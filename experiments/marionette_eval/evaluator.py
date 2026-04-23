"""
Shared inference runner for the Marionette evaluation scripts.

Loads the model + checkpoint once per process and exposes `run_one(...)` to
generate + save one panel per call. Both the cross-identity and same-identity
entry scripts consume this — their only difference is the sampling plan.

The inference path mirrors `marionette/generate.py` exactly, with two
generalisations needed for batched evaluation:

  * A `driver_start_idx` offset threads through `retarget_driver_verts`,
    `prepare_driver_frames`, and the audio window builder — `generate.py`
    always starts the driver at frame 0.
  * Audio is treated as truly optional at runtime: if
    `model.audio_encoder is None` (e.g. trained under `overlays/audio/off.yaml`
    or a future audio-less variant), the audio branch is skipped and no wav
    file is read. Matches the model's own `get_input` behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig

from ldm_base.ldm.util import instantiate_from_config
from marionette.flame.flame import CAP4DFlameSkinner
from marionette.retargeting import (
    prepare_driver_frames, prepare_reference, retarget_driver_verts,
)
from marionette.utils import (
    SAMPLE_RATE, frame_window, load_audio_mono,
    save_labeled_grid, save_video_with_audio, slice_cond_rgb,
)


HEAD_VERT_PATH = "data/assets/flame/head_vertices.txt"


def _load_fit(path: Path) -> dict:
    return {k: v for k, v in np.load(str(path)).items()}


def _load_audio_window_block(
    audio_path: Path,
    driver_start_idx: int,
    n_frames: int,
    samples_per_frame: int,
    audio_context_frames: int,
    n_total_frames: int,
) -> np.ndarray:
    """Per-frame centered audio windows for `[driver_start, driver_start+T)`.

    Returns `(T, window_samples) float32`. Zeros if the wav is missing — the
    audio encoder still runs; the model sees silence for that sample.
    """
    audio = (
        load_audio_mono(audio_path, expected_len=n_total_frames * samples_per_frame)
        if audio_path.exists() else None
    )
    return np.stack([
        frame_window(
            audio, driver_start_idx + t, n_total_frames,
            samples_per_frame, audio_context_frames,
        )
        for t in range(n_frames)
    ], axis=0)


def _load_checkpoint_into(model, ckpt_path: str) -> None:
    """Same contract as `marionette/generate.py::_load_checkpoint_into` —
    strip the Lightning `model.` prefix, tolerate missing/unexpected keys
    only under the frozen `ref_extractor.*` subtree."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw = ckpt.get("state_dict", ckpt)
    sd = {k[len("model."):]: v for k, v in raw.items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    unexpected = [k for k in unexpected if not k.startswith("ref_extractor.")]
    missing    = [k for k in missing    if not k.startswith("ref_extractor.")]
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint load incomplete: {len(missing)} missing, "
            f"{len(unexpected)} unexpected. "
            f"First missing: {missing[:3]}. First unexpected: {unexpected[:3]}."
        )


@dataclass
class EvaluatorPaths:
    flame_root: Path
    video_root: Path
    audio_root: Path


class Evaluator:
    """Holds the loaded model + conditioning module + FLAME skinner, reused
    across every sample in a run. One instance per process."""

    def __init__(
        self,
        cfg: DictConfig,
        checkpoint: str,
        paths: EvaluatorPaths,
        n_frames: int,
        cfg_scale: float,
        n_ddim_steps: int,
        device: torch.device,
    ) -> None:
        self.cfg          = cfg
        self.paths        = paths
        self.n_frames     = n_frames
        self.cfg_scale    = cfg_scale
        self.n_ddim_steps = n_ddim_steps
        self.device       = device

        ds = cfg.train_dataset.params
        self.resolution           = int(ds.resolution)
        self.latent_res           = self.resolution // int(ds.downsample_ratio)
        self.fps                  = float(ds.fps)
        self.samples_per_frame    = int(SAMPLE_RATE / ds.fps)
        self.audio_context_frames = int(ds.audio_context_frames)

        self.model = instantiate_from_config(cfg.model)
        _load_checkpoint_into(self.model, checkpoint)
        self.model.eval().to(device)
        # One single runtime flag gates the entire audio path. If the training
        # config sets `audio_encoder_config: null` (or the arch drops audio
        # entirely later), this code keeps working unchanged.
        self.has_audio = self.model.audio_encoder is not None

        # Dispatch on the config's `target` so condition-ablation arms load
        # their own cond_stage module without any change here.
        self.cond_module = instantiate_from_config(
            cfg.model.params.cond_stage_config,
        ).to(device).eval()

        self.flame_skinner = CAP4DFlameSkinner(
            add_mouth=True, n_shape_params=150, n_expr_params=65,
        )
        self.head_vert_ids = np.genfromtxt(HEAD_VERT_PATH).astype(int)

    @torch.no_grad()
    def run_one(
        self,
        ref_clip: str,
        driver_clip: str,
        ref_frame_idx: int,
        driver_start_idx: int,
        out_dir: Path,
        title: str,
    ) -> None:
        """Generate one T-frame panel + mp4 for the given (ref, driver) tuple.

        Writes `panel.png` + `panel.mp4` into `out_dir`. Does not save
        per-frame PNGs — metric evaluation will read from the mp4."""
        out_dir.mkdir(parents=True, exist_ok=True)
        device   = self.device
        n_frames = self.n_frames

        ref_fit = _load_fit(self.paths.flame_root / ref_clip / "fit.npz")
        drv_fit = _load_fit(self.paths.flame_root / driver_clip / "fit.npz")
        drv_total = int(drv_fit["expr"].shape[0])

        ref_img_norm, _, crop_box = prepare_reference(
            ref_fit, ref_frame_idx,
            self.paths.video_root / f"{ref_clip}.mp4",
            self.resolution, self.flame_skinner, self.head_vert_ids,
        )
        verts_np, offsets_np = retarget_driver_verts(
            ref_fit, drv_fit, crop_box, n_frames, self.flame_skinner,
            driver_start=driver_start_idx,
        )

        # Driver's face-cropped video — used by the 4-row viz panel AND as the
        # natural-video conditioning signal read by condition_ablation arms.
        driver_frames = prepare_driver_frames(
            drv_fit, self.paths.video_root / f"{driver_clip}.mp4",
            n_frames, self.resolution, self.flame_skinner, self.head_vert_ids,
            driver_start=driver_start_idx,
        )
        driver_video_norm = (driver_frames.astype(np.float32) / 127.5) - 1.0

        hint = {
            "driver_verts":  torch.from_numpy(verts_np).unsqueeze(0).to(device),
            "driver_deform": torch.from_numpy(offsets_np).unsqueeze(0).to(device),
            "driver_video":  torch.from_numpy(driver_video_norm).unsqueeze(0).to(device),
        }
        c_cond = self.cond_module(hint)

        ref_tensor = torch.from_numpy(ref_img_norm).permute(2, 0, 1).unsqueeze(0).to(device)
        c_cond["ref_z"] = self.model.get_first_stage_encoding(
            self.model.encode_first_stage(ref_tensor)
        )

        audio_windows = None
        if self.has_audio:
            audio_windows = _load_audio_window_block(
                self.paths.audio_root / f"{driver_clip}.wav",
                driver_start_idx, n_frames,
                self.samples_per_frame, self.audio_context_frames,
                n_total_frames=drv_total,
            )
            audio_t = torch.from_numpy(audio_windows).unsqueeze(0).to(device)
            c_cond["audio_context"] = self.model.audio_encoder(audio_t)
        else:
            c_cond["audio_context"] = None

        c_uncond = {
            k: (torch.zeros_like(v) if torch.is_tensor(v) else v)
            for k, v in c_cond.items()
        }

        latents = self.model.sample_video(
            control=c_cond, control_uncond=c_uncond,
            n_frames=n_frames,
            latent_shape=(4, self.latent_res, self.latent_res),
            n_ddim_steps=self.n_ddim_steps,
            cfg_scale=self.cfg_scale,
        )
        imgs = self.model.decode_first_stage(latents.unsqueeze(0)).squeeze(0)
        imgs = ((imgs.clamp(-1, 1) + 1) / 2 * 255).byte().cpu().numpy()   # (T, 3, H, W)

        # 4-row panel: Reference (static) | Driver Video | Driver Expression | Generated
        ref_rgb_u8 = ((ref_img_norm + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        ref_row = np.broadcast_to(
            ref_rgb_u8.transpose(2, 0, 1)[None],
            (n_frames, 3, self.resolution, self.resolution),
        ).copy()

        driver_row = driver_frames.transpose(0, 3, 1, 2).copy()

        expr_row = slice_cond_rgb(c_cond["spatial_cond"][0], 42, self.resolution)

        rows   = [ref_row, driver_row, expr_row, imgs]
        labels = ["Reference", "Driver Video", "Driver Expression", "Generated"]

        save_labeled_grid(rows, labels, out_dir / "panel.png", title=title)
        save_video_with_audio(
            rows, labels, audio_windows,
            out_dir / "panel.mp4", fps=self.fps, title=title,
        )
