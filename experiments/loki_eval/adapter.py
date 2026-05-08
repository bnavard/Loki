"""
Per-sample inference for Loki evaluation, aligned with the
`sota_comparison/<baseline>/adapter.py` shape.

Differences from a true SOTA baseline adapter:
  * The model lives in-process (Loki is local — no `conda run` shell-out
    to a foreign env). So `Evaluator` holds the loaded checkpoint + cond_stage
    module and reuses them across samples.
  * Sample IDs come from the curated HDTF manifest under
    `experiments/sota_comparison/manifests/hdtf.json` (same UID pool every
    SOTA wrapper consumes), so a glob across baselines hits the same identity
    pair under the same `<sample_id>` folder.

Per sample (`run_one(sample: EvalSample, ref_frame_idx, ...)`):
  1. Load the ref + driver clips' `fit.npz` from
     `<flame_root>/<clip_id>/fit.npz` — for HDTF that's
     `data/benchmark/hdtf/flame_tracking/flowface/`.
  2. `prepare_reference(ref_fit, ref_frame_idx, …)` → face-cropped 512×512
     ref image in `[-1, 1]` + the ref's crop_box.
  3. `retarget_driver_verts(ref_fit, driver_fit, crop_box, n_frames, …,
     driver_start=0)` → `(T, V, 3)` NDC verts + `(T, V, 3)` expression
     deformation, computed as `β_ref + ψ_driver[t] + θ_driver[t]` under the
     reference's camera. `driver_start=0` matches the SOTA convention of
     "first N frames of the trimmed driver."
  4. Encode ref → `ref_z` via VAE.
  5. `model.sample_video(...)` — DDIM with classifier-free guidance.
  6. VAE-decode and write the on-disk artifacts in the SOTA-wrapper shape:
       samples/<sample_id>/panel.mp4    -- 512×512 generation
       scratch/<sample_id>/source.png   -- ref frame
       scratch/<sample_id>/driver.mp4   -- 512×512 driver row

Output goes to `<output_dir>/samples/<sample.sample_id>/panel.mp4` plus a
sibling `<output_dir>/scratch/<sample.sample_id>/{source.png, driver.mp4}`
mirroring how every `experiments/sota_comparison/<baseline>/` adapter
writes its files.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import DictConfig

from ldm_base.ldm.util import instantiate_from_config
from loki.data.video_dataset import FLAME_PARAMS_SCHEMA
from loki.flame.flame import FlameSkinnerExtended
from loki.model.checkpoint_compat import strip_legacy_keys
from loki.retargeting import (
    prepare_driver_frames, prepare_reference, retarget_driver_verts,
)
from experiments.sota_comparison.dataset.pairing import EvalSample


HEAD_VERT_PATH = "data/assets/flame/head_vertices.txt"


def _encode_h264(out_mp4: Path, frames_chw_u8: np.ndarray, fps: float) -> None:
    """Pipe `(T, 3, H, W)` uint8 RGB frames into ffmpeg and encode as
    libx264-ultrafast no-audio mp4. Matches the SOTA wrappers' driver.mp4 /
    panel.mp4 encoding profile so the loki_eval output is byte-shape
    indistinguishable from any baseline run on disk."""
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    T, _, H, W = frames_chw_u8.shape
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo",
         "-vcodec", "rawvideo",
         "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}",
         "-r", str(fps),
         "-i", "-",
         "-an",
         "-c:v", "libx264",
         "-preset", "ultrafast",
         "-pix_fmt", "yuv420p",
         str(out_mp4)],
        stdin=subprocess.PIPE,
    )
    try:
        for t in range(T):
            # rgb24 expects (H, W, 3) bytes per frame.
            proc.stdin.write(frames_chw_u8[t].transpose(1, 2, 0).tobytes())
    finally:
        proc.stdin.close()
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg encode failed (rc={rc}) for {out_mp4}")


def _load_fit(path: Path) -> dict:
    return {k: v for k, v in np.load(str(path)).items()}


def _pack_flame_params(fit: dict, start: int, n: int) -> np.ndarray:
    """Pack the driver window's raw FLAME motion parameters into a fixed
    `(n, FLAME_PARAMS_DIM)` vector. Mirrors
    `TalkingHeadDataset._pack_flame_params` so the `flame_vector` ablation
    arm sees the same 77-dim layout at eval time as at train time —
    schema: `expr | rot | neck_rot | jaw_rot | eye_rot`.
    """
    pieces = []
    for key, dim in FLAME_PARAMS_SCHEMA:
        if key in fit:
            arr = np.asarray(fit[key][start:start + n], dtype=np.float32)
            if arr.shape != (n, dim):
                raise ValueError(
                    f"FLAME fit field {key!r} has shape {arr.shape}; "
                    f"expected ({n}, {dim}). Check the fit.npz schema."
                )
        else:
            arr = np.zeros((n, dim), dtype=np.float32)
        pieces.append(arr)
    return np.concatenate(pieces, axis=-1)


def _load_checkpoint_into(model, ckpt_path: str) -> None:
    """Strip the Lightning `model.` prefix and fail loud on any missing /
    unexpected keys, except under the frozen `ref_extractor.*` subtree
    (its weights may legitimately be absent from a Lightning checkpoint
    if they weren't saved — they get re-loaded from SD 2.1 init separately)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw = ckpt.get("state_dict", ckpt)
    sd = {k[len("model."):]: v for k, v in raw.items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    unexpected = [k for k in unexpected if not k.startswith("ref_extractor.")]
    missing    = [k for k in missing    if not k.startswith("ref_extractor.")]
    unexpected = strip_legacy_keys(unexpected)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint load incomplete: {len(missing)} missing, "
            f"{len(unexpected)} unexpected. "
            f"First missing: {missing[:3]}. First unexpected: {unexpected[:3]}."
        )


@dataclass(frozen=True)
class LokiEvalArgs:
    """Inference-side knobs (parsed in run_inference.py, passed to Evaluator).
    Mirror the structure of the SOTA `<Baseline>Args` dataclasses."""
    n_frames:     int   = 16
    cfg_scale:    float = 2.0
    n_ddim_steps: int   = 50


class Evaluator:
    """Holds the loaded Loki model + cond_stage module + FLAME skinner,
    reused across every sample in a run. One instance per process.

    Distinct from SOTA wrappers (which shell out to a baseline's own env per
    sample and so re-load the model every time): Loki is local, so we
    pay the model-load cost once at runner startup and amortize it across
    all samples."""

    def __init__(
        self,
        cfg:        DictConfig,
        checkpoint: str,
        flame_root: Path,
        device:     torch.device,
        args:       LokiEvalArgs = LokiEvalArgs(),
    ) -> None:
        self.cfg        = cfg
        self.flame_root = Path(flame_root)
        self.device     = device
        self.args       = args

        ds = cfg.train_dataset.params
        self.resolution = int(ds.resolution)
        self.latent_res = self.resolution // int(ds.downsample_ratio)
        self.fps        = float(ds.fps)

        self.model = instantiate_from_config(cfg.model)
        _load_checkpoint_into(self.model, checkpoint)
        self.model.eval().to(device)

        # Dispatch on the config's `target` so condition_ablation arms load
        # their own cond_stage module without any change here.
        #
        # Use `self.model.cond_stage_model` (instantiated by the LDM base
        # class from `cond_stage_config` and populated by the checkpoint
        # load above) rather than a fresh `instantiate_from_config(...)`.
        # The rasterized arms (no_posenc, no_deform, baseline) have zero
        # `nn.Parameter`s — their state is deterministic buffers
        # (PropRenderer faces / UVs, PositionalEncoding.freqs), so a
        # fresh instance is byte-identical to the checkpoint-loaded one.
        # The `flame_vector` arm differs: its MLP has parameters that,
        # while frozen by `cond_stage_trainable=False`, are saved in the
        # checkpoint at training-time random init. The downstream
        # ConditioningEncoder learned to consume that specific projection,
        # so we must use the checkpoint copy. The 4-row panel's third row
        # label + slice come from this module's `VIZ_LABEL` / `VIZ_SLICE`.
        self.cond_module = self.model.cond_stage_model

        self.flame_skinner = FlameSkinnerExtended(
            add_mouth=True, n_shape_params=150, n_expr_params=65,
        )
        self.head_vert_ids = np.genfromtxt(HEAD_VERT_PATH).astype(int)

    @torch.no_grad()
    def run_one(
        self,
        sample:        EvalSample,
        ref_frame_idx: int,
        output_dir:    Path,
        title:         str | None = None,
    ) -> Path:
        """Generate one panel + mp4 for the given EvalSample. Returns the
        path to `panel.mp4`.

        Driver windowing is fixed at `driver_start=0` (matches every SOTA
        wrapper's "first N frames of the trimmed driver" convention).
        Sample ID is taken from `sample.sample_id` directly so the on-disk
        folder name aligns with every other baseline's output tree."""
        device   = self.device
        n_frames = self.args.n_frames

        ref_clip    = sample.ref_clip
        driver_clip = sample.driver_clip

        ref_fit_path = self.flame_root / ref_clip.clip_id / "fit.npz"
        drv_fit_path = self.flame_root / driver_clip.clip_id / "fit.npz"
        if not ref_fit_path.is_file():
            raise FileNotFoundError(
                f"Missing FLAME tracking for ref clip: {ref_fit_path}. "
                f"Loki inference requires `fit.npz` per clip."
            )
        if not drv_fit_path.is_file():
            raise FileNotFoundError(
                f"Missing FLAME tracking for driver clip: {drv_fit_path}."
            )
        ref_fit = _load_fit(ref_fit_path)
        drv_fit = _load_fit(drv_fit_path)

        ref_img_norm, _, crop_box = prepare_reference(
            ref_fit, ref_frame_idx, ref_clip.video_path,
            self.resolution, self.flame_skinner, self.head_vert_ids,
        )
        verts_np, offsets_np = retarget_driver_verts(
            ref_fit, drv_fit, crop_box, n_frames, self.flame_skinner,
            driver_start=0,
        )

        # Driver's face-cropped frames — used for the visual panel only.
        driver_frames = prepare_driver_frames(
            drv_fit, driver_clip.video_path,
            n_frames, self.resolution, self.flame_skinner, self.head_vert_ids,
            driver_start=0,
        )

        hint = {
            "driver_verts":  torch.from_numpy(verts_np).unsqueeze(0).to(device),
            "driver_deform": torch.from_numpy(offsets_np).unsqueeze(0).to(device),
        }
        # The `flame_vector` ablation arm reads `driver_flame_params` directly
        # (raw 77-dim FLAME params, no rasterization). Always populate so any
        # arm whose cond module needs it works without a runner-side dispatch;
        # rasterized arms ignore the extra key via `**_unused`.
        flame_params = _pack_flame_params(drv_fit, start=0, n=n_frames)
        hint["driver_flame_params"] = (
            torch.from_numpy(flame_params).unsqueeze(0).to(device)
        )
        c_cond = self.cond_module(hint)

        ref_tensor = torch.from_numpy(ref_img_norm).permute(2, 0, 1).unsqueeze(0).to(device)
        c_cond["ref_z"] = self.model.get_first_stage_encoding(
            self.model.encode_first_stage(ref_tensor)
        )

        c_uncond = {
            k: (torch.zeros_like(v) if torch.is_tensor(v) else v)
            for k, v in c_cond.items()
        }

        latents = self.model.sample_video(
            control=c_cond, control_uncond=c_uncond,
            n_frames=n_frames,
            latent_shape=(4, self.latent_res, self.latent_res),
            n_ddim_steps=self.args.n_ddim_steps,
            cfg_scale=self.args.cfg_scale,
        )
        imgs = self.model.decode_first_stage(latents.unsqueeze(0)).squeeze(0)
        imgs = ((imgs.clamp(-1, 1) + 1) / 2 * 255).byte().cpu().numpy()   # (T, 3, H, W)

        # On-disk shape mirrors the SOTA wrappers' layout exactly so cross-
        # baseline tooling (compute_metrics, populate_drivers, glob walks)
        # treats loki_eval and any sota_comparison/<baseline>/ run
        # uniformly.
        #
        #   samples/<sample_id>/panel.mp4    -- 512×512 generation
        #   scratch/<sample_id>/source.png   -- ref frame (static across T)
        #   scratch/<sample_id>/driver.mp4   -- 512×512 driver row
        ref_rgb_u8 = ((ref_img_norm + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        driver_chw = driver_frames.transpose(0, 3, 1, 2).copy()   # (T, 3, H, W)

        sample_dir  = output_dir / "samples" / sample.sample_id
        scratch_dir = output_dir / "scratch" / sample.sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        scratch_dir.mkdir(parents=True, exist_ok=True)

        panel_mp4  = sample_dir  / "panel.mp4"
        source_png = scratch_dir / "source.png"
        driver_mp4 = scratch_dir / "driver.mp4"

        # cv2.imwrite expects BGR; ref_rgb_u8 is RGB.
        cv2.imwrite(str(source_png), cv2.cvtColor(ref_rgb_u8, cv2.COLOR_RGB2BGR))
        _encode_h264(panel_mp4,  imgs,       fps=self.fps)
        _encode_h264(driver_mp4, driver_chw, fps=self.fps)
        return panel_mp4
