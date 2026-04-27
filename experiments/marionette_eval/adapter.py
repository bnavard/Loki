"""
Per-sample inference for Marionette evaluation, aligned with the
`sota_comparison/<baseline>/adapter.py` shape.

Differences from a true SOTA baseline adapter:
  * The model lives in-process (Marionette is local — no `conda run` shell-out
    to a foreign env). So `Evaluator` holds the loaded checkpoint + cond_stage
    module and reuses them across samples.
  * Sample IDs come from the curated TalkVid manifest under
    `experiments/sota_comparison/manifests/talkvid.json` (same UID pool every
    SOTA wrapper consumes), so a glob across baselines hits the same identity
    pair under the same `<sample_id>` folder.

Per sample (`run_one(sample: EvalSample, ref_frame_idx, ...)`):
  1. Load the ref clip's `fit.npz` from `<flame_root>/<clip_id>/fit.npz`.
     `flame_root` is dataset-aware: the runner picks
     `cfg.flame_roots[<dataset>]` so TalkVid reads from
     `data/flame_tracking/flowface/` and HDTF reads from
     `data/benchmark/hdtf/flame_tracking/flowface/`.
  2. `prepare_reference(ref_fit, ref_frame_idx, …)` → face-cropped 512×512
     ref image in `[-1, 1]` + the ref's crop_box.
  3. `retarget_driver_verts(ref_fit, driver_fit, crop_box, n_frames, …,
     driver_start=0)` → `(T, V, 3)` NDC verts + `(T, V, 3)` expression
     deformation, computed as `β_ref + ψ_driver[t] + θ_driver[t]` under the
     reference's camera. `driver_start=0` matches the SOTA convention of
     "first N frames of the trimmed driver."
  4. `prepare_driver_frames(driver_fit, …, driver_start=0)` → driver's own
     face-cropped frames for the panel's "Driver Video" row + the
     driver-video conditioning (used by no_flame / no_deform arms; ignored
     by baseline).
  5. Encode ref → `ref_z` via VAE.
  6. If audio: read `sample.driver_clip.audio_path` (TalkVid sidecar WAV),
     build per-frame ±context windows, encode via `model.audio_encoder`.
     Skip the whole audio branch if the checkpoint has no audio encoder
     (the audio_off arm of condition_ablation).
  7. `model.sample_video(...)` — DDIM with classifier-free guidance.
  8. VAE-decode and write the on-disk artifacts in the SOTA-wrapper shape:
       samples/<sample_id>/panel.mp4    -- 512×512 generation, no audio
       scratch/<sample_id>/source.png   -- ref frame
       scratch/<sample_id>/driver.mp4   -- 512×512 driver row, no audio

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
from marionette.flame.flame import CAP4DFlameSkinner
from marionette.retargeting import (
    prepare_driver_frames, prepare_reference, retarget_driver_verts,
)
from marionette.utils import SAMPLE_RATE, frame_window, load_audio_mono
from experiments.sota_comparison.dataset.pairing import EvalSample


HEAD_VERT_PATH = "data/assets/flame/head_vertices.txt"


def _encode_h264(out_mp4: Path, frames_chw_u8: np.ndarray, fps: float) -> None:
    """Pipe `(T, 3, H, W)` uint8 RGB frames into ffmpeg and encode as
    libx264-ultrafast no-audio mp4. Matches the SOTA wrappers' driver.mp4 /
    panel.mp4 encoding profile so the marionette_eval output is byte-shape
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


def _load_audio_window_block(
    audio_path: Path,
    n_frames: int,
    samples_per_frame: int,
    audio_context_frames: int,
    n_total_frames: int,
) -> np.ndarray:
    """Per-frame ±`audio_context_frames` centered windows over
    `[0, n_frames)` of the driver. Zeros if the wav is missing — the audio
    encoder still runs; the model sees silence for that sample."""
    audio = (
        load_audio_mono(audio_path, expected_len=n_total_frames * samples_per_frame)
        if audio_path is not None and audio_path.exists() else None
    )
    return np.stack([
        frame_window(
            audio, t, n_total_frames,
            samples_per_frame, audio_context_frames,
        )
        for t in range(n_frames)
    ], axis=0)


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
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint load incomplete: {len(missing)} missing, "
            f"{len(unexpected)} unexpected. "
            f"First missing: {missing[:3]}. First unexpected: {unexpected[:3]}."
        )


@dataclass(frozen=True)
class MarionetteEvalArgs:
    """Inference-side knobs (parsed in run_inference.py, passed to Evaluator).
    Mirror the structure of the SOTA `<Baseline>Args` dataclasses."""
    n_frames:     int   = 16
    cfg_scale:    float = 2.0
    n_ddim_steps: int   = 50


class Evaluator:
    """Holds the loaded Marionette model + cond_stage module + FLAME skinner,
    reused across every sample in a run. One instance per process.

    Distinct from SOTA wrappers (which shell out to a baseline's own env per
    sample and so re-load the model every time): Marionette is local, so we
    pay the model-load cost once at runner startup and amortize it across
    all samples."""

    def __init__(
        self,
        cfg:        DictConfig,
        checkpoint: str,
        flame_root: Path,
        device:     torch.device,
        args:       MarionetteEvalArgs = MarionetteEvalArgs(),
    ) -> None:
        self.cfg        = cfg
        self.flame_root = Path(flame_root)
        self.device     = device
        self.args       = args

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
        # config sets `audio_encoder_config: null` (the audio_off arm of
        # condition_ablation), this code keeps working unchanged.
        self.has_audio = self.model.audio_encoder is not None

        # Dispatch on the config's `target` so condition_ablation arms load
        # their own cond_stage module without any change here. The 4-row
        # panel's third row label + slice come from the active cond_stage's
        # `VIZ_LABEL` / `VIZ_SLICE` class attrs.
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
                f"Marionette inference requires `fit.npz` per clip."
            )
        if not drv_fit_path.is_file():
            raise FileNotFoundError(
                f"Missing FLAME tracking for driver clip: {drv_fit_path}."
            )
        ref_fit = _load_fit(ref_fit_path)
        drv_fit = _load_fit(drv_fit_path)
        drv_total = int(drv_fit["expr"].shape[0])

        ref_img_norm, _, crop_box = prepare_reference(
            ref_fit, ref_frame_idx, ref_clip.video_path,
            self.resolution, self.flame_skinner, self.head_vert_ids,
        )
        verts_np, offsets_np = retarget_driver_verts(
            ref_fit, drv_fit, crop_box, n_frames, self.flame_skinner,
            driver_start=0,
        )

        # Driver's face-cropped frames — used for the viz panel AND as the
        # natural-video conditioning signal read by no_flame / no_deform arms.
        driver_frames = prepare_driver_frames(
            drv_fit, driver_clip.video_path,
            n_frames, self.resolution, self.flame_skinner, self.head_vert_ids,
            driver_start=0,
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
                driver_clip.audio_path,
                n_frames,
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
            n_ddim_steps=self.args.n_ddim_steps,
            cfg_scale=self.args.cfg_scale,
        )
        imgs = self.model.decode_first_stage(latents.unsqueeze(0)).squeeze(0)
        imgs = ((imgs.clamp(-1, 1) + 1) / 2 * 255).byte().cpu().numpy()   # (T, 3, H, W)

        # On-disk shape mirrors the SOTA wrappers' layout exactly so cross-
        # baseline tooling (compute_metrics, populate_drivers, glob walks)
        # treats marionette_eval and any sota_comparison/<baseline>/ run
        # uniformly.
        #
        #   samples/<sample_id>/panel.mp4    -- 512×512 generation, no audio
        #   scratch/<sample_id>/source.png   -- ref frame (static across T)
        #   scratch/<sample_id>/driver.mp4   -- 512×512 driver row, no audio
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
