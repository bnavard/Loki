"""
HunyuanPortrait adapter: converts our canonical `EvalSample` records into
the on-disk inputs HunyuanPortrait's `inference.py` expects, then shells
out.

Per sample:
  1. Extract a **randomly sampled** frame from `ref_clip.video_path` →
     `<scratch>/source.png`. Frame index comes from the caller's seeded
     RNG so every sample is reproducible under one top-level `--seed`.
  2. ffmpeg-trim the first `clip_duration_s` seconds of
     `driver_clip.video_path` → `<scratch>/driver.mp4`. HunyuanPortrait
     reads every frame of whatever video we hand it (up to
     `cfg.frame_num`, default 10000), so physically trimming the input is
     the cleanest way to pin output duration exactly. 25 fps × 5 s →
     125 output frames, consistent across baselines driven from the same
     pair list.
  3. Write a **patched config** whose `output_dir` is an absolute path
     into our scratch dir. Upstream's inference.py resolves `output_dir`
     relative to cwd, so a patch is the only reliable way to route their
     output without running from inside their repo.
  4. Invoke `python inference.py --config <patched> --video_path <driver>
     --image_path <source>` inside the `hunyuan_portrait` conda env,
     with cwd set to the cloned repo so relative weight paths
     (`pretrained_weights/...`) resolve.
  5. Upstream writes `<output_dir>/<timestamp>_<img>_<vid>/cropped.mp4`
     (512×512 face crop). We glob it and move to
     `<output_dir>/samples/<sample_id>/panel.mp4`.

Cross-baseline alignment
------------------------
The output `panel.mp4` is HunyuanPortrait's `cropped.mp4` (the pure 512×512
face-crop generation), NOT `full_resolution.mp4` (their paste-back onto the
original source image). Reasons:
  * SadTalker and Marionette both produce face-cropped 512×512 output.
    Metric sweeps on lip-sync / identity / sharpness expect that common
    surface.
  * HunyuanPortrait's paste-back layer is useful for demo videos but
    introduces an extra warp that is baseline-specific and would confound
    per-model-quality comparisons.

Audio
-----
HunyuanPortrait is motion-driven only — the output mp4 is silent. We
don't mux any driver audio at write time so the artefact stays identical
to what the model actually produced.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
from omegaconf import OmegaConf

from experiments.sota_comparison.dataset.pairing import EvalSample


@dataclass(frozen=True)
class HunyuanPortraitArgs:
    """Knobs the baseline exposes via its config yaml. Mirrored 1:1 here so
    the runner can surface them as CLI flags for ablation / sweep. Defaults
    match upstream's `config/hunyuan-portrait.yaml`."""
    num_inference_steps:       int   = 25
    motion_bucket_id:          int   = 0
    n_sample_frames:           int   = 25       # inner frame batch; memory-bound
    use_arcface:               bool  = True     # ArcFace identity conditioning
    min_appearance_guidance:   float = 2.0      # CFG scale (appearance)
    max_appearance_guidance:   float = 2.0
    min_motion_guidance:       float = 2.0      # CFG scale (motion)
    max_motion_guidance:       float = 2.0


# ---------------------------------------------------------------------------
# File prep (same pattern SadTalker's adapter uses)
# ---------------------------------------------------------------------------
def _extract_frame(video_path: Path, frame_idx: int, out_png: Path) -> None:
    """Write frame `frame_idx` of `video_path` as a PNG. Random-access seek
    with a sequential-decode fallback for codecs that mis-handle frame-
    accurate seeks across keyframe boundaries."""
    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            for _ in range(frame_idx + 1):
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError(
                        f"Failed to read frame {frame_idx} from {video_path}"
                    )
        out_png.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_png), frame)
    finally:
        cap.release()


def _trim_video(src: Path, out_mp4: Path, duration_s: float) -> None:
    """Copy the first `duration_s` of `src` to `out_mp4`, re-encoding video
    so the cut lands exactly on the requested duration (stream-copy with
    `-c copy` rounds to the nearest keyframe and can produce shorter or
    longer clips).

    Re-encode cost is negligible (~50 ms/second of output on H100 with
    libx264 ultrafast), and getting the duration exact matters because
    HunyuanPortrait's pipeline iterates every frame it sees."""
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-t", str(duration_s),
        "-i", str(src),
        "-an",                           # drop audio — HunyuanPortrait
                                         # is motion-driven only; keeping
                                         # the stream would just confuse
                                         # moviepy downstream.
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True)


def _make_patched_config(
    upstream_config: Path,
    patched_config:  Path,
    output_dir:      Path,
    args:            HunyuanPortraitArgs,
) -> None:
    """Read upstream's `hunyuan-portrait.yaml`, override `output_dir` to an
    absolute path in our scratch dir, and write the copy. Other paths
    (weight locations) stay relative — they resolve against the cloned repo
    which we pass as cwd to the subprocess.

    Exposed-arg overrides land here too, so anything we let the runner tune
    at the CLI lands in the same place as `output_dir`."""
    cfg = OmegaConf.load(upstream_config)
    cfg.output_dir = str(output_dir.resolve())

    cfg.num_inference_steps             = args.num_inference_steps
    cfg.motion_bucket_id                = args.motion_bucket_id
    cfg.n_sample_frames                 = args.n_sample_frames
    cfg.use_arcface                     = args.use_arcface
    cfg.min_appearance_guidance_scale   = args.min_appearance_guidance
    cfg.max_appearance_guidance_scale   = args.max_appearance_guidance
    cfg.min_motion_guidance_scale       = args.min_motion_guidance
    cfg.max_motion_guidance_scale       = args.max_motion_guidance

    patched_config.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, patched_config)


# ---------------------------------------------------------------------------
# Shell out
# ---------------------------------------------------------------------------
def _run_hunyuan_cli(
    impl_dir:      Path,
    patched_cfg:   Path,
    source_png:    Path,
    driver_mp4:    Path,
    result_dir:    Path,
    conda_env:     str,
) -> Path:
    """Run HunyuanPortrait's inference.py in its conda env. Returns the path
    to the generated `cropped.mp4`.

    `cwd=impl_dir` so relative weight paths in the config
    (`pretrained_weights/...`) resolve. The patched config sits outside the
    repo — inference.py happily loads it via `--config <absolute_path>`.
    """
    result_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "conda", "run", "--no-capture-output", "-n", conda_env,
        "python", "inference.py",
        "--config",     str(patched_cfg.resolve()),
        "--video_path", str(driver_mp4.resolve()),
        "--image_path", str(source_png.resolve()),
    ]
    subprocess.run(cmd, check=True, cwd=str(impl_dir))

    # inference.py writes `<output_dir>/<timestamp>_<img>_<vid>/cropped.mp4`.
    # Timestamp is their side; we grab the newest `cropped.mp4` under the
    # result dir (there should be exactly one per sample, but taking newest
    # is robust to a future upstream change that produces multiple).
    cropped = sorted(result_dir.glob("*/cropped.mp4"), key=lambda p: p.stat().st_mtime)
    if not cropped:
        raise RuntimeError(
            f"HunyuanPortrait produced no cropped.mp4 under {result_dir}. "
            f"Check inference.py's stderr above."
        )
    return cropped[-1]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run_one(
    sample:        EvalSample,
    ref_frame_idx: int,
    impl_dir:      Path,
    output_dir:    Path,
    scratch:       Path,
    conda_env:     str = "hunyuan_portrait",
    args:          HunyuanPortraitArgs = HunyuanPortraitArgs(),
) -> Path:
    """Generate one sample end-to-end. Returns the path to `panel.mp4`.

    Layout:
        scratch/<sample_id>/source.png                 # ref frame
        scratch/<sample_id>/driver.mp4                 # trimmed driver video
        scratch/<sample_id>/hunyuan-portrait.yaml      # patched config
        scratch/<sample_id>/result/<timestamp>_...     # HunyuanPortrait raw
        output_dir/samples/<sample_id>/panel.mp4       # canonical location
    """
    if not (0 <= ref_frame_idx < sample.ref_clip.n_frames):
        raise ValueError(
            f"ref_frame_idx={ref_frame_idx} out of range for clip "
            f"{sample.ref_clip.clip_id} (n_frames={sample.ref_clip.n_frames})"
        )

    work       = scratch / sample.sample_id
    source_png = work / "source.png"
    driver_mp4 = work / "driver.mp4"
    patched_cfg = work / "hunyuan-portrait.yaml"
    result_dir = work / "result"

    _extract_frame(sample.ref_clip.video_path, ref_frame_idx, source_png)
    _trim_video(sample.driver_clip.video_path, driver_mp4, sample.clip_duration_s)
    _make_patched_config(
        upstream_config = impl_dir / "config" / "hunyuan-portrait.yaml",
        patched_config  = patched_cfg,
        output_dir      = result_dir,
        args            = args,
    )

    raw_mp4 = _run_hunyuan_cli(
        impl_dir    = impl_dir,
        patched_cfg = patched_cfg,
        source_png  = source_png,
        driver_mp4  = driver_mp4,
        result_dir  = result_dir,
        conda_env   = conda_env,
    )

    final_dir = output_dir / "samples" / sample.sample_id
    final_dir.mkdir(parents=True, exist_ok=True)
    final_mp4 = final_dir / "panel.mp4"
    shutil.move(str(raw_mp4), str(final_mp4))
    return final_mp4
