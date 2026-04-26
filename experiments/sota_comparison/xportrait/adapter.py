"""
X-Portrait adapter: converts our canonical `EvalSample` records into the
on-disk inputs X-Portrait's `core/test_xportrait.py` expects, then shells out.

Per sample:
  1. Extract a **randomly sampled** ref frame from `ref_clip.video_path` →
     `<scratch>/source.png`. Frame index comes from the caller's seeded
     RNG so every sample is reproducible under one top-level `--seed`.
  2. ffmpeg-trim the first `clip_duration_s` seconds of
     `driver_clip.video_path` → `<scratch>/driver.mp4`. X-Portrait reads
     every frame of the driver video (with `--out_frames -1`), so the
     physical trim is the clean way to pin output duration exactly.
  3. Invoke `python core/test_xportrait.py --source_image <png>
     --driving_video <mp4> --resume_dir <ckpt> --output_dir <scratch/result>
     --best_frame -1 --out_frames -1 ...` inside the `xportrait` conda env
     with cwd set to the cloned repo so relative paths
     (`model_lib/ControlNet/...`, `config/...`) resolve.
  4. X-Portrait writes one mp4 under `<output_dir>` whose filename encodes
     source/driver names + config. We grab the newest mp4 from that dir
     and move it to `<output_dir>/samples/<sample_id>/panel.mp4`.

`--best_frame -1` auto-detects the frame in the driver video whose head
pose best matches the source image via face-alignment landmarks (their
`find_best_frame_byheadpose_fa`). That's essential for batched eval — the
`bash scripts/test_xportrait.sh` demo hardcodes `--best_frame 36` for a
specific driver clip; nothing useful to port across hundreds of samples.

X-Portrait is motion-driven; the output mp4 is silent. Same policy as
HunyuanPortrait — don't mux the driver's audio here, leave the on-disk
artefact identical to what the model produced. Muxing, if desired for
qualitative review, happens downstream at analysis time.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2

from experiments.sota_comparison.dataset.pairing import EvalSample


# Default checkpoint filename landed by setup_env.sh under
# `impl/checkpoint/model_state-415001.th`. Upstream's demo script hardcodes
# this exact name — do not rename unless the GDrive artefact changes.
DEFAULT_CKPT = Path("checkpoint") / "model_state-415001.th"

# Upstream's inference model config (relative to impl_dir).
DEFAULT_MODEL_CONFIG = Path("config") / "cldm_v15_appearance_pose_local_mm.yaml"


@dataclass(frozen=True)
class XPortraitArgs:
    """Knobs the baseline exposes via its CLI. Mirrored 1:1 here so the runner
    can surface them uniformly for ablation / sweep. Defaults match upstream's
    `scripts/test_xportrait.sh`."""
    uc_scale:    int = 5          # CFG-like unconditional guidance scale
    ddim_steps:  int = 30
    num_mix:     int = 4          # overlap frames for prompt-travelling
    seed:        int = 999        # upstream demo seed
    best_frame:  int = -1         # -1 → auto-detect via face-alignment


# ---------------------------------------------------------------------------
# File prep (same shape as SadTalker / HunyuanPortrait adapters)
# ---------------------------------------------------------------------------
def _extract_frame(video_path: Path, frame_idx: int, out_png: Path) -> None:
    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            # Sequential fallback for codecs that mis-handle frame-accurate seeks.
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
    """Re-encode the first `duration_s` of `src` to `out_mp4`. Re-encoding
    (rather than `-c copy`) ensures the cut lands exactly on the requested
    duration — X-Portrait iterates every frame it reads, so an over-long
    trim means extra (and wasted) generation steps."""
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-t", str(duration_s),
        "-i", str(src),
        "-an",                           # no audio — X-Portrait ignores it
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Shell out
# ---------------------------------------------------------------------------
def _run_xportrait_cli(
    impl_dir:      Path,
    source_png:    Path,
    driver_mp4:    Path,
    result_dir:    Path,
    ckpt_rel:      Path,
    model_config:  Path,
    conda_env:     str,
    args:          XPortraitArgs,
) -> Path:
    """Run X-Portrait's test script in its conda env. Returns the path to the
    generated mp4.

    `cwd=impl_dir` so relative paths (`model_lib/...`, `config/...`, the
    checkpoint under `checkpoint/`) resolve. The output dir is an absolute
    path in our scratch tree so X-Portrait's writer lands where we expect.
    """
    result_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "conda", "run", "--no-capture-output", "-n", conda_env,
        "python3", "core/test_xportrait.py",
        "--model_config",  str(model_config),                 # relative → impl_dir
        "--resume_dir",    str(ckpt_rel),                     # relative → impl_dir
        "--output_dir",    str(result_dir.resolve()),         # absolute
        "--source_image",  str(source_png.resolve()),
        "--driving_video", str(driver_mp4.resolve()),
        "--seed",          str(args.seed),
        "--uc_scale",      str(args.uc_scale),
        "--ddim_steps",    str(args.ddim_steps),
        "--num_mix",       str(args.num_mix),
        "--best_frame",    str(args.best_frame),              # -1 → auto
        "--out_frames",    "-1",                              # use all trimmed frames
    ]
    subprocess.run(cmd, check=True, cwd=str(impl_dir))

    # Upstream's filename pattern is
    #   `{name}_{control_type}_uc{uc_scale}_{source_name}_by_{driving_video_name}_mix{N}.mp4`
    # We just grab the newest mp4 in the result dir — there should be exactly
    # one per call; picking newest is robust to a future upstream filename
    # change.
    mp4s = sorted(result_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
    if not mp4s:
        raise RuntimeError(
            f"X-Portrait produced no mp4 under {result_dir}. "
            f"Check test_xportrait.py's stderr above."
        )
    return mp4s[-1]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run_one(
    sample:        EvalSample,
    ref_frame_idx: int,
    impl_dir:      Path,
    output_dir:    Path,
    scratch:       Path,
    conda_env:     str = "xportrait",
    ckpt_rel:      Path = DEFAULT_CKPT,
    model_config:  Path = DEFAULT_MODEL_CONFIG,
    args:          XPortraitArgs = XPortraitArgs(),
) -> Path:
    """Generate one sample end-to-end. Returns the path to `panel.mp4`.

    Layout:
        scratch/<sample_id>/source.png                 # ref frame
        scratch/<sample_id>/driver.mp4                 # trimmed driver video
        scratch/<sample_id>/result/*.mp4               # X-Portrait raw output
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
    result_dir = work / "result"

    _extract_frame(sample.ref_clip.video_path, ref_frame_idx, source_png)
    _trim_video(sample.driver_clip.video_path, driver_mp4, sample.clip_duration_s)

    raw_mp4 = _run_xportrait_cli(
        impl_dir     = impl_dir,
        source_png   = source_png,
        driver_mp4   = driver_mp4,
        result_dir   = result_dir,
        ckpt_rel     = ckpt_rel,
        model_config = model_config,
        conda_env    = conda_env,
        args         = args,
    )

    final_dir = output_dir / "samples" / sample.sample_id
    final_dir.mkdir(parents=True, exist_ok=True)
    final_mp4 = final_dir / "panel.mp4"
    shutil.move(str(raw_mp4), str(final_mp4))
    return final_mp4
