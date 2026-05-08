"""
AniTalker adapter: converts our canonical `EvalSample` records into the
on-disk inputs AniTalker's `code/demo.py` expects, then shells out.

Per sample:
  1. Extract a **randomly sampled** frame from `ref_clip.video_path` →
     `<scratch>/source.png`. Frame index comes from the caller's seeded
     RNG so every sample is reproducible under one top-level `--seed`.
  2. Get the driver's audio → `<scratch>/audio.wav`. If
     `driver_clip.audio_path` is set (datasets that ship sidecar WAVs), we
     ffmpeg it straight; otherwise we ffmpeg from the mp4's muxed audio
     stream (HDTF and similar muxed-audio datasets). Either way the output is mono 16 kHz WAV
     — the format AniTalker's HuBERT feature extractor expects.
  3. Hand demo.py a placeholder `--test_hubert_path` pointing at a scratch
     `.npy` file that does NOT yet exist. AniTalker's demo.py detects the
     missing file and auto-extracts HuBERT features from the WAV on the
     fly (using the `ckpts/chinese-hubert-large/` model). Persisted to
     the placeholder path so a re-run skips extraction.
  4. Invoke `python code/demo.py --infer_type hubert_audio_only
     --stage1_checkpoint_path ckpts/stage1.ckpt
     --stage2_checkpoint_path ckpts/stage2_audio_only_hubert.ckpt
     --test_image_path <png> --test_audio_path <wav>
     --test_hubert_path <npy> --result_path <scratch>/result/ [--face_sr]`
     inside the `anitalker` conda env with cwd set to the cloned repo so
     relative weight paths (`ckpts/...`) resolve.
  5. AniTalker writes `<result_path>/<image_stem>-<audio_stem>.mp4` at
     256×256, or additionally `<image_stem>-<audio_stem>_SR.mp4` at 512×512
     when `--face_sr` is passed. We prefer the SR output when present (it
     matches the 512×512 surface every other baseline and Loki
     produce); otherwise we fall back to the 256 file.

Modality
--------
AniTalker is audio-driven (HuBERT features). `cross_identity` means "A's
face + B's audio" — same semantics as SadTalker's cross-identity, and
identical pair-list construction.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2

from experiments.sota_comparison.dataset.pairing import EvalSample


# Default ckpt paths under `impl/ckpts/`. setup_env.sh lands the weights
# from `taocode/anitalker_ckpts` there; upstream's demo.py expects exactly
# these filenames.
DEFAULT_STAGE1_CKPT = Path("ckpts") / "stage1.ckpt"
DEFAULT_STAGE2_CKPT = Path("ckpts") / "stage2_audio_only_hubert.ckpt"


@dataclass(frozen=True)
class AniTalkerArgs:
    """Knobs the baseline exposes on its CLI. Mirror upstream's demo.py
    1:1 so the runner can surface them for ablation / sweep."""
    step_T:        int  = 50        # diffusion denoising steps
    seed:          int  = 0         # AniTalker's internal seed (distinct from
                                    # the runner's sample-selection seed)
    face_sr:       bool = True      # GFPGAN upscale 256 → 512 to match other
                                    # baselines' output surface
    motion_dim:    int  = 20
    decoder_layers: int = 2


# ---------------------------------------------------------------------------
# File prep
# ---------------------------------------------------------------------------
def _extract_frame(video_path: Path, frame_idx: int, out_png: Path) -> None:
    """Same pattern as the other adapters — random-access seek with a
    sequential-decode fallback for codecs that mis-handle seeks across
    keyframe boundaries."""
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


def _trim_driver_video(src: Path, out_mp4: Path, duration_s: float) -> None:
    """Save the first `duration_s` of the driver clip as `driver.mp4`. AniTalker
    itself doesn't read this — the model is audio-only (HuBERT features). We
    write it so the scratch dir matches the on-disk shape of the visual
    baselines (hunyuan, xportrait), which lets evaluation tooling glob
    `*/scratch/<id>/driver.mp4` uniformly across every SOTA wrapper."""
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-t", str(duration_s),
         "-i", str(src),
         "-an",
         "-c:v", "libx264",
         "-preset", "ultrafast",
         "-pix_fmt", "yuv420p",
         str(out_mp4)],
        check=True,
    )


def _extract_audio(src: Path, out_wav: Path, duration_s: float) -> None:
    """Extract the first `duration_s` of `src` as mono-16 kHz WAV. Handles
    both video inputs with muxed audio AND standalone .wav inputs — ffmpeg
    picks the first audio stream regardless of container."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-t", str(duration_s),
        "-i", str(src),
        "-vn",                          # no video
        "-acodec", "pcm_s16le",
        "-ac", "1",                     # mono
        "-ar", "16000",                 # 16 kHz (HuBERT expects this)
        str(out_wav),
    ]
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Shell out
# ---------------------------------------------------------------------------
def _run_anitalker_cli(
    impl_dir:      Path,
    source_png:    Path,
    audio_wav:     Path,
    hubert_npy:    Path,
    result_dir:    Path,
    stage1_ckpt:   Path,
    stage2_ckpt:   Path,
    conda_env:     str,
    args:          AniTalkerArgs,
) -> Path:
    """Run AniTalker's demo.py in its conda env. Returns the path to the
    generated mp4 (SR if face_sr=True and it wrote one, otherwise the 256
    version).

    `cwd=impl_dir` so relative paths (`ckpts/...`, including the
    hardcoded `ckpts/chinese-hubert-large/` HuBERT model) resolve.
    """
    result_dir.mkdir(parents=True, exist_ok=True)
    hubert_npy.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "conda", "run", "--no-capture-output", "-n", conda_env,
        "python", "code/demo.py",
        "--infer_type",              "hubert_audio_only",
        "--stage1_checkpoint_path",  str(stage1_ckpt),                      # rel → impl_dir
        "--stage2_checkpoint_path",  str(stage2_ckpt),                      # rel → impl_dir
        "--test_image_path",         str(source_png.resolve()),
        "--test_audio_path",         str(audio_wav.resolve()),
        "--test_hubert_path",        str(hubert_npy.resolve()),             # auto-extracted if missing
        "--result_path",             str(result_dir.resolve()) + "/",       # demo.py joins with f-string
        "--seed",                    str(args.seed),
        "--step_T",                  str(args.step_T),
        "--motion_dim",              str(args.motion_dim),
        "--decoder_layers",          str(args.decoder_layers),
    ]
    if args.face_sr:
        cmd.append("--face_sr")

    subprocess.run(cmd, check=True, cwd=str(impl_dir))

    # Upstream writes `<result_path>/<img_stem>-<audio_stem>.mp4` (256) and
    # optionally `<img_stem>-<audio_stem>_SR.mp4` (512 via GFPGAN).
    sr_mp4  = list(result_dir.glob("*_SR.mp4"))
    raw_mp4 = [p for p in result_dir.glob("*.mp4") if not p.name.endswith("_SR.mp4")]
    if args.face_sr and sr_mp4:
        return sorted(sr_mp4, key=lambda p: p.stat().st_mtime)[-1]
    if raw_mp4:
        return sorted(raw_mp4, key=lambda p: p.stat().st_mtime)[-1]
    raise RuntimeError(
        f"AniTalker produced no mp4 under {result_dir}. "
        f"Check demo.py's stderr above."
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run_one(
    sample:        EvalSample,
    ref_frame_idx: int,
    impl_dir:      Path,
    output_dir:    Path,
    scratch:       Path,
    conda_env:     str = "anitalker",
    stage1_ckpt:   Path = DEFAULT_STAGE1_CKPT,
    stage2_ckpt:   Path = DEFAULT_STAGE2_CKPT,
    args:          AniTalkerArgs = AniTalkerArgs(),
) -> Path:
    """Generate one sample end-to-end. Returns the path to `panel.mp4`.

    Layout:
        scratch/<sample_id>/source.png     # ref frame
        scratch/<sample_id>/audio.wav      # driver audio, 16 kHz mono
        scratch/<sample_id>/driver.mp4     # driver_clip video, same trim
        scratch/<sample_id>/hubert.npy     # auto-extracted features
        scratch/<sample_id>/result/*.mp4   # AniTalker raw + (opt) _SR output
        output_dir/samples/<sample_id>/panel.mp4
    """
    if not (0 <= ref_frame_idx < sample.ref_clip.n_frames):
        raise ValueError(
            f"ref_frame_idx={ref_frame_idx} out of range for clip "
            f"{sample.ref_clip.clip_id} (n_frames={sample.ref_clip.n_frames})"
        )

    work       = scratch / sample.sample_id
    source_png = work / "source.png"
    audio_wav  = work / "audio.wav"
    driver_mp4 = work / "driver.mp4"
    hubert_npy = work / "hubert.npy"
    result_dir = work / "result"

    _extract_frame(sample.ref_clip.video_path, ref_frame_idx, source_png)

    # Prefer the driver's sidecar audio when the dataset provides one;
    # fall back to the video's muxed audio (HDTF and similar muxed-audio datasets).
    audio_src = sample.driver_clip.audio_path or sample.driver_clip.video_path
    _extract_audio(audio_src, audio_wav, sample.clip_duration_s)
    # `driver.mp4` is a co-output for downstream eval — AniTalker reads only
    # `audio.wav` (HuBERT features). Always sourced from the driver's
    # *video* path so we get a proper mp4 even when audio lives in a
    # separate sidecar.
    _trim_driver_video(sample.driver_clip.video_path, driver_mp4, sample.clip_duration_s)

    raw_mp4 = _run_anitalker_cli(
        impl_dir    = impl_dir,
        source_png  = source_png,
        audio_wav   = audio_wav,
        hubert_npy  = hubert_npy,
        result_dir  = result_dir,
        stage1_ckpt = stage1_ckpt,
        stage2_ckpt = stage2_ckpt,
        conda_env   = conda_env,
        args        = args,
    )

    final_dir = output_dir / "samples" / sample.sample_id
    final_dir.mkdir(parents=True, exist_ok=True)
    final_mp4 = final_dir / "panel.mp4"
    shutil.move(str(raw_mp4), str(final_mp4))
    return final_mp4
