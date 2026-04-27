"""
SadTalker adapter: converts our canonical `EvalSample` records into the
on-disk inputs SadTalker's `inference.py` expects, then shells out.

Per sample:
  1. Extract a **randomly sampled** frame from `ref_clip.video_path` →
     `<scratch>/source.png`. Frame index is drawn from the caller's seeded
     RNG (centralised seeding policy lives in the runner), so `ref_frame_idx`
     is reproducible across runs for a given `(protocol, seed, sample_id)`.
     A random frame beats "always frame 0" because the first frame of many
     clips is a transition / mouth-open / motion-blurred — not a clean
     identity anchor.
  2. Extract the first `clip_duration_s` seconds of audio from
     `driver_clip.video_path` → `<scratch>/audio.wav` (via ffmpeg, mono,
     16 kHz — SadTalker's wav2lip encoder expects 16 k).
  3. Invoke `python inference.py --source_image <png> --driven_audio <wav>
     --result_dir <scratch>/result/` inside the `sadtalker` conda env.
  4. The upstream script writes `<result_dir>/<timestamp>.mp4`. We grab the
     newest mp4 and move it to `<output_dir>/samples/<sample_id>/panel.mp4`.

The shell-out is intentional. SadTalker's inference.py is ~140 lines of
glue: 3DMM extraction → audio2coeff (CVAE) → facerender → ffmpeg mux.
Calling it as a subprocess keeps deps sane (SadTalker's torch 2.1/cu121
lives in its own conda env; the marionette env stays unpolluted) and lets
upstream changes flow through without re-vendoring.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2

from experiments.sota_comparison.dataset.pairing import EvalSample


@dataclass(frozen=True)
class SadTalkerArgs:
    """Knobs the baseline exposes on its inference CLI. These map 1:1 to
    SadTalker's argparse — documented in their README — so our runner can
    surface them uniformly for ablation / sweep."""
    size:        int   = 512          # 256 or 512 — face resolution
    preprocess:  str   = "crop"       # crop | extcrop | resize | full | extfull
    pose_style:  int   = 0            # 0..45 — learned speaker-style bucket
    enhancer:    str | None = None    # None | "gfpgan" | "RestoreFormer"
    still:       bool  = False        # reduce head motion (paper-style)
    batch_size:  int   = 2            # facerender batch; memory-bound


def _extract_frame(video_path: Path, frame_idx: int, out_png: Path) -> None:
    """Write frame `frame_idx` of `video_path` as a PNG. Uses OpenCV's
    random-access seek; falls back to sequential decode if the seek lands on
    a non-keyframe (some codecs / containers silently return stale buffers
    on `CAP_PROP_POS_FRAMES` seeks that cross a keyframe boundary).
    """
    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            # Sequential fallback for codecs that mis-handle frame-accurate
            # seeks. Slower for large frame_idx but always correct.
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
    """Save the first `duration_s` of the driver clip as `driver.mp4`. SadTalker
    itself doesn't read this — the model is audio-only. We write it so the
    scratch dir matches the on-disk shape of the visual baselines (hunyuan,
    xportrait), which lets evaluation tooling glob `*/scratch/<id>/driver.mp4`
    uniformly across every SOTA wrapper. Re-encoding (not `-c copy`) so the
    duration lands exactly where the audio.wav cut did, even when the source
    GOP boundaries don't align."""
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
    """Extract the first `duration_s` seconds of `src` as mono-16 kHz WAV.
    `src` may be either a video with muxed audio (HDTF, VoxCeleb2, CelebV-HQ)
    or a standalone wav (TalkVid's sidecar audio). Either way ffmpeg picks
    the first audio stream and resamples/downmixes to 16 kHz mono — the
    format SadTalker's wav2lip-based audio encoder expects."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-t", str(duration_s),
        "-i", str(src),
        "-vn",                          # no video
        "-acodec", "pcm_s16le",         # uncompressed WAV
        "-ac", "1",                     # mono
        "-ar", "16000",                 # 16 kHz
        str(out_wav),
    ]
    subprocess.run(cmd, check=True)


def _run_sadtalker_cli(
    impl_dir:     Path,
    source_png:   Path,
    driven_wav:   Path,
    result_dir:   Path,
    conda_env:    str,
    args:         SadTalkerArgs,
) -> Path:
    """Shell out to SadTalker's inference.py inside its conda env. Returns
    the path to the generated mp4.

    `conda run -n <env> python inference.py ...` is the simplest reproducible
    way to hop envs from within the marionette runner. `--no-capture-output`
    streams SadTalker's stdout/stderr so the 3DMM / facerender progress
    prints reach the terminal.
    """
    result_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "conda", "run", "--no-capture-output", "-n", conda_env,
        "python", "inference.py",
        "--source_image", str(source_png.resolve()),
        "--driven_audio", str(driven_wav.resolve()),
        "--result_dir",   str(result_dir.resolve()),
        "--size",         str(args.size),
        "--preprocess",   args.preprocess,
        "--pose_style",   str(args.pose_style),
        "--batch_size",   str(args.batch_size),
    ]
    if args.enhancer is not None:
        cmd += ["--enhancer", args.enhancer]
    if args.still:
        cmd += ["--still"]

    subprocess.run(cmd, check=True, cwd=str(impl_dir))

    # Upstream writes `<result_dir>/<timestamp>.mp4`; grab whichever mp4 is
    # newest in the directory so we're robust to their filename format.
    mp4s = sorted(result_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
    if not mp4s:
        raise RuntimeError(
            f"SadTalker produced no mp4 in {result_dir}. Check stderr above."
        )
    return mp4s[-1]


def run_one(
    sample:        EvalSample,
    ref_frame_idx: int,
    impl_dir:      Path,
    output_dir:    Path,
    scratch:       Path,
    conda_env:     str = "sadtalker",
    args:          SadTalkerArgs = SadTalkerArgs(),
) -> Path:
    """Generate one sample end-to-end. Returns the path to `panel.mp4`.

    `ref_frame_idx` is drawn by the caller from a seeded RNG (see
    `run_inference.py`) so the full evaluation is reproducible under a
    single top-level `--seed`.

    Layout:
        scratch/<sample_id>/source.png             # ref_clip, frame ref_frame_idx
        scratch/<sample_id>/audio.wav              # driver_clip, first clip_duration_s
        scratch/<sample_id>/driver.mp4             # driver_clip video, same trim
        scratch/<sample_id>/result/*.mp4           # SadTalker's raw output
        output_dir/samples/<sample_id>/panel.mp4   # canonical location
    """
    if not (0 <= ref_frame_idx < sample.ref_clip.n_frames):
        raise ValueError(
            f"ref_frame_idx={ref_frame_idx} out of range for clip "
            f"{sample.ref_clip.clip_id} (n_frames={sample.ref_clip.n_frames})"
        )

    work       = scratch / sample.sample_id
    source_png = work / "source.png"
    driven_wav = work / "audio.wav"
    driver_mp4 = work / "driver.mp4"
    result_dir = work / "result"

    _extract_frame(sample.ref_clip.video_path, ref_frame_idx, source_png)
    # Prefer the driver's sidecar audio_path when the dataset provides one
    # (TalkVid: silent mp4s + sibling .wav files); fall back to the video
    # itself when audio is muxed in (HDTF, VoxCeleb2, CelebV-HQ).
    audio_src = sample.driver_clip.audio_path or sample.driver_clip.video_path
    _extract_audio(audio_src, driven_wav, sample.clip_duration_s)
    # `driver.mp4` is a co-output for downstream eval — SadTalker reads only
    # `audio.wav`. Always sourced from the driver's *video* path (not
    # `audio_path`) so the file is a proper mp4 even on TalkVid where audio
    # lives separately as .wav.
    _trim_driver_video(sample.driver_clip.video_path, driver_mp4, sample.clip_duration_s)

    raw_mp4 = _run_sadtalker_cli(
        impl_dir=impl_dir,
        source_png=source_png,
        driven_wav=driven_wav,
        result_dir=result_dir,
        conda_env=conda_env,
        args=args,
    )

    final_dir = output_dir / "samples" / sample.sample_id
    final_dir.mkdir(parents=True, exist_ok=True)
    final_mp4 = final_dir / "panel.mp4"
    shutil.move(str(raw_mp4), str(final_mp4))
    return final_mp4
