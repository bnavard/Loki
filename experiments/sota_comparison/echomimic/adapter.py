"""
EchoMimic adapter: converts our canonical `EvalSample` records into the
on-disk inputs EchoMimic's `infer_audio2vid.py` expects, then shells out.

Per sample:
  1. Extract a **randomly sampled** ref frame from `ref_clip.video_path` →
     `<scratch>/source.png`. Same seeded-RNG policy as every other SOTA
     adapter — `(protocol, seed, sample_id)` reproduces the same frame
     across baselines.
  2. Get the driver's audio → `<scratch>/audio.wav`. Prefers
     `driver_clip.audio_path` (TalkVid sidecar WAVs); falls back to
     ffmpeg-extracting from the muxed video stream (HDTF, VoxCeleb2).
     Output is mono 16 kHz — what EchoMimic's whisper-tiny audio encoder
     expects (matching `--sample_rate 16000`).
  3. Generate a per-sample patched config YAML at
     `<scratch>/animation.yaml` — copies upstream's
     `configs/prompts/animation.yaml` and replaces `test_cases` with a
     single mapping `{source.png: [audio.wav]}` using ABSOLUTE paths so
     OmegaConf doesn't need to resolve them against any particular cwd.
     The `pretrained_*_path` and `inference_config` entries stay
     RELATIVE; they resolve against the subprocess's cwd, which is set
     to `impl_dir`.
  4. Invoke `python infer_audio2vid.py --config <patched>.yaml -W 512 -H
     512 -L <num_frames> --fps <fps> --seed <S>` inside the `echomimic`
     conda env with cwd=impl_dir.
  5. EchoMimic writes
     `output/<date>/<HHMM--seed-WxH>/<ref>_<audio>_<H>x<W>_<int(cfg)>_<HHMM>_withaudio.mp4`
     under cwd. We glob for the newest `*_withaudio.mp4` and move it to
     `samples/<sample_id>/panel.mp4`.

Modality
--------
Audio-driven (image + WAV). Same shape as SadTalker / AniTalker. Output is
the lifelike portrait animation with audio muxed in.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
from omegaconf import OmegaConf

from experiments.sota_comparison.dataset.pairing import EvalSample


# Default ckpt sub-paths under impl/pretrained_weights/. setup_env.sh lands
# the audio-only subset of `BadToBest/EchoMimic` here; upstream's
# `configs/prompts/animation.yaml` references them with these exact names.
DEFAULT_UPSTREAM_CONFIG = Path("configs") / "prompts" / "animation.yaml"


@dataclass(frozen=True)
class EchoMimicArgs:
    """Knobs the baseline exposes on its CLI. Mirror upstream's
    `infer_audio2vid.py` argparse so the runner can surface them uniformly
    for ablation / sweep."""
    width:                  int   = 512        # -W
    height:                 int   = 512        # -H
    fps:                    int   = 25         # --fps; matches TalkVid/HDTF
    cfg:                    float = 2.5        # --cfg
    steps:                  int   = 30         # --steps
    seed:                   int   = 420        # --seed (upstream demo default)
    sample_rate:            int   = 16000      # --sample_rate (whisper)
    context_frames:         int   = 12         # --context_frames
    context_overlap:        int   = 3          # --context_overlap
    facemusk_dilation_ratio: float = 0.1
    facecrop_dilation_ratio: float = 0.5


# ---------------------------------------------------------------------------
# File prep
# ---------------------------------------------------------------------------
def _extract_frame(video_path: Path, frame_idx: int, out_png: Path) -> None:
    """Same pattern as the other adapters: random-access seek with a
    sequential-decode fallback for codecs that mis-handle keyframe
    boundaries."""
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


def _extract_audio(src: Path, out_wav: Path, duration_s: float) -> None:
    """Extract first `duration_s` of `src` as mono-16 kHz WAV. ffmpeg
    accepts both video containers (HDTF muxed) and standalone .wav
    (TalkVid sidecar)."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-t", str(duration_s),
        "-i", str(src),
        "-vn",                          # no video
        "-acodec", "pcm_s16le",
        "-ac", "1",                     # mono
        "-ar", "16000",                 # 16 kHz (whisper-tiny expects this)
        str(out_wav),
    ]
    subprocess.run(cmd, check=True)


def _make_patched_config(
    upstream_config: Path,
    patched_config:  Path,
    source_png:      Path,
    audio_wav:       Path,
) -> None:
    """Read upstream's animation.yaml, replace `test_cases` with a single
    mapping {source.png: [audio.wav]} using absolute paths, and write the
    patched copy. Other keys (pretrained_*_path, inference_config,
    weight_dtype) stay untouched — they're relative and resolve from the
    subprocess's cwd (= impl_dir)."""
    cfg = OmegaConf.load(upstream_config)
    cfg.test_cases = {
        str(source_png.resolve()): [str(audio_wav.resolve())],
    }
    patched_config.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, patched_config)


# ---------------------------------------------------------------------------
# Shell out
# ---------------------------------------------------------------------------
def _run_echomimic_cli(
    impl_dir:      Path,
    patched_cfg:   Path,
    duration_s:    float,
    conda_env:     str,
    args:          EchoMimicArgs,
) -> Path:
    """Run EchoMimic's infer_audio2vid.py in its conda env. Returns the
    path to the generated mp4 (the `_withaudio` variant — that's the one
    with audio muxed in, the version we want to keep).

    `cwd=impl_dir` so the relative paths in the patched config
    (`pretrained_weights/...`, `configs/inference/inference_v2.yaml`) and
    the relative output dir (`output/<date>/...`) resolve.
    """
    # `-L` is total frames the diffusion pipeline generates. Round up so we
    # always cover the requested duration; the `_withaudio` mux trims to
    # the audio's actual length.
    num_frames = max(1, int(round(duration_s * args.fps)))

    cmd = [
        "conda", "run", "--no-capture-output", "-n", conda_env,
        "python", "infer_audio2vid.py",
        "--config",                  str(patched_cfg.resolve()),
        "-W",                        str(args.width),
        "-H",                        str(args.height),
        "-L",                        str(num_frames),
        "--seed",                    str(args.seed),
        "--cfg",                     str(args.cfg),
        "--steps",                   str(args.steps),
        "--sample_rate",             str(args.sample_rate),
        "--fps",                     str(args.fps),
        "--context_frames",          str(args.context_frames),
        "--context_overlap",         str(args.context_overlap),
        "--facemusk_dilation_ratio", str(args.facemusk_dilation_ratio),
        "--facecrop_dilation_ratio", str(args.facecrop_dilation_ratio),
    ]
    subprocess.run(cmd, check=True, cwd=str(impl_dir))

    # EchoMimic writes:
    #   output/<date>/<HHMM--seed-WxH>/<ref>_<audio>_<H>x<W>_<cfg>_<HHMM>_withaudio.mp4
    # under cwd. We glob for `*_withaudio.mp4` (the muxed-audio version)
    # and grab the newest by mtime — a per-sample subprocess produces
    # exactly one such file.
    out_root = impl_dir / "output"
    candidates = sorted(out_root.rglob("*_withaudio.mp4"),
                        key=lambda p: p.stat().st_mtime)
    if not candidates:
        # Fall back to the no-audio version (some failure modes skip the
        # mux step but still produce the silent core mp4).
        candidates = sorted(out_root.rglob("*.mp4"),
                            key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise RuntimeError(
            f"EchoMimic produced no mp4 under {out_root}. "
            f"Check infer_audio2vid.py's stderr above."
        )
    return candidates[-1]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run_one(
    sample:        EvalSample,
    ref_frame_idx: int,
    impl_dir:      Path,
    output_dir:    Path,
    scratch:       Path,
    conda_env:     str = "echomimic",
    upstream_config: Path = DEFAULT_UPSTREAM_CONFIG,
    args:          EchoMimicArgs = EchoMimicArgs(),
) -> Path:
    """Generate one sample end-to-end. Returns the path to `panel.mp4`.

    Layout:
        scratch/<sample_id>/source.png       # ref frame
        scratch/<sample_id>/audio.wav        # driver audio, 16 kHz mono
        scratch/<sample_id>/animation.yaml   # patched test_cases config
        impl/output/<date>/<…>/*.mp4         # EchoMimic raw output
        output_dir/samples/<sample_id>/panel.mp4
    """
    if not (0 <= ref_frame_idx < sample.ref_clip.n_frames):
        raise ValueError(
            f"ref_frame_idx={ref_frame_idx} out of range for clip "
            f"{sample.ref_clip.clip_id} (n_frames={sample.ref_clip.n_frames})"
        )

    work        = scratch / sample.sample_id
    source_png  = work / "source.png"
    audio_wav   = work / "audio.wav"
    patched_cfg = work / "animation.yaml"

    _extract_frame(sample.ref_clip.video_path, ref_frame_idx, source_png)

    audio_src = sample.driver_clip.audio_path or sample.driver_clip.video_path
    _extract_audio(audio_src, audio_wav, sample.clip_duration_s)

    _make_patched_config(
        upstream_config = impl_dir / upstream_config,
        patched_config  = patched_cfg,
        source_png      = source_png,
        audio_wav       = audio_wav,
    )

    raw_mp4 = _run_echomimic_cli(
        impl_dir    = impl_dir,
        patched_cfg = patched_cfg,
        duration_s  = sample.clip_duration_s,
        conda_env   = conda_env,
        args        = args,
    )

    final_dir = output_dir / "samples" / sample.sample_id
    final_dir.mkdir(parents=True, exist_ok=True)
    final_mp4 = final_dir / "panel.mp4"
    shutil.move(str(raw_mp4), str(final_mp4))
    return final_mp4
