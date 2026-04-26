"""Audio utilities shared by the dataset (per-sample window loading) and
inference (whole-clip window loading).

The talking-head model consumes a per-frame audio window of
`samples_per_frame * (1 + 2 * context_frames)` samples — i.e. the current
frame's audio ± `context_frames` neighbours on each side — which AudioEncoder
maps to cross-attention tokens.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

try:
    import soundfile as sf
    _HAS_SOUNDFILE = True
except ImportError:
    _HAS_SOUNDFILE = False


SAMPLE_RATE = 16_000


def load_audio_mono(audio_path: Path, expected_len: int) -> Optional[np.ndarray]:
    """Read a wav file, mix to mono, pad to `expected_len` samples.

    Returns None if the file is missing or soundfile isn't available.
    """
    if not _HAS_SOUNDFILE or not audio_path.exists():
        return None
    audio, _ = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio[:, 0]
    if len(audio) < expected_len:
        audio = np.pad(audio, (0, expected_len - len(audio)))
    return audio


def frame_window(
    audio: Optional[np.ndarray],
    frame_id: int,
    n_total_frames: int,
    samples_per_frame: int,
    context_frames: int,
) -> np.ndarray:
    """Centered audio window covering `frame_id` ± `context_frames`.

    Out-of-range neighbouring frames are clamped to the nearest valid frame,
    matching how the dataset constructs training windows. Silence if `audio`
    is None.

    Returns (samples_per_frame * (1 + 2*context_frames),) float32.
    """
    window_samples = samples_per_frame * (1 + 2 * context_frames)
    if audio is None:
        return np.zeros(window_samples, dtype=np.float32)

    window = np.zeros(window_samples, dtype=np.float32)
    for i, f in enumerate(range(frame_id - context_frames, frame_id + context_frames + 1)):
        f_clamp = max(0, min(f, n_total_frames - 1))
        src = f_clamp * samples_per_frame
        dst = i * samples_per_frame
        window[dst:dst + samples_per_frame] = audio[src:src + samples_per_frame]
    return window


def load_clip_audio_windows(
    audio_path: Optional[Path],
    n_frames: int,
    samples_per_frame: int,
    context_frames: int,
) -> np.ndarray:
    """Whole-clip convenience wrapper: stack `frame_window` over [0, n_frames).
    Returns (n_frames, window_samples) float32. Zeros if audio is missing."""
    audio = load_audio_mono(
        audio_path, expected_len=n_frames * samples_per_frame,
    ) if audio_path is not None else None
    return np.stack([
        frame_window(audio, t, n_frames, samples_per_frame, context_frames)
        for t in range(n_frames)
    ], axis=0)
