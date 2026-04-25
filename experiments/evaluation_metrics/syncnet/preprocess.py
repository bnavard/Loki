"""
Preprocessing utilities for SyncNet evaluation.

Video: extract frames at 25 FPS, resize to 224×224.
Audio: compute 13-coefficient MFCC at 100 Hz (10 ms step).

Window alignment:
  - 5 video frames = 1 SyncNet window
  - 4 MFCC frames per video frame → 20 MFCC frames per window
  - For video frame index i, the corresponding MFCC slice is [i*4 : i*4 + 20]
"""

from typing import Optional

import numpy as np
import cv2


def extract_video_frames(
    video_path: str,
    target_size: int = 224,
    max_frames: Optional[int] = None,
) -> np.ndarray:
    """Decode all frames from an mp4, resize to (target_size, target_size).

    Returns: (T, 3, H, W) float32 in [0, 1].
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (target_size, target_size))
        frames.append(frame)
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")

    arr = np.stack(frames, axis=0).astype(np.float32) / 255.0  # (T, H, W, 3)
    return arr.transpose(0, 3, 1, 2)  # (T, 3, H, W)


def compute_mfcc(
    audio_path: str,
    sample_rate: int = 16000,
    num_cepstral: int = 13,
    win_length: float = 0.025,
    win_step: float = 0.01,
) -> np.ndarray:
    """Compute MFCC features from a WAV file.

    Returns: (13, N_mfcc_frames) float32, where N_mfcc_frames ≈ audio_duration / win_step.
    """
    try:
        from python_speech_features import mfcc as compute_mfcc_feats
    except ImportError:
        raise ImportError(
            "python_speech_features is required for MFCC computation. "
            "Install with: pip install python-speech-features"
        )

    import soundfile as sf
    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]

    # Resample if needed
    if sr != sample_rate:
        import scipy.signal
        n_target = int(len(audio) * sample_rate / sr)
        audio = scipy.signal.resample(audio, n_target)

    mfcc_feats = compute_mfcc_feats(
        audio, samplerate=sample_rate, numcep=num_cepstral,
        winlen=win_length, winstep=win_step,
    )  # (N_mfcc_frames, 13)

    return mfcc_feats.T.astype(np.float32)  # (13, N_mfcc_frames)


def build_syncnet_windows(
    video_frames: np.ndarray,
    mfcc: np.ndarray,
    window_size: int = 5,
    mfcc_per_frame: int = 4,
):
    """Yield aligned (video_window, mfcc_window) pairs.

    Args:
        video_frames: (T, 3, H, W) float32
        mfcc: (13, N_mfcc) float32
        window_size: number of consecutive video frames per SyncNet window (5)
        mfcc_per_frame: MFCC frames per video frame (4 at 100Hz / 25fps)

    Yields:
        (video_window, mfcc_window) where:
          video_window: (3, 5, H, W) float32
          mfcc_window:  (1, 13, 20) float32
    """
    n_video = video_frames.shape[0]
    n_mfcc = mfcc.shape[1]
    mfcc_window_size = window_size * mfcc_per_frame  # 20

    for i in range(n_video - window_size + 1):
        mfcc_start = i * mfcc_per_frame
        mfcc_end = mfcc_start + mfcc_window_size
        if mfcc_end > n_mfcc:
            break

        video_window = video_frames[i:i + window_size]      # (5, 3, H, W)
        video_window = video_window.transpose(1, 0, 2, 3)   # (3, 5, H, W)

        mfcc_window = mfcc[:, mfcc_start:mfcc_end]          # (13, 20)
        mfcc_window = mfcc_window[np.newaxis, :, :]          # (1, 13, 20)

        yield video_window, mfcc_window
