"""Video/frame reading. Supports mp4 (via decord) and directories of images.

decord is lazy-imported at call time, not at module scope: loading libav at
import time before torch initialises CUDA can segfault on this env, and
`loki.utils` is imported eagerly by many modules that never actually
touch a video file."""
from pathlib import Path

import numpy as np
import cv2


class FrameReader:
    """Minimal sequence-of-frames reader for directories full of image files.
    Behaves like `decord.VideoReader` for indexing."""

    def __init__(self, video_path):
        self.frame_list = sorted(list(Path(video_path).glob("*.*")))

    def __len__(self):
        return len(self.frame_list)

    def __getitem__(self, index):
        img = cv2.imread(str(self.frame_list[index]))[..., [2, 1, 0]]
        return img


def load_frame(video_path, frame_id):
    """Load one frame from either a video file (mp4) or a directory of images."""
    if Path(video_path).is_dir():
        reader = FrameReader(video_path)
    else:
        from decord import VideoReader
        reader = VideoReader(str(video_path))
    if frame_id >= len(reader):
        frame_id = len(reader) - 1
    frame = reader[frame_id]
    if not isinstance(frame, np.ndarray):
        frame = frame.asnumpy()
    return frame
