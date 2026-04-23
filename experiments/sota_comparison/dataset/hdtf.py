"""
HDTF adapter for the SOTA comparison suite.

On-disk layout observed at `data/benchmark/hdtf/clips/`:

    data/benchmark/hdtf/clips/
    ├── WDA_LucilleRoybal_Allard_000_0_80.mp4
    ├── WDA_LucilleRoybal_Allard_000_1053_1133.mp4
    ├── WRA_MikeRogers_000_0_80.mp4
    └── ...

Filename convention:
    <PREFIX>_<SPEAKER_NAME_WITH_UNDERSCORES>_<SESSION>_<START>_<END>.mp4

The last three underscore-separated tokens are always (session_id, start,
end); everything before them is the identity. Speaker names with multiple
tokens (e.g. `LucilleRoybal_Allard`) are supported by the "strip last three
tokens" rule rather than a fixed token count.

We filter out clips whose filenames start with `RD_Radio` — those entries
don't carry a consistent identity-parseable structure in this mirror of the
dataset and are dropped rather than special-cased.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from .base import BenchmarkVideoDataset

SKIP_PREFIX = "RD_Radio"


class HDTFDataset(BenchmarkVideoDataset):
    """Flat-mp4 HDTF layout under `<root>/clips/*.mp4`."""

    def __init__(
        self,
        root: Path = Path("data/benchmark/hdtf"),
        manifest_path: Path | None = None,
    ):
        super().__init__(root, manifest_path)

    @property
    def name(self) -> str:
        return "hdtf"

    def _discover_clips(self) -> Iterator[tuple[str, str, Path]]:
        clips_dir = self.root / "clips"
        if not clips_dir.is_dir():
            raise FileNotFoundError(
                f"HDTF clips directory not found: {clips_dir}. "
                f"Expected layout: {self.root}/clips/*.mp4"
            )

        # Sorted for reproducible manifest order across filesystems / runs.
        for video_path in sorted(clips_dir.glob("*.mp4")):
            clip_id = video_path.stem
            if clip_id.startswith(SKIP_PREFIX):
                continue

            parts = clip_id.split("_")
            # Expected layout carries at least 4 tokens: one for identity +
            # three for (session, start, end). Anything shorter is malformed
            # — skip rather than guess, so the manifest stays clean.
            if len(parts) < 4:
                continue
            identity_id = "_".join(parts[:-3])
            yield clip_id, identity_id, video_path
