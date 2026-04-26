"""
HDTF adapter for the SOTA comparison suite.

On-disk layout observed at `data/benchmark/hdtf/clips/`:

    data/benchmark/hdtf/clips/
    ├── WDA_LucilleRoybal_Allard_000_0_80.mp4
    ├── WDA_LucilleRoybal_Allard_000_1053_1133.mp4
    ├── WRA_MikeRogers_000_0_80.mp4
    └── ...

Filename convention:
    <PREFIX>_<SPEAKER_NAME_WITH_UNDERSCORES>[<SESSION_DIGIT>]_<SESSION>_<START>_<END>.mp4

The last three underscore-separated tokens are always (session_id, start,
end); everything before them is the identity. Speaker names with multiple
tokens (e.g. `LucilleRoybal_Allard`) are supported by the "strip last three
tokens" rule rather than a fixed token count.

Trailing-digit disambiguation: some speakers appear in the mirror with a
trailing digit on the speaker-name token (e.g. `BarbaraLee0`, `BarbaraLee1`
are the same physical speaker in different sessions). After extracting the
identity tokens we strip trailing ASCII digits from the **last** token so
`WDA_BarbaraLee0` and `WDA_BarbaraLee1` collapse to `WDA_BarbaraLee`. If
the trailing-digit strip empties the token entirely (session index written
as its own token), that token is dropped.

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
            identity_tokens = parts[:-3]

            # Collapse `BarbaraLee0`/`BarbaraLee1` variants into one identity
            # by stripping trailing digits from the last name token. A token
            # that was pure digits becomes empty and is dropped.
            identity_tokens[-1] = identity_tokens[-1].rstrip("0123456789")
            if not identity_tokens[-1]:
                identity_tokens = identity_tokens[:-1]
            if not identity_tokens:
                # No non-digit name content left — skip rather than emit an
                # empty identity_id.
                continue

            identity_id = "_".join(identity_tokens)
            yield clip_id, identity_id, video_path
