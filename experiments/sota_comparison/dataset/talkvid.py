"""
TalkVid adapter for the SOTA comparison suite.

Clip enumeration source
-----------------------
**The TalkVid SOTA manifest is built from Marionette's validation set**
(`data/derived/val_clips.json`), not the full `data/talkvid/talkvid/` disk.
Two reasons:

  1. Centralisation — baselines (SadTalker, LivePortrait, …) under this
     folder and Marionette's own `marionette_eval/` share the same clip
     pool. A single identity on disk maps to the same panel folder in
     every baseline's output tree AND in Marionette's eval tree, so the
     cross-model comparison is 1-to-1 at the clip level.
  2. Cost — the full TalkVid mirror has ~10k clips. Walking + ffprobing
     all of them to then throw 90% away (one clip per identity) is
     ~30 minutes of subprocess time per rebuild for zero gain.

`clip_list_path` is a constructor parameter, defaulting to
`data/derived/val_clips.json`, so a future ablation that wants a different
subset (train set, a custom curated JSON) can drop it in without
subclassing.

On-disk layout at `data/talkvid/`:

    data/talkvid/
    ├── talkvid/                    ← video clips (face-cropped, 512×512, 25fps)
    │   ├── 0-Cekahx2rw_NA_1187.180_1202.660.mp4
    │   └── ...
    └── audio/                      ← sibling WAV files; the mp4s are silent
        ├── 0-Cekahx2rw_NA_1187.180_1202.660.wav
        └── ...

Identity parsing
----------------
Clip names follow `<YOUTUBE_ID>_NA_<start>_<end>`. Identity = the prefix
before the first `_NA_`. Clips that don't contain the `_NA_` sentinel (a
handful of outliers named things like `gender_Female_videovideo_...`) each
become their own single-clip identity — safe fallback since we have no
better signal. This matches the rule
`marionette.experiments.marionette_eval.pairing` already uses for the same
data.

Audio
-----
TalkVid's mp4s are silent — audio lives at `<root>/audio/<clip_id>.wav`.
`_audio_path_for` populates `BenchmarkClip.audio_path` with that sibling so
downstream audio-driven baselines (SadTalker, …) pull audio from the WAV
file instead of the muxed video stream.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Optional

from .base import BenchmarkVideoDataset


IDENTITY_SEP = "_NA_"
DEFAULT_CLIP_LIST = Path("data/derived/val_clips.json")


class TalkVidDataset(BenchmarkVideoDataset):
    """TalkVid-over-Marionette-val-split layout under
    `<root>/talkvid/*.mp4` + `<root>/audio/<clip_id>.wav`, with the clip
    list sourced from a JSON-array list file (default: Marionette's
    `val_clips.json`)."""

    def __init__(
        self,
        root: Path = Path("data/talkvid"),
        manifest_path: Path | None = None,
        clip_list_path: Path = DEFAULT_CLIP_LIST,
    ):
        super().__init__(root, manifest_path)
        self.clip_list_path = Path(clip_list_path)

    @property
    def name(self) -> str:
        return "talkvid"

    @property
    def video_dir(self) -> Path:
        return self.root / "talkvid"

    @property
    def audio_dir(self) -> Path:
        return self.root / "audio"

    def _discover_clips(self) -> Iterator[tuple[str, str, Path]]:
        if not self.clip_list_path.is_file():
            raise FileNotFoundError(
                f"TalkVid clip list not found: {self.clip_list_path}. "
                f"Expected a JSON array of clip IDs (no extension), as written "
                f"by `scripts/manifest/partition_dataset.py`."
            )

        with open(self.clip_list_path) as f:
            clip_ids = json.load(f)

        # Sorted for reproducible manifest order regardless of the source
        # list's ordering. Matches HDTF's behaviour.
        for clip_id in sorted(clip_ids):
            clip_id = str(clip_id).strip()
            if not clip_id:
                continue
            video_path = self.video_dir / f"{clip_id}.mp4"
            if not video_path.exists():
                # A clip listed in the val set but missing on disk is a
                # data-integrity problem, not something to silently skip —
                # surface it immediately so the mirror can be fixed.
                raise FileNotFoundError(
                    f"Clip listed in {self.clip_list_path} is missing from "
                    f"disk: {video_path}"
                )
            identity_id = clip_id.split(IDENTITY_SEP, 1)[0]
            yield clip_id, identity_id, video_path

    def _audio_path_for(self, clip_id: str, video_path: Path) -> Optional[Path]:
        """Sibling WAV path. Existence is not checked here — if a specific
        clip's audio is missing, the baseline's adapter fails loudly at
        ffmpeg-extract time rather than at enumeration time."""
        return self.audio_dir / f"{clip_id}.wav"
