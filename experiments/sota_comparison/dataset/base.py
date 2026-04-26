"""
Canonical record schema + base class for benchmark datasets used in the SOTA
comparison suite.

Each external baseline (SadTalker, LivePortrait, …) consumes data in its own
on-disk layout at inference time (single PNG + WAV, or a JSON manifest, or a
directory of pre-cropped frames, etc.). We decouple "what's on disk" from
"what a baseline wants" by:

  1. `BenchmarkClip` — a stable per-clip record describing the source video
     on disk: identity, path, n_frames, fps, resolution.
  2. `BenchmarkVideoDataset` — an ABC that enumerates `BenchmarkClip`s from
     a dataset root. Subclasses parse dataset-specific layouts (flat mp4s,
     nested `<id>/<vid>/<utt>.mp4`, etc.).
  3. Per-baseline `adapter.py` — given a `BenchmarkClip` + protocol args,
     materialize whatever files the baseline's inference CLI wants (via
     ffmpeg / cv2) in a temp dir, then shell out.

This module owns (1) and (2). Adapters live under each baseline folder.

No torch imports here — this is a pure metadata catalog. Adapters build the
tensors their baseline wants.
"""
from __future__ import annotations

import json
import subprocess
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Optional


@dataclass(frozen=True)
class BenchmarkClip:
    """One video clip's metadata. Stable, hashable, JSON-serialisable.

    `clip_id` must be unique within a dataset and stable across probe runs
    (composed from the on-disk path, not a random UUID). It is the key the
    pairing logic uses to join ref/driver selections back to their records.

    `uid` is the curated-manifest identifier (`id_0457` style, 4-zero-padded)
    assigned by `benchmark_manifest.build_benchmark_manifest` and consumed by
    `pairing.build_samples` to name `EvalSample.sample_id`. It is `None` on
    records loaded from the raw ffprobe cache and populated on records
    loaded from the curated manifest; downstream pairing / runner code
    requires it to be non-None (raises otherwise).

    `audio_path` carries the source of audio for clips where it lives OUTSIDE
    the video container (e.g. TalkVid, whose mp4s are silent and whose audio
    sits as sibling `.wav` files under `data/talkvid/audio/<clip_id>.wav`).
    Baseline adapters that need audio (SadTalker, any audio-driven model)
    should prefer `audio_path` when set and fall back to `video_path` when
    it is None (HDTF, VoxCeleb2, CelebV-HQ — audio muxed into the mp4).
    """
    clip_id:     str
    identity_id: str
    video_path:  Path
    n_frames:    int
    fps:         float
    width:       int
    height:      int
    uid:         Optional[str] = None
    audio_path:  Optional[Path] = None

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.fps if self.fps > 0 else 0.0

    def to_json_dict(self) -> dict:
        d = asdict(self)
        d["video_path"] = str(self.video_path)
        # Only emit audio_path when set — keeps HDTF/VoxCeleb/CelebV-HQ
        # manifests (audio muxed in mp4) free of noisy `"audio_path": null`
        # lines, so the committed JSON diff stays minimal across dataset
        # adapters that don't use the field.
        if self.audio_path is None:
            d.pop("audio_path", None)
        else:
            d["audio_path"] = str(self.audio_path)
        return d

    @classmethod
    def from_json_dict(cls, d: dict) -> "BenchmarkClip":
        audio = d.get("audio_path")
        return cls(
            clip_id     = d["clip_id"],
            identity_id = d["identity_id"],
            video_path  = Path(d["video_path"]),
            n_frames    = int(d["n_frames"]),
            fps         = float(d["fps"]),
            width       = int(d["width"]),
            height      = int(d["height"]),
            uid         = d.get("uid"),
            audio_path  = Path(audio) if audio else None,
        )


def probe_video(video_path: Path) -> tuple[int, float, int, int]:
    """Return `(n_frames, fps, width, height)` via ffprobe.

    Single subprocess call, parsed as JSON. Raises on ffprobe failure so the
    caller sees the clip rather than silently skipping it — a corrupt video
    in the manifest is a bug to fix, not a sample to drop.
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames,r_frame_rate,width,height,duration",
        "-of", "json",
        str(video_path),
    ]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    meta = json.loads(out)["streams"][0]

    # r_frame_rate comes as "num/den"; evaluate to float.
    num, _, den = meta["r_frame_rate"].partition("/")
    fps = float(num) / float(den) if den else float(num)

    # nb_frames is not always populated (some mp4s lack it); fall back to
    # duration * fps. Both branches round down — consumers that need an
    # exact count re-probe with `-count_frames`.
    if "nb_frames" in meta and meta["nb_frames"] not in ("N/A", "0"):
        n_frames = int(meta["nb_frames"])
    else:
        n_frames = int(float(meta["duration"]) * fps)

    return n_frames, fps, int(meta["width"]), int(meta["height"])


class BenchmarkVideoDataset(ABC):
    """Base class: enumerates `BenchmarkClip`s under a dataset root, with a
    JSON manifest cache so ffprobe runs once per clip per dataset.

    Subclasses implement `_discover_clips(root)` — walk the dataset-specific
    directory layout and yield `(clip_id, identity_id, video_path)` triples.
    The base class handles probing and caching.
    """

    def __init__(self, root: Path, manifest_path: Path | None = None):
        self.root = Path(root)
        self.manifest_path = Path(manifest_path) if manifest_path else None
        self._clips: list[BenchmarkClip] | None = None

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------
    @property
    @abstractmethod
    def name(self) -> str:
        """Short dataset name, e.g. "hdtf". Used in the default manifest
        filename and the running table."""

    @abstractmethod
    def _discover_clips(self) -> Iterator[tuple[str, str, Path]]:
        """Yield `(clip_id, identity_id, video_path)` for each video found
        under `self.root`. Ordering is preserved into the manifest so runs
        are reproducible."""

    def _audio_path_for(self, clip_id: str, video_path: Path) -> Optional[Path]:
        """Return an external audio path for this clip, or None if audio is
        muxed into the mp4. Default: None (HDTF / VoxCeleb2 / CelebV-HQ all
        mux). Override in subclasses like TalkVid whose audio lives as
        sibling `.wav` files."""
        return None

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------
    def load(self, rebuild: bool = False) -> list[BenchmarkClip]:
        """Return the full clip list, probing if needed. Cached to
        `manifest_path` (default: `data/derived/<name>_manifest.json`).
        """
        if self._clips is not None and not rebuild:
            return self._clips

        manifest = self._default_manifest_path() if self.manifest_path is None else self.manifest_path

        if manifest.exists() and not rebuild:
            with open(manifest) as f:
                raw = json.load(f)
            self._clips = [BenchmarkClip.from_json_dict(d) for d in raw]
            return self._clips

        clips: list[BenchmarkClip] = []
        for clip_id, identity_id, video_path in self._discover_clips():
            n_frames, fps, w, h = probe_video(video_path)
            clips.append(BenchmarkClip(
                clip_id=clip_id, identity_id=identity_id,
                video_path=video_path, n_frames=n_frames, fps=fps,
                width=w, height=h,
                audio_path=self._audio_path_for(clip_id, video_path),
            ))

        manifest.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest, "w") as f:
            json.dump([c.to_json_dict() for c in clips], f, indent=2)

        self._clips = clips
        return clips

    def _default_manifest_path(self) -> Path:
        return Path("data/derived") / f"{self.name}_manifest.json"

    def get_by_id(self, clip_id: str) -> BenchmarkClip:
        for c in self.load():
            if c.clip_id == clip_id:
                return c
        raise KeyError(f"clip_id={clip_id!r} not found in {self.name}")

    def by_identity(self) -> dict[str, list[BenchmarkClip]]:
        """Group clips by identity_id. Useful for cross-identity pairing
        (draw ref / driver from different groups) and for per-identity
        same-identity sampling."""
        groups: dict[str, list[BenchmarkClip]] = {}
        for c in self.load():
            groups.setdefault(c.identity_id, []).append(c)
        return groups
