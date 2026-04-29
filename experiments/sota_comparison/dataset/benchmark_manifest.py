"""
Curated benchmark manifest: one clip per identity, capped at `n_samples_cap`
per dataset, each entry tagged with a stable `id_XXXX` UID.

Motivation
----------
Raw dataset enumeration (via `BenchmarkVideoDataset.load`) gives every clip on
disk — 15k+ for HDTF. The actual paper comparison wants
(a) one clip per identity so an identity is only evaluated once, and (b) a
stable UID per identity so the same `id_0457` refers to the same physical
person across every baseline's output tree on disk.

This module produces, saves, and loads that curated pool. Once
`build_benchmark_manifest` is written to
`experiments/sota_comparison/manifests/<dataset>.json`, the UIDs are frozen
for that dataset + that seed; downstream runners (`sadtalker/run_inference.py`,
future baselines, Marionette's own eval) all consume the same manifest and
inherit the same UID scheme. That is what makes a single glob over
`outputs/**/samples/<uid>/panel.mp4` align every model against every other
model at the identity level.

Selection rule (`longest_clip_per_identity`)
--------------------------------------------
For each identity group in the raw enumeration, pick the clip with the most
frames. Longest clip is the safest default: it maximises the chance of
surviving `clip_duration_s` filtering later and gives baselines the most
content to work with. Alternatives (first alphabetical / seeded random) can
be added if a future experiment wants them.

Sampling
--------
If the identity count exceeds `n_samples_cap`, `n_samples_cap` identities
are drawn uniformly under `seed`. Otherwise all identities are kept. UIDs
are assigned in alphabetical order of `identity_id` over the retained set
so a given `(dataset, seed)` tuple reproduces the exact same manifest.

File layout
-----------
    experiments/sota_comparison/manifests/<dataset>.json

This directory is checked into git so a published paper's identity pool is
permanent. Rebuild requires explicit `--rebuild` at the CLI.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np

from .base import BenchmarkClip, BenchmarkVideoDataset


MANIFEST_DIR = Path("experiments/sota_comparison/manifests")
SelectionRule = Literal["longest_clip_per_identity"]


def manifest_path(dataset_name: str) -> Path:
    """Canonical location for `<dataset>.json`."""
    return MANIFEST_DIR / f"{dataset_name}.json"


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def _pick_longest_per_identity(
    clips: list[BenchmarkClip],
) -> dict[str, BenchmarkClip]:
    """Group by identity; keep the clip with max `n_frames` per group.

    Ties are broken by `clip_id` (alphabetical) to keep the selection
    deterministic regardless of the input ordering.
    """
    best: dict[str, BenchmarkClip] = {}
    for c in clips:
        cur = best.get(c.identity_id)
        if cur is None:
            best[c.identity_id] = c
            continue
        if (c.n_frames, c.clip_id) > (cur.n_frames, cur.clip_id):
            best[c.identity_id] = c
    return best


def build_benchmark_manifest(
    dataset:         BenchmarkVideoDataset,
    n_samples_cap:   int = 1000,
    seed:            int = 42,
    selection_rule:  SelectionRule = "longest_clip_per_identity",
) -> tuple[list[BenchmarkClip], dict]:
    """Return `(clips_with_uid, meta)`.

    `clips_with_uid` is sorted by `uid` (which itself follows alphabetical
    `identity_id` order after sampling). `meta` is a small metadata dict
    suitable for dumping as the top of the manifest JSON.
    """
    raw = dataset.load()

    if selection_rule == "longest_clip_per_identity":
        picked = _pick_longest_per_identity(raw)
    else:
        raise ValueError(f"Unknown selection_rule: {selection_rule!r}")

    ids_sorted = sorted(picked.keys())

    rng = np.random.default_rng(seed)
    if len(ids_sorted) > n_samples_cap:
        idx = rng.choice(len(ids_sorted), size=n_samples_cap, replace=False)
        ids_sampled = sorted(ids_sorted[int(i)] for i in idx)
    else:
        ids_sampled = ids_sorted

    # UIDs re-assigned in alphabetical order of identity_id on the retained
    # set → a given (dataset, seed) tuple reproduces the exact same UIDs.
    clips_with_uid: list[BenchmarkClip] = []
    for i, ident in enumerate(ids_sampled):
        picked_clip = picked[ident]
        # BenchmarkClip is frozen — rebuild it with the uid field set while
        # forwarding every other field (audio_path included, for datasets
        # where audio lives outside the mp4 container).
        clips_with_uid.append(BenchmarkClip(
            clip_id     = picked_clip.clip_id,
            identity_id = picked_clip.identity_id,
            video_path  = picked_clip.video_path,
            n_frames    = picked_clip.n_frames,
            fps         = picked_clip.fps,
            width       = picked_clip.width,
            height      = picked_clip.height,
            uid         = f"id_{i:04d}",
            audio_path  = picked_clip.audio_path,
        ))

    meta = {
        "dataset":            dataset.name,
        "n_total_identities": len(ids_sorted),
        "n_identities":       len(clips_with_uid),
        "n_samples_cap":      n_samples_cap,
        "seed":               seed,
        "selection_rule":     selection_rule,
    }
    return clips_with_uid, meta


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def save_manifest(
    path:           Path,
    clips_with_uid: list[BenchmarkClip],
    meta:           dict,
) -> None:
    """Write `meta + clips` to `path` as JSON. Creates parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **meta,
        "clips": [c.to_json_dict() for c in clips_with_uid],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_manifest(path: Path) -> tuple[list[BenchmarkClip], dict]:
    """Return `(clips_with_uid, meta)` — same shape as `build_benchmark_manifest`.

    Raises a clear error if any clip in the manifest is missing a uid, so a
    corrupted / half-written manifest fails loudly at load time.
    """
    path = Path(path)
    with open(path) as f:
        payload = json.load(f)

    raw_clips = payload.pop("clips")
    clips = [BenchmarkClip.from_json_dict(d) for d in raw_clips]
    missing = [c.clip_id for c in clips if not c.uid]
    if missing:
        raise ValueError(
            f"Manifest {path} has {len(missing)} clips without a uid "
            f"(first: {missing[:3]}). Rebuild with `build_manifest.py --rebuild`."
        )
    return clips, payload


def load_by_dataset(dataset_name: str) -> tuple[list[BenchmarkClip], dict]:
    """Convenience loader: resolve the default manifest path by dataset name
    and read it. Raises with an actionable hint if the manifest doesn't
    exist yet (must be built first)."""
    path = manifest_path(dataset_name)
    if not path.exists():
        raise FileNotFoundError(
            f"No manifest at {path}. Build it first:\n"
            f"    PYTHONPATH=. python experiments/sota_comparison/dataset/"
            f"build_manifest.py --dataset {dataset_name}"
        )
    return load_manifest(path)
