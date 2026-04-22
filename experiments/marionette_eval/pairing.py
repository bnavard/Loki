"""
Pure pairing helpers for the Marionette evaluation scripts.

Builds the deterministic sample lists consumed by the cross-identity and
same-identity eval runners. All randomness funnels through a single
`np.random.default_rng(seed)` so the same seed reproduces the same schedule.

Clip IDs follow `<YOUTUBE_ID>_NA_<start>_<end>` — the identity is the
YouTube ID (the segment before `_NA_`). Clips from the same YouTube video
share an identity; the cross-identity pairing treats them as one.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


IDENTITY_SEP = "_NA_"


@dataclass(frozen=True)
class CrossSample:
    ref_identity:      str
    driver_identity:   str
    ref_clip:          str
    driver_clip:       str
    ref_frame_idx:     int
    driver_start_idx:  int
    ref_clip_len:      int
    driver_clip_len:   int


@dataclass(frozen=True)
class SameSample:
    identity:          str
    clip:              str
    ref_frame_idx:     int
    driver_start_idx:  int
    clip_len:          int


def identity_of(clip_id: str) -> str:
    """Return the YouTube ID for a clip. Clips that don't contain the `_NA_`
    sentinel are treated as their own identity (safe fallback for any future
    naming variant)."""
    return clip_id.split(IDENTITY_SEP, 1)[0]


def group_by_identity(clip_ids: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for c in clip_ids:
        groups.setdefault(identity_of(c), []).append(c)
    return groups


def read_clip_lengths(clip_ids: list[str], flame_root: Path) -> dict[str, int]:
    """Read per-clip frame counts from `fit.npz` headers.

    Uses `np.load(..., mmap_mode='r')` + `.shape` so we don't pay the cost of
    loading the full array — just the header parse. Missing fit files yield
    length 0 (so the caller's length filter drops them cleanly).
    """
    lengths: dict[str, int] = {}
    for c in clip_ids:
        fit_path = flame_root / c / "fit.npz"
        if not fit_path.exists():
            lengths[c] = 0
            continue
        with np.load(str(fit_path), mmap_mode="r") as z:
            lengths[c] = int(z["expr"].shape[0])
    return lengths


def _usable_clips(
    clips: list[str],
    lengths: dict[str, int],
    min_len: int,
) -> list[str]:
    return [c for c in clips if lengths.get(c, 0) >= min_len]


def build_cross_identity_samples(
    clip_ids: list[str],
    flame_root: Path,
    n_frames: int,
    seed: int,
) -> tuple[list[CrossSample], dict[str, int]]:
    """Option (c): every usable identity appears exactly once as ref and exactly
    once as driver, with ref_identity ≠ driver_identity (a derangement).

    Each sample draws one clip uniformly per identity, then independently
    samples `ref_frame_idx` over the ref clip and `driver_start_idx` over the
    valid window `[0, driver_clip_len - n_frames]`.

    Returns:
        samples — list of CrossSample, ordered by ref_identity.
        stats   — dict with `n_total_identities`, `n_usable_identities`,
                  `n_dropped_short_clips`, `n_samples` for caller print.
    """
    groups = group_by_identity(clip_ids)
    lengths = read_clip_lengths(clip_ids, flame_root)

    usable_by_id: dict[str, list[str]] = {}
    n_dropped_short = 0
    for ident, clips in groups.items():
        ok = _usable_clips(clips, lengths, min_len=n_frames)
        if ok:
            usable_by_id[ident] = ok
        else:
            n_dropped_short += len(clips)

    identities = sorted(usable_by_id.keys())
    if len(identities) < 2:
        raise RuntimeError(
            f"Cross-identity pairing needs ≥2 usable identities with clips of "
            f"≥{n_frames} frames; found {len(identities)}."
        )

    rng = np.random.default_rng(seed)
    # Derangement via rejection sampling — cheap at N=125.
    n = len(identities)
    for _ in range(1000):
        perm = rng.permutation(n)
        if not np.any(perm == np.arange(n)):
            break
    else:
        raise RuntimeError("Failed to draw a derangement after 1000 attempts.")

    samples: list[CrossSample] = []
    for i, ref_ident in enumerate(identities):
        drv_ident = identities[int(perm[i])]
        ref_clip  = usable_by_id[ref_ident][int(rng.integers(0, len(usable_by_id[ref_ident])))]
        drv_clip  = usable_by_id[drv_ident][int(rng.integers(0, len(usable_by_id[drv_ident])))]
        rlen = lengths[ref_clip]
        dlen = lengths[drv_clip]
        ref_frame_idx    = int(rng.integers(0, rlen))
        driver_start_idx = int(rng.integers(0, dlen - n_frames + 1))
        samples.append(CrossSample(
            ref_identity=ref_ident, driver_identity=drv_ident,
            ref_clip=ref_clip, driver_clip=drv_clip,
            ref_frame_idx=ref_frame_idx, driver_start_idx=driver_start_idx,
            ref_clip_len=rlen, driver_clip_len=dlen,
        ))

    stats = {
        "n_total_identities":    len(groups),
        "n_usable_identities":   len(identities),
        "n_dropped_short_clips": n_dropped_short,
        "n_samples":             len(samples),
    }
    return samples, stats


def build_same_identity_samples(
    clip_ids: list[str],
    flame_root: Path,
    n_frames: int,
    samples_per_identity: int,
    min_ref_driver_gap: int,
    seed: int,
) -> tuple[list[SameSample], dict[str, int]]:
    """Every usable identity contributes `samples_per_identity` samples, each
    from a uniformly-sampled clip of that identity. Within the chosen clip:

      * `driver_start_idx ∈ [0, clip_len - n_frames]`
      * `ref_frame_idx` sampled from positions at least `min_ref_driver_gap`
        frames outside the target window `[driver_start, driver_start+n_frames)`
        so the ref never coincides with a target frame.

    Clips that are too short to satisfy the gap are skipped per-attempt (the
    whole identity is dropped only if none of its clips qualify).
    """
    groups = group_by_identity(clip_ids)
    lengths = read_clip_lengths(clip_ids, flame_root)

    min_total_len = n_frames + 2 * min_ref_driver_gap
    usable_by_id: dict[str, list[str]] = {}
    for ident, clips in groups.items():
        ok = _usable_clips(clips, lengths, min_len=min_total_len)
        if ok:
            usable_by_id[ident] = ok

    identities = sorted(usable_by_id.keys())
    rng = np.random.default_rng(seed)
    samples: list[SameSample] = []
    for ident in identities:
        for _ in range(samples_per_identity):
            clip = usable_by_id[ident][int(rng.integers(0, len(usable_by_id[ident])))]
            clen = lengths[clip]
            driver_start_idx = int(rng.integers(0, clen - n_frames + 1))
            ref_frame_idx = _sample_ref_with_gap(
                clen, driver_start_idx, n_frames, min_ref_driver_gap, rng,
            )
            samples.append(SameSample(
                identity=ident, clip=clip,
                ref_frame_idx=ref_frame_idx,
                driver_start_idx=driver_start_idx,
                clip_len=clen,
            ))

    stats = {
        "n_total_identities":   len(groups),
        "n_usable_identities":  len(identities),
        "n_dropped_identities": len(groups) - len(identities),
        "n_samples":            len(samples),
    }
    return samples, stats


def _sample_ref_with_gap(
    clip_len: int,
    driver_start_idx: int,
    n_frames: int,
    min_gap: int,
    rng: np.random.Generator,
) -> int:
    """Uniform draw over frames outside `[driver_start - min_gap,
    driver_start + n_frames + min_gap)`. Caller guarantees the clip is long
    enough for at least one valid position."""
    forbidden_lo = driver_start_idx - min_gap
    forbidden_hi = driver_start_idx + n_frames + min_gap
    valid: list[int] = []
    if forbidden_lo > 0:
        valid.extend(range(0, forbidden_lo))
    if forbidden_hi < clip_len:
        valid.extend(range(forbidden_hi, clip_len))
    if not valid:
        raise RuntimeError(
            f"No valid ref frame for clip_len={clip_len}, driver_start={driver_start_idx}, "
            f"n_frames={n_frames}, min_gap={min_gap}."
        )
    return int(valid[int(rng.integers(0, len(valid)))])
