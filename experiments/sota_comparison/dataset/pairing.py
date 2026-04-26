"""
Protocol helpers for building deterministic (ref, driver) pair lists from
`BenchmarkClip` lists. Baseline- and modality-agnostic: each baseline's
adapter decides what to extract from `driver_clip` (audio only for
audio-driven models, full video for motion-transfer models, both for
multi-modal). The pair list stays the same across baselines on the same
(protocol, seed, dataset manifest).

Two protocols:

  * `same_identity_reconstruction` — ref_clip == driver_clip. Self-
    reconstruction: frame 0 + own audio for SadTalker; ref window + own
    motion for motion-transfer baselines. Matches SadTalker's HDTF-346
    paper protocol when clip_duration_s = 8.

  * `cross_identity` — derangement over identities: ref_identity ≠
    driver_identity, each identity appears at most once as ref and once
    as driver. For SadTalker this reads as "speaker A's face driven by
    speaker B's audio" — the clean test of voice-to-face transfer. For
    motion-transfer baselines it's "A's identity doing B's motion."

All randomness funnels through one `np.random.default_rng(seed)` so a given
(protocol, seed, dataset manifest) tuple reproduces the same list.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .base import BenchmarkClip


@dataclass(frozen=True)
class EvalSample:
    """One inference job. Stable, JSON-serialisable.

    `sample_id` is built from the clips' curated-manifest UIDs so the same
    identity (and the same pair, under cross-identity) maps to the same
    on-disk folder name across every baseline:
        same_identity_reconstruction → "id_0457"
        cross_identity               → "id_0457_id_0009"
    """
    sample_id:       str
    ref_clip:        BenchmarkClip
    driver_clip:     BenchmarkClip
    clip_duration_s: float   # how many seconds of the driver to use


def _require_uid(c: BenchmarkClip) -> str:
    if not c.uid:
        raise ValueError(
            f"BenchmarkClip {c.clip_id!r} has no uid — pairing requires clips "
            f"loaded from the curated manifest. Build it first via "
            f"`experiments/sota_comparison/dataset/build_manifest.py`."
        )
    return c.uid


def same_identity_reconstruction(
    clips:           list[BenchmarkClip],
    n_samples:       int,
    clip_duration_s: float,
    seed:            int = 42,
) -> list[EvalSample]:
    """Draw `n_samples` clips uniformly without replacement; ref == driver.

    Filters to clips whose duration is at least `clip_duration_s`. If fewer
    than `n_samples` are long enough, returns what's available rather than
    raising — lets the caller decide whether short manifests are a problem.
    """
    eligible = [c for c in clips if c.duration_s >= clip_duration_s]
    if not eligible:
        raise ValueError(
            f"No clips with duration ≥ {clip_duration_s}s found in manifest."
        )

    rng = np.random.default_rng(seed)
    take = min(n_samples, len(eligible))
    picks = rng.choice(len(eligible), size=take, replace=False)

    out: list[EvalSample] = []
    for pick_idx in picks:
        picked_clip = eligible[int(pick_idx)]
        out.append(EvalSample(
            sample_id       = _require_uid(picked_clip),    # e.g. "id_0457"
            ref_clip        = picked_clip,
            driver_clip     = picked_clip,
            clip_duration_s = clip_duration_s,
        ))
    return out


def cross_identity(
    clips:           list[BenchmarkClip],
    n_samples:       int,
    clip_duration_s: float,
    seed:            int = 42,
) -> list[EvalSample]:
    """Pair ref and driver from different identities.

    Procedure:
      1. Group clips by `identity_id`; drop identities whose clips are all
         shorter than `clip_duration_s`.
      2. Rejection-sample a permutation of identity indices until it is a
         derangement (no identity maps to itself). Density is ~1/e for
         k ≥ 2 identities, so a handful of retries suffices in practice.
      3. For each paired (ref_id, driver_id) slot, draw one ref_clip from
         ref_id's eligible clips and one driver_clip from driver_id's.
      4. Emit up to `n_samples` samples (capped by the number of identities).
    """
    clips_by_identity: dict[str, list[BenchmarkClip]] = {}
    for clip in clips:
        if clip.duration_s >= clip_duration_s:
            clips_by_identity.setdefault(clip.identity_id, []).append(clip)

    identities = sorted(clips_by_identity.keys())
    if len(identities) < 2:
        raise ValueError(
            f"Cross-identity needs ≥ 2 identities with clips ≥ "
            f"{clip_duration_s}s; got {len(identities)}."
        )

    rng = np.random.default_rng(seed)

    for _ in range(100):
        perm = list(rng.permutation(len(identities)))
        if all(perm[i] != i for i in range(len(identities))):
            break
    else:
        raise RuntimeError(
            "Failed to draw a derangement in 100 tries — identity pool may "
            "be degenerate (only one identity has enough eligible clips?)."
        )

    samples: list[EvalSample] = []
    for i in range(min(n_samples, len(identities))):
        ref_identity    = identities[i]
        driver_identity = identities[perm[i]]
        ref_clip    = clips_by_identity[ref_identity]   [
            int(rng.integers(0, len(clips_by_identity[ref_identity])))
        ]
        driver_clip = clips_by_identity[driver_identity][
            int(rng.integers(0, len(clips_by_identity[driver_identity])))
        ]
        samples.append(EvalSample(
            # e.g. "id_0457_id_0009" — ref's uid followed by driver's uid.
            sample_id       = f"{_require_uid(ref_clip)}_{_require_uid(driver_clip)}",
            ref_clip        = ref_clip,
            driver_clip     = driver_clip,
            clip_duration_s = clip_duration_s,
        ))
    return samples


Protocol = Literal["same_identity_reconstruction", "cross_identity"]

_DISPATCH = {
    "same_identity_reconstruction": same_identity_reconstruction,
    "cross_identity":                cross_identity,
}


def build_samples(
    protocol:        Protocol,
    clips:           list[BenchmarkClip],
    n_samples:       int,
    clip_duration_s: float,
    seed:            int = 42,
) -> list[EvalSample]:
    """Dispatch to the named protocol. Used by baseline runners so their
    CLI stays orthogonal: they take `--protocol` and pass it through."""
    if protocol not in _DISPATCH:
        raise ValueError(
            f"Unknown protocol: {protocol!r}. Pick from {list(_DISPATCH)}."
        )
    return _DISPATCH[protocol](clips, n_samples, clip_duration_s, seed)
