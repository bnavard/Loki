"""Fréchet Video Distance — distribution-level visual quality.

Wraps `cdfvd` (Ge et al., CVPR 2024 — content-debiased FVD), which exposes
both the original Kinetics-400 I3D backbone and a VideoMAE backbone.
Report VideoMAE by default — converges on smaller sample sizes than I3D
(Luo et al. JEDi). I3D is available behind an opt-in for compatibility
with older literature.

FVD compares **distributions**, so it requires aggregated statistics
across a sample. Calling it on a single video is meaningless.

Resolution: both backbones consume 224×224. We do **not** pre-resize the
512×512 panel.mp4 files on disk — `cdfvd.load_videos(resolution=224)`
resizes inside its loader. Single source of truth for the FVD-side
resolution; no extra copy on disk.

Sample size caveat: the talking-head benchmarks in this repo are 125
identities (TalkVid) / 212 (HDTF). I3D FVD is widely held to need ≥ ~2k
clips to stabilize; below that the number is noisy. We report it anyway
and tag a `low_sample` flag on the result so downstream comparisons can
weight accordingly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal


class FVD:
    """One backbone per instance. Construct per evaluation; the cdfvd object
    holds real-side stats internally and accumulates them across `compute`
    calls if you reuse it (we don't — one-shot per protocol).
    """

    def __init__(
        self,
        model:           Literal["i3d", "videomae"] = "videomae",
        resolution:      int   = 224,    # I3D and VideoMAE both consume 224×224
        sequence_length: int   = 16,     # standard FVD clip length
        device:          str   = "cuda",
    ) -> None:
        from cdfvd import fvd as cdfvd_module
        self.model           = model
        self.resolution      = resolution
        self.sequence_length = sequence_length
        self.evaluator       = cdfvd_module.cdfvd(
            model, n_real="full", n_fake="full",
            ckpt_path=None, seed=0, compute_feats=False,
            device=device,
        )

    def compute(self, pred_dir: Path, ref_dir: Path) -> float:
        """
        Args:
            pred_dir: folder containing one prediction `.mp4` per sample.
            ref_dir:  folder containing one ground-truth `.mp4` per sample.
        Returns:
            FVD scalar. Lower is better.
        """
        real_loader = self.evaluator.load_videos(
            str(ref_dir), data_type="video_folder",
            resolution=self.resolution, sequence_length=self.sequence_length,
        )
        fake_loader = self.evaluator.load_videos(
            str(pred_dir), data_type="video_folder",
            resolution=self.resolution, sequence_length=self.sequence_length,
        )
        self.evaluator.compute_real_stats(real_loader)
        self.evaluator.compute_fake_stats(fake_loader)
        return float(self.evaluator.compute_fvd_from_stats())
