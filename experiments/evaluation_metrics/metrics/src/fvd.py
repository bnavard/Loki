"""Fréchet Video Distance — distribution-level visual quality.

Wraps `cdfvd` (Ge et al., CVPR 2024 — content-debiased FVD), which exposes
both the original Kinetics-400 I3D backbone and a VideoMAE-v2 backbone.
Default to VideoMAE-v2 — converges on smaller sample sizes than I3D
(Luo et al. JEDi). I3D is available behind an opt-in for compatibility
with older literature.

FVD compares **distributions**, so it requires aggregated statistics
across a sample. Calling it on a single video is meaningless — see the
`compute(pred_dir, ref_dir)` API which folds many clips into Gaussians
on each side.

Resolution: both backbones consume 224×224. We do **not** pre-resize the
512×512 panel.mp4 files on disk — `cdfvd.load_videos(resolution=224)`
resizes inside its loader. Single source of truth for the FVD-side
resolution; no extra copy on disk.

Sample size caveat: HDTF has 212 same-identity clips. I3D FVD is widely
held to need ≥ ~2k clips to stabilize; below that the number is noisy.
VideoMAE-v2 (the default) is more sample-efficient. Either way, we
report the number and tag the result with `n_real` / `n_fake` so
downstream comparisons can weight accordingly.

VideoMAE backbone availability: upstream `cdfvd` hardcodes a download
URL for the SSv2-finetuned giant checkpoint that points at an Aliyun OSS
bucket which has been taken down. This module assumes the patch at
`patches/cdfvd_videomaev2_utils.py` has been applied (via
`setup_fvd.sh`) so the loader pulls from the HuggingFace mirror at
`OpenGVLab/VideoMAE2/mae-g/`. Without the patch, VideoMAE FVD silently
fails: `requests.get` saves the 404 HTML body as a `.pth`, then
`torch.load` dies with `invalid load key, '<'`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal


class FVD:
    """One backbone per instance. Construct per evaluation; the cdfvd
    object holds real-side stats internally and accumulates them across
    `compute_real_stats` calls if you reuse it (we don't — one-shot
    per protocol).
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

    def compute(self, pred_dir: Path, ref_dir: Path) -> dict:
        """
        Args:
            pred_dir: folder containing one prediction `.mp4` per sample.
            ref_dir:  folder containing one ground-truth `.mp4` per sample.
        Returns:
            {
              "fvd":     float,   # lower is better
              "model":   str,     # "videomae" | "i3d"
              "n_real":  int,     # GT clip count fed into Σ_real
              "n_fake":  int,     # pred clip count fed into Σ_fake
            }
        """
        real_loader = self.evaluator.load_videos(
            str(ref_dir), data_type="video_folder",
            resolution=self.resolution, sequence_length=self.sequence_length,
        )
        fake_loader = self.evaluator.load_videos(
            str(pred_dir), data_type="video_folder",
            resolution=self.resolution, sequence_length=self.sequence_length,
        )
        n_real = len(real_loader.dataset) if hasattr(real_loader, "dataset") else -1
        n_fake = len(fake_loader.dataset) if hasattr(fake_loader, "dataset") else -1
        self.evaluator.compute_real_stats(real_loader)
        self.evaluator.compute_fake_stats(fake_loader)
        return {
            "fvd":    float(self.evaluator.compute_fvd_from_stats()),
            "model":  self.model,
            "n_real": int(n_real),
            "n_fake": int(n_fake),
        }