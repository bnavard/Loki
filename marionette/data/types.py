"""
TypedDict contracts for the DataLoader → SpatialConditioning → UNet pipeline.

⚠️ STALE — DO NOT TRUST. These contracts predate the v3 reorg: they still
reference `ref_mask`, `z_input`, `ref_image`, `ref_verts`, and
`spatial_cond` of width 49, all of which were removed when identity moved
into the frozen `RefFeatureExtractor`. The current schema is documented
inline in:
  - `TalkingHeadDataset.__getitem__` (data/video_dataset.py)
  - `SpatialConditioning.forward` (conditioning/conditioning.py)
  - `MarionetteUNet.forward` (model/unet.py)
  - `MarionetteDiffusion.get_input` (model/diffusion.py)

This file has no runtime effect — pure documentation-as-types — but the
documentation itself is wrong and the file is a candidate for deletion in
a follow-up cleanup. Kept around for now so removing it can be a separate
focused diff.
"""
from __future__ import annotations

from typing import Optional, TypedDict

import torch


class HintDict(TypedDict):
    """Per-sample conditioning dict (before DataLoader collation adds the B dim).

    Shapes are per-sample; the collated batch prepends a leading B.
    """
    driver_verts:  torch.Tensor   # (T, V, 3)     driver's FLAME verts in target-camera NDC
    driver_deform: torch.Tensor   # (T, V, 3)     per-vertex expression deformation
    ref_mask:      torch.Tensor   # (T, 1, h, w)  1 on slot 0 (reference), 0 elsewhere
    ref_image:     torch.Tensor   # (3, H, W)     reference frame in [-1, 1]
    ref_verts:     torch.Tensor   # (V, 3)        reference's FLAME verts in NDC (warp source)


class SampleDict(TypedDict):
    """Output of `TalkingHeadDataset.__getitem__`."""
    target_video: torch.Tensor    # (T, H, W, 3)         frames to reconstruct, [-1, 1]
    audio:        torch.Tensor    # (T, window_samples)  driver audio per frame
    hint:         HintDict


class ControlDict(TypedDict, total=False):
    """Post-SpatialConditioning + post-CFG control dict consumed by the UNet.

    `audio_context` is optional (None when the audio stream is disabled).
    Values are batched (leading B dim).
    """
    spatial_cond:  torch.Tensor              # (B, T, H, W, 49) fused conditioning map
    z_input:       torch.Tensor              # (B, T, 4, h, w)  GT latents (only refs are non-zero)
    ref_mask:      torch.Tensor              # (B, T, 1, h, w)  reference-slot indicator
    audio_context: Optional[torch.Tensor]    # (B, T, S, D) wav2vec2 tokens, or None
