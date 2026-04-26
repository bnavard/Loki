"""Learned Perceptual Image Patch Similarity — Zhang et al. 2018.

`net='alex'` is the default in the original repo and what nearly every
talking-head paper reports. `net='vgg'` is meant for backprop / perceptual
loss, not reporting.

LPIPS expects inputs in `[-1, 1]`, not `[0, 1]` — converted at the call site.
"""
from __future__ import annotations

import torch
import lpips


class LPIPSMetric:
    """Stateful wrapper. Loads AlexNet/VGG once at construction and pins it
    on `device`; pass `(B, T, 3, H, W)` videos in `[0, 1]` to `__call__`.

    The inner LPIPS forward batches `B*T` frames at a time; for long videos
    or large batches that can OOM. Set `chunk_size` to bound the effective
    inner batch.
    """

    def __init__(
        self,
        net:        str = "alex",
        device:     str = "cuda",
        chunk_size: int = 64,
    ) -> None:
        self.device     = device
        self.chunk_size = chunk_size
        self.model      = lpips.LPIPS(net=net).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred, ref: `(B, T, 3, H, W)` in `[0, 1]`.
        Returns:
            `(B,)` mean LPIPS over frames per video.
        """
        if pred.shape != ref.shape:
            raise ValueError(f"shape mismatch: pred {tuple(pred.shape)} vs ref {tuple(ref.shape)}")
        B, T = pred.shape[:2]

        # Flatten time into batch, convert to [-1, 1].
        pred_flat = (pred.reshape(B * T, *pred.shape[2:]).to(self.device) * 2 - 1)
        ref_flat  = (ref.reshape(B * T,  *ref.shape[2:]).to(self.device) * 2 - 1)

        out: list[torch.Tensor] = []
        for i in range(0, pred_flat.shape[0], self.chunk_size):
            d = self.model(
                pred_flat[i : i + self.chunk_size],
                ref_flat[i : i + self.chunk_size],
            )
            out.append(d.flatten())
        d = torch.cat(out, dim=0).view(B, T)
        return d.mean(dim=1)
