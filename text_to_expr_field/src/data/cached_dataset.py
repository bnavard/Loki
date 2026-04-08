"""
Cached latent dataset — loads precomputed VAE latents from disk.

Random temporal slicing provides data augmentation: each __getitem__
returns a different temporal window from the full cached latent.
Works for any channel count (45ch expr field, 3ch deform, etc.)
since it just loads and slices a tensor.
"""

import random
from pathlib import Path

import torch

from text_to_expr_field.src.data.base_dataset import BaseExprFieldDataset


class CachedLatentDataset(BaseExprFieldDataset):
    """
    Loads precomputed VAE latents + text embeddings from disk.

    Args:
        manifest_path:           Path to data/derived/manifest.json
        vae_latent_cache_dir:    Directory with precomputed VAE latent .pt files
        target_latent_T:         Temporal window size to slice from cached latents.
                                 Set to None to use the full latent (no slicing).
        prompt_latent_cache_dir: Directory with precomputed UMT5 text embeddings
        min_frames:              Skip clips shorter than this
        flame_root:              Root directory for FLAME data
    """

    def __init__(
        self,
        manifest_path: str,
        vae_latent_cache_dir: str,
        target_latent_T: int = None,
        prompt_latent_cache_dir: str = None,
        min_frames: int = 80,
        flame_root: str = "data/flowface",
    ):
        super().__init__(
            manifest_path=manifest_path,
            min_frames=min_frames,
            prompt_latent_cache_dir=prompt_latent_cache_dir,
            flame_root=flame_root,
        )

        self.vae_latent_cache_dir = Path(vae_latent_cache_dir)
        self.target_latent_T = target_latent_T

        # Only keep clips that have a cached latent file
        self.samples = [
            entry for entry in self.samples
            if (self.vae_latent_cache_dir / f"{entry['clip_id']}.pt").exists()
        ]

    def _load_latent(self, clip_id: str) -> torch.Tensor:
        cache_path = self.vae_latent_cache_dir / f"{clip_id}.pt"
        cached = torch.load(str(cache_path), map_location="cpu", weights_only=True)
        full_latent = cached["latent"] if isinstance(cached, dict) else cached

        # full_latent: [C, T, H, W]
        if self.target_latent_T is None or self.target_latent_T >= full_latent.shape[1]:
            return full_latent

        max_start = full_latent.shape[1] - self.target_latent_T
        start = random.randint(0, max_start)
        return full_latent[:, start:start + self.target_latent_T, :, :]
