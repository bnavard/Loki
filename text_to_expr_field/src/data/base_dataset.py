"""
Base dataset class for text-to-expression-field training.

Handles manifest parsing, sample filtering, and text embedding loading.
Subclasses implement _load_latent() to provide the VAE latent tensor.
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class BaseExprFieldDataset(Dataset):
    """
    Base dataset: manifest filtering + text embedding loading.

    Each clip is one sample. Clips shorter than min_frames are skipped.
    Text embeddings are loaded from a precomputed cache directory, or
    raw captions are returned as fallback.

    Subclasses must implement:
        _load_latent(clip_id: str) -> torch.Tensor
    """

    def __init__(
        self,
        manifest_path: str,
        min_frames: int,
        prompt_latent_cache_dir: str = None,
        flame_root: str = "data/flowface",
    ):
        with open(manifest_path) as f:
            manifest = json.load(f)

        self.flame_root = Path(flame_root)
        self.prompt_latent_cache_dir = (
            Path(prompt_latent_cache_dir) if prompt_latent_cache_dir else None
        )

        self.samples = [
            entry for entry in manifest
            if entry["num_frames"] >= min_frames
        ]

    def __len__(self):
        return len(self.samples)

    def _load_text_embed(self, entry):
        """Load cached text embedding, or fall back to raw caption string."""
        clip_id = entry["clip_id"]
        result = {}

        if self.prompt_latent_cache_dir:
            cache_path = self.prompt_latent_cache_dir / f"{clip_id}.pt"
            if cache_path.exists():
                text_data = torch.load(str(cache_path), map_location="cpu",
                                       weights_only=True)
                result["text_embed"] = text_data["text_embed"]
                result["caption"] = text_data.get("caption", "")
                return result

        with open(entry["caption_file"]) as f:
            caption_data = json.load(f)
        result["caption"] = caption_data["caption"]
        return result

    def _load_latent(self, clip_id: str) -> torch.Tensor:
        """Return the VAE latent for a clip. Implemented by subclasses."""
        raise NotImplementedError

    def __getitem__(self, idx):
        entry = self.samples[idx]
        clip_id = entry["clip_id"]

        result = {"clip_id": clip_id}
        result.update(self._load_text_embed(entry))
        result["latent"] = self._load_latent(clip_id)

        return result
