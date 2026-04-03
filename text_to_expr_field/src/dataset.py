"""
Dataset for text-to-expression-field training.

Each clip is one training sample. Clips with more than target_frames (default
80) are truncated to the first target_frames frames. Clips shorter than
target_frames are skipped. Fully deterministic — always starts from frame 0.

The expression field is computed on the fly from fit.npz, or loaded from a
precomputed VAE latent cache (data/derived/vae_latent_cache/) if available.

The 45-channel expression field is split into 15 groups of 3 channels, stacked
temporally into a pseudo-video, and encoded through the Wan2.2 VAE.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class ExprFieldDataset(Dataset):
    """
    Dataset for training text-to-expression-field generation.

    Each clip is one sample. Clips longer than target_frames are truncated
    to the first target_frames frames. Clips shorter are skipped.

    Two loading modes:
      1. Cached latents: if precomputed VAE latent and prompt embedding exist
         in their respective cache dirs, load them directly (fast).
      2. On-the-fly: compute expression field from fit.npz → reshape →
         VAE encode (slow, requires GPU — use cache_vae_latents.py first).

    Args:
        manifest_path:           Path to data/derived/manifest.json
        flame_root:              Root directory for FLAME data (data/flowface/)
        target_frames:           Fixed number of frames per sample (default 80).
        resolution:              Expression field spatial resolution (default 512)
        vae:                     Optional frozen VAE for on-the-fly encoding.
        device:                  Device for on-the-fly computation.
        vae_latent_cache_dir:    Directory with precomputed VAE latents.
        prompt_latent_cache_dir: Directory with precomputed UMT5 text embeddings.
        cached_only:             Only include clips with cached VAE latents.
    """

    def __init__(self, manifest_path: str, flame_root: str = "data/flowface",
                 target_frames: int = 80, resolution: int = 512,
                 vae=None, device: str = "cuda",
                 vae_latent_cache_dir: str = "data/derived/vae_latent_cache",
                 prompt_latent_cache_dir: str = "data/derived/prompt_latent_cache",
                 cached_only: bool = False):
        with open(manifest_path) as f:
            manifest = json.load(f)

        self.flame_root = Path(flame_root)
        self.target_frames = target_frames
        self.resolution = resolution
        self.vae = vae
        self.device = device
        self.vae_latent_cache_dir = Path(vae_latent_cache_dir) if vae_latent_cache_dir else None
        self.prompt_latent_cache_dir = Path(prompt_latent_cache_dir) if prompt_latent_cache_dir else None

        # Lazy-init on first use to avoid GPU allocation at import time
        self._conditioning = None
        self._flame_skinner = None
        self._head_vertex_ids = None

        # One clip = one sample. Skip clips shorter than target_frames.
        self.samples = [
            entry for entry in manifest
            if entry["num_frames"] >= target_frames
        ]

        # If cached_only=True, only include clips whose VAE latent has already
        # been precomputed and saved to vae_latent_cache_dir.
        if cached_only and self.vae_latent_cache_dir:
            self.samples = [
                entry for entry in self.samples
                if (self.vae_latent_cache_dir / f"{entry['clip_id']}.pt").exists()
            ]

    def _get_conditioning(self):
        """Lazy-init THConditioning and FLAME skinner on first use."""
        if self._conditioning is None:
            from talkinghead_sd21_unet_cap4d_based.conditioning.th_conditioning import THConditioning
            self._conditioning = THConditioning(
                image_size=self.resolution,
                positional_channels=42,
                positional_multiplier=1.0,
                super_resolution=1,
                use_ray_directions=False,
                use_expr_deformation=True,
                use_crop_mask=False,
            ).eval().to(self.device)

        if self._flame_skinner is None:
            from talkinghead_sd21_unet_cap4d_based.flame.flame import CAP4DFlameSkinner
            self._flame_skinner = CAP4DFlameSkinner(
                add_mouth=True, n_shape_params=150, n_expr_params=65,
            )
            self._head_vertex_ids = np.genfromtxt(
                "data/assets/flame/head_vertices.txt"
            ).astype(int)

        return self._conditioning, self._flame_skinner, self._head_vertex_ids

    def __len__(self):
        return len(self.samples)

    def _compute_expr_field(self, clip_id):
        """
        Compute 45-channel expression field from fit.npz.

        Always takes the first target_frames frames for deterministic behavior.
        Clips longer than target_frames are truncated.

        Returns: [target_frames, 45, H, W] float32 tensor (CPU).
        """
        from talkinghead_sd21_unet_cap4d_based.flame.flame import compute_flame
        from talkinghead_sd21_unet_cap4d_based.data.utils import get_bbox_from_verts, verts_to_pytorch3d

        conditioning, flame_skinner, head_vertex_ids = self._get_conditioning()

        fit = dict(np.load(str(self.flame_root / clip_id / "fit.npz")))
        H = self.resolution

        # Always start from frame 0 for deterministic behavior.
        frame_indices = range(0, self.target_frames)

        frames = []
        for t in frame_indices:
            flame_item = {
                "shape": fit["shape"],
                "expr": fit["expr"][[t]],
                "rot": fit["rot"][[t]],
                "tra": fit["tra"][[t]],
                "eye_rot": fit["eye_rot"][[t]],
                "fx": fit["fx"][[0]], "fy": fit["fy"][[0]],
                "cx": fit["cx"][[0]], "cy": fit["cy"][[0]],
                "extr": fit["extr"][[0]],
            }
            if "jaw_rot" in fit:
                flame_item["jaw_rot"] = fit["jaw_rot"][[t]]

            flame_out = compute_flame(flame_skinner, flame_item)
            verts_2d = flame_out["verts_2d"][0, 0]
            offsets_3d = flame_out["offsets_3d"][0]

            crop_box = get_bbox_from_verts(verts_2d.copy(), head_vertex_ids)
            verts_2d_p3d = verts_to_pytorch3d(verts_2d.copy(), np.array(crop_box))

            # THConditioning requires reference_mask — pass dummy zeros,
            # discard channel 45 (ref_mask) from output.
            dummy_ref_mask = torch.zeros(1, 1, 1, H, H, device=self.device)
            batch = {
                "verts_2d": torch.tensor(verts_2d_p3d).unsqueeze(0).unsqueeze(0).to(self.device),
                "offsets_3d": torch.tensor(offsets_3d).unsqueeze(0).unsqueeze(0).to(self.device),
                "reference_mask": dummy_ref_mask,
            }

            with torch.no_grad():
                out = conditioning(batch, unconditional=False)

            pos_enc = out["pos_enc"][0, 0, :, :, :45].permute(2, 0, 1)
            frames.append(pos_enc.cpu())

        return torch.stack(frames, dim=0)

    def _reshape_for_vae(self, expr_field):
        """[T, 45, H, W] → [T_padded, 3, H, W] pseudo-video, padded to 4k+1."""
        T, C, H, W = expr_field.shape
        assert C == 45

        grouped = expr_field.reshape(T, 15, 3, H, W)
        pseudo = grouped.reshape(T * 15, 3, H, W)

        # Pad to 4k+1
        target = 1 + 4 * (pseudo.shape[0] // 4)
        if target < pseudo.shape[0]:
            target += 4
        while pseudo.shape[0] < target:
            pseudo = torch.cat([pseudo, pseudo[-1:]], dim=0)

        return pseudo

    def _encode_with_vae(self, pseudo_video):
        """Encode pseudo-video through the frozen VAE."""
        with torch.no_grad():
            vae_input = pseudo_video.permute(1, 0, 2, 3).unsqueeze(0)
            vae_input = vae_input.to(self.device, dtype=torch.float32)
            posterior = self.vae.encode(vae_input).latent_dist
            latent = posterior.mode().squeeze(0).cpu()
        return latent

    def __getitem__(self, idx):
        entry = self.samples[idx]
        clip_id = entry["clip_id"]

        result = {"clip_id": clip_id}

        # ---- Prompt embedding: load from cache or fallback to raw caption ----
        text_cached = False
        if self.prompt_latent_cache_dir:
            cache_path = self.prompt_latent_cache_dir / f"{clip_id}.pt"
            if cache_path.exists():
                text_data = torch.load(str(cache_path), map_location="cpu")
                result["text_embed"] = text_data["text_embed"]
                result["caption"] = text_data.get("caption", "")
                text_cached = True

        if not text_cached:
            with open(entry["caption_file"]) as f:
                caption_data = json.load(f)
            result["caption"] = caption_data["caption"]

        # ---- VAE latent: load from cache or compute on the fly ----
        if self.vae_latent_cache_dir:
            cache_path = self.vae_latent_cache_dir / f"{clip_id}.pt"
            if cache_path.exists():
                cached = torch.load(str(cache_path), map_location="cpu")
                result["latent"] = cached["latent"] if isinstance(cached, dict) else cached
                return result

        # On-the-fly: compute expression field → reshape → VAE encode
        expr_field = self._compute_expr_field(clip_id)
        pseudo_video = self._reshape_for_vae(expr_field)

        if self.vae is not None:
            result["latent"] = self._encode_with_vae(pseudo_video)
            return result
        else:
            result["pseudo_video"] = pseudo_video
            return result

    @staticmethod
    def reassemble_from_decoded(decoded_video, num_expr_frames):
        """
        Reassemble a decoded pseudo-video back into a 45-channel expression field.

        Args:
            decoded_video: [T_padded, 3, H, W]
            num_expr_frames: original number of expression frames

        Returns:
            [num_expr_frames, 45, H, W] expression dense field
        """
        T_needed = num_expr_frames * 15
        decoded_video = decoded_video[:T_needed]

        H, W = decoded_video.shape[2], decoded_video.shape[3]
        expr_field = decoded_video.reshape(num_expr_frames, 15, 3, H, W)
        expr_field = expr_field.reshape(num_expr_frames, 45, H, W)

        return expr_field
