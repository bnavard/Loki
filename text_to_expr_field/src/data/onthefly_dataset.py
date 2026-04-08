"""
On-the-fly dataset — computes expression/deform fields from fit.npz
and VAE-encodes them live during training.

Requires PyTorch3D (for rasterization) and a frozen VAE on GPU.
Must use num_workers=0 since both PyTorch3D and VAE use CUDA.
"""

from pathlib import Path

import numpy as np
import torch

from text_to_expr_field.src.data.base_dataset import BaseExprFieldDataset
from text_to_expr_field.src.utils.reshape import to_pseudo_video


class OnTheFlyDataset(BaseExprFieldDataset):
    """
    Computes expression fields from FLAME fit.npz and VAE-encodes on the fly.

    Args:
        manifest_path:           Path to data/derived/manifest.json
        vae:                     Frozen VAE model (on GPU, eval mode)
        mode:                    "expr_field" (45ch) or "deform" (3ch deformation only)
        target_frames:           Frames to use per clip (truncated from start)
        resolution:              Expression field spatial resolution
        prompt_latent_cache_dir: Directory with precomputed UMT5 text embeddings
        min_frames:              Skip clips shorter than this
        flame_root:              Root directory for FLAME data
    """

    def __init__(
        self,
        manifest_path: str,
        vae,
        mode: str = "deform",
        target_frames: int = 81,
        resolution: int = 512,
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

        assert mode in ("expr_field", "deform")
        self.mode = mode
        self.vae = vae
        self.target_frames = target_frames
        self.resolution = resolution

        # Lazy-init to avoid GPU allocation at import time
        self._conditioning = None
        self._flame_skinner = None
        self._head_vertex_ids = None

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
            ).eval().to(self.vae.device)

        if self._flame_skinner is None:
            from talkinghead_sd21_unet_cap4d_based.flame.flame import CAP4DFlameSkinner
            self._flame_skinner = CAP4DFlameSkinner(
                add_mouth=True, n_shape_params=150, n_expr_params=65,
            )
            self._head_vertex_ids = np.genfromtxt(
                "data/assets/flame/head_vertices.txt"
            ).astype(int)

        return self._conditioning, self._flame_skinner, self._head_vertex_ids

    def _compute_field(self, clip_id: str) -> torch.Tensor:
        """
        Compute expression field from fit.npz via FLAME + PyTorch3D.

        Returns:
            [target_frames, C, H, W] float32 tensor (CPU).
            C=45 for expr_field mode, C=3 for deform mode.
        """
        from talkinghead_sd21_unet_cap4d_based.flame.flame import compute_flame
        from talkinghead_sd21_unet_cap4d_based.data.utils import get_bbox_from_verts, verts_to_pytorch3d

        conditioning, flame_skinner, head_vertex_ids = self._get_conditioning()
        device = self.vae.device

        fit = dict(np.load(str(self.flame_root / clip_id / "fit.npz")))
        H = self.resolution

        frames = []
        for t in range(self.target_frames):
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

            dummy_ref_mask = torch.zeros(1, 1, 1, H, H, device=device)
            batch = {
                "verts_2d": torch.tensor(verts_2d_p3d).unsqueeze(0).unsqueeze(0).to(device),
                "offsets_3d": torch.tensor(offsets_3d).unsqueeze(0).unsqueeze(0).to(device),
                "reference_mask": dummy_ref_mask,
            }

            with torch.no_grad():
                out = conditioning(batch, unconditional=False)

            if self.mode == "deform":
                frame = out["pos_enc"][0, 0, :, :, 42:45].permute(2, 0, 1)
            else:
                frame = out["pos_enc"][0, 0, :, :, :45].permute(2, 0, 1)
            frames.append(frame.cpu())

        return torch.stack(frames, dim=0)

    def _encode_with_vae(self, pseudo_video: torch.Tensor) -> torch.Tensor:
        """Encode pseudo-video through the frozen VAE."""
        with torch.no_grad():
            vae_input = pseudo_video.permute(1, 0, 2, 3).unsqueeze(0)
            vae_input = vae_input.to(device=self.vae.device, dtype=self.vae.dtype)
            posterior = self.vae.encode(vae_input).latent_dist
            return posterior.mode().squeeze(0).cpu()

    def _load_latent(self, clip_id: str) -> torch.Tensor:
        field = self._compute_field(clip_id)
        pseudo_video = to_pseudo_video(field)
        return self._encode_with_vae(pseudo_video)
