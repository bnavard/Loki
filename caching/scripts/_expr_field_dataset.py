"""
Minimal dataset for VAE latent caching.

Computes expression fields from fit.npz on the fly and returns pseudo-videos
ready for VAE encoding. No VAE, no text embeddings — just the raw pseudo-video.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class ExprFieldCachingDataset(Dataset):
    """
    Dataset that computes expression fields from fit.npz and returns
    pseudo-videos for VAE encoding.

    Args:
        manifest_path: Path to manifest.json
        flame_root:    Root directory for FLAME data
        target_frames: Frames per clip
        resolution:    Spatial resolution
        device:        CUDA device for PyTorch3D rasterization
    """

    def __init__(self, manifest_path, flame_root="data/flowface",
                 target_frames=80, resolution=512, device="cuda"):
        with open(manifest_path) as f:
            manifest = json.load(f)

        self.flame_root = Path(flame_root)
        self.target_frames = target_frames
        self.resolution = resolution
        self.device = device

        self.samples = [
            entry for entry in manifest
            if entry["num_frames"] >= target_frames
        ]

        self._conditioning = None
        self._flame_skinner = None
        self._head_vertex_ids = None

    def _get_conditioning(self):
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

    def __getitem__(self, idx):
        from talkinghead_sd21_unet_cap4d_based.flame.flame import compute_flame
        from talkinghead_sd21_unet_cap4d_based.data.utils import get_bbox_from_verts, verts_to_pytorch3d

        entry = self.samples[idx]
        clip_id = entry["clip_id"]
        conditioning, flame_skinner, head_vertex_ids = self._get_conditioning()

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

        expr_field = torch.stack(frames, dim=0)  # [T, 45, H, W]

        # Reshape to pseudo-video: [T*15, 3, H, W] padded to 4k+1
        T, C = expr_field.shape[0], expr_field.shape[1]
        pseudo = expr_field.reshape(T, 15, 3, H, H).reshape(T * 15, 3, H, H)

        target = 1 + 4 * (pseudo.shape[0] // 4)
        if target < pseudo.shape[0]:
            target += 4
        while pseudo.shape[0] < target:
            pseudo = torch.cat([pseudo, pseudo[-1:]], dim=0)

        return {"clip_id": clip_id, "pseudo_video": pseudo}
