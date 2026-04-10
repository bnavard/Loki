"""
Single-frame pair dataset for Stage 1 (spatial) Marigold training.

Each __getitem__ randomly samples one frame from a clip, computes
the deformation map for that frame via FLAME + PyTorch3D, and loads
the corresponding natural video frame with the same face crop.

This gives ~900k training pairs from ~7150 clips x ~127 frames/clip.
Random frame sampling across epochs ensures all frames get covered.

Both outputs are [3, H, W] image tensors. The SD3.5 VAE expects [B, 3, H, W].
Used with SD3Transformer2DModel for image-level Marigold training.
"""

import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class FramePairDataset(Dataset):
    """
    Returns (natural_frame, deform_frame, text_embed) triplets at T=1.

    Args:
        manifest_path:  Path to data/derived/manifest.json
        flame_root:     Root directory for FLAME data
        resolution:     Spatial resolution
        min_frames:     Skip clips shorter than this
    """

    def __init__(
        self,
        manifest_path: str,
        flame_root: str = "data/flowface",
        resolution: int = 512,
        min_frames: int = 10,
    ):
        with open(manifest_path) as f:
            manifest = json.load(f)

        self.flame_root = Path(flame_root)
        self.resolution = resolution

        self.samples = [
            entry for entry in manifest
            if entry["num_frames"] >= min_frames
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
            ).eval().cuda()

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

    def _compute_deform_frame(self, clip_id, frame_idx):
        """
        Compute 3ch deformation map + crop box for one frame.

        Returns:
            deform: [3, H, W] float32 tensor
            crop_box: numpy [x0, y0, x1, y1]
        """
        from talkinghead_sd21_unet_cap4d_based.flame.flame import compute_flame
        from talkinghead_sd21_unet_cap4d_based.data.utils import get_bbox_from_verts, verts_to_pytorch3d

        conditioning, flame_skinner, head_vertex_ids = self._get_conditioning()

        fit = dict(np.load(str(self.flame_root / clip_id / "fit.npz")))
        H = self.resolution

        flame_item = {
            "shape": fit["shape"],
            "expr": fit["expr"][[frame_idx]],
            "rot": fit["rot"][[frame_idx]],
            "tra": fit["tra"][[frame_idx]],
            "eye_rot": fit["eye_rot"][[frame_idx]],
            "fx": fit["fx"][[0]], "fy": fit["fy"][[0]],
            "cx": fit["cx"][[0]], "cy": fit["cy"][[0]],
            "extr": fit["extr"][[0]],
        }
        if "jaw_rot" in fit:
            flame_item["jaw_rot"] = fit["jaw_rot"][[frame_idx]]

        flame_out = compute_flame(flame_skinner, flame_item)
        verts_2d = flame_out["verts_2d"][0, 0]
        offsets_3d = flame_out["offsets_3d"][0]

        crop_box = get_bbox_from_verts(verts_2d.copy(), head_vertex_ids)
        verts_2d_p3d = verts_to_pytorch3d(verts_2d.copy(), np.array(crop_box))

        dummy_ref = torch.zeros(1, 1, 1, H, H, device="cuda")
        batch = {
            "verts_2d": torch.tensor(verts_2d_p3d).unsqueeze(0).unsqueeze(0).cuda(),
            "offsets_3d": torch.tensor(offsets_3d).unsqueeze(0).unsqueeze(0).cuda(),
            "reference_mask": dummy_ref,
        }

        with torch.no_grad():
            out = conditioning(batch, unconditional=False)

        deform = out["pos_enc"][0, 0, :, :, 42:45].permute(2, 0, 1).cpu()
        return deform, crop_box

    def _load_natural_frame(self, clip_id, frame_idx, crop_box):
        """Load one natural video frame, crop to face, normalize to [-1, 1]."""
        from talkinghead_sd21_unet_cap4d_based.data.utils import crop_image, rescale_image

        frames_dir = self.flame_root / clip_id / "images" / "cam0"
        H = self.resolution

        img_path = frames_dir / f"{frame_idx:05d}.jpg"
        img = cv2.imread(str(img_path))
        if img is None:
            return torch.zeros(3, H, H)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = crop_image(img, crop_box, bg_value=0)
        img = rescale_image(img, H)
        img = img.astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(img).permute(2, 0, 1)

    def __getitem__(self, idx):
        entry = self.samples[idx]
        clip_id = entry["clip_id"]
        num_frames = entry["num_frames"]

        # Random frame each call — free data augmentation across epochs
        frame_idx = random.randint(0, num_frames - 1)

        deform_frame, crop_box = self._compute_deform_frame(clip_id, frame_idx)
        natural_frame = self._load_natural_frame(clip_id, frame_idx, crop_box)

        return {
            "clip_id": clip_id,
            "natural_frame": natural_frame,    # [3, H, W]
            "target_frame": deform_frame,      # [3, H, W]
        }
