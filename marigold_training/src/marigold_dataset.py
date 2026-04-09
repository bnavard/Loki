"""
Marigold-style dataset: returns (natural_video, deform_video, text) triplets.

Each sample produces a temporally aligned pair:
  - natural_video: RGB frames from the talking-head clip, cropped to the face
    bounding box, resized to target resolution, normalized to [-1, 1]
  - deform_video: 3ch deformation map from FLAME rasterization, same crop/size
  - text_embed: cached UMT5 text embedding of the prosody caption

Both videos are from the same clip, same frames, so they are spatially and
temporally aligned — the deformation map is a structured re-representation
of the facial dynamics in the natural video.

The VAE encoding happens in the training loop (not here), because both
videos need to be encoded through the same frozen VAE on GPU.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from marigold_training.src.reshape import to_pseudo_video


class MarigoldDataset(Dataset):
    """
    Dataset returning (natural_video, deform_video, text_embed) triplets.

    Args:
        manifest_path:           Path to data/derived/manifest.json
        flame_root:              Root directory for FLAME data (data/flowface/)
        video_root:              Root directory for source videos (data/talkvid/talkvid/)
        target_frames:           Frames per clip (must satisfy 4k+1 for VAE)
        resolution:              Spatial resolution for both videos
        prompt_latent_cache_dir: Directory with precomputed UMT5 text embeddings
    """

    def __init__(
        self,
        manifest_path: str,
        flame_root: str = "data/flowface",
        video_root: str = "data/talkvid/talkvid",
        target_frames: int = 81,
        resolution: int = 512,
        prompt_latent_cache_dir: str = "data/derived/prompt_latent_cache",
    ):
        with open(manifest_path) as f:
            manifest = json.load(f)

        self.flame_root = Path(flame_root)
        self.video_root = Path(video_root)
        self.target_frames = target_frames
        self.resolution = resolution
        self.prompt_latent_cache_dir = (
            Path(prompt_latent_cache_dir) if prompt_latent_cache_dir else None
        )

        # Filter to clips with enough frames
        self.samples = [
            entry for entry in manifest
            if entry["num_frames"] >= target_frames
        ]

        # Lazy-init FLAME + conditioning (avoids GPU allocation at import)
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

    def _load_natural_video(self, clip_id, crop_boxes):
        """
        Load natural video frames, crop to face bounding box, resize, normalize.

        Returns:
            [3, T, H, W] float32 tensor in [-1, 1]
        """
        from talkinghead_sd21_unet_cap4d_based.data.utils import crop_image, rescale_image

        frames_dir = self.flame_root / clip_id / "images" / "cam0"
        H = self.resolution
        frames = []

        for t in range(self.target_frames):
            img_path = frames_dir / f"{t:05d}.jpg"
            img = cv2.imread(str(img_path))
            if img is None:
                img = np.zeros((H, H, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = crop_image(img, crop_boxes[t], bg_value=0)
            img = rescale_image(img, H)
            img = img.astype(np.float32) / 127.5 - 1.0
            frames.append(torch.from_numpy(img).permute(2, 0, 1))

        video = torch.stack(frames, dim=1)  # [3, T, H, W]
        return video

    def _compute_deform_and_crops(self, clip_id):
        """
        Compute deformation map and per-frame crop boxes from fit.npz.

        Returns:
            deform: [T, 3, H, W] float32 tensor
            crop_boxes: list of T numpy arrays [x0, y0, x1, y1]
        """
        from talkinghead_sd21_unet_cap4d_based.flame.flame import compute_flame
        from talkinghead_sd21_unet_cap4d_based.data.utils import get_bbox_from_verts, verts_to_pytorch3d

        conditioning, flame_skinner, head_vertex_ids = self._get_conditioning()

        fit = dict(np.load(str(self.flame_root / clip_id / "fit.npz")))
        H = self.resolution

        deform_frames = []
        crop_boxes = []

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
            crop_boxes.append(crop_box)

            verts_2d_p3d = verts_to_pytorch3d(verts_2d.copy(), np.array(crop_box))

            dummy_ref_mask = torch.zeros(1, 1, 1, H, H, device="cuda")
            batch = {
                "verts_2d": torch.tensor(verts_2d_p3d).unsqueeze(0).unsqueeze(0).cuda(),
                "offsets_3d": torch.tensor(offsets_3d).unsqueeze(0).unsqueeze(0).cuda(),
                "reference_mask": dummy_ref_mask,
            }

            with torch.no_grad():
                out = conditioning(batch, unconditional=False)

            frame = out["pos_enc"][0, 0, :, :, 42:45].permute(2, 0, 1)
            deform_frames.append(frame.cpu())

        return torch.stack(deform_frames, dim=0), crop_boxes

    def __getitem__(self, idx):
        entry = self.samples[idx]
        clip_id = entry["clip_id"]

        # 1. Compute deformation map + crop boxes (FLAME + PyTorch3D)
        deform_video, crop_boxes = self._compute_deform_and_crops(clip_id)

        # 2. Load natural video with the same crop boxes
        natural_video = self._load_natural_video(clip_id, crop_boxes)

        # 3. Pad deform to 4k+1 for VAE compatibility
        deform_pseudo = to_pseudo_video(deform_video)
        deform_video_5d = deform_pseudo.permute(1, 0, 2, 3)  # [3, T_padded, H, W]

        # 4. Pad natural video to match
        T_target = deform_pseudo.shape[0]
        T_nat = natural_video.shape[1]
        if T_nat < T_target:
            pad = natural_video[:, -1:].repeat(1, T_target - T_nat, 1, 1)
            natural_video = torch.cat([natural_video, pad], dim=1)
        elif T_nat > T_target:
            natural_video = natural_video[:, :T_target]

        result = {
            "clip_id": clip_id,
            "natural_video": natural_video,    # [3, T, H, W] float32, [-1, 1]
            "target_video": deform_video_5d,   # [3, T, H, W] float32
        }

        # 5. Text embedding
        if self.prompt_latent_cache_dir:
            cache_path = self.prompt_latent_cache_dir / f"{clip_id}.pt"
            if cache_path.exists():
                text_data = torch.load(str(cache_path), map_location="cpu",
                                       weights_only=True)
                result["text_embed"] = text_data["text_embed"]

        return result
