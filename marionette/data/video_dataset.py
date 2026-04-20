"""
Video dataset for talking-head generation training.

Each clip is split into non-overlapping n_frames-sized windows (deterministic).
Frame 0 of each window is the reference frame; frames 1:n_frames are the
generation targets.

Training is strictly same-identity: target, driver (FLAME signal), reference
image — all sourced from the same clip and same window. Cross-identity
retargeting happens at inference time only, in `generate.py`.

NDC = Normalized Device Coordinates — pytorch3d's convention where visible
on-screen content maps to [-1, 1] per axis, +x=left, +y=up. `grid_sample` uses
the other sign convention (+x=right, +y=down); the flip happens inside the
warp step, not here.

Per sample returned:
    target_video      : (T, H, W, 3)   frames to reconstruct, [-1, 1]
    audio             : (T, W)         driver audio window per frame
    hint:
        driver_verts   : (T, V, 3)     driver's FLAME verts in target-camera NDC
        driver_deform  : (T, V, 3)     per-vertex expression deformation
        ref_mask       : (T, 1, h, w)  1 on slot 0 (reference), 0 elsewhere
        ref_image      : (3, H, W)     reference frame in [-1, 1] (= slot 0)
        ref_verts      : (V, 3)        reference's FLAME verts in NDC (warp source)
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from marionette.flame.flame import CAP4DFlameSkinner, compute_flame
from marionette.utils import (
    load_frame, crop_image, rescale_image,
    get_bbox_from_verts, verts_to_pytorch3d,
    SAMPLE_RATE, load_audio_mono, frame_window,
)


HEAD_VERT_PATH = "data/assets/flame/head_vertices.txt"


class TalkingHeadDataset(Dataset):
    """
    Args:
        clip_list_path       : path to a JSON file of clip IDs (produced by
                               `scripts/manifest/partition_dataset.py`).
        video_root           : root directory for videos ({video_root}/{id}.mp4).
        audio_root           : root directory for audio  ({audio_root}/{id}.wav).
        flame_root           : root directory for FLAME  ({flame_root}/{id}/fit.npz).
        n_frames             : total frames per sample (1 ref + n_frames-1 targets).
        resolution           : image resolution (default 512).
        downsample_ratio     : VAE downsampling factor (default 8, so latent is 64²).
        fps                  : video frame rate (used for audio alignment).
        audio_context_frames : number of frames on each side of the current frame
                               to include in the audio window (default 2 → 5-frame window).
        add_mouth            : whether to include mouth vertices in FLAME skinner.
    """

    def __init__(
        self,
        clip_list_path: str,
        video_root: str,
        audio_root: str,
        flame_root: str,
        n_frames: int = 16,
        resolution: int = 512,
        downsample_ratio: int = 8,
        fps: float = 25.0,
        audio_context_frames: int = 2,
        add_mouth: bool = True,
    ):
        self.video_root = Path(video_root)
        self.audio_root = Path(audio_root)
        self.flame_root = Path(flame_root)

        self.n_frames             = n_frames
        self.resolution           = resolution
        self.latent_res           = resolution // downsample_ratio
        self.fps                  = fps
        self.samples_per_frame    = int(SAMPLE_RATE / fps)
        self.audio_context_frames = audio_context_frames
        self.audio_window_samples = self.samples_per_frame * (1 + 2 * audio_context_frames)

        self.flame_skinner = CAP4DFlameSkinner(
            add_mouth=add_mouth,
            n_shape_params=150,
            n_expr_params=65,
        )
        self.head_vertex_ids = np.genfromtxt(HEAD_VERT_PATH).astype(int)

        # Build a flat index of non-overlapping windows across all clips. Each
        # clip is split into (T_total // n_frames) windows; frame 0 of each
        # window is the reference slot, all frames are reconstruction targets
        # (loss masked to the non-reference slots via ref_mask).
        with open(clip_list_path) as f:
            all_ids = json.load(f)
        self.clips = {}    # clip_id -> (fit, n_total)
        self.samples = []  # flat list of (clip_id, window_start)
        for clip_id in all_ids:
            clip_id = clip_id.strip()
            if not clip_id:
                continue
            fit = dict(np.load(str(self.flame_root / clip_id / "fit.npz")))
            n_t = fit["expr"].shape[0]
            if n_t < n_frames:
                continue
            self.clips[clip_id] = (fit, n_t)
            n_windows = n_t // n_frames
            for w in range(n_windows):
                self.samples.append((clip_id, w * n_frames))

    def __len__(self):
        return len(self.samples)

    def _load_and_process_frame(
        self,
        clip_id: str,
        fit: dict,
        timestep_id: int,
        cam_id: int = 0,
    ):
        """Load one video frame + its FLAME conditioning.

        Returns:
            img      : (H, W, 3) float32 in [-1, 1]
            verts    : (V, 3) in pytorch3d NDC
            offsets  : (V, 3)
        """
        flame_item = {
            "shape":   fit["shape"],
            "expr":    fit["expr"][[timestep_id]],
            "rot":     fit["rot"][[timestep_id]],
            "tra":     fit["tra"][[timestep_id]],
            "eye_rot": fit["eye_rot"][[timestep_id]],
            "fx":      fit["fx"][[cam_id]],
            "fy":      fit["fy"][[cam_id]],
            "cx":      fit["cx"][[cam_id]],
            "cy":      fit["cy"][[cam_id]],
            "extr":    fit["extr"][[cam_id]],
        }
        if "jaw_rot" in fit:
            flame_item["jaw_rot"] = fit["jaw_rot"][[timestep_id]]

        flame_out = compute_flame(self.flame_skinner, flame_item)
        verts_2d  = flame_out["verts_2d"][0, 0]
        offsets   = flame_out["offsets_3d"][0]

        crop_box = get_bbox_from_verts(verts_2d.copy(), self.head_vertex_ids)

        video_path = self.video_root / f"{clip_id}.mp4"
        img = load_frame(video_path, timestep_id)
        img = crop_image(img, crop_box, bg_value=255)
        img = rescale_image(img, self.resolution)
        img = ((img / 127.5) - 1.0).astype(np.float32)

        verts_ndc = verts_to_pytorch3d(verts_2d.copy(), np.array(crop_box))
        return img, verts_ndc, offsets

    def __getitem__(self, idx: int) -> dict:
        clip_id, window_start = self.samples[idx]
        fit, n_total = self.clips[clip_id]
        frame_ids = list(range(window_start, window_start + self.n_frames))

        audio_full = load_audio_mono(
            self.audio_root / f"{clip_id}.wav",
            expected_len=n_total * self.samples_per_frame,
        )

        imgs, verts_list, offsets_list = [], [], []
        ref_masks, audio_windows       = [], []

        for t_idx, t_frame in enumerate(frame_ids):
            img, verts, offsets = self._load_and_process_frame(
                clip_id, fit, t_frame, cam_id=0,
            )
            imgs.append(img)
            verts_list.append(verts)
            offsets_list.append(offsets)

            is_ref = int(t_idx == 0)
            ref_masks.append(
                np.ones((1, self.latent_res, self.latent_res), dtype=np.float32) * is_ref
            )
            audio_windows.append(
                frame_window(
                    audio_full, t_frame, n_total,
                    self.samples_per_frame, self.audio_context_frames,
                )
            )

        imgs     = np.stack(imgs,          axis=0)
        verts    = np.stack(verts_list,    axis=0)
        offsets  = np.stack(offsets_list,  axis=0)
        ref_mask = np.stack(ref_masks,     axis=0)
        audio    = np.stack(audio_windows, axis=0)

        # Reference = slot 0. HWC→CHW so grid_sample in SpatialConditioning consumes
        # it directly; verts stay (V, 3) to be broadcast across T downstream.
        ref_image = torch.from_numpy(imgs[0].copy()).permute(2, 0, 1)
        ref_verts = torch.from_numpy(verts[0].copy())

        return {
            "target_video": torch.from_numpy(imgs),
            "audio":        torch.from_numpy(audio),
            "hint": {
                "driver_verts":  torch.from_numpy(verts),
                "driver_deform": torch.from_numpy(offsets),
                "ref_mask":      torch.from_numpy(ref_mask),
                "ref_image":     ref_image,
                "ref_verts":     ref_verts,
            },
        }
