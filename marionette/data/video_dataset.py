"""
Video dataset for talking-head generation training.

Each clip is split into non-overlapping `n_frames`-sized target windows.
Independently per sample, a **reference frame** is drawn uniformly at random
from anywhere in the same clip and prepended as slot 0 of `target_video`.
This is the CAP4D / AnimateAnyone convention: the ref provides an identity
anchor whose pose/expression need not match the first frame of the gen
window, and the model learns to reconstruct any target pose under any ref.

Per-sample output:
    target_video : (T+1, H, W, 3)  slot 0 = ref, slots 1..T = target window
                                    in [-1, 1]. Slot 0 feeds the frozen
                                    RefFeatureExtractor; slots 1..T are the
                                    ε-MSE loss target for the gen UNet.
    audio        : (T, W_audio)    per-frame audio windows for the T gen
                                    slots only (no audio for the ref slot).
    hint:
        driver_verts  : (T, V, 3)  driver (= target) FLAME verts per gen slot,
                                    in pytorch3d NDC relative to the per-slot
                                    face crop. Consumed by the default
                                    FLAME-based conditioning to rasterize the
                                    42ch positional encoding.
        driver_deform : (T, V, 3)  per-vertex expression deformation, rasterized
                                    alongside the vert-position prop to yield
                                    the 3ch deform map in spatial_cond.
        driver_video  : (T, H, W, 3) driver-frame pixels in [-1, 1], face-cropped
                                    using the driver's own FLAME-derived head
                                    bbox (identical recipe to slots 1..T of
                                    target_video — in fact shares storage with
                                    target_video[1:]). Not used by the default
                                    SpatialConditioning (which reads only
                                    driver_verts / driver_deform); it exists
                                    so `experiments/condition_ablation/`
                                    variants that swap FLAME for natural
                                    driver-video pixels can read it without
                                    any dataset flag.

NDC = Normalized Device Coordinates — pytorch3d's convention: +x=left, +y=up,
visible content in [-1, 1] per axis.
"""

import json
from pathlib import Path

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
        clip_list_path       : JSON list of clip IDs (produced by
                               `scripts/manifest/partition_dataset.py`).
        video_root           : root dir for videos ({video_root}/{id}.mp4).
        audio_root           : root dir for audio  ({audio_root}/{id}.wav).
        flame_root           : root dir for FLAME  ({flame_root}/{id}/fit.npz).
        n_frames             : number of gen target frames per sample. The ref
                               slot is added on top, so each returned sample
                               carries n_frames + 1 video frames in
                               `target_video`.
        resolution           : image resolution (default 512).
        downsample_ratio     : VAE downsampling factor (default 8).
        fps                  : video frame rate (used for audio alignment).
        audio_context_frames : number of frames on each side of the current
                               frame to include in the audio window.
        add_mouth            : include mouth vertices in the FLAME skinner.
        ref_sampling_seed    : base seed for per-sample ref draws. Combined
                               with sample index so two workers / epochs
                               agree on which frame is the ref.
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
        ref_sampling_seed: int = 0,
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
        self.ref_sampling_seed    = ref_sampling_seed

        self.flame_skinner = CAP4DFlameSkinner(
            add_mouth=add_mouth,
            n_shape_params=150,
            n_expr_params=65,
        )
        self.head_vertex_ids = np.genfromtxt(HEAD_VERT_PATH).astype(int)

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
        """Load one video frame + its FLAME conditioning at this timestep.

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

        # Deterministic ref draw: same idx always picks the same ref frame.
        rng = np.random.default_rng(self.ref_sampling_seed ^ (idx + 0x9E3779B1))
        ref_frame_id = int(rng.integers(0, n_total))

        frame_ids = [ref_frame_id] + list(
            range(window_start, window_start + self.n_frames)
        )

        audio_full = load_audio_mono(
            self.audio_root / f"{clip_id}.wav",
            expected_len=n_total * self.samples_per_frame,
        )

        imgs, verts_list, offsets_list, audio_windows = [], [], [], []
        for t_idx, t_frame in enumerate(frame_ids):
            img, verts, offsets = self._load_and_process_frame(
                clip_id, fit, t_frame, cam_id=0,
            )
            imgs.append(img)
            verts_list.append(verts)
            offsets_list.append(offsets)

            # Audio is aligned with gen slots only (skip slot 0 = ref).
            if t_idx >= 1:
                audio_windows.append(
                    frame_window(
                        audio_full, t_frame, n_total,
                        self.samples_per_frame, self.audio_context_frames,
                    )
                )

        imgs    = np.stack(imgs,              axis=0)   # (T+1, H, W, 3)
        verts   = np.stack(verts_list[1:],    axis=0)   # (T, V, 3) gen only
        offsets = np.stack(offsets_list[1:],  axis=0)   # (T, V, 3) gen only
        audio   = np.stack(audio_windows,     axis=0)   # (T, W_audio)

        target_video = torch.from_numpy(imgs)                  # (T+1, H, W, 3)
        return {
            "target_video": target_video,
            "audio":        torch.from_numpy(audio),
            "hint": {
                "driver_verts":  torch.from_numpy(verts),
                "driver_deform": torch.from_numpy(offsets),
                # Shares storage with `target_video[1:]` — zero memory overhead,
                # single source of truth for the driver-slot pixels. Variants in
                # `experiments/condition_ablation/` that condition on natural
                # video read this key.
                "driver_video":  target_video[1:],
            },
        }
