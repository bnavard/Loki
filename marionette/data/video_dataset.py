"""
Video dataset for talking-head generation training.

Each clip is split into non-overlapping n_frames-sized windows (deterministic).
Frame 0 of each window is the reference frame; frames 1:n_frames are the
generation targets. Audio windows are extracted from the same temporal segment,
ensuring audio-expression alignment.

The flat sample index maps across all windows from all clips, so a 125-frame
clip at n_frames=16 yields 7 training samples (windows starting at 0, 16, 32, ...).
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from marionette.flame.flame import CAP4DFlameSkinner, compute_flame
from marionette.data.utils import (
    load_frame,
    crop_image,
    rescale_image,
    get_bbox_from_verts,
    verts_to_pytorch3d,
)

try:
    import soundfile as sf
    _HAS_SOUNDFILE = True
except ImportError:
    _HAS_SOUNDFILE = False


HEAD_VERT_PATH = "data/assets/flame/head_vertices.txt"
SAMPLE_RATE    = 16_000   # expected audio sample rate


class TalkingHeadDataset(Dataset):
    """
    Args:
        clip_list_path       : path to a JSON file containing the clip IDs used
                               by this split. Expected format: a flat list of
                               strings, as produced by
                               `scripts/manifest/partition_dataset.py`
                               (`data/derived/train_clips.json` /
                               `data/derived/val_clips.json`).
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
        expression_source    : "gt" (default) — returns FLAME vertex/offset data so
                               that THConditioning can rasterize the deformation
                               map on the fly. "marigold" — loads the pre-generated
                               deformation map video from
                               `{marigold_deform_root}/{clip_id}/deformation.mp4`
                               and packs the requested frames into the hint dict as
                               `marigold_deform`.
        marigold_deform_root : directory containing per-clip Marigold-predicted
                               deformation videos
                               (`{root}/{clip_id}/deformation.mp4`). Required when
                               `expression_source="marigold"`.
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
        expression_source: str = "gt",
        marigold_deform_root: Optional[str] = None,
    ):
        assert expression_source in ("gt", "marigold"), \
            f"expression_source must be 'gt' or 'marigold', got {expression_source!r}"
        if expression_source == "marigold" and marigold_deform_root is None:
            raise ValueError(
                "expression_source='marigold' requires marigold_deform_root to be set"
            )
        self.expression_source = expression_source
        self.marigold_deform_root = (
            Path(marigold_deform_root) if marigold_deform_root is not None else None
        )

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

        # Build a flat index of non-overlapping windows across all clips.
        # Each clip is split into (T_total // n_frames) windows of n_frames.
        # The reference frame is the first frame of each window.
        # This gives deterministic, non-overlapping segments with audio aligned
        # to the same temporal window.
        with open(clip_list_path) as f:
            all_ids = json.load(f)
        self.clips = {}  # clip_id -> (fit, n_total)
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

    # ------------------------------------------------------------------
    # Per-frame helpers
    # ------------------------------------------------------------------
    def _load_audio(self, clip_id: str, n_total_frames: int) -> Optional[np.ndarray]:
        """Load full audio as (n_samples,) float32 array. Returns None if missing."""
        audio_path = self.audio_root / f"{clip_id}.wav"
        if not audio_path.exists() or not _HAS_SOUNDFILE:
            return None
        audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio[:, 0]  # take first channel
        expected_len = n_total_frames * self.samples_per_frame
        if len(audio) < expected_len:
            audio = np.pad(audio, (0, expected_len - len(audio)))
        return audio

    def _get_audio_window(self, audio: Optional[np.ndarray], frame_id: int, n_total: int) -> np.ndarray:
        """
        Extract a context window of audio centred on frame_id.
        Returns (audio_window_samples,) float32.
        """
        if audio is None:
            return np.zeros(self.audio_window_samples, dtype=np.float32)

        ctx = self.audio_context_frames
        start_frame = frame_id - ctx
        end_frame   = frame_id + ctx + 1

        window = np.zeros(self.audio_window_samples, dtype=np.float32)
        for i, f in enumerate(range(start_frame, end_frame)):
            f_clamp = max(0, min(f, n_total - 1))
            src_start = f_clamp * self.samples_per_frame
            src_end   = src_start + self.samples_per_frame
            dst_start = i * self.samples_per_frame
            dst_end   = dst_start + self.samples_per_frame
            window[dst_start:dst_end] = audio[src_start:src_end]

        return window

    def _load_and_process_frame(
        self,
        clip_id: str,
        fit: dict,
        timestep_id: int,
        cam_id: int = 0,
    ):
        """
        Load one video frame and compute its FLAME conditioning.

        Returns:
            img        : (H, W, 3) float32 in [-1, 1]
            verts_2d   : (V, 2)  in PyTorch3D space
            offsets_3d : (V, 3)
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

        flame_out  = compute_flame(self.flame_skinner, flame_item)
        verts_2d   = flame_out["verts_2d"][0, 0]   # (V, 2)
        offsets_3d = flame_out["offsets_3d"][0]     # (V, 3)

        crop_box = get_bbox_from_verts(verts_2d.copy(), self.head_vertex_ids)

        video_path = self.video_root / f"{clip_id}.mp4"
        img = load_frame(video_path, timestep_id)
        img = crop_image(img, crop_box, bg_value=255)
        img = rescale_image(img, self.resolution)
        img = ((img / 127.5) - 1.0).astype(np.float32)

        verts_2d_p3d = verts_to_pytorch3d(verts_2d.copy(), np.array(crop_box))

        return img, verts_2d_p3d, offsets_3d

    def _load_marigold_deform_window(self, clip_id: str, frame_ids: list) -> torch.Tensor:
        """Load Marigold-predicted deformation frames by decoding the cached mp4.

        Reads `{marigold_deform_root}/{clip_id}/deformation.mp4`, extracts only
        the frames in `frame_ids`, and returns a normalized float tensor.

        Returns:
            (T, 3, H, W) float32 in [-1, 1], resized to `self.resolution`.
        """
        video_path = self.marigold_deform_root / clip_id / "deformation.mp4"
        if not video_path.exists():
            raise FileNotFoundError(
                f"Marigold deformation video not found: {video_path}. "
                f"Run scripts/cache/cache_marigold_deform.py first."
            )

        frames = [load_frame(video_path, f) for f in frame_ids]  # list of (H, W, 3) uint8
        frames = [rescale_image(f, self.resolution) for f in frames]
        arr = np.stack(frames, axis=0).astype(np.float32)        # (T, H, W, 3) in [0, 255]
        arr = arr / 127.5 - 1.0                                  # normalize to [-1, 1]
        arr = np.transpose(arr, (0, 3, 1, 2))                    # (T, 3, H, W)
        return torch.from_numpy(arr)

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict:
        clip_id, window_start = self.samples[idx]
        fit, n_total = self.clips[clip_id]

        # Frame 0 of the window is the reference frame.
        # Frames 1 through n_frames-1 are the target (generation) frames.
        # All frames come from the same contiguous temporal segment,
        # ensuring audio and expression are naturally aligned.
        ref_frame_id = window_start
        target_ids = list(range(window_start + 1, window_start + self.n_frames))

        all_frame_ids = [ref_frame_id] + target_ids

        # Load audio once for the whole clip
        audio_full = self._load_audio(clip_id, n_total)

        imgs, verts_list, offsets_list, ref_masks, audio_windows = [], [], [], [], []

        for t_idx, frame_id in enumerate(all_frame_ids):
            img, verts_2d, offsets_3d = self._load_and_process_frame(
                clip_id, fit, frame_id, cam_id=0
            )

            is_ref = int(t_idx == 0)

            # Reference mask as a spatial map (broadcast to latent resolution)
            ref_mask = np.ones((1, self.latent_res, self.latent_res), dtype=np.float32) * is_ref

            imgs.append(img)
            verts_list.append(verts_2d)
            offsets_list.append(offsets_3d)
            ref_masks.append(ref_mask)
            audio_windows.append(self._get_audio_window(audio_full, frame_id, n_total))

        # Stack along time axis
        imgs     = np.stack(imgs,         axis=0)   # (T, H, W, 3)
        verts    = np.stack(verts_list,   axis=0)   # (T, V, 2)
        offsets  = np.stack(offsets_list, axis=0)   # (T, V, 3)
        ref_mask = np.stack(ref_masks,    axis=0)   # (T, 1, h, w)
        audio    = np.stack(audio_windows, axis=0)  # (T, window_samples)

        hint = {
            "verts_2d":       torch.tensor(verts),
            "offsets_3d":     torch.tensor(offsets),
            "reference_mask": torch.tensor(ref_mask),
        }

        if self.expression_source == "marigold":
            # Pre-generated deformation map replaces the rasterization path.
            # THConditioning will detect this key and bypass mesh rasterization.
            hint["marigold_deform"] = self._load_marigold_deform_window(clip_id, all_frame_ids)

        return {
            "jpg":   torch.tensor(imgs),           # (T, H, W, 3) float32 [-1,1]
            "audio": torch.tensor(audio),          # (T, window_samples) float32
            "hint":  hint,
        }