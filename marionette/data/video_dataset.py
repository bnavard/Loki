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
                               `marigold_deform`. "driving_video" — uses the raw
                               natural video frames downsampled to latent resolution
                               as spatial conditioning (3ch RGB in [-1, 1]),
                               packed into `hint["driving_video"]`.
        marigold_deform_root : directory containing per-clip Marigold-predicted
                               deformation videos
                               (`{root}/{clip_id}/deformation.mp4`). Required when
                               `expression_source="marigold"`.
        cross_identity_driving : when True, each sample sources the *driver signal*
                               (expression map / deformation / driving video) and
                               the *audio* from a DIFFERENT clip — deterministically
                               chosen via a seeded permutation that excludes
                               self-pairings and same-identity pairings. The target
                               clip still provides the reference frame and the
                               reconstruction target (`jpg`). Used at evaluation
                               time so the model is judged on its ability to
                               follow external expression cues rather than
                               reproducing signals extracted from the target itself.
                               Training should keep this False (no paired
                               cross-identity ground truth exists).
        pairing_seed         : RNG seed for the cross-identity clip permutation
                               and the per-sample driver window selection.
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
        cross_identity_driving: bool = False,
        pairing_seed: int = 42,
    ):
        assert expression_source in ("gt", "marigold", "driving_video"), \
            f"expression_source must be 'gt', 'marigold', or 'driving_video', got {expression_source!r}"
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

        # Cross-identity driver permutation.
        # Each target clip is mapped to a single driver clip. Pairings are
        # deterministic (seeded), exclude self-pairings, and — where possible —
        # exclude same-identity pairings (two different clips of the same
        # YouTube video). We retry the shuffle up to `max_tries` times; if the
        # set is too small to satisfy the constraint we fall back to a plain
        # derangement.
        self.cross_identity_driving = cross_identity_driving
        self.pairing_seed = pairing_seed
        self.driver_map: dict[str, str] = {}
        if cross_identity_driving:
            self.driver_map = self._build_driver_permutation(
                list(self.clips.keys()), seed=pairing_seed,
            )

    @staticmethod
    def _extract_video_id(clip_id: str) -> str:
        """Pull the YouTube video ID out of a clip_id so we can detect same-identity
        pairings and avoid them. Mirrors scripts/manifest/partition_dataset.py."""
        import re
        if "_NA_" in clip_id:
            return clip_id.split("_NA_")[0]
        m = re.search(r"videovideo(.+?)_scene", clip_id)
        return m.group(1) if m else clip_id

    @classmethod
    def _build_driver_permutation(cls, clip_ids: list, seed: int) -> dict:
        """Build a target -> driver map that minimises same-identity pairings.

        Strategy: shuffle the driver pool (seeded). For each target, scan the
        pool for the first driver with a DIFFERENT identity than the target;
        take it. If no such driver remains, take the first available driver
        (unavoidable same-identity pair). Guarantees:
          - No self-pair (a clip never drives itself).
          - Same-identity pairs only appear when a single identity dominates
            the clip set (>50% of clips) — the theoretical minimum.
        """
        import random as _rnd
        if len(clip_ids) < 2:
            return {clip_ids[0]: clip_ids[0]} if clip_ids else {}

        target_ids = list(clip_ids)
        vid_of = {c: cls._extract_video_id(c) for c in target_ids}
        rng = _rnd.Random(seed)

        driver_pool = target_ids.copy()
        rng.shuffle(driver_pool)

        mapping: dict[str, str] = {}
        # Process targets whose identity is most common first — they have the
        # fewest cross-identity drivers available, so we assign them greedily
        # while the pool is still rich.
        from collections import Counter
        id_count = Counter(vid_of.values())
        ordered_targets = sorted(
            target_ids,
            key=lambda c: (-id_count[vid_of[c]], c),
        )

        for target in ordered_targets:
            t_vid = vid_of[target]
            chosen_idx = None
            # Prefer a different-identity, non-self driver
            for i, driver in enumerate(driver_pool):
                if driver != target and vid_of[driver] != t_vid:
                    chosen_idx = i
                    break
            # Fall back to any non-self driver
            if chosen_idx is None:
                for i, driver in enumerate(driver_pool):
                    if driver != target:
                        chosen_idx = i
                        break
            if chosen_idx is None:
                # Only one distinct clip left and it's the target itself —
                # degenerate case; self-pair as a last resort.
                chosen_idx = 0
            mapping[target] = driver_pool.pop(chosen_idx)

        return mapping

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
        """Load Marigold-predicted deformation frames from the cached fp16 tensor.

        Reads `{marigold_deform_root}/{clip_id}/deform_field.pt`, indexes the
        requested frames, and returns them as a float32 tensor.

        Returns:
            (T, 3, H, W) float32 in original prediction range.
        """
        pt_path = self.marigold_deform_root / clip_id / "deform_field.pt"
        if not pt_path.exists():
            raise FileNotFoundError(
                f"Marigold deformation tensor not found: {pt_path}. "
                f"Run scripts/cache/marigold_deform/cache.py first."
            )

        deform_field = torch.load(str(pt_path), map_location="cpu", weights_only=True)
        frames = deform_field[frame_ids].float()  # (T, 3, H, W)
        if frames.shape[-1] != self.resolution:
            frames = torch.nn.functional.interpolate(
                frames, size=self.resolution, mode="bilinear", align_corners=False,
            )
        return frames

    # ------------------------------------------------------------------
    # Driver window selection
    # ------------------------------------------------------------------
    def _select_driver_window(self, idx: int, target_clip_id: str):
        """Resolve the (driver_clip_id, driver_window_start) for a sample.

        In same-identity mode (training), the driver is the target. In
        cross-identity mode (eval), the driver is determined by the permutation
        built at construction time, and a random window within that clip is
        selected deterministically per sample via `pairing_seed + idx`.
        """
        if not self.cross_identity_driving:
            target_window_start = self.samples[idx][1]
            return target_clip_id, target_window_start

        driver_clip_id = self.driver_map.get(target_clip_id, target_clip_id)
        _, driver_n_total = self.clips[driver_clip_id]
        max_start = max(0, driver_n_total - self.n_frames)
        if max_start == 0:
            return driver_clip_id, 0

        import random as _rnd
        rng = _rnd.Random(self.pairing_seed * 1_000_003 + idx)
        return driver_clip_id, rng.randint(0, max_start)

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict:
        # ---- Target: provides identity (frame 0) + reconstruction target ----
        target_clip_id, target_window_start = self.samples[idx]
        target_fit, _target_n_total = self.clips[target_clip_id]
        target_frame_ids = list(range(
            target_window_start, target_window_start + self.n_frames,
        ))

        # ---- Driver: provides expression signal + audio ----
        driver_clip_id, driver_window_start = self._select_driver_window(idx, target_clip_id)
        driver_fit, driver_n_total = self.clips[driver_clip_id]
        driver_frame_ids = list(range(
            driver_window_start, driver_window_start + self.n_frames,
        ))
        driver_audio_full = self._load_audio(driver_clip_id, driver_n_total)

        target_imgs = []
        driver_imgs = []
        driver_verts_list = []
        driver_offsets_list = []
        ref_masks = []
        audio_windows = []

        for t_idx in range(self.n_frames):
            t_frame = target_frame_ids[t_idx]
            d_frame = driver_frame_ids[t_idx]

            # Target: image (first one becomes the reference; all are reconstruction targets)
            t_img, _, _ = self._load_and_process_frame(
                target_clip_id, target_fit, t_frame, cam_id=0,
            )
            target_imgs.append(t_img)

            # Driver: FLAME + image (image only used by driving_video mode, but
            # loading it here keeps the control flow simple)
            d_img, d_verts, d_offsets = self._load_and_process_frame(
                driver_clip_id, driver_fit, d_frame, cam_id=0,
            )
            driver_imgs.append(d_img)
            driver_verts_list.append(d_verts)
            driver_offsets_list.append(d_offsets)

            is_ref = int(t_idx == 0)
            ref_masks.append(
                np.ones((1, self.latent_res, self.latent_res), dtype=np.float32) * is_ref
            )

            audio_windows.append(
                self._get_audio_window(driver_audio_full, d_frame, driver_n_total)
            )

        target_imgs   = np.stack(target_imgs,         axis=0)  # (T, H, W, 3)
        driver_imgs   = np.stack(driver_imgs,         axis=0)  # (T, H, W, 3)
        verts         = np.stack(driver_verts_list,   axis=0)  # (T, V, 2)
        offsets       = np.stack(driver_offsets_list, axis=0)  # (T, V, 3)
        ref_mask      = np.stack(ref_masks,           axis=0)  # (T, 1, h, w)
        audio         = np.stack(audio_windows,       axis=0)  # (T, window_samples)

        hint = {
            "verts_2d":       torch.tensor(verts),
            "offsets_3d":     torch.tensor(offsets),
            "reference_mask": torch.tensor(ref_mask),
        }

        if self.expression_source == "marigold":
            # Pre-generated deformation map replaces the rasterization path.
            # THConditioning will detect this key and bypass mesh rasterization.
            hint["marigold_deform"] = self._load_marigold_deform_window(
                driver_clip_id, driver_frame_ids,
            )

        elif self.expression_source == "driving_video":
            # Downsample the DRIVER's video frames to latent resolution.
            driving_frames = np.stack(
                [rescale_image(
                    ((img + 1.0) * 127.5).clip(0, 255).astype(np.uint8),
                    self.latent_res,
                ) for img in driver_imgs],
                axis=0,
            ).astype(np.float32)                                # (T, latent_res, latent_res, 3)
            driving_frames = driving_frames / 127.5 - 1.0       # back to [-1, 1]
            driving_frames = np.transpose(driving_frames, (0, 3, 1, 2))  # (T, 3, H, W)
            hint["driving_video"] = torch.from_numpy(driving_frames)

        return {
            "jpg":        torch.tensor(target_imgs),   # (T, H, W, 3) TARGET frames (reconstruction target)
            "driver_jpg": torch.tensor(driver_imgs),   # (T, H, W, 3) DRIVER frames (for vis; same as jpg when same-identity)
            "audio":      torch.tensor(audio),         # (T, window_samples) DRIVER audio
            "hint":       hint,                        # DRIVER-sourced conditioning
        }