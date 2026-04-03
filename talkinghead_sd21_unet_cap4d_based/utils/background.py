"""
Background plate builder for talking-head generation.

Builds a clean background plate from a video clip by aggregating background
pixels across all frames using per-frame foreground masks. Pixels that are
never revealed (always occluded) remain zero and can optionally be inpainted.

The plate is built in **full uncropped resolution** so it can be cropped with
any per-frame crop_box later. This handles the fact that the crop region shifts
as the head moves.

Masks are expected at:
    {flame_root}/{clip_id}/bg/cam0/{frame_id:04d}.png
Format: single-channel uint8 (512x512), where high values = foreground,
low values = background.
"""

import numpy as np
import cv2
from pathlib import Path

from talkinghead_sd21_unet_cap4d_based.data.utils import load_frame, crop_image, rescale_image


def build_background_plate(
    video_path: str,
    mask_dir: str,
    n_frames: int,
    fg_threshold: float = 0.5,
) -> np.ndarray:
    """
    Build a clean background plate in full uncropped resolution by aggregating
    background pixels across frames.

    For each pixel, takes the value from the first frame where the foreground
    mask is below the threshold (i.e. the pixel is background).

    Args:
        video_path:    Path to video file or frame directory.
        mask_dir:      Path to directory containing per-frame fg masks (0000.png, ...).
        n_frames:      Number of frames to scan.
        fg_threshold:  Pixels with mask value below this (in [0,1]) are considered
                       background.

    Returns:
        background_plate: (H_orig, W_orig, 3) uint8 RGB image (uncropped).
    """
    mask_dir = Path(mask_dir)

    # Load first frame to determine original resolution
    first_frame = load_frame(video_path, 0)
    H_orig, W_orig = first_frame.shape[:2]

    plate = np.zeros((H_orig, W_orig, 3), dtype=np.float64)
    filled = np.zeros((H_orig, W_orig), dtype=bool)

    for t in range(n_frames):
        if filled.all():
            break

        mask_path = mask_dir / f"{t:04d}.png"
        if not mask_path.exists():
            continue

        frame = load_frame(video_path, t)  # (H, W, 3) uint8 RGB
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)  # (H, W) uint8

        # Resize mask to match frame resolution if needed
        if mask.shape[:2] != (H_orig, W_orig):
            mask = cv2.resize(mask, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)

        mask_norm = mask.astype(np.float32) / 255.0
        is_bg = (mask_norm < fg_threshold) & (~filled)

        plate[is_bg] = frame[is_bg].astype(np.float64)
        filled[is_bg] = True

    # Fill remaining pixels with mean background color
    if not filled.all():
        mean_color = plate[filled].mean(axis=0) if filled.any() else np.array([128, 128, 128])
        plate[~filled] = mean_color

    return plate.astype(np.uint8)


def crop_background_plate(
    plate: np.ndarray,
    crop_box: np.ndarray,
    resolution: int = 512,
) -> np.ndarray:
    """
    Crop and resize the full-resolution background plate to match a specific
    frame's crop_box.

    Args:
        plate:      (H_orig, W_orig, 3) uint8 RGB background plate.
        crop_box:   (4,) crop box [x1, y1, x2, y2].
        resolution: Target output resolution.

    Returns:
        (resolution, resolution, 3) uint8 RGB.
    """
    cropped = crop_image(plate.astype(np.float32), crop_box, bg_value=128)
    return rescale_image(cropped.astype(np.uint8), resolution)


def composite_frame_with_background(
    frame: np.ndarray,
    bg_cropped: np.ndarray,
    fg_mask: np.ndarray,
    feather_radius: int = 5,
) -> np.ndarray:
    """
    Composite a single frame with a cropped background plate using a soft mask.

    Args:
        frame:          (H, W, 3) float32 in [-1, 1] (same as dataset output).
        bg_cropped:     (H, W, 3) uint8 RGB cropped background plate.
        fg_mask:        (H, W) float32 in [0, 1], high = foreground.
        feather_radius: Gaussian blur radius for soft edges. 0 = hard.

    Returns:
        (H, W, 3) float32 in [-1, 1].
    """
    if feather_radius > 0:
        ksize = feather_radius * 2 + 1
        fg_mask = cv2.GaussianBlur(fg_mask, (ksize, ksize), 0)

    fg_mask_3ch = fg_mask[..., None]  # (H, W, 1)

    # Convert bg plate to [-1, 1] range to match frame
    bg_norm = (bg_cropped.astype(np.float32) / 127.5) - 1.0

    blended = frame * fg_mask_3ch + bg_norm * (1.0 - fg_mask_3ch)
    return blended.astype(np.float32)


def load_fg_mask(mask_dir: str, frame_id: int, crop_box: np.ndarray,
                 resolution: int = 512) -> np.ndarray:
    """
    Load and crop a foreground mask for a specific frame.

    Returns:
        (resolution, resolution) float32 in [0, 1], high = foreground.
    """
    mask_path = Path(mask_dir) / f"{frame_id:04d}.png"
    if not mask_path.exists():
        return np.ones((resolution, resolution), dtype=np.float32)

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    # Crop and resize to match frame processing
    mask_cropped = crop_image(
        mask[..., None].astype(np.float32), crop_box, bg_value=0
    ).astype(np.uint8)[..., 0]
    mask_resized = cv2.resize(mask_cropped, (resolution, resolution),
                              interpolation=cv2.INTER_LINEAR)

    return mask_resized.astype(np.float32) / 255.0


def composite_with_background(
    generated_frames: np.ndarray,
    background_plate: np.ndarray,
    mask_dir: str,
    frame_indices: list,
    crop_box: np.ndarray,
    resolution: int = 512,
    feather_radius: int = 5,
) -> np.ndarray:
    """
    Composite generated frames (inference) with a stable background plate.

    Args:
        generated_frames: (T, 3, H, W) uint8 RGB (channel-first).
        background_plate: (H_orig, W_orig, 3) uint8 RGB (uncropped).
        mask_dir:         Path to per-frame foreground masks.
        frame_indices:    List of frame indices corresponding to generated_frames.
        crop_box:         (4,) crop box used during generation.
        resolution:       Resolution of generated frames.
        feather_radius:   Gaussian blur radius for soft mask edges. 0 = hard.

    Returns:
        composited: (T, 3, H, W) uint8 RGB (channel-first).
    """
    T = generated_frames.shape[0]
    composited = np.zeros_like(generated_frames)

    bg_cropped = crop_background_plate(background_plate, crop_box, resolution)

    for i in range(T):
        t = frame_indices[i]
        gen_hwc = generated_frames[i].transpose(1, 2, 0)  # (H, W, 3)

        fg_mask = load_fg_mask(mask_dir, t, crop_box, resolution)

        if feather_radius > 0:
            ksize = feather_radius * 2 + 1
            fg_mask = cv2.GaussianBlur(fg_mask, (ksize, ksize), 0)

        fg_mask_3ch = fg_mask[..., None]
        blended = (gen_hwc.astype(np.float32) * fg_mask_3ch +
                   bg_cropped.astype(np.float32) * (1.0 - fg_mask_3ch))
        composited[i] = blended.clip(0, 255).astype(np.uint8).transpose(2, 0, 1)

    return composited
