"""
Quick integration test for TalkingHeadDataset.
Loads samples with and without background stabilization,
saves side-by-side comparisons.

Run:
    cd <repo_root>
    python talkinghead/tests/test_dataset.py

Outputs saved to: outputs/dataset_vis/
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import cv2
from pathlib import Path

from marionette.data.video_dataset import TalkingHeadDataset

OUT_DIR = Path("outputs/dataset_vis")


def denormalize_img(img_tensor):
    """Convert from [-1, 1] float tensor to [0, 255] uint8 numpy BGR."""
    img = img_tensor.numpy()
    img = ((img + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(img[..., ::-1])  # RGB → BGR


def add_label(img, label, font_scale=0.7, thickness=2):
    """Add a text label at the top-left of an image."""
    img = img.copy()
    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
    return img


def make_comparison(sample_no_bg, sample_bg, n_show=8):
    """
    Create a 2-row comparison image:
      Top row:    frames WITHOUT background stabilization
      Bottom row: frames WITH background stabilization
    Returns a BGR image with labels.
    """
    T = sample_no_bg["jpg"].shape[0]
    indices = np.linspace(0, T - 1, min(T, n_show), dtype=int)

    label_w = 220
    rows = []
    for sample, label in [(sample_no_bg, "No BG Stab"), (sample_bg, "With BG Stab")]:
        frames = []
        for idx in indices:
            frame = denormalize_img(sample["jpg"][idx])
            cv2.putText(frame, f"t={idx}", (10, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            frames.append(frame)
        strip = np.concatenate(frames, axis=1)

        # Add label column on the left
        H = strip.shape[0]
        label_col = np.zeros((H, label_w, 3), dtype=np.uint8)
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_y = (H + text_size[1]) // 2
        cv2.putText(label_col, label, (10, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)
        rows.append(np.concatenate([label_col, strip], axis=1))

    return np.concatenate(rows, axis=0)


def make_bg_plate_vis(ds_bg, idx):
    """
    Visualize the background plate for a clip:
      Row 1: Original frames (with bg stab applied)
      Row 2: The bg plate cropped to each frame's crop box
      Row 3: The fg mask for each frame
    """
    clip_id, _ = ds_bg.samples[idx]
    fit, n_total = ds_bg.clips[clip_id]

    # Get the cached bg plate
    bg_plate = ds_bg._get_bg_plate(clip_id, n_total)
    if bg_plate is None:
        return None

    from marionette.utils.background import crop_background_plate, load_fg_mask
    from marionette.data.utils import get_bbox_from_verts
    from marionette.flame.flame import compute_flame

    mask_dir = ds_bg._get_mask_dir(clip_id)
    resolution = ds_bg.resolution

    # Pick 8 evenly spaced frames
    n_show = min(n_total, 8)
    frame_ids = np.linspace(0, n_total - 1, n_show, dtype=int)

    frame_row, bg_row, mask_row = [], [], []

    for fid in frame_ids:
        # Compute crop_box for this frame
        flame_item = {
            "shape": fit["shape"],
            "expr":  fit["expr"][[fid]],
            "rot":   fit["rot"][[fid]],
            "tra":   fit["tra"][[fid]],
            "eye_rot": fit["eye_rot"][[fid]],
            "fx":    fit["fx"][[0]],
            "fy":    fit["fy"][[0]],
            "cx":    fit["cx"][[0]],
            "cy":    fit["cy"][[0]],
            "extr":  fit["extr"][[0]],
        }
        flame_out = compute_flame(ds_bg.flame_skinner, flame_item)
        verts_2d = flame_out["verts_2d"][0, 0]
        crop_box = get_bbox_from_verts(verts_2d.copy(), ds_bg.head_vertex_ids)

        # Crop bg plate
        bg_cropped = crop_background_plate(bg_plate, np.array(crop_box), resolution)
        bg_bgr = bg_cropped[..., ::-1].copy()
        cv2.putText(bg_bgr, f"t={fid}", (10, resolution - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        bg_row.append(bg_bgr)

        # Load fg mask
        fg_mask = load_fg_mask(str(mask_dir), fid, np.array(crop_box), resolution)
        mask_vis = (fg_mask * 255).astype(np.uint8)
        mask_bgr = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)
        cv2.putText(mask_bgr, f"t={fid}", (10, resolution - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        mask_row.append(mask_bgr)

        # Load original frame (with bg stab)
        from marionette.data.utils import load_frame, crop_image, rescale_image
        from marionette.utils.background import composite_frame_with_background
        video_path = ds_bg.video_root / f"{clip_id}.mp4"
        frame = load_frame(video_path, fid)
        frame = crop_image(frame, crop_box, bg_value=255)
        frame = rescale_image(frame, resolution)
        frame_norm = ((frame / 127.5) - 1.0).astype(np.float32)
        frame_comp = composite_frame_with_background(frame_norm, bg_cropped, fg_mask, feather_radius=5)
        frame_vis = ((frame_comp + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        frame_bgr = frame_vis[..., ::-1].copy()
        cv2.putText(frame_bgr, f"t={fid}", (10, resolution - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        frame_row.append(frame_bgr)

    label_w = 220
    all_rows = []
    for row_frames, label in [(frame_row, "Composited"), (bg_row, "BG Plate"), (mask_row, "FG Mask")]:
        strip = np.concatenate(row_frames, axis=1)
        H = strip.shape[0]
        label_col = np.zeros((H, label_w, 3), dtype=np.uint8)
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_y = (H + text_size[1]) // 2
        cv2.putText(label_col, label, (10, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)
        all_rows.append(np.concatenate([label_col, strip], axis=1))

    return np.concatenate(all_rows, axis=0)


def main():
    common_args = dict(
        id_list_path="marionette/data/train_ids.txt",
        video_root="data/talkvid/talkvid",
        audio_root="data/talkvid/audio",
        flame_root="data/flowface",
        n_frames=16,
        resolution=512,
        downsample_ratio=8,
        fps=25.0,
        audio_context_frames=2,
        use_ray_directions=False,
        add_mouth=True,
    )

    print("Loading dataset WITHOUT background stabilization...")
    ds_no_bg = TalkingHeadDataset(**common_args, stabilize_background=False)
    print(f"  {len(ds_no_bg)} clips")

    print("Loading dataset WITH background stabilization...")
    ds_bg = TalkingHeadDataset(**common_args, stabilize_background=True, feather_radius=5)
    print(f"  {len(ds_bg)} clips")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Test on a few samples
    test_indices = [0, 50, 98]
    for idx in test_indices:
        if idx >= len(ds_no_bg):
            continue

        clip_id = ds_no_bg.samples[idx][0]
        print(f"\n--- Sample {idx}: {clip_id} ---")

        # Windowing is now deterministic (non-overlapping segments),
        # so both datasets return the same frames for the same index.
        sample_no_bg = ds_no_bg[idx]
        sample_bg = ds_bg[idx]

        # 1. Side-by-side comparison: with vs without bg stabilization
        comp = make_comparison(sample_no_bg, sample_bg)
        path = OUT_DIR / f"comparison_{idx:03d}.png"
        cv2.imwrite(str(path), comp)
        print(f"  Saved {path.name}")

        # 2. Background plate breakdown (composited / plate / mask)
        plate_vis = make_bg_plate_vis(ds_bg, idx)
        if plate_vis is not None:
            path = OUT_DIR / f"bg_breakdown_{idx:03d}.png"
            cv2.imwrite(str(path), plate_vis)
            print(f"  Saved {path.name}")
        else:
            print(f"  No bg masks found for {clip_id}, skipping breakdown")

        # 3. Check that shapes are identical
        for key in ["jpg", "audio"]:
            assert sample_no_bg[key].shape == sample_bg[key].shape, \
                f"Shape mismatch for {key}: {sample_no_bg[key].shape} vs {sample_bg[key].shape}"
        print("  Shape checks passed")

    # Save the full bg plate for one clip (uncropped)
    clip_id, _ = ds_bg.samples[0]
    _, n_total = ds_bg.clips[clip_id]
    bg_plate = ds_bg._get_bg_plate(clip_id, n_total)
    if bg_plate is not None:
        path = OUT_DIR / "bg_plate_full.png"
        cv2.imwrite(str(path), bg_plate[..., ::-1])
        print(f"\nSaved full uncropped bg plate: {path}")

    print(f"\nAll outputs saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
