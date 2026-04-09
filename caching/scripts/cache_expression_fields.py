"""
Precompute and cache 45-channel expression fields for all clips.

For each clip with a fit.npz, computes the full expression field via
FLAME + PyTorch3D rasterization and saves:
  - data/derived/expression_field/{clip_id}/deformation.mp4 (channels 42:45 as video)
  - data/derived/expression_field/{clip_id}/deform_rgb/     (per-frame deformation PNGs)
  - data/derived/expression_field/{clip_id}/expr_field.pt   (45ch tensor, optional with --save_tensor)

Supports multi-GPU via manual sharding (--gpu / --num_gpus).

Usage:
    cd /data/pouyan/baseline/repository/cap4d

    # Single GPU:
    PYTHONPATH=. python caching/scripts/cache_expression_fields.py

    # Also save the full 45ch tensor (~5GB each):
    PYTHONPATH=. python caching/scripts/cache_expression_fields.py --save_tensor

    # Parallel across 4 GPUs:
    PYTHONPATH=. python caching/scripts/cache_expression_fields.py --gpu 0 --num_gpus 4
    PYTHONPATH=. python caching/scripts/cache_expression_fields.py --gpu 1 --num_gpus 4
    PYTHONPATH=. python caching/scripts/cache_expression_fields.py --gpu 2 --num_gpus 4
    PYTHONPATH=. python caching/scripts/cache_expression_fields.py --gpu 3 --num_gpus 4

    # Test on one clip:
    PYTHONPATH=. python caching/scripts/cache_expression_fields.py --test
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
torch.backends.cudnn.enabled = False

from tqdm import tqdm

FLOWFACE_DIR = Path("data/flowface")
OUTPUT_DIR = Path("data/derived/expression_field")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--flowface_dir", type=str, default=str(FLOWFACE_DIR))
    p.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--num_gpus", type=int, default=1)
    p.add_argument("--test", action="store_true")
    p.add_argument("--clip", type=str, default=None, help="Process a specific clip")
    p.add_argument("--save_tensor", action="store_true",
                   help="Save full 45ch expr_field.pt (~5GB each). Default: only save deformation video + PNGs.")
    return p.parse_args()


def normalize_to_uint8(arr):
    abs_max = np.abs(arr).max() + 1e-8
    return ((arr / abs_max + 1) / 2 * 255).clip(0, 255).astype(np.uint8)


def save_deform_outputs(expr_field, clip_dir, fps):
    """
    Save deformation map (channels 42:45) as:
      - clip_dir/deformation.mp4     (video)
      - clip_dir/deform_rgb/*.png    (per-frame images)
    """
    T = expr_field.shape[0]
    H, W = expr_field.shape[2], expr_field.shape[3]
    deform_np = expr_field[:, 42:45].numpy()

    # Video
    writer = cv2.VideoWriter(
        str(clip_dir / "deformation.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H),
    )

    # Per-frame images
    frames_dir = clip_dir / "deform_rgb"
    frames_dir.mkdir(parents=True, exist_ok=True)

    for t in range(T):
        deform = deform_np[t].transpose(1, 2, 0)  # [H, W, 3]
        vis = normalize_to_uint8(deform)
        bgr = vis[..., ::-1]

        writer.write(bgr)
        cv2.imwrite(str(frames_dir / f"{t:05d}.png"), bgr)

    writer.release()


def compute_expr_field(clip_id, flame_root, resolution, device):
    """
    Compute full 45-channel expression field from fit.npz.

    Returns:
        expr_field: [T, 45, H, W] float32 tensor (CPU)
    """
    from talkinghead_sd21_unet_cap4d_based.conditioning.th_conditioning import THConditioning
    from talkinghead_sd21_unet_cap4d_based.flame.flame import CAP4DFlameSkinner, compute_flame
    from talkinghead_sd21_unet_cap4d_based.data.utils import get_bbox_from_verts, verts_to_pytorch3d

    flame_root = Path(flame_root)
    H = resolution

    conditioning = THConditioning(
        image_size=H, positional_channels=42, positional_multiplier=1.0,
        super_resolution=1, use_ray_directions=False,
        use_expr_deformation=True, use_crop_mask=False,
    ).eval().to(device)

    flame_skinner = CAP4DFlameSkinner(
        add_mouth=True, n_shape_params=150, n_expr_params=65,
    )
    head_vertex_ids = np.genfromtxt("data/assets/flame/head_vertices.txt").astype(int)

    fit = dict(np.load(str(flame_root / clip_id / "fit.npz")))
    total_frames = fit["expr"].shape[0]

    frames = []
    for t in range(total_frames):
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

        pos_enc = out["pos_enc"][0, 0, :, :, :45].permute(2, 0, 1)
        frames.append(pos_enc.cpu())

    return torch.stack(frames, dim=0)


def main():
    args = parse_args()
    device = f"cuda:{args.gpu}"
    flowface_dir = Path(args.flowface_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all clips with fit.npz
    if args.clip:
        all_clips = [args.clip]
    else:
        all_clips = sorted([
            d.name for d in flowface_dir.iterdir()
            if d.is_dir() and (d / "fit.npz").exists()
        ])

    print(f"Found {len(all_clips)} clips with fit.npz")

    # Shard across GPUs
    my_clips = all_clips[args.gpu::args.num_gpus]
    print(f"GPU {args.gpu}/{args.num_gpus}: processing {len(my_clips)} clips")

    if args.test:
        my_clips = my_clips[:1]
        print(f"Test mode: processing {my_clips[0]}")

    # Skip already processed
    done_marker = "expr_field.pt" if args.save_tensor else "deformation.mp4"
    to_process = [
        c for c in my_clips
        if not (output_dir / c / done_marker).exists()
    ]
    print(f"To process: {len(to_process)} | Already done: {len(my_clips) - len(to_process)}")

    if not to_process:
        print("Nothing to do.")
        return

    n_ok = n_fail = 0
    for clip_id in tqdm(to_process, desc=f"GPU {args.gpu}"):
        try:
            expr_field = compute_expr_field(
                clip_id, args.flowface_dir, args.resolution, device,
            )

            clip_dir = output_dir / clip_id
            clip_dir.mkdir(parents=True, exist_ok=True)

            # Save full 45ch tensor (optional — ~5GB each)
            if args.save_tensor:
                torch.save(expr_field, str(clip_dir / "expr_field.pt"))

            # Save deformation video + per-frame images
            save_deform_outputs(expr_field, clip_dir, args.fps)

            n_ok += 1

            if args.test:
                print(f"\n  Shape: {expr_field.shape}")
                print(f"  Range: [{expr_field.min():.4f}, {expr_field.max():.4f}]")
                print(f"  Saved to: {clip_dir}")

        except Exception as e:
            tqdm.write(f"[FAIL] {clip_id}: {e}")
            n_fail += 1

    print(f"Done. Success: {n_ok} | Failed: {n_fail}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
