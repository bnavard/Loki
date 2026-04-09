"""
Visualize ground truth expression field from a FLAME fit.npz.

Computes the 45-channel expression field via THConditioning + PyTorch3D
rasterization, then saves visualization videos.

Usage:
    cd <repo_root>
    PYTHONPATH=. python text_to_expr_field/scripts/visualize_ground_truth.py \
        --clip_id 39Y_gFC9SmY_NA_1123.760_1128.801 \
        --num_frames 24 \
        --output_dir outputs/ground_truth
"""

import argparse
from pathlib import Path

import numpy as np
import torch
torch.backends.cudnn.enabled = False

from text_to_expr_field.src.vis import visualize_expr_field


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--clip_id", required=True, help="Clip ID under data/flowface/")
    p.add_argument("--flame_root", default="data/flowface")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--num_frames", type=int, default=24)
    p.add_argument("--output_dir", default="outputs/ground_truth")
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def compute_expr_field(clip_id, flame_root, resolution, num_frames, device):
    """Compute 45-channel expression field from fit.npz via THConditioning."""
    from talkinghead_sd21_unet_cap4d_based.conditioning.th_conditioning import THConditioning
    from talkinghead_sd21_unet_cap4d_based.flame.flame import CAP4DFlameSkinner, compute_flame
    from talkinghead_sd21_unet_cap4d_based.data.utils import get_bbox_from_verts, verts_to_pytorch3d

    conditioning = THConditioning(
        image_size=resolution, positional_channels=42,
        positional_multiplier=1.0, super_resolution=1,
        use_ray_directions=False, use_expr_deformation=True,
        use_crop_mask=False,
    ).eval().to(device)

    flame_skinner = CAP4DFlameSkinner(
        add_mouth=True, n_shape_params=150, n_expr_params=65,
    )
    head_vertex_ids = np.genfromtxt("data/assets/flame/head_vertices.txt").astype(int)

    fit = dict(np.load(str(Path(flame_root) / clip_id / "fit.npz")))
    H = resolution
    n = min(num_frames, fit["expr"].shape[0])
    print(f"Clip {clip_id}: {fit['expr'].shape[0]} total frames, using first {n}")

    frames = []
    for t in range(n):
        flame_item = {
            "shape": fit["shape"],
            "expr": fit["expr"][[t]], "rot": fit["rot"][[t]],
            "tra": fit["tra"][[t]], "eye_rot": fit["eye_rot"][[t]],
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

        frames.append(out["pos_enc"][0, 0, :, :, :45].permute(2, 0, 1).cpu())

        if (t + 1) % 10 == 0 or t == n - 1:
            print(f"  Frame {t+1}/{n}")

    return torch.stack(frames, dim=0)


def main():
    args = parse_args()

    expr_field = compute_expr_field(
        args.clip_id, args.flame_root, args.resolution,
        args.num_frames, args.device,
    )
    print(f"Expression field: {expr_field.shape}")
    print(f"Value range: [{expr_field.min():.4f}, {expr_field.max():.4f}]")

    sample_dir = Path(args.output_dir) / args.clip_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    torch.save(expr_field, str(sample_dir / "expr_field.pt"))
    visualize_expr_field(expr_field, sample_dir, args.fps)


if __name__ == "__main__":
    main()
