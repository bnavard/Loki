"""
Quick test: check how much the 46-channel expression map varies across frames.
Loads one real clip, computes THConditioning output, and reports per-channel
statistics across the T=16 frames.

Run:
    cd <repo_root>
    PYTHONPATH=. python talkinghead/tests/test_expr_map_variation.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
torch.backends.cudnn.enabled = False

import numpy as np
from talkinghead_sd21_unet_cap4d_based.data.video_dataset import TalkingHeadDataset
from talkinghead_sd21_unet_cap4d_based.conditioning.th_conditioning import THConditioning

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    # Load one sample from the dataset
    ds = TalkingHeadDataset(
        id_list_path="talkinghead/data/train_ids.txt",
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
        stabilize_background=False,
    )

    np.random.seed(42)
    sample = ds[0]

    # Build conditioning module
    cond = THConditioning(
        image_size=64,
        positional_channels=42,
        positional_multiplier=1.0,
        super_resolution=2,
        use_ray_directions=False,
        use_expr_deformation=True,
        use_crop_mask=False,
    )
    cond.to(DEVICE)

    # Prepare batch (add batch dim)
    hint = sample["hint"]
    batch = {
        "verts_2d": hint["verts_2d"].unsqueeze(0).to(DEVICE),
        "offsets_3d": hint["offsets_3d"].unsqueeze(0).to(DEVICE),
        "reference_mask": hint["reference_mask"].unsqueeze(0).to(DEVICE),
    }

    with torch.no_grad():
        out = cond(batch, unconditional=False)

    pos_enc = out["pos_enc"]  # (1, T, H, W, 46)
    pos_enc = pos_enc[0]       # (T, H, W, 46)

    T = pos_enc.shape[0]
    C = pos_enc.shape[-1]

    print(f"Expression map shape: {pos_enc.shape}")
    print(f"  T={T} frames, H=W={pos_enc.shape[1]}, C={C} channels\n")

    # Channel groups
    groups = {
        "pos_enc (0:42)": (0, 42),
        "expr_deform (42:45)": (42, 45),
        "ref_mask (45)": (45, 46),
    }

    for group_name, (c_start, c_end) in groups.items():
        channels = pos_enc[:, :, :, c_start:c_end]  # (T, H, W, n_ch)

        # Compare each frame to frame 0
        frame0 = channels[0]  # (H, W, n_ch)
        print(f"--- {group_name} ---")

        diffs = []
        for t in range(1, T):
            diff = (channels[t] - frame0).abs()
            diffs.append(diff.mean().item())
            if t <= 3 or t == T - 1:
                print(f"  Frame {t} vs frame 0: mean_abs_diff={diff.mean():.6f}, "
                      f"max_abs_diff={diff.max():.6f}, "
                      f"ratio_nonzero={((diff > 1e-6).sum() / diff.numel()):.4f}")

        avg_diff = np.mean(diffs)
        print(f"  Average diff across all frames vs frame 0: {avg_diff:.6f}")

        # Check if channels are identical across frames
        all_same = all(d < 1e-6 for d in diffs)
        print(f"  Identical across frames: {'YES' if all_same else 'NO'}\n")

    # Per-channel variance across time (at each spatial position)
    temporal_var = pos_enc.var(dim=0)  # (H, W, 46) — variance over T
    print("--- Per-channel temporal variance (averaged over spatial) ---")
    for c in range(C):
        v = temporal_var[:, :, c].mean().item()
        label = f"ch{c:02d}"
        if c < 42:
            label += " (pos_enc)"
        elif c < 45:
            label += " (expr_deform)"
        else:
            label += " (ref_mask)"
        if v > 1e-6:
            print(f"  {label}: variance={v:.6f}")
        else:
            print(f"  {label}: variance={v:.6f}  ← STATIC")


if __name__ == "__main__":
    main()
