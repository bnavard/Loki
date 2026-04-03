#!/usr/bin/env python3
"""
Generate expression maps from flowface output for conditioning a gen AI model.

For each video in FLOWFACE_DIR with a fit.npz:
  1. Load frames via CAP4DInferenceDataset
  2. Run CAP4DConditioning to produce pos_enc  [T, H, W, C]  float32
  3. Save to EXP_MAP_DIR/{video_name}.npz  (key: 'pos_enc')

Input:  /data/pouyan/flame_expression/flowface/{video_name}/fit.npz
Output: /data/pouyan/flame_expression/exp_map/{video_name}.npz

Usage:
    python generate_exp_map.py                          # all pending
    python generate_exp_map.py --test                   # one video + visualization
    python generate_exp_map.py --test --video NAME      # specific video + visualization
"""

import os
import sys
import argparse
import traceback
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ============================================================================
# PATHS
# ============================================================================

REPO_ROOT    = Path(__file__).parent.resolve()
FLOWFACE_DIR = Path("data/flowface")
EXP_MAP_DIR  = Path("data/exp_map")

# ============================================================================
# CONDITIONING
# ============================================================================

def load_conditioning():
    os.chdir(REPO_ROOT)
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "conditioning"))

    from conditioning.conditioning.cap4dcond import CAP4DConditioning

    conditioning = CAP4DConditioning(
        image_size=64,
        positional_channels=42,
        positional_multiplier=1.0,
        super_resolution=2,
        use_ray_directions=True,
        use_expr_deformation=True,
        use_crop_mask=False,
    )
    conditioning.eval()
    return conditioning


def process_video(video_dir, conditioning, output_path):
    """Run CAP4DConditioning on all frames; save pos_enc [T, H, W, C] float32."""
    from dataset import CAP4DInferenceDataset

    dataset = CAP4DInferenceDataset(data_path=video_dir, resolution=512, downsample_ratio=8)
    if len(dataset) == 0:
        raise RuntimeError(f"Empty dataset: {video_dir.name}")

    all_pos_enc = []
    for idx in range(len(dataset)):
        item  = dataset[idx]
        batch = {
            k: torch.from_numpy(v).unsqueeze(0) if isinstance(v, np.ndarray) else v
            for k, v in item["hint"].items()
        }
        with torch.no_grad():
            out = conditioning(batch, unconditional=False)
        all_pos_enc.append(out["pos_enc"][0, 0].cpu().numpy())  # [H, W, C]

    pos_enc_array = np.stack(all_pos_enc, axis=0).astype(np.float32)  # [T, H, W, C]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(output_path), pos_enc=pos_enc_array)
    return pos_enc_array


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_exp_map(video_dir, pos_enc_array, out_dir, conditioning):
    """Save per-frame deformation map PNGs + a summary grid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vis_dir = out_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    n_frames = pos_enc_array.shape[0]
    sample_indices = list(range(0, n_frames, max(1, n_frames // 8)))[:8]

    fig, axes = plt.subplots(2, len(sample_indices), figsize=(4 * len(sample_indices), 8))
    if len(sample_indices) == 1:
        axes = axes[:, None]

    for col, idx in enumerate(sample_indices):
        pos_enc = pos_enc_array[idx]
        visualizations = conditioning.get_vis(pos_enc)

        for name, vis in visualizations.items():
            if name == "expr_disp":
                abs_max = np.abs(vis).max()
                vis_norm = (vis / (abs_max + 1e-8) + 1) / 2
                vis_norm = np.clip(vis_norm, 0, 1)
                plt.imsave(str(vis_dir / f"frame_{idx:04d}_expr_disp.png"), vis_norm)
            elif name.startswith("pose_map"):
                vmin, vmax = vis.min(), vis.max()
                vis_norm = (vis - vmin) / (vmax - vmin + 1e-8)
                vis_norm = np.clip(vis_norm, 0, 1)
                plt.imsave(str(vis_dir / f"frame_{idx:04d}_{name}.png"), vis_norm)

        # Grid: top = pose_map, bottom = expr_disp
        for name, vis in visualizations.items():
            if name.startswith("pose_map"):
                vmin, vmax = vis.min(), vis.max()
                vis_g = (vis - vmin) / (vmax - vmin + 1e-8)
                axes[0, col].imshow(np.clip(vis_g, 0, 1))
                axes[0, col].set_title(f"frame {idx}\n{name}", fontsize=9)
                axes[0, col].axis("off")
            elif name == "expr_disp":
                abs_max = np.abs(vis).max()
                vis_g = (vis / (abs_max + 1e-8) + 1) / 2
                axes[1, col].imshow(np.clip(vis_g, 0, 1))
                axes[1, col].set_title("expr_disp", fontsize=9)
                axes[1, col].axis("off")

    plt.suptitle(f"{video_dir.name}  —  {n_frames} frames, shape {pos_enc_array.shape}", fontsize=12)
    plt.tight_layout()
    grid_path = out_dir / "expression_map_grid.png"
    plt.savefig(str(grid_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Visualization grid  → {grid_path}")
    print(f"Individual frames   → {vis_dir}/")


# ============================================================================
# BATCH
# ============================================================================

def run_batch():
    """Generate expression maps for all flowface-converted videos."""
    if not FLOWFACE_DIR.exists():
        print("No flowface output found. Run run_parallel_flowface.py first.")
        return

    video_dirs = sorted(d for d in FLOWFACE_DIR.iterdir()
                        if d.is_dir() and (d / "fit.npz").exists())
    print(f"Found {len(video_dirs)} flowface dirs")

    to_process = [d for d in video_dirs if not (EXP_MAP_DIR / f"{d.name}.npz").exists()]
    print(f"To process: {len(to_process)} | Already done: {len(video_dirs) - len(to_process)}")

    if not to_process:
        print("Nothing to do.")
        return

    print("Loading CAP4DConditioning...")
    conditioning = load_conditioning()

    n_ok = n_fail = 0
    for video_dir in tqdm(to_process, desc="Expression maps"):
        output_path = EXP_MAP_DIR / f"{video_dir.name}.npz"
        try:
            arr = process_video(video_dir, conditioning, output_path)
            n_ok += 1
            tqdm.write(f"  {video_dir.name}: {arr.shape}")
        except Exception as e:
            tqdm.write(f"  [FAIL] {video_dir.name}: {e}")
            traceback.print_exc()
            n_fail += 1

    print(f"Done. Success: {n_ok} | Failed: {n_fail}")
    print(f"Expression maps → {EXP_MAP_DIR}")


# ============================================================================
# TEST
# ============================================================================

def run_test(video_name=None):
    """Run on one video + produce visualizations."""
    EXP_MAP_DIR.mkdir(parents=True, exist_ok=True)

    if video_name:
        video_dir = FLOWFACE_DIR / video_name
        if not (video_dir / "fit.npz").exists():
            print(f"No fit.npz found at {video_dir}")
            return
    else:
        candidates = sorted(d for d in FLOWFACE_DIR.iterdir()
                            if d.is_dir() and (d / "fit.npz").exists()) if FLOWFACE_DIR.exists() else []
        if not candidates:
            print("No flowface outputs found. Run run_parallel_flowface.py first.")
            return
        video_dir = candidates[0]

    vid_name = video_dir.name
    output_path = EXP_MAP_DIR / f"{vid_name}.npz"
    print(f"[Test] Video: {vid_name}")

    print("\nLoading CAP4DConditioning...")
    conditioning = load_conditioning()

    print("Generating expression map...")
    pos_enc_array = process_video(video_dir, conditioning, output_path)
    print(f"pos_enc shape: {pos_enc_array.shape}  →  {output_path}")

    print("\nGenerating visualizations...")
    visualize_exp_map(video_dir, pos_enc_array, EXP_MAP_DIR / vid_name, conditioning)

    print("\nTest complete!")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate expression maps from flowface output")
    parser.add_argument("--test", action="store_true",
                        help="Run on ONE video + save visualizations")
    parser.add_argument("--video", type=str, default=None,
                        help="Specific video name for --test (default: first available)")
    args = parser.parse_args()

    if args.test:
        run_test(args.video)
    else:
        run_batch()


if __name__ == "__main__":
    main()
