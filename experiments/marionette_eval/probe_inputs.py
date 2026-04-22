"""
Dry-run probe for the Marionette evaluation pipeline.

Builds a handful of samples from each mode (cross-identity + same-identity) and
saves a panel of exactly what would feed the model — reference image, driver
video frames, and the rasterized FLAME spatial conditioning — WITHOUT loading
the checkpoint or running DDIM. Useful for eyeballing the pairing logic, the
FLAME retargeting, and the spatial_cond rasterization before spending GPU
time on full inference.

Saved rows per panel:
  1. Reference     — the face-cropped ref frame (static across T).
  2. Driver Video  — the driver's own face-cropped video frames over the T-window.
  3. Driver Expr   — spatial_cond[..., 42:45] (the 3-channel per-vertex
                     expression deformation, rasterized under the ref's crop).
  4. Pos Enc [0:3] — first three channels of the 42-channel positional encoding
                     of rasterized FLAME vertex positions. A sanity-check that
                     the pose-map rasterization lands in the right pixel region.

Usage (from repo root):

    conda activate marionette
    PYTHONPATH=. python experiments/marionette_eval/probe_inputs.py \\
        --config    experiments/marionette_eval/configs/cross_identity.yaml \\
        --n_samples 3

By default, probes 3 samples from each of cross-identity and same-identity.
Output lands at outputs/marionette_eval/probe/run_<ts>/{cross,same}/.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from experiments.marionette_eval.pairing import (
    build_cross_identity_samples, build_same_identity_samples,
)
from marionette.conditioning.conditioning import SpatialConditioning
from marionette.config_utils import load_experiment_config
from marionette.flame.flame import CAP4DFlameSkinner
from marionette.retargeting import prepare_driver_frames, prepare_reference, retarget_driver_verts
from marionette.utils import save_labeled_grid, save_video_with_audio, slice_cond_rgb


HEAD_VERT_PATH = "data/assets/flame/head_vertices.txt"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",    required=True,
                   help="Any eval config — only base + val_dataset.params are read.")
    p.add_argument("--n_samples", type=int, default=3,
                   help="Samples per mode (cross-identity and same-identity).")
    p.add_argument("--seed",      type=int, default=None,
                   help="Override the config's seed. Does not affect the full-run reproducibility.")
    p.add_argument("--output_dir", default="outputs/marionette_eval/probe")
    p.add_argument("--device",    default="cuda")
    return p.parse_args()


def _load_fit(path: Path) -> dict:
    return {k: v for k, v in np.load(str(path)).items()}


@torch.no_grad()
def probe_one(
    ref_clip: str,
    driver_clip: str,
    ref_frame_idx: int,
    driver_start_idx: int,
    n_frames: int,
    resolution: int,
    cond_module: SpatialConditioning,
    flame_skinner: CAP4DFlameSkinner,
    head_vert_ids: np.ndarray,
    flame_root: Path,
    video_root: Path,
    device: torch.device,
    out_dir: Path,
    title: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_fit = _load_fit(flame_root / ref_clip / "fit.npz")
    drv_fit = _load_fit(flame_root / driver_clip / "fit.npz")

    ref_img_norm, _, crop_box = prepare_reference(
        ref_fit, ref_frame_idx, video_root / f"{ref_clip}.mp4",
        resolution, flame_skinner, head_vert_ids,
    )
    verts_np, offsets_np = retarget_driver_verts(
        ref_fit, drv_fit, crop_box, n_frames, flame_skinner,
        driver_start=driver_start_idx,
    )

    hint = {
        "driver_verts":  torch.from_numpy(verts_np).unsqueeze(0).to(device),
        "driver_deform": torch.from_numpy(offsets_np).unsqueeze(0).to(device),
    }
    c_cond = cond_module(hint)
    spatial_cond = c_cond["spatial_cond"][0]       # (T, H, W, 45)

    ref_rgb_u8 = ((ref_img_norm + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    ref_row = np.broadcast_to(
        ref_rgb_u8.transpose(2, 0, 1)[None],
        (n_frames, 3, resolution, resolution),
    ).copy()

    driver_frames = prepare_driver_frames(
        drv_fit, video_root / f"{driver_clip}.mp4",
        n_frames, resolution, flame_skinner, head_vert_ids,
        driver_start=driver_start_idx,
    )
    driver_row = driver_frames.transpose(0, 3, 1, 2).copy()

    expr_row   = slice_cond_rgb(spatial_cond, 42, resolution)      # deform channels
    posenc_row = slice_cond_rgb(spatial_cond,  0, resolution)      # first pos_enc triple

    rows = [ref_row, driver_row, expr_row, posenc_row]
    labels = ["Reference", "Driver Video", "Driver Expr (deform)", "Pos Enc [0:3]"]

    save_labeled_grid(rows, labels, out_dir / "panel.png", title=title)
    save_video_with_audio(
        rows, labels, audio_np=None,
        path=out_dir / "panel.mp4", fps=25.0, title=title,
    )


def main():
    args = parse_args()
    cfg  = load_experiment_config(args.config)
    seed = args.seed if args.seed is not None else int(cfg.seed)
    n_frames   = int(cfg.inference.n_frames)
    resolution = int(cfg.val_dataset.params.resolution)
    device = torch.device(args.device)

    torch.manual_seed(seed)
    np.random.seed(seed)

    flame_root = Path(cfg.val_dataset.params.flame_root)
    video_root = Path(cfg.val_dataset.params.video_root)
    with open(cfg.val_dataset.params.clip_list_path) as f:
        val_clips = json.load(f)

    # Reuse the eval pairing so what we probe matches what the full runs will see.
    cross, cross_stats = build_cross_identity_samples(
        val_clips, flame_root, n_frames, seed=seed,
    )
    same, same_stats = build_same_identity_samples(
        val_clips, flame_root, n_frames=n_frames,
        samples_per_identity=2, min_ref_driver_gap=n_frames, seed=seed,
    )

    # Conditioning runs on GPU (pytorch3d rasterizer), no model weights needed.
    cond_module = SpatialConditioning(
        **OmegaConf.to_container(cfg.model.params.cond_stage_config.params),
    ).to(device).eval()
    flame_skinner = CAP4DFlameSkinner(add_mouth=True, n_shape_params=150, n_expr_params=65)
    head_vert_ids = np.genfromtxt(HEAD_VERT_PATH).astype(int)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"run_{timestamp}"
    (run_dir / "cross").mkdir(parents=True, exist_ok=True)
    (run_dir / "same").mkdir(parents=True, exist_ok=True)

    print(f"[probe] run dir: {run_dir}")
    print(f"  n_frames={n_frames}  resolution={resolution}  seed={seed}")
    print(f"  cross stats: {cross_stats}")
    print(f"  same  stats: {same_stats}")
    print(f"  probing {args.n_samples} from each mode")

    for i, s in enumerate(cross[: args.n_samples]):
        tag = f"{i:02d}_ref-{s.ref_identity}__drv-{s.driver_identity}"
        title = (
            f"[Cross probe {i+1}/{args.n_samples}] "
            f"ref={s.ref_clip[:24]} (f{s.ref_frame_idx}) → "
            f"drv={s.driver_clip[:24]} (f{s.driver_start_idx}:{s.driver_start_idx + n_frames})"
        )
        print(f"[cross {i+1}/{args.n_samples}] {tag}")
        probe_one(
            ref_clip=s.ref_clip, driver_clip=s.driver_clip,
            ref_frame_idx=s.ref_frame_idx, driver_start_idx=s.driver_start_idx,
            n_frames=n_frames, resolution=resolution,
            cond_module=cond_module, flame_skinner=flame_skinner,
            head_vert_ids=head_vert_ids,
            flame_root=flame_root, video_root=video_root, device=device,
            out_dir=run_dir / "cross" / tag, title=title,
        )

    for i, s in enumerate(same[: args.n_samples]):
        tag = f"{i:02d}_{s.identity}"
        title = (
            f"[Same probe {i+1}/{args.n_samples}] "
            f"clip={s.clip[:24]} | ref=f{s.ref_frame_idx} "
            f"target=[{s.driver_start_idx}:{s.driver_start_idx + n_frames})"
        )
        print(f"[same {i+1}/{args.n_samples}] {tag}")
        probe_one(
            ref_clip=s.clip, driver_clip=s.clip,
            ref_frame_idx=s.ref_frame_idx, driver_start_idx=s.driver_start_idx,
            n_frames=n_frames, resolution=resolution,
            cond_module=cond_module, flame_skinner=flame_skinner,
            head_vert_ids=head_vert_ids,
            flame_root=flame_root, video_root=video_root, device=device,
            out_dir=run_dir / "same" / tag, title=title,
        )

    print(f"[probe] done → {run_dir}")


if __name__ == "__main__":
    main()
