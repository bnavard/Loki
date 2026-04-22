r"""
Cross-identity talking-head inference.

Given a reference clip (identity source β_ref) and a driver clip (motion
source ψ_driver, θ_driver) — possibly the same clip for same-identity
reconstruction — produces an `n_frames`-long video of the reference identity
performing the driver's expression and head pose, with the driver's audio.

Recipe:
  * `RefFeatureExtractor` runs the frozen SD 2.1 UNet on the VAE-encoded
    reference image once, caching per-layer self-attention inputs.
  * FLAME(β_ref, ψ_driver[t], θ_driver[t], camera=ref) is rasterized into the
    ref's pixel space — `SpatialConditioning` emits a 45-channel `spatial_cond`
    (pos_enc + driver_deform) at every gen slot.
  * DDIM denoising on T pure-noise latents with classifier-free guidance.
    At each step the gen UNet receives both `spatial_cond` (additive to the
    first feature map) and `ref_features` (concatenated into self-attention
    K/V at every layer). The ref never occupies a slot in the output.

Usage:
    python marionette/generate.py \
        --checkpoint  outputs/<run>/<ckpt>.ckpt \
        --config      marionette/configs/base.yaml \
        --ref_clip    <reference_clip_id> \
        --ref_frame   0 \
        --driver_clip <driver_clip_id> \
        --output_dir  outputs/generated/
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import cv2
from omegaconf import OmegaConf

from ldm_base.ldm.util import instantiate_from_config
from marionette.flame.flame import CAP4DFlameSkinner
from marionette.conditioning.conditioning import SpatialConditioning
from marionette.retargeting import (
    prepare_reference, retarget_driver_verts, prepare_driver_frames,
)
from marionette.utils import (
    SAMPLE_RATE, load_clip_audio_windows,
    slice_cond_rgb, save_labeled_grid, save_video_with_audio,
)


HEAD_VERT_PATH = "data/assets/flame/head_vertices.txt"


def _load_fit(path: Path) -> dict:
    return {k: v for k, v in np.load(str(path)).items()}


def _load_checkpoint_into(model, ckpt_path: str):
    """Load a Lightning checkpoint into a bare `MarionetteDiffusion` — strip
    the outer `model.` prefix and fail loud on any missing or unexpected keys
    so we never silently train on partial weights."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw = ckpt.get("state_dict", ckpt)
    sd = {k[len("model."):]: v for k, v in raw.items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # Ref extractor params are frozen and may legitimately be absent from a
    # Lightning checkpoint if they weren't saved; tolerate that one case.
    unexpected = [k for k in unexpected if not k.startswith("ref_extractor.")]
    missing    = [k for k in missing    if not k.startswith("ref_extractor.")]
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint load incomplete: {len(missing)} missing, "
            f"{len(unexpected)} unexpected. "
            f"First missing: {missing[:3]}. First unexpected: {unexpected[:3]}."
        )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",   required=True)
    p.add_argument("--config",       required=True)
    p.add_argument("--ref_clip",     required=True)
    p.add_argument("--ref_frame",    type=int, default=0)
    p.add_argument("--driver_clip",  required=True)
    p.add_argument("--flame_root",   default="data/flame_tracking/flowface")
    p.add_argument("--video_root",   default="data/talkvid/talkvid")
    p.add_argument("--audio_root",   default="data/talkvid/audio")
    p.add_argument("--output_dir",   default="outputs/generated")
    p.add_argument("--n_frames",     type=int, default=16,
                   help="Number of gen target frames; matches the UNet's time_steps.")
    p.add_argument("--cfg_scale",    type=float, default=2.0)
    p.add_argument("--n_ddim_steps", type=int, default=50)
    p.add_argument("--device",       default="cuda")
    return p.parse_args()


@torch.no_grad()
def main():
    args   = parse_args()
    device = torch.device(args.device)
    cfg    = OmegaConf.load(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    model = instantiate_from_config(cfg.model)
    _load_checkpoint_into(model, args.checkpoint)
    model.eval().to(device)

    ds_params = cfg.train_dataset.params
    resolution = ds_params.resolution
    latent_res = resolution // ds_params.downsample_ratio
    samples_per_frame    = int(SAMPLE_RATE / ds_params.fps)
    audio_context_frames = ds_params.audio_context_frames

    n_frames = args.n_frames

    cond_module = SpatialConditioning(
        **OmegaConf.to_container(cfg.model.params.cond_stage_config.params),
    ).to(device).eval()
    flame_skinner = CAP4DFlameSkinner(add_mouth=True, n_shape_params=150, n_expr_params=65)
    head_vert_ids = np.genfromtxt(HEAD_VERT_PATH).astype(int)

    flame_root = Path(args.flame_root)
    video_root = Path(args.video_root)
    audio_root = Path(args.audio_root)

    ref_fit    = _load_fit(flame_root / args.ref_clip / "fit.npz")
    driver_fit = _load_fit(flame_root / args.driver_clip / "fit.npz")
    is_cross_id = args.ref_clip != args.driver_clip
    print(f"[generate] {'cross-identity' if is_cross_id else 'same-identity'}: "
          f"ref={args.ref_clip} driver={args.driver_clip}")

    ref_img_norm, _, crop_box = prepare_reference(
        ref_fit, args.ref_frame, video_root / f"{args.ref_clip}.mp4",
        resolution, flame_skinner, head_vert_ids,
    )
    verts_np, offsets_np = retarget_driver_verts(
        ref_fit, driver_fit, crop_box, n_frames, flame_skinner,
    )

    hint = {
        "driver_verts":  torch.from_numpy(verts_np).unsqueeze(0).to(device),
        "driver_deform": torch.from_numpy(offsets_np).unsqueeze(0).to(device),
    }

    ref_tensor = torch.from_numpy(ref_img_norm).permute(2, 0, 1).unsqueeze(0).to(device)
    ref_z = model.get_first_stage_encoding(model.encode_first_stage(ref_tensor))

    c_cond = cond_module(hint)
    c_cond["ref_z"] = ref_z

    audio_windows = load_clip_audio_windows(
        audio_root / f"{args.driver_clip}.wav",
        n_frames, samples_per_frame, audio_context_frames,
    )
    audio_t = torch.from_numpy(audio_windows).unsqueeze(0).to(device)
    audio_ctx = model.audio_encoder(audio_t) if model.audio_encoder is not None else None
    c_cond["audio_context"] = audio_ctx

    c_uncond = {
        k: (torch.zeros_like(v) if torch.is_tensor(v) else v)
        for k, v in c_cond.items()
    }

    latents = model.sample_video(
        control=c_cond, control_uncond=c_uncond,
        n_frames=n_frames,
        latent_shape=(4, latent_res, latent_res),
        n_ddim_steps=args.n_ddim_steps,
        cfg_scale=args.cfg_scale,
    )

    imgs = model.decode_first_stage(latents.unsqueeze(0)).squeeze(0)
    imgs = ((imgs.clamp(-1, 1) + 1) / 2 * 255).byte().cpu().numpy()   # (T, 3, H, W)

    out_dir = Path(args.output_dir)
    frame_dir = out_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(imgs):
        bgr = img.transpose(1, 2, 0)[..., [2, 1, 0]]
        cv2.imwrite(str(frame_dir / f"{i:05d}.png"), bgr)

    # 4-row panel: Reference (static) | Driver Video | Driver Expression | Generated
    ref_rgb_u8 = ((ref_img_norm + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    ref_row = np.broadcast_to(
        ref_rgb_u8.transpose(2, 0, 1)[None], (n_frames, 3, resolution, resolution),
    ).copy()

    driver_frames = prepare_driver_frames(
        driver_fit, video_root / f"{args.driver_clip}.mp4",
        n_frames, resolution, flame_skinner, head_vert_ids,
    )
    driver_row = driver_frames.transpose(0, 3, 1, 2).copy()

    spatial_cond_t = c_cond["spatial_cond"][0]                       # (T, H, W, 45)
    expr_row = slice_cond_rgb(spatial_cond_t, 42, resolution)

    rows_data = [ref_row, driver_row, expr_row, imgs]
    labels    = ["Reference", "Driver Video", "Driver Expression", "Generated"]
    title = (f"{'Cross' if is_cross_id else 'Same'}-Identity | "
             f"ref={args.ref_clip[:32]} → drv={args.driver_clip[:32]}")

    save_labeled_grid(rows_data, labels, out_dir / "panel.png", title=title)
    save_video_with_audio(
        rows_data, labels, audio_windows,
        out_dir / "panel.mp4", fps=ds_params.fps, title=title,
    )

    print(f"[generate] saved {len(imgs)} frames to {frame_dir}")
    print(f"[generate] saved side-by-side panel → {out_dir / 'panel.png'}")
    print(f"[generate] saved video w/ driver audio → {out_dir / 'panel.mp4'}")


if __name__ == "__main__":
    main()
