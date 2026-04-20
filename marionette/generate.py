"""
Cross-identity retargeted inference for Marionette.

Given a reference clip (identity source β_ref) and a driver clip (motion
source ψ_driver, θ_driver) — possibly the same clip, for same-identity
reconstruction — this script produces a video of the reference identity
performing the driver's expression and head pose, with the driver's audio.

Recipe:
  FLAME(shape=β_ref, expression=ψ_driver[t], pose=θ_driver[t], camera=ref)
  → verts + offsets projected into ref pixel space.
  → SpatialConditioning: rasterize [pos_enc | driver_deform | warped_ref | ref_mask]
    = 49-channel spatial_cond fed to the UNet's ConditioningEncoder.

Usage:
    python marionette/generate.py \\
        --checkpoint  outputs/<run>/<ckpt>.ckpt \\
        --config      marionette/configs/base.yaml \\
        --ref_clip    <reference_clip_id> \\
        --ref_frame   0 \\
        --driver_clip <driver_clip_id> \\
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
from marionette.model.sampler import SlidingWindowSampler
from marionette.flame.flame import CAP4DFlameSkinner
from marionette.conditioning.conditioning import SpatialConditioning
from marionette.retargeting import prepare_reference, retarget_driver_verts
from marionette.utils import SAMPLE_RATE, load_clip_audio_windows


HEAD_VERT_PATH = "data/assets/flame/head_vertices.txt"


def _load_fit(path: Path) -> dict:
    return {k: v for k, v in np.load(str(path)).items()}


def _round_to_sampler_window(n_frames: int, V: int, R: int) -> int:
    """Sampler needs (n_frames - R) % (V - R) == 0. Round DOWN to closest fit."""
    G = V - R
    n_gen = n_frames - R
    valid = (n_gen // G) * G + R
    return max(valid, V)


def _build_hint(
    verts_np: np.ndarray,         # (T, V, 3)
    offsets_np: np.ndarray,       # (T, V, 3)
    ref_image_np: np.ndarray,     # (H, W, 3) in [-1, 1]
    ref_verts_np: np.ndarray,     # (V, 3)
    n_frames: int,
    latent_res: int,
    device: torch.device,
) -> dict:
    ref_mask = np.zeros((n_frames, 1, latent_res, latent_res), dtype=np.float32)
    ref_mask[0] = 1.0
    ref_image = torch.from_numpy(ref_image_np).permute(2, 0, 1)
    return {
        "driver_verts":  torch.from_numpy(verts_np).unsqueeze(0).to(device),
        "driver_deform": torch.from_numpy(offsets_np).unsqueeze(0).to(device),
        "ref_mask":      torch.from_numpy(ref_mask).unsqueeze(0).to(device),
        "ref_image":     ref_image.unsqueeze(0).to(device),
        "ref_verts":     torch.from_numpy(ref_verts_np.astype(np.float32)).unsqueeze(0).to(device),
    }


def _null_token(cond: dict) -> dict:
    """Zero every tensor in `cond`, leave non-tensors alone. The CFG null token."""
    return {
        k: (torch.zeros_like(v) if torch.is_tensor(v) else v)
        for k, v in cond.items()
    }


def _split_ref_gen(cond: dict, n_frames: int) -> tuple[dict, dict]:
    """Split a (1, T, ...) control dict into (R=1 ref, T-1 generated) sampler inputs."""
    def _slice(d, idx):
        return {
            k: (v[:, idx].squeeze(0) if v is not None else None)
            for k, v in d.items()
        }
    return _slice(cond, [0]), _slice(cond, list(range(1, n_frames)))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",   required=True)
    p.add_argument("--config",       required=True)
    p.add_argument("--ref_clip",     required=True,
                   help="Reference clip ID (identity source).")
    p.add_argument("--ref_frame",    type=int, default=0)
    p.add_argument("--driver_clip",  required=True,
                   help="Driver clip ID (motion + audio source). "
                        "Set equal to --ref_clip for same-identity reconstruction.")
    p.add_argument("--flame_root",   default="data/flame_tracking/flowface")
    p.add_argument("--video_root",   default="data/talkvid/talkvid")
    p.add_argument("--audio_root",   default="data/talkvid/audio")
    p.add_argument("--output_dir",   default="outputs/generated")
    p.add_argument("--n_frames",     type=int, default=16)
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
    ckpt  = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
    model.eval().to(device)

    ds_params = cfg.train_dataset.params
    resolution = ds_params.resolution
    latent_res = resolution // ds_params.downsample_ratio
    samples_per_frame = int(SAMPLE_RATE / ds_params.fps)
    audio_context_frames = ds_params.audio_context_frames

    V = cfg.inference.get("n_frames", cfg.model.params.n_frames)
    R = 1
    n_frames = _round_to_sampler_window(args.n_frames, V, R)
    if n_frames != args.n_frames:
        print(f"[generate] rounded n_frames {args.n_frames} → {n_frames} for V={V}, R={R}")

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

    ref_img_norm, ref_verts_ndc, crop_box = prepare_reference(
        ref_fit, args.ref_frame, video_root / f"{args.ref_clip}.mp4",
        resolution, flame_skinner, head_vert_ids,
    )
    verts_np, offsets_np = retarget_driver_verts(
        ref_fit, driver_fit, crop_box, n_frames, flame_skinner,
    )

    hint = _build_hint(
        verts_np, offsets_np, ref_img_norm, ref_verts_ndc,
        n_frames, latent_res, device,
    )

    ref_tensor = torch.from_numpy(ref_img_norm).permute(2, 0, 1).unsqueeze(0).to(device)
    ref_z = model.get_first_stage_encoding(model.encode_first_stage(ref_tensor))
    hint["z"] = torch.cat(
        [ref_z.unsqueeze(0),
         torch.zeros(1, n_frames - 1, 4, latent_res, latent_res, device=device)],
        dim=1,
    )

    c_cond   = cond_module(hint)
    c_uncond = _null_token(c_cond)

    audio_windows = load_clip_audio_windows(
        audio_root / f"{args.driver_clip}.wav",
        n_frames, samples_per_frame, audio_context_frames,
    )
    audio_t = torch.from_numpy(audio_windows).unsqueeze(0).to(device)
    audio_ctx = model.audio_encoder(audio_t) if model.audio_encoder is not None else None
    c_cond["audio_context"]   = audio_ctx
    c_uncond["audio_context"] = (
        torch.zeros_like(audio_ctx) if audio_ctx is not None else None
    )

    ref_cond,   gen_cond   = _split_ref_gen(c_cond,   n_frames)
    ref_uncond, gen_uncond = _split_ref_gen(c_uncond, n_frames)

    sampler = SlidingWindowSampler(model)
    latents = sampler.sample(
        S=args.n_ddim_steps,
        ref_cond=ref_cond, ref_uncond=ref_uncond,
        gen_cond=gen_cond, gen_uncond=gen_uncond,
        latent_shape=(4, latent_res, latent_res),
        V=V, R=R, cfg_scale=args.cfg_scale,
    )

    all_latents = torch.cat([ref_z, latents], dim=0).unsqueeze(0)
    imgs = model.decode_first_stage(all_latents).squeeze(0)
    imgs = ((imgs.clamp(-1, 1) + 1) / 2 * 255).byte().cpu().numpy()

    frame_dir = Path(args.output_dir) / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(imgs):
        bgr = img.transpose(1, 2, 0)[..., [2, 1, 0]]
        cv2.imwrite(str(frame_dir / f"{i:05d}.png"), bgr)
    print(f"[generate] saved {len(imgs)} frames to {frame_dir}")


if __name__ == "__main__":
    main()
