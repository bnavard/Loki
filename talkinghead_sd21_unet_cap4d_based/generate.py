"""
Inference script: generate a talking-head video given a reference image and
a driving source (expression transfer), similar to CAP4D's inference pipeline.

Two FLAME sources (matching CAP4D's ReferenceDataset / GenerationDataset split):
  --ref_data   : path to the *reference subject's* directory (contains fit.npz +
                 images/).  Identity (shape), camera params, and the reference
                 photo come from here.
  --driving_fit: path to the *driving* fit.npz.  Only expression (expr) and
                 eye rotation (eye_rot) are taken from this file — everything
                 else is inherited from the reference subject.

Usage:
    python talkinghead_sd21_unet_cap4d_based/generate.py \
        --checkpoint  outputs/talkinghead/th-100000.ckpt \
        --config      talkinghead_sd21_unet_cap4d_based/configs/talking_head.yaml \
        --ref_data    /path/to/subject_dir/ \
        --ref_frame   0 \
        --driving_fit /path/to/driving/fit.npz \
        --audio       /path/to/audio.wav \
        --output_dir  outputs/generated/ \
        [--n_frames 64] [--fps 25] [--cfg_scale 2.0] [--seed 42]

Pipeline:
  1. Load reference subject's fit.npz → extract identity shape + camera params.
  2. Load reference image from subject dir → crop face → encode to latent z_ref.
  3. Load driving fit.npz → take only expr + eye_rot per frame.
  4. Combine reference identity with driving expressions to build FLAME conditioning.
  5. Load audio → extract per-frame windows → encode with wav2vec2.
  6. Run THSampler with R=1 (reference) + (n_frames-1) generated slots per pass.
  7. Decode latents → save as video frames / mp4.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import cv2
from omegaconf import OmegaConf

from controlnet.ldm.util import instantiate_from_config
from talkinghead_sd21_unet_cap4d_based.model.th_sampler import THSampler
from talkinghead_sd21_unet_cap4d_based.flame.flame import CAP4DFlameSkinner, compute_flame
from talkinghead_sd21_unet_cap4d_based.data.utils import (
    load_frame, crop_image, rescale_image, get_bbox_from_verts, verts_to_pytorch3d,
)
from talkinghead_sd21_unet_cap4d_based.conditioning.th_conditioning import THConditioning
from talkinghead_sd21_unet_cap4d_based.utils.background import build_background_plate, composite_with_background

try:
    import soundfile as sf
    _HAS_SOUNDFILE = True
except ImportError:
    _HAS_SOUNDFILE = False

HEAD_VERT_PATH = "data/assets/flame/head_vertices.txt"
SAMPLE_RATE    = 16_000


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",   required=True)
    p.add_argument("--config",       required=True)
    # Reference subject (identity source)
    p.add_argument("--ref_data",     required=True,
                   help="Subject directory containing fit.npz and images/")
    p.add_argument("--ref_frame",    type=int, default=0,
                   help="Timestep index of the reference frame in the subject video")
    p.add_argument("--ref_cam",      type=int, default=0,
                   help="Camera index for the reference frame (default 0)")
    # Driving source (expression transfer)
    p.add_argument("--driving_fit",  required=True,
                   help="fit.npz from driving video (only expr + eye_rot are used)")
    # Audio
    p.add_argument("--audio",        required=True)
    # Generation params
    p.add_argument("--output_dir",   default="outputs/generated")
    p.add_argument("--n_frames",     type=int, default=64)
    p.add_argument("--fps",          type=float, default=25.0)
    p.add_argument("--cfg_scale",    type=float, default=2.0)
    p.add_argument("--n_ddim_steps", type=int, default=50)
    p.add_argument("--device",       default="cuda")
    # Background stabilization
    p.add_argument("--bg_mask_dir",  default=None,
                   help="Directory with per-frame fg masks (e.g. data/flowface/{clip}/bg/cam0). "
                        "If provided, builds a clean background plate and composites.")
    p.add_argument("--feather_radius", type=int, default=5,
                   help="Gaussian blur radius for soft mask edges (0 = hard)")
    return p.parse_args()


@torch.no_grad()
def main():
    args   = parse_args()
    device = torch.device(args.device)
    cfg    = OmegaConf.load(args.config)

    # Set all seeds for reproducibility
    seed = cfg.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Load model ----
    model = instantiate_from_config(cfg.model)
    ckpt  = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
    model.eval().to(device)

    resolution  = cfg.dataset.params.dataset.params.resolution
    latent_res  = resolution // 8
    n_frames    = args.n_frames
    V           = cfg.inference.get("n_frames", 16)  # UNet window size
    R           = 1

    # ---- FLAME skinner + conditioning module ----
    flame_skinner = CAP4DFlameSkinner(add_mouth=True, n_shape_params=150, n_expr_params=65)
    head_vert_ids = np.genfromtxt(HEAD_VERT_PATH).astype(int)
    cond_module   = THConditioning(**OmegaConf.to_container(cfg.model.params.cond_stage_config.params))
    cond_module.to(device)

    # ---- Load reference subject's FLAME fit (identity source) ----
    ref_data_dir = Path(args.ref_data)
    ref_fit = dict(np.load(ref_data_dir / "fit.npz"))

    # ---- Load driving FLAME fit (expression source) ----
    drv_fit = dict(np.load(args.driving_fit))

    # ---- Build reference FLAME params (identity + ref frame expression) ----
    ref_tid = args.ref_frame
    ref_cid = args.ref_cam
    ref_flame_item = {}
    for key in ref_fit:
        if key in ["expr", "rot", "tra", "eye_rot"]:
            ref_flame_item[key] = ref_fit[key][[ref_tid]]
        elif key in ["fx", "fy", "cx", "cy", "extr", "resolutions"]:
            ref_flame_item[key] = ref_fit[key][[ref_cid]]
        elif key == "shape":
            ref_flame_item[key] = ref_fit[key]

    flame_out      = compute_flame(flame_skinner, ref_flame_item)
    verts_2d_ref   = flame_out["verts_2d"][0, 0]
    offsets_3d_ref = flame_out["offsets_3d"][0]
    crop_box       = get_bbox_from_verts(verts_2d_ref.copy(), head_vert_ids)
    ref_extr       = ref_flame_item["extr"][0]

    # ---- Load reference image ----
    cam_dir = ref_fit["camera_order"][ref_cid] if "camera_order" in ref_fit else "cam0"
    img_dir = ref_data_dir / "images" / cam_dir
    ref_img = load_frame(img_dir, ref_tid)
    ref_img_crop = crop_image(ref_img.astype(np.float32), crop_box, bg_value=255)
    ref_img_crop = rescale_image(ref_img_crop.astype(np.uint8), resolution).astype(np.float32)
    ref_img_norm = ((ref_img_crop / 127.5) - 1.0)

    # Encode reference image to latent
    ref_tensor = torch.tensor(ref_img_norm)[None, None].permute(0, 1, 4, 2, 3).to(device)  # (1,1,3,H,W)
    ref_z = model.get_first_stage_encoding(model.encode_first_stage(ref_tensor.reshape(-1, 3, resolution, resolution)))
    ref_z = ref_z.reshape(1, 1, 4, latent_res, latent_res)

    # ---- Load audio ----
    samples_per_frame    = int(SAMPLE_RATE / args.fps)
    audio_context_frames = cfg.dataset.params.dataset.params.get("audio_context_frames", 2)
    audio_window_samples = samples_per_frame * (1 + 2 * audio_context_frames)

    if _HAS_SOUNDFILE and Path(args.audio).exists():
        audio_full, _ = sf.read(args.audio, dtype="float32", always_2d=False)
        if audio_full.ndim > 1:
            audio_full = audio_full[:, 0]
    else:
        print("WARNING: audio file not found or soundfile not installed — using silence.")
        audio_full = np.zeros(n_frames * samples_per_frame, dtype=np.float32)

    def get_audio_window(frame_id):
        ctx = audio_context_frames
        window = np.zeros(audio_window_samples, dtype=np.float32)
        for i, f in enumerate(range(frame_id - ctx, frame_id + ctx + 1)):
            f_c  = max(0, min(f, n_frames - 1))
            s, e = f_c * samples_per_frame, (f_c + 1) * samples_per_frame
            window[i * samples_per_frame:(i + 1) * samples_per_frame] = audio_full[s:e] if e <= len(audio_full) else 0.
        return window

    # ---- Build per-frame conditioning ----
    # Combine: reference identity (shape, camera) + driving expressions (expr, eye_rot)
    # Rotation and translation come from the reference subject (the driving source
    # only contributes facial expressions, not head pose — matching CAP4D's
    # GenerationDataset which keeps ref_rot / ref_tra from the reference).
    print(f"Building conditioning for {n_frames} frames…")
    n_drv = drv_fit["expr"].shape[0]
    all_verts, all_offsets = [], []
    for t in range(n_frames):
        t_drv = min(t, n_drv - 1)
        fi = {
            "shape":   ref_fit["shape"],
            "expr":    drv_fit["expr"][[t_drv]],
            "eye_rot": drv_fit["eye_rot"][[t_drv]],
            "rot":     ref_flame_item["rot"],      # ref subject's head pose
            "tra":     ref_flame_item["tra"],       # ref subject's translation
            "fx":      ref_flame_item["fx"],
            "fy":      ref_flame_item["fy"],
            "cx":      ref_flame_item["cx"],
            "cy":      ref_flame_item["cy"],
            "extr":    ref_flame_item["extr"],
        }
        fo = compute_flame(flame_skinner, fi)
        v  = verts_to_pytorch3d(fo["verts_2d"][0, 0].copy(), np.array(crop_box))
        all_verts.append(v)
        all_offsets.append(fo["offsets_3d"][0])

    verts_t   = torch.tensor(np.stack(all_verts,   axis=0), dtype=torch.float32)    # (T, V, 2)
    offsets_t = torch.tensor(np.stack(all_offsets, axis=0), dtype=torch.float32)    # (T, V, 3)

    # Reference mask: frame 0 = reference (1), rest = generated (0)
    ref_mask_np = np.zeros((n_frames, 1, latent_res, latent_res), dtype=np.float32)
    ref_mask_np[0] = 1.0
    ref_mask_t = torch.tensor(ref_mask_np)  # (T, 1, h, w)

    cond_batch = {
        "verts_2d":       verts_t[None].to(device),        # (1, T, V, 2)
        "offsets_3d":     offsets_t[None].to(device),      # (1, T, V, 3)
        "reference_mask": ref_mask_t[None].to(device),     # (1, T, 1, h, w)
        "z":              torch.cat([ref_z, torch.zeros(1, n_frames - 1, 4, latent_res, latent_res, device=device)], dim=1),
    }

    c_cond   = cond_module(cond_batch, unconditional=False)
    c_uncond = cond_module(cond_batch, unconditional=True)

    # Audio context
    audio_windows = np.stack([get_audio_window(t) for t in range(n_frames)], axis=0)  # (T, W)
    audio_tensor  = torch.tensor(audio_windows, dtype=torch.float32)[None].to(device)  # (1, T, W)
    audio_ctx     = model.audio_encoder(audio_tensor) if model.audio_encoder is not None else None  # (1,T,S,D)
    audio_ctx_zero = torch.zeros_like(audio_ctx) if audio_ctx is not None else None

    c_cond["audio_context"]   = audio_ctx
    c_uncond["audio_context"] = audio_ctx_zero

    # ---- Separate ref and gen conditioning ----
    def slice_time(d, indices):
        return {k: v[:, indices] for k, v in d.items() if v is not None}

    ref_cond   = slice_time(c_cond,   [0])
    ref_uncond = slice_time(c_uncond, [0])
    gen_cond   = slice_time(c_cond,   list(range(1, n_frames)))
    gen_uncond = slice_time(c_uncond, list(range(1, n_frames)))

    # Move to (N, ...) shape expected by sampler (N = n_frames for each split)
    def reshape_for_sampler(d, squeeze_batch=True):
        """(1, T, ...) → (T, 1, ...)  — sampler indexes along dim 0 = frame index"""
        return {k: v.squeeze(0) if squeeze_batch else v for k, v in d.items() if v is not None}

    ref_cond   = reshape_for_sampler(ref_cond)
    ref_uncond = reshape_for_sampler(ref_uncond)
    gen_cond   = reshape_for_sampler(gen_cond)
    gen_uncond = reshape_for_sampler(gen_uncond)

    # ---- Run sampler ----
    sampler = THSampler(model)
    latents = sampler.sample(
        S=args.n_ddim_steps,
        ref_cond=ref_cond,
        ref_uncond=ref_uncond,
        gen_cond=gen_cond,
        gen_uncond=gen_uncond,
        latent_shape=(4, latent_res, latent_res),
        V=V,
        R=R,
        cfg_scale=args.cfg_scale,
    )  # (n_frames-1, 4, h, w)

    # Prepend reference latent
    all_latents = torch.cat([ref_z.squeeze(0), latents], dim=0)  # (n_frames, 4, h, w)

    # ---- Decode ----
    print("Decoding latents…")
    imgs = model.decode_first_stage(all_latents[None]).squeeze(0)  # (n_frames, 3, H, W)
    imgs = ((imgs.clamp(-1, 1) + 1) / 2 * 255).byte().cpu().numpy()

    # ---- Background stabilization ----
    if args.bg_mask_dir is not None:
        print("Building background plate…")
        # Determine how many mask frames are available
        mask_dir = Path(args.bg_mask_dir)
        n_mask_frames = len(list(mask_dir.glob("*.png")))

        bg_plate = build_background_plate(
            video_path=str(ref_data_dir / "images" / cam_dir) if (ref_data_dir / "images" / cam_dir).is_dir()
                       else str(ref_data_dir / "video.mp4"),
            mask_dir=str(mask_dir),
            n_frames=n_mask_frames,
            crop_box=np.array(crop_box),
            resolution=resolution,
        )

        print("Compositing with stable background…")
        imgs = composite_with_background(
            generated_frames=imgs,
            background_plate=bg_plate,
            mask_dir=str(mask_dir),
            frame_indices=list(range(n_frames)),
            crop_box=np.array(crop_box),
            resolution=resolution,
            feather_radius=args.feather_radius,
        )

    # ---- Save frames ----
    frame_dir = Path(args.output_dir) / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(imgs):
        img_bgr = img.transpose(1, 2, 0)[..., [2, 1, 0]]
        cv2.imwrite(str(frame_dir / f"{i:05d}.png"), img_bgr)

    print(f"Saved {len(imgs)} frames to {frame_dir}")


if __name__ == "__main__":
    main()
