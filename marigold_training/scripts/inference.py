"""
Marigold inference with SD3.5: natural face video → deformation map video.

Processes a video frame-by-frame through the SD3.5-based Marigold model.
Each frame is independently denoised conditioned on the corresponding
natural face frame.

Usage:
    cd <repo_root>

    PYTHONPATH=. python marigold_training/scripts/inference.py \
        --clip_id CLIP_ID \
        --checkpoint outputs/marigold_spatial/run_YYYYMMDD/step_NNNNNN \
        --prompt "A person says: '...' The delivery is calm and measured."
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
torch.backends.cudnn.enabled = False

from marigold_training.src.marigold_model import double_input_channels
from marigold_training.src.vis import visualize_deform, normalize_to_uint8, save_video


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--clip_id", required=True, help="Clip ID for natural video")
    p.add_argument("--prompt", default=None, help="Text prompt (single)")
    p.add_argument("--prompts", default=None, help="JSON prompt file (batch)")
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint step dir")
    p.add_argument("--model_id", default="stabilityai/stable-diffusion-3.5-medium")
    p.add_argument("--flame_root", default="data/flowface")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--num_frames", type=int, default=None, help="Frames to process (default: all)")
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps", type=int, default=25)
    return p.parse_args()


def main():
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint)
    device = "cuda"
    H = args.resolution

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # ---- Load prompts ----
    if args.prompts:
        with open(args.prompts) as f:
            prompts = json.load(f)
    elif args.prompt:
        prompts = [{"id": "single", "prompt": args.prompt}]
    else:
        prompts = [{"id": "default", "prompt": ""}]

    # ---- Load SD3.5 via full pipeline to encode null prompt ----
    from diffusers import StableDiffusion3Pipeline

    print(f"Loading {args.model_id}...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16,
    )

    transformer = pipe.transformer
    vae = pipe.vae.to(device, dtype=torch.float32).eval()
    vae.requires_grad_(False)

    scaling_factor = vae.config.scaling_factor
    shift_factor = vae.config.shift_factor
    pooled_dim = transformer.config["pooled_projection_dim"]

    # Encode null prompt once — move text encoders to device first
    print("Encoding null prompt...")
    pipe.text_encoder.to(device)
    pipe.text_encoder_2.to(device)
    pipe.text_encoder_3.to(device)
    with torch.no_grad():
        null_out = pipe.encode_prompt(prompt="", prompt_2="", prompt_3="", device=device)
        null_text_embeds = null_out[0].to(dtype=torch.bfloat16)
        null_pooled_embeds = null_out[2].to(dtype=torch.bfloat16)

    del pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3
    del pipe.tokenizer, pipe.tokenizer_2, pipe.tokenizer_3
    del pipe
    torch.cuda.empty_cache()

    # Double input + load checkpoint
    transformer = double_input_channels(transformer)

    full_path = checkpoint_dir / "transformer"
    if full_path.exists():
        from safetensors.torch import load_file
        sf = full_path / "diffusion_pytorch_model.safetensors"
        if not sf.exists():
            sf = full_path / "model.safetensors"
        print(f"Loading checkpoint: {sf}")
        transformer.load_state_dict(load_file(str(sf)), strict=True)
    else:
        print(f"WARNING: No checkpoint at {full_path}")

    transformer = transformer.to(device, dtype=torch.bfloat16).eval()

    # ---- Prepare FLAME + conditioning ----
    from marionette.conditioning.th_conditioning import THConditioning
    from marionette.flame.flame import CAP4DFlameSkinner, compute_flame
    from marionette.data.utils import (
        get_bbox_from_verts, verts_to_pytorch3d, crop_image, rescale_image,
    )

    cond = THConditioning(
        image_size=H, positional_channels=42, positional_multiplier=1.0,
        super_resolution=1,
        use_expr_deformation=True, use_crop_mask=False,
    ).eval().to(device)
    flame_skinner = CAP4DFlameSkinner(add_mouth=True, n_shape_params=150, n_expr_params=65)
    head_vids = np.genfromtxt("data/assets/flame/head_vertices.txt").astype(int)

    flame_root = Path(args.flame_root)
    fit = dict(np.load(str(flame_root / args.clip_id / "fit.npz")))
    total_frames = fit["expr"].shape[0]
    n_frames = args.num_frames if args.num_frames else total_frames
    n_frames = min(n_frames, total_frames)
    print(f"Clip {args.clip_id}: {total_frames} total frames, processing {n_frames}")

    # Null text conditioning (unconditional, per original Marigold)
    text_embeds = null_text_embeds
    pooled = null_pooled_embeds

    # ---- Output ----
    run_output_dir = checkpoint_dir / "inference" / args.clip_id
    run_output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Process frame by frame ----
    sbs_frames = []
    pred_frames = []
    gt_frames = []

    for t in range(n_frames):
        if t % 10 == 0:
            print(f"  Frame {t}/{n_frames}")

        # Compute GT deform + crop box
        flame_item = {
            "shape": fit["shape"], "expr": fit["expr"][[t]], "rot": fit["rot"][[t]],
            "tra": fit["tra"][[t]], "eye_rot": fit["eye_rot"][[t]],
            "fx": fit["fx"][[0]], "fy": fit["fy"][[0]],
            "cx": fit["cx"][[0]], "cy": fit["cy"][[0]], "extr": fit["extr"][[0]],
        }
        if "jaw_rot" in fit:
            flame_item["jaw_rot"] = fit["jaw_rot"][[t]]

        flame_out = compute_flame(flame_skinner, flame_item)
        v2d = flame_out["verts_2d"][0, 0]
        off = flame_out["offsets_3d"][0]
        crop_box = get_bbox_from_verts(v2d.copy(), head_vids)
        v2d_p = verts_to_pytorch3d(v2d.copy(), np.array(crop_box))

        with torch.no_grad():
            out = cond({
                "verts_2d": torch.tensor(v2d_p).unsqueeze(0).unsqueeze(0).to(device),
                "offsets_3d": torch.tensor(off).unsqueeze(0).unsqueeze(0).to(device),
                "reference_mask": torch.zeros(1, 1, 1, H, H, device=device),
            }, unconditional=False)
        gt_deform = out["pos_enc"][0, 0, :, :, 42:45].permute(2, 0, 1).cpu()

        # Load natural frame
        img = cv2.imread(str(flame_root / args.clip_id / "images" / "cam0" / f"{t:05d}.jpg"))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = crop_image(img, crop_box, bg_value=0)
            img = rescale_image(img, H)
        else:
            img = np.zeros((H, H, 3), dtype=np.uint8)
        nat_tensor = torch.from_numpy(img.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)

        # VAE encode natural frame
        with torch.no_grad():
            nat_latent = vae.encode(
                nat_tensor.unsqueeze(0).to(device, dtype=vae.dtype)
            ).latent_dist.mode()
            nat_latent = ((nat_latent - shift_factor) * scaling_factor).to(dtype=torch.bfloat16)

        # Euler denoising
        torch.manual_seed(args.seed + t)
        x = torch.randn_like(nat_latent)
        timesteps = torch.linspace(1.0, 0.0, args.num_steps + 1, device=device)

        use_cfg = args.guidance_scale > 1.0

        with torch.no_grad():
            for i in range(args.num_steps):
                dt = timesteps[i + 1] - timesteps[i]
                model_input = torch.cat([x, nat_latent], dim=1)
                t_ms = (timesteps[i] * 1000).long().expand(1)

                if use_cfg:
                    vel_cond = transformer(
                        hidden_states=model_input, timestep=t_ms,
                        encoder_hidden_states=text_embeds,
                        pooled_projections=pooled,
                    ).sample
                    vel_uncond = transformer(
                        hidden_states=model_input, timestep=t_ms,
                        encoder_hidden_states=torch.zeros_like(text_embeds),
                        pooled_projections=torch.zeros_like(pooled),
                    ).sample
                    velocity = vel_uncond + args.guidance_scale * (vel_cond - vel_uncond)
                else:
                    velocity = transformer(
                        hidden_states=model_input, timestep=t_ms,
                        encoder_hidden_states=text_embeds,
                        pooled_projections=pooled,
                    ).sample
                x = x + velocity * dt

        # Decode
        raw_latent = x.float() / scaling_factor + shift_factor
        with torch.no_grad():
            decoded = vae.decode(raw_latent.to(vae.dtype), return_dict=False)[0]
        pred_deform = decoded.squeeze(0).float().cpu()  # [3, H, W]

        # Collect frames for video
        nat_vis = img[..., ::-1]  # RGB→BGR
        pred_vis = normalize_to_uint8(pred_deform.numpy().transpose(1, 2, 0))[..., ::-1]
        gt_vis = normalize_to_uint8(gt_deform.numpy().transpose(1, 2, 0))[..., ::-1]

        sbs_frames.append(np.concatenate([nat_vis, pred_vis, gt_vis], axis=1))
        pred_frames.append(pred_vis)
        gt_frames.append(gt_vis)

    # Save videos
    save_video(sbs_frames, run_output_dir / "side_by_side.mp4", args.fps)
    save_video(pred_frames, run_output_dir / "predicted.mp4", args.fps)
    save_video(gt_frames, run_output_dir / "ground_truth.mp4", args.fps)

    print(f"\nDone. Saved to {run_output_dir}/")
    print(f"  side_by_side.mp4  (input | predicted | ground truth)")
    print(f"  predicted.mp4")
    print(f"  ground_truth.mp4")


if __name__ == "__main__":
    main()
