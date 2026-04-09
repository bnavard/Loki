"""
Marigold-style inference: natural video + text → deformation map video.

Given a natural talking-head video and a text prompt, generates the
corresponding deformation map video. The natural video provides
spatiotemporal anchoring at every denoising step via channel concatenation.

Uses Euler ODE integration (flow matching: t=1 noise → t=0 data).
At each step, concatenates [current_noisy_deform | clean_natural_video]
along channel dim and passes to the modified DiT (32ch input).

Supports multi-GPU parallelism via torchrun.

Usage:
    cd /data/pouyan/baseline/repository/cap4d

    PYTHONPATH=. python text_to_expr_field/scripts/inference_marigold.py \
        --clip_id CLIP_ID \
        --checkpoint outputs/marigold_deform/run_YYYYMMDD/step_NNNNNN \
        --prompt "A person says: '...' The delivery is calm and measured."

    # Or batch from a JSON prompt file:
    PYTHONPATH=. python text_to_expr_field/scripts/inference_marigold.py \
        --prompts text_to_expr_field/configs/eval_prompts_single.json \
        --clip_id CLIP_ID \
        --checkpoint outputs/marigold_deform/run_YYYYMMDD/step_NNNNNN
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
torch.backends.cudnn.enabled = False

from text_to_expr_field.src.model.marigold import double_patch_embedding
from text_to_expr_field.src.vis import visualize_deform


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--clip_id", required=True, help="Clip ID for natural video")
    p.add_argument("--prompt", default=None, help="Text prompt (single)")
    p.add_argument("--prompts", default=None, help="JSON prompt file (batch)")
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint directory")
    p.add_argument("--model_id", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    p.add_argument("--flame_root", default="data/flowface")
    p.add_argument("--target_frames", type=int, default=81)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps", type=int, default=25)
    return p.parse_args()


def encode_video(vae, video_tensor, latents_mean, latents_std):
    """
    VAE-encode a video tensor and normalize with pretrained stats.

    Args:
        vae:           frozen VAE model
        video_tensor:  [B, 3, T, H, W] or [3, T, H, W]
        latents_mean:  [1, C, 1, 1, 1]
        latents_std:   [1, C, 1, 1, 1]

    Returns:
        normalized latent [B, C, T_lat, h, w]
    """
    if video_tensor.ndim == 4:
        video_tensor = video_tensor.unsqueeze(0)

    with torch.no_grad():
        latent = vae.encode(
            video_tensor.to(device=vae.device, dtype=vae.dtype)
        ).latent_dist.mode()
        latent = (latent - latents_mean.to(latent.device, latent.dtype)) / \
                 latents_std.to(latent.device, latent.dtype)
    return latent


def compute_deformation_gt(clip_id, flame_root, target_frames, resolution):
    """
    Compute ground-truth deformation maps from FLAME fits.

    Returns:
        deform_gt:  [T, 3, H, W] float32
        crop_boxes: list of per-frame crop boxes (for use by load_natural_video)
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
    ).eval().cuda()

    flame_skinner = CAP4DFlameSkinner(
        add_mouth=True, n_shape_params=150, n_expr_params=65,
    )
    head_vertex_ids = np.genfromtxt("data/assets/flame/head_vertices.txt").astype(int)
    fit = dict(np.load(str(flame_root / clip_id / "fit.npz")))

    deform_frames = []
    crop_boxes = []

    for t in range(target_frames):
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
        crop_boxes.append(crop_box)

        verts_2d_p3d = verts_to_pytorch3d(verts_2d.copy(), np.array(crop_box))
        dummy_ref = torch.zeros(1, 1, 1, H, H, device="cuda")
        batch = {
            "verts_2d": torch.tensor(verts_2d_p3d).unsqueeze(0).unsqueeze(0).cuda(),
            "offsets_3d": torch.tensor(offsets_3d).unsqueeze(0).unsqueeze(0).cuda(),
            "reference_mask": dummy_ref,
        }
        with torch.no_grad():
            out = conditioning(batch, unconditional=False)
        deform_frames.append(out["pos_enc"][0, 0, :, :, 42:45].permute(2, 0, 1).cpu())

    deform_gt = torch.stack(deform_frames, dim=0)  # [T, 3, H, W]
    return deform_gt, crop_boxes


def load_natural_video(clip_id, flame_root, crop_boxes, target_frames, resolution):
    """
    Load and crop natural video frames using precomputed crop boxes.

    Returns:
        natural_video: [3, T_padded, H, W] float32 in [-1, 1]
    """
    from text_to_expr_field.src.utils.reshape import to_pseudo_video
    from talkinghead_sd21_unet_cap4d_based.data.utils import crop_image, rescale_image

    frames_dir = Path(flame_root) / clip_id / "images" / "cam0"
    H = resolution

    natural_frames = []
    for t in range(target_frames):
        img_path = frames_dir / f"{t:05d}.jpg"
        img = cv2.imread(str(img_path))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = crop_image(img, crop_boxes[t], bg_value=0)
            img = rescale_image(img, H)
            img = img.astype(np.float32) / 127.5 - 1.0
        else:
            img = np.zeros((H, H, 3), dtype=np.float32)
        natural_frames.append(torch.from_numpy(img).permute(2, 0, 1))

    natural_video = torch.stack(natural_frames, dim=1)  # [3, T, H, W]

    # Pad to 4k+1 for VAE
    T_padded = to_pseudo_video(torch.zeros(target_frames, 3, H, H)).shape[0]
    T_nat = natural_video.shape[1]
    if T_nat < T_padded:
        pad = natural_video[:, -1:].repeat(1, T_padded - T_nat, 1, 1)
        natural_video = torch.cat([natural_video, pad], dim=1)

    return natural_video


def main():
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint)
    device = torch.device("cuda")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # ---- Load prompts ----
    if args.prompts:
        with open(args.prompts) as f:
            prompts = json.load(f)
    elif args.prompt:
        prompts = [{"id": "single", "prompt": args.prompt}]
    else:
        raise ValueError("Provide --prompt or --prompts")

    # ---- Load pipeline ----
    from diffusers import WanPipeline
    print(f"Loading {args.model_id}...")
    pipe = WanPipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16)

    vae = pipe.vae.to(device).eval()
    vae.requires_grad_(False)
    transformer = pipe.transformer

    # ---- Encode all prompts with the text encoder before freeing it ----
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder.to(device).eval()

    prompt_embeddings = {}
    for entry in prompts:
        pid = entry["id"]
        text_cache = Path("data/derived/prompt_latent_cache") / f"{args.clip_id}.pt"
        if text_cache.exists():
            text_data = torch.load(str(text_cache), map_location="cpu", weights_only=True)
            prompt_embeddings[pid] = text_data["text_embed"].unsqueeze(0).to(device, dtype=torch.bfloat16)
            print(f"  [{pid}] Loaded cached text embedding")
        else:
            text_inputs = tokenizer(
                [entry["prompt"]],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                text_embeds = text_encoder(**text_inputs)[0]  # (1, seq_len, dim)
            prompt_embeddings[pid] = text_embeds.to(dtype=torch.bfloat16)
            print(f"  [{pid}] Encoded text with text encoder (shape {text_embeds.shape})")

    del text_encoder, tokenizer
    torch.cuda.empty_cache()

    # Double input layer (same modification as training)
    transformer = double_patch_embedding(transformer)

    # Load fine-tuned weights
    full_path = checkpoint_dir / "transformer"
    if full_path.exists():
        from safetensors.torch import load_file
        safetensor_file = full_path / "diffusion_pytorch_model.safetensors"
        if not safetensor_file.exists():
            safetensor_file = full_path / "model.safetensors"
        print(f"Loading checkpoint: {safetensor_file}")
        state_dict = load_file(str(safetensor_file))
        transformer.load_state_dict(state_dict, strict=True)
    else:
        print(f"WARNING: No checkpoint at {full_path}")

    transformer = transformer.to(device, dtype=torch.bfloat16).eval()

    latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
    latents_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)

    del pipe
    torch.cuda.empty_cache()

    # ---- Prepare natural video + ground-truth deformation ----
    print(f"Computing deformation for clip {args.clip_id}...")
    deform_gt, crop_boxes = compute_deformation_gt(
        args.clip_id, args.flame_root, args.target_frames, args.resolution,
    )
    print(f"Loading natural video for clip {args.clip_id}...")
    natural_video = load_natural_video(
        args.clip_id, args.flame_root, crop_boxes, args.target_frames, args.resolution,
    )
    # natural_video: [3, T_padded, H, W]

    # VAE-encode natural video (conditioning — always clean)
    natural_latent = encode_video(vae, natural_video, latents_mean, latents_std)
    # [1, 16, T_lat, h, w]
    natural_latent = natural_latent.to(dtype=torch.bfloat16)
    print(f"Natural video latent: {natural_latent.shape}")

    # ---- Output dir ----
    run_output_dir = checkpoint_dir / "inference_marigold"
    run_output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Generate for each prompt ----
    for i, entry in enumerate(prompts):
        prompt_id = entry["id"]
        prompt_text = entry["prompt"]
        print(f"\n[{i+1}/{len(prompts)}] Generating: {prompt_id}")

        torch.manual_seed(args.seed + i)
        torch.cuda.manual_seed_all(args.seed + i)

        text_embeds = prompt_embeddings[prompt_id]

        # Start from pure noise for the target deformation latent
        x = torch.randn_like(natural_latent)  # [1, 16, T_lat, h, w]

        # ---- Euler ODE integration: t=1 (noise) → t=0 (data) ----
        num_steps = args.num_inference_steps
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

        # Optional CFG
        use_cfg = args.guidance_scale > 1.0
        null_embeds = torch.zeros_like(text_embeds) if use_cfg else None

        with torch.no_grad():
            for step_idx in range(num_steps):
                t_current = timesteps[step_idx]
                dt = timesteps[step_idx + 1] - timesteps[step_idx]  # negative

                # Marigold concatenation: [noisy_target | clean_conditioning]
                model_input = torch.cat([x, natural_latent], dim=1)  # [1, 32, ...]

                t_batch = t_current.expand(1)

                if use_cfg:
                    # Conditioned prediction
                    vel_cond = transformer(
                        model_input, timestep=t_batch,
                        encoder_hidden_states=text_embeds,
                    ).sample
                    # Unconditioned prediction
                    vel_uncond = transformer(
                        model_input, timestep=t_batch,
                        encoder_hidden_states=null_embeds,
                    ).sample
                    velocity = vel_uncond + args.guidance_scale * (vel_cond - vel_uncond)
                else:
                    velocity = transformer(
                        model_input, timestep=t_batch,
                        encoder_hidden_states=text_embeds,
                    ).sample

                # Euler step
                x = x + velocity * dt

                if (step_idx + 1) % 10 == 0:
                    print(f"  Step {step_idx + 1}/{num_steps}")

        # ---- Denormalize + VAE decode ----
        raw_latent = x * latents_std.to(x.device, x.dtype) + latents_mean.to(x.device, x.dtype)

        with torch.no_grad():
            decoded = vae.decode(raw_latent.to(vae.dtype), return_dict=False)[0]

        # [1, 3, T, H, W] → [T, 3, H, W]
        deform_pred = decoded.squeeze(0).permute(1, 0, 2, 3).float().cpu()
        print(f"  Generated deformation: {deform_pred.shape}")
        print(f"  Value range: [{deform_pred.min():.4f}, {deform_pred.max():.4f}]")

        # Save
        sample_dir = run_output_dir / prompt_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        torch.save(deform_pred, str(sample_dir / "deform_field.pt"))
        torch.save(deform_gt, str(sample_dir / "deform_gt.pt"))
        with open(sample_dir / "prompt.txt", "w") as f:
            f.write(prompt_text)

        visualize_deform(deform_pred, sample_dir / "predicted", args.fps)
        visualize_deform(deform_gt, sample_dir / "ground_truth", args.fps)

    print(f"\nDone. Results saved to {run_output_dir}/")


if __name__ == "__main__":
    main()
