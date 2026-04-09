"""
Inference — generate expression dense fields or deformation maps from text.

Supports multi-GPU parallelism via torchrun: each GPU processes a disjoint
subset of prompts. No gradient sync or communication needed.

Usage:
    cd <repo_root>

    # Single GPU:
    PYTHONPATH=. python text_to_expr_field/scripts/inference.py \
        --prompts text_to_expr_field/configs/eval_prompts.json \
        --checkpoint outputs/text_to_expr_field/run_YYYYMMDD/final

    # Multi-GPU:
    PYTHONPATH=. torchrun --nproc_per_node=4 \
        text_to_expr_field/scripts/inference.py \
        --prompts text_to_expr_field/configs/eval_prompts.json \
        --checkpoint outputs/text_to_expr_field/run_YYYYMMDD/final
"""

import argparse
import json
import os
from pathlib import Path

import torch
torch.backends.cudnn.enabled = False

from text_to_expr_field.src.model import load_inference_pipeline
from text_to_expr_field.src.utils import pseudo_video_to_expr_field
from text_to_expr_field.src.vis import visualize_expr_field, visualize_deform


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", required=True,
                   help="Path to JSON file with list of {id, prompt} entries")
    p.add_argument("--checkpoint", required=True,
                   help="Path to checkpoint directory")
    p.add_argument("--model_id", default="Wan-AI/Wan2.2-T2V-A14B-Diffusers")
    p.add_argument("--mode", choices=["expr_field", "deform"], default="expr_field",
                   help="expr_field: full 45ch, deform: 3ch deformation map")
    p.add_argument("--target_real_frames", type=int, default=24,
                   help="Expression frames to generate (must be divisible by 4)")
    p.add_argument("--num_frames", type=int, default=None,
                   help="Override VAE-level frame count (for deform mode)")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps", type=int, default=25)
    return p.parse_args()


def compute_num_frames(args):
    """Derive the pseudo-video frame count from args."""
    assert args.target_real_frames % 4 == 0

    if args.mode == "deform":
        if args.num_frames is not None:
            return args.num_frames
        n = args.target_real_frames
        if (n - 1) % 4 != 0:
            n = 4 * (n // 4) + 1
        return n
    else:
        return 15 * args.target_real_frames + 1


def decode_latents(pipe, latents):
    """Denormalize latents and VAE decode without clamping."""
    latents_mean = (
        torch.tensor(pipe.vae.config.latents_mean)
        .view(1, pipe.vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std_inv = (
        1.0 / torch.tensor(pipe.vae.config.latents_std)
        .view(1, pipe.vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents = latents / latents_std_inv + latents_mean

    with torch.no_grad():
        decoded = pipe.vae.decode(latents.to(pipe.vae.dtype), return_dict=False)[0]

    # [1, C, T, H, W] -> [T, C, H, W]
    return decoded.squeeze(0).permute(1, 0, 2, 3).float().cpu()


def main():
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint)

    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(f"cuda:{rank}")
    num_pseudo_frames = compute_num_frames(args)

    # Load prompts, shard by rank
    with open(args.prompts) as f:
        all_prompts = json.load(f)
    prompts = [p for i, p in enumerate(all_prompts) if i % world_size == rank]
    prompt_indices = [i for i in range(len(all_prompts)) if i % world_size == rank]

    if rank == 0:
        print(f"Distributed: {world_size} GPU(s), {len(all_prompts)} prompts")

    # Load pipeline + checkpoint
    if rank == 0:
        print(f"Loading {args.model_id} + checkpoint {checkpoint_dir}...")
    pipe = load_inference_pipeline(args.model_id, checkpoint_dir, device)

    # Output alongside checkpoint
    run_output_dir = checkpoint_dir / "inference"
    run_output_dir.mkdir(parents=True, exist_ok=True)

    for local_i, (global_i, entry) in enumerate(zip(prompt_indices, prompts)):
        prompt_id = entry["id"]
        prompt_text = entry["prompt"]
        print(f"[GPU {rank}] [{local_i+1}/{len(prompts)}] Generating: {prompt_id}")

        torch.manual_seed(args.seed + global_i)
        torch.cuda.manual_seed_all(args.seed + global_i)

        output = pipe(
            prompt=prompt_text,
            num_frames=num_pseudo_frames,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            output_type="latent",
        )

        generated = decode_latents(pipe, output.frames)
        print(f"[GPU {rank}]   Decoded video: {generated.shape}")

        sample_dir = run_output_dir / prompt_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        if args.mode == "deform":
            print(f"[GPU {rank}]   Deformation: range [{generated.min():.4f}, {generated.max():.4f}]")
            torch.save(generated, str(sample_dir / "deform_field.pt"))
            visualize_deform(generated, sample_dir, args.fps, verbose=(rank == 0))
        else:
            expr_field = pseudo_video_to_expr_field(generated, num_frames=args.target_real_frames)
            print(f"[GPU {rank}]   Expression field: {expr_field.shape}, range [{expr_field.min():.4f}, {expr_field.max():.4f}]")
            torch.save(expr_field, str(sample_dir / "expr_field.pt"))
            visualize_expr_field(expr_field, sample_dir, args.fps, verbose=(rank == 0))

        with open(sample_dir / "prompt.txt", "w") as f:
            f.write(prompt_text)

    print(f"[GPU {rank}] Done. Results saved to {run_output_dir}/")


if __name__ == "__main__":
    main()
