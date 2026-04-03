"""
Step 3.1: Inference — generate expression dense fields from text.

Loads the LoRA-fine-tuned Wan2.2 model, generates a latent from a text
prompt, decodes through the VAE, and reassembles into a 45-channel
expression dense field that can be fed to the rendering UNet.

Usage:
    cd /data/pouyan/baseline/repository/cap4d
    PYTHONPATH=. python text_to_expr_field/scripts/inference.py \
        --prompt "A person says: 'Hello, welcome.' Warm tone, moderate pace." \
        --checkpoint outputs/text_to_expr_field/run_YYYYMMDD/final \
        --output outputs/generated_expr_field.pt
"""

import argparse
from pathlib import Path

import torch
torch.backends.cudnn.enabled = False

from text_to_expr_field.src.utils import pseudo_video_to_expr_field


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", required=True, help="Text caption describing the speech")
    p.add_argument("--checkpoint", required=True, help="Path to LoRA checkpoint directory")
    p.add_argument("--model_id", default="Wan-AI/Wan2.2-T2V-A14B-Diffusers")
    p.add_argument("--output", default="outputs/generated_expr_field.pt")
    p.add_argument("--num_frames", type=int, default=241,
                   help="Total pseudo-video frames (must satisfy 4k+1; 241 for 16 expr frames)")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    checkpoint_dir = Path(args.checkpoint)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Load base pipeline
    from diffusers import WanPipeline
    print(f"Loading {args.model_id}...")
    pipe = WanPipeline.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16,
    )

    # Load LoRA weights for transformer(s)
    from peft import PeftModel

    lora_path = checkpoint_dir / "lora_transformer"
    if lora_path.exists():
        print(f"Loading LoRA (transformer): {lora_path}")
        pipe.transformer = PeftModel.from_pretrained(pipe.transformer, str(lora_path))
    else:
        print(f"WARNING: No LoRA found at {lora_path}")

    # Check for MoE second expert
    lora_path_2 = checkpoint_dir / "lora_transformer_2"
    if lora_path_2.exists() and hasattr(pipe, "transformer_2"):
        print(f"Loading LoRA (transformer_2): {lora_path_2}")
        pipe.transformer_2 = PeftModel.from_pretrained(pipe.transformer_2, str(lora_path_2))

    pipe = pipe.to(device)

    # Generate
    print(f"Generating with prompt: '{args.prompt}'")
    output = pipe(
        prompt=args.prompt,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
    )

    # The pipeline output format depends on the diffusers version.
    # Extract the generated video tensor.
    if hasattr(output, "frames"):
        generated = output.frames
        if isinstance(generated, list):
            generated = torch.stack([torch.tensor(f) for f in generated])
    else:
        generated = output[0]

    # Ensure shape is [T, 3, H, W]
    if generated.ndim == 5:
        generated = generated.squeeze(0)
    if generated.shape[1] != 3 and generated.shape[-1] == 3:
        generated = generated.permute(0, 3, 1, 2)

    print(f"Generated pseudo-video: {generated.shape}")

    # Reassemble to 45-channel expression field
    expr_field = pseudo_video_to_expr_field(generated, num_frames=16)
    print(f"Expression field: {expr_field.shape}")
    print(f"Value range: [{expr_field.min():.4f}, {expr_field.max():.4f}]")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(expr_field, str(output_path))
    print(f"Saved: {output_path}")

    # Also save a visualization
    _visualize(expr_field, output_path.with_suffix(""))


def _visualize(expr_field, output_dir):
    """Save deformation heatmap frames for visual inspection."""
    import cv2
    import numpy as np

    output_dir = Path(output_dir)
    vis_dir = output_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    T = expr_field.shape[0]
    for t in range(T):
        deform = expr_field[t, 42:45].numpy().transpose(1, 2, 0)
        abs_max = np.abs(deform).max() + 1e-8
        vis = ((deform / abs_max + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        cv2.imwrite(str(vis_dir / f"deform_{t:04d}.png"), vis[..., ::-1])

    print(f"Visualizations: {vis_dir}/")


if __name__ == "__main__":
    main()
