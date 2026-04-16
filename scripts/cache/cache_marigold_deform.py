"""
Cache deformation maps predicted by the Marigold-trained SD3.5 model.

For each clip, reads every frame from the preprocessed 512x512 video
(data/talkvid/talkvid/{clip_id}.mp4) and runs it through the Marigold model.
Frames are fed directly at 512x512 — no FLAME crop is applied.
Inference is batched (default 16 frames) for better GPU utilization.

Output per clip:
  {output_dir}/{clip_id}/deform_field.pt    — [T, 3, 512, 512] float16 tensor
  {output_dir}/{clip_id}/deformation.mp4    — visualization video (lossy, for inspection only)

Usage:
    cd <repo_root>

    # Single GPU:
    PYTHONPATH=. python scripts/cache/cache_marigold_deform.py --gpu 0

    # Parallel across 4 GPUs:
    PYTHONPATH=. python scripts/cache/cache_marigold_deform.py --gpu 0 --num_gpus 4 &
    PYTHONPATH=. python scripts/cache/cache_marigold_deform.py --gpu 1 --num_gpus 4 &
    PYTHONPATH=. python scripts/cache/cache_marigold_deform.py --gpu 2 --num_gpus 4 &
    PYTHONPATH=. python scripts/cache/cache_marigold_deform.py --gpu 3 --num_gpus 4 &

    # Test on one clip:
    PYTHONPATH=. python scripts/cache/cache_marigold_deform.py --test
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
torch.backends.cudnn.enabled = False

from tqdm import tqdm

DEFAULT_CHECKPOINT = "outputs/marigold_spatial/run_20260410_180104/step_011000/transformer/diffusion_pytorch_model.safetensors"
DEFAULT_MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    p.add_argument("--manifest", default="data/derived/manifest.json",
                   help="Training manifest produced by scripts/manifest/build_manifest.py; "
                        "the 'clip_id' field of each entry drives caching.")
    p.add_argument("--video_root", default="data/talkvid/talkvid")
    p.add_argument("--output_dir", default="data/derived/marigold_deform")
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--num_gpus", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--test", action="store_true")
    p.add_argument("--clip", type=str, default=None)
    return p.parse_args()


def normalize_to_uint8(arr):
    m = np.abs(arr).max() + 1e-8
    return ((arr / m + 1) / 2 * 255).clip(0, 255).astype(np.uint8)


def load_marigold_model(model_id, checkpoint_path, device):
    """Load SD3.5, encode null prompt, apply Marigold checkpoint."""
    from diffusers import StableDiffusion3Pipeline
    from marigold_training.src.marigold_model import double_input_channels
    from safetensors.torch import load_file

    print(f"Loading {model_id}...")
    pipe = StableDiffusion3Pipeline.from_pretrained(model_id)

    transformer = pipe.transformer.to(dtype=torch.bfloat16)
    vae = pipe.vae.to(device).eval()
    vae.requires_grad_(False)

    scaling_factor = vae.config.scaling_factor
    shift_factor = vae.config.shift_factor

    print("Encoding null prompt...")
    pipe.text_encoder.to(device)
    pipe.text_encoder_2.to(device)
    pipe.text_encoder_3.to(device)
    with torch.no_grad():
        null_out = pipe.encode_prompt(prompt="", prompt_2="", prompt_3="", device=device)
        null_text = null_out[0].to(dtype=torch.bfloat16)
        null_pooled = null_out[2].to(dtype=torch.bfloat16)

    del pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3
    del pipe.tokenizer, pipe.tokenizer_2, pipe.tokenizer_3
    del pipe
    torch.cuda.empty_cache()

    transformer = double_input_channels(transformer)
    print(f"Loading checkpoint: {checkpoint_path}")
    transformer.load_state_dict(load_file(str(checkpoint_path)), strict=True)
    transformer = transformer.to(device, dtype=torch.bfloat16).eval()

    return transformer, vae, scaling_factor, shift_factor, null_text, null_pooled


@torch.no_grad()
def predict_deform_batch(transformer, vae, img_batch, null_text, null_pooled,
                         scaling_factor, shift_factor, num_steps, base_seed, device):
    """
    Run a batch of 512x512 frames through the Marigold pipeline.

    Args:
        img_batch:  [B, 3, H, W] float32 in [-1, 1]. B can be any size
                    (handles the last partial batch of a clip naturally).
        base_seed:  seed for the first frame in this batch; frame i uses
                    base_seed + i.

    Returns:
        [B, 3, H, W] float16 deformation maps
    """
    B = img_batch.shape[0]

    nat_latent = vae.encode(
        img_batch.to(device, dtype=vae.dtype)
    ).latent_dist.mode()
    nat_latent = ((nat_latent - shift_factor) * scaling_factor).to(dtype=torch.bfloat16)

    # Generate per-frame noise with deterministic seeds.
    noise_list = []
    for i in range(B):
        torch.manual_seed(base_seed + i)
        noise_list.append(torch.randn_like(nat_latent[:1]))
    x = torch.cat(noise_list, dim=0)

    timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    null_text_b = null_text.expand(B, -1, -1)
    null_pooled_b = null_pooled.expand(B, -1)

    for i in range(num_steps):
        model_input = torch.cat([x, nat_latent], dim=1)
        t_ms = (timesteps[i] * 1000).long().expand(B)
        velocity = transformer(
            hidden_states=model_input, timestep=t_ms,
            encoder_hidden_states=null_text_b,
            pooled_projections=null_pooled_b,
        ).sample
        x = x + velocity * (timesteps[i + 1] - timesteps[i])

    raw = x.float() / scaling_factor + shift_factor
    decoded = vae.decode(raw.to(vae.dtype), return_dict=False)[0]
    return decoded.float().cpu().to(torch.float16)


def read_video_frames(video_path):
    """Read all frames from an mp4 as a list of [3, H, W] tensors in [-1, 1]."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 127.5 - 1.0
        frames.append(torch.from_numpy(frame).permute(2, 0, 1))
    cap.release()
    return frames


def identity_first_ordering(clip_ids: list[str]) -> list[str]:
    """Reorder clips so that every identity appears once before any repeats.

    Identity is the portion before '_NA_' (the YouTube video ID). The result
    is a round-robin over identities: the first pass picks one clip per
    identity, the second pass picks the next clip from identities that have
    more, and so on. This maximises identity coverage early in the run so
    training can start on a diverse set while caching is still in progress.
    """
    from collections import defaultdict
    buckets = defaultdict(list)
    for cid in clip_ids:
        ident = cid.split("_NA_")[0] if "_NA_" in cid else cid
        buckets[ident].append(cid)

    # Round-robin: take one clip per identity per pass.
    ordered = []
    while buckets:
        empty = []
        for ident in sorted(buckets):
            ordered.append(buckets[ident].pop(0))
            if not buckets[ident]:
                empty.append(ident)
        for ident in empty:
            del buckets[ident]
    return ordered


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_root = Path(args.video_root)

    # Build clip list
    if args.clip:
        all_clips = [args.clip]
    else:
        import json
        with open(args.manifest) as f:
            manifest = json.load(f)
        all_clips = identity_first_ordering(
            [entry["clip_id"] for entry in manifest]
        )

    print(f"Total clips: {len(all_clips)}")

    # Shard
    my_clips = all_clips[args.gpu::args.num_gpus]
    print(f"GPU {args.gpu}/{args.num_gpus}: {len(my_clips)} clips")

    if args.test:
        my_clips = my_clips[:1]

    # Skip already done
    to_process = [c for c in my_clips if not (output_dir / c / "deform_field.pt").exists()]
    print(f"To process: {len(to_process)} | Already done: {len(my_clips) - len(to_process)}")

    if not to_process:
        print("Nothing to do.")
        return

    # Load Marigold model once
    transformer, vae, sf, shf, null_text, null_pooled = load_marigold_model(
        args.model_id, args.checkpoint, device,
    )

    n_ok = n_fail = 0
    for clip_id in tqdm(to_process, desc=f"GPU {args.gpu}"):
        try:
            video_path = video_root / f"{clip_id}.mp4"
            if not video_path.exists():
                tqdm.write(f"[SKIP] {clip_id}: video not found")
                continue

            frames = read_video_frames(video_path)
            if len(frames) == 0:
                tqdm.write(f"[SKIP] {clip_id}: empty video")
                continue

            # Batched inference — process `batch_size` frames at a time.
            # The last batch may be smaller (no padding needed).
            all_frames = torch.stack(frames, dim=0)  # [T, 3, H, W]
            deform_batches = []
            for b_start in range(0, len(frames), args.batch_size):
                batch = all_frames[b_start : b_start + args.batch_size]
                deform = predict_deform_batch(
                    transformer, vae, batch, null_text, null_pooled,
                    sf, shf, args.num_steps, args.seed + b_start, device,
                )
                deform_batches.append(deform)

            deform_field = torch.cat(deform_batches, dim=0)  # [T, 3, H, W] float16

            # Save fp16 tensor (lossless relative to bf16 inference precision)
            clip_dir = output_dir / clip_id
            clip_dir.mkdir(parents=True, exist_ok=True)
            torch.save(deform_field, str(clip_dir / "deform_field.pt"))

            # Save visualization video (lossy, for inspection only)
            T, _, H, W = deform_field.shape
            writer = cv2.VideoWriter(
                str(clip_dir / "deformation.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H),
            )
            for t in range(T):
                vis = normalize_to_uint8(deform_field[t].float().numpy().transpose(1, 2, 0))
                writer.write(vis[..., ::-1])
            writer.release()

            n_ok += 1
            if args.test:
                print(f"  Clip: {clip_id}")
                print(f"  Frames: {len(frames)}")
                print(f"  Shape: {deform_field.shape}")
                print(f"  Range: [{deform_field.min():.4f}, {deform_field.max():.4f}]")
                print(f"  Saved: {clip_dir}")

        except Exception as e:
            tqdm.write(f"[FAIL] {clip_id}: {e}")
            n_fail += 1

    print(f"Done. Success: {n_ok} | Failed: {n_fail}")


if __name__ == "__main__":
    main()
