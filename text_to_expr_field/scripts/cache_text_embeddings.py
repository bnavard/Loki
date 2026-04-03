"""
Precompute and cache UMT5 text embeddings for all clips using DDP.

For each clip, loads the caption from data/derived/captions/{clip_id}.json,
encodes it through the frozen UMT5-XXL text encoder (from Wan2.2), and saves
the embedding tensor to disk.

This eliminates the need to load the ~13B param UMT5 model during training,
freeing significant GPU memory and speeding up the training loop.

Output: data/derived/prompt_latent_cache/{clip_id}.pt

Usage:
    cd /data/pouyan/baseline/repository/cap4d

    # Single GPU:
    PYTHONPATH=. python text_to_expr_field/scripts/cache_text_embeddings.py

    # Multi-GPU DDP:
    PYTHONPATH=. torchrun --nproc_per_node=4 text_to_expr_field/scripts/cache_text_embeddings.py

    # Test on one batch:
    PYTHONPATH=. python text_to_expr_field/scripts/cache_text_embeddings.py --test
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, DistributedSampler

torch.backends.cudnn.enabled = False
mp.set_start_method("spawn", force=True)

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

OUTPUT_DIR = Path("data/derived/prompt_latent_cache")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest_path", type=str, default="data/derived/manifest.json")
    p.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    p.add_argument("--model_id", type=str, default="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
                   help="Wan2.2 model ID (for loading tokenizer + text encoder)")
    p.add_argument("--batch_size", type=int, default=32,
                   help="Captions per batch (text encoding is cheap, use large batches)")
    p.add_argument("--max_length", type=int, default=512,
                   help="Max token length for text encoder")
    p.add_argument("--test", action="store_true")
    return p.parse_args()


class CaptionDataset(Dataset):
    def __init__(self, manifest_path, output_dir):
        with open(manifest_path) as f:
            manifest = json.load(f)

        output_dir = Path(output_dir)

        # Only include clips with captions that haven't been cached yet
        self.samples = []
        for entry in manifest:
            clip_id = entry["clip_id"]
            caption_file = entry.get("caption_file")
            if caption_file is None or not Path(caption_file).exists():
                continue
            if (output_dir / f"{clip_id}.pt").exists():
                continue
            self.samples.append(entry)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry = self.samples[idx]
        with open(entry["caption_file"]) as f:
            caption_data = json.load(f)
        return {
            "clip_id": entry["clip_id"],
            "caption": caption_data["caption"],
        }


def collate_fn(batch):
    return {
        "clip_ids": [item["clip_id"] for item in batch],
        "captions": [item["caption"] for item in batch],
    }


# ---------------------------------------------------------------------------
# DDP setup
# ---------------------------------------------------------------------------

def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda:0")
    return rank, world_size, device


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    args = parse_args()
    rank, world_size, device = setup_distributed()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    is_main = (rank == 0)

    dataset = CaptionDataset(args.manifest_path, args.output_dir)
    
    if is_main:
        print(f"Captions to encode: {len(dataset)}")

    if len(dataset) == 0:
        if is_main:
            print("Nothing to do.")
        cleanup_distributed()
        return

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=False,
    )

    # Load tokenizer + text encoder from Wan2.2
    if is_main:
        print(f"Loading text encoder from {args.model_id}...")

    from diffusers import WanPipeline

    pipe = WanPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder.to(device).eval()

    # Free everything else from the pipeline
    del pipe.transformer, pipe.vae, pipe.scheduler
    del pipe
    torch.cuda.empty_cache()

    if is_main:
        print("Text encoder loaded. Encoding captions...")

    # Encode and save
    n_ok = 0
    pbar = tqdm(dataloader, desc=f"GPU {rank}", disable=(not is_main))

    for batch in pbar:
        clip_ids = batch["clip_ids"]
        captions = batch["captions"]

        text_inputs = tokenizer(
            captions,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            text_embeds = text_encoder(**text_inputs)[0]  # (B, seq_len, dim)

        for i, clip_id in enumerate(clip_ids):
            torch.save({
                "text_embed": text_embeds[i].cpu(),
                "caption": captions[i],
            }, str(output_dir / f"{clip_id}.pt"))
            n_ok += 1

        pbar.set_postfix(cached=n_ok)

        if args.test:
            if is_main:
                print(f"\nTest: encoded {len(clip_ids)} captions")
                print(f"  Embed shape: {text_embeds.shape}")
                print(f"  Clip IDs: {clip_ids[:3]}...")
            break

    if dist.is_initialized():
        dist.barrier()

    print(f"[GPU {rank}] Cached {n_ok} text embeddings.")
    cleanup_distributed()


if __name__ == "__main__":
    main()
