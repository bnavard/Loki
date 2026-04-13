"""
Precompute and cache VAE latents for all clips using DDP.

Uses the ExprFieldDataset to load expression fields deterministically
(always first 80 frames), then encodes them through the frozen Wan2.2
VAE and saves the latent tensors to disk.

Each GPU processes a unique shard of the dataset via DistributedSampler.
DataLoader workers prefetch expression fields while the VAE encodes the
current batch, keeping the GPU utilized.

Output: data/derived/vae_latent_cache/{clip_id}.pt

Usage:
    cd <repo_root>

    # Single GPU:
    PYTHONPATH=. python caching/scripts/cache_vae_latents.py

    # Multi-GPU DDP (e.g. 4 GPUs):
    PYTHONPATH=. torchrun --nproc_per_node=4 caching/scripts/cache_vae_latents.py

    # Specific GPUs:
    CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. torchrun --nproc_per_node=4 \
        caching/scripts/cache_vae_latents.py

    # Test on one batch:
    PYTHONPATH=. python caching/scripts/cache_vae_latents.py --test
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler

torch.backends.cudnn.enabled = False
mp.set_start_method("spawn", force=True)

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

OUTPUT_DIR = Path("data/derived/vae_latent_cache")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest_path", type=str, default="data/derived/manifest.json")
    p.add_argument("--flame_root", type=str, default="data/flowface")
    p.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    p.add_argument("--target_frames", type=int, default=80)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=1,
                   help="Clips per GPU per batch")
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader workers (0 recommended — expression field computation uses CUDA)")
    p.add_argument("--test", action="store_true", help="Process one batch only")
    return p.parse_args()


def setup_distributed():
    """Initialize DDP if launched with torchrun, otherwise run single-GPU."""
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


def collate_fn(batch):
    """Custom collate that handles variable-length pseudo-videos (though they
    should all be the same length with deterministic target_frames)."""
    pseudo_videos = [item["pseudo_video"] for item in batch]
    clip_ids = [item["clip_id"] for item in batch]
    stacked = torch.stack([p.permute(1, 0, 2, 3) for p in pseudo_videos], dim=0)
    return {"video_batch": stacked, "clip_ids": clip_ids}


class CacheFilteredDataset(torch.utils.data.Dataset):
    """Wraps ExprFieldDataset and skips clips that are already cached."""

    def __init__(self, base_dataset, output_dir):
        self.base_dataset = base_dataset
        self.output_dir = Path(output_dir)

        # Build filtered index: only include clips not yet cached
        self.valid_indices = []
        for i, entry in enumerate(base_dataset.samples):
            cache_path = self.output_dir / f"{entry['clip_id']}.pt"
            if not cache_path.exists():
                self.valid_indices.append(i)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        return self.base_dataset[real_idx]


def main():
    args = parse_args()
    rank, world_size, device = setup_distributed()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    is_main = (rank == 0)

    # Build dataset (computes expression fields on the fly, returns pseudo-video)
    from caching.scripts._expr_field_dataset import ExprFieldCachingDataset

    base_dataset = ExprFieldCachingDataset(
        manifest_path=args.manifest_path,
        flame_root=args.flame_root,
        target_frames=args.target_frames,
        resolution=args.resolution,
        device=str(device),
    )

    # skip already-cached clips
    dataset = CacheFilteredDataset(base_dataset, output_dir)

    if is_main:
        print(f"Total clips: {len(base_dataset)}")
        print(f"To process: {len(dataset)} (already cached: {len(base_dataset) - len(dataset)})")

    if len(dataset) == 0:
        if is_main:
            print("Nothing to do.")
        cleanup_distributed()
        return

    # Distributed sampler ensures each GPU gets a unique shard
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    # Load VAE
    if is_main:
        print("Loading Wan2.2 VAE...")
    from diffusers import AutoencoderKLWan
    vae = AutoencoderKLWan.from_pretrained(
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        subfolder="vae",
        torch_dtype=torch.float32,
    ).to(device).eval()

    # Encode and save
    n_ok = 0
    pbar = tqdm(dataloader, desc=f"GPU {rank}", disable=(not is_main))

    for batch_idx, batch in enumerate(pbar):
        if args.test and batch_idx >= 1:
            break

        video_batch = batch["video_batch"].to(device, dtype=torch.float32)  # [B, 3, T, H, W]
        clip_ids = batch["clip_ids"]

        with torch.no_grad():
            posterior = vae.encode(video_batch).latent_dist
            latents = posterior.mode()  # [B, z_dim, T_lat, H_lat, W_lat]

        for i, clip_id in enumerate(clip_ids):
            torch.save({
                "latent": latents[i].cpu(),
                "num_expr_frames": args.target_frames,
            }, str(output_dir / f"{clip_id}.pt"))
            n_ok += 1

        pbar.set_postfix(cached=n_ok)

    # Sync all GPUs before printing summary
    if dist.is_initialized():
        dist.barrier()

    print(f"[GPU {rank}] Cached {n_ok} clips.")

    cleanup_distributed()


if __name__ == "__main__":
    main()
