"""
Train text-to-expression-field diffusion model (Wan DiT, LoRA or full fine-tuning).

Supports two data modes:
  - Cached: loads precomputed VAE latents from disk (fast, any model size)
  - On-the-fly: computes deform/expr fields from fit.npz + VAE-encodes live

Usage:
    cd /data/pouyan/baseline/repository/cap4d

    # Single GPU:
    PYTHONPATH=. python text_to_expr_field/scripts/train.py \
        --config text_to_expr_field/configs/train_config.yaml

    # Multi-GPU:
    PYTHONPATH=. accelerate launch --num_processes 4 --mixed_precision bf16 \
        text_to_expr_field/scripts/train.py \
        --config text_to_expr_field/configs/train_config.yaml
"""

import argparse
import logging
import math
import shutil
from datetime import datetime
from pathlib import Path

import torch
torch.backends.cudnn.enabled = False

import torch.nn.functional as F
from torch.utils.data import DataLoader

from text_to_expr_field.src.data import CachedLatentDataset, OnTheFlyDataset, collate_fn
from text_to_expr_field.src.model import load_pipeline, setup_lora, setup_full_finetune, save_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


def log_rank0(msg, is_main):
    if is_main:
        logger.info(msg)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--resume", default=None)
    return p.parse_args()


def load_config(config_path):
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_dataset(cfg, vae, device):
    """Factory: return CachedLatentDataset or OnTheFlyDataset based on config."""
    on_the_fly = cfg.get("on_the_fly", False)

    if on_the_fly:
        return OnTheFlyDataset(
            manifest_path=cfg.get("manifest_path", "data/derived/manifest.json"),
            vae=vae,
            mode=cfg.get("mode", "deform"),
            target_frames=cfg.get("target_frames", 81),
            resolution=cfg.get("resolution", 512),
            prompt_latent_cache_dir=cfg.get("prompt_latent_cache_dir"),
            min_frames=cfg.get("target_frames", 81),
            flame_root=cfg.get("flame_root", "data/flowface"),
        )
    else:
        return CachedLatentDataset(
            manifest_path=cfg.get("manifest_path", "data/derived/manifest.json"),
            vae_latent_cache_dir=cfg["vae_latent_cache_dir"],
            target_latent_T=cfg.get("target_latent_T"),
            prompt_latent_cache_dir=cfg.get("prompt_latent_cache_dir"),
            min_frames=cfg.get("target_frames", 80),
            flame_root=cfg.get("flame_root", "data/flowface"),
        )


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Reproducibility
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # ---- Accelerate ----
    try:
        from accelerate import Accelerator
        accelerator = Accelerator(
            mixed_precision=cfg.get("mixed_precision", "bf16"),
            gradient_accumulation_steps=cfg.get("gradient_accumulation", 4),
            log_with="tensorboard",
            project_dir=cfg.get("output_dir", "outputs/text_to_expr_field"),
        )
        device = accelerator.device
        is_main = accelerator.is_main_process
    except ImportError:
        accelerator = None
        device = torch.device(f"cuda:{cfg.get('gpu', 0)}")
        is_main = True

    # ---- Run directory ----
    output_dir = Path(cfg.get("output_dir", "outputs/text_to_expr_field"))
    if is_main:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = output_dir / f"run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.config, run_dir / "config.yaml")
        log_rank0(f"Run directory: {run_dir}", is_main)
        marker = output_dir / ".current_run_dir"
        marker.write_text(str(run_dir))
    else:
        import time
        marker = output_dir / ".current_run_dir"
        for _ in range(60):
            if marker.exists():
                break
            time.sleep(0.5)
        run_dir = Path(marker.read_text().strip())

    # ---- Load model ----
    log_rank0(f"Loading pipeline: {cfg.get('model_id')}...", is_main)
    components = load_pipeline(cfg, device)
    transformer = components["transformer"]
    vae = components["vae"]
    latents_mean = components["latents_mean"]
    latents_std = components["latents_std"]

    on_the_fly = cfg.get("on_the_fly", False)
    log_rank0(f"VAE: {'kept for on-the-fly' if on_the_fly else 'freed (using cached latents)'}", is_main)

    # ---- Memory optimizations ----
    if cfg.get("gradient_checkpointing", True):
        if hasattr(transformer, "enable_gradient_checkpointing"):
            transformer.enable_gradient_checkpointing()
        log_rank0("Gradient checkpointing enabled", is_main)

    if cfg.get("group_offload", False):
        from diffusers.hooks import apply_group_offloading
        apply_group_offloading(
            transformer, onload_device=device,
            offload_device=torch.device("cpu"),
            offload_type="leaf_level", use_stream=True,
        )
        log_rank0("Group offloading enabled", is_main)

    # ---- LoRA or full fine-tuning ----
    use_lora = cfg.get("use_lora", True)
    if use_lora:
        transformer = setup_lora(transformer, cfg)
        log_rank0(f"LoRA: rank={cfg.get('lora_rank', 128)}, alpha={cfg.get('lora_alpha', 128)}", is_main)
    else:
        transformer = setup_full_finetune(transformer)
        total = sum(p.numel() for p in transformer.parameters())
        log_rank0(f"Full fine-tuning: {total:,} params", is_main)

    transformer = transformer.to(device)
    if is_main and use_lora:
        transformer.print_trainable_parameters()

    # ---- Dataset ----
    dataset = build_dataset(cfg, vae, device)
    log_rank0(f"Dataset: {len(dataset)} clips", is_main)

    num_workers = 0 if on_the_fly else cfg.get("num_workers", 2)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.get("batch_size", 1),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=not on_the_fly,
        drop_last=True,
        collate_fn=collate_fn,
    )

    # ---- Optimizer ----
    trainable_params = [p for p in transformer.parameters() if p.requires_grad]
    max_steps = cfg.get("max_steps", 20000)
    warmup_steps = cfg.get("warmup_steps", 500)
    lr = cfg.get("lr", 1e-5)

    optimizer = torch.optim.AdamW(
        trainable_params, lr=lr,
        weight_decay=cfg.get("weight_decay", 0.01),
    )

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---- Accelerate prepare ----
    if accelerator is not None:
        transformer, optimizer, dataloader, lr_scheduler = accelerator.prepare(
            transformer, optimizer, dataloader, lr_scheduler,
        )

    # ---- Training loop ----
    cfg_dropout = cfg.get("cfg_dropout", 0.1)
    grad_clip = cfg.get("grad_clip", 1.0)
    save_every = cfg.get("save_every", 2000)
    log_every = cfg.get("log_every", 50)

    global_step = 0
    transformer.train()
    log_rank0(f"Starting training for {max_steps} steps...", is_main)

    while global_step < max_steps:
        for batch in dataloader:
            if global_step >= max_steps:
                break

            raw_latents = batch["latent"].to(device, dtype=torch.bfloat16)
            latents = (raw_latents - latents_mean.to(device, dtype=torch.bfloat16)) / \
                      latents_std.to(device, dtype=torch.bfloat16)
            text_embeds = batch["text_embed"].to(device, dtype=torch.bfloat16)

            # CFG dropout
            B = latents.shape[0]
            drop_mask = torch.rand(B, device=device) < cfg_dropout
            if drop_mask.any():
                text_embeds = torch.where(
                    drop_mask[:, None, None].expand_as(text_embeds),
                    torch.zeros_like(text_embeds), text_embeds,
                )

            # Flow matching
            timesteps = torch.rand(B, device=device, dtype=torch.bfloat16)
            noise = torch.randn_like(latents)
            t_expand = timesteps[:, None, None, None, None]
            noisy_latents = (1 - t_expand) * latents + t_expand * noise
            target_velocity = noise - latents

            velocity_pred = transformer(
                noisy_latents, timestep=timesteps,
                encoder_hidden_states=text_embeds,
            ).sample

            loss = F.mse_loss(velocity_pred.float(), target_velocity.float())

            if accelerator is not None:
                accelerator.backward(loss)
            else:
                loss.backward()

            if grad_clip > 0:
                params = [p for p in transformer.parameters() if p.requires_grad]
                if accelerator is not None:
                    accelerator.clip_grad_norm_(params, grad_clip)
                else:
                    torch.nn.utils.clip_grad_norm_(params, grad_clip)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if is_main and global_step % log_every == 0:
                log_rank0(
                    f"step {global_step}/{max_steps} | "
                    f"loss={loss.item():.4f} | "
                    f"lr={lr_scheduler.get_last_lr()[0]:.2e}",
                    is_main,
                )

            if is_main and global_step % save_every == 0:
                ckpt = save_checkpoint(transformer, global_step, run_dir, use_lora, accelerator)
                log_rank0(f"Saved checkpoint: {ckpt}", is_main)

    # Final save
    if is_main:
        ckpt = save_checkpoint(transformer, global_step, run_dir, use_lora, accelerator)
        log_rank0(f"Training complete. Final checkpoint: {ckpt}", is_main)


if __name__ == "__main__":
    main()
