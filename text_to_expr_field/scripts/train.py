"""
Train text → expression field model (Wan2.2-T2V-A14B, 14B DiT, LoRA).

Loads precomputed VAE latents and UMT5 text embeddings from disk — neither
the VAE encoder nor text encoder is loaded during training, maximizing GPU
memory for the 14B transformer.

Each training step randomly slices a temporal window from the cached latent,
normalizes with pretrained latents_mean/std, and trains with flow matching
(velocity prediction) loss in float32.

Requires: diffusers, peft, accelerate, precomputed caches.

Usage:
    cd /data/pouyan/baseline/repository/cap4d

    # Single GPU:
    PYTHONPATH=. python text_to_expr_field/scripts/train.py \
        --config text_to_expr_field/configs/train_config.yaml

    # 8x H200 distributed:
    PYTHONPATH=. accelerate launch --num_processes 8 --mixed_precision bf16 \
        text_to_expr_field/scripts/train.py \
        --config text_to_expr_field/configs/train_config.yaml
"""

import argparse
import logging
import math
import os
import shutil
from datetime import datetime
from pathlib import Path

import torch
torch.backends.cudnn.enabled = False

import torch.nn.functional as F
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


def log_rank0(msg, is_main):
    if is_main:
        logger.info(msg)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to train_config.yaml")
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    return p.parse_args()


def load_config(config_path):
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Determine if running with accelerate
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

    # Timestamped run directory
    output_dir = Path(cfg.get("output_dir", "outputs/text_to_expr_field"))
    if is_main:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = output_dir / f"run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.config, run_dir / "config.yaml")
        log_rank0(f"Run directory: {run_dir}", is_main)
    else:
        # Wait for rank 0 to create the directory
        import time
        marker = output_dir / ".current_run_dir"
        if is_main:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(str(run_dir))
        else:
            for _ in range(60):
                if marker.exists():
                    break
                time.sleep(0.5)
            run_dir = Path(marker.read_text().strip())

    # Write marker for other ranks (if main)
    if is_main:
        marker = output_dir / ".current_run_dir"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(str(run_dir))

    # ---- Load Wan2.2 transformer via WanPipeline ----
    # We load the full pipeline to get the correct Wan2.2 transformer class,
    # then immediately free the text encoder, tokenizer, and VAE since we
    # use precomputed caches for both text embeddings and VAE latents.
    from diffusers import WanPipeline

    model_id = cfg.get("model_id", "Wan-AI/Wan2.2-T2V-A14B-Diffusers")
    log_rank0(f"Loading pipeline from {model_id}...", is_main)

    pipe = WanPipeline.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
    )

    transformer = pipe.transformer
    scheduler = pipe.scheduler

    # Extract pretrained latent normalization stats before freeing the VAE.
    # The DiT was pretrained with latents normalized by these channel-wise stats.
    # Shape [1, C, 1, 1, 1] to broadcast over [B, C, T, H, W] latents
    latents_mean = torch.tensor(pipe.vae.config.latents_mean).view(1, -1, 1, 1, 1)
    latents_std = torch.tensor(pipe.vae.config.latents_std).view(1, -1, 1, 1, 1)
    log_rank0(f"Latent normalization: mean shape={latents_mean.shape}, std shape={latents_std.shape}", is_main)

    # Free text encoder, tokenizer and VAE to reclaim memory
    del pipe.text_encoder, pipe.tokenizer, pipe.vae
    del pipe
    torch.cuda.empty_cache()
    log_rank0("Freed text encoder + VAE — using cached embeddings and latents", is_main)

    # ---- Memory optimizations (following HF diffusers recommendations) ----

    # 1. Gradient checkpointing: recompute activations during backward pass
    #    instead of storing them. Saves ~40% activation memory, ~30% slower.
    if cfg.get("gradient_checkpointing", True):
        if hasattr(transformer, "enable_gradient_checkpointing"):
            transformer.enable_gradient_checkpointing()
        log_rank0("Gradient checkpointing enabled", is_main)

    # 2. Group offloading: move inactive transformer blocks to CPU, prefetch
    #    the next block to GPU via CUDA streams. Drastically reduces peak GPU
    #    memory for the frozen base model weights.
    if cfg.get("group_offload", True):
        from diffusers.hooks import apply_group_offloading
        apply_group_offloading(
            transformer,
            onload_device=device,
            offload_device=torch.device("cpu"),
            offload_type="leaf_level",
            use_stream=True,
        )
        log_rank0("Group offloading enabled (leaf_level, CUDA streams)", is_main)

    # ---- Apply LoRA ----
    from peft import LoraConfig, get_peft_model

    lora_rank = cfg.get("lora_rank", 128)
    lora_alpha = cfg.get("lora_alpha", 128)

    target_modules = set()
    for name, mod in transformer.named_modules():
        if isinstance(mod, torch.nn.Linear):
            target_modules.add(name.split(".")[-1])

    attn_ffn_keywords = ["to_q", "to_k", "to_v", "to_out", "proj", "ff", "net"]
    target_modules = sorted([m for m in target_modules
                             if any(kw in m for kw in attn_ffn_keywords)])

    log_rank0(f"LoRA rank={lora_rank}, alpha={lora_alpha}, targets={target_modules}", is_main)

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=cfg.get("lora_dropout", 0.05),
    )

    transformer = get_peft_model(transformer, lora_config)
    transformer = transformer.to(device)
    if is_main:
        transformer.print_trainable_parameters()

    # ---- Dataset ----
    from text_to_expr_field.src.dataset import ExprFieldDataset

    target_real_frames = cfg.get("target_real_frames", 24)
    target_latent_T = (15 * target_real_frames) // 4 + 1
    log_rank0(f"Training window: {target_real_frames} real frames → latent T={target_latent_T}", is_main)

    dataset = ExprFieldDataset(
        manifest_path=cfg.get("manifest_path", "data/derived/manifest.json"),
        flame_root=cfg.get("flame_root", "data/flowface"),
        target_frames=cfg.get("target_frames", 80),
        resolution=cfg.get("resolution", 512),
        target_real_frames=target_real_frames,
        vae=None,
        device=str(device),
        vae_latent_cache_dir=cfg.get("vae_latent_cache_dir", "data/derived/vae_latent_cache"),
        prompt_latent_cache_dir=cfg.get("prompt_latent_cache_dir", "data/derived/prompt_latent_cache"),
        cached_only=True,
    )
    log_rank0(f"Dataset: {len(dataset)} clips (VAE and prompt latent pre-cached only)", is_main)

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.get("batch_size", 1),
        shuffle=True,
        num_workers=cfg.get("num_workers", 2),
        pin_memory=True,
        drop_last=True,
    )

    # ---- Optimizer ----
    trainable_params = list(transformer.parameters())

    max_steps = cfg.get("max_steps", 20000)
    warmup_steps = cfg.get("warmup_steps", 500)
    lr = cfg.get("lr", 1e-5)

    # 3. 8-bit Adam: halves optimizer state memory.
    #    Install: pip install bitsandbytes
    use_8bit_adam = cfg.get("use_8bit_adam", False)
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                trainable_params, lr=lr,
                weight_decay=cfg.get("weight_decay", 0.01),
            )
            log_rank0("Using 8-bit AdamW (bitsandbytes)", is_main)
        except ImportError:
            log_rank0("bitsandbytes not found, falling back to standard AdamW", is_main)
            optimizer = torch.optim.AdamW(
                trainable_params, lr=lr,
                weight_decay=cfg.get("weight_decay", 0.01),
            )
    else:
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

    # ---- Accelerate setup (plain DDP, no DeepSpeed) ----
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

            # Load cached latents (already temporally sliced by the dataset)
            # and normalize with pretrained stats so the DiT sees the expected
            # input distribution. Normalization is per-channel over (T, H, W).
            raw_latents = batch["latent"].to(device, dtype=torch.bfloat16)
            latents = (raw_latents - latents_mean.to(device, dtype=torch.bfloat16)) / \
                      latents_std.to(device, dtype=torch.bfloat16)
            text_embeds = batch["text_embed"].to(device, dtype=torch.bfloat16)

            # CFG dropout: randomly replace text with null embeddings
            B = latents.shape[0]
            drop_mask = torch.rand(B, device=device) < cfg_dropout
            if drop_mask.any():
                null_embeds = torch.zeros_like(text_embeds)
                text_embeds = torch.where(
                    drop_mask[:, None, None].expand_as(text_embeds),
                    null_embeds, text_embeds,
                )

            # Sample timesteps (flow matching: uniform [0, 1])
            timesteps = torch.rand(B, device=device, dtype=torch.bfloat16)

            # Sample noise and interpolate
            noise = torch.randn_like(latents)
            t_expand = timesteps[:, None, None, None, None]
            noisy_latents = (1 - t_expand) * latents + t_expand * noise
            target_velocity = noise - latents

            # Forward
            velocity_pred = transformer(
                noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=text_embeds,
            ).sample

            loss = F.mse_loss(velocity_pred.float(), target_velocity.float())

            # Backward
            if accelerator is not None:
                accelerator.backward(loss)
            else:
                loss.backward()

            if grad_clip > 0:
                if accelerator is not None:
                    accelerator.clip_grad_norm_(trainable_params, grad_clip)
                else:
                    torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)

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
                ckpt_dir = run_dir / f"step_{global_step:06d}"
                ckpt_dir.mkdir(exist_ok=True)

                unwrapped = accelerator.unwrap_model(transformer) if accelerator else transformer
                unwrapped.save_pretrained(str(ckpt_dir / "lora_transformer"))
                log_rank0(f"Saved checkpoint: {ckpt_dir}", is_main)

    # Final save
    if is_main:
        final_dir = run_dir / "final"
        final_dir.mkdir(exist_ok=True)
        unwrapped = accelerator.unwrap_model(transformer) if accelerator else transformer
        unwrapped.save_pretrained(str(final_dir / "lora_transformer"))
        log_rank0(f"Training complete. Final checkpoint: {final_dir}", is_main)


if __name__ == "__main__":
    main()
