"""
Marigold-style training: natural video → deformation map video.

Adapts the Marigold depth estimation approach (Ke et al., CVPR 2024) to video:
the DiT's input layer is doubled from 16 to 32 channels to accept concatenated
[noisy_target_deform_latent, clean_natural_video_latent]. The natural video
provides spatiotemporal anchoring at every denoising step.

Both videos are VAE-encoded on the fly. The VAE is frozen. Text conditioning
uses cached UMT5 embeddings (prosody captions, not empty text).

Full fine-tuning of the transformer (not LoRA).

Usage:
    cd /data/pouyan/baseline/repository/cap4d

    PYTHONPATH=. accelerate launch --num_processes 4 --mixed_precision bf16 \
        text_to_expr_field/scripts/train_marigold.py \
        --config text_to_expr_field/configs/train_marigold_config.yaml
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

from text_to_expr_field.src.data.marigold_dataset import MarigoldDataset
from text_to_expr_field.src.data.collate import collate_fn
from text_to_expr_field.src.model.marigold import double_patch_embedding
from text_to_expr_field.src.model.checkpoint import save_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


def log_rank0(msg, is_main):
    if is_main:
        logger.info(msg)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    return p.parse_args()


def load_config(config_path):
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    cfg = load_config(args.config)

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
            project_dir=cfg.get("output_dir", "outputs/marigold_deform"),
        )
        device = accelerator.device
        is_main = accelerator.is_main_process
    except ImportError:
        accelerator = None
        device = torch.device(f"cuda:{cfg.get('gpu', 0)}")
        is_main = True

    # ---- Run directory ----
    output_dir = Path(cfg.get("output_dir", "outputs/marigold_deform"))
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

    # ---- Load pipeline ----
    from diffusers import WanPipeline

    model_id = cfg.get("model_id", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    log_rank0(f"Loading pipeline: {model_id}...", is_main)

    pipe = WanPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)

    transformer = pipe.transformer
    scheduler = pipe.scheduler

    # Frozen VAE for on-the-fly encoding of both natural video and deform video
    vae = pipe.vae.to(device).eval()
    vae.requires_grad_(False)

    # Latent normalization stats
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
    latents_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)

    # Free text encoder — we use cached embeddings
    del pipe.text_encoder, pipe.tokenizer
    del pipe
    torch.cuda.empty_cache()

    # ---- Marigold: double the input layer ----
    # Following Marigold's _replace_unet_conv_in(): clone patch_embedding weights,
    # repeat along input channel dim (16 → 32), scale by 0.5.
    log_rank0(f"Original patch_embedding: {transformer.patch_embedding}", is_main)
    transformer = double_patch_embedding(transformer)
    log_rank0(f"Doubled patch_embedding: {transformer.patch_embedding}", is_main)

    # ---- Memory optimizations ----
    if cfg.get("gradient_checkpointing", True):
        if hasattr(transformer, "enable_gradient_checkpointing"):
            transformer.enable_gradient_checkpointing()
        log_rank0("Gradient checkpointing enabled", is_main)

    # ---- Full fine-tuning ----
    transformer = transformer.to(device)
    transformer.requires_grad_(True)
    total_params = sum(p.numel() for p in transformer.parameters())
    log_rank0(f"Full fine-tuning: {total_params:,} params", is_main)

    # ---- Dataset ----
    dataset = MarigoldDataset(
        manifest_path=cfg.get("manifest_path", "data/derived/manifest.json"),
        flame_root=cfg.get("flame_root", "data/flowface"),
        video_root=cfg.get("video_root", "data/talkvid/talkvid"),
        target_frames=cfg.get("target_frames", 81),
        resolution=cfg.get("resolution", 512),
        prompt_latent_cache_dir=cfg.get("prompt_latent_cache_dir"),
    )
    log_rank0(f"Dataset: {len(dataset)} clips", is_main)

    # num_workers=0: dataset uses CUDA (PyTorch3D rasterization)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.get("batch_size", 1),
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
        collate_fn=collate_fn,
    )

    # ---- Optimizer ----
    trainable_params = [p for p in transformer.parameters() if p.requires_grad]
    max_steps = cfg.get("max_steps", 20000)
    lr = cfg.get("lr", 5e-6)

    optimizer = torch.optim.AdamW(
        trainable_params, lr=lr,
        weight_decay=cfg.get("weight_decay", 0.01),
    )

    # ---- LR scheduler: IterExponential ----
    # Linear warmup for warmup_steps, then exponential decay over decay_iters
    # to lr_min_ratio of initial LR.
    warmup_steps = cfg.get("warmup_steps", 100)
    decay_iters = cfg.get("decay_iters", 25000)
    lr_min_ratio = cfg.get("lr_min_ratio", 0.01)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        decay_step = step - warmup_steps
        return math.exp(math.log(lr_min_ratio) * decay_step / decay_iters)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---- Accelerate prepare ----
    if accelerator is not None:
        transformer, optimizer, dataloader, lr_scheduler = accelerator.prepare(
            transformer, optimizer, dataloader, lr_scheduler,
        )

    # ---- Training loop ----
    cfg_dropout = cfg.get("cfg_dropout", 0.1)
    grad_clip = cfg.get("grad_clip", 1.0)
    save_every = cfg.get("save_every", 250)
    log_every = cfg.get("log_every", 1)

    global_step = 0
    transformer.train()
    log_rank0(f"Starting Marigold training for {max_steps} steps...", is_main)

    while global_step < max_steps:
        for batch in dataloader:
            if global_step >= max_steps:
                break

            # ---- VAE encode both videos ----
            # Following Marigold: encode both through the same frozen VAE,
            # use deterministic encoding (mode, not sample).
            with torch.no_grad():
                natural_5d = batch["natural_video"].unsqueeze(0) if batch["natural_video"].ndim == 4 \
                    else batch["natural_video"]
                target_5d = batch["target_video"].unsqueeze(0) if batch["target_video"].ndim == 4 \
                    else batch["target_video"]

                # [B, 3, T, H, W] → VAE → [B, C_latent, T_latent, h, w]
                natural_latent = vae.encode(
                    natural_5d.to(device=device, dtype=vae.dtype)
                ).latent_dist.mode()
                target_latent = vae.encode(
                    target_5d.to(device=device, dtype=vae.dtype)
                ).latent_dist.mode()

                # Normalize both with the same pretrained stats
                natural_latent = (natural_latent - latents_mean.to(device, dtype=natural_latent.dtype)) / \
                                 latents_std.to(device, dtype=natural_latent.dtype)
                target_latent = (target_latent - latents_mean.to(device, dtype=target_latent.dtype)) / \
                                latents_std.to(device, dtype=target_latent.dtype)

            # Cast to training dtype
            natural_latent = natural_latent.to(dtype=torch.bfloat16)
            target_latent = target_latent.to(dtype=torch.bfloat16)
            text_embeds = batch["text_embed"].to(device, dtype=torch.bfloat16)

            B = target_latent.shape[0]

            # ---- CFG dropout: randomly drop text conditioning ----
            drop_mask = torch.rand(B, device=device) < cfg_dropout
            if drop_mask.any():
                text_embeds = torch.where(
                    drop_mask[:, None, None].expand_as(text_embeds),
                    torch.zeros_like(text_embeds), text_embeds,
                )

            # ---- Flow matching: noise the TARGET latent only ----
            # The natural video latent is ALWAYS clean (Marigold's key principle).
            noise = torch.randn_like(target_latent)
            t = torch.rand(B, device=device, dtype=torch.bfloat16)
            t_expand = t[:, None, None, None, None]

            noisy_target = (1 - t_expand) * target_latent + t_expand * noise
            target_velocity = noise - target_latent

            # ---- Marigold concatenation ----
            # [noisy_target | clean_natural_video] along channel dim
            # Following Marigold's cat order: conditioning first in their code,
            # but the instruction doc puts noisy first. We follow the instruction
            # doc since the weight init treats both halves identically (repeat + /2).
            model_input = torch.cat([noisy_target, natural_latent], dim=1)
            # Shape: [B, 32, T_latent, h, w]

            # ---- Forward: DiT predicts velocity ----
            velocity_pred = transformer(
                model_input, timestep=t,
                encoder_hidden_states=text_embeds,
            ).sample

            # ---- Loss (fp32 for numerical stability) ----
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
                    f"loss={loss.item():.6f} | "
                    f"lr={lr_scheduler.get_last_lr()[0]:.2e}",
                    is_main,
                )

            if is_main and global_step % save_every == 0:
                ckpt = save_checkpoint(
                    transformer, global_step, run_dir,
                    use_lora=False, accelerator=accelerator,
                )
                log_rank0(f"Saved checkpoint: {ckpt}", is_main)

    # Final save
    if is_main:
        ckpt = save_checkpoint(
            transformer, global_step, run_dir,
            use_lora=False, accelerator=accelerator,
        )
        log_rank0(f"Training complete. Final checkpoint: {ckpt}", is_main)


if __name__ == "__main__":
    main()
