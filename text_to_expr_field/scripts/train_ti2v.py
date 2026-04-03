"""
Train text+image-to-expression-field model using Wan2.2-TI2V-5B.

Uses the smaller 5B model (single transformer, not MoE) conditioned on both
text captions and a reference face image. The reference image is loaded from
data/flowface/{clip_id}/images/cam0/00001.jpg during training.

Both VAE latents and text embeddings are loaded from precomputed caches.
The text encoder and VAE are never loaded during training.

Usage:
    cd /data/pouyan/baseline/repository/cap4d

    # Single GPU:
    PYTHONPATH=. python text_to_expr_field/scripts/train_ti2v.py \
        --config text_to_expr_field/configs/train_config_ti2v.yaml

    # Multi-GPU:
    PYTHONPATH=. accelerate launch --num_processes 4 --mixed_precision bf16 \
        text_to_expr_field/scripts/train_ti2v.py \
        --config text_to_expr_field/configs/train_config_ti2v.yaml

    # With DeepSpeed ZeRO-2:
    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --use_deepspeed --zero_stage 2 \
        --num_processes 4 --mixed_precision bf16 \
        text_to_expr_field/scripts/train_ti2v.py \
        --config text_to_expr_field/configs/train_config_ti2v.yaml
"""

import argparse
import logging
import math
import os
import shutil
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image

torch.backends.cudnn.enabled = False

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


def load_reference_image(clip_id, flame_root, image_processor, device):
    """
    Load the reference face image for a clip from flowface data.
    Returns the processed image tensor ready for the pipeline's image encoder.
    """
    img_path = Path(flame_root) / clip_id / "images" / "cam0" / "00001.jpg"
    if not img_path.exists():
        # Fallback to first available frame
        cam_dir = Path(flame_root) / clip_id / "images" / "cam0"
        frames = sorted(cam_dir.glob("*.jpg"))
        if not frames:
            return None
        img_path = frames[0]

    image = Image.open(str(img_path)).convert("RGB")
    # Process through the pipeline's image processor
    processed = image_processor(image, return_tensors="pt")
    if isinstance(processed, dict):
        pixel_values = processed.get("pixel_values", processed.get("images"))
    else:
        pixel_values = processed
    return pixel_values.to(device, dtype=torch.bfloat16)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    try:
        from accelerate import Accelerator
        accelerator = Accelerator(
            mixed_precision=cfg.get("mixed_precision", "bf16"),
            gradient_accumulation_steps=cfg.get("gradient_accumulation", 4),
            log_with="tensorboard",
            project_dir=cfg.get("output_dir", "outputs/text_to_expr_field_ti2v"),
        )
        device = accelerator.device
        is_main = accelerator.is_main_process
    except ImportError:
        accelerator = None
        device = torch.device(f"cuda:{cfg.get('gpu', 0)}")
        is_main = True

    # Timestamped run directory
    output_dir = Path(cfg.get("output_dir", "outputs/text_to_expr_field_ti2v"))
    if is_main:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = output_dir / f"run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.config, run_dir / "config.yaml")
        log_rank0(f"Run directory: {run_dir}", is_main)

        marker = output_dir / ".current_run_dir"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(str(run_dir))
    else:
        import time
        marker = output_dir / ".current_run_dir"
        for _ in range(60):
            if marker.exists():
                break
            time.sleep(0.5)
        run_dir = Path(marker.read_text().strip())

    # ---- Load Wan2.2-TI2V-5B ----
    model_id = cfg.get("model_id", "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    log_rank0(f"Loading {model_id}...", is_main)

    from diffusers import WanImageToVideoPipeline

    pipe = WanImageToVideoPipeline.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
    )

    transformer = pipe.transformer
    scheduler = pipe.scheduler

    # Keep the image processor for encoding reference images
    image_processor = pipe.image_processor if hasattr(pipe, "image_processor") else None
    # Keep the image encoder for encoding reference images during training
    image_encoder = None
    if hasattr(pipe, "image_encoder") and pipe.image_encoder is not None:
        image_encoder = pipe.image_encoder.to(device).eval()
        image_encoder.requires_grad_(False)

    # Free text encoder + VAE — we use cached embeddings and latents
    del pipe.text_encoder, pipe.tokenizer, pipe.vae
    torch.cuda.empty_cache()
    log_rank0("Freed text encoder + VAE — using cached embeddings and latents", is_main)

    # ---- Apply LoRA ----
    from peft import LoraConfig, get_peft_model

    lora_rank = cfg.get("lora_rank", 64)
    lora_alpha = cfg.get("lora_alpha", 64)

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
    if is_main:
        transformer.print_trainable_parameters()

    # ---- Dataset ----
    from text_to_expr_field.src.dataset import ExprFieldDataset

    dataset = ExprFieldDataset(
        manifest_path=cfg.get("manifest_path", "data/derived/manifest.json"),
        flame_root=cfg.get("flame_root", "data/flowface"),
        target_frames=cfg.get("target_frames", 80),
        resolution=cfg.get("resolution", 512),
        vae=None,
        device=str(device),
        vae_latent_cache_dir=cfg.get("vae_latent_cache_dir", "data/derived/vae_latent_cache"),
        prompt_latent_cache_dir=cfg.get("prompt_latent_cache_dir", "data/derived/text_embed_cache"),
        cached_only=cfg.get("cached_only", True),
    )
    log_rank0(f"Dataset: {len(dataset)} clips", is_main)

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.get("batch_size", 1),
        shuffle=True,
        num_workers=cfg.get("num_workers", 2),
        pin_memory=True,
        drop_last=True,
    )

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        transformer.parameters(),
        lr=cfg.get("lr", 1e-5),
        weight_decay=cfg.get("weight_decay", 0.01),
    )

    max_steps = cfg.get("max_steps", 20000)
    warmup_steps = cfg.get("warmup_steps", 500)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---- Accelerate setup ----
    if accelerator is not None:
        transformer, optimizer, dataloader, lr_scheduler = accelerator.prepare(
            transformer, optimizer, dataloader, lr_scheduler,
        )

    # ---- Training loop ----
    cfg_dropout = cfg.get("cfg_dropout", 0.1)
    grad_clip = cfg.get("grad_clip", 1.0)
    save_every = cfg.get("save_every", 2000)
    log_every = cfg.get("log_every", 50)
    flame_root = cfg.get("flame_root", "data/flowface")

    global_step = 0
    transformer.train()

    log_rank0(f"Starting training for {max_steps} steps...", is_main)

    while global_step < max_steps:
        for batch in dataloader:
            if global_step >= max_steps:
                break

            latents = batch["latent"].to(device, dtype=torch.bfloat16)
            text_embeds = batch["text_embed"].to(device, dtype=torch.bfloat16)
            clip_ids = batch["clip_id"]
            B = latents.shape[0]

            # Load and encode reference images for this batch
            # The image provides identity context (what the person looks like)
            image_embeds_list = []
            for clip_id in clip_ids:
                ref_img = load_reference_image(clip_id, flame_root, image_processor, device)
                if ref_img is not None and image_encoder is not None:
                    with torch.no_grad():
                        img_embed = image_encoder(ref_img).image_embeds
                    image_embeds_list.append(img_embed.squeeze(0))
                else:
                    # Null image embedding if no image available
                    image_embeds_list.append(torch.zeros(
                        text_embeds.shape[-1], device=device, dtype=torch.bfloat16
                    ))
            image_embeds = torch.stack(image_embeds_list, dim=0)  # (B, dim)

            # CFG dropout: randomly drop text and/or image conditioning
            drop_text = torch.rand(B, device=device) < cfg_dropout
            drop_image = torch.rand(B, device=device) < cfg_dropout
            if drop_text.any():
                text_embeds = torch.where(
                    drop_text[:, None, None].expand_as(text_embeds),
                    torch.zeros_like(text_embeds),
                    text_embeds,
                )
            if drop_image.any():
                image_embeds = torch.where(
                    drop_image[:, None].expand_as(image_embeds),
                    torch.zeros_like(image_embeds),
                    image_embeds,
                )

            # Flow matching
            timesteps = torch.rand(B, device=device, dtype=torch.bfloat16)
            noise = torch.randn_like(latents)
            t_expand = timesteps[:, None, None, None, None]
            noisy_latents = (1 - t_expand) * latents + t_expand * noise
            target_velocity = noise - latents

            # Forward pass
            velocity_pred = transformer(
                noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=text_embeds,
                image_embeds=image_embeds,
            ).sample

            loss = F.mse_loss(velocity_pred, target_velocity)

            if accelerator is not None:
                accelerator.backward(loss)
            else:
                loss.backward()

            if grad_clip > 0:
                params = list(transformer.parameters())
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
                ckpt_dir = run_dir / f"step_{global_step:06d}"
                ckpt_dir.mkdir(exist_ok=True)
                unwrapped = accelerator.unwrap_model(transformer) if accelerator else transformer
                unwrapped.save_pretrained(str(ckpt_dir / "lora_transformer"))
                log_rank0(f"Saved checkpoint: {ckpt_dir}", is_main)

    if is_main:
        final_dir = run_dir / "final"
        final_dir.mkdir(exist_ok=True)
        unwrapped = accelerator.unwrap_model(transformer) if accelerator else transformer
        unwrapped.save_pretrained(str(final_dir / "lora_transformer"))
        log_rank0(f"Training complete. Final checkpoint: {final_dir}", is_main)


if __name__ == "__main__":
    main()
