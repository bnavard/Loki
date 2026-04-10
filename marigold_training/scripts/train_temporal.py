"""
Stage 2: Temporal Marigold training (video, T=81).

Loads the Stage 1 (spatial) checkpoint and fine-tunes on video pairs
(natural video → deformation map video). The model already understands
spatial mapping from Stage 1 — Stage 2 teaches temporal coherence.

Uses a lower learning rate than Stage 1 to preserve the spatial prior.

Usage:
    cd <repo_root>

    PYTHONPATH=. accelerate launch --num_processes 4 --mixed_precision bf16 \
        marigold_training/scripts/train_temporal.py \
        --config marigold_training/configs/temporal_config.yaml
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

from marigold_training.src.marigold_dataset import MarigoldDataset
from marigold_training.src.marigold_model import double_patch_embedding
from marigold_training.src.collate import collate_fn
from marigold_training.src.checkpoint import save_checkpoint
from marigold_training.src.vis import generate_eval_sample, save_eval_grid, visualize_deform

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


def log_rank0(msg, is_main):
    if is_main:
        logger.info(msg)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    return p.parse_args()


def _save_lr_plot(lr, max_steps, warmup_steps, decay_iters, lr_min_ratio, run_dir):
    """Save a learning rate vs step plot for the IterExponential schedule."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    steps = np.arange(max_steps)
    lr_values = np.zeros(max_steps)
    for s in steps:
        if s < warmup_steps:
            lr_values[s] = lr * s / max(warmup_steps, 1)
        else:
            decay_step = s - warmup_steps
            lr_values[s] = lr * math.exp(math.log(lr_min_ratio) * decay_step / decay_iters)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, lr_values, linewidth=1.5)
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Learning Rate", fontsize=12)
    ax.set_title(
        f"LR Schedule: warmup {warmup_steps} steps → exp decay to "
        f"{lr_min_ratio*100:.0f}% over {decay_iters} iters",
        fontsize=12,
    )
    ax.set_xlim(0, max_steps)
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(-4, -4))
    ax.grid(True, alpha=0.3)
    ax.axvline(x=warmup_steps, color="r", linestyle="--", alpha=0.5, label=f"warmup end ({warmup_steps})")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(str(Path(run_dir) / "learning_rate_vs_step.png"), dpi=150)
    plt.close(fig)
    logger.info(f"LR plot saved: {run_dir}/learning_rate_vs_step.png")


def load_config(config_path):
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_stage1_checkpoint(transformer, checkpoint_path, is_main):
    """
    Load Stage 1 (spatial) weights into the transformer.

    Option A (default): Direct loading — Stage 1 weights already have the
    correct shape (32ch input from Marigold's doubling). The temporal kernel
    weights were updated during Stage 1 even though T=1, because gradients
    still flow through them.
    """
    from safetensors.torch import load_file

    checkpoint_path = Path(checkpoint_path)
    transformer_dir = checkpoint_path / "transformer"

    safetensor_file = transformer_dir / "diffusion_pytorch_model.safetensors"
    if not safetensor_file.exists():
        safetensor_file = transformer_dir / "model.safetensors"

    if not safetensor_file.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {transformer_dir}. "
            f"Expected diffusion_pytorch_model.safetensors or model.safetensors"
        )

    if is_main:
        logger.info(f"Loading Stage 1 checkpoint: {safetensor_file}")

    state_dict = load_file(str(safetensor_file))
    transformer.load_state_dict(state_dict, strict=True)

    return transformer


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
            project_dir=cfg.get("output_dir", "outputs/marigold_temporal"),
        )
        device = accelerator.device
        is_main = accelerator.is_main_process
    except ImportError:
        accelerator = None
        device = torch.device(f"cuda:{cfg.get('gpu', 0)}")
        is_main = True

    # ---- Run directory ----
    output_dir = Path(cfg.get("output_dir", "outputs/marigold_temporal"))
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
    log_rank0(f"Loading base pipeline: {model_id}...", is_main)

    pipe = WanPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    transformer = pipe.transformer

    vae = pipe.vae.to(device).eval()
    vae.requires_grad_(False)

    latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
    latents_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)

    del pipe.text_encoder, pipe.tokenizer
    del pipe
    torch.cuda.empty_cache()

    # ---- Double input layer (must happen BEFORE loading Stage 1 weights) ----
    transformer = double_patch_embedding(transformer)
    log_rank0(f"Doubled patch_embedding: {transformer.patch_embedding}", is_main)

    # ---- Load Stage 1 (spatial) checkpoint ----
    stage1_checkpoint = cfg.get("stage1_checkpoint")
    if stage1_checkpoint is None:
        raise ValueError("stage1_checkpoint must be set in config (path to Stage 1 step directory)")
    transformer = load_stage1_checkpoint(transformer, stage1_checkpoint, is_main)
    log_rank0("Stage 1 spatial weights loaded", is_main)

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

    # ---- Dataset (video pairs, T=81) ----
    dataset = MarigoldDataset(
        manifest_path=cfg.get("manifest_path", "data/derived/manifest.json"),
        flame_root=cfg.get("flame_root", "data/flowface"),
        video_root=cfg.get("video_root", "data/talkvid/talkvid"),
        target_frames=cfg.get("target_frames", 81),
        resolution=cfg.get("resolution", 512),
        prompt_latent_cache_dir=cfg.get("prompt_latent_cache_dir"),
    )
    log_rank0(f"Dataset: {len(dataset)} video clips", is_main)

    dataloader = DataLoader(
        dataset, batch_size=cfg.get("batch_size", 1),
        shuffle=True, num_workers=0, pin_memory=False,
        drop_last=True, collate_fn=collate_fn,
    )

    # ---- Optimizer (lower LR than Stage 1 to preserve spatial prior) ----
    trainable_params = [p for p in transformer.parameters() if p.requires_grad]
    max_steps = cfg.get("max_steps", 25000)
    lr = cfg.get("lr", 3e-6)

    optimizer = torch.optim.AdamW(
        trainable_params, lr=lr,
        weight_decay=cfg.get("weight_decay", 0.01),
    )

    # ---- LR scheduler: IterExponential ----
    warmup_steps = cfg.get("warmup_steps", 100)
    decay_iters = cfg.get("decay_iters", 25000)
    lr_min_ratio = cfg.get("lr_min_ratio", 0.01)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        decay_step = step - warmup_steps
        return math.exp(math.log(lr_min_ratio) * decay_step / decay_iters)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---- Save LR schedule plot ----
    if is_main:
        _save_lr_plot(lr, max_steps, warmup_steps, decay_iters, lr_min_ratio, run_dir)

    # ---- Accelerate prepare ----
    if accelerator is not None:
        transformer, optimizer, dataloader, lr_scheduler = accelerator.prepare(
            transformer, optimizer, dataloader, lr_scheduler,
        )

    # ---- Training loop (identical to Stage 1 but with video tensors) ----
    cfg_dropout = cfg.get("cfg_dropout", 0.1)
    grad_clip = cfg.get("grad_clip", 1.0)
    save_every = cfg.get("save_every", 500)
    log_every = cfg.get("log_every", 1)
    eval_every = cfg.get("eval_every", save_every)

    eval_batch = None

    global_step = 0
    transformer.train()
    log_rank0(f"Starting Stage 2 (temporal) training for {max_steps} steps...", is_main)

    while global_step < max_steps:
        for batch in dataloader:
            if global_step >= max_steps:
                break

            # ---- VAE encode both videos ----
            with torch.no_grad():
                natural_5d = batch["natural_video"]
                target_5d = batch["target_video"]
                if natural_5d.ndim == 4:
                    natural_5d = natural_5d.unsqueeze(0)
                if target_5d.ndim == 4:
                    target_5d = target_5d.unsqueeze(0)

                # [B, 3, T, H, W] → [B, 16, T_lat, h, w]
                natural_latent = vae.encode(
                    natural_5d.to(device=device, dtype=vae.dtype)
                ).latent_dist.mode()
                target_latent = vae.encode(
                    target_5d.to(device=device, dtype=vae.dtype)
                ).latent_dist.mode()

                natural_latent = (natural_latent - latents_mean.to(device, dtype=natural_latent.dtype)) / \
                                 latents_std.to(device, dtype=natural_latent.dtype)
                target_latent = (target_latent - latents_mean.to(device, dtype=target_latent.dtype)) / \
                                latents_std.to(device, dtype=target_latent.dtype)

            natural_latent = natural_latent.to(dtype=torch.bfloat16)
            target_latent = target_latent.to(dtype=torch.bfloat16)
            text_embeds = batch["text_embed"].to(device, dtype=torch.bfloat16)

            B = target_latent.shape[0]

            # ---- CFG dropout ----
            drop_mask = torch.rand(B, device=device) < cfg_dropout
            if drop_mask.any():
                text_embeds = torch.where(
                    drop_mask[:, None, None].expand_as(text_embeds),
                    torch.zeros_like(text_embeds), text_embeds,
                )

            # ---- Flow matching ----
            noise = torch.randn_like(target_latent)
            t = torch.rand(B, device=device, dtype=torch.bfloat16)
            t_expand = t[:, None, None, None, None]

            noisy_target = (1 - t_expand) * target_latent + t_expand * noise
            target_velocity = noise - target_latent

            model_input = torch.cat([noisy_target, natural_latent], dim=1)
            # Shape: [B, 32, T_lat, h, w]

            velocity_pred = transformer(
                model_input, timestep=t,
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
                    f"loss={loss.item():.6f} | "
                    f"lr={lr_scheduler.get_last_lr()[0]:.2e}",
                    is_main,
                )

            if is_main and global_step % save_every == 0:
                ckpt = save_checkpoint(transformer, global_step, run_dir, accelerator)
                log_rank0(f"Saved checkpoint: {ckpt}", is_main)

            # ---- Periodic eval: generate a deformation video from a held sample ----
            if is_main and global_step % eval_every == 0:
                if eval_batch is None:
                    eval_batch = {
                        "natural_video": batch["natural_video"][:1].clone(),
                        "target_video": batch["target_video"][:1].clone(),
                        "text_embed": batch["text_embed"][:1].clone(),
                    }

                with torch.no_grad():
                    eval_nat = eval_batch["natural_video"]
                    if eval_nat.ndim == 4:
                        eval_nat = eval_nat.unsqueeze(0)
                    eval_cond_latent = vae.encode(
                        eval_nat.to(device=device, dtype=vae.dtype)
                    ).latent_dist.mode()
                    eval_cond_latent = (eval_cond_latent - latents_mean.to(device, dtype=eval_cond_latent.dtype)) / \
                                       latents_std.to(device, dtype=eval_cond_latent.dtype)
                    eval_cond_latent = eval_cond_latent.to(dtype=torch.bfloat16)

                eval_text = eval_batch["text_embed"][:1].to(device, dtype=torch.bfloat16)

                unwrapped = accelerator.unwrap_model(transformer) if accelerator else transformer
                pred_deform = generate_eval_sample(
                    unwrapped, vae, eval_cond_latent, eval_text,
                    latents_mean, latents_std,
                )

                # GT deformation (pixel space)
                gt_target = eval_batch["target_video"][0]  # [3, T, H, W]
                gt_deform = gt_target.permute(1, 0, 2, 3).cpu()  # [T, 3, H, W]

                eval_dir = run_dir / "eval" / f"step_{global_step:06d}"

                # Save middle-frame comparison grid
                save_eval_grid(
                    eval_batch["natural_video"][0].cpu(),
                    pred_deform,
                    gt_deform,
                    eval_dir / "comparison.png",
                )

                # Save predicted deformation as video
                visualize_deform(pred_deform, eval_dir / "predicted", fps=25, verbose=False)
                visualize_deform(gt_deform, eval_dir / "ground_truth", fps=25, verbose=False)

                log_rank0(f"Eval saved: {eval_dir}", is_main)

    if is_main:
        ckpt = save_checkpoint(transformer, global_step, run_dir, accelerator)
        log_rank0(f"Stage 2 complete. Final checkpoint: {ckpt}", is_main)


if __name__ == "__main__":
    main()
