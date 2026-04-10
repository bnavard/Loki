"""
Marigold training with SD3.5 Medium: natural face frame → deformation map.

Uses SD3Transformer2DModel (24-layer MMDiT, ~1.5B params) with rectified flow.
The input layer is doubled from 16 to 32 channels to accept concatenated
[noisy_target_deform | clean_natural_frame] latents.

Text conditioning uses cached UMT5 embeddings (4096-dim, compatible with
SD3's T5-XXL joint_attention_dim). CLIP pooled projections are zeroed.

Full fine-tuning of all transformer parameters.

Usage:
    cd <repo_root>

    PYTHONPATH=. accelerate launch --num_processes 4 --mixed_precision bf16 \
        marigold_training/scripts/train_spatial.py \
        --config marigold_training/configs/spatial_config.yaml

    # Resume:
    PYTHONPATH=. accelerate launch --num_processes 4 --mixed_precision bf16 \
        marigold_training/scripts/train_spatial.py \
        --config marigold_training/configs/spatial_config.yaml \
        --resume outputs/marigold_spatial/run_YYYYMMDD/step_001000
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

from marigold_training.src.frame_pair_dataset import FramePairDataset
from marigold_training.src.marigold_model import double_input_channels
from marigold_training.src.collate import collate_fn
from marigold_training.src.checkpoint import save_checkpoint, load_training_state
from marigold_training.src.vis import save_eval_grid

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


def log_rank0(msg, is_main):
    if is_main:
        logger.info(msg)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--resume", default=None,
                   help="Path to checkpoint step directory to resume from")
    return p.parse_args()


def load_config(config_path):
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def _save_lr_plot(lr, max_steps, warmup_steps, decay_iters, lr_min_ratio, run_dir):
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
        f"LR Schedule: warmup {warmup_steps} steps \u2192 exp decay to "
        f"{lr_min_ratio*100:.0f}% over {decay_iters} iters", fontsize=12)
    ax.set_xlim(0, max_steps)
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(-4, -4))
    ax.grid(True, alpha=0.3)
    ax.axvline(x=warmup_steps, color="r", linestyle="--", alpha=0.5, label=f"warmup end ({warmup_steps})")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(str(Path(run_dir) / "learning_rate_vs_step.png"), dpi=150)
    plt.close(fig)
    logger.info(f"LR plot saved: {run_dir}/learning_rate_vs_step.png")


@torch.no_grad()
def _run_eval_inference(
    transformer, vae, cond_latent, text_embeds, pooled_projections,
    scaling_factor, shift_factor, num_steps=50,
):
    """Euler denoising for eval. Returns [3, H, W] decoded deform."""
    device = cond_latent.device
    was_training = transformer.training
    transformer.eval()

    x = torch.randn_like(cond_latent)
    timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

    for i in range(num_steps):
        t_current = timesteps[i]
        dt = timesteps[i + 1] - timesteps[i]

        model_input = torch.cat([x, cond_latent], dim=1)
        t_ms = (t_current * 1000).long().expand(cond_latent.shape[0])

        velocity = transformer(
            hidden_states=model_input,
            timestep=t_ms,
            encoder_hidden_states=text_embeds,
            pooled_projections=pooled_projections,
        ).sample

        x = x + velocity * dt

    raw_latent = x.float() / scaling_factor + shift_factor
    decoded = vae.decode(raw_latent.to(vae.dtype), return_dict=False)[0]
    result = decoded.float().cpu()  # [B, 3, H, W]

    if was_training:
        transformer.train()

    return result


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
            project_dir=cfg.get("output_dir", "outputs/marigold_spatial"),
        )
        device = accelerator.device
        is_main = accelerator.is_main_process
    except ImportError:
        accelerator = None
        device = torch.device(f"cuda:{cfg.get('gpu', 0)}")
        is_main = True

    # ---- Run directory ----
    output_dir = Path(cfg.get("output_dir", "outputs/marigold_spatial"))
    if args.resume:
        resume_dir = Path(args.resume)
        run_dir = resume_dir.parent
        run_dir.mkdir(parents=True, exist_ok=True)
        if is_main:
            marker = output_dir / ".current_run_dir"
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(str(run_dir))
            log_rank0(f"Resuming into: {run_dir}", is_main)
    elif is_main:
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

    # ---- Load SD3.5 components ----
    from diffusers import SD3Transformer2DModel, AutoencoderKL

    model_id = cfg.get("model_id", "stabilityai/stable-diffusion-3.5-medium")
    log_rank0(f"Loading transformer from {model_id}...", is_main)

    transformer = SD3Transformer2DModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=torch.bfloat16,
    )

    log_rank0("Loading VAE...", is_main)
    vae = AutoencoderKL.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float32,
    ).to(device).eval()
    vae.requires_grad_(False)

    scaling_factor = vae.config.scaling_factor
    shift_factor = vae.config.shift_factor
    pooled_dim = transformer.config["pooled_projection_dim"]
    log_rank0(f"VAE: scaling_factor={scaling_factor}, shift_factor={shift_factor}", is_main)

    # ---- Marigold: double input layer (16 → 32) ----
    log_rank0(f"Original pos_embed.proj: {transformer.pos_embed.proj}", is_main)
    transformer = double_input_channels(transformer)
    log_rank0(f"Doubled pos_embed.proj: {transformer.pos_embed.proj}", is_main)

    # ---- Resume: load model weights (after doubling, before optimizer) ----
    if args.resume:
        from safetensors.torch import load_file
        resume_dir = Path(args.resume)
        sf = resume_dir / "transformer" / "diffusion_pytorch_model.safetensors"
        if not sf.exists():
            sf = resume_dir / "transformer" / "model.safetensors"
        log_rank0(f"Loading model weights: {sf}", is_main)
        state_dict = load_file(str(sf))
        transformer.load_state_dict(state_dict, strict=True)

    # ---- Memory optimizations ----
    if cfg.get("gradient_checkpointing", True):
        if hasattr(transformer, "enable_gradient_checkpointing"):
            transformer.enable_gradient_checkpointing()
        log_rank0("Gradient checkpointing enabled", is_main)

    transformer = transformer.to(device)
    transformer.requires_grad_(True)
    total_params = sum(p.numel() for p in transformer.parameters())
    log_rank0(f"Full fine-tuning: {total_params:,} params", is_main)

    # ---- Dataset ----
    dataset = FramePairDataset(
        manifest_path=cfg.get("manifest_path", "data/derived/manifest.json"),
        flame_root=cfg.get("flame_root", "data/flowface"),
        resolution=cfg.get("resolution", 512),
        min_frames=cfg.get("min_frames", 10),
    )
    log_rank0(f"Dataset: {len(dataset)} clips", is_main)

    dataloader = DataLoader(
        dataset, batch_size=cfg.get("batch_size", 8),
        shuffle=True, num_workers=0, pin_memory=False,
        drop_last=True, collate_fn=collate_fn,
    )

    # ---- Optimizer ----
    trainable_params = [p for p in transformer.parameters() if p.requires_grad]
    max_steps = cfg.get("max_steps", 50000)
    lr = cfg.get("lr", 1e-5)

    optimizer = torch.optim.AdamW(
        trainable_params, lr=lr,
        weight_decay=cfg.get("weight_decay", 0.01),
    )

    # ---- LR scheduler ----
    warmup_steps = cfg.get("warmup_steps", 100)
    decay_iters = cfg.get("decay_iters", 50000)
    lr_min_ratio = cfg.get("lr_min_ratio", 0.01)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        decay_step = step - warmup_steps
        return math.exp(math.log(lr_min_ratio) * decay_step / decay_iters)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---- Resume: restore optimizer + scheduler + RNG state ----
    resume_step = 0
    if args.resume:
        resume_step = load_training_state(
            Path(args.resume), optimizer=optimizer, lr_scheduler=lr_scheduler,
        )
        log_rank0(f"Restored training state at step {resume_step}, lr={lr_scheduler.get_last_lr()[0]:.2e}", is_main)

    if is_main:
        _save_lr_plot(lr, max_steps, warmup_steps, decay_iters, lr_min_ratio, run_dir)

    # ---- Accelerate prepare ----
    if accelerator is not None:
        transformer, optimizer, dataloader, lr_scheduler = accelerator.prepare(
            transformer, optimizer, dataloader, lr_scheduler,
        )

    # ---- Training loop ----
    cfg_dropout = cfg.get("cfg_dropout", 0.1)
    grad_clip = cfg.get("grad_clip", 1.0)
    save_every = cfg.get("save_every", 2000)
    log_every = cfg.get("log_every", 1)
    eval_every = cfg.get("eval_every", save_every)
    num_eval_samples = cfg.get("num_eval_samples", 4)

    eval_batches = []
    global_step = resume_step
    transformer.train()

    run_eval_next = resume_step > 0
    log_rank0(f"Starting training from step {global_step} to {max_steps}...", is_main)

    while global_step < max_steps:
        for batch in dataloader:
            if global_step >= max_steps:
                break

            # ---- Collect eval samples from the first few batches ----
            if is_main and len(eval_batches) < num_eval_samples:
                eval_batches.append({
                    "natural_frame": batch["natural_frame"][:1].clone(),
                    "target_frame": batch["target_frame"][:1].clone(),
                })

            # ---- VAE encode ----
            with torch.no_grad():
                natural_latent = vae.encode(
                    batch["natural_frame"].to(device=device, dtype=vae.dtype)
                ).latent_dist.mode()
                target_latent = vae.encode(
                    batch["target_frame"].to(device=device, dtype=vae.dtype)
                ).latent_dist.mode()

                natural_latent = (natural_latent - shift_factor) * scaling_factor
                target_latent = (target_latent - shift_factor) * scaling_factor

            natural_latent = natural_latent.to(dtype=torch.bfloat16)
            target_latent = target_latent.to(dtype=torch.bfloat16)
            B = target_latent.shape[0]

            # Null text conditioning (unconditional, per original Marigold)
            text_embeds = torch.zeros(B, 1, 4096, device=device, dtype=torch.bfloat16)
            pooled_projections = torch.zeros(B, pooled_dim, device=device, dtype=torch.bfloat16)

            # ---- Rectified flow ----
            noise = torch.randn_like(target_latent)
            t = torch.rand(B, device=device, dtype=torch.bfloat16)
            t_expand = t[:, None, None, None]

            noisy_target = (1 - t_expand) * target_latent + t_expand * noise
            target_velocity = noise - target_latent

            model_input = torch.cat([noisy_target, natural_latent], dim=1)
            timestep = (t * 1000).long()

            velocity_pred = transformer(
                hidden_states=model_input,
                timestep=timestep,
                encoder_hidden_states=text_embeds,
                pooled_projections=pooled_projections,
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
                ckpt = save_checkpoint(
                    transformer, global_step, run_dir, accelerator,
                    optimizer=optimizer, lr_scheduler=lr_scheduler, seed=seed,
                )
                log_rank0(f"Saved checkpoint: {ckpt}", is_main)

            # ---- Periodic eval: multiple samples ----
            should_eval = (global_step % eval_every == 0) or run_eval_next
            if is_main and should_eval and len(eval_batches) > 0:
                run_eval_next = False

                unwrapped = accelerator.unwrap_model(transformer) if accelerator else transformer

                import numpy as np
                from marigold_training.src.vis import normalize_to_uint8
                import cv2

                rows = []
                for eb in eval_batches:
                    with torch.no_grad():
                        ec = vae.encode(
                            eb["natural_frame"].to(device=device, dtype=vae.dtype)
                        ).latent_dist.mode()
                        ec = ((ec - shift_factor) * scaling_factor).to(dtype=torch.bfloat16)

                    et = torch.zeros(1, 1, 4096, device=device, dtype=torch.bfloat16)
                    ep = torch.zeros(1, pooled_dim, device=device, dtype=torch.bfloat16)

                    pred = _run_eval_inference(
                        unwrapped, vae, ec, et, ep, scaling_factor, shift_factor,
                    )  # [1, 3, H, W]

                    nat_np = ((eb["natural_frame"][0].numpy().transpose(1, 2, 0) + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
                    pred_np = normalize_to_uint8(pred[0].numpy().transpose(1, 2, 0))
                    gt_np = normalize_to_uint8(eb["target_frame"][0].cpu().numpy().transpose(1, 2, 0))
                    rows.append(np.concatenate([nat_np, pred_np, gt_np], axis=1))

                grid = np.concatenate(rows, axis=0)
                eval_dir = run_dir / "eval"
                eval_dir.mkdir(parents=True, exist_ok=True)
                out_path = eval_dir / f"step_{global_step:06d}.png"
                cv2.imwrite(str(out_path), grid[..., ::-1])
                log_rank0(f"Eval saved: {out_path} ({len(eval_batches)} samples)", is_main)

    if is_main:
        ckpt = save_checkpoint(
            transformer, global_step, run_dir, accelerator,
            optimizer=optimizer, lr_scheduler=lr_scheduler, seed=seed,
        )
        log_rank0(f"Training complete. Final checkpoint: {ckpt}", is_main)


if __name__ == "__main__":
    main()
