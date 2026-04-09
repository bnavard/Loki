"""
Training script for the Talking-Head Diffusion Model.

Usage:
    python talkinghead/train.py --config talkinghead/configs/talking_head.yaml \
                                 [--resume /path/to/checkpoint.ckpt] \
                                 [--gpus 0 1]

Data paths (video_root, audio_root, flame_root, id_list_path) are read from the
YAML config.  The script uses PyTorch Lightning for training loop, checkpointing,
and logging.
"""

import argparse
import os
from pathlib import Path

import torch
torch.backends.cudnn.enabled = False
import numpy as np
import einops
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from controlnet.ldm.util import instantiate_from_config
from talkinghead_sd21_unet_cap4d_based.data.video_dataset import TalkingHeadDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",    required=True, help="Path to talking_head.yaml")
    p.add_argument("--resume",    default=None,  help="Checkpoint to resume from")
    p.add_argument("--gpus",      nargs="+", type=int, default=[0])
    return p.parse_args()


def build_dataloader(ds_cfg, batch_size, shuffle=True, drop_last=True):
    dataset = TalkingHeadDataset(**OmegaConf.to_container(ds_cfg, resolve=True))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        drop_last=drop_last,
    )


def is_rank_zero():
    """Check if current process is rank 0 (or not in DDP)."""
    rank = int(os.environ.get("LOCAL_RANK", 0))
    return rank == 0


def load_model(cfg, init_path=None):
    model = instantiate_from_config(cfg.model)

    if init_path and Path(init_path).exists():
        if is_rank_zero():
            print(f"Loading SD 2.1 weights from {init_path}")
        sd = torch.load(init_path, map_location="cpu")
        state_dict = sd.get("state_dict", sd)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if is_rank_zero():
            print(f"  Missing keys : {len(missing)}")
            print(f"  Unexpected   : {len(unexpected)}")
            # Group missing keys by prefix for readability
            missing_prefixes = {}
            for k in missing:
                prefix = k.split(".")[0] + "." + k.split(".")[1] if "." in k else k
                missing_prefixes.setdefault(prefix, []).append(k)
            print("  Missing key groups:")
            for prefix, keys in sorted(missing_prefixes.items()):
                print(f"    {prefix}: {len(keys)} keys (e.g. {keys[0]})")
            unexpected_prefixes = {}
            for k in unexpected:
                prefix = k.split(".")[0] + "." + k.split(".")[1] if "." in k else k
                unexpected_prefixes.setdefault(prefix, []).append(k)
            print("  Unexpected key groups:")
            for prefix, keys in sorted(unexpected_prefixes.items()):
                print(f"    {prefix}: {len(keys)} keys (e.g. {keys[0]})")

    return model


# ---------------------------------------------------------------------------
# Visualization callback
# ---------------------------------------------------------------------------
class VisualizationCallback(Callback):
    """Generate and log talking-head samples at regular intervals."""

    def __init__(self, cfg, val_loader, output_dir, vis_every_n_steps=2000,
                 n_vis_samples=8, vis_ddim_steps=20):
        super().__init__()
        self.cfg = cfg
        self.val_loader = val_loader
        self.output_dir = Path(output_dir) / "visualizations"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.vis_every_n_steps = vis_every_n_steps
        self.n_vis_samples = n_vis_samples
        self.vis_ddim_steps = vis_ddim_steps
        self._vis_call_count = 0  # used to rotate through val samples

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step == 0:
            return
        if trainer.global_step % self.vis_every_n_steps != 0:
            return
        # Only rank 0 generates visualizations in DDP
        if trainer.global_rank != 0:
            return
        self._generate_samples(trainer, pl_module)

    @torch.no_grad()
    def _generate_samples(self, trainer, pl_module):
        import cv2

        model = pl_module.model
        model.eval()
        device = pl_module.device
        step = trainer.global_step
        fps = self.cfg.get("train_dataset", self.cfg.get("dataset", {})).get(
            "params", {}).get("fps", 25.0)

        step_dir = self.output_dir / f"step_{step:06d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        from talkinghead_sd21_unet_cap4d_based.model.th_sampler import THSampler
        sampler = THSampler(model)

        # Rotate through different val samples each visualization step
        n_skip = self._vis_call_count * self.n_vis_samples
        self._vis_call_count += 1

        n_generated = 0
        n_skipped = 0
        for batch in self.val_loader:
            # Skip batches to reach different samples each time
            if n_skipped < n_skip:
                n_skipped += batch[model.first_stage_key].shape[0]
                continue
            if n_generated >= self.n_vis_samples:
                break

            batch = self._to_device(batch, device)

            z, cond = model.get_input(batch, model.first_stage_key, force_conditional=True)
            ctrl = cond['c_concat'][0]
            c_uncond = cond['c_uncond'][0]

            b = z.shape[0]
            for i in range(b):
                if n_generated >= self.n_vis_samples:
                    break

                ctrl_i = {k: v[[i]] if v is not None else None for k, v in ctrl.items()}
                uncond_i = {k: v[[i]] if v is not None else None for k, v in c_uncond.items()}

                # Split conditioning along time axis into reference (frame 0)
                # and generation (frames 1:) for the Stochastic I/O sampler.
                # squeeze(0) removes the batch dim since we're processing one
                # sample at a time — sampler expects (T, ...) not (1, T, ...).
                # cond = conditioning for the conditional branch (model sees full signal)
                # uncond = conditioning for the unconditional branch (zeroed, for CFG)
                ref_cond   = {k: v[:, :1].squeeze(0) if v is not None else None for k, v in ctrl_i.items()}
                ref_uncond = {k: v[:, :1].squeeze(0) if v is not None else None for k, v in uncond_i.items()}
                gen_cond   = {k: v[:, 1:].squeeze(0) if v is not None else None for k, v in ctrl_i.items()}
                gen_uncond = {k: v[:, 1:].squeeze(0) if v is not None else None for k, v in uncond_i.items()}

                V = z.shape[1]
                R = 1
                latent_shape = z.shape[2:]

                try:
                    gen_latents = sampler.sample(
                        S=self.vis_ddim_steps,
                        ref_cond=ref_cond, ref_uncond=ref_uncond,
                        gen_cond=gen_cond, gen_uncond=gen_uncond,
                        latent_shape=latent_shape, V=V, R=R,
                        cfg_scale=self.cfg.inference.get("cfg_scale", 2.0),
                        verbose=False,
                    )

                    ref_latent = z[i, :1]
                    all_latents = torch.cat([ref_latent, gen_latents], dim=0)

                    gen_imgs = model.decode_first_stage(all_latents[None]).squeeze(0)
                    gen_imgs = ((gen_imgs.clamp(-1, 1) + 1) / 2 * 255).byte().cpu().numpy()

                    gt_imgs = model.decode_first_stage(z[i:i+1]).squeeze(0)
                    gt_imgs = ((gt_imgs.clamp(-1, 1) + 1) / 2 * 255).byte().cpu().numpy()

                    # Expression map visualization from pos_enc
                    pos_enc_i = ctrl_i["pos_enc"][0]  # (T, H, W, C_cond)
                    expr_vis = self._render_expression_maps(pos_enc_i, gen_imgs.shape[-1])

                    # Form 1: image grid (GT / Expression Map / Generated) with labels
                    self._save_labeled_grid(
                        gt_imgs, expr_vis, gen_imgs,
                        step_dir / f"sample_{n_generated:02d}.png",
                    )

                    # Form 2: video with audio
                    audio_i = batch.get("audio", None)
                    audio_np = audio_i[i].cpu().numpy() if audio_i is not None else None
                    self._save_video_with_audio(
                        gt_imgs, expr_vis, gen_imgs, audio_np,
                        step_dir / f"sample_{n_generated:02d}.mp4",
                        fps=fps,
                    )

                    # Log image grid to tensorboard
                    if trainer.logger:
                        grid = self._make_grid_tensor(gt_imgs, expr_vis, gen_imgs)
                        trainer.logger.experiment.add_image(
                            f"vis/sample_{n_generated}", grid, global_step=step,
                        )
                except Exception as e:
                    import traceback
                    print(f"  Visualization failed for sample {n_generated}: {e}")
                    traceback.print_exc()

                n_generated += 1

        model.train()
        if is_rank_zero():
            print(f"  Saved {n_generated} visualization(s) to {step_dir}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _to_device(self, batch, device):
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(device)
            elif isinstance(v, dict):
                out[k] = {kk: vv.to(device) if isinstance(vv, torch.Tensor) else vv
                          for kk, vv in v.items()}
            else:
                out[k] = v
        return out

    def _render_expression_maps(self, pos_enc, target_size):
        """
        Extract a 3-channel expression deformation slice from pos_enc and
        render as a uint8 image array (T, 3, H_out, W_out).

        pos_enc: (T, H, W, C_cond) on GPU or CPU
        """
        import cv2
        C_cond = pos_enc.shape[-1]
        T = pos_enc.shape[0]

        # No FLAME conditioning (only ref_mask = 1 channel) — blank expression row
        if C_cond == 1:
            return np.zeros((T, 3, target_size, target_size), dtype=np.uint8)

        # Deformation-only mode (4 channels: 3 deform + 1 ref_mask) — deform is at [0:3]
        # Full mode (46 channels: 42 pos enc + 3 deform + 1 ref_mask) — deform is at [42:45]
        if C_cond == 4:
            expr = pos_enc[..., 0:3].cpu().numpy()
        else:
            expr = pos_enc[..., 42:45].cpu().numpy()  # (T, H, W, 3)
        T, H, W, _ = expr.shape

        # Normalize to [0, 255] for visualization
        e_min, e_max = expr.min(), expr.max()
        if e_max - e_min > 1e-8:
            expr = (expr - e_min) / (e_max - e_min)
        else:
            expr = np.zeros_like(expr)
        expr = (expr * 255).astype(np.uint8)

        # Resize to match generated image resolution and convert to (T, 3, H, W)
        frames = []
        for t in range(T):
            frame = cv2.resize(expr[t], (target_size, target_size),
                               interpolation=cv2.INTER_NEAREST)
            frames.append(frame.transpose(2, 0, 1))  # (3, H, W)
        return np.stack(frames, axis=0)  # (T, 3, H, W)

    def _add_label(self, row_img, label, font_scale=1.0, thickness=2):
        """Add a text label on the left side of a row image. Returns new image."""
        import cv2
        H, W = row_img.shape[:2]
        label_w = 180
        canvas = np.zeros((H, label_w + W, 3), dtype=np.uint8)
        canvas[:, label_w:] = row_img

        # Center text vertically
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_x = 10
        text_y = (H + text_size[1]) // 2
        cv2.putText(canvas, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        return canvas

    def _save_labeled_grid(self, gt_imgs, expr_imgs, gen_imgs, path):
        """
        Save a 3-row grid: Ground Truth / Expression Map / Generated.
        Each row has a label on the left. Shows up to 8 evenly spaced frames.
        All inputs are (T, 3, H, W) uint8 numpy arrays.
        """
        import cv2
        T = gt_imgs.shape[0]
        n_show = min(T, 8)
        indices = np.linspace(0, T - 1, n_show, dtype=int)

        labels = ["Ground Truth", "Expression", "Generated"]
        all_rows = [gt_imgs, expr_imgs, gen_imgs]
        labeled_rows = []

        for imgs, label in zip(all_rows, labels):
            frames = [imgs[idx].transpose(1, 2, 0) for idx in indices]  # (H, W, 3) RGB
            strip = np.concatenate(frames, axis=1)
            strip_bgr = strip[..., ::-1].copy()
            labeled = self._add_label(strip_bgr, label)
            labeled_rows.append(labeled)

        grid = np.concatenate(labeled_rows, axis=0)
        cv2.imwrite(str(path), grid)

    def _save_video_with_audio(self, gt_imgs, expr_imgs, gen_imgs, audio_np,
                                path, fps=25.0):
        """
        Save a side-by-side video (3 rows stacked) with embedded audio.
        gt_imgs, expr_imgs, gen_imgs: (T, 3, H, W) uint8
        audio_np: (T, window_samples) float32 or None
        """
        import cv2
        import tempfile
        import subprocess

        T = gt_imgs.shape[0]
        H_frame = gt_imgs.shape[2]
        W_frame = gt_imgs.shape[3]

        # Build composite frames: 3 rows with labels
        label_w = 180
        comp_w = label_w + W_frame
        comp_h = H_frame * 3

        labels = ["Ground Truth", "Expression", "Generated"]
        all_rows = [gt_imgs, expr_imgs, gen_imgs]

        frames = []
        for t in range(T):
            rows = []
            for imgs, label in zip(all_rows, labels):
                frame_rgb = imgs[t].transpose(1, 2, 0)  # (H, W, 3)
                frame_bgr = frame_rgb[..., ::-1].copy()
                labeled = self._add_label(frame_bgr, label, font_scale=0.7, thickness=1)
                rows.append(labeled)
            composite = np.concatenate(rows, axis=0)
            frames.append(composite)

        # Write video without audio first
        tmp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp_video_path = tmp_video.name
        tmp_video.close()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp_video_path, fourcc, fps,
                                  (frames[0].shape[1], frames[0].shape[0]))
        for frame in frames:
            writer.write(frame)
        writer.release()

        if audio_np is not None:
            try:
                import soundfile as sf
                # Concatenate per-frame audio windows (take center portion to avoid overlap)
                samples_per_frame = audio_np.shape[1] // 5  # center frame of 5-frame window
                center_offset = samples_per_frame * 2  # skip first 2 context frames
                audio_concat = []
                for t in range(T):
                    chunk = audio_np[t, center_offset:center_offset + samples_per_frame]
                    audio_concat.append(chunk)
                audio_full = np.concatenate(audio_concat)

                tmp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp_audio_path = tmp_audio.name
                tmp_audio.close()
                sf.write(tmp_audio_path, audio_full, 16000)

                # Mux video + audio with ffmpeg
                subprocess.run([
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-i", tmp_video_path,
                    "-i", tmp_audio_path,
                    "-c:v", "libx264", "-crf", "23",
                    "-c:a", "aac", "-b:a", "128k",
                    "-shortest",
                    str(path),
                ], check=True)
                os.unlink(tmp_video_path)
                os.unlink(tmp_audio_path)
                return
            except Exception:
                pass  # Fall through to no-audio path

        # No audio or ffmpeg failed — just rename tmp video
        import shutil
        shutil.move(tmp_video_path, str(path))

    def _make_grid_tensor(self, gt_imgs, expr_imgs, gen_imgs):
        """Create a (3, 3*H, n*W) tensor for tensorboard logging."""
        T = gt_imgs.shape[0]
        n_show = min(T, 8)
        indices = np.linspace(0, T - 1, n_show, dtype=int)

        rows = []
        for imgs in [gt_imgs, expr_imgs, gen_imgs]:
            frames = [imgs[idx] for idx in indices]  # list of (3, H, W)
            rows.append(np.concatenate(frames, axis=2))  # (3, H, n*W)

        grid = np.concatenate(rows, axis=1)  # (3, 3*H, n*W)
        return torch.tensor(grid, dtype=torch.uint8)


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------
class LightningWrapper(pl.LightningModule):
    """Thin Lightning wrapper around THDiffusion."""

    def __init__(self, model, cfg):
        super().__init__()
        self.model = model
        self.cfg   = cfg

    def training_step(self, batch, batch_idx):
        z, cond = self.model.get_input(batch, self.model.first_stage_key)
        loss, loss_dict = self.model(z, cond)
        self.log_dict(loss_dict, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        z, cond = self.model.get_input(batch, self.model.first_stage_key,
                                        force_conditional=True)
        loss, loss_dict = self.model(z, cond)
        self.log_dict(loss_dict, prog_bar=True, on_step=False, on_epoch=True,
                      sync_dist=True)
        return loss

    def configure_optimizers(self):
        return self.model.configure_optimizers()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    cfg  = OmegaConf.load(args.config)

    # Set seeds for reproducibility
    pl.seed_everything(cfg.seed, workers=True)

    # Create a timestamped run directory (rank 0 only to avoid duplicates in DDP)
    output_dir = cfg.output_dir
    from datetime import datetime
    if is_rank_zero():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(output_dir) / f"run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        # Write run_dir to a tmp file so other ranks can read it
        marker = Path(output_dir) / ".current_run_dir"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(str(run_dir))
        # Copy config file to run directory for reproducibility
        import shutil
        shutil.copy2(args.config, run_dir / "config.yaml")
        print(f"Run directory: {run_dir}")
    else:
        # Wait for rank 0 to create the directory
        import time
        marker = Path(output_dir) / ".current_run_dir"
        for _ in range(60):
            if marker.exists():
                break
            time.sleep(0.5)
        run_dir = Path(marker.read_text().strip())
    output_dir = str(run_dir)

    model = load_model(cfg, init_path=cfg.get("init_path"))
    model.learning_rate = cfg.learning_rate

    if is_rank_zero():
        print(f"Building train dataloader...")
    train_loader = build_dataloader(cfg.train_dataset.params, cfg.gpu_batch_size,
                                     shuffle=True, drop_last=True)
    if is_rank_zero():
        print(f"  Train clips: {len(train_loader.dataset)}")
        print(f"Building val dataloader...")
    val_loader   = build_dataloader(cfg.val_dataset.params, cfg.gpu_batch_size,
                                     shuffle=False, drop_last=False)
    if is_rank_zero():
        print(f"  Val clips: {len(val_loader.dataset)}")

    wrapper = LightningWrapper(model, cfg)

    # Checkpoint: save every N steps (keep all)
    periodic_ckpt = ModelCheckpoint(
        dirpath=output_dir,
        filename="th-{step:06d}",
        every_n_train_steps=cfg.save_every_n_steps,
        save_top_k=-1,
    )

    # Checkpoint: save best by val loss
    best_ckpt = ModelCheckpoint(
        dirpath=output_dir,
        filename="th-best-{step:06d}-{val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
    )

    # Visualization callback
    vis_cb = VisualizationCallback(
        cfg=cfg,
        val_loader=val_loader,
        output_dir=output_dir,
        vis_every_n_steps=cfg.get("val_every_n_steps", 2000),
        n_vis_samples=cfg.get("n_vis_samples", 4),
        vis_ddim_steps=cfg.get("vis_ddim_steps", 20),
    )

    logger = TensorBoardLogger(save_dir=output_dir, name="logs")

    trainer = pl.Trainer(
        max_steps=cfg.n_steps,
        accelerator="gpu",
        devices=args.gpus,
        strategy="ddp_find_unused_parameters_true" if len(args.gpus) > 1 else "auto",
        precision=16,
        callbacks=[vis_cb, best_ckpt],
        logger=logger,
        log_every_n_steps=cfg.logger_freq,
        accumulate_grad_batches=cfg.virtual_batch_size // cfg.gpu_batch_size,
        val_check_interval=1.0,
        limit_val_batches=cfg.get("n_val_batches", 20),
        deterministic=True,
    )

    trainer.fit(wrapper, train_loader, val_loader, ckpt_path=args.resume)


if __name__ == "__main__":
    main()
