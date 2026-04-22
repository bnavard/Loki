"""Visualization utilities for Marionette.

Hosts:
  - Pure image helpers (`slice_cond_rgb`, `add_label`, `add_red_border`,
    `save_labeled_grid`, `save_video_with_audio`, `make_grid_tensor`) used by
    both live-training viz and offline inspection.
  - `VisualizationCallback`, the Lightning callback that periodically runs
    same-identity DDIM reconstructions on the val set and writes mp4s /
    labeled grids / tensorboard images.

Kept out of train.py so the orchestrator file stays thin (CLI + run loop only).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
from pytorch_lightning.callbacks import Callback


def slice_cond_rgb(
    spatial_cond: torch.Tensor,
    ch_start: int,
    target_size: int,
    n_channels: int = 3,
) -> np.ndarray:
    """Render a 3-channel slice of a spatial_cond tensor as uint8 (T, 3, H, W).

    Args:
        spatial_cond: (T, H, W, 49) conditioning tensor (channels-last).
        ch_start:     first channel of the slice (42 → driver_deform, 45 → warped_ref).
        target_size:  output spatial size; upsampled bilinearly if different from H/W.
        n_channels:   slice width (default 3 for RGB-like visualization).
    """
    x = spatial_cond[..., ch_start:ch_start + n_channels].cpu().numpy()
    T, H, W, _ = x.shape

    mn, mx = x.min(), x.max()
    if mx - mn > 1e-8:
        x = (x - mn) / (mx - mn)
    else:
        x = np.zeros_like(x)
    x = (x * 255).astype(np.uint8)

    frames = []
    for t in range(T):
        frame = x[t] if (H == target_size and W == target_size) else cv2.resize(
            x[t], (target_size, target_size), interpolation=cv2.INTER_LINEAR,
        )
        frames.append(frame.transpose(2, 0, 1))
    return np.stack(frames, axis=0)


def add_label(row_img: np.ndarray, label: str,
              font_scale: float = 1.0, thickness: int = 2) -> np.ndarray:
    """Prepend a 180-px black strip with the label on the left of a row image."""
    H, W = row_img.shape[:2]
    label_w = 180
    canvas = np.zeros((H, label_w + W, 3), dtype=np.uint8)
    canvas[:, label_w:] = row_img

    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    cv2.putText(
        canvas, label, (10, (H + text_size[1]) // 2),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
    )
    return canvas


def add_red_border(frame_bgr: np.ndarray, border_width: int = 4) -> np.ndarray:
    """Draw a red border around a BGR frame (in-place). Used to mark the ref frame."""
    h, w = frame_bgr.shape[:2]
    cv2.rectangle(frame_bgr, (0, 0), (w - 1, h - 1),
                  color=(0, 0, 255), thickness=border_width)
    return frame_bgr


def save_labeled_grid(
    rows_data: List[np.ndarray],
    labels: List[str],
    path: Path,
    title: Optional[str] = None,
):
    """Write a labeled N-row strip grid to disk.

    Args:
        rows_data: list of (T, 3, H, W) uint8 arrays, one per row.
        labels:    row labels, same length.
        title:     optional title bar drawn at the top.
    The "Generated" row gets a red border on its frame 0 (reference slot).
    """
    T = rows_data[0].shape[0]
    n_show = min(T, 8)
    indices = np.linspace(0, T - 1, n_show, dtype=int)

    labeled_rows = []
    for imgs, label in zip(rows_data, labels):
        frames = []
        for t_idx in indices:
            frame = imgs[t_idx].transpose(1, 2, 0).copy()
            frame_bgr = frame[..., ::-1].copy()
            if "Generated" in label and t_idx == 0:
                add_red_border(frame_bgr)
            frames.append(frame_bgr)
        strip = np.concatenate(frames, axis=1)
        labeled_rows.append(add_label(strip, label))

    grid = np.concatenate(labeled_rows, axis=0)

    if title:
        title_h = 40
        title_bar = np.zeros((title_h, grid.shape[1], 3), dtype=np.uint8)
        text_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = (grid.shape[1] - text_size[0]) // 2
        cv2.putText(
            title_bar, title, (text_x, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
        )
        grid = np.concatenate([title_bar, grid], axis=0)

    cv2.imwrite(str(path), grid)


def save_video_with_audio(
    rows_data: List[np.ndarray],
    labels: List[str],
    audio_np: Optional[np.ndarray],
    path: Path,
    fps: float = 25.0,
    title: Optional[str] = None,
):
    """Write a stacked labeled video (rows stacked vertically) with optional muxed audio.

    Args:
        rows_data: list of (T, 3, H, W) uint8 arrays.
        audio_np:  (T, window_samples) float32 per-frame audio windows, or None.
        path:      output .mp4 path.
    """
    T = rows_data[0].shape[0]

    frames = []
    for t in range(T):
        rows = []
        for imgs, label in zip(rows_data, labels):
            frame_rgb = imgs[t].transpose(1, 2, 0).copy()
            frame_bgr = frame_rgb[..., ::-1].copy()
            if "Generated" in label and t == 0:
                add_red_border(frame_bgr)
            rows.append(add_label(frame_bgr, label, font_scale=0.7, thickness=1))
        composite = np.concatenate(rows, axis=0)

        if title:
            title_h = 30
            title_bar = np.zeros((title_h, composite.shape[1], 3), dtype=np.uint8)
            text_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            text_x = (composite.shape[1] - text_size[0]) // 2
            cv2.putText(
                title_bar, title, (text_x, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
            )
            composite = np.concatenate([title_bar, composite], axis=0)
        frames.append(composite)

    tmp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_video_path = tmp_video.name
    tmp_video.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        tmp_video_path, fourcc, fps, (frames[0].shape[1], frames[0].shape[0]),
    )
    for frame in frames:
        writer.write(frame)
    writer.release()

    if audio_np is not None:
        try:
            import soundfile as sf
            # Per-frame windows overlap by audio_context_frames on each side;
            # take the center chunk to avoid duplicating samples across frames.
            samples_per_frame = audio_np.shape[1] // 5
            center_offset = samples_per_frame * 2
            audio_full = np.concatenate([
                audio_np[t, center_offset:center_offset + samples_per_frame]
                for t in range(T)
            ])

            tmp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_audio_path = tmp_audio.name
            tmp_audio.close()
            sf.write(tmp_audio_path, audio_full, 16000)

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
            pass

    shutil.move(tmp_video_path, str(path))


def make_grid_tensor(rows_data: List[np.ndarray]) -> torch.Tensor:
    """Build a (3, N*H, n*W) uint8 tensor for tensorboard `add_image`."""
    T = rows_data[0].shape[0]
    n_show = min(T, 8)
    indices = np.linspace(0, T - 1, n_show, dtype=int)

    rows = []
    for imgs in rows_data:
        frames = [imgs[idx] for idx in indices]
        rows.append(np.concatenate(frames, axis=2))
    return torch.tensor(np.concatenate(rows, axis=1), dtype=torch.uint8)


def _to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, dict):
            out[k] = {
                kk: vv.to(device) if isinstance(vv, torch.Tensor) else vv
                for kk, vv in v.items()
            }
        else:
            out[k] = v
    return out


class VisualizationCallback(Callback):
    """Periodic same-identity DDIM reconstructions on the val set.

    Fires every `vis_every_n_steps` steps (rank 0 only, DDP-safe via explicit
    barriers). For each of `n_vis_samples` val samples:
      - Runs `model.sample_video` over the T gen slots (ref lives in the
        separate frozen ref UNet, not in the time axis).
      - Builds a 4-row labeled grid [Reference | Ground Truth | Driver Deform
        | Generated], saved as both a PNG grid and an audio-muxed mp4.
      - Logs the grid to tensorboard as a single image.
    """

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
        self._vis_call_count = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step == 0:
            return
        if trainer.global_step % self.vis_every_n_steps != 0:
            return

        # DDP safety: viz runs rank-0 only but takes minutes. Without barriers
        # the other ranks advance into the next step's allreduce and NCCL times
        # out. Brackets around the rank-0 work keep everyone aligned.
        if trainer.world_size > 1:
            torch.distributed.barrier()

        if trainer.global_rank == 0:
            self._generate_samples(trainer, pl_module)

        if trainer.world_size > 1:
            torch.distributed.barrier()

    @torch.no_grad()
    def _generate_samples(self, trainer, pl_module):
        model = pl_module.model
        model.eval()
        device = pl_module.device
        step = trainer.global_step
        fps = self.cfg.get("train_dataset", {}).get("params", {}).get("fps", 25.0)

        val_params = self.cfg.get("val_dataset", {}).get("params", {})
        resolution = val_params.get("resolution", 512)

        step_dir = self.output_dir / f"step_{step:06d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        cfg_scale = self.cfg.inference.get("cfg_scale", 2.0)

        n_skip = self._vis_call_count * self.n_vis_samples
        self._vis_call_count += 1

        n_generated = 0
        n_skipped = 0
        for batch in self.val_loader:
            if n_skipped < n_skip:
                n_skipped += batch[model.first_stage_key].shape[0]
                continue
            if n_generated >= self.n_vis_samples:
                break

            batch = _to_device(batch, device)

            gen_z, cond = model.get_input(batch, model.first_stage_key, force_conditional=True)
            ctrl     = cond['c_concat'][0]
            c_uncond = cond['c_uncond'][0]

            b = gen_z.shape[0]
            for i in range(b):
                if n_generated >= self.n_vis_samples:
                    break

                ctrl_i   = {k: (v[[i]] if v is not None else None) for k, v in ctrl.items()}
                uncond_i = {k: (v[[i]] if v is not None else None) for k, v in c_uncond.items()}

                try:
                    gen_latents = model.sample_video(
                        control=ctrl_i, control_uncond=uncond_i,
                        n_frames=gen_z.shape[1],
                        latent_shape=tuple(gen_z.shape[2:]),
                        n_ddim_steps=self.vis_ddim_steps,
                        cfg_scale=cfg_scale,
                    )

                    gen_imgs = model.decode_first_stage(gen_latents.unsqueeze(0)).squeeze(0)
                    gen_imgs = ((gen_imgs.clamp(-1, 1) + 1) / 2 * 255).byte().cpu().numpy()

                    gt_imgs = model.decode_first_stage(gen_z[i:i+1]).squeeze(0)
                    gt_imgs = ((gt_imgs.clamp(-1, 1) + 1) / 2 * 255).byte().cpu().numpy()

                    spatial_cond_i = ctrl_i["spatial_cond"][0]
                    deform_vis = slice_cond_rgb(spatial_cond_i, 42, resolution)

                    # Static reference row: decode the ref latent from control
                    # and broadcast across T for grid alignment.
                    ref_z_i = ctrl_i["ref_z"]
                    ref_img = model.decode_first_stage(ref_z_i.unsqueeze(1)).squeeze(0)
                    ref_u8  = ((ref_img.clamp(-1, 1) + 1) / 2 * 255).byte().cpu().numpy()[0]
                    T_gen   = gen_z.shape[1]
                    ref_row = np.broadcast_to(ref_u8[None], (T_gen, *ref_u8.shape)).copy()

                    rows_data = [ref_row, gt_imgs, deform_vis, gen_imgs]
                    labels = ["Reference", "Ground Truth", "Driver Deform", "Generated"]
                    title = f"Same-Identity Reconstruction | Step {step}"

                    save_labeled_grid(
                        rows_data, labels,
                        step_dir / f"sample_{n_generated:02d}.png", title=title,
                    )

                    # Only mux audio into the viz mp4 if the model actually
                    # consumes it. The audio-off arm still receives audio in
                    # the batch (dataset stays uniform across ablations), but
                    # surfacing it in the viz would misrepresent what the
                    # model had access to.
                    if model.audio_encoder is not None:
                        audio_i = batch.get("audio", None)
                        audio_np = audio_i[i].cpu().numpy() if audio_i is not None else None
                    else:
                        audio_np = None
                    save_video_with_audio(
                        rows_data, labels, audio_np,
                        step_dir / f"sample_{n_generated:02d}.mp4",
                        fps=fps, title=title,
                    )

                    if trainer.logger:
                        grid = make_grid_tensor(rows_data)
                        trainer.logger.experiment.add_image(
                            f"vis/sample_{n_generated}", grid, global_step=step,
                        )
                except Exception as e:
                    import traceback
                    print(f"  Visualization failed for sample {n_generated}: {e}")
                    traceback.print_exc()

                n_generated += 1

        model.train()
        if trainer.global_rank == 0:
            print(f"  Saved {n_generated} visualization(s) to {step_dir}")
