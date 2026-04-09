"""Visualization for deformation maps."""

from pathlib import Path

import cv2
import numpy as np
import torch


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalize a float array symmetrically around zero to [0, 255] uint8."""
    abs_max = np.abs(arr).max() + 1e-8
    return ((arr / abs_max + 1) / 2 * 255).clip(0, 255).astype(np.uint8)


def save_video(frames: list, path, fps: int):
    """Save a list of uint8 BGR frames as an mp4 video."""
    path = Path(path)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h),
    )
    for frame in frames:
        writer.write(frame)
    writer.release()


def visualize_deform(deform_field: torch.Tensor, output_dir, fps: int, verbose: bool = True):
    """
    Save deformation map as a video.

    Args:
        deform_field: [T, 3, H, W] deformation tensor
        output_dir:   directory to save deformation.mp4
        fps:          video frame rate
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    T = deform_field.shape[0]
    deform_np = deform_field.numpy()

    frames = []
    for t in range(T):
        deform = deform_np[t].transpose(1, 2, 0)
        vis = normalize_to_uint8(deform)
        frames.append(vis[..., ::-1])
    save_video(frames, output_dir / "deformation.mp4", fps)

    if verbose:
        print(f"  Video saved to {output_dir}/deformation.mp4")


@torch.no_grad()
def generate_eval_sample(
    transformer, vae, natural_latent, text_embeds,
    latents_mean, latents_std,
    num_steps=50, guidance_scale=1.0,
):
    """
    Run Euler denoising for evaluation during training.

    Args:
        transformer:    the model (switched to eval mode temporarily)
        vae:            frozen VAE for decoding
        natural_latent: [1, 16, T, h, w] normalized conditioning latent
        text_embeds:    [1, seq_len, dim] text embedding
        latents_mean:   [1, 16, 1, 1, 1] normalization stats
        latents_std:    [1, 16, 1, 1, 1] normalization stats
        num_steps:      Euler steps
        guidance_scale: CFG scale (1.0 = no guidance)

    Returns:
        decoded: [T, 3, H, W] float32 CPU tensor (deformation map in pixel space)
    """
    device = natural_latent.device
    was_training = transformer.training
    transformer.eval()

    x = torch.randn_like(natural_latent)
    timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

    use_cfg = guidance_scale > 1.0
    null_embeds = torch.zeros_like(text_embeds) if use_cfg else None

    for i in range(num_steps):
        t_current = timesteps[i]
        dt = timesteps[i + 1] - timesteps[i]

        model_input = torch.cat([x, natural_latent], dim=1)
        t_batch = t_current.expand(1)

        if use_cfg:
            vel_cond = transformer(
                model_input, timestep=t_batch,
                encoder_hidden_states=text_embeds,
            ).sample
            vel_uncond = transformer(
                model_input, timestep=t_batch,
                encoder_hidden_states=null_embeds,
            ).sample
            velocity = vel_uncond + guidance_scale * (vel_cond - vel_uncond)
        else:
            velocity = transformer(
                model_input, timestep=t_batch,
                encoder_hidden_states=text_embeds,
            ).sample

        x = x + velocity * dt

    # Denormalize + decode
    raw_latent = x * latents_std.to(x.device, x.dtype) + latents_mean.to(x.device, x.dtype)
    decoded = vae.decode(raw_latent.to(vae.dtype), return_dict=False)[0]
    # [1, 3, T, H, W] → [T, 3, H, W]
    decoded = decoded.squeeze(0).permute(1, 0, 2, 3).float().cpu()

    if was_training:
        transformer.train()

    return decoded


def save_eval_grid(natural_pixels, pred_deform, gt_deform, output_path):
    """
    Save a side-by-side comparison image: input | predicted | ground truth.

    For video (T>1), saves the middle frame. For T=1, saves that frame.

    Args:
        natural_pixels: [3, T, H, W] in [-1, 1]
        pred_deform:    [T, 3, H, W] predicted deformation
        gt_deform:      [T, 3, H, W] ground truth deformation (before VAE)
        output_path:    path to save the PNG
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Pick middle frame
    if natural_pixels.shape[0] == 3:
        # [3, T, H, W]
        T = natural_pixels.shape[1]
        mid = T // 2
        nat = natural_pixels[:, mid].numpy().transpose(1, 2, 0)
    else:
        T = natural_pixels.shape[0]
        mid = T // 2
        nat = natural_pixels[mid].numpy().transpose(1, 2, 0)

    mid_pred = min(mid, pred_deform.shape[0] - 1)
    mid_gt = min(mid, gt_deform.shape[0] - 1)

    # Natural: [-1, 1] → [0, 255]
    nat_vis = ((nat + 1) / 2 * 255).clip(0, 255).astype(np.uint8)

    # Deformations: symmetric normalization
    pred_vis = normalize_to_uint8(pred_deform[mid_pred].numpy().transpose(1, 2, 0))
    gt_vis = normalize_to_uint8(gt_deform[mid_gt].numpy().transpose(1, 2, 0))

    # Horizontally: input | predicted | ground truth
    grid = np.concatenate([nat_vis, pred_vis, gt_vis], axis=1)
    cv2.imwrite(str(output_path), grid[..., ::-1])
