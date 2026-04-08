"""Pipeline loading for training and inference."""

import torch


def load_pipeline(cfg, device):
    """
    Load a Wan pipeline and extract components for training.

    Returns:
        dict with keys: transformer, vae (or None), latents_mean, latents_std, scheduler
    """
    from diffusers import WanPipeline

    model_id = cfg.get("model_id", "Wan-AI/Wan2.2-T2V-A14B-Diffusers")
    pipe = WanPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)

    transformer = pipe.transformer
    scheduler = pipe.scheduler

    # Latent normalization stats (per-channel, broadcast over B, T, H, W)
    latents_mean = torch.tensor(pipe.vae.config.latents_mean).view(1, -1, 1, 1, 1)
    latents_std = torch.tensor(pipe.vae.config.latents_std).view(1, -1, 1, 1, 1)

    on_the_fly = cfg.get("on_the_fly", False)
    if on_the_fly:
        vae = pipe.vae.to(device).eval()
        vae.requires_grad_(False)
        del pipe.text_encoder, pipe.tokenizer
    else:
        vae = None
        del pipe.text_encoder, pipe.tokenizer, pipe.vae

    del pipe
    torch.cuda.empty_cache()

    return {
        "transformer": transformer,
        "vae": vae,
        "latents_mean": latents_mean,
        "latents_std": latents_std,
        "scheduler": scheduler,
    }


def load_inference_pipeline(model_id, checkpoint_dir, device):
    """
    Load a Wan pipeline with LoRA or full fine-tuned weights for inference.

    Loads the base pretrained pipeline first, then loads the checkpoint
    weights into the existing transformer to avoid version mismatches.

    Args:
        model_id:       HuggingFace model ID
        checkpoint_dir: Path to checkpoint (contains lora_transformer/ or transformer/)
        device:         torch.device

    Returns:
        WanPipeline ready for generation
    """
    from pathlib import Path
    from diffusers import WanPipeline

    pipe = WanPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)

    checkpoint_dir = Path(checkpoint_dir)
    lora_path = checkpoint_dir / "lora_transformer"
    full_path = checkpoint_dir / "transformer"

    if lora_path.exists():
        from peft import PeftModel
        pipe.transformer = PeftModel.from_pretrained(pipe.transformer, str(lora_path))
    elif full_path.exists():
        # Load saved weights into the pretrained transformer architecture
        # to avoid version mismatches from instantiating a new class.
        from safetensors.torch import load_file
        state_dict = load_file(str(full_path / "model.safetensors"))
        pipe.transformer.load_state_dict(state_dict, strict=False)
    else:
        print(f"WARNING: No checkpoint found at {checkpoint_dir}")

    return pipe.to(device)
