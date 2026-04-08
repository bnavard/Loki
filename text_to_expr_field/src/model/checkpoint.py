"""Checkpoint saving for both LoRA and full fine-tuned models."""

from pathlib import Path


def save_checkpoint(transformer, step, run_dir, use_lora, accelerator=None):
    """
    Save a training checkpoint.

    Args:
        transformer: the model (possibly wrapped by accelerate)
        step:        global training step (used in directory name)
        run_dir:     base run directory
        use_lora:    if True saves as lora_transformer/, else transformer/
        accelerator: optional accelerate.Accelerator for unwrapping
    """
    ckpt_dir = Path(run_dir) / f"step_{step:06d}"
    ckpt_dir.mkdir(exist_ok=True)

    unwrapped = accelerator.unwrap_model(transformer) if accelerator else transformer
    save_name = "lora_transformer" if use_lora else "transformer"
    unwrapped.save_pretrained(str(ckpt_dir / save_name))

    return ckpt_dir
