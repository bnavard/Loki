"""Checkpoint saving."""

from pathlib import Path


def save_checkpoint(transformer, step, run_dir, accelerator=None):
    """
    Save a training checkpoint.

    Args:
        transformer: the model (possibly wrapped by accelerate)
        step:        global training step
        run_dir:     base run directory
        accelerator: optional accelerate.Accelerator for unwrapping
    """
    ckpt_dir = Path(run_dir) / f"step_{step:06d}"
    ckpt_dir.mkdir(exist_ok=True)

    unwrapped = accelerator.unwrap_model(transformer) if accelerator else transformer
    unwrapped.save_pretrained(str(ckpt_dir / "transformer"))

    return ckpt_dir
