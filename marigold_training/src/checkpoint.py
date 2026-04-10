"""Full training state checkpointing for exact resume."""

from pathlib import Path

import torch


def save_checkpoint(transformer, step, run_dir, accelerator=None,
                    optimizer=None, lr_scheduler=None, seed=None):
    """
    Save a full training checkpoint: model weights + training state.

    Saves:
      - transformer/       (model weights via save_pretrained)
      - training_state.pt  (optimizer, lr_scheduler, step, seed)

    Args:
        transformer:  the model (possibly wrapped by accelerate)
        step:         global training step
        run_dir:      base run directory
        accelerator:  optional accelerate.Accelerator for unwrapping
        optimizer:    optimizer state (optional, for resume)
        lr_scheduler: lr scheduler state (optional, for resume)
        seed:         current seed for reproducibility
    """
    ckpt_dir = Path(run_dir) / f"step_{step:06d}"
    ckpt_dir.mkdir(exist_ok=True)

    # Save model weights
    unwrapped = accelerator.unwrap_model(transformer) if accelerator else transformer
    unwrapped.save_pretrained(str(ckpt_dir / "transformer"))

    # Save training state for exact resume
    training_state = {"global_step": step}
    if optimizer is not None:
        training_state["optimizer"] = optimizer.state_dict()
    if lr_scheduler is not None:
        training_state["lr_scheduler"] = lr_scheduler.state_dict()
    if seed is not None:
        training_state["seed"] = seed
        training_state["rng_state"] = torch.random.get_rng_state()
        if torch.cuda.is_available():
            training_state["cuda_rng_state"] = torch.cuda.get_rng_state()

    torch.save(training_state, str(ckpt_dir / "training_state.pt"))

    return ckpt_dir


def load_training_state(ckpt_dir, optimizer=None, lr_scheduler=None):
    """
    Load training state from a checkpoint for exact resume.

    Args:
        ckpt_dir:     path to step directory (e.g. step_001000/)
        optimizer:    optimizer to restore state into
        lr_scheduler: scheduler to restore state into

    Returns:
        global_step from the checkpoint
    """
    ckpt_dir = Path(ckpt_dir)
    state_path = ckpt_dir / "training_state.pt"

    if not state_path.exists():
        # Fallback: extract step from directory name
        step_str = ckpt_dir.name.replace("step_", "")
        return int(step_str)

    state = torch.load(str(state_path), map_location="cpu", weights_only=False)

    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])

    if lr_scheduler is not None and "lr_scheduler" in state:
        lr_scheduler.load_state_dict(state["lr_scheduler"])

    # Restore RNG state for reproducibility
    if "rng_state" in state:
        torch.random.set_rng_state(state["rng_state"])
    if "cuda_rng_state" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda_rng_state"])

    return state["global_step"]
