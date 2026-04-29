"""
Training driver for the Marionette talking-head diffusion model.

Two public entry points:

  * run_training(cfg, output_dir=None, resume=None, gpus=(0,)) — the pure
    function. Experiment scripts (e.g. `experiments/marionette_baseline/run.py`,
    `experiments/condition_ablation/run_*.py`) import this and pass a
    pre-loaded DictConfig produced by
    `marionette.config_utils.load_experiment_config`.

  * main() — thin CLI wrapper that argparse-parses `--config`, `--resume`,
    `--output_dir`, and `--gpus`, then calls run_training. Retained so a bare
    `python marionette/train.py --config path/to/exp.yaml` still works for
    ad-hoc runs.

Data paths (video_root, flame_root, clip_list_path) are read from the YAML
config; SD 2.1 pretrained init from `cfg.init_path`. Uses PyTorch
Lightning for the training loop, checkpointing (`th-<step>.ckpt` periodic +
`th-best-...` on val/loss), and TensorBoard logging.

Side-effects worth knowing about:

  * On rank 0, a `log.txt` is opened at `<run_dir>/log.txt` via
    `marionette.utils.install_log_tee`, mirroring stdout/stderr (line-buffered)
    so the terminal output of a run is recoverable from disk after the
    session ends. Other ranks' stdout stays terminal-only.
  * `VisualizationCallback` fires every `val_every_n_steps`; every rank
    participates in the viz pass — the work is sharded across ranks by
    sample, so a multi-GPU run scales the viz step too.
"""

import argparse
import os
from pathlib import Path
from typing import Optional, Sequence

import torch
torch.backends.cudnn.enabled = False
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader

from ldm_base.ldm.util import instantiate_from_config
from marionette.data.video_dataset import TalkingHeadDataset
from marionette.utils import VisualizationCallback, install_log_tee


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     required=True, help="Path to an experiment YAML (base + overlays).")
    p.add_argument("--output_dir", default=None,  help="Override output_dir from the config.")
    p.add_argument("--resume",     default=None,  help="Checkpoint to resume from.")
    p.add_argument("--gpus",       nargs="+", type=int, default=[0])
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
    from marionette.model.ref_unet import RefFeatureExtractor

    model = instantiate_from_config(cfg.model)

    if init_path and Path(init_path).exists():
        if is_rank_zero():
            print(f"Loading SD 2.1 weights from {init_path}")
        sd = torch.load(init_path, map_location="cpu")
        state_dict = sd.get("state_dict", sd)
        # Duplicate SD 2.1's `model.diffusion_model.*` keys under
        # `ref_extractor.unet.*` so one load call populates both the gen UNet
        # and the frozen reference UNet from the same pretrained source.
        state_dict = RefFeatureExtractor.load_sd21_into_ref(state_dict)
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
# Lightning module
# ---------------------------------------------------------------------------
class LightningWrapper(pl.LightningModule):
    """Thin Lightning wrapper around MarionetteDiffusion."""

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
# Public entry point
# ---------------------------------------------------------------------------
def run_training(
    cfg: DictConfig,
    output_dir: Optional[str] = None,
    resume: Optional[str] = None,
    gpus: Sequence[int] = (0,),
) -> None:
    """Run a full training job against a pre-loaded config.

    Args:
        cfg:        fully-resolved DictConfig (e.g. produced by
                    `marionette.config_utils.load_experiment_config`).
        output_dir: root directory for this run's outputs. If None, falls back
                    to `cfg.output_dir`. A timestamped `run_<YYYYmmdd_HHMMSS>/`
                    subdirectory is always created inside this root.
        resume:     optional path to a Lightning checkpoint to resume from.
        gpus:       GPU indices to train on.
    """
    if output_dir is None:
        if "output_dir" not in cfg:
            raise ValueError(
                "run_training: `output_dir` not provided and `cfg.output_dir` is unset."
            )
        output_dir = cfg.output_dir

    # Set seeds for reproducibility
    pl.seed_everything(cfg.seed, workers=True)

    # When resuming, reuse the checkpoint's directory; otherwise create a new
    # timestamped run directory (rank 0 only to avoid duplicates in DDP).
    from datetime import datetime
    if resume is not None:
        run_dir = Path(resume).parent.parent
        if is_rank_zero():
            print(f"Resuming into existing run directory: {run_dir}")
    elif is_rank_zero():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(output_dir) / f"run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        # Write run_dir to a marker file so other ranks can read it
        marker = Path(output_dir) / ".current_run_dir"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(str(run_dir))
        # Snapshot the fully-resolved config for reproducibility
        OmegaConf.save(cfg, run_dir / "config_resolved.yaml")
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
    run_output_dir = str(run_dir)

    # Mirror terminal output into the run dir — rank 0 only, so multi-rank
    # writes can't interleave. Installed AFTER run_dir is known (so we know
    # where to write) but BEFORE model load, so the SD 2.1 weight-loading
    # diagnostics and any warm-up messages are captured too.
    if is_rank_zero():
        install_log_tee(run_dir / "log.txt")

    model = load_model(cfg, init_path=cfg.get("init_path"))
    model.learning_rate = cfg.learning_rate

    if is_rank_zero():
        print("Building train dataloader...")
    train_loader = build_dataloader(cfg.train_dataset.params, cfg.gpu_batch_size,
                                    shuffle=True, drop_last=True)
    if is_rank_zero():
        print(f"  Train windows: {len(train_loader.dataset)} "
              f"(from {len(train_loader.dataset.clips)} unique clips)")
        print("Building val dataloader...")
    val_loader = build_dataloader(cfg.val_dataset.params, cfg.gpu_batch_size,
                                  shuffle=False, drop_last=False)
    if is_rank_zero():
        print(f"  Val windows: {len(val_loader.dataset)} "
              f"(from {len(val_loader.dataset.clips)} unique clips)")

    wrapper = LightningWrapper(model, cfg)

    ckpt_dir = str(Path(run_output_dir) / "checkpoints")

    # Checkpoint: save every N steps (keep all)
    periodic_ckpt = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="th-{step:06d}",
        every_n_train_steps=cfg.save_every_n_steps,
        save_top_k=-1,
    )

    # Checkpoint: save best by val loss
    best_ckpt = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="th-best-{step:06d}-{val_loss:.4f}",
        auto_insert_metric_name=False,
        monitor="val/loss",
        mode="min",
        save_top_k=1,
    )

    # Visualization callback
    vis_cb = VisualizationCallback(
        cfg=cfg,
        val_loader=val_loader,
        output_dir=run_output_dir,
        vis_every_n_steps=cfg.get("val_every_n_steps", 2000),
        n_vis_samples=cfg.get("n_vis_samples", 4),
        vis_ddim_steps=cfg.get("vis_ddim_steps", 20),
    )

    logger = TensorBoardLogger(save_dir=run_output_dir, name="logs")

    trainer = pl.Trainer(
        max_steps=cfg.n_steps,
        accelerator="gpu",
        devices=list(gpus),
        strategy="ddp_find_unused_parameters_true" if len(gpus) > 1 else "auto",
        precision=16,
        callbacks=[vis_cb, best_ckpt, periodic_ckpt],
        logger=logger,
        log_every_n_steps=cfg.logger_freq,
        accumulate_grad_batches=cfg.virtual_batch_size // cfg.gpu_batch_size,
        val_check_interval=1.0,
        limit_val_batches=cfg.get("n_val_batches", 20),
        deterministic=True,
    )

    trainer.fit(wrapper, train_loader, val_loader, ckpt_path=resume)


# ---------------------------------------------------------------------------
# CLI wrapper
# ---------------------------------------------------------------------------
def main():
    """Ad-hoc CLI entry: `python marionette/train.py --config path/to/exp.yaml`."""
    args = parse_args()
    from marionette.config_utils import load_experiment_config
    cfg = load_experiment_config(args.config)
    run_training(
        cfg=cfg,
        output_dir=args.output_dir,
        resume=args.resume,
        gpus=tuple(args.gpus),
    )


if __name__ == "__main__":
    main()
