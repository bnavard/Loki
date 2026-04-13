"""
Quick integration test for the training pipeline.

Runs 2 training steps + 1 validation step with real data (2 clips) to verify
the full pipeline works end-to-end: dataset loading, VAE encoding, conditioning,
UNet forward, loss computation, backward pass.

Usage:
    cd <repo_root>
    PYTHONPATH=. python talkinghead/tests/test_training_integration.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
torch.backends.cudnn.enabled = False

import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Use 2 real clips for the test
TEST_IDS = [
    "39Y_gFC9SmY_NA_1123.760_1128.801",
    "39Y_gFC9SmY_NA_1128.801_1133.843",
]

REPO_ROOT  = Path(__file__).resolve().parent.parent.parent
ID_FILE    = REPO_ROOT / "talkinghead" / "tests" / "_test_ids.txt"
CONFIG     = REPO_ROOT / "talkinghead" / "configs" / "talking_head.yaml"


def write_test_ids():
    ID_FILE.write_text("\n".join(TEST_IDS) + "\n")


def make_test_cfg():
    cfg = OmegaConf.load(str(CONFIG))
    # Override dataset to use tiny test ID list for both train and val
    for ds_key in ["train_dataset", "val_dataset"]:
        if ds_key in cfg:
            cfg[ds_key].params.id_list_path = str(ID_FILE)
    # Smaller batch for speed
    cfg.gpu_batch_size = 1
    return cfg


def build_loader(ds_cfg, batch_size=1):
    from marionette.data.video_dataset import TalkingHeadDataset
    dataset = TalkingHeadDataset(**OmegaConf.to_container(ds_cfg, resolve=True))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                      drop_last=True)


def to_device(batch, device):
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


# ===========================================================================
# Tests
# ===========================================================================
def test_dataset_loading():
    """Dataset returns correct keys and shapes."""
    cfg = make_test_cfg()
    loader = build_loader(cfg.train_dataset.params, batch_size=1)
    batch = next(iter(loader))

    assert "jpg" in batch,   "Missing 'jpg' key"
    assert "audio" in batch, "Missing 'audio' key"
    assert "hint" in batch,  "Missing 'hint' key"

    T = cfg.train_dataset.params.n_frames
    assert batch["jpg"].shape[1] == T, f"jpg T={batch['jpg'].shape[1]}, expected {T}"
    assert batch["audio"].shape[1] == T, f"audio T={batch['audio'].shape[1]}, expected {T}"
    assert batch["hint"]["reference_mask"].shape[1] == T

    hint = batch["hint"]
    assert "verts_2d" in hint
    assert "offsets_3d" in hint
    assert "reference_mask" in hint
    print(f"  jpg:       {tuple(batch['jpg'].shape)}")
    print(f"  audio:     {tuple(batch['audio'].shape)}")
    print(f"  verts_2d:  {tuple(hint['verts_2d'].shape)}")
    print(f"  ref_mask:  {tuple(hint['reference_mask'].shape)}")


def test_training_step():
    """Full training forward + backward for 2 steps."""
    cfg = make_test_cfg()
    loader = build_loader(cfg.train_dataset.params, batch_size=1)

    from controlnet.ldm.util import instantiate_from_config
    model = instantiate_from_config(cfg.model)

    # Load SD 2.1 weights if available
    init_path = cfg.get("init_path")
    if init_path and Path(init_path).exists():
        sd = torch.load(init_path, map_location="cpu")
        model.load_state_dict(sd.get("state_dict", sd), strict=False)

    model.learning_rate = cfg.learning_rate
    model.to(DEVICE)
    model.train()

    optimizer = model.configure_optimizers()

    for step, batch in enumerate(loader):
        if step >= 2:
            break
        batch = to_device(batch, DEVICE)

        z, cond = model.get_input(batch, model.first_stage_key)
        loss, loss_dict = model(z, cond)

        assert loss.ndim == 0, f"Loss should be scalar, got {loss.shape}"
        assert loss.item() > 0, f"Loss should be positive, got {loss.item()}"
        assert "train/loss" in loss_dict

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"  Step {step}: loss={loss.item():.4f}")

    print(f"  loss_dict keys: {list(loss_dict.keys())}")


def test_validation_step():
    """Validation forward (no backward)."""
    cfg = make_test_cfg()
    loader = build_loader(cfg.val_dataset.params if "val_dataset" in cfg
                          else cfg.train_dataset.params, batch_size=1)

    from controlnet.ldm.util import instantiate_from_config
    model = instantiate_from_config(cfg.model)

    init_path = cfg.get("init_path")
    if init_path and Path(init_path).exists():
        sd = torch.load(init_path, map_location="cpu")
        model.load_state_dict(sd.get("state_dict", sd), strict=False)

    model.learning_rate = cfg.learning_rate
    model.to(DEVICE)
    model.eval()

    batch = next(iter(loader))
    batch = to_device(batch, DEVICE)

    with torch.no_grad():
        z, cond = model.get_input(batch, model.first_stage_key, force_conditional=True)
        loss, loss_dict = model(z, cond)

    assert loss.ndim == 0
    assert "val/loss" in loss_dict
    print(f"  Val loss: {loss.item():.4f}")
    print(f"  loss_dict keys: {list(loss_dict.keys())}")


# ===========================================================================
# Runner
# ===========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print(f"Training Integration Test  (device: {DEVICE})")
    print("=" * 60)

    write_test_ids()

    tests = [
        ("Dataset loading",   test_dataset_loading),
        ("Training step x2",  test_training_step),
        ("Validation step",   test_validation_step),
    ]

    passed, failed = 0, 0
    for name, fn in tests:
        print(f"\n--- {name} ---")
        try:
            fn()
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as e:
            import traceback
            print(f"  [FAIL] {name}")
            print(f"         {e}")
            traceback.print_exc()
            failed += 1

    # Cleanup
    if ID_FILE.exists():
        ID_FILE.unlink()

    print("\n" + "-" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
