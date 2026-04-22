# Audio Ablation

Does the wav2vec2 audio cross-attention pathway meaningfully improve
generation quality — especially lip sync — over the purely visual/FLAME
conditioning path? A controlled, two-arm ablation against the canonical
baseline.

## Arms

| Arm | Recipe | Entry point |
|---|---|---|
| **audio_on**  | Canonical baseline, audio cross-attention active                  | [`experiments/marionette_baseline/run.py`](../marionette_baseline/run.py) |
| **audio_off** | Canonical baseline + [`overlays/audio/off.yaml`](../../marionette/configs/overlays/audio/off.yaml) | [`run.py`](run.py) (this folder)                                        |

The `overlays/audio/off.yaml` overlay:
- sets `model.params.audio_encoder_config: null` — the wav2vec2 encoder is never instantiated.
- sets `model.params.unet_config.params.use_audio_context: false` — every transformer block in the gen UNet skips its cross-attention pass.

`MarionetteDiffusion.__init__` validates that these two flags are consistent,
so there is no way to leave one half of the pathway live by accident.

## Reproducibility guarantees

Both arms inherit the *same* `marionette/configs/base.yaml`. The ablation is
controlled — only the audio pathway differs. What keeps it apples-to-apples:

- **Same seed.** `cfg.seed: 42` lives in the base config; both arms inherit
  it unchanged. `run_training` calls `pl.seed_everything(cfg.seed,
  workers=True)`, so torch / numpy / python / per-worker DataLoader seeds
  are all deterministic and identical across arms.
- **Same data shuffle.** `DataLoader(shuffle=True)` + deterministic worker
  seeds means both arms walk training batches in exactly the same order.
- **Same reference-sampling order.** `TalkingHeadDataset` uses
  `ref_sampling_seed=0` (default) XOR'd with sample index to pick the ref
  frame — this is identical across arms.
- **Same pretrained init.** Both arms load SD 2.1 weights from
  `data/models/v2-1_512-ema-pruned.ckpt` via the same
  `RefFeatureExtractor.load_sd21_into_ref` key-duplication path.
- **Same non-audio hyperparameters.** Learning rate, batch size, gradient
  accumulation, n_steps, noise schedule, CFG probability, validation cadence
  — all inherited from base.yaml and untouched by the overlay.
- **Deterministic ops.** `train.py` sets `torch.backends.cudnn.enabled =
  False` and passes `deterministic=True` to `pl.Trainer`, which forces
  CUDA kernels to produce identical outputs across runs on the same
  hardware.

The audio-off arm does *not* instantiate or train a wav2vec2 encoder, so its
optimiser parameter list is smaller by ~1M params. Step count and wall-clock
step duration are nonetheless directly comparable — the gen UNet is
identical across arms.

## Launching the audio_off arm

```bash
conda activate marionette

# Single GPU
PYTHONPATH=. python experiments/ablate_audio/run.py

# Multi-GPU (DDP)
PYTHONPATH=. python experiments/ablate_audio/run.py --gpus 0 1 2 3

# Resume
PYTHONPATH=. python experiments/ablate_audio/run.py \
    --resume outputs/ablate_audio/audio_off/run_YYYYmmdd_HHMMSS/checkpoints/th-<step>.ckpt
```

The audio_on arm is the existing baseline — launch it (or reuse its
checkpoint) via:

```bash
PYTHONPATH=. python experiments/marionette_baseline/run.py --gpus 0 1 2 3
```

## Outputs

```
outputs/
├── marionette_baseline/run_<ts>/       # audio_on arm (existing baseline)
│   └── checkpoints/
└── ablate_audio/
    └── audio_off/run_<ts>/             # this experiment
        ├── config_resolved.yaml
        ├── checkpoints/
        ├── logs/
        └── visualizations/
```

## Structure

```
experiments/ablate_audio/
├── README.md
├── run.py
└── configs/
    └── audio_off.yaml       # base + audio/off overlay + output_dir
```

`audio_on.yaml` intentionally does not exist here — the audio-on arm is
already covered by `marionette_baseline/`. Duplicating it would invite the
two "audio on" configs to drift out of sync.
