# Talking-Head Tests

Test scripts for verifying the talking-head diffusion pipeline. Each test can be run independently from the repository root.

## Table of Contents

- [test_pipeline.py](#test_pipelinepy) — Unit tests with random tensors (no data required)
- [test_training_integration.py](#test_training_integrationpy) — End-to-end training integration test
- [test_dataset.py](#test_datasetpy) — Visual dataset test with background stabilization comparison

## Tests

### test_pipeline.py

**Purpose:** Unit tests using random tensors to verify tensor shapes flow correctly through every module. No real data, checkpoints, or FLAME assets required.

**What it tests:**
- `THConditioning` — channel count property, conditional/unconditional output shapes
- `AudioEncoder` — waveform to context token conversion (no pretrained download)
- `THUnetModel` — full UNet forward pass with spatial + audio conditioning
- `THDiffusion` — training step (get_input + p_losses) with mock VAE
- `THSampler` — inference loop (2 DDIM steps, fake model)

**Run:**
```bash
python talkinghead/tests/test_pipeline.py
# or
python -m pytest talkinghead/tests/test_pipeline.py -v
```

---

### test_training_integration.py

**Purpose:** End-to-end integration test that runs 2 training steps + 1 validation step with real data (2 clips). Verifies the full pipeline: dataset loading, VAE encoding, conditioning, UNet forward, loss computation, backward pass.

**Requirements:** Real data must exist at `data/talkvid/`, `data/flowface/`, and `data/talkvid/audio/`.

**Run:**
```bash
PYTHONPATH=. python talkinghead/tests/test_training_integration.py
```

---

### test_dataset.py

**Purpose:** Visual integration test for `TalkingHeadDataset` with and without background stabilization. Loads real samples, validates shapes, and saves comparison visualizations to inspect data quality.

**What it outputs** (to `outputs/dataset_vis/`):
- `comparison_XXX.png` — Side-by-side: frames without vs with background stabilization. Use this to verify the background plate is working and jitter is eliminated.
- `bg_breakdown_XXX.png` — Three rows per sample: composited frames, cropped background plate, and foreground mask. Use this to inspect the quality of the background plate and mask alignment.
- `bg_plate_full.png` — The full uncropped background plate for one clip, showing the aggregated background with the head removed.

**Requirements:** Real data must exist at `data/talkvid/`, `data/flowface/` (including `bg/cam0/` masks).

**Run:**
```bash
python talkinghead/tests/test_dataset.py
```
