# HunyuanPortrait — SOTA Comparison Wrapper

Runs HunyuanPortrait (Tencent, [paper](https://arxiv.org/abs/2503.18860),
[code](https://github.com/Tencent-Hunyuan/HunyuanPortrait),
[HuggingFace](https://huggingface.co/tencent/HunyuanPortrait)) against our
benchmark datasets under a uniform CLI so its outputs sit next to
Marionette's and SadTalker's for apples-to-apples comparison.

## At a glance

| | |
|---|---|
| Paper | [CVPR 2025](https://arxiv.org/abs/2503.18860) |
| Input | Source image + driver video (motion-driven; no audio) |
| Output | 512×512 face-cropped mp4 at the driver's fps |
| Env | `hunyuan_portrait` — Python 3.10 + torch 2.1.0+cu121 |
| Pinned commit | *fill in on first clone — see "1. Clone the baseline"* |

## 0. One-shot setup (recommended)

Everything below (env, pip deps, clone, ~5–6 GB of weights) in one
idempotent script:

```bash
bash experiments/sota_comparison/hunyuan_portrait/setup_env.sh
```

Safe to re-run — per-file sentinels skip the big downloads that already
landed, so a flaky HuggingFace connection just needs another invocation.
Manual walkthrough below if you need to debug a specific step.

## 1. Clone the baseline

Upstream is not committed into this repo — it's gitignored under `impl/`.

```bash
cd experiments/sota_comparison/hunyuan_portrait
git clone https://github.com/Tencent-Hunyuan/HunyuanPortrait.git impl
cd impl && git rev-parse HEAD > ../COMMIT_PIN.txt
```

Record the commit hash in the "At a glance" table on first setup.

## 2. Environment

Upstream's README says "pip3 install torch torchvision torchaudio" with no
pin. We fix the stack to `torch==2.1.0+cu121` (same as the `sadtalker` env)
so runs are reproducible and don't drift across pip-solver updates.

```bash
conda env create -f experiments/sota_comparison/hunyuan_portrait/env.yml
conda activate hunyuan_portrait
pip install -r experiments/sota_comparison/hunyuan_portrait/requirements.txt
```

Our `requirements.txt` deviates from upstream's in two spots — see the
inline comments. Briefly: we pin `onnxruntime-gpu==1.19.2` instead of
upstream's unpinned `onnxruntime` + `onnxruntime-gpu` pair (installing both
together is ambiguous; GPU variant covers the CPU EP too).

## 3. Model weights

Follow upstream's download recipe verbatim — it fetches pieces from
three HuggingFace repos plus two wget URLs.

```bash
cd experiments/sota_comparison/hunyuan_portrait/impl
mkdir -p pretrained_weights
cd pretrained_weights

# Stable Video Diffusion configs (JSONs only)
huggingface-cli download --resume-download stabilityai/stable-video-diffusion-img2vid-xt \
    --local-dir . --include "*.json"

# yoloface (face detector used by test_preprocess)
wget -c https://huggingface.co/LeonJoe13/Sonic/resolve/main/yoloface_v5m.pt

# SVD VAE weights
wget -c https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main/vae/diffusion_pytorch_model.fp16.safetensors -P vae

# ArcFace (identity embedding)
wget -c https://huggingface.co/FoivosPar/Arc2Face/resolve/da2f1e9aa3954dad093213acfc9ae75a68da6ffd/arcface.onnx

# HunyuanPortrait's own weights — dino.pth, expression.pth, headpose.pth,
# image_proj.pth, motion_proj.pth, pose_guider.pth, unet.pth
huggingface-cli download --resume-download tencent/HunyuanPortrait \
    --local-dir hyportrait
```

Expected layout after download:

```
impl/pretrained_weights/
├── arcface.onnx
├── hyportrait/
│   ├── dino.pth
│   ├── expression.pth
│   ├── headpose.pth
│   ├── image_proj.pth
│   ├── motion_proj.pth
│   ├── pose_guider.pth
│   └── unet.pth
├── scheduler/
│   └── scheduler_config.json
├── unet/
│   └── config.json
├── vae/
│   ├── config.json
│   └── diffusion_pytorch_model.fp16.safetensors
└── yoloface_v5m.pt
```

Total ~5–6 GB.

## 4. Run

**Prerequisite — curated manifest.** The runner reads
`experiments/sota_comparison/manifests/<dataset>.json`. Build it once via
`dataset/build_manifest.py` (see the top-level
[sota_comparison README](../README.md)).

The runner lives outside the `hunyuan_portrait` env — it orchestrates,
builds the pair list, and hops into the env per sample via `conda run`.
Launch from the `marionette` env:

```bash
conda activate marionette

# TalkVid cross-identity (A's face doing B's motion)
PYTHONPATH=. python experiments/sota_comparison/hunyuan_portrait/run_inference.py \
    --dataset talkvid \
    --protocol cross_identity \
    --n_samples 125 \
    --clip_duration_s 5.0 \
    --seed 42

# HDTF needs --clip_duration_s 3.0 since the mirror's clips are ~3.24 s
PYTHONPATH=. python experiments/sota_comparison/hunyuan_portrait/run_inference.py \
    --dataset hdtf \
    --protocol same_identity_reconstruction \
    --n_samples 346 \
    --clip_duration_s 3.0 \
    --seed 42
```

All four `(dataset, protocol)` combos are in the runner's docstring.

### Protocol notes

- **Motion-driven, not audio-driven.** Output mp4 is silent. For
  qualitative review of cross-identity samples where you want the
  driver's voice, mux the driver's audio at analysis time — we don't do
  it here to keep the on-disk artefact identical to what the model
  produced.
- **Random reference frame.** The adapter draws `ref_frame_idx` uniformly
  from `[0, ref_clip.n_frames)` under the runner's seeded RNG. Shares the
  seed schedule with SadTalker, so a given `(protocol, seed, sample_id)`
  picks the same ref frame across every baseline — keeps cross-baseline
  frame-for-frame comparison honest.
- **Driver trimming.** We ffmpeg-trim the driver to `clip_duration_s`
  before handing it to HunyuanPortrait's `inference.py`. Every frame in
  the trimmed video is processed. At 25 fps × 5 s that's 125 frames
  generated.
- **`clip_duration_s` vs dataset pool length.** The pairing module filters
  out clips shorter than `clip_duration_s`. TalkVid's val manifest is 125
  clips of exactly 5.0 s — passing `5.0` keeps all 125. HDTF's mirror is
  pre-chunked to ~3.24 s — pass `3.0` (anything larger wipes the pool).

## 5. Output layout

```
outputs/sota_comparison/hunyuan_portrait/<dataset>/<protocol>/run_<timestamp>/
├── config_resolved.json          # full CLI args + git rev
├── scratch/<sample_id>/          # per-sample working files
│   ├── source.png                #   ref frame
│   ├── driver.mp4                #   trimmed driver video
│   ├── hunyuan-portrait.yaml     #   patched config (output_dir = absolute)
│   └── result/<ts>_<img>_<vid>/  #   raw HunyuanPortrait output
├── failed.json                   # (only if any sample errored)
└── samples/<sample_id>/
    └── panel.mp4                 # HunyuanPortrait's cropped.mp4
```

`<sample_id>` is UID-based:
- `same_identity_reconstruction` → `id_0457`
- `cross_identity` → `id_0457_id_0009` (ref uid, driver uid)

`panel.mp4` is HunyuanPortrait's `cropped.mp4` (512×512 face-crop
generation), NOT `full_resolution.mp4` (paste-back onto original source).
The cropped face is the surface every other baseline and Marionette
produce, so cross-model metric sweeps operate on a common representation.

## 6. Knobs exposed on the CLI

| Flag | Default | Notes |
|---|---|---|
| `--num_inference_steps` | 25 | Denoising steps. Upstream default. |
| `--motion_bucket_id` | 0 | SVD motion bucket. 0 = modest motion; upstream default. |
| `--n_sample_frames` | 25 | Inner pipeline frame batch; memory-bound. Lower for small GPUs. |
| `--no_arcface` | off | Disable ArcFace identity conditioning. Faster, mild identity-fidelity drop. |
| `--min_appearance_guidance` / `--max_appearance_guidance` | 2.0 / 2.0 | CFG scale range for appearance conditioning. |
| `--min_motion_guidance` / `--max_motion_guidance` | 2.0 / 2.0 | CFG scale range for motion conditioning. |
| `--impl_dir` | `./impl` | Path to the cloned HunyuanPortrait repo. |
| `--conda_env` | `hunyuan_portrait` | Env with HunyuanPortrait's torch stack. |
| `--n_take` | unbounded | Cap pair list for debug runs. |

## 7. Troubleshooting

- **`ImportError: cannot import name 'AutoencoderKLTemporalDecoder'`** —
  `diffusers` got upgraded past the pin. Force-reinstall:
  `pip install --force-reinstall diffusers==0.29.0`.
- **ArcFace silently slow** — `onnxruntime` grabbed the CPU execution
  provider because `onnxruntime-gpu` wasn't picked up. Check
  `python -c "import onnxruntime as ort; print(ort.get_available_providers())"` —
  `CUDAExecutionProvider` must be in the list. If not, force-reinstall
  `onnxruntime-gpu==1.19.2`.
- **`torch.cuda.OutOfMemoryError` during facerender** — drop
  `--n_sample_frames` from 25 → 12 (halves memory at the cost of more
  inner-loop iterations).
- **HuggingFace download hangs** — `huggingface-cli` sometimes gets
  rate-limited on fresh installs. Re-run with `--resume-download`; the
  flag is already in the commands above.
- **yoloface fails on a specific source frame** — random seeding picks
  frames uniformly, so occasional transitions/occlusions land a bad
  frame. Rerun the sample with a different `--seed` or bump
  `--n_take` past the failure to skip.
