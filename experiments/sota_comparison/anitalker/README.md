# AniTalker — SOTA Comparison Wrapper

Runs AniTalker (SJTU X-LANCE, [paper](https://arxiv.org/abs/2405.03121),
[code](https://github.com/X-LANCE/AniTalker),
[HuggingFace](https://huggingface.co/taocode/anitalker_ckpts)) against our
benchmark datasets under a uniform CLI so its outputs sit next to
Loki's, SadTalker's, HunyuanPortrait's, and X-Portrait's for
apples-to-apples comparison.

## At a glance

| | |
|---|---|
| Paper | [ACM MM 2024](https://arxiv.org/abs/2405.03121) |
| Input | Source image + driven audio (WAV) — motion-driven by HuBERT features |
| Output | 512×512 face-cropped mp4 at 25 fps (via `--face_sr` GFPGAN upscale; native is 256×256) |
| Env | `anitalker` — Python 3.9 + torch 2.0.1+cu118 |
| Pinned commit | *fill in on first clone — see "1. Clone the baseline"* |

## 0. One-shot setup (recommended)

Everything below (env, pip deps, clone, ~3 GB ckpt bundle from HuggingFace)
in one idempotent script:

```bash
bash experiments/sota_comparison/anitalker/setup_env.sh
```

Safe to re-run — completed steps are detected and skipped. Manual
walkthrough below if you need to debug a specific step.

## 1. Clone the baseline

Upstream is not committed into this repo — it's gitignored under `impl/`.

```bash
cd experiments/sota_comparison/anitalker
git clone https://github.com/X-LANCE/AniTalker.git impl
cd impl && git rev-parse HEAD > ../COMMIT_PIN.txt
```

Record the commit hash in the "At a glance" table on first setup. If your
network to GitHub is flaky, you can drop a zip extract at `impl/` instead —
`setup_env.sh` accepts either.

## 2. Environment

Upstream's README pins **Python 3.9 + torch 1.8.0 + CUDA 11.1**. torch 1.8
predates Ampere (sm_80+) and does NOT run on modern Hopper/Ada cards — the
kernels error with "no kernel image available for execution on the device"
on sm_89+. We bump to `torch 2.0.1+cu118` (same stack used by the
X-Portrait env) — close enough to AniTalker's era that the model code still
loads without changes, modern enough to run on H100/H200 via PTX fallback.

```bash
conda env create -f experiments/sota_comparison/anitalker/env.yml
conda activate anitalker
pip install -r experiments/sota_comparison/anitalker/requirements.txt
```

Our `requirements.txt` keeps upstream's `transformers==4.19.2` pin (the
HuBERT extractor imports APIs that shifted in later versions) and adds
`gfpgan` for the 256→512 face-SR pass.

## 3. Model weights

All checkpoints live in the `taocode/anitalker_ckpts` HuggingFace repo
(~3 GB total). Pull the whole thing into `impl/ckpts/`:

```bash
cd experiments/sota_comparison/anitalker/impl
mkdir -p ckpts
hf download taocode/anitalker_ckpts --repo-type model --local-dir ckpts
```

Resulting layout (upstream's `demo.py` expects these exact names at
`./ckpts/` relative to the repo root):

```
impl/ckpts/
├── stage1.ckpt                        (~188 MB)
├── stage2_audio_only_hubert.ckpt      (~342 MB)   ← our runner's default
├── stage2_full_control_hubert.ckpt    (~359 MB)
├── stage2_full_control_mfcc.ckpt      (~249 MB)
├── stage2_pose_only_hubert.ckpt       (~348 MB)
├── stage2_pose_only_mfcc.ckpt         (~237 MB)
└── chinese-hubert-large/              (~1.2 GB — HuBERT feature model)
    ├── config.json
    ├── preprocessor_config.json
    └── pytorch_model.bin
```

Our runner uses the **hubert_audio_only** variant
(`stage2_audio_only_hubert.ckpt`) by default — upstream's README flags it
as the highest-quality preset. `--stage2_ckpt` on the runner swaps it for
any of the other stage-2 shards if you want to ablate.

## 4. Run

**Prerequisite — curated manifest.** The runner reads
`experiments/sota_comparison/manifests/<dataset>.json`. Build it once via
`dataset/build_manifest.py` (see the top-level
[sota_comparison README](../README.md)).

The runner lives outside the `anitalker` env — it orchestrates, builds the
pair list, and hops into the env per sample via `conda run`. Launch from
the `loki` env:

```bash
conda activate loki

# HDTF cross-identity (A's face + B's audio)
PYTHONPATH=. python experiments/sota_comparison/anitalker/run_inference.py \
    --dataset hdtf \
    --protocol cross_identity \
    --n_samples 200 \
    --clip_duration_s 3.0 \
    --seed 42

# Same-identity reconstruction
PYTHONPATH=. python experiments/sota_comparison/anitalker/run_inference.py \
    --dataset hdtf \
    --protocol same_identity_reconstruction \
    --n_samples 346 \
    --clip_duration_s 3.0 \
    --seed 42
```

Both protocols are in the runner's docstring.

### Protocol notes

- **Audio-driven, not motion-video-driven.** Same shape as SadTalker. The
  adapter extracts `clip_duration_s` seconds of the driver's audio as
  mono 16 kHz WAV and hands it to AniTalker; the model synthesises lip +
  head motion from HuBERT features.
- **Sidecar WAV preference.** When a dataset ships sidecar `.wav` files,
  the adapter reads `driver_clip.audio_path`. HDTF and similar
  muxed-audio datasets → ffmpeg extracts it from the mp4. Same code path
  as SadTalker.
- **Auto HuBERT feature extraction.** `demo.py` detects a missing
  `--test_hubert_path` and auto-extracts the `.npy` using
  `ckpts/chinese-hubert-large/` on the fly. Our adapter points
  `--test_hubert_path` at a per-sample scratch path so each sample's
  features are cached + not re-extracted on re-runs.
- **Face super-resolution (GFPGAN) is on by default.** AniTalker's native
  output is 256×256; we enable `--face_sr` so the output is 512×512 to
  match every other baseline's surface. Pass `--no_face_sr` to disable
  (faster, produces a 256 output).
- **Random reference frame.** Same seeded RNG schedule as every other
  runner. `(protocol, seed, sample_id)` selects the same ref frame across
  SadTalker, HunyuanPortrait, X-Portrait, AniTalker.

## 5. Output layout

```
outputs/sota_comparison/anitalker/<dataset>/<protocol>/run_<timestamp>/
├── config_resolved.json          # full CLI args + git rev
├── scratch/<sample_id>/          # per-sample working files
│   ├── source.png                #   ref frame
│   ├── audio.wav                 #   driver audio, 16 kHz mono
│   ├── hubert.npy                #   auto-extracted HuBERT features
│   └── result/                   #   AniTalker raw mp4 output(s)
├── failed.json                   # (only if any sample errored)
└── samples/<sample_id>/
    └── panel.mp4                 # 512×512 SR output (or 256 native if --no_face_sr)
```

`<sample_id>` is UID-based:
- `same_identity_reconstruction` → `id_0457`
- `cross_identity` → `id_0457_id_0009` (ref uid, driver uid)

Identical naming to every other SOTA baseline → a single glob like
`outputs/sota_comparison/*/hdtf/cross_identity/run_<ts>/samples/id_0457_id_0009/panel.mp4`
gives every baseline's output for the same identity pair.

## 6. Knobs exposed on the CLI

| Flag | Default | Notes |
|---|---|---|
| `--step_T` | 50 | Diffusion denoising steps. |
| `--anitalker_seed` | 0 | AniTalker's internal seed (distinct from `--seed` which drives pair-list + ref-frame selection). |
| `--no_face_sr` | off | Disable GFPGAN 256→512 upscaling. Default leaves SR on so output is 512×512. |
| `--motion_dim` | 20 | Upstream default — do not change without reason. |
| `--decoder_layers` | 2 | Upstream default. |
| `--impl_dir` | `./impl` | Path to the cloned AniTalker repo. |
| `--conda_env` | `anitalker` | Env with AniTalker's torch 2.0.1+cu118 stack. |
| `--stage1_ckpt` | `ckpts/stage1.ckpt` | Stage-1 ckpt path relative to `impl_dir`. |
| `--stage2_ckpt` | `ckpts/stage2_audio_only_hubert.ckpt` | Stage-2 ckpt relative to `impl_dir`. Swap for a different stage-2 shard to ablate. |
| `--n_take` | unbounded | Cap pair list for debug runs. |

## 7. Troubleshooting

- **`ImportError: cannot import name 'cached_download' from 'huggingface_hub'`** —
  `huggingface_hub` got upgraded past 0.10.x. Force-reinstall:
  `pip install --force-reinstall "huggingface_hub[cli]==0.10.1"`.
- **`transformers` import error — `GraphModule` etc.** — same story with
  `transformers`. Force-reinstall to the pinned 4.19.2.
- **`GFPGAN` fails to find its weights on first run** — it auto-downloads
  them to `~/.cache/gfpgan/`. First `--face_sr` call takes ~30 s longer
  than subsequent ones; not a real error.
- **"no kernel image available for execution on the device"** — you're
  probably on sm_90 (H100) and something in the env reverted to torch 1.8.
  Verify `python -c "import torch; print(torch.version.cuda)"` prints
  `11.8` and `torch.__version__` starts with `2.0.1`.
- **HuBERT auto-extraction fails with "Please install transformers
  module first"** — `transformers==4.19.2` isn't installed. `pip install
  -r experiments/sota_comparison/anitalker/requirements.txt`.
