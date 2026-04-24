# X-Portrait — SOTA Comparison Wrapper

Runs X-Portrait (ByteDance, [paper](https://arxiv.org/abs/2403.15931),
[code](https://github.com/bytedance/X-Portrait),
[project page](https://byteaigc.github.io/x-portrait/)) against our benchmark
datasets under a uniform CLI so its outputs sit next to Marionette's,
SadTalker's, and HunyuanPortrait's for apples-to-apples comparison.

## At a glance

| | |
|---|---|
| Paper | [SIGGRAPH 2024](https://arxiv.org/abs/2403.15931) |
| Input | Source image + driver video (motion-driven; no audio) |
| Output | 512×512 face-cropped mp4 at the driver's fps |
| Env | `xportrait` — Python 3.9 + torch 2.0.1+cu118 |
| Pinned commit | *fill in on first clone — see "1. Clone the baseline"* |

## 0. One-shot setup (recommended)

Everything below (env, pip deps, clone, ~3 GB checkpoint from Google Drive)
in one idempotent script:

```bash
bash experiments/sota_comparison/xportrait/setup_env.sh
```

Safe to re-run — completed steps are detected and skipped. Manual
walkthrough below if you need to debug a specific step.

## 1. Clone the baseline

Upstream is not committed into this repo — it's gitignored under `impl/`.

```bash
cd experiments/sota_comparison/xportrait
git clone https://github.com/bytedance/X-Portrait.git impl
cd impl && git rev-parse HEAD > ../COMMIT_PIN.txt
```

Record the commit hash in the "At a glance" table on first setup.

## 2. Environment

Upstream's README pins **Python 3.9 + CUDA 11.8**. Their own pins force
torch 2.0.x via `xformers>=0.0.22` + `triton==2.0.0`, so we can't reuse the
SadTalker / HunyuanPortrait torch 2.1+cu121 stack here. `torch 2.0.1+cu118`
still runs on Hopper/Ada via PTX fallback (slightly slower kernels than
native cu121 but correct); on Ampere it runs natively.

```bash
conda env create -f experiments/sota_comparison/xportrait/env.yml
conda activate xportrait
pip install -r experiments/sota_comparison/xportrait/requirements.txt
```

Our `requirements.txt` mirrors upstream's pins with one addition: `gdown`,
used by `setup_env.sh` to pull the checkpoint from Google Drive. Upstream's
own `env_install.sh` ends with `sudo apt install python3-tk` — that's only
needed if you use matplotlib's interactive backend, which `test_xportrait.py`
does NOT, so we skip it.

## 3. Model weights

Upstream hosts the checkpoint on Google Drive:
[drive.google.com/drive/folders/1Bq0n-w1VT5l99CoaVg02hFpqE5eGLo9O](https://drive.google.com/drive/folders/1Bq0n-w1VT5l99CoaVg02hFpqE5eGLo9O?usp=sharing).
The specific file we need is `model_state-415001.th` (~3 GB, file id
`1VOpVg25EQTUlbHOvuLEFi8rBhVd2KlxQ`).

```bash
cd experiments/sota_comparison/xportrait/impl
mkdir -p checkpoint
gdown --id 1VOpVg25EQTUlbHOvuLEFi8rBhVd2KlxQ -O checkpoint/model_state-415001.th
```

**Do not rename** — upstream's demo script (`scripts/test_xportrait.sh`) and
our adapter both look for this exact filename.

Expected layout after download:

```
impl/
├── core/test_xportrait.py
├── config/cldm_v15_appearance_pose_local_mm.yaml
└── checkpoint/
    └── model_state-415001.th        (~3 GB)
```

## 4. Run

**Prerequisite — curated manifest.** The runner reads
`experiments/sota_comparison/manifests/<dataset>.json`. Build it once via
`dataset/build_manifest.py` (see the top-level
[sota_comparison README](../README.md)).

The runner lives outside the `xportrait` env — it orchestrates, builds the
pair list, and hops into the env per sample via `conda run`. Launch from
the `marionette` env:

```bash
conda activate marionette

# TalkVid cross-identity (A's face doing B's motion)
PYTHONPATH=. python experiments/sota_comparison/xportrait/run_inference.py \
    --dataset talkvid \
    --protocol cross_identity \
    --n_samples 125 \
    --clip_duration_s 5.0 \
    --seed 42

# HDTF needs --clip_duration_s 3.0 since the mirror's clips are ~3.24 s
PYTHONPATH=. python experiments/sota_comparison/xportrait/run_inference.py \
    --dataset hdtf \
    --protocol same_identity_reconstruction \
    --n_samples 346 \
    --clip_duration_s 3.0 \
    --seed 42
```

All four `(dataset, protocol)` combos are in the runner's docstring.

### Protocol notes

- **Motion-driven, not audio-driven.** Output mp4 is silent. Matches
  HunyuanPortrait's behaviour; mux the driver audio downstream at
  analysis time if needed.
- **Auto-best-frame selection.** `--best_frame -1` (the default) invokes
  X-Portrait's `find_best_frame_byheadpose_fa` — it runs face-alignment
  on the source image and every driver frame, then picks the driver
  frame whose 68-point head pose most closely matches the source. Upstream's
  demo script hardcodes `--best_frame 36` for its specific clip; that's
  useless for a 125-sample sweep. If you want a fixed index (e.g. `--best_frame 0`)
  across every sample, pass it on the CLI.
- **Random reference frame.** Like every other SOTA runner, `ref_frame_idx`
  is drawn from the same seeded RNG schedule — same `(protocol, seed,
  sample_id)` picks the same ref frame across SadTalker, HunyuanPortrait,
  and X-Portrait.
- **Driver trimming.** We ffmpeg-trim the driver to `clip_duration_s` then
  pass `--out_frames -1` so X-Portrait processes every trimmed frame.
  25 fps × 5 s = 125 output frames.
- **`clip_duration_s` vs dataset pool length.** Same rule as the other
  runners: the pairing module filters out clips shorter than
  `clip_duration_s`. TalkVid (5.0 s val clips) accepts 5.0. HDTF (~3.24 s
  chunks) needs 3.0.

## 5. Output layout

```
outputs/sota_comparison/xportrait/<dataset>/<protocol>/run_<timestamp>/
├── config_resolved.json          # full CLI args + git rev
├── scratch/<sample_id>/          # per-sample working files
│   ├── source.png                #   ref frame
│   ├── driver.mp4                #   trimmed driver video
│   └── result/                   #   X-Portrait raw mp4 output
├── failed.json                   # (only if any sample errored)
└── samples/<sample_id>/
    └── panel.mp4                 # X-Portrait's generated video
```

`<sample_id>` is UID-based:
- `same_identity_reconstruction` → `id_0457`
- `cross_identity` → `id_0457_id_0009` (ref uid, driver uid)

Identical naming to SadTalker / HunyuanPortrait → a single
`outputs/sota_comparison/*/talkvid/cross_identity/run_<ts>/samples/id_0457_id_0009/panel.mp4`
glob gives you every baseline's output for the same identity pair.

## 6. Knobs exposed on the CLI

| Flag | Default | Notes |
|---|---|---|
| `--uc_scale` | 5 | Unconditional guidance scale (CFG-like). Upstream demo default. |
| `--ddim_steps` | 30 | Denoising steps. |
| `--num_mix` | 4 | Overlap frames for prompt-travelling inference. |
| `--xp_seed` | 999 | Internal X-Portrait seed. Kept separate from the runner's `--seed` (which drives pair list + ref-frame selection). |
| `--best_frame` | -1 | `-1` → auto. Any non-negative value forces that exact driver frame across every sample. |
| `--impl_dir` | `./impl` | Path to the cloned X-Portrait repo. |
| `--conda_env` | `xportrait` | Env with X-Portrait's torch 2.0.1 stack. |
| `--ckpt_rel` | `checkpoint/model_state-415001.th` | Checkpoint path relative to `impl_dir`. |
| `--model_config` | `config/cldm_v15_appearance_pose_local_mm.yaml` | Model config relative to `impl_dir`. |
| `--n_take` | unbounded | Cap pair list for debug runs. |

## 7. Troubleshooting

- **`gdown` fails with "Permission denied" / "cannot retrieve file"** — the
  Google Drive link is rate-limited after many downloads. Wait, or open the
  folder URL in a browser to warm the session, then rerun `gdown`.
  `gdown --id <id> -O <path>` resumes partial downloads.
- **`xformers` ImportError on Hopper/Ada** — `xformers==0.0.22` is tied to
  torch 2.0.x and compiles against CUDA 11.8. It runs but may emit kernel-
  fallback warnings on sm_89+. Non-fatal; test output is unchanged.
- **`face-alignment` fails to detect a face in the source image** — happens
  on occasional transition/motion-blur frames. The adapter's seeded RNG
  gives a different ref frame per `--seed`, so rerun with a different seed
  to resample.
- **`RuntimeError: CUDA out of memory` during inference** — X-Portrait
  loads the SD1.5-based ControlNet + motion module onto one GPU. On cards
  with < 24 GB VRAM, lower `--ddim_steps` (30 → 20) or `--num_mix` (4 → 2)
  to reduce peak activations.
- **"undetected faces!!" spam during inference** — X-Portrait's face-
  alignment path fails on some driver frames and falls back to default
  crop coords (196,196,320,320). Non-fatal; the model still produces an
  output. Common on clips with rapid pan / occlusion.
