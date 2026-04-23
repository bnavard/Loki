# SadTalker — SOTA Comparison Wrapper

Runs SadTalker (Zhang et al., CVPR 2023 — [paper](https://arxiv.org/abs/2211.12194),
[code](https://github.com/OpenTalker/SadTalker)) against our benchmark
datasets under a uniform CLI so its outputs sit next to Marionette's for
apples-to-apples comparison.

## At a glance

| | |
|---|---|
| Trained on    | VoxCeleb |
| Paper eval on | HDTF, first 8 s of 346 videos, **same-identity reconstruction** |
| Input        | Single source image + driven audio (audio-driven only) |
| Output       | H×W ∈ {256, 256} or {512, 512} mp4 with driver audio muxed in |
| Env          | `sadtalker` — Python 3.8 + torch 2.1.0+cu121 (deviates from upstream; see below) |
| Pinned commit | `<fill in on first clone — see "1. Clone the baseline" below>` |

## 1. Clone the baseline

The upstream repo is not committed into this repo — it's gitignored under
`impl/` so history stays clean. Clone it and pin the commit you ran:

```bash
cd experiments/sota_comparison/sadtalker
git clone https://github.com/OpenTalker/SadTalker.git impl
cd impl && git rev-parse HEAD > ../COMMIT_PIN.txt
```

Record the hash from `COMMIT_PIN.txt` into the "At a glance" table above on
first setup. If you re-clone later and the hash moves, regenerate outputs —
SadTalker has occasional silent behaviour changes on main.

## 2. Environment

Upstream README pins `torch==1.12.1+cu113`, which does **not** run on
modern Hopper/Ada cards (sm_89+). We use `torch==2.1.0+cu121` plus matching
numpy/numba/llvmlite bumps — verified on H200 (sm_90). Use [env.yml](env.yml)
+ [requirements.txt](requirements.txt) in this folder; **do not** use
`impl/requirements.txt` (its pins conflict with torch 2.1).

```bash
conda env create -f experiments/sota_comparison/sadtalker/env.yml
conda activate sadtalker
pip install -r experiments/sota_comparison/sadtalker/requirements.txt
```

## 3. Model weights

```bash
cd experiments/sota_comparison/sadtalker/impl
bash scripts/download_models.sh
```

This pulls checkpoints into `impl/checkpoints/` and `impl/gfpgan/` (the
latter only if you use `--enhancer gfpgan`). ~2 GB total.

## 4. Run

**Prerequisite — curated manifest.** The runner reads
`experiments/sota_comparison/manifests/<dataset>.json`, a one-clip-per-
identity pool with stable `id_XXXX` UIDs. Build it once per dataset (committed
to git, frozen after that):

```bash
conda activate marionette
PYTHONPATH=. python experiments/sota_comparison/dataset/build_manifest.py \
    --dataset hdtf
```

The runner lives outside the `sadtalker` env — it orchestrates, builds the
pair list from the manifest, and hops into the `sadtalker` env per sample
via `conda run`. So you run it from the `marionette` env:

```bash
conda activate marionette

# Paper protocol (same-identity reconstruction) — first pass
PYTHONPATH=. python experiments/sota_comparison/sadtalker/run_inference.py \
    --dataset         hdtf \
    --protocol        same_identity_reconstruction \
    --n_samples       346 \
    --clip_duration_s 3.0 \
    --seed            42

# Cross-identity voice transfer (complements the paper protocol)
PYTHONPATH=. python experiments/sota_comparison/sadtalker/run_inference.py \
    --dataset         hdtf \
    --protocol        cross_identity \
    --n_samples       200 \
    --clip_duration_s 3.0 \
    --seed            42
```

### Protocol notes

- **`clip_duration_s = 3.0` instead of the paper's 8.0** — our HDTF mirror
  (`data/benchmark/hdtf/clips/`) is pre-chunked into ~3.24-s segments. The
  paper protocol's "first 8s of a full video" isn't directly runnable here.
  Option: stitch consecutive chunks per speaker into synthetic 8-s clips
  via ffmpeg — left as a follow-up for strict paper-protocol parity.
- **Random reference frame** — the adapter draws `ref_frame_idx` uniformly
  from `[0, ref_clip.n_frames)` under the top-level `--seed`. Frame 0 is
  often a transition / mouth-open / blurred frame in HDTF, and "always
  frame 0" would bias the identity anchor. Random is reproducible because
  the RNG is seeded once per run.
- **`pose_style = 0`** — SadTalker carries 46 learned speaker-style
  buckets (see the upstream's Audio2Pose CVAE `classbias` parameter).
  These are unlabelled latent clusters from VoxCeleb training; each just
  biases head-motion character, not lip sync. Fixing to 0 across all
  samples keeps the comparison reproducible. If you want to study how
  style choice changes identity metrics later, sweep this as a separate
  axis.
- **`size = 512`, `preprocess = crop`** — matches Marionette's 512×512
  face-cropped output so the two models' panels are visually directly
  comparable.

## 5. Output layout

```
outputs/sota_comparison/sadtalker/hdtf/<protocol>/run_<timestamp>/
├── config_resolved.json          # full CLI args + git rev
├── scratch/<sample_id>/          # per-sample working files (source.png, audio.wav)
├── failed.json                   # (only if any sample errored)
└── samples/<sample_id>/
    └── panel.mp4
```

`<sample_id>` is UID-based:
- `same_identity_reconstruction` → `id_0457`
- `cross_identity` → `id_0457_id_0009` (ref uid, driver uid)

Because UIDs come from the frozen benchmark manifest, `id_0457/panel.mp4`
refers to the same physical person in every baseline's output tree — a
single `outputs/**/samples/id_0457/panel.mp4` glob compares SadTalker,
LivePortrait, Marionette, etc. 1-to-1 on the same identity.

## 6. Knobs exposed on the CLI

| Flag | Default | Notes |
|---|---|---|
| `--size` | 512 | 256 \| 512. Face resolution. |
| `--preprocess` | `crop` | crop \| extcrop \| resize \| full \| extfull. |
| `--pose_style` | 0 | 0..45. Learned speaker-style bucket; head-motion only. |
| `--enhancer` | `None` | None \| gfpgan \| RestoreFormer. Face super-resolution. |
| `--still` | off | Reduce head motion (paper-style). |
| `--batch_size` | 2 | Facerender batch; memory-bound. |
| `--impl_dir` | `./impl` | Path to the cloned SadTalker repo. |
| `--conda_env` | `sadtalker` | Env with SadTalker's torch stack. |
| `--n_take` | unbounded | Cap pair list for debug runs. |

## 7. Troubleshooting

- **`conda: command not found` inside `subprocess.run`** — make sure conda is
  on `PATH` in the shell that launched the runner. If you use conda init
  snippets that only activate interactively, `source ~/.bashrc` first.
- **`ModuleNotFoundError: librosa`** inside the subprocess — you likely
  installed `impl/requirements.txt` instead of our `requirements.txt`. The
  torch 2.1 stack needs our exact pins; upstream's librosa 0.10 removes
  APIs SadTalker's code still calls.
- **`Can't get the coeffs of the input`** printed by inference.py — face
  detection failed on the source frame. Common for HDTF clips whose frame
  0 is a transition. Our adapter picks a random frame, which helps, but
  you can rerun with a different `--seed` to resample.
