# EchoMimic — SOTA Comparison Wrapper

Runs EchoMimicV1 (Ant Group / Alipay, [paper](https://arxiv.org/abs/2407.08136),
[code](https://github.com/antgroup/echomimic),
[HuggingFace](https://huggingface.co/BadToBest/EchoMimic)) against our
benchmark datasets under a uniform CLI so its outputs sit next to
Loki's, SadTalker's, AniTalker's, HunyuanPortrait's, and X-Portrait's
for apples-to-apples comparison.

## At a glance

| | |
|---|---|
| Paper | [AAAI 2025](https://arxiv.org/abs/2407.08136) |
| Input | Source image + driven audio (WAV) — landmark-conditioned diffusion |
| Output | 512×512 face-cropped mp4 with audio muxed in |
| Env | `echomimic` — Python 3.10 + torch 2.1.0+cu121 |
| Pinned commit | *fill in on first clone — see "1. Clone the baseline"* |

## 0. One-shot setup (recommended)

Everything below (env, pip deps, clone, ~10 GB ckpt subset from HuggingFace)
in one idempotent script:

```bash
bash experiments/sota_comparison/echomimic/setup_env.sh
```

Per-file sentinels skip artefacts already on disk, so a re-run after a
flaky HuggingFace connection just resumes the missing files.

## 1. Clone the baseline

Upstream is not committed into this repo — it's gitignored under `impl/`.

```bash
cd experiments/sota_comparison/echomimic
git clone https://github.com/antgroup/echomimic.git impl
cd impl && git rev-parse HEAD > ../COMMIT_PIN.txt
```

Record the commit hash in the "At a glance" table on first setup. If the
github clone is slow on your network, `setup_env.sh` also accepts a
zip-extracted mirror dropped at `impl/`.

## 2. Environment

Upstream is permissive on torch (`>=2.0.1, <=2.2.2`) and tested on Python
3.8 / 3.10 / 3.11. We pick `python=3.10 + torch 2.1.0+cu121` — the same
stack used by SadTalker and HunyuanPortrait — so the audio-driven baselines
share an ABI and we land squarely inside upstream's tested matrix.

```bash
conda env create -f experiments/sota_comparison/echomimic/env.yml
conda activate echomimic
pip install -r experiments/sota_comparison/echomimic/requirements.txt
```

Deviations from upstream's `impl/requirements.txt` are minimal — `torch` /
`torchvision` / `torchaudio` are pinned to the cu121 wheels in `env.yml`,
and we add `huggingface_hub[cli]` so `setup_env.sh` can use the new `hf`
command for the weight download.

## 3. Model weights

The `BadToBest/EchoMimic` HuggingFace repo is **~34 GB** total — it ships
multiple variants:
- `denoising_unet{,_acc,_pose,_pose_acc}.pth`  (3.4 GB each)
- `motion_module{,_acc,_pose,_pose_acc}.pth`   (1.82 GB each)
- `reference_unet{,_pose}.pth`                 (3.26 GB each)
- `face_locator{,_pose}.pth`                   (4.35 MB each)
- `audio_processor/`, `sd-vae-ft-mse/`, `sd-image-variations-diffusers/`

Our runner uses the **audio-only, non-accelerated** path (the variant
referenced by `configs/prompts/animation.yaml`). `setup_env.sh` downloads
only that subset (~10 GB) via `hf download --include`:

```bash
hf download BadToBest/EchoMimic --repo-type model \
    --local-dir impl/pretrained_weights \
    --include \
        "denoising_unet.pth" \
        "reference_unet.pth" \
        "motion_module.pth" \
        "face_locator.pth" \
        "audio_processor/*" \
        "sd-vae-ft-mse/*" \
        "sd-image-variations-diffusers/*"
```

Resulting layout:

```
impl/pretrained_weights/
├── denoising_unet.pth                   (3.4 GB)
├── reference_unet.pth                   (3.26 GB)
├── motion_module.pth                    (1.82 GB)
├── face_locator.pth                     (4.35 MB)
├── audio_processor/
│   └── whisper_tiny.pt
├── sd-vae-ft-mse/
│   └── ...
└── sd-image-variations-diffusers/
    └── ...
```

If you later want to ablate the pose-driven or accelerated variants, add
`denoising_unet_pose.pth`, `motion_module_pose.pth`, `reference_unet_pose.pth`,
`face_locator_pose.pth` (or the `_acc` shards) to the `--include` list.

## 4. Run

**Prerequisite — curated manifest.** The runner reads
`experiments/sota_comparison/manifests/<dataset>.json`. Build it once via
`dataset/build_manifest.py` (see the top-level
[sota_comparison README](../README.md)).

The runner lives outside the `echomimic` env — it orchestrates, builds the
pair list, and hops into the env per sample via `conda run`. Launch from
the `loki` env:

```bash
conda activate loki

# HDTF cross-identity (A's face + B's audio)
PYTHONPATH=. python experiments/sota_comparison/echomimic/run_inference.py \
    --dataset hdtf \
    --protocol cross_identity \
    --n_samples 200 \
    --clip_duration_s 3.0 \
    --seed 42

# Same-identity reconstruction
PYTHONPATH=. python experiments/sota_comparison/echomimic/run_inference.py \
    --dataset hdtf \
    --protocol same_identity_reconstruction \
    --n_samples 346 \
    --clip_duration_s 3.0 \
    --seed 42
```

Both protocols are in the runner's docstring.

### Protocol notes

- **Audio-driven, landmark-conditioned.** Same input shape as SadTalker /
  AniTalker (image + WAV). Output mp4 has the driver's audio muxed in via
  upstream's `_withaudio.mp4` post-step.
- **Sidecar WAV preference.** When a dataset ships sidecar `.wav` files,
  the adapter reads `driver_clip.audio_path`. HDTF and similar
  muxed-audio datasets → ffmpeg extracts from the muxed video stream.
  Same code path as SadTalker.
- **Patched per-sample config.** Upstream's `infer_audio2vid.py` is
  YAML-driven (a `test_cases` mapping in `configs/prompts/animation.yaml`).
  Our adapter writes a per-sample patched config to scratch with
  `test_cases = {<source.png>: [<audio.wav>]}` (absolute paths) and points
  `--config` at it; everything else (weight paths, `inference_config`)
  stays relative and resolves against the subprocess cwd = `impl_dir`.
- **Frame count via `-L`.** `-L = round(clip_duration_s × fps)`. At
  25 fps × 3 s ≈ 75 frames on HDTF. Upstream's demo default `-L 1200` is
  way too long for our use; we override per-sample.
- **Random reference frame.** Same seeded RNG schedule as every other
  runner. `(protocol, seed, sample_id)` selects the same ref frame across
  every audio-driven baseline (SadTalker, AniTalker, EchoMimic).

## 5. Output layout

```
outputs/sota_comparison/echomimic/<dataset>/<protocol>/run_<timestamp>/
├── config_resolved.json          # full CLI args + git rev
├── scratch/<sample_id>/          # per-sample working files
│   ├── source.png                #   ref frame
│   ├── audio.wav                 #   driver audio, 16 kHz mono
│   └── animation.yaml            #   patched test_cases config
├── failed.json                   # (only if any sample errored)
└── samples/<sample_id>/
    └── panel.mp4                 # EchoMimic's *_withaudio.mp4
```

`<sample_id>` is UID-based:
- `same_identity_reconstruction` → `id_0457`
- `cross_identity` → `id_0457_id_0009` (ref uid, driver uid)

Aligned with every other SOTA baseline + Loki → a single glob like
`outputs/sota_comparison/*/hdtf/cross_identity/run_<ts>/samples/id_0457_id_0009/panel.mp4`
gives every model's output for the same identity pair.

## 6. Knobs exposed on the CLI

| Flag | Default | Notes |
|---|---|---|
| `--width / --height` | 512 / 512 | Output spatial size. Defaults match every other baseline. |
| `--fps` | 25 | Output mp4 fps; matches HDTF native. |
| `--cfg` | 2.5 | Classifier-free guidance scale. |
| `--steps` | 30 | Diffusion denoising steps. |
| `--echomimic_seed` | 420 | EchoMimic's internal seed (independent of `--seed` which drives the runner's pair-list / ref-frame). |
| `--sample_rate` | 16000 | WAV sample rate fed to whisper-tiny audio encoder. |
| `--context_frames` | 12 | Per-batch frames in the temporal pipeline. |
| `--context_overlap` | 3 | Overlap between consecutive context windows. |
| `--facemusk_dilation_ratio` | 0.1 | Face-mask padding. |
| `--facecrop_dilation_ratio` | 0.5 | Face-crop padding. |
| `--impl_dir` | `./impl` | Path to the cloned EchoMimic repo. |
| `--conda_env` | `echomimic` | Env with EchoMimic's torch 2.1+cu121 stack. |
| `--upstream_config` | `configs/prompts/animation.yaml` | Path relative to `impl_dir` of the upstream config we patch per sample. |
| `--n_take` | unbounded | Cap pair list for debug runs. |

## 7. Troubleshooting

- **`mediapipe` install fails** — usually a Python version mismatch. Our
  `env.yml` pins `python=3.10`; verify `python --version` reports that
  inside the `echomimic` env.
- **`facenet_pytorch` import error on torch 2.1** — `facenet_pytorch==2.5.0`
  was built against torch <= 2.x, should work; if it complains about
  `torch.load` weights_only changes, force-reinstall: `pip install
  --force-reinstall facenet_pytorch==2.5.0`.
- **HuggingFace download stalls / partial** — the audio-only subset is
  ~10 GB and can take many minutes. `hf download` resumes automatically
  on re-run (the per-file sentinels in `setup_env.sh` mean an `hf
  download` call only re-fetches what's missing).
- **`OSError: ... no kernel image available`** — torch picked up the wrong
  CUDA wheel. Verify `python -c "import torch; print(torch.version.cuda)"`
  prints `12.1`. If not, force-reinstall the cu121 stack from `env.yml`.
- **Output mp4 is silent** — the adapter looks for `*_withaudio.mp4` first
  and falls back to the no-audio variant. A silent output means upstream's
  moviepy mux step failed (often a moviepy / imageio-ffmpeg version
  mismatch). Check that `moviepy==1.0.3` is the resolved version inside
  the env.
