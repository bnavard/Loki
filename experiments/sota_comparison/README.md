# SOTA Comparison Suite

Uniform wrappers around state-of-the-art talking-head models so their
outputs sit next to Marionette's under a common layout for apples-to-apples
comparison. One folder per baseline; one command per `(baseline, dataset,
protocol)` triple.

## Design

- **`dataset/`** — dataset-agnostic manifest + protocol layer.
  - `base.py`: `BenchmarkClip` dataclass, `BenchmarkVideoDataset` ABC with
    a manifest cache (ffprobe once, JSON to `data/derived/<name>_manifest.json`).
  - `hdtf.py`, `celebvhq.py`, `voxceleb2.py`, `talkvid.py` *(added as each
    baseline needs them)*: walk dataset-specific on-disk layouts, emit
    `(clip_id, identity_id, video_path)` triples.
  - `pairing.py`: builds deterministic `EvalSample` lists under one of two
    protocols — `same_identity_reconstruction` or `cross_identity`. All
    randomness funnels through one seeded `np.random.default_rng(seed)` so
    a given `(protocol, seed, manifest)` tuple reproduces the same list.
- **`<baseline>/`** — one folder per upstream model. Each holds:
  - `README.md`: commit pin, env setup, paper protocol notes, CLI knobs.
  - `env.yml` / `requirements.txt`: the conda env that actually runs on
    modern CUDA hardware (may diverge from upstream's README if their pins
    are stale — documented in each baseline's README).
  - `adapter.py`: converts `EvalSample` + protocol args into whatever the
    baseline's inference CLI wants on disk (source image, WAV, manifest,
    …) and shells out.
  - `run_inference.py`: uniform CLI — `--dataset --protocol --n_samples
    --clip_duration_s --seed` plus baseline-specific knobs.
  - `impl/`: gitignored clone of the upstream repo. Commit pin lives in
    the baseline's README.

## Output layout

Every baseline writes to:

```
outputs/sota_comparison/<baseline>/<dataset>/<protocol>/run_<timestamp>/
├── config_resolved.json          # full CLI args + git rev of this repo
├── scratch/<sample_id>/          # per-sample working files
├── failed.json                   # (only if any sample errored)
└── samples/<sample_id>/
    └── panel.mp4
```

Mirrors `experiments/marionette_eval/`'s layout so downstream metric sweeps
can glob `outputs/**/samples/**/panel.mp4` uniformly across baselines and
Marionette.

## Protocols

| Protocol | ref_clip | driver_clip | Audio-driven baseline reads it as | Motion-transfer baseline reads it as |
|---|---|---|---|---|
| `same_identity_reconstruction` | `clip A` | `clip A` | Frame of A + A's own audio (self-reconstruction) | Ref window of A + A's own motion |
| `cross_identity` | `clip A` (identity X) | `clip B` (identity Y ≠ X) | Frame of A + B's audio (voice transfer) | A's identity performing B's motion |

Both are exposed on every baseline's runner. For SadTalker: the paper's
protocol is `same_identity_reconstruction`; `cross_identity` is the
apples-to-apples complement for judging expression / lip transfer under
identity mismatch.

## Baselines

| Baseline | Paper | Trained on | Paper eval | Input modality | Res / FPS | Wrapper | Commit pin |
|---|---|---|---|---|---|---|---|
| [SadTalker](sadtalker/README.md) | [CVPR 2023](https://arxiv.org/abs/2211.12194) | VoxCeleb | HDTF, first 8 s of 346 videos, same-identity | source image + driven audio | 512×512 / 25 fps | [sadtalker/](sadtalker/) | see `sadtalker/COMMIT_PIN.txt` after first clone |

Add rows as new baselines are wrapped. The **Paper eval** column is the
source of truth for how we replicate each baseline's published numbers —
our wrapper's default CLI args aim to match that column modulo
dataset-mirror constraints (documented per baseline).

## Dataset notes

| Dataset | Root | Clip layout | Identity grouping | Caveats |
|---|---|---|---|---|
| HDTF | `data/benchmark/hdtf/clips/` | Flat `<speaker>_<session>_<start>_<end>.mp4` | First N tokens before `_<session>_<start>_<end>` | Pre-chunked into ~3.24-s segments; paper's "first 8s" can't be replicated clip-by-clip. Use `clip_duration_s=3.0` or stitch adjacent chunks (follow-up). `RD_Radio*` clips are skipped — their naming is inconsistent. |
| CelebV-HQ | `data/benchmark/celebvhq/` | Flat `<video_id>_<idx>.mp4` | Prefix before `_<idx>` | *Adapter pending — added when a baseline needs it.* |
| VoxCeleb2 | `data/benchmark/voxceleb2/clips/<speaker>/<video>/<utt>.mp4` | Hierarchical | Top-level folder (`id00012`, …) | Audio partly in `.7z` archives under `audio/aac/` — unpack before use. *Adapter pending.* |
| TalkVid (ours) | `data/talkvid/talkvid/` | Flat, our own preprocessing | Prefix before first `_` | *Adapter pending — belongs here so any baseline can be evaluated on our data with zero wrapper change.* |

## Running a baseline

See the baseline's own README for env setup. The orchestrator launches from
the `marionette` env — the baseline's env only hosts its inference
subprocess, hopped into via `conda run -n <env>`.

```bash
conda activate marionette
PYTHONPATH=. python experiments/sota_comparison/<baseline>/run_inference.py \
    --dataset         hdtf \
    --protocol        same_identity_reconstruction \
    --n_samples       346 \
    --clip_duration_s 3.0 \
    --seed            42
```
