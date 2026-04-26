# Marionette Evaluation

Marionette's eval against a checkpoint, **wired into the same identity-pair
infrastructure every SOTA wrapper uses**. Reads the curated TalkVid manifest
at [`experiments/sota_comparison/manifests/talkvid.json`](../sota_comparison/manifests/talkvid.json)
and emits panels under `outputs/marionette_eval/<dataset>/<protocol>/run_<ts>/samples/<sample_id>/`,
where `<sample_id>` is the same UID-based name (`id_0457` /
`id_0457_id_0009`) every SOTA baseline produces. A single glob across
`outputs/**/samples/<sample_id>/panel.mp4` therefore compares Marionette
1-to-1 against every baseline on the same physical identity (or pair).

## Quick start

```bash
conda activate marionette

# Cross-identity (recommended first)
PYTHONPATH=. python experiments/marionette_eval/run_inference.py \
    --dataset talkvid \
    --protocol cross_identity \
    --n_samples 125 \
    --clip_duration_s 5.0 \
    --seed 42 \
    --checkpoint outputs/marionette_baseline/run_<ts>/checkpoints/<ckpt>.ckpt

# Same-identity reconstruction
PYTHONPATH=. python experiments/marionette_eval/run_inference.py \
    --dataset talkvid \
    --protocol same_identity_reconstruction \
    --n_samples 125 \
    --clip_duration_s 5.0 \
    --seed 42 \
    --checkpoint outputs/marionette_baseline/run_<ts>/checkpoints/<ckpt>.ckpt

# HDTF — cross-identity (212 identities; clips are ~3.24 s, so use a 3.0 s filter)
PYTHONPATH=. python experiments/marionette_eval/run_inference.py \
    --dataset hdtf \
    --protocol cross_identity \
    --n_samples 212 \
    --clip_duration_s 3.0 \
    --seed 42 \
    --checkpoint outputs/marionette_baseline/run_<ts>/checkpoints/<ckpt>.ckpt
```

`--checkpoint` can be set in [`configs/eval.yaml`](configs/eval.yaml) to
avoid the CLI override. Other knobs (`--cfg_scale`, `--n_ddim_steps`,
`--n_frames`) override the YAML on the fly.

**Prerequisite:** the curated manifest must be built once with

```bash
PYTHONPATH=. python experiments/sota_comparison/dataset/build_manifest.py --dataset talkvid
PYTHONPATH=. python experiments/sota_comparison/dataset/build_manifest.py --dataset hdtf  # if running HDTF
```

**HDTF prereq — generate `fit.npz` for the 212 manifest clips.** TalkVid
already has FLAME tracking under `data/flame_tracking/flowface/`. HDTF
doesn't ship with it; generate it once into a dedicated tree under the
HDTF root (kept separate from `data/flame_tracking/` so the two pools
don't mix):

```bash
# 1. Stage symlinks to the 212 manifest clips so the parallel runner
#    only processes those (not all 16k+ HDTF mp4s):
conda activate marionette
PYTHONPATH=. python -c "
import json
from pathlib import Path
m = json.load(open('experiments/sota_comparison/manifests/hdtf.json'))
stage = Path('data/benchmark/hdtf/_eval_inputs')
stage.mkdir(parents=True, exist_ok=True)
for c in m['clips']:
    dst = stage / f\"{c['clip_id']}.mp4\"
    if dst.is_symlink() or dst.exists(): dst.unlink()
    dst.symlink_to(Path(c['video_path']).resolve())
print(f'staged {len(m[\"clips\"])} clips')
"

# 2. Run the FLAME-tracking pipeline against the staged dir, writing all
#    intermediate + final outputs under data/benchmark/hdtf/flame_tracking/:
conda activate expmapgen
PIXEL3DMM_PREPROCESSED_DATA=data/benchmark/hdtf/flame_tracking/preprocessing \
PIXEL3DMM_TRACKING_OUTPUT=data/benchmark/hdtf/flame_tracking/tracking \
FLAME_LOG_DIR=data/benchmark/hdtf/flame_tracking/logs/artifacts \
FLAME_COMPLETED_LOG=data/benchmark/hdtf/flame_tracking/logs/completed_videos.txt \
FLAME_FAILED_LOG=data/benchmark/hdtf/flame_tracking/logs/failed_videos.txt \
bash generate_exp_map/scripts/run_multi_gpu.sh \
    data/benchmark/hdtf/_eval_inputs \
    data/benchmark/hdtf/flame_tracking/flowface \
    8 2
```

HDTF clips are 81 frames (~3.24 s at 25 fps) — pass `--clip_duration_s 3.0`
on the eval CLI so `build_samples`'s length filter doesn't drop everything.

## What `--protocol` means here

| Protocol | ref_clip | driver_clip | Output sample_id |
|---|---|---|---|
| `same_identity_reconstruction` | identity A's clip | identity A's clip (same) | `id_<uid>` |
| `cross_identity` | identity A | identity B (B ≠ A; one identity per ref + one per driver, derangement over the manifest) | `id_<ref_uid>_id_<drv_uid>` |

Both protocols are produced by `experiments.sota_comparison.dataset.pairing.build_samples`
under one seeded RNG, identically to every SOTA wrapper. A given
`(protocol, seed, manifest)` reproduces the same pair list across SOTA
baselines and Marionette.

## Why this is "like sota_comparison" specifically

- **Pair-list source identical.** Both Marionette and every SOTA baseline
  call `load_by_dataset("talkvid")` → curated manifest → `build_samples`.
  Same UID pool, same derangement / sampling logic.
- **Sample-id format identical.** `id_0457` / `id_0457_id_0009`.
- **Output layout identical.** `samples/<sample_id>/panel.{png,mp4}` plus
  `config_resolved.yaml` and `config_resolved.json` snapshots at run root, plus a
  `failed.json` when any sample errors.
- **Seeding policy identical.** `ref_frame_idx` is drawn from a single
  `np.random.default_rng(seed)` — `(protocol, seed, sample_id)` selects the
  same ref frame across every baseline driven from the same manifest.

## Where Marionette differs from a SOTA wrapper

- **In-process, no `conda run`.** Marionette is local; the model loads once
  in the runner and `Evaluator.run_one(...)` is called per sample. No
  subprocess hop, no per-sample model reload.
- **Datasets supported: `talkvid` and `hdtf`.** Both need `fit.npz` per
  clip. The per-dataset FLAME tracking root is read from
  `cfg.flame_roots[<dataset>]` — TalkVid points at
  `data/flame_tracking/flowface/`, HDTF at
  `data/benchmark/hdtf/flame_tracking/flowface/`. If `fit.npz` is missing
  for a dataset, generate via `generate_exp_map/` first; see "HDTF prereq"
  below.
- **`panel.mp4` is shorter than SOTA's.** Marionette generates
  `cfg.inference.n_frames` frames per panel (16 by default = 0.64 s at
  25 fps). SOTA panels are typically 5 s / 125 frames. The `<sample_id>`
  folder name aligns; the mp4 durations don't. `--clip_duration_s` here
  only filters the eligible-clip pool inside `build_samples`; it does NOT
  change Marionette's actual generation length.
- **Driver-window start fixed at frame 0.** Matches every SOTA wrapper's
  "first N frames of the trimmed driver" convention.

## Inference path (per sample)

`Evaluator.run_one(sample: EvalSample, ref_frame_idx, output_dir)`:

1. Load `ref_fit` and `driver_fit` from
   `<flame_root>/<clip_id>/fit.npz` (`flame_root` from
   `cfg.val_dataset.params.flame_root`).
2. `prepare_reference(ref_fit, ref_frame_idx, …)` → face-cropped 512×512
   ref image in `[-1, 1]` + crop_box.
3. `retarget_driver_verts(ref_fit, driver_fit, crop_box, n_frames, …,
   driver_start=0)` → `(T, V, 3)` NDC verts and `(T, V, 3)` expression
   offsets.
4. `prepare_driver_frames(driver_fit, …, driver_start=0)` → driver's own
   face-cropped frames, used for the panel's "Driver Video" row AND as
   the natural-video conditioning signal that the no_flame / no_deform
   ablation arms read (the baseline cond_stage ignores it).
5. Run the active cond_stage module (instantiated via
   `instantiate_from_config(cfg.model.params.cond_stage_config)`, so any
   condition_ablation arm drops in without code changes here) on
   `{driver_verts, driver_deform, driver_video}` → `spatial_cond`.
6. VAE-encode the ref → `ref_z (1, 4, h, w)` for `RefFeatureExtractor`.
7. If `model.audio_encoder is not None`, load `sample.driver_clip.audio_path`
   (TalkVid sidecar WAV), build per-frame ±`audio_context_frames` windows,
   encode. Otherwise skip — handles audio-off checkpoints from
   `condition_ablation/audio_off/`.
8. `model.sample_video(...)` — DDIM with classifier-free guidance.
9. VAE-decode + write 4-row panel:
   - Reference (static) | Driver Video | `<cond preview>` | Generated.
   - Row 3 slice + label come from the active cond_stage's `VIZ_SLICE` /
     `VIZ_LABEL` class attrs, so the row label is correct for every arm
     (`Driver Deform` for baseline, `Driver Video` / `Pos Enc` for
     ablations).

## Output layout

```
outputs/marionette_eval/<dataset>/<protocol>/run_<timestamp>/
├── config_resolved.yaml          # snapshot of the resolved YAML
├── config_resolved.json          # CLI args + git rev + checkpoint path
│                                  # (read by the metrics runner)
├── failed.json                   # (only if any sample errored)
└── samples/<sample_id>/
    ├── panel.png                 # 4-row labeled grid (vertical row labels)
    └── panel.mp4                 # same panel as video, with driver audio if available
```

Compare directly against
`outputs/sota_comparison/<baseline>/talkvid/<protocol>/run_<ts>/samples/<sample_id>/panel.mp4`
with a single glob.

## Structure

```
experiments/marionette_eval/
├── README.md
├── adapter.py            # Evaluator class + run_one(sample: EvalSample, ...)
├── run_inference.py      # CLI orchestrator (single runner, --protocol selects)
└── configs/
    └── eval.yaml         # base + inference knobs + output_dir + checkpoint
```
