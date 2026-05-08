# Loki Evaluation

Loki's eval against a checkpoint, **wired into the same identity-pair
infrastructure every SOTA wrapper uses**. Reads the curated HDTF manifest
at [`experiments/sota_comparison/manifests/hdtf.json`](../sota_comparison/manifests/hdtf.json)
and emits panels under
`outputs/loki_eval/hdtf/<protocol>/run_<ts>/samples/<sample_id>/`,
where `<sample_id>` is the same UID-based name (`id_0457` /
`id_0457_id_0009`) every SOTA baseline produces. A single glob across
`outputs/**/samples/<sample_id>/panel.mp4` therefore compares Loki
1-to-1 against every baseline on the same physical identity (or pair).

## Quick start

```bash
conda activate loki

# Cross-identity (recommended first)
PYTHONPATH=. python experiments/loki_eval/run_inference.py \
    --protocol cross_identity \
    --n_samples 212 \
    --clip_duration_s 3.0 \
    --seed 42 \
    --checkpoint outputs/loki_baseline/run_<ts>/checkpoints/<ckpt>.ckpt

# Same-identity reconstruction
PYTHONPATH=. python experiments/loki_eval/run_inference.py \
    --protocol same_identity_reconstruction \
    --n_samples 212 \
    --clip_duration_s 3.0 \
    --seed 42 \
    --checkpoint outputs/loki_baseline/run_<ts>/checkpoints/<ckpt>.ckpt
```

`--checkpoint` can be set in [`configs/eval.yaml`](configs/eval.yaml) to
avoid the CLI override. Other knobs (`--cfg_scale`, `--n_ddim_steps`,
`--n_frames`) override the YAML on the fly.

**Prerequisite — curated manifest:** build the HDTF manifest once with

```bash
PYTHONPATH=. python experiments/sota_comparison/dataset/build_manifest.py --dataset hdtf
```

**Prerequisite — HDTF FLAME tracking.** The Loki inference path
needs per-clip `fit.npz` under
`data/benchmark/hdtf/flame_tracking/flowface/<clip_id>/`. Generate it
once into a dedicated tree under the HDTF root:

```bash
# 1. Stage symlinks to the manifest clips so the parallel runner only
#    processes those (not all 16k+ HDTF mp4s):
conda activate loki
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
baselines and Loki.

## Why this is "like sota_comparison" specifically

- **Pair-list source identical.** Both Loki and every SOTA baseline
  call `load_by_dataset("hdtf")` → curated manifest → `build_samples`.
  Same UID pool, same derangement / sampling logic.
- **Sample-id format identical.** `id_0457` / `id_0457_id_0009`.
- **Output layout identical.** `samples/<sample_id>/panel.mp4` plus
  `config_resolved.yaml` and `config_resolved.json` snapshots at run root, plus a
  `failed.json` when any sample errors.
- **Seeding policy identical.** `ref_frame_idx` is drawn from a single
  `np.random.default_rng(seed)` — `(protocol, seed, sample_id)` selects the
  same ref frame across every baseline driven from the same manifest.

## Where Loki differs from a SOTA wrapper

- **In-process, no `conda run`.** Loki is local; the model loads once
  in the runner and `Evaluator.run_one(...)` is called per sample. No
  subprocess hop, no per-sample model reload.
- **`panel.mp4` is shorter than SOTA's.** Loki generates
  `cfg.inference.n_frames` frames per panel (16 by default = 0.64 s at
  25 fps). SOTA panels are typically 5 s / 125 frames. The `<sample_id>`
  folder name aligns; the mp4 durations don't. `--clip_duration_s` here
  only filters the eligible-clip pool inside `build_samples`; it does NOT
  change Loki's actual generation length.
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
   face-cropped frames for the panel's "Driver Video" row.
5. Run the active cond_stage module (instantiated via
   `instantiate_from_config(cfg.model.params.cond_stage_config)`, so any
   condition_ablation arm drops in without code changes here) on
   `{driver_verts, driver_deform}` → `spatial_cond`.
6. VAE-encode the ref → `ref_z (1, 4, h, w)` for `RefFeatureExtractor`.
7. `model.sample_video(...)` — DDIM with classifier-free guidance.
8. VAE-decode + write the SOTA-shaped on-disk artifacts (silent mp4s):
   - `samples/<sample_id>/panel.mp4` — generated 512×512 video.
   - `scratch/<sample_id>/source.png` — ref frame.
   - `scratch/<sample_id>/driver.mp4` — driver's face-cropped frames.

## Output layout

```
outputs/loki_eval/hdtf/<protocol>/run_<timestamp>/
├── config_resolved.yaml          # snapshot of the resolved YAML
├── config_resolved.json          # CLI args + git rev + checkpoint path
│                                  # (read by the metrics runner)
├── failed.json                   # (only if any sample errored)
├── samples/<sample_id>/
│   └── panel.mp4                 # 512×512 generation, silent
└── scratch/<sample_id>/
    ├── source.png                # ref frame
    └── driver.mp4                # 512×512 driver row, silent
```

Compare directly against
`outputs/sota_comparison/<baseline>/hdtf/<protocol>/run_<ts>/samples/<sample_id>/panel.mp4`
with a single glob.

## Structure

```
experiments/loki_eval/
├── README.md
├── adapter.py            # Evaluator class + run_one(sample: EvalSample, ...)
├── run_inference.py      # CLI orchestrator (single runner, --protocol selects)
└── configs/
    └── eval.yaml         # base + inference knobs + output_dir + checkpoint
```
