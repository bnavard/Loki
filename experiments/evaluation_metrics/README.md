# Evaluation Metrics

Quantitative metrics for any run dir produced by the SOTA-comparison or
marionette-eval runners. The CLI reads `<run_dir>/config_resolved.json` to
recover `dataset` + `protocol`, then routes the right metric set:

| Protocol                          | Metrics                                       |
|-----------------------------------|-----------------------------------------------|
| `same_identity_reconstruction`    | `head_rot_dist`, `expression_l1`              |
| `cross_identity`                  | `head_rot_dist`, `expression_l1`, `id_cosine` |

Per-sample results land at `<output_dir>/metrics.jsonl` (one line per
sample); aggregates at `<output_dir>/metrics_summary.json`.

## Conventions

- **Inputs**: `<run_dir>/samples/<sample_id>/panel.mp4` for the
  `id_cosine` ArcFace prior; FLAME tracking outputs at
  `data/flame_tracking/preds/<bucket>/<dataset>/<protocol>/<sample_id>/fit.npz`
  for `head_rot_dist` and `expression_l1`. The pred-fit tree is produced
  by running [`generate_exp_map/`](../../generate_exp_map/) on each
  baseline's `panel.mp4` outputs.
- **Ground truth FLAME fits**: per-clip `fit.npz` under
  `data/benchmark/hdtf/flame_tracking/flowface/<clip_id>/`.
- **fps + resolution**: predictions are 25 fps at 512×512 across every
  tool in this repo; the ArcFace path resamples HDTF source clips to
  match (nearest-neighbor — no temporal interpolation).
- **Per-frame handling**:
  - `head_rot_dist` and `expression_l1` read FLAME parameters from
    `fit.npz`. A sample contributes when both pred and target fits
    exist with ≥ 2 frames; per-sample `*_track_rate = T_used / n_frames`
    records how much of the requested window was usable.
  - `id_cosine` (cross-identity only) requires per-frame ArcFace
    detection on the pred. Per-sample `id_detect_rate` records failure
    density.
- **Run-level aggregation**: every headline metric uses a **weighted**
  mean with each sample's track / detect rate as the weight, so a sample
  whose number was computed on 5/16 frames contributes proportionally
  less than one computed on 16/16.

## Setup

Runs in the `marionette` conda env — no separate environment to build.
Head-rot and expression metrics share pytorch3d with model training;
`id_cosine` uses `onnxruntime-gpu` (pinned in [`requirements.txt`](../../requirements.txt))
against torch's bundled cuDNN 9. The launcher scripts source
`_activate.sh` which prepends torch's cuDNN dir to `LD_LIBRARY_PATH` so
onnxruntime-gpu's CUDAExecutionProvider can dlopen it.

## Usage

### Single run

```bash
# Same-identity:
PYTHONPATH=. python experiments/evaluation_metrics/compute_metrics.py \
    --run-dir outputs/sota_comparison/sadtalker/hdtf/same_identity_reconstruction/run_<ts>/

# Cross-identity:
PYTHONPATH=. python experiments/evaluation_metrics/compute_metrics.py \
    --run-dir outputs/sota_comparison/xportrait/hdtf/cross_identity/run_<ts>/

# Top up only one group on a partially-evaluated dir:
PYTHONPATH=. python experiments/evaluation_metrics/compute_metrics.py \
    --run-dir outputs/sota_comparison/anitalker/hdtf/same_identity_reconstruction/run_<ts>/ \
    --metrics head_rot
```

### Unified sweep — Marionette + every SOTA baseline (multi-GPU)

`run_eval_metrics.sh` walks both
`outputs/marionette_eval/<dataset>/<protocol>/run_*/` and
`outputs/sota_comparison/<baseline>/<dataset>/<protocol>/run_*/` into a
single round-robin queue across the available GPUs (default 8). Every
artifact lands in a centralized tree under
`outputs/test_metric/metrics/<bucket>/<dataset>/<protocol>/`, where
`<bucket>` is `marionette` or one of the SOTA baseline names — a single
glob `outputs/test_metric/metrics/*/<dataset>/<protocol>/metrics_summary.json`
picks up every model uniformly for comparison.

```bash
# All 8 GPUs (default), top-up missing groups:
bash experiments/evaluation_metrics/run_eval_metrics.sh

# Recompute one group across every dir:
METRICS=head_rot bash experiments/evaluation_metrics/run_eval_metrics.sh

# Force recomputation, ignore caches:
FRESH=1 bash experiments/evaluation_metrics/run_eval_metrics.sh

# Override GPU count or pin to specific GPUs:
NUM_GPUS=4    bash experiments/evaluation_metrics/run_eval_metrics.sh
GPUS="0 2 4 6" bash experiments/evaluation_metrics/run_eval_metrics.sh

# Background + log:
bash experiments/evaluation_metrics/run_eval_metrics.sh \
    > outputs/test_metric/metrics/_batch.log 2>&1 &
tail -f outputs/test_metric/metrics/_batch.log
```

Idempotent: any run dir whose central `metrics_summary.json` already
contains the headline value for a group is skipped on re-run for that
group. `METRICS=all` forces a full overwrite; `FRESH=1` wipes the central
summary + `metrics.jsonl` first so `auto` recomputes everything.

## Metrics

### `head_rot_dist` — geodesic angular distance over FLAME head pose

Compose the FLAME `rot · neck_rot` rotation per frame, anchor every
trajectory to its own frame 0 (so per-clip camera-fit constants drop
out), and measure the geodesic angular distance between pred's and
target's frame-0-anchored deltas via the standard
`θ = 2·arccos(|q_a · q_b|)` quaternion formula. Reported in degrees,
mean over the available frames. Lower is better. The target side is
the GT clip's fit (same-identity) or the driver clip's fit
(cross-identity).

### `expression_l1` — pose-disentangled FLAME deformation map L1

For each frame, render the target's FLAME fit verbatim, then render a
substituted fit where pred's `(expr, eye_rot, jaw_rot)` are inserted
into the target's `(rot, tra, neck_rot, shape, camera)`. Both renders
land on the same image-space pixels by construction, so the per-pixel
difference between the two deformation maps is purely an expression
difference. Per-pixel L1 averaged across the 3 deform channels, then
averaged over on-mesh pixels (mask-aware — background pixels would
otherwise dilute the score). Lower is better.

### `id_cosine` — ArcFace identity preservation (cross-identity only)

InsightFace `buffalo_l` (RetinaFace + ArcFace R100). The identity prior
is the L2-normalized mean of ArcFace embeddings over **all frames of
the ref clip**; per-frame cosine of the generated panel against the
prior is averaged over time. Bounded in `[-1, 1]`; higher is better.
Per-clip priors are cached so a ref clip used by multiple cross-id
pairs is only embedded once. Run-level mean is weighted by
`id_detect_rate`.

## Sanity check

`sanity_check/` ships visualization tools that *don't* compute the
aggregate metrics — they render per-frame overlays so you can verify by
eye that the metric inputs are sensible (FLAME fit alignment, expression
substitution, head-pose deltas).

```bash
PYTHONPATH=. python experiments/evaluation_metrics/sanity_check/visualize_expression.py \
    --target-fit data/benchmark/hdtf/flame_tracking/flowface/<clip_id>/fit.npz \
    --pred-fit   data/flame_tracking/preds/<bucket>/hdtf/<protocol>/<sample_id>/fit.npz \
    --target-video data/benchmark/hdtf/clips/<clip_id>.mp4 \
    --pred-video   <run_dir>/samples/<sample_id>/panel.mp4 \
    --out-mp4      outputs/test_metric/expr_sanity_swap/<bucket>_<sample_id>.mp4
```

## Layout

```
experiments/evaluation_metrics/
├── README.md
├── _activate.sh                          # sourced by launcher scripts; activates `marionette`
├── compute_metrics.py                    # CLI — auto-routes by protocol
├── run_eval_metrics.sh                   # unified multi-GPU sweep (Marionette + SOTA)
├── metrics/
│   ├── __init__.py
│   ├── io.py                             # video decode + run-tree walk
│   ├── evaluator.py                      # protocol-aware routing + weighted aggregation
│   └── src/
│       ├── head_rot.py                   # FLAME rot·neck_rot geodesic distance
│       ├── expression.py                 # pose-disentangled deformation-map L1
│       └── id_sim.py                     # InsightFace ArcFace cosine (cross-id only)
├── deformation_map_diff/                 # helper scripts for the expression-metric pipeline
└── sanity_check/                         # visual debugging — NOT used by compute_metrics.py
```
