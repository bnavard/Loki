# Evaluation Metrics

Quantitative metrics for any run dir produced by the SOTA-comparison or
loki-eval runners. The CLI reads `<run_dir>/config_resolved.json` to
recover `dataset` + `protocol`, then routes the right metric set:

| Protocol                          | Metrics                                                                                |
|-----------------------------------|----------------------------------------------------------------------------------------|
| `same_identity_reconstruction`    | `head_rot_dist`, `expression_l1`, `psnr_db`, `ssim`, `lpips`, `fvd_videomae` (HDTF only) |
| `cross_identity`                  | `head_rot_dist`, `expression_l1`, `id_cosine`                                          |

The pixel-aligned metrics (`psnr_db`, `ssim`, `lpips`) and `fvd_*` are
**HDTF-only** and **same-identity-only**: they require a pixel-aligned
GT video, which only exists when the predicted clip is the same person
as the reference (and we only report these on HDTF in the paper).
Cross-identity does not get these — there is no aligned GT to compare
against pixel-wise.

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
  - `psnr_db`, `ssim`, `lpips` (HDTF same-identity only) decode pred and
    GT through one shared `load_video` call at 25 fps / 512 px / 16
    frames; per-sample `pixel_track_rate = T_used / 16` is the shared
    weight for all three aggregates.
  - `fvd_videomae` (and optional `fvd_i3d`) is **distribution-level** —
    one scalar per (bucket, dataset, protocol). It does not appear in
    `metrics.jsonl` (no per-sample row); only in `metrics_summary.json`.
    See `compute_fvd.py` for the separate driver.
- **Run-level aggregation**: every headline metric uses a **weighted**
  mean with each sample's track / detect rate as the weight, so a sample
  whose number was computed on 5/16 frames contributes proportionally
  less than one computed on 16/16.

## Setup

Runs in the `loki` conda env — no separate environment to build.
Head-rot and expression metrics share pytorch3d with model training;
`id_cosine` uses `onnxruntime-gpu` (pinned in [`requirements.txt`](../../requirements.txt))
against torch's bundled cuDNN 9. The launcher scripts source
`_activate.sh` which prepends torch's cuDNN dir to `LD_LIBRARY_PATH` so
onnxruntime-gpu's CUDAExecutionProvider can dlopen it.

The pixel-aligned metrics (`psnr`, `ssim`, `lpips`) and FVD pull in
extra deps and a patched `cdfvd`. Run the one-time installer:

```bash
bash experiments/evaluation_metrics/setup_fvd.sh
```

It pip-installs `torchmetrics`, `lpips`, and `cdfvd`; copies our
HuggingFace-mirror patch over `cdfvd/third_party/VideoMAEv2/utils.py`
(the upstream URL is dead, see the patch's header for context); and
pre-stages the 1.9 GB VideoMAE-v2 SSv2-finetuned checkpoint into the
env's site-packages so the first FVD call doesn't block on a download.

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

### Unified sweep — Loki + every SOTA baseline (multi-GPU)

`run_eval_metrics.sh` walks both
`outputs/loki_eval/<dataset>/<protocol>/run_*/` and
`outputs/sota_comparison/<baseline>/<dataset>/<protocol>/run_*/` into a
single round-robin queue across the available GPUs (default 8). Every
artifact lands in a centralized tree under
`outputs/test_metric/metrics/<bucket>/<dataset>/<protocol>/`, where
`<bucket>` is `loki` or one of the SOTA baseline names — a single
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

### `psnr_db`, `ssim`, `lpips` — pixel-aligned image quality (HDTF same-id only)

Per-frame PSNR (`torchmetrics.functional.peak_signal_noise_ratio`,
`data_range=1.0`), per-frame SSIM (Wang et al. 2004 defaults: 11×11
Gaussian, σ=1.5, K1=0.01, K2=0.03), and per-frame LPIPS
(Zhang et al. 2018, `net='alex'`) between the predicted `panel.mp4` and
the ground-truth clip. Predicted and GT videos are decoded at 25 fps /
512 px / 16 frames through a single shared call so the three metrics
sit on identical pixel data. Run-level aggregate is a `pixel_track_rate`-
weighted mean. PSNR / SSIM higher-is-better; LPIPS lower-is-better.

### `fvd_videomae`, `fvd_i3d` — Fréchet Video Distance (HDTF same-id only)

Distribution-level visual quality. Each video is mapped to a single
feature vector by a frozen 3D classifier — VideoMAE-v2 (default,
SSv2-finetuned giant; `cdfvd`'s `videomae` backbone) or Kinetics-400
I3D (opt-in via `--i3d`, less sample-efficient). We fit two multivariate
Gaussians on the resulting feature sets — one over GT clips, one over
generated clips — and report the closed-form Fréchet distance:

```
FVD = ||μ_real − μ_fake||² + tr(Σ_real + Σ_fake − 2(Σ_real Σ_fake)^½)
```

Lower is better. **One scalar per (bucket, dataset, protocol)** — there
is no per-sample decomposition, because the Σs are properties of a set
of feature vectors. With 212 HDTF same-id clips, VideoMAE-v2 is the
recommended backbone (Luo et al. JEDi: more sample-efficient than I3D,
which is widely held to need ≥ 2 k clips for a stable estimate).

Run via the separate driver:

```bash
# All discovered HDTF same-id buckets, VideoMAE-v2 (default).
PYTHONPATH=. python experiments/evaluation_metrics/compute_fvd.py

# Add I3D too.
PYTHONPATH=. python experiments/evaluation_metrics/compute_fvd.py --i3d

# Single bucket.
PYTHONPATH=. python experiments/evaluation_metrics/compute_fvd.py \
    --run-dir outputs/loki_eval/hdtf/same_identity_reconstruction/run_<ts>/
```

Result is folded into the same
`outputs/test_metric/metrics/<bucket>/hdtf/same_identity_reconstruction/metrics_summary.json`
the per-sample evaluator writes, under `metrics.fvd_videomae` / `metrics.fvd_i3d`.

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
├── _activate.sh                          # sourced by launcher scripts; activates `loki`
├── setup_fvd.sh                          # one-time: pip install + cdfvd patch + 1.9 GB pre-stage
├── compute_metrics.py                    # CLI — auto-routes by protocol (per-sample)
├── compute_fvd.py                        # CLI — distribution-level FVD (HDTF same-id)
├── run_eval_metrics.sh                   # unified multi-GPU sweep (Loki + SOTA)
├── metrics/
│   ├── __init__.py
│   ├── io.py                             # video decode + run-tree walk
│   ├── evaluator.py                      # protocol-aware routing + weighted aggregation
│   └── src/
│       ├── head_rot.py                   # FLAME rot·neck_rot geodesic distance
│       ├── expression.py                 # pose-disentangled deformation-map L1
│       ├── id_sim.py                     # InsightFace ArcFace cosine (cross-id only)
│       ├── psnr.py                       # peak signal-to-noise ratio (HDTF same-id only)
│       ├── ssim.py                       # structural similarity index (HDTF same-id only)
│       ├── lpips.py                      # learned perceptual sim. — AlexNet (HDTF same-id only)
│       └── fvd.py                        # Fréchet Video Distance — VideoMAE-v2 / I3D
├── patches/
│   └── cdfvd_videomaev2_utils.py         # HuggingFace-mirror URL patch for cdfvd's broken loader
├── deformation_map_diff/                 # helper scripts for the expression-metric pipeline
└── sanity_check/                         # visual debugging — NOT used by compute_metrics.py
```
