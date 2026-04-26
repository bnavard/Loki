# Evaluation Metrics

Quantitative metrics for any run dir produced by the SOTA-comparison or
marionette-eval runners. The CLI reads `<run_dir>/config_resolved.json` to
recover `dataset` + `protocol`, then routes the right metric set:

| Protocol                          | Per-sample metrics              | Distribution metrics |
|-----------------------------------|---------------------------------|----------------------|
| `same_identity_reconstruction`    | PSNR, SSIM, LPIPS, LMD-F, LMD-M | FVD                  |
| `cross_identity`                  | ID similarity (ArcFace cosine)  | FVD                  |

Per-sample results land at `<run_dir>/metrics.jsonl` (one line per sample);
aggregates + FVD at `<run_dir>/metrics_summary.json`.

## Conventions

- **Inputs**: `<run_dir>/samples/<sample_id>/panel.mp4` — the canonical
  per-sample prediction file every SOTA wrapper writes.
- **Ground truth**: the manifest's `video_path` for the sample's UID,
  resolved from `experiments/sota_comparison/manifests/<dataset>.json`.
  No per-tool GT dump required.
- **fps + resolution**: predictions are 25 fps at 512×512 across every
  tool in this repo; variable-fps GT clips are resampled to 25 fps at
  load time (nearest-neighbor — no temporal interpolation).
- **Face-region cropping** (default-on, same-id only): both pred and GT
  are independently cropped to a square around the detected face
  (`1.3 ×` the RetinaFace bbox), then resized to 512×512. PSNR / SSIM /
  LPIPS / LMD therefore measure face-region quality directly, with no
  framing or scale asymmetry between pred and GT. Disable with
  `--no-face-crop` for raw-framing numbers.
- **Per-frame handling**:
  - PSNR / SSIM / LPIPS run on **every** paired frame — broken pred
    frames produce bad numbers and naturally penalize the tool.
  - LMD-F / LMD-M and ID-cosine require face detection per frame; a
    frame contributes only when MediaPipe (LMD) / ArcFace (ID-cosine)
    detects a face. The per-sample `lmd_detect_rate` /
    `id_detect_rate` is recorded so failure density is visible.
- **Run-level aggregation**:
  - PSNR / SSIM / LPIPS: arithmetic mean across samples.
  - LMD-F / LMD-M / ID-cosine: **weighted** mean using each sample's
    detect rate as the weight, so noisy samples (where many frames
    failed detection) contribute proportionally less.
- **Sample skip**: only when clip-level RetinaFace fails on the first
  10 probe frames of pred or GT (logged as
  `skipped: face_detection_failed_clip` in `metrics.jsonl`). Per-frame
  failures don't trigger a skip.

## Setup

```bash
bash experiments/evaluation_metrics/setup_env.sh
conda activate evaluation_metrics
```

## Usage

### Single run

```bash
# Same-identity:
PYTHONPATH=. python experiments/evaluation_metrics/compute_metrics.py \
    --run-dir outputs/sota_comparison/sadtalker/talkvid/same_identity_reconstruction/run_<ts>/

# Cross-identity:
PYTHONPATH=. python experiments/evaluation_metrics/compute_metrics.py \
    --run-dir outputs/sota_comparison/xportrait/talkvid/cross_identity/run_<ts>/

# Skip the distribution-level FVD step (faster — useful while iterating):
PYTHONPATH=. python experiments/evaluation_metrics/compute_metrics.py \
    --run-dir outputs/sota_comparison/anitalker/hdtf/same_identity_reconstruction/run_<ts>/ \
    --skip-fvd

# Both FVD backbones (default is videomae only):
PYTHONPATH=. python experiments/evaluation_metrics/compute_metrics.py \
    --run-dir outputs/sota_comparison/hunyuan_portrait/talkvid/same_identity_reconstruction/run_<ts>/ \
    --fvd-models videomae i3d
```

### Sweep all SOTA runs

`run_eval_metrics_on_sota.sh` walks every
`outputs/sota_comparison/<baseline>/<dataset>/<protocol>/run_*/`, runs
`compute_metrics.py --skip-fvd` on each, and mirrors every
`metrics_summary.json` into a centralized tree under
`outputs/test_metric/metrics/<baseline>/<dataset>/<protocol>/`:

```bash
bash experiments/evaluation_metrics/run_eval_metrics_on_sota.sh \
    > outputs/test_metric/metrics/_batch.log 2>&1 &
tail -f outputs/test_metric/metrics/_batch.log
```

Idempotent: already-summarized run dirs are skipped on re-run.

## Metrics

### PSNR
`torchmetrics.functional.image.peak_signal_noise_ratio` with
`data_range=1.0`. Per-frame, averaged over time per sample. Higher is
better (dB).

### SSIM
`torchmetrics.functional.image.structural_similarity_index_measure` with
canonical Wang et al. 2004 settings (11×11 Gaussian, σ=1.5, K1=0.01,
K2=0.03). Per-frame, averaged over time. Higher is better.

### LPIPS
`lpips.LPIPS(net='alex')` — the Zhang et al. 2018 paper default and the
talking-head literature standard. Inputs converted to `[-1, 1]` at the
call site. Per-frame, averaged over time. Lower is better.

### LMD-F / LMD-M
MediaPipe FaceLandmarker (Tasks API, `face_landmarker_v2_with_blendshapes.task`).
Per-frame Euclidean distance between corresponding 2D landmarks on pred
vs. GT, normalized by the GT frame's inter-ocular distance (landmarks
33 ↔ 263) so the metric is scale-invariant.
- **LMD-F**: mean over all 478 face landmarks — penalizes pose /
  expression mismatch alongside lip motion.
- **LMD-M**: mean over the 22 lip-region landmarks — proxy for lip-sync
  quality.

`lmd_detect_rate` is reported per-sample; the run-level mean for LMD-F
and LMD-M is **weighted** by this rate.

### ID similarity (cross-identity only)
InsightFace `buffalo_l` — RetinaFace + ArcFace R100. The identity prior
is the L2-normalized mean of ArcFace embeddings over **all frames of the
ref clip**; per-frame cosine of generated vs. prior is averaged over
time. Bounded in `[-1, 1]`; higher is better. Per-clip priors are
cached so a ref clip used by multiple cross-id pairs is only embedded
once. Run-level mean is **weighted** by `id_detect_rate`.

### FVD (distribution metric)
`cd-fvd` (Ge et al., CVPR 2024 — content-debiased FVD). Default backbone
is **VideoMAE** (converges on smaller samples than the I3D one per Luo
et al. JEDi); I3D is opt-in via `--fvd-models videomae i3d`.

I3D FVD is widely held to need ≥ 2k clips for stability. The
talking-head benchmarks here are 125 (TalkVid) / 212 (HDTF), so the
summary tags `low_sample = True` whenever the count is below the
threshold.

Internally the runner stages a `<run_dir>/_fvd/{pred,ref}/` tree where
both sides are face-cropped via the same routine used for per-sample
metrics, so the FVD distribution comparison is on aligned face crops.

## Sanity check

`sanity_check/` ships visualization tools that *don't* compute the
aggregate metrics — they render per-frame overlays so you can verify by
eye that the metric inputs are sensible (face crops, landmark detection,
ArcFace bbox).

```bash
# One sample (same-id or cross-id auto-detected from the run's protocol):
PYTHONPATH=. python experiments/evaluation_metrics/sanity_check/visualize_sample.py \
    --run-dir outputs/sota_comparison/sadtalker/talkvid/same_identity_reconstruction/run_<ts>/ \
    --sample-id id_0042

# Batch — first / middle / last sample of every run dir, mirroring under
# outputs/test_metric/visualizations/<baseline>/<dataset>/<protocol>/<sample_id>/:
bash experiments/evaluation_metrics/sanity_check/visualize_batch.sh \
    > outputs/test_metric/visualizations/_batch.log 2>&1 &
```

## Layout

```
experiments/evaluation_metrics/
├── README.md
├── setup_env.sh                          # creates `evaluation_metrics` conda env
├── env.yml                               # python 3.11 + cuda + torch
├── requirements.txt                      # pip-installable libs
├── compute_metrics.py                    # CLI — auto-routes by protocol
├── run_eval_metrics_on_sota.sh           # sweep every outputs/sota_comparison/**/run_*/
├── metrics/
│   ├── __init__.py
│   ├── io.py                             # decode + manifest GT resolution + face crop
│   ├── psnr.py                           # torchmetrics wrapper
│   ├── ssim.py                           # torchmetrics wrapper
│   ├── lpips_metric.py                   # AlexNet, chunked over (B*T)
│   ├── lmd.py                            # MediaPipe FaceLandmarker (LMD-F + LMD-M)
│   ├── fvd.py                            # cd-fvd (videomae default, i3d opt-in)
│   ├── id_sim.py                         # InsightFace ArcFace cosine
│   └── evaluator.py                      # protocol-aware routing + weighted aggregation
└── sanity_check/                         # visual debugging — NOT used by compute_metrics.py
    ├── visualize_sample.py               # one sample → overlay mp4 + curves PNG
    └── visualize_batch.sh                # 60 samples across the SOTA tree
```
