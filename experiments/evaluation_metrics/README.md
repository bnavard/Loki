# Evaluation Metrics

Quantitative metrics for any run dir produced by the SOTA-comparison or
marionette-eval runners. The CLI reads `<run_dir>/config_resolved.json` to
recover `dataset` + `protocol`, then routes the right metric set:

| Protocol                          | Per-sample metrics             | Distribution metrics |
|-----------------------------------|--------------------------------|----------------------|
| `same_identity_reconstruction`    | PSNR, SSIM, LPIPS, LMD-F, LMD-M | FVD                  |
| `cross_identity`                  | ID similarity (ArcFace cosine)  | FVD                  |

Per-sample results land at `<run_dir>/metrics.jsonl` (one line per sample);
aggregates + FVD at `<run_dir>/metrics_summary.json`.

## Conventions

- Inputs live at `<run_dir>/samples/<sample_id>/panel.mp4` — the canonical
  per-sample prediction file every SOTA wrapper writes.
- Ground truth is the manifest's `video_path` for the sample's UID,
  resolved from `experiments/sota_comparison/manifests/<dataset>.json`.
  No per-tool GT dump required.
- Predictions are 25 fps at 512×512 across every tool in this repo;
  variable-fps GT clips are resampled to 25 fps at load time
  (nearest-neighbor in time — no temporal interpolation).

## Setup

```bash
bash experiments/evaluation_metrics/setup_env.sh
conda activate evaluation_metrics
```

## Usage

```bash
# SOTA wrapper, same-identity reconstruction:
PYTHONPATH=. python experiments/evaluation_metrics/compute_metrics.py \
    --run-dir outputs/sota_comparison/sadtalker/talkvid/same_identity_reconstruction/run_<ts>/

# SOTA wrapper, cross-identity:
PYTHONPATH=. python experiments/evaluation_metrics/compute_metrics.py \
    --run-dir outputs/sota_comparison/xportrait/talkvid/cross_identity/run_<ts>/

# Skip the distribution-level FVD step (e.g. while iterating on per-sample metrics):
PYTHONPATH=. python experiments/evaluation_metrics/compute_metrics.py \
    --run-dir outputs/sota_comparison/anitalker/hdtf/same_identity_reconstruction/run_<ts>/ \
    --skip-fvd

# Both FVD backbones (default is videomae only):
PYTHONPATH=. python experiments/evaluation_metrics/compute_metrics.py \
    --run-dir outputs/sota_comparison/hunyuan_portrait/talkvid/same_identity_reconstruction/run_<ts>/ \
    --fvd-models videomae i3d
```

## Metrics

### PSNR
`torchmetrics.functional.image.peak_signal_noise_ratio` with `data_range=1.0`.
Per-frame, averaged over time per video. Higher is better (dB).

### SSIM
`torchmetrics.functional.image.structural_similarity_index_measure` with
the canonical Wang et al. 2004 settings (11×11 Gaussian, σ=1.5,
K1=0.01, K2=0.03). Per-frame, averaged over time. Higher is better
(bounded in `[-1, 1]`, in practice `[0, 1]` for natural videos).

### LPIPS
`lpips.LPIPS(net='alex')` — the Zhang et al. 2018 paper default and what
nearly every talking-head paper reports. Inputs converted to `[-1, 1]`
at the call site. Per-frame, averaged over time. Lower is better.

### LMD-F / LMD-M
MediaPipe FaceMesh with `refine_landmarks=True` (attention-mesh variant
— sharper iris and lip precision). Per-frame Euclidean distance between
corresponding 2D landmarks on pred vs. GT, normalized by the GT frame's
inter-ocular distance (landmarks 33 ↔ 263) so the metric is
scale-invariant. Reports two variants:
- **LMD-F**: mean over all 468 face landmarks — penalizes pose /
  expression mismatch alongside lip motion.
- **LMD-M**: mean over the 22 lip-region landmarks — proxy for lip-sync
  quality.

`detect_rate` (hits / T) is reported alongside; if it drops below ~0.95
the LMD numbers are biased toward easier frames.

### ID similarity (cross-identity only)
InsightFace `buffalo_l` — RetinaFace + ArcFace R100. The identity prior
is the L2-normalized mean of ArcFace embeddings over **all frames of the
ref clip**; per-frame cosine of generated vs. prior is averaged over
time. Bounded in `[-1, 1]`; higher is better. Per-clip priors are
cached so a ref clip used by multiple cross-id pairs is only embedded
once.

### FVD (distribution metric)
`cdfvd` (Ge et al., CVPR 2024 — content-debiased FVD). Default backbone
is **VideoMAE** (converges on smaller samples than the I3D one per Luo
et al. JEDi); I3D is opt-in via `--fvd-models videomae i3d` for
compatibility with older literature.

I3D FVD is widely held to need ≥ 2k clips for stability. The talking-head
benchmarks here are 125 (TalkVid) / 212 (HDTF), so the summary tags
`low_sample = True` whenever the count is below the threshold.

Internally the runner stages a `<run_dir>/_fvd/{pred,ref}/` symlink tree
so cdfvd's video-folder loader has paired `<sample_id>.mp4` filenames.
Symlinks rather than copies — source GT files are hundreds of MB.

## Tests

```bash
PYTHONPATH=. pytest experiments/evaluation_metrics/tests/ -v
```

## Layout

```
experiments/evaluation_metrics/
├── README.md
├── setup_env.sh                      # creates `evaluation_metrics` conda env
├── env.yml                           # python 3.11 + cuda + torch
├── requirements.txt                  # pip-installable libs
├── compute_metrics.py                # CLI — auto-routes by protocol
├── metrics/
│   ├── __init__.py
│   ├── io.py                         # decode + manifest-aware GT resolution
│   ├── psnr.py                       # torchmetrics wrapper
│   ├── ssim.py                       # torchmetrics wrapper
│   ├── lpips_metric.py               # AlexNet, chunked over (B*T)
│   ├── lmd.py                        # MediaPipe FaceMesh (LMD-F + LMD-M)
│   ├── fvd.py                        # cdfvd (videomae default, i3d opt-in)
│   ├── id_sim.py                     # InsightFace ArcFace cosine
│   └── evaluator.py                  # protocol-aware routing
└── tests/
    ├── test_psnr_ssim.py             # cross-check vs scikit-image
    ├── test_lpips.py                 # identity LPIPS(x,x) ≈ 0
    ├── test_lmd.py                   # MediaPipe roundtrip + sample_id split
    └── test_io.py                    # sample_id splitter + run-metadata parsing
```
