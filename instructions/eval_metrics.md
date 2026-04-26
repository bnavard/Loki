# Video Diffusion Evaluation Metrics — Implementation Guide

This guide specifies how to implement an evaluation suite for a talking-head video diffusion model. The deliverable is a Python package `metrics/` that reports **PSNR, SSIM, LPIPS, FVD, LMD-F, LMD-M** for paired generated/ground-truth videos. The suite must be reproducible, GPU-accelerated where possible, and produce numbers that are comparable across runs and (modulo detector choice for LMD) across papers.

---

## 1. Conventions

All metric functions in this package operate on a single, fixed tensor convention. Convert at the boundary; do not let multiple conventions coexist internally.

- **Video tensor shape**: `(B, T, C, H, W)` where `C=3` (RGB), float32, values in `[0, 1]`.
- **Image tensor shape**: `(B, C, H, W)`, same dtype/range.
- **Frame ordering**: temporal axis is contiguous, no skipping. If the source is variable-fps, resample to a fixed fps (default 25 for talking-head) before passing in.
- **Pairing**: PSNR / SSIM / LPIPS / LMD require frame-level pairing between prediction and reference. FVD does not — it compares distributions.
- **Device**: every metric class accepts a `device` argument and moves all internal state (including LPIPS network and FVD I3D) to that device once at construction. Do not move per-call.
- **Reduction**: per-frame metrics return `(B,)` per-video means by default; aggregate to a scalar at the reporter level, not inside the metric.

A small `metrics/io.py` should provide:

```python
def load_video(path: str | Path, fps: int = 25, resolution: int | None = None) -> torch.Tensor:
    """Load video → (T, 3, H, W) float32 in [0, 1]. Resamples to `fps`. Resizes (bilinear) if `resolution` set."""

def iter_video_pairs(pred_dir: Path, ref_dir: Path, fps: int = 25, resolution: int | None = None
                     ) -> Iterator[tuple[str, torch.Tensor, torch.Tensor]]:
    """Yield (video_id, pred, ref) with matched filenames. Truncate to min(T_pred, T_ref)."""
```

Use `decord` for loading (faster than `torchvision.io` for random access). Fall back to `imageio[ffmpeg]` if `decord` is unavailable.

---

## 2. Dependencies

```
torch>=2.1
torchvision
torchmetrics>=1.3
lpips>=0.1.4
cdfvd                      # content-debiased FVD; gives you both I3D and VideoMAE backbones
mediapipe>=0.10
scikit-image               # reference PSNR/SSIM only, used in tests
decord                     # fast video IO
numpy
einops
tqdm
pyyaml                     # config
```

Install with `pip install -r requirements.txt`. Pin versions in CI.

---

## 3. Project layout

```
metrics/
├── __init__.py
├── io.py                  # video loading, pair iteration
├── psnr.py                # thin wrapper around torchmetrics
├── ssim.py                # thin wrapper around torchmetrics
├── lpips_metric.py        # thin wrapper around `lpips` package
├── fvd.py                 # wrapper around cdfvd
├── lmd.py                 # MediaPipe-based, custom
├── evaluator.py           # unified runner
└── cli.py                 # argparse entry point
tests/
├── test_psnr_ssim.py      # cross-check vs scikit-image
├── test_lpips.py          # cross-check vs official lpips repo
├── test_fvd.py            # identity test: FVD(X, X) == 0 within tolerance
└── test_lmd.py            # synthetic landmark test
configs/
└── default.yaml
```

---

## 4. PSNR

**Library**: `torchmetrics.image.PeakSignalNoiseRatio`.

Per-frame PSNR averaged across frames and across the batch. `data_range=1.0` because inputs are in `[0, 1]`.

```python
# metrics/psnr.py
import torch
from torchmetrics.functional.image import peak_signal_noise_ratio

def psnr_video(pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    pred, ref: (B, T, 3, H, W) in [0, 1]
    returns: (B,) — mean PSNR over frames per video
    """
    assert pred.shape == ref.shape
    B, T = pred.shape[:2]
    pred_flat = pred.reshape(B * T, *pred.shape[2:])
    ref_flat = ref.reshape(B * T, *ref.shape[2:])
    psnr_per_frame = peak_signal_noise_ratio(
        pred_flat, ref_flat, data_range=1.0, reduction="none", dim=(1, 2, 3)
    )  # (B*T,)
    return psnr_per_frame.view(B, T).mean(dim=1)
```

**Notes**:
- `data_range` must be `1.0` (matches our `[0, 1]` convention). If you change the convention, change this.
- Some talking-head papers report PSNR on the cropped face region only. Expose a `mask` argument later if needed; v1 reports full-frame.

---

## 5. SSIM

**Library**: `torchmetrics.image.StructuralSimilarityIndexMeasure`.

Use the default Wang et al. 2004 settings: 11×11 Gaussian kernel, σ=1.5, K1=0.01, K2=0.03. These are the defaults; do not change them — the field implicitly assumes them.

```python
# metrics/ssim.py
import torch
from torchmetrics.functional.image import structural_similarity_index_measure

def ssim_video(pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    pred, ref: (B, T, 3, H, W) in [0, 1]
    returns: (B,) — mean SSIM over frames per video
    """
    assert pred.shape == ref.shape
    B, T = pred.shape[:2]
    pred_flat = pred.reshape(B * T, *pred.shape[2:])
    ref_flat = ref.reshape(B * T, *ref.shape[2:])
    ssim_per_frame = structural_similarity_index_measure(
        pred_flat, ref_flat,
        data_range=1.0,
        gaussian_kernel=True, sigma=1.5, kernel_size=11,
        k1=0.01, k2=0.03,
        reduction="none",
    )  # (B*T,)
    return ssim_per_frame.view(B, T).mean(dim=1)
```

---

## 6. LPIPS

**Library**: official `lpips` package by Richard Zhang.

Use **`net='alex'`**. This is the default in the original repo and what nearly all talking-head papers report. `vgg` is for backprop / perceptual loss, not reporting.

LPIPS expects inputs in **`[-1, 1]`**, not `[0, 1]`. Convert at the call site.

```python
# metrics/lpips_metric.py
import torch
import lpips

class LPIPSMetric:
    def __init__(self, net: str = "alex", device: str = "cuda"):
        self.device = device
        self.model = lpips.LPIPS(net=net).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """
        pred, ref: (B, T, 3, H, W) in [0, 1]
        returns: (B,) — mean LPIPS over frames per video
        """
        assert pred.shape == ref.shape
        B, T = pred.shape[:2]
        pred = (pred.reshape(B * T, *pred.shape[2:]).to(self.device) * 2 - 1)
        ref  = (ref.reshape(B * T,  *ref.shape[2:]).to(self.device) * 2 - 1)
        d = self.model(pred, ref).view(B, T)  # (B, T)
        return d.mean(dim=1)
```

**Notes**:
- Batch size for the inner call is `B*T`. For long videos, chunk to avoid OOM (default chunk size 64 frames).
- Disable grad and `.eval()` once at construction.
- Pin `lpips==0.1.4` so the AlexNet weights are stable.

---

## 7. FVD

**Library**: `cdfvd` (content-debiased FVD by Ge et al., CVPR 2024). It exposes both the original I3D backbone and the VideoMAE backbone. Report **both** — I3D for compatibility with existing literature, VideoMAE for the more reliable signal.

FVD compares **distributions**, so it requires aggregated statistics over a sample (typically ≥ 2048 clips for I3D to converge; fewer for VideoMAE). Do not call it on a single video.

```python
# metrics/fvd.py
from pathlib import Path
import torch
from cdfvd import fvd as cdfvd_module

class FVD:
    def __init__(self, model: str = "i3d", resolution: int = 224, sequence_length: int = 16,
                 device: str = "cuda"):
        """
        model: "i3d" or "videomae"
        resolution: 224 for I3D, 224 for VideoMAE
        sequence_length: number of frames per clip; 16 is standard for I3D
        """
        self.evaluator = cdfvd_module.cdfvd(
            model, n_real="full", n_fake="full",
            ckpt_path=None, seed=0, compute_feats=False,
            device=device,
        )
        self.resolution = resolution
        self.sequence_length = sequence_length

    def compute(self, pred_dir: Path, ref_dir: Path) -> float:
        """
        pred_dir, ref_dir: folders of video files (one .mp4 per sample)
        returns: FVD scalar
        """
        real_loader = self.evaluator.load_videos(
            str(ref_dir), data_type="video_folder",
            resolution=self.resolution, sequence_length=self.sequence_length,
        )
        fake_loader = self.evaluator.load_videos(
            str(pred_dir), data_type="video_folder",
            resolution=self.resolution, sequence_length=self.sequence_length,
        )
        self.evaluator.compute_real_stats(real_loader)
        self.evaluator.compute_fake_stats(fake_loader)
        return float(self.evaluator.compute_fvd_from_stats())
```

**Notes**:
- For I3D, **at least ~2k clips** are needed for the metric to stabilize. Below that, report it but flag it as low-sample.
- VideoMAE features converge faster (~16% of the I3D sample size per Luo et al. JEDi), so on small eval sets prefer that number.
- Sample clips of `sequence_length` frames at uniform stride; do not use heavily overlapping clips (biases the metric).
- Save and cache real-side stats — they don't change across runs of the same eval set.

---

## 8. LMD (Landmark Mean Distance)

LMD is the mean Euclidean distance between corresponding 2D facial landmarks on predicted vs ground-truth frames, averaged over frames and videos. Report **two variants**:

- **LMD-F**: all face landmarks (penalizes pose/expression mismatch — useful when GT pose should be matched).
- **LMD-M**: mouth-region landmarks only (proxy for lip-sync quality).

Use **MediaPipe FaceMesh with `refine_landmarks=True`** (the attention-mesh variant — better lip and iris precision). 468 base landmarks + 10 refined iris landmarks.

### 8.1 Mouth landmark indices (MediaPipe topology)

```python
# Outer lip + inner lip (matches the FaceMesh canonical topology)
MOUTH_LANDMARKS = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,   # outer
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,    # inner
]
```

### 8.2 Normalization

Raw pixel distances are scale-dependent. Normalize each frame's distances by the **inter-ocular distance** (distance between left eye outer corner `33` and right eye outer corner `263`) on the **ground-truth** frame. This makes LMD scale-invariant and comparable across resolutions.

State this normalization explicitly in any reported result. Papers that don't normalize report numbers in pixels at a fixed resolution (commonly 256×256); both are acceptable but you must pick one and stick with it.

### 8.3 Implementation

```python
# metrics/lmd.py
import numpy as np
import torch
import mediapipe as mp

MOUTH_LANDMARKS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
                   78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263

class LMD:
    def __init__(self, normalize_by_iod: bool = True):
        self.normalize_by_iod = normalize_by_iod
        self._mp = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,           # per-frame, no temporal smoothing
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )

    def _extract(self, frame_uint8: np.ndarray) -> np.ndarray | None:
        """frame_uint8: (H, W, 3) RGB uint8. Returns (478, 2) pixel coords or None."""
        H, W, _ = frame_uint8.shape
        res = self._mp.process(frame_uint8)
        if not res.multi_face_landmarks:
            return None
        lms = res.multi_face_landmarks[0].landmark
        return np.array([[p.x * W, p.y * H] for p in lms], dtype=np.float32)

    def __call__(self, pred: torch.Tensor, ref: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        pred, ref: (B, T, 3, H, W) in [0, 1]
        returns: {"lmd_f": (B,), "lmd_m": (B,), "detect_rate": (B,)}
        """
        assert pred.shape == ref.shape
        B, T = pred.shape[:2]
        out_f = torch.full((B,), float("nan"))
        out_m = torch.full((B,), float("nan"))
        detect = torch.zeros(B)

        for b in range(B):
            f_dists, m_dists, hits = [], [], 0
            for t in range(T):
                p = (pred[b, t].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                r = (ref[b, t].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                lp = self._extract(p)
                lr = self._extract(r)
                if lp is None or lr is None:
                    continue
                hits += 1
                if self.normalize_by_iod:
                    iod = np.linalg.norm(lr[LEFT_EYE_OUTER] - lr[RIGHT_EYE_OUTER]) + 1e-8
                else:
                    iod = 1.0
                d = np.linalg.norm(lp - lr, axis=1) / iod  # (468,)
                f_dists.append(d.mean())
                m_dists.append(d[MOUTH_LANDMARKS].mean())
            if hits > 0:
                out_f[b] = float(np.mean(f_dists))
                out_m[b] = float(np.mean(m_dists))
                detect[b] = hits / T

        return {"lmd_f": out_f, "lmd_m": out_m, "detect_rate": detect}

    def close(self):
        self._mp.close()
```

**Notes**:
- `static_image_mode=True` because the per-frame pipeline shouldn't carry temporal state — it would couple consecutive measurements.
- Detection failures are a real failure mode on poor generations. Report `detect_rate` alongside LMD; if it drops below ~0.95 the LMD numbers are not trustworthy.
- This is **not** comparable across detectors. Pin `mediapipe==0.10.x`. If you ever switch to dlib or FAN, re-evaluate everything.
- For batch speed, MediaPipe is single-threaded per `FaceMesh` instance; parallelize across videos with `multiprocessing` if needed (don't share the `FaceMesh` across processes — construct one per worker).

---

## 9. Unified evaluator

```python
# metrics/evaluator.py
from dataclasses import dataclass, field
from pathlib import Path
import torch
from tqdm import tqdm
from .io import iter_video_pairs
from .psnr import psnr_video
from .ssim import ssim_video
from .lpips_metric import LPIPSMetric
from .fvd import FVD
from .lmd import LMD

@dataclass
class EvalConfig:
    fps: int = 25
    resolution: int = 256
    device: str = "cuda"
    fvd_models: list[str] = field(default_factory=lambda: ["i3d", "videomae"])
    fvd_seq_len: int = 16
    lmd_normalize: bool = True

def evaluate(pred_dir: Path, ref_dir: Path, cfg: EvalConfig) -> dict:
    lpips_m = LPIPSMetric(net="alex", device=cfg.device)
    lmd_m = LMD(normalize_by_iod=cfg.lmd_normalize)

    psnr_all, ssim_all, lpips_all = [], [], []
    lmd_f_all, lmd_m_all, det_all = [], [], []

    for vid, pred, ref in tqdm(iter_video_pairs(pred_dir, ref_dir, cfg.fps, cfg.resolution),
                                desc="per-video metrics"):
        pred = pred.unsqueeze(0).to(cfg.device)  # (1, T, 3, H, W)
        ref  = ref.unsqueeze(0).to(cfg.device)
        psnr_all.append(psnr_video(pred, ref).item())
        ssim_all.append(ssim_video(pred, ref).item())
        lpips_all.append(lpips_m(pred, ref).item())
        lmd = lmd_m(pred.cpu(), ref.cpu())
        lmd_f_all.append(lmd["lmd_f"].item())
        lmd_m_all.append(lmd["lmd_m"].item())
        det_all.append(lmd["detect_rate"].item())

    results = {
        "psnr": float(torch.tensor(psnr_all).mean()),
        "ssim": float(torch.tensor(ssim_all).mean()),
        "lpips": float(torch.tensor(lpips_all).mean()),
        "lmd_f": float(torch.tensor(lmd_f_all).nanmean()),
        "lmd_m": float(torch.tensor(lmd_m_all).nanmean()),
        "lmd_detect_rate": float(torch.tensor(det_all).mean()),
    }

    for backbone in cfg.fvd_models:
        fvd = FVD(model=backbone, sequence_length=cfg.fvd_seq_len, device=cfg.device)
        results[f"fvd_{backbone}"] = fvd.compute(pred_dir, ref_dir)

    lmd_m.close()
    return results
```

CLI in `metrics/cli.py` should accept `--pred-dir`, `--ref-dir`, `--config`, `--out results.json`.

---

## 10. Validation tests

These are non-optional. The metric numbers are only meaningful if cross-checked.

- **`test_psnr_ssim.py`**: on 10 random `(256, 256, 3)` uint8 image pairs, `metrics.psnr_video` and `metrics.ssim_video` must match `skimage.metrics.peak_signal_noise_ratio` and `skimage.metrics.structural_similarity` (with `data_range=1.0`, `channel_axis=-1`, `gaussian_weights=True`, `sigma=1.5`, `use_sample_covariance=False`) within `1e-3`.
- **`test_lpips.py`**: identity check — `LPIPS(x, x) < 1e-4` for any `x`.
- **`test_fvd.py`**: identity check — `FVD(real_dir, real_dir) < 5.0` for I3D on a 256-clip sample (it's not exactly zero due to the matrix-sqrt numerics).
- **`test_lmd.py`**: identity check on real video — `LMD(real, real) == 0` exactly when detection succeeds. Synthetic shift test: shift one video by 5 px → LMD should equal 5.0 / IOD.

---

## 11. Common pitfalls

1. **Range mismatch**: LPIPS wants `[-1, 1]`; PSNR/SSIM want `[0, 1]` with `data_range=1.0`. Mixing these is the most common bug.
2. **Resolution drift**: If pred and ref are at different resolutions, resize both to the same target before any metric. PSNR/SSIM/LPIPS at mismatched resolutions are meaningless.
3. **Frame count mismatch**: truncate to `min(T_pred, T_ref)` at load time, do not pad.
4. **FVD on small samples**: any FVD computed on < 1k clips for I3D should be flagged in the report. VideoMAE is more forgiving but still wants a few hundred.
5. **MediaPipe in multi-process**: construct one `FaceMesh` per worker; sharing causes silent corruption.
6. **LMD on failed generations**: if `detect_rate < 0.95`, the LMD numbers are biased toward easier frames. Report both.
7. **fps mismatch**: a 30 fps prediction vs 25 fps GT will pair the wrong frames. Always resample to a common fps at load time.
8. **Color space**: assume RGB throughout. `decord` returns RGB; OpenCV returns BGR — convert if you ever introduce OpenCV.
9. **Cached FVD stats**: real-side stats can be cached as `.pkl`; do this once per benchmark dataset to save ~minutes per run.

---

## 12. Acceptance checklist

- [ ] `pip install -r requirements.txt` succeeds in a clean venv.
- [ ] All four tests in `tests/` pass.
- [ ] `python -m metrics.cli --pred-dir <demo_pred> --ref-dir <demo_ref>` produces a JSON with all 7 numbers.
- [ ] Numbers for PSNR/SSIM/LPIPS on a known pair match a reference implementation within `1e-3`.
- [ ] FVD on `(real, real)` is < 5.0 for I3D and < 1.0 for VideoMAE on a 256-clip set.
- [ ] LMD detect rate is reported and is ≥ 0.95 on the GT side.
