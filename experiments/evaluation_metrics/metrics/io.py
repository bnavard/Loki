"""Video and run-tree IO for the metrics package.

Two responsibilities:

1. **Decode an mp4 to a `(T, 3, H, W)` float32 tensor in `[0, 1]`** at fixed
   fps + resolution. Defaults match this repo's prediction surface — every
   Marionette / SOTA-wrapper `panel.mp4` is 25 fps at 512×512 (AniTalker
   even has a GFPGAN 256→512 upscale specifically to match). Source GT
   clips have variable fps (TalkVid manifest spans 23.97 – 60); the
   resampler normalizes them to the prediction's rate so paired metrics
   line up frame-for-frame. Uses `decord` (fast random access); falls back
   to `imageio.v3` if `decord` isn't available.

2. **Walk a SOTA-comparison-style run dir** — `<run_dir>/samples/<sample_id>/panel.mp4`
   plus a `config_resolved.json` at the run root carrying `dataset` / `protocol` —
   and pair each sample with its ground-truth video resolved from the curated
   manifest under `experiments/sota_comparison/manifests/<dataset>.json`.

   For `same_identity_reconstruction`: GT = the manifest clip whose `uid`
   matches the sample's `id_XXXX`.
   For `cross_identity` (sample_id = `id_XXXX_id_YYYY`): the *ref* clip's
   identity image is the GT for identity-preservation metrics; there is no
   temporal GT.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import torch
import torch.nn.functional as F


# Every prediction in this repo (Marionette + 5 SOTA wrappers) lands at
# 25 fps and 512×512. Defaults match that surface so the GT side is
# normalized to the prediction's rate / resolution rather than the other
# way around.
DEFAULT_FPS        = 25
DEFAULT_RESOLUTION = 512

# Face-crop margin around the raw RetinaFace bbox. 1.3× adds a small
# forehead + jaw cushion to the eyebrows-to-chin RetinaFace box without
# pulling in hair or shoulders, so the cropped face fills ~80% of the
# resized frame on both pred and GT.
DEFAULT_FACE_CROP_MARGIN = 1.3


# ---------------------------------------------------------------------------
# Video decoding
# ---------------------------------------------------------------------------


def load_video(
    path: str | Path,
    fps: Optional[int] = DEFAULT_FPS,
    resolution: Optional[int] = None,
    max_frames: Optional[int] = None,
) -> torch.Tensor:
    """Decode `path` to `(T, 3, H, W)` float32 in `[0, 1]`.

    Args:
        path: video file (mp4 / mkv / etc.).
        fps: target fps. If set and the source fps differs, frames are
             nearest-neighbor resampled along time. None = leave as-is.
             Default 25 — every prediction in this repo is at 25 fps; setting
             this normalizes a variable-fps GT clip to the same rate.
        resolution: target square resolution. If set, frames are bilinearly
             resized to `(resolution, resolution)`. None = leave as-is.
        max_frames: clip the temporal axis to at most this many frames.

    Returns:
        `(T, 3, H, W)` torch.float32 in `[0, 1]`, RGB.
    """
    path = Path(path)
    try:
        import decord  # type: ignore
        decord.bridge.set_bridge("torch")
        vr = decord.VideoReader(str(path))
        src_fps = float(vr.get_avg_fps())
        n_src = len(vr)
        idx = _resample_frame_indices(n_src, src_fps, fps, max_frames)
        frames = vr.get_batch(idx)  # (T, H, W, 3) uint8 torch
    except ImportError:
        frames = _imageio_load(path, fps, max_frames)

    # (T, H, W, 3) → (T, 3, H, W), uint8 → float32 in [0, 1].
    frames = frames.to(torch.float32).permute(0, 3, 1, 2) / 255.0

    if resolution is not None and (frames.shape[-1] != resolution or frames.shape[-2] != resolution):
        frames = F.interpolate(
            frames, size=(resolution, resolution),
            mode="bilinear", align_corners=False,
        )

    return frames


def _resample_frame_indices(
    n_src: int,
    src_fps: float,
    target_fps: Optional[int],
    max_frames: Optional[int],
) -> list[int]:
    """Pick source-frame indices to land at `target_fps`.

    Nearest-neighbor in time — no temporal interpolation, since interpolated
    frames never existed in the source and would inject phantom content into
    the GT that the prediction was never asked to match.

    Identity (returns `range(n_src)`) when `target_fps` is None or matches
    source within 1e-3 fps.
    """
    if target_fps is None or abs(src_fps - target_fps) < 1e-3:
        idx = list(range(n_src))
    else:
        n_target = max(1, int(round(n_src * (target_fps / src_fps))))
        idx = [
            min(n_src - 1, int(round(t * src_fps / target_fps)))
            for t in range(n_target)
        ]
    if max_frames is not None and len(idx) > max_frames:
        idx = idx[:max_frames]
    return idx


def _imageio_load(path: Path, fps: Optional[int], max_frames: Optional[int]) -> torch.Tensor:
    """Fallback decoder. imageio.v3 is slower than decord but reads the same
    files via ffmpeg."""
    import imageio.v3 as iio
    meta = iio.immeta(path, plugin="FFMPEG")
    src_fps = float(meta.get("fps", DEFAULT_FPS))
    arr = iio.imread(path, plugin="FFMPEG")  # (T, H, W, 3) uint8 ndarray
    n_src = arr.shape[0]
    idx = _resample_frame_indices(n_src, src_fps, fps, max_frames)
    return torch.from_numpy(arr[idx])  # (T, H, W, 3) uint8


# ---------------------------------------------------------------------------
# Run-tree walking + GT resolution
# ---------------------------------------------------------------------------


@dataclass
class RunMetadata:
    """Resolved metadata about a single run dir."""
    run_dir:     Path
    dataset:     str   # "talkvid" | "hdtf"
    protocol:    str   # "same_identity_reconstruction" | "cross_identity"
    manifest:    dict
    uid_to_clip: dict[str, dict]   # "id_0457" → manifest clip dict


def load_run_metadata(run_dir: Path) -> RunMetadata:
    """Read `<run_dir>/config_resolved.json` to recover dataset+protocol,
    then load + index the curated manifest. Fail loud if it's missing —
    we'd rather refuse to run than silently mis-route metrics.

    Every runner in this repo (marionette_eval + the 5 SOTA wrappers)
    writes a `config_resolved.json` at the run root with at least
    `dataset` and `protocol` populated.
    """
    run_dir = Path(run_dir)
    args_path = run_dir / "config_resolved.json"
    if not args_path.is_file():
        raise FileNotFoundError(
            f"`{args_path}` not found. The metrics runner needs to read "
            f"`dataset` and `protocol` from the run's recorded config."
        )
    args = json.loads(args_path.read_text())
    dataset  = args.get("dataset")
    protocol = args.get("protocol")
    if dataset is None or protocol is None:
        raise ValueError(f"`dataset` / `protocol` missing from {args_path}.")

    manifest_path = Path("experiments/sota_comparison/manifests") / f"{dataset}.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Curated manifest not found at {manifest_path}. Build it via "
            f"`experiments/sota_comparison/dataset/build_manifest.py --dataset {dataset}`."
        )
    manifest = json.loads(manifest_path.read_text())
    uid_to_clip = {c["uid"]: c for c in manifest["clips"]}

    return RunMetadata(
        run_dir=run_dir, dataset=dataset, protocol=protocol,
        manifest=manifest, uid_to_clip=uid_to_clip,
    )


@dataclass
class SamplePair:
    """One row to evaluate. `gt_video_path` is None when no temporal GT
    exists for the protocol (cross-identity)."""
    sample_id:     str
    pred_path:     Path
    gt_video_path: Optional[Path]   # for same-identity reconstruction
    ref_clip:      dict             # ref-side manifest entry (always present)
    driver_clip:   dict             # driver-side manifest entry (always present)


def iter_samples(meta: RunMetadata) -> Iterator[SamplePair]:
    """Yield one `SamplePair` per `<run_dir>/samples/<sample_id>/panel.mp4`.

    Skips entries with no `panel.mp4` (e.g. samples that errored — see the
    run's `failed.json`).
    """
    samples_root = meta.run_dir / "samples"
    if not samples_root.is_dir():
        raise FileNotFoundError(
            f"`{samples_root}` not found. The runner expects a "
            f"`samples/<sample_id>/panel.mp4` tree under the run dir."
        )

    for sample_dir in sorted(samples_root.iterdir()):
        if not sample_dir.is_dir():
            continue
        pred_path = sample_dir / "panel.mp4"
        if not pred_path.is_file():
            continue

        sample_id = sample_dir.name
        ref_uid, drv_uid = _split_sample_id(sample_id, meta.protocol)
        if ref_uid not in meta.uid_to_clip or drv_uid not in meta.uid_to_clip:
            # Sample folder doesn't match any manifest UID — skip, don't fail
            # (lets us tolerate manual additions / re-runs with stale dirs).
            continue
        ref_clip = meta.uid_to_clip[ref_uid]
        drv_clip = meta.uid_to_clip[drv_uid]

        # Same-identity: GT = the (single) clip's video. Cross-identity: no GT.
        gt_video = (
            Path(ref_clip["video_path"])
            if meta.protocol == "same_identity_reconstruction" else None
        )

        yield SamplePair(
            sample_id     = sample_id,
            pred_path     = pred_path,
            gt_video_path = gt_video,
            ref_clip      = ref_clip,
            driver_clip   = drv_clip,
        )


def _split_sample_id(sample_id: str, protocol: str) -> tuple[str, str]:
    """`id_0457` → (`id_0457`, `id_0457`); `id_0457_id_0009` → (`id_0457`, `id_0009`).
    The two-uid form is recognized by an `_id_` infix."""
    if protocol == "same_identity_reconstruction":
        return sample_id, sample_id
    if "_id_" not in sample_id:
        raise ValueError(
            f"cross_identity sample_id `{sample_id}` doesn't contain `_id_` "
            f"separator. Expected `id_<ref>_id_<drv>`."
        )
    ref_uid, drv_part = sample_id.split("_id_", 1)
    return ref_uid, f"id_{drv_part}"


def truncate_to_match(pred: torch.Tensor, ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Truncate pred/ref to `min(T_pred, T_ref)` frames so paired metrics
    have aligned tensors. Padding is wrong here — it would inject
    near-identical frames that bias PSNR/SSIM upward."""
    T = min(pred.shape[0], ref.shape[0])
    return pred[:T], ref[:T]


# ---------------------------------------------------------------------------
# Face-cropping for paired-metric framing alignment
# ---------------------------------------------------------------------------
# Predictions in this repo are already face-cropped at 512×512 by each
# generation tool, but the tool-specific crop varies (SadTalker is tighter
# than X-Portrait, etc.) and the manifest GT clips aren't face-cropped at
# all. Pixel-aligned metrics (PSNR / SSIM / LPIPS) collapse if pred and GT
# don't have the face filling the same fraction of the frame.
#
# The fix: detect a face bbox on the first hit-frame of each video, expand
# it by `margin` (1.5× matches SadTalker's convention and is roughly the
# talking-head literature default), square it, and crop+resize to the
# target resolution. Apply the same procedure to **both** sides so the
# framing is aligned regardless of each tool's idiosyncratic crop.


def detect_face_bbox_xyxy(
    video: torch.Tensor,
    detect_fn,                       # callable: np_uint8_bgr -> list (insightface Face-likes)
    max_probe_frames: int = 10,
) -> tuple[float, float, float, float] | None:
    """Detect the largest face on the first hit-frame; return its raw
    `(x1, y1, x2, y2)` in pixel coords, or None on no detection within
    `max_probe_frames`.

    Pure detection — no margin, no squaring, no resizing. The caller
    decides what to do with the bbox (measure a face-fill ratio, derive a
    square crop, etc.).
    """
    T = video.shape[0]
    for t in range(min(max_probe_frames, T)):
        img_rgb = (video[t].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        faces = detect_fn(img_rgb[..., ::-1].copy())   # InsightFace expects BGR
        if not faces:
            continue
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        x1, y1, x2, y2 = face.bbox
        return (float(x1), float(y1), float(x2), float(y2))
    return None


def face_crop_video(
    video:             torch.Tensor,
    bbox_xyxy:         tuple[int, int, int, int],
    target_resolution: int = DEFAULT_RESOLUTION,
) -> torch.Tensor:
    """Apply a precomputed `(x1, y1, x2, y2)` bbox crop to every frame of
    `video` and bilinearly resize each crop to `target_resolution`.

    `video`: `(T, 3, H, W)` float32 in `[0, 1]`. Returns the same shape with
    `H == W == target_resolution`.
    """
    x1, y1, x2, y2 = bbox_xyxy
    cropped = video[:, :, y1:y2, x1:x2]
    return F.interpolate(
        cropped, size=(target_resolution, target_resolution),
        mode="bilinear", align_corners=False,
    )


def face_crop_around_detection(
    video:             torch.Tensor,
    detect_fn,
    margin:            float = DEFAULT_FACE_CROP_MARGIN,
    target_resolution: int   = DEFAULT_RESOLUTION,
    max_probe_frames:  int   = 10,
) -> torch.Tensor | None:
    """Detect a face on the first hit-frame of `video`, expand its bbox by
    `margin`, square it (face-centered), and crop+resize every frame to
    `target_resolution`.

    Returns None if face detection fails on the first `max_probe_frames`.

    The same routine is applied to **both** pred and GT in same-identity
    metric evaluation: each video's metrics are computed on a tight
    face-only square. This eliminates the framing mismatch between a
    tool's tool-specific pred crop and the manifest GT's wide crop, and
    matches what talking-head papers report as PSNR / SSIM / LPIPS on the
    face region.

    `margin` controls the padding around the raw RetinaFace bbox.
    RetinaFace returns roughly eyebrows-to-chin already, so 1.3× adds a
    small forehead and jaw cushion without pulling in hair or shoulders.
    """
    bbox = detect_face_bbox_xyxy(video, detect_fn, max_probe_frames)
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half   = max(x2 - x1, y2 - y1) / 2 * margin

    H, W = video.shape[-2], video.shape[-1]
    half = min(half, float(min(H, W)) / 2)   # crop can't exceed the frame

    # Center first; shift back inside the image if the square pokes out
    # (rather than clipping the side, which would change the face-fill
    # ratio between pred and GT).
    cx1, cy1, cx2, cy2 = cx - half, cy - half, cx + half, cy + half
    if cx1 < 0: cx2 -= cx1; cx1 = 0
    if cy1 < 0: cy2 -= cy1; cy1 = 0
    if cx2 > W: cx1 -= (cx2 - W); cx2 = W
    if cy2 > H: cy1 -= (cy2 - H); cy2 = H
    cx1, cy1 = max(0.0, cx1), max(0.0, cy1)
    cx2, cy2 = min(float(W), cx2), min(float(H), cy2)
    if (cx2 - cx1) < 1 or (cy2 - cy1) < 1:
        return None

    return face_crop_video(
        video,
        (int(round(cx1)), int(round(cy1)), int(round(cx2)), int(round(cy2))),
        target_resolution,
    )
