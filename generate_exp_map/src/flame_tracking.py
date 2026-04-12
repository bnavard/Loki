"""
Process a single video through the pixel3dmm FLAME tracking pipeline.

Runs four steps sequentially:
  1. Preprocessing: face cropping + MICA reconstruction + facer segmentation
  2. Normals prediction: pixel3dmm neural network
  3. UV map prediction: pixel3dmm neural network (different mode)
  4. FLAME tracking: fits FLAME parameters → fit.npz

The output fit.npz contains FLAME expression parameters used by all
downstream pipelines (expression field computation, Marigold training, etc.).

Requires the pixel3dmm package (patched) to be installed. Set environment:
  PIXEL3DMM_CODE_BASE         — path to pixel3dmm source
  PIXEL3DMM_PREPROCESSED_DATA — intermediate preprocessing output (default: data/flame_tracking/preprocessing)
  PIXEL3DMM_TRACKING_OUTPUT   — final tracking output (default: data/flame_tracking/tracking)

Usage:
    cd <repo_root>

    # Single video:
    python generate_exp_map/scripts/flame_tracking.py /path/to/video.mp4

    # Without log file:
    python generate_exp_map/scripts/flame_tracking.py /path/to/video.mp4 --no-log
"""

import os
import sys
from pathlib import Path

PIXEL3DMM_CODE_BASE = os.environ.get("PIXEL3DMM_CODE_BASE")
if PIXEL3DMM_CODE_BASE is None:
    raise EnvironmentError(
        "PIXEL3DMM_CODE_BASE not set. See generate_exp_map/README.md for setup."
    )

sys.path.insert(0, PIXEL3DMM_CODE_BASE)
sys.path.insert(0, os.path.join(PIXEL3DMM_CODE_BASE, "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for pixel3dmm_*.py in src/

# Note: PIPNet's FaceBoxesV2 is NOT added to sys.path to avoid
# conflicting with MICA's utils. Instead, faceboxes_detector.py
# uses relative imports (patched during setup).

from omegaconf import OmegaConf
from pixel3dmm import env_paths

from pixel3dmm_preprocessing import main as run_preprocessing_main
from pixel3dmm_inference import main as network_inference_main
from track import main as track_main

LOG_DIR = Path(os.environ.get("FLAME_LOG_DIR", "data/flame_tracking/logs/artifacts"))
COMPLETED_LOG = Path(os.environ.get("FLAME_COMPLETED_LOG", "data/flame_tracking/logs/completed.txt"))
FAILED_LOG = Path(os.environ.get("FLAME_FAILED_LOG", "data/flame_tracking/logs/failed.txt"))


def process_video(video_path: str, log_to_file: bool = True, gpu_id: str = "0"):
    """Process one video through the full pipeline. Returns True on success."""
    video_path = str(Path(video_path).resolve())
    vid_name = Path(video_path).stem

    if COMPLETED_LOG.exists():
        with open(COMPLETED_LOG, "r") as f:
            if video_path in {l.strip() for l in f if l.strip()}:
                print(f"Already completed: {vid_name}, skipping...")
                return True

    original_stdout, original_stderr = sys.stdout, sys.stderr
    log_fh = None

    if log_to_file:
        LOG_DIR.mkdir(exist_ok=True, parents=True)
        COMPLETED_LOG.parent.mkdir(exist_ok=True, parents=True)
        log_fh = open(LOG_DIR / f"{vid_name}_gpu{gpu_id}.log", "w", buffering=1)
        sys.stdout = log_fh
        sys.stderr = log_fh

    print(f"\n{'='*60}\nProcessing: {vid_name}\nGPU: {gpu_id}\n{'='*60}\n")

    try:
        print("[1/4] Preprocessing...")
        run_preprocessing_main(video_or_images_path=video_path)

        print("[2/4] Normals prediction...")
        base_conf = OmegaConf.load(f"{env_paths.CODE_BASE}/configs/base.yaml")
        cfg = OmegaConf.merge(base_conf, OmegaConf.create({
            "model": {"prediction_type": "normals"},
            "data": {"video_name": vid_name},
            "video_name": vid_name, "inference_batch_size": 4,
        }))
        network_inference_main(cfg)

        print("[3/4] UV map prediction...")
        base_conf = OmegaConf.load(f"{env_paths.CODE_BASE}/configs/base.yaml")
        cfg = OmegaConf.merge(base_conf, OmegaConf.create({
            "model": {"prediction_type": "uv_map"},
            "data": {"video_name": vid_name},
            "video_name": vid_name, "inference_batch_size": 4,
        }))
        network_inference_main(cfg)

        print("[4/4] FLAME tracking...")
        base_conf = OmegaConf.load(f"{env_paths.CODE_BASE}/configs/tracking.yaml")
        cfg = OmegaConf.merge(base_conf, OmegaConf.create({
            "data": {"video_name": vid_name},
            "video_name": vid_name,
        }))
        track_main(cfg)

        print(f"\nSuccessfully processed: {vid_name}\n")
        with open(COMPLETED_LOG, "a") as f:
            f.write(f"{video_path}\n")
        return True

    except Exception as e:
        print(f"\nFailed: {vid_name}: {e}\n")
        import traceback
        traceback.print_exc()
        with open(FAILED_LOG, "a") as f:
            f.write(f"{video_path}\n")
        return False

    finally:
        if log_to_file and log_fh:
            sys.stdout, sys.stderr = original_stdout, original_stderr
            log_fh.close()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("video_path")
    p.add_argument("--no-log", action="store_true")
    args = p.parse_args()

    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    success = process_video(args.video_path, log_to_file=not args.no_log, gpu_id=gpu_id)
    sys.exit(0 if success else 1)
