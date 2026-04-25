"""
Multi-GPU parallel FlowFace conversion: pixel3dmm tracking → fit.npz.

For each completed tracking directory, runs convert_to_flowface.py which:
  - Re-fits FLAME parameters from pixel3dmm meshes
  - Adds gaze tracking (L2CS) and background segmentation (RobustVideoMatting)
  - Produces fit.npz, reference_images.json, camera trajectories

Usage:
    cd <repo_root>

    # All pending videos, 4 GPUs:
    PYTHONPATH=. python generate_exp_map/scripts/run_parallel_flowface.py \
        --preprocessing_dir data/flame_tracking/preprocessing \
        --tracking_dir data/flame_tracking/tracking \
        --output_dir data/flowface

    # Specific GPUs:
    PYTHONPATH=. python generate_exp_map/scripts/run_parallel_flowface.py \
        --preprocessing_dir data/flame_tracking/preprocessing \
        --tracking_dir data/flame_tracking/tracking \
        --output_dir data/flowface \
        --gpus 0 1 2 3

    # Test on one video:
    PYTHONPATH=. python generate_exp_map/scripts/run_parallel_flowface.py \
        --preprocessing_dir data/flame_tracking/preprocessing \
        --tracking_dir data/flame_tracking/tracking \
        --output_dir data/flowface \
        --test
"""

import argparse
import os
import traceback
from pathlib import Path
from multiprocessing import Process, Queue
from tqdm import tqdm

TRACKING_SUFFIX = "_nV1_noPho_uv2000.0_n1000.0"


def tracking_name_to_video_name(tracking_dir_name):
    """Strip pixel3dmm tracking suffix to recover the original video name."""
    return tracking_dir_name.removesuffix(TRACKING_SUFFIX)


def find_pending(preprocessing_dir, tracking_dir, output_dir):
    """Return tracking dirs that still need FlowFace conversion."""
    tracking_dir = Path(tracking_dir)
    preprocessing_dir = Path(preprocessing_dir)
    output_dir = Path(output_dir)

    if not tracking_dir.exists():
        return []

    pending = []
    for d in sorted(tracking_dir.iterdir()):
        if not d.is_dir():
            continue
        vid_name = tracking_name_to_video_name(d.name)
        if (output_dir / vid_name / "fit.npz").exists():
            continue
        rgb_dir = preprocessing_dir / vid_name / "rgb"
        checkpoint = d / "checkpoint"
        if rgb_dir.exists() and checkpoint.exists() and any(checkpoint.iterdir()):
            pending.append(d)
    return pending


def worker(gpu_id, tracking_dirs, preprocessing_dir, output_dir, result_queue):
    """Worker: runs convert_to_flowface for each assigned video."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Must import after setting CUDA_VISIBLE_DEVICES
    from generate_exp_map.src.convert_to_flowface import main as convert_main

    for tracking_dir in tracking_dirs:
        vid_name = tracking_name_to_video_name(tracking_dir.name)
        rgb_dir = Path(preprocessing_dir) / vid_name / "rgb"
        preproc_dir = Path(preprocessing_dir) / vid_name
        out_dir = Path(output_dir) / vid_name

        try:
            out_dir.mkdir(parents=True, exist_ok=True)

            class ConvertArgs:
                video_path = str(rgb_dir)
                tracking_path = str(tracking_dir)
                preprocess_path = str(preproc_dir)
                output_path = str(out_dir)
                max_n_ref = 100
                enable_gaze_tracking = 1
                device = "cuda:0"

            convert_main(ConvertArgs())
            result_queue.put({"name": vid_name, "success": True})

        except Exception as e:
            result_queue.put({
                "name": vid_name, "success": False,
                "error": str(e), "tb": traceback.format_exc(),
            })


def parse_args():
    p = argparse.ArgumentParser(description="Parallel FlowFace conversion")
    p.add_argument("--preprocessing_dir", required=True,
                   help="pixel3dmm preprocessing output directory")
    p.add_argument("--tracking_dir", required=True,
                   help="pixel3dmm tracking output directory")
    p.add_argument("--output_dir", required=True,
                   help="FlowFace output directory (fit.npz)")
    p.add_argument("--gpus", type=int, nargs="+", default=list(range(8)))
    p.add_argument("--test", action="store_true")
    p.add_argument("--test_gpu", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    pending = find_pending(args.preprocessing_dir, args.tracking_dir, args.output_dir)

    if not pending:
        print("Nothing to convert.")
        return

    if args.test:
        vid_name = tracking_name_to_video_name(pending[0].name)
        print(f"[Test] Video: {vid_name}, GPU: {args.test_gpu}")
        q = Queue()
        worker(args.test_gpu, [pending[0]], args.preprocessing_dir, args.output_dir, q)
        result = q.get()
        if result["success"]:
            print(f"[Test] Success → {Path(args.output_dir) / vid_name}/")
        else:
            print(f"[Test] Failed: {result.get('error')}")
            if result.get("tb"):
                print(result["tb"])
        return

    print(f"FlowFace conversion: {len(pending)} videos across GPUs {args.gpus}")

    splits = [pending[i::len(args.gpus)] for i in range(len(args.gpus))]
    result_queue = Queue()

    procs = []
    for gpu_id, split in zip(args.gpus, splits):
        if not split:
            continue
        p = Process(target=worker,
                    args=(gpu_id, split, args.preprocessing_dir, args.output_dir, result_queue))
        p.start()
        procs.append(p)

    n_ok = n_fail = 0
    with tqdm(total=len(pending), desc="FlowFace") as pbar:
        done = 0
        while done < len(pending):
            alive = any(p.is_alive() for p in procs)
            try:
                result = result_queue.get(timeout=600 if alive else 2)
            except Exception:
                if not alive:
                    break
                continue
            done += 1
            if result["success"]:
                n_ok += 1
            else:
                n_fail += 1
                tqdm.write(f"  [FAIL] {result['name']}: {result.get('error', '?')}")
            pbar.update(1)
            pbar.set_postfix(ok=n_ok, fail=n_fail)

    for p in procs:
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()

    print(f"Done. Success: {n_ok} | Failed: {n_fail}")


if __name__ == "__main__":
    main()
