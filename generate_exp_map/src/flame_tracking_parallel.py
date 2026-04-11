"""
Multi-GPU parallel scheduler for FLAME tracking.

Distributes videos across GPUs with multiple workers per GPU. Each worker
loads the pixel3dmm models once and processes videos sequentially.

Features:
  - Resume from completed/failed logs
  - Round-robin GPU assignment
  - Per-video log files
  - Progress tracking with periodic status banners

Usage:
    cd <repo_root>

    # Process all mp4 files in the data directory:
    python generate_exp_map/scripts/flame_tracking_parallel.py \
        --data_dirs data/talkvid/talkvid \
        --num_gpus 8 \
        --workers_per_gpu 2
"""

import argparse
import os
import sys
import time
from pathlib import Path
from multiprocessing import Process, Queue, Event
import traceback

PIXEL3DMM_CODE_BASE = os.environ.get("PIXEL3DMM_CODE_BASE")
if PIXEL3DMM_CODE_BASE is None:
    raise EnvironmentError(
        "PIXEL3DMM_CODE_BASE not set. See generate_exp_map/README.md for setup."
    )

os.environ.setdefault("PIXEL3DMM_PREPROCESSED_DATA", "data/flame_tracking/preprocessing")
os.environ.setdefault("PIXEL3DMM_TRACKING_OUTPUT", "data/flame_tracking/tracking")

sys.path.insert(0, PIXEL3DMM_CODE_BASE)
sys.path.insert(0, os.path.join(PIXEL3DMM_CODE_BASE, "scripts"))

LOG_DIR = Path(os.environ.get("FLAME_LOG_DIR", "data/flame_tracking/logs/artifacts"))
COMPLETED_LOG = Path(os.environ.get("FLAME_COMPLETED_LOG", "data/flame_tracking/logs/completed.txt"))
FAILED_LOG = Path(os.environ.get("FLAME_FAILED_LOG", "data/flame_tracking/logs/failed.txt"))


def worker_process(gpu_id, task_queue, result_queue, stop_event):
    """Worker: processes videos on a specific GPU. Models loaded once, reused."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import datetime
    def ts():
        return datetime.datetime.now().strftime("%H:%M:%S")

    from generate_exp_map.src.flame_tracking import process_video

    print(f"[{ts()}] [GPU {gpu_id}] Worker ready", flush=True)

    while not stop_event.is_set():
        try:
            video_path = task_queue.get(timeout=1.0)
        except Exception:
            continue

        if video_path is None:
            break

        video_name = Path(video_path).name
        print(f"[{ts()}] [GPU {gpu_id}] Starting {video_name}", flush=True)

        start_time = time.time()
        try:
            success = process_video(video_path, gpu_id=str(gpu_id))
            duration = time.time() - start_time
            result_queue.put({
                "gpu_id": gpu_id, "video_path": video_path,
                "success": success, "duration": duration,
            })
        except Exception as e:
            result_queue.put({
                "gpu_id": gpu_id, "video_path": video_path,
                "success": False, "duration": time.time() - start_time,
                "error": str(e),
            })
            traceback.print_exc()

    print(f"[{ts()}] [GPU {gpu_id}] Worker stopped", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dirs", nargs="+", required=True,
                   help="Directories containing mp4 files to process")
    p.add_argument("--num_gpus", type=int, default=8)
    p.add_argument("--workers_per_gpu", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()
    LOG_DIR.mkdir(exist_ok=True, parents=True)

    # Collect videos
    mp4_files = []
    for data_path in args.data_dirs:
        data_dir = Path(data_path)
        if data_dir.exists():
            mp4_files.extend(data_dir.rglob("*.mp4"))
        else:
            print(f"Warning: {data_path} not found")
    mp4_files = sorted(mp4_files)

    def resolve(p):
        return str(Path(p).resolve())

    # Load completed + failed for resume
    skip = set()
    for log_path in [COMPLETED_LOG, FAILED_LOG]:
        if log_path.exists():
            with open(log_path) as f:
                skip |= {resolve(l.strip()) for l in f if l.strip()}

    mp4_files = [f for f in mp4_files if resolve(f) not in skip]
    total = len(mp4_files)

    print(f"Found {total} videos to process ({len(skip)} already done/failed)")
    print(f"Using {args.num_gpus} GPUs x {args.workers_per_gpu} workers = "
          f"{args.num_gpus * args.workers_per_gpu} concurrent jobs\n")

    if total == 0:
        print("Nothing to do.")
        return

    # Create queues + workers
    task_queues = [Queue() for _ in range(args.num_gpus)]
    result_queue = Queue()
    stop_event = Event()

    workers = []
    for gpu_id in range(args.num_gpus):
        for _ in range(args.workers_per_gpu):
            p = Process(target=worker_process,
                        args=(gpu_id, task_queues[gpu_id], result_queue, stop_event))
            p.start()
            workers.append(p)

    # Distribute round-robin
    for i, video_path in enumerate(mp4_files):
        task_queues[i % args.num_gpus].put(str(video_path))

    # Poison pills
    for q in task_queues:
        for _ in range(args.workers_per_gpu):
            q.put(None)

    # Monitor results
    import datetime
    def ts():
        return datetime.datetime.now().strftime("%H:%M:%S")

    completed_count = failed_count = 0
    processed = 0
    last_status = time.time()

    while processed < total:
        try:
            result = result_queue.get(timeout=5.0)
            processed += 1

            vid = Path(result["video_path"]).name
            gpu = result["gpu_id"]
            dur = result.get("duration", 0)

            if result["success"]:
                completed_count += 1
                print(f"[{ts()}] [GPU {gpu}] OK {vid} ({dur:.1f}s)")
            else:
                failed_count += 1
                err = result.get("error", "")
                print(f"[{ts()}] [GPU {gpu}] FAIL {vid} ({dur:.1f}s) {err[:80]}")

            print(f"  progress: {processed}/{total} "
                  f"({100*processed/total:.1f}%) "
                  f"ok={completed_count} fail={failed_count} "
                  f"remaining={total-processed}")

        except Exception:
            pass

        if time.time() - last_status >= 60:
            print(f"\n[{ts()}] STATUS: {processed}/{total} "
                  f"({completed_count} ok, {failed_count} fail, {total-processed} remaining)\n")
            last_status = time.time()

    # Cleanup
    stop_event.set()
    for p in workers:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()

    print(f"\nDone. {completed_count}/{total} succeeded, {failed_count} failed.")
    if failed_count > 0:
        print(f"Failed videos: {FAILED_LOG}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
