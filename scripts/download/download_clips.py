"""
Download YouTube video clips from the TalkVid dataset.

Reads a JSON file of (URL, start, end) segments and downloads them using
yt-dlp with multi-section support. Includes rate-limit detection with
exponential backoff, batch cooldowns, and resume via per-clip JSON logs.

Usage:
    cd <repo_root>

    # Download from the filtered dataset:
    python scripts/download/download_clips.py \
        --input scripts/download/talkvid_data.json \
        --output data/talkvid/talkvid

    # With browser cookies for auth (recommended to avoid rate limits):
    python scripts/download/download_clips.py \
        --input scripts/download/talkvid_data.json \
        --output data/talkvid/talkvid \
        --browser chrome

    # Test with a small batch:
    python scripts/download/download_clips.py \
        --input scripts/download/talkvid_data.json \
        --output data/talkvid/talkvid \
        --limit 10
"""

import argparse
import functools
import os
import random
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, Generator, List, Optional, Tuple
import json
import math
import glob
from pathlib import Path
from rich.progress import (
    Progress,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TextColumn,
)  # type: ignore




def get_video_id(url: str) -> str:
    """Extract video ID from a YouTube URL."""
    if 'watch?v=' in url:
        return url.split('watch?v=')[-1].split('&')[0]
    if 'youtu.be/' in url:
        return url.split('youtu.be/')[-1].split('?')[0]
    if '/shorts/' in url:
        return url.split('/shorts/')[-1].split('?')[0]
    # Fallback for other URL formats or just return a hash
    # For simplicity, we'll just use the last part of the URL
    return url.split('/')[-1] or "unknown_id"


def find_executable(candidates: List[str]) -> Optional[str]:
    """Return the first existing candidate executable path or None."""
    for candidate in candidates:
        if shutil.which(candidate):
            return candidate
        # Also consider a relative executable within the current directory
        if os.path.isfile(candidate):
            return candidate
    return None


def safe_mkdir(path: str) -> None:
    if not path:
        return
    os.makedirs(path, exist_ok=True)


def is_clip_downloaded(
    url: str, start: float, end: float, output_dir: str
) -> bool:
    """Check if a clip was already downloaded by inspecting its JSON log."""
    video_id = get_video_id(url)
    # Normalize float format in filename
    log_filename = f"{video_id}_{start:.3f}_{end:.3f}.json".replace(":", "-")
    log_file = os.path.join(output_dir, "json_logs", log_filename)

    if not os.path.exists(log_file):
        return False

    try:
        with open(log_file, "r", encoding="utf-8") as f:
            log_data = json.load(f)
        if log_data.get("download_info", {}).get("status") == "success":
            # Verify the video file actually exists on disk
            video_file = log_data.get("download_info", {}).get("video_clip_file")
            if video_file and os.path.exists(video_file) and os.path.getsize(video_file) > 0:
                return True
    except (json.JSONDecodeError, KeyError):
        return False
    return False


def iter_segments_from_big_json(
    input_json_path: str,
) -> Generator[Tuple[str, float, float], None, None]:
    """
    Load and iterate over a JSON file of video segments.

    Each item (dict) should contain:
        - "Video Link" (or lowercase variant) inside an "info" sub-dict
        - "start-time" / "start"
        - "end-time" / "end"

    Yields (url, start, end) tuples.
    """

    with open(input_json_path, "r", encoding="utf-8") as f:
        try:
            items = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse JSON: {input_json_path} | {exc}") from exc

    if not isinstance(items, list):
        raise ValueError("Expected top-level JSON to be an array (list)")

    for item in items:
        if not isinstance(item, dict):
            continue

        info_dict = item.get("info", {})
        url = info_dict.get("Video Link") or info_dict.get("video_link")
        start_val = item.get("start-time") or item.get("start")
        end_val = item.get("end-time") or item.get("end")

        if url is None or start_val is None or end_val is None:
            continue

        try:
            start_f = float(start_val)
            end_f = float(end_val)
        except (TypeError, ValueError):
            continue

        if end_f > start_f:
            yield (url, start_f, end_f)


def seconds_to_time_string(seconds_value: float) -> str:
    if seconds_value < 0:
        seconds_value = 0.0
    ms = math.floor(seconds_value * 1000 + 1e-6)
    hours, rem = divmod(ms, 3600_000)
    minutes, ms = divmod(rem, 60_000)
    seconds, ms = divmod(ms, 1000)
    if ms == 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{ms:03d}"

def get_yt_dlp_base_cmd(cookies_path: Optional[str], browser: Optional[str]) -> Tuple[Optional[List[str]], Optional[str]]:
    """Build base yt-dlp command, preferring python -m yt_dlp. Returns (cmd, error)."""
    try:
        import importlib.util  # noqa: F401

        if importlib.util.find_spec("yt_dlp") is not None:
            base_cmd = [sys.executable, "-m", "yt_dlp"]
        else:
            raise ImportError
    except Exception:
        yt_dlp_candidates = [
            os.path.join(os.getcwd(), "yt-dlp_x86.exe"),
            os.path.join(os.getcwd(), "yt-dlp.exe"),
            "yt-dlp",
        ]
        yt_dlp_path = find_executable(yt_dlp_candidates)
        if yt_dlp_path is None:
            return None, "yt-dlp not found. Install with: python -m pip install --user -U yt-dlp"
        base_cmd = [yt_dlp_path]

    # Cookies are added by callers depending on the operation (probe/download)
    if browser:
        base_cmd = [*base_cmd, "--cookies-from-browser", browser]
    elif cookies_path and os.path.exists(cookies_path):
        base_cmd = [*base_cmd, "--cookies", cookies_path]

    # Use IPv4 and ignore user config files for reproducibility
    base_cmd = [*base_cmd, "-4", "--ignore-config"]
    return base_cmd, None


RATE_LIMIT_PHRASES = [
    "http error 429",
    "too many requests",
    "rate limit",
    "sign in to confirm",
    "confirm you're not a bot",
]


def is_rate_limited(error_msg: str) -> bool:
    """Check if an error message indicates YouTube rate limiting."""
    lower = error_msg.lower()
    return any(phrase in lower for phrase in RATE_LIMIT_PHRASES)


def sleep_with_jitter(base_seconds: float, jitter: float) -> None:
    """Sleep for base_seconds +/- jitter (random uniform)."""
    actual = max(0.0, base_seconds + random.uniform(-jitter, jitter))
    if actual > 0:
        time.sleep(actual)


def run_yt_dlp_multi_sections(
    url: str,
    segments: List[Tuple[float, float]],
    output_dir: str,
    cookies_path: Optional[str] = None,
    browser: Optional[str] = None,
    extractor_args: Optional[str] = None,
    strict_cuts: bool = False,
) -> Tuple[int, str]:
    """
    Download multiple segments from the same URL in a single yt-dlp call
    using multiple --download-sections flags.
    """
    safe_mkdir(output_dir)
    base_cmd, err = get_yt_dlp_base_cmd(cookies_path, browser)
    if base_cmd is None:
        return 1, err or "Unable to locate yt-dlp"

    # --- Build download command ---
    video_id = get_video_id(url)
    video_output_dir = os.path.join(output_dir, video_id)
    safe_mkdir(video_output_dir)

    # Build multi-section args
    section_args: List[str] = []
    for (s, e) in segments:
        if e <= s:
            continue
        s_str = seconds_to_time_string(s)
        e_str = seconds_to_time_string(e)
        section_args.extend(["--download-sections", f"*{s_str}-{e_str}"])

    if not section_args:
        return 1, "No valid segments for this URL"

    # Output template: files go into a subdirectory named by video_id
    output_template = os.path.join(
        video_output_dir,
        "%(id)s_%(section_number)03d_%(section_start).3f_%(section_end).3f.%(ext)s",
    )

    cmd: List[str] = [
        *base_cmd,
        "-4",
        "--ignore-config",
        "--no-playlist",
        "--retries", "10",
        "--fragment-retries", "10",
        "--concurrent-fragments", "8",
        "-N", "4",
        "--no-warnings",
        "--restrict-filenames",
        "--no-continue", "--no-overwrites",
        "--print", "after_move:filepath",
        "--write-subs",
        "--write-auto-subs",
        "--write-description",
        "--extract-audio",
        "--audio-format", "m4a", "--audio-quality", "0",
        "--keep-video",
        "--no-keep-fragments",
        "--clean-info-json",
        "-o", output_template,
        # Prefer H.264+AAC (lossless remux); fall back to best
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4", 
    ]
    if strict_cuts:
        cmd.append("--force-keyframes-at-cuts")

    if extractor_args:
        cmd.extend(["--extractor-args", extractor_args])

    # Append section args
    cmd.extend(section_args)
    cmd.append(url)

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        if proc.returncode == 0:
            return 0, proc.stdout.strip()
        # Fallback: if requested format unavailable, try best
        err_msg = (proc.stderr.strip() or proc.stdout.strip())
        if "Requested format is not available" in err_msg:
            fallback_cmd = [
                *base_cmd,
                "-4", "--ignore-config", "--no-playlist",
                "--retries", "10", "--fragment-retries", "10",
                "--concurrent-fragments", "8", "-N", "4",
                "--no-warnings", "--restrict-filenames",
                "-c", "--no-overwrites",
                # --- Fallback features ---
                "--print", "after_move:filepath",
                "--write-subs", "--write-auto-subs", "--write-description",
                "--extract-audio", "--audio-format", "m4a", "--keep-video",
                # --- Output template (fallback) ---
                "-o", output_template,
                "-f", "bestvideo[ext=mp4][vcodec!=none]+bestaudio[ext=m4a]/best[ext=mp4][vcodec!=none]",
                "--remux-video", "mp4",
            ]
            if strict_cuts:
                fallback_cmd.append("--force-keyframes-at-cuts")
            if extractor_args:
                fallback_cmd.extend(["--extractor-args", extractor_args])
            fallback_cmd.extend(section_args)
            fallback_cmd.append(url)

            proc2 = subprocess.run(fallback_cmd, capture_output=True, text=True, encoding='utf-8')
            if proc2.returncode == 0:
                return 0, proc2.stdout.strip()
            return proc2.returncode, (proc2.stderr.strip() or proc2.stdout.strip())
        return proc.returncode, err_msg
    except Exception as exc:  # noqa: BLE001
        return 1, f"yt-dlp failed: {exc}"


def probe_url_availability(
    url: str,
    cookies_path: Optional[str],
    browser: Optional[str],
    extractor_args: Optional[str] = None,
) -> Tuple[bool, str]:
    """Check if a URL is available by asking yt-dlp to print the id without downloading."""
    base_cmd, err = get_yt_dlp_base_cmd(cookies_path, browser)
    if base_cmd is None:
        return False, err or "yt-dlp not found"

    cmd: List[str] = [
        *base_cmd,
        "-s",
        "--no-warnings",
        "-O",
        "%(id)s",
        url,
    ]
    if extractor_args:
        cmd.extend(["--extractor-args", extractor_args])
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0 and proc.stdout.strip():
            return True, proc.stdout.strip()
        # Collect an error message
        msg = (proc.stderr or proc.stdout or "unknown error").strip()
        return False, msg
    except Exception as exc:  # noqa: BLE001
        return False, f"probe failed: {exc}"


# ---- Parse download output ------------------------------------------------------------


def _match_segment_from_name(name: str) -> Optional[Tuple[float, float]]:
    """Extract segment (start, end) from filename. Returns None if no match."""
    m = re.search(r"_(\d+\.\d+)_(\d+\.\d+)\.", name)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None


def parse_ytdlp_output(
    output: str,
    segments: List[Tuple[float, float]],
    video_id: str,
    video_output_dir: str,
) -> Dict[Tuple[float, float], Dict]:
    """
    Parse yt-dlp output and associate file paths with original segments.
    Falls back to scanning the output directory for audio/description/subtitles.
    """

    # Extract file paths from stdout
    files_from_stdout = [line.strip() for line in output.splitlines() if line.strip()]

    # Classification containers
    description_file: str = ""
    subtitle_files: List[str] = []
    clip_files: Dict[Tuple[float, float], List[str]] = defaultdict(list)

    for raw in files_from_stdout:
        # Strip prefix markers like "[info] Writing video description to: "
        possible_path = raw.split(": ")[-1].strip()
        path = Path(possible_path)
        if not path.exists():
            # Skip if not a real file path
            continue

        if path.name.endswith(".description"):
            description_file = str(path)
        elif path.suffix.lower() in {".vtt", ".srt", ".ass"}:
            subtitle_files.append(str(path))
        elif path.suffix.lower() in {".mp4", ".m4a", ".webm", ".mkv"}:
            seg_match = _match_segment_from_name(path.name)
            if seg_match:
                closest_seg = min(
                    segments,
                    key=lambda s: abs(s[0]-seg_match[0])+abs(s[1]-seg_match[1])
                )
                clip_files[closest_seg].append(str(path))

    # --- Fallback: scan output directory for missing files ---
    try:
        for file_name in os.listdir(video_output_dir):
            file_path = os.path.join(video_output_dir, file_name)
            path = Path(file_path)
            if path.name.endswith(".description") and not description_file:
                description_file = file_path
            elif path.suffix.lower() in {".vtt", ".srt", ".ass"} and file_path not in subtitle_files:
                subtitle_files.append(file_path)
            elif path.suffix.lower() in {".mp4", ".m4a", ".webm", ".mkv"}:
                seg_match = _match_segment_from_name(path.name)
                if seg_match:
                    closest_seg = min(
                        segments,
                        key=lambda s: abs(s[0]-seg_match[0])+abs(s[1]-seg_match[1])
                    )
                    if file_path not in clip_files[closest_seg]:
                        clip_files[closest_seg].append(file_path)
    except FileNotFoundError:
        pass

    # Assemble final results
    results: Dict[Tuple[float, float], Dict] = {}
    for seg in segments:
        file_list = clip_files.get(seg, [])
        video_file = next((f for f in file_list if f.endswith((".mp4", ".mkv", ".webm"))), "")
        audio_file = next((f for f in file_list if f.endswith(".m4a")), "")

        results[seg] = {
            "video_clip_file": video_file,
            "audio_clip_file": audio_file,
            "description_file": description_file,
            "subtitle_files": subtitle_files,
        }

    return results


def load_unavailable_urls(log_file_path: str) -> set:
    """Load URLs with permanent errors (unavailable, private, etc.) from log file."""
    unavailable_urls = set()
    if not os.path.exists(log_file_path):
        return unavailable_urls

    # Common permanent error phrases (lowercase)
    permanent_error_phrases = [
        "video unavailable",
        "account associated with this video has been terminated",
        "private video",
        "video is private",
        "user has closed their youtube account",
    ]

    try:
        with open(log_file_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    url, reason = parts[0], parts[1].lower()
                    if any(phrase in reason for phrase in permanent_error_phrases):
                        unavailable_urls.add(url)
    except Exception as e:
        print(f"Warning: Could not read or parse unavailable URLs log: {e}")

    return unavailable_urls


def _write_json_log(json_logs_dir: str, video_id: str, start: float, end: float,
                    url: str, files_info: Dict) -> None:
    """Write a json log immediately after a successful download."""
    log_filename = f"{video_id}_{start:.3f}_{end:.3f}.json".replace(":", "-")
    log_file = os.path.join(json_logs_dir, log_filename)
    log_data = {
        "source_info": {
            "url": url,
            "start_time": start,
            "end_time": end,
        },
        "download_info": {
            **files_info,
            "status": "success",
            "error": "",
            "download_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
    }
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)


def download_with_ytdlp(
    input_json_path: str,
    output_dir: str,
    cookies_path: Optional[str],
    browser: Optional[str],
    extractor_args: Optional[str],
    limit: Optional[int],
    workers: int,
    delay: float = 5.0,
    jitter: float = 3.0,
    batch_size: int = 50,
    cooldown: float = 120.0,
    max_backoff: float = 600.0,
) -> None:
    """
    Download clips with rate-limit resilience.

    Key behaviors:
        - Processes URLs sequentially (with optional workers for segments within a URL)
        - Writes json logs immediately after each URL completes (survives interrupts)
        - Delays between URLs with random jitter to avoid rate limiting
        - Pauses for cooldown after every batch_size URLs
        - Detects rate limiting and applies exponential backoff
        - Skips permanently unavailable URLs from prior runs
    """

    url2segments: Dict[str, List[Tuple[float, float]]] = defaultdict(list)

    count = 0
    total_segments = 0
    skipped_due_to_log = 0

    json_logs_dir = os.path.join(output_dir, "json_logs")
    safe_mkdir(json_logs_dir)

    for (url, start, end) in iter_segments_from_big_json(input_json_path):
        total_segments += 1
        if limit is not None and limit >= 0 and count >= limit:
            break

        if is_clip_downloaded(url, start, end, output_dir):
            skipped_due_to_log += 1
            continue

        if end > start:
            url2segments[url].append((start, end))
            count += 1

    print(f"Total segments found: {total_segments}")
    print(f"Skipped (already downloaded): {skipped_due_to_log}")
    print(f"Segments to download: {count}")

    if not url2segments:
        print("No new segments to download.")
        return

    safe_mkdir(output_dir)
    logs_dir = os.path.join(output_dir, "logs")
    safe_mkdir(logs_dir)
    failed_urls_file = os.path.join(logs_dir, "failed_urls.txt")
    failed_segments_file = os.path.join(logs_dir, "failed_segments.txt")

    unavailable_urls = load_unavailable_urls(failed_urls_file)
    if unavailable_urls:
        print(f"Loaded {len(unavailable_urls)} permanently unavailable URLs from logs.")

    progress = Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )

    progress.__enter__()
    task_urls = progress.add_task("URLs", total=len(url2segments))
    task_segments = progress.add_task("Segments", total=count)

    backoff = 0.0
    urls_since_cooldown = 0
    url_items = list(url2segments.items())
    total_urls = len(url_items)
    success_count = 0
    fail_count = 0
    skip_count = 0

    print(f"[config] delay={delay}s, jitter={jitter}s, batch_size={batch_size}, "
          f"cooldown={cooldown}s, max_backoff={max_backoff}s, workers={workers}")
    print(f"[config] cookies={'yes' if cookies_path else 'no'}, "
          f"browser={browser or 'none'}")
    print(f"[start] Processing {total_urls} URLs with {count} segments...\n")

    for url_idx, (url, segs) in enumerate(url_items, 1):
        video_id = get_video_id(url)
        ts = datetime.now().strftime('%H:%M:%S')

        # --- Batch cooldown ---
        if batch_size > 0 and urls_since_cooldown >= batch_size:
            print(f"\n[{ts}][cooldown] Batch of {batch_size} URLs done. "
                  f"Sleeping {cooldown:.0f}s before next batch... "
                  f"(success={success_count}, fail={fail_count}, skip={skip_count})")
            time.sleep(cooldown)
            urls_since_cooldown = 0
            backoff = 0.0

        # --- Exponential backoff on rate limit ---
        if backoff > 0:
            print(f"[{ts}][backoff] Rate limit detected. Sleeping {backoff:.0f}s...")
            time.sleep(backoff)

        # --- Per-URL delay with jitter ---
        sleep_with_jitter(delay, jitter)

        # Skip permanently unavailable URLs
        if url in unavailable_urls:
            reason = "Skipping probe: URL previously marked as permanently unavailable."
            print(f"[{ts}][{url_idx}/{total_urls}] SKIP {video_id} -- permanently unavailable")
            with open(failed_segments_file, "a", encoding="utf-8") as fseg:
                for (s, e) in segs:
                    fseg.write(f"SKIP\t{url}\t{s:.3f}\t{e:.3f}\t{reason}\n")
            progress.update(task_urls, advance=1)
            progress.update(task_segments, advance=len(segs))
            skip_count += 1
            continue

        print(f"[{ts}][{url_idx}/{total_urls}] Probing {video_id} ({len(segs)} segments)...", end=" ", flush=True)
        ok, reason = probe_url_availability(url, cookies_path, browser, extractor_args)
        if not ok:
            print(f"UNAVAILABLE: {reason[:80]}")
            with open(failed_urls_file, "a", encoding="utf-8") as furl:
                furl.write(f"{url}\t{reason}\n")
            for (s, e) in segs:
                with open(failed_segments_file, "a", encoding="utf-8") as fseg:
                    fseg.write(f"SKIP\t{url}\t{s:.3f}\t{e:.3f}\t{reason}\n")
            progress.update(task_urls, advance=1)
            progress.update(task_segments, advance=len(segs))
            skip_count += 1

            if is_rate_limited(reason):
                backoff = min(max_backoff, max(60.0, backoff * 2 if backoff else 60.0))
                print(f"[{ts}][backoff] Rate limit on probe. Will sleep {backoff:.0f}s before next URL.")
            continue

        print("OK. Downloading...", end=" ", flush=True)

        # Successful probe -- reset backoff
        backoff = 0.0

        segs = sorted(set(segs))

        # Download segments for this URL
        rc, msg = run_yt_dlp_multi_sections(
            url, segs, output_dir, cookies_path, browser, extractor_args, True
        )

        if rc != 0:
            print(f"FAILED: {msg[:100]}")
            with open(failed_segments_file, "a", encoding="utf-8") as fseg:
                for (s, e) in segs:
                    fseg.write(f"FAIL\t{url}\t{s:.3f}\t{e:.3f}\t{msg}\n")
            fail_count += 1

            if is_rate_limited(msg):
                backoff = min(max_backoff, max(60.0, backoff * 2 if backoff else 60.0))
                print(f"[{ts}][backoff] Rate limit on download. Will sleep {backoff:.0f}s before next URL.")
        else:
            # Immediately parse and write json logs (survives interrupts)
            parsed_results = parse_ytdlp_output(
                msg, segs, video_id, os.path.join(output_dir, video_id)
            )
            for seg_tuple, files_info in parsed_results.items():
                start, end = seg_tuple
                _write_json_log(json_logs_dir, video_id, start, end, url, files_info)

            succeeded_segs = set(parsed_results.keys())
            failed_segs = [s for s in segs if s not in succeeded_segs]
            if failed_segs:
                with open(failed_segments_file, "a", encoding="utf-8") as fseg:
                    for (s, e) in failed_segs:
                        fseg.write(f"FAIL\t{url}\t{s:.3f}\t{e:.3f}\tNo output file generated\n")

            print(f"SUCCESS ({len(succeeded_segs)}/{len(segs)} segments)")
            success_count += 1

        progress.update(task_urls, advance=1)
        progress.update(task_segments, advance=len(segs))
        urls_since_cooldown += 1

    progress.__exit__(None, None, None)
    print(f"\n[done] URLs: {success_count} success, {fail_count} failed, {skip_count} skipped")


def ensure_ffmpeg() -> None:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found. Install it first (required by yt-dlp for segment cutting).")


def cleanup_final_files(output_dir: str) -> None:
    """Clean up files not recorded in json_logs to keep the output directory tidy."""
    print("\nStarting final file cleanup...")
    json_logs_dir = os.path.join(output_dir, "json_logs")
    if not os.path.isdir(json_logs_dir):
        print("json_logs directory not found, skipping cleanup.")
        return

    # 1. Collect all file paths recorded in logs
    recorded_files = set()
    json_files = glob.glob(os.path.join(json_logs_dir, "*.json"))
    for log_file in json_files:
        recorded_files.add(os.path.abspath(log_file))  # whitelist the log file itself
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            info = data.get("download_info", {})
            if info.get("status") != "success":
                continue

            for key, value in info.items():
                if key.endswith("_file") and isinstance(value, str) and value:
                    recorded_files.add(os.path.abspath(value))
                elif key.endswith("_files") and isinstance(value, list):
                    for item in value:
                        if isinstance(item, str) and item:
                            recorded_files.add(os.path.abspath(item))
        except Exception as e:
            print(f"Error processing log file {log_file}: {e}")

    if not recorded_files:
        print("No recorded files found in logs, skipping cleanup.")
        return

    # 2. Walk output directory, delete unrecorded files
    print(f"Found {len(recorded_files)} files recorded in logs. Scanning for unrecorded files...")
    deleted_count = 0
    for root, _, files in os.walk(output_dir):
        for file in files:
            # Skip failure log files
            if "logs" in root and ("failed_urls.txt" in file or "failed_segments.txt" in file):
                continue

            file_path = os.path.abspath(os.path.join(root, file))
            if file_path not in recorded_files:
                try:
                    os.remove(file_path)
                    deleted_count += 1
                    print(f"Deleted unrecorded file: {file_path}")
                except OSError as e:
                    print(f"Failed to delete {file_path}: {e}")
    print(f"Cleanup complete. Deleted {deleted_count} unrecorded files.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download clips specified in a large JSON file directly with yt-dlp sections on Windows."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        default=os.path.join(os.getcwd(), "filtered_video_clips.json"),
        help="Path to the large JSON file containing 'Video Link', 'start-time', 'end-time' fields.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(os.getcwd(), "clips_output"),
        help="Directory to store the downloaded clips.",
    )
    parser.add_argument(
        "--mode",
        choices=["ytdlp"],
        default="ytdlp",
        help="Download mode (only ytdlp is supported).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many clip segments (for testing).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent workers for yt-dlp mode (default: 1 to avoid rate limits).",
    )
    parser.add_argument(
        "--cookies",
        type=str,
        default=os.path.join(os.getcwd(), "cookies.txt"),
        help="Path to cookies.txt to pass to yt-dlp if present.",
    )
    parser.add_argument(
        "--browser",
        type=str,
        choices=["edge", "chrome", "firefox", "chromium", "brave", "vivaldi", "opera"],
        default=None,
        help="Use --cookies-from-browser <browser> for YouTube auth (recommended).",
    )
    parser.add_argument(
        "--extractor_args",
        type=str,
        default=None,
        help="Pass through to yt-dlp --extractor-args, e.g. 'youtube:player_client=android'",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Run cleanup process after downloading to remove unlogged files.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=5.0,
        help="Base seconds to wait between URL downloads (default: 5).",
    )
    parser.add_argument(
        "--jitter",
        type=float,
        default=3.0,
        help="Random +/- seconds added to delay (default: 3).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Pause for --cooldown seconds after this many URLs (default: 50, 0=disabled).",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=120.0,
        help="Seconds to sleep between batches (default: 120).",
    )
    parser.add_argument(
        "--max-backoff",
        type=float,
        default=600.0,
        help="Max backoff seconds on rate-limit detection (default: 600).",
    )
    return parser.parse_args()


def main() -> None:
    ensure_ffmpeg()
    args = parse_args()
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Start downloading with yt-dlp")

    download_with_ytdlp(
        input_json_path=args.input,
        output_dir=args.output,
        cookies_path=args.cookies if os.path.exists(args.cookies) else None,
        browser=args.browser,
        extractor_args=args.extractor_args,
        limit=args.limit,
        workers=args.workers,
        delay=args.delay,
        jitter=args.jitter,
        batch_size=args.batch_size,
        cooldown=args.cooldown,
        max_backoff=args.max_backoff,
    )
    print("yt-dlp clip downloads completed.")

    if args.cleanup:
        cleanup_final_files(args.output)


if __name__ == "__main__":
    # Force unbuffered stdout so logs appear immediately in terminals/tmux
    print = functools.partial(print, flush=True)  # type: ignore[assignment]
    main()


