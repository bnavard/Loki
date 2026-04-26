"""
Filter input JSON to keep at most N segments per unique URL.

This maximizes URL coverage so the download script hits as many
different videos as possible before cycling back for more segments.

Usage:
    python filter_max_per_url.py --input input.json --output input_filtered.json --max-per-url 2
"""

import argparse
import json
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(
        description="Keep at most N segments per URL to maximize video coverage."
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to the full input JSON."
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to write the filtered JSON."
    )
    parser.add_argument(
        "--max-per-url",
        type=int,
        default=2,
        help="Maximum segments to keep per unique URL (default: 2).",
    )
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        items = json.load(f)

    url_counts = defaultdict(int)
    filtered = []

    for item in items:
        info = item.get("info", {})
        url = info.get("Video Link") or info.get("video_link")
        if not url:
            continue

        if url_counts[url] < args.max_per_url:
            filtered.append(item)
            url_counts[url] += 1

    unique_urls = len(url_counts)
    print(f"Input:  {len(items)} segments")
    print(f"Output: {len(filtered)} segments across {unique_urls} unique URLs")
    print(f"Max per URL: {args.max_per_url}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
