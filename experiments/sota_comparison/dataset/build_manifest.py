r"""
Build the curated benchmark manifest for a dataset.

Produces `experiments/sota_comparison/manifests/<dataset>.json` — one clip
per identity, capped at `--n_samples_cap` (default 1000), with a stable
`id_XXXX` UID attached to each entry. Downstream runners (SadTalker,
LivePortrait, Marionette eval, …) load this file and inherit the same
UID-to-identity mapping, so `outputs/**/samples/id_0457/` refers to the
same physical person across every baseline.

Manifests are committed to git and frozen. Re-running this tool on an
existing manifest refuses to overwrite unless `--rebuild` is passed.

Usage (from repo root):

    PYTHONPATH=. python experiments/sota_comparison/dataset/build_manifest.py \
        --dataset hdtf

    # Override the cap or seed:
    PYTHONPATH=. python experiments/sota_comparison/dataset/build_manifest.py \
        --dataset hdtf --n_samples_cap 500 --seed 0

    # Force overwrite an existing manifest:
    PYTHONPATH=. python experiments/sota_comparison/dataset/build_manifest.py \
        --dataset hdtf --rebuild
"""
from __future__ import annotations

import argparse
import sys

from experiments.sota_comparison.dataset.benchmark_manifest import (
    build_benchmark_manifest,
    manifest_path,
    save_manifest,
)
from experiments.sota_comparison.dataset.hdtf    import HDTFDataset
from experiments.sota_comparison.dataset.talkvid import TalkVidDataset


# Registry of dataset-name → zero-arg constructor. Add entries here when new
# dataset adapters land (celebvhq, voxceleb2).
_DATASETS = {
    "hdtf":    HDTFDataset,
    "talkvid": TalkVidDataset,
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Build a one-clip-per-identity benchmark manifest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset",       required=True, choices=sorted(_DATASETS))
    p.add_argument("--n_samples_cap", type=int, default=1000,
                   help="Max identities to keep; sampled uniformly under --seed "
                        "when the dataset has more than this.")
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--selection_rule", default="longest_clip_per_identity",
                   choices=["longest_clip_per_identity"])
    p.add_argument("--rebuild", action="store_true",
                   help="Overwrite an existing manifest.")
    return p.parse_args()


def main():
    args = parse_args()
    out_path = manifest_path(args.dataset)

    if out_path.exists() and not args.rebuild:
        print(
            f"Manifest already exists at {out_path}.\n"
            f"Pass --rebuild to overwrite. Aborting so existing UIDs stay stable.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    ds = _DATASETS[args.dataset]()
    clips, meta = build_benchmark_manifest(
        dataset         = ds,
        n_samples_cap   = args.n_samples_cap,
        seed            = args.seed,
        selection_rule  = args.selection_rule,
    )
    save_manifest(out_path, clips, meta)

    print(f"[{args.dataset}] wrote manifest → {out_path}")
    print(f"  total identities in dataset: {meta['n_total_identities']}")
    print(f"  identities kept:             {meta['n_identities']} "
          f"(cap={meta['n_samples_cap']}, seed={meta['seed']})")
    print(f"  selection rule:              {meta['selection_rule']}")
    if clips:
        print(f"  first:  uid={clips[0].uid}  identity={clips[0].identity_id}  "
              f"clip={clips[0].clip_id}  n_frames={clips[0].n_frames}")
        print(f"  last:   uid={clips[-1].uid}  identity={clips[-1].identity_id}  "
              f"clip={clips[-1].clip_id}  n_frames={clips[-1].n_frames}")


if __name__ == "__main__":
    main()
