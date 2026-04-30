"""Quantitative evaluation metrics for talking-head video generation.

The package is organized one metric per module so each can be imported and
tested in isolation. The unified entry point is `evaluator.evaluate(run_dir)`.

Conventions (every metric in this package follows them):
  - Video tensors are `(B, T, 3, H, W)` float32 in `[0, 1]`.
  - Per-frame metrics return a `(B,)` per-video mean — aggregation to a
    single scalar happens in the reporter, not inside the metric.
  - LPIPS converts to `[-1, 1]` at its call site.
  - Each stateful metric (LPIPS / FVD / LMD / IDSimilarity) loads its
    network once at construction and pins it on a chosen device.
"""
