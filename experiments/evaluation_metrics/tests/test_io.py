"""IO sanity: sample_id splitter, manifest indexing, frame resampler.

These are pure-python and have no heavy deps, so they always run."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.evaluation_metrics.metrics.io import (
    _resample_frame_indices, _split_sample_id, load_run_metadata,
)


def test_split_sample_id_same_identity():
    assert _split_sample_id("id_0457", "same_identity_reconstruction") == ("id_0457", "id_0457")


def test_split_sample_id_cross_identity():
    assert _split_sample_id("id_0457_id_0009", "cross_identity") == ("id_0457", "id_0009")


def test_split_sample_id_cross_identity_invalid():
    with pytest.raises(ValueError, match="_id_"):
        _split_sample_id("id_0457", "cross_identity")


def test_resample_identity_when_fps_matches():
    idx = _resample_frame_indices(n_src=100, src_fps=25.0, target_fps=25, max_frames=None)
    assert idx == list(range(100))


def test_resample_identity_when_target_none():
    idx = _resample_frame_indices(n_src=10, src_fps=30.0, target_fps=None, max_frames=None)
    assert idx == list(range(10))


def test_resample_downsamples_30_to_25():
    """30 fps → 25 fps: roughly 5/6 of source frames retained."""
    idx = _resample_frame_indices(n_src=30, src_fps=30.0, target_fps=25, max_frames=None)
    assert len(idx) == 25
    assert idx[0] == 0
    assert idx[-1] < 30                          # in-bounds
    # Monotonic-non-decreasing — temporal order preserved.
    assert all(b >= a for a, b in zip(idx, idx[1:]))


def test_resample_max_frames_caps_output():
    idx = _resample_frame_indices(n_src=100, src_fps=25.0, target_fps=25, max_frames=10)
    assert idx == list(range(10))


def test_load_run_metadata_missing_args(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="run_args.json"):
        load_run_metadata(tmp_path)


def test_load_run_metadata_unknown_dataset(tmp_path: Path):
    """`load_run_metadata` fails loud when the dataset's manifest doesn't
    exist on disk — would-be silent mis-routing of metrics is worse than
    refusing to run."""
    (tmp_path / "run_args.json").write_text(json.dumps({
        "dataset": "no_such_dataset", "protocol": "same_identity_reconstruction",
    }))
    with pytest.raises(FileNotFoundError, match="manifest"):
        load_run_metadata(tmp_path)


def test_load_run_metadata_missing_protocol(tmp_path: Path):
    (tmp_path / "run_args.json").write_text(json.dumps({"dataset": "talkvid"}))
    with pytest.raises(ValueError, match="protocol"):
        load_run_metadata(tmp_path)
