"""Stdout / stderr tee to a file.

Used by the training orchestrator to mirror the terminal output of a run into
a `log.txt` inside the run directory so crashes, warnings, load-summary
diagnostics, and per-step status all stay recoverable after the session ends.

Rank-scoping is the caller's responsibility — multiple ranks writing to the
same file will interleave and corrupt it. Call `install_log_tee` only on
rank 0 in DDP.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path


class _TeeStream:
    """Duplicate writes to an additional file handle while forwarding to the
    original stream. Preserves `isatty()` and `fileno()` so tqdm / any other
    TTY-aware library still detects a terminal and renders progress bars
    normally."""

    def __init__(self, original, file_handle):
        self._original = original
        self._file     = file_handle

    def write(self, s):
        self._original.write(s)
        self._file.write(s)

    def flush(self):
        self._original.flush()
        self._file.flush()

    def isatty(self):
        return self._original.isatty()

    def fileno(self):
        return self._original.fileno()

    def __getattr__(self, name):
        # Fallback for any attribute not explicitly defined (e.g. `encoding`).
        return getattr(self._original, name)


def install_log_tee(log_path: Path) -> None:
    """Tee stdout and stderr into `log_path` (append mode, line-buffered).

    Call ONLY on rank 0 in DDP — multi-rank writes to the same file
    interleave. Other ranks' stdout stays terminal-only, which is fine
    because every important progress/status print in this codebase is
    already gated by `is_rank_zero()`.

    Line buffering (`buffering=1`) means `tail -f` sees output in real time
    without waiting for a block flush. Append mode means resuming a run
    preserves the prior log; a banner marks where each session begins.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log_path, "a", buffering=1)
    banner = (
        f"\n{'=' * 70}\n"
        f"=== log tee opened at {datetime.now().isoformat(timespec='seconds')}\n"
        f"{'=' * 70}\n"
    )
    fh.write(banner)
    fh.flush()
    sys.stdout = _TeeStream(sys.__stdout__, fh)
    sys.stderr = _TeeStream(sys.__stderr__, fh)
