"""Helpers for loading Lightning checkpoints into `MarionetteDiffusion`.

`strip_legacy_keys` filters out per-block cross-attention K/V/Q weights
(`*.transformer_blocks.<i>.attn2.*` / `norm2.*`) and standalone
`audio_encoder.*` entries from a checkpoint's "unexpected" list. Older
Lightning checkpoints carry these tensors; the current architecture has no
modules they would bind to, so silently dropping them at load time is the
correct behaviour. Without this filter, `load_state_dict(strict=False)`
would still surface them in `unexpected` and our load shim would raise on a
non-empty list.
"""
from __future__ import annotations

import re

_GEN_ATTN_LEGACY_RE = re.compile(
    r"^model\.diffusion_model\..*\.transformer_blocks\.\d+\.(attn2|norm2)\."
)


def strip_legacy_keys(unexpected: list[str]) -> list[str]:
    """Return `unexpected` with legacy cross-attention and audio-encoder
    keys removed. Other unexpected keys pass through unchanged so genuine
    state-dict mismatches still raise."""
    return [
        k for k in unexpected
        if not k.startswith("audio_encoder.")
        and not _GEN_ATTN_LEGACY_RE.match(k)
    ]
