"""
Config composition utilities for the Marionette training pipeline.

Configs are composed from a base YAML plus an optional sequence of overlay
YAMLs. Overlays are partial configs that modify only the fields they care
about.

Experiment-side usage:

    # experiments/<name>/configs/variant.yaml
    base: marionette/configs/base.yaml
    overlays:
      - <optional overlay yamls>

    # optional inline overrides (applied last) go here

Any other top-level keys in the experiment YAML are treated as inline
overrides and merged last — useful for quick tweaks that don't deserve their
own overlay file.

The resolved config should be snapshotted alongside the run output so the
exact values that trained a checkpoint are recoverable, regardless of later
changes to the base or overlays.
"""

from pathlib import Path
from typing import Union

from omegaconf import OmegaConf, DictConfig


def load_experiment_config(path: Union[str, Path]) -> DictConfig:
    """Resolve an experiment config composed of a base + overlay chain.

    Args:
        path: path to an experiment YAML file. The file must have top-level
            meta-keys `base` (str) and optionally `overlays` (list[str]).
            All paths in these fields are resolved relative to the current
            working directory (typically the repo root).

    Returns:
        A fully merged DictConfig with the `base` and `overlays` meta-keys
        removed. Later overlays override earlier ones; any remaining top-level
        fields in the experiment YAML are merged last as inline overrides.
    """
    exp = OmegaConf.load(path)
    if "base" not in exp:
        raise KeyError(
            f"{path}: experiment config must declare a top-level `base:` key"
        )

    base_path = exp.pop("base")
    overlay_paths = list(exp.pop("overlays", []))

    merged = OmegaConf.load(base_path)
    for ov in overlay_paths:
        merged = OmegaConf.merge(merged, OmegaConf.load(ov))

    # Anything that remained in `exp` is an inline override — applied last.
    if len(exp) > 0:
        merged = OmegaConf.merge(merged, exp)

    return merged  # type: ignore[return-value]