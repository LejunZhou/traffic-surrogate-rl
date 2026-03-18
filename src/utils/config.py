"""
Config loading and merging utilities.

Loads YAML config files and supports recursive dict merging for overrides.
"""

from __future__ import annotations

import copy
import yaml
from pathlib import Path


def load_config(config_path: str) -> dict:
    """Load a YAML config file and return it as a nested dict.

    Args:
        config_path: Path to a .yaml config file.

    Returns:
        Nested dict of config values.

    Raises:
        FileNotFoundError: If config_path does not exist.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r") as f:
        cfg = yaml.safe_load(f)
    return cfg if cfg is not None else {}


def merge_configs(base: dict, overrides: dict) -> dict:
    """Recursively merge overrides into a deep copy of base.

    Nested dicts are merged key-by-key; all other values are replaced.

    Args:
        base: Base config dict.
        overrides: Values to override (nested dicts are merged, not replaced).

    Returns:
        New merged config dict. base is not mutated.
    """
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result
