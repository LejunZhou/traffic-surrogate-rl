"""
Config loading and merging utilities.

Loads YAML config files and supports dot-notation access.
Allows overriding keys from the command line or programmatically.
"""

from __future__ import annotations

import yaml
from pathlib import Path

# TODO: implement config utilities


def load_config(config_path: str) -> dict:
    """Load a YAML config file and return it as a nested dict.

    Args:
        config_path: Path to a .yaml config file.

    Returns:
        Nested dict of config values.
    """
    raise NotImplementedError


def merge_configs(base: dict, overrides: dict) -> dict:
    """Recursively merge overrides into base config.

    Args:
        base: Base config dict.
        overrides: Dict of keys to override (nested dicts are merged, not replaced).

    Returns:
        Merged config dict.
    """
    raise NotImplementedError
