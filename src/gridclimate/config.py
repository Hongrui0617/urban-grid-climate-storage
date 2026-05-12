from pathlib import Path
from typing import Any, Dict

import yaml


REQUIRED_TOP_LEVEL_KEYS = [
    "project",
    "paths",
    "columns",
    "data",
    "features",
    "split",
    "models",
    "evaluation",
]


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Parameters
    ----------
    config_path : str or Path
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        raise ValueError(f"Config file is empty: {config_path}")

    validate_config(config)

    return config


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate that the configuration contains required top-level sections.
    """
    missing_keys = [
        key for key in REQUIRED_TOP_LEVEL_KEYS
        if key not in config
    ]

    if missing_keys:
        raise KeyError(
            f"Missing required config sections: {missing_keys}"
        )


def get_project_root() -> Path:
    """
    Return the project root directory.

    This assumes this file is located at:
    src/gridclimate/config.py
    """
    return Path(__file__).resolve().parents[2]


def resolve_project_path(relative_path: str | Path) -> Path:
    """
    Convert a project-relative path into an absolute path.
    """
    return get_project_root() / Path(relative_path)