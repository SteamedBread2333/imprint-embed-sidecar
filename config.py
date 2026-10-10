"""Read sidecar model ids from environment and imprint.yaml (no threshold constants)."""

from __future__ import annotations

import os
from typing import Any


def _read_yaml(path: str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except OSError:
        return {}
    return data if isinstance(data, dict) else {}


def _embed_config(doc: dict[str, Any]) -> dict[str, Any]:
    plugins = doc.get("plugins")
    if not isinstance(plugins, dict):
        return {}
    embed = plugins.get("embed")
    if not isinstance(embed, dict):
        return {}
    cfg = embed.get("config")
    return cfg if isinstance(cfg, dict) else {}


def relate_model_id() -> str | None:
    """Model id for POST /relate; None disables the endpoint."""
    env = os.environ.get("IMPRINT_RELATE_MODEL", "").strip()
    if env:
        return env
    cfg_path = os.environ.get("IMPRINT_PLUGIN_CONFIG", "").strip()
    if cfg_path:
        cfg = _embed_config(_read_yaml(cfg_path))
        model = cfg.get("relate_model")
        if isinstance(model, str) and model.strip():
            return model.strip()
    return None
