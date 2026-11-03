"""Read sidecar model ids from this package and the process environment.

Embed and relate defaults live in imprint.plugin.json. imprint does not
choose those ids. embed.yaml may set relate_model to off; it does not
select the embed model.
"""

from __future__ import annotations

import json
import os
from typing import Any

_PLUGIN_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imprint.plugin.json")
_RELATE_OFF = {"", "0", "off", "none", "false"}


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
    """Plugin config from either file shape IMPRINT_PLUGIN_CONFIG points at.

    Project imprint.yaml nests it at plugins.embed.config.
    ~/.imprint/embed.yaml puts the same map at top-level config.
    """
    plugins = doc.get("plugins")
    if isinstance(plugins, dict):
        embed = plugins.get("embed")
        if isinstance(embed, dict):
            cfg = embed.get("config")
            if isinstance(cfg, dict):
                return cfg
    cfg = doc.get("config")
    return cfg if isinstance(cfg, dict) else {}


def schema_default(key: str) -> str | None:
    """String default from this plugin's configSchema, if one is declared."""
    try:
        with open(_PLUGIN_JSON, encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    schema = doc.get("configSchema")
    if not isinstance(schema, dict):
        return None
    field = schema.get(key)
    if not isinstance(field, dict):
        return None
    default = field.get("default")
    if isinstance(default, str) and default.strip():
        return default.strip()
    return None


def _model_from_config_value(value: Any) -> str | None:
    """None disables /relate. A missing key is not passed here."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if text.lower() in _RELATE_OFF:
        return None
    return text


def embed_model_id() -> str:
    """Embedding model id for this process.

    IMPRINT_EMBED_MODEL wins when set. Otherwise the configSchema default.
    A model key in embed.yaml is ignored.
    """
    env = os.environ.get("IMPRINT_EMBED_MODEL", "").strip()
    if env:
        return env
    default = schema_default("model")
    if not default:
        raise RuntimeError("imprint.plugin.json configSchema.model.default is empty")
    return default


def relate_model_id() -> str | None:
    """Model id for POST /relate; None disables the endpoint.

    IMPRINT_RELATE_MODEL wins when the process environment sets it.
    Otherwise read relate_model from IMPRINT_PLUGIN_CONFIG.
    A missing key uses this plugin's configSchema default.
    """
    if "IMPRINT_RELATE_MODEL" in os.environ:
        return _model_from_config_value(os.environ.get("IMPRINT_RELATE_MODEL", ""))
    cfg_path = os.environ.get("IMPRINT_PLUGIN_CONFIG", "").strip()
    if cfg_path:
        cfg = _embed_config(_read_yaml(cfg_path))
        if "relate_model" in cfg:
            return _model_from_config_value(cfg.get("relate_model"))
    return schema_default("relate_model")
