"""Prefetch Hugging Face weights before CrossEncoder/fastembed warm (mirror-safe)."""

from __future__ import annotations

import logging
import os

log = logging.getLogger("imprint-embed")

# ONNX bundles are huge and unused by sentence-transformers PyTorch load.
_IGNORE = ["onnx/**", "*.onnx", "*.csv"]


def _hub_env() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")


def prefetch_relate(model_id: str, cache_dir: str) -> None:
    if not model_id.strip():
        return
    _hub_env()
    from huggingface_hub import snapshot_download

    log.info("prefetch: downloading relate model %s (skipping onnx)", model_id)
    snapshot_download(
        repo_id=model_id.strip(),
        cache_dir=cache_dir,
        ignore_patterns=_IGNORE,
    )
    log.info("prefetch: relate model ready in cache")
