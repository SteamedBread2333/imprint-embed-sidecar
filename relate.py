"""Pairwise NLI scores via a local cross-encoder (no lexical polarity tables)."""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

log = logging.getLogger("imprint-embed")


class RelateError(RuntimeError):
    """Raised when the relate backend cannot score pairs."""


class Relater:
    """Lazy cross-encoder for claim-pair relationship scores."""

    def __init__(self, model: str, cache_dir: str) -> None:
        self.model = model
        self.cache_dir = cache_dir
        self._lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._model: Any = None
        self._labels: list[str] = []

    def _load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:  # pragma: no cover
                raise RelateError(
                    "sentence-transformers is not installed; run scripts/setup.sh"
                ) from exc
            os.makedirs(self.cache_dir, exist_ok=True)
            started = time.time()
            try:
                self._model = CrossEncoder(self.model, cache_folder=self.cache_dir)
            except Exception as exc:
                raise RelateError(f"cannot load relate model {self.model!r}: {exc}") from exc
            labels = getattr(self._model, "config", None)
            if labels is not None and hasattr(labels, "id2label"):
                id2label = labels.id2label
                if isinstance(id2label, dict):
                    self._labels = [
                        str(id2label[i]).lower()
                        for i in sorted(id2label.keys(), key=lambda x: int(x))
                    ]
            if not self._labels:
                self._labels = ["contradiction", "neutral", "entailment"]
            log.info("relate model loaded in %.2fs labels=%s", time.time() - started, self._labels)

    @property
    def labels(self) -> list[str]:
        self._load()
        return list(self._labels)

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
        if not pairs:
            return []
        self._load()
        assert self._model is not None
        started = time.time()
        try:
            with self._infer_lock:
                raw = self._model.predict(
                    [(a, b) for a, b in pairs],
                    apply_softmax=True,
                )
        except Exception as exc:
            raise RelateError(f"relate inference failed: {exc}") from exc

        out: list[dict[str, float]] = []
        for row in raw:
            if hasattr(row, "__iter__") and not isinstance(row, (str, bytes)):
                probs = [float(x) for x in row]
            else:
                probs = [float(row)]
            scores = {self._labels[i]: probs[i] for i in range(min(len(probs), len(self._labels)))}
            out.append(scores)
        if len(out) != len(pairs):
            raise RelateError(f"backend returned {len(out)} scores for {len(pairs)} pairs")
        log.debug("scored %d pair(s) in %.3fs", len(pairs), time.time() - started)
        return out
