"""imprint-embed sidecar: persistent embedding + relate service over loopback HTTP.

Endpoints
  GET  /health  -> embed + relate capability
  POST /embed   -> vectors
  POST /relate  -> pairwise NLI-style scores (no duplicate/advisory decisions)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from config import relate_model_id
from relate import RelateError, Relater

DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_PORT = 4174
DEFAULT_CACHE = os.path.expanduser("~/.cache/imprint/models")
DEFAULT_HOST = "127.0.0.1"
MAX_TEXTS_PER_REQUEST = 64
MAX_PAIRS_PER_REQUEST = 64
MAX_TEXT_CHARS = 4000

log = logging.getLogger("imprint-embed")


class EmbedError(RuntimeError):
    """Raised when the backend cannot produce vectors."""


class Embedder:
    """Lazy, thread-safe embedding model."""

    def __init__(self, model: str, cache_dir: str, infer_lock: threading.Lock) -> None:
        self.model = model
        self.cache_dir = cache_dir
        self._infer_lock = infer_lock
        self._lock = threading.Lock()
        self._model: Any = None
        self._dim = 0

    def _load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:
                raise EmbedError(
                    "fastembed is not installed; run scripts/setup.sh first"
                ) from exc
            os.makedirs(self.cache_dir, exist_ok=True)
            started = time.time()
            try:
                self._model = TextEmbedding(
                    model_name=self.model, cache_dir=self.cache_dir
                )
            except Exception as exc:
                raise EmbedError(f"cannot load model {self.model!r}: {exc}") from exc
            log.info("embed model loaded in %.2fs", time.time() - started)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self._load()
        assert self._model is not None
        started = time.time()
        try:
            with self._infer_lock:
                vectors = [list(map(float, v)) for v in self._model.embed(texts)]
        except Exception as exc:
            raise EmbedError(f"embedding failed: {exc}") from exc
        if len(vectors) != len(texts):
            raise EmbedError(
                f"backend returned {len(vectors)} vectors for {len(texts)} texts"
            )
        self._dim = len(vectors[0]) if vectors else self._dim
        log.debug("embedded %d text(s) in %.3fs", len(texts), time.time() - started)
        return vectors

    @property
    def dim(self) -> int:
        return self._dim


class Sidecar:
    def __init__(self, embed_model: str, cache_dir: str, relate_model: str | None) -> None:
        self.cache_dir = cache_dir
        self.infer_lock = threading.Lock()
        self.embedder = Embedder(embed_model, cache_dir, self.infer_lock)
        self.relate_model = relate_model
        self.relater: Relater | None = (
            Relater(relate_model, cache_dir) if relate_model else None
        )

    def health_body(self) -> dict[str, Any]:
        embed = {
            "model": self.embedder.model,
            "dim": self.embedder.dim,
            "enabled": True,
        }
        if self.relater is None:
            relate = {"enabled": False, "model": None, "labels": []}
        else:
            relate = {
                "enabled": True,
                "model": self.relater.model,
                "labels": self.relater.labels if self._relate_loaded() else [],
            }
        return {
            "ok": True,
            "model": self.embedder.model,
            "dim": self.embedder.dim,
            "cache_dir": self.cache_dir,
            "embed": embed,
            "relate": relate,
        }

    def _relate_loaded(self) -> bool:
        return self.relater is not None and self.relater._model is not None  # noqa: SLF001


def _validate_texts(payload: dict[str, Any]) -> list[str]:
    texts = payload.get("texts")
    if not isinstance(texts, list) or not texts:
        raise EmbedError("'texts' must be a non-empty array of strings")
    if len(texts) > MAX_TEXTS_PER_REQUEST:
        raise EmbedError(f"at most {MAX_TEXTS_PER_REQUEST} texts per request")
    out: list[str] = []
    for item in texts:
        if not isinstance(item, str):
            raise EmbedError("'texts' must contain only strings")
        out.append(item[:MAX_TEXT_CHARS])
    return out


def _validate_pairs(payload: dict[str, Any]) -> list[tuple[str, str]]:
    pairs = payload.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise RelateError("'pairs' must be a non-empty array of [a,b] strings")
    if len(pairs) > MAX_PAIRS_PER_REQUEST:
        raise RelateError(f"at most {MAX_PAIRS_PER_REQUEST} pairs per request")
    out: list[tuple[str, str]] = []
    for item in pairs:
        if not isinstance(item, list) or len(item) != 2:
            raise RelateError("each pair must be [string, string]")
        a, b = item[0], item[1]
        if not isinstance(a, str) or not isinstance(b, str):
            raise RelateError("pair elements must be strings")
        out.append((a[:MAX_TEXT_CHARS], b[:MAX_TEXT_CHARS]))
    return out


def make_handler(sidecar: Sidecar) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "imprint-embed/0.2"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            log.info("%s %s", self.address_string(), fmt % args)

        def _send(self, status: int, body: dict[str, Any]) -> None:
            raw = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:  # noqa: N802
            if self.path.split("?")[0] != "/health":
                self._send(404, {"ok": False, "error": "not found"})
                return
            self._send(200, sidecar.health_body())

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?")[0]
            if path == "/embed":
                self._post_embed()
            elif path == "/relate":
                self._post_relate()
            else:
                self._send(404, {"ok": False, "error": "not found"})

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                raise ValueError("empty request body")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise EmbedError("body must be a JSON object")
            return payload

        def _post_embed(self) -> None:
            try:
                payload = self._read_json()
                texts = _validate_texts(payload)
                vectors = sidecar.embedder.embed(texts)
            except EmbedError as exc:
                log.warning("embed rejected: %s", exc)
                self._send(400, {"error": str(exc)})
                return
            except ValueError as exc:
                self._send(400, {"error": str(exc)})
                return
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                self._send(400, {"error": f"invalid JSON: {exc}"})
                return
            except Exception as exc:
                log.exception("embed failed")
                self._send(500, {"error": str(exc)})
                return
            self._send(
                200,
                {
                    "vectors": vectors,
                    "dim": len(vectors[0]) if vectors else 0,
                    "model": sidecar.embedder.model,
                },
            )

        def _post_relate(self) -> None:
            if sidecar.relater is None:
                self._send(
                    503,
                    {"error": "relate disabled: set relate_model in imprint.yaml or IMPRINT_RELATE_MODEL"},
                )
                return
            try:
                payload = self._read_json()
                pairs = _validate_pairs(payload)
                scores = sidecar.relater.score_pairs(pairs)
            except RelateError as exc:
                log.warning("relate rejected: %s", exc)
                self._send(400, {"error": str(exc)})
                return
            except ValueError as exc:
                self._send(400, {"error": str(exc)})
                return
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                self._send(400, {"error": f"invalid JSON: {exc}"})
                return
            except Exception as exc:
                log.exception("relate failed")
                self._send(500, {"error": str(exc)})
                return
            labels = sidecar.relater._labels or [  # noqa: SLF001 — avoid loading model for label names only
                "contradiction",
                "neutral",
                "entailment",
            ]
            self._send(
                200,
                {
                    "model": sidecar.relater.model,
                    "labels": labels,
                    "results": scores,
                },
            )

    return Handler


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="imprint embed sidecar")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--host", default=os.environ.get("IMPRINT_EMBED_HOST", DEFAULT_HOST))
    parser.add_argument("--model", default=os.environ.get("IMPRINT_EMBED_MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("IMPRINT_EMBED_CACHE", DEFAULT_CACHE),
    )
    parser.add_argument(
        "--relate-model",
        default=None,
        help="override relate model id (else imprint.yaml / IMPRINT_RELATE_MODEL)",
    )
    parser.add_argument("--warm", action="store_true", help="load models before serving")
    parser.add_argument(
        "--print-plugin-dir",
        action="store_true",
        help="print directory containing imprint.plugin.json and exit",
    )
    return parser.parse_args(argv)


def plugin_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def resolve_port(cli_port: int | None) -> int:
    if cli_port:
        return cli_port
    env_port = os.environ.get("IMPRINT_PLUGIN_PORT", "").strip()
    if env_port.isdigit():
        return int(env_port)
    return DEFAULT_PORT


def write_pid_file(state_dir: str) -> str:
    if not state_dir:
        return ""
    path = os.path.join(state_dir, "sidecar.pid")
    try:
        os.makedirs(state_dir, exist_ok=True)
        with open(path, "w") as f:
            f.write(str(os.getpid()))
    except OSError as exc:
        log.warning("cannot write pid file %s: %s", path, exc)
        return ""
    return path


def remove_pid_file(path: str) -> None:
    if not path:
        return
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except OSError as exc:
        log.warning("cannot remove pid file %s: %s", path, exc)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s imprint-embed %(message)s",
        stream=sys.stdout,
    )
    args = parse_args(argv)
    if args.print_plugin_dir:
        print(plugin_dir())
        return 0

    port = resolve_port(args.port)
    relate_model = args.relate_model or relate_model_id()
    sidecar = Sidecar(args.model, args.cache_dir, relate_model)
    if args.warm:
        sidecar.embedder.embed(["warmup"])
        if sidecar.relater is not None:
            sidecar.relater.score_pairs([("warm", "warm")])
    server = ThreadingHTTPServer((args.host, port), make_handler(sidecar))
    pid_file = write_pid_file(os.environ.get("IMPRINT_PLUGIN_STATE", ""))
    log.info(
        "listening on http://%s:%d embed=%s relate=%s pid=%d",
        args.host,
        port,
        args.model,
        relate_model or "disabled",
        os.getpid(),
    )

    def _term_to_keyboardinterrupt(_signum, _frame):
        raise KeyboardInterrupt

    prev_term = signal.signal(signal.SIGTERM, _term_to_keyboardinterrupt)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        log.info("shutting down")
    finally:
        server.server_close()
        remove_pid_file(pid_file)
        signal.signal(signal.SIGTERM, prev_term)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
