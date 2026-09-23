"""imprint-embed sidecar: persistent embedding service over loopback HTTP.

Design constraints (from the handover plan):
  * resident process — never fork per call (cold start is 1-3s)
  * HTTP JSON on 127.0.0.1 only, so Go can use a thin 2s-timeout client
  * model weights live outside the project (~/.cache/imprint/models)
  * failures must be visible in logs; the caller degrades to lexical matching

Endpoints
  GET  /health  -> {"ok":true,"model":...,"dim":...}
  POST /embed   -> {"vectors":[[...]],"dim":512,"model":...}
                   (bge-small-zh-v1.5 emits 512-dim vectors; read dim from the
                   response, do not hardcode) {"error":"..."} on failure
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_PORT = 4174
DEFAULT_CACHE = os.path.expanduser("~/.cache/imprint/models")
DEFAULT_HOST = "127.0.0.1"
MAX_TEXTS_PER_REQUEST = 64
MAX_TEXT_CHARS = 4000

log = logging.getLogger("imprint-embed")


class EmbedError(RuntimeError):
    """Raised when the backend cannot produce vectors."""


class Embedder:
    """Lazy, thread-safe model wrapper.

    Loading happens on first use so `imprint up` health-checks stay fast and a
    missing weight directory surfaces as a clear startup error rather than a
    hanging process.
    """

    def __init__(self, model: str, cache_dir: str) -> None:
        self.model = model
        self.cache_dir = cache_dir
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
                from fastembed import TextEmbedding  # imported lazily: heavy
            except ImportError as exc:  # pragma: no cover - env dependent
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
            log.info("model loaded in %.2fs", time.time() - started)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self._load()
        assert self._model is not None
        started = time.time()
        try:
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


def make_handler(embedder: Embedder) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "imprint-embed/0.1"
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

        def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
            if self.path.split("?")[0] != "/health":
                self._send(404, {"ok": False, "error": "not found"})
                return
            self._send(
                200,
                {
                    "ok": True,
                    "model": embedder.model,
                    "dim": embedder.dim,
                    "cache_dir": embedder.cache_dir,
                },
            )

        def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
            if self.path.split("?")[0] != "/embed":
                self._send(404, {"ok": False, "error": "not found"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                self._send(400, {"error": "empty request body"})
                return
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise EmbedError("body must be a JSON object")
                texts = _validate_texts(payload)
                vectors = embedder.embed(texts)
            except EmbedError as exc:
                log.warning("embed rejected: %s", exc)
                self._send(400, {"error": str(exc)})
                return
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                self._send(400, {"error": f"invalid JSON: {exc}"})
                return
            except Exception as exc:  # unexpected: surface, degrade in Go
                log.exception("embed failed")
                self._send(500, {"error": str(exc)})
                return
            self._send(
                200,
                {
                    "vectors": vectors,
                    "dim": len(vectors[0]) if vectors else 0,
                    "model": embedder.model,
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
    parser.add_argument("--warm", action="store_true", help="load the model before serving")
    parser.add_argument(
        "--parent-pid",
        type=int,
        default=None,
        help=(
            "PID of the parent process to watch; sidecar exits when the parent "
            "disappears. Replaces an idle timer, which would misfire on a long-lived "
            "MCP host that keeps calling embed. Falls back to IMPRINT_PARENT_PID env."
        ),
    )
    return parser.parse_args(argv)


def resolve_port(cli_port: int | None) -> int:
    """Port precedence: CLI flag > IMPRINT_PLUGIN_PORT > default.

    The plugin manager always exports IMPRINT_PLUGIN_PORT, so honouring it
    keeps a single source of truth for the port.
    """
    if cli_port:
        return cli_port
    env_port = os.environ.get("IMPRINT_PLUGIN_PORT", "").strip()
    if env_port.isdigit():
        return int(env_port)
    return DEFAULT_PORT


def resolve_parent_pid(cli_pid: int | None) -> int:
    """Parent PID precedence: CLI flag > IMPRINT_PARENT_PID > 0 (disabled).

    0 (or any value <= 0) means "do not watch", which is the safe default for
    ad-hoc CLI runs.
    """
    if cli_pid and cli_pid > 0:
        return cli_pid
    env_pid = os.environ.get("IMPRINT_PARENT_PID", "").strip()
    if env_pid.isdigit():
        v = int(env_pid)
        if v > 0:
            return v
    return 0


def watch_parent(parent_pid: int, interval: float = 30.0) -> None:
    """Poll parent PID; exit when the parent disappears.

    Uses signal 0, which never delivers a real signal but raises ProcessLookupError
    if the PID is gone (or PermissionError if it has changed uid). The daemon
    thread is stopped only via process exit, so a KeyboardInterrupt in serve_forever
    is what tears the server down — this thread only fires on parent death.
    """
    while True:
        try:
            os.kill(parent_pid, 0)
        except (ProcessLookupError, PermissionError):
            log.info("parent pid %d gone, exiting", parent_pid)
            os._exit(0)
        time.sleep(interval)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s imprint-embed %(message)s",
        stream=sys.stdout,
    )
    args = parse_args(argv)
    port = resolve_port(args.port)
    parent_pid = resolve_parent_pid(args.parent_pid)
    embedder = Embedder(args.model, args.cache_dir)
    if args.warm:
        embedder.embed(["warmup"])
    server = ThreadingHTTPServer((args.host, port), make_handler(embedder))
    if parent_pid > 0:
        watcher = threading.Thread(
            target=watch_parent,
            args=(parent_pid,),
            name="parent-watcher",
            daemon=True,
        )
        watcher.start()
        log.info(
            "listening on http://%s:%d model=%s parent_pid=%d",
            args.host, port, args.model, parent_pid,
        )
    else:
        log.info("listening on http://%s:%d model=%s", args.host, port, args.model)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        log.info("shutting down")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
