"""HTTP contract tests for /relate (no model download)."""

from __future__ import annotations

import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from unittest.mock import patch

import server


class RelateHandlerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sidecar = server.Sidecar("stub-embed", "/tmp/imprint-test-cache", "stub-relate")
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(self.sidecar))
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    def test_relate_disabled_without_model(self) -> None:
        sidecar = server.Sidecar("stub-embed", "/tmp/imprint-test-cache", None)
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(sidecar))
        port = httpd.server_address[1]
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        self.addCleanup(httpd.shutdown)
        self.addCleanup(httpd.server_close)
        import urllib.error
        import urllib.request

        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/relate",
            data=json.dumps({"pairs": [["a", "b"]]}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req, timeout=2)
        self.assertEqual(ctx.exception.code, 503)

    @patch.object(server.Relater, "score_pairs", return_value=[{"contradiction": 0.8, "entailment": 0.1, "neutral": 0.1}])
    def test_relate_ok(self, _mock: object) -> None:
        import urllib.request

        body = json.dumps({"pairs": [["Use tabs", "Use spaces"]]}).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/relate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read().decode())
        self.assertEqual(data["results"][0]["contradiction"], 0.8)


if __name__ == "__main__":
    unittest.main()
