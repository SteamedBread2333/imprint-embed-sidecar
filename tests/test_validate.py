"""Validation helpers for embed/relate payloads."""

from __future__ import annotations

import unittest

import server


class ValidatePairsTest(unittest.TestCase):
    def test_rejects_empty_pairs(self) -> None:
        with self.assertRaises(server.RelateError):
            server._validate_pairs({})

    def test_accepts_pair(self) -> None:
        got = server._validate_pairs({"pairs": [["a", "b"]]})
        self.assertEqual(got, [("a", "b")])


if __name__ == "__main__":
    unittest.main()
