"""relate_model resolution stays inside the sidecar plugin."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import config


class RelateModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self._env = {
            "IMPRINT_RELATE_MODEL": os.environ.get("IMPRINT_RELATE_MODEL"),
            "IMPRINT_PLUGIN_CONFIG": os.environ.get("IMPRINT_PLUGIN_CONFIG"),
        }
        os.environ.pop("IMPRINT_RELATE_MODEL", None)
        os.environ.pop("IMPRINT_PLUGIN_CONFIG", None)

    def tearDown(self) -> None:
        for key, value in self._env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_schema_default_matches_plugin_json(self) -> None:
        raw = json.loads(Path(config._PLUGIN_JSON).read_text(encoding="utf-8"))
        self.assertEqual(
            config.schema_default("relate_model"),
            raw["configSchema"]["relate_model"]["default"],
        )

    def test_missing_key_uses_schema(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write("enabled: true\nconfig:\n  port: 4174\n")
            path = f.name
        self.addCleanup(os.unlink, path)
        os.environ["IMPRINT_PLUGIN_CONFIG"] = path
        self.assertEqual(config.relate_model_id(), config.schema_default("relate_model"))

    def test_flat_embed_yaml_overrides_schema(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write("config:\n  relate_model: org/custom-nli\n")
            path = f.name
        self.addCleanup(os.unlink, path)
        os.environ["IMPRINT_PLUGIN_CONFIG"] = path
        self.assertEqual(config.relate_model_id(), "org/custom-nli")

    def test_nested_project_yaml(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write("plugins:\n  embed:\n    config:\n      relate_model: org/nested\n")
            path = f.name
        self.addCleanup(os.unlink, path)
        os.environ["IMPRINT_PLUGIN_CONFIG"] = path
        self.assertEqual(config.relate_model_id(), "org/nested")

    def test_off_disables(self) -> None:
        os.environ["IMPRINT_RELATE_MODEL"] = "off"
        self.assertIsNone(config.relate_model_id())


if __name__ == "__main__":
    unittest.main()
