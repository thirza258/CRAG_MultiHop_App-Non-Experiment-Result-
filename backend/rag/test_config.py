import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rag.config import load_pipeline_config


class DeploymentConfigTests(unittest.TestCase):
    def test_partial_config_preserves_other_module_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yml"
            path.write_text("llm_model: custom/chat\ndense_config:\n  embedding_model: custom/embedding\nchunking:\n  chunk_size: 900\n")
            with mock.patch.dict(os.environ, {"RAG_CONFIG_FILE": str(path)}):
                config = load_pipeline_config()
        self.assertEqual(config["llm_model"], "custom/chat")
        self.assertEqual(config["dense_config"]["embedding_model"], "custom/embedding")
        self.assertEqual(config["dense_config"]["top_k"], 4)
        self.assertEqual(config["chunking"]["chunk_size"], 900)
        self.assertEqual(config["chunking"]["strategy"], "recursive")

    def test_explicit_missing_config_file_is_reported(self):
        with mock.patch.dict(os.environ, {"RAG_CONFIG_FILE": "/does-not-exist/crag.yml"}):
            with self.assertRaisesRegex(ValueError, "does not exist"):
                load_pipeline_config()
