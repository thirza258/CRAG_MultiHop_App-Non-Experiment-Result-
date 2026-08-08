"""Tests for the RAG engine registry recovery behavior."""

import unittest
from unittest import mock

try:
    from rag import rag_service
    IMPORT_ERROR = ""
except Exception as exc:  # heavy pipeline imports need the full image
    rag_service = None
    IMPORT_ERROR = str(exc)


@unittest.skipIf(rag_service is None, f"rag_service unavailable: {IMPORT_ERROR}")
class RagRegistryTests(unittest.TestCase):
    def setUp(self):
        self.registry = rag_service.rag_registry
        self._original_engine = self.registry.engine
        self.registry.engine = None
        self.addCleanup(self._restore)

    def _restore(self):
        self.registry.engine = self._original_engine

    def test_get_engine_retries_failed_initialization(self):
        sentinel = object()
        with mock.patch.object(
            rag_service, "AppRAGPipeline", return_value=sentinel
        ):
            self.assertIs(self.registry.get_engine(), sentinel)

    def test_get_engine_ignores_legacy_method_and_model_args(self):
        sentinel = object()
        with mock.patch.object(
            rag_service, "AppRAGPipeline", return_value=sentinel
        ):
            self.assertIs(
                self.registry.get_engine("hybrid", "some-model"), sentinel
            )

    def test_get_engine_raises_but_recovers_on_next_call(self):
        with mock.patch.object(
            rag_service, "AppRAGPipeline", side_effect=RuntimeError("chroma down")
        ):
            with self.assertRaises(ValueError):
                self.registry.get_engine()

        # Once the dependency is back, the same registry must recover
        # without a container restart.
        sentinel = object()
        with mock.patch.object(
            rag_service, "AppRAGPipeline", return_value=sentinel
        ):
            self.assertIs(self.registry.get_engine(), sentinel)


if __name__ == "__main__":
    unittest.main()
