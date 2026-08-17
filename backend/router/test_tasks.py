"""Tests for the router Celery tasks, called directly as plain functions.

Covers, without a broker, a network, an API key or any of Redis/Postgres/Chroma:

- build_index_task: the document-ingestion path (pending -> indexing -> ready),
  all three duplicate-execution guards, re-indexing after a failure, the
  mark-failed-and-retry path, and the missing-document-id contract.

The RAG engine is always patched at the point of use (the names imported into
router.tasks), never on rag.rag_service - rag_service.rag_registry is a
process-wide singleton that rag/tests.py also manipulates.
"""

import unittest
import uuid
from unittest import mock

from django.test import TestCase

try:
    from pipeline.app_pipeline import AppRAGPipeline
    from router import tasks
    from router.models import (
        AnalysisBatch,
        AnalysisResult,
        Document,
        DocumentChunk,
        GuestUser,
        Job,
        UserCollection,
    )
    IMPORT_ERROR = ""
except Exception as exc:  # needs Django + the heavy pipeline import chain
    AppRAGPipeline = None
    tasks = None
    IMPORT_ERROR = str(exc)


class _RetryRequested(Exception):
    """
    Stand-in for celery.exceptions.Retry.

    Called directly, the real Task.retry() re-raises the original exception, so
    a test could not tell "a retry was requested" from "the task exploded".
    Patching retry with this sentinel makes the request observable and keeps the
    test from ever queueing or sleeping.
    """


class _PatchMixin:
    def _patch(self, target, attribute, new=mock.DEFAULT):
        """mock.patch.object bound to the test's lifetime."""
        patcher = mock.patch.object(target, attribute, new)
        mocked = patcher.start()
        self.addCleanup(patcher.stop)
        return mocked


# ──────────────────────────────────────────────────────────────────────
# build_index_task - the document ingestion path
# ──────────────────────────────────────────────────────────────────────

@unittest.skipIf(tasks is None, f"router.tasks unavailable: {IMPORT_ERROR}")
class BuildIndexTaskTests(_PatchMixin, TestCase):
    def setUp(self):
        self.user = GuestUser.objects.create(
            email="alice@example.com", username="alice"
        )
        self.document = Document.objects.create(
            user=self.user,
            name="paper.pdf",
            source_type="pdf",
            extracted_text_path="/tmp/paper.txt",
            status="pending",
            file_hash="hash-a",
        )

        # autospec so a signature change on AppRAGPipeline._build_index breaks
        # this test instead of silently accepting whatever the task passes.
        self.pipeline = mock.create_autospec(AppRAGPipeline, instance=True)
        self.get_registry = self._patch(tasks, "get_registry")
        self.get_registry.return_value.get_engine.return_value = self.pipeline

    def _call(self, document_id=None, username="alice"):
        return tasks.build_index_task(
            document_id=self.document.pk if document_id is None else document_id,
            username=username,
        )

    def _stored_status(self, document_id=None):
        pk = self.document.pk if document_id is None else document_id
        return Document.objects.get(pk=pk).status

    def test_pending_document_is_indexed_and_marked_ready(self):
        seen_status = []
        self.pipeline._build_index.side_effect = (
            lambda **kwargs: seen_status.append(self._stored_status())
        )

        result = self._call()

        self.assertEqual(
            result, {"status": "success", "document_id": self.document.pk}
        )
        self.get_registry.return_value.get_engine.assert_called_once_with()
        self.pipeline._build_index.assert_called_once()
        _, kwargs = self.pipeline._build_index.call_args
        self.assertEqual(kwargs["username"], "alice")
        self.assertEqual(kwargs["document"].pk, self.document.pk)
        # The claim must be committed before the heavy work starts, otherwise a
        # second worker picking up the same document would index it twice.
        self.assertEqual(seen_status, ["indexing"])
        self.assertEqual(self._stored_status(), "ready")

    def test_ready_document_returns_immediately_without_reindexing(self):
        self.document.status = "ready"
        self.document.save(update_fields=["status"])

        result = self._call()

        self.assertIsNone(result)
        self.pipeline._build_index.assert_not_called()
        # The guard must fire before the engine is even fetched.
        self.get_registry.assert_not_called()
        self.assertEqual(self._stored_status(), "ready")

    def test_indexing_document_returns_immediately(self):
        self.document.status = "indexing"
        self.document.save(update_fields=["status"])

        result = self._call()

        self.assertIsNone(result)
        self.pipeline._build_index.assert_not_called()
        self.get_registry.assert_not_called()
        # Must not be reported as finished by a duplicate delivery.
        self.assertEqual(self._stored_status(), "indexing")

    def test_document_with_existing_chunks_is_marked_ready_without_reindexing(self):
        collection = UserCollection.objects.create(
            user=self.user, collection_name="user_alice_collection"
        )
        DocumentChunk.objects.create(
            document=self.document,
            user_collection=collection,
            chroma_id="alice_1_chunk_0",
            chunk_index=0,
        )
        # Deliberately still "pending" - a "ready"/"indexing" document would
        # return on an earlier guard and this test would pass for the wrong
        # reason.
        self.assertEqual(self._stored_status(), "pending")

        result = self._call()

        self.assertIsNone(result)
        self.pipeline._build_index.assert_not_called()
        self.get_registry.assert_not_called()
        self.assertEqual(self._stored_status(), "ready")

    def test_failed_document_is_reindexed(self):
        # Only "ready" and "indexing" are guarded, so a failed ingest is
        # retryable by re-dispatching the task.
        self.document.status = "failed"
        self.document.save(update_fields=["status"])

        result = self._call()

        self.assertEqual(
            result, {"status": "success", "document_id": self.document.pk}
        )
        self.pipeline._build_index.assert_called_once()
        self.assertEqual(self._stored_status(), "ready")

    def test_build_index_failure_marks_document_failed_and_requests_retry(self):
        error = RuntimeError("chroma down")
        self.pipeline._build_index.side_effect = error
        retry = self._patch(
            tasks.build_index_task,
            "retry",
            mock.Mock(side_effect=_RetryRequested("retry requested")),
        )

        with self.assertRaises(_RetryRequested):
            self._call()

        self.assertEqual(retry.call_count, 1)
        _, retry_kwargs = retry.call_args
        self.assertIs(retry_kwargs["exc"], error)
        self.assertEqual(retry_kwargs["countdown"], 60)  # 60 * 2 ** 0 retries
        # The document must not be left stuck on "indexing", or every later
        # delivery would hit the duplicate guard and never index it.
        self.assertEqual(self._stored_status(), "failed")

    def test_missing_document_id_raises_does_not_exist(self):
        missing_id = self.document.pk + 1000
        retry = self._patch(
            tasks.build_index_task,
            "retry",
            mock.Mock(side_effect=_RetryRequested("retry requested")),
        )

        with self.assertRaises(Document.DoesNotExist):
            self._call(document_id=missing_id)

        # A document that does not exist can never appear, so retrying is
        # pointless - the task must re-raise instead.
        retry.assert_not_called()
        self.pipeline._build_index.assert_not_called()
