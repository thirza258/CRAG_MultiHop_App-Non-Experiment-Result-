"""Tests for the pure, dependency-free helpers the RAG pipeline is built on.

What it covers:

1. ``common.chunker.DocumentChunker`` — strategy dispatch plus the four
   strategies: ``fixed`` (stride/overlap arithmetic), ``paragraph`` (blank-line
   delimiter, single-newline fallback, oversized-paragraph splitting),
   ``recursive`` (the production default: 500/50) and ``semantic`` (driven by a
   canned-vector stub client, never a real embedding API).
2. ``emitter.status.StatusEmitter`` — the event shape published to Redis, the
   ``meta`` default, ``NULL_EMITTER`` and the swallow-and-log contract that stops
   a dead broker from aborting a query.
3. ``common.schema.RAGResponse`` — the JSON envelope every view returns.
4. ``utils.helper`` — the document storage prefix and conversation id generators.

Each block guards its own imports so the thin-environment skips stay
independent: ``emitter.status`` (stdlib only) and ``utils.helper`` (uuid only)
still run when langchain / numpy / scikit-learn are absent.

Nothing here needs the network, an API key, Redis, PostgreSQL or ChromaDB.
"""

import json
import time
import unittest
import uuid
from types import SimpleNamespace
from unittest import mock

try:
    from common.chunker import DocumentChunker
    CHUNKER_IMPORT_ERROR = ""
except Exception as exc:  # needs langchain-text-splitters + numpy + scikit-learn
    DocumentChunker = None
    CHUNKER_IMPORT_ERROR = str(exc)

try:
    from emitter import status
    STATUS_IMPORT_ERROR = ""
except Exception as exc:  # stdlib only — should never trigger
    status = None
    STATUS_IMPORT_ERROR = str(exc)

# JsonResponse reads settings.DEFAULT_CHARSET when it is constructed, so the
# schema tests need a configured Django, not just an importable one.
SCHEMA_SKIP_REASON = ""
try:
    from django.conf import settings as django_settings

    from common import schema
    if not django_settings.configured:
        SCHEMA_SKIP_REASON = "django settings are not configured"
except Exception as exc:
    schema = None
    SCHEMA_SKIP_REASON = f"common.schema unavailable: {exc}"

try:
    from utils import helper
    HELPER_IMPORT_ERROR = ""
except Exception as exc:  # uuid only — should never trigger
    helper = None
    HELPER_IMPORT_ERROR = str(exc)


# ──────────────────────────────────────────────────────────────────────
# common/chunker.py
# ──────────────────────────────────────────────────────────────────────

# 50 printable characters, no whitespace: long enough to be an "oversized"
# paragraph at chunk_size=20 and free of separators, so the split points are
# pure arithmetic.
LONG_PARAGRAPH = "".join(chr(ord("a") + i % 26) for i in range(50))

# Every constructor below passes an explicit overlap. _chunk_fixed advances by
# (chunk_size - overlap), so an overlap >= chunk_size is an infinite loop — the
# default overlap of 50 would hang these small-chunk_size tests instead of
# failing them.


@unittest.skipIf(DocumentChunker is None, f"common.chunker unavailable: {CHUNKER_IMPORT_ERROR}")
class ChunkerDispatchTests(unittest.TestCase):
    def test_defaults_match_the_production_pipeline(self):
        # pipeline/app_pipeline.py builds DocumentChunker() with no arguments.
        chunker = DocumentChunker()
        self.assertEqual(chunker.strategy, "recursive")
        self.assertEqual(chunker.chunk_size, 500)
        self.assertEqual(chunker.overlap, 50)
        self.assertIsNone(chunker.client)

    def test_chunk_dispatches_the_default_strategy_to_recursive(self):
        chunker = DocumentChunker()
        with mock.patch.object(chunker, "recursive", return_value=["only"]) as recursive:
            self.assertEqual(chunker.chunk("some text"), ["only"])
        recursive.assert_called_once_with("some text")

    def test_unknown_strategy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "bogus"):
            DocumentChunker(strategy="bogus", chunk_size=20, overlap=5).chunk("some text")


@unittest.skipIf(DocumentChunker is None, f"common.chunker unavailable: {CHUNKER_IMPORT_ERROR}")
class FixedChunkingTests(unittest.TestCase):
    TEXT = "abcdefghijklmnopqrstuvwx"  # 24 characters

    def _chunker(self, chunk_size=10, overlap=3):
        return DocumentChunker(strategy="fixed", chunk_size=chunk_size, overlap=overlap)

    def test_slices_at_chunk_size_with_the_documented_overlap(self):
        chunks = self._chunker().chunk(self.TEXT)

        # stride = chunk_size - overlap = 7, so the windows are 0:10, 7:17,
        # 14:24 and the 21:24 remainder.
        self.assertEqual(chunks, ["abcdefghij", "hijklmnopq", "opqrstuvwx", "vwx"])
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 10)

    def test_consecutive_chunks_share_the_overlap_window(self):
        chunks = self._chunker().chunk(self.TEXT)

        for first, second in zip(chunks, chunks[1:]):
            self.assertEqual(first[-3:], second[:3])

    def test_chunks_reassemble_into_the_original_text(self):
        chunks = self._chunker().chunk(self.TEXT)

        rebuilt = chunks[0] + "".join(chunk[3:] for chunk in chunks[1:])
        self.assertEqual(rebuilt, self.TEXT)

    def test_text_shorter_than_chunk_size_stays_one_chunk(self):
        self.assertEqual(self._chunker().chunk("abc"), ["abc"])

    def test_empty_text_produces_no_chunks(self):
        self.assertEqual(self._chunker().chunk(""), [])

    def test_whitespace_only_text_is_kept_verbatim(self):
        # Unlike the paragraph strategy, fixed chunking slices the raw string and
        # never strips, so blank input becomes one blank chunk. Pinned because
        # callers that treat "chunks were produced" as "text was extracted" rely
        # on knowing which strategy filters and which does not.
        self.assertEqual(self._chunker().chunk("   \n  "), ["   \n  "])


@unittest.skipIf(DocumentChunker is None, f"common.chunker unavailable: {CHUNKER_IMPORT_ERROR}")
class ParagraphChunkingTests(unittest.TestCase):
    def _chunker(self, chunk_size=20, overlap=5):
        return DocumentChunker(strategy="paragraph", chunk_size=chunk_size, overlap=overlap)

    def test_merges_paragraphs_until_chunk_size_and_keeps_the_blank_line(self):
        text = "\n\n".join(["A" * 10, "B" * 10, "C" * 10])

        chunks = self._chunker(chunk_size=50).chunk(text)

        # The blank-line delimiter is re-inserted when paragraphs are merged, so
        # this is the assertion that catches a delimiter regression.
        self.assertEqual(chunks, ["A" * 10 + "\n\n" + "B" * 10 + "\n\n" + "C" * 10])

    def test_flushes_the_current_chunk_when_the_next_paragraph_overflows(self):
        text = "\n\n".join(["A" * 10, "B" * 10, "C" * 10])

        chunks = self._chunker(chunk_size=20).chunk(text)

        self.assertEqual(chunks, ["A" * 10, "B" * 10, "C" * 10])

    def test_falls_back_to_single_newlines_when_there_are_no_blank_lines(self):
        text = "line one\nline two\nline three"

        chunks = self._chunker(chunk_size=10, overlap=3).chunk(text)

        # With a '\n\n'-only delimiter the whole text would be one 28-character
        # paragraph and get sliced mid-line instead.
        self.assertEqual(chunks, ["line one", "line two", "line three"])

    def test_splits_a_paragraph_longer_than_chunk_size(self):
        chunks = self._chunker(chunk_size=20, overlap=5).chunk(LONG_PARAGRAPH)

        # Fixed-sliced with stride 15: 0:20, 15:35, 30:50 and the 45:50 tail.
        self.assertEqual(
            chunks,
            [
                LONG_PARAGRAPH[0:20],
                LONG_PARAGRAPH[15:35],
                LONG_PARAGRAPH[30:50],
                LONG_PARAGRAPH[45:50],
            ],
        )
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 20)

    def test_tail_of_an_oversized_paragraph_merges_with_the_next_paragraph(self):
        text = LONG_PARAGRAPH + "\n\ntail"

        chunks = self._chunker(chunk_size=20, overlap=5).chunk(text)

        self.assertEqual(
            chunks,
            [
                LONG_PARAGRAPH[0:20],
                LONG_PARAGRAPH[15:35],
                LONG_PARAGRAPH[30:50],
                LONG_PARAGRAPH[45:50] + "\n\ntail",
            ],
        )

    def test_blank_text_produces_no_chunks(self):
        chunker = self._chunker()

        self.assertEqual(chunker.chunk(""), [])
        self.assertEqual(chunker.chunk("  \n \n "), [])


@unittest.skipIf(DocumentChunker is None, f"common.chunker unavailable: {CHUNKER_IMPORT_ERROR}")
class RecursiveChunkingTests(unittest.TestCase):
    def setUp(self):
        # Unique fixed-width tokens: no token is a substring of another, so word
        # membership and ordering can be asserted exactly.
        self.words = [f"w{i:04d}" for i in range(400)]
        self.text = " ".join(self.words)  # 2399 characters
        self.chunker = DocumentChunker()  # recursive, 500/50

    def test_long_text_becomes_several_chunks_within_chunk_size(self):
        chunks = self.chunker.chunk(self.text)

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertTrue(chunk.strip())
            self.assertLessEqual(len(chunk), self.chunker.chunk_size)

    def test_every_word_survives_in_its_original_order(self):
        chunks = self.chunker.chunk(self.text)

        first_seen = {}
        for index, chunk in enumerate(chunks):
            for word in chunk.split():
                first_seen.setdefault(word, index)

        # dict preserves insertion order, so this asserts full coverage *and*
        # that the chunk sequence follows the source text.
        self.assertEqual(list(first_seen), self.words)

    def test_each_chunk_is_a_contiguous_slice_of_the_source(self):
        for chunk in self.chunker.chunk(self.text):
            self.assertIn(chunk, self.text)

    def test_text_shorter_than_chunk_size_stays_one_chunk(self):
        self.assertEqual(self.chunker.chunk("one two three"), ["one two three"])

    def test_empty_text_produces_no_chunks(self):
        self.assertEqual(self.chunker.chunk(""), [])

    def test_whitespace_only_text_produces_no_content(self):
        chunks = self.chunker.chunk("   \n\n \t ")

        self.assertEqual([chunk for chunk in chunks if chunk.strip()], [])


def _stub_embedding_client(vectors=(), error=None):
    """An OpenAI-shaped client returning canned vectors — never touches network."""
    calls = []

    def create(input, model):
        calls.append((list(input), model))
        if error is not None:
            raise error
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=list(vector)) for vector in vectors]
        )

    client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
    client.calls = calls
    return client


@unittest.skipIf(DocumentChunker is None, f"common.chunker unavailable: {CHUNKER_IMPORT_ERROR}")
class SemanticChunkingTests(unittest.TestCase):
    TEXT = "Cats purr. Cats nap. Bridges span rivers."
    SENTENCES = ["Cats purr.", "Cats nap.", "Bridges span rivers."]

    def _chunker(self, client, chunk_size=500):
        return DocumentChunker(
            strategy="semantic",
            chunk_size=chunk_size,
            overlap=5,
            embedding_client=client,
        )

    def test_requires_an_embedding_client(self):
        chunker = DocumentChunker(strategy="semantic", chunk_size=500, overlap=5)

        with self.assertRaisesRegex(ValueError, "embedding_client"):
            chunker.chunk(self.TEXT)

    def test_groups_similar_sentences_and_breaks_on_a_topic_shift(self):
        # Cosine similarity 1.0 between the first two vectors (above the 0.75
        # threshold) and 0.0 across the third.
        client = _stub_embedding_client(
            vectors=[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        )

        chunks = self._chunker(client).chunk(self.TEXT)

        self.assertEqual(chunks, ["Cats purr. Cats nap.", "Bridges span rivers."])
        self.assertEqual(client.calls, [(self.SENTENCES, "text-embedding-3-small")])

    def test_chunk_size_caps_a_group_of_similar_sentences(self):
        client = _stub_embedding_client(
            vectors=[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
        )

        chunks = self._chunker(client, chunk_size=1).chunk(self.TEXT)

        self.assertEqual(chunks, self.SENTENCES)

    def test_embedding_failure_degrades_to_one_chunk_per_sentence(self):
        client = _stub_embedding_client(error=RuntimeError("no network"))

        chunks = self._chunker(client).chunk(self.TEXT)

        self.assertEqual(chunks, self.SENTENCES)

    def test_empty_text_produces_no_chunks_without_calling_the_client(self):
        client = _stub_embedding_client(vectors=[[1.0, 0.0]])

        self.assertEqual(self._chunker(client).chunk(""), [])
        self.assertEqual(client.calls, [])


# ──────────────────────────────────────────────────────────────────────
# emitter/status.py
# ──────────────────────────────────────────────────────────────────────

@unittest.skipIf(status is None, f"emitter.status unavailable: {STATUS_IMPORT_ERROR}")
class StatusEmitterTests(unittest.TestCase):
    def test_forwards_stage_message_and_meta_to_the_callback(self):
        events = []
        emitter = status.StatusEmitter(events.append)

        self.assertIsNone(emitter.emit("dense_retrieval", "searching", {"top_k": 5}))
        self.assertEqual(
            events,
            [{"stage": "dense_retrieval", "message": "searching", "meta": {"top_k": 5}}],
        )

    def test_event_carries_exactly_the_three_documented_keys(self):
        events = []
        status.StatusEmitter(events.append).emit("pipeline_start", "starting")

        # router/consumers.py relays this dict straight to the browser.
        self.assertEqual(set(events[0]), {"stage", "message", "meta"})

    def test_missing_or_falsy_meta_becomes_an_empty_dict(self):
        events = []
        emitter = status.StatusEmitter(events.append)

        emitter.emit("pipeline_start", "no meta argument")
        emitter.emit("degraded", "explicit None", None)
        emitter.emit("degraded", "explicit empty", {})

        self.assertEqual([event["meta"] for event in events], [{}, {}, {}])

    def test_callback_failure_is_swallowed_and_logged(self):
        emitter = status.StatusEmitter(mock.Mock(side_effect=RuntimeError("redis down")))

        with self.assertLogs(status.logger, level="WARNING") as captured:
            self.assertIsNone(emitter.emit("hybrid_rerank", "reranking"))

        logged = "\n".join(captured.output)
        self.assertIn("[StatusEmitter]", logged)
        self.assertIn("hybrid_rerank", logged)
        self.assertIn("redis down", logged)

    def test_a_failed_emit_does_not_disable_later_emits(self):
        seen = []

        def flaky(event):
            seen.append(event["stage"])
            if len(seen) == 1:
                raise ConnectionError("broker blip")

        emitter = status.StatusEmitter(flaky)

        with self.assertLogs(status.logger, level="WARNING"):
            emitter.emit("first", "boom")
        emitter.emit("second", "still publishing")

        self.assertEqual(seen, ["first", "second"])

    def test_null_emitter_is_a_silent_no_op(self):
        self.assertIsInstance(status.NULL_EMITTER, status.StatusEmitter)
        self.assertIsNone(status.NULL_EMITTER._callback)

        with mock.patch.object(status.logger, "warning") as warning:
            self.assertIsNone(
                status.NULL_EMITTER.emit("retrieval_empty", "nothing found", {"n": 0})
            )

        warning.assert_not_called()


# ──────────────────────────────────────────────────────────────────────
# common/schema.py
# ──────────────────────────────────────────────────────────────────────

@unittest.skipIf(bool(SCHEMA_SKIP_REASON), SCHEMA_SKIP_REASON)
class RAGResponseEnvelopeTests(unittest.TestCase):
    def _body(self, response):
        self.assertEqual(response["Content-Type"], "application/json")
        return json.loads(response.content)

    def test_success_envelopes_carry_the_payload_and_a_fixed_message(self):
        payload = {"answer": "hi", "degraded": []}

        for method, code, message in (
            ("response_200", 200, "Response generated successfully"),
            ("response_201", 201, "Object created successfully"),
            ("response_202", 202, "Chat initialization started."),
        ):
            with self.subTest(method=method):
                response = getattr(schema.RAGResponse, method)(payload)

                self.assertEqual(response.status_code, code)
                body = self._body(response)
                self.assertEqual(body["status"], code)
                self.assertEqual(body["message"], message)
                self.assertEqual(body["data"], payload)

    def test_error_envelopes_report_the_error_and_null_data(self):
        # Called on the class, not on get_responses(): response_404 is missing
        # @staticmethod, so an instance call binds the instance to `error` —
        # router/tests.py pins that defect at the view level.
        for method, code in (
            ("response_400", 400),
            ("response_404", 404),
            ("response_500", 500),
        ):
            with self.subTest(method=method):
                response = getattr(schema.RAGResponse, method)("collection not found")

                self.assertEqual(response.status_code, code)
                body = self._body(response)
                self.assertEqual(body["status"], code)
                self.assertEqual(body["message"], "collection not found")
                self.assertIsNone(body["data"])

    def test_envelope_has_exactly_four_keys_and_a_current_timestamp(self):
        body = self._body(schema.RAGResponse.response_200("Chunk Created"))

        self.assertEqual(set(body), {"status", "message", "timestamp", "data"})
        self.assertIsInstance(body["timestamp"], float)
        self.assertLess(abs(body["timestamp"] - time.time()), 300)

    def test_string_payloads_are_passed_through_unchanged(self):
        body = self._body(schema.RAGResponse.response_200("Chunk Created"))

        self.assertEqual(body["data"], "Chunk Created")

    def test_get_responses_returns_a_memoised_instance(self):
        schema._responses = None
        self.addCleanup(setattr, schema, "_responses", None)

        first = schema.get_responses()

        self.assertIsInstance(first, schema.RAGResponse)
        self.assertIs(schema.get_responses(), first)


# ──────────────────────────────────────────────────────────────────────
# utils/helper.py
# ──────────────────────────────────────────────────────────────────────

@unittest.skipIf(helper is None, f"utils.helper unavailable: {HELPER_IMPORT_ERROR}")
class PathHelperTests(unittest.TestCase):
    def test_base_path_namespaces_the_document_under_the_user(self):
        self.assertRegex(
            helper._document_base_path("alice"),
            r"^documents/user_alice/[0-9a-f]{32}$",
        )

    def test_base_path_is_a_relative_three_segment_key(self):
        path = helper._document_base_path("bob")

        self.assertFalse(path.startswith("/"))
        segments = path.split("/")
        self.assertEqual(len(segments), 3)
        self.assertEqual(segments[:2], ["documents", "user_bob"])

    def test_base_path_is_unique_per_call(self):
        # utils/insert_file.py joins the uploaded filename onto this prefix, so
        # two uploads of the same name by the same user must not collide.
        paths = {helper._document_base_path("alice") for _ in range(5)}

        self.assertEqual(len(paths), 5)

    def test_base_path_document_id_is_a_uuid4(self):
        doc_id = helper._document_base_path("alice").rsplit("/", 1)[1]

        self.assertEqual(uuid.UUID(doc_id).version, 4)

    def test_conversation_ids_are_unique_uuid4_hex_strings(self):
        ids = [helper.conversation_id_generator() for _ in range(5)]

        for value in ids:
            self.assertRegex(value, r"^[0-9a-f]{32}$")
            self.assertEqual(uuid.UUID(value).version, 4)
        self.assertEqual(len(set(ids)), 5)


if __name__ == "__main__":
    unittest.main()
