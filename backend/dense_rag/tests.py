"""Tests for DenseRAG: construction guards, embedding batching and the
retrieval fallbacks that keep AppRAGPipeline alive when ChromaDB or the
embedding API is unreachable.

Everything here runs with no network, no API key and no ChromaDB server:
DenseRAG.__init__ is only ever entered with dense_rag.dense_rag.OpenAI and
dense_rag.dense_rag.get_chroma_client patched, plus a fake OPENROUTER_API_KEY.
"""

import os
import unittest
from unittest import mock

try:
    from dense_rag import dense_rag as dense_rag_module
    from dense_rag.dense_rag import DenseRAG
    IMPORT_ERROR = ""
except Exception as exc:  # needs Django configured (chroma -> router.models) + sklearn
    dense_rag_module = None
    DenseRAG = None
    IMPORT_ERROR = str(exc)


API_KEY_ENV = {"OPENROUTER_API_KEY": "test-key"}

# Sentinel so tests can deliberately hand back a None payload, which is one of
# the malformed shapes retrieve() has to survive.
_UNSET = object()


def _collection(count=0, query_result=_UNSET, get_result=_UNSET, name="test_collection"):
    """A stand-in for a chromadb collection object."""
    collection = mock.Mock()
    collection.name = name
    collection.count.return_value = count
    collection.query.return_value = (
        {"documents": [[]], "metadatas": [[]]} if query_result is _UNSET else query_result
    )
    collection.get.return_value = (
        {"documents": []} if get_result is _UNSET else get_result
    )
    return collection


def _make_dense_rag(config=None, collection=None):
    """Construct DenseRAG with the OpenAI client and Chroma lookup stubbed out."""
    full_config = {"collection_name": "test_collection", "top_k": 5}
    full_config.update(config or {})
    stub_collection = collection if collection is not None else _collection()

    with mock.patch.dict(os.environ, API_KEY_ENV), mock.patch.object(
        dense_rag_module, "OpenAI"
    ), mock.patch.object(
        dense_rag_module, "get_chroma_client", return_value=stub_collection
    ):
        return DenseRAG(full_config)


def _embedding_response(vectors):
    """Mimic openai's embeddings.create response shape."""
    response = mock.Mock()
    response.data = [mock.Mock(embedding=vector) for vector in vectors]
    return response


@unittest.skipIf(DenseRAG is None, f"dense_rag unavailable: {IMPORT_ERROR}")
class InitTests(unittest.TestCase):
    def test_init_rejects_an_empty_api_key(self):
        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}), mock.patch.object(
            dense_rag_module, "OpenAI"
        ) as openai_cls, mock.patch.object(
            dense_rag_module, "get_chroma_client"
        ) as get_client:
            with self.assertRaises(ValueError) as ctx:
                DenseRAG({})

        self.assertIn("OPENROUTER_API_KEY", str(ctx.exception))
        # The guard must fire before any client/collection is built.
        openai_cls.assert_not_called()
        get_client.assert_not_called()

    def test_init_rejects_an_absent_api_key(self):
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
            dense_rag_module, "OpenAI"
        ), mock.patch.object(dense_rag_module, "get_chroma_client"):
            with self.assertRaises(ValueError) as ctx:
                DenseRAG({})

        self.assertIn("OPENROUTER_API_KEY", str(ctx.exception))

    def test_init_builds_openrouter_client_and_resolves_the_collection(self):
        stub_collection = _collection(name="my_corpus")

        with mock.patch.dict(os.environ, API_KEY_ENV), mock.patch.object(
            dense_rag_module, "OpenAI"
        ) as openai_cls, mock.patch.object(
            dense_rag_module, "get_chroma_client", return_value=stub_collection
        ) as get_client:
            rag = DenseRAG(
                {
                    "collection_name": "my_corpus",
                    "embedding_model": "openai/some-embedder",
                    "top_k": 3,
                }
            )

        openai_cls.assert_called_once_with(
            base_url="https://openrouter.ai/api/v1", api_key="test-key"
        )
        get_client.assert_called_once_with(collection_name="my_corpus")
        self.assertIs(rag.client, openai_cls.return_value)
        self.assertIs(rag.collection, stub_collection)
        self.assertEqual(rag.embedding_model, "openai/some-embedder")
        self.assertEqual(rag.top_k, 3)
        self.assertEqual(rag.documents, [])

    def test_init_falls_back_to_default_model_and_collection(self):
        with mock.patch.dict(os.environ, API_KEY_ENV), mock.patch.object(
            dense_rag_module, "OpenAI"
        ), mock.patch.object(
            dense_rag_module, "get_chroma_client", return_value=_collection()
        ) as get_client:
            rag = DenseRAG({})

        get_client.assert_called_once_with(collection_name="default_corpus")
        self.assertEqual(rag.embedding_model, "openai/text-embedding-3-small")
        self.assertEqual(rag.top_k, 5)

    def test_emitter_defaults_to_the_no_op_emitter(self):
        rag = _make_dense_rag()

        self.assertIs(rag.emitter, dense_rag_module.NULL_EMITTER)
        # The default must be callable without a Redis callback attached.
        self.assertIsNone(rag.emitter.emit("dense_retrieval", "hello"))

    def test_set_emitter_replaces_the_default(self):
        rag = _make_dense_rag()
        emitter = mock.Mock()

        rag.set_emitter(emitter)

        self.assertIs(rag.emitter, emitter)


@unittest.skipIf(DenseRAG is None, f"dense_rag unavailable: {IMPORT_ERROR}")
class GetEmbeddingsTests(unittest.TestCase):
    def setUp(self):
        self.rag = _make_dense_rag({"embedding_model": "openai/some-embedder"})
        self.rag.client = mock.Mock()
        self.batches = []

    def _record_batches(self, **kwargs):
        batch = list(kwargs["input"])
        self.batches.append(batch)
        return _embedding_response([[0.1, 0.2]] * len(batch))

    def test_batches_input_by_batch_size(self):
        self.rag.client.embeddings.create.side_effect = self._record_batches

        embeddings = self.rag._get_embeddings([f"doc {i}" for i in range(250)], batch_size=100)

        self.assertEqual(self.rag.client.embeddings.create.call_count, 3)
        self.assertEqual([len(batch) for batch in self.batches], [100, 100, 50])
        self.assertEqual(len(embeddings), 250)

    def test_sends_the_configured_model_and_float_encoding(self):
        self.rag.client.embeddings.create.return_value = _embedding_response([[0.1]])

        self.rag._get_embeddings(["only one"])

        self.rag.client.embeddings.create.assert_called_once_with(
            input=["only one"],
            model="openai/some-embedder",
            encoding_format="float",
        )

    def test_strips_newlines_from_inputs(self):
        self.rag.client.embeddings.create.side_effect = self._record_batches

        self.rag._get_embeddings(["first\nsecond\nthird"])

        self.assertEqual(self.batches, [["first second third"]])

    def test_coerces_non_text_inputs_positionally(self):
        # Callers align embeddings with their input by position, so a bad entry
        # must become a placeholder rather than shift everything after it.
        self.rag.client.embeddings.create.side_effect = self._record_batches

        embeddings = self.rag._get_embeddings(["real text", None, 42])

        self.assertEqual(self.batches, [["real text", "", "42"]])
        self.assertEqual(len(embeddings), 3)

    def test_skips_a_batch_whose_api_call_raises_and_keeps_the_rest(self):
        self.rag.client.embeddings.create.side_effect = [
            _embedding_response([[0.1]]),
            ConnectionError("openrouter down"),
            _embedding_response([[0.3]]),
        ]

        embeddings = self.rag._get_embeddings(["a", "b", "c"], batch_size=1)

        self.assertEqual(embeddings, [[0.1], [0.3]])

    def test_returns_empty_when_every_batch_fails(self):
        self.rag.client.embeddings.create.side_effect = ConnectionError("no network")

        self.assertEqual(self.rag._get_embeddings(["a", "b"], batch_size=1), [])

    def test_returns_empty_when_the_response_carries_no_data(self):
        self.rag.client.embeddings.create.return_value = _embedding_response([])

        self.assertEqual(self.rag._get_embeddings(["a"]), [])

    def test_returns_empty_for_empty_input_without_calling_the_api(self):
        for texts in ([], None):
            with self.subTest(texts=texts):
                self.rag.client.embeddings.create.reset_mock()

                self.assertEqual(self.rag._get_embeddings(texts), [])
                self.rag.client.embeddings.create.assert_not_called()

    def test_non_positive_batch_size_falls_back_to_a_single_batch(self):
        # app_pipeline passes min(len(texts), 50), which is 0 for an empty-ish
        # corpus, and range() refuses a step of 0.
        self.rag.client.embeddings.create.side_effect = self._record_batches

        embeddings = self.rag._get_embeddings(["a", "b", "c"], batch_size=0)

        self.assertEqual(self.rag.client.embeddings.create.call_count, 1)
        self.assertEqual(self.batches, [["a", "b", "c"]])
        self.assertEqual(len(embeddings), 3)


@unittest.skipIf(DenseRAG is None, f"dense_rag unavailable: {IMPORT_ERROR}")
class RetrieveTests(unittest.TestCase):
    def _rag(self, collection, top_k=5):
        rag = _make_dense_rag({"top_k": top_k}, collection=collection)
        return rag

    @staticmethod
    def _stub_embeddings(rag, value=None):
        """Patch the query-embedding call at its point of use."""
        return mock.patch.object(
            rag,
            "_get_embeddings",
            return_value=[[0.1, 0.2]] if value is None else value,
        )

    def test_empty_collection_returns_nothing_without_querying(self):
        collection = _collection(count=0)
        rag = self._rag(collection)

        with self._stub_embeddings(rag):
            self.assertEqual(rag.retrieve("who wrote it?"), ([], []))

        collection.query.assert_not_called()

    def test_returns_nothing_when_embedding_yields_no_vector(self):
        collection = _collection(count=4)
        rag = self._rag(collection)

        with self._stub_embeddings(rag, []):
            self.assertEqual(rag.retrieve("who wrote it?"), ([], []))

        collection.query.assert_not_called()

    def test_clamps_n_results_to_the_collection_count(self):
        collection = _collection(
            count=2,
            query_result={
                "documents": [["chunk one", "chunk two"]],
                "metadatas": [[{"url": "u1"}, {"url": "u2"}]],
            },
        )
        rag = self._rag(collection, top_k=5)

        with self._stub_embeddings(rag):
            docs, metas = rag.retrieve("who wrote it?")

        collection.query.assert_called_once_with(
            query_embeddings=[[0.1, 0.2]],
            n_results=2,
            where=None,
            include=["documents", "metadatas"],
        )
        self.assertEqual(docs, ["chunk one", "chunk two"])
        self.assertEqual(metas, [{"url": "u1"}, {"url": "u2"}])

    def test_passes_where_filter_through_to_collection_query(self):
        where_filter = {"document_id": 7}
        collection = _collection(
            count=10,
            get_result={"documents": ["d1", "d2", "d3"]},
            query_result={
                "documents": [["chunk one"]],
                "metadatas": [[{"document_id": 7}]],
            },
        )
        rag = self._rag(collection, top_k=5)

        with self._stub_embeddings(rag):
            docs, _ = rag.retrieve("who wrote it?", where_filter=where_filter)

        collection.get.assert_called_once_with(where=where_filter)
        collection.query.assert_called_once_with(
            query_embeddings=[[0.1, 0.2]],
            # n_results is clamped to the filtered count, not the whole collection
            n_results=3,
            where=where_filter,
            include=["documents", "metadatas"],
        )
        self.assertEqual(docs, ["chunk one"])

    def test_drops_where_filter_when_it_matches_no_documents(self):
        # An over-narrow filter must degrade to a whole-collection search
        # instead of returning nothing at all.
        for get_result in ({"documents": []}, {}, None):
            with self.subTest(get_result=get_result):
                collection = _collection(
                    count=4,
                    get_result=get_result,
                    query_result={
                        "documents": [["chunk one", "chunk two"]],
                        "metadatas": [[{}, {}]],
                    },
                )
                rag = self._rag(collection, top_k=2)

                with self._stub_embeddings(rag):
                    docs, _ = rag.retrieve("who wrote it?", where_filter={"url": "nope"})

                collection.query.assert_called_once_with(
                    query_embeddings=[[0.1, 0.2]],
                    n_results=2,
                    where=None,
                    include=["documents", "metadatas"],
                )
                self.assertEqual(len(docs), 2)

    def test_returns_nothing_when_top_k_leaves_no_results_to_ask_for(self):
        collection = _collection(count=4)
        rag = self._rag(collection, top_k=0)

        with self._stub_embeddings(rag):
            self.assertEqual(rag.retrieve("who wrote it?"), ([], []))

        collection.query.assert_not_called()

    def test_returns_empty_when_the_collection_count_raises(self):
        collection = _collection(count=3)
        collection.count.side_effect = RuntimeError("chroma unreachable")
        rag = self._rag(collection)

        with self._stub_embeddings(rag):
            self.assertEqual(rag.retrieve("who wrote it?"), ([], []))

        collection.query.assert_not_called()

    def test_returns_empty_when_the_collection_query_raises(self):
        collection = _collection(count=3)
        collection.query.side_effect = RuntimeError("chroma unreachable")
        rag = self._rag(collection)

        with self._stub_embeddings(rag):
            self.assertEqual(rag.retrieve("who wrote it?"), ([], []))

    def test_returns_empty_when_embedding_raises(self):
        # AppRAGPipeline treats an empty dense side as survivable, an exception
        # as a degradation of the whole query.
        collection = _collection(count=3)
        rag = self._rag(collection)

        with mock.patch.object(
            rag, "_get_embeddings", side_effect=RuntimeError("embedding api down")
        ):
            self.assertEqual(rag.retrieve("who wrote it?"), ([], []))

    def test_pads_and_coerces_metadata_to_match_documents(self):
        collection = _collection(
            count=3,
            query_result={
                "documents": [["chunk one", "chunk two", "chunk three"]],
                "metadatas": [[{"url": "u1"}, "not-a-dict"]],
            },
        )
        rag = self._rag(collection, top_k=3)

        with self._stub_embeddings(rag):
            docs, metas = rag.retrieve("who wrote it?")

        self.assertEqual(len(docs), 3)
        self.assertEqual(metas, [{"url": "u1"}, {}, {}])

    def test_malformed_query_payload_reads_as_no_results(self):
        malformed = [
            None,
            {},
            {"documents": None},
            {"documents": [[]]},
            {"documents": [None]},
            # not nested one row per query embedding
            {"documents": ["chunk one", "chunk two"]},
        ]
        for query_result in malformed:
            with self.subTest(query_result=query_result):
                collection = _collection(count=3, query_result=query_result)
                rag = self._rag(collection, top_k=3)

                with self._stub_embeddings(rag):
                    self.assertEqual(rag.retrieve("who wrote it?"), ([], []))

    def test_emits_one_dense_retrieval_status_event(self):
        collection = _collection(count=0)
        rag = self._rag(collection)
        emitter = mock.Mock()
        rag.set_emitter(emitter)

        with self._stub_embeddings(rag):
            rag.retrieve("who wrote it?")

        emitter.emit.assert_called_once_with("dense_retrieval", mock.ANY)

    def test_a_non_string_query_does_not_raise(self):
        # The logging path formats the query, so it must survive None.
        collection = _collection(count=3)
        rag = self._rag(collection)
        rag.client = mock.Mock()
        rag.client.embeddings.create.return_value = _embedding_response([])

        self.assertEqual(rag.retrieve(None), ([], []))
        collection.query.assert_not_called()

    def test_a_bare_instance_without_an_emitter_still_retrieves(self):
        # hybrid_rag/tests.py builds retrievers with object.__new__ to skip the
        # model/Chroma work; retrieve() must not need set_emitter() first.
        rag = object.__new__(DenseRAG)
        rag.embedding_model = "openai/some-embedder"
        rag.top_k = 2
        rag.collection = _collection(
            count=2,
            query_result={
                "documents": [["chunk one", "chunk two"]],
                "metadatas": [[{"url": "u1"}, {"url": "u2"}]],
            },
        )
        self.assertFalse(hasattr(rag, "emitter"))

        with self._stub_embeddings(rag):
            docs, metas = rag.retrieve("who wrote it?")

        self.assertEqual(docs, ["chunk one", "chunk two"])
        self.assertEqual(len(metas), 2)


@unittest.skipIf(DenseRAG is None, f"dense_rag unavailable: {IMPORT_ERROR}")
class CollectionSwapTests(unittest.TestCase):
    def test_set_collection_swaps_the_collection_and_records_the_name(self):
        rag = _make_dense_rag()
        new_collection = _collection(name="user_42_collection")

        with mock.patch.object(
            dense_rag_module, "get_chroma_client", return_value=new_collection
        ) as get_client:
            rag.set_collection("user_42_collection")

        get_client.assert_called_once_with(collection_name="user_42_collection")
        self.assertIs(rag.collection, new_collection)
        self.assertEqual(rag.collection_name, "user_42_collection")

    def test_set_collection_propagates_a_chroma_failure(self):
        # AppRAGPipeline detects the "dense_retrieval" degradation from this
        # raise — swallowing it here would hide a dead ChromaDB.
        rag = _make_dense_rag()
        original = rag.collection

        with mock.patch.object(
            dense_rag_module, "get_chroma_client", side_effect=RuntimeError("chroma down")
        ):
            with self.assertRaises(RuntimeError):
                rag.set_collection("user_42_collection")

        self.assertIs(rag.collection, original)


if __name__ == "__main__":
    unittest.main()
