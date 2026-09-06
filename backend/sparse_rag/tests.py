"""Tests for SparseRAG: tokenization, lazy BM25 index loading and retrieval.

Covers _tokenize (lower-casing, punctuation stripping, stop-word handling and
non-string input), _load_index_from_chroma (empty collection, index build from
collection.get(), caching, ChromaDB failure) and retrieve (nothing indexed,
BM25 ranking with the real rank_bm25, top_k, keyword preference over the query,
the where_filter branch, docs/metas alignment) plus the no-op emitter default
that lets retrieve run before set_emitter().

No network, no API key, no ChromaDB server: get_chroma_client is patched and a
hand-written fake collection stands in for the chromadb collection. BM25Okapi
is used for real - it is pure python.
"""

import unittest
from unittest import mock

try:
    from common.runtime import context as runtime_context
    from sparse_rag import sparse_rag as sparse_rag_module
    from sparse_rag.sparse_rag import SparseRAG
    IMPORT_ERROR = ""
except Exception as exc:  # needs Django settings + chromadb + nltk + rank_bm25
    sparse_rag_module = None
    SparseRAG = None
    IMPORT_ERROR = str(exc)


# A corpus where each query below matches exactly one document, so BM25 gives
# that document the only non-zero score and the ranking assertion is stable.
DOCS = [
    "the cat sat on the mat in the kitchen",
    "photosynthesis converts sunlight into chemical energy inside plants",
    "the stock market closed lower on tuesday after the report",
]
METAS = [
    {"idx": 0, "source": "animals"},
    {"idx": 1, "source": "biology"},
    {"idx": 2, "source": "finance"},
]
BIOLOGY_QUERY = "photosynthesis sunlight plants"
FINANCE_QUERY = "stock market tuesday"


class FakeCollection:
    """Stand-in for a chromadb collection, exposing only count() and get().

    Hand-written rather than a MagicMock on purpose: MagicMock().get(...) returns
    a truthy mock whose "documents" is also a mock, which _clean_documents turns
    into [] - the tests would silently run the "reported documents but returned
    none" path while looking like they exercised the happy path.
    """

    def __init__(self, documents=None, metadatas=None, name="sparse_rag_bm25"):
        self.documents = list(documents) if documents is not None else []
        self.metadatas = list(metadatas) if metadatas is not None else []
        self.name = name
        self.count_calls = 0
        self.get_calls = []
        self.count_error = None
        self.get_error = None

    def count(self):
        self.count_calls += 1
        if self.count_error is not None:
            raise self.count_error
        return len(self.documents)

    def get(self, where=None, **kwargs):
        self.get_calls.append(where)
        if self.get_error is not None:
            raise self.get_error
        if where is None:
            return {
                "documents": list(self.documents),
                "metadatas": list(self.metadatas),
            }
        documents, metadatas = [], []
        for doc, meta in zip(self.documents, self.metadatas):
            if isinstance(meta, dict) and all(
                meta.get(key) == value for key, value in where.items()
            ):
                documents.append(doc)
                metadatas.append(meta)
        return {"documents": documents, "metadatas": metadatas}


def _make_rag(collection, **config):
    """Build a SparseRAG without a ChromaDB server.

    remove_stop_words defaults to False so the retrieval tests never depend on
    the NLTK stopwords corpus being present in the image.
    """
    cfg = {"remove_stop_words": False}
    cfg.update(config)
    with mock.patch.object(
        sparse_rag_module, "get_chroma_client", return_value=collection
    ):
        return SparseRAG(cfg)


def _corpus_rag(**config):
    collection = FakeCollection(DOCS, METAS)
    return _make_rag(collection, **config), collection


@unittest.skipIf(SparseRAG is None, f"sparse_rag unavailable: {IMPORT_ERROR}")
class TokenizeTests(unittest.TestCase):
    def test_tokenize_lowercases_and_strips_punctuation(self):
        rag, _ = _corpus_rag()

        self.assertEqual(
            rag._tokenize("Hello, WORLD! It's 2024."),
            ["hello", "world", "its", "2024"],
        )
        # The regex strips separators rather than splitting on them, so a
        # hyphenated phrase collapses into a single token.
        self.assertEqual(rag._tokenize("state-of-the-art"), ["stateoftheart"])

    def test_tokenize_removes_english_stopwords_when_enabled(self):
        with mock.patch.object(sparse_rag_module, "stopwords") as stopwords:
            stopwords.words.return_value = ["the", "is", "on", "a", "in"]
            rag, _ = _corpus_rag(remove_stop_words=True)

            self.assertEqual(rag.stop_words, {"the", "is", "on", "a", "in"})
            self.assertEqual(rag._tokenize("The cat is on a mat"), ["cat", "mat"])

        # Read from NLTK once and cached, however many times it is used.
        stopwords.words.assert_called_once_with("english")

    def test_tokenize_keeps_stopwords_when_disabled(self):
        with mock.patch.object(sparse_rag_module, "stopwords") as stopwords:
            rag, _ = _corpus_rag(remove_stop_words=False)

        stopwords.words.assert_not_called()
        self.assertEqual(rag.stop_words, set())
        self.assertEqual(
            rag._tokenize("The cat is on a mat"),
            ["the", "cat", "is", "on", "a", "mat"],
        )

    def test_tokenize_returns_empty_list_for_non_string_input(self):
        rag, _ = _corpus_rag()

        # ChromaDB can hand back None documents and callers tokenize raw queries.
        self.assertEqual(rag._tokenize(None), [])
        self.assertEqual(rag._tokenize(42), [])
        self.assertEqual(rag._tokenize(b"bytes"), [])
        self.assertEqual(rag._tokenize({"text": "hi"}), [])


@unittest.skipIf(SparseRAG is None, f"sparse_rag unavailable: {IMPORT_ERROR}")
class ConstructionTests(unittest.TestCase):
    def test_collection_name_from_config_is_passed_to_chroma_client(self):
        collection = FakeCollection(DOCS, METAS)
        with mock.patch.object(
            sparse_rag_module, "get_chroma_client", return_value=collection
        ) as get_client:
            rag = SparseRAG({"collection_name": "user_collection_7", "top_k": 3})

        get_client.assert_called_once_with(collection_name="user_collection_7")
        self.assertIs(rag.collection, collection)
        self.assertEqual(rag.top_k, 3)

    def test_emitter_defaults_to_null_emitter(self):
        rag, _ = _corpus_rag()

        self.assertIs(rag.emitter, sparse_rag_module.NULL_EMITTER)

    def test_construction_survives_missing_stopword_corpus(self):
        # A missing NLTK corpus must cost stop-word removal, not the whole app.
        # The list is read on first use rather than in __init__ (a request can
        # switch removal on that the deployment configured off), so the patch
        # has to cover the read too.
        with mock.patch.object(sparse_rag_module, "stopwords") as stopwords:
            stopwords.words.side_effect = LookupError("Resource stopwords not found")
            rag, _ = _corpus_rag(remove_stop_words=True)

            self.assertEqual(rag.stop_words, set())
            self.assertEqual(rag._tokenize("the cat"), ["the", "cat"])


@unittest.skipIf(SparseRAG is None, f"sparse_rag unavailable: {IMPORT_ERROR}")
class LoadIndexFromChromaTests(unittest.TestCase):
    def test_returns_false_for_empty_collection(self):
        collection = FakeCollection([])
        rag = _make_rag(collection)

        self.assertFalse(rag._load_index_from_chroma())
        self.assertFalse(rag._index_loaded)
        self.assertIsNone(rag.bm25)
        # The empty count must short-circuit before any document read.
        self.assertEqual(collection.get_calls, [])

    def test_builds_documents_metadatas_and_bm25_from_collection_get(self):
        # A None document and a metadata list two entries short: both are things
        # ChromaDB really returns, and both must be normalised, not raise.
        collection = FakeCollection(
            ["alpha beta gamma", None, "delta epsilon"],
            [{"idx": 0}],
        )
        rag = _make_rag(collection)

        self.assertTrue(rag._load_index_from_chroma())
        self.assertTrue(rag._index_loaded)
        self.assertEqual(rag.documents, ["alpha beta gamma", "", "delta epsilon"])
        self.assertEqual(rag.metadatas, [{"idx": 0}, {}, {}])
        self.assertEqual(len(rag.tokenized_corpus), 3)
        self.assertEqual(rag.tokenized_corpus[0], ["alpha", "beta", "gamma"])
        self.assertEqual(rag.tokenized_corpus[1], [])
        self.assertIsInstance(rag.bm25, sparse_rag_module.BM25Okapi)

    def test_second_call_is_cached_and_does_not_reread_the_collection(self):
        rag, collection = _corpus_rag()

        self.assertTrue(rag._load_index_from_chroma())
        self.assertEqual(len(collection.get_calls), 1)

        self.assertTrue(rag._load_index_from_chroma())
        self.assertEqual(len(collection.get_calls), 1)

    def test_returns_false_instead_of_raising_when_collection_fails(self):
        counting_fails = FakeCollection(DOCS, METAS)
        counting_fails.count_error = RuntimeError("chroma unreachable")
        rag = _make_rag(counting_fails)

        self.assertFalse(rag._load_index_from_chroma())
        self.assertFalse(rag._index_loaded)

        reading_fails = FakeCollection(DOCS, METAS)
        reading_fails.get_error = RuntimeError("chroma unreachable")
        rag = _make_rag(reading_fails)

        self.assertFalse(rag._load_index_from_chroma())
        self.assertFalse(rag._index_loaded)
        # A failed load leaves the index re-tryable rather than half-built.
        self.assertIsNone(rag.bm25)


@unittest.skipIf(SparseRAG is None, f"sparse_rag unavailable: {IMPORT_ERROR}")
class RetrieveTests(unittest.TestCase):
    def test_returns_empty_lists_when_nothing_is_indexed(self):
        collection = FakeCollection([])
        rag = _make_rag(collection)

        self.assertEqual(rag.retrieve(BIOLOGY_QUERY), ([], []))
        self.assertEqual(collection.get_calls, [])

    def test_ranks_the_relevant_document_first(self):
        rag, _ = _corpus_rag(top_k=3)

        documents, metadatas = rag.retrieve(BIOLOGY_QUERY)

        self.assertEqual(documents[0], DOCS[1])
        self.assertEqual(metadatas[0], METAS[1])

    def test_respects_top_k(self):
        rag, _ = _corpus_rag(top_k=2)

        documents, metadatas = rag.retrieve(BIOLOGY_QUERY)

        self.assertEqual(len(documents), 2)
        self.assertEqual(len(metadatas), 2)
        self.assertEqual(documents[0], DOCS[1])

    def test_top_k_larger_than_the_corpus_returns_the_whole_corpus(self):
        rag, _ = _corpus_rag(top_k=50)

        documents, metadatas = rag.retrieve(BIOLOGY_QUERY)

        self.assertEqual(len(documents), len(DOCS))
        self.assertEqual(len(metadatas), len(DOCS))

    def test_top_k_of_zero_returns_empty_lists_not_the_whole_corpus(self):
        # scores.argsort()[-0:] is the entire corpus reversed, so a misconfigured
        # top_k must bail out instead of flooding the pipeline.
        rag, _ = _corpus_rag(top_k=0)

        self.assertEqual(rag.retrieve(BIOLOGY_QUERY), ([], []))

    def test_keyword_is_preferred_over_the_query_when_supplied(self):
        rag, _ = _corpus_rag(top_k=1)

        with_keyword, _ = rag.retrieve(FINANCE_QUERY, keyword=BIOLOGY_QUERY)
        self.assertEqual(with_keyword, [DOCS[1]])

        # Same query without a keyword ranks the finance document, so the result
        # above came from the keyword and not from the corpus by coincidence.
        without_keyword, _ = rag.retrieve(FINANCE_QUERY)
        self.assertEqual(without_keyword, [DOCS[2]])

    def test_where_filter_restricts_the_corpus_to_matching_documents(self):
        rag, collection = _corpus_rag(top_k=5)

        # The query matches the finance document, but the filter only admits the
        # biology one - the filtered corpus, not the full index, must be scored.
        documents, metadatas = rag.retrieve(
            FINANCE_QUERY, where_filter={"source": "biology"}
        )

        self.assertEqual(documents, [DOCS[1]])
        self.assertEqual(metadatas, [METAS[1]])
        self.assertEqual(collection.get_calls[-1], {"source": "biology"})

    def test_where_filter_matching_nothing_returns_empty_lists(self):
        rag, collection = _corpus_rag(top_k=5)

        self.assertEqual(
            rag.retrieve(BIOLOGY_QUERY, where_filter={"source": "nonexistent"}),
            ([], []),
        )
        self.assertEqual(collection.get_calls[-1], {"source": "nonexistent"})

    def test_returns_equal_length_corresponding_documents_and_metadatas(self):
        rag, _ = _corpus_rag(top_k=3)

        documents, metadatas = rag.retrieve(BIOLOGY_QUERY)

        self.assertEqual(len(documents), len(metadatas))
        self.assertEqual(len(documents), len(DOCS))
        for document, meta in zip(documents, metadatas):
            # Each meta must still describe the document at its own position.
            self.assertEqual(DOCS[meta["idx"]], document)

    def test_collection_emptied_after_the_index_was_cached_returns_empty(self):
        rag, collection = _corpus_rag(top_k=3)

        documents, _ = rag.retrieve(BIOLOGY_QUERY)
        self.assertEqual(documents[0], DOCS[1])

        collection.documents = []
        collection.metadatas = []

        self.assertEqual(rag.retrieve(BIOLOGY_QUERY), ([], []))

    def test_returns_empty_lists_instead_of_raising_when_chromadb_dies(self):
        rag, collection = _corpus_rag(top_k=3)

        documents, _ = rag.retrieve(BIOLOGY_QUERY)
        self.assertEqual(documents[0], DOCS[1])

        # The index is cached, so the next call fails on the redundant count().
        collection.count_error = RuntimeError("chroma unreachable")

        self.assertEqual(rag.retrieve(BIOLOGY_QUERY), ([], []))


@unittest.skipIf(SparseRAG is None, f"sparse_rag unavailable: {IMPORT_ERROR}")
class EmitterFallbackTests(unittest.TestCase):
    def test_retrieve_works_before_set_emitter_is_called(self):
        rag, _ = _corpus_rag(top_k=1)

        documents, _ = rag.retrieve(BIOLOGY_QUERY)

        self.assertEqual(documents, [DOCS[1]])

    def test_retrieve_works_on_an_instance_with_no_emitter_attribute(self):
        # The house pattern for skipping __init__. retrieve() swallows every
        # exception, so a missing emitter would show up as an empty result -
        # getting real documents back is what proves the getattr fallback.
        rag = object.__new__(SparseRAG)
        rag._index_loaded = True
        rag.collection = FakeCollection(DOCS, METAS)
        rag.documents = list(DOCS)
        rag.metadatas = list(METAS)
        rag.bm25 = sparse_rag_module.BM25Okapi(
            [rag._tokenize(doc) for doc in DOCS]
        )
        rag.top_k = 1

        self.assertNotIn("emitter", rag.__dict__)
        self.assertEqual(rag.retrieve(BIOLOGY_QUERY), ([DOCS[1]], [METAS[1]]))

    def test_set_emitter_receives_the_sparse_retrieval_stage(self):
        rag, _ = _corpus_rag(top_k=1)
        emitter = mock.Mock()
        rag.set_emitter(emitter)

        rag.retrieve(BIOLOGY_QUERY)

        self.assertIs(rag.emitter, emitter)
        stages = [call.args[0] for call in emitter.emit.call_args_list]
        self.assertIn("sparse_retrieval", stages)


if __name__ == "__main__":
    unittest.main()


def _with_params(**params):
    """Install a request runtime carrying just these pipeline-config values."""
    return runtime_context.use_runtime(runtime_context.RuntimeSettings(params=params))


@unittest.skipIf(SparseRAG is None, f"sparse_rag unavailable: {IMPORT_ERROR}")
class PerRequestSettingsTests(unittest.TestCase):
    def test_top_k_follows_the_request(self):
        rag, _ = _corpus_rag(top_k=5)
        self.assertEqual(rag.top_k, 5)

        with _with_params(top_k=2):
            self.assertEqual(rag.top_k, 2)

        # And does not outlive it.
        self.assertEqual(rag.top_k, 5)

    def test_assigning_top_k_sets_the_configured_default(self):
        # Assignment is how the old attribute behaved and how tests set it up;
        # a request still overrides it.
        rag, _ = _corpus_rag(top_k=5)
        rag.top_k = 3

        self.assertEqual(rag.top_k, 3)
        with _with_params(top_k=7):
            self.assertEqual(rag.top_k, 7)

    def test_stop_word_removal_follows_the_request(self):
        with mock.patch.object(sparse_rag_module, "stopwords") as stopwords:
            stopwords.words.return_value = ["the", "is"]
            rag, _ = _corpus_rag(remove_stop_words=False)

            # Configured off: the NLTK corpus is not read at all.
            self.assertEqual(rag.stop_words, set())
            stopwords.words.assert_not_called()

            # A request can still switch it on.
            with _with_params(remove_stop_words=True):
                self.assertEqual(rag.stop_words, {"the", "is"})
                self.assertEqual(rag._tokenize("the cat is here"), ["cat", "here"])

    def test_a_request_can_switch_stop_word_removal_off(self):
        with mock.patch.object(sparse_rag_module, "stopwords") as stopwords:
            stopwords.words.return_value = ["the", "is"]
            rag, _ = _corpus_rag(remove_stop_words=True)

            with _with_params(remove_stop_words=False):
                self.assertEqual(rag.stop_words, set())
                self.assertEqual(
                    rag._tokenize("the cat is here"), ["the", "cat", "is", "here"]
                )


@unittest.skipIf(SparseRAG is None, f"sparse_rag unavailable: {IMPORT_ERROR}")
class IndexInvalidationTests(unittest.TestCase):
    """The BM25 index is cached, and both its corpus and its tokenisation can
    change underneath it."""

    def test_switching_collections_rebuilds_the_index(self):
        # Regression guard. set_collection never reset _index_loaded, so on a
        # shared pipeline the first collection queried after startup kept
        # answering for every later one — including another user's documents.
        rag, first = _corpus_rag()
        self.assertTrue(rag._load_index_from_chroma())
        self.assertEqual(rag.documents, DOCS)

        second = FakeCollection(["completely different text"], [{"src": "other"}])
        with mock.patch.object(
            sparse_rag_module, "get_chroma_client", return_value=second
        ):
            rag.set_collection("someone_elses_collection")

        self.assertFalse(rag._index_loaded)
        self.assertEqual(rag.documents, [])

        self.assertTrue(rag._load_index_from_chroma())
        self.assertEqual(rag.documents, ["completely different text"])

    def test_changing_the_stop_word_setting_rebuilds_the_index(self):
        # The query is tokenised the same way the corpus was, so a cached index
        # built under the other setting would be matched against terms it no
        # longer contains.
        with mock.patch.object(sparse_rag_module, "stopwords") as stopwords:
            stopwords.words.return_value = ["the", "is", "on", "a"]
            rag, _ = _corpus_rag(remove_stop_words=True)

            self.assertTrue(rag._load_index_from_chroma())
            with_removal = list(rag.tokenized_corpus)

            with _with_params(remove_stop_words=False):
                self.assertTrue(rag._load_index_from_chroma())
                without_removal = list(rag.tokenized_corpus)

        self.assertNotEqual(with_removal, without_removal)

    def test_an_unchanged_setting_reuses_the_cached_index(self):
        rag, collection = _corpus_rag(remove_stop_words=False)

        self.assertTrue(rag._load_index_from_chroma())
        calls_after_first = len(collection.get_calls)

        self.assertTrue(rag._load_index_from_chroma())
        self.assertEqual(len(collection.get_calls), calls_after_first)
