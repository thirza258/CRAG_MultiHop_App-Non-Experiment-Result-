"""Tests for the root-level startup scripts:

- dl_reranker_model.py  (resumable model downloader)
- insert_base_dataset.py (base corpus preparation / optional indexing)
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import dl_reranker_model as dl
    DL_IMPORT_ERROR = ""
except Exception as exc:  # huggingface_hub not installed in this env
    dl = None
    DL_IMPORT_ERROR = str(exc)

try:
    import pandas as pd
    import insert_base_dataset as ibd
    IBD_IMPORT_ERROR = ""
except Exception as exc:  # needs Django + chromadb + pandas
    ibd = None
    IBD_IMPORT_ERROR = str(exc)


# ──────────────────────────────────────────────────────────────────────
# dl_reranker_model.py
# ──────────────────────────────────────────────────────────────────────

@unittest.skipIf(dl is None, f"dl_reranker_model unavailable: {DL_IMPORT_ERROR}")
class DownloadModelTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.models_dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

        patcher = mock.patch.object(dl, "MODELS_DIR", self.models_dir)
        patcher.start()
        self.addCleanup(patcher.stop)

        sleep_patcher = mock.patch.object(dl.time, "sleep")
        sleep_patcher.start()
        self.addCleanup(sleep_patcher.stop)

    def _local_dir(self, model_id="org/model"):
        return self.models_dir / model_id.replace("/", "--")

    def test_skips_when_marker_present(self):
        local = self._local_dir()
        local.mkdir(parents=True)
        (local / dl.COMPLETE_MARKER).touch()

        with mock.patch.object(dl, "snapshot_download") as snap:
            self.assertTrue(dl.download_model("org/model"))
            snap.assert_not_called()

    def test_success_writes_marker(self):
        with mock.patch.object(dl, "snapshot_download") as snap:
            self.assertTrue(dl.download_model("org/model"))
            snap.assert_called_once()

        self.assertTrue((self._local_dir() / dl.COMPLETE_MARKER).exists())

    def test_partial_download_without_marker_is_retried(self):
        # Simulates an interrupted earlier run: files exist but no marker.
        local = self._local_dir()
        local.mkdir(parents=True)
        (local / "model.safetensors").write_text("partial")

        with mock.patch.object(dl, "snapshot_download") as snap:
            self.assertTrue(dl.download_model("org/model"))
            snap.assert_called_once()

        self.assertTrue((local / dl.COMPLETE_MARKER).exists())

    def test_retries_then_succeeds(self):
        with mock.patch.object(
            dl, "snapshot_download", side_effect=[OSError("network"), None]
        ) as snap:
            self.assertTrue(dl.download_model("org/model"))
            self.assertEqual(snap.call_count, 2)

        self.assertTrue((self._local_dir() / dl.COMPLETE_MARKER).exists())

    def test_gives_up_after_max_attempts_without_marker(self):
        with mock.patch.object(dl, "MAX_ATTEMPTS", 3), mock.patch.object(
            dl, "snapshot_download", side_effect=OSError("network down")
        ) as snap:
            self.assertFalse(dl.download_model("org/model"))
            self.assertEqual(snap.call_count, 3)

        self.assertFalse((self._local_dir() / dl.COMPLETE_MARKER).exists())

    def test_download_models_reports_partial_failure(self):
        def fake_snapshot(repo_id, local_dir):
            if repo_id == "bad/model":
                raise OSError("network")

        with mock.patch.object(dl, "MAX_ATTEMPTS", 1), mock.patch.object(
            dl, "HYBRID_RERANKER_LIST", ["good/model", "bad/model"]
        ), mock.patch.object(dl, "snapshot_download", side_effect=fake_snapshot):
            self.assertFalse(dl.download_models())

        self.assertTrue(
            (self.models_dir / "good--model" / dl.COMPLETE_MARKER).exists()
        )
        self.assertFalse(
            (self.models_dir / "bad--model" / dl.COMPLETE_MARKER).exists()
        )


# ──────────────────────────────────────────────────────────────────────
# insert_base_dataset.py
# ──────────────────────────────────────────────────────────────────────

def _valid_df(rows: int = 2):
    return pd.DataFrame(
        [
            {
                "title": f"t{i}",
                "author": f"a{i}",
                "source": f"s{i}",
                "url": f"https://example.com/{i}",
                "category": "news",
                "published_at": "2024-01-01",
                "body": f"body {i}",
            }
            for i in range(rows)
        ]
    )


@unittest.skipIf(ibd is None, f"insert_base_dataset unavailable: {IBD_IMPORT_ERROR}")
class TruthyTests(unittest.TestCase):
    def test_truthy_values(self):
        for value in ("1", "true", "TRUE", "Yes", "on"):
            self.assertTrue(ibd._truthy(value), value)
        for value in ("", "0", "false", "no", "off", "None"):
            self.assertFalse(ibd._truthy(value), value)


@unittest.skipIf(ibd is None, f"insert_base_dataset unavailable: {IBD_IMPORT_ERROR}")
class ValidateCorpusTests(unittest.TestCase):
    def test_valid_corpus_passes(self):
        df = _valid_df()
        self.assertIs(ibd._validate_corpus(df), df)

    def test_empty_corpus_rejected(self):
        with self.assertRaises(ValueError):
            ibd._validate_corpus(pd.DataFrame())

    def test_missing_columns_rejected(self):
        df = _valid_df().drop(columns=["body"])
        with self.assertRaises(ValueError):
            ibd._validate_corpus(df)


@unittest.skipIf(ibd is None, f"insert_base_dataset unavailable: {IBD_IMPORT_ERROR}")
class EnsureCorpusExistsTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        corpus_dir = Path(self._tmp.name) / "corpus"
        self.corpus_file = corpus_dir / "corpus.json"
        self.addCleanup(self._tmp.cleanup)

        for name, value in (
            ("CORPUS_DIR", corpus_dir),
            ("CORPUS_FILE", self.corpus_file),
        ):
            patcher = mock.patch.object(ibd, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

        sleep_patcher = mock.patch.object(ibd.time, "sleep")
        sleep_patcher.start()
        self.addCleanup(sleep_patcher.stop)

    def test_uses_valid_cached_corpus_without_downloading(self):
        self.corpus_file.parent.mkdir(parents=True)
        _valid_df().to_json(self.corpus_file, orient="records")

        with mock.patch.object(ibd.pd, "read_json", wraps=pd.read_json) as read_json:
            df = ibd.ensure_corpus_exists()

        self.assertEqual(len(df), 2)
        for call in read_json.call_args_list:
            self.assertNotEqual(call.args[0], ibd.CORPUS_URL)

    def test_invalid_cached_corpus_is_redownloaded(self):
        self.corpus_file.parent.mkdir(parents=True)
        self.corpus_file.write_text('[{"only": "junk"}]')

        real_read_json = pd.read_json

        def fake_read_json(source, **kwargs):
            if source == ibd.CORPUS_URL:
                return _valid_df()
            return real_read_json(source, **kwargs)

        with mock.patch.object(ibd.pd, "read_json", side_effect=fake_read_json):
            df = ibd.ensure_corpus_exists()

        self.assertEqual(len(df), 2)
        # The repaired corpus must have been written atomically to disk.
        self.assertTrue(self.corpus_file.exists())
        self.assertFalse(self.corpus_file.with_suffix(".json.tmp").exists())
        self.assertEqual(len(pd.read_json(self.corpus_file)), 2)

    def test_download_retries_then_succeeds(self):
        calls = {"n": 0}
        real_read_json = pd.read_json

        def flaky_read_json(source, **kwargs):
            if source == ibd.CORPUS_URL:
                calls["n"] += 1
                if calls["n"] == 1:
                    raise OSError("network")
                return _valid_df()
            return real_read_json(source, **kwargs)

        with mock.patch.object(ibd.pd, "read_json", side_effect=flaky_read_json):
            df = ibd.ensure_corpus_exists()

        self.assertEqual(calls["n"], 2)
        self.assertEqual(len(df), 2)

    def test_download_failure_raises_after_attempts(self):
        with mock.patch.object(ibd, "CORPUS_DOWNLOAD_ATTEMPTS", 2), mock.patch.object(
            ibd.pd, "read_json", side_effect=OSError("network down")
        ) as read_json:
            with self.assertRaises(RuntimeError):
                ibd.ensure_corpus_exists()
            self.assertEqual(read_json.call_count, 2)

        self.assertFalse(self.corpus_file.exists())


@unittest.skipIf(ibd is None, f"insert_base_dataset unavailable: {IBD_IMPORT_ERROR}")
class ConnectChromaTests(unittest.TestCase):
    def setUp(self):
        sleep_patcher = mock.patch.object(ibd.time, "sleep")
        sleep_patcher.start()
        self.addCleanup(sleep_patcher.stop)

    def test_retries_until_chroma_is_up(self):
        bad = mock.Mock()
        bad.heartbeat.side_effect = ConnectionError("not up yet")
        good = mock.Mock()

        with mock.patch.object(
            ibd.chromadb, "HttpClient", side_effect=[bad, good]
        ) as http_client:
            client = ibd.connect_chroma()

        self.assertIs(client, good)
        self.assertEqual(http_client.call_count, 2)

    def test_raises_when_chroma_never_comes_up(self):
        bad = mock.Mock()
        bad.heartbeat.side_effect = ConnectionError("down")

        with mock.patch.object(ibd, "CHROMA_CONNECT_ATTEMPTS", 3), mock.patch.object(
            ibd.chromadb, "HttpClient", return_value=bad
        ):
            with self.assertRaises(RuntimeError):
                ibd.connect_chroma()


@unittest.skipIf(ibd is None, f"insert_base_dataset unavailable: {IBD_IMPORT_ERROR}")
class MainGatingTests(unittest.TestCase):
    def _run_main(self, *, count, env):
        collection = mock.Mock()
        collection.count.return_value = count
        client = mock.Mock()
        client.get_or_create_collection.return_value = collection

        with mock.patch.object(ibd, "connect_chroma", return_value=client), \
                mock.patch.object(ibd, "ensure_corpus_exists", return_value=_valid_df()) as ensure, \
                mock.patch.object(ibd, "index_corpus") as index, \
                mock.patch.dict(os.environ, env, clear=False):
            ibd.main()

        return ensure, index, client

    def test_collection_always_created_and_indexing_off_by_default(self):
        env = {"INSERT_BASE_DATASET": "", "OPENROUTER_API_KEY": "sk-x"}
        ensure, index, client = self._run_main(count=0, env=env)

        client.get_or_create_collection.assert_called_once()
        ensure.assert_called_once()  # corpus is cached even when indexing is off
        index.assert_not_called()

    def test_indexing_skipped_when_already_populated(self):
        env = {"INSERT_BASE_DATASET": "true", "OPENROUTER_API_KEY": "sk-x"}
        _, index, _ = self._run_main(count=42, env=env)
        index.assert_not_called()

    def test_indexing_skipped_without_api_key(self):
        env = {"INSERT_BASE_DATASET": "true", "OPENROUTER_API_KEY": ""}
        _, index, _ = self._run_main(count=0, env=env)
        index.assert_not_called()

    def test_indexing_runs_when_enabled_empty_and_key_present(self):
        env = {"INSERT_BASE_DATASET": "true", "OPENROUTER_API_KEY": "sk-x"}
        _, index, _ = self._run_main(count=0, env=env)
        index.assert_called_once()


@unittest.skipIf(ibd is None, f"insert_base_dataset unavailable: {IBD_IMPORT_ERROR}")
class IndexCorpusTests(unittest.TestCase):
    def _run(self, embeddings_count, insert_ok=True):
        docs = ["chunk1", "chunk2", "chunk3"]
        metas = [{}, {}, {}]

        dense = mock.Mock()
        dense._get_embeddings.return_value = [[0.1]] * embeddings_count

        with mock.patch("dense_rag.dense_rag.DenseRAG", return_value=dense), \
                mock.patch("common.dataset_settings.prepare_corpus", return_value=(docs, metas)), \
                mock.patch("chroma.chroma_settings.insert_chunk_to_chromadb", return_value=insert_ok) as insert:
            ibd.index_corpus(mock.Mock(), _valid_df())
        return insert

    def test_refuses_to_insert_misaligned_embeddings(self):
        with self.assertRaises(RuntimeError):
            self._run(embeddings_count=2)

    def test_raises_when_insert_reports_failed_batches(self):
        with self.assertRaises(RuntimeError):
            self._run(embeddings_count=3, insert_ok=False)

    def test_inserts_when_everything_aligns(self):
        insert = self._run(embeddings_count=3, insert_ok=True)
        insert.assert_called_once()


if __name__ == "__main__":
    unittest.main()
