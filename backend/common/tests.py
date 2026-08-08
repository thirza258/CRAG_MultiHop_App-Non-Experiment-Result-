"""Tests for common helpers: NLTK bootstrap and local model path resolution."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    from common import nltk_setup
    NLTK_IMPORT_ERROR = ""
except Exception as exc:
    nltk_setup = None
    NLTK_IMPORT_ERROR = str(exc)

try:
    from common import memory
    MEMORY_IMPORT_ERROR = ""
except Exception as exc:  # needs Django configured (router.models import)
    memory = None
    MEMORY_IMPORT_ERROR = str(exc)


@unittest.skipIf(nltk_setup is None, f"nltk_setup unavailable: {NLTK_IMPORT_ERROR}")
class EnsureNltkDataTests(unittest.TestCase):
    def setUp(self):
        nltk_setup._checked = False
        self.addCleanup(setattr, nltk_setup, "_checked", False)

    def test_no_download_when_all_data_present(self):
        with mock.patch.object(nltk_setup.nltk.data, "find") as find, mock.patch.object(
            nltk_setup.nltk, "download"
        ) as download:
            nltk_setup.ensure_nltk_data()
            self.assertEqual(find.call_count, len(nltk_setup.REQUIRED_NLTK_DATA))
            download.assert_not_called()

    def test_downloads_missing_packages(self):
        downloaded = set()

        def fake_find(path):
            package = path.split("/")[-1]
            if package not in downloaded:
                raise LookupError(path)

        def fake_download(package, quiet=True):
            downloaded.add(package)

        with mock.patch.object(
            nltk_setup.nltk.data, "find", side_effect=fake_find
        ), mock.patch.object(
            nltk_setup.nltk, "download", side_effect=fake_download
        ):
            nltk_setup.ensure_nltk_data()

        self.assertEqual(
            downloaded, {pkg for _, pkg in nltk_setup.REQUIRED_NLTK_DATA}
        )

    def test_raises_clear_error_when_download_fails(self):
        with mock.patch.object(
            nltk_setup.nltk.data, "find", side_effect=LookupError("missing")
        ), mock.patch.object(nltk_setup.nltk, "download"):
            with self.assertRaises(RuntimeError):
                nltk_setup.ensure_nltk_data()

    def test_second_call_is_cached(self):
        with mock.patch.object(nltk_setup.nltk.data, "find") as find:
            nltk_setup.ensure_nltk_data()
            first_count = find.call_count
            nltk_setup.ensure_nltk_data()
            self.assertEqual(find.call_count, first_count)


@unittest.skipIf(memory is None, f"common.memory unavailable: {MEMORY_IMPORT_ERROR}")
class LocalModelPathTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.models_dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

        patcher = mock.patch.object(memory, "MODELS_DIR", self.models_dir)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_uses_local_dir_when_marker_present(self):
        local = self.models_dir / "org--model"
        local.mkdir(parents=True)
        (local / ".download_complete").touch()

        self.assertEqual(memory._local_model_path("org/model"), str(local))

    def test_uses_local_dir_with_real_files_but_no_marker(self):
        local = self.models_dir / "org--model"
        local.mkdir(parents=True)
        (local / "config.json").write_text("{}")

        self.assertEqual(memory._local_model_path("org/model"), str(local))

    def test_falls_back_when_dir_only_has_hidden_leftovers(self):
        # An interrupted snapshot_download leaves only a hidden .cache dir —
        # that must not be mistaken for a usable model.
        local = self.models_dir / "org--model"
        (local / ".cache").mkdir(parents=True)

        self.assertEqual(memory._local_model_path("org/model"), "org/model")

    def test_falls_back_when_dir_missing(self):
        self.assertEqual(memory._local_model_path("org/model"), "org/model")


if __name__ == "__main__":
    unittest.main()
