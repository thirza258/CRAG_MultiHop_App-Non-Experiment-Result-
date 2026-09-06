"""Tests for request-scoped model choice and credentials.

The whole design rests on three properties, and each has tests here that would
fail loudly if it broke:

* **Isolation** — the pipeline is one process-wide singleton, so two concurrent
  queries choosing different models with different keys must not see each
  other's. A regression here means one user's credentials answering another
  user's question.
* **Reset** — the REST path runs on a pooled server thread. A runtime left
  installed would hand the *next* request served by that thread the previous
  caller's key.
* **Fallback** — an empty choice means "use what this component was configured
  with", never "use nothing". That is what keeps every pre-existing call site
  behaving exactly as it did.
"""

import os
import threading
import time
import unittest
from unittest import mock

try:
    from common.runtime import context as rc
    IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - import guard
    rc = None
    IMPORT_ERROR = str(exc)

SECRET = "sk-or-v1-0123456789abcdef0123456789abcdef"


@unittest.skipIf(rc is None, f"common.runtime.context unavailable: {IMPORT_ERROR}")
class NormalizeModelIdTests(unittest.TestCase):
    def test_real_openrouter_ids_are_accepted(self):
        # Sampled from the live catalogs, including the shapes that would break
        # a naive "letters and slashes only" check.
        for model_id in (
            "openai/gpt-4o",
            "mistralai/mistral-nemo",
            "qwen/qwen3-30b-a3b-instruct-2507",
            "google/gemini-embedding-2-preview",
            "liquid/lfm-2.5-embedding-350m:free",
            "sentence-transformers/paraphrase-minilm-l6-v2",
            "anthropic/claude-3.5-sonnet",
            "gpt-4o",
        ):
            with self.subTest(model_id=model_id):
                self.assertEqual(rc.normalize_model_id(model_id), model_id)

    def test_surrounding_whitespace_is_stripped(self):
        self.assertEqual(rc.normalize_model_id("  openai/gpt-4o\n"), "openai/gpt-4o")

    def test_path_traversal_and_injection_shapes_are_rejected(self):
        # The id is interpolated into an API request; these must not survive.
        for model_id in (
            "../../etc/passwd",
            "model with spaces",
            "model\r\nX-Injected: yes",
            "model?query=1",
            "model#fragment",
            "model&other",
            "'; DROP TABLE--",
            "/leading-slash",
            "-leading-dash",
        ):
            with self.subTest(model_id=model_id):
                self.assertEqual(rc.normalize_model_id(model_id), "")

    def test_empty_and_non_string_fall_back_to_the_default(self):
        for value in ("", "   ", None, 42, [], {}, True):
            with self.subTest(value=value):
                self.assertEqual(rc.normalize_model_id(value, "fallback"), "fallback")

    def test_over_long_ids_are_rejected(self):
        self.assertEqual(
            rc.normalize_model_id("a" * (rc.MAX_MODEL_ID_LENGTH + 1), "fallback"),
            "fallback",
        )

    def test_the_default_is_returned_verbatim_and_not_re_validated(self):
        # Callers pass their own configured value as the default; second-guessing
        # it would change behaviour for components this feature never touched.
        self.assertEqual(rc.normalize_model_id("bad id!", "whatever/we-had"), "whatever/we-had")


@unittest.skipIf(rc is None, f"common.runtime.context unavailable: {IMPORT_ERROR}")
class FallbackTests(unittest.TestCase):
    def test_no_runtime_installed_means_the_components_own_default(self):
        self.assertEqual(rc.resolve_llm_model("configured/model"), "configured/model")
        self.assertEqual(rc.resolve_embedding_model("configured/embed"), "configured/embed")

    def test_an_empty_choice_does_not_override_the_default(self):
        with rc.use_runtime(rc.RuntimeSettings()):
            self.assertEqual(rc.resolve_llm_model("configured/model"), "configured/model")
            self.assertEqual(rc.resolve_embedding_model("configured/embed"), "configured/embed")

    def test_keys_fall_back_to_the_environment(self):
        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "server-key"}, clear=False):
            with rc.use_runtime(rc.RuntimeSettings()):
                self.assertEqual(rc.openrouter_api_key(), "server-key")

    def test_a_supplied_key_beats_the_environment(self):
        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "server-key"}, clear=False):
            with rc.use_runtime(rc.RuntimeSettings(keys={"openrouter": SECRET})):
                self.assertEqual(rc.openrouter_api_key(), SECRET)

    def test_server_has_key_reflects_the_environment_only(self):
        with mock.patch.dict(os.environ, {"NEWS_API_KEY": "server-news"}, clear=False):
            self.assertTrue(rc.server_has_key("news"))
        with mock.patch.dict(os.environ, {"NEWS_API_KEY": ""}, clear=False):
            self.assertFalse(rc.server_has_key("news"))
        self.assertFalse(rc.server_has_key("not-a-provider"))


@unittest.skipIf(rc is None, f"common.runtime.context unavailable: {IMPORT_ERROR}")
class UseRuntimeTests(unittest.TestCase):
    def test_installs_the_choices_for_the_block(self):
        settings = rc.RuntimeSettings(
            llm_model="openai/gpt-4o",
            embedding_model="baai/bge-m3",
            keys={"openrouter": SECRET, "news": "news-key"},
        )
        with rc.use_runtime(settings):
            self.assertEqual(rc.resolve_llm_model("x"), "openai/gpt-4o")
            self.assertEqual(rc.resolve_embedding_model("x"), "baai/bge-m3")
            self.assertEqual(rc.openrouter_api_key(), SECRET)
            self.assertEqual(rc.news_api_key(), "news-key")

    def test_resets_on_exit(self):
        with rc.use_runtime(rc.RuntimeSettings(llm_model="openai/gpt-4o")):
            pass
        self.assertEqual(rc.current_runtime(), rc.EMPTY_RUNTIME)
        self.assertEqual(rc.resolve_llm_model("configured"), "configured")

    def test_resets_even_when_the_block_raises(self):
        # The REST path's except clause runs after this; a leaked key would go
        # to whatever request that pooled thread serves next.
        with self.assertRaises(ValueError):
            with rc.use_runtime(rc.RuntimeSettings(keys={"openrouter": SECRET})):
                raise ValueError("stage failed")
        self.assertEqual(rc.current_runtime(), rc.EMPTY_RUNTIME)
        self.assertEqual(rc.request_secrets(), {})

    def test_nesting_restores_the_outer_runtime_not_the_empty_one(self):
        outer = rc.RuntimeSettings(llm_model="a/outer", keys={"openrouter": "outer-key"})
        inner = rc.RuntimeSettings(llm_model="b/inner", keys={"openrouter": "inner-key"})
        with rc.use_runtime(outer):
            with rc.use_runtime(inner):
                self.assertEqual(rc.resolve_llm_model("x"), "b/inner")
                self.assertEqual(rc.openrouter_api_key(), "inner-key")
            self.assertEqual(rc.resolve_llm_model("x"), "a/outer")
            self.assertEqual(rc.openrouter_api_key(), "outer-key")
        self.assertEqual(rc.current_runtime(), rc.EMPTY_RUNTIME)

    def test_junk_settings_degrade_to_the_empty_runtime(self):
        with rc.use_runtime("not-a-settings-object"):
            self.assertEqual(rc.current_runtime(), rc.EMPTY_RUNTIME)


@unittest.skipIf(rc is None, f"common.runtime.context unavailable: {IMPORT_ERROR}")
class PinEmbeddingModelTests(unittest.TestCase):
    def test_replaces_the_embedding_model_for_the_rest_of_the_block(self):
        with rc.use_runtime(rc.RuntimeSettings(embedding_model="user/choice")):
            rc.pin_embedding_model("collection/actual")
            self.assertEqual(rc.resolve_embedding_model("x"), "collection/actual")

    def test_leaves_the_llm_and_the_keys_alone(self):
        settings = rc.RuntimeSettings(
            llm_model="openai/gpt-4o",
            embedding_model="user/choice",
            keys={"openrouter": SECRET},
        )
        with rc.use_runtime(settings):
            rc.pin_embedding_model("collection/actual")
            self.assertEqual(rc.resolve_llm_model("x"), "openai/gpt-4o")
            self.assertEqual(rc.openrouter_api_key(), SECRET)

    def test_a_blank_model_is_a_no_op(self):
        # "We could not determine the collection's model" must not be mistaken
        # for "use no model" — the user's choice still applies.
        with rc.use_runtime(rc.RuntimeSettings(embedding_model="user/choice")):
            for value in ("", None, "   "):
                with self.subTest(value=value):
                    rc.pin_embedding_model(value)
                    self.assertEqual(rc.resolve_embedding_model("x"), "user/choice")

    def test_a_malformed_model_is_a_no_op(self):
        with rc.use_runtime(rc.RuntimeSettings(embedding_model="user/choice")):
            rc.pin_embedding_model("not a model id!")
            self.assertEqual(rc.resolve_embedding_model("x"), "user/choice")

    def test_the_pin_does_not_survive_the_enclosing_block(self):
        with rc.use_runtime(rc.RuntimeSettings(embedding_model="user/choice")):
            rc.pin_embedding_model("collection/actual")
        self.assertEqual(rc.current_runtime(), rc.EMPTY_RUNTIME)
        self.assertEqual(rc.resolve_embedding_model("configured"), "configured")


@unittest.skipIf(rc is None, f"common.runtime.context unavailable: {IMPORT_ERROR}")
class ThreadIsolationTests(unittest.TestCase):
    def test_concurrent_queries_never_see_each_others_choices(self):
        # This is the property that lets one shared pipeline serve every model
        # choice. If it regresses, user A's key answers user B's question.
        observed = {}
        errors = []
        barrier = threading.Barrier(6, timeout=10)

        def worker(index):
            try:
                settings = rc.RuntimeSettings(
                    llm_model=f"vendor/model-{index}",
                    embedding_model=f"vendor/embed-{index}",
                    keys={"openrouter": f"key-{index}"},
                )
                with rc.use_runtime(settings):
                    # All threads sit inside their block simultaneously, so any
                    # shared mutable state would be observable.
                    barrier.wait()
                    time.sleep(0.01)
                    observed[index] = (
                        rc.resolve_llm_model("none"),
                        rc.resolve_embedding_model("none"),
                        rc.openrouter_api_key(),
                    )
            except Exception as exc:  # pragma: no cover - surfaces as a failure
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(timeout=10)

        self.assertEqual(errors, [])
        self.assertEqual(
            observed,
            {
                i: (f"vendor/model-{i}", f"vendor/embed-{i}", f"key-{i}")
                for i in range(5)
            },
        )

    def test_a_new_thread_starts_with_no_runtime(self):
        # threading.Thread does not copy the parent's context, which is exactly
        # what the websocket path relies on.
        seen = {}

        def worker():
            seen["llm"] = rc.resolve_llm_model("configured")

        with rc.use_runtime(rc.RuntimeSettings(llm_model="openai/gpt-4o")):
            thread = threading.Thread(target=worker)
            thread.start()
            thread.join(timeout=10)

        self.assertEqual(seen["llm"], "configured")


@unittest.skipIf(rc is None, f"common.runtime.context unavailable: {IMPORT_ERROR}")
class RuntimeSettingsTests(unittest.TestCase):
    def test_from_wire_validates_the_model_ids(self):
        settings = rc.RuntimeSettings.from_wire(
            {"llm_model": "openai/gpt-4o", "embedding_model": "not valid!"},
            {"openrouter": SECRET},
        )
        self.assertEqual(settings.llm_model, "openai/gpt-4o")
        self.assertEqual(settings.embedding_model, "")
        self.assertEqual(settings.keys["openrouter"], SECRET)

    def test_from_wire_tolerates_junk(self):
        for config in (None, "string", 42, []):
            with self.subTest(config=config):
                settings = rc.RuntimeSettings.from_wire(config, None)
                self.assertEqual(settings.llm_model, "")
                self.assertEqual(settings.embedding_model, "")

    def test_from_wire_is_idempotent_over_already_normalised_keys(self):
        from common.runtime.api_keys import normalize_api_keys

        config = {"llm_model": "openai/gpt-4o"}
        once = rc.RuntimeSettings.from_wire(config, {"openrouter": SECRET})
        twice = rc.RuntimeSettings.from_wire(config, normalize_api_keys({"openrouter": SECRET}))
        self.assertEqual(once, twice)

    def test_round_trips_through_to_dict(self):
        # This is what crosses the process boundary to the Celery worker.
        settings = rc.RuntimeSettings(
            llm_model="openai/gpt-4o",
            embedding_model="baai/bge-m3",
            keys={"openrouter": SECRET, "news": "news-key"},
        )
        self.assertEqual(rc.RuntimeSettings.from_dict(settings.to_dict()), settings)

    def test_from_dict_re_validates_instead_of_trusting_the_payload(self):
        settings = rc.RuntimeSettings.from_dict(
            {"llm_model": "bad id!", "embedding_model": "openai/text-embedding-3-small",
             "keys": {"openrouter": "has a space"}}
        )
        self.assertEqual(settings.llm_model, "")
        self.assertEqual(settings.embedding_model, "openai/text-embedding-3-small")
        self.assertEqual(settings.keys["openrouter"], "")

    def test_from_dict_tolerates_junk(self):
        for junk in (None, "string", 42, []):
            with self.subTest(junk=junk):
                self.assertEqual(rc.RuntimeSettings.from_dict(junk), rc.RuntimeSettings())

    def test_describe_never_reveals_a_key(self):
        settings = rc.RuntimeSettings(
            llm_model="openai/gpt-4o", keys={"openrouter": SECRET, "news": "news-key"}
        )
        summary = settings.describe()
        self.assertNotIn(SECRET, summary)
        self.assertNotIn("news-key", summary)
        self.assertIn("openai/gpt-4o", summary)
        self.assertIn("openrouter=own", summary)

    def test_is_immutable(self):
        settings = rc.RuntimeSettings(llm_model="openai/gpt-4o")
        with self.assertRaises(Exception):
            settings.llm_model = "something/else"

    def test_with_embedding_model_returns_a_copy(self):
        settings = rc.RuntimeSettings(llm_model="openai/gpt-4o", embedding_model="a/one")
        pinned = settings.with_embedding_model("b/two")
        self.assertEqual(settings.embedding_model, "a/one")
        self.assertEqual(pinned.embedding_model, "b/two")
        self.assertEqual(pinned.llm_model, "openai/gpt-4o")


if __name__ == "__main__":
    unittest.main()
