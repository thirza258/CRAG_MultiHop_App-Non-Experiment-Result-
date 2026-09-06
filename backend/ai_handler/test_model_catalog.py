"""Tests for the OpenRouter model catalog the pickers are built from.

The property that matters: **the settings panel always gets a usable list.** A
picker with no options is a feature the user cannot reach, so every failure mode
here degrades to something renderable — the live catalog, then the last good
response, then a bundled fallback — and never raises.

The other thing under test is the two-endpoint split. OpenRouter's main
``/models`` catalog contains no embedding models at all; they live behind
``/embeddings/models``. Fetching only the first would leave the embedding picker
offering ids that cannot be embedded with.
"""

import unittest
from unittest import mock

try:
    from ai_handler import model_catalog
    IMPORT_ERROR = ""
except Exception as exc:  # needs requests
    model_catalog = None
    IMPORT_ERROR = str(exc)


# Trimmed from the real responses, keeping the fields the code reads.
CHAT_PAYLOAD = {
    "data": [
        {
            "id": "mistralai/mistral-nemo",
            "name": "Mistral: Mistral Nemo",
            "context_length": 131072,
            "pricing": {"prompt": "0.000000834", "completion": "0.000002501"},
            "architecture": {"output_modalities": ["text"]},
        },
        {
            "id": "deepseek/deepseek-chat:free",
            "name": "DeepSeek: V3 (free)",
            "context_length": 65536,
            "pricing": {"prompt": "0", "completion": "0"},
        },
    ]
}

EMBEDDING_PAYLOAD = {
    "data": [
        {
            "id": "google/gemini-embedding-2-preview",
            "name": "Google: Gemini Embedding 2 (preview)",
            "context_length": 2048,
            "pricing": {"prompt": "0.00000001"},
            "architecture": {"output_modalities": ["embeddings"]},
        }
    ]
}


def _ok(payload):
    response = mock.Mock()
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    return response


def _route(chat=None, embedding=None):
    """A requests.get stand-in that answers each catalog URL separately.

    A value that is an Exception is raised, so a test can fail one endpoint
    while the other succeeds.
    """
    def get(url, **_kwargs):
        payload = chat if url == model_catalog.CHAT_MODELS_URL else embedding
        if isinstance(payload, Exception):
            raise payload
        return _ok(payload)

    return get


@unittest.skipIf(model_catalog is None, f"ai_handler.model_catalog unavailable: {IMPORT_ERROR}")
class CatalogTests(unittest.TestCase):
    def setUp(self):
        model_catalog.reset_cache()
        self.addCleanup(model_catalog.reset_cache)

    # ── the happy path ───────────────────────────────────────────────────────
    def test_fetches_both_catalogs_and_reports_openrouter_as_the_source(self):
        with mock.patch.object(
            model_catalog.requests, "get", side_effect=_route(CHAT_PAYLOAD, EMBEDDING_PAYLOAD)
        ):
            catalog = model_catalog.get_catalog()

        self.assertEqual(catalog["source"], "openrouter")
        self.assertEqual(
            [m["id"] for m in catalog["chat"]],
            ["deepseek/deepseek-chat:free", "mistralai/mistral-nemo"],
        )
        self.assertEqual(
            [m["id"] for m in catalog["embedding"]], ["google/gemini-embedding-2-preview"]
        )

    def test_asks_both_endpoints_because_embeddings_are_not_in_the_main_catalog(self):
        with mock.patch.object(
            model_catalog.requests, "get", side_effect=_route(CHAT_PAYLOAD, EMBEDDING_PAYLOAD)
        ) as get:
            model_catalog.get_catalog()

        requested = {call.args[0] for call in get.call_args_list}
        self.assertEqual(
            requested, {model_catalog.CHAT_MODELS_URL, model_catalog.EMBEDDING_MODELS_URL}
        )

    def test_entries_are_trimmed_to_what_the_picker_renders(self):
        with mock.patch.object(
            model_catalog.requests, "get", side_effect=_route(CHAT_PAYLOAD, EMBEDDING_PAYLOAD)
        ):
            catalog = model_catalog.get_catalog()

        entry = next(m for m in catalog["chat"] if m["id"] == "mistralai/mistral-nemo")
        self.assertEqual(
            set(entry),
            {"id", "name", "context_length", "prompt_price", "completion_price", "free"},
        )
        # Prices arrive as strings and have to be usable as numbers.
        self.assertAlmostEqual(entry["prompt_price"], 0.000000834)
        self.assertEqual(entry["context_length"], 131072)
        self.assertFalse(entry["free"])

    def test_a_zero_priced_model_is_flagged_free(self):
        with mock.patch.object(
            model_catalog.requests, "get", side_effect=_route(CHAT_PAYLOAD, EMBEDDING_PAYLOAD)
        ):
            catalog = model_catalog.get_catalog()

        free = next(m for m in catalog["chat"] if m["id"].endswith(":free"))
        self.assertTrue(free["free"])

    def test_results_are_sorted_by_id(self):
        with mock.patch.object(
            model_catalog.requests, "get", side_effect=_route(CHAT_PAYLOAD, EMBEDDING_PAYLOAD)
        ):
            catalog = model_catalog.get_catalog()

        ids = [m["id"] for m in catalog["chat"]]
        self.assertEqual(ids, sorted(ids))

    # ── caching ──────────────────────────────────────────────────────────────
    def test_a_second_call_is_served_from_the_cache(self):
        with mock.patch.object(
            model_catalog.requests, "get", side_effect=_route(CHAT_PAYLOAD, EMBEDDING_PAYLOAD)
        ) as get:
            model_catalog.get_catalog()
            self.assertEqual(get.call_count, 2)
            second = model_catalog.get_catalog()
            self.assertEqual(get.call_count, 2, "the cache was not used")

        self.assertEqual(second["source"], "cache")
        self.assertTrue(second["chat"])

    def test_force_refresh_bypasses_the_cache(self):
        with mock.patch.object(
            model_catalog.requests, "get", side_effect=_route(CHAT_PAYLOAD, EMBEDDING_PAYLOAD)
        ) as get:
            model_catalog.get_catalog()
            model_catalog.get_catalog(force_refresh=True)

        self.assertEqual(get.call_count, 4)

    def test_an_expired_cache_is_refetched(self):
        with mock.patch.object(
            model_catalog.requests, "get", side_effect=_route(CHAT_PAYLOAD, EMBEDDING_PAYLOAD)
        ) as get:
            model_catalog.get_catalog()
            model_catalog._cache["fetched_at"] -= model_catalog.CACHE_TTL_SECONDS + 1
            catalog = model_catalog.get_catalog()

        self.assertEqual(get.call_count, 4)
        self.assertEqual(catalog["source"], "openrouter")

    # ── degradation ──────────────────────────────────────────────────────────
    def test_one_dead_endpoint_does_not_cost_the_other(self):
        with mock.patch.object(
            model_catalog.requests,
            "get",
            side_effect=_route(CHAT_PAYLOAD, ConnectionError("embeddings down")),
        ):
            catalog = model_catalog.get_catalog()

        self.assertEqual([m["id"] for m in catalog["chat"]][0], "deepseek/deepseek-chat:free")
        # The embedding picker still has to offer something usable.
        self.assertEqual(
            [m["id"] for m in catalog["embedding"]],
            [m["id"] for m in model_catalog.FALLBACK_CATALOG["embedding"]],
        )
        self.assertIn("errors", catalog)

    def test_both_endpoints_dead_serves_the_bundled_fallback(self):
        with mock.patch.object(
            model_catalog.requests,
            "get",
            side_effect=_route(ConnectionError("down"), ConnectionError("down")),
        ):
            catalog = model_catalog.get_catalog()

        self.assertEqual(catalog["source"], "fallback")
        self.assertTrue(catalog["chat"])
        self.assertTrue(catalog["embedding"])

    def test_the_last_good_response_survives_a_later_outage(self):
        with mock.patch.object(
            model_catalog.requests, "get", side_effect=_route(CHAT_PAYLOAD, EMBEDDING_PAYLOAD)
        ):
            model_catalog.get_catalog()

        with mock.patch.object(
            model_catalog.requests,
            "get",
            side_effect=_route(ConnectionError("down"), ConnectionError("down")),
        ):
            catalog = model_catalog.get_catalog(force_refresh=True)

        # Stale beats bundled: it is the real catalog, just older.
        self.assertNotEqual(catalog["source"], "fallback")
        self.assertEqual(
            [m["id"] for m in catalog["chat"]],
            ["deepseek/deepseek-chat:free", "mistralai/mistral-nemo"],
        )

    def test_a_malformed_response_counts_as_a_failure(self):
        for payload in ({}, {"data": "not-a-list"}, {"data": []}, {"data": [{}]}):
            with self.subTest(payload=payload):
                model_catalog.reset_cache()
                with mock.patch.object(
                    model_catalog.requests, "get", side_effect=_route(payload, payload)
                ):
                    catalog = model_catalog.get_catalog()
                self.assertEqual(catalog["source"], "fallback")

    def test_unusable_entries_are_dropped_but_the_rest_survive(self):
        payload = {
            "data": [
                {"name": "no id at all"},
                {"id": "", "name": "blank id"},
                "not even a dict",
                {"id": "openai/gpt-4o", "name": "OpenAI: GPT-4o"},
            ]
        }
        with mock.patch.object(
            model_catalog.requests, "get", side_effect=_route(payload, EMBEDDING_PAYLOAD)
        ):
            catalog = model_catalog.get_catalog()

        self.assertEqual([m["id"] for m in catalog["chat"]], ["openai/gpt-4o"])

    def test_unparseable_pricing_becomes_none_rather_than_raising(self):
        payload = {
            "data": [
                {"id": "a/one", "pricing": {"prompt": "not-a-number"}},
                {"id": "b/two", "pricing": "not-a-dict"},
                {"id": "c/three"},
            ]
        }
        with mock.patch.object(
            model_catalog.requests, "get", side_effect=_route(payload, EMBEDDING_PAYLOAD)
        ):
            catalog = model_catalog.get_catalog()

        self.assertEqual(
            [m["prompt_price"] for m in catalog["chat"]], [None, None, None]
        )

    def test_an_http_error_status_counts_as_a_failure(self):
        response = mock.Mock()
        response.raise_for_status.side_effect = Exception("429 Too Many Requests")
        with mock.patch.object(model_catalog.requests, "get", return_value=response):
            catalog = model_catalog.get_catalog()

        self.assertEqual(catalog["source"], "fallback")

    def test_a_request_timeout_is_always_set(self):
        # Without one this runs inside a user-facing GET with no upper bound.
        with mock.patch.object(
            model_catalog.requests, "get", side_effect=_route(CHAT_PAYLOAD, EMBEDDING_PAYLOAD)
        ) as get:
            model_catalog.get_catalog()

        for call in get.call_args_list:
            self.assertEqual(
                call.kwargs.get("timeout"), model_catalog.REQUEST_TIMEOUT_SECONDS
            )


@unittest.skipIf(model_catalog is None, f"ai_handler.model_catalog unavailable: {IMPORT_ERROR}")
class FallbackCatalogTests(unittest.TestCase):
    def test_every_fallback_entry_has_an_id_and_a_name(self):
        for group in ("chat", "embedding"):
            for entry in model_catalog.FALLBACK_CATALOG[group]:
                with self.subTest(group=group, entry=entry):
                    self.assertTrue(entry.get("id"))
                    self.assertTrue(entry.get("name"))

    def test_every_fallback_id_is_a_valid_model_id(self):
        # A fallback the config normaliser would reject is worse than useless:
        # the picker would offer an option that silently does nothing.
        from common.runtime.context import normalize_model_id

        for group in ("chat", "embedding"):
            for entry in model_catalog.FALLBACK_CATALOG[group]:
                with self.subTest(group=group, model_id=entry["id"]):
                    self.assertEqual(normalize_model_id(entry["id"]), entry["id"])

    def test_the_pipelines_own_default_embedding_model_is_offerable(self):
        # It is what the shipped base corpus is indexed with, so the picker has
        # to be able to express it even with no network.
        ids = {e["id"] for e in model_catalog.FALLBACK_CATALOG["embedding"]}
        self.assertIn("google/gemini-embedding-2-preview", ids)


if __name__ == "__main__":
    unittest.main()
