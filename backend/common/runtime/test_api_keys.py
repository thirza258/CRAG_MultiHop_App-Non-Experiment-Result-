"""Tests for per-request API key handling.

Two load-bearing properties here, and they pull in opposite directions:

* **Liberal on the way in** — a malformed KEYS block must fall back to the
  server's own credentials, never fail the query it was attached to.
* **Airtight on the way out** — a key the user pasted must not appear in a log
  line, a status event, or an error message returned to the browser. The tests
  that assert absence are the reason this module exists.
"""

import unittest
import unittest.mock

try:
    from common.runtime import api_keys as ak
    IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - import guard
    ak = None
    IMPORT_ERROR = str(exc)

SECRET = "sk-or-v1-0123456789abcdef0123456789abcdef"
NEWS_SECRET = "9f8e7d6c5b4a39281706abcdef123456"


@unittest.skipIf(ak is None, f"common.runtime.api_keys unavailable: {IMPORT_ERROR}")
class NormalizeTests(unittest.TestCase):
    def test_none_yields_every_field_empty(self):
        self.assertEqual(
            ak.normalize_api_keys(None), {field: "" for field in ak.KEY_FIELDS}
        )

    def test_non_dict_input_yields_every_field_empty(self):
        for junk in ("string", 42, [SECRET], True, object()):
            with self.subTest(junk=junk):
                self.assertEqual(
                    ak.normalize_api_keys(junk),
                    {field: "" for field in ak.KEY_FIELDS},
                )

    def test_always_returns_every_field(self):
        result = ak.normalize_api_keys({"openrouter": SECRET})
        self.assertEqual(set(result), set(ak.KEY_FIELDS))

    def test_canonical_names_are_accepted(self):
        result = ak.normalize_api_keys({"openrouter": SECRET, "news": NEWS_SECRET})
        self.assertEqual(result["openrouter"], SECRET)
        self.assertEqual(result["news"], NEWS_SECRET)

    def test_env_style_and_mixed_case_aliases_are_accepted(self):
        # A real client built the payload from an env-var name; dropping the key
        # would silently spend the server's credit instead of the user's.
        for name in ("OPENROUTER_API_KEY", "openrouter_key", "OpenRouter"):
            with self.subTest(name=name):
                self.assertEqual(
                    ak.normalize_api_keys({name: SECRET})["openrouter"], SECRET
                )
        for name in ("NEWS_API_KEY", "newsapi", "news_api"):
            with self.subTest(name=name):
                self.assertEqual(
                    ak.normalize_api_keys({name: NEWS_SECRET})["news"], NEWS_SECRET
                )

    def test_surrounding_whitespace_is_stripped(self):
        self.assertEqual(
            ak.normalize_api_keys({"openrouter": f"  {SECRET}\n"})["openrouter"],
            SECRET,
        )

    def test_unknown_keys_are_dropped(self):
        result = ak.normalize_api_keys({"evil": SECRET, "news": NEWS_SECRET})
        self.assertNotIn("evil", result)
        self.assertEqual(result["news"], NEWS_SECRET)

    def test_blank_and_non_string_values_are_ignored(self):
        for value in ("", "   ", None, 42, [], {}):
            with self.subTest(value=value):
                self.assertEqual(
                    ak.normalize_api_keys({"openrouter": value})["openrouter"], ""
                )

    def test_a_value_with_a_newline_is_rejected_not_sanitised(self):
        # This key would go into an Authorization header. Stripping the newline
        # and sending the remainder would look like a mysterious 401; keeping it
        # would be header injection.
        for value in (
            f"{SECRET}\r\nX-Injected: yes",
            f"{SECRET}\nfoo",
            f"{SECRET} with spaces",
        ):
            with self.subTest(value=value):
                self.assertEqual(
                    ak.normalize_api_keys({"openrouter": value})["openrouter"], ""
                )

    def test_an_over_long_value_is_rejected(self):
        self.assertEqual(
            ak.normalize_api_keys({"openrouter": "a" * (ak.MAX_KEY_LENGTH + 1)})["openrouter"],
            "",
        )

    def test_a_value_at_the_length_limit_is_accepted(self):
        at_limit = "a" * ak.MAX_KEY_LENGTH
        self.assertEqual(
            ak.normalize_api_keys({"openrouter": at_limit})["openrouter"], at_limit
        )


@unittest.skipIf(ak is None, f"common.runtime.api_keys unavailable: {IMPORT_ERROR}")
class ResolveTests(unittest.TestCase):
    def test_the_requests_own_key_wins(self):
        with unittest.mock.patch.dict(
            "os.environ", {"OPENROUTER_API_KEY": "server-key"}, clear=False
        ):
            self.assertEqual(ak.resolve("openrouter", {"openrouter": SECRET}), SECRET)

    def test_falls_back_to_the_environment(self):
        with unittest.mock.patch.dict(
            "os.environ", {"OPENROUTER_API_KEY": "server-key"}, clear=False
        ):
            self.assertEqual(ak.resolve("openrouter", {"openrouter": ""}), "server-key")
            self.assertEqual(ak.resolve("openrouter", None), "server-key")

    def test_empty_when_neither_exists(self):
        with unittest.mock.patch.dict(
            "os.environ", {"OPENROUTER_API_KEY": ""}, clear=False
        ):
            self.assertEqual(ak.resolve("openrouter", {}), "")

    def test_unknown_field_has_no_environment_fallback(self):
        self.assertEqual(ak.resolve("nonexistent-provider", {}), "")


@unittest.skipIf(ak is None, f"common.runtime.api_keys unavailable: {IMPORT_ERROR}")
class FingerprintTests(unittest.TestCase):
    def test_never_contains_the_key(self):
        printed = ak.fingerprint(SECRET)
        self.assertNotIn(SECRET, printed)
        # Not even a tail of it: this ends up in a log file that outlives the
        # request, so no partial-reveal convention here.
        self.assertNotIn(SECRET[-4:], printed)

    def test_is_stable_for_the_same_key(self):
        self.assertEqual(ak.fingerprint(SECRET), ak.fingerprint(f"  {SECRET} "))

    def test_differs_between_keys(self):
        self.assertNotEqual(ak.fingerprint(SECRET), ak.fingerprint(NEWS_SECRET))

    def test_missing_key_is_reported_as_none(self):
        for value in ("", "   ", None, 42):
            with self.subTest(value=value):
                self.assertEqual(ak.fingerprint(value), "none")


@unittest.skipIf(ak is None, f"common.runtime.api_keys unavailable: {IMPORT_ERROR}")
class ScrubTests(unittest.TestCase):
    def test_removes_a_supplied_key_from_a_provider_error(self):
        # The shape an OpenAI-SDK exception actually takes.
        message = (
            "Error code: 401 - {'error': {'message': 'No auth credentials found'}} "
            f"request_headers={{'Authorization': 'Bearer {SECRET}'}}"
        )
        scrubbed = ak.scrub(message, {"openrouter": SECRET})
        self.assertNotIn(SECRET, scrubbed)
        self.assertIn(ak.PLACEHOLDER, scrubbed)
        self.assertIn("401", scrubbed, "the useful part of the error must survive")

    def test_removes_a_news_key_from_a_url_in_an_error(self):
        message = f"HTTPError for url: https://newsapi.org/v2/everything?q=x&apiKey={NEWS_SECRET}"
        scrubbed = ak.scrub(message, {"news": NEWS_SECRET})
        self.assertNotIn(NEWS_SECRET, scrubbed)

    def test_removes_every_supplied_key_at_once(self):
        message = f"{SECRET} then {NEWS_SECRET}"
        scrubbed = ak.scrub(message, {"openrouter": SECRET, "news": NEWS_SECRET})
        self.assertNotIn(SECRET, scrubbed)
        self.assertNotIn(NEWS_SECRET, scrubbed)

    def test_a_key_that_prefixes_another_does_not_leave_a_tail(self):
        # Replacing the shorter first would leave the remainder of the longer
        # one visible, which is why secret_values sorts longest-first.
        short = "abcdefghij"
        long = short + "klmnopqrst"
        scrubbed = ak.scrub(f"see {long}", {"a": short, "b": long})
        self.assertNotIn(short, scrubbed)
        self.assertNotIn(long, scrubbed)

    def test_leaves_text_alone_when_no_key_was_supplied(self):
        message = "Error code: 500 - upstream unavailable"
        self.assertEqual(ak.scrub(message, {}), message)
        self.assertEqual(ak.scrub(message, None), message)

    def test_very_short_values_are_not_treated_as_secrets(self):
        # Blanket-replacing a 3-character "key" would mangle unrelated text.
        self.assertEqual(ak.scrub("the cat sat", {"openrouter": "cat"}), "the cat sat")

    def test_non_string_input_never_raises(self):
        self.assertEqual(ak.scrub(None, {"openrouter": SECRET}), "")
        self.assertEqual(ak.scrub(42, {"openrouter": SECRET}), "")


@unittest.skipIf(ak is None, f"common.runtime.api_keys unavailable: {IMPORT_ERROR}")
class RedactTests(unittest.TestCase):
    def _payload(self):
        return {
            "USER": "bob",
            "QUERY": "who wrote it?",
            "CONFIG": {"use_reranker": False, "llm_model": "openai/gpt-4o"},
            "KEYS": {"openrouter": SECRET, "news": NEWS_SECRET},
        }

    def test_keys_values_are_replaced(self):
        redacted = ak.redact(self._payload())
        self.assertEqual(redacted["KEYS"]["openrouter"], ak.PLACEHOLDER)
        self.assertEqual(redacted["KEYS"]["news"], ak.PLACEHOLDER)

    def test_no_secret_survives_anywhere_in_the_repr(self):
        # The consumer logs this with an f-string, so repr() is the real test.
        redacted = repr(ak.redact(self._payload()))
        self.assertNotIn(SECRET, redacted)
        self.assertNotIn(NEWS_SECRET, redacted)

    def test_the_loggable_parts_are_preserved(self):
        redacted = ak.redact(self._payload())
        self.assertEqual(redacted["USER"], "bob")
        self.assertEqual(redacted["QUERY"], "who wrote it?")
        self.assertEqual(redacted["CONFIG"]["llm_model"], "openai/gpt-4o")

    def test_an_absent_key_stays_distinguishable_from_a_redacted_one(self):
        redacted = ak.redact({"KEYS": {"openrouter": SECRET, "news": ""}})
        self.assertEqual(redacted["KEYS"]["openrouter"], ak.PLACEHOLDER)
        self.assertEqual(redacted["KEYS"]["news"], "")

    def test_a_credential_named_field_anywhere_is_redacted(self):
        redacted = ak.redact(
            {"outer": {"inner": {"api_key": SECRET, "auth_token": SECRET}}}
        )
        self.assertNotIn(SECRET, repr(redacted))

    def test_lists_are_traversed(self):
        redacted = ak.redact({"items": [{"api_key": SECRET}, {"safe": "value"}]})
        self.assertNotIn(SECRET, repr(redacted))
        self.assertEqual(redacted["items"][1]["safe"], "value")

    def test_deeply_nested_input_is_capped_instead_of_recursing_forever(self):
        payload = current = {}
        for _ in range(50):
            current["next"] = {}
            current = current["next"]
        self.assertIsNotNone(ak.redact(payload))

    def test_a_self_referential_payload_never_hangs(self):
        payload = {"USER": "bob"}
        payload["self"] = payload
        self.assertIsNotNone(ak.redact(payload))

    def test_non_dict_payloads_pass_through(self):
        self.assertEqual(ak.redact("plain string"), "plain string")
        self.assertEqual(ak.redact(None), None)


@unittest.skipIf(ak is None, f"common.runtime.api_keys unavailable: {IMPORT_ERROR}")
class DescribeTests(unittest.TestCase):
    def test_reports_which_side_supplied_each_key_without_the_value(self):
        summary = ak.describe({"openrouter": SECRET, "news": ""})
        self.assertIn("openrouter=own", summary)
        self.assertIn("news=server", summary)
        self.assertNotIn(SECRET, summary)

    def test_survives_junk(self):
        self.assertIsInstance(ak.describe(None), str)
        self.assertIsInstance(ak.describe("nonsense"), str)


if __name__ == "__main__":
    unittest.main()
