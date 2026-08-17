"""Tests for per-query pipeline configuration normalisation.

The load-bearing property here is backward compatibility: a request that sends
no config, a broken config, or a config full of junk must end up running the
same pipeline the app ran before configuration existed.
"""

import unittest

try:
    from common import pipeline_config as pc
    IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - import guard
    pc = None
    IMPORT_ERROR = str(exc)


@unittest.skipIf(pc is None, f"common.pipeline_config unavailable: {IMPORT_ERROR}")
class NormalizeDefaultsTests(unittest.TestCase):
    def test_none_returns_defaults(self):
        self.assertEqual(
            pc.normalize_pipeline_config(None), pc.DEFAULT_PIPELINE_CONFIG
        )

    def test_empty_dict_returns_defaults(self):
        self.assertEqual(pc.normalize_pipeline_config({}), pc.DEFAULT_PIPELINE_CONFIG)

    def test_defaults_describe_the_historic_full_pipeline(self):
        # If any of these flip, every existing query silently changes behaviour.
        self.assertTrue(pc.DEFAULT_PIPELINE_CONFIG["use_multi_hop"])
        self.assertTrue(pc.DEFAULT_PIPELINE_CONFIG["use_corrective"])
        self.assertTrue(pc.DEFAULT_PIPELINE_CONFIG["use_reranker"])
        self.assertEqual(pc.DEFAULT_PIPELINE_CONFIG["max_hops"], 3)
        self.assertEqual(pc.DEFAULT_PIPELINE_CONFIG["retrievers"], "both")
        self.assertEqual(pc.DEFAULT_PIPELINE_CONFIG["corpus"], "auto")

    def test_result_is_a_copy_not_the_shared_default(self):
        result = pc.normalize_pipeline_config(None)
        result["max_hops"] = 99
        self.assertEqual(pc.DEFAULT_PIPELINE_CONFIG["max_hops"], 3)

    def test_always_returns_every_key(self):
        result = pc.normalize_pipeline_config({"use_multi_hop": False})
        self.assertEqual(set(result), set(pc.DEFAULT_PIPELINE_CONFIG))

    def test_unknown_keys_are_dropped(self):
        result = pc.normalize_pipeline_config({"evil": "payload", "max_hops": 2})
        self.assertNotIn("evil", result)
        self.assertEqual(result["max_hops"], 2)

    def test_non_dict_input_falls_back_to_defaults(self):
        for junk in ("string", 42, [1, 2, 3], True, object()):
            with self.subTest(junk=junk):
                self.assertEqual(
                    pc.normalize_pipeline_config(junk), pc.DEFAULT_PIPELINE_CONFIG
                )


@unittest.skipIf(pc is None, f"common.pipeline_config unavailable: {IMPORT_ERROR}")
class BooleanCoercionTests(unittest.TestCase):
    def test_real_booleans_pass_through(self):
        result = pc.normalize_pipeline_config(
            {"use_multi_hop": False, "use_corrective": False, "use_reranker": False}
        )
        self.assertFalse(result["use_multi_hop"])
        self.assertFalse(result["use_corrective"])
        self.assertFalse(result["use_reranker"])

    def test_string_spellings_are_accepted(self):
        for truthy in ("true", "TRUE", " True ", "1", "yes", "on"):
            with self.subTest(value=truthy):
                self.assertTrue(
                    pc.normalize_pipeline_config({"use_corrective": truthy})[
                        "use_corrective"
                    ]
                )
        for falsy in ("false", "FALSE", " off ", "0", "no"):
            with self.subTest(value=falsy):
                self.assertFalse(
                    pc.normalize_pipeline_config({"use_corrective": falsy})[
                        "use_corrective"
                    ]
                )

    def test_unrecognised_value_keeps_the_default(self):
        result = pc.normalize_pipeline_config({"use_corrective": "maybe"})
        self.assertTrue(result["use_corrective"])

    def test_none_value_keeps_the_default(self):
        self.assertTrue(
            pc.normalize_pipeline_config({"use_reranker": None})["use_reranker"]
        )


@unittest.skipIf(pc is None, f"common.pipeline_config unavailable: {IMPORT_ERROR}")
class MaxHopsTests(unittest.TestCase):
    def test_in_range_values_are_kept(self):
        for hops in (1, 2, 3):
            with self.subTest(hops=hops):
                self.assertEqual(
                    pc.normalize_pipeline_config({"max_hops": hops})["max_hops"], hops
                )

    def test_numeric_strings_are_accepted(self):
        self.assertEqual(
            pc.normalize_pipeline_config({"max_hops": "2"})["max_hops"], 2
        )

    def test_too_high_is_clamped_not_rejected(self):
        self.assertEqual(
            pc.normalize_pipeline_config({"max_hops": 99})["max_hops"], pc.MAX_HOPS
        )

    def test_too_low_is_clamped(self):
        for hops in (0, -5):
            with self.subTest(hops=hops):
                self.assertEqual(
                    pc.normalize_pipeline_config({"max_hops": hops})["max_hops"],
                    pc.MIN_HOPS,
                )

    def test_garbage_falls_back_to_default(self):
        for junk in ("many", None, [3], {}, 1.5e400):
            with self.subTest(junk=junk):
                self.assertEqual(
                    pc.normalize_pipeline_config({"max_hops": junk})["max_hops"],
                    pc.DEFAULT_PIPELINE_CONFIG["max_hops"],
                )

    def test_booleans_are_not_treated_as_hop_counts(self):
        # bool is an int subclass; True must not silently mean "1 hop".
        self.assertEqual(
            pc.normalize_pipeline_config({"max_hops": True})["max_hops"],
            pc.DEFAULT_PIPELINE_CONFIG["max_hops"],
        )

    def test_float_is_truncated_then_clamped(self):
        self.assertEqual(
            pc.normalize_pipeline_config({"max_hops": 2.9})["max_hops"], 2
        )


@unittest.skipIf(pc is None, f"common.pipeline_config unavailable: {IMPORT_ERROR}")
class ChoiceFieldTests(unittest.TestCase):
    def test_every_declared_retriever_choice_is_accepted(self):
        for choice in pc.RETRIEVER_CHOICES:
            with self.subTest(choice=choice):
                self.assertEqual(
                    pc.normalize_pipeline_config({"retrievers": choice})["retrievers"],
                    choice,
                )

    def test_every_declared_corpus_choice_is_accepted(self):
        for choice in pc.CORPUS_CHOICES:
            with self.subTest(choice=choice):
                self.assertEqual(
                    pc.normalize_pipeline_config({"corpus": choice})["corpus"], choice
                )

    def test_choices_are_case_insensitive_and_trimmed(self):
        self.assertEqual(
            pc.normalize_pipeline_config({"retrievers": " DENSE "})["retrievers"],
            "dense",
        )

    def test_invalid_choice_falls_back_to_default(self):
        self.assertEqual(
            pc.normalize_pipeline_config({"retrievers": "quantum"})["retrievers"],
            "both",
        )
        self.assertEqual(
            pc.normalize_pipeline_config({"corpus": "everything"})["corpus"], "auto"
        )

    def test_at_least_one_retriever_is_always_selected(self):
        # The single-choice design means "no retrievers at all" is unreachable.
        for value in (None, "", "nonsense", 0, [], {"dense": False}):
            with self.subTest(value=value):
                self.assertIn(
                    pc.normalize_pipeline_config({"retrievers": value})["retrievers"],
                    pc.RETRIEVER_CHOICES,
                )


@unittest.skipIf(pc is None, f"common.pipeline_config unavailable: {IMPORT_ERROR}")
class HelperTests(unittest.TestCase):
    def test_is_default_true_for_missing_and_explicit_defaults(self):
        self.assertTrue(pc.is_default(None))
        self.assertTrue(pc.is_default({}))
        self.assertTrue(pc.is_default(dict(pc.DEFAULT_PIPELINE_CONFIG)))

    def test_is_default_false_when_a_stage_is_disabled(self):
        self.assertFalse(pc.is_default({"use_corrective": False}))
        self.assertFalse(pc.is_default({"max_hops": 1}))

    def test_describe_mentions_each_stage(self):
        summary = pc.describe(
            {"use_multi_hop": False, "use_corrective": False, "use_reranker": True}
        )
        self.assertIn("multi_hop=off", summary)
        self.assertIn("corrective=off", summary)
        self.assertIn("reranker=on", summary)

    def test_describe_reports_the_hop_ceiling_when_enabled(self):
        self.assertIn("multi_hop=x2", pc.describe({"max_hops": 2}))

    def test_describe_never_raises_on_junk(self):
        self.assertIsInstance(pc.describe("nonsense"), str)


if __name__ == "__main__":
    unittest.main()
