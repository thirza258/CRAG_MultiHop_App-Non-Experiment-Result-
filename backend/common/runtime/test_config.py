"""Tests for per-query pipeline configuration normalisation.

The load-bearing property here is backward compatibility: a request that sends
no config, a broken config, or a config full of junk must end up running the
same pipeline the app ran before configuration existed.
"""

import unittest

try:
    from common.runtime import config as pc
    IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - import guard
    pc = None
    IMPORT_ERROR = str(exc)


@unittest.skipIf(pc is None, f"common.runtime.config unavailable: {IMPORT_ERROR}")
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
        # None, not 3: an untouched request must defer to whatever the
        # deployment put in multi_hop_config rather than push it back to the
        # schema's own number. MAX_HOPS is the ceiling, not the default.
        self.assertIsNone(pc.DEFAULT_PIPELINE_CONFIG["max_hops"])
        self.assertEqual(pc.DEFAULT_PIPELINE_CONFIG["retrievers"], "both")
        self.assertEqual(pc.DEFAULT_PIPELINE_CONFIG["corpus"], "auto")

    def test_result_is_a_copy_not_the_shared_default(self):
        result = pc.normalize_pipeline_config(None)
        result["max_hops"] = 99
        self.assertIsNone(pc.DEFAULT_PIPELINE_CONFIG["max_hops"])

    def test_always_returns_every_key(self):
        result = pc.normalize_pipeline_config({"use_multi_hop": False})
        self.assertEqual(set(result), set(pc.DEFAULT_PIPELINE_CONFIG))

    def test_every_sizing_knob_defers_by_default(self):
        # A number the panel never touched must not overwrite what the
        # deployment configured. "" does this for named choices, None for numbers.
        for field in ("top_k", "rerank_top_k", "external_top_k", "max_hops",
                      "chunk_size", "chunk_overlap", "expansion_queries",
                      "temperature"):
            with self.subTest(field=field):
                self.assertIsNone(pc.DEFAULT_PIPELINE_CONFIG[field])

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


@unittest.skipIf(pc is None, f"common.runtime.config unavailable: {IMPORT_ERROR}")
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


@unittest.skipIf(pc is None, f"common.runtime.config unavailable: {IMPORT_ERROR}")
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


@unittest.skipIf(pc is None, f"common.runtime.config unavailable: {IMPORT_ERROR}")
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


@unittest.skipIf(pc is None, f"common.runtime.config unavailable: {IMPORT_ERROR}")
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

    def test_describe_says_on_rather_than_xnone_for_an_untouched_ceiling(self):
        # max_hops defers by default, and "multi_hop=xNone" would go into every
        # status event the user sees.
        self.assertIn("multi_hop=on", pc.describe({}))
        self.assertNotIn("None", pc.describe({}))

    def test_external_top_k_is_bounded_like_the_other_counts(self):
        self.assertEqual(
            pc.normalize_pipeline_config({"external_top_k": 8})["external_top_k"], 8
        )
        self.assertEqual(
            pc.normalize_pipeline_config({"external_top_k": 999})["external_top_k"],
            pc.MAX_TOP_K,
        )
        self.assertIsNone(
            pc.normalize_pipeline_config({"external_top_k": "lots"})["external_top_k"]
        )

    def test_describe_never_raises_on_junk(self):
        self.assertIsInstance(pc.describe("nonsense"), str)


@unittest.skipIf(pc is None, f"common.runtime.config unavailable: {IMPORT_ERROR}")
class ModelSelectionTests(unittest.TestCase):
    """The per-query OpenRouter model choice.

    An unset model must mean "whatever each stage was already configured with",
    so a client that knows nothing about model selection keeps the exact
    behaviour it had.
    """

    def test_models_default_to_empty_meaning_the_configured_default(self):
        config = pc.normalize_pipeline_config(None)
        self.assertEqual(config["llm_model"], "")
        self.assertEqual(config["embedding_model"], "")

    def test_an_unset_model_still_counts_as_the_default_pipeline(self):
        self.assertTrue(pc.is_default({"llm_model": "", "embedding_model": ""}))

    def test_choosing_a_model_is_not_the_default_pipeline(self):
        self.assertFalse(pc.is_default({"llm_model": "openai/gpt-4o"}))

    def test_real_openrouter_ids_are_accepted(self):
        for model_id in (
            "openai/gpt-4o",
            "mistralai/mistral-nemo",
            "qwen/qwen3-30b-a3b-instruct-2507",
            "liquid/lfm-2.5-embedding-350m:free",
        ):
            with self.subTest(model_id=model_id):
                self.assertEqual(
                    pc.normalize_pipeline_config({"llm_model": model_id})["llm_model"],
                    model_id,
                )

    def test_embedding_model_is_normalised_independently(self):
        config = pc.normalize_pipeline_config(
            {"llm_model": "openai/gpt-4o", "embedding_model": "baai/bge-m3"}
        )
        self.assertEqual(config["llm_model"], "openai/gpt-4o")
        self.assertEqual(config["embedding_model"], "baai/bge-m3")

    def test_a_malformed_model_id_falls_back_instead_of_failing(self):
        # The id is interpolated into an upstream API request, so it is
        # validated rather than passed through.
        for junk in ("../../etc/passwd", "model with spaces", "'; DROP TABLE--", 42, [], None):
            with self.subTest(junk=junk):
                config = pc.normalize_pipeline_config({"llm_model": junk})
                self.assertEqual(config["llm_model"], "")

    def test_describe_reports_both_models(self):
        summary = pc.describe(
            {"llm_model": "openai/gpt-4o", "embedding_model": "baai/bge-m3"}
        )
        self.assertIn("llm=openai/gpt-4o", summary)
        self.assertIn("embedding=baai/bge-m3", summary)

    def test_describe_says_default_when_no_model_was_chosen(self):
        summary = pc.describe(None)
        self.assertIn("llm=default", summary)
        self.assertIn("embedding=default", summary)

    def test_a_credential_shaped_field_cannot_ride_along_in_config(self):
        # CONFIG is logged verbatim and summarised into status events shown to
        # the user, so keys travel in a separate KEYS field. Anything
        # credential-looking put here is dropped as an unknown key.
        secret = "sk-or-v1-0123456789abcdef"
        config = pc.normalize_pipeline_config(
            {"openrouter_api_key": secret, "api_key": secret, "KEYS": {"news": secret}}
        )
        self.assertEqual(set(config), set(pc.DEFAULT_PIPELINE_CONFIG))
        self.assertNotIn(secret, repr(config))
        self.assertNotIn(secret, pc.describe(config))


@unittest.skipIf(pc is None, f"common.runtime.config unavailable: {IMPORT_ERROR}")
class DeferSentinelTests(unittest.TestCase):
    """`None` for a number and `""` for a name mean "use what the deployment
    configured" — deliberately not "use the value written here".

    Giving these concrete defaults would override every deployment's own tuning
    in rag_service.py with one hard-coded number, which is the opposite of
    configurable.
    """

    NUMERIC_DEFER_FIELDS = (
        "top_k", "rerank_top_k", "temperature", "chunk_size", "chunk_overlap",
        "expansion_queries",
    )
    NAMED_DEFER_FIELDS = (
        "llm_model", "embedding_model", "evaluation_llm_model",
        "evaluation_embedding_model", "chunk_strategy", "corrective_strictness",
    )

    def test_every_numeric_knob_defaults_to_defer(self):
        config = pc.normalize_pipeline_config(None)
        for field in self.NUMERIC_DEFER_FIELDS:
            with self.subTest(field=field):
                self.assertIsNone(config[field])

    def test_every_named_knob_defaults_to_defer(self):
        config = pc.normalize_pipeline_config(None)
        for field in self.NAMED_DEFER_FIELDS:
            with self.subTest(field=field):
                self.assertEqual(config[field], "")

    def test_a_deferring_config_is_still_the_default_pipeline(self):
        self.assertTrue(pc.is_default({field: None for field in self.NUMERIC_DEFER_FIELDS}))

    def test_junk_defers_rather_than_inventing_a_number(self):
        for junk in ("abc", [], {}, True, float("nan")):
            with self.subTest(junk=junk):
                config = pc.normalize_pipeline_config({"top_k": junk, "temperature": junk})
                self.assertIsNone(config["top_k"])
                self.assertIsNone(config["temperature"])

    def test_infinity_is_clamped_or_deferred_but_never_passed_on(self):
        # Following the module's existing rule that out-of-range is clamped
        # rather than rejected: as a float it becomes the maximum, and as an int
        # it cannot be converted at all, so it defers. Either way nothing
        # unbounded reaches an API request.
        config = pc.normalize_pipeline_config(
            {"top_k": float("inf"), "temperature": float("inf")}
        )
        self.assertIsNone(config["top_k"])
        self.assertEqual(config["temperature"], pc.MAX_TEMPERATURE)


@unittest.skipIf(pc is None, f"common.runtime.config unavailable: {IMPORT_ERROR}")
class NewStageToggleTests(unittest.TestCase):
    NEW_TOGGLES = (
        "use_evaluation", "use_external_search", "use_wikipedia", "use_news",
        "use_query_expansion", "use_knowledge_refinement", "remove_stop_words",
        "eval_answer_relevancy", "eval_faithfulness",
    )

    def test_every_new_toggle_defaults_on(self):
        # On is the historic behaviour, so a client that knows nothing about
        # these gets exactly the pipeline it had before they existed.
        config = pc.normalize_pipeline_config(None)
        for field in self.NEW_TOGGLES:
            with self.subTest(field=field):
                self.assertTrue(config[field])

    def test_each_can_be_switched_off_independently(self):
        for field in self.NEW_TOGGLES:
            with self.subTest(field=field):
                config = pc.normalize_pipeline_config({field: False})
                self.assertFalse(config[field])
                others = [f for f in self.NEW_TOGGLES if f != field]
                self.assertTrue(all(config[other] for other in others))

    def test_switching_one_off_is_not_the_default_pipeline(self):
        for field in self.NEW_TOGGLES:
            with self.subTest(field=field):
                self.assertFalse(pc.is_default({field: False}))


@unittest.skipIf(pc is None, f"common.runtime.config unavailable: {IMPORT_ERROR}")
class NumericClampTests(unittest.TestCase):
    def test_top_k_is_clamped_to_the_supported_range(self):
        self.assertEqual(pc.normalize_pipeline_config({"top_k": 999})["top_k"], pc.MAX_TOP_K)
        self.assertEqual(pc.normalize_pipeline_config({"top_k": 0})["top_k"], pc.MIN_TOP_K)
        self.assertEqual(pc.normalize_pipeline_config({"top_k": -5})["top_k"], pc.MIN_TOP_K)

    def test_temperature_is_clamped(self):
        self.assertEqual(pc.normalize_pipeline_config({"temperature": 9})["temperature"], pc.MAX_TEMPERATURE)
        self.assertEqual(pc.normalize_pipeline_config({"temperature": -1})["temperature"], pc.MIN_TEMPERATURE)

    def test_temperature_zero_is_a_real_value_not_a_defer(self):
        # 0.0 is the deterministic setting the pipeline ships with, so it has to
        # survive the falsy check that "defer" uses.
        self.assertEqual(pc.normalize_pipeline_config({"temperature": 0})["temperature"], 0.0)

    def test_a_numeric_string_is_accepted(self):
        self.assertEqual(pc.normalize_pipeline_config({"top_k": "6"})["top_k"], 6)
        self.assertEqual(pc.normalize_pipeline_config({"temperature": "0.7"})["temperature"], 0.7)

    def test_chunk_overlap_is_corrected_when_it_would_not_leave_room(self):
        # overlap >= size makes a fixed-size chunker either loop forever or emit
        # one chunk per character.
        config = pc.normalize_pipeline_config({"chunk_size": 500, "chunk_overlap": 500})
        self.assertLess(config["chunk_overlap"], config["chunk_size"])

        config = pc.normalize_pipeline_config({"chunk_size": 400, "chunk_overlap": 900})
        self.assertLess(config["chunk_overlap"], config["chunk_size"])

    def test_a_sane_overlap_is_left_alone(self):
        config = pc.normalize_pipeline_config({"chunk_size": 800, "chunk_overlap": 100})
        self.assertEqual(config["chunk_size"], 800)
        self.assertEqual(config["chunk_overlap"], 100)


@unittest.skipIf(pc is None, f"common.runtime.config unavailable: {IMPORT_ERROR}")
class CorrectiveStrictnessTests(unittest.TestCase):
    """Strictness is a preset rather than three loose floats.

    The grader normalises similarity into [0, 1], so the useful band between
    "accept this chunk" and "escalate to the web" is a few hundredths wide, and
    its decision logic requires upper > lower. A preset cannot express an
    inverted pair; three sliders can.
    """

    def test_it_defaults_to_deferring_to_the_deployment(self):
        self.assertEqual(pc.normalize_pipeline_config(None)["corrective_strictness"], "")
        self.assertIsNone(pc.corrective_thresholds(None))

    def test_every_preset_is_accepted(self):
        for preset in pc.STRICTNESS_CHOICES:
            with self.subTest(preset=preset):
                config = pc.normalize_pipeline_config({"corrective_strictness": preset})
                self.assertEqual(config["corrective_strictness"], preset)

    def test_an_unknown_preset_defers(self):
        self.assertEqual(
            pc.normalize_pipeline_config({"corrective_strictness": "extremely"})[
                "corrective_strictness"
            ],
            "",
        )

    def test_every_preset_keeps_the_ordering_the_grader_depends_on(self):
        for preset in pc.STRICTNESS_CHOICES:
            with self.subTest(preset=preset):
                thresholds = pc.corrective_thresholds(
                    {"corrective_strictness": preset}
                )
                self.assertGreater(thresholds["upper"], thresholds["strip"])
                self.assertGreater(thresholds["strip"], thresholds["lower"])

    def test_presets_are_ordered_from_lenient_to_strict(self):
        lenient = pc.corrective_thresholds({"corrective_strictness": "lenient"})
        balanced = pc.corrective_thresholds({"corrective_strictness": "balanced"})
        strict = pc.corrective_thresholds({"corrective_strictness": "strict"})

        self.assertLess(lenient["upper"], balanced["upper"])
        self.assertLess(balanced["upper"], strict["upper"])

    def test_balanced_reproduces_the_shipped_thresholds(self):
        # rag/rag_service.py has always used these; the preset has to be able to
        # express the behaviour the app had before it existed.
        self.assertEqual(
            pc.corrective_thresholds({"corrective_strictness": "balanced"}),
            {"upper": 0.91, "lower": 0.87, "strip": 0.88},
        )


@unittest.skipIf(pc is None, f"common.runtime.config unavailable: {IMPORT_ERROR}")
class ChunkStrategyTests(unittest.TestCase):
    def test_every_supported_strategy_is_accepted(self):
        for strategy in pc.CHUNK_STRATEGY_CHOICES:
            with self.subTest(strategy=strategy):
                self.assertEqual(
                    pc.normalize_pipeline_config({"chunk_strategy": strategy})[
                        "chunk_strategy"
                    ],
                    strategy,
                )

    def test_an_unknown_strategy_defers(self):
        # DocumentChunker raises ValueError on an unknown strategy, so passing
        # one through would turn a typo into a failed upload.
        self.assertEqual(
            pc.normalize_pipeline_config({"chunk_strategy": "magic"})["chunk_strategy"],
            "",
        )


@unittest.skipIf(pc is None, f"common.runtime.config unavailable: {IMPORT_ERROR}")
class DescribeNewFieldsTests(unittest.TestCase):
    def test_the_core_stages_are_always_named(self):
        summary = pc.describe(None)
        for expected in ("multi_hop=", "corrective=", "reranker=", "evaluation="):
            self.assertIn(expected, summary)

    def test_an_untouched_knob_is_not_listed(self):
        # This runs on every query; a 25-field dump would bury the useful part.
        summary = pc.describe(None)
        self.assertNotIn("use_wikipedia", summary)
        self.assertNotIn("chunk_size", summary)

    def test_a_changed_knob_is_listed(self):
        summary = pc.describe({"use_wikipedia": False, "top_k": 9})
        self.assertIn("use_wikipedia=False", summary)
        self.assertIn("top_k=9", summary)

    def test_describe_never_raises_on_the_new_fields(self):
        self.assertIsInstance(pc.describe({"temperature": "hot", "top_k": []}), str)


if __name__ == "__main__":
    unittest.main()
