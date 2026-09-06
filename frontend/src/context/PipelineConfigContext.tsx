import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { PipelineConfig } from "../types/types";

/**
 * Per-query pipeline configuration, shared between the Sidebar (which edits it)
 * and the Chatbot (which sends it with every query).
 *
 * Keys are snake_case on purpose: this object is serialised straight onto the
 * websocket payload as CONFIG, and the backend contract lives in
 * backend/common/runtime/config.py. Keep the two in sync.
 *
 * API keys deliberately live in a *different* context (ApiKeysContext): this
 * object is persisted to localStorage wholesale and logged verbatim by the
 * server, neither of which a credential should be part of.
 */

export const DEFAULT_PIPELINE_CONFIG: PipelineConfig = {
  // Stage composition — all on, which is the historic pipeline.
  use_multi_hop: true,
  max_hops: null,
  use_corrective: true,
  use_reranker: true,
  use_evaluation: true,
  retrievers: "both",
  corpus: "auto",

  // Corrective sub-stages.
  use_external_search: true,
  use_wikipedia: true,
  use_news: true,
  use_query_expansion: true,
  expansion_queries: null,
  use_knowledge_refinement: true,
  corrective_strictness: "",

  // Retrieval sizing.
  top_k: null,
  rerank_top_k: null,
  external_top_k: null,
  remove_stop_words: true,

  // Generation. "" / null mean "let the server use what it has configured", so
  // a user who never opens these gets exactly the pipeline they had before they
  // existed.
  llm_model: "",
  temperature: null,

  // Embedding / indexing.
  embedding_model: "",
  chunk_strategy: "",
  chunk_size: null,
  chunk_overlap: null,

  // Evaluation detail.
  evaluation_llm_model: "",
  evaluation_embedding_model: "",
  eval_answer_relevancy: true,
  eval_faithfulness: true,
};

export const MIN_HOPS = 1;
export const MAX_HOPS = 3;

/**
 * Bounds mirroring backend/common/runtime/config.py. The server clamps to
 * these anyway; matching them here means a slider cannot offer a value that
 * silently becomes a different one.
 */
export const LIMITS = {
  top_k: { min: 1, max: 20 },
  rerank_top_k: { min: 1, max: 20 },
  temperature: { min: 0, max: 2, step: 0.1 },
  chunk_size: { min: 100, max: 4000, step: 50 },
  chunk_overlap: { min: 0, max: 1000, step: 10 },
  expansion_queries: { min: 1, max: 5 },
  external_top_k: { min: 1, max: 20 },
  max_hops: { min: MIN_HOPS, max: MAX_HOPS },
} as const;

/**
 * Mirrors _MODEL_ID_RE in backend/common/runtime/context.py. Validating here as
 * well means the UI cannot offer (or accept as custom input) an id the backend
 * would silently drop — which would look like a setting that does nothing.
 */
export const MODEL_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._/:-]{0,127}$/;

export const isValidModelId = (value: string): boolean =>
  MODEL_ID_PATTERN.test(value.trim());

const STORAGE_KEY = "pipeline_config";

type PipelineConfigContextValue = {
  config: PipelineConfig;
  setConfig: (patch: Partial<PipelineConfig>) => void;
  resetConfig: () => void;
  isDefault: boolean;
};

const PipelineConfigContext = createContext<PipelineConfigContextValue | null>(null);

/** Every boolean field, so none is forgotten when one is added. */
const BOOLEAN_FIELDS = [
  "use_multi_hop",
  "use_corrective",
  "use_reranker",
  "use_evaluation",
  "use_external_search",
  "use_wikipedia",
  "use_news",
  "use_query_expansion",
  "use_knowledge_refinement",
  "remove_stop_words",
  "eval_answer_relevancy",
  "eval_faithfulness",
] as const;

const MODEL_FIELDS = [
  "llm_model",
  "embedding_model",
  "evaluation_llm_model",
  "evaluation_embedding_model",
] as const;

/** The one nullable number that is genuinely fractional; the rest are counts. */
const FRACTIONAL_NUMBERS = new Set<string>(["temperature"]);

/** Numeric fields where `null` is a real value meaning "defer to the server". */
const NULLABLE_NUMBERS = [
  "top_k",
  "rerank_top_k",
  "temperature",
  "chunk_size",
  "chunk_overlap",
  "expansion_queries",
  "external_top_k",
  // Nullable for the same reason as the rest: a deployment may have tuned
  // multi_hop_config down, and an untouched panel must not push it back up.
  "max_hops",
] as const;

/** Merge stored/unknown input onto the defaults, dropping anything invalid. */
const sanitize = (raw: unknown): PipelineConfig => {
  const base = { ...DEFAULT_PIPELINE_CONFIG };
  if (!raw || typeof raw !== "object") return base;
  const input = raw as Record<string, unknown>;

  BOOLEAN_FIELDS.forEach((key) => {
    if (typeof input[key] === "boolean") {
      (base[key] as boolean) = input[key] as boolean;
    }
  });

  if (input.retrievers === "both" || input.retrievers === "dense" || input.retrievers === "sparse") {
    base.retrievers = input.retrievers;
  }
  if (input.corpus === "auto" || input.corpus === "user" || input.corpus === "base") {
    base.corpus = input.corpus;
  }
  if (
    input.corrective_strictness === "" ||
    input.corrective_strictness === "lenient" ||
    input.corrective_strictness === "balanced" ||
    input.corrective_strictness === "strict"
  ) {
    base.corrective_strictness = input.corrective_strictness;
  }
  if (
    input.chunk_strategy === "" ||
    input.chunk_strategy === "fixed" ||
    input.chunk_strategy === "paragraph" ||
    input.chunk_strategy === "semantic" ||
    input.chunk_strategy === "recursive"
  ) {
    base.chunk_strategy = input.chunk_strategy;
  }

  // null is a real value here ("defer"), so it has to survive the round trip —
  // a plain `typeof === "number"` branch would quietly reset it every write.
  NULLABLE_NUMBERS.forEach((key) => {
    const value = input[key];
    if (value === null) {
      base[key] = null;
      return;
    }
    if (typeof value !== "number" || !Number.isFinite(value)) return;
    const { min, max } = LIMITS[key];
    const whole = FRACTIONAL_NUMBERS.has(key) ? value : Math.trunc(value);
    base[key] = Math.min(max, Math.max(min, whole));
  });

  // A blank model is meaningful ("use the server default"), so it is accepted
  // alongside a valid id; anything else is dropped rather than sent onward.
  MODEL_FIELDS.forEach((key) => {
    const value = input[key];
    if (typeof value !== "string") return;
    const trimmed = value.trim();
    if (trimmed === "" || isValidModelId(trimmed)) {
      base[key] = trimmed;
    }
  });

  return base;
};

const readStored = (): PipelineConfig => {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    return sanitize(stored ? JSON.parse(stored) : null);
  } catch {
    // Corrupt JSON in localStorage must never stop the app from loading.
    return { ...DEFAULT_PIPELINE_CONFIG };
  }
};

export const PipelineConfigProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [config, setConfigState] = useState<PipelineConfig>(readStored);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(config));
    } catch {
      // Private-mode / quota failures are not worth breaking a query over.
    }
  }, [config]);

  const setConfig = useCallback((patch: Partial<PipelineConfig>) => {
    setConfigState((prev) => sanitize({ ...prev, ...patch }));
  }, []);

  const resetConfig = useCallback(() => {
    setConfigState({ ...DEFAULT_PIPELINE_CONFIG });
  }, []);

  const value = useMemo<PipelineConfigContextValue>(
    () => ({
      config,
      setConfig,
      resetConfig,
      isDefault: (Object.keys(DEFAULT_PIPELINE_CONFIG) as (keyof PipelineConfig)[]).every(
        (key) => config[key] === DEFAULT_PIPELINE_CONFIG[key]
      ),
    }),
    [config, setConfig, resetConfig]
  );

  return (
    <PipelineConfigContext.Provider value={value}>{children}</PipelineConfigContext.Provider>
  );
};

/**
 * Falls back to the defaults when used outside a provider, so a component can
 * read the config without the app crashing if the tree changes.
 */
export const usePipelineConfig = (): PipelineConfigContextValue => {
  const ctx = useContext(PipelineConfigContext);
  if (ctx) return ctx;
  return {
    config: { ...DEFAULT_PIPELINE_CONFIG },
    setConfig: () => undefined,
    resetConfig: () => undefined,
    isDefault: true,
  };
};
