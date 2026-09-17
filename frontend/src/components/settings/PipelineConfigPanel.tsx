import React, { useCallback, useEffect, useState } from "react";
import { ChevronDown, RefreshCw, RotateCcw, SlidersHorizontal } from "lucide-react";
import {
  LIMITS,
  usePipelineConfig,
} from "../../context/PipelineConfigContext";
import {
  ChunkStrategy,
  CorpusChoice,
  CorpusInfo,
  ModelCatalog,
  RetrieverChoice,
  StrictnessChoice,
} from "../../types/types";
import ModelPicker from "./ModelPicker";
import { NumberField, Section, Segmented, SegmentedOption, Toggle } from "./SettingsControls";
import service, { DOCUMENTS_CHANGED } from "../../services/service";
import {
  EMPTY_CATALOG,
  cachedCatalog,
  fetchModelCatalog,
} from "../../services/modelCatalog";

/**
 * Every stage, model and threshold the pipeline exposes, grouped so the sidebar
 * stays usable at roughly twenty-five controls.
 *
 * Two things are deliberately not free choices, and the panel says so rather
 * than accepting a setting the pipeline will override:
 *
 * - The **embedding model** is pinned to whatever the searched corpus was built
 *   with. A collection is one vector space; embedding a query with a different
 *   model compares across spaces.
 * - **Corrective strictness** is a preset, not three thresholds. The grader
 *   normalises similarity into [0, 1], so the band between "accept" and
 *   "escalate" is a few hundredths wide and its decision logic needs
 *   upper > lower — three sliders can express a pipeline that grades everything
 *   as wrong.
 */

const CORPUS_OPTIONS: SegmentedOption<CorpusChoice>[] = [
  { value: "auto", label: "Auto", hint: "Your documents when you have any, otherwise the base corpus" },
  { value: "user", label: "My docs", hint: "Only documents you uploaded" },
  { value: "base", label: "Base corpus", hint: "Only the bundled MultiHop-RAG news corpus" },
];

const RETRIEVER_OPTIONS: SegmentedOption<RetrieverChoice>[] = [
  { value: "both", label: "Both", hint: "Dense embeddings + BM25 keyword search" },
  { value: "dense", label: "Dense", hint: "Embedding similarity only" },
  { value: "sparse", label: "BM25", hint: "Keyword search only" },
];

const STRICTNESS_OPTIONS: SegmentedOption<StrictnessChoice>[] = [
  { value: "", label: "Default", hint: "Use the grading thresholds this deployment is configured with" },
  { value: "lenient", label: "Lenient", hint: "Accept more retrieved context; escalates to the web less often" },
  { value: "balanced", label: "Balanced", hint: "The thresholds the app ships with" },
  { value: "strict", label: "Strict", hint: "Demand closer matches; escalates to the web more often" },
];

const CHUNK_STRATEGY_OPTIONS: SegmentedOption<ChunkStrategy>[] = [
  { value: "", label: "Default", hint: "Use the splitter this deployment is configured with" },
  { value: "recursive", label: "Recursive", hint: "Split on paragraph, then sentence, then word boundaries" },
  { value: "paragraph", label: "Paragraph", hint: "Keep paragraphs whole, merging them up to the chunk size" },
  { value: "fixed", label: "Fixed", hint: "Cut every chunk_size characters regardless of structure" },
  { value: "semantic", label: "Semantic", hint: "Split where the topic shifts, measured by sentence embeddings (costs an embedding call per upload)" },
];

/** Which corpus a query will actually search, given the corpus setting. */
const activeCorpus = (
  corpus: CorpusChoice,
  info: CorpusInfo | null
): "user" | "base" => {
  if (corpus === "base") return "base";
  if (corpus === "user") return "user";
  // "auto" searches the user's own documents when they have any.
  return (info?.user_corpus?.chunk_count ?? 0) > 0 ? "user" : "base";
};

/** "2 off" / "off", for the section headers. */
const offBadge = (flags: boolean[]): string | undefined => {
  const off = flags.filter((flag) => !flag).length;
  if (!off) return undefined;
  return off === flags.length ? "all off" : `${off} off`;
};

const PipelineConfigPanel: React.FC = () => {
  const { config, setConfig, resetConfig, isDefault } = usePipelineConfig();
  const [open, setOpen] = useState(true);
  // Paint from the browser cache first so the pickers are usable on the very
  // first render, then replace with whatever the server returns.
  const [catalog, setCatalog] = useState<ModelCatalog>(
    () => cachedCatalog() ?? EMPTY_CATALOG
  );
  const [catalogLoading, setCatalogLoading] = useState(false);
  const [corpusInfo, setCorpusInfo] = useState<CorpusInfo | null>(null);

  const loadCatalog = useCallback(async (force = false) => {
    setCatalogLoading(true);
    try {
      setCatalog(await fetchModelCatalog({ force }));
    } finally {
      setCatalogLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadCatalog();
  }, [loadCatalog]);

  useEffect(() => {
    const username = localStorage.getItem("username");
    if (!username) return;
    let cancelled = false;
    const refresh = () => service.getCorpusInfo(username).then((info) => {
      if (!cancelled) setCorpusInfo(info);
    });
    void refresh();
    window.addEventListener(DOCUMENTS_CHANGED, refresh);
    return () => {
      cancelled = true;
      window.removeEventListener(DOCUMENTS_CHANGED, refresh);
    };
  }, []);

  const target = activeCorpus(config.corpus, corpusInfo);
  const pin = target === "base" ? corpusInfo?.base_corpus : corpusInfo?.user_corpus;
  const embeddingLocked = Boolean(pin?.locked && pin?.embedding_model);

  const lockedReason =
    target === "base"
      ? "The shared base corpus is indexed with this model, and it is not yours to re-index. Switch Corpus to your own documents to choose one."
      : `Your ${pin?.chunk_count ?? 0} indexed chunks were embedded with this model. A query has to use the same one, so switching means deleting your documents and re-uploading them.`;

  // The catalog is only "unavailable" when there is genuinely nothing to show;
  // a stale cached list is fine and needs no warning.
  const catalogUnavailable = !catalogLoading && !catalog.chat.length;

  return (
    <section className="flex-shrink-0 rounded border border-[hsl(var(--border))] bg-[hsl(var(--background))]">
      <div className="flex items-center justify-between px-3 py-2">
        <button
          type="button"
          onClick={() => setOpen(!open)}
          aria-expanded={open}
          className="flex min-w-0 items-center gap-2 text-sm font-semibold text-[hsl(var(--foreground))]"
        >
          <SlidersHorizontal className="h-3.5 w-3.5 text-[hsl(var(--primary))]" aria-hidden="true" />
          Pipeline
          <ChevronDown
            className={`h-3.5 w-3.5 text-[hsl(var(--muted-foreground))] transition-transform ${
              open ? "" : "-rotate-90"
            }`}
            aria-hidden="true"
          />
        </button>
        {!isDefault && (
          <button
            type="button"
            onClick={resetConfig}
            title="Reset every setting to the full default pipeline"
            className="flex items-center gap-1 text-[11px] text-[hsl(var(--muted-foreground))] transition-colors hover:text-[hsl(var(--primary))]"
          >
            <RotateCcw className="h-3 w-3" aria-hidden="true" />
            Reset
          </button>
        )}
      </div>

      {open && (
        <div className="space-y-3 border-t border-[hsl(var(--border))] px-3 py-3">
          {/* ── Stages ─────────────────────────────────────────────────── */}
          <Section
            title="Stages"
            defaultOpen
            badge={offBadge([
              config.use_multi_hop,
              config.use_corrective,
              config.use_reranker,
              config.use_evaluation,
            ])}
          >
            <Toggle
              label="Multi-hop"
              hint="Decompose the question into follow-up lookups"
              checked={config.use_multi_hop}
              onChange={(use_multi_hop) => setConfig({ use_multi_hop })}
            />

            {config.use_multi_hop && (
              <div className="pl-6">
                <NumberField
                  label="Max hops"
                  hint="How many follow-up lookups a question may chain. Each hop costs a decision call plus a retrieval, so the ceiling is 3."
                  value={config.max_hops}
                  fallback={3}
                  min={LIMITS.max_hops.min}
                  max={LIMITS.max_hops.max}
                  unit="hops"
                  onChange={(max_hops) => setConfig({ max_hops })}
                />
              </div>
            )}

            <Toggle
              label="Corrective (CRAG)"
              hint="Grade retrieved context and correct it when weak. The grader is a locally downloaded model, so its strictness is adjustable below but the model itself is fixed."
              checked={config.use_corrective}
              onChange={(use_corrective) => setConfig({ use_corrective })}
            />

            <Toggle
              label="Reranker"
              hint="Reorder merged candidates with the cross-encoder. On/off only — it runs from a model downloaded to the server, not through OpenRouter, so it cannot be swapped per query."
              checked={config.use_reranker}
              onChange={(use_reranker) => setConfig({ use_reranker })}
            />

            <Toggle
              label="Answer scoring"
              hint="Rate the answer for relevancy and faithfulness. Off saves two LLM-judge passes per question."
              checked={config.use_evaluation}
              onChange={(use_evaluation) => setConfig({ use_evaluation })}
            />
          </Section>

          {/* ── Models ─────────────────────────────────────────────────── */}
          <Section title="Models" defaultOpen>
            <div className="flex items-center justify-end">
              <button
                type="button"
                onClick={() => void loadCatalog(true)}
                disabled={catalogLoading}
                title="Re-fetch the model list from OpenRouter"
                className="flex items-center gap-1 text-[10px] text-[hsl(var(--muted-foreground))] transition-colors hover:text-[hsl(var(--primary))] disabled:opacity-50"
              >
                <RefreshCw
                  className={`h-2.5 w-2.5 ${catalogLoading ? "animate-spin" : ""}`}
                  aria-hidden="true"
                />
                {catalogLoading ? "Loading" : "Refresh list"}
              </button>
            </div>

            <ModelPicker
              label="Chat model"
              hint="Used for the answer, hop decisions and query expansion."
              value={config.llm_model}
              options={catalog.chat}
              onChange={(llm_model) => setConfig({ llm_model })}
              catalogNote={
                catalogUnavailable
                  ? "Could not reach the model catalog — type a full OpenRouter id instead."
                  : undefined
              }
            />

            <ModelPicker
              label="Embedding model"
              hint="Applies when you index new documents. Dense retrieval always uses whatever the corpus it searches was built with."
              // When locked, show the model that actually built the vectors —
              // showing the user's stored pick would be showing them a lie.
              value={
                embeddingLocked ? pin?.embedding_model ?? "" : config.embedding_model
              }
              options={catalog.embedding}
              onChange={(embedding_model) => setConfig({ embedding_model })}
              locked={embeddingLocked}
              lockedReason={embeddingLocked ? lockedReason : undefined}
              defaultLabel={
                embeddingLocked ? undefined : corpusInfo?.configured_default || undefined
              }
              conflictingChoice={
                embeddingLocked &&
                config.embedding_model &&
                config.embedding_model !== pin?.embedding_model
                  ? config.embedding_model
                  : undefined
              }
              onClearChoice={() => setConfig({ embedding_model: "" })}
            />

            <NumberField
              label="Temperature"
              hint="How freely the answer is worded. The pipeline's own decisions (which documents to fetch) stay deterministic either way."
              value={config.temperature}
              fallback={0}
              min={LIMITS.temperature.min}
              max={LIMITS.temperature.max}
              step={LIMITS.temperature.step}
              onChange={(temperature) => setConfig({ temperature })}
            />
          </Section>

          {/* ── Retrieval ──────────────────────────────────────────────── */}
          <Section title="Retrieval" badge={config.remove_stop_words ? undefined : "1 off"}>
            <Segmented
              label="Corpus"
              value={config.corpus}
              options={CORPUS_OPTIONS}
              onChange={(corpus) => setConfig({ corpus })}
            />

            <Segmented
              label="Retrievers"
              value={config.retrievers}
              options={RETRIEVER_OPTIONS}
              onChange={(retrievers) => setConfig({ retrievers })}
            />

            <NumberField
              label="Chunks retrieved"
              hint="How many chunks each retriever fetches before merging."
              value={config.top_k}
              fallback={4}
              min={LIMITS.top_k.min}
              max={LIMITS.top_k.max}
              unit="chunks"
              onChange={(top_k) => setConfig({ top_k })}
            />

            <NumberField
              label="Chunks kept"
              hint="How many survive the merge — what the model actually reads."
              value={config.rerank_top_k}
              fallback={4}
              min={LIMITS.rerank_top_k.min}
              max={LIMITS.rerank_top_k.max}
              unit="chunks"
              onChange={(rerank_top_k) => setConfig({ rerank_top_k })}
            />

            <Toggle
              label="Drop stop words"
              hint="Strip 'the', 'is', 'on'… before BM25 keyword matching."
              checked={config.remove_stop_words}
              onChange={(remove_stop_words) => setConfig({ remove_stop_words })}
              disabled={config.retrievers === "dense"}
              disabledReason="Only affects BM25, which this query is not using."
            />
          </Section>

          {/* ── Corrective detail ──────────────────────────────────────── */}
          <Section
            title="Corrective detail"
            badge={
              !config.use_corrective
                ? "stage off"
                : offBadge([
                    config.use_external_search,
                    config.use_wikipedia,
                    config.use_news,
                    config.use_query_expansion,
                    config.use_knowledge_refinement,
                  ])
            }
          >
            <Segmented
              label="Strictness"
              value={config.corrective_strictness}
              options={STRICTNESS_OPTIONS}
              onChange={(corrective_strictness) => setConfig({ corrective_strictness })}
              disabled={!config.use_corrective}
            />

            <Toggle
              label="External search"
              hint="Let weak local context escalate to the web."
              checked={config.use_external_search}
              onChange={(use_external_search) => setConfig({ use_external_search })}
              disabled={!config.use_corrective}
              disabledReason="Corrective is switched off, so nothing escalates."
            />

            <div className="space-y-3 pl-6">
              <Toggle
                label="Wikipedia"
                hint="Fetch and grade Wikipedia articles."
                checked={config.use_wikipedia}
                onChange={(use_wikipedia) => setConfig({ use_wikipedia })}
                disabled={!config.use_corrective || !config.use_external_search}
                disabledReason="External search is switched off."
              />
              <Toggle
                label="News (NewsAPI)"
                hint="Fetch a recent news article. Needs a NewsAPI key — yours or the server's."
                checked={config.use_news}
                onChange={(use_news) => setConfig({ use_news })}
                disabled={!config.use_corrective || !config.use_external_search}
                disabledReason="External search is switched off."
              />

              <NumberField
                label="Web passages kept"
                hint="How many passages the web search keeps after scoring them against the question. Separate from the retrieval settings — these are fetched live, not read from the index."
                value={config.external_top_k}
                fallback={5}
                min={LIMITS.external_top_k.min}
                max={LIMITS.external_top_k.max}
                unit="passages"
                onChange={(external_top_k) => setConfig({ external_top_k })}
                disabled={!config.use_corrective || !config.use_external_search}
              />
            </div>

            <Toggle
              label="Query expansion"
              hint="LLM-rewritten keywords and phrasings. Off also removes the ambiguous-resolution retries, which are built on them."
              checked={config.use_query_expansion}
              onChange={(use_query_expansion) => setConfig({ use_query_expansion })}
              disabled={!config.use_corrective}
              disabledReason="Corrective is switched off."
            />

            {config.use_corrective && config.use_query_expansion && (
              <div className="pl-6">
                <NumberField
                  label="Alternative phrasings"
                  hint="How many rewordings the ambiguous retry may try."
                  value={config.expansion_queries}
                  fallback={3}
                  min={LIMITS.expansion_queries.min}
                  max={LIMITS.expansion_queries.max}
                  onChange={(expansion_queries) => setConfig({ expansion_queries })}
                />
              </div>
            )}

            <Toggle
              label="Knowledge refinement"
              hint="Re-score a borderline chunk sentence-group by sentence-group, keeping only the relevant parts."
              checked={config.use_knowledge_refinement}
              onChange={(use_knowledge_refinement) =>
                setConfig({ use_knowledge_refinement })
              }
              disabled={!config.use_corrective}
              disabledReason="Corrective is switched off."
            />
          </Section>

          {/* ── Indexing ───────────────────────────────────────────────── */}
          <Section title="Indexing">
            <p className="rounded border border-dashed border-[hsl(var(--border))] px-2 py-1.5 text-[11px] leading-snug text-[hsl(var(--muted-foreground))]">
              Applies to documents you upload from now on. Existing chunks are
              already split and are not affected.
            </p>

            <Segmented
              label="Chunking"
              value={config.chunk_strategy}
              options={CHUNK_STRATEGY_OPTIONS}
              onChange={(chunk_strategy) => setConfig({ chunk_strategy })}
            />

            <NumberField
              label="Chunk size"
              hint="Target characters per chunk."
              value={config.chunk_size}
              fallback={500}
              min={LIMITS.chunk_size.min}
              max={LIMITS.chunk_size.max}
              step={LIMITS.chunk_size.step}
              unit="chars"
              onChange={(chunk_size) => setConfig({ chunk_size })}
            />

            <NumberField
              label="Chunk overlap"
              hint="Characters repeated between neighbouring chunks, so a fact split across a boundary survives."
              value={config.chunk_overlap}
              fallback={50}
              min={LIMITS.chunk_overlap.min}
              max={LIMITS.chunk_overlap.max}
              step={LIMITS.chunk_overlap.step}
              unit="chars"
              onChange={(chunk_overlap) => setConfig({ chunk_overlap })}
            />
          </Section>

          {/* ── Scoring detail ─────────────────────────────────────────── */}
          <Section
            title="Scoring detail"
            badge={
              !config.use_evaluation
                ? "stage off"
                : offBadge([config.eval_answer_relevancy, config.eval_faithfulness])
            }
          >
            <Toggle
              label="Answer relevancy"
              hint="Does the answer address the question that was asked?"
              checked={config.eval_answer_relevancy}
              onChange={(eval_answer_relevancy) => setConfig({ eval_answer_relevancy })}
              disabled={!config.use_evaluation}
              disabledReason="Answer scoring is switched off."
            />

            <Toggle
              label="Faithfulness"
              hint="Is every claim in the answer supported by the retrieved chunks?"
              checked={config.eval_faithfulness}
              onChange={(eval_faithfulness) => setConfig({ eval_faithfulness })}
              disabled={!config.use_evaluation}
              disabledReason="Answer scoring is switched off."
            />

            <ModelPicker
              label="Judge model"
              hint="Left on the default, the judge is a different model from the one answering — a model scoring its own output flatters itself."
              value={config.evaluation_llm_model}
              options={catalog.chat}
              onChange={(evaluation_llm_model) => setConfig({ evaluation_llm_model })}
              disabled={!config.use_evaluation}
            />

            <ModelPicker
              label="Judge embedding model"
              hint="Used to measure answer relevancy. Nothing is stored, so any model is safe here."
              value={config.evaluation_embedding_model}
              options={catalog.embedding}
              onChange={(evaluation_embedding_model) =>
                setConfig({ evaluation_embedding_model })
              }
              disabled={!config.use_evaluation}
            />
          </Section>

          {!isDefault && (
            <p className="rounded border border-dashed border-[hsl(var(--border))] px-2 py-1.5 text-[11px] leading-snug text-[hsl(var(--muted-foreground))]">
              Running a customised pipeline. Answers may be less grounded than the
              default configuration.
            </p>
          )}
        </div>
      )}
    </section>
  );
};

export default PipelineConfigPanel;
