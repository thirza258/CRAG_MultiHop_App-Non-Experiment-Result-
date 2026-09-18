import { ChatContextResponse } from "../interface";

export type UserRole = "me" | "bot";

export type EvalScore = {
  answer_relevancy: number | null;
  faithfulness: number | null;
};

export type Message = {
  user: UserRole;
  text: string;
  conversationId?: string;
  context?: ChatContextResponse[]; 
  evaluation?: EvalScore;
  /**
   * Things the pipeline could not honour verbatim — e.g. the corpus is indexed
   * with a different embedding model than the one selected. Shown with the
   * answer, because a setting that was silently overridden is worse than one
   * that was refused out loud.
   */
  notices?: string[];
};

export type EmbeddingStatus = "embedded" | "pending" | "failed";

/**
 * Per-query pipeline configuration, sent as CONFIG on the query websocket.
 * snake_case keys mirror backend/common/runtime/config.py exactly.
 */
export type RetrieverChoice = "both" | "dense" | "sparse";
export type CorpusChoice = "auto" | "user" | "base";
/** "" defers to whatever thresholds the deployment configured. */
export type StrictnessChoice = "" | "lenient" | "balanced" | "strict";
/** "" defers to the deployment's chunker. */
export type ChunkStrategy = "" | "fixed" | "paragraph" | "semantic" | "recursive";

/**
 * Every knob a query carries.
 *
 * Two sentinels mean "the user did not choose", and they matter: `null` for a
 * number and `""` for a model id or named strategy. Both defer to whatever the
 * deployment configured server-side — deliberately not "use the value written
 * here", which would override every deployment's own tuning with one number
 * baked into the client.
 *
 * Booleans are the exception and carry real defaults, all matching the historic
 * behaviour, because on/off leaves no room for a third state.
 */
export type PipelineConfig = {
  // Stage composition
  use_multi_hop: boolean;
  max_hops: number | null;
  use_corrective: boolean;
  use_reranker: boolean;
  use_evaluation: boolean;
  retrievers: RetrieverChoice;
  corpus: CorpusChoice;

  // Corrective sub-stages
  use_external_search: boolean;
  use_wikipedia: boolean;
  use_news: boolean;
  use_query_expansion: boolean;
  expansion_queries: number | null;
  use_knowledge_refinement: boolean;
  corrective_strictness: StrictnessChoice;

  // Retrieval sizing
  top_k: number | null;
  rerank_top_k: number | null;
  external_top_k: number | null;
  remove_stop_words: boolean;

  // Generation
  llm_model: string;
  temperature: number | null;

  // Embedding / indexing
  embedding_model: string;
  chunk_strategy: ChunkStrategy;
  chunk_size: number | null;
  chunk_overlap: number | null;

  // Evaluation detail
  evaluation_llm_model: string;
  evaluation_embedding_model: string;
  eval_answer_relevancy: boolean;
  eval_faithfulness: boolean;
};

/**
 * Per-request API keys ("bring your own key"), sent as a top-level KEYS field
 * beside CONFIG — never inside it. CONFIG is logged verbatim on the server;
 * KEYS is redacted. The backend contract lives in backend/common/runtime/api_keys.py.
 */
export type ApiKeyName = "openrouter" | "news";

export type ApiKeys = Record<ApiKeyName, string>;

/** One entry from the OpenRouter catalog, trimmed to what a picker renders. */
export type ModelOption = {
  id: string;
  name: string;
  context_length?: number | null;
  /** USD per input token, as reported by OpenRouter. */
  prompt_price?: number | null;
  completion_price?: number | null;
  free?: boolean;
};

/**
 * OpenRouter publishes chat and embedding models from two separate endpoints —
 * embedding models are absent from the main /models catalog entirely — so they
 * arrive here as two separate lists.
 */
export type ModelCatalog = {
  chat: ModelOption[];
  embedding: ModelOption[];
  /** "openrouter" | "cache" | "fallback" — shown when it is not live. */
  source?: string;
  fetched_at?: number;
};

/**
 * Which embedding model a corpus's vectors actually came from. A collection is
 * one vector space, so once it holds anything the model cannot change without
 * re-indexing — `locked` is what lets the picker explain itself up front
 * instead of overriding the choice after the query has run.
 */
export type CorpusPin = {
  collection_name: string | null;
  embedding_model: string;
  chunk_count?: number;
  locked: boolean;
};

export type CorpusInfo = {
  user_corpus: CorpusPin;
  base_corpus: CorpusPin;
  configured_default: string;
};

export type SubmitPayload =
  | { type: "file"; file: File }
  | { type: "url"; url: string }
  | { type: "text"; text: string };

export interface DocStep {
  id: number;
  slug: string;
  title: string;
  description: React.ReactNode;
  icon: React.ReactNode;
  imagePath?: string;
  imageAlt: string;
  imagePlaceholderText: string;
}

export type DeepResultContextType = {
  setIds: (ids: { conversationId: string; documentId: string }) => void;
};

export type ErrorState = {
  error: string;
  status: number;
  message: string;
};

export type JobStatus = "PENDING" | "PROCESSING" | "READY" | "FAILED";

export type ChatbotContextType = {
  username: string | null;
  documentId: string | null;
  setContext: (context: { username: string; documentId: string }) => void;
};