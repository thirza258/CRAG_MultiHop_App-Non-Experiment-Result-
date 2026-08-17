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
};

export type EmbeddingStatus = "embedded" | "pending" | "failed";

/**
 * Per-query pipeline configuration, sent as CONFIG on the query websocket.
 * snake_case keys mirror backend/common/pipeline_config.py exactly.
 */
export type RetrieverChoice = "both" | "dense" | "sparse";
export type CorpusChoice = "auto" | "user" | "base";

export type PipelineConfig = {
  use_multi_hop: boolean;
  max_hops: number;
  use_corrective: boolean;
  use_reranker: boolean;
  retrievers: RetrieverChoice;
  corpus: CorpusChoice;
};

export type SubmitPayload =
  | { type: "file"; file: File }
  | { type: "url"; url: string }
  | { type: "text"; text: string };

export interface DocStep {
  id: number;
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