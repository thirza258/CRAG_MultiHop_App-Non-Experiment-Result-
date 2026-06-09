
import { EvalScore } from "./types/types";

export interface ChatContextResponse {
  text: string;
  metadata: {
    document_id: number;
    chunk_index: number;
    username: string;
  };
}

export interface ChatResponse {
  status: number;
  message: string;
  timestamp: number;
  data: {
    answer: string;
    context?: ChatContextResponse[];  // fixed: array of objects, not strings
    source: string;
    conversation_id?: number;
    evaluation?: EvalScore;
  };
}

export interface RetrievedChunk {
  id: string | number;
  text: string;
  score?: number;
}

export interface EvaluationMetric {
  chunk_evaluation: Record<string, number>;
    response_evaluation: Record<string, number>;
    retrieval_score?: { chunk_id: string; score: number }[];
}

export interface AnalysisResult {
  batch_id: string;
  method: string;
  aiModel: string;
  query: string;
  answer: string;
  retrievedChunks: RetrievedChunk[];
  evaluation?: EvaluationMetric;
  progress: number;
  error?: string;
}

export interface WebSocketMessage {
  status?: string;
  method?: string;
  query?: string;
  aiModel?: string;
  progress?: number;
  batch_id?: string;
  error?: string;
  answer?: string;
  context?: {
    text: string;
    chunk_id: number | string;
    score?: number;
  }[];
  evaluation?: EvaluationMetric;
}

export interface NormalizedChunk {
  number: number;
  id: string | number;
  text: string;
  score?: number;
}

export interface StartAnalysisResponse {
  message: string;
  batch_id: string;
  expected_count: number;
}

export interface HistoryItem {
  id: string;
  title: string;
  type: "file" | "url";
  date: string; 
}

export interface TaskData {
  id: string;
  prompt: string;
}

export interface Chunk {
  id: string;
  text: string;
}

export interface DeepAnalysisConfig {
  topK: number;
  chunkSize: number;
  retrievalMethods: string[];
  selectedModels: string[];
}

export interface FileMetadata {
  id: string;
  name: string;
  source_type: string;
  source_path: string;
  extracted_text_path: string;
  created_at: string;
}

export interface ConversationItem {
  query: string;
  response: string;
  created_at: string;
}

export interface DeleteDocumentResponse {
  response: string; 
}