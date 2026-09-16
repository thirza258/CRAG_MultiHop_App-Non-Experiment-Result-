import { DeleteDocumentResponse, DocumentStatus, UploadResponse } from "../interface";
import { ApiKeys, CorpusInfo, PipelineConfig } from "../types/types";
import { apiClient } from "./apiClient";

/**
 * Per-request settings the insert endpoints accept.
 *
 * The embedding model is the reason these are on the *upload* path at all: a
 * ChromaDB collection holds one vector space, so which model embeds a document
 * is decided when it is indexed, not when it is queried. The keys are here so
 * a user who brought their own pays for their own indexing.
 */
export type RequestSettings = {
  config?: Partial<PipelineConfig>;
  keys?: Partial<ApiKeys>;
};

/** JSON body fields, omitted entirely when there is nothing to send. */
const settingsBody = (settings?: RequestSettings) => {
  const body: Record<string, unknown> = {};
  if (settings?.config && Object.keys(settings.config).length) {
    body.CONFIG = settings.config;
  }
  if (settings?.keys && Object.keys(settings.keys).length) {
    body.KEYS = settings.keys;
  }
  return body;
};

/**
 * The same two fields for a multipart upload. multipart/form-data has no nested
 * objects, so they go as JSON text — the backend's JSONObjectField accepts
 * either spelling.
 */
const appendSettings = (formData: FormData, settings?: RequestSettings) => {
  if (settings?.config && Object.keys(settings.config).length) {
    formData.append("CONFIG", JSON.stringify(settings.config));
  }
  if (settings?.keys && Object.keys(settings.keys).length) {
    formData.append("KEYS", JSON.stringify(settings.keys));
  }
};

const signUp = async (email: string, username: string) => {
    const response = await apiClient.post("/sign-up/", {
        "USERNAME": username,
        "EMAIL": email,
    },
    {
        headers: {
            "Content-Type": "application/json",
        },
    });
    return response.data;
};

const submitFile = async ( 
    file: File,
    username: string,
    settings?: RequestSettings
) => {
    const formData = new FormData();
    formData.append("FILE", file);
    formData.append("USER", username);
    appendSettings(formData, settings);

    const response = await apiClient.post<UploadResponse>("/insert-data/", formData, {
        headers: {
            "Content-Type": "multipart/form-data",
        },
    });

    return response.data;
};

const submitURL = async (
    url: string,
    username: string,
    settings?: RequestSettings
) => {
    const response = await apiClient.post<UploadResponse>(
        "/insert-url/",
        {
            URL: url,
            USER: username,
            ...settingsBody(settings),
        },
        {
            headers: {
                "Content-Type": "application/json",
            },
        }
    );

    return response.data;
};

const submitText = async (
    text: string,
    username: string,
    settings?: RequestSettings
) => {
    const response = await apiClient.post<UploadResponse>("/insert-text/", {
        TEXT: text,
        USER: username,
        ...settingsBody(settings),
    },
    {
        headers: {
            "Content-Type": "application/json",
        },
    }
);

    return response.data;
};


const cleanSystem = async () => {
    const response = await apiClient.get("/clean/");
    return response.data;
};

const jsonConfig = {
  headers: {
    "Content-Type": "application/json",
    "Accept": "application/json",
  },
};

const getConversation = async (conversation_id: string) => {
  const response = await apiClient.get(
    `/conversation/${conversation_id}/`,
    jsonConfig
  );
  return response.data;
};

const getDocumentInfo = async (username: string) => {
  const response = await apiClient.get(
    `/document/${encodeURIComponent(username)}/`,
    jsonConfig
  );
  return response.data;
}

const getConversationHistory = async (username: string) => {
  const response = await apiClient.get(
    `/conversation-history/${encodeURIComponent(username)}/`,
    jsonConfig
  );
  return response.data;
}

/**
 * Which embedding model each corpus is pinned to.
 *
 * The settings panel needs this to explain a locked embedding picker up front,
 * rather than letting the user pick a model and only telling them afterwards
 * that the corpus overrode it. Returns null on failure — the panel then simply
 * offers the choice, and the pipeline's own notice covers the override.
 */
const getCorpusInfo = async (username: string): Promise<CorpusInfo | null> => {
  try {
    const response = await apiClient.get(
      `/corpus/${encodeURIComponent(username)}/`,
      jsonConfig
    );
    const body = response.data;
    const info = body && typeof body === "object" && "user_corpus" in body
      ? body
      : body?.data;
    return info && typeof info === "object" ? (info as CorpusInfo) : null;
  } catch (error) {
    console.warn("Could not read corpus info:", error);
    return null;
  }
};


import { AxiosError } from "axios";

const deleteDocument = async (
  documentId: string | number,
  username: string
): Promise<DeleteDocumentResponse> => {
  const response = await apiClient.delete<DeleteDocumentResponse>(
    `/document/${encodeURIComponent(documentId)}/${encodeURIComponent(username)}/`
  );
  return response.data;
};

export const DOCUMENTS_CHANGED = "crag:documents-changed";
export const HISTORY_CHANGED = "crag:history-changed";
export const documentsChanged = () => window.dispatchEvent(new Event(DOCUMENTS_CHANGED));

export const errorMessage = (error: unknown): string => {
  if (error instanceof AxiosError) {
    const data = error.response?.data;
    if (typeof data?.message === "string") return data.message;
    if (data && typeof data === "object") return Object.values(data).flat().join(" ");
  }
  return error instanceof Error ? error.message : "Request failed. Please try again.";
};

const waitForDocument = async (
  initial: DocumentStatus,
  username: string,
  onStatus: (document: DocumentStatus) => void,
  signal: AbortSignal,
): Promise<DocumentStatus> => {
  let document = initial;
  const deadline = Date.now() + 15 * 60_000;
  while (!signal.aborted) {
    onStatus(document);
    if (document.status === "ready") return document;
    if (document.status === "failed") {
      throw new Error(document.error_message || "Indexing failed. Check your API key and upload again.");
    }
    if (Date.now() > deadline) {
      throw new Error("Indexing is taking longer than expected. Check the document status before asking a question.");
    }
    await new Promise<void>((resolve, reject) => {
      const abort = () => { clearTimeout(timer); reject(new DOMException("Cancelled", "AbortError")); };
      const timer = setTimeout(() => { signal.removeEventListener("abort", abort); resolve(); }, 1500);
      signal.addEventListener("abort", abort, { once: true });
    });
    const response = await apiClient.get<{ data: DocumentStatus }>(
      `/document/${document.document_id}/${encodeURIComponent(username)}/`, { signal },
    );
    document = response.data.data;
  }
  throw new DOMException("Cancelled", "AbortError");
};

export default {
    submitFile,
    submitURL,
    cleanSystem,
    signUp,
    submitText,
    getConversation,
    getDocumentInfo,
    getConversationHistory,
    getCorpusInfo,
    deleteDocument,
    waitForDocument,
};
