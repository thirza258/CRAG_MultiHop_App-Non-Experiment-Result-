import { DeleteDocumentResponse } from "../interface";
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

    const response = await apiClient.post("/insert-data/", formData, {
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
    const response = await apiClient.post(
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
    const response = await apiClient.post("/insert-text/", {
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
    `/document/${username}/`,
    jsonConfig
  );
  return response.data;
}

const getConversationHistory = async (username: string) => {
  const response = await apiClient.get(
    `/conversation-history/${username}/`,
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
  documentId: string,
  username: string
): Promise<DeleteDocumentResponse> => {
  try {
    const response = await apiClient.delete<DeleteDocumentResponse>(
      `/document/${encodeURIComponent(documentId)}/${encodeURIComponent(username)}/`
    );

    return response.data;
  } catch (error: unknown) {
    if (error instanceof AxiosError) {
      const apiError = {
        status: error.response?.status ?? 500,
        message:
          error.response?.data?.error ||
          "An error occurred while deleting the document",
        originalError: error,
      };

      throw apiError;
    }

    throw new Error(
      "Network error: unable to reach the document deletion endpoint"
    );
  }
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
};
