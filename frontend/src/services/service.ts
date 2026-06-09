import { DeleteDocumentResponse } from "../interface";
import { apiClient } from "./apiClient";

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
    username: string
) => {
    const formData = new FormData();
    formData.append("FILE", file);
    formData.append("USER", username);

    const response = await apiClient.post("/insert-data/", formData, {
        headers: {
            "Content-Type": "multipart/form-data",
        },
    });

    return response.data;
};

const submitURL = async (
    url: string,
    username: string
) => {
    const response = await apiClient.post(
        "/insert-url/",
        {
            URL: url,
            USER: username,
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
    username: string
) => {
    const response = await apiClient.post("/insert-text/", {
        TEXT: text,
        USER: username,
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
    deleteDocument,
};
