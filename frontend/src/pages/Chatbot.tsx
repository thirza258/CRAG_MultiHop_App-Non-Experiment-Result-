import { useState, useRef, useEffect } from "react";
import service, { documentsChanged, errorMessage, HISTORY_CHANGED } from "../services/service";
import { UploadResponse } from "../interface";
import { ChatMessage } from "../components/ui/chatmessage";
import { Message, SubmitPayload } from "../types/types";
import FileUploadSection from "../components/file/FileInput";
import UrlUploadSection from "../components/file/URLInput";
import TextUploadSection from "../components/file/TextInput";
import { useNavigate } from "react-router-dom";
import { generateChatStream, ChatStream } from "../services/websocket";
import { useSeo } from "../lib/seo";
import { usePipelineConfig } from "../context/PipelineConfigContext";
import { useApiKeys } from "../context/ApiKeysContext";

function Chatbot() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState<string>("");
  const [chatLoading, setChatLoading] = useState<boolean>(false);
  const [fileLocal, setFileLocal] = useState<File | null>(null);
  const [url, setLocalUrl] = useState<string>("");
  const [textInput, setTextInput] = useState<string>("");
  const [isModalOpen, setIsModalOpen] = useState<boolean>(false);
  const [modalSubmitting, setModalSubmitting] = useState<boolean>(false);
  const [statusText, setStatusText] = useState<string>("");

  const wsRef = useRef<ChatStream | null>(null);
  const uploadRef = useRef<AbortController | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const navigate = useNavigate();
  const { config: pipelineConfig, setConfig } = usePipelineConfig();
  const { keysForRequest } = useApiKeys();

  // sendMessage runs inside websocket callbacks, so read the live config from a
  // ref rather than closing over the value at render time.
  const configRef = useRef(pipelineConfig);
  configRef.current = pipelineConfig;

  // Same reason, and the keys change independently of the config.
  const keysRef = useRef(keysForRequest);
  keysRef.current = keysForRequest;

  useSeo({
    title: "Chat with your documents | CRAG MultiHop RAG",
    description:
      "Ask questions about your uploaded PDFs, URLs and notes, and watch the corrective multi-hop retrieval pipeline answer them.",
    path: "/chat",
    noindex: true,
  });

  useEffect(() => {
    const username = localStorage.getItem("username");
    if (!username) {
      navigate("/login");
    }
  }, [navigate]);

  const resetModalInputs = () => {
    setFileLocal(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
    setLocalUrl("");
    setTextInput("");
  };

  const sendMessage = (): void => {
  if (!input.trim() || chatLoading || modalSubmitting) return;

  // close any existing connection first
  if (wsRef.current) {
    wsRef.current.close();
  }

  const userMessage: Message = { user: "me", text: input };
  setMessages((prev) => [...prev, userMessage]);
  setInput("");
  setChatLoading(true);
  setStatusText("Starting...");

  const username = localStorage.getItem("username") || "";

  const ws = generateChatStream(
    input,
    username,
    // onStatus
    (statusMsg) => setStatusText(statusMsg),
    // onResult
    (msg) => {
      sessionStorage.removeItem(`chat_history_${username}`);
      window.dispatchEvent(new Event(HISTORY_CHANGED));
      setMessages((prev) => [...prev, {
        user: "bot",
        text: msg.answer,
        conversationId: msg.conversation_id?.toString(),
        context: msg.context || [],
        evaluation: msg.evaluation,
        // Settings the pipeline overrode for this answer, e.g. the embedding
        // model the searched corpus is actually indexed with.
        notices: [...(msg.notices || []), ...(msg.degraded?.length ? [`Some stages were unavailable: ${msg.degraded.join(", ")}.`] : [])],
      }]);
      console.log("Final answer received:", msg.answer);
      console.log("Evaluation received:", msg.evaluation);
      setStatusText("");
      setChatLoading(false);
    },
    // onError
    (errorMsg) => {
      setMessages((prev) => [...prev, {
        user: "bot",
        text: errorMsg || "Sorry, something went wrong.",
      }]);
      setStatusText("");
      setChatLoading(false);
    },
    // config
    configRef.current,
    // keys — a separate top-level field, never folded into config
    keysRef.current(),
  );
  wsRef.current = ws;
};

// cleanup on unmount
useEffect(() => {
  return () => {
    wsRef.current?.close();
    uploadRef.current?.abort();
  };
}, []);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      const selectedFile = event.target.files[0];
      setFileLocal(selectedFile);
      setLocalUrl("");
      setTextInput("");
    }
  };

  const handleUrlChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const value = event.target.value;
    setLocalUrl(value);
    if (value) {
      setFileLocal(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
      setTextInput("");
    }
  };

  const handleTextChange = (event: React.ChangeEvent<HTMLTextAreaElement>) => {
    const value = event.target.value;
    setTextInput(value);
    if (value) {
      setFileLocal(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
      setLocalUrl("");
    }
  };

  const handleModalSubmit = async () => {
    if (!fileLocal && !url.trim() && !textInput.trim()) {
      alert("Please upload a file, enter a URL, or paste text.");
      return;
    }

    if (modalSubmitting) return;

    let userMessageText = "";
    let payload: SubmitPayload;

    if (fileLocal) {
      userMessageText = `Uploaded file: ${fileLocal.name}`;
      payload = { type: "file", file: fileLocal };
    } else if (url) {
      userMessageText = `Submitted URL: ${url}`;
      payload = { type: "url", url };
    } else {
      userMessageText = textInput;
      payload = { type: "text", text: textInput };
    }

    const userMessage: Message = {
      user: "me",
      text: userMessageText,
    };

    setMessages((prev) => [...prev, userMessage]);

    setIsModalOpen(false);
    setModalSubmitting(true);
    setStatusText("Uploading document…");
    const controller = new AbortController();
    uploadRef.current = controller;

    try {
      const username = localStorage.getItem("username") || "";
      const settings = {
        // Only the embedding model matters for indexing, but sending the whole
        // config keeps one shape on the wire and the backend ignores the rest.
        config: configRef.current,
        keys: keysRef.current(),
      };

      let response: UploadResponse;

      switch (payload.type) {
        case "file":
          response = await service.submitFile(payload.file, username, settings);
          break;

        case "url":
          response = await service.submitURL(payload.url, username, settings);
          break;

        case "text":
          response = await service.submitText(payload.text, username, settings);
          break;

        default:
          throw new Error("Invalid payload type");
      }

      if (response.status !== 200) throw new Error(response.message);
      if (controller.signal.aborted) return;
      documentsChanged();
      await service.waitForDocument(response.data, username, (document) => {
        setStatusText(document.status === "pending" ? "Waiting for indexing to start…" : "Indexing your document…");
      }, controller.signal);
      // An upload makes the intent to chat with the user's documents explicit.
      setConfig({ corpus: "user" });
      documentsChanged();
      resetModalInputs();
      setMessages((prev) => [...prev, {
        user: "bot",
        text: `“${response.data.name}” is ready. Ask a question about your document.`,
      }]);
    } catch (error) {
      if (controller.signal.aborted) return;
      documentsChanged();
      setMessages((prev) => [...prev, { user: "bot", text: errorMessage(error) }]);
    } finally {
      if (!controller.signal.aborted) {
        setModalSubmitting(false);
        setStatusText("");
      }
    }
  };

  const isFileFilled = !!fileLocal;
  const isUrlFilled = url.length > 0;
  const isTextFilled = textInput.length > 0;

  return (
    <div className="flex flex-col h-[calc(100vh-64px)] bg-[hsl(var(--background))] text-[hsl(var(--foreground))]">
      {/* Chat messages area */}
      <div className="flex-grow overflow-y-auto p-4">
        {messages.map((msg, index) => (
          <ChatMessage
            key={index}
            user={msg.user}
            text={msg.text}
            context={msg.context}
            evaluation={msg.evaluation}
            notices={msg.notices}
          />
        ))}
        {(chatLoading || modalSubmitting) && (
          <div role="status" aria-live="polite" className="flex items-center gap-2 text-[hsl(var(--muted-foreground))] text-sm px-4 pb-2">
            <span className="animate-pulse">●</span>
            <span>{statusText || "Thinking..."}</span>
          </div>
        )}
      </div>

      {/* Input area with small button + text input + send button */}
      <div className="flex-shrink-0 flex items-center p-4 bg-[hsl(var(--card))] border-t border-[hsl(var(--input))]">
        {/* Small button to open modal */}
        <button
          onClick={() => setIsModalOpen(true)}
          disabled={chatLoading || modalSubmitting}
          className="mr-2 px-3 py-2 rounded bg-[hsl(var(--secondary))] text-[hsl(var(--secondary-foreground))] hover:bg-[hsl(var(--secondary)/0.8)] transition-colors duration-200 disabled:opacity-50"
          aria-label="Upload file, URL, or text"
          title="Attach content"
        >
          📎
        </button>

        <input
          type="text"
          className="flex-grow border border-[hsl(var(--input))] rounded px-4 py-2 mr-4 bg-[hsl(var(--background))] text-[hsl(var(--foreground))]"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && !chatLoading && !modalSubmitting && sendMessage()}
          placeholder={modalSubmitting ? "Wait for your document to be ready…" : "Ask about your documents"}
          disabled={chatLoading || modalSubmitting}
        />
        <button
          className={`px-6 py-2 rounded font-semibold transition-colors duration-200 ${
            chatLoading
              ? "bg-[hsl(var(--muted))] text-[hsl(var(--muted-foreground))]"
              : "bg-[hsl(var(--primary))] text-[hsl(var(--primary-foreground))]"
          }`}
          onClick={sendMessage}
          disabled={chatLoading || modalSubmitting}
        >
          {modalSubmitting ? "Indexing…" : chatLoading ? "Sending..." : "Send"}
        </button>
      </div>

      {/* Modal overlay */}
      {isModalOpen && (
        <div className="fixed inset-0 bg-[hsl(var(--foreground))]/30 flex items-center justify-center z-50">
          <div className="bg-[hsl(var(--card))] rounded shadow-xl w-full max-w-md p-6 relative">
            <h2 className="text-xl font-semibold mb-4">Attach Content</h2>
            <div className="space-y-4">
              <FileUploadSection
                inputRef={fileInputRef}
                onChange={handleFileChange}
                disabled={isUrlFilled || isTextFilled || modalSubmitting}
                fileName={fileLocal?.name}
                onClear={() => {
                  setFileLocal(null);
                  if (fileInputRef.current) fileInputRef.current.value = "";
                }}
              />
              <UrlUploadSection
                value={url}
                onChange={handleUrlChange}
                disabled={isFileFilled || isTextFilled || modalSubmitting}
              />
              <TextUploadSection
                value={textInput}
                onChange={handleTextChange}
                disabled={isFileFilled || isUrlFilled || modalSubmitting}
              />
            </div>
            <div className="flex justify-end gap-2 mt-6">
              <button
                onClick={() => {
                  setIsModalOpen(false);
                  resetModalInputs();
                }}
                disabled={modalSubmitting}
                className="px-4 py-2 rounded bg-[hsl(var(--muted))] text-[hsl(var(--muted-foreground))]"
              >
                Cancel
              </button>
              <button
                onClick={handleModalSubmit}
                disabled={modalSubmitting || (!fileLocal && !url.trim() && !textInput.trim())}
                className="px-4 py-2 rounded bg-[hsl(var(--primary))] text-[hsl(var(--primary-foreground))] font-semibold"
              >
                {modalSubmitting ? "Processing..." : "Submit"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default Chatbot;
