import { useState, useRef, useEffect } from "react";
import service from "../services/service";
import { ChatResponse } from "../interface";
import { ChatMessage } from "../components/ui/chatmessage";
import { Message, SubmitPayload } from "../types/types";
import FileUploadSection from "../components/file/FileInput";
import UrlUploadSection from "../components/file/URLInput";
import TextUploadSection from "../components/file/TextInput";
import { useNavigate } from "react-router-dom";
import {WS_BASE_URL} from "../services/websocket";

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

  const wsRef = useRef<WebSocket | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const navigate = useNavigate();

  useEffect(() => {
    const username = localStorage.getItem("username");
    if (!username) {
      navigate("/login");
    }
  }, []);

  const resetModalInputs = () => {
    setFileLocal(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
    setLocalUrl("");
    setTextInput("");
  };

  const sendMessage = (): void => {
  if (!input.trim() || chatLoading) return;

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

  const ws = new WebSocket(`${WS_BASE_URL}/ws/query/stream/`);
  wsRef.current = ws;

  ws.onopen = () => {
    ws.send(JSON.stringify({ USER: username, QUERY: input }));
  };

  ws.onmessage = (e) => {
  const msg = JSON.parse(e.data);
  switch (msg.stage) {
    case "result":
      sessionStorage.removeItem(`chat_history_${username}`);
      setMessages((prev) => [...prev, {
        user: "bot",
        text: msg.answer,
        conversationId: msg.conversation_id?.toString(),
        documentId: msg.document_id?.toString(),
        context: msg.context || [],
        evaluation: msg.evaluation || null,
      }]);
      console.log("Final answer received:", msg.answer);
      console.log("Evaluation received:", msg.evaluation);
      setStatusText("");
      setChatLoading(false);
      ws.close();
      break;

    case "error":
      setMessages((prev) => [...prev, {
        user: "bot",
        text: "Sorry, something went wrong.",
      }]);
      setStatusText("");
      setChatLoading(false);
      ws.close();
      break;

    default:
      setStatusText(`${msg.stage} — ${msg.message}`);
      break;
  }
};

  ws.onerror = () => {
    setMessages((prev) => [...prev, { user: "bot", text: "Connection lost." }]);
    setChatLoading(false);
    setStatusText("");
  };
};

// cleanup on unmount
useEffect(() => {
  return () => {
    wsRef.current?.close();
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
    if (!fileLocal && !url && !textInput) {
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
    resetModalInputs();
    setModalSubmitting(true);

    try {
      const username = localStorage.getItem("username") || "";

      let response: ChatResponse;

      switch (payload.type) {
        case "file":
          response = await service.submitFile(payload.file, username);
          break;

        case "url":
          response = await service.submitURL(payload.url, username);
          break;

        case "text":
          response = await service.submitText(payload.text, username);
          break;

        default:
          throw new Error("Invalid payload type");
      }

      if (response.status !== 200) {
        throw new Error(response.message);
      }

      sessionStorage.removeItem(`chat_history_${username}`);

      const botMessage: Message = {
        user: "bot",
        text: response.data.answer,
        conversationId: response.data.conversation_id?.toString(),
        context: response.data.context || [],
        evaluation: response.data.evaluation,
      };

      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      console.error("Error processing modal input:", error);

      setMessages((prev) => [
        ...prev,
        {
          user: "bot",
          text: "Sorry, something went wrong while processing your request.",
        },
      ]);
    } finally {
      setModalSubmitting(false);
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
          />
        ))}
        {chatLoading && (
          <div className="flex items-center gap-2 text-slate-400 text-sm px-4 pb-2">
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
          disabled={chatLoading}
          className="mr-2 px-3 py-2 rounded-lg bg-[hsl(var(--secondary))] text-[hsl(var(--secondary-foreground))] hover:bg-[hsl(var(--secondary)/0.8)] transition-colors duration-200 disabled:opacity-50"
          aria-label="Upload file, URL, or text"
          title="Attach content"
        >
          📎
        </button>

        <input
          type="text"
          className="flex-grow border border-[hsl(var(--input))] rounded-lg px-4 py-2 mr-4 bg-[hsl(var(--background))] text-[hsl(var(--foreground))]"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && !chatLoading && sendMessage()}
          placeholder="Type your message here"
          disabled={chatLoading}
        />
        <button
          className={`px-6 py-2 rounded-lg font-semibold transition-colors duration-200 ${
            chatLoading
              ? "bg-[hsl(var(--muted))] text-[hsl(var(--muted-foreground))]"
              : "bg-[hsl(var(--primary))] text-[hsl(var(--primary-foreground))]"
          }`}
          onClick={sendMessage}
          disabled={chatLoading}
        >
          {chatLoading ? "Sending..." : "Send"}
        </button>
      </div>

      {/* Modal overlay */}
      {isModalOpen && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-[hsl(var(--card))] rounded-lg shadow-xl w-full max-w-md p-6 relative">
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
                className="px-4 py-2 rounded-lg bg-[hsl(var(--muted))] text-[hsl(var(--muted-foreground))]"
              >
                Cancel
              </button>
              <button
                onClick={handleModalSubmit}
                disabled={modalSubmitting}
                className="px-4 py-2 rounded-lg bg-[hsl(var(--primary))] text-[hsl(var(--primary-foreground))] font-semibold"
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
