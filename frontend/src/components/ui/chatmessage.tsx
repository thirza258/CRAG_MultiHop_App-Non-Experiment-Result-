import React, { useState } from "react";
import ReactMarkdown from "react-markdown";
import { ChatContextResponse } from "../../interface"; // adjust import path

type ChatMessageProps = {
  user: "me" | "bot";
  text: string;
  context?: ChatContextResponse[];
  evaluation?: {
    answer_relevancy: number | null;
    faithfulness: number | null;
  };
};

export const ChatMessage: React.FC<ChatMessageProps> = ({
  user,
  text,
  context,
  evaluation,
}) => {
  const isMe = user === "me";
  const [showContext, setShowContext] = useState(false);

  return (
    <div className={`flex w-full mb-4 ${isMe ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-md p-4 rounded-2xl shadow-lg flex flex-col space-y-3
          ${
            isMe
              ? "bg-[hsl(var(--primary))] text-[hsl(var(--primary-foreground))] rounded-br-md"
              : "bg-slate-800 text-slate-100 rounded-bl-md"
          }
        `}
      >
        <ReactMarkdown>{text}</ReactMarkdown>

        {/* Context dropdown (only for bot messages with context) */}
        {!isMe && context && context.length > 0 && (
          <div className="mt-2 border-t border-slate-600 pt-2">
            <button
              onClick={() => setShowContext(!showContext)}
              className="text-xs text-blue-300 hover:text-blue-200 flex items-center gap-1"
            >
              {showContext ? "▼" : "▶"} Retrieved Context (
              {context.length} chunk{context.length !== 1 && "s"})
            </button>
            {showContext && (
              <div className="mt-2 space-y-2 max-h-60 overflow-y-auto text-xs bg-slate-900/50 p-2 rounded">
                {context.map((ctx, idx) => (
                  <div key={idx} className="border-l-2 border-blue-500 pl-2">
                    <div className="text-gray-300">
                      | chunk{" "}
                      {ctx.metadata.chunk_index} | 👤 {ctx.metadata.username}
                    </div>
                    <div className="text-gray-400 mt-1 line-clamp-3">
                      {ctx.text.length > 200
                        ? ctx.text.substring(0, 200) + "..."
                        : ctx.text}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {evaluation && (
          <div className="mt-2 text-xs text-gray-400">
            <div>Answer Relevancy: {evaluation.answer_relevancy?.toFixed(2) || "N/A"}</div>
            <div>Faithfulness: {evaluation.faithfulness?.toFixed(2) || "N/A"}</div>
          </div>
        )}
      </div>
    </div>
  );
};