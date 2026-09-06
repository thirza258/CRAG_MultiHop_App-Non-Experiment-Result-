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
  /**
   * Settings the pipeline could not honour verbatim for this answer — chiefly
   * an embedding model overridden by the one the searched corpus was built
   * with. Shown with the answer, not swallowed: a setting that silently did
   * nothing is what makes a settings panel untrustworthy.
   */
  notices?: string[];
};

export const ChatMessage: React.FC<ChatMessageProps> = ({
  user,
  text,
  context,
  evaluation,
  notices,
}) => {
  const isMe = user === "me";
  const [showContext, setShowContext] = useState(false);

  return (
    <div className={`flex w-full mb-5 ${isMe ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-xl rounded px-4 py-3 flex flex-col space-y-3 text-sm leading-relaxed
          ${
            isMe
              ? "bg-[hsl(var(--primary))] text-[hsl(var(--primary-foreground))]"
              : "border border-[hsl(var(--border))] bg-[hsl(var(--background))] text-[hsl(var(--foreground))]"
          }
        `}
      >
        <div className="[&_p]:m-0 [&_p+p]:mt-3 [&_ul]:my-2 [&_ul]:list-disc [&_ul]:pl-5 [&_ol]:my-2 [&_ol]:list-decimal [&_ol]:pl-5 [&_code]:rounded [&_code]:bg-[hsl(var(--muted))] [&_code]:px-1 [&_code]:py-0.5 [&_code]:text-[0.85em]">
          <ReactMarkdown>{text}</ReactMarkdown>
        </div>

        {!isMe && notices && notices.length > 0 && (
          <ul className="space-y-1 border-l-2 border-[hsl(var(--primary))] bg-[hsl(var(--muted))] px-3 py-2 text-xs text-[hsl(var(--muted-foreground))]">
            {notices.map((notice, idx) => (
              <li key={idx}>{notice}</li>
            ))}
          </ul>
        )}

        {/* Context dropdown (only for bot messages with context) */}
        {!isMe && context && context.length > 0 && (
          <div className="border-t border-[hsl(var(--border))] pt-2">
            <button
              onClick={() => setShowContext(!showContext)}
              className="flex items-center gap-1 text-xs text-[hsl(var(--primary))] hover:underline"
              aria-expanded={showContext}
            >
              {showContext ? "▼" : "▶"} Retrieved context ({context.length} chunk
              {context.length !== 1 && "s"})
            </button>
            {showContext && (
              <div className="mt-2 max-h-60 space-y-3 overflow-y-auto text-xs">
                {context.map((ctx, idx) => (
                  <div key={idx} className="border-l-2 border-[hsl(var(--border))] pl-3">
                    <div className="font-mono text-[11px] text-[hsl(var(--muted-foreground))]">
                      chunk {ctx.metadata.chunk_index} · {ctx.metadata.username}
                    </div>
                    <p className="mt-1 line-clamp-3 text-[hsl(var(--muted-foreground))]">
                      {ctx.text.length > 200
                        ? ctx.text.substring(0, 200) + "..."
                        : ctx.text}
                    </p>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {evaluation && (
          <dl className="flex gap-6 border-t border-[hsl(var(--border))] pt-2 text-xs text-[hsl(var(--muted-foreground))]">
            <div className="flex gap-1.5">
              <dt>Answer relevancy</dt>
              <dd className="font-medium text-[hsl(var(--foreground))]" data-numeric>
                {evaluation.answer_relevancy?.toFixed(2) ?? "n/a"}
              </dd>
            </div>
            <div className="flex gap-1.5">
              <dt>Faithfulness</dt>
              <dd className="font-medium text-[hsl(var(--foreground))]" data-numeric>
                {evaluation.faithfulness?.toFixed(2) ?? "n/a"}
              </dd>
            </div>
          </dl>
        )}
      </div>
    </div>
  );
};
