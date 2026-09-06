import React, { useState } from "react";
import { ChevronDown, Eye, EyeOff, KeyRound, Trash2 } from "lucide-react";
import { ApiKeyName } from "../../types/types";
import {
  ApiKeyProblem,
  MAX_KEY_LENGTH,
  useApiKeys,
} from "../../context/ApiKeysContext";

/**
 * Where the user pastes their own API keys.
 *
 * Keys are held in this browser and sent with each request, for that request
 * only — the server never stores them and redacts them from its logs. Anything
 * left blank falls back to the server's own credentials, so this panel is
 * entirely optional.
 */

type KeyFieldSpec = {
  name: ApiKeyName;
  label: string;
  placeholder: string;
  /** What stops working without it, stated concretely. */
  purpose: string;
  docsUrl: string;
  docsLabel: string;
};

const KEY_FIELDS: KeyFieldSpec[] = [
  {
    name: "openrouter",
    label: "OpenRouter",
    placeholder: "sk-or-v1-…",
    purpose:
      "Runs every model call — answering, hop decisions and embeddings. With your own key you pick any model OpenRouter offers and pay for it yourself.",
    docsUrl: "https://openrouter.ai/keys",
    docsLabel: "openrouter.ai/keys",
  },
  {
    name: "news",
    label: "NewsAPI",
    placeholder: "32-character key",
    purpose:
      "Used by corrective retrieval to pull a recent news article when the indexed documents look too weak to answer from. Without it that one lookup is skipped; Wikipedia still runs.",
    docsUrl: "https://newsapi.org/register",
    docsLabel: "newsapi.org/register",
  },
];

const PROBLEM_TEXT: Record<ApiKeyProblem, string> = {
  "invalid-characters":
    "That does not look like a key — it contains spaces or line breaks. Paste it again without them.",
  "too-long": `That is longer than ${MAX_KEY_LENGTH} characters, so it is probably not a key.`,
};

const ApiKeysPanel: React.FC = () => {
  const { keys, setKey, clearKeys, hasAnyKey, problems } = useApiKeys();
  const [open, setOpen] = useState(false);
  const [revealed, setRevealed] = useState<Partial<Record<ApiKeyName, boolean>>>({});
  // Held separately from the stored key so a half-pasted value can be shown
  // back to the user while it is still invalid, instead of vanishing.
  const [drafts, setDrafts] = useState<Partial<Record<ApiKeyName, string>>>({});

  const valueFor = (name: ApiKeyName) => drafts[name] ?? keys[name];

  const onFieldChange = (name: ApiKeyName, next: string) => {
    setDrafts((prev) => ({ ...prev, [name]: next }));
    setKey({ [name]: next });
  };

  const onClear = () => {
    clearKeys();
    setDrafts({});
    setRevealed({});
  };

  return (
    <section className="flex-shrink-0 rounded border border-[hsl(var(--border))] bg-[hsl(var(--background))]">
      <div className="flex items-center justify-between px-3 py-2">
        <button
          type="button"
          onClick={() => setOpen(!open)}
          aria-expanded={open}
          className="flex min-w-0 items-center gap-2 text-sm font-semibold text-[hsl(var(--foreground))]"
        >
          <KeyRound
            className="h-3.5 w-3.5 text-[hsl(var(--primary))]"
            aria-hidden="true"
          />
          API keys
          {hasAnyKey && (
            <span className="rounded bg-[hsl(var(--primary))]/10 px-1.5 py-0.5 text-[10px] font-medium text-[hsl(var(--primary))]">
              yours
            </span>
          )}
          <ChevronDown
            className={`h-3.5 w-3.5 text-[hsl(var(--muted-foreground))] transition-transform ${
              open ? "" : "-rotate-90"
            }`}
            aria-hidden="true"
          />
        </button>
        {hasAnyKey && (
          <button
            type="button"
            onClick={onClear}
            title="Remove the keys stored in this browser"
            className="flex items-center gap-1 text-[11px] text-[hsl(var(--muted-foreground))] transition-colors hover:text-[hsl(var(--destructive))]"
          >
            <Trash2 className="h-3 w-3" aria-hidden="true" />
            Clear
          </button>
        )}
      </div>

      {open && (
        <div className="space-y-4 border-t border-[hsl(var(--border))] px-3 py-3">
          {KEY_FIELDS.map((field) => {
            const problem = problems[field.name];
            const isRevealed = Boolean(revealed[field.name]);
            const inputId = `api-key-${field.name}`;
            return (
              <div key={field.name}>
                <label
                  htmlFor={inputId}
                  className="text-[11px] font-semibold uppercase tracking-wider text-[hsl(var(--muted-foreground))]"
                >
                  {field.label}
                </label>
                <div
                  className={`mt-1.5 flex items-center gap-1 rounded-md border bg-[hsl(var(--background))] px-2 py-1 ${
                    problem
                      ? "border-[hsl(var(--destructive))]"
                      : "border-[hsl(var(--border))] focus-within:border-[hsl(var(--primary))]"
                  }`}
                >
                  <input
                    id={inputId}
                    type={isRevealed ? "text" : "password"}
                    value={valueFor(field.name)}
                    onChange={(event) => onFieldChange(field.name, event.target.value)}
                    placeholder={field.placeholder}
                    autoComplete="off"
                    spellCheck={false}
                    className="w-full bg-transparent py-0.5 font-mono text-xs text-[hsl(var(--foreground))] outline-none placeholder:font-sans placeholder:text-[hsl(var(--muted-foreground))]"
                  />
                  <button
                    type="button"
                    onClick={() =>
                      setRevealed((prev) => ({
                        ...prev,
                        [field.name]: !prev[field.name],
                      }))
                    }
                    aria-label={isRevealed ? "Hide key" : "Show key"}
                    title={isRevealed ? "Hide key" : "Show key"}
                    className="shrink-0 text-[hsl(var(--muted-foreground))] transition-colors hover:text-[hsl(var(--foreground))]"
                  >
                    {isRevealed ? (
                      <EyeOff className="h-3.5 w-3.5" aria-hidden="true" />
                    ) : (
                      <Eye className="h-3.5 w-3.5" aria-hidden="true" />
                    )}
                  </button>
                </div>

                {problem ? (
                  <p className="mt-1 text-[11px] leading-snug text-[hsl(var(--destructive))]">
                    {PROBLEM_TEXT[problem]}
                  </p>
                ) : (
                  <p className="mt-1 text-[11px] leading-snug text-[hsl(var(--muted-foreground))]">
                    {field.purpose}{" "}
                    <a
                      href={field.docsUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="underline decoration-dotted hover:text-[hsl(var(--primary))]"
                    >
                      {field.docsLabel}
                    </a>
                  </p>
                )}
              </div>
            );
          })}

          <p className="rounded border border-dashed border-[hsl(var(--border))] px-2 py-1.5 text-[11px] leading-snug text-[hsl(var(--muted-foreground))]">
            Keys are stored in this browser only and sent with each request for
            that request. They are never saved on the server, and the server
            redacts them from its logs. Leave a field empty to use the
            deployment's own key.
          </p>
        </div>
      )}
    </section>
  );
};

export default ApiKeysPanel;
