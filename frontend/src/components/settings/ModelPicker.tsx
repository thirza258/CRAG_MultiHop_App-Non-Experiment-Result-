import React, { useEffect, useMemo, useRef, useState } from "react";
import { Check, ChevronDown, Lock, Search, X } from "lucide-react";
import { ModelOption } from "../../types/types";
import { isValidModelId } from "../../context/PipelineConfigContext";
import { formatContext, formatPrice } from "../../services/modelCatalog";

/**
 * Searchable model chooser.
 *
 * A native <select> is not an option here: OpenRouter lists several hundred
 * chat models, so the list has to be filterable. It also accepts a typed id
 * that is not in the catalog — a model released after the cached list, or a
 * deployment-specific one — validated against the same pattern the backend
 * uses, so the UI cannot offer something the server would silently drop.
 *
 * An empty value is a real choice, rendered as "Server default": it means "let
 * the backend use what it has configured", which is what every request sent
 * before model selection existed.
 */

type ModelPickerProps = {
  label: string;
  hint?: string;
  value: string;
  options: ModelOption[];
  onChange: (modelId: string) => void;
  /** Shown instead of the control when the choice is not the user's to make. */
  locked?: boolean;
  lockedReason?: string;
  /** What "" resolves to on the server, shown so the default is not a mystery. */
  defaultLabel?: string;
  disabled?: boolean;
  /** Rendered under the control when the catalog could not be fetched live. */
  catalogNote?: string;
  /**
   * A stored choice that the lock is currently overriding. Surfaced with a way
   * to drop it — otherwise the user gets told about the override on every
   * single query and has no idea where the setting lives.
   */
  conflictingChoice?: string;
  onClearChoice?: () => void;
};

const SERVER_DEFAULT_LABEL = "Server default";

/** How many rows to render at once — the filter is what narrows the list. */
const MAX_VISIBLE = 60;

const ModelPicker: React.FC<ModelPickerProps> = ({
  label,
  hint,
  value,
  options,
  onChange,
  locked = false,
  lockedReason,
  defaultLabel,
  disabled = false,
  catalogNote,
  conflictingChoice,
  onClearChoice,
}) => {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const containerRef = useRef<HTMLDivElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);

  // Close on an outside click or Escape, the two things a user expects of a
  // popover they opened by accident.
  useEffect(() => {
    if (!open) return;

    const onPointerDown = (event: MouseEvent) => {
      if (!containerRef.current?.contains(event.target as Node)) setOpen(false);
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };

    document.addEventListener("mousedown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("mousedown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [open]);

  useEffect(() => {
    if (open) searchRef.current?.focus();
    else setQuery("");
  }, [open]);

  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return options.slice(0, MAX_VISIBLE);
    return options
      .filter(
        (option) =>
          option.id.toLowerCase().includes(needle) ||
          option.name.toLowerCase().includes(needle)
      )
      .slice(0, MAX_VISIBLE);
  }, [options, query]);

  const totalMatches = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return options.length;
    return options.filter(
      (option) =>
        option.id.toLowerCase().includes(needle) ||
        option.name.toLowerCase().includes(needle)
    ).length;
  }, [options, query]);

  const selected = options.find((option) => option.id === value);
  const trimmedQuery = query.trim();
  // Offer the typed text as a choice only when it is a plausible id and is not
  // already in the list, so the common case shows no extra row.
  const customIsOfferable =
    trimmedQuery.length > 0 &&
    isValidModelId(trimmedQuery) &&
    !options.some((option) => option.id === trimmedQuery);

  const select = (modelId: string) => {
    onChange(modelId);
    setOpen(false);
  };

  if (locked) {
    return (
      <div>
        <div className="text-[11px] font-semibold uppercase tracking-wider text-[hsl(var(--muted-foreground))]">
          {label}
        </div>
        <div className="mt-1.5 flex items-center gap-2 rounded-md border border-dashed border-[hsl(var(--border))] bg-[hsl(var(--muted))] px-2 py-1.5">
          <Lock
            className="h-3 w-3 shrink-0 text-[hsl(var(--muted-foreground))]"
            aria-hidden="true"
          />
          <span
            className="min-w-0 flex-1 truncate font-mono text-xs text-[hsl(var(--foreground))]"
            title={value || defaultLabel || SERVER_DEFAULT_LABEL}
          >
            {value || defaultLabel || SERVER_DEFAULT_LABEL}
          </span>
        </div>
        {lockedReason && (
          <p className="mt-1 text-[11px] leading-snug text-[hsl(var(--muted-foreground))]">
            {lockedReason}
          </p>
        )}
        {conflictingChoice && (
          <p className="mt-1 flex flex-wrap items-baseline gap-1 text-[11px] leading-snug text-[hsl(var(--muted-foreground))]">
            <span>
              You have <span className="font-mono">{conflictingChoice}</span> saved,
              which is being overridden on every query.
            </span>
            {onClearChoice && (
              <button
                type="button"
                onClick={onClearChoice}
                className="underline decoration-dotted hover:text-[hsl(var(--primary))]"
              >
                Drop it
              </button>
            )}
          </p>
        )}
      </div>
    );
  }

  return (
    <div ref={containerRef} className="relative">
      <div className="flex items-baseline justify-between gap-2">
        <div
          className="text-[11px] font-semibold uppercase tracking-wider text-[hsl(var(--muted-foreground))]"
          id={`model-picker-${label}`}
        >
          {label}
        </div>
        {value && (
          <button
            type="button"
            onClick={() => onChange("")}
            className="flex items-center gap-0.5 text-[10px] text-[hsl(var(--muted-foreground))] transition-colors hover:text-[hsl(var(--primary))]"
            title="Use the server's configured model"
          >
            <X className="h-2.5 w-2.5" aria-hidden="true" />
            Use default
          </button>
        )}
      </div>

      <button
        type="button"
        onClick={() => setOpen((prev) => !prev)}
        disabled={disabled}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-labelledby={`model-picker-${label}`}
        className="mt-1.5 flex w-full items-center gap-2 rounded-md border border-[hsl(var(--border))] bg-[hsl(var(--background))] px-2 py-1.5 text-left transition-colors hover:border-[hsl(var(--primary))] disabled:cursor-not-allowed disabled:opacity-50"
      >
        <span className="min-w-0 flex-1">
          <span
            className={`block truncate text-xs ${
              value
                ? "font-mono text-[hsl(var(--foreground))]"
                : "text-[hsl(var(--muted-foreground))]"
            }`}
            title={value || undefined}
          >
            {value || SERVER_DEFAULT_LABEL}
          </span>
          {!value && defaultLabel && (
            <span className="block truncate font-mono text-[10px] text-[hsl(var(--muted-foreground))]">
              {defaultLabel}
            </span>
          )}
          {selected && selected.name !== selected.id && (
            <span className="block truncate text-[10px] text-[hsl(var(--muted-foreground))]">
              {selected.name}
            </span>
          )}
        </span>
        <ChevronDown
          className={`h-3.5 w-3.5 shrink-0 text-[hsl(var(--muted-foreground))] transition-transform ${
            open ? "rotate-180" : ""
          }`}
          aria-hidden="true"
        />
      </button>

      {hint && !open && (
        <p className="mt-1 text-[11px] leading-snug text-[hsl(var(--muted-foreground))]">
          {hint}
        </p>
      )}
      {catalogNote && !open && (
        <p className="mt-1 text-[11px] leading-snug text-[hsl(var(--muted-foreground))]">
          {catalogNote}
        </p>
      )}

      {open && (
        <div className="absolute left-0 right-0 z-30 mt-1 overflow-hidden rounded-md border border-[hsl(var(--border))] bg-[hsl(var(--card))] shadow-lg">
          <div className="flex items-center gap-1.5 border-b border-[hsl(var(--border))] px-2 py-1.5">
            <Search
              className="h-3 w-3 shrink-0 text-[hsl(var(--muted-foreground))]"
              aria-hidden="true"
            />
            <input
              ref={searchRef}
              type="text"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search models, or type an id"
              className="w-full bg-transparent text-xs text-[hsl(var(--foreground))] outline-none placeholder:text-[hsl(var(--muted-foreground))]"
            />
          </div>

          <ul role="listbox" className="max-h-56 overflow-y-auto py-1">
            <li>
              <button
                type="button"
                role="option"
                aria-selected={value === ""}
                onClick={() => select("")}
                className="flex w-full items-center gap-2 px-2 py-1.5 text-left hover:bg-[hsl(var(--muted))]"
              >
                <Check
                  className={`h-3 w-3 shrink-0 text-[hsl(var(--primary))] ${
                    value === "" ? "" : "invisible"
                  }`}
                  aria-hidden="true"
                />
                <span className="min-w-0 flex-1">
                  <span className="block text-xs text-[hsl(var(--foreground))]">
                    {SERVER_DEFAULT_LABEL}
                  </span>
                  {defaultLabel && (
                    <span className="block truncate font-mono text-[10px] text-[hsl(var(--muted-foreground))]">
                      {defaultLabel}
                    </span>
                  )}
                </span>
              </button>
            </li>

            {customIsOfferable && (
              <li>
                <button
                  type="button"
                  role="option"
                  aria-selected={false}
                  onClick={() => select(trimmedQuery)}
                  className="flex w-full items-center gap-2 px-2 py-1.5 text-left hover:bg-[hsl(var(--muted))]"
                >
                  <Check className="invisible h-3 w-3 shrink-0" aria-hidden="true" />
                  <span className="min-w-0 flex-1">
                    <span className="block truncate font-mono text-xs text-[hsl(var(--foreground))]">
                      {trimmedQuery}
                    </span>
                    <span className="block text-[10px] text-[hsl(var(--muted-foreground))]">
                      Use this id (not in the catalog)
                    </span>
                  </span>
                </button>
              </li>
            )}

            {filtered.map((option) => {
              const price = formatPrice(option.prompt_price);
              const context = formatContext(option.context_length);
              return (
                <li key={option.id}>
                  <button
                    type="button"
                    role="option"
                    aria-selected={option.id === value}
                    onClick={() => select(option.id)}
                    className="flex w-full items-center gap-2 px-2 py-1.5 text-left hover:bg-[hsl(var(--muted))]"
                  >
                    <Check
                      className={`h-3 w-3 shrink-0 text-[hsl(var(--primary))] ${
                        option.id === value ? "" : "invisible"
                      }`}
                      aria-hidden="true"
                    />
                    <span className="min-w-0 flex-1">
                      <span
                        className="block truncate font-mono text-xs text-[hsl(var(--foreground))]"
                        title={option.id}
                      >
                        {option.id}
                      </span>
                      <span className="flex flex-wrap items-center gap-x-1.5 text-[10px] text-[hsl(var(--muted-foreground))]">
                        {context && <span>{context}</span>}
                        {price && <span>{price}</span>}
                        {option.free && (
                          <span className="rounded bg-[hsl(var(--primary))]/10 px-1 text-[hsl(var(--primary))]">
                            free
                          </span>
                        )}
                      </span>
                    </span>
                  </button>
                </li>
              );
            })}

            {!filtered.length && !customIsOfferable && (
              <li className="px-2 py-2 text-[11px] leading-snug text-[hsl(var(--muted-foreground))]">
                {options.length
                  ? "No model matches that search."
                  : "The model catalog is unavailable — type a full model id to use it."}
              </li>
            )}
          </ul>

          {totalMatches > filtered.length && (
            <div className="border-t border-[hsl(var(--border))] px-2 py-1 text-[10px] text-[hsl(var(--muted-foreground))]">
              Showing {filtered.length} of {totalMatches} — keep typing to narrow it down.
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ModelPicker;
