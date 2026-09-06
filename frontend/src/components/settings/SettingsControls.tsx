import React, { useState } from "react";
import { ChevronDown, RotateCcw } from "lucide-react";

/**
 * The controls the pipeline settings panel is built from.
 *
 * Extracted because the panel now carries roughly twenty-five settings, and the
 * interesting part of each is its label and hint, not its markup.
 *
 * The one non-obvious control is {@link NumberField}: several settings have a
 * third state beyond their range. `null` means "whatever the deployment
 * configured", which is not the same as any particular number — the server
 * tunes these per deployment and the client must be able to say "don't
 * override" rather than pinning a value of its own.
 */

// ── Collapsible group ────────────────────────────────────────────────────────

type SectionProps = {
  title: string;
  /** Short summary shown on the header row, e.g. how many are switched off. */
  badge?: string;
  defaultOpen?: boolean;
  children: React.ReactNode;
};

export const Section: React.FC<SectionProps> = ({
  title,
  badge,
  defaultOpen = false,
  children,
}) => {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div className="border-t border-[hsl(var(--border))] pt-3 first:border-t-0 first:pt-0">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        className="flex w-full items-center gap-1.5 text-left"
      >
        <ChevronDown
          className={`h-3 w-3 shrink-0 text-[hsl(var(--muted-foreground))] transition-transform ${
            open ? "" : "-rotate-90"
          }`}
          aria-hidden="true"
        />
        <span className="text-[11px] font-semibold uppercase tracking-wider text-[hsl(var(--muted-foreground))]">
          {title}
        </span>
        {badge && (
          <span className="ml-auto rounded bg-[hsl(var(--muted))] px-1.5 py-0.5 text-[10px] font-medium text-[hsl(var(--muted-foreground))]">
            {badge}
          </span>
        )}
      </button>
      {open && <div className="mt-3 space-y-3.5 pl-[18px]">{children}</div>}
    </div>
  );
};

// ── Segmented choice ─────────────────────────────────────────────────────────

export type SegmentedOption<T extends string> = {
  value: T;
  label: string;
  hint: string;
};

type SegmentedProps<T extends string> = {
  label: string;
  value: T;
  options: SegmentedOption<T>[];
  onChange: (value: T) => void;
  disabled?: boolean;
};

export function Segmented<T extends string>({
  label,
  value,
  options,
  onChange,
  disabled = false,
}: SegmentedProps<T>) {
  const active = options.find((o) => o.value === value);
  // Four or more choices do not fit a sidebar row at a readable size, so they
  // wrap into two columns instead of shrinking to unreadable slivers.
  const layout =
    options.length > 3 ? "grid grid-cols-2 gap-0.5" : "flex";
  return (
    <div className={disabled ? "opacity-50" : undefined}>
      <div
        className="text-[11px] font-semibold uppercase tracking-wider text-[hsl(var(--muted-foreground))]"
        id={`seg-${label}`}
      >
        {label}
      </div>
      <div
        role="radiogroup"
        aria-labelledby={`seg-${label}`}
        className={`mt-1.5 ${layout} rounded-md border border-[hsl(var(--border))] bg-[hsl(var(--background))] p-0.5`}
      >
        {options.map((option) => (
          <button
            key={option.value}
            type="button"
            role="radio"
            aria-checked={value === option.value}
            title={option.hint}
            disabled={disabled}
            onClick={() => onChange(option.value)}
            className={`flex-1 rounded px-2 py-1 text-xs font-medium transition-colors disabled:cursor-not-allowed ${
              value === option.value
                ? "bg-[hsl(var(--primary))] text-[hsl(var(--primary-foreground))]"
                : "text-[hsl(var(--muted-foreground))] hover:text-[hsl(var(--foreground))]"
            }`}
          >
            {option.label}
          </button>
        ))}
      </div>
      {active && (
        <p className="mt-1 text-[11px] leading-snug text-[hsl(var(--muted-foreground))]">
          {active.hint}
        </p>
      )}
    </div>
  );
}

// ── Toggle ───────────────────────────────────────────────────────────────────

type ToggleProps = {
  label: string;
  hint: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
  /** Why it is disabled — shown instead of the hint, so it is never a mystery. */
  disabledReason?: string;
};

export const Toggle: React.FC<ToggleProps> = ({
  label,
  hint,
  checked,
  onChange,
  disabled = false,
  disabledReason,
}) => (
  <label
    className={`flex items-start gap-2.5 ${
      disabled ? "cursor-not-allowed opacity-50" : "cursor-pointer"
    }`}
  >
    <input
      type="checkbox"
      checked={checked}
      disabled={disabled}
      onChange={(e) => onChange(e.target.checked)}
      className="mt-0.5 h-4 w-4 shrink-0 cursor-pointer accent-[hsl(var(--primary))] disabled:cursor-not-allowed"
    />
    <span className="min-w-0">
      <span className="block text-xs font-medium text-[hsl(var(--foreground))]">
        {label}
      </span>
      <span className="block text-[11px] leading-snug text-[hsl(var(--muted-foreground))]">
        {disabled && disabledReason ? disabledReason : hint}
      </span>
    </span>
  </label>
);

// ── Number with a "server default" state ─────────────────────────────────────

type NumberFieldProps = {
  label: string;
  hint: string;
  /** null means "defer to whatever the deployment configured". */
  value: number | null;
  min: number;
  max: number;
  step?: number;
  /**
   * Slider position while the value is null. Display only — it is never sent,
   * so it cannot override the server. It mirrors what `rag/rag_service.py`
   * ships today; a deployment that retuned that will show the handle in a
   * slightly wrong place until the user moves it, which is the intended
   * trade against shipping a second source of truth to the client.
   */
  fallback: number;
  onChange: (value: number | null) => void;
  disabled?: boolean;
  /** Rendered next to the value, e.g. "chunks". */
  unit?: string;
};

export const NumberField: React.FC<NumberFieldProps> = ({
  label,
  hint,
  value,
  min,
  max,
  step = 1,
  fallback,
  onChange,
  disabled = false,
  unit,
}) => {
  const isDefault = value === null;
  const shown = isDefault ? fallback : value;
  const id = `num-${label.replace(/\s+/g, "-").toLowerCase()}`;

  return (
    <div className={disabled ? "opacity-50" : undefined}>
      <div className="flex items-center justify-between gap-2">
        <label
          htmlFor={id}
          className="text-[11px] font-semibold uppercase tracking-wider text-[hsl(var(--muted-foreground))]"
        >
          {label}
        </label>
        {isDefault ? (
          <span id={`${id}-state`} className="text-[10px] text-[hsl(var(--muted-foreground))]">
            Server default
          </span>
        ) : (
          <span id={`${id}-state`} className="flex items-center gap-1.5">
            <span className="font-mono text-xs text-[hsl(var(--primary))]" data-numeric>
              {shown}
              {unit ? ` ${unit}` : ""}
            </span>
            <button
              type="button"
              onClick={() => onChange(null)}
              disabled={disabled}
              title="Go back to the server's configured value"
              className="text-[hsl(var(--muted-foreground))] transition-colors hover:text-[hsl(var(--primary))]"
            >
              <RotateCcw className="h-2.5 w-2.5" aria-hidden="true" />
            </button>
          </span>
        )}
      </div>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={shown}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
        className={`mt-1 w-full disabled:cursor-not-allowed ${
          // A deferred field has no value, so the track must not look like it
          // holds one. Grey rather than accent: with `fallback` sitting on the
          // maximum for some fields (max hops), a filled blue track reads as a
          // ceiling the user chose, which is the opposite of "not set".
          isDefault
            ? "accent-[hsl(var(--muted-foreground))] opacity-50"
            : "accent-[hsl(var(--primary))]"
        }`}
        aria-describedby={`${id}-state`}
      />
      <p className="mt-0.5 text-[11px] leading-snug text-[hsl(var(--muted-foreground))]">
        {hint}
      </p>
    </div>
  );
};
