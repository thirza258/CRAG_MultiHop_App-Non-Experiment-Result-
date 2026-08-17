import React, { useState } from "react";
import { ChevronDown, RotateCcw, SlidersHorizontal } from "lucide-react";
import {
  MAX_HOPS,
  MIN_HOPS,
  usePipelineConfig,
} from "../context/PipelineConfigContext";
import { CorpusChoice, RetrieverChoice } from "../types/types";

/**
 * Sidebar panel for turning pipeline stages on and off, capping the hop count,
 * and choosing which corpus to search. The value is read by Chatbot and sent
 * with each query as CONFIG.
 */

const CORPUS_OPTIONS: { value: CorpusChoice; label: string; hint: string }[] = [
  { value: "auto", label: "Auto", hint: "Your documents when you have any, otherwise the base corpus" },
  { value: "user", label: "My docs", hint: "Only documents you uploaded" },
  { value: "base", label: "Base corpus", hint: "Only the bundled MultiHop-RAG news corpus" },
];

const RETRIEVER_OPTIONS: { value: RetrieverChoice; label: string; hint: string }[] = [
  { value: "both", label: "Both", hint: "Dense embeddings + BM25 keyword search" },
  { value: "dense", label: "Dense", hint: "Embedding similarity only" },
  { value: "sparse", label: "BM25", hint: "Keyword search only" },
];

type SegmentedProps<T extends string> = {
  label: string;
  value: T;
  options: { value: T; label: string; hint: string }[];
  onChange: (value: T) => void;
};

function Segmented<T extends string>({ label, value, options, onChange }: SegmentedProps<T>) {
  const active = options.find((o) => o.value === value);
  return (
    <div>
      <div
        className="text-[11px] font-semibold uppercase tracking-wider text-[hsl(var(--muted-foreground))]"
        id={`seg-${label}`}
      >
        {label}
      </div>
      <div
        role="radiogroup"
        aria-labelledby={`seg-${label}`}
        className="mt-1.5 flex rounded-md border border-[hsl(var(--border))] bg-[hsl(var(--background))] p-0.5"
      >
        {options.map((option) => (
          <button
            key={option.value}
            type="button"
            role="radio"
            aria-checked={value === option.value}
            title={option.hint}
            onClick={() => onChange(option.value)}
            className={`flex-1 rounded px-2 py-1 text-xs font-medium transition-colors ${
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

type ToggleProps = {
  label: string;
  hint: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
};

const Toggle: React.FC<ToggleProps> = ({ label, hint, checked, onChange }) => (
  <label className="flex cursor-pointer items-start gap-2.5">
    <input
      type="checkbox"
      checked={checked}
      onChange={(e) => onChange(e.target.checked)}
      className="mt-0.5 h-4 w-4 shrink-0 cursor-pointer accent-[hsl(var(--primary))]"
    />
    <span className="min-w-0">
      <span className="block text-xs font-medium text-[hsl(var(--foreground))]">{label}</span>
      <span className="block text-[11px] leading-snug text-[hsl(var(--muted-foreground))]">
        {hint}
      </span>
    </span>
  </label>
);

const PipelineConfigPanel: React.FC = () => {
  const { config, setConfig, resetConfig, isDefault } = usePipelineConfig();
  const [open, setOpen] = useState(true);

  return (
    <section className="flex-shrink-0 rounded-lg border border-[hsl(var(--border))] bg-[hsl(var(--background))]">
      <div className="flex items-center justify-between px-3 py-2">
        <button
          type="button"
          onClick={() => setOpen(!open)}
          aria-expanded={open}
          className="flex min-w-0 items-center gap-2 text-sm font-semibold text-[hsl(var(--foreground))]"
        >
          <SlidersHorizontal className="h-3.5 w-3.5 text-[hsl(var(--primary))]" aria-hidden="true" />
          Pipeline
          <ChevronDown
            className={`h-3.5 w-3.5 text-[hsl(var(--muted-foreground))] transition-transform ${
              open ? "" : "-rotate-90"
            }`}
            aria-hidden="true"
          />
        </button>
        {!isDefault && (
          <button
            type="button"
            onClick={resetConfig}
            title="Reset to the full default pipeline"
            className="flex items-center gap-1 text-[11px] text-[hsl(var(--muted-foreground))] transition-colors hover:text-[hsl(var(--primary))]"
          >
            <RotateCcw className="h-3 w-3" aria-hidden="true" />
            Reset
          </button>
        )}
      </div>

      {open && (
        <div className="space-y-4 border-t border-[hsl(var(--border))] px-3 py-3">
          <Segmented
            label="Corpus"
            value={config.corpus}
            options={CORPUS_OPTIONS}
            onChange={(corpus) => setConfig({ corpus })}
          />

          <Segmented
            label="Retrievers"
            value={config.retrievers}
            options={RETRIEVER_OPTIONS}
            onChange={(retrievers) => setConfig({ retrievers })}
          />

          <div className="space-y-3">
            <div className="text-[11px] font-semibold uppercase tracking-wider text-[hsl(var(--muted-foreground))]">
              Stages
            </div>

            <Toggle
              label="Multi-hop"
              hint="Decompose the question into follow-up lookups"
              checked={config.use_multi_hop}
              onChange={(use_multi_hop) => setConfig({ use_multi_hop })}
            />

            {config.use_multi_hop && (
              <div className="pl-6">
                <label
                  htmlFor="max-hops"
                  className="flex items-center justify-between text-[11px] text-[hsl(var(--muted-foreground))]"
                >
                  <span>Max hops</span>
                  <span className="font-mono text-xs text-[hsl(var(--primary))]">
                    {config.max_hops}
                  </span>
                </label>
                <input
                  id="max-hops"
                  type="range"
                  min={MIN_HOPS}
                  max={MAX_HOPS}
                  step={1}
                  value={config.max_hops}
                  onChange={(e) => setConfig({ max_hops: Number(e.target.value) })}
                  className="mt-1 w-full accent-[hsl(var(--primary))]"
                />
              </div>
            )}

            <Toggle
              label="Corrective (CRAG)"
              hint="Grade retrieved context and correct it when weak"
              checked={config.use_corrective}
              onChange={(use_corrective) => setConfig({ use_corrective })}
            />

            <Toggle
              label="Reranker"
              hint="Reorder merged candidates with the local cross-encoder"
              checked={config.use_reranker}
              onChange={(use_reranker) => setConfig({ use_reranker })}
            />
          </div>

          {!isDefault && (
            <p className="rounded border border-dashed border-[hsl(var(--border))] px-2 py-1.5 text-[11px] leading-snug text-[hsl(var(--muted-foreground))]">
              Running a reduced pipeline. Answers may be less grounded than the
              default configuration.
            </p>
          )}
        </div>
      )}
    </section>
  );
};

export default PipelineConfigPanel;
