import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { PipelineConfig } from "../types/types";

/**
 * Per-query pipeline configuration, shared between the Sidebar (which edits it)
 * and the Chatbot (which sends it with every query).
 *
 * Keys are snake_case on purpose: this object is serialised straight onto the
 * websocket payload as CONFIG, and the backend contract lives in
 * backend/common/pipeline_config.py. Keep the two in sync.
 */

export const DEFAULT_PIPELINE_CONFIG: PipelineConfig = {
  use_multi_hop: true,
  max_hops: 3,
  use_corrective: true,
  use_reranker: true,
  retrievers: "both",
  corpus: "auto",
};

export const MIN_HOPS = 1;
export const MAX_HOPS = 3;

const STORAGE_KEY = "pipeline_config";

type PipelineConfigContextValue = {
  config: PipelineConfig;
  setConfig: (patch: Partial<PipelineConfig>) => void;
  resetConfig: () => void;
  isDefault: boolean;
};

const PipelineConfigContext = createContext<PipelineConfigContextValue | null>(null);

/** Merge stored/unknown input onto the defaults, dropping anything invalid. */
const sanitize = (raw: unknown): PipelineConfig => {
  const base = { ...DEFAULT_PIPELINE_CONFIG };
  if (!raw || typeof raw !== "object") return base;
  const input = raw as Record<string, unknown>;

  const bool = (key: keyof PipelineConfig) => {
    if (typeof input[key] === "boolean") {
      (base[key] as boolean) = input[key] as boolean;
    }
  };
  bool("use_multi_hop");
  bool("use_corrective");
  bool("use_reranker");

  if (typeof input.max_hops === "number" && Number.isFinite(input.max_hops)) {
    base.max_hops = Math.min(MAX_HOPS, Math.max(MIN_HOPS, Math.trunc(input.max_hops)));
  }
  if (input.retrievers === "both" || input.retrievers === "dense" || input.retrievers === "sparse") {
    base.retrievers = input.retrievers;
  }
  if (input.corpus === "auto" || input.corpus === "user" || input.corpus === "base") {
    base.corpus = input.corpus;
  }
  return base;
};

const readStored = (): PipelineConfig => {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    return sanitize(stored ? JSON.parse(stored) : null);
  } catch {
    // Corrupt JSON in localStorage must never stop the app from loading.
    return { ...DEFAULT_PIPELINE_CONFIG };
  }
};

export const PipelineConfigProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [config, setConfigState] = useState<PipelineConfig>(readStored);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(config));
    } catch {
      // Private-mode / quota failures are not worth breaking a query over.
    }
  }, [config]);

  const setConfig = useCallback((patch: Partial<PipelineConfig>) => {
    setConfigState((prev) => sanitize({ ...prev, ...patch }));
  }, []);

  const resetConfig = useCallback(() => {
    setConfigState({ ...DEFAULT_PIPELINE_CONFIG });
  }, []);

  const value = useMemo<PipelineConfigContextValue>(
    () => ({
      config,
      setConfig,
      resetConfig,
      isDefault: (Object.keys(DEFAULT_PIPELINE_CONFIG) as (keyof PipelineConfig)[]).every(
        (key) => config[key] === DEFAULT_PIPELINE_CONFIG[key]
      ),
    }),
    [config, setConfig, resetConfig]
  );

  return (
    <PipelineConfigContext.Provider value={value}>{children}</PipelineConfigContext.Provider>
  );
};

/**
 * Falls back to the defaults when used outside a provider, so a component can
 * read the config without the app crashing if the tree changes.
 */
export const usePipelineConfig = (): PipelineConfigContextValue => {
  const ctx = useContext(PipelineConfigContext);
  if (ctx) return ctx;
  return {
    config: { ...DEFAULT_PIPELINE_CONFIG },
    setConfig: () => undefined,
    resetConfig: () => undefined,
    isDefault: true,
  };
};
