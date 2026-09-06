import { ModelCatalog, ModelOption } from "../types/types";
import { apiClient } from "./apiClient";

/**
 * The model lists the pickers offer.
 *
 * Fetched from the backend rather than from OpenRouter directly: the server
 * caches one copy for every user and does not depend on OpenRouter's CORS
 * policy or per-IP rate limits. It also serves a bundled fallback list when
 * OpenRouter is unreachable, so the picker is never empty.
 *
 * Cached again in this browser so reopening the settings panel is instant and
 * the list survives a page reload with no network at all.
 */

const STORAGE_KEY = "model_catalog_cache";

/** Matches the server's own cache window; the catalog changes over days. */
const CACHE_TTL_MS = 60 * 60 * 1000;

export const EMPTY_CATALOG: ModelCatalog = { chat: [], embedding: [] };

type CachedCatalog = { catalog: ModelCatalog; storedAt: number };

const isModelOption = (raw: unknown): raw is ModelOption =>
  Boolean(raw) &&
  typeof raw === "object" &&
  typeof (raw as ModelOption).id === "string" &&
  (raw as ModelOption).id.length > 0;

/** Keep only well-formed entries; a nameless one still renders by id. */
const sanitizeList = (raw: unknown): ModelOption[] => {
  if (!Array.isArray(raw)) return [];
  return raw.filter(isModelOption).map((entry) => ({
    ...entry,
    name: typeof entry.name === "string" && entry.name ? entry.name : entry.id,
  }));
};

const sanitizeCatalog = (raw: unknown): ModelCatalog => {
  if (!raw || typeof raw !== "object") return { ...EMPTY_CATALOG };
  const input = raw as Record<string, unknown>;
  return {
    chat: sanitizeList(input.chat),
    embedding: sanitizeList(input.embedding),
    source: typeof input.source === "string" ? input.source : undefined,
    fetched_at: typeof input.fetched_at === "number" ? input.fetched_at : undefined,
  };
};

const readCache = (): CachedCatalog | null => {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (!stored) return null;
    const parsed = JSON.parse(stored) as Partial<CachedCatalog>;
    if (typeof parsed?.storedAt !== "number") return null;
    const catalog = sanitizeCatalog(parsed.catalog);
    if (!catalog.chat.length && !catalog.embedding.length) return null;
    return { catalog, storedAt: parsed.storedAt };
  } catch {
    return null;
  }
};

const writeCache = (catalog: ModelCatalog): void => {
  try {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({ catalog, storedAt: Date.now() } satisfies CachedCatalog)
    );
  } catch {
    // Quota or private mode — the catalog is a convenience, not state.
  }
};

/** The cached catalog regardless of age, for painting the panel immediately. */
export const cachedCatalog = (): ModelCatalog | null => readCache()?.catalog ?? null;

/**
 * The catalog, from this browser's cache when it is fresh enough, otherwise
 * from the server.
 *
 * Never rejects. A failed fetch with a stale cache serves the stale copy — an
 * older real catalog beats an empty picker — and a failed fetch with no cache
 * at all returns empty lists, which the picker renders as "type an id".
 */
export const fetchModelCatalog = async (
  options: { force?: boolean } = {}
): Promise<ModelCatalog> => {
  const cached = readCache();

  if (!options.force && cached && Date.now() - cached.storedAt < CACHE_TTL_MS) {
    return { ...cached.catalog, source: cached.catalog.source ?? "cache" };
  }

  try {
    const response = await apiClient.get(
      options.force ? "/models/?refresh=1" : "/models/",
      { headers: { Accept: "application/json" } }
    );
    // The API wraps some responses in { data }, and returns others bare.
    const body = response.data;
    const catalog = sanitizeCatalog(
      body && typeof body === "object" && "chat" in body ? body : body?.data
    );

    if (catalog.chat.length || catalog.embedding.length) {
      writeCache(catalog);
      return catalog;
    }
    throw new Error("model catalog response contained no models");
  } catch (error) {
    if (cached) {
      console.warn("Model catalog refresh failed — using the cached list:", error);
      return { ...cached.catalog, source: "cache" };
    }
    console.warn("Model catalog unavailable:", error);
    return { ...EMPTY_CATALOG };
  }
};

/** Human-readable price, or null when OpenRouter reported none. */
export const formatPrice = (perToken?: number | null): string | null => {
  if (typeof perToken !== "number" || Number.isNaN(perToken)) return null;
  if (perToken === 0) return "free";
  // Per-million is the unit every provider quotes, and per-token values are
  // small enough that they render as 0.00 otherwise.
  const perMillion = perToken * 1_000_000;
  const digits = perMillion < 1 ? 3 : 2;
  return `$${perMillion.toFixed(digits)}/M`;
};

/** "128K" style context length, or null when unknown. */
export const formatContext = (length?: number | null): string | null => {
  if (typeof length !== "number" || !Number.isFinite(length) || length <= 0) {
    return null;
  }
  if (length >= 1_000_000) return `${Math.round(length / 1_000_000)}M ctx`;
  if (length >= 1_000) return `${Math.round(length / 1_000)}K ctx`;
  return `${length} ctx`;
};
