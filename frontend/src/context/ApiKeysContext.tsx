import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import { ApiKeyName, ApiKeys } from "../types/types";

/**
 * The user's own API keys ("bring your own key"), shared between the settings
 * panel that edits them and the chat/upload paths that send them.
 *
 * Kept in a separate context from PipelineConfigContext on purpose. That object
 * is whitelisted by `keyof PipelineConfig` on every write and is logged verbatim
 * by the server; a credential in it would be both silently dropped and, once it
 * wasn't, written to a log file. These travel as their own top-level KEYS field
 * (see backend/common/runtime/api_keys.py), which the server redacts before logging.
 *
 * Storage is this browser's localStorage and nowhere else — the keys are sent
 * with each request, used for that request, and never persisted server-side.
 * That does mean any script running on this origin can read them, which is the
 * standard trade for BYOK and why `clearKeys` is offered prominently.
 */

export const API_KEY_NAMES: ApiKeyName[] = ["openrouter", "news"];

export const EMPTY_API_KEYS: ApiKeys = { openrouter: "", news: "" };

/** Mirrors _ALLOWED_KEY_CHARS in backend/common/runtime/api_keys.py. */
const ALLOWED_KEY_CHARS = /^[A-Za-z0-9._~+/=:-]+$/;

/** Mirrors MAX_KEY_LENGTH in backend/common/runtime/api_keys.py. */
export const MAX_KEY_LENGTH = 400;

const STORAGE_KEY = "byok_api_keys";

export type ApiKeyProblem = "invalid-characters" | "too-long";

type ApiKeysContextValue = {
  keys: ApiKeys;
  /** Update one or more keys. Values are trimmed; "" clears a key. */
  setKey: (patch: Partial<ApiKeys>) => void;
  clearKeys: () => void;
  /** True when at least one key is set — drives the panel's summary badge. */
  hasAnyKey: boolean;
  /**
   * Only the keys that are actually set, ready to send as KEYS. Omitting the
   * blank ones keeps the wire payload honest: an empty string and an absent
   * field both mean "use the server's", so there is no reason to send either.
   */
  keysForRequest: () => Partial<ApiKeys>;
  /**
   * Why a key was rejected, per field, so the input can explain itself. The
   * rejected value is not stored — a key we had to alter is not the user's key,
   * and sending the remainder would surface as an opaque 401.
   */
  problems: Partial<Record<ApiKeyName, ApiKeyProblem>>;
};

const ApiKeysContext = createContext<ApiKeysContextValue | null>(null);

/** Why `value` is unusable as a key, or null when it is fine (or blank). */
export const validateKey = (value: string): ApiKeyProblem | null => {
  const trimmed = value.trim();
  if (!trimmed) return null;
  if (trimmed.length > MAX_KEY_LENGTH) return "too-long";
  if (!ALLOWED_KEY_CHARS.test(trimmed)) return "invalid-characters";
  return null;
};

const sanitize = (raw: unknown): ApiKeys => {
  const base: ApiKeys = { ...EMPTY_API_KEYS };
  if (!raw || typeof raw !== "object") return base;
  const input = raw as Record<string, unknown>;

  API_KEY_NAMES.forEach((name) => {
    const value = input[name];
    if (typeof value !== "string") return;
    const trimmed = value.trim();
    if (trimmed && !validateKey(trimmed)) {
      base[name] = trimmed;
    }
  });

  return base;
};

const readStored = (): ApiKeys => {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    return sanitize(stored ? JSON.parse(stored) : null);
  } catch {
    // Corrupt JSON (or a browser blocking storage) must never stop the app
    // from loading — the user can always paste the key again.
    return { ...EMPTY_API_KEYS };
  }
};

export const ApiKeysProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [keys, setKeys] = useState<ApiKeys>(readStored);
  const [problems, setProblems] = useState<
    Partial<Record<ApiKeyName, ApiKeyProblem>>
  >({});

  useEffect(() => {
    try {
      const anySet = API_KEY_NAMES.some((name) => keys[name]);
      if (anySet) {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(keys));
      } else {
        // Nothing to remember: drop the entry rather than leaving an empty
        // object behind that looks like stored credentials.
        localStorage.removeItem(STORAGE_KEY);
      }
    } catch {
      // Private-mode / quota failures are not worth breaking a query over.
    }
  }, [keys]);

  const setKey = useCallback((patch: Partial<ApiKeys>) => {
    const nextProblems: Partial<Record<ApiKeyName, ApiKeyProblem>> = {};
    const accepted: Partial<ApiKeys> = {};

    (Object.keys(patch) as ApiKeyName[]).forEach((name) => {
      if (!API_KEY_NAMES.includes(name)) return;
      const value = (patch[name] ?? "").trim();
      const problem = validateKey(value);
      if (problem) {
        nextProblems[name] = problem;
        // Store nothing rather than keeping the last valid value: what is on
        // screen is what gets sent, and silently holding a different key than
        // the field shows is exactly the kind of surprise BYOK cannot afford.
        accepted[name] = "";
        return;
      }
      accepted[name] = value;
    });

    setProblems((prev) => {
      const merged = { ...prev };
      (Object.keys(patch) as ApiKeyName[]).forEach((name) => {
        // Clear a stale problem as soon as the field becomes valid again.
        if (nextProblems[name]) merged[name] = nextProblems[name];
        else delete merged[name];
      });
      return merged;
    });

    if (Object.keys(accepted).length) {
      setKeys((prev) => ({ ...prev, ...accepted }));
    }
  }, []);

  const clearKeys = useCallback(() => {
    setKeys({ ...EMPTY_API_KEYS });
    setProblems({});
  }, []);

  const keysForRequest = useCallback((): Partial<ApiKeys> => {
    const payload: Partial<ApiKeys> = {};
    API_KEY_NAMES.forEach((name) => {
      if (keys[name]) payload[name] = keys[name];
    });
    return payload;
  }, [keys]);

  const value = useMemo<ApiKeysContextValue>(
    () => ({
      keys,
      setKey,
      clearKeys,
      hasAnyKey: API_KEY_NAMES.some((name) => Boolean(keys[name])),
      keysForRequest,
      problems,
    }),
    [keys, setKey, clearKeys, keysForRequest, problems]
  );

  return (
    <ApiKeysContext.Provider value={value}>{children}</ApiKeysContext.Provider>
  );
};

/**
 * Falls back to "no keys" outside a provider, so a component can read them
 * without the app crashing if the tree changes. Sending no keys is always a
 * valid request — the server falls back to its own credentials.
 */
export const useApiKeys = (): ApiKeysContextValue => {
  const ctx = useContext(ApiKeysContext);
  if (ctx) return ctx;
  return {
    keys: { ...EMPTY_API_KEYS },
    setKey: () => undefined,
    clearKeys: () => undefined,
    hasAnyKey: false,
    keysForRequest: () => ({}),
    problems: {},
  };
};
