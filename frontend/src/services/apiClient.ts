import axios, { AxiosError, AxiosRequestConfig } from "axios";

const API_BASE = (import.meta.env.VITE_API_URL || import.meta.env.VITE_API_BASE_URL || "/api").replace(/\/$/, "");
const API_VERSION = import.meta.env.VITE_API_VERSION || "v1";

/** Maximum number of retries for transient failures. */
const MAX_RETRIES = 2;
/** Base delay in ms for exponential backoff between retries. */
const RETRY_BASE_DELAY_MS = 1000;

export const apiClient = axios.create({
    baseURL: `${API_BASE}/${API_VERSION}`,
    // 60 s covers the slow /insert-data/ upload + indexing dispatch path.
    timeout: 60_000,
});

/**
 * Returns true for errors that are worth retrying automatically:
 * network failures (no response at all) and 502/503/504 from the
 * reverse proxy.
 */
function isRetryable(error: AxiosError): boolean {
    if (!error.response) return true; // network error / timeout
    return [502, 503, 504].includes(error.response.status);
}

// ── Response interceptor: retry transient errors with backoff ─────────
apiClient.interceptors.response.use(
    (response) => response,
    async (error: AxiosError) => {
        const config = error.config as AxiosRequestConfig & { _retryCount?: number };
        if (!config) return Promise.reject(error);

        config._retryCount = config._retryCount ?? 0;

        if (config.method?.toLowerCase() === "get" && isRetryable(error) && config._retryCount < MAX_RETRIES) {
            config._retryCount += 1;
            const delay = RETRY_BASE_DELAY_MS * Math.pow(2, config._retryCount - 1);
            await new Promise((res) => setTimeout(res, delay));
            return apiClient.request(config);
        }

        return Promise.reject(error);
    }
);
