const protocol = window.location.protocol === "https:" ? "wss" : "ws";

export const WS_BASE_URL =
    // import.meta.env.VITE_WS_URL ||
    `${protocol}://${window.location.host}`; 

/** Maximum number of automatic reconnection attempts. */
const MAX_RECONNECT_ATTEMPTS = 3;
/** Base delay in ms for exponential backoff. */
const RECONNECT_BASE_DELAY_MS = 1000;

export const generateChatStream = (
  query: string,
  username: string,
  onStatus: (message: string) => void,
  onResult: (data: any) => void,
  onError: (message: string) => void,
  config?: Record<string, unknown>,
  /**
   * The user's own API keys, sent as their own top-level KEYS field rather than
   * inside CONFIG. The server logs CONFIG verbatim and summarises it into the
   * status events shown on screen; KEYS is redacted before either.
   */
  keys?: Record<string, unknown>,
) => {
  let attempt = 0;

  const connect = (): WebSocket => {
    const ws = new WebSocket(`${WS_BASE_URL}/ws/query/stream/`);

    ws.onopen = () => {
      attempt = 0; // reset on successful connect
      const payload: Record<string, unknown> = { USER: username, QUERY: query };
      if (config) payload.CONFIG = config;
      if (keys && Object.keys(keys).length) payload.KEYS = keys;
      ws.send(JSON.stringify(payload));
    };

    ws.onmessage = (e) => {
      const msg = JSON.parse(e.data);
      switch (msg.stage) {
        case "result": onResult(msg); ws.close(); break;
        case "error":  onError(msg.message);  ws.close(); break;
        default:       onStatus(msg.stage ? `${msg.stage} — ${msg.message}` : msg.message); break;
      }
    };

    ws.onerror = () => {
      // onerror is always followed by onclose — reconnection is handled there.
    };

    ws.onclose = (ev) => {
      // Normal closure (1000) or explicit close from onmessage — no reconnect.
      if (ev.code === 1000) return;

      if (attempt < MAX_RECONNECT_ATTEMPTS) {
        attempt += 1;
        const delay = RECONNECT_BASE_DELAY_MS * Math.pow(2, attempt - 1);
        onStatus(`Connection lost — reconnecting (attempt ${attempt}/${MAX_RECONNECT_ATTEMPTS})…`);
        setTimeout(() => connect(), delay);
      } else {
        onError("Connection lost. Please try again.");
      }
    };

    return ws;
  };

  return connect();
};