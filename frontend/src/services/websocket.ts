import type { ChatContextResponse } from "../interface";
import type { EvalScore } from "../types/types";

const protocol = window.location.protocol === "https:" ? "wss" : "ws";
export const WS_BASE_URL = (import.meta.env.VITE_WS_URL || `${protocol}://${window.location.host}`).replace(/\/$/, "");
const MAX_RECONNECT_ATTEMPTS = 3;

export type ChatStream = { close: () => void };
export type ChatResult = {
  answer: string;
  conversation_id?: number;
  context?: ChatContextResponse[];
  evaluation?: EvalScore;
  notices?: string[];
  degraded?: string[];
};

/** Retry connection setup only: resending an accepted query can bill the user twice. */
export const generateChatStream = (
  query: string,
  username: string,
  onStatus: (message: string) => void,
  onResult: (data: ChatResult) => void,
  onError: (message: string) => void,
  config?: Record<string, unknown>,
  keys?: Record<string, unknown>,
): ChatStream => {
  let attempt = 0;
  let stopped = false;
  let sent = false;
  let socket: WebSocket | undefined;
  let retryTimer: ReturnType<typeof setTimeout> | undefined;
  let connectionTimer: ReturnType<typeof setTimeout> | undefined;

  const close = () => {
    stopped = true;
    clearTimeout(retryTimer);
    clearTimeout(connectionTimer);
    socket?.close(1000);
  };
  const fail = (message: string) => {
    if (stopped) return;
    close();
    onError(message);
  };
  const connect = () => {
    if (stopped) return;
    try {
      socket = new WebSocket(`${WS_BASE_URL}/ws/query/stream/`);
    } catch {
      fail("Could not connect to chat. Check the WebSocket address.");
      return;
    }
    const ws = socket;
    connectionTimer = setTimeout(() => fail("Connection timed out. Please try again."), 15_000);
    ws.onopen = () => {
      if (stopped) { ws.close(1000); return; }
      clearTimeout(connectionTimer);
      connectionTimer = setTimeout(() => fail("The answer timed out. Please try again."), 330_000);
      const payload: Record<string, unknown> = { USER: username, QUERY: query };
      if (config) payload.CONFIG = config;
      if (keys && Object.keys(keys).length) payload.KEYS = keys;
      ws.send(JSON.stringify(payload));
      sent = true;
    };
    ws.onmessage = (event) => {
      if (stopped) return;
      let message;
      try { message = JSON.parse(event.data); }
      catch { fail("The server sent an invalid response. Please try again."); return; }
      if (!message || typeof message !== "object") {
        fail("The server sent an invalid response. Please try again."); return;
      }
      if (message.stage === "result") {
        if (typeof message.answer !== "string") { fail("The server returned no answer."); return; }
        close();
        onResult(message);
      } else if (message.stage === "error") {
        fail(message.message || "The query failed. Please try again.");
      } else if (typeof message.message === "string") {
        onStatus(message.message);
      }
    };
    ws.onerror = () => { /* onclose handles connection failure */ };
    ws.onclose = () => {
      clearTimeout(connectionTimer);
      if (stopped) return;
      if (!sent && attempt < MAX_RECONNECT_ATTEMPTS) {
        attempt += 1;
        onStatus(`Reconnecting (${attempt}/${MAX_RECONNECT_ATTEMPTS})…`);
        retryTimer = setTimeout(connect, 1000 * 2 ** (attempt - 1));
      } else {
        fail("Connection lost before the answer arrived. Please try again.");
      }
    };
  };
  connect();
  return { close };
};
