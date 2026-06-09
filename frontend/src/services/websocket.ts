const protocol = window.location.protocol === "https:" ? "wss" : "ws";

export const WS_BASE_URL =
    // import.meta.env.VITE_WS_URL ||
    `${protocol}://${window.location.host}`; 

export const generateChatStream = (
  query: string,
  username: string,
  onStatus: (message: string) => void,
  onResult: (data: any) => void,
  onError: (message: string) => void,
) => {
  const ws = new WebSocket(`${WS_BASE_URL}/ws/query/stream/`);

  ws.onopen = () => {
    ws.send(JSON.stringify({ USER: username, QUERY: query }));
  };

  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    switch (msg.stage) {
      case "status": onStatus(msg.message);  break;
      case "result": onResult(msg); ws.close(); break;
      case "error":  onError(msg.message); ws.close(); break;
    }
  };

  ws.onerror = () => { onError("Connection lost."); };

  return ws;
};