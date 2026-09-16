import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import ts from 'typescript';

const source = fs.readFileSync(new URL('../src/services/websocket.ts', import.meta.url), 'utf8')
  .replace('import.meta.env.VITE_WS_URL', 'globalThis.wsUrl');
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }).outputText;
function setup(wsUrl) {
  const sockets = [], timers = new Map(), results = [], errors = [];
  let timerId = 0;
  class Socket {
    constructor(url) { this.url = url; this.sent = []; sockets.push(this); }
    send(data) { this.sent.push(JSON.parse(data)); }
    close(code) { this.closed = code; }
  }
  const env = {
    exports: {}, wsUrl, WebSocket: Socket,
    window: { location: { protocol: 'https:', host: 'docs.example.test' } },
    setTimeout: (callback, delay) => { timers.set(++timerId, { callback, delay }); return timerId; },
    clearTimeout: id => timers.delete(id),
  };
  vm.runInNewContext(compiled, env);
  const start = () => env.exports.generateChatStream('question', 'alice', () => {},
    result => results.push(result), error => errors.push(error),
    { corpus: 'user', use_reranker: false }, { openrouter: 'test-key' });
  const retry = () => {
    const entry = [...timers].find(([, timer]) => timer.delay < 15_000);
    assert.ok(entry, 'a reconnect should be scheduled');
    timers.delete(entry[0]); entry[1].callback();
  };
  return { sockets, timers, results, errors, start, retry };
}

test('configured WebSocket origin and separate settings reach the server', () => {
  const app = setup('wss://api.example.test/'); app.start();
  const socket = app.sockets[0];
  assert.equal(socket.url, 'wss://api.example.test/ws/query/stream/');
  socket.onopen();
  assert.equal(socket.sent[0].CONFIG.corpus, 'user');
  assert.equal(socket.sent[0].KEYS.openrouter, 'test-key');
  assert.equal(socket.sent[0].CONFIG.openrouter, undefined);
});

test('successful answers close the stream and clear timers', () => {
  const app = setup(); app.start(); const socket = app.sockets[0]; socket.onopen();
  socket.onmessage({ data: JSON.stringify({ stage: 'result', answer: 'From your file' }) });
  socket.onclose({ code: 1000 });
  assert.equal(app.results[0].answer, 'From your file');
  assert.equal(app.errors.length, 0);
  assert.equal(app.timers.size, 0);
});

test('connection setup retries are bounded', () => {
  const app = setup(); app.start();
  for (let i = 0; i < 3; i++) { app.sockets[i].onclose({ code: 1006 }); app.retry(); }
  app.sockets[3].onclose({ code: 1006 });
  assert.equal(app.sockets.length, 4);
  assert.equal(app.errors.length, 1);
  assert.equal(app.timers.size, 0);
});

test('a disconnect after sending never automatically resubmits the question', () => {
  const app = setup(); app.start(); app.sockets[0].onopen(); app.sockets[0].onclose({ code: 1006 });
  assert.equal(app.errors.length, 1);
  assert.equal(app.sockets.length, 1);
  assert.equal(app.timers.size, 0);
});

test('cancelling closes the current retry connection and cancels later reconnects', () => {
  const app = setup(); const stream = app.start(); app.sockets[0].onclose({ code: 1006 }); app.retry();
  stream.close(); app.sockets[1].onclose({ code: 1006 });
  assert.equal(app.sockets[1].closed, 1000);
  assert.equal(app.timers.size, 0);
  assert.equal(app.errors.length, 0);
});

test('invalid frames end the loading state with an error', () => {
  const app = setup(); app.start(); app.sockets[0].onopen();
  app.sockets[0].onmessage({ data: 'invalid JSON' });
  assert.equal(app.errors.length, 1);
  assert.equal(app.timers.size, 0);
});

test('normal closure without a result still reports the missing answer', () => {
  const app = setup(); app.start(); app.sockets[0].onopen(); app.sockets[0].onclose({ code: 1000 });
  assert.equal(app.errors.length, 1);
});
