// Press the editor's own Undo N times over the Chrome DevTools Protocol —
// wave CHAR1a.3 audit.
//
//   node tools/demo/undo.mjs [port] [times]
//
// The portrait step turns the hero to face the camera and drops a marker at his
// head, in the open DOCUMENT. Every frame after it must be the level's own pose
// again, and the editor already has one door for that: `scene_undo`, the same
// transaction stack Ctrl+Z drives. Undoing is better than "put it back", which
// would be a second implementation of the same edits and could disagree.

const port = Number(process.argv[2] ?? 9222);
const times = Number(process.argv[3] ?? 1);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const r = await fetch(`http://127.0.0.1:${port}/json`).catch(() => null);
const list = r ? await r.json() : [];
const page = list.find((t) => t.type === "page" && t.webSocketDebuggerUrl);
if (!page) {
  console.log(`NO PAGE TARGET on port ${port}`);
  process.exit(2);
}
const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((res, rej) => {
  ws.onopen = res;
  ws.onerror = rej;
});
let id = 0;
const pending = new Map();
ws.onmessage = (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) {
    pending.get(m.id)(m);
    pending.delete(m.id);
  }
};
const send = (method, params = {}) =>
  new Promise((res) => {
    const i = ++id;
    pending.set(i, res);
    ws.send(JSON.stringify({ id: i, method, params }));
  });
for (let k = 0; k < times; k++) {
  await send("Runtime.evaluate", {
    expression: `window.__TAURI_INTERNALS__.invoke("scene_undo", {})`,
    returnByValue: true,
    awaitPromise: true,
  });
  await sleep(150);
}
console.log(`undo ×${times}`);
ws.close();
process.exit(0);
