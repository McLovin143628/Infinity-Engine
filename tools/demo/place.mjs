// Place the FEMALE committed body beside the level's pawn, over the Chrome
// DevTools Protocol — wave CHAR1a.2.
//
//   node tools/demo/place.mjs [port]
//
// Exit 0 = placed (the new actor's guid is printed); exit 2 = no page target;
// exit 3 = the command refused, with its message.
//
// # Why this exists, and why it does not save
//
// The wave was asked for a frame of a street NPC wearing the second committed
// body. The island's `.inf_lvl` is committed and this wave may not regenerate
// one, so the body cannot be added to the level — but it can be added to the
// open DOCUMENT, which is what an author does and what `Place Actor ▸ Starter
// Character` is for. Nothing is written to disk: the document is left dirty and
// the demo loop never presses Ctrl+S.
//
// PIE then runs the document that has her in it, and because
// `society::level_archetypes` surveys every distinct rigged body a level offers,
// half the street's agents wear her.

const port = Number(process.argv[2] ?? 9222);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function pageTarget() {
  for (let i = 0; i < 60; i++) {
    try {
      const r = await fetch(`http://127.0.0.1:${port}/json`);
      const list = await r.json();
      const page = list.find((t) => t.type === "page" && t.webSocketDebuggerUrl);
      if (page) return page;
    } catch {
      /* the port is not open yet */
    }
    await sleep(1000);
  }
  return null;
}

const page = await pageTarget();
if (!page) {
  console.log(`NO PAGE TARGET on port ${port} after 60 s`);
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
const evalJs = async (expression) => {
  const r = await send("Runtime.evaluate", {
    expression,
    returnByValue: true,
    awaitPromise: true,
  });
  if (r.result?.exceptionDetails) {
    return "EXC: " + JSON.stringify(r.result.exceptionDetails).slice(0, 400);
  }
  return r.result?.result?.value;
};

// `__TAURI_INTERNALS__.invoke` and not the app's own `ipc.ts` wrapper: this
// script drives a RELEASE bundle, whose module graph is not reachable from the
// console. The command name and its argument shape are the contract either way.
const invoke = (cmd, args) =>
  evalJs(
    `window.__TAURI_INTERNALS__.invoke(${JSON.stringify(cmd)}, ${JSON.stringify(args)})`,
  );

const pawn = await invoke("scene_player_pawn", {});
console.log("pawn:", pawn);
// No `at`: the command places her beside the pawn, which on a 50 km2 island is
// the difference between "in the shot" and "in the sea two towns away".
const placed = await invoke("character_place_starter", { at: null, female: true });
console.log("placed:", placed);

// ── NO SKIN WORKAROUND HERE ANY MORE ──────────────────────────────
//
// Wave CHAR1a.2 bound the skin from HERE, through `scene_add_component` +
// `scene_apply_material`, because `SceneDoc::edit_create_character` inserted no
// `Material` and the fix re-blesses committed levels. The CHAR1a audit landed
// that fix: the door takes a `CharacterSkin` and inserts the component, and the
// four committed levels that spawn a character were re-blessed with the cause.
//
// So the workaround is gone, and that is the point: a demo loop that dresses the
// scene before photographing it photographs the loop, not the engine. What these
// frames show now is what an author gets.

ws.close();
process.exit(typeof placed === "string" && placed.length > 8 ? 0 : 3);
