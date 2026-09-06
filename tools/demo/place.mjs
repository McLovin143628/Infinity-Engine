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

// ── THE SKIN, BOUND ─────────────────────────────────────────────────────────
//
// **The finding this wave measured, and the workaround it is allowed.** The
// island hero carries Transform, Visibility, SkeletalMesh, AnimStateMachine,
// RigidBody3D and Collider3D — and NO `Material`. Read off the running editor
// with `scene_details`, not inferred. `SceneDoc::edit_create_character` has never
// inserted one, so the skin the New Character wizard writes
// (`Starter_Skin.inf_mat`, which `inf-import --rebind-character` fills with the
// mannequin's own albedo/normal/ORM) is bound by NOTHING: both hosts read
// `Material` -> None, hand `vt_set_for` a `None` and draw the renderer's neutral
// 0.8 grey. The editor's "untextured white" and PIE's "grey with dark limbs" are
// one fact under two lights, not two bugs.
//
// The fix is one insert in that door — and it re-blesses every committed level
// that spawns a character, which this wave may not do. So the binding is applied
// HERE, through `scene_apply_material`: the same door an author takes when they
// drag a material onto an actor, one undo step, and nothing written to disk.
const SKIN_M = "00000000-0000-0000-0000-00005c1000a1";
const SKIN_F = "00000000-0000-0000-0000-00005c1000b1";
for (const [who, id, guid] of [["hero", SKIN_M, pawn], ["female", SKIN_F, placed]]) {
  if (typeof guid !== "string" || guid.length < 8) continue;
  // **The component first, and that is a finding of its own.**
  // `edit_apply_material` skips any target that does not already carry a
  // `Material` ("Only entities that already carry a Material component") and
  // returns a count — so dragging a material onto a character in the viewport
  // applies to nothing and reports nothing. Measured here: `skin on hero: 0`
  // until this line existed.
  await invoke("scene_add_component", {
    guids: [guid],
    typePath: "inf_ecs::components::Material",
  });
  const n = await invoke("scene_apply_material", { assetId: id, targets: [guid] });
  console.log(`skin on ${who}: ${n}`);
}
ws.close();
process.exit(typeof placed === "string" && placed.length > 8 ? 0 : 3);
