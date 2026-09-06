// Frame the hero's FACE in the editor viewport, over the Chrome DevTools
// Protocol — wave CHAR1a.3 audit.
//
//   node tools/demo/portrait.mjs [port] [--head=0.775] [--dist=1.0] [--guid=…]
//
// Exit 0 = framed (the numbers are printed); 2 = no page target; 3 = the
// commands refused.
//
// # Why this exists
//
// A wave that puts a MetaHuman on the island makes a claim about a FACE, and the
// demo loop's camera sits behind the character: a head is thirty pixels of a
// 1080p frame, and a 12× crop of thirty pixels proves nothing about eyes, lips
// or brows. Wave CHAR1a.3 shipped two BLANK heads — the combined body's face
// tile sampling the body's atlas — and its report called them "real faces". A
// portrait is cheaper than an argument.
//
// # How it frames one WITHOUT a camera command
//
// There is no "put the camera here" IPC door, and there should not be one just
// for this. `viewport_focus` frames the selection, but it queues a viewport
// command whose `selection_center` averages the selection's RENDER INSTANCES —
// an object the projector has not seen yet has none — and it was measured here
// doing nothing at all for a marker created moments before.
//
// So the character moves instead of the camera. The editor opens at
// `EngineHost::player_start_pose`: eye at `p + 1.75·Y − flat·7`, pitch −0.12 rad,
// where `flat = (sin yaw, 0, −cos yaw)`. That is arithmetic this script can do,
// so it puts the hero's HEAD on the camera's own view ray at arm's length. The
// edit is to the open DOCUMENT only; nothing is written to disk, exactly as
// `place.mjs`'s is not, and the demo loop never presses Ctrl+S — `undo.mjs` puts
// him back before Play.
//
// The character is NOT turned around: his mesh's authored forward is +Z while
// the pawn's `flat` is −Z, so at his own rotation he already faces the camera.
// Measured — a 180° turn photographed the back of his skull.

const port = Number(process.argv[2] ?? 9222);
const arg = (k, d) =>
  Number(process.argv.find((a) => a.startsWith(`--${k}=`))?.slice(k.length + 3) ?? d);
const HEAD_M = arg("head", 0.775); // eye height ABOVE THE ENTITY ORIGIN — see below
const DIST_M = arg("dist", 1.0);
const EYE_M = 1.75; // EngineHost::PLAYER_START_EYE_M
const BACK_M = 7.0; // EngineHost::PLAYER_START_BACK_M
const PITCH = -0.12; // EngineHost::PLAYER_START_PITCH, radians

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
const invoke = (cmd, args) =>
  evalJs(
    `window.__TAURI_INTERNALS__.invoke(${JSON.stringify(cmd)}, ${JSON.stringify(args)})`,
  );

const TRANSFORM = "inf_ecs::components::Transform";
const field = (details, name) =>
  details?.components
    ?.find((c) => c.type_path === TRANSFORM)
    ?.fields?.find((f) => f.name === name)?.value?.value;

// Whose face: the pawn by default, anyone with `--guid=…`.
const wantGuid = process.argv.find((a) => a.startsWith("--guid="))?.slice(7);
const subject = wantGuid ?? (await invoke("scene_player_pawn", {}));
if (typeof subject !== "string" || subject.length < 8 || subject.startsWith("EXC:")) {
  console.log("NO SUBJECT:", subject);
  process.exit(3);
}
console.log("subject:", subject);

await invoke("scene_select", { guids: [subject], additive: false });
const before = await invoke("scene_details", {});
const pos = field(before, "translation");
const rot = field(before, "rotation");
console.log("subject transform:", JSON.stringify({ pos, rot }));
if (!Array.isArray(pos) || !Array.isArray(rot)) {
  console.log("NO TRANSFORM ON THE SUBJECT");
  process.exit(3);
}

// The camera the editor opened with, from the level's own player start.
const yaw = (rot[1] * Math.PI) / 180;
const flat = [Math.sin(yaw), 0, -Math.cos(yaw)];
const eye = [pos[0] - flat[0] * BACK_M, pos[1] + EYE_M, pos[2] - flat[2] * BACK_M];
const fwd = [
  Math.sin(yaw) * Math.cos(PITCH),
  Math.sin(PITCH),
  -Math.cos(yaw) * Math.cos(PITCH),
];
// Put the eyes on the view ray, a stated distance out.
//
// `--head` is measured from the ENTITY ORIGIN, which is the pawn's capsule
// CENTRE and not its feet: `character_place_starter`'s own `feet_offset_m` is
// 0.6125 + 0.2625 = 0.875 m (the CHAR1a audit measured it the hard way, having
// floated a character exactly that far into the air). So a 1.78 m MetaHuman's
// eyes at 1.65 m above the ground are 0.775 m above the origin, and that is the
// default. Measured the same hard way: 1.65 put the crown of his head on the
// bottom edge of the frame.
const at = [
  eye[0] + fwd[0] * DIST_M,
  eye[1] + fwd[1] * DIST_M - HEAD_M,
  eye[2] + fwd[2] * DIST_M,
];
console.log(
  "eye:",
  JSON.stringify(eye.map((v) => Number(v.toFixed(3)))),
  "head:",
  JSON.stringify(at.map((v, i) => Number((i === 1 ? v + HEAD_M : v).toFixed(3)))),
);

const res = await invoke("scene_set_property", {
  guids: [subject],
  typePath: TRANSFORM,
  field: "translation",
  value: { kind: "vec3", value: at },
});
if (typeof res === "string" && res.startsWith("EXC:")) {
  console.log("MOVE REFUSED:", res);
  process.exit(3);
}
// Drop the selection: the pawn's collision capsule is drawn around a selected
// character and a portrait is not a picture of a capsule.
await invoke("scene_select", { guids: [], additive: false });
// The document version has to reach the viewport before the shutter.
await sleep(2500);
console.log(`PORTRAIT FRAMED at ${DIST_M} m, head ${HEAD_M} m`);
ws.close();
process.exit(0);
