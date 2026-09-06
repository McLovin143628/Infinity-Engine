//! **P24.1: a skinned character in PIE.**
//!
//! `run_pie` took no `SkinnedRegistry` and `PlayerApp::new` seeded the inert
//! `Content::None` one, so every character in a **windowed** PIE session — the
//! embedded viewport, the new-window preview — drew as a placeholder cube, while
//! the headless PIE the gates drive and the shipped build both drew real posed
//! geometry. Skeletons, clips and machines had ridden the `ScenePayload` since
//! v3; the `.inf_mesh` bytes the skinned pass needs were the one part that never
//! crossed. `ScenePayload` **v7** carries them.
//!
//! P21.4 fixed the identical class for voxel volumes by threading `voxel_assets`
//! into `run_pie`; this is the same fix, and it carries the same lesson forward:
//! **two empty maps agree perfectly**, so every comparison below asserts the
//! payload's `meshes` vector is non-empty, with an exact count taken from the
//! fixture, before it compares anything.
//!
//! The rendering claim is asserted **structurally** — the registry resolves the
//! mesh to real bind-space geometry and a palette — not in pixels: `run_pie`
//! itself needs a GPU and a display, so like every GPU path it is compile-checked
//! and human-verified. What is machine-checked here is the whole chain that feeds
//! it: document → payload → registry → `SkinnedDraw`.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Duration;

use glam::{Mat4, Quat, Vec3};
use uuid::Uuid;

use inf_anim::{
    AnimClip, AnimClipAsset, Interpolation, Joint, JointTrack, JointTransform, QuatTrack, Skeleton,
    SkeletonAsset, SmState, SmTransition, StateMachine, StateMachineAsset,
};
use inf_ecs::components::{AnimStateMachine, SkeletalMesh};
use inf_editor_core::ipc::SpawnKind;
use inf_editor_core::pie::{build_scene_payload, PieSession};
use inf_editor_core::scene::SceneDoc;
use inf_mesh::{MeshAsset, MeshVertex, SubMesh, VertexSkin};
use inf_player::skinned::SkinnedRegistry;
use inf_runtime::pie::{PlayerToEditor, ScenePayload};

const HERO: Uuid = Uuid::from_u128(0x2401_1000_0000_0000_0000_0000_0000_0001);
const MESH: Uuid = Uuid::from_u128(0x2401_1000_0000_0000_0000_0000_0000_0002);
const SKEL: Uuid = Uuid::from_u128(0x2401_1000_0000_0000_0000_0000_0000_0003);
const SM: Uuid = Uuid::from_u128(0x2401_1000_0000_0000_0000_0000_0000_0004);
const IDLE: Uuid = Uuid::from_u128(0x2401_1000_0000_0000_0000_0000_0000_0005);
const WAVE: Uuid = Uuid::from_u128(0x2401_1000_0000_0000_0000_0000_0000_0006);
const SALUTE: Uuid = Uuid::from_u128(0x2401_1000_0000_0000_0000_0000_0000_0007);

fn player_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_inf-player"))
}

// ── the fixture ─────────────────────────────────────────────────────────────

/// A 2-joint chain: root, then a tip 1 m up.
fn rig() -> SkeletonAsset {
    SkeletonAsset::new(
        Skeleton::new(vec![
            Joint {
                name: "root".into(),
                parent: None,
                inverse_bind: Mat4::IDENTITY.to_cols_array(),
                local_bind: JointTransform::IDENTITY,
            },
            Joint {
                name: "tip".into(),
                parent: Some(0),
                inverse_bind: Mat4::from_translation(Vec3::new(0.0, -1.0, 0.0)).to_cols_array(),
                local_bind: JointTransform::from_trs(Vec3::Y, Quat::IDENTITY, Vec3::ONE),
            },
        ])
        .unwrap(),
    )
}

/// A skinned triangle — three vertices, real influences, so
/// `skinned_mesh_data` accepts it and the skinned pass has something to draw.
fn body() -> MeshAsset {
    let v = |x: f32, y: f32| MeshVertex {
        position: [x, y, 0.0],
        normal: [0.0, 0.0, 1.0],
        ..Default::default()
    };
    let w = |j: u16| VertexSkin {
        joints: [j, 0, 0, 0],
        weights: [1.0, 0.0, 0.0, 0.0],
    };
    MeshAsset::new(
        vec![SubMesh {
            name: "tri".into(),
            vertices: vec![v(0.0, 0.0), v(1.0, 0.0), v(0.0, 2.0)],
            indices: vec![0, 1, 2],
            material_slot: None,
            skin: vec![w(0), w(0), w(1)],
        }],
        vec![],
    )
}

/// A clip holding the tip joint at `deg` about +Z (one stepped key, so the pose
/// a state produces is an exact constant).
fn hold(deg: f32) -> AnimClip {
    let q = Quat::from_rotation_z(deg.to_radians()).to_array();
    // Built through `AnimClip::new` rather than as a struct literal since
    // `.inf_anim` v2 (P29.2): the constructor fills the five new tail channels
    // with their empty defaults, and a literal here would have to be edited every
    // time the format grows a field it does not care about. `duration` is derived
    // from the keys, which for a single stepped key at t=0 is 0 — so it is set
    // afterwards, because this fixture wants a 1 s state period.
    let mut clip = AnimClip::new(
        format!("hold{deg}"),
        vec![JointTrack {
            joint: 1,
            translation: None,
            rotation: Some(QuatTrack::new(vec![0.0], vec![q], Interpolation::Step)),
            scale: None,
        }],
    );
    clip.duration = 1.0;
    clip
}

/// idle → wave → salute, both transitions unconditional, so the machine walks
/// its states on consecutive fixed steps with no actor and no Blueprint variable
/// involved.
///
/// **Three states, not two.** Each state holds a *constant* pose, so a two-state
/// chain settles after one transition and every hash from step 2 on is identical
/// — which would make the "the trace moved" arm below fail for the right reason
/// and the equality arm pass for the wrong one.
/// **The second transition has a DURATION** (P29.4, closing P29.2's recorded
/// boundary). P29.2 shipped inertialization as the default blend mode and wrote
/// down what it could not check: *"No **subprocess** arm runs an inertialized
/// transition: `pie_skinned`'s real `--pie` binary walks its machine on
/// zero-duration transitions, so the decay is exercised across the two hosts only
/// in-process."* One number closes it. A quarter-second fade between two constant
/// poses makes every step of the decay a *different* pose, so the trace the real
/// subprocess folds and the trace the shipping build folds are now compared
/// through several frames of a live inertialized blend rather than through a
/// snap.
///
/// The first transition stays at zero, so a snap is still covered too.
fn machine() -> StateMachine {
    StateMachine {
        states: vec![
            SmState::clip("idle", *IDLE.as_bytes()),
            SmState::clip("wave", *WAVE.as_bytes()),
            SmState::clip("salute", *SALUTE.as_bytes()),
        ],
        transitions: vec![
            SmTransition::new(0, 1, 0.0),
            SmTransition::new(1, 2, INERTIAL_FADE_S),
        ],
        entry: 0,
        ..Default::default()
    }
}

/// The second transition's duration, seconds — long enough that a 60 Hz trace of
/// eight steps lands several frames inside it.
const INERTIAL_FADE_S: f64 = 0.25;

/// The asset bytes the editor would resolve out of a project's content root.
fn anim_bytes() -> BTreeMap<Uuid, Vec<u8>> {
    BTreeMap::from([
        (SKEL, inf_asset::encode(&rig()).unwrap()),
        (
            SM,
            inf_asset::encode(&StateMachineAsset::new(machine(), Some(*SKEL.as_bytes()))).unwrap(),
        ),
        (
            IDLE,
            inf_asset::encode(&AnimClipAsset::new(hold(0.0), Some(*SKEL.as_bytes()))).unwrap(),
        ),
        (
            WAVE,
            inf_asset::encode(&AnimClipAsset::new(hold(60.0), Some(*SKEL.as_bytes()))).unwrap(),
        ),
        (
            SALUTE,
            inf_asset::encode(&AnimClipAsset::new(hold(-25.0), Some(*SKEL.as_bytes()))).unwrap(),
        ),
    ])
}

/// A one-character document, streamed exactly as the editor streams it.
fn character_payload(windowed: bool) -> ScenePayload {
    let mut doc = SceneDoc::new();
    let e = doc.create_with_guid(HERO, SpawnKind::Empty, "Hero", None);
    doc.world_mut().world_mut().entity_mut(e).insert((
        SkeletalMesh {
            mesh: Some(MESH),
            skeleton: Some(SKEL),
        },
        AnimStateMachine {
            sm: Some(SM),
            ..Default::default()
        },
    ));
    doc.world_mut().mark_dirty();
    doc.world_mut().propagate();

    let anim = anim_bytes();
    let mesh_bytes = inf_asset::encode(&body()).unwrap();
    build_scene_payload(
        &doc,
        |_| None,
        |_| None,
        |g| anim.get(&g).cloned(),
        |_| None,
        |_| None,
        |_| None,
        |g| (g == MESH).then(|| mesh_bytes.clone()),
        // P26.3b: the cloth / hair / material / texture byte resolver.
        |_| None,
        |_| None,
        0, // tick-hz 0: no per-frame sleep (step-driven determinism)
        windowed,
    )
    .expect("build scene payload")
}

// ── the gates ───────────────────────────────────────────────────────────────

/// **The payload carries a character.** The exact-count guard the P21.4 lesson
/// asks for, asserted at the seam that produces the payload, so nothing below can
/// be two empty maps agreeing.
#[test]
fn the_payload_carries_the_characters_mesh_skeleton_machine_and_clips() {
    let payload = character_payload(false);
    assert_eq!(
        payload.meshes.len(),
        1,
        "the payload carries no `.inf_mesh` — every character in a windowed PIE \
         session would draw as a placeholder cube (this is the P24.1 defect)"
    );
    assert_eq!(payload.meshes[0].0, MESH);
    assert_eq!(payload.skeletons.len(), 1);
    assert_eq!(payload.machines.len(), 1);
    // Both clips, and NEITHER is named by a component: they live inside the
    // machine's own payload and reach the wire only through the transitive walk.
    let clip_ids: Vec<Uuid> = payload.clips.iter().map(|(g, _)| *g).collect();
    assert!(clip_ids.contains(&IDLE), "{clip_ids:?}");
    assert!(clip_ids.contains(&WAVE), "{clip_ids:?}");
    assert!(clip_ids.contains(&SALUTE), "{clip_ids:?}");
    assert_eq!(clip_ids.len(), 3, "{clip_ids:?}");
    assert_eq!(
        payload.schema_version,
        inf_runtime::pie::SCENE_PAYLOAD_VERSION
    );
}

/// **A PIE session draws non-cube geometry.** The registry `run_pie` is now
/// handed resolves the payload's bytes to real bind-space geometry plus a
/// palette — the structural form of "the character is not a placeholder".
///
/// The negative control is the store PIE used to get: `SkinnedRegistry::new()`
/// resolves the same binding to `None`, which is precisely how the projector fell
/// through to its slate cube.
#[test]
fn the_pie_registry_resolves_real_skinned_geometry() {
    let payload = character_payload(true);
    assert_eq!(payload.meshes.len(), 1, "nothing to resolve");

    let store = SkinnedRegistry::from_payload(
        &payload.meshes,
        &payload.skeletons,
        &payload.clips,
        &payload.machines,
    );
    let sm = SkeletalMesh {
        mesh: Some(MESH),
        skeleton: Some(SKEL),
    };
    let draw = store
        .resolve_skinned(&sm, None, None, None)
        .expect("the PIE payload's character resolves to real skinned geometry");
    assert_eq!(draw.mesh.vertices.len(), 3, "the bind-space stream is real");
    assert_eq!(draw.mesh.indices, vec![0, 1, 2]);
    assert_eq!(draw.palette.len(), 2, "one matrix per joint");
    assert_eq!(draw.key, (MESH, SKEL));
    assert_eq!(store.loaded_skinned(), 1);
    assert!(store.has_content());

    // The negative control — the store a windowed PIE session used to be handed.
    assert!(
        SkinnedRegistry::new()
            .resolve_skinned(&sm, None, None, None)
            .is_none(),
        "the inert store must still miss, or this test proves nothing about the \
         one that was threaded in"
    );
}

/// **The machine's pose reaches the drawn palette, through the payload.** Build
/// the sim the PIE subprocess builds, step it, and resolve the same character
/// through the same store: the palette must follow the machine out of its entry
/// state.
#[test]
fn the_payload_built_sim_poses_the_character_it_ships() {
    let payload = character_payload(false);
    let mut sim = inf_player::sim_from_payload(&payload)
        .expect("the payload builds a sim")
        .sim;
    let store = SkinnedRegistry::from_payload(
        &payload.meshes,
        &payload.skeletons,
        &payload.clips,
        &payload.machines,
    );
    let sm = SkeletalMesh {
        mesh: Some(MESH),
        skeleton: Some(SKEL),
    };
    let rest = store
        .resolve_skinned(&sm, None, None, None)
        .unwrap()
        .palette;

    sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
    let posed = inf_ecs::pose::evaluated_pose(sim.world(), HERO).expect(
        "the payload-built sim published a pose — if it did not, the \
                 skeleton or the machine's clips never crossed the wire",
    );
    let drawn = store
        .resolve_skinned(&sm, None, Some(posed), None)
        .unwrap()
        .palette;
    assert_ne!(
        rest[1].to_cols_array(),
        drawn[1].to_cols_array(),
        "the character shipped by the payload still draws its rest pose"
    );
}

/// **PIE == shipping, with a character in it.** The real `--pie` subprocess,
/// driven step by step, must fold the same per-step `state_hash` the in-process
/// shipping build of the same payload does — and since P24.1 that hash carries the
/// evaluated pose, so this arm is now a comparison of *what is drawn* as well as
/// of what is simulated.
#[test]
fn pie_subprocess_trace_matches_shipping_with_a_skinned_character() {
    let payload = character_payload(false);
    // Anti-vacuity FIRST: a payload with no character would make both sides agree
    // about a world containing nothing to pose.
    assert_eq!(payload.meshes.len(), 1);
    assert_eq!(payload.skeletons.len(), 1);
    assert_eq!(payload.machines.len(), 1);

    const N: u32 = 8;
    let mut session = PieSession::spawn_scene(&player_bin(), &payload).expect("scene session");
    session.step(N).expect("step N");
    let mut got = Vec::with_capacity(N as usize);
    for _ in 0..N {
        let ev = session
            .wait_for(Duration::from_secs(10), |e| {
                matches!(e, PlayerToEditor::Frame { .. })
            })
            .expect("a frame per step");
        if let PlayerToEditor::Frame { state_hash, .. } = ev {
            got.push(state_hash);
        }
    }
    let want = inf_player::scene_trace(&payload, N as u64).expect("shipping trace");
    assert_eq!(
        got, want,
        "the streamed PIE trace must equal the shipping trace for a level with a \
         machine-driven character"
    );
    // …and the trace MOVES on the step the machine transitions, which is what
    // makes the equality above a statement about the pose rather than about two
    // static worlds.
    assert!(
        got.windows(2).any(|w| w[0] != w[1]),
        "the trace never changed — the machine never left its entry state, so this \
         comparison says nothing about the pose"
    );
    // **P29.2's recorded boundary, closed** (P29.4). Every state holds a CONSTANT
    // pose, so a machine walking them on zero-duration transitions can only ever
    // produce three distinct hashes. A quarter-second inertialized decay produces
    // a different pose on every step it is running, so a count above three is
    // proof that the real `--pie` subprocess and the shipping build agreed
    // through a live decay rather than through a snap — which is exactly what
    // P29.2 wrote down that it could not check.
    let distinct: std::collections::BTreeSet<_> = got.iter().collect();
    assert!(
        distinct.len() > 3,
        "the subprocess trace holds only {} distinct poses over {N} steps - the \
inertialized decay is not crossing the process boundary, and P29.2's boundary \
is open again",
        distinct.len()
    );
    session.stop(Duration::from_secs(5)).expect("graceful stop");
}
/// **The wiring itself, pinned as source text.**
///
/// Everything above proves the payload carries a character and that a store built
/// from it resolves one. None of it can see whether `run_pie_window` actually
/// *hands that store over* — severing `SkinnedRegistry::from_payload(…)` back to
/// `SkinnedRegistry::new()` in `main.rs` fails **zero** tests otherwise, because
/// the skinned store feeds rendering and nothing else: no trace, no state hash, no
/// simulation result changes. That is the whole shape of the P24.1 defect (three
/// phases of placeholder cubes with every gate green), so the wiring needs a gate
/// aimed at the wiring.
///
/// The repo's own idiom for render-only wiring — `runtime_destruct.rs`'s
/// `the_windowed_player_exempts_pie_from_the_tier_budget` reads `window.rs` as
/// source text for exactly this reason. Two halves, and the ABSENCE is the
/// load-bearing one: a host that passed the payload store *and* left the inert
/// constructor in place would satisfy a presence-only check while some other arm
/// of the same match still handed over nothing.
#[test]
fn the_windowed_pie_host_hands_over_the_payloads_skinned_store() {
    // Normalized: `core.autocrlf = true` checks `.rs` out CRLF on Windows.
    let src = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/main.rs"),
    )
    .expect("main.rs is readable")
    .replace("\r\n", "\n");

    assert!(
        src.contains("inf_player::skinned::SkinnedRegistry::from_payload("),
        "the windowed PIE host no longer builds its skeletal store FROM THE \
         PAYLOAD — every character in an embedded or new-window PIE session would \
         draw as a placeholder cube again, and no trace comparison in this repo \
         can see it"
    );
    assert!(
        !src.contains("SkinnedRegistry::new()"),
        "the windowed PIE host still constructs an INERT `SkinnedRegistry` — that \
         is the exact call `from_payload` replaced, and a host doing both hands \
         over nothing on whichever path reaches it first"
    );
    // …and all three byte vectors reach it: skeletons and clips crossed the wire
    // from v3 and were never handed to the store, which is why the mesh arriving
    // in v7 alone would still have drawn nothing.
    for field in ["&payload.meshes", "&payload.skeletons", "&payload.clips"] {
        assert!(
            src.contains(field),
            "the windowed PIE host does not pass `{field}` to its skeletal store"
        );
    }
    // A guard on the guard: `run_pie` must still TAKE the store, or the argument
    // above is being handed to a signature that drops it.
    let window = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/window.rs"),
    )
    .expect("window.rs is readable")
    .replace("\r\n", "\n");
    assert!(
        window.contains("skinned: Arc<SkinnedRegistry>,")
            && window.contains("app.skinned = skinned;"),
        "`run_pie` no longer accepts and installs a `SkinnedRegistry`"
    );
}

// ── P24.3: the AUTHORED IkTarget, across the real `--pie` boundary ──────────

/// The same character, plus an **authored** [`IkTarget`] reaching for `at`
/// (world metres).
///
/// Deliberately the same builder as [`character_payload`] with one component
/// added: the whole claim of the arms below is that the authored component
/// crosses the wire *inside the level bytes a `ScenePayload` already carries* —
/// no payload field, no `SCENE_PAYLOAD_VERSION` bump, and above all **no
/// test-only injection hook in the player**, which is the P21.4 law and the
/// reason P24.2 could not build this arm at all.
fn character_payload_with_ik(at: inf_ecs::math::Vec3d) -> ScenePayload {
    let mut doc = SceneDoc::new();
    let e = doc.create_with_guid(HERO, SpawnKind::Empty, "Hero", None);
    doc.world_mut().world_mut().entity_mut(e).insert((
        SkeletalMesh {
            mesh: Some(MESH),
            skeleton: Some(SKEL),
        },
        AnimStateMachine {
            sm: Some(SM),
            ..Default::default()
        },
        inf_ecs::IkTarget {
            goals: vec![inf_ecs::IkGoalRecord {
                // Joints 0 → 1 of the fixture skeleton: the one bone it has.
                chain: vec![0, 1],
                target: at,
                ..Default::default()
            }],
        },
    ));
    doc.world_mut().mark_dirty();
    doc.world_mut().propagate();

    let anim = anim_bytes();
    let mesh_bytes = inf_asset::encode(&body()).unwrap();
    build_scene_payload(
        &doc,
        |_| None,
        |_| None,
        |g| anim.get(&g).cloned(),
        |_| None,
        |_| None,
        |_| None,
        |g| (g == MESH).then(|| mesh_bytes.clone()),
        // P26.3b: the cloth / hair / material / texture byte resolver.
        |_| None,
        |_| None,
        0,
        false,
    )
    .expect("build scene payload")
}

/// **The authored IK target survives the wire.** The guard the P21.4 lesson asks
/// for, at the seam that produces the payload — asserted on the *decoded level
/// bytes*, so nothing below can be two empty worlds agreeing.
#[test]
fn the_payload_carries_the_authored_ik_target_inside_the_level_bytes() {
    let at = inf_ecs::math::Vec3d::new(0.6, 0.6, 0.0);
    let payload = character_payload_with_ik(at);
    // No new payload field, and the envelope version did NOT move.
    assert_eq!(
        payload.schema_version,
        inf_runtime::pie::SCENE_PAYLOAD_VERSION,
        "the authored component must ride the LEVEL bytes; a payload bump here \
         would mean it did not"
    );
    let level = inf_scene::RuntimeLevel::decode(&payload.level_bytes)
        .expect("the payload's level bytes decode");
    assert_eq!(
        level.entities[0].guid, HERO,
        "the payload's level lost the character"
    );
    let t = level.entities[0]
        .ik_target
        .clone()
        .expect("the authored IkTarget crossed the wire");
    assert_eq!(t.goals.len(), 1);
    assert_eq!(t.goals[0].chain, vec![0, 1]);
    assert_eq!(t.goals[0].target, at, "the target moved on the way across");
    assert!(t.goals[0].enabled);
}

/// **THE ARM P24.2 LEDGERED AND COULD NOT BUILD** (M-PIE): the *real* `--pie`
/// subprocess, engaging IK through the authored component, folding a trace
/// byte-identical to the shipping build's.
///
/// P24.2's report claimed a subprocess IK arm and there was none — every
/// `pose_parity` arm compared `SimSession` against `RuntimeSim` inside one
/// process. The ruling then was: **do not build a test-only goal-injection hook
/// into the player** (a boot path the shipped binary does not take is the law
/// P21.4 paid for), and the arm closes naturally once the goal is authored. This
/// is that arm. The player is the real binary, the goal arrives in the `.inf_lvl`
/// bytes, and nothing in `inf-player` knows a test is running.
///
/// Three claims, because equality alone is satisfiable by two dead worlds:
///  1. the streamed PIE trace equals the shipping trace, step for step;
///  2. the trace MOVES between steps (the machine is animating under the IK);
///  3. **moving the authored target moves the trace** — the anti-vacuity arm,
///     which is what says the goal was solved rather than merely carried.
#[test]
fn the_real_pie_subprocess_engages_the_authored_ik_target() {
    const N: u32 = 8;
    let run = |at: inf_ecs::math::Vec3d| -> Vec<u64> {
        let payload = character_payload_with_ik(at);
        let mut session = PieSession::spawn_scene(&player_bin(), &payload).expect("scene session");
        session.step(N).expect("step N");
        let mut got = Vec::with_capacity(N as usize);
        for _ in 0..N {
            let ev = session
                .wait_for(Duration::from_secs(10), |e| {
                    matches!(e, PlayerToEditor::Frame { .. })
                })
                .expect("a frame per step");
            if let PlayerToEditor::Frame { state_hash, .. } = ev {
                got.push(state_hash);
            }
        }
        session.stop(Duration::from_secs(5)).expect("graceful stop");
        // (1) the same payload, run in-process as the shipping build does.
        let want = inf_player::scene_trace(&payload, N as u64).expect("shipping trace");
        assert_eq!(
            got, want,
            "the REAL --pie subprocess folded a different trace from the shipping \
             build of the same payload, with an authored IK goal in the level"
        );
        got
    };

    let near = run(inf_ecs::math::Vec3d::new(0.6, 0.6, 0.0));
    // (2) the world is alive: the machine is animating under the solve.
    assert!(
        near.windows(2).any(|w| w[0] != w[1]),
        "the trace never changed — this comparison says nothing about the pose"
    );
    // (3) ANTI-VACUITY: the goal is SOLVED, not merely carried. A player that
    //     decoded the component and never handed it to `step_pose_evaluation`
    //     passes (1) and (2) and fails here.
    let far = run(inf_ecs::math::Vec3d::new(-0.6, 0.6, 0.0));
    assert_ne!(
        near, far,
        "moving the authored IK target did not move the trace — the component \
         crossed the wire and nothing solved it"
    );
}

/// The `ik.*` node kit's doors, **through the two hosts' shared Ring-0 rule**.
///
/// The kit edits the authored component by goal index, so this is the property
/// both handler arms rest on: an in-range index reports `true` and moves the
/// solve; an out-of-range one reports `false` and changes nothing. Written
/// against `inf_ecs::pose` rather than against either host because that is the
/// one implementation both `simulate.rs` and `runtime_sim.rs` call — testing it
/// twice would be testing the mirror, not the rule.
#[test]
fn the_ik_kits_doors_edit_the_authored_goal_and_refuse_by_value() {
    let mut doc = SceneDoc::new();
    let e = doc.create_with_guid(HERO, SpawnKind::Empty, "Hero", None);
    doc.world_mut().world_mut().entity_mut(e).insert((
        SkeletalMesh {
            mesh: Some(MESH),
            skeleton: Some(SKEL),
        },
        AnimStateMachine {
            sm: Some(SM),
            ..Default::default()
        },
        inf_ecs::IkTarget {
            goals: vec![inf_ecs::IkGoalRecord {
                chain: vec![0, 1],
                ..Default::default()
            }],
        },
    ));
    doc.world_mut().mark_dirty();
    let w = doc.world_mut();

    let at = inf_ecs::math::Vec3d::new(0.6, 0.6, 0.0);
    assert!(inf_ecs::pose::set_authored_goal_target(w, HERO, 0, at));
    assert!(inf_ecs::pose::set_authored_goal_weight(w, HERO, 0, 0.5));
    assert!(inf_ecs::pose::set_authored_goal_enabled(w, HERO, 0, false));
    let t = w.world().get::<inf_ecs::IkTarget>(e).unwrap();
    assert_eq!(t.goals[0].target, at);
    assert_eq!(t.goals[0].weight, 0.5);
    assert!(!t.goals[0].enabled);

    // Refusals are VALUES: a goal index past the end, and an entity with no
    // component at all, both answer `false` and leave the world alone.
    assert!(!inf_ecs::pose::set_authored_goal_target(w, HERO, 9, at));
    assert!(!inf_ecs::pose::set_authored_goal_target(
        w,
        Uuid::from_u128(0xDEAD),
        0,
        at
    ));
    // …and a non-finite weight never reaches `pslerp`.
    assert!(!inf_ecs::pose::set_authored_goal_weight(
        w,
        HERO,
        0,
        f32::NAN
    ));
    assert_eq!(
        w.world().get::<inf_ecs::IkTarget>(e).unwrap().goals[0].weight,
        0.5,
        "a refused weight must not have been written"
    );

    // The two queries answer on a world that has never stepped: nothing reached,
    // no error — which is the conservative reading and not a panic.
    assert!(!inf_ecs::pose::ik_reached(w, HERO));
    assert_eq!(inf_ecs::pose::ik_reach_error(w, HERO), 0.0);
}
