//! **Phase 24 gate — the character pipeline, wizard to PIE** (P24.5).
//!
//! `phase24_gate.rs` measures the cloth-and-hair half of the phase's done-when
//! clause. This file measures the other half — *"a biped and a quadruped
//! generate from templates … and both run under existing state machines"* — from
//! the wizard's Ring-1 door to the real `inf-player --pie` subprocess, in six
//! arms:
//!
//! * **(a) THE DOOR** — `inf_editor_core::character::build_character` writes a
//!   biped's and a quadruped's six assets into a real `AssetProject`, and the
//!   `.inf_sm` it wrote names the three `.inf_anim`s it wrote. Driven through the
//!   Ring-1 function, **not** through UI automation: the panel is a caller, and a
//!   gate that went through it would be measuring React.
//! * **(b) THE PROPORTIONS ARE LOAD-BEARING** — a taller spec produces different
//!   `.inf_skel` **bytes** and a different walk clip. Asserted on payload bytes
//!   rather than on a trace, deliberately: `pose_state_bytes` folds the skeleton
//!   *GUID*, so any two builds differ for free and a trace comparison here would
//!   be vacuous by construction.
//! * **(c) PIE == SHIPPING, TWICE** — the real `--pie` subprocess folds the same
//!   per-frame trace as `inf_player::scene_trace` over the same payload, for the
//!   biped and for the quadruped, and a second subprocess run of the same
//!   payload folds the same trace again.
//! * **(d) THE WORLD IS ALIVE** — the trace moves between steps, and the pose
//!   store is non-empty after one step. Two dead worlds agree perfectly (P21.4,
//!   met again in P24.4), so this runs *before* anything is compared.
//! * **(e) A SEVERED MACHINE FAILS** — the same entity, the same generated
//!   assets, the same actor, with `AnimStateMachine.sm` set to `None`: the trace
//!   must move. This is what says the machine is DRIVING the pose rather than
//!   riding along beside it.
//! * **(f) THE GENERATED THRESHOLDS FIRE ACROSS THE REAL WIRE** — one document,
//!   one set of assets, two actors whose `speed` variable defaults either side of
//!   the generated run threshold. Different traces means the machine the wizard
//!   wrote transitioned, in the shipped binary, on the numbers the wizard derived
//!   from this creature's own legs.
//!
//! # One document, several payloads
//!
//! Every variant below is built from the **same** `SceneDoc`, mutated between
//! `build_scene_payload` calls. `SceneDoc::edit_create_character` mints a fresh
//! entity GUID, and `pose_state_bytes` folds it — so two documents would produce
//! different traces for a reason that has nothing to do with what is being
//! measured. One document keeps every comparison honest.
//!
//! # The fixture is generated
//!
//! Nothing here is committed: the rig, the mannequin, the three clips and the
//! machine all come out of the shipped wizard, so there is no binary that can
//! drift from the code that reads it.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Duration;

use glam::DVec3;
use uuid::Uuid;

use inf_anim::{AnimClipAsset, BodyPlan, SkeletonAsset, StateMachineAsset};
use inf_blueprint::semantics::{BlueprintClass, Variable};
use inf_blueprint::{Lit, Ty};
use inf_ecs::components::{ActorClass, AnimStateMachine};
use inf_editor_core::assets::AssetProject;
use inf_editor_core::character::{build_character, CharacterBuild, CharacterSpec};
use inf_editor_core::pie::{build_scene_payload, PieSession};
use inf_editor_core::scene::SceneDoc;
use inf_runtime::pie::{PlayerToEditor, ScenePayload};

/// Steps every trace here is folded over. Long enough that the machine's 0.15 s
/// cross-fade completes and a transition is visible in the pose, at 60 Hz.
const STEPS: u32 = 24;
const HZ: u32 = 60;

/// The actor class GUID the fixture binds. One class, two variable defaults —
/// see arm (f).
const DRIVER: Uuid = Uuid::from_u128(0x2405_0000_0000_0000_0000_0000_0000_0001);

fn player_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_inf-player"))
}

// ── the wizard's output ─────────────────────────────────────────────────────

/// A character the wizard actually built, plus the asset bytes a `ScenePayload`
/// needs.
struct Made {
    /// Kept alive: the `AssetProject` wrote into it and `bytes` was read from it.
    _tmp: tempfile::TempDir,
    build: CharacterBuild,
    /// Every generated asset's on-disk bytes, keyed by GUID — exactly what a cook
    /// would ship and what `build_scene_payload`'s resolvers hand over.
    bytes: BTreeMap<Uuid, Vec<u8>>,
}

/// Run the wizard's Ring-1 door for `spec` into a fresh project.
fn make(spec: &CharacterSpec) -> Made {
    let tmp = tempfile::tempdir().expect("tempdir");
    let mut project = AssetProject::open(tmp.path().join("Content")).expect("open project");
    let build = build_character(&mut project, spec).expect("the wizard builds a character");
    let mut bytes = BTreeMap::new();
    for id in [
        build.skeleton,
        build.mesh,
        build.idle,
        build.walk,
        build.run,
        build.machine,
    ] {
        let entry = project
            .db()
            .get(id)
            .expect("the wizard registered its asset");
        bytes.insert(id.0, std::fs::read(&entry.path).expect("read asset bytes"));
    }
    Made {
        _tmp: tmp,
        build,
        bytes,
    }
}

fn biped_spec() -> CharacterSpec {
    CharacterSpec {
        name: "Hero".into(),
        plan: BodyPlan::Biped,
        ..CharacterSpec::default()
    }
}

/// The twenty-joint canonical-vocabulary rig, which `BodyPlan::Biped` was until
/// SK1a — kept as its own arm so the rig every committed clip in this repository
/// is index-bound to still has a gate on it.
fn canonical_biped_spec() -> CharacterSpec {
    CharacterSpec {
        name: "Canon".into(),
        plan: BodyPlan::BipedCanonical,
        ..CharacterSpec::default()
    }
}

fn quadruped_spec() -> CharacterSpec {
    CharacterSpec {
        name: "Beast".into(),
        plan: BodyPlan::Quadruped,
        ..CharacterSpec::default()
    }
}

/// An actor class whose only member is `speed`, defaulted to `m_s`.
///
/// **No handler and no IR**: `ActorInstance::new` seeds member variables from
/// their class defaults, and `inf_ecs::pose`'s `vars` resolver reads that map, so
/// a class with one variable and no events is the smallest honest way to hand the
/// generated machine a speed. Writing a `Tick` that set it would test the
/// interpreter, which is P6's job and not this file's.
fn driver(m_s: f64) -> BlueprintClass {
    let mut class = BlueprintClass::new("act:p245_driver", "Locomotion Driver");
    class.variables.push(Variable {
        name: inf_anim::SPEED_VAR.to_string(),
        ty: Ty::Float,
        default: Lit::Float(m_s),
        exposed: true,
    });
    class
}

/// The wizard's actor, in a document, wearing what the wizard wrote.
fn doc_with(made: &Made) -> (SceneDoc, Uuid) {
    let mut doc = SceneDoc::new();
    let guid = doc.edit_create_character(
        "Hero",
        made.build.skeleton.0,
        made.build.mesh.0,
        made.build.machine.0,
        Some(inf_editor_core::scene::doc::CharacterSkin::from_material(
            made.build.material.0,
            &inf_editor_core::character::starter_skin_material(),
        )),
        DVec3::ZERO,
        // The wizard's own controller is REPLACED below by this file's `driver`,
        // whose only job is to default `speed` — see its doc comment. Passing
        // `None` here keeps that substitution honest rather than layering two
        // classes on one entity.
        None,
        1.8,
    );
    let e = doc.world().entity_of(guid).expect("the wizard spawned it");
    // **P29.6: the movement component is taken back off.**
    //
    // The wizard now makes a real character — a capsule, a kinematic body and a
    // `CharacterMovement` — and the movement step publishes that character's own
    // `speed` into its machine every step
    // (`inf_ecs::anim_bridge::publish_character_params`), which SHADOWS an actor
    // variable of the same name. That is the right precedence: a character that
    // has a movement component has one authority for how fast it is going.
    //
    // This file's claim is the P24.5 one — *the machine the wizard wrote fires
    // on the thresholds the wizard derived* — and it drives it with a variable
    // on purpose (see `driver`). So the fixture takes the component away and
    // keeps testing exactly what it always tested; the new precedence has its
    // own arm in `movement_3d`.
    doc.world_mut()
        .world_mut()
        .entity_mut(e)
        .remove::<inf_ecs::components::CharacterMovement>();
    doc.world_mut()
        .world_mut()
        .entity_mut(e)
        .insert(ActorClass(DRIVER));
    doc.world_mut().mark_dirty();
    doc.world_mut().propagate();
    (doc, guid)
}

/// A payload from `doc`, carrying `made`'s assets and an actor moving at `speed`.
fn payload(doc: &SceneDoc, made: &Made, speed: f64) -> ScenePayload {
    let bytes = &made.bytes;
    build_scene_payload(
        doc,
        |g| (g == DRIVER).then(|| driver(speed)),
        |_| None,
        |g| bytes.get(&g).cloned(),
        |_| None,
        |_| None,
        |_| None,
        |g| bytes.get(&g).cloned(),
        // P26.3b: the cloth / hair / material / texture byte resolver.
        |_| None,
        |_| None,
        HZ,
        false,
    )
    .expect("build scene payload")
}

/// Run the real `inf-player --pie` subprocess on `payload` and return its
/// streamed per-frame state hashes.
fn pie_trace(payload: &ScenePayload) -> Vec<u64> {
    let mut session = PieSession::spawn_scene(&player_bin(), payload).expect("scene session");
    session.step(STEPS).expect("step the session");
    let mut got = Vec::with_capacity(STEPS as usize);
    for _ in 0..STEPS {
        let ev = session
            .wait_for(Duration::from_secs(20), |e| {
                matches!(e, PlayerToEditor::Frame { .. })
            })
            .expect("a frame per step");
        if let PlayerToEditor::Frame { state_hash, .. } = ev {
            got.push(state_hash);
        }
    }
    session
        .stop(Duration::from_secs(10))
        .expect("graceful stop");
    got
}

/// **(d) THE WORLD IS ALIVE**, asserted before any two traces are compared.
///
/// The P21.4 rule, met again by P24.4's own mutation matrix: a hash comparison
/// between two hosts says they run the same world; it cannot say the world has a
/// character in it. So this steps the reference once and asserts the pose store
/// is non-empty and that the joint count is the rig's, before returning the
/// shipping trace everything else is measured against.
fn shipping_trace(payload: &ScenePayload, joints: usize) -> Vec<u64> {
    let mut sim = inf_player::sim_from_payload(payload)
        .expect("the payload builds a sim")
        .sim;
    sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
    let pose = inf_ecs::pose::pose_state_bytes(sim.world());
    assert!(
        !pose.is_empty(),
        "the generated character posed NOTHING — every comparison below would \
         be two empty worlds agreeing"
    );
    // 16 bytes of entity guid + 16 of skeleton guid + a u32 count + 10 f32 per
    // joint. Checking the count, not just non-emptiness: a rig that lost its
    // joints on the way across the wire still writes a header.
    let counted = u32::from_le_bytes(pose[32..36].try_into().unwrap()) as usize;
    assert_eq!(
        counted, joints,
        "the posed skeleton has {counted} joints and the generated rig has {joints}"
    );
    // **The trace, re-priced** (SK1a). The layout above is the price: a posed
    // character costs 36 bytes of header plus 40 per joint per step, so the
    // default biped went from 836 B to 6 476 B when it became the mannequin —
    // 7.75x, not the 8x a bone count alone would suggest, because the header does
    // not grow. What is *streamed* between the two hosts is a `u64` hash per step
    // and does not grow at all; what grows is the memcpy and the hash over it, per
    // posed character per fixed step. Asserted rather than described so the number
    // cannot drift out of the ledger.
    assert_eq!(
        pose.len(),
        16 + 16 + 4 + joints * 40,
        "the pose trace is not the shape its own price is quoted from"
    );
    println!(
        "pose trace: {joints} joints = {} B per character per step",
        pose.len()
    );

    let trace = inf_player::scene_trace(payload, STEPS as u64).expect("shipping trace");
    assert!(
        trace.windows(2).any(|w| w[0] != w[1]),
        "the trace never changed across {STEPS} steps — the character is standing \
         in a bind pose and every comparison below says nothing about animation"
    );
    trace
}

// ── (a) THE DOOR ────────────────────────────────────────────────────────────

/// **A biped AND a quadruped generate from templates**, through the wizard's own
/// Ring-1 door, with the machine naming the clips that were written beside it.
#[test]
fn the_wizard_generates_a_biped_and_a_quadruped_wired_to_their_clips() {
    for spec in [biped_spec(), quadruped_spec()] {
        let made = make(&spec);
        let skel: SkeletonAsset =
            inf_asset::decode(&made.bytes[&made.build.skeleton.0]).expect("the rig decodes");
        let sm: StateMachineAsset =
            inf_asset::decode(&made.bytes[&made.build.machine.0]).expect("the machine decodes");

        assert_eq!(
            sm.skeleton,
            Some(*made.build.skeleton.0.as_bytes()),
            "the generated machine is not bound to the generated rig"
        );
        assert_eq!(sm.machine.states.len(), 3);
        assert_eq!(sm.machine.entry, 0);
        let played: Vec<Uuid> = sm
            .machine
            .states
            .iter()
            .map(|s| match &s.motion {
                inf_anim::Motion::Clip(c) => Uuid::from_bytes(*c),
                other => panic!("a generated state plays {other:?}"),
            })
            .collect();
        assert_eq!(
            played,
            vec![made.build.idle.0, made.build.walk.0, made.build.run.0],
            "the machine does not name the three clips the wizard wrote"
        );

        // Every clip drives joints this rig actually has, and drives more than
        // the root — a walk cycle addressing only joint 0 would be a bob.
        for id in [made.build.idle.0, made.build.walk.0, made.build.run.0] {
            let clip: AnimClipAsset = inf_asset::decode(&made.bytes[&id]).expect("clip decodes");
            assert!(clip.clip.duration > 0.0);
            assert!(
                clip.clip
                    .tracks
                    .iter()
                    .all(|t| (t.joint as usize) < skel.skeleton.len()),
                "{} addresses a joint the rig does not have",
                clip.clip.name
            );
        }
        let walk: AnimClipAsset =
            inf_asset::decode(&made.bytes[&made.build.walk.0]).expect("walk decodes");
        // The hips, plus an upper and a lower bone per leg. Spelled as a
        // `floor` so an ARM track (which only the biped has) is not a failure.
        let expected_legs = spec.plan.legs() as usize;
        let floor = 1 + expected_legs * 2;
        assert!(
            walk.clip.tracks.len() >= floor,
            "a {:?} walk drives {} tracks; {expected_legs} legs need {floor} at least",
            spec.plan,
            walk.clip.tracks.len(),
        );
    }
}

// ── (b) THE PROPORTIONS ARE LOAD-BEARING ────────────────────────────────────

/// **Shaping the body changes what is written.**
///
/// On the *bytes*, not on a trace: `pose_state_bytes` folds the skeleton GUID, so
/// two separately-built characters differ in their trace for free. The claim
/// worth making is that the payload the author gets is a function of the
/// proportions they set — including the ANIMATION, which is the whole reason the
/// locomotion set is generated rather than shipped.
#[test]
fn a_changed_proportion_changes_the_generated_rig_and_its_walk() {
    let short = make(&biped_spec());
    let tall = make(&CharacterSpec {
        params: inf_anim::BodyParams {
            height_m: 2.6,
            ..inf_anim::BodyParams::default()
        },
        ..biped_spec()
    });

    let a: SkeletonAsset = inf_asset::decode(&short.bytes[&short.build.skeleton.0]).unwrap();
    let b: SkeletonAsset = inf_asset::decode(&tall.bytes[&tall.build.skeleton.0]).unwrap();
    assert_eq!(
        a.skeleton.len(),
        b.skeleton.len(),
        "only the height changed; the joint COUNT must not"
    );
    // **The hips, by NAME and not by index** (SK1a). Joint 0 of the mannequin is
    // `root`, which sits at the origin at every height by construction, so an
    // index-0 comparison here reads two identical zeros and calls the generator
    // broken. The girdle is what carries `hip_height_ratio` on both rigs.
    let hips = |sk: &SkeletonAsset| {
        let i = sk
            .role_index()
            .first(inf_anim::BoneRoleKind::Pelvis, inf_anim::BoneSide::Center)
            .or_else(|| sk.skeleton.index_of("hips"))
            .expect("a biped has a hip girdle") as usize;
        sk.skeleton.joints()[i].local_bind
    };
    assert_ne!(
        hips(&a),
        hips(&b),
        "a 1.75 m and a 2.6 m biped have the same hips — the proportion never \
         reached the generator"
    );

    let aw: AnimClipAsset = inf_asset::decode(&short.bytes[&short.build.walk.0]).unwrap();
    let bw: AnimClipAsset = inf_asset::decode(&tall.bytes[&tall.build.walk.0]).unwrap();
    assert_ne!(
        aw.clip, bw.clip,
        "the two walks are identical — the locomotion set is not derived from the \
         rig, which is the only thing that makes it a DEFAULT rather than a stock \
         clip that fits one skeleton"
    );

    // …and the same spec twice produces the same content, so the difference above
    // is the proportion and not the wind.
    //
    // The RIG is compared as bytes — a `SkeletonAsset` carries no identity, so
    // two generations of one spec are byte-identical. A CLIP is compared as its
    // decoded `AnimClip`, because `AnimClipAsset` wraps the *skeleton's GUID*,
    // which `AssetProject::write_asset` mints fresh on every build; comparing
    // those bytes would be comparing two `AssetId::new()`s and would fail for a
    // reason that has nothing to do with the generator.
    let again = make(&biped_spec());
    assert_eq!(
        short.bytes[&short.build.skeleton.0], again.bytes[&again.build.skeleton.0],
        "two builds of one spec produced different rigs"
    );
    let a2: AnimClipAsset = inf_asset::decode(&again.bytes[&again.build.walk.0]).unwrap();
    assert_eq!(
        aw.clip, a2.clip,
        "two builds of one spec produced different walks"
    );
}

// ── (c) PIE == SHIPPING, (d) alive, (e) severed ─────────────────────────────

/// **THE GATE**: the wizard's biped, in the real `--pie` subprocess, folding the
/// shipping build's trace — twice — and a severed machine that does not.
#[test]
fn the_real_pie_subprocess_runs_the_generated_biped() {
    let made = make(&biped_spec());
    let joints = {
        let skel: SkeletonAsset = inf_asset::decode(&made.bytes[&made.build.skeleton.0]).unwrap();
        // **The default biped is the 161-bone mannequin** (SK1a), so this gate is
        // the wave's PIE == shipping arm on a mannequin pose trace: a rig with
        // twist chains and IK handles that the drive pass writes every step, and a
        // locomotion set derived from `thigh_l` / `calf_l` rather than from a name
        // this engine invented. Pinned, so a plan that quietly reverted would fail
        // here rather than pass a smaller gate.
        assert_eq!(skel.skeleton.len(), inf_anim::MANNY_JOINT_COUNT);
        assert_eq!(
            skel.roles.len(),
            inf_anim::MANNY_JOINT_COUNT,
            "the roles shipped"
        );
        assert_eq!(skel.twists.len(), 16, "the twist drivers shipped");
        assert_eq!(skel.ik_follow.len(), 5, "the IK follow table shipped");
        skel.skeleton.len()
    };
    let (mut doc, guid) = doc_with(&made);
    let live = payload(&doc, &made, 0.0);

    // (d) then (c): the world is alive, and the subprocess agrees with shipping.
    let want = shipping_trace(&live, joints);
    let got = pie_trace(&live);
    assert_eq!(
        got, want,
        "the REAL --pie subprocess folded a different trace from the shipping \
         build over the SAME payload, for a character the wizard generated"
    );
    // …and again, from a second process: the trace is a property of the payload.
    assert_eq!(
        pie_trace(&live),
        want,
        "two --pie runs of one payload folded different traces"
    );

    // (e) ANTI-VACUITY: sever the machine. Same entity, same rig, same body, same
    // actor — only `AnimStateMachine.sm` goes away. A player that carried the
    // component and never handed it to `step_pose_evaluation` passes everything
    // above and fails here.
    let e = doc.world().entity_of(guid).expect("the actor");
    doc.world_mut()
        .world_mut()
        .entity_mut(e)
        .insert(AnimStateMachine {
            sm: None,
            ..Default::default()
        });
    doc.world_mut().mark_dirty();
    doc.world_mut().propagate();
    let severed = payload(&doc, &made, 0.0);
    assert_ne!(
        pie_trace(&severed),
        want,
        "unbinding the generated state machine changed nothing — the pose is not \
         coming from the machine the wizard wrote"
    );
}

/// The same gate for a **quadruped**, which is the other half of the phase's
/// done-when clause and a different code path in the generator (two girdles, no
/// arms, four legs a quarter cycle apart).
#[test]
fn the_real_pie_subprocess_runs_the_generated_quadruped() {
    let made = make(&quadruped_spec());
    let joints = {
        let skel: SkeletonAsset = inf_asset::decode(&made.bytes[&made.build.skeleton.0]).unwrap();
        skel.skeleton.len()
    };
    let (doc, _) = doc_with(&made);
    let live = payload(&doc, &made, 0.0);
    let want = shipping_trace(&live, joints);
    assert_eq!(
        pie_trace(&live),
        want,
        "the REAL --pie subprocess folded a different trace from the shipping \
         build over the SAME payload, for a quadruped the wizard generated"
    );
}

// ── (f) THE GENERATED THRESHOLDS FIRE ACROSS THE REAL WIRE ──────────────────

/// **The machine the wizard wrote transitions in the shipped binary, on the
/// numbers the wizard derived from this creature's legs.**
///
/// One document and one set of assets, so the entity GUID, the skeleton GUID and
/// every clip are identical between the two runs; the *only* difference is the
/// `speed` variable the actor's class defaults to — below the walk threshold on
/// one side, above the run threshold on the other. Different traces means the
/// conditions were evaluated and the states were entered.
///
/// The thresholds are read back off the generated machine rather than assumed,
/// because they are a function of the rig's leg length and hard-coding one here
/// would make this test agree with a stale constant instead of with the wizard.
#[test]
fn the_generated_thresholds_move_the_shipped_player_between_states() {
    let made = make(&biped_spec());
    let sm: StateMachineAsset = inf_asset::decode(&made.bytes[&made.build.machine.0]).unwrap();
    // The run edge is `walk → run` on a `>`; its value is the run threshold.
    let run_at = sm
        .machine
        .transitions
        .iter()
        .find(|t| t.from == inf_anim::SmSource::State(1) && t.to == 2)
        .map(|t| {
            // v2: the condition is a tree, and the generator's edges are the flat
            // float compares v1 authored -- `as_flat_and` is the door that says so.
            t.condition
                .as_flat_and()
                .expect("the generated edges stay flat float compares")[0]
                .value
                .as_f64()
        })
        .expect("the generated machine has a walk→run edge");
    assert!(
        run_at > 0.0,
        "a run threshold of {run_at} is not a threshold"
    );

    let (doc, _) = doc_with(&made);
    let standing = payload(&doc, &made, 0.0);
    let sprinting = payload(&doc, &made, run_at * 2.0);

    let still = pie_trace(&standing);
    let moving = pie_trace(&sprinting);
    assert_ne!(
        still,
        moving,
        "the shipped player folded the same trace at 0 m/s and at {} m/s — the \
         generated machine's `{}` conditions never fired",
        run_at * 2.0,
        inf_anim::SPEED_VAR
    );

    // …and the state it lands in is `run`, read off the reference world rather
    // than inferred from the hashes differing (which a machine that merely
    // wobbled would also satisfy).
    let mut sim = inf_player::sim_from_payload(&sprinting).unwrap().sim;
    for _ in 0..STEPS {
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
    }
    let state = sim
        .world()
        .world()
        .iter_entities()
        .find_map(|e| e.get::<AnimStateMachine>().map(|m| m.runtime.current))
        .expect("the character carries a machine");
    assert_eq!(
        state,
        2,
        "at twice the run threshold the character is in state {} ({}), not `run`",
        state,
        inf_anim::STATE_NAMES.get(state).copied().unwrap_or("?")
    );
}

/// **One arm of** the gate above, for the **canonical-vocabulary biped** — the
/// twenty-joint rig `BodyPlan::Biped` was before SK1a.
///
/// Not redundant with the mannequin arm: it is the rig every committed
/// `.inf_anim` in this repository is index-bound to, it carries **no** side
/// tables, and it is therefore the arm that proves the drive pass's "absent costs
/// nothing" claim end to end — a payload with no twists and no handles folds a
/// trace the shipping build folds too.
///
/// **What it does not do** (SK1a audit; the first write-up said "the same gate",
/// which it is not): it runs the PIE-versus-shipping comparison **once**, from
/// one subprocess. It does not repeat it from a second process and it carries no
/// severed-machine `assert_ne!`. Both of those live on the mannequin arm, where
/// they guard the same two hazards for both rigs — a per-process trace and a
/// vacuous comparison are properties of the harness, not of the bone count.
#[test]
fn the_real_pie_subprocess_runs_the_canonical_biped_with_no_side_tables() {
    let made = make(&canonical_biped_spec());
    let joints = {
        let skel: SkeletonAsset = inf_asset::decode(&made.bytes[&made.build.skeleton.0]).unwrap();
        assert_eq!(skel.skeleton.len(), 20);
        assert!(
            skel.roles.is_empty() && skel.twists.is_empty() && skel.ik_follow.is_empty(),
            "this rig is supposed to carry no side tables, and the claim above \
             about what a rig without them costs rests on it"
        );
        skel.skeleton.len()
    };
    let (doc, _) = doc_with(&made);
    let live = payload(&doc, &made, 0.0);
    let want = shipping_trace(&live, joints);
    assert_eq!(
        pie_trace(&live),
        want,
        "the REAL --pie subprocess folded a different trace from the shipping \
         build over a rig with no drive tables at all"
    );
}
