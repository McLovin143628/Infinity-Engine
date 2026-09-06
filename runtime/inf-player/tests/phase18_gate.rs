//! **THE PHASE 18 GATE** (P18.5): Lumen-class GI + Nanite completion, composed.
//!
//! The `phase18-scatter` sample puts every piece of the phase in one level — a
//! grid of **standing meshlet slabs** streamed out of an mmap'd `.inf_vmesh` under
//! a VRAM budget and culled by two-pass HZB occlusion (P18.1/P18.2), a `Terrain`
//! and a `TimeOfDay` clock feeding **GI v2** with an amortized probe sweep
//! (P18.4), and **102 400 GPU-scattered instances** with LOD cross-fade (P18.5).
//! This file is the gate over it.
//!
//! Each batch of Phase 18 already has its own suite — `vgeom_gate::gate_b2`/`b3`,
//! `inf-render`'s `gi_v2.rs` and `scatter.rs`. What none of them can say is that
//! the four features still hold **together**: every one of them is a *subtractive*
//! or *budgeted* optimisation, and the classic way such a thing breaks is that two
//! of them are individually conservative and jointly are not. So the claims here
//! are composition claims, and the sub-features are exercised only as far as
//! composition can move them.
//!
//! The six gates:
//!
//! * **(a) the composed frame trace is deterministic** — pixels, meshlet
//!   residency, the GI audit and the scatter audit reproduce byte for byte across
//!   two fresh renderers driven over the same camera path. Floats are traced as
//!   **bits**, never with a tolerance (`phase17_gate` says why: an `f64`
//!   comparison that "passes within 1e-12" is exactly the drift a gate exists to
//!   catch).
//! * **(b) cooked == uncooked** — the cooked pack and the editor's PIE payload
//!   build the same world and project the same scene, step for step, with the
//!   clock running.
//! * **(c) the 10.6M-triangle vgeom gate survives composition** — the flagship
//!   scene still exceeds 10.6M source triangles, and occlusion-on is still
//!   byte-identical to occlusion-off **with GI, scatter and a bound streaming
//!   budget all on**. P18.1's subtractive proof is not allowed to be a
//!   single-feature result.
//! * **(d) the instance-cull counters are real** — frustum / occluded / distance /
//!   mesh / impostor are all nonzero on the composed frame and reproduce exactly,
//!   so the scatter is genuinely being culled rather than trivially all-visible
//!   (which would satisfy every pixel comparison in the suite).
//! * **(e) the composed frame is inside the frame budget**, with the per-system
//!   costs measured and printed.
//! * **(f) the golden inventory is exactly the 41 committed PNGs** — nothing is
//!   re-blessed here; this pins the *set*.
//!
//! …plus the **scene-level editor-parity claim**: the PIE payload the editor
//! builds and the cooked pack produce the same `RenderScene` shape for vgeom,
//! scatter and terrain. (The field-for-field source mirror between
//! `inf_viewport::host` and `inf_player::render` is already pinned by
//! `inf-editor-core`'s `tests/projector_mirror.rs` — `project_sky`
//! character-for-character, the `VgeomInstance` literal field-for-field, and both
//! hosts' opt-in to the meshlet path. Duplicating that here would add nothing;
//! what it cannot say is that the two *scenes* agree, which is what this asserts.)
//!
//! Every GPU test skips cleanly with no adapter, like every GPU path in this repo.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use glam::{DVec3, Quat, Vec3};

use inf_editor_core::samples::{
    phase18_height, phase18_world_center, vgeom_demo_scene, vgeom_demo_source_triangles,
    PHASE18_PCG_ASSET_GUID, PHASE18_PCG_GUID, PHASE18_SCATTER_DRAW_DISTANCE,
    PHASE18_SCATTER_INSTANCES, PHASE18_TERRAIN_GUID, VGEOM_DEMO_GRID, VGEOM_DEMO_MESH_GUID,
};
use inf_math::FloatingOrigin;
use inf_packager::{cook, CookOptions};
use inf_player::level::{self, LevelSource, PackLevelSource};
use inf_player::render::{project_scene, project_scene_with_skinned};
use inf_player::runtime_sim::{RuntimeInput, RuntimeSim};
use inf_player::skinned::SkinnedRegistry;
use inf_player::vmesh::{derived_vmesh_id, VmeshRegistry};
use inf_project::ProjectManifest;
use inf_render::{
    EngineRenderer, GiAudit, GiSettings, GpuContext, HeadlessTarget, LightKind, PrimMesh,
    RenderLight, RenderScene, RenderSettings, RenderView, ScatterAudit, ScatterBatch, ScatterData,
    ScatterInstance, VgeomAsset, VgeomAudit, VgeomInstance, VgeomSettings, HEADLESS_FORMAT,
};
use inf_vgeom::test_support::{budget_for_pages, held_pages_for_want};

/// Render target size. Small on purpose: the claims are about counters and byte
/// equality, and a bigger frame buys neither.
const W: u32 = 320;
const H: u32 = 180;

/// Rendered frames per scripted run. Enough for the two-pass occlusion path's
/// temporal early set and the amortized GI probe sweep to have real history —
/// a one-frame comparison would only ever see each subsystem's conservative
/// cold state, which is not the state a shipped game runs in.
const FRAMES: usize = 6;

/// Fixed sim steps between rendered frames. At the sample's 600x clock this is
/// ten minutes of time-of-day per frame, so the sun genuinely sweeps across the
/// run instead of sitting still while a "running clock" is claimed.
const STEPS_PER_FRAME: usize = 60;

/// Hard mean-per-frame budget, in milliseconds — a **mirror** of the §8 tripwire
/// in `crates/inf-render/tests/frame_budget.rs`, which is a private `const` of
/// that test binary and therefore not importable here.
///
/// **This is deliberately NOT a new ratchet.** §8's rule makes every budget
/// constant a standing maintenance obligation, and P17.4 (`sky_stack_cost_per_
/// tier`), P18.1 (`vgeom_two_pass_cost`) and P18.4 (`gi_v2_cost_per_tier`) all set
/// the precedent of measuring a new subsystem against the existing composed-frame
/// budget rather than minting another one. If `frame_budget.rs` ratchets its
/// number down, this copy follows it down; it may never be raised independently,
/// and it may never be raised at all.
use inf_core::FRAME_BUDGET_MS;

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

fn sample_dir() -> PathBuf {
    workspace_root().join("samples/phase18-scatter")
}

fn vgeom_demo_dir() -> PathBuf {
    workspace_root().join("samples/vgeom-demo")
}

fn gpu_or_skip(what: &str) -> Option<GpuContext> {
    match GpuContext::headless() {
        Ok(g) => Some(g),
        Err(e) => {
            eprintln!("SKIP {what}: no GPU adapter ({e})");
            None
        }
    }
}

// ── the fixture: two sample dirs, one project ────────────────────────────────

/// Scaffold a project holding **both** sample directories' content and cook it.
///
/// The phase18 level's slabs reference `samples/vgeom-demo/Dense.inf_mesh` by
/// GUID rather than shipping a second copy of a 1 MB mesh binary (the sample's
/// README says so), so the fixture has to copy both directories into one
/// `Content` tree. The cook then sees one mesh, derives one `.inf_vmesh` from it
/// through the `MeshRef.asset` dependency-closure edge, and ships both.
fn cook_phase18(tmp: &Path) -> PathBuf {
    let proj = tmp.join("proj");
    ProjectManifest::new("Phase 18 Scatter", "blank-3d")
        .save(&proj)
        .unwrap();
    let content = proj.join("Content");
    std::fs::create_dir_all(&content).unwrap();

    let src = sample_dir();
    for f in [
        "Phase18Scatter.inf_lvl",
        "Phase18Scatter.inf_lvl.toml",
        "GroundCover.inf_pcg",
        "GroundCover.inf_pcg.toml",
    ] {
        std::fs::copy(src.join(f), content.join(f)).unwrap_or_else(|e| panic!("copy {f}: {e}"));
    }
    let vsrc = vgeom_demo_dir();
    for f in ["Dense.inf_mesh", "Dense.inf_mesh.toml"] {
        std::fs::copy(vsrc.join(f), content.join(f)).unwrap_or_else(|e| panic!("copy {f}: {e}"));
    }

    let out = tmp.join("out");
    cook(&proj, &out, &CookOptions::default()).expect("cook succeeds");
    out
}

/// Build the composed world off a cooked pack — the shipping `--pack` path.
fn pack_sim(pack_dir: &Path) -> RuntimeSim {
    let source = PackLevelSource::open(pack_dir).expect("pack opens");
    let built = inf_player::build_world_from_pack(&source).expect("pack level builds");
    inf_player::sim_from_built(built)
}

/// Build the composed world off the **editor's PIE payload** — the committed
/// document plus the `.inf_pcg` bytes its `PcgVolume.graph` ref resolves to,
/// exactly as `terrain_demo::pie_payload_matches_shipping_for_terrain_and_pcg`
/// does. This is the "uncooked" arm of gate (b).
fn pie_sim() -> RuntimeSim {
    let dir = sample_dir();
    let doc = inf_editor_core::scene::serialize::load(&dir.join("Phase18Scatter.inf_lvl"))
        .expect("the committed phase18 document loads");
    let pcg_bytes = std::fs::read(dir.join("GroundCover.inf_pcg")).unwrap();
    let payload = inf_editor_core::pie::build_scene_payload(
        &doc,
        |_| None,
        |guid| (guid == PHASE18_PCG_ASSET_GUID).then(|| pcg_bytes.clone()),
        |_| None,
        |_| None,
        |_| None,
        |_| None,
        // P22.3: no destructible meshes in this fixture.
        |_| None,
        // P26.3b: the cloth / hair / material / texture byte resolver.
        |_| None,
        |_| None,
        60,
        false,
    )
    .expect("payload builds");
    assert_eq!(payload.pcgs.len(), 1, "the referenced .inf_pcg rides along");
    let built = inf_player::build_world_from_payload(&payload).expect("PIE world builds");
    inf_player::sim_from_built(built)
}

/// The runtime meshlet registry sliced out of the cooked pack (the mmap the
/// P18.2 streamer pages from).
fn pack_vmeshes(pack_dir: &Path) -> VmeshRegistry {
    let reader = inf_asset::PackReader::open(&pack_dir.join(inf_player::level::PACK_FILE))
        .expect("open pack for registry");
    let reg = VmeshRegistry::from_pack(Arc::new(reader)).expect("vmesh registry");
    assert!(
        !reg.is_empty(),
        "the cook derived no .inf_vmesh — the slabs' MeshRef.asset edge is broken"
    );
    reg
}

// ── the scripted camera ──────────────────────────────────────────────────────

/// A low traverse of the scatter volume, from outside its south-west corner
/// toward its far side.
///
/// Low and moving is the whole point: a distant overview would put every scatter
/// instance in one distance band and every slab in the frustum, which is exactly
/// the frame in which the cull counters mean nothing. From here the LOD bands,
/// the frustum and the occluder set all move under the camera.
fn gate_camera(step: usize) -> RenderView {
    let c = phase18_world_center();
    let t = step as f64;
    let eye = DVec3::new(c.x - 140.0 + t * 22.0, 0.0, c.z - 140.0 + t * 11.0);
    let eye = DVec3::new(eye.x, phase18_height(eye.x, eye.z) + 5.0, eye.z);
    let target = DVec3::new(
        c.x + 90.0,
        phase18_height(c.x + 90.0, c.z + 60.0),
        c.z + 60.0,
    );
    RenderView {
        origin: FloatingOrigin::new(DVec3::ZERO),
        eye_world: eye,
        forward: (target - eye).as_vec3().normalize(),
        up: Vec3::Y,
        fov_y: 60f32.to_radians(),
        near: 0.05,
        width: W,
        height: H,
        ortho: None,
    }
}

// ── settings ─────────────────────────────────────────────────────────────────

/// The composed Phase 18 configuration: the meshlet path on with two-pass HZB
/// occlusion (the shipping default) under `stream_budget` bytes of VRAM, GI v2 on
/// with an amortized probe sweep, and GPU scatter at its shipping defaults.
fn composed_settings(stream_budget: u64) -> RenderSettings {
    RenderSettings {
        vgeom: VgeomSettings {
            enabled: true,
            stream: inf_vgeom::VgeomStreamBudget {
                budget_bytes: stream_budget,
                ..inf_vgeom::VgeomStreamBudget::default()
            },
            ..VgeomSettings::default()
        },
        gi: GiSettings {
            enabled: true,
            // A deliberately amortized sweep (an eighth of the High probe grid per
            // frame), so the P18.4 round-robin cursor is part of what the
            // determinism gate compares rather than a code path nothing exercises.
            probe_budget: 256,
            ..GiSettings::default()
        },
        ..RenderSettings::default()
    }
}

// ── the streaming budget, and the law it is derived under ────────────────────
//
// **meshopt is not cross-platform.** Bit-identical input meshes produce different
// meshlet DAGs — a different page count, and different page *bytes* — on
// aarch64-apple-darwin than on x86_64. So a budget computed as a fraction of any
// byte total measured on one runner is a platform-dependent premise, and every
// anti-vacuity guard resting on it ("the budget must bind") is a platform-dependent
// claim. `inf-render`'s `tests/vgeom_streaming.rs` was the first trip on this and
// `inf-vgeom`'s reimport test the second; the derivation below is the third.
//
// The rule and the two helpers now live in `inf_vgeom::test_support`, next to the
// fixture generator whose header states the law, so the suites share ONE
// implementation instead of mirroring it: `budget_for_pages(source, n)` cuts a
// budget at the midpoint between "the n coarsest pages fit exactly" and "n + 1
// fits", and `held_pages_for_want(want)` is the page count guaranteed to clamp a
// frame that wants `want`.

/// A streaming budget that **binds on every frame** of the scripted run, sized in
/// pages off the live page directory.
///
/// The premise it needs is the run's own per-frame *want*
/// (`VgeomStreamStats::wanted_pages` — the streamer's `ideal_page_count`, a pure
/// function of the camera and the page directory that does not depend on the budget
/// or on residency history), measured here on an unclamped reference run. Cutting
/// the budget at half the **smallest** want over the run means every frame asks for
/// strictly more pages than the budget can grant, so `budget_clamped` holds on
/// every frame by construction, on any platform, whatever meshopt did to the ladder.
///
/// This replaces `(the last frame's resident_bytes / 2).max(root)`, which was a
/// *byte* fraction of the *last* frame's residency asserted to clamp on *every*
/// frame. The camera flies in toward the slabs, so the want ladder rises across the
/// run: half of the last frame's bytes can still cover an early frame's whole want,
/// and then gate (a)'s "the streaming budget never bound" guard fires — which is
/// exactly what happened on macos-arm64 at `a0e398a`, and never on x86_64, because
/// the two platforms' page bytes differ. The guard was right; its premise was not
/// portable.
fn binding_budget(gpu: &GpuContext, pack: &Path, reg: &VmeshRegistry) -> u64 {
    let mut sim = pack_sim(pack);
    let target = HeadlessTarget::new(gpu, W, H);
    let mut r = EngineRenderer::new(gpu, HEADLESS_FORMAT);
    r.set_settings(composed_settings(inf_vgeom::DEFAULT_VGEOM_BUDGET_BYTES));
    let mut scene = RenderScene::default();
    let mut wants = Vec::with_capacity(FRAMES);
    for step in 0..FRAMES {
        for _ in 0..STEPS_PER_FRAME {
            sim.step_once(RuntimeInput::default());
        }
        project_scene(&mut scene, &sim, 0.0, reg);
        r.render(gpu, &scene, &gate_camera(step), &target.view, (W, H));
        let report = r.vgeom_stream_report();
        assert!(
            !report.stats.budget_clamped,
            "the reference run must be unclamped (frame {step})"
        );
        wants.push(report.stats.wanted_pages);
    }
    assert!(
        r.vgeom_stream_report().stats.resident_bytes > 0,
        "the reference run paged nothing — the meshlet path never activated"
    );
    // `wanted_pages` is a SUM across assets, so it is one asset's want only while
    // the scene lists one asset — which is the composed scene's shape, pinned by
    // `cooked_equals_uncooked_on_the_composed_scene`.
    assert_eq!(
        scene.vgeom_assets.len(),
        1,
        "the composed scene must list exactly one meshlet asset for `wanted_pages` \
         to be that asset's want"
    );
    let source = scene.vgeom_assets[0].source.clone();
    let min_want = *wants.iter().min().expect("FRAMES > 0");
    assert!(
        min_want >= 2,
        "some frame of the scripted run wants only the ROOT page ({wants:?} of {} \
         pages): the root is granted unconditionally, so NO budget can clamp that \
         frame and gate (a)'s streamed-scene claim is unstateable. Move the camera \
         script closer to the slabs — do not weaken the guard",
        source.pages().len()
    );
    let held = held_pages_for_want(min_want);
    let budget = budget_for_pages(&source, held);
    eprintln!(
        "phase18 streaming budget: per-frame wants {wants:?} of {} pages, page \
         directory {:?} B ⇒ hold the {held} coarsest pages ⇒ {budget} B (binds on \
         every frame: every want > {held})",
        source.pages().len(),
        source
            .pages()
            .iter()
            .map(|p| p.resident_bytes())
            .collect::<Vec<_>>(),
    );
    budget
}

// ── the recorded trace ───────────────────────────────────────────────────────

/// One rendered frame of the composed scene, as data.
///
/// Every float in here is stored as **bits**. `phase17_gate` states the rule and
/// the reason: a tolerance-based comparison of a determinism trace passes exactly
/// the drift it exists to catch.
#[derive(Debug, Clone, PartialEq)]
struct Frame {
    /// xxh3 of the RGBA readback — the pixels, byte for byte.
    pixels: u128,
    /// How much of the frame is not background (anti-vacuity: an all-black frame
    /// is trivially reproducible).
    painted: usize,
    /// The level clock, `f64` bits.
    seconds: u64,
    /// The projected sun direction, `f32` bits — the GI key light.
    sun: [u32; 3],
    /// Each projected scatter batch's content key and instance count.
    scatter_batches: Vec<(u128, usize)>,
    /// P18.2 residency: `(resident pages, total pages)` per meshlet asset.
    pages: BTreeMap<u128, (usize, usize)>,
    /// P18.2: the residency floor the cut is clamped to, per asset.
    floor_lod: BTreeMap<u128, u32>,
    resident_bytes: u64,
    budget_clamped: bool,
    /// P18.4.
    gi: GiAudit,
    /// P18.1.
    vgeom: VgeomAudit,
    /// P18.5.
    scatter: ScatterAudit,
}

/// Drive one scripted run on a **fresh** sim and a **fresh** renderer, recording
/// a [`Frame`] per rendered frame.
fn record(
    gpu: &GpuContext,
    pack: &Path,
    reg: &VmeshRegistry,
    settings: RenderSettings,
) -> Vec<Frame> {
    let mut sim = pack_sim(pack);
    let target = HeadlessTarget::new(gpu, W, H);
    let mut r = EngineRenderer::new(gpu, HEADLESS_FORMAT);
    r.set_settings(settings);
    r.set_vgeom_audit(true);
    r.set_scatter_audit(true);

    let mut scene = RenderScene::default();
    let mut out = Vec::with_capacity(FRAMES);
    for step in 0..FRAMES {
        for _ in 0..STEPS_PER_FRAME {
            sim.step_once(RuntimeInput::default());
        }
        project_scene(&mut scene, &sim, 0.0, reg);
        r.render(gpu, &scene, &gate_camera(step), &target.view, (W, H));
        let img = target.read_rgba(gpu).expect("readback");
        let report = r.vgeom_stream_report();
        let sun = scene.sun.unit_direction();
        out.push(Frame {
            pixels: xxhash_rust::xxh3::xxh3_128(&img),
            painted: img.chunks(4).filter(|p| p[0] > 12 || p[1] > 12).count(),
            seconds: sim.time_of_day_seconds().to_bits(),
            sun: [sun.x.to_bits(), sun.y.to_bits(), sun.z.to_bits()],
            scatter_batches: scene
                .scatter
                .iter()
                .map(|b| (b.data.key(), b.data.len()))
                .collect(),
            pages: report.pages.clone(),
            floor_lod: report.floor_lod.clone(),
            resident_bytes: report.stats.resident_bytes,
            budget_clamped: report.stats.budget_clamped,
            gi: r.gi_audit(),
            vgeom: r.vgeom_audit(gpu),
            scatter: r.scatter_audit(gpu),
        });
    }
    out
}

// ── GATE (a) — the composed frame trace is deterministic ─────────────────────

/// **GATE (a).** Two fresh renderers, driven over the same scripted camera path
/// on two freshly-built worlds, produce a byte-identical trace: the pixels, the
/// meshlet residency, the GI audit and the scatter audit.
///
/// If this failed, the shipped game would not be replayable and no golden could
/// be trusted: each of the three subsystems in the trace has its own reason to be
/// non-reproducible — the streamer's residency could depend on *what happened to
/// arrive* rather than on (wants, budget); the scatter compaction could have
/// become an atomic append; GI's amortized probe sweep could have keyed its
/// cursor on something outside the frame. Composed, any one of them takes the
/// whole frame with it.
#[test]
fn the_composed_frame_trace_is_deterministic_across_runs() {
    let Some(gpu) = gpu_or_skip("phase18 gate (a)") else {
        return;
    };
    let tmp = tempfile::tempdir().unwrap();
    let pack = cook_phase18(tmp.path());
    let reg = pack_vmeshes(&pack);
    let budget = binding_budget(&gpu, &pack, &reg);
    let settings = composed_settings(budget);

    let a = record(&gpu, &pack, &reg, settings);
    let b = record(&gpu, &pack, &reg, settings);
    assert_eq!(
        a, b,
        "the composed Phase 18 frame trace is not reproducible"
    );

    // ── anti-vacuity: every layer of the trace has to have actually moved ──
    let first = a.first().unwrap();
    let last = a.last().unwrap();
    eprintln!(
        "gate (a): {FRAMES} frames under a {budget} B meshlet budget\n  \
         first: {} px painted, clock {:.1} s, {:?}\n  \
         last:  {} px painted, clock {:.1} s, {:?}\n  \
         vgeom {:?}\n  gi {:?}\n  scatter {:?}",
        first.painted,
        f64::from_bits(first.seconds),
        first.pages,
        last.painted,
        f64::from_bits(last.seconds),
        last.pages,
        last.vgeom,
        last.gi,
        last.scatter,
    );
    assert!(
        first.painted > 4_000,
        "the composed frame painted almost nothing ({} px) — the trace is vacuous",
        first.painted
    );
    assert_ne!(
        first.pixels, last.pixels,
        "every frame is identical — the camera path and the clock did nothing"
    );
    assert_ne!(first.seconds, last.seconds, "the clock never advanced");
    assert_ne!(
        first.sun, last.sun,
        "the sun never moved — GI's key light is frozen"
    );
    // The budget bound on every frame — printed as evidence rather than merely
    // asserted, because "a streamed scene" is the premise the rest of gate (a)
    // rests on and a reader should be able to see the prefix it held. It is true
    // by construction (see [`binding_budget`]): the budget holds half the smallest
    // want in the run, so every frame asks for pages it cannot have.
    let resident: Vec<(usize, usize)> = a.iter().flat_map(|f| f.pages.values().copied()).collect();
    eprintln!(
        "gate (a): under a {budget} B budget the streamer held {resident:?} \
         (resident, total) pages per frame; clamped {:?}",
        a.iter().map(|f| f.budget_clamped).collect::<Vec<_>>()
    );
    assert!(
        a.iter().all(|f| f.budget_clamped),
        "the streaming budget never bound on every frame (held {resident:?} \
         (resident, total) pages under {budget} B) — gate (a) is not testing a \
         streamed scene"
    );
    assert!(
        resident.iter().all(|(res, total)| *res >= 1 && res < total),
        "the clamped run must hold its root floor and strictly fewer than every \
         page ({resident:?} (resident, total) under {budget} B)"
    );
    assert!(
        a.iter().all(|f| f.vgeom.base_cut > 0),
        "the meshlet path never activated"
    );
    assert!(
        a.iter().all(|f| f.gi.terrain_columns > 0),
        "GI never sampled the ground — the terrain is not reaching the volume"
    );
    assert!(
        a.iter().any(|f| f.gi.candidates > 0 && f.gi.voxelized > 0),
        "GI voxelized no primitive on any frame — only the terrain reached it"
    );
    // The probe update is **sliced** — 256 of the High tier's 2048 per frame. That
    // is all this asserts, and the wording matters: it does NOT claim the sweep
    // *advances*, because in this arm it does not — the clock crosses the sun
    // bucket every frame and legitimately restarts it (see the cursor note below).
    // The advancing-cursor claim lives in the sun-still arm that follows.
    assert!(
        a.iter().all(|f| f.gi.probes_updated == 256),
        "the GI probe update is not sliced: a full High-tier update is 2048 probes, \
         so a 256-probe slice is what proves `probe_budget` reached the pass"
    );
    assert!(
        a.iter()
            .all(|f| f.scatter.candidates as usize == PHASE18_SCATTER_INSTANCES + foliage_count()),
        "the scatter cull did not see every instance: {:?}",
        a.iter().map(|f| f.scatter.candidates).collect::<Vec<_>>()
    );

    // **THE AMORTIZATION SWEEP, AND WHY THIS ARM ONLY PRINTS ITS CURSOR.**
    //
    // The P18.4 defect this gate found is FIXED (2026-08-02). It was: `GiSweepKey`
    // held `scene.version`, and the player's `project_scene` calls
    // `RenderScene::mark_dirty()` unconditionally at the end of every projection —
    // so the version moved every frame whether or not any content did, `GiNode`
    // restarted the sweep, and every probe past the first 256-probe slice was never
    // re-integrated. The version has left the key; the sweep's own wrap-around
    // bounds staleness after a content change instead.
    //
    // The cursor still sits at one slice *in this arm*, and legitimately so. The
    // run advances `STEPS_PER_FRAME` steps of the sample's 600x clock between
    // rendered frames — ten sim-minutes per frame — so the sun sweeps ≈2.5° per
    // frame and crosses `gi::sun_bucket`'s ≈0.50° bucket every single frame. A
    // genuinely moved sun invalidates the probes' integration, and restarting there
    // is the documented doctrine, not a defect. No advancing-cursor assertion can
    // hold here without weakening that doctrine, so the cursor is recorded as an
    // observation.
    //
    // The player-side witness for the fix is the sun-still arm below,
    // `the_amortized_sweep_advances_when_only_the_scene_version_churns`: it freezes
    // the sim (and so the clock, the sun and the content) and leaves `scene.version`
    // as the only moving input — the exact shape of the defect.
    let cursors: Vec<u32> = a.iter().map(|f| f.gi.probe_cursor).collect();
    eprintln!("gate (a): GI probe cursor across the run: {cursors:?}");
}

// ── the P18.4 amortization fix, witnessed from the shipped player ────────────

/// **THE AMORTIZED SWEEP ADVANCES WHEN ONLY `scene.version` CHURNS** — the
/// player-side witness for the P18.4 defect this gate found (fixed 2026-08-02).
///
/// `inf_player::render::project_scene` re-projects unconditionally and ends with
/// `RenderScene::mark_dirty()`, so a shipped build's `scene.version` moves every
/// frame whether or not one triangle did. While that version sat in `GiSweepKey`,
/// `GiNode` restarted the probe sweep every frame and every probe past the first
/// 256-probe slice was never re-integrated — amortization paying a full update's
/// price for one slice of freshness, in the *shipped* build only. The editor hid
/// it, because `sync_from_doc` is document-version-gated and a static document
/// holds its version still: a PIE-vs-shipping divergence in everything but name.
///
/// `inf-render`'s `gi_amortization_survives_the_shipped_players_scene_version_churn`
/// pins the claim on a synthetic scene. This pins it through the **real
/// projector**, on the composed Phase 18 level, with the sun held still: the sim
/// is advanced to the composed run's opening pose and then never stepped again, so
/// the clock, the sun and the content are bit-frozen and the version is the only
/// thing in the frame that moves. That isolation is what makes it a witness — with
/// the sun frozen the documented `gi::sun_bucket` reset cannot be what drives the
/// cursor, so an unfixed key would be caught by this churn *alone*. It is also not
/// a contrived pose: a paused game, a pause menu, a cutscene hold and a
/// static-time-of-day level are all exactly this.
///
/// Gate (a) cannot state it, for the reason its cursor note gives: it advances the
/// clock ten sim-minutes per rendered frame, which crosses the ≈0.50° sun bucket
/// every frame and restarts the sweep there for a legitimate reason.
#[test]
fn the_amortized_sweep_advances_when_only_the_scene_version_churns() {
    let Some(gpu) = gpu_or_skip("phase18 amortization witness") else {
        return;
    };
    let tmp = tempfile::tempdir().unwrap();
    let pack = cook_phase18(tmp.path());
    let reg = pack_vmeshes(&pack);
    // The same composed configuration gate (a) runs under, streaming budget and
    // all — the claim is about the shipped renderer's real settings, not a
    // GI-only one.
    let settings = composed_settings(binding_budget(&gpu, &pack, &reg));

    // One warmup block, so the level clock sits at the same mid-morning the
    // composed run opens at — and then the sim is never stepped again.
    let mut sim = pack_sim(&pack);
    for _ in 0..STEPS_PER_FRAME {
        sim.step_once(RuntimeInput::default());
    }

    let target = HeadlessTarget::new(&gpu, W, H);
    let mut r = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
    r.set_settings(settings);
    // One fixed pose. Camera motion is deliberately NOT a sweep reset (the volume
    // follows the camera every frame, so resetting on it would mean never
    // amortizing at all) — but freezing it leaves the version as the single
    // moving input, which is the point of the arm.
    let view = gate_camera(FRAMES / 2);

    /// Frames of pure churn: two full 8-frame sweeps' worth of evidence, so the
    /// wrap is inside the window rather than at its edge.
    const CHURN_FRAMES: usize = 10;

    let mut scene = RenderScene::default();
    let mut cursors = Vec::with_capacity(CHURN_FRAMES);
    let mut versions = Vec::with_capacity(CHURN_FRAMES);
    let mut suns = Vec::with_capacity(CHURN_FRAMES);
    for _ in 0..CHURN_FRAMES {
        // THE PLAYER'S OWN LOOP: project every frame, step the sim never.
        project_scene(&mut scene, &sim, 0.0, &reg);
        r.render(&gpu, &scene, &view, &target.view, (W, H));
        let _ = target.read_rgba(&gpu).expect("readback");
        cursors.push(r.gi_audit().probe_cursor);
        versions.push(scene.version);
        let s = scene.sun.unit_direction();
        suns.push([s.x.to_bits(), s.y.to_bits(), s.z.to_bits()]);
    }
    eprintln!(
        "sun-still churn over {CHURN_FRAMES} frames:\n  cursors  {cursors:?}\n  \
         versions {versions:?}\n  sun bits {:?} (held)",
        suns[0]
    );

    // ── ANTI-VACUITY: the churn the defect needs really is present ──
    assert!(
        versions.windows(2).all(|w| w[1] > w[0]),
        "`project_scene` did not bump `scene.version` on every projection — the \
         churn this arm exists to survive is not happening, so it proves nothing \
         ({versions:?})"
    );
    // ── ISOLATION: nothing else that legitimately resets the sweep moved ──
    assert!(
        suns.windows(2).all(|w| w[0] == w[1]),
        "the projected sun moved with the sim held still — the sun bucket rather \
         than the version could be what drives the cursor here ({suns:?})"
    );

    // ── the claim: 256 → 512 → … → 1792 → wrap, under pure version churn ──
    assert_eq!(
        cursors[0], 256,
        "the first frame did not take one 256-probe slice ({cursors:?})"
    );
    assert!(
        cursors.iter().any(|&c| c > 256),
        "the cursor never advanced past its first slice — the shipped player's \
         per-frame `scene.version` bump is restarting the sweep ({cursors:?})"
    );
    assert_eq!(
        cursors[7], 0,
        "the 2048-probe sweep did not wrap on the 8th 256-probe frame ({cursors:?})"
    );
    for (i, w) in cursors[..8].windows(2).enumerate() {
        if w[1] != 0 {
            assert_eq!(
                w[1],
                w[0] + 256,
                "slice {i} did not advance by the 256-probe budget ({cursors:?})"
            );
        }
    }
}

/// Instances the sample's painted `Foliage` entity contributes to the scatter
/// candidate set (they ride the same cull as the PCG bulk, in their own batches).
fn foliage_count() -> usize {
    let grid = inf_editor_core::samples::PHASE18_FOLIAGE_GRID as usize;
    grid * grid
}

// ── GATE (b) — cooked == uncooked ────────────────────────────────────────────

/// One projected frame of the composed scene, reduced to bit-exact fingerprints —
/// the CPU-side `RenderScene` DTO, which is the whole input to the passes.
#[derive(Debug, Clone, PartialEq)]
struct Projected {
    seconds: u64,
    sun: [u32; 3],
    lights: usize,
    /// `(asset id, page count)` per meshlet asset listed this frame.
    vgeom_assets: Vec<(u128, usize)>,
    /// Every vgeom instance's asset, world translation bits and colour bits.
    vgeom_instances: Vec<(u128, [u64; 3], [u32; 4])>,
    /// Every scatter batch's content key, instance count, anchor bits and the
    /// authored draw distance that rides on it (P18.5).
    scatter: Vec<(u128, usize, [u64; 3], u64)>,
    /// `(renderer id, tile resolution, tile count)` per projected terrain.
    terrains: Vec<(u64, u32, usize)>,
}

fn project(sim: &RuntimeSim, reg: &VmeshRegistry, scene: &mut RenderScene) -> Projected {
    project_scene(scene, sim, 0.0, reg);
    let sun = scene.sun.unit_direction();
    let bits3 = |v: DVec3| [v.x.to_bits(), v.y.to_bits(), v.z.to_bits()];
    Projected {
        seconds: sim.time_of_day_seconds().to_bits(),
        sun: [sun.x.to_bits(), sun.y.to_bits(), sun.z.to_bits()],
        lights: scene.lights.len(),
        vgeom_assets: scene
            .vgeom_assets
            .iter()
            .map(|a| (a.id, a.source.pages().len()))
            .collect(),
        vgeom_instances: scene
            .vgeom_instances
            .iter()
            .map(|i| {
                (
                    i.asset,
                    bits3(i.translation),
                    [
                        i.color[0].to_bits(),
                        i.color[1].to_bits(),
                        i.color[2].to_bits(),
                        i.color[3].to_bits(),
                    ],
                )
            })
            .collect(),
        scatter: scene
            .scatter
            .iter()
            .map(|b| {
                (
                    b.data.key(),
                    b.data.len(),
                    bits3(b.anchor),
                    b.draw_distance.to_bits(),
                )
            })
            .collect(),
        terrains: scene
            .terrains
            .iter()
            .map(|t| (t.id, t.tile_resolution, t.tiles.len()))
            .collect(),
    }
}

/// Run a sim for the same scripted step schedule the GPU gates use, recording the
/// projected scene each frame.
fn projected_trace(mut sim: RuntimeSim, reg: &VmeshRegistry) -> Vec<Projected> {
    let mut scene = RenderScene::default();
    (0..FRAMES)
        .map(|_| {
            for _ in 0..STEPS_PER_FRAME {
                sim.step_once(RuntimeInput::default());
            }
            project(&sim, reg, &mut scene)
        })
        .collect()
}

/// **GATE (b).** The cooked pack and the editor's PIE payload build the same
/// world and project the same scene, step for step, with the clock running.
///
/// If this failed, a shipped Phase 18 build would light, scatter or stream
/// differently from the preview the designer approved — the failure mode is
/// invisible to every unit test and is found by a player.
#[test]
fn cooked_equals_uncooked_on_the_composed_scene() {
    let tmp = tempfile::tempdir().unwrap();
    let pack = cook_phase18(tmp.path());
    let reg = pack_vmeshes(&pack);

    let ship = projected_trace(pack_sim(&pack), &reg);
    let pie = projected_trace(pie_sim(), &reg);

    assert_eq!(
        pie, ship,
        "the PIE payload's projected scene != the cooked pack's — PIE is not shipping"
    );
    // Named sub-claims, so a failure says WHICH layer drifted rather than dumping
    // six structs carrying 100k instances between them.
    assert!(pie.iter().zip(&ship).all(|(a, b)| a.sun == b.sun));
    assert!(pie
        .iter()
        .zip(&ship)
        .all(|(a, b)| a.vgeom_instances == b.vgeom_instances));
    assert!(pie.iter().zip(&ship).all(|(a, b)| a.scatter == b.scatter));
    assert!(pie.iter().zip(&ship).all(|(a, b)| a.terrains == b.terrains));

    // The comparison is only worth anything if the scene really carries all of
    // Phase 18 on both sides.
    let last = ship.last().unwrap();
    assert_eq!(
        last.vgeom_assets.len(),
        1,
        "the composed scene must list exactly one meshlet asset (the shared dense mesh)"
    );
    assert_eq!(
        last.vgeom_assets[0].0,
        derived_vmesh_id(VGEOM_DEMO_MESH_GUID).as_u128(),
        "the slabs must resolve through the cook-derived vmesh id"
    );
    assert!(
        last.vgeom_assets[0].1 >= 2,
        "a single-page vmesh cannot be streamed"
    );
    let scattered: usize = last.scatter.iter().map(|(_, n, _, _)| n).sum();
    assert!(
        scattered >= 100_000,
        "the composed scene must carry 100k+ scattered instances, got {scattered}"
    );
    assert_eq!(
        last.terrains.len(),
        1,
        "the composed scene must project its terrain"
    );
    assert_ne!(
        ship.first().unwrap().seconds,
        last.seconds,
        "the clock never advanced — the trace compares one frozen frame"
    );
}

// ── the scene-level editor-parity claim ──────────────────────────────────────

/// **Editor parity, at the scene level.** The `RenderScene` the editor's PIE
/// payload produces and the one the cooked pack produces have the same *shape*
/// for the three things Phase 18 draws: meshlet assets + instances, scatter
/// batches, and terrain.
///
/// `inf-editor-core`'s `tests/projector_mirror.rs` already pins the two
/// projectors' **source** against each other — `project_sky` character for
/// character, the `VgeomInstance` literal field for field, the surrounding
/// real-geometry rules, and both hosts' opt-in to the meshlet path. What it
/// cannot pin is that the two hosts, fed the same authored level through two
/// completely different loaders (a zero-copy mmap'd pack vs an in-memory payload
/// with a re-evaluated PCG cache), arrive at the same scene. That is this.
#[test]
fn the_editor_payload_and_the_cooked_pack_agree_on_the_render_scene_shape() {
    let tmp = tempfile::tempdir().unwrap();
    let pack = cook_phase18(tmp.path());
    let reg = pack_vmeshes(&pack);

    let mut ship_scene = RenderScene::default();
    let ship = project(&pack_sim(&pack), &reg, &mut ship_scene);
    let mut pie_scene = RenderScene::default();
    let pie = project(&pie_sim(), &reg, &mut pie_scene);

    assert_eq!(
        pie.vgeom_assets, ship.vgeom_assets,
        "the two hosts list different meshlet assets"
    );
    assert_eq!(
        pie.vgeom_instances, ship.vgeom_instances,
        "the two hosts place the meshlet slabs differently"
    );
    assert_eq!(
        pie.scatter, ship.scatter,
        "the two hosts scatter differently — a content key, an anchor or the \
         authored draw distance drifted"
    );
    assert_eq!(
        pie.terrains, ship.terrains,
        "the two hosts project different terrain"
    );

    // …and the shape is the composed one, not an empty agreement.
    assert_eq!(
        ship.vgeom_instances.len(),
        inf_editor_core::samples::PHASE18_SLAB_GRID * inf_editor_core::samples::PHASE18_SLAB_GRID,
    );
    assert!(
        ship.scatter.len() >= 3,
        "expected the PCG bulk plus one batch per painted foliage kind, got {}",
        ship.scatter.len()
    );
    // The PCG bulk is exactly the count the sample's README states, and the
    // authored draw distance rides on the batch (P18.5's fix for the two hosts
    // disagreeing about it).
    let bulk = ship
        .scatter
        .iter()
        .max_by_key(|(_, n, _, _)| *n)
        .expect("a scatter batch");
    assert_eq!(bulk.1, PHASE18_SCATTER_INSTANCES);
    assert_eq!(f64::from_bits(bulk.3), PHASE18_SCATTER_DRAW_DISTANCE);
}

// ── GATE (c) — the 10.6M-triangle vgeom gate survives composition ────────────

/// Scaffold + cook the **vgeom-demo** project (a mirror of `vgeom_gate.rs`'s
/// helper of the same name — test binaries cannot share code, and the flagship
/// scene is the one thing gate (c) genuinely needs from it).
fn cook_vgeom_demo(tmp: &Path) -> PathBuf {
    let proj = tmp.join("vproj");
    ProjectManifest::new("Vgeom Demo", "blank-3d")
        .save(&proj)
        .unwrap();
    let content = proj.join("Content");
    std::fs::create_dir_all(&content).unwrap();
    let src = vgeom_demo_dir();
    for f in ["VgeomDemo.inf_lvl", "Dense.inf_mesh", "Dense.inf_mesh.toml"] {
        std::fs::copy(src.join(f), content.join(f)).unwrap_or_else(|e| panic!("copy {f}: {e}"));
    }
    let out = tmp.join("vout");
    cook(&proj, &out, &CookOptions::default()).expect("cook succeeds");
    out
}

/// Resolve the cooked flagship pack into `(vmesh asset, placed instances)` —
/// the exact `RenderScene` content the player builds for the demo (mirror of
/// `vgeom_gate::scene_from_pack`).
fn flagship_scene_from_pack(pack_dir: &Path) -> (VgeomAsset, Vec<VgeomInstance>) {
    let source = PackLevelSource::open(pack_dir).expect("pack opens");
    let reader = inf_asset::PackReader::open(&pack_dir.join(level::PACK_FILE))
        .expect("open pack for registry");
    let reg = VmeshRegistry::from_pack(Arc::new(reader)).expect("vmesh registry");
    let (asset_id, vmesh_source) = reg
        .pick(VGEOM_DEMO_MESH_GUID, true)
        .expect("the derived vmesh for the dense mesh is present in the pack");
    let asset = VgeomAsset::new(asset_id, vmesh_source);

    let lvl = inf_scene::RuntimeLevel::decode(&source.level_bytes().unwrap()).expect("decode");
    let mut instances = Vec::new();
    let mut id = 1u32;
    for e in &lvl.entities {
        let Some(mr) = &e.mesh else { continue };
        if mr.asset != Some(VGEOM_DEMO_MESH_GUID) {
            continue;
        }
        let t = e.transform;
        instances.push(VgeomInstance::lit(
            asset_id,
            t.translation.to_dvec3(),
            t.quat().as_quat(),
            t.scale.to_dvec3().as_vec3(),
            [0.6, 0.6, 0.6, 1.0],
            id,
        ));
        id += 1;
    }
    (asset, instances)
}

/// A deterministic `n×n` scatter field on the ground plane, spanning `span`
/// metres. Positions come from an integer hash (not `std` trig — the P14 law) so
/// the field is irregular but reproducible on every platform.
fn scatter_field(anchor: DVec3, n: u32, span: f64, scale: f32, id: u32) -> ScatterBatch {
    let step = span / n as f64;
    let mut out = Vec::with_capacity((n * n) as usize);
    for i in 0..n * n {
        let (gx, gz) = ((i % n) as f64, (i / n) as f64);
        let mut h = i.wrapping_mul(2_654_435_761);
        h ^= h >> 15;
        h = h.wrapping_mul(0x27d4_eb2d);
        let jx = ((h & 0xFFFF) as f64 / 65535.0) - 0.5;
        let jz = (((h >> 16) & 0xFFFF) as f64 / 65535.0) - 0.5;
        out.push(ScatterInstance {
            position: anchor
                + DVec3::new(
                    (gx - (n as f64 - 1.0) * 0.5 + jx * 0.6) * step,
                    scale as f64 * 0.5,
                    (gz - (n as f64 - 1.0) * 0.5 + jz * 0.6) * step,
                ),
            rotation: Quat::IDENTITY,
            scale: glam::Vec3::splat(scale),
            color: [0.24, 0.52, 0.20, 1.0],
        });
    }
    ScatterBatch::lit(
        Arc::new(ScatterData::build(PrimMesh::Cube, anchor, out)),
        anchor,
        0.85,
        id,
    )
}

/// A ground-level camera near one corner of the flagship grid (mirror of
/// `vgeom_gate::ground_camera`).
fn flagship_camera() -> RenderView {
    let eye = DVec3::new(-26.0, 1.5, -26.0);
    RenderView {
        origin: FloatingOrigin::new(DVec3::ZERO),
        eye_world: eye,
        forward: (DVec3::ZERO - eye).as_vec3().normalize(),
        up: Vec3::Y,
        fov_y: 60f32.to_radians(),
        near: 0.05,
        width: W,
        height: H,
        ortho: None,
    }
}

/// **GATE (c).** The 10.6M-source-triangle flagship scene still renders
/// **pixel-identically** with two-pass HZB occlusion on and off — with GI v2, 100k
/// GPU-scattered instances and a bound streaming budget all enabled at the same
/// time.
///
/// `vgeom_gate::gate_b2` proved P18.1 subtractive on its own. This is the claim
/// that survives composition, and it is a genuinely different one: scatter builds
/// its **own** HZB from the depth target *after* the late meshlet draw, so if
/// occlusion culling ever removed a meshlet that did contribute a fragment, the
/// error would not merely tint a pixel — it would change the depth pyramid the
/// scatter cull reads, and a different set of 100k instances would draw. Two
/// conservative optimisations composing into a non-conservative one is precisely
/// the failure a per-feature suite cannot see.
#[test]
fn the_ten_million_triangle_gate_survives_composition() {
    // The source-triangle claim is CPU-side and must hold with or without a GPU.
    let per_instance = vgeom_demo_source_triangles() / (VGEOM_DEMO_GRID * VGEOM_DEMO_GRID) as u64;
    assert!(
        vgeom_demo_source_triangles() > 10_600_000,
        "the flagship scene must still exceed 10.6M source triangles, got {} \
         ({per_instance} per instance x {} instances)",
        vgeom_demo_source_triangles(),
        VGEOM_DEMO_GRID * VGEOM_DEMO_GRID
    );
    let _ = vgeom_demo_scene(); // the generator still builds (source-of-truth check)

    let Some(gpu) = gpu_or_skip("phase18 gate (c)") else {
        return;
    };
    let tmp = tempfile::tempdir().unwrap();
    let pack = cook_vgeom_demo(tmp.path());
    let (asset, instances) = flagship_scene_from_pack(&pack);
    assert_eq!(instances.len(), VGEOM_DEMO_GRID * VGEOM_DEMO_GRID);
    let lod0 = asset
        .source
        .to_mesh()
        .expect("materialize the vmesh")
        .classic_lods()[0]
        .triangle_count() as u64;
    assert!(
        lod0 * instances.len() as u64 > 10_600_000,
        "the cooked flagship pack must still carry 10.6M+ source triangles"
    );

    // The composed scene: the flagship meshlet grid + 102 400 scattered instances
    // over the same ground, under one sun.
    let mut scene = RenderScene {
        vgeom_assets: vec![asset],
        vgeom_instances: instances,
        scatter: vec![scatter_field(DVec3::ZERO, 320, 90.0, 0.35, 7)],
        lights: vec![RenderLight {
            kind: LightKind::Directional,
            color: [1.0, 0.97, 0.9],
            intensity: 3.0,
            direction: Vec3::new(0.35, 0.75, 0.55).normalize(),
            position: DVec3::ZERO,
            range: 0.0,
            ..RenderLight::default()
        }],
        ..Default::default()
    };
    scene.mark_dirty();
    let view = flagship_camera();

    let settings = |occlusion: bool, budget: u64| RenderSettings {
        vgeom: VgeomSettings {
            enabled: true,
            occlusion,
            two_pass: occlusion,
            stream: inf_vgeom::VgeomStreamBudget {
                budget_bytes: budget,
                ..inf_vgeom::VgeomStreamBudget::default()
            },
            ..VgeomSettings::default()
        },
        gi: GiSettings {
            enabled: true,
            probe_budget: 256,
            ..GiSettings::default()
        },
        ..RenderSettings::default()
    };

    let shot = |occlusion: bool, budget: u64, frames: usize, audit: bool| {
        let target = HeadlessTarget::new(&gpu, W, H);
        let mut r = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
        r.set_settings(settings(occlusion, budget));
        r.set_vgeom_audit(audit);
        r.set_scatter_audit(audit);
        let mut img = Vec::new();
        for _ in 0..frames {
            r.render(&gpu, &scene, &view, &target.view, (W, H));
            img = target.read_rgba(&gpu).expect("readback");
        }
        (
            img,
            r.vgeom_audit(&gpu),
            r.scatter_audit(&gpu),
            r.vgeom_stream_report(),
        )
    };

    // Size a budget that genuinely withholds pages, in PAGES off the live page
    // directory — never as a fraction of a byte total (see [`binding_budget`] for
    // the law and the CI failure that paid for it). The reference run reports what
    // this static camera wants; the budget holds half of that.
    let (_, _, _, unclamped) = shot(false, inf_vgeom::DEFAULT_VGEOM_BUDGET_BYTES, 1, false);
    assert!(
        !unclamped.stats.budget_clamped,
        "the reference run must be unclamped"
    );
    let want = unclamped.stats.wanted_pages;
    let source = &scene.vgeom_assets[0].source;
    assert!(
        want >= 2,
        "the flagship camera wants only the ROOT page ({want} of {} pages): the root \
         is granted unconditionally, so no budget can clamp this frame and the \
         streaming half of the composition claim is unstateable",
        source.pages().len()
    );
    let held = held_pages_for_want(want);
    let budget = budget_for_pages(source, held);

    let (reference, _, ref_scatter, ref_report) = shot(false, budget, 6, true);
    let (composed, audit, scatter, report) = shot(true, budget, 6, true);

    eprintln!(
        "gate (c): {} of {} pairs in the base cut, {} proven occluded ({:.1}%), \
         drawn {} early + {} late | streamed {}/{} pages ({} B) of a {want}-page \
         want under a {budget} B budget built to hold {held} (clamped {}) | \
         scatter {:?}",
        audit.base_cut,
        asset_pairs(&scene),
        audit.occluded,
        100.0 * audit.occluded as f64 / audit.base_cut.max(1) as f64,
        audit.early_drawn,
        audit.late_drawn,
        report.stats.resident_pages,
        source.pages().len(),
        report.stats.resident_bytes,
        report.stats.budget_clamped,
        scatter,
    );

    // Everything really is on at once, or the equality below proves nothing.
    assert!(audit.base_cut > 0, "the meshlet path never activated");
    assert!(
        audit.occluded > 0,
        "a ground camera looking through 324 dense bodies must occlude something: {audit:?}"
    );
    assert!(
        report.stats.budget_clamped,
        "the streaming budget never bound ({} of {} pages resident against a \
         {want}-page want under {budget} B)",
        report.stats.resident_pages,
        source.pages().len()
    );
    assert!(
        report.stats.resident_bytes <= budget,
        "resident bytes {} exceed the {budget} B ceiling",
        report.stats.resident_bytes
    );
    assert!(
        scatter.candidates == 320 * 320 && scatter.mesh > 0,
        "the scatter path never ran alongside the meshlets: {scatter:?}"
    );
    assert!(
        reference
            .chunks(4)
            .filter(|p| p[0] > 24 || p[1] > 24)
            .count()
            > 2_000,
        "the composed 10.6M-triangle view must paint real geometry"
    );

    // ── the claim ──
    let diff = composed
        .iter()
        .zip(&reference)
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        diff,
        0,
        "two-pass occlusion must stay purely subtractive at 10.6M triangles WITH \
         GI, scatter and streaming on ({diff} of {} bytes differ)",
        reference.len()
    );
    // …and the composition did not quietly disable one of them: the scatter cull
    // saw the identical instance set both ways (its HZB is built from a depth
    // target the meshlet occlusion may not have changed).
    assert_eq!(
        scatter, ref_scatter,
        "meshlet occlusion changed what the scatter cull decided — the two \
         subtractive passes are not composing"
    );
    assert_eq!(
        report.stats, ref_report.stats,
        "meshlet occlusion changed the streamer's residency"
    );
}

/// Total `(instance x meshlet)` pairs in a flagship scene — the denominator the
/// base-cut ratio is reported against.
fn asset_pairs(scene: &RenderScene) -> u64 {
    scene
        .vgeom_assets
        .iter()
        .map(|a| a.source.meshlet_count() as u64)
        .sum::<u64>()
        * scene.vgeom_instances.len() as u64
}

// ── GATE (d) — the instance-cull audit counters ──────────────────────────────

/// **GATE (d).** On the composed frame the scatter cull's counters are all
/// nonzero and reproduce exactly across two runs.
///
/// This is the anti-vacuity gate for the whole P18.5 half of the phase. Every
/// pixel comparison in this file and in `inf-render/tests/scatter.rs` is
/// satisfied by a cull that rejects *nothing* — "occlusion on is identical to
/// occlusion off" is trivially true when nothing is occluded, and "the LOD bands
/// fade" is trivially true when everything is in the near band. What makes those
/// claims mean something is that on this scene the GPU really removed instances
/// by frustum, by hierarchical depth and by distance, and split what survived
/// across the mesh and impostor lists.
#[test]
fn the_instance_cull_counters_are_real_and_deterministic() {
    let Some(gpu) = gpu_or_skip("phase18 gate (d)") else {
        return;
    };
    let tmp = tempfile::tempdir().unwrap();
    let pack = cook_phase18(tmp.path());
    let reg = pack_vmeshes(&pack);
    let settings = composed_settings(inf_vgeom::DEFAULT_VGEOM_BUDGET_BYTES);

    let a = record(&gpu, &pack, &reg, settings);
    let b = record(&gpu, &pack, &reg, settings);
    let counters: Vec<ScatterAudit> = a.iter().map(|f| f.scatter).collect();
    let again: Vec<ScatterAudit> = b.iter().map(|f| f.scatter).collect();
    assert_eq!(
        counters, again,
        "the instance-cull counters are not reproducible — the compaction or the \
         cull is order-dependent"
    );

    let total = PHASE18_SCATTER_INSTANCES + foliage_count();
    for (i, c) in counters.iter().enumerate() {
        eprintln!("gate (d) frame {i}: {c:?}");
        assert_eq!(
            c.candidates as usize, total,
            "every scattered instance must be considered"
        );
    }

    // Each rejection reason fires somewhere on the path, and both draw lists are
    // used. Asserted over the run rather than per frame: a single camera pose is
    // allowed to have (say) nothing occluded, but a traverse of a hilly field
    // behind a grid of standing slabs is not.
    let any = |f: fn(&ScatterAudit) -> u32| counters.iter().map(f).max().unwrap_or(0);
    let frustum = any(|c| c.frustum_culled);
    let occluded = any(|c| c.occluded);
    let distance = any(|c| c.distance_culled);
    let mesh = any(|c| c.mesh);
    let impostor = any(|c| c.impostor);
    eprintln!(
        "gate (d) peaks over {FRAMES} frames of {total} instances: frustum {frustum}, \
         occluded {occluded}, distance {distance}, mesh {mesh}, impostor {impostor}"
    );
    assert!(frustum > 0, "nothing was frustum-culled");
    assert!(
        occluded > 0,
        "nothing was occlusion-culled — the slabs and the terrain hid nothing"
    );
    assert!(
        distance > 0,
        "nothing was distance-culled — the volume's authored \
         {PHASE18_SCATTER_DRAW_DISTANCE} m draw distance did not bind"
    );
    assert!(mesh > 0, "nothing drew as a full mesh");
    assert!(impostor > 0, "nothing drew as an impostor");

    // …and the frame is not trivially all-visible: the surviving draw lists are a
    // strict minority of the candidate set on every frame.
    for c in &counters {
        let drawn = c.mesh + c.impostor;
        assert!(
            (drawn as usize) < total,
            "every instance survived the cull ({drawn} of {total}) — the composed \
             frame is not actually culling"
        );
        // Every candidate is rejected for exactly one reason or survives, and a
        // survivor joins at least one draw list (both, inside the cross-fade
        // band). That sandwiches the accounting exactly — a counter that
        // double-counted or dropped instances breaks one side or the other.
        let rejected = c.frustum_culled + c.occluded + c.distance_culled;
        assert!(
            rejected + c.mesh.max(c.impostor) <= c.candidates
                && c.candidates <= rejected + c.mesh + c.impostor,
            "the cull's outcomes do not account for the candidate set \
             (survivors join the mesh list, the impostor list, or — in the \
             cross-fade band — both): {c:?}"
        );
    }
}

// ── GATE (e) — the composed frame's budget, with per-system costs ────────────

/// **GATE (e).** The composed Phase 18 frame stays inside the §8 frame budget,
/// and the cost of each subsystem is **measured** against the same scene with it
/// off, so the ROADMAP can quote real numbers rather than adjectives.
///
/// Deliberately no new ratchet constant (see [`FRAME_BUDGET_MS`]). The absolute
/// millisecond figures are printed, not asserted, because a number from one
/// machine is not a contract; what is asserted is that the heaviest configuration
/// — everything on at once — is inside the budget the composed frame already has.
/// The strict assertion is skipped on a software adapter exactly as
/// `frame_budget.rs` skips it, for the same reason: a CPU rasterizer's timing is
/// not representative of anything.
#[test]
fn the_composed_frame_stays_inside_the_frame_budget() {
    let Some(gpu) = gpu_or_skip("phase18 gate (e)") else {
        return;
    };
    let info = gpu.adapter.get_info();
    let virtualized = {
        let n = info.name.to_ascii_lowercase();
        n.contains("paravirtual") || n.contains("virtualbox") || n.contains("vmware")
    };
    let software = info.device_type == wgpu::DeviceType::Cpu || virtualized;

    let tmp = tempfile::tempdir().unwrap();
    let pack = cook_phase18(tmp.path());
    let reg = pack_vmeshes(&pack);
    let budget = binding_budget(&gpu, &pack, &reg);

    // ONE projected scene, reused by every configuration: what is being measured
    // is the renderer, and re-projecting 102 400 instances per frame would fold a
    // CPU cost the shipping loop pays once per content change into every number.
    let mut sim = pack_sim(&pack);
    for _ in 0..STEPS_PER_FRAME {
        sim.step_once(RuntimeInput::default());
    }
    let mut scene = RenderScene::default();
    project_scene(&mut scene, &sim, 0.0, &reg);
    let scattered: usize = scene.scatter.iter().map(|b| b.data.len()).sum();
    let view = gate_camera(FRAMES / 2);
    let target = HeadlessTarget::new(&gpu, W, H);

    const WARM: usize = 8;
    const MEASURED: usize = 40;
    let measure = |scene: &RenderScene, settings: RenderSettings| -> f64 {
        let mut r = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
        r.set_settings(settings);
        for _ in 0..WARM {
            r.render(&gpu, scene, &view, &target.view, (W, H));
        }
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        let t0 = std::time::Instant::now();
        for _ in 0..MEASURED {
            r.render(&gpu, scene, &view, &target.view, (W, H));
            let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        }
        t0.elapsed().as_secs_f64() * 1000.0 / MEASURED as f64
    };

    // The baseline: the same world, meshlets on at an unbound budget, GI off, no
    // scatter. Each layer is then added to THAT, one at a time.
    let mut bare = scene.clone();
    bare.scatter.clear();
    bare.mark_dirty();

    let base_settings = RenderSettings {
        vgeom: VgeomSettings {
            enabled: true,
            ..VgeomSettings::default()
        },
        ..RenderSettings::default()
    };
    let mut streamed = base_settings;
    streamed.vgeom.stream.budget_bytes = budget;
    let mut with_gi = base_settings;
    with_gi.gi = GiSettings {
        enabled: true,
        probe_budget: 256,
        ..GiSettings::default()
    };

    let baseline = measure(&bare, base_settings);
    let streaming = measure(&bare, streamed);
    let gi = measure(&bare, with_gi);
    let scatter = measure(&scene, base_settings);
    let everything = measure(&scene, composed_settings(budget));

    eprintln!(
        "phase18 composed frame ({W}x{H}, {} meshlet instances, {scattered} scattered \
         instances, {budget} B meshlet budget, on {}):\n  \
         baseline (meshlets, unbound)   {baseline:.3} ms\n  \
         + streaming (bound budget)     {streaming:.3} ms ({:+.3})\n  \
         + GI v2 (amortized, 256/frame) {gi:.3} ms ({:+.3})\n  \
         + GPU scatter                  {scatter:.3} ms ({:+.3})\n  \
         everything                     {everything:.3} ms ({:+.3}); budget {FRAME_BUDGET_MS} ms{}",
        scene.vgeom_instances.len(),
        info.name,
        streaming - baseline,
        gi - baseline,
        scatter - baseline,
        everything - baseline,
        if software {
            " [software — smoke only]"
        } else {
            ""
        },
    );

    if software {
        // A CPU rasterizer's timing is not representative; the composed frame
        // pipeline having run at all is the whole claim here.
        return;
    }
    assert!(
        everything < FRAME_BUDGET_MS,
        "the composed Phase 18 frame cost {everything:.3} ms, over the \
         {FRAME_BUDGET_MS} ms budget on {} (§8: investigate the regression, never \
         raise it)",
        info.name
    );
}

// ── GATE (f) — the golden inventory ──────────────────────────────────────────

/// Every golden scene committed under `crates/inf-render/tests/goldens/`, by name.
///
/// 39 scenes entered Phase 18; P18.5 added two (`scatter.png`,
/// `scatter_impostors.png`), **P19.2 added `biomes.png`** — the Biomes view
/// mode's overlay, the one thing that phase drew that nothing else did — and
/// **P20.1 added three**: `water_ocean_noon.png`, `water_lake_dusk.png` and
/// `water_river.png`, one per water body kind, because an ocean, a lake and a
/// river are three different tessellations and three different shading regimes
/// (absorption, Fresnel, flow foam) sharing one shader. **P20.3 added two**:
/// `water_underwater_ocean.png` (the camera inside the medium — the absorption
/// story seen from the other side, plus the v1 surface light shafts) and
/// `water_wetness_shore.png` (the darkened, glossier band a water level leaves on
/// the ground it meets).
///
/// P20.3 also **re-blessed the three P20.1 water scenes**, deliberately: shoreline
/// wetness is default-on and all three carry terrain, so their ground at and below
/// the water level is now darker. Every other image is byte-identical.
const GOLDENS: [&str; 62] = [
    "2d_lit.png",
    "aerial_fog.png",
    "billboards.png",
    "biomes.png",
    // P21.2: a hole-punched heightfield with the voxel cave visible through it.
    "cave_mouth.png",
    "clouds_dusk.png",
    "clouds_night.png",
    "clouds_overcast.png",
    "clouds_scattered.png",
    // Wave NPC1b: eight bodies on ONE mesh sharing ONE joint palette, each with
    // its own tint and its own build — the crowd's far tier drawn in one call
    // from one atlas block. The wave's ONE new golden.
    "crowd_variation.png",
    "csm.png",
    "cubes.png",
    // P22.1: a walked footprint trail + vehicle ruts pressed into snow, with the
    // grass band bent out of the tracks. The batch's ONE new golden.
    "deform.png",
    "editor_default.png",
    "gi_bleed.png",
    "gi_emissive.png",
    // Wave VEN1a: a scatter batch of emissive plates lighting a white floor
    // through the GI volume, with a zero-radiance sun -- the venue substrate's
    // proof that SCATTERED emission reaches the bounce. The wave's ONE new
    // golden.
    "gi_scatter_neon.png",
    "gi_specular.png",
    "gi_terrain.png",
    "grid_sky.png",
    // Wave TER2a: the island's own ground -- four PBR materials blended by real
    // splat weights, seen from ONE METRE up. Every terrain golden before it is
    // an overlook, because below about three metres there was nothing to look
    // at.
    "ground_close.png",
    "hdr_bloom.png",
    // Wave VIS1b: the lens trio -- vignette, chromatic aberration, film grain --
    // on one high-contrast scene. All three are zero at the default.
    "lens_trio.png",
    "ortho_2d.png",
    "pbr_materials.png",
    "primitives.png",
    "scatter.png",
    "scatter_impostors.png",
    "selection_gizmo.png",
    // Wave CHAR1a.2: the masked-skin golden — eight hair cards alternating
    // alpha against a 0.5 cutoff, four discarded, which is the picture that
    // proves `SkinnedInstance::blend` reaches the fragment stage.
    "skinned_masked.png",
    "skinned_mesh.png",
    "sky_dawn.png",
    "sky_dusk.png",
    "sky_night.png",
    // Wave GTA1: the arm `sky_night` could not have. That frame pitches 35 deg
    // up and looks AWAY from the sun, so the one place a below-horizon sun
    // could still be seen -- the horizon band facing it -- was outside its
    // frame; this one stands there and bounds RED.
    "sky_night_horizon.png",
    "sky_noon.png",
    "spot_lights.png",
    "sprites_2d.png",
    "ssao.png",
    // Wave VIS1b: the sun glare / ghost chain / halo / anamorphic streak, at
    // dawn with the disc low in a dim sky. Off by default, so nothing else
    // moved -- the ADDITIVE branch of the golden rule.
    "sun_flare.png",
    "terrain.png",
    "terrain_lod.png",
    "terrain_splat.png",
    "tilemap_2d.png",
    "translucency.png",
    "unlit.png",
    // Wave VEN1a: a closed, near-black venue interior lit only by practicals --
    // a red key and a blue rim over a plank stage, a chrome pole standing in
    // the wash, a magenta neon plate throwing a halo, and a festoon chain.
    // The wave's SECOND new golden.
    "venue_interior.png",
    "vgeom_dense.png",
    "vgeom_far.png",
    // P21.1: the carved SDF volume — a slab with a bored tunnel and a dome, the
    // overhang a heightfield cannot represent.
    "voxel.png",
    // P27.4: the four virtual-shadow-map goldens — the clipmap through the page
    // table, the engine's first spot and point shadows, and the derived bias at
    // the grazing angle that breaks a flat constant. VSM is off by default, so
    // none of the fifty above moved.
    "vsm_bias_grazing.png",
    "vsm_directional.png",
    "vsm_point.png",
    "vsm_spot.png",
    "water_lake_dusk.png",
    "water_ocean_noon.png",
    "water_river.png",
    // Wave VIS1a audit: the ONE golden the wave's own signature feature can be
    // pinned by. Opaque SSR samples the previous frame's resolve, so a
    // single-frame golden can never show it — but water marches against THIS
    // frame's private resolve and needs no history, so the boat reflecting
    // itself is a deterministic one-frame image. The wave shipped believing
    // otherwise; see the VIS1a audit ledger.
    "water_ssr.png",
    "water_underwater_ocean.png",
    "water_wetness_shore.png",
    "weather_fog_dawn.png",
    "weather_snow_dusk.png",
    "weather_storm_noon.png",
];

/// **GATE (f).** The golden *inventory* is exactly these 60 PNGs.
///
/// Nothing is re-blessed here, and no pixel is compared — that stays in
/// `inf-render`'s own harness, which is where the renderer and the images live.
/// What this pins is the **set**, because that is the part a phase gate can
/// meaningfully own: a phase that adds a feature and quietly *deletes* a golden
/// (or adds one nobody reviewed) passes every per-image comparison in the repo,
/// since a comparison that is no longer run cannot fail. Listing the names rather
/// than the count makes a swap — one scene removed, another added — fail too.
#[test]
fn the_golden_inventory_is_exactly_the_committed_set() {
    let dir = workspace_root().join("crates/inf-render/tests/goldens");
    let mut found: Vec<String> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("read {}: {e}", dir.display()))
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".png"))
        .collect();
    found.sort();

    let want: Vec<String> = GOLDENS.iter().map(|s| s.to_string()).collect();
    assert!(
        want.windows(2).all(|w| w[0] < w[1]),
        "the GOLDENS list must stay sorted so the diff below is readable"
    );
    assert_eq!(
        found, want,
        "the golden inventory changed. Adding or removing a golden scene is a \
         deliberate act: update `GOLDENS` here in the same commit, and say in the \
         message which scene changed and why. Never re-bless as a side effect."
    );
    assert_eq!(found.len(), GOLDENS.len());
}

// ── the composed scene's own invariants ──────────────────────────────────────

/// Sanity on the committed sample: it round-trips byte-identically, it stays
/// small despite carrying six figures of scatter, and it really does compose all
/// four Phase 18 features.
///
/// The size claim is the load-bearing one. The whole reason the bulk scatter is a
/// `PcgVolume` rather than a `Foliage` is that a volume's `evaluated` cache is
/// derived and never persisted; if someone "simplified" the sample by baking the
/// instances into `Foliage::instances`, every other test here would still pass
/// and the repo would gain a megabyte of committed level.
#[test]
fn the_committed_level_is_small_and_composes_phase_18() {
    let bytes = std::fs::read(sample_dir().join("Phase18Scatter.inf_lvl")).unwrap();
    assert_eq!(
        bytes[0],
        inf_scene::SCHEMA_VERSION as u8,
        "the committed phase18 sample is a current-schema payload"
    );
    assert!(
        bytes.len() < 48 * 1024,
        "the gate level must stay small, got {} bytes — did the 100k scatter get \
         baked into the level?",
        bytes.len()
    );
    let level = inf_scene::RuntimeLevel::decode(&bytes).expect("level decodes");
    assert_eq!(
        level.encode().unwrap(),
        bytes,
        "phase18-scatter decode→encode must be byte-identical"
    );

    // All four Phase 18 features are authored in the one level.
    let slabs = level
        .entities
        .iter()
        .filter(|e| e.mesh.as_ref().and_then(|m| m.asset) == Some(VGEOM_DEMO_MESH_GUID))
        .count();
    assert_eq!(
        slabs,
        inf_editor_core::samples::PHASE18_SLAB_GRID * inf_editor_core::samples::PHASE18_SLAB_GRID,
        "the meshlet slabs must reference the vgeom-demo mesh by GUID"
    );
    assert!(
        level.entities.iter().any(|e| e.terrain.is_some()),
        "the level must carry a terrain (GI's voxelized ground)"
    );
    assert!(
        level
            .entities
            .iter()
            .any(|e| e.foliage.as_ref().is_some_and(|f| !f.instances.is_empty())),
        "the level must carry a painted foliage patch"
    );
    let vol = level
        .entities
        .iter()
        .find_map(|e| e.pcg_volume.as_ref())
        .expect("the level must carry a PcgVolume");
    assert_eq!(vol.graph, Some(PHASE18_PCG_ASSET_GUID));
    assert_eq!(vol.draw_distance, PHASE18_SCATTER_DRAW_DISTANCE);
    assert!(
        vol.evaluated.is_empty(),
        "a PcgVolume's evaluated cache must never be persisted — that is the whole \
         reason 100k instances cost this level nothing"
    );
    assert!(
        level.entities.iter().any(|e| e.time_of_day.is_some()),
        "the level must carry a running clock for GI's key light"
    );
}

/// The composed world really does load 100k+ scattered instances, deterministically
/// — the CPU-side half of the P18.5 claim, which holds with no GPU at all.
#[test]
fn the_cooked_world_evaluates_a_hundred_thousand_scattered_instances() {
    let tmp = tempfile::tempdir().unwrap();
    let pack = cook_phase18(tmp.path());

    let count = |sim: &RuntimeSim| -> usize {
        let w = sim.world();
        w.world()
            .iter_entities()
            .filter_map(|e| e.get::<inf_ecs::components::PcgVolume>())
            .map(|v| v.evaluated.len())
            .sum()
    };
    let a = pack_sim(&pack);
    assert_eq!(
        count(&a),
        PHASE18_SCATTER_INSTANCES,
        "the sample's stated instance count is wrong (see \
         `samples::PHASE18_SCATTER_INSTANCES` for the arithmetic)"
    );
    // The gate's headline number, pinned at compile time so a retune of the
    // sample cannot quietly drop the scene below the 100k target.
    const _: () = assert!(PHASE18_SCATTER_INSTANCES >= 100_000);
    assert_eq!(
        count(&pack_sim(&pack)),
        count(&a),
        "the scatter is not deterministic across two loads off the same pack"
    );
    // …and it is re-evaluated on the PIE side too, from the same committed graph.
    assert_eq!(
        count(&pie_sim()),
        count(&a),
        "PIE scatter != shipping scatter"
    );

    // The volume evaluated over the terrain rather than over a flat fallback: the
    // instances follow the sample's height function.
    let w = a.world();
    let e = w.entity_of(PHASE18_PCG_GUID).expect("the volume entity");
    let vol = w
        .world()
        .get::<inf_ecs::components::PcgVolume>(e)
        .expect("the volume component");
    assert!(
        vol.evaluated
            .iter()
            .take(512)
            .all(|i| { (i.position.y - phase18_height(i.position.x, i.position.z)).abs() < 1.0 }),
        "the scatter did not land on the terrain — it evaluated over a flat fallback"
    );
    assert!(
        w.entity_of(PHASE18_TERRAIN_GUID).is_some(),
        "the terrain entity must survive the cook"
    );
}

// ── PIE == shipping, on the SKINNED POSE ─────────────────────────────────────
//
// P18.3 gave the editor viewport a real `SkeletalMesh` projection and left the
// shipped player with **no `SkeletalMesh` branch at all**: `project_scene` never
// populated `RenderScene::skinned_meshes` / `skinned`, so a level carrying a
// skeletal character previewed correctly in PIE and shipped as nothing. That is a
// PIE-vs-shipping divergence rather than a missing feature, and it is the exact
// shape of failure this file exists for — invisible to every unit test, found by
// a player.
//
// `inf-editor-core`'s `tests/projector_mirror.rs` now pins the two hosts' *source*
// against each other: the pose rule character for character, the bind-space
// rebuild character for character, and the `SkinnedInstance` literal field for
// field. What it cannot say is that the two hosts, fed the same authored character
// through two completely different loaders (a cooked pack vs the editor's in-memory
// PIE payload) and stepped through the same fixed-step schedule, arrive at the same
// **pose**. That is this — the animation twin of gate (b).
//
// The character is built in code rather than in `samples/`, deliberately: the
// committed `character-demo` sample has a skeleton, clips and a state machine but
// **no skeletal MESH asset** (nothing could draw one until this batch), so it
// cannot exercise the branch. `phase17_gate` sets the precedent for a gate that
// authors its own document.

/// The gate character's entity GUID.
const SKINNED_CHARACTER_GUID: uuid::Uuid = uuid::Uuid::from_u128(0x18_3001);

/// Clip length, seconds. Long enough that both sampled play-heads below sit
/// strictly inside one cycle: on a LOOPING player `t == duration` wraps back to
/// the start, which would compare the rest pose to itself and pass for the wrong
/// reason.
const SKINNED_CLIP_SECONDS: f64 = 2.0;

/// The two step counts the pose is sampled at (60 Hz ⇒ 0.25 s and 0.75 s). Two,
/// because a single sample makes "PIE == shipping" true for a frozen palette —
/// the anti-vacuity claim is that the pose *moved* between them.
const POSE_STEPS_A: usize = 15;
const POSE_STEPS_B: usize = 45;

/// A 4-vertex, 2-triangle strip skinned to a 2-joint chain: the lower pair
/// pinned to the root, the upper pair to the tip, so a rotation of the tip joint
/// visibly moves half the vertices and none of the rest.
fn skinned_body() -> inf_mesh::MeshAsset {
    let v = |x: f32, y: f32| inf_mesh::MeshVertex {
        position: [x, y, 0.0],
        normal: [0.0, 0.0, 1.0],
        ..Default::default()
    };
    let root = inf_mesh::VertexSkin {
        joints: [0, 0, 0, 0],
        weights: [1.0, 0.0, 0.0, 0.0],
    };
    let tip = inf_mesh::VertexSkin {
        joints: [1, 0, 0, 0],
        weights: [1.0, 0.0, 0.0, 0.0],
    };
    inf_mesh::MeshAsset::new(
        vec![inf_mesh::SubMesh {
            name: "body".into(),
            vertices: vec![v(-0.5, 0.0), v(0.5, 0.0), v(-0.5, 2.0), v(0.5, 2.0)],
            indices: vec![0, 1, 2, 2, 1, 3],
            material_slot: None,
            skin: vec![root, root, tip, tip],
        }],
        vec![],
    )
}

fn two_joint_skeleton() -> inf_anim::Skeleton {
    inf_anim::Skeleton::new(vec![
        inf_anim::Joint {
            name: "root".into(),
            parent: None,
            local_bind: inf_anim::JointTransform::default(),
            inverse_bind: glam::Mat4::IDENTITY.to_cols_array(),
        },
        inf_anim::Joint {
            name: "tip".into(),
            parent: Some(0),
            local_bind: inf_anim::JointTransform::from_trs(
                Vec3::new(0.0, 1.0, 0.0),
                Quat::IDENTITY,
                Vec3::ONE,
            ),
            inverse_bind: glam::Mat4::from_translation(Vec3::new(0.0, -1.0, 0.0)).to_cols_array(),
        },
    ])
    .expect("the fixture skeleton is valid")
}

/// A clip that rotates the tip joint 90° about X over its whole length, linearly
/// — so *every* play-head inside the cycle is a distinct pose, and two different
/// step counts cannot coincidentally agree.
fn wave_clip() -> inf_anim::AnimClip {
    // `AnimClip::new` since `.inf_anim` v2 (P29.2): `duration` is derived from
    // the keys, which are exactly `SKINNED_CLIP_SECONDS` apart.
    inf_anim::AnimClip::new(
        "wave",
        vec![inf_anim::JointTrack {
            joint: 1,
            translation: None,
            rotation: Some(inf_anim::QuatTrack::new(
                vec![0.0, SKINNED_CLIP_SECONDS as f32],
                vec![
                    Quat::IDENTITY.to_array(),
                    Quat::from_rotation_x(std::f32::consts::FRAC_PI_2).to_array(),
                ],
                inf_anim::Interpolation::Linear,
            )),
            scale: None,
        }],
    )
}

/// The gate's document: one entity bound to the fixture mesh + skeleton, playing
/// the fixture clip, with a deliberately non-default `Material` so the PBR half of
/// the projection is compared too.
fn character_doc(
    mesh: uuid::Uuid,
    skel: uuid::Uuid,
    clip: uuid::Uuid,
) -> inf_editor_core::scene::SceneDoc {
    use inf_ecs::components::{AnimPlayer, Material, SkeletalMesh, Transform};

    let mut doc = inf_editor_core::scene::SceneDoc::new();
    doc.set_title("Phase 18 Skinned Parity");
    doc.create_with_guid(
        SKINNED_CHARACTER_GUID,
        inf_editor_core::ipc::SpawnKind::Empty,
        "Hero",
        None,
    );
    let e = doc
        .world()
        .entity_of(SKINNED_CHARACTER_GUID)
        .expect("the character entity");
    let w = doc.world_mut().world_mut();
    w.entity_mut(e)
        .insert(Transform::from_translation(DVec3::new(3.0, 1.5, -2.0)));
    w.entity_mut(e).insert(SkeletalMesh {
        mesh: Some(mesh),
        skeleton: Some(skel),
    });
    w.entity_mut(e).insert(AnimPlayer {
        clip: Some(clip),
        t: 0.0,
        speed: 1.0,
        looping: true,
        playing: true,
        duration: SKINNED_CLIP_SECONDS,
    });
    w.entity_mut(e).insert(Material {
        base_color: inf_ecs::Color::new(0.31, 0.62, 0.44, 1.0),
        metallic: 0.25,
        roughness: 0.37,
        emissive: inf_ecs::Color::new(0.05, 0.0, 0.11, 1.0),
        ..Material::default()
    });
    doc.world_mut().mark_dirty();
    doc.world_mut().propagate();
    doc
}

/// Scaffold a project holding the fixture character (mesh + skeleton + clip +
/// level) and cook it. Returns the pack directory and the authored document, so
/// the PIE arm runs off the same source of truth the cook consumed.
fn cook_skinned_character(tmp: &Path) -> (PathBuf, inf_editor_core::scene::SceneDoc) {
    let proj = tmp.join("charproj");
    ProjectManifest::new("Phase 18 Skinned", "blank-3d")
        .save(&proj)
        .unwrap();
    let content = proj.join("Content");
    std::fs::create_dir_all(&content).unwrap();

    let mut assets = inf_editor_core::assets::AssetProject::open(&content).expect("asset project");
    let dir = assets.content_dir("Character").expect("content dir");
    let mesh = assets
        .write_asset(&dir, "Body", &skinned_body(), None, vec![], None)
        .expect("write .inf_mesh");
    let skel = assets
        .write_asset(
            &dir,
            "Rig",
            &inf_anim::SkeletonAsset::new(two_joint_skeleton()),
            None,
            vec![],
            None,
        )
        .expect("write .inf_skel");
    let clip = assets
        .write_asset(
            &dir,
            "Wave",
            &inf_anim::AnimClipAsset::new(wave_clip(), Some(*skel.uuid().as_bytes())),
            None,
            vec![],
            None,
        )
        .expect("write .inf_anim");

    let doc = character_doc(mesh.uuid(), skel.uuid(), clip.uuid());
    inf_editor_core::scene::serialize::save(&doc, &dir.join("Character.inf_lvl"), None)
        .expect("save level");

    let out = tmp.join("charout");
    cook(&proj, &out, &CookOptions::default()).expect("cook succeeds");
    (out, doc)
}

/// The shipping arm: the world the cooked pack boots.
fn skinned_pack_sim(pack_dir: &Path) -> RuntimeSim {
    let source = PackLevelSource::open(pack_dir).expect("pack opens");
    let built = inf_player::build_world_from_pack(&source).expect("pack level builds");
    inf_player::sim_from_built(built)
}

/// The PIE arm: the world the editor's in-memory payload boots, from the same doc.
fn skinned_pie_sim(doc: &inf_editor_core::scene::SceneDoc) -> RuntimeSim {
    let payload = inf_editor_core::pie::build_scene_payload(
        doc,
        |_| None,
        |_| None,
        |_| None,
        |_| None,
        |_| None,
        |_| None,
        // P22.3: no destructible meshes in this fixture.
        |_| None,
        // P26.3b: the cloth / hair / material / texture byte resolver.
        |_| None,
        |_| None,
        60,
        false,
    )
    .expect("payload builds");
    let built = inf_player::build_world_from_payload(&payload).expect("PIE world builds");
    inf_player::sim_from_built(built)
}

/// The skeletal store, sliced out of the cooked pack. **One** registry drives both
/// arms, exactly as `pack_vmeshes` does for the vgeom half: what is being compared
/// is the two *worlds*, not two copies of the same immutable asset store — and a
/// shared store is also what makes the `Arc`-identity claim below meaningful.
fn skinned_registry(pack_dir: &Path) -> SkinnedRegistry {
    let reader = inf_asset::PackReader::open(&pack_dir.join(level::PACK_FILE))
        .expect("open pack for the skeletal registry");
    SkinnedRegistry::from_pack(Arc::new(reader))
}

/// One projected `SkinnedInstance`, as bits: `(mesh slot, world translation,
/// colour, (metallic, roughness), the whole joint palette)`.
type SkinnedBits = (usize, [u64; 3], [u32; 4], [u32; 2], Vec<[u32; 16]>);

/// One projected `skinned_meshes` entry, as bits: `(vertex count, index count,
/// FNV-1a of every vertex bit + every index)`.
type SkinnedMeshBits = (usize, usize, u64);

/// One projected frame of the skeletal half of the scene, as **bits**. Same rule
/// as everywhere else in this file: an `f32` palette compared "within 1e-6" is
/// exactly the drift a parity gate exists to catch.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ProjectedSkinned {
    /// The bind-space geometry each slot carries. The counts alone would let two
    /// hosts upload *different geometry* of the same size and still agree, which
    /// is why the fold over every vertex bit is in the key.
    meshes: Vec<SkinnedMeshBits>,
    /// Every skinned instance the projection placed.
    instances: Vec<SkinnedBits>,
    /// How many rigid instances the projection emitted — i.e. how many
    /// `SkeletalMesh` entities fell back to the placeholder cube. Zero is the
    /// claim; a nonzero value with equal `instances` would be two hosts agreeing
    /// that the character does not draw.
    placeholders: usize,
}

fn fnv(bytes: impl Iterator<Item = u32>) -> u64 {
    bytes.fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
        (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn project_skinned(
    sim: &RuntimeSim,
    reg: &SkinnedRegistry,
    scene: &mut RenderScene,
) -> ProjectedSkinned {
    project_scene_with_skinned(
        scene,
        sim,
        0.0,
        &VmeshRegistry::new(),
        reg,
        &inf_voxel::VoxelVolumes::new(),
    );
    ProjectedSkinned {
        meshes: scene
            .skinned_meshes
            .iter()
            .map(|m| {
                let verts = m.vertices.iter().flat_map(|v| {
                    v.pos
                        .iter()
                        .chain(v.normal.iter())
                        .map(|f| f.to_bits())
                        .chain(v.joints)
                        .chain(v.weights.iter().map(|f| f.to_bits()))
                        .collect::<Vec<u32>>()
                });
                (
                    m.vertices.len(),
                    m.indices.len(),
                    fnv(verts.chain(m.indices.iter().copied())),
                )
            })
            .collect(),
        instances: scene
            .skinned
            .iter()
            .map(|i| {
                (
                    i.mesh,
                    [
                        i.translation.x.to_bits(),
                        i.translation.y.to_bits(),
                        i.translation.z.to_bits(),
                    ],
                    [
                        i.color[0].to_bits(),
                        i.color[1].to_bits(),
                        i.color[2].to_bits(),
                        i.color[3].to_bits(),
                    ],
                    [i.metallic.to_bits(), i.roughness.to_bits()],
                    i.palette
                        .iter()
                        .map(|m| m.to_cols_array().map(f32::to_bits))
                        .collect(),
                )
            })
            .collect(),
        placeholders: scene.instances.len(),
    }
}

/// Step `sim` `steps` fixed steps, then project into `scene`.
fn skinned_trace(
    mut sim: RuntimeSim,
    reg: &SkinnedRegistry,
    steps: usize,
    scene: &mut RenderScene,
) -> ProjectedSkinned {
    for _ in 0..steps {
        sim.step_once(RuntimeInput::default());
    }
    project_skinned(&sim, reg, scene)
}

/// **The skinned-pose parity arm.** The cooked pack and the editor's PIE payload,
/// stepped the same number of fixed steps, project the same skinned instances —
/// bit for bit, palettes included.
///
/// If this failed, a shipped character would be posed differently from the one the
/// animator approved in PIE (or, as before this batch, would not be drawn at all),
/// and nothing else in the suite would notice: every existing character gate
/// asserts the *sim* state — transforms and state-machine indices — which is
/// exactly the state that was already correct while the render projection was
/// missing.
#[test]
fn pie_equals_shipping_on_the_projected_skinned_pose() {
    let tmp = tempfile::tempdir().unwrap();
    let (pack, doc) = cook_skinned_character(tmp.path());
    let reg = skinned_registry(&pack);

    let mut ship_scene = RenderScene::default();
    let mut pie_scene = RenderScene::default();
    let ship_a = skinned_trace(skinned_pack_sim(&pack), &reg, POSE_STEPS_A, &mut ship_scene);
    let pie_a = skinned_trace(skinned_pie_sim(&doc), &reg, POSE_STEPS_A, &mut pie_scene);
    assert_eq!(
        pie_a, ship_a,
        "the PIE payload's projected skinned pose != the cooked pack's — PIE is \
         not shipping for skeletal characters"
    );

    // …and again at a different play-head, so the agreement is not an artifact of
    // one sampled instant.
    let ship_b = skinned_trace(skinned_pack_sim(&pack), &reg, POSE_STEPS_B, &mut ship_scene);
    let pie_b = skinned_trace(skinned_pie_sim(&doc), &reg, POSE_STEPS_B, &mut pie_scene);
    assert_eq!(pie_b, ship_b, "PIE != shipping at the second play-head");

    // ── anti-vacuity ──
    //
    // Everything above is satisfied by two hosts that both draw nothing, or both
    // draw a frozen bind pose. These are the claims that make it mean something.
    assert_eq!(
        ship_a.instances.len(),
        1,
        "the shipped player projected no skinned instance — the `SkeletalMesh` \
         branch did not fire (this IS the divergence the arm exists for)"
    );
    assert_eq!(
        ship_a.meshes.len(),
        1,
        "exactly one `(mesh, skeleton)` pair should own a `skinned_meshes` slot"
    );
    assert_eq!(
        ship_a.placeholders, 0,
        "the character fell back to its placeholder cube — the cook did not ship \
         the `.inf_mesh`/`.inf_skel`, or the registry could not resolve them"
    );
    assert_eq!(
        (ship_a.meshes[0].0, ship_a.meshes[0].1),
        (4, 6),
        "the bind-space rebuild lost or invented geometry"
    );
    let palette_a = &ship_a.instances[0].4;
    let palette_b = &ship_b.instances[0].4;
    assert_eq!(palette_a.len(), 2, "one palette matrix per skeleton joint");
    let identity = glam::Mat4::IDENTITY.to_cols_array().map(f32::to_bits);
    assert!(
        palette_a.iter().any(|m| *m != identity),
        "the projected palette is all-identity — the clip is not driving the \
         pose, so the parity above compares two bind poses"
    );
    assert_ne!(
        palette_a, palette_b,
        "the palette did not move between {POSE_STEPS_A} and {POSE_STEPS_B} steps \
         — the play-head is frozen and the parity claim is trivial"
    );

    // The `Arc<SkinnedMeshData>` discipline, end to end: one registry means ONE
    // bind-space upload, shared by both hosts' scenes and stable across
    // re-projection. The skinned pass caches its GPU buffers on that pointer, so a
    // projector that rebuilt the stream per frame would re-upload a character
    // every frame while every assertion above still passed.
    assert!(
        Arc::ptr_eq(&ship_scene.skinned_meshes[0], &pie_scene.skinned_meshes[0]),
        "the two hosts hold different `Arc`s for one asset — the store is not \
         sharing its bind-space buffer"
    );
    let before = Arc::as_ptr(&ship_scene.skinned_meshes[0]);
    project_skinned(&skinned_pack_sim(&pack), &reg, &mut ship_scene);
    assert_eq!(
        before,
        Arc::as_ptr(&ship_scene.skinned_meshes[0]),
        "re-projecting rebuilt the bind-space buffer — every frame would re-upload \
         the whole character"
    );

    // Finally: the narrower four-argument door really is what left the player
    // blind. Same world, no skeletal store ⇒ the placeholder, which is what
    // shipped before this batch.
    let mut inert = RenderScene::default();
    project_scene(
        &mut inert,
        &skinned_pack_sim(&pack),
        0.0,
        &VmeshRegistry::new(),
    );
    assert!(
        inert.skinned.is_empty() && inert.instances.len() == 1,
        "with no skeletal store the character must degrade to exactly one \
         placeholder cube, not vanish"
    );
}
