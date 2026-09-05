//! **The Phase 27 gate** (P27.5, clause 3) — virtual shadow maps, over a real
//! device, on a scripted path, through the shipping projector.
//!
//! Phase 27's own "Done when" is the shape of this file:
//!
//! > directional + spot + point shadows render through the page atlas at ≥8k
//! > effective on High tier, a static scene re-rasterizes **zero** pages after
//! > warm-up (engagement-counter proven, falsifiable), moving one object
//! > invalidates exactly the pages its bounds touch, vgeom/skinned/terrain/rigid
//! > all cast, the page-allocation trace is deterministic and PIE == shipping
//! > holds, and Low tier keeps CSM through `RenderTier::apply`.
//!
//! The six arms the ROADMAP names, and what each is built to falsify:
//!
//! * **(a) the page-allocation trace is deterministic and PIE == shipping.** Two
//!   runs of one scripted walk agree byte for byte, and two *hosts* — a cooked
//!   pack and an editor `ScenePayload` — agree on both the trace and the frame.
//!   Anti-vacuity in both directions: the trace has to MOVE along the walk, and
//!   the shadowed frame has to differ from the same frame with virtual shadows
//!   off. The trace is by **address** (`light/face/level/x/y`) and never by
//!   slot: a slot index is an allocation order and two hosts that seated the
//!   same pages differently describe the same shadow.
//! * **(b) a static scene re-rasterizes ZERO pages after warm-up — and the arm
//!   FAILS if the counter is dead.** The ROADMAP spells the anti-vacuity
//!   requirement in the phase text itself, because a no-op cache also
//!   re-rasterizes zero. So the warm-up must be proven to have rasterized,
//!   drawn and cached first, and the cache's own counter must keep moving while
//!   the raster's does not.
//! * **(c) page-exact invalidation on a scripted mover.** A real entity moved
//!   through the sim and re-projected through the one door, with the expected
//!   page set computed **independently** through `vsm_page_sees_sphere` over
//!   each resident page's own matrix — and the count asserted exactly, against
//!   a fixture where the mover touches some pages and not others.
//! * **(d) point/spot engagement.** Their trees register, their pages become
//!   resident on more than one cube face, the receiver engages — and every one
//!   of those counters is **zero** on the shipped defaults.
//! * **(e) budgets, in the three classes P26.5 settled**: LOAD (once, against
//!   `LOAD_BUDGET_MS`), WORLD (pages and rasters, on every adapter, because a
//!   page is a page everywhere) and CLOCK (milliseconds, only where a
//!   millisecond represents a frame). The sun's time-slicing drain bound — the
//!   drain is **eleven times** faster than the quantum that fills it — is
//!   asserted here, where the gate can see both numbers.
//! * **(f) the golden set is pinned and additive**, by content digest as well as
//!   by count.
//!
//! Two more claims ride this file because the P26.5 audit routed them here:
//!
//! * **refinement under camera approach, stated AT the gate over the scripted
//!   path's trace** — the marked level gets finer as the camera closes, over the
//!   same bit-exact walk arm (a) pins, rather than only over a seeded device
//!   fixture;
//! * **the phase's own ≥8k-effective claim on High**, measured rather than
//!   asserted from a settings literal: the live tree's finest level, times the
//!   page size, is the virtual resolution a receiver addresses.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use inf_asset::{AssetId, AssetKind, AssetSidecar, ContentHash};
use inf_core::FRAME_BUDGET_MS;
use inf_ecs::components::{Light, LightKind, Transform};
use inf_editor_core::scene::{serialize, SceneDoc};
use inf_packager::{cook, CookOptions};
use inf_player::budget::LOAD_BUDGET_MS;
use inf_project::ProjectManifest;
use inf_render::{
    EngineRenderer, GpuContext, RenderScene, RenderSettings, RenderTier, RenderView,
    VsmLightHandle, VsmPage, VsmSettings,
};
use uuid::Uuid;

const LEVEL: Uuid = Uuid::from_u128(0x2705_0000_0000_0001);
/// The mover's NAME — its identity across both hosts (see `mover_entity`).
const MOVER_NAME: &str = "Mover";
const W: u32 = 256;
const H: u32 = 144;
/// Frames the scripted walk runs. The marking ring is pinned at `frame − 2`, so
/// nothing is resident before frame 2; twelve leaves a stretch of settled frames
/// for the trace to differ over and for refinement to show.
const PATH_FRAMES: u64 = 12;
/// The mover's own scale and the step it takes, in metres. Small and short on
/// purpose: arm (c) is about *exactly* the pages it touches, which needs pages
/// it does not touch.
const MOVER_SCALE: f64 = 0.4;
const MOVER_STEP: f64 = 0.7;

fn gpu_or_skip(what: &str) -> Option<GpuContext> {
    match GpuContext::headless() {
        Ok(gpu) => Some(gpu),
        Err(e) => {
            eprintln!("SKIP {what}: no GPU adapter ({e})");
            None
        }
    }
}

fn put(content: &Path, file: &str, guid: Uuid, bytes: &[u8], kind: AssetKind) {
    let path = content.join(file);
    std::fs::write(&path, bytes).expect("write asset");
    AssetSidecar::new(AssetId(guid), kind, ContentHash::of(bytes))
        .save(&path)
        .expect("write sidecar");
}

fn cook_opts() -> CookOptions {
    CookOptions {
        vgeom: inf_packager::VgeomCookOptions {
            enabled: false,
            ..Default::default()
        },
        ..Default::default()
    }
}

/// The fixture level: a floor, three static casters, **one mover with a stable
/// GUID** (arm (c) moves it through the sim), a shadow-casting sun, and — for
/// arm (d) — a shadow-casting spot and a shadow-casting point light.
///
/// `analytic` is the switch arm (d) needs: without the spot and the point, the
/// same level exercises the directional path alone, which is what makes their
/// counters a measurement of *those two lights* rather than of shadows.
fn shadow_project(tmp: &Path, analytic: bool) -> (PathBuf, SceneDoc) {
    let proj = tmp.join(if analytic { "gate" } else { "gate-sun" });
    ProjectManifest::new("Phase 27 Gate", "blank-3d")
        .save(&proj)
        .unwrap();
    let content = proj.join("Content");
    std::fs::create_dir_all(&content).unwrap();

    let mut doc = SceneDoc::new();
    let place = |doc: &mut SceneDoc, name: &str, t: [f64; 3], s: [f64; 3]| -> Uuid {
        let id = doc.edit_create(inf_editor_core::ipc::SpawnKind::Cube, name, None);
        let world = doc.world_mut();
        let e = world.entity_of(id).expect("the cube exists");
        world.world_mut().entity_mut(e).insert(Transform {
            translation: inf_ecs::math::Vec3d::new(t[0], t[1], t[2]),
            scale: inf_ecs::math::Vec3d::new(s[0], s[1], s[2]),
            ..Default::default()
        });
        id
    };
    place(&mut doc, "Floor", [0.0, -0.5, 0.0], [60.0, 1.0, 60.0]);
    for (i, (x, z)) in [(-2.4f64, 0.8f64), (1.4, -1.2), (3.0, 1.6)]
        .into_iter()
        .enumerate()
    {
        place(&mut doc, &format!("Box{i}"), [x, 0.75, z], [1.4, 1.5, 1.4]);
    }
    let mover = place(
        &mut doc,
        "Mover",
        [-1.5, 1.0, 2.4],
        [MOVER_SCALE, MOVER_SCALE, MOVER_SCALE],
    );
    // The mover is found by NAME rather than by a hand-set GUID, and that is a
    // correction rather than a preference: `SceneDoc` keys its own
    // creation-order list by the guid it minted, so replacing the component
    // orphans the entity at save time — measured, and it cost the fixture its
    // fifth instance.
    let _ = mover;

    let mut light = |name: &str, kind: LightKind, t: [f64; 3], rot: [f64; 3]| {
        let id = doc.edit_create(
            inf_editor_core::ipc::SpawnKind::DirectionalLight,
            name,
            None,
        );
        let world = doc.world_mut();
        let e = world.entity_of(id).expect("the light exists");
        world.world_mut().entity_mut(e).insert(Light {
            kind,
            intensity: 3.0,
            range: 40.0,
            cast_shadows: true,
            ..Default::default()
        });
        world.world_mut().entity_mut(e).insert(Transform {
            translation: inf_ecs::math::Vec3d::new(t[0], t[1], t[2]),
            rotation: inf_ecs::math::Vec3d::new(rot[0], rot[1], rot[2]),
            ..Default::default()
        });
    };
    // The sun first, so it is scene light 0 and `sun_slot` names its tree.
    light(
        "Sun",
        LightKind::Directional,
        [0.0, 0.0, 0.0],
        [-55.0, 24.0, 0.0],
    );
    if analytic {
        light("Spot", LightKind::Spot, [-4.0, 3.2, 2.0], [-60.0, 0.0, 0.0]);
        light("Point", LightKind::Point, [3.2, 2.4, -1.0], [0.0, 0.0, 0.0]);
    }

    let level = serialize::encode(&serialize::to_scene_file(&doc)).expect("encode level");
    put(&content, "Gate.inf_lvl", LEVEL, &level, AssetKind::Level);
    (proj, doc)
}

/// The PIE payload for `doc`, served from the project on disk — `vsm_pie.rs`'s
/// `payload_for`, kept identical so the two suites build the wire the same way.
fn payload_for(proj: &Path, doc: &SceneDoc) -> inf_runtime::pie::ScenePayload {
    let content = proj.join("Content");
    let mut by_guid: std::collections::HashMap<Uuid, PathBuf> = std::collections::HashMap::new();
    for e in std::fs::read_dir(&content).expect("content dir") {
        let p = e.expect("dir entry").path();
        if let Ok(side) = AssetSidecar::load(&p) {
            by_guid.insert(side.guid.uuid(), p);
        }
    }
    let read = move |g: Uuid| by_guid.get(&g).and_then(|p| std::fs::read(p).ok());
    inf_editor_core::pie::build_scene_payload(
        doc,
        |_| None,
        |_| None,
        |_| None,
        |_| None,
        |_| None,
        |_| None,
        |_| None,
        read,
        |_| None,
        0,
        false,
    )
    .expect("the payload builds")
}

/// **The scripted path**: the eye walks in toward the caster row. An APPROACH
/// rather than an orbit, deliberately — refinement under camera approach is one
/// of the two claims the P26.5 audit routed to this gate, and it needs the
/// distance to fall.
fn path_view(step: u64) -> RenderView {
    let eye = glam::DVec3::new(0.0, 4.6, 13.0 - 0.85 * step as f64);
    RenderView {
        origin: inf_math::FloatingOrigin::new(glam::DVec3::ZERO),
        eye_world: eye,
        forward: (glam::DVec3::new(0.0, 0.4, 0.0) - eye)
            .as_vec3()
            .normalize(),
        up: glam::Vec3::Y,
        fov_y: 45f32.to_radians(),
        near: 0.05,
        width: W,
        height: H,
        ortho: None,
    }
}

/// The virtual-shadow configuration every arm here runs, **through the High
/// tier's own clamp** so the ≥8k claim is about a configuration a High machine
/// actually gets rather than about a literal in a test.
fn vsm_on() -> VsmSettings {
    RenderTier::High
        .apply(RenderSettings {
            vsm: VsmSettings {
                enabled: true,
                ..Default::default()
            },
            ..Default::default()
        })
        .vsm
}

/// What one scripted run produced.
struct Run {
    /// One line a frame: every resident page, by (light, face, level, x, y),
    /// sorted by ADDRESS.
    trace: Vec<String>,
    /// The finest (numerically smallest) resident level of the sun's tree, per
    /// frame — the refinement claim's own number.
    finest: Vec<u32>,
    /// The last frame's pixels.
    frame: Vec<u8>,
    receiver_frames: u64,
    raster: inf_render::VsmRasterStats,
    marking: inf_render::VsmStreamStats,
    /// Wall time of each frame — `render` plus the poll, and nothing else.
    frame_ms: Vec<f64>,
    /// Pages rasterized on each frame, differenced out of the running total.
    pages_each: Vec<u64>,
    /// Atlas slots the residency was planned with.
    slots: u32,
    /// The finest level's page grid, one side — the ≥8k measurement's input.
    finest_pages_per_side: u32,
}

impl Run {
    fn steady_ms(&self) -> f64 {
        let tail = &self.frame_ms[1.min(self.frame_ms.len())..];
        if tail.is_empty() {
            return 0.0;
        }
        tail.iter().sum::<f64>() / tail.len() as f64
    }
    fn peak_steady_pages(&self) -> u64 {
        self.pages_each[1.min(self.pages_each.len())..]
            .iter()
            .copied()
            .max()
            .unwrap_or(0)
    }
    fn pages_trace(&self) -> String {
        self.pages_each
            .iter()
            .enumerate()
            .map(|(i, p)| format!("{i}:{p}"))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// The resident page set as an address line, and the finest resident level of
/// the **sun's** tree.
///
/// By tree KIND and not by handle 0: a handle is the n-th shadow-casting light
/// in scene order and the projector walks `Guid` order, so which light is
/// handle 0 is a property of the document's minted ids. (Measured: the fixture's
/// sun is handle 2, and the first draft of the refinement assertion was reading
/// the SPOT's levels, which a camera approach does not refine.)
fn residency_line(renderer: &EngineRenderer) -> (String, u32) {
    let Some(sys) = renderer.vsm() else {
        return (String::new(), u32::MAX);
    };
    let sun = sys
        .trees()
        .iter()
        .position(|t| t.kind == inf_render::VsmTreeKind::Clipmap)
        .map(|i| i as u32);
    let res = sys.residency();
    let geom = res.geometry();
    let mut pages: Vec<(u32, u32, u32, u32, u32)> = (0..geom.slot_count())
        .filter_map(|s| res.slot_occupant(s))
        .map(|(l, p)| (l.0, p.face, p.level, p.x, p.y))
        .collect();
    pages.sort_unstable();
    let finest = pages
        .iter()
        .filter(|(l, ..)| Some(*l) == sun)
        .map(|(_, _, lv, _, _)| *lv)
        .min()
        .unwrap_or(u32::MAX);
    let mut line = String::new();
    for (l, f, lv, x, y) in pages {
        line.push_str(&format!("{l}/{f}/{lv}/{x}/{y};"));
    }
    (line, finest)
}

/// Render the scripted walk over `sim`, through the **player's own projector** —
/// the one door a shipped frame's scene comes through.
fn scripted(gpu: &GpuContext, sim: &inf_player::runtime_sim::RuntimeSim, vsm: bool) -> Run {
    let target = inf_render::HeadlessTarget::new(gpu, W, H);
    let mut renderer = EngineRenderer::new(gpu, inf_render::HEADLESS_FORMAT);
    let mut settings = RenderSettings::default();
    if vsm {
        settings.vsm = vsm_on();
    }
    renderer
        .try_set_settings(settings)
        .expect("the gate's settings are legal");
    let vmeshes = inf_player::vmesh::VmeshRegistry::new();

    let mut trace = Vec::with_capacity(PATH_FRAMES as usize);
    let mut finest = Vec::with_capacity(PATH_FRAMES as usize);
    let mut frame_ms = Vec::with_capacity(PATH_FRAMES as usize);
    let mut pages_each = Vec::with_capacity(PATH_FRAMES as usize);
    let mut pages_so_far = 0u64;
    for step in 0..PATH_FRAMES {
        let mut scene = RenderScene::default();
        inf_player::render::project_scene(&mut scene, sim, 0.0, &vmeshes);
        scene.grid_enabled = false;
        scene.mark_dirty();
        let v = path_view(step);
        let t = std::time::Instant::now();
        renderer.render(gpu, &scene, &v, &target.view, (W, H));
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        frame_ms.push(t.elapsed().as_secs_f64() * 1000.0);
        let now = renderer.vsm_raster_stats().map_or(0, |s| s.pages);
        pages_each.push(now - pages_so_far);
        pages_so_far = now;
        let (line, f) = residency_line(&renderer);
        trace.push(line);
        finest.push(f);
    }
    let (slots, finest_pages_per_side) = match renderer.vsm() {
        Some(sys) => (
            sys.residency().geometry().slot_count(),
            sys.trees()
                .iter()
                .find(|t| t.kind == inf_render::VsmTreeKind::Clipmap)
                .map(|t| t.levels[0].pages_x)
                .unwrap_or(0),
        ),
        None => (0, 0),
    };
    Run {
        trace,
        finest,
        frame: target.read_rgba(gpu).expect("readback"),
        receiver_frames: renderer.vsm_receiver_frames(),
        raster: renderer.vsm_raster_stats().unwrap_or_default(),
        marking: renderer.vsm().map(|s| s.stats()).unwrap_or_default(),
        frame_ms,
        pages_each,
        slots,
        finest_pages_per_side,
    }
}

/// Cook the fixture and build the **shipping** world out of the pack.
fn shipped(tmp: &Path, analytic: bool) -> (inf_player::runtime_sim::RuntimeSim, PathBuf) {
    let (proj, _) = shadow_project(tmp, analytic);
    let report = cook(&proj, &tmp.join("out"), &cook_opts()).expect("the project cooks");
    let source =
        inf_player::level::PackLevelSource::open(&report.pack_path).expect("open the cooked pack");
    let built = inf_player::build_world_from_pack(&source).expect("build the world");
    (inf_player::sim_from_built(built), proj)
}

/// **Assert the WORLD before comparing two of them** (the P21.4 law): the sim
/// really projects a shadow-casting sun and the fixture's geometry.
fn assert_world(name: &str, sim: &inf_player::runtime_sim::RuntimeSim, analytic: bool) {
    let mut scene = RenderScene::default();
    inf_player::render::project_scene(
        &mut scene,
        sim,
        0.0,
        &inf_player::vmesh::VmeshRegistry::new(),
    );
    assert!(
        scene
            .lights
            .iter()
            .any(|l| l.cast_shadows && l.kind == inf_render::LightKind::Directional),
        "the {name} world projects no shadow-casting directional light"
    );
    assert!(
        scene.instances.len() >= 5,
        "the {name} world projects {} instances, not the floor, its three \
         casters and the mover",
        scene.instances.len()
    );
    if analytic {
        for kind in [inf_render::LightKind::Spot, inf_render::LightKind::Point] {
            assert!(
                scene
                    .lights
                    .iter()
                    .any(|l| l.cast_shadows && l.kind == kind),
                "the {name} world projects no shadow-casting {kind:?} light"
            );
        }
    }
}

// ══ (a) the trace is deterministic, and PIE == shipping on it ═══════════════

/// **(a) Two runs of one scripted path produce the same page-allocation trace,
/// byte for byte — and two HOSTS do too.**
///
/// The determinism half and the PIE-vs-shipping half are one arm because they
/// share a walk and each is the other's control: two runs of one host prove the
/// loop is a pure function of committed input, and two hosts prove the input is
/// the same world.
///
/// The trace is by **address**, never by slot. A slot index is an allocation
/// order; two hosts that seated the same pages in a different order describe the
/// same shadow, and comparing the integers would be comparing two correct
/// answers and calling them wrong (the P26.3 handle law, one virtual system
/// over).
///
/// **And refinement under camera approach is asserted here** — the P26.5 audit
/// routed it to this gate *"over the page-allocation trace"*, and this is that
/// trace. The finest resident level of the sun's tree must fall as the eye
/// closes, over the same walk the equality above is about.
#[test]
fn the_page_allocation_trace_is_deterministic_and_pie_equals_shipping() {
    let Some(gpu) = gpu_or_skip("the phase-27 gate's trace arm") else {
        return;
    };
    let tmp = tempfile::tempdir().expect("tempdir");
    let (proj, doc) = shadow_project(tmp.path(), true);
    let report = cook(&proj, &tmp.path().join("out"), &cook_opts()).expect("the project cooks");
    let source =
        inf_player::level::PackLevelSource::open(&report.pack_path).expect("open the cooked pack");
    let build_shipped = || {
        inf_player::sim_from_built(
            inf_player::build_world_from_pack(&source).expect("build the world"),
        )
    };
    let previewed_sim = inf_player::sim_from_payload(&payload_for(&proj, &doc))
        .expect("the payload builds a sim")
        .sim;
    assert_world("shipped", &build_shipped(), true);
    assert_world("previewed", &previewed_sim, true);

    let once = scripted(&gpu, &build_shipped(), true);
    let twice = scripted(&gpu, &build_shipped(), true);
    let previewed = scripted(&gpu, &previewed_sim, true);

    // ANTI-VACUITY FIRST, both directions. A column of empty strings agrees with
    // itself perfectly, and a still trace makes the equality a claim about one
    // frame repeated.
    assert_eq!(once.trace.len(), PATH_FRAMES as usize);
    assert!(
        once.trace.iter().filter(|l| !l.is_empty()).count() >= (PATH_FRAMES - 2) as usize,
        "the run held no resident page on most frames: {:?}",
        once.trace
    );
    assert!(
        once.trace.windows(2).any(|w| w[0] != w[1]),
        "residency never changed across {PATH_FRAMES} frames of a scripted \
         approach, so every equality below is about a still frame"
    );
    assert_eq!(once.receiver_frames, PATH_FRAMES);

    // DETERMINISM.
    assert_eq!(
        once.trace, twice.trace,
        "two runs of one scripted path produced different page-allocation traces"
    );
    assert_eq!(once.frame, twice.frame, "the frame is not deterministic");

    // PIE == SHIPPING, on the trace AND on the pixels: two hosts can resolve the
    // same pages and shade them differently.
    assert_eq!(
        once.trace, previewed.trace,
        "a cooked pack and a PIE payload made different shadow pages resident \
         for one level"
    );
    assert_eq!(
        once.frame, previewed.frame,
        "the two boot paths resolved the same pages and shaded them differently"
    );

    // …and the frame really is shadowed, or the agreement above is about two
    // identical un-shadowed pictures.
    let unshadowed = scripted(&gpu, &build_shipped(), false);
    assert_eq!(unshadowed.receiver_frames, 0);
    assert_ne!(
        once.frame, unshadowed.frame,
        "virtual shadows changed no pixel of the shipped frame"
    );

    // **REFINEMENT UNDER CAMERA APPROACH**, over this trace (the P26.5 audit's
    // item, restated AT the gate). The level rule is one shadow texel per screen
    // pixel, so closing on the casters must ask for a FINER level — a smaller
    // number — and the path is an approach for exactly this reason.
    let first = once.finest[2];
    let last = *once.finest.last().expect("frames");
    assert!(
        first != u32::MAX && last != u32::MAX,
        "the sun's tree held no resident page at an end of the walk: {:?}",
        once.finest
    );
    assert!(
        last < first,
        "the finest marked level was {first} at the start of the approach and \
         {last} at the end — it did not refine: {:?}",
        once.finest
    );
    // …and it is monotone in the direction of travel rather than noisy: no frame
    // asks for a coarser finest level than the frame before it.
    for w in once.finest[2..].windows(2) {
        assert!(
            w[1] <= w[0],
            "the finest level went BACKWARDS along an approach ({} -> {}): {:?}",
            w[0],
            w[1],
            once.finest
        );
    }
}

// ══ (b) a static scene re-rasterizes zero pages after warm-up ═══════════════

/// **(b) A static scene re-rasterizes ZERO pages after warm-up, and this arm
/// FAILS if the counter is dead.**
///
/// The ROADMAP spells the anti-vacuity requirement in the phase text itself —
/// *"a no-op cache also re-rasters zero; the counter must first prove rasters
/// happened"* — so the shape is fixed: prove the warm-up rasterized, drew and
/// packed; **then** assert the tail moves none of them; and keep a counter that
/// is still moving, so "nothing happened" and "nothing needed to happen" are
/// distinguishable.
///
/// A parked camera, not a walk: a moving camera moves every page matrix and
/// would invalidate the atlas for a reason that has nothing to do with caching.
#[test]
fn a_static_scene_re_rasterizes_zero_pages_after_warm_up() {
    let Some(gpu) = gpu_or_skip("the phase-27 gate's caching arm") else {
        return;
    };
    let tmp = tempfile::tempdir().expect("tempdir");
    let (sim, _) = shipped(tmp.path(), true);
    assert_world("shipped", &sim, true);

    let target = inf_render::HeadlessTarget::new(&gpu, W, H);
    let mut renderer = EngineRenderer::new(&gpu, inf_render::HEADLESS_FORMAT);
    renderer
        .try_set_settings(RenderSettings {
            vsm: vsm_on(),
            ..Default::default()
        })
        .expect("the gate's settings are legal");
    let vmeshes = inf_player::vmesh::VmeshRegistry::new();
    let view = path_view(PATH_FRAMES / 2);
    let frame = |renderer: &mut EngineRenderer| {
        let mut scene = RenderScene::default();
        inf_player::render::project_scene(&mut scene, &sim, 0.0, &vmeshes);
        scene.grid_enabled = false;
        scene.mark_dirty();
        renderer.render(&gpu, &scene, &view, &target.view, (W, H));
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    };

    // WARM UP, and PROVE it did the work. Eight frames: the ring is pinned at
    // `frame − 2`, so the residency needs several to fill and settle.
    for _ in 0..8 {
        frame(&mut renderer);
    }
    let warm = renderer.vsm_raster_stats().expect("a live system");
    assert!(
        warm.pages > 0,
        "the warm-up rasterized no page at all, so 'zero after warm-up' is a \
         statement about a system that never ran: {warm:?}"
    );
    assert!(warm.draws > 0, "no indirect draw was issued: {warm:?}");
    assert!(warm.casters > 0, "no caster was packed: {warm:?}");
    assert!(
        warm.cleared_pages > 0,
        "no page was cleared, so nothing was re-rasterized to be cached: {warm:?}"
    );

    // …and the atlas really holds it: the page set is non-empty WORLD state, not
    // a counter's opinion.
    let resident = renderer
        .vsm()
        .map(|s| {
            let g = s.residency().geometry();
            (0..g.slot_count())
                .filter(|i| s.residency().slot_occupant(*i).is_some())
                .count()
        })
        .unwrap_or(0);
    assert!(resident > 0, "no slot is occupied after warm-up");

    // THE CLAIM: six more frames of the identical scene from the identical eye.
    const TAIL: u64 = 6;
    for _ in 0..TAIL {
        frame(&mut renderer);
    }
    let after = renderer.vsm_raster_stats().expect("a live system");
    assert_eq!(
        after.pages,
        warm.pages,
        "a static scene re-rasterized {} page(s) over {TAIL} frames ({warm:?} \
         -> {after:?})",
        after.pages - warm.pages
    );
    assert_eq!(after.draws, warm.draws, "a static scene issued a draw");
    assert_eq!(
        after.cleared_pages, warm.cleared_pages,
        "a static scene cleared a page"
    );

    // THE COUNTER IS NOT DEAD. The cache's own counter grows by the resident set
    // every frame — which is what tells "nothing happened" from "nothing needed
    // to happen", and what a no-op cache cannot produce.
    assert!(
        after.cached_pages > warm.cached_pages,
        "the cache counter did not move over {TAIL} static frames, so the zero \
         above is indistinguishable from a system that stopped running: \
         {warm:?} -> {after:?}"
    );
    assert!(
        after.cached_pages - warm.cached_pages >= resident as u64,
        "the cache reported {} hits over {TAIL} frames with {resident} pages \
         resident — fewer than one full atlas a frame, so pages are being \
         dropped rather than cached",
        after.cached_pages - warm.cached_pages
    );
    // …and the marking loop kept running too, so the frames really happened.
    let marking = renderer.vsm().expect("a live system").stats();
    assert!(marking.frames >= 8 + TAIL, "{marking:?}");
    assert!(marking.mark_threads > 0, "{marking:?}");
}

// ══ (c) page-exact invalidation on a scripted mover ═════════════════════════

/// **(c) Moving one object invalidates exactly the pages its bounds touch.**
///
/// The mover is a **real entity** moved in the sim's world and re-projected
/// through `project_scene` — the one door — so what is measured is the shipping
/// path rather than a `RenderScene` edited by hand.
///
/// The expected set is computed **independently**: for every resident page, the
/// page's own matrix (through `VsmSystem::page_matrix`, the shipped door) is
/// tested against the mover's bounding sphere at BOTH positions with
/// `vsm_page_sees_sphere` — the same predicate the GPU cull runs, evaluated on
/// the CPU over an address set read off the residency. Then the count is
/// asserted **exactly**, and the fixture is checked to have pages the mover does
/// NOT touch, or "exactly" cannot be told from "everything".
#[test]
fn a_scripted_mover_invalidates_exactly_the_pages_its_bounds_touch() {
    let Some(gpu) = gpu_or_skip("the phase-27 gate's invalidation arm") else {
        return;
    };
    let tmp = tempfile::tempdir().expect("tempdir");
    let (mut sim, _) = shipped(tmp.path(), false);
    assert_world("shipped", &sim, false);

    let target = inf_render::HeadlessTarget::new(&gpu, W, H);
    let mut renderer = EngineRenderer::new(&gpu, inf_render::HEADLESS_FORMAT);
    renderer
        .try_set_settings(RenderSettings {
            vsm: vsm_on(),
            ..Default::default()
        })
        .expect("the gate's settings are legal");
    let vmeshes = inf_player::vmesh::VmeshRegistry::new();
    let view = path_view(PATH_FRAMES / 2);
    let frame = |sim: &inf_player::runtime_sim::RuntimeSim, renderer: &mut EngineRenderer| {
        let mut scene = RenderScene::default();
        inf_player::render::project_scene(&mut scene, sim, 0.0, &vmeshes);
        scene.grid_enabled = false;
        scene.mark_dirty();
        renderer.render(&gpu, &scene, &view, &target.view, (W, H));
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    };
    for _ in 0..8 {
        frame(&sim, &mut renderer);
    }
    let warm = renderer.vsm_raster_stats().expect("a live system");
    // A settled frame first: the count below is a DELTA, so it has to start from
    // a frame that itself rasterized nothing.
    frame(&sim, &mut renderer);
    let settled = renderer.vsm_raster_stats().expect("a live system");
    assert_eq!(
        settled.pages, warm.pages,
        "the fixture had not settled before the mover moved: {warm:?} -> \
         {settled:?}"
    );

    // The mover's world positions, and its bounding sphere's radius through
    // `pack_casters`'s own derivation.
    let mover = mover_entity(&mut sim);
    let before = mover_translation(&sim, mover);
    let after = glam::DVec3::new(before.x + MOVER_STEP, before.y, before.z);
    let radius = inf_render::PrimMesh::Cube.bounding_radius() * MOVER_SCALE as f32;

    // The independent answer, over the residency's own addresses and matrices.
    let sys = renderer.vsm().expect("a live system");
    let geom = sys.residency().geometry();
    let resident: Vec<(VsmLightHandle, VsmPage)> = (0..geom.slot_count())
        .filter_map(|s| sys.residency().slot_occupant(s))
        .collect();
    let want: BTreeSet<usize> = resident
        .iter()
        .enumerate()
        .filter(|(_, (light, page))| {
            let Some(vp) = sys.page_matrix(*light, *page) else {
                return false;
            };
            inf_render::vsm_page_sees_sphere(&vp, before.as_vec3(), radius)
                || inf_render::vsm_page_sees_sphere(&vp, after.as_vec3(), radius)
        })
        .map(|(i, _)| i)
        .collect();
    assert!(
        !want.is_empty() && want.len() < resident.len(),
        "the mover touches {} of {} resident pages — the fixture cannot tell \
         'exactly' from 'everything'",
        want.len(),
        resident.len()
    );

    // MOVE IT, in the sim, and re-project through the one door.
    {
        let world = sim.world_mut();
        let mut t = *world
            .world()
            .entity(mover)
            .get::<Transform>()
            .expect("the mover has a transform");
        t.translation.x += MOVER_STEP;
        world.world_mut().entity_mut(mover).insert(t);
        // The projector reads `GlobalTransform`, which the propagation pass
        // writes — so a local edit that is never propagated moves nothing.
        world.mark_dirty();
        world.propagate();
    }
    frame(&sim, &mut renderer);
    let moved = renderer.vsm_raster_stats().expect("a live system");
    let touched = moved.pages - settled.pages;
    assert!(
        touched > 0,
        "moving an object rasterized nothing at all: {settled:?} -> {moved:?}"
    );
    assert_eq!(
        touched,
        want.len() as u64,
        "the mover's bounds touch {} resident pages and the frame rasterized \
         {touched} ({settled:?} -> {moved:?})",
        want.len()
    );
    // …and the invalidation really was a SCATTER rather than a product: the
    // touches it walked are bounded by casters x levels x covered pages, not by
    // casters x pages.
    assert!(
        moved.invalidation_touches > 0,
        "the invalidation scatter walked nothing: {moved:?}"
    );
}

/// The mover's entity, by NAME — the one identity a cooked pack and a payload
/// both carry unchanged (a `Guid` is minted per document and an entity index is
/// per world).
fn mover_entity(sim: &mut inf_player::runtime_sim::RuntimeSim) -> inf_ecs::Entity {
    let world = sim.world_mut();
    let ids = world.entities();
    *ids.iter()
        .find(|e| world.name_of(**e) == Some(MOVER_NAME))
        .unwrap_or_else(|| panic!("no entity named {MOVER_NAME} in the world"))
}

/// The mover's world translation.
fn mover_translation(sim: &inf_player::runtime_sim::RuntimeSim, e: inf_ecs::Entity) -> glam::DVec3 {
    let t = sim
        .world()
        .world()
        .entity(e)
        .get::<Transform>()
        .expect("the mover has a transform");
    glam::DVec3::new(t.translation.x, t.translation.y, t.translation.z)
}

// ══ (d) point and spot engagement ══════════════════════════════════════════

/// **(d) The point and spot lights engage — and every counter they move is zero
/// on the shipped defaults.**
///
/// The P20 law's shape: a claim about the command stream needs an engagement
/// counter, and the moving half comes FIRST so the zero half is a comparison
/// rather than an absence of evidence.
///
/// The claim is made on the WORLD rather than on a total: the point light's
/// pages must be resident on **more than one cube face**, which is the thing a
/// cube tree does that a quadtree does not, and the spot's tree must be a
/// quadtree registered for the spot rather than a second clipmap.
#[test]
fn the_point_and_spot_trees_engage_and_are_inert_on_the_defaults() {
    let Some(gpu) = gpu_or_skip("the phase-27 gate's point/spot arm") else {
        return;
    };
    let tmp = tempfile::tempdir().expect("tempdir");
    let (sim, _) = shipped(tmp.path(), true);
    assert_world("shipped", &sim, true);

    let run = scripted(&gpu, &sim, true);
    assert_eq!(run.receiver_frames, PATH_FRAMES);

    // Rebuild one renderer to read the live system (the run consumed its own).
    let target = inf_render::HeadlessTarget::new(&gpu, W, H);
    let mut renderer = EngineRenderer::new(&gpu, inf_render::HEADLESS_FORMAT);
    renderer
        .try_set_settings(RenderSettings {
            vsm: vsm_on(),
            ..Default::default()
        })
        .expect("legal");
    let vmeshes = inf_player::vmesh::VmeshRegistry::new();
    for step in 0..PATH_FRAMES {
        let mut scene = RenderScene::default();
        inf_player::render::project_scene(&mut scene, &sim, 0.0, &vmeshes);
        scene.grid_enabled = false;
        scene.mark_dirty();
        renderer.render(&gpu, &scene, &path_view(step), &target.view, (W, H));
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    }
    let sys = renderer.vsm().expect("a live system");
    let kinds: Vec<inf_render::VsmTreeKind> = sys.trees().iter().map(|t| t.kind).collect();
    // As a SET, because the order is the projector's `Guid` walk and not the
    // document's: a handle is the n-th shadow-casting light in *scene* order and
    // scene order is whatever the ids sorted to.
    for want in [
        inf_render::VsmTreeKind::Clipmap,
        inf_render::VsmTreeKind::Quadtree,
        inf_render::VsmTreeKind::Cube,
    ] {
        assert_eq!(
            kinds.iter().filter(|k| **k == want).count(),
            1,
            "the three lights did not register exactly one {want:?}: {kinds:?}"
        );
    }
    assert_eq!(kinds.len(), 3, "{kinds:?}");
    let handle_of = |k: inf_render::VsmTreeKind| -> u32 {
        kinds
            .iter()
            .position(|t| *t == k)
            .unwrap_or_else(|| panic!("no {k:?} tree")) as u32
    };

    // Resident pages per light, and per FACE for the cube.
    let geom = sys.residency().geometry();
    let mut per_light: BTreeMap<u32, BTreeSet<(u32, u32)>> = BTreeMap::new();
    for s in 0..geom.slot_count() {
        if let Some((l, p)) = sys.residency().slot_occupant(s) {
            per_light.entry(l.0).or_default().insert((p.face, p.level));
        }
    }
    for (light, label) in [
        (handle_of(inf_render::VsmTreeKind::Quadtree), "the spot"),
        (handle_of(inf_render::VsmTreeKind::Cube), "the point"),
    ] {
        let set = per_light.get(&light).cloned().unwrap_or_default();
        assert!(
            !set.is_empty(),
            "{label} light holds no resident page — its tree is registered and \
             nothing marks it"
        );
    }
    let point_faces: BTreeSet<u32> = per_light
        .get(&handle_of(inf_render::VsmTreeKind::Cube))
        .map(|s| s.iter().map(|(f, _)| *f).collect())
        .unwrap_or_default();
    assert!(
        point_faces.len() > 1,
        "the point light's pages are all on face {point_faces:?} — a cube tree \
         that only ever marks one face is a quadtree with five wasted blocks"
    );

    // …and the receiver really names them: every light's slot is the projection
    // its own tree owns, and the sun's is the FIRST directional's.
    let slots = renderer.vsm_light_slots();
    assert_eq!(slots.len(), 3, "one slot per scene light: {slots:?}");
    assert!(
        slots.iter().all(|s| *s != 0),
        "a shadow-casting light got no slot: {slots:?}"
    );
    let params = renderer.vsm_receiver_params().expect("a live system");
    let sun_scene_index = {
        let mut scene = RenderScene::default();
        inf_player::render::project_scene(
            &mut scene,
            &sim,
            0.0,
            &inf_player::vmesh::VmeshRegistry::new(),
        );
        scene
            .lights
            .iter()
            .position(|l| l.cast_shadows && l.kind == inf_render::LightKind::Directional)
            .expect("a shadow-casting sun")
    };
    assert_eq!(
        params.counts[0], slots[sun_scene_index],
        "the receiver's sun slot is not the first shadow-casting directional \
         light's: slots {slots:?}, sun at scene index {sun_scene_index}"
    );

    // ZERO ON THE DEFAULTS. Same world, same walk, shipped settings.
    let off = scripted(&gpu, &sim, false);
    assert_eq!(off.receiver_frames, 0);
    assert_eq!(off.raster, inf_render::VsmRasterStats::default());
    assert_eq!(off.marking, inf_render::VsmStreamStats::default());
    assert!(
        off.trace.iter().all(|l| l.is_empty()),
        "a renderer with the shipped defaults made a shadow page resident"
    );
}

// ══ (e) budgets ════════════════════════════════════════════════════════════

/// **(e) The three budget classes, kept apart** (P26.5's ruling, applied to
/// shadow pages).
///
/// * **LOAD** — cooking and building the world, once, against [`LOAD_BUDGET_MS`].
/// * **WORLD** — what one frame's shadow loop DID, in pages: asserted on every
///   adapter, because a page is a page everywhere and arm (a) proves the
///   sequence is a pure function of committed input. This is the half with
///   teeth in CI.
/// * **CLOCK** — the steady-state milliseconds, against [`FRAME_BUDGET_MS`],
///   **only on an adapter whose timing represents a frame**. Printed always.
///
/// And the **sun's time-slicing drain bound** is asserted here, because this is
/// the one place both of its numbers are in scope: the atlas drains in
/// `slots / VSM_MAX_RASTER_PAGES` frames and the sun's quantum lasts
/// `frames_per_quantum`, and the ratio is what makes `VSM_MAX_RASTER_PAGES` a
/// cap on *work* rather than a cap on *quality*.
#[test]
fn the_shadow_loop_stays_inside_its_budgets() {
    let Some(gpu) = gpu_or_skip("the phase-27 gate's budget arm") else {
        return;
    };
    let info = gpu.adapter.get_info();
    let virtualized = {
        let n = info.name.to_ascii_lowercase();
        n.contains("paravirtual") || n.contains("virtualbox") || n.contains("vmware")
    };
    let software = info.device_type == wgpu::DeviceType::Cpu || virtualized;

    // ── LOAD class ──────────────────────────────────────────────────────────
    let tmp = tempfile::tempdir().expect("tempdir");
    let (proj, _) = shadow_project(tmp.path(), true);
    let report = cook(&proj, &tmp.path().join("out"), &cook_opts()).expect("the project cooks");
    let t0 = std::time::Instant::now();
    let source = inf_player::level::PackLevelSource::open(&report.pack_path).expect("open pack");
    let sim = inf_player::sim_from_built(
        inf_player::build_world_from_pack(&source).expect("build the world"),
    );
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
    assert!(
        load_ms < LOAD_BUDGET_MS,
        "building the shadowed level took {load_ms:.2} ms, over the \
         {LOAD_BUDGET_MS} ms load budget {}",
        inf_player::budget::RATCHET_NOTE
    );

    let run = scripted(&gpu, &sim, true);
    let control = scripted(&gpu, &sim, false);

    println!(
        "phase27 budgets on {} ({:?}){}:\n  \
         steady mean over {} frames {:.2} ms (no virtual shadows: {:.2} ms, \
         so shadows are {:+.2} ms of it)\n  \
         {} pages rasterized / {} cached / {} deferred; peak {} on a steady \
         frame against the {} cap (per frame: {})\n  \
         {} atlas slots, {} marking threads over the path\n  \
         budgets: frame {FRAME_BUDGET_MS} ms, load {LOAD_BUDGET_MS} ms",
        info.name,
        info.device_type,
        if software {
            " [virtual/software — the clock is reported, not asserted]"
        } else {
            ""
        },
        PATH_FRAMES - 1,
        run.steady_ms(),
        control.steady_ms(),
        run.steady_ms() - control.steady_ms(),
        run.raster.pages,
        run.raster.cached_pages,
        run.raster.deferred_pages,
        run.peak_steady_pages(),
        inf_render::VSM_MAX_RASTER_PAGES,
        run.pages_trace(),
        run.slots,
        run.marking.mark_threads,
    );

    // ── WORLD class: everywhere ─────────────────────────────────────────────
    // ANTI-VACUITY FIRST: the frames did shadow work, or every bound below is a
    // bound on an empty loop.
    assert!(
        run.raster.pages > 0 && run.raster.casters > 0,
        "the path rasterized {} pages from {} casters — nothing to bound: {:?}",
        run.raster.pages,
        run.raster.casters,
        run.raster
    );
    assert_eq!(
        run.pages_each.iter().sum::<u64>(),
        run.raster.pages,
        "the per-frame page counts do not sum to the raster's own total, so the \
         ceiling below is asserted against a bookkeeping bug"
    );
    // No frame rasterized more pages than a frame may. This is the assertion
    // that survives on every adapter: it is a measurement of the engine, not of
    // the runner.
    assert!(
        run.peak_steady_pages() <= inf_render::VSM_MAX_RASTER_PAGES as u64,
        "a frame rasterized {} pages, over the {} the caster pass is allowed \
         (per frame: {})",
        run.peak_steady_pages(),
        inf_render::VSM_MAX_RASTER_PAGES,
        run.pages_trace()
    );
    // …and the residency stayed inside the atlas it was planned with.
    assert!(run.slots > 0, "the atlas was planned with no slots");
    let occupied = run
        .trace
        .last()
        .map(|l| l.matches(';').count())
        .unwrap_or(0);
    assert!(
        occupied > 0 && occupied <= run.slots as usize,
        "{occupied} pages are resident in an atlas of {} slots",
        run.slots
    );
    // The marking pass ran at the tier's own density: one thread per pixel on
    // High, and never more than the frame has pixels.
    let per_frame = run.marking.mark_threads / PATH_FRAMES;
    assert!(
        per_frame >= (W * H) as u64 && per_frame < 2 * (W * H) as u64,
        "the marking pass dispatched {per_frame} threads a frame for a \
         {W} x {H} frame at stride {}",
        vsm_on().mark_stride
    );

    // ── THE SUN'S DRAIN BOUND ───────────────────────────────────────────────
    // The stability condition the P27.3 policy rests on: the dirty drain is
    // ELEVEN times faster than the quantum that fills it. Both numbers are in
    // scope here and nowhere else — the quantum is a settings function and the
    // drain is the live atlas over the frame cap.
    let set = vsm_on();
    let quantum = inf_render::vsm_sun_quantum(&set);
    let quanta_per_rev = std::f64::consts::TAU / f64::from(quantum);
    // A twenty-minute day at 60 fps — the P27.3 memo's own reference.
    let frames_per_quantum = (20.0 * 60.0 * 60.0) / quanta_per_rev;
    // A CEILING: 1 024 slots at 256 a frame is four frames, not five. (The
    // first draft wrote `floor + 1` and reported 8.95, which is the same
    // arithmetic error one place over — the ratio is the whole argument for
    // `VSM_MAX_RASTER_PAGES`, so it is worth getting exactly right.)
    let drain_frames = (f64::from(run.slots) / f64::from(inf_render::VSM_MAX_RASTER_PAGES)).ceil();
    let ratio = frames_per_quantum / drain_frames;
    println!(
        "phase27 drain: quantum {quantum:.6} rad, {quanta_per_rev:.1} quanta a \
         revolution, {frames_per_quantum:.2} frames a quantum, {drain_frames} \
         frames to drain {} slots at {} a frame, ratio {ratio:.2}",
        run.slots,
        inf_render::VSM_MAX_RASTER_PAGES,
    );
    assert!(
        (10.0..13.0).contains(&ratio),
        "the drain is {ratio:.2} times faster than the sun's quantum, not the \
         ~11 the policy rests on — `VSM_MAX_RASTER_PAGES` has stopped being a \
         cap on WORK"
    );

    // ── CLOCK class: only where a millisecond represents a frame ─────────────
    if software {
        return;
    }
    let steady = run.steady_ms();
    assert!(
        steady < FRAME_BUDGET_MS,
        "a shadowed frame cost {steady:.2} ms on {}, over the \
         {FRAME_BUDGET_MS} ms frame budget {}",
        info.name,
        inf_player::budget::RATCHET_NOTE
    );
}

// ══ the ≥8k-effective claim, and the tier flip ═════════════════════════════

/// **The phase's own "≥8k effective on High", measured** (P27.5).
///
/// *Effective resolution* is the virtual extent a receiver can address at its
/// finest level: `pages a side × VSM_PAGE_SIZE`. It is measured off the **live
/// tree** rather than off a settings literal, on a configuration produced by
/// `RenderTier::High.apply` — so the number is the one a High machine gets.
///
/// The texel density is the other half and it is what makes "effective" mean
/// anything: at the shipped level-0 extent, one shadow texel is **7.8 mm** under
/// the camera. An equivalent single shadow map would be 8192² of them, which is
/// 268 MB of `Depth32Float` against the atlas's 64 MiB — the sentence the whole
/// phase exists to make true, as arithmetic.
///
/// And the tier ladder is asserted with it, because ≥8k is a **High** claim: the
/// clamp takes Medium to half of it, which is the honest reading of "High/Medium
/// run VSM".
#[test]
fn the_high_tier_addresses_at_least_eight_thousand_shadow_texels() {
    let Some(gpu) = gpu_or_skip("the phase-27 gate's resolution arm") else {
        return;
    };
    let tmp = tempfile::tempdir().expect("tempdir");
    let (sim, _) = shipped(tmp.path(), false);
    let run = scripted(&gpu, &sim, true);

    // MEASURED, off the live tree the renderer built.
    let effective = run.finest_pages_per_side * inf_render::VSM_PAGE_SIZE;
    assert!(
        effective >= 8192,
        "High tier addresses {effective} shadow texels a side ({} pages of {}), \
         short of the 8 192 the phase claims",
        run.finest_pages_per_side,
        inf_render::VSM_PAGE_SIZE
    );

    // The density that makes it mean something, and the comparison the phase is
    // built on.
    let set = vsm_on();
    let texel_m = 2.0 * set.first_level_extent_m / effective as f32;
    let equivalent_bytes = u64::from(effective) * u64::from(effective) * 4;
    println!(
        "phase27 resolution: {} pages a side x {} texels = {effective} virtual \
         texels, {:.2} mm a texel over a +/-{} m level 0; an equivalent single \
         shadow map is {:.0} MB against a {} MiB atlas",
        run.finest_pages_per_side,
        inf_render::VSM_PAGE_SIZE,
        texel_m * 1000.0,
        set.first_level_extent_m,
        equivalent_bytes as f64 / 1e6,
        set.budget_bytes / (1024 * 1024),
    );
    assert!(
        texel_m < 0.01,
        "a level-0 shadow texel is {:.2} mm — 'high' is a claim about density \
         as well as about extent",
        texel_m * 1000.0
    );
    assert!(
        equivalent_bytes > set.budget_bytes * 3,
        "the equivalent single shadow map is {equivalent_bytes} B against a \
         {} B atlas; virtualizing is buying nothing",
        set.budget_bytes
    );

    // THE TIER LADDER: >=8k is a HIGH claim, and the clamp really lowers it.
    let asked = RenderSettings {
        vsm: VsmSettings {
            enabled: true,
            ..Default::default()
        },
        ..Default::default()
    };
    let high = RenderTier::High.apply(asked).vsm;
    let medium = RenderTier::Medium.apply(asked).vsm;
    let low = RenderTier::Low.apply(asked).vsm;
    assert_eq!(high.clipmap_pages_per_side, run.finest_pages_per_side);
    assert!(
        medium.clipmap_pages_per_side * inf_render::VSM_PAGE_SIZE < 8192,
        "Medium addresses the same 8k as High, so the tier ladder is flat"
    );
    assert!(high.enabled && medium.enabled, "High/Medium run VSM");
    assert!(!low.enabled, "Low did not fall back to the cascaded path");
}

// ══ (f) the golden set ═════════════════════════════════════════════════════

/// **(f) The golden set is pinned, and it is additive only** (P27.5).
///
/// Phase 27 renders shadows through a page atlas for the first time and it
/// re-blessed nothing: `VsmSettings::enabled` is `false` by default, so a
/// default frame binds the renderer's permanent EMPTY shadow surface and every
/// receiver's arithmetic is an exact `* 1.0`. P27.4 added four goldens and
/// P27.5 added none.
///
/// Pinned by CONTENT as well as by count, for the reason `phase26_gate` measured
/// (both halves of "a count is enough" are backwards): `check_golden`'s pixel
/// comparison is opt-in, and the harness WRITES any golden it cannot read, so a
/// deletion is a silent re-bless. The digest fails on an edited frame, on a
/// re-bless, and on a delete-then-regenerate whenever the regenerated frame
/// differs.
///
/// Deliberately a **second** copy of `phase26_gate`'s constants rather than a
/// shared helper: the two gates are two claims about the same directory made by
/// two phases, and a shared constant would let one phase's re-bless satisfy the
/// other's pin.
#[test]
fn the_golden_set_is_pinned_and_additive_after_phase_27() {
    /// 50 through P26.5, plus P27.4's four virtual-shadow frames
    /// (`vsm_directional`, `vsm_spot`, `vsm_point`, `vsm_bias_grazing`).
    /// **P27.5 adds none**: every claim it makes is asserted as state or as a
    /// counter, and the one new pixel surface it ships — the shadow-page view
    /// mode — is a debug overlay, which is not a thing to freeze.
    ///
    /// Wave VIS1a's **audit** adds the fifty-fifth, `water_ssr.png`: a new frame,
    /// so this pin and the digest below move together on the additive branch and
    /// no committed image changed. `phase26_gate`'s twin carries the reason.
    const GOLDENS: usize = 62;
    /// `xxh3_128` over `"{file_name} {hex}\n"` for every golden, name-sorted.
    /// **RULE: this may change only in a commit that adds a golden, or in one
    /// whose stated purpose is to change what the engine LOOKS like.**
    ///
    /// **Moved once, at wave SKY2** (from `23d41a61c31c28a17a20871b6c875707`),
    /// the same move `phase26_gate`'s twin records at length — eight cloud-
    /// carrying frames re-blessed on purpose by the volumetric-cloud overhaul,
    /// count unchanged. That both pins had to move separately is the point of
    /// their being separate: a wave that re-blessed for a bad reason would have
    /// to write the reason down twice.
    ///
    /// **Wave NPC1b adds the fifty-ninth**: `crowd_variation.png` — eight bodies
    /// on ONE skinned mesh sharing ONE joint palette, each with its own tint and
    /// its own build, drawn in a single instanced call from a single atlas block.
    /// Additive: `GOLDENS` moves 58 -> 59, the digest moves with it, and **not one
    /// committed image changed** — the instanced path draws a one-mesh scene in
    /// the order it drew it before, which is what the stable sort in
    /// `plan_skinned_batches` is for.
    ///
    /// **Wave GTA1 moves it a seventh time, and this move is BOTH branches of the
    /// rule at once** (from `1db3dd0bf5961ddbfb53aeb2b40a1697`), which is why the
    /// commits are separate and each says which it is:
    ///
    ///  * **additive** — `sky_night_horizon.png`, the frame the night sky's own
    ///    defect lived in. `golden_sky_night` pitches 35 deg up and looks AWAY
    ///    from the sun and bounds only blue, so the horizon band facing a
    ///    below-horizon sun was outside every committed frame in this set.
    ///    `GOLDENS` moves 59 -> 60 beside it.
    ///  * **the look** — `clouds_night.png` and `sky_night.png`, re-blessed with
    ///    the stated purpose that the committed frames depicted the defect: the
    ///    transmittance LUT's horizon-tangent texel leaking under the ground made
    ///    a midnight cloud deck glow at R/G **20.88** (top-half mean
    ///    `[0.27609, 0.01323, 0.01039]` -> `[0.00656, 0.00656, 0.00656]`, diff
    ///    mean 0.0835 / max 0.5064) and the night sky red-biased (`[0.00975,
    ///    0.00802, 0.00941]` -> `[0.00672, ...]`, diff mean 0.0057 / max 0.0500 —
    ///    inside tolerance, and re-blessed anyway, because a frame that quietly
    ///    depicts an engine that no longer exists is one nobody can read a
    ///    regression off).
    ///
    /// **Day is untouched and it is measured**: `sky_noon`, `sky_dawn` and
    /// `sky_dusk` compare at mean 0.000000 / max 0.000000, because the new
    /// visibility factor is exactly 1.0 while the sun is up.
    ///
    /// # Wave IASSET2: `ground_close.png`, re-blessed on the same precedent
    ///
    /// The ground library's normal and detail maps moved from **BC1 to BC5**,
    /// which is the wave's content clause and is measured on those exact bytes:
    /// BC1's worst per-channel error on the two channels a tangent-space normal
    /// lives in is **122 of 255** against BC5's **17** (mean 10.4 -> 1.7). A
    /// normal that wrong does not shade approximately wrong, it faces somewhere
    /// else.
    ///
    /// `ground_close.png` is the one committed frame that depicts those maps.
    /// Its fixture called `TextureCompression::Bc1` for all four slots by hand;
    /// it now calls `inf_material::ground`'s own per-slot settings, so "the real
    /// committed content" stays true rather than remembered. The frame moved by
    /// **mean 0.001212 / max 0.024941** — inside the harness's tolerance, and
    /// re-blessed anyway, for the reason the paragraph above gives verbatim: a
    /// frame that quietly depicts an engine that no longer exists is one nobody
    /// can read a regression off.
    ///
    /// **The albedo and ORM maps did NOT move**, and that is measured too: BC7
    /// would take the albedo's worst error from 11 of 255 to 5 and the ORM's
    /// from 45 to 33, for twice the page bytes — half of what a 24 MiB atlas arm
    /// holds. `GOLDENS` does not move; the set is still 60 files.
    /// **Wave VEN1a (2026-09-02) — the ADDITIVE branch, twice.** Two goldens
    /// join the set and **no existing frame was re-blessed**.
    ///
    /// `gi_scatter_neon.png`: a scatter batch of emissive plates lighting a
    /// white floor through the GI volume under a zero-radiance sun. It exists
    /// because `passes::gi` staged instances, skinned meshes and vgeom and never
    /// **scatter** — so a grammar-built venue's neon, string lights and lit
    /// panes were drawn and bounced nothing. Nothing could have moved with it:
    /// the four `gi_*` goldens hold no scatter and the two `scatter_*` goldens
    /// run with GI off, measured under `INF_GOLDEN_STRICT=1` before the pin did.
    ///
    /// `venue_interior.png`: the wave's visual claim as one frame — a **closed**
    /// near-black room lit only by practicals, with a red key and a blue rim
    /// over a plank stage, a chrome pole (`metallic 1.0, roughness 0.12`)
    /// standing in the wash, a magenta neon plate throwing a halo onto the
    /// boards, and a festoon chain. Its far corner sits at **4.2 of 255** and
    /// its stage at 45; with the room's roof and side walls removed the corner
    /// was 40, which is a frame lit by daylight with some neon in it.
    ///
    /// `GOLDENS` moves 60 -> 62.
    // Wave EDIT1 moved this ONCE, with a stated purpose and a described
    // difference: `gi_specular.png` was re-blessed when the GI ambient stopped
    // being irradiance spent as radiance. The other 61 frames are byte-identical
    // -- four of the five GI goldens hold their images through the
    // `GI_LAMBERT_PI` compensation in `golden.rs`, and `gi_specular` is the one
    // a single multiplier cannot hold because EDIT1 removed a pi from two
    // places and its subject rides both. Mean 0.072099, max 0.242510; every
    // structural assertion it exists for passes on the new frame.
    //
    // **Wave FIX3 moves it again** (from `1a03e55866b2b76958573b4050651d98`), on
    // the re-bless branch of the rule, and the wave's stated purpose IS the look:
    // the ambient term every shaded surface in the engine spends stops being the
    // GI probe field alone and becomes the sky's own irradiance, projected from
    // the P17 medium, with the probe field added as its signed difference from an
    // open sky. Measured before it: a white wall's shaded face read 0.0129 of its
    // sunlit face under a clear noon sky, against the 0.1-0.2 physics gives; the
    // island's Play frame had a third of the hero below luminance 8.
    //
    // **SIX frames move and all six are GI-on scenes**, which is the gate -- the
    // term is reached only through `GiSettings::enabled`, so the other 56 run the
    // instruction stream they always did and are byte-identical:
    //
    //   gi_bleed         mean 0.1558 / max 0.4919   more colour bleed: the red
    //                    wall is lit by the sky now, so it has light to bleed
    //   gi_terrain       0.0715 / 0.2930            the bounce carries a cosine
    //   gi_specular      0.0536 / 0.2940            the sky reflects again
    //   gi_emissive      0.0495 / 0.3170
    //   venue_interior   0.0426 / 0.3347            the room got DARKER (its far
    //                    corner 4.2 -> 2.0): the probe field's `-sky` term takes
    //                    back exactly the sky a closed room blocks
    //   gi_scatter_neon  0.0261 / 0.2783
    //
    // The last four are inside the harness's perceptual tolerance and are
    // re-blessed anyway, for the reason `golden.rs` states: a frame that no
    // longer depicts what the engine draws is a frame nobody can read a
    // regression off. `GOLDENS` does not move. The whole table, with the arms'
    // own before/after numbers, is in `docs/memos/island-progress.md` under
    // *Wave FIX3*.
    //
    // **The FIX3 AUDIT moves it once more** (from
    // `f7310588fca598ba368e4d2aaa89dc83`), on the same re-bless branch of the
    // same rule, and again the stated purpose IS the look. The paragraph above
    // says the `-sky` term "takes back exactly the sky a closed room blocks".
    // Measured, it did not: `inf-render/tests/interior_ambient.rs` photographs a
    // SEALED 24x24x12 m hall -- no door, no window, the sun outside -- and its
    // three faces summed to **0.4307** against a DOORED hall's 0.4348, i.e.
    // **99 %** of the light of a room that has an opening. Two mechanisms, both
    // closed by the audit:
    //
    //   * the probe march lit every surface it hit with the UNOCCLUDED sky, so
    //     an interior wall bounced daylight it cannot see. The sky half of that
    //     bounce is now scaled by the probe's own upward sky-view factor, which
    //     is 0 in a sealed room and 1 on open ground.
    //   * a probe standing INSIDE a wall or a floor slab still voted in the
    //     trilinear blend -- 87 % of the weight for the shaded face
    //     `sky_ambient.rs` is about -- and the blend reached through ceilings.
    //     Buried probes no longer vote and the blend carries the receiver's own
    //     normal.
    //
    // The sealed hall now reads **0.0004** against the doored hall's 0.0059, and
    // the outdoor reading the wave was written for holds and improves:
    // shaded:sunlit **0.1838 -> 0.2151** against a physical 0.1-0.35, with the
    // far row unchanged at 0.2867 and the furnace unchanged at 1.0096.
    //
    // **THREE frames move**, all of them GI-on:
    //
    //   venue_interior   mean 0.0189 / max 0.4381   stage 45.50 -> 40.35, and
    //                    the far corner is UNMOVED at 1.97
    //   gi_emissive      0.0629 / 0.1567            the bar's green bleed nearly
    //                    doubles, green/red 1.996 -> 3.371
    //   gi_scatter_neon  0.0369 / 0.3564
    //
    // …and THREE MORE, on a second pass, because the first write-up of this
    // block said `gi_bleed`, `gi_terrain` and `gi_specular` were "byte-identical
    // through it" and they were not. Their committed PNGs had not moved, which
    // is a different sentence: the RENDER moved, by mean 0.0083 / 0.0095 /
    // 0.0060 — under the harness's perceptual tolerance, so the strict run was
    // green and the claim went unmeasured. Measured on this adapter, all six
    // GI goldens read **0.000000** against their frames at `07f862f9` and three
    // of them did not any more. They are re-blessed on the same branch of the
    // same rule, for the reason `golden.rs` states: a frame that no longer
    // depicts what the engine draws is a frame nobody can read a regression off.
    //
    //   gi_bleed         mean 0.0083 / max 0.0452   near-wall red/green 1.462
    //                    -> 1.479
    //   gi_terrain       0.0095 / 0.0380            green drop 9.85 held; the
    //                    RED drop went +5.07 -> -2.46, i.e. the red ground now
    //                    brightens the wall's red instead of darkening it less
    //   gi_specular      0.0060 / 0.0478            grazing floor 194.5 -> 196.5
    //                    against a flat control 59.3 -> 57.8
    //
    // And the count above it: **ELEVEN** frames of the other 56 carry a
    // pre-existing adapter diff, not thirteen. Measured at `e8451338`,
    // `07f862f9` and here, the same eleven appear with the same means to six
    // decimals (cave_mouth, deform, ground_close, terrain, terrain_lod,
    // terrain_splat and the five water frames). The wave's thirteen counted
    // `gi_bleed` (0.044445) and `gi_terrain` (0.028975) among them, which are
    // not adapter diffs at all: they are the wave's own move, and they are the
    // very numbers its own table reports as the "before" of those two frames.
    const GOLDEN_SET_DIGEST: &str = "cb6c4704b2298cbe0d18729a3d251e29";
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("crates")
        .join("inf-render")
        .join("tests")
        .join("goldens");
    let mut pngs: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("read {}: {e}", dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "png"))
        .collect();
    pngs.sort();
    assert_eq!(
        pngs.len(),
        GOLDENS,
        "the golden set is {} files, not {GOLDENS}. ADDING one is allowed and \
         means moving both constants in the same commit; losing one is not.",
        pngs.len()
    );
    let mut manifest = String::new();
    for p in &pngs {
        let bytes = std::fs::read(p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
        assert!(bytes.len() > 256, "{} is not a golden", p.display());
        manifest.push_str(&format!(
            "{} {}\n",
            p.file_name().expect("a file name").to_string_lossy(),
            ContentHash::of(&bytes).to_hex()
        ));
    }
    assert_eq!(
        ContentHash::of(manifest.as_bytes()).to_hex(),
        GOLDEN_SET_DIGEST,
        "the golden set's CONTENT moved during Phase 27. A golden was \
         re-blessed, deleted (the harness regenerates any golden it cannot \
         read, so the count above stays {GOLDENS}), or replaced.\n{manifest}"
    );
    // …and the four Phase 27 frames are IN it by name, so "additive" is a claim
    // about this phase's own additions rather than about a number.
    let names: BTreeSet<String> = pngs
        .iter()
        .map(|p| p.file_name().expect("name").to_string_lossy().into_owned())
        .collect();
    for want in [
        "vsm_directional.png",
        "vsm_spot.png",
        "vsm_point.png",
        "vsm_bias_grazing.png",
    ] {
        assert!(names.contains(want), "{want} is not in the golden set");
    }
}
