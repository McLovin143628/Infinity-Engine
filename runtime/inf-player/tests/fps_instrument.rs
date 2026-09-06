//! **THE FPS INSTRUMENT** (island wave I4) — what "≥ 60 fps" is allowed to mean
//! in this repository.
//!
//! Before this file the AAA-readiness certification's complaint was exact: *"the
//! only GPU frame harness renders 640 × 360… no test in this repo measures fps at
//! a shipping resolution, so '≥ 60 fps' for the island has no existing
//! instrument."* Every frame number the tree carried was a **mean CPU wall
//! clock** around 484 lit cubes at a resolution nobody ships.
//!
//! # What this measures
//!
//! One scene, composed of the heaviest things the engine already ships, at the
//! two resolutions a desktop title ships at:
//!
//! * the **phase-30 city** — 100 banded `PcgVolume` blocks, 1 000 grammar
//!   buildings, 370 468 solids, a real road mesh through the GIS import door;
//! * a **streamed terrain** beneath it, paging its render cut as the camera moves
//!   (flat at zero, so the composed level is *the same city* wave I3 measured —
//!   see `island_frame_source_png`);
//! * the **phase-29 wizard character**, skinned, with a live state machine;
//! * the sun, and the render settings a **shipped player** builds for this level
//!   on this adapter — through `inf_player::render::shipped_settings`, the one
//!   door, rather than a configuration a harness invented.
//!
//! and it drives them the way a frame is actually driven: a fixed step, a render
//! terrain sync, a full re-projection (`scene.mark_dirty()` every frame, which is
//! the *churning* regime Hardening Wave E proved is the only honest one), and a
//! camera that moves down the middle street so nothing is cached from last frame
//! because nothing moved.
//!
//! # What this frame does NOT draw (the I4 audit)
//!
//! The shipped settings for a level that authors no render block leave
//! **shadows, GI, VSM, TAA, SSAO, bloom and the visbuffer all off** —
//! `RenderSettingsRecord::default()` for five of them, `VsmSettings::default()`
//! for VSM, the tier for the visbuffer. So the headline below is honest about
//! what a shipped player draws today and is **not** a lit AAA frame, and the
//! audit's addition is that the harness says so in its own output and then
//! measures the difference: `THE STACK'S PRICE` runs the same content at 1080p
//! with the authorable half turned on through the same `shipped_settings` door,
//! and prints it — with, since island wave I4b, the same CPU-stage and per-pass
//! tables the shipped configuration gets, because a configuration whose price is
//! one number cannot be optimised.
//!
//! Measured on an RTX 4070 Ti, MIN of rounds, **after wave I4b**: lit p95
//! 38.1-41.8 ms against 15.8-19.5 as shipped, GPU frame 16.1-16.5 against
//! 2.9-6.0, and the pipelined estimate **16.4-16.5 ms lit (60.7-60.9 fps)**.
//! Wave I4 measured the same two configurations at **92.3-92.9 lit against
//! 43.7-44.0, GPU frame 35.8-36.1 against 17.3-19.4**.
//!
//! **Two of those movements are not the engine's**, and the file says so where
//! it prints them: the unlit GPU frame fell further than any change to the unlit
//! path can explain, because I4's frame left the GPU idle two thirds of every
//! frame and a card that is idle downclocks. A GPU millisecond is a measurement
//! of the device *in the state the frame put it in*, so GPU columns are only
//! comparable between runs whose CPU frames are comparable.
//!
//! # How it is measured
//!
//! * **A whole discarded pass** runs first — pipelines compile, the terrain's
//!   render cut converges, the scatter payloads seat.
//! * **p50 / p95 / p99** of the per-frame CPU wall clock, plus per-pass GPU
//!   milliseconds from `inf_render::timing` — the query-set clock this wave
//!   built, because a whole-frame number cannot say where the frame went.
//! * **MIN of rounds**: several independent rounds, and the round with the lowest
//!   p50 is the one reported. A shared machine's slow round is a statement about
//!   the machine; the fastest round is the closest this can get to a statement
//!   about the engine. (`inf-anim`'s `inertialization` harness is the precedent.)
//!   It also means the headline is the *best* of `ROUNDS`, so the printed
//!   distance from 60 fps is a lower bound on the distance; every round's
//!   percentiles are printed beside it.
//! * **The per-stage tables are MEANS and the headline is a PERCENTILE**, which
//!   the output now says on the line above the table (the I4 audit). The CPU
//!   stages are asserted to tile the round's own **mean** frame, so a cost that
//!   sits in no stage — as the timestamp readback did — is a red arm rather than
//!   an unexplained few milliseconds between two tables.
//!
//! # Where it asserts
//!
//! **Nowhere on CI, by name.** The two ceilings this file introduces are wall
//! clocks, and `inf_player::budget`'s header states the law: *"prefer a budget in
//! a unit the machine cannot inflate, and when only a clock will do, condition it
//! the way the rest of the tree already does."* A shared virtualized runner has
//! no GPU worth timing and preempts one leg and not another. So CI **reports**
//! every number in this file and asserts none of them; a real adapter on a real
//! machine — and the I9 certification — asserts.
//!
//! What IS asserted everywhere, unconditionally, because none of it is a clock:
//! the composed scene really carries the city, the terrain and the character; the
//! ground under the city changes not one building; and the per-pass report names
//! every pass the renderer built.

use std::path::{Path, PathBuf};

use glam::{DVec3, Vec3};
use inf_asset::PackReader;
use inf_ecs::components::{Guid, PcgVolume, SkeletalMesh, Terrain};
use inf_editor_core::samples;
use inf_math::FloatingOrigin;
use inf_packager::{cook, CookOptions};
use inf_player::budget::{
    CITY_STEP_BUDGET_MS, RATCHET_NOTE, ROAD_TRIANGLES_CEILING, SHIPPING_FRAME_BUDGET_MS,
    SHIPPING_FRAME_CEILING_MS, SHIPPING_FRAME_P99_CEILING_MS,
};
use inf_player::level::PackLevelSource;
use inf_player::render::{project_scene_full, shipped_settings, sync_voxel_store};
use inf_player::runtime_sim::RuntimeSim;
use inf_project::ProjectManifest;
use inf_render::{
    EngineRenderer, GpuContext, HeadlessTarget, RenderScene, RenderView, HEADLESS_FORMAT,
};

/// The two resolutions a desktop title ships at. 1080p is the floor the
/// certification asked for; 1440p is the second point, because a frame that is
/// fragment-bound and a frame that is draw-bound scale differently and one
/// resolution cannot tell them apart.
const RESOLUTIONS: [(u32, u32, &str); 2] = [(1920, 1080, "1080p"), (2560, 1440, "1440p")];

/// Frames measured per round — one pass of the scripted flythrough.
const FRAMES: usize = 120;
/// Independent rounds; the one with the lowest p50 is reported (MIN-of-rounds).
///
/// **Every round replays the identical camera sequence** (`step` restarts at 0),
/// and a whole discarded pass runs first. The first version of this harness let
/// `step` run on across rounds, so each round flew a *different* stretch of the
/// city and MIN-of-rounds picked the cheapest stretch rather than the least
/// disturbed round. A MIN over samples of different things is not a minimum, it
/// is a selection, and that is why the reset is here.
///
/// *The 28.3 ms the wave attributed to its absence does not reproduce* (the I4
/// audit re-ran the mutation): `CITY_DRIVE_STEP_M` is 0.25 m, so a 120-frame
/// round is **thirty metres** and every round is inside the same district —
/// removing the reset moves the p50 by about 1 %, to 39.104 ms. The discipline is
/// right and is kept; the figure was retired.
const ROUNDS: usize = 3;

/// Steps discarded before the fixed step's own breakdown is measured (island
/// wave I4b) — the `FRAMES`-long discarded pass, one processor over: the band
/// seats, the terrain tiles mesh, and every `structure_stamps` miss there is
/// happens in here.
const STEP_WARMUP: usize = 120;
/// Steps per profiling round.
const STEP_SAMPLES: usize = 120;
/// Independent profiling rounds; the cheapest by the step's own total is
/// reported (MIN-of-rounds, the instrument's discipline).
const STEP_ROUNDS: usize = 3;

/// The CPU stages one frame is split into, in the order the frame runs them.
///
/// A frame that is CPU-bound and cannot say WHERE is the same defect the
/// certification found on the GPU side, one processor over — so the instrument
/// splits the wall clock at the four seams a host actually has.
/// The last stage is **the instrument's own overhead**, and it is here for the
/// reason the GPU segments tile the GPU frame: a breakdown whose parts do not add
/// up to the whole it sits beside is a breakdown of a frame nobody measured. The
/// timestamp readback (`gpu_timings`, a `map_async` + a poll) happens inside the
/// wall clock that produces p50/p95/p99, and a shipped frame does not pay it. It
/// is measured, printed, and subtracted by the reader rather than left as an
/// unnamed residue between a 37.8 ms stage table and a 39.9 ms p50 — which is
/// what the first version of this file left. (Added by the I4 audit.)
const CPU_STAGE_NAMES: [&str; CPU_STAGES] = [
    "sim fixed step",
    "stream sync",
    "projection",
    "render (record)",
    "poll (GPU wait)",
    "timing readback",
];
const CPU_STAGES: usize = 6;

// ── the fixture ─────────────────────────────────────────────────────────────

/// Scaffold a project holding the composed instrument level and cook it.
///
/// The city's **assets** are copied and its level is not: the instrument writes
/// its own `.inf_lvl` (city + ground + character), and two startup levels in one
/// project would make the cook choose.
fn cook_instrument(tmp: &Path) -> PathBuf {
    let proj = tmp.join("proj");
    ProjectManifest::new("Island Frame", "blank-3d")
        .save(&proj)
        .expect("the manifest saves");
    let content = proj.join("Content");
    std::fs::create_dir_all(&content).expect("mkdir Content");

    let city = samples::city_dir();
    for f in [
        "CityBlock.inf_pcg",
        "CityBlock.inf_pcg.toml",
        "CityRoads.inf_mesh",
        "CityRoads.inf_mesh.toml",
    ] {
        std::fs::copy(city.join(f), content.join(f))
            .unwrap_or_else(|e| panic!("copy the city's {f}: {e}"));
    }
    let hero = samples::phase29_locomotion_dir();
    for f in samples::island_frame_character_files() {
        std::fs::copy(hero.join(f), content.join(f))
            .unwrap_or_else(|e| panic!("copy the character's {f}: {e}"));
    }
    samples::write_island_frame_terrain(&content).expect("the ground imports");
    samples::write_island_frame_level(&content).expect("the level saves");

    let out = tmp.join("out");
    cook(&proj, &out, &CookOptions::default()).expect("the instrument scene cooks");
    out
}

/// The pack's world, its sim, and the render stores a shipped player resolves
/// against — assembled exactly as `inf_player::load_render_assets` does, from
/// **one** `Arc<PackReader>` (the P18.2 rule).
struct Fixture {
    sim: RuntimeSim,
    vmeshes: inf_player::vmesh::VmeshRegistry,
    skinned: inf_player::skinned::SkinnedRegistry,
    voxel_assets: inf_player::voxel::VoxelRegistry,
    /// Wave TER2b: the authored meshes scattered ground cover draws. Assembled
    /// from the same reader as everything else, through the same door
    /// `load_scatter_meshes` calls at boot -- so the frame this instrument times
    /// is the frame a shipped run draws, cover meshes and all.
    scatter_meshes: inf_render::ScatterMeshes,
    record: inf_scene::RenderSettingsRecord,
    materials: std::sync::Arc<inf_player::MaterialContent>,
}

fn open(pack: &Path) -> Fixture {
    let source = PackLevelSource::open(pack).expect("the pack opens");
    let built = inf_player::build_world_from_pack(&source).expect("the pack world builds");
    // R-P4: the level's own render block, captured before the world is consumed
    // — exactly where `inf_player::run_windowed` captures it.
    let record = built.render;
    let materials = std::sync::Arc::new(source.material_content());
    let reader = std::sync::Arc::new(
        PackReader::open(&pack.join(inf_player::level::PACK_FILE)).expect("the pack maps"),
    );
    let skinned = inf_player::skinned::SkinnedRegistry::from_pack(reader.clone());
    let voxel_assets = inf_player::voxel::VoxelRegistry::from_pack(reader.clone());
    let scatter_meshes = inf_player::scatter_mesh::from_pack(&reader);
    let vmeshes = inf_player::vmesh::VmeshRegistry::from_pack(reader)
        .expect("the pack's derived meshlet DAGs index");
    Fixture {
        sim: inf_player::sim_from_built(built),
        vmeshes,
        skinned,
        voxel_assets,
        scatter_meshes,
        record,
        materials,
    }
}

/// Register the level's virtual textures on `renderer` — `PlayerRenderHost::rebuild_vt`'s
/// body, through the same `inf_render::build_vt_level` door both hosts call, so
/// the instrument's frame samples textures the way a shipped frame does.
fn bind_virtual_textures(gpu: &GpuContext, renderer: &mut EngineRenderer, fx: &Fixture) -> usize {
    let mats = fx.materials.vt_materials();
    if mats.is_empty() {
        renderer.set_vt_level(None);
        return 0;
    }
    let budget = renderer.settings().vt.budget_bytes;
    let materials = fx.materials.clone();
    match inf_render::build_vt_level(
        &gpu.device,
        &gpu.queue,
        renderer.settings(),
        budget,
        &mats,
        |g| materials.source(g),
    ) {
        Some((textures, pools, report)) => {
            renderer.set_vt_level(Some((textures, pools)));
            report.textures
        }
        None => {
            renderer.set_vt_level(None);
            0
        }
    }
}

/// The flythrough: the camera rides the city's own scripted drive line, at eye
/// height, looking east down the middle street.
///
/// Scripted rather than free, for the phase-16 gate's reason — the frame
/// sequence has to be a function of the level alone, or two runs measure two
/// different worlds. Moving rather than parked, for
/// `frame_budget.rs::frame_stays_under_budget_under_version_churn`'s reason: a
/// still camera measures a frame this engine never draws.
fn fly(step: u64, width: u32, height: u32) -> RenderView {
    let p = samples::city_drive_point(step);
    let eye = DVec3::new(p.x, 2.2, p.z);
    RenderView {
        origin: FloatingOrigin::new(DVec3::ZERO),
        eye_world: eye,
        forward: Vec3::new(1.0, -0.08, 0.0).normalize(),
        up: Vec3::Y,
        fov_y: 70f32.to_radians(),
        near: 0.05,
        width,
        height,
        ortho: None,
    }
}

// ── statistics ──────────────────────────────────────────────────────────────

/// One round's frame-time distribution, in milliseconds.
#[derive(Debug, Clone, Copy)]
struct Round {
    p50: f64,
    p95: f64,
    p99: f64,
    worst: f64,
    /// The **mean** of the same frames. Carried beside the percentiles because
    /// every per-stage number this file prints is a mean and every headline
    /// number is a percentile, and reading one against the other is only honest
    /// if both are on the page. It is also what makes the stage table's tiling
    /// assertion possible.
    mean: f64,
}

/// Nearest-rank percentile over a sorted sample — the definition that always
/// names a frame that actually happened, rather than interpolating one that did
/// not.
fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = (q * sorted.len() as f64).ceil().max(1.0) as usize;
    sorted[rank.min(sorted.len()) - 1]
}

fn round_of(mut frames: Vec<f64>) -> Round {
    frames.sort_by(f64::total_cmp);
    let mean = match frames.is_empty() {
        true => 0.0,
        false => frames.iter().sum::<f64>() / frames.len() as f64,
    };
    Round {
        p50: percentile(&frames, 0.50),
        p95: percentile(&frames, 0.95),
        p99: percentile(&frames, 0.99),
        worst: *frames.last().unwrap_or(&0.0),
        mean,
    }
}

/// Is this adapter one whose milliseconds mean anything?
fn representative(info: &wgpu::AdapterInfo) -> bool {
    let n = info.name.to_ascii_lowercase();
    let virtualized = n.contains("paravirtual") || n.contains("virtualbox") || n.contains("vmware");
    info.device_type != wgpu::DeviceType::Cpu && !virtualized
}

// ── the run ─────────────────────────────────────────────────────────────────

/// One measured configuration: the frame-time rounds and the best round's
/// per-pass GPU breakdown.
struct Measured {
    rounds: Vec<Round>,
    best: usize,
    /// `(name, GPU ms, CPU record ms)` per pass — the CPU column is island
    /// wave I4b's addition, because a lit frame's dearest half turned out to be
    /// the recording rather than the drawing.
    passes: Vec<(&'static str, f64, f64)>,
    gpu_frame_ms: f64,
    cpu_ms: [f64; CPU_STAGES],
    instances: usize,
    scatter_batches: usize,
    /// Instances **inside** those batches (island wave I7b). A batch count says
    /// nothing about how much grows: the island's whole vegetation is one batch
    /// per terrain, because `push_biome_population` goes through the same
    /// `push_scatter` body a `PcgVolume` does.
    scatter_instances: usize,
    vgeom_instances: usize,
    skinned: usize,
    terrain_tiles: usize,
    vt_textures: usize,
    /// The VSM caster pass's own counters over the measured rounds (island wave
    /// I4b) — `pages x groups` is what the pass RECORDS, and the record column
    /// is meaningless without them.
    vsm: Option<inf_render::VsmRasterStats>,
    /// **The record stage's own breakdown** (island wave I7b), meaned over the
    /// best round. The per-pass record column tiles only the marked span, and on
    /// the island two thirds of a 10.874 ms stage was outside it.
    record: inf_render::RecordProfile,
    /// **The fixed step's own phases, taken INSIDE the frame loop** (island wave
    /// I8c) — meaned over the best round, beside `cpu_ms[0]`, which is the wall
    /// clock around the same call.
    ///
    /// The two clocks are the wave's clause 3. `island_gate` and this file's own
    /// isolated block both read the step at **5.67–5.74 ms** against the 6.0 ms
    /// `CITY_STEP_BUDGET_MS` ratchet, while `cpu_ms[0]` reads 6.32–6.85 ms across
    /// the five render configurations — over the ratchet in every row, on a
    /// simulation that is identical in all five. Whether that is the step or the
    /// machine is not answerable from a wall clock alone, and it is answerable
    /// from this: the phases **tile the step by construction** (each mark
    /// measures from the previous one), so `cpu_ms[0] − step_phases_ms` is time
    /// the thread spent not inside the step at all.
    ///
    /// **Measured, the residue is +0.002 ms** — so the answer is not scheduling.
    /// The step really does take longer in a frame than it does alone, and the
    /// extra is *inside its own phases*, which is what the breakdown beside this
    /// number exists to attribute.
    ///
    /// The profile costs one `Instant::now()` per phase — 24 of them, ~25 ns each
    /// on Windows, under a microsecond against a millisecond step — and a
    /// profiled step is byte-identical to an unprofiled one
    /// (`the_profile_does_not_move_the_simulation`).
    step_phases_ms: f64,
    /// The same phases, **phase by phase** (island wave I8c) — meaned over the
    /// best round, in the shape the isolated block prints, so the two clocks can
    /// be compared where they differ rather than only in total.
    step_profile: inf_player::step_profile::StepProfile,
    /// **The GI voxelizer's own counters** (island wave NPC1e), from the last
    /// frame of the measured run.
    ///
    /// Read for its engagement counters: `skinned_rejected`, behind the
    /// per-instance pre-reject that stopped the fifth skinned consumer walking a
    /// crowd per joint, and (wave VEN1a) `scatter_rejected` / `scatter_decimated`
    /// behind the scatter staging's two whole-batch rejects and its walk
    /// ceiling. A pass whose recording got cheaper and a pass with nothing to
    /// reject look identical in a millisecond.
    gi: inf_render::GiAudit,
}

impl Measured {
    fn round(&self) -> Round {
        self.rounds[self.best]
    }
}

/// Render `ROUNDS × FRAMES` frames of the composed scene at `(w, h)` and answer
/// the distribution.
///
/// The projection is the **production** one: `sync_voxel_store` then
/// `project_scene_full`, which is `PlayerRenderHost::project`'s whole body — the
/// two halves are public and named as such precisely so a gate can drive them
/// without a window.
fn measure(
    gpu: &GpuContext,
    fx: &mut Fixture,
    w: u32,
    h: u32,
    settings: inf_render::RenderSettings,
    // I7: the camera path is a PARAMETER, because the instrument now measures
    // two worlds — the composed city and the island — and a scripted flight over
    // one is not a flight over the other. Passing it rather than branching on a
    // flag inside `fly` keeps the frame loop identical for both, which is what
    // makes the two sets of numbers comparable at all.
    path: &dyn Fn(u64, u32, u32) -> RenderView,
) -> Measured {
    let target = HeadlessTarget::new(gpu, w, h);
    let mut renderer = EngineRenderer::new(gpu, HEADLESS_FORMAT);
    renderer.set_settings(settings);
    let vt_textures = bind_virtual_textures(gpu, &mut renderer, fx);
    let timed = renderer.set_gpu_timing(gpu, true);
    let mut scene = RenderScene {
        grid_enabled: false,
        ..Default::default()
    };
    let mut voxels = inf_voxel::VoxelVolumes::new();
    let mut debris = inf_render::DebrisCache::default();

    // **THE SECOND CLOCK ON THE FIXED STEP** (island wave I8c, clause 3). Armed
    // for the whole measurement so every configuration's rows are taken the same
    // way, and left on: it is one `Instant::now()` per phase and the step it
    // profiles is byte-identical to the step it would otherwise run.
    fx.sim.set_step_profiling(true);
    let mut step: u64 = 0;
    let mut frame = |scene: &mut RenderScene,
                     renderer: &mut EngineRenderer,
                     fx: &mut Fixture,
                     step: u64|
     -> (
        Option<inf_render::FrameTimings>,
        [f64; CPU_STAGES],
        inf_render::RecordProfile,
        inf_player::step_profile::StepProfile,
    ) {
        let view = path(step, w, h);
        // The CPU half, stage by stage. A frame that is CPU-bound and cannot say
        // WHERE is the same defect the certification found on the GPU side, one
        // processor over.
        let mut cpu = [0.0f64; CPU_STAGES];
        let t = std::time::Instant::now();
        fx.sim
            .step_once(inf_player::runtime_sim::RuntimeInput::default());
        cpu[0] = t.elapsed().as_secs_f64() * 1000.0;
        // The step's own phases, which tile it by construction — so the residue
        // against `cpu[0]` is time this thread was not inside the step, and the
        // breakdown is where the difference from the isolated block lives.
        let phases = fx.sim.step_profile();
        let t = std::time::Instant::now();
        fx.sim.sync_render_terrain(view.eye_world);
        sync_voxel_store(&mut voxels, &fx.voxel_assets, &fx.sim, view.eye_world);
        cpu[1] = t.elapsed().as_secs_f64() * 1000.0;
        let t = std::time::Instant::now();
        project_scene_full(
            scene,
            &fx.sim,
            1.0,
            &fx.vmeshes,
            &fx.skinned,
            &voxels,
            &mut debris,
            renderer.vt_textures(),
            &fx.scatter_meshes,
            &std::collections::HashMap::new(),
        );
        cpu[2] = t.elapsed().as_secs_f64() * 1000.0;
        let t = std::time::Instant::now();
        renderer.render(gpu, scene, &view, &target.view, (w, h));
        cpu[3] = t.elapsed().as_secs_f64() * 1000.0;
        // The record path's own phases, taken from inside the call that just
        // returned. CPU only, so it blocks on nothing and is not a stage.
        let rec = renderer.record_profile();
        let t = std::time::Instant::now();
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        cpu[4] = t.elapsed().as_secs_f64() * 1000.0;
        // **The instrument's own cost, inside the instrument's own clock.** The
        // readback below is a `map_async` plus a second poll, and it is inside
        // the `t0` span the percentiles are taken over — so it has to be a named
        // stage or the stage table stops tiling the frame it sits under.
        let t = std::time::Instant::now();
        let timings = renderer.gpu_timings(gpu);
        cpu[5] = t.elapsed().as_secs_f64() * 1000.0;
        (timings, cpu, rec, phases)
    };

    // The discarded pass: pipelines compile, the terrain's render cut converges
    // (`max_loads_per_sync` is 16, so a cut several rings deep takes tens of
    // frames to settle), and the scatter payloads seat in their content-keyed
    // buffers. Measuring any of that would be measuring a frame that happens
    // once.
    for _ in 0..FRAMES {
        frame(&mut scene, &mut renderer, fx, step);
        step += 1;
    }

    let mut rounds: Vec<Round> = Vec::with_capacity(ROUNDS);
    let mut best_passes: Vec<(&'static str, f64, f64)> = Vec::new();
    let mut best_gpu = 0.0;
    let mut best_cpu = [0.0f64; CPU_STAGES];
    let mut best_record = inf_render::RecordProfile::default();
    let mut best_phases = 0.0f64;
    let mut best_step = inf_player::step_profile::StepProfile::default();
    let mut best = 0usize;
    for r in 0..ROUNDS {
        let mut ms = Vec::with_capacity(FRAMES);
        let mut sums: Vec<(&'static str, f64, f64)> = Vec::new();
        let mut gpu_total = 0.0;
        let mut cpu_total = [0.0f64; CPU_STAGES];
        let mut rec_total = inf_render::RecordProfile::default();
        let mut phase_total = inf_player::step_profile::StepProfile::default();
        step = 0;
        for _ in 0..FRAMES {
            let t0 = std::time::Instant::now();
            let (timings, cpu, rec, phases) = frame(&mut scene, &mut renderer, fx, step);
            ms.push(t0.elapsed().as_secs_f64() * 1000.0);
            step += 1;
            phase_total.accumulate(&phases);
            for (slot, v) in cpu_total.iter_mut().zip(cpu) {
                *slot += v;
            }
            rec_total.accumulate(&rec);
            if let Some(t) = timings {
                gpu_total += t.total_ms;
                if sums.is_empty() {
                    sums = t.passes.iter().map(|p| (p.name, p.ms, p.cpu_ms)).collect();
                } else {
                    for (slot, p) in sums.iter_mut().zip(&t.passes) {
                        slot.1 += p.ms;
                        slot.2 += p.cpu_ms;
                    }
                }
            }
        }
        let round = round_of(ms);
        if r == 0 || round.p50 < rounds[best].p50 {
            best = r;
            best_gpu = gpu_total / FRAMES as f64;
            best_cpu = cpu_total.map(|v| v / FRAMES as f64);
            best_record = rec_total;
            best_record.scale(1.0 / FRAMES as f64);
            best_step = phase_total;
            best_step.scale(1.0 / FRAMES as f64);
            best_phases = best_step.total_ms();
            best_passes = sums
                .into_iter()
                .map(|(n, gpu, cpu)| (n, gpu / FRAMES as f64, cpu / FRAMES as f64))
                .collect();
        }
        rounds.push(round);
    }
    if !timed {
        best_passes.clear();
    }

    Measured {
        rounds,
        best,
        passes: best_passes,
        gpu_frame_ms: best_gpu,
        cpu_ms: best_cpu,
        instances: scene.instances.len(),
        scatter_batches: scene.scatter.len(),
        scatter_instances: scene.scatter.iter().map(|b| b.data.instances.len()).sum(),
        vgeom_instances: scene.vgeom_instances.len(),
        skinned: scene.skinned.len(),
        terrain_tiles: scene.terrains.iter().map(|t| t.tiles.len()).sum(),
        vt_textures,
        vsm: renderer.vsm_raster_stats(),
        gi: renderer.gi_audit(),
        record: best_record,
        step_phases_ms: best_phases,
        step_profile: best_step,
    }
}

/// Print the record stage's phases, dearest first, and **assert they tile it**.
///
/// The one direction that must hold: the phases are the whole of
/// `EngineRenderer::render`, and `cpu[3]` is the wall clock around exactly that
/// call, so the two are the same span measured twice. The slop is a tenth of a
/// millisecond over a sum of fourteen `Instant` reads.
fn print_record_profile(label: &str, m: &Measured) {
    let total = m.record.total_ms();
    if total <= 0.0 {
        println!("  {label} record profile: not armed (no GPU timing)");
        return;
    }
    println!(
        "  {label} record (record) {:.3} ms, phases sum {total:.3} ms:",
        m.cpu_ms[3]
    );
    for (name, ms) in m.record.dearest_first() {
        if ms <= 0.0005 {
            continue;
        }
        println!(
            "    {name:<16} {ms:7.3} ms ({:5.1} % of the record stage)",
            100.0 * ms / m.cpu_ms[3].max(1.0e-9)
        );
    }
    assert!(
        (total - m.cpu_ms[3]).abs() < 0.1 + 0.02 * m.cpu_ms[3],
        "{label}: the record phases sum to {total:.3} ms beside a {:.3} ms record \
         stage — they are the same call measured twice and must tile it",
        m.cpu_ms[3]
    );
}

/// **THE FIXED STEP'S TWO CLOCKS, RECONCILED** (island wave I8c, clause 3).
///
/// Two harnesses read the same simulation and disagreed. `island_gate`'s
/// step-profile arm and this file's own isolated block both put the island's
/// fixed step at **5.67–5.69 ms** against the 6.0 ms
/// [`CITY_STEP_BUDGET_MS`] ratchet; the `sim fixed step` row of the table above
/// reads **6.35–7.27 ms** — over the ratchet in every row — across five render
/// configurations whose simulation is *identical*. A ratchet that reads met on
/// one clock and breached on another is not a ratchet, and the I8b audit carried
/// it by name.
///
/// It is answerable, and the answer is a measurement rather than an attribution.
/// The step's phases **tile it by construction** (`step_profile`'s own header:
/// each mark measures from the previous one, so the sum *is* the step), and
/// `cpu_ms[0]` is a wall clock around the same call. So:
///
/// * **the phase sum** is what the step spent doing the step — the number the
///   ratchet is about, and the one both other harnesses report;
/// * **the wall minus the phase sum** is time this thread was not inside the
///   step at all: the render thread, the driver and the frame's own poll,
///   sharing a machine with it.
///
/// # And the answer, measured (island wave I8c)
///
/// **The residue is +0.002 ms.** So it is *not* scheduling: the step is not being
/// preempted, it is genuinely doing more work inside a frame than it does alone.
/// The extra is inside the phases, which is why the phases are printed here
/// beside the isolated block's own table — a step interleaved with a projection
/// that walks 389 793 instances and a record stage that walks 192 terrain tiles
/// re-reads its own working set from a colder cache every time, and that is a
/// cost a shipped game pays and a step harness on its own does not.
///
/// Neither number is wrong and only one of them is a fact about the simulation
/// alone. This prints both, the residue, and the breakdown, so the two harnesses
/// stop being two truths.
fn print_step_clocks(label: &str, m: &Measured) {
    let wall = m.cpu_ms[0];
    let phases = m.step_phases_ms;
    if phases <= 0.0 {
        return;
    }
    println!(
        "  {label} fixed step: {phases:.3} ms in its own phases inside a \
         {wall:.3} ms wall clock ({:+.3} ms, {:.1} %, is this thread not being \
         inside the step at all). REPORTED, NOT ASSERTED — the \
         {CITY_STEP_BUDGET_MS} ms ratchet binds the ISOLATED step, which is \
         `island_gate`'s furnish battery over the fixture and this file's own \
         city arm; this row is that same step paying for the frame around it \
         (the island wave I8c audit's LOW-3: the line said 'against a ratchet' \
         on the one clock the ratchet does not bind)",
        wall - phases,
        (wall - phases) / wall.max(1.0e-9) * 100.0,
    );
    for (n, ms) in m.step_profile.dearest_first() {
        if ms <= 0.02 {
            continue;
        }
        println!(
            "  {label}   in-frame step {n:<18} {ms:7.3} ms ({:4.1} %)",
            ms / phases.max(1.0e-9) * 100.0
        );
    }
    // The one direction that cannot hold: the phases tile the step and the wall
    // clock brackets the call that runs them, so a phase sum PAST the wall means
    // the two are not the same span and the reconciliation above is arithmetic
    // about nothing.
    assert!(
        phases <= wall + 0.05,
        "{label}: the fixed step's phases sum to {phases:.3} ms inside a \
         {wall:.3} ms wall clock around the same call"
    );
}

// ── the arms ────────────────────────────────────────────────────────────────

/// **The composed scene really is the city, the ground and the character** —
/// asserted with no GPU, because a frame time over content that is not what the
/// ledger says it is is a number about nothing.
///
/// And the load-bearing half: **the ground under the city changes not one
/// building**. `island_frame_source_png` holds the terrain flat at exactly zero
/// so the composed level's PCG output is bit-identical to the flat city wave I3
/// measured; if that ever stops being true, the instrument stops being an
/// instrument over *that* city and every number in the I3 ledger stops being
/// comparable. The number is the one I3 printed: **370 468 solids**.
#[test]
fn the_instrument_scene_carries_the_city_the_ground_and_the_character() {
    let tmp = tempfile::tempdir().expect("tmp");
    let pack = cook_instrument(tmp.path());
    let source = PackLevelSource::open(&pack).expect("the pack opens");
    let built = inf_player::build_world_from_pack(&source).expect("the pack world builds");
    let w = built.world.world();

    let mut volumes: Vec<(uuid::Uuid, PcgVolume)> = w
        .iter_entities()
        .filter_map(|e| Some((e.get::<Guid>()?.0, e.get::<PcgVolume>()?.clone())))
        .collect();
    volumes.sort_by_key(|(g, _)| *g);
    let solids: usize = volumes.iter().map(|(_, v)| v.structures.len()).sum();
    let buildings: usize = volumes.iter().map(|(_, v)| v.structure_groups.len()).sum();
    let terrains = w
        .iter_entities()
        .filter(|e| e.contains::<Terrain>())
        .count();
    let characters = w
        .iter_entities()
        .filter(|e| e.contains::<SkeletalMesh>())
        .count();

    println!(
        "instrument scene: {} volumes, {buildings} buildings, {solids} solids, \
         {terrains} terrain, {characters} skinned character",
        volumes.len()
    );

    assert_eq!(volumes.len(), 100, "the hundred city blocks must be here");
    assert_eq!(
        buildings, 1_000,
        "a thousand buildings, as wave I3 measured"
    );
    assert_eq!(
        solids, 370_468,
        "the ground under the city MOVED a building: wave I3's city is \
         370 468 solids and this one is {solids}. The instrument's terrain is \
         held flat at exactly zero so the composed level is the same city the \
         I3 ledger describes — if the terrain gains a slope, every number in \
         that ledger stops being comparable with every number in this one."
    );
    assert_eq!(terrains, 1, "the streamed ground must be in the world");
    assert_eq!(characters, 1, "the wizard character must be in the world");
}

/// **THE SHIPPED PLAYER PIPELINES, AND THAT IS WHAT THE ESTIMATE ASSUMES**
/// (island wave I4b).
///
/// Every run of this file prints a `PIPELINED ESTIMATE` — `max(CPU without the
/// wait, GPU frame)` — beside the serialized number it measures, on the stated
/// grounds that "a real presenter overlaps the halves". Wave I4 carried that as
/// arithmetic over two measurements and named a windowed present-to-present
/// harness as the honest closure. A harness needs a window and this battery has
/// none; what it can do, and what I4 did not, is **check the claim about the
/// player** rather than assert it in prose.
///
/// The player's frame path is four calls —
/// `SurfaceChain::acquire` → `EngineRenderer::render` (record + submit) →
/// `Queue::present` — and it contains **no blocking device poll**. The CPU
/// therefore runs ahead into the next frame while the GPU drains this one, and
/// `acquire` blocks only when the swap chain has no free image. That is the
/// overlap the estimate models.
///
/// # …AND THE PLAYER HAS A THIRD TERM THE ESTIMATE DOES NOT (the I4b audit)
///
/// `SurfaceChain::new` sets `PresentMode::AutoVsync`, so what makes the swap
/// chain run out of images is not only the GPU — it is the **display**. The
/// player's real presented cadence is
/// `max(CPU without the wait, GPU frame, the refresh interval)`, and the estimate
/// this file prints is the first two. That makes it a **lower bound on the
/// player's frame time and an upper bound on its fps**, which is the direction
/// that matters for a "≥ 60 fps" claim and is why the claim is written as an
/// estimate everywhere it appears: an engine that computes a frame in 16.5 ms
/// presents at 60 Hz on a 60 Hz panel whatever else is true.
///
/// It also constrains the **owed present-to-present harness**: a windowed harness
/// measuring `AutoVsync` measures the panel, so it has to configure `Immediate`
/// or `Mailbox` to measure the engine — and then say which one it measured.
///
/// This arm is a **source scope** *and* a module ban, because neither alone is
/// enough. The scopes extract `PlayerRenderHost::render`'s body and
/// `PlayerApp::frame`'s, so a wait somewhere else in the crate cannot satisfy
/// them by moving; the module ban sweeps both whole files, because a scope that
/// bans a *substring* is defeated by a one-line helper whose call site contains
/// no `poll(` at all — the P23 byte-pin lesson, met again (the I4b audit).
#[test]
fn the_shipped_players_frame_path_does_not_wait_for_the_gpu() {
    let render_rs = include_str!("../src/render.rs");
    let start = render_rs
        .find("pub fn render(&mut self, view: &RenderView) {")
        .expect("PlayerRenderHost::render is the player's one frame call");
    let body = &render_rs[start..];
    let end = body
        .find("\n    }\n")
        .expect("the function body ends at a de-indented brace");
    let body = &body[..end];
    println!(
        "the player's frame path is {} lines: {}",
        body.lines().count(),
        body.lines()
            .filter(|l| {
                let l = l.trim();
                !l.is_empty() && !l.starts_with("//")
            })
            .map(str::trim)
            .collect::<Vec<_>>()
            .join(" ")
    );
    assert!(
        body.contains("self.gpu.queue.present(frame)"),
        "the extracted body is not the present path — this arm is reading the \
         wrong function and would pass for anything"
    );
    assert!(
        !body.contains("poll("),
        "the shipped player's frame path now polls the device:\n{body}\nA poll \
         serializes the CPU and GPU halves, which is what the harness does on \
         purpose and what a presenter must not do — and it would make every \
         PIPELINED ESTIMATE this file prints a description of a frame the player \
         no longer draws."
    );

    // **AND THE LOOP THAT CALLS IT** (the I4b audit). The paragraph above said
    // this arm extracted "the windowed loop's own frame block" as well; it did
    // not, and a poll one caller up serializes the halves exactly as a poll one
    // caller down does. `PlayerApp::frame` is the block: it runs the fixed steps,
    // projects, and calls `host.render(&view)` — the whole of what a presented
    // frame costs the CPU.
    let window_rs = include_str!("../src/window.rs");
    let start = window_rs
        .find("fn frame(&mut self, event_loop: &ActiveEventLoop) {")
        .expect("PlayerApp::frame is the windowed loop's own frame block");
    let loop_body = &window_rs[start..];
    let end = loop_body
        .find("\n    }\n")
        .expect("the function body ends at a de-indented brace");
    let loop_body = &loop_body[..end];
    println!(
        "the windowed loop's frame block is {} lines",
        loop_body.lines().count()
    );
    assert!(
        loop_body.contains("live.host.render(&view)"),
        "the extracted block is not the windowed frame path — this arm is \
         reading the wrong function and would pass for anything"
    );
    assert!(
        !loop_body.contains("poll("),
        "the windowed loop now polls the device around its frame:\n{loop_body}"
    );

    // **AND THE BAN IS OVER THE MODULE, NOT INSIDE THE SCOPE** (the I4b audit).
    //
    // A substring ban inside a function body is defeated by a one-line helper:
    // `fn wait_for_gpu(&self) { self.gpu.device.poll(..) }` on `PlayerRenderHost`,
    // called from either body, puts no `poll(` at either call site. That is the
    // P23 byte-pin lesson word for word — **read a scope, ban the MODULE** — so
    // the two scopes above are what say "this is the right function", and the
    // sweep below is what says "and nothing they call waits either", over the two
    // files that own the player's frame.
    //
    // Three needles, because `poll(` is not the only spelling of a wait: a
    // `pollster::block_on` over a `map_async` serializes exactly as hard, and
    // `wait_for_submission_index` is wgpu's own name for the same thing. All
    // three are absent from both files today, which is what makes this a
    // tripwire rather than an allowlist.
    for (name, src) in [("render.rs", render_rs), ("window.rs", window_rs)] {
        for needle in [".poll(", "block_on", "wait_for_submission"] {
            assert!(
                !src.contains(needle),
                "`inf_player::{name}` contains `{needle}`, so something on the \
                 player's frame path waits for the device — even if the call site \
                 inside `render` or `frame` reads clean. A wait serializes the CPU \
                 and GPU halves, which is what this harness does on purpose and \
                 what a presenter must not do, and it would make every PIPELINED \
                 ESTIMATE this file prints a description of a frame the player no \
                 longer draws."
            );
        }
    }
}

/// **THE FIXED STEP'S OWN BREAKDOWN** (island wave I4b) — the table wave I4
/// could not print.
///
/// I4 measured the frame, found it CPU-bound, and found the single dearest thing
/// in it to be the fixed step at **13.0–14.9 ms** over this city — of which
/// ~2.2 ms was the I3 collider band **and ~11.5 ms was unattributed**. "Attribute
/// it before prescribing" is what the I4 audit routed to this wave, and this arm
/// is the attribution: `RuntimeSim` marks every phase of its own body and the
/// phases tile the step by construction.
///
/// **No GPU.** The step is CPU work over a cooked pack, so this arm runs
/// everywhere the battery runs — and the number it prints in the `dev` profile
/// is a number about a build nobody ships (the I4 law), which is why the
/// **budget is asserted only in `--release`**, on a machine whose milliseconds
/// mean something, exactly like every other wall clock in this tree.
#[test]
fn the_fixed_steps_own_budget() {
    let tmp = tempfile::tempdir().expect("tmp");
    let pack = cook_instrument(tmp.path());
    let mut fx = open(&pack);

    // The discarded pass, for the frame instrument's reason one processor over:
    // the first steps seat the collider band, mesh the terrain tiles, and take
    // every `structure_stamps` miss there is. Measuring them would be measuring
    // a step that happens once.
    for _ in 0..STEP_WARMUP {
        fx.sim
            .step_once(inf_player::runtime_sim::RuntimeInput::default());
    }
    fx.sim.set_step_profiling(true);

    let mut rounds: Vec<(f64, inf_player::step_profile::StepProfile)> = Vec::new();
    for _ in 0..STEP_ROUNDS {
        let mut acc = inf_player::step_profile::StepProfile::default();
        let t0 = std::time::Instant::now();
        for _ in 0..STEP_SAMPLES {
            fx.sim
                .step_once(inf_player::runtime_sim::RuntimeInput::default());
            acc.accumulate(&fx.sim.step_profile());
        }
        let wall = t0.elapsed().as_secs_f64() * 1000.0 / STEP_SAMPLES as f64;
        acc.scale(1.0 / STEP_SAMPLES as f64);
        rounds.push((wall, acc));
    }
    // MIN of rounds, by the step's own total — the instrument's own discipline.
    let best = rounds
        .iter()
        .enumerate()
        .min_by(|a, b| a.1 .1.total_ms().total_cmp(&b.1 .1.total_ms()))
        .map(|(i, _)| i)
        .expect("at least one round");
    let (wall, prof) = rounds[best];

    println!(
        "\n=== THE FIXED STEP, PHASE BY PHASE === {STEP_ROUNDS} rounds x \
         {STEP_SAMPLES} steps after {STEP_WARMUP} discarded, MIN of rounds; \
         content: the phase-30 city (370 468 solids, 1 000 buildings), a streamed \
         terrain and a skinned character — the fps instrument's own scene"
    );
    for (i, (w, p)) in rounds.iter().enumerate() {
        println!(
            "round {}: step {:.3} ms (wall {:.3} ms)",
            i + 1,
            p.total_ms(),
            w
        );
    }
    println!(
        "STEP {:.3} ms  [round {} of {STEP_ROUNDS}]",
        prof.total_ms(),
        best + 1
    );
    for (n, ms) in prof.dearest_first() {
        if ms <= 0.0005 {
            continue;
        }
        println!(
            "  {n:<18} {ms:7.3} ms  ({:5.1} % of the step)",
            ms / prof.total_ms().max(1.0e-9) * 100.0
        );
    }
    let silent: Vec<&str> = prof
        .rows()
        .filter(|(_, ms)| *ms <= 0.0005)
        .map(|(n, _)| n)
        .collect();
    if !silent.is_empty() {
        println!("  under 0.0005 ms: {}", silent.join(", "));
    }
    println!(
        "  the step's own wall clock is {wall:.3} ms; the phases sum to {:.3} ms",
        prof.total_ms()
    );
    // **What the solver is actually paying for.** A step whose dearest phase is
    // `bridge3d.step` over a world with one moving thing in it is a step paying
    // for its own STATIC geometry, and the pair count is the evidence.
    let (tracked, touching) = fx.sim.bridge3d().world().contact_pair_counts();
    println!(
        "  physics world: {} bodies, {} admitted structure colliders, \
         {tracked} contact pairs tracked ({touching} touching)",
        fx.sim.bridge3d().body_count(),
        fx.sim.bridge3d().admitted_structures(),
    );

    // **THE PHASES TILE THE STEP.** The GPU segments' tiling assertion and the
    // CPU stages', one processor over: a breakdown whose parts do not add up to
    // the whole it sits beside is a breakdown of a step nobody measured. The
    // wall clock also carries `set_input` and the profile does not, which is
    // three `BTreeSet` differences on an empty input — hence a tolerance rather
    // than an equality, and it is in PROPORTION rather than in milliseconds so
    // it means the same thing in `dev` (where the step is slower) as in release.
    //
    // **AND THE TOLERANCE SITS BELOW THE DEFECT IT NAMES** (the I4b audit). It
    // was `0.10`, against a residue that measures **0.000** — the two numbers
    // print equal to three decimals on this scene. A tenth of a 1.25 ms step is
    // 0.125 ms, which is more than nineteen of the twenty-two phases *put
    // together*: 94 % of this step is `physics3d sync`, `solver` and `character
    // move`, so a breakdown that lost every other phase would have passed.
    // Mutation-measured: `StepProfile::total_ms` summing only the first eleven
    // slots drops 5.6 % of the step and the old clause did not fire. 2 % is
    // twenty times the observed residue and a third of that mutation, which is
    // what "compute the defect, then put the threshold under it" asks for.
    let drift = (wall - prof.total_ms()).abs() / wall.max(1.0e-9);
    println!(
        "  the phases account for {:.3} % of the step's wall clock",
        (1.0 - drift) * 100.0
    );
    assert!(
        drift < 0.02,
        "the phases sum to {:.3} ms beside a {wall:.3} ms step — {:.2} % of the \
         step is in no phase, so the breakdown describes a step this arm did not \
         time",
        prof.total_ms(),
        drift * 100.0
    );

    // The §8 budget itself — release only, real machine only, for
    // `inf_player::budget`'s stated reason.
    if cfg!(debug_assertions) {
        eprintln!(
            "\ndev profile (opt-level 1, debug assertions ON): the step is \
             reported, not asserted — re-run with --release for the number \
             CITY_STEP_BUDGET_MS is set from"
        );
        return;
    }
    if std::env::var_os("CI").is_some() {
        eprintln!("\nCI: the step is reported, not asserted (shared runner)");
        return;
    }
    println!(
        "STEP BUDGET: {:.3} ms measured against a {CITY_STEP_BUDGET_MS} ms \
         ceiling {RATCHET_NOTE}",
        prof.total_ms()
    );
    assert!(
        prof.total_ms() <= CITY_STEP_BUDGET_MS,
        "the fixed step cost {:.3} ms over the city, past the \
         {CITY_STEP_BUDGET_MS} ms ceiling {RATCHET_NOTE}",
        prof.total_ms()
    );
}

/// **A stopwatch is not behaviour.** The phase clock reads no sim state, writes
/// none and changes no ordering, so a profiled step and an unprofiled one must
/// produce byte-identical sim state — and this is the arm that says so rather
/// than the comment that claims it.
///
/// Built to falsify: it compares `state_bytes()` (the same buffer the replay
/// fold, `step_state_hash` and every PIE == shipping arm consume) after the same
/// number of steps on two sims built from the same pack, one profiled.
#[test]
fn the_profile_does_not_move_the_simulation() {
    let tmp = tempfile::tempdir().expect("tmp");
    let pack = cook_instrument(tmp.path());
    let mut plain = open(&pack);
    let mut profiled = open(&pack);
    profiled.sim.set_step_profiling(true);
    for _ in 0..24 {
        plain
            .sim
            .step_once(inf_player::runtime_sim::RuntimeInput::default());
        profiled
            .sim
            .step_once(inf_player::runtime_sim::RuntimeInput::default());
    }
    let a = plain.sim.state_bytes();
    let b = profiled.sim.state_bytes();
    println!(
        "24 steps: {} bytes of sim state, profiled and not, identical = {}",
        a.len(),
        a == b
    );
    assert!(
        profiled.sim.step_profile().total_ms() > 0.0,
        "the profiled sim reported a zero step — the clock is not armed, so the \
         comparison below is between two unprofiled runs"
    );
    assert_eq!(
        a, b,
        "a profiled step and an unprofiled one produced different sim state"
    );
}

/// **The instrument, at shipping resolution.** The wave's headline number.
#[test]
fn the_frame_at_shipping_resolution() {
    let Ok(gpu) = GpuContext::headless() else {
        eprintln!("SKIP fps_instrument: no GPU adapter");
        return;
    };
    let info = gpu.adapter.get_info();
    let real = representative(&info);

    let tmp = tempfile::tempdir().expect("tmp");
    let pack = cook_instrument(tmp.path());
    let mut fx = open(&pack);
    let (settings, tier) = shipped_settings(&gpu, fx.record);

    println!(
        "\n=== THE FPS INSTRUMENT === {} ({:?}), tier {tier:?}\n\
         settings: vgeom {} / visbuffer {} / shadows {} / gi {} / vsm {} / taa {} / \
         ssao {} / bloom {} / vt budget {} MiB\n\
         method: {ROUNDS} rounds x {FRAMES} frames after one discarded pass of \
         {FRAMES}, MIN of rounds by p50, every round replaying the SAME camera \
         sequence; scripted drive-through, full re-projection every frame",
        info.name,
        info.device_type,
        settings.vgeom.enabled,
        settings.vgeom.visbuffer,
        settings.shadows.enabled,
        settings.gi.enabled,
        settings.vsm.enabled,
        settings.taa,
        settings.ssao.enabled,
        settings.bloom.enabled,
        settings.vt.budget_bytes / (1024 * 1024),
    );
    // **WHAT THIS FRAME DOES NOT DRAW, said out loud** (the I4 audit).
    //
    // Every one of those flags is `false`, and none of it is the instrument's
    // choice: `RenderSettingsRecord::default()` ships shadows / GI / TAA / SSAO
    // / bloom off, `VsmSettings::default().enabled` is `false` engine-wide
    // ("until P27.4 gives the pages a receiver"), and `VgeomSettings::visbuffer`
    // is off on every tier. So the headline below is an honest measurement of
    // **what a shipped player draws for a level that authors no render block**,
    // and it is emphatically not a measurement of a frame with a lighting stack
    // in it. Quoting the number without this line would make "≥ 60 fps" mean a
    // frame with no shadows in it. `the_price_of_the_lighting_stack` below runs
    // the same content with the authorable half turned on and prints what it
    // costs — reported, never asserted, because the ceilings are set from the
    // shipped configuration.
    println!(
        "NOTE — the measured frame draws NO shadows, NO GI, NO VSM, NO TAA, NO \
         SSAO, NO bloom and NO visbuffer. That is the shipped default for a level \
         with no authored render block, not a choice this harness made; the price \
         of turning the authorable half on is printed at the end of this run."
    );

    let mut worst_p95 = 0.0f64;
    let mut worst_p99 = 0.0f64;
    // The 1080p row's own p95 and GPU frame, kept so the lighting stack's price
    // is a difference between two runs of the SAME resolution.
    let mut shipped_1080 = (0.0f64, 0.0f64);
    for (w, h, label) in RESOLUTIONS {
        let m = measure(&gpu, &mut fx, w, h, settings, &fly);
        let r = m.round();
        if (w, h) == (RESOLUTIONS[0].0, RESOLUTIONS[0].1) {
            shipped_1080 = (r.p95, m.gpu_frame_ms);
        }
        println!(
            "\n{label} ({w}x{h}): p50 {:.3} ms ({:.1} fps) | p95 {:.3} ms | \
             p99 {:.3} ms | worst {:.3} ms   [round {} of {ROUNDS}]",
            r.p50,
            1000.0 / r.p50.max(1.0e-9),
            r.p95,
            r.p99,
            r.worst,
            m.best + 1,
        );
        println!(
            "{label} content: {} mesh instances, {} scatter batches / {} \
             scattered instances, {} vgeom \
             instances, {} skinned, {} terrain tiles, {} virtual textures",
            m.instances,
            m.scatter_batches,
            m.scatter_instances,
            m.vgeom_instances,
            m.skinned,
            m.terrain_tiles,
            m.vt_textures
        );
        for (i, rr) in m.rounds.iter().enumerate() {
            println!(
                "{label} round {}: p50 {:.3} p95 {:.3} p99 {:.3}",
                i + 1,
                rr.p50,
                rr.p95,
                rr.p99
            );
        }
        let cpu_sum: f64 = m.cpu_ms.iter().sum();
        println!(
            "{label} CPU frame {cpu_sum:.3} ms (a MEAN — every stage below is a \
             mean, and the p50/p95/p99 above are percentiles of the same {FRAMES} \
             frames; the round's own mean frame is {:.3} ms), stage by stage:",
            r.mean
        );
        for (n, ms) in CPU_STAGE_NAMES.iter().zip(m.cpu_ms) {
            println!(
                "{label}   {n:<16} {ms:7.3} ms  ({:5.1} % of the CPU frame)",
                ms / cpu_sum.max(1.0e-9) * 100.0
            );
        }
        print_step_clocks(label, &m);
        print_record_profile(label, &m);
        // **THE STAGES TILE THE FRAME** — the CPU twin of the GPU segments'
        // tiling assertion below, and the arm that would have caught the residue
        // the I4 audit found: the timestamp readback sat inside the wall clock the
        // percentiles are taken over and inside no stage, so the table summed to
        // 37.839 ms beside a 39.792 ms headline with nothing naming the gap.
        // 0.5 ms is one `Instant::now()` pair per stage plus the loop's own
        // bookkeeping, which is what the difference is allowed to be.
        assert!(
            (cpu_sum - r.mean).abs() < 0.5,
            "{label}: the CPU stages sum to {cpu_sum:.3} ms beside a {:.3} ms mean \
             frame — {:.3} ms of the measured frame is in no stage, so the \
             breakdown describes a frame this harness did not time",
            r.mean,
            (r.mean - cpu_sum).abs()
        );
        if m.passes.is_empty() {
            println!("{label} per-pass: unavailable (no timestamp queries on this device)");
        } else {
            let mut by_cost = m.passes.clone();
            by_cost.sort_by(|a, b| b.1.total_cmp(&a.1));
            println!(
                "{label} GPU frame {:.3} ms; dearest passes:",
                m.gpu_frame_ms
            );
            for (n, ms, cpu) in by_cost.iter().take(12) {
                println!(
                    "{label}   {n:<16} {ms:7.3} ms  ({:5.1} % of the GPU frame)   \
                     record {cpu:6.3} ms",
                    ms / m.gpu_frame_ms.max(1.0e-9) * 100.0
                );
            }
            // **The CPU column does NOT tile the record stage, and by how much**
            // (the I4b audit). `inf_render::timing` said the CPU segments "tile
            // the record phase exactly as the GPU segments tile the frame"; they
            // do not, and the difference is not an epsilon. `FrameTimer::begin`
            // opens at the frame's first *command*, so everything
            // `EngineRenderer::render` does before that — the view matrices, the
            // light and deform uniform writes, the encoder itself — is inside
            // `render (record)` and inside no segment, and so are
            // `encoder.finish()` and the submit after the last mark. Printed
            // rather than asserted away, and bounded in the one direction that
            // has to hold: a part may not exceed its whole.
            let record_sum: f64 = m.passes.iter().map(|(_, _, cpu)| cpu).sum();
            println!(
                "{label} the per-pass RECORD column sums to {record_sum:.3} ms of \
                 the {:.3} ms `render (record)` stage — the {:.3} ms outside it is \
                 the setup before the timer's first command plus the finish and \
                 submit after its last",
                m.cpu_ms[3],
                m.cpu_ms[3] - record_sum,
            );
            assert!(
                record_sum <= m.cpu_ms[3] + 0.5,
                "{label}: the per-pass record column sums to {record_sum:.3} ms \
                 inside a {:.3} ms record stage — a part cannot exceed its whole, \
                 so one of the two clocks is measuring something else",
                m.cpu_ms[3]
            );
            // Anti-vacuity: a report whose segments do not add up to the frame is
            // a report about a frame the renderer did not draw.
            let sum: f64 = m.passes.iter().map(|(_, ms, _)| ms).sum();
            assert!(
                (sum - m.gpu_frame_ms).abs() < 1.0e-6,
                "{label}: the per-pass segments ({sum:.6} ms) do not tile the \
                 GPU frame ({:.6} ms)",
                m.gpu_frame_ms
            );
        }
        // **The harness frame is SERIALIZED and a presenter's is not.** This loop
        // polls to completion every frame, because a frame time measured without
        // a sync point is a submission time; the price is that the GPU's work
        // lands *after* the CPU's instead of underneath it. A real presenter
        // overlaps them, so the frame it would show is bounded below by the
        // dearer of the two halves. Reported as an estimate and never asserted —
        // it is arithmetic over two measurements, not a third measurement.
        // The wait AND the instrument's own readback come off: neither is work a
        // presenter's frame does.
        let submitted = cpu_sum - m.cpu_ms[4] - m.cpu_ms[5];
        let pipelined = submitted.max(m.gpu_frame_ms);
        println!(
            "{label} PIPELINED ESTIMATE {pipelined:.3} ms ({:.1} fps) = max(CPU without the wait or the stopwatch {submitted:.3}, GPU frame {:.3}) — an estimate, not a measurement; the asserted number is the serialized p95 above",
            1000.0 / pipelined.max(1.0e-9),
            m.gpu_frame_ms
        );
        println!(
            "{label} DISTANCE FROM 60 fps: p50 {:+.3} ms, p95 {:+.3} ms against a \
             {SHIPPING_FRAME_BUDGET_MS} ms frame",
            r.p50 - SHIPPING_FRAME_BUDGET_MS,
            r.p95 - SHIPPING_FRAME_BUDGET_MS
        );
        worst_p95 = worst_p95.max(r.p95);
        worst_p99 = worst_p99.max(r.p99);
    }

    if !real {
        eprintln!(
            "\n{}: timing reported, not asserted (software/paravirtual adapter)",
            info.name
        );
        return;
    }
    if std::env::var_os("CI").is_some() {
        // **Every CI runner reports and does not assert, by name** — the law
        // `inf-anim`'s inertialization harness paid for and `inf_player::budget`'s
        // header states. A frame time on a shared virtualized runner is a
        // measurement of the runner. Locally, and in the I9 certification, the
        // ceilings below still bite.
        eprintln!("\nCI: frame times reported, not asserted (shared runner)");
        return;
    }

    // ── THE PRICE OF THE LIGHTING STACK (the I4 audit) ──────────────────────
    //
    // Everything above is the shipped default, and the shipped default has no
    // shadows in it. A constitution that says what "≥ 60 fps" means over a frame
    // with the expensive half of the renderer switched off would be quoted for
    // years as though it covered a lit frame, so the same content is run once
    // more at 1080p with the stack on and the difference is printed.
    //
    // Through the **authoring door** (`RenderSettingsRecord` → `shipped_settings`),
    // not by poking `RenderSettings`: that is what an author enabling shadows in
    // Project Settings produces, so the number is a number about a level somebody
    // could ship. VSM is the one exception — it has no authorable field, because
    // `VsmSettings::default().enabled` is a *code* default the tier applies over —
    // so it is set beside the record and named as such.
    //
    // **ASSERTED SINCE WAVE CERT1, and folded into `worst_p95`.** It used to be
    // reported and never folded, on the ground that "the ratchets are set from
    // the shipped configuration; a second configuration asserted against them
    // would be a ceiling for a frame nobody has ratcheted". That ground is gone:
    // CERT1 authored the stack into the showcase island and the three 3D starter
    // templates, so THIS is the shipped configuration for every level anybody is
    // meant to look at, and `SHIPPING_FRAME_CEILING_MS`'s own doc required the
    // constant to be re-minted the day that happened. It was — down, 40.0 -> 38.0.
    //
    // **What the stack costs is a SPREAD and not a number**, and the CERT1 audit
    // is why that sentence reads this way. The wave quoted two runs, +0.129 ms
    // and +6.290 ms of p95, and concluded "somewhere between a seventh of a
    // millisecond and six of them". A third run of this same arm on the same
    // machine and the same adapter, at the same head, measured
    // **p95 20.818 lit against 10.660 shipped: +10.158 ms**, with the lit GPU
    // frame 5.414 ms against 2.435 (+2.979, i.e. DEARER, where the first run had
    // it 2.069 ms cheaper). Three runs, three answers, and the shipped baseline
    // is the half that moves: 16.733 / 14.209 / 10.660. So the honest reading is
    // that this content's shipped frame is not reproducible to better than about
    // 1.6x between runs, that the stack's price is bounded by the runs at
    // roughly +0.1 to +10.2 ms, and that the ratchet below is minted on the
    // WORST lit p95 anyone has recorded here (20.818) rather than on a delta.
    // 38.0 is 1.83x that.
    //
    // **Behind the adapter and CI gates, on purpose.** It is 480 more frames of a
    // GI + VSM + TAA frame, which on a software rasterizer is minutes rather than
    // seconds, and the two runners this repository has are software rasterizers.
    // A diagnostic nobody asserts must not be a CI cost.
    {
        let lit_record = inf_scene::RenderSettingsRecord {
            bloom_enabled: true,
            ssao_enabled: true,
            taa: true,
            shadows_enabled: true,
            gi_enabled: true,
            ..fx.record
        };
        let (mut lit, lit_tier) = shipped_settings(&gpu, lit_record);
        lit.vsm.enabled = true;
        let (w, h, label) = RESOLUTIONS[0];
        // **A tier below High clamps the stack straight back off** (`RenderTier::Low`
        // sets `shadows.enabled` and `gi.enabled` to `false`), and a price printed
        // for a configuration the tier refused is the price of nothing. Reported
        // rather than asserted, because "this adapter is Medium" is a fact about a
        // machine and a red build on one is the one-platform hazard P25 paid for.
        let clamped = !(lit.shadows.enabled && lit.gi.enabled);
        if clamped {
            println!(
                "\n{label} the lighting stack is clamped off at tier {lit_tier:?} \
                 (shadows {} / gi {}) — no price to print on this adapter, and the \
                 shipped ceilings below still bite",
                lit.shadows.enabled, lit.gi.enabled
            );
        }
        if !clamped {
            let m = measure(&gpu, &mut fx, w, h, lit, &fly);
            let r = m.round();
            let (base_p95, base_gpu) = shipped_1080;
            println!(
                "\n{label} WITH THE LIGHTING STACK ON (tier {lit_tier:?}; shadows {} / \
             gi {} / vsm {} / taa {} / ssao {} / bloom {}): p50 {:.3} ms ({:.1} fps) \
             | p95 {:.3} ms | p99 {:.3} ms | GPU frame {:.3} ms",
                lit.shadows.enabled,
                lit.gi.enabled,
                lit.vsm.enabled,
                lit.taa,
                lit.ssao.enabled,
                lit.bloom.enabled,
                r.p50,
                1000.0 / r.p50.max(1.0e-9),
                r.p95,
                r.p99,
                m.gpu_frame_ms,
            );
            // **The lit frame gets the SAME two tables the shipped one gets**
            // (island wave I4b). A configuration whose price is quoted as one
            // number cannot be optimised: I4b's first act was to attribute the
            // CPU frame and the sim step, and a lit frame that says only "p95
            // 64.9" sends the next reader to the GPU when a third of it may be
            // on the other processor.
            if let Some(v) = m.vsm.as_ref() {
                println!(
                    "{label} lit VSM raster: {} frames opened the page pass, \
                     {} page rectangles, {} indirect draws, {} casters \
                     ({} from scatter, {} terrain), {} deferred pages, {} \
                     dropped casters — i.e. {:.1} pages and {:.0} draws per \
                     rastering frame",
                    v.frames,
                    v.pages,
                    v.draws,
                    v.casters,
                    v.scatter_casters,
                    v.terrain_casters,
                    v.deferred_pages,
                    v.dropped_casters,
                    v.pages as f64 / v.frames.max(1) as f64,
                    v.draws as f64 / v.frames.max(1) as f64,
                );
                println!(
                    "{label} lit VSM group mask: {} indirect draws skipped \
                     ({:.0} per rastering frame) against {} issued; \
                     {} invalidation touches ({:.0} per frame)",
                    v.skipped_draws,
                    v.skipped_draws as f64 / v.frames.max(1) as f64,
                    v.draws,
                    v.invalidation_touches,
                    v.invalidation_touches as f64 / v.frames.max(1) as f64,
                );
            }
            let lit_cpu: f64 = m.cpu_ms.iter().sum();
            println!("{label} lit CPU frame {lit_cpu:.3} ms (a MEAN), stage by stage:");
            for (n, ms) in CPU_STAGE_NAMES.iter().zip(m.cpu_ms) {
                println!(
                    "{label}   lit {n:<16} {ms:7.3} ms  ({:5.1} % of the lit CPU frame)",
                    ms / lit_cpu.max(1.0e-9) * 100.0
                );
            }
            print_step_clocks(label, &m);
            print_record_profile(label, &m);
            if !m.passes.is_empty() {
                let mut by_cost = m.passes.clone();
                by_cost.sort_by(|a, b| (b.1 + b.2).total_cmp(&(a.1 + a.2)));
                println!(
                    "{label} lit GPU frame {:.3} ms; every pass, with what it cost \
                     to RECORD it:",
                    m.gpu_frame_ms
                );
                for (n, ms, cpu) in by_cost
                    .iter()
                    .filter(|(_, ms, cpu)| *ms >= 0.0005 || *cpu >= 0.0005)
                {
                    println!(
                        "{label}   lit {n:<16} {ms:7.3} ms  ({:5.1} % of the lit \
                         GPU frame)   record {cpu:6.3} ms",
                        ms / m.gpu_frame_ms.max(1.0e-9) * 100.0
                    );
                }
            }
            let lit_submitted = lit_cpu - m.cpu_ms[4] - m.cpu_ms[5];
            println!(
                "{label} lit PIPELINED ESTIMATE {:.3} ms ({:.1} fps) = max(CPU without the wait or the stopwatch {lit_submitted:.3}, GPU frame {:.3})",
                lit_submitted.max(m.gpu_frame_ms),
                1000.0 / lit_submitted.max(m.gpu_frame_ms).max(1.0e-9),
                m.gpu_frame_ms
            );
            println!(
                "{label} THE STACK'S PRICE, same resolution and same content: p95 \
             {:.3} ms lit against {base_p95:.3} ms as shipped ({:+.3} ms), GPU \
             frame {:.3} ms against {base_gpu:.3} ms ({:+.3} ms). The PRICE is \
             reported and never asserted; the lit p95 and p99 themselves ARE \
             asserted, against the same ceilings as the shipped configuration \
             (wave CERT1 — this line said 'reported, never asserted' about the \
             whole row while the fold three assertions below already covered it).",
                r.p95,
                r.p95 - base_p95,
                m.gpu_frame_ms,
                m.gpu_frame_ms - base_gpu,
            );
            // **Anti-vacuity, re-aimed onto counters** (island wave I8b).
            //
            // This used to assert `m.gpu_frame_ms > base_gpu` — the lit frame
            // must be *dearer* than the shipped one — and it stopped being
            // true, on a fixture whose content did not change: with the
            // settlements' parts out of the caster set (`ScatterBatch::
            // casts_shadows`) the lit configuration reads **4.775 ms of GPU
            // against the shipped 8.278**, and its p50 is **14.088 ms against
            // 17.044**. The lighting stack is cheaper than the frame it is
            // added to.
            //
            // Two things are wrong with the old assertion and only one of them
            // is this wave's. The first is I4b's own law, applied to this file:
            // the two figures come from **two different runs**, minutes apart,
            // at two device states, and a GPU that has been boosting for a
            // minute is not the one that measured the first row. The second is
            // that "did the stack run" is a question about *engagement*, not
            // about a clock (the P20.3 law) — and the stack publishes counters
            // that say so exactly.
            //
            // So the arm asserts that the lit configuration DID THE WORK: the
            // VSM opened its page pass, packed casters and issued draws. A
            // renderer that came back with the stack silently clamped off
            // produces zeros here, which is the defect the old comparison was
            // reaching for. The price stays reported and never asserted.
            let vsm = m
                .vsm
                .as_ref()
                .expect("the lit configuration publishes VSM raster stats");
            assert!(
                vsm.frames > 0 && vsm.casters > 0 && vsm.draws > 0,
                "the lit configuration rastered no virtual shadow map at all \
                 ({} frames, {} casters, {} draws) — shadows, GI and VSM came \
                 back enabled and did nothing, so the price printed above is \
                 the price of nothing",
                vsm.frames,
                vsm.casters,
                vsm.draws
            );
            // …and the ceiling covers this frame too (wave CERT1). Folded here
            // rather than asserted separately so there is ONE ratchet and one
            // failure message: whichever configuration is worse is the one the
            // number below is about. The dev-profile gate is still downstream,
            // so the full battery keeps reporting and asserting nothing.
            worst_p95 = worst_p95.max(r.p95);
            worst_p99 = worst_p99.max(r.p99);
        }
    }

    if cfg!(debug_assertions) {
        // **The build is not the build, so this reports rather than asserts** —
        // the paravirtual-adapter law, one layer down. `[profile.dev]` is
        // `opt-level = 1` with debug assertions on for every workspace crate, so
        // the CPU half of the frame here is a measurement of a build nobody
        // ships; the GPU half is unaffected and is printed above either way.
        //
        // The full battery runs in this profile, which is exactly why the
        // ceilings are not asserted in it: a tripwire that only ever sees the
        // slow build would have to be set where the fast build cannot regress
        // past it, and would therefore never fire.
        //
        //   cargo test --release -p inf-player --test fps_instrument -- --nocapture
        //
        // is the run that asserts, and the one the I9 certification makes.
        eprintln!(
            "\ndev profile (opt-level 1, debug assertions ON): frame times \
             reported, not asserted — re-run with --release for the shipping-build \
             number the ceilings are set from"
        );
        return;
    }
    assert!(
        worst_p95 <= SHIPPING_FRAME_CEILING_MS,
        "the 95th-percentile frame cost {worst_p95:.3} ms at the worse of the two \
         shipping resolutions, over the {SHIPPING_FRAME_CEILING_MS} ms ceiling on \
         {} {RATCHET_NOTE}",
        info.name
    );
    assert!(
        worst_p99 <= SHIPPING_FRAME_P99_CEILING_MS,
        "the 99th-percentile frame cost {worst_p99:.3} ms, over the \
         {SHIPPING_FRAME_P99_CEILING_MS} ms hitch ceiling on {} {RATCHET_NOTE}",
        info.name
    );
}

// ── THE ISLAND (wave I7) ────────────────────────────────────────────────────

/// How far the island camera advances per frame, metres.
///
/// 0.9 m at 60 Hz is 54 m/s — a fast car — so a 120-frame round covers 108 m and
/// the four-pass run covers 432 m of a 7 168 m island. Every number below
/// describes one stretch of one route, exactly as the city's does.
const ISLAND_FLY_STEP_M: f64 = 0.9;

/// The island's own flythrough: a low pass east from the first city.
///
/// Scripted, positional and moving, for the same three reasons the city's is —
/// and low, at 40 m, because that is where a player is. A camera at two
/// kilometres would measure the coarse end of the pyramid and nothing else.
fn island_fly(step: u64, width: u32, height: u32, from: DVec3) -> RenderView {
    let eye = DVec3::new(
        from.x + step as f64 * ISLAND_FLY_STEP_M,
        from.y + 40.0,
        from.z,
    );
    RenderView {
        origin: FloatingOrigin::new(DVec3::ZERO),
        eye_world: eye,
        forward: Vec3::new(1.0, -0.22, 0.0).normalize(),
        up: Vec3::Y,
        fov_y: 70f32.to_radians(),
        near: 0.05,
        width,
        height,
        ortho: None,
    }
}

/// **How many NPCs the island's headline arm puts in Harbour City** (wave NPC1b).
///
/// A thousand, because that is the number every NPC1a table is written at and
/// the number the arc's brief names. The block they stand in is 160 m of
/// half-extent about the flight's start, chosen the way `crowd_sweep`'s is:
/// wider than `DEFAULT_CROWD_NEAR_M` (96 m) so the ladder produces a real mix of
/// tiers, inside `DEFAULT_CROWD_FAR_M` (512 m) so the frame is not quietly
/// measuring an empty town.
const ISLAND_CROWD_N: usize = 1_000;

/// Half-extent of the block the island's crowd stands in, metres.
const ISLAND_CROWD_HALF_M: f64 = 160.0;

/// The archetype every test NPC copies: the level's own lowest-`Guid` rigged
/// character, so a thousand of them share one mesh, one skeleton, one `.inf_sm`
/// and every clip.
///
/// A MIRROR of `crowd_sweep::archetype_from_hero`, and deliberately so: two
/// instruments measuring the same subject have to build it the same way, or the
/// island's numbers and the city's are about two different crowds.
fn island_archetype(sim: &inf_player::runtime_sim::RuntimeSim) -> inf_ecs::crowd::CrowdArchetype {
    let w = sim.world().world();
    let mut best: Option<(uuid::Uuid, inf_ecs::crowd::CrowdArchetype)> = None;
    for e in w.iter_entities() {
        let (Some(g), Some(sk)) = (
            e.get::<inf_ecs::Guid>(),
            e.get::<inf_ecs::components::SkeletalMesh>(),
        ) else {
            continue;
        };
        if sk.skeleton.is_none() {
            continue;
        }
        let sm = e
            .get::<inf_ecs::components::AnimStateMachine>()
            .and_then(|a| a.sm);
        let a = inf_ecs::crowd::CrowdArchetype::humanoid(sk.mesh, sk.skeleton, sm);
        if best.as_ref().is_none_or(|(bg, _)| g.0 < *bg) {
            best = Some((g.0, a));
        }
    }
    let (guid, a) = best.expect(
        "the island has no rigged character to copy, so a crowd of it would pose \
         nothing and this row would price an empty pipeline",
    );
    println!(
        "island crowd archetype: hero {guid} — skeleton {:?}",
        a.skeleton
    );
    a
}

/// A thousand NPCs walking short there-and-back routes about `centre`.
///
/// Deterministic lattice placement — no RNG, because the placement is fixture
/// content; the per-agent variation the crowd itself needs comes from
/// `agent_rand` inside the engine.
fn island_population(
    n: usize,
    archetype: inf_ecs::crowd::CrowdArchetype,
    centre: DVec3,
) -> std::collections::BTreeMap<uuid::Uuid, inf_ecs::crowd::CrowdRecord> {
    const NAMESPACE: u128 = 0x4e50_4331_625f_4953_4c41_4e44_0000_0000;
    let mut out = std::collections::BTreeMap::new();
    for i in 0..n {
        let side = (n as f64).sqrt().ceil().max(1.0);
        let (ix, iz) = ((i as f64) % side, (i as f64 / side).floor());
        let span = 2.0 * ISLAND_CROWD_HALF_M;
        let x = centre.x - ISLAND_CROWD_HALF_M + span * (ix + 0.5) / side;
        let z = centre.z - ISLAND_CROWD_HALF_M + span * (iz + 0.5) / side;
        let from = DVec3::new(x, centre.y, z);
        out.insert(
            uuid::Uuid::from_u128(NAMESPACE | i as u128),
            inf_ecs::crowd::CrowdRecord::walking(
                archetype,
                inf_ecs::crowd::CrowdRoute::between(from, from + DVec3::new(0.0, 0.0, 6.0), 1.4),
            ),
        );
    }
    out
}

/// **The level's player-controlled hero, and where it stands** (island wave
/// NPC1e) — a MIRROR of `island_gate`'s `hero_entity` / `set_hero` pair, and
/// deliberately so: the certification row stages the same world the day gate
/// stages, and two instruments that put a hero in a town by different doors are
/// two instruments measuring two towns.
fn hero_at(sim: &inf_player::runtime_sim::RuntimeSim) -> Option<DVec3> {
    let w = sim.world().world();
    for e in w.iter_entities() {
        if e.get::<inf_ecs::components::CharacterMovement>()
            .is_some_and(|m| m.player_controlled)
        {
            return e
                .get::<inf_ecs::components::Transform>()
                .map(|t| t.translation.to_dvec3());
        }
    }
    None
}

/// Move the hero — the streaming anchor every band in the engine is derived
/// from — to `p`.
fn set_hero(sim: &mut inf_player::runtime_sim::RuntimeSim, p: DVec3) {
    let world = sim.world_mut();
    let mut found = None;
    {
        let w = world.world();
        for e in w.iter_entities() {
            if e.get::<inf_ecs::components::CharacterMovement>()
                .is_some_and(|m| m.player_controlled)
            {
                found = Some(e.id());
            }
        }
    }
    let Some(e) = found else { return };
    if let Some(mut t) = world
        .world_mut()
        .get_mut::<inf_ecs::components::Transform>(e)
    {
        t.translation = inf_ecs::math::Vec3d::new(p.x, p.y, p.z);
    }
}

/// **The centre of the town the flythrough is over** — where a player stands to
/// be *in* Harbour City rather than beside it.
///
/// The settlement set comes through the same
/// `inf_editor_core::settlement::settlements` door `island_gate`'s
/// `walk_target_settlement` uses; the choice is **nearest to `from`**, with the
/// name as the tie-break.
///
/// # Nearest, and not largest, and the difference is 3.8 km
///
/// The obvious pick is the biggest town, and on this island that is **Eastgate,
/// 52 blocks, 3 812 m from the flight's start**. Standing the hero there would
/// certify a *different world from the one the camera draws*: residency in this
/// engine is derived from the streaming source and never from a camera (the
/// law), so the cells around the hero would be the resident ones and the flight
/// would be over a world that had streamed out. A certification row needs its
/// town in **both** — under the anchor and in the frame — and only the town the
/// flight passes over is in both.
fn island_town_centre(design: &inf_island::IslandDesign, from: DVec3) -> DVec3 {
    let d = |s: &inf_editor_core::settlement::Settlement| {
        (s.centre.x - from.x).hypot(s.centre.y - from.z)
    };
    let mut plans = inf_editor_core::settlement::settlements(design);
    plans.sort_by(|a, b| d(a).total_cmp(&d(b)).then(a.name.cmp(&b.name)));
    let best = plans
        .into_iter()
        .next()
        .expect("the island design has a settlement");
    println!(
        "island town centre: {} — {} blocks at ({:.0}, {:.0}), {:.0} m from the flight's start",
        best.name,
        best.blocks.len(),
        best.centre.x,
        best.centre.y,
        d(&best),
    );
    DVec3::new(best.centre.x, 0.0, best.centre.y)
}

/// Where the island's recipe lives.
fn island_recipe() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../samples/island/island.toml")
}

/// Build and cook the **full** island — the 51.38 km² one.
///
/// `None` when the tile cache is not filled, which is every machine that has not
/// run `inf island fetch`. That is why the arm below is `#[ignore]`d: this
/// repository's CI never fetches, and an island built from nothing would be a
/// number about an empty world.
fn cook_island(tmp: &Path) -> Option<PathBuf> {
    let recipe = inf_island::IslandRecipe::load(&island_recipe()).ok()?;
    let plan = inf_island::plan_tiles(&recipe).ok()?;
    let missing = plan.missing_in(&recipe.cache_dir());
    if !missing.is_empty() {
        println!(
            "SKIP: {} of {} source tiles are missing from {} — run `inf island fetch`",
            missing.len(),
            plan.len(),
            recipe.cache_dir().display()
        );
        return None;
    }
    let started = std::time::Instant::now();
    let build = inf_island::build_island(&recipe, &inf_island::BuildOptions::default()).ok()?;
    println!("ISLAND BUILD: {:.1} s", started.elapsed().as_secs_f64());
    print!("{}", build.report.summary());
    // **WHAT THE ROADS COST, PER MESH** (wave ROAD1b, clause 4). The ROAD1
    // audit's carried 18 is that the four road meshes were priced by nothing;
    // this is the census, taken where the island is actually built, and the
    // ceiling `inf_player::budget::ROAD_TRIANGLES_CEILING` is read by name
    // beside it. It bounds CONTENT — what the cook carries, the pack holds and
    // the streamer pages — rather than a frame cost: every one of these meshes
    // clears `vgeom.min_triangles` and reaches the raster through a meshlet DAG.
    if let Some(m) = build.mesh.as_ref() {
        let mut total = 0u64;
        if let Some((mesh, _)) = m.carriageway.as_ref() {
            println!(
                "ISLAND ROADS: carriageway {} v / {} t",
                mesh.vertex_count(),
                mesh.triangle_count()
            );
            total += mesh.triangle_count() as u64;
        }
        for (part, a) in &m.furniture {
            println!(
                "ISLAND ROADS: {:>22} {} v / {} t",
                part.label(),
                a.vertex_count(),
                a.triangle_count()
            );
            total += a.triangle_count() as u64;
        }
        println!(
            "ISLAND ROADS: {total} triangles over {} draw(s), against a ceiling of {}",
            1 + m.furniture.len(),
            ROAD_TRIANGLES_CEILING
        );
        assert!(
            total <= ROAD_TRIANGLES_CEILING,
            "the island's road meshes hold {total} triangles against a ceiling of {ROAD_TRIANGLES_CEILING} {RATCHET_NOTE}"
        );
    }
    let proj = tmp.join("island");
    inf_project::ProjectManifest::new(&recipe.name, "blank-3d")
        .save(&proj)
        .ok()?;
    inf_island::write_content(&build, &proj.join("Content")).ok()?;
    let out = tmp.join("island-out");
    let started = std::time::Instant::now();
    cook(&proj, &out, &CookOptions::default()).ok()?;
    println!("ISLAND COOK: {:.1} s", started.elapsed().as_secs_f64());
    Some(out)
}

/// **THE ISLAND'S OWN FRAME NUMBERS**, at shipping resolution, shipped settings
/// and lit.
///
/// `#[ignore]`d because it needs the fetched elevation cache and a real adapter,
/// and because building 51 km² of terrain is a minute of work no unit run should
/// pay. Run it with:
///
/// ```text
/// cargo test --release -p inf-player --test fps_instrument -- --ignored the_island --nocapture
/// ```
///
/// **It REPORTS and never asserts.** The ceilings in `inf_player::budget` are set
/// from the composed city — the scene every previous wave measured — and
/// asserting them over a different world would re-pin a ratchet by accident.
/// What this arm is for is the ledger: the island in the units the budget is
/// written in, beside the city's own numbers, on the same machine.
#[test]
#[ignore = "needs the fetched island cache and a real GPU; builds 51 km2 of terrain"]
fn the_island_at_shipping_resolution() {
    let Ok(gpu) = GpuContext::headless() else {
        println!("SKIP the_island_at_shipping_resolution: no GPU adapter");
        return;
    };
    let info = gpu.adapter.get_info();
    let tmp = tempfile::tempdir().expect("a temp dir");
    let Some(pack) = cook_island(tmp.path()) else {
        return;
    };

    // The island STREAMS: without the cell streamer the world holds only what
    // `AlwaysLoaded` kept, which is a frame of a sun and a heightfield.
    let mut fx = {
        let source = PackLevelSource::open(&pack).expect("the island pack opens");
        let mut built = inf_player::build_world_from_pack(&source).expect("the world builds");
        let partition = built.take_partition();
        let pcg = built.pcg_context();
        let record = built.render;
        let materials = std::sync::Arc::new(source.material_content());
        let reader = std::sync::Arc::new(
            PackReader::open(&pack.join(inf_player::level::PACK_FILE)).expect("the pack maps"),
        );
        let skinned = inf_player::skinned::SkinnedRegistry::from_pack(reader.clone());
        let voxel_assets = inf_player::voxel::VoxelRegistry::from_pack(reader.clone());
        let scatter_meshes = inf_player::scatter_mesh::from_pack(&reader);
        let vmeshes = inf_player::vmesh::VmeshRegistry::from_pack(reader)
            .expect("the island's meshlet DAGs index");
        let mut sim = inf_player::sim_from_built(built);
        // **Cells first, then the terrain** — `run_headless`'s own order, and it
        // matters: terrain residency is derived from the sim's entities and a
        // freshly-activated cell brings some of them in. The first draft of this
        // arm attached neither and measured a frame of sky and water: **0 terrain
        // tiles, 0 instances**, which the anti-vacuity assertion caught.
        inf_player::attach_cell_streaming(&mut sim, &partition, pcg);
        inf_player::attach_terrain_streaming(
            &mut sim,
            &inf_player::TerrainContent::Pack(
                PackLevelSource::open(&pack).expect("the island pack re-opens"),
            ),
        );
        Fixture {
            sim,
            vmeshes,
            skinned,
            voxel_assets,
            scatter_meshes,
            record,
            materials,
        }
    };

    let recipe = inf_island::IslandRecipe::load(&island_recipe()).expect("the recipe loads");
    let design = inf_island::read_design(&recipe).expect("the design reads");
    let s = design.start(0.0);
    let from = DVec3::new(s.x, s.y, s.z);
    let path = move |step: u64, w: u32, h: u32| island_fly(step, w, h, from);
    // Wave NPC1e's certification staging, read once: where the hero starts (so
    // the arc's own rows can be put back exactly there) and where its town is.
    let town_centre = island_town_centre(&design, from);
    let hero_home = hero_at(&fx.sim).expect("the island has a player-controlled hero");
    println!(
        "island hero start: ({:.0}, {:.0}, {:.0}); the town centre is {:.0} m away",
        hero_home.x,
        hero_home.y,
        hero_home.z,
        (DVec3::new(town_centre.x, hero_home.y, town_centre.z) - hero_home).length()
    );

    // **THE REAL ISLAND'S OWN FIXED STEP** (island wave I8a). The hero starts in
    // Harbour City, so the settlement blocks around it activate and the collider
    // band admits their parts — which is what a step over a city costs and what
    // `CITY_STEP_BUDGET_MS` is a ratchet about. Measured here rather than on the
    // CI fixture because the fixture's reservations take the town's 76 m grid and
    // a number about a city has to come from one.
    //
    // **Reported, never asserted**, on the same terms as every frame number
    // below: the ceiling is set from the composed phase-30 city and asserting it
    // over a different world would re-pin a ratchet by accident.
    {
        for _ in 0..STEP_WARMUP {
            fx.sim
                .step_once(inf_player::runtime_sim::RuntimeInput::default());
        }
        fx.sim.set_step_profiling(true);
        let mut acc = inf_player::step_profile::StepProfile::default();
        let t0 = std::time::Instant::now();
        for _ in 0..STEP_SAMPLES {
            fx.sim
                .step_once(inf_player::runtime_sim::RuntimeInput::default());
            acc.accumulate(&fx.sim.step_profile());
        }
        let wall = t0.elapsed().as_secs_f64() * 1000.0 / STEP_SAMPLES as f64;
        acc.scale(1.0 / STEP_SAMPLES as f64);
        fx.sim.set_step_profiling(false);
        let (tracked, touching) = fx.sim.bridge3d().world().contact_pair_counts();
        // **THIS IS THE RATCHET'S CLOCK** (island wave I8c, clause 3), and the
        // rows below are not: nothing is rendering here, so the wall clock and
        // the phase sum agree to a thousandth. Every configuration's own
        // `fixed step` line reports the same phases inside a wall clock that also
        // holds a render thread and a driver, and `print_step_clocks` reconciles
        // the two. `island_gate`'s step-profile arm reads this one.
        println!(
            "=== THE REAL ISLAND'S FIXED STEP === {wall:.3} ms/step (phases sum to \
             {:.3}) against a {:.1} ms ratchet; {} bodies, {} ADMITTED structure \
             colliders, {tracked} contact pairs ({touching} touching)",
            acc.total_ms(),
            inf_player::budget::CITY_STEP_BUDGET_MS,
            fx.sim.bridge3d().body_count(),
            fx.sim.bridge3d().admitted_structures(),
        );
        for (n, ms) in acc.dearest_first() {
            if ms <= 0.02 {
                continue;
            }
            println!(
                "  step {n:<18} {ms:7.3} ms ({:4.1} %)",
                ms / acc.total_ms().max(1.0e-9) * 100.0
            );
        }
    }

    let (shipped, tier) = shipped_settings(&gpu, fx.record);
    // The lit configuration is the SHIPPED one with the authorable half turned
    // on, through the same door — never a hand-built settings block, which would
    // be a price for a frame nobody ships.
    let lit_record = inf_scene::RenderSettingsRecord {
        bloom_enabled: true,
        ssao_enabled: true,
        taa: true,
        shadows_enabled: true,
        gi_enabled: true,
        ..fx.record
    };
    let (mut lit, lit_tier) = shipped_settings(&gpu, lit_record);
    lit.vsm.enabled = true;
    // **THE PHOTOREAL ROW** (wave VIS1a): the lit configuration with screen-space
    // reflections on, through the same authoring door -- `ssr_enabled` is a
    // schema-v26 field of the render record, so this is what a level that ticks
    // "Screen-Space" in World Settings produces, not a hand-poked
    // `RenderSettings`. Its AO is GTAO and its prepass carries terrain, skinned,
    // voxel and fracture geometry, so this row is the wave's whole GPU cost in
    // one number, against the LIT row beside it.
    let (mut lit_ssr, _) = shipped_settings(
        &gpu,
        inf_scene::RenderSettingsRecord {
            ssr_enabled: true,
            ..lit_record
        },
    );
    lit_ssr.vsm.enabled = true;
    // **THE WHOLE VIS STACK** (wave VIS1b): the LIT+SSR row plus auto exposure,
    // the sun glare and the lens trio — every one of them a schema-v26 field of
    // the render record, so this is what a level that ticks all of them in World
    // Settings produces rather than a hand-poked `RenderSettings`.
    //
    // The three costs are separable by construction and the ledger separates
    // them: the exposure histogram is its own `exposure` graph row, the glare is
    // its own `flare` row, and the lens trio has none because it is arithmetic
    // and two texture fetches inside `tonemap`.
    let (mut lit_vis, _) = shipped_settings(
        &gpu,
        inf_scene::RenderSettingsRecord {
            ssr_enabled: true,
            exposure_mode: 1,
            bloom_karis: true,
            flare_enabled: true,
            flare_intensity: 0.35,
            flare_ghost_count: 4,
            flare_halo: 0.3,
            flare_streak: 0.25,
            vignette_intensity: 0.35,
            vignette_smoothness: 0.4,
            chromatic_aberration: 2.0,
            grain_intensity: 0.08,
            grain_size: 1.5,
            ..lit_record
        },
    );
    lit_vis.vsm.enabled = true;
    // **THE ALTERNATIVE, PRICED** (island wave I7b) — **and the control island
    // wave VSM2 beat without taking it.**
    //
    // I7b measured the lit island's whole GPU frame as `vsm-raster`, and the
    // reason was not the world: the dirty split read **0 pages re-cast** every
    // frame — nothing under a page ever moves — against ~530 "moved" and ~400
    // "re-slotted", which is the clipmap grid shifting under a camera that
    // travels 0.9 m a frame against a level-0 page **1.0 m** wide
    // (`2 x first_level_extent_m / clipmap_pages_per_side`, 2 x 32 / 64).
    //
    // So the third configuration is the shipped lit one with the first clipmap
    // level widened 4x, which quarters every level's snap rate. It is
    // **reported, never shipped**: `first_level_extent_m` is a product decision
    // about shadow sharpness, and the wave that measures an alternative is not
    // the wave that gets to pick it.
    //
    // It stays because it is still the honest control for the LIT row above.
    // I7b priced it at `vsm-raster` **15.931 ms** against the shipped ladder's
    // 30.001; wave VSM2 — which keys shadow-page residency to the **world cell**
    // rather than to the grid label, so a scroll re-labels instead of re-drawing
    // — takes the SHIPPED ladder to **4.58-4.61 ms**, three and a half times
    // better than the trade, at level 0's full resolution. The `moved` column
    // below is what changed: **532.0 a frame -> 2.2**.
    let mut lit_coarse = lit;
    lit_coarse.vsm.first_level_extent_m = lit.vsm.first_level_extent_m * 4.0;
    let clamped = !(lit.shadows.enabled && lit.gi.enabled);
    println!(
        "=== THE ISLAND on {} ({:?}) — tier {tier:?}, {ROUNDS} rounds x {FRAMES} frames, MIN of rounds ===",
        info.name, info.device_type
    );
    if clamped {
        println!(
            "the lighting stack is clamped off at tier {lit_tier:?} — the LIT rows below are the shipped ones"
        );
    }
    println!(
        "flythrough from ({:.0}, {:.0}, {:.0}) east at {ISLAND_FLY_STEP_M} m a frame, eye +40 m",
        from.x, from.y, from.z
    );

    for (label, settings) in [
        ("SHIPPED", shipped),
        ("LIT", lit),
        ("LIT+SSR", lit_ssr),
        ("LIT+VIS", lit_vis),
        ("LIT-COARSE-CLIPMAP", lit_coarse),
        // ── **WAVE NPC1e'S CERTIFICATION PAIR** ───────────────────────────
        //
        // **The island a player boots, standing in its town, at the hour that
        // town is on the street.** Every other row in this list has the hero at
        // the level's authored player start and the clock at 02:18 local — a
        // true configuration, and not the one to certify at, because the start
        // is outside the blocks (measured: every one of the level's own
        // residents is `Far` from it) and at two in the morning the town is
        // correctly asleep.
        //
        // So this row does exactly two things, and neither of them installs
        // anything: it winds the level's own `TimeOfDay` to **08:30 local**,
        // leaving the island's authored rate alone, and it stands the hero at
        // **the thickest part of that town's rush hour** — the resident with the
        // most neighbours inside the ladder's own `Full` radius, derived from the
        // level below rather than chosen. The population is the one
        // `inf_ecs::society::sync_society` derives from the level's own
        // buildings; the clock is the one the recipe ships.
        //
        // *(The NPC1e audit corrected this paragraph: it read "its largest
        // settlement's centre, which is `island_gate`'s own staging", and the
        // code below does neither. `island_town_centre` picks the **nearest**
        // settlement and says why in its own doc — the largest is 3.8 km off the
        // flight — and the centre is only ever REPORTED, because standing there
        // is exactly the vacuous staging this row's own finding is about.)*
        //
        // **It comes FIRST of the four**, before the arc's own crowd rows, and
        // that ordering is forced rather than chosen: `set_crowd_population`
        // marks the population hand-installed and the society stops deriving —
        // a one-way door for the session — so the derived row cannot follow a
        // hand-installed one.
        ("LIT+VIS (rush hour, the island's own society)", lit_vis),
        // …and its control, taken immediately after at the same clock and with
        // the hero in the same place, so the town's cost is a difference between
        // two adjacent rows that differ in one thing.
        ("LIT+VIS (rush hour, town cleared)", lit_vis),
        // **THE CROWD'S CONTROL**, and it is adjacent for a reason. The rows
        // above are taken in order over a world that keeps streaming, so the
        // `LIT+VIS` row at the top of the list has seen less of the island than
        // the crowd row at the bottom: its scatter count and its terrain
        // residency are not the same numbers. A delta taken across that gap
        // would attribute the streaming to the crowd. This row is `LIT+VIS`
        // again, immediately before the population goes in, so the two rows
        // differ in one thing.
        //
        // **The hero goes back to its authored start here**, so the three arc
        // rows share one staging and the crowd the row installs stands where
        // every NPC1a/b/c/d table put it.
        ("LIT+VIS (crowd control)", lit_vis),
        // **WAVE NPC1b'S HEADLINE ROW.** The same `LIT+VIS` settings, the same
        // flight, the same island — with a thousand NPCs standing in Harbour
        // City. Last, and through the same loop, so the crowd's cost is the
        // DIFFERENCE between two rows taken the same way rather than a number
        // from a second harness nobody can compare against.
        ("LIT+VIS+CROWD", lit_vis),
        // **THE AFTER-CONTROL, and it is not belt-and-braces.** These rows are
        // taken back to back on one adapter, and a GPU that has been rendering
        // 1080p for five minutes is not the GPU that rendered the first row: the
        // island's `scatter` and `terrain` passes — neither of which a crowd can
        // touch — read materially dearer at the bottom of the list than at the
        // top. One control before the crowd cannot see that ramp; a control on
        // each side measures it, and the crowd's cost is the row against the MEAN
        // of its two brackets rather than against whichever bracket flatters it.
        ("LIT+VIS (crowd cleared)", lit_vis),
    ] {
        // **THE CERTIFICATION STAGING** (island wave NPC1e): wind the level's own
        // clock to the morning commute, stand the hero in the town, and let the
        // town get on the street. The rate is left at the island's authored 18,
        // so the row measures a town that is *walking* rather than one frozen
        // mid-stride.
        if label == "LIT+VIS (rush hour, the island's own society)" {
            {
                let w = fx.sim.world_mut().world_mut();
                let mut q = w.query::<&mut inf_ecs::components::TimeOfDay>();
                let mut wound = 0usize;
                for mut tod in q.iter_mut(w) {
                    tod.seconds = (8.5 * 3600.0 - tod.longitude_deg * 240.0).rem_euclid(86_400.0);
                    wound += 1;
                }
                assert!(wound > 0, "the island carries no clock to wind");
            }
            // **AND THE HERO STANDS AMONG ITS PEOPLE, which is not the same
            // place as the town's centre and the difference is the whole
            // staging.**
            //
            // The obvious placement is the settlement's own centre, which is
            // `island_gate`'s staging for the day gate — and on this island the
            // level's authored player start **already is** Harbour City's
            // centre, 0 m away. Measured from there, every one of the level's
            // own residents is `Far`: **0 `Full`, 0 `Near`, 998 `Far`, 0
            // steering intents**. A settlement's centre is its crossroads, and
            // its people live in the blocks around it.
            //
            // So the hero is placed at the **thickest part of the rush hour**:
            // after one warm-up that lets the society derive and the bodies
            // appear, the agent with the most neighbours inside the ladder's own
            // `Full` radius, and the hero stands where it stands. It is derived
            // from the level rather than chosen, it is what "standing in the
            // crowd" means in the units the ladder is written in, and it is the
            // *worst* plausible station rather than an average one — which is
            // the right thing for a certification to price. NPC1d's own law, met
            // from the other side: *a gate staged where a defect cannot appear
            // is a gate that certifies the defect.*
            //
            // **On the ground it stands on, not at y = 0**: a hero buried in a
            // height field is an anchor with metres of Y between it and every
            // resident, which the ladder measures in three dimensions and would
            // read as `Far`.
            for _ in 0..STEP_WARMUP {
                fx.sim
                    .step_once(inf_player::runtime_sim::RuntimeInput::default());
            }
            let (among, thickest) = {
                let w = fx.sim.world().world();
                let mut at: Vec<DVec3> = Vec::new();
                for e in w.iter_entities() {
                    if e.get::<inf_ecs::crowd::CrowdAgent>().is_none() {
                        continue;
                    }
                    if let Some(t) = e.get::<inf_ecs::components::Transform>() {
                        at.push(t.translation.to_dvec3());
                    }
                }
                assert!(
                    !at.is_empty(),
                    "the town materialized nobody to stand among"
                );
                let r = inf_ecs::crowd::DEFAULT_CROWD_RADII.0;
                let mut best = (0usize, at[0]);
                for p in &at {
                    let n = at.iter().filter(|q| (**q - *p).length() <= r).count();
                    if n > best.0 {
                        best = (n, *p);
                    }
                }
                (best.1, best.0)
            };
            let y = fx.sim.terrain_height_at(among.x, among.z);
            set_hero(&mut fx.sim, DVec3::new(among.x, y + 1.5, among.z));
            fx.sim.world_mut().mark_dirty();
            for _ in 0..STEP_WARMUP {
                fx.sim
                    .step_once(inf_player::runtime_sim::RuntimeInput::default());
            }
            println!(
                "  the town's centre is ({:.0}, {:.0}); the thickest part of its \
                 rush hour is ({:.0}, {:.0}), {:.0} m away, with {thickest} \
                 residents inside a {:.0} m ring",
                town_centre.x,
                town_centre.z,
                among.x,
                among.z,
                (DVec3::new(among.x, town_centre.y, among.z) - town_centre).length(),
                inf_ecs::crowd::DEFAULT_CROWD_RADII.0,
            );
            let soc = fx.sim.society_stats();
            let st = fx.sim.crowd_stats();
            // **WHERE ITS PEOPLE ARE, relative to where the player is standing**
            // (island wave NPC1e). The tier ladder is a distance, so a tier
            // census with no distances beside it cannot tell "the town is not
            // around the hero" from "the ladder is broken" — and the first cut
            // of this row read `0 full / 0 near / 998 far` and could say which
            // only by measuring. Nearest, tenth-nearest and median, in metres.
            let mut d: Vec<f64> = {
                let here = hero_at(&fx.sim).unwrap_or(DVec3::ZERO);
                let w = fx.sim.world().world();
                let mut v = Vec::new();
                for e in w.iter_entities() {
                    if e.get::<inf_ecs::crowd::CrowdAgent>().is_none() {
                        continue;
                    }
                    if let Some(t) = e.get::<inf_ecs::components::Transform>() {
                        v.push((t.translation.to_dvec3() - here).length());
                    }
                }
                v
            };
            d.sort_by(f64::total_cmp);
            println!(
                "  the hero stands at ({:.0}, {:.1}, {:.0}); its {} materialized \
                 neighbours are {:.0} / {:.0} / {:.0} m away (nearest, tenth, median)",
                hero_at(&fx.sim).unwrap_or(DVec3::ZERO).x,
                hero_at(&fx.sim).unwrap_or(DVec3::ZERO).y,
                hero_at(&fx.sim).unwrap_or(DVec3::ZERO).z,
                d.len(),
                d.first().copied().unwrap_or(f64::NAN),
                d.get(9).copied().unwrap_or(f64::NAN),
                d.get(d.len() / 2).copied().unwrap_or(f64::NAN),
            );
            println!(
                "=== THE ISLAND'S OWN SOCIETY AT THE RUSH HOUR === {:.2} h local; \
                 {} volumes folded, {} homes offered, {} agents, {} scheduled; \
                 {} full / {} near / {} far / {} dormant; {} steering intents, \
                 {} blocked, {} arrived",
                inf_ecs::sky::local_hour(fx.sim.world()),
                soc.volumes,
                soc.homes,
                soc.agents,
                soc.agents - soc.pending,
                st.per_tier[0],
                st.per_tier[1],
                st.per_tier[2],
                st.per_tier[3],
                st.steered,
                st.blocked,
                st.arrived,
            );
            // ANTI-VACUITY, both halves. A row that certifies "the island with
            // its town" must have a town in it, and a row about the RUSH HOUR
            // must be at the rush hour — a wound clock that did not take would
            // certify a sleeping village and read like a triumph.
            assert!(
                soc.agents > 0 && st.per_tier.iter().sum::<usize>() > 0,
                "the certification row has no population in it: {soc:?}"
            );
            let hour = inf_ecs::sky::local_hour(fx.sim.world());
            assert!(
                (8.0..10.0).contains(&hour),
                "the clock reads {hour:.2} h, so this row is not the rush hour"
            );
            // …and the third half, which is the one the first cut of this row
            // failed: bodies the mover actually visits. `Full` is the only tier
            // that steers, so a certification row with none of it prices a town
            // that never touches `step_character_movement`.
            assert!(
                st.per_tier[0] > 0,
                "no agent is `Full`, so this row prices a crowd that never \
                 enters the mover — the wall the whole arc is about"
            );
        }
        // …and the hero goes back where the level put it, so the three arc rows
        // below share one staging with every NPC1a-d table.
        if label == "LIT+VIS (crowd control)" {
            set_hero(&mut fx.sim, hero_home);
            fx.sim.world_mut().mark_dirty();
            for _ in 0..STEP_WARMUP {
                fx.sim
                    .step_once(inf_player::runtime_sim::RuntimeInput::default());
            }
        }
        if label == "LIT+VIS (rush hour, town cleared)" || label == "LIT+VIS (crowd cleared)" {
            fx.sim.set_crowd_population(Default::default());
            for _ in 0..STEP_WARMUP {
                fx.sim
                    .step_once(inf_player::runtime_sim::RuntimeInput::default());
            }
            let st = fx.sim.crowd_stats();
            assert_eq!(
                st.per_tier, [0; 4],
                "the population did not clear, so this bracket still holds a crowd"
            );
        }
        if label == "LIT+VIS+CROWD" {
            let archetype = island_archetype(&fx.sim);
            fx.sim
                .set_crowd_population(island_population(ISLAND_CROWD_N, archetype, from));
            // The crowd materializes on its first step and takes every
            // structure-stamp miss there is; measuring that would be measuring a
            // step that happens once.
            for _ in 0..STEP_WARMUP {
                fx.sim
                    .step_once(inf_player::runtime_sim::RuntimeInput::default());
            }
            let st = fx.sim.crowd_stats();
            println!(
                "=== THE CROWD AT HARBOUR CITY === {ISLAND_CROWD_N} NPCs in a {:.0} m block: {} full / {} near / {} far / {} dormant",
                2.0 * ISLAND_CROWD_HALF_M,
                st.per_tier[0],
                st.per_tier[1],
                st.per_tier[2],
                st.per_tier[3],
            );
        }
        let m = measure(&gpu, &mut fx, 1920, 1080, settings, &path);
        let r = m.round();
        let cpu_sum: f64 = m.cpu_ms.iter().sum();
        println!(
            "ISLAND 1080p {label}: p50 {:.3} p95 {:.3} p99 {:.3} worst {:.3} ms ({:.1} fps at p50)",
            r.p50,
            r.p95,
            r.p99,
            r.worst,
            1000.0 / r.p50.max(1.0e-9)
        );
        println!(
            "  content    {} instances, {} scatter batches / {} scattered instances, {} vgeom, {} skinned, {} terrain tiles, {} virtual textures",
            m.instances,
            m.scatter_batches,
            m.scatter_instances,
            m.vgeom_instances,
            m.skinned,
            m.terrain_tiles,
            m.vt_textures
        );
        for (i, name) in CPU_STAGE_NAMES.iter().enumerate() {
            println!("  cpu {name:>16}: {:.3} ms", m.cpu_ms[i]);
        }
        println!("  cpu {:>16}: {cpu_sum:.3} ms", "TOTAL");
        print_step_clocks(label, &m);
        // **WHAT THE STEP IS PAYING FOR, PER ROW** (the NPC1e audit). The island
        // printed this census once, in the isolated fixed-step block above, at
        // the hero's authored start — and then the wave attributed **+21.0 ms of
        // p50** to *"the town's own admitted structure colliders"* on a row taken
        // 189 m away, where nothing had counted them. `physics3d sync` and
        // `solver` climbing is a measurement; *which bodies* did it is a claim,
        // and this is the counter that carries it. It is printed for every row so
        // the collider band's share is a subtraction between two of them rather
        // than an inference from one.
        {
            let (tracked, touching) = fx.sim.bridge3d().world().contact_pair_counts();
            println!(
                "  physics {:>14}: {} bodies, {} ADMITTED structure colliders, \
                 {tracked} contact pairs ({touching} touching)",
                "world",
                fx.sim.bridge3d().body_count(),
                fx.sim.bridge3d().admitted_structures(),
            );
        }
        print_record_profile(label, &m);
        println!("  gpu {:>16}: {:.3} ms", "frame", m.gpu_frame_ms);
        let mut passes = m.passes.clone();
        passes.sort_by(|a, b| b.1.total_cmp(&a.1));
        // **Every pass, not the dearest eight** (wave VIS1a). A `take(8)` cannot
        // report a pass that is cheap *now* and is the subject of the wave —
        // `depth-prepass` and `ssao` were both below the cut on the island, which
        // is precisely the information a before/after table needs. The city's lit
        // table already prints on this rule; the island's did not.
        for (name, ms, rec) in passes
            .iter()
            .filter(|(_, ms, rec)| *ms >= 0.0005 || *rec >= 0.0005)
        {
            println!("  gpu {name:>16}: {ms:.3} ms   (record {rec:.3} ms)");
        }
        // **What the shadow pass actually DID** (island wave I7b). `vsm-raster`
        // was 95.1 % of wave I7's lit GPU frame on a world whose casters are a
        // heightfield and one road mesh, and a millisecond count with no page,
        // draw or caster beside it cannot say whether the cost is the drawing or
        // the asking. One line, and it is the pass's own counters.
        if let Some(v) = m.vsm.as_ref() {
            println!("  {label} {}", v.summary());
            if v.frames > 0 {
                println!(
                    "  {label} per rastering frame: {:.1} pages, {:.0} draws, \
                     {:.0} casters, {:.0} invalidation touches, {:.1} cached \
                     pages, {:.1} deferred",
                    v.pages as f64 / v.frames as f64,
                    v.draws as f64 / v.frames as f64,
                    v.casters as f64 / v.frames as f64,
                    v.invalidation_touches as f64 / v.frames as f64,
                    v.cached_pages as f64 / v.frames as f64,
                    v.deferred_pages as f64 / v.frames as f64,
                );
                // **WHY they were dirty** (island wave I7b). "The cache is
                // thrashing" is not a diagnosis; these three sum to
                // `dirty_pages` and say whether the pages moved under the world
                // or the world moved under the pages.
                //
                // Island wave VSM2 is read off the middle column: it was **532.0
                // a frame** and it is **2.2**, because a page's slot now belongs
                // to its world cell and a clipmap scroll re-labels rather than
                // re-draws. What is left in `re-slotted` is the row and column
                // the window newly exposes, which have never been drawn.
                // The middle column's legend is **not** "the page's own matrix"
                // any more (the VSM2 audit): for a clipmap the geometric stamp is
                // the world cell folded with `ClipmapLayout::content_key`, so a
                // level re-centring and an origin rebase — the two the old legend
                // named — are exactly the two that no longer land here.
                println!(
                    "  {label} dirty per rastering frame: {:.1} re-slotted, \
                     {:.1} moved (the box it draws: the sun's quantum, the \
                     along-light snap), {:.1} re-cast (something under it)",
                    v.dirty_slot as f64 / v.frames as f64,
                    v.dirty_geometry as f64 / v.frames as f64,
                    v.dirty_casters as f64 / v.frames as f64,
                );
                // **WHAT THE PASS HANDS OVER** (island wave I8c). `vsm-raster`
                // was 6.087 ms of an 18.0 ms lit GPU frame with 0.657 ms of
                // recording behind it, so the cost is on the device — and pages,
                // draws and casters are three counts of *asks*. A draw's price is
                // its index count, and a geometry group is one per resident
                // TERRAIN TILE at `6 × VSM_TERRAIN_CASTER_CELLS²` indices against
                // a cube's 36. This is the line that says which of the two the
                // frame is made of, and the level histogram beside it says how
                // much world each of those rectangles covers.
                println!(
                    "  {label} submitted per rastering frame: {:.0} indices \
                     ({:.0} terrain, {:.1} %; {:.0} meshlet-asset, {:.1} %) over \
                     {:.0} draws ({:.0} terrain, {:.0} meshlet at mean classic \
                     level {:.2}); pages by level {}",
                    v.indices_drawn as f64 / v.frames as f64,
                    v.indices_terrain as f64 / v.frames as f64,
                    v.indices_terrain as f64 / (v.indices_drawn.max(1)) as f64 * 100.0,
                    v.indices_vgeom as f64 / v.frames as f64,
                    v.indices_vgeom as f64 / (v.indices_drawn.max(1)) as f64 * 100.0,
                    v.draws as f64 / v.frames as f64,
                    v.draws_terrain as f64 / v.frames as f64,
                    v.draws_vgeom as f64 / v.frames as f64,
                    v.vgeom_level_sum as f64 / v.vgeom_casters.max(1) as f64,
                    v.levels_summary(),
                );
            }
        }
        let submitted = cpu_sum - m.cpu_ms[4] - m.cpu_ms[5];
        let pipelined = submitted.max(m.gpu_frame_ms);
        println!(
            "  PIPELINED ESTIMATE {pipelined:.3} ms ({:.1} fps) = max(CPU without the wait or the stopwatch {submitted:.3}, GPU frame {:.3})",
            1000.0 / pipelined.max(1.0e-9),
            m.gpu_frame_ms
        );
        println!(
            "  DISTANCE FROM 60 fps: p50 {:+.3} ms, p95 {:+.3} ms against a {SHIPPING_FRAME_BUDGET_MS} ms frame",
            r.p50 - SHIPPING_FRAME_BUDGET_MS,
            r.p95 - SHIPPING_FRAME_BUDGET_MS
        );
        // Anti-vacuity: the frame drew the island, not an empty world.
        assert!(
            m.terrain_tiles > 0,
            "{label}: the island's frame drew no terrain tile"
        );
        // **What the crowd cost the RENDERER** (wave NPC1b) — read off a
        // projection of the same sim through the passes' own planners, so the
        // draw count and the palette bytes are the ones the frame above paid.
        // **What GI stopped doing** (island wave NPC1e). Printed for every row,
        // because the number that matters is a *pair*: `candidates` says the
        // volume kept something and `skinned_rejected` says the pre-reject
        // dropped something, and either alone can be read as the other.
        // **…and what SCATTER did** (wave VEN1a audit). The venue wave gave the
        // scatter path its first route into the bounce, with two whole-batch
        // rejects and a walk ceiling in front of it — three numbers that were
        // counted and then read by nobody, which is the state
        // `skinned_rejected` was added to escape. `scatter_decimated` above zero
        // is a reportable loss of fidelity; on the island it is expected to be
        // zero, and a number nobody prints is a number nobody can notice moving.
        println!(
            "  gi {:>16}: {} candidates, {} voxelized, {} dropped, {} skinned instances rejected whole, \
             {} scatter batches rejected whole, {} strided",
            "audit",
            m.gi.candidates,
            m.gi.voxelized,
            m.gi.dropped,
            m.gi.skinned_rejected,
            m.gi.scatter_rejected,
            m.gi.scatter_decimated
        );
        if label == "LIT+VIS+CROWD" || label == "LIT+VIS (rush hour, the island's own society)" {
            let mut scene = RenderScene::default();
            let mut voxels = inf_voxel::VoxelVolumes::new();
            let mut debris = inf_render::DebrisCache::default();
            inf_player::render::project_scene_full(
                &mut scene,
                &fx.sim,
                1.0,
                &fx.vmeshes,
                &fx.skinned,
                &voxels,
                &mut debris,
                None,
                &fx.scatter_meshes,
                &std::collections::HashMap::new(),
            );
            let _ = &mut voxels;
            let plan = inf_render::plan_skinned_batches(&scene);
            let proxies = scene
                .skinned
                .iter()
                .filter(|i| i.shadow == inf_render::SkinnedShadow::Proxy)
                .count();
            // **The crowd shadow LOD's own share** (island wave NPC1e): the
            // instances that cast nothing at all, counted here off the same
            // projection so the ledger's shadow line and its draw line come from
            // one scene.
            let unshadowed = scene
                .skinned
                .iter()
                .filter(|i| i.shadow == inf_render::SkinnedShadow::None)
                .count();
            println!(
                "  CROWD RENDER COST: {} skinned instances in {} draw call(s), {} palette blocks = {:.2} MB of atlas",
                scene.skinned.len(),
                plan.runs.len(),
                plan.blocks,
                plan.matrices as f64 * 64.0 / (1024.0 * 1024.0),
            );
            println!(
                "  CROWD SHADOWS: {proxies} proxy casters in 1 group, {unshadowed} past the shadow LOD, {} skinned groups, {} of a {} ceiling",
                inf_render::skinned_caster_groups(&scene) - usize::from(proxies > 0),
                inf_render::skinned_caster_groups(&scene),
                inf_render::VSM_MAX_GROUPS,
            );
            // **READ THE GPU ROWS ABOVE AGAINST THE BRACKETS, NOT PASS FOR
            // PASS.** This row's frame is CPU-bound — the crowd's fixed step,
            // projection and recording put the CPU total well past the GPU frame,
            // so the device idles for much of it. Measured, every pass inflates,
            // including `scatter` and `terrain`, which a crowd adds nothing to:
            // whatever the cause, a pass-for-pass difference against a bracket is
            // not a crowd cost. The numbers this row publishes AS the crowd's are
            // the two structural lines above and the CPU stages.
            println!(
                "  (this row is CPU-bound; its per-pass GPU milliseconds are not comparable pass-for-pass with the bracket rows)"
            );
        }
    }
    println!(
        "Reported, never asserted: the ceilings in `inf_player::budget` are set from the composed city, and asserting them over a different world would re-pin a ratchet by accident."
    );
}
