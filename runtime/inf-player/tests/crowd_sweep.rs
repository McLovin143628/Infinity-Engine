//! **THE N-SWEEP** (wave NPC1a) — the first crowd measurement this engine has
//! ever taken.
//!
//! # What this is, and what it is deliberately not
//!
//! It is an **instrument**, in the I4 sense: it REPORTS. Nothing in the sweep
//! asserts a millisecond, because a millisecond in the `dev` profile on a shared
//! runner is a fact about the runner, and because the numbers do not exist yet —
//! before NPC1a there was no crowd, no tier, no `NPC_STEP_BUDGET_MS` and nothing
//! anywhere in this tree that had ever put a thousand NPCs in a world and looked
//! at the clock. The three things it *does* assert are the ones that are
//! machine-independent or structural:
//!
//! * **(a) zero cost when absent** — with no population the crowd phase reads
//!   under 0.005 ms (0.0001 measured, i.e. the clock's own resolution: the step
//!   returns on a `contains_resource` before it allocates) and the trace is
//!   byte-identical to the same scene without one;
//! * **(b) the tier ladder is what makes N affordable** — the same N, banded and
//!   all-`Full`, must not produce the same work: the banded run poses strictly
//!   fewer characters and folds strictly fewer trace bytes;
//! * **(c) the trace re-shape's arithmetic** — bytes per step scale with
//!   `AGENT_TRACE_BYTES` per non-posing agent rather than with a posed
//!   character's 6 476.
//!
//! Everything else is a table, printed, for the NPC1b brief to read.
//!
//! # The scene
//!
//! The **fps instrument's own composed scene** — the phase-30 city (1 000
//! grammar buildings, 370 468 solids), a streamed terrain paging beneath it and
//! the phase-29 wizard character — cooked exactly as `inf cook` cooks it. That
//! is a deliberate choice over the island fixture: it is the scene
//! `CITY_STEP_BUDGET_MS` is asserted over and the scene every published fps
//! number in the ledger comes from, so a crowd measured on it can be read
//! straight against those numbers instead of against a second baseline nobody
//! has. The NPCs are spawned along the city's own scripted drive line, which is
//! where a town's population would be.
//!
//! The archetype is read **off the scene's own hero** — its `SkeletalMesh` and
//! `AnimStateMachine` GUIDs — so every test NPC is a copy of the character this
//! scene already has, sharing one vertex buffer, one skeleton, one machine and
//! every clip (the renderer `Arc`-dedupes by `(mesh, skeleton)`); a thousand of
//! them cost one of each.
//!
//! **The GPU columns are NPC1b's** (`draws` / `blocks` / `groups` / `palette`).
//! They are read off the projected `RenderScene` through the renderer's own
//! planners — `plan_skinned_batches` and `skinned_caster_groups`, the functions
//! the passes themselves call — so the palette column is a measurement rather
//! than the multiplication it was in NPC1a. No GPU is involved: both planners are
//! pure functions of a scene.
//!
//! **And that character is a 20-joint rig, not the island's 161-bone hero**
//! (NPC1a audit — this paragraph said "161-bone" while the sweep's own `(c)`
//! line printed "this scene rigs its character at 20 joints", which is derived
//! from the pose section it measured). It matters in one direction and it is
//! worth being explicit about which: a posed character here is **836 B** against
//! the island's 6 476, so every *byte* column below understates an
//! island-class crowd by about 8× on the pose section — and so does the
//! `animation` millisecond, which is a pose evaluation per joint. The palette
//! column does not: wall 3's palette is one 16 KiB power-of-two block per
//! skinned character whatever its joint count. **A thousand island-class NPCs
//! cost more than the 6.8 ms step measured here**, and the ladder is worth
//! correspondingly more than the 2.0× this table prices it at.
//!
//! The **palette** column is the one exception and it is worth being explicit
//! about, because NPC1a's version of this paragraph got it backwards: wall 3's
//! block was one 16 KiB power-of-two allocation per skinned character whatever
//! its joint count, so the column did NOT understate a 161-bone crowd. NPC1b's
//! atlas packs tightly, so it does now — 20 joints is 1 280 B a block where the
//! island's hero is 10 304.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use glam::DVec3;
use uuid::Uuid;

use inf_asset::PackReader;
use inf_ecs::crowd::{
    CrowdArchetype, CrowdBand, CrowdRecord, CrowdRoute, AGENT_TRACE_BYTES, DEFAULT_CROWD_RADII,
};
use inf_editor_core::samples;
use inf_packager::{cook, CookOptions};
use inf_player::level::PackLevelSource;
use inf_player::render::project_scene_full;
use inf_player::runtime_sim::{RuntimeInput, RuntimeSim};
use inf_project::ProjectManifest;
use inf_render::RenderScene;

/// The population sizes the sweep walks. `0` is the control, and it is the arm
/// that says the whole system costs a level without a crowd nothing at all.
const NS: [usize; 5] = [0, 1, 10, 100, 1000];

/// Steps discarded before anything is timed — `fps_instrument`'s discipline: the
/// first steps seat the collider band, mesh the terrain tiles and take every
/// `structure_stamps` miss there is, and they also materialize the whole crowd.
/// Measuring them would be measuring a step that happens once.
const WARMUP: usize = 40;

/// Steps timed per round.
const SAMPLES: usize = 40;

/// Independent rounds; the cheapest by the step's own total is reported
/// (MIN-of-rounds, the instrument's discipline).
const ROUNDS: usize = 3;

/// **Where the crowd stands**: a square block of this half-extent, in metres,
/// centred on the drive line ahead of the camera.
///
/// 160 m is chosen against the ladder rather than for looks: it reaches past
/// `DEFAULT_CROWD_NEAR_M` (96 m) so a realistic spread produces a real mix of
/// `Full`, `Near` and `Far`, and stops short of `DEFAULT_CROWD_FAR_M` (512 m) so
/// the banded run is not quietly measuring an empty world.
const BLOCK_HALF_M: f64 = 160.0;

/// A ladder wide enough that every agent is `Full` — the control the banded run
/// is priced against.
const ALL_FULL: (f64, f64, f64) = (1.0e9, 1.0e9, 1.0e9);

// ── the fixture ─────────────────────────────────────────────────────────────

/// Scaffold and cook the composed instrument level — `fps_instrument`'s
/// `cook_instrument`, so both files measure the same world.
fn cook_scene(tmp: &Path) -> PathBuf {
    let proj = tmp.join("proj");
    ProjectManifest::new("Crowd Sweep", "blank-3d")
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
    cook(&proj, &out, &CookOptions::default()).expect("the sweep scene cooks");
    out
}

/// The pack's sim plus the render stores a projection resolves against.
struct Fixture {
    sim: RuntimeSim,
    vmeshes: inf_player::vmesh::VmeshRegistry,
    skinned: inf_player::skinned::SkinnedRegistry,
    voxel_assets: inf_player::voxel::VoxelRegistry,
    scatter_meshes: inf_render::ScatterMeshes,
}

fn open(pack: &Path) -> Fixture {
    let source = PackLevelSource::open(pack).expect("the pack opens");
    let built = inf_player::build_world_from_pack(&source).expect("the pack world builds");
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
    }
}

/// **The archetype, read off the scene's own hero.**
///
/// A hard-coded GUID table here would be a second place the starter character's
/// identity is written down, and the two would disagree the first time the
/// sample moved — `island_gate`'s `content_assets` doctrine, one level up.
/// Panics if the scene has no rigged character at all, because a sweep whose
/// NPCs pose nothing measures nothing.
fn archetype_from_hero(sim: &RuntimeSim) -> CrowdArchetype {
    let w = sim.world().world();
    let mut best: Option<(Uuid, CrowdArchetype)> = None;
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
        let a = CrowdArchetype::humanoid(sk.mesh, sk.skeleton, sm);
        // Ascending `Guid`, so the pick is a function of the level's contents.
        if best.as_ref().is_none_or(|(bg, _)| g.0 < *bg) {
            best = Some((g.0, a));
        }
    }
    let (guid, a) = best.expect(
        "the sweep scene has no rigged character to copy — every NPC would pose \
         nothing and the whole measurement would be of an empty pipeline",
    );
    println!(
        "archetype: hero {guid} — mesh {:?}, skeleton {:?}, machine {:?}",
        a.mesh, a.skeleton, a.sm
    );
    assert!(
        a.skeleton.is_some() && a.sm.is_some(),
        "the hero has no skeleton or no state machine, so a copy of it would not \
         run the pose pipeline this sweep exists to price"
    );
    a
}

/// **N test NPCs on the city's drive line**, walking short there-and-back routes
/// across the street.
///
/// Deterministic and positional, `island_gate`'s discipline: the population is a
/// pure function of `n` and the drive line, so two runs of the sweep place the
/// same thousand people in the same thousand doorways. GUIDs come off a fixed
/// namespace so they never collide with the level's own.
fn population(n: usize, archetype: CrowdArchetype) -> BTreeMap<Uuid, CrowdRecord> {
    const NAMESPACE: u128 = 0x4e50_4331_615f_4147_454e_5400_0000_0000;
    let centre = samples::city_drive_point(0);
    let mut out = BTreeMap::new();
    for i in 0..n {
        // A deterministic lattice inside the block — no RNG, because the
        // *placement* is fixture content and the house rule is that anything
        // stateful folds into the trace. The per-agent variation the crowd
        // itself needs comes from `agent_rand`, inside the engine.
        let side = (n as f64).sqrt().ceil().max(1.0);
        let (ix, iz) = ((i as f64) % side, (i as f64 / side).floor());
        let span = 2.0 * BLOCK_HALF_M;
        let x = centre.x - BLOCK_HALF_M + span * (ix + 0.5) / side;
        let z = centre.z - BLOCK_HALF_M + span * (iz + 0.5) / side;
        let from = DVec3::new(x, centre.y, z);
        out.insert(
            Uuid::from_u128(NAMESPACE | i as u128),
            CrowdRecord::walking(
                archetype,
                // Six metres across the street and back — long enough that
                // an agent crosses a tier boundary over a sweep, short enough
                // that the block stays a block. `between` is NPC1a's own
                // two-point ping-pong, kept by name so this sweep measures the
                // route it always measured.
                CrowdRoute::between(from, from + DVec3::new(0.0, 0.0, 6.0), 1.4),
            ),
        );
    }
    out
}

// ── one measurement ─────────────────────────────────────────────────────────

/// **The trace, section by section** — what one step's `state_bytes` is made of.
///
/// `state_bytes` is a bincode snapshot followed by nine appended sections, and
/// the whole point of NPC1a's re-shape is which of them a crowd agent lands in.
/// Reading them separately is what turned this wave's own trace arithmetic from
/// a story into a number: the pose is **not** where a Far agent's bytes go, and
/// nothing but a per-section read could have said so.
#[derive(Debug, Clone, Copy, Default)]
struct TraceSections {
    snapshot: usize,
    deform: usize,
    pose: usize,
    cloth: usize,
    hair: usize,
    gameplay: usize,
    crowd: usize,
}

impl TraceSections {
    fn of(sim: &RuntimeSim, total: usize) -> Self {
        let w = sim.world();
        let deform = inf_ecs::deform::deform_state_bytes(w).len();
        let pose = inf_ecs::pose::pose_state_bytes(w).len();
        let cloth = inf_ecs::cloth::cloth_state_bytes(w).len();
        let hair = inf_ecs::hair::hair_state_bytes(w).len();
        let gameplay = inf_ecs::door::door_state_bytes(w).len()
            + inf_ecs::item::item_state_bytes(w).len()
            + inf_ecs::weapon::weapon_state_bytes(w).len()
            + inf_ecs::weapon::health_state_bytes(w).len();
        let crowd = inf_ecs::crowd::crowd_state_bytes(w).len();
        Self {
            // The residue, not an assumption: whatever the nine appended
            // sections do not account for is the leading bincode snapshot.
            snapshot: total.saturating_sub(deform + pose + cloth + hair + gameplay + crowd),
            deform,
            pose,
            cloth,
            hair,
            gameplay,
            crowd,
        }
    }
}

/// Everything the sweep records for one `(N, ladder)` cell.
#[derive(Debug, Clone, Copy)]
struct Row {
    n: usize,
    banded: bool,
    step_ms: f64,
    crowd_ms: f64,
    /// The `society` phase (NPC1d) — on this scene, a settled one: the sweep's
    /// fixture derives its population once and folds nothing again.
    society_ms: f64,
    anim_ms: f64,
    phys_ms: f64,
    solver_ms: f64,
    trace_bytes: usize,
    posed: usize,
    per_tier: [usize; 4],
    instances: usize,
    scatter_batches: usize,
    skinned: usize,
    projection_ms: f64,
    sections: TraceSections,
    gpu: GpuCost,
}

/// **What the crowd costs the RENDERER**, measured off the projected scene
/// through the renderer's own planners rather than multiplied out (wave NPC1b).
///
/// Every field here used to be one line of arithmetic in `Row::palette_mb`, and
/// its comment said the palette was "uploaded twice per frame (the main pass and
/// the prepass)". The prepass half was wrong — `SkinnedMeshNode::sync` is keyed
/// on `(scene.version, origin)` and is free the second time a frame calls it —
/// and the second upload it should have named is the SHADOW path, which keeps its
/// own per-caster palette buffers. Both are read separately now.
#[derive(Debug, Clone, Copy, Default)]
struct GpuCost {
    /// Draw calls the skinned pass issues — one per `(mesh)` run.
    draws: usize,
    /// Distinct palette blocks in the atlas. Fewer than `skinned` is the crowd's
    /// sharing arriving at the renderer.
    blocks: usize,
    /// Bytes the skinned pass uploads a frame: the whole atlas, once.
    atlas_bytes: usize,
    /// Bytes the page raster uploads a frame: one power-of-two block per skinned
    /// caster that really gets one, which is what `vsm_raster::sync_skinned`
    /// writes.
    ///
    /// **The filter has to name every mode that gets no slot** (island wave
    /// NPC1e). It read `!= Proxy` when `Proxy` was the only mode without a
    /// palette; `SkinnedShadow::None` is a second, and a mirror that lists the
    /// exceptions it knew about drifts the moment a third arrives. The arm below
    /// caught it as a **ratio that went the wrong way** — 2.2× where the pass
    /// itself uploads less than it did — which is what a mirror is for.
    vsm_palette_bytes: usize,
    /// Geometry groups the scene's skinned content asks the page raster for,
    /// against a `VSM_MAX_GROUPS` of 1 024.
    vsm_groups: usize,
    /// Instances casting through the crowd's ONE shared proxy group.
    proxies: usize,
    /// **What the pre-NPC1b renderer would have uploaded on this same scene**:
    /// one power-of-two palette block per instance, on the lit path AND on the
    /// shadow path.
    ///
    /// Computed here rather than quoted from NPC1a's table, because NPC1a's
    /// 31.28 MB was `skinned x 16 KiB x 2` — a 161-bone block against a scene
    /// whose character is rigged at 20 joints. A before/after has to be computed
    /// the same way on the same content or it is two different questions.
    legacy_bytes: usize,
}

impl GpuCost {
    /// Read one projected scene, through the same planners the passes call.
    fn of(scene: &RenderScene) -> Self {
        let plan = inf_render::plan_skinned_batches(scene);
        let vsm_palette_bytes = scene
            .skinned
            .iter()
            .filter(|i| {
                !matches!(
                    i.shadow,
                    inf_render::SkinnedShadow::Proxy | inf_render::SkinnedShadow::None
                )
            })
            .map(|i| (i.palette.len().max(1) * 64).next_power_of_two().max(64))
            .sum();
        let legacy_bytes: usize = scene
            .skinned
            .iter()
            .map(|i| 2 * (i.palette.len().max(1) * 64).next_power_of_two().max(64))
            .sum();
        Self {
            legacy_bytes,
            draws: plan.runs.len(),
            blocks: plan.blocks,
            atlas_bytes: plan.matrices * 64,
            vsm_palette_bytes,
            vsm_groups: inf_render::skinned_caster_groups(scene),
            proxies: scene
                .skinned
                .iter()
                .filter(|i| i.shadow == inf_render::SkinnedShadow::Proxy)
                .count(),
        }
    }

    /// Palette megabytes a frame, both paths summed — the column wall 3 is about.
    fn palette_mb(&self) -> f64 {
        (self.atlas_bytes + self.vsm_palette_bytes) as f64 / (1024.0 * 1024.0)
    }
}

impl Row {
    /// The palette megabytes a frame uploads for this row's skinned characters.
    ///
    /// **Measured, since NPC1b** — through `plan_skinned_batches`, the function
    /// the pass itself calls. It used to be `skinned x 16 KiB x 2`: wall 3's
    /// arithmetic, stated rather than measured, and the number NPC1b had to beat.
    fn palette_mb(&self) -> f64 {
        self.gpu.palette_mb()
    }
}

/// Run one cell of the sweep.
fn measure(pack: &Path, n: usize, banded: bool) -> Row {
    let mut fx = open(pack);
    let archetype = if n > 0 {
        archetype_from_hero(&fx.sim)
    } else {
        CrowdArchetype::default()
    };
    if !banded {
        assert!(
            fx.sim.set_crowd_radii(ALL_FULL),
            "the all-Full control ladder was refused"
        );
    }
    if n > 0 {
        fx.sim.set_crowd_population(population(n, archetype));
    }

    for _ in 0..WARMUP {
        fx.sim.step_once(RuntimeInput::default());
    }
    fx.sim.set_step_profiling(true);

    let mut rounds: Vec<inf_player::step_profile::StepProfile> = Vec::new();
    for _ in 0..ROUNDS {
        let mut acc = inf_player::step_profile::StepProfile::default();
        for _ in 0..SAMPLES {
            fx.sim.step_once(RuntimeInput::default());
            acc.accumulate(&fx.sim.step_profile());
        }
        acc.scale(1.0 / SAMPLES as f64);
        rounds.push(acc);
    }
    let prof = *rounds
        .iter()
        .min_by(|a, b| a.total_ms().total_cmp(&b.total_ms()))
        .expect("at least one round");

    use inf_player::step_profile::STEP_PHASE_NAMES;
    // Panics on an unknown phase rather than answering `0.0`: a misspelled name
    // would print a column of zeroes and read as "this phase costs nothing",
    // which is the under-attribution `step_profile` exists to remove, one level
    // up in the instrument that reads it.
    let ms = |name: &str| {
        let i = STEP_PHASE_NAMES
            .iter()
            .position(|n| *n == name)
            .unwrap_or_else(|| panic!("no step phase called `{name}`"));
        prof.ms[i]
    };

    let trace_bytes = fx.sim.state_bytes().len();
    let stats = fx.sim.crowd_stats();
    // **THE TRACE, SECTION BY SECTION.** The first draft of this instrument
    // printed one total and the wave's arithmetic then attributed the growth to
    // the pose — which is inference dressed as measurement, and it was wrong by
    // a factor of six. Every section is read separately here, and whatever the
    // nine of them do not account for is the bincode snapshot, printed as a
    // residue rather than assumed to be one.
    let sections = TraceSections::of(&fx.sim, trace_bytes);
    // Bytes, not a count. `Row::posed` beside it is the COUNT, off the pose
    // store; the two are named apart because a reader comparing "291 posed" to
    // "243 276 B of pose" has to know which is which.
    let pose_bytes = sections.pose;

    // The projection, measured the way `projection_budget.rs` measures it: no
    // GPU at any point, into the same `RenderScene` a host reuses every frame.
    let mut scene = RenderScene::default();
    let mut voxels = inf_voxel::VoxelVolumes::default();
    let mut debris = inf_render::DebrisCache::default();
    let eye = samples::city_drive_point(0);
    fx.sim.sync_render_terrain(eye);
    inf_player::render::sync_voxel_store(&mut voxels, &fx.voxel_assets, &fx.sim, eye);
    let mut best_proj = f64::INFINITY;
    for _ in 0..8 {
        let t = std::time::Instant::now();
        project_scene_full(
            &mut scene,
            &fx.sim,
            1.0,
            &fx.vmeshes,
            &fx.skinned,
            &voxels,
            &mut debris,
            None,
            &fx.scatter_meshes,
        );
        best_proj = best_proj.min(t.elapsed().as_secs_f64() * 1000.0);
    }

    Row {
        n,
        banded,
        step_ms: prof.total_ms(),
        crowd_ms: ms("crowd"),
        society_ms: ms("society"),
        anim_ms: ms("animation"),
        phys_ms: ms("physics3d sync"),
        solver_ms: ms("solver"),
        trace_bytes,
        // A posed character is `36 + 40 x joints` bytes; the count is what the
        // pose store actually holds, derived from the store rather than from an
        // assumed rig size.
        posed: pose_count(fx.sim.world()),
        per_tier: stats.per_tier,
        instances: scene.instances.len(),
        scatter_batches: scene.scatter.len(),
        skinned: scene.skinned.len(),
        projection_ms: best_proj,
        sections,
        gpu: GpuCost::of(&scene),
    }
    .also_print(pose_bytes)
}

impl Row {
    fn also_print(self, pose_bytes: usize) -> Row {
        println!(
            "N={:<5} {:<9} step {:7.3} ms (crowd {:6.3} / anim {:6.3} / phys3d {:6.3} / \
             solver {:6.3})  trace {:>9} B (pose {:>9} B, {} posed)  tiers {}F/{}N/{}Fa/{}D  \
             draws {} inst + {} scatter, {} skinned in {} calls ({} blocks, {} \
             proxies, {} vsm groups), palette {:.2} MB  projection {:.3} ms",
            self.n,
            if self.banded { "banded" } else { "all-Full" },
            self.step_ms,
            self.crowd_ms,
            self.anim_ms,
            self.phys_ms,
            self.solver_ms,
            self.trace_bytes,
            pose_bytes,
            self.posed,
            self.per_tier[0],
            self.per_tier[1],
            self.per_tier[2],
            self.per_tier[3],
            self.instances,
            self.scatter_batches,
            self.skinned,
            self.gpu.draws,
            self.gpu.blocks,
            self.gpu.proxies,
            self.gpu.vsm_groups,
            self.palette_mb(),
            self.projection_ms,
        );
        println!(
            "      sections: snapshot {} B, deform {} B, pose {} B, cloth {} B, hair {} B, gameplay {} B, crowd {} B",
            self.sections.snapshot,
            self.sections.deform,
            self.sections.pose,
            self.sections.cloth,
            self.sections.hair,
            self.sections.gameplay,
            self.sections.crowd,
        );
        self
    }
}

/// How many entities the pose store holds — the count the trace's pose section
/// is `36 + 40 · joints` bytes *each* of.
fn pose_count(world: &inf_ecs::EcsWorld) -> usize {
    inf_ecs::pose::posed_count(world)
}

// ── the sweep ───────────────────────────────────────────────────────────────

/// **THE N-SWEEP.** The table NPC1b's brief is written from.
#[test]
fn the_n_sweep() {
    let tmp = tempfile::tempdir().expect("tmp");
    let pack = cook_scene(tmp.path());

    println!(
        "\n=== THE N-SWEEP (NPC1a) === {ROUNDS} rounds x {SAMPLES} steps after \
         {WARMUP} discarded, MIN of rounds; content: the phase-30 city (1 000 \
         grammar buildings, 370 468 solids), a streamed terrain and the phase-29 \
         character, with N copies of that character standing in a {:.0} m block \
         on the drive line. Ladder: full {} m / near {} m / far {} m.",
        2.0 * BLOCK_HALF_M,
        DEFAULT_CROWD_RADII.0,
        DEFAULT_CROWD_RADII.1,
        DEFAULT_CROWD_RADII.2,
    );

    let mut banded: Vec<Row> = Vec::new();
    let mut full: Vec<Row> = Vec::new();
    for n in NS {
        banded.push(measure(&pack, n, true));
        full.push(measure(&pack, n, false));
    }

    // ── (a) ZERO COST WHEN ABSENT ───────────────────────────────────────────
    //
    // Structural rather than statistical: `step_crowd` returns on a
    // `contains_resource` before it allocates anything, so the phase clock
    // charges it a time that rounds to nothing, and `crowd_state_bytes` returns
    // an empty vec so the trace is byte-for-byte the pre-NPC1a one.
    let zero = banded[0];
    assert_eq!(zero.n, 0);
    assert_eq!(
        zero.per_tier, [0; 4],
        "a scene with no population reported agents"
    );
    assert!(
        zero.crowd_ms < 0.005,
        "the crowd phase cost {:.4} ms on a level with no crowd — 'absent costs \
         nothing' is not structural any more",
        zero.crowd_ms
    );
    println!(
        "\n(a) ZERO COST WHEN ABSENT: the crowd phase reads {:.4} ms and folds 0 \
         trace bytes on a level with no population ({} B of state either way)",
        zero.crowd_ms, zero.trace_bytes
    );

    // ── (b) THE LADDER IS WHAT MAKES N AFFORDABLE ───────────────────────────
    //
    // The comparison that says the tier system does anything at all. Same scene,
    // same N, same routes, same step — one banded and one with every agent
    // forced to `Full`. If the two agreed, the ladder would be a decision
    // nothing reads.
    let (b, f) = (banded[NS.len() - 1], full[NS.len() - 1]);
    println!(
        "\n(b) THE LADDER, at N={}: banded poses {} of {} agents and folds {} B a \
         step; all-Full poses {} and folds {} B — {:.1}x the trace and {:.2}x the \
         animation phase",
        b.n,
        b.posed,
        b.n,
        b.trace_bytes,
        f.posed,
        f.trace_bytes,
        f.trace_bytes as f64 / b.trace_bytes.max(1) as f64,
        f.anim_ms / b.anim_ms.max(1.0e-9),
    );
    assert!(
        b.posed < f.posed,
        "the banded run posed {} characters and the all-Full control posed {} — \
         the ladder is deciding nothing",
        b.posed,
        f.posed
    );
    assert!(
        b.trace_bytes < f.trace_bytes,
        "the banded run folds {} trace bytes and the all-Full control folds {} — \
         the trace re-shape is not riding the tiers",
        b.trace_bytes,
        f.trace_bytes
    );
    assert!(
        b.per_tier[0] < b.n,
        "every one of {} agents came out Full inside a {} m block against a {} m \
         Full radius — the sweep is not posing the problem",
        b.n,
        2.0 * BLOCK_HALF_M,
        DEFAULT_CROWD_RADII.0
    );

    // ── (c) THE TRACE ARITHMETIC ────────────────────────────────────────────
    //
    // A non-posing agent costs `AGENT_TRACE_BYTES` and nothing else. Checked as
    // a DIFFERENCE against the N=0 baseline rather than as an absolute, because
    // the scene's own trace is most of both numbers.
    let base = banded[0].trace_bytes;
    let zero_s = banded[0].sections;
    println!("(c) THE TRACE, against a {base} B baseline, SECTION BY SECTION:");
    for r in &banded {
        if r.n == 0 {
            continue;
        }
        let grew = r.trace_bytes - base;
        let floor = r.n * AGENT_TRACE_BYTES;
        let s = r.sections;
        let per = |a: usize, b: usize| (a - b) as f64 / r.n as f64;
        println!(
            "  N={:<5} +{:>8} B = {:>6.0} B an agent: crowd {:>4.0} + snapshot {:>4.0} + pose {:>5.0} + deform {:>5.0}",
            r.n,
            grew,
            grew as f64 / r.n as f64,
            per(s.crowd, zero_s.crowd),
            per(s.snapshot, zero_s.snapshot),
            per(s.pose, zero_s.pose),
            per(s.deform, zero_s.deform),
        );
        assert!(
            grew >= floor,
            "N={} grew the trace by {} B, under the {} B the crowd section alone must contribute - a section is missing from the fold",
            r.n,
            grew,
            floor
        );
        assert_eq!(
            s.crowd,
            r.n * AGENT_TRACE_BYTES,
            "N={} folded {} crowd bytes against {} agents x {AGENT_TRACE_BYTES}",
            r.n,
            s.crowd,
            r.n
        );
    }
    // The headline ratio, at N=100 and at the tier the wave is about.
    let hundred = banded
        .iter()
        .find(|r| r.n == 100)
        .copied()
        .expect("N=100 is in the sweep");
    let hundred_full = full
        .iter()
        .find(|r| r.n == 100)
        .copied()
        .expect("N=100 is in the sweep");
    let joints = (hundred.sections.pose / hundred.posed.max(1)).saturating_sub(36) / 40;
    println!(
        "  AT N=100: banded {} B a step against all-Full {} B, {:.1}x. This scene rigs its character at {joints} joints, so one posed agent is {} B of pose against {AGENT_TRACE_BYTES} B of crowd section; the island hero is 161 joints and 6 476 B, where the same move is 132x.",
        hundred.trace_bytes,
        hundred_full.trace_bytes,
        hundred_full.trace_bytes as f64 / hundred.trace_bytes.max(1) as f64,
        36 + joints * 40,
    );
    // **THE SECTION THAT ACTUALLY DOMINATES**, named rather than left in the
    // table for a reader to find. The wave arrived believing the trace wall was
    // the pose; the measurement says it is the ground the crowd walks on.
    // NPC1b and NPC1c inherit this.
    let deform_pct = hundred.sections.deform as f64 / hundred.trace_bytes.max(1) as f64 * 100.0;
    let pose_pct = hundred.sections.pose as f64 / hundred.trace_bytes.max(1) as f64 * 100.0;
    println!(
        "  AND THE DEAREST SECTION IS THE GROUND: at N=100 the deformation field is {deform_pct:.0} % of the trace and the pose is {pose_pct:.0} %. A crowd trace wall is the field its feet leave, not the joints it carries.",
    );

    // ── the tables, for the brief ───────────────────────────────────────────
    for (label, rows) in [("BANDED", &banded), ("ALL-FULL", &full)] {
        println!("\n{label}");
        println!(
            "  {:>5} | {:>8} | {:>7} | {:>7} | {:>10} | {:>6} | {:>7} | {:>5} | {:>6} | {:>6} | {:>8} | {:>10}",
            "N",
            "step ms",
            "crowd",
            "anim",
            "trace B",
            "posed",
            "skinned",
            "draws",
            "blocks",
            "groups",
            "palette",
            "proj ms"
        );
        for r in rows.iter() {
            println!(
                "  {:>5} | {:>8.3} | {:>7.3} | {:>7.3} | {:>10} | {:>6} | {:>7} | {:>5} | {:>6} | {:>6} | {:>7.2}M | {:>10.3}",
                r.n,
                r.step_ms,
                r.crowd_ms,
                r.anim_ms,
                r.trace_bytes,
                r.posed,
                r.skinned,
                r.gpu.draws,
                r.gpu.blocks,
                r.gpu.vsm_groups,
                r.palette_mb(),
                r.projection_ms
            );
        }
    }

    // ── (d) THE RENDERER READS THE TIER (wave NPC1b) ────────────────────────
    //
    // Structural, not statistical, and the three claims are the three things the
    // crowd renderer is: ONE draw for a whole crowd on one mesh, ONE palette
    // block for every agent the ladder took off the pose path, and ONE geometry
    // group for every agent that casts through the proxy.
    //
    // The all-`Full` control is what makes it falsifiable. The same thousand
    // agents on the same mesh, banded and unbanded, must differ HERE — a
    // renderer that ignored the tier would report the same blocks and the same
    // groups in both columns, which is exactly what it did before this wave.
    {
        let g = b.gpu;
        let gf = f.gpu;
        println!(
            "\n(d) THE CROWD RENDERER at N={}: {} skinned instances draw in {} call(s)",
            b.n, b.skinned, g.draws
        );
        println!(
            "    from {} palette block(s) = {:.2} MB of atlas; {} of them cast through {} shared proxy group(s).",
            g.blocks,
            g.atlas_bytes as f64 / (1024.0 * 1024.0),
            g.proxies,
            usize::from(g.proxies > 0),
        );
        println!(
            "    The scene asks the page raster for {} groups of a {} ceiling and uploads {:.2} MB of caster palette.",
            g.vsm_groups,
            inf_render::VSM_MAX_GROUPS,
            g.vsm_palette_bytes as f64 / (1024.0 * 1024.0),
        );
        println!(
            "    ALL-FULL control: {} call(s), {} blocks, {} groups, {:.2} MB atlas.",
            gf.draws,
            gf.blocks,
            gf.vsm_groups,
            gf.atlas_bytes as f64 / (1024.0 * 1024.0),
        );
        println!(
            "    THE PALETTE, BEFORE AND AFTER ON THIS SCENE: the pre-NPC1b renderer uploaded one",
        );
        println!(
            "    power-of-two block per instance on each of the two paths = {:.2} MB a frame; the atlas",
            g.legacy_bytes as f64 / (1024.0 * 1024.0),
        );
        println!(
            "    plus the caster palettes are {:.2} MB, {:.1}x. (NPC1a's table said 31.28 MB, which is",
            g.palette_mb(),
            g.legacy_bytes as f64 / (g.atlas_bytes + g.vsm_palette_bytes).max(1) as f64,
        );
        println!(
            "    the same arithmetic at a 161-bone block against a rig this scene carries at {} joints;",
            g.atlas_bytes / 64 / g.blocks.max(1),
        );
        println!(
            "    on the island's own hero the before really is {:.1} MB.)",
            (b.skinned * 2 * 16 * 1024) as f64 / (1024.0 * 1024.0),
        );
        assert!(
            g.legacy_bytes > 4 * (g.atlas_bytes + g.vsm_palette_bytes),
            "the atlas uploads {} B where the per-instance path uploaded {} B, under 4x",
            g.atlas_bytes + g.vsm_palette_bytes,
            g.legacy_bytes
        );
        assert!(
            g.draws <= 2,
            "a crowd of {} on one mesh drew in {} calls, so the batching is not grouping by mesh",
            b.skinned,
            g.draws
        );
        assert!(
            g.blocks < b.skinned / 2,
            "{} skinned instances produced {} palette blocks: the shared pose is not reaching the atlas",
            b.skinned,
            g.blocks
        );
        assert!(
            g.blocks < gf.blocks,
            "banded produced {} palette blocks and all-Full {}: the renderer is not reading the tier",
            g.blocks,
            gf.blocks
        );
        assert!(
            g.vsm_groups < gf.vsm_groups && (g.vsm_groups as u32) < inf_render::VSM_MAX_GROUPS,
            "banded asks for {} shadow groups and all-Full for {}, against a {} ceiling",
            g.vsm_groups,
            gf.vsm_groups,
            inf_render::VSM_MAX_GROUPS
        );
    }

    // The per-agent cost of the crowd phase itself — the number
    // `NPC_STEP_BUDGET_MS` is minted from, and the one NPC1b's parallelism
    // question is decided against.
    if let Some(r) = banded.iter().find(|r| r.n == 1000) {
        println!(
            "\nTHE CROWD PHASE ITSELF: {:.4} ms for {} agents = {:.4} us an agent \
             a step. The whole phase is {:.2} % of a {:.3} ms step.",
            r.crowd_ms,
            r.n,
            r.crowd_ms * 1000.0 / r.n as f64,
            r.crowd_ms / r.step_ms.max(1.0e-9) * 100.0,
            r.step_ms
        );
    }
    // ── THE BUDGET (NPC1a) ──────────────────────────────────────────────────
    //
    // `inf_player::budget`'s conditioning, for its reasons: the crowd phase is a
    // wall clock, `[profile.dev]` is `opt-level = 1` with debug assertions on,
    // and a shared CI runner's milliseconds are a fact about the runner. So the
    // arm reports everywhere and asserts under `cargo test --release` off CI.
    let budgeted = banded
        .iter()
        .find(|r| r.n == inf_player::budget::NPC_BUDGET_AGENTS)
        .copied()
        .expect("the sweep must measure the population the budget is quoted at");
    println!(
        "THE CROWD BUDGET: {:.4} ms at N={} against NPC_STEP_BUDGET_MS = {:.1} ms",
        budgeted.crowd_ms,
        budgeted.n,
        inf_player::budget::NPC_STEP_BUDGET_MS
    );
    // ── THE SOCIETY BUDGET (NPC1d) ──────────────────────────────────────────
    //
    // The SETTLED cost, which is the one a shipped frame pays for ever: one
    // entity walk that folds nothing. The derivation transient is bounded by
    // `SOCIETY_PLANS_PER_STEP` and is measured in the island gate, not here.
    println!(
        "THE SOCIETY BUDGET: {:.4} ms settled at N={} against \
         SOCIETY_STEP_BUDGET_MS = {:.1} ms",
        budgeted.society_ms,
        budgeted.n,
        inf_player::budget::SOCIETY_STEP_BUDGET_MS
    );
    // ── THE PHASE'S WORK IS A FUNCTION OF THE CROWD (NPC1a audit) ───────────
    //
    // `step_crowd` does the same work per agent at every tier — a decision, a
    // transform write, a component write — so the banded run and the all-`Full`
    // control, which hold the SAME thousand agents, must charge this phase the
    // same. The wave's first cut did not: it read **0.282 ms banded and 0.759
    // all-`Full`**, because the phase folded a digest of every entry in the pose
    // store on every step and the control had 1 001 of them against 291. The
    // ratio is therefore the tell for "this phase is being paid for other
    // systems' work", and the budget minted from a contaminated reading was
    // three times what the crowd actually costs.
    //
    // Reported here, and asserted in the same release/off-CI block the budget is
    // asserted in — this file's own discipline, and the reason the *semantic*
    // half is armed as a counter in `inf_ecs::crowd`'s
    // `a_settled_crowd_folds_no_pose_digests_however_many_characters_pose`.
    let control = full[NS.len() - 1];
    let shape = control.crowd_ms / budgeted.crowd_ms.max(1.0e-9);
    println!(
        "THE PHASE'S SHAPE at N={}: banded {:.4} ms against all-Full {:.4} ms ({shape:.2}x) over the same population - the crowd phase must not be a function of how many characters POSE.",
        budgeted.n, budgeted.crowd_ms, control.crowd_ms
    );
    if cfg!(debug_assertions) {
        println!(
            "dev profile (opt-level 1, debug assertions ON): every millisecond above is a number about a build nobody ships, so the budget is reported and not asserted - re-run with --release for the numbers it was minted from."
        );
        return;
    }
    if std::env::var_os("CI").is_some() {
        println!("CI runner: the budget is reported, not asserted (the P26.5 rule).");
        return;
    }
    assert!(
        budgeted.society_ms <= inf_player::budget::SOCIETY_STEP_BUDGET_MS,
        "the society phase cost {:.4} ms on a SETTLED level against a {:.1} ms \
         budget {}",
        budgeted.society_ms,
        inf_player::budget::SOCIETY_STEP_BUDGET_MS,
        inf_player::budget::RATCHET_NOTE
    );
    assert!(
        budgeted.crowd_ms <= inf_player::budget::NPC_STEP_BUDGET_MS,
        "the crowd phase cost {:.4} ms at N={} against a {:.1} ms budget {}",
        budgeted.crowd_ms,
        budgeted.n,
        inf_player::budget::NPC_STEP_BUDGET_MS,
        inf_player::budget::RATCHET_NOTE
    );
    assert!(
        shape < 2.0,
        "the crowd phase charged {:.4} ms banded and {:.4} ms all-Full ({shape:.2}x) over the SAME {} agents - the phase is doing work that scales with how many characters pose rather than with the population, which is what made NPC1a's first budget three times the crowd's real cost",
        budgeted.crowd_ms,
        control.crowd_ms,
        budgeted.n
    );
}

// ── the actor door ──────────────────────────────────────────────────────────

/// **THE Far -> Near POSE POP, MEASURED** (wave NPC1b, clause 5).
///
/// NPC1a carried it as a sentence — *"a `Far -> Near` promotion pops; the machine
/// does not advance while an agent is `Far`, so it resumes where it left off"* —
/// and a sentence is not a size. This arm gives it one, in the only unit that
/// makes it decidable: **how far a joint moves at a promotion, against how far it
/// moves in an ordinary step.**
///
/// The two poses are the two the renderer actually draws. A `Far` agent
/// evaluates nothing, so `resolve_skinned_shared` hands it its rig's rest
/// matrices; the step it is promoted, the machine's pose arrives whole. So the
/// pop is `|rest - posed|` and the ordinary motion is `|posed(t) - posed(t+1)|`,
/// both read off the same registry the projector reads.
///
/// **It REPORTS.** The ratio is a property of the level's animation and of the
/// fixed step, not of this wave, and asserting a number here would pin an
/// author's clip. What it asserts is the two things that make the report mean
/// something: that the ordinary step moves *something* (so the machine is
/// running) and that the pop is strictly larger than it (so the pop is real and
/// the sentence NPC1a carried is not folklore).
#[test]
fn the_far_to_near_promotion_pop_is_measured() {
    let tmp = tempfile::tempdir().expect("tmp");
    let pack = cook_scene(tmp.path());
    let mut fx = open(&pack);

    // Let the machine get past its entry state and into real motion.
    for _ in 0..WARMUP {
        fx.sim.step_once(RuntimeInput::default());
    }

    // The level's own posed character — the archetype every test NPC copies, so
    // the numbers below are about the crowd's own rig.
    let hero = fx
        .sim
        .world()
        .world()
        .iter_entities()
        .filter_map(|e| {
            let g = e.get::<inf_ecs::Guid>()?;
            let sk = e.get::<inf_ecs::components::SkeletalMesh>()?;
            sk.skeleton?;
            Some((g.0, *sk))
        })
        .min_by_key(|(g, _)| *g)
        .expect("the sweep scene has no rigged character");
    let (guid, sm) = hero;

    let palette_now = |fx: &Fixture| -> Vec<glam::Mat4> {
        let posed = inf_ecs::pose::evaluated_pose(fx.sim.world(), guid);
        assert!(
            posed.is_some(),
            "the level's character published no pose, so there is no promotion to \
             measure"
        );
        fx.skinned
            .resolve_skinned(&sm, None, posed, None)
            .expect("the character resolves")
            .palette
            .to_vec()
    };
    let worst = |a: &[glam::Mat4], b: &[glam::Mat4]| -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x.w_axis.truncate() - y.w_axis.truncate()).length())
            .fold(0.0f32, f32::max)
    };

    // What a `Far` agent is drawn in: its rig's rest matrices, shared by the
    // whole bucket. Read through the same door the projector uses.
    let rest = fx
        .skinned
        .resolve_skinned_shared(&sm, None)
        .expect("the character resolves")
        .palette
        .to_vec();

    // An ordinary step's motion, worst joint, over a window rather than one
    // sample: a single fixed step can land between two keyframes and read zero.
    let mut step_motion = 0.0f32;
    let mut prev = palette_now(&fx);
    let mut pop = worst(&rest, &prev);
    for _ in 0..60 {
        fx.sim.step_once(RuntimeInput::default());
        let now = palette_now(&fx);
        step_motion = step_motion.max(worst(&prev, &now));
        pop = pop.max(worst(&rest, &now));
        prev = now;
    }

    println!("\nTHE PROMOTION POP (NPC1b clause 5): a Far agent is drawn in its rig's REST pose;");
    println!(
        "  the step it is promoted, the machine's pose arrives whole. Worst joint over 60 steps:"
    );
    println!("  pop {pop:.4} m against an ordinary step's {step_motion:.4} m");
    println!(
        "  = {:.0}x an ordinary step. The character in this fixture is nearly still, so read the",
        pop / step_motion.max(1.0e-9),
    );
    println!(
        "  ratio as a floor and the metre as the fact: the pop is the WHOLE distance from the bind"
    );
    println!(
        "  pose to the pose the machine is in, and it arrives in one step. It is not new — a Far"
    );
    println!("  agent drew its rest pose before NPC1b too, because it has no EvaluatedPose and no");
    println!("  AnimPlayer — but nothing had measured it.");
    println!("  NOT FIXED IN THIS WAVE. The fix is an inertialization on re-entry (the P29.2");
    println!(
        "  `PoseBlender` seam), which is SIM state on the crowd's own trace section and belongs"
    );
    println!(
        "  with the wave that owns it — NPC1c gives a near agent a controller and re-enters the"
    );
    println!(
        "  machine from a real pose rather than from rest. Carried by name in the NPC1b ledger."
    );

    assert!(
        step_motion > 0.0,
        "the character's machine moved no joint over 60 steps, so this arm is \
         measuring a still character and its ratio means nothing"
    );
    assert!(
        pop > step_motion,
        "the promotion pop ({pop}) is no larger than an ordinary step ({step_motion}), \
         so NPC1a's carried item 4 is wrong and the ledger should say so instead"
    );
}

/// **DOES A CROWD NPC NEED A BLUEPRINT?** — the NPC1a decision, with the
/// arithmetic it was made on.
///
/// # The question
///
/// `cell_stream.rs` carries a named v1 deferral: *"an entity that streams IN
/// does not gain a ticking blueprint… making the actor map dynamic is a
/// `RuntimeSim` change that this batch deliberately did not make."* NPC1a's
/// brief offered two doors — open the dynamic actor map so streamed NPCs can
/// carry Blueprint brains, or build a dedicated crowd system and leave the
/// deferral standing with its sentence updated.
///
/// # Why it is measured here and not argued
///
/// Because an unmeasured prescription can be backwards (P25's law), and because
/// the alternative reasoning available is exactly the kind this repository keeps
/// having to retract: *"`run_all_with_args` is a serial for-loop, therefore
/// Blueprints are too slow"* is a statement about a shape, not about a cost.
///
/// So both phases are driven at the same N in the same process on the same
/// machine: N entities carrying a `Tick` blueprint that does one variable write,
/// against N crowd agents. Both numbers come off `StepProfile`, which is the
/// shipped clock, so this measures the engine rather than a benchmark of it.
///
/// The arm asserts only the **ordering** — that a crowd agent is cheaper per
/// entity than a blueprint actor — because that is the claim the decision rests
/// on and it is a comparison of two numbers taken on one machine in one second,
/// which is the only kind of wall-clock claim that is portable.
#[test]
fn a_crowd_agent_is_cheaper_per_entity_than_a_blueprint_actor() {
    use inf_blueprint::{
        BlueprintClass, BlueprintFn, EventBinding, EventKind, Expr, Lit, Param, Stmt, Ty, Variable,
    };

    /// Entities on each side. Big enough that the per-entity cost dominates the
    /// phase's own fixed overhead, small enough that a dev-profile run of this
    /// arm stays in single-digit seconds.
    const N: usize = 1000;
    const STEPS: usize = 60;

    /// The smallest honest blueprint: one `Tick` that writes one variable.
    ///
    /// Not an empty class — an empty one would measure `run_on_guid`'s
    /// bookkeeping (the map remove, the `RuntimeHost` construction, the
    /// re-insert) and nothing else, and a crowd agent does real work per step.
    /// One `vars::set` is the least a brain can do and still be a brain.
    fn ticker() -> BlueprintClass {
        let mut class = BlueprintClass::new("act:npc1a_ticker", "Crowd Ticker");
        class.variables.push(Variable {
            name: "phase".into(),
            ty: Ty::Float,
            default: Lit::Float(0.0),
            exposed: false,
        });
        class.events.push(EventBinding {
            event: EventKind::Tick,
            body: BlueprintFn {
                id: "tick".into(),
                name: "tick".into(),
                params: vec![Param {
                    name: "dt".into(),
                    ty: Ty::Float,
                }],
                ret: Ty::Unit,
                body: vec![Stmt::ExprStmt(Expr::Call {
                    path: vec!["vars".into(), "set".into()],
                    args: vec![
                        Expr::Lit(Lit::Str("phase".into())),
                        Expr::Lit(Lit::Float(1.0)),
                    ],
                })],
            },
        });
        class
    }

    /// A bare world holding `n` entities at a spread of places, plus one
    /// streaming source at the origin so the crowd band is not unbounded.
    fn world_of(n: usize) -> (inf_ecs::EcsWorld, Vec<Uuid>) {
        let mut w = inf_ecs::EcsWorld::new();
        let src = w.spawn_with_guid(Uuid::from_u128(1), "Source", None);
        w.world_mut()
            .entity_mut(src)
            .insert(inf_ecs::components::StreamingSource { radius_m: 0.0 });
        let mut guids = Vec::with_capacity(n);
        for i in 0..n {
            guids.push(Uuid::from_u128(0x1000 + i as u128));
        }
        w.propagate();
        (w, guids)
    }

    // ── the blueprint side ──────────────────────────────────────────────────
    let (mut w, guids) = world_of(N);
    let mut actors = Vec::with_capacity(N);
    for (i, g) in guids.iter().enumerate() {
        let e = w.spawn_with_guid(*g, "Actor", None);
        inf_ecs::sim::set_translation(&mut w, e, inf_ecs::Vec3d::new(i as f64 * 0.5, 0.0, 0.0));
        actors.push((*g, ticker()));
    }
    w.propagate();
    let mut bp = RuntimeSim::new(w, actors, glam::DVec2::new(0.0, -9.81), 60.0);
    for _ in 0..10 {
        bp.step_once(RuntimeInput::default());
    }
    bp.set_step_profiling(true);
    let bp_ms = mean_phase(&mut bp, "blueprint tick", STEPS);

    // ── the crowd side ──────────────────────────────────────────────────────
    let (w, guids) = world_of(N);
    let mut cr = RuntimeSim::new(w, Vec::new(), glam::DVec2::new(0.0, -9.81), 60.0);
    let mut records = BTreeMap::new();
    for (i, g) in guids.iter().enumerate() {
        records.insert(
            *g,
            CrowdRecord::walking(
                // No assets: this arm prices the crowd SYSTEM against the
                // blueprint SYSTEM, and a rig on one side and not the other
                // would be measuring the pose pipeline.
                CrowdArchetype::default(),
                CrowdRoute::between(
                    DVec3::new(i as f64 * 0.5, 0.0, 0.0),
                    DVec3::new(i as f64 * 0.5, 0.0, 6.0),
                    1.4,
                ),
            ),
        );
    }
    cr.set_crowd_population(records);
    for _ in 0..10 {
        cr.step_once(RuntimeInput::default());
    }
    cr.set_step_profiling(true);
    let cr_ms = mean_phase(&mut cr, "crowd", STEPS);

    let stats = cr.crowd_stats();
    println!(
        "\n=== THE ACTOR DOOR, at N={N} === blueprint tick {bp_ms:.4} ms \
         ({:.3} us an actor) against the crowd phase {cr_ms:.4} ms \
         ({:.3} us an agent) — {:.1}x. Crowd tiers: {}",
        bp_ms * 1000.0 / N as f64,
        cr_ms * 1000.0 / N as f64,
        bp_ms / cr_ms.max(1.0e-9),
        stats.summary(),
    );
    println!(
        "  At 60 Hz a {:.4} ms blueprint tick for {N} actors is {:.1} % of the \
         6.0 ms CITY_STEP_BUDGET_MS, before the crowd does any actual thinking; \
         the crowd phase is {:.1} %.",
        bp_ms,
        bp_ms / 6.0 * 100.0,
        cr_ms / 6.0 * 100.0
    );
    assert!(
        cr_ms < bp_ms,
        "a crowd agent cost {cr_ms:.4} ms and a blueprint actor {bp_ms:.4} ms at \
         N={N} — the NPC1a decision (crowd brains are a dedicated system, \
         Blueprint is for hero-class actors) rests on this ordering and it does \
         not hold on this machine"
    );
    // Anti-vacuity: both sides must actually be doing their work, or two zeroes
    // compare fine.
    assert_eq!(stats.total(), N, "the crowd side lost agents");
    assert!(
        bp_ms > 0.0,
        "the blueprint tick measured exactly zero — the actors are not bound"
    );
}

/// The **mean** charge to the named phase over `steps` steps, in milliseconds.
///
/// A mean and not a minimum, deliberately, and the two sides of the comparison
/// take it the same way: a minimum over per-step charges of a phase that is a
/// hundred microseconds long is dominated by whichever step the scheduler left
/// alone, which is a different thing on each side of the comparison. The
/// ordering the arm asserts is a factor apart, not a percent.
fn mean_phase(sim: &mut RuntimeSim, phase: &str, steps: usize) -> f64 {
    let idx = inf_player::step_profile::STEP_PHASE_NAMES
        .iter()
        .position(|n| *n == phase)
        .unwrap_or_else(|| panic!("no step phase called `{phase}`"));
    let mut acc = 0.0;
    for _ in 0..steps {
        sim.step_once(RuntimeInput::default());
        acc += sim.step_profile().ms[idx];
    }
    acc / steps as f64
}

// ── the parallelism question ────────────────────────────────────────────────

/// **DOES `parallel_map` PAY FOR THE CROWD'S DECISION HALF?** — measured before
/// it is prescribed.
///
/// NPC1a's brief asks for agent decisions to go through `inf_core::job`'s
/// deterministic in-order map *"where parallelism pays (measure)"*. This is the
/// measurement, and it drives [`inf_ecs::crowd::plan_agent`] — the exact
/// function the shipped step calls, not a copy of it — over the same inputs
/// serially and through `parallel_map_ref`.
///
/// It **reports** rather than asserts a speed-up, for the reason every wall
/// clock in this tree reports: a ratio between two thread counts on a shared
/// runner is a fact about the runner. What it *does* assert is the property that
/// would make the parallel path legal at all: the two produce **identical**
/// output, which is `parallel_map_ref`'s own in-order-pure-map contract, checked
/// on the crowd's own data rather than trusted.
#[test]
fn the_parallel_map_over_agent_decisions_is_priced_before_it_is_prescribed() {
    let band = CrowdBand::from_anchors([DVec3::ZERO], DEFAULT_CROWD_RADII);
    println!("\n=== THE CROWD'S DECISION HALF, SERIAL vs PARALLEL ===");
    for n in [1usize, 10, 100, 1000, 10_000] {
        let inputs: Vec<(Uuid, CrowdRecord, DVec3)> = (0..n)
            .map(|i| {
                let at = DVec3::new((i % 97) as f64 * 7.0, 0.0, (i / 97) as f64 * 7.0);
                (
                    Uuid::from_u128(0x2000 + i as u128),
                    CrowdRecord::walking(
                        CrowdArchetype::default(),
                        CrowdRoute::between(at, at + DVec3::new(0.0, 0.0, 6.0), 1.4),
                    ),
                    at,
                )
            })
            .collect();
        let t_s = 3.25;

        let plan_one = |x: &(Uuid, CrowdRecord, DVec3)| {
            // The leg the step resolves once and hands down (island wave NPC1e;
            // `inf_ecs::crowd::ActiveLeg`). These records are unscheduled, so it
            // is `None` — resolved through `leg_at` rather than written as
            // `None` so this benchmark keeps calling what the step calls.
            let clock = inf_ecs::crowd::CrowdClock::at(t_s);
            let leg = x.1.leg_at(x.0, clock);
            inf_ecs::crowd::plan_agent(&band, x.0, &x.1, x.2, clock, leg)
        };
        // MIN of three, the instrument's discipline.
        let serial_out: Vec<_> = inputs.iter().map(plan_one).collect();
        let par_out = inf_core::parallel_map_ref(&inputs, plan_one);
        let mut serial_ms = f64::INFINITY;
        let mut par_ms = f64::INFINITY;
        for _ in 0..3 {
            let t = std::time::Instant::now();
            std::hint::black_box(inputs.iter().map(plan_one).collect::<Vec<_>>());
            serial_ms = serial_ms.min(t.elapsed().as_secs_f64() * 1000.0);
            let t = std::time::Instant::now();
            std::hint::black_box(inf_core::parallel_map_ref(&inputs, plan_one));
            par_ms = par_ms.min(t.elapsed().as_secs_f64() * 1000.0);
        }
        assert_eq!(
            serial_out, par_out,
            "the parallel map produced a different plan at N={n} — \
             `parallel_map_ref`'s in-order pure-map contract does not hold over \
             `plan_agent`, and a crowd that used it would not be deterministic"
        );
        println!(
            "  N={n:<6} serial {serial_ms:8.4} ms ({:7.4} us an agent)   parallel \
             {par_ms:8.4} ms ({:7.4} us an agent)   {:.2}x",
            serial_ms * 1000.0 / n as f64,
            par_ms * 1000.0 / n as f64,
            serial_ms / par_ms.max(1.0e-9)
        );
    }
    println!(
        "  The decision half is a nearest-anchor distance and a ping-pong along a \
         segment: a handful of flops an agent, with no allocation. Read the ratio \
         above before wiring a pool into a fixed step -- the shipped step calls \
         `plan_agent` inline, and NPC1a leaves it there unless this table says \
         otherwise on a real machine."
    );
}
