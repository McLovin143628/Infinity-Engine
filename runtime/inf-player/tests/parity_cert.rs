//! **THE PARITY CERTIFICATION'S OWN INSTRUMENT** (wave CERT1).
//!
//! Every table in `docs/memos/parity-certification.md` that is a *number about
//! the island* is printed by this file. The memo quotes it; it does not restate
//! it, and it does not compute anything by hand — the repository's own law is
//! that prose is never ahead of its arms, and a certification is the document
//! most able to break it.
//!
//! # What it runs on, and why that is the fixture and not the island
//!
//! The shipped island is 51.38 km² and its `.inf_terrain` is 549.9 MB, which is
//! deliberately not committed. So every arm here runs on `samples/island-fixture`
//! — the **same recipe format, the same generator, the same partition, the same
//! biome binding**, at 2.36 km² with its DEM tiles committed — which is exactly
//! `island_gate.rs`'s own doctrine and for its reason: a change that breaks the
//! island breaks this.
//!
//! Where a number is only interesting at the shipped scale, the arm is
//! `#[ignore]`d and reads `INF_CERT_ISLAND_PACK`, following the
//! `block_codec_bakeoff::island_codec_bakeoff` precedent — an un-ignored arm
//! that asserts the property on a corpus everyone has, plus an ignored one that
//! measures the corpus that matters.
//!
//! # What it does NOT do
//!
//! No GPU. The frame numbers the certification quotes come from
//! `fps_instrument.rs`, which owns that subject and has owned it since island
//! wave I4; duplicating a second frame harness here would be two instruments
//! measuring one thing, which this tree has already paid for once (`island_gate`
//! and `fps_instrument` disagreeing about the fixed step).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use glam::DVec3;
use inf_player::runtime_sim::{RuntimeInput, RuntimeSim};
use inf_project::ProjectManifest;

// ── the fixture ─────────────────────────────────────────────────────────────

/// The recipe every un-ignored arm builds — `island_gate`'s own.
fn fixture_recipe() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../samples/island-fixture/island.toml")
}

/// Build + cook the fixture island into `tmp`, through `inf island build`'s own
/// door (`write_content`), so a gate that passed while the command produced
/// something else is impossible.
fn cook_fixture(tmp: &Path) -> PathBuf {
    let recipe =
        inf_island::IslandRecipe::load(&fixture_recipe()).expect("the fixture recipe loads");
    let build = inf_island::build_island(&recipe, &inf_island::BuildOptions::default())
        .expect("the fixture island builds");
    assert!(
        build.report.land_km2 > 0.5,
        "the fixture has only {:.3} km2 of land — two empty worlds agree perfectly",
        build.report.land_km2
    );
    let proj = tmp.join("island");
    ProjectManifest::new(&recipe.name, "blank-3d")
        .save(&proj)
        .expect("the project scaffolds");
    inf_island::write_content(&build, &proj.join("Content")).expect("the content writes");
    let out = tmp.join("out");
    inf_packager::cook(&proj, &out, &inf_packager::CookOptions::default()).expect("it cooks");
    out
}

/// The SHIPPING host over a cooked pack, with both streamers attached —
/// `run_headless`'s own order, and a MIRROR of `island_gate::pack_sim`.
///
/// Attaching both is not optional: a partitioned level's cooked `.inf_lvl`
/// carries no entities at all (they are in the derived `.inf_part`), and a
/// streamed terrain's working set ships blank.
fn pack_sim(pack: &Path) -> RuntimeSim {
    let source = inf_player::level::PackLevelSource::open(pack).expect("the pack opens");
    let mut built = inf_player::build_world_from_pack(&source).expect("the world builds");
    let partition = built.take_partition();
    let pcg = built.pcg_context();
    let mut sim = inf_player::sim_from_built(built);
    inf_player::attach_cell_streaming(&mut sim, &partition, pcg);
    inf_player::attach_terrain_streaming(
        &mut sim,
        &inf_player::TerrainContent::Pack(source.clone()),
    );
    sim
}

/// The level's player-controlled hero and where it stands — `island_gate`'s and
/// `fps_instrument`'s own pair, so three instruments read one hero.
fn hero_at(sim: &RuntimeSim) -> Option<DVec3> {
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

/// Move the hero — the streaming anchor every band is derived from — to `p`.
/// A MIRROR of `island_gate`'s and `fps_instrument`'s pair.
fn set_hero(sim: &mut RuntimeSim, p: DVec3) {
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

/// Freeze this host's day at local solar `hour` — `island_gate::freeze_clock`.
fn freeze_clock(sim: &mut RuntimeSim, hour: f64) {
    let w = sim.world_mut().world_mut();
    let mut q = w.query::<&mut inf_ecs::components::TimeOfDay>();
    let mut wound = 0usize;
    for mut tod in q.iter_mut(w) {
        tod.seconds = (hour * 3600.0 - tod.longitude_deg * 240.0).rem_euclid(86_400.0);
        tod.rate = 0.0;
        wound += 1;
    }
    assert!(wound > 0, "the island carries no clock to freeze");
}

/// Step until the traffic plan queue has drained, or give up — the shape
/// `island_gate`'s rush-hour arm uses, because an unsettled world measures the
/// planner rather than the street.
fn settle(sim: &mut RuntimeSim, max_steps: usize) -> usize {
    // QUIET IS A WINDOW, NOT AN INSTANT. `island_gate::settle_the_society` is
    // the precedent and its reason is exact: a planner drains and refills, so
    // one step with an empty queue is a coincidence rather than a settled world.
    // The first draft of this file broke on a bare `i > 240` guard and reported
    // three identical hours, which is what an unsettled world looks like.
    const QUIET: usize = 120;
    let mut quiet = 0usize;
    let mut i = 0usize;
    while i < max_steps {
        sim.step_once(RuntimeInput::default());
        i += 1;
        let t = sim.traffic_stats();
        let so = sim.society_stats();
        let idle = t.pending == 0 && t.planned_now == 0 && so.pending == 0 && so.planned_now == 0;
        quiet = match idle {
            true => quiet + 1,
            false => 0,
        };
        if quiet >= QUIET {
            break;
        }
    }
    i
}

// ── CP-C2 · the pawn, the ground, and the cold start ────────────────────────

/// **CP-C2 · THE ISLAND'S PAWN COMES TO REST ON THE ISLAND.**
///
/// GTA1's closing audit found every starter level giving its pawn a `MeshRef`
/// plane with nothing physical on it — measured **4.9868 m of fall in one
/// second, still accelerating**, against −0.0201 m with a collider under it.
/// The island's ground is not a plane at all: it is a streamed heightfield, so
/// the question has to be asked again over the real boot path.
///
/// # What "on the ground" is, and why the first version of this arm was wrong
///
/// The first draft asserted the hero barely moves in its first second and
/// measured **0.9883 m of fall** — then found the same 0.9883 m in a world with
/// the terrain streamer left off, which meant the arm was not measuring the
/// ground at all. Both numbers are honest and neither is a defect: the hero is
/// SPAWNED about a metre above its ground and settles onto it, and the ground it
/// settles onto is there either way (the cell streamer brings the settlement's
/// own colliders, and the heightfield query answers before the pages do).
///
/// So the property is **rest**, not stillness: a body on the ground stops, and a
/// body on nothing does not. The control is the same hero lifted 50 m, which
/// must still be moving when the subject has stopped — without it, "it stopped"
/// is a claim an arm with gravity switched off would also pass.
///
/// # The settle itself (wave FIX1)
///
/// The 0.9883 m was **`START_LIFT_M`, and nothing else**. This arm's third
/// assertion used to bound the drop at 2 m, which is a bound the metre passed —
/// so the arm recorded the defect and could not fail on it. FIX1 set the lift to
/// zero, and the bound is now `SETTLE_CEILING_M`, tight enough that putting the
/// metre back reds this arm. What remains inside it is the honest remainder the
/// lift was hiding: the start's HEIGHT comes from the nearest committed road
/// vertex, and the ground under the start comes from the built heightfield, so
/// the two disagree by whatever the road profile rounded — `inf-island`'s own
/// `the_reported_start_is_the_one_the_level_spawns_at` bounds that gap at 5 m and
/// prints it as the START GAP.
#[test]
fn the_islands_pawn_comes_to_rest_on_the_island() {
    const SETTLE: usize = 300;
    const LIFT_M: f64 = 50.0;
    /// How far the hero may settle before the spawn is a DROP (wave FIX1).
    ///
    /// A quarter of a metre: four times the ground snap's own working room and a
    /// quarter of the metre this arm used to permit, so `START_LIFT_M` cannot go
    /// back to 1.0 without this file saying so.
    const SETTLE_CEILING_M: f64 = 0.25;

    let tmp = tempfile::tempdir().expect("a temp dir");
    let pack = cook_fixture(tmp.path());

    let mut sim = pack_sim(&pack);
    let spawn = hero_at(&sim).expect("the island has a player-controlled hero");
    let mut trace = Vec::with_capacity(SETTLE);
    for _ in 0..SETTLE {
        sim.step_once(RuntimeInput::default());
        trace.push(hero_at(&sim).expect("the hero survived").y);
    }
    let settled = *trace.last().expect("a trace");
    let drop = spawn.y - settled;
    // The last second of the run: a body at rest moves no further.
    let last_second = trace[SETTLE - 61] - trace[SETTLE - 1];

    // The CONTROL: the same world, the same hero, fifty metres up.
    let mut control = pack_sim(&pack);
    for _ in 0..30 {
        control.step_once(RuntimeInput::default());
    }
    let c_from = hero_at(&control).expect("the control's hero").y;
    let c_at = hero_at(&control).expect("the control's hero");
    set_hero(&mut control, DVec3::new(c_at.x, c_from + LIFT_M, c_at.z));
    let mut c_trace = Vec::new();
    for _ in 0..60 {
        control.step_once(RuntimeInput::default());
        c_trace.push(hero_at(&control).expect("it survived").y);
    }
    let c_last_second = c_trace[0] - c_trace[59];

    let spawn_y = spawn.y;
    println!(
        "CP-C2 the pawn on the island: spawned at y {spawn_y:.3}, settles at y {settled:.3} (a {drop:+.4} m drop over {SETTLE} steps); it then moves {last_second:+.5} m in the last second"
    );
    println!(
        "  control, the same hero lifted {LIFT_M:.0} m: {c_last_second:+.4} m in ITS first second — a falling body, so the arm above can see one"
    );

    assert!(
        last_second.abs() < 0.01,
        "the island's hero is still moving {last_second:+.5} m/s after {SETTLE} steps — it has not come to rest on anything"
    );
    assert!(
        c_last_second > 1.0,
        "the control fell only {c_last_second:.4} m from {LIFT_M:.0} m up, so the rest assertion above would have passed with no ground in the world"
    );
    // …and the settle is a SETTLE, not a plunge: the hero is spawned near its
    // ground rather than dropped onto it from a height a player would notice.
    assert!(
        drop.abs() < SETTLE_CEILING_M,
        "the hero fell {drop:.3} m before it stopped — it is spawned that far off its own \
         ground, against a {SETTLE_CEILING_M:.2} m ceiling"
    );
}

/// **CP-C2 · A COLD START, MEASURED.**
///
/// How long from "open the pack" to "the first fixed step has returned" — the
/// headless twin of the Play button's time-to-first-frame, minus the window and
/// the GPU, which is the half a test on this machine can hold to a number. The
/// other half is `fps_instrument.rs`'s, and it is a frame rather than a boot.
#[test]
fn the_island_boots_and_takes_its_first_step_inside_the_load_budget() {
    let tmp = tempfile::tempdir().expect("a temp dir");
    let pack = cook_fixture(tmp.path());

    let t0 = std::time::Instant::now();
    let mut sim = pack_sim(&pack);
    let open_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let t1 = std::time::Instant::now();
    sim.step_once(RuntimeInput::default());
    let first_step_ms = t1.elapsed().as_secs_f64() * 1000.0;
    let total = open_ms + first_step_ms;

    println!(
        "CP-C2 cold start (fixture): open+stream {open_ms:.1} ms + first fixed step {first_step_ms:.1} ms = {total:.1} ms, against LOAD_BUDGET_MS {:.1}",
        inf_player::budget::LOAD_BUDGET_MS
    );
    // Reported and not asserted on a shared runner or a dev build — the house
    // rule for every wall clock in this tree (`budget.rs`'s own header).
    if std::env::var_os("CI").is_some() || cfg!(debug_assertions) {
        println!("  (reported only: CI or a dev build)");
        return;
    }
    assert!(
        total < inf_player::budget::LOAD_BUDGET_MS,
        "a cold start took {total:.1} ms against a {:.1} ms budget",
        inf_player::budget::LOAD_BUDGET_MS
    );
}

// ── CP-B2 · the streets are never empty ─────────────────────────────────────

/// Total settlement street length in the resident set, metres.
fn street_metres(sim: &RuntimeSim) -> f64 {
    inf_ecs::traffic::carriageway_of(sim.world())
        .map(|res| res.streets.iter().map(|s| (s.b - s.a).length()).sum())
        .unwrap_or(0.0)
}

/// Bodies on the street right now: `(parked, driving, pedestrians)`.
fn occupancy(sim: &RuntimeSim) -> (usize, usize, usize) {
    let t = sim.traffic_stats();
    let c = sim.crowd_stats();
    let peds: usize = c.per_tier[0] + c.per_tier[1] + c.per_tier[2];
    (t.cars.saturating_sub(t.driving), t.driving, peds)
}

/// **CP-B2 · OCCUPANCY PER 100 m OF STREET, AT THREE HOURS.**
///
/// The reference's own claim is not "there are cars" — it is that a street is
/// never empty at any hour, and the frames that make it are a full hotel lot at
/// midday and a parked row under streetlights at night. So the measurement is a
/// density, at three clock positions, over the street length the world actually
/// has.
#[test]
fn the_streets_are_never_empty_at_any_hour() {
    let tmp = tempfile::tempdir().expect("a temp dir");
    let pack = cook_fixture(tmp.path());

    println!("CP-B2 street occupancy (island fixture, resident set):");
    println!("  hour  street_m  parked  driving  peds   per 100 m");
    let mut rows: Vec<(f64, f64, usize, usize, usize)> = Vec::new();
    for hour in [8.5_f64, 14.0, 21.0] {
        let mut sim = pack_sim(&pack);
        freeze_clock(&mut sim, hour);
        let steps = settle(&mut sim, 3_000);
        let metres = street_metres(&sim);
        let (parked, driving, peds) = occupancy(&sim);
        let per100 = if metres > 0.0 {
            (parked + driving + peds) as f64 * 100.0 / metres
        } else {
            0.0
        };
        println!(
            "  {hour:>4.1}  {metres:8.0}  {parked:6}  {driving:7}  {peds:4}   {per100:9.2}  \
             (settled in {steps} steps)"
        );
        rows.push((hour, per100, parked, driving, peds));
    }

    // The claim, asserted: at every hour there is something on the street, and
    // the street itself is a real length rather than a world with no roads.
    let mut sim = pack_sim(&pack);
    settle(&mut sim, 600);
    assert!(
        street_metres(&sim) > 100.0,
        "the fixture recovered no streets, so every density above is a division \
         by nothing"
    );
    for (hour, per100, parked, driving, peds) in &rows {
        assert!(
            *per100 > 0.0,
            "at {hour:.1} the street is empty: {parked} parked, {driving} driving, {peds} peds"
        );
    }
}

// ── CP-B10 · the light census, and the many-lights wall ─────────────────────

/// **CP-B10 · WHAT IS ACTUALLY LIT, AND WHAT A LIT SETTLEMENT WOULD NEED.**
///
/// The user's ruling is that a glowing window pane is not illumination: *"We
/// should have actual lights in this game and game engine."* So the census
/// counts REAL lights — `Light` components and `PcgVolume::lights` fixtures —
/// against the frame's own ceiling, and prints the arithmetic the PAR arc has
/// to fit under.
///
/// It measures and does not fix: a clustered light-culling pass is PAR0's, and
/// a certification that shipped half of one would have prejudged its design.
#[test]
fn the_light_census_and_the_many_lights_number() {
    let tmp = tempfile::tempdir().expect("a temp dir");
    let pack = cook_fixture(tmp.path());
    let mut sim = pack_sim(&pack);
    settle(&mut sim, 2_000);

    let w = sim.world().world();
    let mut authored = 0usize;
    let mut volumes = 0usize;
    let mut fixtures = 0usize;
    let mut lit_volumes = 0usize;
    let mut instances = 0usize;
    let mut glowing = 0usize;
    for e in w.iter_entities() {
        if e.get::<inf_ecs::components::Light>().is_some() {
            authored += 1;
        }
        if let Some(v) = e.get::<inf_ecs::components::PcgVolume>() {
            volumes += 1;
            fixtures += v.lights.len();
            if !v.lights.is_empty() {
                lit_volumes += 1;
            }
            instances += v.evaluated.len();
            glowing += v.evaluated.iter().filter(|i| i.glow > 0.0).count();
        }
    }

    let max = inf_render::passes::mesh::MAX_LIGHTS;
    let cap = inf_pcg::volume::VOLUME_LIGHT_CAP;
    // The sun/moon key light is pushed as `lights[0]` by the projector before any
    // entity light, so the frame's real budget for everything else is one less.
    let budget = max - 1;
    println!("CP-B10 light census (island fixture, resident set):");
    let row = |k: &str, v: String| println!("  {k:<36}{v}");
    row("authored `Light` entities", authored.to_string());
    row("PcgVolume blocks resident", volumes.to_string());
    row("…of which carry ANY fixture", lit_volumes.to_string());
    row("real PcgLight fixtures", fixtures.to_string());
    row("scattered instances", instances.to_string());
    row("…of which merely GLOW (panes)", glowing.to_string());
    row(
        "MAX_LIGHTS per frame",
        format!("{max}  (VOLUME_LIGHT_CAP {cap} per block)"),
    );
    row(
        "fixtures per lit block",
        format!(
            "{:.2}",
            match lit_volumes > 0 {
                true => fixtures as f64 / lit_volumes as f64,
                false => 0.0,
            }
        ),
    );
    row(
        "blocks at the cap one frame carries",
        format!(
            "{} beside the sun (the {}th overflows)",
            budget / cap.max(1),
            budget / cap.max(1) + 1
        ),
    );

    // THE FINDING, asserted so it cannot quietly stop being true: the world is
    // lit by emissive geometry and not by lights. A block that glows and hangs
    // no fixture is the thing the user rejected, and there are a lot of them.
    assert!(
        volumes > 0,
        "no PcgVolume is resident, so this census counted an empty world"
    );
    assert!(
        glowing > fixtures,
        "the fixture island has {glowing} glowing instances against {fixtures} real \
         fixtures — if this ever inverts, CP-B10 has been closed and this arm \
         should be rewritten rather than relaxed"
    );
    // …and the wall itself. FOUR blocks, not three: three at the cap is twelve
    // fixtures beside the sun, which fits in sixteen. The first draft of this
    // arm said three and the arm said so by failing — the wall is real and it
    // is one block further out than the prose had it.
    assert!(
        cap * 4 + 1 > max,
        "four blocks at VOLUME_LIGHT_CAP ({cap} each) no longer overflow \
         MAX_LIGHTS ({max}) — the many-lights wall this row prices has moved"
    );
}

// ── CP-C3 · the LOD ladder census ───────────────────────────────────────────

/// **CP-C3 · HOW MANY DISTANCE TIERS EACH ASSET KIND HAS.**
///
/// The user's requirement is three: low-poly far, medium mid, high-poly close,
/// for meshes, materials AND textures. This prints the ladder per kind, from the
/// engine's own constants and from the island's own cooked assets — so the memo
/// quotes a census rather than a claim.
#[test]
fn the_lod_ladder_census() {
    let tmp = tempfile::tempdir().expect("a temp dir");
    let pack = cook_fixture(tmp.path());
    let reader = std::sync::Arc::new(
        inf_asset::PackReader::open(&pack.join(inf_player::level::PACK_FILE))
            .expect("the pack maps"),
    );

    println!("CP-C3 LOD ladder census (island fixture):");

    // ── virtualized geometry: the only CONTINUOUS ladder in the engine ──
    let vmeshes =
        inf_player::vmesh::VmeshRegistry::from_pack(reader.clone()).expect("the DAGs index");
    let row = |k: &str, v: String| println!("  {k:<36}{v}");
    row("meshlet DAGs in the pack", vmeshes.len().to_string());

    // ── every `.inf_mesh` the pack carries, and whether it got a DAG ──
    let mut meshes: Vec<(String, usize, bool)> = Vec::new();
    let mesh_ids: Vec<inf_asset::AssetId> = reader
        .index()
        .filter(|e| e.kind == inf_asset::AssetKind::Mesh)
        .map(|e| e.guid)
        .collect();
    for id in mesh_ids {
        let Ok(bytes) = reader.read_ref(id) else {
            continue;
        };
        let Ok(mesh) = inf_asset::decode::<inf_mesh::MeshAsset>(bytes.as_ref()) else {
            continue;
        };
        let has_dag = vmeshes.resolve(id.0).is_some();
        meshes.push((id.0.to_string(), mesh.triangle_count(), has_dag));
    }
    meshes.sort_by_key(|m| std::cmp::Reverse(m.1));
    row("static/skeletal meshes", meshes.len().to_string());
    for (id, tris, dag) in &meshes {
        let (levels, coarsest) = match dag {
            true => {
                let (_, src) = vmeshes
                    .resolve(uuid::Uuid::parse_str(id).expect("a guid"))
                    .expect("it resolved a moment ago");
                match src.to_mesh() {
                    Ok(m) => {
                        let lods = m.classic_lods();
                        let coarsest = lods.last().map(|l| l.triangle_count()).unwrap_or(0);
                        (m.level_count(), coarsest)
                    }
                    Err(_) => (1, *tris),
                }
            }
            false => (1, *tris),
        };
        println!(
            "    {id}  {tris:>7} tris  →  {levels} meshlet level(s), coarsest {coarsest} tris   [{}]",
            match dag {
                true => "vgeom — a CONTINUOUS ladder",
                false => "no DAG — its ladder is whatever its draw path has",
            }
        );
    }

    // ── scatter / vegetation: mesh → impostor → culled, three real bands ──
    let s = inf_render::RenderSettings::default();
    row(
        "scatter / vegetation (High tier)",
        format!(
            "3 bands: mesh 0..{:.0} m, impostor {:.0}..{:.0} m (fade {:.0} m), culled beyond {:.0} m",
            s.scatter.mesh_distance_m,
            s.scatter.mesh_distance_m - s.scatter.fade_band_m,
            s.scatter.cull_distance_m,
            s.scatter.fade_band_m,
            s.scatter.cull_distance_m,
        ),
    );

    // ── grammar buildings: parts ↔ shell, the two-band cut ──
    row(
        "grammar buildings",
        format!(
            "3 bands: fit-out 0..{:.0} m, fabric 0..{:.0} m + reach, shell {:.0} m..draw",
            inf_render::INTERIOR_LOD_M,
            inf_render::STRUCTURE_LOD_M,
            inf_render::STRUCTURE_LOD_M
        ),
    );

    // ── terrain: four clipmap rings with a morph between them ──
    println!(
        "  terrain clipmap {} rings, base {} cells, morph over the \
         last {:.0}% of a band",
        inf_render::passes::terrain::TERRAIN_LOD_COUNT,
        inf_render::passes::terrain::TERRAIN_BASE_CELLS,
        inf_render::passes::terrain::TERRAIN_MORPH_REGION * 100.0
    );

    // ── WHAT A MID TIER FOR A GRAMMAR BUILDING WOULD BE MADE OF ──
    //
    // The two bands above are parts and shell. A mid tier has to be built out of
    // something, and the only classification already in the instance stream is
    // the module FAMILY: the assembler stamps `module_mesh_guid(shape)` onto
    // every instance it places and the projector already buckets on exactly that
    // GUID. So the candidate mid tier is "stop drawing a building's FIT-OUT
    // before you stop drawing its fabric", and its price is the share of a
    // furnished building's instances that fit-out actually is.
    //
    // MEASURED, over every archetype, on a furnished build — not over the
    // fixture's resident set, which would have measured whichever zones happened
    // to be near the hero and whether `settlement::furnishes` turned furniture on
    // for them at all. "Measure the prescription before landing it" is this
    // repository's own law and this is the number it produces.
    let mut fabric_total = 0usize;
    let mut fit_out_total = 0usize;
    println!("  the fit-out share, per archetype, FURNISHED:");
    for arch in inf_pcg::building::ArchetypeId::ALL {
        let out = inf_pcg::building::assemble::build(
            &inf_pcg::building::BuildingParams {
                archetype: arch,
                footprint: inf_pcg::building::Rect2::from_center(
                    glam::DVec2::ZERO,
                    glam::DVec2::new(26.0, 18.0),
                ),
                base_y: 0.0,
                seed: 3,
                floors: 3,
            },
            3,
            true,
        );
        let (mut fabric, mut fit_out) = (0usize, 0usize);
        for i in &out.instances {
            match i.mesh {
                Some(g) if inf_pcg::building::modules::is_fit_out_mesh(g) => fit_out += 1,
                _ => fabric += 1,
            }
        }
        fabric_total += fabric;
        fit_out_total += fit_out;
        println!(
            "    {:<16} {:>6} instances   {fit_out:>5} fit-out ({:>4.1} %)",
            format!("{arch:?}"),
            fabric + fit_out,
            match fabric + fit_out > 0 {
                true => fit_out as f64 * 100.0 / (fabric + fit_out) as f64,
                false => 0.0,
            }
        );
    }
    let all = fabric_total + fit_out_total;
    let share = match all > 0 {
        true => fit_out_total as f64 * 100.0 / all as f64,
        false => 0.0,
    };
    row(
        "a fit-out mid tier would drop",
        format!("{share:.1} % of a furnished building's instances ({fit_out_total} of {all})"),
    );
    assert!(all > 0, "no archetype assembled anything");
    // **THE NUMBER THIS CENSUS EXISTS FOR, ASSERTED** (CERT1 audit). It was
    // printed and nothing held it: mutating `ModuleShape::is_fit_out` to answer
    // `false` for every family makes this row print `0.0 % (0 of 10428)` and the
    // arm stayed GREEN, which is a census that would watch the classification it
    // measures stop existing and say so only to a reader. 12.2 % is what the mid
    // band was landed on, and a floor of 5 % is well under it and well over the
    // 1.0 % artefact the first (mis-aimed) measurement produced — so this fires
    // for a classifier that has stopped classifying, and not for content drift.
    assert!(
        share > 5.0,
        "the fit-out share is {share:.1} % ({fit_out_total} of {all}) — the mid \
         LOD band was landed on 12.2 %, and below 5 % either `is_fit_out` has \
         stopped classifying or `settlement::furnishes` has stopped furnishing"
    );

    // The assertion the census exists to make: the island's real meshes are
    // NOT all on a three-tier ladder, and the grammar buildings — which are most
    // of what a settlement draws — are on two bands.
    let with_dag = meshes.iter().filter(|(_, _, d)| *d).count();
    row(
        "meshes with a meshlet ladder",
        format!("{with_dag} of {}", meshes.len()),
    );
    assert!(
        !meshes.is_empty(),
        "the pack carries no meshes, so this census counted nothing"
    );
}

/// **CP-C3 · A GRAMMAR BUILDING NOW DRAWS IN THREE DISTANCE BANDS.**
///
/// Before this wave it drew in two — every part out to `STRUCTURE_LOD_M`, then
/// one shell box — and the certification's owner asked for three. The mid rung
/// is `INTERIOR_LOD_M`: a FIT-OUT family (a chair, a counter, a television)
/// stops drawing where the building stops having colliders, because past that
/// you cannot be inside it and the window between you and it is an opaque box
/// (CP-B10). The census above prices it: **12.2 %** of a furnished building's
/// instances, 23.6 % of a bar's.
///
/// # Why the world here is hand-built
///
/// The fixture island's resident blocks are Office, Apartment and Industrial
/// zones, and `inf_editor_core::settlement::furnishes` turns furniture OFF for
/// all three — so a projection of the fixture contains no fit-out at all and an
/// arm written over it would assert the band into existence out of an empty set.
/// (It did, on this file's first run, and said so by failing.) So the world is
/// four instances and one group: two fabric, two fit-out, one shell — the
/// smallest world in which three complementary bands can be told apart.
#[test]
fn a_grammar_building_draws_in_three_distance_bands() {
    use inf_ecs::components::{
        PcgVolume, ScatteredInstance, ScatteredSolid, ScatteredSurface, StructureGroup, Transform,
    };

    let fabric = inf_pcg::building::modules::module_mesh_guid(
        inf_pcg::building::modules::ModuleShape::Panel,
    );
    let fit_out = inf_pcg::building::modules::module_mesh_guid(
        inf_pcg::building::modules::ModuleShape::Legged,
    );
    assert!(
        !inf_pcg::building::modules::is_fit_out_mesh(fabric)
            && inf_pcg::building::modules::is_fit_out_mesh(fit_out),
        "the two families this arm bands apart are not on opposite sides of the classification, so nothing below is a test of the band"
    );

    let at = |x: f64| DVec3::new(x, 0.0, 0.0);
    let inst = |mesh, x| ScatteredInstance {
        position: at(x),
        rotation: glam::DQuat::IDENTITY,
        scale: 1.0,
        kind: 0,
        mesh: Some(mesh),
        extent: None,
        glow: 0.0,
        surface: ScatteredSurface::DEFAULT,
    };
    let solid = |x: f64| ScatteredSolid {
        center: at(x),
        half_extents: DVec3::splat(0.25),
        rotation: glam::DQuat::IDENTITY,
    };

    let mut world = inf_ecs::EcsWorld::new();
    let e = world.spawn_with_guid(uuid::Uuid::from_u128(0xCE_1201), "Block", None);
    let mut vol = PcgVolume::default();
    vol.set_population(
        vec![
            inst(fabric, 0.0),
            inst(fabric, 1.0),
            inst(fit_out, 2.0),
            inst(fit_out, 3.0),
        ],
        vec![solid(0.0), solid(1.0), solid(2.0), solid(3.0)],
        vec![StructureGroup {
            shell: ScatteredSolid {
                center: at(1.5),
                half_extents: DVec3::splat(4.0),
                rotation: glam::DQuat::IDENTITY,
            },
            start: 0,
            len: 4,
            inst_start: 0,
            inst_len: 4,
        }],
        Vec::new(),
        Vec::new(),
        Default::default(),
        Vec::new(),
        Vec::new(),
    );
    world
        .world_mut()
        .entity_mut(e)
        .insert(Transform::default())
        .insert(vol);
    world.propagate();
    let mut sim = RuntimeSim::new(world, Vec::new(), glam::DVec2::ZERO, 60.0);
    sim.step_once(RuntimeInput::default());

    // The two families must resolve to real geometry, or both bucket with the
    // meshless placeholder and the key the band reads is `None` for both.
    let mut meshes = inf_render::ScatterMeshes::new();
    for (id, m) in inf_pcg::building::modules::module_meshes() {
        let g = inf_render::ScatterGeometry::from_streams(&m.positions, &m.normals, &m.indices);
        meshes.insert(id.as_u128(), std::sync::Arc::new(g));
    }

    let mut scene = inf_render::RenderScene::default();
    let mut debris = inf_render::DebrisCache::default();
    let voxels = inf_voxel::VoxelVolumes::new();
    inf_player::render::project_scene_full(
        &mut scene,
        &sim,
        0.0,
        &inf_player::vmesh::VmeshRegistry::new(),
        &inf_player::skinned::SkinnedRegistry::new(),
        &voxels,
        &mut debris,
        None,
        &meshes,
        &std::collections::HashMap::new(),
    );

    let mut bands: BTreeMap<(u64, u64), (usize, usize)> = BTreeMap::new();
    for b in &scene.scatter {
        let e = bands
            .entry((b.near_distance.to_bits(), b.draw_distance.to_bits()))
            .or_insert((0, 0));
        e.0 += 1;
        e.1 += b.data.instances.len();
    }
    println!("CP-C3 the bands the projector emits for one grammar building:");
    for ((near, draw), (batches, insts)) in &bands {
        println!(
            "  [{:>7.1}, {:>7.1})  {batches} batch(es)  {insts} instance(s)",
            f64::from_bits(*near),
            f64::from_bits(*draw)
        );
    }

    let count_where = |f: &dyn Fn(&inf_render::ScatterBatch) -> bool| -> usize {
        scene
            .scatter
            .iter()
            .filter(|b| f(b))
            .map(|b| b.data.instances.len())
            .sum()
    };
    let near = count_where(&|b| b.draw_distance == inf_render::INTERIOR_LOD_M);
    let far = count_where(&|b| b.near_distance == inf_render::STRUCTURE_LOD_M);
    let mid = count_where(&|b| {
        b.draw_distance != inf_render::INTERIOR_LOD_M
            && b.near_distance != inf_render::STRUCTURE_LOD_M
    });
    println!("  near (fit-out) {near} · mid (fabric) {mid} · far (shell) {far}");

    assert_eq!(
        near, 2,
        "the fit-out band did not take exactly the two fit-out instances"
    );
    assert_eq!(
        mid, 2,
        "the fabric band did not take exactly the two fabric instances"
    );
    assert_eq!(far, 1, "the shell band is not one box");
    assert!(
        bands.len() >= 3,
        "the projector emitted {} distinct bands — three rungs wearing fewer names",
        bands.len()
    );
    const {
        assert!(
            inf_render::INTERIOR_LOD_M < inf_render::STRUCTURE_LOD_M,
            "the mid rung is not inside the far one, so there are three names and two bands"
        );
    }
    // The tie `INTERIOR_LOD_M`'s own doc claims. `inf-render` cannot name
    // `inf-ecs`, so this is the only place the equality can be checked at all.
    assert_eq!(
        inf_render::INTERIOR_LOD_M,
        inf_ecs::band::DEFAULT_COLLIDER_NEAR_M,
        "INTERIOR_LOD_M has drifted off the collider band it is defined as"
    );
}

/// **CP-C5 · THE TEXTURE LADDER, BY DISTANCE BAND.**
///
/// The certification's owner asked for three tiers on textures as well as on
/// meshes. Textures have been virtual since P26 and their ladder is
/// **continuous**, not three-runged — but "continuous" is a claim, and this is
/// the census that turns it into a table: for each of the island's own textures,
/// the mip level `justified_mip` justifies at a set of distances, through the
/// SAME pure function the CPU floor, the GPU feedback shader and the visbuffer
/// feedback all derive their level from.
///
/// No GPU: `inf-vt` is a GPU-free crate by design and the rule is a pure
/// function of extent, screen pixels and mip count.
#[test]
fn the_texture_ladder_by_distance_band() {
    let tmp = tempfile::tempdir().expect("a temp dir");
    let pack = cook_fixture(tmp.path());
    let reader = inf_asset::PackReader::open(&pack.join(inf_player::level::PACK_FILE))
        .expect("the pack maps");

    // A 1080p view at the fps instrument's own field of view, so the pixels this
    // census counts are the pixels that instrument's frames have.
    let view = inf_render::RenderView {
        origin: inf_math::FloatingOrigin::new(DVec3::ZERO),
        eye_world: DVec3::ZERO,
        forward: glam::Vec3::Z,
        up: glam::Vec3::Y,
        fov_y: 70f32.to_radians(),
        near: 0.05,
        width: 1920,
        height: 1080,
        ortho: None,
    };
    let scale = inf_render::vt_stream::projection_scale(&view);

    // The ground materials the island's terrain layers bind: a 2 m patch of
    // ground is the thing a walking player is looking at, so a 1 m radius is the
    // footprint the level is asked about.
    const BANDS_M: [f32; 6] = [1.0, 4.0, 16.0, 64.0, 256.0, 1024.0];
    let mut extents: Vec<(String, u32, u32)> = Vec::new();
    for e in reader.index() {
        if e.kind != inf_asset::AssetKind::Texture {
            continue;
        }
        let Ok(bytes) = reader.read_ref(e.guid) else {
            continue;
        };
        let Ok(view) = inf_vt::TiledTextureView::new(bytes.as_ref()) else {
            continue;
        };
        let d = view.vt_desc();
        let Some(m0) = d.mips.first() else {
            continue;
        };
        extents.push((e.guid.0.to_string(), m0.width.max(m0.height), d.mip_count()));
    }
    extents.sort();

    println!("CP-C5 the texture ladder, 1080p / 70° fov, a 1 m-radius footprint:");
    print!("  {:<38}{:>6}{:>5}", "texture", "extent", "mips");
    for m in BANDS_M {
        print!("{:>8.0} m", m);
    }
    println!();
    for (id, extent, mips) in &extents {
        print!("  {id:<38}{extent:>6}{mips:>5}");
        for d in BANDS_M {
            let px = inf_render::vt_stream::screen_diameter_px(
                glam::Vec3::new(0.0, 0.0, d),
                1.0,
                glam::Vec3::ZERO,
                scale,
            );
            print!(
                "{:>10}",
                inf_render::vt_stream::justified_mip(*extent, px, *mips)
            );
        }
        println!();
    }
    println!(
        "  the coarsest {} mip level(s) of every texture are resident unconditionally (VT_FLOOR_LEVELS); a tile is {}x{} with a {}-texel border",
        inf_render::VT_FLOOR_LEVELS,
        inf_vt::TILE_SIZE,
        inf_vt::TILE_SIZE,
        inf_vt::TILE_BORDER
    );

    assert!(
        !extents.is_empty(),
        "the island pack carries no virtual textures, so this census counted nothing"
    );
    // The ladder is CONTINUOUS, and the way to say so with an assertion is that
    // it has more rungs than the three the owner asked for: over six decades of
    // distance the justified level must actually move, and by more than two.
    let (_, extent, mips) = &extents[0];
    let level = |d: f32| {
        let px = inf_render::vt_stream::screen_diameter_px(
            glam::Vec3::new(0.0, 0.0, d),
            1.0,
            glam::Vec3::ZERO,
            scale,
        );
        inf_render::vt_stream::justified_mip(*extent, px, *mips)
    };
    let near = level(BANDS_M[0]);
    let far = level(BANDS_M[BANDS_M.len() - 1]);
    assert!(
        far > near + 2,
        "the texture ladder moved only {} level(s) between {:.0} m and {:.0} m, which is fewer rungs than the three the certification asks for",
        far - near,
        BANDS_M[0],
        BANDS_M[BANDS_M.len() - 1]
    );
}

// ── the shipped island, when it is on this machine ──────────────────────────

/// **The same censuses over the SHIPPED 51.38 km² island.**
///
/// `#[ignore]`d and env-gated, following `block_codec_bakeoff::island_codec_bakeoff`:
/// the corpus that matters is not in the repository, the arms above assert the
/// properties on one everybody has, and this prints the numbers the memo quotes.
///
/// ```text
/// INF_CERT_ISLAND_PACK=<...>/island-build/project/Build \
///   cargo test --release -p inf-player --test parity_cert -- --ignored --nocapture
/// ```
#[test]
#[ignore = "needs INF_CERT_ISLAND_PACK pointing at a built island's Build directory"]
fn the_shipped_island_censused() {
    let Ok(dir) = std::env::var("INF_CERT_ISLAND_PACK") else {
        println!("INF_CERT_ISLAND_PACK unset — nothing to measure");
        return;
    };
    let pack = PathBuf::from(dir);

    // **CP-C1, ON THIS MACHINE.** `inf_project::boot`'s arms prove the WALK
    // against a temp directory; this proves the walk finds the showcase from the
    // real executable's real directory, which is the only place the claim "the
    // application opens on the island" can be checked at all. It lives on the
    // ignored arm because it is a statement about a machine that has run
    // `inf island build`, which is the same condition the pack above needs.
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(std::path::Path::to_path_buf));
    match exe_dir.as_deref().and_then(inf_project::find_showcase) {
        Some(root) => println!(
            "CP-C1 the showcase rung, from {}: {}",
            exe_dir.as_deref().unwrap_or(Path::new("?")).display(),
            root.display()
        ),
        None => println!(
            "CP-C1 the showcase rung found nothing from {} — the editor would open \
             its start screen here",
            exe_dir.as_deref().unwrap_or(Path::new("?")).display()
        ),
    }

    let t0 = std::time::Instant::now();
    let mut sim = pack_sim(&pack);
    let open_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let t1 = std::time::Instant::now();
    sim.step_once(RuntimeInput::default());
    let step_ms = t1.elapsed().as_secs_f64() * 1000.0;
    println!(
        "SHIPPED ISLAND cold start: open+stream {open_ms:.1} ms + first fixed step \
         {step_ms:.1} ms = {:.1} ms",
        open_ms + step_ms
    );

    let start = hero_at(&sim).expect("the shipped island has a hero");
    for _ in 0..60 {
        sim.step_once(RuntimeInput::default());
    }
    let after = hero_at(&sim).expect("it survived");
    println!(
        "SHIPPED ISLAND ground under the pawn: {:+.4} m of fall in 1.0 s (start y {:.2})",
        start.y - after.y,
        start.y
    );

    let mut per_hour: BTreeMap<String, (f64, usize, usize, usize)> = BTreeMap::new();
    for hour in [8.5_f64, 14.0, 21.0] {
        let mut s = pack_sim(&pack);
        freeze_clock(&mut s, hour);
        settle(&mut s, 3_000);
        let metres = street_metres(&s);
        let (parked, driving, peds) = occupancy(&s);
        per_hour.insert(format!("{hour:04.1}"), (metres, parked, driving, peds));
    }
    println!("SHIPPED ISLAND street occupancy:");
    for (hour, (metres, parked, driving, peds)) in &per_hour {
        let per100 = match *metres > 0.0 {
            true => (*parked + *driving + *peds) as f64 * 100.0 / *metres,
            false => 0.0,
        };
        let per = |n: usize| match *metres > 0.0 {
            true => n as f64 * 100.0 / *metres,
            false => 0.0,
        };
        println!(
            "  {hour}  {metres:8.0} m  parked {parked:5} ({:5.2}/100m)  driving {driving:4} ({:5.2}/100m)  peds {peds:4} ({:5.2}/100m)  = {per100:.2} per 100 m",
            per(*parked),
            per(*driving),
            per(*peds)
        );
    }
}
