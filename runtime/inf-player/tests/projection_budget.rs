//! **The PROJECTION budget** (Hardening Wave E, 2026-08-14) — a §8 ratchet over
//! the seam between the simulation and the renderer, which had no budget of any
//! class before this file existed.
//!
//! # Why this is a new class rather than a new number
//!
//! §8's existing budgets each cover one side of the frame. `FRAME_BUDGET_MS` is
//! *the renderer given a scene*; `SIM_STEP_BUDGET_MS` and
//! [`STREAMED_STEP_BUDGET_MS`](inf_player::budget::STREAMED_STEP_BUDGET_MS) are
//! *the simulation given a world*; [`LOAD_BUDGET_MS`](inf_player::budget::LOAD_BUDGET_MS)
//! is *the one-shot build*. Between the second and the first sits
//! [`project_scene_full`](inf_player::render::project_scene_full): the function
//! that turns an `EcsWorld` into the `RenderScene` the renderer is then measured
//! against. **Nothing measured it.** A frame could spend more time being
//! projected than being drawn and every gate in the tree would stay green.
//!
//! That gap is not hypothetical. The audit's lens 3 found three producers
//! (terrain tiles, voxel chunks, fracture chunks) rebuilding payloads from
//! scratch every frame against stamps their own consumers honour — work whose
//! entire cost lands here, in a function no budget could see, and whose *result*
//! is then discarded by a version check in the renderer.
//!
//! # What is measured, precisely
//!
//! One call to `project_scene_full` over a fixture-sized streamed world, sixty
//! times, into the **same** `RenderScene` — which is exactly what
//! `PlayerRenderHost::project` does every frame. No GPU is involved at any point,
//! so unlike the frame budget this arm asserts on **every** machine, CI included:
//! a projection is CPU work over CPU data, and the P26 rule ("prefer a budget in
//! a unit the machine cannot inflate") is honoured differently here — the unit is
//! still a millisecond, but the ~4× shared-runner factor from
//! [`LOAD_BUDGET_MS`](inf_player::budget::LOAD_BUDGET_MS)'s docs is a CPU factor
//! and the constant carries it.
//!
//! # The arms
//!
//! * **(a) the budget** — mean projection under
//!   [`PROJECTION_BUDGET_MS`](inf_player::budget::PROJECTION_BUDGET_MS).
//! * **(b) anti-vacuity** — the fixture really carries the heavy payloads (tiles,
//!   chunks, instances). A budget over an empty scene is a number about nothing,
//!   and this file's whole subject is per-tile and per-chunk work.
//! * **(c) the churn is real** — `project_scene_full` bumps `scene.version` on
//!   every call, which is the regime `inf-render`'s
//!   `frame_stays_under_budget_under_version_churn` measures on the other side of
//!   the seam. Pinned here because it is the premise of that arm.
//! * **(d) a re-projection is byte-identical to a cold one** — the arm that makes
//!   memoizing a projector *safe*. A memo keyed on a change stamp is only correct
//!   if a hit and a miss produce the same payload; this projects into a used scene
//!   and into a fresh one and compares the two, field for field, on every heavy
//!   list. It was written before any memo existed (where it holds trivially) so
//!   that it can only ever go red for the right reason.

use glam::{DVec2, DVec3};
use uuid::Uuid;

use inf_ecs::components::{MeshRef, Terrain, Transform, VoxelVolume};
use inf_ecs::EcsWorld;
use inf_player::budget::{PROJECTION_BUDGET_MS, RATCHET_NOTE};
use inf_player::render::{project_scene_full, sync_voxel_store};
use inf_player::runtime_sim::{RuntimeInput, RuntimeSim};
use inf_render::RenderScene;
use inf_terrain::TerrainData;
use inf_voxel::{build_voxel_asset, ChunkKey, VoxelChunk, VoxelData, VoxelVolumes, CHUNK_DIM};

// ── the fixture ─────────────────────────────────────────────────────────────

/// Tile resolution — the Phase-16 gate scene's, so the per-tile payload this
/// measures is the one the streaming gate already ships.
const TILE_RES: u32 = 129;
const MPS: f64 = 1.0;
/// 6 × 6 = 36 level-0 tiles. Below the gate scene's 64 (this runs on every CI
/// runner of three operating systems and a projection is not free), far above
/// the one-tile fixtures every other in-process test uses.
const TILES_PER_SIDE: i32 = 6;
/// Ordinary mesh entities, so the projector's entity walk is a real walk rather
/// than a two-element loop.
const PROPS: u32 = 200;

const TERRAIN_GUID: u128 = 0x1E00_0001;
const CAVE_GUID: u128 = 0x1E00_0002;
const PROP_GUID_BASE: u128 = 0x1E00_1000;

/// A rolling heightfield over [`TILES_PER_SIDE`]² authored tiles.
///
/// `psin`/`pcos` rather than `f64::sin` — the P14 LAW. Nothing here is
/// committed, but a fixture whose content depends on the libm build is a fixture
/// whose *measurement* does too.
fn terrain_data() -> TerrainData {
    let mut data = TerrainData::new(TILE_RES, MPS);
    for tx in 0..TILES_PER_SIDE {
        for tz in 0..TILES_PER_SIDE {
            data.author_tile((tx, tz), |x, z| {
                6.0 * inf_math::psin64(x * 0.02) + 4.0 * inf_math::pcos64(z * 0.017)
            });
        }
    }
    data
}

/// A slab of rock four chunks on a side — a cave-sized volume whose surface
/// meshes into a real vertex stream rather than into nothing.
fn cave_data() -> VoxelData {
    let mut v = VoxelData::new(0.5);
    let surface = f64::from(CHUNK_DIM) * 1.5;
    for cx in 0..4 {
        for cz in 0..4 {
            for cy in 0..2 {
                let key = ChunkKey::new(cx, cy, cz);
                v.insert_chunk(
                    key,
                    VoxelChunk::from_fn(|_, j, _| {
                        f64::from(cy * CHUNK_DIM as i32 + j as i32) - surface
                    }),
                );
            }
        }
    }
    v.clear_dirty();
    v
}

fn world() -> EcsWorld {
    let mut world = EcsWorld::new();

    let t = world.spawn_with_guid(Uuid::from_u128(TERRAIN_GUID), "Terrain", None);
    world
        .world_mut()
        .entity_mut(t)
        .insert(Transform::default())
        .insert(Terrain {
            meters_per_sample: MPS,
            tile_resolution: TILE_RES,
            data: terrain_data(),
            ..Terrain::default()
        });

    let c = world.spawn_with_guid(Uuid::from_u128(CAVE_GUID), "Cave", None);
    world
        .world_mut()
        .entity_mut(c)
        .insert(Transform::from_translation(DVec3::new(8.0, 0.0, 8.0)))
        .insert(VoxelVolume {
            asset: Some(Uuid::from_u128(CAVE_GUID)),
            ..VoxelVolume::default()
        });

    for i in 0..PROPS {
        let e = world.spawn_with_guid(
            Uuid::from_u128(PROP_GUID_BASE + u128::from(i)),
            "Prop",
            None,
        );
        let (x, z) = ((i % 20) as f64 * 3.0, (i / 20) as f64 * 3.0);
        world
            .world_mut()
            .entity_mut(e)
            .insert(Transform::from_translation(DVec3::new(x, 2.0, z)))
            .insert(MeshRef::default());
    }
    world.propagate();
    world
}

/// The camera the residency passes are driven from — inside the cave slab, so
/// the voxel store really pages and meshes.
const EYE: DVec3 = DVec3::new(12.0, 6.0, 12.0);

/// Sim + a synced voxel store, ready to project. Mirrors `PlayerRenderHost`'s own
/// two-step (`sync_voxels` then project) because that is the split being measured.
fn fixture() -> (RuntimeSim, VoxelVolumes, inf_player::voxel::VoxelRegistry) {
    let mut sim = RuntimeSim::new(world(), Vec::new(), DVec2::ZERO, 60.0);
    sim.step_once(RuntimeInput::default());

    let bytes = build_voxel_asset(&cave_data())
        .expect("the fixture volume encodes")
        .as_bytes()
        .to_vec();
    let registry =
        inf_player::voxel::VoxelRegistry::from_payload(&[(Uuid::from_u128(CAVE_GUID), bytes)]);

    let mut voxels = VoxelVolumes::new();
    sync_voxel_store(&mut voxels, &registry, &sim, EYE);
    (sim, voxels, registry)
}

/// One projection into `scene`, exactly as `PlayerRenderHost::project` performs
/// it (the widest door, with a live debris memo).
fn project(scene: &mut RenderScene, sim: &RuntimeSim, voxels: &VoxelVolumes) {
    project_scene_full(
        scene,
        sim,
        1.0,
        &inf_player::vmesh::VmeshRegistry::default(),
        &inf_player::skinned::SkinnedRegistry::new(),
        voxels,
        &mut inf_render::DebrisCache::default(),
        None,
        // This gate's world is a voxel/debris budget fixture with no scatter
        // kinds in it, so an empty table is the whole truth about it.
        &inf_render::ScatterMeshes::new(),
        &std::collections::HashMap::new(),
    );
}

// ── the deliverable ─────────────────────────────────────────────────────────

/// **(a) + (b) + (c).** What one sim→render projection costs, measured in the
/// regime the shipped player runs it in: the same scene, over and over.
#[test]
fn projection_stays_under_budget() {
    let (sim, voxels, _registry) = fixture();
    let mut scene = RenderScene::default();

    const WARMUP: u32 = 5;
    const MEASURED: u32 = 60;
    for _ in 0..WARMUP {
        project(&mut scene, &sim, &voxels);
    }

    let before = scene.version;
    let start = std::time::Instant::now();
    for _ in 0..MEASURED {
        project(&mut scene, &sim, &voxels);
    }
    let mean_ms = start.elapsed().as_secs_f64() * 1000.0 / f64::from(MEASURED);

    let tiles: usize = scene.terrains.iter().map(|t| t.tiles.len()).sum();
    let chunks: usize = scene.voxels.iter().map(|v| v.chunks.len()).sum();
    let verts: usize = scene
        .voxels
        .iter()
        .flat_map(|v| v.chunks.iter())
        .map(|c| c.vertices.len())
        .sum();
    eprintln!(
        "projection_budget: mean {mean_ms:.3} ms/projection over {MEASURED} \
         projections — {} terrains / {tiles} tiles at {TILE_RES}², {} volumes / \
         {chunks} chunks / {verts} vertices, {} instances; budget \
         {PROJECTION_BUDGET_MS} ms",
        scene.terrains.len(),
        scene.voxels.len(),
        scene.instances.len(),
    );

    // (b) ANTI-VACUITY. Every claim above is per-tile and per-chunk work; a
    // fixture that projected an empty scene would sail under any budget and
    // measure nothing at all.
    assert!(
        tiles >= (TILES_PER_SIDE * TILES_PER_SIDE) as usize,
        "the fixture must project every authored tile, got {tiles}"
    );
    assert!(
        chunks > 0 && verts > 0,
        "the fixture's voxel volume must project a meshed surface, got \
         {chunks} chunks / {verts} vertices"
    );
    assert!(
        scene.instances.len() >= PROPS as usize,
        "the fixture must project its props, got {}",
        scene.instances.len()
    );

    // (c) THE CHURN IS REAL. `inf-render`'s
    // `frame_stays_under_budget_under_version_churn` measures the renderer in the
    // regime this line pins: one bump per projection, on the producing side.
    assert_eq!(
        scene.version.wrapping_sub(before),
        u64::from(MEASURED),
        "`project_scene_full` must bump `scene.version` exactly once per call — \
         the premise of the renderer's version-churn budget arm"
    );

    assert!(
        mean_ms < PROJECTION_BUDGET_MS,
        "the sim→render projection cost {mean_ms:.3} ms, over the \
         {PROJECTION_BUDGET_MS} ms projection budget {RATCHET_NOTE}"
    );
}

/// **(d)** A projection into a scene that already holds last frame's payload must
/// equal a projection into a fresh one — field for field, on every heavy list.
///
/// This is the arm that licenses memoizing a projector against a change stamp.
/// Such a memo is correct exactly when a *hit* and a *miss* produce identical
/// bytes, and the two paths here are precisely that pair: `used` takes whatever
/// fast path a re-projection has, `cold` cannot take any. It is written to hold
/// trivially today (there is no memo yet) so that when one lands it can only go
/// red for the reason it exists.
#[test]
fn a_reprojection_is_byte_identical_to_a_cold_one() {
    let (sim, voxels, _registry) = fixture();

    let mut used = RenderScene::default();
    project(&mut used, &sim, &voxels);
    project(&mut used, &sim, &voxels);

    let mut cold = RenderScene::default();
    project(&mut cold, &sim, &voxels);

    assert_eq!(
        used.terrains, cold.terrains,
        "a re-projected terrain differs from a cold one"
    );
    assert_eq!(
        used.voxels, cold.voxels,
        "a re-projected voxel volume differs from a cold one"
    );
    assert_eq!(
        used.fracture_chunks, cold.fracture_chunks,
        "a re-projected fracture set differs from a cold one"
    );
    // `MeshInstance` and `RenderLight` carry no `PartialEq`, so these two compare
    // their `Debug` renderings — which is an exact comparison here: every field is
    // an `f32`/`f64`/integer and `Debug` for a float is the shortest round-tripping
    // decimal, so two renderings are equal exactly when the bits are.
    assert_eq!(
        format!("{:?}", used.instances),
        format!("{:?}", cold.instances),
        "a re-projected instance list differs from a cold one"
    );
    assert_eq!(
        used.scatter.len(),
        cold.scatter.len(),
        "a re-projected scatter list differs from a cold one"
    );
    assert_eq!(
        format!("{:?}", used.lights),
        format!("{:?}", cold.lights),
        "a re-projected light list differs from a cold one"
    );
    // Anti-vacuity for THIS arm: comparing two empty lists proves nothing.
    assert!(!cold.terrains.is_empty() && !cold.voxels.is_empty());
}

/// **The falsifier.** A carried projection must still see an edit — otherwise
/// "carried forward unchanged" is just "never updated again".
///
/// Sculpts one tile through the stamped door (`get_tile_mut`, which is what every
/// brush, splat, biome and carve path in the tree goes through), re-projects into
/// the **same** scene, and asserts the new heights are on screen — and that the
/// tiles nobody touched came through unchanged, so the fix is a memo rather than
/// a rebuild wearing one.
///
/// Mutation-verified: a `tiles_match` hard-wired to `true` leaves the byte-identity
/// arm green and fails here.
#[test]
fn a_carried_projection_still_sees_an_edit() {
    let (mut sim, voxels, _registry) = fixture();
    let mut scene = RenderScene::default();
    project(&mut scene, &sim, &voxels);
    let before = scene.terrains[0].clone();

    // The edit: raise every sample of tile (2, 2) by ten metres.
    let entity = sim
        .world()
        .entity_of(Uuid::from_u128(TERRAIN_GUID))
        .expect("the fixture's terrain");
    {
        let world = sim.world_mut();
        let mut terrain = world
            .world_mut()
            .get_mut::<Terrain>(entity)
            .expect("the fixture's terrain component");
        let tile = terrain
            .data
            .get_tile_mut((2, 2))
            .expect("the fixture authored (2, 2)");
        for h in tile.heights_mut() {
            *h += 10.0;
        }
    }

    project(&mut scene, &sim, &voxels);
    let after = &scene.terrains[0];
    assert_eq!(
        before.tiles.len(),
        after.tiles.len(),
        "the edit must not change the resident set"
    );

    let mut moved = 0usize;
    for (b, a) in before.tiles.iter().zip(after.tiles.iter()) {
        assert_eq!(b.key, a.key, "the tile order must not change");
        if b.heights == a.heights {
            continue;
        }
        moved += 1;
        assert_eq!(a.key.coord, (2, 2), "only the edited tile may move");
        for (b, a) in b.heights.iter().zip(a.heights.iter()) {
            assert!(
                (a - b - 10.0).abs() < 1e-4,
                "the carried projection served a stale height: {b} -> {a}"
            );
        }
        assert_ne!(
            b.version, a.version,
            "the edited tile's stamp must have moved — the memo's whole key"
        );
    }
    assert_eq!(moved, 1, "exactly the edited tile must be re-projected");
}

/// **The second falsifier, for the case a cheaper key could not see.**
///
/// A monotone *maximum* over the tile stamps — the obvious cheap memo key — is
/// blind to the eviction of a non-maximal tile, because a removal deletes the
/// ledger entry instead of minting a new number. So this removes a tile that is
/// *not* the most recently stamped one and asserts the carried projection sheds
/// it. The comparison is a whole-sequence walk precisely so that this passes.
#[test]
fn a_carried_projection_sees_a_tile_leave() {
    let (mut sim, voxels, _registry) = fixture();
    let mut scene = RenderScene::default();
    project(&mut scene, &sim, &voxels);
    let before = scene.terrains[0].tiles.len();

    let entity = sim
        .world()
        .entity_of(Uuid::from_u128(TERRAIN_GUID))
        .expect("the fixture's terrain");
    {
        let world = sim.world_mut();
        let mut terrain = world
            .world_mut()
            .get_mut::<Terrain>(entity)
            .expect("the fixture's terrain component");
        // (5, 5) was authored LAST, so (0, 0) is emphatically not the maximum
        // stamp — which is the entire point of removing that one.
        terrain.data.remove_tile((0, 0));
    }

    project(&mut scene, &sim, &voxels);
    let after = &scene.terrains[0];
    assert_eq!(
        after.tiles.len(),
        before - 1,
        "a removed tile must leave the projection"
    );
    assert!(
        after.tiles.iter().all(|t| t.key.coord != (0, 0)),
        "the removed tile is still being drawn"
    );
}

// ── the SCATTER carry-forward (island wave I8a audit) ───────────────────────
//
// Wave E carried terrain tiles and voxel chunks forward and left the third heavy
// payload alone: a `PcgVolume`'s scatter was re-packed to f32 and re-hashed every
// frame. On the island's 172 settlement blocks that was 365 545 instances and
// **20.2 ms a projection against the 1.5 ms budget above** — a defect the wave's
// own ledger routed with its price.
//
// The arms below have their OWN fixture rather than growing `fixture()`, and that
// is deliberate: `projection_stays_under_budget` pins a RATCHET, and a ratchet
// re-measured over a different world is a ratchet nobody minted.

const VOLUME_GUID: u128 = 0x1E00_0003;
/// Instances in the fixture volume. Enough that a re-pack is a measurable
/// allocation and a carry is not, small enough that this runs on a CI runner.
const SCATTER_N: usize = 20_000;

/// One volume's population — `n` instances in twenty buildings, so the projection
/// takes the grouped path (loose + parts + shells) rather than the flat one.
#[allow(clippy::type_complexity)]
fn population(
    n: usize,
    lift: f64,
) -> (
    Vec<inf_ecs::components::ScatteredInstance>,
    Vec<inf_ecs::components::ScatteredSolid>,
    Vec<inf_ecs::components::StructureGroup>,
) {
    let mut inst = Vec::with_capacity(n);
    let mut solid = Vec::with_capacity(n);
    for i in 0..n {
        let (x, z) = ((i % 200) as f64 * 0.5, (i / 200) as f64 * 0.5);
        inst.push(inf_ecs::components::ScatteredInstance {
            position: DVec3::new(x, lift, z),
            rotation: glam::DQuat::IDENTITY,
            scale: 1.0,
            kind: (i % 5) as u32,
            mesh: None,
            extent: None,
            glow: 0.0,
            surface: inf_ecs::components::ScatteredSurface::DEFAULT,
        });
        solid.push(inf_ecs::components::ScatteredSolid {
            center: DVec3::new(x, lift, z),
            half_extents: DVec3::splat(0.25),
            rotation: glam::DQuat::IDENTITY,
        });
    }
    let per = (n / 20) as u32;
    let groups = (0..20u32)
        .map(|g| inf_ecs::components::StructureGroup {
            shell: inf_ecs::components::ScatteredSolid {
                center: DVec3::new(g as f64 * 5.0, lift, 0.0),
                half_extents: DVec3::splat(4.0),
                rotation: glam::DQuat::IDENTITY,
            },
            start: g * per,
            len: per,
            inst_start: g * per,
            inst_len: per,
        })
        .collect();
    (inst, solid, groups)
}

/// A sim holding one `PcgVolume` with a real population, and nothing else heavy.
fn scatter_fixture() -> RuntimeSim {
    let mut world = EcsWorld::new();
    let e = world.spawn_with_guid(Uuid::from_u128(VOLUME_GUID), "Blocks", None);
    let mut vol = inf_ecs::components::PcgVolume::default();
    let (inst, solid, groups) = population(SCATTER_N, 0.0);
    vol.set_population(
        inst,
        solid,
        groups,
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
    let mut sim = RuntimeSim::new(world, Vec::new(), DVec2::ZERO, 60.0);
    sim.step_once(RuntimeInput::default());
    sim
}

fn project_scatter(scene: &mut RenderScene, sim: &RuntimeSim) {
    project(scene, sim, &VoxelVolumes::new());
}

/// The volume's entity, for a test that wants to rewrite its population.
fn volume_of(sim: &RuntimeSim) -> inf_ecs::Entity {
    sim.world()
        .entity_of(Uuid::from_u128(VOLUME_GUID))
        .expect("the fixture's volume")
}

/// **(e) A CARRIED SCATTER IS THE SAME PAYLOAD, AND IT IS NOT RE-PACKED.**
///
/// Two claims, and the second is what makes the fix a fix rather than a comment:
///
/// * the batches a second projection produces `==` the first's, field for field —
///   which is (d) restated for the list (d) only counted; and
/// * the `Arc` behind each is **pointer-identical**, which no re-pack can
///   produce. `ScatterData::key` would have made the two *equal* either way,
///   which is exactly why this asserts the pointer.
///
/// The wall clock is printed and not asserted (this module's own law), but the
/// ratio is the whole point: a hit is a pointer copy.
#[test]
fn a_carried_scatter_is_the_same_payload_and_is_not_re_packed() {
    let sim = scatter_fixture();
    let mut scene = RenderScene::default();

    let t = std::time::Instant::now();
    project_scatter(&mut scene, &sim);
    let cold_ms = t.elapsed().as_secs_f64() * 1000.0;
    let first: Vec<inf_render::ScatterBatch> = scene.scatter.clone();
    let packed: usize = first.iter().map(|b| b.data.len()).sum();
    assert!(
        !first.is_empty() && packed == SCATTER_N + 20,
        "the fixture volume projected {} batches / {packed} instances — this arm \
         is over the wrong world",
        first.len()
    );

    let t = std::time::Instant::now();
    project_scatter(&mut scene, &sim);
    let warm_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!(
        "SCATTER MEMO: {} batches / {packed} instances; cold {cold_ms:.3} ms, \
         carried {warm_ms:.3} ms; the memo holds {} source(s)",
        first.len(),
        scene.scatter_memo.len()
    );
    assert_eq!(
        scene.scatter, first,
        "a carried batch list is not the cold one"
    );
    for (a, b) in first.iter().zip(&scene.scatter) {
        assert!(
            std::sync::Arc::ptr_eq(&a.data, &b.data),
            "the carried projection re-packed a payload it already had — the \
             batches are equal, so only the POINTER can say so"
        );
    }
    // …and a cold scene reaches the same answer, which is (d) for this list.
    let mut cold = RenderScene::default();
    project_scatter(&mut cold, &sim);
    assert_eq!(
        cold.scatter, scene.scatter,
        "a re-projected scatter list differs from a cold one"
    );
}

/// **The falsifier.** A carried scatter must still see a new population.
///
/// Mutation-verified: a `ScatterMemo::take` that ignored `source.stamp` leaves
/// the arm above green and fails here.
#[test]
fn a_carried_scatter_still_sees_a_new_population() {
    let mut sim = scatter_fixture();
    let mut scene = RenderScene::default();
    project_scatter(&mut scene, &sim);
    let before = scene.scatter.clone();

    let e = volume_of(&sim);
    {
        let (inst, solid, groups) = population(SCATTER_N, 7.0);
        let world = sim.world_mut();
        let mut vol = world
            .world_mut()
            .get_mut::<inf_ecs::components::PcgVolume>(e)
            .expect("the fixture's volume component");
        vol.set_population(
            inst,
            solid,
            groups,
            Vec::new(),
            Vec::new(),
            Default::default(),
            Vec::new(),
            Vec::new(),
        );
    }

    project_scatter(&mut scene, &sim);
    assert_eq!(
        before.len(),
        scene.scatter.len(),
        "the edit must not change the batch count"
    );
    for (b, a) in before.iter().zip(&scene.scatter) {
        assert_ne!(
            b.data.key(),
            a.data.key(),
            "the carried projection served a stale population — every instance \
             moved seven metres up"
        );
    }
}

/// **The stale hit a per-volume counter could not stop** (island wave I8a audit).
///
/// A cell deactivation destroys the `PcgVolume` and a reactivation builds a fresh
/// one **under the same guid**. `structures_gen` used to be a per-component
/// `wrapping_add(1)`, so the first write of both incarnations was `1` — same
/// guid, same stamp, different content, and a memo keyed on the pair would serve
/// the dead volume's instances.
///
/// This is that sequence exactly: project, replace the component with a NEW one
/// carrying a different population, project again. The stamp is drawn from a
/// process-global counter now, so it is a miss.
#[test]
fn a_reincarnated_volume_never_serves_the_old_population() {
    let mut sim = scatter_fixture();
    let mut scene = RenderScene::default();
    project_scatter(&mut scene, &sim);
    let before = scene.scatter.clone();
    let e = volume_of(&sim);
    let old_stamp = sim
        .world()
        .world()
        .get::<inf_ecs::components::PcgVolume>(e)
        .expect("the volume")
        .structures_gen;

    // The reincarnation: a brand-new component, same guid, different content.
    {
        let mut fresh = inf_ecs::components::PcgVolume::default();
        assert_eq!(fresh.structures_gen, 0, "a fresh component is unwritten");
        let (inst, solid, groups) = population(SCATTER_N, 13.0);
        fresh.set_population(
            inst,
            solid,
            groups,
            Vec::new(),
            Vec::new(),
            Default::default(),
            Vec::new(),
            Vec::new(),
        );
        assert_ne!(
            fresh.structures_gen, old_stamp,
            "a reincarnated volume took the dead one's stamp — the memo would \
             serve its instances"
        );
        sim.world_mut().world_mut().entity_mut(e).insert(fresh);
    }

    project_scatter(&mut scene, &sim);
    for (b, a) in before.iter().zip(&scene.scatter) {
        assert_ne!(
            b.data.key(),
            a.data.key(),
            "the memo served the destroyed volume's population to its successor"
        );
    }
}
