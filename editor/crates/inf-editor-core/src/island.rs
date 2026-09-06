//! The island's **level**, authored from its committed design (wave I7).
//!
//! # Why the level is committed and the terrain is not
//!
//! An island is 342 MB of terrain and 260 KB of design. The design is source;
//! the terrain is a build artifact of one machine, because the sampling step
//! goes through the projection modules this repository's portability law exempts
//! by name.
//!
//! So the level has to be authorable **without the terrain**, or it would be a
//! committed document only one machine could produce and nothing could check it
//! had not drifted. It is: `inf_island::read_design` opens five small committed
//! files — the coastline, the roads, the streams, the lakes, the biome masks —
//! and every number in the level comes from those, from the recipe, or from a
//! GUID derived from the island's own name.
//!
//! **Nothing here reads an elevation.** `the_level_is_authored_from_committed_
//! design_alone` is what says so.
//!
//! # "From the recipe" was not the same as "from a committed number"
//!
//! That sentence was true of every *file* and false of three numbers, and the
//! I7 CI-red is what taught the difference. `read_design` used to build the
//! geo-anchor by inverting the recipe's easting/northing through
//! `inf_gis::anchor_at` — i.e. `proj4rs`, i.e. the platform's libm — and the
//! three degrees that came back were serialized straight into this committed
//! `.inf_lvl`. macOS computed `origin_latitude_deg = 49.34307562364772` where
//! Windows had blessed `…773`: one ulp, one byte at offset 14 788, and
//! `committed_sample_matches_generators` red on one platform of three.
//!
//! The recipe now **states** its geodetic origin and `read_design` carries it
//! across untouched, so every byte of the anchor traces to a decimal in a
//! committed TOML. `crates/inf-island/tests/stated_anchor.rs` checks the stated
//! degrees against the projection, and
//! `crates/inf-island/tests/portable_math_law.rs` bans the anchor door from the
//! whole crate so it cannot come back.
//!
//! # One generator, two islands
//!
//! The full island and the CI-scale fixture are the same recipe format and the
//! same scene function. That is not tidiness — it is what makes the fixture a
//! gate: the level CI exercises is built by the code that builds the one that
//! ships, so a change that breaks the shipped level breaks the fixture's too.

use glam::DVec3;
use uuid::Uuid;

use crate::ipc::SpawnKind;
use crate::scene::serialize::{LevelSettings, PartitionSettings};
use crate::scene::SceneDoc;

/// Insert a bundle onto `guid`'s entity, dirtying the doc.
///
/// A second copy of `samples.rs`'s macro rather than a shared function, and the
/// reason is the facade rule: `macro_rules!` is module-scoped, and the only way
/// to write this once as a *function* is a `B: bevy_ecs::bundle::Bundle` bound —
/// which would make this crate name `bevy_ecs`, which is exactly what `inf-ecs`
/// exists to prevent. Eight lines of syntax against a ring violation.
macro_rules! insert {
    ($doc:expr, $guid:expr, $comp:expr $(,)?) => {{
        if let Some(e) = $doc.entity_of($guid) {
            $doc.world_mut().world_mut().entity_mut(e).insert($comp);
            $doc.world_mut().mark_dirty();
        }
    }};
}

/// How far above its resting height a car is authored (island wave VEH1a).
///
/// [`START_LIFT_M`]'s argument for a vehicle: the resting origin is *the springs
/// solved*, and solving them at author time would put a committed number in the
/// level that the first fixed step then re-derives. A hand's lift is a tenth of
/// a second of fall and lets the suspension settle it, which is what an author
/// does.
pub const CAR_LIFT_M: f64 = 0.15;

/// The paint a settlement's car wears.
///
/// A small fixed wheel rather than a random colour: the level is a committed
/// document and a colour drawn from a generator's own RNG state is a byte that
/// moves when anything upstream of it changes. Seven sites, five colours, and
/// the index is the site's own.
fn car_paint(site: usize) -> inf_ecs::math::Color {
    const PAINT: [inf_ecs::math::Color; 5] = [
        inf_ecs::math::Color::new(0.62, 0.10, 0.11, 1.0),
        inf_ecs::math::Color::new(0.12, 0.24, 0.52, 1.0),
        inf_ecs::math::Color::new(0.86, 0.86, 0.88, 1.0),
        inf_ecs::math::Color::new(0.16, 0.17, 0.19, 1.0),
        inf_ecs::math::Color::new(0.20, 0.45, 0.30, 1.0),
    ];
    PAINT[site % PAINT.len()]
}

/// How far above the ground a hero is spawned.
///
/// **Zero since wave FIX1**, and the metre it replaces is the whole of CERT1's
/// D-18. The old value's reason was that "a character placed exactly on the
/// surface is one the first ground snap has to resolve out of the floor" — which
/// is true, and is a description of the ground snap doing its job on a frame
/// nobody sees. What the lift bought instead was a fall the player watches:
/// **0.9883 m** of settle measured on the fixture and **0.9769 m** on the shipped
/// island, 0.45 s of it, every time the Play button is pressed.
///
/// It stays a named constant rather than becoming a literal `0.0` at the call
/// site because it is still the one place the question is answered, and because
/// `the_reported_start_is_the_one_the_level_spawns_at` pins its LINEARITY: a lift
/// of `x` must raise the start by exactly `x` and move nothing else.
pub const START_LIFT_M: f64 = 0.0;

/// How tall the island's hero is, metres — **the starter character's own
/// height**, not a number this file chose (SK1c).
///
/// The hero *is* `samples/starter-character`, so its capsule has to be the one
/// `edit_create_character` derives for that rig. Reading the spec is what makes
/// that true by construction rather than by two literals agreeing: a wizard
/// default that moves re-blesses the sample folder AND re-blesses both island
/// levels, in the same run, which is the loud version of a mismatch.
///
/// It was `1.8` while the hero was a bare capsule with nothing inside it. The
/// wizard's default is 1.75, and a 1.75 m body in a 1.8 m capsule floats 5 cm
/// off the ground it is standing on.
fn hero_height_m() -> f64 {
    crate::samples::starter_character_spec().params.height_m
}

/// How many reaches become real `WaterBody::River` entities.
///
/// See `IslandDesign::rivers` for the measurement behind it: a `RiverPath` holds
/// `segments × 16` frames and `WaterSurface::height_at` walks them, so binding
/// fifty reaches would put tens of thousands of frames behind every buoyancy
/// query. The rest keep their carved channels and are dry beds.
pub const MAX_RIVER_BODIES: usize = 10;

/// **How far the island's sun casts a shadow**, metres (wave CERT1, CP-B5).
///
/// `RenderSettingsRecord::default().shadows_max_distance` is 60 m, which on a
/// settlement of 100 m city blocks (`settlement::CITY_BLOCK_M`) means the
/// building on the far side of the street casts nothing at all. 250 m is two
/// and a half blocks — the depth the parity reference's daytime street frames
/// show shadow in.
///
/// Authored on THIS level rather than moved in the engine's default: a cascade
/// range is a cost every level would otherwise pay, and a level with a 20 m
/// ground plane has no use for it. Read by name by
/// `runtime/inf-player/tests/lit_stack.rs`, so the day someone re-blesses the
/// island without it the arm says which number went missing.
pub const ISLAND_SHADOW_DISTANCE_M: f32 = 250.0;

/// The partition cell an island streams in.
///
/// `DEFAULT_CELL_SIZE_M` (256 m) over a 7 168 m world is 28 × 28 = 784 cells,
/// which is the same lattice the terrain's own level-0 tiles sit on — so a cell
/// and a page activate together instead of at two different distances.
pub const ISLAND_CELL_SIZE_M: f64 = 256.0;

/// The activation radius. One cell, which is what P16's own default is; the
/// terrain's render cut is a separate and wider thing (IB-9).
pub const ISLAND_ACTIVATION_M: f64 = 256.0;

/// The prefetch margin.
pub const ISLAND_PREFETCH_M: f64 = 256.0;

/// The scatter cell the island's vegetation is evaluated on, world metres.
pub const ISLAND_SCATTER_CELL_M: f64 = 32.0;

/// Instances per square metre at density 1.0.
///
/// **This is the island-scale number and it is a budget, not a taste.** 40 km² of
/// land at the phase-18 sample's own 0.05 /m² would be two million instances
/// before a single mask ran. At 0.004 the biome-bound population over the
/// forested 38.5 % is ~620 000 candidates, which the P18.5 GPU scatter path and
/// the I3 draw bands are sized for; the CPU fallback's own ceiling
/// (`MAX_CPU_SCATTER_INSTANCES`, 65 536) is what a tier that cannot reach the
/// GPU path degrades to, nearest-first.
///
/// # Raised at TER2a, with the arithmetic that allowed it
///
/// 0.004 /m² is **one thing per 250 m²** — a fifteen-metre square with a single
/// tuft of grass in it. That number was set when a scatter kind was a bare
/// transform and the cost of being wrong was zero; TER2a gives the three kinds
/// real meshes, and at one per 250 m² an island covered in them still reads as
/// bare ground.
///
/// **The TER2a audit's correction to the sentence above.** The meshes are
/// authored, byte-locked and cooked, and **nothing draws them**: `push_scatter`
/// builds a `PrimMesh::Cube` tinted from a five-entry palette, `ScatterBatch`
/// has no mesh field, and `PcgKind::mesh` never survives evaluation into a
/// `PcgInstance` — see `island_gate::the_cover_meshes_are_shipped_and_are_not_yet_drawn`.
/// So what this raise actually multiplied, today, is the number of **placeholder
/// cubes** in the player's ~1.3 km²: 2 681 → 16 771 at the authored 0.7–1.6 m
/// scale range, about one every 9 m of resident ground. The number is the right
/// one for the day the upload lands, it is measured, and it is inside every
/// budget below — and until that day it is cubes. Reverting to 0.004 until then
/// is the other defensible answer and is named in the wave's carried list rather
/// than taken by the audit.
///
/// What bounds it is not the island's area but the **working set**, and the
/// working set is what the instrument measures. At 0.004 the shipped island
/// frame drew **2 681** scattered instances: the scatter evaluates only where
/// terrain is resident, which is `SIM_MARGIN_TILES` (2.0) of level-0 pages
/// around each observer — about 1.3 km², of which the scattering biomes are a
/// fraction. So the per-frame population scales with the density and nothing
/// else. **Measured at 0.02: 16 771 instances** of 32–128 triangles — about
/// 1.1 M triangles, against the 10 M-triangle gate P13 measured at 2.4 % cull
/// and the 15 k instances `phase19-town` already draws.
///
/// *The scaling was optimistic and the instrument is what said so.* A linear
/// scale from the 0.004 reading predicts 13 405 — **20 % below** the 16 771 the
/// frame actually drew, which is the measurement sitting **25 % above** the
/// prediction. A jittered per-cell scatter does not divide evenly. The
/// prediction is recorded beside the measurement rather than replaced by it,
/// because the lesson is the house one: an inference dressed as a measurement
/// is worse than no measurement.
///
/// The CPU fallback is what caps it: 16 771 is **25.6 %** of
/// `MAX_CPU_SCATTER_INSTANCES` (65 536), so a tier that cannot reach the GPU
/// scatter path still draws every instance rather than a nearest-first subset —
/// with room for three more raises of this size before the two tiers stop
/// drawing the same island. At 0.1 /m² they would.
///
/// *The arm is tighter than that sentence, deliberately* (TER2a audit). "Three
/// more raises" is the distance to the **real** ceiling, 65 536; the arm below
/// trips at a **third** of it (21 845), i.e. after roughly one more raise of this
/// size. A tripwire that only fires when the thing has already broken is not a
/// tripwire, so the two numbers are different on purpose — and are both written
/// down here so neither reads as the other.
///
/// **The honest bound this does not fix**: the scatter is evaluated on the
/// SIMULATION's resident set, and the renderer draws terrain far past it (the
/// clipmap's outer rings are pages the sim never asked for). So ground cover
/// stops at roughly 1.3 km² around the player and bare ground continues to the
/// horizon. Widening `SIM_MARGIN_TILES` would move it and would also widen every
/// physics query and every biome evaluation with it; the right fix is a scatter
/// residency of its own, and it is a wave rather than a constant.
pub const ISLAND_SCATTER_DENSITY: f64 = 0.02;

/// A stable GUID from the island's name and a salt, mirroring
/// `inf_island`'s own derivation so the level and the build agree about which
/// asset is which without either storing a table.
pub(crate) fn derived(name: &str, salt: &str) -> Uuid {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in salt.as_bytes().iter().chain(b"/").chain(name.as_bytes()) {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    let mut lo: u64 = 0x9e37_79b9_7f4a_7c15 ^ h;
    lo = lo.wrapping_mul(0xff51_afd7_ed55_8ccd);
    lo ^= lo >> 33;
    let mut bytes = [0u8; 16];
    bytes[..8].copy_from_slice(&h.to_be_bytes());
    bytes[8..].copy_from_slice(&lo.to_be_bytes());
    Uuid::from_bytes(bytes)
}

/// The island's sun.
pub fn sun_guid(name: &str) -> Uuid {
    derived(name, "island.sun")
}
/// The island's terrain entity.
pub fn terrain_entity_guid(name: &str) -> Uuid {
    derived(name, "island.terrain.entity")
}
/// The Outliner label one road-furniture entity carries (wave ROAD1).
fn road_part_label(part: inf_gis::RoadPart) -> &'static str {
    match part {
        inf_gis::RoadPart::Carriageway => "Roads",
        inf_gis::RoadPart::Kerb => "Kerbs and pavements",
        inf_gis::RoadPart::MarkingWhite => "Road markings",
        inf_gis::RoadPart::MarkingYellow => "Road markings (yellow)",
    }
}

/// **The surface one road-furniture group wears** (wave ROAD1).
///
/// # The kerb binds a `.inf_mat` and the paint does not, and that is a decision
///
/// Concrete is a *surface*: a person stands on it at a metre and its joints,
/// its float finish and its exposed aggregate are what say "laid" rather than
/// "rolled". So it is a real synthesised set (`GroundKind::Concrete`), bound by
/// GUID, tiling every 2 m through the same `uv_tiling_m` the asphalt uses.
///
/// Road paint is **not** a surface. It is a flat, near-lambertian film 100 mm
/// wide, and every texel of a map of it would be the same texel — so the paint
/// entities carry `Material::asset: None`, which `Material`'s own doc calls
/// "the scalars are the whole material, the permanent no-texture path". That is
/// 1.6 MB of committed bytes not spent, twice, on two constant colours.
///
/// The colours are **linear** and they are weathered rather than fresh: road
/// marking that has been driven over reads about 0.55 linear, not 1.0, and a
/// pure white line on a dark carriageway is the thing that makes a screenshot
/// look like a diagram. Roughness 0.55 — thermoplastic is smoother than the
/// aggregate around it, which is why a wet road's lines flare first.
fn road_part_material(part: inf_gis::RoadPart) -> inf_ecs::components::Material {
    use inf_ecs::math::Color;
    match part {
        inf_gis::RoadPart::Kerb | inf_gis::RoadPart::Carriageway => {
            let kind = if matches!(part, inf_gis::RoadPart::Kerb) {
                inf_material::ground::GroundKind::Concrete
            } else {
                inf_material::ground::GroundKind::Asphalt
            };
            let c = kind.base_color();
            inf_ecs::components::Material {
                base_color: Color::new(c[0], c[1], c[2], c[3]),
                roughness: kind.roughness(),
                metallic: 0.0,
                asset: Some(crate::ground::ground_material_guid(kind)),
                ..Default::default()
            }
        }
        inf_gis::RoadPart::MarkingWhite => inf_ecs::components::Material {
            base_color: Color::new(0.55, 0.55, 0.53, 1.0),
            roughness: 0.55,
            metallic: 0.0,
            ..Default::default()
        },
        inf_gis::RoadPart::MarkingYellow => inf_ecs::components::Material {
            base_color: Color::new(0.46, 0.31, 0.035, 1.0),
            roughness: 0.55,
            metallic: 0.0,
            ..Default::default()
        },
    }
}

/// The island's road-surface entity.
pub fn roads_entity_guid(name: &str) -> Uuid {
    derived(name, "island.roads.entity")
}
/// **One road-furniture entity** (wave ROAD1) — the kerbs and pavements, the
/// white paint or the yellow.
///
/// Three entities beside the carriageway, and not one, because an
/// `inf_ecs::Material` component binds **one** `.inf_mat`: a road that wears
/// asphalt, concrete, white paint and yellow paint is four entities however the
/// geometry is stored. Salted per part on `roads_entity_guid`'s own pattern.
pub fn road_part_entity_guid(name: &str, part: inf_gis::RoadPart) -> Uuid {
    match part {
        inf_gis::RoadPart::Carriageway => roads_entity_guid(name),
        inf_gis::RoadPart::Kerb => derived(name, "island.roads.kerb.entity"),
        inf_gis::RoadPart::MarkingWhite => derived(name, "island.roads.marking.white.entity"),
        inf_gis::RoadPart::MarkingYellow => derived(name, "island.roads.marking.yellow.entity"),
    }
}
/// The ocean.
pub fn ocean_guid(name: &str) -> Uuid {
    derived(name, "island.ocean")
}
/// The hero.
pub fn hero_guid(name: &str) -> Uuid {
    derived(name, "island.hero")
}
/// Lake `i`.
pub fn lake_guid(name: &str, i: usize) -> Uuid {
    derived(name, &format!("island.lake.{i}"))
}
/// River `i`.
pub fn river_guid(name: &str, i: usize) -> Uuid {
    derived(name, &format!("island.river.{i}"))
}
/// The scatter volume over site `i`'s quarter of the world.
pub fn cover_volume_guid(name: &str, i: usize) -> Uuid {
    derived(name, &format!("island.cover.{i}"))
}

/// The `.inf_pcg` the biome set binds — the island's ground cover.
///
/// A **document-only** envelope, exactly as `phase18-scatter`'s is: no graph, so
/// no grammar and no building passes, which is right because this is vegetation
/// and the settlements are wave I8's. The biome binding is what restricts it to
/// the biomes that scatter — `bind_document` rewrites the sampler as
/// `Multiply(Biome{id}, authored)` and re-salts the seed per biome, so one
/// document serves six biomes without six copies of it.
pub fn island_cover_document(seed: u64) -> inf_pcg::PcgDocument {
    use inf_pcg::{PcgKind, PcgRule, SamplerDef};
    let rule = PcgRule {
        name: "island-cover".into(),
        // Slope-limited: nothing grows on a 45-degree face, and the feather is
        // what keeps the treeline from being a drawn line.
        sampler: SamplerDef::Slope {
            min_deg: 0.0,
            max_deg: 34.0,
            feather_deg: 6.0,
        },
        scatter: inf_pcg::ScatterParams {
            seed,
            cell_size: ISLAND_SCATTER_CELL_M,
            base_density: ISLAND_SCATTER_DENSITY,
            jitter: 1.0,
            align_to_normal: false,
            scale_range: (0.7, 1.6),
            rotation: inf_pcg::RotationMode::RandomYaw,
            altitude_offset: 0.0,
        },
        // **The three kinds have meshes now** (TER2a, clause 5). All three
        // carried `mesh: None` — a bare transform, which the scatter evaluates,
        // the biome binding restricts, the residency pages and the frame counts,
        // and which draws nothing at all. `CoverKind::ALL`'s order IS this
        // palette's order: `kind_index` on a scattered instance indexes here.
        kinds: vec![
            PcgKind {
                mesh: Some(crate::cover::cover_mesh_guid(
                    crate::cover::CoverKind::GrassTuft,
                )),
                weight: 4.0,
            },
            PcgKind {
                mesh: Some(crate::cover::cover_mesh_guid(
                    crate::cover::CoverKind::Shrub,
                )),
                weight: 2.0,
            },
            PcgKind {
                mesh: Some(crate::cover::cover_mesh_guid(crate::cover::CoverKind::Rock)),
                weight: 1.0,
            },
        ],
    };
    inf_pcg::PcgDocument::single_layer("cover", vec![rule])
}

/// The `.inf_pcg` payload.
pub fn island_cover_payload(seed: u64) -> inf_pcg::PcgAssetPayload {
    inf_pcg::PcgAssetPayload::new(island_cover_document(seed))
}

/// **The island's four ground layers** (wave TER2a, clause 3).
///
/// The order is `inf_island::splat`'s and is load-bearing: the build writes a
/// per-sample `[u8; 4]` whose channels are grass, rock, forest floor and sand in
/// that order, and this is what those four channels *are*. Swapping two here
/// paints the beaches with rock and nothing would report it.
///
/// Each layer names a `.inf_mat` from the engine's committed ground library
/// (`samples/ground/`), which is what turns the terrain shader's four-layer
/// virtual-texture branch from a capability into a picture. The scalar
/// `albedo`/`roughness` beside it are **not decoration**: they are what a
/// surface shades with while its pages stream in, and what it shades with for
/// ever on an adapter with no virtual textures — so they are the ground sets'
/// own base colours rather than a second opinion about them.
///
/// `tex_scale` is metres per tile, and it is also what the procedural triplanar
/// grain is scaled by, so it is one number doing two jobs: at 2 m a 1 024²
/// albedo is 1.95 mm a texel and the grain that breaks up its tiling is a 2 m
/// feature. Both are what those surfaces want.
fn island_ground_layers() -> [inf_ecs::components::TerrainLayer; 4] {
    use inf_ecs::components::TerrainLayer;
    use inf_ecs::math::Color;
    use inf_material::ground::GroundKind;
    let layer = |kind: GroundKind| {
        let c = kind.base_color();
        TerrainLayer {
            albedo: Color::new(c[0], c[1], c[2], 1.0),
            roughness: f64::from(kind.roughness()),
            tex_scale: kind.tex_scale_m(),
            material: Some(crate::ground::ground_material_guid(kind)),
        }
    };
    [
        layer(GroundKind::Grass),
        layer(GroundKind::Rock),
        layer(GroundKind::ForestFloor),
        layer(GroundKind::Sand),
    ]
}

/// Author the island's level from its committed design.
pub fn island_scene(design: &inf_island::IslandDesign) -> SceneDoc {
    use inf_ecs::components::{
        AlwaysLoaded, AudioListener, Light, LightKind, MeshRef, PcgVolume, SkyAtmosphere, Spline,
        SplineInterp, StreamingSource, Terrain, TimeOfDay, Transform, WaterBody, WaterKind,
    };
    use inf_ecs::math::{Color, Vec2d, Vec3d};

    let name = design.recipe.name.as_str();
    let mut doc = SceneDoc::new();
    doc.set_title(name);

    // **Where on Earth the world is.** The sky reads its latitude from this, so a
    // shadow at noon falls the way it falls at 49 N — which is also what pins the
    // world frame: +X east, +Y up, -Z north.
    doc.set_geo(design.anchor.clone());

    // **THE LIT STACK, AUTHORED** (wave CERT1, CP-A1/CP-B5). This is the level
    // the application boots on and the level the parity certification is
    // written about, and a level that authors no render block ships with
    // shadows, GI, VSM, TAA, SSAO and bloom ALL OFF. The showcase played unlit
    // for every wave up to this one.
    //
    // `shadows_max_distance` is the ONE knob overridden beyond
    // `lit_showcase()`: the record's default is 60 m, which on a street of
    // 100 m blocks means the building across the road casts nothing. 250 m is
    // two and a half city blocks — the depth the reference frames' daytime
    // street shots actually show shadow in — and it is a per-level authored
    // number, not an engine default, so no other level pays for it.
    doc.set_settings(LevelSettings {
        partition: PartitionSettings {
            enabled: true,
            cell_size_m: ISLAND_CELL_SIZE_M,
            activation_radius_m: ISLAND_ACTIVATION_M,
            prefetch_margin_m: ISLAND_PREFETCH_M,
        },
        render: crate::scene::serialize::RenderSettingsRecord {
            shadows_max_distance: ISLAND_SHADOW_DISTANCE_M,
            ..crate::scene::serialize::RenderSettingsRecord::lit_showcase()
        },
        ..LevelSettings::default()
    });

    // ── the sky ───────────────────────────────────────────────────────────────
    let sun = sun_guid(name);
    doc.create_with_guid(sun, SpawnKind::Empty, "Sun", None);
    insert!(
        doc,
        sun,
        Transform {
            translation: Vec3d::ZERO,
            rotation: Vec3d::new(-46.0, -28.0, 0.0),
            scale: Vec3d::ONE,
        },
    );
    insert!(
        doc,
        sun,
        Light {
            kind: LightKind::Directional,
            color: Color::WHITE,
            intensity: 3.2,
            ..Default::default()
        },
    );
    insert!(
        doc,
        sun,
        TimeOfDay {
            seconds: 10.5 * 3600.0,
            rate: ISLAND_CLOCK_RATE,
            ..TimeOfDay::default()
        },
    );
    insert!(doc, sun, SkyAtmosphere::default());
    insert!(doc, sun, AlwaysLoaded);

    // ── the ground ────────────────────────────────────────────────────────────
    //
    // Streamed: the terrain ships NO tiles in the level, only the `.inf_terrain`
    // GUID, which is what keeps a 342 MB world out of a 30 KB document. And
    // `AlwaysLoaded`, because a Terrain occupies space and a partitioner would
    // otherwise bin the whole heightfield into the one cell holding its origin —
    // and the ground would despawn under the player.
    // **IDENTITY, AND THE OFFSET IT REPLACES WAS THE ISLAND MOVED HALF A WORLD.**
    //
    // Every other terrain in this repository is built with level-0 tile
    // coordinates starting at `(0, 0)`, so its entity is translated to
    // `-span/2` to centre the grid on the world origin
    // (`island_frame_terrain_origin` is the pattern). **`IslandGrid` does not
    // work that way**: `tile0 = -(tiles / 2)`, so the `.inf_terrain`'s own tile
    // indices are already centred and its sample frame **is** the world frame —
    // which is exactly what the whole build assumes (`CoarseHeights::of(&data,
    // min, max, …)`, the grade audit's `data.height_at(p)`, the channel carve,
    // the biome stamp).
    //
    // Translating the entity as well applied that centring twice. Measured on
    // the fixture, through the shipped host's own `terrain.height_at` seam: the
    // hero's start read **0.000 m of unauthored ground where the design puts
    // 129.916 m**, and `(0, 0)` read 80.000 m off a page 768 m away. On the
    // shipped island the displacement is 3 584 m on both axes, so half the
    // terrain sat outside the world. Nothing caught it because the island gate
    // never attached the terrain streamer, so the simulation's working set was
    // empty and every query answered the unauthored default — the two hosts
    // agreed about no ground at all.
    let terrain_guid = terrain_entity_guid(name);
    doc.create_with_guid(terrain_guid, SpawnKind::Empty, "Ground", None);
    insert!(doc, terrain_guid, Transform::IDENTITY);
    {
        let mut t = Terrain::configured(
            design.recipe.grid.tile_resolution,
            design.recipe.grid.meters_per_sample,
        );
        t.asset = Some(inf_island::terrain_guid(name));
        t.biome_set = Some(inf_island::biome_set_guid(name));
        t.layers = island_ground_layers();
        debug_assert!(t.data.is_empty(), "a streamed terrain ships no tiles");
        insert!(doc, terrain_guid, t);
    }
    insert!(doc, terrain_guid, AlwaysLoaded);

    // ── the roads, as one drawn surface ───────────────────────────────────────
    //
    // No collider, by IB-4's ruling: a road conforms to the terrain, whose
    // heightfield collider already answers there, so a per-segment trimesh would
    // be 3.63 ms a step for nothing a body can reach.
    let roads = roads_entity_guid(name);
    doc.create_with_guid(roads, SpawnKind::Empty, "Roads", None);
    insert!(doc, roads, Transform::IDENTITY);
    insert!(
        doc,
        roads,
        MeshRef {
            asset: Some(inf_island::road_mesh_guid(name)),
            ..Default::default()
        },
    );
    // **THE ROAD'S SURFACE** (wave ASSET0, clause 0). Until this line the
    // `Roads` entity carried a `MeshRef` and nothing else, and the EDIT1 audit
    // measured what that meant on the shipped editor: the street reads
    // (238.9, 239.1, 238.0) lit and **(225.7, 225.7, 225.7) UNLIT** -- 0.8
    // linear through the tonemap, which is `Material::default().base_color`,
    // which is what the projection gives a mesh with no `Material` component at
    // all. The brightest surface in the frame was the engine's debug grey, in
    // BOTH hosts, and EDIT1 made the editor open standing on it.
    //
    // The scalars are carried as well as the `.inf_mat` id, and they are the
    // material's own (`GroundKind::Asphalt::base_color` / `roughness`) rather
    // than a second opinion: `Material::asset` says *these scalars came from
    // that material*, and a host with no virtual textures -- or one whose pages
    // have not arrived yet -- shades off exactly them. Two numbers that
    // disagreed would be a road that changed colour as it streamed.
    {
        use inf_material::ground::GroundKind::Asphalt;
        let c = Asphalt.base_color();
        insert!(
            doc,
            roads,
            inf_ecs::components::Material {
                base_color: Color::new(c[0], c[1], c[2], c[3]),
                roughness: Asphalt.roughness(),
                metallic: 0.0,
                asset: Some(crate::ground::ground_material_guid(Asphalt)),
                ..Default::default()
            },
        );
    }
    insert!(doc, roads, AlwaysLoaded);

    // ── the kerbs, the pavements and the paint ────────────────────────────────
    //
    // **CERT1's CP-B3, closed** (wave ROAD1): *"the pavement is never drawn
    // either — `PAVEMENT_M = 2.0` is a nav ring of eight nodes; there is no kerb
    // geometry, no pavement mesh and no road marking."* Three entities, because
    // a `Material` binds one `.inf_mat` and a road wears four surfaces.
    //
    // No collider, for the `Roads` entity's own reason and one more: a kerb is
    // 150 mm and the terrain's heightfield collider answers under it, so a
    // trimesh here would be a second surface a hand's breadth above the first.
    // The step a pedestrian takes up a kerb is a step they take through it
    // today, and that is the honest limit — carried, not smuggled.
    for part in inf_gis::FURNITURE_PARTS {
        let g = road_part_entity_guid(name, part);
        doc.create_with_guid(g, SpawnKind::Empty, road_part_label(part), None);
        insert!(doc, g, Transform::IDENTITY);
        insert!(
            doc,
            g,
            MeshRef {
                asset: Some(inf_island::road_part_mesh_guid(name, part)),
                ..Default::default()
            },
        );
        insert!(doc, g, road_part_material(part));
        insert!(doc, g, AlwaysLoaded);
    }

    // ── the sea ───────────────────────────────────────────────────────────────
    //
    // One body. `WaterSurface::Ocean` is unbounded in the simulation and the
    // renderer tessellates a patch around the camera, so an island needs exactly
    // one however long its coastline is.
    let ocean = ocean_guid(name);
    doc.create_with_guid(ocean, SpawnKind::Empty, "Ocean", None);
    insert!(
        doc,
        ocean,
        WaterBody {
            kind: WaterKind::Ocean,
            level_m: design.recipe.sea.level_m,
            wave_amplitude_m: 0.6,
            wave_length_m: 34.0,
            wave_steepness: 0.42,
            wave_count: 5,
            wave_seed: 0x0015_1A4D,
            // Body-local wind: a coastline's sea state must not depend on where
            // the weather blend happens to be when a trace is taken.
            wind_from_weather: false,
            wind_x: 6.5,
            wind_z: -2.5,
            ..WaterBody::default()
        },
    );
    insert!(doc, ocean, AlwaysLoaded);

    // ── the lakes ─────────────────────────────────────────────────────────────
    for (i, l) in design.network.lakes.iter().enumerate() {
        let g = lake_guid(name, i);
        doc.create_with_guid(g, SpawnKind::Empty, &format!("Lake {i}"), None);
        insert!(
            doc,
            g,
            Transform::from_translation(DVec3::new(l.centre.x, l.level_m, l.centre.y)),
        );
        insert!(
            doc,
            g,
            WaterBody::lake(l.level_m, Vec2d::new(l.half_extent.x, l.half_extent.y)),
        );
    }

    // ── the rivers ────────────────────────────────────────────────────────────
    //
    // The centreline is the `Spline` on the SAME entity — P20.1's composition
    // rule, so there is nothing to resolve and nothing to dangle — authored in
    // world space under an identity transform.
    for (i, s) in design.rivers(MAX_RIVER_BODIES).into_iter().enumerate() {
        let g = river_guid(name, i);
        doc.create_with_guid(g, SpawnKind::Empty, &format!("River {i}"), None);
        insert!(doc, g, Transform::from_translation(DVec3::ZERO));
        let w = s.width_m();
        let d = s.depth_m();
        insert!(
            doc,
            g,
            WaterBody {
                river_width_start_m: (w * 0.7).max(1.0),
                river_width_end_m: w,
                river_depth_start_m: (d * 0.7).max(0.3),
                river_depth_end_m: d,
                // **The shore band is sized to THIS reach** (wave ROAD1,
                // clause 3), and it has to be, now that a river's drawn column
                // is its own modelled bed: `water.wgsl` takes a river's column
                // FROM `depth · (1 − bank²)` and not from the depth buffer at
                // all (`select`, not `max` — the arm one crate over says why),
                // so a band wider than
                // the reach is deep would leave the whole creek half
                // transparent — a 0.35 m stream against the 1.2 m default
                // reaches `smoothstep(0, 1.2, 0.35) = 0.20` **on its own
                // centreline**.
                //
                // A third of the depth puts the band in the outer third of the
                // ribbon, which is where a bank is, and the clamp keeps it a
                // band rather than a hard edge on a trickle or a smear on the
                // deepest reach.
                shore_fade_m: (d * 0.35).clamp(0.12, 1.2),
                ..WaterBody::river(w, d, 0.5 + 2.0 * s.grade().clamp(0.0, 0.5))
            },
        );
        insert!(
            doc,
            g,
            Spline {
                points: s.points.iter().map(|p| Vec3d::new(p.x, p.y, p.z)).collect(),
                closed: false,
                interp: SplineInterp::CatmullRom,
            },
        );
    }

    // ── the vegetation ────────────────────────────────────────────────────────
    //
    // The binding is on the `.inf_biomes` set, which both hosts resolve through
    // `inf_pcg::BiomeBinding::from_set`, and it evaluates over the **terrain's
    // own bounds** — so there is no `PcgVolume` here and no half-extent to keep
    // in step with the world's. That is the one door: the level names a biome
    // set, the set names a graph, and the graph is masked by the painted ids.

    // ── the settlements ───────────────────────────────────────────────────────
    //
    // **Wave I8a: the seven pads stop being terraces.** One `PcgVolume` per block
    // — a centre, an axis-aligned half-extent, a seed and the GUID of the zone
    // document its archetype names. The plan is
    // `crate::settlement::settlements`, which is a pure function of the same
    // committed design every number above comes from.
    //
    // **Not `AlwaysLoaded`, deliberately.** A settlement block is exactly what
    // the partition is for: 172 volumes over 51 km² is a thousand buildings the
    // simulation must not hold at once, and `PcgVolume` evaluation runs on cell
    // activation (`cell_stream::reconcile`) as well as at load. The blocks
    // therefore stream with their cells, which is the whole reason the level can
    // carry them at all.
    //
    // `draw_distance` is left at its default: the I3 structure bands
    // (`DEFAULT_STRUCTURE_LOD_M`, 96 m) are what decide whether a building draws
    // its parts or its shell, and a second per-volume distance cut on top of
    // them would be a second authority on the same question.
    let plans = crate::settlement::settlements(design);
    for plan in &plans {
        for b in &plan.blocks {
            let g = crate::settlement::block_guid(name, b.site, b.col, b.row);
            doc.create_with_guid(
                g,
                SpawnKind::Empty,
                &format!("{} {} {},{}", plan.name, b.archetype.name(), b.col, b.row),
                None,
            );
            insert!(
                doc,
                g,
                Transform {
                    translation: Vec3d::new(b.centre.x, 0.0, b.centre.y),
                    rotation: Vec3d::ZERO,
                    scale: Vec3d::ONE,
                },
            );
            insert!(
                doc,
                g,
                PcgVolume {
                    graph: Some(crate::settlement::zone_guid(b.archetype)),
                    extent: Vec2d::new(b.half.x, b.half.y),
                    seed: b.seed,
                    ..Default::default()
                },
            );
        }
    }

    // ── the fleet ─────────────────────────────────────────────────────────────
    //
    // **One car where the circuit passes each settlement** (island wave VEH1a).
    // Wave I7 shipped 33.74 km of graded, audited road and left open item 15 on
    // the record: *"THE CIRCUIT IS DRAWN AND AUDITED, AND NOTHING DRIVES IT"* —
    // and the reason it gave was content, *"a vehicle needs a chassis mesh and a
    // tuning block that would be the fourth committed asset in a folder whose
    // whole argument is that it is small."*
    //
    // It needs neither. The body is a union of built-in primitives derived from
    // one Ring-0 table (`inf_ecs::vehicle::VehicleBody`), so **no mesh file is
    // committed**; the tuning is a `VehicleClass`, which the scene has carried
    // since v25; and the fleet is a TOML catalogue this crate holds as a `&str`
    // and reads at generation time, on the `GAMEPLAY_ITEMS_TOML` precedent, so
    // **no schema moves and no asset kind is added**.
    //
    // WHERE: the nearest *planned route vertex* to each settlement's centre —
    // which is the point at which the drivable circuit passes the town, and the
    // only place on this island where a committed number says what the ground
    // height is (`inf_island::roads::nearest_route_vertex`, the same door
    // `player_start` uses). Nothing here reads an elevation, and the level stays
    // authorable without the terrain.
    //
    // NOT `AlwaysLoaded`: a car streams with its partition cell exactly as a
    // settlement block does, so a level with seven of them holds however many
    // are near the simulation and no more.
    let fleet = crate::vehicle::island_vehicles();
    for (n, plan) in plans.iter().enumerate() {
        // **The whole fleet reaches the island** (island wave VEH2a). A city
        // parks the two road cars, a town the working vehicles and the wagon,
        // and everything else the pickup — chosen by the site's own kind and its
        // position in the settlement order rather than by a hash, so a recipe
        // that adds a town changes exactly one car.
        //
        // Five rows and seven settlements, and no row is left in the catalogue:
        // a fleet nothing drives is a table, and this wave's whole argument is
        // that the five rows drive differently.
        let Some(def) = fleet.get(island_vehicle_id(plan.kind, n)) else {
            continue;
        };
        let Some((v, dir)) = inf_island::nearest_route_vertex(&design.routes, plan.centre) else {
            continue;
        };
        let guid = derived(name, &format!("island.car.{}", plan.site));
        // Nose along the road, and a hand's lift over it: the suspension settles
        // the rest on the first step, which is what an author does rather than
        // solving the spring at author time.
        //
        // `patan2_64` and not `f64::atan2` — this yaw is serialized into a
        // committed `.inf_lvl`, which is the P14 law's first class exactly
        // (`portable_math_law` scans every `.rs` under this crate's `src/`).
        // A yaw of θ about `+Y` takes `+Z` to `(sin θ, 0, cos θ)`, so facing a
        // direction `d` is `atan2(d.x, d.z)`.
        let yaw_deg = inf_math::patan2_64(dir.x, dir.y).to_degrees();
        crate::vehicle::spawn_vehicle(
            &mut doc,
            guid,
            def,
            crate::vehicle::VehicleSpawn {
                name: &format!("{} Car", plan.name),
                at: DVec3::new(
                    v.x,
                    crate::vehicle::resting_origin_y(def, v.y) + CAR_LIFT_M,
                    v.z,
                ),
                yaw_deg,
                paint: car_paint(plan.site),
                clip: None,
                livery: None,
                engine_voice: true,
            },
        );
    }

    // ── the emergency fleet ───────────────────────────────────────────────────
    //
    // **Parked on the apron at each institution's own street frontage** (wave
    // EMS1), one row per entry of `station_fleet`.
    //
    // WHERE, honestly: not inside the bay. A station's `ApparatusBay` is a room
    // of a building this generator does not build — the block is a `PcgVolume`
    // and its buildings are derived at load — and the one thing that would place
    // a car inside it is a ground height, which this crate is forbidden to read
    // (`the_settlement_generator_is_authored_from_committed_design_alone` bans
    // `inf_terrain::`). So the fleet goes where every other authored vehicle on
    // this island goes: the nearest planned route vertex, which is the only
    // place a committed number says what the ground is. That is the apron
    // outside the doors, which is where an appliance stands anyway, and it is
    // stated here rather than discovered by somebody looking for a fire engine
    // in a garage.
    //
    // The probe is the INSTITUTION'S OWN BLOCK CENTRE, not the settlement's, so
    // a city's four institutions are four separate placements; the vehicles of
    // one station are then spaced along the road by `EMS_PARK_PITCH_M`, starting
    // one pitch PAST the vertex (see below), so a two-car station does not stack
    // two cars on one point and neither sits on the settlement's own car.
    let fleet_defs = crate::vehicle::island_vehicles();
    for plan in &plans {
        // **ONE SETTLEMENT, ONE APRON REGISTER** (EMS1 audit) — the third act of
        // the parked-fleet trilogy, one level up from the first two.
        //
        // The probe is per BLOCK, and two institution blocks of one city are
        // near each other and near the same road: measured on the shipped
        // island, **Eastgate's fire hall and police station probe route
        // vertices 12.6 m apart with near-parallel directions**, and the
        // appliance and the second cruiser came out **3.40 m apart** — an
        // 8.85-tonne, 7.8 m hull with a saloon 3 m inside its tail, and 0.89 m
        // of a 2.21 m vertical overlap as well. That is exactly the defect this
        // wave already paid for twice, and it was invisible because the arm
        // that catches it reset its list per station.
        //
        // So a settlement's institutions share one register, and a vehicle
        // whose spot is not clear of one already parked **steps one more pitch
        // along its own apron** until it is. Deterministic (block order, then
        // `k`), portable (a length against a constant), and it moves nothing
        // that was already clear — which is every station on the island except
        // Eastgate's.
        let mut parked: Vec<DVec3> = Vec::new();
        for b in plan.blocks.iter().filter(|b| b.archetype.is_institution()) {
            let fleet = station_fleet(b.archetype);
            if fleet.is_empty() {
                continue;
            }
            let Some((v, dir)) = inf_island::nearest_route_vertex(&design.routes, b.centre) else {
                continue;
            };
            let yaw_deg = inf_math::patan2_64(dir.x, dir.y).to_degrees();
            // **The apron is the side the station is on.** The perpendicular of
            // the road direction in XZ, signed by which way the block lies —
            // a dot product and a sign, so no trigonometry reaches a committed
            // transform (the P14 law).
            let perp = glam::DVec2::new(-dir.y, dir.x);
            let side = if perp.dot(b.centre - glam::DVec2::new(v.x, v.z)) >= 0.0 {
                1.0
            } else {
                -1.0
            };
            let apron = perp * (side * EMS_APRON_OFFSET_M);
            // **PAST the vertex, never ON it**, and that is a bug this wave paid
            // for. The first cut centred the fleet on the vertex — and a
            // settlement's own civilian car is parked EXACTLY there, so on a
            // small town whose institution block resolves to the settlement's
            // own route vertex an appliance materialized inside a saloon. The
            // island gate found it the way an interpenetration is always found:
            // not as an overlap, but as a hero who could no longer stand up
            // beside the car it was trying to drive.
            //
            // So the run starts one pitch along and grows from there, which
            // clears the longest vehicle in the catalogue (a 7.8 m appliance)
            // against the longest thing already at the vertex with 1.1 m to
            // spare.
            for (k, id) in fleet.iter().enumerate() {
                // The catalogue is parsed once for the whole pass: the emergency
                // rows live in the same table the civilian ones do.
                let Some(def) = fleet_defs.get(id).copied() else {
                    continue;
                };
                // Derived from the STEP, never accumulated — the P17.4
                // exact-linear rule, which is what keeps two hosts agreeing
                // about a metre. The step starts at `k + 1` and rises only to
                // clear a vehicle this settlement has already parked.
                let spot = |step: usize| {
                    let along = step as f64 * EMS_PARK_PITCH_M;
                    DVec3::new(
                        v.x + dir.x * along + apron.x,
                        crate::vehicle::resting_origin_y(&def, v.y) + CAR_LIFT_M,
                        v.z + dir.y * along + apron.y,
                    )
                };
                let mut at = spot(k + 1);
                for step in k + 2..k + 2 + EMS_PARK_MAX_SHUNT {
                    if parked
                        .iter()
                        .all(|p| (*p - at).length() >= EMS_PARK_PITCH_M)
                    {
                        break;
                    }
                    at = spot(step);
                }
                parked.push(at);
                let guid = derived(
                    name,
                    &format!("island.ems.{}.{}.{}.{k}.{id}", b.site, b.col, b.row),
                );
                crate::vehicle::spawn_vehicle(
                    &mut doc,
                    guid,
                    &def,
                    crate::vehicle::VehicleSpawn {
                        name: &format!("{} {}", plan.name, id),
                        at,
                        yaw_deg,
                        // The livery paints every part it names, so the base
                        // paint only reaches a part the table forgot — and a
                        // white one would make that omission invisible.
                        paint: inf_ecs::math::Color::new(0.35, 0.36, 0.38, 1.0),
                        clip: None,
                        livery: crate::vehicle::island_vehicle_livery(id),
                        // **A parked appliance does not idle.** The emitter
                        // does not follow a car yet (VEH2a's carried item 5),
                        // so seventeen of these across the island would be
                        // seventeen stationary engine loops in a bounded audio
                        // log — which the drive gate caught as a second `Play`
                        // in a stream whose claim is one voice per car. The
                        // fleet gets its voice when EMS2 makes it drive.
                        engine_voice: false,
                    },
                );
            }
        }
    }

    // -- SEA AND AIR (wave VEH2c) ---------------------------------------------
    //
    // **The first two vehicles on this island that are not road vehicles**, and
    // the whole of what they need from this generator is a PLACE. Both ride the
    // same Path A the fleet above does -- the committed catalogue, spawned by
    // `spawn_vehicle` -- and neither reads an elevation, which is the ban this
    // module is held to (`the_settlement_generator_is_authored_from_committed_
    // design_alone` forbids `inf_terrain::`).
    //
    // THE LAUNCH: at the harbour, which is the sea off the city called Harbour
    // City. Its height needs no terrain at all -- a boat floats at the SURFACE,
    // and the recipe states where that is (`[sea] level_m`). Its position is the
    // committed coast ring's nearest vertex to the city, pushed seaward: the
    // nearest shore point to a settlement is on the shore that settlement faces,
    // so continuing away from the city goes out to sea rather than inland. That
    // is one normalize and no elevation.
    //
    // THE HELICOPTER AND ITS PAD: at the same city's police station apron, on
    // the emergency fleet's own door (`nearest_route_vertex` from the
    // institution's block centre), far enough along the road to clear the
    // cruisers parked there. The pad is a drawn slab and a static collider so a
    // machine can stand on it and a player can walk onto it.
    if let Some(plan) = plans.iter().find(|p| p.name == HARBOUR_CITY) {
        let fleet = crate::vehicle::island_vehicles();
        let sea_y = design.recipe.sea.level_m;

        // -- the launch --------------------------------------------------------
        if let Some(def) = fleet.get("launch") {
            let mut best: Option<(f64, glam::DVec2)> = None;
            for ring in &design.coast {
                for v in ring {
                    let d = (*v - plan.centre).length_squared();
                    if best.is_none_or(|(bd, _)| d < bd) {
                        best = Some((d, *v));
                    }
                }
            }
            if let Some((_, shore)) = best {
                let seaward = (shore - plan.centre).normalize_or_zero();
                if seaward != glam::DVec2::ZERO {
                    let at = shore + seaward * HARBOUR_OFFSHORE_M;
                    // Bow toward the open sea, which is the way a boat is left
                    // on a mooring. `patan2_64`, never `f64::atan2`: this yaw is
                    // serialized into a committed level (the P14 law).
                    let yaw_deg = inf_math::patan2_64(seaward.x, seaward.y).to_degrees();
                    crate::vehicle::spawn_vehicle(
                        &mut doc,
                        derived(name, "island.launch"),
                        def,
                        crate::vehicle::VehicleSpawn {
                            name: "Harbour Launch",
                            at: DVec3::new(
                                at.x,
                                crate::vehicle::floating_origin_y(def, sea_y),
                                at.y,
                            ),
                            yaw_deg,
                            paint: inf_ecs::math::Color::new(0.86, 0.87, 0.88, 1.0),
                            clip: None,
                            livery: None,
                            // The boat is the FIRST vehicle on this island whose
                            // engine can follow it: `AudioCommand::SetPosition`
                            // arrived at EMS2 and a moving emitter is real now,
                            // which is VEH2a's carried item 5 closed for the
                            // player's own craft. Traffic's mute ruling is about
                            // seventeen parked cars and does not apply here.
                            engine_voice: true,
                        },
                    );
                }
            }
        }

        // -- the pad, and the machine on it ------------------------------------
        let station = plan
            .blocks
            .iter()
            .find(|b| b.archetype == inf_pcg::ArchetypeId::PoliceStation);
        if let (Some(b), Some(def)) = (station, fleet.get("chopper")) {
            if let Some((v, dir)) = inf_island::nearest_route_vertex(&design.routes, b.centre) {
                let yaw_deg = inf_math::patan2_64(dir.x, dir.y).to_degrees();
                // The same apron the cruisers use, on the same side, and PAST
                // them: `station_fleet` parks three, one pitch apart, starting
                // one pitch along -- so a pad at `HELIPAD_SLOT` pitches clears
                // the last of them by a whole pitch.
                let perp = glam::DVec2::new(-dir.y, dir.x);
                let side = if perp.dot(b.centre - glam::DVec2::new(v.x, v.z)) >= 0.0 {
                    1.0
                } else {
                    -1.0
                };
                let apron = perp * (side * EMS_APRON_OFFSET_M);
                let along = HELIPAD_SLOT as f64 * EMS_PARK_PITCH_M;
                let at = DVec3::new(
                    v.x + dir.x * along + apron.x,
                    v.y,
                    v.z + dir.y * along + apron.y,
                );

                // The pad: a drawn slab with a real collider, so it is something
                // a machine stands on and a character can walk onto rather than
                // a decal. Half a metre of it is buried, which is what stops a
                // 12 cm lip being a kerb the hero trips over.
                let pad = derived(name, "island.helipad");
                doc.create_with_guid(pad, SpawnKind::Empty, "Helipad", None);
                insert!(
                    doc,
                    pad,
                    Transform {
                        // The slab's CENTRE, so that `HELIPAD_LIP_M` of it stands
                        // proud of the apron and `HELIPAD_BURY_M` is under it.
                        translation: Vec3d::new(at.x, at.y - HELIPAD_BURY_M, at.z),
                        rotation: Vec3d::new(0.0, yaw_deg, 0.0),
                        scale: Vec3d::ONE,
                    },
                );
                insert!(
                    doc,
                    pad,
                    inf_ecs::components::RigidBody3D {
                        kind: inf_ecs::components::BodyKind3D::Static,
                        ..Default::default()
                    },
                );
                insert!(
                    doc,
                    pad,
                    inf_ecs::components::Collider3D {
                        shape_kind: inf_ecs::components::ColliderShape3DKind::Box,
                        half_extents: Vec3d::new(
                            HELIPAD_RADIUS_M,
                            HELIPAD_HALF_M,
                            HELIPAD_RADIUS_M
                        ),
                        friction: 0.8,
                        ..Default::default()
                    },
                );
                insert!(
                    doc,
                    pad,
                    inf_ecs::components::MeshRef {
                        primitive: inf_ecs::components::Primitive::Cylinder,
                        asset: None,
                    },
                );
                insert!(
                    doc,
                    pad,
                    inf_ecs::components::Material {
                        base_color: inf_ecs::math::Color::new(0.13, 0.14, 0.15, 1.0),
                        metallic: 0.0,
                        roughness: 0.95,
                        ..Default::default()
                    },
                );

                crate::vehicle::spawn_vehicle(
                    &mut doc,
                    derived(name, "island.chopper"),
                    def,
                    crate::vehicle::VehicleSpawn {
                        name: "Light Helicopter",
                        at: DVec3::new(
                            at.x,
                            crate::vehicle::resting_origin_y(def, at.y + HELIPAD_LIP_M),
                            at.z,
                        ),
                        yaw_deg,
                        paint: inf_ecs::math::Color::new(0.22, 0.30, 0.42, 1.0),
                        clip: None,
                        // **No livery, and therefore no service** (wave VEH2c).
                        // `dispatch::unit_kind_of` recognises a unit off a
                        // bloomed `light_bar` child, and a helicopter wearing one
                        // would be claimed by the dispatcher and sent to drive to
                        // an incident on a road it has no wheels for. Making it a
                        // police unit is a real change to `drive_intent`, not a
                        // paint job, and it is priced in this wave's ledger
                        // rather than taken by accident.
                        livery: None,
                        engine_voice: true,
                    },
                );
            }
        }
    }

    // ── the hero ──────────────────────────────────────────────────────────────
    //
    // **It is the starter character** (SK1c). This used to be forty lines of
    // hand-rolled components ending in `AnimStateMachine { sm: None }` and no
    // `SkeletalMesh` — a capsule that walked, with nothing to draw and nothing to
    // pose — because the one door that knows how to build a character
    // (`SceneDoc::edit_create_character`) minted its own entity GUID and the
    // island derives every one of its own. That door takes a GUID now, so the
    // island spawns a character through the same code path the New Character
    // wizard does, and the assets it names are the ones
    // `samples/starter-character` commits.
    //
    // The assets reach a built project through the recipe's `[content]` list
    // (`inf_island::write_content`), so this crate names a GUID and nothing else:
    // no new crate edge, and the island's own generator still knows nothing about
    // `inf-anim`.
    let ids = crate::samples::starter_character_ids();
    let asset = |id: Option<inf_asset::AssetId>| id.expect("every starter id is fixed").0;
    let feet = design.start(START_LIFT_M);
    let hero = hero_guid(name);
    doc.edit_create_character_with_guid(
        hero,
        "Hero",
        asset(ids.skeleton),
        asset(ids.mesh),
        asset(ids.machine),
        Some(crate::scene::doc::CharacterSkin::from_material(
            asset(ids.material),
            &crate::character::starter_skin_material(),
        )),
        feet,
        Some(asset(ids.actor)),
        hero_height_m(),
    );
    // **Both of these are outside that door, and both are load-bearing.**
    // `StreamingSource` is the partition's activation anchor AND the I3 collider
    // band's — the two cannot disagree about where the simulation is because they
    // read the same component — and `AlwaysLoaded` keeps the hero resident in its
    // own cell. `edit_create_character` deliberately inserts neither: a character
    // is not necessarily a streaming anchor, and putting that opinion in the
    // wizard's door would put it on every character anybody ever spawns.
    insert!(doc, hero, StreamingSource { radius_m: 256.0 });
    insert!(doc, hero, AlwaysLoaded);
    // **…and the EAR** (wave VEN1b). A third component outside the wizard's
    // door, for the two above it's own reason and one of its own.
    //
    // `active_listener` takes the first active `AudioListener` in `Guid` order
    // and, finding none, leaves the engine's listener at its default pose —
    // **the world origin**. The island had none, so every spatial source it has
    // ever carried was mixed against (0, 0, 0): a vehicle's engine loop four
    // hundred metres from the hero was as loud as one under the bonnet, and the
    // occlusion raycast was a line from the origin to the emitter through
    // whatever a kilometre of island happened to contain.
    //
    // Nothing measured it, because nothing on this island made a spatial noise
    // that anybody stood next to — the traffic is silent by ruling, and the
    // footstep cue is a one-shot at the body's own feet. The venue's music is
    // the first, and this is the attachment its own boot path was missing.
    // (`AudioListener` is a scene-v6 component: placing one moves no schema.)
    insert!(doc, hero, AudioListener { active: true });

    doc.world_mut().propagate();
    doc.mark_saved();
    doc
}

/// Write the island's committed halves beside its recipe: the level, the
/// `.inf_pcg` its biome set binds, and the streets its own blocks imply.
pub fn write_island_level(
    design: &inf_island::IslandDesign,
    dir: &std::path::Path,
) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("mkdir: {e}"))?;
    let name = design.recipe.name.as_str();
    let slug = inf_island::slug(name);

    crate::scene::serialize::save(
        &island_scene(design),
        &dir.join(format!("{slug}.inf_lvl")),
        Some(inf_island::level_guid(name)),
    )?;

    // **The street layer** (wave ROAD1b). It is written HERE, beside the level,
    // because it is derived from the same blocks the level writes and by the
    // same door — `inf_ecs::traffic::streets_of_blocks`. The island build
    // cannot derive it: `inf-island` does not link `inf-ecs` and has no
    // settlement planner, which is exactly why the island had two road networks
    // and paved only one of them.
    inf_island::layers::write_streets(
        &dir.join(&design.recipe.roads.streets),
        &design.anchor,
        &island_street_spans(design),
    )
    .map_err(|e| format!("write the island's street layer: {e}"))?;

    let bytes = inf_asset::encode(&island_cover_payload(design.recipe.seed_for("cover")))
        .map_err(|e| format!("encode the island's .inf_pcg: {e}"))?;
    let p = dir.join(format!("{slug}Cover.inf_pcg"));
    std::fs::write(&p, &bytes).map_err(|e| format!("write {}: {e}", p.display()))?;
    let mut side = inf_asset::AssetSidecar::new(
        inf_asset::AssetId(inf_island::cover_pcg_guid(name)),
        inf_asset::AssetKind::Pcg,
        inf_asset::ContentHash::of(&bytes),
    );
    // **The three cover meshes it scatters** (TER2a clause 5). The cook reaches
    // them through its own implicit `Pcg` edge either way; this is the edge the
    // ASSET DATABASE reads — the delete-with-references warning and the Content
    // Drawer's "show references" both walk sidecars, so without it an author can
    // delete a mesh thirteen thousand instances are standing on and be told
    // nothing.
    side.dependencies = crate::cover::CoverKind::ALL
        .iter()
        .map(|k| inf_asset::AssetId(crate::cover::cover_mesh_guid(*k)))
        .collect();
    side.save(&p)
        .map_err(|e| format!("write the .inf_pcg sidecar: {e}"))
}

/// The repository's own root, from this crate's manifest.
pub fn repo_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

/// **How fast the island's day runs** — clock-seconds per simulated second, the
/// `inf_ecs::components::TimeOfDay::rate` the level authors (island wave NPC1d).
///
/// Eighteen is an **eighty-minute day**, and it is set from a measurement rather
/// than chosen for looks. `inf_ecs::society` gives a commute one hour of the
/// level clock and a `ScheduleLeg` walks its route over that window, so the
/// metres per second a commute implies is `length x rate / 3600`. Measured on
/// the CI island's own derived population — 329 residents of Harbour City — the
/// median commute is **320 m**, so the rate that makes the median commute a
/// *walk* is `3600 x 1.65 / 320`, and eighteen is that number rounded to
/// something a reader can hold. At eighteen the island's commutes imply
/// **0.89 / 1.60 / 1.97 m/s** (min / median / max) against the movement model's
/// own `walk_speed_mps` of 1.65 — every one of them a walking pace, and the
/// median within three per cent of it.
///
/// The first draft was thirty (a forty-eight-minute day, which is a nicer number
/// to say) and it made the median commute **2.67 m/s** — a jog. The arm that
/// found that is `the_islands_own_rate_makes_a_commute_a_walk`, and it is in the
/// gate rather than in this comment because a proportion stated in a doc is a
/// claim (the NPC1c law about a 2.4 m capsule wearing a 1.8 m comment).
///
/// It was **zero** — a frozen clock at 10:30 UTC — from wave I7 until this wave,
/// which is why the I8b night-window substrate (`inf_render::night_glow_step`)
/// had never once returned a non-zero step on the shipped island: the sun could
/// not get below the horizon. Turning it on is the whole of clause 3.
pub const ISLAND_CLOCK_RATE: f64 = 18.0;

/// **Which catalogue row parks at one settlement** (island wave VEH2a).
///
/// A city takes the two road cars, a town the working vehicles and the wagon,
/// and everything else the pickup — by the site's own kind and its place in the
/// settlement order, so a recipe that adds a town changes exactly one car.
///
/// A function and not a `match` inline in the generator, because the arm that
/// checks where a car sits has to know which car it is, and a rule restated in
/// a test is the P29.6 A14 defect: the first version of this WAS restated, and
/// the restatement went stale on the very commit that grew the fleet.
pub fn island_vehicle_id(kind: inf_island::SiteKind, index: usize) -> &'static str {
    match (kind, index % 3) {
        (inf_island::SiteKind::City, 0) => "sedan",
        (inf_island::SiteKind::City, _) => "sports",
        (inf_island::SiteKind::Town, 0) => "van",
        (inf_island::SiteKind::Town, 1) => "suv",
        (inf_island::SiteKind::Town, _) => "sedan",
        _ => "truck",
    }
}

/// **The fleet one institution keeps** (wave EMS1) — the catalogue rows parked
/// at its own street frontage, in the order they are parked.
///
/// A function and not a `match` inline in the generator, for
/// [`island_vehicle_id`]'s reason verbatim: the gate has to know which vehicle
/// is whose, and a rule restated in a test is the A14 defect.
///
/// A **clinic keeps nothing**, and that is the falsifying entry: a table where
/// every institution had a fleet would make "an institution has vehicles" a
/// property of the word. A clinic has a car park and the surgery is not an
/// emergency service.
pub fn station_fleet(a: inf_pcg::ArchetypeId) -> &'static [&'static str] {
    match a {
        // Two patrol cars and the tactical van the mandate names.
        inf_pcg::ArchetypeId::PoliceStation => &["cruiser", "cruiser", "swat"],
        inf_pcg::ArchetypeId::FireHall => &["engine"],
        inf_pcg::ArchetypeId::Hospital => &["ambulance", "ambulance"],
        _ => &[],
    }
}

/// Metres between two parked emergency vehicles along the apron.
///
/// A fire appliance is 7.8 m long, which is the longest thing this catalogue
/// parks, so eleven metres is the vehicle plus a clear gap either side — enough
/// that two of them at one station do not begin the level interpenetrating,
/// which is a physics solve nobody authored.
pub const EMS_PARK_PITCH_M: f64 = 11.0;

/// Metres an emergency vehicle stands **off the road's centreline**, on the side
/// its own station is.
///
/// **The apron, not the carriageway**, and the island gate is what taught this
/// wave the difference. The first cut parked the fleet ON the route line — clear
/// of the settlement's own car along it, which was the bug before that — and an
/// appliance eleven metres ahead of a saloon is a saloon that drives 7.8 m in
/// five seconds of throttle and stops. "The car is parked" is what the gate said,
/// and it was right: something was parked in front of it.
///
/// Six metres is `inf_ecs::traffic::KERB_PARK_OFFSET_M`'s five plus the width an
/// appliance has over a saloon, so the widest thing in the catalogue still has
/// its flank out of the lane.
pub const EMS_APRON_OFFSET_M: f64 = 6.0;

/// The city whose harbour the launch is moored in, and whose police station
/// keeps the helicopter (wave VEH2c).
///
/// Named rather than indexed: the recipe's site order is the recipe's business
/// and a wave that added a town would otherwise move the boat to a different
/// island.
pub const HARBOUR_CITY: &str = "Harbour City";

/// How far seaward of the coast ring the launch is moored, metres.
///
/// Past the beach (`[sea] beach_width_m` is 32 m on the committed recipe) and
/// onto the shelf, so the hull is over water deep enough to float in rather than
/// resting on a slipway. Far enough to be honest, close enough to swim to.
pub const HARBOUR_OFFSHORE_M: f64 = 70.0;

/// Which of the station apron's slots the helipad takes.
///
/// `station_fleet(PoliceStation)` parks three vehicles at slots 1, 2 and 3, so
/// the pad at slot 5 clears the last of them by a whole `EMS_PARK_PITCH_M` —
/// which is what keeps a helicopter off the roof of a cruiser.
pub const HELIPAD_SLOT: usize = 5;

/// The pad's radius, metres — comfortably wider than the catalogue's 4.3 m disc.
pub const HELIPAD_RADIUS_M: f64 = 7.0;

/// The pad's half-thickness, metres.
pub const HELIPAD_HALF_M: f64 = 0.30;

/// How much of the pad is buried, metres.
///
/// Most of it: a slab standing 30 cm proud of an apron is a kerb a character
/// trips over, and what is wanted is a surface with a lip.
pub const HELIPAD_BURY_M: f64 = 0.18;

/// The lip the machine's skids stand on, metres — the pad's own proud height.
pub const HELIPAD_LIP_M: f64 = HELIPAD_HALF_M - HELIPAD_BURY_M;

/// How many extra pitches a vehicle may step along its own apron to clear one
/// its settlement has already parked (EMS1 audit).
///
/// A bound rather than a `while`, because a placement loop inside a level load
/// must terminate whatever the road does. Sixteen is five times the largest
/// fleet any one station keeps, and the arm
/// `every_station_parks_its_fleet_in_its_own_livery` asserts the OUTCOME — so if
/// a future island ever exhausts this, the gate says so rather than the level
/// quietly shipping an overlap again.
const EMS_PARK_MAX_SHUNT: usize = 16;

/// Every committed island recipe, as repo-relative paths.
///
/// Exhaustive by hand and in one place, so adding an island is a decision taken
/// once rather than three times (the bless, the byte lock and the level count).
pub const ISLAND_RECIPES: [&str; 2] = [
    "samples/island/island.toml",
    "samples/island-fixture/island.toml",
];

/// **The island's settlement blocks, as the street derivation reads them**
/// (wave ROAD1b).
///
/// The same rectangles [`island_scene`] writes into the level as `PcgVolume`s —
/// `b.centre` and `b.half` — carrying the same GUIDs, because
/// `inf_ecs::traffic::streets_of_blocks` groups by proximity in `Guid` order and
/// a different identity is a different grouping.
///
/// `pad_y` is the settlement's own design height rather than a doorway sill: at
/// author time no block has been evaluated, so it has no doorways, and the
/// paving does not read the height anyway (the ribbon takes its own from the
/// terrain). It is the settlement's centre elevation so the value is *a*
/// walking surface rather than a zero nobody meant.
pub fn island_block_rects(design: &inf_island::IslandDesign) -> Vec<inf_ecs::traffic::BlockRect> {
    let name = design.recipe.name.as_str();
    let mut out = Vec::new();
    for plan in &crate::settlement::settlements(design) {
        for b in &plan.blocks {
            out.push(inf_ecs::traffic::BlockRect {
                guid: crate::settlement::block_guid(name, b.site, b.col, b.row),
                centre: b.centre,
                half: b.half,
                pad_y: 0.0,
            });
        }
    }
    out
}

/// **The streets the island's own blocks imply** (wave ROAD1b) — the
/// centrelines the traffic sim drives, the parked-car lattice sits on and the
/// crowd's pavement ring surrounds, and, since this wave, the ones that get
/// paved.
///
/// One line: `inf_ecs::traffic::streets_of_blocks` over [`island_block_rects`].
/// There is no second derivation and there must not be — the defect this wave
/// was called for is that the island had two road networks and only one of them
/// was drawn.
pub fn island_streets(design: &inf_island::IslandDesign) -> Vec<inf_ecs::traffic::Street> {
    inf_ecs::traffic::streets_of_blocks(&island_block_rects(design))
}

/// **The street network as spans between grid crossings** (wave ROAD1b) — what
/// the committed street layer holds and what the paving is built from.
///
/// The split points are `inf_ecs::traffic::carriageway_graph`'s nodes, which is
/// the graph the traffic sim routes on: so a paved span and a driven edge are
/// the same piece of street by construction, not by a comment. Every node it
/// makes is a crossing or a line's end, and consecutive nodes are linked — the
/// links ARE the spans.
///
/// Sorted at the end so the committed layer is a function of the design and not
/// of a hash walk. The lane count comes from
/// `inf_ecs::traffic::street_lanes`, the one door that turns a reserve into a
/// carriageway.
pub fn island_street_spans(design: &inf_island::IslandDesign) -> Vec<inf_island::StreetSpan> {
    let streets = island_streets(design);
    let graph = inf_ecs::traffic::carriageway_graph(&streets);
    // Which street a node pair belongs to, so a span can state the reserve it
    // came from: a crossing node is shared, but the SEGMENT between two nodes
    // lies on exactly one line — the one whose axis it runs along.
    let mut out: Vec<inf_island::StreetSpan> = Vec::new();
    let mut seen: std::collections::BTreeSet<(u64, u64)> = Default::default();
    for n in graph.nodes() {
        for e in graph.edges_from(n.id) {
            let key = (n.id.min(e.to), n.id.max(e.to));
            if !seen.insert(key) {
                continue;
            }
            let Some(other) = graph.node(e.to) else {
                continue;
            };
            let (a, b) = (n.position, other.position);
            // The reserve: the street this span runs along. A span is axis
            // aligned, so the line that carries it is the one whose own axis
            // matches and whose perpendicular coordinate it sits on.
            let along_x = (a.z - b.z).abs() <= (a.x - b.x).abs();
            let gap = streets
                .iter()
                .filter(|s| s.along_x() == along_x)
                .map(|s| {
                    let d = if along_x {
                        (s.a.y - a.z).abs()
                    } else {
                        (s.a.x - a.x).abs()
                    };
                    (d, s.gap_m)
                })
                .min_by(|p, q| p.0.total_cmp(&q.0))
                .map(|(_, g)| g)
                .unwrap_or(inf_ecs::traffic::MIN_STREET_GAP_M);
            out.push(inf_island::StreetSpan {
                a,
                b,
                lanes: inf_ecs::traffic::street_lanes(gap),
                gap_m: gap,
            });
        }
    }
    out.sort_by(|p, q| {
        p.a.x
            .total_cmp(&q.a.x)
            .then(p.a.z.total_cmp(&q.a.z))
            .then(p.b.x.total_cmp(&q.b.x))
            .then(p.b.z.total_cmp(&q.b.z))
    });
    out
}

/// Read one committed island's design, or `None` when the recipe is not present.
///
/// `None` rather than an error: a tree that has not blessed the samples yet
/// should not fail CI, which is the same rule
/// `committed_sample_matches_generators` already applies to every other sample.
pub fn committed_design(rel: &str) -> Option<inf_island::IslandDesign> {
    let p = repo_root().join(rel);
    if !p.exists() {
        return None;
    }
    let recipe = inf_island::IslandRecipe::load(&p).ok()?;
    inf_island::read_design(&recipe).ok()
}

/// Write every committed island's level and `.inf_pcg`.
pub fn write_island_levels() -> Result<(), String> {
    for rel in ISLAND_RECIPES {
        let Some(d) = committed_design(rel) else {
            continue;
        };
        let dir = repo_root().join(rel).parent().unwrap().to_path_buf();
        write_island_level(&d, &dir)?;
    }
    Ok(())
}

/// **The committed-design source scan**, shared by every module that authors a
/// committed island document (island wave I8a).
///
/// It was `island.rs`'s alone until the settlement generator arrived, and one
/// file's scan is a scan that stops at the file somebody happened to write
/// first: the level's numbers and the settlements' block positions reach the
/// same committed `.inf_lvl`, so both have to be authored from committed design
/// alone or neither is.
#[cfg(test)]
pub(crate) mod scan {
    /// The **non-test, non-comment** lines of a module source, one-based.
    ///
    /// The scan stops at the test module, and it has to: an arm's own needle
    /// list lives in the file it scans, so a whole-file scan matches itself and
    /// fails on the line that declares what it is looking for. (It did.)
    pub fn code_lines(whole: &str) -> Vec<(usize, String)> {
        let src = whole
            .split_once("#[cfg(test)]")
            .map(|(before, _)| before)
            .unwrap_or(whole);
        assert!(
            src.len() < whole.len(),
            "the test module marker moved; this scan is reading itself"
        );
        assert!(
            src.len() > 4_000,
            "the scan is reading {} bytes of a module that is not that small",
            src.len()
        );
        src.lines()
            .enumerate()
            .filter(|(_, l)| !l.trim_start().starts_with("//"))
            .map(|(i, l)| (i + 1, l.to_string()))
            .collect()
    }

    /// Every `inf_island::` item a code listing names, with the line each was
    /// first seen on.
    ///
    /// # It reads a BRACE GROUP as well as a path, and that was a hole
    ///
    /// The first version took the identifier immediately after `inf_island::`
    /// and stopped. `use inf_island::{IslandDesign, Route};` puts a `{` there,
    /// so the extractor read an **empty name and recorded nothing** — a module
    /// that imported every door in the crate by one `use` line scanned clean.
    /// Found the day a second module joined the scan and its own anti-vacuity
    /// arm (*"the module no longer reads the committed design at all"*) fired.
    ///
    /// # …and a brace group that WRAPS, which is the same hole one line down
    ///
    /// (Island wave I8a audit.) Reading `{` to the `}` on the same line still
    /// missed the form `rustfmt` produces the moment an import list is long
    /// enough to wrap:
    ///
    /// ```text
    /// use inf_island::{
    ///     IslandDesign, Route, Site, SiteKind, sample_terrain,
    /// };
    /// ```
    ///
    /// The first line's `{` opens a group with nothing after it, the following
    /// lines never say `inf_island::` at all, and the module scans clean again.
    /// That is not a hypothetical shape: `settlement.rs`'s own import is four
    /// names and 48 characters, and the fifth door anybody adds wraps it. An open
    /// group therefore stays open across lines until its `}`, and every name in
    /// it is recorded against the line the group **started** on.
    pub fn island_doors(code: &[(usize, String)]) -> std::collections::BTreeMap<String, usize> {
        fn record(used: &mut std::collections::BTreeMap<String, usize>, group: &str, line: usize) {
            for name in group.split(',') {
                let name = name.trim();
                if !name.is_empty() {
                    used.entry(name.to_string()).or_insert(line);
                }
            }
        }
        let mut used: std::collections::BTreeMap<String, usize> = Default::default();
        // `Some(line)` while a brace group opened on `line` is still unclosed.
        let mut open: Option<usize> = None;
        for (n, line) in code {
            let mut rest = line.as_str();
            if let Some(started) = open {
                match rest.split_once('}') {
                    Some((group, tail)) => {
                        record(&mut used, group, started);
                        open = None;
                        rest = tail;
                    }
                    None => {
                        record(&mut used, rest, started);
                        continue;
                    }
                }
            }
            while let Some(at) = rest.find("inf_island::") {
                rest = &rest[at + "inf_island::".len()..];
                if let Some(stripped) = rest.strip_prefix('{') {
                    match stripped.split_once('}') {
                        Some((group, tail)) => {
                            record(&mut used, group, *n);
                            rest = tail;
                        }
                        None => {
                            record(&mut used, stripped, *n);
                            open = Some(*n);
                            rest = "";
                        }
                    }
                    continue;
                }
                let name: String = rest
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_')
                    .collect();
                if !name.is_empty() {
                    used.entry(name).or_insert(*n);
                }
            }
        }
        used
    }

    /// Lines that reach `inf_island` under **another name**, which is the one
    /// spelling [`island_doors`] cannot follow (island wave I8a audit).
    ///
    /// The extractor is an allowlist over what it can read, and what it reads is
    /// the literal `inf_island::`. `use inf_island as isl;` — or
    /// `use inf_island::sample_terrain as h;` — renames the door and the scan
    /// walks past it. There is no cheap way to follow an alias without parsing,
    /// so the alias itself is refused: a module authored from committed design
    /// alone has no reason to want one, and a REFUSAL is a rule an author meets
    /// immediately rather than a hole nobody meets at all.
    /// Only `use` lines are read, so `inf_island::clamp(x as i32)` is not an
    /// alias and does not trip it. A `type` alias needs no rule: the aliased path
    /// still spells `inf_island::<Door>` on its own line, so the scan has already
    /// recorded the door by the time anybody uses the short name.
    pub fn aliases(code: &[(usize, String)]) -> Vec<usize> {
        code.iter()
            .filter(|(_, l)| {
                let t = l.trim_start();
                t.starts_with("use ") && t.contains("inf_island") && t.contains(" as ")
            })
            .map(|(n, _)| *n)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn design(rel: &str) -> Option<inf_island::IslandDesign> {
        committed_design(rel)
    }

    /// Lines of the module's **non-test, non-comment** source, with their
    /// one-based numbers — the scan every arm below reads.
    fn module_code() -> Vec<(usize, String)> {
        super::scan::code_lines(include_str!("island.rs"))
    }

    /// **Nothing here reads an elevation.** The level is authored from committed
    /// design alone, which is what makes it a committed document CI can check.
    ///
    /// # An ALLOWLIST, because a ban enumerates only what somebody thought of
    ///
    /// The first version of this arm banned five names — `TileMosaic`,
    /// `plan_tiles`, `build_island`, `elevation_at`, `ProjectionLattice` — and
    /// every other door onto an elevation sailed through it:
    /// `inf_island::sample_terrain`, `inf_island::IslandBuild`,
    /// `inf_terrain::read_terrain_asset`, `TerrainData::height_at`. That is the
    /// P22 law ("a ban enumerates what you thought of, an allowlist what is
    /// allowed") met on the arm that carries this wave's own headline decision.
    ///
    /// So the claim is inverted: the module may name **exactly these** items of
    /// `inf_island`, and no `inf_terrain` item at all. Adding a door here is a
    /// deliberate edit to this list rather than a silent one to the module.
    #[test]
    fn the_level_is_authored_from_committed_design_alone() {
        /// Everything `island.rs` is allowed to reach in the island crate. Every
        /// one of them is a **committed-design** door or a name-derived GUID; not
        /// one of them opens an elevation tile.
        const ALLOWED: &[&str] = &[
            "IslandDesign", // the committed design, read by `read_design`
            "IslandRecipe", // the committed recipe
            "SiteKind",     // a site's own kind, off the committed recipe
            // Wave ROAD1b's street layer: the spans the level's OWN BLOCKS imply,
            // through `inf_ecs::traffic::streets_of_blocks`, written beside the
            // level by `write_island_level`. `StreetSpan` is the record and
            // `layers::write_streets` the writer; neither reads a height — a
            // span carries the settlement's own plan Y and the ribbon takes its
            // elevation from the terrain at BUILD time, in `inf-island`, where
            // the tiles are. That is the same division `write_roads` already
            // has, and it is why this layer can be committed at all.
            "StreetSpan",
            "biome_set_guid", // …and five GUIDs derived from the island's name
            "cover_pcg_guid",
            "level_guid",
            // The nearest vertex of the committed ROAD LAYER, and the direction
            // the route runs there (island wave VEH1a). The routes are the one
            // committed thing on this island that carries a ground height —
            // `player_start` has read them for exactly that since I7 — so a car
            // parked on the circuit is placed from the design and not from a
            // tile. It opens nothing.
            "nearest_route_vertex",
            "layers",      // …its writer (wave ROAD1b; see `StreetSpan` above)
            "read_design", // the one door onto the committed layers
            "road_mesh_guid",
            // Wave ROAD1's three road-furniture meshes. Same shape as
            // `road_mesh_guid` and the same reason it is allowed: a name-derived
            // GUID, a pure function of the recipe's own `name`, which opens no
            // tile and reads no elevation. `road_part_mesh_guid` on the
            // carriageway IS `road_mesh_guid`, so the committed level's existing
            // binding is unmoved.
            "road_part_mesh_guid",
            "slug",
            "terrain_guid",
        ];
        let code = module_code();
        assert!(
            super::scan::aliases(&code).is_empty(),
            "island.rs imports `inf_island` under another name at line(s) {:?} — \
             the scan follows the literal `inf_island::` and an alias walks past \
             it (island wave I8a audit)",
            super::scan::aliases(&code)
        );
        let used = super::scan::island_doors(&code);
        println!("island.rs reaches inf_island::{{{:?}}}", used.keys());
        for (name, line) in &used {
            assert!(
                ALLOWED.contains(&name.as_str()),
                "island.rs:{line} names `inf_island::{name}`, which is not on the \
                 committed-design allowlist. The level must be authorable without \
                 the terrain, or it is a committed document only one machine can \
                 produce — and the terrain is a build artifact of one machine \
                 because the sampling step goes through the projection modules \
                 the portability law exempts. If this door really is design-only, \
                 add it to ALLOWED with the reason."
            );
        }
        // …and the allowlist is not vacuous: the module really does reach the
        // crate, and reaches the ONE door the decision names.
        assert!(used.len() >= 5, "island.rs reaches {} items", used.len());
        assert!(
            used.contains_key("read_design"),
            "the module no longer opens the committed design at all"
        );

        // The terrain crate is out of bounds entirely — `read_terrain_asset`,
        // `TerrainData` and `height_at` all live there, and none of them is a
        // committed-design door.
        for (n, line) in &code {
            assert!(
                !line.contains("inf_terrain::"),
                "island.rs:{n} names `inf_terrain::` — every door onto an \
                 elevation is in that crate: {}",
                line.trim()
            );
        }
    }

    /// **The scan can fail** — the anti-vacuity arm the sibling gate
    /// (`inf-island/tests/portable_math_law.rs`) has and this one did not.
    ///
    /// A source scan whose extraction is broken is indistinguishable from a
    /// module that is clean, and the arm above would have passed over an empty
    /// string, a mis-split file or a needle that never matches.
    #[test]
    fn the_committed_design_scan_finds_a_door_when_one_is_there() {
        // The extractor, run against a line that names a forbidden door.
        let probe = vec![(
            1usize,
            "    let t = inf_island::sample_terrain(&r, &m, &l, &c);".to_string(),
        )];
        let found = super::scan::island_doors(&probe);
        assert_eq!(found.keys().collect::<Vec<_>>(), vec!["sample_terrain"]);
        assert!(
            !["IslandDesign", "read_design", "slug"].contains(&"sample_terrain"),
            "a real door must not be on the allowlist"
        );
        // **And a BRACE GROUP, which the first extractor could not see at all**
        // (island wave I8a): `use inf_island::{A, B};` put a `{` where the
        // identifier was expected, the take-while read an empty string, and a
        // module importing every door in the crate on one line scanned clean.
        let group = vec![(
            2usize,
            "use inf_island::{IslandBuild, sample_terrain, IslandDesign};".to_string(),
        )];
        let doors = super::scan::island_doors(&group);
        let names: Vec<&str> = doors.keys().map(String::as_str).collect();
        assert_eq!(names, vec!["IslandBuild", "IslandDesign", "sample_terrain"]);
        // **AND THE SAME GROUP WRAPPED OVER THREE LINES** (island wave I8a
        // audit), which is what `rustfmt` writes the moment the list is long
        // enough — and which the same-line reader above still could not see: the
        // `{` opened a group with nothing after it and the following lines never
        // say `inf_island::` at all.
        let wrapped = vec![
            (10usize, "use inf_island::{".to_string()),
            (11, "IslandDesign, Route, Site,".to_string()),
            (12, "SiteKind, sample_terrain,".to_string()),
            (13, "};".to_string()),
        ];
        let doors = super::scan::island_doors(&wrapped);
        let mut names: Vec<&str> = doors.keys().map(String::as_str).collect();
        names.sort_unstable();
        assert_eq!(
            names,
            vec![
                "IslandDesign",
                "Route",
                "Site",
                "SiteKind",
                "sample_terrain"
            ],
            "a wrapped brace import scanned clean — the hole moved one line down"
        );
        assert_eq!(
            doors.get("sample_terrain"),
            Some(&10),
            "a name inside a wrapped group is reported against the line the group \
             opened on"
        );
        // …and an ALIAS is refused rather than followed, because the extractor
        // reads one spelling and a rename is the spelling it cannot read.
        assert_eq!(
            super::scan::aliases(&[
                (1usize, "use inf_island as isl;".to_string()),
                (2, "use inf_island::sample_terrain as h;".to_string()),
                (3, "let n = inf_island::clamp(x as i32);".to_string()),
            ]),
            vec![1, 2],
            "the alias probe either missed a rename or called an `as` cast one"
        );
        // …and a comment line is filtered, which is why the real scan drops them.
        assert!("    // inf_island::sample_terrain"
            .trim_start()
            .starts_with("//"));
        // The real module is being read, and it is the module this file is in.
        let code = module_code();
        assert!(code.len() > 200, "the scan read {} lines", code.len());
        assert!(
            code.iter().any(|(_, l)| l.contains("pub fn island_scene")),
            "the scan is not reading island.rs"
        );
    }

    #[test]
    fn the_guids_are_stable_distinct_and_a_function_of_the_island() {
        let a = "Vancouver Island";
        let mut all = std::collections::BTreeSet::new();
        for g in [
            sun_guid(a),
            terrain_entity_guid(a),
            roads_entity_guid(a),
            ocean_guid(a),
            hero_guid(a),
            lake_guid(a, 0),
            lake_guid(a, 1),
            river_guid(a, 0),
            cover_volume_guid(a, 0),
            inf_island::terrain_guid(a),
            inf_island::road_mesh_guid(a),
            inf_island::biome_set_guid(a),
            inf_island::cover_pcg_guid(a),
            inf_island::level_guid(a),
        ] {
            assert!(all.insert(g), "two of the island's guids collide: {g}");
        }
        assert_eq!(sun_guid(a), sun_guid(a));
        assert_ne!(sun_guid(a), sun_guid("Other Island"));
    }

    /// **EVERY STATION HAS ITS FLEET, ON ITS OWN APRON, IN ITS OWN LIVERY**
    /// (wave EMS1).
    ///
    /// Four claims, each of which a plausible-looking bug would break on its
    /// own:
    ///
    /// * the vehicles exist and are rigs the recogniser finds — so "the island
    ///   has an ambulance" is a measurement and not a table;
    /// * they stand at **their own institution's** frontage rather than at the
    ///   settlement's, which is what stops a city's four stations sharing one
    ///   heap of cars;
    /// * two vehicles of one station are at least a vehicle length apart, so a
    ///   two-car station does not begin the level interpenetrating;
    /// * and every one of them wears its livery — asserted on a **part's own
    ///   material**, because a livery that reached the table and not the entity
    ///   is exactly the failure a table-shaped test cannot see.
    ///
    /// The falsifying half is the clinic: it keeps nothing, and the arm says so.
    #[test]
    fn every_station_parks_its_fleet_in_its_own_livery() {
        for recipe in ISLAND_RECIPES {
            let Some(d) = design(recipe) else {
                println!("SKIP: no {recipe} in this tree");
                continue;
            };
            let doc = island_scene(&d);
            let name = d.recipe.name.as_str();
            let fleet = crate::vehicle::island_vehicles();
            let mut total = 0usize;
            let mut per_row: std::collections::BTreeMap<&str, usize> = Default::default();
            for plan in crate::settlement::settlements(&d) {
                let mut at: Vec<(&'static str, glam::DVec3)> = Vec::new();
                for b in plan.blocks.iter().filter(|b| b.archetype.is_institution()) {
                    let want = station_fleet(b.archetype);
                    if b.archetype == inf_pcg::ArchetypeId::Clinic {
                        assert!(want.is_empty(), "a clinic is not an emergency service");
                    }
                    for (k, id) in want.iter().enumerate() {
                        let guid = derived(
                            name,
                            &format!("island.ems.{}.{}.{}.{k}.{id}", b.site, b.col, b.row),
                        );
                        let rig =
                            inf_ecs::vehicle::rig_of(doc.world(), guid).unwrap_or_else(|| {
                                panic!(
                                    "{recipe}: {} {} has no {id} at ({}, {})",
                                    plan.name,
                                    b.archetype.name(),
                                    b.col,
                                    b.row
                                )
                            });
                        assert_eq!(rig.wheels.len(), 4, "{recipe}: {id}");
                        let e = doc.entity_of(guid).expect("the chassis");
                        let t = doc
                            .world()
                            .world()
                            .get::<inf_ecs::components::Transform>(e)
                            .expect("its transform");
                        at.push((b.archetype.name(), t.translation.to_dvec3()));

                        // The apron is THIS block's, not the settlement's.
                        let (v, _) = inf_island::nearest_route_vertex(&d.routes, b.centre)
                            .expect("the island has routes");
                        let def = fleet.get(id).expect("the catalogue row");
                        let want_y = crate::vehicle::resting_origin_y(def, v.y) + CAR_LIFT_M;
                        assert!(
                            (t.translation.y - want_y).abs() < 1e-9,
                            "{recipe}: a {id} sits at {} against a resting {want_y}",
                            t.translation.y
                        );

                        // THE LIVERY REACHED THE ENTITY. Checked on the part
                        // the table names first, and on the light bar, which is
                        // the part that only exists because of the livery.
                        let livery = crate::vehicle::island_vehicle_livery(id)
                            .unwrap_or_else(|| panic!("{id} has no livery"));
                        let (pname, paint) = livery.parts[0];
                        let pe = doc
                            .entity_of(inf_ecs::vehicle::body_part_guid(guid, pname))
                            .unwrap_or_else(|| panic!("{recipe}: no `{pname}` on a {id}"));
                        let m = doc
                            .world()
                            .world()
                            .get::<inf_ecs::components::Material>(pe)
                            .expect("a drawn part has a material");
                        assert_eq!(
                            m.base_color, paint.base_color,
                            "{recipe}: a {id}'s `{pname}` is not its livery's colour"
                        );
                        let be = doc
                            .entity_of(inf_ecs::vehicle::body_part_guid(guid, "light_bar"))
                            .unwrap_or_else(|| panic!("{recipe}: a {id} has no light bar"));
                        let bm = doc
                            .world()
                            .world()
                            .get::<inf_ecs::components::Material>(be)
                            .expect("a light bar is drawn");
                        let lin = bm.emissive_linear();
                        assert!(
                            lin[0].max(lin[1]).max(lin[2]) > 1.0,
                            "{recipe}: a {id}'s light bar is {lin:?} and does not bloom"
                        );

                        total += 1;
                        *per_row.entry(id).or_default() += 1;
                    }
                }
                // **ACROSS THE WHOLE SETTLEMENT, not within one station** (EMS1
                // audit). The first spelling of this check reset its list per
                // BLOCK, so it could only see two vehicles of one station
                // stacked — and the placement's own probe is
                // `nearest_route_vertex(b.centre)`, which two institution blocks
                // of one city may perfectly well answer with the SAME vertex.
                // That is the "car parked in a saloon" defect one level up: an
                // engine and an ambulance materializing on one point, with the
                // arm that exists to catch interpenetration looking at each of
                // them alone. Measured on both committed recipes before the
                // bound was written down.
                for i in 0..at.len() {
                    for j in i + 1..at.len() {
                        let g = (at[i].1 - at[j].1).length();
                        assert!(
                            g >= EMS_PARK_PITCH_M - 1e-9,
                            "{recipe}: {}'s {} and {} stand {g:.2} m apart — two \
                             vehicles begin the level interpenetrating",
                            plan.name,
                            at[i].0,
                            at[j].0
                        );
                    }
                }
            }
            println!("EMS1 fleet: {recipe} parks {total} -> {per_row:?}");
            assert!(
                total > 0,
                "{recipe} parks no emergency vehicle at all, so every claim \
                 above is about an empty list"
            );
            // The shipped island has all four rows on it; the four-block
            // fixture has one fire hall and therefore one appliance.
            if recipe == ISLAND_RECIPES[0] {
                for id in ["cruiser", "ambulance", "swat", "engine"] {
                    assert!(per_row.contains_key(id), "{recipe} parks no {id}");
                }
            }
        }
    }

    /// **THE CIRCUIT HAS SOMETHING ON IT** (island wave VEH1a) — I7's open item
    /// 15, *"the circuit is drawn and audited, and nothing drives it"*, as an
    /// assertion.
    ///
    /// One car per settlement, standing on the drivable circuit at the point it
    /// passes the town, derivable as a rig by the same recogniser the physics
    /// bridge uses, and — the clause the committed car could never have met —
    /// **drawn at the size of its own collider**.
    ///
    /// Run against both committed islands, because "the fixture is what CI
    /// exercises" is only true while the fixture and the shipped island are the
    /// same generator.
    /// **THE HARBOUR HAS A BOAT AND THE STATION HAS A PAD** (wave VEH2c) — and
    /// the boat is over water rather than on a beach.
    ///
    /// The claim that costs something is the last one. Nothing in this module
    /// may read an elevation, so where the launch floats is derived from two
    /// committed numbers — the coast ring and `[sea] level_m` — and the arm
    /// checks the derivation against the design rather than against the
    /// generator's own arithmetic: the mooring must be SEAWARD of the shore,
    /// which is to say further from the city than the shore point is, and clear
    /// of the beach the recipe declares.
    #[test]
    fn the_harbour_moors_a_launch_and_the_station_keeps_a_helicopter() {
        for recipe in ISLAND_RECIPES {
            let Some(d) = design(recipe) else {
                println!("SKIP: no {recipe} in this tree");
                continue;
            };
            let doc = island_scene(&d);
            let name = d.recipe.name.as_str();
            let plans = crate::settlement::settlements(&d);
            let Some(plan) = plans.iter().find(|p| p.name == HARBOUR_CITY) else {
                println!("SKIP: {recipe} has no {HARBOUR_CITY}");
                continue;
            };
            let fleet = crate::vehicle::island_vehicles();

            // ── the launch ──────────────────────────────────────────────────
            let guid = derived(name, "island.launch");
            let rig = inf_ecs::vehicle::rig_of(doc.world(), guid)
                .unwrap_or_else(|| panic!("{recipe}: no launch the recogniser finds"));
            assert!(rig.wheels.is_empty(), "{recipe}: the launch has wheels");
            assert_eq!(rig.parts.len(), 1, "{recipe}: the launch's screw");
            assert_eq!(rig.parts[0].kind, inf_ecs::vehicle::PartKind::Thruster);

            let e = doc.entity_of(guid).expect("the launch's chassis");
            let t = *doc
                .world()
                .world()
                .get::<inf_ecs::components::Transform>(e)
                .expect("its transform");
            // It FLOATS: a `Buoyancy` the water pass will find, at the density
            // its own catalogue row authored rather than at the one it weighs.
            let def = fleet.get("launch").expect("the launch row");
            let b = doc
                .world()
                .world()
                .get::<inf_ecs::components::Buoyancy>(e)
                .unwrap_or_else(|| panic!("{recipe}: the launch does not float"));
            assert!(b.enabled);
            assert_eq!(b.density_kg_m3, def.buoyancy_density_kg_m3);
            assert_eq!(b.linear_drag, def.buoyancy_linear_drag);
            // …at its equilibrium, so the level does not open with a splash.
            assert!(
                (t.translation.y - inf_ecs::vehicle::floating_origin_y(def, d.recipe.sea.level_m))
                    .abs()
                    < 1e-9,
                "{recipe}: the launch floats at {} and the sea is at {}",
                t.translation.y,
                d.recipe.sea.level_m
            );

            // …and it is OUT TO SEA. Measured against the design's own coast
            // ring: the mooring is further from the city than the shore is, by
            // more than the beach the recipe declares.
            let here = glam::DVec2::new(t.translation.x, t.translation.z);
            let mut shore = f64::MAX;
            let mut nearest = glam::DVec2::ZERO;
            for ring in &d.coast {
                for v in ring {
                    let s = (*v - plan.centre).length();
                    if s < shore {
                        shore = s;
                        nearest = *v;
                    }
                }
            }
            let out = (here - plan.centre).length();
            println!(
                "{recipe}: the launch is {out:.0} m from {HARBOUR_CITY} and the \
                 shore is {shore:.0} m — {:.0} m of water",
                out - shore
            );
            assert!(
                out > shore + d.recipe.sea.beach_width_m,
                "{recipe}: the launch is {out:.0} m out and the shore is \
                 {shore:.0} m — it is on the beach"
            );
            // …and it really is seaward of the nearest shore point rather than
            // merely further from the centre by going round the island.
            assert!(
                (here - nearest).length() < HARBOUR_OFFSHORE_M * 1.05,
                "{recipe}: the launch is {:.0} m from its own shore point",
                (here - nearest).length()
            );

            // ── the pad and the machine on it ───────────────────────────────
            let Some(block) = plan
                .blocks
                .iter()
                .find(|b| b.archetype == inf_pcg::ArchetypeId::PoliceStation)
            else {
                println!("SKIP: {recipe}'s {HARBOUR_CITY} has no police station");
                continue;
            };
            let heli = derived(name, "island.chopper");
            let rig = inf_ecs::vehicle::rig_of(doc.world(), heli)
                .unwrap_or_else(|| panic!("{recipe}: no helicopter the recogniser finds"));
            assert!(rig.wheels.is_empty(), "{recipe}: the helicopter has wheels");
            assert_eq!(rig.parts.len(), 1);
            assert_eq!(rig.parts[0].kind, inf_ecs::vehicle::PartKind::Rotor);
            assert!(
                rig.parts[0].size.x > 4.0,
                "{recipe}: a {} m disc",
                rig.parts[0].size.x
            );

            let pad = derived(name, "island.helipad");
            let pe = doc
                .entity_of(pad)
                .unwrap_or_else(|| panic!("{recipe}: no helipad"));
            let pt = *doc
                .world()
                .world()
                .get::<inf_ecs::components::Transform>(pe)
                .expect("the pad's transform");
            assert!(
                doc.world()
                    .world()
                    .get::<inf_ecs::components::Collider3D>(pe)
                    .is_some_and(|c| !c.sensor),
                "{recipe}: the pad is not something a machine can stand on"
            );

            let he = doc.entity_of(heli).expect("the helicopter's chassis");
            let ht = *doc
                .world()
                .world()
                .get::<inf_ecs::components::Transform>(he)
                .expect("its transform");
            // The machine is ON the pad, and inside it.
            let off = glam::DVec2::new(
                ht.translation.x - pt.translation.x,
                ht.translation.z - pt.translation.z,
            );
            assert!(
                off.length() < 1e-9,
                "{recipe}: the helicopter is {:.1} m off its own pad",
                off.length()
            );
            // …and CLEAR of the cruisers on the same apron, which is what
            // `HELIPAD_SLOT` buys: the disc is 4.3 m and the nearest parked
            // vehicle must be further than that.
            let (v, _) = inf_island::nearest_route_vertex(&d.routes, block.centre)
                .expect("the station has a road");
            let along = (glam::DVec2::new(ht.translation.x, ht.translation.z)
                - glam::DVec2::new(v.x, v.z))
            .length();
            println!(
                "{recipe}: the pad is {along:.1} m along the apron; the fleet \
                 parks {} vehicle(s) at {EMS_PARK_PITCH_M} m pitch",
                station_fleet(block.archetype).len()
            );
            assert!(
                along
                    > station_fleet(block.archetype).len() as f64 * EMS_PARK_PITCH_M
                        + rig.parts[0].size.x,
                "{recipe}: the disc overlaps the parked fleet"
            );

            // The pad is a surface with a LIP, not a kerb: what stands proud is
            // a few centimetres, and the skids stand on that.
            let proud = pt.translation.y + HELIPAD_HALF_M - v.y;
            assert!(
                (proud - HELIPAD_LIP_M).abs() < 1e-9,
                "{recipe}: the pad stands {proud:.3} m proud of the apron"
            );
        }
    }

    #[test]
    fn every_settlement_parks_a_car_on_the_circuit() {
        for recipe in ISLAND_RECIPES {
            let Some(d) = design(recipe) else {
                println!("SKIP: no {recipe} in this tree");
                continue;
            };
            let doc = island_scene(&d);
            let name = d.recipe.name.as_str();
            let plans = crate::settlement::settlements(&d);
            assert!(!plans.is_empty(), "{recipe}: no settlements to park at");
            let fleet = crate::vehicle::island_vehicles();

            let mut drawn_parts = 0usize;
            for (n, plan) in plans.iter().enumerate() {
                let guid = derived(name, &format!("island.car.{}", plan.site));
                let rig = inf_ecs::vehicle::rig_of(doc.world(), guid).unwrap_or_else(|| {
                    panic!("{recipe}: {} has no car the recogniser finds", plan.name)
                });
                assert_eq!(rig.wheels.len(), 4, "{recipe}: {}", plan.name);
                assert_eq!(
                    rig.wheels.iter().filter(|w| w.steered()).count(),
                    2,
                    "{recipe}: {} — the front pair steers",
                    plan.name
                );

                // It stands ON the circuit, at the vertex the design commits.
                let (v, _) = inf_island::nearest_route_vertex(&d.routes, plan.centre)
                    .expect("the island has routes");
                let e = doc.entity_of(guid).expect("the chassis");
                let t = doc
                    .world()
                    .world()
                    .get::<inf_ecs::components::Transform>(e)
                    .expect("its transform");
                let def = fleet
                    .get(island_vehicle_id(plan.kind, n))
                    .expect("the catalogue row");
                assert!(
                    (t.translation.x - v.x).abs() < 1e-9 && (t.translation.z - v.z).abs() < 1e-9,
                    "{recipe}: {}'s car is at ({}, {}) and the circuit is at \
                     ({}, {})",
                    plan.name,
                    t.translation.x,
                    t.translation.z,
                    v.x,
                    v.z
                );
                // …and its wheels reach the road: the origin is the springs'
                // resting height plus the authored lift, no more.
                let want = crate::vehicle::resting_origin_y(def, v.y) + CAR_LIFT_M;
                assert!(
                    (t.translation.y - want).abs() < 1e-9,
                    "{recipe}: {}'s car sits at {} against a resting {want}",
                    plan.name,
                    t.translation.y
                );

                // THE CLAUSE THE COMMITTED CAR COULD NOT MEET: every drawn part
                // is inside the collider it is drawn on, and none of them is the
                // unit primitive.
                for part in def.body.parts() {
                    let pe = doc
                        .entity_of(inf_ecs::vehicle::body_part_guid(guid, part.name))
                        .unwrap_or_else(|| panic!("{recipe}: no `{}` part", part.name));
                    let pt = doc
                        .world()
                        .world()
                        .get::<inf_ecs::components::Transform>(pe)
                        .expect("a part transform");
                    assert_ne!(
                        pt.scale,
                        inf_ecs::math::Vec3d::ONE,
                        "{recipe}: `{}` is drawn at scale one — the I8b defect, \
                         on a car",
                        part.name
                    );
                    for (c, s, hull) in [
                        (pt.translation.x, pt.scale.x, def.half_extents.x),
                        (pt.translation.y, pt.scale.y, def.half_extents.y),
                        (pt.translation.z, pt.scale.z, def.half_extents.z),
                    ] {
                        assert!(c.abs() + s / 2.0 <= hull + 1e-9);
                    }
                    drawn_parts += 1;
                }
            }
            // Anti-vacuity: the loop really walked cars, and it walked one per
            // settlement rather than one that happened to exist.
            assert!(
                drawn_parts >= 4 * plans.len(),
                "{recipe}: {drawn_parts} drawn parts over {} settlements",
                plans.len()
            );
            println!(
                "{recipe}: {} settlements, {} cars, {drawn_parts} drawn body parts",
                plans.len(),
                plans.len()
            );
        }
    }

    /// **THE STREET HAS A SURFACE** (wave ASSET0, clause 0).
    ///
    /// The EDIT1 audit measured the island's `Roads` mesh at
    /// (225.7, 225.7, 225.7) UNLIT on the shipped editor — 0.8 linear, which is
    /// `Material::default().base_color`, which is what the projection hands a
    /// `MeshRef` carrying no `Material` component at all. The brightest surface
    /// in the frame was the engine's debug grey, in both hosts, and EDIT1 had
    /// just made the editor open standing on it.
    ///
    /// This arm reads the WORLD rather than the source: it pulls the component
    /// off the authored document, resolves the `.inf_mat` the component names
    /// out of the committed library, and compares the two. Deleting the
    /// `insert!` fails the first assertion; binding the wrong material fails the
    /// GUID; letting the scalars drift from the asset fails the last pair, which
    /// is the case that would show as a road changing colour as its pages
    /// arrive.
    ///
    /// The `< 0.2` ceiling is the anti-vacuity clause and it is aimed at exactly
    /// one number: `Material::default().base_color` is 0.8, so a road that
    /// somehow ended up back on the default cannot pass this arm by carrying a
    /// material id beside it.
    #[test]
    fn the_road_the_hero_stands_on_carries_a_material() {
        for recipe in [
            "samples/island/island.toml",
            "samples/island-fixture/island.toml",
        ] {
            let Some(d) = design(recipe) else {
                println!("SKIP: {recipe} is not in this tree");
                continue;
            };
            let doc = island_scene(&d);
            let name = d.recipe.name.as_str();
            let e = doc
                .entity_of(roads_entity_guid(name))
                .expect("the island draws a road");
            let m = doc
                .world()
                .world()
                .get::<inf_ecs::components::Material>(e)
                .copied()
                .expect(
                    "the `Roads` entity carries no Material -- the street is back on \
                     Material::default()'s 0.8 debug grey (EDIT1 finding 2)",
                );
            let want =
                crate::ground::ground_material_guid(inf_material::ground::GroundKind::Asphalt);
            assert_eq!(m.asset, Some(want), "{recipe}: the road binds a material");

            // …and the scalars are the material's own, so a host with no virtual
            // textures shades the same surface the pages would have.
            let lib = crate::ground::ground_library().expect("the ground library builds");
            let f = lib
                .iter()
                .find(|f| f.name == "Road_Asphalt.inf_mat")
                .expect("the library writes the asphalt material");
            let mat: inf_material::material::MaterialAsset =
                inf_asset::decode(&f.payload).expect("it decodes");
            assert_eq!(f.sidecar.guid.0, want, "the library and the level disagree");
            for (i, c) in [m.base_color.r, m.base_color.g, m.base_color.b]
                .into_iter()
                .enumerate()
            {
                assert!(
                    (c - mat.base_color[i]).abs() < 1e-6,
                    "{recipe}: channel {i} reads {c} and the .inf_mat says {}",
                    mat.base_color[i]
                );
                assert!(
                    c < 0.2,
                    "{recipe}: a road at {c} linear is the debug grey, not asphalt"
                );
            }
            assert!((m.roughness - mat.roughness).abs() < 1e-6);
            println!(
                "ROAD {recipe}: base_color ({:.3}, {:.3}, {:.3}) linear, roughness \
                 {:.2}, .inf_mat {want}",
                m.base_color.r, m.base_color.g, m.base_color.b, m.roughness
            );

            // ── the kerbs, the pavements and the paint (wave ROAD1) ──────────
            //
            // CERT1's CP-B3 measured that none of these existed. This is the
            // arm that keeps them existing: three entities, each with its own
            // mesh binding and its own surface, and the two paint entities
            // deliberately naming **no** `.inf_mat` — a 100 mm film is scalars,
            // and a texture of it would be 1.6 MB of one colour.
            for part in inf_gis::FURNITURE_PARTS {
                let g = road_part_entity_guid(name, part);
                let e = doc
                    .entity_of(g)
                    .unwrap_or_else(|| panic!("{recipe}: no {} entity", part.label()));
                let mr = doc
                    .world()
                    .world()
                    .get::<inf_ecs::components::MeshRef>(e)
                    .copied()
                    .unwrap_or_else(|| panic!("{recipe}: {} draws nothing", part.label()));
                assert_eq!(
                    mr.asset,
                    Some(inf_island::road_part_mesh_guid(name, part)),
                    "{recipe}: {} is bound to the wrong mesh",
                    part.label()
                );
                let pm = doc
                    .world()
                    .world()
                    .get::<inf_ecs::components::Material>(e)
                    .copied()
                    .unwrap_or_else(|| {
                        panic!(
                            "{recipe}: {} carries no Material, so it draws the 0.8 \
                             debug grey the road itself was on before ASSET0",
                            part.label()
                        )
                    });
                match part {
                    inf_gis::RoadPart::Kerb => assert_eq!(
                        pm.asset,
                        Some(crate::ground::ground_material_guid(
                            inf_material::ground::GroundKind::Concrete
                        )),
                        "{recipe}: the kerb binds the concrete set"
                    ),
                    _ => assert_eq!(
                        pm.asset, None,
                        "{recipe}: road paint is the scalars-only path on purpose \
                         (see `road_part_material`)"
                    ),
                }
                // Every furniture entity is `AlwaysLoaded`, like the road: a
                // kerb that streamed out while the carriageway stayed would be
                // a street that lost its edges at distance.
                assert!(
                    doc.world()
                        .world()
                        .get::<inf_ecs::components::AlwaysLoaded>(e)
                        .is_some(),
                    "{recipe}: {} is not AlwaysLoaded",
                    part.label()
                );
                println!(
                    "ROAD {recipe}: {} -> mesh {:?}, material {:?}, base ({:.2}, {:.2}, {:.2})",
                    part.label(),
                    mr.asset.map(|a| a.to_string()),
                    pm.asset.map(|a| a.to_string()),
                    pm.base_color.r,
                    pm.base_color.g,
                    pm.base_color.b
                );
            }
            // Yellow really is yellow and white really is white — the one thing
            // a GUID comparison cannot say, and the whole reason a driver knows
            // which side of the line to be on.
            let paint = |part| {
                doc.world()
                    .world()
                    .get::<inf_ecs::components::Material>(
                        doc.entity_of(road_part_entity_guid(name, part)).unwrap(),
                    )
                    .copied()
                    .unwrap()
                    .base_color
            };
            let w = paint(inf_gis::RoadPart::MarkingWhite);
            let y = paint(inf_gis::RoadPart::MarkingYellow);
            assert!(
                (w.r - w.b).abs() < 0.05 && w.r > 0.3,
                "{recipe}: the white line reads ({:.2}, {:.2}, {:.2})",
                w.r,
                w.g,
                w.b
            );
            assert!(
                y.r > y.g && y.g > y.b * 3.0,
                "{recipe}: the centre line reads ({:.2}, {:.2}, {:.2}) and yellow is \
                 what separates OPPOSING traffic",
                y.r,
                y.g,
                y.b
            );
        }
    }

    /// The fixture's level really is a level: it names the terrain, the biome
    /// set, an ocean, the water the design found and a player-controlled hero.
    #[test]
    fn the_fixture_level_carries_the_island_it_describes() {
        let Some(d) = design("samples/island-fixture/island.toml") else {
            println!("SKIP: no island fixture in this tree");
            return;
        };
        let doc = island_scene(&d);
        let name = d.recipe.name.as_str();

        // The geo-anchor reaches the document, which is what the sky's latitude
        // and every future import are read from.
        assert!(doc.geo().enabled);
        assert_eq!(doc.geo().crs, "EPSG:32610");

        // **And every number in it is the recipe's own, carried bit for bit.**
        // This is the I7 CI-red, as an arm: the three degrees used to be
        // inverted out of the easting/northing through `proj4rs`, which is the
        // platform's libm, so the committed level's `origin_latitude_deg` read
        // 49.34307562364773 where it was blessed and 49.34307562364772 on macOS
        // — one ulp, one byte, one red platform of three. A committed byte has
        // to trace to a committed decimal, and `assert_eq!` on an f64 is the
        // only comparison that says so.
        let g = doc.geo();
        let a = &d.recipe.anchor;
        assert_eq!(g.origin_easting_m, a.easting_m);
        assert_eq!(g.origin_northing_m, a.northing_m);
        assert_eq!(g.origin_height_m, a.height_m);
        assert_eq!(g.origin_latitude_deg, a.latitude_deg);
        assert_eq!(g.origin_longitude_deg, a.longitude_deg);
        assert_eq!(g.grid_convergence_deg, a.convergence_deg);
        assert_eq!(g.vertical_datum, a.vertical_datum);

        // The partition is on, at the terrain's own tile lattice.
        let s = doc.settings();
        assert!(s.partition.enabled);
        assert_eq!(s.partition.cell_size_m, ISLAND_CELL_SIZE_M);

        // The terrain names its asset and its palette and ships no tiles.
        let e = doc
            .entity_of(terrain_entity_guid(name))
            .expect("a ground entity");
        let t = doc
            .world()
            .world()
            .get::<inf_ecs::components::Terrain>(e)
            .expect("a Terrain component");
        assert_eq!(t.asset, Some(inf_island::terrain_guid(name)));
        assert_eq!(t.biome_set, Some(inf_island::biome_set_guid(name)));
        assert!(t.data.is_empty(), "a streamed terrain ships no tiles");
        assert_eq!(t.tile_resolution, d.recipe.grid.tile_resolution);

        // One ocean, every lake, and the bounded set of rivers.
        let waters: Vec<&inf_ecs::components::WaterBody> = doc
            .world()
            .world()
            .iter_entities()
            .filter_map(|e| e.get::<inf_ecs::components::WaterBody>())
            .collect();
        let oceans = waters
            .iter()
            .filter(|w| w.kind == inf_ecs::components::WaterKind::Ocean)
            .count();
        let lakes = waters
            .iter()
            .filter(|w| w.kind == inf_ecs::components::WaterKind::Lake)
            .count();
        let rivers = waters
            .iter()
            .filter(|w| w.kind == inf_ecs::components::WaterKind::River)
            .count();
        println!("WATER ENTITIES: {oceans} ocean, {lakes} lakes, {rivers} rivers");
        assert_eq!(oceans, 1, "an island needs exactly one unbounded sea");
        assert_eq!(lakes, d.network.lakes.len());
        assert_eq!(rivers, d.rivers(MAX_RIVER_BODIES).len());
        assert!(rivers <= MAX_RIVER_BODIES);
        assert!(rivers > 0, "the design found no reach worth a body");

        // Every river carries its own centreline on its own entity.
        for i in 0..rivers {
            let g = river_guid(name, i);
            let e = doc.entity_of(g).expect("a river entity");
            let sp = doc
                .world()
                .world()
                .get::<inf_ecs::components::Spline>(e)
                .expect("a river's centreline is the Spline on its own entity");
            assert!(sp.points.len() >= 2);
            assert!(!sp.closed);
        }

        // The hero is player-controlled, streams the world and stands above the
        // ground the design put under it.
        let e = doc.entity_of(hero_guid(name)).expect("a hero");
        let m = doc
            .world()
            .world()
            .get::<inf_ecs::components::CharacterMovement>(e)
            .expect("CharacterMovement");
        assert!(m.player_controlled);
        assert!(doc
            .world()
            .world()
            .get::<inf_ecs::components::StreamingSource>(e)
            .is_some());
        let tr = doc
            .world()
            .world()
            .get::<inf_ecs::components::Transform>(e)
            .expect("a transform");
        let start = d.start(START_LIFT_M);
        assert!((tr.translation.x - start.x).abs() < 1e-9);
        assert!((tr.translation.z - start.z).abs() < 1e-9);
        assert!(
            tr.translation.y > start.y,
            "the capsule's centre must be above its feet"
        );
        // …and the start is at a settlement, not at the world origin by accident.
        let site = d.recipe.sites.first().expect("a site");
        assert!((start.x - site.x).abs() < 1e-9 && (start.z - site.z).abs() < 1e-9);
        assert!(
            start.y > d.recipe.sea.level_m,
            "the hero starts under water"
        );
        println!(
            "START: ({:.1}, {:.1}, {:.1}) at {:?}",
            start.x, start.y, start.z, site.name
        );

        // The level's dependency closure names every asset the build writes --
        // and, since SK1c, every asset the STARTER CHARACTER ships. The three
        // character GUIDs enter through `SkeletalMesh.{mesh, skeleton}`,
        // `AnimStateMachine.sm` and `ActorClass`, which `level_dependencies`
        // already walks, so this is a closure that grew rather than a new
        // mechanism -- and it is what makes the cook copy the rig.
        let ids = crate::samples::starter_character_ids();
        let deps = crate::scene::serialize::level_dependencies(&doc);
        for want in [
            inf_island::terrain_guid(name),
            inf_island::road_mesh_guid(name),
            inf_island::biome_set_guid(name),
            ids.skeleton.unwrap().0,
            ids.mesh.unwrap().0,
            ids.machine.unwrap().0,
        ] {
            assert!(deps.contains(&want), "the level does not depend on {want}");
        }
        // **The controller is deliberately NOT in it.** `level_dependencies`
        // walks asset REFERENCES on components and `ActorClass` is not one of
        // them — a Blueprint class is code, reached by the cook's own scan, and
        // `samples/phase29-locomotion`'s committed sidecar lists its rig, body
        // and machine and not its `.inf_act` for exactly the same reason. Stated
        // here rather than left as an absence, because the `[content]` list in
        // the recipe is what puts the `.inf_act` in the island's project and
        // somebody reading this loop will wonder.
        assert!(
            !deps.contains(&ids.actor.unwrap().0),
            "an `ActorClass` has started entering the level's asset closure — good, \
             but the recipe's `[content]` list and this comment both assume it does not"
        );
    }

    /// **The island's hero is the starter character, built through the wizard's
    /// own door** (SK1c).
    ///
    /// Two halves, and the second is the one the swap exists for.
    ///
    /// *It is a character.* It carries a `SkeletalMesh` naming the committed rig
    /// and body, a machine that is `Some`, and the controller -- the four things
    /// the hand-rolled capsule had none of. `AnimStateMachine { sm: None }` and
    /// no `SkeletalMesh` is a hero that walks, draws nothing and poses nothing,
    /// which is what shipped until this wave.
    ///
    /// *It is the door's character.* Every field is compared against what
    /// `edit_create_character` builds at the same height and the same feet --
    /// `the_showcase_character_matches_the_wizard_door`'s discipline, applied to
    /// the second generator in the tree that used to hand-roll one. The capsule
    /// arithmetic agreed by coincidence before (two copies of the same three
    /// lines); it agrees by construction now, and this is what says so.
    #[test]
    fn the_island_hero_is_the_starter_character_the_wizard_would_build() {
        use inf_ecs::components::{
            ActorClass, AlwaysLoaded, AnimStateMachine, CharacterController3D, CharacterMovement,
            Collider3D, ColliderShape3DKind, RigidBody3D, SkeletalMesh, StreamingSource, Transform,
        };
        let Some(d) = committed_design(ISLAND_RECIPES[1]) else {
            eprintln!("SKIP: no committed fixture design");
            return;
        };
        let name = d.recipe.name.as_str();
        let doc = island_scene(&d);
        let e = doc.entity_of(hero_guid(name)).expect("a hero");
        let w = doc.world().world();
        let ids = crate::samples::starter_character_ids();

        // -- it is a character --
        assert_eq!(
            w.get::<SkeletalMesh>(e).map(|s| (s.skeleton, s.mesh)),
            Some((Some(ids.skeleton.unwrap().0), Some(ids.mesh.unwrap().0))),
            "the island hero carries no rig -- it is a capsule again"
        );
        assert_eq!(
            w.get::<AnimStateMachine>(e).and_then(|m| m.sm),
            Some(ids.machine.unwrap().0),
            "the island hero's machine is None, so it poses nothing"
        );
        assert_eq!(
            w.get::<ActorClass>(e).map(|a| a.0),
            Some(ids.actor.unwrap().0)
        );
        // …and the two the door does not insert, which the island must.
        assert!(
            w.get::<StreamingSource>(e).is_some() && w.get::<AlwaysLoaded>(e).is_some(),
            "the hero lost its streaming anchor -- the partition and the I3 \
             collider band both read it"
        );

        // -- it is the door's character, field by field --
        let mut door_doc = SceneDoc::new();
        let door_guid = door_doc.edit_create_character(
            "Hero",
            ids.skeleton.unwrap().0,
            ids.mesh.unwrap().0,
            ids.machine.unwrap().0,
            Some(crate::scene::doc::CharacterSkin::from_material(
                ids.material.unwrap().0,
                &crate::character::starter_skin_material(),
            )),
            d.start(START_LIFT_M),
            Some(ids.actor.unwrap().0),
            hero_height_m(),
        );
        let door = door_doc
            .world()
            .entity_of(door_guid)
            .expect("the door built one");
        let dw = door_doc.world().world();
        assert_eq!(
            w.get::<Collider3D>(e)
                .map(|c| (c.shape_kind, c.half_extents, c.radius)),
            dw.get::<Collider3D>(door)
                .map(|c| (c.shape_kind, c.half_extents, c.radius)),
            "the island's capsule is not the one the wizard would build"
        );
        assert_eq!(
            w.get::<Transform>(e).map(|t| t.translation),
            dw.get::<Transform>(door).map(|t| t.translation),
            "the island places its hero at a different height for the same feet"
        );
        let cm = |c: Option<&CharacterMovement>| {
            c.map(|c| {
                (
                    c.player_controlled,
                    c.stand_half_height_m,
                    c.crouch_half_height_m,
                    c.prone_half_height_m,
                )
            })
        };
        assert_eq!(
            cm(w.get::<CharacterMovement>(e)),
            cm(dw.get::<CharacterMovement>(door))
        );
        assert_eq!(
            w.get::<RigidBody3D>(e).map(|b| b.kind),
            dw.get::<RigidBody3D>(door).map(|b| b.kind)
        );
        assert!(
            w.get::<CharacterController3D>(e).is_some()
                && dw.get::<CharacterController3D>(door).is_some()
        );

        // ANTI-VACUITY: a capsule with real dimensions, derived from the STARTER
        // character's height rather than from a number this file used to choose.
        let c = w.get::<Collider3D>(e).expect("a capsule");
        assert_eq!(c.shape_kind, ColliderShape3DKind::Capsule);
        let h = hero_height_m();
        assert!(
            (h - 1.75).abs() < 1e-12,
            "the starter character's height moved: {h}"
        );
        assert!(
            (c.radius - (h * 0.15)).abs() < 1e-12 && c.half_extents.y > 0.3,
            "{c:?}"
        );
    }

    #[test]
    fn the_cover_document_is_slope_limited_and_island_scaled() {
        let d = island_cover_document(7);
        assert_eq!(d.layers.len(), 1);
        let r = &d.layers[0].rules[0];
        assert_eq!(r.scatter.base_density, ISLAND_SCATTER_DENSITY);
        assert_eq!(r.scatter.cell_size, ISLAND_SCATTER_CELL_M);
        assert_eq!(r.scatter.seed, 7);
        assert!(matches!(
            r.sampler,
            inf_pcg::SamplerDef::Slope { max_deg, .. } if (max_deg - 34.0).abs() < 1e-9
        ));
        assert_eq!(r.kinds.len(), 3);

        // **The arithmetic that bounds this density, and the measurement that
        // corrected it.**
        //
        // The island-wide candidate count is the number people reach for and it
        // is NOT the bound: the scatter is evaluated only where terrain is
        // resident, so what a frame pays for is the WORKING SET.
        //
        // This wave predicted the working set by scaling the shipped
        // instrument's 2 681 instances at 0.004 /m2 linearly to 13 405 at 0.02.
        // The instrument then measured **16 771** -- 25 % ABOVE the prediction,
        // which is the same thing as the prediction sitting 20 % BELOW the
        // measurement (the number the line below prints). A jittered per-cell
        // scatter does not divide evenly. The measurement is the number here;
        // the scaling is kept beside it as what it was, an estimate.
        //
        // **The assertion reads the DOCUMENT'S density, not a literal.** A
        // constant compared against a constant is a tautology the compiler folds
        // and clippy refuses; more to the point it would guard nothing. Scaling
        // the measurement by what `island_cover_document` actually authored
        // means the arm fires the day somebody raises the density past what the
        // CPU tier can draw, which is the only thing worth guarding here.
        const MEASURED_WORKING_SET: f64 = 16_771.0;
        const MEASURED_AT_DENSITY: f64 = 0.02;
        const SCALED_PREDICTION: f64 = 13_405.0;
        let land_m2 = 40.65e6;
        let island_wide = land_m2 * r.scatter.base_density;
        let working_set = MEASURED_WORKING_SET * (r.scatter.base_density / MEASURED_AT_DENSITY);
        println!(
            "SCATTER: {} /m2 is {island_wide:.0} candidates over {:.2} km2 of land, and -- the number that matters -- {working_set:.0} in the working set, MEASURED at {MEASURED_WORKING_SET:.0} (a linear scaling from 0.004 predicted {SCALED_PREDICTION:.0}, {:.0} % low)",
            r.scatter.base_density,
            land_m2 / 1e6,
            (1.0 - SCALED_PREDICTION / MEASURED_WORKING_SET) * 100.0
        );
        // The CPU scatter fallback's own ceiling. A tier that cannot reach the
        // GPU path draws a nearest-first subset past this, which is a different
        // world from the one the GPU tier draws. 16 771 is 25.6 % of it -- so
        // both tiers still draw the same island, with room for three more raises
        // of this size before they stop.
        const MAX_CPU_SCATTER_INSTANCES: f64 = 65_536.0;
        assert!(
            working_set < MAX_CPU_SCATTER_INSTANCES / 3.0,
            "{working_set:.0} instances in the working set is past a third of the CPU fallback's {MAX_CPU_SCATTER_INSTANCES:.0} ceiling -- the two tiers would stop drawing the same island"
        );
        assert!(
            island_wide > 500_000.0,
            "the density is back below what a walk over this island can see"
        );

        // A document-only envelope: no graph, so no grammar and no buildings.
        let p = island_cover_payload(7);
        assert!(p.graph_json.is_none(), "vegetation carries no grammar");
        assert_eq!(p.schema_version, inf_pcg::PcgAssetPayload::CURRENT_VERSION);
    }
    /// **The hero starts on the STREET, not inside a building** (wave FIX1).
    ///
    /// The author's demo showed a hero apparently standing in a wall, and the
    /// brief asked for the start to be moved to the nearest clear kerb. **It is
    /// already on the street and this arm is the measurement that says so** —
    /// which is why nothing moves it. `player_start` puts the hero at the first
    /// city site's exact centre, and `settlement::plan_site` lays its grid so
    /// that block `col = 0` begins `half_street` metres from that centre: the
    /// start stands in the middle of the crossroads, with a `setback_m` of lot
    /// inset on top of the reserve before any wall can begin.
    ///
    /// What was missing was any arm at all. The grid ladder is a search over
    /// pitches (`grid_for`), the site radius is authored, and nothing anywhere
    /// asserted that the two leave the start clear — so a recipe edit that
    /// shrank Harbour City would have put a building on the player, silently.
    /// The bound is `half_street`, taken from the settlement's own plan rather
    /// than restated, so it stays true when the ladder changes gear.
    #[test]
    fn the_island_start_stands_clear_of_every_settlement_block() {
        for rel in crate::island::ISLAND_RECIPES {
            let Some(d) = crate::island::committed_design(rel) else {
                continue;
            };
            let start = d.start(0.0);
            let at = glam::DVec2::new(start.x, start.z);
            let plans = crate::settlement::settlements(&d);
            assert!(!plans.is_empty(), "{rel} plans no settlements");
            let mut nearest = f64::MAX;
            let mut nearest_name = String::new();
            let mut inside = 0usize;
            let mut reserve = f64::MAX;
            for plan in &plans {
                // The reserve this settlement's own grid keeps: half a street.
                reserve = reserve.min(plan.street_m * 0.5);
                for b in &plan.blocks {
                    // Signed distance to an axis-aligned rectangle: negative
                    // inside, which is the number a "not inside a building"
                    // claim needs.
                    let d = (at - b.centre).abs() - b.half;
                    let outside = d.max(glam::DVec2::ZERO).length();
                    let sd = if d.x <= 0.0 && d.y <= 0.0 {
                        d.x.max(d.y)
                    } else {
                        outside
                    };
                    if sd < nearest {
                        nearest = sd;
                        nearest_name =
                            format!("{} {} {},{}", plan.name, b.archetype.name(), b.col, b.row);
                    }
                    if sd < 0.0 {
                        inside += 1;
                    }
                }
            }
            println!(
                "FIX1 start clearance — {rel}: start ({:.1}, {:.1}), nearest block `{nearest_name}` \
                 at {nearest:+.3} m, {} blocks over {} settlements, street reserve {reserve:.1} m",
                start.x,
                start.z,
                plans.iter().map(|p| p.blocks.len()).sum::<usize>(),
                plans.len()
            );
            assert_eq!(
                inside, 0,
                "{rel} builds {inside} block(s) OVER the player start"
            );
            assert!(
                nearest >= reserve - 1.0e-9,
                "{rel}'s start is {nearest:.3} m from `{nearest_name}`, inside the \
                 settlement's own {reserve:.1} m street reserve"
            );
        }
    }

    /// **The hero starts ON the ground, not a metre above it** (wave FIX1).
    ///
    /// [`START_LIFT_M`] was `1.0` and CERT1 measured its consequence twice: a
    /// **0.9883 m** settle on the fixture and **0.9769 m** on the shipped island,
    /// which is the lift and nothing else — the island spawned its hero a metre
    /// in the air and let it fall. The lift's own doc gave the reason ("a
    /// character placed exactly on the surface is one the first ground snap has
    /// to resolve out of the floor"), and the ground snap is the thing that
    /// exists to do exactly that: `CharacterController3D` resolves a penetration
    /// on its first step, which is a frame nobody sees, where a metre of fall is
    /// half a second everybody does.
    ///
    /// This arm is the value, pinned, plus the linearity the lift's one caller
    /// depends on. The fall it prevents is measured by `parity_cert`'s
    /// `the_islands_pawn_comes_to_rest_on_the_island`, over a real cooked island.
    #[test]
    fn the_island_start_is_the_ground_the_design_committed() {
        assert_eq!(
            crate::island::START_LIFT_M,
            0.0,
            "a hero lifted off its own ground falls in front of the player"
        );
        for rel in crate::island::ISLAND_RECIPES {
            let Some(d) = crate::island::committed_design(rel) else {
                continue;
            };
            let feet = d.start(crate::island::START_LIFT_M);
            let ground = d.start(0.0);
            assert_eq!(feet.y, ground.y, "{rel}: the start is off its own ground");
            assert!(
                feet.y > d.recipe.sea.level_m,
                "{rel}: the hero starts under water"
            );
        }
    }
}
