//! **The island, driven — PIE == shipping over 51 km² of real ground** (wave I7).
//!
//! # What this gate is and what it is not
//!
//! It runs the **CI-scale island**, because the shipped one's terrain is 549.9 MB
//! and is not committed (342.7 MB was wave I7's figure; wave TER2b's detail band
//! moved it, and the I8a audit re-measured it off `build.report.summary()`).
//! Everything else is the shipped path: the same recipe
//! format, the same scene generator, the same `.inf_terrain`, the same
//! partition, the same water and the same biome binding — so a change that
//! breaks the island breaks this.
//!
//! # The claim
//!
//! A drive across a streamed island, with the terrain paging under the wheels
//! and the partition activating cells as the source moves, produces **the same
//! bytes** in the shipped player and in the editor's PIE. That is the property
//! every gate in this repository exists to protect, met on the largest world it
//! has: if the terrain streamer, the cell activation or the biome-bound
//! population were a function of anything but sim state, the two would diverge.
//!
//! **And the same forest** (island wave I7b). `Terrain::biome_population` is
//! `#[serde(skip)]`, so it reaches no state fold and two hosts growing different
//! vegetation would have compared equal at every step for ever. The drive folds
//! it separately, out and back, so the ground — and the vegetation on it —
//! pages in **and** out under the comparison.
//!
//! # Why the drive is scripted through the sim and not through a camera
//!
//! The collider band, the cell activation and the terrain's sim residency all
//! anchor on `StreamingSource` entities — sim state. A camera-driven trace would
//! be measuring the renderer. The hero **is** the streaming source here, and the
//! script moves it.

use std::path::{Path, PathBuf};

use uuid::Uuid;

use inf_player::budget::{CITY_STEP_BUDGET_MS, LOAD_BUDGET_MS};
use inf_player::runtime_sim::RuntimeSim;
use inf_project::ProjectManifest;

/// The recipe CI builds.
fn fixture_recipe() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../samples/island-fixture/island.toml")
}

/// The fixed step, matching the level's own `sim_hz`.
const HZ: f64 = 60.0;

/// How many fixed steps the drive runs.
const STEPS: u64 = 900;

/// How far the drive advances per step, metres. 0.4 m at 60 Hz is 24 m/s — a car
/// on a highway, and 360 m over the run, which crosses this fixture's own
/// partition cell and several terrain pages.
const STEP_M: f64 = 0.4;

/// Build the island's project: the recipe's own build, into a temp directory.
///
/// **This is `inf island build`'s own door** (`write_content`), not a
/// reimplementation of it — so a gate that passed while the command produced
/// something different is impossible.
fn build_project(tmp: &Path) -> PathBuf {
    let recipe =
        inf_island::IslandRecipe::load(&fixture_recipe()).expect("the fixture recipe loads");
    let build = inf_island::build_island(&recipe, &inf_island::BuildOptions::default())
        .expect("the fixture island builds");
    // The design must be non-vacuous BEFORE anything is compared: two empty
    // worlds agree perfectly.
    assert!(
        build.report.land_km2 > 0.5,
        "the fixture has only {:.3} km2 of land",
        build.report.land_km2
    );
    assert!(!build.network.streams.is_empty(), "no water was derived");
    assert!(!build.routes.is_empty(), "no road was designed");

    let proj = tmp.join("island");
    ProjectManifest::new(&recipe.name, "blank-3d")
        .save(&proj)
        .expect("the project scaffolds");
    inf_island::write_content(&build, &proj.join("Content")).expect("the island's content writes");
    proj
}

/// Cook it, exactly as `inf cook` does.
fn cook(tmp: &Path) -> PathBuf {
    let proj = build_project(tmp);
    let out = tmp.join("out");
    inf_packager::cook(&proj, &out, &inf_packager::CookOptions::default())
        .expect("the island cooks");
    out
}

/// **The shipping side**: a sim off the cooked pack, **with cell streaming
/// attached** — exactly as `inf_player::run_headless` does.
///
/// Attaching it is not optional and the first draft of this gate found out why:
/// a partitioned level's cooked `.inf_lvl` carries **no entities at all** (they
/// are in the derived `.inf_part`), so a shipping sim without the streamer is a
/// world holding only what `AlwaysLoaded` kept — six records against the
/// editor's fifteen. The two agreed for 411 steps and then did not, which reads
/// like a streaming defect and is a gate that forgot to boot the streamer.
fn pack_sim(pack: &Path) -> RuntimeSim {
    let source = inf_player::level::PackLevelSource::open(pack).expect("the pack opens");
    let mut built = inf_player::build_world_from_pack(&source).expect("the world builds");
    let partition = built.take_partition();
    let pcg = built.pcg_context();
    let mut sim = inf_player::sim_from_built(built);
    inf_player::attach_cell_streaming(&mut sim, &partition, pcg);
    // …**and the TERRAIN streamer**, which `run_headless` attaches on the next
    // line and this gate did not. Without it the island's 4.6 MB of pages never
    // move: the `Terrain` component keeps the empty working set a streamed level
    // ships and every `height_at` in the drive answers off nothing. The gate's
    // own headline says "with the terrain paging under the wheels", and
    // `both_hosts_really_streamed` is what now makes that a measurement.
    inf_player::attach_terrain_streaming(
        &mut sim,
        &inf_player::TerrainContent::Pack(source.clone()),
    );
    sim
}

/// **The editor side**: the loose `.inf_lvl` the author saved, binned by the same
/// Ring-0 function the cook used.
///
/// This is the pair P16.5's own gate compares, and it is the right one for a
/// partitioned level: a `ScenePayload` carries **no partition** (see
/// `a_scene_payload_carries_no_partition`), so the PIE wire is not the editor's
/// authoritative reading of a streamed world — the document is.
///
/// **Built the way `build_world`'s own `--level` arm builds it**, and that is not
/// tidiness: the first draft handed the builder `with_defaults(Vec::new())` and
/// nothing else, so the loose host had no biome sets, no PCG payloads and no
/// terrain resolver where the pack host had all three. Two hosts compared for
/// byte equality must be given the same world to disagree about, or the equality
/// is between one real reading and one impoverished one.
fn loose_sim(content: &Path, slug: &str) -> RuntimeSim {
    let source = inf_player::level::DevDirLevelSource::new(content.join(format!("{slug}.inf_lvl")));
    let terrains = inf_player::level::terrain_paths_by_guid_from_dir(content);
    let pcg_terrains = terrains.clone();
    let (skeletons, clips, machines) = inf_player::level::load_anim_assets_from_dir(content);
    let builder = inf_player::level::InfSceneWorldBuilder::with_defaults(
        inf_player::level::load_actor_classes_from_dir(content),
    )
    .with_pcgs(inf_player::level::load_pcg_payloads_by_guid_from_dir(
        content,
    ))
    .with_biome_sets(inf_player::level::load_biome_sets_by_guid_from_dir(content))
    // **The hero's rig, its machine and its clips** (SK1c). This function's own
    // doc already carried the rule -- *two hosts compared for byte equality must
    // be given the same world to disagree about, or the equality is between one
    // real reading and one impoverished one* -- and the anim index was the third
    // thing it was missing, invisible for as long as the island's hero was a
    // capsule with `AnimStateMachine { sm: None }` and nothing to pose. The pack
    // host gets these from the cook's own index; this one reads the same content
    // root the recipe's `[content]` list filled.
    .with_anim_assets(skeletons, clips, machines)
    // **The club loop** (VEN1b), for the anim index's reason exactly and it is
    // the FOURTH thing this function was missing: the pack host gets its
    // `.inf_audio` payloads from the cook's own index, so without this line the
    // two hosts would issue the same `Play` and one of them would resolve it.
    // The command stream is the observable contract, and it stays comparable
    // either way — what this buys is that both hosts are given the same world.
    .with_audio(inf_player::level::load_audio_assets_from_dir(content))
    .with_terrain_resolver(std::sync::Arc::new(move |g| {
        inf_player::level::terrain_source_from_file(pcg_terrains.get(&g)?).ok()
    }));
    let mut built = inf_player::level::load(&source, &builder).expect("the loose level builds");
    let partition = built.take_partition();
    let pcg = built.pcg_context();
    let mut sim = inf_player::sim_from_built(built);
    inf_player::attach_cell_streaming(&mut sim, &partition, pcg);
    inf_player::attach_terrain_streaming(&mut sim, &inf_player::TerrainContent::Dir(terrains));
    sim
}

/// What a host's two streamers actually did: `(cell activations, cell
/// deactivations, cells resident, sim-resident level-0 pages, page loads)`.
///
/// # Why the gate needs this and could not do without it
///
/// **Mutation-measured, and it is the reason this function exists.** Deleting
/// `attach_cell_streaming` from *one* host reds the byte compare — that is the
/// wave's own finding D8. Deleting it from **both** left every arm of this file
/// green: the coverage check still found one terrain, two water bodies and one
/// hero (they are all `AlwaysLoaded`), the trace still had 900 distinct states
/// (the drive moves the hero itself), and two hosts that both refuse to stream
/// agree perfectly. A gate whose subject is streaming has to assert that
/// streaming *happened*, not merely that two readings of it match.
fn streaming_counters(sim: &RuntimeSim) -> (u64, u64, usize, usize, u64) {
    let c = sim.cell_streaming().stats();
    let t = sim.terrain_streaming().stats();
    (
        c.activations,
        // …and the DEactivations, since island wave I7b: the drive turns round,
        // so a cell that streamed in streams back out, and "1 resident at the
        // end" stopped being the reading that says the partition worked.
        c.deactivations,
        c.cells_resident,
        t.sim_resident_level0,
        t.loads,
    )
}

/// **Every asset in a built project's content root, by GUID.**
///
/// The sidecars are the index — a `.toml` beside every payload naming its GUID —
/// which is exactly what `AssetDb`'s own scan reads. A name table here would be
/// a second place the starter character's identity is written down, and the two
/// would disagree the first time a file was renamed.
fn content_assets(content: &Path) -> std::collections::BTreeMap<Uuid, PathBuf> {
    let mut out = std::collections::BTreeMap::new();
    let Ok(dir) = std::fs::read_dir(content) else {
        return out;
    };
    for entry in dir.filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("toml") {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let Ok(table) = text.parse::<toml::Table>() else {
            continue;
        };
        let Some(guid) = table
            .get("guid")
            .and_then(|v| v.as_str())
            .and_then(|s| Uuid::parse_str(s).ok())
        else {
            continue;
        };
        // `Foo.inf_skel.toml` -> `Foo.inf_skel`, which is the payload it indexes.
        let payload = path.with_extension("");
        if payload.exists() {
            out.insert(guid, payload);
        }
    }
    out
}

/// **The PIE side**: the payload the editor really builds, through
/// `sim_from_payload` — the one PIE boot seam the real `--pie` subprocess takes.
fn pie_sim(proj: &Path) -> RuntimeSim {
    let content = proj.join("Content");
    let recipe =
        inf_island::IslandRecipe::load(&fixture_recipe()).expect("the fixture recipe loads");
    let slug = inf_island::slug(&recipe.name);
    let doc = inf_editor_core::scene::serialize::load(&content.join(format!("{slug}.inf_lvl")))
        .expect("the island level loads");

    // **The terrain is named, not read** (wave GTA1, `ScenePayload` v12). The
    // editor hands the PIE player a path for exactly this asset kind, so this
    // gate builds its payload the way the editor does; see the frame-size arm
    // below for what carrying it inline cost.
    let terrain_file = content.join(format!("{slug}.inf_terrain"));
    assert!(terrain_file.is_file(), "the built terrain is on disk");
    let biomes = std::fs::read(content.join(format!("{slug}.inf_biomes")))
        .expect("the built biome set is on disk");
    let pcg = std::fs::read(content.join(format!("{slug}Cover.inf_pcg")))
        .expect("the cover graph is on disk");
    let mesh = std::fs::read(content.join(format!("{slug}Roads.inf_mesh")))
        .expect("the road mesh is on disk");

    let t_guid = inf_island::terrain_guid(&recipe.name);
    let b_guid = inf_island::biome_set_guid(&recipe.name);
    let p_guid = inf_island::cover_pcg_guid(&recipe.name);
    let m_guid = inf_island::road_mesh_guid(&recipe.name);

    // **The character's assets** (SK1c). The hero is `samples/starter-character`
    // now, so the payload has to carry the rig, the machine and the clips or a
    // `--pie` preview poses nothing while the shipped build poses 161 bones.
    //
    // **Nothing drives this host** (SK1c audit, M1). The first draft of this
    // comment said "the state comparison below said so at step 0"; it did not
    // and could not — that comparison is `pack_sim` against `loose_sim`, and the
    // sim this function returns is only ever *counted*. The counts below are the
    // whole of this seam's cover, which is why the class is counted with the
    // rest.
    //
    // Read off the SIDECARS rather than from a hard-coded name table, which is
    // how the editor's own asset database finds them: the recipe's `[content]`
    // list copies the whole character into `Content/`, and every payload there
    // carries a `.toml` naming its GUID. A table of seven file names here would
    // be a second place the character's identity is written down.
    let assets = content_assets(&content);
    let read_asset = |g: Uuid| assets.get(&g).map(|p| std::fs::read(p).expect("an asset"));

    let payload = inf_editor_core::pie::build_scene_payload(
        &doc,
        // resolve (blueprint class), pcg, anim, biome_set, voxel, terrain, mesh,
        // bytes — in that order. Named here because eight closures of the same
        // shape are eight chances to mis-order them, and the first draft did:
        // it put the terrain where the biome set goes and the payload came back
        // with `0 terrain(s)`, which the non-vacuity assertion below caught.
        |g| {
            read_asset(g)
                .and_then(|b| serde_json::from_slice::<inf_blueprint::BlueprintClass>(&b).ok())
        },
        |g| (g == p_guid).then(|| pcg.clone()),
        read_asset,
        |g| (g == b_guid).then(|| biomes.clone()),
        |_| None,
        |g| (g == t_guid).then(|| inf_editor_core::pie::TerrainRef::Path(terrain_file.clone())),
        |g| {
            if g == m_guid {
                Some(mesh.clone())
            } else {
                read_asset(g)
            }
        },
        read_asset,
        |_| None,
        HZ as u32,
        false,
    )
    .expect("the payload builds");

    // **Non-vacuity at the payload.** A payload carrying no terrain would boot a
    // world with no ground, and two hosts with no ground agree perfectly.
    println!(
        "PAYLOAD: {} terrain path(s), {} inline terrain(s), {} biome set(s), {} pcg(s), {} mesh(es)",
        payload.terrain_paths.len(),
        payload.terrains.len(),
        payload.biome_sets.len(),
        payload.pcgs.len(),
        payload.meshes.len()
    );
    assert_eq!(
        payload.terrain_paths.len(),
        1,
        "the terrain must ride the wire — as a path (v12)"
    );
    assert!(
        payload.terrains.is_empty(),
        "the terrain rode inline as well as by path: it must ride exactly one of \
         the two routes, or the player has two sources for one guid"
    );

    // **THE FRAME FITS, AND IT IS THE FRAME THAT REFUSED** (wave GTA1).
    //
    // `write_msg` is the door that killed Play on the island: a `.inf_terrain`
    // carried inline made the payload larger than `MAX_FRAME_LEN` and the writer
    // refused the frame, so pressing Play produced one line in the status bar and
    // no player at all. Measured on the SHIPPED island — the one this fixture
    // stands in for, whose terrain is not committed — that was 549.9 MB against a
    // 268 435 456-byte cap.
    //
    // The assertion is not "under the cap" alone, which this small fixture would
    // pass either way.
    //
    // # It was a size proxy, and wave IASSET2 broke it by growing the CONTENT
    //
    // It used to be "the whole frame is smaller than the terrain file it names",
    // on the argument that this "is false for any payload carrying the bytes, at
    // any island size". True — and it is also false for a payload that carries
    // nothing of the kind and simply got bigger for its own reasons. IASSET2
    // moved the ground library's normal maps from BC1 to BC5 (worst per-channel
    // error 122 of 255 down to 17), which added 1 812 608 B of texture to a
    // payload that legitimately carries textures, and the frame crossed a
    // terrain file it had nothing to do with: **7 737 553 B against 7 043 328**,
    // with `0 inline terrain(s)` printed two lines above.
    //
    // So the proxy is replaced by the thing it was standing in for: **the
    // terrain file's own bytes must not appear in the frame.** A window from the
    // middle of the file (past the header, where the content is) is searched for
    // directly. That is non-vacuous on a fixture, independent of how large the
    // payload's legitimate content grows, and it is what "the ground is riding
    // inline" actually means.
    let mut frame = Vec::new();
    inf_runtime::pie::write_msg(
        &mut frame,
        &inf_runtime::pie::EditorToPlayer::LoadScene(Box::new(payload.clone())),
    )
    .expect("the PIE frame writes — this is the call that refused before v12");
    let terrain_len = std::fs::metadata(&terrain_file)
        .expect("terrain metadata")
        .len();
    println!(
        "PAYLOAD FRAME: {} B against a {} B cap; the terrain it names is {} B",
        frame.len(),
        inf_runtime::pie::MAX_FRAME_LEN,
        terrain_len
    );
    assert!(
        frame.len() < inf_runtime::pie::MAX_FRAME_LEN,
        "the PIE frame is over the cap at {} B",
        frame.len()
    );
    // THE DIRECT CHECK: a 64-byte window from the middle of the terrain file
    // must not occur anywhere in the frame. Scanned by first-byte prefilter so
    // it is linear rather than quadratic over a multi-megabyte frame.
    let terrain_bytes = std::fs::read(&terrain_file).expect("terrain bytes");
    assert!(
        terrain_bytes.len() > 4096,
        "the fixture's terrain is {} B — too small for the window below to be a \
         fingerprint rather than a coincidence",
        terrain_bytes.len()
    );
    let needle = &terrain_bytes[terrain_bytes.len() / 2..terrain_bytes.len() / 2 + 64];
    let carried = frame
        .iter()
        .enumerate()
        .filter(|(i, b)| **b == needle[0] && *i + needle.len() <= frame.len())
        .any(|(i, _)| &frame[i..i + needle.len()] == needle);
    assert!(
        !carried,
        "the terrain file's own bytes are inside the {} B PIE frame — the ground \
         is riding inline again and the shipped island cannot be played",
        frame.len()
    );
    // ANTI-VACUITY: the window really is findable when it IS carried, so the
    // assertion above is about the frame and not about a needle that never
    // matches anything.
    assert!(
        terrain_bytes
            .iter()
            .enumerate()
            .filter(|(i, b)| **b == needle[0] && *i + needle.len() <= terrain_bytes.len())
            .any(|(i, _)| &terrain_bytes[i..i + needle.len()] == needle),
        "the search cannot find the needle in the file it came from"
    );
    assert_eq!(
        payload.biome_sets.len(),
        1,
        "the palette must ride the wire"
    );
    assert_eq!(payload.pcgs.len(), 1, "the cover graph must ride the wire");
    // **And the hero's rig rides it too** (SK1c). Without this the PIE side
    // publishes no pose at all where the shipping side publishes 6 476 bytes of
    // one. One skeleton, one machine, the machine's three clips reached through
    // the transitive hop, **and the controller class**.
    //
    // **These four assertions are the ONLY cover this payload's character has**
    // (SK1c audit, M1). The first draft of this comment said the step-0 state
    // comparison would notice — it would not, and cannot: the drive gate builds
    // its two hosts from `pack_sim` and `loose_sim`, and `pie_sim` is used here
    // and nowhere else, which the arm below this function says in as many words.
    // A payload seam nothing drives is a payload seam whose only witness is what
    // is counted right here, so the class is counted too: reverting the
    // blueprint resolver alone left all six arms green.
    println!(
        "PAYLOAD CHARACTER: {} skeleton(s), {} machine(s), {} clip(s), {} class(es)",
        payload.skeletons.len(),
        payload.machines.len(),
        payload.clips.len(),
        payload.classes.len()
    );
    assert_eq!(
        payload.skeletons.len(),
        1,
        "the hero's rig must ride the wire"
    );
    assert_eq!(
        payload.machines.len(),
        1,
        "the hero's machine must ride the wire"
    );
    assert_eq!(
        payload.clips.len(),
        3,
        "the machine's clips must ride the wire, or PIE poses every state at rest"
    );
    assert_eq!(
        payload.classes.len(),
        1,
        "the hero's controller class must ride the wire — it is the `.inf_act` \
         the recipe's `[content]` list copies and the one thing in the character \
         that `level_dependencies` does NOT reach, so nothing else would notice"
    );

    inf_player::sim_from_payload(&payload)
        .expect("the PIE world builds")
        .sim
}

/// One host's reading of the drive: the state fold per step, **and** the
/// vegetation the ground grew under it.
///
/// The two are separate because `Terrain::biome_population` is `#[serde(skip)]`
/// — it is what the projector draws, never what the sim reads back — so it does
/// not reach `state_bytes` and no amount of comparing state folds would notice
/// two hosts growing different forests. It is folded here instead, which is
/// where the claim belongs: **the vegetation is a function of the resident
/// ground, and the resident ground is sim state.**
///
/// # Why it RETAINS every step, and what that costs (re-aimed, wave NPC1a)
///
/// `states` holds all 900 steps rather than folding them into a running hash,
/// and that is load-bearing rather than lazy: the anti-vacuity arm below asks how
/// many of the 900 are DISTINCT, which is a question about a set and cannot be
/// asked of a hash chain. A gate that compared two rolling digests would pass
/// perfectly on two hosts that both recorded nothing happening.
///
/// The cost is a function of the crowd, which is why this paragraph is dated:
/// a state is 403 B before the hero was a character, 6 879 B after it (SK1c), and
/// **35 366 B with NPC1a's 24-agent crowd standing beside it** — 31.8 MB retained
/// per host, 63.6 MB for the pair. That is affordable and it does not scale: at
/// the thousand agents `crowd_sweep.rs` measures, a state is about 1.1 MB and
/// 900 of them is a gigabyte. **A crowd gate at N in the hundreds folds instead
/// of retaining, and keeps a separate distinctness reading over a sampled
/// prefix** — stated here rather than discovered by an out-of-memory run.
struct Trace {
    states: Vec<Vec<u8>>,
    /// Per step: a digest of every instance's position bits and kind.
    veg: Vec<u128>,
    /// Per step: how many instances stood.
    veg_len: Vec<usize>,
    /// Per step: how many level-0 tiles the simulation held.
    tiles: Vec<usize>,
    /// Per step: agents at each sim-LOD tier (NPC1a), and the running re-tier
    /// count. All zeroes on a drive with no population, which is every arm in
    /// this file but the crowd one.
    crowd: Vec<[usize; 4]>,
    retiered: Vec<u64>,
}

/// Fold a terrain's population into a comparable digest — **positions, not a
/// count** (the I1 law): two forests of the same size in different places must
/// not compare equal.
fn veg_digest(sim: &RuntimeSim) -> (u128, usize) {
    let mut h = xxhash_rust::xxh3::Xxh3::new();
    let mut n = 0usize;
    let world = sim.world().world();
    let mut per_terrain: Vec<(uuid::Uuid, Vec<u8>)> = Vec::new();
    for e in world.iter_entities() {
        let (Some(g), Some(t)) = (
            e.get::<inf_ecs::Guid>(),
            e.get::<inf_ecs::components::Terrain>(),
        ) else {
            continue;
        };
        let mut bytes = Vec::with_capacity(t.biome_population.len() * 28);
        for i in &t.biome_population {
            bytes.extend_from_slice(&i.position.x.to_bits().to_le_bytes());
            bytes.extend_from_slice(&i.position.y.to_bits().to_le_bytes());
            bytes.extend_from_slice(&i.position.z.to_bits().to_le_bytes());
            bytes.extend_from_slice(&i.kind.to_le_bytes());
            // Wave TER2b: and the MESH, which is what the instance actually
            // draws. Two populations that agree on every position and differ on
            // which prop stands there are two different worlds, and before this
            // line the fold could not tell them apart.
            bytes.extend_from_slice(&i.mesh.map_or(0u128, |m| m.as_u128()).to_le_bytes());
        }
        n += t.biome_population.len();
        per_terrain.push((g.0, bytes));
    }
    per_terrain.sort_by_key(|(g, _)| *g);
    for (g, bytes) in per_terrain {
        h.update(g.as_bytes());
        h.update(&bytes);
    }
    (h.digest128(), n)
}

/// How many level-0 terrain tiles the **simulation** holds right now.
fn sim_tiles(sim: &RuntimeSim) -> usize {
    sim.world()
        .world()
        .iter_entities()
        .filter_map(|e| e.get::<inf_ecs::components::Terrain>())
        .map(|t| t.data.tile_count())
        .sum()
}

/// The drive: a run east and back again, sampled every step.
///
/// Deterministic and positional — a *place*, not a time, which is P29's own
/// lesson. Every step the streaming source is moved and the sim advanced, and
/// the sim's own residency sync is what pages the ground.
///
/// **It turns round half way** (island wave I7b), and that is not decoration:
/// out and back is what makes the ground page **in and out**, and what makes the
/// second half of the drive re-enter tiles the first half already visited. A
/// population that depended on the order its ground arrived in — P21's
/// first-sight hazard, which the per-tile memo is keyed against — would read
/// differently on the way home.
fn drive(sim: &mut RuntimeSim, from: glam::DVec3) -> Trace {
    let hero = hero_entity(sim).expect("the island has a player-controlled hero");
    let mut t = Trace {
        states: Vec::with_capacity(STEPS as usize),
        veg: Vec::with_capacity(STEPS as usize),
        veg_len: Vec::with_capacity(STEPS as usize),
        tiles: Vec::with_capacity(STEPS as usize),
        crowd: Vec::with_capacity(STEPS as usize),
        retiered: Vec::with_capacity(STEPS as usize),
    };
    for step in 0..STEPS {
        // Out along +x and back — twice the step so the turn is still 360 m out
        // — with a slow drift along +z so **no two steps stand in the same
        // place**. Without the drift the way home would repeat the way out and
        // the "900 distinct states" anti-vacuity arm would be measuring a
        // palindrome rather than a world.
        let out = step.min(STEPS - step);
        let p = glam::DVec3::new(
            from.x + out as f64 * 2.0 * STEP_M,
            from.y,
            from.z + step as f64 * 0.05,
        );
        set_hero(sim, hero, p);
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
        t.states.push(sim.state_bytes());
        let (d, n) = veg_digest(sim);
        t.veg.push(d);
        t.veg_len.push(n);
        t.tiles.push(sim_tiles(sim));
        let c = sim.crowd_stats();
        t.crowd.push(c.per_tier);
        t.retiered.push(c.retiered);
    }
    t
}

fn hero_entity(sim: &RuntimeSim) -> Option<inf_ecs::Entity> {
    let world = sim.world().world();
    let mut found = None;
    for e in world.iter_entities() {
        if e.get::<inf_ecs::components::CharacterMovement>()
            .is_some_and(|m| m.player_controlled)
        {
            found = Some(e.id());
        }
    }
    found
}

fn set_hero(sim: &mut RuntimeSim, e: inf_ecs::Entity, p: glam::DVec3) {
    if let Some(mut t) = sim
        .world_mut()
        .world_mut()
        .get_mut::<inf_ecs::components::Transform>(e)
    {
        t.translation = inf_ecs::math::Vec3d::new(p.x, p.y, p.z);
    }
}

/// Where the drive starts: the design's own player start, lifted clear.
fn start() -> glam::DVec3 {
    let recipe =
        inf_island::IslandRecipe::load(&fixture_recipe()).expect("the fixture recipe loads");
    let design = inf_island::read_design(&recipe).expect("the design reads");
    let s = design.start(inf_editor_core::island::START_LIFT_M);
    glam::DVec3::new(s.x, s.y + 2.0, s.z)
}

/// **THE HEADLINE.** The same drive, byte for byte, on both hosts.
///
/// Un-fix mutations this is armed against: a terrain streamer that read a camera
/// rather than a streaming source; a cell activation keyed on anything but sim
/// state; a biome-bound population evaluated differently by the two boot paths.
#[test]
fn pie_equals_shipping_on_an_island_drive() {
    let tmp = tempfile::tempdir().expect("a temp dir");
    let pack = cook(tmp.path());
    let proj = tmp.path().join("island");

    let recipe =
        inf_island::IslandRecipe::load(&fixture_recipe()).expect("the fixture recipe loads");
    let slug = inf_island::slug(&recipe.name);
    let from = start();
    let mut ship = pack_sim(&pack);
    let mut pie = loose_sim(&proj.join("Content"), &slug);

    // **Coverage first**, so two identical empty worlds cannot agree their way
    // through: both hosts must have the ground, the water and the population.
    for (who, sim) in [("shipping", &ship), ("pie", &pie)] {
        let world = sim.world().world();
        let terrains = world
            .iter_entities()
            .filter(|e| e.get::<inf_ecs::components::Terrain>().is_some())
            .count();
        let waters = world
            .iter_entities()
            .filter(|e| e.get::<inf_ecs::components::WaterBody>().is_some())
            .count();
        let heroes = world
            .iter_entities()
            .filter(|e| {
                e.get::<inf_ecs::components::CharacterMovement>()
                    .is_some_and(|m| m.player_controlled)
            })
            .count();
        println!("{who}: {terrains} terrain(s), {waters} water bod(ies), {heroes} hero(es)");
        assert_eq!(terrains, 1, "{who} has no ground");
        assert!(waters >= 2, "{who} has {waters} water bodies");
        assert_eq!(heroes, 1, "{who} has no player");
    }

    // What the two had paged before anything moved, so the numbers below are
    // what the DRIVE did rather than what the boot did.
    let (ship0, pie0) = (streaming_counters(&ship), streaming_counters(&pie));

    let a = drive(&mut ship, from);
    let b = drive(&mut pie, from);
    assert_eq!(a.states.len(), STEPS as usize);
    assert_eq!(b.states.len(), STEPS as usize);

    // …and the trace is not a constant, or the comparison below is between two
    // recordings of nothing happening.
    let distinct: std::collections::BTreeSet<&Vec<u8>> = a.states.iter().collect();
    println!(
        "DRIVE: {} steps of {STEP_M} m out and back = {:.0} m, {} distinct \
         states, {} bytes a state",
        STEPS,
        STEPS as f64 * STEP_M,
        distinct.len(),
        a.states[0].len()
    );
    assert!(
        distinct.len() > STEPS as usize / 2,
        "only {} of {STEPS} states differ — the drive is not moving the world",
        distinct.len()
    );

    for (i, (x, y)) in a.states.iter().zip(&b.states).enumerate() {
        assert_eq!(
            x, y,
            "PIE and shipping diverged at step {i} of {STEPS} — the island's \
             streaming or its population is a function of something other than \
             sim state"
        );
    }
    println!("PIE == SHIPPING over {STEPS} steps of an island drive");

    // **AND THE HERO IS A CHARACTER** (SK1c). The comparison above is a byte
    // equality, and a byte equality is blind to two hosts posing NOTHING
    // identically — which is precisely what this gate did for its whole life
    // before this wave, because the island's hero was a capsule carrying
    // `AnimStateMachine { sm: None }` and no `SkeletalMesh`.
    //
    // So the pose section is measured rather than inferred. **6 476 bytes** is
    // SK1a's arithmetic for a 161-bone rig — a 36-byte header (the entity's GUID,
    // its skeleton's GUID and a joint count) plus 40 bytes a joint — and it is
    // pinned as the number rather than as `> 0` for the reason SK1b's grip gate
    // pins the same one: a rig that silently lost its side tables, or a hero that
    // quietly went back to being a capsule, would still be "greater than zero"
    // on one host and equal on both.
    //
    // It is also the whole of this wave's cost on this trace: the drive went from
    // 403 bytes a state to 6 879.
    //
    // **NPC1d re-aims it from a NUMBER to a MULTIPLE**, and the reason is the
    // wave working rather than the arm breaking. The island now derives a
    // population from its own settlements, so the pose store is a *town*: the
    // drive published 2 137 080 bytes, which is 330 × 6 476 — the hero plus
    // Harbour City's 329 residents, every one of them a 161-bone character. The
    // claim the arm exists for is unchanged and is now stronger: not "the store
    // holds one 161-bone character" but "**every character in the store is a
    // 161-bone character**", which a hero that quietly went back to being a
    // capsule still fails, and which a crowd of them cannot satisfy by accident.
    const POSED_BYTES: usize = 36 + 161 * 40;
    let mut posed_counts = Vec::new();
    let mut posed_ceilings = Vec::new();
    for (who, sim) in [("shipping", &mut ship), ("pie", &mut pie)] {
        let bytes = inf_ecs::pose::pose_state_bytes(sim.world());
        // **And the multiple is bounded by something that knows what it is**
        // (NPC1d audit). A multiple alone says "every character in the store has
        // the island's rig" and nothing about how MANY there should be, so an
        // unexplained doubling of the pose store would sail through it. The
        // hero plus the society the level derived is the only source of
        // characters on this level, and that is the ceiling.
        //
        // **Plus the traffic's own drivers** (wave VEH2b). A `Full` traffic car
        // that is on a leg of its day carries a person, built from the level's
        // own archetype through `crowd::spawn_body` — a real character with a
        // real capsule and a real pose, which is what makes it something a
        // player can pull out. It is NOT a `CrowdRecord`, so `society.agents`
        // has never heard of it, and this arm caught exactly that: 333 posed
        // against a ceiling of 330, three drivers. The ceiling is the number of
        // people the level has, and the drivers are people.
        posed_ceilings.push(1 + sim.society_stats().agents + sim.traffic_stats().drivers);
        assert!(
            bytes.len() >= POSED_BYTES,
            "{who} published {} bytes of pose, less than one 161-bone character's \
             {POSED_BYTES} — the island's hero has stopped being the starter \
             character",
            bytes.len()
        );
        assert_eq!(
            bytes.len() % POSED_BYTES,
            0,
            "{who} published {} bytes of pose, which is not a whole number of \
             161-bone characters — somebody in the store has a different rig",
            bytes.len()
        );
        posed_counts.push(bytes.len() / POSED_BYTES);
    }
    assert_eq!(
        posed_counts[0], posed_counts[1],
        "shipping posed {} characters and PIE posed {}",
        posed_counts[0], posed_counts[1]
    );
    assert!(
        posed_counts[0] <= posed_ceilings[0] && posed_counts[1] <= posed_ceilings[1],
        "the pose store holds {} / {} characters against a hero plus a derived \
         society of {} / {} — something is posing that neither the level nor \
         its own buildings put there",
        posed_counts[0],
        posed_counts[1],
        posed_ceilings[0],
        posed_ceilings[1]
    );
    println!(
        "POSE: {} x {POSED_BYTES} B a step on both hosts — the hero and the \
         island's own residents, of a possible {} (403 B a state before the \
         hero was a character, {} now)",
        posed_counts[0],
        posed_ceilings[0],
        a.states[0].len()
    );

    // **AND THE FOREST AGREES TOO** (island wave I7b). `biome_population` is
    // `#[serde(skip)]`, so it reaches no state fold — two hosts growing
    // different vegetation would have compared equal above, every step, for
    // ever. This is the comparison that says they do not.
    let (vmin, vmax) = (
        *a.veg_len.iter().min().expect("900 steps"),
        *a.veg_len.iter().max().expect("900 steps"),
    );
    let (tmin, tmax) = (
        *a.tiles.iter().min().expect("900 steps"),
        *a.tiles.iter().max().expect("900 steps"),
    );
    let shapes: std::collections::BTreeSet<u128> = a.veg.iter().copied().collect();
    println!(
        "VEGETATION over the drive: {vmin}..{vmax} instances on {tmin}..{tmax} \
         sim tiles, {} distinct forests",
        shapes.len()
    );
    assert!(
        vmin > 0,
        "the drive stood on bare ground at some step — the biome binding is \
         not evaluating over the streamed island"
    );
    assert!(
        tmax > tmin,
        "the simulation held {tmin} terrain tile(s) the whole way, so nothing \
         streamed and this arm cannot see a population stream with it"
    );
    assert!(
        shapes.len() > 1,
        "one forest for the whole drive — the population is not following the \
         ground that pages under it"
    );
    for (i, (x, y)) in a.veg.iter().zip(&b.veg).enumerate() {
        assert_eq!(
            x, y,
            "PIE and shipping grew DIFFERENT vegetation at step {i} of {STEPS} \
             ({} instances against {}) — the biome-bound population is a \
             function of something other than the resident ground",
            a.veg_len[i], b.veg_len[i]
        );
    }
    // …and the ground really paged **out** as well as in, which is the half a
    // one-way drive cannot show: the way home re-enters tiles the way out left.
    let shrank = a.tiles.windows(2).any(|w| w[1] < w[0]);
    let grew = a.tiles.windows(2).any(|w| w[1] > w[0]);
    println!("SIM TILES: grew {grew}, shrank {shrank}");
    assert!(
        grew && shrank,
        "the simulation's tile set only ever {} over the drive — vegetation \
         streaming OUT is not covered by this trace",
        if grew { "grew" } else { "held" }
    );

    // **…AND THE DRIVE REALLY STREAMED**, on both hosts, by the same numbers.
    // See `streaming_counters`: without this the whole file survives having the
    // streamers taken off *both* sides, which is the one mutation the byte
    // compare cannot see.
    let sc = streaming_counters(&ship);
    let pc = streaming_counters(&pie);
    for (who, c) in [("shipping", sc), ("document", pc)] {
        println!(
            "STREAMED {who}: {} cell activation(s), {} deactivation(s), {} \
             cell(s) resident, {} sim L0 page(s), {} page load(s)",
            c.0, c.1, c.2, c.3, c.4
        );
        assert!(
            c.0 > 0 && (c.1 > 0 || c.2 > 0),
            "{who} activated {} cell(s) and deactivated {} over {:.0} m of \
             driving — the partition is not streaming and this gate is \
             comparing two static worlds",
            c.0,
            c.1,
            STEPS as f64 * STEP_M
        );
        assert!(
            c.3 > 0 && c.4 > 0,
            "{who} paged {} terrain tile(s) ({} sim-resident) — the ground is not \
             streaming, so `height_at` answered off an empty working set the \
             whole way",
            c.4,
            c.3
        );
    }
    assert_eq!(
        sc, pc,
        "the two hosts streamed DIFFERENTLY over the same drive — the counters \
         are (cell activations, deactivations, cells resident, sim L0 pages, \
         page loads)"
    );
    // …and the ground paged **under the wheels** rather than only at the boot:
    // the drive itself loaded pages the start position had not asked for.
    println!(
        "PAGED BY THE DRIVE: {} load(s) at the start, {} after {:.0} m out and back",
        ship0.4,
        sc.4,
        STEPS as f64 * STEP_M
    );
    assert!(
        sc.4 > ship0.4 && pc.4 > pie0.4,
        "the drive paged nothing the boot had not already: {} loads at the start, \
         {} at the end. The hero moves {:.0} m across a {}-metre tile span, so a \
         streamer that is working has to fetch something on the way",
        ship0.4,
        sc.4,
        STEPS as f64 * STEP_M,
        recipe.grid.tile_span_m()
    );

    // **AND THE DRIVE IS THROUGH A SETTLEMENT NOW** (island wave I8a), which is
    // what re-prices this trace: the design's start is the first site's own
    // centre (`player_start` reads the committed road layer, and the routes run
    // centre to centre), so the 900 steps leave a settlement, cross its edge and
    // come back. Stated with the numbers rather than left as a change in what
    // the world holds.
    let solids = resident_solids(&ship);
    let doorways = inf_ecs::door::volume_doorways(ship.world());
    let volumes = resident_volumes(&ship);
    println!(
        "SETTLEMENT ON THE DRIVE: {} resident volume(s), {} solids, {} doorways \
         after {:.0} m out and back from {}",
        volumes.len(),
        solids.len(),
        doorways.len(),
        STEPS as f64 * STEP_M,
        recipe
            .sites
            .first()
            .map(|s| s.name.as_str())
            .unwrap_or("the start")
    );
    assert!(
        !volumes.is_empty() && !solids.is_empty() && !doorways.is_empty(),
        "the 900-step drive ended holding no settlement at all — this trace is \
         no longer over a world with a city in it"
    );
}

// ── the crowd (wave NPC1a) ──────────────────────────────────────────────────

/// How many test NPCs the crowd arm stands on the island.
///
/// Small on purpose. The claim is that **PIE equals shipping across a tier
/// TRANSITION**, and a transition needs the hero to drive past an agent, not a
/// thousand agents to exist: N is chosen so that a 360 m drive out and back
/// walks every rung of the ladder in both directions, and so that a 900-step
/// two-host comparison of a 161-bone rig stays a test rather than a benchmark.
/// The scale measurement is `crowd_sweep.rs`'s job and lives there.
const CROWD_N: usize = 24;

/// **The archetype the island's crowd wears — its own hero's.**
///
/// Read off the world rather than from a GUID table, `content_assets`'s doctrine
/// one level up: the hero is `samples/starter-character`, a **161-bone** rig, and
/// a table here would be a second place its identity is written down. It is also
/// what makes this arm able to quote the 6 476 B figure the trace re-shape is
/// argued against, because that number *is* this rig.
fn crowd_archetype(sim: &RuntimeSim) -> inf_ecs::crowd::CrowdArchetype {
    let w = sim.world().world();
    let mut best: Option<(Uuid, inf_ecs::crowd::CrowdArchetype)> = None;
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
        if best.as_ref().is_none_or(|(bg, _)| g.0 < *bg) {
            best = Some((
                g.0,
                inf_ecs::crowd::CrowdArchetype::humanoid(sk.mesh, sk.skeleton, sm),
            ));
        }
    }
    let (guid, a) = best.expect(
        "the island has no rigged character to copy, so every test NPC would pose nothing and the arm would compare two empty pipelines",
    );
    assert!(
        a.skeleton.is_some() && a.sm.is_some(),
        "the island hero {guid} has no skeleton or no machine"
    );
    a
}

/// **The population**: `CROWD_N` NPCs strung out along the drive, standing still.
///
/// Standing rather than walking, deliberately. The hero is what moves — it is the
/// `StreamingSource` the band anchors on — so a stationary agent's tier is a pure
/// function of *the hero's* position, and the drive out and back walks each agent
/// down the ladder and back up it. A walking agent would work too and would make
/// the transition depend on two moving things at once, which is a worse test of
/// one rule.
///
/// A pure function of the start point and `n`, so both hosts are given the same
/// people to disagree about.
fn crowd_population(
    from: glam::DVec3,
    n: usize,
    archetype: inf_ecs::crowd::CrowdArchetype,
) -> std::collections::BTreeMap<Uuid, inf_ecs::crowd::CrowdRecord> {
    const NAMESPACE: u128 = 0x4e50_4331_6169_736c_616e_6400_0000_0000;
    let mut out = std::collections::BTreeMap::new();
    for i in 0..n {
        // Along the drive's own axis, spaced so the 360 m run crosses several
        // Full/Near/Far boundaries, and offset off the line so nobody is
        // standing inside the hero.
        let along = i as f64 * (2.0 * STEPS as f64 * STEP_M / n as f64);
        let at = glam::DVec3::new(from.x + along, from.y, from.z + 6.0);
        out.insert(
            Uuid::from_u128(NAMESPACE | i as u128),
            inf_ecs::crowd::CrowdRecord::standing(archetype, at),
        );
    }
    out
}

/// **PIE == SHIPPING WITH A CROWD, ACROSS TIER TRANSITIONS** (wave NPC1a).
///
/// # The claim, and why it needs its own arm
///
/// `pie_equals_shipping_on_an_island_drive` is deliberately left alone: it pins
/// the hero at exactly `36 + 161 x 40 = 6 476` bytes of pose on both hosts, and
/// that pin is what says the sim-LOD ladder did not touch hero-class characters.
/// This arm adds the case that ladder introduced.
///
/// A sim-LOD tier decides three things a trace can see — whether an agent has a
/// rapier body, whether it is posed at all, and whether it has an entity — and it
/// decides them **per step, from sim state**. So a tier that was a function of
/// anything else would put the two hosts on different code paths at the step they
/// disagreed, and the byte compare below is what says they never do. The
/// dangerous case is not the steady state at either end of the ladder; it is the
/// **transition**, where one host might promote an agent a step before the other
/// and pose 6 476 bytes the other did not.
///
/// # What makes it non-vacuous
///
/// Four counters, each of which would be zero if the fixture were not posing the
/// problem, and every one of them is asserted:
///
/// * agents really change tier over the drive (`retiered`);
/// * every rung of the ladder is occupied at some point, `Dormant` included;
/// * agents really materialize **and** dematerialize;
/// * the pose section really moves — a crowd whose agents were all `Far` for the
///   whole run would compare equal on two hosts that had both stopped posing.
#[test]
fn pie_equals_shipping_with_a_crowd_across_tier_transitions() {
    let tmp = tempfile::tempdir().expect("a temp dir");
    let pack = cook(tmp.path());
    let proj = tmp.path().join("island");
    let recipe =
        inf_island::IslandRecipe::load(&fixture_recipe()).expect("the fixture recipe loads");
    let slug = inf_island::slug(&recipe.name);
    let from = start();

    let mut ship = pack_sim(&pack);
    let mut pie = loose_sim(&proj.join("Content"), &slug);

    // The archetype comes off the SHIPPING host and is given to both, so the two
    // are handed the identical people. Reading it twice would be two readings
    // that could disagree, which is the defect this arm exists to catch wearing
    // the fixture's clothes.
    let archetype = crowd_archetype(&ship);
    let pop = crowd_population(from, CROWD_N, archetype);
    assert_eq!(pop.len(), CROWD_N);
    ship.set_crowd_population(pop.clone());
    pie.set_crowd_population(pop);

    let a = drive(&mut ship, from);
    let b = drive(&mut pie, from);

    // ── the ladder was really walked ────────────────────────────────────────
    let occupied = |i: usize| a.crowd.iter().any(|c| c[i] > 0);
    let names = ["full", "near", "far", "dormant"];
    let seen: Vec<&str> = (0..4).filter(|i| occupied(*i)).map(|i| names[i]).collect();
    // The FIRST step re-tiers everything, because every record starts `Dormant`
    // and the band classifies it — so a `max` over the whole run would be
    // satisfied by that initial classification alone and would say nothing about
    // a transition. The claim is about the DRIVE, so the count is taken over the
    // steps after the crowd has settled.
    const SETTLED: usize = 10;
    let transitions = *a.retiered.iter().max().expect("900 steps");
    let on_the_drive = a
        .retiered
        .iter()
        .skip(SETTLED)
        .copied()
        .max()
        .expect("900 steps");
    let moved = a.retiered.iter().skip(SETTLED).filter(|r| **r > 0).count();
    let peak_full = a.crowd.iter().map(|c| c[0]).max().unwrap_or(0);
    let peak_dormant = a.crowd.iter().map(|c| c[3]).max().unwrap_or(0);
    println!(
        "CROWD over the drive: {CROWD_N} agents, tiers occupied {seen:?}, {transitions} re-tierings at peak ({on_the_drive} after the first {SETTLED} steps, on {moved} steps of {STEPS}), peak {peak_full} full / {peak_dormant} dormant"
    );
    assert_eq!(
        seen.len(),
        4,
        "only {seen:?} of the four tiers were ever occupied - the drive does not walk the ladder, so the transition this arm is about never happens"
    );
    assert!(
        on_the_drive > 0,
        "no agent changed tier after step {SETTLED} of a {:.0} m drive - every re-tiering this run saw was the initial classification, and the transition case is untested",
        2.0 * STEPS as f64 * STEP_M
    );
    assert!(
        moved > 1,
        "the crowd re-tiered on {moved} step(s) after settling - one is a single boundary crossing and the arm wants a drive"
    );

    // ── and the two hosts agree, step for step ──────────────────────────────
    let distinct: std::collections::BTreeSet<&Vec<u8>> = a.states.iter().collect();
    println!(
        "CROWD DRIVE: {} distinct states of {STEPS}, {} bytes a state",
        distinct.len(),
        a.states[0].len()
    );
    assert!(
        distinct.len() > STEPS as usize / 2,
        "only {} of {STEPS} states differ - the crowd drive is not moving the world",
        distinct.len()
    );
    for (i, (x, y)) in a.states.iter().zip(&b.states).enumerate() {
        assert_eq!(
            x, y,
            "PIE and shipping diverged at step {i} of {STEPS} with a crowd in the world - a sim-LOD tier, a materialization or a cached pose digest is a function of something other than sim state"
        );
    }
    assert_eq!(
        a.crowd, b.crowd,
        "the two hosts tiered the crowd differently at some step"
    );
    println!("PIE == SHIPPING over {STEPS} steps of an island drive WITH A CROWD");

    // ── the trace re-shape, as arithmetic on the island's own rig ───────────
    //
    // The island hero is 161 bones, so a posed character is exactly the 6 476 B
    // the ledger quotes and a non-posing agent is `AGENT_TRACE_BYTES`. Measured
    // at the end of the drive rather than asserted from the constant, because
    // the whole point of the re-shape is what the fold actually contains.
    let posed = inf_ecs::pose::pose_state_bytes(ship.world()).len();
    let crowd_bytes = inf_ecs::crowd::crowd_state_bytes(ship.world()).len();
    let posed_agents = inf_ecs::pose::posed_count(ship.world());
    const POSED_BYTES: usize = 36 + 161 * 40;
    let stats = ship.crowd_stats();
    let not_posing =
        stats.at(inf_ecs::crowd::CrowdTier::Far) + stats.at(inf_ecs::crowd::CrowdTier::Dormant);
    // The two counts are named apart on purpose (NPC1a audit): `not_posing` is
    // what the ladder SAVED, and the crowd section is what EVERY agent pays at
    // every tier — the first draft of this line multiplied the saving by 49 and
    // printed the section's total beside it, so it read as an identity that is
    // off by the agents still posing (20 x 49 = 980, and the section is 1 176).
    println!(
        "THE RE-SHAPE at the end of the drive: {posed_agents} posed x {POSED_BYTES} B = {posed} B of pose; {not_posing} of {CROWD_N} agents are off the pose path, and all {CROWD_N} pay {} B each = {crowd_bytes} B of crowd section. Had they all been Full the pose section would have been {} B.",
        inf_ecs::crowd::AGENT_TRACE_BYTES,
        (CROWD_N + 1) * POSED_BYTES
    );
    assert_eq!(
        posed % POSED_BYTES,
        0,
        "the pose section is {posed} B, not a whole number of 161-bone characters - the island's rig moved and every arithmetic in this wave's ledger is quoted against it"
    );
    assert_eq!(
        crowd_bytes,
        CROWD_N * inf_ecs::crowd::AGENT_TRACE_BYTES,
        "the crowd section folds {crowd_bytes} B for {CROWD_N} agents"
    );
    assert!(
        not_posing > 0,
        "every agent was posed at the end of the drive, so the re-shape saved nothing and this arm is measuring the pre-NPC1a engine"
    );
    // **EXACTLY the posing tiers, plus the hero** (NPC1a audit). The wave's
    // version of this line asserted `posed_agents < CROWD_N + 1`, which is
    // satisfied by the *Dormant* agents alone — a Dormant agent has no entity,
    // so it cannot be posed however broken the pose door is. Measured: severing
    // `if !tier.poses() { continue; }` in `step_pose_evaluation` took this
    // drive's state from 35 366 B to 119 554 B a step and left every assertion
    // in this arm green. The equality is what falsifies, because it is the pose
    // store's own count against the ladder's own verdict.
    // …plus the traffic's drivers, for the reason the drive arm's ceiling grew
    // (wave VEH2b): a driver is a person the crowd population has never heard
    // of, and it poses because it is `Full` — which it is exactly while its car
    // is.
    let drivers = ship.traffic_stats().drivers;
    let hero_and_posing = 1
        + drivers
        + stats.at(inf_ecs::crowd::CrowdTier::Full)
        + stats.at(inf_ecs::crowd::CrowdTier::Near);
    assert_eq!(
        posed_agents, hero_and_posing,
        "{posed_agents} characters were posed against {hero_and_posing} the ladder admits (the hero, {drivers} traffic driver(s), plus {} Full and {} Near of {CROWD_N} agents) - the pose door is not reading the tier",
        stats.at(inf_ecs::crowd::CrowdTier::Full),
        stats.at(inf_ecs::crowd::CrowdTier::Near),
    );

    // ── THE CACHED DIGEST IS NOT DEAD CODE ─────────────────────────────────
    //
    // A demoted agent's pose is GONE from the store, so the eight bytes
    // `CrowdRecord::pose_digest` contributes are a fold of the last pose it
    // published, carried as history — and history nothing reads is a field that
    // could be permanently zero with every other arm in this file green. This
    // says it moves: at least one agent walked down off the pose path over the
    // drive and took a non-zero digest with it, and both hosts agree on which.
    let digests = |sim: &RuntimeSim| -> Vec<(Uuid, u64)> {
        sim.world()
            .world()
            .get_resource::<inf_ecs::crowd::CrowdPopulationRes>()
            .map(|p| p.records.iter().map(|(g, r)| (*g, r.pose_digest)).collect())
            .unwrap_or_default()
    };
    let (ds, dp) = (digests(&ship), digests(&pie));
    let carried = ds.iter().filter(|(_, d)| *d != 0).count();
    println!("CACHED DIGESTS: {carried} of {CROWD_N} agents carry one");
    assert!(
        carried > 0,
        "no agent carries a cached pose digest after {STEPS} steps - either nothing was ever demoted out of a posing tier, or the capture is dead code and the eight bytes it folds are permanently zero"
    );
    assert_eq!(ds, dp, "the two hosts cached different pose digests");

    // ── AND THE ISLAND ITSELF DID NOT MOVE ──────────────────────────────────
    //
    // The walked-away fix (`CellStreamStats::rehomed`) changed what deactivation
    // does to every streamed entity, not just to crowd agents. This is the
    // control that says the island's own content stands still and therefore
    // traces exactly as it did: zero re-homes over a 360 m drive across cell
    // boundaries in both directions.
    for (who, sim) in [("shipping", &ship), ("pie", &pie)] {
        let st = sim.cell_streaming().stats();
        println!(
            "{who}: {} activation(s) / {} deactivation(s), {} re-homed",
            st.activations, st.deactivations, st.rehomed
        );
        assert_eq!(
            st.rehomed, 0,
            "{who} re-homed {} streamed entit(ies) on an island whose content does not move - the NPC1a deactivation rule is admitting things it must not",
            st.rehomed
        );
    }
}

/// **A `ScenePayload` CARRIES NO PARTITION**, so a PIE preview of the island
/// runs it whole.
///
/// This is a pre-existing engine property, not a defect this wave introduced,
/// and it is measured here rather than described: the wire has `level_bytes`,
/// classes, pcgs, skeletons, clips, machines, biome sets and voxels — and no
/// `.inf_part`, because the partition is **derived at cook** and a payload is
/// what the editor has *before* a cook.
///
/// The consequence for an author is worth stating plainly: previewing a 51 km²
/// island with `--pie` builds every entity in it at once, where the shipped
/// player streams them. For the *fixture* that is fifteen entities against six;
/// for the island it is every lake, river and site at once. It is why
/// `pie_equals_shipping_on_an_island_drive` compares the loose document against
/// the pack — which is the pair P16.5's own gate compares, for this reason.
#[test]
fn a_scene_payload_carries_no_partition() {
    let tmp = tempfile::tempdir().expect("a temp dir");
    let proj = build_project(tmp.path());
    let payload_sim = pie_sim(&proj);

    let recipe =
        inf_island::IslandRecipe::load(&fixture_recipe()).expect("the fixture recipe loads");
    let slug = inf_island::slug(&recipe.name);
    let streamed = loose_sim(&proj.join("Content"), &slug);

    let count = |sim: &RuntimeSim| sim.world().world().iter_entities().count();
    let (whole, part) = (count(&payload_sim), count(&streamed));
    println!(
        "PIE PAYLOAD: {whole} entities built at once; the streamed reading has \
         {part} resident at step 0"
    );
    assert!(
        whole > part,
        "a payload preview ({whole}) should hold MORE than a streamed world's \
         resident set ({part}) — if they are equal the level is not partitioned \
         and this arm is measuring nothing"
    );
    // …and the level really does ask to be partitioned, or the difference above
    // is about something else.
    let doc = inf_editor_core::scene::serialize::load(
        &proj.join("Content").join(format!("{slug}.inf_lvl")),
    )
    .expect("the island level loads");
    assert!(
        doc.settings().partition.enabled,
        "the island level is not partitioned"
    );
}

/// The cooked island really is an island: the pack carries the terrain, the
/// partition, the palette and the vegetation.
#[test]
fn the_cooked_island_carries_every_half_the_recipe_builds() {
    let tmp = tempfile::tempdir().expect("a temp dir");
    let pack = cook(tmp.path());
    let reader = inf_asset::PackReader::open(&pack.join(inf_player::level::PACK_FILE))
        .expect("the pack reader opens");

    let mut kinds: std::collections::BTreeMap<String, usize> = Default::default();
    for e in reader.index() {
        *kinds.entry(e.kind.slug().to_string()).or_default() += 1;
    }
    println!("PACK: {kinds:?}");
    for want in ["terrain", "level", "biome_set", "pcg", "mesh", "partition"] {
        assert!(
            kinds.contains_key(want),
            "the cooked island has no {want}: {kinds:?}"
        );
    }

    // The terrain in the pack is the one the recipe built, with its pyramid.
    let recipe =
        inf_island::IslandRecipe::load(&fixture_recipe()).expect("the fixture recipe loads");
    let guid = inf_asset::AssetId(inf_island::terrain_guid(&recipe.name));
    let bytes = reader.read(guid).expect("the terrain is in the pack");
    let asset = inf_terrain::TerrainAssetReader::new(&bytes[..]).expect("it decodes");
    println!(
        "TERRAIN: {} tiles, {} LOD levels, {}² samples @ {} m, origin {:?}",
        asset.tile_count(),
        asset.lod_levels(),
        asset.tile_resolution(),
        asset.meters_per_sample(),
        asset.origin()
    );
    assert_eq!(asset.tile_resolution(), recipe.grid.tile_resolution);
    assert_eq!(asset.meters_per_sample(), recipe.grid.meters_per_sample);
    assert!(
        asset.tile_count() as u64 > recipe.grid.tile_count(),
        "the pyramid is missing: {} tiles for {} level-0 pages",
        asset.tile_count(),
        recipe.grid.tile_count()
    );
    // **The georeferenced origin survives the cook**, which is what makes the
    // terrain land where the survey says it does.
    let anchor = recipe.anchor().expect("the anchor builds");
    assert_eq!(asset.origin().x, anchor.origin_easting_m);
    assert_eq!(asset.origin().z, anchor.origin_northing_m);
}

/// **PIE == SHIPPING ON THE RENDER HALF** (wave FIX2): the two hosts resolve the
/// SAME meshlet DAG for every rigid `MeshRef.asset` in the island, in the same
/// order.
///
/// # What this could not see before, and why the gate needed it
///
/// `pie_equals_shipping_on_an_island_drive` compares SIMULATION state, and every
/// gap this wave closed is at the frame: a windowed PIE session stepped the
/// island exactly like the shipped build and drew the roads as placeholder cubes
/// at the world origin, 2.7 km from the spawn, because `run_pie` handed the
/// renderer an empty `VmeshRegistry`. No fold could ever have caught it. This
/// arm reads what the renderer was actually given.
///
/// # Why the comparison is over the level's refs and not over the two registries
///
/// The full sets are deliberately different and it would be wrong to assert
/// otherwise: a cooked pack indexes every mesh it virtualized (the cook's
/// `[vgeom] min_triangles` is 2048), while a payload names exactly the DAGs the
/// level's rigid `MeshRef.asset`s resolve to. Both sets are printed for the
/// record and the payload's must be contained in the pack's — the containment is
/// the honest claim — but the *equality* that matters is per mesh: for every
/// mesh the level draws, do the two hosts resolve the same asset id, or does one
/// of them resolve nothing?
///
/// # The sweep runs before the cook, on purpose
///
/// That is the ROAD1b shape: the editor derives a `.inf_vmesh` beside every mesh
/// it draws, at exactly `derived_vmesh_id` of the mesh, and the cook's collision
/// guard used to ask the PROJECT rather than the closure and threw away the DAG
/// it had just built. The cooked island then carried no virtualized geometry at
/// all and the shipped build drew cubes too. Sweeping first puts those files
/// where they were when that happened, so this arm reds if the guard regresses.
#[test]
fn both_hosts_resolve_the_same_dag_for_every_mesh_the_island_draws() {
    let tmp = tempfile::tempdir().expect("a temp dir");
    let proj = build_project(tmp.path());
    let content = proj.join("Content");

    // ── the EDITOR's derived DAGs, through its own sweep ────────────────────
    let mut project =
        inf_editor_core::assets::AssetProject::open(&proj).expect("the project opens");
    let derived = inf_editor_core::assets::vmesh::sweep(&mut project);
    println!("SWEEP: {} derived .inf_vmesh", derived.len());
    assert!(
        !derived.is_empty(),
        "the sweep derived nothing, so both columns below would be empty and agree"
    );

    // ── the level's rigid mesh refs, in document order, deduplicated ────────
    let recipe =
        inf_island::IslandRecipe::load(&fixture_recipe()).expect("the fixture recipe loads");
    let slug = inf_island::slug(&recipe.name);
    let doc = inf_editor_core::scene::serialize::load(&content.join(format!("{slug}.inf_lvl")))
        .expect("the island level loads");
    let mut refs: Vec<Uuid> = Vec::new();
    for &guid in doc.order() {
        let Some(e) = doc.world().entity_of(guid) else {
            continue;
        };
        if let Some(mesh) = doc
            .world()
            .world()
            .get::<inf_ecs::components::MeshRef>(e)
            .and_then(|m| m.asset)
        {
            if !refs.contains(&mesh) {
                refs.push(mesh);
            }
        }
    }
    println!("MESH REFS: {} — {refs:?}", refs.len());
    // The exact expected count, taken from the fixture, asserted BEFORE anything
    // is compared — the P21.4 rule. Two hosts that both draw nothing agree.
    assert_eq!(
        refs.len(),
        4,
        "the island's rigid mesh refs are the four road layers; this fixture has \
         {} and the comparison below would be measuring the wrong thing",
        refs.len()
    );

    // ── the PIE host's registry, built the way `inf-player --pie` builds it ──
    //
    // The PCG, biome-set, mesh and byte resolvers are REAL (read off the sidecars,
    // the way the editor's asset database answers), because the scatter half of
    // this arm needs the graphs and the meshes their kinds name — the payload
    // carries a level's scatter meshes in `meshes` since FIX2, and a table built
    // from `|_| None` would be empty on both sides and agree.
    let assets = content_assets(&content);
    let read_asset = |g: Uuid| assets.get(&g).map(|p| std::fs::read(p).expect("an asset"));
    let payload = inf_editor_core::pie::build_scene_payload(
        &doc,
        |_| None,
        read_asset,
        |_| None,
        read_asset,
        |_| None,
        |_| None,
        read_asset,
        read_asset,
        |g| match inf_editor_core::assets::vmesh::derived_vmesh(&project, inf_asset::AssetId(g)) {
            inf_editor_core::assets::vmesh::DerivedVmesh::Current(p) => {
                Some(inf_editor_core::pie::VmeshRef::Path(p))
            }
            inf_editor_core::assets::vmesh::DerivedVmesh::Stale(_) => {
                Some(inf_editor_core::pie::VmeshRef::Stale)
            }
            inf_editor_core::assets::vmesh::DerivedVmesh::Absent => None,
        },
        HZ as u32,
        false,
    )
    .expect("the payload builds");
    assert_eq!(
        payload.vmesh_paths.len(),
        refs.len(),
        "the payload names {} DAGs for {} rigid mesh refs",
        payload.vmesh_paths.len(),
        refs.len()
    );
    let pie = inf_player::vmeshes_from_payload(&payload);

    // ── the SHIPPED host's registry, off the cooked pack ────────────────────
    let out = tmp.path().join("out");
    inf_packager::cook(&proj, &out, &inf_packager::CookOptions::default())
        .expect("the island cooks");
    let reader = std::sync::Arc::new(
        inf_asset::PackReader::open(&out.join(inf_player::level::PACK_FILE))
            .expect("the pack opens"),
    );
    let reader_for_scatter = reader.clone();
    let ship = inf_player::vmesh::VmeshRegistry::from_pack(reader).expect("the DAGs index");

    // One `println!` and not two: a literal beginning with a run of spaces (to
    // align a second line under the first) is indistinguishable from a `\`
    // continuation the harness ate, and `inf-packager`'s workspace-wide source
    // gate refuses them all rather than trying to tell the two apart.
    println!(
        "REGISTRIES: pie {} — {:?} / ship {} — {:?}",
        pie.len(),
        pie.registered_guids(),
        ship.len(),
        ship.registered_guids()
    );
    for g in pie.registered_guids() {
        assert!(
            ship.contains(g),
            "the PIE host indexed DAG {g}, which the cooked pack does not carry"
        );
    }

    // ── and the SCATTER table, the fifth store of the same class ────────────
    //
    // Wave TER2b closed "a scattered instance draws its authored mesh" for the
    // cooked and dev-dir boots and left `run_pie` assigning no table at all, so a
    // windowed PIE session drew ground cover as placeholder primitives while the
    // shipped build drew props. Both tables are now built through
    // `finish_scatter_meshes`, so the twelve building-module families cannot be
    // added on one path and not the other — which is what a bare `from_payload`
    // vs `from_pack` comparison would have missed.
    let pie_scatter = inf_player::scatter_meshes_from_payload(&payload);
    let ship_scatter = {
        let mut t = inf_player::scatter_mesh::from_pack(&reader_for_scatter);
        inf_player::scatter_mesh::add_building_modules(&mut t);
        t
    };
    let keys = |t: &inf_render::ScatterMeshes| {
        let mut k: Vec<u128> = t.keys().copied().collect();
        k.sort();
        k
    };
    println!(
        "SCATTER: pie {} / ship {}",
        pie_scatter.len(),
        ship_scatter.len()
    );
    assert!(
        pie_scatter.len() > 12,
        "the PIE scatter table is {} entries — the twelve building modules and \
         nothing authored, so the comparison below would only be testing a Ring-0 \
         default",
        pie_scatter.len()
    );
    assert_eq!(
        keys(&pie_scatter),
        keys(&ship_scatter),
        "the two hosts scatter different meshes"
    );

    // ── THE COMPARISON ─────────────────────────────────────────────────────
    let column = |reg: &inf_player::vmesh::VmeshRegistry| -> Vec<Option<u128>> {
        refs.iter()
            .map(|m| reg.resolve(*m).map(|(id, _)| id))
            .collect()
    };
    let (a, b) = (column(&pie), column(&ship));
    assert_eq!(
        a, b,
        "the two hosts resolve different geometry for the meshes this level \
         draws — PIE {a:?} vs shipping {b:?}"
    );
    assert!(
        a.iter().all(|x| x.is_some()),
        "both hosts resolve NOTHING for at least one road mesh, and two hosts \
         drawing nothing agree perfectly: {a:?}"
    );

    // ── AND THE FRAME, not only the registry ────────────────────────────────
    //
    // Everything above is about which DAGs a host indexed. This is what the
    // renderer is actually handed, projected through `project_scene_full` — the
    // function the windowed player calls — with the real registry and with an
    // empty one, which is precisely the registry `run_pie` built for itself
    // before this wave.
    //
    // Two facts, and each falsifies a different half of FIX2:
    //
    //  * with the payload's registry the roads reach the scene as **vgeom**
    //    instances, and with an empty one they reach it as nothing. Drop
    //    `with_vmesh_paths` and the first count goes to zero.
    //  * the two projections push the SAME number of primitive instances. That
    //    is the placeholder cube being gone: before FIX2 the empty registry
    //    produced four extra `MeshInstance`s — 1 m boxes at `Transform::IDENTITY`,
    //    2.7 km from the spawn — and restoring that branch reds this line.
    let sim = inf_player::sim_from_payload(&payload).expect("the payload builds a world");
    let project = |reg: &inf_player::vmesh::VmeshRegistry| {
        let mut scene = inf_render::RenderScene::default();
        inf_player::render::project_scene_full(
            &mut scene,
            &sim.sim,
            1.0,
            reg,
            &inf_player::skinned::SkinnedRegistry::new(),
            &inf_voxel::VoxelVolumes::new(),
            &mut inf_render::DebrisCache::default(),
            None,
            &inf_render::ScatterMeshes::new(),
        );
        (scene.vgeom_instances.len(), scene.instances.len())
    };
    let (drawn, prims) = project(&pie);
    let (none_drawn, prims_empty) = project(&inf_player::vmesh::VmeshRegistry::new());
    println!("FRAME: vgeom {drawn} / prims {prims} — empty registry: vgeom {none_drawn} / prims {prims_empty}");
    assert_eq!(
        drawn,
        refs.len(),
        "the payload's registry put {drawn} vgeom instances in the frame for {} \
         rigid mesh refs",
        refs.len()
    );
    assert_eq!(
        none_drawn, 0,
        "an EMPTY registry still drew vgeom, so the comparison above is not \
         measuring the registry"
    );
    assert_eq!(
        prims,
        prims_empty,
        "a mesh whose DAG is missing drew {} placeholder primitives — the cube \
         FIX2 deleted is back, and a shipped build that cannot find a mesh is \
         claiming a 1 m box stands where the author put a road",
        prims_empty - prims
    );
}

/// The level carries its geo-anchor through the cook, so the sky knows where on
/// Earth it is.
#[test]
fn the_cooked_level_still_knows_where_on_earth_it_is() {
    let tmp = tempfile::tempdir().expect("a temp dir");
    let pack = cook(tmp.path());
    // Read the cooked LEVEL out of the pack and decode it the way the shipped
    // player's own reader does — the geo-anchor is a file-level settings block,
    // not an entity, so it rides the `.inf_lvl` rather than the built world.
    let source = inf_player::level::PackLevelSource::open(&pack).expect("the pack opens");
    let bytes = source
        .reader()
        .read(source.root_level())
        .expect("the root level is in the pack");
    let level = inf_scene::RuntimeLevel::decode(&bytes).expect("the cooked level decodes");
    let geo = &level.geo;
    println!(
        "GEO: enabled {} crs {:?} at {:.5} N {:.5} E, convergence {:.4} deg",
        geo.enabled,
        geo.crs,
        geo.origin_latitude_deg,
        geo.origin_longitude_deg,
        geo.grid_convergence_deg
    );
    assert!(geo.enabled, "the cooked island lost its geo-anchor");
    assert_eq!(geo.crs, "EPSG:32610");
    assert!((49.0..50.0).contains(&geo.origin_latitude_deg));
    assert!((-124.0..-122.0).contains(&geo.origin_longitude_deg));
    // The solar place is what the sky reads, and it is the anchor's.
    let (lat, lon) = geo.solar_place().expect("an enabled anchor has a place");
    assert_eq!(lat, geo.origin_latitude_deg);
    assert_eq!(lon, geo.origin_longitude_deg);
}

/// **The settlements stand on the ground urban reserves** (island wave I8a) —
/// the flipped half of the tripwire above.
///
/// Three claims, all of them about the built project rather than about the
/// generator that wrote it:
///
/// * every block a settlement plans sits inside its own site's reservation
///   circle, so it is on ground the carve levelled and the biome map painted
///   urban — which is what makes the vegetation and the buildings disjoint by
///   construction rather than by luck;
/// * every zone document a block names is **in the project**, resolvable by
///   GUID out of the content root the recipe's `[content]` list filled. A
///   `PcgVolume` whose graph does not resolve evaluates to nothing and says
///   nothing, which is the failure this catches;
/// * a settlement that plans no block at all is named, not skipped.
fn the_settlements_stand_where_urban_is_reserved(
    content: &Path,
    recipe: &inf_island::IslandRecipe,
) {
    let design = inf_island::read_design(recipe).expect("the committed design reads");
    let plans = inf_editor_core::settlement::settlements(&design);
    let assets = content_assets(content);
    let mut blocks = 0usize;
    for p in &plans {
        let site = &recipe.sites[p.site];
        assert!(site.kind.reserves_urban());
        println!(
            "SETTLEMENT {} ({}): {} blocks inside a {:.0} m reservation, {} refused \
             off-pad, {} refused off-land",
            p.name,
            p.kind.label(),
            p.blocks.len(),
            p.radius_m,
            p.refused_off_pad,
            p.refused_off_land
        );
        for b in &p.blocks {
            for c in b.corners() {
                assert!(
                    (c - p.centre).length() <= site.radius_m,
                    "{}'s block {:?} reaches outside the reservation the biome map \
                     paints urban — its buildings would stand in a forest",
                    p.name,
                    (b.col, b.row)
                );
            }
            let g = inf_editor_core::settlement::zone_guid(b.archetype);
            assert!(
                assets.contains_key(&g),
                "{}'s {} block names zone document {g}, which is not in the built \
                 project — the volume would evaluate to nothing, silently",
                p.name,
                b.archetype.name()
            );
        }
        blocks += p.blocks.len();
    }
    println!(
        "SETTLEMENTS: {blocks} blocks over {} settlements, {} distinct zone \
         documents in the project",
        plans.len(),
        inf_pcg::ArchetypeId::ALL
            .iter()
            .filter(|a| assets.contains_key(&inf_editor_core::settlement::zone_guid(**a)))
            .count()
    );
    assert!(
        blocks > 0,
        "the committed design plans no settlement block at all"
    );
}

/// **THE VEGETATION SCATTERS ON THE GROUND THAT IS RESIDENT, AND NOT BEFORE.**
///
/// # What this arm used to say
///
/// Wave I7 measured the gap and asserted it: **4 958 instances with the ground
/// paged by hand and 0 through the shipped boot**, because
/// `evaluate_biome_bindings` ran once at load over `TerrainData::xz_bounds()`
/// and a streamed terrain ships no tiles. Wave I7b closed it — the fixed step
/// refreshes the population from the ground the terrain streamer just paged —
/// so the arm went red as designed and this is its rewrite.
///
/// # What it says now, and why each half is here
///
/// * **not before** — a world built with no terrain streamer attached holds no
///   tiles, so it grows nothing. The population is a function of resident
///   ground and there is none.
/// * **and after** — the shipped boot, with the streamer attached, grows
///   thousands of instances on the pages the hero stands on.
/// * **and it is the SAME forest the author would preview.** Not a count: every
///   instance the streamed world grows is one the fully-paged reading grows, at
///   the same position, and over a tile whose neighbours are all resident the
///   two agree **exactly**. That is the claim "the shipped island grows what
///   the preview shows" reduced to a comparison, and a count could not make it.
#[test]
fn the_biome_binding_scatters_when_its_ground_is_resident_and_not_before() {
    let tmp = tempfile::tempdir().expect("a temp dir");
    let proj = build_project(tmp.path());
    let content = proj.join("Content");
    let recipe =
        inf_island::IslandRecipe::load(&fixture_recipe()).expect("the fixture recipe loads");
    let slug = inf_island::slug(&recipe.name);

    // The palette really binds a graph — the wire the whole thing hangs on.
    let set_bytes = std::fs::read(content.join(format!("{slug}.inf_biomes")))
        .expect("the biome set is written");
    let set = inf_asset::decode::<inf_terrain::BiomeSet>(&set_bytes).expect("it decodes");
    let bound: Vec<&str> = set
        .biomes
        .iter()
        .filter(|b| b.pcg_graph.is_some())
        .map(|b| b.name.as_str())
        .collect();
    println!("BOUND BIOMES: {bound:?}");
    assert_eq!(
        bound.len(),
        6,
        "every biome but urban binds cover: {bound:?}"
    );
    // **THE TRIPWIRE FLIPPED, AND THE SENTENCE IT CARRIED IS SPENT** (island
    // wave I8a). This read *"urban must stay bare for wave I8"* — a wave that
    // has now happened. What is still true is that urban binds no VEGETATION
    // graph, and the reason is no longer "nobody has built the settlements yet":
    // it is that a settlement is a `PcgVolume` in the LEVEL, not a biome
    // binding, so the two authorities never meet. What urban reserving the
    // ground buys is exactly what wave I7 said it would — the settlement
    // generator finds bare ground rather than a forest to clear.
    //
    // So the arm asserts the settlements instead, on the world: the level
    // carries one volume per block, every one of them names a committed zone
    // document, and every one of them sits inside a site's own reservation.
    assert!(
        !bound.contains(&"urban"),
        "urban binds a cover graph — the settlements stand on reserved ground \
         and the vegetation must not grow through them"
    );
    the_settlements_stand_where_urban_is_reserved(&content, &recipe);

    let pcg_bytes = std::fs::read(content.join(format!("{slug}Cover.inf_pcg")))
        .expect("the cover graph is written");
    let pcg = inf_pcg::PcgAssetPayload::decode(&pcg_bytes).expect("the cover graph decodes");
    let binding = inf_pcg::BiomeBinding::from_set(&set, inf_pcg::DEFAULT_BIOME_FEATHER, |g| {
        (g == inf_island::cover_pcg_guid(&recipe.name)).then(|| pcg.document.clone())
    });
    assert_eq!(
        binding.graphs().len(),
        6,
        "the binding resolved {:?}",
        binding.graphs().len()
    );

    // ── with the ground RESIDENT ──
    let asset = inf_terrain::read_terrain_asset(&content.join(format!("{slug}.inf_terrain")))
        .expect("the built terrain reads");
    let reader = asset.reader();
    let mut data =
        inf_terrain::TerrainData::new(recipe.grid.tile_resolution, recipe.grid.meters_per_sample);
    let (min, max) = inf_island::IslandGrid::of(&recipe).bounds();
    let report = inf_terrain::residency::page_region(
        &mut data,
        &reader,
        glam::DVec2::new(min.x, min.y),
        glam::DVec2::new(max.x, max.y),
    );
    println!(
        "PAGED: {} tiles loaded, {} missing",
        report.loaded.len(),
        report.missing.len()
    );
    assert!(!data.is_empty(), "the whole fixture terrain pages");

    // The author's reading: every tile of the island in memory at once, through
    // the same Ring-0 door the shipped step calls tile by tile.
    let instances = binding.evaluate_resident(&data, glam::DVec3::ZERO);
    println!(
        "VEGETATION: {} instances over {:.3} km2 at {} /m2",
        instances.len(),
        (max.x - min.x) * (max.y - min.y) / 1.0e6,
        inf_editor_core::island::ISLAND_SCATTER_DENSITY
    );
    assert!(
        instances.len() > 500,
        "the binding scattered only {} instances with the ground resident",
        instances.len()
    );
    // Every instance is on the ground and inside the world.
    for i in instances.iter().take(200) {
        assert!(i.pos.x >= min.x - 1.0 && i.pos.x <= max.x + 1.0);
        assert!(i.pos.z >= min.y - 1.0 && i.pos.z <= max.y + 1.0);
        assert!(i.pos.y.is_finite());
    }

    // ── NOT BEFORE: a world with no ground paged ──
    //
    // **Two controls since wave I8a, and the second one is why.** This used to be
    // one: cell streaming attached, terrain streaming deliberately not, asserting
    // zero tiles and zero instances. With the settlements standing that is no
    // longer true and the reason is a *feature* — IB-1's rule that **PCG pages
    // its own ground**. A settlement block is a `PcgVolume` with
    // `Ground::Terrain`, so activating its cell runs `page_terrains_for_pcg`
    // before evaluating it, and the terrain the level shipped empty now holds the
    // page that volume needed. The vegetation then grows on it, correctly.
    //
    // So the true zero moves to a world with **neither** streamer (no cells, no
    // volumes, no pre-pass, no ground), and the cell-only world becomes what it
    // actually is: a strict, tiny subset of the shipped reading.
    let pack = cook(tmp.path());
    let source = inf_player::level::PackLevelSource::open(&pack).expect("the pack opens");
    let mut nothing = {
        let built = inf_player::build_world_from_pack(&source).expect("the world builds");
        inf_player::sim_from_built(built)
    };
    nothing.step_once(inf_player::runtime_sim::RuntimeInput::default());
    let (_, nothing_pop) = veg_digest(&nothing);
    println!(
        "NEITHER STREAMER: {} sim tile(s), {nothing_pop} instances",
        sim_tiles(&nothing)
    );
    assert_eq!(sim_tiles(&nothing), 0, "a streamed level ships no tiles");
    assert_eq!(
        nothing_pop, 0,
        "the binding grew {nothing_pop} instances over ground that is not there"
    );

    let mut bare = {
        let mut built = inf_player::build_world_from_pack(&source).expect("the world builds");
        let partition = built.take_partition();
        let pcg = built.pcg_context();
        let mut sim = inf_player::sim_from_built(built);
        inf_player::attach_cell_streaming(&mut sim, &partition, pcg);
        sim // …and deliberately NO terrain streamer.
    };
    bare.step_once(inf_player::runtime_sim::RuntimeInput::default());
    let (_, bare_pop) = veg_digest(&bare);
    let bare_tiles = sim_tiles(&bare);
    println!(
        "CELLS BUT NO TERRAIN STREAMER: {bare_tiles} sim tile(s) paged by the \
         settlements' own PCG pre-pass, {bare_pop} instances"
    );
    assert!(
        bare_tiles > 0 && bare_pop > 0,
        "the settlement volumes paged no ground of their own — IB-1's pre-pass \
         is not running, and a building with `Ground::Terrain` over an unpaged \
         page fails closed and builds nothing"
    );

    // ── AND AFTER: the shipped boot, streamer attached ──
    let mut ship = pack_sim(&pack);
    ship.step_once(inf_player::runtime_sim::RuntimeInput::default());
    let (_, population) = veg_digest(&ship);
    let tiles = sim_tiles(&ship);
    println!("SHIPPED BOOT: {population} instances on {tiles} sim tile(s)");
    assert!(tiles > 0, "the shipped boot paged no ground");
    assert!(
        population > 500,
        "the streamed boot grew only {population} instances on {tiles} paged \
         tile(s) — the refresh is not reaching the resident ground"
    );
    // …and the terrain streamer is what did it: the settlements' own pre-pass
    // pages a page or two, the streamer pages the neighbourhood.
    assert!(
        tiles > bare_tiles && population > bare_pop,
        "the terrain streamer added nothing the settlements' PCG pre-pass had \
         not already paged ({tiles} against {bare_tiles} tiles, {population} \
         against {bare_pop} instances)"
    );

    // ── AND IT IS THE SAME FOREST ──
    let author: std::collections::BTreeSet<(u64, u64, u64)> = instances
        .iter()
        .map(|i| (i.pos.x.to_bits(), i.pos.y.to_bits(), i.pos.z.to_bits()))
        .collect();
    let (shipped, resident): (Vec<_>, Vec<(i32, i32)>) = {
        let w = ship.world().world();
        let t = w
            .iter_entities()
            .find_map(|e| e.get::<inf_ecs::components::Terrain>())
            .expect("the island has ground");
        (
            t.biome_population.clone(),
            t.data.tiles().map(|(&c, _)| c).collect(),
        )
    };
    let stray = shipped
        .iter()
        .filter(|i| {
            !author.contains(&(
                i.position.x.to_bits(),
                i.position.y.to_bits(),
                i.position.z.to_bits(),
            ))
        })
        .count();
    println!(
        "SAME FOREST: {} of {} shipped instances are places the fully-paged \
         reading also grows ({stray} stray)",
        shipped.len() - stray,
        shipped.len()
    );
    assert_eq!(
        stray, 0,
        "the streamed island grew {stray} instance(s) the author's fully-paged \
         reading does not — a streamed forest must be a SUBSET of the whole one, \
         place for place"
    );

    // …and over a tile whose whole neighbourhood is resident, the two are not
    // merely a subset of one another: they are equal. That is the interior of
    // the streamed world reading exactly as the author's does.
    let set: std::collections::BTreeSet<(i32, i32)> = resident.iter().copied().collect();
    let span = recipe.grid.tile_span_m();
    let interior = set
        .iter()
        .copied()
        .find(|c| (-1..=1).all(|dz| (-1..=1).all(|dx| set.contains(&(c.0 + dx, c.1 + dz)))))
        .expect("the sim's resident set has an interior tile");
    let (x0, z0) = (interior.0 as f64 * span, interior.1 as f64 * span);
    let inside = |x: f64, z: f64| (x0..x0 + span).contains(&x) && (z0..z0 + span).contains(&z);
    let mine: std::collections::BTreeSet<(u64, u64, u64)> = shipped
        .iter()
        .filter(|i| inside(i.position.x, i.position.z))
        .map(|i| {
            (
                i.position.x.to_bits(),
                i.position.y.to_bits(),
                i.position.z.to_bits(),
            )
        })
        .collect();
    let theirs: std::collections::BTreeSet<(u64, u64, u64)> = instances
        .iter()
        .filter(|i| inside(i.pos.x, i.pos.z))
        .map(|i| (i.pos.x.to_bits(), i.pos.y.to_bits(), i.pos.z.to_bits()))
        .collect();
    println!(
        "INTERIOR TILE {interior:?}: {} shipped against {} authored",
        mine.len(),
        theirs.len()
    );
    assert!(
        !theirs.is_empty(),
        "the interior tile {interior:?} grows nothing in either reading, so \
         comparing them proves nothing"
    );
    assert_eq!(
        mine, theirs,
        "inside a fully-resident tile the streamed island and the fully-paged \
         reading must place the SAME instances"
    );
}

/// **THE GROUND THE SIMULATION STANDS ON IS THE GROUND THE RECIPE BUILT.**
///
/// # Why this arm exists
///
/// The gate above compares two hosts. Two hosts reading the *same* wrong ground
/// agree perfectly, and for the whole of wave I7 they did: the island's `Terrain`
/// entity carried `Transform::from_translation(grid.bounds().0)` on top of an
/// `.inf_terrain` whose tile indices are **already centred on the world origin**
/// (`IslandGrid::tile0 = -(tiles / 2)`), so the centring was applied twice.
///
/// Measured before the fix, through this same seam: the design's own player start
/// read **0.000 m of unauthored ground where the build puts 129.916 m**, and the
/// world origin read 80.000 m off a page 768 m away. On the shipped island the
/// displacement is 3 584 m on each axis — half the terrain outside the world.
///
/// So the comparison here is host **against the recipe**, not host against host:
/// `RuntimeSim::terrain_height_at` is the exact seam a Blueprint's
/// `terrain.height_at`, the character's ground snap and the physics heightfield
/// all read, and `IslandBuild::terrain` is what `inf island build` wrote.
#[test]
fn the_ground_the_simulation_stands_on_is_the_ground_the_recipe_built() {
    let tmp = tempfile::tempdir().expect("a temp dir");
    let pack = cook(tmp.path());
    let mut ship = pack_sim(&pack);

    let recipe =
        inf_island::IslandRecipe::load(&fixture_recipe()).expect("the fixture recipe loads");
    let build = inf_island::build_island(&recipe, &inf_island::BuildOptions::default())
        .expect("the fixture island builds");
    let s = start();
    let hero = hero_entity(&ship).expect("a hero");

    // The design's own places: where a player starts, the other settlement, the
    // world origin and a point between them. All four are inside the coastline —
    // a probe on the sea shelf would be a probe on a flat surface, which agrees
    // with itself under any displacement.
    let probes: Vec<(f64, f64)> = build
        .recipe
        .sites
        .iter()
        .map(|q| (q.x, q.z))
        .chain([(0.0, 0.0), (200.0, -200.0)])
        .collect();
    let mut seen: Vec<f64> = Vec::new();
    assert!(probes.len() >= 4, "too few probes to say anything");
    for (x, z) in probes {
        // Stand the streaming source there and let the sim page its own
        // neighbourhood in — residency is derived from sim state, so this is the
        // only honest way to ask.
        set_hero(&mut ship, hero, glam::DVec3::new(x, s.y, z));
        for _ in 0..3 {
            ship.step_once(inf_player::runtime_sim::RuntimeInput::default());
        }
        let sim_h = ship.terrain_height_at(x, z);
        let built = build
            .terrain
            .height_at(glam::DVec2::new(x, z))
            .unwrap_or_else(|| panic!("({x}, {z}) is off the built terrain"));
        println!("GROUND ({x:>7.1}, {z:>7.1}): sim {sim_h:9.3} m, recipe {built:9.3} m");
        assert!(
            (sim_h - built).abs() < 1.0e-6,
            "the simulation stands at {sim_h} m where the recipe built {built} m at \
             ({x}, {z}) — the terrain entity and the .inf_terrain disagree about \
             where the world is"
        );
        seen.push(built);
    }
    // Anti-vacuity: four probes that all read the same number would agree under
    // any offset at all, and so would four probes on flat sea.
    let lo = seen.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = seen.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!(
        hi - lo > 20.0,
        "the four probes span only {:.3} m of relief, so a displaced terrain \
         could still match them",
        hi - lo
    );
    assert!(
        lo > recipe.sea.level_m,
        "a probe at {lo} m is under the {} m waterline — this arm must stand on land",
        recipe.sea.level_m
    );
}

/// **The island's ground is four real materials, end to end** (wave TER2a).
///
/// Before this wave the island's four `TerrainLayer`s named no material, so the
/// terrain shader's whole per-layer virtual-texture branch was unreachable and
/// the frame reported *zero* virtual textures over 51 km² of ground. Binding
/// them is not one edit: a material GUID on a layer has to survive the level
/// save, the cook's dependency closure, the pack, and the shipped player's
/// binding walk — four separate places that each had a rule for
/// `Material.asset` and none for a terrain layer.
///
/// So this arm walks the whole chain on the real cooked pack and names which
/// link is broken when it breaks, rather than asserting a count at the end and
/// leaving the reader to bisect four crates.
#[test]
fn the_cooked_island_carries_the_ground_its_layers_bind() {
    let tmp = tempfile::tempdir().expect("tmp");
    let pack = cook(tmp.path());
    let source = inf_player::level::PackLevelSource::open(&pack).expect("the pack opens");

    // 1. THE LEVEL. Its four layers name four DISTINCT materials, in the splat
    //    order `inf_island::splat` writes weights in.
    let mut built = inf_player::build_world_from_pack(&source).expect("the world builds");
    let bound: Vec<uuid::Uuid> = {
        let world = built.world.world_mut();
        let mut q = world.query::<&inf_ecs::components::Terrain>();
        let ids: Vec<uuid::Uuid> = q
            .iter(world)
            .flat_map(|t| t.layer_materials().collect::<Vec<_>>())
            .collect();
        ids
    };
    assert_eq!(
        bound.len(),
        4,
        "the cooked island's terrain binds {} layer materials, not four: {bound:?}",
        bound.len()
    );
    let mut distinct = bound.clone();
    distinct.sort();
    distinct.dedup();
    assert_eq!(distinct.len(), 4, "two layers share a material: {bound:?}");
    for (k, kind) in [
        inf_material::ground::GroundKind::Grass,
        inf_material::ground::GroundKind::Rock,
        inf_material::ground::GroundKind::ForestFloor,
        inf_material::ground::GroundKind::Sand,
    ]
    .into_iter()
    .enumerate()
    {
        assert_eq!(
            bound[k],
            inf_editor_core::ground::ground_material_guid(kind),
            "layer {k} is not {} — the splat writes its weights into that \
             channel, so a swap here paints the beaches with rock",
            kind.label()
        );
    }

    // 1a. **AND THE ROAD** (wave ASSET0, clause 0). It is not a terrain layer —
    //     it is a `MeshRef` with a `Material` on it — so it reaches the pack by
    //     the OTHER edge of the same closure, and naming it here is what stops
    //     the count above being satisfied by a sixth ground set.
    {
        let want = inf_editor_core::ground::ground_material_guid(
            inf_material::ground::GroundKind::Asphalt,
        );
        let content = source.material_content();
        assert!(
            content.materials.contains_key(&want),
            "the cooked island carries no asphalt material — the street is back \
             on Material::default()'s 0.8 debug grey (EDIT1 finding 2)"
        );
        println!("ISLAND GROUND: the road binds {want}");
    }

    // 1b. **AND THE LEVEL ITSELF CARRIES NONE OF IT** — the finding this arm
    //     exists to keep found. A cooked partitioned level ships **zero
    //     entities**; they are all in the derived `.inf_part`. So the walk that
    //     collects a level's material bindings found nothing at all on every
    //     partitioned world, for `Material.asset` as much as for a terrain
    //     layer, and every surface in one shipped untextured. Nothing caught it
    //     because until TER2a no content in this repository had a texture to
    //     lose: the island reported "0 virtual textures" over 51 km² of ground,
    //     which read as "the ground names no material" and was also true.
    //
    //     Asserted rather than commented, so the day the cook stops emptying a
    //     partitioned level this arm says so instead of silently measuring a
    //     path that no longer exists.
    {
        let raw = inf_player::level::LevelSource::level_bytes(&source).expect("level bytes");
        let lvl = inf_scene::decode(&raw).expect("level decodes");
        assert!(
            lvl.entities.is_empty(),
            "the cooked island level carries {} entities — if a partitioned \
             level now keeps them, `material_content`'s partition walk is \
             double-counting and this arm's premise has changed",
            lvl.entities.len()
        );
        println!(
            "ISLAND GROUND: the cooked level carries 0 entities (they are in the \
             .inf_part), so every binding below came from the partition walk"
        );
    }

    // 2. THE PACK. The cook's closure followed the level's sidecar edge to each
    //    `.inf_mat`, derived a `.inf_matd` for it, and followed THAT to the
    //    `.inf_tex` containers.
    let content = source.material_content();
    // **SIX since wave ROAD1, and the sixth is the KERB.** Four come from
    // `Terrain.layers[*].material`, the fifth from the `Roads` entity's
    // `Material.asset` (ASSET0 — the reason the street stopped shading off
    // `Material::default()`'s 0.8 debug grey) and the sixth from the `Kerbs and
    // pavements` entity's. A drop back to five is the kerb's binding lost, and a
    // drop to four is the road's.
    //
    // **The two MARKING entities add none, and that is the design rather than an
    // omission**: road paint is a 100 mm constant-colour film, so they carry
    // `Material::asset: None` — the scalars-only path — and a texture of one
    // would be 1.6 MB of a single texel. So the closure grows by exactly one
    // material for three new entities.
    //
    // **SEVEN since the wave CHAR1a audit, and the seventh is the HERO'S SKIN.**
    // `SceneDoc::edit_create_character` inserted no `Material` at all, so the
    // `Starter_Skin.inf_mat` the New Character wizard writes beside every body —
    // the file `inf-import --rebind-character` fills with the imported
    // mannequin's albedo, normal and ORM — was bound by nothing, reached the
    // cook's closure through nothing, and shipped in no pack. The character drew
    // the renderer's neutral 0.8 grey in both hosts. The door binds it now, the
    // island's `.inf_lvl` was re-blessed with that cause, and the closure follows
    // the edge: a drop back to six is the hero's skin unbound again.
    assert_eq!(
        content.materials.len(),
        7,
        "the pack carries {} derived material records, not seven — the cook's \
         closure did not follow `Terrain.layers[*].material`, the road's \
         `Material.asset`, the kerb's, or the hero's skin",
        content.materials.len()
    );
    // Twenty-one: six albedo + six normal + six ORM + three detail (grass, rock
    // and asphalt are the sets that ship one; the concrete deliberately does
    // not — a pavement is walked on rather than stared at).
    assert_eq!(
        content.textures.len(),
        21,
        "the pack carries {} surface textures, not twenty-one",
        content.textures.len()
    );

    // 3. THE RESIDENCY. The registration set a shipped host builds from this
    //    content, and the deterministic floor it admits — the numbers the frame
    //    instrument's "N virtual textures" line reports.
    let mats = content.vt_materials();
    // Seven since the wave CHAR1a audit — the hero's skin (see the pack
    // assertion above). It contributes no TEXTURE to the counts below: the
    // committed `Starter_Skin.inf_mat` is a scalars-only dielectric, and it is
    // `inf-import --rebind-character` that fills it with an imported body's
    // maps, in a local project this gate never opens.
    assert_eq!(mats.len(), 7, "the host's material map is not the pack's");
    let order = inf_render::registration_order(&mats);
    assert_eq!(
        order.len(),
        21,
        "the registration order names {} textures, not twenty-one — `want_floor` \
         is a pure function of this sequence, so it is the thing two hosts have \
         to agree about",
        order.len()
    );
    println!(
        "ISLAND GROUND: 4 layers + 1 road + 1 kerb + 1 hero skin -> 7 materials \
         -> {} textures; registration order {:?}",
        content.textures.len(),
        order
            .iter()
            .map(|g| format!("{:x}", *g as u64))
            .collect::<Vec<_>>()
    );

    // 3b. **THE GROUND COVER, AS FAR AS THE PACK** (clause 5). The island's three
    //     scatter kinds bind real meshes now — all three carried `mesh: None` —
    //     and the cook had no `.inf_pcg` -> scatter-mesh edge, so this is the arm
    //     that says both halves landed.
    //
    //     **It ends at the pack, and the TER2a audit made that explicit.** What
    //     this block proves is that the bytes are cooked and reachable — the
    //     fourth link of a five-link chain, and never the fifth. Wave TER2b
    //     closed the fifth: `the_scattered_cover_draws_its_authored_meshes`
    //     below drives the shipped projector over the real population and
    //     asserts the three meshes reach three geometry uploads. This block
    //     stays because a mesh that never reaches the pack cannot be drawn
    //     whatever the projector does.
    {
        let reader = std::sync::Arc::new(
            inf_asset::PackReader::open(&pack.join(inf_player::level::PACK_FILE))
                .expect("the pack maps"),
        );
        for kind in inf_editor_core::cover::CoverKind::ALL {
            let id = inf_asset::AssetId(inf_editor_core::cover::cover_mesh_guid(kind));
            assert!(
                reader.contains(id),
                "the cooked island scatters {} and the pack does not carry its \
                 mesh -- the `.inf_pcg` -> scatter-kind-mesh edge did not close",
                kind.label()
            );
            let bytes = reader.read(id).expect("the mesh reads");
            let mesh: inf_mesh::MeshAsset =
                inf_asset::decode(&bytes).expect("the cover mesh decodes");
            let tris: usize = mesh.submeshes.iter().map(|s| s.triangle_count()).sum();
            assert!(tris > 0, "{} ships an empty mesh", kind.label());
            println!(
                "ISLAND COVER: {} -> {tris} triangles, {:.3} m tall",
                kind.label(),
                mesh.bounds.max[1] - mesh.bounds.min[1]
            );
        }
    }

    // 4. AND IT IS NOT VACUOUS. Every texture the records name really is in the
    //    pack — a record naming bytes that are absent renders untextured and the
    //    counts above would still be four and fourteen.
    for g in &order {
        assert!(
            content.textures.contains_key(&uuid::Uuid::from_u128(*g)),
            "the pack names texture {g:x} and does not carry it"
        );
    }
}

/// **The scattered cover draws its authored meshes** (wave TER2b).
///
/// # What this arm replaces
///
/// `the_cover_meshes_are_shipped_and_are_not_yet_drawn` — a TER2a-audit tripwire
/// written to assert the WRONG outcome, so that the day a real mesh reached the
/// scatter path it would go red and take the ledger with it. It has now gone red,
/// and it did so in the strongest form it could: its two struct literals stopped
/// **compiling**, because `inf_pcg::PcgInstance` and
/// `inf_ecs::components::ScatteredInstance` each grew the `mesh` field the arm
/// existed to say did not exist.
///
/// What was wrong, and is not any more:
///
/// * `PcgKind::mesh` was read by the packager's dependency closure and by nothing
///   that draws. `rules::evaluate_with_in` now resolves it where the rule that
///   owns the palette is still in hand, and it rides on the instance. It has to
///   be the GUID and not an index: `kind_index` is **rule-local**, populations
///   from every rule of every layer of every biome graph are concatenated with no
///   run boundaries, and `compose_volume` interleaves grammar module indices into
///   the same `u32`.
/// * `push_scatter` built one `ScatterData::build(PrimMesh::Cube, …)` for every
///   instance. It buckets by mesh now, and builds one batch per authored mesh.
/// * `inf_render::ScatterBatch` had nowhere to put geometry. `ScatterData` now
///   carries an `Option<Arc<ScatterGeometry>>`, folded into its content key, and
///   the scatter raster pulls that batch's own vertices and indices out of two
///   storage buffers instead of the shared built-in pack.
///
/// # It drives the SHIPPED door on the REAL island
///
/// The arm it replaces asserted pack membership, which is the fourth link of a
/// five-link chain — the TER2a audit's law: *when a clause's title is about what
/// the world looks like, at least one arm has to be about what the world looks
/// like.* So this one cooks the island, boots the shipping sim, drives it until
/// the ground is resident and the population has scattered, and projects it
/// through `project_scene_full` — the very function the windowed player calls —
/// with the very table `load_scatter_meshes` builds at boot.
///
/// Its anti-vacuity half is the **same projection with an empty table**: that is
/// the pre-TER2b engine exactly, and it must produce one meshless batch where the
/// real one produces three mesh-carrying ones. An arm that could not tell those
/// two apart would be measuring nothing.
#[test]
fn the_scattered_cover_draws_its_authored_meshes() {
    let tmp = tempfile::tempdir().expect("a temp dir");
    let pack = cook(tmp.path());
    let reader = inf_asset::PackReader::open(&pack.join(inf_player::level::PACK_FILE))
        .expect("the pack maps");

    // ── 1. the table the shipped player's projector is handed ──
    //
    // Through the SHIPPED door — `inf_player::scatter_mesh::from_pack` is what
    // `load_scatter_meshes` calls at boot — so this is a real run's table, not
    // one the test assembled.
    let meshes = inf_player::scatter_mesh::from_pack(&reader);
    assert_eq!(
        meshes.len(),
        inf_editor_core::cover::CoverKind::ALL.len(),
        "the cooked island's scatter kinds resolve to {} meshes, not the three the \
         cover library authors",
        meshes.len()
    );
    for kind in inf_editor_core::cover::CoverKind::ALL {
        let id = inf_editor_core::cover::cover_mesh_guid(kind);
        let g = meshes
            .get(&id.as_u128())
            .unwrap_or_else(|| panic!("{} did not resolve to scatter geometry", kind.label()));
        assert!(
            g.triangle_count() > 0 && g.vertex_count() > 0,
            "{} resolved to an empty mesh",
            kind.label()
        );
        assert!(
            g.radius > 0.0 && g.radius < 2.0,
            "{} has a unit bounding radius of {} m, which is not ground cover -- \
             the cull sphere and the impostor card are both sized from it",
            kind.label(),
            g.radius
        );
        // NOT VACUOUS: the resolved geometry is the committed mesh's, through the
        // same one door, and not something the loader synthesized.
        let bytes = reader.read(inf_asset::AssetId(id)).expect("the mesh reads");
        let mesh: inf_mesh::MeshAsset = inf_asset::decode(&bytes).expect("the cover mesh decodes");
        let (p, n, _u, _t, i) = mesh.vgeom_streams();
        assert_eq!(
            g.key(),
            inf_render::ScatterGeometry::from_streams(&p, &n, &i).key(),
            "{}'s resolved geometry is not the committed mesh's",
            kind.label()
        );
        println!(
            "ISLAND COVER: {} -> {} triangles, {} vertices, r = {:.3} m",
            kind.label(),
            g.triangle_count(),
            g.vertex_count(),
            g.radius
        );
    }

    // ── 2. the island's own document names them ──
    let doc = inf_editor_core::island::island_cover_document(7);
    let kinds: Vec<&inf_pcg::PcgKind> = doc
        .layers
        .iter()
        .flat_map(|l| &l.rules)
        .flat_map(|r| &r.kinds)
        .collect();
    assert_eq!(kinds.len(), 3, "the island's cover document is three kinds");
    for (k, cover) in kinds.iter().zip(inf_editor_core::cover::CoverKind::ALL) {
        assert_eq!(
            k.mesh,
            Some(inf_editor_core::cover::cover_mesh_guid(cover)),
            "the island's {} kind names no mesh",
            cover.label()
        );
    }

    // ── 3. the shipping projection, on the real population ──
    let mut sim = pack_sim(&pack);
    let t = drive(&mut sim, start());
    let placed = *t.veg_len.last().expect("the drive traced");
    assert!(
        placed > 0,
        "nothing scattered on the drive, so the projection below would be vacuous"
    );
    // Every instance the sim placed carries the GUID its kind resolved to — the
    // link the audit found broken, asserted on the WORLD rather than on a
    // constructed record.
    {
        let world = sim.world().world();
        let mut named = 0usize;
        let mut total = 0usize;
        for e in world.iter_entities() {
            let Some(terrain) = e.get::<inf_ecs::components::Terrain>() else {
                continue;
            };
            for i in &terrain.biome_population {
                total += 1;
                if i.mesh.is_some_and(|m| meshes.contains_key(&m.as_u128())) {
                    named += 1;
                }
            }
        }
        assert_eq!(
            named, total,
            "{named} of {total} scattered instances name a resolvable mesh"
        );
    }

    let project = |sim: &RuntimeSim, table: &inf_render::ScatterMeshes| {
        let mut scene = inf_render::RenderScene::default();
        inf_player::render::project_scene_full(
            &mut scene,
            sim,
            1.0,
            &inf_player::vmesh::VmeshRegistry::new(),
            &inf_player::skinned::SkinnedRegistry::new(),
            &inf_voxel::VoxelVolumes::new(),
            &mut inf_render::DebrisCache::default(),
            None,
            table,
        );
        scene
    };

    // **The table the SHIPPED boot actually builds** (island wave I8b):
    // `load_scatter_meshes` is `from_pack` *plus* the twelve building module
    // families, which name no file and are minted from their own names. The
    // cover assertions above deliberately run on `meshes` alone — they are about
    // the `.inf_pcg` -> scatter-kind-mesh edge and adding twelve engine defaults
    // to that count would make them a statement about the palette.
    let shipped = {
        let mut t = meshes.clone();
        inf_player::scatter_mesh::add_building_modules(&mut t);
        assert!(
            t.len() > meshes.len(),
            "the module families added nothing to the table"
        );
        t
    };
    let scene = project(&sim, &shipped);
    let with_geom: Vec<&inf_render::ScatterBatch> = scene
        .scatter
        .iter()
        .filter(|b| b.data.geometry.is_some())
        .collect();
    let drawn: std::collections::BTreeSet<u128> = with_geom
        .iter()
        .filter_map(|b| b.data.geometry.as_ref().map(|g| g.key()))
        .collect();
    // **Three, AMONG the settlements' own** (island wave I8b). This used to be
    // `drawn.len() == 3`, and it could be: the island's whole mesh-carrying
    // population was the biome-bound cover. A settlement block's grammar modules
    // draw their own shape families now, through the same bucketing, so the
    // distinct-upload count is the cover's three plus whichever of the twelve
    // families the resident blocks used. Three distinct uploads FOR THE COVER is
    // still the claim, and the loop below asserts it by name rather than through
    // a total.
    assert!(
        drawn.len() >= 3,
        "the island's three cover meshes must reach three DISTINCT geometry \
         uploads; the projection produced {} of {} scatter batches carrying \
         geometry and {} distinct uploads",
        with_geom.len(),
        scene.scatter.len(),
        drawn.len()
    );
    for kind in inf_editor_core::cover::CoverKind::ALL {
        let id = inf_editor_core::cover::cover_mesh_guid(kind);
        let key = meshes.get(&id.as_u128()).expect("resolved").key();
        assert!(drawn.contains(&key), "{} is not drawn", kind.label());
    }
    let instances: usize = with_geom.iter().map(|b| b.data.len()).sum();
    assert!(
        instances >= placed,
        "{instances} of {placed} scattered instances reached a mesh-carrying \
         batch — the vegetation alone is {placed}"
    );

    // ── 3b. THE ZERO-PLACEHOLDER ARM (island wave I8b) ──
    //
    // *"A shipped city draws zero placeholder batches"* — the routed project's
    // own first requirement, as a measurement over a real cooked pack rather
    // than over a hand-built fixture. (The I4 audit deleted the first attempt at
    // it as a tautology: that scene was built with `..Default::default()`, so
    // `scatter.is_empty()` asserted that `Vec::new()` is empty.)
    //
    // "Placeholder" means exactly one thing here: a batch whose `geometry` is
    // `None`, which is a batch drawing `PrimMesh::Cube` for content that is not
    // a cube. The **shell** batch is the one thing that legitimately is one — a
    // shell is an oriented box by definition (`push_shells`) — so it is counted
    // apart and asserted non-empty rather than waved through.
    let shells: Vec<&inf_render::ScatterBatch> = scene
        .scatter
        .iter()
        .filter(|b| b.near_distance > 0.0)
        .collect();
    let placeholder: Vec<&inf_render::ScatterBatch> = scene
        .scatter
        .iter()
        .filter(|b| b.data.geometry.is_none() && b.near_distance == 0.0)
        .collect();
    println!(
        "ZERO PLACEHOLDER: {} batches, {} carry geometry, {} are shell boxes, \
         {} are placeholders",
        scene.scatter.len(),
        with_geom.len(),
        shells.len(),
        placeholder.len()
    );
    assert!(
        placeholder.is_empty(),
        "{} of {} scatter batches draw a placeholder cube for content that is \
         not one",
        placeholder.len(),
        scene.scatter.len()
    );
    assert!(
        !shells.is_empty(),
        "no shell batch at all, so the count above is not measuring a city"
    );

    // …and the CASTER half. Every settlement PARTS batch says it does not cast,
    // because its own shell does — so the CPU caster pack walks none of them.
    // The number asserted is `considered`, the count BEFORE the ceiling bites,
    // which is what says the instances were never touched rather than packed
    // and thrown away.
    let caster_settings =
        inf_render::shadow_caster_settings(&inf_render::RenderSettings::default().scatter, 400.0);
    // **The eye stands in a settlement**, not at the world origin: a pack taken
    // from a kilometre away considers nothing, and the comparison below would be
    // two zeros.
    let eye = {
        // Standing ON a building part, so the pack has something to consider:
        // a settlement pad sits a couple of hundred metres up and the batch
        // anchor's own Y is the volume entity's, which is zero.
        let b = scene
            .scatter
            .iter()
            .find(|b| !b.casts_shadows && !b.data.instances.is_empty())
            .expect("a settlement parts batch");
        let o = b.data.instances[0].offset;
        b.anchor + glam::DVec3::new(o[0] as f64, o[1] as f64, o[2] as f64)
    };
    let origin = inf_math::FloatingOrigin::new(eye);
    let pack = |batches: &[inf_render::ScatterBatch]| {
        inf_render::pack_fallback(
            &origin,
            batches,
            eye,
            &caster_settings,
            inf_render::MAX_CPU_SCATTER_INSTANCES,
            inf_render::PackPurpose::Casters,
        )
    };
    let quiet = pack(&scene.scatter);
    // **THE FALSIFIER**: the same scene with every batch casting — which is the
    // pre-I8b engine exactly — walks the settlements' parts and considers an
    // order of magnitude more.
    let loud_batches: Vec<inf_render::ScatterBatch> = scene
        .scatter
        .iter()
        .map(|b| inf_render::ScatterBatch {
            casts_shadows: true,
            ..b.clone()
        })
        .collect();
    let loud = pack(&loud_batches);
    let total_instances: usize = scene.scatter.iter().map(|b| b.data.len()).sum();
    let casting: usize = scene
        .scatter
        .iter()
        .filter(|b| b.casts_shadows)
        .map(|b| b.data.len())
        .sum();
    println!(
        "ZERO PLACEHOLDER CASTERS: {casting} of {total_instances} instances sit \
         in a casting batch; the pack considered {} where the pre-I8b engine \
         considered {}",
        quiet.considered, loud.considered
    );
    assert!(
        casting < total_instances,
        "every one of {total_instances} instances is still a caster — the parts \
         batches did not opt out"
    );
    assert!(
        loud.considered > 0,
        "the control pack considered nothing, so the comparison is two zeros"
    );
    assert!(
        quiet.considered * 4 < loud.considered,
        "the caster pack considered {} of the control's {} — the parts are still \
         being walked",
        quiet.considered,
        loud.considered
    );

    // ── 3c. THE NIGHT WINDOWS (island wave I8b clause 3) ──
    //
    // Every scene projected so far was projected at the island's committed
    // hour, and **not one batch emits** — which is the first half of the claim
    // and the reason a level with no night in it is byte-identical to the
    // pre-I8b engine. Then the clock is wound to midnight through the world's
    // own `TimeOfDay`, the SAME projection runs, and the panes light up.
    //
    // Wound in the world rather than passed to the projector, because the hour
    // reaches the projection through `project_sky` -> `scene.sun.direction` and
    // that is the seam a shipped build uses.
    //
    // **AUTHORED EMISSION IS NOT THE NIGHT-WINDOW RAMP** (wave EMS1). This arm
    // used to read "not one batch emits by day", and that was a PROXY: it was
    // true only because no resident block had ever held an authored emitter. A
    // venue's neon and an institution's door lamp emit at every hour BY DESIGN
    // (`PcgSurface::emissive`, chosen so a building an emergency is called to
    // can be found at night), so the proxy went false the moment a fire hall
    // stood on the fixture's crossroads — and the assertion it stood in for was
    // never about them. What has to be true for the comparison below to mean
    // anything is that the GLAZING is dark by day, which is exactly the set of
    // batches that light up when the clock is wound. So the day set is recorded
    // and the night claim is made about the DIFFERENCE.
    let day_lit: Vec<bool> = scene
        .scatter
        .iter()
        .map(|b| b.emissive != [0.0; 3])
        .collect();
    let day_lit_n = day_lit.iter().filter(|b| **b).count();
    println!(
        "AUTHORED BY DAY: {day_lit_n} of {} batches emit before the clock is \
         wound — a venue's neon and a civic lamp, and not one window",
        scene.scatter.len()
    );
    let glowing_instances: usize = {
        let w = sim.world().world();
        let mut n = 0usize;
        for e in w.iter_entities() {
            if let Some(v) = e.get::<inf_ecs::components::PcgVolume>() {
                n += v.evaluated.iter().filter(|i| i.glow > 0.0).count();
            }
        }
        n
    };
    assert!(
        glowing_instances > 0,
        "the settlements hold no glazed module at all, so nothing could light up"
    );
    {
        let w = sim.world_mut().world_mut();
        let mut q = w.query::<&mut inf_ecs::components::TimeOfDay>();
        let mut wound = 0usize;
        for mut tod in q.iter_mut(w) {
            // Local midnight at this island's longitude.
            tod.seconds = (86_400.0 - tod.longitude_deg * 240.0).rem_euclid(86_400.0);
            wound += 1;
        }
        assert!(wound > 0, "the island carries no clock to wind");
    }
    let night = project(&sim, &shipped);
    // The two projections are the same population under two clocks, so the
    // batches pair by index — asserted, because an index pairing that silently
    // slipped would compare a window against a neon sign.
    assert_eq!(
        night.scatter.len(),
        day_lit.len(),
        "the night projection produced a different batch set from the day one, \
         so the day/night difference below pairs nothing"
    );
    // **The batches that were NOT lit by day and are lit now** — the glazing,
    // and the only thing this arm has ever been about.
    let lit: Vec<&inf_render::ScatterBatch> = night
        .scatter
        .iter()
        .zip(day_lit.iter())
        .filter(|(b, was)| !**was && b.emissive != [0.0; 3])
        .map(|(b, _)| b)
        .collect();
    let lit_instances: usize = lit.iter().map(|b| b.data.len()).sum();
    println!(
        "NIGHT WINDOWS: sun y {:.3} -> glow step {}; {} of {} batches emit, over \
         {lit_instances} of {glowing_instances} glazed instances; brightest {:?}",
        night.sun.direction.y,
        inf_render::night_glow_step(night.sun.direction),
        lit.len(),
        night.scatter.len(),
        lit.iter().map(|b| b.emissive[0]).fold(0.0_f32, f32::max)
    );
    assert!(
        night.sun.direction.y < 0.0,
        "the clock did not put the sun below the horizon: y = {}",
        night.sun.direction.y
    );
    assert!(
        !lit.is_empty(),
        "no batch lights up between day and midnight — every emitter in this \
         scene was already on, so nothing here measures the night-window ramp"
    );
    assert_eq!(
        lit_instances, glowing_instances,
        "{lit_instances} instances reached a lit batch of {glowing_instances} \
         that carry a glow"
    );
    for b in &lit {
        assert!(
            b.emissive[0] > b.emissive[1] && b.emissive[1] > b.emissive[2],
            "a lit window is not warm: {:?}",
            b.emissive
        );
        assert!(
            b.data.geometry.is_some(),
            "a lit batch draws a placeholder — a glowing cube is worse than a \
             dark window"
        );
    }
    // Every batch's cull radius is its OWN geometry's, not the proxy's — the one
    // place the proxy must not be used, because a radius that is too small
    // deletes instances at the frustum edge.
    for b in &with_geom {
        let want = b.data.geometry.as_ref().expect("filtered").radius;
        assert_eq!(b.data.bounding_radius(), want);
    }

    // ── 4. …and the anti-vacuity half: the pre-TER2b engine, exactly ──
    let before = project(&sim, &inf_render::ScatterMeshes::new());
    assert!(
        before.scatter.iter().all(|b| b.data.geometry.is_none()),
        "an empty scatter-mesh table must produce the placeholder path -- if it \
         does not, the arm above is not measuring the table"
    );
    // **What "the pre-TER2b engine" is, restated for a world with settlements in
    // it** (island wave I8a). This used to assert `before.scatter.len() == 1`:
    // the island's whole population was the biome-bound vegetation, which is one
    // batch per terrain. Wave I8a stands 172 settlement blocks on the island, and
    // a block's grammar modules go through the same `push_scatter` body, so the
    // batch count is now the vegetation's one plus one per resident block — the
    // fixture measures nine. A count was never the claim; **not one batch carries
    // geometry** is, and it is asserted above.
    assert!(
        !before.scatter.is_empty(),
        "the placeholder projection produced no batch at all, so it is not the \
         same population"
    );
    assert_eq!(
        before.scatter.iter().filter(|b| !b.data.is_empty()).count(),
        before.scatter.len(),
        "an empty batch reached the projection"
    );
    println!(
        "PLACEHOLDER PROJECTION: {} batches, none carrying geometry (the \
         pre-TER2b engine); the settlements are {} of them",
        before.scatter.len(),
        before.scatter.len().saturating_sub(1)
    );

    // The PROXY primitive is still a cube, and that is a CARRIED BOUND rather
    // than an oversight: the impostor card, the CPU fallback and the cascade
    // shadow caster pack all draw it, because those three bind one shared vertex
    // buffer for the whole frame and a per-batch mesh does not fit in it. The
    // impostor is at least sized off the authored radius (`material.w` in
    // `scatter_mesh.wgsl`); the other two are not, and that is named in the
    // wave's carried list.
    assert!(
        scene
            .scatter
            .iter()
            .chain(&before.scatter)
            .all(|b| b.data.mesh == inf_render::PrimMesh::Cube),
        "the scatter proxy primitive moved -- if that is deliberate, the carried \
         bound about impostors, the CPU fallback and shadow casters has to be \
         rewritten with it"
    );

    println!(
        "ISLAND COVER: {placed} instances -> {} batches carrying 3 distinct \
         geometry uploads (was 1 placeholder batch); cull radii off the meshes \
         themselves; proxy primitive still a cube for the impostor / \
         CPU-fallback / shadow-caster paths.",
        with_geom.len()
    );
}

// ── THE SETTLEMENT GATE (island wave I8a) ───────────────────────────────────
//
// Everything below is about the thing wave I8a put on the island's seven pads:
// the blocks, the buildings they stand, the doors those buildings offer, and
// whether a player can walk in through one and go upstairs.
//
// It runs on the CI-scale fixture for the same reason the drive above does — the
// shipped island's terrain is 549.9 MB and is not committed — and it measures a
// SHIPPED-ISLAND city block directly where the block's own size is what is being
// priced (the fixture's reservations take the town's 76 m grid; a Harbour City
// core block is 100 m, and a battery about a city block has to be about one).

/// The band radius the collider band admits solids inside, for the numbers the
/// furnish battery prints. `inf_ecs::band`'s own default, named here so the
/// battery cannot drift from the thing it prices.
const BAND_NEAR_M: f64 = inf_ecs::band::DEFAULT_COLLIDER_NEAR_M;

/// An archetype whose own storey range starts at two or more — a building that
/// is guaranteed to have a stair whatever seed it draws.
///
/// The walk needs one: `Shop` is `(1, 2)` and `House` is `(1, 3)`, so half the
/// buildings in a town have no flight at all and "climb a stair" would be a
/// claim about which seed came up.
fn always_multistorey(a: inf_pcg::ArchetypeId) -> bool {
    inf_pcg::archetype(a).floors.0 >= 2
}

/// Where a settlement walk goes: the settlement holding the most blocks that are
/// **guaranteed** multi-storey, tie-broken by block count and then by name.
///
/// A pure function of the committed design, so both hosts are handed the same
/// number and nothing else.
fn walk_target_settlement(
    design: &inf_island::IslandDesign,
) -> inf_editor_core::settlement::Settlement {
    let tall = |s: &inf_editor_core::settlement::Settlement| {
        s.blocks
            .iter()
            .filter(|b| always_multistorey(b.archetype))
            .count()
    };
    let mut plans = inf_editor_core::settlement::settlements(design);
    plans.sort_by(|a, b| {
        tall(b)
            .cmp(&tall(a))
            .then(b.blocks.len().cmp(&a.blocks.len()))
            .then(a.name.cmp(&b.name))
    });
    let best = plans
        .into_iter()
        .next()
        .expect("the design has a settlement");
    assert!(
        tall(&best) > 0,
        "no settlement on this island has a block that is guaranteed \
         multi-storey, so a walk that climbs a stair has nowhere to go"
    );
    best
}

/// Every `PcgVolume` the simulation currently holds: `(guid, centre, extent,
/// seed)`, in `Guid` order so nothing downstream depends on an archetype walk.
fn resident_volumes(sim: &RuntimeSim) -> Vec<(Uuid, glam::DVec3, glam::DVec2, u32)> {
    let w = sim.world().world();
    let mut out = Vec::new();
    for e in w.iter_entities() {
        let (Some(g), Some(v), Some(t)) = (
            e.get::<inf_ecs::Guid>(),
            e.get::<inf_ecs::components::PcgVolume>(),
            e.get::<inf_ecs::components::GlobalTransform>(),
        ) else {
            continue;
        };
        out.push((
            g.0,
            t.translation(),
            glam::DVec2::new(v.extent.x, v.extent.y),
            v.seed,
        ));
    }
    out.sort_by_key(|(g, _, _, _)| *g);
    out
}

/// Every solid box the simulation currently holds, in `inf_pcg`'s own
/// vocabulary — the one `opening_is_clear` reads.
fn resident_solids(sim: &RuntimeSim) -> Vec<inf_pcg::PcgCollider> {
    let w = sim.world().world();
    let mut out = Vec::new();
    for e in w.iter_entities() {
        let Some(v) = e.get::<inf_ecs::components::PcgVolume>() else {
            continue;
        };
        for s in &v.structures {
            out.push(inf_pcg::PcgCollider {
                center: s.center,
                half_extents: s.half_extents,
                rotation: s.rotation,
            });
        }
    }
    out
}

/// The lowered building passes of one zone document, read out of the built
/// project exactly as the shipped host reads it.
fn zone_passes(content: &Path, a: inf_pcg::ArchetypeId) -> Vec<inf_pcg::BuildingPass> {
    let p = content.join(inf_editor_core::settlement::zone_file_name(a));
    let bytes = std::fs::read(&p).unwrap_or_else(|e| panic!("no {}: {e}", p.display()));
    let payload = inf_pcg::PcgAssetPayload::decode(&bytes).expect("the zone document decodes");
    let graph = payload.graph().expect("the graph is the source of truth");
    let lowered = inf_pcg::lower_graph(&graph, &inf_pcg::pcg_registry());
    assert!(lowered.ok, "{}: {:?}", a.name(), lowered.issues);
    assert_eq!(
        lowered.buildings.len(),
        1,
        "{} lowers to one building pass",
        a.name()
    );
    lowered.buildings
}

/// **EVERY SETTLEMENT BUILDING IS ENTERABLE, AT SETTLEMENT SCALE** (island wave
/// I8a, clause 3).
///
/// The three phase-19 invariants — `rooms_connected`, `floors_reachable`,
/// `opening_is_clear` — run per **sampled building** over the blocks the
/// simulation is actually holding, against the solids the shipped world
/// actually built. Phase 19 ran them over seven hand-placed lots; this runs them
/// over a settlement.
///
/// It also prints what clause 3 asks for: the doorways per settlement, and the
/// share of them the collider band makes solid.
#[test]
fn every_settlement_building_is_enterable() {
    let tmp = tempfile::tempdir().expect("a temp dir");
    let proj = build_project(tmp.path());
    let content = proj.join("Content");
    let pack = cook(tmp.path());
    let recipe =
        inf_island::IslandRecipe::load(&fixture_recipe()).expect("the fixture recipe loads");
    let design = inf_island::read_design(&recipe).expect("the design reads");
    let mut sim = pack_sim(&pack);
    let hero = hero_entity(&sim).expect("a hero");
    let mut total_buildings = 0usize;
    let mut total_doorways = 0usize;

    // **Every settlement, one at a time.** They are kilometres apart and the
    // partition holds one neighbourhood at once, so "per settlement" is what the
    // hero walking to each of them means.
    for plan in inf_editor_core::settlement::settlements(&design) {
        set_hero(
            &mut sim,
            hero,
            glam::DVec3::new(plan.centre.x, 0.0, plan.centre.y),
        );
        for _ in 0..8 {
            sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
        }

        let volumes = resident_volumes(&sim);
        let solids = resident_solids(&sim);
        let doorways = inf_ecs::door::volume_doorways(sim.world());
        println!(
            "SETTLED at {} ({:.0}, {:.0}): {} resident volume(s), {} solids, {} doorways",
            plan.name,
            plan.centre.x,
            plan.centre.y,
            volumes.len(),
            solids.len(),
            doorways.len()
        );
        assert!(
            !volumes.is_empty(),
            "no settlement volume is resident — the battery below would be over bare ground"
        );
        assert!(
            !solids.is_empty(),
            "the resident settlement blocks built no solid at all"
        );
        assert!(
            !doorways.is_empty(),
            "the resident settlement blocks planned no doorway — nothing is enterable"
        );

        // **The band, measured rather than described.** A doorway is SOLID when the
        // band admits the building it belongs to; everything past the near radius
        // simulates as a shell, which is the I3 ruling ("doors and walls cannot be
        // solid at different distances").
        let band = inf_ecs::band::SimBand::from_world(
            sim.world(),
            BAND_NEAR_M,
            inf_ecs::band::DEFAULT_COLLIDER_FAR_M,
        );
        let banded = doorways
            .iter()
            .filter(|(_, _, d)| {
                band.tier(
                    d.hinge,
                    glam::DVec3::splat(d.width_m.max(d.height_m) * 0.5),
                    glam::DQuat::IDENTITY,
                ) == inf_math::Tier::Near
            })
            .count();
        println!(
            "DOORWAYS: {} planned, {banded} inside the {BAND_NEAR_M:.0} m collider band ({:.2} %)",
            doorways.len(),
            100.0 * banded as f64 / doorways.len() as f64
        );
        // **AND IT IS AN ARM, NOT A PRINT** (NPC1c audit). This is the one
        // reading in this file that is derived from the hero's own
        // `GlobalTransform` and asserted nowhere — the exact number wave NPC1c
        // reported as **0 of 1 297** when it found `set_hero` leaving the
        // streaming anchor stale. A share of zero here means the hero the band
        // is anchored on is somewhere else entirely, and the loop above would
        // still be green: it asks whether the blocks are *resident*, which is
        // cell activation, not whether anything near them is *solid*.
        assert!(
            banded > 0,
            "{}: 0 of {} doorways are inside the {BAND_NEAR_M:.0} m band — the \
             band's anchor is not where this loop just put the hero, so every \
             band-derived claim in this file is about a different place",
            plan.name,
            doorways.len()
        );

        // ── the three invariants, per building ──
        let by_guid: std::collections::BTreeMap<Uuid, inf_editor_core::settlement::Block> = plan
            .blocks
            .iter()
            .map(|b| {
                (
                    inf_editor_core::settlement::block_guid(&recipe.name, b.site, b.col, b.row),
                    *b,
                )
            })
            .collect();

        let mut buildings = 0usize;
        let mut floors_total = 0u32;
        let mut stairs_total = 0usize;
        let mut doors_total = 0usize;
        let mut by_zone: std::collections::BTreeMap<&'static str, usize> = Default::default();
        for (guid, centre, extent, seed) in &volumes {
            let Some(block) = by_guid.get(guid) else {
                continue;
            };
            let passes = zone_passes(&content, block.archetype);
            let plans = {
                let w = sim.world().world();
                let terrain = w
                    .iter_entities()
                    .find_map(|e| e.get::<inf_ecs::components::Terrain>())
                    .expect("the island has ground");
                let height = inf_pcg::FnHeight::new(|x: f64, z: f64| {
                    terrain.data.height_at(glam::DVec2::new(x, z))
                });
                let cx = inf_pcg::GrammarContext {
                    entity: Some(*guid),
                    center: *centre,
                    extent: *extent,
                    seed_offset: u64::from(*seed),
                };
                inf_pcg::plans_of(&passes, &inf_pcg::NoSplines, &height, &cx)
            };
            assert!(
                !plans.is_empty(),
                "{}'s {} block resolved no building at all — a `Ground::Terrain` lot \
             over unpaged ground fails closed, which is right, but this block's \
             ground IS resident",
                plan.name,
                block.archetype.name()
            );
            *by_zone.entry(block.archetype.name()).or_default() += plans.len();
            buildings += plans.len();
            for p in &plans {
                floors_total += p.floors;
                stairs_total += p.stairs.len();
                // 1. every floor's room graph is connected through doors.
                for f in 0..p.floors {
                    assert!(
                        p.rooms_connected(f),
                        "{} {}: floor {f}'s room graph is not connected",
                        plan.name,
                        block.archetype.name()
                    );
                }
                // 3. and every floor is reachable from OUTSIDE.
                assert!(
                    p.entrance.is_some(),
                    "{} {}: no entrance — the building is sealed",
                    plan.name,
                    block.archetype.name()
                );
                assert!(
                    p.floors_reachable(),
                    "{} {}: a floor cannot be reached from outside",
                    plan.name,
                    block.archetype.name()
                );
                assert_eq!(
                    p.stairs.len(),
                    (p.floors - 1) as usize,
                    "{} {}: wrong flight count for {} floors",
                    plan.name,
                    block.archetype.name(),
                    p.floors
                );
                // 2. no solid the SHIPPED world built intrudes into a door's void.
                let doors: Vec<&inf_pcg::Opening> = p
                    .openings
                    .iter()
                    .filter(|o| o.kind == inf_pcg::OpeningKind::Door)
                    .collect();
                doors_total += doors.len();
                for (i, d) in doors.iter().enumerate() {
                    assert!(
                        p.opening_is_clear(d, &solids, 0.02),
                        "{} {}: door {i} on wall {} is blocked by a collider the \
                     shipped world built",
                        plan.name,
                        block.archetype.name(),
                        d.wall
                    );
                }
                // **THE CONTROL.** Every assertion above is "no solid is here", and
                // such an assertion passes trivially if the predicate cannot say no.
                // A slab through the whole building must read BLOCKED.
                let f = p.footprint;
                let top = p.floor_y(p.floors);
                let slab = [inf_pcg::PcgCollider {
                    center: glam::DVec3::new(f.center().x, (p.base_y + top) * 0.5, f.center().y),
                    half_extents: glam::DVec3::new(
                        f.size_x(),
                        (top - p.base_y) * 0.5 + 2.0,
                        f.size_z(),
                    ),
                    rotation: glam::DQuat::IDENTITY,
                }];
                for d in &doors {
                    assert!(
                        !p.opening_is_clear(d, &slab, 0.02),
                        "a door reads CLEAR through a solid building — the \
                     enterability predicate is vacuous at settlement scale"
                    );
                }
            }
        }
        println!(
            "ENTERABILITY {}: {buildings} buildings over {} resident blocks, \
         {floors_total} storeys, {stairs_total} flights, {doors_total} door \
         openings; by zone {by_zone:?}",
            plan.name,
            volumes.len()
        );
        assert!(
            buildings >= 8,
            "only {buildings} buildings resident at {} — this is not a \
         settlement-scale sample",
            plan.name
        );
        assert!(
            stairs_total > 0,
            "not one building at {} has a stair — 'climb a stair' has nothing to climb",
            plan.name
        );
        total_buildings += buildings;
        total_doorways += doorways.len();
    }
    println!(
        "ENTERABILITY TOTAL: {total_buildings} buildings and {total_doorways} \
         doorways over every settlement of this island, all enterable"
    );
    assert!(total_buildings > 0);
}

/// Does any solid contain `p`?
///
/// The blocks a settlement plans are axis-aligned, so a solid's XZ bounds are
/// exact rather than conservative — `PcgCollider::xz_half_extents` is the same
/// door `solid_bounds` uses and it needs no trigonometry.
fn solid_contains(solids: &[inf_pcg::PcgCollider], p: glam::DVec3) -> usize {
    solids
        .iter()
        .filter(|s| {
            let (ex, ez) = s.xz_half_extents();
            let (lo, hi) = s.y_band();
            p.x >= s.center.x - ex
                && p.x <= s.center.x + ex
                && p.z >= s.center.z - ez
                && p.z <= s.center.z + ez
                && p.y >= lo
                && p.y <= hi
        })
        .count()
}

/// One host's reading of the walk: the state fold per step, and everything the
/// walk **discovered** rather than was told.
///
/// The discovery is deliberately part of the trace. Two hosts handed a
/// hard-coded door guid would agree about it whatever their worlds held; two
/// hosts each asked *"which door is nearest"* agree only if their worlds are the
/// same world.
struct Walk {
    states: Vec<Vec<u8>>,
    door: Uuid,
    prompt: glam::DVec3,
    open_before: bool,
    open_after: bool,
    verdict_moved: bool,
    inside: glam::DVec3,
    upstairs: glam::DVec3,
    inside_blocked: usize,
    upstairs_blocked: usize,
    doorways: usize,
    solids: usize,
    climb_m: f64,
}

/// Steps spent walking from the settlement's centre to the door.
const WALK_STEPS: usize = 40;
/// Steps spent waiting for the leaf to swing after `use_door`.
const SWING_STEPS: usize = 45;
/// Steps spent standing inside, and then upstairs.
const DWELL_STEPS: usize = 15;

/// **THE DOOR THE WALK USES**, and the one place the rule is written.
///
/// Three clauses, and every one of them was paid for:
///
/// * an **exterior ground-floor** doorway, because that is a front door;
/// * on a block whose archetype is **guaranteed multi-storey**, because a
///   `Shop` is one or two storeys and a `House` one to three, so on any other
///   block "climb a stair" would be a claim about which seed came up;
/// * on a block some **street line is level with** (wave EMS1). Every block
///   takes its datum from the terrain under its own footprint (`ground:
///   Terrain`), so on sloping ground two blocks of one settlement are two
///   storeys — and a walk that starts six metres below its destination walks a
///   body at the cut face of the pad. That rule was already downstream, applied
///   to the STREETS after the building had been chosen, which left the choice
///   free to pick a building no street is level with. Measured when the EMS1
///   civic strip claimed `Fixture Town`'s crossroads block: pad at 125.76 m
///   against street lines from 90.02 to 157.24, and not one within two metres.
///
/// **And it is ONE function because it used to be two.** `town_plan` carried a
/// copy under a comment reading *"the same one `walk_into_a_building` picks"* —
/// the A14 restated-rule defect, and it went stale the moment the third clause
/// was added to only one of them: the walk chose a street-level door and the
/// plan chose the fire hall's, and the arm failed on a building nobody was
/// walking into.
fn walk_door(
    sim: &RuntimeSim,
    content: &Path,
    recipe: &inf_island::IslandRecipe,
    plan: &inf_editor_core::settlement::Settlement,
) -> (Uuid, usize, inf_ecs::components::DoorwaySlot) {
    let centre = glam::DVec3::new(plan.centre.x, 0.0, plan.centre.y);
    let tall_blocks: std::collections::BTreeSet<Uuid> = plan
        .blocks
        .iter()
        .filter(|b| always_multistorey(b.archetype))
        .map(|b| inf_editor_core::settlement::block_guid(&recipe.name, b.site, b.col, b.row))
        .collect();
    let w = sim.world().world();
    let terrain = w
        .iter_entities()
        .find_map(|e| e.get::<inf_ecs::components::Terrain>())
        .expect("the island has ground");
    let street_ys: Vec<f64> = plan
        .street_graph()
        .nodes()
        .filter(|n| inf_nav::domain::of(n.id) == inf_nav::domain::STREET)
        .filter_map(|n| {
            terrain
                .data
                .height_at(glam::DVec2::new(n.position.x, n.position.z))
        })
        .collect();
    assert!(
        !street_ys.is_empty(),
        "{}: no street line found ground under it at all",
        plan.name
    );
    // **The pad is the BUILDING'S, not the block's.** A block is subdivided
    // into lots and every building takes its own datum, so a block centre is
    // two metres out on any slope — measured, and it is why the first cut of
    // this filter still chose the fire hall.
    let height =
        inf_pcg::FnHeight::new(|x: f64, z: f64| terrain.data.height_at(glam::DVec2::new(x, z)));
    let volumes = resident_volumes(sim);
    let arch_of: std::collections::BTreeMap<Uuid, inf_pcg::ArchetypeId> = plan
        .blocks
        .iter()
        .map(|b| {
            (
                inf_editor_core::settlement::block_guid(&recipe.name, b.site, b.col, b.row),
                b.archetype,
            )
        })
        .collect();
    let mut plans_of: std::collections::BTreeMap<Uuid, Vec<inf_pcg::BuildingPlan>> =
        Default::default();
    let mut candidates: Vec<(Uuid, usize, inf_ecs::components::DoorwaySlot)> =
        inf_ecs::door::volume_doorways(sim.world())
            .into_iter()
            .filter(|(v, _, d)| d.exterior && d.floor == 0 && tall_blocks.contains(v))
            .collect();
    candidates.sort_by(|a, b| {
        (a.2.hinge - centre)
            .length_squared()
            .total_cmp(&(b.2.hinge - centre).length_squared())
            .then((a.0, a.1).cmp(&(b.0, b.1)))
    });
    for (v, idx, slot) in candidates {
        let plans = plans_of.entry(v).or_insert_with(|| {
            let Some((_, vcentre, vextent, vseed)) =
                volumes.iter().find(|(g, _, _, _)| *g == v).copied()
            else {
                return Vec::new();
            };
            let Some(a) = arch_of.get(&v).copied() else {
                return Vec::new();
            };
            inf_pcg::plans_of(
                &zone_passes(content, a),
                &inf_pcg::NoSplines,
                &height,
                &inf_pcg::GrammarContext {
                    entity: Some(v),
                    center: vcentre,
                    extent: vextent,
                    seed_offset: u64::from(vseed),
                },
            )
        });
        let owner = plans.iter().find(|p| {
            let mut ds = inf_pcg::building::doorways_of(p);
            inf_pcg::building::place_doorways_in_frame(&mut ds, p.frame);
            ds.iter().any(|d| {
                d.hinge.to_array().map(f64::to_bits) == slot.hinge.to_array().map(f64::to_bits)
            })
        });
        let Some(pad) = owner.map(|p| p.floor_y(0)) else {
            continue;
        };
        if street_ys.iter().any(|y| (pad - y).abs() <= 2.0) {
            return (v, idx, slot);
        }
    }
    panic!(
        "{}: no resident multi-storey block offers an exterior door on a storey \
         one of its street lines is level with",
        plan.name
    )
}

/// **THE WALK**: enter the city, find a door, open it, step through, go up.
///
/// Every target is computed from the host's OWN world; nothing is passed in but
/// the settlement's centre, which is a committed number.
fn walk_into_a_building(
    sim: &mut RuntimeSim,
    content: &Path,
    recipe: &inf_island::IslandRecipe,
    plan: &inf_editor_core::settlement::Settlement,
) -> Walk {
    let hero = hero_entity(sim).expect("a hero");
    let centre = glam::DVec3::new(plan.centre.x, 0.0, plan.centre.y);
    let mut states: Vec<Vec<u8>> = Vec::new();

    // ── 1. ENTER THE CITY ── stand at the crossroads and let the cells activate.
    set_hero(sim, hero, centre);
    for _ in 0..8 {
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
        states.push(sim.state_bytes());
    }

    // ── 2. FIND A DOOR ── see `walk_door`, which is the one place the
    //    rule lives.
    let doorways = inf_ecs::door::volume_doorways(sim.world());
    let solids = resident_solids(sim);
    let (vol, idx, slot) = walk_door(sim, content, recipe, plan);
    let door = inf_physics::d3::door::pcg_doorway_guid(vol, idx);
    let placement = inf_physics::d3::door::placement_of(sim.world(), door)
        .expect("the doorway the walk found resolves to a placement");
    let inside_dir = {
        let yaw = slot.inside_yaw_deg.to_radians();
        // `+Z` at zero, `+X` at +90 — the compass the doorway carries. The
        // PORTABLE trig, not `std`'s: this reaches a position two hosts compare,
        // and the P14 law does not stop at a file that happens to be a test.
        glam::DVec3::new(
            inf_math::portable::psin64(yaw),
            0.0,
            inf_math::portable::pcos64(yaw),
        )
    };
    // **The walk arrives from the STREET**, which is the side away from the room
    // the wall serves — the way a person reaches a front door, and the side the
    // gate wants to prove is walkable.
    //
    // It used to be load-bearing for a second reason that is no longer true, and
    // the sentence is corrected rather than deleted because the trap it names is
    // worth remembering: before island wave I8b, `use_door` pressed from the
    // LOCK side (`DoorSpec::lock_side` is `Inside` for a grammar door) on a shut
    // unlocked leaf **locked** it instead of opening it, and the first draft of
    // this walk stood on `prompt_position`, pressed, and got a verdict that did
    // not move — it had bolted the front door from the hall. E means open or
    // close on either face now; the bolt is `lock_door` and its own control.
    let prompt = inf_ecs::door::prompt_position(&placement);
    let approach = slot.hinge - inside_dir * 1.2;

    // ── 3. WALK TO IT ── straight from the crossroads, one step at a time, so
    //    the trace is a walk and not a teleport.
    for k in 1..=WALK_STEPS {
        let t = k as f64 / WALK_STEPS as f64;
        set_hero(sim, hero, centre + (approach - centre) * t);
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
        states.push(sim.state_bytes());
    }

    // ── 4. OPEN IT ── through `use_door`, which is the function the interact
    //    button and the `door.use` node both dispatch to, pressed from the feet
    //    the hero is standing on.
    let feet = glam::DVec3::new(approach.x, approach.y - slot.height_m * 0.5, approach.z);
    let open_before = inf_physics::d3::door::is_open_near(sim.world(), approach);
    let verdict = inf_physics::d3::door::use_door(sim.world_mut(), door, feet);
    for _ in 0..SWING_STEPS {
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
        states.push(sim.state_bytes());
    }
    let open_after = inf_physics::d3::door::is_open_near(sim.world(), approach);

    // ── 5. STEP THROUGH ── a metre and a half along the inside face's own
    //    normal, at knee height, which is where a body would be.
    let inside = slot.hinge + inside_dir * 1.5;
    for _ in 0..DWELL_STEPS {
        set_hero(
            sim,
            hero,
            glam::DVec3::new(inside.x, inside.y - 1.0, inside.z),
        );
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
        states.push(sim.state_bytes());
    }
    let inside_blocked = solid_contains(&solids, inside);

    // ── 6. CLIMB ── the building this door belongs to, its stair core, at floor
    //    one's own walking height. Found by re-deriving the block's plans and
    //    matching the doorway's hinge BIT FOR BIT against the derivation the
    //    shipped host itself ran — not by a search radius.
    let block = plan
        .blocks
        .iter()
        .find(|b| {
            inf_editor_core::settlement::block_guid(&recipe.name, b.site, b.col, b.row) == vol
        })
        .copied()
        .expect("the doorway's volume is a settlement block");
    let passes = zone_passes(content, block.archetype);
    let (upstairs, climb_m) = {
        let volumes = resident_volumes(sim);
        let (_, vcentre, vextent, vseed) = volumes
            .iter()
            .find(|(g, _, _, _)| *g == vol)
            .copied()
            .expect("the volume is resident");
        let w = sim.world().world();
        let terrain = w
            .iter_entities()
            .find_map(|e| e.get::<inf_ecs::components::Terrain>())
            .expect("the island has ground");
        let height =
            inf_pcg::FnHeight::new(|x: f64, z: f64| terrain.data.height_at(glam::DVec2::new(x, z)));
        let cx = inf_pcg::GrammarContext {
            entity: Some(vol),
            center: vcentre,
            extent: vextent,
            seed_offset: u64::from(vseed),
        };
        let plans = inf_pcg::plans_of(&passes, &inf_pcg::NoSplines, &height, &cx);
        let mut found = None;
        for p in &plans {
            let mut ds = inf_pcg::building::doorways_of(p);
            inf_pcg::building::place_doorways_in_frame(&mut ds, p.frame);
            if ds.iter().any(|d| {
                d.hinge.x.to_bits() == slot.hinge.x.to_bits()
                    && d.hinge.y.to_bits() == slot.hinge.y.to_bits()
                    && d.hinge.z.to_bits() == slot.hinge.z.to_bits()
            }) {
                found = Some(p.clone());
                break;
            }
        }
        let p = found.expect(
            "the doorway the world offered is not in any plan the same resolution \
             derives — the shipped population and `plans_of` disagree",
        );
        assert!(
            p.floors >= 2,
            "the building this walk entered is single-storey, so there is no stair \
             to climb"
        );
        // **The stair is what got the hero up; the ROOM is where the hero
        // stands.** The core's own footprint is full of treads at floor one's
        // height — that is what a flight from one to two IS — so a "no solid is
        // here" test aimed at the core measures the staircase and reads
        // blocked-by-2. The claim the walk is making is that a floor above the
        // ground is *reachable and standable*, so the point is the first
        // non-stair room on floor one, and the stair's part is asserted
        // separately: its room on floor one must be in the set reachable from
        // outside.
        assert!(p.core.is_some(), "a multi-storey plan has a stair core");
        let reach = p.reachable_rooms();
        let up = p
            .stair_room(1)
            .expect("a multi-storey plan has a stair room on floor one");
        assert!(
            reach.get(up).copied().unwrap_or(false),
            "the stair room on floor one is not reachable from outside — the \
             flight lands nowhere"
        );
        let (ri, room) = p
            .rooms_on(1)
            .find(|(i, r)| r.kind != inf_pcg::RoomType::Stair && reach[*i])
            .expect("floor one has a room that is not the stairwell");
        assert!(reach[ri]);
        let c = p.frame.to_world(room.rect.center());
        // Floor one's walking surface, plus a knee — the point a body standing on
        // the first floor occupies.
        let y = p.floor_y(1) + 0.5;
        (glam::DVec3::new(c.x, y, c.y), p.floor_y(1) - p.floor_y(0))
    };
    for _ in 0..DWELL_STEPS {
        set_hero(
            sim,
            hero,
            glam::DVec3::new(upstairs.x, upstairs.y - 0.5, upstairs.z),
        );
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
        states.push(sim.state_bytes());
    }
    let upstairs_blocked = solid_contains(&solids, upstairs);

    Walk {
        states,
        door,
        prompt,
        open_before,
        open_after,
        verdict_moved: verdict.moved(),
        inside,
        upstairs,
        inside_blocked,
        upstairs_blocked,
        doorways: doorways.len(),
        solids: solids.len(),
        climb_m,
    }
}

/// **THE SETTLEMENT GATE** (island wave I8a, clause 4): the shipped player and
/// the editor's document walk into the same building, open the same door and
/// climb the same stair, **byte for byte**.
///
/// # Coverage first, because two empty worlds agree perfectly
///
/// Every claim below is asserted on each host *separately* before the two are
/// compared: the settlement is resident, it built solids, it planned doorways,
/// the door the walk found was shut and is open, the doorway is walkable and the
/// first floor is standable. A gate that compared folds alone would certify two
/// hosts that both found nothing — which is exactly the failure the I7 audit
/// found in the drive gate one file up.
///
/// # Why the walk discovers its own target
///
/// Nothing is passed in but the settlement's centre, which is a committed
/// number. Which door is nearest, which building it belongs to and where that
/// building's stair core is are all read out of the host's own world, so two
/// hosts holding different worlds disagree about the targets as well as about
/// the folds.
#[test]
fn pie_equals_shipping_on_a_walk_into_a_building() {
    let tmp = tempfile::tempdir().expect("a temp dir");
    let proj = build_project(tmp.path());
    let content = proj.join("Content");
    let pack = cook(tmp.path());
    let recipe =
        inf_island::IslandRecipe::load(&fixture_recipe()).expect("the fixture recipe loads");
    let slug = inf_island::slug(&recipe.name);
    let design = inf_island::read_design(&recipe).expect("the design reads");
    let plan = walk_target_settlement(&design);

    let mut ship = pack_sim(&pack);
    let mut editor = loose_sim(&content, &slug);
    let a = walk_into_a_building(&mut ship, &content, &recipe, &plan);
    let b = walk_into_a_building(&mut editor, &content, &recipe, &plan);

    for (label, w) in [("shipping", &a), ("document", &b)] {
        println!(
            "WALK ({label}) at {}: {} doorways / {} solids resident; door {} at \
             ({:.2}, {:.2}, {:.2}); shut {} -> open {}; inside ({:.2}, {:.2}, \
             {:.2}) blocked by {}; upstairs ({:.2}, {:.2}, {:.2}) blocked by {}; \
             climbed {:.2} m",
            plan.name,
            w.doorways,
            w.solids,
            w.door,
            w.prompt.x,
            w.prompt.y,
            w.prompt.z,
            !w.open_before,
            w.open_after,
            w.inside.x,
            w.inside.y,
            w.inside.z,
            w.inside_blocked,
            w.upstairs.x,
            w.upstairs.y,
            w.upstairs.z,
            w.upstairs_blocked,
            w.climb_m
        );
        // ── coverage, on each host, before either is compared to the other ──
        assert!(w.doorways > 0, "{label}: no doorway was resident at all");
        assert!(w.solids > 0, "{label}: the settlement built no solid");
        assert!(
            !w.open_before,
            "{label}: the door was already open, so opening it proves nothing"
        );
        assert!(
            w.verdict_moved,
            "{label}: `use_door` refused — the door the walk found is not usable"
        );
        assert!(
            w.open_after,
            "{label}: the door did not open in {SWING_STEPS} steps"
        );
        assert_eq!(
            w.inside_blocked, 0,
            "{label}: a solid stands where the walk stepped through the doorway — \
             the door opens onto a wall"
        );
        assert_eq!(
            w.upstairs_blocked, 0,
            "{label}: a solid stands in the stair core on the first floor — the \
             stairwell is filled in"
        );
        assert!(
            w.climb_m > 2.0,
            "{label}: the first floor is {:.2} m up, which is not a storey",
            w.climb_m
        );
        assert_eq!(
            w.states.len(),
            8 + WALK_STEPS + SWING_STEPS + 2 * DWELL_STEPS,
            "{label}: the walk did not run its whole script"
        );
    }

    // ── and the two hosts agree, about the targets and about every step ──
    assert_eq!(a.door, b.door, "the two hosts found different doors");
    assert_eq!(
        a.prompt.x.to_bits(),
        b.prompt.x.to_bits(),
        "the two hosts put the same door in different places"
    );
    assert_eq!(a.upstairs.y.to_bits(), b.upstairs.y.to_bits());
    assert_eq!(a.doorways, b.doorways);
    assert_eq!(a.solids, b.solids);
    let mut distinct: std::collections::BTreeSet<&Vec<u8>> = Default::default();
    for (i, (x, y)) in a.states.iter().zip(&b.states).enumerate() {
        assert_eq!(
            x,
            y,
            "the shipping player and the editor's document diverged at step {i} \
             of the walk ({} against {} bytes)",
            x.len(),
            y.len()
        );
        distinct.insert(x);
    }
    println!(
        "SETTLEMENT GATE: {} steps, {} DISTINCT states, byte-identical between \
         the cooked pack and the loose document",
        a.states.len(),
        distinct.len()
    );
    // Anti-vacuity on the fold itself: a walk that never changed the world would
    // produce one state repeated, and comparing it to itself proves nothing.
    assert!(
        distinct.len() > a.states.len() / 2,
        "only {} of {} states are distinct — the walk is not moving the world",
        distinct.len(),
        a.states.len()
    );
}

/// One furnish configuration's price for one block, in counts a machine cannot
/// inflate.
#[derive(Debug, Clone, Copy, Default)]
struct BlockPrice {
    buildings: usize,
    solids: usize,
    instances: usize,
    doorways: usize,
    /// Solids the collider band would admit with an anchor at the block's own
    /// centre — the number a fixed step pays for.
    banded: usize,
}

/// Evaluate one settlement block with `furnish` forced, and price it.
///
/// The ground is held **flat at the site's own datum**, and that is the honest
/// choice for a price rather than a shortcut: a site pad is levelled toward its
/// datum, so a block near a settlement's centre sits on ground that is nearly
/// flat, and holding it exactly flat makes the comparison between three
/// configurations have one variable. What it is NOT is a claim about the
/// island's own relief.
fn price_block(
    passes: &[inf_pcg::BuildingPass],
    furnish: Option<bool>,
    centre: glam::DVec3,
    extent: glam::DVec2,
    seed: u32,
) -> BlockPrice {
    let passes: Vec<inf_pcg::BuildingPass> = passes
        .iter()
        .cloned()
        .map(|mut p| {
            if let Some(f) = furnish {
                p.furnish = f;
            }
            p
        })
        .collect();
    let cx = inf_pcg::GrammarContext {
        entity: None,
        center: centre,
        extent,
        seed_offset: u64::from(seed),
    };
    let height = inf_pcg::FnHeight::new(move |_, _| Some(centre.y));
    let out = inf_pcg::evaluate_buildings(&passes, &inf_pcg::NoSplines, &height, &cx);
    let band = inf_ecs::band::SimBand::from_anchors(
        [centre],
        BAND_NEAR_M,
        inf_ecs::band::DEFAULT_COLLIDER_FAR_M,
    );
    BlockPrice {
        buildings: out.groups.len(),
        solids: out.colliders.len(),
        instances: out.instances.len(),
        doorways: out.doorways.len(),
        banded: out
            .colliders
            .iter()
            .filter(|c| band.tier(c.center, c.half_extents, c.rotation) == inf_math::Tier::Near)
            .count(),
    }
}

/// **THE FURNISH BATTERY** (island wave I8a, clause 3 / ruling 3).
///
/// The ruling was *measure, then decide, default ON*. This is the measurement,
/// and it has three legs because "furnish=true holds" is three different claims:
///
/// 1. **What a city block costs, three ways.** One real **Harbour City** core
///    block — a 100 m block on the shipped island's own grid, not the fixture's
///    76 m one, because a battery about a city block has to be about one —
///    evaluated with furnish off, with furnish as shipped, and with furnish
///    forced on. Counts, which are the same integer on every machine.
/// 2. **What the fixed step pays.** The fixture's own settled world, stepped
///    with the shipped population and again with every resident volume's
///    population replaced by the fully-furnished one, against
///    [`CITY_STEP_BUDGET_MS`]. Same world, same anchors, one variable.
/// 3. **What a load pays**, against [`LOAD_BUDGET_MS`].
///
/// The verdict it produced is `inf_editor_core::settlement::furnishes`, and it
/// is stated in the wave's ledger with these numbers beside it.
#[test]
fn the_furnish_battery_prices_a_city_block_at_island_scale() {
    let tmp = tempfile::tempdir().expect("a temp dir");
    let started = std::time::Instant::now();
    let proj = build_project(tmp.path());
    let content = proj.join("Content");
    let pack = cook(tmp.path());
    let build_ms = started.elapsed().as_secs_f64() * 1000.0;

    // ── 1. one REAL Harbour City block, three ways ──
    let shipped_recipe =
        inf_editor_core::island::repo_root().join(inf_editor_core::island::ISLAND_RECIPES[0]);
    if shipped_recipe.exists() {
        let recipe =
            inf_island::IslandRecipe::load(&shipped_recipe).expect("the island recipe loads");
        let design = inf_island::read_design(&recipe).expect("the island design reads");
        let city = inf_editor_core::settlement::settlements(&design)
            .into_iter()
            .find(|s| s.kind == inf_island::SiteKind::City)
            .expect("the island has a city");
        // The core block's own geometry, priced for **every archetype** rather
        // than for the one that happens to be zoned there. That is what makes
        // this a battery: `Shop` is one to two storeys and `Hotel` is four to
        // ten, furniture is per ROOM, and a measurement of the cheap one would
        // have decided the ruling on the wrong building.
        let block = city
            .blocks
            .iter()
            .find(|b| b.ring == 0)
            .copied()
            .expect("a city has a ring-0 block");
        let centre = glam::DVec3::new(block.centre.x, 0.0, block.centre.y);
        let extent = glam::DVec2::new(block.half.x, block.half.y);
        println!(
            "FURNISH BATTERY on a {:.0} x {:.0} m {} block at ({:.0}, {:.0}), \
             every archetype:",
            block.half.x * 2.0,
            block.half.y * 2.0,
            city.name,
            block.centre.x,
            block.centre.y
        );
        let mut worst = 1.0f64;
        let mut worst_name = "";
        for a in inf_pcg::ArchetypeId::ALL {
            let passes = zone_passes(&content, a);
            let off = price_block(&passes, Some(false), centre, extent, block.seed);
            let on = price_block(&passes, Some(true), centre, extent, block.seed);
            let ratio = on.solids as f64 / off.solids.max(1) as f64;
            println!(
                "  {:>10} ({}-{} storeys): {} buildings, {} doorways; bare {} \
                 solids ({} banded, {} drawn instances) -> furnished {} solids \
                 ({} banded, {} drawn), {ratio:.2}x{}",
                a.name(),
                inf_pcg::archetype(a).floors.0,
                inf_pcg::archetype(a).floors.1,
                off.buildings,
                off.doorways,
                off.solids,
                off.banded,
                off.instances,
                on.solids,
                on.banded,
                on.instances,
                if inf_editor_core::settlement::furnishes(a) {
                    "  <- SHIPS FURNISHED"
                } else {
                    ""
                }
            );
            assert_eq!(
                off.buildings,
                on.buildings,
                "{}: furnishing changed how many BUILDINGS a block stands, which \
                 it must not — furniture is what goes inside them",
                a.name()
            );
            assert_eq!(
                off.doorways,
                on.doorways,
                "{}: furnishing changed the doorway count",
                a.name()
            );
            assert!(
                on.solids > off.solids,
                "{}: furnishing added no solid at all — the battery is measuring \
                 nothing",
                a.name()
            );
            if ratio > worst {
                worst = ratio;
                worst_name = a.name();
            }
        }
        println!("  WORST: {worst_name} at {worst:.2}x the solids of a bare block");
    } else {
        println!("SKIP the shipped-island half: no committed island recipe");
    }

    // ── 2. what the fixed step pays, on a settled world ──
    let recipe =
        inf_island::IslandRecipe::load(&fixture_recipe()).expect("the fixture recipe loads");
    let design = inf_island::read_design(&recipe).expect("the design reads");
    let plan = walk_target_settlement(&design);
    let mut sim = pack_sim(&pack);
    let hero = hero_entity(&sim).expect("a hero");
    set_hero(
        &mut sim,
        hero,
        glam::DVec3::new(plan.centre.x, 0.0, plan.centre.y),
    );
    for _ in 0..12 {
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
    }
    let shipped_solids = resident_solids(&sim).len();
    println!("FIXED STEP at {} (release asserts nothing here — this REPORTS, on this module's own law that a millisecond is a fact about the machine):", plan.name);
    // **THE SAME CONFIGURATION, TWICE, BEFORE ANYTHING IS CHANGED** (island wave
    // I8a audit). This arm's `A` and `B` are separated by one variable and by a
    // stretch of wall clock, and the wall clock is running **inside a test binary
    // whose other eleven arms are executing on other threads** — `cargo test`
    // runs a file's tests concurrently, and nothing here asks it not to.
    //
    // Measured: this same comparison read **+0.663 ms** in the wave's own run,
    // **+0.821 / +0.915 / +0.805 ms** in three runs of this arm ALONE, and
    // **−0.690 ms** — the opposite sign — in a run of the whole file. The step
    // itself read 6.348, 8.1–8.2 and 10.693 ms in those three regimes over an
    // identical world of 21 453 solids.
    //
    // So the first measurement is repeated with nothing changed between them,
    // and `|A' − A|` is printed beside `B − A'` as this run's own noise floor. A
    // difference inside the floor is not a measurement, and a reader who is
    // handed one number cannot tell.
    let (shipped_ms, shipped_prof) = step_profile_of(&mut sim, 60, 90);
    print_step(
        "as shipped",
        shipped_ms,
        &shipped_prof,
        shipped_solids,
        &sim,
    );
    let (control_ms, control_prof) = step_profile_of(&mut sim, 10, 90);
    print_step(
        "as shipped (again)",
        control_ms,
        &control_prof,
        shipped_solids,
        &sim,
    );
    let floor = (control_ms - shipped_ms).abs();
    println!("  NOISE FLOOR: the same configuration twice differs by {floor:.3} ms");

    // …and again with every resident block's population replaced by the fully
    // furnished one, through the same door the host writes it with.
    let by_guid: std::collections::BTreeMap<Uuid, inf_editor_core::settlement::Block> = plan
        .blocks
        .iter()
        .map(|b| {
            (
                inf_editor_core::settlement::block_guid(&recipe.name, b.site, b.col, b.row),
                *b,
            )
        })
        .collect();
    let mut replaced = 0usize;
    for (guid, centre, extent, seed) in resident_volumes(&sim) {
        let Some(block) = by_guid.get(&guid) else {
            continue;
        };
        let passes: Vec<inf_pcg::BuildingPass> = zone_passes(&content, block.archetype)
            .into_iter()
            .map(|mut p| {
                p.furnish = true;
                p
            })
            .collect();
        let out = {
            let w = sim.world().world();
            let terrain = w
                .iter_entities()
                .find_map(|e| e.get::<inf_ecs::components::Terrain>())
                .expect("the island has ground");
            let height = inf_pcg::FnHeight::new(|x: f64, z: f64| {
                terrain.data.height_at(glam::DVec2::new(x, z))
            });
            let cx = inf_pcg::GrammarContext {
                entity: Some(guid),
                center: centre,
                extent,
                seed_offset: u64::from(seed),
            };
            inf_pcg::compose_volume(
                Vec::new(),
                inf_pcg::evaluate_buildings(&passes, &inf_pcg::NoSplines, &height, &cx),
            )
        };
        let (baked, solid, groups, doorways, residents, interior, lights, emitters) =
            inf_player::level::population_of(out);
        let e = sim
            .world()
            .entity_of(guid)
            .expect("the volume the walk found is in the world");
        if let Some(mut v) = sim
            .world_mut()
            .world_mut()
            .get_mut::<inf_ecs::components::PcgVolume>(e)
        {
            v.set_population(
                baked, solid, groups, doorways, residents, interior, lights, emitters,
            );
            replaced += 1;
        }
    }
    // Two steps for the physics bridge to reconcile the new change stamp before
    // the clock starts — the cost being measured is the STEADY step, not the
    // rebuild the swap itself forces.
    for _ in 0..2 {
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
    }
    let furnished_solids = resident_solids(&sim).len();
    let (furnished_ms, furnished_prof) = step_profile_of(&mut sim, 60, 90);
    print_step(
        "fully furnished",
        furnished_ms,
        &furnished_prof,
        furnished_solids,
        &sim,
    );
    let cost = furnished_ms - control_ms;
    println!(
        "  {replaced} volume(s) swapped; furnishing moves the step {cost:+.3} ms \
         against a {floor:.3} ms noise floor — {}. **The COUNT is the half a \
         machine cannot move**: {shipped_solids} -> {furnished_solids} solids \
         ({:.2}x), and that is what the `furnishes` verdict rests on",
        if cost.abs() > floor * 2.0 {
            "a difference"
        } else {
            "INSIDE the floor, i.e. no measurement at all"
        },
        furnished_solids as f64 / shipped_solids.max(1) as f64
    );
    println!("LOAD: build + cook of the whole fixture project took {build_ms:.0} ms against a {LOAD_BUDGET_MS} ms load budget");
    assert!(
        furnished_solids > shipped_solids,
        "the swap changed nothing — the step comparison is between one \
         configuration and itself"
    );
    // **Reported, not asserted, and the reason is this module's own law**: a
    // millisecond is a fact about the machine, `[profile.dev]` is `opt-level = 1`
    // with debug assertions, and every CI runner reports rather than asserts on
    // a wall clock. What IS asserted is the solid count, which is the same
    // integer everywhere.
    assert!(
        shipped_ms.is_finite() && control_ms.is_finite() && furnished_ms.is_finite(),
        "the step clock produced no number"
    );
}

/// The mean fixed-step time over `n` steps, milliseconds, **with the step's own
/// phase breakdown beside it**.
///
/// A step that cannot say where its milliseconds went is the CPU twin of the
/// frame that could not say where its GPU milliseconds went — wave I4b's own
/// finding, and the reason `RuntimeSim` carries a step clock at all. The first
/// draft of this battery printed one number and a reader could not tell a
/// physics regression from a paging one.
///
/// A discarded warm-up pass first: the first steps after a population swap seat
/// the collider band and take every `structure_stamps` miss there is, and
/// measuring them is measuring a step that happens once.
fn step_profile_of(
    sim: &mut RuntimeSim,
    warmup: usize,
    n: usize,
) -> (f64, inf_player::step_profile::StepProfile) {
    for _ in 0..warmup {
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
    }
    sim.set_step_profiling(true);
    let mut acc = inf_player::step_profile::StepProfile::default();
    let t = std::time::Instant::now();
    for _ in 0..n {
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
        acc.accumulate(&sim.step_profile());
    }
    let wall = t.elapsed().as_secs_f64() * 1000.0 / n as f64;
    acc.scale(1.0 / n as f64);
    sim.set_step_profiling(false);
    (wall, acc)
}

/// Print one step profile, dearest phase first, **with what the physics world
/// actually admitted beside it**.
///
/// A step whose dearest phase is the solver over a world with one moving thing
/// in it is a step paying for its own STATIC geometry, and the admitted-collider
/// count is the evidence — the fps instrument's own arrangement, met on a
/// settlement.
fn print_step(
    label: &str,
    wall: f64,
    prof: &inf_player::step_profile::StepProfile,
    solids: usize,
    sim: &RuntimeSim,
) {
    let (tracked, touching) = sim.bridge3d().world().contact_pair_counts();
    println!(
        "  {label:<18} {wall:7.3} ms/step over {solids} resident solids (phases \
         sum to {:.3} ms; the ratchet is {CITY_STEP_BUDGET_MS} ms). Physics: {} \
         bodies, {} ADMITTED structure colliders, {tracked} contact pairs \
         tracked ({touching} touching)",
        prof.total_ms(),
        sim.bridge3d().body_count(),
        sim.bridge3d().admitted_structures(),
    );
    for (n, ms) in prof.dearest_first() {
        if ms <= 0.02 {
            continue;
        }
        println!(
            "      {n:<18} {ms:7.3} ms ({:4.1} %)",
            ms / prof.total_ms().max(1.0e-9) * 100.0
        );
    }
}

/// **THE `is_open` WALK, RE-MEASURED AT SETTLEMENT SCALE** (island wave I8a,
/// clause 3).
///
/// The I6 audit found `door.is_open` walking **all 19 790 doorways** of the
/// composed city to answer a question about one, and fixed it by checking the
/// reach *before* building a placement — so the walk is still `O(doorways)` and
/// the allocation is not. Wave I8a is the first content in this repository that
/// puts real doorways on a streamed world, so the cost class is re-measured
/// here rather than assumed.
///
/// **The number that changed is not the constant, it is the N.** The walk is
/// over the doorways the SIMULATION holds, and a streamed island holds one
/// neighbourhood: the whole island plans two orders of magnitude more doorways
/// than any step ever walks.
#[test]
fn the_is_open_walk_costs_what_it_costs_at_settlement_scale() {
    let tmp = tempfile::tempdir().expect("a temp dir");
    build_project(tmp.path());
    let pack = cook(tmp.path());
    let recipe =
        inf_island::IslandRecipe::load(&fixture_recipe()).expect("the fixture recipe loads");
    let design = inf_island::read_design(&recipe).expect("the design reads");
    let plan = walk_target_settlement(&design);
    let mut sim = pack_sim(&pack);
    let hero = hero_entity(&sim).expect("a hero");
    set_hero(
        &mut sim,
        hero,
        glam::DVec3::new(plan.centre.x, 0.0, plan.centre.y),
    );
    for _ in 0..12 {
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
    }
    let resident = inf_ecs::door::volume_doorways(sim.world()).len();
    let centre = glam::DVec3::new(plan.centre.x, 0.0, plan.centre.y);
    // Two questions, and the difference between them is the whole point: one
    // asked where there IS a door, one asked out at sea.
    let near = {
        let d = inf_ecs::door::volume_doorways(sim.world());
        d.iter()
            .map(|(_, _, s)| s.hinge)
            .min_by(|a, b| {
                (*a - centre)
                    .length_squared()
                    .total_cmp(&(*b - centre).length_squared())
            })
            .expect("a doorway")
    };
    let far = glam::DVec3::new(centre.x + 5_000.0, centre.y, centre.z);
    const CALLS: usize = 200;
    let mut answered = 0usize;
    let t = std::time::Instant::now();
    for _ in 0..CALLS {
        if inf_physics::d3::door::is_open_near(sim.world(), near) {
            answered += 1;
        }
    }
    let near_us = t.elapsed().as_secs_f64() * 1.0e6 / CALLS as f64;
    let t = std::time::Instant::now();
    for _ in 0..CALLS {
        if inf_physics::d3::door::is_open_near(sim.world(), far) {
            answered += 1;
        }
    }
    let far_us = t.elapsed().as_secs_f64() * 1.0e6 / CALLS as f64;
    // **What the walk's N actually is.** The resident set, against the blocks
    // this island holds in all: the ratio is the whole point of the class, and
    // the second number is a block COUNT rather than a doorway count because
    // nothing evaluates the far blocks and inventing a doorway figure for them
    // would be inference dressed as measurement.
    let resident_blocks = resident_volumes(&sim).len();
    let island_blocks: usize = inf_editor_core::settlement::settlements(&design)
        .iter()
        .map(|s| s.blocks.len())
        .sum();
    println!(
        "IS_OPEN WALK: {resident} doorways over {resident_blocks} resident blocks \
         at {}, {near_us:.1} us a call beside a door and {far_us:.1} us a call \
         five kilometres from one ({answered} of {} answered open). This island \
         has {island_blocks} blocks in all and a step walks the resident ones \
         only — the walk is O(RESIDENT doorways), which is what streaming buys \
         and what the I6 measurement (19 790 doorways on an unstreamed city) did \
         not have.",
        plan.name,
        2 * CALLS
    );
    assert!(resident > 100, "only {resident} doorways were resident");
    // The class: the far call does the same walk and allocates nothing, so it
    // must not be dramatically dearer than the near one. A regression that
    // rebuilt every placement would make the far call the expensive one.
    assert!(
        far_us <= near_us * 4.0 + 50.0,
        "a call with NO door in reach costs {far_us:.1} us against {near_us:.1} \
         beside one — the reach check has stopped happening before the \
         allocation (the I6 audit's finding, returned)"
    );
}

// -- NPC1c: an NPC walks across town -----------------------------------------

/// How far a planned town walk may run, metres. A cap and not a target: the
/// walk is driven one fixed step at a time on **two** hosts, so every metre is
/// about 36 steps of a furnished city and the gate has to finish.
const TOWN_WALK_MAX_M: f64 = 140.0;

/// The most fixed steps the walk is given before it is called stuck.
const TOWN_WALK_STEPS: usize = 9_000;

/// How far the body must get along its route to count as still walking, metres.
const STALL_M: f64 = 0.25;

/// How long it may go without making that much progress before the walk is
/// called stalled.
const STALL_STEPS: usize = 600;

/// A planned walk: the network it was routed over, the path, and the two things
/// the walk has to *do* on the way.
struct TownPlan {
    graph: inf_nav::NavGraph,
    nodes: Vec<inf_nav::NavNodeId>,
    path: inf_nav::NavPath,
    kinds: Vec<inf_nav::NavKind>,
    /// The doorway entity the route passes through -- the one the NPC opens.
    door: Uuid,
    /// The first-floor room the route ends in, world metres at its own floor.
    upstairs: glam::DVec3,
    /// Ground-floor walking height of the same building, so "it climbed" is a
    /// difference and not an absolute.
    ground_y: f64,
    cost_m: f64,
    start: glam::DVec3,
    /// Arc length at which the route crosses the entrance threshold.
    entrance_s: f64,
    /// The flight, measured: `(run axis length, treads, rise, run, landing)`.
    flight: (f64, u32, f64, f64, f64),
}

/// **Plan the walk out of the graphs the island already publishes** (NPC1c).
///
/// Three producers, one network, and nothing invented here:
///
/// * the settlement's own `street_graph`, **grounded** on the simulation's own
///   terrain (its nodes are planar -- `settlement.rs` may not name `inf_terrain`
///   under its allowlist gate, so grounding is the caller's, once);
/// * the target building's `interior_nav`, whose rooms already carry `floor_y`;
/// * one link between them, from the entrance doorway to the nearest street
///   node. A lot's frontage is the one join the settlement does not model, and
///   making it here is honest about that rather than hiding it in a tolerance.
///
/// The target building is found the way `walk_into_a_building` finds it -- the
/// exterior ground-floor doorway nearest the crossroads on a block whose
/// archetype is *guaranteed* multi-storey -- so the two walks in this file aim
/// at the same door and a divergence between them is about the walk and not
/// about the target.
fn plan_town_walk(
    sim: &RuntimeSim,
    content: &Path,
    recipe: &inf_island::IslandRecipe,
    plan: &inf_editor_core::settlement::Settlement,
) -> TownPlan {
    // 1. THE STREETS, on the ground the simulation is standing on.
    let mut graph = plan.street_graph();
    {
        let w = sim.world().world();
        let terrain = w
            .iter_entities()
            .find_map(|e| e.get::<inf_ecs::components::Terrain>())
            .expect("the island has ground");
        let planar: Vec<(inf_nav::NavNodeId, glam::DVec3, inf_nav::NavKind)> =
            graph.nodes().map(|n| (n.id, n.position, n.kind)).collect();
        let mut grounded = 0usize;
        for (id, p, kind) in planar {
            if let Some(y) = terrain.data.height_at(glam::DVec2::new(p.x, p.z)) {
                graph.add_node(id, glam::DVec3::new(p.x, y, p.z), kind);
                grounded += 1;
            }
        }
        assert!(
            grounded > 0,
            "no street node of {} found ground under it, so the walk would be \
             planned in the air",
            plan.name
        );
    }

    // 2. THE DOOR — through `walk_door`, the ONE place the rule lives. This
    //    used to be a copy of it under a comment saying so, which is the A14
    //    restated-rule defect and went stale the moment the rule grew a clause.
    let (vol, idx, slot) = walk_door(sim, content, recipe, plan);
    let door = inf_physics::d3::door::pcg_doorway_guid(vol, idx);

    // 3. THE BUILDING, re-derived through the shipped host's own resolution and
    //    matched to the doorway BIT FOR BIT -- not by a search radius.
    let block = plan
        .blocks
        .iter()
        .find(|b| {
            inf_editor_core::settlement::block_guid(&recipe.name, b.site, b.col, b.row) == vol
        })
        .copied()
        .expect("the doorway's volume is a settlement block");
    let passes = zone_passes(content, block.archetype);
    let volumes = resident_volumes(sim);
    let (_, vcentre, vextent, vseed) = volumes
        .iter()
        .find(|(g, _, _, _)| *g == vol)
        .copied()
        .expect("the volume is resident");
    let building = {
        let w = sim.world().world();
        let terrain = w
            .iter_entities()
            .find_map(|e| e.get::<inf_ecs::components::Terrain>())
            .expect("the island has ground");
        let height =
            inf_pcg::FnHeight::new(|x: f64, z: f64| terrain.data.height_at(glam::DVec2::new(x, z)));
        let cx = inf_pcg::GrammarContext {
            entity: Some(vol),
            center: vcentre,
            extent: vextent,
            seed_offset: u64::from(vseed),
        };
        let plans = inf_pcg::plans_of(&passes, &inf_pcg::NoSplines, &height, &cx);
        let mut found = None;
        for p in &plans {
            let mut ds = inf_pcg::building::doorways_of(p);
            inf_pcg::building::place_doorways_in_frame(&mut ds, p.frame);
            if ds.iter().any(|d| {
                d.hinge.x.to_bits() == slot.hinge.x.to_bits()
                    && d.hinge.y.to_bits() == slot.hinge.y.to_bits()
                    && d.hinge.z.to_bits() == slot.hinge.z.to_bits()
            }) {
                found = Some(p.clone());
                break;
            }
        }
        found.expect(
            "the doorway the world offered is not in any plan the same resolution \
             derives -- the shipped population and `plans_of` disagree",
        )
    };
    assert!(
        building.floors >= 2 && building.core.is_some(),
        "the target building has no stair to climb"
    );

    // 4. THE INTERIOR, welded to the street through its own entrance.
    graph.absorb(&building.interior_nav());
    let entrance_wall = building.entrance.expect("a plan with a doorway has one");
    let (entrance_opening, _) = building
        .openings
        .iter()
        .enumerate()
        .find(|(_, o)| o.kind == inf_pcg::OpeningKind::Door && o.wall == entrance_wall)
        .expect("the entrance wall carries a door opening");
    let entrance_node = inf_pcg::building::doorway_node_id(entrance_opening);
    assert!(
        graph.node(entrance_node).is_some(),
        "the entrance opening is not a node of the building's own nav graph"
    );
    let entrance_p = graph.node(entrance_node).expect("checked").position;
    // **THE FRONTAGE**, and it is a node this gate MINTS rather than one the
    // settlement publishes. A city's street graph has a node only where two
    // centrelines cross -- 120 m apart on the island's own pitch -- so the
    // nearest published node to a front door is tens of metres away and a
    // straight link to it cuts diagonally through the block. Measured: the
    // first draft joined at **32.4 m** and the NPC walked 6.5 m into the side of
    // a building and stood there for 4 000 steps.
    //
    // So the entrance is dropped onto the nearest street EDGE at its own foot of
    // perpendicular, the edge is split there, and the frontage link is the short
    // hop across the pavement. That a lot has a frontage at all is the one join
    // the settlement generator does not model, and minting it here says so
    // rather than hiding it in a tolerance.
    let frontage = inf_nav::domain::CALLER | 1;
    let (frontage_p, split) = {
        let mut best: Option<(f64, glam::DVec3, inf_nav::NavNodeId, inf_nav::NavNodeId)> = None;
        let street: Vec<(inf_nav::NavNodeId, glam::DVec3)> = graph
            .nodes()
            .filter(|n| inf_nav::domain::of(n.id) == inf_nav::domain::STREET)
            .map(|n| (n.id, n.position))
            .collect();
        for (a, pa) in &street {
            for e in graph.edges_from(*a) {
                if e.to <= *a {
                    continue;
                }
                let Some(pb) = graph.node(e.to).map(|n| n.position) else {
                    continue;
                };
                let d = pb - *pa;
                let len2 = d.x * d.x + d.z * d.z;
                if len2 <= 0.0 {
                    continue;
                }
                let t = (((entrance_p.x - pa.x) * d.x + (entrance_p.z - pa.z) * d.z) / len2)
                    .clamp(0.0, 1.0);
                let on = *pa + d * t;
                let dist = glam::DVec2::new(entrance_p.x - on.x, entrance_p.z - on.z).length();
                if best.map(|(bd, _, _, _)| dist < bd).unwrap_or(true) {
                    best = Some((dist, on, *a, e.to));
                }
            }
        }
        let (dist, on, a, b) = best.expect("the settlement has a street edge to join");
        (on, (dist, a, b))
    };
    graph.add_node(frontage, frontage_p, inf_nav::NavKind::Street);
    graph.link(frontage, split.1, inf_nav::NavKind::Street, vec![]);
    graph.link(frontage, split.2, inf_nav::NavKind::Street, vec![]);
    graph.link(frontage, entrance_node, inf_nav::NavKind::Street, vec![]);

    // 5. THE DESTINATION: the first non-stair room on floor one that the plan's
    //    own reachability says you can get to.
    //
    // **The FIRST FLOOR'S LANDING**, which is the stair room on floor one, and
    // not a room beyond it. The claim this gate makes is "it climbed a flight of
    // stairs", and the landing is where that finishes; carrying on into a
    // bedroom adds two more doorways and a room CENTRE, and a room centre is
    // where the furnish pass puts the furniture. `interior_nav` routes through
    // rooms and knows nothing about what is standing in them -- a real
    // limitation, named in this wave's carried list rather than papered over by
    // a destination nobody can reach.
    let reach = building.reachable_rooms();
    let dest_room = building
        .stair_room(1)
        .expect("a multi-storey plan has a stair room on floor one");
    assert!(
        reach.get(dest_room).copied().unwrap_or(false),
        "the stair room on floor one is not reachable from outside"
    );
    let room = &building.rooms[dest_room];
    let dest = inf_pcg::building::room_node_id(dest_room);
    let rc = building.frame.to_world(room.rect.center());
    let upstairs = glam::DVec3::new(rc.x, building.floor_y(1), rc.y);

    // 6. THE START: the street node whose route to the door is longest inside
    //    the cap -- "across town", bounded so the gate finishes. Deterministic:
    //    `nodes()` walks in id order and ties keep the first.
    //
    // **ON THE PAD**, and the constraint is a measurement rather than caution. A
    // settlement's street grid runs out to its whole reservation radius, and the
    // *levelled* pad is smaller than that -- so the outer lines lie on raw
    // hillside, and a route that starts on one walks a body up the natural slope
    // into the cut face at the pad's edge. Measured: an agent started on a line
    // grounded at 122.0 m against a ground floor at 128.1 m walked to within
    // 0.7 m of its next waypoint, stopped 3.84 m off the spine at **119.3 m**,
    // and stood there for 4 400 steps. The mover was right; the route was
    // walking it at a wall.
    //
    // So a candidate start is a street node standing on the same storey as the
    // building it is walking to. That leaves a city's whole grid inside the pad,
    // which is what "across town" means anyway.
    let pad_y = building.floor_y(0);
    let street_ids: Vec<inf_nav::NavNodeId> = graph
        .nodes()
        .filter(|n| inf_nav::domain::of(n.id) == inf_nav::domain::STREET)
        .filter(|n| (n.position.y - pad_y).abs() <= 2.0)
        .map(|n| n.id)
        .collect();
    assert!(
        !street_ids.is_empty(),
        "no street line of {} is on the same storey as its own buildings",
        plan.name
    );

    // The ground sampler, taken once. Everything below reads the SIMULATION's
    // own terrain, so a route is planned over the metres the body will meet.
    let height = |xz: glam::DVec2| -> Option<f64> {
        let w = sim.world().world();
        w.iter_entities()
            .find_map(|e| e.get::<inf_ecs::components::Terrain>())
            .and_then(|t| t.data.height_at(xz))
    };

    let mut best: Option<(
        f64,
        inf_nav::NavNodeId,
        Vec<inf_nav::NavNodeId>,
        inf_nav::NavPath,
    )> = None;
    let mut refused_step = 0usize;
    let mut refused_cost = 0usize;
    for id in street_ids {
        let inf_nav::NavVerdict::Found(r) = inf_nav::route(&graph, id, dest) else {
            continue;
        };
        if r.cost_m > TOWN_WALK_MAX_M {
            refused_cost += 1;
            continue;
        }
        let Some((path, rise)) = grounded_route(&graph, &r.nodes, entrance_node, &height) else {
            continue;
        };
        // **A route the mover cannot walk is not a route** (NPC1c). The street
        // half is re-cut every `TOWN_SAMPLE_M` and put on the simulation's own
        // ground, and the biggest rise between two consecutive samples is held
        // against the character's OWN autostep height. That is not caution: a
        // settlement's grid runs to its whole reservation radius while the pad
        // it levels is smaller, so the outer lines lie on raw hillside and end
        // at the cut face. Measured before this filter existed: an agent walked
        // to within 0.7 m of its next waypoint, stopped 3.84 m off the spine
        // 8.8 m BELOW it, and stood there for 4 400 steps. The mover was right
        // and the route was walking it at a wall.
        if rise > inf_ecs::components::CharacterMovement::default().step_height_m {
            refused_step += 1;
            continue;
        }
        if best
            .as_ref()
            .map(|(c, _, _, _)| r.cost_m > *c)
            .unwrap_or(true)
        {
            best = Some((r.cost_m, id, r.nodes.clone(), path));
        }
    }
    let (cost_m, _from, nodes, path) = best.expect(
        "no street node routes to the first-floor room over ground the mover can \\
         walk -- the street graph and the building's interior are one network on \\
         paper and not on the ground",
    );

    let kinds = inf_nav::route::kinds_of(&graph, &nodes);
    let start = path.points()[0];
    println!(
        "NPC1c town walk: {} nodes, {:.1} m of route, {} spine points, kinds {:?}",
        nodes.len(),
        cost_m,
        path.points().len(),
        kinds
    );
    println!(
        "NPC1c town walk: frontage minted {:.1} m from the entrance",
        split.0
    );
    println!("NPC1c town walk: {refused_step} start(s) refused for an unclimbable step");
    println!("NPC1c town walk: {refused_cost} refused for the {TOWN_WALK_MAX_M:.0} m cap");
    // **What the flight IS**, in the numbers the character mover is judged by.
    // Re-derived here with the assembler's own rule (`stairs()` in
    // `inf_pcg::building::assemble`) rather than read off the solids, so the
    // print names the generator's intent and not one box of it.
    let flight = {
        let core = building.core.expect("checked above");
        let arch = inf_pcg::archetype(block.archetype);
        let inset = arch
            .wall_thickness
            .min(core.size_x() * 0.2)
            .min(core.size_z() * 0.2);
        let rect = core.inset(inset);
        let n = ((arch.floor_height / 0.18).round() as u32).clamp(2, 80);
        let run_len = rect.size_x().max(rect.size_z());
        (
            run_len,
            n,
            arch.floor_height / n as f64,
            run_len / n as f64,
            inset,
        )
    };
    let entrance_s = {
        let ep = graph.node(entrance_node).expect("checked").position;
        path.project(ep).s_m
    };
    println!(
        "NPC1c stair: run {:.2} m, {} treads, rise {:.3} m, run {:.3} m, landing {:.2} m",
        flight.0, flight.1, flight.2, flight.3, flight.4
    );
    TownPlan {
        nodes,
        path,
        kinds,
        door,
        upstairs,
        ground_y: pad_y,
        cost_m,
        start,
        graph,
        entrance_s,
        flight,
    }
}

/// How finely the street half of a route is re-cut before it is put on the
/// ground, metres.
///
/// Two metres is a stride and a bit: fine enough that the rise between two
/// samples is a *step* rather than a hill, coarse enough that a hundred-metre
/// walk is fifty points.
const TOWN_SAMPLE_M: f64 = 2.0;

/// **The planned route, with its outdoor half put on the ground** -- and the
/// biggest rise between two consecutive samples of that half.
///
/// The split is at the entrance: everything before it is street and belongs on
/// the terrain, everything after it is interior and already carries its own
/// `floor_y`. Snapping the whole chain would put the first floor in the garden.
///
/// `None` when the route does not pass through the entrance at all, which is a
/// route this gate is not about.
fn grounded_route(
    graph: &inf_nav::NavGraph,
    nodes: &[inf_nav::NavNodeId],
    entrance: inf_nav::NavNodeId,
    height: &impl Fn(glam::DVec2) -> Option<f64>,
) -> Option<(inf_nav::NavPath, f64)> {
    let at = nodes.iter().position(|n| *n == entrance)?;
    let outside = inf_nav::route::chain(graph, &nodes[..=at])
        .resampled(TOWN_SAMPLE_M)
        .snapped(0.0, height);
    let inside = inf_nav::route::chain(graph, &nodes[at..]);
    let mut rise: f64 = 0.0;
    for w in outside.points().windows(2) {
        rise = rise.max((w[1].y - w[0].y).abs());
    }
    let joined: Vec<glam::DVec3> = outside
        .points()
        .iter()
        .chain(inside.points().iter().skip(1))
        .copied()
        .collect();
    Some((inf_nav::NavPath::new(joined), rise))
}

/// What one host's NPC did.
struct TownWalk {
    /// One digest per step, in order -- the two hosts are compared on these.
    digests: Vec<u64>,
    distinct: usize,
    steps: usize,
    arrived: bool,
    /// Where the agent finished.
    end: glam::DVec3,
    /// The highest walking surface the agent's feet reached.
    peak_y: f64,
    /// Steps on which the crowd step wrote a steering intent.
    steered: u64,
    /// Doors the crowd pass pressed / opened over the whole walk.
    pressed: usize,
    opened: usize,
    considered: usize,
    /// Steps the agent spent behind its own clock.
    blocked: u64,
    tiers: [usize; 4],
    /// The furthest the body ever got along its own route, metres.
    reached_m: f64,
}

const TOWN_AGENT: u128 = 0x4E50_4331_0000_0001;

/// **Walk it.** One agent, one route, driven a fixed step at a time, with the
/// streaming source kept beside it so the tier stays one that has a body.
///
/// The hero follows the AGENT rather than a precomputed line, so the drive is a
/// pure function of each host's own state: a host that put its NPC somewhere
/// else would also stream a different neighbourhood, which compounds the
/// divergence instead of masking it.
fn walk_the_town(sim: &mut RuntimeSim, plan: &TownPlan) -> TownWalk {
    let hero = hero_entity(sim).expect("a hero");
    let archetype = crowd_archetype(sim);
    let walk_speed = inf_ecs::components::CharacterMovement::default().walk_speed_mps;
    let mut records = std::collections::BTreeMap::new();
    records.insert(
        Uuid::from_u128(TOWN_AGENT),
        inf_ecs::crowd::CrowdRecord::walking(
            archetype,
            inf_ecs::crowd::CrowdRoute::along(
                plan.path.clone(),
                walk_speed,
                inf_ecs::crowd::RouteMode::Once,
            ),
        ),
    );
    // The hero stands where the walk starts, so the first step tiers the agent
    // `Full` and materializes it with a body.
    set_hero(sim, hero, plan.start + glam::DVec3::new(0.0, 0.9, 0.0));
    sim.world_mut().mark_dirty();
    sim.set_crowd_population(records);

    let mut out = TownWalk {
        digests: Vec::new(),
        distinct: 0,
        steps: 0,
        arrived: false,
        end: plan.start,
        peak_y: plan.start.y,
        steered: 0,
        pressed: 0,
        opened: 0,
        considered: 0,
        blocked: 0,
        tiers: [0; 4],
        reached_m: 0.0,
    };
    // **The walk ends when it stops walking.** A fixed step budget would either
    // cut a working walk short or spend minutes of CI watching a stalled one, so
    // the loop watches the body's own arc length and stops when it has not moved
    // `STALL_M` in `STALL_STEPS`. That number is then part of what the gate
    // reports rather than a thing the harness hides.
    let mut last_progress = (0usize, 0.0f64);
    let mut seen: std::collections::BTreeSet<u64> = Default::default();
    for _ in 0..TOWN_WALK_STEPS {
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
        out.steps += 1;
        let d = digest(&sim.state_bytes());
        seen.insert(d);
        out.digests.push(d);

        let stats = sim.crowd_stats();
        out.steered += stats.steered;
        out.blocked += stats.blocked;
        for (i, n) in stats.per_tier.iter().enumerate() {
            out.tiers[i] += n;
        }
        out.pressed += sim.gameplay().crowd_doors.pressed;
        out.opened += sim.gameplay().crowd_doors.opened;
        out.considered += sim.gameplay().crowd_doors.considered;

        if let Some(e) = sim.world().entity_of(Uuid::from_u128(TOWN_AGENT)) {
            if let Some(t) = sim.world().world().get::<inf_ecs::components::Transform>(e) {
                let p = t.translation.to_dvec3();
                out.end = p;
                let feet = p.y - archetype.feet_offset_m();
                if feet > out.peak_y {
                    out.peak_y = feet;
                }
            }
            // The hero walks with it, so the agent keeps a tier that has a
            // body for the whole walk.
            //
            // **`mark_dirty` is load-bearing**, and this walk is what found out.
            // `set_hero` writes a `Transform` straight through bevy's own
            // `get_mut`, which does not set `EcsWorld`'s dirty flag -- so
            // `propagate` skips, the `GlobalTransform` the streaming source is
            // read off stays where it last settled, and the whole world streams
            // around a ghost. Measured: after 4 000 steps the hero's local
            // transform read (-450.8, 129.3, 364.4) and its global read
            // (-586.2, 72.1, 393.1), 135 m and a hillside apart, with **0** of
            // the town's 1 297 doorways inside the band.
            set_hero(sim, hero, out.end);
            sim.world_mut().mark_dirty();
        }
        {
            let feet = out.end - glam::DVec3::new(0.0, archetype.feet_offset_m(), 0.0);
            let on = plan.path.project(feet).s_m;
            if on > out.reached_m {
                out.reached_m = on;
            }
            if on > last_progress.1 + STALL_M {
                last_progress = (out.steps, on);
            }
        }
        if stats.arrived > 0 {
            out.arrived = true;
            break;
        }
        if out.steps - last_progress.0 >= STALL_STEPS {
            break;
        }
    }
    out.distinct = seen.len();
    out
}

fn digest(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// **THE NPC1c GATE**: an NPC walks across Harbour City -- down a street,
/// through a doorway it OPENS, up a flight of stairs -- and the shipped player
/// and the editor's document do it byte for byte.
///
/// # What is new here, and what it retires
///
/// Every walk in this file before this one moved its subject by writing a
/// transform: `walk_into_a_building` teleports the hero along a straight line,
/// `set_hero` at a time, and asserts that the *world* would have allowed it (no
/// solid at the point, the doorway clear, the first floor standable). That was
/// the right test of a building and it is not a test of walking. Wave I8a's
/// carried remainder says it plainly -- *nothing has ever walked a character up
/// a flight of stairs* -- and this arm is what retires it: the NPC is moved by
/// `move_and_slide`, through `step_character_movement`, over the same colliders
/// a player meets, and the stair is climbed rather than asserted.
///
/// # Coverage first, because two empty worlds agree perfectly
///
/// Every claim is asserted on each host **separately** before the two are
/// compared: the route was planned over a real network (street, doorway and
/// stair all appear in its `kinds`), the agent really was steered, it really
/// pressed and opened a door, it really ended a storey up, and it really
/// arrived. A gate that compared folds alone would certify two hosts that both
/// stood still.
#[test]
fn pie_equals_shipping_when_an_npc_walks_across_town() {
    let tmp = tempfile::tempdir().expect("a temp dir");
    let proj = build_project(tmp.path());
    let content = proj.join("Content");
    let pack = cook(tmp.path());
    let recipe =
        inf_island::IslandRecipe::load(&fixture_recipe()).expect("the fixture recipe loads");
    let slug = inf_island::slug(&recipe.name);
    let design = inf_island::read_design(&recipe).expect("the design reads");
    let settlement = walk_target_settlement(&design);

    let mut hosts: Vec<(&str, RuntimeSim)> = vec![
        ("shipping", pack_sim(&pack)),
        ("PIE", loose_sim(&content, &slug)),
    ];

    let mut walks: Vec<(&str, TownPlan, TownWalk)> = Vec::new();
    for (label, sim) in hosts.iter_mut() {
        // Stand in the town first, so its cells activate and its blocks build.
        let hero = hero_entity(sim).expect("a hero");
        let centre = glam::DVec3::new(settlement.centre.x, 0.0, settlement.centre.y);
        set_hero(sim, hero, centre);
        for _ in 0..8 {
            sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
        }
        let plan = plan_town_walk(sim, &content, &recipe, &settlement);
        let walk = walk_the_town(sim, &plan);
        println!(
            "NPC1c {label}: {} steps, {} distinct states, arrived {}, \
             steered {}, blocked {}, doors {}/{} pressed/opened",
            walk.steps,
            walk.distinct,
            walk.arrived,
            walk.steered,
            walk.blocked,
            walk.pressed,
            walk.opened
        );
        println!(
            "NPC1c {label}: {} blocked-agent look(s) at a door",
            walk.considered
        );
        let feet = walk.end - glam::DVec3::new(0.0, 0.9, 0.0);
        let on = plan.path.project(feet);
        println!(
            "NPC1c {label}: the body reached {:.2} m of {:.2}, {:.2} m off the spine",
            on.s_m,
            plan.path.length_m(),
            on.distance_m
        );
        walks.push((label, plan, walk));
    }

    // -- coverage, per host, before anything is compared --------------------
    for (label, plan, walk) in &walks {
        assert!(
            plan.kinds.contains(&inf_nav::NavKind::Street),
            "{label}: the route walks no street"
        );
        assert!(
            plan.kinds.contains(&inf_nav::NavKind::Doorway),
            "{label}: the route passes through no doorway"
        );
        assert!(
            plan.kinds.contains(&inf_nav::NavKind::Stair),
            "{label}: the route climbs no stair"
        );
        assert!(
            plan.cost_m > 30.0,
            "{label}: the planned walk is {:.1} m, which is not across a town",
            plan.cost_m
        );
        // ONE network out of TWO producers, asserted as node counts rather than
        // inferred from the route: a graph that had absorbed nothing would still
        // hand back a `kinds` list if the search had somewhere to go.
        let streets = plan
            .graph
            .nodes()
            .filter(|n| inf_nav::domain::of(n.id) == inf_nav::domain::STREET)
            .count();
        let interior = plan
            .graph
            .nodes()
            .filter(|n| inf_nav::domain::of(n.id) == inf_nav::domain::BUILDING)
            .count();
        assert!(
            streets > 0 && interior > 0,
            "{label}: the network is {streets} street + {interior} interior node(s)"
        );
        assert!(
            walk.steered > 0,
            "{label}: the crowd step never wrote a steering intent, so the NPC \
             was not walking"
        );
        assert!(
            walk.tiers[2] == 0 && walk.tiers[3] == 0,
            "{label}: the agent dropped out of a tier with a body during the \
             walk ({:?}), so part of it was not steered at all",
            walk.tiers
        );
        assert!(
            walk.opened >= 1,
            "{label}: the NPC never opened a door ({} pressed)",
            walk.pressed
        );
        // **It went THROUGH the door it opened**, which is the claim the opened
        // counter alone does not make: five metres past the threshold is inside
        // the building and past the leaf's own arc.
        assert!(
            walk.reached_m > plan.entrance_s + 5.0,
            "{label}: the NPC reached {:.2} m of its route against an entrance \
             at {:.2} m -- it opened a door and did not walk through it",
            walk.reached_m,
            plan.entrance_s
        );
        // **AND IT CLIMBED THE STAIRS**, which is the claim island wave I8a's
        // carried remainder said nothing in this engine had ever made: *nothing
        // has ever walked a character up a flight*. Something has now. The
        // number is how far up the flight the NPC's own feet got, under
        // `move_and_slide`, over the generator's real treads -- and half a
        // storey is the bar because that is unambiguously *up the stairs* rather
        // than *onto the bottom step*.
        let storey = plan.upstairs.y - plan.ground_y;
        let climbed = walk.peak_y - plan.ground_y;
        assert!(
            climbed > storey * 0.5,
            "{label}: the NPC's feet peaked {climbed:.2} m up a {storey:.2} m \
             storey -- it did not climb the flight"
        );
        println!(
            "NPC1c {label}: feet peaked {climbed:.2} m up a {storey:.2} m storey \
             ({:.0} % of the flight); arrived {}",
            climbed / storey * 100.0,
            walk.arrived
        );
        assert!(
            walk.distinct > walk.steps / 2,
            "{label}: {} distinct states over {} steps -- the world stopped \
             moving",
            walk.distinct,
            walk.steps
        );
        // **THE FLIGHT IS WHERE IT STOPS, AND THAT IS A MEASUREMENT** (NPC1c).
        //
        // The route ends on the first floor and `kinds` above proves the nav
        // layer plans the climb; the body does not make it, and the reason is
        // the *generator's* rather than the follower's. A flight fills its whole
        // core -- the assembler insets it by one wall thickness, so the landing
        // all the way round is `{landing} m` -- and this building's stair door
        // opens onto the SIDE of the run. A 0.6 m capsule cannot walk a 0.2 m
        // landing to reach the bottom step, so it stands against the middle of
        // the flight, about a storey and a half of treads above it.
        //
        // Nothing here is refused for want of autostep: the treads are
        // 0.18 m x 0.23 m against a 0.45 m step height and a 0.15 m minimum
        // width, which the mover climbs. What is missing is somewhere to stand.
        //
        // So this arm holds the SHAPE of the remainder rather than pretending it
        // away, and it fails the day the generator gives a core a landing a body
        // can stand on -- which is the day this ledger sentence is rewritten and
        // the I8a carried item finally dies.
        let landing = plan.flight.4;
        assert!(
            landing < 0.6,
            "{label}: the stair core's landing is {landing:.2} m, wide enough \
             for a 0.6 m capsule to walk round the flight -- the NPC1c \
             remainder about climbing is closed and the ledger has to say so"
        );
        assert!(
            plan.flight.2 <= inf_ecs::components::CharacterMovement::default().step_height_m,
            "{label}: a tread rises {:.3} m against a {:.2} m autostep, so the \
             flight is refused for a REASON THIS GATE DOES NOT NAME",
            plan.flight.2,
            inf_ecs::components::CharacterMovement::default().step_height_m
        );
        assert!(
            plan.flight.3 >= inf_ecs::components::CharacterMovement::default().step_min_width_m,
            "{label}: a tread runs {:.3} m against a {:.2} m minimum width",
            plan.flight.3,
            inf_ecs::components::CharacterMovement::default().step_min_width_m
        );
    }

    // -- and then PIE == shipping, step for step ----------------------------
    let (a_label, a_plan, a) = &walks[0];
    let (b_label, b_plan, b) = &walks[1];
    assert_eq!(
        a_plan.nodes, b_plan.nodes,
        "the two hosts planned different routes over their own networks"
    );
    assert_eq!(
        a_plan.door, b_plan.door,
        "the two hosts aimed at different doors"
    );
    assert_eq!(
        a.steps, b.steps,
        "{a_label} took {} steps and {b_label} took {}",
        a.steps, b.steps
    );
    for (i, (x, y)) in a.digests.iter().zip(b.digests.iter()).enumerate() {
        assert_eq!(
            x, y,
            "{a_label} and {b_label} diverged at step {i} of the town walk"
        );
    }
    assert_eq!(a.distinct, b.distinct);
    println!(
        "NPC1c GATE: {} steps of town walk, {} distinct states, identical on \
         both hosts; route {:.1} m over {} nodes; {} door(s) opened",
        a.steps,
        a.distinct,
        a_plan.cost_m,
        a_plan.nodes.len(),
        a.opened
    );
}

// ── NPC1d: A DAY IN THE LIFE ────────────────────────────────────────────────

/// How many fixed steps of ONE host the day-in-the-life gate runs — one whole
/// in-game day.
const DAY_STEPS: usize = 1_200;

/// **The rate the gate compresses the island's own day to.**
///
/// The island authors `inf_editor_core::island::ISLAND_CLOCK_RATE = 18`, an
/// eighty-minute day. At 60 Hz that is `4 800 x 60 = 288 000` fixed steps a
/// host, and this gate runs two of them — better than two hours of test process
/// for one arm, in a battery that has to finish.
///
/// So a day is `DAY_STEPS`, fifty steps an in-game hour, and the rate is
/// whatever makes those two agree — **4 320 clock-seconds a sim second, which
/// is 240× the island's own 18**. *(The first write-up of this said 72×, which
/// is the number of clock-SECONDS one fixed step advances — 86 400 / 1 200 —
/// rather than a ratio between two rates. NPC1d audit.)*
///
/// So it runs the SAME day faster, and the compression is exact rather than
/// approximate: a `ScheduleLeg`'s position is a *fraction of its own clock
/// window*, so every agent is in the same place at the same HOUR whatever the
/// rate. What compression does not preserve is the metres per second a leg
/// implies — a body at `Full` still walks at its own gait and simply falls
/// further behind its clock — and that is why the walking-pace claim is armed
/// separately, at the rate the island actually authors, in
/// `the_islands_own_rate_makes_a_commute_a_walk`.
const GATE_CLOCK_RATE: f64 = 86_400.0 * 60.0 / DAY_STEPS as f64;

/// The most steps the gate will spend waiting for a settlement to stream in and
/// its society to be derived.
const SETTLE_STEPS: usize = 4_000;

/// How many consecutive steps a settlement has to fold NOTHING before the gate
/// believes it has finished streaming.
///
/// A society is not settled the moment its first blocks have residents: a
/// settlement's volumes evaluate over many steps as the ground under them pages
/// in, and a gate that stopped at the first quiet step measured four blocks of a
/// hundred and seventy. Measured exactly that way, and the number this constant
/// replaced was **one**.
const QUIET_STEPS: usize = 120;

/// How near its own home or workplace an agent has to be to count as *there*,
/// metres in plan.
const THERE_M: f64 = 4.0;

/// How far the hero walks away for the gate's `Dormant` coda, metres.
///
/// Past `CrowdTier::Far`'s own 512 m from every corner of a town about 320 m
/// across, so the whole population loses its entity.
const TOWN_AWAY_M: f64 = 1_000.0;

/// How far from the settlement's crossroads the hero stands for the day.
///
/// Far enough that most of the town is outside the 32 m `Full` ring and is
/// therefore moved by its clock rather than by a controller; near enough that
/// the ladder still has somebody on its top two rungs. The town's own blocks
/// span about 320 m, so this is its edge.
const TOWN_EDGE_M: f64 = 150.0;

/// One reading of the town, at one hour.
#[derive(Debug, Clone, Copy, PartialEq)]
struct DaySample {
    hour: f64,
    /// Of the agents the CLOCK moves — see `sample_town` for why the ones a
    /// controller moves are counted apart.
    at_home: usize,
    at_work: usize,
    walking: usize,
    /// Agents on a tier that steers. Their bodies walk at their own gait
    /// against a clock this gate runs 240x fast, so where they are is a
    /// statement about the compression rather than about the day.
    steered: usize,
    tiers: [usize; 4],
    glow_step: u16,
    sun_y: f64,
}

/// What one host's day looked like.
struct DayRun {
    digests: Vec<u64>,
    distinct: usize,
    settle_steps: usize,
    society: inf_ecs::society::SocietyStats,
    samples: Vec<DaySample>,
    /// The highest count each tier ever reached — over the settle, the day and
    /// the coda together, because the ladder is exercised by where the hero is
    /// and the hero is in three places over this gate.
    tier_peak: [usize; 4],
    /// The coda: how many agents went `Dormant`, and how far the furthest of
    /// them was from where its own schedule says it should be when it came back.
    coda: (usize, f64),
    /// **Steering intents written over the whole gate** — the settle, the day
    /// and the coda. Non-zero is the falsifier for the NPC1d-audit defect: a
    /// scheduled agent that reached `Full` used to wish `ZERO` on every step and
    /// stand still, because the walkability test asked its diagnostic route for
    /// a SPEED and a schedule has none.
    steered_ever: u64,
    /// The most agents that were `Full` at once during the rush-hour coda — the
    /// anti-vacuity half of [`steered_ever`](Self::steered_ever).
    rush: usize,
    /// Commute lengths over the whole population, metres: min / median / max.
    commute_m: (f64, f64, f64),
    /// Scheduled agents, and of those, agents with a full four-leg day.
    scheduled: (usize, usize),
    /// Glazed instances the settlements hold — the things a night lights up.
    glazed: usize,
}

/// The local hour this host's clock is on.
fn local_hour(sim: &RuntimeSim) -> f64 {
    inf_ecs::sky::local_hour(sim.world())
}

/// The glow step the island's own sky implies right now — the I8b substrate,
/// read through `resolve_sky` rather than through a projection, because the
/// projection reads the same direction and this arm is about the CLOCK.
fn glow_now(sim: &RuntimeSim) -> (u16, f64) {
    let sky = inf_ecs::sky::resolve_sky(sim.world()).expect("the island carries a clock");
    let dir = sky.sun.as_vec3();
    (inf_render::night_glow_step(dir), sky.sun.y)
}

/// Step until the settlement has streamed in and every resident has a day.
///
/// **Quiet for [`QUIET_STEPS`] consecutive steps**, not merely quiet once: a
/// settlement's volumes evaluate as the ground under them pages in, so the first
/// step that folds nothing is somewhere in the middle of a town rather than at
/// the end of one.
fn settle_the_society(sim: &mut RuntimeSim, peak: &mut [usize; 4], steered: &mut u64) -> usize {
    let mut quiet = 0usize;
    for i in 0..SETTLE_STEPS {
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
        let st = sim.crowd_stats();
        for (t, n) in st.per_tier.iter().enumerate() {
            peak[t] = peak[t].max(*n);
        }
        *steered += st.steered;
        let s = sim.society_stats();
        if s.folded_now > 0 || s.planned_now > 0 || s.pending > 0 {
            quiet = 0;
        } else {
            quiet += 1;
        }
        if s.agents > 0 && quiet >= QUIET_STEPS {
            return i + 1;
        }
    }
    let s = sim.society_stats();
    panic!(
        "the society never settled in {SETTLE_STEPS} steps: {} volume(s), {} \
         home(s), {} agent(s), {} still pending",
        s.volumes, s.homes, s.agents, s.pending
    );
}

/// Read the population's schedules — the commute lengths and how many agents
/// got a full day.
fn read_schedules(sim: &RuntimeSim) -> ((f64, f64, f64), (usize, usize)) {
    let pop = sim
        .world()
        .world()
        .get_resource::<inf_ecs::crowd::CrowdPopulationRes>()
        .expect("the island installed a population");
    let mut lens: Vec<f64> = Vec::new();
    let mut scheduled = 0usize;
    let mut full = 0usize;
    for rec in pop.records.values() {
        let Some(s) = rec.schedule.as_ref() else {
            continue;
        };
        scheduled += 1;
        if s.legs().len() == 4 {
            full += 1;
        }
        lens.push(s.legs()[0].path.length_m());
    }
    lens.sort_by(f64::total_cmp);
    let stat = if lens.is_empty() {
        (0.0, 0.0, 0.0)
    } else {
        (lens[0], lens[lens.len() / 2], lens[lens.len() - 1])
    };
    ((stat.0, stat.1, stat.2), (scheduled, full))
}

/// Where the town is at this instant — read off the RECORDS' own `last`, which
/// is where the step put every agent (the body's place for a steered tier, the
/// clock's for the rest), not off a re-derived answer.
fn sample_town(sim: &RuntimeSim, hour: f64) -> DaySample {
    let pop = sim
        .world()
        .world()
        .get_resource::<inf_ecs::crowd::CrowdPopulationRes>()
        .expect("a population");
    let mut s = DaySample {
        hour,
        at_home: 0,
        at_work: 0,
        walking: 0,
        steered: 0,
        tiers: [0; 4],
        glow_step: 0,
        sun_y: 0.0,
    };
    let plan_d =
        |a: glam::DVec3, b: glam::DVec3| -> f64 { glam::DVec2::new(a.x - b.x, a.z - b.z).length() };
    for rec in pop.records.values() {
        s.tiers[rec.tier.as_u8() as usize] += 1;
        let Some(sched) = rec.schedule.as_ref() else {
            continue;
        };
        // **A steered body is not a census.** A tier with a controller is moved
        // by `move_and_slide` at its own 1.65 m/s gait, and this gate runs the
        // day 240x fast -- so a `Full` agent is permanently strung out along its
        // route and where it stands says how compressed the clock is, not what
        // time it is. Counted, reported, and kept out of the town's census;
        // every other tier's position IS `route(clock)`, which is the schedule's
        // own answer written into the world by the step.
        if rec.tier.steers() {
            s.steered += 1;
            continue;
        }
        let first = &sched.legs()[0];
        let home = first.path.points()[0];
        let work = first.path.points()[first.path.points().len() - 1];
        let (dh, dw) = (plan_d(rec.last, home), plan_d(rec.last, work));
        if dh <= THERE_M {
            s.at_home += 1;
        } else if dw <= THERE_M {
            s.at_work += 1;
        } else {
            s.walking += 1;
        }
    }
    let (step, y) = glow_now(sim);
    s.glow_step = step;
    s.sun_y = y;
    s
}

/// **THE NPC1d GATE**: a whole day over a settlement, on both hosts, byte for
/// byte.
///
/// # What it claims, and the order the claims are made in
///
/// Coverage first, per host, because two empty towns agree perfectly: the level
/// derived a population from its OWN buildings (nobody installed one), the
/// population is at home at two in the morning and at work at ten, somebody is
/// walking in between, the tier ladder is exercised across the day, and the
/// windows the island has always had are lit at night and dark at noon. Only
/// then are the two hosts' traces compared.
///
/// # The day is compressed, and the compression is exact
///
/// See [`GATE_CLOCK_RATE`]. A leg is a fraction of its own clock window, so the
/// day has the same shape at any rate; the walking-pace claim, which does NOT
/// survive compression, is armed at the island's authored rate in its own arm.
#[test]
fn pie_equals_shipping_over_a_day_in_the_life() {
    let tmp = tempfile::tempdir().expect("a temp dir");
    let proj = build_project(tmp.path());
    let content = proj.join("Content");
    let pack = cook(tmp.path());
    let recipe =
        inf_island::IslandRecipe::load(&fixture_recipe()).expect("the fixture recipe loads");
    let slug = inf_island::slug(&recipe.name);
    let design = inf_island::read_design(&recipe).expect("the design reads");
    let settlement = walk_target_settlement(&design);

    // The hours the town is read at, and what each is for.
    // …and two of them are **past dusk** (wave VEN1b). Twenty-one hundred is
    // the middle of a night out and one in the morning is the small hours, so a
    // town that goes out and a town that does not are two different pictures
    // rather than one blank one. Which of the two THIS settlement is, and why,
    // is asserted below beside the emptiness it explains.
    //
    // **Ascending**, and that is load-bearing rather than tidy: the sampler
    // walks this list forward as the step index passes each threshold, so an
    // hour out of order fires immediately after its predecessor and the reading
    // is taken at the wrong time of day. (Measured: `1.0` written last gave a
    // "22.1 h" sample and `at(1.0)` found nothing.) The day begins at midnight,
    // so the small hours come first.
    let hours: [f64; 8] = [1.0, 2.0, 7.0, 8.5, 10.0, 19.0, 21.0, 22.0];

    let mut hosts: Vec<(&str, RuntimeSim)> = vec![
        ("shipping", pack_sim(&pack)),
        ("PIE", loose_sim(&content, &slug)),
    ];
    let mut runs: Vec<(&str, DayRun)> = Vec::new();

    for (label, sim) in hosts.iter_mut() {
        // Stand in the town so its cells activate, its blocks build and its
        // society is derived. The hero does not move again: this gate is about
        // the CLOCK, and a moving anchor would make the tier counts a statement
        // about a drive line.
        let hero = hero_entity(sim).expect("a hero");
        let centre = glam::DVec3::new(settlement.centre.x, 0.0, settlement.centre.y);
        set_hero(sim, hero, centre);
        sim.world_mut().mark_dirty();
        let mut tier_peak = [0usize; 4];
        let mut steered_ever = 0u64;
        let settle_steps = settle_the_society(sim, &mut tier_peak, &mut steered_ever);
        let society = sim.society_stats();
        let (commute_m, scheduled) = read_schedules(sim);
        let glazed = {
            let w = sim.world().world();
            let mut n = 0usize;
            for e in w.iter_entities() {
                if let Some(v) = e.get::<inf_ecs::components::PcgVolume>() {
                    n += v.evaluated.iter().filter(|i| i.glow > 0.0).count();
                }
            }
            n
        };
        println!(
            "NPC1d {label}: settled in {settle_steps} steps -- {} volume(s), {} \
             home(s), {} agent(s) ({} declined), {} scheduled ({} full days), \
             {} homebound, {} housebound, {} doorless, {} with no walk home, \
             {} with no errand",
            society.volumes,
            society.homes,
            society.agents,
            society.homes_declined,
            scheduled.0,
            scheduled.1,
            society.homebound,
            society.housebound,
            society.doorless,
            society.no_return,
            society.errandless
        );
        println!(
            "NPC1d {label}: network {} nodes / {} edges, {} frontage(s) ({} \
             refused), {} crossing(s), {} salt collision(s); street routes {} \
             searched / {} cached",
            society.nodes,
            society.edges,
            society.frontages,
            society.frontages_refused,
            society.crossings,
            society.salt_collisions,
            society.outer_searches,
            society.outer_cached
        );
        println!(
            "NPC1d {label}: commute {:.1} / {:.1} / {:.1} m (min/median/max); at \
             the island's authored rate {} the median is {:.2} m/s",
            commute_m.0,
            commute_m.1,
            commute_m.2,
            inf_editor_core::island::ISLAND_CLOCK_RATE,
            commute_m.1 * inf_editor_core::island::ISLAND_CLOCK_RATE / 3600.0
        );

        // ── the day itself ──────────────────────────────────────────────────
        //
        // **The hero steps to the town's edge**, and that is not staging. Standing
        // at the crossroads puts every one of the town's agents inside the 32 m
        // `Full` ring within a few in-game hours -- their bodies, strung out along
        // their routes by the compression, drift toward the middle -- and a census
        // taken over three hundred steered bodies measures the compression rather
        // than the day. From the edge NOBODY steers -- measured, `+0 steered` at
        // every one of the six hours -- and every agent is moved by its clock,
        // which is the tier ladder's whole point and is what makes `rec.last`
        // the schedule's own answer rather than a body's opinion of it. The
        // `Full` rung is exercised by the settle, which happens at the
        // crossroads, and the `Dormant` rung by the coda.
        set_hero(sim, hero, centre + glam::DVec3::new(TOWN_EDGE_M, 0.0, 0.0));
        sim.world_mut().mark_dirty();
        for _ in 0..8 {
            sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
        }
        // Wind the clock to local midnight and compress it, so `DAY_STEPS` is
        // exactly one turn and step `i` is hour `24 i / DAY_STEPS`.
        {
            let w = sim.world_mut().world_mut();
            let mut q = w.query::<&mut inf_ecs::components::TimeOfDay>();
            let mut wound = 0usize;
            for mut tod in q.iter_mut(w) {
                tod.seconds = (86_400.0 - tod.longitude_deg * 240.0).rem_euclid(86_400.0);
                tod.rate = GATE_CLOCK_RATE;
                wound += 1;
            }
            assert!(wound > 0, "the island carries no clock to wind");
        }
        let mut run = DayRun {
            digests: Vec::with_capacity(DAY_STEPS),
            distinct: 0,
            settle_steps,
            society,
            samples: Vec::new(),
            tier_peak,
            coda: (0, 0.0),
            steered_ever,
            rush: 0,
            commute_m,
            scheduled,
            glazed,
        };
        let mut seen: std::collections::BTreeSet<u64> = Default::default();
        let mut next = 0usize;
        for i in 0..DAY_STEPS {
            sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
            let d = digest(&sim.state_bytes());
            seen.insert(d);
            run.digests.push(d);
            let st = sim.crowd_stats();
            for (t, n) in st.per_tier.iter().enumerate() {
                run.tier_peak[t] = run.tier_peak[t].max(*n);
            }
            run.steered_ever += st.steered;
            if next < hours.len() && i >= (hours[next] / 24.0 * DAY_STEPS as f64) as usize {
                let s = sample_town(sim, local_hour(sim));
                println!(
                    "NPC1d {label}: {:>5.2} h -- {} home / {} work / {} walking \
                     (+{} steered); tiers {:?}; sun y {:+.3} -> glow {}",
                    s.hour,
                    s.at_home,
                    s.at_work,
                    s.walking,
                    s.steered,
                    s.tiers,
                    s.sun_y,
                    s.glow_step
                );
                run.samples.push(s);
                next += 1;
            }
        }
        run.distinct = seen.len();

        // ── the coda: the hero leaves, and the town keeps its day ───────────
        //
        // **`Dormant` is a cost tier, not a one-way door.** Past 512 m every
        // agent loses its entity; the schedule is a pure function, so when the
        // hero comes back each one is where its CLOCK says rather than where it
        // was when it went away. That sentence is NPC1c's carried item about
        // `Dormant` re-materializing at `last`, and this is the arm for it.
        set_hero(
            sim,
            hero,
            centre + glam::DVec3::new(TOWN_AWAY_M, 0.0, TOWN_AWAY_M),
        );
        sim.world_mut().mark_dirty();
        let mut dormant = 0usize;
        for _ in 0..30 {
            sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
            let st = sim.crowd_stats();
            for (t, n) in st.per_tier.iter().enumerate() {
                run.tier_peak[t] = run.tier_peak[t].max(*n);
            }
            dormant = dormant.max(st.per_tier[3]);
        }
        // …and back. The clock has run on while nobody was watching.
        set_hero(sim, hero, centre + glam::DVec3::new(TOWN_EDGE_M, 0.0, 0.0));
        sim.world_mut().mark_dirty();
        for _ in 0..30 {
            sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
            for (t, n) in sim.crowd_stats().per_tier.iter().enumerate() {
                run.tier_peak[t] = run.tier_peak[t].max(*n);
            }
        }
        // Every re-materialized agent stands where its own schedule puts it at
        // the hour the clock now reads — asserted as the WORST agent's distance
        // from its own clock answer, over the whole population.
        let worst = {
            let hour = local_hour(sim);
            let pop = sim
                .world()
                .world()
                .get_resource::<inf_ecs::crowd::CrowdPopulationRes>()
                .expect("a population");
            let t_s = pop.steps as f64 * (1.0 / 60.0);
            let clock = inf_ecs::crowd::CrowdClock::new(t_s, hour);
            let mut worst = 0.0f64;
            for (g, rec) in &pop.records {
                if rec.tier.steers() {
                    continue;
                }
                let want = rec.position_at(*g, clock);
                worst = worst.max((rec.last - want).length());
            }
            worst
        };
        // ── the rush-hour coda: somebody actually STEERS ────────────────────
        //
        // **Every reading above is of a town the CLOCK moves**, because from the
        // town's edge nothing is `Full`. That is the right way to census a day
        // and it is blind to the one tier that walks its body: a scheduled agent
        // that reached `Full` used to wish ZERO on every step — the walkability
        // test asked its *diagnostic* route for a speed and a schedule has none
        // — and stood perfectly still while its clock walked on. Not one claim
        // this gate makes would have failed.
        //
        // It is also why the settle cannot be the arm: the island's committed
        // clock reads **02:18 local**, so the residents nearest the crossroads
        // are `Full`, on their last leg, and correctly standing at home asleep.
        // So the hero comes back to the crossroads at half past eight in the
        // morning, which is the middle of the commute, and the bodies there have
        // somewhere to be.
        {
            let w = sim.world_mut().world_mut();
            let mut q = w.query::<&mut inf_ecs::components::TimeOfDay>();
            for mut tod in q.iter_mut(w) {
                let local = (8.5 * 3600.0 - tod.longitude_deg * 240.0).rem_euclid(86_400.0);
                tod.seconds = local;
                tod.rate = 0.0;
            }
        }
        set_hero(sim, hero, centre);
        sim.world_mut().mark_dirty();
        for _ in 0..60 {
            sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
            let st = sim.crowd_stats();
            for (t, n) in st.per_tier.iter().enumerate() {
                run.tier_peak[t] = run.tier_peak[t].max(*n);
            }
            run.steered_ever += st.steered;
            run.rush = run.rush.max(st.per_tier[0]);
        }
        run.coda = (dormant, worst);
        println!(
            "NPC1d {label}: coda -- {dormant} agent(s) went dormant at \
             {TOWN_AWAY_M:.0} m; back at {:.2} h the worst re-materialized agent \
             is {worst:.4} m from its own schedule's answer; {} steering \
             intent(s) over the whole gate, {} agent(s) `Full` at the 08:30 \
             rush hour",
            local_hour(sim),
            run.steered_ever,
            run.rush
        );

        println!(
            "NPC1d {label}: {DAY_STEPS} steps, {} distinct states; tier peaks \
             {:?}; crowd trace {} B a step over {} agent(s) = {} B a day",
            run.distinct,
            run.tier_peak,
            run.society.agents * inf_ecs::crowd::AGENT_TRACE_BYTES,
            run.society.agents,
            run.society.agents * inf_ecs::crowd::AGENT_TRACE_BYTES * DAY_STEPS
        );
        runs.push((label, run));
    }

    // ── coverage, per host, before anything is compared ────────────────────
    for (label, r) in &runs {
        assert!(
            r.society.agents > 0,
            "{label}: the level derived NO population from its own buildings"
        );
        assert_eq!(
            r.society.pending, 0,
            "{label}: {} resident(s) never got a day",
            r.society.pending
        );
        assert_eq!(
            r.society.salt_collisions, 0,
            "{label}: {} building(s) share a nav namespace, so a route can walk \
             into the wrong house",
            r.society.salt_collisions
        );
        assert_eq!(
            r.society.guid_refusals, 0,
            "{label}: {} agent guid(s) collided with a level entity",
            r.society.guid_refusals
        );
        assert!(
            r.scheduled.1 > 0,
            "{label}: {} agent(s) are scheduled and NONE has a full four-leg day \
             -- nobody commutes",
            r.scheduled.0
        );
        // **NOBODY GOES TO WORK AND STAYS THERE** (NPC1d audit). An agent whose
        // walk home does not route leaves at eight and stands at its desk
        // through the night; `DayKind::Full` is returned the moment the
        // OUTBOUND commute routes, so before this counter existed the report
        // called that a full day. The difference between `scheduled` and
        // `full four-leg` is `errandless`, which is carried item 2 in numbers
        // and is reported rather than asserted.
        assert_eq!(
            r.society.no_return, 0,
            "{label}: {} agent(s) have a workplace and no walk home",
            r.society.no_return
        );
        // **AND THE NIGHT SHIFT IS THE THIRD WAY** (wave EMS1). A night
        // worker's day is the TWO-leg one — out at eighteen, home at three —
        // and never the four-leg commute; `a_town_with_a_venue_in_it_has_a_night`
        // is where that is stated as a rule. It was absent from this sum only
        // because the CI fixture's one venue is in a settlement whose night job
        // nobody had claimed: the moment a fire hall's round-the-clock bay put
        // nine night workers on the island, the arm read nine agents short with
        // nothing to charge them to. The term is not slack — remove it and the
        // arm fails again.
        assert_eq!(
            r.scheduled.0 - r.scheduled.1,
            r.society.errandless + r.society.homebound + r.society.night_workers,
            "{label}: {} of {} scheduled agents have fewer than four legs, \
             against {} errandless + {} homebound + {} night worker(s) -- with \
             no_return at zero those are the only three ways to be short of a leg",
            r.scheduled.0 - r.scheduled.1,
            r.scheduled.0,
            r.society.errandless,
            r.society.homebound,
            r.society.night_workers
        );
        assert!(
            r.society.crossings > 0 && r.society.frontages > 0,
            "{label}: {} crossing(s) and {} frontage(s) -- the network is not one \
             thing",
            r.society.crossings,
            r.society.frontages
        );
        assert!(
            r.commute_m.1 > 10.0,
            "{label}: the median commute is {:.1} m, which is not across a town",
            r.commute_m.1
        );

        let at = |h: f64| -> &DaySample {
            r.samples
                .iter()
                .find(|s| (s.hour - h).abs() < 1.0 || (s.hour - h).abs() > 23.0)
                .unwrap_or_else(|| panic!("{label}: no sample near {h} h"))
        };
        let night = at(2.0);
        let morning = at(7.0);
        let commuting = at(8.5);
        let working = at(10.0);
        let evening = at(19.0);
        let late = at(22.0);
        let total = |s: &DaySample| s.at_home + s.at_work + s.walking;

        // **THE TOWN IS AT HOME AT NIGHT AND AT WORK BY TEN.** Asserted as a
        // majority rather than as all of them, because a homebound agent has no
        // workplace to be at and stays home all day by design.
        assert!(
            night.at_home * 2 > total(night),
            "{label}: at {:.2} h only {} of {} are at home",
            night.hour,
            night.at_home,
            total(night)
        );
        assert!(
            morning.at_home * 2 > total(morning),
            "{label}: at {:.2} h only {} of {} are at home before work",
            morning.hour,
            morning.at_home,
            total(morning)
        );
        assert!(
            working.at_work > 0,
            "{label}: NOBODY is at work at {:.2} h ({} home, {} walking)",
            working.hour,
            working.at_home,
            working.walking
        );
        assert!(
            working.at_work > night.at_work,
            "{label}: {} at work at ten against {} at two in the morning -- the \
             town does not go to work",
            working.at_work,
            night.at_work
        );
        // **AND SOMEBODY IS ON THE STREET IN BETWEEN**, which is the claim the
        // two census readings above cannot make on their own.
        assert!(
            commuting.walking > 0,
            "{label}: at {:.2} h {} are at home and {} at work and NOBODY is \
             between them",
            commuting.hour,
            commuting.at_home,
            commuting.at_work
        );
        // **AND IT EMPTIES — BECAUSE THIS TOWN HAS NOWHERE TO GO** (wave
        // VEN1b). By ten at night every clock-driven agent is home: the last
        // leg starts at six and takes an hour, jitter included.
        //
        // That was an unconditional claim about the engine until this wave and
        // is now a claim about this settlement's CONTENT, so the arm carries the
        // reason with it. `walk_target_settlement` picks the fixture's four-block
        // `Fixture Town`, which gets **no venue** — a city's nightlife strip
        // starts at ring 1 and `VENUE_SHARE` refuses to spend more than a third
        // of a settlement on it (VEN1a's own ruling, and the reason a hamlet
        // with a nightclub is stranger than a hamlet without one). A town with
        // no bar in it goes to bed, and `leisure_places == 0` is what says the
        // emptiness is the content's answer rather than a schedule that never
        // learned about the evening.
        //
        // The populated night is `pie_equals_shipping_at_a_club_on_a_saturday_
        // night`, at the settlement that HAS one.
        assert_eq!(
            r.society.leisure_places, 0,
            "{label}: this settlement offers {} places to spend an evening, so the arms below are measuring the wrong thing — they say a town empties BECAUSE it has nowhere to go",
            r.society.leisure_places
        );
        // **A NIGHT JOB IS NOT A NIGHT OUT** (wave EMS1). This used to assert
        // zero night jobs, and that was a proxy for the same thing the line
        // above states directly: the only night work on this island was a
        // venue's counter, so "no night job" and "no nightlife" were one
        // sentence. A fire hall's apparatus bay is worked round the clock and
        // sells nobody a drink, so the two came apart the moment a civic strip
        // stood one on the fixture's crossroads. What the arms below need is
        // that the town has nowhere to GO — which is `leisure_places` — and that
        // the people still out are a small watch rather than a population.
        assert!(
            r.society.night_jobs * 10 < r.society.agents.max(1),
            "{label}: {} of {} agents have a night job — that is a shift town, \
             not a civic watch, and 'the town goes to bed' below is measuring \
             the wrong thing",
            r.society.night_jobs,
            r.society.agents
        );
        assert_eq!(r.society.revellers, 0);
        assert!(
            late.at_home * 2 > total(late),
            "{label}: at {:.2} h only {} of {} are home",
            late.hour,
            late.at_home,
            total(late)
        );
        // …and past midnight too, which is the half the six-hour sample never
        // reached: `EVENING_OUT_H` is twenty hundred and `NIGHT_HOME_H` is two
        // in the morning, so a schedule that had grown an evening it could not
        // fill would show it HERE — an agent standing at a leisure slot it
        // never claimed, or walking a leg with no destination.
        let (evening_out, small_hours) = (at(21.0), at(1.0));
        assert!(
            evening_out.at_home * 2 > total(evening_out),
            "{label}: at {:.2} h only {} of {} are home in a town with no bar",
            evening_out.hour,
            evening_out.at_home,
            total(evening_out)
        );
        assert!(
            small_hours.at_home * 2 > total(small_hours),
            "{label}: at {:.2} h only {} of {} are home",
            small_hours.hour,
            small_hours.at_home,
            total(small_hours)
        );
        println!(
            "NPC1d {label}: past dusk — {:.2} h {} home / {} walking; {:.2} h {} home / {} walking; the settlement offers {} leisure place(s) and {} night job(s)",
            evening_out.hour,
            evening_out.at_home,
            evening_out.walking,
            small_hours.hour,
            small_hours.at_home,
            small_hours.walking,
            r.society.leisure_places,
            r.society.night_jobs
        );
        // Seven in the evening is the middle of the walk home and is reported
        // rather than asserted: with half an hour of jitter either way, whether
        // a given agent is home at 19:00 is a statement about its own seed.
        println!(
            "NPC1d {label}: at {:.2} h the town is {} home / {} work / {} \
             walking (+{} steered)",
            evening.hour, evening.at_home, evening.at_work, evening.walking, evening.steered
        );

        // **THE WINDOWS.** The island has always had glazed modules and its
        // clock was frozen at half past ten in the morning, so the I8b substrate
        // had never once returned a lit step on it. It does now, and it goes
        // dark again by day.
        assert!(
            r.glazed > 0,
            "{label}: the settlements hold no glazed module, so the night \
             comparison would be vacuous"
        );
        assert_eq!(
            working.glow_step, 0,
            "{label}: the windows are lit at {:.2} h with the sun at y {:+.3}",
            working.hour, working.sun_y
        );
        assert!(
            night.glow_step > 0 && late.glow_step > 0,
            "{label}: the windows are dark at {:.2} h (sun y {:+.3}) and {:.2} h \
             (sun y {:+.3})",
            night.hour,
            night.sun_y,
            late.hour,
            late.sun_y
        );
        assert!(
            night.sun_y < 0.0 && working.sun_y > 0.0,
            "{label}: the sun did not rise and set over the day (night {:+.3}, \
             noon {:+.3})",
            night.sun_y,
            working.sun_y
        );

        // **THE LADDER IS EXERCISED.** At least three rungs carry somebody at
        // some point in the day; the fourth is reported rather than demanded,
        // because whether a settlement is wider than `Dormant`'s radius is a
        // property of the recipe rather than of this system.
        let rungs = r.tier_peak.iter().filter(|n| **n > 0).count();
        assert_eq!(
            rungs, 4,
            "{label}: only {rungs} of the four tiers ever carried an agent: {:?} \
             -- the hero stands at the crossroads while the town streams in, at \
             the town's edge for the day and a kilometre away for the coda, \
             which is the whole ladder",
            r.tier_peak
        );
        assert!(
            r.coda.0 > 0,
            "{label}: nobody went dormant a kilometre from the town"
        );
        // **AND SOMEBODY ACTUALLY STEERED** (NPC1d audit). While the town streams
        // in the hero stands at the crossroads and the nearest residents are
        // `Full`, which is the tier that walks its body through
        // `move_and_slide`. A scheduled agent that reached `Full` used to wish
        // ZERO on every step — the walkability test asked its *diagnostic* route
        // for a speed, and a schedule has none — so it stood perfectly still
        // while its clock walked on. Every claim this gate makes about the day
        // was true of that build too, because from the town's edge nothing
        // steers; this is the reading that would not have been.
        assert!(
            r.rush > 0,
            "{label}: nobody was `Full` at the crossroads at half past eight, so \
             the steering claim below would be vacuous"
        );
        assert!(
            r.steered_ever > 0,
            "{label}: not one steering intent over the whole gate, with {} agent(s) \
             `Full` in the middle of the morning commute — every scheduled agent \
             that got a controller stood still",
            r.rush
        );
        // **A dormant agent comes back at its SCHEDULE, not at `last`.** The
        // tolerance is a millimetre rather than a metre because the claim is
        // exact: the position law is `route(clock)` at every tier, so a
        // re-materialized agent's transform IS its clock's answer.
        assert!(
            r.coda.1 < 1.0e-3,
            "{label}: the worst re-materialized agent is {:.4} m from where its \
             own schedule says it should be",
            r.coda.1
        );
        assert!(
            r.distinct > DAY_STEPS / 2,
            "{label}: {} distinct states over {DAY_STEPS} steps -- the world \
             stopped moving",
            r.distinct
        );
    }

    // ── and then PIE == shipping, step for step ────────────────────────────
    let (a_label, a) = &runs[0];
    let (b_label, b) = &runs[1];
    assert_eq!(
        a.society.agents, b.society.agents,
        "{a_label} derived {} agents and {b_label} derived {}",
        a.society.agents, b.society.agents
    );
    assert_eq!(
        a.society.nodes, b.society.nodes,
        "the two hosts built different networks ({} nodes against {})",
        a.society.nodes, b.society.nodes
    );
    assert_eq!(
        a.settle_steps, b.settle_steps,
        "{a_label} settled in {} steps and {b_label} in {}",
        a.settle_steps, b.settle_steps
    );
    assert_eq!(a.digests.len(), b.digests.len());
    for (i, (x, y)) in a.digests.iter().zip(b.digests.iter()).enumerate() {
        assert_eq!(
            x,
            y,
            "the two hosts diverged at step {i} of {DAY_STEPS}, hour {:.2}",
            24.0 * i as f64 / DAY_STEPS as f64
        );
    }
    assert_eq!(a.distinct, b.distinct);
    println!(
        "NPC1d GATE: {DAY_STEPS} steps a host, {} distinct states, identical on \
         both hosts to the byte; {} agents from {} homes over {} block(s),",
        a.distinct, a.society.agents, a.society.homes, a.society.volumes
    );
}

/// **The rate the island authors makes a commute a WALK**, and this is the arm
/// the compressed gate above cannot make.
///
/// `ISLAND_CLOCK_RATE` was not chosen for looks. A `ScheduleLeg` walks its route
/// over its own clock window, so the metres per second it implies is
/// `length x rate / 3600` — and the rate is the number that makes the island's
/// own median commute a walking pace. This measures that commute on the
/// committed island's own population and holds the implied speed inside a band a
/// person walks in.
#[test]
fn the_islands_own_rate_makes_a_commute_a_walk() {
    let tmp = tempfile::tempdir().expect("a temp dir");
    let proj = build_project(tmp.path());
    let content = proj.join("Content");
    let recipe =
        inf_island::IslandRecipe::load(&fixture_recipe()).expect("the fixture recipe loads");
    let slug = inf_island::slug(&recipe.name);
    let design = inf_island::read_design(&recipe).expect("the design reads");
    let settlement = walk_target_settlement(&design);

    let mut sim = loose_sim(&content, &slug);
    let hero = hero_entity(&sim).expect("a hero");
    let centre = glam::DVec3::new(settlement.centre.x, 0.0, settlement.centre.y);
    set_hero(&mut sim, hero, centre);
    sim.world_mut().mark_dirty();
    settle_the_society(&mut sim, &mut [0usize; 4], &mut 0u64);

    let rate = inf_editor_core::island::ISLAND_CLOCK_RATE;
    assert!(
        rate > 0.0,
        "the island's clock is frozen, so it has no day for a society to live"
    );
    let pop = sim
        .world()
        .world()
        .get_resource::<inf_ecs::crowd::CrowdPopulationRes>()
        .expect("a population");
    let mut speeds: Vec<f64> = Vec::new();
    for rec in pop.records.values() {
        let Some(s) = rec.schedule.as_ref() else {
            continue;
        };
        speeds.push(s.legs()[0].implied_speed_mps(rate));
    }
    speeds.sort_by(f64::total_cmp);
    assert!(!speeds.is_empty(), "no agent has a leg to walk");
    let (lo, med, hi) = (
        speeds[0],
        speeds[speeds.len() / 2],
        speeds[speeds.len() - 1],
    );
    println!(
        "NPC1d rate: at {rate} the island's {} commutes imply {lo:.2} / \
         {med:.2} / {hi:.2} m/s (min/median/max); a walk is 1.65",
        speeds.len()
    );
    let walk = inf_ecs::components::CharacterMovement::default().walk_speed_mps;
    // **The rate is SET from this number**, so the band is tight rather than
    // generous: a first draft of 30 put the median at 2.67 m/s -- a jog -- and a
    // band wide enough to admit that would have certified it.
    assert!(
        med > walk * 0.75 && med < walk * 1.25,
        "at rate {rate} the island's median commute implies {med:.2} m/s against \
         a {walk:.2} m/s walk -- the rate and the schedule's hours disagree \
         about what a commute is"
    );
    // And EVERY commute is a walk, not just the middle one: the slowest is a
    // stroll and the fastest is not a run.
    assert!(
        lo > 0.4 && hi < walk * 1.6,
        "the island's commutes run {lo:.2} to {hi:.2} m/s at rate {rate}"
    );
    // A day at this rate is a real length, stated in the units a reader has.
    let day_min = 86_400.0 / rate / 60.0;
    println!("NPC1d rate: a day is {day_min:.1} minutes of real time");
    assert!(
        (day_min - 80.0).abs() < 0.01,
        "the island's day is {day_min:.1} minutes"
    );
}

// ── island wave VEH1a: THE CAR DRIVES THE CIRCUIT ───────────────────────────
//
// Wave I7 left two open items about the roads it had just built, and this
// section closes both:
//
//   15. "THE CIRCUIT IS DRAWN AND AUDITED, AND NOTHING DRIVES IT. … The drive
//       trace moves the streaming source at 24 m/s, which is the *streaming*
//       claim; it is not a vehicle simulation."
//   16. "The road network's topology is asserted on a flat fixture, not on the
//       island. … A connectivity walk over the built `RoadGraph` is what would
//       close it."
//
// The first is `pie_equals_shipping_when_the_car_drives_the_circuit`, which
// replaces the scripted teleport with a real `step_vehicles` trace: the hero
// walks to a car, presses the interact the seat door already listens for, and
// holds the throttle. The `StreamingSource` is carried by the car rather than
// by a script — still sim state and never a camera — but the PAGING half of
// item 15's sentence stays with the scripted 360 m arm above: this window is
// 34 m and the streamers do not move over it (measured in the arm's header,
// VEH1a audit). What dies here is *"nothing drives it"*.

/// How many fixed steps the car drive runs — **five** seconds at 60 Hz.
///
/// (Its first cut said "ten seconds", which is what it would have been at 600;
/// the wave's own carried item records the cut to five and the header's prose
/// says five. VEH1a audit.)
const DRIVE_STEPS: u64 = 300;

/// Every chassis the **vehicle recogniser** finds in a live sim, in `Guid`
/// order.
///
/// Found rather than named: `inf_ecs::vehicle::rig_of` is the function the
/// physics bridge itself derives a rig with, so an arm that finds its cars this
/// way is asserting that the cooked pack's entities really are vehicles, not
/// that a generator wrote a GUID somewhere.
fn cars(sim: &RuntimeSim) -> Vec<uuid::Uuid> {
    let world = sim.world();
    let mut out: Vec<uuid::Uuid> = world
        .world()
        .iter_entities()
        .filter_map(|e| e.get::<inf_ecs::Guid>().map(|g| g.0))
        .filter(|g| inf_ecs::vehicle::rig_of(world, *g).is_some())
        .collect();
    out.sort();
    out
}

/// **The tuning of every EMERGENCY catalogue row** (wave EMS1) — read from the
/// two doors the island's own recipe parks a fleet with, never restated.
///
/// `station_fleet` says which rows a station keeps and `island_vehicles` is the
/// catalogue they are built from, so a wave that adds an ambulance to a hospital
/// is covered here without touching this function. A restated list is the A14
/// defect this file already carries a scar from.
fn emergency_classes() -> Vec<inf_ecs::components::VehicleClass> {
    let defs = inf_editor_core::vehicle::island_vehicles();
    let mut out = Vec::new();
    for a in inf_pcg::ArchetypeId::ALL {
        for id in inf_editor_core::island::station_fleet(a) {
            if let Some(d) = defs.get(id) {
                if !out.contains(&d.class) {
                    out.push(d.class);
                }
            }
        }
    }
    assert!(
        !out.is_empty(),
        "no emergency row in the catalogue at all, so the filter below excludes \
         nothing and the arms that use it are not measuring what they name"
    );
    out
}

/// **The civilian cars**, which is what a drive arm means by "a car".
///
/// An 8.85-tonne fire appliance is a road vehicle and it is not a car: it covers
/// 7.8 m in five seconds of throttle where the settlement's saloon covers 27,
/// and a "nearest vehicle" rule quietly made it the subject of the circuit drive
/// the moment wave EMS1 parked one. The exclusion is by CLASS — the tuning the
/// catalogue row carries — rather than by name or by guid, so it cannot go stale
/// against a renamed entity.
fn civilian_cars(sim: &RuntimeSim) -> Vec<uuid::Uuid> {
    let emergency = emergency_classes();
    let world = sim.world();
    cars(sim)
        .into_iter()
        .filter(|g| {
            world
                .entity_of(*g)
                .and_then(|e| {
                    world
                        .world()
                        .get::<inf_ecs::components::VehicleClass>(e)
                        .copied()
                })
                .is_none_or(|c| !emergency.contains(&c))
        })
        .collect()
}

/// Where a chassis is, right now.
fn chassis_at(sim: &RuntimeSim, guid: uuid::Uuid) -> glam::DVec3 {
    let e = sim.world().entity_of(guid).expect("a live chassis");
    sim.world()
        .world()
        .get::<inf_ecs::components::Transform>(e)
        .map(|t| t.translation.to_dvec3())
        .expect("with a transform")
}

/// One host's whole drive: the state bytes, and what its car did each step.
struct CarTrace {
    states: Vec<Vec<u8>>,
    /// Per step: wheels on the ground, forward m/s, engine revs and load.
    wheels: Vec<usize>,
    speed: Vec<f64>,
    revs: Vec<f64>,
    load: Vec<f64>,
    /// The chassis position each step.
    at: Vec<glam::DVec3>,
    /// The audio commands the drive queued, by kind: Play, SetPitch, SetVolume.
    audio: (usize, usize, usize),
    /// Fixed steps taken **before** the drive loop — the settle step, the ones
    /// the hero spends being stood beside the car, and the press/release pairs
    /// the interact edge needs (VEH1a audit).
    ///
    /// Counted rather than known, because it is the other half of the audio
    /// arithmetic: the window below opens before all of them, so the engine's
    /// pitch/volume pair count is `pre_steps + DRIVE_STEPS` and not
    /// `DRIVE_STEPS`. A ledger that quotes 334 against a 300-step drive without
    /// this is a number no reader can check.
    pre_steps: usize,
    entered: bool,
}

/// Walk the hero to the nearest car, get in, and drive.
///
/// The **enter is the shipped door**: one step with `actions::INTERACT` held,
/// resolved by `inf_physics::d3::interact::resolve` exactly as an authored
/// `Interactable` or a doorway is. Nothing here reaches into the seat state.
fn drive_a_car(sim: &mut RuntimeSim) -> CarTrace {
    // The audio window opens HERE and not at the drive loop: the engine's one
    // `Play` is emitted on the first step the car appears in an outcome, which
    // is the step that derives its rig — long before anybody is driving it. A
    // window that opened at the throttle counted 0 Plays and 300 SetPitches and
    // read as a defect; the loop was right and the instrument was late.
    let before_audio = sim.audio_command_log().len();
    let hero = hero_entity(sim).expect("the island has a player-controlled hero");
    // **The CIVILIAN cars** (wave EMS1): an appliance is a road vehicle and not
    // a car, and a "nearest vehicle" rule made it this arm's subject the moment
    // a fire hall parked one. See `civilian_cars`.
    let fleet = civilian_cars(sim);
    assert!(
        !fleet.is_empty(),
        "no civilian vehicle in the resident world — the level parks one at each \
         settlement and the recogniser found none of them"
    );
    // The nearest car to the hero's own start, so the walk is short and the
    // choice is a function of the world rather than of an index.
    let hero_at = sim
        .world()
        .world()
        .get::<inf_ecs::components::Transform>(hero)
        .map(|t| t.translation.to_dvec3())
        .expect("the hero has a transform");
    let car = *fleet
        .iter()
        .min_by(|a, b| {
            (chassis_at(sim, **a) - hero_at)
                .length_squared()
                .total_cmp(&(chassis_at(sim, **b) - hero_at).length_squared())
        })
        .expect("checked non-empty");

    // ── 1. STAND BESIDE IT ── inside `ENTER_REACH_M` of the seat, on the ground
    //    rather than on the bodywork: a character teleported onto a dynamic box
    //    is a depenetration, and `request(Driving)` needs a grounded mode.
    // One step first: the bridge derives its vehicle map inside its own
    // `sync_from_world_sim`, so a seat asked for before the sim has stepped is a
    // seat on a rig nothing has recognised yet.
    //
    // Every `step_once` from here to the drive loop is COUNTED, because the
    // audio window opened above and so the engine's pitch/volume pair count is
    // this number plus the drive's (VEH1a audit).
    let mut pre_steps = 0usize;
    sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
    pre_steps += 1;
    let seat = inf_physics::d3::vehicle::seat_pose(sim.bridge3d(), car)
        .expect("the car has a seat once the bridge has derived its rig")
        .0;
    // Beside the driver's door and level with the ground the car is on, so the
    // hero is standing rather than falling: `step_driving` is entered through
    // `request(Driving)`, which refuses from a non-grounded mode.
    let beside = glam::DVec3::new(seat.x + 1.7, seat.y - 0.2, seat.z);
    for _ in 0..24 {
        set_hero(sim, hero, beside);
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
        pre_steps += 1;
    }
    let mode_before = sim
        .world()
        .world()
        .get::<inf_ecs::components::CharacterMovement>(hero)
        .map(|m| m.mode);

    // ── 2. PRESS E ── the same edge a door takes, through the same resolver.
    //    `just_pressed` is the difference against the previous tick's held set,
    //    so a press has to be a RELEASED step followed by a HELD one — pressing
    //    on every step of a loop is one edge and then nothing.
    let mut entered = false;
    for _ in 0..8 {
        sim.step_once(
            inf_player::runtime_sim::RuntimeInput::default()
                .press(inf_ecs::movement::actions::INTERACT),
        );
        pre_steps += 1;
        entered = sim
            .world()
            .world()
            .get::<inf_ecs::components::CharacterMovement>(hero)
            .is_some_and(|m| m.mode == inf_ecs::components::MovementMode::Driving);
        if entered {
            break;
        }
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
        pre_steps += 1;
    }
    if !entered {
        // Say WHY, in the units the door decides in — a gate that reports
        // "false" about a resolver with three refusals has named none of them.
        let hero_now = sim
            .world()
            .world()
            .get::<inf_ecs::components::Transform>(hero)
            .map(|t| t.translation.to_dvec3())
            .unwrap_or_default();
        let reach = inf_physics::d3::vehicle::try_enter(
            sim.bridge3d(),
            hero_now - glam::DVec3::Y * 0.9,
            &std::collections::BTreeSet::new(),
        );
        println!(
            "THE SEAT REFUSED: hero at {hero_now:?} in mode {mode_before:?}, \
             seat at {seat:?}, feet-to-seat {:.2} m against a {} m reach, \
             nearest_seat = {reach:?}",
            ((hero_now - glam::DVec3::Y * 0.9) - seat).length(),
            inf_physics::d3::vehicle::ENTER_REACH_M
        );
    }

    // ── 3. DRIVE ── the throttle held, and NOTHING teleported: from here the
    //    hero's transform is written by the seat step, so the `StreamingSource`
    //    is carried by the car rather than by a script. Carried, not exercised:
    //    the drive covers 34 m and the streamers do not move over it — see the
    //    arm's own header for the measurement and for which arm owns the paging
    //    claim (VEH1a audit).
    let mut t = CarTrace {
        states: Vec::with_capacity(DRIVE_STEPS as usize),
        wheels: Vec::with_capacity(DRIVE_STEPS as usize),
        speed: Vec::with_capacity(DRIVE_STEPS as usize),
        revs: Vec::with_capacity(DRIVE_STEPS as usize),
        load: Vec::with_capacity(DRIVE_STEPS as usize),
        at: Vec::with_capacity(DRIVE_STEPS as usize),
        audio: (0, 0, 0),
        pre_steps,
        entered,
    };
    for step in 0..DRIVE_STEPS {
        // A gentle weave, so the drive is a drive and not a straight line — and
        // so no two steps stand in the same place, which is what the
        // distinctness arm below needs.
        let steer = if (step / 60) % 2 == 0 { 0.2 } else { -0.2 };
        sim.step_once(
            inf_player::runtime_sim::RuntimeInput::default()
                .axis_at(inf_ecs::movement::actions::MOVE_Y, 1.0)
                .axis_at(inf_ecs::movement::actions::MOVE_X, steer),
        );
        t.states.push(sim.state_bytes());
        let out = sim
            .vehicles()
            .iter()
            .find(|o| o.chassis == car)
            .copied()
            .unwrap_or(inf_physics::d3::VehicleOutcome {
                chassis: car,
                wheels_grounded: 0,
                load_n: 0.0,
                forward_mps: 0.0,
                revs: 0.0,
                load: 0.0,
            });
        t.wheels.push(out.wheels_grounded);
        t.speed.push(out.forward_mps);
        t.revs.push(out.revs);
        t.load.push(out.load);
        t.at.push(chassis_at(sim, car));
    }
    for cmd in &sim.audio_command_log()[before_audio..] {
        match cmd {
            inf_audio::AudioCommand::Play(_) => t.audio.0 += 1,
            inf_audio::AudioCommand::SetPitch { .. } => t.audio.1 += 1,
            inf_audio::AudioCommand::SetVolume { .. } => t.audio.2 += 1,
            _ => {}
        }
    }
    t
}

/// **THE DRIVE GATE.** A real `step_vehicles` trace over the island's own
/// circuit, byte-identical on both hosts.
///
/// This is `pie_equals_shipping_on_an_island_drive` with the scripted teleport
/// replaced by a car. What it adds over that arm is that the thing moving is
/// **simulated**: four wheel rays a step into the resident heightfield, a
/// suspension, a friction circle and an engine curve, every one of them floating
/// point over content the two hosts derived separately.
///
/// # What it is NOT, measured rather than argued (VEH1a audit)
///
/// The first cut of this header called it *"a far sharper instrument for the
/// **streaming** claim"*, on the argument that a tile arriving on one host and
/// not the other would separate the traces within a step. The argument is true
/// about the mechanism and false about this window: 300 steps at 60 Hz is five
/// seconds and **27 metres** (34 before VEH2a re-blessed the trace; `audit:`
/// VEH2a — the re-bless published its deltas and left three stale `34`s in this
/// arm's own prose), and the counters say the drive pages **nothing**
/// — `(activations, deactivations, cells, sim tiles, terrain loads)` is
/// `(1, 0, 1, 16, 15)` before it and `(1, 0, 1, 16, 15)` after, on both hosts.
/// The paging half of the streaming claim therefore stays exactly where wave I7
/// put it, on the scripted 360-metre out-and-back
/// (`pie_equals_shipping_on_an_island_drive`, which asserts those same
/// counters), and wave I7's open item 15 — *"nothing drives it"* — is closed by
/// the **vehicle** half, which is what this arm actually proves.
///
/// What it does add to the streaming record is small and real: the two hosts'
/// streaming counters are asserted **equal**, and none of them reaches
/// `state_bytes`, so that is a comparison the byte equality below cannot make.
///
/// Coverage first, then the comparison (the NPC1d law: a gate staged where a
/// defect cannot appear certifies the defect).
#[test]
fn pie_equals_shipping_when_the_car_drives_the_circuit() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let pack = cook(tmp.path());
    let proj = tmp.path().join("island");
    let recipe = inf_island::IslandRecipe::load(&fixture_recipe()).expect("the recipe loads");
    let slug = inf_island::slug(&recipe.name);
    let mut ship = pack_sim(&pack);
    let mut pie = loose_sim(&proj.join("Content"), &slug);

    // ── COVERAGE ── both hosts hold the same fleet, and it is not empty.
    for (who, sim) in [("shipping", &ship), ("PIE", &pie)] {
        let fleet = cars(sim);
        assert!(
            !fleet.is_empty(),
            "{who}: the resident world holds no vehicle at all — the level parks \
             one at each settlement and the recogniser found none"
        );
        println!("{who}: {} car(s) resident at boot", fleet.len());
    }
    assert_eq!(
        cars(&ship),
        cars(&pie),
        "the two hosts disagree about which cars exist"
    );

    // What the two had paged before anybody got in, so the numbers below are
    // what the DRIVE did rather than what the boot did — `streaming_counters`'s
    // own rule, borrowed from the scripted arm this one joins.
    let (ship0, pie0) = (streaming_counters(&ship), streaming_counters(&pie));

    let a = drive_a_car(&mut ship);
    let b = drive_a_car(&mut pie);

    // ── THE DOOR ── the hero really got in, on both hosts. Asserted before the
    //    comparison, because two hosts that both failed to enter agree
    //    perfectly and would certify a seat door that does not work.
    assert!(
        a.entered && b.entered,
        "the hero did not enter the car (shipping {}, PIE {}) — the interact \
         edge reached no seat, so the drive below is a car nobody is in",
        a.entered,
        b.entered
    );

    // ── IT DROVE ── the trace is a vehicle simulation, not a parked car.
    let travelled = (a.at[a.at.len() - 1] - a.at[0]).length();
    let top = a.speed.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let grounded: usize = a.wheels.iter().sum();
    let steps = DRIVE_STEPS as usize;
    println!(
        "THE DRIVE: {travelled:.1} m in {DRIVE_STEPS} steps, top {top:.2} m/s, \
         {grounded} wheel contacts of a possible {}, engine \
         Play/SetPitch/SetVolume = {:?} over {} pre-drive steps + {steps} \
         driving ones",
        steps * 4,
        a.audio,
        a.pre_steps
    );
    // 15 m against a measured **27.0** on the authoring machine. The margin is
    // deliberate and it is not slack: the fixture's terrain is SAMPLED, and the
    // sampling step is the one this repository's portability law exempts by
    // name, so the ground a wheel finds is a fact about the machine that built
    // it. The claim this clause has to carry is "it drove", and a car that
    // covers fifteen metres in five seconds is driving on anybody's hillside.
    //
    // # RE-BLESSED AT WAVE VEH2a, and here is what moved
    //
    // Nothing about this arm's *purpose* changed: it is still the one place that
    // says the editor's Simulate and the shipped player step a vehicle to the
    // same bytes. What changed is the model underneath it, so the numbers it
    // prints are published rather than quietly replaced:
    //
    // | | VEH1a | VEH2a | why |
    // |---|---|---|---|
    // | distance in 300 steps | 34.1 m | **27.0 m** | the settlement now parks a row from the grown fleet rather than always the saloon, and this one is a working vehicle |
    // | top speed | 16.81 m/s | **7.65 m/s** | same, plus a real torque curve that has to rev through a real gearbox instead of a flat force available at once |
    // | wheel contacts | 1 018 of 1 200 | **1 200 of 1 200** | the suspension keeps every wheel down for the whole drive now; VEH1a's model bounced one off the ground for a sixth of it |
    // | peak revs | 0.4944 | **0.9269** | revs are the engine's own rpm between idle and the redline, not road speed against a top speed |
    //
    // The contact count is the one worth reading twice: it is not a tuning
    // artefact, it is 182 wheel-steps that used to be spent in the air.
    assert!(
        travelled > 15.0,
        "the car covered {travelled} m in five seconds of throttle — that is a \
         parked car, and every comparison below is about nothing"
    );
    assert!(top > 5.0, "the car's best forward speed was {top} m/s");
    assert!(
        grounded > steps * 2,
        "{grounded} wheel contacts over {steps} steps (of a possible {}) — the \
         car spent the drive in the air, so the suspension was never solved",
        steps * 4
    );
    // The ENGINE spoke, and it spoke every step: the loop is one `Play` and a
    // pitch/volume pair per step per car, which is what makes the stream a pure
    // function of sim state rather than an event somebody remembered to fire.
    assert_eq!(
        a.audio.0, 1,
        "the engine loop queued {} `Play`s for one car — a voice is started \
         ONCE and then addressed, or the clip restarts sixty times a second",
        a.audio.0
    );
    // …and the count is ARITHMETIC, not a floor (VEH1a audit). The window opens
    // before the hero has even walked to the car, so it also holds the settle
    // step, the twenty-four the hero spends standing beside it and the
    // press/release pairs the interact edge needs — `pre_steps`, counted rather
    // than assumed, because the fixture's ground is sampled and the number of
    // enter attempts is a fact about the machine.
    //
    // An equality rather than `>= steps` because equality is the claim: ONE car
    // is being addressed, on EVERY step it is in an outcome, exactly once each.
    // `>=` passed a second resident car paging in halfway and doubling the
    // stream, and it also passed the 334 in the ledger without anybody being
    // able to derive it.
    assert_eq!(
        (a.audio.1, a.audio.2),
        (a.pre_steps + steps, a.pre_steps + steps),
        "the engine loop queued {:?} (Play, SetPitch, SetVolume) over \
         {} pre-drive steps plus {steps} driving ones — a loop that is a pure \
         function of sim state emits exactly one pair per car per step it is \
         published on",
        a.audio,
        a.pre_steps
    );
    assert_eq!(
        a.audio, b.audio,
        "the two hosts queued different engine audio for the same drive"
    );
    // …and the pitch really MOVED: a constant cue would satisfy every count.
    let (rlo, rhi) = a
        .revs
        .iter()
        .fold((f64::MAX, f64::MIN), |(l, h), v| (l.min(*v), h.max(*v)));
    println!(
        "THE ENGINE: revs {rlo:.4} to {rhi:.4}, load {:.2}",
        a.load[10]
    );
    assert!(
        rhi - rlo > 0.05,
        "the engine's revs ran {rlo}..{rhi} over the drive — the cue is a \
         constant, so `SetPitch` is carrying no information"
    );

    // ── ANTI-VACUITY ── the states are not one state repeated.
    let distinct: std::collections::BTreeSet<&Vec<u8>> = a.states.iter().collect();
    assert!(
        distinct.len() > steps / 2,
        "only {} of {steps} drive states are distinct",
        distinct.len()
    );

    // ── THE HEADLINE ── byte for byte, step for step.
    assert_eq!(a.states.len(), steps);
    assert_eq!(b.states.len(), steps);
    for (i, (x, y)) in a.states.iter().zip(&b.states).enumerate() {
        assert_eq!(
            x, y,
            "PIE and shipping diverged at step {i} of {DRIVE_STEPS} of a DRIVE — \
             the vehicle step, the ground its wheels are cast into, or the \
             streaming that paged that ground is a function of something other \
             than sim state"
        );
    }
    // …and their cars agree about what they did, which is the half a byte
    // comparison of the whole world can hide behind a big number.
    for i in 0..steps {
        assert_eq!(a.wheels[i], b.wheels[i], "wheels grounded at step {i}");
        assert_eq!(
            a.speed[i].to_bits(),
            b.speed[i].to_bits(),
            "forward speed at step {i}"
        );
    }

    // ── WHAT THE DRIVE STREAMED ── measured, because the claim was made
    //    (VEH1a audit). This arm's own header called itself *"a far sharper
    //    instrument for the streaming claim"* on the argument that four wheel
    //    rays read whatever ground has paged in — which is true of the
    //    MECHANISM and says nothing about whether this 27-metre, five-second
    //    window pages anything at all. The scripted arm it joins covers 360 m
    //    and asserts its counters; this one has to print its own or the
    //    sentence is prose ahead of the arm.
    //
    //    The counters are `(cell activations, cell deactivations, cells
    //    resident, sim-resident terrain tiles, terrain loads)` and **none of
    //    them is in `state_bytes`** — which is what makes the equality below a
    //    claim the byte compare cannot make, and the same argument the scripted
    //    arm gives for asserting them at all.
    let (ship1, pie1) = (streaming_counters(&ship), streaming_counters(&pie));
    println!(
        "THE DRIVE'S STREAMING (activations, deactivations, cells, sim tiles, \
         terrain loads):\n  shipping {ship0:?} -> {ship1:?}\n  PIE      {pie0:?} \
         -> {pie1:?}"
    );
    assert_eq!(
        ship1, pie1,
        "the two hosts' streamers disagree after the same drive — and none of \
         these counters reaches `state_bytes`, so the byte comparison above is \
         blind to it"
    );
    assert_eq!(
        ship0, pie0,
        "the two hosts had paged differently before anybody moved"
    );
    // …and the honest reading of the deltas. A five-second drive over 27 m is
    // NOT the scripted arm's 360 m out-and-back, and this says which of the two
    // claims it can carry: the residency it inherited, or a page that arrived
    // under a moving car.
    let paged = ship1.0 > ship0.0 || ship1.1 > ship0.1 || ship1.4 > ship0.4;
    println!(
        "THE DRIVE'S STREAMING: the car {} over its {travelled:.0}-metre window — the \
         PAGING half of the streaming claim is the scripted 360 m arm's \
         (`pie_equals_shipping_on_an_island_drive`), and what THIS arm adds is \
         that the two hosts' streamers agree exactly while a simulated body \
         reads the ground they hold",
        if paged {
            "moved a page"
        } else {
            "paged nothing new"
        }
    );
    // Whatever it paged, it must be standing on ground that is really resident:
    // a drive over zero sim-resident terrain tiles is a car falling through a
    // world neither host has.
    assert!(
        ship1.3 > 0 && pie1.3 > 0,
        "the drive ran with {} / {} sim-resident terrain tiles — the wheels were \
         casting into nothing",
        ship1.3,
        pie1.3
    );

    // ── THE SINK, ON THE REAL CIRCUIT ── the ~2 cm clause 1 priced on a
    //    fixture, named where the road actually is. Roads carry NO colliders
    //    (IB-4's priced ruling), so a wheel rides the terrain heightfield and
    //    the tarmac it looks like it is on is the ribbon, drawn at
    //    `DEFAULT_ROAD_LIFT_M` above the ground it drapes on.
    let lift = inf_island::DEFAULT_ROAD_LIFT_M;
    println!(
        "THE SINK: the road ribbon is drawn {lift} m above the heightfield the \
         wheels are cast into, so a wheel rides that far under the visible \
         tarmac at every ribbon vertex, plus whatever the cross-section's own \
         chord adds between them"
    );
    assert!(
        lift > 0.0 && lift < 0.05,
        "the road lift is {lift} m — the sink this wave priced at two \
         centimetres is a different number now"
    );
}

/// **EVERY SITE IS REACHABLE FROM EVERY OTHER** — wave I7's open item 16, over
/// the graph the road builder actually builds.
///
/// I7's own words: *"on the real island the same planner produced 11 links and
/// 7 junctions, which is consistent with it and is not the same as an assertion
/// that every site is reachable from every other. A connectivity walk over the
/// built `RoadGraph` is what would close it."*
///
/// So the walk is over a **real `RoadGraph`** (`inf_island::road_graph`), built by
/// `RoadGraph::from_layer` from the committed roads layer — the same function
/// `inf_island::roads::build_mesh` builds one with, on the same bytes — and
/// routed with `inf_nav::route` over the graph's own `nav_graph()`. Not over the
/// route list, which is the *input* to the junction derivation and cannot
/// falsify it: two roads that cross without sharing an endpoint are two routes
/// and one junction-free graph, and that is exactly the defect a topology claim
/// has to be able to see.
///
/// Run against **both** committed islands, because a fixture that agrees with
/// itself proves nothing about the island that ships.
#[test]
fn every_settlement_is_reachable_from_every_other_over_the_built_road_graph() {
    for rel in [
        "../../samples/island-fixture/island.toml",
        "../../samples/island/island.toml",
    ] {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(rel);
        let Ok(recipe) = inf_island::IslandRecipe::load(&path) else {
            println!("SKIP: no {rel} in this tree");
            continue;
        };
        let anchor = inf_island::read_design(&recipe)
            .expect("the design reads")
            .anchor;
        // Through `inf_island::road_graph`, which resolves the layer path off
        // the recipe exactly as `read_design` does and builds the graph with the
        // same `RoadGraph::from_layer` the island's own mesh builder uses. Not
        // by naming `inf_gis` here: that crate's transcendental exemption is
        // kept by a manifest ban this file's crate is subject to.
        let graph = inf_island::road_graph(&recipe, &anchor).expect("the roads layer reads");
        let nav = graph.nav_graph();
        println!(
            "{}: {} segments, {} intersections, {} junctions, {:.2} km; nav has \
             {} nodes",
            recipe.name,
            graph.segments.len(),
            graph.intersections.len(),
            graph.junctions().count(),
            graph.total_length_m() / 1000.0,
            nav.nodes().count()
        );
        assert!(
            !graph.segments.is_empty() && nav.nodes().count() > 1,
            "{}: the built graph has nothing in it, so the walk below is vacuous",
            recipe.name
        );

        // Each site's own entry point: the nearest node of the graph to it.
        let sites: Vec<(String, inf_nav::NavNodeId)> = recipe
            .sites
            .iter()
            .filter_map(|s| {
                nav.nearest_planar(glam::DVec3::new(s.x, 0.0, s.z), f64::INFINITY)
                    .map(|n| (s.name.clone(), n))
            })
            .collect();
        assert_eq!(
            sites.len(),
            recipe.sites.len(),
            "{}: a site found no road node at all",
            recipe.name
        );

        // THE WALK: every ordered pair.
        let mut worst = 0.0f64;
        let mut pairs = 0usize;
        for (from_name, from) in &sites {
            for (to_name, to) in &sites {
                if from == to {
                    continue;
                }
                pairs += 1;
                let verdict = inf_nav::route(&nav, *from, *to);
                let r = match verdict {
                    inf_nav::NavVerdict::Found(r) => r,
                    other => panic!(
                        "{}: {from_name} cannot reach {to_name} over the built \
                         road graph — {other:?}. The circuit is not connected, \
                         which is what wave I7 open item 16 could not tell.",
                        recipe.name
                    ),
                };
                worst = worst.max(r.cost_m);
            }
        }
        assert!(
            pairs >= recipe.sites.len() * (recipe.sites.len() - 1),
            "{}: only {pairs} ordered pairs walked",
            recipe.name
        );
        println!(
            "{}: {pairs} ordered site pairs, ALL reachable; the longest is \
             {:.2} km of road",
            recipe.name,
            worst / 1000.0
        );
        // A route that costs nothing is two sites welded onto one node, which
        // would make "reachable" true and meaningless.
        assert!(
            worst > 100.0,
            "{}: the longest route between any two sites is {worst} m",
            recipe.name
        );
    }
}

/// **WHAT A CAR COSTS THE ISLAND'S STEP** (island wave VEH1a) — the `vehicle`
/// phase's first numbers on real content, and where `VEHICLE_STEP_BUDGET_MS` is
/// asserted.
///
/// The phase is new, so the whole-step total cannot see it: `CITY_STEP_BUDGET_MS`
/// is one number with an island inside it, and a row that appeared this wave
/// would have to grow to a visible share of six milliseconds before it moved.
/// That is [`NPC_STEP_BUDGET_MS`]'s argument verbatim and it is why this phase
/// carries its own ceiling.
///
/// # A clock, so: release only, real machine only
///
/// `CITY_STEP_BUDGET_MS`'s conditioning, for its reasons — `[profile.dev]` is
/// `opt-level = 1` with debug assertions, and a shared CI runner's milliseconds
/// are a fact about the runner. Reported everywhere; asserted under
/// `cargo test --release` off CI.
///
/// [`NPC_STEP_BUDGET_MS`]: inf_player::budget::NPC_STEP_BUDGET_MS
#[test]
fn the_vehicle_phase_costs_what_it_costs_on_the_island() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let pack = cook(tmp.path());
    let mut sim = pack_sim(&pack);
    sim.set_step_profiling(true);
    let trace = drive_a_car(&mut sim);
    assert!(
        trace.entered,
        "the hero did not get in, so this is a profile of a parked car"
    );

    // MIN of rounds is the discipline everywhere else in this tree; a step
    // profile is one step, so the mean over the drive's own last quarter is what
    // stands in for it — by then the ground has paged and the suspension has
    // settled, which is the state a shipped frame is in.
    let mut mean = inf_player::step_profile::StepProfile::default();
    let rounds = 90u32;
    for _ in 0..rounds {
        sim.step_once(
            inf_player::runtime_sim::RuntimeInput::default()
                .axis_at(inf_ecs::movement::actions::MOVE_Y, 1.0),
        );
        mean.accumulate(&sim.step_profile());
    }
    mean.scale(1.0 / rounds as f64);
    let vehicle = mean.ms[inf_player::step_profile::STEP_PHASE_NAMES
        .iter()
        .position(|n| *n == "vehicle")
        .expect("the `vehicle` phase exists")];

    println!(
        "\nTHE ISLAND'S STEP WITH A CAR IN IT ({} build), {:.4} ms total over \
         {rounds} driving steps:",
        if cfg!(debug_assertions) {
            "dev"
        } else {
            "release"
        },
        mean.total_ms()
    );
    for (name, ms) in mean.dearest_first() {
        if ms > 0.0005 {
            println!("  {name:>18}  {ms:.4} ms");
        }
    }
    println!(
        "  the `vehicle` row: {vehicle:.4} ms for {} car(s) against a \
         {:.1} ms budget at {} cars",
        cars(&sim).len(),
        inf_player::budget::VEHICLE_STEP_BUDGET_MS,
        inf_player::budget::VEHICLE_BUDGET_CARS
    );

    // The phase really ran: a budget met by a door that returned early is a
    // budget about nothing.
    assert!(
        !sim.vehicles().is_empty(),
        "the `vehicle` phase reported no vehicle at all on a level that parks \
         one at every settlement"
    );
    if cfg!(debug_assertions) {
        eprintln!("dev build: the vehicle phase is reported, not asserted");
        return;
    }
    if std::env::var_os("CI").is_some() {
        eprintln!("CI: the vehicle phase is reported, not asserted (shared runner)");
        return;
    }
    assert!(
        vehicle <= inf_player::budget::VEHICLE_STEP_BUDGET_MS,
        "the `vehicle` phase cost {vehicle:.4} ms against a {} ms ceiling {}",
        inf_player::budget::VEHICLE_STEP_BUDGET_MS,
        inf_player::budget::RATCHET_NOTE
    );
}

/// **THE RE-DERIVED GRID IS THE SETTLEMENT'S OWN** (island wave VEH2b).
///
/// `inf_ecs::traffic::streets_of` recovers a street grid from nothing but the
/// block rectangles a committed level carries, because the PLAN that placed
/// those blocks — `inf_editor_core::settlement::Settlement` — is Ring 1 and the
/// shipped player cannot have it. That is the whole design of clause 1b, and
/// until this arm existed it was a claim in a module doc with nothing behind it.
///
/// So the plan is loaded here, where a dev-dependency may see it, and the
/// runtime's answer is held against it: **every line the runtime recovered lies
/// on a line the plan drew**, to a centimetre, and it recovered the interior
/// ones rather than none.
///
/// The inequality is the honest half. The plan draws the outermost line of a
/// settlement too, and that line has ground on one side and no block to bound
/// it — so it is not a gap, the runtime does not find it, and the recovered
/// network is a strict subset. That is stated in `streets_of`'s own doc and it
/// is measured here.
#[test]
fn the_derived_carriageway_is_the_settlements_own_street_grid() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let _proj = build_project(tmp.path());
    let pack = cook(tmp.path());
    let mut sim = pack_sim(&pack);
    let design = inf_island::read_design(
        &inf_island::IslandRecipe::load(&fixture_recipe()).expect("recipe"),
    )
    .expect("design");
    let settlement = walk_target_settlement(&design);
    let centre = glam::DVec3::new(settlement.centre.x, 0.0, settlement.centre.y);
    let hero = hero_entity(&sim).expect("a hero");
    set_hero(&mut sim, hero, centre);
    sim.world_mut().mark_dirty();
    freeze_clock(&mut sim, SETTLE_HOUR);
    let mut peak = [0usize; 4];
    let mut steered = 0u64;
    settle_the_society(&mut sim, &mut peak, &mut steered);

    let derived = inf_ecs::traffic::streets_of(sim.world());
    assert!(
        !derived.is_empty(),
        "the runtime recovered no street at all from a settled settlement"
    );

    // The PLAN's own lines, as the pairs of world-XZ ends it drew.
    let planned = &settlement.streets;
    assert!(
        planned.len() > derived.len(),
        "the runtime recovered {} line(s) of a plan that drew {} — the interior \
         grid is a strict subset and this fixture no longer shows it",
        derived.len(),
        planned.len()
    );

    // **Every recovered line lies on a planned one.** A settlement's grid is
    // axis-aligned, so a line is identified by its own constant coordinate and
    // which axis it runs along.
    let mut worst = 0.0f64;
    for d in &derived {
        let along_x = d.along_x();
        let mine = if along_x { d.a.y } else { d.a.x };
        let best = planned
            .iter()
            .filter(|p| {
                let p_along_x = p.a.y == p.b.y;
                p_along_x == along_x
            })
            .map(|p| {
                let theirs = if along_x { p.a.y } else { p.a.x };
                (theirs - mine).abs()
            })
            .fold(f64::INFINITY, f64::min);
        worst = worst.max(best);
    }
    assert!(
        worst < 0.01,
        "a recovered street line is {worst:.4} m off the nearest line the \
         settlement plan drew — the runtime is recovering a different grid"
    );

    // …and the reserve it recovered is the reserve the plan reserved.
    for d in &derived {
        assert!(
            (d.gap_m - settlement.street_m).abs() < 0.01,
            "a recovered street is {:.3} m wide against the plan's {:.3}",
            d.gap_m,
            settlement.street_m
        );
    }
    println!(
        "VEH2b: the runtime recovered {} of the plan's {} street line(s) at {:.3} m \
         wide, worst offset {worst:.5} m",
        derived.len(),
        planned.len(),
        settlement.street_m
    );
}

/// **WHAT A STREET COSTS THE ISLAND'S STEP** (island wave VEH2b) — the
/// `traffic` phase's first numbers on real content, and where
/// `TRAFFIC_STEP_BUDGET_MS` is asserted.
///
/// The vehicle arm above profiles ONE car with a hero in it. This one profiles
/// the settlement: the derivation, the band, the tier decision for every record
/// and the steering for the ones that are rigs — at the busiest hour, with the
/// hero standing in the middle of it.
///
/// # A clock, so: release only, real machine only
///
/// `CITY_STEP_BUDGET_MS`'s conditioning, for its reasons.
#[test]
fn the_traffic_phase_costs_what_it_costs_on_the_island() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let _proj = build_project(tmp.path());
    let pack = cook(tmp.path());
    let mut sim = pack_sim(&pack);
    let design = inf_island::read_design(
        &inf_island::IslandRecipe::load(&fixture_recipe()).expect("recipe"),
    )
    .expect("design");
    let settlement = walk_target_settlement(&design);
    let centre = glam::DVec3::new(settlement.centre.x, 0.0, settlement.centre.y);
    let hero = hero_entity(&sim).expect("a hero");
    set_hero(&mut sim, hero, centre);
    sim.world_mut().mark_dirty();
    freeze_clock(&mut sim, SETTLE_HOUR);
    let mut peak = [0usize; 4];
    let mut steered = 0u64;
    settle_the_society(&mut sim, &mut peak, &mut steered);
    freeze_clock(&mut sim, RUSH_HOUR);
    for _ in 0..200 {
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
    }
    sim.set_step_profiling(true);

    // MIN of three rounds, the discipline everywhere else in this tree; each
    // round is the mean over sixty settled steps, because a step profile is one
    // step and a single one of them is a fact about a scheduler.
    let rounds = 3u32;
    let per_round = 60u32;
    let mut best: Option<inf_player::step_profile::StepProfile> = None;
    for _ in 0..rounds {
        let mut mean = inf_player::step_profile::StepProfile::default();
        for _ in 0..per_round {
            sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
            mean.accumulate(&sim.step_profile());
        }
        mean.scale(1.0 / f64::from(per_round));
        let take = match &best {
            None => true,
            Some(b) => mean.total_ms() < b.total_ms(),
        };
        if take {
            best = Some(mean);
        }
    }
    let mean = best.expect("three rounds");
    let idx = |name: &str| {
        inf_player::step_profile::STEP_PHASE_NAMES
            .iter()
            .position(|n| *n == name)
            .unwrap_or_else(|| panic!("the `{name}` phase exists"))
    };
    let row = |name: &str| mean.ms[idx(name)];
    let traffic = row("traffic");
    let st = sim.traffic_stats();

    // ── THE CONTROL. The same settlement, the same hour, the same hero, with
    //    the traffic taken out -- because a phase table with a new row in it
    //    prices nothing unless the alternative is priced too (the I1 audit law).
    //    `set_traffic` with an empty population takes every body down AND stops
    //    the derivation (`hand_installed`), so what is measured is a street the
    //    engine is not putting cars on rather than a street it has not got to
    //    yet.
    sim.set_traffic_population(Default::default());
    for _ in 0..90 {
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
    }
    let mut control: Option<inf_player::step_profile::StepProfile> = None;
    for _ in 0..rounds {
        let mut m = inf_player::step_profile::StepProfile::default();
        for _ in 0..per_round {
            sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
            m.accumulate(&sim.step_profile());
        }
        m.scale(1.0 / f64::from(per_round));
        if control.as_ref().is_none_or(|b| m.total_ms() < b.total_ms()) {
            control = Some(m);
        }
    }
    let control = control.expect("three rounds");
    assert_eq!(
        sim.traffic_stats().cars,
        0,
        "the control still has cars in it, so it is not a control"
    );

    println!(
        "\nTHE ISLAND'S STEP WITH A STREET IN IT ({} build), {:.4} ms total, \
         MIN of {rounds} rounds of {per_round}:",
        if cfg!(debug_assertions) {
            "dev"
        } else {
            "release"
        },
        mean.total_ms()
    );
    for (name, ms) in mean.dearest_first() {
        if ms > 0.0005 {
            println!("  {name:>18}  {ms:.4} ms");
        }
    }
    println!(
        "  the `traffic` row: {traffic:.4} ms for {} car(s) -- tiers {:?}, {} \
         driving, {} with a driver -- against a {:.1} ms budget",
        st.cars,
        st.per_tier,
        st.driving,
        st.drivers,
        inf_player::budget::TRAFFIC_STEP_BUDGET_MS
    );
    println!(
        "  the `vehicle` row: {:.4} ms for {} rig(s) -- {:.2} us a car; the \
         `crowd` row {:.4} ms and `society` {:.4} ms for {} agent(s)",
        row("vehicle"),
        st.per_tier[0],
        row("vehicle") * 1000.0 / st.per_tier[0].max(1) as f64,
        row("crowd"),
        row("society"),
        sim.crowd_stats().per_tier.iter().sum::<usize>()
    );
    println!(
        "\nTHE SAME STEP WITH THE TRAFFIC TAKEN OUT: {:.4} ms total against \
         {:.4} -- traffic {:.4} -> {:.4}, vehicle {:.4} -> {:.4}, solver \
         {:.4} -> {:.4}, physics3d sync {:.4} -> {:.4}",
        control.total_ms(),
        mean.total_ms(),
        traffic,
        control.ms[idx("traffic")],
        row("vehicle"),
        control.ms[idx("vehicle")],
        row("solver"),
        control.ms[idx("solver")],
        row("physics3d sync"),
        control.ms[idx("physics3d sync")],
    );
    println!(
        "  THE WHOLE COST OF A STREET: {:.4} ms of a {:.4} ms step -- {:.1} %",
        mean.total_ms() - control.total_ms(),
        mean.total_ms(),
        100.0 * (mean.total_ms() - control.total_ms()) / mean.total_ms()
    );

    // The phase really ran: a budget met by a door that returned early is a
    // budget about nothing.
    assert!(
        st.cars > 0,
        "the `traffic` phase reported no car at all on a settlement whose kerbs \
         it derives"
    );
    assert!(
        st.per_tier[0] > 0,
        "no car is a rig, so the dearest half of this phase never ran"
    );
    if cfg!(debug_assertions) {
        eprintln!("dev build: the traffic phase is reported, not asserted");
        return;
    }
    if std::env::var_os("CI").is_some() {
        eprintln!("CI: the traffic phase is reported, not asserted (shared runner)");
        return;
    }
    assert!(
        traffic <= inf_player::budget::TRAFFIC_STEP_BUDGET_MS,
        "the `traffic` phase cost {traffic:.4} ms against a {} ms ceiling {}",
        inf_player::budget::TRAFFIC_STEP_BUDGET_MS,
        inf_player::budget::RATCHET_NOTE
    );
}

// ── VEH2b: RUSH HOUR, WITH CARS ─────────────────────────────────────────────
//
// The wave's gate. NPC1d proved a town WALKS; VEH2a proved one car DRIVES. This
// is the arm that says the two happen at once, on committed content, on both
// hosts, byte for byte — and that the player can walk up to one of the cars in
// it, pull the driver out and take it.
//
// Everything it asserts is a count with a floor on it, because a rush-hour gate
// that passes with zero cars is a rush-hour gate that certifies nothing.

/// How many steps the rush-hour window runs on each host.
///
/// Six hundred is ten seconds at 60 Hz, which is long enough for a `Full`-tier
/// traffic car to cover a city block at town speed and for the carjack's own
/// press/release edges to land inside it.
const RUSH_STEPS: usize = 600;

/// The local hour the gate freezes the island's day at for the window.
///
/// Twenty-four minutes past eight: `inf_ecs::society::WORK_START_H` is 8 and
/// `COMMUTE_H` is 1, and `CrowdRecord::hour_of` shifts each agent's own day by
/// up to `SCHEDULE_JITTER_H` either way -- so at 8.4 the great majority of cars
/// with a commute are inside their morning leg rather than either side of it.
/// **Frozen** (`rate = 0`), because a census of a rush hour has to be a census
/// of one hour rather than of however far a compressed clock got.
const RUSH_HOUR: f64 = 8.4;

/// The hour the gate SETTLES at, before it winds on to the rush.
///
/// Noon, and it is load-bearing rather than arbitrary. A `Full`-tier traffic car
/// is steered along its whole route by `drive_intent` -- the controller follows
/// the PATH, not the clock -- so any long phase that runs at the rush hour
/// consumes the drive before the window opens. The first cut settled at
/// whatever hour the level carried, and by the time the window started every
/// commuter had **arrived**: measured at `s_m` 176 m of a 176 m route,
/// `remaining_m` 0.0, holding the handbrake. Nothing was broken; the instrument
/// was late.
///
/// At noon a commuter's morning leg is finished on the CLOCK, so
/// `TrafficRecord::is_driving` is false, the car holds the handbrake at its own
/// kerb and nothing is consumed. The society is at work, which is the quiet the
/// settle wants anyway.
const SETTLE_HOUR: f64 = 12.0;

/// How long the gate holds the throttle down in the car it stole.
///
/// Ten seconds. What it asserts on is **the player's own throttle reaching the
/// car** (`stolen_throttle`) rather than a distance, because the fixture's town
/// is two streets wide with three hundred and twenty-nine residents walking
/// across them and traffic in this engine YIELDS to anything in its lane. A
/// distance floor on that street would be a measurement of the jam.
///
/// **`audit:` VEH2b — this doc used to say the arm asserts "revs off idle and a
/// top speed", and it does not.** Both are collected and PRINTED, and on the
/// fixture both read 0.00: the car is walled in by kinematic capsules so
/// `forward_mps` never leaves zero, and this engine's `engine_state` has no idle
/// floor. That is a VEH2a property of the model rather than anything this wave
/// did, and the sentence is corrected instead of the arm being weakened.
const STOLEN_STEPS: usize = 600;

/// What one host's rush hour looked like.
struct RushRun {
    digests: Vec<u64>,
    distinct: usize,
    /// The carriageway the level re-derived for itself: street lines, lanes,
    /// and how many times the derivation actually ran.
    streets: usize,
    lanes: usize,
    derivations: u64,
    /// The traffic census at the end of the window.
    traffic: inf_ecs::traffic::TrafficStats,
    /// The tiers a kilometre away, and at the town's edge -- the ladder's own
    /// two ends, read off the same population.
    away_tiers: [usize; 4],
    edge_tiers: [usize; 4],
    /// …and the crowd's, so "the society walks AND drives" is two numbers rather
    /// than a sentence.
    crowd: inf_ecs::crowd::CrowdStats,
    /// Steering intents written to walking agents over the window.
    walked: u64,
    /// The furthest any one traffic car travelled over the window, metres.
    car_moved_m: f64,
    /// The carjack: whether the driver came out, whether the hero got in, how
    /// many presses it took, and how far the stolen car was then driven.
    jacked: bool,
    seated: bool,
    presses: usize,
    stolen_m: f64,
    /// The highest revs and the top speed the stolen car reached under the
    /// player's own throttle.
    stolen_revs: f64,
    stolen_top_mps: f64,
    /// The throttle the SEAT STEP wrote into the car from the player's stick --
    /// `VehicleOutcome::load`. This is the arm's real claim; see its assertion.
    stolen_throttle: f64,
    /// Where the victim ended up, relative to the car it was pulled out of.
    victim_away_m: f64,
    /// **`audit:` VEH2b — the steps of the STEAL, digested.**
    ///
    /// `digests` above covers the six hundred steps of the rush hour, which end
    /// before the hero has touched anything. Without this the wave's own
    /// headline — *the player pulls a driver out and takes their car,
    /// byte-identical on both hosts* — was compared as three booleans and a
    /// press count: the seat swap, the ejection pose, the victim's adopted
    /// route and ten seconds of driving the stolen car were in no equality at
    /// all. These are the same `state_bytes` digest over the approach, every
    /// press and the whole stolen drive.
    jack_digests: Vec<u64>,
}

/// Freeze this host's day at `hour`, local.
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

/// Run one host's rush hour, and steal a car out of it.
fn rush_hour(sim: &mut RuntimeSim, centre: glam::DVec3) -> RushRun {
    let hero = hero_entity(sim).expect("a hero");
    set_hero(sim, hero, centre);
    sim.world_mut().mark_dirty();
    // Settle at NOON -- see `SETTLE_HOUR`. The clock is frozen first, so no
    // commuter spends the settle driving the route the window is about.
    freeze_clock(sim, SETTLE_HOUR);
    let mut peak = [0usize; 4];
    let mut steered = 0u64;
    settle_the_society(sim, &mut peak, &mut steered);

    // Let the traffic's own plan queue drain. It is bounded
    // (`TRAFFIC_PLANS_PER_STEP` a step, `MAX_COMMUTERS` in total), so this
    // terminates or the arm says which number it stopped at.
    let mut drained = 0usize;
    for i in 0..2_000 {
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
        if sim.traffic_stats().pending == 0 && i > 8 {
            drained = i + 1;
            break;
        }
    }
    assert!(
        drained > 0,
        "the traffic plan queue never drained: {} still pending",
        sim.traffic_stats().pending
    );

    // -- the ladder, before the window. The fixture's whole town is inside the
    //    64 m `Full` ring from its own crossroads, so standing there and
    //    counting rungs measures the fixture rather than the band. Walk the hero
    //    out and back: away, everything is `Dormant`; at the town's edge, the
    //    ladder has cars on both of the rungs a car has.
    set_hero(sim, hero, centre + glam::DVec3::new(TOWN_AWAY_M, 0.0, 0.0));
    sim.world_mut().mark_dirty();
    for _ in 0..20 {
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
    }
    let away_tiers = sim.traffic_stats().per_tier;
    set_hero(sim, hero, centre + glam::DVec3::new(TOWN_EDGE_M, 0.0, 0.0));
    sim.world_mut().mark_dirty();
    for _ in 0..20 {
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
    }
    let edge_tiers = sim.traffic_stats().per_tier;
    set_hero(sim, hero, centre);
    sim.world_mut().mark_dirty();
    for _ in 0..20 {
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
    }

    // -- and NOW the rush hour, so the window is the drive rather than its
    //    aftermath.
    freeze_clock(sim, RUSH_HOUR);

    let res = inf_ecs::traffic::carriageway_of(sim.world()).expect("a carriageway");
    let (streets, lanes, derivations) = (res.streets.len(), res.lanes.len(), res.derivations);

    // ── the window ──────────────────────────────────────────────────────────
    let start: std::collections::BTreeMap<uuid::Uuid, glam::DVec3> =
        inf_physics::d3::traffic::records(sim.world())
            .into_iter()
            .map(|(g, r)| (g, r.last))
            .collect();
    let mut digests = Vec::with_capacity(RUSH_STEPS);
    let mut seen: std::collections::BTreeSet<u64> = Default::default();
    let mut walked = 0u64;
    for _ in 0..RUSH_STEPS {
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
        let d = digest(&sim.state_bytes());
        seen.insert(d);
        digests.push(d);
        walked += sim.crowd_stats().steered;
    }
    let traffic = sim.traffic_stats();
    let crowd = sim.crowd_stats();
    let car_moved_m = inf_physics::d3::traffic::records(sim.world())
        .into_iter()
        .filter_map(|(g, r)| start.get(&g).map(|s| (r.last - *s).length()))
        .fold(0.0f64, f64::max);

    // ── the carjack ─────────────────────────────────────────────────────────
    //
    // The target is the LOWEST-GUID car that currently has somebody at the
    // wheel, so both hosts pick the same one out of the same world rather than
    // out of an index.
    // **The most ISOLATED driven car**, not the first one. The interact rule is
    // nearest-wins over every candidate, so a gate that stood at the door of a
    // car with a parked one a metre and a half behind it measures the ranking
    // rather than the carjack -- which is exactly what the first cut did
    // (resolve answered Enter on a neighbour at 1.65 m against the target's
    // 2.02). Deterministic, and a function of the world.
    let seats: Vec<(uuid::Uuid, glam::DVec3)> = cars(sim)
        .into_iter()
        .filter_map(|g| {
            inf_physics::d3::vehicle::seat_pose(sim.bridge3d(), g).map(|(p, _, _)| (g, p))
        })
        .collect();
    let target = inf_physics::d3::traffic::records(sim.world())
        .into_iter()
        .filter(|(g, r)| {
            r.tier == inf_ecs::crowd::CrowdTier::Full
                && inf_physics::d3::carjack::occupant_of(sim.world(), *g).is_some()
        })
        .filter_map(|(g, _)| {
            let mine = seats.iter().find(|(h, _)| *h == g)?.1;
            let nearest = seats
                .iter()
                .filter(|(h, _)| *h != g)
                .map(|(_, p)| (*p - mine).length())
                .fold(f64::INFINITY, f64::min);
            Some((g, nearest))
        })
        .max_by(|a, b| a.1.total_cmp(&b.1).then(b.0.cmp(&a.0)))
        .map(|(g, _)| g);
    let mut run = RushRun {
        digests,
        distinct: seen.len(),
        streets,
        lanes,
        derivations,
        traffic,
        away_tiers,
        edge_tiers,
        crowd,
        walked,
        car_moved_m,
        jacked: false,
        seated: false,
        presses: 0,
        stolen_m: 0.0,
        stolen_revs: 0.0,
        stolen_top_mps: 0.0,
        stolen_throttle: 0.0,
        victim_away_m: 0.0,
        jack_digests: Vec::new(),
    };
    let Some(chassis) = target else {
        return run;
    };
    let victim = inf_physics::d3::carjack::occupant_of(sim.world(), chassis).expect("checked");

    // Stand at the DRIVER'S door — the `+X` side, which is the side the exit
    // puts a driver out on and the only side the carjack candidate is offered
    // from.
    let (seat, rot, _) =
        inf_physics::d3::vehicle::seat_pose(sim.bridge3d(), chassis).expect("a seat");
    let beside = seat + (rot * glam::DVec3::X) * 1.7 - glam::DVec3::Y * 0.2;
    for _ in 0..24 {
        set_hero(sim, hero, beside);
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
        run.jack_digests.push(digest(&sim.state_bytes()));
    }
    // Press E, release, press again — `just_pressed` is an edge, and the
    // carjack's resist draw is a function of the step, so a real player presses
    // more than once.
    for _ in 0..24 {
        sim.step_once(
            inf_player::runtime_sim::RuntimeInput::default()
                .press(inf_ecs::movement::actions::INTERACT),
        );
        run.jack_digests.push(digest(&sim.state_bytes()));
        run.presses += 1;
        if inf_physics::d3::carjack::occupant_of(sim.world(), chassis) != Some(victim) {
            run.jacked = true;
        }
        run.seated = sim
            .world()
            .world()
            .get::<inf_ecs::components::CharacterMovement>(hero)
            .is_some_and(|m| {
                m.mode == inf_ecs::components::MovementMode::Driving
                    && m.runtime.seat.vehicle == chassis
            });
        if run.seated {
            break;
        }
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
        run.jack_digests.push(digest(&sim.state_bytes()));
    }
    let took = chassis_at(sim, chassis);
    // ...and drive it away. The throttle held, through the shipped input door,
    // exactly as `drive_a_car` holds it.
    for _ in 0..STOLEN_STEPS {
        sim.step_once(
            inf_player::runtime_sim::RuntimeInput::default()
                .axis_at(inf_ecs::movement::actions::MOVE_Y, 1.0),
        );
        run.jack_digests.push(digest(&sim.state_bytes()));
        if let Some(o) = sim.vehicles().iter().find(|o| o.chassis == chassis) {
            run.stolen_revs = run.stolen_revs.max(o.revs);
            run.stolen_top_mps = run.stolen_top_mps.max(o.forward_mps);
            // `VehicleOutcome::load` IS `controls.throttle.abs()` — the number
            // the seat step wrote from the player's own stick. It is the claim
            // this arm can make on a street this crowded; see below.
            run.stolen_throttle = run.stolen_throttle.max(o.load);
        }
    }
    run.stolen_m = (chassis_at(sim, chassis) - took).length();
    run.victim_away_m = sim
        .world()
        .entity_of(victim)
        .and_then(|e| sim.world().world().get::<inf_ecs::components::Transform>(e))
        .map(|t| (t.translation.to_dvec3() - seat).length())
        .unwrap_or(0.0);
    run
}

// ── wave VEN1a: the venue ────────────────────────────────────────────────────

/// How many steps the venue arm runs its traces for, after the town has
/// settled and the clock has been walked to closing time.
const VENUE_STEPS: usize = 240;

/// The hour the venue gate reads the town at. Twenty-two hundred is when a
/// bar is a bar; it is also well past `night_glow_step`'s ramp, so the window
/// glow is at full and the emissive path is carrying its whole load.
const VENUE_HOUR: f64 = 22.0;

/// What one host saw inside the venue.
#[derive(Debug, Default)]
struct VenueRun {
    /// Per-step sim digests over `VENUE_STEPS`.
    digests: Vec<u64>,
    /// Distinct digests — the anti-vacuity counter.
    distinct: usize,
    /// The hour the trace was taken at.
    hour: f64,
    /// Resident `PcgVolume`s, and instances in them.
    volumes: usize,
    instances: usize,
    /// Instances carrying an authored, *coloured* emission (neon, screens,
    /// festoons) — the thing the venue wave added.
    emitters: usize,
    /// The most saturated emitter's brightest channel.
    peak_emissive: f32,
    /// Instances that pulse.
    pulsing: usize,
    /// Instances whose surface is metal — the chrome pole and the brass stool.
    metal: usize,
    /// Real lights the rigs hang, and how many of them are spots.
    fixtures: usize,
    spots: usize,
    /// Interior nav nodes reachable in the resident volumes.
    nav_nodes: usize,
    /// Errand slots — somewhere the town can be sent at night.
    errands: usize,
    /// What the PROJECTOR made of it: lights in the frame, scatter batches, and
    /// batches carrying emission.
    frame_lights: usize,
    frame_batches: usize,
    frame_emissive_batches: usize,
}

/// **THE VEN1a GATE.** A settlement at ten at night, with the player standing
/// in its venue: the neon is lit and coloured, the rig is hung and aimed, the
/// chrome is metal, the interior is walkable — identically on both hosts, byte
/// for byte, over 240 fixed steps.
///
/// Coverage first, then the comparison (the NPC1d law: a gate staged where a
/// defect cannot appear certifies the defect). Two empty streets agree
/// perfectly, and so do two venues with nothing in them, so every claim below
/// is a count with a floor on it.
///
/// **The fixture's venue is a BAR**, and that is measured rather than assumed:
/// `settlement::tests::the_nightlife_strip_is_one_per_settlement_and_three_kinds
/// _per_city` prints the strip for both committed recipes, and the four-block
/// `Fixture Town` correctly gets none while `Fixture Camp` gets a bar. The
/// shipped island's cities get all three kinds; what CI can afford to build and
/// cook twenty-odd times is the fixture.
#[test]
fn pie_equals_shipping_inside_a_venue_at_night() {
    let tmp = tempfile::tempdir().expect("a temp dir");
    let proj = build_project(tmp.path());
    let content = proj.join("Content");
    let pack = cook(tmp.path());
    let recipe =
        inf_island::IslandRecipe::load(&fixture_recipe()).expect("the fixture recipe loads");
    let slug = inf_island::slug(&recipe.name);
    let design = inf_island::read_design(&recipe).expect("the design reads");

    // The venue block, from the same generator the level was built with.
    let plans = inf_editor_core::settlement::settlements(&design);
    let venue = plans
        .iter()
        .flat_map(|p| p.blocks.iter())
        .find(|b| b.archetype.is_venue())
        .copied()
        .expect("the fixture places at least one venue");
    println!(
        "VEN1a: standing in a {} at ({:.0}, {:.0})",
        venue.archetype.name(),
        venue.centre.x,
        venue.centre.y
    );

    let mut hosts: Vec<(&str, RuntimeSim)> = vec![
        ("shipping", pack_sim(&pack)),
        ("PIE", loose_sim(&content, &slug)),
    ];
    let mut runs: Vec<(&str, VenueRun)> = Vec::new();

    for (label, sim) in hosts.iter_mut() {
        let hero = hero_entity(sim).expect("a hero");
        let centre = glam::DVec3::new(venue.centre.x, 0.0, venue.centre.y);
        set_hero(sim, hero, centre);
        sim.world_mut().mark_dirty();
        let mut tier_peak = [0usize; 4];
        let mut steered = 0u64;
        settle_the_society(sim, &mut tier_peak, &mut steered);

        // **Wind the clock to closing time**, on the day-in-the-life arm's own
        // pattern (`GATE_CLOCK_RATE`) and for its reason: the island authors an
        // eighty-minute day, so *walking* to 22:00 is hundreds of thousands of
        // fixed steps a host and this battery has to finish. Measured before it
        // was wound: 2 400 steps reached 10.72 h.
        //
        // `local_hour` is solar time, so the UTC seconds a longitude needs are
        // `hour × 3600 − longitude × 240` — the same arithmetic the day arm
        // winds midnight with.
        {
            let w = sim.world_mut().world_mut();
            let mut q = w.query::<&mut inf_ecs::components::TimeOfDay>();
            let mut wound = 0usize;
            for mut tod in q.iter_mut(w) {
                tod.seconds =
                    (VENUE_HOUR * 3_600.0 - tod.longitude_deg * 240.0).rem_euclid(86_400.0);
                tod.rate = GATE_CLOCK_RATE;
                wound += 1;
            }
            assert!(wound > 0, "{label}: the island carries no clock to wind");
        }
        sim.world_mut().mark_dirty();
        // One step, so every reader of the clock has seen the new hour.
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
        let hour = local_hour(sim);
        assert!(
            (hour - VENUE_HOUR).abs() < 0.5,
            "{label}: the clock reads {hour:.2} h after being wound to {VENUE_HOUR:.1} — \
             this gate is about what a venue looks like at night"
        );
        // …and the night-glow ramp is at FULL, or "at night" is a claim about a
        // number nobody downstream is reading.
        let (step, sun_y) = glow_now(sim);
        assert_eq!(
            step,
            inf_render::NIGHT_GLOW_STEPS,
            "{label}: the glow step is {step} at {hour:.2} h (sun y {sun_y:+.3})"
        );

        let mut r = VenueRun {
            hour,
            ..VenueRun::default()
        };
        {
            let w = sim.world().world();
            for e in w.iter_entities() {
                let Some(v) = e.get::<inf_ecs::components::PcgVolume>() else {
                    continue;
                };
                if v.evaluated.is_empty() && v.lights.is_empty() {
                    continue;
                }
                r.volumes += 1;
                r.instances += v.evaluated.len();
                for i in &v.evaluated {
                    if i.surface.emits() {
                        let e = i.surface.emissive;
                        let hi = e[0].max(e[1]).max(e[2]);
                        let lo = e[0].min(e[1]).min(e[2]);
                        // SATURATED, not merely lit: a warm-white window pane
                        // has glowed since I8b and is not what this wave added.
                        if hi > 1.0 && hi > lo * 2.0 {
                            r.emitters += 1;
                            r.peak_emissive = r.peak_emissive.max(hi);
                        }
                    }
                    if i.surface.pulse_hz > 0.0 {
                        r.pulsing += 1;
                    }
                    if i.surface.metallic > 0.5 {
                        r.metal += 1;
                    }
                }
                r.fixtures += v.lights.len();
                r.spots += v.lights.iter().filter(|l| l.outer_deg < 180.0).count();
                r.nav_nodes += v.interior_nav.len();
                r.errands += v
                    .residents
                    .iter()
                    .filter(|s| s.role == inf_ecs::components::SlotRole::Errand)
                    .count();
            }
        }

        // …and what the PROJECTOR made of it, through the widest door the
        // shipped player itself uses.
        {
            let mut scene = inf_render::RenderScene::default();
            let voxels = inf_voxel::VoxelVolumes::default();
            // The module-mesh table through the SAME Ring-0 door both hosts
            // use, so this arm's scatter is bucketed against the table a
            // shipped frame is bucketed against.
            let mut meshes = inf_render::ScatterMeshes::new();
            inf_player::scatter_mesh::add_building_modules(&mut meshes);
            inf_player::render::project_scene_full(
                &mut scene,
                sim,
                1.0,
                &inf_player::vmesh::VmeshRegistry::default(),
                &inf_player::skinned::SkinnedRegistry::new(),
                &voxels,
                &mut inf_render::DebrisCache::default(),
                None,
                &meshes,
            );
            r.frame_lights = scene.lights.len();
            r.frame_batches = scene.scatter.len();
            r.frame_emissive_batches = scene
                .scatter
                .iter()
                .filter(|b| b.emissive.iter().any(|c| *c > 0.0))
                .count();
        }

        for _ in 0..VENUE_STEPS {
            sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
            r.digests.push(digest(&sim.state_bytes()));
        }
        let mut d = r.digests.clone();
        d.sort_unstable();
        d.dedup();
        r.distinct = d.len();

        println!(
            "VEN1a {label}: {:.2} h -- {} volume(s), {} instance(s); {} coloured emitter(s) \
             (peak {:.2}), {} pulsing, {} metal; {} fixture(s) ({} spot(s)); {} nav node(s), \
             {} errand slot(s); FRAME: {} light(s), {} scatter batch(es), {} of them emissive",
            r.hour,
            r.volumes,
            r.instances,
            r.emitters,
            r.peak_emissive,
            r.pulsing,
            r.metal,
            r.fixtures,
            r.spots,
            r.nav_nodes,
            r.errands,
            r.frame_lights,
            r.frame_batches,
            r.frame_emissive_batches
        );
        runs.push((label, r));
    }

    // ── the per-host arms, BEFORE the compare ────────────────────────────────
    for (label, r) in &runs {
        assert!(r.volumes > 0, "{label}: no volume streamed in at all");
        assert!(r.instances > 100, "{label}: {} instances", r.instances);
        // THE WAVE'S OWN CLAIM: a venue emits colour.
        assert!(
            r.emitters > 0,
            "{label}: not one coloured emitter in the whole town — a venue whose neon \
             is a warm-white window pane is the thing this wave replaced"
        );
        assert!(
            r.peak_emissive > 1.5,
            "{label}: the brightest authored emission is {:.2}, which is a lit window",
            r.peak_emissive
        );
        assert!(r.pulsing > 0, "{label}: nothing in this town breathes");
        assert!(
            r.metal > 0,
            "{label}: no metal anywhere — a chrome pole shaded as plaster is what \
             `pcg_kind_color` and a fixed roughness of 0.75 produced for four waves"
        );
        // The rig.
        assert!(r.fixtures > 0, "{label}: the venue hangs no light");
        // The interior is walkable, which is what `is_errand_destination`
        // earns: a venue with no slots gets an EMPTY nav graph from `pass.rs`.
        assert!(r.nav_nodes > 0, "{label}: the venue's interior is orphaned");
        assert!(
            r.errands > 0,
            "{label}: the venue is nowhere anybody can be sent"
        );
        // The frame.
        assert!(
            r.frame_emissive_batches > 0,
            "{label}: the projector produced no emissive scatter batch — the bucket \
             key is not carrying the surface"
        );
        // **THE LIGHT-BUDGET MEASUREMENT, at frame scale.** `MAX_LIGHTS` is 16
        // for the whole scene and the truncation is first-N in projection order
        // with no distance priority, so a rig that overran it would silently
        // drop the sun.
        assert!(
            r.frame_lights <= inf_render::passes::mesh::MAX_LIGHTS,
            "{label}: {} lights in one frame against a ceiling of {} — a venue rig \
             has pushed the sun out of the uniform",
            r.frame_lights,
            inf_render::passes::mesh::MAX_LIGHTS
        );
        assert!(
            r.frame_lights > 1,
            "{label}: {} light(s) in the frame — the rig did not reach the projector",
            r.frame_lights
        );
        // The trace evolves, or a byte comparison of two frozen worlds proves
        // nothing.
        assert!(
            r.distinct > VENUE_STEPS / 4,
            "{label}: only {} distinct states across {VENUE_STEPS} steps",
            r.distinct
        );
    }

    // ── the byte compare ─────────────────────────────────────────────────────
    let (a, b) = (&runs[0].1, &runs[1].1);
    assert_eq!(
        a.volumes, b.volumes,
        "the two hosts streamed different worlds"
    );
    assert_eq!(a.emitters, b.emitters, "the two hosts lit different venues");
    assert_eq!(a.fixtures, b.fixtures, "the two hosts hung different rigs");
    assert_eq!(
        a.frame_lights, b.frame_lights,
        "the two hosts projected different frames"
    );
    for (i, (x, y)) in a.digests.iter().zip(b.digests.iter()).enumerate() {
        assert_eq!(
            x, y,
            "PIE and shipping diverged at venue step {i} of {VENUE_STEPS}"
        );
    }
}

/// **THE VEH2b GATE.** A settlement at half past eight: the residents walk to
/// work, the traffic drives past them, and the player pulls one of the drivers
/// out and takes their car — identically on both hosts, byte for byte.
#[test]
fn pie_equals_shipping_at_rush_hour_with_cars_on_the_streets() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let proj = build_project(tmp.path());
    let pack = cook(tmp.path());
    let content = proj.join("Content");
    let design = inf_island::read_design(
        &inf_island::IslandRecipe::load(&fixture_recipe()).expect("recipe"),
    )
    .expect("design");
    let slug = inf_island::slug(&design.recipe.name);
    let settlement = walk_target_settlement(&design);
    let centre = glam::DVec3::new(settlement.centre.x, 0.0, settlement.centre.y);

    let mut hosts: Vec<(&str, RuntimeSim)> = vec![
        ("shipping", pack_sim(&pack)),
        ("PIE", loose_sim(&content, &slug)),
    ];
    let mut runs: Vec<(&str, RushRun)> = Vec::new();
    for (label, sim) in hosts.iter_mut() {
        let run = rush_hour(sim, centre);
        println!(
            "VEH2b {label}: {} street line(s) -> {} lane(s), derived {} time(s); \
             {} car(s): {} with a day, {} driving, {} with a driver, tiers {:?}; \
             the busiest car moved {:.1} m",
            run.streets,
            run.lanes,
            run.derivations,
            run.traffic.cars,
            run.traffic.commuters,
            run.traffic.driving,
            run.traffic.drivers,
            run.traffic.per_tier,
            run.car_moved_m
        );
        println!(
            "VEH2b {label}: the society WALKS -- {} agent(s), tiers {:?}, {} \
             steering intent(s) over the window -- and DRIVES: {} car(s) on the \
             road at {RUSH_HOUR:.1} local",
            run.crowd.per_tier.iter().sum::<usize>(),
            run.crowd.per_tier,
            run.walked,
            run.traffic.driving
        );
        println!(
            "VEH2b {label}: the ladder -- a kilometre away {:?}, at the town \
             edge {:?}, at the crossroads {:?}",
            run.away_tiers, run.edge_tiers, run.traffic.per_tier
        );
        println!(
            "VEH2b {label}: carjack -- {} press(es), driver out {}, hero seated \
             {}; the player's throttle reached the car at {:.2}, which revved to \
             {:.2}, topped {:.2} m/s and covered {:.1} m through the crowd; the \
             victim is {:.1} m from the seat; {} step(s) of the steal digested",
            run.presses,
            run.jacked,
            run.seated,
            run.stolen_throttle,
            run.stolen_revs,
            run.stolen_top_mps,
            run.stolen_m,
            run.victim_away_m,
            run.jack_digests.len()
        );
        runs.push((label, run));
    }

    // ── the per-host arms, BEFORE the compare. Two empty streets agree
    //    perfectly, so every claim below is armed with a count.
    for (label, r) in &runs {
        assert!(
            r.streets > 0,
            "{label}: the level re-derived NO streets from its own blocks"
        );
        assert!(r.lanes >= r.streets * 2, "{label}: {} lanes", r.lanes);
        assert!(
            r.traffic.cars > 0,
            "{label}: a rush hour with no cars in it certifies nothing"
        );
        assert!(
            r.traffic.commuters > 0,
            "{label}: no car has a day, so nothing can be driving"
        );
        assert!(
            r.traffic.driving > 0,
            "{label}: {} cars and not one of them is on the road at half past \
             eight",
            r.traffic.cars
        );
        assert!(
            r.traffic.drivers > 0,
            "{label}: {} cars are driving and none of them has a person in it",
            r.traffic.driving
        );
        assert!(
            r.traffic.per_tier[0] > 0,
            "{label}: no car is a rig — nothing is simulated, so nothing can be \
             stolen"
        );
        assert_eq!(
            r.traffic.per_tier[2], 0,
            "{label}: a car reached the Far rung, which a car does not have"
        );
        // The ladder's own two ends, read off the same population: a kilometre
        // away NOTHING is built, and at the town's edge the middle rung has
        // cars on it. Both are measured by moving the anchor, which is the only
        // thing the band reads.
        // A kilometre away NOTHING is built. On the CI fixture nothing is
        // *recorded* either, and that is the honest reading rather than a
        // weaker arm: the settlement's own cells deactivate, its blocks stop
        // being volumes, and the carriageway derived from those blocks goes
        // with them. **The traffic streams with the town it belongs to** --
        // which is also why `derivations` is more than one on this gate.
        assert_eq!(
            r.away_tiers.iter().take(3).sum::<usize>(),
            0,
            "{label}: a car a kilometre from the hero still has a body: {:?}",
            r.away_tiers
        );
        assert!(
            r.edge_tiers[1] > 0,
            "{label}: at the town edge no car is on the middle rung, so the \
             band is not banding: {:?}",
            r.edge_tiers
        );
        assert!(
            r.car_moved_m > 5.0,
            "{label}: the busiest car in town moved {:.2} m in {RUSH_STEPS} \
             steps — the street is a photograph",
            r.car_moved_m
        );
        // THE SOCIETY WALKS AND DRIVES — the wave's own sentence, as two
        // numbers taken from the same step.
        assert!(
            r.crowd.per_tier.iter().sum::<usize>() > 0,
            "{label}: nobody lives here"
        );
        assert!(
            r.walked > 0,
            "{label}: the town has a population and not one of them took a step"
        );
        // THE CARJACK.
        assert!(
            r.jacked,
            "{label}: {} press(es) and the driver never came out",
            r.presses
        );
        assert!(
            r.seated,
            "{label}: the driver came out and the hero never got in"
        );
        assert!(
            r.victim_away_m > 1.0,
            "{label}: the victim is still inside the car ({:.2} m)",
            r.victim_away_m
        );
        // **THE PLAYER'S THROTTLE REACHES THE CAR THEY STOLE.** That is what
        // this arm can honestly assert on this street, and it is the whole of
        // the control claim: the stick goes through `apply_intent`, the seat
        // step, `VehicleControls::from_intent` and `Vehicle::control`, and the
        // vehicle phase reports it back as `load`.
        //
        // What it deliberately does NOT assert is a speed. The fixture's
        // crossroads holds **eighty-one `Full` crowd agents**, and a `Full`
        // crowd agent is a KINEMATIC capsule — a thing a car cannot push, at
        // any torque. The stolen car is walled in by people, moves a few metres
        // and stops. `a_stolen_car_answers_the_throttle_on_an_empty_street` in
        // `inf-physics` is the falsifier that says that is the CROWD and not the
        // car: the same press on a street with one car and nobody on it is
        // **87.4 m at 18.14 m/s**. Carried, and named.
        assert!(
            r.stolen_throttle > 0.5,
            "{label}: the player's throttle never reached the car they stole \
             ({:.3}) -- the seat step is not driving it",
            r.stolen_throttle
        );
        // Anti-vacuity: a window whose every step hashed the same would compare
        // equal and mean nothing.
        assert!(
            r.distinct > RUSH_STEPS / 2,
            "{label}: only {} distinct states over {RUSH_STEPS} steps",
            r.distinct
        );
        // **`audit:` VEH2b — and the STEAL's own steps are a trace, with the
        // same floor under it.** An empty or constant `jack_digests` would make
        // the equality below certify nothing.
        assert!(
            r.jack_digests.len() >= 24 + STOLEN_STEPS,
            "{label}: the steal produced only {} digested step(s)",
            r.jack_digests.len()
        );
        let distinct_jack: std::collections::BTreeSet<u64> =
            r.jack_digests.iter().copied().collect();
        assert!(
            distinct_jack.len() > r.jack_digests.len() / 2,
            "{label}: only {} distinct states over {} steps of the steal",
            distinct_jack.len(),
            r.jack_digests.len()
        );
    }

    // ── and the two hosts agree, step for step.
    let (a, b) = (&runs[0].1, &runs[1].1);
    assert_eq!(
        a.streets, b.streets,
        "the two hosts derived different streets"
    );
    assert_eq!(a.lanes, b.lanes);
    assert_eq!(a.derivations, b.derivations);
    assert_eq!(a.traffic.cars, b.traffic.cars);
    assert_eq!(a.traffic.commuters, b.traffic.commuters);
    assert_eq!(a.traffic.per_tier, b.traffic.per_tier);
    assert_eq!(a.away_tiers, b.away_tiers);
    assert_eq!(a.edge_tiers, b.edge_tiers);
    assert_eq!(
        a.presses, b.presses,
        "the carjack took a different number of presses"
    );
    assert_eq!(a.digests.len(), b.digests.len());
    for (i, (x, y)) in a.digests.iter().zip(b.digests.iter()).enumerate() {
        assert_eq!(
            x, y,
            "PIE and shipping diverged at rush-hour step {i} of {RUSH_STEPS}"
        );
    }
    // ── **`audit:` VEH2b — and so does the STEAL**, step for step. The rush
    //    window above ends before the hero has touched anything; this is the
    //    approach, every press, the seat swap, the ejection, the victim's
    //    adopted route and ten seconds of driving the car away.
    assert_eq!(
        a.jacked, b.jacked,
        "the driver came out on one host and not the other"
    );
    assert_eq!(
        a.seated, b.seated,
        "the hero got in on one host and not the other"
    );
    assert_eq!(
        a.jack_digests.len(),
        b.jack_digests.len(),
        "the steal ran for a different number of steps on the two hosts"
    );
    for (i, (x, y)) in a.jack_digests.iter().zip(b.jack_digests.iter()).enumerate() {
        assert_eq!(
            x,
            y,
            "PIE and shipping diverged at step {i} of {} of the carjack and the drive away",
            a.jack_digests.len()
        );
    }
}

// ── wave VEN1b: the night ────────────────────────────────────────────────────

/// The hour the club gate reads the town at.
///
/// Half past ten: past `EVENING_OUT_H + EVENING_H`, so every reveller has
/// arrived and nobody has started for home; past `NIGHT_WORK_START_H + 1`, so
/// the keeper is behind the counter; and past `night_glow_step`'s ramp, so the
/// venue's own neon is at full while the population is read.
const CLUB_HOUR: f64 = 22.5;

/// How many steps the club arm compares its traces over, after the clock is
/// wound.
///
/// Half `VENUE_STEPS`, and the difference is the whole reason this arm is dear:
/// the hero stands INSIDE the club rather than on the street outside it, so the
/// settlement's agents are on the `Full` rung with controllers under them —
/// which is the ladder working, and is the 0.25 ms-an-agent
/// `step_character_movement` NPC1c priced. Measured: 240 steps put this arm at
/// 569 s, and the trace's own anti-vacuity floor needs a fraction of that.
const CLUB_STEPS: usize = 120;

/// How near a slot a body has to be to count as AT it, metres in plan.
///
/// A metre. `THERE_M` is four, which is right for "did this agent reach the
/// building it works in" and far too loose for "is this body on the stool it
/// claimed": the seats of one bench are 0.6 m apart, so a four-metre radius
/// would count one body at five of them.
const AT_SLOT_M: f64 = 1.0;

/// How near the venue's own block a body has to be to count as AT the club,
/// metres in plan.
///
/// Forty: a venue lot is 22-32 m of frontage on a 76 m fixture block, so this
/// is "inside the block the club stands on" rather than "inside the room",
/// which is the claim a plan-space distance can honestly make.
const AT_THE_VENUE_M: f64 = 40.0;

/// How much of the venue block opens its doors for the second reading, metres
/// from the speaker.
///
/// Sixty — the block and its pavement. **Every** doorway inside it, interior
/// ones included, and that is a measurement rather than thoroughness:
/// `portal_of` takes the opening nearest the LISTENER, the fixture's one venue
/// block is six bars, and with only the six FRONT doors opened a listener on the
/// pavement kept reading `shut` off the neighbouring bar's *interior* door. A
/// club that is open is open.
const CLUB_OPENS_M: f64 = 60.0;

/// One reading of the doorway model, at one listener position.
///
/// `Default` is "nothing heard at all" and is never a measurement: every field
/// is overwritten by `listen` before anything reads one, and a run that somehow
/// reached the arms carrying it would fail the first verdict comparison rather
/// than pass quietly.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Heard {
    db: f64,
    lowpass_hz: Option<f64>,
    verdict: &'static str,
}

impl Default for Heard {
    fn default() -> Self {
        Self {
            db: f64::NEG_INFINITY,
            lowpass_hz: None,
            verdict: "unmeasured",
        }
    }
}

/// What one host saw at the club.
#[derive(Debug, Default)]
struct ClubRun {
    /// Per-step sim digests over `CLUB_STEPS`.
    digests: Vec<u64>,
    /// Distinct digests — the anti-vacuity counter.
    distinct: usize,
    /// The whole audio command stream, as `physics_demo`'s gate (c) compares
    /// it: a `Debug` rendering of every command both hosts queued.
    audio: String,
    /// The hour the reading was taken at.
    hour: f64,
    /// The society's counters.
    society: inf_ecs::society::SocietyStats,
    /// The venue-audio counters.
    venue: inf_ecs::venue::VenueAudioStats,
    /// **The club's own population**, by what the body is DOING — read off the
    /// records' active arrival rather than off a report.
    seated: usize,
    dancing: usize,
    /// Agents whose active leg ends at a `Work`/night slot — the keeper, the
    /// act, the door.
    on_shift: usize,
    /// Of `seated` + `dancing` + `on_shift`, how many are actually **inside the
    /// venue's own block** — the anti-vacuity half, because a posture is a
    /// property of a schedule and a body in the right posture in the wrong town
    /// is not a club.
    at_the_venue: usize,
    /// Bodies within `AT_SLOT_M` of the venue's counter station.
    at_the_counter: usize,
    /// Agents the crowd step turned to face something, and how many wore a
    /// posture other than standing, on the last step of the window.
    faced: u64,
    posed_apart: u64,
    /// The three listening points, door SHUT then door OPEN.
    shut: [Heard; 3],
    open: [Heard; 3],
    /// `Play` commands and per-step `SetOcclusion` commands over the window,
    /// and the share of the latter that name the speaker the probe listened to.
    plays: usize,
    occlusions: usize,
    occlusions_here: usize,
    /// **How many commands fell off the front of the log's ring** (VEN1b
    /// audit).
    ///
    /// `audio_command_log` is a `BoundedLog` of 8 192 and its own doc says a
    /// test that reasons about the stream must assert this is zero first. The
    /// `audio` field above calls itself *the whole* stream and is compared
    /// between the hosts; the counts beside it are counts over the same slice.
    /// Non-zero would make all three claims about a **tail** — and this wave is
    /// what put the island's first per-step `SetListener` into that ring, by
    /// giving the hero an ear, so the headroom is a fact worth pinning rather
    /// than assuming.
    dropped_audio: u64,
    /// **Every door the doorway rule walks, per occluded source, per step**
    /// (VEN1b audit) — `d3::door::placements` is the UNBANDED list.
    doors: usize,
    /// The mean `audio` phase cost over the window, ms — the phase this wave
    /// added per-step work to.
    audio_ms: f64,
    /// How far the hero stood from that speaker, metres.
    heard_from_m: f64,
    /// How many exterior doors the block opened for the second reading.
    opened_doors: usize,
    /// Whether the venue's own music `Play` named a clip the host resolved.
    music_resolved: bool,
    /// The mean society + crowd phase cost over the window, ms.
    society_ms: f64,
    crowd_ms: f64,
}

/// The venue emitter, its block's exterior doorway, and the three points a
/// listener is put at — all derived from the level rather than authored.
fn club_probe(
    sim: &RuntimeSim,
    listener: glam::DVec3,
) -> Option<(uuid::Uuid, glam::DVec3, [glam::DVec3; 3])> {
    // **The NEAREST speaker, not the lowest-guid one.** A venue is placed per
    // LOT and the fixture's one venue block is six bars (VEN1a's own
    // measurement), so `venue_emitters()` returns six and the first of them is
    // whichever hashed lowest — up to a block's diagonal from where the player
    // is standing, and past `AudioSource::max_distance`. The one a listener
    // hears is the one nearest the listener.
    let (emitter_guid, emitter) = inf_ecs::venue::venue_emitters(sim.world())
        .into_iter()
        .min_by(|a, b| {
            (a.1 - listener)
                .length()
                .total_cmp(&(b.1 - listener).length())
        })?;
    // The nearest EXTERIOR ground-floor doorway to it — the venue's own front
    // door, found the way `society` finds a front door: by asking the level.
    let mut best: Option<(f64, glam::DVec3)> = None;
    for (_, _, d) in inf_ecs::door::volume_doorways(sim.world()) {
        if !(d.exterior && d.floor == 0 && d.hinge.is_finite()) {
            continue;
        }
        let dist = (d.hinge - emitter).length();
        if best.is_none_or(|(b, _)| dist < b) {
            best = Some((dist, d.hinge));
        }
    }
    let (_, hinge) = best?;
    // Outward, in plan, with no trigonometry: away from the room the emitter
    // hangs in.
    let flat = glam::DVec2::new(hinge.x - emitter.x, hinge.z - emitter.z);
    let m = flat.length();
    // `is_finite` first, then the comparison: a NaN length must refuse rather
    // than divide, and `!(m > eps)` says that in a spelling clippy reads as a
    // negated partial order.
    if !m.is_finite() || m <= 1e-3 {
        return None;
    }
    let out = glam::DVec3::new(flat.x / m, 0.0, flat.y / m);
    // Ear height, taken off the door's own mid-height rather than off a
    // constant, so the probe follows the storey the venue actually built.
    let ear = |p: glam::DVec3| glam::DVec3::new(p.x, hinge.y + 0.6, p.z);
    Some((
        emitter_guid,
        emitter,
        [
            ear(emitter - out * 2.0), // inside, two metres off the speaker
            ear(hinge + out * 0.2),   // in the opening
            ear(hinge + out * 4.0),   // out in the street
        ],
    ))
}

/// Read the doorway model at the three points.
fn listen(sim: &mut RuntimeSim, emitter: glam::DVec3, points: [glam::DVec3; 3]) -> [Heard; 3] {
    points.map(|p| {
        let g = sim.audio_portal(p, emitter);
        Heard {
            db: g.db(),
            lowpass_hz: g.lowpass_hz,
            verdict: g.verdict.name(),
        }
    })
}

/// **THE VEN1b GATE.** A night at the club: the town goes out, the keeper is
/// behind the counter, the patrons are on the stools, and the music comes
/// through the door — identically on both hosts, byte for byte, on the sim
/// trace AND on the audio command stream.
///
/// # Coverage first, then the comparison
///
/// The NPC1d law, applied to a club: two empty bars agree perfectly, and so do
/// two bars whose patrons are standing in the street. So every claim below is a
/// count with a floor, and the *posture* counts are held against a count of
/// bodies actually **inside the venue's block** — because a posture is a
/// property of a schedule and an agent sitting down in the wrong town is not
/// nightlife.
///
/// # What the CI fixture can and cannot show
///
/// The fixture's one venue is a **Bar**: `venue_strip` gives a town `[Bar]` and
/// `Fixture Town`'s four blocks fall under `VENUE_SHARE`. A bar has a `BarRoom`
/// and therefore stools, a counter and a door — so *seated*, *the keeper* and
/// *the bouncer* are live claims here. A **dance floor** and a **stage** belong
/// to the `Nightclub` and `StripClub` a city gets, and the shipped island's two
/// cities get all three; what says so at this scale is
/// `a_venue_offers_a_countable_number_of_places_to_be` (16 standing stations
/// and a performer's deck, per archetype, over 4 seeds x 2 storeys) and
/// `a_nightclubs_dance_floor_fills_with_dancers` below, which runs a real
/// nightclub through the production door and settles a society on it without
/// cooking a second island. Stated rather than implied, because a gate that
/// quietly measured a bar and reported a nightclub would be the vacuity this
/// file's own laws are about.
#[test]
fn pie_equals_shipping_at_a_club_on_a_saturday_night() {
    let tmp = tempfile::tempdir().expect("a temp dir");
    let proj = build_project(tmp.path());
    let content = proj.join("Content");
    let pack = cook(tmp.path());
    let recipe =
        inf_island::IslandRecipe::load(&fixture_recipe()).expect("the fixture recipe loads");
    let slug = inf_island::slug(&recipe.name);
    let design = inf_island::read_design(&recipe).expect("the design reads");
    let plans = inf_editor_core::settlement::settlements(&design);
    let venue = plans
        .iter()
        .flat_map(|p| p.blocks.iter())
        .find(|b| b.archetype.is_venue())
        .copied()
        .expect("the fixture places at least one venue");
    println!(
        "VEN1b: a night at the {} at ({:.0}, {:.0})",
        venue.archetype.name(),
        venue.centre.x,
        venue.centre.y
    );

    let mut hosts: Vec<(&str, RuntimeSim)> = vec![
        ("shipping", pack_sim(&pack)),
        ("PIE", loose_sim(&content, &slug)),
    ];
    let mut runs: Vec<(&str, ClubRun)> = Vec::new();

    for (label, sim) in hosts.iter_mut() {
        let hero = hero_entity(sim).expect("a hero");
        let centre = glam::DVec3::new(venue.centre.x, 0.0, venue.centre.y);
        set_hero(sim, hero, centre);
        sim.world_mut().mark_dirty();
        let mut tier_peak = [0usize; 4];
        let mut steered = 0u64;
        settle_the_society(sim, &mut tier_peak, &mut steered);

        // Wind to closing time on the day arm's own pattern, then let one step
        // carry the new hour into every reader of the clock.
        {
            let w = sim.world_mut().world_mut();
            let mut q = w.query::<&mut inf_ecs::components::TimeOfDay>();
            let mut wound = 0usize;
            for mut tod in q.iter_mut(w) {
                tod.seconds =
                    (CLUB_HOUR * 3_600.0 - tod.longitude_deg * 240.0).rem_euclid(86_400.0);
                tod.rate = GATE_CLOCK_RATE;
                wound += 1;
            }
            assert!(wound > 0, "{label}: the island carries no clock to wind");
        }
        sim.world_mut().mark_dirty();
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
        let hour = local_hour(sim);
        assert!(
            (hour - CLUB_HOUR).abs() < 0.5,
            "{label}: the clock reads {hour:.2} h after being wound to {CLUB_HOUR:.1}"
        );

        let mut r = ClubRun {
            hour,
            society: sim.society_stats(),
            venue: sim.venue_audio_stats(),
            ..ClubRun::default()
        };

        // ── who is at the club, and what are they doing ─────────────────────
        //
        // Read off each record's ACTIVE ARRIVAL — the same pure function of
        // `(schedule, clock)` the crowd step publishes onto `CrowdAgent` — so
        // this is the world's own answer and not a second derivation of it.
        // **THE PLAYER WALKS IN.** The settle above put the hero at the venue
        // block's centre with `y = 0` — which is the pattern every arm in this
        // file uses and is *under the ground* on an island whose settlements
        // stand on a levelled pad. Measured here for the first time, because
        // this is the first arm that asks how far the hero is from anything:
        // after four thousand steps of being pushed out of a hillside the body
        // was **212.8 m** from the nearest speaker, well past
        // `AudioSource::max_distance`, so nothing was ever re-evaluated and the
        // whole clause would have certified an empty street.
        //
        // So the hero is put IN the club, at the speaker's own XZ and a body's
        // height under it, and given a few steps to stand up. A gate about what
        // a night at the club sounds like has to have somebody at the club.
        let anchor = inf_ecs::venue::venue_emitters(sim.world())
            .into_iter()
            .min_by(|a, b| (a.1 - centre).length().total_cmp(&(b.1 - centre).length()))
            .map(|(_, at)| at)
            .expect("the venue block carries a speaker");
        set_hero(
            sim,
            hero,
            glam::DVec3::new(anchor.x, anchor.y - 1.5, anchor.z),
        );
        sim.world_mut().mark_dirty();
        for _ in 0..30 {
            sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
        }
        // **The profiler is OPT-IN**, and this arm's budget claims were vacuous
        // until it was armed: `RuntimeSim::step_profile` is "all zeroes until
        // `set_step_profiling` is armed" by its own doc, so `society 0.000 ms /
        // crowd 0.000 ms` passed a ratchet of 0.5 and 1.0 while measuring
        // nothing at all.
        sim.set_step_profiling(true);
        let hero_at = sim
            .world()
            .world()
            .get::<inf_ecs::components::Transform>(hero)
            .map(|t| t.translation.to_dvec3())
            .unwrap_or(centre);
        let probe = club_probe(sim, hero_at);
        let venue_at = glam::DVec3::new(venue.centre.x, 0.0, venue.centre.y);
        {
            let clock = inf_ecs::crowd::CrowdClock::from_world(
                sim.world(),
                inf_ecs::crowd::population_steps(sim.world()) as f64 / HZ,
            );
            let pop = sim
                .world()
                .world()
                .get_resource::<inf_ecs::crowd::CrowdPopulationRes>()
                .expect("the island installed a population");
            for (guid, rec) in &pop.records {
                let leg = rec.leg_at(*guid, clock);
                let arrival = rec.arrival_on(leg);
                let near = glam::DVec2::new(rec.last.x - venue_at.x, rec.last.z - venue_at.z)
                    .length()
                    <= AT_THE_VENUE_M;
                // **A shift is named by the ARRIVAL, not by the hour.**
                // `HOME_H` and `NIGHT_WORK_START_H` are both eighteen hundred —
                // the town comes home at the hour a keeper leaves for work — so
                // an hour test cannot tell a night shift from a walk home.
                // Measured on the nightclub arm below when this read the hour:
                // **155 night workers in a town with 31 night jobs**.
                let working = arrival.role == Some(inf_ecs::components::SlotRole::Work)
                    && matches!(leg, Some((_, u)) if u >= 1.0);
                match arrival.posture {
                    inf_ecs::components::SlotPosture::Sit => r.seated += 1,
                    inf_ecs::components::SlotPosture::Dance => r.dancing += 1,
                    // EMS2 appended `Kneel`, which no building PLAN produces —
                    // it is chosen by the dispatcher for a crew member at a
                    // scene, and a `ResidentSlot`'s arrival can never carry it.
                    // Named rather than swept into a wildcard so the day a room
                    // does offer one, this census has to say what it counts it
                    // as.
                    inf_ecs::components::SlotPosture::Kneel
                    | inf_ecs::components::SlotPosture::Stand => {}
                }
                if working {
                    r.on_shift += 1;
                }
                if near && (working || arrival.posture != inf_ecs::components::SlotPosture::Stand) {
                    r.at_the_venue += 1;
                }
                // A night worker's leg ENDS at its own station -- the counter,
                // the deck, the door -- so "is the keeper at the counter" is
                // "is the body within a metre of where its own leg ends".
                if working {
                    let pts = rec.path_on(leg).points();
                    let end = pts[pts.len() - 1];
                    if glam::DVec2::new(rec.last.x - end.x, rec.last.z - end.z).length()
                        <= AT_SLOT_M
                    {
                        r.at_the_counter += 1;
                    }
                }
            }
        }
        let st = sim.crowd_stats();
        r.faced = st.faced;
        r.posed_apart = st.posed_apart;

        // ── the three points, door SHUT then door OPEN ──────────────────────
        let Some((music_guid, emitter, points)) = probe else {
            panic!("{label}: the venue has no music emitter, or no front door");
        };
        r.heard_from_m = (emitter - hero_at).length();
        r.shut = listen(sim, emitter, points);
        // **THE CLUB OPENS — every door on the block, not one.**
        //
        // `portal_of` takes the opening nearest the LISTENER, which is right and
        // is what this measured: a venue block is six bars, so a listener eight
        // metres out from one bar's door is nearer some OTHER bar's, and with
        // only the probed door opened the street point kept reading `shut` at
        // -24 dB. Opening one door of six and reporting "the club is open" is
        // the vacuity; opening the block is the claim.
        let opened = {
            let mut want: Vec<uuid::Uuid> = Vec::new();
            for (vol, i, d) in inf_ecs::door::volume_doorways(sim.world()) {
                if (d.hinge - emitter).length() <= CLUB_OPENS_M {
                    want.push(inf_physics::d3::door::pcg_doorway_guid(vol, i));
                }
            }
            let specs: Vec<(uuid::Uuid, inf_ecs::door::DoorSpec)> = want
                .into_iter()
                .filter_map(|g| {
                    inf_physics::d3::door::placement_of(sim.world(), g).map(|p| (g, p.spec))
                })
                .collect();
            let n = specs.len();
            let f = inf_ecs::door::door_field_mut(sim.world_mut());
            for (g, spec) in specs {
                f.entry(g, &spec).open_deg = spec.open_limit_deg;
            }
            n
        };
        assert!(
            opened > 0,
            "{label}: the venue block offers no exterior door to open"
        );
        r.opened_doors = opened;
        r.open = listen(sim, emitter, points);
        println!(
            "VEN1b {label}: {:.2} h — {} agent(s); {} night job(s) -> {} worker(s); \
             {} leisure place(s) -> {} reveller(s), {} turned away; {} seated, \
             {} dancing, {} on shift, {} of them at the venue ({} at the \
             counter); {} homebound, {} housebound, {} with no walk home, {} \
             errandless",
            r.hour,
            r.society.agents,
            r.society.night_jobs,
            r.society.night_workers,
            r.society.leisure_places,
            r.society.revellers,
            r.society.turned_away,
            r.seated,
            r.dancing,
            r.on_shift,
            r.at_the_venue,
            r.at_the_counter,
            r.society.homebound,
            r.society.housebound,
            r.society.no_return,
            r.society.errandless
        );
        for (state, row) in [("SHUT", &r.shut), ("OPEN", &r.open)] {
            println!(
                "VEN1b {label}: the door {state:<4} — inside {:+6.1} dB ({:<7}), \
                 doorway {:+6.1} dB ({:<7}), street {:+6.1} dB ({:<7}); low-pass \
                 {:?} / {:?} / {:?}",
                row[0].db,
                row[0].verdict,
                row[1].db,
                row[1].verdict,
                row[2].db,
                row[2].verdict,
                row[0].lowpass_hz,
                row[1].lowpass_hz,
                row[2].lowpass_hz
            );
        }

        // ── the traces ──────────────────────────────────────────────────────
        let mut seen: std::collections::BTreeSet<u64> = Default::default();
        let (mut soc_ms, mut crowd_ms, mut audio_ms) = (0.0f64, 0.0f64, 0.0f64);
        for _ in 0..CLUB_STEPS {
            sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
            let d = digest(&sim.state_bytes());
            seen.insert(d);
            r.digests.push(d);
            // By NAME rather than by index: `step_profile::phase` is
            // `pub(crate)`, and `STEP_PHASE_NAMES` is the public face of the
            // same list -- which is also the spelling that goes red rather than
            // silently measuring the wrong phase if the order ever moves.
            for (name, ms) in sim.step_profile().rows() {
                match name {
                    "society" => soc_ms += ms,
                    "crowd" => crowd_ms += ms,
                    // **THE PHASE THIS WAVE ACTUALLY GREW** (VEN1b audit). The
                    // society and the crowd rows above are the two the wave
                    // barely touched; what it added per-step work to is the
                    // AUDIO step -- a `portal_gain` for every looping occluded
                    // source in earshot, every step, each one an unbanded walk
                    // over every door in the resident world. A new phase cost
                    // that no arm measures is `set_debris_budget`'s shape, and
                    // the file already has the row.
                    "audio" => audio_ms += ms,
                    _ => {}
                }
            }
        }
        r.distinct = seen.len();
        r.society_ms = soc_ms / CLUB_STEPS as f64;
        r.crowd_ms = crowd_ms / CLUB_STEPS as f64;
        r.audio_ms = audio_ms / CLUB_STEPS as f64;
        r.dropped_audio = sim.dropped_audio_commands();
        r.doors = inf_physics::d3::door::placements(sim.world()).len();

        let log = sim.audio_command_log();
        // The audio source key is the guid's low bits -- `guid_source_key`'s
        // own rule, spelled here because it is a private helper of the host.
        let key = music_guid.as_u128() as u64;
        for c in log {
            match c {
                inf_audio::AudioCommand::Play(p) => {
                    r.plays += 1;
                    if p.source == key {
                        r.music_resolved = true;
                    }
                }
                inf_audio::AudioCommand::SetOcclusion { source, .. } => {
                    r.occlusions += 1;
                    if *source == key {
                        r.occlusions_here += 1;
                    }
                }
                _ => {}
            }
        }
        r.audio = format!("{log:?}");
        println!(
            "VEN1b {label}: {} speaker(s) ({} spawned), the nearest {:.1} m off, \
             {} door(s) opened; {} Play(s), {} SetOcclusion(s) ({} for that \
             speaker), {} command(s) dropped from the log's ring; society \
             {:.3} ms / crowd {:.3} ms / audio {:.3} ms a step over \
             {CLUB_STEPS}, with {} door(s) in the resident world",
            r.venue.live,
            r.venue.spawned,
            r.heard_from_m,
            r.opened_doors,
            r.plays,
            r.occlusions,
            r.occlusions_here,
            r.dropped_audio,
            r.society_ms,
            r.crowd_ms,
            r.audio_ms,
            r.doors
        );
        runs.push((label, r));
    }

    // ── the per-host arms, BEFORE the compare ────────────────────────────────
    for (label, r) in &runs {
        // THE SOCIETY AFTER DARK.
        assert!(r.society.agents > 0, "{label}: nobody lives here");
        assert!(
            r.society.night_jobs > 0,
            "{label}: the venue offers no night job — a bar with nobody behind \
             the counter is a bar that is shut"
        );
        assert!(
            r.society.leisure_places > 0,
            "{label}: the venue offers nowhere to spend an evening"
        );
        assert!(
            r.society.night_workers > 0,
            "{label}: {} night job(s) and nobody took one",
            r.society.night_jobs
        );
        assert!(
            r.society.revellers > 0,
            "{label}: {} agents, {} leisure places, and not one of them went out",
            r.society.agents,
            r.society.leisure_places
        );
        assert!(
            r.society.revellers <= r.society.leisure_places,
            "{label}: {} revellers claimed {} places — the occupancy cap is not \
             capping",
            r.society.revellers,
            r.society.leisure_places
        );
        // THE POSTURES, and the anti-vacuity half beside them.
        assert!(
            r.seated > 0,
            "{label}: at half past ten not one body in this town is sitting down"
        );
        assert!(
            r.on_shift > 0,
            "{label}: nobody is working the night shift at {:.1} h",
            r.hour
        );
        assert!(
            r.at_the_venue > 0,
            "{label}: {} seated / {} dancing / {} on shift, and NONE of them is \
             within {AT_THE_VENUE_M:.0} m of the venue — a posture in the wrong town \
             is not a club",
            r.seated,
            r.dancing,
            r.on_shift
        );
        assert!(
            r.at_the_counter > 0,
            "{label}: the keeper's leg ends at the counter and the keeper is not \
             standing at it"
        );
        assert!(
            r.faced > 0,
            "{label}: not one body was turned to face anything — `face_body` \
             wrote through the wrong door for every tier"
        );
        assert!(
            r.posed_apart > 0,
            "{label}: the crowd step published no posture other than standing"
        );
        // THE MUSIC.
        // A venue is placed per LOT and a block is subdivided into them, so
        // the fixture's one venue BLOCK is six bars (VEN1a's own measurement) —
        // six speakers, one apiece. The claim is a floor and an equality
        // between the hosts, not a number.
        assert!(
            r.venue.live > 0,
            "{label}: the venue block carries no speaker at all"
        );
        assert!(
            r.music_resolved,
            "{label}: the venue's own emitter never issued a Play"
        );
        assert!(
            r.occlusions_here >= CLUB_STEPS,
            "{label}: {} SetOcclusion(s) for the speaker the probe listened to \
             ({:.1} m off) over {CLUB_STEPS} steps, {} for all of them — the \
             loop's occlusion is not being re-evaluated",
            r.occlusions_here,
            r.heard_from_m,
            r.occlusions
        );
        // THE DOORWAY, as three numbers with the order between them asserted.
        assert_eq!(r.shut[0].verdict, "clear", "{label}: inside is not clear");
        assert!(
            r.shut[0].db > -0.001,
            "{label}: standing inside the club the music is {:.1} dB down",
            r.shut[0].db
        );
        assert_eq!(
            r.shut[1].verdict, "shut",
            "{label}: the venue's front door is not shut at the start"
        );
        assert_eq!(
            r.shut[1].lowpass_hz,
            Some(inf_physics::d3::audio::DOOR_SHUT_LOWPASS_HZ),
            "{label}: a shut door does not muffle, it only quietens"
        );
        assert_eq!(r.open[0].verdict, "clear");
        assert_eq!(
            r.open[1].verdict, "doorway",
            "{label}: the open door still reads as a wall"
        );
        assert_eq!(
            r.open[1].lowpass_hz, None,
            "{label}: an OPEN door filters the sound"
        );
        // …and the swell: opening the door is louder at the opening and in the
        // street, and the opening is louder than the street.
        assert!(
            r.open[1].db > r.shut[1].db + 6.0,
            "{label}: opening the door changed the doorway by {:.1} dB",
            r.open[1].db - r.shut[1].db
        );
        assert!(
            r.open[2].db > r.shut[2].db + 6.0,
            "{label}: opening the door changed the street by {:.1} dB",
            r.open[2].db - r.shut[2].db
        );
        assert!(
            r.open[1].db > r.open[2].db,
            "{label}: the opening ({:.1} dB) is no louder than the street \
             ({:.1} dB)",
            r.open[1].db,
            r.open[2].db
        );
        // **THE LOG IS WHOLE** (VEN1b audit). `audio_command_log` is a ring of
        // `inf_core::DEFAULT_LOG_CAPACITY` and its own doc says a test that
        // reasons about the stream must assert this first. Three claims below
        // and above rest on it: the `Play`/`SetOcclusion` counts, the
        // `occlusions_here >= CLUB_STEPS` floor, and the host-to-host equality
        // of `r.audio`, which calls itself *the whole* stream. This wave is
        // also what put the island's first per-step `SetListener` into that
        // ring — the hero had no ear before it — so the headroom moved and is
        // pinned rather than assumed.
        assert_eq!(
            r.dropped_audio, 0,
            "{label}: {} audio command(s) fell off the front of an 8 192-entry \
             ring, so the {} Play(s) and {} SetOcclusion(s) counted here are a \
             TAIL and the two hosts are being compared on one",
            r.dropped_audio, r.plays, r.occlusions
        );
        // THE BUDGETS, on the phases this wave added work to.
        //
        // **Reported, not asserted, on a dev build or on CI** (wave VEH2c) —
        // the condition every other budget arm in this file already carries and
        // the one this one needed. It is a WALL CLOCK read inside a
        // `cargo test --workspace -j 3` battery, where twenty-six tests of this
        // binary run in parallel threads beside two other crates' binaries, and
        // under that load it measures the scheduler rather than the engine.
        //
        // Measured both ways at wave VEH2c, on the same commit: **0.120 ms
        // (shipping) and 0.122 ms (PIE)** when this arm is run alone, against
        // the 1.0 ms ratchet — better than eight times over. Under the full
        // battery the same work reads **0.611 and 1.084**, and it read **0.425
        // and 1.008** on the run before it, which is a spread of 2.5x on an
        // unchanged program. A number that moves by 2.5x between two runs of
        // the same code is not a budget, and asserting on it makes a green
        // battery a matter of what else the machine was doing.
        //
        // The ratchet is UNCHANGED at 1.0 and the number is still printed every
        // run, so a real regression is still visible in the line above.
        //
        // The three wall-clock ones only, and the VACUITY checks below stay
        // unconditional: a window whose every step hashed the same is a defect
        // on any build.
        let timed = !cfg!(debug_assertions) && std::env::var_os("CI").is_none();
        if !timed {
            eprintln!(
                "{label}: dev build or CI — the phase costs are reported, not \
                 asserted (audio {:.3}, society {:.3}, crowd {:.3} ms)",
                r.audio_ms, r.society_ms, r.crowd_ms
            );
        }
        assert!(
            !timed || r.audio_ms <= inf_player::budget::AUDIO_STEP_BUDGET_MS,
            "{label}: the audio phase costs {:.3} ms a step against a ratchet \
             of {} — {} door(s) in the resident world, walked once per occluded \
             looping source per step ({} SetOcclusion over {CLUB_STEPS}). {}",
            r.audio_ms,
            inf_player::budget::AUDIO_STEP_BUDGET_MS,
            r.doors,
            r.occlusions,
            inf_player::budget::RATCHET_NOTE
        );
        assert!(
            !timed || r.society_ms <= inf_player::budget::SOCIETY_STEP_BUDGET_MS,
            "{label}: the society phase costs {:.3} ms a step against a ratchet \
             of {}",
            r.society_ms,
            inf_player::budget::SOCIETY_STEP_BUDGET_MS
        );
        assert!(
            !timed || r.crowd_ms <= inf_player::budget::NPC_STEP_BUDGET_MS,
            "{label}: the crowd phase costs {:.3} ms a step against a ratchet of \
             {}",
            r.crowd_ms,
            inf_player::budget::NPC_STEP_BUDGET_MS
        );
        // Anti-vacuity: a window whose every step hashed the same would compare
        // equal and mean nothing.
        assert!(
            r.distinct > CLUB_STEPS / 4,
            "{label}: only {} distinct states across {CLUB_STEPS} steps",
            r.distinct
        );
        assert!(!r.audio.is_empty() && r.plays > 0);
    }

    // ── and the two hosts agree, step for step and command for command ───────
    let (a, b) = (&runs[0].1, &runs[1].1);
    assert_eq!(a.society.agents, b.society.agents);
    assert_eq!(a.society.night_workers, b.society.night_workers);
    assert_eq!(a.society.revellers, b.society.revellers);
    assert_eq!(
        (a.seated, a.dancing, a.on_shift, a.at_the_venue),
        (b.seated, b.dancing, b.on_shift, b.at_the_venue),
        "the two hosts put different people in the club"
    );
    assert_eq!(a.venue.live, b.venue.live);
    assert_eq!(a.shut, b.shut, "the two hosts heard different shut doors");
    assert_eq!(a.open, b.open, "the two hosts heard different open doors");
    for (i, (x, y)) in a.digests.iter().zip(b.digests.iter()).enumerate() {
        assert_eq!(x, y, "PIE and shipping diverged at club step {i}");
    }
    // **AND THE AUDIO STREAM**, which `state_bytes` does not cover: the sim
    // snapshot folds eleven sections and audio is not one of them, so a
    // divergence in what the two hosts PLAYED would be invisible to every arm
    // above. `physics_demo`'s gate (c) compares the same rendering on the
    // playground; this is the same claim about a club.
    assert_eq!(
        a.audio, b.audio,
        "PIE and shipping queued different audio for the same night"
    );
}

/// **A NIGHTCLUB'S DANCE FLOOR FILLS WITH DANCERS** — the half of the wave the
/// CI fixture's bar cannot show, run through the production door on a real
/// `Nightclub`.
///
/// The fixture places one venue and it is a `Bar` (see the gate above for why),
/// so *standing room* and *a performer's deck* have no live subject there. This
/// builds a nightclub from the **committed zone document** — the same
/// `BuildingPass` a settlement block lowers to — puts it in a world with two
/// blocks of homes, settles a society over it, and reads what the schedules put
/// on the floor at half past ten.
///
/// No cook, no second island: it is the *society* that is being measured, and a
/// society is a function of `PcgVolume::residents`, which this builds honestly.
#[test]
fn a_nightclubs_dance_floor_fills_with_dancers() {
    let tmp = tempfile::tempdir().expect("a temp dir");
    let proj = build_project(tmp.path());
    let content = proj.join("Content");

    let mut world = inf_ecs::EcsWorld::new();
    let mut place = |guid: uuid::Uuid, a: inf_pcg::ArchetypeId, centre: glam::DVec3| {
        let passes = zone_passes(&content, a);
        let extent = glam::DVec2::new(38.0, 38.0);
        let cx = inf_pcg::GrammarContext {
            entity: Some(guid),
            center: centre,
            extent,
            seed_offset: 0x5E_1B,
        };
        let height = inf_pcg::FnHeight::new(move |_, _| Some(centre.y));
        let out = inf_pcg::evaluate_buildings(&passes, &inf_pcg::NoSplines, &height, &cx);
        let (baked, solid, groups, doorways, residents, interior, lights, emitters) =
            inf_player::level::population_of(inf_pcg::compose_volume(Vec::new(), out));
        world.spawn_with_guid(guid, a.name(), None);
        let e = world.entity_of(guid).expect("the block");
        let mut vol = inf_ecs::components::PcgVolume {
            extent: inf_ecs::math::Vec2d::new(extent.x, extent.y),
            ..Default::default()
        };
        vol.set_population(
            baked, solid, groups, doorways, residents, interior, lights, emitters,
        );
        world.world_mut().entity_mut(e).insert((
            inf_ecs::components::Transform {
                translation: inf_ecs::math::Vec3d::new(centre.x, centre.y, centre.z),
                ..Default::default()
            },
            inf_ecs::components::GlobalTransform(glam::DAffine3::from_translation(centre)),
            vol,
        ));
    };
    place(
        uuid::Uuid::from_u128(0x0C01),
        inf_pcg::ArchetypeId::Nightclub,
        glam::DVec3::ZERO,
    );
    place(
        uuid::Uuid::from_u128(0x0C02),
        inf_pcg::ArchetypeId::Apartment,
        glam::DVec3::new(90.0, 0.0, 0.0),
    );
    place(
        uuid::Uuid::from_u128(0x0C03),
        inf_pcg::ArchetypeId::Office,
        glam::DVec3::new(0.0, 0.0, 90.0),
    );

    // Settle: the first sync folds, and every one after it plans eight days.
    let mut stats = inf_ecs::society::sync_society(&mut world);
    for _ in 0..600 {
        stats = inf_ecs::society::sync_society(&mut world);
        if stats.pending == 0 && stats.planned_now == 0 {
            break;
        }
    }
    println!(
        "VEN1b nightclub: {} agent(s); {} night job(s) -> {} worker(s); {} \
         leisure place(s) -> {} reveller(s), {} turned away",
        stats.agents,
        stats.night_jobs,
        stats.night_workers,
        stats.leisure_places,
        stats.revellers,
        stats.turned_away
    );

    // What the schedules put on the floor at half past ten.
    let clock = inf_ecs::crowd::CrowdClock::new(0.0, CLUB_HOUR);
    let pop = world
        .world()
        .get_resource::<inf_ecs::crowd::CrowdPopulationRes>()
        .expect("a population");
    let (mut sitting, mut dancing, mut performing, mut tending) = (0usize, 0usize, 0usize, 0usize);
    for (guid, rec) in &pop.records {
        let leg = rec.leg_at(*guid, clock);
        let arrival = rec.arrival_on(leg);
        // The ARRIVAL, not the hour — see the club gate above for the 155-of-31
        // measurement that says why.
        let night = arrival.role == Some(inf_ecs::components::SlotRole::Work)
            && matches!(leg, Some((_, u)) if u >= 1.0);
        match (arrival.posture, night) {
            (inf_ecs::components::SlotPosture::Sit, _) => sitting += 1,
            (inf_ecs::components::SlotPosture::Dance, true) => performing += 1,
            (inf_ecs::components::SlotPosture::Dance, false) => dancing += 1,
            (inf_ecs::components::SlotPosture::Stand, true) => tending += 1,
            _ => {}
        }
    }
    println!(
        "VEN1b nightclub at {CLUB_HOUR:.1} h: {sitting} seated, {dancing} on the \
         dance floor, {performing} performing, {tending} on the door or behind \
         the counter"
    );
    assert!(
        stats.leisure_places > 10,
        "a nightclub offers {} places to spend an evening",
        stats.leisure_places
    );
    assert!(
        dancing > 0,
        "{} leisure places, {} revellers, and NOBODY is on the dance floor",
        stats.leisure_places,
        stats.revellers
    );
    assert!(sitting > 0, "nobody in the club is sitting down");
    assert!(
        performing > 0,
        "the nightclub raised a deck and nobody is performing on it"
    );
    assert!(
        tending > 0,
        "the nightclub has a counter and a door and nobody is at either"
    );
    // …and the whole night population is capped by the CONTENT rather than by a
    // constant: every reveller claimed a distinct place.
    assert!(
        stats.revellers <= stats.leisure_places,
        "{} revellers claimed {} places",
        stats.revellers,
        stats.leisure_places
    );
    assert_eq!(
        stats.night_workers, stats.night_jobs,
        "a nightclub's three night jobs are not all taken: {} of {}",
        stats.night_workers, stats.night_jobs
    );
    drop(tmp);
    let _ = proj;
}

/// **STREAMING AT SPEED — the plane-speed mandate's first number** (wave VEH2c;
/// the route and the arms rewritten by that wave's audit).
///
/// The 2026-08-24 mandate asked for a photoreal island a player crosses at
/// aircraft speed, and every wave since has streamed at a car's. This is the
/// measurement that says what happens when it does not.
///
/// # What is being measured, and why it is the honest question
///
/// Cell activation is **SYNCHRONOUS**: `CellStreaming::sync_sim` activates what
/// the source's radius wants, and a cell the bounded prefetch did not reach is
/// loaded **blocking the step** — the documented v1 semantic, counted by
/// `CellStreamStats::blocking_loads`. Terrain's sim pages are the same, in
/// ascending key order. Neither is CLAMPED, and that is deliberate: the tree
/// states the rule twice, in `CellStreamBudget`'s own doc and in
/// `inf_terrain::StreamBudget`'s — *render* wants are clamped because a missing
/// page costs detail, *sim* wants never are, because a missing page changes the
/// simulation.
///
/// So going faster cannot make the world wrong. It can only make the step
/// **expensive**, and the question is how expensive.
///
/// # THE ROUTE IS FIXED GROUND, NOT A FIXED NUMBER OF STEPS
///
/// **This is the audit's correction and it inverts the headline.** The first cut
/// of this arm walked every speed for the same *900 steps* along the same
/// heading, which is 360 m at 24 m/s and 1 746 m at 110 m/s — and this island is
/// 1 536 m square. Measured on that run: at 110 m/s the source had **zero**
/// resident partition cells from step 231 onward and a terrain set frozen at its
/// last eviction, so **669 of its 900 steps were stepping empty space**. The
/// table said the step got *cheaper* with speed, and what it was reading is that
/// a fast source leaves the island sooner. A source that has left the content is
/// not a measurement of streaming; it is a measurement of nothing.
///
/// So the route is **the same ground at every speed** — the design's own player
/// start (which is `Fixture Town`) to `Fixture Camp`, the fixture's two
/// settlements, read off the recipe rather than typed here — and the *step
/// count* is what varies. Every speed pages the same tiles and activates the
/// same cells; the only difference is how many steps it has to do it in, which
/// is exactly the question the mandate asked.
///
/// The ratchet it is measured against is `STREAMED_STEP_BUDGET_MS`, which is the
/// budget for a fixed step on a streamed world, and it is asserted at the speed
/// the mandate names rather than at a car's.
#[test]
fn streaming_holds_at_aircraft_speed_and_the_table_says_what_it_costs() {
    let tmp = tempfile::tempdir().expect("a temp dir");
    let pack = cook(tmp.path());
    let from = start();

    /// The speeds, m/s: a car, then the mandate's 60 and past it. 60 m/s is
    /// 216 km/h, which is a fast light aircraft and above this wave's own
    /// helicopter (38.7 m/s measured).
    const SPEEDS: [f64; 4] = [24.0, 60.0, 80.0, 110.0];

    // THE ROUTE, from the recipe's own sites: the start (Fixture Town) to
    // Fixture Camp. Both ends are settlements and everything between them is
    // this island's ground, so a source on this line is over content for the
    // whole of it — which arm (a) below measures rather than assumes.
    let recipe =
        inf_island::IslandRecipe::load(&fixture_recipe()).expect("the fixture recipe loads");
    let camp = recipe
        .sites
        .iter()
        .find(|s| s.name == "Fixture Camp")
        .expect("the fixture has a camp");
    let to = glam::DVec3::new(camp.x, from.y, camp.z);
    let route_m = (to - from).length();
    let dir = (to - from) / route_m;
    // The island's own square, from the grid that authored it. A route leaving
    // it is the defect this arm was rewritten to stop making.
    let half = recipe.grid.half_extent_m();
    for p in [from, to] {
        assert!(
            p.x.abs() <= half && p.z.abs() <= half,
            "the route's ({:.1}, {:.1}) is outside the island's own {:.0} m square",
            p.x,
            p.z,
            half * 2.0
        );
    }

    struct Row {
        speed: f64,
        steps: u64,
        blocking: u64,
        activations: u64,
        loads: u64,
        churn: usize,
        mean_ms: f64,
        peak_cells: usize,
        /// The floor of the terrain sim residency over the run — the number
        /// that says whether the source stayed over ground at all.
        min_pages: usize,
        /// Where the milliseconds went: the two streaming phases, and the
        /// vehicle phase, in that order.
        phases: (f64, f64, f64),
    }

    /// One phase's accumulated milliseconds, by the name the profile publishes.
    fn at(prof: &inf_player::step_profile::StepProfile, name: &str) -> f64 {
        prof.ms[inf_player::step_profile::STEP_PHASE_NAMES
            .iter()
            .position(|n| *n == name)
            .unwrap_or_else(|| panic!("no `{name}` phase"))]
    }

    let mut rows: Vec<Row> = Vec::new();
    for speed in SPEEDS {
        let mut sim = pack_sim(&pack);
        let hero = hero_entity(&sim).expect("the island has a hero");
        // Profiling ON, or `step_profile` is all zeroes by its own doc and the
        // budget below would be asserted against nothing (the VEN1b finding).
        sim.set_step_profiling(true);
        // Settle, so the boot's own paging is not charged to the flight.
        set_hero(&mut sim, hero, from);
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
        let base = sim.cell_streaming().stats();
        let base_t = sim.terrain_streaming().stats();
        let (blocking0, acts0, loads0) = (base.blocking_loads, base.activations, base_t.loads);

        let step_m = speed / HZ;
        // The SAME GROUND at every speed: as many steps as this speed needs to
        // fly the route, and not one more.
        let run = (route_m / step_m).ceil() as u64;
        let mut churn = 0usize;
        let mut peak = 0usize;
        let mut min_pages = usize::MAX;
        let mut prev: std::collections::BTreeSet<(i32, i32)> =
            sim.cell_streaming().resident().collect();
        let mut wall = std::time::Duration::ZERO;
        let mut prof = inf_player::step_profile::StepProfile::default();
        for step in 0..run {
            let along = (step as f64 * step_m).min(route_m);
            let p = from + dir * along;
            set_hero(&mut sim, hero, p);
            let t0 = std::time::Instant::now();
            sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
            wall += t0.elapsed();
            prof.accumulate(&sim.step_profile());
            let now: std::collections::BTreeSet<(i32, i32)> =
                sim.cell_streaming().resident().collect();
            churn += now.symmetric_difference(&prev).count();
            peak = peak.max(now.len());
            min_pages = min_pages.min(sim.terrain_streaming().stats().sim_resident_level0);
            prev = now;
        }
        let c = sim.cell_streaming().stats();
        let t = sim.terrain_streaming().stats();
        rows.push(Row {
            speed,
            steps: run,
            blocking: c.blocking_loads - blocking0,
            activations: c.activations - acts0,
            loads: t.loads - loads0,
            churn,
            mean_ms: wall.as_secs_f64() * 1000.0 / run as f64,
            peak_cells: peak,
            min_pages,
            // By NAME through `STEP_PHASE_NAMES`, which is the public face of a
            // `pub(crate)` index table — the idiom this file already uses for
            // the vehicle phase.
            phases: (
                at(&prof, "cell stream") / run as f64,
                at(&prof, "terrain stream") / run as f64,
                at(&prof, "vehicle") / run as f64,
            ),
        });
    }

    println!(
        "STREAMING AT SPEED (the fixture island's {route_m:.0} m town-to-camp route at {HZ} Hz, \
         cooked pack — the SAME GROUND at every speed):"
    );
    println!(
        "   m/s   km/h   steps  blocking  acts  pages  churn  peak  min pg   mean step  \
         cell/terr/veh"
    );
    for r in &rows {
        println!(
            "  {:5.0}  {:5.0}  {:6}  {:8}  {:4}  {:5}  {:5}  {:4}  {:6}  {:8.4} ms  \
             {:.4}/{:.4}/{:.4}",
            r.speed,
            r.speed * 3.6,
            r.steps,
            r.blocking,
            r.activations,
            r.loads,
            r.churn,
            r.peak_cells,
            r.min_pages,
            r.mean_ms,
            r.phases.0,
            r.phases.1,
            r.phases.2
        );
    }

    // (a) THE SOURCE STAYED OVER THE CONTENT, at every speed and at every step.
    //     This is the arm the audit added and it is the one that matters: the
    //     first cut of this measurement ran a fixed 900 steps and its fast rows
    //     spent two thirds of the run off the island, where a step is cheap
    //     because there is nothing in it. A terrain sim residency that ever
    //     reaches zero is a source that has left the world.
    for r in &rows {
        assert!(
            r.min_pages > 0,
            "at {:.0} m/s the terrain sim residency fell to {} pages — the source \
             left the island and the rest of this table is a reading of empty space",
            r.speed,
            r.min_pages
        );
        assert!(
            r.churn > 0 && r.loads > 0,
            "at {:.0} m/s the route paged nothing: {} cell transitions, {} terrain \
             loads",
            r.speed,
            r.churn,
            r.loads
        );
    }

    // (b) THE SAME GROUND PAGES THE SAME. Every speed flies one route, so the
    //     tiles it needs are the same tiles and the pages it loads are the same
    //     count — a streamer whose *work* grew with speed would be doing the
    //     journey differently, and one whose work shrank would be skipping it.
    //     This replaces the first cut's "faster loads more", which was true only
    //     because a faster source covered more ground per step and stopped being
    //     true the moment the ground was held fixed.
    let slow = &rows[0];
    for r in &rows {
        assert_eq!(
            r.loads, slow.loads,
            "at {:.0} m/s the route loaded {} terrain pages against {} at 24 m/s \
             over the same ground",
            r.speed, r.loads, slow.loads
        );
        assert_eq!(
            r.activations, slow.activations,
            "at {:.0} m/s the route activated {} cells against {} at 24 m/s over \
             the same ground",
            r.speed, r.activations, slow.activations
        );
    }

    // (c) RESIDENCY DOES NOT RUN AWAY. The activation radius is a radius, not a
    //     history: a source at 110 m/s holds the same working set a source at
    //     24 m/s does, it just replaces it faster. A peak that grew with speed
    //     would be a leak.
    let peak_slow = slow.peak_cells;
    for r in &rows {
        assert!(
            r.peak_cells <= peak_slow.max(1) * 2,
            "at {:.0} m/s the working set peaked at {} cells against {} at \
             24 m/s — residency is a function of speed, which is a leak",
            r.speed,
            r.peak_cells,
            peak_slow
        );
    }

    // (d) THE BUDGET, at the mandate's own speed. Reported at every speed and
    //     asserted at 60 m/s against the ratchet for a streamed step — on a
    //     release build on a real machine, which is the condition every budget
    //     arm in this file already carries and the one this measurement needs
    //     most: a dev build spends 8 ms a step on this fixture at a CAR's
    //     speed, so a dev assertion here would be measuring rustc.
    let fast = rows.iter().find(|r| r.speed >= 60.0).expect("a 60 m/s row");
    if cfg!(debug_assertions) {
        eprintln!("dev build: the streamed step is reported, not asserted");
        return;
    }
    if std::env::var_os("CI").is_some() {
        eprintln!("CI: the streamed step is reported, not asserted (shared runner)");
        return;
    }
    assert!(
        fast.mean_ms < inf_player::budget::STREAMED_STEP_BUDGET_MS,
        "at 60 m/s the fixed step averaged {:.4} ms against the {} ms budget {}",
        fast.mean_ms,
        inf_player::budget::STREAMED_STEP_BUDGET_MS,
        inf_player::budget::RATCHET_NOTE
    );
}
