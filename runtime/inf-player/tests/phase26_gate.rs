//! **The Phase 26 gate — the wire half (P26.3b).**
//!
//! P26.1 built the tiled container, P26.2 the pool and the residency, P26.3 the
//! WGSL sample — and the P26.3 ledger closed with the honest remainder that
//! spec clause 4 was **not** done: *".inf_mat texture references do not yet
//! resolve in the viewport or the player. Both projectors fill
//! `vt: Default::default()` … what is missing is the persisted binding, which
//! needs a scene schema bump (v21 → v22) and pack dependency edges so a cooked
//! pack carries the `.inf_tex` payloads a level names."*
//!
//! This file holds the arms for that wire. They are deliberately about **bytes
//! and worlds**, not pixels: what P26.3b built is the path from an authored
//! `.inf_mat` to a runtime record, along routes that must not disagree.
//!
//! * **(a) ONE DOOR.** The `.inf_matd` bytes a cooked pack carries are
//!   byte-identical to the ones the PIE payload carries, for the same project.
//!   The P22.2 law made executable: the cook and the payload builder both call
//!   `inf_material::derive_material_bytes` and are compared on their OUTPUT
//!   rather than trusted on their comment. (`fracture_equivalence.rs` is the
//!   precedent, one asset kind over.) **Two producers, not three** — the P26.3b
//!   audit's count: the editor viewport resolves no material at all yet and
//!   `render_assets` does not call the door, so the third path is P26.4's.
//! * **(b) The dependency edges are real.** A level binds a material, the
//!   material names a texture, and the cooked pack contains both — at exact
//!   counts, because `!is_empty()` would pass on a pack that lost the second of
//!   two.
//! * **(c) The payload carries the garment and the hairstyle.** The P24.4 debt.
//!   `phase24_gate::the_pie_payload_carries_no_garment_and_that_is_measured` was
//!   written to fail the day `ScenePayload` grew `cloths`; it fired, it is
//!   retired, and this is the positive arm that replaces its source read.
//! * **(d) PIE == shipping on a textured, clothed scene**, with the anti-vacuity
//!   control the P24.4 mutation matrix earned: the same level with nothing bound
//!   must fold a **different** trace, or the equality above is a statement about
//!   two empty worlds agreeing. A **real `--pie` subprocess** folds it too, since
//!   a boot path that drops an attachment does not crash, it agrees with itself
//!   (P21.4).
//! * **(e) The two hosts build the same material content** (P26.3b audit) — the
//!   maps a host binds and pages FROM, not just the bytes on the wire, and the
//!   registration order that the residency is a pure function of.
//! * **(f) Every silent material hazard raises its advisory** (P26.3b audit): a
//!   missing material, a **material instance**, a missing `.inf_tex` and a **v1**
//!   `.inf_tex` — four fixtures, plus the healthy control.
//!
//! The fixture carries an **unbound** material and texture on purpose, reached
//! through a mesh's own sidecar edges the way a glTF import writes them. Without
//! it (e) cannot tell a walk of the pack index from a walk of the level's
//! bindings, which is exactly the defect it found.
//!
//! # P26.5: the streaming half
//!
//! Everything above is about bytes reaching a runtime *record*. The arms below
//! are the phase's own "Done when", and they run the real streaming loop through
//! the real `EngineRenderer` on a real device:
//!
//! * **(a) a scripted path's residency trace is bit-exact twice per run.** Both
//!   halves, because the P26.4 ledger says they are different claims: the floor
//!   is a pure function of `(camera, bounds, registry)` and is exact
//!   unconditionally; the *feedback* is exact **given the same arrival pattern**,
//!   and the arrival pattern is a GPU-timing fact. The device is pumped to
//!   completion between frames so arrival is pinned, and `feedback_misses` is
//!   asserted at exactly the ring's latency — which is what proves the pattern
//!   was pinned rather than hoped for.
//! * **(b) PIE == shipping on that trace.** The same level's content resolved
//!   two ways — out of a cooked pack and out of a streamed payload — folded
//!   through one renderer, compared **by GUID** (the P26.3 LAW: a handle is a
//!   per-registry index and the two hosts mint different ones).
//! * **(c) an over-budget scene stays inside the pool and every sample lands on
//!   the finest resident ancestor** — asserted as residency STATE over every tile
//!   of every level, never as pixels.
//! * **(d) the engagement counters move on a VT scene and stay at zero on a
//!   textureless one**, with the anti-vacuity half first.
//! * **(e) budgets**, in **three** classes rather than two (rescoped
//!   2026-08-13 after `macos-latest` went red at 49.55 ms with nothing
//!   regressed): the level build against the LOAD-class ceiling, the pages one
//!   frame admits against a per-frame ceiling — the portable half, asserted on
//!   every adapter — and the steady-state milliseconds against
//!   `FRAME_BUDGET_MS`, on an adapter whose clock represents a frame. The P20
//!   law, extended by the observation that a *page* is machine-independent and
//!   a *millisecond* is not; and the P26.4 remainder that the feedback's own
//!   budget is a page cap and not a millisecond one, which turns out to be the
//!   sounder of the two. See `docs/memos/p26-frame-budget-scope.md`.
//! * **(f) the golden set is pinned and additive.**

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::time::Duration;

use inf_asset::{AssetId, AssetKind, AssetSidecar, ContentHash, PackReader};
use inf_ecs::components::{ClothSim, HairGuides, Material, SkeletalMesh};
use inf_editor_core::pie::PieSession;
use inf_editor_core::scene::{serialize, SceneDoc};
use inf_material::{MatBlend, MaterialAsset, TextureCompression, TextureImportSettings};
use inf_packager::{cook, CookOptions};
use inf_project::ProjectManifest;
use inf_runtime::pie::PlayerToEditor;
use uuid::Uuid;

// ── the fixture's stable ids ────────────────────────────────────────────────

const LEVEL: Uuid = Uuid::from_u128(0x2603_0000_0000_0001);
const MAT: Uuid = Uuid::from_u128(0x2603_0000_0000_0002);
const ALBEDO: Uuid = Uuid::from_u128(0x2603_0000_0000_0003);
const ORM: Uuid = Uuid::from_u128(0x2603_0000_0000_0004);
const CLOTH: Uuid = Uuid::from_u128(0x2603_0000_0000_0005);
const HAIR: Uuid = Uuid::from_u128(0x2603_0000_0000_0006);
/// A mesh whose SIDECAR depends on a material nothing in the level binds — the
/// ordinary glTF-import shape (P26.3b audit). `inf_editor_core::assets::import`
/// writes exactly these edges: mesh → its materials → their textures, and none
/// of them is a `Material.asset` binding.
const DECOR_MESH: Uuid = Uuid::from_u128(0x2603_0000_0000_0007);
const DECOR_MAT: Uuid = Uuid::from_u128(0x2603_0000_0000_0008);
const DECOR_TEX: Uuid = Uuid::from_u128(0x2603_0000_0000_0009);

/// The pack must hold exactly these, and the counts are exact rather than
/// non-zero for the P21.4 reason: a walk that lost the second texture of two
/// still satisfies `!is_empty()`.
///
/// **Two materials and three textures, and only ONE of each pair is bound.** The
/// unbound one arrives through the mesh's sidecar edges, which is how a real
/// project gets most of its materials — and it is what makes "the two hosts
/// build the same `MaterialContent`" a claim that can fail (P26.3b audit: the
/// pack side walked the pack INDEX and therefore saw both).
const EXPECTED_MATERIALS_IN_PACK: usize = 2;
const EXPECTED_DERIVED_MATERIALS_IN_PACK: usize = 2;
const EXPECTED_TEXTURES_IN_PACK: usize = 3;
/// …of which the LEVEL binds exactly one material and two textures.
const BOUND_MATERIALS: usize = 1;
const BOUND_TEXTURES: usize = 2;

// ── the fixture ─────────────────────────────────────────────────────────────

/// A `.inf_tex` **v2 tiled container** with a left-to-right red ramp, through
/// `inf_material::build_tiled_texture` — the one writer, so the bytes the pack
/// carries are the bytes a runtime pages.
fn texture_bytes(n: u32, tint: u8) -> Vec<u8> {
    let mut rgba = Vec::with_capacity((n * n * 4) as usize);
    for _y in 0..n {
        for x in 0..n {
            rgba.extend_from_slice(&[(x * 255 / (n - 1)) as u8, tint, 200, 255]);
        }
    }
    inf_material::build_tiled_texture(
        rgba,
        n,
        n,
        TextureImportSettings {
            srgb: true,
            generate_mips: true,
            compression: TextureCompression::None,
            hdr: false,
        },
    )
    .expect("the fixture tiles")
    .into_bytes()
}

/// The authored material: two texture slots bound, the third deliberately empty,
/// so `texture_dependencies()`'s slot order is exercised with a hole in it.
fn material() -> MaterialAsset {
    MaterialAsset {
        base_color: [0.9, 0.4, 0.2, 1.0],
        metallic: 0.25,
        roughness: 0.75,
        base_color_texture: Some(AssetId(ALBEDO)),
        normal_texture: None,
        metallic_roughness_texture: Some(AssetId(ORM)),
        blend: MatBlend::Masked,
        alpha_cutoff: 0.375,
        ..Default::default()
    }
}

/// The level: one materialed, **bound** cube plus one character wearing a
/// garment and a hairstyle. `bound` is the anti-vacuity switch — `false` leaves
/// every component authored and every reference `None`, which is the same level
/// with nothing to resolve.
fn doc_with_binding(bound: bool) -> SceneDoc {
    let mut doc = SceneDoc::new();
    let cube = doc.edit_create(inf_editor_core::ipc::SpawnKind::Cube, "Wall", None);
    let hero = doc.edit_create(inf_editor_core::ipc::SpawnKind::Empty, "Hero", None);
    {
        let world = doc.world_mut();
        let e = world.entity_of(cube).expect("the cube exists");
        // A real `.inf_mesh` reference, ALWAYS (not gated on `bound`): its
        // sidecar drags a material and a texture into the closure that nothing
        // binds, which is how a project gets most of its materials and what makes
        // "the two hosts build the same MaterialContent" falsifiable.
        world
            .world_mut()
            .entity_mut(e)
            .insert(inf_ecs::components::MeshRef {
                asset: Some(DECOR_MESH),
                ..Default::default()
            });
        world.world_mut().entity_mut(e).insert(Material {
            base_color: inf_ecs::math::Color::new(0.9, 0.4, 0.2, 1.0),
            metallic: 0.25,
            roughness: 0.75,
            // P26.3b: the persisted binding scene v22 added. `None` is the
            // scalars-only surface — which is exactly what the control builds.
            asset: bound.then_some(MAT),
            ..Default::default()
        });
        let h = world.entity_of(hero).expect("the hero exists");
        world
            .world_mut()
            .entity_mut(h)
            .insert(SkeletalMesh::default());
        world.world_mut().entity_mut(h).insert(ClothSim {
            asset: bound.then_some(CLOTH),
            enabled: true,
            quality: 1,
        });
        world.world_mut().entity_mut(h).insert(HairGuides {
            asset: bound.then_some(HAIR),
            enabled: true,
            quality: 1,
        });
    }
    doc
}

/// Write `payload` under `content` with a sidecar, so the asset database finds
/// it at the id the level names.
fn put(content: &Path, file: &str, guid: Uuid, bytes: &[u8], kind: AssetKind) {
    put_with_deps(content, file, guid, bytes, kind, &[]);
}

/// [`put`] with explicit sidecar dependency edges — what the glTF importer writes
/// (a mesh names its materials, a material names its textures) and what
/// `dependency_closure` follows through `AssetDb::references_of`.
fn put_with_deps(
    content: &Path,
    file: &str,
    guid: Uuid,
    bytes: &[u8],
    kind: AssetKind,
    deps: &[Uuid],
) {
    let path = content.join(file);
    std::fs::write(&path, bytes).expect("write asset");
    let mut side = AssetSidecar::new(AssetId(guid), kind, ContentHash::of(bytes));
    side.dependencies = deps.iter().copied().map(AssetId).collect();
    side.save(&path).expect("write sidecar");
}

/// A project on disk: the level plus every asset it names.
fn scaffold(tmp: &Path, bound: bool) -> (PathBuf, SceneDoc) {
    let proj = tmp.join("proj");
    ProjectManifest::new("Phase 26 Wire", "blank-3d")
        .save(&proj)
        .unwrap();
    let content = proj.join("Content");
    std::fs::create_dir_all(&content).unwrap();

    let doc = doc_with_binding(bound);
    let level = serialize::encode(&serialize::to_scene_file(&doc)).expect("encode level");
    put(&content, "Wire.inf_lvl", LEVEL, &level, AssetKind::Level);
    put(
        &content,
        "Wall.inf_mat",
        MAT,
        &inf_asset::encode(&material()).expect("encode material"),
        AssetKind::Material,
    );
    put(
        &content,
        "Albedo.inf_tex",
        ALBEDO,
        &texture_bytes(128, 40),
        AssetKind::Texture,
    );
    put(
        &content,
        "Orm.inf_tex",
        ORM,
        &texture_bytes(128, 90),
        AssetKind::Texture,
    );
    // The glTF-import shape: a mesh whose sidecar names a material, whose sidecar
    // names a texture — and no `Material.asset` binding anywhere near them.
    let (decor_mesh, _) =
        inf_dcc::to_mesh_asset(&inf_dcc::cube(0.5), &inf_dcc::ExportOptions::default());
    put_with_deps(
        &content,
        "Decor.inf_mesh",
        DECOR_MESH,
        &inf_asset::encode(&decor_mesh).expect("encode mesh"),
        AssetKind::Mesh,
        &[DECOR_MAT],
    );
    put_with_deps(
        &content,
        "Decor.inf_mat",
        DECOR_MAT,
        &inf_asset::encode(&MaterialAsset {
            base_color_texture: Some(AssetId(DECOR_TEX)),
            ..Default::default()
        })
        .expect("encode decor material"),
        AssetKind::Material,
        &[DECOR_TEX],
    );
    put(
        &content,
        "Decor.inf_tex",
        DECOR_TEX,
        &texture_bytes(128, 200),
        AssetKind::Texture,
    );
    put(
        &content,
        "Coat.inf_cloth",
        CLOTH,
        &inf_asset::encode(&garment()).expect("encode cloth"),
        AssetKind::Cloth,
    );
    put(
        &content,
        "Mane.inf_hair",
        HAIR,
        &inf_asset::encode(&hairstyle()).expect("encode hair"),
        AssetKind::Hair,
    );
    (proj, doc)
}

/// A garment through the Model Editor's own door (`inf_editor_core::groom`), so
/// the bytes on the wire are bytes an author could have made.
fn garment() -> inf_anim::ClothAsset {
    let mesh = inf_dcc::plane(1.0);
    let mut sel = inf_dcc::SelectionSet::new(0);
    for v in mesh.vert_ids().take(2) {
        sel.set_vert(v, true);
    }
    let (asset, report) = inf_editor_core::groom::garment_from_session(
        &mesh,
        &sel,
        *Uuid::from_u128(0x2603_0000_0000_00A1).as_bytes(),
        inf_editor_core::groom::GarmentSpec::default(),
        None,
    )
    .expect("the plane is a garment");
    assert!(report.pinned > 0, "the fixture must pin something");
    asset
}

/// The hairstyle twin.
fn hairstyle() -> inf_anim::HairAsset {
    let mesh = inf_dcc::plane(1.0);
    let mut sel = inf_dcc::SelectionSet::new(0);
    for f in mesh.face_ids() {
        sel.set_face(f, true);
    }
    let (asset, report) = inf_editor_core::groom::groom_from_session(
        &mesh,
        &sel,
        *Uuid::from_u128(0x2603_0000_0000_00A2).as_bytes(),
        inf_editor_core::groom::GroomSpec {
            length_m: 0.3,
            segments: 4,
            ..Default::default()
        },
        None,
    )
    .expect("the plane grows guides");
    assert!(report.strands > 0, "the fixture must grow strands");
    asset
}

fn player_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_inf-player"))
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

/// The PIE payload for `doc`, with every one of the fixture's assets served from
/// the project on disk — the shape the Ring-2 command builds, minus Tauri.
fn payload_for(proj: &Path, doc: &SceneDoc) -> inf_runtime::pie::ScenePayload {
    // Indexed by walking the sidecars directly rather than through
    // `render_assets::content_paths_by_guid`, whose `INDEXED_EXTENSIONS` list is
    // the *render* store's four kinds and does not include `.inf_mat` /
    // `.inf_tex` / `.inf_cloth` / `.inf_hair`. Using it here would have made
    // every arm below fail for a reason that has nothing to do with the wire.
    let content = proj.join("Content");
    let mut by_guid: HashMap<Uuid, PathBuf> = HashMap::new();
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

// ── (a) ONE DOOR, three paths ───────────────────────────────────────────────

/// **The `.inf_matd` a pack carries and the one a payload carries are the same
/// BYTES** (P22.2's law, executable).
///
/// The cook and the PIE payload builder are separate code walking separate
/// representations — a `RuntimeLevel` decoded off disk and a live `SceneDoc` —
/// and P22.2's finding was that exactly this arrangement, with comments claiming
/// agreement "by construction", did not agree: one walked archetype order and
/// the other document order, and one skipped a refusal.
///
/// So they are compared on their output, not trusted on their comment. Both call
/// `inf_material::derive_material_bytes`; a second flattening on either side
/// fails here, and so does a divergent id derivation, because the KEY is
/// compared too.
#[test]
fn the_pack_and_the_payload_derive_the_same_material_bytes() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let (proj, doc) = scaffold(tmp.path(), true);
    let report = cook(&proj, &tmp.path().join("out"), &cook_opts()).expect("the project cooks");

    let derived_id = inf_asset::derived_material_id(AssetId(MAT));
    let reader = PackReader::open(&report.pack_path).expect("open pack");
    let from_pack = reader
        .read(derived_id)
        .expect("the pack carries the record");

    let payload = payload_for(&proj, &doc);
    assert_eq!(
        payload.materials.len(),
        1,
        "the payload carries no derived material — the comparison below would be \
         about nothing"
    );
    let (payload_key, from_payload) = payload.materials[0].clone();
    assert_eq!(
        payload_key,
        derived_id.uuid(),
        "the payload keys its material differently from the pack, so a runtime \
         would need two lookup rules"
    );
    assert_eq!(
        from_payload, from_pack,
        "the cook and the PIE payload builder flattened the same .inf_mat into \
         different bytes — there are two doors, not one"
    );

    // …and the record really says what the material said, so the equality above
    // is not two identical defaults agreeing.
    let rec: inf_asset::DerivedMaterial = inf_asset::decode(&from_pack).expect("decode record");
    assert_eq!(rec.albedo, Some(AssetId(ALBEDO)));
    assert_eq!(rec.normal, None, "the empty slot must stay empty");
    assert_eq!(rec.orm, Some(AssetId(ORM)));
    assert_eq!(rec.blend, inf_asset::DerivedBlend::Masked);
    assert_eq!(rec.alpha_cutoff, 0.375);
}

// ── (b) the dependency edges ────────────────────────────────────────────────

/// **A cooked pack carries the material a level binds and the textures that
/// material names**, at exact counts.
///
/// Two edges, and neither existed before this batch: `Material.asset` (scene
/// v22) and `.inf_mat` → its `.inf_tex` maps. Without the first the pack has no
/// material at all; without the second it has a record naming textures the
/// player asks for and cannot find — which renders as an untextured surface, is
/// indistinguishable from an authored flat colour, and is exactly what the
/// advisory doctrine exists for.
#[test]
fn the_cooked_pack_carries_the_material_and_its_textures() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let (proj, _doc) = scaffold(tmp.path(), true);
    let report = cook(&proj, &tmp.path().join("out"), &cook_opts()).expect("the project cooks");
    let reader = PackReader::open(&report.pack_path).expect("open pack");
    let kinds: Vec<AssetKind> = reader.index().map(|e| e.kind).collect();
    let count = |k: AssetKind| kinds.iter().filter(|x| **x == k).count();

    assert_eq!(
        count(AssetKind::Material),
        EXPECTED_MATERIALS_IN_PACK,
        "the level → .inf_mat edge did not close — kinds: {kinds:?}"
    );
    assert_eq!(
        count(AssetKind::Texture),
        EXPECTED_TEXTURES_IN_PACK,
        "the .inf_mat → .inf_tex edge did not close — kinds: {kinds:?}"
    );
    assert_eq!(
        count(AssetKind::DerivedMaterial),
        EXPECTED_DERIVED_MATERIALS_IN_PACK,
        "no .inf_matd was derived — kinds: {kinds:?}"
    );
    assert_eq!(report.materials_derived, EXPECTED_DERIVED_MATERIALS_IN_PACK);

    // The pack's texture entries really are the tiled containers, readable by the
    // door the player pages through — not merely bytes of the right length.
    for tex in [ALBEDO, ORM] {
        let bytes = reader.read(AssetId(tex)).expect("texture bytes");
        inf_vt::TiledTextureReader::new(bytes).expect("the pack's .inf_tex is a v2 container");
    }

    // And the CONTROL: an unbound level pulls none of it in. Without this the
    // three counts above would pass on a cook that packed the whole content root
    // regardless of what the level references.
    //
    // Sharper than a count since the fixture grew its glTF-shaped decor
    // (P26.3b audit): the unbound level still pulls the MESH's material through
    // the mesh's own sidecar — a real edge, and not this batch's — so what must
    // disappear is the material the LEVEL BOUND, and what must not is the one it
    // never bound. A count of zero would now be measuring the wrong thing.
    let bare = tempfile::tempdir().expect("tempdir");
    let (bare_proj, _) = scaffold(bare.path(), false);
    let bare_report =
        cook(&bare_proj, &bare.path().join("out"), &cook_opts()).expect("the bare project cooks");
    let bare_reader = PackReader::open(&bare_report.pack_path).expect("open bare pack");
    assert!(
        !bare_reader.contains(inf_asset::derived_material_id(AssetId(MAT))),
        "an unbound level still dragged its BOUND material into the pack — the \
         counts above are measuring the content root, not the closure"
    );
    assert!(
        !bare_reader.contains(AssetId(ALBEDO)) && !bare_reader.contains(AssetId(ORM)),
        "the bound material's textures reached a pack whose level binds no material"
    );
    assert!(
        bare_reader.contains(inf_asset::derived_material_id(AssetId(DECOR_MAT)))
            && bare_reader.contains(AssetId(DECOR_TEX)),
        "the MESH's material and its texture vanished from the unbound pack too, \
         so this control is measuring a cook that ships nothing rather than the \
         level→material edge"
    );
}

// ── (c) the payload carries the garment, the hairstyle and the surface ──────

/// **The P24.4 debt, discharged and measured** (P26.3b).
///
/// `phase24_gate::the_pie_payload_carries_no_garment_and_that_is_measured` read
/// `ScenePayload`'s own declaration and was built to fail the day it grew
/// `cloths`. It fired. This is what replaces it, and the difference is the
/// point: that arm could only say the field exists, and this one says the field
/// is FILLED — at an exact count, from a real project, through the real builder.
#[test]
fn the_payload_carries_the_garment_the_hairstyle_and_the_material() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let (proj, doc) = scaffold(tmp.path(), true);
    let payload = payload_for(&proj, &doc);

    assert_eq!(
        payload.schema_version,
        inf_runtime::pie::SCENE_PAYLOAD_VERSION
    );
    assert_eq!(payload.cloths.len(), 1, "no .inf_cloth crossed the wire");
    assert_eq!(payload.hairs.len(), 1, "no .inf_hair crossed the wire");
    assert_eq!(payload.materials.len(), 1, "no material record crossed");
    assert_eq!(
        payload.textures.len(),
        2,
        "the material's two maps did not both cross"
    );
    assert_eq!(payload.cloths[0].0, CLOTH);
    assert_eq!(payload.hairs[0].0, HAIR);

    // The bytes are the authored ones, not a default: decode them back through
    // the same door the player takes.
    let coat: inf_anim::ClothAsset =
        inf_asset::decode(&payload.cloths[0].1).expect("the garment decodes");
    assert_eq!(coat, garment(), "the wire carried a different garment");

    // …and the world the payload builds really wears it. The P21.4 rule: assert
    // the WORLD before comparing two of them.
    let mut sim = inf_player::sim_from_payload(&payload)
        .expect("the payload builds a sim")
        .sim;
    sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
    assert!(
        !inf_ecs::cloth::cloth_state_bytes(sim.world()).is_empty(),
        "the PIE world simulates NO garment — the payload's cloths reached no sim"
    );
    assert!(
        !inf_ecs::hair::hair_state_bytes(sim.world()).is_empty(),
        "the PIE world simulates NO hair"
    );

    // The CONTROL: the same level with nothing bound carries nothing, so the
    // counts above are a measurement of the bindings rather than of the resolver.
    let bare = tempfile::tempdir().expect("tempdir");
    let (bare_proj, bare_doc) = scaffold(bare.path(), false);
    let bare_payload = payload_for(&bare_proj, &bare_doc);
    assert!(bare_payload.cloths.is_empty());
    assert!(bare_payload.hairs.is_empty());
    assert!(bare_payload.materials.is_empty());
    assert!(bare_payload.textures.is_empty());
}

// ── (d) PIE == shipping ─────────────────────────────────────────────────────

/// The determinism trace of a world built from a cooked pack.
fn pack_trace(pack: &Path, frames: u64) -> u128 {
    let source = inf_player::level::PackLevelSource::open(pack).expect("open pack");
    let built = inf_player::build_world_from_pack(&source).expect("build world from pack");
    let mut sim = inf_player::sim_from_built(built);
    // ASSERT THE WORLD BEFORE COMPARING TWO OF THEM (P21.4, and the P24.4
    // mutation that proved it again): with no garment in the pack both sides
    // simulate nothing and agree perfectly.
    sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
    assert!(
        !inf_ecs::cloth::cloth_state_bytes(sim.world()).is_empty(),
        "the cooked pack's character simulates NO garment — the comparison would \
         be two empty worlds agreeing"
    );
    inf_player::fold_trace_sim(sim, frames, None)
}

/// The same, from a streamed payload.
///
/// **The same one-step guard**, and the symmetry is load-bearing rather than
/// tidy: `pack_trace` steps once before folding in order to assert the world has
/// a coat in it, so a payload side that folded from step 0 would be comparing
/// frames 0..n against frames 1..n+1 and reporting a phase offset as a content
/// divergence. (Measured — that is exactly what the first draft did.)
fn payload_trace(payload: &inf_runtime::pie::ScenePayload, frames: u64) -> u128 {
    let mut sim = inf_player::sim_from_payload(payload)
        .expect("the payload builds a sim")
        .sim;
    sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
    inf_player::fold_trace_sim(sim, frames, None)
}

/// **PIE == shipping on a textured, clothed scene.**
///
/// The same level, one world built from a cooked pack and one from the streamed
/// payload, folded over the same fixed steps. They must agree — and the control
/// below must NOT, or the agreement is a statement about two worlds in which
/// nothing was bound.
#[test]
fn pie_equals_shipping_on_a_textured_clothed_scene() {
    const FRAMES: u64 = 16;
    let tmp = tempfile::tempdir().expect("tempdir");
    let (proj, doc) = scaffold(tmp.path(), true);
    let report = cook(&proj, &tmp.path().join("out"), &cook_opts()).expect("the project cooks");

    let shipped = pack_trace(&report.pack_path, FRAMES);
    let previewed = payload_trace(&payload_for(&proj, &doc), FRAMES);
    assert_eq!(
        previewed, shipped,
        "a PIE preview and the shipped build folded different worlds out of one \
         level — the payload and the pack are not carrying the same content"
    );

    // ANTI-VACUITY: unbind everything and the trace must MOVE. Without this the
    // equality above is satisfied by two worlds that simulate nothing, which is
    // exactly the failure the P24.4 matrix produced by severing one cook edge.
    let bare = tempfile::tempdir().expect("tempdir");
    let (bare_proj, bare_doc) = scaffold(bare.path(), false);
    let bare_payload = payload_for(&bare_proj, &bare_doc);
    assert_ne!(
        payload_trace(&bare_payload, FRAMES),
        previewed,
        "the same level with and without its garment folded the SAME trace — \
         nothing in the payload is being simulated"
    );
}

/// **The REAL `--pie` subprocess folds the clothed level** — the arm
/// `phase24_gate`'s retirement note said lived here (P26.3b audit).
///
/// It did not. The note claims the replacement for the retired trip-wire includes
/// "a real `--pie` subprocess folds them", and every arm above this one runs
/// in-process. That distinction is not pedantry in this repository: P21.4's
/// finding was a `--pie` binary that built its sim with a bare `RuntimeSim::new`
/// and therefore *agreed with itself* about a world missing an attachment, with
/// no gate running the binary. `sim_from_payload` is the one seam now, and this
/// is what keeps it that way for the four slots v8 added.
///
/// The anti-vacuity guard is first and the trace must MOVE, or a subprocess that
/// folded an empty world would match a reference that folded an empty world.
#[test]
fn the_real_pie_subprocess_folds_the_garment_and_the_binding() {
    const N: u32 = 8;
    let tmp = tempfile::tempdir().expect("tempdir");
    let (proj, doc) = scaffold(tmp.path(), true);
    let payload = payload_for(&proj, &doc);
    assert_eq!(payload.cloths.len(), 1, "nothing to fold");
    assert_eq!(payload.hairs.len(), 1, "nothing to fold");
    assert_eq!(payload.materials.len(), 1, "no binding on the wire");

    let mut session = PieSession::spawn_scene(&player_bin(), &payload).expect("scene session");
    session.step(N).expect("step N");
    let mut got = Vec::with_capacity(N as usize);
    for _ in 0..N {
        let ev = session
            .wait_for(Duration::from_secs(20), |e| {
                matches!(e, PlayerToEditor::Frame { .. })
            })
            .expect("a frame per step");
        if let PlayerToEditor::Frame { state_hash, .. } = ev {
            got.push(state_hash);
        }
    }
    let want = inf_player::scene_trace(&payload, N as u64).expect("in-process trace");
    assert_eq!(
        got, want,
        "the REAL --pie subprocess folded a different world from the in-process \
         one built out of the SAME payload — a boot path is dropping one of the \
         four slots v8 added"
    );
    assert!(
        got.windows(2).any(|w| w[0] != w[1]),
        "the trace never changed across {N} steps — the garment is not being \
         simulated, so the equality above compares two static worlds"
    );
    session
        .stop(Duration::from_secs(10))
        .expect("graceful stop");
}

// ── (e) the two hosts' material content ─────────────────────────────────────

/// **The pack path and the payload path hand a runtime the SAME material
/// content** (P26.3b audit).
///
/// Arm (a) proves the derived BYTES agree. It cannot see the thing a residency
/// trace is actually a function of: the *maps a host binds from*.
/// `PackLevelSource::material_content` walks a pack index and
/// `materials_from_payload` walks a wire vector — two collectors over two
/// containers, which is exactly the P22.2 arrangement that did not agree. Before
/// this arm neither function had a caller **or a test** anywhere in the tree,
/// under a batch whose headline is that the two wires carry one answer.
///
/// The first version of the pack side collected **every** `.inf_tex` and every
/// `.inf_matd` in the pack rather than what the level binds, so the two sides
/// diverged the moment a closure contained a material no entity bound — which is
/// the ordinary glTF-import case, since a `.inf_mesh` sidecar depends on its
/// materials and they depend on their textures. The sets are compared here, not
/// just their lengths.
#[test]
fn the_pack_and_the_payload_build_the_same_material_content() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let (proj, doc) = scaffold(tmp.path(), true);
    let report = cook(&proj, &tmp.path().join("out"), &cook_opts()).expect("the project cooks");

    let source = inf_player::level::PackLevelSource::open(&report.pack_path).expect("open pack");
    let from_pack = source.material_content();
    let from_payload = inf_player::materials_from_payload(&payload_for(&proj, &doc));

    // ANTI-VACUITY, both halves: a comparison of two empty maps passes.
    assert!(!from_pack.is_empty(), "the pack path resolved no material");
    assert!(
        !from_payload.is_empty(),
        "the payload path resolved no material"
    );
    // BOUND, not PACKED. The pack carries the mesh's material and its texture as
    // well, and a host that registered those would page a set the payload side
    // can never produce — which is exactly what the first cut did, because it
    // walked the pack INDEX.
    assert_eq!(from_pack.materials.len(), BOUND_MATERIALS);
    assert_eq!(from_pack.textures.len(), BOUND_TEXTURES);
    // A `const` block on purpose (clippy's `assertions_on_constants` is right
    // that this is constant-valued, and its remedy is the stronger one): the day
    // someone trims the fixture back to a bound-only closure, this arm stops
    // being able to tell a walk of the pack INDEX from a walk of the level's
    // bindings — and that must fail the BUILD rather than pass a test that has
    // quietly become a tautology.
    const {
        assert!(
            EXPECTED_DERIVED_MATERIALS_IN_PACK > BOUND_MATERIALS
                && EXPECTED_TEXTURES_IN_PACK > BOUND_TEXTURES,
            "the fixture's pack holds no UNBOUND material, so this arm cannot \
             tell a walk of the pack index from a walk of the level's bindings"
        );
    }
    assert!(
        !from_pack.materials.contains_key(&DECOR_MAT)
            && !from_pack.textures.contains_key(&DECOR_TEX),
        "the pack path resolved a material the level never bound — its \
         registration order, and therefore its residency, would be a function of \
         the pack rather than of the level"
    );

    // Keyed by the `.inf_mat` GUID on BOTH sides — the salt is inverted at the
    // boundary, so a projector never sees it. A side that forgot to un-salt keys
    // its map by `derived_material_id(MAT)` and fails here rather than as a
    // lookup miss in a frame.
    let pack_records: BTreeMap<_, _> = from_pack.materials.iter().collect();
    let payload_records: BTreeMap<_, _> = from_payload.materials.iter().collect();
    assert_eq!(
        pack_records, payload_records,
        "the pack and the payload resolved different material records for one level"
    );
    assert!(
        pack_records.contains_key(&MAT),
        "the records are not keyed by the .inf_mat GUID the scene names"
    );

    let pack_textures: BTreeMap<_, _> = from_pack.textures.iter().collect();
    let payload_textures: BTreeMap<_, _> = from_payload.textures.iter().collect();
    assert_eq!(
        pack_textures, payload_textures,
        "the pack and the payload carry different texture BYTES for one level"
    );

    // **The registration order is the residency trace.** `want_floor` is a pure
    // function of the registration SEQUENCE (the P26.3 handle law: the handles
    // may differ across hosts, the pages may not), so the two hosts must walk
    // these identically — and in the fixed slot order, with the empty normal slot
    // skipped rather than shifting the ORM into its place.
    assert_eq!(from_pack.registration_order(), vec![ALBEDO, ORM]);
    assert_eq!(
        from_payload.registration_order(),
        from_pack.registration_order()
    );
}

/// **THE REGISTRATION GAP, CLOSED** (P26.4, clause 0): a cooked pack's bound
/// material becomes a per-instance texture set, and the streamed payload's
/// becomes the same one **by GUID**.
///
/// Every layer under this shipped in P26.1–P26.3b and none of them had a
/// non-test caller: the container, the residency, the WGSL sample, the
/// registration door, the material rule and the want floor were all built and
/// exercised while `no projector called VtTextures::register` and both filled
/// `vt: Default::default()`. So a `.inf_mat`'s textures reached a runtime record
/// on every path and were sampled by nothing.
///
/// GPU-free on purpose: what this asserts is the *decision* — which textures are
/// registered, in what order, and which handles a surface's three slots name —
/// and that decision is `inf-render`'s registry, which needs no adapter. The
/// pixels are `inf-render`'s own `a_virtual_texture_reaches_the_lit_pixel`.
///
/// The cross-host comparison is **by GUID and never by handle**, which is the
/// P26.3 LAW: the editor walks document order and the player walks `Guid` order,
/// so one level mints different handles on the two sides and comparing the
/// integers would be comparing two correct answers and calling them wrong.
#[test]
fn a_bound_material_becomes_a_per_instance_texture_set_on_both_wires() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let (proj, doc) = scaffold(tmp.path(), true);
    let report = cook(&proj, &tmp.path().join("out"), &cook_opts()).expect("the project cooks");

    let build = |content: &inf_player::MaterialContent| {
        let mats = content.vt_materials();
        assert_eq!(mats.len(), BOUND_MATERIALS, "no binding to register");
        let (mut lib, _) = inf_render::VtTextures::new(inf_vt::VtPoolConfig {
            // The fixture's containers are written with `TextureCompression::None`,
            // so their stored pages ARE RGBA8 — the same decision
            // `build_vt_level` makes from the payload's own header, spelled here
            // because this arm builds the registry without a device.
            format: inf_vt::PageFormat::Rgba8,
            stored_tile_size: inf_vt::STORED_TILE_SIZE,
            budget_bytes: inf_vt::DEFAULT_VT_BUDGET_BYTES,
            max_texture_dim: 8192,
            trilinear: false,
            // **Unthrottled** (IB-16). This arm builds a registry without a
            // device to compare TWO REGISTRIES' residency; a per-frame upload
            // budget would bound both identically and prove nothing, and the
            // gate's own budget arm (e) measures the loop's admits directly.
            upload_budget_bytes: 0,
        });
        let n = lib.register_materials(&mats, |g| content.source(g));
        assert_eq!(n, BOUND_TEXTURES, "the door registered the wrong count");
        assert!(
            lib.refusals().is_empty(),
            "a bound texture was refused: {:?}",
            lib.refusals()
        );
        // The transaction that carries the root pages — until it runs, the warm
        // gate correctly refuses to name anything.
        assert!(
            lib.set_for_material(MAT.as_u128()).is_none(),
            "a cold registry named a texture"
        );
        let floor = lib.want_floor();
        let txn = lib.residency_mut().apply_wants(&floor);
        assert_eq!(txn.deferred, 0, "the floor did not fit: {}", txn.trace());
        lib
    };

    let source = inf_player::level::PackLevelSource::open(&report.pack_path).expect("open pack");
    let shipped = build(&source.material_content());
    let previewed = build(&inf_player::materials_from_payload(&payload_for(
        &proj, &doc,
    )));

    // The set a surface bound to MAT gets. Not `NONE` — which is what every
    // projector produced before this batch, on every path, for every level.
    let ship_set = shipped.set_for_material(MAT.as_u128());
    let prev_set = previewed.set_for_material(MAT.as_u128());
    assert!(
        !ship_set.is_none(),
        "the shipped path resolved no textures for a bound material"
    );
    assert_eq!(ship_set.normal, 0, "the empty normal slot must stay empty");
    assert_ne!(ship_set.albedo, 0);
    assert_ne!(ship_set.orm, 0);
    assert_ne!(
        ship_set.albedo, ship_set.orm,
        "one texture is serving two slots"
    );

    // BY GUID (the P26.3 LAW), slot by slot: each side's slot must resolve to the
    // texture the material actually names.
    for (slot, guid) in [(ship_set.albedo, ALBEDO), (ship_set.orm, ORM)] {
        assert_eq!(
            shipped.handle(guid.as_u128()).map(|h| h.0 + 1),
            Some(slot),
            "the shipped set's slot does not name {guid}"
        );
    }
    for (slot, guid) in [(prev_set.albedo, ALBEDO), (prev_set.orm, ORM)] {
        assert_eq!(
            previewed.handle(guid.as_u128()).map(|h| h.0 + 1),
            Some(slot),
            "the previewed set's slot does not name {guid}"
        );
    }

    // …and the residency the two hosts arrive at is the same WORLD: the same
    // tiles of the same textures, keyed by GUID. Handles may differ; pages may
    // not.
    let resident = |lib: &inf_render::VtTextures| {
        let mut out: Vec<(Uuid, inf_vt::TileCoord)> = Vec::new();
        for tex in [ALBEDO, ORM] {
            let h = lib.handle(tex.as_u128()).expect("registered");
            let desc = lib.residency().desc(h).expect("registered").clone();
            for mip in 0..desc.mip_count() {
                let m = desc.mips[mip as usize];
                for y in 0..m.tiles_y {
                    for x in 0..m.tiles_x {
                        let at = inf_vt::TileCoord::new(mip, x, y);
                        if lib.residency().is_resident(h, at) {
                            out.push((tex, at));
                        }
                    }
                }
            }
        }
        out
    };
    let ship_res = resident(&shipped);
    assert!(!ship_res.is_empty(), "nothing is resident");
    assert_eq!(
        ship_res,
        resident(&previewed),
        "a cooked pack and a PIE payload paged different tiles for one level"
    );

    // ANTI-VACUITY: the same level with nothing bound registers nothing, so every
    // equality above is a measurement of the binding rather than of two empty
    // registries agreeing.
    let bare = tempfile::tempdir().expect("tempdir");
    let (bare_proj, bare_doc) = scaffold(bare.path(), false);
    let bare_content = inf_player::materials_from_payload(&payload_for(&bare_proj, &bare_doc));
    assert!(bare_content.vt_materials().is_empty());
}

/// **`registration_order` is a function of the SET, not of how the map was
/// built** (P26.3b audit) — the property that makes cross-host agreement
/// possible at all.
///
/// The P26.3 LAW says a handle is a per-registry index, so the editor (document
/// order) and the player (`Guid` order) mint different handles for one texture
/// and compare by GUID. It explicitly does **not** license the two hosts to page
/// different tiles, and `want_floor` is pure in the registration *sequence* — so
/// a `HashMap` walk would give one level two page sets on two machines, silently,
/// and only under a pool small enough to matter.
///
/// Built twice from opposite insertion orders. `std`'s `RandomState` is seeded
/// per map, so two separately-built `HashMap`s of the same keys genuinely iterate
/// differently; six materials make an accidental agreement negligible. The
/// anti-vacuity assertion is that the sorted answer is NOT the insertion order,
/// or this would be a statement about a constant.
#[test]
fn the_registration_order_is_sorted_not_inserted() {
    let mat = |n: u128| Uuid::from_u128(0x2603_0000_1000_0000_0000_0000_0000_0000 + n);
    let tex = |n: u128| Uuid::from_u128(0x2603_0000_2000_0000_0000_0000_0000_0000 + n);
    // Material `i` names texture `i` in its albedo slot, so the expected order is
    // the materials' GUID order projected onto textures.
    let build = |rev: bool| {
        let mut c = inf_player::MaterialContent::default();
        let mut ids: Vec<u128> = (0..6).collect();
        if rev {
            ids.reverse();
        }
        for i in ids {
            c.materials.insert(
                mat(i),
                inf_asset::DerivedMaterial {
                    albedo: Some(AssetId(tex(i))),
                    ..Default::default()
                },
            );
        }
        c
    };
    let forward = build(false);
    let reversed = build(true);
    let want: Vec<Uuid> = (0..6).map(tex).collect();

    assert_eq!(forward.registration_order(), want);
    assert_eq!(
        reversed.registration_order(),
        want,
        "the registration order depends on the order the map was BUILT — two \
         hosts would page different tiles for one level"
    );
    // Anti-vacuity: the sorted answer is not the reversed insertion order, so the
    // equality above is a statement about sorting rather than about a constant.
    let mut backwards = want.clone();
    backwards.reverse();
    assert_ne!(want, backwards);
}

// ── (f) the advisories fire ─────────────────────────────────────────────────

/// **Every silent material hazard raises a named advisory** (P26.3b audit).
///
/// P26.3b shipped two advisories and this audit added two more, and *none* of
/// them had a caller in any test: a `dangling_material_refs` that returned
/// `Vec::new()` unconditionally was invisible, which is the same shape as the
/// counter that never moves. Four levels, four fixtures, four messages — plus the
/// control, because an advisory list that is never empty is noise.
///
/// The two this audit added are the ones the P16 law names directly: a `.inf_mat`
/// whose `.inf_tex` is **missing**, and one whose `.inf_tex` is a **v1** payload
/// `inf_vt::TiledTextureReader` refuses. Both ship a pack that loads, renders and
/// is textureless.
#[test]
fn every_silent_material_hazard_raises_its_advisory() {
    // The healthy control FIRST: none of the four fires on the good fixture, so
    // the four below are measuring their triggers and not a cook that warns
    // about everything.
    let ok = tempfile::tempdir().expect("tempdir");
    let (ok_proj, _) = scaffold(ok.path(), true);
    let ok_report = cook(&ok_proj, &ok.path().join("out"), &cook_opts()).expect("cooks");
    for w in &ok_report.warnings {
        assert!(
            !w.contains("bound to") && !w.contains("references texture"),
            "the healthy fixture raised a material advisory: {w}"
        );
    }

    // 1. A binding naming an asset the project does not have.
    let case = |mutate: &dyn Fn(&Path, &mut SceneDoc)| -> Vec<String> {
        let tmp = tempfile::tempdir().expect("tempdir");
        let proj = tmp.path().join("proj");
        ProjectManifest::new("Advisory", "blank-3d")
            .save(&proj)
            .unwrap();
        let content = proj.join("Content");
        std::fs::create_dir_all(&content).unwrap();
        let mut doc = doc_with_binding(true);
        mutate(&content, &mut doc);
        let level = serialize::encode(&serialize::to_scene_file(&doc)).expect("encode");
        put(&content, "Wire.inf_lvl", LEVEL, &level, AssetKind::Level);
        put(
            &content,
            "Coat.inf_cloth",
            CLOTH,
            &inf_asset::encode(&garment()).unwrap(),
            AssetKind::Cloth,
        );
        put(
            &content,
            "Mane.inf_hair",
            HAIR,
            &inf_asset::encode(&hairstyle()).unwrap(),
            AssetKind::Hair,
        );
        cook(&proj, &tmp.path().join("out"), &cook_opts())
            .expect("an advisory is not a build failure")
            .warnings
    };

    // (a) the binding names nothing — no `.inf_mat` is written at all.
    let dangling = case(&|_content, _doc| {});
    assert!(
        dangling.iter().any(|w| w.contains("is bound to")
            && w.contains(&MAT.to_string())
            && w.contains("not in \nthe project".replace('\n', "").as_str())),
        "a binding naming a missing material cooked silently: {dangling:?}"
    );

    // (b) the binding names an asset of the WRONG KIND — the `.inf_mati` case,
    // reachable through the shipped Content Drawer before this audit.
    let wrong_kind = case(&|content, _doc| {
        put(
            content,
            "Wall.inf_mati",
            MAT,
            &inf_asset::encode(&inf_material::MaterialInstance::new(AssetId(ALBEDO))).unwrap(),
            AssetKind::MaterialInstance,
        );
    });
    assert!(
        wrong_kind
            .iter()
            .any(|w| w.contains("material_instance") && w.contains("TEXTURELESS")),
        "a binding naming a material INSTANCE cooked silently — the asset is in \
         the project, so nothing looked dangling: {wrong_kind:?}"
    );

    // (c) the material names a texture the project does not have.
    let missing_tex = case(&|content, _doc| {
        put(
            content,
            "Wall.inf_mat",
            MAT,
            &inf_asset::encode(&material()).unwrap(),
            AssetKind::Material,
        );
    });
    assert!(
        missing_tex
            .iter()
            .any(|w| w.contains("references texture") && w.contains("not in the")),
        "a material naming a missing .inf_tex cooked silently: {missing_tex:?}"
    );

    // (d) the material names a **v1** `.inf_tex`: present, valid, and unpageable.
    let v1 = case(&|content, _doc| {
        put(
            content,
            "Wall.inf_mat",
            MAT,
            &inf_asset::encode(&material()).unwrap(),
            AssetKind::Material,
        );
        for id in [ALBEDO, ORM] {
            let payload = inf_asset::encode(&inf_material::TextureAsset {
                schema_version: inf_material::TextureAsset::CURRENT_VERSION,
                width: 4,
                height: 4,
                format: inf_material::TextureFormat::Rgba8,
                srgb: true,
                mips: vec![inf_material::TextureMip {
                    width: 4,
                    height: 4,
                    data: vec![0u8; 4 * 4 * 4],
                }],
            })
            .unwrap();
            assert!(
                !inf_material::tiles::is_v2(&payload),
                "the v1 fixture is not v1, so this case measures nothing"
            );
            put(
                content,
                &format!("Tex{}.inf_tex", id.as_u128() & 0xF),
                id,
                &payload,
                AssetKind::Texture,
            );
        }
    });
    assert!(
        v1.iter()
            .any(|w| w.contains("v1 \n.inf_tex".replace('\n', "").as_str())
                || (w.contains("v1") && w.contains("tiled container"))),
        "a material naming a v1 .inf_tex cooked silently — the pack is bigger and \
         the surface is textureless: {v1:?}"
    );
}

// ── (g) the level's own dependency edges ────────────────────────────────────

/// **A level's sidecar records what it binds, and `has_referrers` sees it**
/// (P26.4) — the P26.3b remainder: *"Level sidecars carry no `dependencies`, so
/// `AssetDb::has_referrers` — the delete-with-refs guard — does not know the
/// level→material edge … deleting a bound texture warns and deleting a bound
/// material does not."*
///
/// The consequence P26.4 creates is what makes it worth closing now: before this
/// batch a deleted `.inf_mat` cost a level nothing visible, because nothing
/// sampled a material. Now the surface loses its maps, and the author gets no
/// warning at the moment they can still say no.
///
/// Asserted through a REAL save and a REAL scan, because the two halves live in
/// two crates that have never agreed about this file: the writer is
/// `inf_editor_core::scene::serialize` and the reader is `inf_asset::AssetDb`,
/// which cannot parse a level sidecar as its own and therefore reads the
/// `dependencies` key out of the raw TOML.
#[test]
fn a_levels_sidecar_records_its_bindings_and_the_delete_guard_sees_them() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let (proj, doc) = scaffold(tmp.path(), true);
    let content = proj.join("Content");

    // Re-save the level through the shipped writer so the sidecar this batch
    // added is the one on disk.
    let level_path = content.join("Wire.inf_lvl");
    let enc = serialize::encode_scene(&doc, Some(LEVEL)).expect("encode");
    std::fs::write(&level_path, &enc.payload).expect("write payload");
    std::fs::write(format!("{}.toml", level_path.display()), &enc.sidecar_toml)
        .expect("write sidecar");

    // The bindings the fixture actually has: the bound material, the decor mesh,
    // the cloth and the hair. NOT the textures — a level names its material and
    // the material names its maps, and composing the two is what the graph is
    // for.
    let side: serialize::Sidecar = toml::from_str(&enc.sidecar_toml).expect("parse sidecar");
    for (label, id) in [
        ("the bound material", MAT),
        ("the mesh", DECOR_MESH),
        ("the garment", CLOTH),
        ("the hairstyle", HAIR),
    ] {
        assert!(
            side.dependencies.contains(&id),
            "{label} is not in the level's dependencies: {:?}",
            side.dependencies
        );
    }
    assert!(
        !side.dependencies.contains(&ALBEDO),
        "the level named a texture directly — the edge must go through the \
         material, or one level's sidecar becomes a transitive closure"
    );
    let mut sorted = side.dependencies.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(
        sorted, side.dependencies,
        "the list is not sorted + deduped"
    );

    // …and the asset database reads them, so the delete guard fires.
    let mut db = inf_asset::AssetDb::new(&content);
    db.scan().expect("scan");
    assert!(
        db.has_referrers(AssetId(MAT)),
        "deleting the bound .inf_mat would warn about nothing"
    );
    assert!(
        db.referenced_by(AssetId(MAT))
            .iter()
            .any(|r| r.uuid() == LEVEL),
        "the referrer is not the level"
    );
    // A material's OWN edges still work (they always did), so this arm is not
    // measuring a database that thinks everything is referenced — and the pair
    // is the point: `DECOR_TEX` is referenced through a real sidecar edge the
    // importer writes, `MAT` through the level edge this batch added.
    assert!(db.has_referrers(AssetId(DECOR_TEX)));
    // …and a texture nothing declares an edge to is still unreferenced, so
    // `has_referrers` has not become "true for everything".
    assert!(
        !db.has_referrers(AssetId(ALBEDO)),
        "the fixture's Wall.inf_mat declares no sidecar dependencies, so its albedo must have no referrer; if it does, this arm is measuring a database that says yes to everything"
    );

    // ANTI-VACUITY: the unbound level does NOT refer to the material, so the
    // assertion above is about the binding and not about the scan.
    let bare = tempfile::tempdir().expect("tempdir");
    let (bare_proj, bare_doc) = scaffold(bare.path(), false);
    let bare_content = bare_proj.join("Content");
    let bare_enc = serialize::encode_scene(&bare_doc, Some(LEVEL)).expect("encode");
    std::fs::write(bare_content.join("Wire.inf_lvl"), &bare_enc.payload).unwrap();
    std::fs::write(
        format!("{}.toml", bare_content.join("Wire.inf_lvl").display()),
        &bare_enc.sidecar_toml,
    )
    .unwrap();
    let mut bare_db = inf_asset::AssetDb::new(&bare_content);
    bare_db.scan().expect("scan");
    assert!(
        !bare_db
            .referenced_by(AssetId(MAT))
            .iter()
            .any(|r| r.uuid() == LEVEL),
        "an unbound level still refers to the material"
    );
}

// ══ P26.5: the streaming gate ═══════════════════════════════════════════════

use inf_core::FRAME_BUDGET_MS;
use inf_player::budget::LOAD_BUDGET_MS;
use inf_render::{GpuContext, VtPopIn};
use inf_vt::{TileCoord, VtTextureHandle};

/// The over-budget fixture's ids.
const BIG_MAT: [Uuid; 3] = [
    Uuid::from_u128(0x2605_0000_0000_0011),
    Uuid::from_u128(0x2605_0000_0000_0012),
    Uuid::from_u128(0x2605_0000_0000_0013),
];
const BIG_TEX: [Uuid; 6] = [
    Uuid::from_u128(0x2605_0000_0000_0021),
    Uuid::from_u128(0x2605_0000_0000_0022),
    Uuid::from_u128(0x2605_0000_0000_0023),
    Uuid::from_u128(0x2605_0000_0000_0024),
    Uuid::from_u128(0x2605_0000_0000_0025),
    Uuid::from_u128(0x2605_0000_0000_0026),
];
/// Six 512² containers, and a pool that holds a fraction of them.
const BIG_EXTENT: u32 = 512;
/// **2 MiB.** The RGBA8 page is 136² × 4 = 73 984 B, so this is 28 slots against
/// a referenced set of six 28-tile pyramids — **six times** the pool, which is
/// the "severalfold" the phase's own "Done when" asks for. Named as a fraction
/// of the content rather than as a round number, and asserted as one below.
const BIG_BUDGET_BYTES: u64 = 2 * 1024 * 1024;

/// The number of frames every scripted arm runs. Twelve is the path length
/// `the_floor_holds_across_a_whole_scripted_path_and_repeats_exactly` uses one
/// crate down, kept so the two are talking about the same walk.
const PATH_FRAMES: u64 = 12;

fn gpu_or_skip(what: &str) -> Option<GpuContext> {
    match GpuContext::headless() {
        Ok(gpu) => Some(gpu),
        Err(e) => {
            eprintln!("SKIP {what}: no GPU adapter ({e})");
            None
        }
    }
}

/// A project whose level binds three materials, six textures, and far more
/// texture bytes than the pool below will hold.
fn over_budget_project(tmp: &Path) -> (PathBuf, SceneDoc) {
    let proj = tmp.join("big");
    ProjectManifest::new("Phase 26 Streaming", "blank-3d")
        .save(&proj)
        .unwrap();
    let content = proj.join("Content");
    std::fs::create_dir_all(&content).unwrap();

    let mut doc = SceneDoc::new();
    for (i, mat) in BIG_MAT.iter().enumerate() {
        put(
            &content,
            &format!("Big{i}.inf_mat"),
            *mat,
            &inf_asset::encode(&MaterialAsset {
                base_color_texture: Some(AssetId(BIG_TEX[i * 2])),
                metallic_roughness_texture: Some(AssetId(BIG_TEX[i * 2 + 1])),
                ..Default::default()
            })
            .expect("encode material"),
            AssetKind::Material,
        );
        for k in 0..2 {
            put(
                &content,
                &format!("BigTex{}.inf_tex", i * 2 + k),
                BIG_TEX[i * 2 + k],
                &texture_bytes(BIG_EXTENT, (40 + i * 40 + k * 15) as u8),
                AssetKind::Texture,
            );
        }
        let cube = doc.edit_create(
            inf_editor_core::ipc::SpawnKind::Cube,
            &format!("Wall{i}"),
            None,
        );
        let world = doc.world_mut();
        let e = world.entity_of(cube).expect("the cube exists");
        world.world_mut().entity_mut(e).insert(Material {
            asset: Some(*mat),
            ..Default::default()
        });
        // Spread them along −Z so a forward walk brings them in one at a time,
        // which is what makes the residency trace a *path* rather than a still.
        world
            .world_mut()
            .entity_mut(e)
            .insert(inf_ecs::components::Transform {
                translation: inf_ecs::math::Vec3d::new(0.0, 0.0, -4.0 * i as f64),
                ..Default::default()
            });
    }
    let level = serialize::encode(&serialize::to_scene_file(&doc)).expect("encode level");
    put(&content, "Big.inf_lvl", LEVEL, &level, AssetKind::Level);
    (proj, doc)
}

/// One frame of a scripted walk: the eye marches toward the wall row.
fn path_view(step: u64) -> inf_render::RenderView {
    inf_render::RenderView {
        origin: inf_math::FloatingOrigin::new(glam::DVec3::ZERO),
        eye_world: glam::DVec3::new(0.0, 0.6, 9.0 - 0.7 * step as f64),
        forward: glam::Vec3::new(0.0, 0.0, -1.0),
        up: glam::Vec3::Y,
        fov_y: 60f32.to_radians(),
        near: 0.05,
        width: 320,
        height: 180,
        ortho: None,
    }
}

/// What one scripted run produced.
struct StreamRun {
    /// Per frame: the whole resident set, named by **asset GUID** and walked in
    /// `MaterialContent::registration_order` — not by handle, which is the P26.3
    /// LAW (the two hosts mint different handles for one texture, so comparing
    /// the integers would be comparing two correct answers and calling them
    /// wrong), and not sorted here, because the registration order is already
    /// the one sequence both hosts walk and arm (b) asserts they agree on it
    /// before comparing a single frame.
    trace: Vec<String>,
    pop_in: VtPopIn,
    /// Slots the pool was planned with, and the peak it held.
    slots: u32,
    peak_resident: u32,
    /// Every (texture, tile) of mip 0 that resolved to something OTHER than
    /// itself, with what it resolved to — the fallback record arm (c) reads.
    fallbacks: Vec<(Uuid, TileCoord, TileCoord)>,
    /// How many mip-0 addresses the FINEST-ancestor walk in [`scripted_run`]
    /// actually checked (P26.5 audit) — arm (c)'s anti-vacuity for a walk that
    /// asserts inside the helper, where the residency is still alive.
    addresses_checked: usize,
    engaged_frames: u64,
    /// Wall time of **each** frame — `render` plus the poll that lets the ring's
    /// maps fire — and **nothing else**. The registry build and the renderer's
    /// own construction (which compiles every shader in the tree) are outside
    /// it: timing those as part of a frame is how a budget arm comes to report
    /// 177 ms and mean nothing. Measured that way once, on purpose, before this
    /// note was written.
    ///
    /// Kept **per frame** rather than as a mean, because frame 0 is a different
    /// kind of frame from the eleven after it and a mean over the two says
    /// neither number (P26.5 follow-up, `docs/memos/p26-frame-budget-scope.md`).
    /// See [`StreamRun::cold_ms`] and [`StreamRun::steady_ms`].
    frame_ms_each: Vec<f64>,
    /// Pages admitted on **each** frame — the same work, counted in a unit no
    /// adapter can inflate.
    ///
    /// One admit is one `queue.write_texture` of one page: `VtPools::apply`
    /// writes exactly the transaction's admits and nothing else, so this is the
    /// upload count as well as the residency decision. It is a pure function of
    /// committed input (arm (a) asserts the trace it comes from is bit-exact),
    /// which is what lets a **ceiling** on it be asserted on every adapter while
    /// the milliseconds cannot be.
    admits_each: Vec<u64>,
    /// Wants offered on each frame, both classes — the size of the scan itself.
    ///
    /// The second world number, and it is not redundant with the first: admits
    /// are **clamped by the pool** (a fixture with 28 slots cannot admit 200
    /// pages however badly it asks), so a want set that stopped being bounded
    /// shows up here and barely there. Mutation-measured: a `justified_mip` two
    /// levels too fine moves the peak admits from 6 to 10 and the peak wants
    /// from 36 to 66. The regression `inf_player::budget` names — *"a want scan
    /// that walks the whole pyramid"* — is a wants regression, not an admits
    /// one.
    wants_each: Vec<u64>,
}

impl StreamRun {
    /// Frame 0 — the cold one, and a different animal: it admits the whole
    /// analytic floor into an empty pool, and it is the frame a lazily-built GPU
    /// resource is built on. Measured and **printed**, never asserted, exactly as
    /// `vgeom_streaming::streaming_overhead_is_bounded` treats its own cold frame
    /// ("a one-off whose absolute cost depends entirely on how much content the
    /// first frame sees").
    fn cold_ms(&self) -> f64 {
        self.frame_ms_each.first().copied().unwrap_or(0.0)
    }

    /// The mean of every frame **after** the cold one — the steady state, which
    /// is the thing a per-frame budget is about.
    fn steady_ms(&self) -> f64 {
        let tail = &self.frame_ms_each[1.min(self.frame_ms_each.len())..];
        if tail.is_empty() {
            return 0.0;
        }
        tail.iter().sum::<f64>() / tail.len() as f64
    }

    /// The mean over **every** frame, cold included — the number the first
    /// version of the budget arm asserted, kept so the printed line says what
    /// changed and by how much.
    fn mean_ms(&self) -> f64 {
        if self.frame_ms_each.is_empty() {
            return 0.0;
        }
        self.frame_ms_each.iter().sum::<f64>() / self.frame_ms_each.len() as f64
    }

    /// The most pages any one frame admitted, **cold frame excluded** — the
    /// steady state, for the same reason [`steady_ms`](Self::steady_ms) exists.
    /// Frame 0 admits the whole floor into an empty pool and is legitimately the
    /// largest frame of the run; a ceiling that had to be above it could not see
    /// a loop re-admitting the whole pool every frame, which is the regression
    /// worth catching.
    fn peak_steady_admits(&self) -> u64 {
        self.admits_each[1.min(self.admits_each.len())..]
            .iter()
            .copied()
            .max()
            .unwrap_or(0)
    }

    /// What the cold frame admitted — printed, never asserted.
    fn cold_admits(&self) -> u64 {
        self.admits_each.first().copied().unwrap_or(0)
    }

    /// The largest want set any frame offered. Not split cold/steady: the floor
    /// is bounded on **every** frame by construction
    /// (`visible surfaces × VT_FLOOR_MAX_TILES` plus the camera-free coarse
    /// levels), so frame 0 is not a special case here the way it is for admits.
    fn peak_wants(&self) -> u64 {
        self.wants_each.iter().copied().max().unwrap_or(0)
    }

    /// `frame:wants` for every frame.
    fn wants_trace(&self) -> String {
        self.wants_each
            .iter()
            .enumerate()
            .map(|(i, w)| format!("{i}:{w}"))
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// `frame:admits` for every frame, for a failure message that says *which*
    /// frame did the work.
    fn admits_trace(&self) -> String {
        self.admits_each
            .iter()
            .enumerate()
            .map(|(i, a)| format!("{i}:{a}"))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// **Run the scripted path through the real renderer** and record the world.
///
/// The device is pumped to completion after every frame. That is not tidiness:
/// `take(frame)` returns the mask recorded at `frame − 2` **or nothing**, and
/// which frames get "or nothing" is a GPU-timing fact — so without the pump the
/// refinement half of the trace is a function of how fast the device drained.
/// With it, arrival is a constant and `feedback_misses` says so.
fn scripted_run(
    gpu: &GpuContext,
    content: &inf_player::MaterialContent,
    budget: u64,
    frames: u64,
) -> StreamRun {
    let mats = content.vt_materials();
    assert!(!mats.is_empty(), "the fixture binds no material");
    let (lib, pools, report) = inf_render::build_vt_level(
        &gpu.device,
        &gpu.queue,
        &inf_render::RenderSettings::default(),
        budget,
        &mats,
        |g| content.source(g),
    )
    .expect("the level builds");
    assert_eq!(report.refused, 0, "a bound texture was refused");
    let slots = lib.residency().stats().slots;
    // GUIDs in registration order, so the trace can name them.
    let order = content.registration_order();
    let handles: Vec<(Uuid, VtTextureHandle)> = order
        .iter()
        .filter_map(|g| lib.handle(g.as_u128()).map(|h| (*g, h)))
        .collect();
    assert_eq!(
        handles.len(),
        order.len(),
        "a bound texture did not register"
    );

    let target = inf_render::HeadlessTarget::new(gpu, 320, 180);
    let mut renderer = inf_render::EngineRenderer::new(gpu, inf_render::HEADLESS_FORMAT);
    renderer.set_vt_level(Some((lib, pools)));

    let mut trace = Vec::with_capacity(frames as usize);
    let mut peak_resident = 0u32;
    let mut frame_ms_each = Vec::with_capacity(frames as usize);
    let mut admits_each = Vec::with_capacity(frames as usize);
    let mut wants_each = Vec::with_capacity(frames as usize);
    let mut admits_so_far = 0u64;
    let mut wants_so_far = 0u64;
    for step in 0..frames {
        // The projector's job, once per frame: the sets come out of the live
        // registry, so a texture that is not yet warm is simply not named.
        let mut scene = inf_render::RenderScene::default();
        for (i, mat) in BIG_MAT.iter().enumerate() {
            let set = inf_render::vt_set_for(renderer.vt_textures(), Some(mat.as_u128()));
            scene.instances.push(inf_render::MeshInstance {
                vt: set,
                translation: glam::DVec3::new(0.0, 0.0, -4.0 * i as f64),
                rotation: glam::Quat::IDENTITY,
                scale: glam::Vec3::splat(2.0),
                color: [1.0, 1.0, 1.0, 1.0],
                metallic: 0.0,
                roughness: 0.5,
                emissive: [0.0; 3],
                id: i as u32 + 1,
                mesh: inf_render::PrimMesh::Cube,
                blend: 0,
                cutoff: 0.5,
            });
        }
        scene.mark_dirty();
        let t = std::time::Instant::now();
        renderer.render(gpu, &scene, &path_view(step), &target.view, (320, 180));
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        frame_ms_each.push(t.elapsed().as_secs_f64() * 1000.0);
        // The frame's own admits and wants, differenced out of the running
        // totals — the work THIS frame asked the queue for, in pages, and the
        // size of the scan that asked for it.
        let pop = renderer.vt_pop_in();
        admits_each.push(pop.admits - admits_so_far);
        admits_so_far = pop.admits;
        let wants_now = pop.floor_wants + pop.refine_wants;
        wants_each.push(wants_now - wants_so_far);
        wants_so_far = wants_now;

        let res = renderer
            .vt_textures()
            .expect("the level is live")
            .residency();
        let mut line = String::new();
        let mut resident = 0u32;
        for (guid, h) in &handles {
            let desc = res.desc(*h).expect("registered").clone();
            for mip in 0..desc.mip_count() {
                let m = desc.mips[mip as usize];
                for y in 0..m.tiles_y {
                    for x in 0..m.tiles_x {
                        let at = TileCoord::new(mip, x, y);
                        if res.is_resident(*h, at) {
                            resident += 1;
                            line.push_str(&format!("{guid} {mip} {x} {y};"));
                        }
                    }
                }
            }
        }
        peak_resident = peak_resident.max(resident);
        trace.push(line);
    }

    // The fallback record, taken at the END of the path (the interesting state:
    // the pool is full and the camera has moved).
    //
    // **The two halves of "the FINEST resident ancestor" are asserted HERE**
    // (P26.5 audit), because this is where the residency is still alive. Arm (c)
    // used to compare each fallback against `VtTextureDesc::ancestor` at the
    // level it was served, and that is only the *"is an ancestor"* half:
    // measured, a `VtResidency::resolve` that named the always-pinned ROOT for
    // every address passed the whole arm — the root IS the correct ancestor at
    // the root's level — while failing three `inf-vt` arms. So the arm named
    // after the phase's own "Done when" clause could not see the defect that
    // clause is about. What is missing from `ancestor(at, got.mip) == got` is
    // (i) that the named tile is RESIDENT and (ii) that nothing FINER is, which
    // is what makes it the finest rather than merely one of them.
    let lib = renderer.vt_textures().expect("the level is live");
    let res = lib.residency();
    let mut fallbacks = Vec::new();
    let mut addresses_checked = 0usize;
    for (guid, h) in &handles {
        let desc = res.desc(*h).expect("registered").clone();
        let m = desc.mips[0];
        for y in 0..m.tiles_y {
            for x in 0..m.tiles_x {
                let at = TileCoord::new(0, x, y);
                let got = res.resolve(*h, at).expect("every address resolves").tile;
                addresses_checked += 1;
                // (i) RESIDENT. An address that resolves to a tile holding no
                // page is a slot the shader samples for stale texels — the
                // premise `VtPools`' evict path rests on.
                assert!(
                    res.is_resident(*h, got),
                    "{guid} {at:?} resolves to {got:?}, which is NOT resident"
                );
                if got == at {
                    continue;
                }
                // (ii) FINEST. Every level strictly between the wanted one and
                // the served one must be absent, or a finer ancestor was
                // available and the table handed out a coarser answer.
                for lv in (at.mip + 1)..got.mip {
                    let finer = desc
                        .ancestor(at, lv)
                        .expect("a level between two levels of one pyramid");
                    assert!(
                        !res.is_resident(*h, finer),
                        "{guid} {at:?} was served {got:?} while its ancestor \
                         {finer:?} is resident — the fallback is an ancestor but \
                         not the FINEST one, which is the clause this gate closes"
                    );
                }
                fallbacks.push((*guid, at, got));
            }
        }
    }
    StreamRun {
        trace,
        pop_in: renderer.vt_pop_in(),
        slots,
        peak_resident,
        fallbacks,
        addresses_checked,
        engaged_frames: renderer.vt_engaged_frames(),
        frame_ms_each,
        admits_each,
        wants_each,
    }
}

/// The over-budget project, cooked, with its pack content resolved.
fn big_pack(tmp: &Path) -> (inf_player::MaterialContent, u64) {
    let (proj, _doc) = over_budget_project(tmp);
    let report = cook(&proj, &tmp.join("out"), &cook_opts()).expect("the project cooks");
    let source = inf_player::level::PackLevelSource::open(&report.pack_path).expect("open pack");
    let content = source.material_content();
    let bytes: u64 = content
        .registration_order()
        .iter()
        .filter_map(|g| {
            content
                .source(g.as_u128())
                .map(|s| s.payload().len() as u64)
        })
        .sum();
    (content, bytes)
}

// ── (a) the scripted-path residency trace, bit-exact twice ──────────────────

/// **(a) Two runs of one scripted path produce the same residency trace, byte
/// for byte** — and the arrival pattern that makes the refinement half of that
/// claim true is pinned rather than assumed (P26.5).
///
/// The P26.4 ledger's own reading, which this arm is built to honour:
///
/// > The floor is bit-exact unconditionally: it is a pure function of
/// > `(camera, bounds, registry)` with no clock, no frame history and no GPU in
/// > it. The *feedback* is bit-exact **given the same arrival pattern**, and the
/// > arrival pattern is a GPU-timing fact … That is the honest reading of "a
/// > dropped feedback frame degrades to the floor, deterministically" — the
/// > *degradation* is deterministic, the *timing* is not.
///
/// So the device is pumped to completion between frames and
/// `VtPopIn::feedback_misses` is asserted at **exactly** the ring's latency:
/// two frames can have no mask to read because nothing was recorded two frames
/// earlier, and every frame after that must land. A run that missed more than
/// that did not have a pinned arrival pattern, and the trace equality below
/// would be luck.
#[test]
fn the_scripted_paths_residency_trace_is_bit_exact_twice() {
    let Some(gpu) = gpu_or_skip("the P26 residency trace") else {
        return;
    };
    let tmp = tempfile::tempdir().expect("tempdir");
    let (content, _) = big_pack(tmp.path());

    let a = scripted_run(&gpu, &content, BIG_BUDGET_BYTES, PATH_FRAMES);
    let b = scripted_run(&gpu, &content, BIG_BUDGET_BYTES, PATH_FRAMES);

    // ANTI-VACUITY FIRST: the trace is not a column of empty strings, and it
    // MOVES along the path. Two runs of a streamer that admitted nothing agree
    // perfectly.
    assert_eq!(a.trace.len(), PATH_FRAMES as usize);
    assert!(
        a.trace.iter().all(|l| !l.is_empty()),
        "a frame of the path had nothing resident at all"
    );
    assert!(
        a.trace.windows(2).any(|w| w[0] != w[1]),
        "residency never changed across {PATH_FRAMES} frames of a scripted \
         approach, so the equality below is about a still frame"
    );
    assert_eq!(
        a.trace, b.trace,
        "two runs of one scripted path produced different residency traces"
    );

    // THE ARRIVAL PATTERN, pinned — and the number is `latency + 1`, measured
    // rather than assumed. `READBACK_LATENCY_FRAMES` frames miss because nothing
    // was recorded that far back; the extra one is the COLD FIRST FRAME. On
    // frame 0 the registry is not yet warm, so `vt_set_for` names nothing, the
    // coverage list is empty, `VtFeedback::record` dispatches nothing and hands
    // the ring no copy — which frame 2 then misses. That is "a dropped feedback
    // frame degrades to the floor" taken for real, on the one frame of a level's
    // life where it is guaranteed.
    let latency = inf_render::READBACK_LATENCY_FRAMES + 1;
    for run in [&a, &b] {
        assert_eq!(
            run.pop_in.frames,
            PATH_FRAMES,
            "the streaming loop did not run once per frame: {}",
            run.pop_in.summary()
        );
        assert_eq!(
            run.pop_in.feedback_misses,
            latency,
            "the ring missed {} frames of {PATH_FRAMES}, not the {latency} its \
             pinned latency and its cold first frame account for — the arrival \
             pattern is not pinned, so the trace equality above is luck: {}",
            run.pop_in.feedback_misses,
            run.pop_in.summary()
        );
        assert_eq!(
            run.pop_in.feedback_frames + run.pop_in.feedback_misses,
            PATH_FRAMES,
            "a frame neither read a mask nor recorded a miss"
        );
        // …and both want classes are populated, or the per-class pop-in counters
        // are measuring one thing.
        assert!(
            run.pop_in.floor_wants > 0 && run.pop_in.refine_wants > 0,
            "one want class is empty: {}",
            run.pop_in.summary()
        );
    }
    assert_eq!(
        a.pop_in, b.pop_in,
        "the pop-in counters differ between runs"
    );
}

// ── (b) PIE == shipping, on that trace ──────────────────────────────────────

/// **(b) The cooked pack and the streamed payload page the same tiles, frame for
/// frame** (P26.5).
///
/// Arm (a) says the streamer is deterministic; this says the two *hosts* feed it
/// the same thing. They resolve differently and must — the player reads derived
/// `.inf_matd` records out of a pack, the editor-side payload builder reads
/// authored `.inf_mat` files — so what is compared is the residency each one
/// arrives at, **by GUID**.
///
/// The control is the one the P24.4 mutation matrix earned: an unbound level
/// must page nothing, or the equality is two empty worlds agreeing.
#[test]
fn pie_equals_shipping_on_the_residency_trace() {
    let Some(gpu) = gpu_or_skip("the P26 PIE-vs-shipping trace") else {
        return;
    };
    let tmp = tempfile::tempdir().expect("tempdir");
    let proj_dir = tmp.path().join("big");
    let (content_from_pack, referenced) = big_pack(tmp.path());

    // The payload side, over the same project.
    let doc = serialize::load(&proj_dir.join("Content").join("Big.inf_lvl")).expect("load level");
    let content_from_payload = inf_player::materials_from_payload(&payload_for(&proj_dir, &doc));

    // Both resolve SOMETHING, or the comparison is about two empty maps.
    assert_eq!(content_from_pack.materials.len(), BIG_MAT.len());
    assert_eq!(
        content_from_payload.registration_order(),
        content_from_pack.registration_order(),
        "the two hosts walk different registration orders, so their residencies \
         cannot be compared at all"
    );
    assert!(referenced > 0);

    let shipped = scripted_run(&gpu, &content_from_pack, BIG_BUDGET_BYTES, PATH_FRAMES);
    let previewed = scripted_run(&gpu, &content_from_payload, BIG_BUDGET_BYTES, PATH_FRAMES);
    assert_eq!(
        shipped.trace, previewed.trace,
        "a cooked pack and a PIE payload paged different tiles for one level"
    );
    assert_eq!(shipped.pop_in, previewed.pop_in);

    // ANTI-VACUITY: an unbound level registers nothing at all, so every equality
    // above is a measurement of the binding rather than of two empty registries.
    let bare = tempfile::tempdir().expect("tempdir");
    let (bare_proj, bare_doc) = scaffold(bare.path(), false);
    let bare_content = inf_player::materials_from_payload(&payload_for(&bare_proj, &bare_doc));
    assert!(bare_content.vt_materials().is_empty());
}

// ── (c) the over-budget scene ───────────────────────────────────────────────

/// **(c) A scene referencing several times the physical pool renders inside it,
/// and every sample lands on the finest resident ancestor** (P26.5) — the
/// phase's own "Done when", asserted as residency STATE and never as pixels.
#[test]
fn an_over_budget_scene_stays_in_the_pool_and_falls_back_to_the_finest_ancestor() {
    let Some(gpu) = gpu_or_skip("the P26 over-budget scene") else {
        return;
    };
    let tmp = tempfile::tempdir().expect("tempdir");
    let (content, referenced) = big_pack(tmp.path());

    // THE PREMISE, measured rather than asserted by construction: the level
    // references several times what the pool can hold. A fixture that fitted
    // would make every claim below vacuous.
    assert!(
        referenced > BIG_BUDGET_BYTES * 4,
        "the fixture references {referenced} B against a {BIG_BUDGET_BYTES} B \
         pool — not 'severalfold', so nothing below is an over-budget test"
    );

    let run = scripted_run(&gpu, &content, BIG_BUDGET_BYTES, PATH_FRAMES);

    // 1. INSIDE THE POOL. The slot count is what the budget bought, and the
    // residency never exceeded it — the invariant the whole phase exists for.
    assert!(
        u64::from(run.slots) * inf_vt::PageFormat::Rgba8.page_bytes(inf_vt::STORED_TILE_SIZE)
            <= BIG_BUDGET_BYTES,
        "the pool planner granted {} slots, which is more than the budget",
        run.slots
    );
    assert!(
        run.peak_resident <= run.slots,
        "residency peaked at {} of {} slots",
        run.peak_resident,
        run.slots
    );
    // …and it really filled up, or "stayed inside the budget" is a statement
    // about a streamer that never streamed.
    assert!(
        run.peak_resident * 10 >= run.slots * 9,
        "the pool peaked at {} of {} slots — the fixture never pressed it, so \
         the bound above is not a measurement",
        run.peak_resident,
        run.slots
    );
    assert!(
        run.pop_in.deferred > 0,
        "nothing was ever deferred under a pool six times too small: {}",
        run.pop_in.summary()
    );

    // 2. EVERY SAMPLE LANDS ON THE FINEST RESIDENT ANCESTOR. `resolve` reads the
    // maintained table — exactly what the shader reads — and three things have
    // to hold of what it names: it is an ANCESTOR of the address (below), it is
    // RESIDENT, and nothing FINER is resident. The last two are asserted inside
    // `scripted_run`, where the residency is still alive; this counter is their
    // anti-vacuity, because an assertion in a helper that walked zero addresses
    // is an assertion about nothing.
    //
    // The P26.5 audit is why they exist at all: `desc.ancestor(at, got.mip) ==
    // got` alone is satisfied by a `resolve` that names the always-pinned ROOT
    // for every address in the pyramid — measured, and it passed this whole arm
    // while failing three `inf-vt` arms. A gate must aim at the thing it names
    // (the P23 law), and this one is named after the phase's own "Done when".
    assert!(
        run.addresses_checked >= run.fallbacks.len(),
        "the finest-ancestor walk checked {} addresses against {} recorded \
         fallbacks",
        run.addresses_checked,
        run.fallbacks.len()
    );
    // Six 512² pyramids, four tiles a side at mip 0 — 16 each, 96 in all.
    // Exact rather than "> 0": the walk is over the whole level and a walk that
    // quietly visited one texture would satisfy every assertion in it.
    let want_addresses = BIG_TEX.len() * ((BIG_EXTENT / 128) * (BIG_EXTENT / 128)) as usize;
    assert_eq!(
        run.addresses_checked, want_addresses,
        "the finest-ancestor walk visited {} mip-0 addresses, not the {} the \
         fixture has — the residency assertions inside `scripted_run` did not \
         cover the level",
        run.addresses_checked, want_addresses
    );
    assert!(
        !run.fallbacks.is_empty(),
        "no mip-0 address fell back at all under a pool six times too small, so \
         this arm never exercised the fallback"
    );
    let order = content.registration_order();
    let (lib, _pools, _r) = inf_render::build_vt_level(
        &gpu.device,
        &gpu.queue,
        &inf_render::RenderSettings::default(),
        BIG_BUDGET_BYTES,
        &content.vt_materials(),
        |g| content.source(g),
    )
    .expect("the level builds");
    let descs: BTreeMap<Uuid, inf_vt::VtTextureDesc> = order
        .iter()
        .filter_map(|g| {
            lib.handle(g.as_u128())
                .and_then(|h| lib.residency().desc(h).cloned())
                .map(|d| (*g, d))
        })
        .collect();
    for (guid, at, got) in &run.fallbacks {
        let desc = descs.get(guid).expect("a registered texture");
        assert!(got.mip > at.mip, "a 'fallback' resolved to its own level");
        assert_eq!(
            desc.ancestor(*at, got.mip),
            Some(*got),
            "{guid} {at:?} resolved to {got:?}, which is not its ancestor at that \
             level — the shader walks the clamped chain and would read another \
             tile's texels"
        );
    }
}

// ── (d) engagement counters ─────────────────────────────────────────────────

/// **(d) The counters move on a VT scene and are zero on a textureless one**
/// (P26.5), anti-vacuity first.
///
/// The P20 law: a claim about the command stream needs an engagement counter,
/// not a pixel. And the P26.3 audit's correction to it — *"the engagement-counter
/// arm asserted zero twice without ever drawing a frame with a pool"* — so the
/// moving half comes first here, and the zero half is what it is compared
/// against.
#[test]
fn the_engagement_counters_move_on_a_vt_scene_and_are_zero_without_one() {
    let Some(gpu) = gpu_or_skip("the P26 engagement counters") else {
        return;
    };
    let tmp = tempfile::tempdir().expect("tempdir");
    let (content, _) = big_pack(tmp.path());

    // FIRST: they move.
    let run = scripted_run(&gpu, &content, BIG_BUDGET_BYTES, PATH_FRAMES);
    assert_eq!(run.engaged_frames, PATH_FRAMES, "a frame drew with no pool");
    assert!(run.pop_in.admits > 0, "{}", run.pop_in.summary());
    assert!(run.pop_in.frames == PATH_FRAMES, "{}", run.pop_in.summary());

    // THEN: a textureless scene touches nothing. Same renderer, same frames, no
    // VT level at all — which is the state all 50 goldens record.
    let target = inf_render::HeadlessTarget::new(&gpu, 320, 180);
    let mut renderer = inf_render::EngineRenderer::new(&gpu, inf_render::HEADLESS_FORMAT);
    let mut scene = inf_render::RenderScene::default();
    scene.instances.push(inf_render::MeshInstance {
        vt: inf_render::VtTextureSet::NONE,
        translation: glam::DVec3::ZERO,
        rotation: glam::Quat::IDENTITY,
        scale: glam::Vec3::splat(2.0),
        color: [1.0, 1.0, 1.0, 1.0],
        metallic: 0.0,
        roughness: 0.5,
        emissive: [0.0; 3],
        id: 1,
        mesh: inf_render::PrimMesh::Cube,
        blend: 0,
        cutoff: 0.5,
    });
    scene.mark_dirty();
    for step in 0..PATH_FRAMES {
        renderer.render(&gpu, &scene, &path_view(step), &target.view, (320, 180));
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    }
    assert_eq!(
        renderer.vt_engaged_frames(),
        0,
        "a textureless scene engaged virtual texturing"
    );
    assert_eq!(
        renderer.vt_pop_in(),
        VtPopIn::default(),
        "a textureless scene moved a streaming counter — the command stream the \
         goldens recorded is not the one being drawn"
    );
    assert!(
        renderer.vt_summary().is_none(),
        "a textureless renderer produced a virtual-texture summary"
    );
}

// ── (e) budgets ─────────────────────────────────────────────────────────────

/// **(e) The level build is inside the LOAD budget, one frame's streaming work
/// is inside its page ceiling, and the frame's milliseconds are inside the FRAME
/// budget where a millisecond means something** (P26.5, rescoped 2026-08-13).
///
/// Three classes now, and keeping them apart is the P20 law taken one step
/// further than P26.5 took it. The law says *"loads assert `LOAD_BUDGET_MS`,
/// never `FRAME_BUDGET_MS`"*, because a load happens once and a frame happens
/// thirty times a second. The same reasoning splits the frame half in two: what
/// one frame's streaming loop **does** is a world fact, and what it **costs in
/// milliseconds** is a fact about the machine that ran it.
///
/// * **LOAD** — registry + pool, once, against [`LOAD_BUDGET_MS`]. Everywhere.
/// * **WORLD** — what one frame's loop did, in two numbers, everywhere, because
///   a page is a page on every adapter and arm (a) proves both sequences are a
///   pure function of committed input. This is the half with teeth in CI.
///   Pages **admitted** per steady frame against
///   [`inf_player::budget::VT_ADMITS_PER_FRAME_CEILING`] (`VtPools::apply`
///   issues exactly one `write_texture` per admit, so it is the upload count),
///   and **wants offered** per frame against
///   [`inf_player::budget::VT_WANTS_PER_FRAME_CEILING`] — the second because
///   admits are clamped by the pool, so a scan that walked the whole pyramid
///   would barely move them (measured: +4 admits, +30 wants, +218 deferrals).
/// * **CLOCK** — the steady-state mean, against
///   [`inf_player::budget::VT_STREAM_STEP_BUDGET_MS`] and [`FRAME_BUDGET_MS`],
///   **only on an adapter whose timing represents something**. Printed on every
///   adapter, always.
///
/// # Why the clock is conditional, measured rather than argued
///
/// The first version of this arm asserted the mean frame time unconditionally
/// and went red on `macos-latest` at **49.55 ms** against a 33 ms budget, with
/// nothing regressed: the runner's adapter is the "Apple Paravirtual device"
/// that `inf-render`'s `frame_budget.rs` has named by that string since P15.1,
/// and every other wall-clock arm in the tree — `frame_budget.rs` (five sites),
/// `vgeom_streaming::streaming_overhead_is_bounded`,
/// `phase18_gate::the_composed_frame_stays_inside_the_frame_budget`,
/// `mvs_gate::parity_is_strict` — already declines to assert on it. This arm was
/// the outlier, and the number it asserted was never the streaming loop's: it is
/// `render` plus a blocking pump to idle, i.e. a whole GPU frame, on a
/// virtualized GPU, held against a budget derived on a discrete one. `budget.rs`
/// says §8 numbers *"are not hardware claims"*; this one was.
/// `docs/memos/p26-frame-budget-scope.md` carries the ruling and the numbers.
///
/// # And why the cold frame is separated
///
/// Frame 0 admits the whole analytic floor into an empty pool and builds
/// whatever the renderer builds lazily; frames 1..n admit a handful. A mean over
/// the two describes neither. `vgeom_streaming` set that precedent
/// ("the cold frame is measured separately and printed rather than asserted…
/// because a regression that moved work from the cold frame into the steady
/// state would otherwise look like an improvement"), and both numbers are
/// printed here for the same reason.
///
/// This is still the P26.4 remainder discharged: *"the feedback's own budget is a
/// page cap, not a millisecond cap. `VT_FEEDBACK_MAX_TILES` and
/// `VT_FEEDBACK_REQUEST_CAP` bound the work; what the sync costs in LOAD-class
/// milliseconds is not yet ratcheted, and the `phase26_gate` budget arm is where
/// that lands."* It lands as two ratchets rather than one, and the page-shaped
/// one is the portable half.
#[test]
fn the_streaming_loop_stays_inside_its_budgets() {
    let Some(gpu) = gpu_or_skip("the P26 budgets") else {
        return;
    };
    // The classification `frame_budget.rs` has carried since P15.1, spelled the
    // same way (a CPU rasterizer and a paravirtualized GPU have
    // non-representative, run-to-run-noisy timing).
    let info = gpu.adapter.get_info();
    let virtualized = {
        let n = info.name.to_ascii_lowercase();
        n.contains("paravirtual") || n.contains("virtualbox") || n.contains("vmware")
    };
    let software = info.device_type == wgpu::DeviceType::Cpu || virtualized;

    let tmp = tempfile::tempdir().expect("tempdir");
    let (content, _) = big_pack(tmp.path());

    // ── LOAD class: registry + pool, once ───────────────────────────────────
    let t0 = std::time::Instant::now();
    let built = inf_render::build_vt_level(
        &gpu.device,
        &gpu.queue,
        &inf_render::RenderSettings::default(),
        BIG_BUDGET_BYTES,
        &content.vt_materials(),
        |g| content.source(g),
    );
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
    assert!(built.is_some(), "the level did not build");
    println!(
        "phase26 budgets: level build {load_ms:.2} ms (load budget {} ms)",
        LOAD_BUDGET_MS
    );
    assert!(
        load_ms < LOAD_BUDGET_MS,
        "building the virtual-texture level took {load_ms:.2} ms, over the \
         {LOAD_BUDGET_MS} ms load budget {}",
        inf_player::budget::RATCHET_NOTE
    );
    drop(built);

    // The scripted path, timed **per frame** — `render` plus the poll, and
    // nothing else. The renderer's construction compiles every shader in the
    // tree and the registry build is the LOAD-class number above; timing either
    // as part of a frame is how a budget arm comes to report 177 ms and mean
    // nothing (measured, before this comment existed).
    let run = scripted_run(&gpu, &content, BIG_BUDGET_BYTES, PATH_FRAMES);
    // The same path with **no virtual texturing at all** — the control that says
    // how much of a frame the streaming loop is, on whatever adapter is running
    // it. The P18 gate's (e) precedent: measure a subsystem against the same
    // scene with it off, rather than against an adjective.
    let control = vt_free_control(&gpu, PATH_FRAMES);
    let control_steady = mean_of_tail(&control);

    println!(
        "phase26 budgets on {} ({:?}){}:\n  \
         cold frame (the floor, into an empty pool) {:.2} ms\n  \
         steady mean over {} frames {:.2} ms (all {} frames: {:.2} ms)\n  \
         the same path with no VT at all {control_steady:.2} ms \
         (streaming is {:+.2} ms of it)\n  \
         {} admits, {} deferred; {} on the cold frame, peak {} on a steady one \
         (per frame: {})\n  \
         wants peak {} in one frame (per frame: {})\n  \
         budgets: frame {FRAME_BUDGET_MS} ms, streaming ratchet {} ms, \
         steady admits/frame ceiling {}, wants/frame ceiling {}",
        info.name,
        info.device_type,
        if software {
            " [virtual/software — the clock is reported, not asserted]"
        } else {
            ""
        },
        run.cold_ms(),
        PATH_FRAMES - 1,
        run.steady_ms(),
        PATH_FRAMES,
        run.mean_ms(),
        run.steady_ms() - control_steady,
        run.pop_in.admits,
        run.pop_in.deferred,
        run.cold_admits(),
        run.peak_steady_admits(),
        run.admits_trace(),
        run.peak_wants(),
        run.wants_trace(),
        inf_player::budget::VT_STREAM_STEP_BUDGET_MS,
        inf_player::budget::VT_ADMITS_PER_FRAME_CEILING,
        inf_player::budget::VT_WANTS_PER_FRAME_CEILING,
    );

    // ── WORLD class: what one frame's streaming loop DID ─────────────────────
    // ANTI-VACUITY FIRST: the frames did streaming work, or every bound below is
    // a bound on an empty loop.
    assert!(
        run.pop_in.admits > 0 && run.pop_in.deferred > 0,
        "the path admitted {} and deferred {} — nothing to bound: {}",
        run.pop_in.admits,
        run.pop_in.deferred,
        run.pop_in.summary()
    );
    assert_eq!(
        run.admits_each.iter().sum::<u64>(),
        run.pop_in.admits,
        "the per-frame admits do not sum to the streamer's own total, so the \
         ceiling below is being asserted against a bookkeeping bug"
    );
    // …and no steady frame asked the queue for more pages than a frame may.
    // This is the assertion that survives on every adapter, and it is a
    // measurement of the engine rather than of the runner: one admit is one
    // `write_texture` of one page. The cold frame is excluded and printed — it
    // admits the whole floor the pool can hold, and a ceiling that had to clear
    // it could not see a loop re-admitting that same pool on every frame after.
    assert!(
        run.peak_steady_admits() <= inf_player::budget::VT_ADMITS_PER_FRAME_CEILING,
        "a steady frame admitted {} pages, over the {} the streaming loop is \
         allowed per frame {} (per frame: {})",
        run.peak_steady_admits(),
        inf_player::budget::VT_ADMITS_PER_FRAME_CEILING,
        inf_player::budget::RATCHET_NOTE,
        run.admits_trace()
    );
    // ANTI-VACUITY for the ceiling itself: the steady frames really did admit
    // pages, or the bound above is a bound on zero.
    assert!(
        run.peak_steady_admits() > 0,
        "every admit of the run happened on the cold frame, so the steady-state \
         ceiling is asserted against nothing (per frame: {})",
        run.admits_trace()
    );
    // …and the SCAN that asked for them stayed bounded. Admits alone cannot say
    // this: they are clamped by the pool, so a want set that walked the whole
    // pyramid would show up as a handful more admits and a mountain of
    // deferrals. Measured (mutation): `justified_mip` two levels too fine moves
    // the peak admits 6 → 10, inside any ceiling a 6 justifies, and the peak
    // wants 36 → 66 with deferrals 8 → 226. This is the arm's answer to the
    // regression `inf_player::budget` names by name.
    assert!(
        run.peak_wants() <= inf_player::budget::VT_WANTS_PER_FRAME_CEILING,
        "one frame offered {} wants, over the {} a bounded scan produces {} \
         (per frame: {})",
        run.peak_wants(),
        inf_player::budget::VT_WANTS_PER_FRAME_CEILING,
        inf_player::budget::RATCHET_NOTE,
        run.wants_trace()
    );
    assert!(
        run.peak_wants() > 0,
        "no frame offered a want at all: {}",
        run.wants_trace()
    );

    // ── CLOCK class: only where a millisecond represents a frame ─────────────
    if software {
        return;
    }
    let steady = run.steady_ms();
    assert!(
        steady < FRAME_BUDGET_MS,
        "a streamed frame cost {steady:.2} ms on {}, over the {FRAME_BUDGET_MS} \
         ms frame budget {}",
        info.name,
        inf_player::budget::RATCHET_NOTE
    );
    assert!(
        steady < inf_player::budget::VT_STREAM_STEP_BUDGET_MS,
        "a streamed frame cost {steady:.2} ms on {}, over the {} ms \
         virtual-texture streaming ratchet {}",
        info.name,
        inf_player::budget::VT_STREAM_STEP_BUDGET_MS,
        inf_player::budget::RATCHET_NOTE
    );
}

/// The mean of everything after the first element — a slice's steady state.
fn mean_of_tail(ms: &[f64]) -> f64 {
    let tail = &ms[1.min(ms.len())..];
    if tail.is_empty() {
        return 0.0;
    }
    tail.iter().sum::<f64>() / tail.len() as f64
}

/// The scripted path's frames with **no virtual texturing**: the same renderer,
/// the same three cubes, the same views, `VtTextureSet::NONE` and no level.
///
/// What it isolates is the whole point of measuring it: a frame's cost is the
/// renderer's pass stack plus the streaming loop, and only the second is this
/// phase's. On an adapter where the first is 40 ms the difference is the only
/// number that says anything about the second — which is exactly what the
/// `macos-latest` failure could not distinguish.
fn vt_free_control(gpu: &GpuContext, frames: u64) -> Vec<f64> {
    let target = inf_render::HeadlessTarget::new(gpu, 320, 180);
    let mut renderer = inf_render::EngineRenderer::new(gpu, inf_render::HEADLESS_FORMAT);
    let mut out = Vec::with_capacity(frames as usize);
    for step in 0..frames {
        let mut scene = inf_render::RenderScene::default();
        for i in 0..BIG_MAT.len() {
            scene.instances.push(inf_render::MeshInstance {
                vt: inf_render::VtTextureSet::NONE,
                translation: glam::DVec3::new(0.0, 0.0, -4.0 * i as f64),
                rotation: glam::Quat::IDENTITY,
                scale: glam::Vec3::splat(2.0),
                color: [1.0, 1.0, 1.0, 1.0],
                metallic: 0.0,
                roughness: 0.5,
                emissive: [0.0; 3],
                id: i as u32 + 1,
                mesh: inf_render::PrimMesh::Cube,
                blend: 0,
                cutoff: 0.5,
            });
        }
        scene.mark_dirty();
        let t = std::time::Instant::now();
        renderer.render(gpu, &scene, &path_view(step), &target.view, (320, 180));
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        out.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    assert_eq!(
        renderer.vt_pop_in(),
        VtPopIn::default(),
        "the VT-free control ran the streaming loop, so it is not a control"
    );
    out
}

// ── (f) the golden set ──────────────────────────────────────────────────────

/// **(f) The golden set is pinned, and it is additive only** (P26.5).
///
/// Phase 26 renders material textures into the interactive renderer for the
/// first time, and it re-blessed nothing: a textureless frame does not enter the
/// streaming loop at all, which is a claim about the command stream that
/// `vt_engaged_frames` and `VtPopIn::frames` both measure (arm (d)) and that this
/// counts.
///
/// # A count is not enough, and the P26.5 audit measured why
///
/// The first version of this arm counted, and justified counting like this:
/// *"what 'additive only' forbids is a golden changing, and a changed PNG is
/// what the harness's own comparison catches. What a comparison cannot catch is
/// a golden being deleted."* **Both halves are backwards**, and both were
/// measured rather than reasoned:
///
/// * a **changed** golden is not caught. `check_golden`'s pixel comparison runs
///   only under `INF_GOLDEN_STRICT`, which is opt-in precisely because exact
///   cross-adapter pixels differ — no battery sets it. Measured: `unlit.png`
///   overwritten with `voxel.png`'s bytes, `golden_unlit` green;
/// * a **deleted** golden is not caught for long either, because `check_golden`
///   writes any golden it cannot read (`read_png(&path).is_none() =>
///   write_png`). Measured: `rm unlit.png`, run the golden suite, the file is
///   back and the count is 50 again. Deleting a golden is therefore a *silent
///   re-bless from the current code* — which is the shape of the thing the
///   ledger promises never happened. The count only bites in the window before
///   `golden.rs` runs, and they are separate test binaries with no ordering
///   between them.
///
/// So the set is pinned by CONTENT as well as by count. The digest fails on an
/// edited frame, on a re-bless, and on a delete-then-regenerate **whenever the
/// regenerated frame differs from the committed one** — which is exactly the
/// case that matters, since a deletion made to turn a red build green is a
/// deletion whose replacement will not match. (Measured on this machine: an
/// unchanged renderer regenerates `unlit.png` byte for byte, so the pin is quiet
/// about a deletion that changed nothing, and loud about one that did.)
///
/// Adding a golden is still allowed and still means moving a constant in the
/// same commit — two of them now, which is the honest price of the claim the
/// ledger makes about this phase.
#[test]
fn the_golden_set_is_pinned_and_additive() {
    /// The count P26.1 through P26.5 each carried forward untouched, and that
    /// P27.4 added four to — the virtual-shadow-map set (`vsm_directional`,
    /// `vsm_spot`, `vsm_point`, `vsm_bias_grazing`). P27.1 through P27.3 moved
    /// it by nothing, because virtual shadows were inert until a receiver
    /// existed. Wave VIS1a's **audit** added the fifty-fifth — `water_ssr.png`,
    /// the one frame the wave's own reflection feature can be pinned by.
    const GOLDENS: usize = 62;
    /// `xxh3_128` over `"{file_name} {hex}\n"` for every golden, name-sorted —
    /// the CONTENT pin (P26.5 audit). Committed PNGs are `-text` in
    /// `.gitattributes`, so these bytes are the same on every checkout.
    ///
    /// **RULE: this may change only in a commit that adds a golden, or in one
    /// whose stated purpose is to change what the engine LOOKS like** — never as
    /// a side effect, and never to turn a red build green. Phase 26 re-blessed
    /// none, and neither did P27.4 — `git status` over this directory across that
    /// batch reports four additions and nothing else.
    ///
    /// **Moved once, at wave SKY2** (from `23d41a61c31c28a17a20871b6c875707`),
    /// and the second half of the rule above is the amendment that wave had to
    /// write. SKY2 is the volumetric-cloud overhaul: its deliverable IS the look,
    /// so eight frames with clouds in them — the four `clouds_*`, plus
    /// `editor_default` and the three `weather_*` presets — were re-blessed on
    /// purpose, with the differences described one by one in the commit message
    /// and in `docs/memos/island-progress.md` under *Wave SKY2*. The count did
    /// not move.
    ///
    /// This gate is why the eighth frame is on that list. The wave's brief named
    /// five; `weather_fog_dawn`, `weather_snow_dusk` and `weather_storm_noon`
    /// enable clouds through a `WeatherState` preset and were not on it. A
    /// count-only pin would have said nothing about any of the three.
    ///
    /// **Moved a second time, at wave VIS1a** (from
    /// `7ff2b3702b825f00707a71dff282f400`), for **one** frame: `ssao.png`. That
    /// wave replaces the ambient-occlusion estimator — a hemisphere-kernel
    /// sample-count for GTAO's horizon-search visibility integral, and a 4×4 box
    /// blur for a depth-aware bilateral one — so the committed frame of the SSAO
    /// scene is a picture of an estimator that no longer exists.
    ///
    /// The difference is **mean 0.001456 / max 0.038588**, measured against the
    /// previous frame before it was replaced, and it is *inside* the harness's own
    /// perceptual tolerance — which is exactly why it was re-blessed rather than
    /// left alone. A golden that no longer depicts what the engine draws is a
    /// golden nobody can read a regression off. Every other one of the 54 was
    /// **0.000000 / 0.000000**, printed by the harness and checked one by one.
    ///
    /// **And a third time, in the same wave** (from
    /// `838d18fbffeea0c43ea5d84a8f5fbc63`), for **28** frames — the one commit in
    /// VIS1a whose stated purpose is the look. The GGX specular lobe gained
    /// **multi-scatter energy compensation**: a single-scatter GGX accounts for
    /// the microfacets the Smith term masks and then drops the light they masked,
    /// which the furnace test measures at up to **55 %** of the lobe, and which a
    /// metal — having no diffuse term to hide it — wears as being systematically
    /// too dark. Every frame with a lit surface in it therefore moves, by a little
    /// (mean ≤ **0.000252**, on `weather_fog_dawn`) or, where there is a metal, by
    /// more (max **0.031216**, on `pbr_materials` — the metallic/roughness grid,
    /// which is the frame this change exists for). The other 26 are byte-identical.
    ///
    /// Both moves are inside one wave and each has its own commit; neither turned
    /// a red build green.
    ///
    /// **And a fourth time, in that wave's audit** (from
    /// `84e38ed2a762a2e9a0cca84d0fc80b8b`), on the **additive** branch of the rule
    /// rather than the re-bless one: `water_ssr.png` is a *new* frame, `GOLDENS`
    /// moves 54 → 55 beside it, and not one committed image changed. It exists because VIS1a shipped carrying "no golden can ever
    /// capture SSR" — which is true of the **opaque** path only. Water marches
    /// against its own private same-frame resolve and needs no colour history, so
    /// the boat reflecting itself is a deterministic one-frame image, and the
    /// wave's signature feature now has a pixel pin instead of a sentence saying
    /// it could not have one.
    ///
    /// **And a fifth time, wave VIS1b, additive again** (from
    /// `d6da45fdbc8ef842fcea3acf10c8bf84`): `sun_flare.png`, the sun glare /
    /// ghost chain / halo / anamorphic streak, at dawn with the disc low in a dim
    /// sky. `GOLDENS` moves 55 -> 56 beside it and not one committed image
    /// changed -- the flare is off by default, off is a clear of its own half-res
    /// target, and the tonemap's add sits behind a uniform branch it does not
    /// take.
    ///
    /// **And a sixth, still additive, in the same wave**: `lens_trio.png` -- the
    /// vignette, the chromatic aberration and the film grain on one high-contrast
    /// scene. `GOLDENS` moves 56 -> 57, every `FilmSettings` field is zero at the
    /// default, and the tonemap's three branches are not taken there.
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
    // `gi_bleed`, `gi_terrain` and `gi_specular` are byte-identical through it.
    const GOLDEN_SET_DIGEST: &str = "51c225fa4f1afc63d4fafd5ff3f31d46";
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
         means moving this constant in the same commit; losing one is not.",
        pngs.len()
    );
    // …and every one of them is a real image rather than a truncated file that
    // would compare equal to nothing.
    for p in &pngs {
        let len = std::fs::metadata(p).expect("stat").len();
        assert!(len > 256, "{} is {len} B — not a golden", p.display());
    }

    // THE CONTENT PIN. Name and bytes, in name order, folded into one digest —
    // so a re-blessed frame, a silently regenerated one and a deleted one are
    // three ways of failing the same assertion, none of which the harness's own
    // opt-in comparison sees.
    let mut manifest = String::new();
    for p in &pngs {
        let bytes = std::fs::read(p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
        manifest.push_str(&format!(
            "{} {}\n",
            p.file_name().expect("a file name").to_string_lossy(),
            inf_asset::ContentHash::of(&bytes).to_hex()
        ));
    }
    let digest = inf_asset::ContentHash::of(manifest.as_bytes()).to_hex();
    assert_eq!(
        digest, GOLDEN_SET_DIGEST,
        "the golden set's CONTENT moved. A golden was re-blessed, deleted (the \
         harness regenerates any golden it cannot read, so the count above still \
         matches), or replaced. Adding one is allowed and means moving both constants \
         in the same commit. Changing an existing frame is allowed ONLY in a \
         commit whose stated purpose is to change the look, with the difference \
         described — never as a side effect of something else.\n{manifest}"
    );
}
