//! **The Infini side of the Unreal bridge** (wave ASSET0, clause 2): a
//! `manifest.json` written by `tools/ue-export/export.py` → `.inf_tex`,
//! `.inf_mat` and `.inf_mesh` assets in a project's `Content`.
//!
//! # One door, not a second importer
//!
//! Meshes go through [`super::import::import_file`] — the same call the Content
//! Drawer's drag-and-drop makes — because a bridge that decoded glTF itself
//! would be a second producer of `.inf_mesh` bytes and the two would agree only
//! until one was touched. What this module adds is everything glTF cannot
//! carry: which of five loose PNGs is a roughness map, that a Megascans surface
//! has no ORM and one has to be *packed*, which material is a tiling surface
//! rather than a mesh's skin, and where a light sits on a lamp post.
//!
//! # The PBR remap, and why it is not a pass-through
//!
//! Every Megascans instance in the reference project parents
//! `Standard_MasterMaterial` and names four slots: `albedo`, `normal`,
//! `roughness`, `displacement`. **There is no ORM anywhere** — and this engine's
//! `.inf_mat` has one `metallic_roughness_texture`, glTF-channel-ordered, which
//! is what `vt_sample.wgsl` reads. So the import PACKS one:
//! occlusion → R, roughness → G, metallic → B, through
//! [`inf_material::pack_orm`], with 255/255/0 standing in for a map the pack
//! does not ship. Downtown_West ships a real AO and a real metallic; the
//! Megascans surfaces ship neither, and both import correctly.
//!
//! [`inf_material::pack_orm`] has existed since Wave T with **no caller at
//! all**. This is its first.
//!
//! Its sibling [`inf_material::plan_map_set`] still has none, and deliberately:
//! it recovers a map's role from a FILENAME, which is the right door for a
//! folder of loose Megascans files dragged into the Content Drawer and the wrong
//! one here — the manifest states every role explicitly, so planning by name
//! would be guessing at something already known. Said rather than left as an
//! absence, because "the planner has no caller" is a fact somebody will check.
//!
//! # The clamp
//!
//! The bridge exports at source resolution, which for most of these surfaces is
//! 8 192 square — 268 MB of RGBA a map. [`UeImportOptions::max_texture`] halves
//! through the mip chain's own box filter before the tiler and the BC encode
//! run, which is where nearly all of an import's time and disk goes.
//!
//! # The REBIND, and why a material can be written at somebody else's GUID
//!
//! An imported surface is worth nothing to a level that does not name it, and
//! the levels this repository commits name the **ground library's** GUIDs — the
//! island's four `TerrainLayer::material`s and, since clause 0, its `Roads`
//! entity's `Material::asset`. Those levels are committed and their bytes are
//! locked, and the content they would have to name is licensed content that
//! must never enter this repository.
//!
//! So the bridge writes the imported material **at the committed GUID**, into
//! the local project only. `samples/ground/Road_Asphalt.inf_mat` (synthesised,
//! committed, licence-free) and `Content/Road_Asphalt.inf_mat` (Megascans, local,
//! never committed) are the same asset identity with different texels, and the
//! island level does not know or care which one it got. That is what makes the
//! public repository buildable by anyone and this machine's build photoreal.
//!
//! # …and the law that arrangement rests on is enforced here
//!
//! [`import_manifest`] **refuses a destination inside the engine checkout**,
//! before it decodes anything — see [`engine_checkout_above`]. The audit found
//! the rule stated in three places and checked in none.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use inf_asset::{AssetError, AssetId, Result};
use inf_material::{MapKind, MatBlend, MaterialAsset, TextureImportSettings};
use serde::Deserialize;

use super::AssetProject;

/// How to run one manifest import.
#[derive(Debug, Clone)]
pub struct UeImportOptions {
    /// Pack names to import. Empty imports every pack the manifest carries.
    pub packs: Vec<String>,
    /// Ceiling on a texture's longest side. `0` keeps the source resolution.
    pub max_texture: u32,
    /// Subfolder of the project's `Content` the assets land in.
    pub dest: String,
    /// `(asset stem, manifest material key)` — write the imported material at
    /// the GUID the committed library assigns that stem. See the module note.
    pub rebinds: Vec<(String, String)>,
    /// Import meshes as well as materials. Meshes are the slow half and a
    /// materials-only run is the common one.
    pub meshes: bool,
    /// How many LOD rungs of a **character** to store (wave CHAR1a).
    ///
    /// Three, and the number is a consequence rather than a taste: a skinned
    /// mesh never reaches the meshlet path (`MeshAsset::vgeom_streams` drops the
    /// skin stream), so unlike a rigid mesh a character's ladder has to be
    /// stored rung by rung, and every stored rung is a whole `.inf_mesh` in the
    /// pack. Three is what the crowd tiers can actually select between --
    /// `CrowdTier::{Full, Near, Far}` -- so a fourth would be bytes nothing asks
    /// for.
    pub character_lods: usize,
    /// The `/Game/...` object path of the skeletal mesh whose rig every imported
    /// **clip** is retargeted onto. `None` uses the first skeleton the run
    /// imports, which is right when the manifest carries one body.
    pub retarget_to: Option<String>,
    /// The manifest key of the skeletal mesh to write **at the starter
    /// character's committed GUIDs** — the REBIND, for a body.
    ///
    /// # Why a body needs the same door a road surface needed
    ///
    /// The island's hero entity names three fixed GUIDs (`0x5C10_00A0 + 0/1/2`
    /// — the starter rig, its skin material and its body mesh), the level that
    /// names them is committed and byte-locked, and the body a demo wants to
    /// show is licensed content that may never enter this repository. Exactly
    /// the arrangement clause 0 of ASSET0 solved for the road: write the
    /// imported asset **at the committed identity**, into the local project
    /// only. `samples/starter-character/Starter_Body.inf_mesh` (our own,
    /// committed, licence-free) and `Content/UE/.../Starter_Body.inf_mesh`
    /// (Unreal's mannequin, local, never committed) become the same asset
    /// identity with different vertices, and the island level does not know or
    /// care which one it got.
    ///
    /// Three assets move together or none do, because a mesh whose joint
    /// indices address one rig cannot be posed by another: the LOD-0 mesh, the
    /// skeleton it was skinned to, and the material in its first slot.
    pub rebind_character: Option<String>,
    /// The manifest key of a SECOND skeletal mesh to write at the **female**
    /// starter character's committed GUIDs (wave CHAR1a.3).
    ///
    /// # Why a second target rather than a second run
    ///
    /// The island's crowd wears whatever `(mesh, skeleton, machine)` triples the
    /// LEVEL's own entities carry (`inf_ecs::society::level_archetypes`), and the
    /// second committed body — `samples/starter-character-f`, whose GUIDs are
    /// `0x5C10_00B0 + n` — is the one the demo loop places to give the crowd a
    /// plural wardrobe. Rebinding it in the same run as the hero's is what makes
    /// the two MetaHumans a MALE and a FEMALE default rather than one body
    /// twice: two keys, two identities, one import.
    pub rebind_character_f: Option<String>,
}

impl Default for UeImportOptions {
    fn default() -> Self {
        Self {
            packs: Vec::new(),
            // 2 048, and it is a measurement rather than a round number: the
            // Megascans surfaces here tile at 2-4 m, so 2 048 is a 1-2 mm texel
            // — the same class the committed ground library spends 1 024 to
            // reach at half the tile size, and finer than the 3.9 mm the
            // synthesised asphalt it replaces achieves.
            max_texture: 2048,
            dest: "UE".to_string(),
            rebinds: Vec::new(),
            meshes: true,
            character_lods: 3,
            retarget_to: None,
            rebind_character: None,
            rebind_character_f: None,
        }
    }
}

/// What one manifest import produced.
#[derive(Debug, Clone, Default)]
pub struct UeImportReport {
    /// `(manifest key, asset)` per `.inf_mat` written.
    pub materials: Vec<(String, AssetId)>,
    /// `(manifest key, asset, rungs the pack shipped, triangles imported)`.
    pub meshes: Vec<(String, AssetId, usize, usize)>,
    /// Every `.inf_tex` written.
    pub textures: Vec<AssetId>,
    /// `(stem, asset)` per rebind performed — the committed GUIDs now carrying
    /// imported texels in this project.
    pub rebinds: Vec<(String, AssetId)>,
    /// The light fixtures the manifest carried, converted to this engine's frame.
    pub fixtures: Vec<UeFixture>,
    /// Non-fatal notices, in the P16 cook-advisory shape.
    pub advisories: Vec<String>,
    /// `(manifest key, mesh asset, skeleton asset, rungs, triangles, joints)`
    /// per skeletal mesh imported (wave CHAR1a). The skeleton is `None` when the
    /// glTF carried a mesh with no skin, which is a defect worth seeing rather
    /// than a shape to tolerate.
    pub skeletal: Vec<(String, AssetId, Option<AssetId>, usize, usize, usize)>,
    /// `(manifest key, clip asset, tracks after retarget)` per clip imported.
    pub clips: Vec<(String, AssetId, usize)>,
    /// **The per-pack licence positions this import relied on**, carried out of
    /// the manifest so a caller can write them into its own ledger rather than
    /// re-deriving them: `(pack, licence text, may-ship)`.
    pub licences: Vec<(String, String, bool)>,
    /// Bytes of `.inf_tex` + `.inf_mat` + `.inf_mesh` written.
    pub bytes: u64,
    /// **Which pack each written asset came from** (wave CHAR1a.3, carried 96),
    /// so the licence position can be stamped onto the asset ON DISK rather than
    /// only printed.
    ///
    /// Collected as the import goes because that is the only place the answer is
    /// known: a `.inf_tex` is written inside `import_material` and has no key of
    /// its own, and a run that imports two packs cannot recover which is which
    /// from the finished report.
    pub asset_packs: Vec<(AssetId, String)>,
}

/// A prop's light, in **this engine's units and frame**.
#[derive(Debug, Clone, PartialEq)]
pub struct UeFixture {
    /// The Blueprint it came from.
    pub name: String,
    /// The mesh key the light hangs off, when the Blueprint had one.
    pub mesh: Option<String>,
    /// Offset from the prop's origin, **metres**, in `+X east, +Y up, -Z north`.
    pub offset_m: [f64; 3],
    /// The lamp's colour, **as UE stores it: 8-bit sRGB**.
    ///
    /// Not converted. The transfer function is a `powf`, and this crate's
    /// `portable_math_law` gate refuses one — correctly, because the engine
    /// already has exactly one sRGB decode (`inf_material::ground`'s sqrt
    /// ladder, transcendental-free so committed texels are byte-identical on
    /// every platform) and a second approximation here would be a second
    /// answer to one question. A bridge carries the source value; the
    /// conversion belongs where a `Light` is authored from it, next to
    /// whatever door that authoring path already uses.
    pub color_srgb8: [u8; 3],
    /// Range in metres (UE's attenuation radius).
    pub range_m: f32,
    /// Candela. See [`ue_intensity_to_candela`].
    pub intensity: f32,
}

/// **The engine checkout `path` sits inside, if it sits inside one** (ASSET0
/// audit).
///
/// A directory is this engine's checkout when it holds **both** a `.git` and
/// `tools/ue-export/export.py` — the second marker on purpose, because it is
/// the very script that produces the bytes this law is about, and a user's own
/// game repository must not be mistaken for ours.
///
/// # Why a mechanism and not a sentence
///
/// The wave's licence law — *nothing derived from the reference project's
/// marketplace, Fab or Megascans content may enter this repository* — was
/// upheld by three sentences of documentation and the author's care.
/// Measured at the audit: `export.py` takes its output directory from
/// `INF_UE_OUT` and `inf-import` takes its destination from `--into`, and
/// **neither refused a path inside the checkout**. A single mistyped
/// destination puts 4.1 GB of Megascans PNGs, or 892 MB of `.inf_tex`
/// converted from them, in the working tree of a PUBLIC repository, untracked
/// and one `git add -A` from being published. A law with no door that can say
/// no is a preference.
pub fn engine_checkout_above(path: &Path) -> Option<PathBuf> {
    let mut p: PathBuf = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir().ok()?.join(path)
    };
    loop {
        if p.join(".git").exists() && p.join("tools/ue-export/export.py").is_file() {
            return Some(p);
        }
        if !p.pop() {
            return None;
        }
    }
}

/// **UE centimetres, Z up, left handed → Infini metres, Y up, right handed.**
///
/// One function, because the conversion is the bridge's single most reversible
/// mistake: `(x, y, z)_ue → (x/100, z/100, -y/100)`. UE's own glTF exporter
/// applies exactly this to the geometry (0.01 scale, Y and Z swapped, one axis
/// negated for handedness), so a fixture converted here lands where the mesh
/// beside it landed — which is the property [`UeImportReport::fixtures`] is
/// checked against, rather than asserted.
pub fn ue_cm_to_world_m(cm: [f64; 3]) -> [f64; 3] {
    [cm[0] / 100.0, cm[2] / 100.0, -cm[1] / 100.0]
}

/// UE's point-light `Intensity` (lumens by default) → candela.
///
/// A point light radiates over 4π steradians, so `cd = lm / 4π`. UE's default
/// unit for a `PointLightComponent` is lumens and the reference lamp posts carry
/// 7 500 of them, which is 597 cd — a street lamp. Naming the conversion is the
/// difference between a lamp and a floodlight; the reference project's own
/// number is unusable without it.
pub fn ue_intensity_to_candela(lumens: f32) -> f32 {
    lumens / (4.0 * std::f32::consts::PI)
}

// ── the manifest, as this side reads it ──────────────────────────────────────
//
// Only the fields the import uses, every one `#[serde(default)]`: the manifest
// is written by a script in another repository's language and a field it grows
// must not fail an import that does not read it.

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
struct Manifest {
    schema_version: u32,
    packs: Vec<Pack>,
    meshes: Vec<Mesh>,
    /// v2 (wave CHAR1a): skeletal meshes, each with its own LOD ladder.
    skeletal_meshes: Vec<SkeletalMesh>,
    /// v2 (wave CHAR1a): animation clips, exported one glTF each.
    clips: Vec<Clip>,
    materials: Vec<Material>,
    textures: Vec<Texture>,
    fixtures: Vec<Fixture>,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
struct Pack {
    name: String,
    license: String,
    /// v2: whether this pack's licence permits SHIPPING the content, as opposed
    /// to using it as a local reference. Recorded per pack because the three
    /// character packs differ: ALS is MIT (ship), the mannequins are Epic's
    /// UE-Only content (reference), MetaHumans ship in a cooked pack and are
    /// never committed.
    ship: bool,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
struct SkeletalMesh {
    key: String,
    pack: String,
    source: String,
    skeleton: Option<String>,
    bones: u32,
    lods: Vec<SkelLod>,
    material_slots: Vec<Option<String>>,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
struct SkelLod {
    level: u32,
    file: Option<String>,
    /// Joints in the exported skin — read off the written glTF, not asserted.
    joints: u32,
    joint_names: Vec<String>,
    /// How many `JOINTS_n` attribute sets the exporter wrote. UE writes **two**
    /// (eight influences a vertex) and `.inf_mesh`'s `VertexSkin` holds four.
    influence_sets: u32,
    /// Triangles in the written glTF, counted off its accessors by the exporter.
    /// The number a ladder is chosen by — see [`distinct_rungs`].
    triangles: u32,
    primitives: u32,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
struct Clip {
    key: String,
    pack: String,
    name: String,
    source: String,
    file: Option<String>,
    skeleton: Option<String>,
    skeleton_bones: u32,
    seconds: f32,
    joints: u32,
    joint_names: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
struct Mesh {
    key: String,
    pack: String,
    lods: Vec<Lod>,
    material_slots: Vec<Option<String>>,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
struct Lod {
    level: u32,
    file: Option<String>,
    screen_size: f32,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
struct Material {
    key: String,
    pack: String,
    surface: bool,
    maps: BTreeMap<String, String>,
    base_color: [f32; 4],
    metallic: f32,
    roughness: f32,
    emissive: [f32; 3],
    opacity: f32,
    blend: String,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
struct Texture {
    key: String,
    file: Option<String>,
    map: String,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
struct Fixture {
    key: String,
    lights: Vec<Light>,
    meshes: Vec<FixtureMesh>,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
struct Light {
    location_cm: [f64; 3],
    color_srgb8: [u8; 3],
    intensity: f32,
    radius_cm: f32,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
struct FixtureMesh {
    mesh: String,
}

/// The manifest schema this build reads. A newer one is refused by name rather
/// than half-read: every field here is `default`, so a bump would otherwise
/// import an empty manifest and report success.
/// **v2 (wave CHAR1a)**: `skeletal_meshes` and `clips` joined the manifest.
///
/// The bump is not cosmetic. Every field on every manifest record is
/// `#[serde(default)]` — the deliberate choice at ASSET0, so a manifest that
/// grows a field does not fail an import that never reads it — and the exact
/// cost of that choice is that a v1 reader handed a v2 manifest imports the
/// meshes, ignores the two new **sections** entirely, and reports success
/// having imported no character at all. So the version gates the container and
/// a reader that grows an arm keys it on the version, which is the same law
/// `.ipack`'s header carries.
pub const MANIFEST_SCHEMA_VERSION: u32 = 2;

/// **Import one manifest.**
pub fn import_manifest(
    project: &mut AssetProject,
    manifest_path: &Path,
    opts: &UeImportOptions,
) -> Result<UeImportReport> {
    // **THE LICENCE LAW, AS A DOOR.** Before a single byte is decoded: this
    // bridge's output is licensed content, and the one place it may never land
    // is the public engine repository. See [`engine_checkout_above`].
    if let Some(root) = engine_checkout_above(project.root()) {
        return Err(AssetError::Import(format!(
            "refusing to import into {} — it is inside the engine checkout at \
             {}, and NOTHING this bridge writes may enter this repository. The \
             reference project's packs are Marketplace/Fab/Megascans content \
             whose licence for use outside Unreal is unestablished (see the \
             ASSET0 licence table in docs/memos/island-progress.md). Point the \
             destination at a project outside the checkout — the island's is \
             ../island-build/project.",
            project.root().display(),
            root.display()
        )));
    }
    let raw = std::fs::read_to_string(manifest_path)?;
    let m: Manifest = serde_json::from_str(&raw)
        .map_err(|e| AssetError::Import(format!("{}: {e}", manifest_path.display())))?;
    if m.schema_version > MANIFEST_SCHEMA_VERSION {
        return Err(AssetError::Import(format!(
            "{} is manifest schema v{}, and this build reads v{MANIFEST_SCHEMA_VERSION} — \
             re-export it with this tree's tools/ue-export/export.py",
            manifest_path.display(),
            m.schema_version
        )));
    }
    let base = manifest_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let dest = project.root().join(&opts.dest);
    std::fs::create_dir_all(&dest)?;

    let wanted = |pack: &str| opts.packs.is_empty() || opts.packs.iter().any(|p| p == pack);
    let mut report = UeImportReport::default();
    for p in &m.packs {
        if wanted(&p.name) {
            report.advisories.push(format!(
                "pack {}: licence {} [{}]",
                p.name,
                p.license,
                if p.ship {
                    "MAY SHIP"
                } else {
                    "LOCAL REFERENCE ONLY - never cook, never commit"
                }
            ));
            report
                .licences
                .push((p.name.clone(), p.license.clone(), p.ship));
        }
    }

    // Texture records by key, so a material can find the file behind a map name.
    let by_key: BTreeMap<&str, &Texture> = m.textures.iter().map(|t| (t.key.as_str(), t)).collect();

    // ── 1. materials, each with its own map set ──────────────────────────────
    let mut mat_ids: BTreeMap<String, AssetId> = BTreeMap::new();
    let rebind_of = |key: &str| -> Option<&str> {
        opts.rebinds
            .iter()
            .find(|(_, k)| k == key)
            .map(|(stem, _)| stem.as_str())
    };
    for mat in &m.materials {
        if !wanted(&mat.pack) {
            continue;
        }
        let stem = rebind_of(&mat.key);
        let before = report.textures.len();
        let id = import_material(project, &base, &dest, mat, &by_key, opts, stem, &mut report)?;
        mat_ids.insert(mat.key.clone(), id);
        report.materials.push((mat.key.clone(), id));
        // The material and every texture IT wrote belong to its pack — the only
        // place a `.inf_tex`'s provenance is known, since a texture record has no
        // pack of its own.
        report.asset_packs.push((id, mat.pack.clone()));
        for t in report.textures[before..].to_vec() {
            report.asset_packs.push((t, mat.pack.clone()));
        }
        if let Some(stem) = stem {
            report.rebinds.push((stem.to_string(), id));
        }
    }

    // ── 2. meshes, through the one importer door ─────────────────────────────
    if opts.meshes {
        for mesh in &m.meshes {
            if !wanted(&mesh.pack) {
                continue;
            }
            // LOD 0 is the asset. The coarser rungs are RECORDED and not stored:
            // see the wave ledger — every drawn `.inf_mesh` in this engine goes
            // through a derived `.inf_vmesh`, whose LOD is a continuous meshlet
            // cut, so a second authored discrete ladder would be bytes nothing
            // reads. The census is what a future wave that seeds the DAG from
            // the pack's own rungs will need, and it is in the sidecar.
            let Some(lod0) = mesh.lods.iter().find(|l| l.level == 0) else {
                report
                    .advisories
                    .push(format!("{}: no LOD 0 in the manifest", mesh.key));
                continue;
            };
            let Some(file) = lod0.file.as_ref() else {
                report
                    .advisories
                    .push(format!("{}: LOD 0 exported no file", mesh.key));
                continue;
            };
            let src = base.join(file);
            if !src.is_file() {
                report
                    .advisories
                    .push(format!("{}: {} is not on disk", mesh.key, src.display()));
                continue;
            }
            let out = super::import::import_file(project, &src, &dest)?;
            // Everything the one importer door produced belongs to this pack —
            // the mesh, and the skeleton, materials and textures a glTF carries
            // inside it. Collected here because `import_file` is where the set is
            // known, and because a licence stamped onto the mesh and not onto the
            // rig it needs is a licence somebody will read half of.
            let produced = out.produced.clone();
            report.advisories.extend(out.advisories);
            let Some(id) = out.primary else {
                report
                    .advisories
                    .push(format!("{}: produced no mesh", mesh.key));
                continue;
            };
            let tris = project
                .load_payload::<inf_mesh::MeshAsset>(id)
                .map(|m| m.triangle_count())
                .unwrap_or(0);
            record_rungs(project, id, mesh, &mut report);
            for a in dependency_closure(project, &[&[id][..], &produced].concat()) {
                report.asset_packs.push((a, mesh.pack.clone()));
            }
            report
                .meshes
                .push((mesh.key.clone(), id, mesh.lods.len(), tris));
        }
    }

    // -- 2b. SKELETAL meshes, through the same one door (wave CHAR1a) --------
    //
    // Same `import_file` the rigid meshes use -- it has parsed glTF `skins`,
    // `inverseBindMatrices`, `JOINTS_0`/`WEIGHTS_0` and `animations` since
    // P11.1, and writes the `.inf_skel` plus the dependency edge from the mesh
    // onto it. What this loop adds is the LADDER: unlike a rigid mesh (whose
    // coarser rungs are RECORDED and thrown away, because every rigid draw goes
    // through a continuous meshlet cut) a **skinned** mesh never reaches the
    // vgeom path at all -- `MeshAsset::vgeom_streams` drops the skin stream --
    // so a character that wants a LOD ladder has to have one stored, rung by
    // rung.
    let mut skeletons_by_source: BTreeMap<String, (AssetId, inf_anim::SkeletonAsset)> =
        BTreeMap::new();
    if opts.meshes {
        for sk in &m.skeletal_meshes {
            if !wanted(&sk.pack) {
                continue;
            }
            let mut rungs: Vec<(u32, AssetId, usize)> = Vec::new();
            let mut skel_id: Option<AssetId> = None;
            for lod in distinct_rungs(&sk.lods, opts.character_lods) {
                let Some(file) = lod.file.as_ref() else {
                    report
                        .advisories
                        .push(format!("{}: LOD {} exported no file", sk.key, lod.level));
                    continue;
                };
                let src = base.join(file);
                if !src.is_file() {
                    report
                        .advisories
                        .push(format!("{}: {} is not on disk", sk.key, src.display()));
                    continue;
                }
                let out = super::import::import_file(project, &src, &dest)?;
                // …and the same for a body: its rig, its baked materials and
                // their textures all cross under the pack's licence.
                let produced = out.produced.clone();
                report.advisories.extend(out.advisories);
                let Some(id) = out.primary else {
                    report
                        .advisories
                        .push(format!("{}: LOD {} produced no mesh", sk.key, lod.level));
                    continue;
                };
                // The skeleton is whichever product decoded as one. Taken from
                // LOD 0 only: every rung of one character shares one rig, and
                // importing rung 1's copy would give the ladder two skeletons
                // whose joint ORDER agrees only by luck.
                if lod.level == 0 {
                    // The products of THIS call, plus the mesh's own dependency
                    // edges. The second half is not belt-and-braces: on a
                    // re-import the content-hash dedupe reuses the assets that
                    // are already there and `produced` can be empty, so a body
                    // imported twice would look to this loop like a body with no
                    // rig -- which `rebind_character` then refuses, correctly and
                    // uselessly. The dependency edge is written by `import_file`
                    // when the mesh is skinned and survives the dedupe.
                    let mut candidates: Vec<AssetId> = out.produced.clone();
                    candidates.extend(
                        project
                            .db()
                            .get(id)
                            .map(|e| e.sidecar.dependencies.clone())
                            .unwrap_or_default(),
                    );
                    for pid in &candidates {
                        if project.db().get(*pid).map(|e| e.kind())
                            == Some(inf_asset::AssetKind::Skeleton)
                        {
                            if let Ok(asset) = project.load_payload::<inf_anim::SkeletonAsset>(*pid)
                            {
                                skeletons_by_source.insert(sk.source.clone(), (*pid, asset));
                            }
                            skel_id = Some(*pid);
                            break;
                        }
                    }
                }
                let tris = project
                    .load_payload::<inf_mesh::MeshAsset>(id)
                    .map(|mesh| mesh.triangle_count())
                    .unwrap_or(0);
                for a in dependency_closure(project, &[&[id][..], &produced].concat()) {
                    report.asset_packs.push((a, sk.pack.clone()));
                }
                // **THE SLOT TABLE** (wave CHAR1a.3, `.inf_mesh` v3). The
                // manifest states this mesh's material slots as manifest KEYS and
                // section 1 has already imported each of them, so this is the one
                // place in the tree where "slot 3 is that asset" is known. Written
                // into the payload rather than the sidecar because neither a
                // cooked `.ipack` nor a PIE `ScenePayload` carries a sidecar, and
                // a face whose eye slots resolved in the editor and not in the
                // game is the divergence PIE == shipping exists to stop.
                bind_slots(project, id, sk, &mat_ids, &mut report);
                rungs.push((lod.level, id, tris));
            }
            let Some((_, lod0, tris0)) = rungs.first().copied() else {
                report
                    .advisories
                    .push(format!("{}: no rung imported", sk.key));
                continue;
            };
            // **The influence-set notice.** UE writes eight influences a vertex
            // (`JOINTS_0` + `JOINTS_1`); this engine's `VertexSkin` holds four.
            // Said once per mesh with the number, because a silently halved
            // weight set is a body whose shoulders crease and nobody knows why.
            if let Some(l0) = sk.lods.first() {
                if l0.influence_sets > 1 {
                    report.advisories.push(format!(
                        "{}: the export carries {} influence sets ({} influences a \
                         vertex); this engine's VertexSkin holds 4, so the import \
                         keeps the four heaviest per vertex and renormalizes",
                        sk.key,
                        l0.influence_sets,
                        l0.influence_sets * 4
                    ));
                }
            }
            record_character_ladder(project, lod0, sk, &rungs, &mut report);
            let joints = skel_id
                .and_then(|id| project.load_payload::<inf_anim::SkeletonAsset>(id).ok())
                .map(|s| s.skeleton.len())
                .unwrap_or(0);
            if opts.rebind_character.as_deref() == Some(sk.key.as_str()) {
                rebind_character(
                    project,
                    lod0,
                    skel_id,
                    sk,
                    &mat_ids,
                    &crate::samples::starter_character_ids(),
                    ("Starter", "Starter_Body", "Starter_Skin"),
                    ("Starter_Idle", "Starter_Walk", "Starter_Run"),
                    &mut report,
                )?;
            }
            if opts.rebind_character_f.as_deref() == Some(sk.key.as_str()) {
                rebind_character(
                    project,
                    lod0,
                    skel_id,
                    sk,
                    &mat_ids,
                    &crate::samples::starter_character_f_ids(),
                    ("Starter_F", "Starter_F_Body", "Starter_F_Skin"),
                    ("Starter_F_Idle", "Starter_F_Walk", "Starter_F_Run"),
                    &mut report,
                )?;
            }
            report
                .skeletal
                .push((sk.key.clone(), lod0, skel_id, rungs.len(), tris0, joints));
        }
    }

    // -- 2c. CLIPS, retargeted onto the rig they will be played on ------------
    //
    // NOT through `import_file`: a clip glTF carries its own copy of the source
    // skin, and one hundred and twenty-six ALS clips would have written one
    // hundred and twenty-six identical `.inf_skel` assets that nothing plays and
    // whose joint indices differ from the body's. So the glTF is decoded here,
    // its clip is retargeted BY NAME onto the skeleton the body imported, and
    // one `.inf_anim` is written with a dependency edge onto that skeleton.
    for c in &m.clips {
        if !wanted(&c.pack) {
            continue;
        }
        let Some(file) = c.file.as_ref() else {
            report
                .advisories
                .push(format!("{}: clip exported no file", c.key));
            continue;
        };
        let src = base.join(file);
        if !src.is_file() {
            report
                .advisories
                .push(format!("{}: {} is not on disk", c.key, src.display()));
            continue;
        }
        let g = match inf_mesh::import_gltf(&src) {
            Ok(g) => g,
            Err(e) => {
                report.advisories.push(format!("{}: {e}", c.key));
                continue;
            }
        };
        let (Some(imported), Some(src_skel)) = (g.clips.first(), g.skeletons.first()) else {
            report.advisories.push(format!(
                "{}: the glTF carries {} clips and {} skins -- a clip needs one of each",
                c.key,
                g.clips.len(),
                g.skeletons.len()
            ));
            continue;
        };
        // The rig to play it on: the one this manifest imported for this pack's
        // body, else the first skeleton imported at all. Named rather than
        // guessed, so a manifest that exported clips and no body says so.
        let Some((target_id, target)) = opts
            .retarget_to
            .as_ref()
            .and_then(|s| skeletons_by_source.get(s))
            .or_else(|| skeletons_by_source.values().next())
            .cloned()
        else {
            report.advisories.push(format!(
                "{}: no skeleton was imported to retarget onto -- export a skeletal \
                 mesh in the same manifest, or name one with --retarget-to",
                c.key
            ));
            continue;
        };
        let map =
            inf_anim::retarget::RetargetMap::shared_names(&src_skel.skeleton, &target.skeleton);
        let (payload, rep) = inf_anim::retarget::retarget_clip(
            &imported.clip,
            &src_skel.skeleton,
            &target.skeleton,
            &map,
            true,
        );
        if rep.is_vacuous() {
            // The silent failure, named. A clip with no tracks plays as a
            // perfect bind pose, which on this rig is a T.
            report.advisories.push(format!(
                "{}: retarget produced NO tracks ({} source joints, none named on \
                 the target rig) -- the clip would play as a bind pose",
                c.key,
                src_skel.skeleton.len()
            ));
            continue;
        }
        let asset = inf_anim::AnimClipAsset::new(payload, Some(*target_id.uuid().as_bytes()));
        let name = format!("{}_{}", c.pack, c.name);
        let import = super::skeleton_binding::import_table(project, Some(target_id));
        // **IDENTITY-IDEMPOTENT** (carried item 94). `write_asset` allocates a
        // fresh GUID and side-steps a name collision by writing `X_1.inf_anim`;
        // four import runs of one manifest therefore left 656 `.inf_anim` in the
        // island project where 164 belong. The id is now a pure function of the
        // manifest key and the path a pure function of the name, so a re-import
        // overwrites its own output — the same rule the texture writer has
        // followed since ASSET0, applied to the one kind that did not.
        let id = project.write_asset_at_with_id(
            &dest.join(format!("{name}.inf_anim")),
            &asset,
            clip_guid(&c.key),
            vec![target_id],
            import,
        )?;
        // The source note the allocating door used to write.
        if let Some(entry) = project.db().get(id) {
            let path = entry.path.clone();
            if let Ok(mut side) = inf_asset::AssetSidecar::load(&path) {
                side.source = Some(c.source.clone());
                let _ = side.save(&path);
            }
        }
        report.asset_packs.push((id, c.pack.clone()));
        if !rep.dropped.is_empty() {
            report
                .advisories
                .push(format!("{}: {}", c.key, rep.summary()));
        }
        report.clips.push((c.key.clone(), id, rep.tracks_out));
    }

    // -- 2d. the REBOUND character's clips --------------------------------
    //
    // **Found by looking at the picture.** Rebinding the body and its rig alone
    // produced a hero standing in the street with one arm over its head and its
    // legs splayed, which is what a valid clip played on the wrong rig looks
    // like: `manny.rs` generates a rig whose every bind rotation is the
    // IDENTITY (deliberately -- it is what lets the inverse bind be a
    // translation and keeps the P14 no-trig law), and the shipped mannequin's
    // bind carries a real rotation on 137 of its 162 nodes. The committed
    // `Starter_Idle/Walk/Run` write absolute local rotations computed against
    // the first, and `sample_clip` seeds every untouched joint from the
    // second's `local_bind`. The two disagree everywhere.
    //
    // So a body rebind is not three assets, it is SIX: mesh, rig, skin, and the
    // three clips the hero's state machine names. The clips come from the same
    // pack as the body, so they were authored on exactly the rig that is now
    // underneath it.
    if opts.rebind_character.is_some() && !report.clips.is_empty() {
        rebind_character_clips(
            project,
            &m.clips,
            &report.clips.clone(),
            &crate::samples::starter_character_ids(),
            ("Starter_Idle", "Starter_Walk", "Starter_Run"),
            false,
            &mut report,
        )?;
    }
    if opts.rebind_character_f.is_some() && !report.clips.is_empty() {
        rebind_character_clips(
            project,
            &m.clips,
            &report.clips.clone(),
            &crate::samples::starter_character_f_ids(),
            ("Starter_F_Idle", "Starter_F_Walk", "Starter_F_Run"),
            true,
            &mut report,
        )?;
    }

    // ── 3. fixtures ──────────────────────────────────────────────────────────
    for f in &m.fixtures {
        for l in &f.lights {
            report.fixtures.push(UeFixture {
                name: f.key.clone(),
                mesh: f.meshes.first().map(|fm| fm.mesh.clone()),
                offset_m: ue_cm_to_world_m(l.location_cm),
                color_srgb8: l.color_srgb8,
                range_m: l.radius_cm / 100.0,
                intensity: ue_intensity_to_candela(l.intensity),
            });
        }
    }

    // **WHAT THE RUNG CENSUS ACTUALLY CAPTURED** (ASSET0 audit). The LOD ruling
    // — store no authored ladder, record the pack's rungs in the sidecar for the
    // wave that seeds the meshlet DAG from them — rests on that census being
    // worth inheriting. Measured on this project: `screen_size` is UE's
    // auto-compute sentinel `-1` for **18 of 18 rungs across nine packs**, so
    // what the sidecars hold is rung counts and material slots, and no
    // thresholds at all. Said once per import, with the count, rather than left
    // as a column of `-1.0` for whoever tries to use it.
    let (auto, laddered) = m
        .meshes
        .iter()
        .filter(|x| wanted(&x.pack) && x.lods.len() > 1)
        .fold((0usize, 0usize), |(a, n), x| {
            let all_auto = x.lods.iter().all(|l| l.screen_size < 0.0);
            (a + usize::from(all_auto), n + 1)
        });
    if auto > 0 {
        report.advisories.push(format!(
            "{auto} of {laddered} multi-rung meshes state no LOD screen sizes — \
             UE reports its auto-compute sentinel (-1), so the sidecar census is \
             rung counts and material slots with no thresholds in it"
        ));
    }

    // **THE LICENCE, ON DISK** (carried 96). Last, because it stamps everything
    // the run produced and the run is now over.
    let mut stamped = stamp_licences(project, &mut report);
    let (swept, exact) = sweep_licences(project, &dest, &mut report);
    stamped += swept;
    if swept > 0 {
        report.advisories.push(format!(
            "{swept} asset(s) in {} were produced by the import and are the              dependency of nothing, so their licence is {} — see `sweep_licences`",
            dest.display(),
            if exact {
                "this run's single pack"
            } else {
                "every pack this run imported, at the most conservative ship position"
            }
        ));
    }
    if stamped > 0 {
        report.advisories.push(format!(
            "licence position written into {stamped} asset sidecar(s) on disk"
        ));
    } else if !report.licences.is_empty() {
        report.advisories.push(
            "NO asset sidecar carries a licence position — the packs' licences \
             exist only in this report, which is carried item 96"
                .to_string(),
        );
    }
    report.bytes = written_bytes(project, &report);
    Ok(report)
}

/// **A deterministic asset GUID for one manifest key** (wave CHAR1a.3, carried
/// item 94).
///
/// # The defect
///
/// The clip importer wrote every `.inf_anim` through `write_asset`, which
/// ALLOCATES a fresh GUID and, on a name collision, writes `X_1.inf_anim` beside
/// the one already there. Measured at the CHAR1a audit: the island project held
/// **656** `.inf_anim` from four import runs, 656 distinct GUIDs, and 134 of the
/// 164 sources with two byte-different payloads on disk. The importer was
/// content-deterministic and not **identity**-idempotent, and the difference is
/// the difference between a re-import that updates a project and one that grows
/// it.
///
/// # The rule
///
/// The GUID is a pure function of the manifest key — the UE object path — so a
/// re-import of the same source overwrites its own output, exactly as the texture
/// writer's deterministic PATH already does. FNV-1a over the key, spread across
/// sixteen bytes, written out rather than pulled from a hasher crate for the same
/// reason `short_name`'s digest is: this is an IDENTITY, and an identity whose
/// bytes depend on a dependency's default hasher is an identity that can move
/// under a `cargo update`.
///
/// The high nibble is forced to a UUID v4 shape so the value round-trips through
/// every reader that pretty-prints one, and the salt keeps a clip's id away from
/// any other derived id of the same key.
pub fn clip_guid(key: &str) -> AssetId {
    let mut bytes = [0u8; 16];
    for (i, chunk) in bytes.chunks_mut(8).enumerate() {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        // The salt, so `clip_guid("X")` and any future `something_guid("X")`
        // cannot collide on one key.
        for b in b"inf_anim:ue-clip:".iter().chain(key.as_bytes()) {
            h ^= u64::from(*b);
            h = h.wrapping_mul(0x0000_0100_0000_01B3);
        }
        chunk.copy_from_slice(&h.to_le_bytes());
    }
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    AssetId(uuid::Uuid::from_bytes(bytes))
}

/// **Every asset reachable from `roots`**, dependencies included, to a depth of
/// three.
///
/// The licence stamp needs the CLOSURE and not the products of one call: on a
/// re-import the content-hash dedupe reuses the assets that are already there and
/// `ImportOutput::produced` comes back EMPTY — the same fact `rebind_character`'s
/// skeleton search already had to work around — so a stamp keyed on `produced`
/// gets every asset on the first run and none on the second, which is the run
/// somebody checks.
///
/// Three is the depth this content actually has: mesh → skeleton, mesh →
/// material → texture. Bounded rather than transitive-until-fixpoint because a
/// dependency graph read off disk is somebody else's bytes.
fn dependency_closure(project: &AssetProject, roots: &[AssetId]) -> Vec<AssetId> {
    let mut seen: std::collections::BTreeSet<AssetId> = roots.iter().copied().collect();
    let mut frontier: Vec<AssetId> = roots.to_vec();
    for _ in 0..3 {
        let mut next: Vec<AssetId> = Vec::new();
        for id in frontier.drain(..) {
            let Some(e) = project.db().get(id) else {
                continue;
            };
            for d in &e.sidecar.dependencies {
                if seen.insert(*d) {
                    next.push(*d);
                }
            }
        }
        frontier = next;
        if frontier.is_empty() {
            break;
        }
    }
    seen.into_iter().collect()
}

/// The sidecar `import` keys the licence position is written to.
///
/// Named once, here, because a gate reads them back: **carried item 96** was
/// that the MetaHuman licence row existed in the export manifest and in the
/// printed import report and **nowhere on disk** — a grep for `licen` over all
/// 272 imported sidecars returned nothing. A licence that travels only in a
/// console line is a licence nobody can find six months later, which for content
/// that MAY SHIP and MAY NOT BE COMMITTED is the one fact that has to be
/// attached to the bytes.
pub const LICENCE_KEY: &str = "licence";
/// Whether the pack's licence permits SHIPPING this asset — see [`LICENCE_KEY`].
pub const LICENCE_SHIP_KEY: &str = "licence_may_ship";
/// Which pack the asset came from — see [`LICENCE_KEY`].
pub const LICENCE_PACK_KEY: &str = "licence_pack";

/// **Write the pack's licence position into every asset this run produced.**
///
/// Sidecar-only: no payload moves, no schema window, and the text is where a
/// human looking at the asset will look. Returns how many sidecars were stamped,
/// which is the number the report prints and the gate asserts non-zero.
fn stamp_licences(project: &mut AssetProject, report: &mut UeImportReport) -> usize {
    let by_pack: BTreeMap<&str, (&str, bool)> = report
        .licences
        .iter()
        .map(|(p, l, s)| (p.as_str(), (l.as_str(), *s)))
        .collect();
    let mut stamped = 0usize;
    let pairs = report.asset_packs.clone();
    let mut failures: Vec<String> = Vec::new();
    for (id, pack) in pairs {
        let Some((licence, ship)) = by_pack.get(pack.as_str()).copied() else {
            continue;
        };
        let Some(entry) = project.db().get(id) else {
            continue;
        };
        let path = entry.path.clone();
        let Ok(mut side) = inf_asset::AssetSidecar::load(&path) else {
            continue;
        };
        let mut t = side.import.take().unwrap_or_default();
        t.insert(LICENCE_KEY.into(), licence.to_string().into());
        t.insert(LICENCE_SHIP_KEY.into(), ship.into());
        t.insert(LICENCE_PACK_KEY.into(), pack.clone().into());
        side.import = Some(t);
        match side.save(&path) {
            Ok(()) => stamped += 1,
            Err(e) => failures.push(format!("{}: {e}", path.display())),
        }
    }
    for f in failures {
        report
            .advisories
            .push(format!("licence not recorded on disk for {f}"));
    }
    stamped
}

/// **The sweep**: anything in the destination folder this run wrote to that the
/// per-asset pass did not reach.
///
/// # Why a sweep is needed at all
///
/// `import_file` produces more than it reports. A glTF carries its own materials,
/// and a body's four LOD rungs each carry a copy of them, so a `Content/UE/...`
/// folder ends up holding materials that are the product of an import and the
/// dependency of nothing — measured: 345 of 441 sidecars were reached by the
/// closure and **96** were not. An asset with no licence beside it is the state
/// carried item 96 is about, and "most of them have one" is not the claim.
///
/// A run that imported ONE pack attributes them exactly. A run that imported
/// several cannot — nothing on the asset says which — so the row names every
/// candidate and the ship position is the CONSERVATIVE one: local-only if any of
/// the packs is local-only, because the cost of getting that wrong in the shipping
/// direction is a licence breach and in the other direction is a missing texture.
fn sweep_licences(
    project: &AssetProject,
    dest: &Path,
    report: &mut UeImportReport,
) -> (usize, bool) {
    if report.licences.is_empty() {
        return (0, true);
    }
    let exact = report.licences.len() == 1;
    let pack = if exact {
        report.licences[0].0.clone()
    } else {
        format!(
            "(unattributed: {})",
            report
                .licences
                .iter()
                .map(|(p, _, _)| p.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    let licence = report
        .licences
        .iter()
        .map(|(p, l, _)| {
            if exact {
                l.clone()
            } else {
                format!("{p}: {l}")
            }
        })
        .collect::<Vec<_>>()
        .join("  ||  ");
    let ship = report.licences.iter().all(|(_, _, s)| *s);
    let _ = project;
    let mut n = 0usize;
    let Ok(entries) = std::fs::read_dir(dest) else {
        return (0, exact);
    };
    for e in entries.flatten() {
        let path = e.path();
        if path.extension().is_none_or(|x| x != "toml") {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        if text.contains(LICENCE_PACK_KEY) {
            continue;
        }
        let Ok(mut side) = inf_asset::AssetSidecar::load(&path.with_extension("")) else {
            continue;
        };
        let mut t = side.import.take().unwrap_or_default();
        t.insert(LICENCE_KEY.into(), licence.clone().into());
        t.insert(LICENCE_SHIP_KEY.into(), ship.into());
        t.insert(LICENCE_PACK_KEY.into(), pack.clone().into());
        side.import = Some(t);
        if side.save(&path.with_extension("")).is_ok() {
            n += 1;
        }
    }
    (n, exact)
}

/// The rung census, into the mesh's sidecar `import` table.
///
/// Sidecar-only: no payload moves, no schema window, and a human reading the
/// TOML can see what the pack shipped and what this import kept.
fn record_rungs(project: &mut AssetProject, id: AssetId, mesh: &Mesh, report: &mut UeImportReport) {
    let Some(entry) = project.db().get(id) else {
        return;
    };
    let path = entry.path.clone();
    let Ok(mut side) = inf_asset::AssetSidecar::load(&path) else {
        return;
    };
    let mut t = side.import.take().unwrap_or_default();
    t.insert("ue_source".into(), mesh.key.clone().into());
    t.insert("ue_pack".into(), mesh.pack.clone().into());
    t.insert("ue_lod_rungs".into(), (mesh.lods.len() as i64).into());
    t.insert(
        "ue_lod_screen_sizes".into(),
        toml::Value::Array(
            mesh.lods
                .iter()
                .map(|l| toml::Value::Float(f64::from(l.screen_size)))
                .collect(),
        ),
    );
    t.insert(
        "ue_material_slots".into(),
        toml::Value::Array(
            mesh.material_slots
                .iter()
                .map(|s| toml::Value::String(s.clone().unwrap_or_default()))
                .collect(),
        ),
    );
    side.import = Some(t);
    if let Err(e) = side.save(&path) {
        report
            .advisories
            .push(format!("{}: rung census not recorded ({e})", mesh.key));
    }
}

/// **Bind a skeletal mesh's material slots into its payload** (`.inf_mesh` v3).
///
/// The manifest's `material_slots` are manifest KEYS in slot order and
/// `mat_ids` maps each to the asset section 1 imported it as, so this is a
/// straight zip — and it is the only place both halves are in scope.
///
/// A slot naming a material the run did not import is left `None` and REPORTED:
/// the alternative is a face whose eyelash slot silently inherits the skin, which
/// is the state this whole feature exists to leave behind.
fn bind_slots(
    project: &mut AssetProject,
    mesh: AssetId,
    sk: &SkeletalMesh,
    mat_ids: &BTreeMap<String, AssetId>,
    report: &mut UeImportReport,
) {
    if sk.material_slots.len() < 2 {
        // One slot (or none) is what every body in this tree has, and an
        // unsectioned mesh is what every reader already draws. Writing a
        // one-entry table would cost a payload rewrite per rung for a decision
        // nobody makes.
        return;
    }
    let Ok(mut asset) = project.load_payload::<inf_mesh::MeshAsset>(mesh) else {
        return;
    };
    let mut bound = 0usize;
    let mut missing: Vec<String> = Vec::new();
    let pairs: Vec<(u32, AssetId)> = sk
        .material_slots
        .iter()
        .enumerate()
        .filter_map(|(i, key)| {
            let Some(key) = key else {
                missing.push(format!("slot {i} names no material"));
                return None;
            };
            match mat_ids.get(key) {
                Some(id) => {
                    bound += 1;
                    Some((i as u32, *id))
                }
                None => {
                    missing.push(format!("slot {i} ({key}) did not import"));
                    None
                }
            }
        })
        .collect();
    // The payload's own slot NAMES have to exist for the table to be indexable
    // by the same index `SubMesh::material_slot` carries; a glTF import writes
    // one per primitive material, so a mesh whose slots the manifest knows and
    // whose payload has none is a mesh this table cannot address.
    if asset.material_slots.len() < sk.material_slots.len() {
        report.advisories.push(format!(
            "{}: the manifest states {} material slots and the imported mesh has \
             {} — the slot table is bound over the shorter list",
            sk.key,
            sk.material_slots.len(),
            asset.material_slots.len()
        ));
    }
    asset.bind_material_slots(pairs);
    let Some(entry) = project.db().get(mesh) else {
        return;
    };
    let path = entry.path.clone();
    let deps = entry.sidecar.dependencies.clone();
    let import = entry.sidecar.import.clone();
    if let Err(e) = project.write_asset_at_with_id(&path, &asset, mesh, deps, import) {
        report
            .advisories
            .push(format!("{}: slot table not written ({e})", sk.key));
        return;
    }
    report.advisories.push(format!(
        "{}: {bound} of {} material slots bound into the mesh{}",
        sk.key,
        sk.material_slots.len(),
        if missing.is_empty() {
            String::new()
        } else {
            format!(" ({})", missing.join(", "))
        }
    ));
}

/// The character LOD ladder, into the LOD-0 mesh's sidecar `import` table.
///
/// Unlike [`record_rungs`], which records a census of rungs that were **not**
/// stored, this records the rungs that **were** -- their asset ids and their
/// measured triangle counts -- plus the switch distances derived from them.
///
/// # Where the switch distances come from
///
/// Not from the pack: UE reports its auto-compute sentinel (-1) for every
/// `screen_size` in this project, measured 18 of 18 at the ASSET0 audit and
/// again here. So they are derived from the thing that is actually known, which
/// is each rung's own triangle count against the engine's crowd bands: a rung
/// takes over where a body's on-screen height makes its triangles cost about
/// what the next rung's cost at the band above. The bands themselves are
/// `inf_ecs::crowd`'s, so a character's geometry ladder and its simulation
/// ladder switch at the same three distances instead of at two unrelated sets
/// of numbers.
fn record_character_ladder(
    project: &mut AssetProject,
    lod0: AssetId,
    sk: &SkeletalMesh,
    rungs: &[(u32, AssetId, usize)],
    report: &mut UeImportReport,
) {
    let Some(entry) = project.db().get(lod0) else {
        return;
    };
    let path = entry.path.clone();
    let Ok(mut side) = inf_asset::AssetSidecar::load(&path) else {
        return;
    };
    let mut t = side.import.take().unwrap_or_default();
    t.insert("ue_source".into(), sk.key.clone().into());
    t.insert("ue_pack".into(), sk.pack.clone().into());
    t.insert("ue_bones".into(), i64::from(sk.bones).into());
    t.insert(
        "ue_lod_rungs_exported".into(),
        (sk.lods.len() as i64).into(),
    );
    t.insert("ue_lod_rungs_stored".into(), (rungs.len() as i64).into());
    t.insert(
        "character_lod_assets".into(),
        toml::Value::Array(
            rungs
                .iter()
                .map(|(_, id, _)| toml::Value::String(id.uuid().to_string()))
                .collect(),
        ),
    );
    t.insert(
        "character_lod_triangles".into(),
        toml::Value::Array(
            rungs
                .iter()
                .map(|(_, _, tris)| toml::Value::Integer(*tris as i64))
                .collect(),
        ),
    );
    t.insert(
        "character_lod_switch_m".into(),
        toml::Value::Array(
            character_lod_switch_m(rungs.len())
                .iter()
                .map(|d| toml::Value::Float(*d))
                .collect(),
        ),
    );
    side.import = Some(t);
    if let Err(e) = side.save(&path) {
        report
            .advisories
            .push(format!("{}: ladder not recorded ({e})", sk.key));
    }
}

/// **Write an imported body at the starter character's committed GUIDs.**
///
/// See [`UeImportOptions::rebind_character`] for why. The three assets are
/// copied rather than moved: the originals keep their own ids and stay in the
/// drawer, so an author can see both, and the rebind is a second file with a
/// borrowed identity exactly as the material rebind is.
///
/// A body whose skeleton did not import is REFUSED rather than half-rebound: a
/// mesh at the hero's mesh GUID with the old rig still at the hero's skeleton
/// GUID is 92 000 triangles addressed to the wrong joints, which draws as an
/// explosion and is worse than the low-poly body it replaced.
fn rebind_character(
    project: &mut AssetProject,
    mesh: AssetId,
    skeleton: Option<AssetId>,
    sk: &SkeletalMesh,
    mat_ids: &BTreeMap<String, AssetId>,
    // **Which committed character this body becomes** (wave CHAR1a.3) — the
    // male starter's identity set or the female's. The stems ride with it,
    // because a rebind writes at a committed GUID *and* at that GUID's committed
    // FILE NAME, so the asset scan finds one asset rather than two claiming one
    // id.
    ids: &crate::character::CharacterIds,
    stems: (&str, &str, &str),
    clip_stems: (&str, &str, &str),
    report: &mut UeImportReport,
) -> Result<()> {
    let (Some(want_mesh), Some(want_skel), Some(want_mat)) = (ids.mesh, ids.skeleton, ids.material)
    else {
        return Ok(());
    };
    let Some(skeleton) = skeleton else {
        return Err(AssetError::Import(format!(
            "{}: refusing to rebind a body whose skeleton did not import — the \
             mesh's joint indices would address the rig it replaced",
            sk.key
        )));
    };
    let skel: inf_anim::SkeletonAsset = project.load_payload(skeleton)?;
    let body: inf_mesh::MeshAsset = project.load_payload(mesh)?;
    let root = project.root().to_path_buf();
    // **The rig this identity WORE**, read before it is replaced — see
    // `retarget_committed_clips`. `None` on a first rebind into a project that
    // has never had this character, which is also the case where there is
    // nothing to re-retarget.
    let previous: Option<inf_anim::Skeleton> = project
        .load_payload::<inf_anim::SkeletonAsset>(want_skel)
        .ok()
        .map(|a| a.skeleton);

    let skel_path = root.join(format!("{}.inf_skel", stems.0));
    project.write_asset_at_with_id(&skel_path, &skel, want_skel, vec![], None)?;
    let mesh_path = root.join(format!("{}.inf_mesh", stems.1));
    project.write_asset_at_with_id(&mesh_path, &body, want_mesh, vec![want_skel], None)?;
    report
        .rebinds
        .push((format!("{}.inf_skel", stems.0), want_skel));
    report
        .rebinds
        .push((format!("{}.inf_mesh", stems.1), want_mesh));

    // The skin. The first material slot, because that is the one the body's
    // torso uses on both mannequins (measured: `M_torso` on Manny, `Quinn_01`
    // on Quinn) and this engine's `SkeletalMesh` draw binds ONE material to the
    // whole body — a per-submesh material on a skinned mesh is CHAR1b's.
    if let Some(mat) = sk
        .material_slots
        .first()
        .and_then(|s| s.as_ref())
        .and_then(|k| mat_ids.get(k))
    {
        let payload: MaterialAsset = project.load_payload(*mat)?;
        let deps = payload.texture_dependencies();
        let path = root.join(format!("{}.inf_mat", stems.2));
        project.write_asset_at_with_id(&path, &payload, want_mat, deps, None)?;
        report
            .rebinds
            .push((format!("{}.inf_mat", stems.2), want_mat));
    } else {
        report.advisories.push(format!(
            "{}: rebound the body and its rig, but slot 0 named no imported \
             material — the hero keeps the starter's neutral skin",
            sk.key
        ));
    }
    report.advisories.push(format!(
        "{}: REBOUND at the starter character's GUIDs — this project's hero is \
         now that body ({} triangles, {} joints). Local only.",
        sk.key,
        body.triangle_count(),
        skel.skeleton.len()
    ));
    // **AND THE CLIPS COME WITH IT** — see `retarget_committed_clips`.
    retarget_committed_clips(
        project,
        ids,
        clip_stems,
        previous.as_ref(),
        &skel.skeleton,
        report,
    );
    Ok(())
}

/// **Re-retarget the clips a rebound identity already owns onto its NEW rig**
/// (wave CHAR1a.3).
///
/// # Why a rebind is not finished without it
///
/// A clip's coupling to a rig is POSITIONAL: `QuatTrack::joint` is a `u16` index
/// into `Pose::locals`, index-aligned to `Skeleton::joints`. So writing a
/// different skeleton at a character's committed GUID re-points every track in
/// its clips at a different bone — in range, so nothing refuses, and the
/// character animates with an arm where a spine should be. Wave CHAR1a found
/// exactly that picture ("one arm over its head and its legs splayed") when it
/// rebound a body without its clips, and fixed it by taking the clips from the
/// same PACK as the body.
///
/// That fix does not reach a pack with no clips of its own, which is what the
/// MetaHumans are: the assembly writes skeletal meshes and no `AnimSequence` at
/// all. Their hero would take a 342-joint rig and keep three clips indexed
/// against a 161-joint one.
///
/// So the clips are re-retargeted BY NAME, from the rig the identity wore to the
/// rig it now wears — `RetargetMap::shared_names` plus `CHAIN_INFILL`, the same
/// door every other clip in this bridge crosses. A joint the new rig has and the
/// old one did not (a MetaHuman body publishes the mannequin's 161 names among
/// its 342: twist chains, correctives, the face's neck) is simply untouched and
/// plays at its bind, which is the honest answer for a bone no source clip ever
/// moved.
///
/// A no-op when the rig did not change, and when there was no previous rig at
/// all.
fn retarget_committed_clips(
    project: &mut AssetProject,
    ids: &crate::character::CharacterIds,
    stems: (&str, &str, &str),
    previous: Option<&inf_anim::Skeleton>,
    target: &inf_anim::Skeleton,
    report: &mut UeImportReport,
) {
    let Some(previous) = previous else { return };
    if previous.len() == target.len()
        && previous
            .joints()
            .iter()
            .zip(target.joints())
            .all(|(a, b)| a.name == b.name)
    {
        return;
    }
    // The body the clips are settled against: the mesh that was just written at
    // this identity's own GUID, read back off disk so the reader goes where the
    // runtime goes.
    let ground: Option<Vec<inf_anim::retarget::GroundVertex>> = ids
        .mesh
        .and_then(|id| project.load_payload::<inf_mesh::MeshAsset>(id).ok())
        .map(|m| {
            m.submeshes
                .iter()
                .filter(|s| s.is_skinned())
                .flat_map(|s| {
                    s.vertices
                        .iter()
                        .zip(s.skin.iter())
                        .map(|(v, k)| (v.position, k.joints, k.weights))
                })
                .collect()
        });
    let map = inf_anim::retarget::RetargetMap::shared_names(previous, target);
    let slots = [
        (format!("{}.inf_anim", stems.0), ids.idle),
        (format!("{}.inf_anim", stems.1), ids.walk),
        (format!("{}.inf_anim", stems.2), ids.run),
    ];
    for (file, want) in slots {
        let Some(want) = want else { continue };
        let Ok(mut payload) = project.load_payload::<inf_anim::AnimClipAsset>(want) else {
            continue;
        };
        let (clip, rep) =
            inf_anim::retarget::retarget_clip(&payload.clip, previous, target, &map, true);
        if rep.is_vacuous() {
            report.advisories.push(format!(
                "{file}: retarget onto the rebound rig produced NO tracks ({} source \
                 joints, none named on the {} of the new rig) — the clip is left as \
                 it was and will animate the wrong bones",
                previous.len(),
                target.len()
            ));
            continue;
        }
        payload.clip = clip;
        payload.skeleton = ids.skeleton.map(|s| *s.uuid().as_bytes());
        if let (Some(mesh), Some(_)) = (ground.as_ref(), ids.mesh) {
            let d = inf_anim::retarget::settle_to_ground_with_skin(&mut payload.clip, target, mesh);
            if d.abs() >= 0.001 {
                report.advisories.push(format!(
                    "{file}: settled {:.1} mm onto the rebound body's ground plane",
                    d * 1000.0
                ));
            }
        }
        let path = project.root().join(&file);
        let deps: Vec<AssetId> = ids.skeleton.into_iter().collect();
        let import = super::skeleton_binding::import_table(project, ids.skeleton);
        match project.write_asset_at_with_id(&path, &payload, want, deps, import) {
            Ok(_) => report.advisories.push(format!(
                "{file}: re-retargeted onto the rebound rig — {} of {} tracks kept, \
                 {} dropped ({} → {} joints)",
                rep.tracks_out,
                rep.tracks_in,
                rep.dropped.len(),
                previous.len(),
                target.len()
            )),
            Err(e) => report
                .advisories
                .push(format!("{file}: re-retarget not written ({e})")),
        }
    }
}

/// **The three clips the hero's state machine names, from the rebound body's own
/// pack.**
///
/// The mapping is a stated table rather than a heuristic, because "which clip is
/// the idle" is a decision:
///
/// | starter slot | mannequin clip | why |
/// |---|---|---|
/// | idle | `MM_Idle` / `MF_Idle` | the only idle either mannequin ships |
/// | walk | `MM_Walk_Fwd` / `MF_Walk_Fwd` | forward, root-motion, not the in-place variant |
/// | run  | `MM_Run_Fwd` / `MF_Run_Fwd` | ditto |
///
/// `MM_Walk_InPlace` is deliberately NOT the walk: this engine's locomotion
/// machine drives the body from the movement component and reads the clip for
/// the pose, so an in-place walk would slide the feet at exactly the speed the
/// character travels.
///
/// A slot with no matching clip keeps the committed generated one and is named
/// in an advisory — which is the honest failure, because the alternative is a
/// hero whose walk is somebody else's idle.
fn rebind_character_clips(
    project: &mut AssetProject,
    manifest: &[Clip],
    imported: &[(String, AssetId, usize)],
    ids: &crate::character::CharacterIds,
    stems: (&str, &str, &str),
    // **Which mannequin's clips this identity prefers** (wave CHAR1a.3). The
    // table below names two candidates per slot and `find` takes the FIRST the
    // manifest carries, which is sorted by object path -- so Manny sorts before
    // Quinn and BOTH identities took `MM_Idle`. A male default and a female
    // default that walk identically are one default twice.
    prefer_female: bool,
    report: &mut UeImportReport,
) -> Result<()> {
    // The body and the rig the hero was just rebound to, for the ground settle
    // below. Read back off disk rather than threaded through from
    // `rebind_character`: they were written at fixed GUIDs a moment ago, and a
    // reader that goes to the same place the runtime will is a reader that
    // cannot disagree with it.
    let rig: Option<inf_anim::Skeleton> = ids
        .skeleton
        .and_then(|id| project.load_payload::<inf_anim::SkeletonAsset>(id).ok())
        .map(|a| a.skeleton);
    let ground: Option<Vec<inf_anim::retarget::GroundVertex>> = ids
        .mesh
        .and_then(|id| project.load_payload::<inf_mesh::MeshAsset>(id).ok())
        .map(|m| {
            m.submeshes
                .iter()
                .filter(|s| s.is_skinned())
                .flat_map(|s| {
                    s.vertices
                        .iter()
                        .zip(s.skin.iter())
                        .map(|(v, k)| (v.position, k.joints, k.weights))
                })
                .collect()
        });
    let order = |m: &'static str, f: &'static str| -> [&'static str; 2] {
        if prefer_female {
            [f, m]
        } else {
            [m, f]
        }
    };
    let slots: [(&str, Option<AssetId>, [&str; 2]); 3] = [
        (stems.0, ids.idle, order("MM_Idle", "MF_Idle")),
        (stems.1, ids.walk, order("MM_Walk_Fwd", "MF_Walk_Fwd")),
        (stems.2, ids.run, order("MM_Run_Fwd", "MF_Run_Fwd")),
    ];
    let by_key: BTreeMap<&str, AssetId> = imported
        .iter()
        .map(|(k, id, _)| (k.as_str(), *id))
        .collect();
    for (stem, want, names) in slots {
        let Some(want) = want else { continue };
        // The PREFERENCE is honoured before the manifest's order: the first
        // name that exists wins, not the first record that matches either name.
        let found = names
            .iter()
            .find_map(|want| manifest.iter().find(|c| c.name == *want))
            .and_then(|c| by_key.get(c.key.as_str()).copied());
        let Some(found) = found else {
            report.advisories.push(format!(
                "{stem}: the rebound body's pack ships none of {names:?}, so the \
                 hero keeps the generated clip -- which was authored against a \
                 rig whose bind pose is the identity and will look wrong on this \
                 body"
            ));
            continue;
        };
        let mut payload: inf_anim::AnimClipAsset = project.load_payload(found)?;
        // **SETTLED AGAINST THIS BODY** (wave CHAR1a.2). `retarget_clip` already
        // settled the clip against the target RIG, and that pins the lowest
        // ball/foot joint — but a foot rotated at toe-off lifts its sole while
        // its joint stays put, so the mannequin's run still dipped **17.8 mm**
        // into the road after it. Here the rebound body is on disk, so the
        // question can be asked of the mesh: the lowest skinned vertex over the
        // cycle, the same arithmetic the shader runs. Measured after: the hero's
        // idle plants within 1.6 mm and its run within 1.9 mm.
        if let (Some(sk), Some(mesh)) = (rig.as_ref(), ground.as_ref()) {
            let d = inf_anim::retarget::settle_to_ground_with_skin(&mut payload.clip, sk, mesh);
            if d.abs() >= 0.001 {
                report.advisories.push(format!(
                    "{stem}: settled {:.1} mm onto the rebound body's ground plane",
                    d * 1000.0
                ));
            }
        }
        let deps: Vec<AssetId> = project
            .db()
            .get(found)
            .map(|e| e.sidecar.dependencies.clone())
            .unwrap_or_default();
        let path = project.root().join(format!("{stem}.inf_anim"));
        // **AND THE SKELETON HASH MOVES WITH THE RIG** (carried item 95). The
        // rebind writes a clip authored against the PACK's rig at the starter
        // character's GUID, over a rig that has just been replaced by that same
        // pack's — so the clip and the rig ARE authored together, and the sidecar
        // said otherwise: the island's four rebound assets recorded `8d06c1ee…`
        // while `Starter.inf_skel` hashed `5c7c1647…`, and the editor's content
        // scan printed *"the character animates the wrong bones"* on every boot
        // for content that was in fact correct. A false positive that will hide a
        // true one — so the table is rebuilt here, against the rig on disk.
        let import = super::skeleton_binding::import_table(project, ids.skeleton);
        project.write_asset_at_with_id(&path, &payload, want, deps, import)?;
        report.rebinds.push((format!("{stem}.inf_anim"), want));
    }
    Ok(())
}

/// The rungs of `lods` worth storing: at most `keep`, each strictly coarser
/// than the one before.
///
/// # Why "the first `keep` rungs" is the wrong rule
///
/// **Measured on `SKM_Manny`**: its four exported rungs are 92 178, 92 178,
/// 26 998 and 12 998 triangles — LOD 1 is a *copy* of LOD 0. Taking the first
/// three would have stored 92 178 triangles twice, shipped 4.1 MB of duplicate
/// `.inf_mesh` per body, and given the ladder a switch at 32 m that changes
/// nothing on screen while costing a mesh swap. Selecting by the triangle count
/// the exporter measured gives 92 178 / 26 998 / 12 998 — a real 3.4:1 and
/// 2.1:1 ladder — and the rule is a property of the content rather than of the
/// pack's LOD-settings asset.
///
/// A rung with an unknown (zero) triangle count is kept if it is the first, and
/// otherwise skipped: an unmeasured rung cannot be shown to be coarser than the
/// one before it, and a ladder built on a guess is worse than a short one.
fn distinct_rungs(lods: &[SkelLod], keep: usize) -> Vec<&SkelLod> {
    let mut out: Vec<&SkelLod> = Vec::new();
    let mut last = u32::MAX;
    for lod in lods {
        if out.len() >= keep {
            break;
        }
        if out.is_empty() {
            out.push(lod);
            last = lod.triangles;
            continue;
        }
        if lod.triangles > 0 && lod.triangles < last {
            out.push(lod);
            last = lod.triangles;
        }
    }
    out
}

/// The distances, in metres, at which each stored character rung takes over.
///
/// One entry per rung; entry `i` is the distance at which rung `i` starts being
/// drawn, so entry 0 is always 0. They are `inf_ecs::crowd`'s own tier radii,
/// which is the point: the geometry a character draws and the simulation it
/// runs change at the same distance, so a body cannot be posed by the full
/// animation graph while drawing its cheapest mesh, or the reverse.
pub fn character_lod_switch_m(rungs: usize) -> Vec<f64> {
    let bands = [
        0.0,
        inf_ecs::crowd::DEFAULT_CROWD_FULL_M,
        inf_ecs::crowd::DEFAULT_CROWD_NEAR_M,
        inf_ecs::crowd::DEFAULT_CROWD_FAR_M,
    ];
    (0..rungs)
        .map(|i| {
            bands
                .get(i)
                .copied()
                .unwrap_or(inf_ecs::crowd::DEFAULT_CROWD_FAR_M)
        })
        .collect()
}

/// Total bytes on disk of everything this run produced.
fn written_bytes(project: &AssetProject, report: &UeImportReport) -> u64 {
    let mut n = 0;
    for id in report
        .textures
        .iter()
        .copied()
        .chain(report.materials.iter().map(|(_, id)| *id))
        .chain(report.meshes.iter().map(|(_, id, _, _)| *id))
    {
        if let Some(e) = project.db().get(id) {
            n += std::fs::metadata(&e.path).map(|m| m.len()).unwrap_or(0);
        }
    }
    n
}

/// One material and its whole map set.
#[allow(clippy::too_many_arguments)]
fn import_material(
    project: &mut AssetProject,
    base: &Path,
    dest: &Path,
    mat: &Material,
    by_key: &BTreeMap<&str, &Texture>,
    opts: &UeImportOptions,
    rebind: Option<&str>,
    report: &mut UeImportReport,
) -> Result<AssetId> {
    // Decode every map this material names, clamped. `BTreeMap` so the walk is
    // ordered by role name and two runs of one manifest write the same assets in
    // the same order — the GUID-stability property every import in this tree has.
    let mut planes: BTreeMap<MapKind, (Vec<u8>, u32, u32)> = BTreeMap::new();
    // Roles the manifest states and this engine has nowhere to put. Collected
    // and REPORTED rather than skipped in silence (ASSET0 audit): "the material
    // imported" and "the material imported with half its maps" looked identical
    // in the report, and one of the silent roles was clobbering the albedo.
    let mut unplaced: Vec<&str> = Vec::new();
    for (role, key) in &mat.maps {
        let targets = role_to_planes(role);
        if targets.is_empty() {
            unplaced.push(role.as_str());
            continue;
        };
        let Some(tex) = by_key.get(key.as_str()) else {
            report
                .advisories
                .push(format!("{}: no texture record for {key}", mat.key));
            continue;
        };
        let Some(file) = tex.file.as_ref() else {
            continue;
        };
        let path = base.join(file);
        let Ok(bytes) = std::fs::read(&path) else {
            report
                .advisories
                .push(format!("{}: {} is not on disk", mat.key, path.display()));
            continue;
        };
        let (rgba, w, h) = inf_material::decode_image_rgba8(&bytes)
            .map_err(|e| AssetError::Import(format!("{}: {e}", path.display())))?;
        let (rgba, w, h) = inf_material::downscale_rgba8(rgba, w, h, opts.max_texture)
            .map_err(|e| AssetError::Import(format!("{}: {e}", path.display())))?;
        // One source texture may fill more than one slot (a packed UE mask), and
        // a slot a role already claimed is NOT overwritten — a dedicated
        // `roughness` map beats the roughness channel of a packed mask, and the
        // walk is `BTreeMap`-ordered so which one that is is a property of the
        // manifest rather than of an iteration.
        for (kind, channel) in targets {
            if planes.contains_key(kind) {
                continue;
            }
            let plane = match channel {
                Some(c) => broadcast_channel(&rgba, *c),
                None => rgba.clone(),
            };
            planes.insert(*kind, (plane, w, h));
        }
    }
    if !unplaced.is_empty() {
        report.advisories.push(format!(
            "{}: the manifest names {} and this engine's `.inf_mat` has no slot \
             for {} — not imported. The material keeps its scalar values for \
             those channels.",
            mat.key,
            unplaced.join(", "),
            if unplaced.len() == 1 { "it" } else { "them" }
        ));
    }

    let name = short_name(&mat.key);
    let write = |kind: MapKind,
                 slot: &str,
                 rgba: Vec<u8>,
                 w: u32,
                 h: u32,
                 project: &mut AssetProject,
                 report: &mut UeImportReport|
     -> Result<AssetId> {
        // The SLOT's settings, from the engine's own table — sRGB for exactly
        // one map, BC5 for a normal, BC1 for the rest. `source_is_float` is
        // false: everything the bridge exports is 8-bit PNG.
        let settings: TextureImportSettings = kind.settings(false);
        let image = inf_material::build_tiled_texture(rgba, w, h, settings)
            .map_err(|e| AssetError::Import(format!("{}_{slot}: {e}", mat.key)))?;
        // **A DETERMINISTIC PATH**, so a second run over an unchanged manifest
        // overwrites its own output instead of writing `X_1.inf_tex` beside it.
        // Measured before this line existed: the second import wrote 106
        // duplicate assets and doubled the project's texture bytes.
        let id = project.write_tiled_texture_at(
            &dest.join(format!("{name}_{slot}.inf_tex")),
            &image,
            Some(mat.source_note()),
            None,
        )?;
        report.textures.push(id);
        Ok(id)
    };

    let albedo = match planes.remove(&MapKind::Albedo) {
        Some((px, w, h)) => Some(write(MapKind::Albedo, "Albedo", px, w, h, project, report)?),
        None => None,
    };
    let normal = match planes.remove(&MapKind::Normal) {
        Some((px, w, h)) => Some(write(MapKind::Normal, "Normal", px, w, h, project, report)?),
        None => None,
    };

    // **The ORM, packed.** Occlusion → R, roughness → G, metallic → B, with
    // 255/255/0 where the pack ships nothing — which is every Megascans surface
    // in this project, none of which has an AO or a metallic map. The extent is
    // the SMALLEST of the three: packing a 2 048 roughness into a 4 096 grid
    // would read past the end of it, and `pack_orm` refuses that rather than
    // guessing (it returns `None`).
    let orm_planes = [MapKind::Occlusion, MapKind::Roughness, MapKind::Metallic];
    let orm = if orm_planes.iter().any(|k| planes.contains_key(k)) {
        let (ew, eh) = orm_planes
            .iter()
            .filter_map(|k| planes.get(k))
            .map(|(_, w, h)| (*w, *h))
            .fold((u32::MAX, u32::MAX), |(aw, ah), (w, h)| {
                (aw.min(w), ah.min(h))
            });
        let plane = |k: MapKind| -> Result<Option<Vec<u8>>> {
            let Some((px, w, h)) = planes.get(&k) else {
                return Ok(None);
            };
            if (*w, *h) == (ew, eh) {
                return Ok(Some(px.clone()));
            }
            let (px, _, _) = inf_material::downscale_rgba8(px.clone(), *w, *h, ew.max(eh))
                .map_err(|e| AssetError::Import(format!("{}: {e}", mat.key)))?;
            Ok(Some(px))
        };
        let o = plane(MapKind::Occlusion)?;
        let r = plane(MapKind::Roughness)?;
        let mt = plane(MapKind::Metallic)?;
        match inf_material::pack_orm(o.as_deref(), r.as_deref(), mt.as_deref(), ew, eh) {
            Some(px) => Some(write(
                MapKind::Roughness,
                "ORM",
                px,
                ew,
                eh,
                project,
                report,
            )?),
            None => {
                report.advisories.push(format!(
                    "{}: its occlusion/roughness/metallic maps are not one size, so no ORM \
                     was packed and the material falls back to its scalar roughness",
                    mat.key
                ));
                None
            }
        }
    } else {
        None
    };

    let asset = MaterialAsset {
        schema_version: MaterialAsset::CURRENT_VERSION,
        base_color: mat.base_color,
        metallic: mat.metallic,
        // A Megascans instance carries `roughness = 1.0` as the MULTIPLIER on
        // its roughness map, not as a roughness. Kept as-is when a map is bound
        // (the map is the signal); it is only the fallback that matters, and a
        // surface with no map that says 1.0 really is fully rough.
        roughness: mat.roughness,
        emissive: mat.emissive,
        base_color_texture: albedo,
        normal_texture: normal,
        metallic_roughness_texture: orm,
        blend: match mat.blend.as_str() {
            "masked" => MatBlend::Masked,
            "blend" => MatBlend::Translucent,
            _ => MatBlend::Opaque,
        },
        ..Default::default()
    };
    let deps = asset.texture_dependencies();
    match rebind {
        // **At the committed GUID**, and at the committed FILE NAME, so the
        // asset scan finds one asset rather than two claiming one id.
        Some(stem) => {
            let guid = crate::ground::ground_material_guid(stem_kind(stem).ok_or_else(|| {
                AssetError::Import(format!(
                    "{stem} is not a ground-library surface; the stems are {}",
                    inf_material::ground::GroundKind::ALL
                        .iter()
                        .map(|k| k.stem())
                        .collect::<Vec<_>>()
                        .join(", ")
                ))
            })?);
            let path = project.root().join(format!("{stem}.inf_mat"));
            let id = project.write_asset_at_with_id(&path, &asset, AssetId(guid), deps, None)?;
            report.advisories.push(format!(
                "{stem} now carries {} in this project — the committed synthesised \
                 material of the same GUID is overwritten LOCALLY and stays unchanged in \
                 the repository",
                mat.key
            ));
            Ok(id)
        }
        // …and the material likewise, for the same reason and by the same door.
        None => project.write_asset_at(&dest.join(format!("{name}.inf_mat")), &asset, deps, None),
    }
}

/// The ground-library kind a rebind stem names.
fn stem_kind(stem: &str) -> Option<inf_material::ground::GroundKind> {
    inf_material::ground::GroundKind::ALL
        .into_iter()
        .find(|k| k.stem() == stem)
}

/// A manifest role name → the engine's [`MapKind`]s, with a channel each.
///
/// **Exactly the roles this engine has somewhere to put**, and nothing else:
/// albedo and normal get their own slot, and occlusion/roughness/metallic are
/// packed into the one ORM. A role absent from this table is reported by
/// `import_material` rather than dropped in silence.
///
/// `displacement` is deliberately absent: this engine has no displacement slot
/// on `.inf_mat`, and importing a height map as a texture nothing samples would
/// be 2 MB an asset for a channel no shader reads.
///
/// # Two that were here and should not have been (ASSET0 audit)
///
/// `"emissive" => MapKind::Albedo` was a **silent clobber**. `planes` is keyed
/// by `MapKind` and `mat.maps` is a `BTreeMap`, so on any material carrying both
/// maps `"emissive"` sorts after `"albedo"` and *replaced* it — the material
/// would have shipped its glow map as its base colour. Nothing in the reference
/// project's thirty materials names an emissive map, so the defect was
/// unreachable today and one export-list edit from being reached. There is no
/// emissive texture slot on `.inf_mat`; the scalar `emissive` the manifest
/// states is what crosses.
///
/// `"opacity" => MapKind::Opacity` was a **silent drop with a bill attached**:
/// the plane was decoded and clamped — an 8 K source is 268 MB of RGBA — and
/// then never read, because only Albedo, Normal and the ORM trio are consumed.
/// The engine carries alpha in the base colour's own channel, so an opacity map
/// would have to be composited into the albedo to mean anything; until it is,
/// the honest answer is to say so and not pay for the decode.
///
/// # A packed UE mask, unpacked into the slots this engine has (wave CHAR1a.2)
///
/// Some roles are not one map: `msr` is Unreal's `T_*_MSR_MSK`, which carries
/// **metallic in R, specular in G and roughness in B** — the name is the spec and
/// a channel census of `T_Manny_01_MSR_MSK` confirms it (R bimodal 0/255 = a
/// metal mask; G flat at 92–118 = the 0.5 specular constant; B 0/116/255 = the
/// roughness). This engine's ORM is occlusion/roughness/metallic, so it is a
/// SWIZZLE and not a rename, which is exactly what CHAR1a carried as item 76.
///
/// `aniso_ao_paint` is `T_*_AS?AO?MASK_MSK`: anisotropy in R, **ambient
/// occlusion in G**, a paint mask in B. The importer used to take plane R,
/// whose mean over the mannequin is **3.5 of 255** — so every character imported
/// through this bridge had its ambient term multiplied by 0.014. That is not a
/// subtle wrong: it is the body reading almost unlit in shade.
///
/// `Some(channel)` means "broadcast that channel of the decoded RGBA into a grey
/// plane"; `pack_orm` reads channel 0 of whatever it is handed, so the broadcast
/// is what makes one source texture fill two different ORM channels.
///
/// Returns an EMPTY slice for a role this engine has nowhere to put, which the
/// caller reports as unplaced — `tangent`, `normal_second`, `decal` and
/// `clearcoat` are all real maps with no home here, and saying so is the ASSET0
/// audit's rule. Wave CHAR1a.3 adds four more of them from the MetaHuman
/// materials: `scatter` (the subsurface amount — PAR5's, and this engine has no
/// SSS term to feed), `detail_mask` (a 32-pixel micro-tiling mask; the engine's
/// own detail slot is a whole texture with a tiling rate, not a mask), and
/// `animated_delta` (the facial rig's per-curve basecolor/normal deltas, which
/// need the 875-joint face rig FACE1 will build). Each is a real map this bridge
/// carries across and this engine cannot yet spend, and the import log says so
/// per material.
pub fn role_to_planes(role: &str) -> &'static [(MapKind, Option<usize>)] {
    match role {
        "albedo" => &[(MapKind::Albedo, None)],
        "normal" => &[(MapKind::Normal, None)],
        "roughness" => &[(MapKind::Roughness, None)],
        "metallic" => &[(MapKind::Metallic, None)],
        "ao" => &[(MapKind::Occlusion, None)],
        "msr" => &[(MapKind::Metallic, Some(0)), (MapKind::Roughness, Some(2))],
        "aniso_ao_paint" => &[(MapKind::Occlusion, Some(1))],
        // **The MetaHuman mask** (wave CHAR1a.3). `T_*_SRMF` is
        // Specular / Roughness / Metallic / Fuzz and `T_Teeth_SRM` the same
        // three without the fourth -- the name is the spec, and a channel census
        // of three of them agrees: `T_Body_SRMF` R median 150 (the specular
        // constant), G median **187** (the roughness), B **exactly 0 at every
        // percentile including the max** (skin is not a metal), A median 200.
        // So roughness is G and metallic is B, which is a different swizzle from
        // `msr`'s and the reason both are in this table by name rather than one
        // "packed mask" rule that would have to guess.
        //
        // The specular plane has nowhere to go: this engine's PBR is
        // metallic-roughness and reads no specular map. It is dropped in silence
        // here rather than reported, because unlike `tangent` or `clearcoat` it
        // is a CHANNEL of a texture that IS imported -- the advisory would say
        // "srmf is unplaced" about a map two thirds of which just landed.
        "srmf" => &[(MapKind::Roughness, Some(1)), (MapKind::Metallic, Some(2))],
        _ => &[],
    }
}

/// One channel of an RGBA plane, broadcast into a fresh grey RGBA plane.
pub fn broadcast_channel(rgba: &[u8], channel: usize) -> Vec<u8> {
    let mut out = vec![255u8; rgba.len()];
    for (i, px) in rgba.chunks_exact(4).enumerate() {
        let v = px[channel];
        out[i * 4] = v;
        out[i * 4 + 1] = v;
        out[i * 4 + 2] = v;
    }
    out
}

/// A readable asset name out of a manifest key.
///
/// The keys are full object paths with the separators flattened and are up to
/// 150 characters long; a Content Drawer full of those is unusable. The last two
/// path-ish segments are what a human recognises.
///
/// # The digest, and the silent overwrite it closes (wave CHAR1a audit)
///
/// The tail alone is **not unique**, and the first content that proved it was
/// the pair of MetaHumans wave CHAR1a.2 imported. Their material keys are
///
/// ```text
/// INF_Built_INF_Dominic_Body_Materials_MI_Body_Baked_MI_Body_Baked
/// INF_Built_INF_Vivian_Body_Materials_MI_Body_Baked_MI_Body_Baked
/// ```
///
/// — identical in their last six segments, because the character's name sits at
/// index 3 and the tail always drops it. The texture writer's path is
/// deliberately DETERMINISTIC (`dest.join(format!("{name}_{slot}.inf_tex"))`, so
/// a re-import overwrites its own output instead of writing `X_1.inf_tex`), so
/// the second character's maps overwrote the first's. Measured on the wave's own
/// import: 32 textures were written and **16 files** exist on disk, every
/// surviving sidecar recording a Vivian source and not one recording Dominic's.
/// Nothing raised an advisory, because from each material's point of view the
/// write succeeded.
///
/// So the name carries a four-hex-digit digest of the WHOLE key. It is still
/// deterministic — the same key gives the same name for ever, which is what the
/// overwrite rule needs — and two keys that differ anywhere now differ here.
fn short_name(key: &str) -> String {
    let parts: Vec<&str> = key.split('_').collect();
    let tail = if parts.len() > 6 {
        parts[parts.len() - 6..].join("_")
    } else {
        key.to_string()
    };
    let clean: String = tail
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect();
    // FNV-1a over the full key, 16 bits of it. Written out rather than pulled
    // from a hasher crate: this is a NAME, and a name whose bytes depend on a
    // dependency's default hasher is a name that can move under a `cargo update`.
    let mut h: u32 = 0x811c_9dc5;
    for b in key.as_bytes() {
        h ^= *b as u32;
        h = h.wrapping_mul(0x0100_0193);
    }
    format!("{clean}_{:04x}", (h ^ (h >> 16)) & 0xffff)
}

impl Material {
    /// What the sidecar records as this asset's source. Not a path — the source
    /// is in another engine's content tree and re-importing it needs the whole
    /// bridge — so it is the UE object path, which is what a human would look up.
    fn source_note(&self) -> String {
        format!("ue:{}", self.key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **TWO CHARACTERS' MATERIALS ARE TWO NAMES** (wave CHAR1a audit).
    ///
    /// The two MetaHuman body materials wave CHAR1a.2 imported differ only in
    /// their fourth underscore segment, which `short_name`'s six-segment tail
    /// always drops — so both resolved to `MI_Body_Baked_MI_Body_Baked`, and the
    /// texture writer's deterministic path (which exists so a re-import
    /// overwrites its own output) made the second character's maps overwrite the
    /// first's. Measured on the wave's own import: 32 textures written, **16
    /// files** on disk, every survivor recording a Vivian source.
    ///
    /// **The mutation**: drop the digest from `short_name`. `a == b` and the
    /// first assertion fails with both names printed. Verified.
    #[test]
    fn two_characters_materials_do_not_shorten_to_one_name() {
        let a = short_name("INF_Built_INF_Dominic_Body_Materials_MI_Body_Baked_MI_Body_Baked");
        let b = short_name("INF_Built_INF_Vivian_Body_Materials_MI_Body_Baked_MI_Body_Baked");
        assert_ne!(
            a, b,
            "two characters' body materials shorten to one asset name, so the \
             second import silently overwrites the first's textures"
        );
        // …and the readable part survives, because a Content Drawer full of
        // digests is the problem this function exists to avoid.
        assert!(a.starts_with("MI_Body_Baked_MI_Body_Baked_"), "{a}");
        assert!(b.starts_with("MI_Body_Baked_MI_Body_Baked_"), "{b}");
        // …and it is DETERMINISTIC, which is what the overwrite rule needs: the
        // same key names the same file on every run, on every machine.
        assert_eq!(
            a,
            short_name("INF_Built_INF_Dominic_Body_Materials_MI_Body_Baked_MI_Body_Baked")
        );
    }
}
