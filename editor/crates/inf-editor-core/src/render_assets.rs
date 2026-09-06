//! **The interactive viewport's asset store** (P18.3): the loose-file half of
//! what `inf_player::vmesh::VmeshRegistry` is for a cooked pack.
//!
//! This closes the oldest documented gap in the engine. Since P4 a `MeshRef.asset`
//! — an entity bound to an imported glTF/OBJ — has drawn a **placeholder
//! primitive** in the editor while the shipped player drew its real geometry, for
//! one structural reason: the viewport thread had no asset database and
//! `.inf_vmesh` was cook-only. [`assets::vmesh`](crate::assets::vmesh) fixed the
//! second half; this module is the first.
//!
//! # Why it lives in Ring 1 and not in the viewport host
//!
//! Verbatim the reasoning that put [`crate::terrain_stream`] here:
//! `inf_viewport::host` is `#[cfg(any(windows, target_os = "macos"))]`, so
//! anything written there is invisible to the Linux CI leg. Keeping the store, the
//! resolution rules and the skinning-pose derivation here — platform neutral, GPU
//! free — means the tests below run on all three OSes and the host is left with
//! nothing but call sites. Same shape as [`EditorTerrainStreams`], same reason.
//!
//! [`EditorTerrainStreams`]: crate::terrain_stream::EditorTerrainStreams
//!
//! # Resolution is computed, never indexed
//!
//! `MeshRef.asset` names a `.inf_mesh` GUID. Its DAG's GUID is
//! [`inf_vgeom::derived_vmesh_id`] of it — the same Ring-0 bijection the cook
//! writes with and the player reads with — so a lookup is an XOR plus a hash-map
//! hit, with no side table that could disagree with the filesystem.
//!
//! # The one deliberate difference from the player, and why
//!
//! The player keys [`inf_render::VgeomAsset::id`] by the derived **GUID**. This
//! store keys it by the derived payload's **content hash**
//! ([`content_key`](LoadedVgeom::id)). The reason is that the two hosts have
//! different guarantees about their content:
//!
//! * a cooked pack is **immutable** — an id names one sequence of bytes forever;
//! * a project's content root is **not** — a re-import, an external tool, or a
//!   `rewrite_payload` can hand the same GUID different geometry mid-session.
//!
//! Both renderer paths cache GPU state by `VgeomAsset::id`
//! (`ClassicVgeomNode::geom` never evicts; `VgeomStreamer` holds pool blocks sized
//! from the source it registered), so a content change under a stable id is a
//! *stale render*, not a reload. Making the id content-addressed makes that
//! unrepresentable: changed bytes are a different asset, the old one leaves
//! `wants` and is fully evicted, and the new one pages in from scratch. It also
//! deduplicates — two entities on two mesh assets with identical geometry share
//! one upload. Everything else in the projection is byte-for-byte the player's
//! rule, which is what the mirror gate pins.
//!
//! # Skeletal meshes
//!
//! [`resolve_skinned`](EditorRenderAssets::resolve_skinned) **is a mirror since
//! P18.5**, and this note used to say the opposite. When P18.3 wrote it the shipped
//! player had no `SkeletalMesh` branch at all, so there was nothing to keep in sync
//! and GPU skinning had only ever been exercised by the `golden_skinned_mesh`
//! headless golden. P18.5 gave the player the matching branch
//! (`inf_player::skinned`), which closed that as the PIE-vs-shipping divergence it
//! had become — so this function and [`skinned_mesh_data`] are now pinned **character
//! for character** against their player twins by `tests/projector_mirror.rs`
//! (`the_skinned_pose_rule_is_identical_in_both_stores`,
//! `the_bind_space_rebuild_is_identical_in_both_stores`), with the receiver
//! (`&mut self` here, `&self` there — a difference in who owns the cache, not in the
//! rule) as the single normalized token. Edit either side and the other must follow.
//!
//! The rule itself (P24.1): the pose the **sim** evaluated for this entity when a
//! state machine published one, else the entity's [`AnimPlayer`] play-head sampled
//! through [`inf_anim::sample_clip`], else the rest pose.
//!
//! The **ONE permitted difference** between the two copies is the receiver:
//! `&mut self` here (the viewport owns its store mutably) against `&self` there
//! (that store memoizes behind its own locks because the projector holds it
//! immutably). That is a difference in who owns the cache, not in the rule, and
//! `projector_mirror.rs` normalizes exactly that token — doc block included —
//! before comparing.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use glam::Mat4;
use inf_anim::{AnimClip, Skeleton};
use inf_ecs::components::{AnimPlayer, AnimStateMachine, SkeletalMesh};
use inf_ecs::pose::EvaluatedPose;
use inf_mesh::MeshAsset;
use inf_render::{SkinnedMeshData, SkinnedVertex};
use inf_vgeom::VgeomSource;
use uuid::Uuid;

/// Depth cap on the content-root walk — deep enough for any content layout,
/// shallow enough that a symlink loop cannot hang project open. Mirrors
/// `terrain_stream`'s cap for the same reason.
const MAX_CONTENT_DEPTH: u32 = 16;

/// The loose asset extensions this store indexes.
///
/// P26.4 added the three material kinds: the viewport resolves a level's
/// `Material.asset` bindings out of this same index, so a `.inf_mat` (or the
/// `.inf_mati` a binding may name, which resolves to its root) and the
/// `.inf_tex` containers behind it have to be findable by GUID here.
///
/// **`inf_sm` arrived with wave CHAR1a.2's preview idle, and its absence is the
/// reason that feature shipped inert for one demo run.** `resolve_skinned` asks
/// `machine_entry_clip` for the machine's entry clip; that asks
/// `load_payload::<StateMachineAsset>`; and a `.inf_sm` that is not in this list
/// is not in the index, so the lookup missed and every character in the viewport
/// stood in its bind pose exactly as before. The gate could not see it — its arm
/// registers the machine directly and never touches an index — and the demo
/// frame could: a just-placed body in a plain T. The arm now takes both doors.
const INDEXED_EXTENSIONS: [&str; 8] = [
    "inf_vmesh",
    "inf_mesh",
    "inf_skel",
    "inf_anim",
    "inf_sm",
    "inf_mat",
    "inf_mati",
    "inf_tex",
];

/// One indexed, opened `.inf_vmesh`.
#[derive(Clone)]
pub struct LoadedVgeom {
    /// The **content-addressed** render key (see the module docs) — the payload's
    /// `ContentHash` as a `u128`. Stable for stable bytes, different for different
    /// bytes, which is exactly the invalidation contract both render nodes' caches
    /// need.
    pub id: u128,
    pub source: Arc<VgeomSource>,
}

/// **A level's virtual-texture content as the editor resolves it** (P26.4) — the
/// viewport's counterpart to `inf_player::MaterialContent`.
///
/// Two hosts, two resolutions, **one order**: the shipped player reads derived
/// `.inf_matd` records out of a pack and this reads authored `.inf_mat` files off
/// disk, because a derived record is a cook product and the editor is what
/// authors its source. What must not differ is the sequence the textures are
/// registered in, and that is `inf_render::registration_order` over
/// [`materials`](Self::materials) on both sides.
#[derive(Debug, Default, Clone)]
pub struct EditorMaterialContent {
    /// Root `.inf_mat` GUID → its three texture GUIDs.
    pub materials: std::collections::BTreeMap<u128, inf_render::VtMaterialMaps>,
    /// `.inf_tex` v2 container bytes by GUID. Owned (a loose file has no mapping
    /// to slice) and `Arc`-shared, so re-resolving a level does not re-copy them.
    pub textures: std::collections::BTreeMap<u128, Arc<Vec<u8>>>,
    /// **The derived RECORD behind each of those materials** (wave CHAR1a.3),
    /// keyed the same way.
    ///
    /// `materials` above is the three texture GUIDs the virtual-texture registry
    /// wants; this is the surface an instance draws with — colour, the three PBR
    /// terms, blend mode, cutoff. It is the same record the cook writes as a
    /// `.inf_matd` and the shipped player reads, produced by the same
    /// `derive_material` call two lines apart, so the viewport and the game
    /// cannot disagree about what a material IS while agreeing about its maps.
    ///
    /// Its consumer is the skinned material SECTION: a slot named by an
    /// `.inf_mesh` is bound by no component, so the projector needs a way to ask
    /// "what does this GUID draw as?" about a material nothing else resolved.
    pub records: std::collections::BTreeMap<u128, inf_asset::DerivedMaterial>,
}

impl EditorMaterialContent {
    /// One texture's bytes as the registration door takes them.
    pub fn source(&self, guid: u128) -> Option<Arc<dyn inf_render::VtTileSource>> {
        let bytes = self.textures.get(&guid)?.clone();
        Some(bytes as Arc<dyn inf_render::VtTileSource>)
    }

    /// Whether this level binds any material with any texture at all.
    pub fn is_empty(&self) -> bool {
        self.materials.is_empty()
    }
}

/// **One skinned mesh's drawn sections** (wave CHAR1a.3): `(first index, index
/// count, material guid)` per material slot, EMPTY for a mesh whose submeshes
/// all want one slot.
///
/// A named type rather than an inline tuple because the map it keys is a
/// `Mutex<HashMap<_, Arc<Vec<_>>>>` on one side of the mirror, which is exactly
/// the shape `clippy::type_complexity` exists to stop — the same reason
/// `RestPaletteKey` next to it has a name.
///
/// **MIRROR**: keep byte-identical with the other host's.
type SkinnedSections = Arc<Vec<(u32, u32, Option<u128>)>>;

/// The key a shared palette is cached under: `(mesh, skeleton, entry clip)`.
///
/// A named type rather than an inline tuple because the map it keys is a
/// `Mutex<HashMap<_, Arc<Vec<Mat4>>>>` on one side of the mirror, which is
/// exactly the shape `clippy::type_complexity` exists to stop.
///
/// The third term arrived with wave CHAR1a.2's preview idle. `None` is the rest
/// pose — every caller before that wave — and `Some(clip)` is that clip's t = 0
/// pose, which is what `resolve_skinned` builds for an agent carrying the same
/// state machine. One entry per distinct ANSWER, which is what "derived once per
/// `(mesh, skeleton)`" was always claiming and could no longer deliver on its
/// own: two crowds of the same body running different machines share a mesh and
/// a rig and do not share a pose.
type RestPaletteKey = (Uuid, Uuid, Option<Uuid>);

/// A skeletal mesh resolved to something the skinned pass can draw.
#[derive(Clone)]
pub struct SkinnedDraw {
    /// Bind-space geometry, shared between every entity using the same
    /// `(mesh, skeleton)` pair.
    pub mesh: Arc<SkinnedMeshData>,
    /// `global · inverse_bind` per joint, for this entity's current pose.
    ///
    /// **`Arc`, since NPC1b**: the crowd's non-posing tiers all resolve to one
    /// rest palette per `(mesh, skeleton)`, derived once here and handed to every
    /// agent that shares it, so the renderer's atlas deduplicates it by pointer
    /// identity instead of uploading a thousand copies of one answer.
    pub palette: Arc<Vec<Mat4>>,
    /// The `(mesh, skeleton)` pair, so a caller can deduplicate
    /// [`RenderScene::skinned_meshes`](inf_render::RenderScene::skinned_meshes)
    /// entries across instances.
    pub key: (Uuid, Uuid),
    /// **The mesh's material SECTIONS** (wave CHAR1a.3): one `(range, material)` per material slot, or EMPTY for a mesh whose submeshes all want
    /// one slot — which is every committed character in this tree.
    ///
    /// The RANGE addresses the mesh's own index buffer; the `Uuid` is the slot's
    /// material, which the PROJECTOR resolves into a surface and a virtual-texture
    /// set — a `VtTextureSet` is a warm-gated snapshot of a per-frame residency,
    /// and this value is cached for the life of the store.
    pub sections: SkinnedSections,
}

/// Every loose render asset the interactive viewport can draw, indexed by GUID
/// over a project's content root.
#[derive(Default)]
pub struct EditorRenderAssets {
    /// Where loose assets are looked up. `None` (the default) disables the whole
    /// store — a primitive-only document is unaffected either way, which is what
    /// keeps a viewport with no project byte-identical to its pre-P18.3 self.
    content_root: Option<PathBuf>,
    /// Asset GUID → loose file, rebuilt when the root changes and, lazily, when a
    /// lookup misses (see [`resolve_path`](Self::resolve_path)).
    index: HashMap<Uuid, PathBuf>,
    /// GUIDs a rescan has already failed to find, so a genuinely dangling
    /// reference costs **one** directory walk rather than one per frame. Cleared
    /// whenever the index is rebuilt.
    rescanned_for: HashSet<Uuid>,
    /// Opened vmesh sources by **mesh** GUID. `None` is a negative entry: this
    /// mesh has no usable DAG, so stop trying every frame.
    vgeom: HashMap<Uuid, Option<LoadedVgeom>>,
    /// Bind-space skinned geometry by **mesh GUID** — see `skinned_geometry`
    /// (round-2 finding R2-1). The skeleton is not part of the value and was
    /// never part of it.
    skinned: HashMap<Uuid, Option<Arc<SkinnedMeshData>>>,
    /// Decoded skeletons by GUID (shared by every entity bound to one).
    skeletons: HashMap<Uuid, Option<Arc<Skeleton>>>,
    /// Decoded clips by GUID.
    clips: HashMap<Uuid, Option<Arc<AnimClip>>>,
    /// **Decoded `.inf_sm` machines by GUID** (wave CHAR1a.2) — read for one
    /// thing only: which clip the ENTRY state plays, so an unplayed character can
    /// be previewed standing in its idle instead of its bind pose. The machine is
    /// never stepped here; stepping it is the fixed step's job and this is the
    /// projection.
    state_machines: HashMap<Uuid, Option<Arc<inf_anim::StateMachine>>>,
    /// **The rest-pose palette per `(mesh, skeleton)` pair** (wave NPC1b) — the
    /// one a crowd's non-posing tiers all draw.
    ///
    /// Keyed by the pair and not by the mesh alone, unlike `skinned` above: the
    /// value is `skinning_matrices` over a SKELETON's rest pose, so it is a
    /// property of the rig and re-binding one mesh to two rigs is two answers.
    /// **The drawn sections of each mesh**, cached hit or miss (wave CHAR1a.3).
    ///
    /// Keyed by the MESH alone, exactly like the geometry cache beside it and for
    /// the same reason: the ranges and the slot materials are properties of the
    /// `.inf_mesh`, not of the rig it is played on.
    sections: HashMap<Uuid, SkinnedSections>,
    rest_palettes: HashMap<RestPaletteKey, Arc<Vec<Mat4>>>,
    /// **Scatter geometry by mesh GUID** (wave TER2b) — the flat pull arrays a
    /// `PcgKind`'s `.inf_mesh` becomes so that scattered ground cover draws its
    /// authored shape instead of the placeholder cube every instance drew from
    /// P18.5 until this wave. `None` is a negative entry on the `vgeom` terms:
    /// this GUID has no usable geometry, so stop opening the file.
    scatter: HashMap<Uuid, Option<Arc<inf_render::ScatterGeometry>>>,
    /// **Bumped whenever the bytes behind a GUID might have moved** (P26.4
    /// audit): the index was rebuilt, or the content root changed.
    ///
    /// A consumer that caches something *derived* from this store — the
    /// viewport's virtual-texture level is the one that exists — cannot gate on
    /// the document version (a gizmo drag bumps that per input event, and
    /// building a VT level creates an atlas) and cannot gate on the binding SET
    /// alone: re-importing a `.inf_tex`, or editing a material's graph, changes
    /// neither. Before this counter, `sync_vt_bindings` did exactly that and its
    /// own doc claimed otherwise — *"`refresh_asset_index` forces a
    /// re-projection and clears the set"*, which nothing did — so the viewport's
    /// atlas held the bytes it read the first time for the rest of the session.
    index_generation: u64,
}

impl EditorRenderAssets {
    pub fn new() -> Self {
        Self::default()
    }

    /// Point the store at a project's content root (or `None` to disable it).
    ///
    /// Rescans the loose-asset index and drops **everything** loaded, so a project
    /// switch can never serve the previous project's geometry.
    pub fn set_content_root(&mut self, root: Option<PathBuf>) {
        self.clear();
        self.index = match &root {
            Some(dir) => content_paths_by_guid(dir),
            None => HashMap::new(),
        };
        self.rescanned_for.clear();
        self.content_root = root;
        self.index_generation += 1;
    }

    /// The active content root.
    pub fn content_root(&self) -> Option<&Path> {
        self.content_root.as_deref()
    }

    /// **How many times the bytes behind a GUID might have moved** — see
    /// [`index_generation`](Self::index_generation)'s field docs (P26.4 audit).
    ///
    /// Monotone, and the only correct third term for a consumer caching a
    /// *derived* GPU object: the document version is too noisy and the binding
    /// set is too coarse.
    #[inline]
    pub fn index_generation(&self) -> u64 {
        self.index_generation
    }

    /// Number of indexed loose assets.
    pub fn index_len(&self) -> usize {
        self.index.len()
    }

    /// Number of meshes with an opened DAG (negative entries excluded).
    pub fn loaded_vgeom(&self) -> usize {
        self.vgeom.values().filter(|v| v.is_some()).count()
    }

    /// Number of opened bind-space skinned meshes.
    pub fn loaded_skinned(&self) -> usize {
        self.skinned.values().filter(|v| v.is_some()).count()
    }

    /// Rebuild the loose-asset index **and drop every opened payload**.
    ///
    /// Called when the content database changed (an import landed, the watcher saw
    /// an external edit). Unlike terrain's `refresh_index`, which keeps live
    /// streams because a terrain's *tiles* are re-read per page, this drops what it
    /// holds: a `.inf_vmesh` is opened once and then only sliced, so a payload
    /// rewritten under the same GUID would otherwise be served from the old
    /// mapping forever. Re-opening costs a header + page-directory parse per
    /// referenced asset — a few hundred bytes each — which is the price of never
    /// rendering stale geometry.
    pub fn refresh_index(&mut self) {
        // Bumped before the early-out, because "there is no root" is itself a
        // state a derived cache has to be told about (P26.4 audit).
        self.index_generation += 1;
        let Some(dir) = self.content_root.clone() else {
            return;
        };
        self.index = content_paths_by_guid(&dir);
        self.rescanned_for.clear();
        self.drop_loaded();
    }

    /// Release every opened payload, keeping the index and the root.
    ///
    /// The level-switch door (File ▸ Open / File ▸ New): the store is keyed on
    /// *asset* GUIDs rather than entity GUIDs, so nothing here is invalidated by a
    /// new document — but every byte it holds belongs to the old one's working set
    /// and nothing else would ever free it.
    pub fn clear(&mut self) {
        self.drop_loaded();
    }

    /// Drop everything opened for meshes **not** in `keep` — the per-projection
    /// audit ([`EditorTerrainStreams::retain_only`]'s twin).
    ///
    /// [`EditorTerrainStreams::retain_only`]: crate::terrain_stream::EditorTerrainStreams::retain_only
    ///
    /// `keep` is **every asset GUID the projection referenced** — `MeshRef.asset`,
    /// `SkeletalMesh.mesh`, `SkeletalMesh.skeleton`, `AnimPlayer.clip`. An asset
    /// that left the document — deleted, hidden, unbound, or belonging to a level
    /// that has since been replaced — is a whole `.inf_vmesh` mapping, or decoded
    /// skinned geometry, held for nothing. This is the only place that knows which
    /// assets are live, so it is the only place that can do it. (The P16.4b lesson,
    /// in mesh form.)
    ///
    /// Passing the *whole* referenced set rather than just meshes is what lets
    /// skeletons and clips be collected too: they are keyed independently of the
    /// mesh that reaches them, so a mesh-only `keep` would leak every skeleton a
    /// session ever posed.
    pub fn retain_only(&mut self, keep: impl IntoIterator<Item = Uuid>) {
        let live: HashSet<Uuid> = keep.into_iter().collect();
        self.vgeom.retain(|mesh, _| live.contains(mesh));
        self.skinned.retain(|mesh, _| live.contains(mesh));
        self.skeletons.retain(|id, _| live.contains(id));
        self.clips.retain(|id, _| live.contains(id));
        self.state_machines.retain(|id, _| live.contains(id));
        self.scatter.retain(|mesh, _| live.contains(mesh));
        // **The crowd's shared rest palette is a cache like the others** (NPC1b
        // audit). Wave NPC1b added `rest_palettes` and wired it into none of this
        // store's three eviction doors, so it was the one map a session could
        // only grow — the P21 "a pin with no release is a leak with a deadline"
        // shape, and here also a *staleness*: a re-import drops the decoded
        // skeleton and the palette derived from it would have survived.
        self.rest_palettes.retain(|(mesh, skel, entry), _| {
            live.contains(mesh)
                && live.contains(skel)
                && entry.is_none_or(|clip| live.contains(&clip))
        });
    }

    /// Forget one mesh asset's opened payloads (a targeted re-import invalidation).
    pub fn invalidate(&mut self, mesh_id: Uuid) {
        self.vgeom.remove(&mesh_id);
        self.skinned.remove(&mesh_id);
        self.scatter.remove(&mesh_id);
        self.rescanned_for.clear();
    }

    fn drop_loaded(&mut self) {
        self.vgeom.clear();
        self.skinned.clear();
        self.skeletons.clear();
        self.clips.clear();
        self.state_machines.clear();
        self.scatter.clear();
        // The rest palette is DERIVED from a skeleton this line just dropped
        // (NPC1b audit). Left standing it would outlive the bytes it was built
        // from: `rescan` calls this, so a re-imported `.inf_skel` would give the
        // per-agent path the new rig and the crowd's shared path the old one —
        // which is exactly the disagreement
        // `the_shared_palette_is_the_one_the_per_agent_path_would_have_built`
        // asserts cannot happen, from the one direction that arm cannot see.
        self.rest_palettes.clear();
    }

    // ── resolution ──────────────────────────────────────────────────────────

    /// Resolve a `MeshRef.asset` to its virtualized geometry.
    ///
    /// **MIRROR** of `inf_player::vmesh::VmeshRegistry::resolve`: the derived id is
    /// computed the same way, resolution is deliberately *independent of the render
    /// setting* (the tier decides whether the meshlet path or the classic
    /// discrete-LOD fallback draws the resolved content — the scene content is the
    /// same either way), and `None` means "no DAG ⇒ the primitive placeholder",
    /// which stays correct content for a primitive `MeshRef`.
    ///
    /// Opens the payload on first use and caches both hits and misses.
    pub fn resolve_vgeom(&mut self, mesh_id: Uuid) -> Option<LoadedVgeom> {
        if let Some(hit) = self.vgeom.get(&mesh_id) {
            return hit.clone();
        }
        let loaded = self.open_vgeom(mesh_id);
        self.vgeom.insert(mesh_id, loaded.clone());
        loaded
    }

    fn open_vgeom(&mut self, mesh_id: Uuid) -> Option<LoadedVgeom> {
        let vmesh_id = inf_vgeom::derived_vmesh_id(inf_asset::AssetId(mesh_id)).uuid();
        let path = self.resolve_path(vmesh_id)?;
        // **Is this DAG the one this mesh would derive?** (wave FIX2, carried 38.)
        //
        // This function used to read the payload and ask nothing about it, so a
        // mesh rewritten under the same GUID — which is exactly what a re-import,
        // a DCC save and `inf island build` all do — kept drawing the DAG built
        // from the PREVIOUS bytes until the project-open sweep reached it, about
        // two and a half minutes into a session on the island. The ROAD1b audit
        // photographed it: two of the four road DAGs in the wave's own proof frame
        // were the pre-paving build, so the frame showed kerbs and double yellow
        // and no carriageway, and nothing anywhere said so.
        //
        // Stale is REFUSED, not drawn: the caller's `None` arm draws nothing (the
        // placeholder cube is gone), which is the honest frame for "the project
        // does not have current geometry for this mesh yet". The sweep replaces
        // it and `refresh_index` drops this negative entry.
        //
        // `derived_is_current` is the SAME comparison `plan_vmesh` makes, so the
        // viewport refuses exactly what the sweep would rebuild.
        //
        // A `.inf_vmesh` with no readable sidecar is not one this editor derived
        // (a cooked artifact copied in by hand, say), and a mesh with none is not
        // in the database at all. Neither is a staleness CLAIM, so neither is a
        // refusal — the `?` chain below falls through to the read, which is what
        // the editor did before this wave.
        let stale = self.resolve_path(mesh_id).and_then(|mesh_path| {
            let mesh = inf_asset::AssetSidecar::load(&mesh_path).ok()?;
            let derived = inf_asset::AssetSidecar::load(&path).ok()?;
            (!crate::assets::vmesh::derived_is_current(&mesh, &derived)).then_some(mesh_path)
        });
        if let Some(mesh_path) = stale {
            tracing::warn!(
                "inf-editor-core: .inf_vmesh {} was derived from different bytes than {} \
                 holds now, so it is NOT drawn — the project-open meshlet sweep rebuilds \
                 it, and until it does this mesh has no geometry the project can honestly \
                 show",
                path.display(),
                mesh_path.display()
            );
            return None;
        }
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!("inf-editor-core: .inf_vmesh {}: {e}", path.display());
                return None;
            }
        };
        // The render key is the payload's content hash (see the module docs), read
        // from the bytes actually loaded rather than from a sidecar that could be
        // stale relative to them.
        let id = inf_asset::ContentHash::of(&bytes).0;
        match VgeomSource::from_payload(bytes) {
            Ok(source) => Some(LoadedVgeom {
                id,
                source: Arc::new(source),
            }),
            Err(e) => {
                tracing::warn!("inf-editor-core: bad .inf_vmesh {}: {e}", path.display());
                None
            }
        }
    }

    /// Resolve a [`SkeletalMesh`] (+ its optional [`AnimPlayer`], + the pose the
    /// sim evaluated for this entity) to a drawable skinned instance: bind-space
    /// geometry plus the skinning palette for the pose that is actually in force.
    ///
    /// The pose rule, in order:
    ///  1. no skeleton asset, or a skeleton with no joints ⇒ `None` (the caller
    ///     keeps the placeholder — a skinned mesh with no skeleton is not a
    ///     renderable thing);
    ///  2. a `posed` entry from the sim ([`inf_ecs::pose`]) whose skeleton is the
    ///     one bound here and whose joint count matches ⇒ **that pose**, because
    ///     an `AnimStateMachine` beats an `AnimPlayer` and this is where "the
    ///     machine wins" stops being a comment;
    ///  3. no sim pose and no `AnimPlayer` clip, but an `AnimStateMachine` whose
    ///     entry state names a clip that resolves ⇒ **that clip at t = 0**, the
    ///     preview idle (wave CHAR1a.2);
    ///  4. otherwise no `AnimPlayer`, or one with no clip, or a clip that will
    ///     not resolve ⇒ **rest pose** ([`inf_anim::Pose::rest`]);
    ///  5. otherwise the clip sampled at the play-head, honouring `looping`.
    ///
    /// Rule 2's two guards are not defensive noise. The projector reads the pose
    /// out of a sim-side store keyed by entity and resolves the skeleton out of
    /// its own asset store: if a character were re-bound to a different rig
    /// between the fixed step and the projection, applying the old pose would
    /// deform it by another skeleton's hierarchy — silently, and only on the
    /// frames where it happened. A mismatch falls through to rules 3/4, which is
    /// always a pose the bound skeleton can wear.
    ///
    /// Rule 4 is what makes a freshly dropped character *visible* rather than
    /// invisible-until-you-press-play, and it is deliberately the same fallback
    /// the components document (`AnimPlayer::clip: None → the bind pose`).
    ///
    /// **Rule 3 is why the viewport stopped drawing a bind pose** (wave CHAR1a.2;
    /// CHAR1a carried it as item 72 with a photograph). Outside Play there is no
    /// sim pose, and a character authored with a state machine carries no
    /// `AnimPlayer` at all — so every rig in the editor fell to rule 4 and stood
    /// in its bind pose: a T on the generated rig, an A on the mannequin's. The
    /// machine already says what the character does when nothing is happening;
    /// that is what an entry state IS. Sampling its clip at t = 0 is the smallest
    /// true answer.
    ///
    /// It is **render-side only**: no sim state is read or written, nothing here
    /// can move a trace byte, and a character the sim IS posing still takes
    /// rule 2 — so the P29 gates that read sim poses are untouched by it.
    ///
    /// Deliberately t = 0 rather than a clock. A preview that animated would make
    /// two screenshots of the same document differ, and the editor has no
    /// play-head to be at.
    pub fn resolve_skinned(
        &mut self,
        sm: &SkeletalMesh,
        player: Option<&AnimPlayer>,
        posed: Option<&EvaluatedPose>,
        machine: Option<&AnimStateMachine>,
    ) -> Option<SkinnedDraw> {
        let mesh_id = sm.mesh?;
        let skeleton_id = sm.skeleton?;
        let skeleton = self.skeleton(skeleton_id)?;
        if skeleton.is_empty() {
            return None;
        }
        let mesh = self.skinned_geometry(mesh_id, skeleton_id)?;
        let from_sim =
            posed.filter(|p| p.skeleton == skeleton_id && p.pose.len() == skeleton.len());
        let pose = match (from_sim, player.and_then(|p| p.clip.map(|c| (p, c)))) {
            (Some(p), _) => p.pose.clone(),
            (None, Some((p, clip_id))) => match self.clip(clip_id) {
                Some(clip) => inf_anim::sample_clip(&skeleton, &clip, p.t as f32, p.looping),
                None => inf_anim::Pose::rest(&skeleton),
            },
            (None, None) => match self
                .machine_entry_clip(machine.and_then(|m| m.sm))
                .and_then(|clip_id| self.clip(clip_id))
            {
                Some(clip) => inf_anim::sample_clip(&skeleton, &clip, 0.0, true),
                None => inf_anim::Pose::rest(&skeleton),
            },
        };
        Some(SkinnedDraw {
            mesh,
            palette: Arc::new(inf_anim::skinning_matrices(&skeleton, &pose)),
            key: (mesh_id, skeleton_id),
            sections: self.skinned_sections(mesh_id),
        })
    }

    /// **The tier's shared pose** (wave NPC1b): the palette every crowd agent
    /// that is not evaluating one draws, derived ONCE per `(mesh, skeleton)`.
    ///
    /// It is deliberately not a new pose rule. A crowd agent the ladder took off
    /// the pose path has no [`EvaluatedPose`] — `step_pose_evaluation` rebuilds
    /// the store from the tier's own target set, so a `Far` agent has no entry —
    /// and carries no `AnimPlayer`, so `resolve_skinned` falls through rules 2 and
    /// 5 and returns exactly this. What changes is how many times it is derived:
    /// once, instead of once per agent per frame, and the four joint-length
    /// allocations wall 3 counted stop happening for that tier entirely.
    /// `the_shared_palette_is_the_one_the_per_agent_path_would_have_built` is the
    /// arm that keeps the two answers equal.
    ///
    /// **It takes the machine since wave CHAR1a.2, and it has to.** The per-agent
    /// path now prefers a machine's entry pose over the rest pose (rule 3), so a
    /// crowd whose agents all carry one would draw an idle at the tiers that pose
    /// and a bind pose at the tiers that do not — one character with two
    /// silhouettes, swapping at the tier radius. The cache key grew the entry clip
    /// with it, so it is still one palette per distinct answer.
    pub fn resolve_skinned_shared(
        &mut self,
        sm: &SkeletalMesh,
        machine: Option<&AnimStateMachine>,
    ) -> Option<SkinnedDraw> {
        let mesh_id = sm.mesh?;
        let skeleton_id = sm.skeleton?;
        let skeleton = self.skeleton(skeleton_id)?;
        if skeleton.is_empty() {
            return None;
        }
        let mesh = self.skinned_geometry(mesh_id, skeleton_id)?;
        let entry = self.machine_entry_clip(machine.and_then(|m| m.sm));
        Some(SkinnedDraw {
            mesh,
            palette: self.rest_palette(&skeleton, (mesh_id, skeleton_id, entry)),
            key: (mesh_id, skeleton_id),
            sections: self.skinned_sections(mesh_id),
        })
    }

    /// The shared palette for one `(mesh, skeleton, entry clip)` triple, cached.
    ///
    /// With no entry clip this is the rest pose it has been since NPC1b; with one
    /// it is that clip's t = 0 pose, which is the answer `resolve_skinned`'s
    /// rule 3 builds for an agent carrying the same machine. Deriving it here
    /// rather than at the call site is what keeps the two equal.
    fn rest_palette(&mut self, skeleton: &Skeleton, key: RestPaletteKey) -> Arc<Vec<Mat4>> {
        if let Some(hit) = self.rest_palettes.get(&key) {
            return hit.clone();
        }
        let pose = match key.2.and_then(|clip_id| self.clip(clip_id)) {
            Some(clip) => inf_anim::sample_clip(skeleton, &clip, 0.0, true),
            None => inf_anim::Pose::rest(skeleton),
        };
        let built = Arc::new(inf_anim::skinning_matrices(skeleton, &pose));
        self.rest_palettes.insert(key, built.clone());
        built
    }

    /// **The entry clip of a state machine**, or `None` when there is no machine,
    /// no asset, or an entry state that plays something other than a single clip.
    ///
    /// MIRROR: keep this byte-identical with the other store's, doc block
    /// included — `projector_mirror.rs` compares it, because it is the input to
    /// the preview pose and two hosts that disagreed about it would disagree
    /// about how an unplayed character stands.
    ///
    /// A blend space or a sub-machine returns `None` on purpose: they have no
    /// single clip to sample at t = 0, and inventing one (the first sample of a
    /// blend at its default parameter) would be a preview of a pose the machine
    /// never actually enters.
    fn machine_entry_clip(&mut self, sm: Option<Uuid>) -> Option<Uuid> {
        let machine = self.state_machine(sm?)?;
        let state = machine.states.get(machine.entry)?;
        match &state.motion {
            inf_anim::state_machine::Motion::Clip(clip) => Some(Uuid::from_bytes(*clip)),
            _ => None,
        }
    }

    /// A decoded `.inf_sm`, cached (hit or miss) by GUID — the twin of
    /// [`clip`](Self::clip), for the same reason and with the same negative
    /// caching.
    fn state_machine(&mut self, id: Uuid) -> Option<Arc<inf_anim::StateMachine>> {
        if let Some(hit) = self.state_machines.get(&id) {
            return hit.clone();
        }
        let loaded = self
            .load_payload::<inf_anim::StateMachineAsset>(id)
            .map(|a| Arc::new(a.machine));
        self.state_machines.insert(id, loaded.clone());
        loaded
    }

    /// A decoded skeleton, cached (hit or miss) by GUID.
    pub fn skeleton(&mut self, id: Uuid) -> Option<Arc<Skeleton>> {
        if let Some(hit) = self.skeletons.get(&id) {
            return hit.clone();
        }
        let loaded = self
            .load_payload::<inf_anim::SkeletonAsset>(id)
            .map(|a| Arc::new(a.skeleton));
        self.skeletons.insert(id, loaded.clone());
        loaded
    }

    /// A decoded animation clip, cached (hit or miss) by GUID.
    pub fn clip(&mut self, id: Uuid) -> Option<Arc<AnimClip>> {
        if let Some(hit) = self.clips.get(&id) {
            return hit.clone();
        }
        let loaded = self
            .load_payload::<inf_anim::AnimClipAsset>(id)
            .map(|a| Arc::new(a.clip));
        self.clips.insert(id, loaded.clone());
        loaded
    }

    /// **The drawn sections of one skeletal mesh**, cached — `(first index, index
    /// count, material guid)` per material slot, EMPTY when the mesh wants one
    /// slot, which is every committed character in this tree.
    ///
    /// # Why the store answers this and not the projector
    ///
    /// The ranges address the concatenated index buffer `skinned_mesh_data`
    /// builds, and the slot materials come from the `.inf_mesh`'s own v3 table.
    /// Both are properties of the ASSET, so they are derived once per mesh here —
    /// beside the geometry, under the same cache, out of the same decode — rather
    /// than per entity per frame in a projector that would then have to re-derive
    /// the concatenation rule and hope it matched.
    ///
    /// What is NOT here is the surface. A section's colour, PBR scalars, blend
    /// mode and virtual-texture set are resolved by the PROJECTOR, from the same
    /// per-host material map that decides what is registered for the level: a
    /// `VtTextureSet` is a warm-gated snapshot of a per-frame residency, and this
    /// value is cached for the life of the store.
    ///
    /// **MIRROR**: keep byte-identical with the other host's, doc block included
    /// (`projector_mirror.rs`).
    fn skinned_sections(&mut self, mesh_id: Uuid) -> SkinnedSections {
        if let Some(hit) = self.sections.get(&mesh_id) {
            return hit.clone();
        }
        let built = Arc::new(self.build_skinned_sections(mesh_id));
        self.sections.insert(mesh_id, built.clone());
        built
    }

    /// Decode a mesh asset and derive its sections.
    ///
    /// **MIRROR** — see [`skinned_sections`](Self::skinned_sections).
    fn build_skinned_sections(&mut self, mesh_id: Uuid) -> Vec<(u32, u32, Option<u128>)> {
        let Some(mesh) = self.load_payload::<MeshAsset>(mesh_id) else {
            return Vec::new();
        };
        mesh.skinned_sections()
            .into_iter()
            .map(|(first, count, slot)| {
                (
                    first,
                    count,
                    mesh.material_for_slot(slot).map(|a| a.uuid().as_u128()),
                )
            })
            .collect()
    }

    /// Bind-space skinned geometry for a `(mesh, skeleton)` pair, cached.
    /// **Keyed by the MESH alone** (round-2 finding R2-1), matching the player
    /// mirror, whose own doc states the rule.
    ///
    /// The cache was keyed `(mesh, skeleton)` and its value never looked at the
    /// skeleton: `skinned_mesh_data` reads the `.inf_mesh`'s bind-space
    /// vertices and its per-vertex `VertexSkin` stream, both of which are
    /// properties of the mesh asset. So N characters sharing one mesh on N
    /// different rigs — the whole point of the P24 modular-rigging work — held
    /// N byte-identical copies in RAM, uploaded N times to the GPU (the skinned
    /// pass keys its upload on `Arc` identity, and these were N different
    /// pointers), and voxelized N times into GI.
    ///
    /// The half-blind `retain_only` was the other half of the same key: it
    /// dropped an entry only when the MESH left, so an entry whose skeleton had
    /// gone survived under a key nothing could ever look up again. With one
    /// term there is nothing left to be blind about.
    fn skinned_geometry(
        &mut self,
        mesh_id: Uuid,
        _skeleton_id: Uuid,
    ) -> Option<Arc<SkinnedMeshData>> {
        if let Some(hit) = self.skinned.get(&mesh_id) {
            return hit.clone();
        }
        let built = self
            .load_payload::<MeshAsset>(mesh_id)
            .and_then(|m| skinned_mesh_data(&m))
            .map(Arc::new);
        self.skinned.insert(mesh_id, built.clone());
        built
    }

    // ── virtual textures (P26.4, clause 0) ──────────────────────────────────

    /// **What the viewport's live virtual-texture level is a function of**
    /// (P26.5) — the cache key its rebuild is gated on, as a value.
    ///
    /// The viewport projects on every document version and building a VT level
    /// creates an atlas, so the rebuild has to be gated. Both terms are
    /// load-bearing and the P26.4 audit found the second one missing:
    ///
    /// * the **binding set**, because that is what decides *which* textures are
    ///   registered and in what order;
    /// * the **index generation**, because a re-imported `.inf_tex` changes
    ///   neither the binding set nor the document version — so without it the
    ///   viewport held the bytes it read the first time for the rest of the
    ///   session, on a path Ring 2 drives from two commands.
    ///
    /// It is a **value with a name** rather than two fields compared inline, and
    /// that is the whole point of the change: the host's early-out is now one
    /// `!=` over a type, so a third term cannot be added to the rebuild without
    /// coming through here, and the rule is testable on every CI leg instead of
    /// only where a window exists (`EngineHost::new` takes a real surface, so
    /// nothing headless can execute the host's own copy — see
    /// `tests/vt_level_key.rs`).
    pub fn vt_level_key(&mut self, doc: &crate::scene::SceneDoc) -> VtLevelKey {
        let mut bindings: BTreeSet<Uuid> = BTreeSet::new();
        // The skinned meshes, collected first and asked afterwards: the slot
        // lookup borrows `self` mutably (it caches) and the entity walk borrows
        // the document.
        let mut skinned: BTreeSet<Uuid> = BTreeSet::new();
        let world = doc.world();
        let w = world.world();
        for &guid in doc.order() {
            let Some(entity) = world.entity_of(guid) else {
                continue;
            };
            if let Some(asset) = w
                .get::<inf_ecs::components::Material>(entity)
                .and_then(|m| m.asset)
            {
                bindings.insert(asset);
            }
            // TER2a: a terrain's four splat layers bind materials too, through
            // the one door `Terrain::layer_materials`. The ground is by far the
            // largest textured surface a world has, and it was invisible to this
            // walk — so a terrain layer could name a material and the viewport
            // would never register it, never build a level for it, and show the
            // flat albedo for ever.
            if let Some(terrain) = w.get::<inf_ecs::components::Terrain>(entity) {
                bindings.extend(terrain.layer_materials());
            }
            skinned.extend(
                w.get::<inf_ecs::components::SkeletalMesh>(entity)
                    .and_then(|s| s.mesh),
            );
        }
        // **AND THE SKINNED MESH'S MATERIAL SLOTS** (wave CHAR1a.3). A
        // slot is named by the `.inf_mesh`, not by any component, so a body
        // whose face wants twelve materials binds exactly ONE of them here
        // and eleven surfaces registered nothing, resolved nothing, and drew
        // off their scalar colour. It is the same hole TER2a found for a
        // terrain's four splat layers, one asset kind over.
        //
        // Empty for every mesh written before the slot table existed, so
        // this adds nothing to any level in this repository and the
        // registration SEQUENCE -- which `VtTextures::want_floor` is a pure
        // function of -- is unmoved for all of them.
        for mesh in skinned {
            for (_, _, mat) in self.skinned_sections(mesh).iter() {
                bindings.extend(mat.map(Uuid::from_u128));
            }
        }
        VtLevelKey {
            bindings,
            index_generation: self.index_generation(),
        }
    }

    /// **Resolve a level's `Material.asset` bindings into virtual-texture
    /// content** — the editor half of what a cooked pack's
    /// `PackLevelSource::material_content` is for the shipped player.
    ///
    /// The two hosts *must* resolve differently and *must not* order
    /// differently. Differently, because the editor has loose authored files and
    /// no `.inf_matd` (a derived record is a cook product); identically, because
    /// `inf_render::registration_order` is the one walk both then hand to the
    /// registry, and `VtTextures::want_floor` is a pure function of it.
    ///
    /// **A binding that does not name a `.inf_mat` contributes nothing**, and
    /// that is a mirror decision rather than a limitation. A `.inf_mati` binding
    /// resolves *nowhere* on the shipped path — the cook derives a `.inf_matd`
    /// for `AssetKind::Material` only and raises its own advisory for the
    /// wrong-kind case (the P26.3b audit's finding) — so a viewport that walked
    /// the instance chain and textured the surface would show the author
    /// something the shipped build does not. `scene_apply_material` persists the
    /// ROOT since P26.3b, so this is the pre-existing-content case only, and the
    /// honest picture of it is the untextured surface the cook already warns
    /// about.
    ///
    /// A binding that resolves to nothing likewise contributes nothing: the
    /// surface renders off its scalar attributes, the permanent no-texture path.
    pub fn material_content(
        &mut self,
        bindings: impl IntoIterator<Item = Uuid>,
    ) -> EditorMaterialContent {
        let mut out = EditorMaterialContent::default();
        let mut seen: HashSet<Uuid> = HashSet::new();
        for bound in bindings {
            if !seen.insert(bound) {
                continue;
            }
            let Some(root) = self.material_root(bound) else {
                continue;
            };
            let Some(mat) = self.load_payload::<inf_material::MaterialAsset>(root) else {
                continue;
            };
            // THE ONE DOOR (P22.2's law): the same flattening the cook and the
            // PIE payload builder call, so the viewport cannot resolve a
            // material's maps differently from the two wires.
            let rec = inf_material::derive_material(&mat);
            out.records.insert(root.as_u128(), rec.clone());
            out.materials.insert(
                root.as_u128(),
                inf_render::VtMaterialMaps {
                    albedo: rec.albedo.map(|t| t.uuid().as_u128()),
                    normal: rec.normal.map(|t| t.uuid().as_u128()),
                    orm: rec.orm.map(|t| t.uuid().as_u128()),
                    // Wave G: bound exactly as the player binds it
                    // (`inf_player::vt_materials`), through the record's own
                    // `detail_scale_q8` so neither host owns a second
                    // metres→8.8 conversion — see the note there.
                    detail: rec.detail.map(|t| t.uuid().as_u128()),
                    detail_scale_q8: rec.detail_scale_q8(),
                    // Wave ROAD1's physical tiling rate, bound the same way and
                    // for the same reason: `uv_tiling_q8` is on the RECORD, so
                    // there is one metres->8.8 conversion in the tree and not
                    // one per host.
                    uv_tiling_q8: rec.uv_tiling_q8(),
                },
            );
            for tex in rec.texture_dependencies() {
                let g = tex.uuid();
                if out.textures.contains_key(&g.as_u128()) {
                    continue;
                }
                let Some(path) = self.resolve_path(g) else {
                    continue;
                };
                match std::fs::read(&path) {
                    Ok(bytes) => {
                        out.textures.insert(g.as_u128(), Arc::new(bytes));
                    }
                    Err(e) => {
                        tracing::warn!("inf-editor-core: .inf_tex {}: {e}", path.display())
                    }
                }
            }
        }
        out
    }

    /// The `.inf_mat` a binding names, or `None` for anything else.
    ///
    /// The kind comes from the FILE EXTENSION rather than from a sidecar field:
    /// this store indexes GUID → path and never opens a sidecar twice, and the
    /// extension is what an `AssetKind` is written from in the first place.
    fn material_root(&mut self, id: Uuid) -> Option<Uuid> {
        let path = self.resolve_path(id)?;
        (path.extension().and_then(|s| s.to_str()) == Some("inf_mat")).then_some(id)
    }

    /// The authored scatter geometry for one mesh GUID, loaded once and cached.
    ///
    /// # Why the vgeom store next door cannot answer this
    ///
    /// A `.inf_vmesh` is a meshlet DAG: `meshlet_vertices` + `meshlet_triangles`
    /// and no plain index list, and the cook does not derive one for a small prop
    /// in any case (`[vgeom] min_triangles`). The scatter raster pulls a flat
    /// vertex array and a flat index array out of two storage buffers, so it
    /// wants the `.inf_mesh` itself. This is the editor's twin of
    /// `inf_player::scatter_mesh` and it goes through the same one door,
    /// `ScatterGeometry::from_streams`.
    ///
    /// The triangle ceiling is the player's, by name, so the editor and a shipped
    /// build refuse the same content: a scatter kind draws its geometry once per
    /// instance with no LOD behind it.
    pub fn resolve_scatter_geometry(
        &mut self,
        mesh_id: Uuid,
    ) -> Option<Arc<inf_render::ScatterGeometry>> {
        if let Some(hit) = self.scatter.get(&mesh_id) {
            return hit.clone();
        }
        let built = self.build_scatter_geometry(mesh_id);
        self.scatter.insert(mesh_id, built.clone());
        built
    }

    fn build_scatter_geometry(
        &mut self,
        mesh_id: Uuid,
    ) -> Option<Arc<inf_render::ScatterGeometry>> {
        let mesh: inf_mesh::MeshAsset = self.load_payload(mesh_id)?;
        let (positions, normals, _uvs, _tangents, indices) = mesh.vgeom_streams();
        let geom = inf_render::ScatterGeometry::from_streams(&positions, &normals, &indices);
        if geom.is_empty() {
            tracing::warn!("inf-editor-core: scatter mesh {mesh_id} has no drawable triangles");
            return None;
        }
        if geom.triangle_count() > inf_render::MAX_SCATTER_MESH_TRIANGLES {
            tracing::warn!(
                "inf-editor-core: scatter mesh {mesh_id} is {} triangles, past the \
                 {} the scatter path draws per instance; it falls back to the \
                 placeholder primitive",
                geom.triangle_count(),
                inf_render::MAX_SCATTER_MESH_TRIANGLES
            );
            return None;
        }
        Some(Arc::new(geom))
    }

    fn load_payload<T: inf_asset::AssetPayload>(&mut self, id: Uuid) -> Option<T> {
        let path = self.resolve_path(id)?;
        let bytes = std::fs::read(&path).ok()?;
        match inf_asset::decode::<T>(&bytes) {
            Ok(v) => Some(v),
            Err(e) => {
                tracing::warn!("inf-editor-core: decode {}: {e}", path.display());
                None
            }
        }
    }

    /// The loose file for an asset GUID, rescanning **once** on a miss.
    ///
    /// The index is a snapshot taken when the root was set, so an asset written
    /// after it — a fresh import — would otherwise never be found. A GUID that is
    /// genuinely absent is remembered in `rescanned_for`, so a dangling reference
    /// costs one directory walk for the session rather than one per frame.
    fn resolve_path(&mut self, id: Uuid) -> Option<PathBuf> {
        if let Some(p) = self.index.get(&id) {
            return Some(p.clone());
        }
        let root = self.content_root.clone()?;
        if !self.rescanned_for.insert(id) {
            return None;
        }
        self.index = content_paths_by_guid(&root);
        self.index.get(&id).cloned()
    }
}

/// **The viewport's virtual-texture cache key** (P26.5) — see
/// [`EditorRenderAssets::vt_level_key`].
///
/// `BTreeSet` and not `HashSet`: the set is compared for equality here, but it
/// is also what `material_content` is asked for, and `registration_order`'s
/// purity in the registration SEQUENCE is the property both hosts' residency
/// rests on (the P26.3b audit's `HashMap`-walk finding). A key that iterates
/// differently on two runs would put that back.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct VtLevelKey {
    /// Every `Material.asset` the document binds, deduplicated and sorted.
    pub bindings: BTreeSet<Uuid>,
    /// `EditorRenderAssets::index_generation` when the key was taken.
    pub index_generation: u64,
}

impl VtLevelKey {
    /// Whether this level has any binding at all — `false` is the textureless
    /// path, where the host hands the renderer `None` and the command stream is
    /// the one all 50 goldens recorded.
    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }
}

/// Build bind-space [`SkinnedMeshData`] from a `.inf_mesh` that carries skinning
/// influences, or `None` if none of its submeshes do.
///
/// Submeshes are concatenated into one vertex + index buffer (indices rebased),
/// exactly like the thumbnailer's `combined_geometry` — the skinned pass draws one
/// instance per entity, and per-submesh material slots are the same follow-up
/// they are on the rigid path. An **unskinned** submesh inside a skinned mesh is
/// kept and pinned to joint 0 with weight 1, which is what a rigid part welded to
/// a skeleton's root means; dropping it would silently lose geometry.
///
/// **The uv crosses with the position and the normal** (P26.5). It is the reason
/// the box projection could be retired: a skinned character is the case where
/// `vt_box_uv` was visibly wrong, and the authored uv has been sitting in
/// `inf_mesh::MeshVertex` since P4 — this copy simply did not read it. An
/// unwrapped submesh contributes its `[0, 0]` default, which is what it has on
/// disk and not a value invented here.
///
/// **MIRROR** of the other host's `skinned_mesh_data` — keep the two
/// byte-identical, **this doc block included** (`projector_mirror.rs`). It has to
/// be: the two hosts would otherwise upload *different vertex buffers* for the
/// same asset, which no scene-level comparison can see. Side-neutral wording on
/// purpose — a note that names the *other* file is a note only one copy can carry,
/// and a comment that exists on one side only is the drift this gate is for.
pub fn skinned_mesh_data(mesh: &MeshAsset) -> Option<SkinnedMeshData> {
    if !mesh.submeshes.iter().any(|s| s.is_skinned()) {
        return None;
    }
    let mut vertices: Vec<SkinnedVertex> = Vec::with_capacity(mesh.vertex_count());
    let mut indices: Vec<u32> = Vec::new();
    for sm in &mesh.submeshes {
        let base = vertices.len() as u32;
        for (i, v) in sm.vertices.iter().enumerate() {
            let skin = sm.skin.get(i).copied().unwrap_or_default().normalized();
            vertices.push(SkinnedVertex {
                pos: v.position,
                normal: v.normal,
                uv: v.uv,
                joints: skin.joints.map(u32::from),
                weights: skin.weights,
            });
        }
        indices.extend(sm.indices.iter().map(|&i| i + base));
    }
    if vertices.is_empty() || indices.len() < 3 {
        return None;
    }
    Some(SkinnedMeshData { vertices, indices })
}

/// Index every loose render asset under `dir` by its GUID, read from the sibling
/// `inf_asset` `.toml` sidecar.
///
/// Recursive over a whole project **content root** (imports land in subfolders),
/// deterministic (each directory's entries are path-sorted before descending), and
/// the editor's own dot-directories (`.inf` import cache / thumbnails,
/// `.infinity` settings) are not walked — the same rules, and the same reasons, as
/// `terrain_stream::terrain_paths_by_guid`.
pub fn content_paths_by_guid(dir: &Path) -> HashMap<Uuid, PathBuf> {
    let mut files = Vec::new();
    collect_files(dir, 0, &mut files);
    let mut out = HashMap::new();
    for p in files {
        if let Ok(side) = inf_asset::AssetSidecar::load(&p) {
            out.insert(side.guid.uuid(), p);
        }
    }
    out
}

fn collect_files(dir: &Path, depth: u32, out: &mut Vec<PathBuf>) {
    if depth > MAX_CONTENT_DEPTH {
        return;
    }
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path())).collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            let hidden = path
                .file_name()
                .and_then(|s| s.to_str())
                .is_some_and(|n| n.starts_with('.'));
            if !hidden {
                collect_files(&path, depth + 1, out);
            }
        } else if path
            .extension()
            .and_then(|s| s.to_str())
            .is_some_and(|e| INDEXED_EXTENSIONS.contains(&e))
        {
            out.push(path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::{vmesh, AssetProject};
    use inf_anim::{AnimClip, Joint, JointTrack, JointTransform, QuatTrack};
    use inf_asset::AssetId;
    use inf_mesh::{MeshVertex, SubMesh, VertexSkin};

    /// A grid dense enough for a real multi-meshlet DAG, cheap enough for CI.
    fn grid_mesh(n: u32) -> MeshAsset {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        for j in 0..=n {
            for i in 0..=n {
                vertices.push(MeshVertex {
                    position: [i as f32, ((i * j) % 5) as f32 * 0.25, j as f32],
                    normal: [0.0, 1.0, 0.0],
                    uv: [i as f32 / n as f32, j as f32 / n as f32],
                    ..Default::default()
                });
            }
        }
        let idx = |i: u32, j: u32| j * (n + 1) + i;
        for j in 0..n {
            for i in 0..n {
                indices.extend_from_slice(&[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)]);
                indices.extend_from_slice(&[idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)]);
            }
        }
        MeshAsset::new(
            vec![SubMesh {
                name: "grid".into(),
                vertices,
                indices,
                material_slot: None,
                skin: Vec::new(),
            }],
            vec![],
        )
    }

    /// A 3-vertex triangle skinned to a 2-joint chain.
    fn skinned_mesh() -> MeshAsset {
        let v = |x: f32, y: f32| MeshVertex {
            position: [x, y, 0.0],
            normal: [0.0, 0.0, 1.0],
            // The AUTHORED parametrization, distinct per vertex and distinct
            // from the position, so an arm can tell "the uv crossed" from "some
            // number crossed" (P26.5 audit).
            uv: [x * 0.25 + 0.5, y * 0.125 + 0.25],
            ..Default::default()
        };
        MeshAsset::new(
            vec![SubMesh {
                name: "tri".into(),
                vertices: vec![v(0.0, 0.0), v(1.0, 0.0), v(0.0, 2.0)],
                indices: vec![0, 1, 2],
                material_slot: None,
                skin: vec![
                    VertexSkin {
                        joints: [0, 0, 0, 0],
                        weights: [1.0, 0.0, 0.0, 0.0],
                    },
                    VertexSkin {
                        joints: [0, 0, 0, 0],
                        weights: [1.0, 0.0, 0.0, 0.0],
                    },
                    VertexSkin {
                        joints: [1, 0, 0, 0],
                        weights: [1.0, 0.0, 0.0, 0.0],
                    },
                ],
            }],
            vec![],
        )
    }

    fn two_joint_skeleton() -> Skeleton {
        Skeleton::new(vec![
            Joint {
                name: "root".into(),
                parent: None,
                local_bind: JointTransform::default(),
                inverse_bind: Mat4::IDENTITY.to_cols_array(),
            },
            Joint {
                name: "tip".into(),
                parent: Some(0),
                local_bind: JointTransform::from_trs(
                    glam::Vec3::new(0.0, 1.0, 0.0),
                    glam::Quat::IDENTITY,
                    glam::Vec3::ONE,
                ),
                inverse_bind: Mat4::from_translation(glam::Vec3::new(0.0, -1.0, 0.0))
                    .to_cols_array(),
            },
        ])
        .unwrap()
    }

    /// A one-second clip rotating the tip joint 90° about X.
    fn wave_clip() -> AnimClip {
        // `AnimClip::new` since `.inf_anim` v2 (P29.2): `duration` is derived
        // from the keys (1.0 here, unchanged).
        AnimClip::new(
            "wave",
            vec![JointTrack {
                joint: 1,
                translation: None,
                rotation: Some(QuatTrack::new(
                    vec![0.0, 1.0],
                    vec![
                        glam::Quat::IDENTITY.to_array(),
                        glam::Quat::from_rotation_x(std::f32::consts::FRAC_PI_2).to_array(),
                    ],
                    inf_anim::Interpolation::Linear,
                )),
                scale: None,
            }],
        )
    }

    /// A project with one derived mesh; returns `(dir, root, mesh guid)`.
    fn project_with_derived_mesh(n: u32) -> (tempfile::TempDir, PathBuf, Uuid) {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_path_buf();
        let mut proj = AssetProject::open(&root).unwrap();
        let d = proj.content_dir("Meshes").unwrap();
        let mesh_id = proj
            .write_asset(&d, "Grid", &grid_mesh(n), None, vec![], None)
            .unwrap();
        assert!(vmesh::ensure_vmesh(&mut proj, mesh_id).unwrap().rebuilt());
        (dir, root, mesh_id.uuid())
    }

    /// **A character's own uv reaches the bind-space stream** (P26.5 audit).
    ///
    /// P26.5's headline for the skinned path is that a character samples the
    /// artist's parametrization rather than a box projection — *"a box projection
    /// on a character is visibly wrong"*, carried in three ledgers since P26.3.
    /// What pinned it was `projector_mirror`'s
    /// `the_bind_space_rebuild_is_identical_in_both_stores`, which compares this
    /// function's text with the player's copy of it. Measured: dropping
    /// `uv: v.uv` in **both** copies at once is invisible to the whole of
    /// `inf-editor-core` and `inf-player` — the mirror is satisfied, because the
    /// two hosts agree perfectly about building the same wrong buffer. Two hosts
    /// agreeing is not the world being right (the P24 law).
    ///
    /// So this asserts the world: the uv on the GPU vertex is the uv on the
    /// asset vertex, and it VARIES, because `[0, 0]` everywhere is a uv that
    /// satisfies "equals a function of the vertex" for a function that ignores
    /// its argument — the same trap `every_primitive_spans_its_own_uv_square`
    /// and the deformed-garment arm are shaped around.
    #[test]
    fn a_skinned_meshs_authored_uv_reaches_the_bind_space_stream() {
        let mesh = skinned_mesh();
        let data = skinned_mesh_data(&mesh).expect("the fixture is skinned");
        let authored: Vec<[f32; 2]> = mesh
            .submeshes
            .iter()
            .flat_map(|s| s.vertices.iter().map(|v| v.uv))
            .collect();
        assert_eq!(data.vertices.len(), authored.len());
        for (i, (got, want)) in data.vertices.iter().zip(&authored).enumerate() {
            assert_eq!(
                got.uv, *want,
                "vertex {i}: the bind-space stream carries {:?} and the asset says \
                 {want:?} — the upload dropped the authored parametrization, which \
                 is what put a box projection on a character through P26.4",
                got.uv
            );
        }
        // ANTI-VACUITY: the asset's own uvs are not all one value, so the
        // equality above is a measurement rather than a comparison of two
        // constants.
        let span = |k: usize| {
            let vals: Vec<f32> = authored.iter().map(|u| u[k]).collect();
            vals.iter().cloned().fold(f32::MIN, f32::max)
                - vals.iter().cloned().fold(f32::MAX, f32::min)
        };
        assert!(
            span(0) > 0.0 && span(1) > 0.0,
            "the fixture's authored uv is flat, so a stream of zeros would pass"
        );
    }

    /// **The headline gate**: an imported mesh, derived and indexed, resolves to
    /// real streamable geometry — which is precisely what the editor viewport could
    /// not do before P18.3.
    #[test]
    fn an_imported_mesh_resolves_to_real_geometry() {
        let (_dir, root, mesh_id) = project_with_derived_mesh(12);
        let mut store = EditorRenderAssets::new();
        store.set_content_root(Some(root));

        let loaded = store.resolve_vgeom(mesh_id).expect("real geometry");
        assert!(loaded.source.meshlet_count() > 0, "the DAG has meshlets");
        assert!(!loaded.source.pages().is_empty(), "and a page directory");
        assert!(
            loaded.source.bounds().1 > 0.0,
            "with a real bounding sphere"
        );
        assert_eq!(store.loaded_vgeom(), 1);
    }

    /// **The drop gap, end to end** (Wave E).
    ///
    /// From P4 to this wave, dragging a mesh asset into the viewport spawned a
    /// placeholder cube named after the asset and **nothing in the editor ever
    /// wrote `MeshRef::asset`** — only samples, migration fixtures and tests
    /// did. The gap was recorded as a rendering limitation ("the viewport thread
    /// has no asset DB"), and P18.3 — the test directly above — removed that
    /// reason without anyone noticing the write was still missing.
    ///
    /// This arm walks the whole path in one place, because each half passes on
    /// its own while the feature stays broken: the payload parses, the spawn
    /// binds, the component holds the guid, and the store resolves that guid to
    /// real streamable geometry.
    ///
    /// **What it does and does not reach** (corrected by the Wave E audit, A6).
    /// The first write-up of this arm claimed "delete the `mesh_asset()` branch
    /// in `spawn_drop` and the second assertion fails" — measured, that mutation
    /// left this test GREEN, because `spawn_drop` lives in `inf-viewport`, which
    /// this crate does not depend on and cannot call. Step 2 now goes through
    /// `viewport_drop::spawn_asset_entity`, the ONE door both real drop paths
    /// (`EngineHost::spawn_drop` and `scene_spawn_asset`) were refactored onto,
    /// so deleting the binding there DOES turn this red. That each of those two
    /// callers still reaches the door is a source-level fact, pinned by
    /// `inf-viewport`'s `drop_gate` and `inf-studio`'s `the_spawn_command_uses_the_shared_drop_door`.
    #[test]
    fn a_dropped_mesh_is_bound_to_the_entity_and_resolves_to_real_geometry() {
        use crate::scene::SceneDoc;
        use crate::viewport_drop::{parse_drop_payload, spawn_asset_entity};

        let (_dir, root, mesh_id) = project_with_derived_mesh(12);

        // 1. What the Content Drawer sends when a mesh is dropped on the hole.
        let payload = format!("asset:mesh:{mesh_id}:Barrel");
        let parsed = parse_drop_payload(&payload);
        let bound = parsed.mesh_asset().expect("a mesh payload binds");
        assert_eq!(bound, mesh_id);

        // 2. What both drop doors do with it — the entity carries the asset.
        let mut doc = SceneDoc::new();
        let guid = spawn_asset_entity(&mut doc, "Barrel", Some(bound), None);
        assert_eq!(
            doc.mesh_asset_of(guid),
            Some(mesh_id),
            "the spawned prop must carry the mesh it was dropped from"
        );

        // 3. What the renderer then finds: real geometry, not a placeholder.
        let mut store = EditorRenderAssets::new();
        store.set_content_root(Some(root));
        let loaded = store
            .resolve_vgeom(doc.mesh_asset_of(guid).unwrap())
            .expect("the bound asset resolves to real geometry");
        assert!(loaded.source.meshlet_count() > 0);

        // 4. And undo removes the prop rather than leaving a bound orphan.
        assert!(doc.undo());
        assert_eq!(doc.mesh_asset_of(guid), None);
    }

    /// Resolution is a pure function of content: two stores over the same root
    /// agree on the id and on everything the renderer reads from the source.
    #[test]
    fn resolution_is_deterministic_across_stores() {
        let (_dir, root, mesh_id) = project_with_derived_mesh(10);
        let mut a = EditorRenderAssets::new();
        let mut b = EditorRenderAssets::new();
        a.set_content_root(Some(root.clone()));
        b.set_content_root(Some(root));
        let (x, y) = (
            a.resolve_vgeom(mesh_id).unwrap(),
            b.resolve_vgeom(mesh_id).unwrap(),
        );
        assert_eq!(x.id, y.id, "the content key is reproducible");
        assert_eq!(x.source.meshlet_count(), y.source.meshlet_count());
        assert_eq!(x.source.pages(), y.source.pages());
    }

    /// An unresolvable mesh (no DAG on disk) is a clean `None` — the caller keeps
    /// its primitive placeholder — and it costs exactly **one** directory walk,
    /// not one per frame.
    #[test]
    fn a_dangling_mesh_reference_misses_cheaply() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = EditorRenderAssets::new();
        store.set_content_root(Some(dir.path().to_path_buf()));
        let ghost = Uuid::from_u128(0xDEAD_BEEF);
        for _ in 0..5 {
            assert!(store.resolve_vgeom(ghost).is_none());
        }
        assert_eq!(store.loaded_vgeom(), 0);
    }

    /// With no content root the store is inert — which is what makes a viewport
    /// with no project open behave exactly as it did before this batch.
    #[test]
    fn no_content_root_means_no_resolution() {
        let mut store = EditorRenderAssets::new();
        assert!(store.resolve_vgeom(Uuid::from_u128(1)).is_none());
        assert_eq!(store.index_len(), 0);
    }

    /// **Re-import invalidation.** Rewriting the mesh under the same GUID must
    /// change the render key, because both renderer nodes cache GPU state by it —
    /// a stable key across changed bytes is a permanently stale draw.
    #[test]
    fn a_reimport_changes_the_content_key() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_path_buf();
        let mut proj = AssetProject::open(&root).unwrap();
        let d = proj.content_dir("Meshes").unwrap();
        let mesh_id = proj
            .write_asset(&d, "Grid", &grid_mesh(8), None, vec![], None)
            .unwrap();
        vmesh::ensure_vmesh(&mut proj, mesh_id).unwrap();

        let mut store = EditorRenderAssets::new();
        store.set_content_root(Some(root));
        let before = store.resolve_vgeom(mesh_id.uuid()).unwrap();

        // Re-import (same GUID, new geometry) + re-derive.
        proj.rewrite_payload(mesh_id, &grid_mesh(14), vec![])
            .unwrap();
        assert!(vmesh::ensure_vmesh(&mut proj, mesh_id).unwrap().rebuilt());

        // Without the refresh the store still serves the old mapping — which is
        // exactly why Ring 2 pushes `refresh_index` on `assets://changed`.
        assert_eq!(store.resolve_vgeom(mesh_id.uuid()).unwrap().id, before.id);
        store.refresh_index();
        let after = store.resolve_vgeom(mesh_id.uuid()).unwrap();
        assert_ne!(after.id, before.id, "the render key must follow the bytes");
        assert_ne!(after.source.meshlet_count(), before.source.meshlet_count());
    }

    /// **A DAG derived from bytes the mesh no longer holds is NOT drawn** (wave
    /// FIX2, carried 38).
    ///
    /// The re-import arm above re-derives before it looks; this one does not,
    /// which is the state every session is in between an import landing and the
    /// meshlet sweep reaching that mesh — about two and a half minutes on the
    /// island, and the reason the ROAD1b audit's proof frame showed the previous
    /// build's roads. `open_vgeom` used to read the payload and ask nothing, so
    /// the viewport drew it with complete confidence.
    ///
    /// Non-vacuous by construction: the same store resolves the mesh happily one
    /// line earlier, so this is the staleness being refused and not the fixture
    /// failing to resolve at all.
    #[test]
    fn a_stale_derived_vmesh_is_refused_rather_than_drawn() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_path_buf();
        let mut proj = AssetProject::open(&root).unwrap();
        let d = proj.content_dir("Meshes").unwrap();
        let mesh_id = proj
            .write_asset(&d, "Grid", &grid_mesh(8), None, vec![], None)
            .unwrap();
        vmesh::ensure_vmesh(&mut proj, mesh_id).unwrap();

        let mut store = EditorRenderAssets::new();
        store.set_content_root(Some(root));
        assert!(
            store.resolve_vgeom(mesh_id.uuid()).is_some(),
            "the fixture does not resolve even when current"
        );

        // The mesh moves; nothing re-derives. Same GUID, same files, new bytes.
        proj.rewrite_payload(mesh_id, &grid_mesh(14), vec![])
            .unwrap();
        store.refresh_index();
        assert!(
            store.resolve_vgeom(mesh_id.uuid()).is_none(),
            "the viewport drew a DAG built from bytes this mesh no longer holds"
        );

        // …and the sweep returns it, so a session is not stuck in the refusal.
        assert!(vmesh::ensure_vmesh(&mut proj, mesh_id).unwrap().rebuilt());
        store.refresh_index();
        assert!(store.resolve_vgeom(mesh_id.uuid()).is_some());
    }

    /// Deleting the mesh asset stops it resolving — no stale geometry, no panic,
    /// and since wave FIX2 no placeholder box either: a bound `MeshRef` whose DAG
    /// is gone draws nothing.
    #[test]
    fn deleting_a_mesh_stops_it_resolving() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_path_buf();
        let mut proj = AssetProject::open(&root).unwrap();
        let d = proj.content_dir("Meshes").unwrap();
        let mesh_id = proj
            .write_asset(&d, "Grid", &grid_mesh(8), None, vec![], None)
            .unwrap();
        vmesh::ensure_vmesh(&mut proj, mesh_id).unwrap();
        let mut store = EditorRenderAssets::new();
        store.set_content_root(Some(root));
        assert!(store.resolve_vgeom(mesh_id.uuid()).is_some());

        assert!(proj.delete(mesh_id, false).unwrap().is_empty());
        store.refresh_index();
        assert!(
            store.resolve_vgeom(mesh_id.uuid()).is_none(),
            "a deleted mesh must stop resolving — no stale render"
        );
    }

    /// **Level switch releases streams** (the P16.4b lesson, tested rather than
    /// asserted): the per-projection audit drops what the new document does not
    /// reference, and `clear` drops everything.
    #[test]
    fn a_level_switch_releases_every_opened_payload() {
        let (_dir, root, mesh_id) = project_with_derived_mesh(9);
        let mut store = EditorRenderAssets::new();
        store.set_content_root(Some(root));
        store.resolve_vgeom(mesh_id).unwrap();
        assert_eq!(store.loaded_vgeom(), 1);

        // A projection that no longer references it releases it…
        store.retain_only(std::iter::empty());
        assert_eq!(store.loaded_vgeom(), 0, "retain_only released the mapping");

        // …and a projection that does keeps it.
        store.resolve_vgeom(mesh_id).unwrap();
        store.retain_only([mesh_id]);
        assert_eq!(store.loaded_vgeom(), 1);

        // The document-replaced door drops everything, index intact.
        store.clear();
        assert_eq!(store.loaded_vgeom(), 0);
        assert!(store.index_len() > 0, "the index survives a level switch");
    }

    // ── skeletal ────────────────────────────────────────────────────────────

    fn project_with_character() -> (tempfile::TempDir, PathBuf, Uuid, Uuid, Uuid) {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_path_buf();
        let mut proj = AssetProject::open(&root).unwrap();
        let d = proj.content_dir("Character").unwrap();
        let mesh = proj
            .write_asset(&d, "Body", &skinned_mesh(), None, vec![], None)
            .unwrap();
        let skel = proj
            .write_asset(
                &d,
                "Rig",
                &inf_anim::SkeletonAsset::new(two_joint_skeleton()),
                None,
                vec![],
                None,
            )
            .unwrap();
        let clip = proj
            .write_asset(
                &d,
                "Wave",
                &inf_anim::AnimClipAsset::new(wave_clip(), Some(*skel.uuid().as_bytes())),
                None,
                vec![],
                None,
            )
            .unwrap();
        (dir, root, mesh.uuid(), skel.uuid(), clip.uuid())
    }

    /// **SkeletalMesh rest-pose projection**: a character with no `AnimPlayer`
    /// resolves to real bind-space geometry and an identity-ish palette, so it is
    /// visible the moment it is placed.
    #[test]
    fn a_skeletal_mesh_projects_its_rest_pose() {
        let (_dir, root, mesh, skel, _clip) = project_with_character();
        let mut store = EditorRenderAssets::new();
        store.set_content_root(Some(root));

        let sm = SkeletalMesh {
            mesh: Some(mesh),
            skeleton: Some(skel),
        };
        let draw = store
            .resolve_skinned(&sm, None, None, None)
            .expect("rest pose draws");
        assert_eq!(draw.mesh.vertices.len(), 3);
        assert_eq!(draw.mesh.indices, vec![0, 1, 2]);
        assert_eq!(draw.palette.len(), 2, "one matrix per joint");
        // Rest pose: `global · inverse_bind` is the identity for a bind-pose
        // skeleton, which is what makes the drawn mesh its authored shape.
        for m in draw.palette.iter() {
            assert!(
                (*m - Mat4::IDENTITY)
                    .to_cols_array()
                    .iter()
                    .all(|v| v.abs() < 1e-5),
                "rest palette must be identity, got {m:?}"
            );
        }
        // The influences survived the bind-space rebuild.
        assert_eq!(draw.mesh.vertices[2].joints[0], 1);
        assert_eq!(draw.key, (mesh, skel));
    }

    /// **The shared tier palette is the one the per-agent path would have built**
    /// (wave NPC1b) — the whole claim behind `resolve_skinned_shared`, as an
    /// equality rather than as a sentence.
    ///
    /// A crowd agent the ladder took off the pose path has no `EvaluatedPose`
    /// (`step_pose_evaluation` rebuilds its store from the tier's own target set)
    /// and no `AnimPlayer`, so the per-agent call falls through to rule 3. If the
    /// two ever disagreed, a crowd would visibly change pose at the tier boundary
    /// — and it would do so in whichever host called which door.
    #[test]
    fn the_shared_palette_is_the_one_the_per_agent_path_would_have_built() {
        let (_dir, root, mesh, skel, _clip) = project_with_character();
        let mut store = EditorRenderAssets::new();
        store.set_content_root(Some(root));
        let sm = SkeletalMesh {
            mesh: Some(mesh),
            skeleton: Some(skel),
        };
        let per_agent = store.resolve_skinned(&sm, None, None, None).unwrap();
        let shared = store.resolve_skinned_shared(&sm, None).unwrap();
        assert_eq!(*shared.palette, *per_agent.palette);
        assert_eq!(shared.key, per_agent.key);
        assert!(Arc::ptr_eq(&shared.mesh, &per_agent.mesh));

        // …and it is SHARED: two agents get one allocation, which is the thing
        // the atlas deduplicates on. An equality that held while every call
        // allocated a fresh `Vec` would satisfy the arm above and buy nothing.
        let again = store.resolve_skinned_shared(&sm, None).unwrap();
        assert!(
            Arc::ptr_eq(&shared.palette, &again.palette),
            "the shared palette was rebuilt — the cache is not a cache"
        );
        assert!(
            !Arc::ptr_eq(&shared.palette, &per_agent.palette),
            "the per-agent path handed back the shared block, so this arm proves nothing"
        );
    }

    /// **THE SHARED REST PALETTE IS EVICTED BY THE DOORS THAT EVICT THE SKELETON
    /// IT CAME FROM** (NPC1b audit).
    ///
    /// Wave NPC1b added `rest_palettes` and wired it into **none** of this
    /// store's three eviction doors — `drop_loaded` (which `clear`,
    /// `set_content_root` and `refresh_index` all reach), and `retain_only`. That
    /// is two defects in one omission:
    ///
    /// * **stale.** `refresh_index`'s own doc says it drops what it holds because
    ///   *"a payload rewritten under the same GUID would otherwise be served from
    ///   the old mapping forever"*. The rest palette is `skinning_matrices` over a
    ///   **skeleton** that line drops, so after a re-import the per-agent path
    ///   would read the new rig and the crowd's shared path the old one — the
    ///   exact disagreement the arm above asserts cannot happen, arriving from the
    ///   one direction that arm cannot see.
    /// * **a leak.** `retain_only` is the only place that knows which assets are
    ///   live, so a map missing from it is a map a session can only grow (P21's
    ///   "a pin with no release is a leak with a deadline").
    ///
    /// The arm holds both doors *and* that it is still a cache in between, so a
    /// "fix" that stopped caching altogether fails it too.
    #[test]
    fn the_shared_rest_palette_is_dropped_with_the_skeleton_it_was_derived_from() {
        let (_dir, root, mesh, skel, _clip) = project_with_character();
        let mut store = EditorRenderAssets::new();
        store.set_content_root(Some(root));
        let sm = SkeletalMesh {
            mesh: Some(mesh),
            skeleton: Some(skel),
        };

        let first = store.resolve_skinned_shared(&sm, None).unwrap();
        // …and it really is a cache, so the two negatives below mean something.
        assert!(
            Arc::ptr_eq(
                &first.palette,
                &store.resolve_skinned_shared(&sm, None).unwrap().palette
            ),
            "the rest palette is not cached at all, so nothing below is a test of \
             eviction"
        );

        // Door 1: `clear` / `set_content_root` / `refresh_index`, all through
        // `drop_loaded`. The old `Arc` is still held here, so its allocation
        // cannot be recycled under the new one — the pointer inequality is real.
        store.clear();
        let after_clear = store.resolve_skinned_shared(&sm, None).unwrap();
        assert!(
            !Arc::ptr_eq(&first.palette, &after_clear.palette),
            "`clear` kept the rest palette, so it outlives the skeleton bytes it \
             was derived from"
        );
        assert_eq!(*after_clear.palette, *first.palette, "the rig changed");

        // Door 2: `retain_only`, with the skeleton no longer referenced.
        assert!(
            Arc::ptr_eq(
                &after_clear.palette,
                &store.resolve_skinned_shared(&sm, None).unwrap().palette
            ),
            "the cache stopped caching after `clear`"
        );
        store.retain_only([mesh]);
        assert!(
            !Arc::ptr_eq(
                &after_clear.palette,
                &store.resolve_skinned_shared(&sm, None).unwrap().palette
            ),
            "`retain_only` kept a rest palette whose skeleton left the document"
        );
        // …and the mesh half of the key counts too: an entry survives only while
        // BOTH halves are live.
        let held = store.resolve_skinned_shared(&sm, None).unwrap();
        store.retain_only([skel]);
        assert!(
            !Arc::ptr_eq(
                &held.palette,
                &store.resolve_skinned_shared(&sm, None).unwrap().palette
            ),
            "`retain_only` keyed on the skeleton alone, so a deleted mesh's entry \
             lives for ever"
        );
    }

    /// With an `AnimPlayer` the palette follows the play-head — the same clip
    /// sampling the runtime does, so the viewport shows the pose the game would.
    #[test]
    fn an_anim_player_drives_the_palette() {
        let (_dir, root, mesh, skel, clip) = project_with_character();
        let mut store = EditorRenderAssets::new();
        store.set_content_root(Some(root));
        let sm = SkeletalMesh {
            mesh: Some(mesh),
            skeleton: Some(skel),
        };

        let rest = store
            .resolve_skinned(&sm, None, None, None)
            .unwrap()
            .palette;
        let posed = store
            .resolve_skinned(
                &sm,
                Some(&AnimPlayer {
                    clip: Some(clip),
                    // Mid-clip on purpose: `t == duration` on a LOOPING player
                    // wraps back to 0, which would compare the rest pose to
                    // itself and pass for the wrong reason.
                    t: 0.5,
                    ..AnimPlayer::default()
                }),
                None,
                None,
            )
            .unwrap()
            .palette;
        assert_ne!(rest[1].to_cols_array(), posed[1].to_cols_array());
        // …and it is deterministic: the same play-head yields the same palette.
        let again = store
            .resolve_skinned(
                &sm,
                Some(&AnimPlayer {
                    clip: Some(clip),
                    // Mid-clip on purpose: `t == duration` on a LOOPING player
                    // wraps back to 0, which would compare the rest pose to
                    // itself and pass for the wrong reason.
                    t: 0.5,
                    ..AnimPlayer::default()
                }),
                None,
                None,
            )
            .unwrap()
            .palette;
        assert_eq!(posed[1].to_cols_array(), again[1].to_cols_array());
    }

    /// An `AnimPlayer` whose clip cannot be resolved falls back to the rest pose
    /// rather than to nothing — a missing clip must not make a character vanish.
    #[test]
    fn an_unresolvable_clip_falls_back_to_rest() {
        let (_dir, root, mesh, skel, _clip) = project_with_character();
        let mut store = EditorRenderAssets::new();
        store.set_content_root(Some(root));
        let sm = SkeletalMesh {
            mesh: Some(mesh),
            skeleton: Some(skel),
        };
        let rest = store
            .resolve_skinned(&sm, None, None, None)
            .unwrap()
            .palette;
        let ghost = store
            .resolve_skinned(
                &sm,
                Some(&AnimPlayer {
                    clip: Some(Uuid::from_u128(0xC0FFEE)),
                    t: 0.5,
                    ..AnimPlayer::default()
                }),
                None,
                None,
            )
            .unwrap()
            .palette;
        assert_eq!(rest[1].to_cols_array(), ghost[1].to_cols_array());
    }

    /// Half-bound skeletal entities keep the placeholder (no skeleton ⇒ nothing to
    /// deform with), and an unskinned mesh is not a skinned draw.
    #[test]
    fn an_unbound_skeletal_mesh_stays_a_placeholder() {
        let (_dir, root, mesh, skel, _clip) = project_with_character();
        let mut store = EditorRenderAssets::new();
        store.set_content_root(Some(root));

        assert!(store
            .resolve_skinned(&SkeletalMesh::default(), None, None, None)
            .is_none());
        assert!(store
            .resolve_skinned(
                &SkeletalMesh {
                    mesh: Some(mesh),
                    skeleton: None
                },
                None,
                None,
                None,
            )
            .is_none());
        assert!(store
            .resolve_skinned(
                &SkeletalMesh {
                    mesh: Some(Uuid::from_u128(7)),
                    skeleton: Some(skel)
                },
                None,
                None,
                None,
            )
            .is_none());
        // A rigid mesh with no skin stream is not skinned geometry.
        assert!(skinned_mesh_data(&grid_mesh(4)).is_none());
    }

    /// The GUID index skips the editor's own dot-directories, so a `.inf_vmesh`
    /// parked in the import cache can never be served as content.
    #[test]
    fn the_index_skips_dot_directories() {
        let dir = tempfile::tempdir().unwrap();
        let hidden = dir.path().join(".inf").join("import-cache");
        std::fs::create_dir_all(&hidden).unwrap();
        let payload = hidden.join("Stray.inf_vmesh");
        std::fs::write(&payload, b"not a real payload").unwrap();
        let side = inf_asset::AssetSidecar::new(
            AssetId::new(),
            inf_asset::AssetKind::MeshletMesh,
            inf_asset::ContentHash::of(b"not a real payload"),
        );
        side.save(&payload).unwrap();

        assert!(content_paths_by_guid(dir.path()).is_empty());
    }
    /// An [`EvaluatedPose`] on `skeleton` with the tip joint bent 30° about X —
    /// what a state machine in a non-entry state publishes.
    fn bent_pose(skeleton: Uuid) -> EvaluatedPose {
        let sk = two_joint_skeleton();
        let mut pose = inf_anim::Pose::rest(&sk);
        pose.locals[1].rotation = glam::Quat::from_rotation_x(30f32.to_radians()).to_array();
        EvaluatedPose {
            skeleton,
            pose,
            sockets: Vec::new(),
        }
    }

    /// A pose with the WRONG joint count whose acceptance would be **visible**.
    ///
    /// The load-bearing detail is which joint is bent (P24.1 audit M4). The first
    /// cut bent the tip and then truncated the tip away, so the short pose
    /// evaluated byte-identical to rest — `global_transforms` back-fills a missing
    /// joint from its bind transform — and the assertion below passed with or
    /// without the length guard it exists to prove. Bending the **root**, which
    /// survives the truncation, makes an accepted short pose produce a different
    /// palette from rest, so deleting the guard fails the test.
    fn short_pose_a_guardless_store_would_wear(skeleton: Uuid) -> EvaluatedPose {
        let sk = two_joint_skeleton();
        let mut pose = inf_anim::Pose::rest(&sk);
        pose.locals[0].rotation = glam::Quat::from_rotation_x(40f32.to_radians()).to_array();
        pose.locals.truncate(1);
        EvaluatedPose {
            skeleton,
            pose,
            sockets: Vec::new(),
        }
    }

    /// **The P24.1 headline gate, store half**: when the sim published a pose for
    /// this entity, THAT is what the palette is built from — the `AnimPlayer` is
    /// overruled, because an `AnimStateMachine` beats a clip play-head and this is
    /// the only place that fact reaches the renderer.
    #[test]
    fn a_sim_evaluated_pose_beats_the_anim_player() {
        let (_dir, root, mesh, skel, clip) = project_with_character();
        let mut store = EditorRenderAssets::new();
        store.set_content_root(Some(root));
        let sm = SkeletalMesh {
            mesh: Some(mesh),
            skeleton: Some(skel),
        };
        let play = AnimPlayer {
            clip: Some(clip),
            t: 0.5,
            ..AnimPlayer::default()
        };

        let rest = store
            .resolve_skinned(&sm, None, None, None)
            .unwrap()
            .palette;
        let from_clip = store
            .resolve_skinned(&sm, Some(&play), None, None)
            .unwrap()
            .palette;
        let posed = bent_pose(skel);
        let from_sim = store
            .resolve_skinned(&sm, Some(&play), Some(&posed), None)
            .unwrap()
            .palette;

        // All three are distinct: the machine's pose is neither the rest pose nor
        // the clip's. (Comparing only against rest would pass for a store that
        // simply kept sampling the clip.)
        assert_ne!(from_sim[1].to_cols_array(), rest[1].to_cols_array());
        assert_ne!(from_sim[1].to_cols_array(), from_clip[1].to_cols_array());
        // …and with no player at all the sim pose still wins over rest.
        let alone = store
            .resolve_skinned(&sm, None, Some(&posed), None)
            .unwrap()
            .palette;
        assert_eq!(alone[1].to_cols_array(), from_sim[1].to_cols_array());
    }

    /// Rule 2's guards: a pose evaluated against a **different skeleton**, or one
    /// whose joint count does not match, is refused and the store falls through to
    /// the `AnimPlayer` / rest arms. Applying it would deform a character by
    /// another rig's hierarchy — silently, and only on the frames it happened.
    #[test]
    fn a_pose_from_another_skeleton_is_refused() {
        let (_dir, root, mesh, skel, _clip) = project_with_character();
        let mut store = EditorRenderAssets::new();
        store.set_content_root(Some(root));
        let sm = SkeletalMesh {
            mesh: Some(mesh),
            skeleton: Some(skel),
        };
        let rest = store
            .resolve_skinned(&sm, None, None, None)
            .unwrap()
            .palette;

        let mut wrong_rig = bent_pose(skel);
        wrong_rig.skeleton = Uuid::from_u128(0xBAD_5CE1);
        assert_eq!(
            store
                .resolve_skinned(&sm, None, Some(&wrong_rig), None)
                .unwrap()
                .palette[1]
                .to_cols_array(),
            rest[1].to_cols_array(),
            "a pose from another rig must not be worn"
        );

        let wrong_len = short_pose_a_guardless_store_would_wear(skel);
        let drawn = store
            .resolve_skinned(&sm, None, Some(&wrong_len), None)
            .unwrap()
            .palette;
        // Both joints, because a short pose that IS worn moves both: the bent root
        // is joint 0's palette outright and rides through joint 1's global.
        assert_eq!(
            drawn[0].to_cols_array(),
            rest[0].to_cols_array(),
            "a pose with the wrong joint count must not be worn"
        );
        assert_eq!(
            drawn[1].to_cols_array(),
            rest[1].to_cols_array(),
            "a pose with the wrong joint count must not be worn"
        );
        // ANTI-VACUITY: the refused pose really would have changed the palette —
        // otherwise the two assertions above hold whether the guard exists or not,
        // which is exactly the hole the audit found.
        let worn = inf_anim::skinning_matrices(&two_joint_skeleton(), &wrong_len.pose);
        assert_ne!(
            worn[0].to_cols_array(),
            rest[0].to_cols_array(),
            "the short pose evaluates to the rest palette, so refusing it is \
             indistinguishable from accepting it"
        );
    }

    // ── virtual textures (P26.4, clause 0) ──────────────────────────────────

    /// A real `.inf_tex` **v2 tiled container** of `n × n`, through the one
    /// writer — so the bytes the store hands the registry are bytes a runtime
    /// pages, not a plausible blob.
    fn tiled(n: u32, srgb: bool) -> Vec<u8> {
        inf_material::build_tiled_texture(
            vec![180u8; (n * n * 4) as usize],
            n,
            n,
            inf_material::TextureImportSettings {
                srgb,
                generate_mips: true,
                compression: inf_material::TextureCompression::None,
                hdr: false,
            },
        )
        .expect("the fixture tiles")
        .into_bytes()
    }

    /// **The editor resolves a level's bindings into virtual-texture content, in
    /// the ONE registration order** (P26.4, clause 0).
    ///
    /// The viewport half of the gap the P26.3b ledger named: before this, no
    /// projector called `VtTextures::register` and every instance shipped
    /// `vt: Default::default()`. What is asserted here is the *decision* — which
    /// `.inf_mat`s resolve, which `.inf_tex` bytes come with them, and in what
    /// sequence — because that sequence IS the residency, and it is the one
    /// thing the two hosts must agree on exactly.
    #[test]
    fn a_levels_bindings_resolve_to_registrable_material_content() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_path_buf();
        let mut proj = AssetProject::open(&root).unwrap();
        let d = proj.content_dir("Materials").unwrap();

        // The RAW-IMAGE door (P26.1): a v2 container is never written through
        // `inf_asset::encode`, so a fixture that used the generic writer would be
        // a fixture the reader refuses.
        let put_tex = |proj: &mut AssetProject, name: &str, bytes: Vec<u8>| {
            let path = proj.unique_asset_path(&d, name, "inf_tex").unwrap();
            std::fs::write(&path, &bytes).unwrap();
            proj.register_written_asset(
                path,
                inf_asset::AssetKind::Texture,
                inf_asset::ContentHash::of(&bytes),
                None,
                None,
                None,
            )
            .unwrap()
        };
        let albedo = put_tex(&mut proj, "Albedo", tiled(128, true));
        let orm = put_tex(&mut proj, "Orm", tiled(64, false));
        let mat = proj
            .write_asset(
                &d,
                "Wall",
                &inf_material::MaterialAsset {
                    base_color_texture: Some(albedo),
                    normal_texture: None,
                    metallic_roughness_texture: Some(orm),
                    ..Default::default()
                },
                None,
                vec![albedo, orm],
                None,
            )
            .unwrap();
        // A second material naming ONE of the same textures — the dedupe case,
        // and what makes "first sight wins" a claim that can fail.
        let mat2 = proj
            .write_asset(
                &d,
                "Trim",
                &inf_material::MaterialAsset {
                    base_color_texture: Some(orm),
                    ..Default::default()
                },
                None,
                vec![orm],
                None,
            )
            .unwrap();

        let mut store = EditorRenderAssets::new();
        store.set_content_root(Some(root));
        // Bindings arrive in DOCUMENT order — deliberately the reverse of GUID
        // order here, so the sort below is measuring the rule.
        let mut bound = [mat.uuid(), mat2.uuid()];
        bound.sort();
        bound.reverse();
        let content = store.material_content(bound);

        assert_eq!(content.materials.len(), 2, "a binding did not resolve");
        assert_eq!(
            content.textures.len(),
            2,
            "the shared texture was loaded twice or one was lost"
        );
        assert!(content.source(albedo.uuid().as_u128()).is_some());
        assert!(content.source(orm.uuid().as_u128()).is_some());
        assert!(content.source(0).is_none());

        // THE ORDER: materials by GUID, then albedo → normal → ORM, first sight
        // wins — the same `inf_render::registration_order` the player walks.
        let mut want: Vec<u128> = Vec::new();
        let (a, b) = if mat.uuid() < mat2.uuid() {
            (mat, mat2)
        } else {
            (mat2, mat)
        };
        for m in [a, b] {
            for t in if m == mat {
                vec![albedo, orm]
            } else {
                vec![orm]
            } {
                if !want.contains(&t.uuid().as_u128()) {
                    want.push(t.uuid().as_u128());
                }
            }
        }
        assert_eq!(inf_render::registration_order(&content.materials), want);

        // …and the content REGISTERS: the door takes it, the floor fits, and a
        // surface bound to the material names two of its three slots.
        let (mut lib, _) = inf_render::VtTextures::new(inf_render::VtPoolConfig {
            format: inf_render::PageFormat::Rgba8,
            stored_tile_size: inf_render::STORED_TILE_SIZE,
            budget_bytes: inf_render::DEFAULT_VT_BUDGET_BYTES,
            max_texture_dim: 8192,
            trilinear: false,
            // **Unthrottled** (IB-16): a per-frame upload budget would make a want
            // that was ASKED FOR and a want that was SEATED two different things in
            // every count here, which is not what this file is about.
            upload_budget_bytes: 0,
        });
        let n = lib.register_materials(&content.materials, |g| content.source(g));
        assert_eq!(n, 2);
        assert!(lib.refusals().is_empty(), "{:?}", lib.refusals());
        let floor = lib.want_floor();
        assert_eq!(lib.residency_mut().apply_wants(&floor).deferred, 0);
        let set = lib.set_for_material(mat.uuid().as_u128());
        assert!(!set.is_none());
        assert_eq!(set.normal, 0, "the empty slot must stay empty");
        assert_eq!(
            lib.handle(albedo.uuid().as_u128()).map(|h| h.0 + 1),
            Some(set.albedo)
        );
        assert_eq!(
            lib.handle(orm.uuid().as_u128()).map(|h| h.0 + 1),
            Some(set.orm)
        );

        // **THE INVALIDATION** (P26.4 audit). The viewport's VT level is gated on
        // the binding SET, and a re-imported `.inf_tex` changes neither that set
        // nor the document version — so before `index_generation` existed the
        // atlas held the bytes it read the first time for the rest of the
        // session, while `sync_vt_bindings`'s own doc claimed
        // `refresh_asset_index` "clears the set". Nothing cleared it; nothing in
        // the codebase ever had.
        //
        // Asserted on the BYTES, not on the counter: a monotone number that
        // tracks nothing is exactly as useless as no number.
        let before = store.index_generation();
        let first = store
            .material_content([mat.uuid()])
            .source(albedo.uuid().as_u128())
            .map(|s| s.payload().to_vec())
            .expect("the albedo resolved");
        let reimported = tiled(64, true);
        assert_ne!(first, reimported, "the fixture re-imported the same bytes");
        std::fs::write(proj.db().get(albedo).unwrap().path.clone(), &reimported).unwrap();
        store.refresh_index();
        assert!(
            store.index_generation() > before,
            "a re-import did not move the index generation, so a consumer caching \
             a derived GPU object has nothing to notice it by"
        );
        assert_eq!(
            store
                .material_content([mat.uuid()])
                .source(albedo.uuid().as_u128())
                .map(|s| s.payload().to_vec()),
            Some(reimported),
            "the store served the pre-import bytes"
        );
        // …and a root change is the same event.
        let gen_before_root = store.index_generation();
        store.set_content_root(store.content_root().map(Path::to_path_buf));
        assert!(store.index_generation() > gen_before_root);

        // ANTI-VACUITY: an unbound level resolves nothing, so the counts above
        // measure the bindings and not the content root.
        assert!(store.material_content([]).is_empty());
        // …and a binding naming something that is not a `.inf_mat` resolves to
        // nothing, which is the MIRROR of the shipped path (the cook derives a
        // record for `AssetKind::Material` only and warns about the rest).
        assert!(store.material_content([albedo.uuid()]).is_empty());
    }
}
