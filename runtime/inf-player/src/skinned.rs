//! **The shipped player's skeletal render-asset store** — the matching follow-up
//! P18.3 left open, and the close of a live PIE-vs-shipping divergence.
//!
//! P18.3 gave the *editor viewport* a real `SkeletalMesh` projection
//! ([`inf_editor_core::render_assets`]): a bound skeletal mesh draws its actual
//! skinned geometry, posed. The shipped player had **no `SkeletalMesh` branch at
//! all** — `crate::render::project_scene` never touched
//! [`RenderScene::skinned_meshes`](inf_render::RenderScene::skinned_meshes) — so a
//! level with a skeletal character previewed correctly in PIE and shipped as
//! nothing. This module is the player's half; [`crate::render`] is the projection
//! that consumes it.
//!
//! It is to [`inf_editor_core::render_assets::EditorRenderAssets`] what
//! [`crate::vmesh::VmeshRegistry`] is to that store's vgeom half: the same rules,
//! reading a cooked pack (or a `--level` dev dir) instead of a project's loose
//! content root.
//!
//! # What is mirrored, and what is host-local
//!
//! Two functions are kept **character for character** identical to the editor's,
//! and `inf-editor-core`'s `tests/projector_mirror.rs` pins them as source text
//! (it links neither crate, so it runs on the Linux CI leg):
//!
//! * [`SkinnedRegistry::resolve_skinned`] — the **pose rule**. No skeleton (or a
//!   skeleton with no joints) ⇒ `None`, and the caller keeps the placeholder; the
//!   pose the SIM evaluated for this entity (P24.1) when there is one; else no
//!   `AnimPlayer`, no clip, or a clip that will not resolve ⇒ the **rest pose**;
//!   else the clip sampled at the play-head, honouring `looping`. The rest-pose
//!   arm is what makes a character *visible* rather than invisible-until-it-plays,
//!   and it is the same fallback the components document.
//!
//!   The **ONE permitted difference** between the two copies is the receiver:
//!   `&self` in this store (it memoizes behind its own locks because the projector
//!   holds it immutably) against `&mut self` in the editor's (the viewport owns
//!   its store mutably). That is a difference in who owns the cache, not in the
//!   rule, and `projector_mirror.rs` normalizes exactly that token — doc block
//!   included — before comparing.
//! * [`skinned_mesh_data`] — the bind-space rebuild (submeshes concatenated,
//!   indices rebased, an unskinned submesh pinned to joint 0 with weight 1).
//!
//! What is deliberately host-local is *where the bytes come from* and *when*: the
//! editor walks a mutable content root and re-scans on a miss, the player reads an
//! immutable pack entry (or a dev-dir index taken once). That is the same
//! asymmetry [`crate::vmesh`] already documents — but note the P18.3 **content
//! hash vs GUID** difference does **not** apply here. That one exists because both
//! vgeom render nodes cache GPU state by `VgeomAsset::id`, so a re-import under a
//! stable id would render stale; the skinned pass caches by the **pointer
//! identity** of the `Arc<SkinnedMeshData>` instead, and a re-import that produced
//! a different `Arc` is a different key by construction. There is no id to
//! content-address, so both hosts key skinned geometry by asset GUID.
//!
//! # The `Arc<SkinnedMeshData>` discipline
//!
//! `RenderScene::skinned_meshes` has been `Vec<Arc<SkinnedMeshData>>` since P18.3
//! *because* the skinned pass caches its GPU uploads by pointer identity. A
//! projector that rebuilt the bind-space stream every frame would re-upload
//! megabytes per frame while claiming to be cached. So this store owns exactly one
//! `Arc` per mesh asset for as long as it lives, hands out clones, and the
//! projection pushes the clone straight into the scene — no copy, no re-upload.
//! **The sharing is the convention this projector follows**; `skinned_geometry`
//! and the unit test `the_same_mesh_asset_hands_out_one_shared_arc` below are what
//! keep it true. It is invalidated by exactly one thing: dropping the registry
//! (a level switch builds a new one), because a cooked pack entry is immutable for
//! the life of the process.
//!
//! [`inf_editor_core::render_assets`]: https://docs.rs/inf-editor-core
//! [`inf_editor_core::render_assets::EditorRenderAssets`]: https://docs.rs/inf-editor-core

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use glam::Mat4;
use inf_anim::{AnimClip, AnimClipAsset, Skeleton, SkeletonAsset, StateMachineAsset};
use inf_asset::{AssetId, PackReader};
use inf_ecs::components::{AnimPlayer, AnimStateMachine, SkeletalMesh};
use inf_ecs::pose::EvaluatedPose;
use inf_render::SkinnedMeshData;
use uuid::Uuid;

// Native only — see `SkinnedRegistry::build_skinned_geometry` and this crate's
// `Cargo.toml` for why the browser player carries no `inf-mesh`. Gated at the
// import as well as the item because the wasm CI leg runs with `-D warnings`,
// where an import used by nothing is a build failure.
#[cfg(not(target_arch = "wasm32"))]
use inf_mesh::MeshAsset;
#[cfg(not(target_arch = "wasm32"))]
use inf_render::SkinnedVertex;

/// The loose asset extensions a dev-dir index covers — the four a
/// `SkeletalMesh` + `AnimPlayer` + `AnimStateMachine` triple can reach.
///
/// **`inf_sm` arrived with wave CHAR1a.2's preview idle, and its absence is the
/// reason that feature shipped inert for one demo run.** `resolve_skinned` asks
/// `machine_entry_clip` for the machine's entry clip; `machine_entry_clip` asks
/// `load_payload::<StateMachineAsset>`; and a `.inf_sm` that is not in this list
/// is not in the index, so the lookup missed, the rule fell through to the rest
/// pose, and every character in the editor stood in its bind pose exactly as
/// before. The gate did not see it because its arm registers the machine through
/// `insert_state_machine` and never touches the index — the frame saw it. The
/// arm now takes both doors.
const INDEXED_EXTENSIONS: [&str; 4] = ["inf_mesh", "inf_skel", "inf_anim", "inf_sm"];

/// A skeletal mesh resolved to something the skinned pass can draw.
///
/// **MIRROR** of `inf_editor_core::render_assets::SkinnedDraw`, field for field.
#[derive(Clone)]
pub struct SkinnedDraw {
    /// Bind-space geometry, shared between every entity using the same
    /// `(mesh, skeleton)` pair — and, across projections, the *same* `Arc`.
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
    pub sections: Arc<Vec<(u32, u32, Option<u128>)>>,
}

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

/// Where a registry reads payload bytes from.
enum Content {
    /// Nothing — the `--demo` world, the PIE window, the browser player. Every
    /// lookup misses, so every `SkeletalMesh` keeps its placeholder, which is
    /// exactly the pre-existing behaviour for those hosts.
    None,
    /// A cooked pack. The `Arc<PackReader>` is the **same mapping**
    /// [`crate::vmesh::VmeshRegistry`] pages meshlets out of — opened once per
    /// run, not once per registry.
    Pack(Arc<PackReader>),
    /// A `--level` dev dir: asset GUID → loose file, indexed once at load from
    /// the sibling `inf_asset` sidecars (the twin of
    /// [`VmeshRegistry::from_dir`](crate::vmesh::VmeshRegistry::from_dir)).
    Dir(HashMap<Uuid, PathBuf>),
    /// Bytes that arrived over the PIE wire (P24.1, `ScenePayload` v7). The editor
    /// has the project's asset DB and the player subprocess has no filesystem
    /// handle to it at all, so a PIE session's skeletal source *is* the payload —
    /// the same shape [`crate::voxel::VoxelRegistry`]'s `Memory` source took for
    /// the identical reason in P21.4.
    Memory(HashMap<Uuid, Vec<u8>>),
}

/// Every skeletal render asset the shipped player can draw, resolved by GUID out
/// of a cooked pack or a dev dir.
///
/// Resolution is **lazy and memoized** (hits *and* misses), like the editor's
/// store: a level whose partition streams a character in at minute ten pays for
/// that character then, and a dangling reference costs one failed lookup rather
/// than one per frame.
pub struct SkinnedRegistry {
    content: Content,
    /// Bind-space geometry by **mesh** GUID. `None` is a negative entry: this
    /// asset has no usable skin stream, so stop trying every frame.
    ///
    /// Keyed by the mesh alone — the editor's store keys the same cache by
    /// `(mesh, skeleton)`, but the value is a pure function of the `.inf_mesh`
    /// (see [`skinned_mesh_data`], which never looks at a skeleton), so the extra
    /// key component only splits the cache. The **slot** key the projection
    /// deduplicates on is still the full pair, which is what has to match.
    meshes: Mutex<HashMap<Uuid, Option<Arc<SkinnedMeshData>>>>,
    /// Decoded skeletons by GUID (shared by every entity bound to one).
    skeletons: Mutex<HashMap<Uuid, Option<Arc<Skeleton>>>>,
    /// Decoded clips by GUID.
    clips: Mutex<HashMap<Uuid, Option<Arc<AnimClip>>>>,
    /// **Decoded `.inf_sm` machines by GUID** (wave CHAR1a.2) — read for one
    /// thing only: which clip the ENTRY state plays, so an unplayed character can
    /// be previewed standing in its idle instead of its bind pose. The machine is
    /// never stepped here; stepping it is the fixed step's job and this is the
    /// projection.
    state_machines: Mutex<HashMap<Uuid, Option<Arc<inf_anim::StateMachine>>>>,
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
    sections: Mutex<HashMap<Uuid, Arc<Vec<(u32, u32, Option<u128>)>>>>,
    rest_palettes: Mutex<HashMap<RestPaletteKey, Arc<Vec<Mat4>>>>,
}

impl Default for SkinnedRegistry {
    fn default() -> Self {
        Self {
            content: Content::None,
            meshes: Mutex::default(),
            skeletons: Mutex::default(),
            clips: Mutex::default(),
            state_machines: Mutex::default(),
            sections: Mutex::default(),
            rest_palettes: Mutex::default(),
        }
    }
}

/// A poisoned cache lock is not a reason to take the game down: the map is a pure
/// memo of immutable content, so whatever a panicking thread left behind is still
/// a valid (possibly incomplete) cache.
fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

impl SkinnedRegistry {
    /// An inert registry: every `SkeletalMesh` keeps its placeholder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Resolve skeletal assets out of a cooked pack, sharing the caller's
    /// mapping.
    pub fn from_pack(reader: Arc<PackReader>) -> Self {
        Self {
            content: Content::Pack(reader),
            ..Self::default()
        }
    }

    /// Index the loose `.inf_mesh` / `.inf_skel` / `.inf_anim` files in `dir`
    /// (non-recursive) by their sidecar asset GUIDs — the dev-dir twin of
    /// [`from_pack`](Self::from_pack). Deterministic (path-sorted); files without
    /// a readable sidecar are skipped.
    pub fn from_dir(dir: &Path) -> Self {
        Self {
            content: Content::Dir(index_dir(dir)),
            ..Self::default()
        }
    }

    /// Serve skeletal payloads carried by a PIE `ScenePayload` (P24.1).
    ///
    /// All three byte vectors go into ONE map because they are keyed by asset
    /// GUID and asset GUIDs are unique across kinds — `load_payload` decodes by
    /// the type its caller asked for, so a mesh looked up as a skeleton would
    /// fail to decode rather than be mistaken for one.
    ///
    /// Before this door the windowed PIE player was handed
    /// [`SkinnedRegistry::new`] — an inert store — so every character in an
    /// embedded or new-window PIE session drew as a placeholder cube while the
    /// headless PIE the gates drive, and the shipped build, drew real geometry.
    /// P21.4 fixed the identical class for voxel volumes; this is the same fix.
    pub fn from_payload(
        meshes: &[(Uuid, Vec<u8>)],
        skeletons: &[(Uuid, Vec<u8>)],
        clips: &[(Uuid, Vec<u8>)],
        // **The MACHINES, since the wave CHAR1a audit — and they were missing.**
        //
        // `resolve_skinned`'s rule 3 and `resolve_skinned_shared` both ask
        // `machine_entry_clip`, which asks this store for the `.inf_sm`. Wave
        // CHAR1a.2 found that lookup broken because `"inf_sm"` was not an indexed
        // extension, and fixed `Content::Dir` in BOTH hosts — and this store has
        // THREE content sources. `Content::Memory` is the one a PIE session uses,
        // it was built from meshes + skeletons + clips, and `ScenePayload` has
        // carried a `machines` list since P24.1. So every `.inf_sm` lookup missed
        // in Play, every crowd agent off the pose path fell to `Pose::rest`, and
        // the street was full of characters standing in the mannequin's bind pose
        // while the hero beside them ran. Photographed by this audit's demo run:
        // an NPC with its arms out at 40 degrees below the horizontal (the
        // mannequin's A-pose bind: `hand_l` at (+0.478, +1.045) from a shoulder at
        // (+0.190, +1.436)) against the hero's idle, which puts `hand_l` at
        // (+0.222, +0.894) — nearly straight down.
        machines: &[(Uuid, Vec<u8>)],
    ) -> Self {
        let map: HashMap<Uuid, Vec<u8>> = meshes
            .iter()
            .chain(skeletons)
            .chain(clips)
            .chain(machines)
            .cloned()
            .collect();
        Self {
            content: Content::Memory(map),
            ..Self::default()
        }
    }

    /// Whether this registry can resolve anything at all (`false` for
    /// [`new`](Self::new)).
    pub fn has_content(&self) -> bool {
        !matches!(self.content, Content::None)
    }

    /// Number of mesh assets with resolved bind-space geometry (negative entries
    /// excluded) — the mirror of `EditorRenderAssets::loaded_skinned`.
    pub fn loaded_skinned(&self) -> usize {
        lock(&self.meshes).values().filter(|v| v.is_some()).count()
    }

    /// Register bind-space geometry directly, under a mesh asset GUID — the door
    /// for tests and for a host that built the stream rather than loading one
    /// (the twin of [`VmeshRegistry::insert_mesh`](crate::vmesh::VmeshRegistry::insert_mesh)).
    pub fn insert_mesh(&mut self, mesh_id: Uuid, data: SkinnedMeshData) {
        lock(&self.meshes).insert(mesh_id, Some(Arc::new(data)));
    }

    /// Register a decoded skeleton directly.
    pub fn insert_skeleton(&mut self, id: Uuid, skeleton: Skeleton) {
        lock(&self.skeletons).insert(id, Some(Arc::new(skeleton)));
    }

    /// Register a decoded clip directly.
    pub fn insert_clip(&mut self, id: Uuid, clip: AnimClip) {
        lock(&self.clips).insert(id, Some(Arc::new(clip)));
    }

    /// Register a decoded state machine directly (wave CHAR1a.2) — the door the
    /// preview-idle rule is measured through, and the twin of
    /// [`insert_clip`](Self::insert_clip).
    pub fn insert_state_machine(&mut self, id: Uuid, machine: inf_anim::StateMachine) {
        lock(&self.state_machines).insert(id, Some(Arc::new(machine)));
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
        &self,
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
        &self,
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
    fn rest_palette(&self, skeleton: &Skeleton, key: RestPaletteKey) -> Arc<Vec<Mat4>> {
        if let Some(hit) = lock(&self.rest_palettes).get(&key) {
            return hit.clone();
        }
        let pose = match key.2.and_then(|clip_id| self.clip(clip_id)) {
            Some(clip) => inf_anim::sample_clip(skeleton, &clip, 0.0, true),
            None => inf_anim::Pose::rest(skeleton),
        };
        let built = Arc::new(inf_anim::skinning_matrices(skeleton, &pose));
        lock(&self.rest_palettes).insert(key, built.clone());
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
    fn machine_entry_clip(&self, sm: Option<Uuid>) -> Option<Uuid> {
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
    fn state_machine(&self, id: Uuid) -> Option<Arc<inf_anim::StateMachine>> {
        let mut cache = lock(&self.state_machines);
        if let Some(hit) = cache.get(&id) {
            return hit.clone();
        }
        let loaded = self
            .load_payload::<StateMachineAsset>(id)
            .map(|a| Arc::new(a.machine));
        cache.insert(id, loaded.clone());
        loaded
    }

    /// A decoded skeleton, cached (hit or miss) by GUID.
    pub fn skeleton(&self, id: Uuid) -> Option<Arc<Skeleton>> {
        let mut cache = lock(&self.skeletons);
        if let Some(hit) = cache.get(&id) {
            return hit.clone();
        }
        let loaded = self
            .load_payload::<SkeletonAsset>(id)
            .map(|a| Arc::new(a.skeleton));
        cache.insert(id, loaded.clone());
        loaded
    }

    /// A decoded animation clip, cached (hit or miss) by GUID.
    pub fn clip(&self, id: Uuid) -> Option<Arc<AnimClip>> {
        let mut cache = lock(&self.clips);
        if let Some(hit) = cache.get(&id) {
            return hit.clone();
        }
        let loaded = self
            .load_payload::<AnimClipAsset>(id)
            .map(|a| Arc::new(a.clip));
        cache.insert(id, loaded.clone());
        loaded
    }

    /// Bind-space skinned geometry for a `(mesh, skeleton)` pair, cached.
    ///
    /// **The `Arc` sharing point.** One `Arc<SkinnedMeshData>` per mesh asset for
    /// the life of the registry: every projection of every entity bound to that
    /// mesh gets a clone of the same pointer, which is the identity the skinned
    /// pass caches its GPU upload on. Rebuilding it per projection would silently
    /// re-upload the whole bind-space stream every frame.
    fn skinned_geometry(&self, mesh_id: Uuid, _skeleton_id: Uuid) -> Option<Arc<SkinnedMeshData>> {
        let mut cache = lock(&self.meshes);
        if let Some(hit) = cache.get(&mesh_id) {
            return hit.clone();
        }
        let built = self.build_skinned_geometry(mesh_id).map(Arc::new);
        cache.insert(mesh_id, built.clone());
        built
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
    fn skinned_sections(&self, mesh_id: Uuid) -> Arc<Vec<(u32, u32, Option<u128>)>> {
        if let Some(hit) = lock(&self.sections).get(&mesh_id) {
            return hit.clone();
        }
        let built = Arc::new(self.build_skinned_sections(mesh_id));
        lock(&self.sections).insert(mesh_id, built.clone());
        built
    }

    /// Decode a mesh asset and derive its sections.
    ///
    /// **MIRROR** — see [`skinned_sections`](Self::skinned_sections).
    fn build_skinned_sections(&self, mesh_id: Uuid) -> Vec<(u32, u32, Option<u128>)> {
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

    /// Decode a mesh asset and rebuild its bind-space skinned stream.
    #[cfg(not(target_arch = "wasm32"))]
    fn build_skinned_geometry(&self, mesh_id: Uuid) -> Option<SkinnedMeshData> {
        self.load_payload::<MeshAsset>(mesh_id)
            .and_then(|m| skinned_mesh_data(&m))
    }

    /// **wasm32**: the browser player carries no `inf-mesh` — its `meshopt`
    /// dependency builds meshoptimizer's C++ through `cc`, which the wasm target
    /// has no toolchain/sysroot for (measured: `cargo check --target
    /// wasm32-unknown-unknown -p inf-player` fails on it), the same reason
    /// `inf-vgeom` gates `meshopt` off wasm32. So a `SkeletalMesh` keeps its
    /// placeholder in the browser — unchanged from before this batch — while
    /// every native target draws the real skinned geometry.
    #[cfg(target_arch = "wasm32")]
    fn build_skinned_geometry(&self, _mesh_id: Uuid) -> Option<SkinnedMeshData> {
        None
    }

    /// Read + decode one asset payload out of whatever content backs this
    /// registry. A missing entry is a clean `None` (⇒ the placeholder); a
    /// *corrupt* one warns and is also `None`, because one bad asset must not
    /// take a level down — the same rule `VmeshRegistry` loads under.
    fn load_payload<T: inf_asset::AssetPayload>(&self, id: Uuid) -> Option<T> {
        let bytes = match &self.content {
            Content::None => return None,
            Content::Pack(reader) => {
                let asset = AssetId(id);
                if !reader.contains(asset) {
                    return None;
                }
                match reader.read(asset) {
                    Ok(b) => b,
                    Err(e) => {
                        tracing::warn!("inf-player: read skeletal asset {id}: {e}");
                        return None;
                    }
                }
            }
            Content::Dir(index) => {
                let path = index.get(&id)?;
                match std::fs::read(path) {
                    Ok(b) => b,
                    Err(e) => {
                        tracing::warn!("inf-player: read {}: {e}", path.display());
                        return None;
                    }
                }
            }
            Content::Memory(map) => map.get(&id)?.clone(),
        };
        match inf_asset::decode::<T>(&bytes) {
            Ok(v) => Some(v),
            Err(e) => {
                tracing::warn!("inf-player: decode skeletal asset {id}: {e}");
                None
            }
        }
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
#[cfg(not(target_arch = "wasm32"))]
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

/// Index every loose skeletal asset in `dir` (non-recursive) by its GUID, read
/// from the sibling `inf_asset` `.toml` sidecar. Deterministic (path-sorted).
fn index_dir(dir: &Path) -> HashMap<Uuid, PathBuf> {
    let mut files: Vec<PathBuf> = match std::fs::read_dir(dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.extension()
                    .and_then(|s| s.to_str())
                    .is_some_and(|e| INDEXED_EXTENSIONS.contains(&e))
            })
            .collect(),
        Err(_) => return HashMap::new(),
    };
    files.sort();
    let mut out = HashMap::new();
    for p in files {
        match inf_asset::AssetSidecar::load(&p) {
            Ok(side) => {
                out.insert(side.guid.uuid(), p);
            }
            Err(_) => tracing::warn!(
                "inf-player: skeletal asset without a sidecar {}",
                p.display()
            ),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use inf_anim::{Interpolation, Joint, JointTrack, JointTransform, QuatTrack};

    const MESH: Uuid = Uuid::from_u128(0x5EED_0001);
    const SKEL: Uuid = Uuid::from_u128(0x5EED_0002);
    const CLIP: Uuid = Uuid::from_u128(0x5EED_0003);

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

    fn wave_clip() -> AnimClip {
        // `AnimClip::new` since `.inf_anim` v2 (P29.2): it derives `duration`
        // from the keys (2.0 here, unchanged) and fills the five new tail
        // channels with their empty defaults.
        AnimClip::new(
            "wave",
            vec![JointTrack {
                joint: 1,
                translation: None,
                rotation: Some(QuatTrack::new(
                    vec![0.0, 2.0],
                    vec![
                        glam::Quat::IDENTITY.to_array(),
                        glam::Quat::from_rotation_x(std::f32::consts::FRAC_PI_2).to_array(),
                    ],
                    Interpolation::Linear,
                )),
                scale: None,
            }],
        )
    }

    fn triangle() -> SkinnedMeshData {
        let v = |x: f32, y: f32, j: u32| SkinnedVertex {
            pos: [x, y, 0.0],
            normal: [0.0, 0.0, 1.0],
            uv: [x, y],
            joints: [j, 0, 0, 0],
            weights: [1.0, 0.0, 0.0, 0.0],
        };
        SkinnedMeshData {
            vertices: vec![v(0.0, 0.0, 0), v(1.0, 0.0, 0), v(0.0, 2.0, 1)],
            indices: vec![0, 1, 2],
        }
    }

    fn registry() -> SkinnedRegistry {
        let mut reg = SkinnedRegistry::new();
        reg.insert_mesh(MESH, triangle());
        reg.insert_skeleton(SKEL, two_joint_skeleton());
        reg.insert_clip(CLIP, wave_clip());
        reg
    }

    fn bound() -> SkeletalMesh {
        SkeletalMesh {
            mesh: Some(MESH),
            skeleton: Some(SKEL),
        }
    }

    /// Rule 2 of the pose rule: no `AnimPlayer` ⇒ the rest pose, so a character
    /// is visible the moment it is spawned rather than only once it plays.
    #[test]
    fn a_skeletal_mesh_projects_its_rest_pose() {
        let reg = registry();
        let draw = reg
            .resolve_skinned(&bound(), None, None, None)
            .expect("rest pose draws");
        assert_eq!(draw.mesh.vertices.len(), 3);
        assert_eq!(draw.mesh.indices, vec![0, 1, 2]);
        assert_eq!(draw.palette.len(), 2, "one matrix per joint");
        for m in draw.palette.iter() {
            assert!(
                (*m - Mat4::IDENTITY)
                    .to_cols_array()
                    .iter()
                    .all(|v| v.abs() < 1e-5),
                "rest palette must be identity, got {m:?}"
            );
        }
        assert_eq!(draw.key, (MESH, SKEL));
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
        let reg = registry();
        let per_agent = reg.resolve_skinned(&bound(), None, None, None).unwrap();
        let shared = reg.resolve_skinned_shared(&bound(), None).unwrap();
        assert_eq!(*shared.palette, *per_agent.palette);
        assert_eq!(shared.key, per_agent.key);
        assert!(std::sync::Arc::ptr_eq(&shared.mesh, &per_agent.mesh));

        // …and it is SHARED: two agents get one allocation, which is the thing
        // the atlas deduplicates on. An equality that held while every call
        // allocated a fresh `Vec` would satisfy the arm above and buy nothing.
        let again = reg.resolve_skinned_shared(&bound(), None).unwrap();
        assert!(
            std::sync::Arc::ptr_eq(&shared.palette, &again.palette),
            "the shared palette was rebuilt — the cache is not a cache"
        );
        assert!(
            !std::sync::Arc::ptr_eq(&shared.palette, &per_agent.palette),
            "the per-agent path handed back the shared block, so this arm proves nothing"
        );
    }

    /// Rule 3: the palette follows the play-head, deterministically.
    #[test]
    fn an_anim_player_drives_the_palette() {
        let reg = registry();
        let rest = reg
            .resolve_skinned(&bound(), None, None, None)
            .unwrap()
            .palette;
        let play = AnimPlayer {
            clip: Some(CLIP),
            // Mid-clip on purpose: `t == duration` on a LOOPING player wraps back
            // to 0, which would compare the rest pose to itself and pass for the
            // wrong reason.
            t: 0.5,
            ..AnimPlayer::default()
        };
        let posed = reg
            .resolve_skinned(&bound(), Some(&play), None, None)
            .unwrap()
            .palette;
        assert_ne!(rest[1].to_cols_array(), posed[1].to_cols_array());
        let again = reg
            .resolve_skinned(&bound(), Some(&play), None, None)
            .unwrap()
            .palette;
        assert_eq!(posed[1].to_cols_array(), again[1].to_cols_array());
    }

    /// Rule 2 again: an `AnimPlayer` whose clip cannot be resolved falls back to
    /// the rest pose rather than to nothing — a missing clip must not make a
    /// character vanish.
    #[test]
    fn an_unresolvable_clip_falls_back_to_rest() {
        let reg = registry();
        let rest = reg
            .resolve_skinned(&bound(), None, None, None)
            .unwrap()
            .palette;
        let ghost = reg
            .resolve_skinned(
                &bound(),
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

    /// Rule 1: half-bound (or unbacked) skeletal entities keep the placeholder,
    /// and an inert registry resolves nothing at all.
    #[test]
    fn an_unbound_skeletal_mesh_stays_a_placeholder() {
        let reg = registry();
        assert!(reg
            .resolve_skinned(&SkeletalMesh::default(), None, None, None)
            .is_none());
        assert!(reg
            .resolve_skinned(
                &SkeletalMesh {
                    mesh: Some(MESH),
                    skeleton: None
                },
                None,
                None,
                None,
            )
            .is_none());
        assert!(reg
            .resolve_skinned(
                &SkeletalMesh {
                    mesh: Some(Uuid::from_u128(7)),
                    skeleton: Some(SKEL)
                },
                None,
                None,
                None,
            )
            .is_none());
        assert!(SkinnedRegistry::new()
            .resolve_skinned(&bound(), None, None, None)
            .is_none());
        assert!(!SkinnedRegistry::new().has_content());
    }

    /// **The `Arc` discipline.** Two projections of the same mesh asset hand out
    /// the *same pointer*, which is the identity the skinned pass caches its GPU
    /// upload on. If this ever became a fresh `Arc` per resolve, every frame
    /// would re-upload the whole bind-space stream while the cache reported a hit
    /// rate of zero — invisible in every pixel comparison in the repo.
    #[test]
    fn the_same_mesh_asset_hands_out_one_shared_arc() {
        let reg = registry();
        let a = reg
            .resolve_skinned(&bound(), None, None, None)
            .unwrap()
            .mesh;
        let b = reg
            .resolve_skinned(&bound(), None, None, None)
            .unwrap()
            .mesh;
        assert!(
            Arc::ptr_eq(&a, &b),
            "the store must share one Arc<SkinnedMeshData> per mesh asset"
        );
        assert_eq!(reg.loaded_skinned(), 1);
    }

    /// A miss is memoized too: a dangling reference costs one failed lookup for
    /// the session, not one per frame.
    #[test]
    fn a_dangling_reference_misses_cheaply() {
        let reg = SkinnedRegistry::from_dir(Path::new("does-not-exist"));
        let ghost = SkeletalMesh {
            mesh: Some(Uuid::from_u128(1)),
            skeleton: Some(Uuid::from_u128(2)),
        };
        for _ in 0..5 {
            assert!(reg.resolve_skinned(&ghost, None, None, None).is_none());
        }
        assert_eq!(reg.loaded_skinned(), 0);
    }

    /// A rigid mesh with no skin stream is not skinned geometry (the caller keeps
    /// its placeholder).
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn an_unskinned_mesh_is_not_skinned_geometry() {
        use inf_mesh::{MeshVertex, SubMesh};
        let rigid = MeshAsset::new(
            vec![SubMesh {
                name: "quad".into(),
                vertices: vec![MeshVertex::default(); 3],
                indices: vec![0, 1, 2],
                material_slot: None,
                skin: Vec::new(),
            }],
            vec![],
        );
        assert!(skinned_mesh_data(&rigid).is_none());
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
        let reg = registry();
        let sm = bound();
        let play = AnimPlayer {
            clip: Some(CLIP),
            t: 0.5,
            ..AnimPlayer::default()
        };

        let rest = reg.resolve_skinned(&sm, None, None, None).unwrap().palette;
        let from_clip = reg
            .resolve_skinned(&sm, Some(&play), None, None)
            .unwrap()
            .palette;
        let posed = bent_pose(SKEL);
        let from_sim = reg
            .resolve_skinned(&sm, Some(&play), Some(&posed), None)
            .unwrap()
            .palette;

        // All three are distinct: the machine's pose is neither the rest pose nor
        // the clip's. (Comparing only against rest would pass for a store that
        // simply kept sampling the clip.)
        assert_ne!(from_sim[1].to_cols_array(), rest[1].to_cols_array());
        assert_ne!(from_sim[1].to_cols_array(), from_clip[1].to_cols_array());
        // …and with no player at all the sim pose still wins over rest.
        let alone = reg
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
        let reg = registry();
        let sm = bound();
        let rest = reg.resolve_skinned(&sm, None, None, None).unwrap().palette;

        let mut wrong_rig = bent_pose(SKEL);
        wrong_rig.skeleton = Uuid::from_u128(0xBAD_5CE1);
        assert_eq!(
            reg.resolve_skinned(&sm, None, Some(&wrong_rig), None)
                .unwrap()
                .palette[1]
                .to_cols_array(),
            rest[1].to_cols_array(),
            "a pose from another rig must not be worn"
        );

        let wrong_len = short_pose_a_guardless_store_would_wear(SKEL);
        let drawn = reg
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
}
