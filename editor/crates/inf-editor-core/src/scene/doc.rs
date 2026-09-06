//! The authoritative scene document (P3.2).
//!
//! [`SceneDoc`] wraps an [`inf_ecs::EcsWorld`] with editor state — selection,
//! a monotonic version, a dirty flag, an explicit creation-order list (so the
//! Outliner and the serialized file are deterministic) — and exposes the
//! primitive mutations (create / rename / delete / reparent / visibility /
//! select). Undo (P3.4) wraps these; the snapshot layer projects them to the
//! frontend DTOs; the viewport reads through `world()` to render + pick.
//!
//! GUIDs, not bevy `Entity` ids, are the identity that crosses every boundary:
//! entity ids are reused across despawn and never serialized.

use std::collections::{BTreeSet, HashMap};

use glam::{DVec2, DVec3};
use inf_ecs::components::{
    ActorClass, AnimStateMachine, AtlasRect, BodyKind3D, Camera, CharacterController3D,
    CharacterMovement, Collider3D, ColliderShape3DKind, Foliage, FoliageInstance,
    FoliagePaletteEntry, GlobalTransform, Light, Light2D, LightKind, Material, MeshRef, NineSlice,
    Primitive, RigidBody3D, SkeletalMesh, Spline, SplineInterp, Sprite, Terrain, Text2D, Tilemap,
    TimeOfDay, Transform, Visibility, Volume, VolumeKind, WaterBody, WaterKind,
};
use inf_ecs::{Color, ComputedVisibility, EcsWorld, Entity, PropValue, Vec2d, Vec3d};
use inf_terrain::{
    BiomeDelta, BiomeStroke, BrushOp, BrushParams, DataMapDelta, HeightDelta, SplatDelta,
    SplatStroke, Stroke, TerrainData,
};
use uuid::Uuid;

use crate::ipc::{SceneDelta, SceneNode, SceneSnapshot, SpawnKind};
use crate::scene::serialize::{EntityRecord, LevelSettings};
use crate::scene::undo::{EditCommand, EditHistory};

/// **The surface a character is created wearing** (wave CHAR1a audit) — the
/// `.inf_mat` GUID plus the scalars flattened off it, which is exactly what
/// [`SceneDoc::edit_apply_material`] writes onto a `Material` component.
///
/// # Why the door takes the numbers and not just the id
///
/// The renderer reads the *component*: `host.rs`/`render.rs` take
/// `base_color`/`metallic`/`roughness` from it and use `Material::asset` only to
/// resolve the virtual-texture set (`inf_render::vt_set_for`). So a door that
/// bound the id and left the scalars at their defaults would have written a
/// component whose numbers do not describe the material it names — the exact
/// state P26.3b's "the binding *adds* the texture edge, it never becomes the
/// only copy of the numbers" forbids. Four fields, and they are the four the
/// flattening has always carried.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CharacterSkin {
    /// The `.inf_mat` this surface is bound to.
    pub asset: Uuid,
    /// Linear RGBA base colour, flattened off the material.
    pub base_color: [f32; 4],
    /// 0 = dielectric, 1 = metal.
    pub metallic: f32,
    /// Perceptual roughness.
    pub roughness: f32,
}

impl CharacterSkin {
    /// Flatten an authored material into the four fields the component carries.
    pub fn from_material(asset: Uuid, m: &inf_material::MaterialAsset) -> Self {
        Self {
            asset,
            base_color: m.base_color,
            metallic: m.metallic,
            roughness: m.roughness,
        }
    }
}

/// Per-bake accounting returned by [`SceneDoc::edit_erode_region`], derived from
/// the committed height delta (so it is adapter-independent, unlike GPU float
/// order). Richer stats (sediment moved, water balance) come from the CPU
/// reference's [`inf_terrain::ErosionStats`] on the CPU path only.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ErodeReport {
    /// Height samples the bake actually changed (`before != after`).
    pub cells_changed: usize,
    /// Data-map samples the bake actually changed (P19.1) — the flow /
    /// deposition / wear accumulators it moved. Always **at least**
    /// `cells_changed`: a height only moves through the erode/deposit pass,
    /// which writes a map in the same breath.
    pub map_cells_changed: usize,
    /// Net terrain-volume change `Σ(after − before)·l²` (world m³); negative =
    /// net-eroded.
    pub mass_delta: f64,
}

pub struct SceneDoc {
    world: EcsWorld,
    /// Creation order of every live entity — the single ordering source for the
    /// Outliner tree and the serialized file (bevy iteration order is unstable).
    order: Vec<Uuid>,
    selection: Vec<Uuid>,
    version: u64,
    dirty: bool,
    title: String,
    /// File-level simulation settings (gravity + rate), persisted in `.inf_lvl`
    /// (schema v3). Defaults preserve pre-v3 behaviour.
    settings: LevelSettings,
    /// Where on Earth this level is (schema v24). Disabled by default.
    geo: inf_math::geo::GeoAnchor,
    history: EditHistory,
    /// The pre-preview authored scalars captured when a sequencer scrub arms a
    /// live preview — `(target, reflection-path, authored-value)` — or `None`
    /// when no preview is active. A save mid-scrub temporarily rolls the world
    /// back to these so the file carries the AUTHORED values, then re-applies
    /// the preview (see [`Self::with_authored_scene`]).
    preview: Option<Vec<(Uuid, String, f64)>>,
    /// **Process-unique identity of this document instance** (P21.3 audit).
    ///
    /// Not persisted and not part of the level: it exists so a *different
    /// thread* can tell that the document under it was replaced wholesale.
    /// `scene_open` and `scene_new` do `*doc = …` under the lock; the viewport
    /// thread wakes up holding gestures whose state — a stroke's terrain entity,
    /// a gizmo drag's selection, an open transaction — refers to a document that
    /// no longer exists. Committing one then writes the *old* level's edit into
    /// the new one, and a single Ctrl+Z applies it.
    ///
    /// A monotone counter rather than a hash of the contents, because two
    /// identical levels opened in sequence are still two different documents to
    /// anything holding a mid-gesture reference into one of them.
    doc_id: u64,
    /// **What has changed since the last projection** (IB-13).
    ///
    /// `Some(set)` names the guids a mutation moved; `None` means "everything",
    /// which is what [`SceneDoc::touch`] records and what
    /// [`SceneDoc::project_delta`] pays a full walk for. It is a *conservative
    /// union* — one `touch` in a batch widens the batch — so a call site that has
    /// not been narrowed to [`SceneDoc::touch_at`] is slow and never wrong.
    scope: Option<BTreeSet<Uuid>>,
    /// The guid set the last projection published.
    ///
    /// Replaces the retained full `SceneSnapshot` Ring 2 used to diff against: a
    /// removal is `projected − live`, and a scoped projection asks only whether
    /// each named guid still exists. 100 000 uuids is 1.6 MB against a snapshot's
    /// hundred thousand nodes and their strings.
    projected: BTreeSet<Uuid>,
    /// The root list the last projection published.
    ///
    /// `SceneDelta` carries `roots` whole on every emit, and deriving it costs a
    /// walk of `order` — which is exactly the O(n) this item exists to remove
    /// from the drag path. Only a full projection refreshes it, and only a
    /// structural change (create / delete / reparent) can move it; both of those
    /// go through [`touch`](Self::touch), which forces a full projection. That is
    /// the same invariant [`touch_at`](Self::touch_at) states, read from the
    /// other end.
    roots_cache: Vec<String>,
    /// Each published guid's position in [`order`](Self::order) — **creation
    /// rank** (I3 audit).
    ///
    /// A `SceneNode` states its children *in creation order*, and the full
    /// projection gets that for free by walking `order`. A scoped projection
    /// cannot walk `order`, so it asks the world instead — and `Children` is
    /// bevy's own list, in the order the links were last *inserted*. The two
    /// agree until something re-parents, and then a rename of the parent shipped
    /// a re-ordered child list that the next full projection put back: the
    /// Outliner's tree reordered under the cursor and nothing reported it
    /// (`a_scoped_node_lists_its_children_in_creation_order`).
    ///
    /// Refreshed by [`hierarchy_index`](Self::hierarchy_index), i.e. exactly
    /// where [`roots_cache`](Self::roots_cache) is, and it is stale in exactly
    /// the same circumstances — `order` only moves on a create, a delete or a
    /// load, and all three take the full path. ~2 MB at 100 000 entities, which
    /// buys the scoped branch an `O(children)` sort instead of an `O(entities)`
    /// walk.
    order_rank: HashMap<Uuid, u32>,
}

/// Source of [`SceneDoc::doc_id`]. Never wraps in any plausible session (one
/// document per open/new; `u64` at one per nanosecond is 584 years).
static NEXT_DOC_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// The reflect type path of [`inf_ecs::components::PcgVolume`], spelled the way
/// `scene::details` spells the four asset-reference components it names.
///
/// A `const` rather than `type_path_for::<PcgVolume>()` because it is compared
/// once per reflected field write, which is once per frame of a numeric drag.
const PCG_VOLUME_TYPE_PATH: &str = "inf_ecs::components::PcgVolume";

impl Default for SceneDoc {
    fn default() -> Self {
        Self::new()
    }
}

impl SceneDoc {
    pub fn new() -> Self {
        Self {
            world: EcsWorld::new(),
            order: Vec::new(),
            selection: Vec::new(),
            version: 0,
            dirty: false,
            title: "Untitled".to_string(),
            settings: LevelSettings::default(),
            geo: Default::default(),
            history: EditHistory::default(),
            preview: None,
            doc_id: NEXT_DOC_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            scope: None,
            projected: BTreeSet::new(),
            roots_cache: Vec::new(),
            order_rank: HashMap::new(),
        }
    }

    /// This document instance's process-unique id — see [`SceneDoc::doc_id`]'s
    /// field docs for why another thread needs it.
    pub fn doc_id(&self) -> u64 {
        self.doc_id
    }

    // ── accessors ────────────────────────────────────────────────────────

    pub fn world(&self) -> &EcsWorld {
        &self.world
    }

    pub fn world_mut(&mut self) -> &mut EcsWorld {
        &mut self.world
    }

    pub fn version(&self) -> u64 {
        self.version
    }

    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    pub fn title(&self) -> &str {
        &self.title
    }

    pub fn set_title(&mut self, title: impl Into<String>) {
        self.title = title.into();
    }

    /// File-level simulation settings (gravity + rate), persisted in `.inf_lvl`.
    pub fn settings(&self) -> LevelSettings {
        self.settings
    }

    /// Replace the level settings (the loader + a settings editor call this).
    /// Loader-only: does **not** dirty the document or bump the version.
    pub fn set_settings(&mut self, settings: LevelSettings) {
        self.settings = settings;
    }

    /// Where on Earth world `(0, 0, 0)` is (schema v24, Wave G).
    ///
    /// Disabled on an ordinary made-up world. Every GIS import door reads this
    /// to know what to reproject into, and the cook compares it against
    /// `TimeOfDay`'s hand-authored latitude.
    pub fn geo(&self) -> &inf_math::geo::GeoAnchor {
        &self.geo
    }

    /// Replace the geo-anchor. **Loader-only**: does not dirty the document or
    /// bump the version, exactly like [`set_settings`](Self::set_settings).
    pub fn set_geo(&mut self, geo: inf_math::geo::GeoAnchor) {
        self.geo = geo;
    }

    /// Replace the geo-anchor **as an author edit**: dirties the document and
    /// bumps the version so the viewport and the panels re-sync.
    ///
    /// Not an undo step. Anchoring a level is a once-per-project act that every
    /// subsequent import is interpreted against, and quietly reversing it with a
    /// Ctrl+Z aimed at something else would silently move every georeferenced
    /// asset in the level. The panel is the place to change it back.
    pub fn edit_geo(&mut self, geo: inf_math::geo::GeoAnchor) {
        if self.geo == geo {
            return;
        }
        self.geo = geo;
        self.touch();
    }

    /// Raw (non-recording) level-settings write for the undo layer + [`Self::edit_settings`].
    /// Marks the document dirty + bumps the version (a settings change is a real edit),
    /// unlike the loader-only [`Self::set_settings`].
    pub(crate) fn raw_set_settings(&mut self, settings: LevelSettings) {
        self.settings = settings;
        self.touch();
    }

    /// Edit the level settings (the World Settings panel), recorded as one undo
    /// step. No-op (and records nothing) when `new` equals the current settings.
    pub fn edit_settings(&mut self, new: LevelSettings) {
        let old = self.settings;
        if old == new {
            return;
        }
        self.raw_set_settings(new);
        self.history
            .record("World Settings", EditCommand::SetLevelSettings { old, new });
    }

    pub fn selection(&self) -> &[Uuid] {
        &self.selection
    }

    /// Live entities in creation order.
    pub fn order(&self) -> &[Uuid] {
        &self.order
    }

    pub fn entity_of(&self, guid: Uuid) -> Option<Entity> {
        self.world.entity_of(guid)
    }

    /// Bump the version; mark unsaved. Every mutation funnels through here.
    ///
    /// **Widens the projection scope to everything** (IB-13): a caller that does
    /// not say what it moved has said it might have moved anything, and the next
    /// [`project_delta`](Self::project_delta) pays a full walk. That is the
    /// conservative direction — slow, never wrong — and it is why narrowing a
    /// call site to [`touch_at`](Self::touch_at) is a strict improvement rather
    /// than a migration.
    pub(crate) fn touch(&mut self) {
        self.version += 1;
        self.dirty = true;
        self.scope = None;
    }

    /// [`touch`](Self::touch), naming exactly the entities whose projection
    /// changed.
    ///
    /// # The contract
    ///
    /// **Only for changes that cannot move an entity in the hierarchy.** A
    /// `SceneNode`'s `children` and a projection's `roots` are derived from
    /// *other* entities' parent links, so a create, a delete or a reparent
    /// changes nodes this call has no way to name — those must use
    /// [`touch`](Self::touch). Renames, visibility, transforms and property
    /// edits are what this is for, and between them they are every frame of a
    /// gizmo drag.
    ///
    /// Widening is monotone: once a batch has seen one `touch`, later
    /// `touch_at`s cannot narrow it back.
    pub(crate) fn touch_at(&mut self, guids: impl IntoIterator<Item = Uuid>) {
        self.version += 1;
        self.dirty = true;
        if let Some(scope) = self.scope.as_mut() {
            scope.extend(guids);
        }
    }

    /// [`touch_at`](Self::touch_at) over `guid` **and every descendant**.
    ///
    /// The door for anything that changes `effective_visible`, which is a
    /// property of an entity's whole ancestry: hiding a folder changes the
    /// projection of everything under it, and naming only the folder would leave
    /// the Outliner drawing its children at full brightness. `O(subtree)`, which
    /// is `O(1)` for the leaf every gizmo drag touches.
    pub(crate) fn touch_subtree(&mut self, guid: Uuid) {
        let mut guids = vec![guid];
        if let Some(e) = self.world.entity_of(guid) {
            guids.extend(self.world.subtree(e).into_iter().filter_map(|se| {
                let g = self.world.guid_of(se)?;
                (g != guid).then_some(g)
            }));
        }
        self.touch_at(guids);
    }

    /// [`touch_at`](Self::touch_at) without dirtying — the selection's door.
    ///
    /// Selection is not an unsaved edit (a saved file does not record who was
    /// clicked), but it *is* a projection change: the delta's tail carries the
    /// selection and the frontend refetches Details from it.
    pub(crate) fn touch_selection(&mut self) {
        self.version += 1;
    }

    /// Push a prepared [`EditCommand`] onto the history — the door for edits
    /// whose `edit_*` entry point lives in [`crate::scene::undo`] rather than
    /// here.
    ///
    /// The `history` field is private to this module, so a command assembled
    /// elsewhere in the crate has no other way in. Deliberately narrow: it does
    /// **not** touch or dirty the document, because the callers that use it have
    /// already applied their edit live (a brush stroke) and record only at
    /// commit — exactly like `edit_commit_sculpt` beside them.
    pub(crate) fn record_edit(&mut self, label: &str, cmd: EditCommand) {
        self.history.record(label, cmd);
    }

    /// Bump the version **without** dirtying — the Simulate loop (P8.4) calls
    /// this after mutating the ECS world so the viewport re-syncs, but a live
    /// preview must not mark the document unsaved (exit restores it anyway).
    ///
    /// **Widens the projection scope to everything** (IB-13). The Simulate loop
    /// writes straight into the ECS world without going through any `SceneDoc`
    /// mutation, so there is nothing to name — treating this as a narrow change
    /// would leave the Outliner showing the pre-Simulate world while the viewport
    /// showed the live one.
    pub fn bump_version_for_runtime(&mut self) {
        self.version += 1;
        self.scope = None;
    }

    /// Clear the dirty flag (after a successful save) without bumping version.
    pub fn mark_saved(&mut self) {
        self.dirty = false;
    }

    // ── mutations ────────────────────────────────────────────────────────

    /// Create an entity of `kind` under `parent`, returning its GUID. When
    /// `name` is empty a kind-appropriate default is used.
    pub fn create(&mut self, kind: SpawnKind, name: &str, parent: Option<Uuid>) -> Uuid {
        let guid = Uuid::new_v4();
        self.create_with_guid(guid, kind, name, parent);
        guid
    }

    /// Create with an explicit GUID (load + deterministic tests + undo redo).
    pub fn create_with_guid(
        &mut self,
        guid: Uuid,
        kind: SpawnKind,
        name: &str,
        parent: Option<Uuid>,
    ) -> Entity {
        let label = if name.is_empty() {
            default_name(kind)
        } else {
            name.to_string()
        };
        let parent_entity = parent.and_then(|p| self.world.entity_of(p));
        let entity = self.world.spawn_with_guid(guid, &label, parent_entity);
        attach_kind(&mut self.world, entity, kind);
        self.order.push(guid);
        self.touch();
        entity
    }

    /// Low-level spawn for the loader: explicit identity, no kind components
    /// (the caller inserts concrete components). Does not dirty the document.
    pub(crate) fn spawn_bare(&mut self, guid: Uuid, name: &str, parent: Option<Uuid>) -> Entity {
        let parent_entity = parent.and_then(|p| self.world.entity_of(p));
        let entity = self.world.spawn_with_guid(guid, name, parent_entity);
        self.order.push(guid);
        entity
    }

    /// Loader / undo parent fix-up: re-attach `guid` under `parent` WITHOUT
    /// dirtying the document or bumping the version (mirrors [`Self::spawn_bare`]).
    /// A no-op when `guid` is already correctly parented — so an already-valid,
    /// parents-precede-children file re-attaches nothing and re-saves
    /// byte-identically. The two-pass spawn paths (loader + delete→undo) call
    /// this after every GUID exists, so a child recorded BEFORE its parent (a
    /// node reparented under a later-created one) lands under the right parent
    /// instead of silently falling back to the root.
    pub(crate) fn raw_fixup_parent(&mut self, guid: Uuid, parent: Option<Uuid>) {
        let Some(child) = self.world.entity_of(guid) else {
            return;
        };
        let current = self
            .world
            .parent_of(child)
            .and_then(|p| self.world.guid_of(p));
        if current == parent {
            return;
        }
        let parent_entity = parent.and_then(|p| self.world.entity_of(p));
        self.world.reparent(child, parent_entity);
    }

    /// Empty the document (before a load). Keeps title/version bookkeeping to
    /// the caller.
    ///
    /// **The geo-anchor is cleared here for the same reason `settings` is**
    /// (Wave-G audit A4). Leaving it behind made the loader's own regression arm
    /// blind: the arm seeded a document with the anchor it then asserted, and
    /// `reset` preserved it, so deleting `apply_to_doc`'s `set_geo` line — the
    /// exact line the wave confessed to having forgotten once — left all 666 of
    /// this crate's tests green. Measured. A load path that forgets an
    /// attachment does not crash, it agrees with itself (the P21 law), and a
    /// gate cannot watch a line whose absence it cannot feel.
    pub(crate) fn reset(&mut self) {
        self.world.clear();
        self.order.clear();
        self.selection.clear();
        self.settings = LevelSettings::default();
        self.geo = inf_math::geo::GeoAnchor::default();
        self.preview = None;
    }

    /// Read one entity's editable component properties (Details, P3.3).
    pub fn entity_props(&self, guid: Uuid) -> Vec<inf_ecs::ComponentProps> {
        match self.world.entity_of(guid) {
            Some(e) => self.world.read_props(e),
            None => Vec::new(),
        }
    }

    /// The Outliner label for `guid`.
    pub fn display_name(&self, guid: Uuid) -> String {
        self.world
            .entity_of(guid)
            .and_then(|e| self.world.name_of(e))
            .unwrap_or("")
            .to_string()
    }

    /// The UE-style type label for `guid`.
    pub fn kind_of_guid(&self, guid: Uuid) -> String {
        self.world
            .entity_of(guid)
            .map(|e| kind_of(&self.world, e))
            .unwrap_or_default()
    }

    /// The Details view of the current selection (P3.3).
    pub fn details(&self) -> crate::ipc::DetailsDto {
        crate::scene::details::build(self)
    }

    /// Reset one component field to its `Default` value (P3.3.4), recorded for
    /// undo. Returns whether it changed.
    pub fn edit_reset_prop(&mut self, guid: Uuid, type_path: &str, field: &str) -> bool {
        match inf_ecs::default_field(self.world.registry(), type_path, field) {
            Some(default) => self.edit_set_prop(guid, type_path, field, &default),
            None => false,
        }
    }

    /// Write one component field on `guid`. Returns whether it applied.
    pub fn write_prop(
        &mut self,
        guid: Uuid,
        type_path: &str,
        field: &str,
        value: &inf_ecs::PropValue,
    ) -> bool {
        let Some(e) = self.world.entity_of(guid) else {
            return false;
        };
        let ok = self.world.write_prop(e, type_path, field, value);
        if ok {
            // IB-13: a reflected field write cannot re-parent an entity (the
            // hierarchy is not an editable component), so the projection change
            // is bounded by this entity's subtree — the subtree rather than the
            // entity because `Visibility::visible` is one of the fields that
            // arrives here, and it changes every descendant's
            // `effective_visible`. This is the Details panel's write and every
            // frame of a numeric drag in it.
            self.touch_subtree(guid);
            self.invalidate_pcg_population(guid, type_path);
        }
        ok
    }

    /// **An edited PCG volume is a stale PCG volume** (wave EDIT1, clause 3).
    ///
    /// `PcgVolume::evaluated` is a `#[serde(skip)]` cache derived from the
    /// graph, the seed, the extent and the ground. Three of those four are
    /// editable in the Details panel, and before this the cache simply outlived
    /// them: an author who widened a block's extent kept looking at the old
    /// block until they right-clicked Evaluate. Dropping the population is what
    /// makes the camera streamer (or the explicit command) put the new one
    /// there — the P10 node-editor loop, on a component instead of a graph.
    ///
    /// Dropped rather than re-evaluated in place, because this is called from
    /// inside a numeric drag: re-evaluating here would run a five-millisecond
    /// city build on every frame of a slider, where the streamer's next tick
    /// does it once, off the thread that draws.
    ///
    /// The write goes through `set_population` for the reason the streamer's
    /// release does: that call is what stamps `structures_gen`, and the physics
    /// bridge and the sim→render fold both read the stamp.
    fn invalidate_pcg_population(&mut self, guid: Uuid, type_path: &str) {
        if type_path != PCG_VOLUME_TYPE_PATH {
            return;
        }
        let Some(e) = self.world.entity_of(guid) else {
            return;
        };
        let w = self.world.world_mut();
        let Some(mut vol) = w.get_mut::<inf_ecs::components::PcgVolume>(e) else {
            return;
        };
        if vol.evaluated.is_empty() && vol.structures.is_empty() {
            return;
        }
        vol.set_population(
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            inf_nav::NavGraph::default(),
            Vec::new(),
            Vec::new(),
        );
    }

    /// Write one component field for LIVE PREVIEW (the sequencer scrub, P11.4):
    /// applies the value and bumps the version so the native viewport re-syncs,
    /// but does **not** dirty the document. A scrub session restores the captured
    /// originals on stop, so previewing a sequence never marks the scene unsaved
    /// (mirrors [`Self::bump_version_for_runtime`], which the Simulate loop uses).
    pub fn write_prop_preview(
        &mut self,
        guid: Uuid,
        type_path: &str,
        field: &str,
        value: &inf_ecs::PropValue,
    ) -> bool {
        let Some(e) = self.world.entity_of(guid) else {
            return false;
        };
        let ok = self.world.write_prop(e, type_path, field, value);
        if ok {
            self.bump_version_for_runtime();
        }
        ok
    }

    // ── sequencer preview guard (P11.4) ──────────────────────────────────────
    //
    // A scrub writes interpolated values into the live world through
    // `write_prop_preview` (non-dirtying). While that preview is live the world
    // no longer holds the AUTHORED scene, so a save mid-scrub would persist the
    // interpolated values. These seams let the scrub arm a guard holding the
    // pre-preview authored scalars; [`Self::with_authored_scene`] rolls the world
    // back to them for the duration of a serialize, then re-applies the preview.

    /// Whether a non-dirtying sequencer preview (scrub) is currently live.
    pub fn is_previewing(&self) -> bool {
        self.preview.is_some()
    }

    /// Arm the preview guard with the pre-preview authored scalars to restore on
    /// save — `(target, reflection-path, authored-value)`. Called by the scrub
    /// seam ([`crate::sequencer::apply_sequence_at`]) as it starts writing
    /// interpolated values; [`Self::end_preview`] disarms it.
    pub fn begin_preview(&mut self, authored: Vec<(Uuid, String, f64)>) {
        self.preview = Some(authored);
    }

    /// Disarm the preview guard (the scrub restored the authored values). Called
    /// by [`crate::sequencer::restore_snapshot`].
    pub fn end_preview(&mut self) {
        self.preview = None;
    }

    /// Run `f` against the AUTHORED scene while a sequencer preview is live: the
    /// world holds interpolated scrub scalars, so this rolls the previewed
    /// scalars back to their pre-preview authored values, runs `f` (e.g.
    /// serialize), then re-applies the preview — leaving the editor still showing
    /// the preview. A plain passthrough when no preview is armed, so a normal
    /// save is byte-for-byte unchanged.
    pub fn with_authored_scene<R>(&mut self, f: impl FnOnce(&SceneDoc) -> R) -> R {
        let Some(authored) = self.preview.take() else {
            return f(self);
        };
        // Remember the live preview scalars so we can re-apply them after `f`.
        let live: Vec<(Uuid, String, f64)> = authored
            .iter()
            .filter_map(|(t, p, _)| {
                crate::sequencer::read_scalar(self, *t, p).map(|v| (*t, p.clone(), v))
            })
            .collect();
        // Roll the world back to the authored values, run `f`, then restore the
        // preview so the editor keeps showing it.
        for (t, p, v) in &authored {
            crate::sequencer::write_scalar_preview(self, *t, p, *v);
        }
        let out = f(self);
        for (t, p, v) in &live {
            crate::sequencer::write_scalar_preview(self, *t, p, *v);
        }
        self.preview = Some(authored);
        out
    }

    pub fn rename(&mut self, guid: Uuid, name: &str) {
        if let Some(e) = self.world.entity_of(guid) {
            self.world.rename(e, name);
            // IB-13: a rename moves one node and nothing else — not even its
            // children, which carry guids and not names.
            self.touch_at([guid]);
        }
    }

    pub fn set_visible(&mut self, guid: Uuid, visible: bool) {
        if let Some(e) = self.world.entity_of(guid) {
            self.world.set_visible(e, visible);
            // IB-13: `effective_visible` is a property of the ancestry, so
            // hiding a folder changes every node under it.
            self.touch_subtree(guid);
        }
    }

    /// Re-parent `guid` under `parent` (`None` → root). Returns false if the
    /// move would create a cycle or the guid is unknown.
    pub fn reparent(&mut self, guid: Uuid, parent: Option<Uuid>) -> bool {
        let Some(child) = self.world.entity_of(guid) else {
            return false;
        };
        let parent_entity = match parent {
            Some(p) => match self.world.entity_of(p) {
                Some(e) => Some(e),
                None => return false,
            },
            None => None,
        };
        let ok = self.world.reparent(child, parent_entity);
        if ok {
            self.touch();
        }
        ok
    }

    /// Delete each guid (and its descendants). Prunes selection + order.
    pub fn delete(&mut self, guids: &[Uuid]) {
        for &guid in guids {
            if let Some(e) = self.world.entity_of(guid) {
                let subtree: Vec<Uuid> = self
                    .world
                    .subtree(e)
                    .into_iter()
                    .filter_map(|se| self.world.guid_of(se))
                    .collect();
                self.world.despawn(e);
                self.order.retain(|g| !subtree.contains(g));
                self.selection.retain(|g| !subtree.contains(g));
            }
        }
        self.touch();
    }

    // ── selection ────────────────────────────────────────────────────────

    /// Set (or extend, when `additive`) the selection. Unknown guids are
    /// dropped. Selection changes bump the version (so the UI re-syncs) but do
    /// **not** dirty the document.
    pub fn select(&mut self, guids: &[Uuid], additive: bool) {
        let valid: Vec<Uuid> = guids
            .iter()
            .copied()
            .filter(|g| self.world.entity_of(*g).is_some())
            .collect();
        if additive {
            for g in valid {
                if let Some(pos) = self.selection.iter().position(|s| *s == g) {
                    self.selection.remove(pos); // toggle off
                } else {
                    self.selection.push(g);
                }
            }
        } else {
            self.selection = valid;
        }
        // IB-13: a selection change moves no NODE — only the delta's tail —
        // so it does not widen the projection scope.
        self.touch_selection();
    }

    /// **Append without toggling** — Shift+click in the viewport (Wave E).
    ///
    /// A separate method rather than a second meaning for `additive`, because
    /// `additive` is documented and relied on as a TOGGLE (Ctrl+click removes an
    /// object that is already selected, which is exactly what Shift+click must
    /// NOT do — shift-clicking across a group would deselect half of it).
    /// Already-selected guids keep their position, so the primary (the first
    /// entry, which the gizmo centres on) does not move under the user.
    pub fn select_append(&mut self, guids: &[Uuid]) {
        let mut changed = false;
        for g in guids {
            if self.world.entity_of(*g).is_none() || self.selection.contains(g) {
                continue;
            }
            self.selection.push(*g);
            changed = true;
        }
        if changed {
            self.touch_selection();
        }
    }

    pub fn clear_selection(&mut self) {
        if !self.selection.is_empty() {
            self.selection.clear();
            self.touch_selection();
        }
    }

    // ── raw mutations for undo (non-recording) ───────────────────────────
    //
    // These are the same primitives the public methods use, exposed so
    // `EditCommand::apply`/`revert` can drive the world without re-entering the
    // recorder. They still `touch()` (a revert is a real change).

    pub(crate) fn raw_rename(&mut self, guid: Uuid, name: &str) {
        self.rename(guid, name);
    }

    pub(crate) fn raw_reparent(&mut self, guid: Uuid, parent: Option<Uuid>) -> bool {
        self.reparent(guid, parent)
    }

    pub(crate) fn raw_set_visible(&mut self, guid: Uuid, visible: bool) {
        self.set_visible(guid, visible);
    }

    pub(crate) fn raw_delete(&mut self, guids: &[Uuid]) {
        self.delete(guids);
    }

    pub(crate) fn raw_write_prop(
        &mut self,
        guid: Uuid,
        type_path: &str,
        field: &str,
        value: &PropValue,
    ) -> bool {
        self.write_prop(guid, type_path, field, value)
    }

    pub(crate) fn raw_set_transform(&mut self, guid: Uuid, t: Transform) {
        if let Some(e) = self.world.entity_of(guid) {
            self.world.world_mut().entity_mut(e).insert(t);
            self.world.mark_dirty();
            // IB-13, and this is **the** gizmo-drag path: a `SceneNode` carries
            // no transform at all (guid, name, kind, visible, effective_visible,
            // parent, children), so moving an entity changes no node. The
            // version bump is what the viewport re-syncs on; the delta carries
            // the doc tail and no nodes.
            self.touch_at([guid]);
        }
    }

    /// Read an entity's [`Sprite`] component (the fields — `texture`,
    /// `atlas_rect` — that the reflection Details grid can't reach).
    pub(crate) fn raw_get_sprite(&self, guid: Uuid) -> Option<Sprite> {
        let e = self.world.entity_of(guid)?;
        self.world.world().get::<Sprite>(e).cloned()
    }

    /// Insert (`Some`) or remove (`None`) an entity's [`Sprite`] component.
    pub(crate) fn raw_set_sprite(&mut self, guid: Uuid, sprite: Option<Sprite>) {
        if let Some(e) = self.world.entity_of(guid) {
            match sprite {
                Some(s) => {
                    self.world.world_mut().entity_mut(e).insert(s);
                }
                None => {
                    self.world.world_mut().entity_mut(e).remove::<Sprite>();
                }
            }
            self.world.mark_dirty();
            self.touch();
        }
    }

    /// Read an entity's [`ActorClass`] blueprint-class binding GUID, if any.
    pub(crate) fn raw_get_actor(&self, guid: Uuid) -> Option<Uuid> {
        let e = self.world.entity_of(guid)?;
        self.world
            .world()
            .get::<inf_ecs::components::ActorClass>(e)
            .map(|a| a.0)
    }

    /// The `.inf_mat` an entity's surface is bound to (P26.3b · scene v22), if
    /// any. Public because the Details projection reads it: the field is
    /// `#[reflect(ignore)]`, so the reflection walker cannot.
    pub fn material_asset_of(&self, guid: Uuid) -> Option<Uuid> {
        self.raw_get_material_asset(guid)
    }

    /// Read an entity's `Material.asset` `.inf_mat` binding GUID, if any
    /// (P26.3b · scene v22).
    pub(crate) fn raw_get_material_asset(&self, guid: Uuid) -> Option<Uuid> {
        let e = self.world.entity_of(guid)?;
        self.world
            .world()
            .get::<inf_ecs::components::Material>(e)
            .and_then(|m| m.asset)
    }

    /// Set (`Some`) or clear (`None`) an entity's `Material.asset` binding
    /// (P26.3b · scene v22).
    ///
    /// **Never inserts a `Material`.** An entity with no material has no surface
    /// to bind, and conjuring one here would give apply-material a second,
    /// silent behaviour on exactly the targets `edit_apply_material` skips.
    pub(crate) fn raw_set_material_asset(&mut self, guid: Uuid, asset: Option<Uuid>) {
        let Some(e) = self.world.entity_of(guid) else {
            return;
        };
        let Some(mut m) = self
            .world
            .world()
            .get::<inf_ecs::components::Material>(e)
            .copied()
        else {
            return;
        };
        if m.asset == asset {
            return;
        }
        m.asset = asset;
        self.world.world_mut().entity_mut(e).insert(m);
        self.world.mark_dirty();
        self.touch();
    }

    // ── the asset links the reflection walker cannot see (Wave E) ────────
    //
    // `MeshRef::asset`, `SkeletalMesh::{mesh, skeleton}` and `ActorClass` are
    // all `#[reflect(ignore)]` (or unreflected), which is correct — they are
    // identity links assigned by drag-drop, not editable numbers. The cost was
    // that the frontend had **no way at all** to learn which mesh, rig or
    // blueprint a selected entity carries, so "open this object in the Model
    // Editor" had nothing to open. These four accessors are the fix, and they
    // are exact copies of `material_asset_of`'s shape and justification.

    /// The `.inf_mesh` an entity's [`MeshRef`] is bound to, if any.
    ///
    /// Public for the same reason `material_asset_of` is: the field is
    /// `#[reflect(ignore)]`, so the reflection walker cannot reach it.
    pub fn mesh_asset_of(&self, guid: Uuid) -> Option<Uuid> {
        let e = self.world.entity_of(guid)?;
        self.world.world().get::<MeshRef>(e).and_then(|m| m.asset)
    }

    /// Read an entity's `PcgVolume.graph` — which `.inf_pcg` document scatters
    /// into it (wave EDIT1, clause 3).
    ///
    /// The field is `#[reflect(ignore)]`, so before this the Details panel
    /// showed a settlement block's Extent, Seed and Draw Distance and no way at
    /// all to tell which of the island's fourteen zone documents grew the
    /// buildings standing in it. The same escape hatch `Material.asset` and
    /// `MeshRef.asset` use, for the same reason.
    pub fn pcg_graph_of(&self, guid: Uuid) -> Option<Uuid> {
        let e = self.world.entity_of(guid)?;
        self.world
            .world()
            .get::<inf_ecs::components::PcgVolume>(e)
            .and_then(|v| v.graph)
    }

    /// How many scattered instances and structural solids a `PcgVolume`
    /// currently holds — the Details panel's read-only "has this been evaluated"
    /// row (wave EDIT1, clause 3). `None` when the entity carries no volume.
    pub fn pcg_population_of(&self, guid: Uuid) -> Option<(usize, usize)> {
        let e = self.world.entity_of(guid)?;
        self.world
            .world()
            .get::<inf_ecs::components::PcgVolume>(e)
            .map(|v| (v.evaluated.len(), v.structures.len()))
    }

    /// The built-in primitive an entity's [`MeshRef`] draws, if it has one.
    ///
    /// `Some` even when [`Self::mesh_asset_of`] is `Some` — the component
    /// carries both, and the primitive is what the placeholder path draws.
    pub fn primitive_of(&self, guid: Uuid) -> Option<Primitive> {
        let e = self.world.entity_of(guid)?;
        self.world.world().get::<MeshRef>(e).map(|m| m.primitive)
    }

    /// An entity's [`SkeletalMesh`] pair — `(mesh, skeleton)` — if it has one.
    ///
    /// Returns `Some((None, None))` for an entity that carries the component
    /// with nothing bound: "it is a skeletal mesh with no rig" and "it is not a
    /// skeletal mesh at all" are different answers and the caller needs both.
    pub fn skeletal_mesh_of(&self, guid: Uuid) -> Option<(Option<Uuid>, Option<Uuid>)> {
        let e = self.world.entity_of(guid)?;
        self.world
            .world()
            .get::<SkeletalMesh>(e)
            .map(|s| (s.mesh, s.skeleton))
    }

    /// The `.inf_act` blueprint class an entity is bound to, if any.
    ///
    /// The public face of the crate-private `raw_get_actor`, which stayed
    /// `pub(crate)` because until Wave E nothing outside the undo layer read it.
    pub fn actor_class_of(&self, guid: Uuid) -> Option<Uuid> {
        self.raw_get_actor(guid)
    }

    /// Insert (`Some`) or remove (`None`) an entity's [`ActorClass`] binding.
    pub(crate) fn raw_set_actor(&mut self, guid: Uuid, actor: Option<Uuid>) {
        if let Some(e) = self.world.entity_of(guid) {
            match actor {
                Some(a) => {
                    self.world
                        .world_mut()
                        .entity_mut(e)
                        .insert(inf_ecs::components::ActorClass(a));
                }
                None => {
                    self.world
                        .world_mut()
                        .entity_mut(e)
                        .remove::<inf_ecs::components::ActorClass>();
                }
            }
            self.world.mark_dirty();
            self.touch();
        }
    }

    /// Read an entity's [`Tilemap`] component (the chunk map the Details grid
    /// can't reach — the tile-painting panel reads it to render the grid).
    pub fn raw_get_tilemap(&self, guid: Uuid) -> Option<Tilemap> {
        let e = self.world.entity_of(guid)?;
        self.world.world().get::<Tilemap>(e).cloned()
    }

    /// Apply a batch of `(x, y, index)` tile writes to an entity's [`Tilemap`]
    /// in place (non-recording; the undo layer + [`Self::edit_paint_tiles`] drive
    /// this). No-op if the entity has no `Tilemap`.
    pub(crate) fn raw_set_tiles(&mut self, guid: Uuid, cells: &[(i32, i32, u32)]) {
        let Some(e) = self.world.entity_of(guid) else {
            return;
        };
        if let Some(mut tm) = self.world.world_mut().get_mut::<Tilemap>(e) {
            for &(x, y, idx) in cells {
                tm.set_tile(x, y, idx);
            }
        } else {
            return;
        }
        self.world.mark_dirty();
        self.touch();
    }

    // ── terrain sculpting (P10.2b) ───────────────────────────────────────
    //
    // Sculpting mirrors the tile-paint contract one layer down (height samples,
    // not tile cells): the viewport thread accumulates a mouse-down→up gesture
    // into an `inf_terrain::Stroke`, mutating the entity's live `TerrainData` per
    // dab (so the render cache re-uploads mid-drag on the version bump), then on
    // release commits the merged `HeightDelta` as ONE undo step. Because the
    // stroke already applied its dabs, `edit_commit_sculpt` only *records* — undo
    // replays `before`, redo replays `after`.

    /// Borrow an entity's [`Terrain`] paged heightfield + its world translation
    /// (for the sculpt raycast / brush-ring overlay). `None` if the entity has no
    /// `Terrain`.
    pub fn terrain_data_and_origin(&self, guid: Uuid) -> Option<(&TerrainData, DVec3)> {
        let e = self.world.entity_of(guid)?;
        let w = self.world.world();
        let data = &w.get::<Terrain>(e)?.data;
        let translation = w
            .get::<GlobalTransform>(e)
            .map(|g| g.translation())
            .unwrap_or(DVec3::ZERO);
        Some((data, translation))
    }

    // ── streamed-terrain editing (P16.4b) ────────────────────────────────
    //
    // An asset-backed terrain's `Terrain.data` is its **editable working set**:
    // tiles page into it on demand from the `.inf_terrain`, brushes sculpt them
    // exactly as they sculpt an inline terrain's, and the dirty set is what a
    // save writes back. See `crate::terrain_edit` for the whole design note —
    // these are only the document's half of the seam.

    /// The `.inf_terrain` asset `guid`'s terrain streams from, or `None` for an
    /// inline terrain (or a non-terrain entity).
    pub fn terrain_asset_of(&self, guid: Uuid) -> Option<Uuid> {
        let e = self.world.entity_of(guid)?;
        self.world.world().get::<Terrain>(e)?.asset
    }

    /// Every asset-backed terrain entity in the document, in creation order.
    pub fn streamed_terrain_entities(&self) -> Vec<Uuid> {
        self.order
            .iter()
            .copied()
            .filter(|&g| self.terrain_asset_of(g).is_some())
            .collect()
    }

    /// The tiles of `guid`'s terrain awaiting write-back to its `.inf_terrain`.
    pub fn terrain_dirty_tiles(&self, guid: Uuid) -> Vec<inf_terrain::TileKey> {
        let Some(e) = self.world.entity_of(guid) else {
            return Vec::new();
        };
        match self.world.world().get::<Terrain>(e) {
            Some(t) => t.data.dirty_tiles(),
            None => Vec::new(),
        }
    }

    /// Whether **any** streamed terrain carries unsaved edits — what flips the
    /// viewport's status chip and what the crash-recovery note reports.
    pub fn has_unsaved_terrain_edits(&self) -> bool {
        self.streamed_terrain_entities()
            .into_iter()
            .any(|g| !self.terrain_dirty_tiles(g).is_empty())
    }

    /// Run `f` against `guid`'s live `TerrainData` — the raw residency door the
    /// streamer pages a brush footprint through.
    ///
    /// Deliberately **non-touching**: `f` is a residency operation, not an edit
    /// (the brush that follows does its own touching), so this neither dirties the
    /// document nor bumps its version. `None` when the entity has no `Terrain`.
    pub(crate) fn with_terrain_data_mut<R>(
        &mut self,
        guid: Uuid,
        f: impl FnOnce(&mut TerrainData) -> R,
    ) -> Option<R> {
        let e = self.world.entity_of(guid)?;
        let mut terrain = self.world.world_mut().get_mut::<Terrain>(e)?;
        Some(f(&mut terrain.data))
    }

    /// Clear the write-back marks of the tiles a save actually wrote, returning
    /// how many were cleared.
    ///
    /// `written` is `(key, stamp-at-staging)`. A mark is cleared **only** when the
    /// tile still carries that stamp: the rewrite ran with the document unlocked,
    /// so anything sculpted during it is newer than the bytes that reached disk
    /// and must stay dirty for the next save
    /// ([`TerrainData::clear_dirty_if_unchanged`](inf_terrain::TerrainData::clear_dirty_if_unchanged)).
    ///
    /// Bumps the version **without** dirtying (like the Simulate loop), so the
    /// viewport re-reads the "unsaved terrain edits" state on the next frame while
    /// the freshly saved document stays clean.
    pub fn terrain_mark_written_back(
        &mut self,
        guid: Uuid,
        written: &[(inf_terrain::TileKey, u64)],
    ) -> usize {
        let Some(e) = self.world.entity_of(guid) else {
            return 0;
        };
        let mut cleared = 0;
        if let Some(mut terrain) = self.world.world_mut().get_mut::<Terrain>(e) {
            for &(key, stamp) in written {
                if terrain.data.clear_dirty_if_unchanged(key, stamp) {
                    cleared += 1;
                }
            }
        }
        if cleared > 0 {
            self.bump_version_for_runtime();
        }
        cleared
    }

    /// Apply one brush dab into an ongoing `stroke`, mutating the entity's live
    /// `TerrainData`, and bump the version so the render terrain cache re-uploads
    /// the dirtied tiles next frame (live sculpt feedback). No-op without a
    /// `Terrain`. Non-recording — the merged delta is recorded at commit.
    pub fn sculpt_apply_dab(
        &mut self,
        guid: Uuid,
        stroke: &mut Stroke,
        op: BrushOp,
        params: BrushParams,
    ) {
        let Some(e) = self.world.entity_of(guid) else {
            return;
        };
        if let Some(mut terrain) = self.world.world_mut().get_mut::<Terrain>(e) {
            stroke.add_dab(&mut terrain.data, op, params);
        } else {
            return;
        }
        self.world.mark_dirty();
        self.touch();
    }

    /// Finish a sculpt `stroke` and record it as one undo step. The stroke's dabs
    /// already mutated the terrain (via [`Self::sculpt_apply_dab`]), so this only
    /// finalizes the merged [`HeightDelta`] and pushes the command — an empty
    /// stroke records nothing. Returns whether an undo entry was recorded.
    pub fn edit_commit_sculpt(&mut self, guid: Uuid, stroke: Stroke) -> bool {
        let Some(e) = self.world.entity_of(guid) else {
            return false;
        };
        let Some(terrain) = self.world.world().get::<Terrain>(e) else {
            return false;
        };
        let delta = stroke.finish(&terrain.data);
        if delta.is_empty() {
            return false;
        }
        self.history.record(
            "Sculpt Terrain",
            EditCommand::SculptTerrain {
                guid,
                delta: Box::new(delta),
            },
        );
        true
    }

    /// Redo a sculpt stroke: replay its `after` height samples (recreating any
    /// tiles it authored). Non-recording; the undo layer drives it.
    pub(crate) fn raw_apply_terrain_delta(&mut self, guid: Uuid, delta: &HeightDelta) {
        let Some(e) = self.world.entity_of(guid) else {
            return;
        };
        if let Some(mut terrain) = self.world.world_mut().get_mut::<Terrain>(e) {
            terrain.data.apply_delta(delta);
        } else {
            return;
        }
        self.world.mark_dirty();
        self.touch();
    }

    /// Undo a sculpt stroke: replay its `before` height samples and drop the
    /// tiles it authored from nothing, returning the terrain byte-identical to
    /// before the stroke. Non-recording; the undo layer drives it.
    pub(crate) fn raw_revert_terrain_delta(&mut self, guid: Uuid, delta: &HeightDelta) {
        let Some(e) = self.world.entity_of(guid) else {
            return;
        };
        if let Some(mut terrain) = self.world.world_mut().get_mut::<Terrain>(e) {
            terrain.data.revert_delta(delta);
        } else {
            return;
        }
        self.world.mark_dirty();
        self.touch();
    }

    // ── terrain splat painting (P10.4) ───────────────────────────────────
    //
    // The exact twin of the sculpt seam above, one layer over: the viewport
    // thread accumulates a paint gesture into an `inf_terrain::SplatStroke`,
    // mutating the entity's live weight buffers per dab (so the render weight
    // texture re-uploads mid-drag on the version bump), then commits the merged
    // `SplatDelta` as ONE undo step. `PaintSplat` is a separate `EditCommand`
    // from `SculptTerrain` — height and weight deltas are genuinely different
    // payloads, and keeping them apart leaves the existing sculpt-undo path
    // byte-stable rather than folding both into one enum.

    /// The linear albedo of a terrain entity's splat `layer` (`0..=3`), for the
    /// paint brush-ring colour. `None` if the entity has no `Terrain`.
    pub fn terrain_layer_albedo(&self, guid: Uuid, layer: u8) -> Option<[f32; 4]> {
        let e = self.world.entity_of(guid)?;
        let terrain = self.world.world().get::<Terrain>(e)?;
        Some(
            terrain.layers[(layer as usize) % terrain.layers.len()]
                .albedo
                .to_array(),
        )
    }

    /// Apply one paint dab into an ongoing `stroke`, mutating the entity's live
    /// weight buffers, and bump the version so the render weight texture
    /// re-uploads next frame (live paint feedback). No-op without a `Terrain`.
    /// Non-recording — the merged delta is recorded at commit.
    pub fn paint_apply_dab(&mut self, guid: Uuid, stroke: &mut SplatStroke, params: BrushParams) {
        let Some(e) = self.world.entity_of(guid) else {
            return;
        };
        if let Some(mut terrain) = self.world.world_mut().get_mut::<Terrain>(e) {
            stroke.add_dab(&mut terrain.data, params);
        } else {
            return;
        }
        self.world.mark_dirty();
        self.touch();
    }

    /// Finish a paint `stroke` and record it as one undo step. The dabs already
    /// mutated the weights (via [`Self::paint_apply_dab`]); this finalizes the
    /// merged [`SplatDelta`] and pushes the command. An empty stroke records
    /// nothing. Returns whether an undo entry was recorded.
    pub fn edit_commit_paint(&mut self, guid: Uuid, stroke: SplatStroke) -> bool {
        let Some(e) = self.world.entity_of(guid) else {
            return false;
        };
        let Some(terrain) = self.world.world().get::<Terrain>(e) else {
            return false;
        };
        let delta = stroke.finish(&terrain.data);
        if delta.is_empty() {
            return false;
        }
        self.history.record(
            "Paint Splat",
            EditCommand::PaintSplat {
                guid,
                delta: Box::new(delta),
            },
        );
        true
    }

    /// Redo a paint stroke: replay its `after` weights. Non-recording.
    pub(crate) fn raw_apply_splat_delta(&mut self, guid: Uuid, delta: &SplatDelta) {
        let Some(e) = self.world.entity_of(guid) else {
            return;
        };
        if let Some(mut terrain) = self.world.world_mut().get_mut::<Terrain>(e) {
            terrain.data.apply_splat_delta(delta);
        } else {
            return;
        }
        self.world.mark_dirty();
        self.touch();
    }

    /// Undo a paint stroke: replay its `before` weights and drop any weight
    /// buffers the stroke materialized, returning the terrain byte-identical to
    /// before the stroke. Non-recording.
    pub(crate) fn raw_revert_splat_delta(&mut self, guid: Uuid, delta: &SplatDelta) {
        let Some(e) = self.world.entity_of(guid) else {
            return;
        };
        if let Some(mut terrain) = self.world.world_mut().get_mut::<Terrain>(e) {
            terrain.data.revert_splat_delta(delta);
        } else {
            return;
        }
        self.world.mark_dirty();
        self.touch();
    }

    // ── terrain biome painting (P19.2) ───────────────────────────────────
    //
    // The splat seam again, one layer over — and deliberately a *separate*
    // `EditCommand` for the same reason `PaintSplat` is separate from
    // `SculptTerrain`: the payloads are genuinely different (per-sample `u8` ids
    // versus `[u8; 4]` weights), and keeping them apart leaves both existing undo
    // paths byte-stable instead of folding an always-empty buffer into every
    // stroke on the stack.

    /// The GUID of the `.inf_biomes` set a terrain entity's ids name, or `None`
    /// when no set is bound (and every sample therefore reads *unassigned*).
    /// `None` too if the entity has no `Terrain`.
    pub fn terrain_biome_set(&self, guid: Uuid) -> Option<Uuid> {
        let e = self.world.entity_of(guid)?;
        self.world.world().get::<Terrain>(e)?.biome_set
    }

    /// Bind (or clear) the `.inf_biomes` set a terrain's biome ids name, as one
    /// undo step.
    ///
    /// Recorded as a whole-component swap rather than a reflected property write
    /// because `Terrain.biome_set` is `#[reflect(ignore)]` — an asset reference,
    /// not a Details-grid scalar — exactly like `MeshRef.asset`. Returns whether
    /// anything changed (rebinding to the same set records nothing).
    pub fn edit_set_terrain_biome_set(&mut self, guid: Uuid, set: Option<Uuid>) -> bool {
        if self.terrain_biome_set(guid) == set && self.has_terrain(guid) {
            return false;
        }
        let Some(before) = crate::scene::serialize::record_of(self, guid) else {
            return false;
        };
        let Some(e) = self.world.entity_of(guid) else {
            return false;
        };
        {
            let Some(mut terrain) = self.world.world_mut().get_mut::<Terrain>(e) else {
                return false;
            };
            terrain.biome_set = set;
        }
        self.world.mark_dirty();
        self.touch();
        let Some(after) = crate::scene::serialize::record_of(self, guid) else {
            return false;
        };
        self.history.record(
            "Set Biome Set",
            EditCommand::SwapComponents {
                guid,
                before: Box::new(before),
                after: Box::new(after),
            },
        );
        true
    }

    /// `true` when `guid` names an entity carrying a `Terrain` component.
    fn has_terrain(&self, guid: Uuid) -> bool {
        self.world
            .entity_of(guid)
            .is_some_and(|e| self.world.world().get::<Terrain>(e).is_some())
    }

    /// Apply one biome dab into an ongoing `stroke`, mutating the entity's live
    /// biome-id buffers, and bump the version so the render biome texture
    /// re-uploads next frame (live paint feedback). No-op without a `Terrain`.
    /// Non-recording — the merged delta is recorded at commit.
    pub fn biome_apply_dab(&mut self, guid: Uuid, stroke: &mut BiomeStroke, params: BrushParams) {
        let Some(e) = self.world.entity_of(guid) else {
            return;
        };
        if let Some(mut terrain) = self.world.world_mut().get_mut::<Terrain>(e) {
            stroke.add_dab(&mut terrain.data, params);
        } else {
            return;
        }
        self.world.mark_dirty();
        self.touch();
    }

    /// Finish a biome `stroke` and record it as one undo step. An empty stroke
    /// records nothing. Returns whether an undo entry was recorded.
    pub fn edit_commit_biome(&mut self, guid: Uuid, stroke: BiomeStroke) -> bool {
        let Some(e) = self.world.entity_of(guid) else {
            return false;
        };
        // The label names what the stroke did, so the undo menu distinguishes
        // erasing from assigning without the user having to remember.
        let label = if stroke.is_eraser() {
            "Erase Biome"
        } else {
            "Paint Biome"
        };
        let Some(terrain) = self.world.world().get::<Terrain>(e) else {
            return false;
        };
        let delta = stroke.finish(&terrain.data);
        if delta.is_empty() {
            return false;
        }
        self.history.record(
            label,
            EditCommand::PaintBiome {
                guid,
                delta: Box::new(delta),
            },
        );
        true
    }

    /// Fill polygons of biome ids into a terrain and record the lot as **one**
    /// undo step (IB-5).
    ///
    /// `fill` is a [`inf_terrain::BiomeFill`] the caller has already driven —
    /// a land-cover import fills a hundred polygons at eight classes into one,
    /// because a hundred undo steps is not an import an author can take back.
    /// Records through the same `EditCommand::PaintBiome` a brush stroke does,
    /// so undo, redo and the render re-upload are one path and not two.
    pub fn edit_commit_biome_fill(
        &mut self,
        guid: Uuid,
        label: &str,
        fill: inf_terrain::BiomeFill,
    ) -> bool {
        let Some(e) = self.world.entity_of(guid) else {
            return false;
        };
        let Some(terrain) = self.world.world().get::<Terrain>(e) else {
            return false;
        };
        let delta = fill.finish(&terrain.data);
        if delta.is_empty() {
            return false;
        }
        self.history.record(
            label,
            EditCommand::PaintBiome {
                guid,
                delta: Box::new(delta),
            },
        );
        self.world.mark_dirty();
        self.touch();
        true
    }

    /// Redo a biome stroke: replay its `after` ids. Non-recording.
    pub(crate) fn raw_apply_biome_delta(&mut self, guid: Uuid, delta: &BiomeDelta) {
        let Some(e) = self.world.entity_of(guid) else {
            return;
        };
        if let Some(mut terrain) = self.world.world_mut().get_mut::<Terrain>(e) {
            terrain.data.apply_biome_delta(delta);
        } else {
            return;
        }
        self.world.mark_dirty();
        self.touch();
    }

    /// Undo a biome stroke: replay its `before` ids and drop any id buffers the
    /// stroke materialized, returning the terrain byte-identical to before the
    /// stroke. Non-recording.
    pub(crate) fn raw_revert_biome_delta(&mut self, guid: Uuid, delta: &BiomeDelta) {
        let Some(e) = self.world.entity_of(guid) else {
            return;
        };
        if let Some(mut terrain) = self.world.world_mut().get_mut::<Terrain>(e) {
            terrain.data.revert_biome_delta(delta);
        } else {
            return;
        }
        self.world.mark_dirty();
        self.touch();
    }

    // ── erosion data maps (P19.1) ────────────────────────────────────────

    /// Redo an erosion bake's **data-map** half: replay its `after` texels,
    /// materializing any tile that needs it. Non-recording.
    pub(crate) fn raw_apply_data_map_delta(&mut self, guid: Uuid, delta: &DataMapDelta) {
        let Some(e) = self.world.entity_of(guid) else {
            return;
        };
        if let Some(mut terrain) = self.world.world_mut().get_mut::<Terrain>(e) {
            terrain.data.apply_data_map_delta(delta);
        } else {
            return;
        }
        self.world.mark_dirty();
        self.touch();
    }

    /// Undo an erosion bake's data-map half: replay its `before` texels and drop
    /// any map buffers the bake materialized, returning the terrain
    /// byte-identical to before the bake. Non-recording.
    pub(crate) fn raw_revert_data_map_delta(&mut self, guid: Uuid, delta: &DataMapDelta) {
        let Some(e) = self.world.entity_of(guid) else {
            return;
        };
        if let Some(mut terrain) = self.world.world_mut().get_mut::<Terrain>(e) {
            terrain.data.revert_data_map_delta(delta);
        } else {
            return;
        }
        self.world.mark_dirty();
        self.touch();
    }

    // ── foliage painting (E-P6) ──────────────────────────────────────────
    //
    // The foliage brush mirrors the sculpt seam one component over: the viewport
    // thread accumulates a mouse-down→up scatter gesture, live-mutating the target
    // `Foliage` component's instance list per tick (so the render projection
    // refreshes mid-stroke on the version bump), then commits ONE `PaintFoliage`
    // undo step. A stroke either APPENDS (`added`) or ERASES (`removed`) — never
    // both — so apply/revert stay exact inverses.

    /// Whether `guid` names a `Foliage` entity (the brush's target-resolution rule
    /// paints into the selected foliage entity, else auto-creates one).
    pub fn has_foliage(&self, guid: Uuid) -> bool {
        self.world
            .entity_of(guid)
            .map(|e| self.world.world().get::<Foliage>(e).is_some())
            .unwrap_or(false)
    }

    /// The foliage entity's world translation (instances are entity-local, so the
    /// brush converts world hit points through this). `None` if no `Foliage`.
    pub fn foliage_origin(&self, guid: Uuid) -> Option<DVec3> {
        let e = self.world.entity_of(guid)?;
        let w = self.world.world();
        w.get::<Foliage>(e)?;
        Some(
            w.get::<GlobalTransform>(e)
                .map(|g| g.translation())
                .unwrap_or(DVec3::ZERO),
        )
    }

    /// A clone of the entity's current foliage instances (the brush snapshots this
    /// at stroke start for min-spacing + erase). `None` if no `Foliage`.
    pub fn foliage_instances(&self, guid: Uuid) -> Option<Vec<FoliageInstance>> {
        let e = self.world.entity_of(guid)?;
        Some(self.world.world().get::<Foliage>(e)?.instances.clone())
    }

    /// Append instances to the entity's `Foliage` in place (non-recording; the
    /// brush drives it per tick, the merged set is recorded at commit). Bumps the
    /// version so the render projection refreshes. No-op without a `Foliage`.
    pub fn foliage_append(&mut self, guid: Uuid, instances: &[FoliageInstance]) {
        let Some(e) = self.world.entity_of(guid) else {
            return;
        };
        if let Some(mut fol) = self.world.world_mut().get_mut::<Foliage>(e) {
            fol.instances.extend_from_slice(instances);
        } else {
            return;
        }
        self.world.mark_dirty();
        self.touch();
    }

    /// Replace the entity's foliage instance list in place (non-recording; the
    /// erase brush rebuilds it per tick). No-op without a `Foliage`.
    pub fn foliage_set_instances(&mut self, guid: Uuid, instances: Vec<FoliageInstance>) {
        let Some(e) = self.world.entity_of(guid) else {
            return;
        };
        if let Some(mut fol) = self.world.world_mut().get_mut::<Foliage>(e) {
            fol.instances = instances;
        } else {
            return;
        }
        self.world.mark_dirty();
        self.touch();
    }

    /// Finish a foliage stroke and record it as ONE `PaintFoliage` undo step. The
    /// live instances already changed (via [`Self::foliage_append`] /
    /// [`Self::foliage_set_instances`]), so this only records — `apply` replays the
    /// net add/remove, `revert` inverts it. An empty stroke records nothing.
    /// Returns whether an undo entry was recorded.
    pub fn edit_commit_foliage(
        &mut self,
        guid: Uuid,
        added: Vec<FoliageInstance>,
        removed: Vec<(usize, FoliageInstance)>,
    ) -> bool {
        if added.is_empty() && removed.is_empty() {
            return false;
        }
        self.history.record(
            "Paint Foliage",
            EditCommand::PaintFoliage {
                guid,
                added,
                removed,
            },
        );
        self.touch();
        true
    }

    /// Redo a foliage stroke: remove the `removed` indices (descending, so earlier
    /// indices stay valid) then push the `added`. Non-recording; the undo layer
    /// drives it. The live instances already hold this state when first committed,
    /// so this only matters on an actual redo.
    pub(crate) fn raw_apply_foliage(
        &mut self,
        guid: Uuid,
        added: &[FoliageInstance],
        removed: &[(usize, FoliageInstance)],
    ) {
        let Some(e) = self.world.entity_of(guid) else {
            return;
        };
        if let Some(mut fol) = self.world.world_mut().get_mut::<Foliage>(e) {
            let mut idx: Vec<usize> = removed.iter().map(|(i, _)| *i).collect();
            idx.sort_unstable();
            for &i in idx.iter().rev() {
                if i < fol.instances.len() {
                    fol.instances.remove(i);
                }
            }
            fol.instances.extend_from_slice(added);
        } else {
            return;
        }
        self.world.mark_dirty();
        self.touch();
    }

    /// Undo a foliage stroke: pop the `added` (they were appended last) then re-
    /// insert the `removed` at their original indices (ascending), restoring the
    /// pre-stroke instance list exactly. Non-recording; the undo layer drives it.
    pub(crate) fn raw_revert_foliage(
        &mut self,
        guid: Uuid,
        added: &[FoliageInstance],
        removed: &[(usize, FoliageInstance)],
    ) {
        let Some(e) = self.world.entity_of(guid) else {
            return;
        };
        if let Some(mut fol) = self.world.world_mut().get_mut::<Foliage>(e) {
            for _ in 0..added.len() {
                fol.instances.pop();
            }
            let mut pairs: Vec<(usize, FoliageInstance)> = removed.to_vec();
            pairs.sort_by_key(|(i, _)| *i);
            for (i, inst) in pairs {
                let at = i.min(fol.instances.len());
                fol.instances.insert(at, inst);
            }
        } else {
            return;
        }
        self.world.mark_dirty();
        self.touch();
    }

    // ── terrain erosion bake (P10.3b) ────────────────────────────────────
    //
    // Erosion runs the GPU compute pipeline (or the CPU reference as the
    // no-adapter fallback) over an extracted `HeightRegion`, writes the result
    // back through the same seam as sculpt, and commits it as ONE
    // `SculptTerrain` height delta — so an erosion bake is undoable for free and
    // the viewport re-uploads on the version bump. ERODED TERRAIN IS DATA: the
    // delta is stored in the level; only the *bake action* varies by GPU adapter
    // (GPU f32 vs the partly-f64 CPU reference), scene determinism is unaffected.

    /// The world-XZ AABB `[min, max]` covering the entity's authored terrain
    /// tiles, in terrain-local coordinates (the space [`edit_erode_region`] and
    /// `extract_region` expect). `None` when the entity has no `Terrain` or the
    /// terrain is empty.
    ///
    /// [`edit_erode_region`]: Self::edit_erode_region
    pub fn terrain_bounds(&self, guid: Uuid) -> Option<(DVec2, DVec2)> {
        let e = self.world.entity_of(guid)?;
        let data = &self.world.world().get::<Terrain>(e)?.data;
        if data.is_empty() {
            return None;
        }
        let span = data.tile_span();
        let mut min = DVec2::splat(f64::INFINITY);
        let mut max = DVec2::splat(f64::NEG_INFINITY);
        for (&(tx, tz), _) in data.tiles() {
            let o = DVec2::new(tx as f64 * span, tz as f64 * span);
            min = min.min(o);
            max = max.max(o + DVec2::splat(span));
        }
        Some((min, max))
    }

    /// Run an erosion bake over the terrain-local world AABB `[min, max]`
    /// (expanded by `margin` samples so open-boundary draining happens outside
    /// the area of interest) on `guid`'s terrain: `run` transforms the extracted
    /// [`HeightRegion`](inf_terrain::HeightRegion) in place (the GPU pipeline, or
    /// the CPU reference on the no-adapter fallback), and the result is committed
    /// as ONE undoable `SculptTerrain` delta with a version bump. Returns
    /// per-bake accounting (cells changed + net mass change) derived from the
    /// committed delta — adapter-independent, unlike GPU float order. `None` when
    /// the entity has no `Terrain`.
    pub fn edit_erode_region<F>(
        &mut self,
        guid: Uuid,
        min: DVec2,
        max: DVec2,
        margin: u32,
        run: F,
    ) -> Option<ErodeReport>
    where
        F: FnOnce(&mut inf_terrain::HeightRegion),
    {
        let e = self.world.entity_of(guid)?;
        let (mps, delta, maps) = {
            let mut terrain = self.world.world_mut().get_mut::<Terrain>(e)?;
            let mps = terrain.data.meters_per_sample();
            let (delta, maps) = terrain.data.edit_region_with_maps(min, max, margin, run);
            (mps, delta, maps)
        };
        // Cells actually changed + net terrain-volume change, from the delta.
        let area = mps * mps;
        let mut cells_changed = 0usize;
        let mut mass_delta = 0.0f64;
        for p in &delta.patches {
            for k in 0..p.before.len() {
                let d = p.after[k] - p.before[k];
                if d != 0.0 {
                    cells_changed += 1;
                    mass_delta += d as f64;
                }
            }
        }
        mass_delta *= area;
        let map_cells_changed = maps
            .patches
            .iter()
            .map(|p| {
                (0..p.before.len())
                    .filter(|&k| p.after[k] != p.before[k])
                    .count()
            })
            .sum();
        // ONE undo step, two layers (P19.1). The height and data-map deltas are
        // separate records — a sculpt brush only ever produces the first — but
        // they are recorded inside a single transaction, so one Ctrl+Z restores
        // heights AND maps, byte-identically, and neither can be undone without
        // the other.
        //
        // THE ORDER IS LOAD-BEARING, and this is the only place it is chosen.
        // `SceneDoc::undo` reverts a transaction's commands in **reverse** order,
        // so recording heights-then-maps means undo runs maps-then-heights. That
        // is the safe direction: `revert_delta` may *remove* tiles the stroke
        // authored from nothing, and `revert_data_map_delta` skips a patch whose
        // tile is gone — so a map revert after a height revert could silently
        // lose writes. Reversed, the maps are restored while every tile is still
        // present. It is benign either way for erosion specifically, because
        // `HeightRegion::write_back` never creates a tile (an erosion bake's
        // `created_tiles` is always empty), but the ordering must not depend on
        // that: swap these two `record` calls and the invariant is gone.
        if !delta.is_empty() || !maps.is_empty() {
            self.history.begin("Erode Terrain");
            if !delta.is_empty() {
                self.history.record(
                    "Erode Terrain",
                    EditCommand::SculptTerrain {
                        guid,
                        delta: Box::new(delta),
                    },
                );
            }
            if !maps.is_empty() {
                self.history.record(
                    "Erode Terrain",
                    EditCommand::WriteDataMaps {
                        guid,
                        delta: Box::new(maps),
                    },
                );
            }
            self.history.commit();
            self.world.mark_dirty();
            self.touch();
        }
        Some(ErodeReport {
            cells_changed,
            map_cells_changed,
            mass_delta,
        })
    }

    /// Recreate an entity from a serialized record at order slot `at`.
    pub(crate) fn raw_spawn_record(&mut self, rec: &EntityRecord, at: usize) {
        let e = self.spawn_bare(rec.guid, &rec.name, rec.parent);
        // `spawn_bare` appended the guid; move it to its original slot.
        if let Some(pos) = self.order.iter().position(|g| *g == rec.guid) {
            let g = self.order.remove(pos);
            let at = at.min(self.order.len());
            self.order.insert(at, g);
        }
        self.apply_record_components(e, rec);
        self.touch();
    }

    /// Insert every component carried by a serialized [`EntityRecord`] onto `e`.
    /// Shared by [`Self::raw_spawn_record`] (delete→undo) and the duplicate /
    /// paste spawn path — both re-materialize an entity from a record. Does not
    /// dirty the document; the caller `touch()`es.
    fn apply_record_components(&mut self, e: Entity, rec: &EntityRecord) {
        let w = self.world.world_mut();
        w.entity_mut(e).insert((
            rec.transform,
            Visibility {
                visible: rec.visible,
            },
        ));
        if let Some(m) = rec.mesh {
            w.entity_mut(e).insert(m);
        }
        if let Some(m) = rec.material {
            w.entity_mut(e).insert(m);
        }
        if let Some(l) = rec.light {
            w.entity_mut(e).insert(l);
        }
        if let Some(c) = rec.camera {
            w.entity_mut(e).insert(c);
        }
        if let Some(s) = &rec.sprite {
            w.entity_mut(e).insert(s.clone());
        }
        if let Some(t) = &rec.tilemap {
            w.entity_mut(e).insert(t.clone());
        }
        if let Some(n) = &rec.nine_slice {
            w.entity_mut(e).insert(n.clone());
        }
        if let Some(t) = &rec.text2d {
            w.entity_mut(e).insert(t.clone());
        }
        if let Some(l) = &rec.light_2d {
            w.entity_mut(e).insert(*l);
        }
        if let Some(c) = &rec.rigid_body_2d {
            w.entity_mut(e).insert(*c);
        }
        if let Some(c) = &rec.collider_2d {
            w.entity_mut(e).insert(*c);
        }
        if let Some(c) = &rec.character_controller_2d {
            w.entity_mut(e).insert(*c);
        }
        if let Some(c) = &rec.rigid_body_3d {
            w.entity_mut(e).insert(*c);
        }
        if let Some(c) = &rec.collider_3d {
            w.entity_mut(e).insert(*c);
        }
        if let Some(c) = &rec.character_controller_3d {
            w.entity_mut(e).insert(*c);
        }
        if let Some(actor) = rec.actor {
            w.entity_mut(e)
                .insert(inf_ecs::components::ActorClass(actor));
        }
        // v4 world components: a deleted-then-undone terrain / PCG volume restores
        // its full paged data + graph ref (the P10.6 fix for the "delete → undo
        // loses Terrain" gap the v3 batch documented).
        if let Some(t) = &rec.terrain {
            w.entity_mut(e).insert(t.clone());
        }
        if let Some(v) = &rec.pcg_volume {
            w.entity_mut(e).insert(v.clone());
        }
        // v5 animation / character components: a deleted-then-undone character
        // restores its full SkeletalMesh / AnimPlayer / AnimStateMachine /
        // RootMotion / AttachedTo set (mirrors the v4 terrain/PCG restore above).
        if let Some(c) = &rec.skeletal_mesh {
            w.entity_mut(e).insert(*c);
        }
        if let Some(c) = &rec.anim_player {
            w.entity_mut(e).insert(*c);
        }
        if let Some(c) = &rec.anim_state_machine {
            w.entity_mut(e).insert(*c);
        }
        if let Some(c) = &rec.root_motion {
            w.entity_mut(e).insert(*c);
        }
        if let Some(c) = &rec.attached_to {
            w.entity_mut(e).insert(c.clone());
        }
        // v6 (P12.4): a delete→undo also restores the joints/audio component set.
        if let Some(c) = &rec.joint_2d {
            w.entity_mut(e).insert(*c);
        }
        if let Some(c) = &rec.joint_3d {
            w.entity_mut(e).insert(*c);
        }
        if let Some(c) = &rec.audio_source {
            w.entity_mut(e).insert(c.clone());
        }
        if let Some(c) = &rec.audio_listener {
            w.entity_mut(e).insert(*c);
        }
    }

    fn prop_value(&self, guid: Uuid, type_path: &str, field: &str) -> Option<PropValue> {
        let comps = self.entity_props(guid);
        let c = comps.iter().find(|c| c.type_path == type_path)?;
        c.fields
            .iter()
            .find(|f| f.name == field)
            .map(|f| f.value.clone())
    }

    // ── recorded mutations (Ring 2 / gizmo call these) ───────────────────

    /// Create + record for undo. Returns the new GUID.
    pub fn edit_create(&mut self, kind: SpawnKind, name: &str, parent: Option<Uuid>) -> Uuid {
        let guid = self.create(kind, name, parent);
        let at = self.order.iter().position(|g| *g == guid).unwrap_or(0);
        if let Some(record) = crate::scene::serialize::record_of(self, guid) {
            self.history.record(
                "Create",
                EditCommand::Create {
                    at,
                    record: Box::new(record),
                },
            );
        }
        guid
    }

    /// Create an entity **bound to a mesh asset** — a dropped `.inf_mesh` (Wave E).
    ///
    /// # The gap this closes
    ///
    /// Since P4 every editor path that placed a dragged-in mesh called
    /// `edit_create(SpawnKind::Cube, …)` and stopped: the entity was a
    /// **placeholder cube named after the asset**, and *nothing in the editor
    /// ever wrote `MeshRef::asset`* — only samples, migration fixtures and
    /// tests did. The viewport has resolved that field to real geometry since
    /// P18.3 ([`crate::render_assets`]), so the placeholder was not a rendering
    /// limitation any more; it was a missing assignment. It also meant the Wave-E
    /// routing ("edit the mesh of this object") had nothing to open on any prop
    /// the user had actually placed.
    ///
    /// The `primitive` stays `Cube`: it is the fallback the renderer draws when
    /// the asset cannot be resolved, exactly as `MeshRef`'s own docs describe.
    ///
    /// One `Create` undo step — the component is attached **before** the record
    /// is snapshotted, so redo restores a prop that is still wearing its mesh
    /// (the `edit_create_streamed_terrain` pattern, for its reason).
    pub fn edit_create_mesh_asset(
        &mut self,
        name: &str,
        asset: Uuid,
        parent: Option<Uuid>,
    ) -> Uuid {
        let guid = self.create(SpawnKind::Cube, name, parent);
        if let Some(entity) = self.world.entity_of(guid) {
            let mut mesh = self
                .world
                .world()
                .get::<MeshRef>(entity)
                .copied()
                .unwrap_or_default();
            mesh.asset = Some(asset);
            self.world.world_mut().entity_mut(entity).insert(mesh);
        }
        let at = self.order.iter().position(|g| *g == guid).unwrap_or(0);
        if let Some(record) = crate::scene::serialize::record_of(self, guid) {
            self.history.record(
                "Place Mesh",
                EditCommand::Create {
                    at,
                    record: Box::new(record),
                },
            );
        }
        self.touch();
        guid
    }

    /// Create a **streamed** terrain: a `Terrain` whose heightfield lives in a
    /// `.inf_terrain` asset rather than in the document (P16.4a).
    ///
    /// The component carries the asset's own grid configuration and an **empty**
    /// `data` — that is the whole point, so a 131 km world costs the level a few
    /// hundred bytes and the editor streamer pages it from the content root.
    ///
    /// The asset is attached **before** the undo record is snapshotted, so redo
    /// restores a streamed terrain and not the starter sine-hill the plain
    /// `Add ▸ Terrain` spawns. Returns the new GUID.
    pub fn edit_create_streamed_terrain(
        &mut self,
        name: &str,
        asset: Uuid,
        tile_resolution: u32,
        meters_per_sample: f64,
    ) -> Uuid {
        let guid = self.create(SpawnKind::Terrain, name, None);
        if let Some(entity) = self.world.entity_of(guid) {
            let mut terrain = Terrain::configured(tile_resolution.max(2), meters_per_sample);
            terrain.asset = Some(asset);
            self.world.world_mut().entity_mut(entity).insert(terrain);
        }
        let at = self.order.iter().position(|g| *g == guid).unwrap_or(0);
        if let Some(record) = crate::scene::serialize::record_of(self, guid) {
            self.history.record(
                "Create Streamed Terrain",
                EditCommand::Create {
                    at,
                    record: Box::new(record),
                },
            );
        }
        self.touch();
        guid
    }

    /// Create a **character**: an entity carrying the `SkeletalMesh` +
    /// `AnimStateMachine` + `Material` trio a generated character needs (P24.5;
    /// the surface since the wave CHAR1a audit).
    ///
    /// One `Create` undo step, on the `edit_create_lake` pattern — the components
    /// are attached *before* the record is snapshotted, so undo removes a
    /// character and redo restores one that is still wearing its rig. A wizard
    /// that spawned the entity and then set the components through three separate
    /// calls would put three steps on the stack for one button.
    ///
    /// Placed at `at` (world metres). The whole point of the pair is that neither
    /// half is useful alone: a `SkeletalMesh` with no machine draws a bind pose
    /// forever, and a machine with no skeleton steps its states and poses nothing
    /// (`inf_ecs::pose`'s rule 3).
    #[allow(clippy::too_many_arguments)]
    pub fn edit_create_character(
        &mut self,
        name: &str,
        skeleton: Uuid,
        mesh: Uuid,
        machine: Uuid,
        skin: Option<CharacterSkin>,
        at: DVec3,
        controller: Option<Uuid>,
        height_m: f64,
    ) -> Uuid {
        self.edit_create_character_with_guid(
            Uuid::new_v4(),
            name,
            skeleton,
            mesh,
            machine,
            skin,
            at,
            controller,
            height_m,
        )
    }

    /// [`edit_create_character`](Self::edit_create_character), **with the entity's
    /// GUID supplied** (SK1c).
    ///
    /// # Why a generator needs this
    ///
    /// A committed level's entity GUIDs are part of its bytes. `island.rs`
    /// derives every one of its own from the island's name — `hero_guid(name)`
    /// — so the level a fresh checkout authors is the level the last one did,
    /// and so the gate can find the hero without walking components. The
    /// interactive door mints, and the island's hero used to be forty lines of
    /// hand-rolled components for exactly that reason: the one door that knows
    /// how to build a character could not be told which entity to build.
    ///
    /// **What is NOT here, and is load-bearing**: `StreamingSource` and
    /// `AlwaysLoaded`. A character is not necessarily a streaming anchor, and
    /// making this door insert one would put a partition opinion on every
    /// character the wizard spawns. The island re-inserts both after this
    /// returns, which is where that decision belongs.
    #[allow(clippy::too_many_arguments)]
    pub fn edit_create_character_with_guid(
        &mut self,
        guid: Uuid,
        name: &str,
        skeleton: Uuid,
        mesh: Uuid,
        machine: Uuid,
        // **The skin the wizard already wrote** (wave CHAR1a audit) — see
        // [`CharacterSkin`]. `None` leaves the entity with no `Material`, which
        // is what every character in this engine had until this parameter
        // existed, and is still the right answer for a caller that has no
        // surface to name.
        skin: Option<CharacterSkin>,
        at: DVec3,
        controller: Option<Uuid>,
        height_m: f64,
    ) -> Uuid {
        self.create_with_guid(guid, SpawnKind::Empty, name, None);
        if let Some(entity) = self.world.entity_of(guid) {
            // ── P29.6 ── the character is a CHARACTER now: a capsule sized from
            //    the rig it was generated with, a kinematic body, the movement
            //    component with the catalogue defaults, and the controller the
            //    wizard wrote. §13's gap list said the wizard emitted none of
            //    this; a `SkeletalMesh` and a machine alone is a puppet.
            //
            //    The capsule is derived rather than authored: half the creature's
            //    own height, less a radius that is a quarter of it, so a 1.2 m
            //    character and a 2 m one both get a capsule that fits. `at` is
            //    the FEET (which is what an author places), and the transform is
            //    the capsule CENTRE, so the two differ by exactly what
            //    `inf_ecs::movement::feet_offset_m` subtracts — the P29.6
            //    character-space ruling, applied at the one place a character is
            //    created.
            let height = if height_m.is_finite() && height_m > 0.2 {
                height_m
            } else {
                1.8
            };
            let radius = (height * 0.15).clamp(0.1, 0.5);
            let half = (height * 0.5 - radius).max(0.05);
            let mut movement = CharacterMovement {
                player_controlled: true,
                ..Default::default()
            };
            movement.stand_half_height_m = half;
            movement.crouch_half_height_m = (half * 0.5).max(0.05);
            movement.prone_half_height_m = (radius * 0.6).max(0.03);
            let mut t = Transform::IDENTITY;
            t.translation = Vec3d::new(at.x, at.y + half + radius, at.z);
            self.world.world_mut().entity_mut(entity).insert((
                SkeletalMesh {
                    mesh: Some(mesh),
                    skeleton: Some(skeleton),
                },
                AnimStateMachine {
                    sm: Some(machine),
                    ..Default::default()
                },
                RigidBody3D {
                    kind: BodyKind3D::Kinematic,
                    ..Default::default()
                },
                Collider3D {
                    shape_kind: ColliderShape3DKind::Capsule,
                    half_extents: Vec3d::new(radius, half, radius),
                    radius,
                    ..Default::default()
                },
                CharacterController3D::default(),
                movement,
                t,
            ));
            // ── THE SKIN (wave CHAR1a audit) ──
            //
            // **Nothing in this engine bound a character's skin.** The New
            // Character wizard writes `<Name> Skin.inf_mat` beside the body and
            // names it as the mesh's dependency, and `inf-import
            // --rebind-character` fills it with the imported body's own albedo,
            // normal and ORM — and then both hosts read `Material` → `None`,
            // hand `vt_set_for` a `None`, and draw the renderer's neutral 0.8
            // grey. Read off the running editor with `scene_details`, not
            // inferred: the island hero carried `Transform, Visibility,
            // SkeletalMesh, AnimStateMachine, RigidBody3D, Collider3D` and no
            // `Material`. That is why every character frame this campaign has
            // taken shows a grey or white body.
            //
            // The component is inserted here, in the same pre-record window as
            // the rig, for the same reason the rig is: undo removes a character
            // and redo restores one that is still wearing its skin.
            if let Some(skin) = skin {
                self.world.world_mut().entity_mut(entity).insert(Material {
                    base_color: Color::new(
                        skin.base_color[0],
                        skin.base_color[1],
                        skin.base_color[2],
                        skin.base_color[3],
                    ),
                    metallic: skin.metallic,
                    roughness: skin.roughness,
                    asset: Some(skin.asset),
                    ..Default::default()
                });
            }
            if let Some(actor) = controller {
                self.world
                    .world_mut()
                    .entity_mut(entity)
                    .insert(ActorClass(actor));
            }
        }
        self.record_create(guid, "Create Character");
        guid
    }

    // ── P20.4 hydrology authoring ────────────────────────────────────────
    //
    // Three recorded mutations, and between them they are the whole water tool.
    // They exist here rather than in the Ring-2 command layer for the reason
    // every other `edit_*` does: the undo record has to be taken around the
    // *complete* change, and a caller that spawned an entity and then inserted
    // components through separate calls would put three steps on the stack for
    // one click.

    /// Create a **lake**: an entity carrying a `WaterBody` at `center`, sized
    /// `half_extent` (metres, XZ) with its surface at `level_m` (P20.4).
    ///
    /// One `Create` undo step, on the `edit_create_streamed_terrain` pattern —
    /// the component is attached *before* the record is snapshotted, so redo
    /// restores a lake rather than the bare entity the spawn kind alone makes.
    ///
    /// The entity's transform is the lake's **centre**, which is the convention
    /// the projector reads (`center = affine.translation.xz`); `level_m` is an
    /// absolute world elevation and deliberately not derived from the transform's
    /// `y`, because a sea level that moved when you nudged the entity would be a
    /// trap (`WaterBody::level_m`'s own doc says so).
    pub fn edit_create_lake(
        &mut self,
        name: &str,
        center: DVec3,
        half_extent: Vec2d,
        level_m: f64,
    ) -> Uuid {
        let guid = self.create(SpawnKind::Empty, name, None);
        if let Some(entity) = self.world.entity_of(guid) {
            let mut t = Transform::IDENTITY;
            t.translation = Vec3d::new(center.x, center.y, center.z);
            self.world.world_mut().entity_mut(entity).insert((
                WaterBody::lake(
                    level_m,
                    Vec2d::new(half_extent.x.abs().max(0.0), half_extent.y.abs().max(0.0)),
                ),
                t,
            ));
        }
        self.record_create(guid, "Create Lake");
        guid
    }

    /// Create a **river**: an entity carrying a `WaterBody::river` *and* the
    /// `Spline` that is its centreline, on the same entity (P20.4).
    ///
    /// `points` are **world** space; they are stored relative to the entity's
    /// transform, which is placed at the first point. That keeps the spline's
    /// authored numbers small and makes "drag the river" move the whole
    /// centreline, which is what the same-entity composition buys.
    ///
    /// Fewer than two points yields a river with no ribbon — an authoring state
    /// (`RenderWater::drawable` skips it), not an error, so the tool can create
    /// the entity on the first click and keep appending.
    pub fn edit_create_river(
        &mut self,
        name: &str,
        points: &[DVec3],
        width_m: f64,
        depth_m: f64,
        flow_m_s: f64,
    ) -> Uuid {
        let origin = points.first().copied().unwrap_or(DVec3::ZERO);
        let guid = self.create(SpawnKind::Empty, name, None);
        if let Some(entity) = self.world.entity_of(guid) {
            let mut t = Transform::IDENTITY;
            t.translation = Vec3d::new(origin.x, origin.y, origin.z);
            self.world.world_mut().entity_mut(entity).insert((
                WaterBody::river(width_m.max(0.0), depth_m.max(0.0), flow_m_s),
                Spline {
                    points: points
                        .iter()
                        .map(|p| {
                            let l = *p - origin;
                            Vec3d::new(l.x, l.y, l.z)
                        })
                        .collect(),
                    closed: false,
                    interp: SplineInterp::CatmullRom,
                },
                t,
            ));
        }
        self.record_create(guid, "Create River");
        guid
    }

    /// Append a **world-space** control point to `guid`'s `Spline` (P20.4) — one
    /// undo step per click, recorded as a `SwapComponents` because a `Vec<Vec3d>`
    /// is not a reflection-addressable scalar.
    ///
    /// **Creates the `Spline` when the entity has none.** That is not a
    /// convenience: an entity carrying a `WaterKind::River` and no spline is
    /// precisely the state "I added the component through the Details menu and
    /// have not drawn the path yet", and it is the state the water tool meets
    /// most often. Refusing it wedged the tool — every click resolved to the same
    /// spline-less selection, did nothing, and said nothing (found in the P20.4
    /// audit). `SwapComponents` round-trips a component *addition* natively, so
    /// undo removes the spline again exactly as `edit_add_component` does.
    ///
    /// Returns `false` only when the entity is gone.
    ///
    /// The point is converted into the entity's own frame through its
    /// `GlobalTransform`, so appending to a river under a moved or rotated parent
    /// puts the point where the author clicked rather than where the untransformed
    /// numbers would land.
    pub fn edit_append_spline_point(&mut self, guid: Uuid, world: DVec3) -> bool {
        let Some(before) = crate::scene::serialize::record_of(self, guid) else {
            return false;
        };
        let Some(e) = self.world.entity_of(guid) else {
            return false;
        };
        let affine = self
            .world
            .world()
            .get::<GlobalTransform>(e)
            .map(|g| g.0)
            .unwrap_or(glam::DAffine3::IDENTITY);
        let local = affine.inverse().transform_point3(world);
        let point = Vec3d::new(local.x, local.y, local.z);
        {
            let w = self.world.world_mut();
            match w.get_mut::<Spline>(e) {
                Some(mut spline) => spline.points.push(point),
                None => {
                    w.entity_mut(e).insert(Spline {
                        points: vec![point],
                        closed: false,
                        interp: SplineInterp::CatmullRom,
                    });
                }
            }
        }
        let Some(after) = crate::scene::serialize::record_of(self, guid) else {
            return false;
        };
        self.touch();
        self.world.mark_dirty();
        self.world.propagate();
        self.history.record(
            "Add River Point",
            EditCommand::SwapComponents {
                guid,
                before: Box::new(before),
                after: Box::new(after),
            },
        );
        true
    }

    /// Rewrite `guid`'s river profile (width / depth / flow) in one undo step
    /// (P20.4) — the per-river half of profile editing, beside the per-control-
    /// point half the Details list already gives.
    ///
    /// **No schema growth**: every field already exists on `WaterBody` from
    /// P20.1. Returns `false` when the entity has no `WaterBody`.
    pub fn edit_set_river_profile(
        &mut self,
        guid: Uuid,
        width_start_m: f64,
        width_end_m: f64,
        depth_start_m: f64,
        depth_end_m: f64,
        flow_m_s: f64,
    ) -> bool {
        let Some(before) = crate::scene::serialize::record_of(self, guid) else {
            return false;
        };
        let Some(e) = self.world.entity_of(guid) else {
            return false;
        };
        {
            let w = self.world.world_mut();
            let Some(mut body) = w.get_mut::<WaterBody>(e) else {
                return false;
            };
            body.river_width_start_m = width_start_m.max(0.0);
            body.river_width_end_m = width_end_m.max(0.0);
            body.river_depth_start_m = depth_start_m.max(0.0);
            body.river_depth_end_m = depth_end_m.max(0.0);
            body.river_flow_m_s = flow_m_s;
        }
        let Some(after) = crate::scene::serialize::record_of(self, guid) else {
            return false;
        };
        if before == after {
            return false;
        }
        self.touch();
        self.world.mark_dirty();
        self.world.propagate();
        self.history.record(
            "Set River Profile",
            EditCommand::SwapComponents {
                guid,
                before: Box::new(before),
                after: Box::new(after),
            },
        );
        true
    }

    /// Move a lake's surface (P20.4) — the fill-level edit the tool's slider
    /// makes, in one undo step. Returns `false` for an entity with no
    /// `WaterBody`, or when nothing changed.
    pub fn edit_set_water_level(&mut self, guid: Uuid, level_m: f64, half_extent: Vec2d) -> bool {
        let Some(before) = crate::scene::serialize::record_of(self, guid) else {
            return false;
        };
        let Some(e) = self.world.entity_of(guid) else {
            return false;
        };
        {
            let w = self.world.world_mut();
            let Some(mut body) = w.get_mut::<WaterBody>(e) else {
                return false;
            };
            body.level_m = level_m;
            if body.kind == WaterKind::Lake {
                body.extent = Vec2d::new(half_extent.x.abs(), half_extent.y.abs());
            }
        }
        let Some(after) = crate::scene::serialize::record_of(self, guid) else {
            return false;
        };
        if before == after {
            return false;
        }
        self.touch();
        self.world.mark_dirty();
        self.world.propagate();
        self.history.record(
            "Set Water Level",
            EditCommand::SwapComponents {
                guid,
                before: Box::new(before),
                after: Box::new(after),
            },
        );
        true
    }

    /// Snapshot a freshly-created entity into one `Create` undo step. Shared by
    /// the P20.4 water spawns; the record is taken *after* their components are
    /// attached, which is what makes redo restore a lake rather than an empty.
    fn record_create(&mut self, guid: Uuid, label: &str) {
        let at = self.order.iter().position(|g| *g == guid).unwrap_or(0);
        if let Some(record) = crate::scene::serialize::record_of(self, guid) {
            self.history.record(
                label,
                EditCommand::Create {
                    at,
                    record: Box::new(record),
                },
            );
        }
        self.world.mark_dirty();
        self.world.propagate();
        self.touch();
    }

    /// Delete + record for undo (the whole subtree round-trips on undo).
    pub fn edit_delete(&mut self, guids: &[Uuid]) {
        use std::collections::HashSet;
        let mut set: HashSet<Uuid> = HashSet::new();
        for &g in guids {
            if let Some(e) = self.world.entity_of(g) {
                for se in self.world.subtree(e) {
                    if let Some(sg) = self.world.guid_of(se) {
                        set.insert(sg);
                    }
                }
            }
        }
        let items: Vec<(usize, EntityRecord)> = self
            .order
            .iter()
            .enumerate()
            .filter(|(_, g)| set.contains(g))
            .filter_map(|(i, g)| crate::scene::serialize::record_of(self, *g).map(|r| (i, r)))
            .collect();
        let tops: Vec<Uuid> = guids
            .iter()
            .copied()
            .filter(|g| self.world.entity_of(*g).is_some())
            .collect();
        if items.is_empty() {
            return;
        }
        self.delete(&tops);
        self.history
            .record("Delete", EditCommand::Delete { items, tops });
    }

    // ── duplicate / clipboard (editor seams) ─────────────────────────────
    //
    // Duplicate and copy/cut/paste share one core: snapshot a forest of subtree
    // records (parents-first via `world.subtree`, nested selection de-duped),
    // mint fresh GUIDs, remap internal parent links to the new GUIDs (a parent
    // OUTSIDE the copied set keeps its ORIGINAL GUID so the copy lands as a
    // sibling), then re-spawn them as ONE transaction of `Create` commands (so a
    // duplicate / paste is a single undo step). Clipboard records never cross a
    // process boundary — Ring 2 holds the `Vec<EntityRecord>` directly.

    /// Snapshot the subtree records for `roots`, **parents-first**, with nested
    /// selection de-duped (a root whose ancestor is also selected is skipped, so
    /// its subtree isn't copied twice) and repeated GUIDs collapsed. This is what
    /// the clipboard stores for copy / cut, and what [`Self::edit_duplicate`]
    /// re-spawns.
    pub fn collect_subtree_records(&self, roots: &[Uuid]) -> Vec<EntityRecord> {
        use std::collections::HashSet;
        let root_set: HashSet<Uuid> = roots.iter().copied().collect();
        let mut seen: HashSet<Uuid> = HashSet::new();
        let mut out = Vec::new();
        for &g in roots {
            if !seen.insert(g) {
                continue; // a repeated root in the input
            }
            let Some(e) = self.world.entity_of(g) else {
                continue;
            };
            // Skip if an ancestor is also selected — that ancestor's copy already
            // carries this node, so copying it again would double it.
            let mut anc = self.world.parent_of(e);
            let mut nested = false;
            while let Some(p) = anc {
                if self
                    .world
                    .guid_of(p)
                    .is_some_and(|pg| root_set.contains(&pg))
                {
                    nested = true;
                    break;
                }
                anc = self.world.parent_of(p);
            }
            if nested {
                continue;
            }
            for se in self.world.subtree(e) {
                if let Some(sg) = self.world.guid_of(se) {
                    if let Some(rec) = crate::scene::serialize::record_of(self, sg) {
                        out.push(rec);
                    }
                }
            }
        }
        out
    }

    /// Re-spawn a forest of `records` with fresh GUIDs, recording one `Create`
    /// per entity (the caller opens the enclosing transaction so the whole thing
    /// is one undo step). Internal parent links remap to the new GUIDs; a parent
    /// outside the set keeps its original GUID (sibling placement, or the scene
    /// root if that parent no longer exists). When `rename_roots`, each root
    /// record's name gets a unique " Copy" suffix. Returns the new root GUIDs.
    /// `records` MUST be parents-first (as [`Self::collect_subtree_records`]
    /// produces) so each child's remapped parent already exists on redo.
    fn spawn_records_remapped(
        &mut self,
        records: &[EntityRecord],
        rename_roots: bool,
    ) -> Vec<Uuid> {
        use std::collections::{HashMap, HashSet};
        let old_ids: HashSet<Uuid> = records.iter().map(|r| r.guid).collect();
        let idmap: HashMap<Uuid, Uuid> = records.iter().map(|r| (r.guid, Uuid::new_v4())).collect();
        let mut new_roots = Vec::new();
        for rec in records {
            let is_root = match rec.parent {
                Some(p) => !old_ids.contains(&p),
                None => true,
            };
            let mut nr = rec.clone();
            nr.guid = idmap[&rec.guid];
            nr.parent = match rec.parent {
                Some(p) if old_ids.contains(&p) => Some(idmap[&p]),
                other => other,
            };
            if is_root {
                if rename_roots {
                    nr.name = self.unique_copy_name(&nr.name);
                }
                new_roots.push(nr.guid);
            }
            let at = self.order.len();
            self.raw_spawn_record(&nr, at);
            self.history.record(
                "Create",
                EditCommand::Create {
                    at,
                    record: Box::new(nr),
                },
            );
        }
        new_roots
    }

    /// Duplicate each selected root's subtree: fresh GUIDs, internal parent links
    /// preserved, each copy a sibling of its original, root names " Copy"-suffixed
    /// — all one undo step. Nested selection is de-duped. Returns the new root
    /// GUIDs (Ring 2 selects them).
    pub fn edit_duplicate(&mut self, roots: &[Uuid]) -> Vec<Uuid> {
        let records = self.collect_subtree_records(roots);
        if records.is_empty() {
            return Vec::new();
        }
        self.begin_transaction("Duplicate");
        let new_roots = self.spawn_records_remapped(&records, true);
        self.commit_transaction();
        new_roots
    }

    /// Paste a forest of clipboard `records` (from [`Self::collect_subtree_records`]):
    /// fresh GUIDs, internal parent links preserved; a record whose parent isn't
    /// in the set becomes a root, landing under its original parent if that still
    /// exists, else at the scene root — one undo step. Returns the new root GUIDs.
    pub fn edit_paste_records(&mut self, records: &[EntityRecord]) -> Vec<Uuid> {
        if records.is_empty() {
            return Vec::new();
        }
        self.begin_transaction("Paste");
        let new_roots = self.spawn_records_remapped(records, false);
        self.commit_transaction();
        new_roots
    }

    /// A unique " Copy" / " Copy N" suffixed variant of `base` not already used
    /// by a live entity (UE-style duplicate naming; see [`default_name`]).
    fn unique_copy_name(&self, base: &str) -> String {
        let first = format!("{base} Copy");
        if !self.name_in_use(&first) {
            return first;
        }
        let mut n = 2;
        loop {
            let candidate = format!("{base} Copy {n}");
            if !self.name_in_use(&candidate) {
                return candidate;
            }
            n += 1;
        }
    }

    fn name_in_use(&self, name: &str) -> bool {
        self.order.iter().any(|&g| self.display_name(g) == name)
    }

    pub fn edit_rename(&mut self, guid: Uuid, name: &str) {
        let Some(e) = self.world.entity_of(guid) else {
            return;
        };
        let before = self.world.name_of(e).unwrap_or("").to_string();
        if before == name {
            return;
        }
        self.rename(guid, name);
        self.history.record(
            "Rename",
            EditCommand::Rename {
                guid,
                before,
                after: name.to_string(),
            },
        );
    }

    pub fn edit_reparent(&mut self, guid: Uuid, parent: Option<Uuid>) -> bool {
        let before = self
            .world
            .entity_of(guid)
            .and_then(|e| self.world.parent_of(e))
            .and_then(|p| self.world.guid_of(p));
        if before == parent {
            return true;
        }
        let ok = self.reparent(guid, parent);
        if ok {
            self.history.record(
                "Reparent",
                EditCommand::Reparent {
                    guid,
                    before,
                    after: parent,
                },
            );
        }
        ok
    }

    pub fn edit_set_visible(&mut self, guid: Uuid, visible: bool) {
        let Some(e) = self.world.entity_of(guid) else {
            return;
        };
        let before = self
            .world
            .world()
            .get::<Visibility>(e)
            .map(|v| v.visible)
            .unwrap_or(true);
        if before == visible {
            return;
        }
        self.set_visible(guid, visible);
        self.history.record(
            "Set Visibility",
            EditCommand::SetVisible {
                guid,
                before,
                after: visible,
            },
        );
    }

    pub fn edit_set_transform(&mut self, guid: Uuid, t: Transform) {
        let Some(e) = self.world.entity_of(guid) else {
            return;
        };
        let before = self
            .world
            .world()
            .get::<Transform>(e)
            .copied()
            .unwrap_or(Transform::IDENTITY);
        if before == t {
            return;
        }
        self.raw_set_transform(guid, t);
        self.history.record(
            "Move",
            EditCommand::SetTransform {
                guid,
                before,
                after: t,
            },
        );
    }

    /// Write a **world-space** TRS back onto an entity, composing the correct
    /// local transform for its parent (Wave 2). `world_trs` is a [`Transform`]
    /// whose euler-deg rotation / translation / scale describe the world pose the
    /// gizmo produced; this recovers `local = parent_global⁻¹ · world` so a child
    /// of a rotated/scaled parent keeps its edited world pose (the previous
    /// writeback wrote world TRS straight into the local `Transform`, which was
    /// only correct for roots / identity parents — the documented P3 bug).
    ///
    /// Records a `SetTransform` undo entry via [`Self::edit_set_transform`], so a
    /// gizmo drag stays one undo step inside the enclosing transaction. Parent
    /// `GlobalTransform`s are refreshed (`propagate`) up front so a **multi-select
    /// drag applied parents-first** composes each child against its parent's
    /// already-written pose.
    pub fn edit_set_world_transform(&mut self, guid: Uuid, world_trs: Transform) {
        // Ensure parent GlobalTransforms reflect any earlier writebacks in this
        // batch (and the committed scene). Cheap when nothing is dirty.
        self.world.propagate();
        let Some(e) = self.world.entity_of(guid) else {
            return;
        };
        let parent_global = self
            .world
            .parent_of(e)
            .and_then(|p| self.world.world().get::<GlobalTransform>(p).map(|g| g.0))
            .unwrap_or(glam::DAffine3::IDENTITY);
        // World matrix from the incoming TRS → parent-relative local matrix.
        let local_affine = parent_global.inverse() * world_trs.affine();
        let (scale, rotation, translation) = local_affine.to_scale_rotation_translation();
        let mut local = Transform::from_translation(translation);
        local.set_quat(rotation);
        local.scale = inf_ecs::Vec3d::from_dvec3(scale);
        self.edit_set_transform(guid, local);
    }

    pub fn edit_set_prop(
        &mut self,
        guid: Uuid,
        type_path: &str,
        field: &str,
        value: &PropValue,
    ) -> bool {
        let Some(before) = self.prop_value(guid, type_path, field) else {
            return false;
        };
        if before == *value {
            return true;
        }
        let ok = self.write_prop(guid, type_path, field, value);
        if ok {
            let label = format!("Edit {field}");
            self.history.record(
                &label,
                EditCommand::SetProp {
                    guid,
                    type_path: type_path.to_string(),
                    field: field.to_string(),
                    before,
                    after: value.clone(),
                },
            );
        }
        ok
    }

    /// The default value for a single **element** of the list at `path` on
    /// `type_path`'s component (E-P1 ListField "add element"). `None` if the path
    /// is not a list.
    pub fn list_default(&self, type_path: &str, path: &str) -> Option<PropValue> {
        inf_ecs::default_list_element(self.world.registry(), type_path, path)
    }

    /// Add a `Default` instance of `type_path`'s component to `guid` (E-P1),
    /// recorded as one `SwapComponents` undo step. Returns `false` (no-op) when
    /// the component is not addable, is already present, or the entity is gone.
    pub fn edit_add_component(&mut self, guid: Uuid, type_path: &str) -> bool {
        if !self.is_addable(type_path) {
            return false;
        }
        // Idempotent: adding a component that already exists is a no-op.
        if self
            .entity_props(guid)
            .iter()
            .any(|c| c.type_path == type_path)
        {
            return false;
        }
        let Some(before) = crate::scene::serialize::record_of(self, guid) else {
            return false;
        };
        let Some(e) = self.world.entity_of(guid) else {
            return false;
        };
        if !self.world.insert_default_component(e, type_path) {
            return false;
        }
        let Some(after) = crate::scene::serialize::record_of(self, guid) else {
            return false;
        };
        self.touch();
        self.world.mark_dirty();
        self.world.propagate();
        self.history.record(
            "Add Component",
            EditCommand::SwapComponents {
                guid,
                before: Box::new(before),
                after: Box::new(after),
            },
        );
        true
    }

    /// Remove `type_path`'s component from `guid` (E-P1), recorded as one
    /// `SwapComponents` undo step. Returns `false` when the component is
    /// structural (only the addable set may be removed — so `Transform` is
    /// rejected), absent, or the entity is gone.
    pub fn edit_remove_component(&mut self, guid: Uuid, type_path: &str) -> bool {
        if !self.is_addable(type_path) {
            return false; // structural (Transform) / unknown → refuse
        }
        if !self
            .entity_props(guid)
            .iter()
            .any(|c| c.type_path == type_path)
        {
            return false; // not present
        }
        let Some(before) = crate::scene::serialize::record_of(self, guid) else {
            return false;
        };
        let Some(e) = self.world.entity_of(guid) else {
            return false;
        };
        if !self.world.remove_component(e, type_path) {
            return false;
        }
        let Some(after) = crate::scene::serialize::record_of(self, guid) else {
            return false;
        };
        self.touch();
        self.world.mark_dirty();
        self.world.propagate();
        self.history.record(
            "Remove Component",
            EditCommand::SwapComponents {
                guid,
                before: Box::new(before),
                after: Box::new(after),
            },
        );
        true
    }

    /// Whether `type_path` is a user-addable/removable component (the editable
    /// set minus the structural `Transform`).
    fn is_addable(&self, type_path: &str) -> bool {
        self.world
            .registry()
            .addable()
            .any(|c| c.type_path == type_path)
    }

    /// Re-apply an [`EntityRecord`]'s full component set onto `guid`, removing any
    /// optional component the record leaves `None` (E-P1 add/remove-component
    /// undo — [`EditCommand::SwapComponents`]). The record is the complete truth.
    pub(crate) fn raw_apply_record_components(
        &mut self,
        guid: Uuid,
        rec: &crate::scene::serialize::EntityRecord,
    ) {
        let Some(e) = self.world.entity_of(guid) else {
            return;
        };
        crate::scene::serialize::write_record_components(self.world_mut(), e, rec, true);
        self.touch();
        self.world.mark_dirty();
        self.world.propagate();
    }

    /// Apply a material's PBR parameters to each target entity's `Material`
    /// component as one undo step (Content-Drawer apply-by-drag / "Apply to
    /// Selection", P7.1). **A target without a `Material` gets one inserted**
    /// (wave CHAR1a audit — it used to be skipped, silently, which is why
    /// dragging a skin onto a character applied to zero entities); a target that
    /// is gone, or that the registry will not let carry a `Material`, is refused
    /// with a warning. Returns how many entities were updated.
    ///
    /// # P26.3b — it also writes the BINDING
    ///
    /// Since P7.1 this flattened a `.inf_mat`'s scalars onto the component and
    /// threw the reference away, so nothing on disk recorded which material a
    /// surface uses — which is why `.inf_mat` texture references resolved in
    /// neither host however much virtual-texturing machinery stood behind them
    /// (the P26.3 spec-clause-4 gap). `asset` now carries that reference into
    /// scene v22.
    ///
    /// **The flattening stays**, for two reasons that are not the same reason: a
    /// pre-v22 level's scalars are the only copy it has, and a material with no
    /// textures must keep rendering off the per-instance attributes rather than
    /// through a resolution that can fail. So the binding is *added* to the
    /// scalars and never replaces them — `None` is the permanent no-texture
    /// path, structurally, not a branch someone has to remember to write.
    // A flat PBR-block + blend forwarding seam (mirrors the ECS `Material`
    // fields); bundling them into a struct would just move the noise.
    #[allow(clippy::too_many_arguments)]
    pub fn edit_apply_material(
        &mut self,
        targets: &[Uuid],
        // The `.inf_mat` asset these parameters came from (P26.3b). `None`
        // CLEARS the binding rather than leaving the previous one over the new
        // scalars (P26.3b audit) — the `edit_apply_sprite_slice(texture: Option)`
        // shape. A caller with no asset to name must not invent one; it must also
        // not bequeath one.
        asset: Option<Uuid>,
        base_color: [f32; 4],
        metallic: f32,
        roughness: f32,
        emissive: [f32; 3],
        // R-P5: the material's blend mode, as the ECS `BlendMode` **variant name**
        // ("Opaque" / "Masked" / "Translucent"). The Ring-2 glue maps the
        // inf-material `MatBlend` enum onto this string (inf-material does not
        // depend on inf-ecs), and it's written through the enum reflection path.
        blend: &str,
        alpha_cutoff: f32,
    ) -> usize {
        let Some(tp) = self.world.registry().type_path_for("Material") else {
            return 0;
        };
        self.begin_transaction("Apply Material");
        let mut applied = 0;
        for &g in targets {
            // **A target with no `Material` GETS ONE** (wave CHAR1a audit).
            //
            // This used to `continue`, and it returned a count nobody reads —
            // so dragging a material onto a character in the viewport applied to
            // *nothing* and said *nothing*. Measured on the running editor:
            // `skin on hero: 0`, with no warning anywhere. A door whose refusal
            // is a number in a return value is a silent skip.
            //
            // Inserting is the same answer [`edit_apply_sprite_slice`] gives one
            // door over ("a target without a `Sprite` gets one inserted"), and it
            // is the right one for the same reason: the caller named a surface
            // and a material, and the component is the *place the surface lives*,
            // not a precondition the author was supposed to have arranged.
            // `edit_add_component` is the door, so the insert is the ordinary
            // recorded `SwapComponents` step — inside this transaction, so the
            // whole apply is still ONE undo.
            //
            // An entity that is gone, or one the registry will not let carry a
            // `Material`, is refused OUT LOUD.
            if self.prop_value(g, tp, "base_color").is_none() && !self.edit_add_component(g, tp) {
                tracing::warn!(
                    "inf-editor-core: apply-material refused {g}: it carries no \
                     `Material` and one could not be inserted (entity gone, or the \
                     component is not addable)"
                );
                continue;
            }
            self.edit_set_prop(g, tp, "base_color", &PropValue::Color(base_color));
            self.edit_set_prop(g, tp, "metallic", &PropValue::Number(metallic as f64));
            self.edit_set_prop(g, tp, "roughness", &PropValue::Number(roughness as f64));
            self.edit_set_prop(
                g,
                tp,
                "emissive",
                &PropValue::Color([emissive[0], emissive[1], emissive[2], 1.0]),
            );
            self.edit_set_prop(
                g,
                tp,
                "blend",
                &PropValue::Enum {
                    value: blend.to_string(),
                    options: Vec::new(),
                },
            );
            self.edit_set_prop(
                g,
                tp,
                "alpha_cutoff",
                &PropValue::Number(alpha_cutoff as f64),
            );
            // P26.3b — the binding, inside the SAME transaction as the scalars.
            // A separate step would let an undo leave a surface whose numbers
            // came from one material and whose textures come from another.
            //
            // **Written unconditionally** (P26.3b audit): `None` CLEARS, exactly
            // as `edit_apply_sprite_slice`'s `texture: Option<Uuid>` does one
            // door over. The first cut wrote only `Some`, which meant applying a
            // material this caller could not name left the PREVIOUS binding
            // standing over the new scalars — the surface's numbers from one
            // material and its textures from another, which is the very state
            // the same-transaction write exists to prevent. "Do not invent a
            // binding" and "do not leave a stale one" are not the same rule, and
            // only the second one is structural.
            let before = self.raw_get_material_asset(g);
            if before != asset {
                self.raw_set_material_asset(g, asset);
                self.history.record(
                    "Apply Material",
                    EditCommand::SetMaterialAsset {
                        guid: g,
                        before,
                        after: asset,
                    },
                );
            }
            applied += 1;
        }
        self.commit_transaction();
        applied
    }

    /// Bind a blueprint-class asset GUID to each target entity's [`ActorClass`]
    /// link as one undo step (Content-Drawer drag a `.inf_act` onto an entity,
    /// P9.5 — mirrors [`Self::edit_apply_material`]). Returns how many entities
    /// were (re)bound; a target already bound to the same GUID is skipped.
    pub fn edit_apply_actor(&mut self, targets: &[Uuid], actor: Uuid) -> usize {
        self.begin_transaction("Bind Actor");
        let mut applied = 0;
        for &g in targets {
            if self.world.entity_of(g).is_none() {
                continue;
            }
            let before = self.raw_get_actor(g);
            let after = Some(actor);
            if before == after {
                continue;
            }
            self.raw_set_actor(g, after);
            self.history.record(
                "Bind Actor",
                EditCommand::SetActor {
                    guid: g,
                    before,
                    after,
                },
            );
            applied += 1;
        }
        self.commit_transaction();
        applied
    }

    /// Apply a sprite-sheet slice to each target's [`Sprite`] component as one
    /// undo step (P8.2a "Apply to Selection"). A target without a `Sprite` gets
    /// one inserted (defaults + the slice); an existing one keeps its other
    /// fields (pivot, color, sorting layer, flips). `size`, when `Some`, sets the
    /// quad extent from the slice's pixel aspect. Returns how many were updated.
    pub fn edit_apply_sprite_slice(
        &mut self,
        targets: &[Uuid],
        texture: Option<uuid::Uuid>,
        uv_min: [f64; 2],
        uv_max: [f64; 2],
        size: Option<[f64; 2]>,
    ) -> usize {
        let atlas_rect = AtlasRect {
            min: Vec2d::new(uv_min[0], uv_min[1]),
            max: Vec2d::new(uv_max[0], uv_max[1]),
        };
        self.begin_transaction("Apply Sprite Slice");
        let mut applied = 0;
        for &g in targets {
            if self.world.entity_of(g).is_none() {
                continue;
            }
            let before = self.raw_get_sprite(g);
            let mut sprite = before.clone().unwrap_or_default();
            sprite.texture = texture;
            sprite.atlas_rect = atlas_rect;
            if let Some(sz) = size {
                sprite.size = Vec2d::new(sz[0], sz[1]);
            }
            let after = Some(sprite);
            if before == after {
                continue;
            }
            self.raw_set_sprite(g, after.clone());
            self.history.record(
                "Apply Sprite Slice",
                EditCommand::SetSprite {
                    guid: g,
                    before,
                    after,
                },
            );
            applied += 1;
        }
        self.commit_transaction();
        applied
    }

    /// Paint a batch of tiles onto an entity's [`Tilemap`] as **one** undo step
    /// (a full mouse-down→up stroke, P8.2b). `cells` is `(x, y, index)` with
    /// `index == 0` erasing. Only cells that actually change are touched, and the
    /// recorded command stores just those cells' pre/post index (not the chunk
    /// map). Returns how many cells changed. No-op if the entity has no `Tilemap`.
    pub fn edit_paint_tiles(&mut self, guid: Uuid, cells: &[(i32, i32, u32)]) -> usize {
        let Some(e) = self.world.entity_of(guid) else {
            return 0;
        };
        // Diff against the current map. A stroke may touch a coordinate more than
        // once (drag back over a cell); collapse to one entry per coordinate —
        // original map value → final painted value — in first-touch order, and
        // keep only cells that actually change.
        let changed: Vec<(i32, i32, u32, u32)> = {
            use std::collections::HashMap;
            let Some(tm) = self.world.world().get::<Tilemap>(e) else {
                return 0;
            };
            let mut orig: HashMap<(i32, i32), u32> = HashMap::new();
            let mut last: HashMap<(i32, i32), u32> = HashMap::new();
            let mut order: Vec<(i32, i32)> = Vec::new();
            for &(x, y, after) in cells {
                orig.entry((x, y)).or_insert_with(|| tm.get_tile(x, y));
                if last.insert((x, y), after).is_none() {
                    order.push((x, y));
                }
            }
            order
                .into_iter()
                .filter_map(|k| {
                    let (b, a) = (orig[&k], last[&k]);
                    (b != a).then_some((k.0, k.1, b, a))
                })
                .collect()
        };
        if changed.is_empty() {
            return 0;
        }
        let after_cells: Vec<(i32, i32, u32)> =
            changed.iter().map(|&(x, y, _, a)| (x, y, a)).collect();
        self.raw_set_tiles(guid, &after_cells);
        let n = changed.len();
        self.history.record(
            "Paint Tiles",
            EditCommand::SetTiles {
                guid,
                cells: changed,
            },
        );
        n
    }

    // ── time of day (P17.1) ──────────────────────────────────────────────

    /// The level's clock, as the World Settings panel shows it: the **sky
    /// authority's** [`TimeOfDay`], or `None` when the level has none (which is
    /// every pre-P17.1 level, and renders under the retired `SUN_DIR`).
    ///
    /// The authority is resolved by `inf_ecs::sky` (lowest `Guid` wins) — the same
    /// rule both scene projectors use, so the panel can never show a different
    /// clock from the one the viewport is rendering.
    pub fn time_of_day(&self) -> Option<TimeOfDay> {
        inf_ecs::sky::resolve_sky(&self.world).map(|s| s.time_of_day)
    }

    /// The `Guid` of the entity carrying the level's clock, if any.
    pub fn sky_authority(&self) -> Option<Uuid> {
        inf_ecs::sky::sky_authority(&self.world).and_then(|e| self.world.guid_of(e))
    }

    /// The level's atmosphere (P17.2): the **sky authority's** `SkyAtmosphere`,
    /// or `None` when the level has no clock.
    ///
    /// Resolved through the same authority rule as [`Self::time_of_day`], so the
    /// panel can never show one entity's atmosphere while the viewport renders
    /// another's. Note this returns `None` for a clockless level even if some
    /// entity carries a stray `SkyAtmosphere` — which is correct, because such a
    /// component is inert (`inf_ecs::sky` warns about exactly that shape).
    pub fn sky_atmosphere(&self) -> Option<inf_ecs::components::SkyAtmosphere> {
        inf_ecs::sky::resolve_sky(&self.world).map(|s| s.atmosphere)
    }

    /// Edit the level's clock as **one** undo step (World Settings; the frontend
    /// debounces the calls).
    ///
    /// `create` decides what happens when the level has **no** clock yet:
    ///
    /// * `true` — create the sky authority (an empty entity named `Sky` carrying
    ///   [`TimeOfDay`] + `SkyAtmosphere` at their defaults) inside the same
    ///   transaction, so a single undo removes the whole opt-in. This is
    ///   deliberately the only way a level gains a dynamic sun from this panel:
    ///   the components are the source of truth (they persist, animate from
    ///   Blueprints and key from the sequencer), and World Settings is a view onto
    ///   them rather than a second place the sun could live.
    /// * `false` — do **nothing at all**: no entity, no undo entry, no version
    ///   bump, no dirty flag. This is what a settings write that merely *echoed
    ///   back* the previewed defaults must do — the World Settings panel sends the
    ///   whole settings block on every edit, including the gravity and partition
    ///   rows, and none of those may conjure a sun as a side effect.
    ///
    /// A level that already has a clock is written either way (`create` only
    /// governs the create).
    ///
    /// Returns the authority's `Guid`, or `None` when nothing was done. A no-op
    /// edit (same values) returns the existing `Guid` and records nothing.
    pub fn edit_time_of_day(&mut self, tod: TimeOfDay, create: bool) -> Option<Uuid> {
        // Reflect type paths come through the registry facade — `bevy_reflect`
        // never leaves `inf-ecs` (architecture rule). Both components are
        // registered in `inf_ecs::registry`, so this cannot fail in practice; it
        // is `None` rather than a panic so a settings write can never take the
        // editor down.
        let reg = self.world.registry();
        let (Some(tp), Some(atmos_tp)) = (
            reg.type_path_for("TimeOfDay"),
            reg.type_path_for("SkyAtmosphere"),
        ) else {
            debug_assert!(false, "TimeOfDay/SkyAtmosphere must be registered");
            return None;
        };
        let existing = self.sky_authority();
        if existing.is_none() && !create {
            return None;
        }
        if let Some(guid) = existing {
            if self.time_of_day() == Some(tod) {
                return Some(guid);
            }
        }

        self.begin_transaction("Time of Day");
        let guid = match existing {
            Some(g) => g,
            None => {
                let g = self.edit_create(SpawnKind::Empty, "Sky", None);
                // Both must land: the entity is brand new, so neither component can
                // already be present, and both are addable. A silent `false` here
                // would leave a `Sky` actor that does nothing.
                // Clock FIRST: the intermediate state is then "a clock with a
                // defaulted atmosphere" (benign) rather than "an atmosphere with no
                // clock", which is the shape `inf_ecs::sky` warns about.
                let added_clock = self.edit_add_component(g, tp);
                let added_atmos = self.edit_add_component(g, atmos_tp);
                debug_assert!(
                    added_atmos && added_clock,
                    "the sky authority must gain both components"
                );
                g
            }
        };
        // Written through the reflection edit path so each field records a normal
        // `SetProp` (the transaction collapses them into the one step) and the
        // Details grid, the sequencer and this panel all go through one door.
        self.edit_set_prop(guid, tp, "seconds", &PropValue::Number(tod.seconds));
        self.edit_set_prop(
            guid,
            tp,
            "day_of_year",
            &PropValue::Number(f64::from(tod.day_of_year)),
        );
        self.edit_set_prop(
            guid,
            tp,
            "latitude_deg",
            &PropValue::Number(tod.latitude_deg),
        );
        self.edit_set_prop(
            guid,
            tp,
            "longitude_deg",
            &PropValue::Number(tod.longitude_deg),
        );
        self.edit_set_prop(guid, tp, "rate", &PropValue::Number(tod.rate));
        self.commit_transaction();
        Some(guid)
    }

    /// Edit the level's **atmosphere** as one undo step (P17.2) — the sibling of
    /// [`Self::edit_time_of_day`], with identical `create` semantics.
    ///
    /// The two are separate entry points rather than one combined write because
    /// the panel debounces them independently and a user dragging the fog slider
    /// should not be told they edited "Time of Day" in the undo stack. They share
    /// the authority, so whichever runs first on a clockless level creates it —
    /// and `edit_time_of_day` deliberately creates *both* components, so the
    /// second call always finds a target.
    ///
    /// `create: false` on a clockless level is a total no-op: no entity, no undo
    /// entry, no version bump, no dirty flag. That matters for the same reason it
    /// does on the clock — the panel posts the whole settings block on every edit,
    /// so without it, nudging gravity would conjure a sky out of the previewed
    /// defaults.
    ///
    /// Only the fields [`crate::ipc::SkyAtmosphereDto`] exposes are written; the
    /// component's five `Color` fields are left alone, so a Details-authored sun
    /// colour survives a fog edit.
    pub fn edit_sky_atmosphere(
        &mut self,
        atmos: inf_ecs::components::SkyAtmosphere,
        create: bool,
    ) -> Option<Uuid> {
        let reg = self.world.registry();
        let Some(tp) = reg.type_path_for("SkyAtmosphere") else {
            debug_assert!(false, "SkyAtmosphere must be registered");
            return None;
        };
        let existing = self.sky_authority();
        if existing.is_none() && !create {
            return None;
        }
        if let Some(guid) = existing {
            if self.sky_atmosphere() == Some(atmos) {
                return Some(guid);
            }
        }

        self.begin_transaction("Sky Atmosphere");
        // Creating goes through `edit_time_of_day`, not through a second
        // hand-rolled spawn: it is the one place that knows the authority must
        // gain the clock FIRST (the intermediate "atmosphere with no clock" state
        // is the shape `inf_ecs::sky` warns about), and reusing it means there is
        // exactly one definition of what a sky authority is.
        let guid = match existing {
            Some(g) => g,
            None => match self.edit_time_of_day(TimeOfDay::default(), true) {
                Some(g) => g,
                None => {
                    self.commit_transaction();
                    return None;
                }
            },
        };
        let num = |v: f32| PropValue::Number(f64::from(v));
        self.edit_set_prop(guid, tp, "enabled", &PropValue::Bool(atmos.enabled));
        self.edit_set_prop(guid, tp, "physical", &PropValue::Bool(atmos.physical));
        self.edit_set_prop(guid, tp, "sky_intensity", &num(atmos.sky_intensity));
        self.edit_set_prop(guid, tp, "turbidity", &num(atmos.turbidity));
        self.edit_set_prop(guid, tp, "mie_anisotropy", &num(atmos.mie_anisotropy));
        self.edit_set_prop(guid, tp, "sun_disc_deg", &num(atmos.sun_disc_deg));
        self.edit_set_prop(guid, tp, "moon_disc_deg", &num(atmos.moon_disc_deg));
        self.edit_set_prop(guid, tp, "star_intensity", &num(atmos.star_intensity));
        self.edit_set_prop(guid, tp, "tint_strength", &num(atmos.tint_strength));
        self.edit_set_prop(
            guid,
            tp,
            "aerial_perspective",
            &num(atmos.aerial_perspective),
        );
        self.edit_set_prop(guid, tp, "fog_density", &num(atmos.fog_density));
        self.edit_set_prop(guid, tp, "fog_falloff", &num(atmos.fog_falloff));
        self.edit_set_prop(guid, tp, "fog_height", &num(atmos.fog_height));
        // ── volumetric clouds (P17.3) ──
        self.edit_set_prop(
            guid,
            tp,
            "clouds_enabled",
            &PropValue::Bool(atmos.clouds_enabled),
        );
        self.edit_set_prop(guid, tp, "cloud_coverage", &num(atmos.cloud_coverage));
        self.edit_set_prop(guid, tp, "cloud_type", &num(atmos.cloud_type));
        self.edit_set_prop(guid, tp, "cloud_bottom", &num(atmos.cloud_bottom));
        self.edit_set_prop(guid, tp, "cloud_top", &num(atmos.cloud_top));
        self.edit_set_prop(guid, tp, "cloud_density", &num(atmos.cloud_density));
        self.edit_set_prop(guid, tp, "cloud_detail", &num(atmos.cloud_detail));
        self.edit_set_prop(
            guid,
            tp,
            "cloud_seed",
            &PropValue::Number(f64::from(atmos.cloud_seed)),
        );
        self.edit_set_prop(guid, tp, "cloud_wind_x", &num(atmos.cloud_wind_x));
        self.edit_set_prop(guid, tp, "cloud_wind_z", &num(atmos.cloud_wind_z));
        self.edit_set_prop(guid, tp, "cloud_phase_g", &num(atmos.cloud_phase_g));
        self.edit_set_prop(guid, tp, "cloud_shadow", &num(atmos.cloud_shadow));
        self.edit_set_prop(guid, tp, "cloud_ambient", &num(atmos.cloud_ambient));
        self.commit_transaction();
        Some(guid)
    }

    /// Edit the level's **weather** block as one undo step (P17.4) — the third
    /// sibling of [`Self::edit_time_of_day`] / [`Self::edit_sky_atmosphere`],
    /// with identical `create` semantics.
    ///
    /// Separate from `edit_sky_atmosphere` even though both write the same
    /// component, for the reason that split them in the first place: the undo
    /// entry is named after what the user did. Clicking "Storm" should read
    /// **Weather** in the history, not "Sky Atmosphere".
    ///
    /// Only the eleven `weather_*` fields are written, so an atmosphere edit and
    /// a weather edit in either order compose rather than overwrite — which is
    /// exactly what the panel does, since it posts the whole settings block on
    /// every change.
    pub fn edit_weather(
        &mut self,
        atmos: inf_ecs::components::SkyAtmosphere,
        create: bool,
    ) -> Option<Uuid> {
        let reg = self.world.registry();
        let Some(tp) = reg.type_path_for("SkyAtmosphere") else {
            debug_assert!(false, "SkyAtmosphere must be registered");
            return None;
        };
        let existing = self.sky_authority();
        if existing.is_none() && !create {
            return None;
        }
        // A no-op records nothing: no undo entry, no version bump, no dirty flag.
        if let Some(guid) = existing {
            if self.sky_atmosphere().is_some_and(|a| {
                a.weather_params() == atmos.weather_params()
                    && a.weather_enabled == atmos.weather_enabled
                    && a.weather_target == atmos.weather_target
                    && a.weather_blend_seconds == atmos.weather_blend_seconds
                    && a.weather_blend_remaining == atmos.weather_blend_remaining
            }) {
                return Some(guid);
            }
        }

        self.begin_transaction("Weather");
        let guid = match existing {
            Some(g) => g,
            None => match self.edit_time_of_day(inf_ecs::components::TimeOfDay::default(), true) {
                Some(g) => g,
                None => {
                    self.commit_transaction();
                    return None;
                }
            },
        };
        let num = |v: f32| PropValue::Number(f64::from(v));
        self.edit_set_prop(
            guid,
            tp,
            "weather_enabled",
            &PropValue::Bool(atmos.weather_enabled),
        );
        // The preset is a reflected unit enum, written by **variant name** — the
        // Rust spelling (`"Storm"`), not the lowercase wire spelling, because that
        // is what `bevy_reflect`'s `DynamicEnum` matches on.
        self.edit_set_prop(
            guid,
            tp,
            "weather_target",
            &PropValue::Enum {
                value: atmos.weather_target.variant_name().to_string(),
                options: Vec::new(),
            },
        );
        self.edit_set_prop(
            guid,
            tp,
            "weather_blend_seconds",
            &num(atmos.weather_blend_seconds),
        );
        self.edit_set_prop(
            guid,
            tp,
            "weather_blend_remaining",
            &num(atmos.weather_blend_remaining),
        );
        self.edit_set_prop(guid, tp, "weather_coverage", &num(atmos.weather_coverage));
        self.edit_set_prop(
            guid,
            tp,
            "weather_cloud_type",
            &num(atmos.weather_cloud_type),
        );
        self.edit_set_prop(guid, tp, "weather_wind_x", &num(atmos.weather_wind_x));
        self.edit_set_prop(guid, tp, "weather_wind_z", &num(atmos.weather_wind_z));
        self.edit_set_prop(
            guid,
            tp,
            "weather_fog_density",
            &num(atmos.weather_fog_density),
        );
        self.edit_set_prop(
            guid,
            tp,
            "weather_precipitation",
            &num(atmos.weather_precipitation),
        );
        self.edit_set_prop(guid, tp, "weather_snowiness", &num(atmos.weather_snowiness));
        self.commit_transaction();
        Some(guid)
    }

    // ── history control ──────────────────────────────────────────────────

    /// Open an undo transaction; every recorded edit until [`Self::commit_transaction`]
    /// collapses into one entry (a gizmo drag is one undo step, P3.4.2).
    pub fn begin_transaction(&mut self, label: &str) {
        self.history.begin(label);
    }

    pub fn commit_transaction(&mut self) {
        self.history.commit();
    }

    /// `true` while an undo transaction is open — i.e. while some gesture owns
    /// the history and every recorded edit is being folded into its entry.
    pub fn has_open_transaction(&self) -> bool {
        self.history.has_open()
    }

    /// **Close a transaction whose owner will never close it**, whatever its
    /// nesting depth. Returns `true` when one was closed and had edits in it.
    ///
    /// See [`EditHistory::settle_open`](crate::scene::undo::EditHistory) for the
    /// failure: one unmatched `begin_transaction` makes `undo_len()` stop
    /// growing and **kills Ctrl+Z for the rest of the session**, silently. The
    /// viewport pump calls this when no gesture that owns a transaction is in
    /// flight, which is the same settlement discipline the stroke settlers use.
    ///
    /// **Only safe from a caller that holds the document lock and knows no
    /// gesture of its own is mid-transaction.** Ring-2 commands open and close
    /// theirs synchronously inside one function under that same lock, so the
    /// viewport thread can never observe one of those half-open.
    pub fn settle_open_transaction(&mut self) -> bool {
        let settled = self.history.settle_open();
        if settled {
            self.touch();
        }
        settled
    }

    pub fn can_undo(&self) -> bool {
        self.history.can_undo()
    }

    pub fn can_redo(&self) -> bool {
        self.history.can_redo()
    }

    /// Undo-stack depth (bounded by the history limit). Used by the P15 memory
    /// diagnostics + the soak test's "memory doesn't grow unboundedly" invariant.
    pub fn undo_len(&self) -> usize {
        self.history.undo_len()
    }

    /// Redo-stack depth.
    pub fn redo_len(&self) -> usize {
        self.history.redo_len()
    }

    /// **Approximate bytes the undo + redo stacks are holding** (Hardening D).
    ///
    /// The number the P15 memory diagnostics report instead of
    /// `undo_depth × 512`: a sculpt or paint stroke on the stack is megabytes,
    /// and a flat per-entry charge under-reported the largest thing the editor
    /// holds by orders of magnitude. See `EditCommand::memory_bytes`.
    pub fn undo_bytes(&self) -> usize {
        self.history.memory_bytes()
    }

    /// Undo the most recent transaction. Returns whether anything was undone.
    pub fn undo(&mut self) -> bool {
        self.history.commit();
        if let Some(txn) = self.history.take_undo() {
            for cmd in txn.commands.iter().rev() {
                cmd.revert(self);
            }
            self.history.push_redo(txn);
            // IB-13: an undo replays whole `EditCommand` inverses, which can
            // create, delete and reparent — so the projection resyncs.
            self.version += 1;
            self.scope = None;
            true
        } else {
            false
        }
    }

    /// Redo the most recently undone transaction.
    pub fn redo(&mut self) -> bool {
        if let Some(txn) = self.history.take_redo() {
            for cmd in txn.commands.iter() {
                cmd.apply(self);
            }
            self.history.push_undo(txn);
            self.version += 1;
            self.scope = None;
            true
        } else {
            false
        }
    }

    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    // ── snapshot ─────────────────────────────────────────────────────────

    /// Project the world to a full [`SceneSnapshot`] (propagates first so
    /// effective visibility + world transforms are current).
    pub fn snapshot(&mut self) -> SceneSnapshot {
        self.world.propagate();

        // **One pass for the whole hierarchy** (lens 3, P23). `node_of` used to
        // derive each node's `children` by scanning the ENTIRE `order` list and
        // resolving `entity_of`/`parent_of`/`guid_of` for every candidate — so a
        // snapshot was O(n^2) entity lookups, and a snapshot is what every
        // `world://delta` costs, on every gizmo-drag mouse-move and every sculpt
        // dab. Measured on this machine (release, `tests/delta_cost_bench.rs`):
        // 13.1 ms at 1 000 entities, 338.0 ms at 5 000, **3 277.5 ms at 15 000**.
        // The lens filed it as ~30 000 string allocations; the strings were never
        // the term.
        //
        // The parent link is read once per entity here, and creation order is
        // preserved because `order` is walked in order.
        let (children_of, roots) = self.hierarchy_index();

        let nodes: Vec<SceneNode> = self
            .order
            .iter()
            .filter_map(|&guid| self.node_of(guid, &children_of))
            .collect();

        // A snapshot IS a projection: the frontend that received it holds exactly
        // these guids, so the next delta must diff against them. Recording it
        // here is what lets `scene_snapshot`/`scene_open`/`scene_new` hand the
        // client a full tree and the very next `world://delta` be scoped.
        self.projected = nodes
            .iter()
            .filter_map(|n| n.guid.parse::<Uuid>().ok())
            .collect();
        self.roots_cache = roots.clone();
        self.scope = Some(BTreeSet::new());

        SceneSnapshot {
            version: self.version,
            roots,
            nodes,
            selection: self.selection.iter().map(|g| g.to_string()).collect(),
            dirty: self.dirty,
            title: self.title.clone(),
            can_undo: self.history.can_undo(),
            can_redo: self.history.can_redo(),
            undo_label: self.history.undo_label().map(str::to_string),
            redo_label: self.history.redo_label().map(str::to_string),
        }
    }

    /// **What changed since the last projection** (IB-13) — the door
    /// `world://delta` goes through.
    ///
    /// # The measurement this exists for
    ///
    /// Every `world://delta` used to cost a full [`snapshot`](Self::snapshot)
    /// *plus* a full `diff` of it against a retained copy, and one is emitted on
    /// every gizmo-drag mouse-move, every sculpt dab and every click-select.
    /// Measured on this machine (release, `tests/delta_cost_bench.rs`), moving
    /// **one** entity:
    ///
    /// | entities | snapshot | diff | round trip |
    /// |---|---|---|---|
    /// | 15 000 | 3.173 ms | 2.251 ms | 5.423 ms |
    /// | 50 000 | 11.802 ms | 7.707 ms | 19.508 ms |
    /// | 100 000 | 24.117 ms | 19.999 ms | **44.116 ms** |
    ///
    /// The certification's IB-13 named 34 ms at 100 000 and counted only the
    /// snapshot half; the round trip is worse, and it is past the 33 ms frame
    /// tripwire *before anything renders*.
    ///
    /// # The scope, and the contract that makes it safe
    ///
    /// A mutation declares what it moved. `touch` means "I do not know" and
    /// costs a full projection; `touch_at` names the guids and costs
    /// `O(named)`. (Both are `pub(crate)`, so they are named rather than linked
    /// — a public doc that links a private item is a rustdoc warning and a dead
    /// link on the rendered page.) A scope is a **conservative union**: one
    /// `touch` anywhere in a batch widens the whole batch to everything, so
    /// converting a call site is a strict improvement and forgetting to convert
    /// one is only slow.
    ///
    /// **`touch_at` may only be used for changes that cannot move an entity in
    /// the hierarchy**, because a `SceneNode`'s `children` and the snapshot's
    /// `roots` are derived from *other* entities' parent links. Create, delete
    /// and reparent must use `touch`. That is the whole invariant, it is stated
    /// on `touch_at`, and `a_reparent_reaches_the_delta_through_every_node_it_
    /// moves` is what measures it.
    ///
    /// The removal half needs no journal: the doc remembers the guid **set** it
    /// last published (`projected`), so a full projection derives `removed` by
    /// difference and a scoped one by asking whether each named guid still
    /// exists. A set of 100 000 uuids is 1.6 MB against the retained snapshot it
    /// replaces, which was the nodes *and* their strings.
    pub fn project_delta(&mut self) -> SceneDelta {
        self.world.propagate();
        let scope = self.scope.replace(BTreeSet::new());
        let tail_selection: Vec<String> = self.selection.iter().map(|g| g.to_string()).collect();
        // Only a full projection can have moved the roots — a scoped change is a
        // rename, a visibility or a property write by contract, none of which
        // re-parents anything. Cloning the list anyway cost 3.496 ms of an
        // otherwise 0.03 ms delta at 100 000 entities. Read AFTER the walk, which
        // is what refreshes the cache.
        let was_full = scope.is_none();

        let (added, updated, removed) = match scope {
            // Everything: the same walk `snapshot` performs, and the roots cache
            // is refreshed from it rather than by a second pass.
            None => {
                let (children_of, roots) = self.hierarchy_index();
                self.roots_cache = roots;
                let mut added = Vec::new();
                let mut updated = Vec::new();
                let mut live: BTreeSet<Uuid> = BTreeSet::new();
                for &guid in &self.order {
                    let Some(node) = self.node_of(guid, &children_of) else {
                        continue;
                    };
                    live.insert(guid);
                    if self.projected.contains(&guid) {
                        updated.push(node);
                    } else {
                        added.push(node);
                    }
                }
                let removed: Vec<String> = self
                    .projected
                    .difference(&live)
                    .map(|g| g.to_string())
                    .collect();
                self.projected = live;
                (added, updated, removed)
            }
            // Only what was named. `children_of` is built for the named guids
            // alone — `EcsWorld::children_of` is a per-entity read, so the index
            // costs O(children of the named), not O(world) — and it is **ranked**
            // (I3 audit), because the world's `Children` is insertion order and a
            // `SceneNode` states creation order.
            Some(named) => {
                let mut children_of: HashMap<Uuid, Vec<String>> = HashMap::new();
                for &guid in &named {
                    let kids = self.ranked_children(guid);
                    if !kids.is_empty() {
                        children_of.insert(guid, kids);
                    }
                }
                let mut added = Vec::new();
                let mut updated = Vec::new();
                let mut removed = Vec::new();
                for guid in named {
                    match self.node_of(guid, &children_of) {
                        Some(node) => {
                            if self.projected.insert(guid) {
                                added.push(node);
                            } else {
                                updated.push(node);
                            }
                        }
                        None => {
                            if self.projected.remove(&guid) {
                                removed.push(guid.to_string());
                            }
                        }
                    }
                }
                (added, updated, removed)
            }
        };

        SceneDelta {
            version: self.version,
            added,
            removed,
            updated,
            roots: was_full.then(|| self.roots_cache.clone()),
            selection: tail_selection,
            dirty: self.dirty,
            title: self.title.clone(),
            can_undo: self.history.can_undo(),
            can_redo: self.history.can_redo(),
            undo_label: self.history.undo_label().map(str::to_string),
            redo_label: self.history.redo_label().map(str::to_string),
        }
    }

    /// The parent→children index and the root list, in **one** walk of `order`.
    ///
    /// Shared by [`snapshot`](Self::snapshot) and the full arm of
    /// [`project_delta`](Self::project_delta) so the two cannot build the
    /// hierarchy differently — a snapshot and the delta that follows it
    /// disagreeing about who owns a child is a tree the Outliner cannot draw.
    ///
    /// It also **refreshes [`order_rank`](Self::order_rank)**, because the same
    /// walk is what defines creation rank and the scoped projection has no other
    /// way to state a child list in the order this one does (I3 audit).
    fn hierarchy_index(&mut self) -> (HashMap<Uuid, Vec<String>>, Vec<String>) {
        let mut children_of: HashMap<Uuid, Vec<String>> = HashMap::new();
        let mut roots: Vec<String> = Vec::new();
        let mut rank: HashMap<Uuid, u32> = HashMap::with_capacity(self.order.len());
        for (i, &g) in self.order.iter().enumerate() {
            let Some(e) = self.world.entity_of(g) else {
                continue;
            };
            rank.insert(g, i as u32);
            match self.world.parent_of(e).and_then(|p| self.world.guid_of(p)) {
                Some(parent) => children_of.entry(parent).or_default().push(g.to_string()),
                None => roots.push(g.to_string()),
            }
        }
        self.order_rank = rank;
        (children_of, roots)
    }

    /// `guid`'s children as a [`SceneNode`] states them: **creation order**, the
    /// order [`hierarchy_index`](Self::hierarchy_index) produces (I3 audit).
    ///
    /// The world's own `Children` is insertion order, so this sorts by
    /// [`order_rank`](Self::order_rank) — `O(children log children)`, which is
    /// `O(1)` for the leaf a gizmo drag names. A guid the rank does not know is
    /// one no projection has published; it sorts last, by its text, so the
    /// answer is still a function of the content.
    fn ranked_children(&self, guid: Uuid) -> Vec<String> {
        let Some(e) = self.world.entity_of(guid) else {
            return Vec::new();
        };
        let mut kids: Vec<(u32, String)> = self
            .world
            .children_of(e)
            .into_iter()
            .filter_map(|c| self.world.guid_of(c))
            .map(|g| {
                (
                    self.order_rank.get(&g).copied().unwrap_or(u32::MAX),
                    g.to_string(),
                )
            })
            .collect();
        kids.sort_unstable();
        kids.into_iter().map(|(_, s)| s).collect()
    }

    /// `children_of` is [`snapshot`](Self::snapshot)'s single-pass parent index.
    /// Passed in rather than rebuilt because rebuilding it per node is what made
    /// a 15 000-entity snapshot take three and a quarter seconds.
    fn node_of(&self, guid: Uuid, children_of: &HashMap<Uuid, Vec<String>>) -> Option<SceneNode> {
        let e = self.world.entity_of(guid)?;
        let name = self.world.name_of(e).unwrap_or("").to_string();
        let visible = self
            .world
            .world()
            .get::<Visibility>(e)
            .map(|v| v.visible)
            .unwrap_or(true);
        let effective_visible = self
            .world
            .world()
            .get::<ComputedVisibility>(e)
            .map(|c| c.0)
            .unwrap_or(true);
        let parent = self
            .world
            .parent_of(e)
            .and_then(|p| self.world.guid_of(p))
            .map(|g| g.to_string());
        // Children in creation order — from the index built once per snapshot.
        let children: Vec<String> = children_of.get(&guid).cloned().unwrap_or_default();

        Some(SceneNode {
            guid: guid.to_string(),
            name,
            kind: kind_of(&self.world, e),
            visible,
            effective_visible,
            parent,
            children,
        })
    }
}

/// UE-style type-column label from the components an entity carries.
fn kind_of(world: &EcsWorld, e: Entity) -> String {
    let w = world.world();
    if let Some(light) = w.get::<Light>(e) {
        return match light.kind {
            LightKind::Directional => "Directional Light",
            LightKind::Point => "Point Light",
            LightKind::Spot => "Spot Light",
        }
        .to_string();
    }
    if w.get::<Camera>(e).is_some() {
        return "Camera".to_string();
    }
    // Before Wave E this arm did not exist, so `kind_of` reported every rigged
    // character as "Static Mesh" (or "Actor") and the Outliner's type column
    // could not tell a prop from a character. Checked BEFORE `MeshRef` because a
    // character generated by the P24.5 wizard carries both.
    if w.get::<SkeletalMesh>(e).is_some() {
        return "Skeletal Mesh".to_string();
    }
    if w.get::<MeshRef>(e).is_some() {
        return "Static Mesh".to_string();
    }
    if w.get::<Sprite>(e).is_some() {
        return "Sprite".to_string();
    }
    if w.get::<Tilemap>(e).is_some() {
        return "Tilemap".to_string();
    }
    if w.get::<NineSlice>(e).is_some() {
        return "Nine-Slice".to_string();
    }
    if w.get::<Text2D>(e).is_some() {
        return "Text".to_string();
    }
    if w.get::<Light2D>(e).is_some() {
        return "2D Light".to_string();
    }
    if w.get::<Terrain>(e).is_some() {
        return "Terrain".to_string();
    }
    if let Some(vol) = w.get::<Volume>(e) {
        return match vol.kind {
            VolumeKind::Trigger => "Trigger Volume",
            VolumeKind::Blocking => "Blocking Volume",
        }
        .to_string();
    }
    // No renderable payload: a folder if it has children, else a plain actor.
    if !world.children_of(e).is_empty() {
        "Folder".to_string()
    } else {
        "Actor".to_string()
    }
}

fn default_name(kind: SpawnKind) -> String {
    match kind {
        SpawnKind::Empty => "Empty",
        SpawnKind::Cube => "Cube",
        SpawnKind::Sphere => "Sphere",
        SpawnKind::Plane => "Plane",
        SpawnKind::Cylinder => "Cylinder",
        SpawnKind::Cone => "Cone",
        SpawnKind::DirectionalLight => "DirectionalLight",
        SpawnKind::PointLight => "PointLight",
        SpawnKind::SpotLight => "SpotLight",
        SpawnKind::Camera => "Camera",
        SpawnKind::Sprite => "Sprite",
        SpawnKind::Tilemap => "Tilemap",
        SpawnKind::Text2d => "Text",
        SpawnKind::NineSlice => "NineSlice",
        SpawnKind::Light2d => "Light2D",
        SpawnKind::Terrain => "Terrain",
        SpawnKind::TriggerVolume => "TriggerVolume",
        SpawnKind::BlockingVolume => "BlockingVolume",
        SpawnKind::Spline => "Spline",
        SpawnKind::Foliage => "Foliage",
    }
    .to_string()
}

/// Insert the components that make an entity the requested kind.
fn attach_kind(world: &mut EcsWorld, entity: Entity, kind: SpawnKind) {
    let w = world.world_mut();
    let primitive = match kind {
        SpawnKind::Cube => Some(Primitive::Cube),
        SpawnKind::Sphere => Some(Primitive::Sphere),
        SpawnKind::Plane => Some(Primitive::Plane),
        SpawnKind::Cylinder => Some(Primitive::Cylinder),
        SpawnKind::Cone => Some(Primitive::Cone),
        _ => None,
    };
    if let Some(primitive) = primitive {
        w.entity_mut(entity).insert((
            MeshRef {
                primitive,
                asset: None,
            },
            Material::default(),
        ));
        return;
    }
    match kind {
        SpawnKind::DirectionalLight => {
            w.entity_mut(entity).insert(Light {
                kind: LightKind::Directional,
                ..Light::default()
            });
        }
        SpawnKind::PointLight => {
            w.entity_mut(entity).insert(Light {
                kind: LightKind::Point,
                ..Light::default()
            });
        }
        SpawnKind::SpotLight => {
            w.entity_mut(entity).insert(Light {
                kind: LightKind::Spot,
                ..Light::default()
            });
        }
        SpawnKind::Camera => {
            w.entity_mut(entity).insert(Camera::default());
        }
        // ── 2D kinds (P8.2b): sensible authorable defaults ────────────────
        SpawnKind::Sprite => {
            w.entity_mut(entity).insert(Sprite::default());
        }
        SpawnKind::Tilemap => {
            // A 4×4 starter atlas so the paint palette is immediately usable.
            w.entity_mut(entity).insert(Tilemap {
                atlas_cols: 4,
                atlas_rows: 4,
                ..Tilemap::default()
            });
        }
        SpawnKind::Text2d => {
            w.entity_mut(entity).insert(Text2D {
                text: "Text".to_string(),
                ..Text2D::default()
            });
        }
        SpawnKind::NineSlice => {
            w.entity_mut(entity).insert(NineSlice::default());
        }
        SpawnKind::Light2d => {
            w.entity_mut(entity).insert(Light2D::default());
        }
        // ── 3D terrain (P10): a small starter sine-hill so it's visible on spawn ─
        SpawnKind::Terrain => {
            let mut terrain = Terrain::configured(64, 1.0);
            let span = terrain.data.tile_span();
            terrain
                .data
                .write_region(glam::DVec2::ZERO, glam::DVec2::splat(span), |x, z| {
                    // **Portable trig, because this writes COMMITTED content**
                    // (the P14 law). These samples land in the author's
                    // `Terrain` component, are serialized into their `.inf_lvl`
                    // and are cooked into a pack — so the heights a level holds
                    // must not depend on which machine spawned the terrain.
                    // `std`'s `sin`/`cos` route through the platform libm and
                    // are entitled to disagree in the last ulp; `psin64`/
                    // `pcos64` are IEEE add/mul/floor only.
                    2.0 * inf_math::psin64(x * 0.1) * inf_math::pcos64(z * 0.1)
                });
            w.entity_mut(entity).insert(terrain);
        }
        // ── Gameplay volumes (E-P4): a box Volume + implicit-static box collider,
        //    NO MeshRef so they stay invisible in PIE by construction. Trigger =
        //    sensor (overlaps only); Blocking = solid. Distinct default tints. ──
        SpawnKind::TriggerVolume | SpawnKind::BlockingVolume => {
            let is_trigger = matches!(kind, SpawnKind::TriggerVolume);
            let (vkind, tint) = if is_trigger {
                (VolumeKind::Trigger, Color::new(1.0, 0.6, 0.1, 1.0))
            } else {
                (VolumeKind::Blocking, Color::new(0.2, 0.5, 1.0, 1.0))
            };
            w.entity_mut(entity).insert((
                Volume { kind: vkind, tint },
                Collider3D {
                    shape_kind: ColliderShape3DKind::Box,
                    half_extents: Vec3d::splat(1.0),
                    sensor: is_trigger,
                    ..Collider3D::default()
                },
            ));
        }
        // ── Utility (E-P5): a default control-point spline. The component
        //    Default supplies the two-point starter path; the viewport draws it
        //    as a polyline and the Details List editor edits the points. ──
        SpawnKind::Spline => {
            w.entity_mut(entity).insert(Spline::default());
        }
        // ── Utility (E-P6): a Foliage scatter seeded with a 1-entry palette so
        //    the brush places something immediately. A green-tinted Cone reads as
        //    a stand-in "tree/shrub" (real mesh-asset palettes are the follow-up);
        //    instances start empty (the brush fills them). ──
        SpawnKind::Foliage => {
            w.entity_mut(entity).insert(Foliage {
                palette: vec![FoliagePaletteEntry {
                    primitive: Primitive::Cone,
                    tint: Color::new(0.30, 0.62, 0.28, 1.0),
                }],
                instances: Vec::new(),
            });
        }
        SpawnKind::Empty
        | SpawnKind::Cube
        | SpawnKind::Sphere
        | SpawnKind::Plane
        | SpawnKind::Cylinder
        | SpawnKind::Cone => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **AN EDITED PCG VOLUME IS A STALE PCG VOLUME** (wave EDIT1, clause 3 --
    /// the arm the audit found missing).
    ///
    /// The commit that added `invalidate_pcg_population` is *titled* for this
    /// behaviour ("an edit makes it stale") and nothing asserted it: the three
    /// arms it shipped are clause 1's, in another crate, about a different door.
    /// So a `write_prop` that stopped dropping the cache -- or a type-path
    /// constant that drifted from `PcgVolume`'s -- would have left an author
    /// looking at the old block after widening it, with a green battery.
    ///
    /// **Both halves, because one of them is not the claim.** "It drops the
    /// population" is satisfied perfectly by a `write_prop` that drops every
    /// volume's population on every write anywhere in the document, which would
    /// make an unrelated slider re-build a city four times a second. The control
    /// is a write to a DIFFERENT component on the SAME entity.
    #[test]
    fn editing_a_pcg_volume_drops_the_population_it_no_longer_describes() {
        use inf_ecs::components::{PcgVolume, ScatteredInstance, ScatteredSurface};

        let populate = |doc: &mut SceneDoc, g: Uuid| {
            let e = doc.world.entity_of(g).expect("entity");
            let one = ScatteredInstance {
                position: DVec3::ZERO,
                rotation: glam::DQuat::IDENTITY,
                scale: 1.0,
                kind: 0,
                mesh: None,
                extent: None,
                glow: 0.0,
                surface: ScatteredSurface::DEFAULT,
            };
            let w = doc.world.world_mut();
            let mut vol = w.get_mut::<PcgVolume>(e).expect("volume");
            vol.set_population(
                vec![one],
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                inf_nav::NavGraph::default(),
                Vec::new(),
                Vec::new(),
            );
        };

        let mut doc = SceneDoc::new();
        let g = doc.create(SpawnKind::Empty, "Block", None);
        {
            let e = doc.world.entity_of(g).expect("entity");
            doc.world.world_mut().entity_mut(e).insert(PcgVolume {
                seed: 7,
                ..PcgVolume::default()
            });
        }
        populate(&mut doc, g);
        assert_eq!(
            doc.pcg_population_of(g),
            Some((1, 0)),
            "the fixture did not put a population there, so nothing below is measuring a drop"
        );

        // THE CLAIM: one of the volume's own fields is written, and the cache
        // that was derived from it goes.
        assert!(doc.write_prop(
            g,
            "inf_ecs::components::PcgVolume",
            "seed",
            &PropValue::Number(9.0)
        ));
        assert_eq!(
            doc.pcg_population_of(g),
            Some((0, 0)),
            "editing a volume's seed left the population it no longer describes in place -- an author who re-seeds a block keeps looking at the old block"
        );

        // THE CONTROL: a write to another component on the same entity must
        // leave it alone.
        populate(&mut doc, g);
        assert_eq!(doc.pcg_population_of(g), Some((1, 0)));
        assert!(doc.write_prop(
            g,
            "inf_ecs::components::Transform",
            "translation",
            &PropValue::Vec3([1.0, 0.0, 0.0])
        ));
        assert_eq!(
            doc.pcg_population_of(g),
            Some((1, 0)),
            "moving the entity dropped its population -- the invalidation is not keyed on the volume's own type path, so every slider in the Details panel rebuilds a city"
        );
    }

    #[test]
    fn create_rename_reparent_delete() {
        let mut doc = SceneDoc::new();
        let a = doc.create(SpawnKind::Empty, "A", None);
        let b = doc.create(SpawnKind::Cube, "", Some(a));
        // default name for a cube
        let snap = doc.snapshot();
        let bn = snap.nodes.iter().find(|n| n.guid == b.to_string()).unwrap();
        assert_eq!(bn.name, "Cube");
        assert_eq!(bn.kind, "Static Mesh");
        assert_eq!(bn.parent.as_deref(), Some(a.to_string().as_str()));

        doc.rename(b, "Crate");
        // Reparent b to root.
        assert!(doc.reparent(b, None));
        let snap = doc.snapshot();
        assert_eq!(snap.roots.len(), 2);

        doc.delete(&[a]);
        let snap = doc.snapshot();
        // a gone, b (now a root) survives.
        assert!(snap.nodes.iter().all(|n| n.guid != a.to_string()));
        assert_eq!(snap.nodes.len(), 1);
    }

    #[test]
    fn apply_material_sets_pbr_and_undoes_as_one_step() {
        let mut doc = SceneDoc::new();
        let cube = doc.create(SpawnKind::Cube, "Cube", None);
        let tp = doc.world().registry().type_path_for("Material").unwrap();

        let bound = Uuid::from_u128(0xFA7E_0026);
        let applied = doc.edit_apply_material(
            &[cube],
            Some(bound),
            [1.0, 0.0, 0.0, 1.0],
            1.0,
            0.2,
            [0.5, 0.0, 0.0],
            "Translucent",
            0.25,
        );
        assert_eq!(applied, 1);
        assert_eq!(
            doc.prop_value(cube, tp, "metallic"),
            Some(PropValue::Number(1.0))
        );
        assert_eq!(
            doc.prop_value(cube, tp, "base_color"),
            Some(PropValue::Color([1.0, 0.0, 0.0, 1.0]))
        );
        // R-P5: blend + alpha_cutoff apply through the same undo step.
        assert!(matches!(
            doc.prop_value(cube, tp, "blend"),
            Some(PropValue::Enum { value, .. }) if value == "Translucent"
        ));
        assert_eq!(
            doc.prop_value(cube, tp, "alpha_cutoff"),
            Some(PropValue::Number(0.25))
        );

        // P26.3b: the BINDING landed, beside the scalars it does not replace.
        // The commit that added it asserted nothing about it — the arm passed
        // `Some(bound)` and never looked.
        assert_eq!(doc.material_asset_of(cube), Some(bound));

        // The field writes collapse into one undo step (back to defaults) — and
        // that ONE step takes the binding with them. A binding recorded outside
        // the transaction would survive here, leaving a surface whose numbers
        // came from one material and whose textures come from another.
        assert!(doc.undo());
        assert_eq!(
            doc.prop_value(cube, tp, "metallic"),
            Some(PropValue::Number(0.0))
        );
        assert_eq!(
            doc.material_asset_of(cube),
            None,
            "one undo reverted the scalars and left the binding — the two are not \
             in one transaction"
        );
        // …and redo restores both, so the command round-trips rather than only
        // reverting.
        assert!(doc.redo());
        assert_eq!(doc.material_asset_of(cube), Some(bound));
        assert_eq!(
            doc.prop_value(cube, tp, "metallic"),
            Some(PropValue::Number(1.0))
        );
    }

    /// **Applying a material this caller cannot name CLEARS the binding**
    /// (P26.3b audit), rather than leaving the previous one standing over the new
    /// scalars.
    ///
    /// The state that would otherwise persist is the exact one
    /// `EditCommand::SetMaterialAsset` was introduced to make impossible: a
    /// surface whose numbers come from the material just applied and whose
    /// textures come from the one before it. Reachable through the shipped UI —
    /// `scene_apply_material` passes `None` when a dropped `.inf_mati`'s parent
    /// chain is broken.
    #[test]
    fn applying_an_unnameable_material_clears_the_binding_it_replaces() {
        let mut doc = SceneDoc::new();
        let cube = doc.create(SpawnKind::Cube, "Cube", None);
        let first = Uuid::from_u128(0xFA7E_0001);

        assert_eq!(
            doc.edit_apply_material(
                &[cube],
                Some(first),
                [1.0, 0.0, 0.0, 1.0],
                1.0,
                0.25,
                [0.0; 3],
                "Opaque",
                0.5,
            ),
            1
        );
        // Anti-vacuity: there is a binding to lose.
        assert_eq!(doc.material_asset_of(cube), Some(first));

        // A second material, with different scalars and no nameable asset.
        assert_eq!(
            doc.edit_apply_material(
                &[cube],
                None,
                [0.0, 0.0, 1.0, 1.0],
                0.0,
                0.75,
                [0.0; 3],
                "Opaque",
                0.5,
            ),
            1
        );
        let tp = doc.world().registry().type_path_for("Material").unwrap();
        assert_eq!(
            doc.prop_value(cube, tp, "roughness"),
            Some(PropValue::Number(0.75)),
            "the second material's scalars did not apply, so the clear below is \
             measuring nothing"
        );
        assert_eq!(
            doc.material_asset_of(cube),
            None,
            "the previous binding survived a material it did not come from — the \
             surface's numbers and its textures are from two different materials"
        );
        // …and the clear is undoable as one step with the scalars it arrived
        // with.
        assert!(doc.undo());
        assert_eq!(doc.material_asset_of(cube), Some(first));
        assert_eq!(
            doc.prop_value(cube, tp, "roughness"),
            Some(PropValue::Number(0.25))
        );
    }

    #[test]
    fn apply_sprite_slice_inserts_and_undoes_as_one_step() {
        let mut doc = SceneDoc::new();
        let e = doc.create(SpawnKind::Empty, "Sprite", None);
        assert!(doc.raw_get_sprite(e).is_none(), "no sprite to start");

        let tex = uuid::Uuid::from_u128(0xABCD);
        let applied =
            doc.edit_apply_sprite_slice(&[e], Some(tex), [0.25, 0.5], [0.5, 1.0], Some([2.0, 1.0]));
        assert_eq!(applied, 1);
        let s = doc.raw_get_sprite(e).expect("sprite inserted");
        assert_eq!(s.texture, Some(tex));
        assert_eq!(
            s.atlas_rect,
            AtlasRect {
                min: Vec2d::new(0.25, 0.5),
                max: Vec2d::new(0.5, 1.0),
            }
        );
        assert_eq!(s.size, Vec2d::new(2.0, 1.0));
        assert_eq!(doc.kind_of_guid(e), "Sprite");

        // One undo removes the whole Sprite (it didn't exist before).
        assert!(doc.undo());
        assert!(doc.raw_get_sprite(e).is_none(), "undo removes the sprite");
        assert!(doc.redo());
        assert_eq!(doc.raw_get_sprite(e).unwrap().texture, Some(tex));
    }

    #[test]
    fn spawns_2d_kinds_with_defaults_and_labels() {
        let mut doc = SceneDoc::new();
        let cases = [
            (SpawnKind::Sprite, "Sprite", "Sprite"),
            (SpawnKind::Tilemap, "Tilemap", "Tilemap"),
            (SpawnKind::Text2d, "Text", "Text"),
            (SpawnKind::NineSlice, "NineSlice", "Nine-Slice"),
            (SpawnKind::Light2d, "Light2D", "2D Light"),
        ];
        for (kind, name, label) in cases {
            let g = doc.create(kind, "", None);
            assert_eq!(doc.display_name(g), name, "default name for {kind:?}");
            assert_eq!(doc.kind_of_guid(g), label, "type label for {kind:?}");
        }
    }

    #[test]
    fn spawned_2d_entity_round_trips_through_undo() {
        // A spawned tilemap (with a starter atlas) survives delete→undo intact.
        let mut doc = SceneDoc::new();
        let g = doc.edit_create(SpawnKind::Tilemap, "Map", None);
        doc.edit_paint_tiles(g, &[(0, 0, 2), (1, 1, 3)]);
        assert_eq!(
            crate::scene::tilemap::build_dto(&doc, g)
                .unwrap()
                .cells
                .len(),
            2
        );

        doc.edit_delete(&[g]);
        assert!(doc.entity_of(g).is_none());

        // Undo delete → undo paint. The tilemap (and its painted cells) return.
        assert!(doc.undo()); // undo delete
        let dto = crate::scene::tilemap::build_dto(&doc, g).expect("tilemap restored");
        assert_eq!(dto.cells.len(), 2, "painted tiles survive delete→undo");
        assert_eq!(dto.atlas_cols, 4);
    }

    #[test]
    fn spawns_terrain_with_starter_hill_and_undo_removes_it() {
        // P10: a spawned Terrain carries a non-empty starter heightfield, reports
        // the "Terrain" type label, and undoing the spawn removes it.
        let mut doc = SceneDoc::new();
        let g = doc.edit_create(SpawnKind::Terrain, "", None);
        assert_eq!(doc.display_name(g), "Terrain", "default name");
        assert_eq!(doc.kind_of_guid(g), "Terrain", "type label");

        // The starter sine-hill authored at least one tile (visible immediately).
        let e = doc.entity_of(g).unwrap();
        let terrain = doc.world().world().get::<Terrain>(e).unwrap();
        assert!(
            !terrain.data.is_empty(),
            "starter terrain has authored tiles"
        );

        // Undo the spawn → the terrain entity is gone.
        assert!(doc.undo(), "undo removes the spawned terrain");
        assert!(doc.entity_of(g).is_none(), "terrain despawned by undo");
    }

    #[test]
    fn spawns_trigger_and_blocking_volumes() {
        // E-P4: each volume kind gets a Volume + box Collider3D (sensor iff
        // Trigger), the expected default name/label, and NO MeshRef (invisible in
        // PIE). Distinct default tints per kind.
        let mut doc = SceneDoc::new();
        let cases = [
            (
                SpawnKind::TriggerVolume,
                "TriggerVolume",
                "Trigger Volume",
                VolumeKind::Trigger,
                true,
                Color::new(1.0, 0.6, 0.1, 1.0),
            ),
            (
                SpawnKind::BlockingVolume,
                "BlockingVolume",
                "Blocking Volume",
                VolumeKind::Blocking,
                false,
                Color::new(0.2, 0.5, 1.0, 1.0),
            ),
        ];
        for (kind, name, label, vkind, sensor, tint) in cases {
            let g = doc.create(kind, "", None);
            assert_eq!(doc.display_name(g), name, "default name for {kind:?}");
            assert_eq!(doc.kind_of_guid(g), label, "type label for {kind:?}");

            let e = doc.entity_of(g).unwrap();
            let w = doc.world().world();
            let vol = w.get::<Volume>(e).expect("Volume component present");
            assert_eq!(vol.kind, vkind, "volume kind for {kind:?}");
            assert_eq!(vol.tint, tint, "default tint for {kind:?}");

            let col = w.get::<Collider3D>(e).expect("Collider3D present");
            assert_eq!(col.shape_kind, ColliderShape3DKind::Box);
            assert_eq!(col.half_extents, Vec3d::splat(1.0));
            assert_eq!(col.sensor, sensor, "sensor flag for {kind:?}");

            assert!(
                w.get::<MeshRef>(e).is_none(),
                "volumes carry no MeshRef (invisible in PIE) for {kind:?}"
            );
        }
    }

    #[test]
    fn sculpt_stroke_is_one_undo_step_and_round_trips() {
        use glam::DVec2;
        use inf_terrain::{BrushOp, BrushParams, Stroke};

        let mut doc = SceneDoc::new();
        let g = doc.edit_create(SpawnKind::Terrain, "", None);
        let e = doc.entity_of(g).unwrap();

        // Probe a point inside the starter terrain and record its height.
        let (data, _origin) = doc.terrain_data_and_origin(g).unwrap();
        let center = {
            let span = data.tile_span();
            DVec2::new(span * 0.5, span * 0.5)
        };
        let before = data.height_at(center).unwrap();

        // A raise stroke of a few overlapping dabs → one merged undo step.
        let mut stroke = Stroke::begin();
        let params = BrushParams::new(center, 6.0, 3.0);
        for _ in 0..3 {
            doc.sculpt_apply_dab(g, &mut stroke, BrushOp::Raise, params);
        }
        assert!(
            doc.edit_commit_sculpt(g, stroke),
            "non-empty stroke records"
        );

        let raised = doc
            .world()
            .world()
            .get::<Terrain>(e)
            .unwrap()
            .data
            .height_at(center)
            .unwrap();
        assert!(raised > before + 1.0, "raise lifted the surface: {raised}");

        // One undo reverts the whole stroke (not per-dab) back to byte-identical.
        assert!(doc.undo(), "one undo step for the stroke");
        let undone = doc
            .world()
            .world()
            .get::<Terrain>(e)
            .unwrap()
            .data
            .height_at(center)
            .unwrap();
        assert!((undone - before).abs() < 1e-6, "undo restored height");

        // Redo replays it exactly.
        assert!(doc.redo(), "redo the stroke");
        let redone = doc
            .world()
            .world()
            .get::<Terrain>(e)
            .unwrap()
            .data
            .height_at(center)
            .unwrap();
        assert!((redone - raised).abs() < 1e-6, "redo replayed the stroke");
    }

    #[test]
    fn foliage_paint_undo_redo_restores_exact_instances() {
        let mut doc = SceneDoc::new();
        let g = doc.edit_create(SpawnKind::Foliage, "Foliage", None);
        assert!(doc.has_foliage(g));
        assert!(
            doc.foliage_instances(g).unwrap().is_empty(),
            "seeded with an empty instance list"
        );

        // Simulate a scatter stroke: live-append, then commit as ONE undo step.
        let added: Vec<FoliageInstance> = (0..4)
            .map(|i| FoliageInstance {
                position: Vec3d::new(i as f64, 0.0, 0.0),
                rotation: Vec3d::new(0.0, 10.0 * i as f64, 0.0),
                scale: 1.0,
                kind: 0,
            })
            .collect();
        doc.foliage_append(g, &added);
        assert_eq!(doc.foliage_instances(g).unwrap(), added);
        assert!(doc.edit_commit_foliage(g, added.clone(), Vec::new()));

        assert!(doc.undo(), "one undo step for the stroke");
        assert!(
            doc.foliage_instances(g).unwrap().is_empty(),
            "undo pops the whole scatter"
        );
        assert!(doc.redo(), "redo replays the stroke");
        assert_eq!(
            doc.foliage_instances(g).unwrap(),
            added,
            "redo restores the exact instance vec"
        );
    }

    #[test]
    fn foliage_erase_round_trips_exact_order() {
        let mut doc = SceneDoc::new();
        let g = doc.edit_create(SpawnKind::Foliage, "Foliage", None);
        let original: Vec<FoliageInstance> = (0..5)
            .map(|i| FoliageInstance {
                position: Vec3d::new(i as f64, 0.0, 0.0),
                rotation: Vec3d::ZERO,
                scale: 1.0,
                kind: 0,
            })
            .collect();
        doc.foliage_set_instances(g, original.clone());

        // Erase indices 1 and 3: rebuild the live list + record the removed pairs
        // (against the ORIGINAL indices, as the brush does).
        let removed = vec![(1usize, original[1]), (3usize, original[3])];
        let kept: Vec<FoliageInstance> = original
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != 1 && *i != 3)
            .map(|(_, x)| *x)
            .collect();
        doc.foliage_set_instances(g, kept.clone());
        assert!(doc.edit_commit_foliage(g, Vec::new(), removed));

        assert_eq!(doc.foliage_instances(g).unwrap(), kept);
        assert!(doc.undo(), "undo the erase");
        assert_eq!(
            doc.foliage_instances(g).unwrap(),
            original,
            "erase undo restores the pre-stroke list in exact order"
        );
        assert!(doc.redo(), "redo the erase");
        assert_eq!(doc.foliage_instances(g).unwrap(), kept);
    }

    #[test]
    fn edit_commit_foliage_ignores_empty_strokes() {
        let mut doc = SceneDoc::new();
        let g = doc.edit_create(SpawnKind::Foliage, "Foliage", None);
        assert!(!doc.edit_commit_foliage(g, Vec::new(), Vec::new()));
        // The only undo entry is the entity Create — the empty stroke recorded
        // none, so a second undo has nothing to pop.
        assert!(doc.undo(), "undo the create");
        assert!(!doc.undo(), "an empty stroke records nothing");
    }

    #[test]
    fn paint_stroke_is_one_undo_step_and_byte_identical_revert() {
        use glam::DVec2;
        use inf_terrain::{BrushParams, SplatStroke};

        let mut doc = SceneDoc::new();
        let g = doc.edit_create(SpawnKind::Terrain, "", None);
        let e = doc.entity_of(g).unwrap();

        // Serialize the pristine (unpainted) terrain data for a byte-identical
        // revert check.
        let pristine =
            serde_json::to_string(&doc.world().world().get::<Terrain>(e).unwrap().data).unwrap();

        let (data, _origin) = doc.terrain_data_and_origin(g).unwrap();
        let center = {
            let span = data.tile_span();
            DVec2::new(span * 0.5, span * 0.5)
        };

        // A paint stroke of a few overlapping dabs onto layer 2 → one undo step.
        let mut stroke = SplatStroke::begin(2);
        let params = BrushParams::new(center, 6.0, 1.0);
        for _ in 0..3 {
            doc.paint_apply_dab(g, &mut stroke, params);
        }
        assert!(doc.edit_commit_paint(g, stroke), "non-empty paint records");

        // Some tile is now painted (its weight buffer materialized).
        let painted_any = doc
            .world()
            .world()
            .get::<Terrain>(e)
            .unwrap()
            .data
            .tiles()
            .any(|(_, t)| !t.weights_are_default());
        assert!(painted_any, "paint materialized a weight buffer");
        let painted =
            serde_json::to_string(&doc.world().world().get::<Terrain>(e).unwrap().data).unwrap();

        // One undo → byte-identical to the pristine terrain (materialized buffers
        // dropped back to the sparse default).
        assert!(doc.undo(), "one undo step for the paint stroke");
        let undone =
            serde_json::to_string(&doc.world().world().get::<Terrain>(e).unwrap().data).unwrap();
        assert_eq!(
            undone, pristine,
            "paint undo restored byte-identical terrain"
        );

        // Redo replays the painted weights exactly.
        assert!(doc.redo(), "redo the paint stroke");
        let redone =
            serde_json::to_string(&doc.world().world().get::<Terrain>(e).unwrap().data).unwrap();
        assert_eq!(redone, painted, "redo replayed the paint stroke");
    }

    // ── duplicate / clipboard (editor seams) ─────────────────────────────

    #[test]
    fn duplicate_single_entity_copies_components_fresh_guid_sibling_and_suffix() {
        let mut doc = SceneDoc::new();
        let parent = doc.edit_create(SpawnKind::Empty, "Parent", None);
        let cube = doc.edit_create(SpawnKind::Cube, "Box", Some(parent));

        let copies = doc.edit_duplicate(&[cube]);
        assert_eq!(copies.len(), 1);
        let copy = copies[0];
        assert_ne!(copy, cube, "fresh guid");

        // The copy is a SIBLING under the same parent as the original.
        let ce = doc.entity_of(copy).unwrap();
        let cp = doc
            .world()
            .parent_of(ce)
            .and_then(|p| doc.world().guid_of(p));
        assert_eq!(cp, Some(parent), "copy sits under the same parent");

        // Components copied → still a Static Mesh; name " Copy"-suffixed.
        assert_eq!(doc.kind_of_guid(copy), "Static Mesh");
        assert_eq!(doc.display_name(copy), "Box Copy");

        // A second duplicate bumps the suffix number.
        let more = doc.edit_duplicate(&[cube]);
        assert_eq!(doc.display_name(more[0]), "Box Copy 2");
    }

    #[test]
    fn duplicate_subtree_preserves_internal_links_with_remapped_guids() {
        let mut doc = SceneDoc::new();
        let a = doc.edit_create(SpawnKind::Empty, "A", None);
        let b = doc.edit_create(SpawnKind::Cube, "B", Some(a));
        let c = doc.edit_create(SpawnKind::Sphere, "C", Some(b));

        let roots = doc.edit_duplicate(&[a]);
        assert_eq!(roots.len(), 1);
        let a2 = roots[0].to_string();
        let snap = doc.snapshot();

        // Copied root "A Copy" has one child "B".
        let a2n = snap.nodes.iter().find(|n| n.guid == a2).unwrap();
        assert_eq!(a2n.name, "A Copy");
        assert_eq!(a2n.children.len(), 1);
        let b2 = a2n.children[0].clone();
        let b2n = snap.nodes.iter().find(|n| n.guid == b2).unwrap();
        assert_eq!(b2n.name, "B");

        // The copied C hangs off the copied B — the internal link was remapped.
        assert_eq!(b2n.children.len(), 1);
        let c2 = b2n.children[0].clone();
        let c2n = snap.nodes.iter().find(|n| n.guid == c2).unwrap();
        assert_eq!(c2n.name, "C");
        assert_eq!(c2n.parent.as_deref(), Some(b2.as_str()));

        // Fresh guids throughout.
        assert_ne!(b2, b.to_string());
        assert_ne!(c2, c.to_string());
    }

    #[test]
    fn duplicate_is_one_undo_step() {
        let mut doc = SceneDoc::new();
        let a = doc.edit_create(SpawnKind::Empty, "A", None);
        let _b = doc.edit_create(SpawnKind::Cube, "B", Some(a));
        assert_eq!(doc.snapshot().nodes.len(), 2);

        let roots = doc.edit_duplicate(&[a]);
        assert_eq!(roots.len(), 1);
        assert_eq!(doc.snapshot().nodes.len(), 4, "root + child duplicated");

        assert!(doc.undo(), "one undo removes the whole duplicate");
        assert_eq!(doc.snapshot().nodes.len(), 2, "back to the original two");
        assert!(doc.redo(), "redo restores all copies in one step");
        assert_eq!(doc.snapshot().nodes.len(), 4);
    }

    #[test]
    fn nested_selection_is_deduped_on_duplicate() {
        // Selecting BOTH a parent and its child duplicates the subtree once.
        let mut doc = SceneDoc::new();
        let a = doc.edit_create(SpawnKind::Empty, "A", None);
        let b = doc.edit_create(SpawnKind::Cube, "B", Some(a));

        let roots = doc.edit_duplicate(&[a, b]);
        assert_eq!(roots.len(), 1, "nested child folds into the parent copy");
        assert_eq!(doc.snapshot().nodes.len(), 4, "exactly one extra subtree");
    }

    #[test]
    fn copy_delete_paste_restores_equivalent_subtree_with_fresh_guids() {
        let mut doc = SceneDoc::new();
        let a = doc.edit_create(SpawnKind::Empty, "A", None);
        let b = doc.edit_create(SpawnKind::Cube, "B", Some(a));

        // Copy snapshots the clipboard records; delete then removes the originals.
        let clip = doc.collect_subtree_records(&[a]);
        assert_eq!(clip.len(), 2);
        doc.edit_delete(&[a]);
        assert!(doc.snapshot().nodes.is_empty());

        // Paste re-materializes an equivalent subtree with fresh guids.
        let roots = doc.edit_paste_records(&clip);
        assert_eq!(roots.len(), 1);
        let snap = doc.snapshot();
        assert_eq!(snap.nodes.len(), 2, "subtree restored");
        assert!(
            snap.nodes
                .iter()
                .all(|n| n.guid != a.to_string() && n.guid != b.to_string()),
            "every pasted node has a fresh guid"
        );
        // Names preserved (paste does NOT add a Copy suffix).
        assert!(snap.nodes.iter().any(|n| n.name == "A"));
        assert!(snap.nodes.iter().any(|n| n.name == "B"));
    }

    #[test]
    fn cut_then_paste_is_move_with_fresh_guids() {
        let mut doc = SceneDoc::new();
        let a = doc.edit_create(SpawnKind::Cube, "A", None);

        // Cut = snapshot clipboard + delete.
        let clip = doc.collect_subtree_records(&[a]);
        doc.edit_delete(&[a]);
        assert!(doc.entity_of(a).is_none());

        // Paste = re-spawn with a fresh guid (a move, identity not preserved).
        let roots = doc.edit_paste_records(&clip);
        assert_eq!(roots.len(), 1);
        assert_ne!(roots[0], a, "pasted entity has a fresh guid");
        assert_eq!(doc.display_name(roots[0]), "A");
        assert_eq!(doc.snapshot().nodes.len(), 1);
    }

    #[test]
    fn deleting_parent_removes_children() {
        let mut doc = SceneDoc::new();
        let a = doc.create(SpawnKind::Empty, "A", None);
        let _b = doc.create(SpawnKind::Cube, "B", Some(a));
        doc.delete(&[a]);
        assert!(doc.snapshot().nodes.is_empty());
    }

    #[test]
    fn selection_additive_toggles() {
        let mut doc = SceneDoc::new();
        let a = doc.create(SpawnKind::Empty, "A", None);
        let b = doc.create(SpawnKind::Empty, "B", None);
        doc.select(&[a], false);
        doc.select(&[b], true);
        assert_eq!(doc.selection(), &[a, b]);
        doc.select(&[a], true); // toggle a off
        assert_eq!(doc.selection(), &[b]);
    }

    /// **Shift APPENDS; Ctrl TOGGLES** (Wave E) — two verbs, deliberately not
    /// one. Overloading `additive` would make a shift-click across a group
    /// deselect the half that was already in it, which is the opposite of what
    /// the gesture means everywhere else in every editor.
    #[test]
    fn selection_append_never_removes() {
        let mut doc = SceneDoc::new();
        let a = doc.create(SpawnKind::Empty, "A", None);
        let b = doc.create(SpawnKind::Empty, "B", None);
        let c = doc.create(SpawnKind::Empty, "C", None);
        doc.select(&[a], false);
        doc.select_append(&[b]);
        assert_eq!(doc.selection(), &[a, b]);

        // The distinction: appending something already selected is a NO-OP,
        // where `select(.., true)` would have removed it.
        doc.select_append(&[a]);
        assert_eq!(doc.selection(), &[a, b], "append must never toggle off");

        // …and the primary (the entry the gizmo centres on) does not move.
        doc.select_append(&[c]);
        assert_eq!(doc.selection(), &[a, b, c]);
        assert_eq!(doc.selection()[0], a);

        // Unknown guids are dropped, exactly as `select` drops them.
        let before = doc.version;
        doc.select_append(&[uuid::Uuid::from_u128(0xDEAD)]);
        assert_eq!(doc.selection(), &[a, b, c]);
        assert_eq!(
            before, doc.version,
            "a no-op append does not bump the version"
        );
    }

    #[test]
    fn effective_visibility_follows_ancestors() {
        let mut doc = SceneDoc::new();
        let a = doc.create(SpawnKind::Empty, "A", None);
        let b = doc.create(SpawnKind::Cube, "B", Some(a));
        doc.set_visible(a, false);
        let snap = doc.snapshot();
        let bn = snap.nodes.iter().find(|n| n.guid == b.to_string()).unwrap();
        assert!(bn.visible, "b's own toggle is still on");
        assert!(!bn.effective_visible, "b hidden because A is hidden");
    }

    /// The terrain's **saved bytes** — its `.inf_lvl` bincode encoding, which
    /// covers heights, splat weights and the P19.1 data maps in one shot.
    ///
    /// Comparing these rather than a field-by-field hash is the stronger claim
    /// and the one that matters: "undo restored it" means *the file it would
    /// write is the file it would have written*, with no layer left out because
    /// the test's hash forgot about it.
    fn saved_bytes(data: &inf_terrain::TerrainData) -> Vec<u8> {
        bincode::serde::encode_to_vec(data, bincode::config::standard())
            .expect("a terrain always encodes")
    }

    /// **THE UNDO GATE (P19.1).** An erosion bake writes two layers through two
    /// separate `EditCommand`s, and the claim that pays for that split is that
    /// **one** Ctrl+Z restores both, byte-identically. Nothing exercised the
    /// transaction before this: the delta types were tested at the `TerrainData`
    /// layer, but the editor's grouping — `history.begin` / two `record`s /
    /// `commit`, then `undo` reverting them in reverse order — was not.
    #[test]
    fn an_erosion_bake_is_one_undo_step_covering_heights_and_maps() {
        let mut doc = SceneDoc::new();
        let g = doc.edit_create(SpawnKind::Terrain, "", None);
        let undo_depth_before = doc.history.undo_len();

        let (data, _) = doc.terrain_data_and_origin(g).unwrap();
        let before = saved_bytes(data);
        assert!(
            data.data_maps_are_default(),
            "a fresh terrain has never been eroded"
        );
        let (min, max) = doc
            .terrain_bounds(g)
            .expect("the starter terrain has bounds");

        let params = inf_terrain::ErosionParams {
            rain_rate: 0.05,
            ..inf_terrain::ErosionParams::default()
        };
        let report = doc
            .edit_erode_region(g, min, max, 0, |region| {
                inf_terrain::erode(region, &params, 40);
            })
            .expect("the entity has a terrain");

        // The bake really moved both layers …
        assert!(report.cells_changed > 0, "the bake changed no heights");
        // The map layer covers at least every cell the heights moved: a cell can
        // accumulate flow without its height budging, but the reverse cannot
        // happen — a height only moves through the erode/deposit pass, which
        // writes wear or deposition in the same breath. (On ordinary terrain the
        // two counts coincide, because `min_tilt` gives every wet cell a non-zero
        // capacity and so something to erode or settle.)
        assert!(
            report.map_cells_changed >= report.cells_changed,
            "every height the bake moved must have moved a data map too ({} maps vs \
             {} heights)",
            report.map_cells_changed,
            report.cells_changed
        );
        let eroded = {
            let (data, _) = doc.terrain_data_and_origin(g).unwrap();
            assert!(!data.data_maps_are_default(), "the maps must have landed");
            saved_bytes(data)
        };
        assert_ne!(eroded, before);

        // … as exactly ONE undo step, whatever it took to record it.
        assert_eq!(
            doc.history.undo_len(),
            undo_depth_before + 1,
            "the bake must be a single transaction, not one entry per layer"
        );
        assert_eq!(doc.history.undo_label(), Some("Erode Terrain"));

        assert!(doc.undo(), "one undo");
        let (data, _) = doc.terrain_data_and_origin(g).unwrap();
        assert!(
            data.data_maps_are_default(),
            "undo must drop the materialized map buffers back to the sparse default"
        );
        assert_eq!(
            saved_bytes(data),
            before,
            "one undo step must restore heights AND maps byte-identically"
        );

        // Redo replays both layers exactly, and is still one step.
        assert!(doc.redo(), "one redo");
        let (data, _) = doc.terrain_data_and_origin(g).unwrap();
        assert_eq!(saved_bytes(data), eroded, "redo must reproduce both layers");
        assert!(doc.undo());
        assert_eq!(doc.history.undo_len(), undo_depth_before);
    }

    /// **THE BIOME UNDO GATE (P19.2).** One stroke is one undo step, and that one
    /// step returns the terrain's **saved bytes** to what they were — including
    /// dropping the materialized id buffers back to the sparse default, so an
    /// undone stroke costs the file nothing.
    ///
    /// Compared as saved bytes (the whole encode: heights + weights + maps + ids)
    /// rather than as a biome-layer hash, for the reason `saved_bytes` states:
    /// "undo restored it" has to mean the file it would write is the file it
    /// would have written.
    #[test]
    fn a_biome_stroke_is_one_undo_step_and_undoes_to_the_saved_bytes() {
        use inf_terrain::{BiomeStroke, BrushParams};

        let mut doc = SceneDoc::new();
        let g = doc.edit_create(SpawnKind::Terrain, "", None);
        let undo_depth_before = doc.history.undo_len();

        let (data, _) = doc.terrain_data_and_origin(g).unwrap();
        let before = saved_bytes(data);
        assert!(data.biomes_are_default(), "a fresh terrain is unpainted");
        let (min, max) = doc
            .terrain_bounds(g)
            .expect("the starter terrain has bounds");
        let centre = (min + max) * 0.5;
        let radius = ((max.x - min.x).min(max.y - min.y) * 0.25).max(1.0);

        // One stroke, several dabs — exactly what a mouse drag produces.
        let mut stroke = BiomeStroke::begin(2);
        for k in 0..4 {
            let c = centre + glam::DVec2::new(k as f64 * radius * 0.3, 0.0);
            doc.biome_apply_dab(g, &mut stroke, BrushParams::new(c, radius, 1.0));
        }
        assert!(doc.edit_commit_biome(g, stroke), "the stroke must record");

        let painted = {
            let (data, _) = doc.terrain_data_and_origin(g).unwrap();
            assert!(!data.biomes_are_default(), "the ids must have landed");
            assert_eq!(data.biome_at(centre), Some(2));
            saved_bytes(data)
        };
        assert_ne!(painted, before);

        assert_eq!(
            doc.history.undo_len(),
            undo_depth_before + 1,
            "a stroke must be ONE undo step, not one per dab"
        );
        assert_eq!(doc.history.undo_label(), Some("Paint Biome"));

        assert!(doc.undo(), "one undo");
        let (data, _) = doc.terrain_data_and_origin(g).unwrap();
        assert!(
            data.biomes_are_default(),
            "undo must drop the materialized id buffers back to the sparse default"
        );
        assert_eq!(saved_bytes(data), before, "undo must be byte-identical");

        assert!(doc.redo(), "one redo");
        let (data, _) = doc.terrain_data_and_origin(g).unwrap();
        assert_eq!(saved_bytes(data), painted, "redo must reproduce the stroke");
    }

    /// The **eraser** is a stroke like any other: it records its own undo step,
    /// labelled so the menu distinguishes it, and it undoes exactly.
    #[test]
    fn an_erasing_stroke_is_labelled_and_undoes_exactly() {
        use inf_terrain::{BiomeStroke, BrushParams, UNASSIGNED_BIOME};

        let mut doc = SceneDoc::new();
        let g = doc.edit_create(SpawnKind::Terrain, "", None);
        let (min, max) = doc.terrain_bounds(g).unwrap();
        let centre = (min + max) * 0.5;
        let radius = ((max.x - min.x).min(max.y - min.y) * 0.25).max(1.0);
        let params = BrushParams::new(centre, radius, 1.0);

        let mut paint = BiomeStroke::begin(5);
        doc.biome_apply_dab(g, &mut paint, params);
        assert!(doc.edit_commit_biome(g, paint));
        let painted = saved_bytes(doc.terrain_data_and_origin(g).unwrap().0);

        let mut erase = BiomeStroke::begin(UNASSIGNED_BIOME);
        doc.biome_apply_dab(g, &mut erase, params);
        assert!(doc.edit_commit_biome(g, erase), "erasing is a real edit");
        assert_eq!(doc.history.undo_label(), Some("Erase Biome"));
        assert_eq!(
            doc.terrain_data_and_origin(g).unwrap().0.biome_at(centre),
            Some(UNASSIGNED_BIOME)
        );

        assert!(doc.undo());
        assert_eq!(
            saved_bytes(doc.terrain_data_and_origin(g).unwrap().0),
            painted,
            "undoing an erase must restore the painted bytes"
        );
    }

    /// Binding a `.inf_biomes` to a terrain is one undo step, and rebinding to the
    /// same set records nothing (so a toolbar that re-pushes its selection cannot
    /// flood the undo stack).
    #[test]
    fn binding_a_biome_set_is_one_undo_step_and_is_idempotent() {
        let mut doc = SceneDoc::new();
        let g = doc.edit_create(SpawnKind::Terrain, "", None);
        let depth = doc.history.undo_len();
        let set = Uuid::from_u128(0xB10E);

        assert_eq!(doc.terrain_biome_set(g), None);
        assert!(doc.edit_set_terrain_biome_set(g, Some(set)));
        assert_eq!(doc.terrain_biome_set(g), Some(set));
        assert_eq!(doc.history.undo_len(), depth + 1);

        assert!(
            !doc.edit_set_terrain_biome_set(g, Some(set)),
            "rebinding the same set must record nothing"
        );
        assert_eq!(doc.history.undo_len(), depth + 1);

        assert!(doc.undo());
        assert_eq!(doc.terrain_biome_set(g), None, "undo must unbind");
        assert!(doc.redo());
        assert_eq!(doc.terrain_biome_set(g), Some(set));

        // Clearing it is also a step, and an entity with no terrain is a no-op.
        assert!(doc.edit_set_terrain_biome_set(g, None));
        assert_eq!(doc.terrain_biome_set(g), None);
        let cube = doc.edit_create(SpawnKind::Cube, "", None);
        assert!(!doc.edit_set_terrain_biome_set(cube, Some(set)));
    }
}

/// Wave 2: world-space gizmo writeback (`edit_set_world_transform`) — verifies a
/// child of a rotated/scaled parent keeps its edited world pose, and roots are
/// unchanged from the old world-as-local behaviour.
#[cfg(test)]
mod wave2_world_transform_tests {
    use super::*;

    fn v3(x: f64, y: f64, z: f64) -> inf_ecs::Vec3d {
        inf_ecs::Vec3d::from_dvec3(DVec3::new(x, y, z))
    }

    #[test]
    fn child_of_rotated_scaled_parent_keeps_world_position() {
        let mut doc = SceneDoc::new();
        let parent = doc.create(SpawnKind::Empty, "Parent", None);
        let child = doc.create(SpawnKind::Cube, "Child", Some(parent));

        // Parent: translated, yawed 90°, uniformly scaled 2×.
        let mut pt = Transform::from_translation(DVec3::new(10.0, 0.0, 4.0));
        pt.rotation = v3(0.0, 90.0, 0.0);
        pt.scale = v3(2.0, 2.0, 2.0);
        doc.edit_set_transform(parent, pt);

        // Ask the gizmo writeback to place the child at a specific WORLD point.
        let target = DVec3::new(5.0, 1.0, -3.0);
        doc.edit_set_world_transform(child, Transform::from_translation(target));

        doc.world_mut().propagate();
        let ce = doc.world().entity_of(child).unwrap();
        let wp = doc.world().world_translation(ce).unwrap();
        assert!(
            (wp - target).length() < 1e-6,
            "child world pos {wp:?} should equal target {target:?}"
        );
    }

    #[test]
    fn root_writeback_is_world_as_local() {
        let mut doc = SceneDoc::new();
        let e = doc.create(SpawnKind::Cube, "Root", None);
        let target = DVec3::new(3.0, 4.0, 5.0);
        doc.edit_set_world_transform(e, Transform::from_translation(target));

        // A root's local translation is exactly the world translation.
        let ent = doc.world().entity_of(e).unwrap();
        let local = doc.world().world().get::<Transform>(ent).copied().unwrap();
        assert!((local.translation.to_dvec3() - target).length() < 1e-9);

        doc.world_mut().propagate();
        let wp = doc.world().world_translation(ent).unwrap();
        assert!((wp - target).length() < 1e-9);
    }

    #[test]
    fn multi_select_parents_first_preserves_child_world() {
        // Move BOTH a parent and its child in one drag: applied parents-first,
        // the child's local is composed against the parent's already-written
        // pose, so both land at their target world points.
        let mut doc = SceneDoc::new();
        let parent = doc.create(SpawnKind::Empty, "Parent", None);
        let child = doc.create(SpawnKind::Cube, "Child", Some(parent));

        let parent_target = DVec3::new(7.0, 2.0, 0.0);
        let child_target = DVec3::new(9.0, 2.0, 1.0);
        // Parents-first order (as the win32 writeback sorts).
        doc.edit_set_world_transform(parent, Transform::from_translation(parent_target));
        doc.edit_set_world_transform(child, Transform::from_translation(child_target));

        doc.world_mut().propagate();
        let pe = doc.world().entity_of(parent).unwrap();
        let ce = doc.world().entity_of(child).unwrap();
        assert!((doc.world().world_translation(pe).unwrap() - parent_target).length() < 1e-6);
        assert!((doc.world().world_translation(ce).unwrap() - child_target).length() < 1e-6);
    }

    // ── E-P1 add / remove component ─────────────────────────────────────────

    fn has_component(doc: &SceneDoc, guid: Uuid, type_path: &str) -> bool {
        doc.entity_props(guid)
            .iter()
            .any(|c| c.type_path == type_path)
    }

    #[test]
    fn add_remove_component_round_trips_and_undoes() {
        let mut doc = SceneDoc::new();
        let g = doc.create(SpawnKind::Cube, "Body", None);
        let joint_tp = doc.world().registry().type_path_for("Joint 3D").unwrap();

        // Absent to start.
        assert!(!has_component(&doc, g, joint_tp));

        // Add → present; undo → absent; redo → present.
        assert!(doc.edit_add_component(g, joint_tp));
        assert!(has_component(&doc, g, joint_tp));
        assert!(doc.undo());
        assert!(!has_component(&doc, g, joint_tp));
        assert!(doc.redo());
        assert!(has_component(&doc, g, joint_tp));

        // Remove → absent; undo → present.
        assert!(doc.edit_remove_component(g, joint_tp));
        assert!(!has_component(&doc, g, joint_tp));
        assert!(doc.undo());
        assert!(has_component(&doc, g, joint_tp));

        // Idempotency: adding a present component / removing an absent one no-op.
        assert!(!doc.edit_add_component(g, joint_tp)); // already there
        assert!(doc.edit_remove_component(g, joint_tp));
        assert!(!doc.edit_remove_component(g, joint_tp)); // already gone
    }

    #[test]
    fn removing_transform_is_rejected() {
        let mut doc = SceneDoc::new();
        let g = doc.create(SpawnKind::Cube, "Body", None);
        let transform_tp = doc.world().registry().type_path_for("Transform").unwrap();
        assert!(has_component(&doc, g, transform_tp));
        // Transform is structural — not addable/removable.
        assert!(!doc.edit_remove_component(g, transform_tp));
        assert!(has_component(&doc, g, transform_tp));
        assert!(!doc.edit_add_component(g, transform_tp));
    }

    #[test]
    fn fifty_add_remove_steps_undo_and_redo_cleanly() {
        let mut doc = SceneDoc::new();
        let g = doc.create(SpawnKind::Cube, "Body", None);
        // Two independent components toggled across 50 recorded steps.
        let joint = doc.world().registry().type_path_for("Joint 3D").unwrap();
        let audio = doc
            .world()
            .registry()
            .type_path_for("Audio Source")
            .unwrap();
        let comps = [joint, audio];
        let mut steps = 0usize;
        // Build 50 add/remove edits (each component toggled 25 times → ends present).
        while steps < 50 {
            for tp in comps {
                if steps >= 50 {
                    break;
                }
                if has_component(&doc, g, tp) {
                    assert!(doc.edit_remove_component(g, tp));
                } else {
                    assert!(doc.edit_add_component(g, tp));
                }
                steps += 1;
            }
        }
        // Post-50-step state: 25 toggles each (odd) → both present.
        assert!(has_component(&doc, g, joint));
        assert!(has_component(&doc, g, audio));
        // Undo everything → both absent (started absent).
        while doc.undo() {}
        assert!(!has_component(&doc, g, joint));
        assert!(!has_component(&doc, g, audio));
        // Redo everything → back to the post-50-step state (both present).
        while doc.redo() {}
        assert!(has_component(&doc, g, joint));
        assert!(has_component(&doc, g, audio));
    }

    #[test]
    fn add_component_defaults_and_swap_preserves_other_components() {
        // Adding/removing a component must not disturb sibling components.
        let mut doc = SceneDoc::new();
        let g = doc.create(SpawnKind::Cube, "Body", None);
        let mat_tp = doc.world().registry().type_path_for("Material").unwrap();
        let joint_tp = doc.world().registry().type_path_for("Joint 3D").unwrap();
        // The cube already has a Material; edit it, then add a Joint and undo.
        assert!(has_component(&doc, g, mat_tp));
        doc.edit_set_prop(g, mat_tp, "metallic", &PropValue::Number(0.5));
        assert!(doc.edit_add_component(g, joint_tp));
        assert!(doc.undo()); // undo the joint add
                             // Material (and its edited value) survives the swap-revert.
        assert!(has_component(&doc, g, mat_tp));
        let m = doc.entity_props(g);
        let metallic = m
            .iter()
            .find(|c| c.type_path == mat_tp)
            .unwrap()
            .fields
            .iter()
            .find(|f| f.name == "metallic")
            .unwrap();
        assert_eq!(metallic.value, PropValue::Number(0.5));
    }
}
