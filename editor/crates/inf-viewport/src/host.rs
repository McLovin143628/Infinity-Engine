//! Platform-shared engine host: owns the GPU stack (context, swapchain,
//! renderer), the render scene, and the floating origin. The per-OS modules
//! (win32, macos) own the native window/layer + input and drive this.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use glam::{DQuat, DVec2, DVec3, Vec2, Vec3};
use inf_ecs::components::{
    BlendMode, Collider2D, Collider3D, ColliderShape2DKind, ColliderShape3DKind,
    ComputedVisibility, Foliage, FoliageInstance, GlobalTransform, Joint2D, Joint3D, Light,
    Light2D, LightKind as EcsLightKind, Material, MeshRef, NineSlice, PcgVolume, Primitive,
    ScatteredInstance, SkeletalMesh, Spline, SplineInterp as EcsSplineInterp, Sprite, Terrain,
    Text2D, TextAlign, Tilemap, Volume, VoxelVolume, WaterBody, WaterKind,
};
use inf_ecs::{Transform as EcsTransform, Vec3d};
use inf_editor_core::ipc::SpawnKind;
use inf_editor_core::scene::serialize::RenderSettingsRecord;
use inf_editor_core::scene::SceneDoc;
use inf_math::{FloatingOrigin, SplineInterp};
// P21.2: the carve tools' brush primitive and the op it wraps it in. The rest of
// this crate's voxel surface stays path-qualified (`inf_voxel::…`) — these two
// are named because they appear in every line of the tool.
use inf_voxel::{VoxelOp, VoxelShape};
// R-P4: scene-persisted post/exposure/lighting settings applied to the live
// renderer (see `apply_record` + `sync_from_doc`).
use inf_render::{
    collider_outline_2d, collider_outline_3d, expand_nine_slice, expand_text, gizmo,
    handle_from_guid, AtmosphereParams, CloudParams, ColliderOutline2D, ColliderOutline3D,
    DebugDraw, EngineRenderer, GizmoDelta, GizmoDrag, GizmoMode, GpuContext, HAlign, HeightFog,
    LightKind, MeshInstance, NineSliceParams, OrthoParams, Picker, PrebatchedRun, PrecipParams,
    PrimMesh, RenderChunk, RenderLight, RenderLight2D, RenderScene, RenderTerrain,
    RenderTerrainLayer, RenderTerrainTile, RenderTilemap, RenderView, RenderWater, ScatterBatch,
    ScatterData, ScatterInstance, SkyParams, SpriteInstance, SunParams, SurfaceChain,
    TerrainTileKey, TextParams, TilemapParams, BUILTIN_FONT_TEXTURE,
};
use inf_render::{
    detect_tier, BloomSettings, ExposureMode, ExposureSettings, FilmSettings, FlareSettings,
    GiSettings, RenderSettings, RenderTier, ShadowSettings, SsaoSettings, SsrQuality, SsrSettings,
};
use uuid::Uuid;

use inf_terrain::{
    raycast_terrain, BiomeStroke, BrushOp, BrushParams, Falloff, FlattenTarget, SplatStroke,
    Stroke, TerrainData,
};

use crate::camera::{
    BiomeSettings, Camera2D, EditorCamera, FoliageSettings, GizmoSpace, SculptFalloff, SculptOp,
    SculptSettings, Snap2DSettings, SnapSettings, SpoilMode, ToolMode, ViewportMode, VoxelOpMode,
    VoxelSettings, VoxelToolKind, WaterSettings, WaterToolKind, TWO_D_FAR, TWO_D_NEAR,
};
use crate::SurfaceTarget;

/// Frames of *page movement* between terrain-streaming diagnostics lines
/// (P16.3b2). Roughly 5 s at 60 fps of continuous paging; a settled camera never
/// reaches it because the counter only ticks when the cut actually changed.
const STREAM_LOG_INTERVAL_FRAMES: u32 = 300;

/// How far back the `Home` action stands from the player start, metres (wave
/// EDIT1, clause 2) — the **2D** and fallback path.
///
/// `EditorCamera::focus_goal` reads its argument as the radius of a sphere to
/// fit, so this is "frame a twenty-metre ball around the character": the pawn
/// plus the street it stands in plus the fronts of the buildings either side.
/// The pawn's own radius would be under two metres and would put the author's
/// nose against a shirt with no way to tell a harbour from a hillside.
const PLAYER_START_FRAME_M: f64 = 20.0;

/// Where the 3D camera actually stands for `Home`: eye height, metres.
///
/// **Not a framed sphere**, and the reason is a measured one. Framing a sphere
/// puts the camera wherever its current pitch happens to point it, and on the
/// showcase island — where the start stands 14 m from a settlement block — the
/// first cut of this clause came to rest above a building and filled the lower
/// half of the frame with a roof. What an author opening a level wants is the
/// shot the GAME opens on: standing behind the character at head height,
/// looking the way it faces, down the street it is standing in.
const PLAYER_START_EYE_M: f64 = 1.75;
/// How far behind the character that camera stands, metres — a third-person
/// boom, not a shoulder.
const PLAYER_START_BACK_M: f64 = 7.0;
/// Its pitch, radians. Barely down: enough to put the character's feet and the
/// road in the lower third without looking at the pavement.
const PLAYER_START_PITCH: f32 = -0.12;

/// Where the level's player-controlled character stands, in world metres.
///
/// **One door with the runtime.** `inf_ecs::movement::camera_subject` is what
/// the shipped player re-resolves every step to decide who the camera follows,
/// and what `scene_player_pawn` already asks before offering to Play. Asking it
/// here is what keeps "where the editor opens" and "who the game follows" from
/// becoming two answers that drift.
///
/// `GlobalTransform` first, the local `Transform` second — the same order
/// `inf_player::cell_stream::streaming_sources` reads a source's position in, and
/// for the same reason: a pawn parented under a rig has no meaningful local
/// translation, and a document that has not propagated yet has no global one.
fn player_start_of(doc: &SceneDoc) -> Option<(DVec3, f32)> {
    let guid = inf_ecs::movement::camera_subject(doc.world())?;
    let entity = doc.world().entity_of(guid)?;
    let w = doc.world().world();
    let (translation, rotation) = if let Some(g) = w.get::<GlobalTransform>(entity) {
        let (_, r, t) = g.0.to_scale_rotation_translation();
        (t, r)
    } else {
        let t = w.get::<EcsTransform>(entity)?;
        (t.translation.to_dvec3(), t.quat())
    };
    // Which way it faces, as the editor camera's own yaw. `forward()` is
    // `(sin y, 0, -cos y)` at zero pitch, so `-Z` is yaw zero and the inverse is
    // `atan2(x, -z)`. A character with no meaningful facing (an identity
    // rotation) gives yaw 0, which is -Z, which is where the default camera
    // already looks — so the degenerate case is the old behaviour and not a
    // surprise.
    let f = rotation * glam::DVec3::NEG_Z;
    let yaw = f64::atan2(f.x, -f.z) as f32;
    Some((translation, yaw))
}

/// Map an ECS [`Primitive`] to the renderer's [`PrimMesh`] (R-P1).
///
/// MIRROR: keep identical to `inf_player::render::prim_mesh` (the player's
/// ECS→RenderScene projection). Both seams must agree so the editor viewport and
/// the shipped player draw the same geometry for a given primitive.
fn prim_mesh(p: Primitive) -> PrimMesh {
    match p {
        Primitive::Cube => PrimMesh::Cube,
        Primitive::Sphere => PrimMesh::Sphere,
        Primitive::Plane => PrimMesh::Plane,
        Primitive::Cylinder => PrimMesh::Cylinder,
        Primitive::Cone => PrimMesh::Cone,
    }
}

/// Project the ECS [`BlendMode`] into the renderer's packed `blend` code (R-P5):
/// 0 opaque, 1 masked, 2 translucent. Mirrored in the player's `render.rs`.
fn blend_code(b: BlendMode) -> u8 {
    match b {
        BlendMode::Opaque => 0,
        BlendMode::Masked => 1,
        BlendMode::Translucent => 2,
    }
}

/// Parse a [`SpawnKind`] from its snake_case wire string (the tail of a
/// `"spawn:<kind>"` drop payload). Mirrors the `serde(rename_all = "snake_case")`
/// on the DTO — kept as an explicit match so the drop path stays serde-free.
fn spawn_kind_from_str(s: &str) -> Option<SpawnKind> {
    Some(match s {
        "empty" => SpawnKind::Empty,
        "cube" => SpawnKind::Cube,
        "sphere" => SpawnKind::Sphere,
        "plane" => SpawnKind::Plane,
        "cylinder" => SpawnKind::Cylinder,
        "cone" => SpawnKind::Cone,
        "directional_light" => SpawnKind::DirectionalLight,
        "point_light" => SpawnKind::PointLight,
        "spot_light" => SpawnKind::SpotLight,
        "camera" => SpawnKind::Camera,
        "sprite" => SpawnKind::Sprite,
        "tilemap" => SpawnKind::Tilemap,
        "text2d" => SpawnKind::Text2d,
        "nine_slice" => SpawnKind::NineSlice,
        "light2d" => SpawnKind::Light2d,
        "terrain" => SpawnKind::Terrain,
        "spline" => SpawnKind::Spline,
        "foliage" => SpawnKind::Foliage,
        "trigger_volume" => SpawnKind::TriggerVolume,
        "blocking_volume" => SpawnKind::BlockingVolume,
        _ => return None,
    })
}

pub struct EngineHost {
    target: SurfaceTarget,
    gpu: GpuContext,
    chain: SurfaceChain,
    renderer: EngineRenderer,
    // Only the Windows input layer picks today; macOS input lands with its
    // hardware pass (kept constructed so the field is ready when it does).
    #[cfg_attr(not(windows), allow(dead_code))]
    picker: Picker,
    pub scene: RenderScene,
    pub origin: FloatingOrigin,
    /// Active transform-gizmo mode; the gizmo shows only with a selection.
    pub gizmo_mode: GizmoMode,
    /// Gizmo orientation frame (Wave 2): world-aligned handles or local
    /// (selection-rotation) handles. 2D mode always draws/edits in World.
    gizmo_space: GizmoSpace,
    /// 3D transform-gizmo snap increments pushed from the toolbar (Wave 2),
    /// replacing the previously-hardcoded 1 m / 15° / 0.1 constants. Only the
    /// Windows input layer applies it (via [`EngineHost::snap_3d`]).
    #[cfg_attr(not(windows), allow(dead_code))]
    snap_3d: SnapSettings,
    gizmo_drag: Option<GizmoDrag>,
    /// Mesh-instance transforms captured at gizmo-drag start, keyed by instance
    /// id. The cumulative gizmo delta (measured from the ORIGINAL grab anchor,
    /// see [`EngineHost::update_gizmo`]) is applied to THESE each frame — the
    /// live instances are never accumulated frame-to-frame — so snapping
    /// quantizes total displacement, not per-frame deltas (M2).
    gizmo_initial: HashMap<u32, InstanceXform>,
    /// Same as [`Self::gizmo_initial`] for selected 2D (non-mesh) entities.
    gizmo_initial_2d: HashMap<Uuid, Sel2D>,
    fov_y: f32,
    /// Render-instance id → entity GUID, rebuilt each projection (P3.2). Lets a
    /// pick resolve to a scene entity and a gizmo write back to the document.
    id_to_guid: HashMap<u32, Uuid>,
    guid_to_id: HashMap<Uuid, u32>,
    /// World-space 2D collider outlines by GUID (P8.3b), rebuilt each projection.
    /// Rendered as debug lines for the current selection only.
    collider_outlines: HashMap<Uuid, ColliderDebug>,
    /// World-space 3D collider wireframes by GUID (P9.1), rebuilt each projection.
    /// Rendered as debug lines for the current selection only.
    collider_outlines_3d: HashMap<Uuid, ColliderDebug3D>,
    /// World-space joint debug segments by GUID (P12.1), rebuilt each projection.
    /// Rendered as debug lines for the current selection only (2D + 3D joints).
    joint_lines: HashMap<Uuid, JointDebug>,
    /// World-space `Volume` wireframes by GUID (E-P4), rebuilt each projection.
    /// Unlike collider outlines these are drawn ALWAYS (not selection-gated), in
    /// the volume's tint, so trigger/blocking regions stay visible while editing.
    volume_outlines: HashMap<Uuid, VolumeDebug>,
    /// Spot-light cone gizmos by GUID (R-P3), rebuilt each projection. Drawn as
    /// debug lines for the current selection only.
    spot_lights: HashMap<Uuid, SpotDebug>,
    /// World-space `Spline` polylines by GUID (E-P5), rebuilt each projection.
    /// The sampled curve is drawn ALWAYS (neutral cyan); a selected spline
    /// additionally shows a 3-axis cross at each control point.
    spline_polylines: HashMap<Uuid, SplineDebug>,
    /// GUIDs of the document's current selection (for collider debug draw —
    /// covers every selected entity, not just the mesh instances).
    selected_guids: Vec<Uuid>,
    /// Document version the current projection reflects (skip redundant rebuilds).
    synced_version: Option<u64>,
    /// **The seam owes a recompute** (round-2 finding B9).
    ///
    /// Hardening Wave E made `apply_seam` skippable: when every volume and
    /// every terrain was carried forward unchanged, last frame's per-vertex
    /// terms are still right. That argument is sound for the player, whose
    /// only writer of `scene.terrains` is `project_scene` itself. This host has
    /// **two more**: `sync_streamed_terrain` (the camera advanced the cut) and
    /// `after_terrain_edit` (a sculpt dab), both of which re-project a terrain
    /// **in place**, outside `rebuild_scene`. By the time the next projection
    /// runs, its carry comparison is against the already-updated list — every
    /// signature matches, nothing is dropped, and the skip fires. A cave mouth
    /// beside a streamed terrain then keeps seam terms computed against
    /// pre-dab, pre-page heights for the rest of the session, while the shipped
    /// build recomputes them — a new editor-vs-shipping divergence in the exact
    /// seam the mirrored-pair discipline exists to protect.
    ///
    /// Invalidating `synced_version` alone does not close it: it forces the
    /// projection to *run*, and the run then carries everything and skips.
    /// This is the term that makes the run recompute.
    seam_dirty: bool,
    /// Active projection: perspective (3D) or orthographic (2D editor). Drives
    /// the gizmo handle set and the grid plane; the camera itself lives in the
    /// platform loop (which keeps a separate pose per mode). (P8.2c)
    pub mode: ViewportMode,
    /// 2D-mode snapping config pushed from the toolbar (P8.2c). Only the Windows
    /// input layer reads it (via [`EngineHost::snap_2d_translate`]).
    #[cfg_attr(not(windows), allow(dead_code))]
    snap_2d: Snap2DSettings,
    /// Working transforms of selected **non-mesh** entities (sprites, text, …)
    /// for the gizmo: captured from the document on each projection, mutated
    /// during a drag, written back on release. Mesh entities use the render
    /// instances instead (`scene.instances`). (P8.2c)
    selected_2d: HashMap<Uuid, Sel2D>,
    /// Active tool: pick/gizmo (`Select`) or terrain sculpt (`Sculpt`). (P10.2b)
    tool_mode: ToolMode,
    /// Sculpt brush configuration pushed from the toolbar (P10.2b).
    sculpt: SculptSettings,
    /// Biome brush configuration pushed from the toolbar (P19.2).
    biome: BiomeSettings,
    /// Per-terrain biome overlay palettes, indexed by biome id — pushed from
    /// Ring 2, which is the only layer that has both the scene and the asset
    /// database and can therefore resolve a `Terrain.biome_set` to a
    /// `BiomeSet::palette()`.
    ///
    /// Viewport-local *render* state, exactly like the view mode: it is derived
    /// from an asset, never authored here, and nothing in Ring 0/1 has to learn
    /// about the asset DB for the overlay to work. An entry is absent until Ring 2
    /// pushes one, and an absent palette renders every sample as *unassigned* —
    /// which is the honest picture of a terrain with no biome set bound.
    biome_palettes: std::collections::BTreeMap<Uuid, Vec<[f32; 4]>>,
    /// GUID of the terrain entity the terrain tools currently target (P16.6).
    ///
    /// Set to the **first** projected terrain on each projection — so a
    /// single-terrain document behaves exactly as it did before — and then moved
    /// to whichever terrain the cursor is actually over by
    /// [`sculpt_pick`](Self::sculpt_pick)'s nearest-hit resolution, which the
    /// hover and stroke paths both run through. `None` ⇒ no terrain is projected.
    terrain_guid: Option<Uuid>,
    /// Every visible, non-empty terrain in the current projection, in the
    /// document's order — **index-aligned with `scene.terrains`** (P16.6), which
    /// is what lets a re-projection of one terrain (a streamed cut advancing, a
    /// dab landing) write back into the right slot instead of rebuilding the list.
    terrain_slots: Vec<TerrainSlot>,
    /// Camera-driven streaming for asset-backed terrains (P16.3b2). The policy
    /// lives in `inf_editor_core::terrain_stream` (Ring 1, so Linux CI compiles
    /// and tests it); the host only calls it — at projection time
    /// ([`rebuild_scene`](Self::rebuild_scene)) and at the render-sync point
    /// ([`render_frame`](Self::render_frame)).
    ///
    /// **The determinism seam.** Its *camera-driven* pages land in the streamer's
    /// own working set, never in the document's `Terrain.data`, so moving the
    /// editor camera cannot dirty the document, change a `height_at` answer, or
    /// desync a Simulate session from a shipped run. (An **edit** does page into
    /// the document — synchronously, footprint-shaped — which is a different
    /// thing entirely; see `inf_editor_core::terrain_edit`.) Disabled until a
    /// content root is set, which makes inline terrain behaviour bit-identical to
    /// before.
    terrain_streams: inf_editor_core::terrain_stream::EditorTerrainStreams,
    /// Where the player starts and which way it faces (yaw, radians), cached
    /// from the document at projection time (wave EDIT1, clause 2).
    ///
    /// Asked through `inf_ecs::movement::camera_subject` -- the runtime's own
    /// door, the same one `scene_player_pawn` asks and the same one the shipped
    /// player re-resolves every step -- so "where the editor opens" and "who the
    /// game follows" cannot become two different answers.
    ///
    /// Cached rather than asked on demand because the two readers are the `Home`
    /// action and the per-frame gizmo, and neither holds the document lock at the
    /// point it needs the answer. `None` on a level with no player-controlled
    /// character, which is a real level (a cinematic, a blockout) and not an
    /// error -- the same judgement `NoPawnPlayDialog` makes.
    player_start: Option<(DVec3, f32)>,
    /// The loaded voxel volumes (P21.1) — the caves, tunnels and excavations that
    /// locally extend the heightfield. The Ring-1 store resolves a
    /// `VoxelVolume.asset` to a loose `.inf_voxel` under the content root and hands
    /// the bytes to the Ring-0 `inf_voxel::VoxelVolumes`, which owns parsing,
    /// residency and meshing — so the viewport and the shipped player cannot mesh
    /// the same field differently. Disabled until a content root is set, which
    /// makes a document with no volumes bit-identical to its pre-P21.1 self.
    /// **Shared** rather than owned (P21.2): the undo history has to be able to
    /// put a carve back, and `SceneDoc::undo` runs on the Ring-2 command thread
    /// with no viewport. The type is spelled out rather than written as its
    /// `SharedVoxelVolumes` alias so the mirror gate can still see which store
    /// this host loads through. Lock order everywhere: **document, then this**.
    voxel_volumes:
        std::sync::Arc<std::sync::Mutex<inf_editor_core::voxel_store::EditorVoxelVolumes>>,
    /// The Simulate session's fracture states (P22.3), published in by Ring 2
    /// after each fixed step.
    ///
    /// Empty while nothing is simulating — and empty is exactly "no actor has
    /// broken", so an authoring session projects not one extra byte. The type is
    /// spelled out rather than written as its `SharedFractures` alias for the same
    /// reason `voxel_volumes` above is: the mirror gate can then see which state
    /// this host draws from.
    /// The P22.4 sub-chunk rubble memo, keyed by each broken actor's fracture
    /// generation, so a collapse packs its instance payload when the live chunk
    /// set changes rather than on every projection.
    debris_cache: inf_render::DebrisCache,
    fractures: std::sync::Arc<
        std::sync::Mutex<std::collections::BTreeMap<uuid::Uuid, inf_physics::d3::FractureState>>,
    >,
    /// The loose-file render-asset store (P18.3) — the editor's answer to the
    /// player's `VmeshRegistry`. Resolves a `MeshRef.asset` to its derived
    /// `.inf_vmesh` and a `SkeletalMesh` to bind-space geometry + a posed skinning
    /// palette, both from the project's content root.
    ///
    /// Owned here, and released here: the projection is the only thing that knows
    /// which mesh assets a document actually references, so it is the only thing
    /// that can free the rest ([`EditorRenderAssets::retain_only`]). The policy
    /// itself lives in Ring 1 for the same reason terrain streaming does — Linux CI
    /// compiles and tests it, this file it does not.
    ///
    /// [`EditorRenderAssets::retain_only`]: inf_editor_core::render_assets::EditorRenderAssets::retain_only
    render_assets: inf_editor_core::render_assets::EditorRenderAssets,
    /// The authored meshes scattered content draws (wave TER2b) — resolved out of
    /// `render_assets` at the head of each projection and read by `push_scatter`.
    ///
    /// Separate from the store it comes from because the projector must not hold
    /// a `&mut` to the store while it walks the world, and because this is the
    /// *table* the player's shipped host also hands its projector: one shape, two
    /// hosts, and the mirror gate compares the body that reads it.
    scatter_meshes: inf_render::ScatterMeshes,
    /// Every mesh GUID this document's scattered instances **named**, whether or
    /// not it resolved (wave TER2b audit).
    ///
    /// The live-asset audit has to be keyed on the question and not on the
    /// answer: a GUID that failed to resolve is a *negative* entry in
    /// `render_assets`, and keying the audit on `scatter_meshes` — which holds
    /// only the hits — evicted every miss on the very projection that took it, so
    /// a scatter kind naming a mesh past `MAX_SCATTER_MESH_TRIANGLES` re-opened
    /// and re-decoded that file on every document version, which is once per
    /// input event during a gizmo drag.
    scatter_wanted: std::collections::BTreeSet<Uuid>,
    /// **What the live virtual-texture level was built from** (P26.5) — the
    /// binding set AND the asset index's generation, as one value.
    ///
    /// A projection runs on every document version and building a VT level
    /// creates an atlas, so the rebuild is gated on this changing (the
    /// `terrain_slots` pattern, for materials). It was two fields compared
    /// inline until P26.5; the rule now lives in
    /// [`inf_editor_core::render_assets::VtLevelKey`], where a test can execute
    /// it — this file needs a window, so the P26.4 audit could only pin the
    /// early-out as source text.
    vt_level_key: inf_editor_core::render_assets::VtLevelKey,
    /// The last tool-rejection message, for a Ring-2 caller to surface. Drained by
    /// [`take_tool_status`](Self::take_tool_status).
    tool_status: Option<String>,
    /// Frames until the next terrain-streaming diagnostics line. Counts down only
    /// while pages are actually moving, so a settled camera logs nothing.
    stream_log_countdown: u32,
    /// In-flight sculpt stroke: the accumulating brush gesture (`None` = idle).
    sculpt_drag: Option<SculptDrag>,
    /// World-space brush-ring loop points (following terrain height), rebuilt as
    /// the cursor hovers/sculpts terrain; drawn as debug lines in Sculpt mode.
    /// Shared with the Foliage brush (same hover-ring buffer, different colour).
    sculpt_ring: Vec<DVec3>,
    /// Colour of the brush ring (encodes the active op / foliage brush).
    sculpt_ring_color: [f32; 4],
    /// Foliage-brush configuration pushed from the toolbar (E-P6).
    foliage: FoliageSettings,
    /// In-flight foliage scatter stroke (`None` = idle).
    foliage_drag: Option<FoliageDrag>,
    /// Monotonic per-session stroke counter: folded into the scatter RNG so each
    /// stroke is independent yet the same input sequence reproduces identical
    /// instances (determinism law — no wall-clock / thread-rng).
    foliage_stroke_seq: u32,
    /// Water-tool configuration pushed from the toolbar (P20.4).
    water_tool: WaterSettings,
    /// Per-terrain **water-level hints by biome id** (P19.2's
    /// `BiomeDef::water_hint`, read at last), pushed from Ring 2 alongside the
    /// biome palette.
    ///
    /// Pushed rather than resolved here for the same reason the palette is: the
    /// viewport thread holds a document, not an asset database, and a
    /// `.inf_biomes` is an asset. Indexed by biome id (`0..=255`); an id past the
    /// end, or a `None` entry, means "this biome has no hint" and the tool falls
    /// back to the ground.
    water_hints: std::collections::BTreeMap<Uuid, Vec<Option<f64>>>,
    /// The river the next click extends (`None` = the next click starts one).
    /// Cleared when the tool is left, so switching away and back does not silently
    /// append to a river the author has stopped thinking about.
    water_active_river: Option<Uuid>,
    /// In-flight lake drag: the world point the press landed on. The release
    /// point completes the rectangle.
    water_lake_drag: Option<DVec3>,
    /// The preview the water tool draws: the pending river segment, or the lake
    /// rectangle plus its waterline contour. World space, rebuilt on hover and on
    /// drag; drawn as debug lines and never persisted.
    water_preview: Vec<[DVec3; 2]>,
    /// Voxel-tool configuration pushed from the toolbar (P21.2).
    voxel_tool: VoxelSettings,
    /// In-flight carve stroke (`None` = idle), plus the terrains the verdict said
    /// it may open. The terrain list is resolved ONCE at mouse-down: re-deriving
    /// it per dab would let a drag wander onto ground the author was never warned
    /// about, and if that ground were inline its mouths would be sealed by the
    /// next save.
    voxel_stroke: Option<inf_editor_core::scene::undo::CarveStroke>,
    voxel_stroke_terrains: Vec<Uuid>,
    /// World point of the stroke's last dab, so a fast drag is resampled at even
    /// arc length rather than leaving gaps between dabs.
    voxel_stroke_last: Option<DVec3>,
    /// The spline tunnel's waypoints so far (world metres). Cleared when the tool
    /// or the sub-mode is left — an author who walked away must not find a
    /// half-drawn tunnel waiting to be committed.
    voxel_path: Vec<DVec3>,
    /// The **box cut**'s drag anchor: the surface point the press landed on
    /// (P21.3). `Some` for exactly as long as the button is down, and cleared by
    /// a commit, a tool switch or a settings push — an abandoned pit rectangle
    /// must not be waiting on return.
    voxel_box_anchor: Option<DVec3>,
    /// …and the surface point under the cursor while that drag runs, so the
    /// preview and the readout describe the same rectangle the release will cut.
    voxel_box_cursor: Option<DVec3>,
    /// Where the author put the **spoil site** marker (P21.3), or `None` for the
    /// deterministic default.
    ///
    /// Host state and not a setting, because the author picks it in the viewport
    /// and the toolbar has no way to name a world point. It survives sub-mode
    /// changes and tool switches on purpose: a heap site is a decision about the
    /// *level*, not about the gesture, and re-picking it after every trip to the
    /// Select tool would be busywork.
    voxel_spoil_site: Option<DVec3>,
    /// The `SceneDoc::doc_id` this host last saw, so it can notice the document
    /// being replaced under it (P21.3 audit). `None` before the first frame.
    last_doc_id: Option<u64>,
    /// The box cut's **resolved** plan, refreshed while the drag runs so the
    /// preview draws the pit the release would actually cut rather than a
    /// rubber-band between the two corner picks (P21.3 audit).
    voxel_box_plan: Option<inf_editor_core::voxel_tool::BoxCutPlan>,
    /// The preview the voxel tool draws: the tunnel path plus the segment the next
    /// click would add, the pit rectangle being dragged, the brush's cut sphere,
    /// and the spoil marker. Debug lines, never persisted.
    voxel_preview: Vec<[DVec3; 2]>,
    /// Terrains already reported as carrying holes they cannot save (P21.2's
    /// defensive advisory). One warning per terrain per document, because the
    /// check runs on every projection and a per-frame status event is an
    /// unusable seam, not a louder one.
    voxel_hole_warned: BTreeSet<Uuid>,
    /// Camera eye captured on the last rendered frame (P10.5b). PCG scatter
    /// instances are draw-distance-culled against it at projection time; because
    /// projection is doc-version-gated (not per-frame), the cull set refreshes
    /// whenever the document changes (a `pcg_evaluate` bumps the version) rather
    /// than continuously as the camera moves — a documented v1 simplification.
    last_eye_world: DVec3,
    /// The GPU capability tier detected once from the adapter (R-P4). `None` until
    /// the first `sync_from_doc` probes it; it clamps the scene-persisted render
    /// settings down (never up) via [`RenderTier::apply`], exactly like the player.
    render_tier: Option<RenderTier>,
    /// The adapter capabilities probed once alongside [`Self::render_tier`]
    /// (P18.3). Needed because the editor now *requests* the meshlet path, so the
    /// occlusion capability floor (`clamp_occlusion`) has to be applied on top of
    /// the tier — the pair that [`inf_render::detect_and_clamp`] is. Cached rather
    /// than re-probed because `apply_render_settings` runs on every document
    /// version, which during a gizmo drag is every frame.
    render_caps: Option<inf_render::AdapterCaps>,
    /// The last [`RenderSettings`] pushed to the renderer (R-P4), so a redundant
    /// `set_settings` (which would reset TAA history) is skipped when the mapped
    /// value is unchanged.
    applied_render: Option<RenderSettings>,
}

/// Map the scene-persisted [`RenderSettingsRecord`] onto a live
/// [`RenderSettings`] (R-P4). The record carries the authorable subset; every
/// other field (hdr, vgeom, tier_override, and the shadow/GI tuning knobs the
/// panel doesn't expose) stays at `RenderSettings::default()`, so
/// `apply_record(&RenderSettingsRecord::default()) == RenderSettings::default()`
/// — the mapping is pinned by a unit test on both sides.
///
/// MIRROR: keep identical to `inf_player::render::apply_record` (the player's
/// copy over `inf_scene::RenderSettingsRecord`). Both seams must agree so the
/// editor viewport and the shipped player apply a level's render block the same.
fn apply_record(r: &RenderSettingsRecord) -> RenderSettings {
    let d = RenderSettings::default();
    RenderSettings {
        exposure: r.exposure,
        dither: r.dither,
        bloom: BloomSettings {
            enabled: r.bloom_enabled,
            threshold: r.bloom_threshold,
            knee: r.bloom_knee,
            intensity: r.bloom_intensity,
            karis: r.bloom_karis,
        },
        ssao: SsaoSettings {
            enabled: r.ssao_enabled,
            radius: r.ssao_radius,
            intensity: r.ssao_intensity,
            bias: r.ssao_bias,
        },
        taa: r.taa,
        // MIRROR (wave SKY2): the cloud's own temporal accumulation follows the
        // level's TAA switch. See the twin in `inf_player::render::apply_record`
        // for why it is two mechanisms behind one authored bit.
        cloud_temporal: r.taa,
        shadows: ShadowSettings {
            enabled: r.shadows_enabled,
            max_distance: r.shadows_max_distance,
            ..d.shadows
        },
        gi: GiSettings {
            enabled: r.gi_enabled,
            intensity: r.gi_intensity,
            ..d.gi
        },
        // ── schema v26 (wave VIS1a): the photoreal arc's authorable surface ──
        //
        // EVERY v26 field is applied here, including the ones whose consumer
        // lands in VIS1b. An unread field is cheap; an unread *wire* field is
        // not — a value an author can type, a codec persists and no seam ever
        // reads is a promise the engine is not keeping, and the seam is the only
        // place that can be checked before the consumer exists.
        ssr: SsrSettings {
            enabled: r.ssr_enabled,
            max_distance: r.ssr_distance,
            thickness: r.ssr_thickness,
            quality: SsrQuality::from_code(r.ssr_quality),
            intensity: r.ssr_intensity,
            roughness_cutoff: r.ssr_roughness_cutoff,
        },
        exposure_control: ExposureSettings {
            mode: ExposureMode::from_code(r.exposure_mode),
            compensation_ev: r.exposure_compensation_ev,
            min_luminance: r.exposure_min_luminance,
            max_luminance: r.exposure_max_luminance,
            adaptation_speed: r.exposure_adaptation_speed,
        },
        flare: FlareSettings {
            enabled: r.flare_enabled,
            intensity: r.flare_intensity,
            ghost_count: r.flare_ghost_count,
            halo: r.flare_halo,
            streak: r.flare_streak,
        },
        film: FilmSettings {
            vignette_intensity: r.vignette_intensity,
            vignette_smoothness: r.vignette_smoothness,
            chromatic_aberration: r.chromatic_aberration,
            grain_intensity: r.grain_intensity,
            grain_size: r.grain_size,
        },
        ..d
    }
}

/// Nearest positive ray/sphere intersection distance, or `None` when the ray
/// misses (P18.3 analytic pick fallback).
///
/// A ray starting *inside* the sphere counts as a hit at `t = 0` — clicking while
/// the camera is inside an object must select it, not fall through it. Pure, so
/// the rule is unit-testable without a GPU.
fn ray_sphere_t(ro: DVec3, rd: DVec3, center: DVec3, radius: f64) -> Option<f64> {
    let m = ro - center;
    let b = m.dot(rd);
    let c = m.dot(m) - radius * radius;
    if c > 0.0 && b > 0.0 {
        return None; // pointing away from a sphere we are outside of
    }
    let disc = b * b - c;
    if disc < 0.0 {
        return None;
    }
    Some((-b - disc.sqrt()).max(0.0))
}

/// The settings the editor viewport **requests** before any tier clamp (P18.3).
///
/// MIRROR of the player's request in `PlayerRenderHost::new`: take the level's
/// authored block and turn the meshlet path **on**, then let the tier and the
/// adapter capability floor clamp it down. Without this the editor's real-mesh
/// content would always fall through to the classic discrete-LOD node — the same
/// geometry, but none of P18.2's streaming, budget or eviction, and a claim of
/// preview-==-shipping that is not true.
///
/// A free function on purpose: it is the whole editor-side render-settings
/// decision, so it unit-tests without a GPU (below), and
/// `tests/projector_mirror.rs` pins the opt-in against the player's copy.
///
/// The editor deliberately does **not** apply `clamp_mobile`: there is no mobile
/// editor, and the player's own mobile branch is `cfg`-gated to targets this crate
/// does not build for.
fn requested_render_settings(record: &RenderSettingsRecord) -> RenderSettings {
    let base = apply_record(record);
    RenderSettings {
        vgeom: inf_render::VgeomSettings {
            enabled: true,
            ..base.vgeom
        },
        ..base
    }
}

/// One projected terrain (P16.6) — the per-terrain state the old single-terrain
/// `terrain_streamed` / `terrain_editable` / `terrain_unsaved_edits` fields held,
/// now one record per terrain and index-aligned with `scene.terrains`.
struct TerrainSlot {
    /// The terrain entity.
    guid: Uuid,
    /// Asset-backed (streamed from a `.inf_terrain`) rather than inline.
    streamed: bool,
    /// Streamed **and** its asset is a writable file the save path can fold edits
    /// into. Always `false` for an inline terrain, which needs no asset at all —
    /// read together with `streamed`: *streamed && !editable* is the one case the
    /// terrain tools refuse.
    editable: bool,
    /// Carries tiles not yet written back to its asset.
    unsaved: bool,
}

/// One terrain the cursor-resolution helpers below consider (P16.6): the entity,
/// the heightfield **actually under the cursor**, and the terrain's world
/// translation.
///
/// The middle field is the load-bearing one. For an inline terrain it is the
/// document's own `TerrainData`; for a **streamed** one the document's set is
/// empty by design (its tiles live in the `.inf_terrain`, and only what the
/// streamer has paged in is real), so it must be the streamer's render working
/// set. Everything that resolves a cursor against terrain — sculpt, paint,
/// drag-drop, foliage — funnels through [`EngineHost::terrain_probes`], which is
/// the one place that choice is made.
struct TerrainProbe<'a> {
    guid: Uuid,
    data: &'a inf_terrain::TerrainData,
    translation: DVec3,
}

/// Where a ray met a terrain: the entity, the hit in that terrain's local XZ, the
/// local surface height, and the world-space point.
#[derive(Debug, Clone, Copy, PartialEq)]
struct TerrainRayHit {
    guid: Uuid,
    local_xz: DVec2,
    local_height: f64,
    world: DVec3,
}

/// Build a [`TerrainProbe`] per slot, in order, resolving each one's heightfield
/// through `resolve` and honouring `restrict` (P16.6).
///
/// Free + generic over the resolver so the **restrict** rule — the thing that
/// pins a stroke to the terrain it started on — unit-tests without a GPU, which
/// an `EngineHost` method could not.
fn terrain_probes_of<'a>(
    slots: &[TerrainSlot],
    restrict: Option<Uuid>,
    mut resolve: impl FnMut(Uuid) -> Option<(&'a inf_terrain::TerrainData, DVec3)>,
) -> Vec<TerrainProbe<'a>> {
    slots
        .iter()
        .filter(|s| !restrict.is_some_and(|g| g != s.guid))
        .filter_map(|s| {
            resolve(s.guid).map(|(data, translation)| TerrainProbe {
                guid: s.guid,
                data,
                translation,
            })
        })
        .collect()
}

/// The **nearest** terrain hit along a world-space ray (P16.6).
///
/// Nearest-along-the-ray is the only defensible rule once terrains can overlap or
/// nest: the surface you can see is the surface a brush must write, and
/// "whichever terrain the document happens to list first" is neither. Ties (two
/// coincident surfaces) resolve to the earlier probe, i.e. document order, so the
/// choice is deterministic rather than dependent on iteration luck.
///
/// Pure, so the rule unit-tests without a GPU.
fn nearest_terrain_hit(
    probes: &[TerrainProbe<'_>],
    ro_w: DVec3,
    rd: DVec3,
) -> Option<TerrainRayHit> {
    let mut best: Option<(f64, TerrainRayHit)> = None;
    for probe in probes {
        let Some(hit) = raycast_terrain(probe.data, ro_w - probe.translation, rd, 1.0e6) else {
            continue;
        };
        let world = probe.translation + hit.point;
        let d = (world - ro_w).length();
        if best.as_ref().is_none_or(|(bd, _)| d < *bd) {
            best = Some((
                d,
                TerrainRayHit {
                    guid: probe.guid,
                    local_xz: DVec2::new(hit.point.x, hit.point.z),
                    local_height: hit.point.y,
                    world,
                },
            ));
        }
    }
    best.map(|(_, hit)| hit)
}

/// The **topmost** terrain surface at world XZ `p`, as `(entity, world height)`.
///
/// Topmost rather than nearest, because this answers "what ground is here?" for
/// things scattered from above (foliage) rather than "what did the cursor hit?".
/// Ties resolve to the earlier probe. Pure, so it unit-tests without a GPU.
fn topmost_surface(probes: &[TerrainProbe<'_>], p: DVec2) -> Option<(Uuid, f64)> {
    let mut best: Option<(f64, Uuid)> = None;
    for probe in probes {
        let local = DVec2::new(p.x - probe.translation.x, p.y - probe.translation.z);
        let Some(h) = probe.data.height_at(local) else {
            continue;
        };
        let y = probe.translation.y + h;
        if best.as_ref().is_none_or(|(by, _)| y > *by) {
            best = Some((y, probe.guid));
        }
    }
    best.map(|(y, g)| (g, y))
}

/// An in-flight sculpt gesture (P10.2b): the mouse-down→up stroke accumulating
/// dabs into one [`Stroke`], plus the state to resample the drag path and, on
/// release, commit one [`inf_terrain::HeightDelta`] undo step.
struct SculptDrag {
    /// Target terrain entity.
    guid: Uuid,
    /// The accumulating stroke (merged into one delta at commit) — a height
    /// [`Stroke`] for the sculpt ops, or a [`SplatStroke`] for the Paint sub-mode.
    kind: DragStroke,
    /// The effective op (Ctrl may flip Raise↔Lower).
    op: SculptOp,
    /// Last dab centre in terrain-local XZ (for even path resampling).
    last_local: DVec2,
    /// Local surface height under the stroke's first touch — the Flatten target.
    flatten_height: f64,
}

/// The in-flight stroke of a [`SculptDrag`]: a height sculpt or a splat paint.
enum DragStroke {
    Height(Stroke),
    Splat(SplatStroke),
    /// A biome-id paint gesture (P19.2). Like the splat arm it captures its
    /// target at `begin`, so retargeting the toolbar mid-drag cannot corrupt an
    /// in-flight stroke.
    Biome(BiomeStroke),
}

/// An in-flight foliage scatter gesture (E-P6): the mouse-down→up stroke that
/// live-mutates the target [`Foliage`] component per tick and, on release,
/// commits ONE `PaintFoliage` undo step. Either adds (`added`) or erases
/// (`removed`) — never both in one stroke.
struct FoliageDrag {
    /// Target foliage entity (selected, or auto-created at stroke start).
    guid: Uuid,
    /// Erase (remove within radius) vs place.
    erase: bool,
    /// This stroke's index, folded into the scatter RNG for determinism.
    stroke_seq: u32,
    /// Running scatter-sample index (monotonic across the whole stroke) so every
    /// candidate draws a distinct deterministic RNG draw.
    next_sample: u64,
    /// Entity world translation captured at stroke start — foliage instances are
    /// entity-local, so world hit points convert through this.
    origin: DVec3,
    /// Local XZ of every instance known this stroke (pre-existing + added), for
    /// O(n) min-spacing rejection (add mode). A v1 simplification — fine at brush
    /// scale; a spatial hash is the follow-up for very dense components.
    positions: Vec<DVec2>,
    /// Instances placed this stroke, in push order (append-only; the undo record
    /// pops exactly these off the end on revert).
    added: Vec<FoliageInstance>,
    /// Snapshot of the component's instances at stroke start (erase mode only).
    original: Vec<FoliageInstance>,
    /// Original-vector indices removed so far this stroke (erase mode).
    removed: BTreeSet<usize>,
}

/// One deterministic scatter candidate produced by [`foliage_samples`]: a world
/// XZ position within the brush disk plus a yaw + uniform scale. The host lifts it
/// to the terrain (or ground) height and converts to entity-local before placing.
#[derive(Debug, Clone, Copy, PartialEq)]
struct FoliageCandidate {
    /// World-space XZ (the `y`/height is resolved by the host against the terrain).
    pos_xz: DVec2,
    /// Yaw about +Y, degrees (euler-deg YXZ, the `Transform` convention).
    yaw_deg: f64,
    /// Uniform scale (`1 ± scale_jitter`).
    scale: f64,
    /// Palette slot.
    kind: u32,
}

/// Hard cap on candidate samples placed per brush tick (keeps a huge-radius
/// high-density brush from stalling the interaction thread).
const FOLIAGE_MAX_PER_TICK: u32 = 64;

/// Deterministic disk sampler for one foliage brush tick (E-P6). **Pure** — the
/// output is a function of the inputs alone (no wall-clock / thread-rng), so the
/// same stroke input sequence reproduces identical instances (unit-tested). Each
/// candidate `i` derives its uniforms from `xxh3_64(seed, stroke_seq,
/// base_index + i)`; the disk sample is area-uniform (`r = R·√u`).
#[allow(clippy::too_many_arguments)] // brush params are a flat list; a struct here would just shuffle them
fn foliage_samples(
    center_xz: DVec2,
    radius: f64,
    count: u32,
    seed: u32,
    stroke_seq: u32,
    base_index: u64,
    scale_jitter: f64,
    kind: u32,
) -> Vec<FoliageCandidate> {
    let jitter = scale_jitter.max(0.0);
    (0..count)
        .map(|i| {
            let h = foliage_hash(seed, stroke_seq, base_index + i as u64);
            // Split the 64-bit hash into four independent [0,1) uniforms.
            let u0 = unit_from_bits(h);
            let u1 = unit_from_bits(h.rotate_left(16).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let u2 = unit_from_bits(h.rotate_left(32).wrapping_mul(0xC2B2_AE3D_27D4_EB4F));
            let u3 = unit_from_bits(h.rotate_left(48).wrapping_mul(0x1656_67B1_9E37_79F9));
            let r = radius * u0.sqrt();
            let theta = std::f64::consts::TAU * u1;
            let pos_xz = center_xz + DVec2::new(r * theta.cos(), r * theta.sin());
            let yaw_deg = 360.0 * u2;
            let scale = (1.0 + (u3 * 2.0 - 1.0) * jitter).max(0.01);
            FoliageCandidate {
                pos_xz,
                yaw_deg,
                scale,
                kind,
            }
        })
        .collect()
}

/// xxh3-64 of the three-word RNG key `(seed, stroke_seq, sample_index)`, packed
/// little-endian. Shared hash family with `inf-graph`/`inf-asset`.
fn foliage_hash(seed: u32, stroke_seq: u32, sample_index: u64) -> u64 {
    let mut bytes = [0u8; 16];
    bytes[0..4].copy_from_slice(&seed.to_le_bytes());
    bytes[4..8].copy_from_slice(&stroke_seq.to_le_bytes());
    bytes[8..16].copy_from_slice(&sample_index.to_le_bytes());
    xxhash_rust::xxh3::xxh3_64(&bytes)
}

/// Map 64 hash bits to a `[0, 1)` uniform (53-bit mantissa, exactly like the
/// canonical `u64 → f64` construction).
fn unit_from_bits(bits: u64) -> f64 {
    (bits >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

/// Min-spacing (world metres) below which a candidate is rejected against an
/// existing instance: derived from the target density so a denser brush packs
/// tighter. `area/instance = 1/density`, so nominal spacing ≈ √(1/density); a
/// 0.7 factor lets the disk fill without a hard grid look. Clamped to a small
/// floor so `density → ∞` can't reject everything.
fn foliage_min_spacing(density: f64) -> f64 {
    if density <= 0.0 {
        return 0.05;
    }
    (0.7 * (1.0 / density).sqrt()).max(0.05)
}

/// Euler-degrees (YXZ) → quaternion, matching `inf_ecs::Transform::quat` exactly
/// so a foliage instance's stored rotation reads the same everywhere.
fn foliage_rot_quat(rot: Vec3d) -> glam::Quat {
    DQuat::from_euler(
        glam::EulerRot::YXZ,
        rot.y.to_radians(),
        rot.x.to_radians(),
        rot.z.to_radians(),
    )
    .as_quat()
}

/// Sample a flat brush ring at `y = 0` around a world XZ centre (the foliage
/// brush's ground-plane fallback when there's no terrain under the cursor).
fn ground_ring(center_xz: DVec2, radius: f64) -> Vec<DVec3> {
    const SEGMENTS: u32 = 32;
    (0..SEGMENTS)
        .map(|i| {
            let a = std::f64::consts::TAU * (i as f64) / (SEGMENTS as f64);
            DVec3::new(
                center_xz.x + radius * a.cos(),
                0.0,
                center_xz.y + radius * a.sin(),
            )
        })
        .collect()
}

/// Vertex `i` of an `n`-segment circle of radius `r` around `center`, in the
/// plane perpendicular to `axis` (`0 = X`, `1 = Y`, `2 = Z`) — the voxel tool's
/// three-circle cut silhouette (P21.2).
///
/// A *sphere* preview and not a ground ring, because the cut is a volume whose
/// depth is the parameter an author most needs to see: a flat ring would draw a
/// 4 m ball and a 4 m disc identically. Editor overlay geometry only — nothing
/// here reaches committed content, which is why plain `std` trig is fine (the
/// Push the twelve edges of an axis-aligned wire box onto a debug-line list.
///
/// One writer, so the pit preview and the dig-to-grade column preview cannot
/// draw the same shape two different ways.
fn push_box_wire(out: &mut Vec<[DVec3; 2]>, lo: DVec3, hi: DVec3) {
    for &y in &[hi.y, lo.y] {
        let c = [
            DVec3::new(lo.x, y, lo.z),
            DVec3::new(hi.x, y, lo.z),
            DVec3::new(hi.x, y, hi.z),
            DVec3::new(lo.x, y, hi.z),
        ];
        for i in 0..4 {
            out.push([c[i], c[(i + 1) % 4]]);
        }
    }
    for (x, z) in [(lo.x, lo.z), (hi.x, lo.z), (hi.x, hi.z), (lo.x, hi.z)] {
        out.push([DVec3::new(x, hi.y, z), DVec3::new(x, lo.y, z)]);
    }
}

/// P14 portability law governs what is *authored*, not what is drawn).
fn circle_point(center: DVec3, r: f64, axis: usize, i: usize, n: usize) -> DVec3 {
    let a = std::f64::consts::TAU * (i % n) as f64 / n as f64;
    let (s, c) = (r * a.sin(), r * a.cos());
    center
        + match axis {
            0 => DVec3::new(0.0, c, s),
            1 => DVec3::new(c, 0.0, s),
            _ => DVec3::new(c, s, 0.0),
        }
}

/// A selected 2D (non-mesh) entity's working transform for the gizmo. World
/// space; mirrors what a mesh instance carries so the writeback path is uniform.
/// Only `translation` is read off Windows (the selection center); the rest feed
/// the gizmo writeback, which is Windows-input-only for now.
#[cfg_attr(not(windows), allow(dead_code))]
#[derive(Debug, Clone, Copy)]
struct Sel2D {
    translation: DVec3,
    rotation: DQuat,
    scale: DVec3,
    /// Half-size estimate (world units) for the focus radius.
    extent: f64,
}

/// A mesh instance's transform captured at gizmo-drag start (M2). The cumulative
/// gizmo delta is applied to this snapshot each frame so snapping is exact.
#[derive(Debug, Clone, Copy)]
struct InstanceXform {
    translation: DVec3,
    rotation: glam::Quat,
    scale: Vec3,
}

/// A selected entity's collider, resolved to world space for debug outlining.
struct ColliderDebug {
    shape: ColliderOutline2D,
    /// Collider offset in the body frame (world units, XY).
    offset: Vec2,
    /// Entity world translation (Z kept so the outline sits in the sprite plane).
    world_pos: DVec3,
    /// Z rotation of the body (radians).
    z_rot: f64,
}

/// A selected entity's 3D collider, resolved to world space for debug outlining.
struct ColliderDebug3D {
    shape: ColliderOutline3D,
    /// Collider offset in the body frame (world units).
    offset: DVec3,
    /// Entity world translation.
    world_pos: DVec3,
    /// Full world orientation of the body.
    rotation: DQuat,
}

/// A selected entity's joint (P12.1), resolved to world-space debug segments: the
/// anchor-to-anchor link plus a small cross marking each anchor.
struct JointDebug {
    segments: Vec<[DVec3; 2]>,
}

/// A [`Volume`]'s editor wireframe (E-P4): the entity's box collider resolved to
/// world space plus the volume's tint. Drawn ALWAYS (not selection-gated) so
/// trigger/blocking regions read while editing; the selection just brightens it.
struct VolumeDebug {
    collider: ColliderDebug3D,
    tint: [f32; 4],
}

/// **The centre and radius that frame a sampled curve** (audit ROAD1).
///
/// Split out of [`EngineHost::spline_focus`] so the arithmetic has a seam a test
/// can reach: the host itself cannot be built without a GPU adapter, and the
/// thing worth pinning is that a long thin run is framed by its whole length and
/// not by a point.
///
/// The radius is half the bounding box's diagonal with a 4 m floor. Half the
/// diagonal rather than the longest side because the camera frames a sphere;
/// the floor because a two-metre puddle framed at two metres is a picture of the
/// inside of a puddle.
fn curve_focus(line: &[DVec3]) -> Option<(DVec3, f64)> {
    let first = *line.first()?;
    let (mut lo, mut hi) = (first, first);
    for p in line {
        lo = lo.min(*p);
        hi = hi.max(*p);
    }
    let centre = (lo + hi) * 0.5;
    Some((centre, ((hi - lo).length() * 0.5).max(4.0)))
}

/// A [`Spline`]'s editor visualization (E-P5): the sampled curve as a world-space
/// polyline plus the world-space control points (for the selected-only markers).
/// Points are cached in world space (the entity transform already applied) and
/// rebased through the floating origin at draw time.
struct SplineDebug {
    /// Sampled curve vertices in world space (consecutive pairs form segments).
    line: Vec<DVec3>,
    /// Control points in world space (a 3-axis cross is drawn at each when the
    /// spline is selected).
    control: Vec<DVec3>,
}

/// A spot [`Light`]'s editor cone gizmo (R-P3): the beam apex, its world-space
/// emission axis, the outer half-angle, an effective draw distance, and the
/// light's colour. Drawn for the current selection only (cheap, per-selection).
struct SpotDebug {
    /// Beam apex (the light's world position).
    apex: DVec3,
    /// Normalized world emission direction (`rot · −Z`).
    axis: DVec3,
    /// Outer-cone half-angle (radians) — the drawn rim.
    outer_rad: f64,
    /// Draw distance: the light's `range`, or 5 m when unbounded (`range == 0`).
    dist: f64,
    /// The light's colour (rgb, opaque).
    color: [f32; 4],
}

impl EngineHost {
    pub fn new(target: SurfaceTarget, width: u32, height: u32) -> Result<Self, String> {
        let (gpu, chain, renderer) = Self::build_gpu_stack(target, width, height)?;
        let picker = Picker::new(&gpu);
        Ok(Self {
            target,
            gpu,
            chain,
            renderer,
            picker,
            scene: RenderScene {
                grid_enabled: true,
                ..Default::default()
            },
            origin: FloatingOrigin::default(),
            gizmo_mode: GizmoMode::Translate,
            gizmo_space: GizmoSpace::World,
            snap_3d: SnapSettings::default(),
            gizmo_drag: None,
            gizmo_initial: HashMap::new(),
            gizmo_initial_2d: HashMap::new(),
            fov_y: 60f32.to_radians(),
            id_to_guid: HashMap::new(),
            guid_to_id: HashMap::new(),
            collider_outlines: HashMap::new(),
            collider_outlines_3d: HashMap::new(),
            joint_lines: HashMap::new(),
            volume_outlines: HashMap::new(),
            spot_lights: HashMap::new(),
            spline_polylines: HashMap::new(),
            selected_guids: Vec::new(),
            synced_version: None,
            seam_dirty: false,
            mode: ViewportMode::Perspective,
            snap_2d: Snap2DSettings::default(),
            selected_2d: HashMap::new(),
            tool_mode: ToolMode::Select,
            sculpt: SculptSettings::default(),
            biome: BiomeSettings::default(),
            biome_palettes: std::collections::BTreeMap::new(),
            terrain_guid: None,
            terrain_slots: Vec::new(),
            terrain_streams: inf_editor_core::terrain_stream::EditorTerrainStreams::new(),
            player_start: None,
            voxel_volumes: inf_editor_core::voxel_store::shared_volumes(),
            debris_cache: inf_render::DebrisCache::default(),
            fractures: inf_editor_core::simulate::shared_fractures(),
            render_assets: inf_editor_core::render_assets::EditorRenderAssets::new(),
            scatter_meshes: inf_render::ScatterMeshes::new(),
            scatter_wanted: Default::default(),
            vt_level_key: Default::default(),
            tool_status: None,
            stream_log_countdown: STREAM_LOG_INTERVAL_FRAMES,
            sculpt_drag: None,
            sculpt_ring: Vec::new(),
            sculpt_ring_color: [1.0; 4],
            foliage: FoliageSettings::default(),
            foliage_drag: None,
            water_tool: WaterSettings::default(),
            water_hints: std::collections::BTreeMap::new(),
            water_active_river: None,
            water_lake_drag: None,
            water_preview: Vec::new(),
            voxel_tool: VoxelSettings::default(),
            voxel_stroke: None,
            voxel_stroke_terrains: Vec::new(),
            voxel_stroke_last: None,
            voxel_path: Vec::new(),
            voxel_box_anchor: None,
            voxel_box_cursor: None,
            voxel_spoil_site: None,
            last_doc_id: None,
            voxel_box_plan: None,
            voxel_preview: Vec::new(),
            voxel_hole_warned: BTreeSet::new(),
            foliage_stroke_seq: 0,
            last_eye_world: DVec3::ZERO,
            render_tier: None,
            render_caps: None,
            applied_render: None,
        })
    }

    fn build_gpu_stack(
        target: SurfaceTarget,
        width: u32,
        height: u32,
    ) -> Result<(GpuContext, SurfaceChain, EngineRenderer), String> {
        let instance = inf_render::create_instance();
        // SAFETY: the native handle outlives the host (the platform module
        // destroys the host before the window/layer).
        let surface = unsafe { target.create_surface(&instance) }?;
        let gpu = GpuContext::for_surface(instance, &surface)?;
        // Interactive viewport: DEGRADE on GPU validation/OOM errors (log +
        // count, keep rendering) instead of aborting the whole editor process,
        // which is wgpu's default. The headless golden/thumbnail paths keep that
        // fatal default so CI still fails hard on validation bugs (M1).
        gpu.install_lenient_error_handler();
        let chain = SurfaceChain::new(&gpu, surface, width, height)?;
        let renderer = EngineRenderer::new(&gpu, chain.target_format());
        Ok((gpu, chain, renderer))
    }

    /// **Forget every cached answer that was about the DEAD device** (Hardening D).
    ///
    /// A fresh `EngineRenderer` starts with `vt: None`, `vt_textures: None`,
    /// `RenderSettings::default()` and `ViewMode::Lit`. The host, meanwhile, keeps
    /// *memos* of what it already pushed into the old one, and every one of those
    /// memos gates an early return:
    ///
    /// * `vt_level_key` — `sync_vt_bindings` returns early on `key == self.…`, so
    ///   the fresh renderer never receives `set_vt_level` and **every
    ///   virtual-textured surface renders untextured for the rest of the
    ///   session**;
    /// * `applied_render` — `apply_render_settings` returns early when the mapped
    ///   block is unchanged, so the level's authored post/exposure/vgeom/VSM/SSAO
    ///   block is never pushed and the viewport silently runs crate defaults;
    /// * `synced_version` — the projection itself is version-gated, and the new
    ///   renderer holds no scene at all;
    /// * `render_tier` / `render_caps` — probes of the OLD adapter, and the new
    ///   device may not be the same one (a TDR can move the app to a different
    ///   GPU).
    ///
    /// The view mode lives only in the renderer, so it is re-pushed rather than
    /// cleared. The old comment on the loss branch — "every remaining field on
    /// `self` is plain CPU/scene data" — was true when it was written and stopped
    /// being true at P26.3/R-P4; the player has done this correctly since P14
    /// (`PlayerRenderHost::rebuild_vt` exists for exactly this case).
    fn reset_device_scoped_state(&mut self, view_mode: inf_render::ViewMode) {
        self.vt_level_key = Default::default();
        self.applied_render = None;
        self.synced_version = None;
        self.render_tier = None;
        self.render_caps = None;
        self.renderer.set_view_mode(view_mode);
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.chain.request_resize(width, height);
    }

    /// The perspective render view for `camera` at the current surface size.
    pub fn view_for(&self, camera: &EditorCamera) -> RenderView {
        let (width, height) = self.chain.requested_size();
        RenderView {
            origin: self.origin,
            eye_world: camera.pos,
            forward: camera.forward(),
            up: Vec3::Y,
            fov_y: self.fov_y,
            near: 0.05,
            width,
            height,
            ortho: None,
        }
    }

    /// The orthographic render view for the 2D camera at the current surface
    /// size: eye above the XY plane looking down -Z, up = +Y, reverse-Z ortho.
    pub fn view_2d(&self, cam: &Camera2D) -> RenderView {
        let (width, height) = self.chain.requested_size();
        RenderView {
            origin: self.origin,
            eye_world: cam.eye(),
            forward: Vec3::NEG_Z,
            up: Vec3::Y,
            fov_y: self.fov_y,
            near: 0.05,
            width,
            height,
            ortho: Some(OrthoParams {
                half_height: cam.half_height as f32,
                near: TWO_D_NEAR,
                far: TWO_D_FAR,
            }),
        }
    }

    /// Current surface size in physical pixels (for camera/gizmo math). Only the
    /// Windows input layer drives the cameras today.
    #[cfg_attr(not(windows), allow(dead_code))]
    pub fn surface_size(&self) -> (u32, u32) {
        self.chain.requested_size()
    }

    /// Switch the active projection (perspective ↔ 2D ortho). The platform loop
    /// keeps a separate camera pose per mode, so switching preserves both.
    pub fn set_mode(&mut self, mode: ViewportMode) {
        self.mode = mode;
    }

    /// Set the renderer's shading view mode (Lit / Unlit / Wireframe / Biomes /
    /// VtResidency). Pure passthrough to the renderer, which clamps
    /// Wireframe→Unlit if the adapter lacks `POLYGON_MODE_LINE`.
    /// Editor-transient (never persisted).
    ///
    /// **The one exception to "pure passthrough"** (P26.5): switching INTO the
    /// residency heat-map writes the streamer's summary to the Output Log. The
    /// ramp answers "which pixels are behind"; the line answers "by how much,
    /// out of how big a pool, and did the budget clamp" — and an author who has
    /// just asked the first question is the one person who wants the second.
    /// It is logged on the transition rather than per frame, because a line
    /// every 16 ms is not a log, it is a denial of service on the panel.
    pub fn set_view_mode(&mut self, mode: inf_render::ViewMode) {
        let entering = self.renderer.view_mode() != mode;
        self.renderer.set_view_mode(mode);
        if entering && mode == inf_render::ViewMode::VtResidency {
            match self.renderer.vt_summary() {
                Some(line) => tracing::info!("inf-viewport: {line}"),
                None => tracing::info!(
                    "inf-viewport: this level binds no virtual textures — the \
                     residency view will be uniformly grey"
                ),
            }
        }
        // P27.5: the same rule for the shadow-page view, and it closes the
        // P27.1 remainder that *"nothing logs `vsm_summary` in a host and no
        // editor surface exists"* — both halves at once, because the ramp
        // answers "which pixels are behind" and the line answers "by how much,
        // out of how big an atlas, and did the budget defer".
        if entering && mode == inf_render::ViewMode::VsmPages {
            match self.renderer.vsm_summary() {
                Some(line) => tracing::info!("inf-viewport: {line}"),
                None => tracing::info!(
                    "inf-viewport: this level runs no virtual shadow maps (the \
                     setting is off, or no light casts) — the page view will be \
                     uniformly grey"
                ),
            }
        }
    }

    /// Replace the 2D-mode snapping configuration (from the toolbar).
    pub fn set_snap_2d(&mut self, snap: Snap2DSettings) {
        self.snap_2d = snap;
    }

    /// Switch the active tool (Select / Sculpt / Foliage / Biome / Water / Voxel)
    /// from the toolbar. Leaving the brush tools drops any hovered brush ring, and
    /// leaving the water tool drops its pending state — an author who switched to
    /// Select and back should not find the next click extending a river they had
    /// finished.
    pub fn set_tool_mode(&mut self, mode: ToolMode) {
        self.tool_mode = mode;
        if !matches!(mode, ToolMode::Sculpt | ToolMode::Foliage | ToolMode::Biome) {
            self.sculpt_ring.clear();
        }
        if mode != ToolMode::Water {
            self.water_active_river = None;
            self.water_lake_drag = None;
            self.water_preview.clear();
        }
        if mode != ToolMode::Voxel {
            // The same rule the water tool follows, and it matters more here: an
            // uncommitted tunnel path is geometry that has not been cut yet, and
            // finding it half-drawn on return would make the next click carve a
            // shape the author stopped thinking about several tools ago.
            self.voxel_path.clear();
            self.voxel_preview.clear();
            // A box-cut drag is the same case as a pending tunnel path: it has
            // cut nothing (the pit is committed on release), so dropping it
            // loses no work and leaves no un-undoable edit behind.
            self.voxel_box_anchor = None;
            self.voxel_box_cursor = None;
            // An in-flight BRUSH stroke is the opposite case — it has already cut
            // — and it is closed by [`settle_orphaned_carve`], which the pump calls
            // with the document in hand. It cannot be closed here: committing an
            // undo entry needs a `&mut SceneDoc` and this seam has none.
        }
    }

    /// Replace the voxel-tool configuration (from the toolbar, P21.2).
    ///
    /// Drops any pending tunnel path for the same reason
    /// [`set_water`](Self::set_water) drops a lake drag: switching Brush ↔ Tunnel
    /// (or changing the radius) mid-path would commit a tube the author described
    /// with one setting and cut with another.
    pub fn set_voxel(&mut self, voxel: VoxelSettings) {
        self.voxel_tool = voxel;
        self.voxel_path.clear();
        self.voxel_preview.clear();
        // …and any half-dragged pit rectangle, for the same reason: a pit
        // described at one depth and cut at another is not the pit the author
        // dragged. The spoil SITE deliberately survives — it is a decision about
        // the level, not about the gesture.
        self.voxel_box_anchor = None;
        self.voxel_box_cursor = None;
        self.voxel_box_plan = None;
    }

    /// Replace the water-tool configuration (from the toolbar, P20.4).
    ///
    /// Drops any in-flight lake drag: flipping Lake → River mid-drag would
    /// otherwise leave the anchor set, and the next River click would finish a
    /// rectangle the author had abandoned.
    pub fn set_water(&mut self, water: WaterSettings) {
        self.water_tool = water;
        self.water_lake_drag = None;
        self.water_preview.clear();
    }

    /// Push a terrain entity's resolved per-biome water-level hints (P20.4) —
    /// the `set_biome_palette` twin. An empty table clears the entry, so
    /// unbinding a biome set stops the tool suggesting levels from a set that is
    /// no longer bound.
    pub fn set_water_hints(&mut self, guid: Uuid, hints: Vec<Option<f64>>) {
        if hints.iter().all(|h| h.is_none()) {
            self.water_hints.remove(&guid);
        } else {
            self.water_hints.insert(guid, hints);
        }
    }

    /// Replace the sculpt brush configuration (from the toolbar).
    pub fn set_sculpt(&mut self, sculpt: SculptSettings) {
        self.sculpt = sculpt;
    }

    /// Replace the biome brush configuration (from the toolbar, P19.2).
    pub fn set_biome(&mut self, biome: BiomeSettings) {
        self.biome = biome;
    }

    /// Push a terrain entity's resolved biome palette (P19.2) — see
    /// [`biome_palettes`](Self::biome_palettes). An empty palette clears the
    /// entry, so unbinding a biome set does not leave a stale overlay behind.
    pub fn set_biome_palette(&mut self, guid: Uuid, palette: Vec<[f32; 4]>) {
        if palette.is_empty() {
            self.biome_palettes.remove(&guid);
        } else {
            self.biome_palettes.insert(guid, palette);
        }
    }

    /// Replace the foliage brush configuration (from the toolbar, E-P6).
    pub fn set_foliage(&mut self, foliage: FoliageSettings) {
        self.foliage = foliage;
    }

    /// Translate snap increment (world units) for 2D mode, `0.0` ⇒ none. Only
    /// the Windows input layer applies it during a gizmo drag.
    #[cfg_attr(not(windows), allow(dead_code))]
    pub fn snap_2d_translate(&self) -> f32 {
        self.snap_2d.translate_snap()
    }

    /// The world transform of the render instance carrying pick id `id`, wherever
    /// it lives (P18.3).
    ///
    /// Until this batch every renderable entity was a [`MeshInstance`], so the
    /// selection-driven affordances — the gizmo snapshot, focus framing, the
    /// Local-space basis, the transform write-back — all searched one list. A
    /// `MeshRef.asset` is now a [`VgeomInstance`](inf_render::VgeomInstance) and a
    /// bound `SkeletalMesh` a [`SkinnedInstance`](inf_render::SkinnedInstance), and
    /// **an imported mesh must be exactly as manipulable as a cube**. Every one of
    /// those call sites reads through here instead, so that is true by construction
    /// rather than by remembering to add a third branch each time.
    ///
    /// Ids are unique across the three lists (one `next_id` feeds them all), so
    /// the first hit is the only hit.
    fn instance_xform(&self, id: u32) -> Option<InstanceXform> {
        if let Some(i) = self.scene.instances.iter().find(|i| i.id == id) {
            return Some(InstanceXform {
                translation: i.translation,
                rotation: i.rotation,
                scale: i.scale,
            });
        }
        if let Some(i) = self.scene.vgeom_instances.iter().find(|i| i.id == id) {
            return Some(InstanceXform {
                translation: i.translation,
                rotation: i.rotation,
                scale: i.scale,
            });
        }
        self.scene
            .skinned
            .iter()
            .find(|i| i.id == id)
            .map(|i| InstanceXform {
                translation: i.translation,
                rotation: i.rotation,
                scale: i.scale,
            })
    }

    /// Write a transform back onto whichever render list carries `id` — the
    /// mutable twin of [`instance_xform`](Self::instance_xform).
    fn set_instance_xform(&mut self, id: u32, x: InstanceXform) {
        if let Some(i) = self.scene.instances.iter_mut().find(|i| i.id == id) {
            i.translation = x.translation;
            i.rotation = x.rotation;
            i.scale = x.scale;
            return;
        }
        if let Some(i) = self.scene.vgeom_instances.iter_mut().find(|i| i.id == id) {
            i.translation = x.translation;
            i.rotation = x.rotation;
            i.scale = x.scale;
            return;
        }
        if let Some(i) = self.scene.skinned.iter_mut().find(|i| i.id == id) {
            i.translation = x.translation;
            i.rotation = x.rotation;
            i.scale = x.scale;
        }
    }

    /// World-space center of the current selection, if any. Reads LIVE working
    /// positions (render instances + selected 2D entities) so it tracks a gizmo
    /// drag in progress.
    fn selection_center(&self) -> Option<DVec3> {
        let mut sum = DVec3::ZERO;
        let mut n = 0.0;
        for id in &self.scene.selected {
            if let Some(inst) = self.instance_xform(*id) {
                sum += inst.translation;
                n += 1.0;
            }
        }
        // …and the selected entities that are a CURVE rather than an instance.
        // `scene.selected` holds render-instance ids and a river has none, so
        // the document's own selection is the list that can see one.
        for guid in &self.selected_guids {
            if let Some((mid, _)) = self.spline_focus(*guid) {
                sum += mid;
                n += 1.0;
            }
        }
        for s in self.selected_2d.values() {
            sum += s.translation;
            n += 1.0;
        }
        (n > 0.0).then(|| sum / n)
    }

    /// **Where a curve-shaped entity actually is** — the midpoint of its sampled
    /// spline and a radius that holds the whole run (audit ROAD1).
    ///
    /// # The actor whose position is not where it is
    ///
    /// A river, a road spline or a tunnel path is an entity whose geometry lives
    /// entirely in its [`Spline`] component: the `Transform` is the identity and
    /// there is no `MeshRef`, so it never becomes a render instance. `Focus
    /// Selection` therefore found nothing for it, `selection_center` fell
    /// through to the player start, and on the island — whose rivers are exactly
    /// this shape — three attempts at framing one landed on the world origin.
    /// That is wave ROAD1's carried item 12, and it is why that wave could not
    /// photograph the river it had just measured.
    ///
    /// The polyline is already projected for the editor's own spline overlay
    /// (`spline_polylines`, world space, entity transform applied), so this
    /// costs a lookup and a walk over points the frame already holds.
    ///
    /// The radius is half the run's diagonal with a 4 m floor, which frames a
    /// whole watercourse rather than one bend — a 2 km river framed at its
    /// midpoint with a 2 m radius is a picture of some water.
    ///
    /// An entity that HAS a render instance is answered `None`: the instance
    /// already contributes its own position and extent, and counting it twice
    /// would drag the framed centre toward whichever half of it is a curve.
    fn spline_focus(&self, id: uuid::Uuid) -> Option<(DVec3, f64)> {
        if self.guid_to_id.contains_key(&id) {
            return None;
        }
        curve_focus(&self.spline_polylines.get(&id)?.line)
    }

    /// Rebuild the render projection from the shared document when it changed
    /// (P3.2). Renderable entities (those with a `MeshRef`) become instances;
    /// the id↔GUID maps let picks and gizmo writeback cross back to the world.
    /// Skipped mid-drag so an in-flight gizmo edit isn't clobbered.
    pub fn sync_from_doc(&mut self, doc: &SceneDoc) {
        if self.gizmo_drag.is_some() {
            return;
        }
        let version = doc.version();
        if self.synced_version == Some(version) {
            return;
        }
        self.synced_version = Some(version);
        self.rebuild_scene(doc);
        self.apply_render_settings(doc);
        self.check_inline_holes(doc);
    }

    /// Apply the scene-persisted render block (post/exposure/lighting) to the live
    /// renderer (R-P4). The tier is probed once from the adapter and clamps the
    /// mapped settings down (never up), mirroring the player. A redundant push is
    /// skipped (cached in `applied_render`) so an unrelated document edit doesn't
    /// needlessly reset TAA history. Runs from `sync_from_doc` (version-gated), so
    /// an `edit_settings` — which bumps the version — flows straight through.
    ///
    /// **P18.3 — the editor asks for the meshlet path.** It never did, which meant
    /// every vgeom asset the viewport now carries would have drawn through
    /// `ClassicVgeomNode`'s discrete-LOD fallback: correct pixels, but not the
    /// streamed, budgeted, evicting P18.2 path the player uses, and therefore not
    /// "the editor streams meshlets exactly as the player does". The request is the
    /// player's, character for character (see `requested_render_settings`); the
    /// clamps below are what decide whether it is granted.
    ///
    /// Both clamps apply, from **cached** probes: `RenderTier::apply` (no meshlet
    /// path below High) and `AdapterCaps::clamp_occlusion` (the storage-texture
    /// floor two-pass occlusion needs). That pair is exactly
    /// [`inf_render::detect_and_clamp`], inlined only so the adapter is probed once
    /// per host rather than on every document version — and it closes P18.1's
    /// honest remainder (1) for the editor, which noted that this host applied the
    /// tier without the occlusion floor.
    fn apply_render_settings(&mut self, doc: &SceneDoc) {
        let tier = *self
            .render_tier
            .get_or_insert_with(|| detect_tier(&self.gpu, &RenderSettings::default()));
        let caps = *self
            .render_caps
            .get_or_insert_with(|| inf_render::AdapterCaps::probe(&self.gpu));
        let requested = requested_render_settings(&doc.settings().render);
        let mapped = caps.clamp_occlusion(tier.apply(requested));
        if self.applied_render != Some(mapped) {
            self.renderer.set_settings(mapped);
            self.applied_render = Some(mapped);
        }
    }

    fn rebuild_scene(&mut self, doc: &SceneDoc) {
        // P21.1 — the voxel bind PRE-PASS, before anything is projected.
        //
        // Loading a volume is a `&mut` act and projecting is not, so it runs here
        // rather than inside the entity loop, for three reasons that are really
        // one: (1) the projection stays read-only, which is what lets
        // `projector_mirror`'s "neither host meshes inside its projection" claim be
        // true rather than aspirational; (2) the live set is built from
        // **bound-ness**, not from whether the volume happened to produce
        // triangles — a volume that meshes to no surface (a chunk set that is all
        // air, or all rock) used to be released at the end of every projection and
        // re-read from disk on the next one, which at a gizmo drag's document-bump
        // rate is a disk read per input event; (3) it is the exact shape
        // `inf_player::render::PlayerRenderHost::sync_voxels` has, so the two hosts
        // page the same way.
        self.sync_voxels(doc);
        self.scene.instances.clear();
        self.scene.lights.clear();
        self.scene.sprites.clear();
        self.scene.tilemaps.clear();
        self.scene.prebatched.clear();
        self.scene.lights_2d.clear();
        // P16.3b1 + Hardening Wave E: terrains are NOT cleared — they are
        // **stamp-gated**, like `deform` below and unlike everything above. Last
        // frame's list is taken out and each entry is either carried forward whole
        // (nothing about its grid, layers or per-tile stamps moved) or dropped and
        // rebuilt. What is left in `prev_terrains` at the end of the walk is
        // exactly the terrains that left the scene, which is how a disappearance
        // is seen.
        let mut prev_terrains = std::mem::take(&mut self.scene.terrains);
        // P18.3: real geometry. `MeshRef.asset` entities project as virtualized
        // geometry (meshlet path, or the classic discrete-LOD fallback — the tier
        // decides which node draws it), and `SkeletalMesh` entities project as
        // real skinned draws. Both lists are rebuilt from scratch every projection,
        // exactly like `instances`.
        self.scene.vgeom_assets.clear();
        self.scene.vgeom_instances.clear();
        self.scene.skinned_meshes.clear();
        self.scene.skinned.clear();
        // P18.5 + island wave I8a's audit: the scatter LIST is rebuilt from
        // scratch every projection, and the PAYLOADS behind it are not. The memo
        // itself is taken out further down — AFTER `sync_scatter_meshes`, because
        // this host rebuilds its mesh table per projection and the table's stamp
        // is part of the memo's key.
        self.scene.scatter.clear();
        // P20.1: water bodies are rebuilt from scratch every projection, like
        // `scatter` — a body's whole state is a pure function of its component, its
        // spline and the level clock, so there is nothing to carry over.
        self.scene.waters.clear();
        // P21.1 + Hardening Wave E: volumetric terrain, on the same stamp gate.
        // The meshed surface behind each chunk lives in the store and is re-meshed
        // only where the field moved, so a settled cave now costs the comparison
        // of its chunk stamps and no copying at all — it used to cost two rebases
        // of every vertex stream, a mapped copy and an index clone, every frame.
        let mut prev_voxels = std::mem::take(&mut self.scene.voxels);
        let (terrains_before, voxels_before) = (prev_terrains.len(), prev_voxels.len());
        self.scene.fracture_chunks.clear();
        // P22.1: the deformation field is NOT cleared — it is epoch-gated, because
        // it is the one projected thing that is usually identical to last frame's.
        // See `project_deform`, which is also where `None` is written when there is
        // no field at all. It reads the DOCUMENT's world, which during Simulate is
        // the very world `SimSession::fixed_step` presses footprints into — so the
        // editor draws what it simulates, with no Ring-2 fold in between.
        project_deform(&mut self.scene, inf_ecs::deform::deform_field(doc.world()));
        // Where the player starts (wave EDIT1, clause 2). Refreshed with every
        // projection because a level open, an undone delete and a dragged pawn
        // all move it, and all three bump the version this walk is gated on.
        self.player_start = player_start_of(doc);
        self.terrain_slots.clear();
        // `terrain_guid` (the tool target) is deliberately NOT cleared here — it
        // is re-validated against the new slot list at the end of the projection,
        // so a sculpt stroke (which bumps the document version on every dab, and
        // therefore re-projects) keeps aiming at the terrain it is editing.
        self.id_to_guid.clear();
        self.guid_to_id.clear();
        self.collider_outlines.clear();
        self.collider_outlines_3d.clear();
        self.joint_lines.clear();
        self.volume_outlines.clear();
        self.spot_lights.clear();
        self.spline_polylines.clear();

        let world = doc.world();
        // Hardening D — the two per-entity side tables that are NOT rebuilt by the
        // projection. `biome_palettes` and `water_hints` are pushed in from Ring 2
        // (`set_biome_palette` / `set_water_hints`), so they cannot be cleared at
        // the top of a sync like the tables above — clearing would drop a palette
        // set between two document versions. What they *can* be is retained to the
        // document: a `Guid` the document no longer carries is an entity that was
        // deleted, and its palette (a `Vec` per terrain, and a
        // `Vec<Option<f64>>` per water body sized by its spline) outlived it for
        // the life of the host.
        {
            let live: std::collections::BTreeSet<Uuid> = doc.order().iter().copied().collect();
            self.biome_palettes.retain(|guid, _| live.contains(guid));
            self.water_hints.retain(|guid, _| live.contains(guid));
        }
        // The sky authority first (P17.1): it writes `scene.sun` / `scene.sky` and,
        // when a clock is present, pushes the sun/moon directional light as
        // `lights[0]` — a stable index on both projector sides.
        project_sky(&mut self.scene, world);

        // **The hour, resolved once** (island wave I8b clause 3). `project_sky` has
        // just written `scene.sun`, so the sun's own height is in hand and the night
        // glow is a function of it rather than of a second read of the clock. The
        // step is QUANTIZED because it is part of the scatter memo's key -- see
        // `inf_render::night_glow_step`.
        //
        // **Wave VEN1a pairs it with the pulse tick**: a venue's string lights
        // breathe as a pure function of the LEVEL clock (never a frame index, never
        // a wall clock), quantized on the same argument. `water_environment` below
        // is the Ring-0 door that says what "now" is, so both hosts read one clock.
        // MIRROR: the same lines in the other host, in the same place.
        // The LEVEL clock, resolved once, in seconds. `water_environment` is
        // the Ring-0 door that says what "now" is (P20.4), so a venue's stage
        // wash, a festoon's breath and a river's swell all read one number.
        let clock_s = inf_ecs::sky::water_environment(world).0;
        let clock = inf_render::ScatterClock {
            glow_step: inf_render::night_glow_step(self.scene.sun.direction),
            pulse_tick: inf_render::pulse_tick(clock_s),
        };
        // The clock and wind every water body responds to, resolved ONCE per
        // projection in Ring 0 (`inf_ecs::sky`) so the two MIRROR projectors cannot
        // disagree about what "now" and "the wind" mean — the same reasoning that put
        // `ResolvedSky::cloud_time_s` there.
        let water_env = inf_ecs::sky::water_environment(world);
        // P20.4: the level's terrains, borrowed once, so a river's foam can read
        // the P19.1 flow map. Resolved HERE rather than per body for the same
        // reason `water_env` is — the rule for "which terrain answers" is Ring-0
        // (`inf_ecs::hydro`) so the two MIRROR projectors cannot each invent one.
        let water_flow = inf_ecs::hydro::terrain_flow(world);
        // P26.4, clause 0 — THE REGISTRATION, before anything is projected.
        // Every `Material.asset` the document binds becomes a virtual-texture
        // registry + atlas, and the per-instance sets below are looked up out of
        // it. Rebuilt only when the BINDING SET changes, because building it
        // creates GPU resources and a projection runs on every document version.
        self.sync_vt_bindings(doc);
        // Wave TER2b, on the same terms and for the same reason: every `.inf_mesh`
        // this document's scattered instances name, resolved into flat pull arrays
        // BEFORE the walk. `push_scatter` gets a finished table because the
        // shipped player's projector runs per frame and must not open a file, and
        // the two projectors are pinned character for character.
        self.sync_scatter_meshes(doc);
        // …and now the scatter memo, whose key folds that table's own stamp
        // (island wave I8a audit). Last projection's payloads are taken out here
        // and re-filled by `carry_or_push_pcg_scatter` as the walk goes; what is
        // left at the end is exactly the scatter that left the scene, and it is
        // dropped. MIRROR: `inf_player::render::project_scene_full` takes the same
        // two locals under the same names.
        let mut prev_scatter = std::mem::take(&mut self.scene.scatter_memo);
        let scatter_table = inf_render::scatter_table_stamp(&self.scatter_meshes);
        let w = world.world();
        // The live registry, borrowed for the whole projection: every instance's
        // texture set is a lookup in it (P26.4). `None` on a level with no
        // bindings, and then every set below is `VtTextureSet::NONE`.
        let vt = self.renderer.vt_textures();
        let mut next_id: u32 = 1;
        // Which vgeom assets this projection has already listed (the render node
        // caches GPU geometry by id, but the asset list must not duplicate), and
        // which `(mesh, skeleton)` pairs already own a `skinned_meshes` slot.
        // MIRROR: `vgeom_seen` is the player's `project_scene` local of the same
        // name and the same purpose.
        let mut vgeom_seen: BTreeSet<u128> = BTreeSet::new();
        let mut skinned_slots: HashMap<(Uuid, Uuid), usize> = HashMap::new();
        // Every render asset this projection actually referenced (meshes,
        // skeletons, clips) — the input to the end-of-projection `retain_only`
        // audit (P16.4b's lesson in mesh form).
        let mut live_render_assets: BTreeSet<Uuid> = BTreeSet::new();
        for &guid in doc.order() {
            let Some(entity) = world.entity_of(guid) else {
                continue;
            };
            let visible = w
                .get::<ComputedVisibility>(entity)
                .map(|c| c.0)
                .unwrap_or(true);

            // Lights project into the renderer's light list (P7.1).
            if let Some(light) = w.get::<Light>(entity) {
                if visible {
                    let affine = w
                        .get::<GlobalTransform>(entity)
                        .map(|g| g.0)
                        .unwrap_or(glam::DAffine3::IDENTITY);
                    self.scene.lights.push(project_light(light, &affine));
                    // Cache a cone gizmo for spot lights (R-P3), drawn for the
                    // selection only in `render_frame`.
                    if light.kind == EcsLightKind::Spot {
                        let (_, rot, translation) = affine.to_scale_rotation_translation();
                        let c = light.color.to_array();
                        self.spot_lights.insert(
                            guid,
                            SpotDebug {
                                apex: translation,
                                axis: (rot * -DVec3::Z).normalize(),
                                outer_rad: light.outer_cone_deg.to_radians() as f64,
                                dist: if light.range > 0.0 {
                                    light.range as f64
                                } else {
                                    5.0
                                },
                                color: [c[0], c[1], c[2], 1.0],
                            },
                        );
                    }
                }
            }

            // Sprites project into the 2D sprite list (P8.1a). A sprite entity
            // usually has no MeshRef, so this happens before the mesh gate.
            if let Some(sprite) = w.get::<Sprite>(entity) {
                if visible {
                    let translation = w
                        .get::<GlobalTransform>(entity)
                        .map(|g| g.translation())
                        .unwrap_or(DVec3::ZERO);
                    self.scene.sprites.push(project_sprite(sprite, translation));
                }
            }

            // 2D lights project into the sprite pass's light list (P8.1c).
            if let Some(light2d) = w.get::<Light2D>(entity) {
                if visible {
                    let translation = w
                        .get::<GlobalTransform>(entity)
                        .map(|g| g.translation())
                        .unwrap_or(DVec3::ZERO);
                    self.scene
                        .lights_2d
                        .push(project_light2d(light2d, translation));
                }
            }

            // 9-slices expand to nine quads (P8.1c), pushed as one prebatched run.
            if let Some(nine) = w.get::<NineSlice>(entity) {
                if visible {
                    let translation = w
                        .get::<GlobalTransform>(entity)
                        .map(|g| g.translation())
                        .unwrap_or(DVec3::ZERO);
                    self.scene
                        .prebatched
                        .push(project_nine_slice(nine, translation));
                }
            }

            // Text blocks expand to one quad per glyph (P8.1c), one prebatched run.
            if let Some(text) = w.get::<Text2D>(entity) {
                if visible {
                    let translation = w
                        .get::<GlobalTransform>(entity)
                        .map(|g| g.translation())
                        .unwrap_or(DVec3::ZERO);
                    if let Some(run) = project_text(text, translation) {
                        self.scene.prebatched.push(run);
                    }
                }
            }

            // Tilemaps project into the 2D tilemap list (P8.1b); the sprite pass
            // culls + expands their chunks each frame.
            if let Some(tilemap) = w.get::<Tilemap>(entity) {
                if visible && !tilemap.is_empty() {
                    let translation = w
                        .get::<GlobalTransform>(entity)
                        .map(|g| g.translation())
                        .unwrap_or(DVec3::ZERO);
                    self.scene
                        .tilemaps
                        .push(project_tilemap(tilemap, translation));
                }
            }

            // Heightfield terrain (P10.1) projects into the render scene's terrain
            // list; the terrain pass assembles clipmap LOD rings around the camera
            // for each one, every frame. Each tile carries its own change stamp
            // (P16.3b1), so re-projecting on a document change re-uploads only the
            // tiles a sculpt/paint stroke actually touched.
            //
            // P16.6 — MULTI-TERRAIN: "first visible terrain wins" is gone. EVERY
            // visible, non-empty terrain projects, in **document order**, and the
            // parallel `terrain_slots` records which of them is streamed/editable/
            // dirty.
            //
            // MIRROR, precisely: `inf_player::render::project_scene` runs the same
            // projection but walks its world in `Guid` order — the editor has a
            // document and the player does not. Both orders are deterministic for
            // their own side; what makes a PIE-vs-shipping comparison meaningful is
            // that both stamp the SAME `RenderTerrain::id` from the entity `Guid`,
            // so the two lists match up by identity rather than by index.
            //
            // P16.3b2 — THE SIM/RENDER SPLIT: an asset-backed terrain draws the
            // **streamer's** camera-driven working set; the document's `data` stays
            // exactly as authored (empty, for a streamed terrain) and is never
            // written by the camera. An inline terrain has no stream and projects
            // its own data, unchanged.
            if let Some(terrain) = w.get::<Terrain>(entity) {
                if visible {
                    let translation = w
                        .get::<GlobalTransform>(entity)
                        .map(|g| g.translation())
                        .unwrap_or(DVec3::ZERO);
                    let streamed = self.terrain_streams.ensure(
                        guid,
                        terrain,
                        translation,
                        self.last_eye_world,
                    );
                    // P16.4b — the document's authored tiles are the authority
                    // for a streamed terrain, so mirror them into the render
                    // set (pinned) before projecting. Copies nothing when
                    // nothing was edited.
                    if streamed {
                        self.terrain_streams.overlay_document_edits(guid, doc);
                    }
                    let palette: &[[f32; 4]] = self
                        .biome_palettes
                        .get(&guid)
                        .map(|p| p.as_slice())
                        .unwrap_or(&[]);
                    let projected = if streamed {
                        self.terrain_streams
                            .render_data(guid)
                            .filter(|d| d.tile_count() + d.coarse_tile_count() > 0)
                            .map(|d| {
                                project_terrain(
                                    guid,
                                    terrain,
                                    d,
                                    translation,
                                    palette,
                                    vt,
                                    &mut prev_terrains,
                                )
                            })
                    } else if !terrain.data.is_empty() {
                        Some(project_terrain(
                            guid,
                            terrain,
                            &terrain.data,
                            translation,
                            palette,
                            vt,
                            &mut prev_terrains,
                        ))
                    } else {
                        None
                    };
                    if let Some(rt) = projected {
                        self.scene.terrains.push(rt);
                        self.terrain_slots.push(TerrainSlot {
                            guid,
                            streamed,
                            editable: streamed && self.terrain_streams.is_editable(guid),
                            unsaved: streamed && terrain.data.has_dirty_tiles(),
                        });
                    }
                }
            }

            // Volumetric terrain (P21.1): the SDF chunk volume that locally
            // extends the heightfield — caves, tunnels, overhangs. Its chunks live
            // in a `.inf_voxel` the Ring-1 store resolves under the content root;
            // a volume it cannot serve simply has no slot and draws nothing.
            // MIRROR: `inf_player::render` runs the same branch (minus the
            // visibility gate and the pick-id map, both host-local) through the
            // same `project_voxel` body.
            if visible && w.get::<VoxelVolume>(entity).is_some() {
                let translation = w
                    .get::<GlobalTransform>(entity)
                    .map(|g| g.translation())
                    .unwrap_or(DVec3::ZERO);
                // READ-ONLY: `sync_voxels` above did the binding, so nothing here
                // loads, parses or meshes.
                let projected = self.voxel_volumes.lock().ok().and_then(|v| {
                    v.slot(guid).and_then(|slot| {
                        project_voxel(
                            slot,
                            w.get::<Terrain>(entity),
                            translation,
                            inf_render::terrain_id_from_guid(guid.as_u128()),
                            vt,
                            &mut prev_voxels,
                        )
                    })
                });
                if let Some(rv) = projected {
                    self.scene.voxels.push(rv);
                }
            }

            // PCG scatter volumes (P10.5b): project the cached evaluated
            // instances (refreshed on demand by `pcg_evaluate`) as ONE GPU-instanced
            // scatter batch of placeholder cubes — kind→GUID→real-mesh upload is
            // the same documented viewport gap as sprites/tilemaps, so PCG proves
            // placement/density/orientation with primitives.
            //
            // P18.5 — **THE CPU DRAW-DISTANCE CULL IS GONE.** This branch used to
            // skip instances farther than `vol.draw_distance` from
            // `self.last_eye_world`; the field now rides on the batch and the GPU
            // cull honours it. That is what finally makes the two hosts agree about
            // it: the player ignored the field entirely, so a shipped build drew
            // strictly more scatter than its preview.
            //
            // A pick on the scatter resolves to the volume entity (id→guid), so the
            // volume is selectable by clicking its content.
            // **`.filter(|_| visible)` since wave FIX3, and it is a lighting fix.**
            // The player's whole projection loop sits behind
            // `if !visible { continue; }` (`inf_player::render`), so hiding a
            // venue volume in a shipped build turns its lamps off. The editor
            // applies visibility per COMPONENT, and the rig block below sat
            // outside every guard — so hiding the volume in the viewport left the
            // stage lit and the same level went dark in PIE. Two different worlds
            // on one machine, on the half of the frame the MIRROR comment inside
            // this block claims is identical. The scatter half below already had
            // its own `visible &&`, so this changes nothing for it.
            if let Some(vol) = w.get::<PcgVolume>(entity).filter(|_| visible) {
                // **THE VENUE RIG** (wave VEN1a): the real, coloured, cone-shaped
                // lights a grammar-built venue hangs over its stage and behind its
                // bar. PCG produced no light of any kind until this wave.
                //
                // Emission -- a lit pane, a neon plate, a festoon -- is free against
                // the 16-light frame ceiling and bounces through the GI volume,
                // which is why clause 1 spent itself on it. What emission cannot
                // throw is a shaped pool with a soft edge, and that is a cone.
                //
                // **This is the ONE place a fixture's colour is resolved.** The
                // derived `ScatteredLight` carries the two colours it sweeps between
                // and its own phase slot; which of them it is showing NOW is a pure
                // function of the LEVEL clock through `inf_render::swept_colour`, so
                // PIE and the shipped player agree byte for byte and a golden
                // renders one frame from cold.
                //
                // Outside the `evaluated.is_empty()` guard below, deliberately: a
                // rig is derived beside the population and is not part of it, and a
                // volume with lights and no instances is a defect this branch should
                // make visible rather than hide.
                //
                // MIRROR: the same block in the other host, in the same place.
                for l in &vol.lights {
                    let colour =
                        inf_render::swept_colour(l.sweep, l.cycle_hz, l.phase, l.phases, clock_s);
                    // A cone that covers the sphere IS a point light, and the rig
                    // says so with a 180-degree outer angle rather than a second
                    // type. Resolved here, once, because `RenderLight` has a kind.
                    let point = l.outer_deg >= 180.0;
                    self.scene.lights.push(RenderLight {
                        kind: if point {
                            LightKind::Point
                        } else {
                            LightKind::Spot
                        },
                        color: colour,
                        intensity: l.intensity,
                        // The renderer's convention is TOWARD the light; a fixture
                        // carries the direction its beam is emitted along.
                        direction: (-l.dir).as_vec3(),
                        position: l.at,
                        range: l.range_m,
                        inner_cos: l.inner_deg.to_radians().cos(),
                        outer_cos: l.outer_deg.to_radians().cos(),
                        // **Never a shadow caster.** A venue rig is three lamps in
                        // one room; giving each its own virtual-shadow quadtree
                        // would spend `VSM_MAX_PROJECTIONS` on a pool whose whole
                        // content is a stage floor and two benches.
                        cast_shadows: false,
                    });
                }
                if visible && !vol.evaluated.is_empty() {
                    let translation = w
                        .get::<GlobalTransform>(entity)
                        .map(|g| g.translation())
                        .unwrap_or(DVec3::ZERO);
                    let id = next_id;
                    next_id += 1;
                    carry_or_push_pcg_scatter(
                        &mut self.scene,
                        &mut prev_scatter,
                        scatter_table,
                        guid,
                        vol,
                        &self.scatter_meshes,
                        translation,
                        id,
                        clock,
                    );
                    // ONE row per batch, not one per instance: a pick anywhere in
                    // the scatter selects the owning volume, and the map stops
                    // carrying 100k rows to say the same thing.
                    self.id_to_guid.insert(id, guid);
                }
            }

            // P19.3 — THE TERRAIN'S BIOME POPULATION: the terrain-level sibling of
            // the volume branch above. Each painted biome's `.inf_pcg` graph is
            // evaluated over the region its id owns and the merged result lives in
            // the derived, never-persisted `Terrain::biome_population` (rebuilt by
            // the editor's evaluate command, and by the player on level load —
            // which is exactly what makes the two paths comparable).
            //
            // It goes through the SAME `push_scatter` body as a volume, so a
            // population cannot be packed, shaded, culled or picked differently
            // from a volume's scatter. A pick resolves to the terrain entity.
            //
            // MIRROR: `inf_player::render` runs the same branch (minus the
            // visibility gate and the pick-id map, both host-local).
            if let Some(terrain) = w.get::<Terrain>(entity) {
                if visible && !terrain.biome_population.is_empty() {
                    let translation = w
                        .get::<GlobalTransform>(entity)
                        .map(|g| g.translation())
                        .unwrap_or(DVec3::ZERO);
                    let id = next_id;
                    next_id += 1;
                    push_biome_population(
                        &mut self.scene,
                        terrain,
                        &self.scatter_meshes,
                        translation,
                        id,
                        clock,
                    );
                    self.id_to_guid.insert(id, guid);
                }
            }

            // Water surfaces (P20.1): an ocean, a lake or a spline river. A river
            // reads the `Spline` on THIS SAME ENTITY for its centreline — component
            // composition, not a reference, so there is nothing to resolve and
            // nothing to dangle.
            // MIRROR: `inf_player::render` runs the same branch (minus the pick-id
            // map, which is host-local), through the same `project_water` body.
            if let Some(water) = w.get::<WaterBody>(entity) {
                if visible {
                    let affine = w
                        .get::<GlobalTransform>(entity)
                        .map(|g| g.0)
                        .unwrap_or(glam::DAffine3::IDENTITY);
                    let id = next_id;
                    next_id += 1;
                    let body = project_water(
                        water,
                        w.get::<Spline>(entity),
                        &affine,
                        water_env,
                        &water_flow,
                        id,
                    );
                    if body.drawable() {
                        self.scene.waters.push(body);
                        // ONE row per body: a pick anywhere on the surface selects
                        // the owning entity.
                        self.id_to_guid.insert(id, guid);
                    }
                }
            }
            // Foliage scatter (P18.5): painted instances project as GPU-instanced
            // scatter batches, mesh + tint taken from the referenced palette slot,
            // one batch per primitive kind the palette resolves. Instances are
            // entity-LOCAL, so the entity's world translation is the batch ANCHOR
            // (the auto-created container sits at the origin; applying the
            // container's rotation/scale to instances is a documented v1
            // follow-up). A pick on the scatter selects the owning Foliage entity
            // (id→guid), so it is selectable by clicking its content.
            // MIRROR: the player's `render.rs` runs the same projection (no pick).
            if let Some(fol) = w.get::<Foliage>(entity) {
                if visible && !fol.instances.is_empty() {
                    let translation = w
                        .get::<GlobalTransform>(entity)
                        .map(|g| g.translation())
                        .unwrap_or(DVec3::ZERO);
                    let id = next_id;
                    next_id += 1;
                    push_foliage_scatter(&mut self.scene, fol, translation, id);
                    // Every batch this entity emits shares the one id, so this is
                    // one row however many primitive kinds the palette uses.
                    self.id_to_guid.insert(id, guid);
                }
            }

            // 2D colliders cache a world-space outline (P8.3b); drawn as debug
            // lines for the selection only (in `render_frame`). Independent of
            // MeshRef — a collider often sits on a sprite or bare entity.
            if let Some(col) = w.get::<Collider2D>(entity) {
                let affine = w
                    .get::<GlobalTransform>(entity)
                    .map(|g| g.0)
                    .unwrap_or(glam::DAffine3::IDENTITY);
                let (_, rot, translation) = affine.to_scale_rotation_translation();
                let (_, _, z_rot) = rot.to_euler(glam::EulerRot::YXZ);
                self.collider_outlines
                    .insert(guid, project_collider(col, translation, z_rot));
            }

            // 3D colliders cache a world-space wireframe (P9.1); drawn as debug
            // lines for the selection only, with full body rotation + offset.
            if let Some(col) = w.get::<Collider3D>(entity) {
                let affine = w
                    .get::<GlobalTransform>(entity)
                    .map(|g| g.0)
                    .unwrap_or(glam::DAffine3::IDENTITY);
                let (_, rotation, translation) = affine.to_scale_rotation_translation();
                self.collider_outlines_3d
                    .insert(guid, project_collider_3d(col, translation, rotation));
            }

            // Volumes (E-P4) cache a tinted box wireframe drawn ALWAYS (not
            // selection-gated) so trigger/blocking regions stay visible while
            // editing. Reuses the entity's Collider3D projection; skipped when the
            // entity is hidden (respect the visibility flag).
            if visible {
                if let (Some(vol), Some(col)) =
                    (w.get::<Volume>(entity), w.get::<Collider3D>(entity))
                {
                    let affine = w
                        .get::<GlobalTransform>(entity)
                        .map(|g| g.0)
                        .unwrap_or(glam::DAffine3::IDENTITY);
                    let (_, rotation, translation) = affine.to_scale_rotation_translation();
                    self.volume_outlines.insert(
                        guid,
                        VolumeDebug {
                            collider: project_collider_3d(col, translation, rotation),
                            tint: vol.tint.to_array(),
                        },
                    );
                }
            }

            // Splines (E-P5) cache a world-space polyline sampled from the
            // control points, drawn ALWAYS (the curve is the only editor cue) so
            // long as the entity is visible. Points are entity-local, so they are
            // lifted through the entity's world transform first; Catmull-Rom /
            // linear are both affine combinations, so transforming the control
            // points then sampling is identical to sampling then transforming (and
            // cheaper). 16 samples per segment. The selected-only control markers
            // reuse the same world control points.
            if visible {
                if let Some(spline) = w.get::<Spline>(entity) {
                    let n = spline.points.len();
                    if n >= 2 {
                        let affine = w
                            .get::<GlobalTransform>(entity)
                            .map(|g| g.0)
                            .unwrap_or(glam::DAffine3::IDENTITY);
                        let control: Vec<DVec3> = spline
                            .points
                            .iter()
                            .map(|p| affine.transform_point3(p.to_dvec3()))
                            .collect();
                        let interp = match spline.interp {
                            EcsSplineInterp::Linear => SplineInterp::Linear,
                            EcsSplineInterp::CatmullRom => SplineInterp::CatmullRom,
                        };
                        let seg_count = if spline.closed { n } else { n - 1 };
                        let steps = seg_count * 16;
                        let mut line = Vec::with_capacity(steps + 1);
                        for i in 0..=steps {
                            let t = i as f64 / steps as f64;
                            line.push(inf_math::eval_spline(&control, spline.closed, interp, t));
                        }
                        self.spline_polylines
                            .insert(guid, SplineDebug { line, control });
                    }
                }
            }

            // Joints cache world-space debug segments (P12.1): the anchor→anchor
            // link + a cross at each anchor. Resolves the OTHER body's world pose
            // via the doc's guid index. Drawn for the selection only.
            let self_pose = || {
                let affine = w
                    .get::<GlobalTransform>(entity)
                    .map(|g| g.0)
                    .unwrap_or(glam::DAffine3::IDENTITY);
                let (_, rot, tr) = affine.to_scale_rotation_translation();
                (tr, rot)
            };
            let other_pose = |other: Uuid| -> Option<(DVec3, DQuat)> {
                let oe = world.entity_of(other)?;
                let affine = w.get::<GlobalTransform>(oe).map(|g| g.0)?;
                let (_, rot, tr) = affine.to_scale_rotation_translation();
                Some((tr, rot))
            };
            if let Some(j) = w.get::<Joint3D>(entity) {
                if let Some(other) = j.other.get() {
                    if let Some((op, orot)) = other_pose(other) {
                        let (sp, srot) = self_pose();
                        let a = sp + srot * j.local_anchor.to_dvec3();
                        let b = op + orot * j.other_anchor.to_dvec3();
                        self.joint_lines.insert(guid, project_joint(a, b));
                    }
                }
            } else if let Some(j) = w.get::<Joint2D>(entity) {
                if let Some(other) = j.other.get() {
                    if let Some((op, orot)) = other_pose(other) {
                        let (sp, srot) = self_pose();
                        let a = sp + srot * DVec3::new(j.local_anchor.x, j.local_anchor.y, 0.0);
                        let b = op + orot * DVec3::new(j.other_anchor.x, j.other_anchor.y, 0.0);
                        self.joint_lines.insert(guid, project_joint(a, b));
                    }
                }
            }

            // Skeletal meshes (P11.1 → **P18.3**): a `SkeletalMesh` entity now
            // draws its REAL skinned geometry. The bind-space mesh comes from the
            // referenced `.inf_mesh`'s skin streams, the palette from the
            // `.inf_skel` posed by the entity's `AnimPlayer` — rest pose when there
            // is no player, no clip, or an unresolvable one, so a freshly dropped
            // character is visible immediately rather than only once it plays.
            // Both the resolution and the pose rule live in Ring 1
            // (`inf_editor_core::render_assets`), which is the only part of this
            // that Linux CI can see.
            //
            // The **placeholder cube survives** as the honest fallback: a
            // `SkeletalMesh` with no assets bound (or with a mesh carrying no skin
            // stream) is still authorable content and must stay selectable.
            //
            // MIRROR of `inf_player::render`'s skeletal branch since P18.5, pinned
            // field for field by `inf-editor-core`'s `tests/projector_mirror.rs`.
            // Until that batch the player had no `SkeletalMesh` branch at all, so a
            // level with a character previewed here and shipped as nothing — this
            // comment used to describe that as "editor-only rendering, not a
            // divergence", which was true only for as long as nobody shipped one.
            // Host-local: `translation` (the editor has no fixed-step interpolation
            // to do) and `id`/`mesh` (each host numbers from its own counter over
            // its own iteration order).
            // ── P24.4 cloth ── the garment the sim folded on this entity this
            //    step, drawn beside whatever else it draws. NOT inside the
            //    `MeshRef`-absent branch below: a garment is worn by an entity,
            //    not instead of its geometry, so a character with a static mesh
            //    and a cloak draws both. Its own affine, because the branch below
            //    computes one only when the entity is skeletal. (MIRROR of the
            //    other host's call.)
            if visible {
                let affine = w
                    .get::<GlobalTransform>(entity)
                    .map(|g| g.0)
                    .unwrap_or(glam::DAffine3::IDENTITY);
                let (cloth_scale, cloth_rot, cloth_t) = affine.to_scale_rotation_translation();
                // ── P29.7 character space ── a garment's vertices are in the
                //    wearer's MODEL space (feet at the origin) and this
                //    translation is the entity's (a capsule CENTRE), so the lift
                //    goes through the one door that knows the difference.
                //    Without it a coat is drawn nearly a metre above the
                //    character wearing it. Zero for an entity with no movement
                //    component, which is every prop.
                let cloth_at = cloth_t + inf_ecs::pose::model_offset_world(world, entity);
                project_cloth(
                    &mut self.scene,
                    world,
                    guid,
                    cloth_at,
                    cloth_rot.as_quat(),
                    cloth_scale.as_vec3(),
                );
                project_hair(
                    &mut self.scene,
                    world,
                    guid,
                    cloth_at,
                    cloth_rot.as_quat(),
                    cloth_scale.as_vec3(),
                );
            }
            if w.get::<MeshRef>(entity).is_none() {
                if let (true, Some(sm)) = (visible, w.get::<SkeletalMesh>(entity).copied()) {
                    // ── P29.6 character space ── a rig's origin is its FEET and a
                    //    character's entity transform is its capsule CENTRE, so
                    //    the pose is drawn through the one door that knows the
                    //    difference (`inf_ecs::pose::model_to_world`,
                    //    identity-composed for everything that is not a
                    //    character). (MIRROR of the other host's call.)
                    let affine = inf_ecs::pose::model_to_world(world, entity);
                    let (scale, rot, translation) = affine.to_scale_rotation_translation();
                    let id = next_id;
                    next_id += 1;
                    live_render_assets.extend(sm.mesh);
                    live_render_assets.extend(sm.skeleton);
                    let player = w.get::<inf_ecs::components::AnimPlayer>(entity).copied();
                    live_render_assets.extend(player.and_then(|p| p.clip));
                    // P24.1: the pose the SIM evaluated for this entity this fixed
                    // step, if its `AnimStateMachine` published one. Read here rather
                    // than derived here — the machine's pose is deterministic sim
                    // state, folded into the trace, and a projector that re-evaluated
                    // it would be a second opinion about what the character is doing.
                    let posed = inf_ecs::pose::evaluated_pose(world, guid);
                    // PBR params come from the entity's `Material` exactly as they do
                    // on the rigid path; an unmaterialed character gets the renderer's
                    // neutral. Read BEFORE the match, so the placeholder branch below
                    // carries the same virtual-texture set the real geometry would —
                    // a character whose skeleton has not resolved is still a surface
                    // bound to a material (P26.4).
                    let (color, metallic, roughness, emissive, vt) = w
                        .get::<Material>(entity)
                        .map(|m| {
                            (
                                m.base_color.to_array(),
                                m.metallic,
                                m.roughness,
                                m.emissive_linear(),
                                inf_render::vt_set_for(vt, m.asset.map(|a| a.as_u128())),
                            )
                        })
                        .unwrap_or((
                            [0.8, 0.8, 0.8, 1.0],
                            0.0,
                            0.5,
                            [0.0; 3],
                            inf_render::VtTextureSet::NONE,
                        ));

                    // ── THE RENDERER READS THE TIER (wave NPC1b) ──
                    //
                    // `CrowdAgent` is the verdict `inf_ecs::crowd::step_crowd`
                    // published this fixed step. Three things follow from it, all
                    // of them render-side and none of them sim state — so nothing
                    // here can move a trace byte, and both hosts derive the same
                    // answers from the same `Guid`.
                    let agent = w.get::<inf_ecs::crowd::CrowdAgent>(entity).copied();
                    let look = agent.map(|a| inf_ecs::crowd::agent_look_in(world, a.guid));
                    let color = look.map_or(color, |l| l.over(color));
                    let body = look.map_or(1.0, |l| l.build);
                    let shadow = crowd_shadow(agent);
                    // An agent the ladder took OFF the pose path resolves to
                    // its rig's shared rest palette — the same matrices the
                    // per-agent call would have produced for it, derived once per
                    // `(mesh, skeleton)` instead of once per agent per frame.
                    let resolved = match agent {
                        Some(a) if !a.tier.poses() => {
                            self.render_assets.resolve_skinned_shared(&sm)
                        }
                        _ => self
                            .render_assets
                            .resolve_skinned(&sm, player.as_ref(), posed),
                    };
                    match resolved {
                        Some(draw) => {
                            // One `skinned_meshes` entry per (mesh, skeleton)
                            // pair, and the entry is the store's own `Arc` — no
                            // copy here, and the pass keys its GPU upload on that
                            // pointer, so re-projecting an unchanged character
                            // costs neither a memcpy nor a re-upload (P18.3).
                            let slot = *skinned_slots.entry(draw.key).or_insert_with(|| {
                                self.scene.skinned_meshes.push(draw.mesh);
                                self.scene.skinned_meshes.len() - 1
                            });
                            self.scene.skinned.push(inf_render::SkinnedInstance {
                                vt,
                                translation,
                                rotation: rot.as_quat(),
                                scale: scale.as_vec3() * body,
                                color,
                                metallic,
                                roughness,
                                emissive,
                                id,
                                mesh: slot,
                                palette: draw.palette,
                                shadow,
                            });
                        }
                        // Unbound (or unskinned) — the pre-P18.3 placeholder,
                        // unchanged down to its slate tint, so authoring a skeletal
                        // entity before its assets exist looks exactly as it did.
                        None => self.scene.instances.push(MeshInstance {
                            vt,
                            translation,
                            rotation: rot.as_quat(),
                            scale: scale.as_vec3(),
                            color: [0.55, 0.60, 0.72, 1.0],
                            metallic: 0.0,
                            roughness: 0.6,
                            emissive: [0.0; 3],
                            id,
                            // Skeletal placeholder is always a cube (no primitive kind).
                            mesh: PrimMesh::Cube,
                            // R-P5: skeletal placeholders are opaque.
                            blend: 0,
                            cutoff: 0.5,
                        }),
                    }
                    self.id_to_guid.insert(id, guid);
                    self.guid_to_id.insert(guid, id);
                }
                continue; // only meshes become draw instances
            }
            if !visible {
                continue;
            }
            let affine = w
                .get::<GlobalTransform>(entity)
                .map(|g| g.0)
                .unwrap_or(glam::DAffine3::IDENTITY);
            let (scale, rot, translation) = affine.to_scale_rotation_translation();
            // MIRROR: this Material→MeshInstance projection is duplicated in the
            // player's `render.rs` (inf-player) — keep the two in sync, R-P5 blend
            // + cutoff included.
            let (color, metallic, roughness, emissive, blend, cutoff, vt) = w
                .get::<Material>(entity)
                .map(|m| {
                    (
                        m.base_color.to_array(),
                        m.metallic,
                        m.roughness,
                        m.emissive_linear(),
                        blend_code(m.blend),
                        m.alpha_cutoff,
                        inf_render::vt_set_for(vt, m.asset.map(|a| a.as_u128())),
                    )
                })
                .unwrap_or((
                    [0.8, 0.8, 0.8, 1.0],
                    0.0,
                    0.5,
                    [0.0; 3],
                    0,
                    0.5,
                    inf_render::VtTextureSet::NONE,
                ));
            let mesh_ref = w.get::<MeshRef>(entity).copied().unwrap_or_default();
            let id = next_id;
            next_id += 1;
            live_render_assets.extend(mesh_ref.asset);
            // P18.3 — THE OLDEST DOCUMENTED GAP, CLOSED. A `MeshRef.asset` with a
            // derived vmesh renders REAL geometry: the GPU meshlet path (vgeom on)
            // or the classic discrete-LOD fallback (vgeom off), both driven by the
            // same vgeom scene content, with the tier deciding which node draws it.
            // An unresolved asset (or a primitive-only `MeshRef`) falls back to the
            // built-in primitive — which stays *legitimate content*, not a
            // placeholder, for the Cube/Sphere/Plane/Cylinder/Cone kinds.
            //
            // MIRROR of `inf_player::render::project_scene`'s `MeshRef` branch,
            // field for field. The one deliberate difference is where the asset id
            // comes from — the player uses the derived GUID (a pack is immutable),
            // the editor uses the derived payload's content hash (a content root is
            // not) — and the reasoning lives in `inf_editor_core::render_assets`,
            // once, rather than in two comments that could disagree.
            // P22.3 — THE ATOMIC SWAP. A destructible that has started breaking
            // draws its CHUNKS and not its mesh: never both, never neither. The
            // predicate is `FractureState::is_intact`, the same one
            // `PhysicsBridge3D::sync_from_world_sim` reads to decide whether the
            // actor keeps its collider, so the two halves of the swap cannot
            // disagree. An intact (or absent) state projects nothing and the mesh
            // branch below runs exactly as it always has.
            // P22.4 — and its RUBBLE. `project_debris` rides the same `and_then`
            // so the two projections cannot disagree about which actors are
            // broken; it is dressing only, keyed by content, and it never touches
            // the sim.
            let broken = {
                let debris = &mut self.debris_cache;
                self.fractures.lock().ok().and_then(|f| {
                    f.get(&guid).and_then(|state| {
                        project_fracture(
                            state,
                            w.get::<Material>(entity),
                            inf_render::terrain_id_from_guid(guid.as_u128()),
                        )
                        .map(|chunks| {
                            (
                                chunks,
                                project_debris(
                                    state,
                                    w.get::<Material>(entity),
                                    inf_render::terrain_id_from_guid(guid.as_u128()),
                                    debris,
                                ),
                            )
                        })
                    })
                })
            };
            let fractured = broken.is_some();
            if let Some((chunks, rubble)) = broken {
                self.scene.fracture_chunks.extend(chunks);
                self.scene.scatter.extend(rubble);
            }
            let vgeom = (!fractured)
                .then(|| {
                    mesh_ref
                        .asset
                        .and_then(|mesh_id| self.render_assets.resolve_vgeom(mesh_id))
                })
                .flatten();
            match vgeom {
                Some(loaded) => {
                    if vgeom_seen.insert(loaded.id) {
                        // The scene carries the PAGED source, not a decoded DAG
                        // (P18.2): the render node's streamer decides what of it is
                        // resident from the camera's own screen-error wants.
                        self.scene
                            .vgeom_assets
                            .push(inf_render::VgeomAsset::new(loaded.id, loaded.source));
                    }
                    self.scene.vgeom_instances.push(inf_render::VgeomInstance {
                        vt,
                        asset: loaded.id,
                        translation,
                        rotation: rot.as_quat(),
                        scale: scale.as_vec3(),
                        color,
                        metallic,
                        roughness,
                        emissive,
                        id,
                    });
                }
                // A broken destructible has already pushed its chunks.
                None if fractured => {}
                // **Wave FIX2: a bound mesh with no DAG draws NOTHING.**
                //
                // It used to draw `prim_mesh(mesh_ref.primitive)` — a 1 m cube for
                // `MeshRef::default()`. `resolve_vgeom` misses when the derived
                // `.inf_vmesh` is absent OR (since FIX2) STALE, and a viewport
                // that answers "stale" with a box is a viewport that photographs
                // something that is not in the project. `open_vgeom` states which
                // and names the remedy.
                //
                // Both hosts: `inf_player::render::project_scene` has the
                // identical arm.
                None if mesh_ref.asset.is_some() => {}
                // R-P1: an unresolved / primitive-only MeshRef draws its built-in
                // primitive kind (Sphere/Plane/Cylinder/Cone), not always a cube.
                None => self.scene.instances.push(MeshInstance {
                    vt,
                    translation,
                    rotation: rot.as_quat(),
                    scale: scale.as_vec3(),
                    color,
                    metallic,
                    roughness,
                    emissive,
                    id,
                    mesh: prim_mesh(mesh_ref.primitive),
                    blend,
                    cutoff,
                }),
            }
            self.id_to_guid.insert(id, guid);
            self.guid_to_id.insert(guid, id);
        }

        // Selection outline mirrors the document's selection.
        self.scene.selected = doc
            .selection()
            .iter()
            .filter_map(|g| self.guid_to_id.get(g).copied())
            .collect();
        // Full selection (any entity) drives the collider debug outlines.
        self.selected_guids = doc.selection().to_vec();

        // Capture working transforms for selected NON-mesh entities (sprites,
        // text, tilemaps, …) so the gizmo can move them in 2D. Mesh entities are
        // covered by their render instances instead.
        self.selected_2d.clear();
        for &guid in doc.selection() {
            if self.guid_to_id.contains_key(&guid) {
                continue;
            }
            let Some(entity) = world.entity_of(guid) else {
                continue;
            };
            let affine = w
                .get::<GlobalTransform>(entity)
                .map(|g| g.0)
                .unwrap_or(glam::DAffine3::IDENTITY);
            let (scale, rotation, translation) = affine.to_scale_rotation_translation();
            let extent = w
                .get::<Sprite>(entity)
                .map(|s| (s.size.x.max(s.size.y) * 0.5).max(0.25))
                .unwrap_or(0.5);
            self.selected_2d.insert(
                guid,
                Sel2D {
                    translation,
                    rotation,
                    scale,
                    extent,
                },
            );
        }

        // Release every terrain stream the projection did not just use (P16.4b
        // audit; P16.6: **all** live terrains, not just the first). A stream keyed
        // on an entity the projection did not touch — a terrain that became
        // invisible, was deleted, or belonged to a document that has since been
        // replaced — is dead memory holding a whole `.inf_terrain` payload, plus
        // any tile it pinned for an unsaved edit (which nothing would ever unpin).
        // This is the only place that knows which terrains are live, so it is the
        // only place that can do it.
        self.terrain_streams
            .retain_only(self.terrain_slots.iter().map(|s| s.guid));

        // The same audit for mesh assets (P18.3). A `.inf_vmesh` mapping plus its
        // decoded skinned geometry is real memory held on behalf of entities that
        // may no longer exist: a mesh unbound in the Details panel, an entity
        // deleted, or — the case P16.4b was written about — a whole document
        // replaced by File ▸ Open. The projection is the only place that knows the
        // live set, so this is the only place that can release the rest.
        // …and the scatter-kind meshes are live render assets too (wave TER2b):
        // they are named by an instance list rather than by a component field, so
        // the walk above never sees them and an audit without them would free the
        // island's ground cover on the very projection that drew it.
        //
        // Keyed on what the instances **named**, not on what resolved (TER2b
        // audit): `resolve_scatter_geometry` caches its misses as well as its
        // hits, and reading the resolved table here threw every miss away again
        // one line later — so an unresolvable kind re-read and re-decoded its
        // `.inf_mesh` on every document version.
        live_render_assets.extend(self.scatter_wanted.iter().copied());
        self.render_assets.retain_only(live_render_assets);

        // Re-validate the tool target: keep it if that terrain is still projected
        // (so a stroke's status stays about the terrain being sculpted), else fall
        // back to the first projected terrain — which is exactly the pre-P16.6
        // behaviour for a single-terrain document.
        if !self
            .terrain_guid
            .is_some_and(|g| self.terrain_slots.iter().any(|s| s.guid == g))
        {
            self.terrain_guid = self.terrain_slots.first().map(|s| s.guid);
        }

        // **Round-2 finding R2-2**: the debris memo is retained against the
        // live fracture set. `DebrisCache::batch` drops an entry when the actor
        // sheds nothing, which covers a reclaim; a DESPAWN never calls it again
        // and left the packed payload for the session. MIRROR:
        // `inf_player::render::project_scene` does the same, from the same
        // authority.
        {
            let live: std::collections::BTreeSet<u64> = self
                .fractures
                .lock()
                .map(|f| {
                    f.keys()
                        .map(|g| inf_render::terrain_id_from_guid(g.as_u128()))
                        .collect()
                })
                .unwrap_or_default();
            self.debris_cache.retain_live(&live);
        }

        // P21.2 SEAM. Fill every projected volume's per-vertex seam terms from
        // the heightfields projected beside them, and arm the blend at
        // [`DEFAULT_SEAM_BAND_M`]. `inf_render::apply_seam` is the ONE
        // implementation both hosts call — a hand-synced per-vertex loop on each
        // side is exactly the shape that eventually stops agreeing, and a cave
        // mouth that shades one way in the editor and another in the shipped
        // build is the failure the mirrored-pair discipline exists to prevent.
        //
        // Runs last, because it needs BOTH halves projected, and once per
        // projection (a change stamp) rather than per frame.
        //
        // Hardening Wave E: and now not even that. The seam is a pure function of
        // the volumes' vertices and the terrains, so when EVERY volume and EVERY
        // terrain in this scene was carried forward unchanged — nothing dropped,
        // nothing added, nothing rebuilt — last frame's terms are still the right
        // ones, and the per-vertex walk (which samples every terrain per vertex)
        // is skipped whole. `carried == pushed && prev.is_empty()` is the exact
        // statement of "nothing changed on this axis": the first half sees an
        // addition or a rebuild, the second sees a removal.
        //
        // Round-2 finding B9: `seam_dirty` is the third term, and it is what
        // makes the other two honest in THIS host. The two out-of-band writers
        // re-project a terrain in place, so the comparison above runs against
        // an already-updated list and answers "nothing changed" about a change
        // it made itself. See the field's doc.
        if self.seam_dirty
            || !(prev_terrains.is_empty()
                && prev_voxels.is_empty()
                && terrains_before - prev_terrains.len() == self.scene.terrains.len()
                && voxels_before - prev_voxels.len() == self.scene.voxels.len())
        {
            inf_render::apply_seam(
                &mut self.scene.voxels,
                &self.scene.terrains,
                inf_render::DEFAULT_SEAM_BAND_M,
            );
            self.seam_dirty = false;
        }

        self.scene.hovered = None;
        self.scene.mark_dirty();
    }
}

/// Project a [`Collider2D`] (+ its world pose) into a world-space debug outline.
fn project_collider(col: &Collider2D, world_pos: DVec3, z_rot: f64) -> ColliderDebug {
    let shape = match col.shape_kind {
        ColliderShape2DKind::Box => ColliderOutline2D::Box {
            half: Vec2::new(col.half_extents.x as f32, col.half_extents.y as f32),
        },
        ColliderShape2DKind::Circle => ColliderOutline2D::Circle {
            radius: col.radius as f32,
        },
        ColliderShape2DKind::Capsule => ColliderOutline2D::Capsule {
            half_height: col.half_extents.y as f32,
            radius: col.radius as f32,
        },
    };
    ColliderDebug {
        shape,
        offset: Vec2::new(col.offset.x as f32, col.offset.y as f32),
        world_pos,
        z_rot,
    }
}

/// Stroke a collider outline into the debug-line layer, rebasing through the
/// floating origin. Points are generated in the collider's local XY frame,
/// rotated by the body's Z rotation, offset, and lifted onto the entity's world
/// position (Z preserved so the outline sits in the sprite plane).
fn draw_collider_outline(debug: &mut DebugDraw, origin: &FloatingOrigin, cd: &ColliderDebug) {
    const COLLIDER_COLOR: [f32; 4] = [0.30, 0.95, 0.55, 1.0];
    /// Circle/capsule tessellation for the debug outline.
    const CIRCLE_SEGMENTS: u32 = 32;

    let (sin, cos) = (cd.z_rot.sin() as f32, cd.z_rot.cos() as f32);
    let rotate = |p: Vec2| Vec2::new(cos * p.x - sin * p.y, sin * p.x + cos * p.y);
    let offset = rotate(cd.offset);

    let pts = collider_outline_2d(cd.shape, CIRCLE_SEGMENTS);
    if pts.is_empty() {
        return;
    }
    // World-space (then render-local) point for a local outline vertex.
    let to_local = |p: Vec2| {
        let r = rotate(p) + offset;
        let world = cd.world_pos + DVec3::new(r.x as f64, r.y as f64, 0.0);
        origin.to_render(world)
    };
    for i in 0..pts.len() {
        let a = to_local(pts[i]);
        let b = to_local(pts[(i + 1) % pts.len()]);
        debug.line(a, b, COLLIDER_COLOR);
    }
}

/// Project a [`Collider3D`] (+ its world pose) into a world-space debug wireframe.
fn project_collider_3d(col: &Collider3D, world_pos: DVec3, rotation: DQuat) -> ColliderDebug3D {
    let shape = match col.shape_kind {
        ColliderShape3DKind::Box => ColliderOutline3D::Box {
            half: Vec3::new(
                col.half_extents.x as f32,
                col.half_extents.y as f32,
                col.half_extents.z as f32,
            ),
        },
        ColliderShape3DKind::Sphere => ColliderOutline3D::Sphere {
            radius: col.radius as f32,
        },
        ColliderShape3DKind::Capsule => ColliderOutline3D::Capsule {
            half_height: col.half_extents.y as f32,
            radius: col.radius as f32,
        },
    };
    ColliderDebug3D {
        shape,
        offset: col.offset.to_dvec3(),
        world_pos,
        rotation,
    }
}

/// Build joint debug segments (world space): the anchor-to-anchor link plus a
/// small axis cross at each anchor so the joint reads even when the anchors
/// coincide with the body origins.
fn project_joint(anchor_a: DVec3, anchor_b: DVec3) -> JointDebug {
    const CROSS: f64 = 0.12;
    let mut segments = vec![[anchor_a, anchor_b]];
    for anchor in [anchor_a, anchor_b] {
        for axis in [DVec3::X, DVec3::Y, DVec3::Z] {
            segments.push([anchor - axis * CROSS, anchor + axis * CROSS]);
        }
    }
    JointDebug { segments }
}

/// Stroke joint debug segments into the debug-line layer, rebasing each endpoint
/// through the floating origin.
fn draw_joint_lines(debug: &mut DebugDraw, origin: &FloatingOrigin, jd: &JointDebug) {
    const JOINT_COLOR: [f32; 4] = [0.95, 0.75, 0.20, 1.0];
    for [a, b] in &jd.segments {
        debug.line(origin.to_render(*a), origin.to_render(*b), JOINT_COLOR);
    }
}

/// Stroke a spot-light cone gizmo into the debug-line layer (R-P3): an 8-segment
/// rim circle at the beam's outer-cone radius, plus four apex→rim spokes, in the
/// light's colour. The rim sits at distance `dist` down the emission `axis`, with
/// radius `dist · tan(outer_rad)`. Rebased through the floating origin.
fn draw_spot_cone(debug: &mut DebugDraw, origin: &FloatingOrigin, sd: &SpotDebug) {
    const SEGMENTS: usize = 8;
    let axis = sd.axis;
    // Two axis-perpendicular basis vectors for the rim plane.
    let seed = if axis.x.abs() < 0.9 {
        DVec3::X
    } else {
        DVec3::Y
    };
    let t1 = axis.cross(seed).normalize();
    let t2 = axis.cross(t1); // already unit (axis ⟂ t1, both unit)
    let center = sd.apex + axis * sd.dist;
    let radius = sd.dist * sd.outer_rad.tan();

    let rim = |i: usize| -> DVec3 {
        let a = std::f64::consts::TAU * i as f64 / SEGMENTS as f64;
        center + (t1 * a.cos() + t2 * a.sin()) * radius
    };
    let apex_local = origin.to_render(sd.apex);
    for i in 0..SEGMENTS {
        let a = origin.to_render(rim(i));
        let b = origin.to_render(rim((i + 1) % SEGMENTS));
        debug.line(a, b, sd.color); // rim
        if i % (SEGMENTS / 4) == 0 {
            debug.line(apex_local, a, sd.color); // apex → rim spoke (×4)
        }
    }
}

/// Stroke a 3D collider wireframe into the debug-line layer, rebasing through the
/// floating origin. Segments are generated in the collider's local frame, offset
/// in the body frame, rotated by the body's world orientation, and lifted onto
/// the entity's world position.
fn draw_collider_outline_3d(debug: &mut DebugDraw, origin: &FloatingOrigin, cd: &ColliderDebug3D) {
    const COLLIDER_COLOR: [f32; 4] = [0.30, 0.95, 0.55, 1.0];
    /// Ring/arc tessellation for the debug wireframe.
    const CIRCLE_SEGMENTS: u32 = 32;

    // Local frame point → render-local: offset in body frame, rotate, translate.
    let to_local = |p: Vec3| {
        let local = DVec3::new(p.x as f64, p.y as f64, p.z as f64) + cd.offset;
        let world = cd.world_pos + cd.rotation * local;
        origin.to_render(world)
    };
    for [a, b] in collider_outline_3d(cd.shape, CIRCLE_SEGMENTS) {
        debug.line(to_local(a), to_local(b), COLLIDER_COLOR);
    }
}

/// Stroke a [`Volume`]'s box wireframe into the debug-line layer in its tint,
/// rebasing through the floating origin. Drawn unconditionally (the region is
/// invisible in PIE, so the editor outline is the only cue). The debug-line API
/// has no width, so a selected volume gets a second inset ring in a brightened
/// tint to read as "thicker/highlighted".
fn draw_volume_outline(
    debug: &mut DebugDraw,
    origin: &FloatingOrigin,
    vd: &VolumeDebug,
    selected: bool,
) {
    const CIRCLE_SEGMENTS: u32 = 32;
    let cd = &vd.collider;
    // Local (optionally scaled) collider point → render-local: offset in the body
    // frame, rotate by the body orientation, translate onto the entity.
    let stroke = |debug: &mut DebugDraw, scale: f64, color: [f32; 4]| {
        let to_local = |p: Vec3| {
            let local = DVec3::new(p.x as f64, p.y as f64, p.z as f64) * scale + cd.offset;
            origin.to_render(cd.world_pos + cd.rotation * local)
        };
        for [a, b] in collider_outline_3d(cd.shape, CIRCLE_SEGMENTS) {
            debug.line(to_local(a), to_local(b), color);
        }
    };
    stroke(debug, 1.0, vd.tint);
    if selected {
        let brighten = |c: f32| (c * 1.5).min(1.0);
        let bright = [
            brighten(vd.tint[0]),
            brighten(vd.tint[1]),
            brighten(vd.tint[2]),
            vd.tint[3],
        ];
        stroke(debug, 0.9, bright);
    }
}

/// A distinct placeholder colour per PCG kind index, so a multi-kind scatter
/// reads as varied content even before real meshes upload (P10.5b). Cycles
/// through a small foliage/rock palette. Shared by the volume path and P19.3's
/// terrain biome population — both index the same palette space.
fn pcg_kind_color(kind: u32) -> [f32; 4] {
    const PALETTE: [[f32; 4]; 5] = [
        [0.28, 0.52, 0.24, 1.0], // foliage green
        [0.55, 0.40, 0.22, 1.0], // trunk brown
        [0.62, 0.60, 0.55, 1.0], // rock grey
        [0.75, 0.68, 0.35, 1.0], // dry grass
        [0.35, 0.58, 0.45, 1.0], // shrub teal
    ];
    PALETTE[(kind as usize) % PALETTE.len()]
}

/// **Register the building module meshes** (island wave I8b) — the twelve shape
/// families every palette module draws.
///
/// They name no `.inf_mesh` file and never will: the geometry is a function of
/// the module's own name, minted under a private salt by
/// `inf_pcg::building::modules`. So neither loader above can find them by
/// scanning a pack or a directory, and both hosts add them to the table they
/// just built instead — from one Ring-0 source, through one Ring-0 flattener.
///
/// **Existing entries win.** A project that really does ship an `.inf_mesh`
/// under one of these ids has authored it deliberately, and an engine default
/// must not overwrite authored content.
///
/// MIRROR: identical in `inf_player::scatter_mesh`, pinned by
/// `inf-editor-core`'s `tests/projector_mirror.rs`.
pub fn add_building_modules(table: &mut inf_render::ScatterMeshes) {
    // MIRROR-BEGIN building_module_table
    for (id, m) in inf_pcg::building::modules::module_meshes() {
        let key = id.as_u128();
        if table.contains_key(&key) {
            continue;
        }
        let g = inf_render::ScatterGeometry::from_streams(&m.positions, &m.normals, &m.indices);
        if g.is_empty() || g.triangle_count() > inf_render::MAX_SCATTER_MESH_TRIANGLES {
            continue;
        }
        table.insert(key, std::sync::Arc::new(g));
    }
    // MIRROR-END building_module_table
}

/// ONE `ScatterBatch` from a list of [`ScatteredInstance`]s anchored at
/// `translation` (P18.5) — the whole body of every scatter path that speaks in
/// scattered instances: a [`PcgVolume`]'s evaluated cache and P19.3's terrain
/// **biome population**. Written once so the two cannot drift, which is the same
/// argument that pins the two hosts against each other.
///
/// # The cover draws its own meshes (wave TER2b)
///
/// From P18.5 until this wave every scattered instance drew a `PrimMesh::Cube`
/// tinted from a five-entry debug palette, whatever mesh its `PcgKind` named —
/// the island shipped 16 771 tinted cubes with three authored props sitting
/// unread in its pack. `ScatteredInstance::mesh` now carries the GUID
/// (`kind_index` could not: it is rule-local, and populations are concatenated),
/// `meshes` is the host's resolved table of geometry, and instances **bucket by
/// mesh** — one `ScatterBatch` per authored mesh plus one for everything that has
/// none. The bucket map is a `BTreeMap` and not a `HashMap` because the batch
/// order a projection emits is what the content-keyed GPU caches see, and an
/// order that depends on a hash seed is a re-upload nobody asked for.
///
/// **The cube does not leave; it becomes the proxy.** `PrimMesh::Cube` is still
/// passed, and it is what the impostor card, the CPU fallback and the shadow
/// caster pack use — those three bind one shared vertex buffer for the whole
/// frame, so a per-batch mesh does not fit in them. What changed is the full-mesh
/// raster, which pulls its vertices out of a storage buffer and can therefore be
/// handed any geometry at all. The impostor is sized off the authored radius
/// rather than the cube's (`impostor_radius` in `scatter_mesh.wgsl`).
///
/// The instance tint is still the placeholder palette: the scatter pull buffer is
/// position + normal and carries no uv, so there is nowhere for a material to be
/// sampled. Texturing a scattered mesh is the named follow-up.
///
/// **`draw_distance` rides on the batch now.** The editor used to cull it against
/// its own camera eye on the CPU and the player ignored the field entirely, so a
/// shipped build drew strictly more scatter than its preview. The cull compute
/// honours it for both hosts, which is what finally makes them agree. `0` means
/// unlimited — the renderer's own bands then have sole charge.
///
/// The whole batch takes ONE pick `id`: a scatter is authored, moved and deleted
/// as a whole, so it is one object as far as selection is concerned.
///
/// MIRROR: identical in `inf_viewport::host` and `inf_player::render`, pinned by
/// `inf-editor-core`'s `tests/projector_mirror.rs`.
#[allow(clippy::too_many_arguments)]
fn push_scatter(
    scene: &mut RenderScene,
    instances: &[ScatteredInstance],
    meshes: &inf_render::ScatterMeshes,
    translation: DVec3,
    draw_distance: f64,
    near_distance: f64,
    id: u32,
    clock: inf_render::ScatterClock,
    casts_shadows: bool,
) {
    // MIRROR-BEGIN scatter_mesh_buckets
    if instances.is_empty() {
        return;
    }
    // **The bucket key grew a second half** (island wave I8b): the mesh a
    // batch draws AND the emission it draws with. `ScatterBatch::emissive` is
    // one value for a whole batch, so two instances that glow differently
    // cannot share one -- and a window pane and the wall beside it are exactly
    // that pair. Bucketing on the authored glow costs one extra batch per
    // volume that has any, and nothing at all for one that has none.
    //
    // **…and a third** (wave VEN1a): the SURFACE it draws with. `metallic`,
    // `roughness` and the authored `emissive` are all per-batch fields on
    // `ScatterBatch`, so the same argument reaches all four numbers at once and
    // `ScatteredSurface::batch_key` states them in one place rather than
    // letting a fifth reader forget one. A volume grows one extra batch per
    // distinct surface it holds and NONE at all for a volume whose modules all
    // answer `ScatteredSurface::DEFAULT` -- which is every level that predates
    // the venue archetypes, so no committed content changes its batch count.
    //
    // The TINT is deliberately not in the key: a scattered instance has carried
    // its own colour since P18.5, so a venue's six neon hues cost one draw.
    type BucketKey = (Option<u128>, u32, [u32; 6]);
    let mut buckets: std::collections::BTreeMap<BucketKey, Vec<ScatterInstance>> =
        std::collections::BTreeMap::new();
    for si in instances {
        // A GUID the host could not resolve buckets with the meshless ones, so an
        // absent mesh degrades to the placeholder it always was rather than to an
        // empty batch.
        let key = si
            .mesh
            .map(|m| m.as_u128())
            .filter(|k| meshes.contains_key(k));
        // **The box the instance occupies, not a unit cube** (I8b). `scale` is
        // one uniform f64 and every building module carries 1.0, so before this
        // a 10 m slab drew the same size as a 0.3 m mullion. Both the authored
        // geometry and the fallback primitive are unit-box shaped -- they span
        // [-0.5, 0.5] -- so the scale is twice the half-extent for either.
        let scale = si.extent.map_or_else(
            || glam::Vec3::splat(si.scale as f32),
            |e| glam::Vec3::new(e[0] * 2.0, e[1] * 2.0, e[2] * 2.0),
        );
        buckets
            .entry((key, si.glow.to_bits(), si.surface.batch_key()))
            .or_default()
            .push(ScatterInstance {
                position: si.position,
                rotation: si.rotation.as_quat(),
                scale,
                // **The authored tint wins over the placeholder palette** (wave
                // VEN1a). `pcg_kind_color` is a five-entry debug ramp indexed by
                // a RULE-LOCAL kind, so a chrome pole and a brick wall could
                // draw the same green; a module whose family states a colour
                // states it here, and the palette survives for everything that
                // does not.
                color: si.surface.tint.unwrap_or_else(|| pcg_kind_color(si.kind)),
            });
    }
    for ((key, glow_bits, surface), bucket) in buckets {
        // **THE MID BAND** (wave CERT1, CP-C3). A building used to draw in two
        // tiers: everything it holds out to `STRUCTURE_LOD_M`, then one shell
        // box. The third rung is here, and it costs no new data: the bucket key
        // is already the mesh GUID, and a FIT-OUT family's GUID is a fact about
        // the content, so a chair stops drawing at `INTERIOR_LOD_M` while the
        // wall behind it keeps the band it had. A bucket that is not a module at
        // all -- ground cover, an authored `.inf_mesh` -- answers `false` and is
        // untouched, so no level without grammar buildings changes one batch.
        //
        // `draw_distance == 0.0` is "no limit", so it is a MATCH and not a
        // `min`: `0.0.min(64.0)` is zero, which would cull the fit-out of every
        // volume that never set a distance.
        let bucket_draw = match key.map(uuid::Uuid::from_u128) {
            Some(g) if inf_pcg::building::modules::is_fit_out_mesh(g) => {
                match draw_distance > 0.0 {
                    true => draw_distance.min(inf_render::INTERIOR_LOD_M),
                    false => inf_render::INTERIOR_LOD_M,
                }
            }
            _ => draw_distance,
        };
        let data = ScatterData::build_with_geometry(
            PrimMesh::Cube,
            key.and_then(|k| meshes.get(&k)).cloned(),
            translation,
            bucket,
        );
        // The batch's emission is the night-window ramp PLUS the module's own
        // authored colour, pulsed. Added and not chosen between, because a lit
        // shopfront pane is both: a window that glows warm at dusk and a sign
        // that is magenta at every hour.
        let authored = inf_render::pulse_emissive(
            [
                f32::from_bits(surface[0]),
                f32::from_bits(surface[1]),
                f32::from_bits(surface[2]),
            ],
            f32::from_bits(surface[3]),
            clock.pulse_tick,
        );
        let glow = inf_render::glow_emissive(f32::from_bits(glow_bits), clock.glow_step);
        scene.scatter.push(ScatterBatch {
            data: Arc::new(data),
            anchor: translation,
            metallic: f32::from_bits(surface[4]),
            roughness: f32::from_bits(surface[5]),
            emissive: [
                glow[0] + authored[0],
                glow[1] + authored[1],
                glow[2] + authored[2],
            ],
            id,
            draw_distance: bucket_draw,
            near_distance,
            casts_shadows,
        });
    }
    // MIRROR-END scatter_mesh_buckets
}

/// The shell batch: one oriented **box** per structure group, banded
/// `[near, far)`.
///
/// Separate from [`push_scatter`] because a shell is the one scatter instance
/// that is not uniformly scaled — its three half-extents are the building's, and
/// a cube of the wrong proportions is a different building rather than a coarser
/// one.
///
/// MIRROR: identical in `inf_viewport::host` and `inf_player::render`.
#[allow(clippy::too_many_arguments)]
fn push_shells(
    scene: &mut RenderScene,
    shells: &[ScatteredInstance],
    vol: &PcgVolume,
    translation: DVec3,
    draw_distance: f64,
    near_distance: f64,
    id: u32,
) {
    // MIRROR-BEGIN pcg_shell_batch
    if shells.is_empty() {
        return;
    }
    let data = ScatterData::build(
        PrimMesh::Cube,
        translation,
        shells.iter().zip(&vol.structure_groups).map(|(si, g)| {
            let h = g.shell.half_extents;
            ScatterInstance {
                position: si.position,
                rotation: si.rotation.as_quat(),
                scale: glam::Vec3::new((h.x * 2.0) as f32, (h.y * 2.0) as f32, (h.z * 2.0) as f32),
                color: pcg_kind_color(si.kind),
            }
        }),
    );
    scene.scatter.push(ScatterBatch {
        data: Arc::new(data),
        anchor: translation,
        metallic: 0.0,
        roughness: 0.75,
        emissive: [0.0; 3],
        id,
        draw_distance,
        near_distance,
        // **The shell is the building's shadow** (island wave I8b). Its parts
        // are packed with `casts_shadows: false` precisely because this box
        // contains them, so this is the one batch of a building that must
        // always be `true`.
        casts_shadows: true,
    });
    // MIRROR-END pcg_shell_batch
}

/// **Carry a volume's scatter batches forward, or pack them** (island wave I8a
/// audit) — [`push_pcg_scatter`] behind the projection's own change stamp.
///
/// # The defect this closes
///
/// `push_pcg_scatter` re-packs a volume's whole population every frame: every
/// instance to f32 render space and every packed byte through xxh3 to derive the
/// content key. The key then tells the GPU cache nothing changed. That is
/// precisely the shape Hardening Wave E found in the terrain and voxel payloads
/// — *"what no consumer could do is stop the payload from being built"* — and on
/// the island's 172 settlement blocks it was **365 545 instances and 20.2 ms a
/// projection against a 1.5 ms `PROJECTION_BUDGET_MS`.**
///
/// # Why the key is sound
///
/// [`ScatterSource`](inf_render::ScatterSource) carries the volume's `Guid`, its
/// **process-global** `structures_gen`, the authored `draw_distance` the stamp
/// does not cover, the mesh table's own stamp and the world anchor the offsets
/// were packed against. The population stamp is drawn from a process-global
/// monotone counter (`inf_ecs`'s `NEXT_STRUCTURES_GEN`), so it names one
/// population of one volume for the life of the process — including across the
/// destroy-and-rebuild a cell deactivation and reactivation performs under the
/// same guid, which is what a per-volume counter could not do. `0` is "never
/// written" and is a forced miss.
///
/// A hit copies an `Arc` and a handful of scalars per batch and not one instance;
/// a miss packs exactly what it packed before. **The two produce byte-identical
/// scenes**, which is what `a_reprojection_is_byte_identical_to_a_cold_one`
/// already exists to say.
///
/// The pick `id` is rewritten on a carried batch, deliberately: it is the
/// projection's own numbering of this frame's entities and not content.
///
/// MIRROR: identical in `inf_viewport::host` and `inf_player::render`, pinned by
/// `inf-editor-core`'s `tests/projector_mirror.rs`.
#[allow(clippy::too_many_arguments)]
fn carry_or_push_pcg_scatter(
    scene: &mut RenderScene,
    prev: &mut inf_render::ScatterMemo,
    table: u64,
    guid: Uuid,
    vol: &PcgVolume,
    meshes: &inf_render::ScatterMeshes,
    translation: DVec3,
    id: u32,
    clock: inf_render::ScatterClock,
) {
    // MIRROR-BEGIN pcg_scatter_memo
    let source = inf_render::ScatterSource {
        entity: guid.as_u128(),
        stamp: vol.structures_gen,
        draw_distance_bits: vol.draw_distance.to_bits(),
        table,
        glow_step: clock.glow_step,
        // **Zero unless this volume actually pulses** (wave VEN1a). The tick
        // must be in the key or a carried batch keeps the phase it was packed
        // at and a club's string lights freeze mid-breath -- but a key that
        // always carried it would re-pack all 172 of the island's settlement
        // volumes eight times a second for a festoon in one of them.
        //
        // **`PcgVolume::pulses` and NOT a scan of `evaluated`** (VEN1a audit).
        // This expression builds the memo's KEY, so it runs before the lookup
        // -- on a hit as much as a miss. Spelled as `evaluated.iter().any(..)`
        // it made a carried projection walk every instance it was carrying
        // precisely so as not to touch them: measured on `projection_budget`'s
        // 20 020-instance fixture, the hit path went 0.009 ms -> 0.350 ms.
        pulse_tick: if vol.pulses { clock.pulse_tick } else { 0 },
        anchor: translation,
    };
    if let Some(batches) = prev.take(source) {
        for b in &batches {
            scene
                .scatter
                .push(inf_render::ScatterBatch { id, ..b.clone() });
        }
        scene.scatter_memo.insert(source, batches);
        return;
    }
    let at = scene.scatter.len();
    push_pcg_scatter(scene, vol, meshes, translation, id, clock);
    let packed = scene.scatter[at..].to_vec();
    scene.scatter_memo.insert(source, packed);
    // MIRROR-END pcg_scatter_memo
}

/// Project a [`PcgVolume`]'s evaluated cache into GPU-instanced scatter batches
/// (P18.5), anchored at the volume entity's world `translation`, carrying the
/// volume's authored content draw distance.
///
/// # The structure LOD (IB-2b)
///
/// A volume that grew **buildings** carries `structure_groups`, and its parts are
/// swapped for one **shell** box per building past
/// [`STRUCTURE_LOD_M`](inf_render::STRUCTURE_LOD_M).
///
/// | batch | band | what it holds |
/// |---|---|---|
/// | ungrouped | `[0, draw_distance)` | scatter and fences — content that has no shell to stand in for it |
/// | parts | `[0, lod + reach)` | every building's own boxes |
/// | shells | `[lod, draw_distance)` | one oriented box per building |
///
/// **The two bands overlap by `reach`, and that is the I3 audit's own
/// correction** (met here as a stale doc block, island wave I8a audit): a part
/// sits up to its shell's half-diagonal nearer the eye than the shell's centre
/// does, so cutting both at `lod` leaves a hole through the back of a building
/// rather than a level of detail.
///
/// **The camera is nowhere in this function**, which is the whole point: a
/// selection made on the CPU would put the eye inside a batch's *content key*,
/// and a content key that moves with the camera re-uploads a city's instance
/// buffer every time the player walks. The bands are constants of the content;
/// the cull compute — which has already computed each instance's distance —
/// makes the choice per instance, per frame.
///
/// A volume with no groups takes exactly the pre-I3 path: one batch, one band,
/// byte-identical.
///
/// MIRROR: identical in `inf_viewport::host` and `inf_player::render`, pinned by
/// `inf-editor-core`'s `tests/projector_mirror.rs`.
fn push_pcg_scatter(
    scene: &mut RenderScene,
    vol: &PcgVolume,
    meshes: &inf_render::ScatterMeshes,
    translation: DVec3,
    id: u32,
    clock: inf_render::ScatterClock,
) {
    // MIRROR-BEGIN pcg_scatter_lod
    if vol.structure_groups.is_empty() {
        push_scatter(
            scene,
            &vol.evaluated,
            meshes,
            translation,
            vol.draw_distance,
            0.0,
            id,
            clock,
            true,
        );
        return;
    }
    let lod = if vol.draw_distance > 0.0 {
        inf_render::STRUCTURE_LOD_M.min(vol.draw_distance)
    } else {
        inf_render::STRUCTURE_LOD_M
    };
    let first = vol.structure_groups[0].inst_start as usize;
    let last = vol
        .structure_groups
        .last()
        .map(|g| g.instance_range().end)
        .unwrap_or(first)
        .min(vol.evaluated.len());
    // Content the groups do not cover keeps its full band: a fence has no shell
    // to be replaced by, so cutting it at the LOD distance would delete it.
    let mut loose: Vec<ScatteredInstance> = Vec::new();
    loose.extend_from_slice(&vol.evaluated[..first.min(vol.evaluated.len())]);
    loose.extend_from_slice(&vol.evaluated[last..]);
    push_scatter(
        scene,
        &loose,
        meshes,
        translation,
        vol.draw_distance,
        0.0,
        id,
        clock,
        // Loose content -- a fence, a scatter -- has no shell standing in for
        // it, so it casts its own shadow exactly as it always has.
        true,
    );
    // **The two bands are complementary in the GROUP's distance, and the cull is
    // per INSTANCE** (I3 audit). A part sits up to its shell's own half-diagonal
    // nearer the eye than the shell's centre does, so cutting both at `lod`
    // leaves a building whose shell is just inside the line without the parts
    // that are just outside it — and with no shell to stand in for them. That is
    // a hole through the back of a building, not a level of detail. Widening the
    // parts band by exactly that reach makes a gap impossible; the price is a
    // `reach`-wide overlap in which a building draws its parts INSIDE its own
    // shell, which is bounded, contained, and never a hole.
    let reach = vol
        .structure_groups
        .iter()
        .map(|g| g.shell.half_extents.length())
        .fold(0.0_f64, f64::max);
    let parts_far = if vol.draw_distance > 0.0 {
        (lod + reach).min(vol.draw_distance)
    } else {
        lod + reach
    };
    push_scatter(
        scene,
        &vol.evaluated[first.min(last)..last],
        meshes,
        translation,
        parts_far,
        0.0,
        id,
        clock,
        // **THE PARTS DO NOT CAST; THEIR SHELL DOES** (island wave I8b). Every
        // instance in this batch stands inside the oriented box `push_shells`
        // is about to emit, and that box is packed as a caster at every
        // distance -- `pack_fallback` reads `draw_distance` and ignores
        // `near_distance`. So the silhouette is unchanged and the CPU caster
        // pack stops walking a city: 365 545 instances tested to keep 16 384 of
        // them was 16.7 ms a frame and 70 % of the record stage.
        false,
    );
    let shells: Vec<ScatteredInstance> = vol
        .structure_groups
        .iter()
        .map(|g| ScatteredInstance {
            position: g.shell.center,
            rotation: g.shell.rotation,
            // The primitive is a UNIT cube (half-extent 0.5), so a box of
            // half-extents `h` is a scale of `2h`.
            scale: 1.0,
            // The shell wears the colour of the building's first part, so a
            // district of offices does not turn grey at the LOD distance.
            kind: vol
                .evaluated
                .get(g.inst_start as usize)
                .map_or(0, |i| i.kind),
            // A shell is a BOX by definition (see `push_shells`), so it names no
            // mesh however many its parts name -- and `push_shells` reads its
            // three half-extents off the group rather than off the instance, so
            // the I8b extent has nothing to say here either.
            mesh: None,
            extent: None,
            glow: 0.0,
            // **A shell never emits** (wave VEN1a). It wears the first part's
            // metal and roughness for the same reason it wears its colour --
            // a district of chrome-fronted clubs must not turn to plaster at
            // the LOD distance -- but its EMISSION is dropped, because a shell
            // is one box standing for a whole building and a building whose
            // first part happened to be a neon plate would become a
            // building-sized neon plate at 96 m.
            surface: inf_ecs::components::ScatteredSurface {
                emissive: [0.0; 3],
                pulse_hz: 0.0,
                ..vol
                    .evaluated
                    .get(g.inst_start as usize)
                    .map_or(inf_ecs::components::ScatteredSurface::DEFAULT, |i| {
                        i.surface
                    })
            },
        })
        .collect();
    push_shells(scene, &shells, vol, translation, vol.draw_distance, lod, id);
    // MIRROR-END pcg_scatter_lod
}

/// Project a [`Terrain`]'s **biome population** — P19.3's biome→PCG binding, i.e.
/// each painted biome's `.inf_pcg` graph evaluated over the region its id owns —
/// into ONE GPU-instanced scatter batch. Body: [`push_scatter`], so a population
/// and a volume are packed, shaded, culled and picked by the very same code.
///
/// Instance positions are already ABSOLUTE world positions (the binding evaluates
/// against the terrain's world heightfield), so the batch anchors at the terrain's
/// own origin exactly as a volume anchors at its centre.
///
/// **Draw distance `0` = UNLIMITED, deliberately.** A `PcgVolume` has an authored
/// per-volume knob and it can only clamp the renderer's bands DOWN
/// (`inf_render::ScatterSettings`); a terrain population has no such authored
/// field, so `0` leaves the global `ScatterSettings` — the host's own tier-clamped
/// budget — in sole charge, rather than inventing a content limit nobody authored.
///
/// MIRROR: identical in `inf_viewport::host` and `inf_player::render`, pinned by
/// `inf-editor-core`'s `tests/projector_mirror.rs`.
fn push_biome_population(
    scene: &mut RenderScene,
    terrain: &Terrain,
    meshes: &inf_render::ScatterMeshes,
    translation: DVec3,
    id: u32,
    clock: inf_render::ScatterClock,
) {
    push_scatter(
        scene,
        &terrain.biome_population,
        meshes,
        translation,
        0.0,
        0.0,
        id,
        clock,
        true,
    )
}

/// Project a [`Foliage`] component's painted instances into GPU-instanced scatter
/// batches (P18.5): mesh + tint from the referenced palette slot.
///
/// Instances are entity-LOCAL, so the batch anchor is the entity `translation` and
/// the packed offsets are the local positions with **no conversion**. That is what
/// makes the payload a pure function of the paint stroke: the same stroke placed
/// twice content-hashes to one GPU upload however far apart the two entities sit
/// (the anchor is deliberately not part of `ScatterData::key`).
///
/// The palette resolves a primitive kind PER INSTANCE and one batch draws exactly
/// one kind, so instances bucket by resolved kind in authored order and the buckets
/// emit in [`PrimMesh::ALL`] order — deterministic, and independent of which kinds
/// the palette happens to use.
///
/// Every batch of one entity shares ONE pick `id` (see [`push_pcg_scatter`]).
///
/// MIRROR: identical in `inf_viewport::host` and `inf_player::render`, pinned by
/// `inf-editor-core`'s `tests/projector_mirror.rs`.
fn push_foliage_scatter(scene: &mut RenderScene, fol: &Foliage, translation: DVec3, id: u32) {
    if fol.instances.is_empty() {
        return;
    }
    let mut buckets: [Vec<ScatterInstance>; PrimMesh::ALL.len()] = Default::default();
    for fi in &fol.instances {
        let (mesh, color) = fol
            .palette
            .get(fi.kind as usize)
            .map(|p| (prim_mesh(p.primitive), p.tint.to_array()))
            .unwrap_or((PrimMesh::Cube, [0.28, 0.52, 0.24, 1.0]));
        buckets[mesh.index()].push(ScatterInstance {
            // Entity-LOCAL, paired with the ZERO build-anchor below.
            position: fi.position.to_dvec3(),
            rotation: foliage_rot_quat(fi.rotation),
            scale: glam::Vec3::splat(fi.scale as f32),
            color,
        });
    }
    for (k, bucket) in buckets.into_iter().enumerate() {
        if bucket.is_empty() {
            continue;
        }
        let data = ScatterData::build(PrimMesh::ALL[k], DVec3::ZERO, bucket);
        scene.scatter.push(ScatterBatch {
            data: Arc::new(data),
            anchor: translation,
            metallic: 0.0,
            roughness: 0.85,
            emissive: [0.0; 3],
            id,
            draw_distance: 0.0,
            near_distance: 0.0,
            casts_shadows: true,
        });
    }
}

/// Project a [`WaterBody`] (+ the [`Spline`] on the **same entity**, for a river)
/// into a [`RenderWater`] (P20.1).
///
/// MIRROR: this body is byte-identical in `inf_viewport::host` and
/// `inf_player::render`, and `projector_mirror.rs` compares it character for
/// character — like `project_sky`, and for the same reason: neither Ring-0 crate
/// can host it (`inf-render` does not depend on `inf-ecs`, and `inf-ecs` must not
/// depend on `inf-render`), so it is written twice on purpose and gated.
///
/// The two things that *could* silently diverge live in Ring 0 instead:
/// [`inf_ecs::sky::water_environment`] decides what clock and wind a body sees,
/// and [`WaterBody::effective_wind`] decides whether this body follows them. A
/// host that inlined either would be exactly the drift this gate exists to stop.
///
/// `env` is `(level clock in seconds, weather wind (m/s))` — resolved once per
/// projection, never per body, and never from a wall clock.
fn project_water(
    water: &WaterBody,
    spline: Option<&Spline>,
    affine: &glam::DAffine3,
    env: (f64, (f64, f64)),
    flow: &inf_ecs::hydro::TerrainFlow<'_>,
    id: u32,
) -> RenderWater {
    let (time_s, weather_wind) = env;
    let (wind_x, wind_z) = water.effective_wind(weather_wind);
    // A river's ripple travels DOWNSTREAM: its wave frame is (arc length,
    // lateral), so the "wind" is +1 along the river rather than a world
    // direction. Everything else responds to the level's wind.
    let river = water.kind == WaterKind::River;
    let spec = inf_render::WaveSpec {
        amplitude_m: water.wave_amplitude_m,
        wavelength_m: water.wave_length_m,
        steepness: water.wave_steepness,
        wind_x: if river { 1.0 } else { wind_x },
        wind_z: if river { 0.0 } else { wind_z },
        // Degrees at the component boundary, radians below it (the units
        // doctrine); the conversion is a multiply, so it stays bit-portable.
        spread_rad: water.wave_spread_deg.to_radians(),
        seed: water.wave_seed,
        count: water.wave_count,
    };
    let mut out = RenderWater {
        id,
        kind: match water.kind {
            WaterKind::Ocean => inf_render::WaterKindGpu::Ocean,
            WaterKind::Lake => inf_render::WaterKindGpu::Lake,
            WaterKind::River => inf_render::WaterKindGpu::River,
        },
        level_m: water.level_m,
        center: glam::DVec2::new(affine.translation.x, affine.translation.z),
        half_extent: glam::DVec2::new(water.extent.x.max(0.0), water.extent.y.max(0.0)),
        frames: Vec::new(),
        // Forwarded, not dropped (P20.3): `RenderWater::surface` hands it back to
        // the Ring-0 `RiverPath` so the renderer's reconstruction is the path the
        // projector built, flag and all.
        spline_closed: spline.is_some_and(|sp| sp.closed),
        waves: inf_render::WaveField::from_spec(&spec),
        time_s,
        flow_speed_m_s: 0.0,
        shallow_color: [
            water.shallow_color.r,
            water.shallow_color.g,
            water.shallow_color.b,
        ],
        deep_color: [water.deep_color.r, water.deep_color.g, water.deep_color.b],
        absorption: [
            water.absorption.x.max(0.0) as f32,
            water.absorption.y.max(0.0) as f32,
            water.absorption.z.max(0.0) as f32,
        ],
        roughness: water.roughness.clamp(0.0, 1.0) as f32,
        refraction_m: water.refraction_m.max(0.0) as f32,
        shore_fade_m: water.shore_fade_m.max(0.0) as f32,
        opacity: water.opacity.clamp(0.0, 1.0) as f32,
        foam_color: [water.foam_color.r, water.foam_color.g, water.foam_color.b],
        foam_crest_threshold: water.foam_crest_threshold.clamp(0.0, 1.0) as f32,
        foam_shore_m: water.foam_shore_m.max(0.0) as f32,
        foam_flow_m_s: water.foam_flow_m_s.max(0.0) as f32,
    };
    // A river's centreline is the spline on this same entity, in world space.
    // No spline ⇒ no ribbon, and `RenderWater::drawable` skips it: an authoring
    // state, not an error.
    if river {
        if let Some(sp) = spline {
            let points: Vec<DVec3> = sp
                .points
                .iter()
                .map(|p| affine.transform_point3(p.to_dvec3()))
                .collect();
            let interp = match sp.interp {
                inf_ecs::components::SplineInterp::Linear => inf_math::spline::SplineInterp::Linear,
                inf_ecs::components::SplineInterp::CatmullRom => {
                    inf_math::spline::SplineInterp::CatmullRom
                }
            };
            // ONE sanitizer, in Ring 0 (P20.4): the cook, the fixed step and both
            // projectors all build their profile here, so a negative authored
            // depth cannot taper one of them differently from the others.
            let profile = inf_render::RiverProfile::authored(
                water.river_width_start_m,
                water.river_width_end_m,
                water.river_depth_start_m,
                water.river_depth_end_m,
                water.river_flow_m_s,
            );
            let path = inf_render::RiverPath::from_points(&points, sp.closed, interp, &profile);
            out.flow_speed_m_s = path.flow_speed_m_s;
            out.level_m = path
                .frames
                .first()
                .map(|f| f.center.y)
                .unwrap_or(water.level_m);
            // P20.4: the P19.1 flow map modulates each frame's foam. The gain
            // is `1.0` wherever the terrain was never eroded, so this loop is
            // the identity on every level that has no bake — and the whole query
            // is skipped when the level has none, which is the common case.
            let mapped = flow.is_mapped();
            out.frames = path
                .frames
                .iter()
                .map(|f| {
                    let mut wf = inf_render::WaterFrame::from(f);
                    if mapped {
                        wf.flow_gain = flow.foam_gain_at(glam::DVec2::new(f.center.x, f.center.z));
                    }
                    wf
                })
                .collect();
        }
    }
    out
}

/// Project the level's **sky authority** into the renderer's sun + sky blocks
/// (P17.1) — the seam that retired `inf_render::camera::SUN_DIR`.
///
/// **MIRROR**: byte-for-byte identical in both hosts' `project_sky`, this doc
/// block included, and pinned by `projector_mirror.rs`. (It sat above
/// `project_water` in both files until the P21.2 re-audit — the mirror gate could
/// not see it there, and neither could a reader looking for it here.)
/// The one thing that *could* silently diverge — *which* entity is the authority,
/// since the editor walks document order and the player walks `Guid` order —
/// deliberately does not live here: [`inf_ecs::sky::resolve_sky`] answers it once,
/// in Ring 0, by lowest `Guid`.
///
/// With no authority the renderer's own defaults stand: the retired constant's
/// direction and the historic three-colour gradient, so every level that has not
/// opted into time of day renders exactly the pixels it always did.
///
/// When a clock is present the sun (or, once it has set, the moon) is also pushed
/// as a **directional light**, so shadows, GI and the PBR loop all follow the
/// clock without any of those passes knowing time of day exists. It goes in
/// first, before the entity loop, so its index is stable on both sides. A level
/// that would rather author its own suns sets `SkyAtmosphere::enabled = false`,
/// which keeps the clock and the tint but projects no light.
fn project_sky(scene: &mut RenderScene, world: &inf_ecs::EcsWorld) {
    let Some(sky) = inf_ecs::sky::resolve_sky(world) else {
        scene.sun = SunParams::default();
        scene.sky = SkyParams::default();
        scene.atmosphere = AtmosphereParams::default();
        return;
    };
    let a = &sky.atmosphere;
    let phase = sky.moon_phase as f32;
    scene.sun = SunParams {
        direction: sky.sun.as_vec3(),
        color: [a.sun_color.r, a.sun_color.g, a.sun_color.b],
        intensity: a.sun_intensity,
        moon_direction: sky.moon.as_vec3(),
        moon_color: [a.moon_color.r, a.moon_color.g, a.moon_color.b],
        moon_intensity: a.moon_intensity,
        moon_phase: phase,
    };
    let [zenith, horizon, ground] = sky.sky_gradient();
    scene.sky = SkyParams {
        zenith,
        horizon,
        ground,
    };
    // The **weather in force** (P17.4), resolved once in Ring 0: when the
    // weather block is enabled it *drives* cloud coverage/type, the wind and the
    // fog density; when it is not, those come from the authored fields exactly
    // as they did in v13. Which of the two applies is decided by
    // `ResolvedSky::weather`, not here — it is precisely the kind of one-line
    // derivation two byte-identical MIRROR bodies would eventually stop agreeing
    // about, which is the same reasoning that put `cloud_time_s` in Ring 0.
    let w = sky.weather();
    // The physical atmosphere (P17.2). Only the *multipliers* come from the
    // level: the Rayleigh / Mie / ozone coefficients themselves are physical
    // constants of Earth's air and stay at `AtmosphereParams::default()`, so
    // "atmosphere" cannot be mis-authored into something that is not one.
    scene.atmosphere = AtmosphereParams {
        enabled: a.physical,
        turbidity: a.turbidity,
        mie_g: a.mie_anisotropy,
        sky_intensity: a.sky_intensity,
        aerial_perspective: a.aerial_perspective,
        tint_strength: a.tint_strength,
        sun_disc_deg: a.sun_disc_deg,
        moon_disc_deg: a.moon_disc_deg,
        moon_phase: phase,
        star_intensity: a.star_intensity,
        fog: HeightFog {
            density: w.fog_density,
            falloff: a.fog_falloff,
            height: a.fog_height,
            color: [a.fog_color.r, a.fog_color.g, a.fog_color.b],
        },
        // Volumetric clouds (P17.3). `time_s` is the one field here that is
        // *derived* rather than authored: the wind drifts with the level's clock
        // (`ResolvedSky::cloud_time_s`, defined once in Ring 0) and with nothing
        // else, so two runs at the same time of day see the same sky.
        clouds: CloudParams {
            enabled: a.clouds_enabled,
            coverage: w.cloud_coverage,
            cloud_type: w.cloud_type,
            bottom: a.cloud_bottom,
            top: a.cloud_top,
            density: a.cloud_density,
            detail: a.cloud_detail,
            seed: a.cloud_seed,
            wind_x: w.wind_x,
            wind_z: w.wind_z,
            time_s: sky.cloud_time_s(),
            phase_g: a.cloud_phase_g,
            shadow_strength: a.cloud_shadow,
            ambient: a.cloud_ambient,
            color: [a.cloud_color.r, a.cloud_color.g, a.cloud_color.b],
        },
        // Precipitation (P17.4). Entirely derived: the weather block decides
        // whether it falls, how hard and how frozen, and the same wind that
        // drifts the clouds slants it. `time_s` is the level's clock again, so
        // the rain is a function of the document and of nothing else. The tint
        // is the cloud droplet colour on purpose — rain and the cloud it fell
        // out of are the same water, and a second colour field would be one more
        // thing to keep consistent for a stylised sky.
        precip: PrecipParams {
            enabled: w.precipitation > 0.0,
            intensity: w.precipitation,
            snowiness: w.snowiness,
            wind_x: w.wind_x,
            wind_z: w.wind_z,
            time_s: sky.cloud_time_s(),
            color: [a.cloud_color.r, a.cloud_color.g, a.cloud_color.b],
        },
        ..AtmosphereParams::default()
    };
    if let Some((direction, color, intensity)) = sky.key_light() {
        scene.lights.push(RenderLight {
            kind: LightKind::Directional,
            color,
            intensity,
            direction: direction.as_vec3(),
            position: DVec3::ZERO,
            range: 0.0,
            cast_shadows: true,
            ..RenderLight::default()
        });
    }
}

/// Project the surface deformation field (P22.1) onto the scene.
///
/// **MIRROR** — byte-identical in `inf_viewport::host` and `inf_player::render`,
/// including this doc block, and pinned by `projector_mirror.rs`.
///
/// The projection is **epoch-gated rather than rebuilt**, which is the one thing
/// this differs from every other list in the two projectors. The field only moves
/// when something walked; a standing character would otherwise pay a copy of the
/// whole live cell set sixty times a second for a projection identical to the last
/// one. `RenderScene::deform` is an `Arc`, so an unchanged field costs one integer
/// compare and nothing else — and the renderer's upload gate keys on the same
/// epoch, so an unchanged field also uploads nothing (`inf_render::deform`).
///
/// The camera is nowhere in here. What is projected is the whole live field in
/// its own lattice coordinates; which 128 m of it gets drawn is decided later, in
/// the renderer, where a camera is legal.
fn project_deform(scene: &mut RenderScene, field: Option<&inf_terrain::deform::DeformField>) {
    let Some(field) = field.filter(|f| !f.is_empty()) else {
        scene.deform = None;
        return;
    };
    if scene
        .deform
        .as_ref()
        .is_some_and(|d| d.epoch == field.epoch())
    {
        return;
    }
    scene.deform = Some(std::sync::Arc::new(inf_render::RenderDeform {
        cell_samples: inf_terrain::deform::DEFORM_CELL_SAMPLES,
        texel_m: inf_terrain::deform::DEFORM_SAMPLE_PITCH_M,
        epoch: field.epoch(),
        cells: field
            .cells()
            .map(|(coord, cell)| inf_render::RenderDeformCell {
                coord: *coord,
                depths: cell.depths().to_vec(),
            })
            .collect(),
    }));
}

/// Project an ECS `Light` (+ its world transform) into a renderer light (R-P3).
///
/// Direction conventions (**mirrored byte-for-byte** in the player's
/// `inf_player::render::project_light` — the parity tests in both crates pin
/// them so the classic mirror bug can never drift):
///  * Directional/spot store the vector *toward* the light = `rot * +Z` (an
///    entity's forward is `-Z`, so this is the anti-emission direction);
///  * the renderer derives a spot's beam emission as `-direction = rot * -Z`.
///
/// Cone half-angles convert to cosines CPU-side (std trig is fine — this is not
/// committed content). `range` and `cast_shadows` pass through for all kinds
/// (fixing the earlier point-range-hardcoded-0 bug); `cast_shadows` is inert for
/// point/spot (shadow maps deferred).
fn project_light(light: &Light, affine: &glam::DAffine3) -> RenderLight {
    let (_, rot, translation) = affine.to_scale_rotation_translation();
    let c = light.color.to_array();
    let color = [c[0], c[1], c[2]];
    match light.kind {
        EcsLightKind::Directional => RenderLight {
            kind: LightKind::Directional,
            color,
            intensity: light.intensity,
            // Direction *toward* the light: the transform's +Z (emission is −Z).
            direction: (rot * DVec3::Z).as_vec3(),
            position: DVec3::ZERO,
            range: 0.0,
            cast_shadows: light.cast_shadows,
            ..RenderLight::default()
        },
        EcsLightKind::Point => RenderLight {
            kind: LightKind::Point,
            color,
            intensity: light.intensity,
            direction: Vec3::ZERO,
            position: translation,
            range: light.range,
            cast_shadows: light.cast_shadows,
            ..RenderLight::default()
        },
        EcsLightKind::Spot => RenderLight {
            kind: LightKind::Spot,
            color,
            intensity: light.intensity,
            // Toward-the-light (like directional); emission = -direction = rot·−Z.
            direction: (rot * DVec3::Z).as_vec3(),
            position: translation,
            range: light.range,
            inner_cos: light.inner_cone_deg.to_radians().cos(),
            outer_cos: light.outer_cone_deg.to_radians().cos(),
            cast_shadows: light.cast_shadows,
        },
    }
}

/// Project an ECS [`Sprite`] (+ its world position) into a renderer sprite.
///
/// The texture GUID maps to a `TextureHandle`, but the viewport thread has no
/// asset-DB access yet, so no RGBA bytes are pushed to
/// `RenderScene::pending_texture_uploads` — referenced sprites render as the
/// renderer's white fallback tinted by `color` (a colored quad). Resolving the
/// texture bytes in the viewport is the same documented follow-up as rendering
/// imported mesh geometry (both need the asset DB threaded into the viewport;
/// the headless golden test exercises the full textured path). Rotation is left
/// at 0 for P8.1a (2D rotation tooling arrives in P8.2).
fn project_sprite(sprite: &Sprite, translation: DVec3) -> SpriteInstance {
    SpriteInstance {
        position: translation,
        size: Vec2::new(sprite.size.x as f32, sprite.size.y as f32),
        pivot: Vec2::new(sprite.pivot.x as f32, sprite.pivot.y as f32),
        rotation: 0.0,
        uv_min: Vec2::new(
            sprite.atlas_rect.min.x as f32,
            sprite.atlas_rect.min.y as f32,
        ),
        uv_max: Vec2::new(
            sprite.atlas_rect.max.x as f32,
            sprite.atlas_rect.max.y as f32,
        ),
        color: sprite.color.to_array(),
        texture: sprite
            .texture
            .map(|u| handle_from_guid(u.as_u128()))
            .unwrap_or(inf_render::WHITE_TEXTURE),
        sorting_layer: sprite.sorting_layer,
        order: sprite.order,
        flip_x: sprite.flip_x,
        flip_y: sprite.flip_y,
        billboard: billboard_mode(sprite.billboard),
    }
}

/// Map the ECS [`BillboardMode`] enum onto the renderer's `u8` billboard flag
/// (P8.4a) — the sprite pass orients the quad by the camera basis for the
/// non-planar modes.
fn billboard_mode(mode: inf_ecs::BillboardMode) -> u8 {
    match mode {
        inf_ecs::BillboardMode::None => inf_render::BILLBOARD_NONE,
        inf_ecs::BillboardMode::Spherical => inf_render::BILLBOARD_SPHERICAL,
        inf_ecs::BillboardMode::Cylindrical => inf_render::BILLBOARD_CYLINDRICAL,
    }
}

/// Project an ECS [`Tilemap`] (+ its world position) into a [`RenderTilemap`].
///
/// The atlas texture GUID maps to a `TextureHandle`, but — like [`project_sprite`]
/// — the viewport thread has no asset-DB access yet, so no RGBA bytes are pushed:
/// referenced tilemaps render as the white fallback tinted by `tint` (colored
/// cells). The headless golden test exercises the full textured path. The chunk
/// data is copied out of the sparse ECS store once per document version; the
/// sprite pass culls + expands it per frame.
fn project_tilemap(tilemap: &Tilemap, translation: DVec3) -> RenderTilemap {
    let params = TilemapParams {
        origin: translation,
        tile_size: Vec2::new(tilemap.tile_size.x as f32, tilemap.tile_size.y as f32),
        atlas_cols: tilemap.atlas_cols,
        atlas_rows: tilemap.atlas_rows,
        texture: tilemap
            .texture
            .map(|u| handle_from_guid(u.as_u128()))
            .unwrap_or(inf_render::WHITE_TEXTURE),
        color: tilemap.tint.to_array(),
        sorting_layer: tilemap.sorting_layer,
        order: tilemap.order,
    };
    let chunks = tilemap
        .occupied_chunks()
        .map(|(&coord, chunk)| RenderChunk {
            coord,
            tiles: chunk.tiles().to_vec(),
        })
        .collect();
    RenderTilemap { params, chunks }
}

/// The `(key, world origin, stamp)` signature of `data`'s tile list, in exactly
/// the order [`project_terrain`] emits it: level 0 ascending, then the coarse
/// pyramid ascending.
///
/// It is the **key** of Hardening Wave E's P1 memo, which is why it is a named
/// function rather than an expression inside one: a host whose signature
/// disagreed with what its own projection builds would carry a stale terrain
/// forward for ever, and nothing but a screenshot would say so.
///
/// **MIRROR**: byte-identical in `inf_viewport::host` and `inf_player::render`,
/// pinned by `inf-editor-core`'s `tests/projector_mirror.rs`.
fn tile_signature(
    data: &inf_terrain::TerrainData,
    translation: DVec3,
) -> impl Iterator<Item = (TerrainTileKey, DVec3, u64)> + '_ {
    data.tiles()
        .map(|(&coord, tile)| (inf_terrain::TileKey::lod0(coord), tile))
        .chain(data.coarse_tiles().map(|(&key, tile)| (key, tile)))
        .map(move |(key, tile)| {
            (
                TerrainTileKey::new(key.lod, key.coord),
                tile.origin + translation,
                data.tile_version(key),
            )
        })
}

/// Project an ECS [`Terrain`] (+ its world translation) into a [`RenderTerrain`].
///
/// `data` is the working set to draw and is passed **explicitly** (P16.3b2): for
/// an inline terrain it is `terrain.data`, for a streamed one it is the
/// streamer's camera-driven set. `terrain` still supplies the layers and macro
/// variation, which are authored, not streamed. Making the choice a parameter is
/// what keeps "which residency am I drawing?" a decision at the call site rather
/// than an assumption buried here.
///
/// Each **resident** tile becomes a [`RenderTerrainTile`] with its `f64` origin
/// offset by the entity's world translation (so the terrain follows its
/// transform), its `f32` height buffer copied out of the paged data, its height
/// bounds precomputed for the terrain pass's per-tile frustum cull, and its
/// monotone change stamp so the GPU cache re-uploads only what actually moved
/// (P16.3b1).
///
/// Level 0 (the authored heightfield) is emitted first, then the resident coarse
/// pyramid pages in ascending key order — both from `BTreeMap`s, so the tile list
/// is globally `TileKey`-ascending and the upload/draw order is deterministic. An
/// inline (non-asset) terrain holds no coarse pages, so it projects exactly the
/// level-0 list it always did.
///
/// **MIRROR** of `inf_player::render::project_terrain` — keep the two in sync.
///
/// `prev` is the previous frame's terrain list, which the caller has taken out
/// of the scene: a terrain whose grid, layers and whole per-tile stamp sequence
/// are unchanged is **carried forward from it** rather than rebuilt (Hardening
/// Wave E's P1 memo — see [`inf_render::take_unchanged_terrain`] for why the key
/// is sound). Pass `&mut Vec::new()` for a one-shot projection with nothing to
/// carry.
fn project_terrain(
    guid: Uuid,
    terrain: &Terrain,
    data: &TerrainData,
    translation: DVec3,
    biome_palette: &[[f32; 4]],
    // The live virtual-texture registry (TER2a) — `None` on a level with no
    // bindings, and then every layer's set is `VtTextureSet::NONE`.
    vt: Option<&inf_render::VtTextures>,
    prev: &mut Vec<RenderTerrain>,
) -> RenderTerrain {
    let res = data.tile_resolution();
    let n = (res * res) as usize;
    // P16.6: the terrain entity's identity, folded exactly as the player's
    // mirror folds it — needed BEFORE the tiles are built, because it is the
    // memo's key.
    let id = inf_render::terrain_id_from_guid(guid.as_u128());
    let layers = std::array::from_fn(|k| RenderTerrainLayer {
        albedo: terrain.layers[k].albedo.to_array(),
        roughness: terrain.layers[k].roughness as f32,
        tex_scale: terrain.layers[k].tex_scale as f32,
        // TER2a: the authoring field Wave G added, resolved through the SAME
        // door every mesh instance's set goes through. Before this the two hosts
        // both spelled `vt: Default::default()` with a comment saying
        // `TerrainLayer` carried no texture reference -- which stopped being
        // true at Wave G and left the four-layer VT branch unreachable by
        // construction on every terrain in the engine.
        vt: inf_render::vt_set_for(vt, terrain.layers[k].material.map(|m| m.as_u128())),
    });
    let macro_variation = terrain.macro_variation as f32;
    // Hardening Wave E's P1 memo: the signature of the tile list this projection
    // WOULD build — the same walk `project_tile` takes below, reduced to
    // `(key, world origin, stamp)`. When it matches the carried terrain's, the
    // carried one IS the answer and not one height buffer is copied.
    if let Some(kept) = inf_render::take_unchanged_terrain(
        prev,
        id,
        res,
        data.meters_per_sample(),
        &layers,
        macro_variation,
        biome_palette,
        tile_signature(data, translation),
    ) {
        return kept;
    }
    let project_tile = |key: inf_terrain::TileKey, tile: &inf_terrain::TerrainTile| {
        // Resolve the sparse weight store into a full res² buffer for upload
        // (an unpainted tile → uniform default layer 0; a coarse pyramid page is
        // always unpainted — the pyramid is heights-only).
        let weights: Vec<[u8; 4]> = if tile.weights_are_default() {
            vec![inf_terrain::DEFAULT_WEIGHT; n]
        } else {
            (0..res)
                .flat_map(|j| (0..res).map(move |i| (i, j)))
                .map(|(i, j)| tile.weight_sample(res, i, j))
                .collect()
        };
        // Same sparse resolution for the P19.2 biome ids: an unpainted tile — and
        // every coarse pyramid page, which is a streaming page rather than
        // authored content — projects as uniform `UNASSIGNED_BIOME`.
        let biomes: Vec<u8> = if tile.biomes_are_default() {
            vec![inf_terrain::UNASSIGNED_BIOME; n]
        } else {
            (0..res)
                .flat_map(|j| (0..res).map(move |i| (i, j)))
                .map(|(i, j)| tile.biome_sample(res, i, j))
                .collect()
        };
        RenderTerrainTile {
            key: TerrainTileKey::new(key.lod, key.coord),
            origin: tile.origin + translation,
            heights: tile.heights().to_vec(),
            weights,
            biomes,
            height_bounds: tile.height_bounds(),
            // P21.2: the packed hole mask, row-aligned for the GPU by the one
            // function that knows the tile's own bit layout. A tile nothing has
            // carved projects an empty vec — the sparse default, intact.
            holes: inf_terrain::pack_hole_rows(tile, res),
            version: data.tile_version(key),
        }
    };
    let tiles = data
        .tiles()
        .map(|(&coord, tile)| project_tile(inf_terrain::TileKey::lod0(coord), tile))
        .chain(
            data.coarse_tiles()
                .map(|(&key, tile)| project_tile(key, tile)),
        )
        .collect();
    RenderTerrain {
        // The 64-bit fold computed above — what keeps two terrains' GPU tile
        // caches and splat uniforms apart when their grids share coordinates.
        id,
        tile_resolution: res,
        meters_per_sample: data.meters_per_sample(),
        tiles,
        layers,
        macro_variation,
        biome_palette: biome_palette.to_vec(),
    }
}

/// Project one loaded voxel volume into a [`RenderVoxelVolume`] (P21.1).
///
/// `slot` is the host's loaded working set for this entity — its resident chunks
/// and its meshed surface — owned by the Ring-0 `inf_voxel::VoxelVolumes` store, so
/// the two hosts cannot mesh the same field differently. What is left here is the
/// mapping into renderer types, and three rules that would drift silently if each
/// host wrote its own:
///
/// * **The asset's voxel scale wins.** Chunk origins and vertex positions are both
///   derived from `slot.data.voxel_size_m()` — the scale recorded in the
///   `.inf_voxel` header — so the geometry is self-consistent. Reading the
///   component's `voxel_size_m` here instead would scale vertices against origins
///   derived from the asset's and tear the volume apart wherever the two disagree.
///   The component's value is what a *new* asset is authored at, and the cook
///   **does** report the disagreement: `inf_packager::cook`'s
///   `voxel_scale_mismatches` raises one advisory per level + volume, naming both
///   numbers and which one wins (P21.2).
/// * **A voxel material index IS a terrain splat index**, so a volume shades with
///   the `Terrain` on **this same entity** when there is one — composition, not a
///   reference, so no cook edge and nothing to dangle — and with the default
///   palette otherwise. A cave mouth has to shade continuously into the hillside it
///   opens out of, and it cannot if the two sides read different palettes.
/// * **A volume with no surface projects `None`**, never an empty chunk list: the
///   scene's `voxels` being empty is exactly what keeps the voxel pass off the
///   command encoder, and every existing golden depends on that.
///
/// **MIRROR** of the other host's `project_fracture` — keep the two
/// byte-identical, **this doc block included** (the P21.2 lesson recorded on
/// `project_voxel`: the mirror gate compares the comment too).
///
/// Project one destructible actor's live chunks (P22.3).
///
/// # The atomicity contract lives here
///
/// An actor is drawn as its own mesh **or** as its chunks — never both, never
/// neither. `None` means "still whole, draw the mesh"; `Some` means "broken, do
/// not". Both halves read ONE fact, `FractureState::is_intact`, which is the same
/// fact `PhysicsBridge3D::sync_from_world_sim` reads to decide whether the actor
/// keeps its collider. Three consumers, one predicate, so the swap cannot become
/// an ordering accident between passes.
///
/// # Chunk-local geometry against an `f64` anchor
///
/// The `.inf_fracture`'s vertices are in the SOURCE MESH's local space and the
/// state's placement maps them into the world — but a *detached* chunk's pose is
/// solver-owned, so its geometry is re-anchored on its own centre of mass and its
/// world pose rides on the instance. That is what lets the vertex buffer be
/// uploaded once per break while the pose changes every step.
///
/// A **reclaimed** chunk is emitted by nobody: it leaves the render set and the
/// physics world on the same generation, which is what makes the debris budget's
/// despawn one event rather than two.
fn project_fracture(
    state: &inf_physics::d3::FractureState,
    material: Option<&Material>,
    id: u64,
) -> Option<Vec<inf_render::RenderFractureChunk>> {
    if state.is_intact() {
        return None;
    }
    let (color, metallic, roughness, emissive) = match material {
        Some(m) => (m.base_color.to_array(), m.metallic, m.roughness, {
            let e = m.emissive_linear();
            [e[0], e[1], e[2], 1.0]
        }),
        None => ([0.8, 0.8, 0.8, 1.0], 0.0_f32, 0.5_f32, [0.0; 4]),
    };
    let placement = state.placement();
    let asset = state.asset();
    let version = state.generation();
    let mut out = Vec::new();
    for (i, chunk) in state.chunks().iter().enumerate() {
        if chunk.gone {
            continue;
        }
        let Some(src) = asset.chunks.get(i) else {
            continue;
        };
        let centre = placement.transform_point3(DVec3::from_array(src.center_of_mass));
        let vertices: Vec<inf_render::RenderFractureVertex> = src
            .vertices
            .iter()
            .map(|v| {
                let p = placement.transform_point3(DVec3::new(
                    v.position[0] as f64,
                    v.position[1] as f64,
                    v.position[2] as f64,
                )) - centre;
                let n = placement.matrix3
                    * DVec3::new(v.normal[0] as f64, v.normal[1] as f64, v.normal[2] as f64);
                let n = n.normalize_or_zero();
                inf_render::RenderFractureVertex {
                    pos: [p.x as f32, p.y as f32, p.z as f32],
                    normal: [n.x as f32, n.y as f32, n.z as f32],
                    // The chunk's own uv (P26.5). A placement rotates positions
                    // and normals; a parametrization is not a direction, so it
                    // rides across untouched.
                    uv: v.uv,
                }
            })
            .collect();
        out.push(inf_render::RenderFractureChunk {
            entity: id,
            chunk: i as u32,
            translation: chunk.translation,
            rotation: chunk.rotation.as_quat(),
            vertices,
            indices: src.indices.clone(),
            color,
            metallic,
            roughness,
            emissive: [emissive[0], emissive[1], emissive[2]],
            version,
        });
    }
    Some(out)
}

/// **MIRROR** of the other host's `project_debris` — keep the two
/// byte-identical, **this doc block included** (the P21.2 lesson recorded on
/// `project_voxel`: the mirror gate compares the comment too).
///
/// Project one broken destructible's **sub-chunk rubble** (P22.4).
///
/// # It is dressing, and it never reaches the sim
///
/// P22.3 already draws the chunks themselves — real solver-owned bodies. This is
/// the visual rubble *below* that scale: fragments too small to be worth a convex
/// hull, laid deterministically around each live chunk by
/// [`inf_render::debris_instances`] and shipped as ONE `ScatterBatch` per broken
/// actor down the P18.5 GPU instance path. Nothing here is a body, nothing here
/// is queried, and deleting this function would leave the simulation bit-identical.
///
/// # Why the site is the REST centre and not the live pose
///
/// `ScatterData::key` is a content hash over the packed instance bytes, so a
/// batch whose instances moved every step would re-upload its whole buffer every
/// step — the exact cost the scatter path exists to avoid. The rubble is
/// therefore anchored on each chunk's `chunk_rest_center`, which `FractureState`
/// freezes when the first chunk detaches, so the batch's content changes only
/// when the live chunk set does: **one upload per break**, structurally, with no
/// stamp to get wrong. It is also the honest picture — the fragments are the
/// spall left *at* the break, not a second swarm of projectiles.
///
/// A **reclaimed** chunk sheds nothing, so the budget's despawn takes the chunk,
/// its collider and its rubble out together — one event, not three.
///
/// # The bound, stated exactly
///
/// The payload is memoized against the actor's fracture **generation**
/// ([`inf_render::DebrisCache`]), so the CPU pack happens when the live set
/// changes rather than every frame. The GPU upload follows the same cadence, and
/// that is once per *generation bump* — not, as an earlier version of this
/// comment claimed, once per break: a generation moves on every detach AND every
/// reclaim, so a collapsing actor uploads once per chunk that comes off. The
/// bound is `<= 2 x chunk_count` uploads per actor per session.
fn project_debris(
    state: &inf_physics::d3::FractureState,
    material: Option<&Material>,
    id: u64,
    cache: &mut inf_render::DebrisCache,
) -> Option<inf_render::ScatterBatch> {
    if state.is_intact() {
        return None;
    }
    let (color, roughness) = match material {
        Some(m) => (m.base_color.to_array(), m.roughness),
        None => ([0.8, 0.8, 0.8, 1.0], 0.5_f32),
    };
    let sites: Vec<inf_render::DebrisSite> = state
        .chunks()
        .iter()
        .enumerate()
        .filter(|(_, c)| c.detached && !c.gone)
        .map(|(i, c)| inf_render::DebrisSite {
            entity: id,
            chunk: i as u32,
            order: c.detach_order,
            center: state.chunk_rest_center(i),
            radius_m: state.chunk_radius_m(i),
        })
        .collect();
    cache.batch(
        id,
        state.generation(),
        &sites,
        inf_render::DEBRIS_RUBBLE_PER_CHUNK,
        color,
        roughness,
    )
}

/// **MIRROR** of the other host's `project_voxel` — keep the two byte-identical,
/// **this doc block included**. `projector_mirror`'s `extract_fn` anchors on the
/// real function item and takes the comment above it with it; until the P21.2
/// re-audit it anchored on the first `fn project_voxel(` token in the file, which
/// was a backticked self-reference inside this very comment — so these lines were
/// never compared, and the two copies drifted to 23 lines against 17 with the gate
/// green throughout.
fn project_voxel(
    slot: &inf_voxel::VolumeSlot,
    terrain: Option<&Terrain>,
    translation: DVec3,
    id: u64,
    vt: Option<&inf_render::VtTextures>,
    prev: &mut Vec<inf_render::RenderVoxelVolume>,
) -> Option<inf_render::RenderVoxelVolume> {
    let voxel_size_m = slot.data.voxel_size_m();
    let layers = std::array::from_fn(|k| match terrain {
        Some(t) => RenderTerrainLayer {
            albedo: t.layers[k].albedo.to_array(),
            roughness: t.layers[k].roughness as f32,
            tex_scale: t.layers[k].tex_scale as f32,
            // TER2a: the same binding the heightfield takes, through the same
            // door -- a voxel surface shades off the SAME four terrain layers,
            // so a cave mouth whose rock came from a different place than the
            // cliff above it would be visible as a seam.
            vt: inf_render::vt_set_for(vt, t.layers[k].material.map(|m| m.as_u128())),
        },
        None => RenderTerrainLayer::default(),
    });
    // Hardening Wave E's P2 memo: a volume whose chunk set, stamps, origins and
    // palette are all unchanged is carried forward from `prev` — the previous
    // frame's list — rather than having every resident chunk's vertex stream
    // rebased, mapped and cloned again for a consumer that gates the upload on
    // the very stamps compared here. See `inf_render::take_unchanged_voxel`.
    if let Some(kept) = inf_render::take_unchanged_voxel(
        prev,
        id,
        &layers,
        slot.meshes.meshes().map(|(&key, _)| {
            (
                inf_render::VoxelChunkKey::new(key.x, key.y, key.z),
                slot.data.chunk_origin_world(key) + translation,
                slot.meshes.version(key),
            )
        }),
    ) {
        return Some(kept);
    }
    let chunks: Vec<inf_render::RenderVoxelChunk> = slot
        .meshes
        .meshes()
        .map(|(&key, mesh)| inf_render::RenderVoxelChunk {
            key: inf_render::VoxelChunkKey::new(key.x, key.y, key.z),
            origin: slot.data.chunk_origin_world(key) + translation,
            vertices: mesh
                .local_positions_m(voxel_size_m)
                .into_iter()
                .enumerate()
                .map(|(i, pos)| inf_render::RenderVoxelVertex {
                    pos,
                    normal: mesh.normals[i],
                    material: mesh.materials[i] as u32,
                    seam_nh: inf_render::RenderVoxelVertex::NO_SEAM,
                    seam_albedo: [0.0; 4],
                })
                .collect(),
            indices: mesh.indices.clone(),
            bounds: mesh.local_bounds_m(voxel_size_m),
            version: slot.meshes.version(key),
        })
        .collect();
    if chunks.is_empty() {
        return None;
    }
    Some(inf_render::RenderVoxelVolume {
        id,
        chunks,
        layers,
        seam_band_m: 0.0,
    })
}

/// Project an ECS [`Light2D`] (+ world position) into a renderer 2D light.
fn project_light2d(light: &Light2D, translation: DVec3) -> RenderLight2D {
    let c = light.color.to_array();
    RenderLight2D {
        color: [c[0], c[1], c[2]],
        intensity: light.intensity,
        radius: light.radius,
        position: translation,
    }
}

/// Project an ECS [`NineSlice`] (+ world position) into a prebatched run of nine
/// cell quads centered on the entity. Like [`project_sprite`], the texture GUID
/// maps to a handle but no bytes are uploaded from the viewport thread yet
/// (referenced panels render as the tinted white fallback; the headless golden
/// exercises the textured path).
fn project_nine_slice(nine: &NineSlice, translation: DVec3) -> PrebatchedRun {
    let params = NineSliceParams {
        position: translation,
        pivot: Vec2::splat(0.5),
        size: Vec2::new(nine.size.x as f32, nine.size.y as f32),
        border_uv: [
            nine.border_uv[0] as f32,
            nine.border_uv[1] as f32,
            nine.border_uv[2] as f32,
            nine.border_uv[3] as f32,
        ],
        border_world: Vec2::new(nine.border_world.x as f32, nine.border_world.y as f32),
        color: nine.tint.to_array(),
        texture: nine
            .texture
            .map(|u| handle_from_guid(u.as_u128()))
            .unwrap_or(inf_render::WHITE_TEXTURE),
        sorting_layer: nine.sorting_layer,
        order: nine.order,
    };
    let instances = expand_nine_slice(&params).to_vec();
    PrebatchedRun {
        texture: params.texture,
        sorting_layer: params.sorting_layer,
        order: params.order,
        instances,
    }
}

/// Project an ECS [`Text2D`] (+ world position) into a prebatched run of glyph
/// quads. A `None` font asset resolves to the renderer's built-in 8×8 bitmap
/// font ([`BUILTIN_FONT_TEXTURE`], always uploaded by the sprite pass). Returns
/// `None` when the string produces no glyphs (nothing to draw).
fn project_text(text: &Text2D, translation: DVec3) -> Option<PrebatchedRun> {
    let texture = text
        .font_texture
        .map(|u| handle_from_guid(u.as_u128()))
        .unwrap_or(BUILTIN_FONT_TEXTURE);
    let halign = match text.halign {
        TextAlign::Left => HAlign::Left,
        TextAlign::Center => HAlign::Center,
        TextAlign::Right => HAlign::Right,
    };
    let params = TextParams {
        position: translation,
        text: &text.text,
        glyph_cols: text.glyph_cols,
        glyph_rows: text.glyph_rows,
        first_codepoint: text.first_codepoint,
        glyph_size: Vec2::new(text.glyph_size.x as f32, text.glyph_size.y as f32),
        tracking: text.tracking as f32,
        color: text.tint.to_array(),
        texture,
        sorting_layer: text.sorting_layer,
        order: text.order,
        halign,
    };
    let instances = expand_text(&params);
    if instances.is_empty() {
        return None;
    }
    Some(PrebatchedRun {
        texture,
        sorting_layer: text.sorting_layer,
        order: text.order,
        instances,
    })
}

/// The pointer-driven interaction API (select, hover, gizmo drag). Currently
/// only the Windows input layer (`win32.rs`) calls into it; macOS input is not
/// wired yet (the camera holds its default pose), so on non-Windows these are
/// legitimately unused until the macOS hardware pass drives them.
#[cfg_attr(not(windows), allow(dead_code))]
impl EngineHost {
    /// Set the hovered instance (drives the weak outline). `None` clears it.
    pub fn set_hover(&mut self, view: &RenderView, px: u32, py: u32) {
        // Don't recompute hover mid-drag (keeps the outline stable).
        if self.gizmo_drag.is_some() {
            return;
        }
        self.scene.hovered = self.pick_id(view, px, py);
    }

    /// Pick the entity GUID under the cursor (`None` = empty space). Selection
    /// itself lives in the document — the caller applies the pick to it.
    pub fn pick_guid(&mut self, view: &RenderView, px: u32, py: u32) -> Option<Uuid> {
        let id = self.pick_id(view, px, py)?;
        self.id_to_guid.get(&id).copied()
    }

    /// The render-instance id under a viewport pixel.
    ///
    /// The GPU id-buffer pass rasterizes [`RenderScene::instances`] only — the
    /// rigid primitive path — so P18.3's real geometry (virtualized meshes and
    /// skinned characters, which live in their own scene lists) would be
    /// **unclickable**: the whole point of the batch is that an imported mesh is
    /// as much an object as a cube, and an object you cannot click is not one.
    /// P18.5 put GPU-instanced scatter in exactly the same position — a PCG
    /// volume's cubes and a foliage stroke used to BE `instances`, and moving them
    /// into their own storage-buffer path would have quietly made a whole class of
    /// authored content unselectable.
    ///
    /// Extending the ID pass to a vertex-pulled indirect meshlet draw is a
    /// renderer change and belongs with the selection-outline work (see the
    /// remainder recorded in ROADMAP §12 P18.3). The stopgap is the technique the
    /// gizmo already uses and this codebase already trusts — **analytic
    /// picking**: on an id-buffer miss, ray-test the cursor against each vgeom /
    /// skinned / scattered instance's world bounding sphere and take the nearest
    /// hit.
    ///
    /// It is deliberately a *fallback*, not a first choice: whenever the id buffer
    /// answers, that answer wins, so nothing about picking a primitive changes.
    /// Ties are resolved by distance along the ray and then by id, so the result
    /// is a deterministic function of the scene and the pixel. Its honest
    /// limitation is that a bounding sphere is coarser than the silhouette: a
    /// click just outside a concave mesh can select it.
    fn pick_id(&mut self, view: &RenderView, px: u32, py: u32) -> Option<u32> {
        if let Some(id) = self.picker.pick(&self.gpu, &self.scene, view, px, py) {
            return Some(id);
        }
        if self.scene.vgeom_instances.is_empty()
            && self.scene.skinned.is_empty()
            && self.scene.scatter.is_empty()
        {
            return None;
        }
        let (ro, rd) = view.pixel_ray(px as f32, py as f32);
        let ro_world = self.origin.to_world(ro);
        let rd = rd.as_dvec3();

        let bounds_of: std::collections::BTreeMap<u128, ([f32; 3], f32)> = self
            .scene
            .vgeom_assets
            .iter()
            .map(|a| (a.id, a.bounds()))
            .collect();
        let mut best: Option<(f64, u32)> = None;
        let mut consider = |center: DVec3, radius: f64, id: u32| {
            if let Some(t) = ray_sphere_t(ro_world, rd, center, radius) {
                if best.is_none_or(|(bt, bid)| t < bt || (t == bt && id < bid)) {
                    best = Some((t, id));
                }
            }
        };
        for inst in &self.scene.vgeom_instances {
            let Some((c, r)) = bounds_of.get(&inst.asset).copied() else {
                continue;
            };
            let local = Vec3::from_array(c);
            let center = inst.translation + (inst.rotation * (local * inst.scale)).as_dvec3();
            consider(center, (r * inst.scale.abs().max_element()) as f64, inst.id);
        }
        for inst in &self.scene.skinned {
            // Skinned geometry has no cached bounding sphere on the scene DTO, so
            // the bind-space vertex extent stands in. It is computed from the same
            // buffer the pass draws, so it can never disagree with what is on
            // screen — only with where the *pose* moved it, which is the same
            // approximation the rest of this fallback makes.
            let Some(mesh) = self.scene.skinned_meshes.get(inst.mesh) else {
                continue;
            };
            let r = mesh
                .vertices
                .iter()
                .map(|v| Vec3::from_array(v.pos).length())
                .fold(0.0f32, f32::max);
            consider(
                inst.translation,
                (r * inst.scale.abs().max_element()).max(0.05) as f64,
                inst.id,
            );
        }
        // P18.5 scatter. The instances live in a storage buffer the cull compute
        // reads, so nothing about them is rasterized into the id buffer; the same
        // ray-test keeps a PCG volume and a foliage stroke clickable. Each
        // scattered instance is a primitive at a known scale, so its bound is
        // exact-by-construction (`bounding_radius` × scale) rather than the
        // approximation the vgeom/skinned cases settle for. The batch carries ONE
        // id, so a hit on any instance selects the owning volume / foliage entity —
        // which is also what makes the tie-break by id well-defined here.
        for batch in &self.scene.scatter {
            let r = batch.data.mesh.bounding_radius();
            for inst in &batch.data.instances {
                let center = batch.anchor + Vec3::from_array(inst.offset).as_dvec3();
                consider(center, (r * inst.scale.abs()) as f64, batch.id);
            }
        }
        best.map(|(_, id)| id)
    }

    /// Screen-constant gizmo world size for the current view (perspective uses
    /// distance × fov; ortho uses the zoom half-height).
    fn gizmo_size(&self, view: &RenderView, origin_local: Vec3) -> f32 {
        match view.ortho {
            Some(o) => gizmo::gizmo_world_size_ortho(o.half_height),
            None => gizmo::gizmo_world_size(origin_local, view.eye_local(), self.fov_y),
        }
    }

    /// World-space transforms of the current selection after a gizmo drag, keyed
    /// by GUID — the caller writes them back to the document as one undo entry.
    /// (Local == world for the roots/identity-parent objects the gizmo edits;
    /// full parent-relative solve lands with nested transforms.)
    pub fn selected_world_transforms(&self) -> Vec<(Uuid, EcsTransform)> {
        let mut out: Vec<(Uuid, EcsTransform)> = self
            .scene
            .selected
            .iter()
            .filter_map(|id| {
                let guid = self.id_to_guid.get(id)?;
                let inst = self.instance_xform(*id)?;
                let mut t = EcsTransform::from_translation(inst.translation);
                t.set_quat(inst.rotation.as_dquat());
                t.scale = Vec3d::from_dvec3(inst.scale.as_dvec3());
                Some((*guid, t))
            })
            .collect();
        // Selected 2D (non-mesh) entities the gizmo moved (P8.2c).
        for (guid, s) in &self.selected_2d {
            let mut t = EcsTransform::from_translation(s.translation);
            t.set_quat(s.rotation);
            t.scale = Vec3d::from_dvec3(s.scale);
            out.push((*guid, t));
        }
        out
    }

    pub fn set_gizmo_mode(&mut self, mode: GizmoMode) {
        self.gizmo_mode = mode;
    }

    /// Switch the gizmo orientation frame (World ↔ Local) from the toolbar
    /// (Wave 2).
    pub fn set_gizmo_space(&mut self, space: GizmoSpace) {
        self.gizmo_space = space;
    }

    /// Replace the 3D transform-gizmo snap increments (from the toolbar, Wave 2).
    pub fn set_snap_3d(&mut self, snap: SnapSettings) {
        self.snap_3d = snap;
    }

    /// The active 3D snap settings (read by the Windows input layer during a
    /// gizmo drag).
    #[cfg_attr(not(windows), allow(dead_code))]
    pub fn snap_3d(&self) -> SnapSettings {
        self.snap_3d
    }

    /// The gizmo's orientation basis for the current selection: `IDENTITY` in
    /// World space (or 2D, which is always world-aligned), otherwise the primary
    /// selection's world rotation for Local space. The "primary" is the first
    /// selected mesh instance, else the first selected 2D entity (Wave 2).
    fn gizmo_basis(&self) -> glam::Quat {
        if self.mode == ViewportMode::TwoD || self.gizmo_space == GizmoSpace::World {
            return glam::Quat::IDENTITY;
        }
        if let Some(id) = self.scene.selected.first() {
            if let Some(inst) = self.instance_xform(*id) {
                return inst.rotation;
            }
        }
        if let Some(s) = self.selected_2d.values().next() {
            return s.rotation.as_quat();
        }
        glam::Quat::IDENTITY
    }

    /// World-space point under a viewport pixel for drag-spawn (Wave 2, feature
    /// A). UE-like precedence: the terrain surface under the cursor (if a terrain
    /// exists), else the ground plane `y = 0`, else — looking at the sky /
    /// near-parallel — a fixed 10 m down the ray from the eye. In 2D mode the
    /// point lands on the `z = 0` sprite plane. Deterministic (no randomness).
    pub fn pick_world_point(&self, doc: &SceneDoc, view: &RenderView, px: u32, py: u32) -> DVec3 {
        let (ro, rd) = view.pixel_ray(px as f32, py as f32);
        let ro_w = self.origin.to_world(ro);
        let rd = rd.as_dvec3();
        // 2D editor: intersect the sprite plane z = 0.
        if view.ortho.is_some() {
            if rd.z.abs() > 1e-9 {
                let t = -ro_w.z / rd.z;
                if t.is_finite() {
                    return ro_w + rd * t;
                }
            }
            return ro_w;
        }
        // Terrain surface under the cursor — the NEAREST hit across every
        // projected terrain (P16.6), resolved through `terrain_probes` so a
        // STREAMED terrain answers from the pages the streamer has actually paged
        // in. (Reading the document's own set here was the bug: it is empty for a
        // streamed terrain, so every drop fell through to the ground plane.)
        if let Some(hit) = nearest_terrain_hit(&self.terrain_probes(doc, None), ro_w, rd) {
            return hit.world;
        }
        // Ground plane y = 0 (in front of the eye).
        if rd.y.abs() > 1e-6 {
            let t = -ro_w.y / rd.y;
            if (0.0..1.0e6).contains(&t) {
                return ro_w + rd * t;
            }
        }
        // Sky / near-parallel: place 10 m down the ray.
        ro_w + rd * 10.0
    }

    /// Handle a drag-drop that ended over the viewport (Wave 2, feature A): pick
    /// the world point under the cursor and spawn there as ONE undo step, then
    /// select the new entity. Returns `true` when something was spawned (the
    /// caller emits `WorldChanged`).
    ///
    /// Payload convention lives in [`inf_editor_core::viewport_drop`] (parsed
    /// there so the Linux CI leg exercises it — this module is Windows/macOS
    /// only). In short: `spawn:<kind>` places a primitive; `asset:<kind>:<id>:
    /// <name>` places a Content-Drawer asset.
    ///
    /// **A dropped mesh is bound to the entity** (Wave E). This used to spawn a
    /// placeholder cube for every asset kind and never write `MeshRef::asset`,
    /// on the belief that the viewport thread's lack of an asset DB made it
    /// impossible — but the binding is a GUID, not a lookup, and the payload can
    /// carry the kind. The interactive viewport has resolved that field to real
    /// geometry since P18.3, so from here the prop draws as itself.
    pub fn spawn_drop(
        &self,
        doc: &mut SceneDoc,
        view: &RenderView,
        px: u32,
        py: u32,
        payload: &str,
    ) -> bool {
        use inf_editor_core::viewport_drop::{parse_drop_payload, spawn_asset_entity, DropPayload};

        let parsed = parse_drop_payload(payload);
        let (kind, name) = match parsed {
            DropPayload::Spawn { kind } => match spawn_kind_from_str(kind) {
                Some(k) => (k, ""),
                None => {
                    tracing::warn!("inf-viewport: unknown drop spawn kind '{kind}'");
                    return false;
                }
            },
            // A non-mesh asset keeps the honest placeholder: there is no
            // geometry to bind, and the entity is still real, named and
            // selectable.
            DropPayload::Asset { name, .. } => (SpawnKind::Cube, name),
        };
        let mesh_asset = parsed.mesh_asset();

        let point = self.pick_world_point(doc, view, px, py);
        doc.begin_transaction("Spawn");
        let guid = match parsed {
            DropPayload::Spawn { .. } => doc.edit_create(kind, name, None),
            // An asset goes through the door `scene_spawn_asset` also uses, so
            // the binding rule cannot be true on one drop path and false on the
            // other (Wave E audit, A6 — one door for two paths).
            DropPayload::Asset { .. } => spawn_asset_entity(doc, name, mesh_asset, None),
        };
        doc.edit_set_transform(guid, EcsTransform::from_translation(point));
        doc.select(&[guid], false);
        doc.commit_transaction();
        true
    }

    /// Focus target for the current selection: its center and a radius that
    /// bounds every selected object. `None` when nothing is selected.
    pub fn selection_focus(&self) -> Option<(DVec3, f64)> {
        let center = self.selection_center()?;
        let mut radius: f64 = 1.0;
        for id in &self.scene.selected {
            if let Some(inst) = self.instance_xform(*id) {
                let extent = inst.scale.abs().max_element() as f64;
                radius = radius.max((inst.translation - center).length() + extent);
            }
        }
        for guid in &self.selected_guids {
            // A curve-shaped entity — see `spline_focus` for the river this was
            // written for.
            if let Some((mid, extent)) = self.spline_focus(*guid) {
                radius = radius.max((mid - center).length() + extent);
            }
        }
        for s in self.selected_2d.values() {
            radius = radius.max((s.translation - center).length() + s.extent);
        }
        Some((center, radius))
    }

    /// Where the player starts and how wide a frame it wants — the `Home`
    /// action's target (wave EDIT1, clause 2).
    ///
    /// The radius is [`PLAYER_START_FRAME_M`] and not the pawn's own size: a
    /// character is under two metres, and framing two metres from a mountain
    /// vista puts the author's nose against a shirt. What an author opening a
    /// level wants to see is the character AND the street it stands in.
    pub fn player_start_focus(&self) -> Option<(DVec3, f64)> {
        self.player_start.map(|(p, _)| (p, PLAYER_START_FRAME_M))
    }

    /// Where the 3D camera stands for `Home`: `(eye, yaw, pitch)` — behind the
    /// character at head height, looking the way it faces.
    ///
    /// The shot the game opens on, rather than a framed bounding sphere; see
    /// [`PLAYER_START_EYE_M`] for the roof that made the difference.
    pub fn player_start_pose(&self) -> Option<(DVec3, f32, f32)> {
        let (p, yaw) = self.player_start?;
        let (sy, cy) = (yaw as f64).sin_cos();
        let flat = DVec3::new(sy, 0.0, -cy);
        Some((
            p + DVec3::Y * PLAYER_START_EYE_M - flat * PLAYER_START_BACK_M,
            yaw,
            PLAYER_START_PITCH,
        ))
    }

    /// If the cursor is over a gizmo handle, begin a drag and return true. The
    /// handle set is constrained to the sprite plane in 2D (ortho `view`).
    pub fn try_begin_gizmo(&mut self, view: &RenderView, px: u32, py: u32) -> bool {
        let Some(center) = self.selection_center() else {
            return false;
        };
        let origin_local = self.origin.to_render(center);
        let size = self.gizmo_size(view, origin_local);
        let two_d = view.ortho.is_some();
        let basis = self.gizmo_basis();
        let cursor = Vec2::new(px as f32, py as f32);
        let Some(axis) = gizmo::pick_axis(
            self.gizmo_mode,
            origin_local,
            basis,
            size,
            view.view_proj(),
            cursor,
            view.width as f32,
            view.height as f32,
            two_d,
        ) else {
            return false;
        };
        let (ro, rd) = view.pixel_ray(px as f32, py as f32);
        self.gizmo_drag = Some(GizmoDrag::begin(
            self.gizmo_mode,
            axis,
            basis,
            origin_local,
            ro,
            rd,
        ));
        // Snapshot the selection's transforms at drag start (M2): every frame's
        // cumulative delta is applied to these, not accumulated onto the live
        // instances, so snapping quantizes total displacement.
        self.gizmo_initial.clear();
        for id in self.scene.selected.clone() {
            if let Some(inst) = self.instance_xform(id) {
                self.gizmo_initial.insert(id, inst);
            }
        }
        self.gizmo_initial_2d = self.selected_2d.clone();
        true
    }

    pub fn is_dragging_gizmo(&self) -> bool {
        self.gizmo_drag.is_some()
    }

    /// Apply a gizmo drag update from the cursor. `snap` > 0 quantizes.
    ///
    /// The drag is NOT re-anchored between frames: [`GizmoDrag::update`] measures
    /// the delta from the original grab point, so `delta` is the CUMULATIVE
    /// motion since the gesture began. Snapping therefore quantizes the total
    /// displacement (a slow sub-snap drag holds still until it crosses a snap
    /// boundary, then jumps exactly one step; total motion is always a multiple
    /// of the step). The cumulative delta is applied to the drag-start snapshot
    /// (`gizmo_initial`), never accumulated onto the live instances (M2).
    pub fn update_gizmo(&mut self, view: &RenderView, px: u32, py: u32, snap: f32) {
        let Some(drag) = self.gizmo_drag else {
            return;
        };
        let (ro, rd) = view.pixel_ray(px as f32, py as f32);
        let delta = drag.update(ro, rd, snap);
        self.apply_delta(delta, drag.origin);
    }

    /// Apply the cumulative gizmo `delta` to the drag-start snapshot, writing the
    /// result onto the live selection. `pivot_local` is the gizmo origin at drag
    /// start (render-local) — fixed for the whole gesture so cumulative rotation
    /// orbits about a stable point.
    fn apply_delta(&mut self, delta: GizmoDelta, pivot_local: Vec3) {
        let pivot = self.origin.to_world(pivot_local);
        let selected = self.scene.selected.clone();
        for id in &selected {
            let Some(init) = self.gizmo_initial.get(id).copied() else {
                continue;
            };
            let mut next = init;
            match delta {
                GizmoDelta::Translate(t) => next.translation = init.translation + t,
                GizmoDelta::Rotate { axis, radians } => {
                    let q = glam::Quat::from_axis_angle(axis, radians);
                    next.rotation = q * init.rotation;
                    // Orbit the translation about the pivot too.
                    let rel = (init.translation - pivot).as_vec3();
                    next.translation = pivot + (q * rel).as_dvec3();
                }
                GizmoDelta::Scale(s) => next.scale = init.scale * s,
            }
            self.set_instance_xform(*id, next);
        }
        // Selected 2D (non-mesh) entities move the same way, in f64 (P8.2c).
        for (guid, s) in self.selected_2d.iter_mut() {
            let Some(init) = self.gizmo_initial_2d.get(guid).copied() else {
                continue;
            };
            match delta {
                GizmoDelta::Translate(t) => s.translation = init.translation + t,
                GizmoDelta::Rotate { axis, radians } => {
                    let q = DQuat::from_axis_angle(axis.as_dvec3(), radians as f64);
                    s.rotation = q * init.rotation;
                    let rel = init.translation - pivot;
                    s.translation = pivot + q * rel;
                }
                GizmoDelta::Scale(sc) => s.scale = init.scale * sc.as_dvec3(),
            }
        }
        self.scene.mark_dirty();
    }

    pub fn end_gizmo(&mut self) {
        self.gizmo_drag = None;
        self.gizmo_initial.clear();
        self.gizmo_initial_2d.clear();
    }

    // ── terrain sculpting (P10.2b) ────────────────────────────────────────

    /// The active tool (pick/gizmo vs terrain sculpt).
    pub fn tool_mode(&self) -> ToolMode {
        self.tool_mode
    }

    // ── streamed content: terrain (P16.3b2) + meshes (P18.3) ──────────────

    /// Point the viewport's loose-asset streaming at a project's content root (or
    /// `None` to disable it). Rescans the `.inf_terrain` **and** render-asset
    /// indexes and drops every live stream and opened payload, so a project switch
    /// can never serve the previous project's pages or geometry.
    ///
    /// Until a root is set nothing streams, an asset-backed terrain draws its
    /// (empty) inline data and a `MeshRef.asset` draws its primitive placeholder —
    /// so an editor that never calls this behaves exactly as it did before P16.3b2
    /// / P18.3.
    ///
    /// Pushed from Ring 2's `project://changed` flow: the open project's `Content`
    /// directory, so a `Terrain.asset` authored by the import wizard resolves to a
    /// loose `.inf_terrain` and starts paging (P16.4a) and a `MeshRef.asset`
    /// resolves to its derived `.inf_vmesh` (P18.3). Both *policies* are
    /// unit-tested on all three OSes in `inf_editor_core`; this is only the call
    /// site.
    pub fn set_content_root(&mut self, root: Option<std::path::PathBuf>) {
        self.terrain_streams.set_content_root(root.clone());
        if let Ok(mut v) = self.voxel_volumes.lock() {
            v.set_content_root(root.clone());
        }
        self.render_assets.set_content_root(root);
        self.terrain_slots.clear();
        self.synced_version = None; // force a re-projection
    }

    /// Resolve every `.inf_mesh` this document's scattered instances name into
    /// [`scatter_meshes`](Self::scatter_meshes) — wave TER2b.
    ///
    /// # Why the GUIDs are read off the INSTANCES and not off a component
    ///
    /// Because there is no component that has them. A `PcgVolume` names a
    /// `.inf_pcg`, whose *rules* name meshes, and the viewport thread holds a
    /// document rather than an asset database — the same wall the biome palette
    /// and the water hints are pushed over. What the viewport does have is the
    /// evaluated population, and since wave TER2b every instance in it carries
    /// the GUID its kind resolved to. So the walk is over the answer rather than
    /// over the question.
    ///
    /// **It is O(instances) and it runs per projection**, which is per *document
    /// version* and not per frame. On the island's 16 771 instances that is
    /// 16 771 inserts into a set that never exceeds three entries; the loads
    /// behind it happen once each, because `resolve_scatter_geometry` caches its
    /// misses as well as its hits.
    ///
    /// The table is rebuilt rather than accumulated so that a mesh unbound in one
    /// document does not go on being drawn in the next; the *geometry* behind each
    /// entry is `Arc`-shared with the store, so rebuilding costs pointer copies.
    fn sync_scatter_meshes(&mut self, doc: &SceneDoc) {
        let world = doc.world();
        let w = world.world();
        let mut wanted: BTreeSet<Uuid> = BTreeSet::new();
        for &guid in doc.order() {
            let Some(entity) = world.entity_of(guid) else {
                continue;
            };
            if let Some(vol) = w.get::<PcgVolume>(entity) {
                wanted.extend(vol.evaluated.iter().filter_map(|i| i.mesh));
            }
            if let Some(terrain) = w.get::<Terrain>(entity) {
                wanted.extend(terrain.biome_population.iter().filter_map(|i| i.mesh));
            }
        }
        self.scatter_meshes.clear();
        for &id in &wanted {
            if let Some(g) = self.render_assets.resolve_scatter_geometry(id) {
                self.scatter_meshes.insert(id.as_u128(), g);
            }
        }
        // I8b: the twelve building module families, which name no `.inf_mesh`
        // and cannot be resolved through the asset database. Added AFTER the
        // resolved ones so an authored asset under one of these ids wins --
        // the same order the player's loader uses.
        add_building_modules(&mut self.scatter_meshes);
        self.scatter_wanted = wanted;
    }

    /// **Register the document's bound materials as virtual textures** (P26.4,
    /// clause 0) — the editor half of `PlayerRenderHost::set_material_content`,
    /// and the reason a `.inf_mat`'s maps appear on a surface in the viewport.
    ///
    /// Run once at the top of [`rebuild_scene`](Self::rebuild_scene), before
    /// anything is projected, because the per-instance sets the projection reads
    /// come out of the registry this builds.
    ///
    /// **Gated on an `inf_editor_core::render_assets::VtLevelKey`** — the
    /// binding SET *and* the asset-index generation, as one value — and not on
    /// the document version. A projection runs on every version bump (a gizmo
    /// drag bumps it per input event) and building a VT level creates an atlas
    /// texture and an indirection buffer, so the work must not happen per frame.
    ///
    /// The generation is the second half, and it was missing (P26.4 audit). This
    /// gate read the binding set alone, and its own doc claimed
    /// `refresh_asset_index` "forces a re-projection and clears the set" — it
    /// forces the re-projection and clears nothing: **nothing in the codebase
    /// ever cleared the binding set**. So re-importing a `.inf_tex`, or editing
    /// a material's graph, changed neither the version-independent set nor
    /// anything else this early-out read, and the viewport kept the atlas it
    /// built the first time for the rest of the session.
    ///
    /// **Where the rule is now, and where it is tested** (P26.5). Both terms
    /// live in `EditorRenderAssets::vt_level_key`, in Ring 1, because
    /// `EngineHost::new` takes a real surface and nothing headless constructs
    /// one — so the audit could only pin this early-out as source text.
    /// `inf-editor-core`'s `tests/vt_level_key.rs` executes the whole sequence
    /// on a real device through the real `build_vt_level` and asserts the
    /// rebuilt registry's own descriptor carries a re-imported texture's extent;
    /// `render_assets`'s
    /// `a_levels_bindings_resolve_to_registrable_material_content` pins the
    /// underlying store on the BYTES rather than on the counter. What is pinned
    /// as source here is one call and one whole-value compare
    /// (`projector_mirror`).
    ///
    /// Bindings are collected in **document order** and handed to the registry,
    /// which sorts them (`inf_render::registration_order`). The P26.3 LAW says
    /// the handles this mints differ from the player's, and the sort is what
    /// keeps the *pages* the same anyway.
    ///
    /// (Wave TER2b inserted `sync_scatter_meshes` **between** this block and the
    /// function it documents; the TER2b audit put it back. A doc comment that
    /// slides onto the next function does not warn — it just describes the wrong
    /// thing, and here it described a P26.4 finding on a wave-TER2b walk.)
    fn sync_vt_bindings(&mut self, doc: &SceneDoc) {
        // **The rule is a value, and it lives in Ring 1** (P26.5). The two terms
        // used to be two fields compared inline here, where nothing headless can
        // execute them: `EngineHost::new` takes a real surface, so the P26.4
        // audit could only pin this early-out as SOURCE TEXT. `VtLevelKey` moves
        // the decision to a crate that compiles and tests on every leg, leaves
        // this line as the call, and makes a third term impossible to add
        // without going through the type.
        let key = self.render_assets.vt_level_key(doc);
        if key == self.vt_level_key {
            return;
        }
        self.vt_level_key = key.clone();
        if key.is_empty() {
            self.renderer.set_vt_level(None);
            return;
        }
        let content = self.render_assets.material_content(key.bindings);
        if content.is_empty() {
            self.renderer.set_vt_level(None);
            return;
        }
        // THE DOOR, and it is the player's: same materials, same order, same
        // pool-format ruling, same floor.
        // The budget is the TIER's (P26.5) — the same settings field the player
        // reads, so the two hosts plan the same pool on the same machine.
        let budget = self.renderer.settings().vt.budget_bytes;
        let level = inf_render::build_vt_level(
            &self.gpu.device,
            &self.gpu.queue,
            self.renderer.settings(),
            budget,
            &content.materials,
            |g| content.source(g),
        );
        match level {
            Some((textures, pools, report)) => {
                for a in &report.advisories {
                    tracing::warn!("inf-viewport: virtual textures: {a}");
                }
                // **Every arm, not the first one's format** (wave IASSET2
                // audit). This said `pool_format` — the FIRST arm's — which was
                // the whole answer while a pool had one atlas and reads as the
                // whole answer still: a BC1 + BC5 level logged "Bc1 pages" with
                // half its content in another atlas. `demoted` is here for the
                // same reason it exists at all: it is a cost an author can
                // avoid, and one nothing else surfaces per level.
                tracing::info!(
                    "inf-viewport: {} virtual texture(s) registered for {} bound material(s) \
                     ({:?} page arm(s), {:?} demoted, {} refused)",
                    report.textures,
                    content.materials.len(),
                    report.pool_formats,
                    report.demoted,
                    report.refused
                );
                self.renderer.set_vt_level(Some((textures, pools)));
            }
            None => self.renderer.set_vt_level(None),
        }
    }

    /// Bind, place, page (and release) voxel volumes to match the document — the
    /// `&mut` half of the P21.1/P21.2 path, run once before
    /// [`rebuild_scene`](Self::rebuild_scene) projects anything.
    ///
    /// **MIRROR** of `inf_player::render::PlayerRenderHost::sync_voxels`, and the
    /// two agree on the load-bearing rule: the live set is whatever is **bound**,
    /// never whatever happened to draw. A volume whose chunks mesh to no surface —
    /// all air, all rock, or an asset whose caves are entirely outside the view of
    /// this projection — is still a *loaded* volume, and releasing it because it
    /// produced no triangles means re-reading and re-meshing it from disk on the
    /// very next document bump. A gizmo drag bumps the document per input event.
    ///
    /// Walks `doc.order()` so which volume binds first is a function of the
    /// document rather than of the ECS archetype layout.
    ///
    /// **P21.2 — three acts, not one.** `ensure` binds (parses the payload, indexes
    /// its directory, pages nothing); `place` tells Ring 0 where the entity put the
    /// volume, because residency is a world-space radius and a cave placed a
    /// kilometre from its authoring anchor would otherwise page the chunks nobody
    /// is standing in; `sync_camera` executes the policy. A bound volume with no
    /// `sync_camera` draws nothing, which is why the three are one function on both
    /// hosts and why the mirror gate requires all three calls on both sides.
    fn sync_voxels(&mut self, doc: &SceneDoc) {
        let world = doc.world();
        let w = world.world();
        let mut live: Vec<Uuid> = Vec::new();
        for &guid in doc.order() {
            let Some(entity) = world.entity_of(guid) else {
                continue;
            };
            let Some(volume) = w.get::<VoxelVolume>(entity).copied() else {
                continue;
            };
            let Ok(mut volumes) = self.voxel_volumes.lock() else {
                continue;
            };
            if volumes.ensure(guid, &volume) {
                let translation = w
                    .get::<GlobalTransform>(entity)
                    .map(|g| g.translation())
                    .unwrap_or(DVec3::ZERO);
                volumes.place(guid, translation);
                live.push(guid);
            }
        }
        // Release every volume that is no longer bound (its entity was deleted, its
        // reference was cleared, or a File ▸ Open replaced the document). A loaded
        // volume holds its whole decoded chunk set AND its meshed surface — real
        // megabytes — so this is not bookkeeping.
        let Ok(mut volumes) = self.voxel_volumes.lock() else {
            return;
        };
        volumes.retain_only(live);
        // THE CAMERA-DRIVEN RESIDENCY PASS (P21.2). The eye is the *editor* camera,
        // which the simulation has no reference to: `terrain.height_at` reads the
        // sim's own volume map, seeded from sim state alone, so a fly-through can
        // never change a gameplay answer. Same determinism seam as `sync_render`
        // for terrain, and it is the absence of a path rather than a convention.
        volumes.sync_camera(
            self.last_eye_world,
            &inf_voxel::VoxelWantsParams::default(),
            inf_voxel::VoxelStreamBudget::default(),
        );
    }

    /// Rebuild the loose-asset indexes after the content database changed.
    ///
    /// Pushed from Ring 2 when an import finishes or the watcher sees an external
    /// edit: an index built when the project opened does not contain assets
    /// written after it, so a freshly imported terrain or mesh would resolve to
    /// nothing and the entity the user just spawned would draw empty (P16.4a).
    ///
    /// The two halves treat live state differently, deliberately. Terrain streams
    /// are **kept** (re-pointing the root would re-page terrain the user is flying
    /// over, and a terrain's tiles are re-read per page anyway). Opened
    /// `.inf_vmesh` payloads are **dropped**: a vmesh is opened once and then only
    /// sliced, so a payload rewritten under the same GUID would otherwise be served
    /// from the stale mapping forever. Re-opening costs a header + page-directory
    /// parse each.
    pub fn refresh_asset_index(&mut self) {
        self.terrain_streams.refresh_index();
        if let Ok(mut v) = self.voxel_volumes.lock() {
            v.refresh_index();
        }
        self.render_assets.refresh_index();
        self.synced_version = None; // force a re-projection
    }

    /// The slot for a projected terrain, if it is one (P16.6).
    fn terrain_slot(&self, guid: Uuid) -> Option<&TerrainSlot> {
        self.terrain_slots.iter().find(|s| s.guid == guid)
    }

    /// The slot the terrain tools currently target — the terrain under the cursor
    /// at the last pick, else the first projected one (P16.6).
    fn active_terrain_slot(&self) -> Option<&TerrainSlot> {
        self.terrain_guid
            .and_then(|g| self.terrain_slot(g))
            .or(self.terrain_slots.first())
    }

    /// Whether the terrain the tools are aimed at streams from a `.inf_terrain`
    /// asset.
    ///
    /// Polled each frame by the platform loop and published on
    /// `viewport://tool-status`, where the shell uses it to label the terrain.
    /// As of P16.4b it no longer greys the brush tools out — see
    /// [`terrain_is_editable`](Self::terrain_is_editable), which does.
    ///
    /// P16.6: with several terrains projected this describes the **targeted** one
    /// (the cursor's, else the first) rather than "the" terrain — which is what
    /// the status bar has to say, since it is the terrain a stroke would land on.
    pub fn terrain_is_streamed(&self) -> bool {
        self.active_terrain_slot().is_some_and(|s| s.streamed)
    }

    /// Whether the targeted **streamed** terrain can be sculpted/painted, i.e.
    /// its `.inf_terrain` is a writable file the save path can fold edits into.
    ///
    /// `false` for an inline terrain (which is always editable and needs no
    /// asset) — read it together with [`terrain_is_streamed`](Self::terrain_is_streamed):
    /// *streamed && !editable* is the one case the tools refuse.
    pub fn terrain_is_editable(&self) -> bool {
        self.active_terrain_slot().is_some_and(|s| s.editable)
    }

    /// Whether **any** projected terrain carries tiles not yet written back to its
    /// `.inf_terrain` — the toolbar's "unsaved terrain edits" chip.
    ///
    /// Deliberately the aggregate, not the targeted terrain's: the chip warns that
    /// Ctrl+S has work to do, and a stroke on terrain A must not stop reading as
    /// unsaved because the cursor has since drifted over terrain B.
    pub fn terrain_has_unsaved_edits(&self) -> bool {
        self.terrain_slots.iter().any(|s| s.unsaved)
    }

    /// Release every terrain stream — its resident pages, its edit pins, and its
    /// `.inf_terrain` payload — **and** every opened render asset (P18.3): the
    /// `.inf_vmesh` mappings and decoded skinned geometry the previous level
    /// referenced.
    ///
    /// Pushed by `File ▸ Open` / `File ▸ New` (P16.4b audit): those replace the
    /// document wholesale, so every terrain stream is keyed on entity GUIDs that no
    /// longer exist. Without this the old document's payload and any tile it pinned
    /// for an unsaved edit stay alive for the life of the process. Render assets
    /// are keyed on *asset* GUIDs rather than entity ones, so nothing there is
    /// invalidated by the swap — but everything it holds belongs to the outgoing
    /// level's working set, and the incoming projection's `retain_only` would only
    /// free it after the first frame that already paid to keep it.
    pub fn clear_streams(&mut self) {
        self.terrain_streams.clear();
        self.render_assets.clear();
        self.terrain_slots.clear();
        self.terrain_guid = None;
        // P21.2: the "already warned about unsaveable holes" ledger is keyed on
        // the PREVIOUS document's entity GUIDs, so a level opened after one that
        // carried them would inherit the suppression and never say a word.
        self.voxel_hole_warned.clear();
        self.synced_version = None; // force a re-projection against the new doc
    }

    /// Refresh the cold store of every live terrain stream **in place** — called
    /// after a save rewrote the `.inf_terrain` files.
    ///
    /// Live streams keep their resident pages and their published cut, so saving
    /// does not blink the terrain the user is looking at; only the bytes behind
    /// them, the catalog, and the edit pins change. See
    /// `EditorTerrainStreams::reload_store`.
    pub fn reload_terrain_stores(&mut self) {
        self.terrain_streams.reload_stores();
        for slot in &mut self.terrain_slots {
            slot.unsaved = false;
        }
        self.synced_version = None; // re-project from the refreshed store
    }

    /// Terrain-streaming counters, for the diagnostics path.
    pub fn terrain_stream_stats(&self) -> &inf_terrain::TerrainStreamStats {
        self.terrain_streams.stats()
    }

    /// Take the last tool-rejection message (e.g. a sculpt stroke refused on a
    /// streamed terrain), leaving none.
    ///
    /// The **status seam**: drained once per frame by the platform loop and
    /// emitted as [`ViewportEvent::ToolStatus`], which Ring 2 forwards on
    /// `viewport://tool-status` and the shell shows in the status bar. It is also
    /// still in the Output Log via `tracing`, where every other host-side
    /// diagnostic surfaces.
    pub fn take_tool_status(&mut self) -> Option<String> {
        self.tool_status.take()
    }

    /// Record a tool rejection: remember it for the caller and log it once.
    fn reject_tool(&mut self, message: &str) {
        if self.tool_status.as_deref() != Some(message) {
            tracing::warn!("inf-viewport: {message}");
        }
        self.tool_status = Some(message.to_string());
    }

    /// Report a tool **readout** on the same seam without logging it (P20.4).
    ///
    /// The status channel carries two kinds of message: a *rejection* (something
    /// was refused — worth a line in the Output Log) and a *readout* (a live
    /// measurement, e.g. the lake drag's coverage and depth, which changes every
    /// frame and would drown the log). `reject_tool` is the first; this is the
    /// second, and the split exists so the second can exist at all.
    fn report_tool(&mut self, message: String) {
        self.tool_status = Some(message);
    }

    /// Advance every voxel volume's camera-driven residency and, when it changed,
    /// force a re-projection (P21.2).
    ///
    /// The editor-local half of what the shipped player gets for free: the player
    /// projects its whole world every frame, so its one `sync_voxels` call is
    /// enough. `sync_from_doc` is **version-gated** and the camera does not bump
    /// the document version — nor should it — so without this a chunk that paged in
    /// while flying would sit in the store, meshed, and never reach the scene.
    /// Exactly the gap `sync_streamed_terrain` above exists to close, for exactly
    /// the same reason.
    ///
    /// The re-projection is coarser than terrain's, deliberately: a terrain slot is
    /// index-aligned with `scene.terrains` and can be refreshed in place, while a
    /// voxel volume's projection also needs the `Terrain` palette and the transform
    /// on its own entity — i.e. the document walk. Rebuilding the walk is one
    /// projection, and only on a pass that actually moved a chunk (hysteresis makes
    /// that intermittent even under a flying camera), which is cheaper than keeping
    /// a second copy of the branch in step with the first.
    ///
    /// [`sync_voxels`](Self::sync_voxels) calls the same store method once more, on
    /// the document path, and that overlap is deliberate: a volume bound by a
    /// document change pages in the same pass rather than a frame later, and the
    /// second call is a no-op sync (one box-distance test per available chunk, then
    /// a mesh-cache walk that rebuilds nothing) — which is exactly the property
    /// `a_no_op_sync_moves_no_mesh_stamp` pins in Ring 0.
    fn sync_streamed_voxels(&mut self) {
        let Ok(mut volumes) = self.voxel_volumes.lock() else {
            return;
        };
        let report = volumes.sync_camera(
            self.last_eye_world,
            &inf_voxel::VoxelWantsParams::default(),
            inf_voxel::VoxelStreamBudget::default(),
        );
        drop(volumes);
        if report.is_noop() {
            return;
        }
        self.synced_version = None;
    }

    /// Advance the streamed terrain's camera-driven cut and, when it changed,
    /// re-project the render terrain from the streamer's working set.
    ///
    /// Re-projecting here (rather than waiting for a document change) is what lets
    /// pages appear as the camera flies: `sync_from_doc` is version-gated and the
    /// camera does not bump the document version — nor should it.
    fn sync_streamed_terrain(&mut self) {
        if !self.terrain_streams.sync_render(self.last_eye_world) {
            return;
        }
        // Streaming diagnostics on the existing debug path (`tracing` → the Output
        // Log / log file), throttled so a flying camera doesn't flood it. No new
        // panel, no new IPC channel.
        self.stream_log_countdown = self.stream_log_countdown.saturating_sub(1);
        if self.stream_log_countdown == 0 {
            self.stream_log_countdown = STREAM_LOG_INTERVAL_FRAMES;
            tracing::info!("inf-viewport: {}", self.terrain_stream_stats().summary());
        }
        // P16.6: every streamed terrain advances its own cut, so re-project each
        // of them into its own slot (`terrain_slots` is index-aligned with
        // `scene.terrains`, so no lookup and no reordering).
        for i in 0..self.terrain_slots.len() {
            let slot_guid = self.terrain_slots[i].guid;
            if !self.terrain_slots[i].streamed {
                continue;
            }
            // Hardening Wave E: hand the slot's CURRENT projection to
            // `project_terrain` as the carry source. A camera step moves the cut
            // by a page or two and leaves every other resident tile stamped
            // exactly as it was, so this re-projection now copies only what
            // actually paged — it used to deep-copy the whole resident set on
            // every frame the cut advanced.
            let mut prev: Vec<RenderTerrain> = self
                .scene
                .terrains
                .get_mut(i)
                .map(std::mem::take)
                .into_iter()
                .collect();
            let mut projected = None;
            if let Some((component, data, translation)) =
                self.terrain_streams.projection_inputs(slot_guid)
            {
                if data.tile_count() + data.coarse_tile_count() > 0 {
                    let palette: &[[f32; 4]] = self
                        .biome_palettes
                        .get(&slot_guid)
                        .map(|p| p.as_slice())
                        .unwrap_or(&[]);
                    projected = Some(project_terrain(
                        slot_guid,
                        component,
                        data,
                        translation,
                        palette,
                        self.renderer.vt_textures(),
                        &mut prev,
                    ));
                }
            }
            if let Some(dst) = self.scene.terrains.get_mut(i) {
                // The new projection, or — when there was nothing to project —
                // the one that was taken out, put straight back.
                *dst = projected.or_else(|| prev.pop()).unwrap_or_default();
            }
        }
        // **Round-2 finding B9.** `sync_render` said the cut moved, so heights
        // under any neighbouring voxel volume's seam band may have changed.
        // `synced_version` forces the projection to run at all (the camera does
        // not bump the document version); `seam_dirty` makes that run recompute
        // the seam rather than carry the list it has just written itself.
        self.synced_version = None;
        self.seam_dirty = true;
    }

    /// `true` while a sculpt stroke is in progress.
    pub fn is_sculpting(&self) -> bool {
        self.sculpt_drag.is_some()
    }

    /// The heightfield the cursor is actually looking at, plus the terrain's world
    /// translation.
    ///
    /// For an **inline** terrain that is the document's own `TerrainData`. For a
    /// **streamed** one (P16.4b) it is the streamer's render working set — the
    /// surface being drawn, with the document's unsaved edits already mirrored in
    /// (`overlay_document_edits`). Raycasting the document's set instead would
    /// find nothing until something had already been sculpted, which is
    /// unusable: you cannot click ground the document has not paged in yet, and
    /// paging is what a click is *for*.
    ///
    /// The consequence, stated plainly: a stroke can only start where a level-0
    /// page is resident — i.e. within the render cut's fine ring around the
    /// camera. Aiming at distant terrain that is only covered by a coarse page
    /// finds no hit and starts no stroke. Fly closer; the ring follows.
    fn terrain_probe<'a>(
        &'a self,
        doc: &'a SceneDoc,
        guid: Uuid,
    ) -> Option<(&'a inf_terrain::TerrainData, DVec3)> {
        if self.terrain_slot(guid).is_some_and(|s| s.streamed) {
            if let Some((_, data, translation)) = self.terrain_streams.projection_inputs(guid) {
                return Some((data, translation));
            }
        }
        doc.terrain_data_and_origin(guid)
    }

    /// A [`TerrainProbe`] per projected terrain, in document order — **the one
    /// place** the "which heightfield is under the cursor?" choice is made
    /// (P16.6).
    ///
    /// Every terrain-resolving path (sculpt, paint, drag-drop spawn, foliage) goes
    /// through here, so none of them can drift back to reading the document's own
    /// `TerrainData` — which is *empty by design* for a streamed terrain and would
    /// silently drop every cursor onto the `y = 0` ground plane.
    ///
    /// `restrict` narrows to a single terrain (a stroke in progress; see
    /// [`terrain_pick`](Self::terrain_pick)).
    fn terrain_probes<'a>(
        &'a self,
        doc: &'a SceneDoc,
        restrict: Option<Uuid>,
    ) -> Vec<TerrainProbe<'a>> {
        terrain_probes_of(&self.terrain_slots, restrict, |guid| {
            self.terrain_probe(doc, guid)
        })
    }

    /// Raycast the cursor against **every** projected terrain and return the
    /// NEAREST hit (P16.6): the terrain entity, the hit centre in that terrain's
    /// local XZ, and the local surface height there.
    ///
    /// Reuses the same screen→world ray as picking/gizmo drags, rebased through
    /// the floating origin and shifted into each terrain entity's local frame by
    /// [`nearest_terrain_hit`], which is where the rule (and its tie-break) lives.
    fn sculpt_pick(
        &self,
        doc: &SceneDoc,
        view: &RenderView,
        px: u32,
        py: u32,
    ) -> Option<(Uuid, DVec2, f64)> {
        self.terrain_pick(doc, view, px, py, None)
    }

    /// [`sculpt_pick`](Self::sculpt_pick), optionally **restricted to one
    /// terrain**.
    ///
    /// A stroke in progress restricts to the terrain it started on: dragging the
    /// cursor over a neighbouring terrain must not silently move the brush onto
    /// it (the dabs would land in a different document entity, and the single
    /// `HeightDelta` the stroke commits belongs to exactly one terrain).
    fn terrain_pick(
        &self,
        doc: &SceneDoc,
        view: &RenderView,
        px: u32,
        py: u32,
        restrict: Option<Uuid>,
    ) -> Option<(Uuid, DVec2, f64)> {
        let (ro, rd) = view.pixel_ray(px as f32, py as f32);
        let ro_w = self.origin.to_world(ro);
        let probes = self.terrain_probes(doc, restrict);
        nearest_terrain_hit(&probes, ro_w, rd.as_dvec3())
            .map(|h| (h.guid, h.local_xz, h.local_height))
    }

    /// Page the level-0 tiles a dab at `center` needs into the **document's**
    /// working set, synchronously, before the dab runs (P16.4b).
    ///
    /// A no-op for an inline terrain, whose tiles are all already there. See
    /// `inf_editor_core::terrain_edit` for why the document — and not this
    /// host's streamer — owns the tiles a brush writes.
    fn page_brush_footprint(&mut self, doc: &mut SceneDoc, guid: Uuid, center: DVec2) {
        if !self.terrain_slot(guid).is_some_and(|s| s.streamed) {
            return;
        }
        self.terrain_streams
            .page_brush_footprint(guid, doc, center, self.brush_radius());
    }

    /// The radius of whichever terrain brush is active — the biome tool has its
    /// own, and residency paging / the hover ring must follow the tool the user
    /// is actually holding, not the sculpt slider.
    fn brush_radius(&self) -> f64 {
        if self.tool_mode == ToolMode::Biome {
            self.biome.radius
        } else {
            self.sculpt.radius
        }
    }

    /// The ring colour for the biome tool: the **selected** biome's palette
    /// entry, or the *unassigned* grey when nothing resolves — no set bound, an
    /// id the set no longer defines, or the eraser (id `0`), whose palette slot
    /// **is** that grey. The same fallback the shader applies, so the ring under
    /// the cursor never promises a colour the terrain will not take.
    ///
    /// It follows the toolbar selection, not the Ctrl modifier: Ctrl is a
    /// momentary flip read at mouse-down, and recolouring the hover ring on a
    /// keypress would make the brush look like it had changed tools.
    fn biome_swatch(&self, guid: Option<Uuid>) -> [f32; 4] {
        let id = self.biome.biome as usize;
        guid.and_then(|g| self.biome_palettes.get(&g))
            .and_then(|p| p.get(id))
            .copied()
            .unwrap_or(inf_terrain::UNASSIGNED_BIOME_COLOR)
    }

    /// Mirror the document's edited tiles into the render set and refresh the
    /// unsaved-edits flag — run after every dab so the stroke is visible as it is
    /// made and the status chip lights up on the first sample changed.
    fn after_terrain_edit(&mut self, doc: &SceneDoc, guid: Uuid) {
        let Some(index) = self.terrain_slots.iter().position(|s| s.guid == guid) else {
            return;
        };
        if !self.terrain_slots[index].streamed {
            return;
        }
        self.terrain_streams.overlay_document_edits(guid, doc);
        self.terrain_slots[index].unsaved = !doc.terrain_dirty_tiles(guid).is_empty();
        // Hardening Wave E: as in `sync_streamed_terrain`, the slot's current
        // projection is the carry source. A sculpt dab stamps the handful of
        // tiles the brush touched and leaves the rest alone, so what this
        // re-projection rebuilds is now the brush's footprint rather than the
        // whole resident set — on every dab of every stroke.
        let mut prev: Vec<RenderTerrain> = self
            .scene
            .terrains
            .get_mut(index)
            .map(std::mem::take)
            .into_iter()
            .collect();
        let mut projected = None;
        if let Some((component, data, translation)) = self.terrain_streams.projection_inputs(guid) {
            if data.tile_count() + data.coarse_tile_count() > 0 {
                let palette: &[[f32; 4]] = self
                    .biome_palettes
                    .get(&guid)
                    .map(|p| p.as_slice())
                    .unwrap_or(&[]);
                projected = Some(project_terrain(
                    guid,
                    component,
                    data,
                    translation,
                    palette,
                    self.renderer.vt_textures(),
                    &mut prev,
                ));
            }
        }
        if let Some(dst) = self.scene.terrains.get_mut(index) {
            *dst = projected.or_else(|| prev.pop()).unwrap_or_default();
        }
        // **Round-2 finding B9**, the other out-of-band writer: a sculpt dab
        // moves ground a neighbouring cave mouth blends against. Same pair of
        // invalidations, same reason — see `seam_dirty`.
        self.synced_version = None;
        self.seam_dirty = true;
    }

    /// Rebuild the brush-ring loop points around `center` (terrain-local XZ on
    /// `guid`), coloured by the active op. Clears the ring if the terrain vanished.
    ///
    /// P16.6: the terrain is passed in rather than read off "the" terrain field —
    /// the ring must follow the surface the cursor actually resolved to.
    fn refresh_ring(&mut self, doc: &SceneDoc, guid: Option<Uuid>, center: DVec2) {
        let op = self
            .sculpt_drag
            .as_ref()
            .map(|d| d.op)
            .unwrap_or(self.sculpt.op);
        // Paint recolours the ring by the target layer's albedo (so the swatch
        // under the cursor reads as the layer being painted); the biome tool by
        // its target biome's palette colour, for the same reason; sculpt ops use
        // their fixed op colour.
        let color = if self.tool_mode == ToolMode::Biome {
            self.biome_swatch(guid)
        } else if op == SculptOp::Paint {
            guid.and_then(|g| doc.terrain_layer_albedo(g, self.sculpt.paint_layer))
                .unwrap_or_else(|| op_color(op))
        } else {
            op_color(op)
        };
        self.sculpt_ring_color = color;
        if let Some(guid) = guid {
            if let Some((data, translation)) = self.terrain_probe(doc, guid) {
                let ring = build_ring(data, translation, center, self.brush_radius());
                self.sculpt_ring = ring;
                return;
            }
        }
        self.sculpt_ring.clear();
    }

    /// Update the hovered brush ring (idle Sculpt mode): raycast the cursor and
    /// rebuild the ring, or clear it off-terrain.
    ///
    /// P16.6: the pick resolves which terrain is under the cursor, so hovering
    /// also **retargets the tools** — the editable/read-only decision below is
    /// then made against that terrain, not against whichever one came first.
    pub fn update_sculpt_hover(&mut self, doc: &SceneDoc, view: &RenderView, px: u32, py: u32) {
        let hit = self.sculpt_pick(doc, view, px, py);
        if let Some((guid, _, _)) = hit {
            self.terrain_guid = Some(guid);
        }
        // A streamed terrain whose asset cannot be written has nowhere to save a
        // stroke to, so showing an inviting ring would be a lie (P16.4b — an
        // editable streamed terrain rings exactly like an inline one).
        if self.terrain_is_streamed() && !self.terrain_is_editable() {
            self.sculpt_ring.clear();
            return;
        }
        match hit {
            Some((guid, center, _)) => self.refresh_ring(doc, Some(guid), center),
            None => self.sculpt_ring.clear(),
        }
    }

    /// Begin a sculpt stroke under the cursor. Raycasts the terrain; on a hit,
    /// opens a [`Stroke`], lays the first dab, and returns `true`. `ctrl` flips
    /// Raise↔Lower for a temporary inverse brush (UE convention).
    pub fn begin_sculpt(
        &mut self,
        doc: &mut SceneDoc,
        view: &RenderView,
        px: u32,
        py: u32,
        ctrl: bool,
    ) -> bool {
        // P16.6: resolve which terrain the cursor is on FIRST, then judge that
        // terrain. Refusing on "the" terrain's writability while the click lands on
        // a different one is exactly the class of bug multi-terrain introduces.
        let hit = self.sculpt_pick(doc, view, px, py);
        if let Some((guid, _, _)) = hit {
            self.terrain_guid = Some(guid);
        }
        // P16.4b: a streamed terrain IS editable — its tiles page into the
        // document on demand and the save path writes them back. The only refusal
        // left is the honest one: an asset the editor cannot write, where a stroke
        // would be lost at Ctrl+S rather than saved.
        if self.terrain_is_streamed() && !self.terrain_is_editable() {
            self.reject_tool(inf_editor_core::terrain_stream::STREAMED_TERRAIN_READONLY_REJECTION);
            return false;
        }
        let Some((guid, center, height)) = hit else {
            // A miss has two very different causes on a streamed terrain. If some
            // streamed asset really has ground under the cursor, it is simply
            // paged at coarse detail and there is no level-0 page to sculpt — say
            // so (P16.4b audit: a silent no-op reads as a broken tool). Clicking
            // past the edge of every terrain is not a problem and stays silent.
            let p = self.pick_world_point(doc, view, px, py);
            for i in 0..self.terrain_slots.len() {
                let g = self.terrain_slots[i].guid;
                if !self.terrain_slots[i].streamed {
                    continue;
                }
                let local = self
                    .terrain_streams
                    .projection_inputs(g)
                    .map(|(_, _, t)| DVec2::new(p.x - t.x, p.z - t.z))
                    .unwrap_or(DVec2::new(p.x, p.z));
                if self.terrain_streams.covers_level0(g, local) {
                    self.reject_tool(
                        inf_editor_core::terrain_stream::STREAMED_TERRAIN_COARSE_REJECTION,
                    );
                    break;
                }
            }
            return false;
        };
        // Make the footprint resident in the DOCUMENT before the first dab — the
        // brush must never author over ground it has not actually read.
        self.page_brush_footprint(doc, guid, center);
        let op = effective_op(self.sculpt.op, ctrl);
        let settings = self.sculpt;
        let kind = if self.tool_mode == ToolMode::Biome {
            // Ctrl is the eraser modifier — the biome twin of Ctrl flipping
            // Raise↔Lower, and the only way to unpaint without hunting for the
            // "Unassigned" entry in the picker.
            let mut stroke = BiomeStroke::begin(effective_biome(&self.biome, ctrl));
            doc.biome_apply_dab(guid, &mut stroke, biome_params(&self.biome, center));
            DragStroke::Biome(stroke)
        } else if op == SculptOp::Paint {
            let mut stroke = SplatStroke::begin(settings.paint_layer);
            doc.paint_apply_dab(guid, &mut stroke, paint_params(&settings, center));
            DragStroke::Splat(stroke)
        } else {
            let mut stroke = Stroke::begin();
            let (brush, params) = brush_of(op, &settings, center, height);
            doc.sculpt_apply_dab(guid, &mut stroke, brush, params);
            DragStroke::Height(stroke)
        };
        self.after_terrain_edit(doc, guid);
        self.sculpt_drag = Some(SculptDrag {
            guid,
            kind,
            op,
            last_local: center,
            flatten_height: height,
        });
        self.refresh_ring(doc, Some(guid), center);
        true
    }

    /// Continue the stroke: resample the path from the last dab to the cursor at
    /// even spacing (~⅓ radius) and lay a dab at each, mutating the live terrain
    /// (which re-uploads next frame via the version bump).
    pub fn update_sculpt(&mut self, doc: &mut SceneDoc, view: &RenderView, px: u32, py: u32) {
        let Some(drag) = self.sculpt_drag.as_ref() else {
            return;
        };
        let (guid, last, op, flatten_h) =
            (drag.guid, drag.last_local, drag.op, drag.flatten_height);
        // Restricted to the stroke's own terrain (P16.6): sliding over a
        // neighbouring terrain holds the stroke exactly as sliding off the world
        // does, rather than teleporting the brush onto different ground.
        let Some((_, cur, _)) = self.terrain_pick(doc, view, px, py, Some(guid)) else {
            return; // cursor slid off the terrain — hold the stroke, add nothing
        };
        let settings = self.sculpt;
        let biome = self.biome;
        let spacing = (0.35 * self.brush_radius()).max(0.05);
        // `dab_positions` re-emits the start (`last`); skip it — already placed.
        //
        // Through the Ring-1 wrapper, and **capped** (P21.3 audit): the terrain
        // brushes had no per-frame bound at all — only the carve brush did — so
        // a drag whose pick landed far away built the whole list every frame and
        // laid every dab in it. The remainder rides to the next frame from the
        // last dab actually placed, exactly as the carve brush's does.
        let dabs = inf_editor_core::voxel_tool::dab_centers_2d_capped(
            &[last, cur],
            spacing,
            Self::MAX_DABS_PER_UPDATE,
        );
        let mut new_last = last;
        for &c in dabs.iter().skip(1) {
            // Every dab pages its own footprint: a drag walks across tiles, and a
            // dab must never write ground it has not read (P16.4b).
            self.page_brush_footprint(doc, guid, c);
            if let Some(d) = self.sculpt_drag.as_mut() {
                match &mut d.kind {
                    DragStroke::Height(stroke) => {
                        let (brush, params) = brush_of(op, &settings, c, flatten_h);
                        doc.sculpt_apply_dab(guid, stroke, brush, params);
                    }
                    DragStroke::Splat(stroke) => {
                        doc.paint_apply_dab(guid, stroke, paint_params(&settings, c));
                    }
                    DragStroke::Biome(stroke) => {
                        doc.biome_apply_dab(guid, stroke, biome_params(&biome, c));
                    }
                }
            }
            new_last = c;
        }
        if let Some(d) = self.sculpt_drag.as_mut() {
            d.last_local = new_last;
        }
        self.after_terrain_edit(doc, guid);
        self.refresh_ring(doc, Some(guid), cur);
    }

    /// **Close a terrain-brush stroke the active tool can no longer finish**
    /// (P21.3, the P21.2 audit's N2 ledger item). Returns `true` when an orphaned
    /// stroke was recorded.
    ///
    /// [`settle_orphaned_carve`](Self::settle_orphaned_carve)'s sibling, and the
    /// identical bug one tool over. A [`DragStroke`] — **height sculpt, splat
    /// paint or biome paint**, all three — mutates the terrain *live* per dab,
    /// and only [`finish_sculpt`](Self::finish_sculpt) turns those dabs into the
    /// undo entry that describes them. `finish_sculpt` is reached from the pump's
    /// `sculpting` branch, which is gated on the **active tool** — so a tool
    /// switch arriving between two frames of a drag (the toolbar and the `tool.*`
    /// shortcuts both push one down a command channel, mid-gesture or not) left
    /// the stroke open forever: its edits stayed in the document, saved like any
    /// other edit, and Ctrl+Z could not reach them. That is the un-undoable
    /// committed edit `a4e5844` ruled worse than any partial one.
    ///
    /// Committing rather than reverting, for the carve's reason: the author *did*
    /// move that ground and can see it. One undo step, exactly as a mouse-up
    /// would have produced.
    ///
    /// Called unconditionally by the platform pump with the document in hand —
    /// `set_tool_mode` cannot do it itself (committing needs a `&mut SceneDoc`
    /// and that seam has none), and a deferred flag would be a second piece of
    /// state saying what `tool_mode` and `sculpt_drag` already say.
    pub fn settle_orphaned_sculpt(&mut self, doc: &mut SceneDoc) -> bool {
        if self.sculpt_drag.is_none() {
            return false;
        }
        // The pump's own `sculpting` flag, derived here so the two cannot
        // disagree: 2D mode keeps Select regardless of the tool, so switching
        // projections mid-drag strands a stroke exactly as switching tools does.
        // The Paint sub-mode rides `ToolMode::Sculpt`, so the two names below
        // cover all three `DragStroke` kinds.
        let sculpt_branch_runs = matches!(self.tool_mode, ToolMode::Sculpt | ToolMode::Biome)
            && self.mode != ViewportMode::TwoD;
        if sculpt_branch_runs {
            return false;
        }
        let recorded = self.finish_sculpt(doc);
        if recorded {
            self.reject_tool(Self::SCULPT_STROKE_SETTLED_ON_TOOL_SWITCH);
        }
        recorded
    }

    /// What the author is told when a tool switch closed their brush stroke.
    const SCULPT_STROKE_SETTLED_ON_TOOL_SWITCH: &'static str =
        "Brush: the tool changed while a stroke was still down, so the edit so far was committed \
         as one undo step. Ctrl+Z takes it back.";

    /// **Settle every in-flight gesture, unconditionally** — the door the
    /// branch-gated settlers share, and the one the render loop's panic exits
    /// call on their way out (P21.3 audit).
    ///
    /// Returns `true` when anything was recorded.
    ///
    /// The three strokes each already mutated the world per dab; the gizmo drag
    /// and the foliage stroke additionally hold an open undo transaction. When
    /// the render thread is about to stop — a caught panic in the interaction
    /// block or in `render_frame` — the editor process survives, the document
    /// survives, and every one of those edits is in it with **no `EditCommand`
    /// describing it**. The author is then looking at a level they cannot undo
    /// back to, in a session that otherwise still works.
    ///
    /// Deliberately not a `Drop` guard: settling needs `&mut SceneDoc`, which a
    /// destructor on the host cannot have.
    pub fn settle_all_gestures(&mut self, doc: &mut SceneDoc) -> bool {
        let mut settled = false;
        if self.voxel_stroke.is_some() {
            settled |= self.finish_voxel(doc);
        }
        // A box-cut drag has cut nothing (the pit commits on release), so it is
        // dropped rather than settled — committing would dig a pit the author
        // abandoned.
        self.voxel_box_anchor = None;
        self.voxel_box_cursor = None;
        self.voxel_box_plan = None;
        if self.sculpt_drag.is_some() {
            settled |= self.finish_sculpt(doc);
        }
        if self.foliage_drag.is_some() {
            settled |= self.finish_foliage(doc);
        }
        // Last, because the two above close transactions of their own: whatever
        // is still open belongs to a gizmo drag with no owner left.
        self.gizmo_drag = None;
        if doc.has_open_transaction() {
            settled |= doc.settle_open_transaction();
        }
        settled
    }

    /// **Abandon every in-flight gesture when the document is replaced under
    /// us** (P21.3 audit). Returns `true` when something was dropped.
    ///
    /// `scene_open` / `scene_new` do `*doc = …` under the scene lock, on the
    /// Ring-2 thread. The viewport thread then wakes up holding gestures whose
    /// state points into a document that no longer exists: a sculpt stroke's
    /// terrain entity, a carve stroke's hole builders, a gizmo drag's selection,
    /// an open transaction. `clear_streams` — the one thing the swap does notify
    /// us about — travels down a command channel and only releases *streams*.
    ///
    /// The consequence is not a stale gesture, it is a **wrong edit**: the
    /// settlers above would faithfully commit the old level's terrain deltas
    /// into the new document, where one Ctrl+Z then applies them.
    ///
    /// So these are **abandoned, not settled** — the opposite ruling to every
    /// other settler here, and for a reason that inverts the usual argument. A
    /// settler commits because *the author can see the edit*; after a document
    /// swap they cannot, because the world it belonged to is gone. There is
    /// nothing to make undoable and everything to keep out of the new level.
    pub fn abandon_gestures_on_document_swap(&mut self, doc: &SceneDoc) -> bool {
        let id = doc.doc_id();
        if self.last_doc_id == Some(id) {
            return false;
        }
        let first_sight = self.last_doc_id.is_none();
        self.last_doc_id = Some(id);
        if first_sight {
            return false; // binding to the first document is not a swap
        }
        let had = self.sculpt_drag.is_some()
            || self.foliage_drag.is_some()
            || self.voxel_stroke.is_some()
            || self.voxel_box_anchor.is_some()
            || self.gizmo_drag.is_some()
            || !self.voxel_path.is_empty();
        self.sculpt_drag = None;
        self.foliage_drag = None;
        self.voxel_stroke = None;
        self.voxel_stroke_terrains.clear();
        self.voxel_stroke_last = None;
        self.voxel_box_anchor = None;
        self.voxel_box_cursor = None;
        self.voxel_box_plan = None;
        self.voxel_path.clear();
        self.voxel_preview.clear();
        self.gizmo_drag = None;
        self.water_active_river = None;
        self.water_lake_drag = None;
        self.water_preview.clear();
        // **And the carved chunks themselves** (P21.3 re-audit).
        //
        // Dropping the stroke handle above is not enough, because the shared
        // voxel store is a **host** field: it outlives the document swap
        // entirely. Its slots still hold whatever the abandoned stroke cut, and
        // those chunks are *dirty*. When the new document binds the same
        // `(entity, asset)` pair — which is exactly what File ▸ Open on the same
        // level does — `sync_voxels`' `retain_only` keeps the slot and `ensure`
        // short-circuits on `is_bound`, so the half-carve is silently **reused**
        // and the next Ctrl+S writes it into the `.inf_voxel`. With no
        // `EditCommand` anywhere: opening a level to *discard* changes would
        // commit them.
        //
        // Clearing is the whole fix. The new document re-`ensure`s from disk, so
        // every volume comes back as its **last saved** state, which is what
        // "the level that just closed is gone" means. A swap to a *different*
        // level would have dropped these slots on the next projection anyway;
        // this makes the same-level case behave like it.
        if let Ok(mut volumes) = self.voxel_volumes.lock() {
            volumes.clear();
        }
        if had {
            self.reject_tool(Self::GESTURE_ABANDONED_ON_DOCUMENT_SWAP);
        }
        had
    }

    /// What the author is told when a File ▸ Open dropped their gesture.
    const GESTURE_ABANDONED_ON_DOCUMENT_SWAP: &'static str =
        "Edit: the level changed while a drag was in progress, so the drag was dropped. It \
         belonged to the level that just closed.";

    /// **Close an undo transaction no gesture is going to close** (P21.3 audit —
    /// the coordinator's ruling). Returns `true` when one was settled.
    ///
    /// The stroke settlers above close *strokes*; this closes the other half of
    /// the same failure — a bare `begin_transaction` with no owner object. The
    /// win32 pump opens `"Move"` when a gizmo drag starts and commits it on
    /// release, **both inside the tool-gated select branch**, so *hold a
    /// translate handle → Ctrl+Shift+P → `tool.sculpt` → release* leaves it open
    /// forever. From then on every begin/commit pair bounces the nesting depth
    /// 1 → 2 → 1 without closing, every edit is folded into the stranded entry,
    /// `undo_len()` stops growing, and **Ctrl+Z is silently dead for the rest of
    /// the session**. Nothing else surfaces it: the edits land, the document is
    /// dirty, the save works.
    ///
    /// The guard is "no gesture of MINE owns one": a gizmo drag and a foliage
    /// stroke are the two that hold a transaction across frames. Ring-2 commands
    /// open and close theirs synchronously inside one function under the
    /// document lock this thread also takes, so none of those can be observed
    /// half-open from here.
    pub fn settle_orphaned_transaction(&mut self, doc: &mut SceneDoc) -> bool {
        if !doc.has_open_transaction() {
            return false;
        }
        if self.is_dragging_gizmo() || self.is_painting_foliage() {
            return false; // its owner is still alive and will close it
        }
        let settled = doc.settle_open_transaction();
        if settled {
            self.reject_tool(Self::TRANSACTION_SETTLED_ON_TOOL_SWITCH);
        }
        settled
    }

    /// What the author is told when a stranded transaction was closed for them.
    const TRANSACTION_SETTLED_ON_TOOL_SWITCH: &'static str =
        "Edit: a drag was interrupted before it finished, so the change so far was committed as \
         one undo step. Ctrl+Z takes it back.";

    /// Finish the stroke: commit the merged height [`inf_terrain::HeightDelta`] or
    /// splat [`inf_terrain::SplatDelta`] as one undo step. Returns `true` if a
    /// non-empty stroke was recorded.
    pub fn finish_sculpt(&mut self, doc: &mut SceneDoc) -> bool {
        let Some(drag) = self.sculpt_drag.take() else {
            return false;
        };
        let recorded = match drag.kind {
            DragStroke::Height(stroke) => doc.edit_commit_sculpt(drag.guid, stroke),
            DragStroke::Splat(stroke) => doc.edit_commit_paint(drag.guid, stroke),
            DragStroke::Biome(stroke) => doc.edit_commit_biome(drag.guid, stroke),
        };
        self.after_terrain_edit(doc, drag.guid);
        recorded
    }

    // ── foliage painting (E-P6) ───────────────────────────────────────────

    /// `true` while a foliage scatter stroke is in progress.
    pub fn is_painting_foliage(&self) -> bool {
        self.foliage_drag.is_some()
    }

    /// The world point under the cursor for the foliage brush centre: the terrain
    /// surface (reusing [`Self::pick_world_point`]'s terrain-then-ground rule).
    fn foliage_center(&self, doc: &SceneDoc, view: &RenderView, px: u32, py: u32) -> DVec3 {
        self.pick_world_point(doc, view, px, py)
    }

    /// Rebuild the foliage brush ring around a world-space cursor point (terrain
    /// height when over terrain, else a flat ground-plane ring), coloured green.
    fn refresh_foliage_ring(&mut self, doc: &SceneDoc, center: DVec3) {
        const FOLIAGE_RING: [f32; 4] = [0.35, 0.85, 0.40, 1.0];
        self.sculpt_ring_color = FOLIAGE_RING;
        let center_xz = DVec2::new(center.x, center.z);
        // P16.6: the ring follows the TOPMOST terrain covering the brush centre —
        // the same surface `foliage_surface_height` lifts instances onto — and it
        // is resolved through `terrain_probes`, so a streamed terrain rings on the
        // pages it has actually paged in rather than not at all.
        let probes = self.terrain_probes(doc, None);
        if let Some((guid, _)) = topmost_surface(&probes, center_xz) {
            if let Some(probe) = probes.iter().find(|p| p.guid == guid) {
                let local = DVec2::new(
                    center_xz.x - probe.translation.x,
                    center_xz.y - probe.translation.z,
                );
                self.sculpt_ring =
                    build_ring(probe.data, probe.translation, local, self.foliage.radius);
                return;
            }
        }
        self.sculpt_ring = ground_ring(center_xz, self.foliage.radius);
    }

    // ── the water tool (P20.4) ───────────────────────────────────────────
    //
    // Not a brush. A river click *appends a control point*; a lake press-drag
    // *defines a rectangle*. Both resolve the world point the same way every
    // other terrain tool does (`pick_world_point`), so a river lands on the
    // ground the author is looking at rather than on a plane at y = 0.
    //
    // Every mutation goes through a `SceneDoc::edit_*`, so each one is exactly
    // one undo step and the document — not this host — owns the change. The
    // host's own state is the *pending* gesture and the preview, both of which
    // are thrown away when the tool is left.

    /// Why the water tool refused a click that missed the ground.
    const WATER_NO_GROUND_REJECTION: &'static str =
        "Water: no terrain under the cursor. Aim at ground that has paged in — a water \
         placement here would commit geometry at sea level.";
    /// …and why it refused a lake drag too small to be a lake.
    const WATER_LAKE_TOO_SMALL_REJECTION: &'static str =
        "Water: that lake is under a metre across. Drag a larger rectangle — a zero-extent \
         lake draws nothing and would be an invisible entity in the outliner.";

    /// **The water tool's world pick** — like [`pick_world_point`](Self::pick_world_point)
    /// but it **refuses** rather than falling through to the `y = 0` plane
    /// (P20.4 audit).
    ///
    /// That fallback is right for a drag-drop (a cube on the ground plane is a
    /// defensible guess) and wrong here, because a water click **commits
    /// geometry**: over a streamed terrain that has only paged in coarsely, or
    /// over a hole, the fallback would silently plant a river control point or a
    /// lake corner at sea level — and two authors at different camera distances
    /// would commit *different* geometry from the same click. The sculpt brush
    /// already guards its own commits through `reject_tool`; this routes the
    /// water tool through the same seam.
    ///
    /// A level with **no terrain at all** is not a miss: there the ground plane
    /// is the only ground there is, and placing water on it is exactly right. The
    /// refusal is specifically "there IS terrain and the ray did not hit it".
    fn water_pick(&self, doc: &SceneDoc, view: &RenderView, px: u32, py: u32) -> Option<DVec3> {
        let probes = self.terrain_probes(doc, None);
        if probes.is_empty() {
            // No terrain in the level: the ground plane is the ground.
            return Some(self.pick_world_point(doc, view, px, py));
        }
        let (ro, rd) = view.pixel_ray(px as f32, py as f32);
        let ro_w = self.origin.to_world(ro);
        nearest_terrain_hit(&probes, ro_w, rd.as_dvec3()).map(|h| h.world)
    }

    /// The still-water level a new body takes at world point `p`.
    ///
    /// The biome's `water_hint` wins over the ground when Ring 2 has pushed one
    /// (P19.2's field, read at last), then the tool's own offset is added. The
    /// same rule as `inf_editor_core::hydro::water_defaults`, and it is stated in
    /// both places because the viewport thread cannot resolve a `.inf_biomes`
    /// asset — it gets the resolved hint pushed, exactly as it gets a biome
    /// *palette* pushed rather than a biome *set*.
    fn water_level_for(&self, doc: &SceneDoc, p: DVec3) -> f64 {
        let base = self
            .biome_water_hint(doc, DVec2::new(p.x, p.z))
            .unwrap_or(p.y);
        base + self.water_tool.level_offset_m
    }

    /// The water-level hint of the biome painted under world XZ `p`, if the
    /// terrain there has one.
    ///
    /// Resolved through the **topmost-ground rule** — the same one the brush ring
    /// and the foliage drop height use here, and the same one
    /// `inf_editor_core::hydro::topmost_ground` uses for the `water_defaults`
    /// command (P20.4 audit: those two were a topmost rule and a
    /// lowest-`Guid` rule under a doc claiming they matched). The two resolve it
    /// against different *data* — this against the streamer's paged working set,
    /// Ring 1 against the document's resident tiles — which is the streamed-
    /// terrain fact of life and is why the tool answers from what is on screen.
    fn biome_water_hint(&self, doc: &SceneDoc, p: DVec2) -> Option<f64> {
        if self.water_hints.is_empty() {
            return None;
        }
        let probes = self.terrain_probes(doc, None);
        let (guid, _) = topmost_surface(&probes, p)?;
        let probe = probes.iter().find(|q| q.guid == guid)?;
        let local = DVec2::new(p.x - probe.translation.x, p.y - probe.translation.z);
        let id = probe.data.biome_at(local)?;
        self.water_hints
            .get(&guid)
            .and_then(|h| h.get(id as usize).copied())
            .flatten()
    }

    /// Begin a water gesture. For a **river** this is the whole interaction (one
    /// click = one control point); for a **lake** it records the first corner.
    ///
    /// Returns `true` when the document changed, so the caller emits a delta.
    pub fn begin_water(&mut self, doc: &mut SceneDoc, view: &RenderView, px: u32, py: u32) -> bool {
        let Some(p) = self.water_pick(doc, view, px, py) else {
            self.reject_tool(Self::WATER_NO_GROUND_REJECTION);
            self.water_lake_drag = None;
            self.water_preview.clear();
            return false;
        };
        match self.water_tool.kind {
            WaterToolKind::Lake => {
                self.water_lake_drag = Some(p);
                self.water_preview.clear();
                false
            }
            WaterToolKind::River => {
                // Extend the active river if there is one, else the selected
                // river, else start a new one. Consulting the selection is what
                // makes "click a river, keep drawing it" work after a reload.
                let target = self
                    .water_active_river
                    .filter(|g| doc.world().entity_of(*g).is_some())
                    .or_else(|| {
                        doc.selection()
                            .iter()
                            .copied()
                            .find(|g| Self::is_river(doc, *g))
                    });
                let changed = match target {
                    Some(guid) => {
                        // `edit_append_spline_point` CREATES the `Spline` when the
                        // entity has none (P20.4 audit): a `WaterKind::River` added
                        // through the Details menu is exactly the state "I have
                        // declared a river and not drawn it yet", and refusing it
                        // wedged the tool — every click resolved to the same
                        // spline-less selection and did nothing, with no message.
                        doc.edit_append_spline_point(guid, p)
                    }
                    None => {
                        // A river needs two points to be a ribbon; the first click
                        // lays both, a metre apart along +X, so the author sees
                        // something immediately and the second click moves the
                        // mouth rather than creating a degenerate path.
                        let guid = doc.edit_create_river(
                            "River",
                            &[p, p + DVec3::new(1.0, 0.0, 0.0)],
                            self.water_tool.width_m,
                            self.water_tool.depth_m,
                            self.water_tool.flow_m_s,
                        );
                        doc.select(&[guid], false);
                        self.water_active_river = Some(guid);
                        true
                    }
                };
                if let Some(g) = target {
                    self.water_active_river = Some(g);
                }
                self.water_preview.clear();
                changed
            }
        }
    }

    /// Continue a lake drag: redraw the rectangle + its waterline preview. A
    /// river has nothing to continue (its gesture is the click).
    pub fn update_water(&mut self, doc: &SceneDoc, view: &RenderView, px: u32, py: u32) {
        let Some(anchor) = self.water_lake_drag else {
            return;
        };
        let Some(p) = self.water_pick(doc, view, px, py) else {
            self.reject_tool(Self::WATER_NO_GROUND_REJECTION);
            self.water_preview.clear();
            return;
        };
        self.rebuild_lake_preview(doc, anchor, p);
    }

    /// Finish a lake drag: create the lake. Returns whether the document changed.
    ///
    /// A drag smaller than a metre on either side is treated as a mis-click and
    /// creates nothing — a zero-extent lake is undrawable
    /// (`RenderWater::drawable`) and would be an invisible entity in the outliner.
    pub fn finish_water(
        &mut self,
        doc: &mut SceneDoc,
        view: &RenderView,
        px: u32,
        py: u32,
    ) -> bool {
        let Some(anchor) = self.water_lake_drag.take() else {
            return false;
        };
        self.water_preview.clear();
        let Some(p) = self.water_pick(doc, view, px, py) else {
            self.reject_tool(Self::WATER_NO_GROUND_REJECTION);
            return false;
        };
        let half = DVec2::new((p.x - anchor.x).abs() * 0.5, (p.z - anchor.z).abs() * 0.5);
        if half.x < 0.5 || half.y < 0.5 {
            // Refusing silently is how a mis-click looks like a broken tool.
            self.reject_tool(Self::WATER_LAKE_TOO_SMALL_REJECTION);
            return false;
        }
        let center = DVec3::new((anchor.x + p.x) * 0.5, anchor.y, (anchor.z + p.z) * 0.5);
        let guid = doc.edit_create_lake(
            "Lake",
            center,
            inf_ecs::Vec2d::new(half.x, half.y),
            self.water_level_for(doc, anchor),
        );
        doc.select(&[guid], false);
        true
    }

    /// Idle hover in water mode: a river shows the segment the next click would
    /// add; a lake shows nothing until the drag starts.
    pub fn update_water_hover(&mut self, doc: &SceneDoc, view: &RenderView, px: u32, py: u32) {
        if self.water_tool.kind != WaterToolKind::River {
            return;
        }
        self.water_preview.clear();
        let Some(p) = self.water_pick(doc, view, px, py) else {
            return;
        };
        let tail = self
            .water_active_river
            .or_else(|| {
                doc.selection()
                    .iter()
                    .copied()
                    .find(|g| Self::is_river(doc, *g))
            })
            .and_then(|g| Self::river_tail(doc, g));
        if let Some(tail) = tail {
            self.water_preview.push([tail, p]);
        }
    }

    /// Whether `guid` is a river (a `WaterBody` of kind River).
    fn is_river(doc: &SceneDoc, guid: Uuid) -> bool {
        doc.world()
            .entity_of(guid)
            .and_then(|e| doc.world().world().get::<WaterBody>(e))
            .is_some_and(|b| b.kind == inf_ecs::components::WaterKind::River)
    }

    /// The world position of a river's last control point, for the hover preview.
    fn river_tail(doc: &SceneDoc, guid: Uuid) -> Option<DVec3> {
        let world = doc.world();
        let e = world.entity_of(guid)?;
        let w = world.world();
        let spline = w.get::<Spline>(e)?;
        let last = spline.points.last()?;
        let affine = w
            .get::<GlobalTransform>(e)
            .map(|g| g.0)
            .unwrap_or(glam::DAffine3::IDENTITY);
        Some(affine.transform_point3(last.to_dvec3()))
    }

    /// Rebuild the lake preview: the rectangle the drag describes, plus the
    /// **waterline** — where the still-water level meets the ground inside it.
    ///
    /// The waterline comes from `inf_editor_core::hydro::lake_preview`, the same
    /// function the `water_lake_preview` command answers with, so what the author
    /// sees and what the panel says are one computation. Camera-independent: the
    /// contour is a function of the rectangle, the level and the terrain, and a
    /// preview that moved with the eye would be the P18.2 residency law broken in
    /// authoring.
    fn rebuild_lake_preview(&mut self, doc: &SceneDoc, anchor: DVec3, cursor: DVec3) {
        const PREVIEW_RESOLUTION: u32 = 48;
        self.water_preview.clear();
        let half = DVec2::new(
            (cursor.x - anchor.x).abs() * 0.5,
            (cursor.z - anchor.z).abs() * 0.5,
        );
        if half.x <= 0.0 || half.y <= 0.0 {
            return;
        }
        let center = DVec2::new((anchor.x + cursor.x) * 0.5, (anchor.z + cursor.z) * 0.5);
        let level = self.water_level_for(doc, anchor);
        // The rectangle itself, at the water level.
        let corners = [
            DVec3::new(center.x - half.x, level, center.y - half.y),
            DVec3::new(center.x + half.x, level, center.y - half.y),
            DVec3::new(center.x + half.x, level, center.y + half.y),
            DVec3::new(center.x - half.x, level, center.y + half.y),
        ];
        for i in 0..4 {
            self.water_preview.push([corners[i], corners[(i + 1) % 4]]);
        }
        // …and the waterline contour inside it.
        let preview =
            inf_editor_core::hydro::lake_preview(doc, center, half, level, PREVIEW_RESOLUTION);
        for [a, b] in &preview.waterline {
            self.water_preview
                .push([DVec3::new(a.x, level, a.y), DVec3::new(b.x, level, b.y)]);
        }
        // The measurements the preview computes reach the AUTHOR, on the status
        // seam, live as the rectangle is dragged (P20.4 audit: they were computed
        // and dropped). `known == 0` gets its own sentence, because "there is no
        // ground under this rectangle" and "this lake is empty" are different
        // facts that would otherwise both render as 0 %.
        let msg = if !preview.has_ground() {
            format!(
                "Lake {:.0} × {:.0} m at {level:.1} m — NO GROUND under this rectangle",
                half.x * 2.0,
                half.y * 2.0
            )
        } else {
            let partial = if preview.known < preview.samples {
                format!(
                    " ({} of {} samples have ground)",
                    preview.known, preview.samples
                )
            } else {
                String::new()
            };
            format!(
                "Lake {:.0} × {:.0} m at {level:.1} m — covers {:.0}%, up to {:.1} m deep \
                 (mean {:.1} m){partial}",
                half.x * 2.0,
                half.y * 2.0,
                preview.covered_fraction * 100.0,
                preview.max_depth_m,
                preview.mean_depth_m,
            )
        };
        self.report_tool(msg);
    }

    // ── the voxel carve and dig tools (P21.2 + P21.3) ─────────────────────
    //
    // FOUR sub-modes over three gestures, and this module owns none of the
    // shapes they cut — those are `inf_editor_core::voxel_tool`, Ring 1, tested
    // on every CI leg (the M11 move). What is here is the input plumbing:
    //
    // * **brush** — the sculpt brush's gesture exactly: press, drag, release,
    //   dabs spaced by arc length so drag speed cannot change what is dug;
    // * **tunnel** and **trench** — the river tool's: click waypoints,
    //   Ctrl+click to close and cut the whole run as one step (a round bore for
    //   the tunnel, a rectangular section open to the sky for the trench);
    // * **box cut** — the lake tool's: press-drag a rectangle, release
    //   excavates it.
    //
    // Every cut goes through `SceneDoc::edit_dig` (or, for the live brush,
    // `carve_verdict` per dab) FIRST. That is the gate the coordinator's
    // inline-terrain ruling lives behind: schema v19 cannot persist a hole mask
    // on an inline terrain, so a surface-crossing cut there is refused whole —
    // voxels included — rather than half-applied into a cave with no mouth. A
    // cut that never reaches a surface is legal anywhere. P21.3 adds a size gate
    // in the same pass, for the same reason: you find out a dig is too big by
    // doing it, which is exactly what "judged whole" forbids.

    /// Why the voxel tool refused a click with no volume to cut.
    const VOXEL_NO_VOLUME_REJECTION: &'static str =
        "Carve: no voxel volume is loaded. Select an entity with a VoxelVolume whose .inf_voxel \
         resolved under the project's Content root — a carve needs chunks to cut, and this level \
         has none.";
    /// …and why it refused a click that found no ground to aim at.
    const VOXEL_NO_GROUND_REJECTION: &'static str =
        "Carve: no terrain under the cursor. Aim at ground that has paged in — a cut placed at \
         sea level would dig somewhere the author never pointed at.";

    /// …and what the spoil-site picker says while it is armed with nowhere
    /// picked yet — a hint, not a refusal: the dig still happens, at the
    /// documented default site.
    const VOXEL_NO_SPOIL_SITE_HINT: &'static str =
        "Spoil: no site picked yet, so the soil goes to the default spot — east of the cut, \
         clear of its rim. Turn on \"Set spoil site\" and click where you want the heap.";

    /// `true` while a carve stroke **or a box-cut drag** is in progress.
    ///
    /// One predicate for both because the pump gates `update_voxel` and
    /// `finish_voxel` on it: a box drag that answered `false` here would never
    /// see a move or a release, which is a rectangle the author drags and the
    /// editor never cuts.
    pub fn is_carving(&self) -> bool {
        self.voxel_stroke.is_some() || self.voxel_box_anchor.is_some()
    }

    /// Where this dig's material goes — the **state**; the rule is
    /// [`SpoilMode::choice`] in `camera.rs`, which is not `#[cfg]`-gated and is
    /// therefore tested on every CI leg (the M11 argument, completed in the
    /// P21.3 audit round).
    fn spoil_choice(&self) -> inf_editor_core::scene::undo::SpoilChoice {
        self.voxel_tool.spoil.choice(self.voxel_spoil_site)
    }

    /// The volume the next cut goes into — the **lookup**; the rule is
    /// [`inf_editor_core::voxel_tool::voxel_target`] (the M11 move).
    ///
    /// All this contributes is the answer to "does the shared store hold chunks
    /// for this entity?", which needs the mutex this thread owns and is
    /// therefore the one half of the question that cannot live in Ring 1.
    fn voxel_target(&self, doc: &SceneDoc) -> Option<Uuid> {
        inf_editor_core::voxel_tool::voxel_target(doc.order(), doc.selection(), |g| {
            self.voxel_volumes
                .lock()
                .map(|v| v.slot(g).is_some())
                .unwrap_or(false)
        })
    }

    /// The sunk cut centre for the current depth — the rule is
    /// [`inf_editor_core::voxel_tool::cut_center`] (the M11 move).
    fn voxel_sink(&self, surface: DVec3) -> DVec3 {
        inf_editor_core::voxel_tool::cut_center(surface, self.voxel_tool.depth_m)
    }

    /// The **raw** ground point under the cursor, before the depth sink.
    ///
    /// Split out for P21.3: a box cut, a trench and a dig-to-depth brush dab all
    /// need to know where *daylight* is (their tops clear the surface), while
    /// the sphere brush and the tunnel want the sunk centre. Two callers, one
    /// pick, so the two can never disagree about which ground the click found.
    fn voxel_surface_pick(
        &self,
        doc: &SceneDoc,
        view: &RenderView,
        px: u32,
        py: u32,
    ) -> Option<DVec3> {
        let probes = self.terrain_probes(doc, None);
        if probes.is_empty() {
            // No terrain in the level: the ground plane is the only ground there
            // is, exactly as the water tool treats it.
            return Some(self.pick_world_point(doc, view, px, py));
        }
        let (ro, rd) = view.pixel_ray(px as f32, py as f32);
        let ro_w = self.origin.to_world(ro);
        nearest_terrain_hit(&probes, ro_w, rd.as_dvec3()).map(|h| h.world)
    }

    /// The op one dab of the current settings makes at the **surface** point
    /// `surface`.
    ///
    /// The shape itself is [`inf_editor_core::voxel_tool::brush_dab_shape`] —
    /// Ring 1, pure, tested on every CI leg — because "what does one dab dig" is
    /// a rule about committed geometry and this module is
    /// `#[cfg(any(windows, macos))]` (the M11 argument).
    fn voxel_op_at(&self, surface: DVec3) -> VoxelOp {
        self.voxel_shape_op(inf_editor_core::voxel_tool::brush_dab_shape(
            surface,
            self.voxel_tool.radius_m,
            self.voxel_tool.depth_m,
            self.voxel_tool.dig_to_depth,
        ))
    }

    /// Wrap `shape` in the carve-or-fill the toolbar selected. One place, so the
    /// brush and the tunnel cannot disagree about which way the cut runs.
    fn voxel_shape_op(&self, shape: VoxelShape) -> VoxelOp {
        match self.voxel_tool.mode {
            VoxelOpMode::Carve => VoxelOp::carve(shape),
            VoxelOpMode::Fill => VoxelOp::fill(shape, self.voxel_tool.material),
        }
    }

    /// Judge one cut and, when it is allowed, page the heightfield footprint it
    /// will write so the hole rule sees real tiles.
    ///
    /// Returns the terrains this cut may open, or `None` when it is refused (the
    /// refusal is reported on the status seam here, so no caller can drop it).
    ///
    /// Judged **per dab** rather than once per gesture: a drag walks, and the
    /// terrain a dab actually reaches is the one whose container has to be able to
    /// save the mouth it opens. Judging only the first dab would let a stroke
    /// wander onto an inline terrain and seal its mouths at the next save — the
    /// exact failure the ruling exists to prevent, arrived at sideways.
    fn voxel_authorize(&mut self, doc: &mut SceneDoc, shape: &VoxelShape) -> Option<Vec<Uuid>> {
        self.page_cut_footprint(doc, shape);
        match doc.carve_verdict(shape) {
            inf_editor_core::scene::undo::CarveVerdict::RefusedInline { .. } => {
                self.reject_tool(inf_editor_core::scene::undo::INLINE_TERRAIN_CARVE_REFUSAL);
                None
            }
            v => Some(v.terrains().to_vec()),
        }
    }

    /// Page every streamed terrain's tiles under `shape` into the **document's**
    /// working set.
    ///
    /// Run before the verdict, not after: `cut_crosses_surface` only sees
    /// authored tiles, so on a streamed terrain an unpaged footprint would answer
    /// "this cut reaches no surface" and wave through a breakthrough nobody
    /// checked — and then the carve would open a mouth on whichever tiles
    /// happened to be in memory. The sculpt brush's `page_brush_footprint` rule,
    /// with the cut's XZ silhouette in place of the brush radius.
    fn page_cut_footprint(&mut self, doc: &mut SceneDoc, shape: &VoxelShape) {
        let (lo, hi) = shape.aabb_m(0.0);
        let center = DVec2::new((lo.x + hi.x) * 0.5, (lo.z + hi.z) * 0.5);
        let radius = ((hi.x - lo.x).max(hi.z - lo.z) * 0.5).max(0.0);
        self.page_terrain_disc(doc, center, radius);
    }

    /// Page the ground under the **spoil site** before the dig resolves it
    /// (P21.3 audit, B3).
    ///
    /// `SceneDoc::spoil_site`'s `Auto` rule drops the heap onto
    /// `ground_surface_y`, which reads resident tiles only — so on a streamed
    /// terrain the heap's height, and therefore the committed geometry, would
    /// depend on which tiles the camera had visited. `Ring 1` cannot fix this
    /// itself: it has the rule and no streamer.
    ///
    /// The exact site is not known until the cut has run (it is offset by the
    /// pile's own radius), so this pages a **conservative band**: from the cut's
    /// eastern face out by the radius the dig's own sample bound allows, plus
    /// the gap. Deterministic, and a superset of wherever the rule lands.
    fn page_spoil_ground(&mut self, doc: &mut SceneDoc, bounds: (DVec3, DVec3), voxels: u64) {
        if self.voxel_tool.spoil != SpoilMode::Auto {
            return; // a picked site is paged by the click that placed it
        }
        let (lo, hi) = bounds;
        if !(lo.is_finite() && hi.is_finite()) {
            return;
        }
        let size = self
            .voxel_target(doc)
            .map(|v| self.voxel_size_of(v))
            .unwrap_or(0.5);
        let reach = inf_voxel::pile_base_radius_m(voxels, size) + inf_voxel::SPOIL_GAP_M;
        let center = DVec2::new(hi.x + reach * 0.5, (lo.z + hi.z) * 0.5);
        let radius = (reach + (hi.z - lo.z) * 0.5).max(1.0);
        self.page_terrain_disc(doc, center, radius);
    }

    /// Begin a carve gesture. For the **brush** this opens a stroke and lays the
    /// first dab; for the **tunnel** it appends a waypoint (and, with `ctrl`,
    /// closes the path and carves the whole tube).
    ///
    /// Returns `true` when the document changed, so the caller emits a delta.
    pub fn begin_voxel(
        &mut self,
        doc: &mut SceneDoc,
        view: &RenderView,
        px: u32,
        py: u32,
        ctrl: bool,
    ) -> bool {
        // The spoil-site mode owns the click before anything else does: while it
        // is on, the author is placing a marker, not digging. Checked ahead of
        // the volume gate so a level with no volume can still be marked up.
        if self.voxel_tool.pick_spoil_site {
            match self.voxel_surface_pick(doc, view, px, py) {
                Some(p) => {
                    self.voxel_spoil_site = Some(p);
                    self.report_tool(format!(
                        "Spoil site set at ({:.1}, {:.1}, {:.1}). Turn off \"Set spoil site\" to \
                         dig again.",
                        p.x, p.y, p.z
                    ));
                    self.refresh_voxel_preview(p);
                }
                None => self.reject_tool(Self::VOXEL_NO_GROUND_REJECTION),
            }
            return false;
        }
        let Some(volume) = self.voxel_target(doc) else {
            self.reject_tool(Self::VOXEL_NO_VOLUME_REJECTION);
            return false;
        };
        let Some(surface) = self.voxel_surface_pick(doc, view, px, py) else {
            self.reject_tool(Self::VOXEL_NO_GROUND_REJECTION);
            return false;
        };
        let p = self.voxel_sink(surface);
        match self.voxel_tool.kind {
            VoxelToolKind::BoxCut => {
                // A pit is committed on RELEASE, judged whole: the press only
                // anchors the rectangle. Nothing is cut here, which is what
                // makes an abandoned drag cost nothing.
                self.voxel_box_anchor = Some(surface);
                self.voxel_box_cursor = Some(surface);
                self.refresh_voxel_preview(surface);
                self.report_box_cut(doc, volume);
                false
            }
            VoxelToolKind::Brush => {
                let op = self.voxel_op_at(surface);
                let Some(terrains) = self.voxel_authorize(doc, &op.shape) else {
                    return false;
                };
                let mut stroke = inf_editor_core::scene::undo::CarveStroke::begin(
                    volume,
                    self.voxel_volumes.clone(),
                );
                // A refusal from the first dab ends the gesture before it starts:
                // the volume was released or its working set is poisoned, so
                // nothing was cut and nothing above it was opened. Report the
                // reason the stroke gave rather than a paraphrase — it is the
                // only thing that tells the author which of the three it was.
                let tally = match stroke.dab(doc, &op, &terrains) {
                    Ok(t) => t,
                    Err(refusal) => {
                        self.reject_tool(refusal.message());
                        return false;
                    }
                };
                self.voxel_stroke = Some(stroke);
                self.voxel_stroke_terrains = terrains.clone();
                // The stroke's path is resampled in SURFACE space, not in sunk
                // space: the two differ by a constant only while the depth does,
                // and a dig-to-depth column has no sunk centre at all.
                self.voxel_stroke_last = Some(surface);
                for g in terrains {
                    self.after_terrain_edit(doc, g);
                }
                self.report_carve(volume, tally);
                self.refresh_voxel_preview(p);
                // The dabs mutate live; the undo entry lands at mouse-up.
                !tally.is_noop()
            }
            // The two path tools share the gesture and differ only in the shape
            // the commit sweeps — a round bore for a tunnel, a square section
            // for a trench (see `commit_path`).
            VoxelToolKind::Tunnel | VoxelToolKind::Trench => {
                // A tunnel's waypoints are its centreline (already sunk by the
                // depth); a trench's are the SURFACE it is cut down from, and
                // `trench_shapes` applies the depth itself.
                self.voxel_path
                    .push(if self.voxel_tool.kind == VoxelToolKind::Trench {
                        surface
                    } else {
                        p
                    });
                if ctrl && self.voxel_path.len() >= 2 {
                    return self.commit_path(doc, volume);
                }
                self.refresh_voxel_preview(p);
                false
            }
        }
    }

    /// The most dabs one [`update_voxel`](Self::update_voxel) will lay before it
    /// stops and **carries the rest of the stroke to the next frame**.
    ///
    /// The spacing floor is 0.05 m and a drag has no length limit, so one fast
    /// swing across a hundred metres asks for two thousand dabs — and every dab
    /// pages a streamed terrain's footprint off disk, walks every terrain for a
    /// verdict, cuts, and re-meshes what it moved, **all inside the one scene
    /// mutex** that autosave and every Ring-2 command also need. Uncapped, that is
    /// a multi-second freeze in which the crash-recovery write cannot run.
    ///
    /// Capping without carrying would be worse than the freeze: the stroke would
    /// skip to the cursor and leave a gap in the middle of the cut. Resuming from
    /// the last dab actually placed keeps the gesture continuous, and the pump
    /// calls this **every frame while the button is down** (not only when the
    /// cursor moves), so a stroke that outran its budget still drains itself while
    /// the author holds still.
    const MAX_DABS_PER_UPDATE: usize = 32;

    /// Continue a brush stroke: resample the path from the last dab to the cursor
    /// at even arc length (~⅔ radius, the sculpt brush's rule at the coarser
    /// spacing a *volume* wants) and cut at each, up to
    /// [`MAX_DABS_PER_UPDATE`](Self::MAX_DABS_PER_UPDATE) per frame.
    pub fn update_voxel(&mut self, doc: &mut SceneDoc, view: &RenderView, px: u32, py: u32) {
        // The box cut's drag: rubber-band the rectangle and quote what releasing
        // would excavate. Nothing is cut until the release, so this branch never
        // mutates the document.
        if self.voxel_box_anchor.is_some() {
            if let Some(surface) = self.voxel_surface_pick(doc, view, px, py) {
                self.voxel_box_cursor = Some(surface);
                self.refresh_voxel_preview(surface);
                if let Some(volume) = self.voxel_target(doc) {
                    self.report_box_cut(doc, volume);
                }
            }
            return;
        }
        if self.voxel_stroke.is_none() {
            return;
        }
        let Some(last) = self.voxel_stroke_last else {
            return;
        };
        let Some(cur) = self.voxel_surface_pick(doc, view, px, py) else {
            return; // the cursor slid off the ground — hold the stroke, cut nothing
        };
        // The resampling is Ring 1's (`voxel_tool::dab_centers`, itself a wrapper
        // over `inf_terrain::dab_positions`) — the M11 move. `skip(1)`: the
        // first entry is `last`, which is already cut.
        let spacing = inf_editor_core::voxel_tool::dab_spacing(self.voxel_tool.radius_m);
        // **Capped before it is materialized**, not filtered afterwards (P21.3
        // audit). `.take(32)` on a full list is not a bound: at the 0.05 m
        // spacing floor a drag whose pick landed a hundred kilometres away —
        // `pick_world_point` admits `t` out to 1e6 — builds two million points,
        // 80 MB and 27 ms, every frame, before discarding all but 32 of them.
        let centers = inf_editor_core::voxel_tool::dab_centers_capped(
            &[last, cur],
            spacing,
            Self::MAX_DABS_PER_UPDATE,
        );
        let mut tally = self
            .voxel_stroke
            .as_ref()
            .map(|s| s.tally())
            .unwrap_or_default();
        let mut refused = None;
        if centers.len() > 1 {
            let mut placed = last;
            for &center in centers.iter().skip(1) {
                let op = self.voxel_op_at(center);
                if let Some(terrains) = self.voxel_authorize(doc, &op.shape) {
                    if let Some(stroke) = self.voxel_stroke.as_mut() {
                        match stroke.dab(doc, &op, &terrains) {
                            Ok(cut) => tally = cut,
                            // The rock could not be cut, so nothing above it was
                            // opened either. Stop the stroke here — every later
                            // dab would refuse the same way — and let mouse-up
                            // commit whatever the earlier dabs did cut, which is
                            // the only way it stays undoable.
                            Err(r) => {
                                refused = Some(r);
                                break;
                            }
                        }
                    }
                    for g in terrains {
                        if !self.voxel_stroke_terrains.contains(&g) {
                            self.voxel_stroke_terrains.push(g);
                        }
                    }
                }
                placed = center;
            }
            // The remainder rides to the next frame from here, so the cut has no
            // gap where the budget ran out.
            self.voxel_stroke_last = Some(placed);
        }
        for g in self.voxel_stroke_terrains.clone() {
            self.after_terrain_edit(doc, g);
        }
        // A refusal wins the readout: a measurement line over it would tell the
        // author how much they dug while the brush had silently stopped digging.
        match refused {
            Some(r) => self.reject_tool(r.message()),
            None => {
                if let Some(volume) = self.voxel_stroke.as_ref().map(|s| s.volume()) {
                    self.report_carve(volume, tally);
                }
            }
        }
        self.refresh_voxel_preview(cur);
    }

    /// Finish a brush stroke **or a box-cut drag**: displace the soil and commit
    /// every half as ONE undo entry. Returns `true` if anything was recorded.
    pub fn finish_voxel(&mut self, doc: &mut SceneDoc) -> bool {
        if self.voxel_box_anchor.is_some() {
            return self.commit_box_cut(doc);
        }
        let Some(stroke) = self.voxel_stroke.take() else {
            return false;
        };
        self.voxel_stroke_last = None;
        let volume = stroke.volume();
        // The Auto spoil rule reads the ground east of the stroke's own
        // footprint, so that ground has to be paged before it does (P21.3
        // audit, B3).
        if let Some(bounds) = stroke.bounds() {
            let removed: u64 = stroke.tally().carved_by_material.iter().sum();
            self.page_spoil_ground(doc, bounds, removed);
        }
        // The spoil rides the SAME stroke, so the cut, its cave mouths and the
        // heap it produced are one `EditCommand` and one Ctrl+Z.
        let (tally, recorded, refusal) = doc.edit_commit_dig(stroke, self.spoil_choice());
        for g in std::mem::take(&mut self.voxel_stroke_terrains) {
            self.after_terrain_edit(doc, g);
        }
        match refusal {
            // The rock is committed either way (it is in the world and the author
            // can see it); what failed is the heap, and the readout says which.
            Some(r) => self.reject_tool(r.message()),
            None => self.report_carve(volume, tally),
        }
        recorded
    }

    /// Commit the pit the author dragged: judged whole, cut whole, spoiled whole
    /// (P21.3).
    ///
    /// Returns `true` when the document changed. Clears the drag either way — a
    /// refused pit that left its rectangle on screen would invite the author to
    /// release again and be refused again.
    fn commit_box_cut(&mut self, doc: &mut SceneDoc) -> bool {
        let version_before = doc.version();
        let anchor = self.voxel_box_anchor.take();
        let cursor = self.voxel_box_cursor.take();
        self.voxel_box_plan = None;
        self.voxel_preview.clear();
        let (Some(anchor), Some(cursor)) = (anchor, cursor) else {
            return false;
        };
        let Some(volume) = self.voxel_target(doc) else {
            self.reject_tool(Self::VOXEL_NO_VOLUME_REJECTION);
            return false;
        };
        // **Page the rectangle BEFORE the plan probes it** (P21.3 audit, B3) —
        // otherwise the pit's roof and floor are a function of camera residency.
        self.page_box_footprint(doc, anchor, cursor);
        let Some(plan) = self.box_cut_plan(doc, anchor, cursor) else {
            // A click rather than a drag. Silent: missing a drag is not an error
            // an author needs a sentence about.
            return false;
        };
        let op = self.voxel_shape_op(plan.shape);
        // …and page again around the resolved shape before the verdict, exactly
        // as the path tools do: `cut_crosses_surface` reads authored tiles only,
        // and the plan's box is taller than the rectangle that produced it.
        self.page_cut_footprint(doc, &op.shape);
        // The Auto spoil rule drops the heap onto the ground east of the cut, so
        // that ground has to be in the document before the rule reads it.
        let (blo, bhi) = op.shape.aabb_m(0.0);
        self.page_spoil_ground(
            doc,
            (blo, bhi),
            op.shape.affected_sample_count(self.voxel_size_of(volume)),
        );
        let tally = match doc.edit_dig(volume, &self.voxel_volumes, &[op], self.spoil_choice()) {
            Ok(tally) => tally,
            Err(refusal) => {
                self.reject_tool(refusal.message());
                // A refused dig cuts nothing, but paging its footprint DID move
                // the document (tiles arrived in the working set), and the
                // caller's `world_changed` is what re-emits the delta. Returning
                // a flat `false` here left the viewport's own projection a
                // version behind (P21.3 audit).
                return doc.version() != version_before;
            }
        };
        for i in 0..self.terrain_slots.len() {
            let guid = self.terrain_slots[i].guid;
            self.after_terrain_edit(doc, guid);
        }
        self.report_carve(volume, tally);
        !tally.is_noop() || doc.version() != version_before
    }

    /// Resolve the dragged rectangle into a pit through the Ring-1 planner.
    ///
    /// The ground query is this document's own topmost surface
    /// (`SceneDoc::ground_surface_y`), which is what makes a pit dragged across
    /// a slope reach daylight along its whole rim — see
    /// [`inf_editor_core::voxel_tool::box_cut_plan`] for the rule and its
    /// documented limit.
    fn box_cut_plan(
        &self,
        doc: &SceneDoc,
        anchor: DVec3,
        cursor: DVec3,
    ) -> Option<inf_editor_core::voxel_tool::BoxCutPlan> {
        inf_editor_core::voxel_tool::box_cut_plan(
            anchor,
            cursor,
            self.voxel_tool.depth_m.max(0.0),
            |x, z| doc.ground_surface_y(x, z),
        )
    }

    /// **Page the drag rectangle into the document before anything probes it**
    /// (P21.3 audit, B3).
    ///
    /// `box_cut_plan` reads the ground through `SceneDoc::ground_surface_y`,
    /// which answers `None` on a tile that is not resident and whose `None` the
    /// planner treats as "no ground here, skip". On a streamed terrain that
    /// makes the pit's floor and roof a function of **camera residency**: two
    /// authors at different distances dig different pits from the same drag, and
    /// the committed geometry depends on where the session had been. The
    /// standing law says it must not.
    ///
    /// Called before *both* the commit and the readout, so the number the author
    /// reads while dragging is the number the release cuts.
    fn page_box_footprint(&mut self, doc: &mut SceneDoc, anchor: DVec3, cursor: DVec3) {
        let center = DVec2::new((anchor.x + cursor.x) * 0.5, (anchor.z + cursor.z) * 0.5);
        let radius = ((anchor.x - cursor.x).abs().max((anchor.z - cursor.z).abs())) * 0.5;
        self.page_terrain_disc(doc, center, radius);
    }

    /// Page every streamed terrain's tiles under an XZ disc into the document's
    /// working set — the shared half of [`page_cut_footprint`](Self::page_cut_footprint)
    /// and [`page_box_footprint`](Self::page_box_footprint).
    fn page_terrain_disc(&mut self, doc: &mut SceneDoc, center: DVec2, radius: f64) {
        if !center.is_finite() || !radius.is_finite() {
            return;
        }
        for i in 0..self.terrain_slots.len() {
            let guid = self.terrain_slots[i].guid;
            if !self.terrain_slots[i].streamed {
                continue;
            }
            let translation = self
                .terrain_streams
                .projection_inputs(guid)
                .map(|(_, _, t)| t)
                .unwrap_or(DVec3::ZERO);
            let local = DVec2::new(center.x - translation.x, center.y - translation.z);
            self.terrain_streams
                .page_brush_footprint(guid, doc, local, radius.max(0.0));
        }
    }

    /// The pit-drag readout: what releasing right now would excavate.
    fn report_box_cut(&mut self, doc: &mut SceneDoc, volume: Uuid) {
        let (Some(anchor), Some(cursor)) = (self.voxel_box_anchor, self.voxel_box_cursor) else {
            return;
        };
        // Page first, then probe — the same order the commit uses, so the
        // readout cannot quote a pit the release would not cut.
        self.page_box_footprint(doc, anchor, cursor);
        let Some(plan) = self.box_cut_plan(doc, anchor, cursor) else {
            self.voxel_box_plan = None;
            self.report_tool(
                "Box cut: drag a rectangle on the ground — a click digs nothing.".to_string(),
            );
            return;
        };
        // The preview reads this, so it is refreshed here — where the document
        // is in hand and the ground has just been paged and probed.
        self.voxel_box_plan = Some(plan);
        let size = self.voxel_size_of(volume);
        // An UPPER bound on the excavation, not a promise: the pit's box may
        // span air and already-hollow rock, and only the samples that actually
        // held material become spoil. Quoted as "up to" for exactly that reason
        // — the committed readout after the release quotes the real number.
        let gross = plan.area_m2() * (plan.top_y - plan.floor_y);
        self.report_tool(format!(
            "Box cut {:.1} × {:.1} m, floor at {:.1} m ({:.1} m below grade) — up to {gross:.1} m³, \
             {} m voxels{}",
            plan.size_x_m,
            plan.size_z_m,
            plan.floor_y,
            self.voxel_tool.depth_m.max(0.0),
            size,
            self.spoil_note(),
        ));
    }

    /// The voxel scale a readout must quote: the **asset's**, never the
    /// component's (the `project_voxel` rule).
    fn voxel_size_of(&self, volume: Uuid) -> f64 {
        self.voxel_volumes
            .lock()
            .ok()
            .and_then(|v| v.slot(volume).map(|s| s.data.voxel_size_m()))
            .unwrap_or(inf_ecs::components::VoxelVolume::default().voxel_size_m)
    }

    /// The clause every dig readout ends with: where the soil is going.
    fn spoil_note(&self) -> String {
        match self.voxel_tool.spoil {
            SpoilMode::Off => String::new(),
            SpoilMode::Auto => ", spoil piled east of the cut".to_string(),
            SpoilMode::Site => match self.voxel_spoil_site {
                Some(p) => format!(", spoil piled at ({:.0}, {:.0})", p.x, p.z),
                None => ", spoil piled east of the cut (no site picked)".to_string(),
            },
        }
    }

    /// **Close a carve stroke the active tool can no longer finish** (P21.2
    /// audit). Returns `true` when an orphaned stroke was recorded.
    ///
    /// A stroke's dabs mutate the volume and the heightfield *live*, and only
    /// `finish_voxel` turns them into the undo entry that describes them. But
    /// `finish_voxel` is reached from the pump's `else if voxel` branch, which is
    /// gated on the **active tool** — so a tool switch arriving between two frames
    /// of a drag (the toolbar and the `tool.*` shortcuts both push one down a
    /// command channel, mid-gesture or not) left the stroke open forever: its cuts
    /// stayed in the world, saved like any other edit, and Ctrl+Z could not reach
    /// them. That is the un-undoable committed edit `a4e5844` ruled worse than any
    /// partial one.
    ///
    /// Committing rather than reverting, deliberately: the author *did* dig that
    /// rock and can see it, and reverting would need the document the tool switch
    /// does not have. One undo step, exactly as a mouse-up would have produced.
    ///
    /// Called unconditionally by the platform pump with the document in hand —
    /// `set_tool_mode` cannot do it itself (no `SceneDoc` on that seam), and a
    /// deferred flag would be a second piece of state saying what
    /// `tool_mode != Voxel && voxel_stroke.is_some()` already says.
    pub fn settle_orphaned_carve(&mut self, doc: &mut SceneDoc) -> bool {
        if self.voxel_stroke.is_none() && self.voxel_box_anchor.is_none() {
            return false;
        }
        // The condition is the pump's own `voxel` flag, derived here so the two
        // cannot disagree: 2D mode keeps Select regardless of the tool, so
        // switching projections mid-drag strands a stroke exactly as switching
        // tools does.
        let carve_branch_runs =
            self.tool_mode == ToolMode::Voxel && self.mode != ViewportMode::TwoD;
        if carve_branch_runs {
            return false;
        }
        // A box-cut drag has cut NOTHING (the pit is committed on release), so
        // it is dropped rather than settled — there is no edit to make undoable
        // and committing one would dig a pit the author abandoned. Only the
        // brush stroke is settled, and only it earns the sentence below.
        if self.voxel_stroke.is_none() {
            self.voxel_box_anchor = None;
            self.voxel_box_cursor = None;
            self.voxel_preview.clear();
            return false;
        }
        let recorded = self.finish_voxel(doc);
        self.reject_tool(Self::VOXEL_STROKE_SETTLED_ON_TOOL_SWITCH);
        recorded
    }

    /// What the author is told when a tool switch closed their carve for them.
    const VOXEL_STROKE_SETTLED_ON_TOOL_SWITCH: &'static str =
        "Carve: the tool changed while a stroke was still down, so the cut so far was committed \
         as one undo step. Ctrl+Z takes it back.";

    /// Cut the pending path: one swept shape per segment, all into a single
    /// stroke, so the whole run is one undo step.
    ///
    /// **One function for both path tools**, because they differ only in the
    /// section they sweep:
    ///
    /// * a **tunnel** sweeps a capsule — a round bore at depth. Capsules and not
    ///   spheres: a chain of spheres at waypoint spacing leaves gaps between the
    ///   beads, and the swept-sphere primitive exists in `inf_voxel::VoxelShape`
    ///   precisely for this.
    /// * a **trench** sweeps a rectangle from the surface down (P21.3), through
    ///   [`inf_editor_core::voxel_tool::trench_shapes`] — Ring 1, which owns the
    ///   miter allowance that stops a bend leaving un-cut ground on its outside.
    ///
    /// **Judged whole, then cut whole, then spoiled whole** — which is
    /// `SceneDoc::edit_dig`'s job, not this file's. All this does is page every
    /// segment's footprint, hand the chain over, and mirror the result into the
    /// render set. The atomicity rule lives in Ring 1 so every CI leg tests it;
    /// this module is `#[cfg(any(windows, macos))]`.
    fn commit_path(&mut self, doc: &mut SceneDoc, volume: Uuid) -> bool {
        let path = std::mem::take(&mut self.voxel_path);
        self.voxel_preview.clear();
        if path.len() < 2 {
            return false;
        }
        let version_before = doc.version();
        let radius_m = self.voxel_tool.radius_m.max(0.0);
        let ops: Vec<VoxelOp> = if self.voxel_tool.kind == VoxelToolKind::Trench {
            // **Page every leg's footprint BEFORE `trench_shapes` probes it**
            // (P21.3 audit, B3): the sky rule reads the ground through
            // `ground_surface_y`, whose `None` on a non-resident tile the
            // planner treats as "no ground here" — which would make a committed
            // trench's roof a function of camera residency.
            for seg in path.windows(2) {
                let mid = DVec2::new((seg[0].x + seg[1].x) * 0.5, (seg[0].z + seg[1].z) * 0.5);
                let half = (seg[1] - seg[0]).length() * 0.5 + radius_m;
                self.page_terrain_disc(doc, mid, half);
            }
            inf_editor_core::voxel_tool::trench_shapes(
                &path,
                radius_m,
                self.voxel_tool.depth_m.max(0.0),
                |x, z| doc.ground_surface_y(x, z),
            )
            .into_iter()
            .map(|s| self.voxel_shape_op(s))
            .collect()
        } else {
            inf_editor_core::voxel_tool::tunnel_shapes(&path, radius_m)
                .into_iter()
                .map(|s| self.voxel_shape_op(s))
                .collect()
        };
        if ops.is_empty() {
            return false;
        }
        // Page every segment BEFORE the verdict runs: it reads authored tiles
        // only, and an unpaged streamed footprint would answer "this reaches no
        // surface" for a tunnel that plainly breaks out of the hillside.
        for op in &ops {
            self.page_cut_footprint(doc, &op.shape);
        }
        // The refusal carries its OWN sentence. Quoting the inline-terrain one for
        // every empty answer is what this used to do, and it sent an author off to
        // convert a terrain when the real problem was a volume that never loaded.
        // The Auto spoil rule needs the ground east of the whole run paged.
        let size = self.voxel_size_of(volume);
        let mut lo = DVec3::splat(f64::INFINITY);
        let mut hi = DVec3::splat(f64::NEG_INFINITY);
        let mut budget = 0u64;
        for op in &ops {
            let (l, h) = op.shape.aabb_m(0.0);
            lo = lo.min(l);
            hi = hi.max(h);
            budget = budget.saturating_add(op.shape.affected_sample_count(size));
        }
        self.page_spoil_ground(doc, (lo, hi), budget);
        let tally = match doc.edit_dig(volume, &self.voxel_volumes, &ops, self.spoil_choice()) {
            Ok(tally) => tally,
            Err(refusal) => {
                self.reject_tool(refusal.message());
                return doc.version() != version_before;
            }
        };
        // Every projected terrain, not the segments' own lists: a tunnel can run
        // across several, `edit_dig` keeps its per-segment plan private,
        // and `after_terrain_edit` is a no-op on a terrain that is not streamed
        // and a cheap re-projection on one that is (a level has a handful).
        for i in 0..self.terrain_slots.len() {
            let guid = self.terrain_slots[i].guid;
            self.after_terrain_edit(doc, guid);
        }
        self.report_carve(volume, tally);
        !tally.is_noop()
    }

    /// Idle hover in voxel mode: show where the next cut lands, and — for the
    /// tunnel — how much path is pending and how to commit it.
    pub fn update_voxel_hover(&mut self, doc: &SceneDoc, view: &RenderView, px: u32, py: u32) {
        // The spoil-site mode has its own hover: the marker follows the cursor,
        // and the line says what the next click does. Reported unconditionally,
        // because the *silent* version of this mode is a tool where clicking
        // digs nothing and nothing explains why.
        if self.voxel_tool.pick_spoil_site {
            match self.voxel_surface_pick(doc, view, px, py) {
                Some(p) => self.refresh_voxel_preview(p),
                None => self.voxel_preview.clear(),
            }
            self.report_tool(match self.voxel_spoil_site {
                Some(p) => format!(
                    "Set spoil site: click to MOVE the heap from ({:.0}, {:.0}) — turn the button \
                     off to dig again.",
                    p.x, p.z
                ),
                None => Self::VOXEL_NO_SPOIL_SITE_HINT.to_string(),
            });
            return;
        }
        match self.voxel_surface_pick(doc, view, px, py) {
            Some(surface) => {
                let p = match self.voxel_tool.kind {
                    // A trench's preview follows the SURFACE it is cut from; the
                    // other three follow the sunk centre they cut at.
                    VoxelToolKind::Trench | VoxelToolKind::BoxCut => surface,
                    _ => self.voxel_sink(surface),
                };
                self.refresh_voxel_preview(p);
            }
            None => self.voxel_preview.clear(),
        }
        match self.voxel_tool.kind {
            VoxelToolKind::Tunnel | VoxelToolKind::Trench if !self.voxel_path.is_empty() => {
                let length: f64 = self
                    .voxel_path
                    .windows(2)
                    .map(|w| (w[1] - w[0]).length())
                    .sum();
                let what = if self.voxel_tool.kind == VoxelToolKind::Trench {
                    "Trench"
                } else {
                    "Tunnel"
                };
                self.report_tool(format!(
                    "{what}: {} waypoint(s), {length:.1} m so far, {:.1} m across — Ctrl+click to \
                     cut it{}",
                    self.voxel_path.len(),
                    self.voxel_tool.radius_m * 2.0,
                    self.spoil_note(),
                ));
            }
            VoxelToolKind::BoxCut => self.report_tool(format!(
                "Box cut: drag a rectangle on the ground to excavate it {:.1} m below grade{}",
                self.voxel_tool.depth_m.max(0.0),
                self.spoil_note(),
            )),
            _ => {}
        }
    }

    /// Rebuild the voxel tool's preview: the cut's silhouette at `center`, plus
    /// the tunnel path and the segment the next click would add.
    ///
    /// Three orthogonal circles rather than a ring on the ground: the cut is a
    /// **volume** and its depth is the parameter an author most needs to see, and
    /// a flat ring would draw a 4 m sphere and a 4 m disc identically.
    fn refresh_voxel_preview(&mut self, center: DVec3) {
        const SEGMENTS: usize = 24;
        self.voxel_preview.clear();
        let r = self.voxel_tool.radius_m.max(0.0);
        // **The preview draws the shape the commit CUTS** (P21.3 audit). The
        // brush's dig-to-grade mode cuts a column, not a ball, and drawing three
        // circles for it showed the author a sphere at depth while the release
        // dug a shaft to daylight. The shape comes from the same Ring-1 function
        // the dab uses, so the two cannot drift.
        if r > 0.0 && self.voxel_tool.kind == VoxelToolKind::Brush && self.voxel_tool.dig_to_depth {
            // `center` is the sunk point for the brush; the column is measured
            // from the surface above it.
            let surface = center + DVec3::new(0.0, self.voxel_tool.depth_m.max(0.0), 0.0);
            if let VoxelShape::Box {
                center: c,
                half_extents: h,
            } = inf_editor_core::voxel_tool::brush_dab_shape(
                surface,
                r,
                self.voxel_tool.depth_m,
                true,
            ) {
                push_box_wire(&mut self.voxel_preview, c - h, c + h);
            }
        } else if r > 0.0 {
            for axis in 0..3 {
                for i in 0..SEGMENTS {
                    let a = circle_point(center, r, axis, i, SEGMENTS);
                    let b = circle_point(center, r, axis, i + 1, SEGMENTS);
                    self.voxel_preview.push([a, b]);
                }
            }
        }
        if matches!(
            self.voxel_tool.kind,
            VoxelToolKind::Tunnel | VoxelToolKind::Trench
        ) {
            for seg in self.voxel_path.windows(2) {
                self.voxel_preview.push([seg[0], seg[1]]);
            }
            if let Some(&tail) = self.voxel_path.last() {
                self.voxel_preview.push([tail, center]);
            }
        }
        // The pit being dragged, as the box the release would cut.
        //
        // **Drawn from the resolved plan, not from the two picks** (P21.3
        // audit): the plan's floor is `depth` below the LOWEST ground it spans
        // and its top clears the HIGHEST, so a rubber-band drawn between the two
        // corner heights showed a different pit from the one being committed on
        // any ground that was not flat. `voxel_box_plan` is refreshed by
        // `report_box_cut`, which runs on the same frame as the drag update and
        // has the document (this seam does not).
        if let Some(plan) = self.voxel_box_plan {
            if let VoxelShape::Box {
                center: c,
                half_extents: h,
            } = plan.shape
            {
                push_box_wire(&mut self.voxel_preview, c - h, c + h);
            }
        }
        // The spoil marker: a small cross at the site, so an author can see where
        // the heap will land without digging to find out.
        if self.voxel_tool.spoil == SpoilMode::Site {
            if let Some(p) = self.voxel_spoil_site {
                const ARM: f64 = 1.0;
                for d in [DVec3::X, DVec3::Y, DVec3::Z] {
                    self.voxel_preview.push([p - d * ARM, p + d * ARM]);
                }
            }
        }
    }

    /// Publish a carve's running totals on the status seam — the tool's live
    /// measurement readout, the twin of the lake drag's coverage line.
    ///
    /// Quotes cubic metres as well as sample counts because a volume is what an
    /// author is actually digging, and — once spoil is on — it is the number the
    /// conservation ledger balances.
    ///
    /// **The excavation line quotes both sides of that ledger** (P21.3): m³ out
    /// of the hole, m³ into the heap, and the per-material breakdown, because
    /// "did the soil actually go somewhere" is the question an author asks about
    /// a dig and the one number that proves it is the pair.
    fn report_carve(&mut self, volume: Uuid, tally: inf_editor_core::scene::undo::CarveTally) {
        if tally.is_noop() {
            return;
        }
        // The ASSET's scale, never the component's — the same rule `project_voxel`
        // obeys, and reading the component here would quote a volume computed
        // against one cell size for geometry cut at another. The volume is passed
        // in rather than read off the in-flight stroke, because the path tools
        // report *after* committing (and therefore after the stroke is gone),
        // which is exactly when a stroke-derived scale would silently fall back
        // to a default that is right for most assets and wrong for the rest.
        let size = self.voxel_size_of(volume);
        if tally.spoiled() > 0 || self.voxel_tool.spoil != SpoilMode::Off {
            let breakdown = tally.material_breakdown();
            let by_layer = if breakdown.is_empty() {
                String::new()
            } else {
                format!(" [{breakdown}]")
            };
            // `conserved()` and not "the numbers look close": the identity is
            // integer and per material, and a readout that said "balanced" for a
            // near miss would be the one place the gate is not the gate.
            let ledger = if tally.conserved() {
                "balanced"
            } else {
                "DISCARDED"
            };
            self.report_tool(format!(
                "Excavate: removed {:.2} m³ ({} voxels){by_layer}, spoiled {:.2} m³ ({} voxels) — \
                 {ledger}; {} cave-mouth sample(s) opened",
                tally.removed_m3(size),
                tally.carved,
                tally.spoiled_m3(size),
                tally.spoiled(),
                tally.holes,
            ));
            return;
        }
        let m3 = tally.net_removed_m3(size);
        let verb = if m3 >= 0.0 { "removed" } else { "added" };
        self.report_tool(format!(
            "Carve: {verb} {:.2} m³ ({} voxels), {} cave-mouth sample(s) opened/closed",
            m3.abs(),
            tally.touched,
            tally.holes,
        ));
    }

    /// Adopt an externally-created voxel working set, replacing the private one
    /// [`new`](Self::new) made — called by each platform's `spawn` immediately
    /// after construction and never again.
    ///
    /// **Why the store is created outside the host** (P21.2 deliverable 7): the
    /// carved chunks are the thing Ctrl+S writes into the `.inf_voxel`, and the
    /// save path is Ring 2's `commands/scene.rs`, which has a `ViewportHandle`
    /// and no host — the host lives on the render thread behind a command
    /// channel. A terrain's working set is reachable because it is *in the
    /// document*; a volume's cannot be (scene schema v19 is frozen and carries no
    /// chunks), so the handle has to be able to hand the store out. `spawn`
    /// therefore makes it, keeps a clone on the handle, and gives this one to the
    /// host. The `Arc<Mutex<…>>` is the same object either way, and the lock
    /// order is unchanged: **document first, volumes second.**
    ///
    /// The alternative — a `Cmd` that flushed edits on the render thread — would
    /// put a whole-payload asset rewrite between two frames and hand the report
    /// back across a channel that has no reply path. The other one — moving the
    /// store into `SceneDoc` — is refused where the type is defined.
    #[must_use]
    pub fn with_voxel_volumes(
        mut self,
        volumes: inf_editor_core::voxel_store::SharedVoxelVolumes,
    ) -> Self {
        self.voxel_volumes = volumes;
        self
    }

    /// Adopt the handle's shared Simulate fracture states (P22.3) — the
    /// `with_voxel_volumes` twin, so Ring 2 and this host hold clones of one
    /// `Arc` and a publish reaches the projection.
    pub fn with_fractures(mut self, fractures: inf_editor_core::simulate::SharedFractures) -> Self {
        self.fractures = fractures;
        self
    }

    /// Reopen every loaded volume's `.inf_voxel` index in place — the twin of
    /// [`reload_terrain_stores`](Self::reload_terrain_stores), pushed by the save
    /// path once it has folded carve edits back into the assets.
    ///
    /// `refresh_index` and **not** `clear`: the loaded chunks are the working set
    /// the author is looking at, a save does not change their contents, and
    /// dropping them would blink every cave in the level at Ctrl+S. What it does
    /// change is which GUIDs resolve — a carve saved into a *new* `.inf_voxel` is
    /// only findable after a re-walk — and it forgets past resolution failures,
    /// which is exactly the state a just-written asset was in a moment ago.
    pub fn reload_voxel_stores(&mut self) {
        if let Ok(mut v) = self.voxel_volumes.lock() {
            v.refresh_index();
        }
        self.synced_version = None; // re-project against the refreshed index
    }

    /// **The defensive advisory** (P21.2): a terrain carrying holes it cannot
    /// save.
    ///
    /// The carve tools refuse to create that state, so reaching it means the
    /// document arrived in it — an older session, a hand-edited level, a terrain
    /// converted back to inline after a carve. Saving would seal every cave mouth
    /// in it, so the author is told **before** that happens rather than after a
    /// reload. One report per terrain per host, because this runs on every
    /// projection and a per-frame status event is an unusable seam rather than a
    /// louder one.
    ///
    /// The *rule* is `inf_voxel::inline_hole_advisory` (Ring 0, unit-tested on all
    /// three OSes); this is only the call site. The save path is the other place
    /// it belongs, and that half is `voxel_edit`'s to wire.
    fn check_inline_holes(&mut self, doc: &SceneDoc) {
        for &guid in doc.order() {
            if self.voxel_hole_warned.contains(&guid) {
                continue;
            }
            let Some((data, _)) = doc.terrain_data_and_origin(guid) else {
                continue;
            };
            let backed = doc.terrain_asset_of(guid).is_some();
            let Some(note) = inf_voxel::inline_hole_advisory(data, backed) else {
                continue;
            };
            self.voxel_hole_warned.insert(guid);
            self.reject_tool(&note.message());
        }
    }

    /// The world height foliage lands on at world XZ `p`: the topmost terrain
    /// surface covering it, else `0.0` (the ground plane) — the pre-P16.6 answer
    /// for a world with no terrain, unchanged.
    fn foliage_surface_height(&self, doc: &SceneDoc, p: DVec2) -> f64 {
        topmost_surface(&self.terrain_probes(doc, None), p)
            .map(|(_, y)| y)
            .unwrap_or(0.0)
    }

    /// Hover update (idle Foliage mode): move the brush ring to the cursor.
    pub fn update_foliage_hover(&mut self, doc: &SceneDoc, view: &RenderView, px: u32, py: u32) {
        let center = self.foliage_center(doc, view, px, py);
        self.refresh_foliage_ring(doc, center);
    }

    /// Begin a foliage scatter stroke. Resolves the target Foliage entity — the
    /// first SELECTED foliage entity, or a new one auto-created at the origin and
    /// selected — inside one undo transaction, then lays the first tick. Returns
    /// `true` (a stroke always starts; an empty result just records nothing).
    pub fn begin_foliage(
        &mut self,
        doc: &mut SceneDoc,
        view: &RenderView,
        px: u32,
        py: u32,
    ) -> bool {
        let target = doc
            .selection()
            .iter()
            .copied()
            .find(|g| doc.has_foliage(*g));
        // One undo entry for the whole stroke (auto-create + scatter, or scatter).
        doc.begin_transaction("Paint Foliage");
        let guid = match target {
            Some(g) => g,
            None => {
                let g = doc.edit_create(SpawnKind::Foliage, "Foliage", None);
                doc.select(&[g], false);
                g
            }
        };
        let origin = doc.foliage_origin(guid).unwrap_or(DVec3::ZERO);
        let original = doc.foliage_instances(guid).unwrap_or_default();
        let stroke_seq = self.foliage_stroke_seq;
        self.foliage_stroke_seq = self.foliage_stroke_seq.wrapping_add(1);
        let positions = original
            .iter()
            .map(|i| DVec2::new(i.position.x, i.position.z))
            .collect();
        self.foliage_drag = Some(FoliageDrag {
            guid,
            erase: self.foliage.erase,
            stroke_seq,
            next_sample: 0,
            origin,
            positions,
            added: Vec::new(),
            original,
            removed: BTreeSet::new(),
        });
        let center = self.foliage_center(doc, view, px, py);
        self.foliage_dab(doc, center);
        self.refresh_foliage_ring(doc, center);
        true
    }

    /// Continue the stroke: lay a tick at the current cursor. (Per-tick placement;
    /// path resampling for very fast strokes is a documented follow-up — min-
    /// spacing rejection keeps a held cursor from stacking.)
    pub fn update_foliage(&mut self, doc: &mut SceneDoc, view: &RenderView, px: u32, py: u32) {
        if self.foliage_drag.is_none() {
            return;
        }
        let center = self.foliage_center(doc, view, px, py);
        self.foliage_dab(doc, center);
        self.refresh_foliage_ring(doc, center);
    }

    /// One brush tick: place (or erase) instances around `center` (world). Live-
    /// mutates the target component so the scatter renders immediately.
    fn foliage_dab(&mut self, doc: &mut SceneDoc, center: DVec3) {
        let Some(drag) = self.foliage_drag.as_ref() else {
            return;
        };
        let (guid, erase, origin, stroke_seq, base) = (
            drag.guid,
            drag.erase,
            drag.origin,
            drag.stroke_seq,
            drag.next_sample,
        );
        let s = self.foliage;
        let center_xz = DVec2::new(center.x, center.z);

        if erase {
            let r2 = s.radius * s.radius;
            let mut newly: Vec<usize> = Vec::new();
            for (i, inst) in drag.original.iter().enumerate() {
                if drag.removed.contains(&i) {
                    continue;
                }
                let wx = origin.x + inst.position.x;
                let wz = origin.z + inst.position.z;
                let d2 = (wx - center_xz.x).powi(2) + (wz - center_xz.y).powi(2);
                if d2 <= r2 {
                    newly.push(i);
                }
            }
            if newly.is_empty() {
                return;
            }
            let kept = {
                let d = self.foliage_drag.as_mut().unwrap();
                for i in newly {
                    d.removed.insert(i);
                }
                d.original
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| !d.removed.contains(i))
                    .map(|(_, x)| *x)
                    .collect::<Vec<_>>()
            };
            doc.foliage_set_instances(guid, kept);
            return;
        }

        // Place: target count = density × brush area, capped per tick.
        let area = std::f64::consts::PI * s.radius * s.radius;
        let target =
            ((s.density * area).round() as i64).clamp(0, FOLIAGE_MAX_PER_TICK as i64) as u32;
        if target == 0 {
            return;
        }
        let cands = foliage_samples(
            center_xz,
            s.radius,
            target,
            s.seed,
            stroke_seq,
            base,
            s.scale_jitter,
            s.kind,
        );
        if let Some(d) = self.foliage_drag.as_mut() {
            d.next_sample += target as u64;
        }
        // Lift each candidate onto the topmost terrain covering it (P16.6), else
        // the ground plane (scoped immutable doc borrow).
        let heights: Vec<f64> = cands
            .iter()
            .map(|c| self.foliage_surface_height(doc, c.pos_xz))
            .collect();
        let ms2 = foliage_min_spacing(s.density).powi(2);
        let mut accepted: Vec<FoliageInstance> = Vec::new();
        {
            let d = self.foliage_drag.as_mut().unwrap();
            for (c, y) in cands.iter().zip(heights) {
                let local = DVec3::new(c.pos_xz.x - origin.x, y - origin.y, c.pos_xz.y - origin.z);
                let lxz = DVec2::new(local.x, local.z);
                if d.positions
                    .iter()
                    .any(|p| (p.x - lxz.x).powi(2) + (p.y - lxz.y).powi(2) < ms2)
                {
                    continue;
                }
                let inst = FoliageInstance {
                    position: Vec3d::new(local.x, local.y, local.z),
                    rotation: Vec3d::new(0.0, c.yaw_deg, 0.0),
                    scale: c.scale,
                    kind: c.kind,
                };
                d.positions.push(lxz);
                d.added.push(inst);
                accepted.push(inst);
            }
        }
        if !accepted.is_empty() {
            doc.foliage_append(guid, &accepted);
        }
    }

    /// Finish the stroke: commit ONE `PaintFoliage` undo step (added or removed)
    /// and close the transaction opened in [`Self::begin_foliage`]. Returns `true`
    /// if the stroke changed anything (so the caller emits `WorldChanged`).
    /// **Close a foliage stroke the active tool can no longer finish** (P21.3
    /// audit) — the third settler, and the one that also carries a transaction.
    ///
    /// `begin_foliage` opens `"Paint Foliage"` and `finish_foliage` closes it,
    /// both inside the pump's tool-gated `foliage` branch. A tool switch between
    /// two frames of a drag therefore stranded *two* things at once: the
    /// scattered instances (in the world, with no undo entry) and the
    /// transaction itself (which then killed Ctrl+Z for the session — see
    /// [`settle_orphaned_transaction`](Self::settle_orphaned_transaction)).
    /// Settling here fixes both, because `finish_foliage` commits the entry and
    /// closes the transaction on its way out.
    pub fn settle_orphaned_foliage(&mut self, doc: &mut SceneDoc) -> bool {
        if self.foliage_drag.is_none() {
            return false;
        }
        // The pump's own `foliage` flag, derived here so the two cannot disagree.
        let foliage_branch_runs =
            self.tool_mode == ToolMode::Foliage && self.mode != ViewportMode::TwoD;
        if foliage_branch_runs {
            return false;
        }
        let recorded = self.finish_foliage(doc);
        self.reject_tool(Self::SCULPT_STROKE_SETTLED_ON_TOOL_SWITCH);
        recorded
    }

    pub fn finish_foliage(&mut self, doc: &mut SceneDoc) -> bool {
        let Some(drag) = self.foliage_drag.take() else {
            return false;
        };
        let changed = if drag.erase {
            let removed: Vec<(usize, FoliageInstance)> = drag
                .removed
                .iter()
                .map(|&i| (i, drag.original[i]))
                .collect();
            doc.edit_commit_foliage(drag.guid, Vec::new(), removed)
        } else {
            doc.edit_commit_foliage(drag.guid, drag.added, Vec::new())
        };
        // Always close the transaction begin_foliage opened (a Create may have
        // been recorded even when the scatter itself was empty).
        doc.commit_transaction();
        changed
    }
}

/// Effective op after a Ctrl modifier: Ctrl temporarily inverts Raise↔Lower (UE
/// convention); other ops are unaffected.
fn effective_op(op: SculptOp, ctrl: bool) -> SculptOp {
    match (op, ctrl) {
        (SculptOp::Raise, true) => SculptOp::Lower,
        (SculptOp::Lower, true) => SculptOp::Raise,
        (op, _) => op,
    }
}

/// Build the `inf_terrain` brush op + params for one dab from the toolbar
/// settings, filling in the op-specific parameters the flat UI enum omits.
fn falloff_of(f: SculptFalloff) -> Falloff {
    match f {
        SculptFalloff::Smooth => Falloff::Smooth,
        SculptFalloff::Linear => Falloff::Linear,
        SculptFalloff::Sphere => Falloff::Sphere,
        SculptFalloff::Sharp => Falloff::Sharp,
    }
}

/// The biome a dab actually writes: `Ctrl` erases (writes the reserved
/// *unassigned* id) — the biome twin of Ctrl flipping Raise↔Lower.
fn effective_biome(s: &BiomeSettings, ctrl: bool) -> u8 {
    if ctrl {
        inf_terrain::UNASSIGNED_BIOME
    } else {
        s.biome
    }
}

/// Brush params for a biome dab (P19.2). `strength` is **not** a blend fraction
/// here — it selects which falloff contour the hard boundary lands on; see
/// `inf_terrain::biomepaint`.
fn biome_params(s: &BiomeSettings, center: DVec2) -> BrushParams {
    BrushParams {
        center,
        radius: s.radius,
        strength: s.strength,
        falloff: falloff_of(s.falloff),
    }
}

/// Brush params for a splat-paint dab (P10.4): `strength` is the per-dab flow
/// rate toward the target layer, `falloff` shapes it across the radius.
fn paint_params(s: &SculptSettings, center: DVec2) -> BrushParams {
    BrushParams {
        center,
        radius: s.radius,
        strength: s.strength,
        falloff: falloff_of(s.falloff),
    }
}

fn brush_of(
    op: SculptOp,
    s: &SculptSettings,
    center: DVec2,
    flatten_height: f64,
) -> (BrushOp, BrushParams) {
    let params = BrushParams {
        center,
        radius: s.radius,
        strength: s.strength,
        falloff: falloff_of(s.falloff),
    };
    let brush = match op {
        SculptOp::Raise => BrushOp::Raise,
        SculptOp::Lower => BrushOp::Lower,
        SculptOp::Smooth => BrushOp::Smooth { iterations: 1 },
        SculptOp::Flatten => BrushOp::Flatten {
            target: FlattenTarget::PickedHeight(flatten_height),
        },
        SculptOp::Noise => BrushOp::Noise {
            seed: 0x5EED_1234,
            frequency: 0.05,
            octaves: 4,
            amplitude: s.strength,
        },
        // Paint is routed to the splat path before `brush_of` is reached; map it
        // to a no-op-ish Raise for totality (never actually applied).
        SculptOp::Paint => BrushOp::Raise,
    };
    (brush, params)
}

/// The brush-ring colour for an op (green raise / red lower / blue smooth /
/// yellow flatten / violet noise).
fn op_color(op: SculptOp) -> [f32; 4] {
    match op {
        SculptOp::Raise => [0.35, 0.90, 0.45, 1.0],
        SculptOp::Lower => [0.95, 0.45, 0.35, 1.0],
        SculptOp::Smooth => [0.40, 0.70, 1.00, 1.0],
        SculptOp::Flatten => [0.95, 0.85, 0.35, 1.0],
        SculptOp::Noise => [0.75, 0.50, 0.95, 1.0],
        // Fallback only — the ring is normally recoloured to the target layer's
        // albedo (see `refresh_ring`).
        SculptOp::Paint => [0.90, 0.90, 0.90, 1.0],
    }
}

/// Sample a closed ring of world-space points around `center` (terrain-local XZ)
/// at `radius`, each lifted to the terrain surface height there (falling back to
/// the centre height over holes), then shifted by the terrain's world
/// translation. Connect consecutive points (and last→first) to stroke the ring.
fn build_ring(data: &TerrainData, translation: DVec3, center: DVec2, radius: f64) -> Vec<DVec3> {
    const SEGMENTS: u32 = 32;
    let base_h = data.height_at(center).unwrap_or(0.0);
    (0..SEGMENTS)
        .map(|i| {
            let a = std::f64::consts::TAU * (i as f64) / (SEGMENTS as f64);
            let p = center + DVec2::new(radius * a.cos(), radius * a.sin());
            let h = data.height_at(p).unwrap_or(base_h);
            translation + DVec3::new(p.x, h, p.y)
        })
        .collect()
}

impl EngineHost {
    /// Render one frame for the resolved [`RenderView`] (the platform loop
    /// builds it from whichever camera is active and rebases the floating origin
    /// first). Handles crash-safe device-lost recovery internally; only errors
    /// that survive a full stack rebuild are returned.
    ///
    /// Returns `Ok(true)` when a frame was presented and `Ok(false)` when the
    /// swapchain had no image to acquire (surface occluded/minimized/hidden) so
    /// nothing was drawn. The caller uses this to pace itself: a presented FIFO
    /// frame blocks at vsync, but a non-present must be throttled by the loop or
    /// it busy-spins the CPU (M3).
    pub fn render_frame(&mut self, view: &RenderView) -> Result<bool, String> {
        // Remember the camera eye for PCG draw-distance culling on the next
        // projection (see `last_eye_world`).
        self.last_eye_world = view.eye_world;
        // THE RENDER-SYNC POINT (P16.3b2): advance every streamed terrain's
        // camera-driven cut exactly once per frame, here. Unlike the document
        // projection this is *not* version-gated — the cut follows the camera, and
        // the camera moves without the document changing. Nothing it does is
        // visible to the document, which is the whole point.
        self.sync_streamed_terrain();
        self.sync_streamed_voxels();
        if self.gpu.is_lost() {
            tracing::warn!("inf-viewport: device lost — rebuilding GPU stack");
            let (w, h) = self.chain.requested_size();
            // Read the shading mode off the DYING renderer, which is the only
            // place it is stored, before it is replaced by a fresh one that
            // starts at `Lit`. See `reset_device_scoped_state`.
            let view_mode = self.renderer.view_mode();
            // Drop the dead device's swapchain BEFORE a second one is created on
            // the same window: `build_gpu_stack` makes a fresh `Instance` +
            // `Surface`, and two live swapchains on one HWND/CAMetalLayer is
            // driver-dependent state nobody gains anything from. See
            // `SurfaceChain::release`.
            self.chain.release();
            let (gpu, chain, renderer) = Self::build_gpu_stack(self.target, w, h)?;
            self.gpu = gpu;
            self.chain = chain;
            self.renderer = renderer;
            // The picker holds its own device-scoped GPU resources (pipeline,
            // ID-buffer target, readback buffer) created against the OLD device.
            // Rebuild it on the fresh device too — otherwise the next pick (which
            // runs in the interaction block, OUTSIDE the render catch_unwind)
            // hits a device-mismatch validation error and kills the thread with
            // the scene mutex poisoned (H1).
            self.picker = Picker::new(&self.gpu);
            self.reset_device_scoped_state(view_mode);
        }

        // Per-frame debug primitives: world-origin axes tripod, plus the
        // transform gizmo at the selection center (screen-constant size). In 2D
        // the gizmo shows only the sprite-plane handles (X/Y, Z ring).
        self.scene.debug.clear();
        self.scene
            .debug
            .axes(self.origin.to_render(glam::DVec3::ZERO), 1.0);
        if let Some(center) = self.selection_center() {
            let origin_local = self.origin.to_render(center);
            let size = self.gizmo_size(view, origin_local);
            let active = self.gizmo_drag.map(|d| d.axis);
            // Draw with the in-flight drag's fixed basis, else recompute from the
            // current selection (so idle Local-mode handles track selection).
            let basis = self
                .gizmo_drag
                .map(|d| d.basis)
                .unwrap_or_else(|| self.gizmo_basis());
            gizmo::build_geometry(
                &mut self.scene.debug,
                self.gizmo_mode,
                origin_local,
                basis,
                size,
                active,
                view.ortho.is_some(),
            );
        }
        // 2D + 3D collider outlines for the current selection (P8.3b / P9.1).
        for guid in &self.selected_guids {
            if let Some(cd) = self.collider_outlines.get(guid) {
                draw_collider_outline(&mut self.scene.debug, &self.origin, cd);
            }
            if let Some(cd) = self.collider_outlines_3d.get(guid) {
                draw_collider_outline_3d(&mut self.scene.debug, &self.origin, cd);
            }
            if let Some(jd) = self.joint_lines.get(guid) {
                draw_joint_lines(&mut self.scene.debug, &self.origin, jd);
            }
            if let Some(sd) = self.spot_lights.get(guid) {
                draw_spot_cone(&mut self.scene.debug, &self.origin, sd);
            }
        }

        // Volume wireframes (E-P4) draw ALWAYS in the volume's tint (invisible in
        // PIE — this is the only editor cue); a selected volume brightens.
        for (guid, vd) in &self.volume_outlines {
            let selected = self.selected_guids.contains(guid);
            draw_volume_outline(&mut self.scene.debug, &self.origin, vd, selected);
        }

        // **The player start** (wave EDIT1, clause 2): a ring on the ground with
        // a mast and a forward tick, so an author can see at a glance where Play
        // will put them without selecting anything. Drawn ALWAYS, like the
        // splines above and unlike the tool previews below — it is a property of
        // the level, not of a mode — and, like every other primitive in this
        // block, it is an editor-only debug line that is never projected, never
        // persisted and invisible in PIE.
        if let Some((start, _)) = self.player_start {
            const START_COLOR: [f32; 4] = [1.0, 0.78, 0.25, 1.0];
            const START_RADIUS_M: f64 = 0.45; // a character's own footprint
            const START_MAST_M: f64 = 1.8; // eye height, so it reads at street scale
            const START_SEGMENTS: usize = 24;
            let mut prev: Option<DVec3> = None;
            for i in 0..=START_SEGMENTS {
                let a = i as f64 / START_SEGMENTS as f64 * std::f64::consts::TAU;
                let p = start + DVec3::new(a.cos() * START_RADIUS_M, 0.0, a.sin() * START_RADIUS_M);
                if let Some(q) = prev {
                    self.scene.debug.line(
                        self.origin.to_render(q),
                        self.origin.to_render(p),
                        START_COLOR,
                    );
                }
                prev = Some(p);
            }
            self.scene.debug.line(
                self.origin.to_render(start),
                self.origin.to_render(start + DVec3::Y * START_MAST_M),
                START_COLOR,
            );
        }

        // Spline polylines (E-P5) draw ALWAYS in a neutral cyan; a selected
        // spline additionally shows a brighter 3-axis cross at each control point.
        for (guid, sd) in &self.spline_polylines {
            const SPLINE_COLOR: [f32; 4] = [0.25, 0.85, 0.95, 1.0];
            const SPLINE_MARKER: [f32; 4] = [0.6, 1.0, 1.0, 1.0];
            const MARKER_ARM: f64 = 0.15; // world-metre half-length of a cross arm
            for pair in sd.line.windows(2) {
                let a = self.origin.to_render(pair[0]);
                let b = self.origin.to_render(pair[1]);
                self.scene.debug.line(a, b, SPLINE_COLOR);
            }
            if self.selected_guids.contains(guid) {
                for &p in &sd.control {
                    for axis in [DVec3::X, DVec3::Y, DVec3::Z] {
                        let a = self.origin.to_render(p - axis * MARKER_ARM);
                        let b = self.origin.to_render(p + axis * MARKER_ARM);
                        self.scene.debug.line(a, b, SPLINE_MARKER);
                    }
                }
            }
        }

        // Water tool preview (P20.4): the river segment the next click would add,
        // or the lake rectangle plus the **waterline** — where the still-water
        // level meets the ground inside it. Editor-only, like every other debug
        // primitive here; nothing in it is persisted or projected.
        if self.tool_mode == ToolMode::Water {
            const WATER_PREVIEW: [f32; 4] = [0.35, 0.75, 1.0, 1.0];
            for [a, b] in &self.water_preview {
                let a = self.origin.to_render(*a);
                let b = self.origin.to_render(*b);
                self.scene.debug.line(a, b, WATER_PREVIEW);
            }
        }

        // Voxel tool preview (P21.2): the cut's three orthogonal silhouette
        // circles at the point under the cursor, plus the pending tunnel path and
        // the segment the next click would add. Editor-only debug lines; a cut is
        // never previewed by mutating the field.
        if self.tool_mode == ToolMode::Voxel {
            const VOXEL_PREVIEW: [f32; 4] = [1.0, 0.62, 0.28, 1.0];
            for [a, b] in &self.voxel_preview {
                let a = self.origin.to_render(*a);
                let b = self.origin.to_render(*b);
                self.scene.debug.line(a, b, VOXEL_PREVIEW);
            }
        }

        // Sculpt / foliage brush ring: a closed loop following the terrain height
        // under the cursor, coloured by the active op (Sculpt) or green (Foliage).
        if matches!(self.tool_mode, ToolMode::Sculpt | ToolMode::Foliage)
            && self.sculpt_ring.len() >= 2
        {
            let n = self.sculpt_ring.len();
            for i in 0..n {
                let a = self.origin.to_render(self.sculpt_ring[i]);
                let b = self.origin.to_render(self.sculpt_ring[(i + 1) % n]);
                self.scene.debug.line(a, b, self.sculpt_ring_color);
            }
        }

        let Some(frame) = self.chain.acquire(&self.gpu) else {
            return Ok(false); // transient (occluded/timeout) — nothing presented
        };
        let out_view = self.chain.target_view(&frame);
        self.renderer.render(
            &self.gpu,
            &self.scene,
            view,
            &out_view,
            self.chain.configured_size(),
        );
        self.gpu.queue.present(frame);
        Ok(true)
    }
}

/// **How a crowd agent casts** — the one mapping from the sim-LOD tier to the
/// renderer's caster mode (wave NPC1b).
///
/// `Full` is 32 m, which is where a viewer can read that an arm moved, so those
/// agents keep a real skinned caster and get the EXACT posed bound
/// (`SkinnedShadow::Posed`) rather than the 50 % `SKINNED_POSE_MARGIN` inflation
/// — a tighter caster sphere is fewer invalidated shadow pages, which is what an
/// animating character costs the page cache. Everything further out casts a box
/// out of the crowd's single shared proxy group, because `VSM_MAX_GROUPS` is
/// 1 024 and a skinned caster is one group each.
///
/// An entity with no `CrowdAgent` — every hero, every garment, every hair ribbon,
/// every authored character in this tree — answers `BindSphere`, which is exactly
/// what it was before this wave.
///
/// **And past `Near` it casts nothing** (island wave NPC1e): the proxy answered
/// the group ceiling and not the page churn — 968 boxes walking through Harbour
/// City dirtied 168.6 shadow pages a frame against the island's own 56.3 — so
/// `CrowdTier::casts_shadow` is the radius NPC1b's carried item 4 asked for.
fn crowd_shadow(agent: Option<inf_ecs::crowd::CrowdAgent>) -> inf_render::SkinnedShadow {
    match agent {
        None => inf_render::SkinnedShadow::BindSphere,
        Some(a) if a.tier.skinned_caster() => inf_render::SkinnedShadow::Posed,
        Some(a) if a.tier.casts_shadow() => inf_render::SkinnedShadow::Proxy,
        Some(_) => inf_render::SkinnedShadow::None,
    }
}

/// **A simulated garment, as a skinned draw** (P24.4).
///
/// The sim's `inf_ecs::cloth` store already holds this wearer's particle
/// positions *and* the garment's triangle list, so a projector needs no asset
/// store and no mesh lookup at all: it reads the sim world, the way it reads an
/// evaluated pose, and builds the vertex stream through the one Ring-0 function
/// `inf_render::deformed_skinned_mesh`.
///
/// It rides the **skinned** path rather than a pass of its own, because that pass
/// applies its palette before the model matrix — so a model-space garment with a
/// one-entry identity palette lands exactly where the sim put it. No new node, no
/// new shader, no golden re-blessed.
///
/// The instance carries `ID_NONE`, so a garment is not separately pickable:
/// clicking a coat selects nothing and clicking the character selects the
/// character, which is the v1 answer and is ledgered. Its geometry is a fresh
/// `Arc` every projection, which is correct for a surface that moves every step
/// and is the reason a garment costs one vertex-buffer upload per frame.
///
/// Emits nothing for a wearer the sim is not simulating, and nothing for a
/// garment whose triangles all dropped out — an empty draw is a draw call for no
/// pixels.
///
/// **MIRROR** of the other host's `project_cloth` — keep the two byte-identical,
/// **this doc block included** (the P21.2 lesson: the mirror gate compares the
/// comment too). Side-neutral wording on purpose.
fn project_cloth(
    scene: &mut inf_render::RenderScene,
    world: &inf_ecs::EcsWorld,
    guid: uuid::Uuid,
    translation: glam::DVec3,
    rotation: glam::Quat,
    scale: glam::Vec3,
) {
    let Some(live) = inf_ecs::cloth::live_cloth(world, guid) else {
        return;
    };
    let mesh = inf_render::deformed_skinned_mesh(&live.state.x, &live.state.indices);
    if mesh.indices.len() < 3 {
        return;
    }
    scene.skinned_meshes.push(std::sync::Arc::new(mesh));
    let slot = scene.skinned_meshes.len() - 1;
    scene.skinned.push(inf_render::SkinnedInstance {
        vt: Default::default(),
        translation,
        rotation,
        scale,
        color: inf_render::CLOTH_TINT,
        metallic: 0.0,
        roughness: 0.85,
        emissive: [0.0; 3],
        id: inf_render::ID_NONE,
        mesh: slot,
        palette: inf_render::identity_palette(),
        shadow: inf_render::SkinnedShadow::BindSphere,
    });
}

/// **A simulated hairstyle, as a skinned draw** (P24.4).
/// The twin of `project_cloth`, and deliberately the same shape: the sim's
/// `inf_ecs::hair` store already holds this wearer's ribbon positions and index
/// list, both rebuilt inside the fixed step, so a projector reads the sim world
/// and builds nothing of its own. The ribbons are computed there rather than here
/// precisely because there are two projectors and one fixed step.
///
/// Rides the **skinned** path with a one-entry identity palette, carries
/// `ID_NONE`, and is a fresh `Arc` per projection — see `project_cloth` for each
/// of those three, which hold here for the same reasons.
///
/// **MIRROR** of the other host's `project_hair` — keep the two byte-identical,
/// **this doc block included**. Side-neutral wording on purpose.
fn project_hair(
    scene: &mut inf_render::RenderScene,
    world: &inf_ecs::EcsWorld,
    guid: uuid::Uuid,
    translation: glam::DVec3,
    rotation: glam::Quat,
    scale: glam::Vec3,
) {
    let Some(live) = inf_ecs::hair::live_hair(world, guid) else {
        return;
    };
    let mesh = inf_render::deformed_skinned_mesh(&live.ribbon_positions, &live.ribbon_indices);
    if mesh.indices.len() < 3 {
        return;
    }
    scene.skinned_meshes.push(std::sync::Arc::new(mesh));
    let slot = scene.skinned_meshes.len() - 1;
    scene.skinned.push(inf_render::SkinnedInstance {
        vt: Default::default(),
        translation,
        rotation,
        scale,
        color: inf_render::HAIR_TINT,
        metallic: 0.0,
        roughness: 0.45,
        emissive: [0.0; 3],
        id: inf_render::ID_NONE,
        mesh: slot,
        palette: inf_render::identity_palette(),
        shadow: inf_render::SkinnedShadow::BindSphere,
    });
}
#[cfg(test)]
mod curve_focus_tests {
    use super::curve_focus;
    use glam::DVec3;

    /// **A river is framed by its whole run, not by a point** (audit ROAD1).
    ///
    /// Wave ROAD1 could not photograph the river it had just measured: `Focus
    /// Selection` read a river's `Transform`, which is the identity, so three
    /// attempts landed on the world origin. The entity's geometry is in its
    /// `Spline`, and this is the arithmetic that reads it.
    ///
    /// Falsification: return the first point instead of the box centre and the
    /// centre assertion reds; drop the `max(4.0)` and the puddle reds.
    #[test]
    fn a_long_thin_run_is_framed_by_its_length() {
        // 400 m of river running +X, wandering 6 m in Z and dropping 10 m.
        let line: Vec<DVec3> = (0..=40)
            .map(|i| {
                let t = f64::from(i) / 40.0;
                DVec3::new(100.0 + 400.0 * t, 50.0 - 10.0 * t, -20.0 + 6.0 * t)
            })
            .collect();
        let (centre, radius) = curve_focus(&line).expect("a curve has a focus");
        assert!((centre.x - 300.0).abs() < 1.0e-9, "{centre:?}");
        assert!((centre.y - 45.0).abs() < 1.0e-9, "{centre:?}");
        assert!((centre.z - (-17.0)).abs() < 1.0e-9, "{centre:?}");
        // Half the diagonal of a 400 x 10 x 6 box.
        let want = (400.0f64 * 400.0 + 10.0 * 10.0 + 6.0 * 6.0).sqrt() * 0.5;
        assert!((radius - want).abs() < 1.0e-9, "{radius} vs {want}");
        assert!(
            radius > 190.0,
            "a 400 m river framed at {radius} m is a picture of one bend"
        );
    }

    /// A two-metre puddle takes the floor rather than a two-metre radius, and an
    /// empty curve has no focus at all.
    #[test]
    fn a_short_curve_takes_the_floor_and_an_empty_one_has_none() {
        let line = [DVec3::new(0.0, 0.0, 0.0), DVec3::new(2.0, 0.0, 0.0)];
        let (centre, radius) = curve_focus(&line).expect("a focus");
        assert_eq!(centre, DVec3::new(1.0, 0.0, 0.0));
        assert_eq!(radius, 4.0);
        assert!(curve_focus(&[]).is_none());
    }
}

#[cfg(test)]
mod render_settings_tests {
    use super::{apply_record, RenderSettings, RenderSettingsRecord};

    /// The default record maps to the byte-stable renderer default — this pins the
    /// mapping so the editor viewport starts identical to today's pure defaults
    /// (and identical to the player's mirror).
    #[test]
    fn default_record_maps_to_default_settings() {
        assert_eq!(
            apply_record(&RenderSettingsRecord::default()),
            RenderSettings::default()
        );
    }

    /// A non-default record flows each authored field onto the live settings.
    #[test]
    fn non_default_fields_map_through() {
        let rec = RenderSettingsRecord {
            exposure: 2.0,
            dither: false,
            bloom_enabled: true,
            bloom_intensity: 0.3,
            ssao_enabled: true,
            taa: true,
            shadows_enabled: true,
            shadows_max_distance: 120.0,
            gi_enabled: true,
            gi_intensity: 1.5,
            ..RenderSettingsRecord::default()
        };
        let s = apply_record(&rec);
        assert_eq!(s.exposure, 2.0);
        assert!(!s.dither);
        assert!(s.bloom.enabled && (s.bloom.intensity - 0.3).abs() < 1e-6);
        assert!(s.ssao.enabled);
        assert!(s.taa);
        assert!(s.shadows.enabled && (s.shadows.max_distance - 120.0).abs() < 1e-6);
        assert!(s.gi.enabled && (s.gi.intensity - 1.5).abs() < 1e-6);
        // Untouched tuning knobs stay at their defaults.
        assert_eq!(s.shadows.lambda, RenderSettings::default().shadows.lambda);
        assert_eq!(s.gi.extent, RenderSettings::default().gi.extent);
    }
}

/// Spot-light seam parity (R-P3). The **identical** fixture + hardcoded
/// expectations live in `inf_player::render`'s mirror test; both must agree so
/// the toward-the-light / emission direction convention can never drift between
/// the editor viewport and the player.
#[cfg(test)]
mod project_light_parity {
    use super::{project_light, EcsLightKind, Light, LightKind};
    use glam::{DAffine3, DQuat, DVec3};

    #[test]
    fn spot_projects_with_shared_convention() {
        let light = Light {
            kind: EcsLightKind::Spot,
            intensity: 2.0,
            range: 12.0,
            inner_cone_deg: 20.0,
            outer_cone_deg: 35.0,
            cast_shadows: false,
            ..Light::default()
        };
        // Rotate 30° about X at (1, 2, 3).
        let affine = DAffine3::from_rotation_translation(
            DQuat::from_rotation_x(30f64.to_radians()),
            DVec3::new(1.0, 2.0, 3.0),
        );
        let rl = project_light(&light, &affine);

        assert!(matches!(rl.kind, LightKind::Spot));
        // Direction is *toward* the light = rot · +Z = (0, -sin30, cos30).
        let d = rl.direction;
        assert!((d.x - 0.0).abs() < 1e-5, "dir.x {}", d.x);
        assert!((d.y - (-0.5)).abs() < 1e-5, "dir.y {}", d.y);
        assert!((d.z - 0.866_025_4).abs() < 1e-5, "dir.z {}", d.z);
        assert!((rl.position.x - 1.0).abs() < 1e-9);
        assert!((rl.position.y - 2.0).abs() < 1e-9);
        assert!((rl.position.z - 3.0).abs() < 1e-9);
        assert!((rl.range - 12.0).abs() < 1e-6);
        assert!(
            (rl.inner_cos - 0.939_692_6).abs() < 1e-5,
            "inner {}",
            rl.inner_cos
        );
        assert!(
            (rl.outer_cos - 0.819_152).abs() < 1e-5,
            "outer {}",
            rl.outer_cos
        );
        assert!(!rl.cast_shadows);
    }
}

/// The deterministic foliage scatter sampler (E-P6): a pure function of its
/// inputs, so the same stroke input sequence reproduces identical instances (the
/// determinism law — no wall-clock / thread-rng).
#[cfg(test)]
mod foliage_sampler {
    use super::{foliage_min_spacing, foliage_samples, FOLIAGE_MAX_PER_TICK};
    use glam::DVec2;

    #[test]
    fn sampler_is_pure_and_reproducible() {
        let c = DVec2::new(5.0, -3.0);
        let a = foliage_samples(c, 3.0, 32, 1, 7, 0, 0.2, 2);
        let b = foliage_samples(c, 3.0, 32, 1, 7, 0, 0.2, 2);
        assert_eq!(a, b, "identical inputs must reproduce identical candidates");
        assert_eq!(a.len(), 32);
    }

    #[test]
    fn candidates_stay_in_disk_and_carry_kind_and_jitter() {
        let c = DVec2::new(0.0, 0.0);
        let radius = 4.0;
        let jitter = 0.25;
        let cs = foliage_samples(c, radius, FOLIAGE_MAX_PER_TICK, 42, 3, 100, jitter, 5);
        for s in &cs {
            let d = (s.pos_xz - c).length();
            assert!(d <= radius + 1e-9, "sample outside brush disk: {d}");
            assert!((0.0..=360.0).contains(&s.yaw_deg));
            assert!((1.0 - jitter - 1e-9..=1.0 + jitter + 1e-9).contains(&s.scale));
            assert_eq!(s.kind, 5);
        }
    }

    #[test]
    fn different_strokes_and_indices_diverge() {
        let c = DVec2::new(1.0, 1.0);
        let s0 = foliage_samples(c, 2.0, 8, 1, 0, 0, 0.2, 0);
        let s_stroke = foliage_samples(c, 2.0, 8, 1, 1, 0, 0.2, 0);
        let s_index = foliage_samples(c, 2.0, 8, 1, 0, 8, 0.2, 0);
        assert_ne!(s0, s_stroke, "a new stroke re-seeds the scatter");
        assert_ne!(s0, s_index, "advancing the sample index draws fresh values");
    }

    #[test]
    fn min_spacing_tightens_with_density_and_has_a_floor() {
        assert!(foliage_min_spacing(0.1) > foliage_min_spacing(4.0));
        assert!(foliage_min_spacing(0.0) >= 0.05);
        assert!(foliage_min_spacing(1e9) >= 0.05);
    }
}

/// **P16.6 — how the cursor resolves against N terrains.**
///
/// These pin the two rules the multi-terrain tool paths are built on, as pure
/// functions: nearest-along-the-ray for a pick, topmost for a scatter, plus the
/// `restrict` filter that keeps a stroke on the terrain it started on. An
/// `EngineHost` needs a GPU, so the rules live in free functions and the host is
/// a one-line caller of each — which is what makes them testable at all.
#[cfg(test)]
mod terrain_resolution {
    use super::{
        nearest_terrain_hit, terrain_probes_of, topmost_surface, TerrainProbe, TerrainSlot,
    };
    use glam::{DVec2, DVec3};
    use uuid::Uuid;

    fn guid(n: u128) -> Uuid {
        Uuid::from_u128(n)
    }

    /// A flat 4 × 4-tile heightfield at local height `h` (9² samples @ 2 m ⇒ a
    /// 64 m square).
    fn flat(h: f64) -> inf_terrain::TerrainData {
        let mut t = inf_terrain::TerrainData::new(9, 2.0);
        for tz in 0..4 {
            for tx in 0..4 {
                t.author_tile((tx, tz), |_, _| h);
            }
        }
        t
    }

    fn slot(g: Uuid) -> TerrainSlot {
        TerrainSlot {
            guid: g,
            streamed: false,
            editable: false,
            unsaved: false,
        }
    }

    fn probe<'a>(g: Uuid, data: &'a inf_terrain::TerrainData, at: DVec3) -> TerrainProbe<'a> {
        TerrainProbe {
            guid: g,
            data,
            translation: at,
        }
    }

    /// A straight-down ray from high above `(x, z)`.
    fn down(x: f64, z: f64) -> (DVec3, DVec3) {
        (DVec3::new(x, 500.0, z), DVec3::new(0.0, -1.0, 0.0))
    }

    /// **Nearest wins.** Two overlapping terrains under one cursor: the pick lands
    /// on the surface you can actually see, not on whichever the document lists
    /// first. Ties resolve to document order, so the answer is deterministic.
    #[test]
    fn a_pick_takes_the_nearest_of_overlapping_terrains() {
        let (low, high) = (flat(10.0), flat(0.0));
        let (a, b) = (guid(1), guid(2));
        // A sits on the ground (surface y = 10); B is a raised platform over the
        // SAME footprint (surface y = 50) — so B is nearer to a camera above.
        let probes = vec![
            probe(a, &low, DVec3::ZERO),
            probe(b, &high, DVec3::new(0.0, 50.0, 0.0)),
        ];
        let (ro, rd) = down(20.0, 20.0);
        let hit = nearest_terrain_hit(&probes, ro, rd).expect("the ray must hit something");
        assert_eq!(hit.guid, b, "the pick fell through the nearer terrain");
        assert!((hit.world.y - 50.0).abs() < 1e-6, "{:?}", hit.world);
        assert!((hit.local_height - 0.0).abs() < 1e-6);

        // Listing order must not change the answer.
        let reversed = vec![
            probe(b, &high, DVec3::new(0.0, 50.0, 0.0)),
            probe(a, &low, DVec3::ZERO),
        ];
        assert_eq!(nearest_terrain_hit(&reversed, ro, rd).unwrap().guid, b);

        // Two coincident surfaces tie — resolved to the EARLIER probe (document
        // order), never to iteration luck.
        let same = flat(10.0);
        let tied = vec![probe(a, &low, DVec3::ZERO), probe(b, &same, DVec3::ZERO)];
        assert_eq!(nearest_terrain_hit(&tied, ro, rd).unwrap().guid, a);

        // A ray that misses every terrain footprint reports nothing (the caller
        // then falls back to the ground plane).
        let (miss_ro, miss_rd) = down(9_000.0, 9_000.0);
        assert!(nearest_terrain_hit(&probes, miss_ro, miss_rd).is_none());
        assert!(nearest_terrain_hit(&[], ro, rd).is_none());
    }

    /// **Restrict pins a stroke.** Mid-stroke, a nearer terrain appearing under
    /// the cursor must not move the brush: the dabs belong to one document entity
    /// and the single `HeightDelta` the stroke commits belongs to one terrain.
    #[test]
    fn a_restricted_pick_stays_on_the_terrain_the_stroke_started_on() {
        let (low, high) = (flat(10.0), flat(0.0));
        let (a, b) = (guid(1), guid(2));
        let slots = [slot(a), slot(b)];
        let resolve = |g: Uuid| {
            if g == a {
                Some((&low, DVec3::ZERO))
            } else {
                Some((&high, DVec3::new(0.0, 50.0, 0.0)))
            }
        };
        let (ro, rd) = down(20.0, 20.0);

        // Unrestricted, the nearer terrain B wins (as the test above pins).
        let free = terrain_probes_of(&slots, None, resolve);
        assert_eq!(free.len(), 2);
        assert_eq!(nearest_terrain_hit(&free, ro, rd).unwrap().guid, b);

        // Restricted to A — the terrain a stroke started on — B is not even a
        // candidate, and the hit stays on A's surface.
        let pinned = terrain_probes_of(&slots, Some(a), resolve);
        assert_eq!(pinned.len(), 1);
        let hit = nearest_terrain_hit(&pinned, ro, rd).expect("A is still under the cursor");
        assert_eq!(hit.guid, a);
        assert!((hit.world.y - 10.0).abs() < 1e-6);

        // Restricting to a terrain that is not projected yields no candidates at
        // all — the stroke holds rather than jumping (`update_sculpt` returns).
        assert!(terrain_probes_of(&slots, Some(guid(99)), resolve).is_empty());
    }

    /// **Topmost wins for a scatter.** Foliage falls from above, so the ground at
    /// a point is the highest surface covering it — not the first listed, and not
    /// the nearest to a camera that may be underneath.
    #[test]
    fn a_scatter_takes_the_topmost_surface() {
        let (low, high) = (flat(10.0), flat(0.0));
        let (a, b) = (guid(1), guid(2));
        let probes = vec![
            probe(a, &low, DVec3::ZERO),
            probe(b, &high, DVec3::new(0.0, 50.0, 0.0)),
        ];
        let p = DVec2::new(20.0, 20.0);
        let (g, y) = topmost_surface(&probes, p).expect("covered");
        assert_eq!(g, b);
        assert!((y - 50.0).abs() < 1e-6);

        // Off both footprints ⇒ nothing (the caller uses the y = 0 ground plane).
        assert!(topmost_surface(&probes, DVec2::new(9_000.0, 9_000.0)).is_none());
        assert!(topmost_surface(&[], p).is_none());

        // A terrain's own world translation lifts its surface.
        let lifted = vec![probe(a, &low, DVec3::new(0.0, 7.0, 0.0))];
        assert!((topmost_surface(&lifted, p).unwrap().1 - 17.0).abs() < 1e-6);
    }

    /// **Why every terrain path must resolve through `terrain_probe`** (the P16.6
    /// audit fix): a streamed terrain's *document* heightfield is EMPTY by design —
    /// its tiles live in the `.inf_terrain` — so resolving a cursor against it
    /// finds nothing and drops silently to the `y = 0` ground plane. Only the
    /// streamer's render working set has real ground in it.
    ///
    /// This is the fixture the sculpt path already used and the drag-drop/foliage
    /// paths did not; it fails loudly if the document's set ever becomes the
    /// answer again.
    #[test]
    fn a_streamed_terrains_document_set_is_empty_but_its_streamer_has_ground() {
        use inf_editor_core::samples::{
            streamed_terrain_scene, write_streamed_terrain_asset, STREAMED_TERRAIN_TERRAIN_GUID,
        };
        use inf_editor_core::terrain_stream::EditorTerrainStreams;

        let dir = tempfile::tempdir().unwrap();
        write_streamed_terrain_asset(dir.path()).unwrap();
        let doc = streamed_terrain_scene();
        let terrain = {
            let world = doc.world();
            let e = world.entity_of(STREAMED_TERRAIN_TERRAIN_GUID).unwrap();
            world
                .world()
                .get::<inf_ecs::components::Terrain>(e)
                .unwrap()
                .clone()
        };

        // (1) The DOCUMENT's set — what the buggy call sites read — is empty, so
        //     no probe over it resolves any surface anywhere.
        let (doc_data, doc_origin) = doc
            .terrain_data_and_origin(STREAMED_TERRAIN_TERRAIN_GUID)
            .expect("the entity exists");
        assert!(doc_data.is_empty(), "a streamed terrain ships no tiles");
        let doc_probes = vec![probe(STREAMED_TERRAIN_TERRAIN_GUID, doc_data, doc_origin)];
        let p = DVec2::new(64.0, 64.0);
        assert!(
            topmost_surface(&doc_probes, p).is_none(),
            "the document's set must have no ground — that is the bug's cause"
        );
        let (ro, rd) = down(p.x, p.y);
        assert!(nearest_terrain_hit(&doc_probes, ro, rd).is_none());

        // (2) The STREAMER's render set does have ground there, once the camera
        //     has paged it in — which is what `terrain_probe` hands back.
        let mut streams = EditorTerrainStreams::new();
        streams.set_content_root(Some(dir.path().to_path_buf()));
        let eye = DVec3::new(p.x, 40.0, p.y);
        assert!(
            streams.ensure(STREAMED_TERRAIN_TERRAIN_GUID, &terrain, DVec3::ZERO, eye),
            "the fixture terrain must stream"
        );
        for _ in 0..32 {
            streams.sync_render(eye);
        }
        let (_, live, translation) = streams
            .projection_inputs(STREAMED_TERRAIN_TERRAIN_GUID)
            .expect("the stream is live");
        assert!(!live.is_empty(), "the camera paged nothing in");
        let live_probes = vec![probe(STREAMED_TERRAIN_TERRAIN_GUID, live, translation)];

        let (_, y) = topmost_surface(&live_probes, p).expect("streamed ground under the cursor");
        assert!(
            y.abs() > 1e-9,
            "the streamed surface read as flat zero — the generator has relief here"
        );
        let hit = nearest_terrain_hit(&live_probes, ro, rd).expect("the ray must hit the ground");
        // The two agree to within the raycaster's marching step: `height_at`
        // bilinearly interpolates the samples while `raycast_terrain` walks the
        // ray and interpolates at the crossing, so they differ by the sub-cell
        // residual — not by "one of them found nothing", which is the claim here.
        assert!((hit.world.y - y).abs() < 1e-3, "{hit:?} vs {y}");
    }
}

#[cfg(test)]
mod sky_projection_tests {
    use super::{project_sky, RenderScene, SkyParams, SunParams};
    use inf_ecs::components::{SkyAtmosphere, TimeOfDay};
    use inf_ecs::EcsWorld;
    use uuid::Uuid;

    fn world_with(tod: TimeOfDay, atmos: SkyAtmosphere) -> EcsWorld {
        let mut w = EcsWorld::new();
        let e = w.spawn_with_guid(Uuid::from_u128(1), "Sky", None);
        w.world_mut().entity_mut(e).insert(tod);
        w.world_mut().entity_mut(e).insert(atmos);
        w
    }

    // NOTE: the **MIRROR gate** that compares this crate's `project_sky` against
    // the shipped player's, character for character, deliberately does NOT live
    // here: this module is `#[cfg(any(windows, target_os = "macos"))]`, so a test
    // inside it is invisible to the Linux CI leg. It lives in
    // `inf-editor-core/tests/projector_mirror.rs`, which compiles on all three
    // platforms and reads both files as source text.

    /// No time-of-day authority ⇒ the renderer's own defaults stand, which are
    /// bit-for-bit the retired `SUN_DIR` and the historic gradient. This is the
    /// promise that keeps every pre-P17.1 level and golden byte-identical.
    #[test]
    fn a_clockless_world_projects_the_retired_defaults() {
        let mut scene = RenderScene::default();
        scene.sun.intensity = 999.0; // deliberately dirty, to prove it is reset
        scene.lights.clear();
        project_sky(&mut scene, &EcsWorld::new());
        assert_eq!(scene.sun, SunParams::default());
        assert_eq!(scene.sky, SkyParams::default());
        assert!(scene.lights.is_empty(), "no clock ⇒ no sun light");
    }

    /// A daytime clock publishes the sun as `lights[0]` and leaves the authored
    /// gradient untouched (the sky only dims once the sun is near the horizon).
    #[test]
    fn a_daytime_clock_publishes_the_sun() {
        let mut scene = RenderScene::default();
        let w = world_with(TimeOfDay::default(), SkyAtmosphere::default());
        project_sky(&mut scene, &w);
        assert!(scene.sun.direction.y > 0.5, "{:?}", scene.sun);
        assert_eq!(scene.sun.intensity, 3.0);
        assert_eq!(scene.sky, SkyParams::default(), "daytime tint is untouched");
        assert_eq!(scene.lights.len(), 1);
        assert_eq!(scene.lights[0].kind, inf_render::LightKind::Directional);
        assert!(scene.lights[0].cast_shadows);
        assert_eq!(
            scene.lights[0].direction, scene.sun.direction,
            "the key light must be the projected sun"
        );
    }

    /// At night the moon takes over as the key light and the gradient darkens.
    #[test]
    fn a_night_clock_publishes_the_moon_and_darkens_the_sky() {
        let mut scene = RenderScene::default();
        let w = world_with(
            TimeOfDay {
                seconds: 0.0,
                ..TimeOfDay::default()
            },
            SkyAtmosphere::default(),
        );
        project_sky(&mut scene, &w);
        assert!(scene.sun.direction.y < 0.0, "the sun has set");
        assert_eq!(scene.lights.len(), 1);
        assert_eq!(scene.lights[0].intensity, 0.15);
        assert_eq!(scene.lights[0].direction, scene.sun.moon_direction);
        assert!(
            scene.sky.zenith[2] < SkyParams::default().zenith[2],
            "the night sky must darken: {:?}",
            scene.sky
        );
    }

    /// `enabled: false` keeps the clock and the tint but authors no light.
    #[test]
    fn a_disabled_atmosphere_projects_no_light() {
        let mut scene = RenderScene::default();
        let w = world_with(
            TimeOfDay::default(),
            SkyAtmosphere {
                enabled: false,
                ..SkyAtmosphere::default()
            },
        );
        project_sky(&mut scene, &w);
        assert!(scene.lights.is_empty());
        assert!(scene.sun.direction.y > 0.5, "the sun is still projected");
    }
}

/// The analytic pick fallback that keeps real geometry clickable (P18.3).
///
/// The rule is pure, so it is testable here without a GPU — which matters,
/// because the ID-buffer pass it backs up cannot be exercised headlessly at all.
#[cfg(test)]
mod analytic_pick {
    use super::ray_sphere_t;
    use glam::DVec3;

    const FWD: DVec3 = DVec3::new(0.0, 0.0, -1.0);

    #[test]
    fn a_ray_through_a_sphere_hits_at_its_near_surface() {
        // Sphere of radius 1 at z = -10, looking down -Z from the origin.
        let t = ray_sphere_t(DVec3::ZERO, FWD, DVec3::new(0.0, 0.0, -10.0), 1.0)
            .expect("a ray straight at a sphere hits it");
        assert!((t - 9.0).abs() < 1e-9, "near surface, got {t}");
    }

    #[test]
    fn a_ray_beside_a_sphere_misses() {
        assert!(ray_sphere_t(DVec3::ZERO, FWD, DVec3::new(3.0, 0.0, -10.0), 1.0).is_none());
    }

    /// Pointing away from a sphere is a miss — otherwise clicking the sky would
    /// select whatever happens to be behind the camera.
    #[test]
    fn a_sphere_behind_the_eye_misses() {
        assert!(ray_sphere_t(DVec3::ZERO, FWD, DVec3::new(0.0, 0.0, 10.0), 1.0).is_none());
    }

    /// Standing inside an object and clicking must select it, not fall through.
    #[test]
    fn a_ray_starting_inside_hits_at_zero() {
        let t = ray_sphere_t(DVec3::ZERO, FWD, DVec3::new(0.0, 0.0, -0.5), 5.0).unwrap();
        assert_eq!(t, 0.0);
    }

    /// Nearest-along-the-ray is the rule the fallback resolves overlaps with, so
    /// the ordering it depends on has to be the real one.
    #[test]
    fn nearer_spheres_report_smaller_t() {
        let near = ray_sphere_t(DVec3::ZERO, FWD, DVec3::new(0.0, 0.0, -5.0), 1.0).unwrap();
        let far = ray_sphere_t(DVec3::ZERO, FWD, DVec3::new(0.0, 0.0, -50.0), 1.0).unwrap();
        assert!(near < far, "{near} !< {far}");
    }
}

/// The editor's render-settings request (P18.3). Pure, so the decision that puts
/// the viewport on the streamed meshlet path is testable without an adapter —
/// which matters, because the bug this pins was invisible: the classic fallback
/// draws the *same geometry*, so nothing looked wrong while the editor silently
/// skipped every part of P18.2.
/// **The on-screen rejection strings (C4-14).**
///
/// Every constant here is printed verbatim to a user through `reject_tool`, and
/// two of them shipped with a swallowed `\`-continuation: the source's own
/// indentation appeared mid-sentence in the viewport. B33 claimed the class had
/// a producer-side guard "where it reaches a user"; nothing guarded these.
///
/// The gate is over the *values*, not the source text, because that is what the
/// user reads — and it is a list the compiler forces an author to extend, since
/// a new rejection constant that is not in it is simply not covered and the
/// omission is visible right here.
#[cfg(test)]
mod rejection_text {
    use super::EngineHost;

    #[test]
    fn no_rejection_a_user_reads_carries_an_eaten_continuation() {
        let all = [
            ("WATER_NO_GROUND", EngineHost::WATER_NO_GROUND_REJECTION),
            (
                "WATER_LAKE_TOO_SMALL",
                EngineHost::WATER_LAKE_TOO_SMALL_REJECTION,
            ),
        ];
        for (name, msg) in all {
            assert!(
                !msg.contains("  "),
                "{name} carries a run of spaces — a continuation was eaten: {msg:?}"
            );
            assert!(
                !msg.is_empty() && msg.ends_with('.'),
                "{name} must be a finished sentence: {msg:?}"
            );
        }
    }
}

#[cfg(test)]
mod requested_settings {
    use super::{apply_record, requested_render_settings, RenderSettings, RenderSettingsRecord};
    use inf_render::RenderTier;

    #[test]
    fn the_editor_asks_for_the_meshlet_path() {
        let req = requested_render_settings(&RenderSettingsRecord::default());
        assert!(
            req.vgeom.enabled,
            "the editor must REQUEST vgeom — `VgeomSettings::default()` is off, so \
             without this every imported mesh draws through the classic fallback"
        );
    }

    /// The request changes **only** the vgeom master switch: the level's authored
    /// block still decides everything else, exactly as before P18.3.
    #[test]
    fn nothing_else_moves() {
        let rec = RenderSettingsRecord {
            exposure: 1.75,
            taa: true,
            gi_enabled: true,
            ..RenderSettingsRecord::default()
        };
        let base = apply_record(&rec);
        let req = requested_render_settings(&rec);
        assert_eq!(
            RenderSettings {
                vgeom: base.vgeom,
                ..req
            },
            base,
            "the opt-in must touch nothing but `vgeom.enabled`"
        );
        // …and within vgeom, only `enabled`.
        assert_eq!(
            inf_render::VgeomSettings {
                enabled: base.vgeom.enabled,
                ..req.vgeom
            },
            base.vgeom
        );
    }

    /// The tier still has the last word: a machine without the meshlet path gets
    /// the classic fallback, exactly as the player does. Requesting is not forcing.
    #[test]
    fn the_tier_still_clamps_it_away() {
        let req = requested_render_settings(&RenderSettingsRecord::default());
        assert!(!RenderTier::Medium.apply(req).vgeom.enabled);
        assert!(!RenderTier::Low.apply(req).vgeom.enabled);
        assert!(RenderTier::High.apply(req).vgeom.enabled);
    }
}
