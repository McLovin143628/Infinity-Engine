//! The player's render host + its own ECS→[`RenderScene`] projection (P9.3
//! item 1). Ring-2 engine code with **no editor deps**: it uses `inf-render`
//! exactly as `inf-viewport`'s `EngineHost` does (floating origin, reverse-Z, all
//! existing passes) but reads a plain `EcsWorld` instead of the editor's
//! `SceneDoc`.
//!
//! The projection (`project_scene`) mirrors the *shape* of
//! `inf_viewport::host::rebuild_scene` — meshes, lights, sprites, tilemaps, text,
//! 9-slices, 2D lights, billboards — but is duplicated here rather than shared,
//! because the editor version depends on `inf-editor-core`. A shared Ring-0
//! projection crate is a documented follow-up (both would then read `&EcsWorld`).
//!
//! Textures behave as in the editor viewport: the player has no asset-DB in this
//! thread yet, so referenced sprites/tilemaps render as the renderer's white
//! fallback tinted by their color (a colored quad). Uploading real texture bytes
//! is the same follow-up the editor documents.

use std::sync::Arc;

use glam::{DVec3, Vec2, Vec3};
use uuid::Uuid;

use inf_ecs::components::{
    BlendMode, ComputedVisibility, Foliage, GlobalTransform, Light, Light2D,
    LightKind as EcsLightKind, Material, MeshRef, NineSlice, PcgVolume, Primitive,
    ScatteredInstance, SkeletalMesh, Spline, Sprite, Terrain, Text2D, TextAlign, Tilemap,
    VoxelVolume, WaterBody, WaterKind,
};
use inf_ecs::{Guid, Vec3d};
use inf_math::FloatingOrigin;
use inf_render::{
    detect_tier, expand_nine_slice, expand_text, handle_from_guid, AdapterCaps, AtmosphereParams,
    BloomSettings, CloudParams, EngineRenderer, ExposureMode, ExposureSettings, FilmSettings,
    FlareSettings, GiSettings, GpuContext, HAlign, HeightFog, LightKind, MeshInstance,
    NineSliceParams, PrebatchedRun, PrecipParams, PrimMesh, RenderChunk, RenderLight,
    RenderLight2D, RenderScene, RenderSettings, RenderTerrain, RenderTerrainLayer,
    RenderTerrainTile, RenderTilemap, RenderView, RenderWater, ScatterBatch, ScatterData,
    ScatterInstance, ShadowSettings, SkinnedInstance, SkyParams, SsaoSettings, SsrQuality,
    SsrSettings, SunParams, SurfaceChain, TerrainTileKey, TextParams, TilemapParams, VgeomAsset,
    VgeomInstance, BUILTIN_FONT_TEXTURE,
};
use inf_scene::RenderSettingsRecord;

use crate::runtime_sim::RuntimeSim;
use crate::skinned::SkinnedRegistry;
use crate::vmesh::VmeshRegistry;
use crate::voxel::VoxelRegistry;

/// Owns the GPU stack + the render scene the player draws each frame.
pub struct PlayerRenderHost {
    gpu: GpuContext,
    chain: SurfaceChain,
    renderer: EngineRenderer,
    scene: RenderScene,
    origin: FloatingOrigin,
    /// The cook-derived `.inf_vmesh` DAGs a `MeshRef.asset` resolves to (P13.4);
    /// empty for the `--demo` / primitive-only worlds. Set via [`set_vmeshes`].
    ///
    /// [`set_vmeshes`]: PlayerRenderHost::set_vmeshes
    vmeshes: Arc<VmeshRegistry>,
    /// The skeletal render assets a `SkeletalMesh` resolves to (bind-space
    /// geometry + skeletons + clips), from the loaded pack / dev-dir; inert for
    /// the `--demo` / browser worlds (PIE has carried real bytes since P24.1's
    /// `ScenePayload` v7). Set via [`set_skinned`].
    ///
    /// [`set_skinned`]: PlayerRenderHost::set_skinned
    skinned: Arc<SkinnedRegistry>,
    /// The authored meshes scattered content draws (wave TER2b) — every
    /// `.inf_mesh` a loaded `.inf_pcg`'s scatter kinds name, flattened into the
    /// two arrays the scatter raster pulls from. Set via [`set_scatter_meshes`];
    /// empty means every scattered instance draws its placeholder primitive,
    /// which is what the whole engine did until this wave.
    ///
    /// [`set_scatter_meshes`]: PlayerRenderHost::set_scatter_meshes
    scatter_meshes: Arc<inf_render::ScatterMeshes>,
    /// Where a `VoxelVolume.asset` finds its `.inf_voxel` bytes (P21.1) — the
    /// pack, a dev directory, or nothing. Set via [`set_voxel_assets`].
    ///
    /// [`set_voxel_assets`]: PlayerRenderHost::set_voxel_assets
    voxel_assets: Arc<VoxelRegistry>,
    /// The loaded voxel volumes this world draws, keyed by entity. Owned here
    /// rather than passed in because loading is a `&mut` act and projection is
    /// not: [`project`](Self::project) syncs the store first, then projects from
    /// it — the same split the editor viewport uses, and the reason
    /// `project_scene_with_skinned` can keep taking an immutable borrow.
    voxels: inf_voxel::VoxelVolumes,
    /// Whether the auto-picked [`RenderTier`](inf_render::RenderTier) enables the
    /// GPU meshlet path (High). Off → the classic discrete-LOD fallback renders the
    /// same vgeom content (the renderer's `ClassicVgeomNode`).
    vgeom_enabled: bool,
    /// The P22.4 sub-chunk rubble memo, keyed by each broken actor's fracture
    /// generation, so a collapse packs its instance payload when the live chunk
    /// set changes rather than on every frame.
    debris_cache: inf_render::DebrisCache,
    /// The level's material bindings + their `.inf_tex` containers (P26.4),
    /// kept so a device-loss rebuild can re-register them without reopening the
    /// pack. Empty for `--demo` / primitive-only worlds.
    materials: Arc<crate::MaterialContent>,
    /// The auto-picked capability tier itself (P22.4).
    ///
    /// Kept because the tier now decides one thing that is **not** a renderer
    /// setting: the level's debris budget. The window owns the session, so it —
    /// not this host — reads this and calls `set_debris_budget`; see
    /// [`inf_render::debris_budget_for`] for why that mapping lives at the host
    /// and never inside physics.
    tier: inf_render::RenderTier,
}

impl PlayerRenderHost {
    /// Build the render host over an already-created surface + GPU context (the
    /// window module owns the winit window and makes the surface from it). `record`
    /// is the loaded level's scene-persisted render block (R-P4) — post / exposure
    /// / lighting — mapped onto the base [`RenderSettings`]; pass
    /// [`RenderSettingsRecord::default`] for content with no authored block.
    pub fn new(
        gpu: GpuContext,
        surface: wgpu::Surface<'static>,
        width: u32,
        height: u32,
        record: RenderSettingsRecord,
    ) -> Result<Self, String> {
        let chain = SurfaceChain::new(&gpu, surface, width, height)?;
        // Record the GPU adapter for the crash report (P15.2) as a first-class
        // field (it also already appears in the tracing log tail).
        crate::log::set_adapter_info(format!("{:?}", gpu.adapter.get_info()));
        let mut renderer = EngineRenderer::new(&gpu, chain.target_format());

        // R-P4: start from the level's persisted render block (exposure / dither /
        // bloom / ssao / taa / shadows / gi) instead of pure defaults, so the
        // shipped player looks like the authored scene — the mirror of the editor
        // viewport's `apply_render_settings`.
        //
        // Auto-tier (P13.4.2): probe the adapter, pick a render tier, and apply it
        // to the renderer's settings. High enables the GPU meshlet path; Medium/Low
        // fall back to the classic discrete-LOD path (and Low drops the expensive
        // post effects). The decision is logged by `detect_tier`.
        let (settings, tier) = shipped_settings(&gpu, record);
        let vgeom_enabled = settings.vgeom.enabled;
        renderer.set_settings(settings);

        Ok(Self {
            gpu,
            chain,
            renderer,
            scene: RenderScene {
                grid_enabled: false,
                ..Default::default()
            },
            origin: FloatingOrigin::default(),
            vmeshes: Arc::new(VmeshRegistry::new()),
            skinned: Arc::new(SkinnedRegistry::new()),
            scatter_meshes: Arc::new(inf_render::ScatterMeshes::new()),
            voxel_assets: Arc::new(VoxelRegistry::new()),
            voxels: inf_voxel::VoxelVolumes::new(),
            vgeom_enabled,
            debris_cache: inf_render::DebrisCache::default(),
            materials: Arc::new(crate::MaterialContent::default()),
            tier,
        })
    }

    /// The GPU capability tier this host auto-picked from the adapter (P22.4).
    ///
    /// Exposed for exactly one caller — the window, which maps it onto the
    /// session's debris budget. A HEADLESS boot path (`run_headless`,
    /// `scene_trace`, and the `--pie` protocol loop's `windowed == false`
    /// branch) builds no host at all and therefore never sees a tier, which is
    /// what keeps every determinism gate in this repository comparing the
    /// *unclamped* engine defaults.
    ///
    /// **The `--pie` loop is not unconditionally headless, and this doc used to
    /// say it was** (corrected, wave CERT1). `main.rs`'s protocol loop branches
    /// on `payload.windowed`, and the windowed branch goes
    /// `run_pie_window` → `window::run_pie` → `PlayerApp::new` → `build_host` →
    /// `PlayerRenderHost::new` → `shipped_settings` → `detect_tier`. So embedded
    /// and new-window PIE **do** build a host and **do** see a tier — the P22.4
    /// law ("a preview must run what it previews") in the one sentence that had
    /// gone stale against it.
    pub fn tier(&self) -> inf_render::RenderTier {
        self.tier
    }

    /// Attach the cook-derived vmesh registry (from the loaded pack / dev-dir) so
    /// `MeshRef.asset` entities render their real geometry — through the GPU meshlet
    /// path (High tier) or the classic discrete-LOD fallback (otherwise). Empty for
    /// primitive-only worlds.
    pub fn set_vmeshes(&mut self, vmeshes: Arc<VmeshRegistry>) {
        self.vmeshes = vmeshes;
    }

    /// Attach the skeletal render-asset store (from the loaded pack / dev-dir) so
    /// a bound `SkeletalMesh` renders its real, posed skinned geometry instead of
    /// a placeholder cube — the shipped half of the editor viewport's P18.3
    /// projection. Inert for primitive-only worlds.
    ///
    /// `Arc`-shared with nothing else today, but `Arc` on purpose: the registry
    /// owns one `Arc<SkinnedMeshData>` per mesh asset, and a device-loss rebuild
    /// has to hand the *same* store to the new host or every skinned upload would
    /// be cold again for no reason.
    pub fn set_skinned(&mut self, skinned: Arc<SkinnedRegistry>) {
        self.skinned = skinned;
    }

    /// Attach the scatter-kind mesh table (wave TER2b) so a `PcgKind` that names
    /// an `.inf_mesh` draws it instead of the placeholder cube every scattered
    /// instance drew from P18.5 until this wave.
    ///
    /// `Arc` for `set_skinned`'s reason: a device-loss rebuild must hand the same
    /// table to the new host, and the table is the only thing standing between a
    /// projection and a file read.
    pub fn set_scatter_meshes(&mut self, meshes: Arc<inf_render::ScatterMeshes>) {
        self.scatter_meshes = meshes;
    }

    /// Attach the `.inf_voxel` source (P21.1) so a `VoxelVolume` entity renders
    /// the cave it references instead of nothing. Inert for primitive-only worlds.
    ///
    /// Swapping the source drops every loaded volume: a different pack (or dev
    /// directory) is a different world, and a GUID that resolved under the old one
    /// says nothing about the new one.
    pub fn set_voxel_assets(&mut self, voxel_assets: Arc<VoxelRegistry>) {
        self.voxel_assets = voxel_assets;
        self.voxels.clear();
    }

    /// **Attach the level's material bindings** (P26.4, clause 0) so a bound
    /// `.inf_mat`'s textures reach the surfaces that name it.
    ///
    /// This is the shipped half of what the P26.3b ledger called the missing
    /// glue. Everything downstream of it was already built: the registration
    /// door, the material rule, the want floor and the WGSL sample. What did not
    /// exist was any non-test caller, so a cooked pack carried the `.inf_matd`
    /// records and the `.inf_tex` containers and nothing ever looked at them.
    ///
    /// Registration goes through `inf_render::build_vt_level` — the same call
    /// the editor viewport makes, with the same materials in the same order — so
    /// "PIE == shipping" for texture residency is a property of the code. A
    /// world with no bindings hands the renderer `None`, which restores the
    /// textureless command stream exactly.
    pub fn set_material_content(&mut self, materials: Arc<crate::MaterialContent>) {
        self.materials = materials;
        self.rebuild_vt();
    }

    /// (Re)build the virtual-texture level from `self.materials`.
    ///
    /// Separate from the setter because a **device-loss rebuild** must do it
    /// again: the atlas and the indirection buffer belong to the dead device,
    /// and the registry's readers do not — they are container slices, which is
    /// why re-registering costs a header parse per texture and no re-read.
    fn rebuild_vt(&mut self) {
        let mats = self.materials.vt_materials();
        if mats.is_empty() {
            self.renderer.set_vt_level(None);
            return;
        }
        let materials = self.materials.clone();
        // The budget is the TIER's (P26.5), read off the settings the host
        // already clamped through `RenderTier::apply` — not the crate default,
        // which is what both hosts passed through P26.4 and which made the pool
        // the one piece of the renderer a weak GPU paid full price for.
        let budget = self.renderer.settings().vt.budget_bytes;
        let level = inf_render::build_vt_level(
            &self.gpu.device,
            &self.gpu.queue,
            self.renderer.settings(),
            budget,
            &mats,
            |g| materials.source(g),
        );
        match level {
            Some((textures, pools, report)) => {
                for a in &report.advisories {
                    tracing::warn!("inf-player: virtual textures: {a}");
                }
                // **Every arm, not the first one's format** (wave IASSET2
                // audit) — the editor viewport's line says the same, for the
                // same reason: `pool_format` is the FIRST arm's and reads as the
                // whole answer, so a BC1 + BC5 level logged "Bc1 pages" with
                // half its content in another atlas. `demoted` is a cost an
                // author can avoid and nothing else surfaces it per level.
                tracing::info!(
                    "inf-player: {} virtual texture(s) registered for {} bound material(s) \
                     ({:?} page arm(s), {:?} demoted, {} refused)",
                    report.textures,
                    mats.len(),
                    report.pool_formats,
                    report.demoted,
                    report.refused
                );
                self.renderer.set_vt_level(Some((textures, pools)));
            }
            None => self.renderer.set_vt_level(None),
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.chain.request_resize(width, height);
    }

    /// Current requested surface size (physical px).
    pub fn size(&self) -> (u32, u32) {
        self.chain.requested_size()
    }

    /// Drop this host's swapchain, without dropping the host (Hardening D).
    ///
    /// Called on the device-loss path **before** `build_host` creates a second
    /// `Instance` + `Surface` for the same window; see
    /// [`inf_render::SurfaceChain::release`]. The host renders nothing
    /// afterwards (`acquire` answers `None`, which is the occluded-window path)
    /// and its caller replaces it immediately.
    pub fn release_surface(&mut self) {
        self.chain.release();
    }

    /// The floating origin (the camera rebases against it before rendering).
    pub fn origin(&self) -> FloatingOrigin {
        self.origin
    }

    /// **Commit a camera pose at a fixed step** (P28.4) — the predictive
    /// prefetcher's only input, and the door that keeps it honest.
    ///
    /// `tick` must be the *sim's* step count and not a frame index, and the
    /// renderer refuses a tick that does not strictly advance — which is what a
    /// frame that ran zero fixed steps produces, and what a host wired to the
    /// wrong loop would produce every frame. This player qualifies to call it
    /// because its camera is `RuntimeSim::camera_focus`, a fold of actor
    /// positions, which is a pure function of the committed input.
    pub fn commit_camera(&mut self, tick: u64, view: &RenderView) -> bool {
        self.renderer.commit_camera(tick, view)
    }

    /// Rebuild the render scene from the sim's world, interpolated by `alpha`.
    ///
    /// `camera_world` is this frame's eye in world metres — the same point
    /// [`RuntimeSim::sync_render_terrain`] is driven from, and used for the same
    /// thing: camera-driven residency (P21.2). It reaches only the render host's
    /// own stores, never the sim, which is what keeps a fixed step's answers
    /// independent of where the camera happens to be.
    pub fn project(&mut self, sim: &RuntimeSim, alpha: f64, camera_world: DVec3) {
        self.sync_voxels(sim, camera_world);
        project_scene_full(
            &mut self.scene,
            sim,
            alpha,
            &self.vmeshes,
            &self.skinned,
            &self.voxels,
            &mut self.debris_cache,
            self.renderer.vt_textures(),
            &self.scatter_meshes,
            &self.materials.materials,
        );
    }

    fn sync_voxels(&mut self, sim: &RuntimeSim, camera_world: DVec3) {
        sync_voxel_store(&mut self.voxels, &self.voxel_assets, sim, camera_world);
    }

    /// Stroke the **world-partition cell overlay** into the debug-line layer
    /// (P16.5) — one wireframe box per streamed cell, coloured by its state.
    ///
    /// A deliberately separate, opt-in step *after* [`project`](Self::project)
    /// rather than a branch inside it: this is engine debug geometry behind
    /// `--debug-cells`, and keeping it out of the projection makes it obvious at
    /// the call site that a shipped player draws none of it. It reads only
    /// [`CellStreaming`](crate::cell_stream::CellStreaming) state and writes only
    /// into `scene.debug`, so there is no path from here back into the sim.
    ///
    /// The boxes are 1 m tall slabs sitting on the cell footprint at `y = 0` —
    /// enough to read the grid from a ground camera without occluding the world.
    pub fn draw_cell_overlay(&mut self, sim: &RuntimeSim) {
        use crate::cell_stream::CellState;
        let cells = sim.cell_streaming();
        if cells.is_empty() {
            return;
        }
        self.scene.debug.clear();
        for coord in cells.available() {
            let color = match cells.cell_state(coord) {
                CellState::Active => [0.25, 0.95, 0.35, 1.0],
                CellState::Loaded => [0.95, 0.80, 0.20, 1.0],
                CellState::Cold => [0.35, 0.38, 0.45, 1.0],
                CellState::Failed => [0.95, 0.25, 0.25, 1.0],
            };
            let (min, max) = cells.cell_bounds(coord);
            let center = DVec3::new(
                (min[0] + max[0]) * 0.5,
                CELL_OVERLAY_HALF_HEIGHT_M,
                (min[1] + max[1]) * 0.5,
            );
            let half = Vec3::new(
                ((max[0] - min[0]) * 0.5) as f32,
                CELL_OVERLAY_HALF_HEIGHT_M as f32,
                ((max[1] - min[1]) * 0.5) as f32,
            );
            self.scene.debug.wire_box(
                self.origin.to_render(center),
                half,
                glam::Quat::IDENTITY,
                color,
            );
        }
    }

    /// **Draw this step's tracers** (island wave I6) — one line per round that
    /// left a barrel, from the muzzle to wherever it stopped.
    ///
    /// Minimal on purpose and honest about it: a tracer here is a debug-line
    /// segment, which is the substrate this engine has. There is no particle
    /// system (the P22 remainder, still open), so a muzzle flash is the first
    /// twenty centimetres of the same line drawn brighter, and a bullet leaves
    /// no smoke.
    ///
    /// **Read-only, like every other overlay**: it copies out of the report the
    /// fixed step produced and writes into `scene.debug`, so there is no path
    /// from here back into the sim, and a frame that drew no tracer and a frame
    /// that drew ten are the same simulation.
    ///
    /// Drawn AFTER `draw_cell_overlay`, which clears the list, so the two do not
    /// erase each other.
    pub fn draw_tracers(&mut self, sim: &RuntimeSim) {
        let hits = &sim.gameplay().hits;
        if hits.is_empty() {
            return;
        }
        for hit in hits {
            // **Only a LOUD attack draws one** (wave WPN1). A swing arrives in
            // this same list — it goes through the same trigger, the same clock
            // and the same `hits` vector as a round — and a fist that drew a
            // tracer and a muzzle flash would be a bullet trail coming out of an
            // empty hand. `WeaponHit::loud` is the discriminator the audio
            // queue already uses, which is what keeps the two answers one
            // answer.
            if !hit.loud {
                continue;
            }
            let from = self.origin.to_render(hit.from);
            let to = self.origin.to_render(hit.to);
            self.scene.debug.line(from, to, TRACER_COLOR);
            // The flash: the first stretch of the same line, brighter. A
            // fraction rather than a length, so a point-blank shot still has one.
            let flash = from + (to - from) * MUZZLE_FLASH_FRACTION;
            self.scene.debug.line(from, flash, MUZZLE_COLOR);
        }
    }

    /// **Draw this step's extinguish lines** (wave EMS2) — one per fire crew
    /// working a scene, from the crew member's shoulder to what is burning.
    ///
    /// `draw_tracers`' sentence verbatim, one system along: there is no particle
    /// system, so a hose is a debug-line segment, which is the substrate this
    /// engine has. Read-only — it copies out of `inf_physics::d3::dispatch`'s
    /// own list and writes into `scene.debug`, so there is no path from here
    /// back into the sim.
    ///
    /// # THE LEDGER SENTENCE: this is a SHIPPED-HOST overlay only
    ///
    /// `draw_tracers` has exactly one caller (`window.rs`) and no editor twin,
    /// and this follows it. So a beam is drawn by the player and by embedded
    /// PIE, and **not** in the editor's Simulate viewport — which is a fact
    /// about where the tracer path was built rather than a decision this wave
    /// took, and it is written down here so nobody has to discover it.
    pub fn draw_extinguish(&mut self, sim: &RuntimeSim) {
        let beams = inf_physics::d3::dispatch::extinguish_beams(sim.world());
        for (from, to) in beams {
            let a = self.origin.to_render(from);
            let b = self.origin.to_render(to);
            self.scene.debug.line(a, b, EXTINGUISH_COLOR);
        }
    }

    /// Whether the GPU meshlet path is active (the auto-picked tier is High).
    /// **The one line a host logs about virtual shadow maps** (P27.5) — the
    /// P27.1 remainder *"nothing logs `vsm_summary` in a host"*, closed on the
    /// shipping side.
    ///
    /// `None` when the renderer has no system, which is not the same as a line
    /// of zeros: a level with virtual shadows off and a level whose atlas is
    /// empty are different states and a host that printed zeros for both would
    /// say neither.
    pub fn vsm_summary(&self) -> Option<String> {
        self.renderer.vsm_summary()
    }

    /// **The three lines P28.5 gave a host** — virtual texturing, the unified
    /// streamer, and the predictor.
    ///
    /// `vsm_summary` above closed the P27.1 remainder *"nothing logs
    /// `vsm_summary` in a host"*. These three were the same shape and nobody
    /// noticed: `EngineRenderer::vt_summary` (P26.5),
    /// `EngineRenderer::stream_summary` (P28.3) and
    /// `EngineRenderer::predict_summary` (P28.5) each carry the doc comment
    /// *"the one line a host logs about …"* and, until this batch, **no host
    /// logged any of them** — a gate read `stream_report` and nothing read the
    /// line. The class ends with the plan.
    ///
    /// Each is `None` rather than a line of zeros when its subject is absent,
    /// for the reason `vsm_summary` gives: a level with no virtual textures and
    /// a level whose pool is idle are different states.
    pub fn vt_summary(&self) -> Option<String> {
        self.renderer.vt_summary()
    }

    /// See [`PlayerRenderHost::vt_summary`].
    pub fn stream_summary(&self) -> Option<String> {
        self.renderer.stream_summary()
    }

    /// See [`PlayerRenderHost::vt_summary`].
    pub fn predict_summary(&self) -> Option<String> {
        self.renderer.predict_summary()
    }

    pub fn vgeom_enabled(&self) -> bool {
        self.vgeom_enabled
    }

    /// Render one frame for `view`. Handles device-lost recovery like the editor
    /// host (rebuilds nothing here — the caller rebuilds the whole stack on loss;
    /// transient acquire failures skip the frame).
    pub fn render(&mut self, view: &RenderView) {
        let Some(frame) = self.chain.acquire(&self.gpu) else {
            return; // transient (occluded/timeout) — skip
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
    }

    /// Whether the GPU device was lost (the caller rebuilds the stack).
    pub fn is_lost(&self) -> bool {
        self.gpu.is_lost()
    }

    /// **Hand the frame's in-game UI to the renderer** (island wave I5).
    ///
    /// The one seam a screen-space overlay needs, and it is a *setter* rather
    /// than a `scene_mut()`: everything else in `RenderScene` is projected from
    /// the sim by `project_scene_full`, which clears and rebuilds it every
    /// frame, and a UI built by the host would be wiped by the next projection.
    /// A named door says which field the host owns and gives the projector no
    /// reason to touch it.
    ///
    /// Called **between** `project` and `render`, which is the only window in
    /// which both halves of the frame exist.
    pub fn set_ui(&mut self, ui: &inf_ui::UiDrawList) {
        // Cloned rather than moved: the host keeps its list across frames so a
        // menu costs no allocation per frame, and the two buffers are a few
        // hundred quads at the very most.
        self.scene.ui.clone_from(ui);
    }

    /// The render surface's configured size in physical pixels — what the UI
    /// lays itself out for.
    ///
    /// The **configured** size and not the window's: they differ for the frames
    /// a resize debounce is pending, and a UI laid out for a size the swap chain
    /// does not have would be stretched by exactly that ratio.
    pub fn surface_size(&self) -> (u32, u32) {
        self.chain.configured_size()
    }
}

/// Half-height (metres) of a cell-overlay wireframe slab. Low enough to read the
/// grid from a ground camera without boxing the world in.
const CELL_OVERLAY_HALF_HEIGHT_M: f64 = 0.5;

/// A tracer's colour — a warm streak, bright enough to read against daylight.
const TRACER_COLOR: [f32; 4] = [1.0, 0.82, 0.35, 0.85];
/// The muzzle flash's, drawn over the first stretch of the same line.
const MUZZLE_COLOR: [f32; 4] = [1.0, 0.96, 0.80, 1.0];
/// How much of a shot's own length the flash covers — a FRACTION, so a
/// point-blank shot still has one and a four-hundred-metre one does not have a
/// forty-metre flash.
const MUZZLE_FLASH_FRACTION: f32 = 0.02;

/// An extinguish line's colour (wave EMS2) — a pale blue-white, so a hose does
/// not read as a tracer.
const EXTINGUISH_COLOR: [f32; 4] = [0.72, 0.86, 1.0, 0.9];

/// Fill `scene` from `sim`'s world, blending actor positions by `alpha`.
/// Deterministic `Guid` iteration order. `vmeshes` resolves a `MeshRef.asset` to
/// its cook-derived meshlet DAG (P13.4) — a resolved mesh renders real geometry
/// (meshlet path or classic fallback), an unresolved one falls back to a placeholder
/// cube instance (as before).
///
/// `pub` for the same reason [`project_terrain`] is: this DTO is the **entire**
/// input the renderer consumes, so a gate can assert what a frame would draw —
/// two terrains, their ids, their resident pages — without a GPU, and compare it
/// between a cooked run and an editor-document run.
pub fn project_scene(
    scene: &mut RenderScene,
    sim: &RuntimeSim,
    alpha: f64,
    vmeshes: &VmeshRegistry,
) {
    project_scene_with_skinned(
        scene,
        sim,
        alpha,
        vmeshes,
        &SkinnedRegistry::new(),
        &inf_voxel::VoxelVolumes::new(),
    );
}

/// [`project_scene`] plus the skeletal store a bound `SkeletalMesh` resolves
/// against — the **whole** projection, and what the shipped render host calls.
///
/// The four-argument [`project_scene`] is kept as the narrower door because a
/// dozen existing gates drive it and none of them carry a character; it projects
/// with an inert registry, which makes every `SkeletalMesh` fall back to its
/// placeholder exactly as it did before this batch. A host that has skeletal
/// content must call **this**.
pub fn project_scene_with_skinned(
    scene: &mut RenderScene,
    sim: &RuntimeSim,
    alpha: f64,
    vmeshes: &VmeshRegistry,
    skinned: &SkinnedRegistry,
    voxels: &inf_voxel::VoxelVolumes,
) {
    // The narrow doors pack their rubble fresh — which is what they always did,
    // and none of the dozen gates that drive them carries a broken destructible.
    // The shipped host holds a real cache and calls `project_scene_full`.
    project_scene_full(
        scene,
        sim,
        alpha,
        vmeshes,
        skinned,
        voxels,
        &mut inf_render::DebrisCache::default(),
        None,
        // The narrow doors carry no scatter-mesh table for the same reason they
        // carry no skeletal store: a dozen gates drive them and none has cover
        // content. Every scattered instance falls back to its placeholder
        // primitive, which is exactly what it did before wave TER2b.
        &inf_render::ScatterMeshes::new(),
        // …and no material records, for the same reason: a narrow door has no
        // material store, so a sectioned body keeps its entity's own surface on
        // every section. Unchanged for every caller, all of which carry
        // one-slot content.
        &std::collections::HashMap::new(),
    );
}

/// [`project_scene_with_skinned`] plus the host's **debris memo** (P22.4) — the
/// widest door, and the one the shipped render host calls.
///
/// The memo is a parameter rather than a global because a projection must stay a
/// pure function of `(sim, stores)`: two hosts projecting the same sim must
/// produce the same scene, and a cache that outlived a level would make the
/// second projection depend on the first. It is keyed by fracture generation, so
/// a hit and a miss produce byte-identical payloads — the memo can only change
/// how often the packing runs, never what it packs.
///
/// `vt` is the level's virtual-texture registry (P26.4), or `None` for a world
/// with no material bindings — in which case every instance's set is
/// `VtTextureSet::NONE` and the surfaces render off their scalar attributes,
/// exactly as they did before P26.
/// **One material's surface, as `inf_render::skinned_sections` takes it**:
/// `(colour, metallic, roughness, emissive, blend code, alpha cutoff)`.
///
/// A named type rather than an inline tuple for the reason
/// `clippy::type_complexity` gives, and it is the shape the ONE Ring-0 door
/// takes — so naming it here names it for both hosts' lookups.
///
/// **MIRROR**: keep byte-identical with the other host's.
type DerivedSurface = ([f32; 4], f32, f32, [f32; 3], u8, f32);

/// One derived material's surface, as `inf_render::skinned_sections` takes it.
///
/// The colour, the three PBR terms, the blend CODE (`0` opaque, `1` masked, `2`
/// translucent — `blend_code`'s own numbering, because a section is a surface and
/// the fragment stage reading it is the same one) and the alpha cutoff. `None`
/// for a GUID this host has no record for, which makes the section keep the
/// entity's own surface rather than draw in the renderer's neutral grey.
///
/// **MIRROR** of the viewport host's `derived_surface`, byte for byte.
fn derived_surface(
    materials: &std::collections::HashMap<uuid::Uuid, inf_asset::DerivedMaterial>,
    guid: u128,
) -> Option<DerivedSurface> {
    let m = materials.get(&uuid::Uuid::from_u128(guid))?;
    Some((
        m.base_color,
        m.metallic,
        m.roughness,
        m.emissive,
        match m.blend {
            inf_asset::DerivedBlend::Masked => 1,
            inf_asset::DerivedBlend::Translucent => 2,
            _ => 0,
        },
        m.alpha_cutoff,
    ))
}

#[allow(clippy::too_many_arguments)]
pub fn project_scene_full(
    scene: &mut RenderScene,
    sim: &RuntimeSim,
    alpha: f64,
    vmeshes: &VmeshRegistry,
    skinned: &SkinnedRegistry,
    voxels: &inf_voxel::VoxelVolumes,
    debris: &mut inf_render::DebrisCache,
    vt: Option<&inf_render::VtTextures>,
    scatter_meshes: &inf_render::ScatterMeshes,
    // **The level's derived materials** (wave CHAR1a.3), keyed by the `.inf_mat`
    // GUID a scene binds — the same map this host hands `build_vt_level`. TENTH
    // and last rather than beside `vt`, which it most resembles, because
    // appending is what the tail of this signature has always done and a reader
    // comparing two call sites across a wave should not have to count.
    //
    // A skinned mesh's material SECTIONS need a surface per slot, and a slot is
    // named by the `.inf_mesh` rather than by any component — so the projector
    // has to be able to ask "what does this GUID draw as?" about a material no
    // entity binds. It is the same question `Material.asset` asks, of the same
    // store, one indirection further out.
    materials: &std::collections::HashMap<uuid::Uuid, inf_asset::DerivedMaterial>,
) {
    // The virtual-texture LIBRARY, bound under a name the entity walk does not
    // shadow: inside it, `vt` is the SET one entity's material resolved to.
    let vt_lib = vt;
    scene.instances.clear();
    scene.lights.clear();
    scene.sprites.clear();
    scene.tilemaps.clear();
    scene.prebatched.clear();
    scene.lights_2d.clear();
    scene.vgeom_assets.clear();
    scene.vgeom_instances.clear();
    // P18.3's follow-up: real skinned geometry. Both lists are rebuilt from
    // scratch every projection, exactly like `instances` — the bind-space payload
    // behind each entry is `Arc`-shared with the store, so re-projecting an
    // unchanged character re-uses the GPU upload even though the list is rebuilt.
    scene.skinned_meshes.clear();
    scene.skinned.clear();
    // P16.3b1 + Hardening Wave E: terrains are NOT cleared — they are
    // **stamp-gated**, like `deform` below and unlike everything above. Last
    // frame's list is taken out and each entry is either carried forward whole
    // (nothing about its grid, layers or per-tile stamps moved) or dropped and
    // rebuilt. What is left in `prev_terrains` at the end of the walk is exactly
    // the terrains that left the scene, which is how a disappearance is seen.
    let mut prev_terrains = std::mem::take(&mut scene.terrains);
    // P21.1 + Hardening Wave E: volumetric terrain, on the same stamp gate. The
    // meshed surface behind each chunk lives in the host's store and is re-meshed
    // only where the field moved, so a settled cave now costs the comparison of
    // its chunk stamps and no copying at all — it used to cost two rebases of
    // every vertex stream, a mapped copy and an index clone, every frame.
    let mut prev_voxels = std::mem::take(&mut scene.voxels);
    let (terrains_before, voxels_before) = (prev_terrains.len(), prev_voxels.len());
    scene.fracture_chunks.clear();
    // P18.5 + island wave I8a's audit: the scatter LIST is rebuilt from scratch
    // every projection, and the PAYLOADS behind it are not. Last projection's
    // memo is taken out here and re-filled by `carry_or_push_pcg_scatter` as the
    // walk goes; what is left in it at the end is exactly the scatter that left
    // the scene, and it is dropped — the terrain memo's own arrangement.
    //
    // MIRROR: `inf_viewport::host::rebuild_scene` takes the same two locals under
    // the same names, and takes them AFTER its own `sync_scatter_meshes` because
    // the editor rebuilds the mesh table per projection.
    scene.scatter.clear();
    let mut prev_scatter = std::mem::take(&mut scene.scatter_memo);
    let scatter_table = inf_render::scatter_table_stamp(scatter_meshes);
    // P20.1: water bodies are rebuilt from scratch every projection, like
    // `scatter` — a body's whole state is a pure function of its component, its
    // spline and the level clock, so there is nothing to carry over.
    scene.waters.clear();
    // P22.1: the deformation field is NOT cleared — it is epoch-gated, because it
    // is the one projected thing that is usually identical to last frame's. See
    // `project_deform`, which is also where `None` is written when there is no
    // field at all.
    project_deform(scene, inf_ecs::deform::deform_field(sim.world()));
    // Track which vmesh assets are already listed this frame (dedup — the render
    // node caches GPU geometry by id, but the asset list must not duplicate), and
    // which `(mesh, skeleton)` pairs already own a `skinned_meshes` slot.
    // MIRROR: both are `inf_viewport::host::rebuild_scene`'s locals of the same
    // names and the same purpose.
    let mut vgeom_seen: std::collections::HashSet<u128> = std::collections::HashSet::new();
    let mut skinned_slots: std::collections::HashMap<(Uuid, Uuid), usize> =
        std::collections::HashMap::new();

    let world = sim.world();
    // The sky authority first (P17.1): it writes `scene.sun` / `scene.sky` and,
    // when a clock is present, pushes the sun/moon directional light as
    // `lights[0]` — a stable index on both projector sides.
    project_sky(scene, world);

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
        glow_step: inf_render::night_glow_step(scene.sun.direction),
        pulse_tick: inf_render::pulse_tick(clock_s),
    };
    // The clock and wind every water body responds to, resolved ONCE per
    // projection in Ring 0 (`inf_ecs::sky`) so the two MIRROR projectors cannot
    // disagree about what "now" and "the wind" mean — the same reasoning that put
    // `ResolvedSky::cloud_time_s` there.
    let water_env = inf_ecs::sky::water_environment(world);
    // P20.4: the level's terrains, borrowed once, so a river's foam can read the
    // P19.1 flow map. MIRROR of the editor host's line.
    let water_flow = inf_ecs::hydro::terrain_flow(world);
    let w = world.world();

    // Guid-sorted entity list (mirrors doc.order()'s determinism without a doc).
    let mut ents: Vec<(Uuid, inf_ecs::Entity)> = w
        .iter_entities()
        .filter_map(|e| e.get::<Guid>().map(|g| (g.0, e.id())))
        .collect();
    ents.sort_by_key(|(g, _)| *g);

    let mut next_id: u32 = 1;
    for (guid, entity) in ents {
        let visible = w
            .get::<ComputedVisibility>(entity)
            .map(|c| c.0)
            .unwrap_or(true);
        if !visible {
            continue;
        }

        // Interpolated translation for actors; static geometry uses its global.
        let base = w
            .get::<GlobalTransform>(entity)
            .map(|g| g.translation())
            .unwrap_or(DVec3::ZERO);
        let translation = sim.interp_translation(guid, alpha).unwrap_or(base);

        if let Some(light) = w.get::<Light>(entity) {
            let affine = w
                .get::<GlobalTransform>(entity)
                .map(|g| g.0)
                .unwrap_or(glam::DAffine3::IDENTITY);
            scene.lights.push(project_light(light, &affine));
        }
        if let Some(sprite) = w.get::<Sprite>(entity) {
            scene.sprites.push(project_sprite(sprite, translation));
        }
        if let Some(light2d) = w.get::<Light2D>(entity) {
            scene.lights_2d.push(project_light2d(light2d, translation));
        }
        if let Some(nine) = w.get::<NineSlice>(entity) {
            scene.prebatched.push(project_nine_slice(nine, translation));
        }
        if let Some(text) = w.get::<Text2D>(entity) {
            if let Some(run) = project_text(text, translation) {
                scene.prebatched.push(run);
            }
        }
        if let Some(tilemap) = w.get::<Tilemap>(entity) {
            if !tilemap.is_empty() {
                scene.tilemaps.push(project_tilemap(tilemap, translation));
            }
        }
        // Heightfield terrain (P10.6): the player projects **every** visible,
        // non-empty terrain into the render scene's terrain list, exactly like the
        // editor viewport host (`inf_viewport::host::project_terrain`), so
        // cooked/PIE terrain renders. Per-tile change stamps ride along (P16.3b1),
        // so the terrain pass re-uploads a height texture only when that tile
        // really changed.
        //
        // P16.6 — MULTI-TERRAIN: the old "first visible terrain wins" rule is
        // gone. Terrains arrive in `Guid` order (this loop's order), and each
        // carries `terrain_id_from_guid(guid)` so the renderer's per-tile texture
        // cache and per-terrain splat uniform stay separate — two terrains
        // routinely share tile coordinates.
        //
        // MIRROR, precisely: the editor viewport emits terrains in the DOCUMENT's
        // entity order, not `Guid` order. Both are deterministic for their own
        // side; what makes a PIE-vs-shipping comparison of the projected scene
        // meaningful is that both stamp the SAME `id` from the entity `Guid`, so
        // the two lists match up by identity rather than by index.
        //
        // P16.3b2 — THE SIM/RENDER SPLIT: an asset-backed terrain draws the
        // **streamer's** camera-driven working set, not the component's. The
        // component's set is the sim's (level-0 pages around the sim's entities);
        // projecting it would put the camera's cut and the sim's residency in the
        // same container, which is exactly the coupling the doctrine forbids.
        // An inline terrain has no streamer and projects its own data, unchanged.
        if let Some(terrain) = w.get::<Terrain>(entity) {
            let data = sim
                .terrain_streaming()
                .render_data(guid)
                .unwrap_or(&terrain.data);
            if !data.is_empty() || data.coarse_tile_count() > 0 {
                let id = inf_render::terrain_id_from_guid(guid.as_u128());
                scene.terrains.push(project_terrain(
                    terrain,
                    data,
                    translation,
                    id,
                    vt,
                    &mut prev_terrains,
                ));
            }
        }
        // Volumetric terrain (P21.1): the SDF chunk volume that locally extends
        // the heightfield — caves, tunnels, overhangs. Its chunks live in a
        // `.inf_voxel` the host loaded before this walk started; a volume the world
        // could not serve simply has no slot and draws nothing.
        // MIRROR: `inf_viewport::host` runs the same branch (minus the visibility
        // gate and the pick-id map, both host-local) through the same
        // `project_voxel` body.
        if let Some(volume) = w.get::<VoxelVolume>(entity) {
            let projected = volume
                .asset
                .and_then(|_| voxels.get(guid.as_u128()))
                .and_then(|slot| {
                    project_voxel(
                        slot,
                        w.get::<Terrain>(entity),
                        translation,
                        inf_render::terrain_id_from_guid(guid.as_u128()),
                        vt,
                        &mut prev_voxels,
                    )
                });
            if let Some(rv) = projected {
                scene.voxels.push(rv);
            }
        }
        // PCG scatter volumes (P18.5): the volume's evaluated instance cache
        // (populated on load by the level builder) projects as ONE GPU-instanced
        // scatter batch instead of one `MeshInstance` per instance. The payload
        // uploads once per content change and the cull compute does frustum + HZB
        // + distance banding per instance.
        //
        // The volume's authored `draw_distance` now RIDES ON THE BATCH rather than
        // being culled by the host, and that is what finally makes the two hosts
        // agree about it: the editor used to cull against its own camera eye on the
        // CPU while the player ignored the field entirely, so a shipped build drew
        // strictly more scatter than its preview.
        //
        // MIRROR: `push_pcg_scatter` matches `inf_viewport::host`'s PCG projection
        // (minus its pick-id map).
        if let Some(vol) = w.get::<PcgVolume>(entity) {
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
                scene.lights.push(RenderLight {
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
            if !vol.evaluated.is_empty() {
                let id = next_id;
                next_id += 1;
                carry_or_push_pcg_scatter(
                    scene,
                    &mut prev_scatter,
                    scatter_table,
                    guid,
                    vol,
                    scatter_meshes,
                    translation,
                    id,
                    clock,
                );
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
        // from a volume's scatter.
        //
        // MIRROR: `inf_viewport::host` runs the same branch (plus the visibility
        // gate and the pick-id map, both host-local).
        if let Some(terrain) = w.get::<Terrain>(entity) {
            if !terrain.biome_population.is_empty() {
                let id = next_id;
                next_id += 1;
                push_biome_population(scene, terrain, scatter_meshes, translation, id, clock);
            }
        }
        // Water surfaces (P20.1): an ocean, a lake or a spline river. A river
        // reads the `Spline` on THIS SAME ENTITY for its centreline — component
        // composition, not a reference, so there is nothing to resolve and
        // nothing to dangle.
        // MIRROR: `inf_viewport::host` runs the same branch (minus the pick-id
        // map, which is host-local), through the same `project_water` body.
        if let Some(water) = w.get::<WaterBody>(entity) {
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
                scene.waters.push(body);
            }
        }
        // Foliage scatter (P18.5): painted instances project as GPU-instanced
        // scatter batches, one per primitive kind the palette resolves.
        // MIRROR: `push_foliage_scatter` matches `inf_viewport::host`'s Foliage
        // projection so the shipped player and the editor viewport draw the same
        // scatter.
        if let Some(fol) = w.get::<Foliage>(entity) {
            if !fol.instances.is_empty() {
                let id = next_id;
                next_id += 1;
                push_foliage_scatter(scene, fol, translation, id);
            }
        }
        // Skeletal meshes (P11.1 → the P18.3 follow-up): a `SkeletalMesh` entity
        // draws its REAL skinned geometry. The bind-space mesh comes from the
        // referenced `.inf_mesh`'s skin streams, the palette from the `.inf_skel`
        // posed by the entity's `AnimPlayer` — rest pose when there is no player,
        // no clip, or an unresolvable one, so a character in a freshly loaded
        // level is visible immediately rather than only once it plays. Both the
        // resolution and the pose rule live in [`crate::skinned`], which is where
        // the editor's Ring-1 store is mirrored character for character.
        //
        // The **placeholder cube survives** as the honest fallback: a
        // `SkeletalMesh` with no assets bound (or with a mesh carrying no skin
        // stream) is still authorable content that must draw as *something*.
        //
        // MIRROR of `inf_viewport::host::rebuild_scene`'s skeletal branch, pinned
        // field for field by `inf-editor-core`'s `tests/projector_mirror.rs`.
        // Until this batch the editor had this branch and the shipped player had
        // none at all, so a level with a character previewed in PIE and shipped as
        // nothing — a live PIE-vs-shipping divergence, not a missing feature.
        //
        // Host-local, as on the vgeom path: `translation` is the sim's
        // **interpolated** actor position here and the raw affine's in the editor
        // (the editor has no fixed-step interpolation to do), and `id` comes from
        // this host's own counter over `Guid` order.
        // ── P24.4 cloth ── the garment the sim folded on this entity this step,
        //    drawn beside whatever else it draws. NOT inside the `MeshRef`-absent
        //    branch below: a garment is worn by an entity, not instead of its
        //    geometry, so a character with a static mesh and a cloak draws both.
        //    Its own affine, because the branch below computes one only when the
        //    entity is skeletal. (MIRROR of the other host's call.)
        {
            let affine = w
                .get::<GlobalTransform>(entity)
                .map(|g| g.0)
                .unwrap_or(glam::DAffine3::IDENTITY);
            let (cloth_scale, cloth_rot, _t) = affine.to_scale_rotation_translation();
            // ── P29.7 character space ── a garment's vertices are in the
            //    wearer's MODEL space (feet at the origin) and this translation
            //    is the entity's (a capsule CENTRE), so the lift goes through
            //    the one door that knows the difference. Without it a coat is
            //    drawn nearly a metre above the character wearing it. Zero for
            //    an entity with no movement component, which is every prop.
            let cloth_at = translation + inf_ecs::pose::model_offset_world(world, entity);
            project_cloth(
                scene,
                world,
                guid,
                cloth_at,
                cloth_rot.as_quat(),
                cloth_scale.as_vec3(),
            );
            project_hair(
                scene,
                world,
                guid,
                cloth_at,
                cloth_rot.as_quat(),
                cloth_scale.as_vec3(),
            );
        }
        if w.get::<MeshRef>(entity).is_none() {
            if let Some(sm) = w.get::<SkeletalMesh>(entity).copied() {
                // ── P29.6 character space ── a rig's origin is its FEET and a
                //    character's entity transform is its capsule CENTRE, so the
                //    pose is drawn through the one door that knows the difference
                //    (`inf_ecs::pose::model_to_world`, identity-composed for
                //    everything that is not a character). Applied to the
                //    *interpolated* translation as a world delta rather than
                //    taken from the affine, because this host interpolates actor
                //    positions and the editor does not — the drop is the same
                //    number either way. (MIRROR of the other host's call.)
                let affine = inf_ecs::pose::model_to_world(world, entity);
                let drop = affine.translation
                    - w.get::<GlobalTransform>(entity)
                        .map(|g| g.translation())
                        .unwrap_or(affine.translation);
                let translation = translation + drop;
                let (scale, rot, _t) = affine.to_scale_rotation_translation();
                let id = next_id;
                next_id += 1;
                let player = w.get::<inf_ecs::components::AnimPlayer>(entity).copied();
                // **The machine, for the preview idle** (wave CHAR1a.2). Read the same
                // way the player is, and handed to the same door: with no sim pose and no
                // `AnimPlayer`, a character that carries a state machine is drawn in that
                // machine's ENTRY state at t = 0 instead of its bind pose. Outside Play
                // that is every authored character in the level, which is why the viewport
                // used to be full of T-poses.
                let machine = w
                    .get::<inf_ecs::components::AnimStateMachine>(entity)
                    .copied();
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
                // **`blend` and `cutoff` ride here too since wave CHAR1a.2**, from
                // the same `Material` and through the same `blend_code` door as the
                // rigid path's: a hair card, an eyelash sheet or a cut-out garment
                // is a skinned surface, and until this line the skinned path had no
                // way to carry "this fragment is a hole".
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
                        skinned.resolve_skinned_shared(&sm, machine.as_ref())
                    }
                    _ => skinned.resolve_skinned(&sm, player.as_ref(), posed, machine.as_ref()),
                };
                match resolved {
                    Some(draw) => {
                        // One `skinned_meshes` entry per (mesh, skeleton) pair,
                        // and the entry is the store's own `Arc` — no copy here,
                        // and the pass keys its GPU upload on that pointer, so
                        // re-projecting an unchanged character costs neither a
                        // memcpy nor a re-upload (P18.3). **This is the sharing
                        // convention the projector has to follow**, and it is the
                        // reason `skinned_meshes` is a `Vec<Arc<_>>` at all.
                        let slot = *skinned_slots.entry(draw.key).or_insert_with(|| {
                            scene.skinned_meshes.push(draw.mesh);
                            scene.skinned_meshes.len() - 1
                        });
                        let inst = SkinnedInstance {
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
                            blend,
                            cutoff,
                            palette: draw.palette,
                            shadow,
                            sections: Vec::new(),
                        };
                        // **THE SECTIONS** (wave CHAR1a.3): one drawn range per
                        // material slot the `.inf_mesh` names, each with that
                        // slot's own surface and virtual textures, all sharing
                        // this instance's palette. EMPTY for a one-slot body —
                        // every committed character, every crowd agent, and the
                        // MetaHuman full-body mesh — which then draws exactly the
                        // one whole-buffer command it drew before sections
                        // existed. Through `inf_render::skinned_sections`, which
                        // is the ONE door: the rule is Ring 0's, the material
                        // lookup is this host's.
                        let sections = inf_render::skinned_sections(
                            &draw.sections,
                            &inst,
                            |g| derived_surface(materials, g),
                            |g| inf_render::vt_set_for(vt_lib, Some(g)),
                        );
                        scene.skinned.push(SkinnedInstance { sections, ..inst });
                    }
                    // Unbound (or unskinned) — the editor's placeholder, down to
                    // its slate tint, so the two hosts also agree about content
                    // whose assets are missing.
                    None => scene.instances.push(MeshInstance {
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
                        // Wave CHAR1a.2: the placeholder wears the SAME blend the
                        // resolved character would. A masked surface whose skeleton
                        // has not loaded is still a masked surface; the hard-coded
                        // `0`/`0.5` was only ever right because the skinned path
                        // could not be masked at all.
                        blend,
                        cutoff,
                    }),
                }
            }
        }
        if let Some(mesh_ref) = w.get::<MeshRef>(entity) {
            let affine = w
                .get::<GlobalTransform>(entity)
                .map(|g| g.0)
                .unwrap_or(glam::DAffine3::IDENTITY);
            let (scale, rot, _t) = affine.to_scale_rotation_translation();
            // MIRROR: this Material→MeshInstance projection is duplicated in the
            // editor viewport's `host.rs` (inf-viewport) — keep the two in sync,
            // R-P5 blend + cutoff included. (The vgeom path below is opaque-only —
            // vgeom translucency is deferred.)
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

            // P13.4: a MeshRef.asset with a cook-derived vmesh renders REAL geometry
            // — the GPU meshlet path (vgeom on) or the classic discrete-LOD fallback
            // (vgeom off), both driven by the same vgeom scene content. The tier the
            // renderer settings carry picks which node draws it. An unresolved asset
            // (or a primitive-only MeshRef) falls back to a placeholder cube.
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
            let broken = sim.fractures().get(&guid).and_then(|state| {
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
            });
            let fractured = broken.is_some();
            if let Some((chunks, rubble)) = broken {
                scene.fracture_chunks.extend(chunks);
                scene.scatter.extend(rubble);
            }
            let vgeom = (!fractured)
                .then(|| mesh_ref.asset.and_then(|mesh_id| vmeshes.resolve(mesh_id)))
                .flatten();
            if fractured {
                // Its chunks are already in the scene; nothing else to push.
            } else if let Some((asset_id, source)) = vgeom {
                if vgeom_seen.insert(asset_id) {
                    // The scene carries the PAGED source, not a decoded DAG
                    // (P18.2): the render node's streamer decides what of it is
                    // resident from the camera's own screen-error wants.
                    scene.vgeom_assets.push(VgeomAsset::new(asset_id, source));
                }
                scene.vgeom_instances.push(VgeomInstance {
                    vt,
                    asset: asset_id,
                    translation,
                    rotation: rot.as_quat(),
                    scale: scale.as_vec3(),
                    color,
                    metallic,
                    roughness,
                    emissive,
                    id: next_id,
                });
            } else if mesh_ref.asset.is_some() {
                // **Wave FIX2: a bound mesh with no DAG draws NOTHING.**
                //
                // It used to draw `prim_mesh(mesh_ref.primitive)` — a 1 m cube for
                // `MeshRef::default()` — which is a claim about the world no
                // author made, and on the island it hid four missing streets
                // behind four boxes at the origin 2.7 km from the spawn. The
                // registry states the miss once per asset
                // (`VmeshRegistry::report_missing`), which is the Output Log line
                // this branch replaced the box with.
                //
                // Both hosts, not just PIE: the editor viewport's projector has
                // the identical arm, because a placeholder deleted on one side
                // only is a new way for a preview and a build to differ.
            } else {
                // R-P1: an unresolved / primitive-only MeshRef draws its built-in
                // primitive kind (Sphere/Plane/Cylinder/Cone), not always a cube.
                scene.instances.push(MeshInstance {
                    vt,
                    translation,
                    rotation: rot.as_quat(),
                    scale: scale.as_vec3(),
                    color,
                    metallic,
                    roughness,
                    emissive,
                    id: next_id,
                    mesh: prim_mesh(mesh_ref.primitive),
                    blend,
                    cutoff,
                });
            }
            next_id += 1;
        }
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
    // Hardening Wave E: and now not even that. The seam is a pure function of the
    // volumes' vertices and the terrains, so when EVERY volume and EVERY terrain
    // in this scene was carried forward unchanged — nothing dropped, nothing
    // added, nothing rebuilt — last frame's terms are still the right ones, and
    // the per-vertex walk (which samples every terrain per vertex) is skipped
    // whole. `carried == pushed && prev.is_empty()` is the exact statement of
    // "nothing changed on this axis": the first half sees an addition or a
    // rebuild, the second sees a removal.
    if !(prev_terrains.is_empty()
        && prev_voxels.is_empty()
        && terrains_before - prev_terrains.len() == scene.terrains.len()
        && voxels_before - prev_voxels.len() == scene.voxels.len())
    {
        inf_render::apply_seam(
            &mut scene.voxels,
            &scene.terrains,
            inf_render::DEFAULT_SEAM_BAND_M,
        );
    }

    // **Round-2 finding R2-2**, MIRROR of the editor host: the debris memo is
    // retained against the live fracture set. `DebrisCache::batch` drops an
    // entry when the actor sheds nothing, which covers a reclaim; a DESPAWN —
    // a deleted actor, a streamed cell that unloads — never calls it again and
    // left the packed payload for the session.
    {
        let live: std::collections::BTreeSet<u64> = sim
            .fractures()
            .keys()
            .map(|g| inf_render::terrain_id_from_guid(g.as_u128()))
            .collect();
        debris.retain_live(&live);
    }

    scene.mark_dirty();
}

/// Map an ECS [`Primitive`] to the renderer's [`PrimMesh`] (R-P1).
///
/// MIRROR: keep identical to `inf_viewport::host::prim_mesh` (the editor
/// viewport's ECS→RenderScene projection). Both seams must agree so the shipped
/// player and the editor viewport draw the same geometry for a given primitive.
fn prim_mesh(p: Primitive) -> PrimMesh {
    match p {
        Primitive::Cube => PrimMesh::Cube,
        Primitive::Sphere => PrimMesh::Sphere,
        Primitive::Plane => PrimMesh::Plane,
        Primitive::Cylinder => PrimMesh::Cylinder,
        Primitive::Cone => PrimMesh::Cone,
    }
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

/// Project a [`PcgVolume`]'s evaluated cache into GPU-instanced scatter batches
/// (P18.5), anchored at the volume entity's world `translation`, carrying the
/// volume's authored content draw distance.
///
/// # The structure LOD (IB-2b)
///
/// A volume that grew **buildings** carries `structure_groups`, and its parts are
/// swapped for one **shell** box per building past
/// [`STRUCTURE_LOD_M`](inf_render::STRUCTURE_LOD_M). That is
/// three batches, not one, and their distance bands are *complementary*:
///
/// | batch | band | what it holds |
/// |---|---|---|
/// | ungrouped | `[0, draw_distance)` | scatter and fences — content that has no shell to stand in for it |
/// | parts | `[0, lod + reach)` | every building's own boxes |
/// | shells | `[lod, draw_distance)` | one oriented box per building |
///
/// `reach` is the widest shell's own half-diagonal, and it is there because the
/// bands are complementary in the **group's** distance while the cull is per
/// **instance** (I3 audit): without it a building straddling the line loses its
/// far parts and grows no shell — a hole through its back. The two bands
/// therefore overlap by `reach` rather than meeting, and in that overlap a
/// building draws its parts inside its own shell, which contains them.
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
/// [`ScatterSource`] carries the volume's `Guid`, its **process-global**
/// `structures_gen`, the authored `draw_distance` the stamp does not cover, the
/// mesh table's own stamp and the world anchor the offsets were packed against.
/// The population stamp is drawn from a process-global monotone counter
/// (`inf_ecs`'s `NEXT_STRUCTURES_GEN`), so it names one population of one volume
/// for the life of the process — including across the destroy-and-rebuild a cell
/// deactivation and reactivation performs under the same guid, which is what a
/// per-volume counter could not do. `0` is "never written" and is a forced miss.
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

/// Euler-degrees (YXZ) → quaternion for a foliage instance's stored rotation,
/// matching `inf_ecs::Transform::quat` (and the editor viewport's mirror) exactly.
fn foliage_rot_quat(rot: Vec3d) -> glam::Quat {
    glam::DQuat::from_euler(
        glam::EulerRot::YXZ,
        rot.y.to_radians(),
        rot.x.to_radians(),
        rot.z.to_radians(),
    )
    .as_quat()
}

/// Project the ECS [`BlendMode`] into the renderer's packed `blend` code (R-P5):
/// 0 opaque, 1 masked, 2 translucent. Mirrored in the editor viewport's `host.rs`.
fn blend_code(b: BlendMode) -> u8 {
    match b {
        BlendMode::Opaque => 0,
        BlendMode::Masked => 1,
        BlendMode::Translucent => 2,
    }
}

/// **The settings a shipped player runs**, for `record` on `gpu`'s adapter —
/// the level's authored render block, the platform request, the auto-tier and
/// the capability clamp, in that order.
///
/// Extracted from [`PlayerRenderHost::new`] (island wave I4) and called by it, so
/// there is exactly **one** answer to "what does the player render with". The
/// caller that made this necessary is the fps instrument: a harness that rebuilt
/// this chain by hand would measure a configuration nobody ships, and the first
/// thing it would get wrong is the pair of clamps below — which is the very
/// mistake P18.1 found this host making against the editor's.
///
/// Returns the settings **and** the tier, because a caller that reports a frame
/// time without saying which tier produced it has reported a number about an
/// unnamed machine.
pub fn shipped_settings(
    gpu: &GpuContext,
    record: RenderSettingsRecord,
) -> (RenderSettings, inf_render::RenderTier) {
    let base = apply_record(&record);
    let tier = detect_tier(gpu, &base);
    // Desktop requests the meshlet path (the tier clamps it on Medium/Low);
    // mobile/web (P14.1) clamps the level's block down to the mobile ceiling —
    // no vgeom, no SSAO/GI/TAA/bloom/shadows — then the live-adapter tier
    // applies on top (Low still drops what little remains).
    #[cfg(any(target_arch = "wasm32", target_os = "android"))]
    let requested = inf_render::RenderTier::clamp_mobile(base);
    #[cfg(not(any(target_arch = "wasm32", target_os = "android")))]
    let requested = RenderSettings {
        vgeom: inf_render::VgeomSettings {
            enabled: true,
            ..base.vgeom
        },
        ..base
    };
    // BOTH gates, not just the tier — the pair `inf_render::detect_and_clamp`
    // is, inlined only so the adapter is probed once here rather than a
    // second time. The editor viewport has applied both since P18.1; this
    // host applied the tier alone, which meant every *capability* clamp
    // (`vgeom.occlusion`/`two_pass`, scatter, and P26.1's `vt.bc_tiles`) was
    // granted in the shipped player on adapters that cannot run it, and
    // clamped in the editor — two different renderers on one machine.
    // `clamp_occlusion` only ever turns things off, so this can never grant
    // the level's block more than it asked for.
    (
        AdapterCaps::probe(gpu).clamp_occlusion(tier.apply(requested)),
        tier,
    )
}

/// Map the level's scene-persisted [`RenderSettingsRecord`] onto a live
/// [`RenderSettings`] (R-P4). The record carries the authorable subset; every
/// other field (hdr, vgeom, tier_override, and the shadow/GI tuning knobs) stays
/// at `RenderSettings::default()`, so
/// `apply_record(&RenderSettingsRecord::default()) == RenderSettings::default()`
/// — pinned by the unit test below.
///
/// MIRROR: keep identical to `inf_viewport::host::apply_record` (the editor
/// viewport's copy over the editor-core `RenderSettingsRecord`). Both seams must
/// agree so the shipped player and the editor viewport apply a level's render
/// block the same way (preview == shipping).
///
/// **Public since wave FIX3** so `tests/lit_stack.rs` can restate the editor
/// viewport's settings chain without carrying a third copy of this mapping. No
/// crate in the repository depends on both hosts, so that arm has to transcribe
/// one of them; using this copy is sound precisely because
/// `tests/apply_record_mirror.rs` pins the two bodies character-for-character.
pub fn apply_record(r: &RenderSettingsRecord) -> RenderSettings {
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
        // The cloud's own temporal accumulation follows the level's TAA switch
        // (wave SKY2). One authored bit, two mechanisms — and it has to be two,
        // because `passes::taa` reprojects through the depth prepass and a cloud
        // writes no depth, so it cannot ride that pass. What the author is
        // deciding with `taa` is a POLICY ("this level accepts a frame that
        // depends on the frames before it"), and that policy governs both.
        // Riding the existing bit is also what keeps the wave at zero new
        // authored fields: a `cloud_temporal` of its own would be a
        // `RenderSettingsRecord` field, and that record is scene schema.
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

/// Project an ECS [`Terrain`] (+ world translation) into a [`RenderTerrain`],
/// mirroring `inf_viewport::host::project_terrain`: each **resident** tile becomes
/// a [`RenderTerrainTile`] (heights + resolved RGBA8 splat weights + resolved
/// biome ids + precomputed height bounds + its monotone change stamp), plus the
/// four material layers + macro variation.
///
/// `data` is the working set to draw and is passed **explicitly** (P16.3b2): for
/// an inline terrain it is `terrain.data`, for a streamed one it is the
/// streamer's camera-driven set. `terrain` still supplies the layers and macro
/// variation, which are authored, not streamed. Making the choice a parameter is
/// what keeps "which residency am I drawing?" a decision at the call site rather
/// than an assumption buried here.
///
/// Level 0 (the authored heightfield) is emitted first, then the resident coarse
/// pyramid pages in ascending key order — both from `BTreeMap`s, so the tile list
/// is globally `TileKey`-ascending and the upload/draw order is deterministic.
///
/// The stamps are what keep the GPU cache hot: `project_scene` rebuilds this DTO
/// every frame, but a tile's stamp only advances when the tile is actually
/// mutated, so a static (or streamed-but-settled) terrain re-uploads nothing.
/// That replaces the old constant `TERRAIN_VERSION` — which was correct only
/// while terrain could never change (P16.3b1).
///
/// **MIRROR** of `inf_viewport::host::project_terrain` — keep the two in sync.
///
/// `pub` so the streaming gate can assert **rendered-frame determinism** without a
/// GPU: the DTO this returns is the entire input to the terrain pass, so hashing
/// it across two runs is exactly "the same frame was drawn". (Excluding
/// `version`, which is a process-global cache stamp and deliberately not
/// reproducible — see `TerrainData::tile_version`.)
///
/// `id` is the terrain entity's identity fold (P16.6), and `prev` is the
/// previous frame's terrain list, which the caller has taken out of the scene:
/// a terrain whose grid, layers and whole per-tile stamp sequence are unchanged
/// is **carried forward from it** rather than rebuilt (Hardening Wave E's P1
/// memo — see [`inf_render::take_unchanged_terrain`] for why the key is sound).
/// Pass `&mut Vec::new()` for a one-shot projection with nothing to carry.
pub fn project_terrain(
    terrain: &Terrain,
    data: &inf_terrain::TerrainData,
    translation: DVec3,
    id: u64,
    // The live virtual-texture registry (TER2a) — `None` on a level with no
    // bindings, and then every layer's set is `VtTextureSet::NONE`.
    vt: Option<&inf_render::VtTextures>,
    prev: &mut Vec<RenderTerrain>,
) -> RenderTerrain {
    let res = data.tile_resolution();
    let n = (res * res) as usize;
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
    // EMPTY on purpose (P19.2) — see the `biome_palette` note at the tail of this
    // function. Named here so the memo compares exactly what the projection
    // produces.
    let biome_palette: Vec<[f32; 4]> = Vec::new();
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
        &biome_palette,
        tile_signature(data, translation),
    ) {
        return kept;
    }
    let project_tile = |key: inf_terrain::TileKey, tile: &inf_terrain::TerrainTile| {
        // A coarse pyramid page is always unpainted (the pyramid is heights-only),
        // so it resolves to the uniform default like any unpainted tile.
        let weights: Vec<[u8; 4]> = if tile.weights_are_default() {
            vec![inf_terrain::DEFAULT_WEIGHT; n]
        } else {
            (0..res)
                .flat_map(|j| (0..res).map(move |i| (i, j)))
                .map(|(i, j)| tile.weight_sample(res, i, j))
                .collect()
        };
        // Biome ids (P19.2) resolve exactly like the weights: the sparse default is
        // expanded here so the DTO the renderer sees is always dense, and a coarse
        // pyramid page — never painted — is all-`UNASSIGNED_BIOME`.
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
        // The terrain entity's identity (P16.6), now a parameter rather than a
        // field the caller patched afterwards — the memo above has to key on it
        // before the tiles are built, so it cannot be stamped after them. `0` is
        // the "unkeyed" value the single-terrain callers (the gates' DTO
        // fingerprints) pass, exactly as they got before.
        id,
        tile_resolution: res,
        meters_per_sample: data.meters_per_sample(),
        tiles,
        layers,
        macro_variation,
        // EMPTY on purpose (P19.2). The palette is a property of the level's
        // `BiomeSet` asset, and `Terrain::biome_set` is a GUID: resolving it needs
        // an asset DB, which this projection deliberately does not have (it takes
        // an ECS component + a heightfield and nothing else — the same reason
        // layer *textures* never reached it either). The renderer pads every slot
        // with the unassigned colour, so a shipped build that somehow lands in the
        // Biomes view draws uniform neutral rather than reading a stale palette.
        // The mode is an EDITOR view mode — the viewport host, which does hold the
        // DB, is where a real palette is projected from.
        biome_palette,
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

/// A distinct placeholder colour per PCG kind index (mirrors the viewport host's
/// `pcg_kind_color`), so a multi-kind scatter reads as varied content before real
/// meshes upload.
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
/// **MIRROR** of `inf_viewport::host::project_light` — kept byte-for-byte
/// identical (the parity tests in both crates pin the shared conventions):
///  * directional/spot store the vector *toward* the light = `rot · +Z` (forward
///    is `−Z`, so this is the anti-emission direction); the renderer derives a
///    spot's beam emission as `−direction = rot · −Z`;
///  * cone half-angles → cosines CPU-side; `range`/`cast_shadows` pass through
///    for all kinds. `cast_shadows` stopped being inert for point and spot at
///    **P27.4**: a virtual shadow map gives a spot its own quadtree and a point
///    six cube-face ones, and `inf_render::vsm_light_trees` reads exactly this
///    flag. The cascaded path still shadows the first directional light only,
///    which is what P27.5 demotes.
fn project_light(light: &Light, affine: &glam::DAffine3) -> RenderLight {
    let (_, rot, translation) = affine.to_scale_rotation_translation();
    let c = light.color.to_array();
    let color = [c[0], c[1], c[2]];
    match light.kind {
        EcsLightKind::Directional => RenderLight {
            kind: LightKind::Directional,
            color,
            intensity: light.intensity,
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

fn project_sprite(sprite: &Sprite, translation: DVec3) -> inf_render::SpriteInstance {
    inf_render::SpriteInstance {
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

fn billboard_mode(mode: inf_ecs::BillboardMode) -> u8 {
    match mode {
        inf_ecs::BillboardMode::None => inf_render::BILLBOARD_NONE,
        inf_ecs::BillboardMode::Spherical => inf_render::BILLBOARD_SPHERICAL,
        inf_ecs::BillboardMode::Cylindrical => inf_render::BILLBOARD_CYLINDRICAL,
    }
}

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

fn project_light2d(light: &Light2D, translation: DVec3) -> RenderLight2D {
    let c = light.color.to_array();
    RenderLight2D {
        color: [c[0], c[1], c[2]],
        intensity: light.intensity,
        radius: light.radius,
        position: translation,
    }
}

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
        // OPAQUE, stated rather than defaulted (wave CHAR1a.2): a simulated
        // garment's surface is its own tint constant, not a material asset, so
        // there is no authored blend to read here. A CUT-OUT garment needs the
        // material identity the skinned path still does not carry — named in the
        // wave ledger, not implied by a zero.
        blend: 0,
        cutoff: 0.5,
        palette: inf_render::identity_palette(),
        shadow: inf_render::SkinnedShadow::BindSphere,
        sections: Vec::new(),
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
        // OPAQUE, stated rather than defaulted (wave CHAR1a.2): a simulated
        // garment's surface is its own tint constant, not a material asset, so
        // there is no authored blend to read here. A CUT-OUT garment needs the
        // material identity the skinned path still does not carry — named in the
        // wave ledger, not implied by a zero.
        blend: 0,
        cutoff: 0.5,
        palette: inf_render::identity_palette(),
        shadow: inf_render::SkinnedShadow::BindSphere,
        sections: Vec::new(),
    });
}
#[cfg(test)]
mod render_settings_tests {
    use super::{apply_record, RenderSettings, RenderSettingsRecord};

    /// The default record maps to the byte-stable renderer default — this pins the
    /// mapping so a settings-less level renders exactly as today's defaults (and
    /// identical to the editor viewport's mirror `apply_record`).
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

/// Spot-light seam parity (R-P3). This is the **byte-identical mirror** of
/// `inf_viewport::host`'s `project_light_parity` test — same fixture, same
/// hardcoded expectations — so the toward-the-light / emission direction
/// convention can never drift between the player and the editor viewport.
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

/// Foliage projection mirror (E-P6, reshaped by P18.5): a `Foliage` component's
/// instances survive a serde round-trip and project — with no instance lost or
/// invented — into GPU-instanced scatter batches, one per primitive kind the
/// palette resolves. The shipped player relies on this so a level with painted
/// foliage draws in PIE == shipping (the editor viewport runs the identical
/// projection).
#[cfg(test)]
mod foliage_projection {
    use super::{push_foliage_scatter, PrimMesh, RenderScene};
    use glam::DVec3;
    use inf_ecs::components::{Foliage, FoliageInstance, FoliagePaletteEntry, Primitive};
    use inf_ecs::{Color, Vec3d};

    fn demo_foliage() -> Foliage {
        Foliage {
            palette: vec![
                FoliagePaletteEntry {
                    primitive: Primitive::Cone,
                    tint: Color::new(0.3, 0.6, 0.28, 1.0),
                },
                FoliagePaletteEntry {
                    primitive: Primitive::Sphere,
                    tint: Color::new(0.6, 0.5, 0.2, 1.0),
                },
            ],
            instances: (0..7)
                .map(|i| FoliageInstance {
                    position: Vec3d::new(i as f64, 0.0, (i % 3) as f64),
                    rotation: Vec3d::new(0.0, 20.0 * i as f64, 0.0),
                    scale: 1.0 + 0.05 * i as f64,
                    kind: (i % 2) as u32,
                })
                .collect(),
        }
    }

    #[test]
    fn foliage_round_trips_and_projects_to_matching_instance_count() {
        let fol = demo_foliage();
        // Round-trip the whole component (instances are serde-persisted).
        let bytes = serde_json::to_string(&fol).unwrap();
        let back: Foliage = serde_json::from_str(&bytes).unwrap();
        assert_eq!(
            back.instances, fol.instances,
            "instances survive round-trip"
        );

        // Project into a fresh scene. P18.5: the instances no longer expand into
        // `RenderScene::instances` one by one — they pack into scatter batches, so
        // the count that must be preserved is the SUM across batches.
        let mut scene = RenderScene::default();
        let anchor = DVec3::new(10.0, 0.0, -5.0);
        push_foliage_scatter(&mut scene, &back, anchor, 7);
        let total: usize = scene.scatter.iter().map(|b| b.data.len()).sum();
        assert_eq!(total, back.instances.len(), "no instance lost or invented");
        assert!(
            scene.instances.is_empty(),
            "scatter must not also expand into the per-instance mesh path"
        );

        // Two kinds are painted (alternating), so exactly two batches — emitted in
        // `PrimMesh::ALL` order (Sphere before Cone), NOT in first-use order.
        assert_eq!(scene.scatter.len(), 2);
        assert!(matches!(scene.scatter[0].data.mesh, PrimMesh::Sphere));
        assert!(matches!(scene.scatter[1].data.mesh, PrimMesh::Cone));

        // Kind 1 → the second palette slot (Sphere): it lands in that primitive's
        // batch, tinted by that slot. Kinds alternate over 7 instances, so kind 0
        // (Cone) has 4 and kind 1 (Sphere) has 3.
        assert_eq!(scene.scatter[0].data.len(), 3);
        assert_eq!(scene.scatter[1].data.len(), 4);
        assert_eq!(
            scene.scatter[0].data.instances[0].color,
            [0.6, 0.5, 0.2, 1.0]
        );
        assert_eq!(
            scene.scatter[1].data.instances[0].color,
            [0.3, 0.6, 0.28, 1.0]
        );

        for b in &scene.scatter {
            // The entity translation is the ANCHOR, not baked into the offsets…
            assert_eq!(b.anchor, anchor);
            // …and every batch of one entity carries the one pick id it was given.
            assert_eq!(b.id, 7);
            assert_eq!(b.draw_distance, 0.0);
        }
        // Instance 0 is kind 0 (Cone) at the local origin — offsets are the LOCAL
        // positions with no conversion.
        assert_eq!(scene.scatter[1].data.instances[0].offset, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn empty_foliage_projects_nothing() {
        let mut scene = RenderScene::default();
        push_foliage_scatter(&mut scene, &Foliage::default(), DVec3::ZERO, 1);
        assert!(scene.scatter.is_empty());
        assert!(scene.instances.is_empty());
    }

    /// **Content addressing** (P18.5): two foliage entities painted with the same
    /// stroke share one GPU upload, however far apart they sit. That only holds
    /// because the packed offsets are the entity-LOCAL positions and the world
    /// anchor is deliberately *not* part of `ScatterData::key` — pack against the
    /// world position instead and every duplicated prop becomes a second upload.
    #[test]
    fn identical_foliage_content_hashes_to_the_same_key() {
        let a_fol = demo_foliage();
        let b_fol = demo_foliage();
        let mut a = RenderScene::default();
        let mut b = RenderScene::default();
        push_foliage_scatter(&mut a, &a_fol, DVec3::new(10.0, 0.0, -5.0), 1);
        push_foliage_scatter(&mut b, &b_fol, DVec3::new(-4000.0, 12.0, 900.0), 2);

        assert_eq!(a.scatter.len(), b.scatter.len());
        assert!(!a.scatter.is_empty(), "the fixture must actually scatter");
        for (x, y) in a.scatter.iter().zip(&b.scatter) {
            assert_eq!(
                x.data.key(),
                y.data.key(),
                "the same stroke must content-key the same at a different anchor"
            );
        }
        assert_ne!(a.scatter[0].anchor, b.scatter[0].anchor);
    }
}

/// **The CPU half of the render host's voxel sync** — bind, place, mirror the
/// simulation's runtime carves, then page against the camera.
///
/// A free function rather than a method because it needs **no GPU**, and the
/// claim it carries — that the shipped player's render side sees what its
/// Blueprints carve — has to be checkable in CI, where no adapter exists.
/// `PlayerRenderHost::sync_voxels` is a one-line call to it, so the gate and the
/// player run the same code rather than two arrangements of it.
///
/// **P21.2 — three acts, not one.** `ensure` binds (parses the payload, indexes
/// its directory, pages nothing); `place` tells Ring 0 where the entity put the
/// volume, because residency is a world-space radius and a cave placed a
/// kilometre from its authoring anchor would otherwise page the chunks nobody is
/// standing in; `sync_camera` executes the policy. A bound volume with no
/// `sync_camera` draws nothing, which is why the three are one function on both
/// hosts and why the mirror gate requires all three calls on both sides.
///
/// **P21.4 — a fourth act, before the camera one.** `overlay_sim` mirrors the
/// chunks a runtime carve changed out of the *simulation's* map. Without it the
/// player gives a dug floor a collider, lets gameplay stand on it, and keeps
/// drawing the rock.
pub fn sync_voxel_store(
    voxels: &mut inf_voxel::VoxelVolumes,
    assets: &VoxelRegistry,
    sim: &RuntimeSim,
    camera_world: DVec3,
) {
    let world = sim.world();
    let w = world.world();
    let mut wants: Vec<(Uuid, Uuid, DVec3)> = w
        .iter_entities()
        .filter_map(|e| {
            let guid = e.get::<Guid>()?.0;
            let asset = e.get::<VoxelVolume>()?.asset?;
            let translation = e
                .get::<GlobalTransform>()
                .map(|g| g.translation())
                .unwrap_or(DVec3::ZERO);
            Some((guid, asset, translation))
        })
        .collect();
    wants.sort_by_key(|(g, _, _)| *g);
    // Kept for the overlay pass below: it needs the entity→asset binding to refuse
    // a cross-asset copy (an author can re-point `VoxelVolume.asset` in the
    // Details panel mid-session, and asset A's chunks must never land in a slot
    // bound to asset B).
    let bindings = wants.clone();
    // Release volumes whose entity is gone (or whose component lost its
    // reference) BEFORE loading, so a level switch never holds both.
    let live: std::collections::BTreeSet<u128> =
        wants.iter().map(|(g, _, _)| g.as_u128()).collect();
    voxels.retain_only(&live);
    for (guid, asset, translation) in wants {
        if !voxels.is_bound(guid.as_u128(), asset.as_u128()) {
            let Some(bytes) = assets.load(asset) else {
                continue;
            };
            if let Err(e) = voxels.ensure(guid.as_u128(), asset.as_u128(), &bytes) {
                tracing::warn!("inf-player: bad .inf_voxel {asset}: {e}");
                continue;
            }
        }
        voxels.place(guid.as_u128(), translation);
    }
    // THE CAMERA-DRIVEN RESIDENCY PASS (P21.2). The eye is the *render*
    // camera, which the simulation has no reference to: `terrain.height_at`
    // reads the sim's own volume map, seeded from sim state alone, so where the
    // player is looking can never change a gameplay answer. Same determinism
    // seam as `sync_render_terrain`, and it is the absence of a path rather
    // than a convention.
    let report = voxels.sync_camera(
        camera_world,
        &inf_voxel::VoxelWantsParams::default(),
        inf_voxel::VoxelStreamBudget::default(),
    );
    for (key, err) in &report.failed {
        tracing::warn!(
            "inf-player: voxel chunk ({}, {}, {}) failed to decode: {err}",
            key.x,
            key.y,
            key.z
        );
    }
    // **ACT FOUR — THE SIM'S CARVES, AFTER THE CAMERA PASS** (P21.4).
    //
    // A Blueprint that dug changed the *simulation's* volume map; without this the
    // shipped player gives the new floor a collider, lets gameplay stand on it,
    // and keeps drawing the rock that is no longer there. `sim -> render` is the
    // legal direction (the simulation is authoritative and the renderer projects
    // from it); the camera above never reaches back.
    //
    // **After**, so residency stays the camera's business alone and the carve is
    // re-applied on top of whatever it decided. A chunk the camera evicted and
    // later paged back in arrives as the asset's pre-carve bytes with a fresh
    // stamp, which is exactly what `overlay_sim` re-copies on — so nothing has to
    // be pinned and a session's resident set cannot grow without bound. `resync`
    // follows because `sync_camera`'s own re-mesh has already run by then.
    let volumes = sim.voxel_volumes();
    for (guid, asset, _) in &bindings {
        let Some(data) = volumes.get(guid) else {
            continue;
        };
        if voxels.overlay_sim(guid.as_u128(), asset.as_u128(), data) > 0 {
            voxels.resync(guid.as_u128());
        }
    }
}
