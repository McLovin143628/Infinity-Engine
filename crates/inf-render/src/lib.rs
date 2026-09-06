//! Renderer: wgpu device/surface, render graph, WGSL pipeline cache,
//! GPU-driven draws, ID picking.
//!
//! Ring 0 — no editor or Tauri concepts. Hosts (the editor's `inf-viewport`,
//! headless tests, the future thumbnailer) provide a [`GpuContext`], describe
//! a [`RenderScene`] + [`RenderView`] each frame, and get pixels.
//!
//! Coordinate contract (architecture rule 3): scene/instance positions are
//! f64 world space; the [`RenderView`]'s floating origin converts to f32
//! render-local at upload. Depth is reverse-infinite Z.

pub mod atmosphere;
pub mod bluenoise;
pub mod camera;
pub mod caps;
pub mod clouds;
pub mod csm;
pub mod debris;
pub mod debug_draw;
pub mod deform;
pub mod exposure;
pub mod gi;
pub mod gizmo;
pub mod golden;
pub mod gpu;
pub mod graph;
pub mod headless;
pub mod passes;
pub mod pick;
pub mod pipeline;
pub mod precip;
pub mod primitives;
/// **The ray-query shadow experiment** (P28.5) — the ROADMAP's last clause,
/// built to be measured and never load-bearing. Nothing in
/// [`EngineRenderer::render`] reaches it.
pub mod raytrace;
/// The non-blocking GPU→CPU readback ring at a pinned frame latency (P26.4).
/// Not a virtual-texturing detail: P27's shadow-page marking and P28's unified
/// streamer read through this same primitive.
pub mod readback;
pub mod renderer;
pub mod scene;
pub mod settings;
/// **The unified streamer's audit** (P28.3): the renderer's side of
/// `inf-stream` — the three page systems' residency, the arbiter's verdict
/// against the LIVE floors, the coupling's size, and the readback ledger.
pub mod stream;
pub mod surface;
/// **The per-pass GPU clock** (island wave I4) — one `QuerySet` written between
/// encoder commands, so no pass has to know it is being measured and a frame
/// with timing off records byte-identical commands.
pub mod timing;
/// The P28.1 visibility buffer's **packing contract** — the thirty-two bits that
/// name a triangle, the typed refusal a scene past them takes, and the Rust twin
/// of the resolve's barycentric solve.
pub mod visbuffer;
/// Virtual-shadow-map geometry (P27.1): the per-light projections, the
/// clipmap's centring rule, the depth-convention ruling and the one level rule
/// the marking pass mirrors.
pub mod vsm;
/// The P27.1 GPU mirror: one `Depth32Float` page atlas and one indirection
/// buffer, executing an `inf_vsm::VsmTransaction`.
pub mod vsm_atlas;
/// The P27.1 marking loop: the screen-driven page-marking compute pass, the
/// readback ring that reads it at a pinned latency, and the residency step
/// that turns its bits into allocations.
pub mod vsm_mark;
/// The P27.2 caster pass: the per-page GPU cull, the one render pass that owns
/// the whole atlas, and the viewport/scissor pair that pins each page to its
/// slot.
pub mod vsm_raster;
/// The P27.4 receiver: the address walk a lit fragment makes through the page
/// table, the clamped PCF kernel and its measured ruling, the clipmap level
/// blend, and the bias derived from a page's own texel density.
pub mod vsm_receiver;
pub mod vt;
/// The P26.3 registration door: `.inf_tex` v2 payloads become virtual
/// textures here, for both hosts, through one rule.
pub mod vt_library;
/// The P26.4 streaming loop: the analytic want floor, the GPU coverage feedback
/// that refines it, and the pop-in instruments.
pub mod vt_stream;
pub mod water;
pub mod wetness;

pub use atmosphere::{
    camera_radius_km, extinction, height_fog_optical_depth, height_fog_transmittance,
    transmittance_to_top, AtmosphereParams, AtmosphereQuality, HeightFog,
};
pub use camera::{
    ortho_reverse_z, OrthoParams, RenderView, DEPTH_CLEAR, DEPTH_COMPARE, DEPTH_FORMAT,
};
pub use caps::{
    choose_tier, detect_and_clamp, detect_tier, hair_detail_for, AdapterCaps, HairDetailSpec,
    RenderTier,
};
pub use clouds::{
    detail_texel, half_to_f32, shape_texel, wind_offset, CloudParams, CloudQuality, CloudVolumes,
    CPU_GPU_EXACT_CHANNEL_FRACTION, CPU_GPU_SHADOW_TOLERANCE, CPU_GPU_STEP_TOLERANCE,
    CPU_GPU_VALUE_ESCAPE_FRACTION, CPU_GPU_VALUE_TOLERANCE,
};
pub use debug_draw::{
    collider_outline_2d, collider_outline_3d, ColliderOutline2D, ColliderOutline3D, DebugDraw,
    DebugVertex,
};
// The P18.4 GI v2 surface: the cost tier, the amortization schedule, the
// voxelization audit, and the pure SH/terrain math the shaders mirror.
pub use exposure::{ExposureResources, ExposureState};
pub use gi::{
    bin_macro_cells, env_brdf_ab, intersects_volume, priority_order, sample_terrain_column,
    sh_dominant_direction, sh_radiance, sun_bucket, voxelization_tiles, GiAudit, GiBounds,
    GiQuality, ProbeSchedule, TerrainColumn, EMISSIVE_MAX, GI_DIM, MACRO_DIM, PROBE_DIMS,
};
pub use gizmo::{GizmoAxis, GizmoDelta, GizmoDrag, GizmoMode};
pub use gpu::{create_instance, GpuContext};
pub use headless::{HeadlessTarget, HEADLESS_FORMAT};
pub use passes::composite::BlitMode;
pub use passes::terrain::{
    assemble_patches, cells_at_lod, lod_for_distance, lod_thresholds, morph_band, morph_factor,
    patch_mesh_lod, plan_tile_cache, ring_source_lod, superseded, CachedTile, TerrainPatch,
    TileCacheKey, TileCachePlan, TERRAIN_BASE_CELLS, TERRAIN_LOD_COUNT,
};
// The P21.1 voxel-surface cache gate — the pure planner the volumetric-terrain
// pass drives its per-chunk uploads/evictions from, exported like `plan_tile_cache`
// so hosts and gates can reason about residency without a GPU.
pub use passes::voxel::{
    plan_chunk_cache, CachedChunk, ChunkCacheKey, ChunkCachePlan, VoxelReport,
};
// Wave NPC1b: the skinned pass's own batch planner, exported for the same reason
// `plan_tile_cache` is — an instrument that wants this frame's skinned draw count
// or its palette bytes calls the function the pass calls, not a re-derivation.
pub use passes::skinned::{
    plan_skinned_batches, SkinnedBatches, SkinnedRun, SKINNED_PALETTE_MATRICES,
};
pub use pick::Picker;
pub use precip::{
    particle_offset, precip_base, wrap_signed, PrecipParams, PrecipQuality, PRECIP_BOX_XZ_M,
    PRECIP_BOX_Y_M, RAIN_FALL_SPEED, SNOW_FALL_SPEED,
};
pub use primitives::{PrimGpu, PrimMesh, PrimRange};
pub use readback::{ReadbackRing, READBACK_LATENCY_FRAMES};
pub use renderer::{
    EngineRenderer, ViewMode, AO_FORMAT, HDR_FORMAT, LDR_FORMAT, MASK_FORMAT, SCENE_FORMAT,
    SCENE_SAMPLES,
};
pub use scene::{
    apply_seam, box_uv, deformed_skinned_mesh, identity_palette, skinned_sections,
    RenderFractureChunk,
    RenderFractureVertex, RenderTilemap, RenderVoxelChunk, RenderVoxelVertex, RenderVoxelVolume,
    ScatterBatch, ScatterClock, ScatterData, ScatterGeometry, ScatterInstance, ScatterInstanceRaw,
    ScatterMemo, ScatterMeshes, ScatterSource, SkinnedInstance, SkinnedMeshData, SkinnedSection, SkinnedShadow,
    SkinnedVertex, SkyParams, SpriteInstance, SpriteTextureUpload, SunParams, TerrainTileKey,
    TextureHandle, TilemapParams, VgeomAsset, VgeomInstance, VgeomMesh, VoxelChunkKey, CLOTH_TINT,
    DEFAULT_SUN_DIR, HAIR_TINT, ID_GIZMO_BASE, ID_NONE, INTERIOR_LOD_M, MAX_SCATTER_MESH_TRIANGLES,
    STRUCTURE_LOD_M,
};
pub use scene::{
    detail_scale_q8, glow_emissive, night_glow_step, pulse_emissive, pulse_tick,
    scatter_table_stamp, swept_colour, take_unchanged_terrain, take_unchanged_voxel,
    terrain_id_from_guid, uv_tiling_q8, Ambient2D, LightKind, MeshInstance, PrebatchedRun,
    RenderChunk, RenderLight, RenderLight2D, RenderScene, RenderTerrain, RenderTerrainLayer,
    RenderTerrainTile, SeamSample, VtTextureSet, DEFAULT_SEAM_BAND_M, NIGHT_GLOW_STEPS,
};
pub use settings::{
    adapt_exposure_ev, exposure_bin, exposure_bin_luminance, exposure_compensation_factor,
    exposure_log_average, exposure_target_ev, halton, halton_jitter, luminance,
    manual_exposure_multiplier, mip_chain_sizes, soft_knee_factor, ssao_hemisphere_kernel,
    BloomSettings, ExposureMode, ExposureSettings, FilmSettings, FlareSettings, GiSettings,
    PredictSettings, RaytraceSettings, RenderSettings, ScatterSettings, ShadowSettings,
    SsaoSettings, SsrQuality, SsrSettings, StreamSettings, VgeomSettings, VirtualTextureSettings,
    VsmSettings, VsmSettingsError, DEFAULT_PREDICT_HORIZON_TICKS, DEFAULT_STREAM_BUDGET_BYTES,
    EXPOSURE_BINS, EXPOSURE_KEY, EXPOSURE_LOG_MAX, EXPOSURE_LOG_MIN, ROADMAP_PREDICT_HORIZON_TICKS,
    STREAM_BUDGET_LOW_BYTES, STREAM_BUDGET_MEDIUM_BYTES, VGEOM_BUDGET_LOW_BYTES,
    VGEOM_BUDGET_MEDIUM_BYTES, VSM_BUDGET_LOW_BYTES, VSM_BUDGET_MEDIUM_BYTES,
    VSM_CLIPMAP_PAGES_MEDIUM, VSM_MARK_STRIDE_MEDIUM, VSM_MAX_MARK_STRIDE, VSM_MAX_PCF_RADIUS,
    VSM_PCF_RADIUS_MEDIUM,
};
pub use timing::{
    record, FrameTimer, FrameTimings, PassTime, RecordProfile, MAX_FRAME_MARKS, RECORD_PHASES,
    RECORD_PHASE_NAMES,
};
// P28.3: the streamer's audit, and the arbiter's own vocabulary re-exported for
// the same reason `inf_vt`'s is below — a host or a gate that reads a
// `StreamReport` must be able to name a `Consumer` without adding `inf-stream`
// to its manifest.
pub use inf_stream::{
    arbitrate, BudgetGrant, BudgetRequest, Consumer, Coupling, RingLedger, StreamError,
    LANE_FEEDBACK, LANE_FLOOR, LANE_PREDICT,
};
pub use stream::StreamReport;
// P28.5: the visibility path's parity criterion, hoisted out of the P28.1
// nucleus so the phase gate executes the SAME rule rather than a second spelling
// of it. The recorded criterion is one definition; `phase28_gate` reads it.
pub use visbuffer::{
    parity_ok, parity_verdict, ParityVerdict, PARITY_MAX_STEP, PARITY_TEXTURED_MAX_FRACTION,
    PARITY_TEXTURED_MAX_SOLID_CENTRES, PARITY_TEXTURED_MIN_BORDERING,
    PARITY_UNTEXTURED_MAX_FRACTION,
};
// P27.1 virtual shadow maps: the projections and the level rules (pure,
// unit-tested with no adapter), the mirror, and the marking loop's counters.
pub use vsm::{
    clipmap_centre, clipmap_containing_level, clipmap_layout, clipmap_level_ndc, clipmap_matrix,
    clipmap_page_world, cube_face_matrix, light_basis, mark_page_for, page_clip_planes,
    quantize_light_dir, spot_fov_y, spot_matrix, vsm_justified_level, vsm_light_trees,
    vsm_page_matrix, vsm_page_sees_sphere, vsm_projections, vsm_sun_quantum, ClipmapLayout,
    VsmMarkParams, VsmProjection, VsmTreeSet, CUBE_FACE_BASES, VSM_DEPTH_CLEAR, VSM_DEPTH_COMPARE,
    VSM_MAX_PROJECTIONS, VSM_PROJ_ORTHO, VSM_PROJ_PERSPECTIVE, VSM_SUN_REFERENCE_HEIGHT_M,
};
pub use vsm_atlas::{VsmApplyReport, VsmPools, VSM_PAGE_FORMAT};
pub use vsm_mark::{VsmMarker, VsmStreamStats, VsmSystem, VSM_PROJECTION_CAP};
pub use vsm_raster::{
    skinned_caster_groups, PageGeometry, VsmCasterRaw, VsmRaster, VsmRasterStats,
    SKINNED_POSE_MARGIN, VSM_ARG_WORDS, VSM_LEVEL_BUCKETS, VSM_MAX_CASTERS, VSM_MAX_GROUPS,
    VSM_MAX_RASTER_PAGES, VSM_PAGE_DRAW_STRIDE, VSM_RIGID_GROUPS, VSM_TERRAIN_CASTER_CELLS,
};
// P27.4 the receiver: the pure half (the address walk, the clamped kernel, the
// level blend, the two derived bias terms) plus the resources the shared
// environment bind group names.
pub use vsm_receiver::{
    clipmap_resolution_reads, is_clipmap, pcf_crossing_fraction, pcf_resolution_cost,
    receiver_slots, sun_slot, vsm_atlas_header, vsm_bias_ndc, vsm_blend_weight, vsm_block_header,
    vsm_cube_face, vsm_level_factor, vsm_level_ndc, vsm_ndc_per_metre, vsm_page_of, vsm_pcf_taps,
    vsm_receiver_level, vsm_receiver_site, vsm_shadow_factor, vsm_slope_bias_texels, vsm_slope_tan,
    vsm_table_entry, vsm_to_light, VsmEmptyPool, VsmReceiverParams, VsmReceiverResources,
    VsmReceiverSite, VsmTableEntry, VSM_DEPTH_ULP_BIAS, VSM_MAX_SLOPE, VSM_NORMAL_BIAS_TEXELS,
    VSM_NO_DATA, VSM_PCF_RADIUS, VSM_PCF_TAPS, VSM_SLOPE_BIAS_TEXELS,
};
// The shadow page space's own vocabulary, re-exported on `inf_vt`'s precedent so
// a host or a gate that reads a residency does not have to add the GPU-free
// crate to its own manifest to spell a page address.
pub use inf_vsm::{
    VsmAtlasConfig, VsmLightDesc, VsmLightHandle, VsmMarkLayout, VsmPage, VsmResidency,
    VsmTreeKind, VsmWant, DEFAULT_VSM_BUDGET_BYTES, VSM_PAGE_SIZE,
};
// The P26.3 registration door + P26.4's one registration ORDER: both projectors
// build a level's virtual textures through exactly these, so "PIE == shipping"
// for texture residency is a property of the code rather than of two hosts
// agreeing by inspection.
pub use vt_library::{
    build_vt_level, registration_order, vt_set_for, VtLevelReport, VtMaterialMaps, VtRefusal,
    VtTextures, VtTileSource, VT_FLOOR_LEVELS,
};
// The P26.4 streaming loop: the floor's rules (pure, unit-tested with no
// adapter), the GPU feedback pass, and the pop-in counters a gate asserts on.
pub use vt_stream::{
    analytic_floor, camera_wants, feedback_requests, justified_mip, ndc_margin, on_screen,
    predicted_view, projection_scale, scene_coverage, screen_diameter_px, speculative_wants,
    VtCoverage, VtFeedback, VtPopIn, VT_FEEDBACK_MAX_TILES, VT_FLOOR_MAX_TILES,
};
// The pool's own vocabulary, re-exported so a host that builds a VT level does
// not have to depend on `inf-vt` itself: `VtTextures::new` takes a
// `VtPoolConfig`, so a caller that cannot name one cannot call it, and adding
// the crate to two hosts' manifests to spell four constants would put the
// GPU-free half in dependency graphs that have no other use for it.
pub use inf_vt::{
    PageFormat, VtPoolConfig, VtWant, DEFAULT_MAX_TEXTURE_DIM, DEFAULT_VT_BUDGET_BYTES,
    STORED_TILE_SIZE, VT_PRIORITY_FEEDBACK, VT_PRIORITY_FLOOR, VT_PRIORITY_PREDICT,
};
pub use water::{
    camera_underwater, RenderWater, RiverFrame, RiverPath, RiverProfile, Underwater, WaterFrame,
    WaterKindGpu, WaterQuality, WaterSettings, WaterSurface, Wave, WaveField, WaveSpec, MAX_WAVES,
    OCEAN_EXTENT_M, OCEAN_SNAP_M, SHAFT_DECAY, SHAFT_GLOW_POWER, SHAFT_INTENSITY, SHAFT_REACH,
    SHAFT_TINT_DEPTH_M, UNDERWATER_FAR_M, UNDERWATER_RAMP_M,
};
// P22.4 small-debris instancing + the per-tier debris budget: the deterministic
// sub-chunk rubble both hosts lay through the P18.5 scatter path, and the one
// place `RenderTier` is mapped onto a budget (physics stays tier-blind).
pub use debris::{
    debris_batch, debris_budget_for, debris_budget_for_session, debris_instances, DebrisBudgetSpec,
    DebrisCache, DebrisSite, DEBRIS_BUDGET_HIGH, DEBRIS_MAX_SCALE, DEBRIS_MIN_SCALE,
    DEBRIS_RUBBLE_PER_CHUNK,
};
// P22.1 surface deformation: the projected field, the camera-following window's
// packing, and the engine constants argued in `deform.rs` rather than authored.
pub use deform::{
    deform_depth_reference, pack_deform_window, window_origin_texels, DeformResources,
    DeformUniform, RenderDeform, RenderDeformCell, DEFORM_BEND_GAIN, DEFORM_MAX_DEPTH_M,
    DEFORM_TEXEL_M, DEFORM_WINDOW_M, DEFORM_WINDOW_TEXELS, WIND_SWAY, WIND_WAVELENGTH_M,
};
// P20.3 shoreline wetness: the packing the renderer feeds the lit passes, and the
// engine constants whose values are argued in `wetness.rs` rather than authored.
pub use wetness::{
    pack_wetness, WetnessResources, WetnessUniform, MAX_WET_BODIES, MAX_WET_SEGMENTS,
    WET_ALBEDO_SCALE, WET_BAND_M, WET_ROUGHNESS_SCALE, WET_SHORE_MARGIN_M,
};
// The P18.5 scatter instruments: the GPU instance-cull counters (off by default,
// free when off) and the pure band rule both the compute pass and the CPU
// fallback derive their distances from.
pub use passes::scatter::{
    effective_bands, pack_fallback, shadow_caster_settings, PackPurpose, ScatterAudit, ScatterPack,
    MAX_CPU_SCATTER_INSTANCES, SHADOW_CASTER_MARGIN,
};
// The GPU meshlet cull readback (P13.1b) — the CPU-vs-GPU parity gate + the
// player's vgeom-activation check drive it. `VgeomAudit` + `is_camera_cut` are
// the P18.1 two-pass occlusion instruments.
// `cull_visible_streamed` is the same call with the residency it culled under, so
// the parity gate can drive a PUNCHED-OUT resident set (P18.2) rather than only
// the fully-paged case.
pub use passes::vgeom::{
    cull_visible, cull_visible_source, cull_visible_streamed, is_camera_cut, CullReadback,
    VgeomAudit, VgeomStreamReport,
};
// The classic-LOD fallback selection (P13.4) — the CI-provable probe of what the
// classic path draws when vgeom is off (the meshlet path's complement).
pub use passes::classic_vgeom::{classic_lod_selection, ClassicSelection};
// 2D batcher API surfaced through the renderer for hosts.
pub use inf_render_2d::{
    aabb_visible, atlas_uv, batch_scene, batch_sprites, billboard_basis, builtin_font_rgba8,
    chunk_world_aabb, corner_offset_billboard, expand_chunk, expand_nine_slice, expand_text,
    handle_from_guid, BatchedSprites, HAlign, NineSliceParams, SpriteBatch, TextParams,
    BILLBOARD_CYLINDRICAL, BILLBOARD_NONE, BILLBOARD_SPHERICAL, BUILTIN_FONT_COLS,
    BUILTIN_FONT_FIRST_CP, BUILTIN_FONT_ROWS, BUILTIN_FONT_TEXTURE, TILE_CHUNK_DIM, WHITE_TEXTURE,
};
pub use surface::{SurfaceChain, RECONFIGURE_DEBOUNCE};
