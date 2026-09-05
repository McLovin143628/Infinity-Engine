//! Dynamic global-illumination pass (P13.3b, **rebuilt in P18.4**): the two-stage
//! compute front end for the real-time single-bounce diffuse+specular GI. Each
//! frame (when enabled) it
//!
//! 1. **voxelizes** the scene into a camera-centred [`GiQuality::voxel_dim`]³
//!    occupancy+albedo+emissive storage volume (`gi_voxelize.wgsl`), then
//! 2. **marches** the [`GiQuality::probe_dims`] probe grid through it, projecting
//!    single-bounce radiance to L1 SH per probe (`gi_probes.wgsl`).
//!
//! The lit passes then sample the SH buffer through [`crate::passes::EnvBinding`]
//! (their ambient term becomes the probe-interpolated irradiance, and — P18.4 —
//! their ambient *specular* becomes radiance along the reflection vector). The SH
//! buffer + voxel buffer + shared [`GiData`](GiDataGpu) uniform live in the
//! renderer-owned [`GiResources`]; this node owns the two compute pipelines and the
//! per-frame primitive/bin/terrain buffers.
//!
//! ## The boundary with virtual shadow maps (P18 law, restated at P27.1)
//!
//! **GI never reads a virtual shadow page, its residency, or its atlas — and it
//! must not start.** The law is P18's, written when meshlet residency arrived:
//! *camera-driven residency never feeds lighting*. A VSM page exists because
//! **a visible pixel's depth asked for it** (`inf_render::vsm_mark`), so the set
//! of resident pages is a function of where the camera is looking; a probe that
//! consulted them would make the irradiance at a fixed point in the world depend
//! on the viewer, and two hosts looking at one scene from two angles would light
//! it differently. That is the same defect the meshlet path avoids by
//! voxelizing the **always-resident root page** rather than whatever is
//! streamed in.
//!
//! So the two systems meet nowhere: this pass's occlusion is its own voxel
//! volume, and it stays that way when P27.4 gives the pages a receiver — the
//! lit passes' *direct* term is what a shadow page darkens, and the *ambient*
//! term is this pass's. `the_gi_pass_never_reads_a_shadow_page` is the source
//! gate that keeps it true rather than remembered.
//!
//! ## What P18.4 changed
//!
//! * **Coverage.** v1 voxelized rigid `MeshInstance` boxes and nothing else —
//!   terrain, skinned characters and vgeom meshes were invisible to GI, so a
//!   character cast no bounce and a landscape neither occluded nor coloured one.
//!   Now:
//!   * rigid instances → oriented boxes (unchanged),
//!   * skinned instances → **per-joint boxes**, the bind-space AABB of each joint's
//!     dominant vertices carried by the live skinning palette (cached per mesh by
//!     `Arc` pointer identity, so a character costs one AABB pass ever and a
//!     palette transform per frame),
//!   * vgeom instances → the **per-meshlet spheres of the always-resident root
//!     page** (cached per asset id),
//!   * terrain → one height + splat-blended albedo per voxel **column**.
//! * **The cap is gone.** `MAX_GI_INSTANCES = 256` silently dropped everything past
//!   the 257th instance in *scene order* — so which geometry lit the room depended
//!   on the outliner. Primitives are now ordered nearest-volume-first, clipped to a
//!   budget whose overflow is reported ([`GiAudit`]), and binned into macro cells so
//!   the per-voxel gather stays short.
//! * **Sky.** The ray-miss term reads the P17.2 sky-view LUT (bound into this pass)
//!   instead of two authored constants.
//! * **Amortization.** Probe updates can be spread across frames on a deterministic
//!   cursor ([`ProbeSchedule`]); full update is the default.
//! * **Resizable.** [`GiResources`] is sized by [`GiQuality`] and carries a
//!   `generation` that [`crate::passes::ResourceKey`] now includes.
//!
//! ## Scope (documented)
//!
//! * Revoxelizes **every frame** at the default settings (correct; the volume
//!   follows the camera, so it always covers the near field). Probe amortization is
//!   the opt-in half of the temporal story; a *voxel* cache keyed on the volume's
//!   snapped origin is the remaining follow-up.
//! * Every primitive is a box or a sphere: a `PrimMesh::Sphere` instance still
//!   voxelizes as its bounding box, exactly as in v1.
//! * **Off path:** disabled → the node writes the shared uniform with `enabled = 0`
//!   and dispatches nothing, so receivers keep the byte-stable hemispheric ambient.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use glam::{Mat4, Quat, Vec3};

use crate::gi::{
    bin_macro_cells, intersects_volume, priority_order, sample_terrain_column_in, sun_bucket,
    voxelization_tiles, GiAudit, GiBounds, GiQuality, ProbeSchedule, GI_DIM, PROBE_DIMS,
};
use crate::gpu::GpuContext;
use crate::graph::RenderNode;
use crate::renderer::FrameData;
use crate::scene::LightKind;

/// Primitive kind tag in `GiInstanceGpu::albedo.w`.
const KIND_BOX: f32 = 0.0;
const KIND_SPHERE: f32 = 1.0;

/// Minimum weight for a skinned vertex to count toward a joint's box. Below this a
/// vertex is a blend tail — including it would inflate every joint's AABB toward
/// its neighbours until a character voxelized as one slab.
const JOINT_DOMINANT_WEIGHT: f32 = 0.35;

/// **The most instances one scatter batch may be walked for** (wave VEN1a).
///
/// Four times the largest tier budget, so a batch can never be the reason the
/// per-frame cost of *staging* stops being a function of the per-frame cost of
/// *voxelizing*. Past it the batch is walked with a stride — every `n`th
/// instance, `n` chosen so the walk lands here — which is deterministic, is a
/// pure function of the payload's own length, and is reported as
/// [`GiAudit::scatter_decimated`] rather than being silent.
///
/// **It bites on the PAYLOAD's length, not on how much of it is in the volume**
/// (VEN1a audit — the doc said the second and the code has always said the
/// first). `stride` is `data.len().div_ceil(…)`, chosen before a single
/// instance is examined, because examining them is the cost it exists to bound.
/// The walk is uniform over the authored order, so a batch that overruns is
/// thinned by the same factor everywhere — including inside the volume, where
/// only a fraction of it may stand.
///
/// What makes that acceptable rather than merely cheap: a batch big enough to
/// trip it and *mostly inside* the volume is one the grid cannot represent
/// anyway — 16 384 primitives against a 64³ grid of 262 144 voxels and a budget
/// of 4 096, so the grid is saturated several times over before the stride
/// removes anything a probe could have seen. A batch big enough to trip it and
/// mostly *outside* loses fidelity it would have had; no committed level holds
/// one, and [`GiAudit::scatter_decimated`] is what says so rather than a hope.
const SCATTER_WALK_CEILING: usize = 4 * 4096;

/// One cached mesh's joint boxes, **and the `Arc` whose address is its key**.
///
/// The pair is the point, not an implementation detail: see
/// [`GiNode::joint_boxes`] for why an entry that does not hold its allocation is
/// a stale-lighting bug rather than a leak.
type JointBoxEntry = (Arc<crate::scene::SkinnedMeshData>, Arc<Vec<(Vec3, Vec3)>>);

/// A shared handle to the last frame's [`GiAudit`], published by the node and read
/// through [`crate::EngineRenderer::gi_audit`]. The same shape as the P18.2
/// streaming report: CPU counters the pass already computes, so it is free and
/// always on.
pub type SharedGiAudit = Arc<Mutex<GiAudit>>;

/// The shared GI uniform (`std140`), written by [`GiNode`] and read by the two
/// compute shaders **and** every lit pass. Mirrors `struct GiData` across
/// `gi_voxelize.wgsl` / `gi_probes.wgsl` / `env_lighting.wgsl`.
///
/// The P18.4 fields are **appended**, so every pre-existing offset is unchanged.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GiDataGpu {
    /// xyz = render-local volume min corner, w = voxel size (m).
    pub vol_min: [f32; 4],
    /// xyz = render-local probe grid min corner, w = extent (m).
    pub probe_min: [f32; 4],
    /// x = voxel dim, yzw = probe dims.
    pub dims: [f32; 4],
    /// x = enabled, y = intensity, z = rays, w = **macro-cell dim** (was the
    /// instance count in v1, which the CSR bins made redundant).
    pub params: [f32; 4],
    /// xyz = unit direction toward the sun.
    pub sun_dir: [f32; 4],
    /// rgb = sun radiance (colour × intensity).
    pub sun_color: [f32; 4],
    /// rgb = sky zenith radiance (ray miss, upward) — the gradient fallback.
    pub sky_zenith: [f32; 4],
    /// rgb = sky horizon radiance (ray miss, sideways) — the gradient fallback.
    pub sky_horizon: [f32; 4],
    /// P18.4: x = SH specular on, y = SSR on, z = SSR distance (m), w = SSR
    /// relative thickness.
    pub params2: [f32; 4],
    /// P18.4: x = probe update start, y = probe update count, z = probe total,
    /// w = sky source (0 = authored gradient, 1 = the P17.2 sky-view LUT).
    pub sched: [f32; 4],
    /// **Wave VIS1a, the SSR block**: x = march steps, y = roughness cutoff,
    /// z = intensity, w = whether the previous frame's resolved colour is usable
    /// (`0` on the first frame after a resize, when the history texture is a
    /// freshly zero-initialized allocation and reflecting it would be reflecting
    /// black).
    pub ssr: [f32; 4],
    /// **Wave VIS1a**: the PREVIOUS frame's jittered view-projection, so an SSR
    /// hit found in this frame's depth is sampled from the colour buffer at the
    /// place it occupied when that colour was written.
    ///
    /// It rides here rather than in `ViewUniforms` because this uniform is
    /// already bound by every lit pass through `EnvBinding` and by nothing else,
    /// so a matrix added here reaches exactly the shaders that need it and
    /// changes no other pass's uniform layout.
    pub prev_view_proj: [f32; 16],

    // ── the sky's own irradiance (wave FIX3) ──────────────────────────────
    //
    // Four L1 spherical-harmonic coefficients (`rgb` per lane, `w` reserved) of
    // the sky's radiance over the whole sphere, projected once per frame by
    // `crate::atmosphere::sky_irradiance_sh` from the SAME medium the sky pass
    // draws — or, for a scene with no time-of-day authority, from its authored
    // `SkyParams` gradient.
    //
    // **Why it rides in the GI uniform.** Two reasons, and the second is the
    // load-bearing one:
    //
    //  1. This uniform is bound by every lit pass (`EnvBinding`) and by nothing
    //     else, so an ambient term added here reaches every lit shader for zero
    //     new bindings — and the env group has no room for a fifth fragment
    //     storage buffer (`passes::visbuffer` asserts the four it already
    //     spends against `Limits::default()`'s eight).
    //  2. `env_lighting.wgsl` is composed BEFORE the atmosphere library, and
    //     WGSL has no forward declarations. `sky_irradiance` has to sit beside
    //     `gi_irradiance` — the term it is added to, and whose folded cosine
    //     lobe it shares — so its data has to be reachable there. `atmos` is
    //     not; `gi` is.
    //
    // It also puts it beside the two lanes it supersedes: `sky_zenith` /
    // `sky_horizon` are the probe march's own two-colour approximation of this
    // same sky, and the day the march stops needing them they can go.
    pub sky_sh: [[f32; 4]; 4],
}

/// Pad an SH triple set into the uniform's `vec4` lanes.
fn sky_sh_lanes(c: [[f32; 3]; 4]) -> [[f32; 4]; 4] {
    [
        [c[0][0], c[0][1], c[0][2], 0.0],
        [c[1][0], c[1][1], c[1][2], 0.0],
        [c[2][0], c[2][1], c[2][2], 0.0],
        [c[3][0], c[3][1], c[3][2], 0.0],
    ]
}

impl GiDataGpu {
    fn disabled() -> Self {
        Self {
            vol_min: [0.0, 0.0, 0.0, 1.0],
            probe_min: [0.0, 0.0, 0.0, 1.0],
            dims: [
                GI_DIM as f32,
                PROBE_DIMS[0] as f32,
                PROBE_DIMS[1] as f32,
                PROBE_DIMS[2] as f32,
            ],
            params: [0.0, 1.0, 48.0, 0.0],
            sun_dir: [0.0, 1.0, 0.0, 0.0],
            sun_color: [0.0; 4],
            sky_zenith: [0.0; 4],
            sky_horizon: [0.0; 4],
            params2: [0.0; 4],
            sched: [0.0; 4],
            ssr: [0.0; 4],
            prev_view_proj: [0.0; 16],
            // GI off ⇒ the ambient term keeps the hemispheric constant, so the
            // consumer never reads these. See `sky_irradiance` in
            // `shaders/env_lighting.wgsl` for the gate and the ledger for why
            // this wave did not widen it.
            sky_sh: [[0.0; 4]; 4],
        }
    }

    /// The SSR half of the uniform, packed from the settings and the frame.
    ///
    /// A free function of two arguments rather than an inline literal, because it
    /// is written from **two** places — the GI-on path below and the wave VIS1a
    /// GI-off path that exists so SSR no longer needs probes — and two copies of
    /// a packing rule is two chances for the off-GI frame to march differently
    /// from the on-GI one.
    fn with_ssr(mut self, frame: &FrameData) -> Self {
        let s = &frame.settings.ssr;
        self.params2[1] = if s.enabled { 1.0 } else { 0.0 };
        self.params2[2] = s.max_distance.max(0.0);
        self.params2[3] = s.thickness.clamp(1e-4, 1.0);
        self.ssr = [
            s.quality.steps() as f32,
            s.roughness_cutoff.clamp(0.0, 1.0),
            s.intensity.clamp(0.0, 1.0),
            if frame.scene_history_valid { 1.0 } else { 0.0 },
        ];
        self.prev_view_proj = frame.taa_prev_view_proj;
        self
    }
}

/// One voxelized primitive (`struct GiInstance` in `gi_voxelize.wgsl`).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GiInstanceGpu {
    inv_model: [f32; 16],
    /// rgb = albedo, w = kind ([`KIND_BOX`] / [`KIND_SPHERE`]).
    albedo: [f32; 4],
    /// rgb = self-emitted radiance, w unused.
    emissive: [f32; 4],
}

/// One voxel column's terrain occupancy (`struct GiTerrainColumn` in the shader).
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct GiTerrainColumnGpu {
    /// Render-local Y the column is solid up to.
    height: f32,
    /// Packed RGB8 splat-blended albedo.
    albedo: u32,
    /// 0 = no resident tile covers this column.
    present: u32,
    pad: u32,
}

/// Renderer-owned GI GPU resources, shared with the lit passes via [`FrameData`].
///
/// **Resizable since P18.4** (unlike the shadow resources): a change of
/// [`GiQuality`] recreates the voxel + SH buffers and bumps
/// [`generation`](Self::generation), which every bind-group cache that embeds them
/// must key on — see [`crate::passes::ResourceKey`].
pub struct GiResources {
    /// Occupancy+albedo+emissive volume (`dim³ × 2` packed `u32`s).
    pub voxels: wgpu::Buffer,
    /// L1 SH per probe (`probe_count × 4 vec4<f32>`).
    pub sh: wgpu::Buffer,
    /// Shared `GiData` uniform (written by the node, read by everyone).
    pub uniform: wgpu::Buffer,
    /// Bumped on every recreation. Bind-group caches MUST key on this.
    pub generation: u64,
    /// The quality these buffers were sized for.
    pub quality: GiQuality,
}

impl GiResources {
    pub fn new(gpu: &GpuContext, quality: GiQuality, generation: u64) -> Self {
        let dim = quality.voxel_dim() as u64;
        let voxels = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gi-voxels"),
            // Two words per voxel: albedo+occupancy, then emissive.
            size: dim * dim * dim * 8,
            // COPY_SRC so the residency-independence gate can byte-compare the
            // volume itself rather than inferring it from pixels. Costs nothing at
            // runtime — a usage flag only widens what the buffer may be bound as.
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let probes = crate::gi::probe_count_of(quality.probe_dims()) as u64;
        let sh = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gi-sh"),
            // 4 vec4<f32> per probe.
            size: probes * 4 * 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let uniform = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gi-data"),
            size: std::mem::size_of::<GiDataGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue
            .write_buffer(&uniform, 0, bytemuck::bytes_of(&GiDataGpu::disabled()));
        Self {
            voxels,
            sh,
            uniform,
            generation,
            quality,
        }
    }

    /// Blocking readback of the occupancy+albedo+emissive volume as raw bytes.
    ///
    /// Exists for the gates that need to compare the GI *inputs* rather than the
    /// pixels they eventually produce — chiefly
    /// `gi_terrain_voxelization_is_independent_of_residency`, where two residency
    /// states legitimately draw different terrain detail and only the volume can
    /// say whether GI saw the same world. A test/tools path, never the hot path.
    pub fn read_voxels(&self, gpu: &GpuContext) -> Vec<u8> {
        read_buffer(gpu, &self.voxels, "gi-voxels-readback")
    }

    /// Blocking readback of the L1 SH probe buffer. See [`Self::read_voxels`].
    pub fn read_sh(&self, gpu: &GpuContext) -> Vec<u8> {
        read_buffer(gpu, &self.sh, "gi-sh-readback")
    }
}

/// Copy a storage buffer into a staging buffer and map it. Blocking.
fn read_buffer(gpu: &GpuContext, src: &wgpu::Buffer, label: &str) -> Vec<u8> {
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: src.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
    encoder.copy_buffer_to_buffer(src, 0, &staging, 0, src.size());
    gpu.queue.submit([encoder.finish()]);

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    let _ = rx.recv();
    let data = slice.get_mapped_range().expect("map gi readback buffer");
    let out = data.to_vec();
    drop(data);
    staging.unmap();
    out
}

/// The inputs that, when any of them changes, restart the amortized probe sweep.
///
/// Camera *motion* is deliberately absent: the volume follows the camera every
/// frame, so a reset on camera movement would mean never amortizing at all. What is
/// here is everything that makes the previously-integrated probes **meaningless**
/// — the probe geometry, the GI configuration, the volume generation, and the sun
/// (which is how a time-of-day change propagates into the bounce).
///
/// # `scene.version` is NOT here, and that is the P18.4 amortization fix
///
/// It used to be, and it made amortized GI a **no-op in the shipped player**.
/// `inf_player::render::project_scene` re-projects unconditionally every frame and
/// ends with `RenderScene::mark_dirty()`, so `scene.version` increments every
/// frame; the key therefore differed every frame, `ProbeSchedule::reset()` ran
/// every frame, and the cursor never left its first slice. That is *precisely* the
/// failure [`sun_bucket`](crate::gi::sun_bucket) exists to prevent, quoted from its
/// own doc: "amortization paying full price for a fraction of the freshness,
/// exactly where it is supposed to help". The editor viewport hid it, because
/// `sync_from_doc` is version-gated there and a static document holds its version
/// still — so this was a PIE-vs-shipping divergence in everything but name.
///
/// Quantizing it the way the sun is quantized is not available: a version counter
/// has no metric to bucket on. The right reading is that a content change does not
/// invalidate the *integration* — it makes some probes stale, and the sweep already
/// bounds staleness by construction. The cursor wraps, so every probe is revisited
/// every `ceil(total / budget)` frames (8 at the documented 256-probe budget), and
/// a probe that has not been revisited lags by at most one sweep — the identical
/// guarantee, in the identical words, that the sun bucket's doc already accepts.
/// Resetting on top of that bought nothing an unreset sweep does not already give,
/// and cost the whole feature wherever content moves every frame — which is every
/// frame of any game that is actually running.
///
/// What a *reset* is still for is the case where the previous integration cannot
/// be aged into the new one at all: a different probe count, a different volume, a
/// different ray budget, a different sky source. Those are the fields below.
#[derive(Clone, Copy, PartialEq)]
struct GiSweepKey {
    generation: u64,
    probe_total: u32,
    settings: [u32; 5],
    /// The **bucketed** sun ([`sun_bucket`]) — direction and radiance. Not raw
    /// bits: see the construction site.
    sun: [i32; 6],
    sky_from_atmosphere: bool,
}

pub struct GiNode {
    voxelize_pipeline: wgpu::ComputePipeline,
    voxelize_bgl: wgpu::BindGroupLayout,
    probe_pipeline: wgpu::ComputePipeline,
    probe_bgl: wgpu::BindGroupLayout,
    instances: wgpu::Buffer,
    instance_cap: usize,
    cell_offsets: wgpu::Buffer,
    cell_offsets_cap: usize,
    cell_items: wgpu::Buffer,
    cell_items_cap: usize,
    terrain: wgpu::Buffer,
    terrain_cap: usize,
    /// Whether the disabled-GI uniform is already published to `frame.gi.uniform`.
    /// Gates the constant re-write while GI stays off; the enabled path clears it so
    /// a later disable re-publishes.
    published_disabled: bool,
    /// Per-skinned-mesh joint boxes in **bind space**, keyed by the shared
    /// `SkinnedMeshData`'s pointer identity (the same key the skinned pass caches
    /// its GPU buffers on since P18.3).
    ///
    /// **The entry holds the `Arc`, and that is what makes the key sound.** An
    /// address is only an identity while the allocation it names is alive: with
    /// the mesh dropped, the allocator may hand the same address to a *different*
    /// `SkinnedMeshData`, which would then hit this entry and voxelize the
    /// previous mesh's joint boxes into the bounce lighting — no validation error,
    /// no black frame, just wrong light. Verbatim `passes::skinned`'s
    /// `mesh_cache` and `vsm_raster`'s `SkinnedCasterGeom::mesh`, which is the
    /// half the P27.3 audit put back there for the same reason.
    joint_boxes: HashMap<usize, JointBoxEntry>,
    /// Per-vgeom-asset root-page meshlet spheres (local space), keyed by asset id.
    ///
    /// Retained to the frame's asset list: in the editor an asset id is
    /// **content-addressed**, so re-importing a mesh mints a new key and the
    /// superseded entry would accumulate for the life of the session — verbatim
    /// the reasoning P18.3 used to give `ClassicVgeomNode` its `retain_live`.
    meshlet_spheres: HashMap<u128, Arc<Vec<(Vec3, f32)>>>,
    /// The deterministic amortization cursor + the key that resets it.
    schedule: ProbeSchedule,
    sweep_key: Option<GiSweepKey>,
    audit: SharedGiAudit,
}

impl GiNode {
    pub fn new(gpu: &GpuContext, audit: SharedGiAudit) -> Self {
        let vox_shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("gi-voxelize"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/gi_voxelize.wgsl").into(),
                ),
            });
        let probe_shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("gi-probes"),
                source: wgpu::ShaderSource::Wgsl(super::shader_source("gi_probes").into()),
            });

        let uniform_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let storage_entry = |binding, ro| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: ro },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let texture_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };

        let voxelize_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gi-voxelize"),
                entries: &[
                    uniform_entry(0),
                    storage_entry(1, true),  // instances
                    storage_entry(2, false), // voxels (write)
                    storage_entry(3, true),  // macro-cell offsets (CSR)
                    storage_entry(4, true),  // macro-cell items (CSR)
                    storage_entry(5, true),  // terrain columns
                ],
            });
        let probe_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gi-probes"),
                entries: &[
                    uniform_entry(0),
                    storage_entry(1, true),  // voxels (read)
                    storage_entry(2, false), // sh (write)
                    // ── P17.2 atmosphere: the probe miss term's sky radiance ──
                    uniform_entry(3),
                    texture_entry(4), // transmittance LUT
                    texture_entry(5), // sky-view LUT
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let mk = |label, bgl: &wgpu::BindGroupLayout, module, entry: &str| {
            let layout = gpu
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(label),
                    bind_group_layouts: &[Some(bgl)],
                    immediate_size: 0,
                });
            gpu.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(label),
                    layout: Some(&layout),
                    module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                })
        };
        let voxelize_pipeline = mk("gi-voxelize", &voxelize_bgl, &vox_shader, "cs_voxelize");
        let probe_pipeline = mk("gi-probes", &probe_bgl, &probe_shader, "cs_probes");

        let storage_buf = |label, size: u64| {
            gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };

        Self {
            voxelize_pipeline,
            voxelize_bgl,
            probe_pipeline,
            probe_bgl,
            instances: storage_buf(
                "gi-instances",
                (std::mem::size_of::<GiInstanceGpu>() * 16) as u64,
            ),
            instance_cap: 16,
            cell_offsets: storage_buf("gi-cell-offsets", 16 * 4),
            cell_offsets_cap: 16,
            cell_items: storage_buf("gi-cell-items", 16 * 4),
            cell_items_cap: 16,
            terrain: storage_buf(
                "gi-terrain-columns",
                (std::mem::size_of::<GiTerrainColumnGpu>() * 16) as u64,
            ),
            terrain_cap: 16,
            published_disabled: false,
            joint_boxes: HashMap::new(),
            meshlet_spheres: HashMap::new(),
            schedule: ProbeSchedule::new(),
            sweep_key: None,
            audit,
        }
    }

    /// Grow one of the per-frame storage buffers to hold at least `needed`
    /// elements of `stride` bytes (next power of two, never shrinking).
    fn ensure(
        gpu: &GpuContext,
        buf: &mut wgpu::Buffer,
        cap: &mut usize,
        needed: usize,
        stride: usize,
        label: &'static str,
    ) {
        if needed <= *cap {
            return;
        }
        let new_cap = needed.next_power_of_two().max(16);
        *buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (new_cap * stride) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        *cap = new_cap;
    }

    /// Per-joint bind-space boxes `(center, half_extent)` for a skinned mesh,
    /// computed once and cached by the shared buffer's pointer identity.
    ///
    /// A vertex contributes to a joint only where that joint's weight dominates
    /// ([`JOINT_DOMINANT_WEIGHT`]); a joint that ends up with no dominant vertex
    /// gets a zero-extent box and is skipped, so a skeleton with helper bones does
    /// not scatter degenerate primitives through the volume.
    fn joint_boxes_for(
        &mut self,
        mesh: &Arc<crate::scene::SkinnedMeshData>,
    ) -> Arc<Vec<(Vec3, Vec3)>> {
        let key = Arc::as_ptr(mesh) as usize;
        // `ptr_eq` and not `contains_key`, so the soundness condition is written
        // in the code that depends on it (`vsm_raster::sync_skinned`'s rule): an
        // entry is reused only when it is the same allocation the scene is
        // handing over. With the `Arc` held this can never be false for a live
        // key — and that is the point: the day someone drops the field, this line
        // recomputes instead of lighting with a dead mesh's boxes.
        if let Some((held, cached)) = self.joint_boxes.get(&key) {
            if Arc::ptr_eq(held, mesh) {
                return cached.clone();
            }
        }
        let mut lo: Vec<Vec3> = Vec::new();
        let mut hi: Vec<Vec3> = Vec::new();
        for v in &mesh.vertices {
            let p = Vec3::from(v.pos);
            for (j, w) in v.joints.iter().zip(v.weights) {
                if w < JOINT_DOMINANT_WEIGHT {
                    continue;
                }
                let j = *j as usize;
                if j >= lo.len() {
                    lo.resize(j + 1, Vec3::splat(f32::INFINITY));
                    hi.resize(j + 1, Vec3::splat(f32::NEG_INFINITY));
                }
                lo[j] = lo[j].min(p);
                hi[j] = hi[j].max(p);
            }
        }
        let boxes: Vec<(Vec3, Vec3)> = lo
            .into_iter()
            .zip(hi)
            .map(|(l, h)| {
                if l.x > h.x {
                    (Vec3::ZERO, Vec3::ZERO)
                } else {
                    // A minimum half-extent so a planar joint (a flat fan of
                    // vertices) still occupies voxels instead of a zero-volume
                    // slab the point test can never land inside.
                    ((l + h) * 0.5, ((h - l) * 0.5).max(Vec3::splat(0.02)))
                }
            })
            .collect();
        let arc = Arc::new(boxes);
        // The clone is the whole point: it is what stops the address this entry is
        // filed under from being handed to another mesh while the entry lives.
        self.joint_boxes.insert(key, (mesh.clone(), arc.clone()));
        arc
    }

    /// Drop cache entries the frame's scene no longer names.
    ///
    /// Both maps grew for the session before this existed. The joint boxes are
    /// keyed on a pointer whose allocation the entry now holds, so eviction is
    /// what stops a session-long hold on every skinned mesh that ever appeared;
    /// the meshlet spheres are keyed on a content-addressed asset id, so eviction
    /// is what stops a re-import from accreting its own history.
    ///
    /// Runs before the staging loops below, on the live scene — the placement
    /// `ClassicVgeomNode::run` uses, and for the same reason: it also covers the
    /// frames where the content left entirely.
    fn retain_live(&mut self, scene: &crate::scene::RenderScene) {
        if !self.joint_boxes.is_empty() {
            let live: std::collections::BTreeSet<usize> = scene
                .skinned_meshes
                .iter()
                .map(|m| Arc::as_ptr(m) as usize)
                .collect();
            self.joint_boxes.retain(|k, _| live.contains(k));
        }
        if !self.meshlet_spheres.is_empty() {
            let live: std::collections::BTreeSet<u128> =
                scene.vgeom_assets.iter().map(|a| a.id).collect();
            self.meshlet_spheres.retain(|k, _| live.contains(k));
        }
    }

    /// How many entries the two per-content caches hold (Hardening D).
    ///
    /// `(joint-box meshes, meshlet-sphere assets)`. Published for the reason every
    /// release instrument in this tree is: a cache that was never evicted lights
    /// exactly the same frames an evicted one does.
    pub fn cached_counts(&self) -> (usize, usize) {
        (self.joint_boxes.len(), self.meshlet_spheres.len())
    }

    /// Local-space meshlet spheres of a vgeom asset's **root page** — the coarsest
    /// cut, which the streamer guarantees is always resident (P18.2's "never a hole,
    /// only softer detail"). Cached per asset id.
    ///
    /// Reading the *live* resident cut instead would make GI a function of what the
    /// streamer happened to have paged in, i.e. of frame history — the opposite of
    /// what a determinism gate can hold. The root page is a property of the asset,
    /// so this is both cheaper and reproducible. What it costs is fidelity: a
    /// character-sized prop voxelizes as its coarse cluster spheres, which at the
    /// GI volume's 0.6 m voxels is at or below the grid's own resolution anyway.
    fn meshlet_spheres_for(&mut self, asset: &crate::scene::VgeomAsset) -> Arc<Vec<(Vec3, f32)>> {
        if let Some(cached) = self.meshlet_spheres.get(&asset.id) {
            return cached.clone();
        }
        let spheres = asset
            .source
            .with_page_sections(0, |s| {
                let recs: &[inf_vgeom::MeshletRec] = bytemuck::cast_slice(
                    &s.meshlets[..s.meshlets.len()
                        - s.meshlets.len() % std::mem::size_of::<inf_vgeom::MeshletRec>()],
                );
                // The root page spans levels; the COARSEST level in it is the
                // complete, cheapest cover of the whole mesh.
                let coarsest = recs.iter().map(|r| r.lod_level).max().unwrap_or(0);
                recs.iter()
                    .filter(|r| r.lod_level == coarsest)
                    .map(|r| (Vec3::from(r.center), r.radius))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        // A source whose root page could not be read still contributes: fall back
        // to the whole-mesh bounding sphere from the header, which needs no page.
        let spheres = if spheres.is_empty() {
            let (c, r) = asset.bounds();
            vec![(Vec3::from(c), r)]
        } else {
            spheres
        };
        let arc = Arc::new(spheres);
        self.meshlet_spheres.insert(asset.id, arc.clone());
        arc
    }
}

/// A primitive staged for voxelization: its GPU record + the bounds that order and
/// bin it.
struct Staged {
    bounds: GiBounds,
    gpu: GiInstanceGpu,
}

/// Conservative bounding-sphere radius of a unit cube (±0.5) under `m`'s linear
/// part: half the sum of the transformed axis lengths.
fn box_radius(m: &Mat4) -> f32 {
    0.5 * (m.x_axis.truncate().length()
        + m.y_axis.truncate().length()
        + m.z_axis.truncate().length())
}

fn stage_box(model: Mat4, albedo: [f32; 3], emissive: [f32; 3]) -> Option<Staged> {
    let inv = model.inverse();
    if !inv.is_finite() {
        return None; // a degenerate (zero-scale) transform voxelizes as nothing
    }
    Some(Staged {
        bounds: GiBounds {
            center: model.w_axis.truncate(),
            radius: box_radius(&model),
        },
        gpu: GiInstanceGpu {
            inv_model: inv.to_cols_array(),
            albedo: [albedo[0], albedo[1], albedo[2], KIND_BOX],
            emissive: [emissive[0], emissive[1], emissive[2], 0.0],
        },
    })
}

fn stage_sphere(center: Vec3, radius: f32, albedo: [f32; 3], emissive: [f32; 3]) -> Option<Staged> {
    if !(radius.is_finite() && radius > 0.0) || !center.is_finite() {
        return None;
    }
    // The shader's sphere test is `|local| <= 0.5`, so the model scales the unit
    // sphere by the diameter.
    let model =
        Mat4::from_scale_rotation_translation(Vec3::splat(radius * 2.0), Quat::IDENTITY, center);
    Some(Staged {
        bounds: GiBounds { center, radius },
        gpu: GiInstanceGpu {
            inv_model: model.inverse().to_cols_array(),
            albedo: [albedo[0], albedo[1], albedo[2], KIND_SPHERE],
            emissive: [emissive[0], emissive[1], emissive[2], 0.0],
        },
    })
}

impl RenderNode for GiNode {
    fn run(&mut self, gpu: &GpuContext, encoder: &mut wgpu::CommandEncoder, frame: &FrameData) {
        let s = &frame.settings.gi;
        if !s.enabled {
            // **Wave VIS1a: SSR without GI.** SSR's parameters ride this uniform
            // and its reprojection matrix moves every frame, so a scene that wants
            // reflections without probes needs the block written every frame even
            // though every GI field in it is zero. The `published_disabled` cache
            // stays for the far commoner case — neither feature on — where the
            // constant really is a constant and re-writing it is waste.
            if frame.settings.ssr.enabled {
                self.published_disabled = false;
                gpu.queue.write_buffer(
                    &frame.gi.uniform,
                    0,
                    bytemuck::bytes_of(&GiDataGpu::disabled().with_ssr(frame)),
                );
            } else if !self.published_disabled {
                // Publish the disabled uniform once; the buffer is created once per
                // quality and only this node writes it, so re-writing the constant
                // every frame is redundant.
                gpu.queue.write_buffer(
                    &frame.gi.uniform,
                    0,
                    bytemuck::bytes_of(&GiDataGpu::disabled()),
                );
                self.published_disabled = true;
            }
            if let Ok(mut a) = self.audit.lock() {
                *a = GiAudit::default();
            }
            // GI off ⇒ nothing reads either cache. The off-transition is the one
            // the guard would otherwise hide (`VoxelNode::run`'s rule), and the
            // joint-box entries hold an `Arc` to a whole skinned mesh's vertices.
            self.joint_boxes.clear();
            self.meshlet_spheres.clear();
            return;
        }
        // The enabled path writes the real uniform below, so a later disable must
        // re-publish the disabled block.
        self.published_disabled = false;
        // Hold exactly what this frame's scene names (see `retain_live`).
        self.retain_live(frame.scene);

        let quality = frame.gi.quality;
        let dim = quality.voxel_dim();
        let macro_dim = quality.macro_dim();
        let probe_dims = quality.probe_dims();
        let probe_total = crate::gi::probe_count_of(probe_dims);

        let origin = &frame.view.origin;
        let eye = frame.view.eye_local();
        let extent = s.extent.max(1.0);
        let vsize = extent / dim as f32;
        let vol_min = eye - Vec3::splat(extent * 0.5);

        // ── stage every GI-relevant primitive ────────────────────────────────
        let mut staged: Vec<Staged> = Vec::new();

        // Rigid mesh instances (v1's only source), as oriented boxes.
        for inst in &frame.scene.instances {
            let model = origin.model_matrix(inst.translation, inst.rotation, inst.scale);
            if let Some(p) = stage_box(
                model,
                [inst.color[0], inst.color[1], inst.color[2]],
                inst.emissive,
            ) {
                staged.push(p);
            }
        }

        // Skinned instances, as per-joint boxes carried by the live palette —
        // **behind a whole-instance reject** (island wave NPC1e).
        //
        // The NPC1b audit's finding 6: this is the fifth consumer of
        // `scene.skinned` and the only one that walks it PER JOINT, so a
        // thousand island-class NPCs staged 1 001 × 161 ≈ 161 000 oriented boxes
        // a frame — each an `inverse()` and a 112-byte push — and the volume
        // clip below threw nearly all of them away. `gi`'s CPU recording went
        // 0.162 → 0.778 ms on the island's crowd row, for a pass a crowd adds
        // one draw to. It was routed here with the words *"making it cheap needs
        // a conservative per-instance pre-reject the staging path has no bound
        // for today"*, and that bound is `gi::skinned_model_bound`.
        //
        // **Memoized on `(mesh, palette pointer)`, and the map is a LOCAL.** A
        // crowd's far tier shares one rest-pose palette `Arc` between every
        // agent of an archetype (NPC1b's clause 1), so the bound is derived once
        // for hundreds of instances; and because the map lives for this call and
        // the scene is borrowed for it, no key can outlive its allocation — the
        // same argument `plan_skinned_batches` rests on, and deliberately not
        // the field-cache shape, which would need the `Arc` held.
        //
        // **It fails open**: a bound that cannot be computed (`None`) stages the
        // instance exactly as before. What is dropped is only an instance whose
        // *whole* conservative extent misses the volume, which the per-joint
        // clip would have dropped box by box.
        let mut skinned_bounds: HashMap<(usize, usize), Option<crate::gi::GiBounds>> =
            HashMap::new();
        let mut skinned_rejected = 0_u32;
        for inst in &frame.scene.skinned {
            let Some(mesh) = frame.scene.skinned_meshes.get(inst.mesh) else {
                continue;
            };
            let boxes = self.joint_boxes_for(mesh);
            let model = origin.model_matrix(inst.translation, inst.rotation, inst.scale);
            let key = (inst.mesh, Arc::as_ptr(&inst.palette) as usize);
            let bound = *skinned_bounds
                .entry(key)
                .or_insert_with(|| crate::gi::skinned_model_bound(&boxes, &inst.palette));
            if let Some(b) = bound {
                let world = crate::gi::skinned_world_bound(model, b);
                if b.radius <= 0.0 || !intersects_volume(&world, vol_min, extent) {
                    skinned_rejected += 1;
                    continue;
                }
            }
            for (j, (center, half)) in boxes.iter().enumerate() {
                if half.max_element() <= 0.0 {
                    continue;
                }
                let Some(joint) = inst.palette.get(j) else {
                    continue;
                };
                let local =
                    Mat4::from_scale_rotation_translation(*half * 2.0, Quat::IDENTITY, *center);
                if let Some(p) = stage_box(
                    model * *joint * local,
                    [inst.color[0], inst.color[1], inst.color[2]],
                    inst.emissive,
                ) {
                    staged.push(p);
                }
            }
        }

        // vgeom instances, as the root page's meshlet spheres.
        if !frame.scene.vgeom_instances.is_empty() {
            let spheres: HashMap<u128, Arc<Vec<(Vec3, f32)>>> = frame
                .scene
                .vgeom_assets
                .iter()
                .map(|a| (a.id, self.meshlet_spheres_for(a)))
                .collect();
            for inst in &frame.scene.vgeom_instances {
                let Some(list) = spheres.get(&inst.asset) else {
                    continue;
                };
                let model = origin.model_matrix(inst.translation, inst.rotation, inst.scale);
                let scale = inst.scale.abs().max_element();
                for (c, r) in list.iter() {
                    if let Some(p) = stage_sphere(
                        model.transform_point3(*c),
                        r * scale,
                        [inst.color[0], inst.color[1], inst.color[2]],
                        inst.emissive,
                    ) {
                        staged.push(p);
                    }
                }
            }
        }

        // **Scattered instances, as spheres** (wave VEN1a) — the path that had
        // never reached the bounce.
        //
        // # Why this was the cheapest real win the venue wave had
        //
        // Every practical light in a night interior — a neon sign plate, a
        // string-light chain, a stage skirt, a lit shopfront pane — is a
        // *scattered module*, and a scattered module is what `RenderScene`
        // carries in `scatter` rather than in `instances`. Until this wave GI
        // staged instances, skinned meshes and vgeom and nothing else, so a
        // grammar-built building's every window and every sign contributed
        // exactly zero bounced light: the emissive was drawn, and the wall
        // opposite it stayed at the ambient term. Staging them costs no light
        // slot at all — the 16-light frame ceiling is untouched — because an
        // emissive voxel lights the room through `gi_probes.wgsl`'s injection,
        // which needs no analytic light to exist.
        //
        // # The three rejects, and why each is O(1) before it is O(instances)
        //
        // * **Dust.** A batch whose largest instance is under half a voxel
        //   across cannot be *represented* by the grid: it would fill a voxel
        //   with its albedo and occlude everything behind it, which is how a
        //   meadow of 0.3 m grass tufts becomes an opaque slab at 0.6 m voxels.
        //   So a non-emitting batch below that size is skipped whole, by two
        //   scalars the payload already carries (`bounding_radius × max_scale`)
        //   and without touching one instance. **An emitting batch is never
        //   dust** — a string-light bulb is 40 mm and is the entire point.
        // * **The far band.** A structure-LOD *shell* batch is banded
        //   `[STRUCTURE_LOD_M, cull)`, and no point of a volume 40 m across
        //   centred on the eye is 96 m from it, so the whole batch is provably
        //   invisible here. One compare against the volume's half-diagonal.
        // * **Per instance**, the band test the cull compute applies and the
        //   volume test the generic filter below applies — done *inline*, so a
        //   settlement's 2 000-instance batch pushes only what survives instead
        //   of allocating 2 000 × 112 bytes for the filter to throw away.
        let mut scatter_rejected = 0_u32;
        let mut scatter_decimated = 0_u32;
        for batch in &frame.scene.scatter {
            let data = &batch.data;
            if data.is_empty() {
                continue;
            }
            let emits = batch.emissive.iter().any(|c| *c > 0.0);
            if !crate::gi::scatter_batch_stages(
                emits,
                data.bounding_radius() * data.max_scale(),
                batch.near_distance,
                vsize,
                extent,
            ) {
                scatter_rejected += 1;
                continue;
            }
            let emissive = batch.emissive;
            let unit_radius = data.bounding_radius();
            let stride = data.len().div_ceil(SCATTER_WALK_CEILING).max(1);
            if stride > 1 {
                scatter_decimated += 1;
            }
            for inst in data.instances.iter().step_by(stride) {
                let model = crate::passes::scatter::instance_model(origin, batch.anchor, inst);
                let center = model.w_axis.truncate();
                // The band the cull compute would apply to this instance. The
                // volume is 40 m across and a draw distance is hundreds, so this
                // is inert for almost every batch — and it is what stops a
                // *shell* whose near cut the O(1) test above could not settle
                // from voxelizing a building around the camera's ears.
                let d = f64::from((center - eye).length());
                if d < batch.near_distance
                    || (batch.draw_distance > 0.0 && d >= batch.draw_distance)
                {
                    continue;
                }
                // The instance's own radius: the payload's unit bound scaled by
                // this instance's own largest axis, not the batch's widest — a
                // batch holds one mesh at many sizes.
                let s = inst
                    .scale
                    .abs()
                    .max(inst.scale_yz[0].abs())
                    .max(inst.scale_yz[1].abs());
                let radius = unit_radius * s;
                let bounds = GiBounds { center, radius };
                if !intersects_volume(&bounds, vol_min, extent) {
                    continue;
                }
                // **Albedo per instance, emission per batch** — the one
                // asymmetry this path has that the others do not. A scatter's
                // tint rides on the instance (`ScatterInstanceRaw::color`) and
                // its emission on the batch, because the batch is what the
                // raster binds as a uniform; the projectors bucket on the
                // emission for exactly that reason.
                let albedo = [inst.color[0], inst.color[1], inst.color[2]];
                if let Some(p) = stage_sphere(center, radius, albedo, emissive) {
                    staged.push(p);
                }
            }
        }

        // ── clip to the volume, prioritize, budget ──────────────────────────
        let mut kept: Vec<Staged> = staged
            .into_iter()
            .filter(|p| intersects_volume(&p.bounds, vol_min, extent))
            .collect();
        let candidates = kept.len() as u32;
        let bounds: Vec<GiBounds> = kept.iter().map(|p| p.bounds).collect();
        let order = priority_order(&bounds, eye);
        let budget = (s.instance_budget as usize).min(quality.instance_budget());
        let take = order.len().min(budget);

        // Reorder `kept` into priority order and truncate to the budget — the
        // shader indexes `instances[rank]`, so the upload IS the ordering.
        let mut ordered: Vec<Staged> = Vec::with_capacity(take);
        let mut ordered_bounds: Vec<GiBounds> = Vec::with_capacity(take);
        {
            let mut slots: Vec<Option<Staged>> = kept.drain(..).map(Some).collect();
            for &i in order.iter().take(take) {
                if let Some(p) = slots[i as usize].take() {
                    ordered_bounds.push(p.bounds);
                    ordered.push(p);
                }
            }
        }
        let voxelized = ordered.len() as u32;

        // ── macro-cell bins ─────────────────────────────────────────────────
        let identity: Vec<u32> = (0..ordered_bounds.len() as u32).collect();
        let (cell_offsets, cell_items) =
            bin_macro_cells(&ordered_bounds, &identity, vol_min, extent, macro_dim);

        // ── terrain columns ─────────────────────────────────────────────────
        let mut columns = vec![GiTerrainColumnGpu::default(); (dim * dim) as usize];
        let mut terrain_columns = 0u32;
        if !frame.scene.terrains.is_empty() {
            let world_min = origin.to_world(vol_min);
            let world_max = origin.to_world(vol_min + Vec3::splat(extent));
            for terrain in &frame.scene.terrains {
                let candidates = voxelization_tiles(
                    terrain,
                    (world_min.x, world_min.z),
                    (world_max.x, world_max.z),
                );
                if candidates.is_empty() {
                    continue;
                }
                for z in 0..dim {
                    for x in 0..dim {
                        let local_x = vol_min.x + (x as f32 + 0.5) * vsize;
                        let local_z = vol_min.z + (z as f32 + 0.5) * vsize;
                        let wx = origin.origin().x + local_x as f64;
                        let wz = origin.origin().z + local_z as f64;
                        let Some(col) = sample_terrain_column_in(terrain, &candidates, wx, wz)
                        else {
                            continue;
                        };
                        let height = (col.height - origin.origin().y) as f32;
                        let slot = &mut columns[(z * dim + x) as usize];
                        // Overlapping terrains resolve to the highest surface —
                        // occupancy is a union, not a last-writer-wins.
                        if slot.present == 0 || height > slot.height {
                            let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u32;
                            slot.height = height;
                            slot.albedo = q(col.albedo[0])
                                | (q(col.albedo[1]) << 8)
                                | (q(col.albedo[2]) << 16);
                            slot.present = 1;
                        }
                    }
                }
            }
            terrain_columns = columns.iter().filter(|c| c.present != 0).count() as u32;
        }

        // ── sun + sky ────────────────────────────────────────────────────────
        let (sun_dir, sun_color) = frame
            .scene
            .lights
            .iter()
            .find(|l| l.kind == LightKind::Directional)
            .map(|l| {
                (
                    l.direction.normalize_or_zero(),
                    [
                        l.color[0] * l.intensity,
                        l.color[1] * l.intensity,
                        l.color[2] * l.intensity,
                    ],
                )
            })
            // No directional light: fall back to the scene's **projected** sun
            // (P17.1), so probe radiance tracks the time of day instead of a
            // compile-time constant. A scene with no time-of-day authority
            // projects the retired constant, so this is byte-identical for
            // content that has not opted in.
            .unwrap_or_else(|| {
                let sun = &frame.scene.sun;
                let i = sun.intensity;
                (
                    sun.unit_direction(),
                    [sun.color[0] * i, sun.color[1] * i, sun.color[2] * i],
                )
            });
        // P18.4: the probes read the P17.2 sky-view LUT whenever the scene has a
        // physical atmosphere. Without one the authored gradient still answers, so
        // a pre-P17.2 scene's bounce is unchanged.
        let sky_from_atmosphere = frame.scene.atmosphere.enabled;

        // ── amortization schedule ───────────────────────────────────────────
        // NOTE: `frame.scene.version` is deliberately absent — see `GiSweepKey`.
        // It churns every frame in the shipped player and pinned the cursor in its
        // first slice forever; the sweep's own wrap-around is what bounds staleness
        // after a content change.
        let key = GiSweepKey {
            generation: frame.gi.generation,
            probe_total,
            settings: [
                s.extent.to_bits(),
                s.rays,
                s.intensity.to_bits(),
                s.probe_budget,
                s.instance_budget,
            ],
            // QUANTIZED, not `to_bits()`. A raw sun changes in the low bits every
            // frame under a running TimeOfDay clock, which would reset the sweep
            // every frame and pin the cursor in its first slice forever —
            // amortization paying full price for a fraction of the freshness,
            // exactly where it is supposed to help. See `gi::sun_bucket` for the
            // bucket size and the bounded-staleness consequence.
            sun: sun_bucket(sun_dir, sun_color),
            sky_from_atmosphere,
        };
        if self.sweep_key != Some(key) {
            self.schedule.reset();
            self.sweep_key = Some(key);
        }
        let (probe_start, probe_count) = self.schedule.next(probe_total, s.probe_budget);

        if let Ok(mut a) = self.audit.lock() {
            *a = GiAudit {
                candidates,
                voxelized,
                dropped: candidates - voxelized,
                cell_entries: cell_items.len() as u32,
                terrain_columns,
                probes_updated: probe_count,
                probe_cursor: self.schedule.cursor(),
                skinned_rejected,
                scatter_rejected,
                scatter_decimated,
            };
        }

        // ── uploads ──────────────────────────────────────────────────────────
        let packed: Vec<GiInstanceGpu> = ordered.iter().map(|p| p.gpu).collect();
        Self::ensure(
            gpu,
            &mut self.instances,
            &mut self.instance_cap,
            packed.len().max(1),
            std::mem::size_of::<GiInstanceGpu>(),
            "gi-instances",
        );
        Self::ensure(
            gpu,
            &mut self.cell_offsets,
            &mut self.cell_offsets_cap,
            cell_offsets.len(),
            4,
            "gi-cell-offsets",
        );
        Self::ensure(
            gpu,
            &mut self.cell_items,
            &mut self.cell_items_cap,
            cell_items.len().max(1),
            4,
            "gi-cell-items",
        );
        Self::ensure(
            gpu,
            &mut self.terrain,
            &mut self.terrain_cap,
            columns.len(),
            std::mem::size_of::<GiTerrainColumnGpu>(),
            "gi-terrain-columns",
        );
        if !packed.is_empty() {
            gpu.queue
                .write_buffer(&self.instances, 0, bytemuck::cast_slice(&packed));
        }
        gpu.queue
            .write_buffer(&self.cell_offsets, 0, bytemuck::cast_slice(&cell_offsets));
        if !cell_items.is_empty() {
            gpu.queue
                .write_buffer(&self.cell_items, 0, bytemuck::cast_slice(&cell_items));
        }
        gpu.queue
            .write_buffer(&self.terrain, 0, bytemuck::cast_slice(&columns));

        let sky = &frame.scene.sky;
        let data = GiDataGpu {
            vol_min: [vol_min.x, vol_min.y, vol_min.z, vsize],
            probe_min: [vol_min.x, vol_min.y, vol_min.z, extent],
            dims: [
                dim as f32,
                probe_dims[0] as f32,
                probe_dims[1] as f32,
                probe_dims[2] as f32,
            ],
            params: [1.0, s.intensity, s.rays.max(1) as f32, macro_dim as f32],
            sun_dir: [sun_dir.x, sun_dir.y, sun_dir.z, 0.0],
            sun_color: [sun_color[0], sun_color[1], sun_color[2], 0.0],
            sky_zenith: [sky.zenith[0], sky.zenith[1], sky.zenith[2], 0.0],
            sky_horizon: [sky.horizon[0], sky.horizon[1], sky.horizon[2], 0.0],
            // The SSR half is filled by `with_ssr` below — one packing rule, so
            // the GI-on frame and the GI-off frame march identically.
            params2: [if s.specular { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0],
            sched: [
                probe_start as f32,
                probe_count as f32,
                probe_total as f32,
                if sky_from_atmosphere { 1.0 } else { 0.0 },
            ],
            ssr: [0.0; 4],
            prev_view_proj: [0.0; 16],
            // **The SKY's sun, not the probe march's.** The two lines above
            // take the first analytic directional light, because that is what
            // actually lights the voxels a probe ray hits. The sky is lit by
            // `scene.sun` — the projected time-of-day body the sky pass draws
            // from and `AtmosphereGpu::build` bakes the LUT for — so the
            // ambient and the sky the player can see behind it can never
            // disagree about where the sun is. One door, and it is the sky's.
            sky_sh: sky_sh_lanes(crate::passes::sky_lut::sky_irradiance_memo(
                &frame.scene.atmosphere,
                crate::atmosphere::camera_radius_km(
                    &frame.scene.atmosphere,
                    frame.view.eye_world.y as f32,
                ),
                frame.scene.sun.unit_direction().into(),
                {
                    let i = frame.scene.sun.intensity.max(0.0);
                    [
                        frame.scene.sun.color[0] * i,
                        frame.scene.sun.color[1] * i,
                        frame.scene.sun.color[2] * i,
                    ]
                },
                sky.horizon,
                sky.zenith,
            )),
        }
        .with_ssr(frame);
        gpu.queue
            .write_buffer(&frame.gi.uniform, 0, bytemuck::bytes_of(&data));

        // Voxelize → probe march.
        let vox_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gi-voxelize"),
            layout: &self.voxelize_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: frame.gi.uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.instances.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: frame.gi.voxels.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.cell_offsets.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.cell_items.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.terrain.as_entire_binding(),
                },
            ],
        });
        let probe_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gi-probes"),
            layout: &self.probe_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: frame.gi.uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: frame.gi.voxels.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: frame.gi.sh.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: frame.atmosphere.uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&frame.atmosphere.transmittance),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&frame.atmosphere.sky_view),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Sampler(&frame.atmosphere.sampler),
                },
            ],
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gi-voxelize"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.voxelize_pipeline);
            pass.set_bind_group(0, &vox_bg, &[]);
            let total = dim * dim * dim;
            pass.dispatch_workgroups(total.div_ceil(64), 1, 1);
        }
        if probe_count > 0 {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gi-probes"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.probe_pipeline);
            pass.set_bind_group(0, &probe_bg, &[]);
            pass.dispatch_workgroups(probe_count.div_ceil(64), 1, 1);
        }
    }

    fn name(&self) -> &'static str {
        "gi"
    }
}

#[cfg(test)]
mod boundary_tests {
    /// Code only. The module docs above discuss the GI/VSM boundary at length,
    /// and a ban that could not tell a doc block from a call would have to be
    /// written without the explanation — so full-line comments are stripped and
    /// this `#[cfg(test)]` module is cut off, and what is scanned is the code
    /// that runs. A *trailing* comment naming a shadow page trips the gate on
    /// purpose: move it to its own line.
    ///
    /// `lines()` rather than a `\n` search, so the scan is CRLF-tolerant on a
    /// Windows checkout (the P22 law — the trig gate aborted on exactly that).
    fn code_only(source: &str) -> String {
        source
            .split("#[cfg(test)]")
            .next()
            .unwrap_or(source)
            .lines()
            .filter(|l| !l.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// **GI never reads a virtual shadow page, its residency or its atlas** —
    /// the P18 law (*camera-driven residency never feeds lighting*) as a source
    /// gate, restated at P27.1 because P27 is what creates the second
    /// camera-driven page system for it to be broken by.
    ///
    /// A VSM page exists because a visible pixel's depth asked for it, so a
    /// probe that consulted one would make the irradiance at a fixed world point
    /// a function of where the viewer is standing. The failure would be a
    /// *plausible* picture, which is why it needs a check rather than a comment.
    #[test]
    fn the_gi_pass_never_reads_a_shadow_page() {
        for (label, source) in [
            ("passes/gi.rs", include_str!("gi.rs")),
            ("gi.rs", include_str!("../gi.rs")),
            (
                "gi_voxelize.wgsl",
                include_str!("../shaders/gi_voxelize.wgsl"),
            ),
            ("gi_probes.wgsl", include_str!("../shaders/gi_probes.wgsl")),
        ] {
            let code = code_only(source).to_ascii_lowercase();
            assert!(
                !code.contains("vsm"),
                "{label} names a virtual shadow map in CODE — camera-driven \
                 residency never feeds lighting (the P18 law, restated at P27.1). \
                 GI's occlusion is its own voxel volume; a shadow page darkens the \
                 DIRECT term and never the ambient one."
            );
            // ANTI-VACUITY: the stripper left real code behind, so a `contains`
            // over an empty string is not what passed.
            assert!(
                code.len() > 500 && (code.contains("fn ") || code.contains("@compute")),
                "{label}: the scan found no code at all ({} bytes)",
                code.len()
            );
        }
    }

    /// **A shadow page darkens the DIRECT term and never the AMBIENT one**
    /// (P27.5) — the other half of the seam, and the half P27.5 created.
    ///
    /// The arm above bans VSM from GI's own sources. It could not see the new
    /// meeting point this batch made: `vgeom_mesh.wgsl` and `scatter_mesh.wgsl`
    /// now call `shadow_factor`, so for the first time **four** lit shaders
    /// compute a virtual-shadow factor and a GI irradiance in one fragment, a
    /// few lines apart. Multiplying the second by the first is a one-character
    /// edit and would make the ambient term at a fixed world point a function of
    /// where the viewer stands — the P18 law, broken from the other side.
    ///
    /// Scoped rather than spelled (the P23 law): what is scanned is the region
    /// from the ambient block to the end of the fragment shader, which is where
    /// the irradiance is used, and the ban is on any shadow term appearing in
    /// it. The anti-vacuity is that the region really contains the GI call.
    #[test]
    fn a_shadow_page_never_reaches_the_ambient_term() {
        for (label, source) in [
            ("mesh.wgsl", include_str!("../shaders/mesh.wgsl")),
            (
                "skinned_mesh.wgsl",
                include_str!("../shaders/skinned_mesh.wgsl"),
            ),
            (
                "vgeom_mesh.wgsl",
                include_str!("../shaders/vgeom_mesh.wgsl"),
            ),
            (
                "scatter_mesh.wgsl",
                include_str!("../shaders/scatter_mesh.wgsl"),
            ),
        ] {
            let at = source
                .find("let up = clamp(")
                .unwrap_or_else(|| panic!("{label}: the ambient block moved"));
            let tail: String = source[at..]
                .lines()
                .filter(|l| !l.trim_start().starts_with("//"))
                .collect::<Vec<_>>()
                .join("\n");
            assert!(
                tail.contains("ambient_irradiance("),
                "{label}: the scanned region holds no ambient call, so the ban below is about nothing"
            );
            for banned in ["shadow_factor", "vsm_light_shadow", "vsm_shadow("] {
                assert!(
                    !tail.contains(banned),
                    "{label} applies `{banned}` to the ambient term — a virtual \
                     shadow page is camera-driven residency, and the irradiance \
                     at a fixed world point must not be a function of where the \
                     viewer is standing (the P18 law, from the other side)"
                );
            }
            // …and the DIRECT term above really does take it, or the ban is
            // satisfied by a shader with no shadows in it at all.
            let head = &source[..at];
            assert!(
                head.contains("shadow_factor(in.world_pos, n)"),
                "{label} takes no sun shadow at all, so this arm is measuring a \
                 shader that has nothing to keep out of its ambient term"
            );
        }

        // **AND THE PLACE THE AMBIENT IS NOW COMPUTED** (wave FIX3). The six lit
        // passes used to spell the composition inline — `amb = gi_irradiance(..)`
        // — and the scan above was over the region that held it. They spell it as
        // one call now (`ambient_irradiance`), so the region above proves only
        // that the CALL SITE takes no shadow and the ban has to follow the
        // arithmetic into `env_lighting.wgsl`. Three functions: the composition,
        // the sky term and the probe term.
        let env = include_str!("../shaders/env_lighting.wgsl");
        let mut scanned = 0usize;
        for sig in [
            "fn ambient_irradiance(",
            "fn sky_irradiance(",
            "fn gi_indirect(",
        ] {
            let at = env
                .find(sig)
                .unwrap_or_else(|| panic!("env_lighting.wgsl: no `{sig}`"));
            let open = env[at..]
                .find('{')
                .expect("a signature is followed by a brace")
                + at;
            let mut depth = 0usize;
            let mut end = open;
            for (i, c) in env[open..].char_indices() {
                match c {
                    '{' => depth += 1,
                    '}' => {
                        depth -= 1;
                        if depth == 0 {
                            end = open + i;
                            break;
                        }
                    }
                    _ => {}
                }
            }
            let body: String = env[open..end]
                .lines()
                .filter(|l| !l.trim_start().starts_with("//"))
                .collect::<Vec<_>>()
                .join("\n");
            assert!(
                body.len() > 40,
                "env_lighting.wgsl: `{sig}` extracted as {} bytes — the extraction is broken",
                body.len()
            );
            for banned in ["shadow_factor", "vsm_light_shadow", "vsm_shadow("] {
                assert!(
                    !body.contains(banned),
                    "env_lighting.wgsl's `{sig}` applies `{banned}` to the ambient term — a virtual shadow page is camera-driven residency, and the irradiance at a fixed world point must not be a function of where the viewer is standing (the P18 law, from the other side)"
                );
            }
            scanned += 1;
        }
        assert_eq!(
            scanned, 3,
            "the ambient composition is not three functions any more"
        );
    }
}

#[cfg(test)]
mod sky_sh_tests {
    /// The body of the WGSL function named `sig`, from its opening brace to the
    /// matching close, with whitespace collapsed. `lines()` rather than a
    /// newline search, so the extraction survives a Windows CRLF checkout (the
    /// P22 law).
    fn body(src: &str, sig: &str, what: &str) -> String {
        let at = src
            .find(sig)
            .unwrap_or_else(|| panic!("{what}: no `{sig}`"));
        let open = src[at..]
            .find('{')
            .expect("a signature is followed by a brace")
            + at;
        let mut depth = 0usize;
        let mut end = open;
        for (i, c) in src[open..].char_indices() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = open + i;
                        break;
                    }
                }
                _ => {}
            }
        }
        assert!(end > open, "{what}: unbalanced braces");
        src[open + 1..end]
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .collect::<Vec<_>>()
            .join(
                "
",
            )
    }

    /// **THE MIRROR THE COMMENT PROMISES** (wave FIX3).
    ///
    /// `sky_irradiance` exists twice — once in `env_lighting.wgsl`, where six
    /// lit passes read it, and once in `gi_probes.wgsl`, where the probe march
    /// uses it to light the surfaces its rays hit. They cannot share a snippet:
    /// each file declares its own `GiData` (the probe pass binds the uniform at
    /// `@group(0)`, the lit passes at their env group) and WGSL has no forward
    /// declarations, so a shared fragment would have to be composed before a
    /// struct that does not exist yet.
    ///
    /// A comment saying "keep these identical" is not a pin — the VIS1a audit's
    /// ruling, met again. If the two drift, a probe's idea of how bright the sky
    /// is stops being the shading surface's, and the `-sky` correction in the
    /// march stops cancelling the `+sky` in the shader: the frame would be lit
    /// twice in some directions and not at all in others.
    ///
    /// Mutation-verified: changing `0.66666667` to `0.6667` in either copy fails
    /// this arm.
    #[test]
    fn the_two_sky_irradiance_bodies_are_the_same() {
        const ENV: &str = include_str!("../shaders/env_lighting.wgsl");
        const PROBES: &str = include_str!("../shaders/gi_probes.wgsl");
        let sig = "fn sky_irradiance(n: vec3<f32>) -> vec3<f32>";
        let a = body(ENV, sig, "env_lighting.wgsl");
        let b = body(PROBES, sig, "gi_probes.wgsl");
        assert!(
            a.contains("sky_sh0") && a.contains("0.66666667"),
            "the extraction is broken, so an empty comparison would pass: {a:?}"
        );
        assert_eq!(
            a, b,
            "the two `sky_irradiance` copies have drifted. The probe march's
             `-sky` correction and the lit passes' `+sky` term would stop
             cancelling.

env_lighting:
{a}

gi_probes:
{b}"
        );
    }

    /// The probe march must not gather the sky on a miss any more — that is the
    /// whole no-double-count ruling, and it is one line that a future edit could
    /// put back without any test noticing.
    #[test]
    fn the_probe_miss_term_is_zero() {
        const PROBES: &str = include_str!("../shaders/gi_probes.wgsl");
        let code: String = PROBES
            .lines()
            .filter(|l| !l.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join(
                "
",
            );
        assert!(
            code.contains("radiance = vec3<f32>(0.0);"),
            "the probe march no longer zeroes its miss term"
        );
        assert!(
            code.contains("radiance = radiance - gi_sky_radiance(dir);"),
            "the probe march no longer subtracts the sky it blocks — the lit
             passes add the whole sky, so without this the open sky is counted
             twice"
        );
    }
}
