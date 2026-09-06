//! GPU skinning pass (P11.1): the **additive** skinned variant of the rigid mesh
//! pass ([`super::mesh`]). It draws [`SkinnedInstance`]s — each a bind-space
//! [`SkinnedMeshData`] deformed in the vertex shader by a per-instance joint
//! **palette** — into the same MSAA scene targets, right after the rigid meshes.
//!
//! ## Byte-stability
//!
//! When `scene.skinned` is empty the pass emits **no commands** (an early return
//! in [`RenderNode::run`]), so every pre-P11 scene — and the unskinned pipeline —
//! renders identically. The rigid pipeline in [`super::mesh`] is untouched.
//!
//! ## Pipeline design
//!
//! * **Vertex layout** — buffer 0 is the mesh's [`SkinnedVertex`] stream
//!   (`pos @0`, `normal @1`, `joints @2` = `Uint32x4`, `weights @3` =
//!   `Float32x4`); buffer 1 is the per-instance [`InstanceRaw`] (model / normal
//!   matrix / color / pick id / pbr / emissive) at `@location(4..=14)`, selected
//!   per draw via `base_instance`.
//! * **Bind groups** — `@group(0)` view uniforms + `@group(1)` lights (shared
//!   with the rigid pass), `@group(2)` **reserved for the material seam** (an
//!   empty bind group today, mirroring the `inf-material` convention that binds
//!   material textures at group 2), and `@group(3)` the joint-matrix
//!   **palette atlas** (`array<mat4x4<f32>>`).
//! * **Batching** — ONE draw per `(mesh)` run over a shared atlas, which is the
//!   optimization this file called "the documented P15 optimization" and did not
//!   build for four years of waves. See [`plan_skinned_batches`].
//!
//! ## The atlas (wave NPC1b)
//!
//! Before this wave the pass allocated **one storage buffer and one bind group
//! per skinned instance**, wrote each palette with its own `write_buffer`, and
//! issued **one draw per instance** — twice, because the depth prepass walks the
//! same list. At a thousand NPCs that is 2 002 draws, 2 002 bind-group binds and
//! 31.28 MB of palette upload a frame (`crowd_sweep`'s wall-3 column), and the
//! palettes were *byte-identical* across the 710 agents the sim-LOD ladder had
//! decided not to pose at all.
//!
//! Now: every distinct palette (by `Arc` pointer, so a shared one is uploaded
//! **once**) is appended to one atlas buffer at a matrix granularity, and each
//! instance carries its block's `(offset, joint count)` in two channels of the
//! instance stream that have been reserved-zero since P7.1 — `emissive.w` and
//! `pbr.z`. There is no seventeenth vertex attribute, because there is no
//! seventeenth address: this pipeline is at the `max_vertex_attributes: 16` wall
//! exactly (`docs/memos/p26-5-vertex-streams.md`), and channel packing into an
//! existing attribute is the route that memo names.
//!
//! The **count** is not bookkeeping. `pad_palette` used to fill a power-of-two
//! tail with the last live matrix so that a vertex whose joint index ran past its
//! own skeleton read something of its own rig (round-2 finding R2-6); in an atlas
//! that vertex would read the *next character's* palette, which is the same
//! defect made worse. The clamp moved into the shader — `palette[base +
//! min(j, count - 1)]` — where it is exact, needs no padding at all, and holds
//! for a packed atlas that padding cannot.
//!
//! ## The third part of P15, which is STILL NOT BUILT (NPC1b audit)
//!
//! The sentence this file carried since P11.1 named **three** things: *"a shared
//! palette atlas + instanced draw **(and a GPU palette compute pass)** is the
//! documented P15 optimization"*. Wave NPC1b built the first two, and rewrote the
//! sentence in a form that no longer mentions the third — so a forward reference
//! was retired by a wave that met two thirds of it. It is restored here because
//! it is the lever for the two CPU stations that wave's own headline measured as
//! a crowd's real cost:
//!
//! * the **projection** builds a fresh `Vec<Mat4>` per posed character per frame
//!   (`inf_anim::skinning_matrices`) — 288 × 161 matrices and 2.9 MB of fresh
//!   allocation a frame at the island's N = 1 000, measured at **+2.95 ms**
//!   against a 1.5 ms `PROJECTION_BUDGET_MS`;
//! * this pass then **copies** them into the atlas — the loop
//!   `fill_palette_atlas` makes once-per-block.
//!
//! A compute pass that composed `global · inverse_bind` on the GPU from a
//! joint-local pose stream would delete both. It is not built, and unlike the
//! atlas it is not a two-hundred-line change: it needs the pose to reach the
//! renderer as *data* rather than as a finished palette, which is a projector
//! contract and not a pass.

use glam::Mat3;
use inf_math::FloatingOrigin;

use crate::camera::{DEPTH_COMPARE, DEPTH_FORMAT};
use crate::gpu::GpuContext;
use crate::graph::RenderNode;
use crate::passes::mesh::{InstanceRaw, LightsUniform};
use crate::renderer::{FrameData, SCENE_FORMAT, SCENE_SAMPLES};
use crate::scene::{SkinnedInstance, SkinnedMeshData};
use crate::settings::RenderSettings;

/// Vertex attributes for one [`SkinnedVertex`] (buffer 0).
///
/// **Five attributes, and the fifth is at 15** (P26.5). The instance block owns
/// `4..=14` here (one location further along than the rigid path, which starts
/// at 3), so the uv had exactly one address left: `@location(15)`, the last one
/// `Limits::default()`'s `max_vertex_attributes: 16` allows. That is the wall
/// `docs/memos/p26-5-vertex-streams.md` measures — this pipeline is now **full**,
/// and a tangent stream cannot join it without packing two channels into one
/// attribute or raising a limit the renderer has never raised.
const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 5] = [
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x3,
        offset: 0,
        shader_location: 0,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x3,
        offset: 12,
        shader_location: 1,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x2,
        offset: 24,
        shader_location: 15,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Uint32x4,
        offset: 32,
        shader_location: 2,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 48,
        shader_location: 3,
    },
];

/// Per-instance [`InstanceRaw`] attributes (buffer 1), at `@location(4..=14)` so
/// they clear the vertex attributes above.
const INSTANCE_ATTRIBUTES: [wgpu::VertexAttribute; 11] = [
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 0,
        shader_location: 4,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 16,
        shader_location: 5,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 32,
        shader_location: 6,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 48,
        shader_location: 7,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 64,
        shader_location: 8,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 80,
        shader_location: 9,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 96,
        shader_location: 10,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 112,
        shader_location: 11,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Uint32x4,
        offset: 128,
        shader_location: 12,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 144,
        shader_location: 13,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 160,
        shader_location: 14,
    },
];

/// `Limits::default()`'s ceiling on vertex attributes **across every buffer of
/// one pipeline** — the wall `docs/memos/p26-5-vertex-streams.md` measures.
///
/// The renderer has never raised a limit, so this is the number that decides
/// whether a tangent stream can join the skinned path. It is a `const` rather
/// than a sentence in a doc block because the memo's whole argument for
/// deferring the tangent to P28.2 rests on it, and the assertion below turns
/// "this pipeline is full" from a claim into a build failure (P26.5 audit).
const MAX_VERTEX_ATTRIBUTES: usize = 16;

/// **The skinned pipeline is at the wall, exactly.** Five vertex attributes plus
/// eleven instance ones is sixteen, and the uv took the last address there is
/// (`@location(15)`). Adding a seventeenth is a `wgpu` validation failure on
/// every adapter at `create_render_pipeline`; this fails the build instead, at
/// the file that would have to change.
const _: () = {
    assert!(VERTEX_ATTRIBUTES.len() + INSTANCE_ATTRIBUTES.len() == MAX_VERTEX_ATTRIBUTES);
    // …and no location is past the last one the limit allows, which is a
    // different fact from the count: `@location(15)` with a gap would satisfy
    // one and not the other.
    let mut i = 0;
    while i < VERTEX_ATTRIBUTES.len() {
        assert!((VERTEX_ATTRIBUTES[i].shader_location as usize) < MAX_VERTEX_ATTRIBUTES);
        i += 1;
    }
    let mut j = 0;
    while j < INSTANCE_ATTRIBUTES.len() {
        assert!((INSTANCE_ATTRIBUTES[j].shader_location as usize) < MAX_VERTEX_ATTRIBUTES);
        j += 1;
    }
};

fn vertex_layouts() -> [Option<wgpu::VertexBufferLayout<'static>>; 2] {
    [
        Some(wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<crate::scene::SkinnedVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &VERTEX_ATTRIBUTES,
        }),
        Some(wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<InstanceRaw>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &INSTANCE_ATTRIBUTES,
        }),
    ]
}

/// Matrices the joint-palette atlas may hold in one frame.
///
/// **16 MiB** at 64 bytes a matrix — an eighth of `Limits::default()`'s
/// `max_storage_buffer_binding_size` (128 MiB), and **1 628 distinct**
/// island-class 161-bone poses. The number that binds is distinct *poses*, not
/// agents, because a crowd shares its palettes: the N = 1 000 sweep fills 292
/// blocks of it, and a level would have to hold sixteen hundred separately-posed
/// characters *inside one frame* to reach it.
///
/// It is a **counted** ceiling, not a silent one — see
/// [`SkinnedBatches::dropped`], the same rule `VSM_MAX_GROUPS` follows.
pub const SKINNED_PALETTE_MATRICES: usize = 1 << 18;

/// The atlas offset rides an `f32` channel, so the ceiling has to stay inside the
/// integers an `f32` represents exactly (`2^24`). It also has to fit the default
/// storage-buffer binding size, which is what actually stops a device error.
const _: () = {
    assert!(SKINNED_PALETTE_MATRICES <= (1 << 24));
    assert!(SKINNED_PALETTE_MATRICES * 64 <= 128 * 1024 * 1024);
};

/// One draw: a contiguous run of instances sharing a mesh.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SkinnedRun {
    /// Index into `RenderScene::skinned_meshes`.
    pub mesh: usize,
    /// First instance of the run in the packed instance buffer — the draw's
    /// `base_instance`.
    pub first_instance: u32,
    /// Instances in the run.
    pub count: u32,
}

/// **What one frame's skinned draws are, derived from the scene alone.**
///
/// Pure: no device, no adapter, no frame. `SkinnedMeshNode::sync` calls this
/// and then executes it, so an instrument that wants the draw count or the
/// palette bytes calls the same function the pass does rather than a
/// re-derivation of it (the "a gate must aim at the thing it names" rule — the
/// N-sweep's palette column was a *multiplication* before this wave, and this is
/// what makes it a measurement).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SkinnedBatches {
    /// Scene-instance indices, in draw order — grouped by mesh, stable within a
    /// mesh, and with dropped instances removed.
    pub order: Vec<u32>,
    /// Per entry of [`order`](Self::order): its palette block's `(offset in
    /// matrices, live joint count)`, which is what the instance stream carries.
    pub palette: Vec<(u32, u32)>,
    /// One draw each.
    pub runs: Vec<SkinnedRun>,
    /// Distinct palette blocks in the atlas. **Fewer than `order.len()` is the
    /// whole point**: a crowd's far tier shares one.
    pub blocks: usize,
    /// Matrices the atlas holds — `matrices * 64` bytes uploaded per frame.
    pub matrices: usize,
    /// Instances not drawn because [`SKINNED_PALETTE_MATRICES`] was reached.
    pub dropped: usize,
}

/// Plan one frame's skinned draws: group by mesh, deduplicate palettes by `Arc`
/// identity, and lay the surviving blocks out in one atlas.
///
/// The ordering is a **stable** sort on the mesh index, so a scene with one
/// skinned mesh — every character scene in this tree, and every golden — keeps
/// exactly the instance order it had before this wave and draws exactly the same
/// pixels through one call instead of N.
///
/// # The offsets rise at first use, and `fill_palette_atlas` depends on it
///
/// A block's offset is `matrices` at the moment it is created and `matrices`
/// only grows, so the entry of [`SkinnedBatches::palette`] whose offset is at or
/// past everything before it **is** that block's first instance, and every later
/// entry naming the block is a repeat. That is what lets the atlas be filled once
/// per block rather than once per instance;
/// `the_block_offsets_rise_at_first_use_and_repeat_below_the_watermark` is the
/// arm that keeps the invariant true, and it is stated here rather than only at
/// the loop that uses it because it is a property of *this* function.
pub fn plan_skinned_batches(scene: &crate::scene::RenderScene) -> SkinnedBatches {
    let mut order: Vec<u32> = (0..scene.skinned.len() as u32).collect();
    order.sort_by_key(|i| scene.skinned[*i as usize].mesh);

    // Palette blocks, keyed on the `Arc`'s address. Sound for the same reason
    // `skinned_meshes`' upload cache is: the scene holds the `Arc` for the whole
    // frame, so the allocation cannot be freed and re-used under a live key.
    // Key 0 is the sentinel for an EMPTY palette — one shared identity block, so
    // a scene full of cloth ribbons costs one matrix and not one each.
    let mut blocks: std::collections::HashMap<usize, (u32, u32)> = std::collections::HashMap::new();
    let mut matrices = 0usize;
    let mut kept: Vec<u32> = Vec::with_capacity(order.len());
    let mut palette: Vec<(u32, u32)> = Vec::with_capacity(order.len());
    let mut dropped = 0usize;

    for i in order {
        let inst = &scene.skinned[i as usize];
        let key = if inst.palette.is_empty() {
            0
        } else {
            std::sync::Arc::as_ptr(&inst.palette) as usize
        };
        let block = match blocks.get(&key) {
            Some(hit) => *hit,
            None => {
                let live = inst.palette.len().max(1);
                if matrices + live > SKINNED_PALETTE_MATRICES {
                    dropped += 1;
                    continue;
                }
                let fresh = (matrices as u32, live as u32);
                matrices += live;
                blocks.insert(key, fresh);
                fresh
            }
        };
        kept.push(i);
        palette.push(block);
    }

    // One run per contiguous mesh, over the instances that survived.
    let mut runs: Vec<SkinnedRun> = Vec::new();
    for (slot, i) in kept.iter().enumerate() {
        let mesh = scene.skinned[*i as usize].mesh;
        match runs.last_mut() {
            Some(run) if run.mesh == mesh => run.count += 1,
            _ => runs.push(SkinnedRun {
                mesh,
                first_instance: slot as u32,
                count: 1,
            }),
        }
    }

    SkinnedBatches {
        order: kept,
        palette,
        runs,
        blocks: blocks.len(),
        matrices,
        dropped,
    }
}

/// **Fill the atlas staging buffer from a plan — once per BLOCK.** Returns the
/// number of blocks written, which is the claim (NPC1b audit).
///
/// The wave shipped this loop keyed on the *instance*, under a comment that said
/// "written once per BLOCK, not once per instance" and then described writing it
/// once per instance ("the second agent sharing an `Arc` re-writes the same bytes
/// it already holds"). Idempotent, yes — and `O(instances × joints)` in the one
/// loop the whole clause exists to make `O(blocks × joints)`. On the island at
/// N = 1 000 that is 1 001 × 161 matrix copies a frame where 290 × 161 are
/// needed: **3.45× the memcpy**, in `SkinnedMeshNode::sync`, which is CPU render
/// time — the station the wave's own headline says the crowd actually costs.
///
/// The skip is exact rather than a cache: [`plan_skinned_batches`] assigns block
/// offsets at the running `matrices` watermark, so an entry at or past the
/// watermark is that block's first instance and everything below it is a repeat.
/// The identity prefill is what serves an EMPTY palette, whose block is one
/// matrix nobody writes.
pub(crate) fn fill_palette_atlas(
    scratch: &mut Vec<[f32; 16]>,
    scene: &crate::scene::RenderScene,
    plan: &SkinnedBatches,
) -> usize {
    scratch.clear();
    scratch.resize(plan.matrices.max(1), glam::Mat4::IDENTITY.to_cols_array());
    let mut watermark = 0usize;
    let mut written = 0usize;
    for (slot, i) in plan.order.iter().enumerate() {
        let (offset, live) = plan.palette[slot];
        if (offset as usize) < watermark {
            continue;
        }
        let inst = &scene.skinned[*i as usize];
        for (j, m) in inst.palette.iter().take(live as usize).enumerate() {
            scratch[offset as usize + j] = m.to_cols_array();
        }
        watermark = offset as usize + live as usize;
        written += 1;
    }
    written
}

/// Build an [`InstanceRaw`] for a skinned instance (origin-relative model matrix,
/// inverse-transpose normal matrix), mirroring `InstanceRaw::pack`.
///
/// `block` is the instance's `(atlas offset, live joint count)`, packed into the
/// two `f32` channels of the shared instance stream that the skinned path had
/// never used: `pbr.z` (the rigid path's alpha cutoff) and `emissive.w`
/// (reserved on **both** paths since P7.1, and dropped by both vertex stages).
/// Both values are integers well under `2^24`, so the `f32` round-trip is exact
/// rather than approximately exact.
///
/// # `pbr.w` carries the blend code AND the cutoff (wave CHAR1a.2)
///
/// The sentence above used to end "— a skinned surface is opaque", and that was
/// the whole defect: hair cards, eyelashes and a cut-out garment are skinned
/// surfaces that are *not* opaque, and there was no channel left to say so. This
/// pipeline is at the `max_vertex_attributes: 16` wall exactly
/// (`docs/memos/p26-5-vertex-streams.md`), so there is no seventeenth address and
/// no sixteenth channel — `pbr.z` is the joint count, `emissive.w` the atlas
/// offset, `misc.yzw` the virtual-texture set, `misc.x` the pick id, `color.a`
/// the alpha the test reads. `pbr.w` is the one that was still zero.
///
/// So both ride it: **`blend * 4.0 + cutoff`**, with `cutoff` clamped to
/// `[0, 1]`. The multiplier is 4 rather than 2 so that `cutoff == 1.0` cannot
/// carry into the next blend code; `floor(w / 4)` recovers the code and the
/// remainder is the threshold, and both are exact in `f32` (the largest value is
/// 9.0, whose ulp is 2⁻²⁰). The vertex stage unpacks them into `pbr.zw` in the
/// **rigid path's own order** — `z` the cutoff, `w` the code — so the two
/// fragment stages read a masked surface with the identical two lines.
///
/// An opaque instance packs `0 * 4 + 0.5 = 0.5` where it used to pack `0.0`. The
/// fragment's test is `w > 0.5 && w < 1.5`, which `0.0` and `0.5` both fail, so
/// every committed skinned golden renders the identical pixels — proven, not
/// assumed, by running them under `INF_GOLDEN_STRICT=1`.
fn instance_raw(origin: &FloatingOrigin, inst: &SkinnedInstance, block: (u32, u32)) -> InstanceRaw {
    let model = origin.model_matrix(inst.translation, inst.rotation, inst.scale);
    let inv_scale = inst.scale.max(glam::Vec3::splat(1e-6)).recip();
    let nrm = Mat3::from_quat(inst.rotation) * Mat3::from_diagonal(inv_scale);
    let n = nrm.to_cols_array_2d();
    InstanceRaw {
        model: model.to_cols_array(),
        normal_mat: [
            n[0][0], n[0][1], n[0][2], 0.0, //
            n[1][0], n[1][1], n[1][2], 0.0, //
            n[2][0], n[2][1], n[2][2], 0.0,
        ],
        color: inst.color,
        // P26.3: the same three reserved words the rigid path uses, packed by
        // the same rule — so a skinned surface and a rigid one cannot disagree
        // about what "this instance samples nothing" looks like on the wire.
        misc: {
            let s = inst.vt.slots();
            [inst.id, s[0], s[1], s[2]]
        },
        pbr: [
            inst.metallic,
            inst.roughness,
            block.1 as f32,
            inst.blend as f32 * 4.0 + inst.cutoff.clamp(0.0, 1.0),
        ],
        emissive: [
            inst.emissive[0],
            inst.emissive[1],
            inst.emissive[2],
            block.0 as f32,
        ],
    }
}

/// GPU buffers for one uploaded [`SkinnedMeshData`].
struct GpuSkinnedMesh {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_count: u32,
}

pub struct SkinnedMeshNode {
    pipeline: wgpu::RenderPipeline,
    /// Wireframe (`PolygonMode::Line`) variant (R-P2), present only with the
    /// `POLYGON_MODE_LINE` feature; selected when the frame's view mode is
    /// [`ViewMode::Wireframe`](crate::renderer::ViewMode::Wireframe).
    pipeline_wire: Option<wgpu::RenderPipeline>,
    /// The single-sample, fragment-less depth-prepass twin (wave VIS1a). Binds
    /// view at `@group(0)` and the joint palette at `@group(1)`.
    pipeline_depth: wgpu::RenderPipeline,
    joints_bgl: wgpu::BindGroupLayout,
    /// AO + shadows + GI env bind at `@group(2)` (P13.3b; was the AO-only bind).
    env: super::EnvBinding,
    lights_buf: wgpu::Buffer,
    lights_bg: wgpu::BindGroup,
    /// Per `RenderScene::skinned_meshes` entry, its key into
    /// [`mesh_cache`](Self::mesh_cache) — so `SkinnedInstance::mesh` (an index into
    /// the scene's list) still resolves in one hop.
    meshes: Vec<usize>,
    /// Uploaded geometry by `Arc` identity (P18.3). Holds the `Arc` alongside the
    /// buffers so the pointer used as the key can never be recycled under a live
    /// entry; entries not referenced by a sync are dropped, which frees them.
    mesh_cache: std::collections::HashMap<usize, (std::sync::Arc<SkinnedMeshData>, GpuSkinnedMesh)>,
    /// **One** joint-palette storage buffer for the whole frame — the atlas — and
    /// its bind group, both persistent and both grown to a power of two so a
    /// crowd that gains an agent reallocates `O(log N)` times over a session
    /// rather than every frame (`GpuSkinnedInstance`'s rule, applied to one
    /// object instead of N).
    atlas: Option<(wgpu::Buffer, wgpu::BindGroup)>,
    /// The atlas's allocated size in bytes — the grow test.
    atlas_capacity: u64,
    /// The CPU staging buffer the atlas is written from, **kept between frames**.
    /// This is the fourth of the four joint-length allocations wall 3 counted; the
    /// other three are the projector's, and the crowd's far tier no longer pays
    /// them at all (it shares one derivation).
    scratch: Vec<[f32; 16]>,
    /// This frame's draws (see [`plan_skinned_batches`]).
    runs: Vec<SkinnedRun>,
    /// One [`InstanceRaw`] per *drawn* instance, in `runs` order.
    instance_buf: Option<wgpu::Buffer>,
    uploaded_version: Option<(u64, glam::DVec3)>,
    active: bool,
}

impl SkinnedMeshNode {
    pub fn new(gpu: &GpuContext, view_bgl: &wgpu::BindGroupLayout) -> Self {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("skinned-mesh"),
                source: wgpu::ShaderSource::Wgsl(super::shader_source("skinned").into()),
            });

        // @group(1) lights (same layout/contents as the rigid mesh pass).
        let lights_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("skinned-lights"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let lights_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("skinned-lights"),
            size: std::mem::size_of::<LightsUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let lights_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("skinned-lights"),
            layout: &lights_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: lights_buf.as_entire_binding(),
            }],
        });

        // @group(2) AO + shadows + GI env bind (byte-stable when all are off).
        let env = super::EnvBinding::new(gpu);

        // @group(3) joint-matrix storage buffer.
        let joints_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("skinned-joints"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("skinned-mesh"),
                bind_group_layouts: &[
                    Some(view_bgl),
                    Some(&lights_bgl),
                    Some(&env.bgl),
                    Some(&joints_bgl),
                ],
                immediate_size: 0,
            });

        // Fill + (feature-gated) R-P2 wireframe variants, identical but for the
        // primitive state — see the rigid `mesh` pass for the same shape.
        let make_pipeline = |label: &str, primitive: wgpu::PrimitiveState| {
            gpu.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(&layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("vs"),
                        compilation_options: Default::default(),
                        buffers: &vertex_layouts(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some("fs"),
                        compilation_options: Default::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: SCENE_FORMAT,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    primitive,
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: DEPTH_FORMAT,
                        depth_write_enabled: Some(true),
                        depth_compare: Some(DEPTH_COMPARE),
                        stencil: Default::default(),
                        bias: Default::default(),
                    }),
                    multisample: wgpu::MultisampleState {
                        count: SCENE_SAMPLES,
                        ..Default::default()
                    },
                    multiview_mask: None,
                    cache: None,
                })
        };

        let pipeline = make_pipeline(
            "skinned-mesh",
            wgpu::PrimitiveState {
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
        );
        let pipeline_wire = gpu
            .device
            .features()
            .contains(wgpu::Features::POLYGON_MODE_LINE)
            .then(|| {
                make_pipeline(
                    "skinned-mesh-wire",
                    wgpu::PrimitiveState {
                        polygon_mode: wgpu::PolygonMode::Line,
                        ..Default::default()
                    },
                )
            });

        // **The depth-prepass pipeline** (wave VIS1a). Two groups — view and the
        // joint palette — because a depth-only pass binds no lights and no
        // environment; and the env group in particular must NOT be bound here,
        // since it carries the prepass texture at `ENV_SCENE_DEPTH` and this pass
        // has that texture as its depth attachment.
        //
        // No fragment stage: `skinned_mesh.wgsl`'s `fs` never discards.
        let depth_shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("skinned-depth"),
                source: wgpu::ShaderSource::Wgsl(super::shader_source("skinned_depth").into()),
            });
        let depth_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("skinned-depth"),
                bind_group_layouts: &[Some(view_bgl), Some(&joints_bgl)],
                immediate_size: 0,
            });
        let pipeline_depth = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("skinned-depth"),
                layout: Some(&depth_layout),
                vertex: wgpu::VertexState {
                    module: &depth_shader,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    // The SAME vertex layouts as the colour pipeline, so one
                    // geometry buffer and one instance buffer serve both. The
                    // attributes the depth entry does not read are simply unread.
                    buffers: &vertex_layouts(),
                },
                fragment: None,
                primitive: wgpu::PrimitiveState {
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(DEPTH_COMPARE),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState::default(), // single-sample
                multiview_mask: None,
                cache: None,
            });

        Self {
            pipeline,
            pipeline_wire,
            pipeline_depth,
            joints_bgl,
            env,
            lights_buf,
            lights_bg,
            meshes: Vec::new(),
            mesh_cache: std::collections::HashMap::new(),
            atlas: None,
            atlas_capacity: 0,
            scratch: Vec::new(),
            runs: Vec::new(),
            instance_buf: None,
            uploaded_version: None,
            active: false,
        }
    }

    /// Re-upload geometry, instance data, and palettes when the scene changed or
    /// the floating origin rebased.
    fn sync(&mut self, gpu: &GpuContext, frame: &FrameData) {
        let key = (frame.scene.version, frame.view.origin.origin());
        if self.uploaded_version == Some(key) {
            return;
        }
        self.uploaded_version = Some(key);

        // Lights (same projection as the rigid pass).
        let lights =
            LightsUniform::from_scene(frame.scene, &frame.view.origin, frame.vsm_light_slots);
        gpu.queue
            .write_buffer(&self.lights_buf, 0, bytemuck::bytes_of(&lights));

        // ── Mesh geometry, keyed on IDENTITY rather than on the frame (P18.3) ──
        //
        // Bind-space geometry does not change when a pose does, but this used to
        // re-upload every vertex of every skinned mesh whenever `scene.version`
        // moved — which for an editor is every gizmo tick of an unrelated entity.
        // The scene now shares each buffer as an `Arc`, so "is this the same
        // geometry I already uploaded?" is a pointer comparison. The cache holds
        // the `Arc` itself, which is what makes the pointer a sound key: the
        // allocation cannot be freed and reused under a live entry.
        //
        // Palettes below are deliberately NOT cached — they are the per-frame part.
        let mut cache = std::mem::take(&mut self.mesh_cache);
        self.meshes = frame
            .scene
            .skinned_meshes
            .iter()
            .map(|m| {
                let key = std::sync::Arc::as_ptr(m) as usize;
                let entry = cache
                    .remove(&key)
                    .unwrap_or_else(|| (m.clone(), upload_mesh(gpu, m)));
                self.mesh_cache.insert(key, entry);
                key
            })
            .collect();
        // Whatever `cache` still holds was not referenced this frame — dropped
        // here, which releases its GPU buffers.

        // ── the plan, then the atlas, then the instances (wave NPC1b) ──────
        let plan = plan_skinned_batches(frame.scene);
        if plan.dropped > 0 {
            tracing::warn!(
                "inf-render: {} skinned instances past the {}-matrix joint-palette \
                 atlas drew nothing this frame",
                plan.dropped,
                SKINNED_PALETTE_MATRICES
            );
        }
        self.runs.clone_from(&plan.runs);

        // The atlas: every distinct palette once, laid out at the offsets the plan
        // assigned. Written from a scratch buffer that outlives the frame, so a
        // thousand characters cost one allocation and one `write_buffer` rather
        // than a thousand of each. `max(1)` keeps the storage buffer non-empty on
        // a frame whose only skinned instances were dropped.
        //
        // **Once per BLOCK** — see `fill_palette_atlas`. The wave shipped this
        // loop keyed on the instance, so the 710 `Far` agents sharing one `Arc`
        // re-copied the same 161 matrices 710 times a frame.
        let blocks_written = fill_palette_atlas(&mut self.scratch, frame.scene, &plan);
        debug_assert_eq!(
            blocks_written, plan.blocks,
            "the atlas fill wrote {blocks_written} blocks of {} — the watermark \
             skip and the plan's offsets disagree",
            plan.blocks
        );
        let bytes = std::mem::size_of_val(self.scratch.as_slice()) as u64;
        if self.atlas_capacity < bytes || self.atlas.is_none() {
            let capacity = bytes.next_power_of_two().max(64);
            let buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("skinned-palette-atlas"),
                size: capacity,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("skinned-palette-atlas"),
                layout: &self.joints_bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf.as_entire_binding(),
                }],
            });
            self.atlas = Some((buf, bg));
            self.atlas_capacity = capacity;
        }
        if let Some((buf, _)) = self.atlas.as_ref() {
            gpu.queue
                .write_buffer(buf, 0, bytemuck::cast_slice(&self.scratch));
        }

        // Instance transforms, in the plan's draw order — one `InstanceRaw` per
        // DRAWN instance, selected per draw by `base_instance` over a contiguous
        // run rather than one instance at a time.
        let raws: Vec<InstanceRaw> = plan
            .order
            .iter()
            .zip(&plan.palette)
            .map(|(i, block)| {
                instance_raw(
                    &frame.view.origin,
                    &frame.scene.skinned[*i as usize],
                    *block,
                )
            })
            .collect();
        self.active = !raws.is_empty();
        self.instance_buf = (!raws.is_empty()).then(|| {
            let buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("skinned-instances"),
                size: std::mem::size_of_val(raws.as_slice()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            gpu.queue.write_buffer(&buf, 0, bytemuck::cast_slice(&raws));
            buf
        });
    }

    /// Bind the atlas, the geometry of one run, and issue its draw.
    ///
    /// Shared by the colour pass and the depth prepass because the two differ in
    /// exactly one thing — the group index the palette layout sits at (3 and 1) —
    /// and a second copy of the loop is how the two come to disagree about which
    /// instances exist (the P27.3 mirror finding, one file over).
    fn draw_runs(&self, pass: &mut wgpu::RenderPass<'_>, palette_group: u32) {
        let Some((_, atlas_bg)) = self.atlas.as_ref() else {
            return;
        };
        pass.set_bind_group(palette_group, atlas_bg, &[]);
        for run in &self.runs {
            let Some(gpu_mesh) = self
                .meshes
                .get(run.mesh)
                .and_then(|key| self.mesh_cache.get(key))
                .map(|(_, m)| m)
            else {
                continue;
            };
            if gpu_mesh.index_count == 0 {
                continue;
            }
            pass.set_vertex_buffer(0, gpu_mesh.vertices.slice(..));
            pass.set_index_buffer(gpu_mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
            let first = run.first_instance;
            pass.draw_indexed(0..gpu_mesh.index_count, 0, first..first + run.count);
        }
    }
}

fn upload_mesh(gpu: &GpuContext, mesh: &SkinnedMeshData) -> GpuSkinnedMesh {
    let vertices = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("skinned-vertices"),
        size: std::mem::size_of_val(mesh.vertices.as_slice()).max(4) as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue
        .write_buffer(&vertices, 0, bytemuck::cast_slice(&mesh.vertices));
    let indices = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("skinned-indices"),
        size: std::mem::size_of_val(mesh.indices.as_slice()).max(4) as u64,
        usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue
        .write_buffer(&indices, 0, bytemuck::cast_slice(&mesh.indices));
    GpuSkinnedMesh {
        vertices,
        indices,
        index_count: mesh.indices.len() as u32,
    }
}

impl RenderNode for SkinnedMeshNode {
    fn name(&self) -> &'static str {
        "skinned-mesh"
    }

    /// **The character in the prepass** (wave VIS1a). `sync` is keyed on
    /// `(scene.version, origin)` and is therefore free the second time it is called
    /// in one frame — so the prepass and the colour pass share one upload, and the
    /// depth this writes is the pose the colour pass will draw, not one frame's
    /// worth of skew.
    fn depth_prepass(
        &mut self,
        gpu: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        frame: &FrameData,
    ) {
        if !RenderSettings::needs_depth_prepass(frame.settings) {
            return;
        }
        self.sync(gpu, frame);
        if !self.active {
            return;
        }
        let Some(instance_buf) = self.instance_buf.as_ref() else {
            return;
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("skinned-depth"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &frame.targets.depth_prepass,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline_depth);
        pass.set_bind_group(0, frame.view_bg, &[]);
        pass.set_vertex_buffer(1, instance_buf.slice(..));
        // The atlas bind group is built against `joints_bgl`; this pipeline puts
        // that layout at slot 1 rather than 3, and the same object binds.
        self.draw_runs(&mut pass, 1);
    }

    fn run(&mut self, gpu: &GpuContext, encoder: &mut wgpu::CommandEncoder, frame: &FrameData) {
        self.sync(gpu, frame);
        if !self.active {
            return;
        }
        let Some(instance_buf) = self.instance_buf.as_ref() else {
            return;
        };
        let env_bg = self.env.bind_group(gpu, frame).clone();

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("skinned-mesh"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &frame.targets.color_msaa,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &frame.targets.depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        // Wireframe view mode (R-P2) selects the line-raster variant when present.
        let pipeline = match &self.pipeline_wire {
            Some(wire) if frame.view_mode.wireframe() => wire,
            _ => &self.pipeline,
        };
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, frame.view_bg, &[]);
        pass.set_bind_group(1, &self.lights_bg, &[]);
        pass.set_bind_group(2, &env_bg, &[]);
        pass.set_vertex_buffer(1, instance_buf.slice(..));
        self.draw_runs(&mut pass, 3);
    }
}

#[cfg(test)]
mod tests {
    use super::{plan_skinned_batches, SkinnedRun, SKINNED_PALETTE_MATRICES};
    use crate::scene::{RenderScene, SkinnedInstance, SkinnedMeshData, SkinnedShadow};
    use std::sync::Arc;

    fn palette(n: usize, tag: f32) -> Arc<Vec<glam::Mat4>> {
        Arc::new(vec![glam::Mat4::from_scale(glam::Vec3::splat(tag)); n])
    }

    fn instance(mesh: usize, palette: Arc<Vec<glam::Mat4>>) -> SkinnedInstance {
        SkinnedInstance {
            blend: 0,
            cutoff: 0.5,
            translation: glam::DVec3::ZERO,
            rotation: glam::Quat::IDENTITY,
            scale: glam::Vec3::ONE,
            color: [1.0; 4],
            metallic: 0.0,
            roughness: 1.0,
            emissive: [0.0; 3],
            id: 1,
            mesh,
            palette,
            shadow: SkinnedShadow::BindSphere,
            vt: crate::scene::VtTextureSet::NONE,
        }
    }

    fn scene(instances: Vec<SkinnedInstance>, meshes: usize) -> RenderScene {
        RenderScene {
            skinned_meshes: (0..meshes)
                .map(|_| {
                    Arc::new(SkinnedMeshData {
                        vertices: Vec::new(),
                        indices: Vec::new(),
                    })
                })
                .collect(),
            skinned: instances,
            ..Default::default()
        }
    }

    /// **The wave's headline, as arithmetic.** A crowd whose far tier shares one
    /// palette must upload ONE block for all of them, and draw the whole mesh in
    /// one call — not `N` blocks and `N` calls.
    #[test]
    fn a_shared_palette_is_one_atlas_block_and_one_draw() {
        let shared = palette(20, 1.0);
        let far: Vec<SkinnedInstance> = (0..1000).map(|_| instance(0, shared.clone())).collect();
        let plan = plan_skinned_batches(&scene(far, 1));

        assert_eq!(
            plan.runs.len(),
            1,
            "one mesh must be one draw, not a thousand"
        );
        assert_eq!(
            plan.runs[0],
            SkinnedRun {
                mesh: 0,
                first_instance: 0,
                count: 1000
            }
        );
        assert_eq!(
            (plan.blocks, plan.matrices),
            (1, 20),
            "a thousand agents sharing one `Arc` uploaded {} blocks / {} matrices \
             — the sharing is not reaching the atlas",
            plan.blocks,
            plan.matrices
        );
        // Every instance reads the same block, and none of them is at a stale
        // offset: the anti-vacuity half, because a plan that assigned (0, 0) to
        // everything would satisfy the counts above.
        assert!(plan.palette.iter().all(|b| *b == (0, 20)));
        assert_eq!(plan.order.len(), 1000);
    }

    /// The control: a thousand agents with a thousand *different* poses pay for a
    /// thousand blocks, and still draw in one call. Without this the arm above is
    /// satisfied by a plan that ignores the palette entirely.
    #[test]
    fn distinct_palettes_are_distinct_blocks_at_ascending_offsets() {
        let posed: Vec<SkinnedInstance> = (0..1000)
            .map(|i| instance(0, palette(20, i as f32 + 1.0)))
            .collect();
        let plan = plan_skinned_batches(&scene(posed, 1));

        assert_eq!(plan.runs.len(), 1);
        assert_eq!((plan.blocks, plan.matrices), (1000, 20_000));
        for (slot, (offset, live)) in plan.palette.iter().enumerate() {
            assert_eq!((*offset, *live), (slot as u32 * 20, 20));
        }
    }

    /// Two meshes are two draws, and the ordering is a **stable** sort — so a
    /// one-mesh scene (every golden in this tree) keeps the instance order it had
    /// before the atlas and draws the same pixels.
    #[test]
    fn runs_group_by_mesh_and_the_order_within_a_mesh_is_stable() {
        let p = palette(2, 1.0);
        let mixed = vec![
            instance(1, p.clone()),
            instance(0, p.clone()),
            instance(1, p.clone()),
            instance(0, p.clone()),
        ];
        let plan = plan_skinned_batches(&scene(mixed, 2));

        assert_eq!(
            plan.order,
            vec![1, 3, 0, 2],
            "the sort is not stable by mesh"
        );
        assert_eq!(
            plan.runs,
            vec![
                SkinnedRun {
                    mesh: 0,
                    first_instance: 0,
                    count: 2
                },
                SkinnedRun {
                    mesh: 1,
                    first_instance: 2,
                    count: 2
                },
            ]
        );
    }

    /// An empty palette is ONE shared identity block — the cloth/hair case, where
    /// every ribbon in a level names `vec![Mat4::IDENTITY]` or nothing at all.
    #[test]
    fn empty_palettes_share_one_identity_block() {
        let empties: Vec<SkinnedInstance> =
            (0..8).map(|_| instance(0, Arc::new(Vec::new()))).collect();
        let plan = plan_skinned_batches(&scene(empties, 1));
        assert_eq!((plan.blocks, plan.matrices), (1, 1));
        assert!(plan.palette.iter().all(|b| *b == (0, 1)));
    }

    /// **The ceiling counts what it refuses**, and the instances that fit still
    /// draw. A cap that silently offset a block to zero would draw a character in
    /// another character's pose, which is worse than not drawing it.
    #[test]
    fn the_atlas_ceiling_refuses_and_counts_rather_than_aliasing() {
        // One joint each, so the ceiling is reached at exactly its own count.
        let over = SKINNED_PALETTE_MATRICES + 7;
        let many: Vec<SkinnedInstance> = (0..over)
            .map(|i| instance(0, palette(1, i as f32 + 1.0)))
            .collect();
        let plan = plan_skinned_batches(&scene(many, 1));

        assert_eq!(plan.dropped, 7);
        assert_eq!(plan.matrices, SKINNED_PALETTE_MATRICES);
        assert_eq!(plan.order.len(), SKINNED_PALETTE_MATRICES);
        assert_eq!(
            plan.runs,
            vec![SkinnedRun {
                mesh: 0,
                first_instance: 0,
                count: SKINNED_PALETTE_MATRICES as u32
            }]
        );
        // Every surviving block is inside the atlas — the aliasing check.
        assert!(plan
            .palette
            .iter()
            .all(|(o, n)| (*o as usize + *n as usize) <= SKINNED_PALETTE_MATRICES));
    }

    /// The empty scene stays empty: no runs, no blocks, no atlas.
    #[test]
    fn a_scene_with_no_skinned_instances_plans_nothing() {
        let plan = plan_skinned_batches(&RenderScene::default());
        assert_eq!(plan, super::SkinnedBatches::default());
    }

    /// **THE ATLAS IS FILLED ONCE PER BLOCK, AND IT IS THE SAME ATLAS**
    /// (NPC1b audit).
    ///
    /// The wave's own loop was keyed on the *instance* under a comment saying it
    /// was keyed on the block, so the far tier's 710 shared agents re-copied one
    /// palette 710 times a frame in `SkinnedMeshNode::sync`. Two halves, and both
    /// are needed: the fill must produce **byte-identical** bytes to the naive
    /// per-instance fill (or the skip is a correctness bug, not an optimization),
    /// and it must perform exactly `plan.blocks` writes (or it is not a skip).
    #[test]
    fn the_atlas_is_filled_once_per_block_and_matches_the_per_instance_fill() {
        // A mixed scene: a shared far-tier block over two meshes, distinct posed
        // blocks, and an empty palette — the four shapes the fill has to serve.
        let shared = palette(4, 7.0);
        let mut insts: Vec<SkinnedInstance> = Vec::new();
        for i in 0..40 {
            insts.push(instance(i % 2, shared.clone()));
            insts.push(instance(i % 2, palette(3, i as f32 + 1.0)));
            insts.push(instance(i % 2, Arc::new(Vec::new())));
        }
        let sc = scene(insts, 2);
        let plan = plan_skinned_batches(&sc);
        assert!(plan.blocks < plan.order.len(), "nothing is shared here");

        let mut fast = Vec::new();
        let written = super::fill_palette_atlas(&mut fast, &sc, &plan);
        assert_eq!(
            written, plan.blocks,
            "the fill wrote {written} blocks of {} — it is still keyed on the \
             instance",
            plan.blocks
        );

        // The naive fill the wave shipped, spelled out here so the equality is
        // against a thing and not against itself.
        let mut naive = vec![glam::Mat4::IDENTITY.to_cols_array(); plan.matrices.max(1)];
        for (slot, i) in plan.order.iter().enumerate() {
            let (offset, live) = plan.palette[slot];
            for (j, m) in sc.skinned[*i as usize]
                .palette
                .iter()
                .take(live as usize)
                .enumerate()
            {
                naive[offset as usize + j] = m.to_cols_array();
            }
        }
        assert_eq!(fast, naive, "the once-per-block fill is not the same atlas");
    }

    /// **The invariant the skip rests on** (NPC1b audit): a block's offset is the
    /// running matrix watermark at the moment it is created, so the entry at or
    /// past the watermark IS that block's first instance and everything below it
    /// is a repeat.
    ///
    /// Stated as an arm because `fill_palette_atlas` is exact only while it holds
    /// — a planner that packed blocks in any other order would silently stop
    /// writing some of them.
    #[test]
    fn the_block_offsets_rise_at_first_use_and_repeat_below_the_watermark() {
        let a = palette(5, 1.0);
        let b = palette(2, 2.0);
        let insts = vec![
            instance(0, a.clone()),
            instance(0, b.clone()),
            instance(0, a.clone()),
            instance(1, b.clone()),
            instance(1, palette(3, 3.0)),
            instance(1, a),
            instance(1, b),
        ];
        let sc = scene(insts, 2);
        let plan = plan_skinned_batches(&sc);

        let mut watermark = 0usize;
        let mut firsts = 0usize;
        let mut covered = vec![false; plan.matrices];
        for (offset, live) in plan.palette.iter() {
            let (o, n) = (*offset as usize, *live as usize);
            if o >= watermark {
                assert_eq!(
                    o,
                    watermark,
                    "a first use skipped {} matrices",
                    o - watermark
                );
                for c in covered.iter_mut().take(o + n).skip(o) {
                    *c = true;
                }
                watermark = o + n;
                firsts += 1;
            } else {
                assert!(o + n <= watermark, "a repeat block ran past the watermark");
            }
        }
        assert_eq!(firsts, plan.blocks, "first uses are not the block count");
        assert_eq!(watermark, plan.matrices, "the atlas has a hole in it");
        assert!(covered.iter().all(|c| *c), "a matrix is written by nobody");
    }
}
