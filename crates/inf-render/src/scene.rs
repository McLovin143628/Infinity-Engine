//! The renderer's input: a flat, engine-agnostic scene description.
//!
//! Phase 2 scope: unit-cube instances with f64 world transforms (ECS binding
//! arrives in Phase 3 — the host converts whatever it has into this). The
//! `version` counter gates GPU re-uploads: bump it on any instance change.

use std::sync::Arc;

use glam::{DVec3, Mat4, Quat, Vec3};

use crate::debug_draw::DebugDraw;
use crate::primitives::PrimMesh;

pub use inf_vgeom::{VgeomMesh, VgeomSource};

pub use inf_render_2d::{
    PrebatchedRun, RenderChunk, RenderTilemap, SpriteInstance, TextureHandle, TilemapParams,
};

/// A one-shot request to upload an RGBA8 texture into the sprite pass's GPU
/// cache, keyed by [`TextureHandle`]. The pass dedups by handle, so re-listing
/// an already-uploaded texture is a cheap no-op. Straight RGBA8 rows,
/// `width*height*4` bytes, sRGB-encoded base color.
#[derive(Debug, Clone, PartialEq)]
pub struct SpriteTextureUpload {
    pub handle: TextureHandle,
    pub width: u32,
    pub height: u32,
    pub rgba8: Vec<u8>,
}

/// Reserved instance id meaning "nothing" (ID buffer clear value).
pub const ID_NONE: u32 = 0;

/// **The per-instance virtual-texture set** (P26.3): which registered virtual
/// textures a surface samples for base colour, tangent-space normal and ORM.
///
/// # Why a `handle + 1` and not an `Option`
///
/// Because the wire form has to be a plain `u32` — these three ride in words
/// that have shipped as **zero** since P7.1 (`InstanceRaw::misc.yzw`, and
/// `VgeomInstanceGpu`'s three padding words), so an instance that names no
/// texture packs to exactly the bytes it always did and the no-texture fallback
/// is *structural* rather than a flag somebody has to remember to clear.
/// `VtTextureHandle(0)` is a perfectly good texture, so the biased encoding is
/// what makes 0 mean "nothing" without a second field.
///
/// # Why per-instance words and not a texture-set table
///
/// The alternative — one index into a per-frame table of sets — was weighed and
/// costs a fourth VT binding (a storage buffer of sets) to save eight bytes per
/// instance that were **already reserved and already uploaded**. It buys nothing
/// until a surface wants more than three maps, which is where it becomes the
/// right answer; recorded rather than taken. **No vertex-stream budget moves in
/// this batch**: `InstanceRaw` stays 176 bytes and the vgeom instance record
/// stays 176, both pinned by their existing layout arms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct VtTextureSet {
    /// Base colour, sRGB-decoded on sample. `0` = none.
    pub albedo: u32,
    /// Tangent-space normal map, never sRGB-decoded. `0` = none.
    pub normal: u32,
    /// Occlusion (R) / roughness (G) / metallic (B) — the glTF packing the
    /// importer already writes. `0` = none.
    pub orm: u32,
    /// **The detail map** (Wave T — the texture document's §3 A): a shared,
    /// tileable, high-frequency normal/roughness map blended over the three
    /// above. `0` = none, which is every instance written before Wave T.
    pub detail: u32,
    /// How many times [`detail`](Self::detail) tiles across one unit of the base
    /// uv, as unsigned **8.8 fixed point** — see [`detail_scale_q8`].
    ///
    /// Fixed point rather than an `f32` for two reasons that both matter: the
    /// GPU reads it out of 16 spare bits of an existing instance word, and this
    /// struct derives `Eq`, which an `f32` field would take away from every
    /// consumer that compares two sets (the projector's material diff among
    /// them).
    pub detail_scale_q8: u16,
    /// **World metres per texture repeat**, as unsigned **8.8 fixed point**
    /// (wave ROAD1). `0` = "the mesh's uv is already in tile units and is
    /// sampled as authored", which is every instance written before ROAD1.
    ///
    /// It rides in the spare top half of the *third* instance word — `orm` is a
    /// slot index and a slot index is 16 bits, the same free space Wave T's
    /// detail scale found in the first two. So the physical tiling rate costs no
    /// instance byte, no vertex attribute and no bind group either.
    ///
    /// Fixed point rather than an `f32` for Wave T's two reasons unchanged: the
    /// GPU reads it out of sixteen bits, and this struct derives `Eq`, which the
    /// projector's material diff needs.
    ///
    /// **It is a LENGTH, and the detail scale beside it is a RATE.** Larger here
    /// means a *coarser* surface; larger there means a *finer* detail. The two
    /// travel together and mean opposite things, which is exactly why each says
    /// so on its own line.
    pub uv_tiling_q8: u16,
}

/// **Encode a physical tiling rate as unsigned 8.8 fixed point** (wave ROAD1) —
/// **metres per texture repeat**, `0` for "sample the uv as authored".
///
/// The renderer-side twin of `inf_asset::DerivedMaterial::uv_tiling_q8`, and the
/// door a host with a bare `f32` in hand uses. Same saturation rule as
/// [`detail_scale_q8`]: a non-finite or non-positive length is "no rate", a rate
/// that rounds below one 1/256th is floored to one rather than silently becoming
/// "no rate", and a nonsense 300 m saturates instead of wrapping to a plausible
/// 44 m.
#[inline]
pub fn uv_tiling_q8(metres: f32) -> u16 {
    // NaN spelled out, for `detail_scale_q8`'s reason: every ordering comparison
    // a NaN takes part in is false, so `metres <= 0.0` alone lets one through.
    if metres.is_nan() || metres <= 0.0 {
        return 0;
    }
    let q = (metres * 256.0).round();
    if q >= 65535.0 {
        return u16::MAX;
    }
    (q as u16).max(1)
}

/// **Encode a detail tiling multiplier as unsigned 8.8 fixed point** (Wave T).
///
/// Pure integer arithmetic on the way out, so the value an author sets is the
/// value the shader reads on every target — the same reasoning that keeps
/// `inf_material::bc` free of floats, applied to the one number that travels
/// beside a texture handle. Out-of-range and non-finite inputs saturate rather
/// than wrap: `0` means "no detail", so a scale that rounded to zero from a tiny
/// positive value would silently switch the feature off, and the floor of `1`
/// below is what stops that.
#[inline]
pub fn detail_scale_q8(scale: f32) -> u16 {
    // NaN is spelled out rather than caught by a negated comparison: every
    // ordering comparison a NaN takes part in is false, so `scale <= 0.0` alone
    // would let one through into `(scale * 256.0).round() as u16`, where a NaN
    // saturates to 0 — silently switching the feature off for a value that is
    // not a number rather than saying so. (The P29 blend-space finding, one
    // crate over, is the same shape.)
    if scale.is_nan() || scale <= 0.0 {
        return 0;
    }
    let q = (scale * 256.0).round();
    if q >= 65535.0 {
        return u16::MAX;
    }
    (q as u16).max(1)
}

impl VtTextureSet {
    /// Samples nothing — the scalar per-instance attributes stand alone.
    pub const NONE: Self = Self {
        albedo: 0,
        normal: 0,
        orm: 0,
        detail: 0,
        detail_scale_q8: 0,
        uv_tiling_q8: 0,
    };

    /// Whether this instance samples any virtual texture at all.
    #[inline]
    pub fn is_none(&self) -> bool {
        *self == Self::NONE
    }

    /// **The three GPU words**, with Wave T's detail lane folded into the spare
    /// top half of the first two.
    ///
    /// A handle is bounded by the registry's texture count and an atlas slot
    /// index is 16 bits (`inf_vt::table::MAX_SLOT_INDEX`), so the top half of
    /// each of these words has been zero on every instance ever uploaded — which
    /// is why the detail channel costs no instance bytes, no vertex attribute
    /// (the skinned pipeline sits exactly on `max_vertex_attributes: 16`) and no
    /// bind group. The masks are belt-and-braces: a handle that somehow exceeded
    /// 16 bits would otherwise corrupt the *detail* slot rather than merely
    /// itself, which is the kind of failure that reads as a texture bug in a
    /// different material.
    ///
    /// **This is the GPU's question, and it is not the streamer's.** A packed
    /// word is not a slot — `word - 1` is not a texture handle once the top half
    /// carries a detail lane. Anything asking *"which textures does this surface
    /// name"* wants [`handles`](Self::handles); see the Wave-T audit note there
    /// for what the two being one function cost.
    #[inline]
    pub fn slots(&self) -> [u32; 3] {
        const M: u32 = 0xFFFF;
        [
            (self.albedo & M) | ((self.detail & M) << 16),
            (self.normal & M) | ((self.detail_scale_q8 as u32) << 16),
            // ROAD1's tiling rate takes the third word's spare half, for the
            // same reason and under the same mask: `orm` is a slot index and a
            // slot index is 16 bits, so this half has been zero on every
            // instance ever uploaded.
            (self.orm & M) | ((self.uv_tiling_q8 as u32) << 16),
        ]
    }

    /// **The texture slots this set names**, unpacked — `handle + 1` each, `0`
    /// for a slot that names nothing, and the detail map among them.
    ///
    /// The door for every consumer that wants *textures* rather than *instance
    /// words*: the streamer's analytic wants and the feedback request list both
    /// turn a slot into `VtTextureHandle(slot - 1)`, and a packed word is not a
    /// slot.
    ///
    /// **Wave T audit (the defect this exists to close).** Both of those loops
    /// read [`slots`](Self::slots). The moment a material bound a detail map,
    /// word 0 became `albedo | (detail << 16)` and word 1
    /// `normal | (scale << 16)`, so `slot - 1` named a handle far past the
    /// registry, `res.desc` answered `None`, and the surface's **albedo and
    /// normal silently stopped being requested** by both lanes — while the
    /// detail map, which is in neither word as a bare slot, was never requested
    /// at all. Everything fell back to the pinned floor and stayed at its
    /// coarsest three levels for ever. The two questions are now two functions,
    /// which is the only arrangement in which neither can be asked by mistake.
    #[inline]
    pub fn handles(&self) -> [u32; 4] {
        [self.albedo, self.normal, self.orm, self.detail]
    }
}

/// Instance ids at or above this are gizmo parts, not scene objects
/// (see `gizmo.rs`).
pub const ID_GIZMO_BASE: u32 = 0xffff_ff00;

#[derive(Debug, Clone, Copy)]
pub struct MeshInstance {
    /// World-space translation (f64 — architecture rule 3).
    pub translation: DVec3,
    pub rotation: Quat,
    pub scale: Vec3,
    /// Linear-space base color (rgba).
    pub color: [f32; 4],
    /// Metallic-roughness PBR parameters.
    pub metallic: f32,
    pub roughness: f32,
    /// Linear self-emitted color (rgb).
    pub emissive: [f32; 3],
    /// Stable pick id; `ID_NONE` is reserved, ids ≥ `ID_GIZMO_BASE` are
    /// reserved for gizmo parts.
    pub id: u32,
    /// Which built-in primitive geometry to draw (R-P1). Defaults to
    /// [`PrimMesh::Cube`], so a caller that doesn't set it — and every pre-R-P1
    /// scene — renders exactly as before.
    pub mesh: PrimMesh,
    /// Blend mode (R-P5): `0` opaque, `1` masked (alpha-test), `2` translucent
    /// (alpha-blend). Defaults to `0` so every pre-R-P5 scene renders exactly as
    /// before. Projected from the ECS `Material::blend` at the seams; drives both
    /// the bucketing partition ([`crate::passes::mesh::pack_bucketed`]) and the
    /// packed `pbr.w` the shader reads for the masked discard.
    pub blend: u8,
    /// Alpha-test threshold used when `blend == 1` (masked): fragments with base
    /// color alpha below this are discarded. Defaults to `0.5`. Packed into
    /// `pbr.z`.
    pub cutoff: f32,
    /// P26.3: the virtual textures this instance samples. [`VtTextureSet::NONE`]
    /// on every instance that names none, which is every instance before this
    /// batch — and then the fragment shader runs the arithmetic it always ran.
    pub vt: VtTextureSet,
}

impl MeshInstance {
    /// A plain lit **cube** instance (metallic 0, roughness 0.5, no emission,
    /// opaque) — the common case for tests and simple callers.
    pub fn lit(translation: DVec3, rotation: Quat, scale: Vec3, color: [f32; 4], id: u32) -> Self {
        Self {
            vt: Default::default(),
            translation,
            rotation,
            scale,
            color,
            metallic: 0.0,
            roughness: 0.5,
            emissive: [0.0; 3],
            id,
            mesh: PrimMesh::Cube,
            blend: 0,
            cutoff: 0.5,
        }
    }
}

/// A virtualized-geometry (meshlet DAG) asset referenced by one or more
/// [`VgeomInstance`]s (P13.1b, **streamed since P18.2**).
///
/// The scene carries a [`VgeomSource`] — a *lazily indexed* `.inf_vmesh`, header
/// and page directory only — rather than a decoded [`VgeomMesh`]. The renderer's
/// [`VgeomStreamer`](inf_vgeom::VgeomStreamer) pages levels of it in and out of
/// shared GPU pools against a byte budget, so a scene that references a hundred
/// meshes costs a hundred *directory parses* up front, not a hundred full
/// decodes, and VRAM tracks what the camera can actually see.
///
/// `id` is the cook-derived `.inf_vmesh` asset GUID (as a `u128`), so the host
/// keys stable content and the streamer's residency survives across frames.
///
/// [`id`]: VgeomAsset::id
#[derive(Clone)]
pub struct VgeomAsset {
    /// Stable asset id (the derived `.inf_vmesh` GUID as a `u128`).
    pub id: u128,
    /// The paged `.inf_vmesh` this asset streams from (shared; one per asset).
    pub source: Arc<VgeomSource>,
}

impl VgeomAsset {
    /// Reference an already-indexed source.
    pub fn new(id: u128, source: Arc<VgeomSource>) -> Self {
        Self { id, source }
    }

    /// Lay an in-memory [`VgeomMesh`] out as a `.inf_vmesh` image and index it —
    /// the door for tests, the editor's in-memory builds, and any host that has a
    /// decoded DAG rather than a packed asset. Identical downstream to a cooked
    /// pack: there is only one paged path.
    pub fn from_mesh(id: u128, mesh: &VgeomMesh) -> Result<Self, String> {
        Ok(Self {
            id,
            source: Arc::new(VgeomSource::from_mesh(mesh)?),
        })
    }

    /// Whole-mesh bounding sphere (local space) — read from the header, so the
    /// per-instance LOD projection never pages anything in.
    pub fn bounds(&self) -> ([f32; 3], f32) {
        self.source.bounds()
    }
}

impl std::fmt::Debug for VgeomAsset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VgeomAsset")
            .field("id", &format_args!("{:#034x}", self.id))
            .field("source", &self.source)
            .finish()
    }
}

/// One placed instance of a [`VgeomAsset`] (P13.1b) — the meshlet-path twin of a
/// [`MeshInstance`]. Multiple instances of the same `asset` share its GPU buffers;
/// the cull compute emits one visible-list entry per surviving (instance, meshlet)
/// pair. World transform is f64 (architecture rule 3); the renderer projects it to
/// an origin-relative model matrix at upload, exactly like [`MeshInstance`].
#[derive(Debug, Clone, Copy)]
pub struct VgeomInstance {
    /// Which [`VgeomAsset`] (by [`VgeomAsset::id`]) this instance draws.
    pub asset: u128,
    pub translation: DVec3,
    pub rotation: Quat,
    pub scale: Vec3,
    /// Linear-space base color (rgba).
    pub color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    /// Linear self-emitted color (rgb).
    pub emissive: [f32; 3],
    /// Stable pick id (`ID_NONE` reserved).
    pub id: u32,
    /// P26.3: the virtual textures this instance samples. [`VtTextureSet::NONE`]
    /// on every instance that names none, which is every instance before this
    /// batch — and then the fragment shader runs the arithmetic it always ran.
    pub vt: VtTextureSet,
}

impl VgeomInstance {
    /// A plain lit instance of `asset` (metallic 0, roughness 0.5, no emission).
    pub fn lit(
        asset: u128,
        translation: DVec3,
        rotation: Quat,
        scale: Vec3,
        color: [f32; 4],
        id: u32,
    ) -> Self {
        Self {
            vt: Default::default(),
            asset,
            translation,
            rotation,
            scale,
            color,
            metallic: 0.0,
            roughness: 0.5,
            emissive: [0.0; 3],
            id,
        }
    }
}

// ── GPU-instanced scatter (P18.5) ────────────────────────────────────────────

/// One scattered instance as a **host** authors it: world position, orientation,
/// uniform scale, tint (P18.5).
///
/// This is the shape both projectors already had in hand — `PcgVolume::evaluated`
/// and `Foliage::instances` — and it is *not* what reaches the GPU. Packing into
/// [`ScatterInstanceRaw`] happens once, in [`ScatterData::build`], which is Ring 0
/// so the editor viewport and the shipped player cannot pack differently.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScatterInstance {
    /// World-space position (f64 — architecture rule 3).
    pub position: DVec3,
    pub rotation: Quat,
    /// Per-axis scale in the instance's **own rotated frame**.
    ///
    /// Uniform for every caller that predates IB-2b (`Vec3::splat`, and
    /// [`ScatterInstance::uniform`] spells it). It is a vector because a
    /// building's far-LOD **shell** is an oriented *box* with three different
    /// half-extents, and a shell drawn as a cube of the wrong proportions is not
    /// a level of detail — it is a different building. The same argument as
    /// "a shell that is not a barrier is a hole, not a LOD", one pass over.
    pub scale: Vec3,
    /// Linear-space tint (rgba).
    pub color: [f32; 4],
}

impl ScatterInstance {
    /// An instance scaled the same on every axis — what every scatter that is
    /// not a structure shell wants.
    pub fn uniform(position: DVec3, rotation: Quat, scale: f32, color: [f32; 4]) -> Self {
        Self {
            position,
            rotation,
            scale: Vec3::splat(scale),
            color,
        }
    }
}

/// One scattered instance as the **GPU** stores it (P18.5) — 64 bytes, `Pod`, and
/// deliberately **origin-independent**.
///
/// # Why 64 and not 48 (IB-2b)
///
/// The first fourty-eight bytes are **byte-identical** to the P18.5 record, on
/// purpose: `offset`, `scale`, `rotation` and `color` sit exactly where the two
/// shaders already read them, so the only shader line the growth touches is the
/// one that builds the local vertex position. The eight bytes of padding are the
/// price of keeping `scale_yz` in its own 16-byte slot rather than re-cutting a
/// hot record two shaders and four passes read.
///
/// The cost is 33% of an instance buffer — 6.8 MB on the thousand-building city
/// fixture's 427 351 instances — bought for the ability to draw an oriented
/// **box**, which is what a structure's far-LOD shell is.
///
/// `offset` is relative to the batch's [`ScatterBatch::anchor`], not to the
/// floating origin. That is the load-bearing part: a render-local pack would be
/// invalidated by every origin rebase, so a camera flying across the world would
/// re-upload every instance buffer it can see. Anchor-relative offsets are a pure
/// function of the *content*, so the buffer is uploaded once per content change
/// and the anchor rides in a per-frame uniform instead.
///
/// Precision: f32 relative to the batch anchor, so a batch spanning 1 km resolves
/// to ~6e-5 m and one spanning 100 km to ~6e-3 m. Scatter volumes are authored at
/// tens-to-hundreds of metres (`PcgVolume::extent` defaults to 50 m), so this is
/// several orders of margin; a single batch covering a continent would not be.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ScatterInstanceRaw {
    /// Position relative to the batch anchor (metres).
    pub offset: [f32; 3],
    /// Scale along the instance's own **X**. Named `scale` and left in place
    /// because both shaders already read it here and a uniform instance sets all
    /// three to the same number.
    pub scale: f32,
    /// Orientation quaternion, `xyzw`.
    pub rotation: [f32; 4],
    /// Linear-space tint (rgba).
    pub color: [f32; 4],
    /// Scale along the instance's own **Y** and **Z**.
    pub scale_yz: [f32; 2],
    /// Padding to a 16-byte slot — WGSL storage layout, not spare capacity.
    pub _pad: [f32; 2],
}

/// The immutable, content-addressed payload of a [`ScatterBatch`] (P18.5).
///
/// **Content-keyed, like everything else since P18.3.** [`key`](Self::key) is an
/// `xxh3` 128-bit hash over the packed instance bytes *and* the primitive kind, so
/// two batches with identical geometry share one GPU upload (two foliage entities
/// painted from the same stroke, or the editor and the player rendering the same
/// level) and a changed batch is a *different* asset rather than a stale one under
/// a reused id. The renderer's `GenCache` keys on it directly; nothing hashes per
/// frame.
///
/// The hash is folded **while packing**, in one pass over data the projector was
/// building anyway — and the path it replaces built one 96-byte `MeshInstance` per
/// scattered instance and pushed it into `RenderScene::instances`, so this is
/// strictly cheaper than what it supersedes.
#[derive(Debug, Clone, PartialEq)]
pub struct ScatterData {
    /// Which built-in primitive every instance of this batch draws — and, when
    /// [`geometry`](Self::geometry) is `Some`, which primitive stands in for it
    /// in the paths that cannot yet draw a real mesh (the impostor card, the CPU
    /// fallback, the shadow caster pack). See [`ScatterGeometry`].
    pub mesh: PrimMesh,
    /// **The authored mesh every instance of this batch draws** (wave TER2b), or
    /// `None` for the built-in primitive named by [`mesh`](Self::mesh).
    pub geometry: Option<Arc<ScatterGeometry>>,
    /// Packed, anchor-relative instance records in authored order.
    pub instances: Vec<ScatterInstanceRaw>,
    key: u128,
    max_scale: f32,
}

/// Real triangle geometry for a scatter batch (wave TER2b) — an authored
/// `.inf_mesh` in the layout `scatter_mesh.wgsl` already pulls from.
///
/// # Why this is a payload and not a `PrimMesh`
///
/// The scatter raster does **not** bind a vertex buffer. It reads
/// `vertices[idx * 6u + k]` and `indices[...]` out of two storage buffers and
/// takes its draw count from an indirect args block the cull compute wrote — the
/// P18.5 arrangement. Everything a mesh needs to reach that path is therefore the
/// same two flat arrays [`PrimStorage`](crate::primitives::PrimStorage) already
/// holds for the five built-ins, which is why plumbing an authored mesh through
/// costs a payload and a bind group and not a second pipeline.
///
/// **Stride six: position then normal, and no uv.** That is
/// [`SCATTER_PULL_STRIDE`](crate::primitives::SCATTER_PULL_STRIDE), and it is the
/// reason a scattered instance is tinted rather than textured — there is no uv in
/// the buffer for a material to be sampled at. Widening it is the named follow-up
/// (see the module note in `scatter_mesh.wgsl`).
///
/// # Content-addressed, like everything else the scatter path uploads
///
/// [`key`](Self::key) is an `xxh3` over the two arrays, and `ScatterData::key`
/// folds it in — so two batches of the same placements drawn as *different*
/// meshes are two different payloads, and the same mesh scattered in two places
/// is one geometry upload.
#[derive(Debug, Clone, PartialEq)]
pub struct ScatterGeometry {
    /// `position` then `normal` per vertex, flat — `SCATTER_PULL_STRIDE` floats
    /// each, in the authored vertex order.
    pub vertices: Vec<f32>,
    /// Triangle indices into [`vertices`](Self::vertices), by vertex (not float).
    pub indices: Vec<u32>,
    /// The mesh's bounding-sphere radius about its own origin, at unit scale —
    /// the cull radius, and `PrimMesh::bounding_radius`'s counterpart.
    pub radius: f32,
    key: u128,
}

/// The most triangles one scattered mesh may carry.
///
/// A ceiling, not a quality knob — the same shape as
/// [`MAX_CPU_SCATTER_INSTANCES`](crate::MAX_CPU_SCATTER_INSTANCES). Scatter draws
/// its geometry **per instance** with no meshlet LOD behind it, so the cost of a
/// kind is its triangle count times its live population: the island's 16 771
/// instances at this ceiling would be **68.7 M** triangles before culling,
/// against the 10 M-triangle scene P13 gated at 2.4 % cull. The three
/// ground-cover props are 32, 20 and 128 triangles, so this is three orders of
/// magnitude of headroom for the content this path is *for*, and a refusal for
/// content that wants the virtualized path instead.
///
/// Here rather than in either host **because both hosts must refuse the same
/// content**: a mesh the editor draws and a shipped build does not is the exact
/// divergence the projector-mirror gate exists to prevent, one level up.
/// A refused mesh falls back to the placeholder primitive, inertly and with a
/// warning, exactly as an unresolvable GUID does.
pub const MAX_SCATTER_MESH_TRIANGLES: usize = 4_096;

/// The authored meshes a scatter projection may draw, by mesh-asset GUID
/// (`Uuid::as_u128` — this crate keys everything content-addressed on `u128` and
/// deliberately does not name `uuid` in a projector-facing signature).
///
/// **Populated by the host, read by the projector.** A projection runs per frame
/// in the shipped player, so it must not open files; each host fills this from
/// its own store — the player's pack at level load, the editor's content root on
/// a document change — and hands the projector a finished table. A GUID that is
/// absent is a mesh the host could not resolve, and the projector draws the
/// placeholder primitive for it, which is exactly what it did before there were
/// meshes at all.
pub type ScatterMeshes = std::collections::HashMap<u128, Arc<ScatterGeometry>>;

impl ScatterGeometry {
    /// Build from the streams `inf_mesh::MeshAsset::vgeom_streams` produces.
    ///
    /// **The one door.** Both hosts resolve a scatter kind's mesh GUID to a
    /// `MeshAsset` through their own asset store — the editor's loose content
    /// root, the player's pack — and both arrive here, so the flattening, the
    /// radius and the content key are one piece of code rather than a pair that
    /// agree by inspection. `positions` and `normals` must be the same length; a
    /// shorter `normals` list pads with `+Y`, which is what an unnormalled import
    /// already means everywhere else in this engine.
    pub fn from_streams(positions: &[[f32; 3]], normals: &[[f32; 3]], indices: &[u32]) -> Self {
        let mut vertices =
            Vec::with_capacity(positions.len() * crate::primitives::SCATTER_PULL_STRIDE);
        let mut r2: f32 = 0.0;
        for (i, p) in positions.iter().enumerate() {
            vertices.extend_from_slice(p);
            vertices.extend_from_slice(normals.get(i).unwrap_or(&[0.0, 1.0, 0.0]));
            r2 = r2.max(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
        }
        // Indices past the end would read garbage out of a storage buffer, which
        // is a hang or a spike of noise rather than an error. Dropping the
        // offending triangle is the same refusal the mesh reader makes.
        let n = positions.len() as u32;
        let indices: Vec<u32> = indices
            .chunks_exact(3)
            .filter(|t| t.iter().all(|i| *i < n))
            .flatten()
            .copied()
            .collect();
        let mut bytes = Vec::with_capacity(vertices.len() * 4 + indices.len() * 4);
        bytes.extend_from_slice(bytemuck::cast_slice(&vertices));
        bytes.extend_from_slice(bytemuck::cast_slice(&indices));
        let key = xxhash_rust::xxh3::xxh3_128(&bytes);
        Self {
            vertices,
            indices,
            radius: r2.sqrt(),
            key,
        }
    }

    /// The content key — the renderer's GPU-cache identity for this geometry.
    pub fn key(&self) -> u128 {
        self.key
    }

    /// Triangles in the mesh.
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Vertices in the mesh.
    pub fn vertex_count(&self) -> usize {
        self.vertices.len() / crate::primitives::SCATTER_PULL_STRIDE
    }

    /// Nothing to draw — an empty mesh, or one whose every triangle was refused.
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty() || self.vertices.is_empty()
    }
}

impl ScatterData {
    /// Pack world-space instances into an anchor-relative GPU payload and derive
    /// the content key.
    ///
    /// Deterministic: the output is a pure function of `(mesh, anchor, instances)`
    /// in authored order, with no floating-origin, camera or frame input.
    pub fn build(
        mesh: PrimMesh,
        anchor: DVec3,
        instances: impl IntoIterator<Item = ScatterInstance>,
    ) -> Self {
        Self::build_with_geometry(mesh, None, anchor, instances)
    }

    /// [`build`](Self::build), for a batch that draws an **authored mesh** (wave
    /// TER2b).
    ///
    /// `mesh` stays, and is the *proxy*: the impostor card's shape, the CPU
    /// fallback's geometry and the shadow caster's are all still the built-in
    /// primitive, because those three paths bind one shared vertex buffer for the
    /// whole frame and a per-batch mesh does not fit in it. What changes is the
    /// full-mesh raster, which pulls from a storage buffer and therefore can be
    /// handed any geometry at all.
    pub fn build_with_geometry(
        mesh: PrimMesh,
        geometry: Option<Arc<ScatterGeometry>>,
        anchor: DVec3,
        instances: impl IntoIterator<Item = ScatterInstance>,
    ) -> Self {
        let mut packed = Vec::new();
        let mut max_scale: f32 = 0.0;
        for i in instances {
            let o = i.position - anchor;
            // The cull radius is one conservative scalar for the whole batch, so
            // a non-uniform instance contributes its LARGEST axis — an
            // over-approximation, which is all the subtractive cull proof needs.
            max_scale = max_scale
                .max(i.scale.x.abs())
                .max(i.scale.y.abs())
                .max(i.scale.z.abs());
            packed.push(ScatterInstanceRaw {
                offset: [o.x as f32, o.y as f32, o.z as f32],
                scale: i.scale.x,
                rotation: i.rotation.to_array(),
                color: i.color,
                scale_yz: [i.scale.y, i.scale.z],
                _pad: [0.0; 2],
            });
        }
        // The primitive kind is part of the identity: the same placements drawn as
        // cubes and as spheres are two different batches, and an id that did not
        // say so would serve one from the other's cached buffers.
        let mut bytes =
            Vec::with_capacity(packed.len() * std::mem::size_of::<ScatterInstanceRaw>() + 20);
        bytes.extend_from_slice(&(mesh as u32).to_le_bytes());
        // …and so is the authored geometry, on exactly the same argument: the
        // same placements drawn as a grass tuft and as a rock are two batches.
        // A `None` folds nothing, so every batch that predates wave TER2b keys
        // byte-identically to what it always did.
        if let Some(g) = &geometry {
            bytes.extend_from_slice(&g.key().to_le_bytes());
        }
        bytes.extend_from_slice(bytemuck::cast_slice(&packed));
        let key = xxhash_rust::xxh3::xxh3_128(&bytes);
        Self {
            mesh,
            geometry,
            instances: packed,
            key,
            max_scale,
        }
    }

    /// The batch's cull radius at unit scale: the authored mesh's own bounding
    /// sphere when there is one, and the proxy primitive's otherwise.
    ///
    /// A cull radius that is too *small* deletes instances at the frustum edge,
    /// so this is the one place the proxy must not be used for a real mesh — a
    /// 0.74 m shrub inside a `√3/2` unit cube would be culled correctly by
    /// accident, and a 3 m one would not.
    pub fn bounding_radius(&self) -> f32 {
        self.geometry
            .as_ref()
            .map(|g| g.radius)
            .unwrap_or_else(|| self.mesh.bounding_radius())
    }

    /// The content key — the renderer's GPU-cache identity for this payload.
    pub fn key(&self) -> u128 {
        self.key
    }

    /// The largest uniform scale in the batch. Multiplied by the primitive's own
    /// bounding radius it gives one conservative cull radius for every instance,
    /// which is what lets the cull compute carry a single scalar instead of a
    /// per-instance one.
    pub fn max_scale(&self) -> f32 {
        self.max_scale
    }

    pub fn len(&self) -> usize {
        self.instances.len()
    }

    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }
}

/// A GPU-scattered instance batch (P18.5) — the render-side form of a
/// `PcgVolume`'s evaluated cache or a `Foliage` component's placed instances.
///
/// Before this batch both projectors expanded those lists into one
/// [`MeshInstance`] each and pushed them onto [`RenderScene::instances`], so a
/// 100k-instance scatter cost 100k CPU-side structs, 100k packed
/// `InstanceRaw`s and a per-frame vertex-buffer upload of ~17 MB. Now the payload
/// is uploaded **once per content change** into a storage buffer and culled
/// per-instance on the GPU (frustum + the P18.1 HZB), with distance bands
/// selecting full mesh / impostor / nothing.
///
/// The batch is one *object* as far as selection is concerned — a scatter is
/// authored, moved and deleted as a whole — so it carries one pick [`id`](ScatterBatch::id)
/// rather than one per instance.
/// Metres beyond which a grammar structure draws as its **shell** instead of its
/// parts (IB-2b) — the draw-side twin of `inf_ecs::DEFAULT_COLLIDER_NEAR_M`, and
/// deliberately a *different* number.
///
/// # The three constraints, and why exactly one number satisfies all three
///
/// * **Strictly greater than the collider band** (`DEFAULT_COLLIDER_NEAR_M`,
///   64 m — what a 4.0 ms fixed step affords). That is the invariant that keeps
///   "what I can touch" and "what I can see" the same building: every structure
///   a body can collide with is drawn as its parts.
/// * **At or below the scatter path's own mesh band**
///   (`ScatterSettings::mesh_distance_m`, 120 m). A shell batch is banded
///   `[STRUCTURE_LOD_M, draw_distance)` and `effective_bands` ends the
///   full-mesh band at `min(mesh_distance_m, cull)`, so a swap distance past
///   that band means **a shell is never rasterized as geometry at any
///   distance** — it is a billboard from the moment it exists — while the parts
///   it replaces go on drawing as hundreds of billboards each through the whole
///   annulus in between.
/// * **The same number `inf_pcg::building::DEFAULT_STRUCTURE_LOD_M` already
///   is** (96 m), because the sim's band and the draw's band naming different
///   distances is how a building comes to draw as a shell it can still be walked
///   into.
///
/// 96 m is the only value in the tree that satisfies all three, and island wave
/// I8c is where the second constraint was priced: at 192 m the island's
/// settlements drew their parts as impostor cards through the whole 120 m–230 m
/// annulus and their shells as one card past 192. Re-ordering them moved the
/// island's lit `scatter` pass **7.013 → 5.135 ms**, reproduced at 5.148 in a
/// second run. The pop the change exposes is measured in
/// `tests/structure_lod_pop.rs`, which reads this constant as its own camera
/// distance and re-aims with it — the two carried impostor bounds move to
/// `2 × STRUCTURE_LOD_M`, which is the same 192 m they were always measured at,
/// and re-read to the same figures to the pixel.
///
/// # Why it lives here
///
/// In `inf-render`, because **both hosts' projections need it and neither can
/// see the other's crate**: the shipped player links `inf-pcg` and the editor
/// viewport does not (it hand-mirrors `pcg_kind_color` for exactly that reason),
/// so a constant in `inf-pcg` would have to be duplicated.
///
/// *(Wave CERT1 correction: the viewport HAS linked `inf-pcg` since island wave
/// I8b, for the module-mesh table. The sentence above is kept because the
/// conclusion still holds — one constant in the crate both projections already
/// name is better than two — but the premise is no longer true.)*
pub const STRUCTURE_LOD_M: f64 = 96.0;

/// **How far a grammar building draws its FIT-OUT**, metres — the mid rung of
/// the three-band ladder (wave CERT1, CP-C3).
///
/// # The band it makes
///
/// A building used to draw in two tiers: everything it holds out to
/// [`STRUCTURE_LOD_M`], then one shell box. The certification's owner asked for
/// three — *"low poly far / medium mid-range / high poly close"* — and the
/// third rung is here:
///
/// | band | metres | what draws |
/// |---|---|---|
/// | near | `[0, INTERIOR_LOD_M)` | the whole building, fit-out included |
/// | mid | `[0, STRUCTURE_LOD_M + reach)` | its fabric only — walls, glazing, decks, signs, string lights |
/// | far | `[STRUCTURE_LOD_M, draw)` | one oriented shell box |
///
/// The bands are complementary per BUCKET rather than per instance, which is
/// what keeps the camera out of the projection: `push_scatter` already groups a
/// volume's instances by the mesh GUID they draw, and a fit-out family's GUID is
/// a fact about the content. Nothing is added to the instance stream, no schema
/// moves, and a level with no grammar buildings in it changes not one batch.
///
/// # Why 64 m
///
/// It is `inf_ecs::band::DEFAULT_COLLIDER_NEAR_M` — the radius inside which a
/// building has colliders at all. Past it you cannot be *inside* the building,
/// so its furniture can only be seen through a door opening from outside; and
/// the window between you and it is an opaque emissive box (CP-B10), so there is
/// no second way in. Tying the two by value rather than by coincidence means the
/// day the collider band moves, the question "can I be in this room" and the
/// question "is this room furnished" still have one answer.
///
/// `inf-render` cannot name `inf-ecs` (Ring 0 render has no ECS edge), so the
/// equality is asserted by the projector's own gate rather than by a `use`.
pub const INTERIOR_LOD_M: f64 = 64.0;

#[derive(Debug, Clone, PartialEq)]
pub struct ScatterBatch {
    /// The content-addressed payload. `Arc` so a projection that re-runs without
    /// the scatter changing costs a pointer copy.
    pub data: Arc<ScatterData>,
    /// World-space anchor the payload's offsets are relative to. **Not** part of
    /// the content key: a batch whose anchor moves (an interpolated actor) keeps
    /// its buffer and only its uniform changes.
    pub anchor: DVec3,
    pub metallic: f32,
    pub roughness: f32,
    /// Linear self-emitted color (rgb).
    pub emissive: [f32; 3],
    /// Stable pick id for the whole batch (`ID_NONE` reserved).
    pub id: u32,
    /// Authored draw distance in metres; `0` ⇒ unlimited. Clamps the renderer's
    /// own impostor/cull band **down**, never up — content may ask for less detail
    /// than the tier allows, never for more.
    ///
    /// This is the one *content* LOD knob (`PcgVolume::draw_distance`, which has
    /// existed since P10.5). Honouring it inside the cull compute is what finally
    /// makes both hosts agree about it: the editor used to cull against its own
    /// camera eye on the CPU while the player ignored the field entirely, so a
    /// shipped build drew strictly more scatter than its preview.
    pub draw_distance: f64,
    /// **Inner** draw distance in metres; `0` ⇒ no inner cut (IB-2b).
    ///
    /// An instance closer to the eye than this is culled, so a batch occupies the
    /// half-open interval `[near_distance, draw_distance)` instead of
    /// `[0, draw_distance)`.
    ///
    /// # Why the engine needed this
    ///
    /// A level of detail is two batches whose bands are **complementary**: a
    /// building's parts inside the structure-LOD distance, its shell outside it.
    /// Without an inner cut the only way to express that is to decide, on the
    /// CPU, which instances to pack — which puts the camera inside the batch's
    /// *content*, and a content key that moves with the camera re-uploads a
    /// city's instance buffer every time the player walks. The inner cut keeps
    /// both batches a pure function of the content and lets the cull compute —
    /// which has already computed the distance — do the selection per instance,
    /// per frame, for free.
    ///
    /// Clamped against `draw_distance` by nothing: a batch whose inner cut
    /// exceeds its outer one draws nothing, which is the honest reading of an
    /// empty interval and is what an author who inverted the two asked for.
    pub near_distance: f64,
    /// **Whether this batch's instances are shadow casters** (island wave I8b).
    /// `true` for everything that predates the field.
    ///
    /// # What it is for, and it is not an author's knob
    ///
    /// The CPU caster pack (`passes::scatter::pack_fallback`, which both the
    /// cascade and the virtual shadow map draw their scatter casters from)
    /// walks **every instance of every batch** to distance-test it, then keeps
    /// at most `VSM_MAX_CASTERS` of them. On the island's settlements that was
    /// 365 545 instances walked, 16 384 kept and 49 000 thrown away *per frame*
    /// — 16.7 ms of CPU, and 70 % of the whole record stage.
    ///
    /// The content already had the answer. A building's parts stand inside its
    /// own **shell**, which is in the same scene and casts already, so 1 500
    /// interior boxes contribute nothing to a silhouette their own bounding box
    /// draws — and being packed at all is what pushed a *forest* out of the
    /// caster budget. A batch that another batch's geometry contains says so
    /// here, and `pack_fallback` skips it in `O(1)` rather than in
    /// `O(instances)`.
    ///
    /// It is not "cast no shadow": the shell casts. Setting it on a batch that
    /// nothing contains would be, and that is the one thing this field must not
    /// be used for.
    ///
    /// # It is read by the CASTER pack alone (island wave I8b audit)
    ///
    /// `pack_fallback` has two consumers, and the other one is the **visible**
    /// CPU raster — the picture on every tier below `RenderTier::High`, where
    /// `ScatterSettings::gpu` is off. The first cut read this field in the
    /// shared body, so the settlements stopped being *drawn* on a Medium
    /// machine as well as ceasing to cast. The purpose is an argument now
    /// (`passes::scatter::PackPurpose`), and a batch that opts out of casting
    /// still draws.
    pub casts_shadows: bool,
}

impl ScatterBatch {
    /// A plain lit batch (metallic 0, no emission, unlimited draw distance).
    pub fn lit(data: Arc<ScatterData>, anchor: DVec3, roughness: f32, id: u32) -> Self {
        Self {
            data,
            anchor,
            metallic: 0.0,
            roughness,
            emissive: [0.0; 3],
            id,
            draw_distance: 0.0,
            near_distance: 0.0,
            casts_shadows: true,
        }
    }
}

/// **The identity of the thing a [`ScatterBatch`] was packed out of** — the key
/// the sim→render projection's scatter carry-forward reads (island wave I8a
/// audit).
///
/// # Why the projection needed one
///
/// `RenderTerrainTile::version` and `ScatterData::key` both gate the GPU
/// *upload*, and Hardening Wave E's memo already says what that is not enough
/// for: *"what no consumer could do is stop the payload from being built."* The
/// scatter path was the one large payload left rebuilding every frame — a
/// settled city block's instances re-packed to f32 and re-hashed sixty times a
/// second for content that changes only when a cell activates. Measured on the
/// island at 172 settlement blocks: **365 545 instances, ~20.2 ms a projection
/// against a 1.5 ms budget.**
///
/// # What is in the key, and why each half is
///
/// * `entity` + `stamp` are the volume's `Guid` and its
///   `PcgVolume::structures_gen`. The stamp is drawn from a **process-global**
///   monotone counter, so `(entity, stamp)` names one population of one volume
///   for the life of the process — including across the destroy/rebuild a cell
///   deactivation and reactivation performs under the same guid. `0` in either
///   field means "not in the ledger" and is a forced miss, exactly as `0` is for
///   the terrain and voxel stamps.
/// * `draw_distance_bits` is `PcgVolume::draw_distance`'s bit pattern. It is
///   **authored**, so the population stamp does not cover it, and it decides
///   both of a volume's LOD bands: an author dragging it in the Details panel
///   must re-pack.
/// * `table` is [`scatter_table_stamp`] over the host's resolved mesh table,
///   which decides which bucket each instance packs into and therefore how many
///   batches a volume becomes.
///
/// The world anchor is part of the key: the packed offsets are relative to it,
/// so a carried batch whose anchor moved would translate its whole population.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ScatterSource {
    /// The projecting entity's `Guid` as `u128`; `0` ⇒ not memoizable.
    pub entity: u128,
    /// The source's process-global population stamp; `0` ⇒ forced miss.
    pub stamp: u64,
    /// The authored draw distance's bit pattern (not covered by `stamp`).
    pub draw_distance_bits: u64,
    /// [`scatter_table_stamp`] of the mesh table the batch was bucketed against.
    pub table: u64,
    /// **The quantized night-glow step the emissive was written at** (island
    /// wave I8b), from [`night_glow_step`].
    ///
    /// A batch's emission is `glow x gain`, and the gain moves with the sun. It
    /// is not part of `ScatterData` -- the uploaded instance bytes are unmoved
    /// by it -- but it IS part of the `ScatterBatch` record, so a carried batch
    /// would keep the emission of the hour it was packed in. Keying on the
    /// quantized step is what makes dusk a re-pack instead of a city whose
    /// windows are lit at noon.
    pub glow_step: u16,
    /// **The quantized level-clock tick a pulsing emitter's gain was written
    /// at** (wave VEN1a), from [`pulse_tick`] -- or `0` for a volume that holds
    /// no pulsing module, which is every volume of every level before the venue
    /// archetypes.
    ///
    /// Exactly [`glow_step`](Self::glow_step)'s argument, one clock over: the
    /// pulse gain is not part of `ScatterData` (the uploaded bytes are unmoved
    /// by it) but it IS part of the `ScatterBatch` record, so a carried batch
    /// would keep the phase of the frame it was packed in and a club's string
    /// lights would freeze mid-breath the moment the volume stopped changing.
    ///
    /// **The zero is what keeps it cheap.** A key that always carried the tick
    /// would re-pack all 172 of the island's settlement volumes -- 365 545
    /// instances -- eight times a second for a festoon in one of them.
    pub pulse_tick: u32,
    /// The world anchor the payload's offsets were packed against.
    pub anchor: DVec3,
}

impl ScatterSource {
    /// A source nothing may carry forward — painted foliage and a terrain's
    /// biome population.
    ///
    /// Those are not memo-hostile by nature; they simply have no monotone stamp
    /// today, and a memo with no stamp behind it is the stale hit this type
    /// exists to make unreachable.
    pub const NONE: Self = Self {
        entity: 0,
        stamp: 0,
        draw_distance_bits: 0,
        table: 0,
        glow_step: 0,
        pulse_tick: 0,
        anchor: DVec3::ZERO,
    };

    /// Is this a key a carry-forward may match on at all?
    pub fn is_memoizable(&self) -> bool {
        self.entity != 0 && self.stamp != 0
    }
}

/// A stamp over the host's resolved scatter-mesh table.
///
/// Order-independent (the table is a `HashMap`), and sensitive to a geometry
/// being *replaced* under an existing GUID as well as to one being added or
/// removed — each entry folds its own map key together with its geometry's
/// content key, so a swap between two GUIDs moves the stamp too.
///
/// It is a 64-bit digest rather than an identity, which is the same argument
/// `ScatterData::key` already makes one word wider; the table is a handful of
/// entries and is rebuilt at most once per projection, so this is O(table) once
/// a frame and never per volume.
pub fn scatter_table_stamp(meshes: &ScatterMeshes) -> u64 {
    let mut acc = meshes.len() as u64;
    for (k, g) in meshes {
        let key = g.key();
        let e = (*k as u64)
            ^ ((*k >> 64) as u64).rotate_left(17)
            ^ (key as u64).rotate_left(31)
            ^ ((key >> 64) as u64).rotate_left(43);
        acc ^= e.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    }
    acc
}

/// How many steps the night-glow gain is quantized to (island wave I8b).
///
/// The gain is a continuous function of the sun's height and it decides a
/// `ScatterBatch`'s emission, which is part of the scatter memo's key. A
/// continuous key would re-pack every settlement volume on every frame of dusk;
/// sixteen steps make it sixteen re-packs across a whole sunset, each costing
/// what one cell activation costs. The alternative -- an emission the carried
/// batch keeps -- is a city whose windows are lit at noon.
pub const NIGHT_GLOW_STEPS: u16 = 16;

/// Where the ramp begins and ends, as the sun direction's own `y`.
///
/// Windows come on while the sun is still a little above the horizon (people
/// switch lights on at dusk, not at astronomical night) and are fully lit a
/// little below it. Both ends are the sun's height rather than a clock, so a
/// polar summer simply never lights the city, which is correct.
const GLOW_SUN_DAY_Y: f32 = 0.10;
const GLOW_SUN_NIGHT_Y: f32 = -0.08;

/// The **quantized** night-glow step for a sun direction, `0` (full day) to
/// [`NIGHT_GLOW_STEPS`] (full night).
///
/// One door, called by both projectors, so the memo key and the emission it
/// stands for cannot be computed two ways. A non-finite direction reads as day,
/// which is the answer that changes nothing.
pub fn night_glow_step(sun_dir: Vec3) -> u16 {
    let y = sun_dir.y;
    if !y.is_finite() {
        return 0;
    }
    let t = ((GLOW_SUN_DAY_Y - y) / (GLOW_SUN_DAY_Y - GLOW_SUN_NIGHT_Y)).clamp(0.0, 1.0);
    // The same `t*t*(3-2t)` ease every ramp in this engine uses, so the lights
    // come up rather than switching.
    let eased = t * t * (3.0 - 2.0 * t);
    (eased * NIGHT_GLOW_STEPS as f32)
        .round()
        .clamp(0.0, NIGHT_GLOW_STEPS as f32) as u16
}

/// The linear emission a glowing scatter instance contributes at `step`.
///
/// `glow` is the instance's authored multiplier (`0.0` for everything that does
/// not glow); the tint is a warm interior white, because a lit window is warm
/// whatever colour the wall it sits in is tinted. Returns exactly `[0, 0, 0]`
/// for a step of zero, so a daytime batch is byte-identical to the pre-I8b one.
pub fn glow_emissive(glow: f32, step: u16) -> [f32; 3] {
    // **`!is_finite() || <= 0.0`, and not `!(glow > 0.0)`** — the island wave
    // I8a clippy finding, one crate over. A negated comparison on a partially
    // ordered type is the readable-looking spelling; the obvious rewrite
    // `glow <= 0.0` is FALSE for a NaN, and a NaN glow would multiply straight
    // through into an emissive a batch carries for ever.
    if !glow.is_finite() || glow <= 0.0 || step == 0 {
        return [0.0; 3];
    }
    let g = glow * (step as f32 / NIGHT_GLOW_STEPS as f32);
    [g, g * 0.86, g * 0.62]
}

/// **How many times a second a pulsing emitter's gain is re-derived** (wave
/// VEN1a).
///
/// The gain is a continuous function of the level clock and it decides a
/// `ScatterBatch`'s emission, which is part of the scatter memo's key — exactly
/// the shape [`NIGHT_GLOW_STEPS`] already has, for exactly its reason. A
/// continuous key would re-pack every volume that holds a pulsing module on
/// every frame; eight ticks a second make it eight, and at the slowest rate a
/// venue authors (0.27 Hz, a filament's breath) that is thirty samples a cycle,
/// which reads as a breath and not as a staircase.
///
/// The projector writes `0` for a volume that holds **no** pulsing module,
/// which is every volume of every level that predates the venue archetypes — so
/// the tick joins the memo key without making a settlement re-pack for a club
/// three blocks away.
pub const PULSE_TICKS_PER_S: f64 = 8.0;

/// The **quantized** level-clock tick a pulsing emitter's gain is computed at.
///
/// One door, called by both projectors, so the memo key and the emission it
/// stands for cannot be computed two ways — the argument [`night_glow_step`]
/// makes verbatim. A non-finite or negative clock reads as `0`, which is the
/// answer that changes nothing.
pub fn pulse_tick(clock_s: f64) -> u32 {
    if !clock_s.is_finite() || clock_s <= 0.0 {
        return 0;
    }
    // Saturating rather than wrapping: a level clock running for a year of
    // wall time reaches 2.5e8 ticks, well inside `u32`, and a saturated tick is
    // a frozen pulse rather than a discontinuity.
    (clock_s * PULSE_TICKS_PER_S).min(f64::from(u32::MAX)) as u32
}

/// **The floor a pulsing emitter dims to.** A neon tube that goes fully dark
/// reads as a fault; the reference's string lights breathe between about a third
/// and full.
const PULSE_FLOOR: f32 = 0.34;

/// The linear emission an authored emitter contributes at `tick`.
///
/// `hz` of `0.0` — every emitter in the engine except a festoon — returns
/// `emissive` **unchanged and untouched by any arithmetic**, so a steady sign is
/// byte-identical to what a projector without this function would have written.
///
/// # A pure function of the clock, and that is the whole contract
///
/// PIE and the shipped player read the same `ResolvedSky` clock and must agree
/// byte for byte on the trace; a golden renders one frame from cold. So there is
/// no state, no frame index and no generator here — the same rule the scatter
/// dither hash and the foliage wind phase are written to. The sine is
/// [`inf_math::psin64`], not `f64::sin`, because `std` trig is not bit-portable
/// across targets (the P14 law) and this value reaches a `ScatterBatch` that two
/// hosts compare.
pub fn pulse_emissive(emissive: [f32; 3], hz: f32, tick: u32) -> [f32; 3] {
    // `!is_finite() || <= 0.0` rather than `<= 0.0`, the island wave I8a
    // spelling: the readable rewrite is FALSE for a NaN, and a NaN rate would
    // multiply straight through into an emission a batch carries for ever.
    if !hz.is_finite() || hz <= 0.0 {
        return emissive;
    }
    let t = f64::from(tick) / PULSE_TICKS_PER_S;
    let phase = std::f64::consts::TAU * f64::from(hz) * t;
    // Reduced in f64 before the sine, like the foliage wind's phase: at a
    // level's fourth hour `TAU * 0.27 * 14400` is 24 400 radians, and f32 has
    // no fractional radians left at that magnitude.
    let s = inf_math::psin64(phase % std::f64::consts::TAU) as f32;
    let gain = PULSE_FLOOR + (1.0 - PULSE_FLOOR) * (0.5 + 0.5 * s);
    [emissive[0] * gain, emissive[1] * gain, emissive[2] * gain]
}

/// **The colour a sweeping fixture is showing at `clock_s`** (wave VEN1a).
///
/// One door, called by both projectors, so a stage wash cannot be one colour in
/// the editor and another in the shipped player — the argument
/// [`night_glow_step`] and [`pulse_emissive`] both make, and the one that makes
/// a PIE-versus-shipping byte comparison mean anything at all.
///
/// # A triangle, and no trigonometry anywhere
///
/// The sweep is `lerp(a, b, tri(x))` with `tri(x) = 2·|frac(x) − 0.5|` — a pure
/// function of the clock built from `floor` and `abs`, which are exact on every
/// target. A sine would have been prettier by a hair and would have put a libm
/// call on a path two hosts compare byte for byte (the P14 law). The triangle
/// also *dwells* at each end, which is what a moving head actually does.
///
/// # The phase is what makes a rig a rig
///
/// `phase / phases` offsets each fixture around the cycle, so a three-lamp rig
/// shows three colours at once. Without it the three lamps are one lamp with
/// three positions, which is a floodlight.
///
/// A `cycle_hz` of `0.0` — every fixture that is not a stage wash — returns `a`
/// **untouched by any arithmetic**, so a steady lamp is exactly its authored
/// colour.
pub fn swept_colour(
    sweep: ([f32; 3], [f32; 3]),
    cycle_hz: f32,
    phase: u32,
    phases: u32,
    clock_s: f64,
) -> [f32; 3] {
    let (a, b) = sweep;
    // `!is_finite() || <= 0.0`, the island wave I8a spelling: the readable
    // rewrite is FALSE for a NaN, and a NaN rate would carry a NaN colour into
    // the lights uniform.
    if !cycle_hz.is_finite() || cycle_hz <= 0.0 || !clock_s.is_finite() {
        return a;
    }
    let off = if phases > 1 {
        f64::from(phase % phases) / f64::from(phases)
    } else {
        0.0
    };
    // Reduced in f64 before it reaches an f32, like the foliage wind's phase:
    // at a level's fourth hour `0.19 × 14400` is 2 736 cycles and an f32 has
    // little fraction left there.
    let x = (clock_s * f64::from(cycle_hz) + off).fract();
    let t = (2.0 * (x - 0.5).abs()) as f32;
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

/// **The two quantized clocks a scatter projection reads** (wave VEN1a).
///
/// Bundled rather than passed as two scalars because they travel together
/// through five functions in each of two mirrored projectors, and a tenth
/// parameter that one host adds and the other does not is exactly the drift the
/// projector mirror test exists to catch. Both are *quantized* for the same
/// reason — see [`night_glow_step`] and [`pulse_tick`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ScatterClock {
    /// The night-glow step, from [`night_glow_step`].
    pub glow_step: u16,
    /// The pulse tick, from [`pulse_tick`].
    pub pulse_tick: u32,
}

/// **The scatter carry-forward memo** — the scatter twin of what
/// [`take_unchanged_terrain`] does for a terrain (island wave I8a audit).
///
/// Held on [`RenderScene`] rather than beside `scatter`, so there is no parallel
/// list to fall out of step with: the memo OWNS the batches it remembers and the
/// scene's `scatter` list is filled from it by `Arc` clone, which copies a
/// pointer and a handful of scalars per batch and not one instance.
///
/// # How a projection uses it
///
/// Take it out at the top of the walk, ask it for each memoizable source, and
/// build the new one as you go. What is left in the taken-out copy when the walk
/// ends is exactly the scatter that **left** the scene — the terrain memo's own
/// arrangement, and the reason a removal is seen at all.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ScatterMemo {
    by_entity: std::collections::BTreeMap<u128, (ScatterSource, Vec<ScatterBatch>)>,
}

impl ScatterMemo {
    /// The batches this source contributed last projection, removed from the
    /// memo — or `None` when the key does not match exactly.
    ///
    /// A key that differs in any field is a miss, and a miss is silent: the
    /// caller packs the population as it always did.
    pub fn take(&mut self, source: ScatterSource) -> Option<Vec<ScatterBatch>> {
        if !source.is_memoizable() {
            return None;
        }
        match self.by_entity.get(&source.entity) {
            Some((k, _)) if *k == source => self.by_entity.remove(&source.entity).map(|(_, b)| b),
            _ => None,
        }
    }

    /// Remember this source's batches for the next projection.
    ///
    /// A source that is not memoizable is dropped on the floor rather than
    /// stored under a key nothing can ever match.
    pub fn insert(&mut self, source: ScatterSource, batches: Vec<ScatterBatch>) {
        if source.is_memoizable() {
            self.by_entity.insert(source.entity, (source, batches));
        }
    }

    /// How many sources are remembered.
    pub fn len(&self) -> usize {
        self.by_entity.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_entity.is_empty()
    }

    /// Every batch the memo holds, for a caller that wants to price it.
    pub fn batches(&self) -> impl Iterator<Item = &ScatterBatch> {
        self.by_entity.values().flat_map(|(_, b)| b.iter())
    }
}

/// One vertex of a [`SkinnedMeshData`] — position + normal + **uv** in **bind
/// (rest) space**, plus the four joint influences that deform it. `#[repr(C)]` +
/// `Pod` so it uploads straight to a GPU vertex buffer (64 bytes, no padding).
///
/// **The uv is P26.5's**, and it is the half of the change that mattered: a box
/// projection on a *character* is the case the P26.3 ledger named as visibly
/// wrong, because the seams fall on the dominant axis rather than on the
/// artist's and a face's texture will not line up with its head. A skinned
/// `.inf_mesh` has carried authored uvs since P4; only the upload dropped them.
/// It rides `@location(15)` — the last attribute `Limits::default()`'s
/// `max_vertex_attributes: 16` has room for, which is also why **no tangent
/// stream joins it here** (`docs/memos/p26-5-vertex-streams.md`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SkinnedVertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    /// Texture coordinate, `@location(15)`.
    pub uv: [f32; 2],
    /// Joint indices into the instance's skinning palette.
    pub joints: [u32; 4],
    /// Normalized influence weights (`Σ = 1`).
    pub weights: [f32; 4],
}

/// Bind-space geometry for a skinned mesh: an interleaved [`SkinnedVertex`]
/// buffer + a 32-bit index buffer. Referenced by [`SkinnedInstance::mesh`].
#[derive(Debug, Clone, PartialEq)]
pub struct SkinnedMeshData {
    pub vertices: Vec<SkinnedVertex>,
    pub indices: Vec<u32>,
}

/// The tint a simulated garment or hair ribbon draws in (linear rgba).
///
/// **v1 has no per-garment material.** A `.inf_cloth` / `.inf_hair` carries no
/// material slot, so there is nothing to bind a texture *from*; giving the
/// garment the *wearer's* `Material` would draw a coat in exactly the body's
/// colour, which is worse than useless for telling them apart. So both draw in
/// one neutral cloth tint, defined once here so the two projectors cannot
/// disagree about it. Ledgered in ROADMAP §12's P24 block.
///
/// (The original wording added *"and `SkinnedVertex` has no UV channel"*. P26.5
/// gave it one, and [`deformed_skinned_mesh`] fills it with [`box_uv`] — so the
/// remaining obstacle is the missing material slot alone, which is the honest
/// statement of what a v2 has to add.)
pub const CLOTH_TINT: [f32; 4] = [0.42, 0.46, 0.58, 1.0];

/// The tint a simulated hair ribbon draws in (linear rgba).
///
/// Distinct from [`CLOTH_TINT`] for the same reason that one is distinct from the
/// wearer's `Material`: a coat and a head of hair on one character have to be
/// tellable apart, and v1 has no per-asset material to tell them apart with.
pub const HAIR_TINT: [f32; 4] = [0.20, 0.14, 0.10, 1.0];

/// **A CPU-deformed mesh, as something the skinned pass can draw** (P24.4).
///
/// # Why cloth and hair reuse the skinned path instead of getting a pass
///
/// [`SkinnedInstance`]'s palette is applied in the vertex shader **before** the
/// model matrix, so a skinned vertex stream is exactly "model-space positions,
/// transformed by the instance". A simulated garment is model-space positions.
/// So a garment is a skinned mesh with a **one-entry identity palette** and every
/// vertex pinned to joint 0 at weight 1 — no new render node, no new shader, no
/// `SHADER_TABLE` entry, and no golden re-blessed. The alternative was a cloth
/// pass that would have been the skinned pass with the palette deleted.
///
/// # Normals are recomputed, and must be
///
/// A deforming surface's normals are not a property of its topology. They are
/// area-weighted face normals accumulated per vertex (the cross product is
/// area-weighted by construction — no normalization per face, which is both
/// cheaper and the correct weighting), then normalized once. A degenerate or
/// unreferenced vertex keeps `+Y` rather than a zero normal, because a zero
/// normal is a black pixel and a wrong-but-unit one is a shading artefact.
///
/// # The `Arc` discipline does NOT apply here, and that is deliberate
///
/// The skinned pass caches its GPU upload by the pointer identity of the
/// `Arc<SkinnedMeshData>` (see [`RenderScene::skinned_meshes`]). A garment's
/// vertices change **every step**, so a fresh `Arc` per projection is not a bug
/// to be fixed but the honest statement that this geometry has to be re-uploaded.
/// Ledgered: a garment costs one vertex-buffer upload per frame, which is what
/// bounds how many characters can wear one.
///
/// **The dominant-axis box projection** — the uv a surface with no authored one
/// samples with (P26.3's `vt_box_uv`, moved to Rust in P26.5).
///
/// It lived in `vt_sample.wgsl` and ran in the fragment stage of *every* rigid
/// and skinned draw, because the render vertex streams carried no uv at all.
/// They carry one now, so the projection has exactly one producer left — the
/// deformed cloth/hair geometry below, which is positions and topology and has
/// no parametrization to inherit. Computing it once per vertex where the stream
/// is built, rather than once per fragment where it is read, is strictly less
/// work and puts the fallback next to the one thing that needs it.
///
/// Object space (so the projection rides the instance instead of sliding when it
/// moves), `[0,1]²` per face, with the same winding the WGSL had — character for
/// character, so a surface that used to sample this way still does.
pub fn box_uv(p: [f32; 3], n: [f32; 3]) -> [f32; 2] {
    let a = [n[0].abs(), n[1].abs(), n[2].abs()];
    if a[0] >= a[1] && a[0] >= a[2] {
        return [-p[2] * n[0].signum() + 0.5, -p[1] + 0.5];
    }
    if a[1] >= a[2] {
        return [p[0] + 0.5, -p[2] * n[1].signum() + 0.5];
    }
    [p[0] * n[2].signum() + 0.5, -p[1] + 0.5]
}

/// `indices` is passed through unchanged. An index out of range is **dropped**
/// with its whole triangle rather than clamped, so a malformed garment draws less
/// instead of drawing a spike to the origin.
///
/// # The uv is box-projected, and that is the last caller of the projection
///
/// P26.5 gave [`SkinnedVertex`] a uv and every other producer of one fills it
/// from the asset. This geometry has no asset uv to fill it from — a garment is
/// simulated positions over a topology — so it takes [`box_uv`] against the
/// recomputed normal. Nothing samples it yet (a garment's set is
/// [`VtTextureSet::NONE`]; see [`CLOTH_TINT`]), which is precisely why it must
/// not be left at zero: an all-zero uv is a whole coat sampling one texel, and
/// it would arrive silently on the day a `.inf_cloth` grows a material slot.
pub fn deformed_skinned_mesh(positions: &[[f32; 3]], indices: &[u32]) -> SkinnedMeshData {
    let n = positions.len();
    let mut accum = vec![glam::Vec3::ZERO; n];
    let mut kept: Vec<u32> = Vec::with_capacity(indices.len());
    for tri in indices.chunks_exact(3) {
        let (i, j, k) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        if i >= n || j >= n || k >= n {
            continue;
        }
        let (a, b, c) = (
            glam::Vec3::from_array(positions[i]),
            glam::Vec3::from_array(positions[j]),
            glam::Vec3::from_array(positions[k]),
        );
        let face = (b - a).cross(c - a);
        if !face.is_finite() {
            continue;
        }
        accum[i] += face;
        accum[j] += face;
        accum[k] += face;
        kept.extend_from_slice(&[tri[0], tri[1], tri[2]]);
    }
    let vertices = positions
        .iter()
        .zip(&accum)
        .map(|(p, acc)| {
            let len = acc.length();
            let normal = if len > 1.0e-12 && acc.is_finite() {
                *acc / len
            } else {
                glam::Vec3::Y
            };
            let normal = normal.to_array();
            SkinnedVertex {
                pos: *p,
                normal,
                uv: box_uv(*p, normal),
                joints: [0; 4],
                weights: [1.0, 0.0, 0.0, 0.0],
            }
        })
        .collect();
    SkinnedMeshData {
        vertices,
        indices: kept,
    }
}

/// **How one skinned instance casts its shadow** (wave NPC1b).
///
/// A skinned caster is a *geometry group* in the page raster, and
/// [`VSM_MAX_GROUPS`](crate::VSM_MAX_GROUPS) is 1 024 — so a thousand NPCs is a
/// thousand groups, and past the ceiling a crowd's shadows are refused (counted,
/// but refused). This enum is the renderer's whole reading of the crowd tier: the
/// near few keep a real skinned caster, everything further out casts through
/// **one shared proxy group** for the entire crowd.
///
/// [`BindSphere`](Self::BindSphere) is the [`Default`] and is what every
/// instance before this wave was: nothing in the tree opts into the other two
/// unless a projector reads a tier, so every committed level, every golden and
/// every P27 page-cache arm is byte-identical to its pre-NPC1b self.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SkinnedShadow {
    /// A skinned caster culled on the mesh's **bind** bounding sphere inflated by
    /// [`SKINNED_POSE_MARGIN`](crate::SKINNED_POSE_MARGIN) — the pre-NPC1b rule,
    /// and still the rule for every hero, garment and hair ribbon.
    #[default]
    BindSphere,
    /// A skinned caster culled on the **exact** bound of the pose it is drawing:
    /// the union of this instance's per-joint bind spheres transformed by its own
    /// palette. Strictly tighter than [`BindSphere`](Self::BindSphere) and
    /// therefore strictly fewer invalidated pages.
    Posed,
    /// **No skinned caster at all.** This instance casts from the crowd's shared
    /// box proxy, which is ONE group however many agents are in it.
    Proxy,
    /// **This instance casts nothing** — the crowd's shadow LOD (island wave
    /// NPC1e).
    ///
    /// [`Proxy`](Self::Proxy) answered `VSM_MAX_GROUPS` and did not answer the
    /// other half of what a moving crowd costs the shadow map: NPC1b's carried
    /// item 4 measured 968 proxy boxes scattering page invalidation over
    /// **168.6 pages a frame against the island's own 56.3**, at 1 236 page
    /// draws against 328, and the NPC1b audit added that
    /// `VsmRasterStats::deferred_pages` doubled with it — so the crowd was not
    /// only re-rasterizing more, it was serving more *stale* shadow. One group
    /// is the right answer to a group ceiling and is no answer at all to *how
    /// many pages a moving crowd dirties*. That item named the lever —
    /// *"proxies that stop casting past a radius"* — and this is it.
    ///
    /// The radius is the sim ladder's own: an agent past `CrowdTier::Near`
    /// (96 m) casts nothing. It is a **visible** cost and is stated as one — a
    /// person a hundred metres off loses their shadow — which is the same trade
    /// `Proxy` already makes one rung in (an NPC's arms and legs do not move its
    /// shadow past 32 m).
    None,
}

/// **The one-entry identity palette**, shared process-wide.
///
/// A CPU-deformed garment or hair ribbon arrives already in model space, so its
/// "skinning" is the identity — see [`deformed_skinned_mesh`]. Every one of them
/// naming the *same* `Arc` means a level full of garments costs the palette atlas
/// exactly one matrix rather than one per ribbon, through the same pointer-identity
/// dedup a crowd's shared rest pose uses. It is also the reason this is a function
/// and not a `vec![]` at each call site: two `vec![]`s are two blocks.
pub fn identity_palette() -> std::sync::Arc<Vec<Mat4>> {
    static ONE: std::sync::OnceLock<std::sync::Arc<Vec<Mat4>>> = std::sync::OnceLock::new();
    ONE.get_or_init(|| std::sync::Arc::new(vec![Mat4::IDENTITY]))
        .clone()
}

/// One skinned draw: a [`SkinnedMeshData`] (by index into
/// [`RenderScene::skinned_meshes`]) placed by a world transform and deformed by a
/// per-instance **skinning palette** (`global · inverse_bind` per joint, computed
/// CPU-side by the host).
///
/// The palette is applied in the vertex shader **before** the model matrix, so it
/// stays in bind/model space (no floating-origin adjustment needed — only the
/// model translation is origin-relative, exactly like [`MeshInstance`]).
#[derive(Debug, Clone, PartialEq)]
pub struct SkinnedInstance {
    /// World-space translation (f64 — architecture rule 3).
    pub translation: DVec3,
    pub rotation: Quat,
    pub scale: Vec3,
    /// Linear-space base color (rgba).
    pub color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub emissive: [f32; 3],
    /// Stable pick id (`ID_NONE` reserved).
    pub id: u32,
    /// Index into [`RenderScene::skinned_meshes`].
    pub mesh: usize,
    /// Blend mode (wave CHAR1a.2): `0` opaque, `1` masked (alpha-test), `2`
    /// translucent. **The same codes and the same meaning as
    /// [`MeshInstance::blend`]**, projected from the same `Material::blend` at
    /// the same seams — because a hair card, an eyelash sheet and a garment are
    /// skinned surfaces, and until this field existed the skinned path had no way
    /// to say "this fragment is a hole".
    ///
    /// Defaults to `0` at every construction site that predates it, so every
    /// committed skinned golden runs the identical arithmetic: the discard below
    /// is present-and-false for an opaque body.
    ///
    /// **`2` draws opaque, and that is stated rather than hidden.** The rigid
    /// path buckets translucent instances into their own draw range
    /// (`crate::passes::mesh::pack_bucketed`); the skinned pipeline has one
    /// blend state and one range per mesh, so a translucent skinned surface is a
    /// pipeline this pass does not have yet. Masked is the case the characters
    /// need and the case this wave built.
    pub blend: u8,
    /// Alpha-test threshold used when `blend == 1`: fragments whose base-colour
    /// alpha is below this are discarded. Defaults to `0.5`, exactly like
    /// [`MeshInstance::cutoff`].
    ///
    /// Packed with [`blend`](Self::blend) into the ONE free channel of the shared
    /// instance stream (`pbr.w`) — see
    /// [`crate::passes::skinned`]'s `instance_raw`, which explains why there is
    /// only one.
    pub cutoff: f32,
    /// The skinning palette: one matrix per skeleton joint, indexed by the
    /// vertex `joints`. Uploaded into the pass's **palette atlas** at an offset
    /// this instance carries in a packed channel.
    ///
    /// **`Arc`, since NPC1b, and the sharing is the point.** A crowd's far tier
    /// evaluates no pose, so every agent of one archetype out there resolves to
    /// the *same* rest-pose palette — a thousand byte-identical 16 KiB blocks,
    /// derived a thousand times and uploaded a thousand times. The projector now
    /// hands one `Arc` to all of them and
    /// [`plan_skinned_batches`](crate::plan_skinned_batches) deduplicates the
    /// atlas by pointer identity, exactly as `skinned_meshes` deduplicates
    /// geometry. A posed character still gets its own, because a pose is the part
    /// that actually changes.
    pub palette: std::sync::Arc<Vec<Mat4>>,
    /// How this instance casts into the virtual shadow map (wave NPC1b).
    pub shadow: SkinnedShadow,
    /// P26.3: the virtual textures this instance samples. [`VtTextureSet::NONE`]
    /// on every instance that names none, which is every instance before this
    /// batch — and then the fragment shader runs the arithmetic it always ran.
    pub vt: VtTextureSet,
    /// **The mesh's material SECTIONS** (wave CHAR1a.3) - one drawn index range
    /// per material slot, each with its own surface. **Empty** means what it has
    /// always meant: one draw over the whole index buffer, in this instance's own
    /// colour / metallic / roughness / blend / cutoff / `vt`.
    ///
    /// # Why the ranges ride the instance and not the geometry
    ///
    /// [`SkinnedMeshData`] is cached by `Arc` identity and uploaded once; a
    /// [`VtTextureSet`] is a **warm-gated snapshot** that changes as the
    /// residency pages in. Putting the surfaces on the cached geometry would
    /// freeze the first frame's answer into every later one. The RANGES are a
    /// property of the mesh and are derived from it
    /// (`inf_mesh::MeshAsset::skinned_sections`), which is why both projectors
    /// call that one function rather than re-deriving the concatenation rule.
    ///
    /// A body with one slot - every committed character in this tree, every
    /// crowd agent, every garment, and the MetaHuman full-body mesh UE's own
    /// `CreateCombinedFaceAndBodyMesh` writes - leaves this empty and draws
    /// exactly the command stream it drew before this field existed. That is not
    /// a hope: the two committed skinned goldens are sha256-identical across the
    /// change.
    pub sections: Vec<SkinnedSection>,
}

/// **THE ONE DOOR both hosts build a skinned mesh's sections through** (wave
/// CHAR1a.3).
///
/// `ranges` is `(first index, index count, material guid)` per section, as the
/// two skinned stores derive it from the `.inf_mesh`'s v3 slot table.
/// `instance` is the surface the entity's own `Material` resolved to — the
/// fallback for a slot that names nothing, and for a slot whose material the
/// host cannot resolve. `surface` reads a material GUID's
/// `(colour, metallic, roughness, emissive, blend, cutoff)`; `vt_slots` reads
/// the same GUID's virtual-texture set, which is the half that has to be asked
/// per frame because residency warms.
///
/// # Why it is here and not twice in the projectors
///
/// Both hosts have to answer the same question — "what does slot 3 of this body
/// draw as?" — from two different material stores, and the *rule* is the same
/// while the *lookup* is not. Putting the rule in Ring 0 with the lookups as
/// closures is the only shape where the two hosts cannot answer it differently;
/// `projector_mirror` pins the two call sites, and this function is what those
/// two call sites both reduce to.
///
/// Returns an EMPTY vector for an empty `ranges`, which is what an unsectioned
/// mesh has: the instance then draws exactly the one whole-buffer command it drew
/// before sections existed.
pub fn skinned_sections(
    ranges: &[(u32, u32, Option<u128>)],
    instance: &SkinnedInstance,
    mut surface: impl FnMut(u128) -> Option<([f32; 4], f32, f32, [f32; 3], u8, f32)>,
    mut vt_slots: impl FnMut(u128) -> VtTextureSet,
) -> Vec<SkinnedSection> {
    ranges
        .iter()
        .map(|(first_index, index_count, material)| {
            let mut out = SkinnedSection {
                first_index: *first_index,
                index_count: *index_count,
                color: instance.color,
                metallic: instance.metallic,
                roughness: instance.roughness,
                emissive: instance.emissive,
                blend: instance.blend,
                cutoff: instance.cutoff,
                vt: instance.vt,
            };
            if let Some(guid) = *material {
                if let Some((color, metallic, roughness, emissive, blend, cutoff)) =
                    surface(guid)
                {
                    out.color = color;
                    out.metallic = metallic;
                    out.roughness = roughness;
                    out.emissive = emissive;
                    out.blend = blend;
                    out.cutoff = cutoff;
                }
                out.vt = vt_slots(guid);
            }
            out
        })
        .collect()
}

/// One drawn range of a [`SkinnedMeshData`]'s index buffer, with the surface it
/// draws in (wave CHAR1a.3).
///
/// The engine drew ONE material per skinned mesh until this type existed -
/// measured, and it is not a corner case: `SKM_Manny` ships **two** material
/// slots and a MetaHuman face **twelve**, and every one of them was drawn with
/// slot 0's material. A face whose eyes, lashes, teeth and skin are one surface
/// is not a face.
///
/// Every section of one instance shares that instance's **palette**: they are
/// ranges of one skinned mesh deformed by one skeleton, so the atlas block is
/// looked up once and the extra sections cost draw calls and instance rows, not
/// palette bytes.
#[derive(Debug, Clone, PartialEq)]
pub struct SkinnedSection {
    /// First index of the range, into the mesh's own 32-bit index buffer.
    pub first_index: u32,
    /// Indices in the range. A section with none is skipped rather than drawn.
    pub index_count: u32,
    /// Linear-space base colour (rgba) for this range.
    pub color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub emissive: [f32; 3],
    /// `0` opaque, `1` masked, `2` translucent - [`SkinnedInstance::blend`]'s
    /// codes exactly, because a section is a surface and the fragment stage that
    /// reads it is the same one.
    pub blend: u8,
    /// Alpha-test threshold when `blend == 1`.
    pub cutoff: f32,
    /// The virtual textures this range samples.
    pub vt: VtTextureSet,
}

/// Directional, point, or spot light (R-P3). Spot is a point light with a cone
/// mask; its emission axis is `-direction` (see [`RenderLight`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LightKind {
    Directional,
    Point,
    Spot,
}

/// A scene light in world space. The mesh pass converts point/spot positions to
/// render-local (floating-origin-relative) space at upload, exactly like
/// instance transforms.
///
/// ## Direction conventions
///
/// * [`direction`](Self::direction) is the unit vector *toward* the light for
///   [`Directional`](LightKind::Directional) — the existing convention.
/// * For [`Spot`](LightKind::Spot) the same "toward-the-light" vector is stored
///   in `direction`; the beam **emission** axis is therefore `-direction`. The
///   seams project an entity whose forward is `-Z` as `direction = rot * +Z`, so
///   emission = `-direction = rot * -Z`.
///
/// ## Shadows (R-P3 scope)
///
/// [`cast_shadows`](Self::cast_shadows) is honoured only for
/// [`Directional`](LightKind::Directional) lights, where it gates CSM caster
/// selection (see [`crate::passes::shadow`]). Point/spot shadow maps are
/// **deferred**, so the flag is inert (stored, never sampled) for those kinds.
#[derive(Debug, Clone, Copy)]
pub struct RenderLight {
    pub kind: LightKind,
    /// Linear light color.
    pub color: [f32; 3],
    /// Radiant intensity multiplier.
    pub intensity: f32,
    /// Unit direction *toward* the light (directional + spot; see the type docs
    /// — spot emission is `-direction`).
    pub direction: Vec3,
    /// World-space position (point + spot).
    pub position: DVec3,
    /// Influence radius in metres (point + spot); 0 ⇒ unbounded.
    pub range: f32,
    /// Spot inner-cone cosine (full brightness where `cos(angle) ≥ inner_cos`).
    /// Unused for directional/point. Default = `cos(30°)`.
    pub inner_cos: f32,
    /// Spot outer-cone cosine (zero where `cos(angle) ≤ outer_cos`; `outer_cos <
    /// inner_cos`). Unused for directional/point. Default = `cos(40°)`.
    pub outer_cos: f32,
    /// Whether this light casts shadows. Honoured for directional (CSM caster
    /// selection); inert for point/spot (shadow maps deferred).
    pub cast_shadows: bool,
}

impl Default for RenderLight {
    fn default() -> Self {
        Self {
            kind: LightKind::Directional,
            color: [1.0, 1.0, 1.0],
            intensity: 1.0,
            direction: Vec3::Y,
            position: DVec3::ZERO,
            range: 0.0,
            // 30° / 40° half-angles, mirroring the ECS `Light` cone defaults.
            inner_cos: 0.866_025_4,  // cos(30°)
            outer_cos: 0.766_044_44, // cos(40°)
            cast_shadows: true,
        }
    }
}

/// A minimal 2D light (P8.1c): a soft radial falloff in the sprite plane. The
/// sprite pass converts `position` to render-local (floating-origin-relative) at
/// upload, exactly like 3D point lights, and lights every sprite/tile/text/
/// 9-slice fragment by world-XY distance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderLight2D {
    /// Linear light color.
    pub color: [f32; 3],
    /// Brightness multiplier.
    pub intensity: f32,
    /// World-space falloff radius; the contribution is `smoothstep(radius, 0,
    /// dist)`, so it is full at the light and zero at/after `radius`.
    pub radius: f32,
    /// World-space position (the sprite plane's XY is what matters).
    pub position: DVec3,
}

/// Scene-level 2D ambient term. **Defaults to white (`1,1,1`)** so that with no
/// [`RenderLight2D`] present every sprite renders exactly as before
/// (`texel·tint·1`) — the byte-stability guarantee for pre-P8.1c goldens.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ambient2D(pub [f32; 3]);

impl Default for Ambient2D {
    fn default() -> Self {
        Self([1.0, 1.0, 1.0])
    }
}

/// A projected terrain tile's identity: its **asset LOD level** plus its grid
/// coordinate *at that level* (P16.3b1).
///
/// This is the renderer-local mirror of `inf_terrain::TileKey` — `inf-render` is
/// Ring 0 and deliberately does **not** depend on `inf-terrain`, so the two
/// documented projectors (`inf_viewport::host::project_terrain` and
/// `inf_player::render::project_terrain`) map one onto the other, exactly like
/// every other scene DTO.
///
/// Level `0` is the authored, full-resolution heightfield. A level-`n` tile
/// covers `2ⁿ ×` the world span at the same sample count (metres-per-sample
/// doubles per level), so level-`n` tile `(TX, TZ)` is the 2 × 2 block of
/// level-`(n−1)` tiles `(2TX+a, 2TZ+b)` decimated 2:1.
///
/// `Ord` sorts by **`lod` first, then `coord`** (matching `inf_terrain::TileKey`),
/// so a projection that emits level 0 and then each coarse level in ascending key
/// order is globally key-ascending — the order the tile list is documented to
/// arrive in.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TerrainTileKey {
    /// Asset LOD level; `0` is the authored full-resolution level.
    pub lod: u32,
    /// Tile grid coordinate `(tx, tz)` **within that level**.
    pub coord: (i32, i32),
}

impl TerrainTileKey {
    /// A level-0 (authored, full-resolution) key.
    #[inline]
    pub const fn lod0(coord: (i32, i32)) -> Self {
        Self { lod: 0, coord }
    }

    /// A key at an explicit level.
    #[inline]
    pub const fn new(lod: u32, coord: (i32, i32)) -> Self {
        Self { lod, coord }
    }

    /// The coarser key one level up that contains this tile (`lod + 1`, coordinate
    /// halved with **floor** semantics so negative coordinates group correctly —
    /// tile `−1` belongs to block `−1`, not block `0`).
    #[inline]
    pub const fn parent(self) -> Self {
        Self {
            lod: self.lod + 1,
            coord: (self.coord.0.div_euclid(2), self.coord.1.div_euclid(2)),
        }
    }

    /// The four finer keys this tile decimates (`lod − 1`, coordinate doubled),
    /// in a fixed scan order. Empty at level 0 (nothing is finer).
    ///
    /// Saturating doubling: a coordinate that would overflow `i32` cannot name a
    /// real tile anyway, so the clamped key simply never matches a resident one.
    #[inline]
    pub fn children(self) -> [Self; 4] {
        let lod = self.lod.saturating_sub(1);
        let (x, z) = (
            self.coord.0.saturating_mul(2),
            self.coord.1.saturating_mul(2),
        );
        [
            Self::new(lod, (x, z)),
            Self::new(lod, (x.saturating_add(1), z)),
            Self::new(lod, (x, z.saturating_add(1))),
            Self::new(lod, (x.saturating_add(1), z.saturating_add(1))),
        ]
    }
}

/// One terrain tile handed to the [`TerrainNode`](crate::passes::terrain): its
/// [`TerrainTileKey`] (asset LOD + grid coordinate), the `f64` world origin of
/// sample `(0,0)`, the row-major `f32` height offsets (from `origin.y`), and the
/// **change stamp** the GPU cache gates its upload on. Mirrors
/// `inf_terrain::TerrainTile` but stays renderer-agnostic (the host projects it,
/// like `RenderTilemap`).
#[derive(Debug, Clone, PartialEq)]
pub struct RenderTerrainTile {
    /// Asset LOD level + grid coordinate at that level.
    pub key: TerrainTileKey,
    /// World position of sample `(0,0)` (`f64` anchor).
    pub origin: DVec3,
    /// `resolution²` row-major height offsets (metres) from `origin.y`.
    pub heights: Vec<f32>,
    /// `resolution²` row-major RGBA8 splat weights (P10.4), resolved from the
    /// tile's sparse store (an unpainted tile projects the uniform default). The
    /// terrain pass uploads these into a per-tile `Rgba8Unorm` weight texture
    /// beside the height texture. Coarse (LOD ≥ 1) tiles carry no painted weights
    /// — the pyramid is heights-only — so they project the uniform default.
    pub weights: Vec<[u8; 4]>,
    /// `resolution²` row-major **biome ids** (P19.2), one `u8` per sample, resolved
    /// from the tile's sparse store exactly like [`weights`](Self::weights) — this
    /// vec is ALWAYS dense, the projector expands the sparse default. A tile that
    /// has never been painted projects all-`0` (`inf_terrain::UNASSIGNED_BIOME`),
    /// and coarse (LOD ≥ 1) pyramid pages carry no painted data at all — the
    /// pyramid is heights-only — so they project that same uniform default. The
    /// terrain pass uploads these into a per-tile `R8Uint` texture beside the
    /// height + weight ones; only the Biomes view mode reads it.
    ///
    /// A biome id is **categorical**: it indexes
    /// [`RenderTerrain::biome_palette`] and is never filtered or interpolated (the
    /// shader loads it nearest — the midpoint of ids 1 and 3 is not id 2).
    pub biomes: Vec<u8>,
    /// `ceil(resolution / 32) * resolution` **row-packed hole bits** (P21.2), or
    /// **empty** for a tile nothing has carved. Word `w + j * words_per_row` holds
    /// the bits for samples `[w*32, w*32+32)` of row `j`, LSB-first.
    ///
    /// Row-packed, not tile-packed like `inf_terrain::TerrainTile`'s own mask: the
    /// fragment's index becomes `(i >> 5, j)`, which needs no division by the
    /// resolution and maps one-to-one onto an R32Uint texture's texel grid. The
    /// projector does that repack, once per carved tile per edit, and only for
    /// tiles that have holes at all — a hole-free tile projects an empty `Vec`
    /// and the pass binds a 1x1 zero texture for it (four bytes, no permutation).
    ///
    /// Empty is therefore not "unknown", it is **"nothing is holed"** — the same
    /// sparse-default rule the source layer carries, projected intact.
    pub holes: Vec<u32>,
    /// Inclusive `(min, max)` of `heights` (for the tile's AABB cull bound).
    pub height_bounds: (f32, f32),
    /// The tile's **monotone change stamp** (P16.3b1), projected from
    /// `inf_terrain::TerrainData::tile_version`. The GPU texture cache re-uploads
    /// this tile if — and only if — the stamp differs from the one its cached
    /// copy was built at. `0` means "no stamp" (a tile the source could not
    /// version) and is treated conservatively as *always re-upload*, never as a
    /// cache hit.
    pub version: u64,
}

/// **Fill every voxel volume's P21.2 seam terms from the heightfields beside
/// them**, and arm the blend at `band_m` metres.
///
/// The one implementation both host projectors call, and that is the point
/// rather than a convenience: the editor viewport and the shipped player must
/// agree pixel for pixel about where a cave mouth stops being cave, and two
/// hand-synced copies of a per-vertex loop is exactly the shape that eventually
/// does not. (Same reasoning as [`RenderTerrain::seam_sample`] living on the
/// DTO.) A host calls this once, after it has projected both halves.
///
/// A vertex over **no** terrain — or over a holed sample — keeps
/// [`RenderVoxelVertex::NO_SEAM`] and blends nothing, which is what makes the
/// mouth of a cave shade into the hillside while its interior does not. Passing
/// `band_m <= 0` disarms every volume, so a host with no terrain at all produces
/// byte-identical frames to its pre-P21.2 self.
///
/// Projections happen on a change stamp, not per frame.
///
/// # Which heightfield a vertex seams against, when several overlap (P16.6)
///
/// **The nearest surface wins — not the first one in the list.** Every terrain that
/// answers at the vertex's `(x, z)` is sampled, and the one whose surface is
/// vertically closest to that vertex takes it; an exact distance tie goes to the
/// lower [`RenderTerrain::id`], which both hosts fold from the same entity `Guid`.
///
/// This was a `find_map` — first answer wins — and that is **list-order-dependent**,
/// which is the one thing this function cannot afford to be: the editor projects
/// its terrains in document order (`doc.order()`) and the shipped player in `Guid`
/// order. With two overlapping heightfields the same voxel vertex therefore seamed
/// against a *different* terrain in each host, and a cave mouth came out with a
/// different albedo, roughness and shading normal in the preview than in the build
/// — the PIE ≠ shipping failure `project_voxel`'s mirror gate exists to prevent,
/// reached from underneath it, where no character-for-character comparison of the
/// two projectors could ever have seen it.
///
/// Making the **rule** order-free was chosen over forcing one iteration order onto
/// both hosts. Those orders are load-bearing where they are (the editor's is the
/// document the author is looking at; the player has no document), pinning them
/// would couple two crates that share no dependency, and a gate asserting "both
/// walked the same way" would still say nothing about the next consumer. A rule
/// that cannot read the list order cannot be broken by one.
///
/// Nearest-in-Y rather than, say, the largest footprint overlap, because it is the
/// answer the content means: a mouth opens into the ground it was cut through, so
/// where a valley terrain and a plateau terrain overlap in plan, a mouth at
/// `y ≈ 0` belongs to the valley however wide the plateau is.
///
/// A terrain that answers and is then vetoed (below) has still answered: the search
/// does not fall through to another heightfield, because "the ground here is carved
/// away" is an answer.
///
/// Cost is one `seam_sample` per terrain per vertex instead of per vertex, so a
/// single-terrain level — every golden, every unit test, every level that has been
/// through one Terrain Import — pays exactly what it paid before. (Two terrains
/// sharing one `id` would leave a distance tie unbroken and fall back to list
/// position; that is already forbidden — see [`RenderTerrain::id`].)
///
/// # What paging cannot change here, and by which mechanism (P21.2 re-audit)
///
/// Two independent residency systems move under a seam, and only one of them is
/// the answer for each axis:
///
/// * **The voxel-chunk axis** is closed by `inf_voxel::MeshSourceKey`'s 27-neighbour
///   residency mask: a chunk re-meshes when a neighbour pages in *or* out, so the
///   vertices walked below are the ones a fully-resident neighbourhood produces.
///   That mask knows nothing about terrain.
/// * **The terrain axis** is closed by [`RenderTerrain::max_lod`] and by nothing
///   else. [`RenderTerrain::seam_sample`] reads the streamer's residency floor,
///   which is pinned for the stream's whole life, so fine-page churn under a cave
///   mouth cannot move a single seam value. An earlier round named the residency
///   mask for this axis; it is the wrong mechanism, and the correction is recorded
///   rather than quietly swapped, because "the right property, guaranteed by the
///   wrong thing" is what survives a review and fails a refactor.
///
/// The **residual**, stated rather than discovered: a streamed terrain whose floor
/// has not arrived yet projects *no tiles*, so its host pushes no `RenderTerrain`
/// at all, this function returns at the `terrains.is_empty()` line, and the volume
/// keeps [`RenderVoxelVertex::NO_SEAM`]. Both hosts gate re-projection on the
/// document version, so nothing repairs it until the document next changes. That is
/// accepted behaviour for now — the window is one stream's first projection, and
/// the alternative (re-projecting on residency) is the camera-driven path the P18
/// law keeps out of lighting — and it is pinned by
/// `a_terrain_with_no_tiles_leaves_the_volume_unseamed` so it cannot become an
/// unnoticed one.
///
/// # The mask-free veto, and where it applies
///
/// [`RenderTerrain::seam_sample`] reads the residency floor, so on a **streamed**
/// terrain it reads a coarse pyramid page — which carries no hole mask, so its
/// poison rule cannot fire (see
/// [`seam_holes_are_known`](RenderTerrain::seam_holes_are_known)). Blending
/// anyway would put grass on a cave ceiling: at a mouth the roof is *at* the
/// heightfield's surface, which is exactly where the band is widest.
///
/// So where the mask is missing, one rule that needs no mask stands in for it: a
/// voxel surface only *continues* a heightfield if it faces the same way it does,
/// `dot(vertex normal, heightfield normal) > 0`. A cave roof faces down and a
/// heightfield normal always faces up, so a roof is refused; a cave wall is
/// perpendicular and is refused too; the mouth's outward-turning lip — the one
/// surface the blend exists for — faces up and keeps its seam.
///
/// It is deliberately applied **only** where the mask is absent, and that is not
/// timidity: it is strictly weaker than the mask (it cannot see a hole in flat
/// ground at all), so using it in place of a mask that *is* present would trade a
/// correct answer for an approximate one — and would silently move every existing
/// golden, which has an inline terrain and therefore a mask.
///
/// # It writes every vertex it visits, refusals included (Hardening Wave E)
///
/// Originally it *skipped* a vertex it refused to seam, which was correct while
/// the only input it ever saw was a freshly projected volume — one whose terms
/// are already [`RenderVoxelVertex::NO_SEAM`]. Both hosts now carry an unchanged
/// volume forward from the previous frame rather than rebuilding it (the P2
/// memo), and a carried volume arrives holding **last frame's** seam terms: a
/// vertex that used to be seamed and is now refused would have kept a stale
/// blend, and a terrain that left the scene entirely would have frozen the seam
/// it was last sampled at.
///
/// So the refusals write the sentinel explicitly, and the "nothing to sample"
/// case clears rather than returns. The result is a **pure function of
/// `(positions, normals, terrains, band_m)`** — which is what makes re-running it
/// over a carried volume produce the same bytes as running it over a fresh one,
/// and therefore what makes the memo safe. For every caller that existed before
/// this change the behaviour is byte-identical: a fresh vertex already carries
/// the sentinel.
pub fn apply_seam(volumes: &mut [RenderVoxelVolume], terrains: &[RenderTerrain], band_m: f32) {
    let armed = band_m > 0.0 && !terrains.is_empty();
    for volume in volumes {
        volume.seam_band_m = if armed { band_m } else { 0.0 };
        for chunk in &mut volume.chunks {
            let base = chunk.origin;
            for v in &mut chunk.vertices {
                // Cleared FIRST, so every path below — including each `continue`
                // — leaves the sentinel rather than whatever was there before.
                v.seam_nh = RenderVoxelVertex::NO_SEAM;
                v.seam_albedo = [0.0; 4];
                if !armed {
                    continue;
                }
                let wx = base.x + v.pos[0] as f64;
                let wy = base.y + v.pos[1] as f64;
                let wz = base.z + v.pos[2] as f64;
                // The nearest answering surface, by a total order that does not
                // mention the list: |Δy| first, then `id`. See the doc block —
                // `find_map` here made a cave mouth's material a function of which
                // host projected it.
                let mut best: Option<(&RenderTerrain, SeamSample, f64)> = None;
                for t in terrains {
                    let Some(sample) = t.seam_sample(wx, wz) else {
                        continue;
                    };
                    let dy = (sample.height - wy).abs();
                    let wins = match &best {
                        None => true,
                        Some((bt, _, bdy)) => dy < *bdy || (dy == *bdy && t.id < bt.id),
                    };
                    if wins {
                        best = Some((t, sample, dy));
                    }
                }
                let Some((terrain, sample, _)) = best else {
                    continue;
                };
                if !terrain.seam_holes_are_known() && !continues_surface(v.normal, sample.normal) {
                    continue;
                }
                // `origin.y` of the chunk is already the anchor the vertex
                // positions are relative to, so the packed height is measured
                // in the same space the shader compares it against.
                let (nh, albedo) = sample.pack(base.y);
                v.seam_nh = nh;
                v.seam_albedo = albedo;
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The producer memos (Hardening Wave E, 2026-08-14)
// ─────────────────────────────────────────────────────────────────────────────
//
// **The class, stated once for both functions below.**
//
// `RenderTerrainTile::version` and `RenderVoxelChunk::version` are monotone
// change stamps, and every *consumer* in this crate honours them: the terrain
// pass's `plan_tile_cache` and the voxel pass's `plan_chunk_cache` both discard
// an incoming payload whose stamp matches the one their cached GPU copy was
// built at. What no consumer could do is stop the payload from being *built*.
//
// So both hosts rebuilt, every frame, per resident tile: a `heights().to_vec()`,
// a resolved `weights` buffer, a resolved `biomes` buffer, a packed hole mask
// and a full res² min/max rescan for the bounds — and per resident chunk: two
// full rebases of the position stream (`local_bounds_m` called
// `local_positions_m` a second time), a mapped vertex stream and an
// `indices.clone()`. At the Phase-16 gate scene's 84 tiles that is ~12.6 MB of
// allocation and memcpy per frame, ~760 MB/s at 60 Hz, thrown away by a
// consumer that already knew nothing had changed.
//
// The two functions below close it the cheapest way there is: the payload the
// host built LAST frame is still sitting in the scene, so it is carried forward
// rather than rebuilt. That makes a hit cost a `Vec` move and zero bytes copied,
// and it needs no cache, no eviction policy and no lifetime — the memo IS the
// previous frame's scene.
//
// **Why the key is sound.** Both stamps are drawn from a *process-global*
// atomic (`inf_terrain`'s `NEXT_TILE_VERSION`, `inf_voxel`'s
// `NEXT_MESH_VERSION`), so a stamp is unique across every terrain, every volume
// and every level in the process — a stale hit is not merely unlikely, it is
// unreachable. `0` means "not in the ledger" on both sides and is treated as a
// forced miss, exactly as the GPU caches treat it.
//
// **Why a max-fold would NOT have been sound**, recorded because it is the
// obvious cheaper thing: a removal (`TerrainData::evict_tile`,
// `VoxelMeshCache::sync`'s drop pass) *deletes* the ledger entry rather than
// minting a new number, so the maximum stamp over a set is blind to the
// eviction of a non-maximal member. Both functions therefore walk the whole
// sequence and compare it element for element against the carried list —
// lengths, keys, `f64` origins and stamps — which sees an insertion, a removal
// and a reorder alike. It is O(tiles) scalar comparisons against O(tiles × res²)
// bytes.

/// Carry an unchanged [`RenderTerrain`] forward from the previous frame's scene
/// instead of rebuilding it — see the block above for the class.
///
/// `prev` is last frame's list, which the caller has taken out of the scene; a
/// match is **removed** from it, so the leftovers at the end of a projection are
/// exactly the terrains that left the scene. Returns `None` when anything the
/// projection would produce differs, in which case the caller projects normally.
///
/// Everything a projection derives is compared: the identity and grid
/// (`id`, `tile_resolution`, `meters_per_sample`), the authored surface
/// (`layers`, `macro_variation`, `biome_palette`) and the whole tile sequence in
/// the order it is emitted — level 0 ascending, then the coarse pyramid
/// ascending, which is the order both hosts build it in and the order this walks.
// Eight, and the alternative is worse: a struct grouping the six scalars would
// be constructed at each of the two mirrored call sites and destructured here,
// which is the same six values written twice more with a name in between. The
// precedent is `project_scene_full`'s and `VsmRaster::record`'s.
#[allow(clippy::too_many_arguments)]
pub fn take_unchanged_terrain(
    prev: &mut Vec<RenderTerrain>,
    id: u64,
    tile_resolution: u32,
    meters_per_sample: f64,
    layers: &[RenderTerrainLayer; 4],
    macro_variation: f32,
    biome_palette: &[[f32; 4]],
    tiles: impl Iterator<Item = (TerrainTileKey, DVec3, u64)>,
) -> Option<RenderTerrain> {
    let at = prev.iter().position(|t| t.id == id)?;
    // The scalars first, so a layer edit or a grid change short-circuits before
    // the tile walk rather than after it.
    if prev[at].tile_resolution != tile_resolution
        || prev[at].meters_per_sample != meters_per_sample
        || prev[at].macro_variation != macro_variation
        || prev[at].layers != *layers
        || prev[at].biome_palette != biome_palette
        || !prev[at].tiles_match(tiles)
    {
        return None;
    }
    Some(prev.remove(at))
}

/// Carry an unchanged [`RenderVoxelVolume`] forward from the previous frame's
/// scene instead of rebuilding it — the voxel twin of
/// [`take_unchanged_terrain`], see the block above for the class.
///
/// `chunks` yields `(key, world origin, stamp)` per resident chunk, in the order
/// the projection emits them (ascending [`VoxelChunkKey`]).
///
/// The carried volume keeps the **seam terms** [`apply_seam`] wrote into it last
/// frame. That is safe precisely because `apply_seam` writes every vertex it
/// visits (including its refusals) and clears when there is nothing to sample —
/// so re-running it produces the same bytes on a carried volume as on a fresh
/// one, and a caller that skips it because nothing changed keeps terms that are
/// still correct. Neither property was true before this wave, which is why they
/// are stated on `apply_seam` itself as well.
pub fn take_unchanged_voxel(
    prev: &mut Vec<RenderVoxelVolume>,
    id: u64,
    layers: &[RenderTerrainLayer; 4],
    chunks: impl Iterator<Item = (VoxelChunkKey, DVec3, u64)>,
) -> Option<RenderVoxelVolume> {
    let at = prev.iter().position(|v| v.id == id)?;
    if prev[at].layers != *layers {
        return None;
    }
    let mut walked = 0usize;
    let matched = chunks.into_iter().all(|(key, origin, version)| {
        let ok = prev[at].chunks.get(walked).is_some_and(|c| {
            version != 0 && c.version == version && c.key == key && c.origin == origin
        });
        walked += 1;
        ok
    });
    if !matched || walked != prev[at].chunks.len() {
        return None;
    }
    Some(prev.remove(at))
}

/// Does a voxel surface with normal `voxel_n` *continue* a heightfield whose
/// surface normal there is `terrain_n`? — the mask-free half of the poison rule
/// (see [`apply_seam`]).
///
/// Strictly positive, not `>= 0`: a perpendicular surface (a cave wall against
/// flat ground) is refused. Blending it would smear hillside down a vertical face
/// for the band's whole width, which is the loudest form of the artefact this
/// guards, and a wall is not a continuation of the ground it is cut into.
#[inline]
fn continues_surface(voxel_n: [f32; 3], terrain_n: [f32; 3]) -> bool {
    voxel_n[0] * terrain_n[0] + voxel_n[1] * terrain_n[1] + voxel_n[2] * terrain_n[2] > 0.0
}

/// What the heightfield looks like at one world `(x, z)`, for the P21.2 voxel
/// seam blend — the output of [`RenderTerrain::seam_sample`].
///
/// Deliberately the *resolved* surface (a normal, a height, one blended colour)
/// rather than the inputs it was resolved from. The consumer is a vertex
/// attribute, and handing four weights plus a palette across that boundary would
/// mean the voxel shader carrying the terrain's layers — a second place for the
/// blend to be implemented, and a second place for it to be wrong.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SeamSample {
    /// Unit heightfield normal (+Y strictly positive).
    pub normal: [f32; 3],
    /// World surface height (`f64`, the world-space precision doctrine).
    pub height: f64,
    /// Splat-blended linear albedo.
    pub albedo: [f32; 3],
    /// Splat-blended perceptual roughness.
    pub roughness: f32,
}

impl SeamSample {
    /// Pack into the two [`RenderVoxelVertex`] seam attributes, given the
    /// vertex's own world position (only its Y matters — the shader measures the
    /// band as `|world.y - height|`).
    ///
    /// The height rides as `f32` because it is only ever used as a *difference*
    /// against an already render-local vertex Y; the `f64` above is the world
    /// value, and the projector subtracts the floating origin before calling this.
    pub fn pack(&self, origin_y: f64) -> ([f32; 4], [f32; 4]) {
        (
            [
                self.normal[0],
                self.normal[1],
                self.normal[2],
                (self.height - origin_y) as f32,
            ],
            [
                self.albedo[0],
                self.albedo[1],
                self.albedo[2],
                self.roughness,
            ],
        )
    }
}

impl RenderTerrainTile {
    /// Words per packed hole-mask row: `ceil(resolution / 32)`. Zero-width for a
    /// degenerate resolution, which cannot happen through a projector but keeps
    /// the arithmetic total.
    #[inline]
    pub fn hole_words_per_row(resolution: u32) -> u32 {
        resolution.div_ceil(32)
    }

    /// `true` when **some** sample of this tile is holed — the cheap gate the GPU
    /// cache takes before sizing a hole texture at all.
    #[inline]
    pub fn has_holes(&self) -> bool {
        self.holes.iter().any(|&w| w != 0)
    }

    /// Is sample `(i, j)` holed? An empty mask reads `false` everywhere — the
    /// sparse default, projected intact. Out-of-range words read `false` too,
    /// matching the shader, whose `textureLoad` past the texture returns zero.
    #[inline]
    pub fn is_hole(&self, resolution: u32, i: u32, j: u32) -> bool {
        if self.holes.is_empty() {
            return false;
        }
        let stride = Self::hole_words_per_row(resolution) as usize;
        match self.holes.get(j as usize * stride + (i >> 5) as usize) {
            Some(word) => word & (1u32 << (i & 31)) != 0,
            None => false,
        }
    }

    /// The splat weight at sample `(i, j)`, clamping out-of-range indices to the
    /// edge. The projected `weights` vec is always dense (the projector expands
    /// the source's sparse default), so this never resolves a default itself —
    /// but an empty vec still answers, uniformly layer 0, rather than panicking on
    /// a projection that forgot.
    #[inline]
    pub fn weight_sample(&self, resolution: u32, i: u32, j: u32) -> [u8; 4] {
        if self.weights.is_empty() {
            return [255, 0, 0, 0];
        }
        let r = resolution.max(1);
        let idx = (j.min(r - 1) * r + i.min(r - 1)) as usize;
        self.weights.get(idx).copied().unwrap_or([255, 0, 0, 0])
    }
}

/// One terrain splat material layer (P10.4), projected from the ECS
/// `TerrainLayer`. `tex_scale` is world metres per procedural detail-grain tile.
/// (Layer texture GUIDs are deferred — the viewport can't upload asset textures
/// yet; the shader proves the blend with albedo + procedural triplanar grain.)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderTerrainLayer {
    /// Linear albedo (rgba).
    pub albedo: [f32; 4],
    /// Perceptual roughness `[0, 1]`.
    pub roughness: f32,
    /// World metres per detail-grain tile.
    pub tex_scale: f32,
    /// **The layer's shared, tileable PBR material** (Wave T — the texture
    /// document's §3 B, *"store blend weights in a virtual texture mask and
    /// sample shared, tileable PBR materials"*).
    ///
    /// [`VtTextureSet::NONE`] — the default, and what every projector writes
    /// today — is exactly the flat-colour layer this has always been: the
    /// terrain fragment shader's textured branch is not taken and the arithmetic
    /// is instruction-for-instruction what it was, which is what keeps the
    /// committed terrain goldens byte-stable.
    ///
    /// When it names textures, the layer samples them at `world.xz / tex_scale`
    /// and the four layers' samples are weight-blended by the same per-sample
    /// splat mask that already blends their colours. Because virtual textures
    /// are deduplicated by GUID, four terrains sharing one rock material pay for
    /// it once — which is the "minimal unique texture storage" half of the
    /// document's claim, and it is structural rather than a policy.
    ///
    /// **The authoring field is not here.** `inf_ecs::TerrainLayer` — the
    /// persisted, scene-schema half — carries no texture reference yet, and the
    /// scene schema is frozen for this wave; see
    /// `docs/memos/wave-t-textures-disposition.md`, item T26. A host binds this
    /// directly today.
    pub vt: VtTextureSet,
}

impl Default for RenderTerrainLayer {
    fn default() -> Self {
        Self {
            albedo: [0.35, 0.35, 0.35, 1.0],
            roughness: 0.9,
            tex_scale: 8.0,
            vt: VtTextureSet::NONE,
        }
    }
}

/// The renderer's terrain input: the **resident** page set of a paged heightfield
/// projected from the ECS `Terrain` component. The
/// [`TerrainNode`](crate::passes::terrain) uploads a per-tile R32Float height
/// texture, a per-tile Rgba8Unorm splat-weight texture and a per-tile R8Uint
/// biome-id texture (all cached, gated by each tile's own
/// [`version`](RenderTerrainTile::version)) and assembles concentric clipmap LOD
/// rings around the camera each frame, blending the four `layers` by the splat
/// weights.
///
/// ## Residency (P16.3b1)
///
/// `tiles` is whatever the projector handed over — it is **not** assumed to be a
/// complete terrain. A missing tile simply produces no patch (a hole, exactly as
/// an unauthored tile always did), and coarse (LOD ≥ 1) pyramid tiles may ride
/// beside the level-0 ones to cover the outer rings. The renderer never invents a
/// want: it faithfully draws the set it is given (camera-driven residency
/// selection lives above this DTO).
///
/// There is deliberately **no whole-terrain version counter** (P16.3b1 removed
/// it): the per-tile stamps are strictly more precise, and a single global
/// counter is exactly the field a projector forgets to bump — the shipped player
/// pinned it to a constant, which would have frozen the GPU cache the moment
/// residency started changing. The terrain-wide GPU uploads left (the splat
/// material uniform and the P19.2 biome palette) are each gated by comparing the
/// packed value, which cannot desync.
///
/// ## Multi-terrain (P16.6)
///
/// A scene carries **N** of these ([`RenderScene::terrains`]); each is an
/// independent heightfield with its own grid, its own residency and its own splat
/// material. [`id`](Self::id) is what keeps their GPU caches apart — see there.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RenderTerrain {
    /// Stable identity of the terrain this projection describes (P16.6).
    ///
    /// The per-tile GPU texture cache and the splat-material uniform are keyed by
    /// `(id, tile key)` / `id`, so two terrains whose grids share a coordinate
    /// cannot overwrite each other's pages, and a terrain that leaves the scene
    /// releases exactly its own resources. Both projectors derive it from the
    /// terrain entity's `Guid` ([`terrain_id_from_guid`]); `0` is the "unkeyed"
    /// value a single-terrain caller (and every pre-P16.6 test) leaves at its
    /// default, which is why single-terrain scenes stay byte-identical.
    ///
    /// **Distinct terrains in one scene must carry distinct ids.** The renderer
    /// cannot check that — two projections claiming one id simply share a cache
    /// slot and fight over it.
    pub id: u64,
    /// Samples per tile side (the height/weight-texture dimension). Terrain-wide:
    /// a coarse level keeps the resolution and doubles the spacing.
    pub tile_resolution: u32,
    /// World units between samples **at level 0**.
    pub meters_per_sample: f64,
    /// The resident tiles, ascending by [`TerrainTileKey`] (level 0 first, then
    /// each coarse level) — a deterministic upload/draw order.
    pub tiles: Vec<RenderTerrainTile>,
    /// The four splat material layers the per-sample weights blend (P10.4).
    pub layers: [RenderTerrainLayer; 4],
    /// Amplitude of the large-scale fBm albedo modulation (`0` = off).
    pub macro_variation: f32,
    /// Biome id → debug colour for the **Biomes** view mode (P19.2), **indexed by
    /// id**: `biome_palette[id]` is the colour a sample carrying that id is tinted
    /// with. The array position IS the id — never an ordinal into the level's
    /// biome list, which is a sparse set of authored ids. Slot 0 is the reserved
    /// "unassigned" colour, and an id the set never defined reads it too;
    /// `inf_terrain::BiomeSet::palette` builds exactly that shape.
    ///
    /// **May be empty**: a terrain with no `BiomeSet` bound — or a projector that
    /// cannot resolve one — passes `Vec::new()` and the renderer pads every slot
    /// with the unassigned colour, so the mode still draws something honest
    /// (uniform neutral grey = "no biome vocabulary here") instead of failing.
    /// Ids past the end of a short palette pad the same way.
    pub biome_palette: Vec<[f32; 4]>,
}

impl RenderTerrain {
    /// Whether this terrain's tile list is exactly the sequence `tiles`
    /// describes — same length, same keys, same `f64` origins, same stamps, and
    /// no stamp of `0` (the "not in the ledger" value, which is a forced miss on
    /// the same rule the GPU tile cache applies).
    ///
    /// The whole sequence, element for element, rather than a fold: a removal
    /// *deletes* a stamp from the source's ledger instead of minting a new one,
    /// so a maximum over the set is blind to the eviction of a non-maximal
    /// member. See the memo block above [`take_unchanged_terrain`].
    fn tiles_match(&self, tiles: impl Iterator<Item = (TerrainTileKey, DVec3, u64)>) -> bool {
        let mut walked = 0usize;
        let ok = tiles.into_iter().all(|(key, origin, version)| {
            let ok = self.tiles.get(walked).is_some_and(|t| {
                version != 0 && t.version == version && t.key == key && t.origin == origin
            });
            walked += 1;
            ok
        });
        ok && walked == self.tiles.len()
    }

    /// World edge length of one **level-0** tile: `(resolution − 1) · mps`. Also
    /// the unit the clipmap ring thresholds are scaled by.
    pub fn tile_span(&self) -> f64 {
        (self.tile_resolution.max(2) as f64 - 1.0) * self.meters_per_sample
    }

    /// World edge length of a tile at asset LOD `lod`: `tile_span · 2^lod` (the
    /// metres-per-sample doubling, at a constant sample count).
    pub fn tile_span_at(&self, lod: u32) -> f64 {
        self.tile_span() * (1u64 << lod.min(62)) as f64
    }

    /// **The P21.2 seam sample**: the heightfield's surface normal, world height
    /// and splat-blended material at world `(x, z)`, or `None` where this
    /// projection has no level-0 surface there — no resident tile, or a tile whose
    /// sample is holed.
    ///
    /// This is what a projector calls per voxel vertex to fill
    /// [`RenderVoxelVertex::seam_nh`] / [`seam_albedo`](RenderVoxelVertex::seam_albedo),
    /// and it lives **here**, on the already-projected terrain, for a specific
    /// reason: the alternative is for each of the two host projectors to sample
    /// `inf_terrain` and re-implement the splat blend, which would make the seam
    /// colour a function of *which host is drawing* — exactly the class of
    /// divergence the mirrored-pair discipline exists to prevent. One
    /// implementation, over the DTO both hosts already built, cannot diverge.
    ///
    /// ## THE RESIDENCY FLOOR, and not the finest resident page (P21.2 audit)
    ///
    /// This reads **only** the projection's coarsest asset level
    /// ([`max_lod`](Self::max_lod)) — the same restriction, for the same reason,
    /// that [`gi::voxelization_tiles`](crate::gi::voxelization_tiles) puts on the
    /// GI voxelizer. [`tiles`](Self::tiles) is the streamer's **camera-driven**
    /// working set, so a seam resolved against "whichever level-0 page happens to
    /// be paged in" makes a voxel surface's albedo, roughness and shading normal —
    /// and therefore its **lighting** — a function of where the camera has *been*.
    /// That is the P18 law (`camera-driven residency never feeds lighting`), and
    /// the first cut of this function broke it: level 0 is exactly the part of the
    /// cut that pages, so walking away from a cave mouth silently turned its blend
    /// off.
    ///
    /// The coarsest level is the terrain's always-resident root:
    /// `inf_terrain::TerrainStreamer` pins it as its **residency floor** and
    /// reseeds the published cut from it, which is what makes `max_lod` a property
    /// of the *asset* rather than of the camera (see `TerrainStreamer::
    /// residency_floor`). An inline, non-streamed terrain has `max_lod() == 0`, so
    /// for every such terrain — every unit test, every golden, every level that has
    /// not been through the Terrain Import wizard — this is byte-identical to
    /// sampling level 0, because level 0 *is* the coarsest level.
    ///
    /// **Two consequences on a streamed terrain, stated rather than discovered.**
    /// A coarse pyramid page carries downsampled heights and **no painted
    /// weights**, so the seam colour there is the uniform layer-0 blend rather than
    /// the painted one — the identical fidelity trade the GI voxelizer already
    /// took, and the right way round: a slightly flat seam everywhere beats a
    /// differently-coloured one depending on where the player walked. And a coarse
    /// page carries **no hole mask** (`inf_terrain::pyramid::downsample_block`
    /// reduces heights, biome ids and data maps and nothing else — pinned by
    /// `a_coarse_page_carries_no_hole_mask`), which is why
    /// [`seam_holes_are_known`](Self::seam_holes_are_known) exists and why
    /// [`apply_seam`] carries a mask-free veto for the case where it answers
    /// `false`.
    ///
    /// The hole test below is the **same poison rule** the fragment shader and
    /// `inf_terrain::TerrainData::height_at` apply: one holed corner of the
    /// bilinear cell removes it. Which is what makes a cave mouth work — the
    /// vertices *inside* the hole get no seam, the ones just outside it do, and
    /// the band falls off across the rim.
    pub fn seam_sample(&self, x: f64, z: f64) -> Option<SeamSample> {
        let res = self.tile_resolution.max(2);
        let lod = self.max_lod();
        let mps = self.meters_per_sample * (1u64 << lod.min(62)) as f64;
        let span = self.tile_span_at(lod);
        let coord = ((x / span).floor() as i32, (z / span).floor() as i32);
        let tile = self
            .tiles
            .iter()
            .find(|t| t.key.lod == lod && t.key.coord == coord)?;

        let u = ((x - coord.0 as f64 * span) / mps).clamp(0.0, (res - 1) as f64);
        let v = ((z - coord.1 as f64 * span) / mps).clamp(0.0, (res - 1) as f64);
        let (i0, j0) = (u.floor() as u32, v.floor() as u32);
        let (i1, j1) = ((i0 + 1).min(res - 1), (j0 + 1).min(res - 1));
        if tile.is_hole(res, i0, j0)
            || tile.is_hole(res, i1, j0)
            || tile.is_hole(res, i0, j1)
            || tile.is_hole(res, i1, j1)
        {
            return None;
        }
        let (fx, fz) = (u - i0 as f64, v - j0 as f64);
        let h = |i: u32, j: u32| tile.heights[(j * res + i) as usize] as f64;
        let lerp2 = |a: f64, b: f64, c: f64, d: f64| {
            let x0 = a + (b - a) * fx;
            let x1 = c + (d - c) * fx;
            x0 + (x1 - x0) * fz
        };
        let height = tile.origin.y + lerp2(h(i0, j0), h(i1, j0), h(i0, j1), h(i1, j1));

        // Central differences on the sample lattice — the same gradient the
        // terrain fragment shader takes, so the two normals agree at the seam.
        let e = mps as f32;
        let im = i0.saturating_sub(1);
        let ip = (i0 + 1).min(res - 1);
        let jm = j0.saturating_sub(1);
        let jp = (j0 + 1).min(res - 1);
        let dx = (h(ip, j0) - h(im, j0)) as f32 / ((ip - im).max(1) as f32 * e);
        let dz = (h(i0, jp) - h(i0, jm)) as f32 / ((jp - jm).max(1) as f32 * e);
        let n = {
            let v = [-dx, 1.0, -dz];
            let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-6);
            [v[0] / l, v[1] / l, v[2] / l]
        };

        // Splat blend, against THIS terrain's layers.
        let w = |i: u32, j: u32| tile.weight_sample(res, i, j);
        let wq = [w(i0, j0), w(i1, j0), w(i0, j1), w(i1, j1)];
        let mut weights = [0.0f32; 4];
        for k in 0..4 {
            let a = wq[0][k] as f32 / 255.0;
            let b = wq[1][k] as f32 / 255.0;
            let c = wq[2][k] as f32 / 255.0;
            let d = wq[3][k] as f32 / 255.0;
            weights[k] = lerp2(a as f64, b as f64, c as f64, d as f64) as f32;
        }
        let sum: f32 = weights.iter().sum();
        if sum > 1e-4 {
            for w in &mut weights {
                *w /= sum;
            }
        } else {
            weights = [1.0, 0.0, 0.0, 0.0];
        }
        let mut albedo = [0.0f32; 3];
        let mut roughness = 0.0f32;
        for (k, layer) in self.layers.iter().enumerate() {
            for (c, out) in albedo.iter_mut().enumerate() {
                *out += weights[k] * layer.albedo[c];
            }
            roughness += weights[k] * layer.roughness;
        }

        Some(SeamSample {
            normal: n,
            height,
            albedo,
            roughness: roughness.clamp(0.04, 1.0),
        })
    }

    /// The coarsest asset LOD present in the projection (`0` for a level-0-only
    /// terrain — every inline, non-streamed terrain).
    pub fn max_lod(&self) -> u32 {
        self.tiles.iter().map(|t| t.key.lod).max().unwrap_or(0)
    }

    /// Whether the level [`seam_sample`](Self::seam_sample) reads — the residency
    /// floor, [`max_lod`](Self::max_lod) — carries the per-sample hole mask.
    ///
    /// `true` exactly when that level is 0, because holes live **only** on
    /// authored level-0 tiles: `inf_terrain::pyramid::downsample_block` reduces
    /// heights, biome ids and erosion data maps into a coarse page and carries no
    /// hole mask upward at all (the P21.2 remainder, pinned by
    /// `a_coarse_page_carries_no_hole_mask`).
    ///
    /// The consequence is the one [`apply_seam`] acts on: where this is `false`
    /// the poison rule inside `seam_sample` **cannot fire**, so a cave ceiling
    /// under a mouth would answer "there is hillside here" and wear the hillside's
    /// material. Not knowing about a hole is not the same as there being none, and
    /// this is the predicate that says which of the two the caller is holding.
    pub fn seam_holes_are_known(&self) -> bool {
        self.max_lod() == 0
    }

    /// The projected `(key → change stamp)` ledger, in tile order — the input the
    /// GPU texture cache gates its uploads on.
    pub fn tile_versions(&self) -> impl Iterator<Item = (TerrainTileKey, u64)> + '_ {
        self.tiles.iter().map(|t| (t.key, t.version))
    }
}

/// Fold a terrain entity's 128-bit `Guid` into the 64-bit
/// [`RenderTerrain::id`] both projectors use (P16.6).
///
/// XOR-folding the halves, then forcing a non-zero result: `0` is reserved for
/// "unkeyed" (the default a single-terrain caller leaves in place), so a GUID that
/// happens to fold to zero must not silently become it. Pure, so the editor
/// viewport and the shipped player derive the same id for the same entity — which
/// is what makes a PIE-vs-shipping comparison of the projected scene meaningful.
#[inline]
pub fn terrain_id_from_guid(guid: u128) -> u64 {
    let folded = (guid as u64) ^ ((guid >> 64) as u64);
    if folded == 0 {
        1
    } else {
        folded
    }
}

/// A projected voxel chunk's identity: its integer position in the volume's
/// chunk grid (P21.1).
///
/// This is the renderer-local mirror of `inf_voxel::ChunkKey` — `inf-render` is
/// Ring 0 and deliberately does **not** depend on `inf-voxel`, exactly as it does
/// not depend on `inf-terrain` (see [`TerrainTileKey`]). The two documented
/// projectors map one onto the other, like every other scene DTO, which is what
/// keeps the meshed surface the renderer draws *triangle soup* rather than a
/// second copy of the SDF model.
///
/// A chunk is a fixed-size cube of the volume's grid, so the key is a plain 3D
/// lattice coordinate: unlike a terrain tile there is no LOD component, because a
/// voxel volume is a **local** extension of the heightfield (a cave, an
/// excavation, an overhang) rather than a paged world-scale surface — P21.1 meshes
/// every resident chunk at full resolution.
///
/// `Ord` is the **derived, natural field order** (`x`, then `y`, then `z`) rather
/// than a hand-written `(z, y, x)`: the projector hands chunks over in
/// `inf_voxel`'s `BTreeMap` order, which is that same derived field order, so the
/// two orders agree *by construction* instead of by a comment nobody re-checks.
/// A projection walked in key order is therefore also key-ascending here — the
/// order [`RenderVoxelVolume::chunks`] is documented to arrive in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct VoxelChunkKey {
    /// Chunk grid coordinate along world X.
    pub x: i32,
    /// Chunk grid coordinate along world Y (voxel volumes are genuinely 3D —
    /// this is the axis a heightfield tile key does not have).
    pub y: i32,
    /// Chunk grid coordinate along world Z.
    pub z: i32,
}

impl VoxelChunkKey {
    /// A key at an explicit lattice coordinate.
    #[inline]
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }
}

/// One meshed voxel-surface vertex (P21.1) — the output of the isosurface
/// extraction, handed to the [`VoxelNode`](crate::passes::voxel) as plain
/// triangle soup.
///
/// Positions are **chunk-local `f32` metres**: the owning
/// [`RenderVoxelChunk::origin`] carries the `f64` world anchor and the pass builds
/// a per-chunk model matrix from it against the frame's floating origin. That is
/// architecture rule 3 applied at the natural seam — a chunk is metres across, so
/// its interior needs no `f64` precision at all, and the one place the world's
/// magnitude enters is the anchor the origin is subtracted from.
///
/// Normals arrive **already normalized** (the mesher takes them from the SDF
/// gradient, which is what makes a voxel surface smooth without a smoothing pass);
/// the shader re-normalizes after interpolation, which is a different thing.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RenderVoxelVertex {
    /// Chunk-local position (metres, relative to the chunk's `origin`).
    pub pos: [f32; 3],
    /// Unit surface normal (chunk-local == world, chunks are unrotated).
    pub normal: [f32; 3],
    /// Splat-layer index `0..=3`, selecting into [`RenderVoxelVolume::layers`].
    ///
    /// **Categorical**, exactly like [`RenderTerrainTile::biomes`]: it names a
    /// layer, it is not a quantity, so it is passed to the fragment stage
    /// `@interpolate(flat)` — the midpoint of layers 1 and 3 is not layer 2.
    /// Values past `3` clamp in the shader rather than reading out of bounds, so a
    /// projector that grows a fifth material before the renderer does degrades to
    /// the last layer instead of to undefined behaviour.
    pub material: u32,
    /// **Seam (P21.2)**: the heightfield's unit surface normal at this vertex's
    /// world XZ in `xyz`, and the heightfield's world surface height in `w`.
    ///
    /// [`NO_SEAM`](RenderVoxelVertex::NO_SEAM) — all zeros — means "no
    /// heightfield over this point", and it is a sentinel that cannot be confused
    /// with data: a heightfield normal is the gradient of a single-valued
    /// function of `(x, z)`, so `y` is strictly positive for every real sample.
    /// The shader tests `y <= 0` and skips the blend entirely, which is what keeps
    /// a volume with no terrain over it byte-identical to its pre-P21.2 render.
    pub seam_nh: [f32; 4],
    /// **Seam (P21.2)**: the terrain's splat-blended albedo in `rgb` and its
    /// blended perceptual roughness in `a` at this vertex's world XZ, resolved
    /// against the **terrain's** layer palette (not this volume's) by
    /// [`RenderTerrain::seam_sample`]. Ignored when
    /// [`seam_nh`](Self::seam_nh) is the sentinel.
    pub seam_albedo: [f32; 4],
}

impl RenderVoxelVertex {
    /// The "no heightfield here" seam sentinel — see [`seam_nh`](Self::seam_nh).
    /// A projector with no terrain to sample writes this and the blend is off.
    pub const NO_SEAM: [f32; 4] = [0.0; 4];
}

/// One meshed chunk handed to the [`VoxelNode`](crate::passes::voxel) (P21.1).
///
/// The renderer never sees the SDF: it sees the triangles a chunk currently
/// extracts to, plus the stamp that says whether they changed. Everything about
/// *why* the surface has this shape — the density field, the brush history, the
/// meshing algorithm — lives in `inf-voxel` and stops at this boundary.
#[derive(Debug, Clone, PartialEq)]
pub struct RenderVoxelChunk {
    /// Grid position of this chunk within its volume.
    pub key: VoxelChunkKey,
    /// `f64` world position of this chunk's local origin — the anchor
    /// [`vertices`](Self::vertices) are relative to.
    pub origin: DVec3,
    /// The chunk's meshed surface vertices (chunk-local, see
    /// [`RenderVoxelVertex`]).
    pub vertices: Vec<RenderVoxelVertex>,
    /// Triangle list into [`vertices`](Self::vertices). Length must be a multiple
    /// of 3 and every index must be in range; a chunk that fails either is
    /// **dropped** by the cache planner rather than uploaded (see
    /// [`plan_chunk_cache`](crate::passes::voxel::plan_chunk_cache)) — a bad page
    /// never becomes silent geometry.
    pub indices: Vec<u32>,
    /// Inclusive chunk-local `(min, max)` AABB of [`vertices`](Self::vertices) —
    /// the frustum-cull bound. Projected rather than recomputed per frame because
    /// the mesher already knows it.
    pub bounds: ([f32; 3], [f32; 3]),
    /// The chunk's **monotone change stamp**. The GPU buffer cache re-uploads this
    /// chunk if — and only if — the stamp differs from the one its cached copy was
    /// built at. `0` means "no stamp" (a chunk the source could not version) and is
    /// treated conservatively as *always re-upload*, never as a cache hit — exactly
    /// like [`RenderTerrainTile::version`], and for the same reason: a source that
    /// cannot version its chunks must degrade to re-uploading, not to a stale
    /// frame.
    pub version: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Fracture debris (P22.3)
// ─────────────────────────────────────────────────────────────────────────────

/// One live chunk of a broken destructible, ready to draw.
///
/// # Why this is not a `MeshInstance`
///
/// A `MeshInstance` selects one of five built-in primitives; the mesh pass owns a
/// single static vertex buffer for them and has no door for custom geometry. A
/// fracture chunk *is* custom geometry — a Voronoi cell of an authored mesh — and
/// there are up to sixty-four of them per actor, each with its own solver-owned
/// pose. So it carries its triangles the way [`RenderVoxelChunk`] does, and its
/// material the way [`MeshInstance`] does: the two halves of the problem, taken
/// from the two places that already solved each.
///
/// # Chunk-local vertices against an `f64` anchor
///
/// [`vertices`](Self::vertices) are **chunk-local metres** relative to
/// [`translation`](Self::translation), which is the chunk's `f64` world centre of
/// mass. That is the floating-origin split every other world-space DTO in this
/// file makes, and here it is also what lets the vertex buffer be uploaded **once
/// per break** while the pose changes sixty times a second: a tumbling chunk
/// moves its instance, never its geometry.
#[derive(Debug, Clone, PartialEq)]
pub struct RenderFractureChunk {
    /// The destructible actor this chunk belongs to — the same fold of the
    /// entity `Guid` [`RenderVoxelVolume::id`] uses, so a PIE-vs-shipping diff
    /// matches actors up by identity rather than by list position.
    pub entity: u64,
    /// The chunk's index in its `.inf_fracture`.
    pub chunk: u32,
    /// `f64` world position of the chunk's centre of mass — the anchor
    /// [`vertices`](Self::vertices) are relative to.
    pub translation: DVec3,
    /// The chunk's world orientation. Identity while it is still attached.
    pub rotation: Quat,
    /// Chunk-local surface vertices.
    pub vertices: Vec<RenderFractureVertex>,
    /// Triangle list into [`vertices`](Self::vertices). Length must be a multiple
    /// of 3 and every index in range; a chunk that fails either is **dropped** by
    /// the cache planner rather than uploaded — bad geometry never becomes silent
    /// triangles. (The [`RenderVoxelChunk::indices`] rule, verbatim.)
    pub indices: Vec<u32>,
    /// Linear base colour, rgba.
    pub color: [f32; 4],
    /// PBR metallic in `[0, 1]`.
    pub metallic: f32,
    /// PBR roughness in `[0, 1]`.
    pub roughness: f32,
    /// Linear emissive colour, rgb.
    pub emissive: [f32; 3],
    /// The owning actor's **fracture generation** — a monotone stamp that moves
    /// when a chunk detaches or is reclaimed and **not** when one merely tumbles.
    ///
    /// That distinction is the whole design: a pose changes every step and the
    /// geometry never does, so keying the upload on the pose would re-upload every
    /// chunk of every collapsing building sixty times a second. `0` means "no
    /// stamp" and is treated conservatively as *always re-upload*, never as a
    /// cache hit — [`RenderVoxelChunk::version`]'s rule, for its reason.
    pub version: u64,
}

/// A fracture chunk's vertex: position + normal + uv, chunk-local.
///
/// Deliberately the **same 32-byte layout** as
/// [`crate::passes::mesh::MeshVertex`], so the fracture pass can bind
/// `mesh.wgsl`'s own vertex stage against a per-chunk buffer instead of shipping
/// a second PBR shader that would then have to be kept in step with it. The
/// `uv` followed `MeshVertex`'s in P26.5 and it is a *layout* obligation before
/// it is a feature: a `RenderFractureChunk` carries no
/// [`VtTextureSet`], so nothing samples with it— but a
/// buffer whose stride disagrees with the pipeline's is not a subtle bug, it is
/// every chunk drawn from the wrong bytes. The projectors fill it from the
/// `.inf_fracture` chunk's own `inf_mesh::MeshVertex::uv`, which the Voronoi cut
/// carries for the hull faces and defaults on the interior ones.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RenderFractureVertex {
    /// Chunk-local position, metres.
    pub pos: [f32; 3],
    /// Unit normal.
    pub normal: [f32; 3],
    /// Texture coordinate, `@location(2)` — the chunk's own, unrotated.
    pub uv: [f32; 2],
}

/// A voxel volume's meshed, resident chunk set (P21.1) — the renderer's
/// volumetric-terrain input, projected from the ECS `VoxelVolume` component.
///
/// ## Residency
///
/// `chunks` is whatever the projector handed over — it is **not** assumed to be a
/// complete volume. A missing chunk simply produces no geometry (a hole, exactly
/// as an unmeshed region always was); the renderer never invents a want. This is
/// the same contract [`RenderTerrain::tiles`] carries, and for the same reason:
/// residency selection is a decision about the world, and it lives above this DTO.
///
/// ## Relationship to the heightfield
///
/// A volume *locally extends* the heightfield rather than replacing it — a cave
/// mouth, an excavated pit, an overhang the 2.5D surface cannot express. That is
/// why [`layers`](Self::layers) is deliberately the SAME
/// [`RenderTerrainLayer`] type the terrain splat uses, at deliberately the same
/// indices: a cave mouth must shade continuously into the hillside it opens out
/// of, and two independently-authored material vocabularies could not.
///
/// Seam *blending* across that boundary (and shadow/GI participation, and the
/// depth prepass) is **P21.2** — P21.1 draws the surfaces with their own simple
/// lit pass and says so in `shaders/voxel.wgsl`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RenderVoxelVolume {
    /// Stable identity of the volume this projection describes.
    ///
    /// The per-chunk GPU buffer cache is keyed by `(id, chunk key)`, so two
    /// volumes whose grids share a coordinate cannot overwrite each other's
    /// chunks, and a volume that leaves the scene releases exactly its own
    /// buffers. Both projectors derive it from the volume entity's `Guid`
    /// ([`terrain_id_from_guid`] — the same fold, so a voxel volume and a terrain
    /// are identified by the same rule); `0` is the "unkeyed" value a
    /// single-volume caller (and every unit test) leaves at its default.
    ///
    /// **Distinct volumes in one scene must carry distinct ids.** The renderer
    /// cannot check that — two projections claiming one id simply share a cache
    /// slot and fight over it.
    pub id: u64,
    /// The resident meshed chunks, ascending by [`VoxelChunkKey`] — a
    /// deterministic upload/draw order (see the key's `Ord` note).
    pub chunks: Vec<RenderVoxelChunk>,
    /// The four splat material layers a vertex's
    /// [`material`](RenderVoxelVertex::material) index selects.
    ///
    /// Deliberately the SAME [`RenderTerrainLayer`] the heightfield uses, and
    /// deliberately indices `0..=3` **aligned with the terrain splat layers** — a
    /// cave mouth must shade continuously into the hillside it opens out of, which
    /// it cannot do if layer 2 means "rock" on one side of the seam and "moss" on
    /// the other. (`tex_scale` is carried but unused: the voxel shader has no
    /// triplanar detail grain — P21.2 closed the seam through the per-vertex blend
    /// below instead, which is a different mechanism.)
    pub layers: [RenderTerrainLayer; 4],
    /// **Seam blend band (P21.2), in metres.** A voxel fragment within this
    /// distance of the heightfield surface mixes toward the terrain's own albedo,
    /// roughness and normal there; `0` disables the blend for this volume.
    ///
    /// A width, not a switch, because the right value is a property of the
    /// content: a cave mouth cut into 1 m-per-sample ground wants a band about a
    /// metre or two wide — wide enough that the transition is not a visible line,
    /// narrow enough that the cave interior does not turn into hillside. The
    /// projector defaults it to [`DEFAULT_SEAM_BAND_M`].
    ///
    /// What it buys and what it does not is stated in `voxel.wgsl`: material and
    /// normal continuity, **not** geometric welding.
    pub seam_band_m: f32,
}

/// Default width of the P21.2 voxel-to-heightfield seam blend band, in metres.
///
/// Two metres: about two samples of a default 1 m terrain, so the band spans a
/// few pixels of hillside at any reasonable viewing distance and reads as a
/// gradient rather than a boundary, while a player standing in a cave four metres
/// under the surface is entirely outside it.
pub const DEFAULT_SEAM_BAND_M: f32 = 2.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SkyParams {
    /// Linear colors.
    pub zenith: [f32; 3],
    pub horizon: [f32; 3],
    pub ground: [f32; 3],
}

impl Default for SkyParams {
    fn default() -> Self {
        // Editor-dark sky tuned to the infinity-dark theme.
        Self {
            zenith: [0.012, 0.021, 0.038],
            horizon: [0.055, 0.081, 0.120],
            ground: [0.009, 0.011, 0.015],
        }
    }
}

/// The sun this renderer shipped with from Phase 2 to Phase 16, as a
/// compile-time `camera::SUN_DIR` constant. **P17.1 retired the constant**: the
/// direction is now projected from the scene's `TimeOfDay` + `SkyAtmosphere`
/// components, and this value survives only as [`SunParams::default`] — the
/// fallback a scene with no time-of-day authority (every unit test, every
/// pre-P17.1 golden, a bare `RenderScene::default()`) still renders with, so
/// those pixels are byte-identical to what they always were.
///
/// Kept **un-normalized**, exactly as the deleted constant was: every one of its
/// three call sites wrote `SUN_DIR.normalize()`, and
/// [`SunParams::unit_direction`] does the same multiplication on the same bits.
/// Hand-transcribing the normalized triple would risk a 1-ULP drift that moves
/// every pre-P17.1 golden, so the arithmetic is reproduced rather than the
/// result — pinned by `default_sun_is_the_retired_constant_normalized`, which
/// compares raw `to_bits()`.
pub const DEFAULT_SUN_DIR: Vec3 = Vec3::new(0.45, 0.75, 0.3);

/// The scene's **sun and moon** (P17.1) — direction, colour and intensity for
/// each, projected from the `TimeOfDay` + `SkyAtmosphere` component pair by both
/// scene builders (`inf_viewport::host` and `inf_player::render`).
///
/// This is the renderer's single source of a sun direction. It feeds:
///
/// * `ViewUniforms::sun_dir` — read by the sky gradient's glow, the terrain
///   shader, and the mesh/skinned/vgeom shaders' no-light fallback;
/// * the CSM caster fallback ([`crate::passes::shadow`]) when a scene has no
///   directional light at all;
/// * the GI sun fallback ([`crate::passes::gi`]), so probe radiance tracks the
///   time of day.
///
/// A scene that authors its own directional light still wins over the fallbacks —
/// exactly the precedence the renderer had before P17.1, with the constant
/// swapped for a projected value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SunParams {
    /// Unit direction **toward** the sun (same convention as
    /// [`RenderLight::direction`]).
    pub direction: Vec3,
    /// Linear sun colour.
    pub color: [f32; 3],
    /// Sun radiant-intensity multiplier.
    pub intensity: f32,
    /// Unit direction **toward** the moon.
    pub moon_direction: Vec3,
    /// Linear moon colour.
    pub moon_color: [f32; 3],
    /// Moon radiant-intensity multiplier (used while the sun is below the
    /// horizon).
    pub moon_intensity: f32,
    /// Lunar phase, `[0, 1)` — `0` new, `0.5` full (P17.2). Only the sky pass
    /// reads it, to place the moon disc's terminator; it lights nothing.
    pub moon_phase: f32,
}

impl Default for SunParams {
    /// The retired [`DEFAULT_SUN_DIR`] at the intensity the shaders' hard-coded
    /// fallback used (`3.0`), so a scene that never opts into time of day renders
    /// exactly the pixels it rendered before P17.1. `direction` is the raw
    /// constant; consumers read [`unit_direction`](SunParams::unit_direction).
    fn default() -> Self {
        Self {
            direction: DEFAULT_SUN_DIR,
            color: [1.0, 1.0, 1.0],
            intensity: 3.0,
            // Straight down — a moon nobody projected is a moon nobody sees. The
            // renderer never reads it unless the projection filled it in.
            moon_direction: Vec3::NEG_Y,
            moon_color: [0.62, 0.72, 1.0],
            moon_intensity: 0.0,
            // Full — the phase a projector that never filled it in would want if
            // anything ever drew this moon, which nothing does at intensity 0.
            moon_phase: 0.5,
        }
    }
}

impl SunParams {
    /// The sun direction as a unit vector — what every consumer actually reads.
    ///
    /// A projector that hands over a degenerate (zero / non-finite) vector gets
    /// the retired default rather than a `NaN` uniform, which would otherwise
    /// black out the sky glow and the CSM cascade fit.
    pub fn unit_direction(&self) -> Vec3 {
        let d = self.direction.normalize_or_zero();
        if d.length_squared() > 0.5 {
            d
        } else {
            DEFAULT_SUN_DIR.normalize()
        }
    }

    /// The moon direction as a unit vector; degenerate input reads as "straight
    /// down", i.e. a moon below the world.
    pub fn unit_moon_direction(&self) -> Vec3 {
        let d = self.moon_direction.normalize_or_zero();
        if d.length_squared() > 0.5 {
            d
        } else {
            Vec3::NEG_Y
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct RenderScene {
    pub instances: Vec<MeshInstance>,
    /// Bind-space geometry for skinned meshes (P11.1), referenced by
    /// [`SkinnedInstance::mesh`]. Empty ⇒ the skinned pass is a no-op (every
    /// pre-P11 scene stays byte-identical).
    ///
    /// **`Arc`, since P18.3, and for two reasons that are really one.** A host
    /// rebuilds this list on every projection; a character's bind-space stream is
    /// megabytes, so an owned `SkinnedMeshData` meant a full CPU copy per document
    /// change *and* — because the pass keyed its uploads on `scene.version` — a
    /// full GPU re-upload of geometry that had not moved. Sharing the buffer lets
    /// the pass cache by **pointer identity** instead: same `Arc`, same GPU
    /// buffers, no copy and no upload.
    ///
    /// **Palettes take the same discipline since NPC1b** (see
    /// [`SkinnedInstance::palette`]). The original sentence here said they are
    /// "still rebuilt every projection, which is correct — they are the part that
    /// actually changes", and that is true of a *posed* character and false of a
    /// crowd: a thousand agents that evaluate no pose share one rest palette, and
    /// rebuilding it a thousand times is a thousand copies of one answer.
    pub skinned_meshes: Vec<std::sync::Arc<SkinnedMeshData>>,
    /// GPU-skinned instances (P11.1). Each names a joint palette (shared, by
    /// `Arc`) and how it casts; drawn by the skinned mesh pass after the rigid
    /// mesh pass, into the same targets.
    pub skinned: Vec<SkinnedInstance>,
    /// Virtualized-geometry (meshlet DAG) assets referenced by
    /// [`vgeom_instances`](Self::vgeom_instances) (P13.1b). Each is uploaded to GPU
    /// storage buffers once (cached by [`VgeomAsset::id`]). Empty ⇒ the meshlet
    /// pass is a no-op (every scene without vmesh content stays byte-identical).
    pub vgeom_assets: Vec<VgeomAsset>,
    /// Placed meshlet-path instances (P13.1b). Drawn by the GPU-driven
    /// [`crate::passes::vgeom`] pass (cull+LOD compute → vertex-pulled indirect
    /// draw) when [`RenderSettings::vgeom`](crate::RenderSettings) is enabled,
    /// after the rigid mesh pass, into the same MSAA targets.
    pub vgeom_instances: Vec<VgeomInstance>,
    /// GPU-scattered instance batches (P18.5) — PCG volumes and painted foliage.
    /// Empty ⇒ the [`crate::passes::scatter`] node emits no commands, so every
    /// scene without scatter content (including all 39 goldens) is untouched.
    ///
    /// Version-gated like everything else, but the *upload* is gated by each
    /// batch's own [`ScatterData::key`] instead: a projection that rebuilds the
    /// list without the scatter changing re-uses the GPU buffers.
    pub scatter: Vec<ScatterBatch>,
    /// **The carry-forward memo behind [`scatter`](Self::scatter)** (island wave
    /// I8a audit) — the batches each memoizable source contributed last
    /// projection, so a settled city is not re-packed sixty times a second.
    ///
    /// `ScatterData::key` gates the *upload*, and Hardening Wave E's own memo
    /// says why that is not enough: *"what no consumer could do is stop the
    /// payload from being built."* On the island's 172 settlement blocks the
    /// payload was 365 545 instances re-packed to f32 and re-hashed every frame —
    /// **20.2 ms of projection against a 1.5 ms budget** — for content that
    /// changes only when a cell activates.
    ///
    /// Read and written by the two MIRROR projectors and by nobody else; a
    /// renderer pass must keep reading [`scatter`](Self::scatter).
    pub scatter_memo: ScatterMemo,
    /// Water bodies (P20.1) — oceans, lakes and spline rivers. Empty ⇒ the
    /// [`crate::passes::water`] node returns before touching the encoder, so every
    /// scene without water (including all 42 pre-P20.1 goldens) records the exact
    /// command stream it always did.
    ///
    /// Ordering is the projector's and is what the draw order follows, so it must
    /// be deterministic per side. It is **not** the same order in both projectors
    /// (the player walks `Guid` order, the viewport document order) — the same
    /// arrangement `terrains` has, and for the same reason: what makes a
    /// cross-side comparison meaningful is each body's `id`, which both derive
    /// from the entity.
    pub waters: Vec<crate::water::RenderWater>,
    /// Scene lights (directional + point). Empty ⇒ the shader falls back to a
    /// default editor sun so unlit demo scenes still render.
    pub lights: Vec<RenderLight>,
    /// 2D sprites (batched + drawn by the sprite pass over the 3D scene).
    pub sprites: Vec<SpriteInstance>,
    /// Heightfield terrains (P10.1; **N of them** since P16.6). The terrain pass
    /// draws each one's clipmap LOD rings around the camera, in list order; an
    /// empty list ⇒ the pass is a no-op (so scenes without terrain — every
    /// pre-P10.1 golden — stay byte-identical). Each tile's own
    /// [`version`](RenderTerrainTile::version) stamp gates its height/weight
    /// texture upload (P16.3b1), keyed per terrain by
    /// [`RenderTerrain::id`] (P16.6).
    ///
    /// Ordering is the projector's, and it is what the draw order follows — so it
    /// must be deterministic. It is **not** the same order in both projectors: the
    /// player walks its world in `Guid` order, the editor viewport walks the
    /// document's own entity order. Each is deterministic for its own side, which
    /// is what a per-side determinism gate needs; what makes a *cross-side*
    /// comparison meaningful is [`id`](RenderTerrain::id), which both derive from
    /// the terrain entity's `Guid`, so a PIE-vs-shipping diff matches terrains up
    /// by identity rather than by position in a list.
    ///
    /// The terrains are independent: one's residency, grid and splat material say
    /// nothing about another's, and they may legitimately overlap in world space
    /// (the depth test resolves it, exactly as it does for two meshes).
    pub terrains: Vec<RenderTerrain>,
    /// Volumetric-terrain volumes (P21.1) — SDF voxel chunk sets, already meshed
    /// into triangle soup by the projector, that locally extend the heightfield
    /// above. Empty ⇒ the [`crate::passes::voxel`] node returns before touching the
    /// encoder, so every scene without volumetric terrain (including all 47
    /// pre-P21.1 goldens) records the exact command stream it always did.
    ///
    /// Ordering is the projector's, and it is what the draw order follows — so it
    /// must be deterministic. It is **not** the same order in both projectors: the
    /// player walks its world in `Guid` order, the editor viewport walks the
    /// document's own entity order. Each is deterministic for its own side, which
    /// is what a per-side determinism gate needs; what makes a *cross-side*
    /// comparison meaningful is [`id`](RenderVoxelVolume::id), which both derive
    /// from the volume entity's `Guid`, so a PIE-vs-shipping diff matches volumes
    /// up by identity rather than by position in a list. The same arrangement
    /// [`terrains`](Self::terrains) and `waters` have, for the same reason.
    ///
    /// Each chunk's own [`version`](RenderVoxelChunk::version) stamp gates its
    /// vertex/index buffer upload, keyed per volume by
    /// [`RenderVoxelVolume::id`].
    pub voxels: Vec<RenderVoxelVolume>,
    /// Live chunks of broken destructibles (P22.3) — the debris of everything
    /// that has come apart this session.
    ///
    /// **Empty for every intact level**, and that is the atomicity contract: an
    /// actor is drawn as its own `MeshRef` or as its chunks, never as both and
    /// never as neither. Both halves read one fact — the sim's
    /// `FractureState::is_intact` — so the swap cannot become an ordering accident
    /// between two passes.
    ///
    /// Each chunk's own [`version`](RenderFractureChunk::version) stamp gates its
    /// vertex/index buffer upload, keyed by `(entity, chunk)`.
    pub fracture_chunks: Vec<RenderFractureChunk>,
    /// The surface deformation field (P22.1) — the sparse map of how far the
    /// ground has been pressed down by whatever has been standing on it.
    ///
    /// `None` on every level where nothing has touched a terrain (which is every
    /// pre-P22.1 golden), and then [`crate::deform::DeformResources::update`]
    /// writes a **disabled** uniform, no texture upload happens at all, and the
    /// shader branches that read the window are present-but-false — so the
    /// command stream and the arithmetic are both unchanged.
    ///
    /// Behind an `Arc` because it is the one projected field that is *usually
    /// identical to last frame's*: the field only moves when something walked,
    /// and a projector that rebuilt this for a standing character would pay the
    /// copy the sparse representation exists to avoid. Hosts cache it on
    /// [`crate::RenderDeform::epoch`] and clone the pointer.
    ///
    /// Sim-authoritative, always: the camera decides which part of this is *drawn*
    /// (see [`crate::deform`]) and never what is in it.
    pub deform: Option<std::sync::Arc<crate::deform::RenderDeform>>,
    /// 2D tilemaps (P8.1b). The sprite pass culls each tilemap's chunks against
    /// the camera and expands the visible ones into prebatched sprite runs, then
    /// batches them together with the loose `sprites`. Because culling depends on
    /// the live camera, the pass re-expands tilemaps every frame (not gated by
    /// `version`) while any tilemap is present — a documented v1 cost (a
    /// camera-delta / dirty-region optimization is a follow-up).
    pub tilemaps: Vec<RenderTilemap>,
    /// Host-expanded 2D primitives that are already in draw order and share one
    /// `(layer, order, texture)` per run — 9-slices (nine quads) and text blocks
    /// (one quad per glyph), expanded by `inf-render-2d`. The sprite pass merges
    /// these with the loose `sprites` and the tilemap runs in one painter sort.
    /// Version-gated (the host rebuilds them on document change), unlike
    /// tilemaps which additionally re-expand per frame for culling.
    pub prebatched: Vec<PrebatchedRun>,
    /// Minimal 2D lights (P8.1c). Empty ⇒ only `ambient_2d` lights the sprites.
    pub lights_2d: Vec<RenderLight2D>,
    /// Scene-level 2D ambient (default white → unlit sprites unchanged).
    pub ambient_2d: Ambient2D,
    /// Textures to hand to the sprite pass's GPU cache (drained/deduped by
    /// handle). The host populates this once per newly-referenced texture.
    pub pending_texture_uploads: Vec<SpriteTextureUpload>,
    /// Bump on every change to `instances`/`lights`/`sprites`/`tilemaps` — gates
    /// buffer re-upload (tilemaps additionally re-expand per frame for culling).
    pub version: u64,
    pub sky: SkyParams,
    /// The sun and moon (P17.1). Defaults to the retired `SUN_DIR` constant, so a
    /// scene whose projector found no time-of-day authority renders exactly as it
    /// did before. See [`SunParams`].
    pub sun: SunParams,
    /// The physically-based atmosphere (P17.2): the LUT-driven sky, the sun/moon
    /// discs, the starfield, aerial perspective and height fog. **Disabled** by
    /// default, so a scene with no time-of-day authority draws the P17.1 gradient
    /// and its lit passes take the byte-identical no-atmosphere arithmetic. See
    /// [`crate::atmosphere::AtmosphereParams`].
    pub atmosphere: crate::atmosphere::AtmosphereParams,
    pub grid_enabled: bool,
    /// Ids drawn with the selection outline.
    pub selected: Vec<u32>,
    /// Id drawn with the hover outline (weaker), if any.
    pub hovered: Option<u32>,
    /// Immediate-mode debug lines, rebuilt by the host each frame
    /// (render-local space — not gated by `version`).
    pub debug: DebugDraw,
    /// **The in-game UI**, in SCREEN pixels (island wave I5).
    ///
    /// Rebuilt by the host each frame from `inf_ui`'s pure projections, and — like
    /// `debug` and unlike `sprites` — **not gated by `version`**, for the reason
    /// the same word appears on both: a menu is a function of what the player has
    /// pressed, not of what the document contains, so a version that only moves
    /// when the scene does would freeze the cursor on the row it started on.
    ///
    /// Empty on every frame nobody opened a menu on, and `passes::ui` returns
    /// before it touches the encoder when it is — which is what the frozen
    /// goldens rest on.
    pub ui: inf_ui::UiDrawList,
}

impl RenderScene {
    pub fn mark_dirty(&mut self) {
        self.version = self.version.wrapping_add(1);
    }
}

#[cfg(test)]
mod tests {

    /// **A steady fixture is untouched, and a sweeping rig shows three colours
    /// at once** (wave VEN1a).
    #[test]
    fn a_rig_sweeps_in_phase_and_a_steady_lamp_does_not_move() {
        let warm = [1.0, 0.62, 0.30];
        let sweep = ([3.0, 0.12, 0.22], [2.4, 0.18, 1.9]);
        // Steady: the FIRST colour, exactly, at every clock.
        for t in [0.0, 0.5, 13.7, 1e6] {
            assert_eq!(swept_colour((warm, warm), 0.0, 0, 1, t), warm);
            assert_eq!(swept_colour(sweep, 0.0, 2, 3, t), sweep.0);
        }
        // A NaN rate or a NaN clock holds the first colour rather than carrying
        // a NaN into the lights uniform.
        assert_eq!(swept_colour(sweep, f32::NAN, 0, 3, 4.0), sweep.0);
        assert_eq!(swept_colour(sweep, 0.11, 0, 3, f64::NAN), sweep.0);

        // A three-lamp rig at one instant: three DIFFERENT colours.
        let at = |t: f64| {
            (0..3)
                .map(|k| swept_colour(sweep, 0.11, k, 3, t))
                .collect::<Vec<_>>()
        };
        let now = at(4.0);
        assert_ne!(now[0], now[1], "two lamps of a rig show one colour");
        assert_ne!(now[1], now[2]);
        // …and every one of them is ON the segment between the two authored
        // colours, so a sweep cannot invent a hue nobody chose.
        for c in &now {
            for k in 0..3 {
                let (lo, hi) = (sweep.0[k].min(sweep.1[k]), sweep.0[k].max(sweep.1[k]));
                assert!(
                    c[k] >= lo - 1e-5 && c[k] <= hi + 1e-5,
                    "{c:?} left the sweep"
                );
            }
        }
        // It really moves over a cycle, and it reaches BOTH ends: the triangle
        // dwells at each, which a mid-range-only wobble would not.
        let hz = 0.11_f32;
        let period = 1.0 / f64::from(hz);
        let mut lo = f32::MAX;
        let mut hi = f32::MIN;
        for i in 0..64 {
            let c = swept_colour(sweep, hz, 0, 1, period * f64::from(i) / 64.0);
            lo = lo.min(c[2]);
            hi = hi.max(c[2]);
        }
        assert!(
            lo < sweep.0[2] + 0.05,
            "the sweep never reached its first end"
        );
        assert!(
            hi > sweep.1[2] - 0.05,
            "the sweep never reached its second end"
        );
        // Pure: the same clock is the same colour.
        assert_eq!(
            swept_colour(sweep, hz, 1, 3, 9.25),
            swept_colour(sweep, hz, 1, 3, 9.25)
        );
    }

    /// **A steady emitter is untouched, and a pulsing one breathes** (wave
    /// VEN1a).
    ///
    /// The first half is the byte-stability guarantee every emitter in the
    /// engine except a festoon relies on: `hz == 0` must not merely *round* to
    /// the input, it must **be** the input, or every neon sign and every screen
    /// would acquire a rounding nobody asked for.
    #[test]
    fn a_steady_emitter_is_untouched_and_a_pulsing_one_breathes() {
        let neon = [2.6, 0.4, 2.2];
        for tick in [0u32, 1, 7, 4096, u32::MAX] {
            assert_eq!(
                pulse_emissive(neon, 0.0, tick),
                neon,
                "a steady emitter moved at tick {tick}"
            );
        }
        // A NaN or negative rate is steady too, not a NaN emission carried for
        // ever (the island wave I8a spelling).
        assert_eq!(pulse_emissive(neon, f32::NAN, 5), neon);
        assert_eq!(pulse_emissive(neon, -1.0, 5), neon);

        // A festoon at 0.27 Hz: one cycle is 3.70 s, which is 29.6 ticks.
        let hz = 0.27_f32;
        let gains: Vec<f32> = (0..60)
            .map(|t| pulse_emissive([1.0, 1.0, 1.0], hz, t)[0])
            .collect();
        let lo = gains.iter().copied().fold(f32::MAX, f32::min);
        let hi = gains.iter().copied().fold(f32::MIN, f32::max);
        // It really moves, it stays inside the authored envelope, and it never
        // goes dark -- a neon tube that reaches zero reads as a fault.
        assert!(hi - lo > 0.5, "the pulse barely moved ({lo:.3}..{hi:.3})");
        assert!(
            lo >= PULSE_FLOOR - 1e-4,
            "the pulse went below its floor ({lo})"
        );
        assert!(
            hi <= 1.0 + 1e-4,
            "the pulse exceeded the authored value ({hi})"
        );
        // The hue is preserved: every channel takes the same gain.
        let p = pulse_emissive(neon, hz, 11);
        let g = p[0] / neon[0];
        assert!((p[1] / neon[1] - g).abs() < 1e-5);
        assert!((p[2] / neon[2] - g).abs() < 1e-5);
        // …and it is a pure function of the tick, twice over.
        assert_eq!(pulse_emissive(neon, hz, 11), p);
    }

    /// **The tick quantizes the clock and refuses nonsense** (wave VEN1a).
    #[test]
    fn the_pulse_tick_quantizes_the_level_clock() {
        assert_eq!(pulse_tick(0.0), 0);
        assert_eq!(pulse_tick(-3.0), 0, "a clock before zero reads as zero");
        assert_eq!(pulse_tick(f64::NAN), 0);
        assert_eq!(pulse_tick(f64::INFINITY), 0, "not finite, so no tick");
        assert_eq!(pulse_tick(1.0), PULSE_TICKS_PER_S as u32);
        // Eight ticks a second: two clocks inside one eighth share a tick, and
        // the memo therefore does not re-pack between them.
        assert_eq!(pulse_tick(2.010), pulse_tick(2.100));
        assert_ne!(pulse_tick(2.100), pulse_tick(2.200));
        // A year of level clock still fits, so the saturation is unreachable in
        // practice and is a guard rather than a policy.
        assert!(pulse_tick(365.0 * 24.0 * 3600.0) < u32::MAX);
    }

    /// **The night glow is a ramp on the sun's own height, quantized** (island
    /// wave I8b clause 3).
    ///
    /// Four claims, and the first is the one the memo key rests on: a daytime
    /// step is EXACTLY zero and a daytime emission is exactly `[0, 0, 0]`, so a
    /// level with no night in it packs the bytes it packed before this feature
    /// existed.
    #[test]
    fn the_night_glow_ramps_on_the_sun_and_quantizes() {
        use glam::Vec3;
        // High noon: nothing glows, and nothing about the batch moves.
        assert_eq!(night_glow_step(Vec3::new(0.0, 1.0, 0.0)), 0);
        assert_eq!(glow_emissive(1.6, 0), [0.0; 3]);
        // Deep night: the full step.
        assert_eq!(night_glow_step(Vec3::new(0.0, -1.0, 0.0)), NIGHT_GLOW_STEPS);
        // Monotone through dusk, and it really does pass through the middle
        // rather than switching: a ramp that only ever answered 0 or 16 would
        // satisfy the two ends above.
        let mut seen_middle = false;
        let mut last = 0u16;
        for k in 0..=40 {
            let y = 0.15 - 0.01 * k as f32;
            let step = night_glow_step(Vec3::new(0.0, y, 0.0));
            assert!(step >= last, "the ramp went backwards at y = {y}");
            if step > 0 && step < NIGHT_GLOW_STEPS {
                seen_middle = true;
            }
            last = step;
        }
        assert!(seen_middle, "the glow switches rather than ramping");
        // A non-finite direction reads as day — the answer that changes nothing.
        assert_eq!(night_glow_step(Vec3::new(0.0, f32::NAN, 0.0)), 0);
        // And the emission is a warm white scaled by both terms.
        let half = glow_emissive(1.0, NIGHT_GLOW_STEPS / 2);
        let full = glow_emissive(1.0, NIGHT_GLOW_STEPS);
        assert!(full[0] > half[0] && half[0] > 0.0);
        assert!(
            full[1] < full[0] && full[2] < full[1],
            "{full:?} is not warm"
        );
        assert_eq!(glow_emissive(0.0, NIGHT_GLOW_STEPS), [0.0; 3]);
    }
    use super::*;

    /// **A deformed garment carries a uv, and it is the box projection** (P26.5).
    ///
    /// This is the last producer of [`box_uv`] and the reason the projection was
    /// moved to Rust rather than deleted: a garment is simulated positions over a
    /// topology and has no authored parametrization to inherit. Nothing samples
    /// it yet — a garment's set is `VtTextureSet::NONE` — which is exactly why
    /// the stream must not be left at zero: an all-zero uv is a whole coat
    /// sampling one texel, and it would arrive silently the day a `.inf_cloth`
    /// grows a material slot.
    #[test]
    fn a_deformed_garment_carries_a_projected_uv_and_never_a_flat_one() {
        // A quad in XZ, so its recomputed normal is +Y and the projection's
        // dominant axis is unambiguous.
        let positions = [
            [-0.5, 0.0, -0.5],
            [0.5, 0.0, -0.5],
            [0.5, 0.0, 0.5],
            [-0.5, 0.0, 0.5],
        ];
        let mesh = deformed_skinned_mesh(&positions, &[0, 2, 1, 0, 3, 2]);
        assert_eq!(mesh.vertices.len(), 4);
        for v in &mesh.vertices {
            assert_eq!(
                v.uv,
                box_uv(v.pos, v.normal),
                "the garment's uv is not the projection"
            );
        }
        // ANTI-VACUITY: it VARIES. A `uv: [0.0; 2]` default satisfies "equals a
        // function of position" for a function that ignores its argument, and
        // this quad is the shape that would hide it.
        let u: Vec<f32> = mesh.vertices.iter().map(|v| v.uv[0]).collect();
        let w: Vec<f32> = mesh.vertices.iter().map(|v| v.uv[1]).collect();
        let span = |xs: &[f32]| {
            xs.iter().cloned().fold(f32::MIN, f32::max)
                - xs.iter().cloned().fold(f32::MAX, f32::min)
        };
        assert!(span(&u) > 0.9 && span(&w) > 0.9, "the garment's uv is flat");
    }

    #[test]
    fn default_sun_is_the_retired_constant_normalized() {
        // `DEFAULT_SUN_DIR` must be `normalize(0.45, 0.75, 0.3)` — the value the
        // deleted `camera::SUN_DIR` produced at every one of its call sites. This
        // identity is the whole reason every pre-P17.1 golden stays byte-identical.
        let legacy = Vec3::new(0.45, 0.75, 0.3).normalize();
        let s = SunParams::default();
        assert_eq!(
            s.unit_direction().to_array().map(f32::to_bits),
            legacy.to_array().map(f32::to_bits),
            "the default sun moved — every pre-P17.1 golden would move with it"
        );
        assert!((s.unit_direction().length() - 1.0).abs() < 1e-6);
        assert_eq!(RenderScene::default().sun, s);
        // A scene that never opts in never lights anything with the moon.
        assert_eq!(s.moon_intensity, 0.0);
        assert_eq!(s.unit_moon_direction(), Vec3::NEG_Y);
    }

    #[test]
    fn degenerate_sun_direction_falls_back() {
        let s = SunParams {
            direction: Vec3::ZERO,
            moon_direction: Vec3::ZERO,
            ..SunParams::default()
        };
        assert_eq!(s.unit_direction(), DEFAULT_SUN_DIR.normalize());
        assert_eq!(s.unit_moon_direction(), Vec3::NEG_Y);
        let nan = SunParams {
            direction: Vec3::splat(f32::NAN),
            ..SunParams::default()
        };
        assert!(nan.unit_direction().is_finite());
    }

    #[test]
    fn dirty_bumps_version() {
        let mut s = RenderScene::default();
        let v0 = s.version;
        s.instances.push(MeshInstance::lit(
            DVec3::ZERO,
            Quat::IDENTITY,
            Vec3::ONE,
            [1.0; 4],
            1,
        ));
        s.mark_dirty();
        assert_ne!(s.version, v0);
    }
    // ── P21.2 seam ──────────────────────────────────────────────

    /// A 5×5 tile at y = 10, sloping +1 m per sample along +X, optionally with
    /// sample `(2, 2)` holed.
    fn seam_terrain(holed: bool) -> RenderTerrain {
        const RES: u32 = 5;
        let heights: Vec<f32> = (0..RES).flat_map(|_| (0..RES).map(|i| i as f32)).collect();
        let mut holes = Vec::new();
        if holed {
            holes = vec![0u32; RES as usize];
            holes[2] |= 1 << 2;
        }
        RenderTerrain {
            id: 1,
            tile_resolution: RES,
            meters_per_sample: 1.0,
            tiles: vec![RenderTerrainTile {
                key: TerrainTileKey::lod0((0, 0)),
                origin: DVec3::new(0.0, 10.0, 0.0),
                heights,
                weights: vec![[255, 0, 0, 0]; (RES * RES) as usize],
                biomes: vec![0; (RES * RES) as usize],
                holes,
                height_bounds: (0.0, 4.0),
                version: 1,
            }],
            layers: [
                RenderTerrainLayer {
                    albedo: [0.2, 0.4, 0.1, 1.0],
                    roughness: 0.8,
                    tex_scale: 1.0,
                    ..RenderTerrainLayer::default()
                },
                RenderTerrainLayer::default(),
                RenderTerrainLayer::default(),
                RenderTerrainLayer::default(),
            ],
            macro_variation: 0.0,
            biome_palette: Vec::new(),
        }
    }

    /// The sampler answers with the surface it was given — the interpolated
    /// height, a normal that leans away from the slope, and the layer-0 albedo
    /// the weights select.
    #[test]
    fn seam_sample_resolves_height_normal_and_layer() {
        let t = seam_terrain(false);
        let s = t.seam_sample(1.5, 1.0).expect("inside the tile");
        assert!((s.height - 11.5).abs() < 1e-6, "height {}", s.height);
        // Heights rise with +X, so the normal tilts toward -X, and +Y stays
        // positive — which is what makes `y <= 0` a usable sentinel.
        assert!(s.normal[0] < 0.0 && s.normal[1] > 0.0, "{:?}", s.normal);
        assert!((s.normal[2]).abs() < 1e-6, "{:?}", s.normal);
        assert_eq!(s.albedo, [0.2, 0.4, 0.1]);
        assert!((s.roughness - 0.8).abs() < 1e-6);

        // Outside the projected tile there is no answer at all — not a guess.
        assert!(t.seam_sample(500.0, 500.0).is_none());
    }

    /// **The poison rule, projector half.** A holed sample removes the seam from
    /// every cell that interpolates it, and from no other — the same rule
    /// `terrain.wgsl` and `inf_terrain::TerrainData::height_at` apply.
    #[test]
    fn seam_sample_refuses_a_holed_cell_and_only_that_cell() {
        let t = seam_terrain(true);
        // The four cells around sample (2, 2).
        for (x, z) in [(1.5, 1.5), (2.5, 1.5), (1.5, 2.5), (2.5, 2.5)] {
            assert!(t.seam_sample(x, z).is_none(), "({x}, {z}) survived");
        }
        // One cell further out is untouched.
        for (x, z) in [(0.5, 0.5), (3.5, 3.5)] {
            assert!(t.seam_sample(x, z).is_some(), "({x}, {z}) was poisoned");
        }
        // The same query on the un-carved twin answers everywhere — so the test
        // above is about the hole and not about the fixture.
        let clean = seam_terrain(false);
        assert!(clean.seam_sample(2.0, 2.0).is_some());
    }

    /// `apply_seam` arms the band and fills the vertices that have terrain over
    /// them, leaves the rest at the sentinel, and is a **no-op** when disarmed —
    /// which is what keeps a terrain-free scene byte-identical to its pre-P21.2
    /// render.
    #[test]
    fn apply_seam_fills_only_where_there_is_ground() {
        let terrain = seam_terrain(true);
        let vertex = |x: f32, z: f32| RenderVoxelVertex {
            pos: [x, 0.0, z],
            normal: [0.0, 1.0, 0.0],
            material: 0,
            seam_nh: RenderVoxelVertex::NO_SEAM,
            seam_albedo: [0.0; 4],
        };
        let mut volumes = vec![RenderVoxelVolume {
            id: 1,
            chunks: vec![RenderVoxelChunk {
                key: VoxelChunkKey::default(),
                origin: DVec3::ZERO,
                // Under the hole, on clear ground, and off the terrain entirely.
                vertices: vec![vertex(2.0, 2.0), vertex(0.5, 0.5), vertex(400.0, 400.0)],
                indices: vec![0, 1, 2],
                bounds: ([0.0; 3], [1.0; 3]),
                version: 1,
            }],
            layers: [RenderTerrainLayer::default(); 4],
            seam_band_m: 0.0,
        }];

        // Disarmed: nothing moves, and the band stays off.
        let mut off = volumes.clone();
        apply_seam(&mut off, std::slice::from_ref(&terrain), 0.0);
        assert_eq!(off[0].seam_band_m, 0.0);
        assert!(off[0].chunks[0]
            .vertices
            .iter()
            .all(|v| v.seam_nh == RenderVoxelVertex::NO_SEAM));

        apply_seam(&mut volumes, std::slice::from_ref(&terrain), 2.0);
        assert_eq!(volumes[0].seam_band_m, 2.0);
        let vs = &volumes[0].chunks[0].vertices;
        assert_eq!(
            vs[0].seam_nh,
            RenderVoxelVertex::NO_SEAM,
            "a vertex under the hole must not pick up a seam"
        );
        assert!(
            vs[1].seam_nh[1] > 0.0,
            "clear ground must seam: {:?}",
            vs[1]
        );
        assert_eq!(vs[1].seam_albedo, [0.2, 0.4, 0.1, 0.8]);
        assert_eq!(
            vs[2].seam_nh,
            RenderVoxelVertex::NO_SEAM,
            "a vertex off the terrain must not pick up a seam"
        );
    }

    /// **The idempotency arm** (Hardening Wave E) — the property that makes it
    /// safe for a host to carry an unchanged volume forward instead of rebuilding
    /// it.
    ///
    /// Before this wave `apply_seam` *skipped* a vertex it refused, which is
    /// indistinguishable from clearing it as long as the input is always a fresh
    /// projection. It no longer is: `take_unchanged_voxel` hands back the volume
    /// this function wrote into last frame. So a vertex that was seamed and is now
    /// refused — because the terrain under it moved, was holed, or left the scene
    /// entirely — has to come back to the sentinel, or a cave mouth keeps a blend
    /// against ground that is no longer there for the rest of the session.
    ///
    /// Three refusal routes, one arm: the terrain moves out of reach, the terrain
    /// list empties, and the band is disarmed. All three re-run over an
    /// **already-seamed** volume, which is the case that did not exist before.
    #[test]
    fn re_seaming_an_already_seamed_volume_clears_what_it_refuses() {
        let terrain = seam_terrain(false);
        let mut volumes = vec![RenderVoxelVolume {
            id: 1,
            chunks: vec![RenderVoxelChunk {
                key: VoxelChunkKey::default(),
                origin: DVec3::ZERO,
                // Directly under the cell `seam_terrain(true)` holes, so the
                // SAME vertex is seamed by the unholed terrain and refused by
                // the holed one — a carve under a cave mouth, exactly.
                vertices: vec![RenderVoxelVertex {
                    pos: [2.0, 0.0, 2.0],
                    normal: [0.0, 1.0, 0.0],
                    material: 0,
                    seam_nh: RenderVoxelVertex::NO_SEAM,
                    seam_albedo: [0.0; 4],
                }],
                indices: vec![0, 0, 0],
                bounds: ([0.0; 3], [1.0; 3]),
                version: 1,
            }],
            layers: [RenderTerrainLayer::default(); 4],
            seam_band_m: 0.0,
        }];

        apply_seam(&mut volumes, std::slice::from_ref(&terrain), 2.0);
        let seamed = volumes[0].chunks[0].vertices[0];
        assert!(
            seamed.seam_nh != RenderVoxelVertex::NO_SEAM && volumes[0].seam_band_m == 2.0,
            "the fixture must actually seam, or every case below is vacuous: {seamed:?}"
        );

        // (1) The ground under it was carved through — the poison rule refuses.
        let holed = seam_terrain(true);
        let mut carried = volumes.clone();
        apply_seam(&mut carried, std::slice::from_ref(&holed), 2.0);
        assert_eq!(
            carried[0].chunks[0].vertices[0].seam_nh,
            RenderVoxelVertex::NO_SEAM,
            "a carried volume kept a seam against ground that was carved away"
        );
        assert_eq!(carried[0].chunks[0].vertices[0].seam_albedo, [0.0; 4]);

        // (2) The terrain left the scene entirely.
        let mut carried = volumes.clone();
        apply_seam(&mut carried, &[], 2.0);
        assert_eq!(
            carried[0].chunks[0].vertices[0].seam_nh,
            RenderVoxelVertex::NO_SEAM,
            "a carried volume kept a seam after every terrain left"
        );
        assert_eq!(carried[0].seam_band_m, 0.0);

        // (3) The band was disarmed.
        let mut carried = volumes.clone();
        apply_seam(&mut carried, std::slice::from_ref(&terrain), 0.0);
        assert_eq!(
            carried[0].chunks[0].vertices[0].seam_nh,
            RenderVoxelVertex::NO_SEAM,
            "a carried volume kept a seam after the band was disarmed"
        );
        assert_eq!(carried[0].seam_band_m, 0.0);

        // …and re-running with the SAME inputs is a fixed point, which is the
        // half that says a memo hit and a memo miss produce the same bytes.
        let mut again = volumes.clone();
        apply_seam(&mut again, std::slice::from_ref(&terrain), 2.0);
        assert_eq!(again, volumes, "re-seaming an unchanged pair moved bytes");
    }

    /// **THE F3 GATE: two hosts, two list orders, one seam.**
    ///
    /// The editor projects terrains in document order and the shipped player in
    /// `Guid` order, so with overlapping heightfields a first-answer-wins rule made
    /// the same vertex seam against a different terrain per host: PIE ≠ shipping in
    /// albedo, roughness and shading normal, and invisible to every gate that
    /// compares the two *projectors* character for character.
    ///
    /// Asserted the only way that means anything — the identical input presented in
    /// **both** orders, with the answer pinned to the near terrain rather than
    /// merely to "the same one either way" (two orders agreeing on the wrong
    /// terrain would pass a self-consistency check).
    #[test]
    fn the_seam_is_the_same_in_both_terrain_orderings() {
        // The ground the cave is cut through, at y ≈ 10, and a plateau 50 m over it
        // that covers the same plan. Distinct ids, as P16.6 requires.
        let near = seam_terrain(false);
        let far = {
            let mut t = seam_terrain(false);
            t.id = 2;
            t.tiles[0].origin.y += 50.0;
            t.layers[0].albedo = [0.9, 0.1, 0.8, 1.0];
            t.layers[0].roughness = 0.15;
            t
        };
        assert_ne!(
            near.seam_sample(0.5, 0.5).unwrap().albedo,
            far.seam_sample(0.5, 0.5).unwrap().albedo,
            "the two fixtures shade the same, so neither ordering could tell them apart"
        );

        let volume = || RenderVoxelVolume {
            id: 7,
            chunks: vec![RenderVoxelChunk {
                key: VoxelChunkKey::default(),
                origin: DVec3::ZERO,
                // A mouth vertex just under the near ground, and one just under the
                // plateau — each must take the surface it belongs to.
                vertices: vec![
                    RenderVoxelVertex {
                        pos: [0.5, 10.0, 0.5],
                        normal: [0.0, 1.0, 0.0],
                        material: 0,
                        seam_nh: RenderVoxelVertex::NO_SEAM,
                        seam_albedo: [0.0; 4],
                    },
                    RenderVoxelVertex {
                        pos: [0.5, 60.0, 0.5],
                        normal: [0.0, 1.0, 0.0],
                        material: 0,
                        seam_nh: RenderVoxelVertex::NO_SEAM,
                        seam_albedo: [0.0; 4],
                    },
                ],
                indices: vec![0, 1, 0],
                bounds: ([0.0; 3], [1.0; 3]),
                version: 1,
            }],
            layers: [RenderTerrainLayer::default(); 4],
            seam_band_m: 0.0,
        };

        let seam_of = |terrains: &[RenderTerrain]| {
            let mut v = vec![volume()];
            apply_seam(&mut v, terrains, DEFAULT_SEAM_BAND_M);
            v[0].chunks[0].vertices.clone()
        };
        let doc_order = seam_of(&[near.clone(), far.clone()]);
        let guid_order = seam_of(&[far.clone(), near.clone()]);
        assert_eq!(
            doc_order, guid_order,
            "the seam depends on the order the host happened to project its \
             terrains in — the editor walks `doc.order()` and the player walks \
             `Guid` order, so this IS a PIE-vs-shipping difference"
        );
        // …and it is the near surface that answers, in both.
        assert_eq!(
            doc_order[0].seam_albedo,
            [0.2, 0.4, 0.1, 0.8],
            "the mouth at y = 10 must seam against the ground it was cut through"
        );
        assert_eq!(
            doc_order[1].seam_albedo,
            [0.9, 0.1, 0.8, 0.15],
            "the vertex under the plateau must seam against the plateau"
        );
    }

    /// **The stated residual** behind [`apply_seam`]'s paging note: a streamed
    /// terrain whose residency floor has not arrived projects no tiles, so its host
    /// pushes no `RenderTerrain` at all and the volume stays unseamed until the next
    /// document bump re-projects.
    ///
    /// Pinned rather than fixed — the repair would have to re-project on residency,
    /// which is the camera-driven path the P18 law keeps out of lighting. A test is
    /// what stops it from silently becoming something else.
    #[test]
    fn a_terrain_with_no_tiles_leaves_the_volume_unseamed() {
        let mut volumes = vec![RenderVoxelVolume {
            id: 1,
            chunks: vec![RenderVoxelChunk {
                key: VoxelChunkKey::default(),
                origin: DVec3::ZERO,
                vertices: vec![RenderVoxelVertex {
                    pos: [0.5, 10.0, 0.5],
                    normal: [0.0, 1.0, 0.0],
                    material: 0,
                    seam_nh: RenderVoxelVertex::NO_SEAM,
                    seam_albedo: [0.0; 4],
                }],
                indices: vec![0, 0, 0],
                bounds: ([0.0; 3], [1.0; 3]),
                version: 1,
            }],
            layers: [RenderTerrainLayer::default(); 4],
            seam_band_m: 0.0,
        }];

        // What a host projects for a stream with nothing resident: nothing. (The
        // hosts filter on `tile_count() + coarse_tile_count() > 0`, so an empty
        // `RenderTerrain` is not pushed either — this is the empty slice.)
        apply_seam(&mut volumes, &[], DEFAULT_SEAM_BAND_M);
        assert_eq!(
            volumes[0].seam_band_m, 0.0,
            "the band must stay disarmed, or the shader would blend toward a \
             sentinel"
        );
        assert_eq!(
            volumes[0].chunks[0].vertices[0].seam_nh,
            RenderVoxelVertex::NO_SEAM
        );

        // The repair is a re-projection, and only a re-projection: the identical
        // volume seams the moment the floor is there and the host projects again.
        apply_seam(
            &mut volumes,
            std::slice::from_ref(&seam_terrain(false)),
            DEFAULT_SEAM_BAND_M,
        );
        assert!(volumes[0].chunks[0].vertices[0].seam_nh[1] > 0.0);
    }

    /// The row-packed mask reads back the bit that was set, and an empty mask
    /// reads `false` everywhere — the sparse default, which is what lets the
    /// pass bind a 1×1 sentinel texture.
    #[test]
    fn the_projected_hole_mask_reads_back() {
        let t = seam_terrain(true);
        let tile = &t.tiles[0];
        assert!(tile.has_holes());
        assert!(tile.is_hole(5, 2, 2));
        for (i, j) in [(1, 2), (3, 2), (2, 1), (2, 3)] {
            assert!(!tile.is_hole(5, i, j), "({i},{j})");
        }
        let clean = seam_terrain(false);
        assert!(!clean.tiles[0].has_holes());
        assert!(!clean.tiles[0].is_hole(5, 2, 2));
    }

    // ── P21.2 audit: the seam may not read camera-driven residency ──────

    /// A **streamed** seam terrain: the level-0 page of [`seam_terrain`] (holed)
    /// beside the coarse pyramid page the streamer pins as its residency floor.
    ///
    /// `level_0` is what the camera controls — near the terrain it is paged in,
    /// far away it is not — so the two states this builds are two genuinely
    /// different *residency histories* over one asset.
    ///
    /// The coarse page is deliberately **not** a decimation of the fine one (flat
    /// at +5 m, where level 0 slopes from +0 to +4): a fixture whose two levels
    /// agreed could not say which one answered, and that is the whole question
    /// here. It carries no weights and no hole mask, which is exactly what a real
    /// `downsample_block` page carries.
    fn streamed_seam_terrain(level_0: bool) -> RenderTerrain {
        const RES: u32 = 5;
        let coarse = RenderTerrainTile {
            key: TerrainTileKey::new(1, (0, 0)),
            origin: DVec3::new(0.0, 10.0, 0.0),
            heights: vec![5.0; (RES * RES) as usize],
            weights: Vec::new(),
            biomes: Vec::new(),
            holes: Vec::new(),
            height_bounds: (5.0, 5.0),
            version: 2,
        };
        let mut t = seam_terrain(true);
        if !level_0 {
            t.tiles.clear();
        }
        t.tiles.push(coarse);
        t
    }

    /// **THE B1 REGRESSION, projector half.** The seam is resolved against the
    /// terrain's *residency floor* — never against whichever fine page the camera
    /// dragged in — so two residency histories over one asset produce the identical
    /// sample.
    ///
    /// Before the fix `seam_sample` took `key.lod == 0`, which is precisely the
    /// part of the published cut that pages: the same point answered with a full
    /// blend near the camera and `None` far from it, and a voxel surface's albedo,
    /// roughness and shading normal — its **lighting** — moved with it.
    #[test]
    fn seam_sample_reads_the_residency_floor_not_the_finest_page() {
        let near = streamed_seam_terrain(true);
        let far = streamed_seam_terrain(false);
        assert_eq!(near.max_lod(), 1);
        assert_eq!(far.max_lod(), 1);
        assert!(
            far.tiles.len() < near.tiles.len(),
            "the far state must actually be a smaller residency set"
        );

        for (x, z) in [(0.5, 0.5), (2.0, 2.0), (3.75, 1.25)] {
            let a = near.seam_sample(x, z);
            let b = far.seam_sample(x, z);
            assert_eq!(a, b, "({x}, {z}) answered differently across residency");
            let s = a.expect("the floor covers the whole terrain");
            // …and it is the FLOOR's surface, not the fine page's: level 0 is at
            // 10 + x here, the coarse page is flat at 15.
            assert!((s.height - 15.0).abs() < 1e-6, "height {}", s.height);
            assert_eq!(s.normal, [0.0, 1.0, 0.0]);
        }

        // The level-0 page is genuinely a different surface, so the equality above
        // is a claim about which level was read and not about a flat fixture.
        let inline = seam_terrain(false);
        assert_eq!(inline.max_lod(), 0);
        let s = inline.seam_sample(0.5, 0.5).expect("inside");
        assert!((s.height - 10.5).abs() < 1e-6, "height {}", s.height);
    }

    /// Holes do **not** propagate into the pyramid, so a coarse-floor projection
    /// cannot answer the hole question — and says so rather than implying "no
    /// hole". The stated remainder is `downsample_block` carrying a hole mask; the
    /// day it does, this flips and the veto below becomes dead weight.
    #[test]
    fn a_streamed_projection_does_not_know_where_the_holes_are() {
        assert!(seam_terrain(true).seam_holes_are_known());
        let streamed = streamed_seam_terrain(true);
        assert!(!streamed.seam_holes_are_known());
        // The level-0 poison rule refuses (2, 2); the coarse floor has never heard
        // of it. That is the divergence, pinned rather than left to be found.
        assert!(seam_terrain(true).seam_sample(2.0, 2.0).is_none());
        assert!(streamed.seam_sample(2.0, 2.0).is_some());
    }

    /// **THE B1 GATE, CPU half.** Every seam attribute of every vertex is
    /// bit-identical across two residency histories — asserted on the raw `f32`
    /// arrays, because a lighting input that is "close" across camera history is
    /// still a lighting input that depends on camera history.
    #[test]
    fn the_seam_is_bit_identical_across_two_residency_histories() {
        let vertex = |x: f32, z: f32, n: [f32; 3]| RenderVoxelVertex {
            pos: [x, 0.0, z],
            normal: n,
            material: 0,
            seam_nh: RenderVoxelVertex::NO_SEAM,
            seam_albedo: [0.0; 4],
        };
        let volume = RenderVoxelVolume {
            id: 1,
            chunks: vec![RenderVoxelChunk {
                key: VoxelChunkKey::default(),
                origin: DVec3::ZERO,
                vertices: vec![
                    vertex(0.5, 0.5, [0.0, 1.0, 0.0]),
                    vertex(2.0, 2.0, [0.0, 1.0, 0.0]),
                    vertex(3.5, 1.5, [0.3, 0.9, 0.3]),
                    vertex(1.0, 3.0, [0.0, -1.0, 0.0]),
                    vertex(400.0, 400.0, [0.0, 1.0, 0.0]),
                ],
                indices: vec![0, 1, 2],
                bounds: ([0.0; 3], [1.0; 3]),
                version: 1,
            }],
            layers: [RenderTerrainLayer::default(); 4],
            seam_band_m: 0.0,
        };

        let seamed = |terrain: RenderTerrain| {
            let mut v = vec![volume.clone()];
            apply_seam(&mut v, std::slice::from_ref(&terrain), DEFAULT_SEAM_BAND_M);
            v
        };
        let near = seamed(streamed_seam_terrain(true));
        let far = seamed(streamed_seam_terrain(false));
        for (i, (a, b)) in near[0].chunks[0]
            .vertices
            .iter()
            .zip(&far[0].chunks[0].vertices)
            .enumerate()
        {
            assert_eq!(
                (a.seam_nh, a.seam_albedo),
                (b.seam_nh, b.seam_albedo),
                "vertex {i} was lit differently depending on where the camera had \
                 been — camera-driven residency is feeding lighting"
            );
        }
        // Not vacuous: the seam really did fire on the ground-facing vertices.
        assert!(
            near[0].chunks[0].vertices[..3]
                .iter()
                .all(|v| v.seam_nh[1] > 0.0),
            "no vertex picked up a seam at all"
        );
    }

    /// The mask-free veto, where the mask is missing: a surface that does not
    /// **continue** the heightfield gets no seam, so a coarse floor that cannot see
    /// a hole still does not paint hillside onto a cave ceiling.
    ///
    /// The second half is the part that keeps every existing golden byte-stable:
    /// over an *inline* terrain, whose mask IS present, the veto does not apply and
    /// a down-facing vertex seams exactly as it always did.
    #[test]
    fn a_coarse_seam_refuses_the_surfaces_that_do_not_continue_the_ground() {
        let vertex = |n: [f32; 3]| RenderVoxelVertex {
            pos: [0.5, 0.0, 0.5],
            normal: n,
            material: 0,
            seam_nh: RenderVoxelVertex::NO_SEAM,
            seam_albedo: [0.0; 4],
        };
        let volume = RenderVoxelVolume {
            id: 1,
            chunks: vec![RenderVoxelChunk {
                key: VoxelChunkKey::default(),
                origin: DVec3::ZERO,
                // A cave floor (up), a cave wall (perpendicular), a cave roof (down).
                vertices: vec![
                    vertex([0.0, 1.0, 0.0]),
                    vertex([1.0, 0.0, 0.0]),
                    vertex([0.0, -1.0, 0.0]),
                ],
                indices: vec![0, 1, 2],
                bounds: ([0.0; 3], [1.0; 3]),
                version: 1,
            }],
            layers: [RenderTerrainLayer::default(); 4],
            seam_band_m: 0.0,
        };

        let mut streamed = vec![volume.clone()];
        apply_seam(
            &mut streamed,
            std::slice::from_ref(&streamed_seam_terrain(true)),
            DEFAULT_SEAM_BAND_M,
        );
        let vs = &streamed[0].chunks[0].vertices;
        assert!(vs[0].seam_nh[1] > 0.0, "the floor must still seam");
        assert_eq!(
            vs[1].seam_nh,
            RenderVoxelVertex::NO_SEAM,
            "a wall is perpendicular to the ground, not a continuation of it"
        );
        assert_eq!(
            vs[2].seam_nh,
            RenderVoxelVertex::NO_SEAM,
            "hillside was blended onto a cave ceiling"
        );

        // Inline terrain: the mask answers, so the veto stays out of it.
        let mut inline = vec![volume];
        apply_seam(
            &mut inline,
            std::slice::from_ref(&seam_terrain(false)),
            DEFAULT_SEAM_BAND_M,
        );
        assert!(
            inline[0].chunks[0]
                .vertices
                .iter()
                .all(|v| v.seam_nh[1] > 0.0),
            "the veto fired over a terrain whose hole mask is present — every \
             pre-P21.2-audit golden depends on it not doing that"
        );
    }
}
