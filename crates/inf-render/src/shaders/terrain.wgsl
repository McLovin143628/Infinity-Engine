// Heightfield terrain pass (P10.1): geometry-clipmap LOD patches, vertex-shader
// displaced by a per-tile R32Float height texture. One draw per visible tile
// patch (its LOD mesh + its height texture); per-patch data rides the instance
// buffer. LOD morphing (blend toward the coarser grid) kills popping; skirt
// vertices drop the patch boundary down to hide cracks between differing LODs.
//
// Shading is a debug slope+altitude ramp lit by the scene sun — the P10.4 splat
// material hook is marked below. Opaque, reverse-Z, depth-writing.
//
// The height texture is sampled with `textureLoad` + manual bilinear (not a
// filtering sampler), so R32Float works as UnfilterableFloat everywhere —
// no FLOAT32_FILTERABLE feature needed (keeps headless CI adapters happy).

// The per-tile height texture (R32Float; texel = f32 metre offset from the tile
// origin's Y), bound per patch at @group(1).
@group(1) @binding(0) var height_tex: texture_2d<f32>;
// The per-tile splat-weight texture (Rgba8Unorm; texel = the four normalized
// layer weights), bound per patch beside the height texture (P10.4).
@group(1) @binding(1) var weight_tex: texture_2d<f32>;
// The per-tile biome-id texture (R8Uint; texel = one categorical biome id, 0 =
// unassigned), bound per patch beside the two above (P19.2). Integer-typed on
// purpose — see `load_biome`.
@group(1) @binding(2) var biome_tex: texture_2d<u32>;
// The per-tile HOLE MASK (P21.2): R32Uint, **row-packed** — texel `(w, j)` holds
// the hole bits for samples `[w*32, w*32+32)` of row `j`, LSB-first. A tile with
// no holes binds a **1×1 zero texture** instead of a `ceil(res/32) × res` one, so
// the un-carved case costs four bytes and needs no shader permutation: every
// `textureLoad` past a 1×1 texture's bounds returns zero by WGSL's own rule, and
// zero means "not holed" everywhere. That is the entire sparse-upload mechanism.
//
// Bits rather than a byte-per-sample R8Uint texture, for the same reason the CPU
// layer packs (see `inf_terrain::hole_mask_bytes`): this is a predicate, and 8×
// the bandwidth to carry seven zero bits per sample buys nothing. Row-packed
// rather than tile-packed so the fragment's index is `(i >> 5, j)` — no division
// by a non-constant, and no dependence on the tile resolution at all.
@group(1) @binding(3) var hole_tex: texture_2d<u32>;

// Terrain splat material (@group(2)): four layers + macro variation. Mirrors
// `MaterialRaw` in passes/terrain.rs. `params[k].x` = roughness, `.y` = tex_scale.
struct TerrainMaterial {
    albedo: array<vec4<f32>, 4>,
    params: array<vec4<f32>, 4>,
    macro_amp: vec4<f32>,
    // **WAVE T (the texture document's §3 B): each splat layer's virtual-texture
    // slots**, in `VtTextureSet::slots()` order — albedo (+ detail in the top
    // half), normal (+ detail scale in the top half), ORM, unused.
    //
    // All zero on every terrain that does not bind layer materials, which is
    // every terrain that has ever existed: `layer_is_textured` is then false for
    // all four layers and the fragment takes the flat-colour path below it,
    // instruction for instruction. That is what keeps the committed terrain
    // goldens byte-stable while the capability ships on by construction.
    slots: array<vec4<u32>, 4>,
};
@group(2) @binding(0) var<uniform> material: TerrainMaterial;

// Biome id → debug colour (P19.2), INDEXED BY ID. Mirrors `BiomePaletteRaw` in
// passes/terrain.rs: a fixed 256 slots so a `u8` id is always in range and the
// fragment needs no bounds logic. Every undefined id (including the reserved 0)
// was padded with the unassigned colour on the CPU side. A separate binding from
// `material` so the layer material's bytes are untouched by this mode.
struct BiomePalette {
    colors: array<vec4<f32>, 256>,
};
@group(2) @binding(1) var<uniform> biome_palette: BiomePalette;

// AO + cascaded shadows + dynamic GI ride the shared env bind group at @group(3)
// (declared in env_lighting.wgsl, prepended by `lit_scene_shader`): `ao_tex`/`ao_smp`
// (SSAO, white when off), `shadow_factor()`, and `ambient_irradiance()`.

struct VIn {
    // Vertex: unit patch coordinates in [0,1]² + skirt flag (z = 1 on the
    // boundary skirt ring, else 0).
    @location(0) uv_skirt: vec3<f32>,
    // Instance (per patch): origin_local.xyz + world tile span.
    @location(1) o_span: vec4<f32>,
    // Instance: the ring's morph BAND (start, end — metres of horizontal camera
    // distance), grid cells at this LOD, texture resolution. See `morph_at`.
    @location(2) params: vec4<f32>,
    // Instance: skirt depth (m). Its own attribute because the band takes two of
    // `params`' four slots.
    @location(3) skirt_depth: f32,
};

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_local: vec3<f32>,
    @location(1) uv: vec2<f32>,
    // span, resolution, grid cells at this LOD
    @location(2) @interpolate(flat) span_res_cells: vec3<f32>,
    // The patch's render-local origin XZ + its morph band: everything
    // `morphed_height` needs that the fragment cannot derive from `world_local`.
    @location(3) @interpolate(flat) origin_band: vec4<f32>,
};

fn load_texel(ij: vec2<i32>, res: f32) -> f32 {
    let m = i32(res) - 1;
    let c = clamp(ij, vec2<i32>(0, 0), vec2<i32>(m, m));
    return textureLoad(height_tex, c, 0).r;
}

// Bilinearly-sampled height offset (metres) at unit patch coord `uv`.
fn sample_height(uv: vec2<f32>, res: f32) -> f32 {
    let p = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)) * (res - 1.0);
    let i0 = floor(p);
    let f = p - i0;
    let ii = vec2<i32>(i0);
    let h00 = load_texel(ii, res);
    let h10 = load_texel(ii + vec2<i32>(1, 0), res);
    let h01 = load_texel(ii + vec2<i32>(0, 1), res);
    let h11 = load_texel(ii + vec2<i32>(1, 1), res);
    let hx0 = mix(h00, h10, f.x);
    let hx1 = mix(h01, h11, f.x);
    return mix(hx0, hx1, f.y);
}

// ── P22.1 SURFACE DEFORMATION ────────────────────────────────────────────────
//
// The ground height a patch actually draws at: the authored heightfield MINUS
// whatever has been pressed into it. `deform.wgsl` (composed above) supplies
// `deform_depth`, which returns 0 outside the window and 0 in every scene with
// no deformation — so this is the identity function on every pre-P22.1 golden.
//
// **This wrapper is the point.** A vertex-only offset would dent the geometry
// and leave the shading flat, which reads as a texture bug rather than a
// footprint: the fragment's central-difference normal must see the same surface
// the vertex stage moved. Both stages call THIS, and neither computes the offset
// itself, so they cannot drift apart.
//
// `uv` is the patch coordinate (for the heightfield) and `world_xz` the SAME
// point in render-local space (for the window) — two parameters rather than one
// derivation, because the vertex stage has the patch origin to hand and the
// fragment stage has `world_local`, and re-deriving either would be a second
// place for the mapping to be wrong.
//
// Holes are handled by composition order, not by a second test: the fragment
// discards a holed cell before it reaches any of this, and a holed vertex draws
// nothing that survives. Testing again here would only let the two stages
// disagree about which samples are ground.
fn ground_height(uv: vec2<f32>, res: f32, world_xz: vec2<f32>) -> f32 {
    return sample_height(uv, res) - deform_depth(world_xz);
}

// ── LOD MORPH (wave CERT1) ───────────────────────────────────────────────────
//
// **One rule, two evaluation points.** `passes::terrain::morph_band` owns the
// geometry of the ramp — a ring's `[start, end]` band of horizontal camera
// distance — and ships it per patch in `params.xy`; `morph_at` below is the same
// smoothstep `passes::terrain::morph_factor` evaluates on the CPU. What the GPU
// adds is the evaluation POINT: the vertex's own horizontal distance, not the
// tile centre's.
//
// # Why per-vertex, as a number
//
// The morph factor used to be one `f32` per patch, computed from the distance to
// the tile CENTRE. Along the shared edge of two same-ring neighbours the two
// height textures agree sample for sample (the pyramid's shared-edge invariant),
// so the morph was the only thing that could separate the two surfaces — and it
// always did, maximally. On the island the ring-0 ramp is
// `TERRAIN_MORPH_REGION · band width` = 134.4 m wide while adjacent tile centres
// are 256 m apart, so two neighbours can never both be inside the ramp: one is
// pinned at 0 and the other at 1. `tests/terrain_continuity.rs` measured the
// resulting crack at **3.83 m** on a 60 m-relief field, and **0.0000 m** with the
// rule below.
//
// The band is `(0, 0)` for the coarsest ring — a non-positive width is the same
// clause `morph_band` states by returning `None`.
fn morph_at(dist: f32, band: vec2<f32>) -> f32 {
    let width = band.y - band.x;
    if (width <= 0.0) {
        return 0.0;
    }
    let t = clamp((dist - band.x) / width, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

// The height of the **next-coarser grid** at `uv`: bilinear on the coarse
// lattice, i.e. the CHORD the coarser mesh actually rasterizes between its own
// vertices.
//
// # Why a chord and not `round(uv / step) * step` (wave CERT1)
//
// The previous target was a nearest-vertex SNAP, and WGSL's `round` is
// ties-to-even, so consecutive odd vertices snapped in opposite directions:
// index 1 → 0, index 3 → 4, index 5 → 4, index 7 → 8. At full morph the grid
// therefore did not become the coarser grid at all — it became a staircase of
// duplicated heights whose surviving segments span 4 m instead of 8, i.e. **twice
// the local slope**. Two defects in one: the LOD transition never actually
// converged on the mesh it was morphing toward (so the pop it exists to kill
// survived it), and the morph band was a slope amplifier — the "jagged and sharp"
// this wave was called for.
//
// Measured on the island's numbers, at the worst shared-edge vertex of a 60 m
// relief field: the nearest-snap target sat **3.8262 m** off the fine surface
// (`tests/terrain_continuity.rs`, the arm's BEFORE column — a whole 4 m fine cell,
// because the snap jumps a vertex rather than splitting the difference between
// two).
//
// The chord is also what makes the fragment's normal well-posed. A snapped target
// is piecewise CONSTANT in `uv`, so the gradient of the morphed field goes to
// zero at full morph and the far half of every ring would shade dead flat; the
// chord's gradient is the coarse mesh's real slope, which is the surface the
// geometry is being blended onto.
//
// Four `ground_height` taps, not four `textureLoad`s: the coarse lattice lands on
// an exact texel only when `(res − 1)` divides the cell count, which the island's
// 257/64 does and a 17/64 test fixture does not. One rule that holds for every
// resolution beats a faster one that holds for the shipped one.
fn coarse_height(uv: vec2<f32>, res: f32, origin_xz: vec2<f32>, span: f32,
                 coarse_step: f32) -> f32 {
    let g = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)) / coarse_step;
    let g0 = floor(g);
    let f = g - g0;
    let u00 = g0 * coarse_step;
    let u10 = u00 + vec2<f32>(coarse_step, 0.0);
    let u01 = u00 + vec2<f32>(0.0, coarse_step);
    let u11 = u00 + vec2<f32>(coarse_step, coarse_step);
    let h00 = ground_height(u00, res, origin_xz + u00 * span);
    let h10 = ground_height(u10, res, origin_xz + u10 * span);
    let h01 = ground_height(u01, res, origin_xz + u01 * span);
    let h11 = ground_height(u11, res, origin_xz + u11 * span);
    return mix(mix(h00, h10, f.x), mix(h01, h11, f.x), f.y);
}

// **The ground the terrain pass draws.** Both stages call THIS and neither
// computes the blend itself, so they cannot drift apart — the same construction,
// and for the same stated reason, as `ground_height` wrapping `deform_depth`
// above: *"the fragment's central-difference normal must see the same surface the
// vertex stage moved"*.
//
// Before wave CERT1 they did drift: the vertex wrote `mix(h_fine, h_coarse,
// morph)` and the fragment central-differenced the un-morphed `ground_height`, so
// over the last 35 % of every band the geometry moved toward the coarser grid and
// the shading kept lighting the finer one. `tests/terrain_continuity.rs` measured
// the disagreement at **10.586°** of surface normal at full morph, rising
// linearly with it (1.08° at morph 0.1, 5.35° at 0.5).
//
// `uv` is the patch coordinate and `origin_xz` the patch's render-local origin,
// so the world point is derived HERE, once, rather than by each caller — the
// fragment's `world_local.xz` and the vertex's `o_span.xz + uv * span` are the
// same point and re-deriving either was a second place for the mapping to be
// wrong.
//
// **The `m <= 0` early-out is `mix(a, b, 0) = a` spelled out**, not an
// approximation: the 65 % of each band that does not morph pays exactly the four
// texel loads it paid before this wave, which is also why the goldens' near
// ground is untouched by the coarse path.
fn morphed_height(uv: vec2<f32>, res: f32, origin_xz: vec2<f32>, span: f32,
                  cells: f32, band: vec2<f32>) -> f32 {
    let world_xz = origin_xz + uv * span;
    let h_fine = ground_height(uv, res, world_xz);
    let m = morph_at(length(world_xz - view.eye.xz), band);
    if (m <= 0.0) {
        return h_fine;
    }
    let h_coarse = coarse_height(uv, res, origin_xz, span, 2.0 / max(cells, 1.0));
    return mix(h_fine, h_coarse, m);
}

// ── splat weights + procedural detail (P10.4) ────────────────────────────────

fn load_weight(ij: vec2<i32>, res: f32) -> vec4<f32> {
    let m = i32(res) - 1;
    let c = clamp(ij, vec2<i32>(0, 0), vec2<i32>(m, m));
    return textureLoad(weight_tex, c, 0);
}

// Bilinearly-sampled RGBA splat weights at unit patch coord `uv`, renormalized so
// the four channels sum to 1 (a defensive guard on hand-authored/interpolated
// weights; a zeroed sample falls back to pure layer 0).
fn sample_weights(uv: vec2<f32>, res: f32) -> vec4<f32> {
    let p = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)) * (res - 1.0);
    let i0 = floor(p);
    let f = p - i0;
    let ii = vec2<i32>(i0);
    let w00 = load_weight(ii, res);
    let w10 = load_weight(ii + vec2<i32>(1, 0), res);
    let w01 = load_weight(ii + vec2<i32>(0, 1), res);
    let w11 = load_weight(ii + vec2<i32>(1, 1), res);
    let wx0 = mix(w00, w10, f.x);
    let wx1 = mix(w01, w11, f.x);
    var w = mix(wx0, wx1, f.y);
    let s = w.x + w.y + w.z + w.w;
    if (s > 1e-4) {
        w = w / s;
    } else {
        w = vec4<f32>(1.0, 0.0, 0.0, 0.0);
    }
    return w;
}

// ── hole mask (P21.2) ──────────────────────────────────────────────────────

// Is sample `(i, j)` holed? Out-of-range reads return zero (WGSL's rule), which
// is exactly what the 1×1 sentinel texture of an un-carved tile relies on.
fn load_hole(ij: vec2<i32>) -> bool {
    let word = textureLoad(hole_tex, vec2<i32>(ij.x >> 5u, ij.y), 0).r;
    return ((word >> (u32(ij.x) & 31u)) & 1u) != 0u;
}

// **THE POISON RULE, raster half.** A holed sample removes the surface from every
// cell that interpolates it, so one holed corner of the bilinear cell under `uv`
// kills the whole cell — character for character the rule
// `inf_terrain::TerrainData::height_at` applies on the CPU. The two MUST agree:
// if the raster poisoned less than the query, a capsule would stand on ground
// nobody draws; if it poisoned more, the cave's rim would be eaten away.
fn is_holed(uv: vec2<f32>, res: f32) -> bool {
    let p = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)) * (res - 1.0);
    let m = i32(res) - 1;
    let i0 = clamp(vec2<i32>(floor(p)), vec2<i32>(0), vec2<i32>(m));
    let i1 = min(i0 + vec2<i32>(1, 1), vec2<i32>(m, m));
    return load_hole(i0)
        || load_hole(vec2<i32>(i1.x, i0.y))
        || load_hole(vec2<i32>(i0.x, i1.y))
        || load_hole(i1);
}

// ── biome ids (P19.2) ────────────────────────────────────────────────────────

// One biome id at integer texel `ij`, clamped to the tile (mirrors `load_weight`).
// `textureLoad` on an integer texture: a biome id is CATEGORICAL, so it must never
// be filtered — the "average" of ids 1 and 3 is not id 2, it is nonsense.
fn load_biome(ij: vec2<i32>, res: f32) -> u32 {
    let m = i32(res) - 1;
    let c = clamp(ij, vec2<i32>(0, 0), vec2<i32>(m, m));
    return textureLoad(biome_tex, c, 0).r;
}

// The biome id at unit patch coord `uv`, resolved NEAREST (round to the closest
// sample) for the same reason: ids are labels, not quantities. This is the one
// terrain input that is deliberately not bilinear, so a painted boundary shows as
// the hard edge it actually is instead of a gradient through ids nobody painted.
fn sample_biome(uv: vec2<f32>, res: f32) -> u32 {
    let p = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)) * (res - 1.0);
    return load_biome(vec2<i32>(round(p)), res);
}

// Cheap 2D value-noise (hash lattice → smooth interpolation), range [0, 1].
fn hash21(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    p3 = p3 + dot(p3, vec3<f32>(p3.y, p3.z, p3.x) + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn vnoise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash21(i);
    let b = hash21(i + vec2<f32>(1.0, 0.0));
    let c = hash21(i + vec2<f32>(0.0, 1.0));
    let d = hash21(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// 4-octave fBm, range ~[0, 1].
fn fbm2(p: vec2<f32>) -> f32 {
    var v = 0.0;
    var amp = 0.5;
    var freq = p;
    for (var i = 0; i < 4; i = i + 1) {
        v = v + amp * vnoise(freq);
        freq = freq * 2.0;
        amp = amp * 0.5;
    }
    return v;
}

// Triplanar axis weights: normalized pow(|n|, sharpness) over the YZ/XZ/XY planes
// (mirrors `triplanar_axis_weights` in passes/terrain.rs, which unit-tests it).
fn triplanar_weights(n: vec3<f32>, sharpness: f32) -> vec3<f32> {
    var w = pow(abs(n), vec3<f32>(sharpness));
    let s = w.x + w.y + w.z;
    if (s > 0.0) {
        w = w / s;
    } else {
        w = vec3<f32>(0.0, 1.0, 0.0);
    }
    return w;
}

// A world-space triplanar detail grain in [0, 1]: value-noise projected on the
// three world planes at `1/tex_scale`, blended by the triplanar axis weights so
// steep faces read the vertical (XY/YZ) projections instead of a stretched top.
fn triplanar_grain(world: vec3<f32>, n: vec3<f32>, tex_scale: f32) -> f32 {
    let scale = 1.0 / max(tex_scale, 0.001);
    let gx = vnoise(world.yz * scale);
    let gy = vnoise(world.xz * scale);
    let gz = vnoise(world.xy * scale);
    let tw = triplanar_weights(n, 4.0);
    return gx * tw.x + gy * tw.y + gz * tw.z;
}

/// What [`terrain_layers`] resolved: the weight-blended surface of whichever
/// splat layers bind a virtual material, and how much of the fragment they cover.
struct TerrainLayered {
    albedo: vec3<f32>,
    roughness: f32,
    /// Sum of the weights of the layers that were textured, in `[0, 1]`. It is
    /// the mix factor against the flat-colour blend, so a terrain where only two
    /// of four layers carry materials fades correctly into the colours of the
    /// two that do not, rather than darkening toward black.
    coverage: f32,
    any: bool,
};

/// **The axis weight above which a fragment keeps the single planar fetch**
/// (wave TER2a, clause 4).
///
/// `triplanar_weights` normalises `pow(abs(n), 4)`, so a surface at angle `t`
/// from horizontal reads `tw.y` about `cos^4 / (cos^4 + sin^4)`: 0.995 at 15
/// degrees, 0.983 at 20, 0.90 at 30, 0.5 at 45. At 0.98 everything flatter than
/// about twenty degrees keeps the one fetch it has always had and pays nothing
/// for this feature but two compares.
///
/// **Why a cut rather than a continuous blend.** A continuous blend is three
/// fetches everywhere, which is the price Wave T's own comment named ("three
/// times the fetches for the axis-weighted version") and the reason it was a
/// follow-up rather than a feature. The cut spends them where the planar
/// projection is visibly wrong — the alpine faces and the sea cliffs, which on
/// this island is about a fifteenth of the ground.
const TRIPLANAR_PLANAR_CUT: f32 = 0.98;

/// A plane whose axis weight is under this cannot change a texel of an 8-bit
/// output and is skipped. The layer weight's own 0.0039, one level up.
const TRIPLANAR_AXIS_CUT: f32 = 0.004;

/// **Weight-blend the four splat layers' virtual materials** (Wave T, §3 B).
///
/// A planar XZ projection, deliberately: the shared material is authored as a
/// tiling ground texture and terrain is a heightfield, so `world.xz / tex_scale`
/// is the parametrization the content is drawn for. It stretches on a cliff
/// face — the honest bound, and the same one every heightfield engine has until
/// it pays for a triplanar variant (three times the fetches for the axis-weighted
/// version). The procedural grain above already runs triplanar, so the *break-up*
/// on a steep face survives; only the material's own pattern stretches. A
/// triplanar layer sample is the named follow-up.
///
/// The gradients are taken **once, at the top, outside every branch**: `dpdx` of
/// a value computed inside a divergent branch has no neighbour to difference
/// against. That is the same rule `vt_surface`'s callers follow and the reason it
/// takes its derivatives as parameters.
fn terrain_layers(world: vec3<f32>, n: vec3<f32>, w: vec4<f32>, tex_scale: f32) -> TerrainLayered {
    var out: TerrainLayered;
    out.albedo = vec3<f32>(0.0);
    out.roughness = 0.0;
    out.coverage = 0.0;
    out.any = false;
    if (!vt_active()) {
        return out;
    }
    let inv = 1.0 / max(tex_scale, 0.001);
    // **ABSOLUTE world XZ, not render-local.** `world_local` is rebased every
    // time the floating origin snaps (10 m steps), so a uv derived from it would
    // slide the material across the ground as the player walks — a shift of
    // `10 / tex_scale` tiles, which for a 4 m tiling is two and a half tiles of
    // visible jump. `grid_axis_viewport.xy` is the render-local position of the
    // world X/Z axes, i.e. `-origin.xz`, so subtracting it undoes the rebase.
    // The derivatives are unaffected either way (a constant offset), so this
    // costs one subtraction and buys a material that stays where it was put.
    //
    // The bound, stated: this is an f32 world coordinate, so its resolution is
    // about 4 mm at 50 km from the origin. That is under a texel of a 2K
    // material tiled at 4 m out to roughly that range, and past it the tiling
    // begins to quantise. The procedural grain above it has always been
    // render-local and therefore *does* slide; it is a ±15 % tint and nobody has
    // seen it. A material is not.
    let uv = (world.xz - view.grid_axis_viewport.xy) * inv;
    // **Six derivative lanes, not twelve** (TER2a). The three projections are
    // three swizzles of one position, so their six ddx/ddy pairs are six
    // swizzles of the position's own two — taken here, once, unconditionally.
    //
    // Measured on the island: with the derivatives taken per projection
    // (dpdx(uv), dpdx(uv_x), dpdx(uv_z), and the same for dpdy) the terrain pass
    // paid +0.252 ms even with every fragment on the planar path, against
    // +0.126 ms for the extra fetches themselves. The derivative work was twice
    // the price of the thing it was for.
    let dwx = dpdx(world);
    let dwy = dpdy(world);
    let ddx = dwx.xz * inv;
    let ddy = dwy.xz * inv;
    // **THE OTHER TWO PLANES** (TER2a, clause 4). Wave T shipped the planar XZ
    // projection with its own bound written on it: "it stretches on a cliff
    // face — the honest bound... A triplanar layer sample is the named
    // follow-up." This is it.
    //
    // A heightfield's cliff is where a horizontal projection is *worst*: the
    // surface is nearly vertical, so a metre of face maps onto centimetres of
    // uv and the material smears into vertical streaks. The island's showcase is
    // the 948.7 m North Shore peak, whose faces run past sixty degrees.
    //
    // World-anchored on the same terms as `uv`, and derived UNCONDITIONALLY —
    // `dpdx` of a value computed inside a divergent branch has no neighbour to
    // difference against, which is the same rule the planar pair above follows.
    // Six extra ALU derivatives cost nothing; the FETCHES are what is gated.
    // World-anchored on **all three** axes. `grid_axis_viewport.xy` is
    // `-origin.xz` and `mode_axis.y` is `-origin.y` (the same recovery
    // `atmosphere_apply` and `cloud` already make), so the vertical projections
    // stay put when the floating origin snaps in Y as well. Leaving `y`
    // render-local would slide a cliff's material vertically by `10 / tex_scale`
    // tiles every rebase — the exact defect the XZ subtraction above exists to
    // prevent, on the axis a cliff's material actually runs along.
    let wx = world - vec3<f32>(
        view.grid_axis_viewport.x,
        view.mode_axis.y,
        view.grid_axis_viewport.y,
    );
    let uv_x = wx.zy * inv;
    let uv_z = wx.xy * inv;
    let ddx_x = dwx.zy * inv;
    let ddy_x = dwy.zy * inv;
    let ddx_z = dwx.xy * inv;
    let ddy_z = dwy.xy * inv;
    // The same axis weights the procedural grain has always used, at the same
    // sharpness, so the grain and the material break up on the same plane.
    let tw = triplanar_weights(n, 4.0);
    let planar = tw.y >= TRIPLANAR_PLANAR_CUT;
    let weights = array<f32, 4>(w.x, w.y, w.z, w.w);
    for (var k = 0u; k < 4u; k = k + 1u) {
        let wk = weights[k];
        // Below a 256th the layer cannot change a texel of an 8-bit output, and
        // the weights are stored as bytes summing to 255 — so this threshold is
        // the weight's own quantum rather than a tuned epsilon.
        if (wk < 0.0039) {
            continue;
        }
        let slots = material.slots[k].xyz;
        if (!vt_bound(slots.x) && !vt_bound(slots.y) && !vt_bound(slots.z)) {
            continue;
        }
        // **A LAYER'S SCALARS ARE A FALLBACK, NOT A glTF FACTOR** (wave TER2a).
        //
        // `vt_surface` multiplies: `out.albedo = albedo * texel`, and
        // `out.roughness = roughness * orm.g`. That is right for a mesh, where
        // `base_color` is glTF's `baseColorFactor` and a separate field from any
        // fallback. A terrain layer has ONE colour, and it is what the surface
        // shades with when nothing is bound — so passing it as a factor renders
        // a bound ground material at the product of its own colour and itself.
        //
        // Measured on the island's own numbers when its four layers were first
        // bound: the textured frame came out at a mean luminance of 56.4 against
        // the same frame's 114.3 untextured — **2.03x too dark** — because grass
        // shaded at 0.086 x 0.078 instead of 0.078. Wave T could not have seen
        // it: no `.inf_mat` in the repository named a texture, so this branch had
        // never run over content.
        //
        // The fallback still reaches the pixel: the call site mixes
        // `material.albedo` toward this result by `coverage`, so a layer with no
        // albedo texture contributes its colour exactly as before.
        var tint = material.albedo[k].rgb;
        if (vt_bound(slots.x)) {
            tint = vec3<f32>(1.0);
        }
        var rough = material.params[k].x;
        if (vt_bound(slots.z)) {
            rough = 1.0;
        }
        var la = vec3<f32>(0.0);
        var lr = 0.0;
        if (planar) {
            let s = vt_surface(slots, uv, ddx, ddy, tint, 1.0, 0.0, rough);
            la = s.albedo;
            lr = s.roughness;
        } else {
            // Axis-weighted over the three planes, each one skipped when it
            // cannot contribute a texel. A vertical face is `tw.y = 0` and pays
            // for TWO fetches, not three — the worst case is a 45-degree corner.
            var acc = vec3<f32>(0.0);
            var accr = 0.0;
            var accw = 0.0;
            if (tw.x > TRIPLANAR_AXIS_CUT) {
                let sx = vt_surface(slots, uv_x, ddx_x, ddy_x, tint, 1.0, 0.0, rough);
                acc = acc + sx.albedo * tw.x;
                accr = accr + sx.roughness * tw.x;
                accw = accw + tw.x;
            }
            if (tw.y > TRIPLANAR_AXIS_CUT) {
                let sy = vt_surface(slots, uv, ddx, ddy, tint, 1.0, 0.0, rough);
                acc = acc + sy.albedo * tw.y;
                accr = accr + sy.roughness * tw.y;
                accw = accw + tw.y;
            }
            if (tw.z > TRIPLANAR_AXIS_CUT) {
                let sz = vt_surface(slots, uv_z, ddx_z, ddy_z, tint, 1.0, 0.0, rough);
                acc = acc + sz.albedo * tw.z;
                accr = accr + sz.roughness * tw.z;
                accw = accw + tw.z;
            }
            // Renormalise inside the planes that fired, for the same reason the
            // layer blend renormalises inside its own coverage: a face that
            // skipped a plane must not go dark by however much that plane
            // weighed.
            let inv_w = 1.0 / max(accw, 1e-4);
            la = acc * inv_w;
            lr = accr * inv_w;
        }
        out.albedo = out.albedo + la * wk;
        out.roughness = out.roughness + lr * wk;
        out.coverage = out.coverage + wk;
        out.any = true;
    }
    if (out.coverage > 0.0) {
        // Renormalise INSIDE the covered fraction, so the textured layers'
        // colours are their own rather than scaled by how much of the fragment
        // they happen to own; `coverage` then carries that information to the
        // mix at the call site. Getting this wrong is how a half-textured
        // terrain goes dark.
        out.albedo = out.albedo / out.coverage;
        out.roughness = out.roughness / out.coverage;
    }
    return out;
}

@vertex
fn vs(in: VIn) -> VOut {
    let uv = in.uv_skirt.xy;
    let skirt = in.uv_skirt.z;
    let span = in.o_span.w;
    let band = in.params.xy;
    let cells = max(in.params.z, 1.0);
    let res = in.params.w;

    // The LOD morph, the deformation field and the heightfield, in ONE call the
    // fragment stage makes too (P22.1's construction, extended to the morph by
    // wave CERT1 — see `morphed_height`).
    let h = morphed_height(uv, res, in.o_span.xz, span, cells, band);

    // Render-local position: tile origin + planar offset + displaced height.
    var pos = in.o_span.xyz + vec3<f32>(uv.x * span, h, uv.y * span);
    // Skirt: drop the boundary ring straight down to seal cracks.
    pos.y = pos.y - skirt * in.skirt_depth;

    var out: VOut;
    out.clip = view.view_proj * vec4<f32>(pos, 1.0);
    out.world_local = pos;
    out.uv = uv;
    out.span_res_cells = vec3<f32>(span, res, cells);
    out.origin_band = vec4<f32>(in.o_span.xz, band);
    return out;
}

// **The depth-prepass fragment** (wave VIS1a). No colour target, no lighting, no
// splat — the one thing a depth-only terrain draw still needs is the P21.2 hole
// discard, because a carved cell has no surface and a prepass that wrote depth
// there would occlude the cave mouth for SSAO, TAA and SSR while the colour pass
// draws the voxel volume through it. Character for character the test `fs` makes
// below, and it must stay that way: two hole rules that disagree is a cave whose
// AO is a wall.
@fragment
fn fs_depth(in: VOut) {
    if (is_holed(in.uv, max(in.span_res_cells.y, 2.0))) {
        discard;
    }
}

@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
    let span = in.span_res_cells.x;
    let res = max(in.span_res_cells.y, 2.0);
    let cells = max(in.span_res_cells.z, 1.0);
    let origin_xz = in.origin_band.xy;
    let band = in.origin_band.zw;
    let texel = 1.0 / (res - 1.0);

    // ── P21.2 HOLE DISCARD ─────────────────────────────────────────
    // The first statement in the fragment, before anything is computed and
    // before depth is written: a carved cell has no surface, so the clipmap
    // draws nothing there and whatever the voxel volume put behind it shows
    // through. An un-carved tile's 1×1 sentinel makes this a texture fetch that
    // always says "no", so every pre-P21.2 terrain golden stays byte-stable.
    //
    // **This also handles the skirt**, and structurally rather than by a second
    // rule. A skirt vertex carries its boundary `uv`, so a skirt fragment tests
    // the very sample it exists to seal the crack under — which means a skirt
    // wall can never survive inside a hole. The honest consequence, stated
    // because it is a real one: where a hole reaches a patch boundary the crack
    // seal goes with it, so a fine↔coarse LOD gap can open at that edge. That is
    // acceptable and preferable — a gap where there is no ground reads as the
    // cave it is, while a skirt wall hanging in a cave mouth reads as a bug.
    if (is_holed(in.uv, res)) {
        discard;
    }

    // Central-difference normal, through the SAME `morphed_height` the vertex
    // stage displaced with — the heightfield, the P22.1 deformation and now the
    // LOD morph, so the shading follows the geometry into a footprint AND through
    // a morph band instead of lighting the un-morphed fine surface (wave CERT1;
    // the disagreement was **10.586°** at full morph).
    //
    // ── THE TAP SPACING IS MEASURED, NOT ASSUMED (wave CERT1, defect E) ───────
    //
    // A patch binds ONLY its own page — there is no apron ring in the upload —
    // and both `load_texel` and `sample_height` clamp. So a tap at `uv.x < texel`
    // does not read the neighbour's `h[-1]`, it re-reads `h[0]`, and dividing that
    // difference by `2 · world_step` measured **exactly half** the true gradient:
    // a flattened, discontinuous shading line one texel wide down every tile edge,
    // 1 m every 256 m at ring 0 across the whole 7.2 km island. Measured at
    // **0.4998×** the analytic gradient, an **18.290°** step between a tile's edge
    // column and the column one texel inside it.
    //
    // The fix is to clamp the tap uv HERE and divide by the distance the two taps
    // are actually apart, which degrades a central difference into a correctly
    // scaled one-sided difference exactly at the edge and is unchanged one texel
    // in. Measured on the gate's 60 m-relief field: the edge column's |dh/dx|
    // goes **0.4875× → 0.9751×** of the analytic gradient, and the step between
    // that column and the one texel inside it goes **14.688° → 0.707°**.
    //
    // **Stated plainly: this removes the systematic HALVING, not the seam.** The
    // two sides of an edge now estimate the slope with opposite one-sided
    // differences, which disagree by the surface's curvature over one texel, so
    // the CROSS-tile step rises from **0.871° to 1.318°** — the two sides used to
    // agree with each other precisely because both were half the truth, which is a
    // seam invisible to a cross-tile comparison and glaring in the frame. Trading
    // a 14.688° flattened line for a 1.318° estimator difference is the whole of
    // this fix, and it is a trade rather than a cure: full continuity needs the
    // neighbour's texel, i.e. a one-sample APRON RING in the page upload — a
    // `.inf_terrain` page-format change, and out of scope for this wave.
    let ul = max(in.uv.x - texel, 0.0);
    let ur = min(in.uv.x + texel, 1.0);
    let vd = max(in.uv.y - texel, 0.0);
    let vu = min(in.uv.y + texel, 1.0);
    let hl = morphed_height(vec2<f32>(ul, in.uv.y), res, origin_xz, span, cells, band);
    let hr = morphed_height(vec2<f32>(ur, in.uv.y), res, origin_xz, span, cells, band);
    let hd = morphed_height(vec2<f32>(in.uv.x, vd), res, origin_xz, span, cells, band);
    let hu = morphed_height(vec2<f32>(in.uv.x, vu), res, origin_xz, span, cells, band);
    let dhdx = (hr - hl) / max((ur - ul) * span, 1e-6);
    let dhdz = (hu - hd) / max((vu - vd) * span, 1e-6);
    let n = normalize(vec3<f32>(-dhdx, 1.0, -dhdz));

    // ── P10.4 SPLAT MATERIAL HOOK ──────────────────────────────────────────
    // Splat-blended layered material: blend the four layers' albedo/roughness by
    // the per-sample weight texture, add a world-space triplanar detail grain
    // (so steep faces don't stretch), then a large-scale fBm macro variation.
    // The lighting below is the shared PBR-lite path (now roughness-aware).
    let w = sample_weights(in.uv, res);
    var albedo = w.x * material.albedo[0].rgb
        + w.y * material.albedo[1].rgb
        + w.z * material.albedo[2].rgb
        + w.w * material.albedo[3].rgb;
    var roughness = clamp(
        w.x * material.params[0].x + w.y * material.params[1].x
            + w.z * material.params[2].x + w.w * material.params[3].x,
        0.04, 1.0);
    let tex_scale = w.x * material.params[0].y + w.y * material.params[1].y
        + w.z * material.params[2].y + w.w * material.params[3].y;

    // ── WAVE T · MATERIAL LAYERING VIA VIRTUAL TEXTURES (§3 B) ─────────────
    // *"For terrain or large props, store blend weights in a virtual texture
    // mask and sample shared, tileable PBR materials. This delivers infinite
    // visual variety with minimal unique texture storage."*
    //
    // The weights half has shipped since P10.4 — a per-sample RGBA8 mask that
    // renormalises to exactly 255. What was missing is the other half: a layer
    // was a FLAT COLOUR, and the component said so in its own doc comment
    // ("a texture GUID is deliberately absent... per-layer albedo/normal/ORM
    // texture refs are the documented follow-up"). This is that follow-up.
    //
    // Each layer that binds one samples a **shared, tileable** virtual material
    // at `world.xz / tex_scale`, which is the same `tex_scale` the procedural
    // grain already used — so a layer's authored tiling means one thing, not
    // two. Sharing is what makes the storage claim true: `VtTextures` dedupes
    // by GUID, so four terrains using one rock material register it once, and a
    // 2K rock set covers a 50 km world at whatever density `tex_scale` asks for.
    //
    // Weight-gated per layer: a layer whose weight is zero here contributes no
    // texture fetch at all, so the cost is what is actually visible at this
    // fragment (typically one or two layers) rather than four unconditionally.
    let layered = terrain_layers(in.world_local, n, w, tex_scale);
    if (layered.any) {
        albedo = mix(albedo, layered.albedo, layered.coverage);
        roughness = clamp(mix(roughness, layered.roughness, layered.coverage), 0.04, 1.0);
    }

    // Triplanar detail grain (subtle multiplicative tint, ±15%).
    let grain = triplanar_grain(in.world_local, n, tex_scale);
    albedo = albedo * (0.85 + 0.30 * grain);

    // Macro variation: large-scale fBm brightening/darkening (signed, ±amp).
    let macro_fbm = 2.0 * fbm2(in.world_local.xz * 0.01) - 1.0;
    albedo = albedo * (1.0 + material.macro_amp.x * macro_fbm);
    albedo = clamp(albedo, vec3<f32>(0.0), vec3<f32>(1.0));

    // P22.1 COMPACTION DARKENING. Pressed ground is denser ground: the air is
    // squeezed out of the snow, the mud closes over, and what light gets in
    // scatters fewer times before it is absorbed. A single multiplicative term
    // on the splat blend, at most −25% at full depth — deliberately NOT a second
    // copy of the P20.3 wetness model, which is a different physical story (a
    // film of water on top) with its own roughness term and its own uniform.
    // `deform_depth` is 0 in every scene without deformation, so this is a
    // multiply by 1 and every pre-P22.1 terrain golden stays byte-stable.
    if (deform_enabled()) {
        let pressed = clamp(deform_depth(in.world_local.xz) / max(dfm.params.y, 1e-4), 0.0, 1.0);
        albedo = albedo * (1.0 - 0.25 * pressed);
    }
    // ───────────────────────────────────────────────────────────────────────

    // Biomes view mode (P19.2): tint by the per-sample biome id instead of by the
    // splat material — the terrain reads as a map of the painted vocabulary. This
    // is checked BEFORE the unlit branch because Biomes sets BOTH flags (`flags.x`
    // makes every other kind of geometry in the frame render unlit); testing unlit
    // first would swallow the tint.
    //
    // The colour is the palette entry, shaped by a wrapped N·L so the landform's
    // relief still reads under a flat fill (0.55…1.0 of the tint). It is a pure
    // function of the surface normal and the view's sun direction — no lights, no
    // shadows, no GI, no fog — so the mode is deterministic frame to frame and
    // between adapters, which is what makes it goldenable.
    //
    // Like `flags.x`, `flags.y` is EXACTLY 0.0 in every other mode, so this branch
    // is present-but-false and the arithmetic below runs instruction-for-instruction
    // unchanged — every pre-P19.2 terrain golden stays byte-stable.
    if (view.flags.y > 0.5) {
        let id = sample_biome(in.uv, res);
        let tint = biome_palette.colors[id].rgb;
        let ndl_b = clamp(dot(n, normalize(view.sun_dir.xyz)), 0.0, 1.0);
        return vec4<f32>(tint * (0.55 + 0.45 * ndl_b), 1.0);
    }

    // Unlit view mode (R-P2): return the splat-blended albedo directly, skipping
    // the sun/ambient/spec lighting below. Terrain carries no emissive term. The
    // flag is 0 in the default Lit mode, so the terrain golden stays byte-stable.
    if (view.flags.x > 0.5) {
        return vec4<f32>(albedo, 1.0);
    }

    // P20.3 SHORELINE WETNESS. Terrain is where this reads: a coastline is a
    // terrain crossing a water level, and the darkened, glossier band at that
    // crossing is what makes water look like it is sitting IN the landscape
    // rather than on top of it. A pure function of the fragment's world position
    // and the frame's water bodies — no camera, no screen space, no map (see
    // `shaders/wetness.wgsl`). Placed AFTER the Biomes and Unlit returns so both
    // debug views keep showing the authored data rather than a wetted version of
    // it. `wet.dims.x` is 0 on every scene without water, so the branch is
    // present-but-false and every pre-P20.3 terrain golden is byte-stable.
    if (wet.dims.x > 0u) {
        let wetted = wet_apply(in.world_local, albedo, roughness);
        albedo = wetted.rgb;
        roughness = clamp(wetted.a, 0.04, 1.0);
    }

    let sun = normalize(view.sun_dir.xyz);
    let ndl = max(dot(n, sun), 0.0);
    // Hemispheric ambient (sky above / ground below), or the dynamic-GI probe
    // irradiance when GI is on.
    let up = clamp(n.y * 0.5 + 0.5, 0.0, 1.0);
    // Wave FIX3: one door, shared with the five mesh passes. The terrain's own
    // hemispheric constant is brighter than theirs (it is ground, and ground
    // sees more sky), and it stays the fallback for a level that asked for no
    // computed ambient.
    let ambient = ambient_irradiance(
        in.world_local,
        n,
        mix(vec3<f32>(0.05, 0.06, 0.07), vec3<f32>(0.16, 0.20, 0.26), up),
    );
    // A cheap roughness-aware specular glint (Blinn-ish): smoother layers (lower
    // roughness) get a tighter, brighter highlight, so roughness reads visibly.
    let view_dir = normalize(view.eye.xyz - in.world_local);
    let half_v = normalize(sun + view_dir);
    let gloss = (1.0 - roughness) * (1.0 - roughness);
    let spec_power = mix(8.0, 128.0, gloss);
    let spec = pow(max(dot(n, half_v), 0.0), spec_power) * gloss * 0.4;
    // The direct sun (+ its glint) receives the cascaded shadow factor; SSAO
    // modulates only the ambient term.
    var direct = ndl * vec3<f32>(1.15, 1.10, 1.0);
    var spec_term = vec3<f32>(spec);
    if (sun_shadowing_enabled()) {
        let sf = shadow_factor(in.world_local, n);
        direct = direct * sf;
        spec_term = spec_term * sf;
    }
    // P17.3: the cloud layer's soft, large-scale sun occlusion. Terrain is where
    // this reads most — a kilometre-wide cloud shadow drifting over a valley is
    // the whole point of baking the map. Guarded like the CSM block above, so a
    // cloudless scene is byte-identical.
    if (atmos.clouds.x > 0.5 && atmos.cloud_shadow.x > 0.0) {
        let cf = cloud_shadow_factor(in.world_local);
        direct = direct * cf;
        spec_term = spec_term * cf;
    }
    let ao = textureSampleLevel(ao_tex, ao_smp, in.clip.xy / view.grid_axis_viewport.zw, 0.0).r;
    var lo = albedo * (ambient * ao + direct) + spec_term;
    // P18.4 GI specular. Terrain has no `f0` of its own (it is a dielectric splat
    // blend, and its existing glint is a direct-sun Blinn lobe), so this is an
    // ADDITIVE environment term at the dielectric 0.04 rather than a replacement —
    // which also means a terrain golden with GI off runs the identical arithmetic.
    if (gi.params.x > 0.5 && gi.params2.x > 0.5) {
        lo = lo + gi_specular(in.world_local, n, view_dir, roughness, vec3<f32>(0.04)) * ao;
    }

    // HDR-linear haze; the post tonemap pass (ACES + exposure) runs afterward.
    // P17.2: replaced by physical aerial perspective + height fog when the scene
    // has an atmosphere. Terrain is the pass that shows this off — it is the only
    // geometry that reliably reaches the horizon.
    let dist = length(in.world_local - view.eye.xyz);
    let haze = 1.0 - exp(-dist * 0.0025);
    var col = mix(lo, vec3<f32>(0.055, 0.081, 0.120), haze * 0.5);
    if (atmos.params.x > 0.5) {
        col = atmos_apply(lo, in.world_local);
    }
    return vec4<f32>(col, 1.0);
}
