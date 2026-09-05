// vis_resolve.wgsl — the material-resolve pass (P28.1). Composed
// `ShaderKind::Lit(2)`, so `common_view` + the VSM receiver + the environment
// lighting + the atmosphere + cloud shadows + wetness + virtual-texture sampling
// are all prepended, and this shader reaches every one of them through the
// SAME bindings the forward paths use. That is not convenience: `vsm_receive`'s
// own arm asserts `vsm_atlas` is declared exactly once in the tree, so a resolve
// that reached the shadow atlas by any other door would be a second declaration
// of a binding whose whole point is that it has one.
//
// One fullscreen triangle. Per pixel:
//   1. load the `Rg32Uint` visibility id and unpack instance ⊕ meshlet ⊕ triangle
//      (IB-8: sixty-four bits in two words, because WGSL has no `u64`);
//   2. pull the triangle's three vertices from the shared pools and put them
//      through the instance's model matrix and `view.view_proj` — the same
//      jittered matrix the buffer was rasterized with (the P27.1 jitter law);
//   3. solve for the perspective-correct barycentrics AND their exact screen
//      gradients (see `vis_bary`);
//   4. shade, with `dpdx`/`dpdy` replaced by those gradients everywhere.
//
// **Why the gradients are solved and not sampled.** A fragment shader's `dpdx`
// is a difference against the pixel next door inside a 2 x 2 quad. In a resolve
// pass the pixel next door is a *different triangle*, so that difference is a
// step across a silhouette and the mip level it justifies is nonsense — one
// wrong tile per edge pixel, every frame. Solving the barycentrics analytically
// makes the gradient a property of the triangle rather than of the neighbourhood.
//
// **Screen-space material bins.** The bin is the id itself. A `u32` visibility
// value is constant across a triangle, so every fragment of one triangle takes
// the same branch — the divergence a resolve pass pays is bounded by the number
// of distinct triangles in a wave, which is what "binned by material" buys in a
// tiled implementation. A second pass that sorts pixels into per-material
// indirect dispatches is the classic refinement and is measured-and-not-taken in
// `docs/memos/p28-1-visbuffer.md` §4: with one lit pipeline for every meshlet
// instance (materials here are per-instance CONSTANTS, not per-instance shader
// permutations) the bins would all name the same program.

struct Meshlet {
    center: vec3<f32>,
    radius: f32,
    cone_axis: vec3<f32>,
    cone_cutoff: f32,
    vertex_offset: u32,
    triangle_offset: u32,
    vertex_count: u32,
    triangle_count: u32,
    error: f32,
    parent_error: f32,
    lod_level: u32,
    // **The material slot** (Wave T, the texture document's section 3 C). This
    // word carries `MeshletRec::group` on disk and was `pad` here because no
    // shader read it; `inf_vgeom::stream::stage_page` now overwrites it with the
    // meshlet's material slot on the way into the pool, unconditionally, so the
    // word means one thing on the GPU for every container version. `0` is the
    // instance's own material, which is what every mesh cooked before Wave T
    // resolves to.
    material_slot: u32,
};

// The flat instance table rides an `Rgba32Float` TEXTURE, one row an instance,
// eleven used texels of sixteen. Not a storage buffer, and the reason is a
// measured device ceiling rather than taste: `Limits::default()` grants eight
// storage buffers a stage, the shared environment group already spends four (GI
// SH, the VT table, the VSM table, its projections) and the four meshlet pools
// are the other four. The ninth is refused by `create_pipeline_layout`. See
// `VIS_FRAGMENT_STORAGE_BINDINGS` in `crates/inf-render/src/passes/visbuffer.rs`.
//
// `textureLoad` is valid in the vertex, fragment AND compute stages, so all
// three visibility passes read the table through one representation.
const VIS_INSTANCE_TEXELS: i32 = 16;

struct Instance {
    model: mat4x4<f32>,
    n0: vec3<f32>,
    n1: vec3<f32>,
    n2: vec3<f32>,
    color: vec4<f32>,
    emissive: vec3<f32>,
    metallic: f32,
    roughness: f32,
    vt: vec3<u32>,
};

fn vis_instance(id: u32) -> Instance {
    let y = i32(id);
    let c0 = textureLoad(v_instance_tex, vec2<i32>(0, y), 0);
    let c1 = textureLoad(v_instance_tex, vec2<i32>(1, y), 0);
    let c2 = textureLoad(v_instance_tex, vec2<i32>(2, y), 0);
    let c3 = textureLoad(v_instance_tex, vec2<i32>(3, y), 0);
    let n0 = textureLoad(v_instance_tex, vec2<i32>(4, y), 0);
    let n1 = textureLoad(v_instance_tex, vec2<i32>(5, y), 0);
    let n2 = textureLoad(v_instance_tex, vec2<i32>(6, y), 0);
    let col = textureLoad(v_instance_tex, vec2<i32>(7, y), 0);
    let emi = textureLoad(v_instance_tex, vec2<i32>(8, y), 0);
    // (threshold, metallic, roughness, max_scale)
    let pbr = textureLoad(v_instance_tex, vec2<i32>(9, y), 0);
    // (pick_id, vt0, vt1, vt2) — u32 words, so they are bitcast rather than
    // converted. `bitcast` is the right call and the FIRST version of this
    // comment gave the wrong reason for it: it said an f32 round-trip breaks
    // "anything past 2^24", which is where the danger is NOT. A virtual-texture
    // slot is `handle + 1`, a small integer, and every u32 below 2^23 has an
    // all-zero f32 exponent field — so as a float every real handle here is
    // SUBNORMAL, and WGSL permits an implementation to flush subnormals to zero.
    // Converting would turn slot 1 into slot 0, which reads as "this surface
    // binds no texture" and shows up as a silently untextured frame rather than
    // as an error (P28.1 audit). A bitcast of a loaded texel is a
    // reinterpretation and takes no arithmetic, which is why it survives;
    // `parity_textured_virtual_texture`'s closing `assert_ne!` is what witnesses
    // that on whatever adapter runs, because a flushed slot makes the textured
    // frame byte-identical to the untextured one.
    let ids = textureLoad(v_instance_tex, vec2<i32>(10, y), 0);
    var out: Instance;
    out.model = mat4x4<f32>(c0, c1, c2, c3);
    out.n0 = n0.xyz;
    out.n1 = n1.xyz;
    out.n2 = n2.xyz;
    out.color = col;
    out.emissive = emi.xyz;
    out.metallic = pbr.y;
    out.roughness = pbr.z;
    out.vt = vec3<u32>(bitcast<u32>(ids.y), bitcast<u32>(ids.z), bitcast<u32>(ids.w));
    return out;
}

// Group 3: the visibility buffer + the shared pools + the FLAT instance table.
@group(3) @binding(0) var vis_buf: texture_2d<u32>;
// **The meshlet vertex record's stride, in floats** (P28.2). One `VgeomVertex`
// is position(3) + normal(3) + uv(2) + a packed tangent word(1) = 9, and this
// constant is the Rust `VERTEX_REC_LEN / 4` read back — `the_shaders_vertex_
// stride_is_the_rusts` fails the day the two stop agreeing, in every shader that
// reads the pool. A wrong stride here does not error: it reads a neighbouring
// vertex's bytes as this one's position, which is a mesh that renders and is
// wrong everywhere.
const VGEOM_VSTRIDE: u32 = 9u;
@group(3) @binding(1) var<storage, read> v_positions: array<f32>;
@group(3) @binding(2) var<storage, read> v_meshlets: array<Meshlet>;
@group(3) @binding(3) var<storage, read> v_meshlet_verts: array<u32>;
@group(3) @binding(4) var<storage, read> v_meshlet_tris: array<u32>;
@group(3) @binding(5) var v_instance_tex: texture_2d<f32>;

// ── the packing contract (crates/inf-render/src/visbuffer.rs) ────────────────
const VIS_TRI_BITS: u32 = 7u;
const VIS_MESHLET_BITS: u32 = 25u;
const VIS_INSTANCE_BITS: u32 = 24u;
const VIS_EMPTY: u32 = 0u;
const VIS_MESHLET_SHIFT: u32 = VIS_TRI_BITS;
const VIS_INSTANCE_SHIFT: u32 = 0u;
// The degenerate floor, one number in two languages
// (`the_degenerate_floor_is_one_number`).
const VIS_DET_EPS: f32 = 1e-20;

const MAX_LIGHTS: u32 = 16u;

struct GpuLight {
    color: vec4<f32>,
    pos_dir: vec4<f32>,
    params: vec4<f32>,
    spot_dir: vec4<f32>,
};
struct Lights {
    count: vec4<u32>,
    items: array<GpuLight, MAX_LIGHTS>,
};
@group(1) @binding(0) var<uniform> lights: Lights;

const PI: f32 = 3.14159265359;

fn distribution_ggx(n_dot_h: f32, rough: f32) -> f32 {
    let a = rough * rough;
    let a2 = a * a;
    let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, 1e-7);
}

fn geometry_smith(n_dot_v: f32, n_dot_l: f32, rough: f32) -> f32 {
    let r = rough + 1.0;
    let k = (r * r) / 8.0;
    let gv = n_dot_v / (n_dot_v * (1.0 - k) + k);
    let gl = n_dot_l / (n_dot_l * (1.0 - k) + k);
    return gv * gl;
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn shade_light(
    n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, radiance: vec3<f32>,
    albedo: vec3<f32>, metallic: f32, rough: f32, f0: vec3<f32>,
) -> vec3<f32> {
    let h = normalize(v + l);
    let n_dot_l = max(dot(n, l), 0.0);
    if (n_dot_l <= 0.0) {
        return vec3<f32>(0.0);
    }
    let n_dot_v = max(dot(n, v), 1e-4);
    let n_dot_h = max(dot(n, h), 0.0);
    let v_dot_h = max(dot(v, h), 0.0);

    let d = distribution_ggx(n_dot_h, rough);
    let g = geometry_smith(n_dot_v, n_dot_l, rough);
    let f = fresnel_schlick(v_dot_h, f0);

    // Wave VIS1a: multi-scatter energy compensation. A single-scatter GGX
    // drops whatever the Smith term masked instead of letting it bounce again,
    // which is about a third of the lobe at roughness 1.0 and is why every
    // rough metal in this engine has been too dark since P7.1. See
    // `ggx_energy_compensation` in `env_lighting.wgsl`.
    let spec = (d * g) * f / max(4.0 * n_dot_v * n_dot_l, 1e-4)
        * ggx_energy_compensation(f0, rough, n_dot_v);
    let kd = (vec3<f32>(1.0) - f) * (1.0 - metallic);
    let diffuse = kd * albedo / PI;
    return (diffuse + spec) * radiance * n_dot_l;
}

fn point_attenuation(dist: f32, range: f32) -> f32 {
    let inv_sq = 1.0 / max(dist * dist, 1e-4);
    if (range <= 0.0) {
        return inv_sq;
    }
    let t = clamp(1.0 - pow(dist / range, 4.0), 0.0, 1.0);
    return inv_sq * t * t;
}

fn read_tri_byte(byte_addr: u32) -> u32 {
    let word = v_meshlet_tris[byte_addr >> 2u];
    let shift = (byte_addr & 3u) * 8u;
    return (word >> shift) & 0xFFu;
}

// ── barycentrics + their exact screen gradients ─────────────────────────────
//
// The Rust twin is `crate::visbuffer::VisBary::of`, and its doc block carries
// the derivation. In one line: with `M = [c0.xyw | c1.xyw | c2.xyw]`,
// `mu = M^-1 * (ndc.x, ndc.y, 1)` is LINEAR in the pixel, so its NDC gradients
// are two columns of `M^-1` and nothing is differenced anywhere.
struct VisBary {
    lambda: vec3<f32>,
    d_dx: vec3<f32>,
    d_dy: vec3<f32>,
};

fn vis_bary(c0: vec4<f32>, c1: vec4<f32>, c2: vec4<f32>, ndc: vec2<f32>,
            wh: vec2<f32>) -> VisBary {
    var out: VisBary;
    let m = mat3x3<f32>(
        vec3<f32>(c0.x, c0.y, c0.w),
        vec3<f32>(c1.x, c1.y, c1.w),
        vec3<f32>(c2.x, c2.y, c2.w),
    );
    let det = determinant(m);
    // `det != det` is the NaN clause, and it is load-bearing rather than
    // defensive: every comparison against NaN is FALSE, so `abs(det) <
    // VIS_DET_EPS` alone lets a NaN clip coordinate walk through the guard and
    // out as a NaN colour (measured on the Rust twin — P28.1 audit).
    if (abs(det) < VIS_DET_EPS || det != det) {
        out.lambda = vec3<f32>(1.0, 0.0, 0.0);
        out.d_dx = vec3<f32>(0.0);
        out.d_dy = vec3<f32>(0.0);
        return out;
    }
    let inv = mat3x3_inverse(m, det);
    let mu = inv * vec3<f32>(ndc.x, ndc.y, 1.0);
    let dmu_du = inv * vec3<f32>(1.0, 0.0, 0.0);
    let dmu_dv = inv * vec3<f32>(0.0, 1.0, 0.0);
    let d = mu.x + mu.y + mu.z;
    if (abs(d) < VIS_DET_EPS || d != d) {
        out.lambda = vec3<f32>(1.0, 0.0, 0.0);
        out.d_dx = vec3<f32>(0.0);
        out.d_dy = vec3<f32>(0.0);
        return out;
    }
    let inv_d = 1.0 / d;
    let lambda = mu * inv_d;
    let su = dmu_du.x + dmu_du.y + dmu_du.z;
    let sv = dmu_dv.x + dmu_dv.y + dmu_dv.z;
    out.lambda = lambda;
    // NDC per pixel: +2/w across, -2/h down (NDC y points up).
    out.d_dx = ((dmu_du - lambda * su) * inv_d) * (2.0 / max(wh.x, 1.0));
    out.d_dy = ((dmu_dv - lambda * sv) * inv_d) * (-2.0 / max(wh.y, 1.0));
    return out;
}

// WGSL has no `inverse`. Spelled as the adjugate over the determinant the caller
// already computed, so the guard and the division cannot disagree about which
// determinant they mean.
fn mat3x3_inverse(m: mat3x3<f32>, det: f32) -> mat3x3<f32> {
    let a = m[0];
    let b = m[1];
    let c = m[2];
    let r0 = cross(b, c);
    let r1 = cross(c, a);
    let r2 = cross(a, b);
    let inv_det = 1.0 / det;
    // rows of the adjugate become columns of the inverse.
    return mat3x3<f32>(
        vec3<f32>(r0.x, r1.x, r2.x) * inv_det,
        vec3<f32>(r0.y, r1.y, r2.y) * inv_det,
        vec3<f32>(r0.z, r1.z, r2.z) * inv_det,
    );
}

struct VsOut {
    @builtin(position) pos: vec4<f32>,
};

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VsOut {
    var out: VsOut;
    out.pos = vec4<f32>(fullscreen_ndc(i), 0.0, 1.0);
    return out;
}

struct FsOut {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};

@fragment
fn fs(in: VsOut) -> FsOut {
    var out: FsOut;
    let px = vec2<i32>(i32(in.pos.x), i32(in.pos.y));
    let id = textureLoad(vis_buf, px, 0).rg;
    // Word 1 carries `instance + 1`, so a zero there is the cleared texel and
    // nothing else. Testing word 1 alone rather than the pair is exact, not a
    // shortcut: the bias is what makes it so.
    if (id.y == VIS_EMPTY) {
        // `discard` is a statement and not a terminator, so the `return` is
        // load-bearing: without it the unpack below would run on the cleared
        // word, and `0u - 1u` is `0xFFFFFFFF` — a bounds-clamped read of an
        // unrelated instance, every frame, for every pixel the meshlets do not
        // cover.
        discard;
        out.color = vec4<f32>(0.0);
        out.depth = 0.0;
        return out;
    }

    let instance_id = (id.y >> VIS_INSTANCE_SHIFT) - 1u;
    let meshlet_id = (id.x >> VIS_MESHLET_SHIFT) & ((1u << VIS_MESHLET_BITS) - 1u);
    let tri = id.x & ((1u << VIS_TRI_BITS) - 1u);

    let inst = vis_instance(instance_id);
    let m = v_meshlets[meshlet_id];

    // The three corners, pulled exactly as the raster pulled them.
    var wp: array<vec3<f32>, 3>;
    var nrm3: array<vec3<f32>, 3>;
    var uv3: array<vec2<f32>, 3>;
    var tan3: array<vec4<f32>, 3>;
    var clip: array<vec4<f32>, 3>;
    let nmat = mat3x3<f32>(inst.n0, inst.n1, inst.n2);
    let tmat = mat3x3<f32>(inst.model[0].xyz, inst.model[1].xyz, inst.model[2].xyz);
    for (var k = 0u; k < 3u; k = k + 1u) {
        let local = read_tri_byte(m.triangle_offset + tri * 3u + k);
        let global_v = v_meshlet_verts[m.vertex_offset + local];
        let base = global_v * VGEOM_VSTRIDE;
        let p = vec3<f32>(v_positions[base], v_positions[base + 1u], v_positions[base + 2u]);
        let nn = vec3<f32>(v_positions[base + 3u], v_positions[base + 4u], v_positions[base + 5u]);
        let w = inst.model * vec4<f32>(p, 1.0);
        wp[k] = w.xyz;
        nrm3[k] = nmat * nn;
        uv3[k] = vec2<f32>(v_positions[base + 6u], v_positions[base + 7u]);
        // P28.2: the vertex tangent, carried by the model matrix's LINEAR part —
        // a tangent is contravariant and rides `rotation · scale`, not the normal
        // matrix's `rotation · scale⁻¹` (P28.2 audit; the two part company under
        // non-uniform instance scale). The resolve interpolates it by the solved
        // barycentrics below, so its frame is the raster's frame with
        // better-conditioned weights — the same relationship the whole pass has
        // to the forward path, and `vgeom_mesh` does the identical thing.
        let t4 = vgeom_unpack_tangent(bitcast<u32>(v_positions[base + 8u]));
        tan3[k] = vec4<f32>(tmat * t4.xyz, t4.w);
        clip[k] = view.view_proj * w;
    }

    let wh = view.grid_axis_viewport.zw;
    let ndc = vec2<f32>(2.0 * in.pos.x / max(wh.x, 1.0) - 1.0,
                        1.0 - 2.0 * in.pos.y / max(wh.y, 1.0));
    let b = vis_bary(clip[0], clip[1], clip[2], ndc, wh);
    let l = b.lambda;

    // The rasterized depth, recovered rather than stored. `vis_bary` solves
    // `M * mu = (ndc.x, ndc.y, 1)` EXACTLY, so the reconstructed clip point is
    // `(ndc.x, ndc.y, dot(mu, z), 1)` and its w is one — which makes the ndc
    // depth `dot(lambda, z) / dot(lambda, w)`, spelled out because the two
    // dot products are what the identity claims are equal to `dot(mu, z)` and 1.
    // Written this way it survives a `mu` that is scaled, which a future
    // homogeneous formulation might be.
    let zc = vec3<f32>(clip[0].z, clip[1].z, clip[2].z);
    let wc = vec3<f32>(clip[0].w, clip[1].w, clip[2].w);
    out.depth = dot(l, zc) / max(dot(l, wc), 1e-20);

    let world_pos = wp[0] * l.x + wp[1] * l.y + wp[2] * l.z;
    // The handedness is a per-vertex CONSTANT of an authored parametrization, so
    // it is taken from the nearest corner rather than blended: averaging +1 and
    // -1 across a seam would produce 0, which this frame reads as "no tangent".
    let tw = select(select(tan3[2].w, tan3[1].w, l.y >= l.z),
                    tan3[0].w, l.x >= max(l.y, l.z));
    let tangent_i = vec4<f32>(
        tan3[0].xyz * l.x + tan3[1].xyz * l.y + tan3[2].xyz * l.z, tw);
    let normal_i = nrm3[0] * l.x + nrm3[1] * l.y + nrm3[2] * l.z;
    let uv = uv3[0] * l.x + uv3[1] * l.y + uv3[2] * l.z;

    // The analytic replacements for `dpdx`/`dpdy`. Same three weights, the
    // gradient row instead of the value row.
    let vt_ddx = uv3[0] * b.d_dx.x + uv3[1] * b.d_dx.y + uv3[2] * b.d_dx.z;
    let vt_ddy = uv3[0] * b.d_dy.x + uv3[1] * b.d_dy.y + uv3[2] * b.d_dy.z;
    let vt_dpx = wp[0] * b.d_dx.x + wp[1] * b.d_dx.y + wp[2] * b.d_dx.z;
    let vt_dpy = wp[0] * b.d_dy.x + wp[1] * b.d_dy.y + wp[2] * b.d_dy.z;

    let vt_slots = inst.vt;

    // The view-mode ramps, in the forward path's own order and above the unlit
    // short-circuit for the reason `vgeom_mesh.wgsl` records: `VtResidency` and
    // `VsmPages` set the unlit flag too, so a heat branch below the unlit return
    // would never execute.
    if (view.flags.z > 0.5) {
        out.color = vec4<f32>(vt_heat(vt_slots, uv, vt_ddx, vt_ddy), 1.0);
        return out;
    }
    if (view.flags.w > 0.5) {
        out.color = vec4<f32>(vsm_heat(world_pos, normalize(normal_i)), 1.0);
        return out;
    }
    if (view.flags.x > 0.5) {
        out.color = vec4<f32>(inst.color.rgb + inst.emissive, inst.color.a);
        return out;
    }

    var n = normalize(normal_i);
    let v = normalize(view.eye.xyz - world_pos);
    var albedo = inst.color.rgb;
    var metallic = clamp(inst.metallic, 0.0, 1.0);
    var rough = clamp(inst.roughness, 0.04, 1.0);
    var vt_ao = 1.0;
    if (vt_slots.x != 0u || vt_slots.y != 0u || vt_slots.z != 0u) {
        let s = vt_surface(vt_slots, uv, vt_ddx, vt_ddy,
                           albedo, inst.color.a, metallic, rough);
        albedo = s.albedo;
        metallic = clamp(s.metallic, 0.0, 1.0);
        rough = clamp(s.roughness, 0.04, 1.0);
        vt_ao = s.occlusion;
        if (s.has_normal) {
            n = vt_apply_normal_t(
                n, tangent_i, vt_dpx, vt_dpy, vt_ddx, vt_ddy, s.normal_ts
            );
        }
    }
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);

    var lo = vec3<f32>(0.0);
    let count = lights.count.x;
    if (count == 0u) {
        var d = shade_light(n, v, normalize(view.sun_dir.xyz), vec3<f32>(3.0),
                            albedo, metallic, rough, f0);
        if (sun_shadowing_enabled()) {
            d = d * shadow_factor(world_pos, n);
        }
        lo += d;
    } else {
        var shadowed = false;
        for (var i = 0u; i < count && i < MAX_LIGHTS; i = i + 1u) {
            let light = lights.items[i];
            let radiance_base = light.color.rgb * light.color.a;
            if (light.pos_dir.w < 0.5) {
                var d = shade_light(n, v, normalize(light.pos_dir.xyz), radiance_base,
                                    albedo, metallic, rough, f0);
                if (sun_shadowing_enabled() && !shadowed) {
                    d = d * shadow_factor(world_pos, n);
                    shadowed = true;
                }
                lo += d;
            } else {
                let to_light = light.pos_dir.xyz - world_pos;
                let dist = length(to_light);
                let ll = to_light / max(dist, 1e-4);
                let att = point_attenuation(dist, light.params.x);
                var cone = 1.0;
                if (light.pos_dir.w > 1.5) {
                    let cos_dir = dot(ll, -light.spot_dir.xyz);
                    cone = smoothstep(light.params.z, light.params.y, cos_dir);
                }
                let vsm_f = vsm_light_shadow(world_pos, n, light.params.w);
                lo += shade_light(n, v, ll, radiance_base * att * cone * vsm_f,
                                 albedo, metallic, rough, f0);
            }
        }
    }

    let up = clamp(n.y * 0.5 + 0.5, 0.0, 1.0);
    // Wave FIX3: one door. The hemispheric constant is the fallback for a level
    // that asked for no computed ambient; otherwise the sky's own irradiance
    // plus the probe field's signed difference from it. See
    // `ambient_irradiance` / `sky_irradiance` in `env_lighting.wgsl`.
    let amb = ambient_irradiance(
        world_pos,
        n,
        mix(vec3<f32>(0.03, 0.03, 0.035), vec3<f32>(0.10, 0.13, 0.18), up),
    );
    let ao = textureSampleLevel(ao_tex, ao_smp, in.pos.xy / view.grid_axis_viewport.zw, 0.0).r
        * vt_ao;
    lo += amb * albedo * (1.0 - metallic) * ao;
    lo += gi_ambient_specular(world_pos, n, v, rough, f0, amb) * ao;

    lo += inst.emissive;

    let dist = length(world_pos - view.eye.xyz);
    let haze = 1.0 - exp(-dist * 0.004);
    var col = mix(lo, vec3<f32>(0.055, 0.081, 0.120), haze * 0.4);
    if (atmos.params.x > 0.5) {
        col = atmos_apply(lo, world_pos);
    }

    out.color = vec4<f32>(col, inst.color.a);
    return out;
}
