// scatter_mesh.wgsl — the P18.5 scatter raster: one vertex-pulled indirect draw
// for the full-mesh band and one for the impostor band, sharing a fragment stage.
// Composed by `lit_scene_shader(.., 2)`, so `common_view.wgsl`, the env bind group
// (AO + cascaded shadows + GI + atmosphere + cloud shadows) and the atmosphere
// library are all prepended — scatter receives exactly what every other lit pass
// receives.
//
// ── Why vertex pulling, when a scatter is the textbook case for classic
//    instancing ────────────────────────────────────────────────────────────────
//
// The cull compute produces a *compacted index list*, not a compacted payload.
// Feeding that to classic instancing needs `first_instance` on the indirect args
// to address a sub-range of a shared vertex buffer, and `INDIRECT_FIRST_INSTANCE`
// is a non-portable wgpu feature — the same wall the meshlet path hit in P13.1b.
// Compacting the 48-byte payloads instead would make the compaction pass write 12×
// the bytes it does now. So the list stays indices and the vertex stage reads
// `visible[instance_index]`, which is the vgeom precedent, one array subscript, and
// portable everywhere.
//
// ── The dithered cross-fade, and why it has no holes ─────────────────────────
//
// In the `fade`-metre band before `mesh_end` an instance is in BOTH lists. Its
// mesh keeps the pixels where a screen-position hash falls below the mesh weight
// `m = (mesh_end - d)/fade`, and its impostor keeps exactly the complement
// (`h >= m`). One hash, two complementary tests ⇒ every pixel in the overlap is
// covered exactly once, so the transition never thins the silhouette and never
// double-shades it. Past `mesh_end` the mesh is not in the list at all and the
// impostor keeps everything; in the last `fade` metres before the cull distance
// the impostor thins out against `e = (cull - d)/fade` and vanishes.
//
// The hash is a pure function of the pixel — **no temporal jitter, no frame
// index, no instance salt** — because a golden renders one frame from cold and a
// determinism gate renders two: anything remembered between frames would make a
// fade band a function of history.

// 64 B — mirrors `scene::ScatterInstanceRaw`. The first 48 bytes are the P18.5
// record unchanged; `scale_yz` (IB-2b) makes an instance an oriented box.
struct ScatterInst {
    offset: vec3<f32>,
    scale: f32,
    rotation: vec4<f32>,
    color: vec4<f32>,
    scale_yz: vec2<f32>,
    _pad: vec2<f32>,
};

struct ScatterParams {
    // xyz = the batch anchor in render-local space, w = the impostor/cull radius
    // at unit scale (the primitive's bounding radius).
    anchor: vec4<f32>,
    // x = metallic, y = roughness, **z = the primitive kind** (island wave I4b,
    // for `impostor_radius`), w unused.
    material: vec4<f32>,
    // rgb = emissive, w unused.
    emissive: vec4<f32>,
    // x = full-mesh band end (m), y = cull distance (m), z = fade width (m),
    // w = impostors enabled (1/0).
    bands: vec4<f32>,
    // x = first index of this batch's primitive in the shared index buffer,
    // y = its base vertex, z = pick id, w = the compacted list capacity.
    geom: vec4<u32>,
};

@group(3) @binding(0) var<uniform> sp: ScatterParams;
@group(3) @binding(1) var<storage, read> s_instances: array<ScatterInst>;
@group(3) @binding(2) var<storage, read> s_visible: array<u32>;
// Shared primitive geometry: 6 f32 per vertex (position then normal), and the
// packed u16 index list widened to u32.
@group(3) @binding(3) var<storage, read> s_verts: array<f32>;
@group(3) @binding(4) var<storage, read> s_indices: array<u32>;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) world_pos: vec3<f32>,
    // x = mesh weight `m`, y = impostor far weight `e`, z = impostor flag,
    // w = the billboard's radial coordinate² (impostor only).
    @location(3) @interpolate(flat) fade: vec4<f32>,
    @location(4) disc: vec2<f32>,
};

fn qrot(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

fn read_vertex(idx: u32) -> array<vec3<f32>, 2> {
    let b = idx * 6u;
    return array<vec3<f32>, 2>(
        vec3<f32>(s_verts[b], s_verts[b + 1u], s_verts[b + 2u]),
        vec3<f32>(s_verts[b + 3u], s_verts[b + 4u], s_verts[b + 5u]),
    );
}

// The two band weights for an instance at distance `d`. `m` is the mesh's, `e`
// the impostor's far ramp; both clamp to [0,1] so the common case (well inside a
// band) is an exact 1.0 and the discard test below is never taken.
fn band_weights(d: f32) -> vec2<f32> {
    let fade = max(sp.bands.z, 1.0e-3);
    return vec2<f32>(
        clamp((sp.bands.x - d) / fade, 0.0, 1.0),
        clamp((sp.bands.y - d) / fade, 0.0, 1.0),
    );
}

// ── P22.1 FOLIAGE BEND ───────────────────────────────────────────────────────
//
// Two things make a scatter instance stop reading as a plastic model glued to
// the ground, and both are horizontal displacements that grow with height:
//
//  (a) **trampling** — grass in a footprint lies down. `deform_gradient` points
//      INTO the dent, so `-gradient` is "out of the print", which is the way a
//      crushed blade actually splays. At the exact centre of a print the
//      gradient vanishes and nothing leans, which is why the vertical squash
//      below exists as well: the middle of a footprint is flattened, not swept.
//  (b) **wind** — a travelling sine along the wind direction, phased by
//      `dfm.params.w`, which is `ResolvedSky::cloud_time_s`: the LEVEL's clock.
//      No frame index, nothing remembered between frames — the same rule the
//      dither hash above is written to, and for the same reason (a golden
//      renders one frame from cold; a determinism gate renders two).
//
// The two ride SEPARATE switches — `deform_enabled()` and
// `deform_wind_enabled()` — because they are separate claims. Trampling needs a
// field; sway does not, and gating it on "is there a live deform cell anywhere"
// made an ambient effect flick on and off with a footprint expiring half a
// kilometre away. Sway is `ScatterSettings::foliage_wind`, OFF by default, which
// is also what keeps all 49 pre-P22.1 goldens byte-identical.
//
// ── HONEST REMAINDER: THE SHADOW DOES NOT BEND ───────────────────────────────
//
// A scatter instance's shadow is not cast through this shader. Casters are
// packed on the CPU by `passes::shadow` (`scatter::pack_fallback` →
// `shadow_depth.wgsl`), which draws them as rigid instanced boxes, so a blade
// that has been laid flat by a boot still casts the shadow of an upright one —
// and under the grazing sun the deformation golden uses, that is visible.
//
// Not fixed here because the shadow path does NOT share this vertex stage:
// making it agree means teaching the caster pack the same bend, i.e. a second
// implementation of `scatter_bend` on the CPU (in the shadow pass's own
// coordinate space, with no deformation texture bound), which is exactly the
// kind of duplicated derivation the mirror rule exists to prevent. The right fix
// is to route scatter casters through a shared vertex path; that is a shadow-pass
// change, not a P22.1 one, and it is on the ledger.
//
// Returns `(lean_xz, squash)`: a horizontal displacement per metre of height,
// and the vertical scale a trampled instance keeps.
fn scatter_bend(base_xz: vec2<f32>) -> vec3<f32> {
    var lean = vec2<f32>(0.0, 0.0);
    var squash = 1.0;
    if (deform_enabled()) {
        // `deform_depth` already carries the window's rim fade, so the trample
        // term goes continuously to zero at the window edge instead of snapping
        // a row of grass upright 64 m from the camera.
        let depth = deform_depth(base_xz);
        let trample = clamp(depth / max(dfm.params.y, 1e-4), 0.0, 1.0);
        let g = deform_gradient(base_xz);
        let gl = length(g);
        var out_dir = vec2<f32>(0.0, 0.0);
        if (gl > 1e-4) {
            out_dir = -g / gl;
        }
        lean = out_dir * (dfm.params.z * trample);
        squash = 1.0 - 0.6 * trample;
    }
    if (deform_wind_enabled()) {
        // The phase offset is precomputed and reduced on the CPU in f64, so this
        // is origin-invariant and cannot freeze at a day-long clock — see the
        // `Deform.wind` comment in deform.wgsl.
        let phase = dot(base_xz, dfm.wind.xy) * dfm.wind.z + dfm.wind.w;
        lean = lean + dfm.wind.xy * (0.12 * sin(phase));
    }
    return vec3<f32>(lean.x, lean.y, squash);
}

@vertex
fn vs_mesh(@builtin(vertex_index) vidx: u32, @builtin(instance_index) iidx: u32) -> VsOut {
    let inst = s_instances[s_visible[iidx]];
    let vert = read_vertex(s_indices[sp.geom.x + vidx] + sp.geom.y);
    let center = sp.anchor.xyz + inst.offset;
    // P22.1: shear at the transform site. `local.y` is the vertex's height above
    // the instance's own origin, so a displacement proportional to `h · h/r`
    // leaves the base planted (h = 0 ⇒ no motion) and grows quadratically toward
    // the tip — a bend, not a translation. `bend` is `vec3(lean_x, lean_z,
    // squash)` and is exactly `vec3(0, 0, 1)` on every scene with no deformation
    // field, so this is `local` unchanged and every pre-P22.1 scatter frame is
    // byte-identical.
    // IB-2b: the scale is per-axis in the instance's OWN frame, so it multiplies
    // the vertex BEFORE the rotation. `scale_yz == vec2(scale, scale)` for every
    // uniform instance, which is every caller that is not a structure shell, and
    // the arithmetic is then bit-identical to the P18.5 `vert[0] * inst.scale`.
    let iscale = vec3<f32>(inst.scale, inst.scale_yz.x, inst.scale_yz.y);
    var local = qrot(inst.rotation, vert[0] * iscale);
    let bend = scatter_bend(center.xz);
    let radius = max(sp.anchor.w * max(iscale.x, max(iscale.y, iscale.z)), 1e-4);
    let h = max(local.y, 0.0);
    let shear = h * clamp(h / radius, 0.0, 1.0);
    local = vec3<f32>(
        local.x + bend.x * shear,
        local.y * bend.z,
        local.z + bend.y * shear,
    );
    let world = center + local;
    // Uniform scale ⇒ the inverse-transpose of R·S is R·S⁻¹, which normalizes back
    // to R. So a UNIFORM scatter normal needs no normal matrix at all, and the
    // branch below is not taken for any instance that predates IB-2b — which is
    // what keeps every committed golden byte-identical, because `normalize` is
    // not a no-op in floating point even where it is one in algebra.
    //
    // A non-uniform instance does need the correction: the inverse-transpose of
    // R·S is R·S⁻¹, i.e. divide by the scale, rotate, renormalize. Skipping it
    // would light a squat shell as though it were a cube.
    var nrm = vert[1];
    if (iscale.x != iscale.y || iscale.x != iscale.z) {
        nrm = normalize(nrm / max(iscale, vec3<f32>(1.0e-6)));
    }
    let normal = qrot(inst.rotation, nrm);

    var out: VsOut;
    out.pos = view.view_proj * vec4<f32>(world, 1.0);
    out.normal = normal;
    out.color = inst.color;
    out.world_pos = world;
    let w = band_weights(distance(center, view.eye.xyz));
    out.fade = vec4<f32>(w.x, w.y, 0.0, 0.0);
    out.disc = vec2<f32>(0.0, 0.0);
    return out;
}

// **The impostor card's radius: the instance's OWN bounding sphere** (island
// wave I4b).
//
// This used to be `unit_radius x max(sx, sy, sz)` — the primitive's uniform
// bounding radius scaled by the largest axis, which is exact for a uniform
// instance and wildly generous for anything else. Wave I4 measured what that
// costs on a city: a building's scatter impostor covers **19.2x** its mesh's
// screen area (55 868 px against 2 903 at 192 m), because a 20 x 30 x 7.4 m box
// scaled from a unit cube got a card of radius `0.866 x 30 = 25.98 m` where its
// bounding sphere is `0.5 x |(20, 30, 7.4)| = 18.38 m`. The card is a screen
// facing quad with a disc discard, so the drawn area falls with `r^2`.
//
// Every branch answers the SMALLEST sphere that contains the scaled primitive,
// which is what a billboard has to cover and no more:
//
// * a **cube**'s bounding sphere is half the box's own diagonal;
// * a **sphere** scales into an ellipsoid, whose bounding sphere is its largest
//   semi-axis — the old formula, which is why this is not a blanket `length`;
// * a **plane** is a quad in XZ, so Y contributes nothing;
// * a **cylinder** and a **cone** are a disc of radius 0.5 swept along Y, so the
//   rim is the farthest point and the radius is the hypotenuse of (disc radius,
//   half height).
//
// The kind rides in `material.z`, a slot the struct already carried unused.
//
// **An AUTHORED mesh has no kind, and says so in `material.w`** (wave TER2b).
// The five branches below all answer "the smallest sphere containing the scaled
// PRIMITIVE", and a scattered `.inf_mesh` is not one: a 0.307 m grass tuft drawn
// at instance scale 1.0 would take the unit cube's `0.866` and get a card almost
// three times its own height. So a batch carrying real geometry passes its own
// unit bounding radius in `material.w` -- zero means "no authored mesh, use the
// kind" -- and the card is that radius times the instance's largest axis, which
// is exactly the rule the cull compute already uses for the same batch
// (`anchor.w` on line 195). One radius, two consumers, no third opinion.
fn impostor_radius(kind: u32, s: vec3<f32>) -> f32 {
    if (sp.material.w > 0.0) {
        return sp.material.w * max(s.x, max(s.y, s.z));
    }
    if (kind == 1u) {
        return 0.5 * max(s.x, max(s.y, s.z));
    }
    if (kind == 0u) {
        return 0.5 * length(s);
    }
    if (kind == 2u) {
        return 0.5 * length(vec2<f32>(s.x, s.z));
    }
    let rim = 0.5 * max(s.x, s.z);
    let half_h = 0.5 * s.y;
    return sqrt(rim * rim + half_h * half_h);
}

@vertex
fn vs_impostor(@builtin(vertex_index) vidx: u32, @builtin(instance_index) iidx: u32) -> VsOut {
    let inst = s_instances[s_visible[sp.geom.w + iidx]];
    // Two triangles, corners (-1,-1) (1,-1) (1,1) / (-1,-1) (1,1) (-1,1).
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
    );
    let c = corners[vidx];
    let base = sp.anchor.xyz + inst.offset;
    let r = impostor_radius(
        u32(sp.material.z),
        abs(vec3<f32>(inst.scale, inst.scale_yz.x, inst.scale_yz.y)),
    );
    // P22.1: an impostor leans as a WHOLE CARD, evaluated once at its centre —
    // never per vertex. A card is a billboard: shearing its corners individually
    // would rotate the quad out of its screen-facing plane and it would stop
    // being a billboard, so what a distant bent blade needs is for its silhouette
    // to sit where the bent mesh's silhouette was. The lean is taken at the
    // card's own mid height (`r`), which is where the mesh path's quadratic
    // shear has moved to by then.
    let bend = scatter_bend(base.xz);
    let center = base + vec3<f32>(bend.x * r, -r * (1.0 - bend.z), bend.y * r);
    let world = center + view.cam_right.xyz * (c.x * r) + view.cam_up.xyz * (c.y * r);

    // A spherical normal over the disc, so the card shades as a blob of the right
    // size rather than as a flat sticker facing the camera. The z term points at
    // the eye, which is what makes the terminator run across the impostor the way
    // it ran across the mesh it replaced.
    let toward_eye = normalize(view.eye.xyz - center);
    let n = normalize(view.cam_right.xyz * c.x + view.cam_up.xyz * c.y
                      + toward_eye * sqrt(max(1.0 - min(dot(c, c), 1.0), 0.0)));

    var out: VsOut;
    out.pos = view.view_proj * vec4<f32>(world, 1.0);
    out.normal = n;
    out.color = inst.color;
    out.world_pos = world;
    // The band weight is taken at the UNBENT base, exactly as `vs_mesh` takes it
    // at its unbent centre and the cull compute takes it at the same offset. The
    // dithered cross-fade is only hole-free because the mesh's `m` and the
    // impostor's complement are the same number; letting a lean move one of them
    // by a few centimetres would have opened pixels on the boundary.
    let w = band_weights(distance(base, view.eye.xyz));
    out.fade = vec4<f32>(w.x, w.y, 1.0, 0.0);
    out.disc = c;
    return out;
}

// ── Lights (must match LightsUniform / MAX_LIGHTS in passes/mesh.rs) ──
const MAX_LIGHTS: u32 = 16u;

struct GpuLight {
    color: vec4<f32>,
    pos_dir: vec4<f32>, // w = kind (0 dir, 1 point, 2 spot)
    params: vec4<f32>,  // x = range, y = spot inner_cos, z = spot outer_cos
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

// A pure-integer avalanche of the pixel coordinate → [0,1). Wang hash: no trig,
// no float reinterpretation, identical on every backend, and pinned bit-for-bit
// by `scatter::tests::dither_hash_matches_the_rust_side`.
//
// **The result is folded to 24 bits before the divide, and that is a fix.** The
// first cut returned `f32(h) * (1.0 / 4294967296.0)` over the full 32-bit word:
// `h = 0xFFFFFFFF` gives 0.99999999977, which has no f32 neighbour below 1.0 and
// rounds to **exactly 1.0**. Every test below is `h < weight` with `weight <= 1`,
// so those pixels were discarded by both the mesh and the impostor even at full
// weight — a permanent scattering of holes, one pixel in 2^32 per hash draw but
// deterministic and therefore *always in the same places*, and it falsified the
// "the common case is never taken" comment on the discard. `h >> 8` maxes at
// 16777215/16777216 = 0.99999994, which is exactly representable and strictly
// below 1, so a full-weight band now really does keep every pixel. 24 bits is
// ~16.7M dither levels against a fade band that is a few hundred pixels wide.
fn scatter_dither(px: vec2<f32>) -> f32 {
    var h = (u32(px.x) & 0xFFFFu) | ((u32(px.y) & 0xFFFFu) << 16u);
    h = (h ^ 61u) ^ (h >> 16u);
    h = h + (h << 3u);
    h = h ^ (h >> 4u);
    h = h * 0x27d4eb2du;
    h = h ^ (h >> 15u);
    return f32(h >> 8u) * (1.0 / 16777216.0);
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let is_impostor = in.fade.z > 0.5;
    if (is_impostor) {
        // Round silhouette: the one thing a square card gets visibly wrong at the
        // distances an impostor covers.
        if (dot(in.disc, in.disc) > 1.0) {
            discard;
        }
    }

    // The complementary dither. Both tests are skipped entirely outside a fade
    // band (`m == 1` for the mesh, `m == 0 && e == 1` for the impostor), so the
    // common case pays one compare.
    let h = scatter_dither(in.pos.xy);
    if (is_impostor) {
        if (h < in.fade.x || h >= in.fade.y) {
            discard;
        }
    } else if (h >= in.fade.x) {
        discard;
    }

    // P27.5 VsmPages: the shadow-page residency ramp. Above the unlit
    // short-circuit, which `VsmPages` also sets — a heat branch below it would
    // never execute (the `VtResidency` precedent, and the defect a source read
    // does not catch). `flags.w` is 0.0 in every other mode, so this is
    // present-and-false and every golden runs the identical arithmetic.
    if (view.flags.w > 0.5) {
        return vec4<f32>(vsm_heat(in.world_pos, normalize(in.normal)), 1.0);
    }
    if (view.flags.x > 0.5) {
        return vec4<f32>(in.color.rgb + sp.emissive.rgb, in.color.a);
    }

    let n = normalize(in.normal);
    let v = normalize(view.eye.xyz - in.world_pos);
    let albedo = in.color.rgb;
    let metallic = clamp(sp.material.x, 0.0, 1.0);
    let rough = clamp(sp.material.y, 0.04, 1.0);
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);

    var lo = vec3<f32>(0.0);
    let count = lights.count.x;
    if (count == 0u) {
        var d = shade_light(n, v, normalize(view.sun_dir.xyz), vec3<f32>(3.0),
                            albedo, metallic, rough, f0);
        if (sun_shadowing_enabled()) {
            d = d * shadow_factor(in.world_pos, n);
        }
        lo += d;
    } else {
        // The first directional light receives the sun's shadow factor — the
        // virtual page atlas when one is bound, the cascade otherwise.
        var shadowed = false;
        for (var i = 0u; i < count && i < MAX_LIGHTS; i = i + 1u) {
            let light = lights.items[i];
            let radiance_base = light.color.rgb * light.color.a;
            if (light.pos_dir.w < 0.5) {
                // **P27.5: the sun's shadow reaches this surface at last.**
                // Before this batch neither this path nor the other of the two
                // had ever called `shadow_factor` — `git log -S` over both
                // files' whole history returns nothing — so virtualized geometry
                // and foliage took the analytic term and never the directional
                // one, from the cascade either. A phase named Virtual Shadow
                // MAPS whose flagship geometry path cannot RECEIVE a sun shadow
                // is a hole, and this closes it.
                //
                // Spelled exactly as `mesh.wgsl` spells it, first directional
                // only, and `sun_shadowing_enabled()` is false on every scene
                // with both shadow paths off — which is every committed golden
                // outside the four `vsm_*` ones, and those hold no meshlet and
                // no scattered geometry. So this is present-and-false everywhere
                // a golden looks.
                var d = shade_light(n, v, normalize(light.pos_dir.xyz), radiance_base,
                                    albedo, metallic, rough, f0);
                if (sun_shadowing_enabled() && !shadowed) {
                    d = d * shadow_factor(in.world_pos, n);
                    shadowed = true;
                }
                lo += d;
            } else {
                let to_light = light.pos_dir.xyz - in.world_pos;
                let dist = length(to_light);
                let l = to_light / max(dist, 1e-4);
                let att = point_attenuation(dist, light.params.x);
                var cone = 1.0;
                if (light.pos_dir.w > 1.5) {
                    let cos_dir = dot(l, -light.spot_dir.xyz);
                    cone = smoothstep(light.params.z, light.params.y, cos_dir);
                }
                // **P27.4: the engine's FIRST point/spot shadows.** A spot
                // resolves through its single quadtree, a point through the
                // cube-face quadtree its own direction selects. `params.w` is 0
                // on every light without a page tree — which is every light of
                // every scene with virtual shadows off — and `vsm_light_shadow`
                // returns exactly 1.0 there, so this is a present-and-inert
                // `* 1.0` on every committed golden.
                let vsm_f = vsm_light_shadow(in.world_pos, n, light.params.w);
                lo += shade_light(n, v, l, radiance_base * att * cone * vsm_f,
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
        in.world_pos,
        n,
        mix(vec3<f32>(0.03, 0.03, 0.035), vec3<f32>(0.10, 0.13, 0.18), up),
    );
    let ao = textureSampleLevel(ao_tex, ao_smp, in.pos.xy / view.grid_axis_viewport.zw, 0.0).r;
    lo += amb * albedo * (1.0 - metallic) * ao;
    lo += gi_ambient_specular(in.world_pos, n, v, rough, f0, amb) * ao;
    lo += sp.emissive.rgb;

    let dist = length(in.world_pos - view.eye.xyz);
    let haze = 1.0 - exp(-dist * 0.004);
    var col = mix(lo, vec3<f32>(0.055, 0.081, 0.120), haze * 0.4);
    if (atmos.params.x > 0.5) {
        col = atmos_apply(lo, in.world_pos);
    }

    return vec4<f32>(col, in.color.a);
}
