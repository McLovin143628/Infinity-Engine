// GPU skinning pass (P11.1): the skinned variant of mesh.wgsl. The vertex stage
// deforms a bind-space vertex by the per-instance joint palette (@group(3),
// weighted across its four joints) BEFORE the model matrix; the fragment stage is
// the same metallic-roughness PBR (duplicated here so mesh.wgsl — and every
// existing golden's pixels — stays byte-stable). `common_view.wgsl` is prepended.

struct VsIn {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) joints: vec4<u32>,
    @location(3) weights: vec4<f32>,
    // P26.5: the character's AUTHORED uv, at the one location left. The instance
    // block owns 4..=14 here, so 15 is the last address
    // `max_vertex_attributes: 16` allows — this pipeline is now full, which is
    // the wall `docs/memos/p26-5-vertex-streams.md` measures against a tangent.
    @location(15) uv: vec2<f32>,
    // Instance data (buffer 1), @location(4..=14).
    @location(4) model_0: vec4<f32>,
    @location(5) model_1: vec4<f32>,
    @location(6) model_2: vec4<f32>,
    @location(7) model_3: vec4<f32>,
    @location(8) nrm_0: vec4<f32>,
    @location(9) nrm_1: vec4<f32>,
    @location(10) nrm_2: vec4<f32>,
    @location(11) color: vec4<f32>,
    @location(12) misc: vec4<u32>,
    @location(13) pbr: vec4<f32>,
    @location(14) emissive: vec4<f32>,
};

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) world_pos: vec3<f32>,
    @location(3) @interpolate(flat) id: u32,
    @location(4) @interpolate(flat) pbr: vec4<f32>,
    @location(5) @interpolate(flat) emissive: vec3<f32>,
    // P26.5, exactly as mesh.wgsl: the character's own uv and the instance's
    // virtual-texture set. The bind-space frame the box projection needed is
    // gone with the projection.
    @location(6) uv: vec2<f32>,
    @location(8) @interpolate(flat) vt: vec3<u32>,
};

// **The joint-palette ATLAS** (wave NPC1b): every instance's palette, packed at
// matrix granularity into one storage buffer. skin[i] = global[i] · inverse_bind[i].
//
// An instance finds its own block through two channels of the instance stream
// that have been reserved-zero since P7.1 — `emissive.w` is the block's offset in
// matrices and `pbr.z` is its live joint count — because this pipeline is at the
// `max_vertex_attributes: 16` wall exactly and there is no seventeenth address to
// put them at (docs/memos/p26-5-vertex-streams.md). Both are small integers, so
// the f32 round-trip is exact.
@group(3) @binding(0) var<storage, read> palette: array<mat4x4<f32>>;

// **The clamp that replaced `pad_palette`.** A vertex whose joint index runs past
// its own skeleton (a mesh bound to the wrong rig, an index the CPU clamp did not
// see) used to read a power-of-two tail the CPU had filled with this skeleton's
// last live matrix. In an ATLAS that vertex would read the next character's
// palette instead — the same round-2 finding R2-6, made worse — so the rule moved
// here, where it is exact, costs no padding at all, and cannot be defeated by a
// buffer whose length says nothing about this instance's rig.
fn joint(base: u32, count: u32, index: u32) -> mat4x4<f32> {
    return palette[base + min(index, max(count, 1u) - 1u)];
}

@vertex
fn vs(in: VsIn) -> VsOut {
    let base = u32(in.emissive.w);
    let count = u32(in.pbr.z);
    // Linear blend skinning: weighted sum of the four influencing joint matrices.
    let skin = in.weights.x * joint(base, count, in.joints.x)
             + in.weights.y * joint(base, count, in.joints.y)
             + in.weights.z * joint(base, count, in.joints.z)
             + in.weights.w * joint(base, count, in.joints.w);
    let skinned_pos = (skin * vec4<f32>(in.pos, 1.0)).xyz;
    let skin3 = mat3x3<f32>(skin[0].xyz, skin[1].xyz, skin[2].xyz);
    let skinned_normal = skin3 * in.normal;

    let model = mat4x4<f32>(in.model_0, in.model_1, in.model_2, in.model_3);
    let nrm = mat3x3<f32>(in.nrm_0.xyz, in.nrm_1.xyz, in.nrm_2.xyz);
    var out: VsOut;
    let wp = model * vec4<f32>(skinned_pos, 1.0);
    out.pos = view.view_proj * wp;
    out.world_pos = wp.xyz;
    out.normal = nrm * skinned_normal;
    out.color = in.color;
    out.id = in.misc.x;
    // **The day someone gives this pipeline an alpha test is wave CHAR1a.2.**
    // `pbr.z` is this path's joint count (the vertex stage's business and no
    // further), and `pbr.w` carries the blend code and the cutoff together —
    // `blend * 4 + cutoff` — because this pipeline is at the vertex-attribute
    // wall and that channel was the last one still zero. Unpacked HERE, into the
    // rigid path's own `pbr.zw` order (z = cutoff, w = code), so the fragment
    // stage below tests a masked surface with the identical two lines mesh.wgsl
    // uses. `emissive.w` (the atlas offset) is dropped by `.rgb` already.
    let blend_code = floor(in.pbr.w * 0.25);
    out.pbr = vec4<f32>(in.pbr.xy, in.pbr.w - blend_code * 4.0, blend_code);
    out.emissive = in.emissive.rgb;
    // The uv is a property of the SURFACE, not of the pose, so it rides through
    // unchanged — which is the whole difference between an authored
    // parametrization and a projection that had to be re-derived from the
    // skinned position every frame to stop it sliding.
    out.uv = in.uv;
    out.vt = in.misc.yzw;
    return out;
}

// The uv this path samples with is the character's own (P26.5) — the same rule
// as the rigid path's, so a skinned surface and a rigid one cannot texture
// differently, and neither of them box-projects any more.

// ── Lights (must match LightsUniform / MAX_LIGHTS in passes/mesh.rs) ──
const MAX_LIGHTS: u32 = 16u;

struct GpuLight {
    color: vec4<f32>,
    pos_dir: vec4<f32>, // w = kind (0 dir, 1 point, 2 spot)
    params: vec4<f32>,  // x = range, y = spot inner_cos, z = spot outer_cos
    spot_dir: vec4<f32>, // xyz = normalized spot emission direction (spot only)
};
struct Lights {
    count: vec4<u32>,
    items: array<GpuLight, MAX_LIGHTS>,
};
@group(1) @binding(0) var<uniform> lights: Lights;

// AO + cascaded shadows + dynamic GI ride the shared env bind group at @group(2)
// (declared in env_lighting.wgsl, prepended by `lit_scene_shader`): `ao_tex`/`ao_smp`
// (SSAO, white when off), `shadow_factor()`, and `ambient_irradiance()`.

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

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    // Masked alpha-test, wave CHAR1a.2 — the same two lines as mesh.wgsl:147,
    // reading the code and the threshold the vertex stage unpacked out of
    // `pbr.w`. Opaque (code 0) and translucent (code 2) never take this branch,
    // so every pre-CHAR1a.2 skinned golden stays byte-identical (the branch is
    // present and always false for them). Runs before the unlit short-circuit so
    // hair cards and eyelashes show their holes in every view mode.
    if (in.pbr.w > 0.5 && in.pbr.w < 1.5 && in.color.a < in.pbr.z) {
        discard;
    }
    // P26.5 RESIDENCY HEAT-MAP (`ViewMode::VtResidency`): every virtual-textured
    // surface painted by how far behind the streamer is at that pixel.
    //
    // **Before the unlit short-circuit, not after.** `VtResidency` sets
    // `flags.x` too (the `Biomes` precedent: everything that is not the overlay
    // renders flat so the ramp is readable), so a heat branch placed below the
    // unlit return would never execute — measured, and it is the whole defect a
    // real-frame arm catches and a source read does not.
    //
    // `flags.z` is 0.0 in every other mode, so this is present-and-false and
    // every golden runs the identical arithmetic. The derivatives are taken
    // here, in uniform control flow: `flags.z` is a uniform, so the return is
    // uniform, and `in.uv` is an interpolated input either way.
    if (view.flags.z > 0.5) {
        return vec4<f32>(vt_heat(in.vt, in.uv, dpdx(in.uv), dpdy(in.uv)), 1.0);
    }
    // P27.5 VsmPages: the shadow-page residency ramp, beside the texture one and
    // for the same reason. `flags.w` is 0.0 in every other mode, so this is
    // present-and-false and every golden runs the identical arithmetic. Above
    // the unlit short-circuit, which `VsmPages` also sets (the `VtResidency`
    // precedent, and the defect a source read does not catch).
    if (view.flags.w > 0.5) {
        return vec4<f32>(vsm_heat(in.world_pos, normalize(in.normal)), 1.0);
    }
    // Unlit view mode (R-P2): albedo + emissive, no lighting (Unlit + Wireframe).
    if (view.flags.x > 0.5) {
        return vec4<f32>(in.color.rgb + in.emissive, in.color.a);
    }
    var n = normalize(in.normal);
    let v = normalize(view.eye.xyz - in.world_pos);
    var albedo = in.color.rgb;
    var metallic = clamp(in.pbr.x, 0.0, 1.0);
    var rough = clamp(in.pbr.y, 0.04, 1.0);
    // P26.3 VIRTUAL TEXTURES. Zero on every instance that names no texture, so
    // the branch is present-and-false for every scene that predates this batch
    // and the arithmetic below is byte-identical.
    // The CHARACTER'S OWN uv (P26.5).
    let uv = in.uv;
    // The screen derivatives, taken in UNIFORM control flow — a fragment shader
    // may only difference against its neighbours outside a divergent branch, and
    // the VT branch below is per instance. Cheap when nothing samples: a
    // derivative of an interpolated value is two subtractions.
    let vt_ddx = dpdx(uv);
    let vt_ddy = dpdy(uv);
    let vt_dpx = dpdx(in.world_pos);
    let vt_dpy = dpdy(in.world_pos);
    var vt_ao = 1.0;
    if (in.vt.x != 0u || in.vt.y != 0u || in.vt.z != 0u) {
        let s = vt_surface(in.vt, uv, vt_ddx, vt_ddy,
                           albedo, in.color.a, metallic, rough);
        albedo = s.albedo;
        metallic = clamp(s.metallic, 0.0, 1.0);
        rough = clamp(s.roughness, 0.04, 1.0);
        vt_ao = s.occlusion;
        if (s.has_normal) {
            n = vt_apply_normal(n, vt_dpx, vt_dpy, vt_ddx, vt_ddy, s.normal_ts);
        }
    }
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);

    var lo = vec3<f32>(0.0);
    let count = lights.count.x;
    if (count == 0u) {
        var d = shade_light(n, v, normalize(view.sun_dir.xyz), vec3<f32>(3.0),
                          albedo, metallic, rough, f0);
        if (sun_shadowing_enabled()) {
            d = d * shadow_factor(in.world_pos, n);
        }
        // P17.3: large-scale cloud shadowing of the sun. Guarded exactly like the
        // CSM block above, so a scene without clouds runs the identical
        // instruction stream and its goldens stay byte-identical.
        if (atmos.clouds.x > 0.5 && atmos.cloud_shadow.x > 0.0) {
            d = d * cloud_shadow_factor(in.world_pos);
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
                    d = d * shadow_factor(in.world_pos, n);
                    shadowed = true;
                }
                // P17.3: cloud shadows darken every directional light, not just
                // the first — a cloud layer is above all of them.
                if (atmos.clouds.x > 0.5 && atmos.cloud_shadow.x > 0.0) {
                    d = d * cloud_shadow_factor(in.world_pos);
                }
                lo += d;
            } else {
                // Point (w == 1) / spot (w == 2); `cone` stays 1.0 for a point
                // light, so the point path is byte-stable.
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
    // The material's own occlusion map multiplies the screen-space AO: both
    // modulate the AMBIENT term only, never the direct light above.
    let ao = textureSampleLevel(ao_tex, ao_smp, in.pos.xy / view.grid_axis_viewport.zw, 0.0).r
        * vt_ao;
    lo += amb * albedo * (1.0 - metallic) * ao;
    // P18.4 GI specular (see mesh.wgsl); the constant otherwise.
    lo += gi_ambient_specular(in.world_pos, n, v, rough, f0, amb) * ao;

    lo += in.emissive;

    // HDR-linear haze; the post tonemap pass runs afterward. P17.2: replaced by
    // physical aerial perspective + height fog when the scene has an atmosphere.
    let dist = length(in.world_pos - view.eye.xyz);
    let haze = 1.0 - exp(-dist * 0.004);
    var col = mix(lo, vec3<f32>(0.055, 0.081, 0.120), haze * 0.4);
    if (atmos.params.x > 0.5) {
        col = atmos_apply(lo, in.world_pos);
    }

    return vec4<f32>(col, in.color.a);
}
