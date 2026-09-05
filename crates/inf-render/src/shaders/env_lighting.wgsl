// Shared environment lighting (P13.3b, grown in P18.4): AO + cascaded shadows +
// dynamic GI bindings and sampling functions, prepended (after common_view.wgsl) to
// the lit scene shaders by `passes::lit_scene_shader`. The `GROUP_ENV` token is
// substituted with each pipeline's env bind-group index (mesh/skinned/vgeom = 2,
// terrain = 3) so the same source serves every lit pass while the bindings land in
// the right group. Mirrors `EnvBinding` (passes/mod.rs) and the CPU math in
// `crate::csm` / `crate::gi`.
//
// AO stays at bindings 0/1 (declarations moved here from the individual shaders, so
// the existing inline `textureSampleLevel(ao_tex, ao_smp, …)` fragment lines are
// unchanged → byte-stable). Shadows (2,3,4), GI (5,6) and the P18.4 scene-depth
// binding (12, SSR) are appended and only touched when their `enabled` flag is set,
// so the off-path instruction stream is identical.

const SHADOW_RES: f32 = 2048.0; // must equal crate::csm::SHADOW_RESOLUTION

struct ShadowData {
    cascade_vp: array<mat4x4<f32>, 3>,
    splits: vec4<f32>,       // xyz = cascade far distances, w = cascade blend fraction
    texel_world: vec4<f32>,  // per-cascade world texel size (xyz)
    params: vec4<f32>,       // x=enabled, y=depth_bias, z=normal_bias, w=cascade_count
};

struct GiData {
    vol_min: vec4<f32>,    // xyz render-local min, w = voxel_size
    probe_min: vec4<f32>,  // xyz render-local probe grid min, w = extent
    dims: vec4<f32>,       // x = gi_dim, yzw = probe dims
    params: vec4<f32>,     // x=enabled, y=intensity, z=rays, w=macro dim
    sun_dir: vec4<f32>,
    sun_color: vec4<f32>,
    sky_zenith: vec4<f32>,
    sky_horizon: vec4<f32>,
    // P18.4: x = SH specular on, y = SSR on, z = SSR max distance (m),
    // w = SSR relative thickness.
    params2: vec4<f32>,
    // P18.4 amortization: x = probe start, y = probe count, z = probe total,
    // w = sky source (0 gradient, 1 atmosphere LUT).
    sched: vec4<f32>,
    // Wave VIS1a, the SSR block: x = march steps, y = roughness cutoff,
    // z = intensity, w = whether the previous frame's resolved colour is usable.
    ssr: vec4<f32>,
    // Wave VIS1a: the PREVIOUS frame's jittered view-projection, so a hit found in
    // this frame's depth is sampled from the colour buffer at the place it
    // occupied when that colour was written.
    prev_view_proj: mat4x4<f32>,
    // Wave FIX3: the L1 spherical-harmonic projection of THE SKY's own radiance
    // over the whole sphere (rgb per lane, w unused), computed on the CPU by
    // `crate::atmosphere::sky_irradiance_sh` from the same medium the sky pass
    // draws. It is the ambient every shaded surface receives, at any distance
    // from the camera and with no probe volume in the way; `sky_irradiance`
    // below is its consumer. Mirrored in `gi_probes.wgsl`.
    sky_sh0: vec4<f32>,
    sky_sh1: vec4<f32>,
    sky_sh2: vec4<f32>,
    sky_sh3: vec4<f32>,
};

@group(GROUP_ENV) @binding(0) var ao_tex: texture_2d<f32>;
@group(GROUP_ENV) @binding(1) var ao_smp: sampler;
@group(GROUP_ENV) @binding(2) var shadow_map: texture_depth_2d_array;
@group(GROUP_ENV) @binding(3) var shadow_smp: sampler_comparison;
@group(GROUP_ENV) @binding(4) var<uniform> shadow: ShadowData;
@group(GROUP_ENV) @binding(5) var<storage, read> gi_sh: array<vec4<f32>>;
@group(GROUP_ENV) @binding(6) var<uniform> gi: GiData;
// P18.4: the single-sample scene depth (the SSAO/TAA prepass target), read by the
// SSR raymarch with `textureLoad` — no sampler, so this costs one binding, not two.
// `RenderSettings::needs_depth_prepass` is true whenever SSR is on, so the texture
// this reads was written by THIS frame's prepass.
@group(GROUP_ENV) @binding(12) var gi_scene_depth: texture_depth_2d;
// Wave VIS1a: the PREVIOUS frame's resolved opaque colour — SSR's colour source.
// Sampled (not loaded), because the reprojected coordinate is fractional. The
// sampler is the atmosphere's (clamp-to-edge, linear), which is what a reprojected
// fetch wants; see `ENV_SCENE_COLOR` in `passes/mod.rs` for why the source is the
// previous frame rather than this one.
@group(GROUP_ENV) @binding(21) var ssr_scene_color: texture_2d<f32>;

// 3×3-PCF shadow factor of ONE cascade. Returns `-1.0` when the receiver falls
// outside this cascade's projection (the caller's "try the next one" signal), so
// the selection loop and the P18.4 blend can share the same sampling code instead
// of keeping two copies of the bias + PCF arithmetic in sync.
fn csm_cascade_pcf(world_pos: vec3<f32>, n: vec3<f32>, c: i32) -> f32 {
    let offset_pos = world_pos + n * shadow.texel_world[c] * shadow.params.z;
    let clip = shadow.cascade_vp[c] * vec4<f32>(offset_pos, 1.0);
    let ndc = clip.xyz / clip.w;
    let t = ndc.xy * vec2<f32>(0.5, -0.5) + 0.5; // clip → uv (flip y)
    if (!(all(t >= vec2<f32>(0.0)) && all(t <= vec2<f32>(1.0)) && ndc.z >= 0.0 && ndc.z <= 1.0)) {
        return -1.0;
    }
    let compare = ndc.z - shadow.params.y;
    let texel = 1.0 / SHADOW_RES;
    var sum = 0.0;
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let o = vec2<f32>(f32(dx), f32(dy)) * texel;
            sum = sum + textureSampleCompareLevel(shadow_map, shadow_smp, t + o, c, compare);
        }
    }
    return sum / 9.0;
}

// Cascaded shadow factor for the first directional light. Returns 1.0 (fully lit)
// when shadows are off or the receiver is beyond the last cascade. Selects the
// first cascade whose projected UV/depth are in range, then — P18.4 — **blends
// into the next cascade** across a band at the end of the current one, so the
// resolution change no longer shows up as a hard seam across the ground. The band
// is `splits.w` of the cascade's own range (0 ⇒ the pre-P18.4 hard switch, and the
// branch below is not taken at all).
fn shadow_factor(world_pos: vec3<f32>, n: vec3<f32>) -> f32 {
    // **P27.4: the sun's shadow through the page table, when there is one.**
    // VSM replaces the cascades rather than adding to them — P27.5 demotes this
    // path to the fallback it already is on Low, where `RenderTier::apply`
    // clears `VsmSettings::enabled` and the branch below is not taken.
    //
    // The sun's tree is named by the receiver uniform rather than by a light
    // record, because `terrain.wgsl` shades the sun from `view.sun_dir` and has
    // no light index at all — one door for the sun, and `GpuLight.params.w` for
    // every analytic light. `vsm_sun_bound()` is false on every frame without a
    // system (the empty table's first word is not the magic), so every
    // pre-P27.4 golden runs the identical instruction stream below.
    if (vsm_sun_bound()) {
        return vsm_shadow(world_pos, n, vsm.counts.x);
    }
    if (shadow.params.x < 0.5) {
        return 1.0;
    }
    let cascades = i32(shadow.params.w);
    var ci = -1;
    var factor = 1.0;
    for (var c = 0; c < cascades; c = c + 1) {
        let f = csm_cascade_pcf(world_pos, n, c);
        if (f >= 0.0) {
            ci = c;
            factor = f;
            break;
        }
    }
    if (ci < 0) {
        return 1.0;
    }
    let blend = shadow.splits.w;
    if (blend > 0.0 && ci + 1 < cascades) {
        let far = shadow.splits[ci];
        var near_edge = 0.0;
        if (ci > 0) {
            near_edge = shadow.splits[ci - 1];
        }
        let band = max(far - near_edge, 1e-4) * blend;
        let dist = length(world_pos - view.eye.xyz);
        let w = clamp((dist - (far - band)) / max(band, 1e-4), 0.0, 1.0);
        if (w > 0.0) {
            let nf = csm_cascade_pcf(world_pos, n, ci + 1);
            // A receiver at the very edge can fall outside the NEXT cascade too
            // (it is fit to a bounding sphere, not a slab); keep this cascade's
            // answer rather than fading toward "lit", which would flash.
            if (nf >= 0.0) {
                factor = mix(factor, nf, w);
            }
        }
    }
    return factor;
}

// **Is there a sun shadow term at all?** The one door every lit pass's "is it
// worth calling `shadow_factor`" guard goes through since P27.4, because the
// answer is now two flags rather than one and three shaders were each spelling
// the CSM half inline.
//
// Both halves are uniform (a uniform buffer's field and a storage buffer's first
// word), so this is a uniform branch and a scene with neither shadow kind on
// takes exactly the instruction stream it always did.
fn sun_shadowing_enabled() -> bool {
    return shadow.params.x > 0.5 || vsm_sun_bound();
}

fn sh_basis(d: vec3<f32>) -> vec4<f32> {
    return vec4<f32>(0.282095, 0.488603 * d.y, 0.488603 * d.z, 0.488603 * d.x);
}

/// The trilinearly probe-interpolated L1 SH coefficients at `world_pos`, as four
/// RGB triples in a 3×4 matrix (`[c0, c1, c2, c3]` in columns). Both the diffuse
/// irradiance and the P18.4 specular reconstruction start here, so the 8-tap fetch
/// is written once.
fn gi_fetch_sh(world_pos: vec3<f32>, n: vec3<f32>) -> mat4x3<f32> {
    let pmin = gi.probe_min.xyz;
    let extent = gi.probe_min.w;
    let pd = vec3<f32>(gi.dims.y, gi.dims.z, gi.dims.w);
    let coord = clamp((world_pos - pmin) / extent, vec3<f32>(0.0), vec3<f32>(1.0)) * (pd - 1.0);
    let base = floor(coord);
    let f = coord - base;

    var c0 = vec3<f32>(0.0);
    var c1 = vec3<f32>(0.0);
    var c2 = vec3<f32>(0.0);
    var c3 = vec3<f32>(0.0);
    // The same blend restricted to probes that are not buried inside geometry,
    // and the weight it accumulated — see below.
    var v0 = vec3<f32>(0.0);
    var v1 = vec3<f32>(0.0);
    var v2 = vec3<f32>(0.0);
    var v3 = vec3<f32>(0.0);
    var vw = 0.0;
    let maxc = vec3<i32>(pd) - vec3<i32>(1);
    for (var i = 0; i < 8; i = i + 1) {
        let off = vec3<f32>(f32(i & 1), f32((i >> 1) & 1), f32((i >> 2) & 1));
        let w = mix(1.0 - f, f, off);
        let weight = w.x * w.y * w.z;
        let gc = clamp(vec3<i32>(base + off), vec3<i32>(0), maxc);
        let flat = u32((gc.z * i32(pd.y) + gc.y) * i32(pd.x) + gc.x) * 4u;
        let s0 = gi_sh[flat + 0u];
        let s1 = gi_sh[flat + 1u].rgb;
        let s2 = gi_sh[flat + 2u].rgb;
        let s3 = gi_sh[flat + 3u].rgb;
        c0 = c0 + weight * s0.rgb;
        c1 = c1 + weight * s1;
        c2 = c2 + weight * s2;
        c3 = c3 + weight * s3;
        // `w` of the first coefficient is the probe's **validity**, written by
        // `gi_probes.wgsl`: zero when the probe stands inside solid geometry.
        //
        // …times the **backface weight**: a probe on the far side of the
        // surface's own plane is a probe in the next room. The ceiling of a
        // sealed hall is the case — its nearest probe vertically is the one
        // floating in the open sky above the roof, which the plain blend gave
        // 72 % of the weight, and that is light arriving through a slab.
        // `max(dot(., n), 0)` is the standard irradiance-volume weight; the
        // probe's own cell is never fully behind a surface inside it, so a
        // surface with no probe in front of it falls through to the plain blend
        // below rather than going black.
        // From `gc` and not from `base + off`: at the volume's far face
        // `base + off` names a probe one cell past the grid, which `gc` clamps
        // for the FETCH — so a position built from the unclamped index would be
        // a cell away from the probe whose coefficients are being weighted.
        // Mirrors `cs_probes`' own `probe_min.xyz + frac * extent`.
        let probe_pos = pmin + vec3<f32>(gc) * (extent / max(pd - 1.0, vec3<f32>(1.0)));
        let to_probe = probe_pos - world_pos;
        let facing = max(dot(normalize(to_probe + n * 1.0e-4), n), 0.0);
        let vwi = weight * s0.w * facing;
        v0 = v0 + vwi * s0.rgb;
        v1 = v1 + vwi * s1;
        v2 = v2 + vwi * s2;
        v3 = v3 + vwi * s3;
        vw = vw + vwi;
    }
    // **BURIED PROBES DO NOT VOTE** (wave FIX3 audit). A probe inside a wall
    // marched every ray into that wall, so what it reports is the inside of a
    // brick — and the trilinear blend gave it a corner's worth of weight over
    // every surface near it regardless. Measured: on `sky_ambient.rs`'s wall a
    // buried probe carried 87 % of the shaded face's fetch.
    //
    // When every corner is buried there is no opinion to be had and the plain
    // blend stands, because a zero here would read as black rather than as
    // "unknown" — and the eight-buried case is a surface inside solid geometry,
    // which is not visible anyway.
    if (vw > 1.0e-5) {
        let inv = 1.0 / vw;
        return mat4x3<f32>(v0 * inv, v1 * inv, v2 * inv, v3 * inv);
    }
    return mat4x3<f32>(c0, c1, c2, c3);
}

// The probe field's contribution to a Lambert surface's **exit radiance** at
// `world_pos` with normal `n`. The caller only invokes this when GI is enabled,
// so it needn't re-check the flag.
//
// # The Lambert 1/π, and where it went (wave EDIT1)
//
// Cosine-lobe convolution of an L1 set is Ramamoorthi's `A₀ = π`, `A₁ = 2π/3`,
// and it yields IRRADIANCE — on a uniform field of radiance `L`, exactly `π·L`.
// A Lambert surface then leaves `(ρ/π)·E = ρ·L`, which is why every lit pass's
// *direct* term spells the BRDF out as `kd * albedo / PI`.
//
// The ambient term does not, and never did:
//
// ```wgsl
//     lo += amb * albedo * (1.0 - metallic) * ao;    // mesh.wgsl and five more
// ```
//
// There is no `/π` on that line in any of the six lit shaders, and this
// function's own retired comment had written the obligation down — *"folds the
// Lambert 1/π into the caller"* — for a caller that never discharged it. So the
// ambient half of the engine ran **π times** the direct half per unit of
// incident radiance, `gi_intensity = 1.0` meant "π× a normalised gather", and
// the showcase island's daylight street clipped 13.5 % of its pixels to white
// with nothing in the frame darker than 91/255 (the FIX1 audit's table).
//
// The 1/π is folded into the two convolution constants rather than applied as a
// trailing divide, so this is the same instruction count it always was:
//
// ```text
//     A₀/π = 1                A₁/π = 2/3
// ```
//
// The hemispheric constant this replaces (`mix(0.03…, 0.10…)` at each call site)
// has always been in exit-radiance units — it is multiplied by albedo with no
// π either — so after this fold the two ambient sources finally mean the same
// thing, and switching GI on changes a surface's *shape* rather than its scale.
//
// Measured by `tests/gi_normalisation.rs` (the white furnace): before, a white
// Lambert cube in a uniform environment returned **3.22×** the environment's own
// radiance; after, 1.0. The five GI goldens hold their images by carrying `π` in
// their authored `intensity`, which is itself the proof that the change is
// exactly a scale and touches nothing else.
fn gi_indirect(world_pos: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let c = gi_fetch_sh(world_pos, n);
    let b = sh_basis(n);
    // Cosine-lobe convolution (Ramamoorthi) with the Lambert 1/π folded in:
    // A0/π = 1, A1/π = 2/3.
    let a0 = 1.0;
    let a1 = 0.66666667;
    let e = a0 * c[0] * b.x + a1 * (c[1] * b.y + c[2] * b.z + c[3] * b.w);
    // **No clamp here since wave FIX3.** This term is a DIFFERENCE and it is
    // meant to be able to go negative; the clamp lives at the call site, on the
    // sum with `sky_irradiance`, which is the quantity that has to be a
    // radiance. Clamping here would throw away exactly the occlusion.
    return e * gi.params.y;
}

// ── the sky's own irradiance (wave FIX3) ────────────────────────────────────
//
// **THE AMBIENT TERM THIS ENGINE DID NOT HAVE.**
//
// Before this wave a shaded surface was lit by `gi_irradiance` or by a
// hard-coded hemispheric constant, and by nothing else. The probe field is a
// **40 m box centred on the camera** (`GiSettings::extent`), so:
//
//   * a wall beyond the box reads a clamped boundary probe, which is a probe
//     about somewhere else;
//   * a wall inside it, close enough that the probes around it are *in* it,
//     reads probes whose every ray hit that same unlit wall;
//   * and trilinear interpolation drags a surface toward any of its eight
//     probes that happens to sit inside geometry, which reads as black.
//
// Measured, through the whole renderer, in `tests/sky_ambient.rs`: a white
// wall's shaded face read **0.0129** of its sunlit face under a clear noon sky
// with GI on, against the **0.1–0.2** a clear sky actually delivers. The
// FIX2 audit's photograph of that number is a hero at 1.8/255 in Play with a
// building's shaded wall at 8.5.
//
// So the sky becomes a term of its own: the cosine-weighted hemisphere of the
// *actual* sky, projected to L1 on the CPU once per frame from the same medium
// the sky pass draws (`crate::atmosphere::sky_irradiance_sh`), reaching every
// lit shader through this include. It has no volume, no convergence transient
// and no distance limit, because it is a property of the sky rather than of
// where the camera is standing.
//
// # How it composes with the probe field, exactly
//
// The two would double-count the sky if both carried it — an open field would
// be lit twice. So the probe march's **miss term is now zero and its hit term
// carries a `-sky` correction** (`gi_probes.wgsl`), which makes the probe field
// the *difference* between the world with geometry in it and the open sky:
//
// ```text
//     ambient = sky_irradiance(n)  +  Σ_hit (bounce - blocked sky)
//             = (the open sky)     -  (the sky the geometry blocks)  +  bounce
// ```
//
// which is the correct occluded sky plus the bounce, to the quadrature, with
// nothing counted twice and no occlusion thrown away. It is also why the term
// above is signed: a probe inside a closed room gathers `-sky` in nearly every
// direction, and the sum is the small bounce that is really there.
//
// Same units and the same folded cosine lobe as `gi_indirect`, so on a uniform
// sky of radiance `L` an albedo-1 Lambert surface leaves exactly `L` — the
// white furnace, asserted on the CPU by
// `atmosphere::tests::a_uniform_sky_projects_to_its_own_radiance` and through
// the renderer by `tests/gi_normalisation.rs`.
fn sky_irradiance(n: vec3<f32>) -> vec3<f32> {
    let b = sh_basis(n);
    return gi.sky_sh0.rgb * b.x
        + 0.66666667 * (gi.sky_sh1.rgb * b.y + gi.sky_sh2.rgb * b.z + gi.sky_sh3.rgb * b.w);
}

// Sky RADIANCE along `d` from the same L1 set, sharpened by `lobe` — the
// specular twin of `sky_irradiance`, and the term that gives `gi_specular` back
// the sky the probe field stopped carrying. Signed, like its neighbours; the
// caller clamps the sum.
fn sky_sh_radiance(d: vec3<f32>, lobe: f32) -> vec3<f32> {
    let b = sh_basis(d);
    return gi.sky_sh0.rgb * b.x
        + lobe * (gi.sky_sh1.rgb * b.y + gi.sky_sh2.rgb * b.z + gi.sky_sh3.rgb * b.w);
}

// **The one door for "what does a shaded surface receive"** (wave FIX3). Every
// lit pass spells its ambient as this call, so the composition of the sky term
// and the probe field is written once rather than six times.
//
// `fallback` is the pass's own hemispheric constant, used when the level has
// not asked for a computed ambient (`GiSettings::enabled`) — which is what
// keeps every GI-off golden byte-identical.
fn ambient_irradiance(world_pos: vec3<f32>, n: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    if (gi.params.x > 0.5) {
        return max(sky_irradiance(n) + gi_indirect(world_pos, n), vec3<f32>(0.0));
    }
    return fallback;
}

// ── P18.4 specular ──────────────────────────────────────────────────────────

// Reconstruct RADIANCE along `d` from an L1 SH set, sharpened by `lobe` (1 = the
// full directional reconstruction, 0 = the DC term alone — what a fully rough
// surface sees). Mirrors `crate::gi::sh_radiance`.
//
// The projection in `gi_probes.wgsl` normalizes by `4π/rays`, which makes this an
// identity on a constant field: a probe that saw uniform radiance L reconstructs
// exactly L. That is why the specular term below reduces to the flat
// `amb * f0 * 0.5` it replaced instead of being a new light source.
// The reconstruction without the clamp (wave FIX3). The probe field is a signed
// difference now, so a caller that adds the sky term to it has to add before
// clamping or it clamps away the occlusion. Declared first because WGSL has no
// forward declarations.
fn gi_sh_radiance_signed(c: mat4x3<f32>, d: vec3<f32>, lobe: f32) -> vec3<f32> {
    let b = sh_basis(d);
    return c[0] * b.x + lobe * (c[1] * b.y + c[2] * b.z + c[3] * b.w);
}

fn gi_sh_radiance(c: mat4x3<f32>, d: vec3<f32>, lobe: f32) -> vec3<f32> {
    return max(gi_sh_radiance_signed(c, d, lobe), vec3<f32>(0.0));
}

// The dominant light direction of an L1 SH set (per-channel luma of the linear
// band). Mirrors `crate::gi::sh_dominant_direction`.
fn gi_sh_dominant_dir(c: mat4x3<f32>) -> vec3<f32> {
    let luma = vec3<f32>(0.2126, 0.7152, 0.0722);
    // Basis order is [Y00, y, z, x] → the vector is (c3, c1, c2).
    let v = vec3<f32>(dot(c[3], luma), dot(c[1], luma), dot(c[2], luma));
    if (dot(v, v) > 1e-12) {
        return normalize(v);
    }
    return vec3<f32>(0.0);
}

// Karis' analytic split-sum environment BRDF: `(a, b)` such that a surface with
// reflectance f0 responds `f0 * a + b`. Mirrors `crate::gi::env_brdf_ab`.
fn gi_env_brdf_ab(rough: f32, nov: f32) -> vec2<f32> {
    let c0 = vec4<f32>(-1.0, -0.0275, -0.572, 0.022);
    let c1 = vec4<f32>(1.0, 0.0425, 1.04, -0.04);
    let r = rough * c0 + c1;
    let a004 = min(r.x * r.x, exp2(-9.28 * nov)) * r.x + r.y;
    return vec2<f32>(-1.04, 1.04) * a004 + r.zw;
}

// ── multi-scatter energy compensation (wave VIS1a) ──────────────────────────
//
// **A single-scatter GGX loses light, and the loss is a function of roughness.**
// The Smith masking-shadowing term accounts for microfacets that occlude the
// view or the light, and then the model simply *drops* what they occluded — but
// a photon that hits a neighbouring facet does not vanish, it bounces again and
// leaves. The missing energy is small at roughness 0.1 and about a third of the
// lobe by roughness 1.0, and because metals have no diffuse term to hide it, a
// rough metal in this engine has been visibly, systematically too dark since
// P7.1. Brushed steel reads as grey plastic; a matte gold reads as brown.
//
// Kulla & Conty's correction, in the cheap form Filament ships: the directional
// albedo of the single-scatter lobe at `f0 = 1` is exactly `a + b` from the
// split-sum fit already used for the ambient specular, and multiplying the lobe
// by `1 + f0·(1/(a+b) − 1)` returns precisely the energy the single-scatter model
// dropped — no more, because at `f0 = 1` the product is `(a+b)·(1/(a+b)) = 1` by
// construction, which is the white-furnace test written as arithmetic. A darker
// `f0` gets proportionally less back, which is right: a dielectric absorbs what
// it re-scatters.
//
// **It is a multiplier on the SPECULAR term only.** The diffuse lobe is
// unaffected, and so is everything the ambient path does — `gi_ambient_specular`
// already integrates the same split-sum response and is energy-correct.
//
// The CPU mirror is `crate::gi::ggx_energy_compensation`, and the furnace test
// that pins the pair is `the_ggx_furnace_test_is_white`.
fn ggx_energy_compensation(f0: vec3<f32>, rough: f32, nov: f32) -> vec3<f32> {
    let ab = gi_env_brdf_ab(rough, nov);
    // The directional albedo at f0 = 1: how much of the incoming energy the
    // single-scatter lobe actually returns.
    let e = max(ab.x + ab.y, 1e-3);
    return vec3<f32>(1.0) + f0 * (1.0 / e - 1.0);
}


// ── SSR v2 (wave VIS1a) ─────────────────────────────────────────────────────
//
// **What changed, and why it is a different feature rather than a tuning pass.**
// SSR v1 (P18.4) was a *hit finder*: it marched the depth buffer to find where the
// reflection ray landed and then re-anchored the GI probe fetch there. That made it
// meaningless without dynamic GI — there were no probes to re-anchor — which is
// why it lived on `GiSettings` and could not be turned on alone. v2 fetches
// **colour**, so it is its own feature with its own settings block, its own clause
// in `needs_depth_prepass`, and no dependency on probes at all.
//
// **The colour comes from the previous frame.** The renderer is forward with 4x
// MSAA: when a lit fragment shades, this frame's opaque colour does not exist and
// there is no G-buffer to defer against — deferring is what the
// visbuffer-off-by-default ruling refuses, on coverage grounds. So the fetch reads
// `targets.scene_hdr`, which `passes::resolve` wrote at the end of the previous
// frame, **reprojected through that frame's view-projection** so the sample lands
// where the hit point was when the colour was written. The price is one frame of
// latency; the bargain is the one `taa` and `cloud_temporal` already strike, and
// SSR is off by default on the same terms.
//
// **The march is geometric, not uniform.** v1 spent 24 evenly-spaced samples over
// 8 m, which is 33 cm of near-field resolution — coarse enough that a contact
// reflection lands a third of a metre from the object making it. `t = max_dist *
// u²` puts the first step 2 cm out at 32 samples over 24 m and lets the tail
// coarsen, which is where a miss is cheap and a hit is a distant, low-detail
// reflection anyway. A crossing is then bisected three times, so the reported hit
// is inside the step it was found in rather than at its far end.
//
// Reverse-infinite-Z: a LARGER ndc.z is NEARER and view distance is proportional
// to 1/ndc.z, so the penetration test is a ratio and needs no near plane.

// How strongly this surface reflects at all. Full up to half the authored cutoff,
// fading to nothing at it — a hard stop would put a visible edge across a surface
// whose roughness varies.
fn ssr_roughness_weight(rough: f32) -> f32 {
    let cutoff = max(gi.ssr.y, 1e-4);
    return 1.0 - smoothstep(cutoff * 0.5, cutoff, rough);
}

// The previous frame's resolved colour at render-local point `p`, with an edge
// fade in `w`. `w == 0` means "off screen last frame" — a miss, not black.
fn ssr_prev_color(p: vec3<f32>) -> vec4<f32> {
    let clip = gi.prev_view_proj * vec4<f32>(p, 1.0);
    if (clip.w <= 0.0) {
        return vec4<f32>(0.0);
    }
    let ndc = clip.xyz / clip.w;
    if (any(abs(ndc.xy) > vec2<f32>(1.0))) {
        return vec4<f32>(0.0);
    }
    let uv = ndc.xy * vec2<f32>(0.5, -0.5) + 0.5;
    // Fade over the outer 8 % of the frame: a reflection that walks off the edge
    // of the screen must dissolve rather than end in a line.
    let e = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
    let edge = smoothstep(0.0, 0.08, e);
    let c = textureSampleLevel(ssr_scene_color, atmos_lut_smp, uv, 0.0).rgb;
    return vec4<f32>(c, edge);
}

// March `dir` from `origin` against this frame's depth. Returns
// `vec4(reflected_colour, confidence)`; confidence 0 is a miss.
fn ssr_trace(origin: vec3<f32>, dir: vec3<f32>) -> vec4<f32> {
    let max_dist = gi.params2.z;
    let thickness = gi.params2.w;
    let vp = view.grid_axis_viewport.zw;
    let steps = i32(max(gi.ssr.x, 1.0));
    let inv = 1.0 / f32(steps);

    var prev_t = 0.0;
    for (var s = 1; s <= steps; s = s + 1) {
        let u = f32(s) * inv;
        let t = max_dist * u * u;
        let p = origin + dir * t;
        let clip = view.view_proj * vec4<f32>(p, 1.0);
        if (clip.w <= 0.0) {
            return vec4<f32>(0.0); // behind the camera — nothing further can hit
        }
        let ndc = clip.xyz / clip.w;
        if (any(abs(ndc.xy) > vec2<f32>(1.0)) || ndc.z <= 0.0) {
            return vec4<f32>(0.0); // left the frame — every later sample is too
        }
        let texel = vec2<i32>(clamp(ndc.xy * vec2<f32>(0.5, -0.5) * vp + 0.5 * vp,
                                    vec2<f32>(0.0), vp - vec2<f32>(1.0)));
        let scene_z = textureLoad(gi_scene_depth, texel, 0);
        if (scene_z <= 0.0) {
            prev_t = t;
            continue; // sky / nothing rasterized here
        }
        // The scene surface is nearer than the ray sample ⇒ the ray went behind
        // it. Accept only a shallow penetration, or a ray passing far behind a
        // foreground object would report a spurious hit on its silhouette.
        let penetration = 1.0 - ndc.z / scene_z;
        if (penetration > 0.0 && penetration < thickness) {
            // Bisect the crossing: the step that hit may be metres long in the
            // tail, and the colour fetched at its far end is the colour of
            // whatever is behind the thing being reflected.
            var lo = prev_t;
            var hi = t;
            for (var b = 0; b < 3; b = b + 1) {
                let mid = 0.5 * (lo + hi);
                let q = origin + dir * mid;
                let qc = view.view_proj * vec4<f32>(q, 1.0);
                let qn = qc.xyz / max(qc.w, 1e-6);
                let qt = vec2<i32>(clamp(qn.xy * vec2<f32>(0.5, -0.5) * vp + 0.5 * vp,
                                         vec2<f32>(0.0), vp - vec2<f32>(1.0)));
                let qz = textureLoad(gi_scene_depth, qt, 0);
                if (qz > 0.0 && qn.z < qz) {
                    hi = mid; // already behind the surface
                } else {
                    lo = mid;
                }
            }
            let hit = origin + dir * hi;
            let c = ssr_prev_color(hit);
            // Confidence also falls off with march distance: a reflection found
            // at the very end of the ray is the one most likely to be a
            // silhouette artefact, and the one the fallback approximates best.
            let far = 1.0 - smoothstep(0.75, 1.0, hi / max(max_dist, 1e-4));
            return vec4<f32>(c.rgb, c.w * far);
        }
        prev_t = t;
    }
    return vec4<f32>(0.0);
}

// The GI specular term: prefiltered environment radiance along the reflection
// vector, times the split-sum environment BRDF. Replaces the flat
// `ambient * f0 * 0.5` the lit passes used, and is EXACTLY that value in the limit
// of a uniform radiance field with a fully rough surface (see `gi_sh_radiance`) —
// so turning GI specular on adds directionality, not energy.
//
// `rough` is perceptual roughness, `f0` the surface reflectance at normal
// incidence. The caller multiplies by AO.
fn gi_specular(world_pos: vec3<f32>, n: vec3<f32>, v: vec3<f32>, rough: f32, f0: vec3<f32>) -> vec3<f32> {
    let r = reflect(-v, n);
    // **The SSR re-anchoring left this function in wave VIS1a.** It existed
    // because v1 had no colour to fetch, so the best a hit could do was move the
    // probe fetch point; v2 fetches the colour itself and blends it over this
    // term as the fallback (`gi_ambient_specular`). Keeping both would apply the
    // parallax twice.
    let c = gi_fetch_sh(world_pos, n);
    // A rough surface integrates the lobe away; a smooth one keeps it.
    let lobe = clamp(1.0 - rough, 0.0, 1.0);
    // **The sky half is explicit since wave FIX3.** The probe field no longer
    // carries the sky (see `sky_irradiance`), so a reflection that read only
    // the probes would have lost it — a wet road would reflect the buildings
    // and not the sky above them. Reconstructed from the same L1 set the
    // diffuse ambient uses and added before the clamp, because the probe term
    // is signed and its negative half IS the sky this surface cannot see.
    let radiance = max(
        gi_sh_radiance_signed(c, r, lobe) * gi.params.y + sky_sh_radiance(r, lobe),
        vec3<f32>(0.0),
    );
    let ab = gi_env_brdf_ab(rough, clamp(dot(n, v), 0.0, 1.0));
    // **No π here either, since wave EDIT1.** This term carried a `GI_PI` whose
    // only job was to match `gi_irradiance`'s π·L convention — the comment said
    // so — and the split-sum form `L_prefiltered · (f0·a + b)` is in exit-radiance
    // units on its own. With the diffuse half now honest, multiplying here would
    // put the same factor back on the specular half alone. The retired
    // `ambient × f0 × 0.5` is still recovered exactly, because `ambient` at this
    // call site is `gi_irradiance`'s output and moved by the same π: what the
    // split-sum BRDF gives a fully rough, face-on surface is
    // `env_brdf_ab(1, 1) ≈ (0.45, 0)` against the retired 0.5. Pinned by
    // `gi::tests::specular_reduces_to_the_retired_ambient_constant`.
    return radiance * (f0 * ab.x + vec3<f32>(ab.y));
}

// The full ambient specular for a lit pass. Written once here so every lit shader
// takes the identical off-path instruction stream.
//
// Three layers, outermost last:
//   1. the pre-P18.4 constant `ambient · f0 · 0.5`;
//   2. P18.4's SH reconstruction along the reflection vector, when GI and its
//      specular flag are on;
//   3. wave VIS1a's SSR, blended over whichever of those is the fallback.
//
// **The fallback is the design, not a failure path.** A screen-space march can
// only ever answer for what is on screen; everything it cannot see — behind the
// camera, off the edge, occluded, too rough to be worth a mirror ray — is
// answered by the term underneath, which is the probe field when there is one and
// the sky-tinted constant when there is not. So the reflection dissolves into
// something plausible instead of into black.
fn gi_ambient_specular(
    world_pos: vec3<f32>,
    n: vec3<f32>,
    v: vec3<f32>,
    rough: f32,
    f0: vec3<f32>,
    ambient: vec3<f32>,
) -> vec3<f32> {
    var base: vec3<f32>;
    if (gi.params.x > 0.5 && gi.params2.x > 0.5) {
        base = gi_specular(world_pos, n, v, rough, f0);
    } else {
        base = ambient * f0 * 0.5;
    }
    // `params2.y` is 0 on every scene with SSR off, so this branch is untaken and
    // the arithmetic above is exactly what it has been since P18.4.
    if (gi.params2.y > 0.5) {
        // `ssr.w` is the history flag: on the first frame after a resize the
        // colour buffer is a zero-initialized allocation, and reflecting it would
        // be reflecting black.
        let w = ssr_roughness_weight(rough) * gi.ssr.z * gi.ssr.w;
        if (w > 0.0) {
            let hit = ssr_trace(world_pos + n * 0.02, reflect(-v, n));
            if (hit.w > 0.0) {
                // The same split-sum response the fallback is expressed in, so
                // the blend is between two terms in one set of units.
                let ab = gi_env_brdf_ab(rough, clamp(dot(n, v), 0.0, 1.0));
                let refl = hit.rgb * (f0 * ab.x + vec3<f32>(ab.y));
                base = mix(base, refl, clamp(w * hit.w, 0.0, 1.0));
            }
        }
    }
    return base;
}
