// Dynamic-GI probe march (P13.3b, rebuilt in P18.4): one compute thread per
// **scheduled** probe marches `rays` fixed golden-spiral directions through the
// voxel volume, gathers single-bounce radiance, and projects it to L1 spherical
// harmonics (4 coeffs × RGB) written to a storage buffer the lit passes sample.
// Deterministic (fixed directions, no temporal jitter). Mirrors `crate::gi`.
//
//   hit  → radiance = albedo × sun_radiance × sun_visibility(hit) + emissive
//   miss → radiance = the P17.2 SKY-VIEW LUT in that direction, or the authored
//          gradient when the scene has no atmosphere
//
// P18.4 changes, all visible in the two lines above plus the dispatch shape:
//
// * **Sky from the atmosphere.** `sched.w` selects the source. With a time-of-day
//   authority the miss term samples the same Hillaire sky-view LUT the sky pass
//   draws, so the bounce tracks dawn/noon/dusk instead of two authored constants —
//   the tracked P17 deferral. Without one, the gradient path runs the identical
//   arithmetic it always did, which is what keeps `gi_bleed`'s sky term stable.
// * **Emissive injection.** A voxel's second word carries emissive radiance, added
//   on hit — so an emissive surface lights the room with no analytic light at all.
// * **Temporal amortization.** The dispatch covers `sched.y` probes starting at
//   `sched.x` (wrapping modulo `sched.z`), a deterministic round-robin driven by a
//   renderer-side cursor (`crate::gi::ProbeSchedule`), never by a frame index.
//   Full update = `sched.x = 0, sched.y = sched.z`, which is what the goldens and
//   the determinism gates render with.

// The atmosphere library (medium at binding 3, LUTs at 4/5/6) is composed in
// front of this file by `passes::gi_probe_shader`.

struct GiData {
    vol_min: vec4<f32>,
    probe_min: vec4<f32>,
    dims: vec4<f32>,
    params: vec4<f32>,
    sun_dir: vec4<f32>,
    sun_color: vec4<f32>,
    sky_zenith: vec4<f32>,
    sky_horizon: vec4<f32>,
    params2: vec4<f32>,
    sched: vec4<f32>,
    // Declared through to the end since wave FIX3, because `sky_sh*` is at the
    // tail and a uniform struct reads its buffer positionally. The two VIS1a
    // lanes are unread here and named so the layout is legible.
    ssr: vec4<f32>,
    prev_view_proj: mat4x4<f32>,
    sky_sh0: vec4<f32>,
    sky_sh1: vec4<f32>,
    sky_sh2: vec4<f32>,
    sky_sh3: vec4<f32>,
};
@group(0) @binding(0) var<uniform> gi: GiData;
@group(0) @binding(1) var<storage, read> voxels: array<u32>;
@group(0) @binding(2) var<storage, read_write> sh: array<vec4<f32>>;

const PI: f32 = 3.14159265359;
const GI_EMISSIVE_MAX: f32 = 16.0;

fn unpack_rgba8(v: u32) -> vec4<f32> {
    return vec4<f32>(
        f32(v & 0xffu),
        f32((v >> 8u) & 0xffu),
        f32((v >> 16u) & 0xffu),
        f32((v >> 24u) & 0xffu),
    ) / 255.0;
}

// Mirrors `gi::unpack_emissive`.
fn gi_unpack_emissive(v: u32) -> vec3<f32> {
    let c = unpack_rgba8(v);
    return c.rgb * (c.a * GI_EMISSIVE_MAX);
}

struct GiVoxel {
    albedo: vec3<f32>,
    solid: f32,
    emissive: vec3<f32>,
};

fn voxel_at(c: vec3<i32>) -> GiVoxel {
    let dim = i32(gi.dims.x);
    var out: GiVoxel;
    out.albedo = vec3<f32>(0.0);
    out.solid = 0.0;
    out.emissive = vec3<f32>(0.0);
    if (any(c < vec3<i32>(0)) || any(c >= vec3<i32>(dim))) {
        return out; // outside → empty
    }
    let d = u32(dim);
    let uc = vec3<u32>(c);
    let idx = (uc.z * d + uc.y) * d + uc.x;
    let a = unpack_rgba8(voxels[idx * 2u + 0u]);
    out.albedo = a.rgb;
    out.solid = a.a;
    out.emissive = gi_unpack_emissive(voxels[idx * 2u + 1u]);
    return out;
}

// Sample occupancy+albedo+emissive at render-local point `p`.
fn sample_point(p: vec3<f32>) -> GiVoxel {
    let coord = (p - gi.vol_min.xyz) / gi.vol_min.w;
    return voxel_at(vec3<i32>(floor(coord)));
}

fn spiral_dir(i: u32, n: u32) -> vec3<f32> {
    let fn_ = f32(max(n, 1u));
    let fi = f32(i);
    let phi = PI * (3.0 - sqrt(5.0));
    let y = 1.0 - 2.0 * (fi + 0.5) / fn_;
    let r = sqrt(max(1.0 - y * y, 0.0));
    let theta = phi * fi;
    return vec3<f32>(cos(theta) * r, y, sin(theta) * r);
}

fn sh_basis(d: vec3<f32>) -> vec4<f32> {
    return vec4<f32>(0.282095, 0.488603 * d.y, 0.488603 * d.z, 0.488603 * d.x);
}

// **MIRROR of `sky_irradiance` in `shaders/env_lighting.wgsl`** — kept
// character-for-character with it and pinned by
// `passes::gi::tests::the_two_sky_irradiance_bodies_are_the_same`. The two files
// cannot share a snippet: each declares its own `GiData` (this pass binds the
// uniform at `@group(0)`, the lit passes at their env group) and WGSL has no
// forward declarations, so a shared fragment would have to be composed before a
// struct that does not exist yet.
fn sky_irradiance(n: vec3<f32>) -> vec3<f32> {
    let b = sh_basis(n);
    return gi.sky_sh0.rgb * b.x
        + 0.66666667 * (gi.sky_sh1.rgb * b.y + gi.sky_sh2.rgb * b.z + gi.sky_sh3.rgb * b.w);
}

// March toward the sun from `p`; 0 if an occluder is hit before leaving the volume.
fn sun_visibility(p: vec3<f32>) -> f32 {
    let vsize = gi.vol_min.w;
    let dir = normalize(gi.sun_dir.xyz);
    let dim = gi.dims.x;
    var pos = p + dir * vsize * 1.5; // step off the surface
    let steps = i32(dim);
    for (var s = 0; s < steps; s = s + 1) {
        if (sample_point(pos).solid > 0.5) {
            return 0.0;
        }
        pos = pos + dir * vsize;
    }
    return 1.0;
}

// The miss term. `sched.w > 0.5` ⇒ the scene has a physical atmosphere and the
// probes read the SAME sky-view LUT the sky pass draws (P18.4, closing the P17
// deferral); otherwise the authored two-colour gradient, byte-identical to v1.
fn gi_sky_radiance(dir: vec3<f32>) -> vec3<f32> {
    if (gi.sched.w > 0.5) {
        return atmos_sample_skyview(atmos.planet.z, dir);
    }
    let t = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
    return mix(gi.sky_horizon.rgb, gi.sky_zenith.rgb, t);
}

@compute @workgroup_size(64)
fn cs_probes(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = u32(gi.dims.y);
    let py = u32(gi.dims.z);
    let pz = u32(gi.dims.w);
    let total = max(u32(gi.sched.z), 1u);
    let scheduled = u32(gi.sched.y);
    if (gid.x >= scheduled) {
        return;
    }
    // Round-robin slice, wrapping so no probe is starved when `total` is not a
    // multiple of the budget.
    let pi = (u32(gi.sched.x) + gid.x) % total;
    if (pi >= px * py * pz) {
        return;
    }
    let ix = pi % px;
    let iy = (pi / px) % py;
    let iz = pi / (px * py);
    let extent = gi.probe_min.w;
    let frac = vec3<f32>(
        select(f32(ix) / f32(px - 1u), 0.5, px <= 1u),
        select(f32(iy) / f32(py - 1u), 0.5, py <= 1u),
        select(f32(iz) / f32(pz - 1u), 0.5, pz <= 1u),
    );
    let origin = gi.probe_min.xyz + frac * extent;

    let vsize = gi.vol_min.w;
    let dim = gi.dims.x;
    let rays = u32(gi.params.z);

    var c0 = vec3<f32>(0.0);
    var c1 = vec3<f32>(0.0);
    var c2 = vec3<f32>(0.0);
    var c3 = vec3<f32>(0.0);

    for (var r = 0u; r < rays; r = r + 1u) {
        let dir = spiral_dir(r, rays);
        // March the ray through the volume.
        var pos = origin + dir * vsize * 0.5;
        var radiance = vec3<f32>(0.0);
        var hit = false;
        let steps = i32(dim) * 2;
        for (var s = 0; s < steps; s = s + 1) {
            let v = sample_point(pos);
            if (v.solid > 0.5) {
                let vis = sun_visibility(pos);
                // Single bounce + injected emission. Emissive is added rather
                // than multiplied by visibility: a light source does not need the
                // sun's permission to glow.
                //
                // **The Lambert 1/π, here too** (wave EDIT1). What a probe ray
                // gathers is the RADIANCE LEAVING the surface it hit, and a
                // Lambert surface under a light of radiance `S` leaves
                // `albedo/π · S`, which is what `shade_light` spells out as
                // `kd * albedo / PI` two files away. Without the divide this
                // term was π times a lit wall — and on a sunlit street, where
                // most rays hit a wall rather than the sky, it is the DOMINANT
                // half of the gather, not the miss term. Measured: with the
                // consumer's π removed and this one still in, GI lifted a
                // daylight street's p95 by 49 levels under the engine's own
                // dim default sky, which is a number no sky that dim can pay
                // for. Emissive is already a radiance and keeps its units.
                //
                // The cosine is still missing and stays missing: the voxel
                // carries no normal, so `n·l` cannot be evaluated. That makes
                // this an over-estimate by the average of a cosine over the lit
                // hemisphere rather than by a factor of π, and it is a
                // modelling approximation of single-bounce voxel GI rather than
                // a unit error. Carried, with the number, in the ledger.
                // **The surface the ray hit is lit by the sky too** (wave
                // FIX3). Before this it was lit by the sun and by nothing
                // else, so every wall facing away from the sun bounced
                // EXACTLY ZERO — and on a street, where most of a probe's rays
                // end on a wall, that is most of the gather. It is the
                // mechanism behind the FIX2 audit's photograph: a hero two
                // metres from a shaded building, whose nearest probes see that
                // building and nothing else, lit by the sum of a set of blacks.
                //
                // The voxel still carries no normal, so the ray's own
                // direction is the proxy: a ray travelling along `dir` can only
                // have struck a face pointing back along `-dir`. That is exact
                // for a flat surface met head-on and an over-estimate at
                // grazing incidence, and it is what finally puts the **cosine**
                // on the sun term — EDIT1's carried "GI bounce has no n·l",
                // closed here with the same proxy that pays for the sky.
                let nh = -dir;
                let ndl = max(dot(nh, normalize(gi.sun_dir.xyz)), 0.0);
                let incident = gi.sun_color.rgb * vis * ndl * (1.0 / PI)
                    + max(sky_irradiance(nh), vec3<f32>(0.0));
                radiance = v.albedo * incident + v.emissive;
                hit = true;
                break;
            }
            pos = pos + dir * vsize;
        }
        // **The probe field is a DIFFERENCE from the open sky since wave FIX3.**
        //
        // The lit passes now add `sky_irradiance(n)` — the whole unoccluded sky
        // — to this gather, so a gather that also carried the sky would light an
        // open field twice. What this field carries instead is what the geometry
        // CHANGES: the bounce it adds, minus the sky it blocks.
        //
        // ```text
        //     ambient = sky_irradiance(n) + Σ_hit (bounce - blocked sky)
        //             = Σ_miss sky + Σ_hit bounce
        // ```
        //
        // which is the same integral the pre-FIX3 gather computed, to the
        // quadrature — occlusion intact, nothing counted twice — and it is the
        // reason `gi_indirect` is no longer clamped at zero: a probe inside a
        // closed room legitimately gathers `-sky` in nearly every direction.
        if (hit) {
            radiance = radiance - gi_sky_radiance(dir);
        } else {
            radiance = vec3<f32>(0.0);
        }
        let b = sh_basis(dir);
        c0 = c0 + radiance * b.x;
        c1 = c1 + radiance * b.y;
        c2 = c2 + radiance * b.z;
        c3 = c3 + radiance * b.w;
    }

    let norm = 4.0 * PI / f32(max(rays, 1u));
    let base = pi * 4u;
    sh[base + 0u] = vec4<f32>(c0 * norm, 0.0);
    sh[base + 1u] = vec4<f32>(c1 * norm, 0.0);
    sh[base + 2u] = vec4<f32>(c2 * norm, 0.0);
    sh[base + 3u] = vec4<f32>(c3 * norm, 0.0);
}
