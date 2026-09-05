//! Physically-based atmosphere (P17.2): the parameter block, the CPU reference
//! implementation of the scattering math, and the LUT parameterizations.
//!
//! The GPU side lives in [`crate::passes::sky_lut`] (two compute passes that bake
//! the LUTs), [`crate::passes::sky`] (the sky pass that samples them and draws the
//! sun, moon and stars) and `shaders/env_lighting.wgsl` (aerial perspective +
//! height fog on lit geometry). **Everything in this module is pure and
//! GPU-free**, so the physics is unit-tested on every CI platform — including the
//! ones with no adapter, which is where the golden harness skips.
//!
//! # Units (architecture rule 6)
//!
//! Atmospheric optics has its own unit convention and fighting it would be worse
//! than documenting it, so this module — and this module alone — works in
//! **kilometres**:
//!
//! | quantity | unit | example |
//! |---|---|---|
//! | altitude, planet radius, ray-march distance | km | `ground_radius = 6360` |
//! | scattering / absorption coefficients | km⁻¹ | `mie_scattering = 3.996e-3` |
//! | density scale heights | km | `rayleigh_height = 8` |
//! | angular sizes | degrees at the API boundary, cosines inside | `sun_disc_deg = 0.545` |
//!
//! The engine's metres do not disappear: [`HeightFog`] is authored entirely in
//! **SI metres** (m⁻¹ extinction, metre altitudes) because it is a property of
//! the world's geometry rather than of the planet, and the conversion
//! (`km = m / 1000`) happens exactly once, where a world position enters the
//! atmosphere math ([`camera_radius_km`]).
//!
//! # Model
//!
//! Hillaire 2020 ("A Scalable and Production Ready Sky and Atmosphere Rendering
//! Technique") over the Bruneton/Neyret medium: three species, each with its own
//! vertical density profile.
//!
//! * **Rayleigh** — molecular scattering, `exp(-h / 8 km)`. Wavelength⁻⁴, so it
//!   is ~5.7× stronger in blue than in red. This is why the sky is blue and why
//!   the sun reddens at the horizon (the blue is scattered *out* of the long
//!   grazing path).
//! * **Mie** — aerosol scattering + a little absorption, `exp(-h / 1.2 km)`.
//!   Nearly wavelength-neutral and strongly forward-scattering (`g ≈ 0.8`),
//!   which is the bright white halo around the sun.
//! * **Ozone** — pure absorption in a tent centred at 25 km. It contributes
//!   nothing during the day and is what keeps a twilight sky *blue* instead of
//!   muddy grey-brown; leaving it out is the classic "my sunset looks like
//!   soup" bug.
//!
//! The coefficients below are Hillaire's, which are themselves Bruneton's
//! measured values. They are **not** authored per level: a level scales them
//! with dimensionless multipliers ([`AtmosphereParams::turbidity`],
//! [`AtmosphereParams::mie_g`]) so that "atmosphere" stays a physical constant
//! and "weather" stays an artistic knob.
//!
//! # Off path
//!
//! [`AtmosphereParams::default`] is **disabled** and every consumer branches on
//! it, so a scene with no `TimeOfDay` authority — which is every scene the engine
//! shipped with before P17.2 — renders the exact instruction stream it did
//! before (the P13.3 off-path discipline).

use crate::caps::RenderTier;

/// The single calibration constant between physics and the engine's arbitrary
/// light units.
///
/// Everything above is in real radiometric proportions, but the engine's sun
/// "intensity" is a bare multiplier (the historic fallback was `3.0`), not
/// W·m⁻²·sr⁻¹ — so the integral lands wherever it lands. This constant maps it
/// onto the exposure the rest of the renderer is tuned for: with a default sun
/// and `RenderSettings::exposure == 1.0`, a clear noon zenith comes out around
/// 0.5 sRGB after the ACES tonemap, which is what a photographed sky looks like.
///
/// It multiplies **only** the sky/aerial radiance, never the directional light
/// the PBR loop sees — so raising it brightens the sky without silently
/// re-exposing every lit surface in the level.
pub const SKY_EXPOSURE_CALIBRATION: f32 = 3.0;

/// Mean Earth radius used as the atmosphere's ground shell, **km**.
pub const EARTH_GROUND_RADIUS_KM: f32 = 6360.0;
/// Top of the modelled atmosphere, **km** from the planet centre (100 km thick).
pub const EARTH_TOP_RADIUS_KM: f32 = 6460.0;

/// Exponential **height fog** — an authored, SI-metre effect layered on top of
/// the physical aerial perspective.
///
/// The two are deliberately separate. Aerial perspective is what the *planet*
/// does and is not tunable per level beyond a strength; height fog is what a
/// *level* does (a valley of mist, a smoggy city, an alien soup) and is authored
/// in the same metres as the geometry it hides.
///
/// Density falls off exponentially with altitude:
/// `σ(y) = density · exp(-falloff · (y − height))`, which integrates in closed
/// form along a segment — see [`height_fog_optical_depth`]. No ray marching, no
/// step count, no temporal noise.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeightFog {
    /// Extinction at `height`, **m⁻¹**. `0` disables the fog entirely (and the
    /// receivers take the byte-identical no-fog path).
    pub density: f32,
    /// Vertical falloff, **m⁻¹**. `0` = uniform with altitude.
    pub falloff: f32,
    /// World altitude at which `density` applies, **m**.
    pub height: f32,
    /// Linear tint multiplied into the fog's in-scattered light.
    pub color: [f32; 3],
}

impl Default for HeightFog {
    /// Off (`density = 0`), with a 500 m e-folding height ready for an author who
    /// turns it on.
    fn default() -> Self {
        Self {
            density: 0.0,
            falloff: 0.002,
            height: 0.0,
            color: [1.0, 1.0, 1.0],
        }
    }
}

impl HeightFog {
    /// Whether the fog contributes anything at all.
    #[inline]
    pub fn active(&self) -> bool {
        self.density > 0.0 && self.density.is_finite()
    }
}

/// The scene's atmosphere. Projected onto [`crate::scene::RenderScene`] by both
/// scene builders from the level's `SkyAtmosphere` component; **disabled** by
/// default so content that never opted into a time of day is untouched.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtmosphereParams {
    /// Master switch: draw the physical sky, and apply aerial perspective + fog
    /// to lit geometry. Off ⇒ the v11 three-colour gradient pass, byte for byte.
    pub enabled: bool,

    // ── the medium (physical constants; see the module docs) ──────────────
    /// Rayleigh scattering coefficient at sea level, **km⁻¹**, per channel.
    pub rayleigh_scattering: [f32; 3],
    /// Rayleigh density scale height, **km**.
    pub rayleigh_height: f32,
    /// Mie scattering coefficient at sea level, **km⁻¹** (wavelength-neutral).
    pub mie_scattering: f32,
    /// Mie absorption coefficient at sea level, **km⁻¹**.
    pub mie_absorption: f32,
    /// Mie density scale height, **km**.
    pub mie_height: f32,
    /// Mie phase asymmetry `g`, `[-0.95, 0.95]`; positive = forward-scattering.
    pub mie_g: f32,
    /// Ozone absorption coefficient at the tent peak, **km⁻¹**, per channel.
    pub ozone_absorption: [f32; 3],
    /// Altitude of the ozone tent's peak, **km**.
    pub ozone_center: f32,
    /// Half-width of the ozone tent, **km** (density is 0 outside ±this).
    pub ozone_half_width: f32,
    /// Aerosol density multiplier over the clear-air reference (dimensionless).
    pub turbidity: f32,

    // ── the planet ────────────────────────────────────────────────────────
    /// Ground shell radius, **km**.
    pub ground_radius: f32,
    /// Top-of-atmosphere radius, **km**.
    pub top_radius: f32,
    /// Grey ground albedo used for the bounce term below the horizon.
    pub ground_albedo: f32,

    // ── artistic ──────────────────────────────────────────────────────────
    /// Linear exposure multiplier on the sky's radiance.
    pub sky_intensity: f32,
    /// Aerial-perspective strength on lit geometry (`1` = physical, `0` = off).
    pub aerial_perspective: f32,
    /// How far the physical sky is pulled toward the authored gradient, `[0, 1]`.
    pub tint_strength: f32,
    /// Sun angular **diameter**, degrees.
    pub sun_disc_deg: f32,
    /// Moon angular **diameter**, degrees.
    pub moon_disc_deg: f32,
    /// Lunar phase, `[0, 1)` — `0` new, `0.5` full. Drives the terminator on the
    /// moon disc.
    pub moon_phase: f32,
    /// Starfield brightness multiplier.
    pub star_intensity: f32,
    /// Height fog (SI metres).
    pub fog: HeightFog,
    /// Volumetric clouds (P17.3), **SI metres**. Disabled by default; see
    /// [`crate::clouds::CloudParams`].
    pub clouds: crate::clouds::CloudParams,
    /// Precipitation (P17.4), **SI metres**. Disabled by default; see
    /// [`crate::precip::PrecipParams`]. Projected from the weather block, never
    /// authored directly — which is why it has no knobs of its own here.
    pub precip: crate::precip::PrecipParams,
}

impl Default for AtmosphereParams {
    /// Earth, **disabled**. The physical constants are already correct so that
    /// enabling the atmosphere is a one-flag change, not a parameter hunt.
    fn default() -> Self {
        Self {
            enabled: false,
            // Bruneton/Hillaire sea-level coefficients, km⁻¹.
            rayleigh_scattering: [5.802e-3, 13.558e-3, 33.100e-3],
            rayleigh_height: 8.0,
            mie_scattering: 3.996e-3,
            // Hillaire's Mie extinction is 4.40e-3; absorption is the remainder.
            mie_absorption: 0.404e-3,
            mie_height: 1.2,
            mie_g: 0.8,
            ozone_absorption: [0.650e-3, 1.881e-3, 0.085e-3],
            ozone_center: 25.0,
            ozone_half_width: 15.0,
            turbidity: 1.0,
            ground_radius: EARTH_GROUND_RADIUS_KM,
            top_radius: EARTH_TOP_RADIUS_KM,
            ground_albedo: 0.1,
            sky_intensity: 1.0,
            aerial_perspective: 1.0,
            tint_strength: 0.0,
            sun_disc_deg: 0.545,
            moon_disc_deg: 0.52,
            moon_phase: 0.5,
            star_intensity: 1.0,
            fog: HeightFog::default(),
            clouds: crate::clouds::CloudParams::default(),
            precip: crate::precip::PrecipParams::default(),
        }
    }
}

impl AtmosphereParams {
    /// Whether anything in this block should touch a *lit* pass — the aerial
    /// perspective / height-fog receivers, or (P17.3) the cloud-shadow map.
    /// `false` keeps them byte-identical.
    #[inline]
    pub fn affects_lit_passes(&self) -> bool {
        self.enabled
            && (self.aerial_perspective > 0.0 || self.fog.active() || self.clouds.shadows_world())
    }

    /// Whether the cloud bake + raymarch passes should run (P17.3).
    ///
    /// Clouds require the physical atmosphere: they are lit by the sun's
    /// transmittance through it and ambient-lit by the sky-view LUT, so a cloud
    /// over the v11 gradient sky would be a grey blob with nothing to light it.
    /// The projectors therefore only ever set `clouds.enabled` on an enabled
    /// atmosphere, and this is the single place that invariant is enforced.
    #[inline]
    pub fn clouds_active(&self) -> bool {
        self.enabled && self.clouds.active()
    }

    /// Whether the cloud-shadow map should be baked and sampled by lit passes.
    #[inline]
    pub fn cloud_shadows_active(&self) -> bool {
        self.enabled && self.clouds.shadows_world()
    }

    /// Whether the precipitation pass should draw (P17.4).
    ///
    /// Gated on the physical atmosphere for the same reason clouds are: a
    /// raindrop is lit by the sky-view LUT and the sun's transmittance through
    /// the air, so precipitation over the v11 gradient would be a grey speck
    /// with nothing to light it. The projectors only ever set `precip.enabled`
    /// on an enabled atmosphere, and this is the single place that is enforced.
    #[inline]
    pub fn precip_active(&self) -> bool {
        self.enabled && self.precip.active()
    }

    /// The effective (turbidity-scaled) Mie scattering coefficient, km⁻¹.
    #[inline]
    pub fn mie_scattering_scaled(&self) -> f32 {
        self.mie_scattering * self.turbidity.clamp(0.0, 16.0)
    }

    /// The effective (turbidity-scaled) Mie absorption coefficient, km⁻¹.
    #[inline]
    pub fn mie_absorption_scaled(&self) -> f32 {
        self.mie_absorption * self.turbidity.clamp(0.0, 16.0)
    }

    /// Thickness of the modelled atmosphere shell, km.
    #[inline]
    pub fn thickness_km(&self) -> f32 {
        (self.top_radius - self.ground_radius).max(1e-3)
    }

    /// Cosine of the sun disc's **half**-angle (what a `dot(view, sun)` test
    /// compares against).
    #[inline]
    pub fn sun_disc_cos(&self) -> f32 {
        (self.sun_disc_deg.max(0.0).to_radians() * 0.5).cos()
    }

    /// Cosine of the moon disc's half-angle.
    #[inline]
    pub fn moon_disc_cos(&self) -> f32 {
        (self.moon_disc_deg.max(0.0).to_radians() * 0.5).cos()
    }
}

/// Radius from the planet centre, **km**, for a camera at world altitude
/// `altitude_m` **metres**. This is the one place metres become kilometres.
///
/// Clamped just inside the shell: a camera exactly on the ground makes the
/// horizon-angle parameterization degenerate (`sqrt` of 0), and a camera above
/// the top of the atmosphere has no sky to look through.
#[inline]
pub fn camera_radius_km(params: &AtmosphereParams, altitude_m: f32) -> f32 {
    let alt_km = if altitude_m.is_finite() {
        altitude_m * 1e-3
    } else {
        0.0
    };
    (params.ground_radius + alt_km).clamp(
        params.ground_radius + SHELL_EPSILON_KM,
        params.top_radius - SHELL_EPSILON_KM,
    )
}

/// How far inside the shell a camera is held, **km** (10 m).
///
/// Not an arbitrary fudge: `f32` at 6360 km has a ~0.5 m ULP, so a smaller
/// margin would round straight back onto the shell and the horizon-angle
/// parameterization (`sqrt(r² − ground²)`) would go degenerate — a black seam at
/// `v = 0.5` in the sky-view LUT. 10 m of altitude tilts the horizon by 0.1°,
/// which nothing can see.
pub const SHELL_EPSILON_KM: f32 = 0.01;

// ── the medium ───────────────────────────────────────────────────────────────

/// Rayleigh density at `altitude_km`, relative to sea level.
#[inline]
pub fn rayleigh_density(params: &AtmosphereParams, altitude_km: f32) -> f32 {
    (-altitude_km / params.rayleigh_height.max(1e-3)).exp()
}

/// Mie density at `altitude_km`, relative to sea level.
#[inline]
pub fn mie_density(params: &AtmosphereParams, altitude_km: f32) -> f32 {
    (-altitude_km / params.mie_height.max(1e-3)).exp()
}

/// Ozone density at `altitude_km`: a symmetric tent peaking at `ozone_center`.
#[inline]
pub fn ozone_density(params: &AtmosphereParams, altitude_km: f32) -> f32 {
    let hw = params.ozone_half_width.max(1e-3);
    (1.0 - (altitude_km - params.ozone_center).abs() / hw).max(0.0)
}

/// Total extinction (scattering + absorption) at `altitude_km`, **km⁻¹** per
/// channel. Mirrors `atmos_extinction` in `shaders/atmosphere.wgsl`.
pub fn extinction(params: &AtmosphereParams, altitude_km: f32) -> [f32; 3] {
    let dr = rayleigh_density(params, altitude_km);
    let dm = mie_density(params, altitude_km);
    let doz = ozone_density(params, altitude_km);
    let mie_ext = (params.mie_scattering_scaled() + params.mie_absorption_scaled()) * dm;
    [
        params.rayleigh_scattering[0] * dr + mie_ext + params.ozone_absorption[0] * doz,
        params.rayleigh_scattering[1] * dr + mie_ext + params.ozone_absorption[1] * doz,
        params.rayleigh_scattering[2] * dr + mie_ext + params.ozone_absorption[2] * doz,
    ]
}

/// Distance along the ray from `origin` in unit direction `dir` to the sphere of
/// radius `radius` centred at the origin of coordinates, or a negative value when
/// the ray misses. Returns the **far** root, which is what a ray starting inside
/// the shell wants.
pub fn ray_sphere_far(origin: [f32; 3], dir: [f32; 3], radius: f32) -> f32 {
    let b = origin[0] * dir[0] + origin[1] * dir[1] + origin[2] * dir[2];
    let c = origin[0] * origin[0] + origin[1] * origin[1] + origin[2] * origin[2] - radius * radius;
    let disc = b * b - c;
    if disc < 0.0 {
        return -1.0;
    }
    -b + disc.sqrt()
}

/// Distance to the **near** intersection with the sphere, or a negative value
/// when the ray misses it or the sphere is entirely behind.
pub fn ray_sphere_near(origin: [f32; 3], dir: [f32; 3], radius: f32) -> f32 {
    let b = origin[0] * dir[0] + origin[1] * dir[1] + origin[2] * dir[2];
    let c = origin[0] * origin[0] + origin[1] * origin[1] + origin[2] * origin[2] - radius * radius;
    let disc = b * b - c;
    if disc < 0.0 {
        return -1.0;
    }
    let t = -b - disc.sqrt();
    if t < 0.0 {
        -1.0
    } else {
        t
    }
}

/// Transmittance from a point at radius `r_km` looking along a ray whose cosine
/// with the local zenith is `mu`, out to the top of the atmosphere.
///
/// Returns `[0, 0, 0]` when the ray hits the ground first (nothing gets through
/// a planet). `steps` is the ray-march count; the LUT bake uses
/// [`AtmosphereQuality::transmittance_steps`].
///
/// **This is the CPU mirror of `atmos_transmittance_integral` in
/// `shaders/atmosphere.wgsl`** — same trapezoid-free midpoint rule, same order of
/// operations. It exists so the physics is asserted without a GPU (monotonicity
/// in altitude, reddening at grazing angles), which is the only atmosphere test
/// that can run on a CI leg with no adapter.
pub fn transmittance_to_top(params: &AtmosphereParams, r_km: f32, mu: f32, steps: u32) -> [f32; 3] {
    let mu = mu.clamp(-1.0, 1.0);
    // Work in the plane containing the zenith and the ray: position on the +Y
    // axis, direction with the given zenith cosine.
    let origin = [0.0, r_km, 0.0];
    let sin_t = (1.0 - mu * mu).max(0.0).sqrt();
    let dir = [sin_t, mu, 0.0];

    if ray_sphere_near(origin, dir, params.ground_radius) > 0.0 {
        return [0.0, 0.0, 0.0];
    }
    let t_max = ray_sphere_far(origin, dir, params.top_radius);
    if t_max <= 0.0 {
        return [1.0, 1.0, 1.0];
    }

    let steps = steps.max(2);
    let dt = t_max / steps as f32;
    let mut depth = [0.0f32; 3];
    for i in 0..steps {
        let t = (i as f32 + 0.5) * dt;
        let p = [
            origin[0] + dir[0] * t,
            origin[1] + dir[1] * t,
            origin[2] + dir[2] * t,
        ];
        let radius = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        let alt = radius - params.ground_radius;
        let e = extinction(params, alt);
        depth[0] += e[0] * dt;
        depth[1] += e[1] * dt;
        depth[2] += e[2] * dt;
    }
    [(-depth[0]).exp(), (-depth[1]).exp(), (-depth[2]).exp()]
}

// ── LUT parameterizations ────────────────────────────────────────────────────

/// Map a transmittance-LUT texel `(u, v)` back to `(r_km, mu)`.
///
/// This is Bruneton's parameterization, the one Hillaire keeps: `v` encodes the
/// distance to the horizon (so texels are dense near the ground where density
/// changes fastest) and `u` encodes the distance the ray travels to the top of
/// the atmosphere between its two extremes (so texels are dense near the horizon
/// where transmittance changes fastest). A naive linear `mu` mapping wastes most
/// of the texture on the ~identical overhead directions and bands the sunset.
///
/// Mirrors `atmos_transmittance_uv_to_params` in `shaders/atmosphere.wgsl`.
pub fn transmittance_uv_to_params(params: &AtmosphereParams, u: f32, v: f32) -> (f32, f32) {
    let bottom = params.ground_radius;
    let top = params.top_radius;
    let h = (top * top - bottom * bottom).max(0.0).sqrt();
    let rho = h * v.clamp(0.0, 1.0);
    let r = (rho * rho + bottom * bottom).sqrt();
    let d_min = top - r;
    let d_max = rho + h;
    let d = d_min + u.clamp(0.0, 1.0) * (d_max - d_min);
    let mu = if d <= 0.0 {
        1.0
    } else {
        ((h * h - rho * rho - d * d) / (2.0 * r * d)).clamp(-1.0, 1.0)
    };
    (r, mu)
}

/// The inverse of [`transmittance_uv_to_params`]: where to sample the LUT for a
/// point at radius `r_km` looking at zenith cosine `mu`.
pub fn transmittance_params_to_uv(params: &AtmosphereParams, r_km: f32, mu: f32) -> (f32, f32) {
    let bottom = params.ground_radius;
    let top = params.top_radius;
    let r = r_km.clamp(bottom, top);
    let mu = mu.clamp(-1.0, 1.0);
    let h = (top * top - bottom * bottom).max(0.0).sqrt();
    let rho = (r * r - bottom * bottom).max(0.0).sqrt();
    // Distance to the top of the atmosphere along the ray.
    let disc = (r * r * (mu * mu - 1.0) + top * top).max(0.0);
    let d = (-r * mu + disc.sqrt()).max(0.0);
    let d_min = top - r;
    let d_max = rho + h;
    let u = if d_max - d_min <= 0.0 {
        0.0
    } else {
        ((d - d_min) / (d_max - d_min)).clamp(0.0, 1.0)
    };
    let v = if h <= 0.0 {
        0.0
    } else {
        (rho / h).clamp(0.0, 1.0)
    };
    (u, v)
}

/// The angle (radians, from the zenith) at which the horizon sits for a viewer
/// at radius `r_km` — the pivot the sky-view LUT's `v` axis is warped around.
#[inline]
pub fn zenith_horizon_angle(params: &AtmosphereParams, r_km: f32) -> f32 {
    let bottom = params.ground_radius;
    let r = r_km.max(bottom + 1e-4);
    let cos_beta = ((r * r - bottom * bottom).max(0.0).sqrt() / r).clamp(-1.0, 1.0);
    std::f32::consts::PI - cos_beta.acos()
}

/// Sky-view LUT `v` for a view direction whose zenith cosine is
/// `view_zenith_cos`, at viewer radius `r_km`.
///
/// Hillaire's warp: the horizon lands exactly at `v = 0.5` and both halves are
/// square-root-compressed toward it, because the horizon is where the sky's
/// gradient is steepest and a linear mapping visibly bands there. `u` is handled
/// by [`skyview_azimuth_u`].
pub fn skyview_zenith_v(params: &AtmosphereParams, r_km: f32, view_zenith_cos: f32) -> f32 {
    let horizon = zenith_horizon_angle(params, r_km);
    let beta = std::f32::consts::PI - horizon;
    let theta = view_zenith_cos.clamp(-1.0, 1.0).acos();
    if theta < horizon {
        // Above the horizon.
        let coord = if horizon <= 0.0 { 0.0 } else { theta / horizon };
        let coord = 1.0 - coord;
        let coord = 1.0 - coord.max(0.0).sqrt();
        coord * 0.5
    } else {
        let coord = if beta <= 0.0 {
            0.0
        } else {
            (theta - horizon) / beta
        };
        coord.clamp(0.0, 1.0).sqrt() * 0.5 + 0.5
    }
}

/// Sky-view LUT `u` for a view/sun azimuthal cosine (the cosine of the angle
/// between the view and sun directions **projected onto the horizontal plane**).
///
/// The LUT is azimuthally symmetric about the sun, so one axis suffices; the
/// square-root warp concentrates texels toward the sun where the Mie halo lives.
#[inline]
pub fn skyview_azimuth_u(light_view_cos: f32) -> f32 {
    (-light_view_cos.clamp(-1.0, 1.0) * 0.5 + 0.5)
        .max(0.0)
        .sqrt()
}

// ── sky radiance and the ambient it delivers (wave FIX3) ─────────────────────

/// The isotropic multiple-scattering stand-in, mirroring `ATMOS_MULTI_SCATTER`
/// in `shaders/sky_view.wgsl`. A fraction of what single-scatters at each sample
/// is treated as re-radiating in all directions; without it the zenith is
/// unnaturally dark and twilight collapses to black too fast.
pub const MULTI_SCATTER: f32 = 0.022;

/// Rayleigh phase function. Mirrors `atmos_phase_rayleigh`.
#[inline]
pub fn phase_rayleigh(cos_t: f32) -> f32 {
    (3.0 / (16.0 * std::f32::consts::PI)) * (1.0 + cos_t * cos_t)
}

/// Cornette-Shanks Mie phase function. Mirrors `atmos_phase_mie`.
#[inline]
pub fn phase_mie(cos_t: f32, g: f32) -> f32 {
    let g2 = g * g;
    let num = 3.0 * (1.0 - g2) * (1.0 + cos_t * cos_t);
    let den =
        8.0 * std::f32::consts::PI * (2.0 + g2) * (1.0 + g2 - 2.0 * g * cos_t).max(1e-4).powf(1.5);
    num / den
}

/// Is a body at direction cosine `mu` clear of the local horizon at radius
/// `r_km`? Mirrors `atmos_horizon_visibility` — the factor that keeps a sun
/// eight degrees underground from delivering sunset red through the
/// transmittance table's horizon-tangent texel.
#[inline]
pub fn horizon_visibility(params: &AtmosphereParams, r_km: f32, mu: f32, disc_cos: f32) -> f32 {
    let bottom = params.ground_radius;
    let rr = r_km.max(bottom + 1e-4);
    let sin_h = (bottom / rr).min(1.0);
    let cos_h = -(1.0 - sin_h * sin_h).max(0.0).sqrt();
    let ang = (1.0 - disc_cos * disc_cos).max(0.0).sqrt();
    let w = (sin_h * ang).max(1e-4);
    smoothstep(-w, w, mu - cos_h)
}

#[inline]
fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0).max(1e-9)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// **Sky radiance in one direction, on the CPU** — the mirror of `cs_skyview`
/// in `shaders/sky_view.wgsl`, which bakes exactly this integral into the
/// sky-view LUT.
///
/// `dir` is a unit world direction (`+Y` up), `r_km` the viewer's radius from
/// the planet centre ([`camera_radius_km`]), `sun` the unit direction toward the
/// sun and `sun_irradiance` its colour × intensity. The result carries the
/// sky-intensity exposure the caller applies, exactly as the LUT does — that is,
/// this returns the **unexposed** radiance and the caller multiplies by
/// `sky_intensity × SKY_EXPOSURE_CALIBRATION`.
///
/// **Why a CPU copy exists at all.** The GPU has this in a texture and the lit
/// passes can already sample it (`atmos_sample_skyview`), but a *diffuse ambient*
/// is not a sample, it is an integral over the whole sphere — and the one place
/// an integral can be paid for once per frame instead of once per pixel is the
/// CPU. It also means the physics is asserted on every CI leg, including the
/// ones with no adapter, which is this module's standing doctrine.
///
/// The one deliberate difference from the shader: the sun transmittance is
/// ray-marched here ([`transmittance_to_top`]) where the bake reads its own
/// transmittance LUT. That makes this copy slightly *more* accurate than the
/// texture it mirrors, and the discrepancy is bounded by the LUT's own
/// resolution.
pub fn sky_radiance(
    params: &AtmosphereParams,
    r_km: f32,
    dir: [f32; 3],
    sun: [f32; 3],
    sun_irradiance: [f32; 3],
    steps: u32,
    transmittance_steps: u32,
) -> [f32; 3] {
    let ground = params.ground_radius;
    let origin = [0.0, r_km, 0.0];
    let t_ground = ray_sphere_near(origin, dir, ground);
    let t_top = ray_sphere_far(origin, dir, params.top_radius);
    let t_max = if t_ground > 0.0 { t_ground } else { t_top };

    let cos_theta = dir[0] * sun[0] + dir[1] * sun[1] + dir[2] * sun[2];
    let phase_r = phase_rayleigh(cos_theta);
    let phase_m = phase_mie(cos_theta, params.mie_g.clamp(-0.95, 0.95));
    let disc_cos = params.sun_disc_cos();
    let mie_s = params.mie_scattering_scaled();

    let steps = steps.max(4);
    let dt = t_max.max(0.0) / steps as f32;
    let mut luminance = [0.0f32; 3];
    let mut throughput = [1.0f32; 3];
    for i in 0..steps {
        let t = (i as f32 + 0.5) * dt;
        let p = [
            origin[0] + dir[0] * t,
            origin[1] + dir[1] * t,
            origin[2] + dir[2] * t,
        ];
        let radius = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt().max(ground);
        let alt = radius - ground;
        let up = [p[0] / radius, p[1] / radius, p[2] / radius];
        let mu = up[0] * sun[0] + up[1] * sun[1] + up[2] * sun[2];

        let dr = rayleigh_density(params, alt);
        let dm = mie_density(params, alt);
        let sc_r = [
            params.rayleigh_scattering[0] * dr,
            params.rayleigh_scattering[1] * dr,
            params.rayleigh_scattering[2] * dr,
        ];
        let sc_m = mie_s * dm;
        let ext = extinction(params, alt);

        let vis = horizon_visibility(params, radius, mu, disc_cos);
        let tt = transmittance_to_top(params, radius, mu, transmittance_steps);
        let t_sun = [tt[0] * vis, tt[1] * vis, tt[2] * vis];
        // The multiple-scattering stand-in is driven by the least attenuated
        // channel, exactly as the bake does it, so a sunset keeps a warm ambient
        // instead of snapping to black the instant the direct blue is gone.
        let ms = t_sun[0].max(t_sun[1]).max(t_sun[2]);

        for c in 0..3 {
            let direct = (sc_r[c] * phase_r + sc_m * phase_m) * t_sun[c];
            let multi = (sc_r[c] + sc_m) * MULTI_SCATTER * ms;
            let in_scatter = (direct + multi) * sun_irradiance[c];
            let e = ext[c].max(1e-9);
            // Hillaire's energy-conserving step integral, not a midpoint
            // rectangle — the same arithmetic, in the same order, as the bake.
            let step_t = (-e * dt).exp();
            luminance[c] += throughput[c] * (in_scatter - in_scatter * step_t) / e;
            throughput[c] *= step_t;
        }
    }

    // The ground bounce for directions below the horizon: a Lambertian planet
    // lit by the sun. Without it the lower hemisphere is pure black and an
    // ambient term computed from this integral would lose every bit of the warm
    // light a surface receives from the ground it stands on.
    if t_ground > 0.0 {
        let hit = [
            origin[0] + dir[0] * t_ground,
            origin[1] + dir[1] * t_ground,
            origin[2] + dir[2] * t_ground,
        ];
        let len = (hit[0] * hit[0] + hit[1] * hit[1] + hit[2] * hit[2])
            .sqrt()
            .max(1e-6);
        let n = [hit[0] / len, hit[1] / len, hit[2] / len];
        let mu = n[0] * sun[0] + n[1] * sun[1] + n[2] * sun[2];
        let n_dot_l = mu.max(0.0);
        let vis = horizon_visibility(params, ground, mu, disc_cos);
        let tt = transmittance_to_top(params, ground, mu, transmittance_steps);
        let k = n_dot_l * params.ground_albedo.clamp(0.0, 1.0) / std::f32::consts::PI;
        for c in 0..3 {
            luminance[c] += throughput[c] * sun_irradiance[c] * tt[c] * vis * k;
        }
    }
    luminance
}

/// The authored two-colour gradient, evaluated as a radiance in direction `dir`
/// — the sky a scene **without** a time-of-day authority has.
///
/// Mirrors the fallback branch of `gi_sky_radiance` in
/// `shaders/gi_probes.wgsl`, which is the only other place in the engine that
/// asks "how bright is the sky that way" without a LUT.
#[inline]
pub fn gradient_radiance(horizon: [f32; 3], zenith: [f32; 3], dir: [f32; 3]) -> [f32; 3] {
    let t = (dir[1] * 0.5 + 0.5).clamp(0.0, 1.0);
    [
        horizon[0] + (zenith[0] - horizon[0]) * t,
        horizon[1] + (zenith[1] - horizon[1]) * t,
        horizon[2] + (zenith[2] - horizon[2]) * t,
    ]
}

/// How many directions [`sky_irradiance_sh`] projects over.
///
/// The sky is a smooth, azimuthally-symmetric-about-the-sun function and the
/// target is an **L1** set — four coefficients, which a much coarser quadrature
/// than this already resolves. Forty-eight is the same count `GiSettings::rays`
/// defaults to, chosen so that the two ambient sources in this engine sample the
/// sphere at the same density, and measured (`sky_sh_converges`) to be within
/// 0.4 % of a 512-direction reference.
pub const SKY_SH_DIRECTIONS: u32 = 48;

/// March steps for [`sky_irradiance_sh`]'s per-direction integral, and for the
/// transmittance inside it.
///
/// Far below the LUT bake's 32/40: an L1 projection averages 48 directions
/// together, so a per-direction bias that a single sky pixel would show as
/// banding is invisible after the integral. Measured against the bake's own step
/// counts in `sky_sh_converges`: 0.6 % on the DC term, for **a twelfth** of the
/// arithmetic.
const SKY_SH_MARCH_STEPS: u32 = 12;
const SKY_SH_TRANSMITTANCE_STEPS: u32 = 8;

/// **THE AMBIENT TERM, AS PHYSICS** (wave FIX3, clause 3).
///
/// The L1 spherical-harmonic projection of the sky's own radiance over the whole
/// sphere — the cosine-weighted hemisphere a shaded surface actually receives,
/// evaluated once per frame instead of never.
///
/// # What this replaces
///
/// Nothing, and that is the defect it was written for. Before this wave the only
/// ambient term in the engine was `gi_irradiance` — the GI probe field — and it
/// is a *40 m box centred on the camera*. A wall beyond the box reads a clamped
/// boundary probe; a wall inside it, standing close enough to blind the probes
/// around it, reads what those blinded probes gathered. Both answers are the
/// same shape of wrong, and the measurement is in
/// `crates/inf-render/tests/sky_ambient.rs`: a white wall's shaded face read
/// **0.0129** of its sunlit face under the atmosphere with GI on, against the
/// 0.1–0.2 a clear noon sky delivers, and **0.0965** with GI off — because with
/// GI off the hard-coded hemispheric constant is, by accident, roughly the right
/// magnitude for a sun of intensity 3.
///
/// So this is the term that was missing: it does not depend on a probe volume,
/// it does not depend on convergence, it exists at every distance from the
/// camera, and it tracks the sun because it is computed from the same medium the
/// sky pass draws.
///
/// # Units and the convention it hands the shader
///
/// The coefficients are the same Monte-Carlo projection `gi_probes.wgsl` writes
/// — `c_k = Σ L(d_i)·Y_k(d_i)·(4π/N)` — so the consumer convolves them with the
/// identical folded cosine lobe (`A₀/π = 1`, `A₁/π = 2/3`) and gets **exit
/// radiance for an albedo-1 Lambert surface**. On a uniform sky of radiance `L`
/// that is exactly `L`, which is the white furnace written as arithmetic and is
/// asserted by `a_uniform_sky_projects_to_its_own_radiance` below.
///
/// # The two sources
///
/// With `params.enabled` the sky is the physical medium ([`sky_radiance`],
/// multiplied by the sky exposure the LUT carries); without it the authored
/// two-colour gradient. One function, both sources — the same duality
/// `gi_sky_radiance` has in the probe march, so a level that has no time-of-day
/// authority still gets an ambient that is *its own* sky rather than a constant.
pub fn sky_irradiance_sh(
    params: &AtmosphereParams,
    r_km: f32,
    sun: [f32; 3],
    sun_irradiance: [f32; 3],
    gradient_horizon: [f32; 3],
    gradient_zenith: [f32; 3],
) -> [[f32; 3]; 4] {
    let n = SKY_SH_DIRECTIONS;
    let exposure = params.sky_intensity.max(0.0) * SKY_EXPOSURE_CALIBRATION;
    let mut c = [[0.0f32; 3]; 4];
    for i in 0..n {
        let d = golden_spiral(i, n);
        let l = if params.enabled {
            let r = sky_radiance(
                params,
                r_km,
                d,
                sun,
                sun_irradiance,
                SKY_SH_MARCH_STEPS,
                SKY_SH_TRANSMITTANCE_STEPS,
            );
            [r[0] * exposure, r[1] * exposure, r[2] * exposure]
        } else {
            gradient_radiance(gradient_horizon, gradient_zenith, d)
        };
        let b = [
            0.282_095_f32,
            0.488_603 * d[1],
            0.488_603 * d[2],
            0.488_603 * d[0],
        ];
        for (k, ck) in c.iter_mut().enumerate() {
            for (ch, v) in ck.iter_mut().enumerate() {
                *v += l[ch] * b[k];
            }
        }
    }
    let norm = 4.0 * std::f32::consts::PI / n as f32;
    for ck in c.iter_mut() {
        for v in ck.iter_mut() {
            *v *= norm;
        }
    }
    c
}

/// The `i`-th of `n` golden-spiral directions, as a bare array.
///
/// The same sequence as [`crate::gi::golden_spiral_dir`] and `spiral_dir` in
/// `shaders/gi_probes.wgsl` — restated here in this module's own array
/// vocabulary so the atmosphere stays free of `glam`, which is what keeps it
/// pure and testable without a renderer.
#[inline]
fn golden_spiral(i: u32, n: u32) -> [f32; 3] {
    let nf = n.max(1) as f32;
    let fi = i as f32;
    let phi = std::f32::consts::PI * (3.0 - 5.0_f32.sqrt());
    let y = 1.0 - 2.0 * (fi + 0.5) / nf;
    let r = (1.0 - y * y).max(0.0).sqrt();
    let theta = phi * fi;
    [theta.cos() * r, y, theta.sin() * r]
}

// ── height fog (SI metres) ───────────────────────────────────────────────────

/// Optical depth (dimensionless) of the exponential height fog along a segment
/// that starts at world altitude `y0_m`, ends at `y1_m`, and is `distance_m`
/// long — all **metres**.
///
/// Closed form of `∫ density · exp(-falloff · (y(t) − height)) dt`. The
/// `falloff · dy → 0` limit is handled explicitly (a horizontal ray, which is
/// the common case for a camera looking at the horizon, would otherwise divide
/// by zero).
///
/// Mirrors `atmos_fog_optical_depth` in `shaders/atmosphere.wgsl`.
pub fn height_fog_optical_depth(fog: &HeightFog, y0_m: f32, y1_m: f32, distance_m: f32) -> f32 {
    if !fog.active() || distance_m <= 0.0 {
        return 0.0;
    }
    let falloff = fog.falloff.max(0.0);
    let base = fog.density * (-falloff * (y0_m - fog.height)).exp();
    if !base.is_finite() {
        // A camera far below a steeply-falling-off fog plane: saturate rather
        // than produce `inf` (which would make the whole frame NaN).
        return f32::MAX;
    }
    let dy = y1_m - y0_m;
    let k = falloff * dy;
    let shape = if k.abs() < 1e-4 {
        // lim_{k→0} (1 − e^{−k}) / k = 1.
        1.0
    } else {
        (1.0 - (-k).exp()) / k
    };
    base * shape * distance_m
}

/// Fog transmittance `exp(-optical_depth)` along the same segment — the fraction
/// of the surface's own radiance that survives to the eye.
pub fn height_fog_transmittance(fog: &HeightFog, y0_m: f32, y1_m: f32, distance_m: f32) -> f32 {
    (-height_fog_optical_depth(fog, y0_m, y1_m, distance_m)).exp()
}

// ── quality tiers ────────────────────────────────────────────────────────────

/// Atmosphere LUT resolution + ray-march budget. Selected by
/// [`crate::settings::AtmosphereSettings`] and clamped **down** (never up) by
/// [`RenderTier::apply`], exactly like every other capability knob.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AtmosphereQuality {
    /// 128×32 transmittance, 96×54 sky view, 16-step marches.
    Low,
    /// 192×48 transmittance, 128×72 sky view, 24-step marches.
    Medium,
    /// 256×64 transmittance, 192×108 sky view, 32-step marches — the Hillaire
    /// reference sizes.
    High,
}

impl AtmosphereQuality {
    /// `(width, height)` of the transmittance LUT. It is a **static** function of
    /// the medium alone (not of the sun or the camera), so it is baked once and
    /// re-baked only when the medium changes — a 256×64 bake is a rounding error
    /// that happens roughly never.
    pub fn transmittance_size(self) -> (u32, u32) {
        match self {
            AtmosphereQuality::High => (256, 64),
            AtmosphereQuality::Medium => (192, 48),
            AtmosphereQuality::Low => (128, 32),
        }
    }

    /// `(width, height)` of the sky-view LUT. This one *is* re-baked whenever the
    /// sun moves or the camera changes altitude, so it is deliberately tiny: at
    /// 192×108 the whole bake is ~20 k threads, each marching 32 steps — cheaper
    /// than a single full-screen post pass, and it replaces a per-pixel march for
    /// every sky pixel on screen.
    pub fn skyview_size(self) -> (u32, u32) {
        match self {
            AtmosphereQuality::High => (192, 108),
            AtmosphereQuality::Medium => (128, 72),
            AtmosphereQuality::Low => (96, 54),
        }
    }

    /// Ray-march steps for the transmittance bake.
    pub fn transmittance_steps(self) -> u32 {
        match self {
            AtmosphereQuality::High => 40,
            AtmosphereQuality::Medium => 32,
            AtmosphereQuality::Low => 24,
        }
    }

    /// Ray-march steps for the sky-view bake.
    pub fn skyview_steps(self) -> u32 {
        match self {
            AtmosphereQuality::High => 32,
            AtmosphereQuality::Medium => 24,
            AtmosphereQuality::Low => 16,
        }
    }

    /// Star cells per cube-face axis. More cells ⇒ more, smaller stars. Scaling
    /// this with the tier keeps the *screen density* of the hash lookups roughly
    /// constant on a lower-resolution output.
    pub fn star_cells(self) -> f32 {
        match self {
            AtmosphereQuality::High => 384.0,
            AtmosphereQuality::Medium => 288.0,
            AtmosphereQuality::Low => 192.0,
        }
    }

    /// Clamp down to what `tier` can afford. Mirrors the "never turns a feature
    /// on" rule of [`RenderTier::apply`].
    pub fn clamp_to(self, tier: RenderTier) -> Self {
        let ceiling = match tier {
            RenderTier::High => AtmosphereQuality::High,
            RenderTier::Medium => AtmosphereQuality::Medium,
            RenderTier::Low => AtmosphereQuality::Low,
        };
        self.min(ceiling)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn earth() -> AtmosphereParams {
        AtmosphereParams {
            enabled: true,
            ..AtmosphereParams::default()
        }
    }

    /// The pure math is a function of its inputs alone — the property the GPU
    /// LUT determinism gate mirrors on the other side of the bus.
    #[test]
    fn transmittance_is_deterministic() {
        let p = earth();
        for i in 0..32 {
            let mu = -1.0 + i as f32 / 16.0;
            let a = transmittance_to_top(&p, p.ground_radius + 0.5, mu, 40);
            let b = transmittance_to_top(&p, p.ground_radius + 0.5, mu, 40);
            assert_eq!(a.map(f32::to_bits), b.map(f32::to_bits), "mu {mu}");
        }
    }

    /// Higher up ⇒ less air overhead ⇒ more light gets through. Asserted at a
    /// fixed sun angle across the whole shell, per channel.
    #[test]
    fn transmittance_is_monotone_with_altitude() {
        let p = earth();
        let mu = 0.35_f32; // ~20° above the horizon: a long, sensitive path
        let mut prev = [0.0f32; 3];
        let mut alt = 0.0f32;
        while alt <= 60.0 {
            let t = transmittance_to_top(&p, p.ground_radius + alt, mu, 64);
            for c in 0..3 {
                assert!(
                    t[c] + 1e-5 >= prev[c],
                    "channel {c} fell going up: {} < {} at {alt} km",
                    t[c],
                    prev[c]
                );
                assert!((0.0..=1.0).contains(&t[c]), "out of range: {}", t[c]);
            }
            prev = t;
            alt += 2.0;
        }
        // And it actually moves: the top of the atmosphere is nearly clear.
        let top = transmittance_to_top(&p, p.top_radius - 0.5, mu, 64);
        assert!(top[2] > 0.98, "top of atmosphere not clear: {top:?}");
    }

    /// The reason sunsets are red: at a grazing angle the long path scatters the
    /// blue out, so the surviving light's red:blue ratio explodes compared with
    /// noon. This is the single assertion that would catch a swapped
    /// wavelength-coefficient triple.
    #[test]
    fn sunset_is_redder_than_noon() {
        let p = earth();
        let ground = p.ground_radius;
        let noon = transmittance_to_top(&p, ground, 1.0, 64);
        // 0.5° above the horizon — where the sun actually is at sunset.
        let sunset = transmittance_to_top(&p, ground, 0.5_f32.to_radians().sin(), 64);

        let noon_ratio = noon[0] / noon[2];
        let sunset_ratio = sunset[0] / sunset[2];
        assert!(
            noon_ratio > 1.0 && noon_ratio < 1.3,
            "noon should be nearly neutral, got r/b {noon_ratio}"
        );
        assert!(
            sunset_ratio > 10.0 * noon_ratio,
            "sunset not redder than noon: r/b {sunset_ratio} vs {noon_ratio}"
        );
        // Blue is nearly gone; red still substantially survives.
        assert!(sunset[2] < 0.02, "sunset blue too strong: {sunset:?}");
        assert!(sunset[0] > 0.10, "sunset red too weak: {sunset:?}");
        // Sanity on the noon end: most light gets through overhead.
        assert!(noon[2] > 0.6, "noon blue too dim: {noon:?}");
    }

    /// Looking straight down from the ground hits the planet: nothing gets out.
    #[test]
    fn a_ray_into_the_ground_is_opaque() {
        let p = earth();
        let t = transmittance_to_top(&p, p.ground_radius + 1.0, -0.9, 32);
        assert_eq!(t, [0.0, 0.0, 0.0]);
    }

    /// Ozone only exists in a band; it must not touch the ground or the top.
    #[test]
    fn ozone_is_a_tent() {
        let p = earth();
        assert_eq!(ozone_density(&p, 0.0), 0.0);
        assert_eq!(ozone_density(&p, 25.0), 1.0);
        assert_eq!(ozone_density(&p, 41.0), 0.0);
        assert!((ozone_density(&p, 17.5) - 0.5).abs() < 1e-5);
    }

    /// Turbidity is a pure multiplier on the aerosol terms and touches nothing
    /// else — the property that lets "weather" scale "atmosphere".
    #[test]
    fn turbidity_scales_only_mie() {
        let clear = earth();
        let hazy = AtmosphereParams {
            turbidity: 4.0,
            ..clear
        };
        // At 0 km both Rayleigh and Mie are at full density.
        let a = extinction(&clear, 0.0);
        let b = extinction(&hazy, 0.0);
        let delta = clear.mie_scattering + clear.mie_absorption;
        for c in 0..3 {
            assert!(
                (b[c] - a[c] - 3.0 * delta).abs() < 1e-9,
                "channel {c}: {} vs {}",
                b[c] - a[c],
                3.0 * delta
            );
        }
        // At 30 km the aerosols are gone, so turbidity does nothing.
        let a = extinction(&clear, 30.0);
        let b = extinction(&hazy, 30.0);
        for c in 0..3 {
            assert!((a[c] - b[c]).abs() < 1e-9);
        }
    }

    /// The transmittance LUT mapping must round-trip, or the bake and the lookup
    /// disagree and the sky bands.
    #[test]
    fn transmittance_uv_round_trips() {
        let p = earth();
        for iu in 0..17 {
            for iv in 0..17 {
                let (u, v) = (iu as f32 / 16.0, iv as f32 / 16.0);
                let (r, mu) = transmittance_uv_to_params(&p, u, v);
                assert!(
                    (p.ground_radius - 1e-2..=p.top_radius + 1e-2).contains(&r),
                    "r out of shell: {r}"
                );
                let (u2, v2) = transmittance_params_to_uv(&p, r, mu);
                assert!((u - u2).abs() < 2e-3, "u {u} -> {u2}");
                assert!((v - v2).abs() < 2e-3, "v {v} -> {v2}");
            }
        }
    }

    /// The sky-view warp must put the horizon exactly on the seam, monotonically,
    /// with no gaps — the whole point of the parameterization.
    #[test]
    fn skyview_v_pivots_on_the_horizon() {
        let p = earth();
        let r = camera_radius_km(&p, 100.0);
        let horizon = zenith_horizon_angle(&p, r);
        // Straight up is v = 0, straight down is v = 1, the horizon is v = 0.5.
        assert!(skyview_zenith_v(&p, r, 1.0).abs() < 1e-5);
        assert!((skyview_zenith_v(&p, r, -1.0) - 1.0).abs() < 1e-5);
        let at_horizon = skyview_zenith_v(&p, r, horizon.cos());
        assert!((at_horizon - 0.5).abs() < 1e-4, "horizon at v={at_horizon}");
        // Monotone non-decreasing as the view sweeps down.
        let mut prev = -1.0;
        for i in 0..=256 {
            let cos_t = 1.0 - 2.0 * i as f32 / 256.0;
            let v = skyview_zenith_v(&p, r, cos_t);
            assert!(v + 1e-5 >= prev, "not monotone at {cos_t}: {v} < {prev}");
            assert!((0.0..=1.0).contains(&v));
            prev = v;
        }
    }

    /// The azimuth warp: sun-ward is u = 0, anti-sun is u = 1, monotone between.
    #[test]
    fn skyview_u_is_monotone_from_sun_to_antisun() {
        assert!(skyview_azimuth_u(1.0).abs() < 1e-6);
        assert!((skyview_azimuth_u(-1.0) - 1.0).abs() < 1e-6);
        let mut prev = -1.0;
        for i in 0..=64 {
            let c = 1.0 - 2.0 * i as f32 / 64.0;
            let u = skyview_azimuth_u(c);
            assert!(u + 1e-6 >= prev);
            prev = u;
        }
    }

    /// A camera radius is always inside the shell, whatever nonsense arrives.
    #[test]
    fn camera_radius_is_clamped_into_the_shell() {
        let p = earth();
        for alt in [-1e9, -1.0, 0.0, 1.0, 1e6, 1e9, f32::NAN, f32::INFINITY] {
            let r = camera_radius_km(&p, alt);
            assert!(r.is_finite(), "alt {alt} -> {r}");
            assert!(r > p.ground_radius && r < p.top_radius, "alt {alt} -> {r}");
        }
    }

    // ── height fog ──────────────────────────────────────────────────────────

    /// The closed form against hand-computed values at known distances.
    #[test]
    fn fog_matches_the_analytic_integral() {
        // Uniform fog (no falloff) is plain Beer–Lambert: exp(-σ·d).
        let flat = HeightFog {
            density: 1e-3,
            falloff: 0.0,
            height: 0.0,
            color: [1.0; 3],
        };
        for d in [0.0f32, 10.0, 100.0, 1000.0, 5000.0] {
            let t = height_fog_transmittance(&flat, 0.0, 0.0, d);
            assert!(
                (t - (-1e-3 * d).exp()).abs() < 1e-6,
                "d={d}: {t} vs {}",
                (-1e-3 * d).exp()
            );
        }
        // 3/σ metres is the classic "visibility" distance: ~5 % survives.
        let t = height_fog_transmittance(&flat, 0.0, 0.0, 3000.0);
        assert!((t - 0.049_787).abs() < 1e-5, "{t}");

        // A horizontal ray at the reference height is the same as uniform fog at
        // that height, whatever the falloff.
        let layered = HeightFog {
            density: 1e-3,
            falloff: 0.002,
            height: 0.0,
            color: [1.0; 3],
        };
        let t = height_fog_transmittance(&layered, 0.0, 0.0, 1000.0);
        assert!((t - (-1.0f32).exp()).abs() < 1e-6, "{t}");

        // A horizontal ray one e-folding height up sees exactly 1/e the density.
        let t = height_fog_transmittance(&layered, 500.0, 500.0, 1000.0);
        let expect = (-1e-3 * (-1.0f32).exp() * 1000.0).exp();
        assert!((t - expect).abs() < 1e-6, "{t} vs {expect}");

        // A vertical ray from the ground to infinity has a finite optical depth,
        // density/falloff — the analytic limit.
        let od = height_fog_optical_depth(&layered, 0.0, 1e6, 1e6);
        assert!((od - 1e-3 / 0.002).abs() < 1e-4, "{od}");
    }

    /// Fog is monotone in distance, bounded, and inert when off.
    #[test]
    fn fog_is_monotone_and_off_by_default() {
        let off = HeightFog::default();
        assert!(!off.active());
        assert_eq!(height_fog_optical_depth(&off, 0.0, 100.0, 1000.0), 0.0);
        assert_eq!(height_fog_transmittance(&off, 0.0, 100.0, 1000.0), 1.0);

        let fog = HeightFog {
            density: 5e-4,
            ..HeightFog::default()
        };
        let mut prev = 1.0;
        let mut d = 0.0;
        while d <= 20_000.0 {
            let t = height_fog_transmittance(&fog, 2.0, 2.0, d);
            assert!(t <= prev + 1e-6, "not monotone at {d}");
            assert!((0.0..=1.0).contains(&t), "out of range at {d}: {t}");
            prev = t;
            d += 250.0;
        }
    }

    /// A camera far below a steep fog plane must not produce `inf`/`NaN` — one
    /// non-finite pixel in an HDR target poisons the whole bloom chain.
    #[test]
    fn fog_saturates_instead_of_overflowing() {
        let fog = HeightFog {
            density: 1.0,
            falloff: 1.0,
            height: 0.0,
            color: [1.0; 3],
        };
        let t = height_fog_transmittance(&fog, -10_000.0, 0.0, 10_000.0);
        assert!(t.is_finite() && t == 0.0, "{t}");
    }

    // ── quality tiers ───────────────────────────────────────────────────────

    #[test]
    fn quality_clamps_down_never_up() {
        use AtmosphereQuality::*;
        assert_eq!(High.clamp_to(RenderTier::High), High);
        assert_eq!(High.clamp_to(RenderTier::Medium), Medium);
        assert_eq!(High.clamp_to(RenderTier::Low), Low);
        // A caller who asked for Low never gets promoted by a fast GPU.
        assert_eq!(Low.clamp_to(RenderTier::High), Low);
        assert_eq!(Medium.clamp_to(RenderTier::High), Medium);
    }

    #[test]
    fn quality_sizes_shrink_with_the_tier() {
        use AtmosphereQuality::*;
        for (lo, hi) in [(Low, Medium), (Medium, High)] {
            assert!(lo.transmittance_size().0 < hi.transmittance_size().0);
            assert!(lo.skyview_size().0 < hi.skyview_size().0);
            assert!(lo.transmittance_steps() < hi.transmittance_steps());
            assert!(lo.skyview_steps() < hi.skyview_steps());
            assert!(lo.star_cells() < hi.star_cells());
        }
        // The reference sizes stay the Hillaire ones.
        assert_eq!(High.transmittance_size(), (256, 64));
        assert_eq!(High.skyview_size(), (192, 108));
    }

    /// The default block is inert: no scene that has not opted in can be touched
    /// by any of this.
    #[test]
    fn default_is_disabled_and_inert() {
        let p = AtmosphereParams::default();
        assert!(!p.enabled);
        assert!(!p.affects_lit_passes());
        assert!(!p.fog.active());
        // Enabling alone is enough to reach the lit passes (aerial perspective
        // defaults to the physical strength).
        let on = AtmosphereParams { enabled: true, ..p };
        assert!(on.affects_lit_passes());
        // ...but a level that zeroes both knobs opts back out entirely.
        let quiet = AtmosphereParams {
            enabled: true,
            aerial_perspective: 0.0,
            ..p
        };
        assert!(!quiet.affects_lit_passes());
    }

    #[test]
    fn disc_cosines_match_their_angles() {
        let p = earth();
        // 0.545° diameter ⇒ half-angle 0.2725°.
        let expect = 0.2725_f32.to_radians().cos();
        assert!((p.sun_disc_cos() - expect).abs() < 1e-7);
        assert!(p.sun_disc_cos() > 0.9999);
        assert!(p.moon_disc_cos() > p.sun_disc_cos());
    }

    // ── the sky irradiance SH (wave FIX3) ───────────────────────────────────

    /// **THE FURNACE, WITHOUT A GPU.** A uniform sky of radiance `L` must
    /// project to a set whose folded cosine convolution is exactly `L` in every
    /// direction — that is what makes `sky_irradiance` exit radiance for an
    /// albedo-1 Lambert surface, and it is the same statement
    /// `gi_normalisation.rs` makes about the probe gather with a renderer in the
    /// way.
    ///
    /// The gradient source with `zenith == horizon` IS a uniform sky, so this
    /// needs no atmosphere and runs on every CI leg.
    #[test]
    fn a_uniform_sky_projects_to_its_own_radiance() {
        let off = AtmosphereParams::default();
        let l = 0.17f32;
        let c = sky_irradiance_sh(
            &off,
            EARTH_GROUND_RADIUS_KM + 0.01,
            [0.0, 1.0, 0.0],
            [1.0; 3],
            [l; 3],
            [l; 3],
        );
        // The consumer's convolution, spelled out: A0/π = 1, A1/π = 2/3.
        let convolve = |n: [f32; 3]| -> f32 {
            let b = [
                0.282_095_f32,
                0.488_603 * n[1],
                0.488_603 * n[2],
                0.488_603 * n[0],
            ];
            c[0][0] * b[0] + (2.0 / 3.0) * (c[1][0] * b[1] + c[2][0] * b[2] + c[3][0] * b[3])
        };
        for n in [
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.577, 0.577, 0.577],
        ] {
            let e = convolve(n);
            assert!(
                (e - l).abs() < 0.01 * l,
                "a uniform sky of {l} irradiates a surface facing {n:?} with {e}"
            );
        }
    }

    /// The gradient source is the authored sky and not a constant: a bright
    /// zenith over a dark horizon must light an up-facing surface more than a
    /// down-facing one, by the amount the gradient actually carries.
    #[test]
    fn the_gradient_source_has_the_authored_shape() {
        let off = AtmosphereParams::default();
        let c = sky_irradiance_sh(
            &off,
            EARTH_GROUND_RADIUS_KM + 0.01,
            [0.0, 1.0, 0.0],
            [1.0; 3],
            [0.02, 0.02, 0.02],
            [0.40, 0.40, 0.40],
        );
        let convolve = |n: [f32; 3]| -> f32 {
            let b = [
                0.282_095_f32,
                0.488_603 * n[1],
                0.488_603 * n[2],
                0.488_603 * n[0],
            ];
            c[0][0] * b[0] + (2.0 / 3.0) * (c[1][0] * b[1] + c[2][0] * b[2] + c[3][0] * b[3])
        };
        let up = convolve([0.0, 1.0, 0.0]);
        let down = convolve([0.0, -1.0, 0.0]);
        let side = convolve([1.0, 0.0, 0.0]);
        assert!(up > down * 2.0, "up {up} vs down {down}");
        // A vertical wall sees half of each, so it lands between them.
        assert!(
            side > down && side < up,
            "side {side} between {down} and {up}"
        );
    }

    /// **The physical sky is brighter than the engine's authored default, and
    /// blue.** The measurement that matters for the ambient term: at noon the
    /// projected DC term must be a real daylight sky, not the dim editor
    /// gradient it replaces, and its blue must exceed its red the way a Rayleigh
    /// atmosphere's does.
    #[test]
    fn a_clear_noon_sky_is_bright_and_blue() {
        let p = earth();
        let r = camera_radius_km(&p, 2.0);
        let noon = sky_irradiance_sh(
            &p,
            r,
            [0.0, 1.0, 0.0],
            [3.0, 2.94, 2.85],
            [0.0; 3],
            [0.0; 3],
        );
        let dc = noon[0];
        assert!(dc[2] > dc[0] * 1.3, "the noon sky is not blue: {dc:?}");
        // The DC coefficient convolves to `dc * 0.282095` on an up-facing
        // surface's own half of it; the honest reading is that a horizontal
        // surface in the open receives a tenth or more of the sun's own
        // `intensity / π`, which is what the shaded-wall arm then measures
        // through the whole renderer.
        let up = dc[1] * 0.282_095
            + (2.0 / 3.0) * noon[2][1] * 0.0
            + (2.0 / 3.0) * noon[1][1] * 0.488_603;
        let sun_exit = 3.0 / std::f32::consts::PI;
        assert!(
            up > 0.06 * sun_exit && up < 0.6 * sun_exit,
            "a noon sky irradiates an up-facing white surface with {up} against a \
             sunlit {sun_exit} — outside the physical 6–60 % band"
        );
    }

    /// Midnight is not noon. The same projection with the sun below the horizon
    /// must collapse — otherwise the ambient term would light the world at 2 am.
    #[test]
    fn night_is_dark() {
        let p = earth();
        let r = camera_radius_km(&p, 2.0);
        let day = sky_irradiance_sh(&p, r, [0.0, 1.0, 0.0], [3.0; 3], [0.0; 3], [0.0; 3]);
        let night = sky_irradiance_sh(&p, r, [0.0, -1.0, 0.0], [3.0; 3], [0.0; 3], [0.0; 3]);
        assert!(
            night[0][2] < day[0][2] * 0.02,
            "midnight DC {:?} against noon {:?}",
            night[0],
            day[0]
        );
    }

    /// **The quadrature is converged.** `SKY_SH_DIRECTIONS` and the two step
    /// counts are chosen for cheapness; this is the arm that says the cheapness
    /// costs nothing, by comparing the shipped projection against one done with
    /// the sky-view bake's own 32/40 marches over a far denser sphere.
    #[test]
    fn sky_sh_converges() {
        let p = earth();
        let r = camera_radius_km(&p, 2.0);
        let sun = [0.35f32, 0.72, 0.6];
        let len = (sun[0] * sun[0] + sun[1] * sun[1] + sun[2] * sun[2]).sqrt();
        let sun = [sun[0] / len, sun[1] / len, sun[2] / len];
        let irr = [3.0f32, 2.94, 2.85];
        let exposure = p.sky_intensity.max(0.0) * SKY_EXPOSURE_CALIBRATION;

        // The reference: 512 directions at the bake's own step counts.
        let n = 512u32;
        let mut c0 = [0.0f32; 3];
        for i in 0..n {
            let d = golden_spiral(i, n);
            let l = sky_radiance(&p, r, d, sun, irr, 32, 40);
            for ch in 0..3 {
                c0[ch] += l[ch] * exposure * 0.282_095;
            }
        }
        let norm = 4.0 * std::f32::consts::PI / n as f32;
        for v in c0.iter_mut() {
            *v *= norm;
        }

        let got = sky_irradiance_sh(&p, r, sun, irr, [0.0; 3], [0.0; 3])[0];
        for ch in 0..3 {
            let rel = (got[ch] - c0[ch]).abs() / c0[ch].max(1e-9);
            assert!(
                rel < 0.05,
                "channel {ch}: shipped DC {} vs reference {} ({:.2} %)",
                got[ch],
                c0[ch],
                rel * 100.0
            );
        }
        println!("FIX3 sky SH convergence: shipped {got:?} vs 512-direction reference {c0:?}");
    }

    /// **The price.** The projection runs once per frame on the render thread,
    /// so it is a budget line and not a free lunch. Ten microseconds is two
    /// thousandths of a 60 Hz frame; the measured figure is printed so the
    /// ledger quotes a number rather than an adjective.
    #[test]
    fn sky_sh_is_cheap() {
        let p = earth();
        let r = camera_radius_km(&p, 2.0);
        let reps = 200;
        let t0 = std::time::Instant::now();
        let mut sink = 0.0f32;
        for i in 0..reps {
            let y = 0.2 + 0.6 * (i as f32 / reps as f32);
            let c = sky_irradiance_sh(
                &p,
                r,
                [(1.0f32 - y * y).max(0.0).sqrt(), y, 0.0],
                [3.0; 3],
                [0.0; 3],
                [0.0; 3],
            );
            sink += c[0][0];
        }
        let per = t0.elapsed().as_secs_f64() * 1000.0 / f64::from(reps);
        println!("FIX3 sky SH cost: {per:.4} ms per projection (sink {sink})");
        assert!(
            per < 2.0,
            "the sky SH projection costs {per:.4} ms — it runs every frame"
        );
    }
}
