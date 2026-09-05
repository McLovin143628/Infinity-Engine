//! Atmosphere LUT bake (P17.2): the two compute passes that turn
//! [`crate::atmosphere::AtmosphereParams`] + the projected sun into the pair of
//! lookup textures the sky pass and every lit pass sample.
//!
//! ```text
//! AtmosphereNode ──▶ transmittance LUT  (medium only; baked ~never)
//!                └─▶ sky-view LUT       (medium + sun + camera altitude; per-change)
//! ```
//!
//! Both live in the renderer-owned [`AtmosphereResources`], created once per
//! quality level and shared with the sky pass and [`super::EnvBinding`] through
//! [`crate::renderer::FrameData`] — the same arrangement
//! [`super::gi::GiResources`] uses, with one crucial difference: **these textures
//! are resizable**, because [`AtmosphereQuality`] can be clamped down by the
//! render tier at runtime. That is exactly the case the P13 `EnvBinding` cache
//! invariant warned about, so [`AtmosphereResources::generation`] exists and the
//! env bind-group cache keys on it. See [`super::EnvBinding::bind_group`].
//!
//! ## Version gating
//!
//! Neither LUT is re-baked unless its own inputs changed:
//!
//! * the **transmittance** LUT is a function of the medium and the planet shell
//!   alone ([`MediumKey`]) — not of the sun, not of the camera — so it is baked
//!   on the first enabled frame and then essentially never again;
//! * the **sky-view** LUT additionally depends on the sun direction, the sun's
//!   radiance, the exposure and the camera's altitude ([`SkyViewKey`]). The
//!   altitude is quantized to whole metres, so a hovering flycam does not
//!   re-bake 60 times a second over sub-millimetre jitter — a metre of altitude
//!   changes nothing visible at a 6360 km planet radius.
//!
//! ## Off path
//!
//! Disabled ⇒ the node dispatches **nothing** and only publishes the
//! `enabled = 0` uniform (once — a latch, like the GI node's). The sky pass and
//! the lit passes then take their pre-P17.2 arithmetic, so every golden without a
//! `TimeOfDay` authority is byte-identical.

use crate::atmosphere::{
    camera_radius_km, AtmosphereParams, AtmosphereQuality, SKY_EXPOSURE_CALIBRATION,
};
use crate::clouds::CloudQuality;
use crate::gpu::GpuContext;
use crate::graph::RenderNode;
use crate::renderer::FrameData;
use crate::scene::SunParams;

/// LUT storage format. `Rgba16Float` is storage-capable in core WebGPU and holds
/// the >1 radiances the sky-view LUT carries near the sun without clipping.
pub const LUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Bytes per LUT texel (four halves).
const LUT_TEXEL_BYTES: u32 = 8;

/// Side of the cloud-shadow map's world-XZ footprint, **metres** (P17.3).
///
/// 20 km at 512 texels is 39 m per texel. That is deliberately coarse: a cloud
/// four kilometres up casts a shadow whose penumbra is hundreds of metres wide,
/// so a crisper map would only be storing detail that is physically wrong. It is
/// also what keeps the map worth re-baking — 512² × a 16-step march is a few
/// hundred microseconds, and it only runs when the sun, the wind or the camera's
/// snapped centre actually moved.
pub const CLOUD_SHADOW_EXTENT_M: f32 = 20_000.0;

/// Storage format of the cloud-shadow map.
///
/// The obvious choice, `r16float`, is not a core WebGPU **storage** format; the
/// next one, `r32float`, is — but it is not **filterable** without the optional
/// `float32-filterable` feature, and a nearest-sampled cloud shadow is a grid of
/// 39 m squares rather than a soft penumbra. So the map spends four channels to
/// get one that is both storable and filterable everywhere. At 512² that is 2 MB
/// for a texture that must be bilinear, which is the right trade; it also reads
/// back as f16, whose ~1e-3 relative precision is two orders inside the parity
/// gate's envelope ([`crate::clouds::CPU_GPU_SHADOW_TOLERANCE`]).
pub const CLOUD_SHADOW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Storage format of both cloud noise volumes.
///
/// **`Rgba16Float` since wave SKY2**, and the reason is what happens *after* the
/// texel is read. The density function does not display the value, it
/// `remap`s it — twice, against the coverage floor and against the erosion — and
/// a remap is a division by a small window. An 8-bit channel carries 256 levels
/// across the whole `[0, 1]` range; the coverage dissolve at a typical setting
/// keeps a window about a tenth of that wide, so what survives is ~25 distinct
/// densities, and 25 levels stretched across a cloud's soft shoulder is
/// *terracing*. Sixteen-bit float carries ~2 048 levels over the same window.
///
/// `Rgba16Unorm` would be the tighter fit for a `[0, 1]` field and is **not** a
/// core WebGPU storage format (it needs `TEXTURE_FORMAT_16BIT_NORM`);
/// `Rgba16Float` is core, filterable everywhere, and its precision is worst at
/// the top of the range, which is where a cloud is opaque and nobody can see it.
///
/// The cost is exactly a doubling: 16.78 MB for a High-tier 128³ shape volume
/// against 8.39 MB, and 0.26 MB for the 32³ detail volume against 0.13 MB.
pub const CLOUD_NOISE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Bytes one [`CLOUD_NOISE_FORMAT`] texel occupies — four binary16 channels.
pub const CLOUD_NOISE_TEXEL_BYTES: u32 = 8;

/// Format of the [`crate::bluenoise`] tile the raymarch jitters against.
///
/// `r32float`, and its non-filterability is the point: the tile is a *rank* over
/// 4 096 texels, read with `textureLoad` at an integer texel and never
/// interpolated — interpolating a blue-noise tile would smooth away exactly the
/// high-frequency content it exists to supply. A `r8unorm` alternative would
/// quantize 4 096 ranks into 256 levels, putting sixteen texels of the tile at
/// every value.
pub const CLOUD_BLUE_NOISE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;

/// The shared atmosphere uniform (`std140`), written by [`AtmosphereNode`] and
/// read by both LUT compute shaders, the sky pass and every lit pass. Mirrors
/// `struct AtmosphereData` in `shaders/atmosphere.wgsl` field for field.
///
/// Units follow [`crate::atmosphere`]: kilometres and km⁻¹ everywhere except the
/// `fog` block, which is SI metres.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AtmosphereGpu {
    /// x = enabled, y = sky exposure, z = aerial-perspective strength,
    /// w = gradient tint strength.
    pub params: [f32; 4],
    /// rgb = Rayleigh scattering (km⁻¹), w = Rayleigh scale height (km).
    pub rayleigh: [f32; 4],
    /// x = Mie scattering (km⁻¹, turbidity-scaled), y = Mie absorption (km⁻¹),
    /// z = Mie scale height (km), w = Mie phase `g`.
    pub mie: [f32; 4],
    /// rgb = ozone absorption (km⁻¹), w = transmittance-bake step count.
    pub ozone: [f32; 4],
    /// x = ozone centre (km), y = ozone half-width (km), z = ground albedo,
    /// w = sky-view-bake step count.
    pub ozone_shape: [f32; 4],
    /// x = ground radius (km), y = top radius (km), z = camera radius (km),
    /// w = camera world altitude (m).
    pub planet: [f32; 4],
    /// xyz = unit direction toward the sun, w = cos(sun disc half-angle).
    pub sun_dir: [f32; 4],
    /// rgb = sun irradiance (colour × intensity), w reserved.
    pub sun_color: [f32; 4],
    /// xyz = unit direction toward the moon, w = cos(moon disc half-angle).
    pub moon_dir: [f32; 4],
    /// rgb = moon radiance (colour × intensity), w = lunar phase `[0, 1)`.
    pub moon_color: [f32; 4],
    /// x = star intensity, y = star cells per cube-face axis, z = night fade,
    /// w reserved.
    pub stars: [f32; 4],
    /// **SI metres**: x = fog density (m⁻¹), y = fog falloff (m⁻¹),
    /// z = fog reference height (m), w reserved.
    pub fog: [f32; 4],
    /// rgb = fog tint, w reserved.
    pub fog_color: [f32; 4],

    // ── volumetric clouds (P17.3), SI metres ──────────────────────────────
    //
    // The cloud block rides in the *atmosphere* uniform rather than one of its
    // own. That is a deliberate cost decision: a second uniform would add a
    // binding to `EnvBinding`, the sky pass, both bake layouts and the raymarch,
    // for data that is already a property of the same authored component. As it
    // is, the whole feature costs the lit passes exactly one new binding (the
    // shadow map).
    /// x = clouds enabled, y = coverage `[0,1]`, z = cloud type `[0,1]`,
    /// w = erosion detail strength `[0,1]`.
    pub clouds: [f32; 4],
    /// x = layer bottom (m), y = layer top (m), z = extinction at full density
    /// (m⁻¹), w = seed as an integer-valued f32 (see
    /// [`crate::clouds::CloudGpuSeed`]).
    pub cloud_layer: [f32; 4],
    /// x = wind offset X (m), y = wind offset Z (m), z = forward phase `g`,
    /// w = ambient multiplier.
    pub cloud_wind: [f32; 4],
    /// x = primary march steps, y = sun-transmittance steps, z = shadow-bake
    /// steps, w = 1 when this tier reads the detail volume.
    pub cloud_march: [f32; 4],
    /// x = world-shadow strength `[0,1]`, y = shadow-map world extent (m),
    /// z/w = shadow-map centre in **world** X/Z metres (texel-quantized).
    pub cloud_shadow: [f32; 4],
    /// rgb = cloud droplet albedo tint, w = the raymarch's blue-noise jitter
    /// phase (SKY2) — an integer-valued f32 in `[0, 64)` derived from the
    /// **level clock**, never from a frame index. See
    /// [`crate::clouds::jitter_phase`]. It rides here rather than in a lane of
    /// its own because `cloud_color.w` was the reserved slot the cloud block
    /// already carried, so the whole temporal path costs the uniform nothing.
    pub cloud_color: [f32; 4],

    // ── precipitation (P17.4), SI metres ──────────────────────────────────
    //
    // Rides in the atmosphere uniform for exactly the reason the cloud block
    // does: a uniform of its own would add a binding to a pass that already
    // binds this one, for data that is a property of the same authored
    // component. The precipitation pass therefore binds nothing new at all.
    /// x = particle count (integer-valued), y = intensity `[0,1]`,
    /// z = snowiness `[0,1]`, w = fall speed (m/s).
    pub precip: [f32; 4],
    /// x = horizontal box period (m), y = vertical box period (m),
    /// z = particle half-length along its velocity (m), w = particle radius (m).
    pub precip_box: [f32; 4],
    /// x = wind drift X (m, wrapped into the box), y = drift Z (m, wrapped),
    /// z = distance already fallen (m, wrapped), w = alpha scale.
    pub precip_phase: [f32; 4],
    /// xyz = the camera's world position **modulo** the box (m) — the
    /// world-anchoring term; w reserved.
    pub precip_eye: [f32; 4],
    /// rgb = droplet albedo tint, w reserved.
    pub precip_color: [f32; 4],
    /// x = wind X (m/s), y = wind Z (m/s) — the **raw rates**, not the wrapped
    /// drift, because the fall direction is `normalize(wind_x, −fall, wind_z)`
    /// and a wrapped drift is no longer proportional to the wind. zw reserved.
    pub precip_wind: [f32; 4],
}

/// What the sky's irradiance is a function of — the memo key below, as raw bit
/// patterns so two calls in one frame hash identical rather than nearly so.
#[derive(Clone, Copy, PartialEq, Eq)]
struct SkyShKey {
    medium: [u32; 16],
    sun: [u32; 3],
    irradiance: [u32; 3],
    gradient: [u32; 6],
    radius: u32,
    enabled: bool,
}

/// **The one memo in this file, and why it earns its place.**
///
/// [`crate::atmosphere::sky_irradiance_sh`] marches 48 directions through the
/// medium; measured at **0.095 ms** (`atmosphere::tests::sky_sh_is_cheap`). That
/// is 0.6 % of a 60 Hz frame — small, and paid two or three times per frame,
/// because [`AtmosphereGpu::build`] is called by the LUT node *and* re-derived
/// by the cloud node to read its cache keys out. Worse, it would be paid on
/// every frame of a scene whose sun has not moved, which is most of them.
///
/// The memo is keyed on the exact bits of everything the integral reads, so it
/// returns the identical answer the uncached call would: this is a cache, not an
/// approximation, and the determinism gates cannot tell it is here. One entry is
/// enough — the two or three calls in a frame share one key, and a moving sun
/// re-marches once per frame, which is the cost the feature is worth.
pub(crate) fn sky_irradiance_memo(
    p: &AtmosphereParams,
    r_km: f32,
    sun: [f32; 3],
    sun_irradiance: [f32; 3],
    gradient_horizon: [f32; 3],
    gradient_zenith: [f32; 3],
) -> [[f32; 3]; 4] {
    let key = SkyShKey {
        medium: [
            p.rayleigh_scattering[0].to_bits(),
            p.rayleigh_scattering[1].to_bits(),
            p.rayleigh_scattering[2].to_bits(),
            p.rayleigh_height.to_bits(),
            p.mie_scattering_scaled().to_bits(),
            p.mie_absorption_scaled().to_bits(),
            p.mie_height.to_bits(),
            p.mie_g.to_bits(),
            p.ozone_absorption[0].to_bits(),
            p.ozone_absorption[1].to_bits(),
            p.ozone_absorption[2].to_bits(),
            p.ozone_center.to_bits(),
            p.ozone_half_width.to_bits(),
            p.ground_radius.to_bits(),
            p.top_radius.to_bits(),
            p.ground_albedo.to_bits(),
        ],
        sun: [sun[0].to_bits(), sun[1].to_bits(), sun[2].to_bits()],
        irradiance: [
            sun_irradiance[0].to_bits(),
            sun_irradiance[1].to_bits(),
            sun_irradiance[2].to_bits(),
        ],
        gradient: [
            gradient_horizon[0].to_bits(),
            gradient_horizon[1].to_bits(),
            gradient_horizon[2].to_bits(),
            gradient_zenith[0].to_bits(),
            gradient_zenith[1].to_bits(),
            gradient_zenith[2].to_bits(),
        ],
        // The sky exposure rides here rather than in `medium` because it scales
        // the answer without changing the march.
        radius: r_km.to_bits() ^ p.sky_intensity.to_bits() ^ p.sun_disc_deg.to_bits(),
        enabled: p.enabled,
    };
    static MEMO: std::sync::Mutex<Option<(SkyShKey, [[f32; 3]; 4])>> = std::sync::Mutex::new(None);
    let mut slot = MEMO.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((k, v)) = slot.as_ref() {
        if *k == key {
            return *v;
        }
    }
    let v = crate::atmosphere::sky_irradiance_sh(
        p,
        r_km,
        sun,
        sun_irradiance,
        gradient_horizon,
        gradient_zenith,
    );
    *slot = Some((key, v));
    v
}

impl AtmosphereGpu {
    /// The `enabled = 0` block every pre-P17.2 scene renders under. The medium
    /// values are still the physical ones so a debugger shows something sane, but
    /// nothing reads them.
    ///
    pub fn disabled() -> Self {
        Self::build(
            &AtmosphereParams::default(),
            &SunParams::default(),
            glam::DVec3::ZERO,
            AtmosphereQuality::High,
            CloudQuality::High,
        )
    }

    /// Pack a scene's atmosphere + sun (+ clouds) into the shared uniform.
    ///
    /// `eye_world` is the camera's **world** position in metres, taken as `f64`
    /// (architecture rule 3): `y` is the only input the planet block takes
    /// ([`camera_radius_km`] converts it to the atmosphere's kilometres), `xz`
    /// centres the cloud-shadow map, and all three feed
    /// [`crate::precip::PrecipParams::eye_mod`] — which is the one consumer that
    /// genuinely needs the `f64`, because it takes a modulo and an `f32` metre
    /// at 3 200 km already has 0.25 m of resolution.
    pub fn build(
        p: &AtmosphereParams,
        sun: &SunParams,
        eye_world: glam::DVec3,
        quality: AtmosphereQuality,
        cloud_quality: CloudQuality,
    ) -> Self {
        let eye_world_m = [eye_world.x as f32, eye_world.y as f32, eye_world.z as f32];
        let eye_altitude_m = eye_world_m[1];
        let sd = sun.unit_direction();
        let md = sun.unit_moon_direction();
        let i = sun.intensity.max(0.0);
        let mi = sun.moon_intensity.max(0.0);
        // ── clouds (P17.3) ──
        let c = &p.clouds;
        // A degenerate slab (top ≤ bottom, or non-finite) collapses to zero
        // thickness, which the shader's `h` test rejects everywhere — so hostile
        // authoring produces an empty sky rather than a divide by zero.
        let bottom = if c.bottom.is_finite() { c.bottom } else { 0.0 };
        let top = if c.top.is_finite() {
            c.top.max(bottom)
        } else {
            bottom
        };
        let wind = c.wind_offset();
        let shadow_centre =
            Self::cloud_shadow_centre([eye_world_m[0], eye_world_m[2]], cloud_quality);
        // ── precipitation (P17.4) ──
        let precip = &p.precip;
        let precip_quality = crate::precip::PrecipQuality::from_atmosphere(quality);
        let precip_offsets = precip.offsets();
        let precip_eye = crate::precip::PrecipParams::eye_mod(eye_world);
        Self {
            params: [
                if p.enabled { 1.0 } else { 0.0 },
                p.sky_intensity.max(0.0) * SKY_EXPOSURE_CALIBRATION,
                p.aerial_perspective.clamp(0.0, 4.0),
                p.tint_strength.clamp(0.0, 1.0),
            ],
            rayleigh: [
                p.rayleigh_scattering[0],
                p.rayleigh_scattering[1],
                p.rayleigh_scattering[2],
                p.rayleigh_height.max(1e-3),
            ],
            mie: [
                p.mie_scattering_scaled(),
                p.mie_absorption_scaled(),
                p.mie_height.max(1e-3),
                p.mie_g.clamp(-0.95, 0.95),
            ],
            ozone: [
                p.ozone_absorption[0],
                p.ozone_absorption[1],
                p.ozone_absorption[2],
                quality.transmittance_steps() as f32,
            ],
            ozone_shape: [
                p.ozone_center,
                p.ozone_half_width.max(1e-3),
                p.ground_albedo.clamp(0.0, 1.0),
                quality.skyview_steps() as f32,
            ],
            planet: [
                p.ground_radius,
                p.top_radius,
                camera_radius_km(p, eye_altitude_m),
                eye_altitude_m,
            ],
            sun_dir: [sd.x, sd.y, sd.z, p.sun_disc_cos()],
            sun_color: [sun.color[0] * i, sun.color[1] * i, sun.color[2] * i, 0.0],
            moon_dir: [md.x, md.y, md.z, p.moon_disc_cos()],
            moon_color: [
                sun.moon_color[0] * mi,
                sun.moon_color[1] * mi,
                sun.moon_color[2] * mi,
                p.moon_phase.rem_euclid(1.0),
            ],
            stars: [
                p.star_intensity.max(0.0),
                quality.star_cells(),
                night_fade(sd.y),
                0.0,
            ],
            fog: [
                p.fog.density.max(0.0),
                p.fog.falloff.max(0.0),
                p.fog.height,
                0.0,
            ],
            fog_color: [p.fog.color[0], p.fog.color[1], p.fog.color[2], 0.0],
            clouds: [
                if p.clouds_active() { 1.0 } else { 0.0 },
                c.coverage.clamp(0.0, 1.0),
                c.cloud_type.clamp(0.0, 1.0),
                c.detail.clamp(0.0, 1.0),
            ],
            cloud_layer: [
                bottom,
                top,
                c.density.max(0.0),
                crate::clouds::CloudGpuSeed::encode(c.seed),
            ],
            cloud_wind: [
                wind[0],
                wind[1],
                c.phase_g.clamp(0.0, 0.95),
                c.ambient.clamp(0.0, 4.0),
            ],
            cloud_march: [
                cloud_quality.march_steps() as f32,
                cloud_quality.light_steps() as f32,
                cloud_quality.shadow_steps() as f32,
                if cloud_quality.uses_detail() {
                    1.0
                } else {
                    0.0
                },
            ],
            cloud_shadow: [
                if p.cloud_shadows_active() {
                    c.shadow_strength.clamp(0.0, 1.0)
                } else {
                    0.0
                },
                CLOUD_SHADOW_EXTENT_M,
                shadow_centre[0],
                shadow_centre[1],
            ],
            cloud_color: [
                c.color[0],
                c.color[1],
                c.color[2],
                crate::clouds::jitter_phase(c.time_s),
            ],
            // ── precipitation (P17.4) ──
            precip: [
                if p.precip_active() {
                    precip.count(precip_quality) as f32
                } else {
                    0.0
                },
                precip.intensity.clamp(0.0, 1.0),
                precip.snowiness.clamp(0.0, 1.0),
                precip.fall_speed(),
            ],
            precip_box: [
                crate::precip::PRECIP_BOX_XZ_M,
                crate::precip::PRECIP_BOX_Y_M,
                precip.half_length(),
                precip.radius(),
            ],
            precip_phase: [
                precip_offsets[0],
                precip_offsets[1],
                precip_offsets[2],
                crate::precip::PRECIP_ALPHA,
            ],
            precip_eye: [precip_eye[0], precip_eye[1], precip_eye[2], 0.0],
            precip_color: [precip.color[0], precip.color[1], precip.color[2], 0.0],
            precip_wind: [precip.wind_x, precip.wind_z, 0.0, 0.0],
        }
    }

    /// The cloud-shadow map's world-XZ centre for a camera at `eye_xz`, snapped
    /// to a whole texel.
    ///
    /// Snapping is not optional. The map is re-baked whenever its centre moves,
    /// and an unsnapped centre moves every frame a flycam breathes — which would
    /// both re-bake constantly and, worse, slide the shadow pattern by a fraction
    /// of a texel each frame so that a *static* scene shimmered. Snapped, the map
    /// jumps by exactly one texel at a time and the bilinear lookup lands on the
    /// same values it did before.
    pub fn cloud_shadow_centre(eye_xz: [f32; 2], quality: CloudQuality) -> [f32; 2] {
        let texel = CLOUD_SHADOW_EXTENT_M / quality.shadow_res().max(1) as f32;
        let snap = |v: f32| {
            if v.is_finite() {
                (v / texel).round() * texel
            } else {
                0.0
            }
        };
        [snap(eye_xz[0]), snap(eye_xz[1])]
    }

    /// The uniform block for this frame — the single definition of how a
    /// [`FrameData`] becomes `AtmosphereData`, shared by the LUT bake node (which
    /// writes it) and the cloud bake node (which re-derives it to read its keys
    /// out). Two hand-rolled copies of this would be two chances for the cache
    /// keys and the uniform to disagree about what was baked.
    pub(crate) fn from_frame(frame: &FrameData) -> Self {
        Self::build(
            &frame.scene.atmosphere,
            &frame.scene.sun,
            frame.view.eye_world,
            frame.atmosphere.quality,
            frame.atmosphere.cloud_quality,
        )
    }

    fn medium_key(&self) -> MediumKey {
        MediumKey {
            rayleigh: self.rayleigh.map(f32::to_bits),
            mie: self.mie.map(f32::to_bits),
            ozone: self.ozone.map(f32::to_bits),
            ozone_shape: self.ozone_shape.map(f32::to_bits),
            shell: [self.planet[0].to_bits(), self.planet[1].to_bits()],
        }
    }

    fn skyview_key(&self) -> SkyViewKey {
        SkyViewKey {
            medium: self.medium_key(),
            sun_dir: [
                self.sun_dir[0].to_bits(),
                self.sun_dir[1].to_bits(),
                self.sun_dir[2].to_bits(),
            ],
            sun_color: [
                self.sun_color[0].to_bits(),
                self.sun_color[1].to_bits(),
                self.sun_color[2].to_bits(),
            ],
            exposure: self.params[1].to_bits(),
            // Camera radius, quantized to whole metres: a sub-metre flycam
            // wobble must not re-bake the LUT, and cannot change it visibly.
            radius_m: (self.planet[2] as f64 * 1000.0).round() as i64,
        }
    }

    /// What the two baked cloud **noise volumes** are a function of: the seed,
    /// and nothing else.
    ///
    /// Not the coverage, not the wind, not the sun — those all shape the field at
    /// *sample* time, which is the whole reason the volumes are worth baking. So
    /// a level dragging the coverage slider re-bakes nothing at all, and 8.4 MB
    /// of 3D noise is written once and then never again.
    pub(crate) fn cloud_field_key(&self) -> u32 {
        self.cloud_layer[3].to_bits()
    }

    /// What the cloud-**shadow** map is a function of: the field, the layer
    /// geometry, the weather, the wind's current displacement, the sun direction,
    /// the march budget and the map's snapped centre.
    ///
    /// Everything that moves a shadow, and nothing that does not — the camera's
    /// *altitude*, for instance, is absent, because the map is a property of the
    /// world rather than of the viewer.
    pub(crate) fn cloud_shadow_key(&self) -> CloudShadowKey {
        CloudShadowKey {
            enabled: self.clouds[0].to_bits(),
            field: self.cloud_field_key(),
            shape: [
                self.clouds[1].to_bits(),
                self.clouds[2].to_bits(),
                self.clouds[3].to_bits(),
            ],
            layer: [
                self.cloud_layer[0].to_bits(),
                self.cloud_layer[1].to_bits(),
                self.cloud_layer[2].to_bits(),
            ],
            wind: [self.cloud_wind[0].to_bits(), self.cloud_wind[1].to_bits()],
            sun_dir: [
                self.sun_dir[0].to_bits(),
                self.sun_dir[1].to_bits(),
                self.sun_dir[2].to_bits(),
            ],
            steps: [self.cloud_march[2].to_bits(), self.cloud_march[3].to_bits()],
            centre: [
                self.cloud_shadow[2].to_bits(),
                self.cloud_shadow[3].to_bits(),
            ],
            extent: self.cloud_shadow[1].to_bits(),
        }
    }
}

/// What the cloud-shadow map is a function of. Bit patterns, like every other key
/// here, so a `NaN` parameter cannot make the cache thrash.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct CloudShadowKey {
    enabled: u32,
    field: u32,
    shape: [u32; 3],
    layer: [u32; 3],
    wind: [u32; 2],
    sun_dir: [u32; 3],
    steps: [u32; 2],
    centre: [u32; 2],
    extent: u32,
}

/// How far past sunset the stars have fully faded in, as a sine of elevation.
/// `-0.12` ≈ 6.9° below the horizon — the end of civil twilight, which is
/// exactly when a human first sees stars.
const NIGHT_FADE_ELEVATION_SIN: f32 = 0.12;

/// Star/night blend from the sun's elevation sine: `0` while the sun is up,
/// smoothly `1` by the end of civil twilight.
fn night_fade(sun_y: f32) -> f32 {
    let t = (-sun_y / NIGHT_FADE_ELEVATION_SIN).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// What the transmittance LUT is a function of. Bit patterns rather than floats
/// so the key is `Eq` and a `NaN` parameter cannot make the cache thrash.
#[derive(Clone, Copy, PartialEq, Eq)]
struct MediumKey {
    rayleigh: [u32; 4],
    mie: [u32; 4],
    ozone: [u32; 4],
    ozone_shape: [u32; 4],
    shell: [u32; 2],
}

/// What the sky-view LUT is a function of.
#[derive(Clone, Copy, PartialEq, Eq)]
struct SkyViewKey {
    medium: MediumKey,
    sun_dir: [u32; 3],
    sun_color: [u32; 3],
    exposure: u32,
    radius_m: i64,
}

/// Renderer-owned atmosphere GPU resources: the two LUTs, their sampler and the
/// shared uniform.
///
/// **Resizable** (unlike the shadow/GI resources): a change of
/// [`AtmosphereQuality`] recreates the textures and bumps
/// [`generation`](Self::generation), which every bind-group cache that embeds
/// them must key on.
pub struct AtmosphereResources {
    transmittance_tex: wgpu::Texture,
    sky_view_tex: wgpu::Texture,
    shape_tex: wgpu::Texture,
    detail_tex: wgpu::Texture,
    cloud_shadow_tex: wgpu::Texture,
    /// Transmittance LUT (sun-zenith-angle × altitude).
    pub transmittance: wgpu::TextureView,
    /// Sky-view LUT (view direction, sun-relative).
    pub sky_view: wgpu::TextureView,
    /// Clamped bilinear sampler both LUTs are read through. Also serves the
    /// cloud-shadow map, which wants exactly the same addressing.
    pub sampler: wgpu::Sampler,
    /// Cloud **shape** volume (Perlin–Worley base + 3 Worley octaves), P17.3.
    pub cloud_shape: wgpu::TextureView,
    /// Cloud **detail** (erosion) volume.
    pub cloud_detail: wgpu::TextureView,
    /// Cloud sun-transmittance map, world-XZ (P17.3).
    pub cloud_shadow: wgpu::TextureView,
    /// The wave-SKY2 blue-noise tile the raymarch offsets its first sample by
    /// ([`crate::bluenoise`]). 16 KB, generated once on the CPU and uploaded
    /// here — it lives beside the cloud textures because it is only ever read by
    /// a cloud pass, and it rides the same generation for the same reason the
    /// volumes do.
    pub cloud_blue_noise: wgpu::TextureView,
    /// **Repeating** bilinear sampler the two tileable cloud volumes are read
    /// through — the whole point of baking them tileable.
    pub cloud_sampler: wgpu::Sampler,
    /// The shared `AtmosphereData` uniform.
    pub uniform: wgpu::Buffer,
    /// Bumped on every recreation. Bind-group caches MUST key on this.
    pub generation: u64,
    /// The quality these textures were sized for.
    pub quality: AtmosphereQuality,
    /// The cloud tier the three cloud textures were sized for. Derived from
    /// [`quality`](Self::quality) — one knob, not two — and therefore covered by
    /// the same `generation`, because the two are always recreated together.
    pub cloud_quality: CloudQuality,
}

impl AtmosphereResources {
    pub fn new(gpu: &GpuContext, quality: AtmosphereQuality, generation: u64) -> Self {
        let cloud_quality = CloudQuality::from_atmosphere(quality);
        let make = |label: &str, (w, h): (u32, u32)| {
            gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: LUT_FORMAT,
                usage: wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    // The LUT-determinism gate reads these back on the CPU.
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        };
        let transmittance_tex = make("atmos-transmittance-lut", quality.transmittance_size());
        let sky_view_tex = make("atmos-skyview-lut", quality.skyview_size());
        let transmittance = transmittance_tex.create_view(&Default::default());
        let sky_view = sky_view_tex.create_view(&Default::default());

        // ── cloud volumes + shadow map (P17.3) ──
        //
        // Allocated unconditionally, like `ShadowResources`' 48 MB cascade array
        // and `GiResources`' voxel grid: ~9.6 MB at High is well inside the
        // house's existing always-on budget, and a lazily-grown texture would
        // still need a 1×1 placeholder bound at every binding so that the bind
        // groups stay valid — trading real complexity for a fraction of what the
        // shadow map already costs.
        let volume = |label: &str, res: u32| {
            gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: res,
                    height: res,
                    depth_or_array_layers: res,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: CLOUD_NOISE_FORMAT,
                usage: wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    // The CPU/GPU noise-parity gate reads these back.
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        };
        let shape_tex = volume("cloud-shape", cloud_quality.shape_res());
        let detail_tex = volume("cloud-detail", cloud_quality.detail_res());
        let cloud_shadow_tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("cloud-shadow-map"),
            size: wgpu::Extent3d {
                width: cloud_quality.shadow_res(),
                height: cloud_quality.shadow_res(),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: CLOUD_SHADOW_FORMAT,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let cloud_shape = shape_tex.create_view(&Default::default());
        let cloud_detail = detail_tex.create_view(&Default::default());
        let cloud_shadow = cloud_shadow_tex.create_view(&Default::default());

        // ── the blue-noise tile (SKY2) ──
        // Uploaded rather than baked on the GPU: it is 16 KB, it is the same tile
        // at every tier, and the CPU generator IS the definition — a compute
        // shader would be a second implementation of void-and-cluster to keep in
        // step for no saving. `R32Float` because the tile is a *rank* over 4 096
        // texels and an 8-bit channel could only carry 256 of them; it is read
        // with `textureLoad`, so nothing filters it and the format's
        // non-filterability costs nothing.
        let bn_res = crate::bluenoise::BLUE_NOISE_RES;
        let blue_noise_tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("cloud-blue-noise"),
            size: wgpu::Extent3d {
                width: bn_res,
                height: bn_res,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: CLOUD_BLUE_NOISE_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &blue_noise_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(crate::bluenoise::blue_noise_tile()),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bn_res * 4),
                rows_per_image: Some(bn_res),
            },
            wgpu::Extent3d {
                width: bn_res,
                height: bn_res,
                depth_or_array_layers: 1,
            },
        );
        let cloud_blue_noise = blue_noise_tex.create_view(&Default::default());
        // Repeat, not clamp: both volumes are tileable by construction, and that
        // is what lets a 30 km march wrap an 8 km texture with no seam. Getting
        // this wrong would smear the last texel across the whole sky.
        let cloud_sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("cloud-noise"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        // Clamp-to-edge is load-bearing, not a default: the transmittance
        // parameterization puts the horizon at the very edge of the u axis, and a
        // repeating sampler would wrap a grazing ray onto an overhead one.
        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("atmos-lut"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let uniform = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("atmos-data"),
            size: std::mem::size_of::<AtmosphereGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue
            .write_buffer(&uniform, 0, bytemuck::bytes_of(&AtmosphereGpu::disabled()));
        Self {
            transmittance_tex,
            sky_view_tex,
            shape_tex,
            detail_tex,
            cloud_shadow_tex,
            transmittance,
            sky_view,
            sampler,
            cloud_shape,
            cloud_detail,
            cloud_shadow,
            cloud_blue_noise,
            cloud_sampler,
            uniform,
            generation,
            quality,
            cloud_quality,
        }
    }

    /// Blocking readback of the transmittance LUT as tightly-packed
    /// `Rgba16Float` texels. The LUT-determinism gate compares two of these.
    pub fn read_transmittance(&self, gpu: &GpuContext) -> Result<Vec<u8>, String> {
        read_lut(
            gpu,
            &self.transmittance_tex,
            self.quality.transmittance_size(),
        )
    }

    /// Blocking readback of the sky-view LUT. See [`Self::read_transmittance`].
    pub fn read_sky_view(&self, gpu: &GpuContext) -> Result<Vec<u8>, String> {
        read_lut(gpu, &self.sky_view_tex, self.quality.skyview_size())
    }

    /// Blocking readback of the baked cloud **shape** volume as tightly-packed
    /// `Rgba16Float` texels in x-major order — eight bytes each, four
    /// little-endian halves. The CPU/GPU noise-parity gate compares this against
    /// [`crate::clouds::shape_texel`].
    pub fn read_cloud_shape(&self, gpu: &GpuContext) -> Result<Vec<u8>, String> {
        let r = self.cloud_quality.shape_res();
        read_texture(gpu, &self.shape_tex, (r, r, r), CLOUD_NOISE_TEXEL_BYTES)
    }

    /// Blocking readback of the baked cloud **detail** volume.
    pub fn read_cloud_detail(&self, gpu: &GpuContext) -> Result<Vec<u8>, String> {
        let r = self.cloud_quality.detail_res();
        read_texture(gpu, &self.detail_tex, (r, r, r), CLOUD_NOISE_TEXEL_BYTES)
    }

    /// Blocking readback of the cloud-shadow map's transmittances, decoded from
    /// f16 (see [`CLOUD_SHADOW_FORMAT`]). Row-major, `shadow_res()²` values.
    pub fn read_cloud_shadow(&self, gpu: &GpuContext) -> Result<Vec<f32>, String> {
        let r = self.cloud_quality.shadow_res();
        let bytes = read_texture(gpu, &self.cloud_shadow_tex, (r, r, 1), LUT_TEXEL_BYTES)?;
        // Four halves per texel; the transmittance is the red channel.
        Ok(bytes
            .chunks_exact(8)
            .map(|c| half_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect())
    }
}

// IEEE-754 binary16 → f32 lives in `crate::clouds` since wave SKY2, beside the
// f32 → binary16 the 16-bit noise volumes needed. Two directions of one format,
// in one place.
use crate::clouds::half_to_f32;

/// 256-byte-row-aligned texture → CPU copy, unpadded on the way out (the same
/// shape as [`crate::headless::HeadlessTarget::read_rgba`], for a 4×f16 format).
fn read_lut(gpu: &GpuContext, tex: &wgpu::Texture, (w, h): (u32, u32)) -> Result<Vec<u8>, String> {
    read_texture(gpu, tex, (w, h, 1), LUT_TEXEL_BYTES)
}

/// The general form: any 2D or 3D texture → a tightly-packed CPU copy.
///
/// `rows_per_image` is what makes the 3D case work — without it a depth slice's
/// rows are padded but its *slices* are not, and every slice after the first
/// lands at the wrong offset. That is a silent corruption (the readback still has
/// the right length), which is exactly the kind of thing a parity gate would
/// blame on the shader.
fn read_texture(
    gpu: &GpuContext,
    tex: &wgpu::Texture,
    (w, h, d): (u32, u32, u32),
    texel_bytes: u32,
) -> Result<Vec<u8>, String> {
    let unpadded = (w * texel_bytes) as usize;
    let padded = unpadded.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize)
        * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;
    let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("atmos-lut-readback"),
        size: (padded * h as usize * d as usize) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("atmos-lut-readback"),
        });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded as u32),
                rows_per_image: Some(h),
            },
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: d,
        },
    );
    gpu.queue.submit([encoder.finish()]);

    let slice = buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| format!("poll: {e}"))?;
    rx.recv()
        .map_err(|e| format!("map_async dropped: {e}"))?
        .map_err(|e| format!("map_async: {e}"))?;
    let data = slice
        .get_mapped_range()
        .map_err(|e| format!("get_mapped_range: {e}"))?;
    let mut out = Vec::with_capacity(unpadded * h as usize * d as usize);
    for row in data.chunks(padded) {
        out.extend_from_slice(&row[..unpadded]);
    }
    drop(data);
    buffer.unmap();
    Ok(out)
}

/// The graph node that owns the two bake pipelines.
pub struct AtmosphereNode {
    transmittance_pipeline: wgpu::ComputePipeline,
    transmittance_bgl: wgpu::BindGroupLayout,
    sky_view_pipeline: wgpu::ComputePipeline,
    sky_view_bgl: wgpu::BindGroupLayout,
    /// The uniform value currently in `frame.atmosphere.uniform`.
    uploaded: Option<AtmosphereGpu>,
    /// `(resource generation, medium)` the transmittance LUT holds.
    transmittance_baked: Option<(u64, MediumKey)>,
    /// `(resource generation, inputs)` the sky-view LUT holds.
    sky_view_baked: Option<(u64, SkyViewKey)>,
    /// Bind groups, cached against the resource generation (see
    /// [`super::GenCache`]). Nothing here is size-dependent, so the frame-target
    /// generation is deliberately not part of the key.
    transmittance_bg: super::GenCache<u64, wgpu::BindGroup>,
    sky_view_bg: super::GenCache<u64, wgpu::BindGroup>,
}

impl AtmosphereNode {
    pub fn new(gpu: &GpuContext) -> Self {
        let uniform_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let storage_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: LUT_FORMAT,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        };

        let transmittance_bgl =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("atmos-transmittance"),
                    entries: &[uniform_entry(0), storage_entry(1)],
                });
        let sky_view_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("atmos-skyview"),
                entries: &[
                    uniform_entry(0),
                    storage_entry(1),
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let pipeline = |label: &str, source: String, bgl: &wgpu::BindGroupLayout, entry: &str| {
            let module = gpu
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(label),
                    source: wgpu::ShaderSource::Wgsl(source.into()),
                });
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
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                })
        };

        Self {
            transmittance_pipeline: pipeline(
                "atmos-transmittance",
                super::shader_source("atmos_transmittance"),
                &transmittance_bgl,
                "cs_transmittance",
            ),
            transmittance_bgl,
            sky_view_pipeline: pipeline(
                "atmos-skyview",
                super::shader_source("atmos_skyview"),
                &sky_view_bgl,
                "cs_skyview",
            ),
            sky_view_bgl,
            uploaded: None,
            transmittance_baked: None,
            sky_view_baked: None,
            transmittance_bg: super::GenCache::default(),
            sky_view_bg: super::GenCache::default(),
        }
    }
}

/// Compute workgroup edge; must match `@workgroup_size(8, 8, 1)` in both bakes.
const WG: u32 = 8;

impl RenderNode for AtmosphereNode {
    fn name(&self) -> &'static str {
        "atmosphere"
    }

    fn run(&mut self, gpu: &GpuContext, encoder: &mut wgpu::CommandEncoder, frame: &FrameData) {
        let res = frame.atmosphere;
        let params = &frame.scene.atmosphere;
        let data = AtmosphereGpu::from_frame(frame);

        // Publish the uniform when it changed. `AtmosphereResources::new` already
        // wrote the disabled block, so a scene that never enables the atmosphere
        // writes this buffer exactly zero times.
        if self.uploaded != Some(data) {
            gpu.queue
                .write_buffer(&res.uniform, 0, bytemuck::bytes_of(&data));
            self.uploaded = Some(data);
        }
        if !params.enabled {
            return;
        }

        // ── transmittance LUT: medium only ──
        let medium = data.medium_key();
        if self.transmittance_baked != Some((res.generation, medium)) {
            let layout = &self.transmittance_bgl;
            let bg = self
                .transmittance_bg
                .get_or_build(res.generation, || {
                    gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("atmos-transmittance"),
                        layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: res.uniform.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(&res.transmittance),
                            },
                        ],
                    })
                })
                .clone();
            let (w, h) = res.quality.transmittance_size();
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("atmos-transmittance"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.transmittance_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(w.div_ceil(WG), h.div_ceil(WG), 1);
            drop(pass);
            self.transmittance_baked = Some((res.generation, medium));
            // A fresh transmittance LUT invalidates the sky view that was baked
            // from the old one.
            self.sky_view_baked = None;
        }

        // ── sky-view LUT: medium + sun + camera altitude ──
        let key = data.skyview_key();
        if self.sky_view_baked == Some((res.generation, key)) {
            return;
        }
        let layout = &self.sky_view_bgl;
        let bg = self
            .sky_view_bg
            .get_or_build(res.generation, || {
                gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("atmos-skyview"),
                    layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: res.uniform.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&res.sky_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&res.transmittance),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::Sampler(&res.sampler),
                        },
                    ],
                })
            })
            .clone();
        let (w, h) = res.quality.skyview_size();
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("atmos-skyview"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.sky_view_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(w.div_ceil(WG), h.div_ceil(WG), 1);
        drop(pass);
        self.sky_view_baked = Some((res.generation, key));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atmosphere::HeightFog;

    fn enabled() -> AtmosphereParams {
        AtmosphereParams {
            enabled: true,
            ..AtmosphereParams::default()
        }
    }

    #[test]
    fn disabled_block_is_flagged_off() {
        let d = AtmosphereGpu::disabled();
        assert_eq!(d.params[0], 0.0);
        // The medium is still physical, so a capture shows something meaningful.
        assert!(d.rayleigh[2] > d.rayleigh[0]);
        assert_eq!(d.fog[0], 0.0);
    }

    /// The whole point of the version gate: the transmittance key must ignore the
    /// sun and the camera, and the sky-view key must not.
    #[test]
    fn lut_keys_track_the_right_inputs() {
        let p = enabled();
        let noon = SunParams {
            direction: glam::Vec3::new(0.0, 1.0, 0.0),
            ..SunParams::default()
        };
        let dusk = SunParams {
            direction: glam::Vec3::new(1.0, 0.05, 0.0),
            ..SunParams::default()
        };
        let q = AtmosphereQuality::High;
        let cq = CloudQuality::High;
        let a = AtmosphereGpu::build(&p, &noon, glam::DVec3::new(0.0, 2.0, 0.0), q, cq);
        let b = AtmosphereGpu::build(&p, &dusk, glam::DVec3::new(0.0, 2.0, 0.0), q, cq);
        assert!(a.medium_key() == b.medium_key(), "sun moved the medium key");
        assert!(
            a.skyview_key() != b.skyview_key(),
            "sun missed the view key"
        );

        // Altitude: sub-metre jitter must NOT re-bake; a real climb must.
        let c = AtmosphereGpu::build(&p, &noon, glam::DVec3::new(0.0, 2.4, 0.0), q, cq);
        let d = AtmosphereGpu::build(&p, &noon, glam::DVec3::new(0.0, 250.0, 0.0), q, cq);
        assert!(
            a.skyview_key() == c.skyview_key(),
            "sub-metre jitter re-baked"
        );
        assert!(
            a.skyview_key() != d.skyview_key(),
            "a 248 m climb was ignored"
        );

        // Turbidity is a medium change, so both LUTs must fall out of date.
        let hazy = AtmosphereGpu::build(
            &AtmosphereParams {
                turbidity: 3.0,
                ..p
            },
            &noon,
            glam::DVec3::new(0.0, 2.0, 0.0),
            q,
            cq,
        );
        assert!(a.medium_key() != hazy.medium_key());
        assert!(a.skyview_key() != hazy.skyview_key());

        // Fog is a *receiver* parameter: it changes neither LUT.
        let foggy = AtmosphereGpu::build(
            &AtmosphereParams {
                fog: HeightFog {
                    density: 1e-3,
                    ..HeightFog::default()
                },
                ..p
            },
            &noon,
            glam::DVec3::new(0.0, 2.0, 0.0),
            q,
            cq,
        );
        assert!(a.medium_key() == foggy.medium_key());
        assert!(a.skyview_key() == foggy.skyview_key());
        assert_ne!(a, foggy, "the uniform itself must still change");
    }

    /// The **other half** of the "clouds need no key component" argument (the
    /// first half is `passes::gen_cache_tests::cloud_quality_is_a_total_function_of_atmosphere_quality`).
    ///
    /// An injective tier mapping is not enough on its own: if some other code path
    /// could assign `AtmosphereResources::cloud_quality` after construction, the
    /// cloud textures could change under an *unchanged* `generation` and every
    /// bind group keyed on it would go quietly stale — the exact failure the
    /// `EnvBinding` invariant exists to prevent, and one wgpu keeps silent by
    /// holding the old textures alive.
    ///
    /// So: the field is written in exactly one place, `new`, which is also the
    /// only place that takes a fresh `generation`. Asserted over the crate's own
    /// source because that is what the claim is *about* — there is no runtime
    /// observation that could distinguish "assigned once" from "assigned twice
    /// with the same value", and a type-level guarantee would mean making the
    /// field private to a module that already contains its only writer.
    #[test]
    fn cloud_quality_is_only_assigned_at_construction() {
        fn walk(dir: &std::path::Path, out: &mut Vec<(std::path::PathBuf, usize, String)>) {
            for entry in std::fs::read_dir(dir).expect("readable src dir") {
                let path = entry.expect("dir entry").path();
                if path.is_dir() {
                    walk(&path, out);
                } else if path.extension().is_some_and(|e| e == "rs") {
                    let text = std::fs::read_to_string(&path).expect("readable source");
                    for (i, line) in text.lines().enumerate() {
                        let trimmed = line.trim();
                        // Code only. Prose that *discusses* the field — including
                        // the comments on this very test — is not an assignment,
                        // and scanning it produces nothing but false positives.
                        if trimmed.starts_with("//") {
                            continue;
                        }
                        if trimmed.contains("cloud_quality") {
                            out.push((path.clone(), i + 1, trimmed.to_string()));
                        }
                    }
                }
            }
        }
        let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut hits = Vec::new();
        walk(&src, &mut hits);
        assert!(
            !hits.is_empty(),
            "the source scan found nothing — wrong path?"
        );

        // The needles are assembled at run time rather than written as literals:
        // this file is inside the tree being scanned, and a literal needle would
        // match the line that declares it. (It did, on the first run.)
        let field = "cloud_quality";
        let assign = format!(".{field} =");
        let bind = format!("let {field} =");

        // No field assignment anywhere: a dotted assignment to the field is the
        // shape that would break the invariant.
        for (path, line, text) in &hits {
            assert!(
                !text.contains(&assign),
                "{}:{line} assigns the cloud tier after construction, which would \
                 change the cloud textures under an unchanged generation:
  {text}",
                path.display()
            );
        }
        // ...and exactly one binding, the one `new` computes from the atmosphere
        // tier it was handed.
        let bindings: Vec<_> = hits
            .iter()
            .filter(|(_, _, t)| t.starts_with(&bind))
            .collect();
        assert_eq!(
            bindings.len(),
            1,
            "expected exactly one `let cloud_quality =` (in `AtmosphereResources::new`), found {bindings:#?}"
        );
        assert!(
            bindings[0].2.contains("CloudQuality::from_atmosphere"),
            "the one binding no longer derives the tier from the atmosphere tier: {}",
            bindings[0].2
        );
    }

    /// Stars appear only after the sun is down, and are fully out by the end of
    /// civil twilight.
    #[test]
    fn night_fade_matches_civil_twilight() {
        assert_eq!(night_fade(1.0), 0.0);
        assert_eq!(night_fade(0.0), 0.0);
        assert_eq!(night_fade(-1.0), 1.0);
        let dusk = night_fade(-0.06); // half way
        assert!(dusk > 0.4 && dusk < 0.6, "{dusk}");
        // Monotone.
        let mut prev = 0.0;
        for i in 0..=64 {
            let y = -(i as f32) / 64.0 * 0.2;
            let f = night_fade(y);
            assert!(f + 1e-6 >= prev, "not monotone at {y}");
            prev = f;
        }
    }

    /// The uniform is a pure function of its inputs — the CPU half of the LUT
    /// determinism gate (the GPU half reads the baked texels back).
    #[test]
    fn uniform_is_deterministic() {
        let p = enabled();
        let s = SunParams::default();
        let eye = glam::DVec3::new(3.5, 12.5, -7.25);
        let a = AtmosphereGpu::build(&p, &s, eye, AtmosphereQuality::Medium, CloudQuality::Medium);
        let b = AtmosphereGpu::build(&p, &s, eye, AtmosphereQuality::Medium, CloudQuality::Medium);
        assert_eq!(bytemuck::bytes_of(&a), bytemuck::bytes_of(&b));
    }

    /// Nonsense from a projector must not reach the GPU as `NaN`/`inf`.
    #[test]
    fn degenerate_input_stays_finite() {
        let p = AtmosphereParams {
            enabled: true,
            sky_intensity: f32::NAN,
            rayleigh_height: 0.0,
            mie_height: -1.0,
            mie_g: 12.0,
            ozone_half_width: 0.0,
            turbidity: 1e30,
            ..AtmosphereParams::default()
        };
        let s = SunParams {
            direction: glam::Vec3::ZERO,
            intensity: -5.0,
            ..SunParams::default()
        };
        let g = AtmosphereGpu::build(
            &p,
            &s,
            glam::DVec3::new(f64::NAN, f64::INFINITY, f64::NEG_INFINITY),
            AtmosphereQuality::Low,
            CloudQuality::Low,
        );
        assert!(g.rayleigh[3] > 0.0 && g.mie[2] > 0.0 && g.ozone_shape[1] > 0.0);
        assert!((-0.95..=0.95).contains(&g.mie[3]));
        assert!(g.planet[2].is_finite() && g.planet[2] > g.planet[0]);
        assert!(g.sun_dir[0].is_finite() && g.sun_color[0] >= 0.0);
        // A NaN exposure is the one thing that would poison the whole HDR buffer —
        // so assert it cannot BE one. `f32::max` returns the non-NaN operand, so
        // `NaN.max(0.0)` is 0.0 and the NaN never reaches the uniform. That is a
        // real property of the clamp, not a hope about it.
        assert!(
            g.params[1].is_finite() && g.params[1] >= 0.0,
            "a NaN sky intensity reached the GPU: {}",
            g.params[1]
        );
    }
}
