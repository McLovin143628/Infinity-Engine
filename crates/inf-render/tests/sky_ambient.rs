//! **A SHADED WALL UNDER A CLEAR NOON SKY** (wave FIX3, clauses 1 and 3).
//!
//! # The reading this file was written for
//!
//! The FIX2 audit measured the showcase island's hero three ways at one view:
//! **232/255 in the editor viewport, 1.8 in Play, 1.7–2.0 in the shipped
//! player**. In the same shipped frame the hero's sunlit rim reads 166 and a
//! building's shaded wall reads **8.5**. Play and shipping agree; every surface
//! facing away from the sun is *black* in both of them.
//!
//! `gi_normalisation.rs` already answers "what does `gi_intensity = 1.0` mean" —
//! one times a normalised gather, measured 1.0110. It cannot answer this one,
//! because a furnace has no sun: it is a statement about the ambient term's
//! **units** and this is a statement about the ambient term's **existence**.
//!
//! # The instrument
//!
//! A white Lambert wall standing in the open under the P17 atmosphere at noon,
//! photographed twice from the same distance — once from the sun's side and once
//! from the other. Both faces see the same half-sky; only one sees the sun. The
//! physical expectation is a textbook one:
//!
//! ```text
//!   direct normal irradiance, clear noon   ≈ 100 klux
//!   diffuse horizontal irradiance          ≈  10–20 klux
//!   ⇒ a vertical surface facing away from the sun receives roughly half the
//!     diffuse (≈ 5–10 klux) plus what the ground bounces back, against the
//!     ≈ 77 klux + diffuse the sunlit face receives.
//!   ⇒ shaded : sunlit ≈ 0.1 – 0.2
//! ```
//!
//! **The readings are inverted back to scene radiance before the ratio is
//! taken.** ACES is not linear, so a ratio of 8-bit codes is a ratio of nothing;
//! [`Ladder`] measures the post chain's own radiance→code curve with a
//! full-screen emitter at fourteen known radiances and inverts each face
//! reading through it. Bloom, TAA and flare are off in this fixture precisely so
//! that the chain is a per-pixel function and one curve serves every frame.
//!
//! # What it found at `e8451338`
//!
//! The table this file prints, and the ledger's "## Wave FIX3" quotes it.

use glam::{DVec3, Quat, Vec3};
use inf_math::FloatingOrigin;
use inf_render::{
    AtmosphereParams, EngineRenderer, GiSettings, GpuContext, HeadlessTarget, LightKind,
    MeshInstance, RenderLight, RenderScene, RenderSettings, RenderView, ShadowSettings, SunParams,
    HEADLESS_FORMAT,
};

const W: u32 = 256;
const H: u32 = 192;

/// Sun elevation for the fixture, degrees above the horizon.
///
/// Not the zenith: a sun straight overhead leaves *every* vertical wall shaded
/// and the instrument has nothing to compare. Forty degrees is a mid-morning sun
/// at the island's latitude and gives the sunlit face `cos(50°) = 0.643` of the
/// normal irradiance — high enough that the shaded reading is a small fraction
/// of it and low enough that both faces are genuinely vertical walls.
const SUN_ELEVATION_DEG: f32 = 40.0;

/// The engine's sun "intensity" for the fixture — the island's own value
/// (`inf_editor_core::island`, the Sun entity's `Light`) is 3.2 and the
/// atmosphere goldens use 3.0. Three, so this fixture reads beside `tod_scene`.
const SUN_INTENSITY: f32 = 3.0;

fn gpu_or_skip() -> Option<GpuContext> {
    match GpuContext::headless() {
        Ok(gpu) => Some(gpu),
        Err(e) => {
            eprintln!("SKIP sky_ambient: no GPU adapter ({e})");
            None
        }
    }
}

/// Unit vector toward the sun: due +Z on the horizontal, raised by
/// [`SUN_ELEVATION_DEG`]. Chosen axis-aligned so the wall needs no rotation and
/// its two large faces have normals exactly `+Z` (sunlit) and `-Z` (shaded).
fn sun_dir() -> Vec3 {
    let e = SUN_ELEVATION_DEG.to_radians();
    Vec3::new(0.0, e.sin(), e.cos())
}

/// The settings every shot in this file renders with, plus whatever the caller
/// varies.
///
/// Bloom, TAA, flare and film grain stay **off**: each of them makes a pixel a
/// function of its neighbours or of the frames before it, and the ladder below
/// depends on the post chain being a per-pixel function of radiance. Shadows
/// stay **on**, because the ground in front of the shaded wall is in that wall's
/// shadow and the bounce it returns is part of the physical answer.
fn base_settings() -> RenderSettings {
    RenderSettings {
        shadows: ShadowSettings {
            enabled: true,
            max_distance: 250.0,
            ..ShadowSettings::default()
        },
        ..RenderSettings::default()
    }
}

fn gi_on() -> RenderSettings {
    RenderSettings {
        gi: GiSettings {
            enabled: true,
            intensity: 1.0,
            ..GiSettings::default()
        },
        ..base_settings()
    }
}

/// The wall, the ground and the sun — under the physical atmosphere unless
/// `atmosphere` says otherwise.
///
/// `aerial_perspective: 0.0` because this fixture measures a **surface**, not
/// the air in front of it: at 18 m the physical in-scattering is small, but it
/// is not zero, and leaving it in would put a distance-dependent additive term
/// inside a ratio the whole file is about.
fn wall_scene(atmosphere: bool, sky: Option<([f32; 3], [f32; 3])>) -> RenderScene {
    let mut scene = RenderScene {
        grid_enabled: false,
        sun: SunParams {
            direction: sun_dir(),
            color: [1.0, 0.98, 0.95],
            intensity: SUN_INTENSITY,
            ..SunParams::default()
        },
        atmosphere: AtmosphereParams {
            enabled: atmosphere,
            aerial_perspective: 0.0,
            ..AtmosphereParams::default()
        },
        ..Default::default()
    };
    // The authored two-colour gradient, when the caller wants the no-atmosphere
    // source measured. `SkyParams::default()` is deliberately NOT a daylight sky
    // — its own doc calls it "editor-dark, tuned to the infinity-dark theme" —
    // so a fixture that leaves it there and then asks a *noon* question is
    // asking about a UI colour. Both are measured below, for different claims.
    if let Some((horizon, zenith)) = sky {
        scene.sky.horizon = horizon;
        scene.sky.zenith = zenith;
    }
    // The ground: a wide, mid-grey slab. Its bounce is part of what a shaded
    // wall legitimately receives, so it is in the fixture rather than excluded
    // from it.
    scene.instances.push(ground());
    // The wall: 30 m wide, 14 m tall, 2 m thick, standing on the ground. White
    // and fully rough, so what it leaves is `albedo × (incident radiance)` and
    // nothing else.
    let mut wall = MeshInstance::lit(
        DVec3::new(0.0, 7.0, 0.0),
        Quat::IDENTITY,
        Vec3::new(30.0, 14.0, 2.0),
        [0.9, 0.9, 0.9, 1.0],
        2,
    );
    wall.roughness = 1.0;
    wall.metallic = 0.0;
    scene.instances.push(wall);
    // The scene's own sun, as an analytic light. **A non-empty `lights` list is
    // load-bearing**: `mesh.wgsl` falls back to a hard-coded editor sun of
    // radiance 3.0 when the list is empty, which is a different sun from the one
    // the sky is drawn for (`gi_normalisation.rs` documents the same trap).
    scene.lights.push(RenderLight {
        kind: LightKind::Directional,
        color: [1.0, 0.98, 0.95],
        intensity: SUN_INTENSITY,
        direction: sun_dir(),
        position: DVec3::ZERO,
        range: 0.0,
        ..RenderLight::default()
    });
    scene.mark_dirty();
    scene
}

fn ground() -> MeshInstance {
    let mut g = MeshInstance::lit(
        DVec3::new(0.0, -0.5, 0.0),
        Quat::IDENTITY,
        Vec3::new(400.0, 1.0, 400.0),
        [0.35, 0.35, 0.35, 1.0],
        1,
    );
    g.roughness = 1.0;
    g.metallic = 0.0;
    g
}

/// A camera `dist` metres from the wall on the `+Z` (sunlit) or `-Z` (shaded)
/// side, framed so the wall subtends the same solid angle either way — the ratio
/// this file reports is only meaningful if both readings come off the same
/// number of wall pixels.
fn wall_view(dist: f64, sunlit_side: bool) -> RenderView {
    let z = if sunlit_side { dist } else { -dist };
    let forward = if sunlit_side {
        Vec3::new(0.0, 0.0, -1.0)
    } else {
        Vec3::new(0.0, 0.0, 1.0)
    };
    // Half the wall's height over the distance to its face: the wall fills the
    // frame vertically at every distance.
    let half = 7.0;
    let face = dist - 1.0;
    RenderView {
        origin: FloatingOrigin::new(DVec3::ZERO),
        eye_world: DVec3::new(0.0, 7.0, z),
        forward,
        up: Vec3::Y,
        fov_y: 2.0 * (half / face).atan() as f32,
        near: 0.05,
        width: W,
        height: H,
        ortho: None,
    }
}

fn shot(gpu: &GpuContext, scene: &RenderScene, view: &RenderView, s: RenderSettings) -> Vec<u8> {
    let target = HeadlessTarget::new(gpu, W, H);
    let mut renderer = EngineRenderer::new(gpu, HEADLESS_FORMAT);
    renderer.set_settings(s);
    let mut last = Vec::new();
    // Four frames: the voxelize/probe nodes run inside the frame graph, so the
    // first is the one that fills them; `probe_budget = 0` is a full update
    // every frame, so the last is converged by construction.
    for _ in 0..4 {
        renderer.render(gpu, scene, view, &target.view, (W, H));
        last = target.read_rgba(gpu).expect("readback");
    }
    last
}

/// Mean luminance (0..255) of the central patch — well inside the wall's
/// silhouette, so no ground or sky pixel is averaged in.
fn patch(rgba: &[u8]) -> f64 {
    let (mut sum, mut n) = (0.0f64, 0usize);
    for y in (H * 35 / 100)..(H * 60 / 100) {
        for x in (W * 35 / 100)..(W * 65 / 100) {
            let i = ((y * W + x) * 4) as usize;
            sum += 0.2126 * f64::from(rgba[i])
                + 0.7152 * f64::from(rgba[i + 1])
                + 0.0722 * f64::from(rgba[i + 2]);
            n += 1;
        }
    }
    sum / n as f64
}

// ── inverting the post chain ────────────────────────────────────────────────

/// **The radiance→code curve of the shipped post chain, measured.**
///
/// With bloom, TAA, flare and grain off, `exposure → ACES → sRGB encode →
/// dither` is a pure per-pixel function of scene radiance, so one curve measured
/// on any scene inverts a patch mean from any other rendered at the same
/// exposure. Dither is left ON — it is ±½ LSB of blue noise and averaging a
/// 76×48 patch removes it — because turning it off would measure a chain the
/// engine does not ship.
struct Ladder(Vec<(f64, f64)>);

impl Ladder {
    /// A black, fully-rough emitter filling the frame at each of fourteen
    /// radiances. Emissive is added after the BRDF and before the tonemap, so
    /// the patch reads exactly that radiance through exactly the shipped chain.
    fn measure(gpu: &GpuContext) -> Self {
        let mut rungs = Vec::new();
        for &l in &[
            0.0f32, 0.002, 0.005, 0.01, 0.02, 0.04, 0.07, 0.11, 0.16, 0.25, 0.4, 0.6, 0.9, 1.4,
            2.2, 3.5,
        ] {
            let mut scene = RenderScene {
                grid_enabled: false,
                ..Default::default()
            };
            let mut slab = MeshInstance::lit(
                DVec3::ZERO,
                Quat::IDENTITY,
                Vec3::new(60.0, 60.0, 1.0),
                [0.0, 0.0, 0.0, 1.0],
                1,
            );
            slab.roughness = 1.0;
            slab.emissive = [l, l, l];
            scene.instances.push(slab);
            // The zero-intensity light that keeps `mesh.wgsl`'s empty-list
            // fallback sun out of the measurement.
            scene.lights.push(RenderLight {
                kind: LightKind::Directional,
                color: [0.0; 3],
                intensity: 0.0,
                direction: Vec3::Y,
                position: DVec3::ZERO,
                range: 0.0,
                ..RenderLight::default()
            });
            scene.mark_dirty();
            let view = RenderView {
                origin: FloatingOrigin::new(DVec3::ZERO),
                eye_world: DVec3::new(0.0, 0.0, 10.0),
                forward: Vec3::new(0.0, 0.0, -1.0),
                up: Vec3::Y,
                fov_y: 45f32.to_radians(),
                near: 0.05,
                width: W,
                height: H,
                ortho: None,
            };
            let img = shot(gpu, &scene, &view, base_settings());
            rungs.push((f64::from(l), patch(&img)));
        }
        Self(rungs)
    }

    /// The scene radiance whose rendered code is `code`, by linear interpolation
    /// between the bracketing rungs. The chain is monotone, so the bracket is
    /// unique; a reading above the top rung saturates and returns `NaN` rather
    /// than a number the caller would believe.
    fn radiance(&self, code: f64) -> f64 {
        for w in self.0.windows(2) {
            let ((l0, c0), (l1, c1)) = (w[0], w[1]);
            if (c0 - code) * (c1 - code) <= 0.0 && (c1 - c0).abs() > 1.0e-9 {
                return l0 + (l1 - l0) * (code - c0) / (c1 - c0);
            }
        }
        f64::NAN
    }
}

// ── the readings ────────────────────────────────────────────────────────────

/// One configuration's answer: both faces, in scene radiance, and their ratio.
struct Faces {
    sunlit_code: f64,
    shaded_code: f64,
    sunlit: f64,
    shaded: f64,
}

impl Faces {
    fn ratio(&self) -> f64 {
        self.shaded / self.sunlit
    }
}

fn faces(
    gpu: &GpuContext,
    ladder: &Ladder,
    scene: &RenderScene,
    dist: f64,
    s: RenderSettings,
) -> Faces {
    let sunlit_code = patch(&shot(gpu, scene, &wall_view(dist, true), s));
    let shaded_code = patch(&shot(gpu, scene, &wall_view(dist, false), s));
    Faces {
        sunlit_code,
        shaded_code,
        sunlit: ladder.radiance(sunlit_code),
        shaded: ladder.radiance(shaded_code),
    }
}

/// The physical band a shaded vertical wall's exit radiance must fall in,
/// relative to the same wall's sunlit face, at noon under a clear sky.
///
/// Derived above from the standard clear-sky split (≈ 100 klux direct normal
/// against ≈ 10–20 klux diffuse horizontal) and widened at the top for the
/// ground bounce this fixture deliberately includes: a 0.35-albedo slab under
/// that sun returns enough to a vertical wall to carry the honest answer to the
/// high teens. **0.35 and not 0.20** because the wall is white (0.9) over a grey
/// ground and single-bounce inter-reflection between the two is real.
const SHADE_RATIO_MIN: f64 = 0.07;
const SHADE_RATIO_MAX: f64 = 0.35;

/// **THE ARM.** Every configuration the engine ships a lit level in must put a
/// shaded wall inside the physical band — with dynamic GI on and with it off,
/// with the wall inside the probe volume and beyond it.
///
/// Mutation-verified: reverting `mesh.wgsl`'s ambient line to `amb =
/// gi_irradiance(...)` (the replacement this wave turned into an addition) fails
/// the `gi on, far` row at 0.0xx, which is the frame the FIX2 audit
/// photographed.
#[test]
fn a_shaded_wall_under_a_clear_noon_sky_is_a_tenth_of_a_sunlit_one() {
    let Some(gpu) = gpu_or_skip() else { return };
    let ladder = Ladder::measure(&gpu);
    eprintln!("FIX3 clause 1 — the post chain's radiance ladder (exposure 1.0)");
    for (l, c) in &ladder.0 {
        eprintln!("  radiance {l:6.3}  ->  code {c:7.3}");
    }

    // A daylight gradient for the no-atmosphere rows. The numbers are not an
    // art direction: they are chosen so the gradient's mean radiance over the
    // sphere equals the physical medium's at this sun elevation — the DC term
    // `atmosphere::tests::sky_sh_converges` measures, [0.157, 0.193, 0.246] —
    // so the two rows differ in their SOURCE and not in how bright a sky they
    // are asking about. An author typing a sky by eye would type something
    // brighter and get a brighter world, which is correct and is not what this
    // row is for.
    const DAY_HORIZON: [f32; 3] = [0.22, 0.26, 0.30];
    const DAY_ZENITH: [f32; 3] = [0.09, 0.13, 0.19];
    let atmos = wall_scene(true, None);
    let daylight = wall_scene(false, Some((DAY_HORIZON, DAY_ZENITH)));
    let dark = wall_scene(false, None);
    // NEAR: the wall stands inside the 40 m camera-centred probe volume.
    // FAR: the camera is 60 m back, so every probe the wall's fetch can reach is
    // a clamped boundary probe — the configuration the island's Play frame is,
    // for every building further away than half the volume.
    let rows: Vec<(&str, Faces)> = vec![
        (
            "atmosphere, GI off, near 18 m",
            faces(&gpu, &ladder, &atmos, 18.0, base_settings()),
        ),
        (
            "atmosphere, GI on,  near 18 m",
            faces(&gpu, &ladder, &atmos, 18.0, gi_on()),
        ),
        (
            "atmosphere, GI on,  far  60 m",
            faces(&gpu, &ladder, &atmos, 60.0, gi_on()),
        ),
        (
            "daylight gradient, GI off, near",
            faces(&gpu, &ladder, &daylight, 18.0, base_settings()),
        ),
        (
            "daylight gradient, GI on,  near",
            faces(&gpu, &ladder, &daylight, 18.0, gi_on()),
        ),
    ];
    // The control, scored separately because it is a different claim: the
    // engine's OWN `SkyParams::default()` is a dark editor gradient, and a wall
    // under it must come out darker than one under a daylight sky. That is the
    // property this wave actually added — the ambient is a function of the
    // scene's sky rather than of a constant — and asserting the physical band on
    // a sky that is not daylight would be asserting that the fixture lies.
    let dark_row = faces(&gpu, &ladder, &dark, 18.0, gi_on());

    eprintln!("FIX3 clause 1 — a white wall at noon, sunlit face vs shaded face");
    eprintln!(
        "  {:<30} {:>17} {:>17} {:>8}",
        "", "sunlit", "shaded", "ratio"
    );
    for (name, f) in &rows {
        eprintln!(
            "  {name:<30} {:>7.4} @{:5.1} {:>7.4} @{:5.1} {:>8.4}",
            f.sunlit,
            f.sunlit_code,
            f.shaded,
            f.shaded_code,
            f.ratio()
        );
    }

    eprintln!(
        "  {:<30} {:>7.4} @{:5.1} {:>7.4} @{:5.1} {:>8.4}   (control)",
        "engine-default dark gradient",
        dark_row.sunlit,
        dark_row.sunlit_code,
        dark_row.shaded,
        dark_row.shaded_code,
        dark_row.ratio()
    );

    let mut bad = Vec::new();
    for (name, f) in &rows {
        let r = f.ratio();
        if !(SHADE_RATIO_MIN..=SHADE_RATIO_MAX).contains(&r) {
            bad.push(format!("{name}: {r:.4}"));
        }
    }
    assert!(
        bad.is_empty(),
        "a shaded wall under a clear noon sky must read {SHADE_RATIO_MIN}–{SHADE_RATIO_MAX} of a \
         sunlit one; these configurations do not: {bad:?}"
    );

    // …and the ambient is the SCENE's sky and not a constant: the same wall
    // under the engine's dark editor gradient is materially darker in the shade.
    let daylight_ratio = rows
        .iter()
        .find(|(n, _)| n.starts_with("daylight gradient, GI on"))
        .map(|(_, f)| f.ratio())
        .expect("the daylight row");
    assert!(
        dark_row.ratio() < daylight_ratio * 0.75,
        "the shaded wall reads {:.4} under a dark authored sky and {daylight_ratio:.4} under a \
         daylight one — the ambient term is not tracking the scene's sky",
        dark_row.ratio()
    );
}
