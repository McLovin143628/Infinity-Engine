//! **What does `gi_intensity = 1.0` mean?** (wave EDIT1, clause 0.)
//!
//! The FIX1 audit measured the showcase island's Play frame seven ways and found
//! the wash is dynamic GI: with GI on nothing in that frame is darker than
//! 91/255 and 13.5 % of it clips to white, while with GI off the same street is
//! legible with real blacks. It could not say *why*, and routed the magnitude.
//! This file answers it with a number instead of a reading.
//!
//! # The white furnace
//!
//! The oldest instrument in rendering. Put a Lambert surface of albedo `ρ` in an
//! environment of uniform radiance `L` and light it with nothing else. The
//! surface must leave exactly `ρ·L`:
//!
//! ```text
//!   E = ∫_hemisphere L cosθ dω = π·L        (irradiance arriving)
//!   L_out = (ρ/π)·E = ρ·L                   (Lambert BRDF is ρ/π)
//! ```
//!
//! At `ρ = 1` the surface must become **indistinguishable from the environment
//! behind it** — it neither creates nor destroys light. So the arm needs no
//! calibration curve, no HDR readback and no tonemap inversion: it renders the
//! same cube twice through the *same* post chain, once white and lit only by the
//! probe field and once black and emitting `L` on its own, and requires the two
//! images to agree.
//!
//! **Not the same furnace as `gi::tests::the_ggx_furnace_test_is_white`**
//! (wave VIS1a), which is a CPU arm about the GGX lobe's multi-scatter
//! compensation — a *specular* energy question with no probes and no renderer in
//! it. This one is about the *ambient* term's units and runs on the GPU, through
//! the whole shipped post chain, because that is where the missing divide was.
//!
//! # What it found
//!
//! `gi_probes.wgsl` projects the gather with `4π/rays`, which is the textbook
//! Monte-Carlo SH projection and is right — `gi::sh_radiance_is_identity_on_a_
//! uniform_field` already pins it. `gi_irradiance` then convolves it with the
//! Ramamoorthi cosine lobe (`A₀ = π`, `A₁ = 2π/3`) and so returns **irradiance**,
//! `E = π·L`. Every lit pass then spends it as if it were exit radiance:
//!
//! ```text
//!   lo += amb * albedo * (1.0 - metallic) * ao;      // mesh.wgsl and five more
//! ```
//!
//! There is no `/π` anywhere on that line, and the same shader's *direct* term
//! two hundred lines above spells the Lambert BRDF out in full — `let diffuse =
//! kd * albedo / PI`. So the ambient half of the engine was **π times** the
//! direct half, per unit of incident radiance, and `gi_intensity = 1.0` meant
//! "π× a normalised gather". The measured factor is printed below: **3.2224**
//! before the fix and **1.0110** after it, against π = 3.14159 — the residue
//! either side is the ambient specular this arm deliberately leaves in.
//!
//! The lit shaders' own comment had already written the obligation down —
//! *"`gi_irradiance` … folds the Lambert 1/π into the caller"* — and no caller
//! ever discharged it.

use glam::{DVec3, Quat, Vec3};
use inf_math::FloatingOrigin;
use inf_render::{
    EngineRenderer, GiSettings, GpuContext, HeadlessTarget, LightKind, MeshInstance, RenderLight,
    RenderScene, RenderSettings, RenderView, HEADLESS_FORMAT,
};

const W: u32 = 128;
const H: u32 = 128;

/// The environment's radiance. Chosen so that both the honest answer (`L`) and
/// the defect's answer (`π·L`) land inside the ACES curve's readable range —
/// neither crushed nor clipped — so the *factor* is measurable and not merely
/// "too bright". At exposure 1.0 these come out around 127 and 202 of 255.
const L: f32 = 0.15;

/// How far the white furnace may miss, in 8-bit levels of the reference frame.
///
/// Not zero, and the reason is a term this arm deliberately does not remove: a
/// dielectric's ambient **specular**, `amb · f0 · 0.5` with `f0 = 0.04`, is 2 %
/// of the ambient on the white cube and ~0 on the black one, because the black
/// one's diffuse is zero but its Fresnel is not. Two percent of `L` is about
/// three levels here. Six is that with room, and the defect this file was
/// written for missed by **71.8**.
const WHITE_FURNACE_CEILING_LSB: f64 = 6.0;

/// The factor the furnace must measure, and how far it may drift. `1.0` is the
/// physical answer; the defect measured 3.2224. The tolerance is set by the
/// 2 % ambient-specular residue (measured: 1.0110), not by adapter noise.
const NORMALISED_GATHER: f64 = 1.0;
const GATHER_FACTOR_TOLERANCE: f64 = 0.10;

fn gpu_or_skip() -> Option<GpuContext> {
    match GpuContext::headless() {
        Ok(gpu) => Some(gpu),
        Err(e) => {
            eprintln!("SKIP gi_normalisation: no GPU adapter ({e})");
            None
        }
    }
}

/// A cube small enough that it perturbs the probe field it is measured against
/// by nothing that matters.
///
/// The probe grid is 16×8×16 over `extent` metres, so at 40 m the probes stand
/// 2.5–5.7 m apart. A 0.3 m cube seen from the nearest probe subtends about
/// 0.009 sr of 4π — **0.07 %** of the sphere — so every probe in the volume sees
/// the uniform sky and only the uniform sky. That is what makes this a furnace
/// and not a study of probe interpolation across a surface: a full-screen floor
/// would have put the fetch point exactly between a probe that sees the sky and
/// a probe the floor has blinded, and measured the grid instead of the gather.
fn furnace_scene(white: bool) -> RenderScene {
    let mut scene = RenderScene {
        grid_enabled: false,
        ..Default::default()
    };
    // Uniform in every direction: zenith == horizon, so `gi_sky_radiance`'s
    // gradient `mix` is a constant and the gather sees a true furnace.
    scene.sky.zenith = [L, L, L];
    scene.sky.horizon = [L, L, L];
    let mut cube = MeshInstance::lit(
        DVec3::ZERO,
        Quat::IDENTITY,
        Vec3::splat(0.3),
        if white {
            [1.0, 1.0, 1.0, 1.0]
        } else {
            [0.0, 0.0, 0.0, 1.0]
        },
        1,
    );
    cube.roughness = 1.0;
    cube.metallic = 0.0;
    if !white {
        cube.emissive = [L, L, L];
    }
    scene.instances.push(cube);
    // **A furnace is lit by its walls, and an EMPTY `lights` vector is not "no
    // light".** `mesh.wgsl` reads `if (count == 0u)` and falls back to a
    // hard-coded editor sun at radiance 3.0 ("so unlit demo scenes still
    // render"), which is why the first cut of this file measured a sunlit cube:
    // the white cube read 192 with GI *off* where the hemispheric ambient alone
    // predicts 87, and the ladder could not bracket it. Zeroing `scene.sun`
    // changed nothing — `SunParams` carries the *direction* the fallback uses,
    // never its magnitude. So the furnace pushes one directional light of
    // intensity 0: `count` becomes 1, the fallback branch is not taken, and
    // `radiance_base = color × intensity` is exactly zero.
    scene.lights.push(RenderLight {
        kind: LightKind::Directional,
        color: [1.0, 1.0, 1.0],
        intensity: 0.0,
        direction: Vec3::Y,
        position: DVec3::ZERO,
        range: 0.0,
        ..RenderLight::default()
    });
    scene.mark_dirty();
    scene
}

/// The same scene with the reference cube emitting `L · scale` — the ladder the
/// measured factor is read off.
fn reference_scene(scale: f32) -> RenderScene {
    let mut scene = furnace_scene(false);
    scene.instances[0].emissive = [L * scale, L * scale, L * scale];
    scene.mark_dirty();
    scene
}

fn view() -> RenderView {
    RenderView {
        origin: FloatingOrigin::new(DVec3::ZERO),
        eye_world: DVec3::new(0.0, 0.0, 1.2),
        forward: Vec3::new(0.0, 0.0, -1.0),
        up: Vec3::Y,
        fov_y: 45f32.to_radians(),
        near: 0.05,
        width: W,
        height: H,
        ortho: None,
    }
}

/// GI as the showcase island ships it: on, at the record default `intensity`.
///
/// `specular: false` so the one term measured here is the **diffuse** ambient —
/// the term every surface in that street spends. The SH specular rides the same
/// scale and is checked by `golden_gi_specular`.
fn gi_on() -> RenderSettings {
    RenderSettings {
        gi: GiSettings {
            enabled: true,
            extent: 40.0,
            rays: 64,
            intensity: 1.0,
            specular: false,
            ..GiSettings::default()
        },
        ..RenderSettings::default()
    }
}

/// Mean luminance of the cube's face — the central patch, well inside the
/// silhouette so no background pixel is averaged in.
fn face(rgba: &[u8]) -> f64 {
    let (mut sum, mut n) = (0.0f64, 0usize);
    for y in (H * 45 / 100)..(H * 55 / 100) {
        for x in (W * 45 / 100)..(W * 55 / 100) {
            let i = ((y * W + x) * 4) as usize;
            sum += 0.2126 * f64::from(rgba[i])
                + 0.7152 * f64::from(rgba[i + 1])
                + 0.0722 * f64::from(rgba[i + 2]);
            n += 1;
        }
    }
    sum / n as f64
}

fn shot(gpu: &GpuContext, scene: &RenderScene, settings: RenderSettings) -> f64 {
    let target = HeadlessTarget::new(gpu, W, H);
    let mut renderer = EngineRenderer::new(gpu, HEADLESS_FORMAT);
    renderer.set_settings(settings);
    // Three frames: the voxelize/probe nodes run inside the frame graph, so the
    // first frame is the one that fills them. `probe_budget = 0` means a full
    // update every frame, so the third is the converged one by construction.
    let mut last = Vec::new();
    for _ in 0..3 {
        renderer.render(gpu, scene, &view(), &target.view, (W, H));
        last = target.read_rgba(gpu).expect("readback");
    }
    face(&last)
}

/// Invert the ladder: the emissive scale whose rendered face matches `want`,
/// found by linear interpolation between the two bracketing rungs. The post
/// chain is monotone in radiance, so the bracket is unique.
fn factor_for(ladder: &[(f64, f64)], want: f64) -> f64 {
    for w in ladder.windows(2) {
        let ((s0, v0), (s1, v1)) = (w[0], w[1]);
        if (v0 - want) * (v1 - want) <= 0.0 && (v1 - v0).abs() > 1.0e-9 {
            return s0 + (s1 - s0) * (want - v0) / (v1 - v0);
        }
    }
    f64::NAN
}

/// **The arm.** A white Lambert cube lit only by a uniform environment must be
/// the environment's own brightness, and `gi_intensity = 1.0` must mean one
/// times a normalised gather.
#[test]
fn a_white_lambert_surface_in_a_uniform_environment_returns_the_environment() {
    let Some(gpu) = gpu_or_skip() else { return };

    let white = shot(&gpu, &furnace_scene(true), gi_on());
    let unit = shot(&gpu, &reference_scene(1.0), gi_on());

    // The ladder, wide enough to bracket both the honest answer and π.
    let scales = [0.5f32, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
    let ladder: Vec<(f64, f64)> = scales
        .iter()
        .map(|&s| {
            (
                f64::from(s),
                shot(&gpu, &reference_scene(s), RenderSettings::default()),
            )
        })
        .collect();

    println!("EDIT1 clause 0 — the white furnace, sky radiance L = {L}");
    for (s, v) in &ladder {
        println!("  emissive reference x{s:<5.3} face {v:7.3}");
    }
    let measured = factor_for(&ladder, white);
    println!("  GI-lit white lambert      face {white:7.3}  -> gather factor {measured:.4}");
    println!("  emissive reference x1.000 face {unit:7.3}  (the same post chain)");

    assert!(
        (white - unit).abs() <= WHITE_FURNACE_CEILING_LSB,
        "the furnace does not close: a white Lambert cube read {white:.3} where the \
         environment it stands in reads {unit:.3} — {:.3} levels against a ceiling of \
         {WHITE_FURNACE_CEILING_LSB}. `gi_intensity = 1.0` is not one times a \
         normalised gather.",
        (white - unit).abs()
    );
    assert!(
        (measured - NORMALISED_GATHER).abs() <= GATHER_FACTOR_TOLERANCE,
        "gi_intensity = 1.0 multiplies the gather by {measured:.4}, not \
         {NORMALISED_GATHER} (tolerance {GATHER_FACTOR_TOLERANCE})"
    );
}

/// The control, and the reason the arm above is about **normalisation** and not
/// about GI being on at all: the same cube with GI off is not the furnace, and
/// must not be. Nothing here should ever make a GI-off scene move.
#[test]
fn the_furnace_is_a_statement_about_the_gather_and_not_about_the_ambient_fallback() {
    let Some(gpu) = gpu_or_skip() else { return };
    let off = shot(&gpu, &furnace_scene(true), RenderSettings::default());
    let unit = shot(&gpu, &reference_scene(1.0), RenderSettings::default());
    println!("  GI OFF, white lambert     face {off:7.3} (hemispheric constant ambient)");
    println!("  emissive reference x1.000 face {unit:7.3}");
    assert!(
        off < unit,
        "with no probe field the authored hemispheric ambient is a dim constant, not \
         the environment: {off:.3} vs {unit:.3}"
    );
}

// ── the lit street, and the ceiling the island's frame is judged by ──────────

/// How far switching dynamic GI ON may lift a daylight street's 95th percentile
/// luminance, in 8-bit levels.
///
/// **A LIFT and not an absolute**, and that is the whole design of this arm. The
/// FIX1 audit's table is a pair of readings of one frame — p95 **246.6** with GI
/// on against **169.7** with it off, `frac>240` **0.135** against **0.000** —
/// and what it indicts is the DIFFERENCE. An absolute ceiling would instead be
/// measuring how bright this fixture's own sun is, which is a number a fixture
/// author picks and has nothing to do with the defect. The first cut of this
/// file used absolutes and failed on a tree the furnace says is correct, because
/// the street it invented is sunnier than the island.
///
/// Bounce light is real and it must be allowed to lift a frame. Measured on this
/// street: **+25.1** with the units fixed, **+49.4** with the pre-EDIT1 ones.
///
/// Thirty and not twenty, and the reason is a term this wave did NOT fix: the
/// gather's hit radiance carries no `n·l`, because a voxel stores an albedo and
/// no normal. That makes every bounce an over-estimate by the average of a
/// cosine over the lit hemisphere rather than by a factor of π — a modelling
/// approximation of single-bounce voxel GI, not a unit error, and carried with
/// its number rather than hidden inside a ceiling chosen to fit it.
const GI_P95_LIFT_CEILING: f64 = 30.0;

/// How much of the frame dynamic GI may push over 240 that was not there
/// already. The audit measured **+0.135** of the whole frame; a fiftieth is a
/// generous reading of "a bounce brightened some sunlit walls".
///
/// **This is the sharp one.** Measured on this street: **+0.0000** with the
/// units fixed and **+0.1048** with the pre-EDIT1 ones. Bounce light is allowed
/// to brighten a frame and is not allowed to destroy it, and the difference
/// between those two readings is the difference between a lit street and the
/// author's screenshot.
const GI_CLIPPED_LIFT_CEILING: f64 = 0.02;

const STREET_W: u32 = 320;
const STREET_H: u32 = 180;

/// The island's own render block, as `RenderSettingsRecord::lit_showcase`
/// authors it and `apply_record` maps it: shadows, GI, bloom, SSAO, TAA and
/// flare, with `gi_intensity` at the record default.
///
/// `gi_intensity` is `Option` so one function serves all three rows: `None` is
/// the GI-off baseline, `Some(1.0)` is this tree, and `Some(π)` is the frame as
/// it shipped — putting the factor back into the level value is exactly the
/// arithmetic this wave took out of `gi_irradiance`.
fn lit_showcase_settings(gi_intensity: Option<f32>) -> RenderSettings {
    RenderSettings {
        gi: GiSettings {
            enabled: gi_intensity.is_some(),
            intensity: gi_intensity.unwrap_or(1.0),
            ..GiSettings::default()
        },
        shadows: inf_render::ShadowSettings {
            enabled: true,
            ..inf_render::ShadowSettings::default()
        },
        bloom: inf_render::BloomSettings {
            enabled: true,
            ..inf_render::BloomSettings::default()
        },
        ssao: inf_render::SsaoSettings {
            enabled: true,
            ..inf_render::SsaoSettings::default()
        },
        taa: true,
        flare: inf_render::FlareSettings {
            enabled: true,
            ..inf_render::FlareSettings::default()
        },
        ..RenderSettings::default()
    }
}

/// A daylight street: pale ground, two rows of pale blocks, one sun.
///
/// **A fixture, and the report says so.** It is not the showcase island — it has
/// no atmosphere LUT, no meshlets, no virtual textures and no city. What it does
/// have is the thing the FIX1 audit's table was about: a lot of unoccluded sky
/// over a lot of pale albedo, which is the configuration in which an ambient
/// term that is π times too strong stops being a look and becomes a wash. The
/// real-host reading is the demo loop's own frame, scored in the ledger.
fn lit_street() -> RenderScene {
    let mut scene = RenderScene {
        grid_enabled: false,
        ..Default::default()
    };
    // **The engine's OWN sky**, left at `SkyParams::default()` and not authored
    // brighter. The first cut of this fixture set the zenith to 0.35-0.85 linear
    // radiance under a sun of intensity 3, and measured a 43-level lift on a
    // tree the white furnace says is exactly normalised -- because a sky that
    // bright IS a second sun, and a physically correct ambient term will spend
    // it. That reading was the fixture's exposure, not the engine's units, and
    // it is exactly the mistake an absolute ceiling would have enshrined.
    scene.instances.push(MeshInstance::lit(
        DVec3::new(0.0, -0.5, 0.0),
        Quat::IDENTITY,
        Vec3::new(60.0, 1.0, 400.0),
        [0.42, 0.46, 0.38, 1.0],
        1,
    ));
    let mut id = 2;
    for k in 0..24 {
        let z = 70.0 - f64::from(k) * 14.0;
        for side in [-7.0, 7.0] {
            let mut b = MeshInstance::lit(
                DVec3::new(side, 6.0, z),
                Quat::IDENTITY,
                Vec3::new(6.0, 14.0, 10.0),
                [0.80, 0.82, 0.78, 1.0],
                id,
            );
            b.roughness = 0.85;
            scene.instances.push(b);
            id += 1;
        }
    }
    scene.lights.push(RenderLight {
        kind: LightKind::Directional,
        color: [1.0, 0.97, 0.9],
        intensity: 3.0,
        direction: Vec3::new(0.35, 0.85, 0.4).normalize(),
        position: DVec3::ZERO,
        range: 0.0,
        ..RenderLight::default()
    });
    scene.mark_dirty();
    scene
}

/// p95 luminance and the fraction over 240 — the two numbers the FIX1 audit
/// scored `03-pie-b.png` with, so this table reads beside that one.
fn frame_stats(rgba: &[u8]) -> (f64, f64) {
    let mut lum: Vec<f64> = rgba
        .chunks_exact(4)
        .map(|p| 0.2126 * f64::from(p[0]) + 0.7152 * f64::from(p[1]) + 0.0722 * f64::from(p[2]))
        .collect();
    let hot = lum.iter().filter(|l| **l > 240.0).count() as f64 / lum.len() as f64;
    lum.sort_by(f64::total_cmp);
    let p95 = lum[(lum.len() as f64 * 0.95) as usize];
    (p95, hot)
}

fn street_shot(gpu: &GpuContext, gi_intensity: Option<f32>) -> (f64, f64) {
    let scene = lit_street();
    let target = HeadlessTarget::new(gpu, STREET_W, STREET_H);
    let mut renderer = EngineRenderer::new(gpu, HEADLESS_FORMAT);
    renderer.set_settings(lit_showcase_settings(gi_intensity));
    let view = RenderView {
        origin: FloatingOrigin::new(DVec3::ZERO),
        eye_world: DVec3::new(0.0, 1.6, 60.0),
        forward: Vec3::new(0.0, 0.0, -1.0),
        up: Vec3::Y,
        fov_y: 60f32.to_radians(),
        near: 0.05,
        width: STREET_W,
        height: STREET_H,
        ortho: None,
    };
    let mut last = Vec::new();
    // Eight frames: TAA is on, so the first few are a converging history.
    for _ in 0..8 {
        renderer.render(gpu, &scene, &view, &target.view, (STREET_W, STREET_H));
        last = target.read_rgba(gpu).expect("readback");
    }
    frame_stats(&last)
}

/// **The frame arm** (wave EDIT1, clause 0): switching dynamic GI on must add
/// bounce to a daylight street, not a wash — and the control is the value
/// `gi_intensity = 1.0` used to mean.
#[test]
fn dynamic_gi_lights_a_daylight_street_rather_than_washing_it_out() {
    let Some(gpu) = gpu_or_skip() else { return };
    // The baseline: the same street with GI off. This is the audit's own second
    // row, and it is what the lift is measured from.
    let (p95_off, hot_off) = street_shot(&gpu, None);
    let (p95_on, hot_on) = street_shot(&gpu, Some(1.0));
    // The row EDIT1 controlled with: putting π back into the intensity was
    // EXACTLY the arithmetic that wave removed from `gi_irradiance`.
    let (p95_old, hot_old) = street_shot(&gpu, Some(std::f32::consts::PI));
    println!("EDIT1 clause 0 - the lit street under `lit_showcase`");
    // Aligned with width specifiers rather than runs of spaces inside the
    // literal: `inf_packager`'s eaten-continuation sweep reads such a run as the
    // defect it usually is, and it is right to — see this wave's ledger.
    let row = |label: &str, p95: f64, hot: f64, base: Option<(f64, f64)>| {
        let lift = base.map_or(String::new(), |(bp, bh)| {
            format!("  lift {:+.1} / {:+.4}", p95 - bp, hot - bh)
        });
        println!("  {label:<30} p95 {p95:6.1}  frac>240 {hot:.4}{lift}");
    };
    row("GI off", p95_off, hot_off, None);
    row(
        "GI on, intensity 1.0",
        p95_on,
        hot_on,
        Some((p95_off, hot_off)),
    );
    row(
        "GI on, intensity pi",
        p95_old,
        hot_old,
        Some((p95_off, hot_off)),
    );
    assert!(
        p95_on - p95_off <= GI_P95_LIFT_CEILING,
        "GI lifted p95 by {:.1} levels, over the {GI_P95_LIFT_CEILING} ceiling",
        p95_on - p95_off
    );
    assert!(
        hot_on - hot_off <= GI_CLIPPED_LIFT_CEILING,
        "GI pushed {:.4} more of the frame over 240, past {GI_CLIPPED_LIFT_CEILING}",
        hot_on - hot_off
    );
    // **THE FALSIFICATION LEFT THIS FILE IN WAVE FIX3, AND THAT IS A RESULT.**
    //
    // EDIT1 ended here with `assert!(p95_old - p95_off > CEILING || ...)` — the
    // π row had to TRIP the ceilings, or they were not measuring anything. It
    // does not any more, and the reason is the shape of the fix rather than a
    // weakening: since FIX3 the ambient's SCALE is set by `sky_irradiance`,
    // which `gi_intensity` does not touch, and what `gi_intensity` multiplies is
    // a signed DIFFERENCE from the open sky — so π darkens the occluded half
    // exactly as fast as it brightens the bounce. Measured on this street: the π
    // row lifts p95 by **+13.1** where it lifted it by +49.4 before, and clips
    // nothing at all. "π times the ambient" is no longer a state this renderer
    // can be put in from a fixture.
    //
    // Every remaining way to wash the street out needs a SHADER edit, so the
    // falsification moved to mutation plus source gates, and both were measured
    // rather than argued. Mutating `gi_probes.wgsl` back to the pre-FIX3 gather
    // — `ndl = 1.0` and the sky restored to the miss term, i.e. the cosine gone
    // and the sky counted twice — and changing nothing else:
    //
    // ```text
    //   GI on, intensity 1.0   p95 221.6   lift +28.4 / +0.0000
    //   GI on, intensity pi    p95 243.9   lift +50.7 / +0.1393   <- trips BOTH
    // ```
    //
    // So the ceilings still see the defect they were built for; what changed is
    // that reaching it needs the shader. The two lines that would do it are
    // pinned by `inf_render::passes::gi::sky_sh_tests::the_probe_miss_term_is_
    // zero`, and the magnitude of the ambient itself — in BOTH directions, since
    // that arm has a maximum as well as a minimum — by
    // `tests/sky_ambient.rs`.
    //
    // What stays here is the non-vacuity check EDIT1's control also provided:
    // these ceilings must not be met by a renderer in which switching GI on does
    // nothing at all.
    assert!(
        (p95_on - p95_off).abs() > 1.0,
        "switching dynamic GI on moved this street's p95 by {:.2} levels — the \
          ceilings above would be met by a renderer with no GI in it",
        p95_on - p95_off
    );
}
