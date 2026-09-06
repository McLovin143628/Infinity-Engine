//! **DOES A METAL REFLECT THE SKY, AND DOES A SKINNED METAL REFLECT THE SAME
//! SKY?** — the wave CHAR1a audit's priority (b).
//!
//! # The claim this file was written to test
//!
//! Wave CHAR1a.2 measured the engine's Manny as *"a dark glossy android with
//! white panel seams"* against UE's light-grey one, and explained it:
//!
//! > `MSR_MSK`'s metal mask has median 247, so more than half the mannequin's
//! > surface is flagged `metallic = 1`, and **this engine has no environment
//! > specular for metals — a metal with nothing to reflect is black**.
//!
//! …and routed it to PAR0/PAR5 as a missing feature. The second half of that
//! sentence is the part worth measuring, because a metal with `metallic = 1`
//! has **no diffuse term at all** (`lo += amb * albedo * (1 - metallic)` zeroes
//! it), so if the ambient specular were genuinely absent a metal really would be
//! black away from the sun's mirror direction, and *that* would be the whole
//! explanation of a dark character.
//!
//! `env_lighting.wgsl` says otherwise on inspection — `gi_ambient_specular` has
//! three layers (the pre-P18.4 constant `ambient · f0 · 0.5`, P18.4's SH
//! reconstruction along the reflection vector *including an explicit sky term
//! since FIX3*, and VIS1a's SSR over the top) — and `mesh.wgsl` and
//! `skinned_mesh.wgsl` call it with identical arguments. Inspection is not
//! measurement, so this file renders it.
//!
//! # The instrument
//!
//! One sphere, `metallic = 1`, `roughness = 0.3`, under the P17 atmosphere with
//! GI on — rendered twice: once as a rigid [`MeshInstance`] on
//! [`PrimMesh::Sphere`], once as a [`SkinnedInstance`] over **the same
//! `sphere_geometry()` vertices** on a one-joint identity palette. Same camera,
//! same lights, same settings, same silhouette. What is compared is the median
//! luminance of the sphere's own pixels.
//!
//! Two questions, two assertions:
//!
//! 1. **Is a metal black?** Both medians against a floor. A body-coloured metal
//!    under a bright sky that reads near zero is the claim as written.
//! 2. **Do the two paths agree?** The rigid and skinned medians against each
//!    other. A skinned-path divergence here would be one door for both hosts and
//!    a fix; agreement says the lighting a character receives is the lighting
//!    everything else receives, and any remaining darkness is a property of the
//!    *material*, not of the skinned path.

use glam::{DVec3, Mat4, Quat, Vec3};
use inf_math::FloatingOrigin;
use inf_render::{
    primitives::sphere_geometry, AtmosphereParams, EngineRenderer, GiSettings, GpuContext,
    HeadlessTarget, LightKind, MeshInstance, PrimMesh, RenderLight, RenderScene, RenderSettings,
    RenderView, ShadowSettings, SkinnedInstance, SkinnedMeshData, SkinnedVertex, SunParams,
    HEADLESS_FORMAT,
};

const W: u32 = 256;
const H: u32 = 256;
/// Mid-morning, so the sphere has a lit side, a terminator and a shaded side —
/// and the shaded side is where "does a metal reflect the sky" is answered.
const SUN_ELEVATION_DEG: f32 = 40.0;
/// The island's own sun is 3.2; the atmosphere goldens use 3.0.
const SUN_INTENSITY: f32 = 3.0;

fn gpu_or_skip() -> Option<GpuContext> {
    match GpuContext::headless() {
        Ok(gpu) => Some(gpu),
        Err(e) => {
            eprintln!("SKIP metal_sky: no GPU adapter ({e})");
            None
        }
    }
}

fn sun_dir() -> Vec3 {
    let e = SUN_ELEVATION_DEG.to_radians();
    Vec3::new(0.0, e.sin(), e.cos())
}

fn settings() -> RenderSettings {
    RenderSettings {
        shadows: ShadowSettings {
            enabled: true,
            max_distance: 250.0,
            ..ShadowSettings::default()
        },
        gi: GiSettings {
            enabled: true,
            intensity: 1.0,
            ..GiSettings::default()
        },
        ..RenderSettings::default()
    }
}

/// The sky, the sun and nothing else — the caller adds the sphere.
///
/// No ground slab: a ground would bounce light onto the sphere's underside and
/// the question here is what the SKY delivers. `aerial_perspective: 0.0` for the
/// reason `sky_ambient.rs` gives — this measures a surface, not the air.
fn empty_scene() -> RenderScene {
    let mut scene = RenderScene {
        grid_enabled: false,
        sun: SunParams {
            direction: sun_dir(),
            color: [1.0, 0.98, 0.95],
            intensity: SUN_INTENSITY,
            ..SunParams::default()
        },
        atmosphere: AtmosphereParams {
            enabled: true,
            aerial_perspective: 0.0,
            ..AtmosphereParams::default()
        },
        ..Default::default()
    };
    // **A non-empty light list is load-bearing**: both lit shaders fall back to a
    // hard-coded editor sun of radiance 3.0 when it is empty, which is a
    // different sun from the one the sky is drawn for.
    scene.lights.push(RenderLight {
        kind: LightKind::Directional,
        color: [1.0, 0.98, 0.95],
        intensity: SUN_INTENSITY,
        direction: sun_dir(),
        position: DVec3::ZERO,
        range: 0.0,
        ..RenderLight::default()
    });
    scene
}

/// The camera: 6 m from the origin on the shaded side (`-Z`), so the sphere's
/// visible hemisphere is the one facing AWAY from the sun. That is the half of
/// the question a diffuse fallback cannot answer.
fn view() -> RenderView {
    RenderView {
        origin: FloatingOrigin::new(DVec3::ZERO),
        eye_world: DVec3::new(0.0, 0.0, -6.0),
        forward: Vec3::new(0.0, 0.0, 1.0),
        up: Vec3::Y,
        fov_y: 0.35,
        near: 0.05,
        width: W,
        height: H,
        ortho: None,
    }
}

/// The primitive sphere's own geometry as a one-joint skinned mesh.
///
/// The palette is a single identity matrix, so the skinned vertex program's
/// linear blend is the identity and the two paths are drawing the SAME surface
/// at the same place — which is the only way a luminance comparison between them
/// means anything.
fn skinned_sphere() -> SkinnedMeshData {
    let (verts, idx) = sphere_geometry();
    SkinnedMeshData {
        vertices: verts
            .iter()
            .map(|v| SkinnedVertex {
                pos: v.pos,
                normal: v.normal,
                uv: v.uv,
                joints: [0, 0, 0, 0],
                weights: [1.0, 0.0, 0.0, 0.0],
            })
            .collect(),
        indices: idx.iter().map(|i| u32::from(*i)).collect(),
    }
}

fn shot(gpu: &GpuContext, scene: &RenderScene, s: RenderSettings) -> Vec<u8> {
    let target = HeadlessTarget::new(gpu, W, H);
    let mut renderer = EngineRenderer::new(gpu, HEADLESS_FORMAT);
    renderer.set_settings(s);
    let v = view();
    let mut last = Vec::new();
    // Four frames: the voxelize/probe nodes run inside the frame graph and the
    // first is the one that fills them.
    for _ in 0..4 {
        renderer.render(gpu, scene, &v, &target.view, (W, H));
        last = target.read_rgba(gpu).expect("readback");
    }
    last
}

/// Median luminance of the pixels the sphere covers, and how many there were.
///
/// The sphere is found rather than assumed: a pixel belongs to it when it
/// differs from the same pixel of a sphere-less frame. That way the mask is
/// exactly the silhouette in *this* frame, and a difference in tessellation or
/// placement between the two paths would show up as a different pixel count
/// rather than quietly biasing a fixed rectangle.
fn body(rgba: &[u8], empty: &[u8]) -> (f64, usize) {
    let mut lum: Vec<f64> = Vec::new();
    for i in (0..rgba.len()).step_by(4) {
        let d = (i32::from(rgba[i]) - i32::from(empty[i])).abs()
            + (i32::from(rgba[i + 1]) - i32::from(empty[i + 1])).abs()
            + (i32::from(rgba[i + 2]) - i32::from(empty[i + 2])).abs();
        if d > 8 {
            lum.push(
                0.2126 * f64::from(rgba[i])
                    + 0.7152 * f64::from(rgba[i + 1])
                    + 0.0722 * f64::from(rgba[i + 2]),
            );
        }
    }
    lum.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = lum.len();
    if n == 0 {
        return (0.0, 0);
    }
    (lum[n / 2], n)
}

/// **A METAL IS NOT BLACK, AND A SKINNED METAL IS NOT DARKER THAN A RIGID ONE.**
///
/// **The mutation**: in `env_lighting.wgsl`'s `gi_ambient_specular`, replace the
/// whole body with `return vec3<f32>(0.0);` — the state wave CHAR1a.2's sentence
/// describes. Both medians collapse and the floor assertion names it. Verified.
///
/// **The second mutation**: delete the `gi_ambient_specular` line from
/// `skinned_mesh.wgsl` only. The rigid reading is unchanged and the agreement
/// assertion goes red — which is the arm's other half, and the reason the two
/// are measured in one test rather than two.
#[test]
fn a_metal_reflects_the_sky_and_the_skinned_path_reflects_the_same_sky() {
    let Some(gpu) = gpu_or_skip() else { return };
    let s = settings();

    // The reference frame: sky and sun, no sphere. Also the mask's background.
    let empty = shot(&gpu, &empty_scene(), s);

    // ── the rigid metal ──
    let mut rigid = empty_scene();
    let mut ball = MeshInstance::lit(
        DVec3::ZERO,
        Quat::IDENTITY,
        Vec3::splat(2.0),
        [0.8, 0.8, 0.8, 1.0],
        1,
    );
    ball.mesh = PrimMesh::Sphere;
    ball.metallic = 1.0;
    ball.roughness = 0.3;
    rigid.instances.push(ball);
    rigid.mark_dirty();
    let (rigid_p50, rigid_px) = body(&shot(&gpu, &rigid, s), &empty);

    // ── the same metal, skinned ──
    let mut skin = empty_scene();
    skin.skinned_meshes
        .push(std::sync::Arc::new(skinned_sphere()));
    skin.skinned.push(SkinnedInstance {
        vt: Default::default(),
        translation: DVec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: Vec3::splat(2.0),
        color: [0.8, 0.8, 0.8, 1.0],
        metallic: 1.0,
        roughness: 0.3,
        emissive: [0.0; 3],
        id: 1,
        mesh: 0,
        blend: 0,
        cutoff: 0.5,
        palette: std::sync::Arc::new(vec![Mat4::IDENTITY]),
        shadow: Default::default(),
    });
    skin.mark_dirty();
    let (skinned_p50, skinned_px) = body(&shot(&gpu, &skin, s), &empty);

    eprintln!(
        "METAL UNDER THE SKY (metallic 1.0, roughness 0.3, shaded hemisphere):\n  \
         rigid   p50 {rigid_p50:6.2} over {rigid_px} px\n  \
         skinned p50 {skinned_p50:6.2} over {skinned_px} px"
    );

    assert!(
        rigid_px > 2_000 && skinned_px > 2_000,
        "the sphere covers {rigid_px} / {skinned_px} pixels — the mask found no \
         body and the medians below are of nothing"
    );
    // 1. Not black. A metal with nothing to reflect would read within a code or
    //    two of the background; the floor is set well above dither and well
    //    below any plausible reflection, so it separates "no term" from "a dim
    //    term" and not "bright enough" from "not bright enough".
    assert!(
        rigid_p50 > 8.0,
        "a metallic sphere under a clear sky reads p50 {rigid_p50:.2} — the \
         engine has no environment specular and a metal really is black"
    );
    assert!(
        skinned_p50 > 8.0,
        "a SKINNED metallic sphere reads p50 {skinned_p50:.2} while the rigid one \
         reads {rigid_p50:.2} — the skinned path has no ambient specular"
    );
    // 2. The two paths agree. Not bit-equality: the rigid path draws an indexed
    //    primitive through the prim pipeline and the skinned path draws the same
    //    vertices through the skinning pipeline, so rasterisation coverage at the
    //    silhouette differs by a pixel here and there. Ten per cent of the
    //    reading is far tighter than any missing lighting term could hide in.
    let rel = (rigid_p50 - skinned_p50).abs() / rigid_p50.max(1e-6);
    assert!(
        rel < 0.10,
        "the skinned metal reads p50 {skinned_p50:.2} against the rigid \
         {rigid_p50:.2} ({:.1}% apart) — the two lit paths are not receiving the \
         same environment",
        rel * 100.0
    );
}
