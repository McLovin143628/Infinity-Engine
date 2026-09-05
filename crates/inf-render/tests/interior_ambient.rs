//! **A ROOM, FROM THE INSIDE** (wave FIX3 audit — priorities (a) and (c)).
//!
//! `sky_ambient.rs` photographs a wall standing in the open. That is the
//! outdoor half of the ambient question and this file is the indoor half,
//! because the formulation wave FIX3 shipped —
//!
//! ```text
//!     ambient = sky_irradiance(n) + Σ_hit (bounce − blocked sky)
//! ```
//!
//! — is a *difference*, and a difference is only as good as the thing it is
//! subtracted from. Outdoors the two halves cancel to the correct occluded sky.
//! Indoors they do not.
//!
//! # The rooms
//!
//! Boxes of white-ish Lambert slabs with 1 m walls (comfortably thicker than
//! the 0.625 m voxel the GI volume resolves at 40 m / 64³), lit by the same
//! 40°-elevation sun as `sky_ambient.rs`, under the same P17 atmosphere,
//! photographed from the **same** eye in every shot of a size so that the
//! camera-centred probe volume never moves between two readings:
//!
//! * **sealed** — no opening at all. Direct sun cannot enter, so every code in
//!   the frame is ambient and nothing else. The physical answer is the room's
//!   own inter-reflection of zero, i.e. **zero**.
//! * **door** — an opening in the +Z wall, which is the wall the sun is on.
//!   The physical answer is the doorway's solid angle of sky plus what the
//!   sunbeam on the floor bounces.
//! * **open** — the ceiling and the +Z wall removed: a courtyard, and a sanity
//!   rung between the other two.
//!
//! Two sizes, and the difference between them is itself a measurement:
//!
//! * [`ROOM`] is 12 × 12 × 4 m, an ordinary room. The probe grid is
//!   16 × 8 × 16 over a 40 m volume, so probes are **5.71 m apart vertically**
//!   and a 4 m room *contains none of them*: its ambient is interpolated
//!   between a probe buried in the ground below and a probe floating in the
//!   open sky above the roof.
//! * [`HALL`] is 24 × 24 × 12 m, which holds two probe layers and eight probe
//!   columns, so its interior is genuinely sampled and the reading is about
//!   the difference term rather than about probe density.
//!
//! Readings are inverted back to scene radiance through the post chain's own
//! measured curve, exactly as `sky_ambient.rs` does it, because a ratio of
//! ACES-encoded codes is a ratio of nothing.
//!
//! # The quadrature, measured here
//!
//! The sealed hall is also the sharpest instrument in the repository for the
//! ledger's carried *"the sky's own quadrature and the probe march's are
//! different sizes"*. With the bounce's sky scaled away by the sky-view factor,
//! what is left in a sealed room is the CPU's 48-direction L1 projection of the
//! sky **minus** the GPU sky-view LUT integrated over the probe's own rays, and
//! it should be zero:
//!
//! ```text
//!     rays = 48 (GiSettings::default, and every committed level)   0.0004
//!     rays = 64 (what the GI goldens author)                       0.0011
//! ```
//!
//! against an open courtyard's 0.52 — so **0.08 %** when the two quadratures
//! sample the same 48 directions and **0.21 %** when they do not. The arm below
//! is aimed at the shipped configuration; at `rays = 64` the doored hall's own
//! reading falls into the same residue and the ratio stops being meaningful.

use glam::{DVec3, Quat, Vec3};
use inf_math::FloatingOrigin;
use inf_render::{
    AtmosphereParams, EngineRenderer, GiSettings, GpuContext, HeadlessTarget, LightKind,
    MeshInstance, RenderLight, RenderScene, RenderSettings, RenderView, ShadowSettings, SunParams,
    HEADLESS_FORMAT,
};

const W: u32 = 256;
const H: u32 = 192;

/// Same sun as `sky_ambient.rs`, so the two files' numbers sit beside each
/// other: 40° above the horizon, due +Z on the horizontal.
const SUN_ELEVATION_DEG: f32 = 40.0;
const SUN_INTENSITY: f32 = 3.0;

/// Slab thickness, metres. **Load-bearing**: the GI volume is 40 m across a
/// 64³ grid, so a voxel is 0.625 m and a wall thinner than that can fail to
/// register as occupancy at all — which would make a "sealed" room leak and
/// turn this whole file into a measurement of nothing.
const THICK: f64 = 1.0;

/// A room's dimensions and the eye it is photographed from.
struct Dims {
    name: &'static str,
    /// Interior half-extent in X and Z, metres.
    half: f64,
    /// Interior height, metres.
    tall: f64,
    door_w: f64,
    door_h: f64,
    /// Eye height. Every shot of this size is taken from `(0, eye_y, 0)`.
    eye_y: f64,
}

/// An ordinary room — and one the 40 m probe grid puts no probe inside.
const ROOM: Dims = Dims {
    name: "room 12x12x4",
    half: 6.0,
    tall: 4.0,
    door_w: 3.0,
    door_h: 2.5,
    eye_y: 2.0,
};

/// A hall big enough to hold probes: at `extent = 40` the probe grid is
/// 2.67 m in X/Z and 5.71 m in Y, so this holds eight probe columns and two
/// probe layers.
const HALL: Dims = Dims {
    name: "hall 24x24x12",
    half: 12.0,
    tall: 12.0,
    door_w: 6.0,
    door_h: 5.0,
    eye_y: 5.0,
};

fn gpu_or_skip() -> Option<GpuContext> {
    match GpuContext::headless() {
        Ok(gpu) => Some(gpu),
        Err(e) => {
            eprintln!("SKIP interior_ambient: no GPU adapter ({e})");
            None
        }
    }
}

fn sun_dir() -> Vec3 {
    let e = SUN_ELEVATION_DEG.to_radians();
    Vec3::new(0.0, e.sin(), e.cos())
}

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

/// One white-ish Lambert slab.
fn slab(id: u32, centre: DVec3, size: Vec3) -> MeshInstance {
    let mut m = MeshInstance::lit(centre, Quat::IDENTITY, size, [0.7, 0.7, 0.7, 1.0], id);
    m.roughness = 1.0;
    m.metallic = 0.0;
    m
}

/// Which room to build.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Room {
    /// No opening anywhere.
    Sealed,
    /// A door in the +Z wall — the wall the sun is on.
    Door,
    /// Ceiling and +Z wall removed: a courtyard.
    Open,
}

/// The room, the ground it stands on and the sun.
fn room_scene(d: &Dims, kind: Room) -> RenderScene {
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

    // The ground the room stands on, extending well past it.
    let mut ground = MeshInstance::lit(
        DVec3::new(0.0, -THICK - 0.5, 0.0),
        Quat::IDENTITY,
        Vec3::new(400.0, 1.0, 400.0),
        [0.35, 0.35, 0.35, 1.0],
        1,
    );
    ground.roughness = 1.0;
    ground.metallic = 0.0;
    scene.instances.push(ground);

    let outer = (d.half + THICK) as f32 * 2.0;
    let h = THICK as f32;
    let tall = d.tall as f32;

    // Floor: its top face is exactly y = 0.
    scene.instances.push(slab(
        2,
        DVec3::new(0.0, -THICK / 2.0, 0.0),
        Vec3::new(outer, h, outer),
    ));
    // Ceiling: its underside is exactly y = d.tall.
    if kind != Room::Open {
        scene.instances.push(slab(
            3,
            DVec3::new(0.0, d.tall + THICK / 2.0, 0.0),
            Vec3::new(outer, h, outer),
        ));
    }
    // The two X walls and the -Z wall are always solid.
    scene.instances.push(slab(
        4,
        DVec3::new(-d.half - THICK / 2.0, d.tall / 2.0, 0.0),
        Vec3::new(h, tall, outer),
    ));
    scene.instances.push(slab(
        5,
        DVec3::new(d.half + THICK / 2.0, d.tall / 2.0, 0.0),
        Vec3::new(h, tall, outer),
    ));
    scene.instances.push(slab(
        6,
        DVec3::new(0.0, d.tall / 2.0, -d.half - THICK / 2.0),
        Vec3::new(outer, tall, h),
    ));

    // The +Z wall — solid, doored, or absent.
    let zc = d.half + THICK / 2.0;
    match kind {
        Room::Sealed => scene.instances.push(slab(
            7,
            DVec3::new(0.0, d.tall / 2.0, zc),
            Vec3::new(outer, tall, h),
        )),
        Room::Door => {
            let side = (outer as f64 - d.door_w) / 2.0;
            let sx = d.door_w / 2.0 + side / 2.0;
            scene.instances.push(slab(
                7,
                DVec3::new(-sx, d.tall / 2.0, zc),
                Vec3::new(side as f32, tall, h),
            ));
            scene.instances.push(slab(
                8,
                DVec3::new(sx, d.tall / 2.0, zc),
                Vec3::new(side as f32, tall, h),
            ));
            // The lintel over the opening.
            scene.instances.push(slab(
                9,
                DVec3::new(0.0, (d.door_h + d.tall) / 2.0, zc),
                Vec3::new(d.door_w as f32, (d.tall - d.door_h) as f32, h),
            ));
        }
        Room::Open => {}
    }

    // The analytic sun. A non-empty light list is load-bearing: `mesh.wgsl`
    // falls back to a hard-coded editor sun when the list is empty.
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

fn view(d: &Dims, forward: Vec3, up: Vec3) -> RenderView {
    RenderView {
        origin: FloatingOrigin::new(DVec3::ZERO),
        eye_world: DVec3::new(0.0, d.eye_y, 0.0),
        forward,
        up,
        fov_y: 45f32.to_radians(),
        near: 0.05,
        width: W,
        height: H,
        ortho: None,
    }
}

fn shot(gpu: &GpuContext, scene: &RenderScene, v: &RenderView, s: RenderSettings) -> Vec<u8> {
    let target = HeadlessTarget::new(gpu, W, H);
    let mut renderer = EngineRenderer::new(gpu, HEADLESS_FORMAT);
    renderer.set_settings(s);
    let mut last = Vec::new();
    for _ in 0..4 {
        renderer.render(gpu, scene, v, &target.view, (W, H));
        last = target.read_rgba(gpu).expect("readback");
    }
    last
}

/// Mean luminance (0..255) of the central patch.
fn patch(rgba: &[u8]) -> f64 {
    let (mut sum, mut n) = (0.0f64, 0usize);
    for y in (H * 40 / 100)..(H * 60 / 100) {
        for x in (W * 40 / 100)..(W * 60 / 100) {
            let i = ((y * W + x) * 4) as usize;
            sum += 0.2126 * f64::from(rgba[i])
                + 0.7152 * f64::from(rgba[i + 1])
                + 0.0722 * f64::from(rgba[i + 2]);
            n += 1;
        }
    }
    sum / n as f64
}

/// The post chain's radiance→code curve, measured the same way
/// `sky_ambient.rs` measures it (a full-screen emitter at sixteen known
/// radiances, bloom/TAA/flare off so the chain is per-pixel). Restated here
/// rather than shared because this file has to compile and run against the
/// **pre-FIX3** tree as well, where `sky_ambient.rs` does not exist.
struct Ladder(Vec<(f64, f64)>);

impl Ladder {
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
            let mut s = MeshInstance::lit(
                DVec3::ZERO,
                Quat::IDENTITY,
                Vec3::new(60.0, 60.0, 1.0),
                [0.0, 0.0, 0.0, 1.0],
                1,
            );
            s.roughness = 1.0;
            s.emissive = [l, l, l];
            scene.instances.push(s);
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
            let v = RenderView {
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
            let img = shot(gpu, &scene, &v, base_settings());
            rungs.push((f64::from(l), patch(&img)));
        }
        Self(rungs)
    }

    fn radiance(&self, code: f64) -> f64 {
        // Below the black rung the chain has bottomed out and the only signal
        // left is the dither's own half-LSB. Report zero rather than `NaN`: a
        // surface that renders under the black rung IS black, and a `NaN`
        // silently disables every ratio downstream of it.
        if code <= self.0[0].1 {
            return 0.0;
        }
        for w in self.0.windows(2) {
            let ((l0, c0), (l1, c1)) = (w[0], w[1]);
            if (c0 - code) * (c1 - code) <= 0.0 && (c1 - c0).abs() > 1.0e-9 {
                return l0 + (l1 - l0) * (code - c0) / (c1 - c0);
            }
        }
        f64::NAN
    }
}

struct Faces {
    ceiling: f64,
    wall: f64,
    floor: f64,
    ceiling_code: f64,
    wall_code: f64,
    floor_code: f64,
}

fn faces(
    gpu: &GpuContext,
    ladder: &Ladder,
    d: &Dims,
    scene: &RenderScene,
    s: RenderSettings,
) -> Faces {
    // Straight up at the ceiling, at the -X wall (perpendicular to the sun, so
    // no direct beam lands in the patch), and straight down at the floor.
    let c = patch(&shot(
        gpu,
        scene,
        &view(d, Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 0.0, 1.0)),
        s,
    ));
    let w = patch(&shot(
        gpu,
        scene,
        &view(d, Vec3::new(-1.0, 0.0, 0.0), Vec3::Y),
        s,
    ));
    let f = patch(&shot(
        gpu,
        scene,
        &view(d, Vec3::new(0.0, -1.0, 0.0), Vec3::new(0.0, 0.0, 1.0)),
        s,
    ));
    Faces {
        ceiling: ladder.radiance(c),
        wall: ladder.radiance(w),
        floor: ladder.radiance(f),
        ceiling_code: c,
        wall_code: w,
        floor_code: f,
    }
}

/// **THE INDOOR MEASUREMENT.** Prints both room sizes × {sealed, door, open} ×
/// {GI off, GI on}, in scene radiance and in code, and asserts the one property
/// a difference formulation must have and can lose: **a sealed room is dark**.
///
/// A room with no opening receives no light. Whatever ambient a shaded surface
/// in it reads is light the renderer invented. The doored room beside it is the
/// reference for how much light there legitimately is to be had at all.
///
/// Mutation-verified against the defect it was written for: with
/// `gi_probes.wgsl`'s hit term lighting the surface it hit by
/// `sky_irradiance(nh)` — the **unoccluded** sky — a sealed hall's wall reads
/// the same as a doored one's, because the light it reads never came through
/// the door.
#[test]
fn a_sealed_room_is_dark_and_a_doored_one_is_lit_by_its_door() {
    let Some(gpu) = gpu_or_skip() else { return };
    let ladder = Ladder::measure(&gpu);

    eprintln!("FIX3 audit — a room from the inside (radiance, and @code)");
    eprintln!(
        "  {:<32} {:>16} {:>16} {:>16}",
        "", "ceiling", "wall(-X)", "floor"
    );

    let mut sealed_on = Vec::new();
    let mut door_on = Vec::new();
    for d in [&ROOM, &HALL] {
        for (kind, label) in [
            (Room::Sealed, "sealed"),
            (Room::Door, "door  "),
            (Room::Open, "open  "),
        ] {
            let scene = room_scene(d, kind);
            for (s, gi) in [(base_settings(), "GI off"), (gi_on(), "GI on ")] {
                let f = faces(&gpu, &ladder, d, &scene, s);
                eprintln!(
                    "  {:<14} {label} {gi}   {:>7.4} @{:5.1} {:>7.4} @{:5.1} {:>7.4} @{:5.1}",
                    d.name, f.ceiling, f.ceiling_code, f.wall, f.wall_code, f.floor, f.floor_code
                );
                if gi == "GI on " {
                    match kind {
                        Room::Sealed => sealed_on.push((d.name, f)),
                        Room::Door => door_on.push((d.name, f)),
                        Room::Open => {}
                    }
                }
            }
        }
    }

    // The arm, on the hall — the size whose interior the probe grid actually
    // samples, so a failure is about the difference term and not about probe
    // density. A sealed room has no light in it; the doored room says how much
    // light a room of this size and albedo can hold when it does have an
    // opening. A sealed room reading a material fraction of that is the
    // renderer inventing light.
    let (_, sealed) = sealed_on
        .iter()
        .find(|(n, _)| *n == HALL.name)
        .expect("the sealed hall row");
    let (_, doored) = door_on
        .iter()
        .find(|(n, _)| *n == HALL.name)
        .expect("the doored hall row");
    // Summed over the three faces, with a floor under the denominator so a
    // doored room that is itself black cannot make the ratio look healthy.
    let sealed_total = sealed.ceiling + sealed.wall + sealed.floor;
    let doored_total = doored.ceiling + doored.wall + doored.floor;
    let ratio = sealed_total / doored_total.max(1.0e-4);
    eprintln!(
        "  sealed:doored hall, GI on — summed faces {sealed_total:.4} : {doored_total:.4} = \
         {ratio:.4}  (a sealed room should be a small fraction of a doored one)"
    );
    assert!(
        ratio < 0.40,
        "a SEALED hall's three faces sum to {sealed_total:.4} against the DOORED hall's \
         {doored_total:.4} — {ratio:.2}× the light of a room that has an opening, with no \
         opening at all. The ambient term is lighting a room that cannot see the sky."
    );
}
