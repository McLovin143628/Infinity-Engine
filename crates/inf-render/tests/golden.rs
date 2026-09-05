//! Golden-image harness (P2.5). For each scene we:
//!   1. render it twice and assert the two frames match — the renderer is
//!      deterministic on a fixed adapter (catches nondeterminism/races);
//!   2. assert scene-specific structural properties (sky is sky, the cube is
//!      where it should be) — adapter-independent;
//!   3. compare against the committed golden PNG. This runs when
//!      `INF_GOLDEN_STRICT=1` (a matched-adapter run) and always writes the
//!      golden when it's missing or `INF_BLESS_GOLDENS=1`.
//!
//! Exact cross-GPU pixels differ (AA/rasterization), so strict pixel diffing
//! is opt-in; CI relies on the determinism + structural gates, which are
//! adapter-robust. Regenerate goldens with `INF_BLESS_GOLDENS=1 cargo test -p
//! inf-render --test golden`. The harness skips entirely with no GPU adapter
//! (headless CI without lavapipe/WARP).

use std::path::PathBuf;
use std::sync::Arc;

use glam::{DVec3, Mat3, Mat4, Quat, Vec2, Vec3};
use inf_math::FloatingOrigin;
use inf_render::gizmo::{self, GizmoAxis, GizmoMode};
use inf_render::golden::{image_diff, within_tolerance};
use inf_render::passes::vgeom::{cpu_visible_set, cull_flags, frustum_planes, lod_threshold};
use inf_render::{
    assemble_patches, cull_visible, cull_visible_streamed, detail_texel, expand_text, half_to_f32,
    shape_texel, Ambient2D, AtmosphereParams, AtmosphereQuality, BloomSettings, CloudParams,
    CloudQuality, CloudVolumes, EngineRenderer, FilmSettings, FlareSettings, GiAudit, GiQuality,
    GiSettings, GpuContext, HAlign, HeadlessTarget, HeightFog, LightKind, MeshInstance,
    PrebatchedRun, PrecipParams, PrecipQuality, PrimMesh, RenderChunk, RenderDeform,
    RenderDeformCell, RenderLight, RenderLight2D, RenderScene, RenderSettings, RenderTerrain,
    RenderTerrainLayer, RenderTerrainTile, RenderTilemap, RenderView, RenderVoxelChunk,
    RenderVoxelVertex, RenderVoxelVolume, RenderWater, ScatterBatch, ScatterData, ScatterInstance,
    ShadowSettings, SkinnedInstance, SkinnedMeshData, SkinnedVertex, SpriteInstance,
    SpriteTextureUpload, SsaoSettings, SunParams, TerrainTileKey, TextParams, TilemapParams,
    VgeomAsset, VgeomInstance, VgeomMesh, VgeomSettings, ViewMode, VoxelChunkKey, WaterKindGpu,
    WaterQuality, WaveField, WaveSpec, BILLBOARD_CYLINDRICAL, BILLBOARD_NONE, BILLBOARD_SPHERICAL,
    BUILTIN_FONT_COLS, BUILTIN_FONT_FIRST_CP, BUILTIN_FONT_ROWS, BUILTIN_FONT_TEXTURE,
    CPU_GPU_EXACT_CHANNEL_FRACTION, CPU_GPU_SHADOW_TOLERANCE, CPU_GPU_STEP_TOLERANCE,
    CPU_GPU_VALUE_ESCAPE_FRACTION, CPU_GPU_VALUE_TOLERANCE, HEADLESS_FORMAT, TILE_CHUNK_DIM,
};

const W: u32 = 320;
const H: u32 = 180;

fn goldens_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("goldens")
}

fn gpu_or_skip() -> Option<GpuContext> {
    match GpuContext::headless() {
        Ok(gpu) => Some(gpu),
        Err(e) => {
            eprintln!("SKIP golden: no GPU adapter ({e})");
            None
        }
    }
}

fn overlook_view() -> RenderView {
    RenderView {
        origin: FloatingOrigin::new(DVec3::ZERO),
        eye_world: DVec3::new(6.0, 4.5, 9.0),
        forward: (DVec3::ZERO - DVec3::new(6.0, 4.5, 9.0))
            .as_vec3()
            .normalize(),
        up: Vec3::Y,
        fov_y: 60f32.to_radians(),
        near: 0.05,
        width: W,
        height: H,
        ortho: None,
    }
}

fn render(gpu: &GpuContext, scene: &RenderScene, view: &RenderView) -> Vec<u8> {
    render_with(gpu, scene, view, RenderSettings::default())
}

/// Render one frame with explicit HDR/post settings (bloom/SSAO/exposure). TAA is
/// intentionally not exercised here — single-frame determinism goldens keep it
/// off; the multi-frame convergence is covered by `taa_multiframe_stable`.
fn render_with(
    gpu: &GpuContext,
    scene: &RenderScene,
    view: &RenderView,
    settings: RenderSettings,
) -> Vec<u8> {
    let target = HeadlessTarget::new(gpu, W, H);
    let mut renderer = EngineRenderer::new(gpu, HEADLESS_FORMAT);
    renderer.set_settings(settings);
    renderer.render(gpu, scene, view, &target.view, (W, H));
    target.read_rgba(gpu).expect("readback")
}

fn write_png(path: &std::path::Path, rgba: &[u8]) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let file = std::fs::File::create(path).unwrap();
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), W, H);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header().unwrap().write_image_data(rgba).unwrap();
}

fn read_png(path: &std::path::Path) -> Option<Vec<u8>> {
    let file = std::fs::File::open(path).ok()?;
    let dec = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = dec.read_info().ok()?;
    let mut buf = vec![0; reader.output_buffer_size()?];
    let info = reader.next_frame(&mut buf).ok()?;
    buf.truncate(info.buffer_size());
    Some(buf)
}

fn px(rgba: &[u8], x: u32, y: u32) -> [u8; 4] {
    let i = ((y * W + x) * 4) as usize;
    [rgba[i], rgba[i + 1], rgba[i + 2], rgba[i + 3]]
}

/// The shared gate: determinism, then golden write/compare.
fn check_golden(gpu: &GpuContext, name: &str, scene: &RenderScene, view: &RenderView) -> Vec<u8> {
    check_golden_with(gpu, name, scene, view, RenderSettings::default())
}

/// [`check_golden`] with explicit post settings (bloom/SSAO goldens).
fn check_golden_with(
    gpu: &GpuContext,
    name: &str,
    scene: &RenderScene,
    view: &RenderView,
    settings: RenderSettings,
) -> Vec<u8> {
    let a = render_with(gpu, scene, view, settings);
    let b = render_with(gpu, scene, view, settings);
    let (mean, max) = image_diff(&a, &b, W, H);
    assert!(
        mean < 0.005 && max < 0.05,
        "{name}: renderer not deterministic (mean {mean}, max {max})"
    );

    let path = goldens_dir().join(format!("{name}.png"));
    let bless = std::env::var("INF_BLESS_GOLDENS").is_ok();
    let strict = std::env::var("INF_GOLDEN_STRICT").is_ok();

    if bless || read_png(&path).is_none() {
        write_png(&path, &a);
        eprintln!("golden {name}: wrote {}", path.display());
    } else if strict {
        let golden = read_png(&path).expect("golden png");
        let (mean, max) = image_diff(&a, &golden, W, H);
        // Printed on the PASSING path too (wave VIS1a). `within_tolerance` is
        // perceptual, not byte-exact, so a real change to what the engine draws
        // can sit inside it and leave the committed frame quietly depicting an
        // engine that no longer exists. A number nobody can read is a number
        // nobody can notice moving.
        eprintln!("golden {name}: mean {mean:.6}, max {max:.6} against the committed frame");
        assert!(
            within_tolerance(mean, max),
            "{name}: differs from golden (mean {mean}, max {max})"
        );
    }
    a
}

/// **THE UI NODE MOVES NO PIXEL UNTIL SOMETHING IS IN IT** (island wave I5).
///
/// This is what the frozen goldens rest on. A new node in the graph is a new
/// chance for a frame to change, so the claim is measured on both sides rather
/// than argued: with an empty [`inf_ui::UiDrawList`] — which is every golden
/// scene, and the default — the frame is **byte-identical** to the same frame,
/// and with one quad in it the frame **differs**.
///
/// The second half is what makes the first mean something: without it, a node
/// that silently failed to draw at all would pass.
#[test]
fn the_ui_node_is_a_no_op_on_an_empty_list_and_draws_when_it_is_not() {
    let Some(gpu) = gpu_or_skip() else { return };
    let scene = RenderScene {
        grid_enabled: true,
        ..Default::default()
    };
    assert!(
        scene.ui.is_empty(),
        "the default scene ships a UI, so every golden already has one in it"
    );
    let view = overlook_view();
    let bare = render_with(&gpu, &scene, &view, RenderSettings::default());
    let again = render_with(&gpu, &scene, &view, RenderSettings::default());
    assert_eq!(
        bare, again,
        "the frame is not bit-deterministic, so the comparison below means nothing"
    );

    // One opaque white rect over the middle of the frame.
    let mut with_ui = scene.clone();
    with_ui.ui = inf_ui::UiDrawList::new(glam::Vec2::new(W as f32, H as f32));
    with_ui.ui.rect(
        inf_ui::Rect::new(
            W as f32 * 0.25,
            H as f32 * 0.25,
            W as f32 * 0.5,
            H as f32 * 0.5,
        ),
        [1.0, 1.0, 1.0, 1.0],
    );
    let drawn = render_with(&gpu, &with_ui, &view, RenderSettings::default());
    assert_ne!(
        bare, drawn,
        "the UI node drew nothing, so the emptiness arm above cannot fail"
    );
    // …and it drew where it was told: the centre pixel is white and a corner is
    // not, so the node is not merely clearing the frame.
    let centre = px(&drawn, W / 2, H / 2);
    let corner = px(&drawn, 2, 2);
    assert!(
        centre[0] > 240 && centre[1] > 240 && centre[2] > 240,
        "the centre is {centre:?}, not the white rect that was asked for"
    );
    assert_eq!(
        corner,
        px(&bare, 2, 2),
        "a corner outside the rect moved, so the node is drawing over the whole frame"
    );
}

#[test]
fn golden_grid_and_sky() {
    let Some(gpu) = gpu_or_skip() else { return };
    let scene = RenderScene {
        grid_enabled: true,
        ..Default::default()
    };
    let img = check_golden(&gpu, "grid_sky", &scene, &overlook_view());

    // Structural: top rows are sky (not black, leaning blue); bottom rows show
    // the lit grid on the ground plane.
    let sky = px(&img, W / 2, 3);
    assert!(
        sky[2] as u16 + 4 >= sky[0] as u16,
        "sky not bluish: {sky:?}"
    );
    assert!(sky[2] > 3, "sky too dark: {sky:?}");
    // The frame is not uniform (grid present).
    let ground = px(&img, W / 2, H - 20);
    assert_ne!(sky, ground);
}

#[test]
fn golden_cubes() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mut scene = RenderScene {
        grid_enabled: true,
        ..Default::default()
    };
    for (i, (x, z, c)) in [
        (0.0, 0.0, [0.80, 0.20, 0.20]),
        (2.5, -1.0, [0.20, 0.70, 0.30]),
        (-2.0, 1.5, [0.25, 0.45, 0.95]),
    ]
    .into_iter()
    .enumerate()
    {
        scene.instances.push(MeshInstance::lit(
            DVec3::new(x, 0.5, z),
            Quat::from_rotation_y(0.3),
            Vec3::ONE,
            [c[0], c[1], c[2], 1.0],
            i as u32 + 1,
        ));
    }
    scene.mark_dirty();
    let img = check_golden(&gpu, "cubes", &scene, &overlook_view());

    // The central red cube dominates the middle of the frame.
    let center = px(&img, W / 2, H / 2);
    assert!(
        center[0] > center[2] && center[0] > 40,
        "expected the red cube at center: {center:?}"
    );
}

/// Render one frame in a given [`ViewMode`] (R-P2), default post settings.
fn render_view_mode(
    gpu: &GpuContext,
    scene: &RenderScene,
    view: &RenderView,
    mode: ViewMode,
) -> Vec<u8> {
    let target = HeadlessTarget::new(gpu, W, H);
    let mut renderer = EngineRenderer::new(gpu, HEADLESS_FORMAT);
    renderer.set_view_mode(mode);
    renderer.render(gpu, scene, view, &target.view, (W, H));
    target.read_rgba(gpu).expect("readback")
}

/// Unlit view-mode golden (R-P2): the same three cubes as [`golden_cubes`], but
/// rendered with `set_view_mode(Unlit)` so the lit passes short-circuit to
/// albedo+emissive (no lighting). Determinism gate (render twice), a new committed
/// golden `unlit.png` (bless with `INF_BLESS_GOLDENS=1`), and a structural gate:
/// the unlit frame must differ from the lit one (proving the flag actually flipped
/// the shading), each cube still shows its flat base colour, and — crucially — the
/// *lit* frame stays byte-identical to `golden_cubes` (view mode never perturbs the
/// default Lit path; every pre-R-P2 golden is unaffected). Wireframe is NOT
/// goldened — line raster is adapter-fragile (feature-gated + AA-dependent) — so it
/// is covered by the naga compose test + the caps/degrade unit tests instead.
#[test]
fn golden_unlit() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mut scene = RenderScene {
        grid_enabled: true,
        ..Default::default()
    };
    for (i, (x, z, c)) in [
        (0.0, 0.0, [0.80, 0.20, 0.20]),
        (2.5, -1.0, [0.20, 0.70, 0.30]),
        (-2.0, 1.5, [0.25, 0.45, 0.95]),
    ]
    .into_iter()
    .enumerate()
    {
        scene.instances.push(MeshInstance::lit(
            DVec3::new(x, 0.5, z),
            Quat::from_rotation_y(0.3),
            Vec3::ONE,
            [c[0], c[1], c[2], 1.0],
            i as u32 + 1,
        ));
    }
    scene.mark_dirty();
    let view = overlook_view();

    // Determinism + golden write/compare for the Unlit render.
    let a = render_view_mode(&gpu, &scene, &view, ViewMode::Unlit);
    let b = render_view_mode(&gpu, &scene, &view, ViewMode::Unlit);
    let (mean, max) = image_diff(&a, &b, W, H);
    assert!(
        mean < 0.005 && max < 0.05,
        "unlit: renderer not deterministic (mean {mean}, max {max})"
    );
    let path = goldens_dir().join("unlit.png");
    if std::env::var("INF_BLESS_GOLDENS").is_ok() || read_png(&path).is_none() {
        write_png(&path, &a);
        eprintln!("golden unlit: wrote {}", path.display());
    } else if std::env::var("INF_GOLDEN_STRICT").is_ok() {
        let golden = read_png(&path).expect("golden png");
        let (m, mx) = image_diff(&a, &golden, W, H);
        assert!(
            within_tolerance(m, mx),
            "unlit: differs from golden (mean {m}, max {mx})"
        );
    }

    // The unlit render differs from the lit one (the flag genuinely changed the
    // shading — unlit is flatter/brighter, no GGX/ambient/haze).
    let lit = render_view_mode(&gpu, &scene, &view, ViewMode::Lit);
    let (dmean, _dmax) = image_diff(&a, &lit, W, H);
    assert!(dmean > 0.002, "unlit should differ from lit (mean {dmean})");

    // The default Lit path is byte-stable vs the plain-renderer `golden_cubes`
    // frame — view mode never perturbs Lit (the byte-identical guarantee).
    let plain = render(&gpu, &scene, &view);
    let (lmean, lmax) = image_diff(&lit, &plain, W, H);
    assert!(
        lmean < 1e-6 && lmax < 1e-6,
        "Lit view mode must match the default renderer exactly (mean {lmean}, max {lmax})"
    );

    // The central red cube still reads as red under unlit shading.
    let center = px(&a, W / 2, H / 2);
    assert!(
        center[0] > center[2] && center[0] > 40,
        "expected the red cube at center (unlit): {center:?}"
    );
}

/// Primitive-geometry golden (R-P1): one of each of the five built-in kinds
/// (Cube, Sphere, Plane, Cylinder, Cone) in a row on the ground grid, each a
/// distinct colour. Proves every kind renders as its real shape through the whole
/// mesh path. Structural gate: swapping all kinds to Cube changes the frame (so
/// the per-kind geometry genuinely varies) and the row is lit. Determinism gate
/// via `check_golden`; strict pixel diff opt-in.
#[test]
fn golden_primitives() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mut scene = RenderScene {
        grid_enabled: true,
        ..Default::default()
    };
    let kinds = [
        PrimMesh::Cube,
        PrimMesh::Sphere,
        PrimMesh::Plane,
        PrimMesh::Cylinder,
        PrimMesh::Cone,
    ];
    let colors = [
        [0.85, 0.25, 0.25],
        [0.25, 0.75, 0.35],
        [0.30, 0.45, 0.95],
        [0.85, 0.75, 0.30],
        [0.75, 0.35, 0.85],
    ];
    for (i, (&kind, c)) in kinds.iter().zip(colors).enumerate() {
        scene.instances.push(MeshInstance {
            vt: Default::default(),
            translation: DVec3::new(-4.0 + i as f64 * 2.0, 0.5, 0.0),
            rotation: Quat::from_rotation_y(0.3),
            scale: Vec3::ONE,
            color: [c[0], c[1], c[2], 1.0],
            metallic: 0.0,
            roughness: 0.5,
            emissive: [0.0; 3],
            id: i as u32 + 1,
            mesh: kind,
            blend: 0,
            cutoff: 0.5,
        });
    }
    scene.mark_dirty();

    let img = check_golden(&gpu, "primitives", &scene, &overlook_view());

    // Swapping every kind to Cube must change the image — proof the per-kind
    // geometry (not just the cube) actually reaches the rasterizer.
    let mut cubes = scene.clone();
    for inst in &mut cubes.instances {
        inst.mesh = PrimMesh::Cube;
    }
    cubes.mark_dirty();
    let cube_img = render(&gpu, &cubes, &overlook_view());
    let (mean, _max) = image_diff(&img, &cube_img, W, H);
    assert!(
        mean > 0.002,
        "primitive kinds should differ from an all-cube row (mean {mean})"
    );

    let lit = img.chunks(4).any(|p| p[0] > 60 || p[1] > 60 || p[2] > 60);
    assert!(lit, "expected a lit primitive pixel");
}

#[test]
fn golden_selection_gizmo() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mut scene = RenderScene {
        grid_enabled: true,
        selected: vec![1],
        ..Default::default()
    };
    scene.instances.push(MeshInstance::lit(
        DVec3::new(0.0, 0.5, 0.0),
        Quat::IDENTITY,
        Vec3::ONE,
        [0.30, 0.55, 0.65, 1.0],
        1,
    ));
    scene.mark_dirty();
    let view = overlook_view();
    // Translate gizmo at the cube, screen-constant size.
    let origin_local = view.origin.to_render(DVec3::new(0.0, 0.5, 0.0));
    let size = gizmo::gizmo_world_size(origin_local, view.eye_local(), view.fov_y);
    gizmo::build_geometry(
        &mut scene.debug,
        GizmoMode::Translate,
        origin_local,
        glam::Quat::IDENTITY,
        size,
        Some(GizmoAxis::X),
        false,
    );

    let img = check_golden(&gpu, "selection_gizmo", &scene, &view);

    // The selection outline paints the composite's orange edge (linear
    // (1.0, 0.42, 0.05) → sRGB ≈ [255, 171, 63]) somewhere in the central
    // band: red-dominant, mid green, low blue.
    let mut found_outline = false;
    'scan: for y in (H / 4)..(3 * H / 4) {
        for x in (W / 4)..(3 * W / 4) {
            let p = px(&img, x, y);
            if p[0] > 200 && (100..=210).contains(&p[1]) && p[2] < 120 && p[0] > p[2] + 90 {
                found_outline = true;
                break 'scan;
            }
        }
    }
    assert!(found_outline, "selection outline (orange) not found");
}

/// PBR material scene (P7.4 golden): metallic/roughness/emissive variation lit by
/// a directional key + a coloured point light. Exercises the P7.1 shading path
/// in CI (determinism gate via `check_golden`; strict pixel diff is opt-in).
#[test]
fn golden_pbr_materials() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mut scene = RenderScene {
        grid_enabled: true,
        ..Default::default()
    };
    // A row of cubes sweeping roughness at metallic = 1.
    for (i, &(x, rough, metallic, emissive)) in [
        (-3.0f64, 0.1f32, 1.0f32, [0.0f32, 0.0, 0.0]),
        (-1.0, 0.4, 1.0, [0.0, 0.0, 0.0]),
        (1.0, 0.7, 0.0, [0.0, 0.0, 0.0]),
        (3.0, 0.5, 0.0, [0.6, 0.15, 0.0]), // emissive
    ]
    .iter()
    .enumerate()
    {
        scene.instances.push(MeshInstance {
            vt: Default::default(),
            translation: DVec3::new(x, 0.5, 0.0),
            rotation: Quat::from_rotation_y(0.4),
            scale: Vec3::ONE,
            color: [0.85, 0.78, 0.55, 1.0],
            metallic,
            roughness: rough,
            emissive,
            id: i as u32 + 1,
            mesh: PrimMesh::Cube,
            blend: 0,
            cutoff: 0.5,
        });
    }
    scene.lights.push(RenderLight {
        kind: LightKind::Directional,
        color: [1.0, 0.98, 0.9],
        intensity: 3.0,
        direction: Vec3::new(0.4, 0.8, 0.4).normalize(),
        position: DVec3::ZERO,
        range: 0.0,
        ..RenderLight::default()
    });
    scene.lights.push(RenderLight {
        kind: LightKind::Point,
        color: [0.3, 0.5, 1.0],
        intensity: 30.0,
        direction: Vec3::ZERO,
        position: DVec3::new(0.0, 2.5, 2.0),
        range: 12.0,
        ..RenderLight::default()
    });
    scene.mark_dirty();

    let img = check_golden(&gpu, "pbr_materials", &scene, &overlook_view());
    // The scene is lit: some pixel is clearly brighter than the dark backdrop.
    let lit = img.chunks(4).any(|p| p[0] > 90 || p[1] > 90 || p[2] > 90);
    assert!(lit, "expected a lit PBR pixel");
}

/// Translucency golden (R-P5): opaque cubes behind, TWO overlapping tinted
/// translucent panes (`blend == 2`, 50% alpha) in front — proving alpha blending +
/// the deterministic back-to-front sort — plus one **masked** cube (`blend == 1`)
/// whose uniform alpha is below its cutoff, so the alpha-test discards it entirely
/// (a "cutout" hole; per-fragment texture opacity is deferred). Determinism gate
/// via `check_golden`; strict pixel diff opt-in, blessed on a GPU host. Structural
/// gates prove that (a) the panes genuinely blend (vs the same scene made opaque)
/// and (b) the masked instance is genuinely alpha-tested away (vs made opaque).
#[test]
fn golden_translucency() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mut scene = RenderScene {
        grid_enabled: true,
        ..Default::default()
    };
    // A directional key so the panes + cubes are lit.
    scene.lights.push(RenderLight {
        kind: LightKind::Directional,
        color: [1.0, 0.98, 0.9],
        intensity: 3.0,
        direction: Vec3::new(0.4, 0.8, 0.4).normalize(),
        position: DVec3::ZERO,
        range: 0.0,
        ..RenderLight::default()
    });
    // Two opaque cubes at the back (−z).
    scene.instances.push(MeshInstance::lit(
        DVec3::new(-1.4, 0.5, -0.6),
        Quat::from_rotation_y(0.3),
        Vec3::ONE,
        [0.85, 0.22, 0.22, 1.0],
        1,
    ));
    scene.instances.push(MeshInstance::lit(
        DVec3::new(1.4, 0.5, -0.6),
        Quat::from_rotation_y(0.3),
        Vec3::ONE,
        [0.22, 0.72, 0.32, 1.0],
        2,
    ));
    // A masked cube up front whose uniform alpha (0.3) is below its cutoff (0.5)
    // → the mesh fs discards every fragment (the visible cutout: when drawn opaque
    // it would occlude the panes + cubes behind it; masked, it vanishes).
    scene.instances.push(MeshInstance {
        vt: Default::default(),
        translation: DVec3::new(0.0, 0.9, 3.2),
        rotation: Quat::from_rotation_y(0.3),
        scale: Vec3::splat(1.3),
        color: [0.90, 0.85, 0.20, 0.30],
        metallic: 0.0,
        roughness: 0.5,
        emissive: [0.0; 3],
        id: 3,
        mesh: PrimMesh::Cube,
        blend: 1,
        cutoff: 0.5,
    });
    // Two overlapping translucent panes (thin cubes) in front (+z, toward the
    // camera), tinted blue then orange at 50% alpha. The farther one draws first.
    scene.instances.push(MeshInstance {
        vt: Default::default(),
        translation: DVec3::new(-0.4, 0.9, 1.2),
        rotation: Quat::IDENTITY,
        scale: Vec3::new(2.4, 2.4, 0.06),
        color: [0.25, 0.45, 1.0, 0.5],
        metallic: 0.0,
        roughness: 0.5,
        emissive: [0.0; 3],
        id: 4,
        mesh: PrimMesh::Cube,
        blend: 2,
        cutoff: 0.5,
    });
    scene.instances.push(MeshInstance {
        vt: Default::default(),
        translation: DVec3::new(0.5, 0.7, 2.1),
        rotation: Quat::IDENTITY,
        scale: Vec3::new(2.4, 2.4, 0.06),
        color: [1.0, 0.5, 0.15, 0.5],
        metallic: 0.0,
        roughness: 0.5,
        emissive: [0.0; 3],
        id: 5,
        mesh: PrimMesh::Cube,
        blend: 2,
        cutoff: 0.5,
    });
    scene.mark_dirty();

    let img = check_golden(&gpu, "translucency", &scene, &overlook_view());

    // The scene renders lit + blended content.
    let lit = img.chunks(4).any(|p| p[0] > 60 || p[1] > 60 || p[2] > 60);
    assert!(lit, "expected a lit/blended pixel");

    // (a) The translucent panes genuinely blend: making them opaque changes the
    // frame (an opaque pane would fully hide what's behind it).
    let mut solid_panes = scene.clone();
    for inst in &mut solid_panes.instances {
        if inst.blend == 2 {
            inst.blend = 0;
            inst.color[3] = 1.0;
        }
    }
    solid_panes.mark_dirty();
    let solid_panes_img = render(&gpu, &solid_panes, &overlook_view());
    let (mean, _max) = image_diff(&img, &solid_panes_img, W, H);
    assert!(
        mean > 0.002,
        "translucent blending should differ from opaque panes (mean {mean})"
    );

    // (b) The masked instance is genuinely alpha-tested away: making it opaque
    // (fully drawn) changes the frame.
    let mut solid_mask = scene.clone();
    for inst in &mut solid_mask.instances {
        if inst.blend == 1 {
            inst.blend = 0;
            inst.color[3] = 1.0;
        }
    }
    solid_mask.mark_dirty();
    let solid_mask_img = render(&gpu, &solid_mask, &overlook_view());
    let (m2, _max) = image_diff(&img, &solid_mask_img, W, H);
    assert!(
        m2 > 0.001,
        "masked discard should differ from an opaque instance (mean {m2})"
    );
}

/// Spot-light golden (R-P3): a ground plane + a few cubes lit by a single spot
/// aimed obliquely, so the cone's lit ellipse and its soft outer-cone falloff are
/// both on screen. No directional light — the spot shapes the frame. Exercises
/// the shaders' `w == 2` branch (cone `smoothstep` × windowed inverse-square)
/// through the mesh path headlessly (determinism gate via `check_golden`; strict
/// pixel diff opt-in, blessed on a GPU host). Structural gate: the spot frame
/// differs from the same scene lit by a plain point light — proving the cone mask
/// actually clips the illumination (a point light would light the plane broadly).
#[test]
fn golden_spot_lights() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mut scene = RenderScene {
        grid_enabled: true,
        ..Default::default()
    };
    // A large ground plane to catch the cone.
    scene.instances.push(MeshInstance {
        vt: Default::default(),
        translation: DVec3::new(0.0, 0.0, 0.0),
        rotation: Quat::IDENTITY,
        scale: Vec3::new(20.0, 1.0, 20.0),
        color: [0.60, 0.60, 0.62, 1.0],
        metallic: 0.0,
        roughness: 0.7,
        emissive: [0.0; 3],
        id: 1,
        mesh: PrimMesh::Plane,
        blend: 0,
        cutoff: 0.5,
    });
    // A few cubes standing in and around the beam.
    for (i, (x, z)) in [(1.5, 1.5), (3.0, 0.5), (-1.0, 2.5)]
        .into_iter()
        .enumerate()
    {
        scene.instances.push(MeshInstance::lit(
            DVec3::new(x, 0.5, z),
            Quat::from_rotation_y(0.3),
            Vec3::ONE,
            [0.80, 0.75, 0.70, 1.0],
            i as u32 + 2,
        ));
    }
    // One spot high above, aimed obliquely toward (2, 0, 2). `direction` is the
    // toward-the-light vector; the emission axis is its negation.
    let emit = Vec3::new(2.0, -5.0, 2.0).normalize();
    scene.lights.push(RenderLight {
        kind: LightKind::Spot,
        color: [1.0, 0.95, 0.8],
        intensity: 60.0,
        direction: -emit,
        position: DVec3::new(0.0, 5.0, 0.0),
        range: 20.0,
        inner_cos: 15f32.to_radians().cos(),
        outer_cos: 25f32.to_radians().cos(),
        cast_shadows: false,
    });
    scene.mark_dirty();

    let img = check_golden(&gpu, "spot_lights", &scene, &overlook_view());

    // The cone lights a bright patch.
    let bright = img.chunks(4).any(|p| p[0] > 90 || p[1] > 90 || p[2] > 90);
    assert!(bright, "expected a lit spot pixel");

    // Swapping the spot for a plain point light (same position/intensity) lights
    // the ground far more broadly — the frames must differ, proving the cone mask.
    let mut point = scene.clone();
    point.lights[0].kind = LightKind::Point;
    point.mark_dirty();
    let point_img = render(&gpu, &point, &overlook_view());
    let (mean, _max) = image_diff(&img, &point_img, W, H);
    assert!(
        mean > 0.002,
        "spot cone should differ from a point light (mean {mean})"
    );
}

/// 2.5D billboard golden (P8.4a): an **angled perspective** camera over a row of
/// sprites — one flat (planar, in the world XY plane), one spherical billboard,
/// one cylindrical billboard — plus a ground grid for depth context. Under the
/// oblique view the planar sprite is seen edge-on/foreshortened while the two
/// billboards turn to face the camera, proving the vertex-shader orientation
/// (determinism gate via `check_golden`; strict pixel diff opt-in). The camera
/// basis rides in the view uniform (`cam_right`/`cam_up`).
#[test]
fn golden_billboards() {
    let Some(gpu) = gpu_or_skip() else { return };

    const TEX: u64 = 0xB1;
    let mut scene = RenderScene {
        grid_enabled: true,
        pending_texture_uploads: vec![SpriteTextureUpload {
            handle: TEX,
            width: 64,
            height: 64,
            rgba8: checkerboard(64, 4, [230, 80, 40], 255),
        }],
        ..Default::default()
    };

    // Three cards standing on the ground plane (pivot bottom-centre), spread
    // along X, each with a distinct billboard mode + tint.
    for (x, mode, tint) in [
        (-2.2f64, BILLBOARD_NONE, [1.0f32, 0.3, 0.3, 1.0]),
        (0.0, BILLBOARD_SPHERICAL, [0.3, 1.0, 0.4, 1.0]),
        (2.2, BILLBOARD_CYLINDRICAL, [0.4, 0.5, 1.0, 1.0]),
    ] {
        scene.sprites.push(SpriteInstance {
            position: DVec3::new(x, 1.0, 0.0),
            size: Vec2::new(1.6, 2.0),
            pivot: Vec2::new(0.5, 0.5),
            color: tint,
            texture: TEX,
            sorting_layer: 1,
            billboard: mode,
            ..Default::default()
        });
    }
    scene.mark_dirty();

    // The angled overlook view (perspective, camera at (6,4.5,9) → origin).
    let img = check_golden(&gpu, "billboards", &scene, &overlook_view());

    // Each tinted billboard shows up: a red, a green and a blue sprite region.
    let (mut red, mut green, mut blue) = (false, false, false);
    for chunk in img.chunks(4) {
        let (r, g, b) = (chunk[0] as i32, chunk[1] as i32, chunk[2] as i32);
        if r > 130 && r - g > 60 && r - b > 60 {
            red = true;
        }
        if g > 130 && g - r > 50 && g - b > 40 {
            green = true;
        }
        if b > 130 && b - r > 40 && b - g > 40 {
            blue = true;
        }
    }
    assert!(red, "expected the planar (red) sprite");
    assert!(green, "expected the spherical (green) billboard");
    assert!(blue, "expected the cylindrical (blue) billboard");
}

/// The default four-layer splat palette (grass / rock / dirt / snow), mirroring
/// `inf_ecs::components::default_terrain_layers` (kept inline so inf-render stays
/// free of an inf-ecs dep). Used by the terrain goldens so the layer-blended
/// shading is exercised.
fn default_layers() -> [RenderTerrainLayer; 4] {
    [
        RenderTerrainLayer {
            albedo: [0.20, 0.34, 0.14, 1.0], // grass
            roughness: 0.92,
            tex_scale: 6.0,
            vt: Default::default(),
        },
        RenderTerrainLayer {
            albedo: [0.33, 0.30, 0.27, 1.0], // rock
            roughness: 0.85,
            tex_scale: 4.0,
            vt: Default::default(),
        },
        RenderTerrainLayer {
            albedo: [0.42, 0.30, 0.18, 1.0], // dirt
            roughness: 0.95,
            tex_scale: 5.0,
            vt: Default::default(),
        },
        RenderTerrainLayer {
            albedo: [0.86, 0.89, 0.94, 1.0], // snow
            roughness: 0.65,
            tex_scale: 10.0,
            vt: Default::default(),
        },
    ]
}

/// A procedural sine-hills terrain across `ntx × ntz` tiles, authored from one
/// global height function so tile edges are seamless. `res` samples/tile, `mps`
/// metres/sample. Tiles are pushed in `(i32,i32)`-sorted order (matching the
/// host's BTreeMap projection). Unpainted (uniform layer 0 = grass) — the splat
/// golden authors real weight gradients.
fn hill_terrain(res: u32, mps: f64, ntx: i32, ntz: i32) -> RenderTerrain {
    let span = (res as f64 - 1.0) * mps;
    let f = |x: f64, z: f64| 4.0 * (x * 0.15).sin() * (z * 0.15).cos() + 3.5;
    let mut tiles = Vec::new();
    for tx in 0..ntx {
        for tz in 0..ntz {
            let (ox, oz) = (tx as f64 * span, tz as f64 * span);
            let mut heights = vec![0f32; (res * res) as usize];
            let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
            for j in 0..res {
                for i in 0..res {
                    let h = f(ox + i as f64 * mps, oz + j as f64 * mps) as f32;
                    heights[(j * res + i) as usize] = h;
                    lo = lo.min(h);
                    hi = hi.max(h);
                }
            }
            tiles.push(RenderTerrainTile {
                key: TerrainTileKey::lod0((tx, tz)),
                origin: DVec3::new(ox, 0.0, oz),
                heights,
                weights: Vec::new(),
                biomes: Vec::new(),
                height_bounds: (lo, hi),
                holes: Vec::new(),
                version: 1,
            });
        }
    }
    RenderTerrain {
        id: 0,
        tile_resolution: res,
        meters_per_sample: mps,
        tiles,
        layers: default_layers(),
        macro_variation: 0.15,
        biome_palette: Vec::new(),
    }
}

/// A perspective view from `eye` looking at `target`.
fn look_view(eye: DVec3, target: DVec3) -> RenderView {
    RenderView {
        origin: FloatingOrigin::new(DVec3::ZERO),
        eye_world: eye,
        forward: (target - eye).as_vec3().normalize(),
        up: Vec3::Y,
        fov_y: 60f32.to_radians(),
        near: 0.05,
        width: W,
        height: H,
        ortho: None,
    }
}

/// Terrain golden (P10.1 geometry, P10.4 shading): a sine-hills heightfield
/// across 2×2 tiles under an angled perspective camera, showing the terrain
/// silhouette + **splat-blended layer shading** against the sky. Unpainted, so
/// weights are uniform layer 0 (grass) — this golden was **regenerated for
/// P10.4** because the shading changed from the old slope/altitude debug ramp to
/// the layer-based blend (albedo + triplanar grain + macro variation). Exercises
/// the clipmap patch assembly → height/weight texture → vertex displacement path
/// headlessly (determinism gate via `check_golden`; strict pixel diff opt-in).
#[test]
fn golden_terrain() {
    let Some(gpu) = gpu_or_skip() else { return };
    let res = 33;
    let terrain = hill_terrain(res, 1.0, 2, 2); // 2×2 tiles, ~64 m square
    let scene = RenderScene {
        grid_enabled: true,
        terrains: vec![terrain],
        ..Default::default()
    };
    // Angled overlook of the terrain centre (~(32, ·, 32)).
    let view = look_view(DVec3::new(32.0, 24.0, -12.0), DVec3::new(32.0, 3.0, 32.0));
    let img = check_golden(&gpu, "terrain", &scene, &view);

    // The lower band (terrain) is lit and clearly differs from the sky band above.
    let sky = px(&img, W / 2, 6);
    let ground = px(&img, W / 2, H - 12);
    assert_ne!(sky, ground, "terrain band should differ from sky");
    let lit = img
        .chunks(4)
        .any(|p| p[0] as u16 + p[1] as u16 + p[2] as u16 > 150);
    assert!(lit, "expected a lit terrain pixel");
}

/// Terrain LOD golden (P10.1): the same hills across a long 6×2 strip with the
/// camera at the near end, so tiles resolve to ≥2 distinct clipmap LOD rings by
/// distance. Structural gate: assembly yields multiple LODs + the frame renders
/// deterministically.
#[test]
fn golden_terrain_lod() {
    let Some(gpu) = gpu_or_skip() else { return };
    let res = 33;
    let mps = 1.0;
    let terrain = hill_terrain(res, mps, 6, 2); // long strip along +X
                                                // Camera at the near (-X) end looking down the strip: near tiles LOD 0, far
                                                // tiles coarsen → concentric rings.
    let view = look_view(DVec3::new(-6.0, 40.0, 32.0), DVec3::new(140.0, 0.0, 32.0));

    // The pure assembly must produce ≥2 distinct LOD levels (the "≥2 rings" gate).
    let patches = assemble_patches(&terrain, &view, &view.origin);
    let mut lods: Vec<u32> = patches.iter().map(|p| p.ring).collect();
    lods.sort_unstable();
    lods.dedup();
    assert!(
        lods.len() >= 2,
        "expected ≥2 LOD rings, got LODs {lods:?} from {} patches",
        patches.len()
    );

    let scene = RenderScene {
        grid_enabled: true,
        terrains: vec![terrain],
        ..Default::default()
    };
    let img = check_golden(&gpu, "terrain_lod", &scene, &view);
    let lit = img
        .chunks(4)
        .any(|p| p[0] as u16 + p[1] as u16 + p[2] as u16 > 150);
    assert!(lit, "expected a lit terrain pixel");
}

/// A splat-painted terrain (P10.4): 2×2 tiles with hand-authored weight gradients
/// banding all four layers across +X (grass → dirt → rock → snow), plus a **steep
/// cliff** wall so the triplanar detail path is exercised on near-vertical faces.
/// Seamless across tile edges (weights authored from one global world function).
fn splat_terrain(res: u32, mps: f64, ntx: i32, ntz: i32) -> RenderTerrain {
    let span = (res as f64 - 1.0) * mps;
    let total_w = ntx as f64 * span;
    // A steep cliff wall at ~63% of the width (6 m rise over a ~4% band) over a
    // gently rolling base — the wall's near-vertical normals drive triplanar.
    let smoothstep = |e0: f64, e1: f64, x: f64| {
        let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    };
    let height = |x: f64, z: f64| {
        2.0 + 6.0 * smoothstep(0.60 * total_w, 0.64 * total_w, x) + 0.6 * (z * 0.2).sin()
    };
    // Four tent bands across the normalized width → four distinct, blended layers.
    let weight = |x: f64| -> [u8; 4] {
        let u = (x / total_w).clamp(0.0, 1.0);
        let tent = |c: f64| (1.0 - (u - c).abs() * 3.0).max(0.0);
        let raw = [tent(0.0), tent(1.0 / 3.0), tent(2.0 / 3.0), tent(1.0)];
        let s: f64 = raw.iter().sum::<f64>().max(1e-6);
        let mut out = [0u8; 4];
        let mut acc = 0i32;
        for k in 0..4 {
            out[k] = (raw[k] / s * 255.0).round() as u8;
            acc += out[k] as i32;
        }
        out[0] = (out[0] as i32 + (255 - acc)).clamp(0, 255) as u8; // exact sum 255
        out
    };
    let mut tiles = Vec::new();
    for tx in 0..ntx {
        for tz in 0..ntz {
            let (ox, oz) = (tx as f64 * span, tz as f64 * span);
            let mut heights = vec![0f32; (res * res) as usize];
            let mut weights = vec![[0u8; 4]; (res * res) as usize];
            let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
            for j in 0..res {
                for i in 0..res {
                    let (wx, wz) = (ox + i as f64 * mps, oz + j as f64 * mps);
                    let h = height(wx, wz) as f32;
                    heights[(j * res + i) as usize] = h;
                    weights[(j * res + i) as usize] = weight(wx);
                    lo = lo.min(h);
                    hi = hi.max(h);
                }
            }
            tiles.push(RenderTerrainTile {
                key: TerrainTileKey::lod0((tx, tz)),
                origin: DVec3::new(ox, 0.0, oz),
                heights,
                weights,
                // Unpainted — the biomes golden paints this fixture's ids itself.
                biomes: Vec::new(),
                height_bounds: (lo, hi),
                holes: Vec::new(),
                version: 1,
            });
        }
    }
    RenderTerrain {
        id: 0,
        tile_resolution: res,
        meters_per_sample: mps,
        tiles,
        layers: default_layers(),
        macro_variation: 0.15,
        biome_palette: Vec::new(),
    }
}

/// Terrain splat golden (P10.4): a heightfield with hand-authored weight gradients
/// banding all four material layers across +X plus a steep cliff, proving the
/// splat blend + triplanar path headlessly (determinism gate via `check_golden`;
/// strict pixel diff opt-in).
#[test]
fn golden_terrain_splat() {
    let Some(gpu) = gpu_or_skip() else { return };
    let res = 33;
    let terrain = splat_terrain(res, 1.0, 2, 2); // ~64 m square, banded layers
    let scene = RenderScene {
        grid_enabled: true,
        terrains: vec![terrain],
        ..Default::default()
    };
    // Angled overlook of the banded terrain, side-on to the cliff.
    let view = look_view(DVec3::new(4.0, 22.0, -10.0), DVec3::new(40.0, 3.0, 32.0));
    let img = check_golden(&gpu, "terrain_splat", &scene, &view);

    // The four layers span from a green (grass) low band to a bright (snow) high
    // band — assert both a greenish and a bright near-white terrain pixel exist.
    let mut green = false;
    let mut snow = false;
    for chunk in img.chunks(4) {
        let (r, g, b) = (chunk[0] as i32, chunk[1] as i32, chunk[2] as i32);
        if g > 60 && g - r > 20 && g - b > 20 {
            green = true;
        }
        if r > 180 && g > 180 && b > 180 {
            snow = true;
        }
    }
    assert!(green, "expected the grass (green) layer band");
    assert!(snow, "expected the snow (bright) layer band");
}

/// The reserved "no biome" colour (mirrors `inf_terrain::UNASSIGNED_BIOME_COLOR`,
/// which the renderer also mirrors locally — see `passes::terrain`).
const UNASSIGNED_BIOME_COLOR: [f32; 4] = [0.22, 0.22, 0.24, 1.0];

/// A biome palette in the shape `inf_terrain::BiomeSet::palette` produces:
/// **index = biome id**, slot 0 reserved, three strongly-separated hues so a
/// structural per-channel assertion can name which biome a pixel came from.
fn biome_palette() -> Vec<[f32; 4]> {
    vec![
        UNASSIGNED_BIOME_COLOR,  // 0 — reserved / unpainted
        [0.90, 0.10, 0.10, 1.0], // 1 — red
        [0.10, 0.85, 0.15, 1.0], // 2 — green
        [0.15, 0.25, 0.95, 1.0], // 3 — blue
    ]
}

/// The [`splat_terrain`] fixture with **biome ids painted** across +X (P19.2):
/// four equal bands carrying ids 0, 1, 2, 3, so one frame shows both a real biome
/// vocabulary and the reserved "nothing painted here" id. Authored from one global
/// world function, so the bands are seamless across the tile edges.
fn biome_terrain(res: u32, mps: f64, ntx: i32, ntz: i32, palette: Vec<[f32; 4]>) -> RenderTerrain {
    let mut terrain = splat_terrain(res, mps, ntx, ntz);
    let span = (res as f64 - 1.0) * mps;
    let total_w = ntx as f64 * span;
    for tile in &mut terrain.tiles {
        let ox = tile.origin.x;
        let mut ids = vec![0u8; (res * res) as usize];
        for j in 0..res {
            for i in 0..res {
                let u = ((ox + i as f64 * mps) / total_w).clamp(0.0, 1.0);
                ids[(j * res + i) as usize] = (u * 4.0).floor().clamp(0.0, 3.0) as u8;
            }
        }
        tile.biomes = ids;
    }
    terrain.biome_palette = palette;
    terrain
}

/// Biomes view-mode golden (P19.2): the splat-painted heightfield with biome ids
/// banded across +X, rendered with `set_view_mode(Biomes)` so the terrain is
/// tinted by `RenderTerrain::biome_palette` instead of shaded. Determinism gate
/// (render twice), a new committed golden `biomes.png` (bless with
/// `INF_BLESS_GOLDENS=1`), and four structural gates:
///
/// * the Biomes frame differs from the Lit one (the flag genuinely changed the
///   shading);
/// * the painted bands read the colours the palette gave them (a per-channel
///   dominance count, not a pixel hash — adapter-robust);
/// * **swapping two palette entries changes the frame** — the proof that ids index
///   the palette *by id*, rather than the shader inventing a colour per band;
/// * the *Lit* frame stays byte-identical to the plain renderer, which is the
///   OFF-path guarantee: `flags.y` is exactly 0 in every other mode, so no
///   pre-P19.2 golden can move.
///
/// The grid is off on purpose: its world axes draw in red/blue, which would put
/// those hues in the frame from something that is not a biome.
#[test]
fn golden_biomes() {
    let Some(gpu) = gpu_or_skip() else { return };
    let res = 33;
    let palette = biome_palette();
    let terrain = biome_terrain(res, 1.0, 2, 2, palette.clone());
    let scene = RenderScene {
        grid_enabled: false,
        terrains: vec![terrain],
        ..Default::default()
    };
    // The splat golden's camera: angled overlook, side-on to the cliff, all four
    // bands in frame.
    let view = look_view(DVec3::new(4.0, 22.0, -10.0), DVec3::new(40.0, 3.0, 32.0));

    // Determinism + golden write/compare for the Biomes render.
    let a = render_view_mode(&gpu, &scene, &view, ViewMode::Biomes);
    let b = render_view_mode(&gpu, &scene, &view, ViewMode::Biomes);
    let (mean, max) = image_diff(&a, &b, W, H);
    assert!(
        mean < 0.005 && max < 0.05,
        "biomes: renderer not deterministic (mean {mean}, max {max})"
    );
    let path = goldens_dir().join("biomes.png");
    if std::env::var("INF_BLESS_GOLDENS").is_ok() || read_png(&path).is_none() {
        write_png(&path, &a);
        eprintln!("golden biomes: wrote {}", path.display());
    } else if std::env::var("INF_GOLDEN_STRICT").is_ok() {
        let golden = read_png(&path).expect("golden png");
        let (m, mx) = image_diff(&a, &golden, W, H);
        assert!(
            within_tolerance(m, mx),
            "biomes: differs from golden (mean {m}, max {mx})"
        );
    }

    // The tinted frame differs from the lit one.
    let lit = render_view_mode(&gpu, &scene, &view, ViewMode::Lit);
    let (dmean, _dmax) = image_diff(&a, &lit, W, H);
    assert!(
        dmean > 0.002,
        "biomes should differ from lit (mean {dmean})"
    );

    // The painted bands read their palette colours. The tint is scaled by a
    // wrapped N·L (0.55…1.0), which cannot move a hue between channels, so
    // per-channel dominance is a safe structural claim.
    let (mut red, mut green) = (0u32, 0u32);
    for chunk in a.chunks(4) {
        let (r, g, b) = (chunk[0] as i32, chunk[1] as i32, chunk[2] as i32);
        if r > 60 && r - g > 40 && r - b > 40 {
            red += 1;
        }
        if g > 60 && g - r > 40 && g - b > 40 {
            green += 1;
        }
    }
    assert!(
        red > 200,
        "expected the id-1 (red) biome band, saw {red} px"
    );
    assert!(
        green > 200,
        "expected the id-2 (green) biome band, saw {green} px"
    );

    // Swap the colours of ids 1 and 2 in the palette — the SAME painted ids must
    // now draw differently. This is what pins "the palette is indexed by id": a
    // shader that derived a colour from the band, or read the palette by array
    // position within the level's biome list, would be unmoved by this.
    let mut swapped_palette = palette.clone();
    swapped_palette.swap(1, 2);
    let swapped_scene = RenderScene {
        grid_enabled: false,
        terrains: vec![biome_terrain(res, 1.0, 2, 2, swapped_palette)],
        ..Default::default()
    };
    let swapped = render_view_mode(&gpu, &swapped_scene, &view, ViewMode::Biomes);
    let (smean, _smax) = image_diff(&a, &swapped, W, H);
    assert!(
        smean > 0.002,
        "swapping two palette entries must repaint the bands (mean {smean})"
    );

    // The default Lit path is byte-stable vs the plain renderer — view mode never
    // perturbs Lit, so no committed golden can move (the OFF-path proof).
    let plain = render(&gpu, &scene, &view);
    let (lmean, lmax) = image_diff(&lit, &plain, W, H);
    assert!(
        lmean < 1e-6 && lmax < 1e-6,
        "Lit view mode must match the default renderer exactly (mean {lmean}, max {lmax})"
    );
}

/// A synthetic **streamed** terrain (P16.3b1): three asset LOD levels over one
/// global height function, handed to the renderer as a quadtree cut instead of a
/// fully-resident heightfield.
///
/// * level 0 — only the 2 × 2 block at the origin is resident (a deliberately
///   *partial* level-0 residency, the streaming shape);
/// * level 1 — the 2 × 2 block covering 4 × that area (the mid ring);
/// * level 2 — the 2 × 2 block covering 16 × it, **minus `(1,1)`** so one far
///   quadrant is genuinely uncovered (the renderer must render the hole, not
///   invent coverage).
///
/// Coarse pages sample the same global function at `2^lod ·` the spacing, so —
/// exactly like the real `inf_terrain::pyramid` decimation — every coarse sample
/// *is* one of the fine samples, and the shared edges agree bit-for-bit.
fn streamed_terrain(res: u32, mps: f64) -> RenderTerrain {
    let f = |x: f64, z: f64| 5.0 * (x * 0.04).sin() * (z * 0.04).cos() + 4.0;
    let span0 = (res as f64 - 1.0) * mps;
    let page = |lod: u32, coord: (i32, i32), version: u64| {
        let step = mps * (1u64 << lod) as f64;
        let span = span0 * (1u64 << lod) as f64;
        let (ox, oz) = (coord.0 as f64 * span, coord.1 as f64 * span);
        let mut heights = vec![0f32; (res * res) as usize];
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for j in 0..res {
            for i in 0..res {
                let h = f(ox + i as f64 * step, oz + j as f64 * step) as f32;
                heights[(j * res + i) as usize] = h;
                lo = lo.min(h);
                hi = hi.max(h);
            }
        }
        RenderTerrainTile {
            key: TerrainTileKey::new(lod, coord),
            origin: DVec3::new(ox, 0.0, oz),
            heights,
            weights: Vec::new(),
            biomes: Vec::new(),
            height_bounds: (lo, hi),
            holes: Vec::new(),
            version: 1 + version,
        }
    };
    let block = [(0, 0), (0, 1), (1, 0), (1, 1)];
    // Key-ascending (level 0, then level 1, then level 2) — the projection order.
    let mut tiles = Vec::new();
    for (lod, coords) in [
        (0u32, &block[..]),
        (1, &block[..]),
        (2, &block[..3]), // (1,1) deliberately absent
    ] {
        for &c in coords {
            tiles.push(page(lod, c, tiles.len() as u64));
        }
    }
    RenderTerrain {
        id: 0,
        tile_resolution: res,
        meters_per_sample: mps,
        tiles,
        layers: default_layers(),
        macro_variation: 0.15,
        biome_palette: Vec::new(),
    }
}

/// Streamed-terrain headless gate (P16.3b1). A partially-resident level 0 with
/// coarse pyramid pages covering the outer rings must render deterministically —
/// **including across frames that share one renderer**, where the second frame's
/// per-tile version gate finds every stamp unchanged and uploads nothing. A
/// regression that dropped or half-refreshed a cached page would show up here as
/// a differing frame.
///
/// Deliberately **not** a committed golden PNG: the pixels exercise the same
/// shading path the three terrain goldens already pin, while everything this
/// batch adds (which page sources which patch) is asserted structurally, which is
/// adapter-robust — the harness's stated bar for what CI can actually check.
#[test]
fn streamed_terrain_renders_partial_residency() {
    let Some(gpu) = gpu_or_skip() else { return };
    let res = 33;
    let terrain = streamed_terrain(res, 1.0);
    // Overlook from the near corner down the +X/+Z diagonal, so the resident
    // level-0 block is close and the coarse pages recede into the outer rings.
    let view = look_view(
        DVec3::new(-20.0, 60.0, -20.0),
        DVec3::new(120.0, 0.0, 120.0),
    );

    // ── structural: source selection over the residency set ──────────────────
    let patches = assemble_patches(&terrain, &view, &view.origin);
    assert_eq!(
        patches,
        assemble_patches(&terrain, &view, &view.origin),
        "assembly must be a pure function of (residency set, view)"
    );
    let drawn: Vec<TerrainTileKey> = patches.iter().map(|p| p.key).collect();
    // Fine wins: every resident level-0 page draws …
    for c in [(0, 0), (0, 1), (1, 0), (1, 1)] {
        assert!(
            drawn.contains(&TerrainTileKey::lod0(c)),
            "level-0 page {c:?} must draw (fine wins)"
        );
    }
    // … and the coarse pages whose whole footprint they cover stand down.
    assert!(
        !drawn.contains(&TerrainTileKey::new(1, (0, 0))),
        "the fully-subdivided level-1 page must not double-draw over level 0"
    );
    assert!(
        !drawn.contains(&TerrainTileKey::new(2, (0, 0))),
        "the fully-subdivided level-2 page must not double-draw"
    );
    // Coarse pages serve the outer coverage, at ≥2 distinct asset levels.
    let mut levels: Vec<u32> = drawn.iter().map(|k| k.lod).collect();
    levels.sort_unstable();
    levels.dedup();
    assert!(
        levels.len() >= 2 && levels.contains(&0),
        "expected fine + coarse sources, got asset levels {levels:?}"
    );
    // The absent page is simply not drawn — a hole, faithfully rendered.
    assert!(!drawn.contains(&TerrainTileKey::new(2, (1, 1))));
    // Nothing is drawn twice, and a coarse patch keeps the full-density grid its
    // level already decimated for (ring − lod).
    let mut unique = drawn.clone();
    unique.sort_unstable();
    unique.dedup();
    assert_eq!(unique.len(), drawn.len(), "a page was assembled twice");
    for p in &patches {
        assert_eq!(p.mesh_lod, inf_render::patch_mesh_lod(p.ring, p.key.lod));
        assert_eq!(terrain.tiles[p.tile].key, p.key);
    }

    // ── determinism: two fresh renderers, then two frames on a warm cache ─────
    let scene = RenderScene {
        grid_enabled: true,
        terrains: vec![terrain],
        ..Default::default()
    };
    let cold_a = render(&gpu, &scene, &view);
    let cold_b = render(&gpu, &scene, &view);
    assert_eq!(
        cold_a, cold_b,
        "streamed terrain must render deterministically"
    );

    let target = HeadlessTarget::new(&gpu, W, H);
    let mut renderer = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
    renderer.render(&gpu, &scene, &view, &target.view, (W, H));
    let warm_a = target.read_rgba(&gpu).expect("readback");
    renderer.render(&gpu, &scene, &view, &target.view, (W, H));
    let warm_b = target.read_rgba(&gpu).expect("readback");
    assert_eq!(
        warm_a, warm_b,
        "a second frame over an unchanged residency set uploads nothing and must \
         be byte-identical"
    );
    assert_eq!(
        cold_a, warm_a,
        "a warm tile cache must render the cold frame"
    );

    let lit = warm_a
        .chunks(4)
        .any(|p| p[0] as u16 + p[1] as u16 + p[2] as u16 > 150);
    assert!(lit, "expected a lit terrain pixel");
}

/// Shift every tile of `t` by `offset` and stamp it with `id` — a second terrain
/// placed elsewhere in the world while keeping the *same* tile coordinates, which
/// is the collision case the P16.6 cache key exists for.
fn placed_terrain(mut t: RenderTerrain, id: u64, offset: DVec3) -> RenderTerrain {
    t.id = id;
    for tile in &mut t.tiles {
        tile.origin += offset;
    }
    t
}

/// **P16.6 multi-terrain headless gate.** Two independent terrains — same tile
/// coordinates, different world anchors, different splat layers — render in one
/// frame, deterministically, and both are actually drawn.
///
/// Deliberately **not** a committed golden PNG, following the streamed-terrain
/// precedent above: the shading path is already pinned by the three terrain
/// goldens, and everything this batch adds (per-terrain cache slots, per-terrain
/// material uniforms, one instance buffer across both patch lists) is asserted
/// structurally, which is adapter-robust — the harness's stated bar for what CI
/// can actually check. The single-terrain goldens are what pin byte-identity.
#[test]
fn two_terrains_render_independently_in_one_frame() {
    let Some(gpu) = gpu_or_skip() else { return };
    let res = 17;
    let a = placed_terrain(hill_terrain(res, 1.0, 2, 2), 1, DVec3::ZERO);
    // B sits 40 m down +X, so both are in frame at once, and it paints from a
    // different layer set so a shared material uniform would be visible.
    let mut b = placed_terrain(hill_terrain(res, 1.0, 2, 2), 2, DVec3::new(40.0, 6.0, 0.0));
    b.layers[0].albedo = [0.75, 0.18, 0.12, 1.0];
    b.macro_variation = 0.0;

    // The fixture only bites while the two terrains share tile coordinates.
    let keys_a: Vec<_> = a.tiles.iter().map(|t| t.key).collect();
    let keys_b: Vec<_> = b.tiles.iter().map(|t| t.key).collect();
    assert_eq!(keys_a, keys_b, "the two terrains must share tile keys");

    let view = look_view(DVec3::new(16.0, 34.0, -26.0), DVec3::new(36.0, 4.0, 16.0));

    // Both terrains assemble patches under this view (each against its OWN grid).
    let pa = assemble_patches(&a, &view, &view.origin);
    let pb = assemble_patches(&b, &view, &view.origin);
    assert!(!pa.is_empty() && !pb.is_empty(), "both must be visible");

    let one = RenderScene {
        grid_enabled: true,
        terrains: vec![a.clone()],
        ..Default::default()
    };
    let both = RenderScene {
        grid_enabled: true,
        terrains: vec![a, b],
        ..Default::default()
    };

    // Determinism: two fresh renderers over the two-terrain scene agree…
    let cold_a = render(&gpu, &both, &view);
    let cold_b = render(&gpu, &both, &view);
    assert_eq!(cold_a, cold_b, "two terrains must render deterministically");

    // …and a warm cache (second frame, every stamp unchanged, nothing uploaded)
    // reproduces the cold frame — the per-terrain cache slots stay in step.
    let target = HeadlessTarget::new(&gpu, W, H);
    let mut renderer = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
    renderer.render(&gpu, &both, &view, &target.view, (W, H));
    let warm_a = target.read_rgba(&gpu).expect("readback");
    renderer.render(&gpu, &both, &view, &target.view, (W, H));
    let warm_b = target.read_rgba(&gpu).expect("readback");
    assert_eq!(warm_a, warm_b, "a warm two-terrain frame must be stable");
    assert_eq!(cold_a, warm_a, "warm != cold for two terrains");

    // The second terrain really contributed pixels.
    let solo = render(&gpu, &one, &view);
    assert_ne!(
        solo, cold_a,
        "adding the second terrain changed nothing — it never drew"
    );

    // Dropping a terrain from the scene mid-session frees its cache and still
    // renders the survivor exactly as a fresh renderer would.
    renderer.render(&gpu, &one, &view, &target.view, (W, H));
    let after_drop = target.read_rgba(&gpu).expect("readback");
    assert_eq!(
        after_drop, solo,
        "evicting terrain B's pages perturbed terrain A"
    );
}

/// A view looking straight down -Z at the world XY plane, so sprites (which lie
/// in that plane facing +Z) face the camera head-on.
fn front_view() -> RenderView {
    RenderView {
        origin: FloatingOrigin::new(DVec3::ZERO),
        eye_world: DVec3::new(0.0, 0.0, 6.0),
        forward: Vec3::NEG_Z,
        up: Vec3::Y,
        fov_y: 60f32.to_radians(),
        near: 0.05,
        width: W,
        height: H,
        ortho: None,
    }
}

/// A top-down orthographic view over the world XY plane (2D editor mode): eye at
/// +Z looking down -Z, up = +Y. Half-height 4 world units frames a small patch.
fn ortho_view() -> RenderView {
    RenderView {
        origin: FloatingOrigin::new(DVec3::ZERO),
        eye_world: DVec3::new(0.0, 0.0, 100.0),
        forward: Vec3::NEG_Z,
        up: Vec3::Y,
        fov_y: 60f32.to_radians(),
        near: 1.0,
        width: W,
        height: H,
        ortho: Some(inf_render::OrthoParams {
            half_height: 4.0,
            near: 1.0,
            far: 200.0,
        }),
    }
}

/// Procedural checkerboard: `cells×cells` grid over `size×size` px alternating
/// `color` and white; `alpha` sets the color cells' opacity.
fn checkerboard(size: u32, cells: u32, color: [u8; 3], alpha: u8) -> Vec<u8> {
    let cell = (size / cells).max(1);
    let mut v = Vec::with_capacity((size * size * 4) as usize);
    for y in 0..size {
        for x in 0..size {
            let on = ((x / cell) + (y / cell)).is_multiple_of(2);
            if on {
                v.extend_from_slice(&[color[0], color[1], color[2], alpha]);
            } else {
                v.extend_from_slice(&[235, 235, 235, 255]);
            }
        }
    }
    v
}

/// 2D sprite golden (P8.1a): two textured, alpha-blended sprites on distinct
/// sorting layers, backed by in-test procedural checkerboards (no binary
/// fixtures). Exercises the batcher → texture cache → sprite pass path in CI
/// (determinism gate via `check_golden`; strict pixel diff is opt-in).
#[test]
fn golden_sprites_2d() {
    let Some(gpu) = gpu_or_skip() else { return };

    const TEX_A: u64 = 0xA1;
    const TEX_B: u64 = 0xB2;
    let mut scene = RenderScene {
        grid_enabled: true,
        pending_texture_uploads: vec![
            SpriteTextureUpload {
                handle: TEX_A,
                width: 64,
                height: 64,
                rgba8: checkerboard(64, 8, [220, 40, 40], 255),
            },
            SpriteTextureUpload {
                handle: TEX_B,
                width: 64,
                height: 64,
                rgba8: checkerboard(64, 8, [40, 90, 220], 255),
            },
        ],
        ..Default::default()
    };

    // Two overlapping sprites: the blue one (higher layer) draws over the red.
    scene.sprites.push(SpriteInstance {
        position: DVec3::new(-0.6, 0.0, 0.0),
        size: Vec2::new(2.4, 2.4),
        color: [1.0, 1.0, 1.0, 1.0],
        texture: TEX_A,
        sorting_layer: 0,
        ..Default::default()
    });
    scene.sprites.push(SpriteInstance {
        position: DVec3::new(0.6, 0.0, 0.0),
        size: Vec2::new(2.4, 2.4),
        color: [1.0, 1.0, 1.0, 1.0],
        texture: TEX_B,
        sorting_layer: 1,
        ..Default::default()
    });
    scene.mark_dirty();

    let img = check_golden(&gpu, "sprites_2d", &scene, &front_view());

    // Both sprites are visible: a red checker cell (texture A, its non-overlapped
    // left half) and a blue one (texture B, drawn on top on the higher layer),
    // plus the shared bright checker cells.
    let mut red = false;
    let mut blue = false;
    let mut bright = false;
    for chunk in img.chunks(4) {
        let (r, g, b) = (chunk[0] as i32, chunk[1] as i32, chunk[2] as i32);
        if r > 140 && r - b > 60 && r - g > 60 {
            red = true;
        }
        if b > 120 && b - r > 40 {
            blue = true;
        }
        if r > 200 && g > 200 && b > 200 {
            bright = true;
        }
    }
    assert!(red, "expected a red sprite checker cell");
    assert!(blue, "expected a blue sprite checker cell");
    assert!(bright, "expected bright sprite checker cells");
}

/// A `size×size` RGBA atlas of four solid-color quadrants (2×2 grid), laid out
/// so 1-based tile indices 1..=4 map to top-left, top-right, bottom-left,
/// bottom-right respectively (row-major, row 0 = the atlas top row).
fn quad_atlas(size: u32, cells: [[u8; 3]; 4]) -> Vec<u8> {
    let half = size / 2;
    let mut v = Vec::with_capacity((size * size * 4) as usize);
    for y in 0..size {
        for x in 0..size {
            let q = (y >= half) as usize * 2 + (x >= half) as usize; // 0=TL,1=TR,2=BL,3=BR
            let c = cells[q];
            v.extend_from_slice(&[c[0], c[1], c[2], 255]);
        }
    }
    v
}

/// 2D tilemap golden (P8.1b): an in-test procedural 4-cell atlas painted across
/// a patch of tiles that straddles **two** chunks, with one loose sprite on a
/// higher sorting layer drawn over the tiles (an ordering proof). Exercises the
/// chunk cull → expansion → prebatched-run → sprite-pass path headlessly
/// (determinism gate via `check_golden`; strict pixel diff opt-in).
#[test]
fn golden_tilemap_2d() {
    let Some(gpu) = gpu_or_skip() else { return };

    const ATLAS: u64 = 0x71;
    // 1→red(TL), 2→green(TR), 3→blue(BL), 4→yellow(BR).
    let atlas = quad_atlas(
        64,
        [[220, 40, 40], [40, 200, 60], [60, 90, 220], [230, 210, 40]],
    );

    let tile = 0.3_f64;
    let dim = TILE_CHUNK_DIM as f64;
    let params = TilemapParams {
        // Place the vertical chunk boundary (global tile x=32) at world x=0 and
        // the row gy=16 at world y=0, so the painted patch centers on screen.
        origin: DVec3::new(-dim * tile, -dim * 0.5 * tile, 0.0),
        tile_size: Vec2::new(tile as f32, tile as f32),
        atlas_cols: 2,
        atlas_rows: 2,
        texture: ATLAS,
        color: [1.0, 1.0, 1.0, 1.0],
        sorting_layer: 0,
        order: 0,
    };

    // Paint tiles gx∈[28,36) gy∈[14,18): 8×4 = 32 tiles spanning chunk (0,0)
    // (gx 28..31) and chunk (1,0) (gx 32..35). Index cycles 1..=4.
    let n = (TILE_CHUNK_DIM * TILE_CHUNK_DIM) as usize;
    let mut chunk0 = vec![0u32; n];
    let mut chunk1 = vec![0u32; n];
    for gy in 14..18i32 {
        for gx in 28..36i32 {
            let idx = (((gx + gy).rem_euclid(4)) + 1) as u32;
            let (cx, lx) = (gx.div_euclid(TILE_CHUNK_DIM), gx.rem_euclid(TILE_CHUNK_DIM));
            let ly = gy.rem_euclid(TILE_CHUNK_DIM);
            let slot = (ly * TILE_CHUNK_DIM + lx) as usize;
            match cx {
                0 => chunk0[slot] = idx,
                1 => chunk1[slot] = idx,
                _ => unreachable!(),
            }
        }
    }

    let mut scene = RenderScene {
        grid_enabled: true,
        pending_texture_uploads: vec![SpriteTextureUpload {
            handle: ATLAS,
            width: 64,
            height: 64,
            rgba8: atlas,
        }],
        tilemaps: vec![RenderTilemap {
            params,
            chunks: vec![
                RenderChunk {
                    coord: (0, 0),
                    tiles: chunk0,
                },
                RenderChunk {
                    coord: (1, 0),
                    tiles: chunk1,
                },
            ],
        }],
        ..Default::default()
    };

    // A loose magenta sprite on a HIGHER sorting layer, centered over the tiles:
    // it must draw on top (proving loose-vs-prebatched ordering).
    scene.sprites.push(SpriteInstance {
        position: DVec3::new(0.0, 0.0, 0.0),
        size: Vec2::new(0.8, 0.8),
        color: [0.95, 0.15, 0.95, 1.0],
        sorting_layer: 1,
        ..Default::default()
    });
    scene.mark_dirty();

    let img = check_golden(&gpu, "tilemap_2d", &scene, &front_view());

    // At least two distinct atlas cells are visible (proves 1-based indexing +
    // ≥2 chunks expanded), and the loose magenta sprite paints over the center.
    let mut red = false;
    let mut blue = false;
    let mut magenta = false;
    for chunk in img.chunks(4) {
        let (r, g, b) = (chunk[0] as i32, chunk[1] as i32, chunk[2] as i32);
        if r > 140 && r - g > 80 && r - b > 80 {
            red = true;
        }
        if b > 140 && b - r > 60 && b - g > 40 {
            blue = true;
        }
        if r > 150 && b > 150 && r - g > 60 && b - g > 60 {
            magenta = true;
        }
    }
    assert!(red, "expected a red tile (atlas cell 1)");
    assert!(blue, "expected a blue tile (atlas cell 3)");
    assert!(magenta, "expected the loose magenta sprite over the tiles");
}

/// 2D lighting golden (P8.1c): a dark scene ambient with two colored 2D lights
/// (red on the left, blue on the right) over a big white sprite patch. The
/// `smoothstep` falloff paints a red glow on the left half, a blue glow on the
/// right, and near-black between/outside the radii — proving the sprite shader's
/// 2D-light path (determinism gate via `check_golden`; strict pixel diff opt-in).
#[test]
fn golden_2d_lit() {
    let Some(gpu) = gpu_or_skip() else { return };

    let mut scene = RenderScene {
        // Grid off so the sprite lighting reads without the grid underneath.
        grid_enabled: false,
        // Fully dark ambient: the two lights alone shape the image, and the
        // sprite's far corners stay black (the "dark region" assertion below).
        // (sRGB encoding lifts small linear values a lot, so a truly dark region
        // needs ~0 linear ambient.)
        ambient_2d: Ambient2D([0.0, 0.0, 0.0]),
        ..Default::default()
    };

    // One big white quad (the untextured white fallback) covering the frame, so
    // every sampled pixel is the lit sprite (no sky leaking into the readback).
    scene.sprites.push(SpriteInstance {
        position: DVec3::new(0.0, 0.0, 0.0),
        size: Vec2::new(40.0, 24.0),
        color: [1.0, 1.0, 1.0, 1.0],
        sorting_layer: 0,
        ..Default::default()
    });

    // Red light on the left, blue on the right.
    scene.lights_2d.push(RenderLight2D {
        color: [1.0, 0.1, 0.1],
        intensity: 1.5,
        radius: 2.2,
        position: DVec3::new(-1.4, 0.0, 0.0),
    });
    scene.lights_2d.push(RenderLight2D {
        color: [0.1, 0.2, 1.0],
        intensity: 1.5,
        radius: 2.2,
        position: DVec3::new(1.4, 0.0, 0.0),
    });
    scene.mark_dirty();

    let img = check_golden(&gpu, "2d_lit", &scene, &front_view());

    // The left glow is red-dominant, the right glow blue-dominant, and some
    // pixel is near-black (outside both radii / dark ambient).
    let mut red = false;
    let mut blue = false;
    let mut dark = false;
    for chunk in img.chunks(4) {
        let (r, g, b) = (chunk[0] as i32, chunk[1] as i32, chunk[2] as i32);
        if r > 90 && r - g > 50 && r - b > 50 {
            red = true;
        }
        if b > 90 && b - r > 50 && b - g > 30 {
            blue = true;
        }
        if r < 20 && g < 20 && b < 20 {
            dark = true;
        }
    }
    assert!(red, "expected the red 2D light glow");
    assert!(blue, "expected the blue 2D light glow");
    assert!(dark, "expected a dark region outside the light radii");
}

/// Orthographic 2D-editor golden (P8.2c): the ortho camera over a tile patch, a
/// loose sprite, and a built-in-font text run, with the XY grid enabled. Proves
/// the ortho projection + XY-grid shader path + the 2D content passes render
/// coherently under a parallel projection (determinism gate via `check_golden`;
/// strict pixel diff opt-in, blessed on a GPU host).
#[test]
fn golden_ortho_2d() {
    let Some(gpu) = gpu_or_skip() else { return };

    const ATLAS: u64 = 0x51;
    // 1→red(TL), 2→green(TR), 3→blue(BL), 4→yellow(BR).
    let atlas = quad_atlas(
        64,
        [[220, 40, 40], [40, 200, 60], [60, 90, 220], [230, 210, 40]],
    );
    let tile = 0.5_f64;
    let params = TilemapParams {
        origin: DVec3::new(-1.5, -1.5, 0.0),
        tile_size: Vec2::new(tile as f32, tile as f32),
        atlas_cols: 2,
        atlas_rows: 2,
        texture: ATLAS,
        color: [1.0, 1.0, 1.0, 1.0],
        sorting_layer: 0,
        order: 0,
    };
    // A 6×6 patch of tiles in chunk (0,0), index cycling 1..=4.
    let n = (TILE_CHUNK_DIM * TILE_CHUNK_DIM) as usize;
    let mut chunk = vec![0u32; n];
    for gy in 0..6i32 {
        for gx in 0..6i32 {
            let idx = (((gx + gy).rem_euclid(4)) + 1) as u32;
            chunk[(gy * TILE_CHUNK_DIM + gx) as usize] = idx;
        }
    }

    let mut scene = RenderScene {
        grid_enabled: true,
        pending_texture_uploads: vec![SpriteTextureUpload {
            handle: ATLAS,
            width: 64,
            height: 64,
            rgba8: atlas,
        }],
        tilemaps: vec![RenderTilemap {
            params,
            chunks: vec![RenderChunk {
                coord: (0, 0),
                tiles: chunk,
            }],
        }],
        ..Default::default()
    };

    // A loose magenta sprite over the tiles (higher sorting layer).
    scene.sprites.push(SpriteInstance {
        position: DVec3::new(0.4, 0.4, 0.0),
        size: Vec2::new(1.0, 1.0),
        color: [0.95, 0.15, 0.95, 1.0],
        sorting_layer: 2,
        ..Default::default()
    });

    // A short text run in the built-in 8×8 bitmap font (exercises the text path
    // under ortho).
    let text_params = TextParams {
        position: DVec3::new(-2.4, 1.8, 0.0),
        text: "2D",
        glyph_cols: BUILTIN_FONT_COLS,
        glyph_rows: BUILTIN_FONT_ROWS,
        first_codepoint: BUILTIN_FONT_FIRST_CP,
        glyph_size: Vec2::new(0.6, 0.6),
        tracking: 0.1,
        color: [1.0, 1.0, 1.0, 1.0],
        texture: BUILTIN_FONT_TEXTURE,
        sorting_layer: 1,
        order: 0,
        halign: HAlign::Left,
    };
    let glyphs = expand_text(&text_params);
    if !glyphs.is_empty() {
        scene.prebatched.push(PrebatchedRun {
            texture: BUILTIN_FONT_TEXTURE,
            sorting_layer: 1,
            order: 0,
            instances: glyphs,
        });
    }
    scene.mark_dirty();

    let img = check_golden(&gpu, "ortho_2d", &scene, &ortho_view());

    // Under the ortho camera: a red tile cell and the magenta sprite are both
    // visible (proving the parallel projection places 2D content correctly).
    let mut red = false;
    let mut magenta = false;
    for chunk in img.chunks(4) {
        let (r, g, b) = (chunk[0] as i32, chunk[1] as i32, chunk[2] as i32);
        if r > 140 && r - g > 80 && r - b > 80 {
            red = true;
        }
        if r > 150 && b > 150 && r - g > 60 && b - g > 60 {
            magenta = true;
        }
    }
    assert!(red, "expected a red tile under the ortho camera");
    assert!(magenta, "expected the magenta sprite over the tiles");
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// A procedural skinned cylinder (P11.1) + a 2-joint bend skeleton + a rotation
/// clip. The lower half is weighted to the root joint, the upper half to the
/// child joint (blended across the middle band), so rotating the child bends the
/// top of the cylinder. Returns the skeleton, the clip, and the bind-space mesh.
fn skinned_cylinder() -> (inf_anim::Skeleton, inf_anim::AnimClip, SkinnedMeshData) {
    use inf_anim::{
        AnimClip, Interpolation, Joint, JointTrack, JointTransform, QuatTrack, Skeleton,
    };

    // Skeleton: root at the origin, child 1 unit up (+Y). Inverse binds are the
    // inverse of each joint's global bind, so the rest pose is undeformed.
    let j0 = JointTransform::IDENTITY;
    let j1 = JointTransform::from_trs(Vec3::Y, Quat::IDENTITY, Vec3::ONE);
    let g0 = j0.to_mat4();
    let g1 = g0 * j1.to_mat4();
    let skeleton = Skeleton::new(vec![
        Joint {
            name: "root".into(),
            parent: None,
            inverse_bind: g0.inverse().to_cols_array(),
            local_bind: j0,
        },
        Joint {
            name: "upper".into(),
            parent: Some(0),
            inverse_bind: g1.inverse().to_cols_array(),
            local_bind: j1,
        },
    ])
    .unwrap();

    // Clip: rotate the child joint 0° → 60° about +Z over 1 second.
    let mut jt = JointTrack::new(1);
    jt.rotation = Some(QuatTrack::new(
        vec![0.0, 1.0],
        vec![
            Quat::IDENTITY.to_array(),
            Quat::from_rotation_z(60f32.to_radians()).to_array(),
        ],
        Interpolation::Linear,
    ));
    let clip = AnimClip::new("bend", vec![jt]);

    // A radial cylinder along +Y, height 2, radius 0.35.
    let (radial, rings, radius, height) = (16usize, 8usize, 0.35f32, 2.0f32);
    let mut vertices = Vec::new();
    for r in 0..=rings {
        let y = height * r as f32 / rings as f32;
        let w1 = smoothstep(0.5, 1.5, y);
        let w0 = 1.0 - w1;
        for s in 0..radial {
            let a = std::f32::consts::TAU * s as f32 / radial as f32;
            let (c, sn) = (a.cos(), a.sin());
            vertices.push(SkinnedVertex {
                pos: [radius * c, y, radius * sn],
                normal: [c, 0.0, sn],
                // P26.5: the stream carries a uv now. This fixture binds no
                // virtual texture, so nothing reads it — a cylindrical unwrap is
                // written anyway rather than a zero, because a golden fixture
                // that ships a degenerate stream is where a degenerate stream
                // becomes normal.
                uv: [s as f32 / radial as f32, y / height],
                joints: [0, 1, 0, 0],
                weights: [w0, w1, 0.0, 0.0],
            });
        }
    }
    let mut indices = Vec::new();
    for r in 0..rings {
        for s in 0..radial {
            let s1 = (s + 1) % radial;
            let a = (r * radial + s) as u32;
            let b = (r * radial + s1) as u32;
            let c = ((r + 1) * radial + s) as u32;
            let d = ((r + 1) * radial + s1) as u32;
            // Outward-facing winding (CCW seen from outside the tube).
            indices.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }
    (skeleton, clip, SkinnedMeshData { vertices, indices })
}

/// The skinning palette (`global · inverse_bind` per joint) for a clip at time `t`.
fn palette_at(
    sk: &inf_anim::Skeleton,
    clip: &inf_anim::AnimClip,
    t: f32,
) -> std::sync::Arc<Vec<Mat4>> {
    let pose = inf_anim::sample_clip(sk, clip, t, false);
    std::sync::Arc::new(inf_anim::skinning_matrices(sk, &pose))
}

/// Skinned-mesh golden (P11.1): a procedural skinned cylinder driven by a real
/// `inf-anim` clip, rendered at `t=0` (rest, straight) vs `t=mid` (bent). The
/// committed golden is the bent pose; the structural gate proves **deformation**
/// — the two poses render meaningfully differently — and that the skinned pixels
/// are lit (the GPU skinning path actually ran). Determinism gate via
/// `check_golden`; strict pixel diff opt-in. The unskinned pipeline is untouched,
/// so every other golden stays byte-stable.
#[test]
fn golden_skinned_mesh() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (sk, clip, mesh) = skinned_cylinder();

    let make = |palette: std::sync::Arc<Vec<Mat4>>| SkinnedInstance {
        vt: Default::default(),
        translation: DVec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
        color: [0.75, 0.55, 0.35, 1.0],
        metallic: 0.0,
        roughness: 0.6,
        emissive: [0.0; 3],
        id: 1,
        mesh: 0,
        palette,
        shadow: inf_render::SkinnedShadow::BindSphere,
    };

    // P18.3: the scene shares bind-space geometry as an `Arc`, so a host that
    // re-projects every frame neither copies nor re-uploads it. Both scenes here
    // deliberately hold the SAME `Arc`, which is also what makes this golden prove
    // that two frames sharing geometry still render their own poses.
    let mesh = std::sync::Arc::new(mesh);
    let mut rest = RenderScene {
        grid_enabled: true,
        skinned_meshes: vec![mesh.clone()],
        ..Default::default()
    };
    rest.skinned.push(make(palette_at(&sk, &clip, 0.0)));
    rest.mark_dirty();

    let mut bent = RenderScene {
        grid_enabled: true,
        skinned_meshes: vec![mesh],
        ..Default::default()
    };
    bent.skinned.push(make(palette_at(&sk, &clip, 0.5)));
    bent.mark_dirty();

    // Angled view framing the ~2 m tall cylinder around its middle.
    let view = look_view(DVec3::new(3.2, 1.6, 3.6), DVec3::new(0.0, 1.0, 0.0));
    let rest_img = render(&gpu, &rest, &view);
    let bent_img = check_golden(&gpu, "skinned_mesh", &bent, &view);

    // Deformation: the bent pose differs meaningfully from the rest pose.
    let (mean, max) = image_diff(&rest_img, &bent_img, W, H);
    assert!(
        mean > 0.002,
        "expected visible skinning deformation between t=0 and t=mid (mean {mean}, max {max})"
    );
    // The skinned cylinder is actually lit (the GPU skinning path ran).
    let lit = bent_img
        .chunks(4)
        .any(|p| p[0] as u16 + p[1] as u16 + p[2] as u16 > 150);
    assert!(lit, "expected a lit skinned pixel");
}

/// **A CROWD IS NOT A THOUSAND CLONES, AND IT IS ONE DRAW** (wave NPC1b).
///
/// Eight copies of one skinned mesh, sharing **one** joint palette by `Arc` — the
/// shape a crowd's far tier has, where the sim-LOD ladder evaluates no pose and
/// every agent resolves to the same rest matrices — each with its own tint and
/// its own build. Three claims, and the golden is only the third:
///
///  1. the batch planner puts all eight in **one draw** from **one** atlas block,
///     so the sharing survives all the way to the GPU;
///  2. the drawn bodies are **different heights**, so a per-instance scale rides
///     the instance stream rather than being flattened by the shared palette;
///  3. the drawn bodies are **different colours** — counted, from the image, so
///     "a crowd is not a thousand clones" is a measurement rather than a claim
///     about a `color` field nothing was proven to read.
///
/// The tints here are the renderer's business only: the *table* a crowd draws
/// them from lives in `inf_ecs::crowd::CROWD_LOOKS` and is armed there, because
/// this crate cannot see the sim and should not learn to.
#[test]
fn golden_crowd_variation() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (sk, clip, mesh) = skinned_cylinder();
    let mesh = std::sync::Arc::new(mesh);

    // ONE palette, shared: the far tier's shape.
    let shared = palette_at(&sk, &clip, 0.0);
    // Eight looks and eight builds, spread the way a real crowd's are.
    let looks: [([f32; 3], f32); 8] = [
        ([1.00, 0.98, 0.94], 1.06),
        ([0.52, 0.60, 0.82], 0.94),
        ([0.58, 0.62, 0.42], 1.02),
        ([0.86, 0.52, 0.36], 0.96),
        ([0.38, 0.40, 0.44], 1.08),
        ([0.92, 0.82, 0.62], 0.92),
        ([0.40, 0.70, 0.70], 1.04),
        ([0.66, 0.34, 0.40], 0.98),
    ];

    let mut scene = RenderScene {
        grid_enabled: false,
        skinned_meshes: vec![mesh],
        ..Default::default()
    };
    for (i, (tint, build)) in looks.iter().enumerate() {
        scene.skinned.push(SkinnedInstance {
            vt: Default::default(),
            translation: DVec3::new(i as f64 * 1.15 - 4.0, 0.0, 0.0),
            rotation: Quat::IDENTITY,
            scale: Vec3::splat(*build),
            color: [tint[0], tint[1], tint[2], 1.0],
            metallic: 0.0,
            roughness: 0.6,
            emissive: [0.0; 3],
            id: 1 + i as u32,
            mesh: 0,
            palette: shared.clone(),
            shadow: inf_render::SkinnedShadow::Proxy,
        });
    }
    scene.mark_dirty();

    // (1) ONE draw, ONE block — the planner the pass itself calls.
    let plan = inf_render::plan_skinned_batches(&scene);
    assert_eq!(
        plan.runs.len(),
        1,
        "eight clones drew in {} calls",
        plan.runs.len()
    );
    assert_eq!(
        (plan.blocks, plan.matrices),
        (1, shared.len()),
        "the shared far-tier palette did not deduplicate into one atlas block"
    );

    let view = look_view(DVec3::new(0.0, 1.6, 8.0), DVec3::new(0.0, 1.0, 0.0));
    let img = check_golden(&gpu, "crowd_variation", &scene, &view);

    // **The bodies are found against a CONTROL, not against brightness.** The sky
    // is the brightest thing in the frame, so "the topmost lit row" is row 0 in
    // every column and would read as eight identical heights however wrong the
    // scale was. The control is the same frame with the crowd taken out; a pixel
    // that differs from it is a body pixel.
    let mut empty = scene.clone();
    empty.skinned.clear();
    empty.mark_dirty();
    let bg = render(&gpu, &empty, &view);
    let (w, h) = (W as usize, H as usize);
    let body = |x: usize, y: usize| -> bool {
        let i = (y * w + x) * 4;
        let d = |k: usize| (img[i + k] as i16 - bg[i + k] as i16).abs();
        d(0) + d(1) + d(2) > 24
    };

    // The eight bodies, as the frame's own connected column runs, left to right —
    // rather than as eight equal slices of the frame, which assumes a framing.
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for x in 0..w {
        let hit = (0..h).any(|y| body(x, y));
        match runs.last_mut() {
            Some(r) if hit && r.1 == x => r.1 = x + 1,
            _ if hit => runs.push((x, x + 1)),
            _ => {}
        }
    }
    assert_eq!(
        runs.len(),
        looks.len(),
        "the frame holds {} separated bodies, not {}: they overlap or are off-screen",
        runs.len(),
        looks.len()
    );

    // (2) different HEIGHTS: the tallest body's silhouette starts higher up the
    // frame than the shortest's.
    let column_top = |slot: usize| -> usize {
        let (lo, hi) = runs[slot];
        (0..h)
            .find(|y| (lo..hi).any(|x| body(x, *y)))
            .expect("a run with no body pixel")
    };
    let (tall, short) = (column_top(4), column_top(5));
    assert!(
        tall < short,
        "the tallest body's top row is {tall} and the shortest's {short}: the build is not reaching the vertex stream"
    );

    // (3) different COLOURS: the brightest BODY pixel of each run, quantized.
    // Eight looks must not collapse to one.
    let mut hues: Vec<[u8; 3]> = Vec::new();
    for (slot, &(lo, hi)) in runs.iter().enumerate() {
        let mut best = [0u8; 3];
        let mut best_sum = 0u16;
        for y in 0..h {
            for x in lo..hi {
                if !body(x, y) {
                    continue;
                }
                let p = &img[(y * w + x) * 4..];
                let sum = p[0] as u16 + p[1] as u16 + p[2] as u16;
                if sum > best_sum {
                    best_sum = sum;
                    best = [p[0] / 24, p[1] / 24, p[2] / 24];
                }
            }
        }
        assert!(best_sum > 0, "body {slot} drew no body pixel at all");
        hues.push(best);
    }
    hues.sort_unstable();
    hues.dedup();
    assert!(
        hues.len() >= 6,
        "eight looks produced {} distinct sampled colours — a crowd of clones",
        hues.len()
    );
}

// ── P13.3a: HDR post pipeline (bloom, SSAO, TAA) ─────────────────────────────

/// HDR bloom golden (P13.3a): a dark scene with a few **strongly emissive** cubes
/// (linear emissive ≫ 1) so the bloom threshold prefilter + blur mip chain lights
/// up a soft glow the tonemap adds back. Structural gate: with bloom ON the frame
/// carries more total energy than with bloom OFF (the additive blurred glow),
/// while the emitters stay bright and coloured (no NaN blowout). Determinism via
/// `check_golden_with`; strict pixel diff opt-in.
#[test]
fn golden_hdr_bloom() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mut scene = RenderScene {
        grid_enabled: false,
        ..Default::default()
    };
    for (i, (x, emissive)) in [
        (-2.6f64, [8.0f32, 1.0, 0.4]),
        (0.0, [0.5, 7.0, 1.0]),
        (2.6, [0.6, 1.2, 9.0]),
    ]
    .into_iter()
    .enumerate()
    {
        scene.instances.push(MeshInstance {
            vt: Default::default(),
            translation: DVec3::new(x, 0.5, 0.0),
            rotation: Quat::from_rotation_y(0.3),
            scale: Vec3::splat(0.6),
            color: [0.02, 0.02, 0.02, 1.0],
            metallic: 0.0,
            roughness: 0.5,
            emissive,
            id: i as u32 + 1,
            mesh: PrimMesh::Cube,
            blend: 0,
            cutoff: 0.5,
        });
    }
    scene.mark_dirty();
    let view = look_view(DVec3::new(0.0, 1.5, 7.0), DVec3::new(0.0, 0.5, 0.0));

    let bloom_on = RenderSettings {
        bloom: BloomSettings {
            enabled: true,
            threshold: 1.0,
            knee: 0.6,
            intensity: 0.5,
            ..BloomSettings::default()
        },
        ..RenderSettings::default()
    };

    let img = check_golden_with(&gpu, "hdr_bloom", &scene, &view, bloom_on);
    let img_off = render_with(&gpu, &scene, &view, RenderSettings::default());

    let sum = |img: &[u8]| -> u64 {
        img.chunks(4)
            .map(|p| p[0] as u64 + p[1] as u64 + p[2] as u64)
            .sum()
    };
    let (sum_on, sum_off) = (sum(&img), sum(&img_off));
    assert!(
        sum_on > sum_off + (img.len() as u64 / 4),
        "bloom should add glow energy (on {sum_on} vs off {sum_off})"
    );
    let bright = img
        .chunks(4)
        .any(|p| p[0] > 180 || p[1] > 180 || p[2] > 180);
    assert!(bright, "expected the emissive cubes to stay bright");
}

// ── wave VIS1b: sun glare / lens flare ───────────────────────────────────────

/// A scene aimed at the real sun: the physical atmosphere on, its disc in the
/// upper part of the frame, dark ground under it, and optionally a slab standing
/// in front of the disc.
///
/// **The disc has to be the atmosphere's, not the gradient sky's.** The first
/// draft of this fixture used `RenderScene::default()`, whose sky is the
/// pre-P17.2 three-colour gradient plus a `pow(dot, 48) * 0.105` warm glow —
/// nothing in it reaches the flare's threshold of 1.0 in exposed units, and the
/// arm measured a byte-identical frame. `sky.wgsl` draws a disc only on the
/// atmosphere path, where `SUN_DISC_GAIN = 4.0` exists *precisely so it clips to
/// white through ACES and drives bloom* — which is the same property a lens
/// flare needs, because a flare is what a lens does with light that has blown
/// past the display's white.
///
/// `occluder` is `Some(radiance)` to stand a slab in the line of sight to the
/// disc, emissive at that radiance. **`Some(0.0)` and `Some(30.0)` are two
/// different questions** (VIS1b audit): a dark slab hides the disc *and* the
/// bright pixels a gather would have found there, so it dims the glare whether
/// or not anything tests occlusion; a bright one hides the disc and leaves the
/// bright pixels, so only an occlusion test can put the glare out.
fn sun_flare_scene(occluder: Option<f32>) -> (RenderScene, RenderView) {
    // 04:30 UTC on the solstice at 48.9°N — the dawn goldens' own hour. A LOW
    // sun, deliberately: the sky around a noon disc is already near the top of
    // the ACES curve, so glare added there is invisible on an 8-bit frame, and
    // the first draft of this fixture (08:20, a high sun) measured the whole
    // effect at +0.08% of frame luminance. At dawn the disc is the only bright
    // thing in a dim sky, which is both the shot a lens flare is taken in and the
    // one an assertion can read.
    let (mut scene, bodies) = tod_scene(16_200.0);
    scene.grid_enabled = false;
    let eye = DVec3::new(0.0, 2.0, 0.0);
    scene.instances.push(MeshInstance::lit(
        eye + DVec3::new(0.0, -3.0, 0.0),
        Quat::IDENTITY,
        Vec3::new(60.0, 0.5, 60.0),
        [0.05, 0.05, 0.06, 1.0],
        1,
    ));
    if let Some(radiance) = occluder {
        // Straight along the line of sight to the sun, twenty metres out. It is
        // the *disc* this hides, not the frame: the ghost chain and the halo have
        // other bright pixels to work from, and that difference is what the
        // occlusion arm reads.
        let mut slab = MeshInstance::lit(
            eye + bodies.sun * 20.0,
            Quat::IDENTITY,
            Vec3::splat(6.0),
            [0.02, 0.02, 0.02, 1.0],
            2,
        );
        slab.emissive = [radiance; 3];
        scene.instances.push(slab);
    }
    scene.mark_dirty();
    // Pitched eight degrees below the sun's own elevation, so the disc sits high
    // in frame with ground under it — the shot a veiling glare, a ghost chain
    // running back through the centre and a halo are all visible in at once.
    // Derived from the body rather than fixed, for `horizon_view`'s own reason:
    // a refined solar model must not turn this into a picture of empty sky.
    let elevation = bodies.sun.y.clamp(-1.0, 1.0).asin().to_degrees();
    let view = horizon_view(bodies.sun, elevation - 10.0);
    (scene, view)
}

fn flare_on() -> RenderSettings {
    RenderSettings {
        flare: FlareSettings {
            enabled: true,
            intensity: 3.0,
            ghost_count: 5,
            halo: 0.6,
            streak: 0.45,
        },
        ..RenderSettings::default()
    }
}

/// **Sun glare golden** (wave VIS1b, clause 3) — the 56th.
///
/// The **additive** branch of the golden rule: `GOLDENS` moves in four gates, all
/// three `GOLDEN_SET_DIGEST` pins move, the phase18 name array grows, and not one
/// committed image changes — because the flare is off by default and off is a
/// clear of its own target plus a branch the tonemap does not take.
///
/// The structural arm beside it is for the CI legs that do not compare pixels
/// strictly, and it measures the effect against **the same frame with the feature
/// off** rather than against an absolute: a glare is light added around a source,
/// so the frame gets brighter and the pixels that brighten are near the sun.
#[test]
fn golden_sun_flare() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (scene, view) = sun_flare_scene(None);

    let img = check_golden_with(&gpu, "sun_flare", &scene, &view, flare_on());
    let off = render_with(&gpu, &scene, &view, RenderSettings::default());

    let luma = |img: &[u8]| -> f64 {
        img.chunks(4)
            .map(|p| 0.2126 * p[0] as f64 + 0.7152 * p[1] as f64 + 0.0722 * p[2] as f64)
            .sum::<f64>()
    };
    let (on_l, off_l) = (luma(&img), luma(&off));
    assert!(
        on_l > off_l * 1.02,
        "the flare added no light: on {on_l:.0} vs off {off_l:.0}"
    );
    // And it added it as a *lens* does — brightest near the sun. The sun sits in
    // the upper half by construction (`sun_flare_view`), so the upper half must
    // gain proportionally more than the lower one.
    let half = |img: &[u8], top: bool| -> f64 {
        let rows = if top { 0..H / 2 } else { H / 2..H };
        rows.flat_map(|y| (0..W).map(move |x| (x, y)))
            .map(|(x, y)| {
                let p = px(img, x, y);
                0.2126 * p[0] as f64 + 0.7152 * p[1] as f64 + 0.0722 * p[2] as f64
            })
            .sum()
    };
    let top_gain = half(&img, true) - half(&off, true);
    let bot_gain = half(&img, false) - half(&off, false);
    eprintln!(
        "sun_flare: luma {off_l:.0} -> {on_l:.0} (+{:.2}%); top gain {top_gain:.0}, \
         bottom gain {bot_gain:.0}",
        (on_l / off_l - 1.0) * 100.0
    );
    assert!(
        top_gain > bot_gain,
        "the glare is not centred on the sun: top {top_gain:.0} vs bottom {bot_gain:.0}"
    );
}

/// **The glare goes out when the sun does** — the occlusion test, measured.
///
/// This is the claim that separates a lens flare from a screen-space smear: a
/// veiling glare is light scattered inside the lens *by the source*, so putting a
/// wall in front of the sun must take it away. The arm reads the difference
/// between flare-on and flare-off in each of two scenes, so the wall's own effect
/// on the frame cancels out and what is left is the glare alone.
///
/// # The dark wall does not pin the occlusion test, and that is the audit's finding
///
/// A **dark** slab in front of the disc removes the sun's screen position *and*
/// every bright pixel a radial gather would have found there, so the glare falls
/// away whether or not anything asks about occlusion. Measured: with
/// `flare_sun_visibility` severed to a constant `1.0`, the dark-wall reading goes
/// only 18 901 → 51 197 against a threshold of 113 073 — the arm stayed green with
/// the feature's headline claim deleted, and 84 % of the "92 % cut" the wave
/// reported is the slab hiding the disc rather than the test.
///
/// So the second half stands a **bright** slab there instead, with the ghost
/// chain, the halo and the streak turned off so only the veil is in the
/// measurement. Now the frame is *brighter* where the sun was and the only thing
/// that can put the glare out is the depth test. Severed, that reading goes from
/// near zero to larger than the clear sky's.
#[test]
fn the_sun_glare_is_extinguished_when_the_sun_is_occluded() {
    let Some(gpu) = gpu_or_skip() else { return };
    let luma = |img: &[u8]| -> f64 {
        img.chunks(4)
            .map(|p| 0.2126 * p[0] as f64 + 0.7152 * p[1] as f64 + 0.0722 * p[2] as f64)
            .sum::<f64>()
    };
    let added = |occluder: Option<f32>, settings: RenderSettings| -> f64 {
        let (scene, view) = sun_flare_scene(occluder);
        let on = render_with(&gpu, &scene, &view, settings);
        let off = render_with(&gpu, &scene, &view, RenderSettings::default());
        luma(&on) - luma(&off)
    };
    let clear = added(None, flare_on());
    let hidden = added(Some(0.0), flare_on());
    eprintln!("sun glare: clear sky adds {clear:.0}, sun behind a wall adds {hidden:.0}");
    assert!(clear > 0.0, "the clear-sky fixture produced no glare");
    assert!(
        hidden < clear * 0.5,
        "occluding the sun barely dimmed the glare: {hidden:.0} against {clear:.0}"
    );

    // ── the half that pins the depth test itself ──
    //
    // The veil alone: no ghosts, no halo, no streak. Those three are images of
    // whatever is bright and are deliberately NOT gated on the sun's visibility,
    // so leaving them in would put the bright slab's own ghosts in the number the
    // assertion reads.
    let veil_only = RenderSettings {
        flare: FlareSettings {
            enabled: true,
            intensity: 3.0,
            ghost_count: 0,
            halo: 0.0,
            streak: 0.0,
        },
        ..RenderSettings::default()
    };
    let veil_clear = added(None, veil_only);
    let veil_behind_a_lamp = added(Some(30.0), veil_only);
    eprintln!(
        "veil only: clear sky {veil_clear:.0}, sun behind a BRIGHT slab \
         {veil_behind_a_lamp:.0}"
    );
    assert!(
        veil_clear > 0.0,
        "the veil-only fixture produced no glare at all"
    );
    assert!(
        veil_behind_a_lamp < veil_clear * 0.1,
        "the veil did not read the sun's OCCLUSION — a slab brighter than the sky \
         hid the disc and the glare stayed on: {veil_behind_a_lamp:.0} against a \
         clear sky's {veil_clear:.0}"
    );
}

/// **The flare is off by default, and off costs the frame nothing.**
///
/// `INF_GOLDEN_STRICT=1` over all committed frames is the pixel half of this; the
/// engagement half is here, and it is the stronger claim: the two frames must
/// **differ** when the feature is on, so the golden above cannot be passing
/// because the pass never ran.
#[test]
fn flare_off_is_byte_identical_and_on_is_not() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (scene, view) = sun_flare_scene(None);
    let a = render_with(&gpu, &scene, &view, RenderSettings::default());
    let b = render_with(
        &gpu,
        &scene,
        &view,
        RenderSettings {
            flare: FlareSettings::default(),
            ..RenderSettings::default()
        },
    );
    assert_eq!(a, b, "the default flare block is not the default");
    let on = render_with(&gpu, &scene, &view, flare_on());
    assert_ne!(a, on, "turning the flare on changed nothing");
}

// ── wave VIS1b: the lens trio (vignette / chromatic aberration / film grain) ──

/// A high-contrast scene: a pale floor with saturated blocks on it, lit hard.
///
/// Contrast is what the trio needs to be *measurable* — a vignette needs light in
/// the corners to take away, chromatic aberration needs edges to fringe, and
/// grain needs mid-tones to sit on.
fn lens_trio_scene(clock: f64) -> RenderScene {
    let mut scene = RenderScene {
        grid_enabled: false,
        ..Default::default()
    };
    scene.atmosphere.clouds.time_s = clock;
    scene.instances.push(MeshInstance::lit(
        DVec3::new(0.0, -0.3, 0.0),
        Quat::IDENTITY,
        Vec3::new(14.0, 0.4, 14.0),
        [0.78, 0.78, 0.80, 1.0],
        1,
    ));
    let colors = [
        [0.9, 0.12, 0.10, 1.0],
        [0.10, 0.85, 0.20, 1.0],
        [0.10, 0.20, 0.95, 1.0],
        [0.95, 0.85, 0.10, 1.0],
        [0.05, 0.05, 0.06, 1.0],
    ];
    for (i, c) in colors.into_iter().enumerate() {
        let a = i as f64 * 1.2566;
        scene.instances.push(MeshInstance::lit(
            DVec3::new(a.cos() * 3.0, 0.6, a.sin() * 3.0),
            Quat::from_rotation_y(0.4 * i as f32),
            Vec3::splat(0.85),
            c,
            i as u32 + 2,
        ));
    }
    scene.lights.push(RenderLight {
        kind: LightKind::Directional,
        direction: Vec3::new(0.4, 0.8, 0.45).normalize(),
        color: [1.0, 0.97, 0.92],
        intensity: 3.2,
        ..Default::default()
    });
    scene.mark_dirty();
    scene
}

fn lens_trio_on() -> RenderSettings {
    RenderSettings {
        film: FilmSettings {
            vignette_intensity: 0.65,
            vignette_smoothness: 0.45,
            chromatic_aberration: 7.0,
            grain_intensity: 0.22,
            grain_size: 2.0,
        },
        ..RenderSettings::default()
    }
}

/// Mean channel value over a screen rectangle, 0..255.
fn mean_channel(img: &[u8], c: usize, x0: u32, y0: u32, x1: u32, y1: u32) -> f64 {
    let mut acc = 0.0f64;
    let mut n = 0.0f64;
    for y in y0..y1 {
        for x in x0..x1 {
            acc += px(img, x, y)[c] as f64;
            n += 1.0;
        }
    }
    acc / n.max(1.0)
}

/// Mean `|R − B|` over a screen rectangle — what lateral chromatic aberration
/// puts at an edge and what nothing else in this frame does.
fn mean_rb_split(img: &[u8], x0: u32, y0: u32, x1: u32, y1: u32) -> f64 {
    let mut acc = 0.0f64;
    let mut n = 0.0f64;
    for y in y0..y1 {
        for x in x0..x1 {
            let p = px(img, x, y);
            acc += (p[0] as f64 - p[2] as f64).abs();
            n += 1.0;
        }
    }
    acc / n.max(1.0)
}

/// **The lens trio golden** (wave VIS1b, clause 5) — the 57th.
///
/// One composite-stage uber-post: vignette before the ACES curve (a lens loses
/// light at the corner, it does not paint the corner black), chromatic aberration
/// as a radial three-tap, film grain after the curve in display space. All three
/// zero at the default, so this is the **additive** branch of the golden rule
/// again and no committed image moved.
#[test]
fn golden_lens_trio() {
    let Some(gpu) = gpu_or_skip() else { return };
    let scene = lens_trio_scene(0.0);
    let view = look_view(DVec3::new(0.0, 3.4, 7.5), DVec3::new(0.0, 0.4, 0.0));

    let img = check_golden_with(&gpu, "lens_trio", &scene, &view, lens_trio_on());
    let off = render_with(&gpu, &scene, &view, RenderSettings::default());

    // The vignette: the corners lose light relative to the centre.
    let corner = |img: &[u8]| {
        (mean_channel(img, 1, 0, 0, W / 8, H / 8)
            + mean_channel(img, 1, W - W / 8, H - H / 8, W, H))
            / 2.0
    };
    let centre = |img: &[u8]| mean_channel(img, 1, W * 3 / 8, H * 3 / 8, W * 5 / 8, H * 5 / 8);
    let ratio_on = corner(&img) / centre(&img).max(1e-6);
    let ratio_off = corner(&off) / centre(&off).max(1e-6);
    eprintln!(
        "lens_trio: corner/centre {ratio_off:.4} -> {ratio_on:.4}; \
         |R-B| edge band {:.3} -> {:.3}, centre {:.3} -> {:.3}",
        mean_rb_split(&off, 0, 0, W / 5, H),
        mean_rb_split(&img, 0, 0, W / 5, H),
        mean_rb_split(&off, W * 2 / 5, H * 2 / 5, W * 3 / 5, H * 3 / 5),
        mean_rb_split(&img, W * 2 / 5, H * 2 / 5, W * 3 / 5, H * 3 / 5),
    );
    assert!(
        ratio_on < ratio_off * 0.85,
        "the vignette did not darken the corners: {ratio_on:.4} against {ratio_off:.4}"
    );

    // The aberration, measured ALONE.
    //
    // It cannot be read off the trio frame, and that is a finding rather than a
    // fixture detail: `|R − B|` is an absolute difference, the vignette takes 47 %
    // of the light out of the very corners the fringe is strongest in, and the
    // first draft of this arm therefore measured the split going *down* at the
    // edge (19.501 → 18.606) with the aberration plainly working. Three effects
    // in one frame need three measurements.
    let ca_only = RenderSettings {
        film: FilmSettings {
            chromatic_aberration: 7.0,
            ..FilmSettings::default()
        },
        ..RenderSettings::default()
    };
    let ca = render_with(&gpu, &scene, &view, ca_only);
    let edge_gain = mean_rb_split(&ca, 0, 0, W / 5, H) - mean_rb_split(&off, 0, 0, W / 5, H);
    let centre_gain = mean_rb_split(&ca, W * 2 / 5, H * 2 / 5, W * 3 / 5, H * 3 / 5)
        - mean_rb_split(&off, W * 2 / 5, H * 2 / 5, W * 3 / 5, H * 3 / 5);
    eprintln!("chromatic aberration alone: edge +{edge_gain:.3}, centre +{centre_gain:.3}");
    assert!(
        edge_gain > 0.0 && edge_gain > centre_gain * 2.0,
        "the fringe is not radial: edge {edge_gain:.3} vs centre {centre_gain:.3}"
    );
}

/// **The grain is a function of the level clock, and of nothing else.**
///
/// The SKY2 jitter precedent, applied to a lens artefact: a frame index would
/// make the same document render differently on a machine that dropped a frame,
/// and would make a one-frame golden a lottery. Three claims:
///
/// * two renders at the same clock reading are **byte-identical**;
/// * two renders at clock readings a grain period apart are **not**;
/// * a clock advanced by less than a grain period does not re-roll — which is
///   what makes it a 24 Hz grain rather than noise.
#[test]
fn film_grain_follows_the_level_clock_and_never_the_frame_index() {
    let Some(gpu) = gpu_or_skip() else { return };
    let view = look_view(DVec3::new(0.0, 3.4, 7.5), DVec3::new(0.0, 0.4, 0.0));
    let grain_only = RenderSettings {
        film: FilmSettings {
            grain_intensity: 0.5,
            grain_size: 2.0,
            ..FilmSettings::default()
        },
        ..RenderSettings::default()
    };
    let at = |t: f64| render_with(&gpu, &lens_trio_scene(t), &view, grain_only);

    assert_eq!(
        at(10.0),
        at(10.0),
        "the grain is not a function of the clock"
    );
    assert_ne!(
        at(10.0),
        at(10.0 + 1.0 / 24.0 + 1e-3),
        "the grain never re-rolled across a grain period"
    );
    assert_eq!(
        at(10.0),
        at(10.0 + 1.0 / 240.0),
        "the grain re-rolled inside one 24 Hz period"
    );
    // And zero strength is not merely invisible, it is the untouched frame.
    let plain = render_with(
        &gpu,
        &lens_trio_scene(10.0),
        &view,
        RenderSettings::default(),
    );
    let zero = render_with(
        &gpu,
        &lens_trio_scene(10.0),
        &view,
        RenderSettings {
            film: FilmSettings::default(),
            ..RenderSettings::default()
        },
    );
    assert_eq!(plain, zero, "the default film block is not the default");
    assert_ne!(plain, at(10.0), "turning the grain on changed nothing");
}

/// SSAO golden (P13.3a): a cluster of boxes forming **crevices** (a floor slab
/// with blocks pressed together and one stacked), lit by a single soft
/// directional key, SSAO ON. Structural gate: SSAO **darkens** the frame overall
/// (ambient occluded in the contact creases) while the scene stays lit — proving
/// the depth-prepass → half-res AO → ambient-multiply path ran. Determinism via
/// `check_golden_with`; strict pixel diff opt-in.
#[test]
fn golden_ssao() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mut scene = RenderScene {
        grid_enabled: false,
        ..Default::default()
    };
    scene.instances.push(MeshInstance::lit(
        DVec3::new(0.0, -0.25, 0.0),
        Quat::IDENTITY,
        Vec3::new(8.0, 0.5, 8.0),
        [0.6, 0.6, 0.62, 1.0],
        1,
    ));
    for (p, id) in [
        (DVec3::new(-0.55, 0.5, 0.0), 2u32),
        (DVec3::new(0.55, 0.5, 0.0), 3),
        (DVec3::new(0.0, 0.5, -0.9), 4),
        (DVec3::new(0.0, 1.5, 0.0), 5),
    ] {
        scene.instances.push(MeshInstance::lit(
            p,
            Quat::IDENTITY,
            Vec3::ONE,
            [0.7, 0.65, 0.6, 1.0],
            id,
        ));
    }
    scene.lights.push(RenderLight {
        kind: LightKind::Directional,
        color: [1.0, 0.98, 0.95],
        intensity: 1.2,
        direction: Vec3::new(0.3, 0.9, 0.3).normalize(),
        position: DVec3::ZERO,
        range: 0.0,
        ..RenderLight::default()
    });
    scene.mark_dirty();
    let view = look_view(DVec3::new(3.0, 2.6, 4.2), DVec3::new(0.0, 0.6, 0.0));

    let ssao_on = RenderSettings {
        ssao: SsaoSettings {
            enabled: true,
            radius: 0.7,
            intensity: 1.0,
            bias: 0.03,
        },
        ..RenderSettings::default()
    };

    let img = check_golden_with(&gpu, "ssao", &scene, &view, ssao_on);
    let img_off = render_with(&gpu, &scene, &view, RenderSettings::default());

    let sum = |img: &[u8]| -> u64 {
        img.chunks(4)
            .map(|p| p[0] as u64 + p[1] as u64 + p[2] as u64)
            .sum()
    };
    let (sum_on, sum_off) = (sum(&img), sum(&img_off));
    assert!(
        sum_on < sum_off,
        "SSAO should darken the ambient term (on {sum_on} vs off {sum_off})"
    );
    let lit = img
        .chunks(4)
        .any(|p| p[0] as u16 + p[1] as u16 + p[2] as u16 > 200);
    assert!(lit, "expected the SSAO scene to stay lit");
}

/// The SSAO settings the prepass-coverage arms below turn on. A generous radius,
/// because these scenes are metres across rather than the SSAO golden's
/// centimetres.
fn coverage_ssao() -> RenderSettings {
    RenderSettings {
        ssao: SsaoSettings {
            enabled: true,
            radius: 1.5,
            intensity: 1.0,
            bias: 0.03,
        },
        ..RenderSettings::default()
    }
}

/// **A ROUGH METAL IS NOT A DARK METAL** (wave VIS1a) — the furnace test's
/// consequence, on the GPU, in a real frame.
///
/// `gi::tests::the_ggx_furnace_test_is_white` pins the arithmetic: a
/// single-scatter GGX loses up to 55 % of the lobe's energy, and the compensation
/// returns exactly that much. This is the same claim measured where it is visible
/// — the *direct* specular response of a metal slab under a sun, at two
/// roughnesses, from the reflecting angle.
///
/// A perfectly reflective surface returns what it is given whatever its
/// roughness: the lobe spreads, it does not dim. So the two must land close
/// together. Before the compensation they did not, and the numbers are in the
/// assertion below.
#[test]
fn a_rough_metal_keeps_its_energy() {
    let Some(gpu) = gpu_or_skip() else { return };
    let slab = |roughness: f32| {
        let mut scene = RenderScene {
            grid_enabled: false,
            ..Default::default()
        };
        scene.instances.push(MeshInstance {
            metallic: 1.0,
            roughness,
            ..MeshInstance::lit(
                DVec3::new(0.0, -0.25, 0.0),
                Quat::IDENTITY,
                Vec3::new(24.0, 0.5, 24.0),
                [0.95, 0.93, 0.88, 1.0],
                1,
            )
        });
        scene.lights.push(RenderLight {
            kind: LightKind::Directional,
            color: [1.0, 1.0, 1.0],
            intensity: 3.0,
            direction: Vec3::new(0.0, 0.6, 1.0).normalize(),
            ..RenderLight::default()
        });
        scene.mark_dirty();
        scene
    };
    // Looking along the mirror direction of the sun, so the lobe is on screen at
    // both roughnesses rather than only at the smooth one.
    let view = look_view(DVec3::new(0.0, 2.4, -4.0), DVec3::new(0.0, 0.0, 4.0));
    // The metal only, not the sky above it: the sky is identical in both renders
    // and averaging it in would dilute the very quantity being measured.
    let lum = |img: &[u8]| -> f64 {
        let y0 = H / 2;
        let mut sum = 0.0f64;
        for y in y0..H {
            for x in 0..W {
                let i = ((y * W + x) * 4) as usize;
                sum += f64::from(img[i]) + f64::from(img[i + 1]) + f64::from(img[i + 2]);
            }
        }
        sum / f64::from((H - y0) * W)
    };
    let smooth = lum(&render_with(
        &gpu,
        &slab(0.15),
        &view,
        RenderSettings::default(),
    ));
    let rough = lum(&render_with(
        &gpu,
        &slab(0.85),
        &view,
        RenderSettings::default(),
    ));
    eprintln!(
        "ggx energy: a metal slab reads {smooth:.2} at roughness 0.15 and \
         {rough:.2} at 0.85 — a ratio of {:.3}",
        rough / smooth
    );
    // **The smooth slab is the internal control**, and that is what makes the
    // ratio mean something: at roughness 0.15 the compensation factor is
    // essentially 1 (measured on this fixture: the smooth slab moves 314.63 →
    // 317.57, **+0.9 %**), so the ratio is a direct read of how much energy the
    // ROUGH slab got back — 572.69 → 648.20, **+13.2 %**.
    //
    // The ratio is above 1 rather than below it because a smooth metal
    // concentrates its lobe into a narrow highlight while a rough one spreads it
    // across the whole slab; the framing is the mirror direction either way. What
    // this arm asserts is the *gain*, not a fixed physical constant.
    //
    // The threshold is mutation-measured rather than chosen: with the
    // compensation the ratio is **2.041**, with the factor forced to 1 — the
    // single-scatter lobe this wave replaced — it is **1.820**.
    assert!(
        rough / smooth > 1.93,
        "a metal at roughness 0.85 returned {:.3}x what the same metal returns at \
         0.15 — the specular lobe is losing energy the multi-scatter compensation \
         exists to give back (the single-scatter lobe reads 1.820 on this fixture)",
        rough / smooth
    );
}

/// **THE BLUR DOES NOT LEAK ACROSS A SILHOUETTE** (wave VIS1a) — the arm the 4×4
/// box blur could not have passed, and the reason the bilateral one replaced it.
///
/// A box blur over an AO buffer averages a pixel with its neighbours *whatever
/// surface they belong to*, so occlusion computed for a near object bleeds onto
/// whatever is visible past its silhouette — the classic halo. At half resolution
/// one leaked texel is four on screen.
///
/// The scene makes the claim checkable: a creased cluster of boxes on a floor
/// (strong contact AO, exactly the P13.3a golden's arrangement) with a **wall 25
/// metres behind them**. The AO radius is 1.5 m, so no wall pixel can be occluded
/// by anything: physically the wall must not change at all when AO is switched on.
/// Whatever darkening the wall *does* pick up is the blur leaking, and the arm
/// measures it right at the silhouette, where a box blur puts all of it.
#[test]
fn the_ao_blur_does_not_leak_across_a_silhouette() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mut scene = RenderScene {
        grid_enabled: false,
        ..Default::default()
    };
    // Two tall slabs with a 0.4 m slot between them. The slot's inner faces are
    // the most heavily occluded surfaces the AO integral can produce, and they
    // are directly adjacent — in screen space, within one blur footprint — to the
    // backdrop seen THROUGH the slot. That adjacency is the whole fixture: a box
    // blur averages the two together, a depth-aware one refuses to.
    for (x, id) in [(-1.2f64, 2u32), (1.2, 3)] {
        scene.instances.push(MeshInstance::lit(
            DVec3::new(x, 2.0, 0.0),
            Quat::IDENTITY,
            Vec3::new(2.0, 4.0, 2.0),
            [0.72, 0.68, 0.62, 1.0],
            id,
        ));
    }
    // The backdrop, 25 m behind the slabs and a colour nothing else carries so it
    // can be identified in the image without a depth readback.
    scene.instances.push(MeshInstance::lit(
        DVec3::new(0.0, 6.0, -25.0),
        Quat::IDENTITY,
        Vec3::new(60.0, 20.0, 0.5),
        [0.10, 0.30, 0.85, 1.0],
        9,
    ));
    scene.lights.push(RenderLight {
        kind: LightKind::Directional,
        color: [1.0, 0.98, 0.95],
        intensity: 1.2,
        direction: Vec3::new(0.3, 0.9, 0.3).normalize(),
        ..RenderLight::default()
    });
    scene.mark_dirty();
    let view = look_view(DVec3::new(0.0, 2.0, 7.0), DVec3::new(0.0, 2.0, -25.0));

    let off = render_with(&gpu, &scene, &view, RenderSettings::default());
    let on = render_with(&gpu, &scene, &view, coverage_ssao());

    // Blue-dominant pixels are the backdrop; everything else is the cluster, the
    // floor or the sky.
    let is_wall = |img: &[u8], i: usize| -> bool {
        let (r, g, b) = (img[i] as i32, img[i + 1] as i32, img[i + 2] as i32);
        b > 60 && b - r > 30 && b - g > 20
    };
    let wall: Vec<bool> = (0..(W * H) as usize)
        .map(|k| is_wall(&off, k * 4))
        .collect();
    // A wall pixel is "at the silhouette" when a non-wall pixel sits within three
    // texels of it — the footprint a 4×4 half-res blur reaches across.
    let near_edge = |k: usize| -> bool {
        let (x, y) = ((k as u32 % W) as i32, (k as u32 / W) as i32);
        for dy in -3i32..=3 {
            for dx in -3i32..=3 {
                let (nx, ny) = (x + dx, y + dy);
                if nx < 0 || ny < 0 || nx >= W as i32 || ny >= H as i32 {
                    continue;
                }
                if !wall[(ny as u32 * W + nx as u32) as usize] {
                    return true;
                }
            }
        }
        false
    };

    let (mut edge_sum, mut edge_n) = (0i64, 0usize);
    let (mut open_sum, mut open_n) = (0i64, 0usize);
    for (k, _) in wall.iter().enumerate().filter(|(_, w)| **w) {
        let i = k * 4;
        let d = i64::from(off[i]) + i64::from(off[i + 1]) + i64::from(off[i + 2])
            - i64::from(on[i])
            - i64::from(on[i + 1])
            - i64::from(on[i + 2]);
        if near_edge(k) {
            edge_sum += d;
            edge_n += 1;
        } else {
            open_sum += d;
            open_n += 1;
        }
    }
    assert!(
        edge_n > 200 && open_n > 200,
        "the fixture did not produce both a silhouette band ({edge_n} px) and an \
         open backdrop ({open_n} px) — the measurement below would be vacuous"
    );
    let edge = edge_sum as f64 / edge_n as f64;
    let open = open_sum as f64 / open_n as f64;
    eprintln!(
        "ao halo: the backdrop darkens by {edge:.3} luminance at the silhouette \
         ({edge_n} px) against {open:.3} in the open ({open_n} px)"
    );
    // **6.0, and the number is mutation-measured rather than chosen.** On this
    // fixture the depth-aware blur reads **5.14** and the same blur with its
    // weight forced to 1 — i.e. the 4×4 box this wave replaced — reads **6.59**.
    // So the threshold falsifies exactly what it names. (Two decimals, not three:
    // consecutive runs on the same adapter read 5.144 and 5.149, so the third one
    // is the GPU's, not the blur's.)
    //
    // **And the residue is a finding, not slack.** The honest answer here is
    // zero: the backdrop is 25 m from the nearest occluder against an AO radius
    // of 1.5 m, and the open backdrop confirms the integral itself contributes
    // 0.719. The 5.1 that remains at the silhouette is not the blur — it is the
    // **half-resolution upsample**. An AO texel covers 2×2 full-res pixels; where
    // one straddles the slot's edge the texel's own depth sample is the slab's, so
    // it is computed dark, and the lit passes' bilinear fetch then spreads it one
    // full-res pixel onto the wall. Removing that needs a depth-aware upsample in
    // the lit passes or a full-res AO buffer — routed as **VIS-C4b**, and the
    // reason this arm asserts an upper bound rather than a zero.
    assert!(
        edge < 6.0,
        "the backdrop darkened by {edge:.3} at the silhouette against {open:.3} \
         in the open — the AO blur is leaking occlusion across the edge, which is \
         a halo 25 metres away from the only occluder in the frame (a 4×4 box \
         blur reads 6.591 on this fixture; the depth-aware one reads 5.144)"
    );
}

/// A rigid receiver: an 8 m floor slab with its top face at `y = 0`, and — where
/// the occluder is not itself the ground — the only surface in these scenes that
/// samples AO at all.
fn ao_floor() -> MeshInstance {
    MeshInstance::lit(
        DVec3::new(0.0, -0.25, 0.0),
        Quat::IDENTITY,
        Vec3::new(8.0, 0.5, 8.0),
        [0.6, 0.6, 0.62, 1.0],
        1,
    )
}

/// **THE PREPASS SEES THE WORLD** (wave VIS1a) — the falsifiable half of widening
/// the depth prepass past rigid meshes.
///
/// Before this wave `DepthPrepassNode` drew rigid meshes and nothing else, so a
/// terrain, a character, a voxel volume or a fracture chunk wrote *nothing* into
/// the prepass: `ssao.wgsl` read `DEPTH_CLEAR` wherever they stood, took its
/// "sky ⇒ fully unoccluded" branch, and left the AO texture at a uniform 1.0 —
/// which multiplies the ambient term by one.
///
/// The measurement is **the occlusion this geometry casts onto pixels it does not
/// itself cover**, which isolates the claim exactly. Four frames are rendered:
/// the scene with and without the occluder, each with SSAO off and on. The
/// SSAO-off pair identifies the pixels the occluder does not draw over — where
/// they agree, the *same* surface is visible in both scenes. On exactly those
/// pixels the SSAO-on pair may then differ for one reason only: the occluder's
/// depth reached the prepass and the AO kernel found it.
///
/// A removed `depth_prepass` implementation collapses that count to zero. Nothing
/// else in the renderer can move it.
fn assert_prepass_reaches(
    gpu: &GpuContext,
    what: &str,
    with: &RenderScene,
    without: &RenderScene,
    view: &RenderView,
) {
    let off_w = render_with(gpu, with, view, RenderSettings::default());
    let off_o = render_with(gpu, without, view, RenderSettings::default());
    let on_w = render_with(gpu, with, view, coverage_ssao());
    let on_o = render_with(gpu, without, view, coverage_ssao());

    let (mut shared, mut darker, mut brighter) = (0usize, 0usize, 0usize);
    let mut net: i64 = 0;
    for i in (0..off_w.len()).step_by(4) {
        if off_w[i..i + 3] != off_o[i..i + 3] {
            continue; // the occluder is drawn here — not a shared surface
        }
        shared += 1;
        let a = i32::from(on_w[i]) + i32::from(on_w[i + 1]) + i32::from(on_w[i + 2]);
        let b = i32::from(on_o[i]) + i32::from(on_o[i + 1]) + i32::from(on_o[i + 2]);
        net += i64::from(a - b);
        if a < b - 1 {
            darker += 1;
        } else if a > b + 1 {
            brighter += 1;
        }
    }
    eprintln!(
        "prepass coverage — {what}: of {shared} pixels showing the SAME surface \
         in both scenes, {darker} darkened and {brighter} brightened once the \
         occluder joined the prepass; net luminance {net:+}"
    );
    // 100 rather than something near the measured counts (1 583 skinned, 7 033
    // fracture, and the voxel probe's smaller share): the claim is "not zero",
    // and a threshold set against one adapter's exact number is a flake waiting
    // for a different rasterizer.
    assert!(
        darker > 100,
        "{what}: adding this geometry darkened only {darker} of {shared} shared \
         pixels — its depth is not in the prepass, so the AO texture is 1.0 where \
         it stands and it occludes nothing"
    );
    // **The claim is the NET, not "no pixel got brighter"** — and that is a wave
    // VIS1a correction rather than a loosening. The AO blur is depth-aware since
    // this wave, so adding an occluder changes which *neighbours* a probe pixel's
    // blur accepts, and a pixel next to the new silhouette can legitimately end up
    // sampling a less-occluded tap than it did before. That is a second-order
    // effect of the bilateral weight, not occlusion adding light. What may never
    // happen is the frame getting brighter overall.
    assert!(
        net < 0,
        "{what}: the shared surface gained {net} luminance once the occluder \
         joined the prepass — occlusion may only ever remove ambient light"
    );
    assert!(
        brighter * 2 < darker,
        "{what}: {brighter} shared pixels got BRIGHTER against {darker} darker — \
         too many for the bilateral blur's edge redistribution to explain"
    );

    // **AND IT LANDS ON THE OBJECT** (wave VIS1a audit). Counting darkened
    // pixels says the occluder's depth reached the prepass; it does not say the
    // prepass wrote the depth the COLOUR pass writes. A contributor whose
    // depth-only pipeline skinned with a different palette, morphed with a
    // different clipmap or simply multiplied its model matrix in the other order
    // would still darken plenty of pixels — somewhere else. That is the failure
    // this widening actually risks (self-shadowing, a halo one silhouette away
    // from the thing casting it), and the count above is blind to it.
    //
    // So: the occluder's own screen footprint is exactly the mask the loop skips,
    // and a contact shadow must sit against it. The ladder is printed before it
    // is asserted — the threshold below is read off these numbers rather than
    // guessed.
    let occ: Vec<bool> = (0..off_w.len())
        .step_by(4)
        .map(|i| off_w[i..i + 3] != off_o[i..i + 3])
        .collect();
    let dark: Vec<bool> = (0..off_w.len())
        .step_by(4)
        .map(|i| {
            let a = i32::from(on_w[i]) + i32::from(on_w[i + 1]) + i32::from(on_w[i + 2]);
            let b = i32::from(on_o[i]) + i32::from(on_o[i + 1]) + i32::from(on_o[i + 2]);
            off_w[i..i + 3] == off_o[i..i + 3] && a < b - 1
        })
        .collect();
    let mut grown = occ.clone();
    let mut near = [0usize; 4];
    for (band, step) in [4usize, 4, 8, 16].into_iter().enumerate() {
        grown = dilate(&grown, W as usize, H as usize, step);
        near[band] = dark.iter().zip(&grown).filter(|(d, g)| **d && **g).count();
    }
    eprintln!(
        "prepass placement — {what}: of {darker} darkened pixels, {} lie within \
         4 px of the occluder's own footprint, {} within 8, {} within 16, {} \
         within 32",
        near[0], near[1], near[2], near[3]
    );
    // 32 px at 320x180 is a fifth of the frame's height, and the AO radius on
    // these fixtures (1.5 m at 2-5 m of camera distance) projects to well under
    // that. A displaced prepass moves the whole contact shadow bodily, so this
    // falls off a cliff rather than degrading: mutation-measured by translating
    // one contributor's depth-pass model matrix by a metre.
    assert!(
        near[3] * 5 > darker * 4,
        "{what}: only {} of {darker} darkened pixels are within 32 px of the \
         occluder's own screen footprint — the geometry is in the prepass but at \
         a different place from where the colour pass draws it, which is a \
         self-shadowing pattern rather than ambient occlusion",
        near[3]
    );
}

/// Chebyshev dilation of a boolean mask by `r`, separably. Used by
/// [`assert_prepass_reaches`] to ask "is this darkened pixel anywhere near the
/// thing that is supposed to have darkened it".
fn dilate(mask: &[bool], w: usize, h: usize, r: usize) -> Vec<bool> {
    let mut row = vec![false; mask.len()];
    for y in 0..h {
        for x in 0..w {
            let lo = x.saturating_sub(r);
            let hi = (x + r).min(w - 1);
            row[y * w + x] = mask[y * w + lo..=y * w + hi].iter().any(|b| *b);
        }
    }
    let mut out = vec![false; mask.len()];
    for y in 0..h {
        let lo = y.saturating_sub(r);
        let hi = (y + r).min(h - 1);
        for x in 0..w {
            out[y * w + x] = (lo..=hi).any(|yy| row[yy * w + x]);
        }
    }
    out
}

/// Terrain — the one that matters most, because a level whose content *is* its
/// ground had no ambient occlusion at all before this wave.
///
/// Terrain is both the occluder and the receiver (`terrain.wgsl` is composed
/// `LitDeform` and samples `ao_tex` at `@group(3)`), so there is no separate
/// control to take: the frame with SSAO off IS the control, and before this wave
/// the frame with SSAO on was byte-identical to it. Measured, so that is what is
/// asserted.
#[test]
fn the_prepass_sees_terrain() {
    let Some(gpu) = gpu_or_skip() else { return };
    let scene = RenderScene {
        grid_enabled: false,
        terrains: vec![hill_terrain(33, 1.0, 2, 2)],
        ..Default::default()
    };
    let view = look_view(DVec3::new(32.0, 24.0, -12.0), DVec3::new(32.0, 3.0, 32.0));
    let off = render_with(&gpu, &scene, &view, RenderSettings::default());
    let on = render_with(&gpu, &scene, &view, coverage_ssao());
    let (mean, max) = image_diff(&off, &on, W, H);
    let sum = |img: &[u8]| -> u64 {
        img.chunks(4)
            .map(|p| u64::from(p[0]) + u64::from(p[1]) + u64::from(p[2]))
            .sum()
    };
    eprintln!(
        "prepass coverage — terrain: SSAO moved the frame by mean {mean:.6} / max \
         {max:.6}; total luminance {} lit against {} occluded",
        sum(&off),
        sum(&on)
    );
    assert!(
        mean > 0.0005,
        "terrain: SSAO changed nothing (mean {mean}, max {max}) — the ground is \
         not in the depth prepass, so the AO texture is a uniform 1.0"
    );
    assert!(
        sum(&on) < sum(&off),
        "terrain: SSAO brightened the frame — occlusion may only ever remove \
         ambient light"
    );
}

/// A skinned character standing on a rigid floor: the contact shadow around its
/// feet is the thing a rigid-only prepass could not draw.
#[test]
fn the_prepass_sees_a_skinned_character() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (sk, clip, mesh) = skinned_cylinder();
    let mut floor_only = RenderScene {
        grid_enabled: false,
        ..Default::default()
    };
    floor_only.instances.push(ao_floor());
    floor_only.lights.push(RenderLight {
        kind: LightKind::Directional,
        color: [1.0, 0.98, 0.95],
        intensity: 1.2,
        direction: Vec3::new(0.3, 0.9, 0.3).normalize(),
        position: DVec3::ZERO,
        range: 0.0,
        ..RenderLight::default()
    });
    floor_only.mark_dirty();

    let mut scene = floor_only.clone();
    scene.skinned_meshes.push(std::sync::Arc::new(mesh));
    scene.skinned.push(SkinnedInstance {
        vt: Default::default(),
        translation: DVec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
        color: [0.75, 0.55, 0.35, 1.0],
        metallic: 0.0,
        roughness: 0.6,
        emissive: [0.0; 3],
        id: 2,
        mesh: 0,
        palette: palette_at(&sk, &clip, 0.5),
        shadow: inf_render::SkinnedShadow::BindSphere,
    });
    scene.mark_dirty();
    let view = look_view(DVec3::new(3.2, 1.6, 3.6), DVec3::new(0.0, 0.6, 0.0));
    assert_prepass_reaches(&gpu, "skinned", &scene, &floor_only, &view);
}

/// A carved SDF volume, measured against a rigid probe box resting on its slab.
///
/// **The probe is not decoration.** `voxel.wgsl` is composed `Plain` and binds no
/// environment group at all — the P21.1 ruling — so a voxel surface samples no AO
/// and never will until that ruling changes. What putting a voxel volume in the
/// prepass buys is therefore the occlusion it casts on *everything else*, and
/// that is what this measures.
#[test]
fn the_prepass_sees_a_voxel_volume() {
    let Some(gpu) = gpu_or_skip() else { return };
    // The fixture's slab top is the sample plane y = 15 at 0.5 m per voxel, i.e.
    // world y = 7.5; the dome is centred on world (8, 7.5, 8) with a 4.5 m radius.
    // The probe sits on the flat slab, clear of the dome.
    let probe = MeshInstance::lit(
        DVec3::new(13.0, 8.25, 8.0),
        Quat::IDENTITY,
        Vec3::new(3.0, 1.5, 3.0),
        [0.75, 0.72, 0.68, 1.0],
        7,
    );
    let mut scene = voxel_scene();
    scene.instances.push(probe);
    scene.mark_dirty();
    let mut probe_only = RenderScene {
        grid_enabled: false,
        lights: scene.lights.clone(),
        ..Default::default()
    };
    probe_only.instances.push(probe);
    probe_only.mark_dirty();
    let view = look_view(DVec3::new(20.0, 11.0, 20.0), DVec3::new(12.0, 8.0, 8.0));
    assert_prepass_reaches(&gpu, "voxel", &scene, &probe_only, &view);
}

/// Fracture debris on a rigid floor. There is no committed fracture golden to
/// hang this on, so the fixture is built here: two cube-shaped chunks side by
/// side, which is all the arm needs — the claim is about the *seam*, not about
/// the geometry.
#[test]
fn the_prepass_sees_fracture_debris() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mut vertices = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    // Six independent quads so every face carries its own outward normal.
    let faces: [([f32; 3], [f32; 3], [f32; 3]); 6] = [
        ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ([0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ([1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]),
        ([-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
        ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]),
        ([0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
    ];
    for (n, u, v) in faces {
        let base = vertices.len() as u32;
        for (su, sv) in [(-1.0f32, -1.0f32), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
            vertices.push(inf_render::scene::RenderFractureVertex {
                pos: [
                    (n[0] + u[0] * su + v[0] * sv) * 0.5,
                    (n[1] + u[1] * su + v[1] * sv) * 0.5,
                    (n[2] + u[2] * su + v[2] * sv) * 0.5,
                ],
                normal: n,
                uv: [su * 0.5 + 0.5, sv * 0.5 + 0.5],
            });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    let mut floor_only = RenderScene {
        grid_enabled: false,
        ..Default::default()
    };
    floor_only.instances.push(ao_floor());
    floor_only.lights.push(RenderLight {
        kind: LightKind::Directional,
        color: [1.0, 0.98, 0.95],
        intensity: 1.4,
        direction: Vec3::new(0.3, 0.9, 0.3).normalize(),
        position: DVec3::ZERO,
        range: 0.0,
        ..RenderLight::default()
    });
    floor_only.mark_dirty();

    let mut scene = floor_only.clone();
    // Two chunks side by side, so there is a crevice for the occlusion to find.
    for (i, x) in [-0.52f64, 0.52].into_iter().enumerate() {
        scene
            .fracture_chunks
            .push(inf_render::scene::RenderFractureChunk {
                entity: 1,
                chunk: i as u32,
                translation: DVec3::new(x, 0.5, 0.0),
                rotation: Quat::IDENTITY,
                vertices: vertices.clone(),
                indices: indices.clone(),
                color: [0.7, 0.68, 0.65, 1.0],
                metallic: 0.0,
                roughness: 0.8,
                emissive: [0.0; 3],
                version: 1,
            });
    }
    scene.mark_dirty();
    let view = look_view(DVec3::new(2.2, 1.7, 2.6), DVec3::new(0.0, 0.5, 0.0));
    assert_prepass_reaches(&gpu, "fracture", &scene, &floor_only, &view);
}

/// TAA multi-frame stability smoke (P13.3a): with TAA ON and a **static** camera,
/// render N frames on one renderer (the history accumulates). Asserts (1) no NaN /
/// out-of-range garbage ever appears, and (2) after convergence consecutive frames
/// differ by a small, bounded amount (jitter+history settle to a steady image) —
/// not a pixel golden, since TAA is intentionally non-deterministic frame to
/// frame. Skips with no GPU adapter.
#[test]
fn taa_multiframe_stable() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mut scene = RenderScene {
        grid_enabled: true,
        ..Default::default()
    };
    for (i, (x, z, c)) in [
        (0.0, 0.0, [0.80, 0.20, 0.20]),
        (2.0, -1.0, [0.20, 0.70, 0.30]),
        (-1.8, 1.2, [0.25, 0.45, 0.95]),
    ]
    .into_iter()
    .enumerate()
    {
        scene.instances.push(MeshInstance::lit(
            DVec3::new(x, 0.5, z),
            Quat::from_rotation_y(0.3),
            Vec3::ONE,
            [c[0], c[1], c[2], 1.0],
            i as u32 + 1,
        ));
    }
    scene.mark_dirty();
    let view = overlook_view();

    let settings = RenderSettings {
        taa: true,
        ..RenderSettings::default()
    };
    let target = HeadlessTarget::new(&gpu, W, H);
    let mut renderer = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
    renderer.set_settings(settings);

    let mut prev: Option<Vec<u8>> = None;
    let mut last_delta = (1.0f32, 1.0f32);
    for f in 0..12 {
        renderer.render(&gpu, &scene, &view, &target.view, (W, H));
        let img = target.read_rgba(&gpu).expect("readback");
        // A NaN blowout would clamp the whole buffer to black or white.
        let nonblack = img.chunks(4).any(|p| p[0] > 5 || p[1] > 5 || p[2] > 5);
        let nonwhite = img
            .chunks(4)
            .any(|p| p[0] < 250 || p[1] < 250 || p[2] < 250);
        assert!(nonblack && nonwhite, "frame {f} degenerate (NaN blowout?)");
        if let Some(p) = &prev {
            last_delta = image_diff(p, &img, W, H);
        }
        prev = Some(img);
    }
    let (mean, max) = last_delta;
    assert!(
        mean < 0.02 && max < 0.35,
        "TAA did not converge: last frame delta mean {mean}, max {max}"
    );
}

// ── P13.1b: GPU-driven virtualized-geometry (meshlet) path ───────────────────

// The fixture: an `n×n` grid quad-plane over x,z ∈ [-1, 1] displaced by a smooth
// bi-sinusoid, so it has real curvature (nontrivial normal cones + several LOD
// levels) and `2·n·n` triangles' worth of meshlets.
//
// It lives in `inf_vgeom::test_support` — ONE generator, shared with the occlusion
// / streaming / frame-budget suites and with the player's activation gate — and its
// trig is bit-portable, so every platform cooks it to the identical meshlet DAG.
// `vgeom_dense.png` and `vgeom_far.png` were re-blessed exactly once, when it moved
// off `std` sin/cos; no other golden is affected, because no other golden draws it.
use inf_vgeom::test_support::dense_grid_mesh;

const VGEOM_ASSET: u128 = 0x1313_1b00_dead_beef;

fn vgeom_scene(mesh: Arc<VgeomMesh>, scale: f32) -> RenderScene {
    let mut scene = RenderScene {
        grid_enabled: true,
        vgeom_assets: vec![VgeomAsset::from_mesh(VGEOM_ASSET, &mesh).expect("index the vmesh")],
        ..Default::default()
    };
    scene.vgeom_instances.push(VgeomInstance::lit(
        VGEOM_ASSET,
        DVec3::ZERO,
        Quat::IDENTITY,
        Vec3::splat(scale),
        [0.72, 0.52, 0.30, 1.0],
        1,
    ));
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

fn vgeom_settings() -> RenderSettings {
    RenderSettings {
        vgeom: VgeomSettings {
            enabled: true,
            ..VgeomSettings::default()
        },
        ..RenderSettings::default()
    }
}

/// vgeom **dense** golden (P13.1b): the dense mesh at close range under an angled
/// key light, drawn entirely through the GPU meshlet path (cull+LOD compute →
/// vertex-pulled indirect draw). Structural gate: the meshlet surface is lit (the
/// path actually rasterized geometry) and the visible-meshlet count read back from
/// the cull compute is > 0. Determinism via `check_golden_with`; strict pixel diff
/// opt-in. The classic path is untouched (no `MeshInstance`s), so every other
/// golden stays byte-identical.
#[test]
fn golden_vgeom_dense() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mesh = Arc::new(dense_grid_mesh(40));
    let scene = vgeom_scene(mesh.clone(), 2.0);
    // Close overlook of the ~4 m mesh.
    let view = look_view(DVec3::new(0.0, 3.2, 4.6), DVec3::new(0.0, 0.0, 0.0));

    let img = check_golden_with(&gpu, "vgeom_dense", &scene, &view, vgeom_settings());
    let lit = img
        .chunks(4)
        .any(|p| p[0] as u16 + p[1] as u16 + p[2] as u16 > 150);
    assert!(lit, "expected the meshlet surface to be lit");

    // The cull compute selected some meshlets for this frame.
    let visible = cull_visible(
        &gpu,
        &mesh,
        &scene.vgeom_instances,
        &view,
        &vgeom_settings().vgeom,
    );
    assert!(
        !visible.is_empty(),
        "expected visible meshlets at close range"
    );
}

/// vgeom **far** golden (P13.1b) + the **LOD proof**: the same dense mesh viewed
/// from far away resolves to a COARSER cut — the cull compute selects strictly
/// FEWER meshlets than at close range (larger projected screen-error threshold ⇒
/// coarser LOD). Determinism via `check_golden_with`; strict pixel diff opt-in.
#[test]
fn golden_vgeom_far() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mesh = Arc::new(dense_grid_mesh(40));
    let scene = vgeom_scene(mesh.clone(), 2.0);

    let close = look_view(DVec3::new(0.0, 3.2, 4.6), DVec3::new(0.0, 0.0, 0.0));
    let far = look_view(DVec3::new(0.0, 26.0, 38.0), DVec3::new(0.0, 0.0, 0.0));

    let img = check_golden_with(&gpu, "vgeom_far", &scene, &far, vgeom_settings());
    // Something rendered (the mesh is small but present).
    let any = img.chunks(4).any(|p| p[0] > 8 || p[1] > 8 || p[2] > 8);
    assert!(any, "expected the far meshlet mesh to render");

    let s = vgeom_settings().vgeom;
    let n_close = cull_visible(&gpu, &mesh, &scene.vgeom_instances, &close, &s).len();
    let n_far = cull_visible(&gpu, &mesh, &scene.vgeom_instances, &far, &s).len();
    eprintln!(
        "vgeom LOD proof: {} total meshlets, close cut = {n_close}, far cut = {n_far}",
        mesh.meshlet_count()
    );
    assert!(
        n_close > 0 && n_far > 0,
        "both cuts non-empty (close {n_close}, far {n_far})"
    );
    assert!(
        n_far < n_close,
        "LOD proof: far cut should select fewer meshlets (close {n_close}, far {n_far})"
    );
}

/// CPU-vs-GPU cut parity (P13.1b — the strongest gate): for a fixed camera with
/// the whole mesh comfortably in-frustum and cone culling off, the GPU cull
/// compute's visible meshlet set (read back) must **exactly** equal the CPU
/// reference `cpu_visible_set` (the identical LOD cut + frustum filter), which in
/// turn equals `VgeomMesh::select(t)` (the offline reference rule). The
/// per-instance threshold `t` is a single scalar uploaded verbatim, so the
/// branchless cut is bit-identical on both sides.
#[test]
fn vgeom_cpu_gpu_cut_parity() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mesh = dense_grid_mesh(40);

    // The whole mesh (spans ±1, scale 1) sits well inside a 60° frustum at this
    // distance — no meshlet is near a frustum boundary, so float divergence can't
    // flip a cull. Cone culling is off (its per-normal boundary is the only place
    // CPU/GPU could disagree); it is exercised by the pure `cpu_visible_set` unit
    // tests instead.
    let view = look_view(DVec3::new(0.0, 2.2, 4.2), DVec3::new(0.0, 0.0, 0.0));
    let inst = VgeomInstance::lit(
        VGEOM_ASSET,
        DVec3::ZERO,
        Quat::IDENTITY,
        Vec3::ONE,
        [0.7, 0.7, 0.7, 1.0],
        1,
    );
    let settings = VgeomSettings {
        enabled: true,
        cone_cull: false,
        frustum_cull: true,
        occlusion: false,
        two_pass: false,
        pixel_error: 1.0,
        debug_meshlets: false,
        ..VgeomSettings::default()
    };

    let readback =
        cull_visible_streamed(&gpu, &mesh, std::slice::from_ref(&inst), &view, &settings);
    // The streamer pages in what this camera can USE, which is a prefix strictly
    // shorter than the whole asset — this instance's threshold cannot reach the
    // finest pages. That is the interesting case, not a defect: the assertion
    // below is that the clamped cut it produces is nevertheless IDENTICAL to the
    // unclamped one.
    assert!(
        readback.resident_pages >= 1 && readback.resident_pages <= readback.total_pages,
        "residency must be a prefix ({} of {})",
        readback.resident_pages,
        readback.total_pages
    );
    // Single instance ⇒ instance index is always 0; extract the meshlet ids.
    let gpu_meshlets: Vec<u32> = readback.pairs.iter().map(|e| e[1]).collect();

    // CPU reference (same math as the shader).
    let origin = view.origin;
    let model = origin.model_matrix(inst.translation, inst.rotation, inst.scale);
    let max_scale = inst.scale.abs().max_element().max(1e-6);
    let inv_scale = inst.scale.max(Vec3::splat(1e-6)).recip();
    let normal_mat = Mat3::from_quat(inst.rotation) * Mat3::from_diagonal(inv_scale);
    let eye = view.eye_local();
    let center_world = model.transform_point3(Vec3::from(mesh.center));
    let radius = mesh.radius * max_scale;
    let t = lod_threshold(
        eye,
        center_world,
        radius,
        max_scale,
        &view,
        settings.pixel_error,
    );
    let planes = frustum_planes(view.view_proj());
    let cpu_meshlets = cpu_visible_set(
        &mesh,
        model,
        normal_mat,
        eye,
        t,
        max_scale,
        &planes,
        cull_flags(&settings),
        readback.floor_lod,
    );

    assert!(!cpu_meshlets.is_empty(), "reference cut is empty");
    assert_eq!(
        gpu_meshlets, cpu_meshlets,
        "GPU visible set must equal the CPU reference (frustum + LOD)"
    );

    // **The equivalence gate, at the cut.** The streamer paged in a strict prefix,
    // yet the clamped cut it produced is byte-for-byte the cut the pre-P18.2
    // whole-upload path produced. That is not luck: `ideal_page_count` grants
    // exactly the pages whose `max_parent_error` still exceeds this instance's
    // threshold, and a page below that bound holds only meshlets whose
    // `parent_error <= t` — which the cut rejects anyway. So the pages streaming
    // declines to load are precisely the ones the cut could never have selected,
    // and clamping to the wanted floor is a no-op on the drawn set. This is why
    // all 36 goldens stay byte-identical with streaming on.
    let unclamped = cpu_visible_set(
        &mesh,
        model,
        normal_mat,
        eye,
        t,
        max_scale,
        &planes,
        cull_flags(&settings),
        0,
    );
    assert_eq!(
        cpu_meshlets, unclamped,
        "the cut at the streamer's own wanted floor must equal the unclamped cut — \
         streaming may cost VRAM, never detail the camera asked for"
    );

    // And the CPU reference (frustum passes everything here) equals the offline
    // rule VgeomMesh::select(t) — the meshlet DAG cut.
    let select_ids: Vec<u32> = mesh.select(t).map(|(i, _)| i as u32).collect();
    assert_eq!(
        cpu_meshlets, select_ids,
        "frustum passes all in-view meshlets ⇒ cut == VgeomMesh::select(t)"
    );
}

// ── P13.3b: cascaded shadow maps + dynamic GI ────────────────────────────────

/// CSM golden (P13.3b): a caster/receiver scene — three boxes standing on a large
/// white floor slab, lit by a single **low** directional sun so they cast long
/// shadows across the floor. Shadows ON. Structural gate: the cascaded shadows
/// **darken** the frame overall (direct light removed in the occluded floor
/// regions) while the scene stays lit — proving the cascade render → PCF sample
/// path ran. Determinism via `check_golden_with`; strict pixel diff opt-in. With
/// shadows off every other golden stays byte-stable (verified).
#[test]
fn golden_csm() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mut scene = RenderScene {
        grid_enabled: false,
        ..Default::default()
    };
    // A large receiver floor slab (top surface at y = 0).
    scene.instances.push(MeshInstance::lit(
        DVec3::new(0.0, -0.25, 0.0),
        Quat::IDENTITY,
        Vec3::new(12.0, 0.5, 12.0),
        [0.78, 0.78, 0.80, 1.0],
        1,
    ));
    // Three caster boxes.
    for (i, (x, z)) in [(-2.2, 0.5), (1.0, -1.5), (2.6, 1.2)]
        .into_iter()
        .enumerate()
    {
        scene.instances.push(MeshInstance::lit(
            DVec3::new(x, 0.9, z),
            Quat::from_rotation_y(0.3),
            Vec3::new(0.9, 1.8, 0.9),
            [0.80, 0.42, 0.32, 1.0],
            i as u32 + 2,
        ));
    }
    // A low directional sun (grazing → long shadows).
    scene.lights.push(RenderLight {
        kind: LightKind::Directional,
        color: [1.0, 0.97, 0.9],
        intensity: 3.0,
        direction: Vec3::new(0.55, 0.32, 0.45).normalize(),
        position: DVec3::ZERO,
        range: 0.0,
        ..RenderLight::default()
    });
    scene.mark_dirty();
    let view = look_view(DVec3::new(5.0, 5.5, 8.5), DVec3::new(0.0, 0.5, 0.0));

    let shadows_on = RenderSettings {
        shadows: ShadowSettings {
            enabled: true,
            ..ShadowSettings::default()
        },
        ..RenderSettings::default()
    };

    let img = check_golden_with(&gpu, "csm", &scene, &view, shadows_on);
    let img_off = render_with(&gpu, &scene, &view, RenderSettings::default());

    let sum = |img: &[u8]| -> u64 {
        img.chunks(4)
            .map(|p| p[0] as u64 + p[1] as u64 + p[2] as u64)
            .sum()
    };
    let (sum_on, sum_off) = (sum(&img), sum(&img_off));
    assert!(
        sum_on < sum_off,
        "CSM should darken shadowed regions (on {sum_on} vs off {sum_off})"
    );
    let lit = img
        .chunks(4)
        .any(|p| p[0] as u16 + p[1] as u16 + p[2] as u16 > 200);
    assert!(lit, "expected the CSM scene to stay lit");
}

/// **Cascade blending** (P18.4 deliverable 6) — the P13 deferral: cascades used to
/// switch instantly, so the resolution change drew a hard line across the ground
/// wherever a split fell inside the frame.
///
/// The shadow range is squeezed to 14 m so both split boundaries land on the
/// visible floor. The metric is the largest **row-to-row** luminance step down the
/// receiving floor: a hard switch puts a step there that the blend band spreads
/// out. `cascade_blend = 0` must reproduce the pre-P18.4 frame exactly.
#[test]
fn csm_cascade_blend_softens_the_split_seam() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mut scene = RenderScene {
        grid_enabled: false,
        ..Default::default()
    };
    // A long floor running away from the camera, so successive cascades cover
    // successive screen rows.
    scene.instances.push(MeshInstance::lit(
        DVec3::new(0.0, -0.25, -8.0),
        Quat::IDENTITY,
        Vec3::new(30.0, 0.5, 40.0),
        [0.80, 0.80, 0.82, 1.0],
        1,
    ));
    // A picket of casters spread across X and Z: their long shadows sweep across
    // both split lines, so the seam has plenty of penumbra to show up in (a
    // single caster leaves the split crossing only a handful of pixels).
    let mut id = 2u32;
    for z in [1.0f64, -1.0, -3.0, -5.5] {
        for x in [-3.0f64, -0.5, 2.0, 4.5] {
            scene.instances.push(MeshInstance::lit(
                DVec3::new(x, 0.7, z),
                Quat::from_rotation_y(0.3),
                Vec3::new(0.7, 1.4, 0.7),
                [0.80, 0.42, 0.32, 1.0],
                id,
            ));
            id += 1;
        }
    }
    scene.lights.push(RenderLight {
        kind: LightKind::Directional,
        color: [1.0, 0.97, 0.9],
        intensity: 3.0,
        direction: Vec3::new(0.55, 0.42, 0.45).normalize(),
        position: DVec3::ZERO,
        range: 0.0,
        ..RenderLight::default()
    });
    scene.mark_dirty();
    // Low and pitched down, so the two split boundaries (≈1.6 m and ≈4.3 m at a
    // 14 m shadow range) land inside the frame rather than under the camera.
    let view = look_view(DVec3::new(0.0, 1.2, 4.0), DVec3::ZERO);

    let with_blend = |blend: f32| RenderSettings {
        shadows: ShadowSettings {
            enabled: true,
            max_distance: 14.0,
            cascade_blend: blend,
            ..ShadowSettings::default()
        },
        ..RenderSettings::default()
    };

    let hard = render_with(&gpu, &scene, &view, with_blend(0.0));
    let soft = render_with(&gpu, &scene, &view, with_blend(0.3));
    assert_ne!(hard, soft, "the cascade blend changed nothing");

    // Where the blend acts, by screen row. A floor row maps monotonically to a
    // view distance, so this is a distance profile: the blend must move pixels
    // ONLY in the bands ending at a split, and must leave the last cascade (there
    // is no cascade 3 to blend into) untouched.
    let row_diffs: Vec<u32> = (0..H)
        .map(|y| {
            (0..W)
                .filter(|&x| px(&hard, x, y) != px(&soft, x, y))
                .count() as u32
        })
        .collect();
    for d in 0..10u32 {
        let s: u32 = row_diffs[(H * d / 10) as usize..(H * (d + 1) / 10) as usize]
            .iter()
            .sum();
        eprintln!("csm blend decile {d}: {s} differing pixels");
    }
    let touched: u32 = row_diffs.iter().sum();
    assert!(
        touched > 200,
        "the blend barely touched anything ({touched})"
    );
    // The far field — the LAST cascade and the sky above it — must be untouched:
    // `shadow_factor` only blends where there IS a next cascade to blend into.
    let far: u32 = row_diffs[..(H * 30 / 100) as usize].iter().sum();
    assert_eq!(
        far, 0,
        "the blend reached the last cascade / the sky, which have nothing to \
         blend into ({far} pixels)"
    );
    // ...and the near field, well inside cascade 0, is untouched too: this is a
    // band at the boundary, not a global change of how shadows are sampled.
    let near: u32 = row_diffs[(H * 80 / 100) as usize..].iter().sum();
    assert_eq!(
        near, 0,
        "the blend reached the middle of cascade 0 ({near})"
    );

    // The escape hatch really is an escape hatch: `0` is the old code path, and a
    // project that wants it back pays nothing for the option.
    assert_eq!(
        hard,
        render_with(&gpu, &scene, &view, with_blend(0.0)),
        "the zero-blend path is not deterministic"
    );
}

/// Mean red/green ratio of the floor pixels in a screen band (rows `y0..y1`,
/// central columns), skipping near-black (unlit / off-floor) pixels. The proof
/// metric for `golden_gi_bleed`.
fn band_red_ratio(img: &[u8], y0: u32, y1: u32) -> f32 {
    let (mut r, mut g, mut n) = (0.0f64, 0.0f64, 0u32);
    for y in y0..y1 {
        for x in (W * 30 / 100)..(W * 70 / 100) {
            let p = px(img, x, y);
            // Skip near-black pixels (not lit floor).
            if (p[0] as u16 + p[1] as u16 + p[2] as u16) < 30 {
                continue;
            }
            r += p[0] as f64;
            g += p[1] as f64;
            n += 1;
        }
    }
    if n == 0 || g == 0.0 {
        return 0.0;
    }
    (r / g.max(1.0)) as f32
}

/// The GI proof golden (P13.3b) — **`golden_gi_bleed`**: a white floor and a tall
/// **red** wall, with the sun angled so the wall's front face is lit and the floor
/// receives grazing light. With dynamic GI ON, the floor **near the wall** picks up
/// a red single-bounce, so its mean red/green ratio exceeds the far floor's by a
/// clear margin — asserted structurally over two screen bands (the region assert,
/// not a pixel compare). Also asserts determinism (two renders byte-identical).
/// GI off keeps the hemispheric ambient path byte-stable (verified).
#[test]
fn golden_gi_bleed() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mut scene = RenderScene {
        grid_enabled: false,
        ..Default::default()
    };
    // White floor slab (top surface at y = 0), extending toward the wall (−Z).
    scene.instances.push(MeshInstance::lit(
        DVec3::new(0.0, -0.25, 0.5),
        Quat::IDENTITY,
        Vec3::new(12.0, 0.5, 11.0),
        [0.90, 0.90, 0.90, 1.0],
        1,
    ));
    // A tall RED wall along the far (−Z) edge, front face toward +Z (the floor).
    scene.instances.push(MeshInstance::lit(
        DVec3::new(0.0, 1.5, -4.0),
        Quat::IDENTITY,
        Vec3::new(11.0, 3.0, 0.5),
        [0.90, 0.05, 0.05, 1.0],
        2,
    ));
    // Sun from +Z and above: lights the wall's +Z face and grazes the floor. Kept
    // moderate so the single-bounce GI (not the direct white light) shapes the
    // floor's near-wall colour.
    scene.lights.push(RenderLight {
        kind: LightKind::Directional,
        color: [1.0, 0.98, 0.95],
        intensity: 2.0,
        direction: Vec3::new(0.0, 0.5, 1.0).normalize(),
        position: DVec3::ZERO,
        range: 0.0,
        ..RenderLight::default()
    });
    scene.mark_dirty();

    // Look toward the wall base: the wall sits high in the frame, the floor fills
    // the lower two thirds (near-wall floor above, far floor below).
    let view = look_view(DVec3::new(0.0, 4.5, 7.0), DVec3::new(0.0, 0.0, -1.5));

    // Through `gi_settings` like every other GI scene in this file — it was the
    // one that spelled its block out inline, which meant EDIT1's π would have had
    // to be written twice. Same extent, rays and bounce as before.
    let gi_on = gi_settings(40.0, 48, 2.5);

    let img = check_golden_with(&gpu, "gi_bleed", &scene, &view, gi_on);

    // Determinism: a second render is byte-identical to the golden render.
    let img2 = render_with(&gpu, &scene, &view, gi_on);
    let (mean, max) = image_diff(&img, &img2, W, H);
    assert!(
        mean == 0.0 && max == 0.0,
        "GI must be deterministic (mean {mean}, max {max})"
    );

    // Region assert: the near-wall floor band is redder than the far floor band.
    let near = band_red_ratio(&img, H * 40 / 100, H * 52 / 100);
    let far = band_red_ratio(&img, H * 74 / 100, H * 90 / 100);
    eprintln!("gi_bleed red/green ratio: near-wall {near:.3}, far {far:.3}");
    assert!(
        near > far + 0.05,
        "expected red colour bleed near the wall (near {near:.3} vs far {far:.3})"
    );

    // Sanity: the near-wall floor actually picks up red (ratio clearly > 1).
    assert!(
        near > 1.03,
        "near-wall floor not reddened (ratio {near:.3})"
    );
}

// ── P18.4 GI v2 ──────────────────────────────────────────────────────────────
//
// Everything below drives the rebuilt GI: full-scene voxelization (terrain,
// skinned, vgeom), the lifted instance cap, the atmosphere sky term, emissive
// injection, the specular term + SSR, temporal amortization, and the resizable
// resources joining the `ResourceKey`.

/// Mean `num`/`den` channel ratio of the LIT pixels in a screen rectangle,
/// skipping near-black ones. The generalization of [`band_red_ratio`] the P18.4
/// scenes need (green bleed from an emissive, red bleed from a wall).
fn region_channel_ratio(
    img: &[u8],
    x: std::ops::Range<u32>,
    y: std::ops::Range<u32>,
    num: usize,
    den: usize,
) -> f32 {
    let (mut a, mut b, mut n) = (0.0f64, 0.0f64, 0u32);
    for yy in y {
        for xx in x.clone() {
            let p = px(img, xx, yy);
            if (p[0] as u16 + p[1] as u16 + p[2] as u16) < 12 {
                continue;
            }
            a += p[num] as f64;
            b += p[den] as f64;
            n += 1;
        }
    }
    if n == 0 || b == 0.0 {
        return 0.0;
    }
    (a / b.max(1.0)) as f32
}

/// Mean luminance (0..255) of a screen rectangle.
fn region_mean(img: &[u8], x: std::ops::Range<u32>, y: std::ops::Range<u32>) -> f32 {
    let (mut s, mut n) = (0.0f64, 0u32);
    for yy in y {
        for xx in x.clone() {
            let p = px(img, xx, yy);
            s += (p[0] as f64 + p[1] as f64 + p[2] as f64) / 3.0;
            n += 1;
        }
    }
    if n == 0 {
        0.0
    } else {
        (s / n as f64) as f32
    }
}

/// Render `frames` consecutive frames through **one** renderer (so its temporal
/// state — the GI probe cursor — actually advances) and return the last frame plus
/// the GI audit it published.
fn render_frames_with(
    gpu: &GpuContext,
    scene: &RenderScene,
    view: &RenderView,
    settings: RenderSettings,
    frames: u32,
) -> (Vec<u8>, GiAudit) {
    let target = HeadlessTarget::new(gpu, W, H);
    let mut renderer = EngineRenderer::new(gpu, HEADLESS_FORMAT);
    renderer.set_settings(settings);
    for _ in 0..frames.max(1) {
        renderer.render(gpu, scene, view, &target.view, (W, H));
    }
    (
        target.read_rgba(gpu).expect("readback"),
        renderer.gi_audit(),
    )
}

/// GI settings with the P18.4 defaults and an explicit extent/rays/intensity.
/// The factor every GI scene in this file carries in its authored `intensity`
/// since **wave EDIT1**, and why the images below did not move when the engine's
/// ambient normalisation did.
///
/// `gi_irradiance` used to return **irradiance** — `π·L` on a uniform field —
/// which the lit passes then spent as if it were exit radiance, so the ambient
/// half of the engine ran π times the direct half (`crates/inf-render/tests/
/// gi_normalisation.rs` measures it: a white Lambert cube in a uniform
/// environment returned **3.22×** that environment's own radiance). EDIT1 folds
/// the Lambert `1/π` into the convolution constants, so `gi.intensity` now
/// multiplies a quantity that is π times smaller.
///
/// These scenes exist to pin GI's *behaviour* — the red bleed near a wall, an
/// emissive bar lighting a floor with no analytic light, a reflection that has a
/// direction. Every one of them was authored to show that behaviour at a chosen
/// **bounce strength**, and a bounce strength is preserved across a unit change
/// by scaling the multiplier. So the argument below is still the pre-EDIT1
/// number and the π rides here, once.
///
/// **Four of the five hold their images this way and one does not**, and that is
/// worth reading rather than smoothing over. EDIT1 removed a π from TWO places:
/// the consumer (`gi_irradiance`, which every term rides) and the gather's HIT
/// radiance (`gi_probes.wgsl`, which only the sun-bounce term rides). One
/// multiplier on `intensity` restores the sky-miss term exactly and the
/// sun-bounce term only partly, so the compensation is exact for a scene the sky
/// dominates and approximate for one a lit wall does:
///
/// | golden | subject | mean diff |
/// |---|---|---|
/// | `gi_emissive` | an emissive bar (no π either side) | 0.000000 |
/// | `gi_scatter_neon` | emissive instances | 0.000000 |
/// | `gi_terrain` | a bounce off sunlit ground | 0.028975 |
/// | `gi_bleed` | a bounce off a sunlit wall | 0.044445 |
/// | `gi_specular` | that bounce, in a reflection | **0.072099** |
///
/// The first four are inside the 0.06 mean tolerance and are NOT re-blessed. The
/// fifth is re-blessed, once, with that cause — and every structural assertion it
/// exists for still passes on the new frame (grazing specular 210.9 against a
/// flat 144.0, near-wall red/green 1.367 against a far 1.091, a smooth surface
/// moving 0.1280 against a matte 0.0590).
///
/// **The showcase island does not get this compensation** — it keeps
/// `gi_intensity: 1.0`, and that is the whole point of the wave.
const GI_LAMBERT_PI: f32 = std::f32::consts::PI;

fn gi_settings(extent: f32, rays: u32, bounce: f32) -> RenderSettings {
    RenderSettings {
        gi: GiSettings {
            enabled: true,
            extent,
            rays,
            intensity: bounce * GI_LAMBERT_PI,
            ..GiSettings::default()
        },
        ..RenderSettings::default()
    }
}

/// **Emissive injection** (P18.4 deliverable 5) — `golden_gi_emissive`.
///
/// A white floor and a **green emissive** bar, and *no analytic light at all*: the
/// scene's one directional light has zero intensity, so `sun_color` in the GI
/// uniform is black and a voxel's bounce term is `albedo × 0 + emissive`. Anything
/// green on that floor arrived through the voxel volume's emissive word.
#[test]
fn golden_gi_emissive() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mut scene = RenderScene {
        grid_enabled: false,
        ..Default::default()
    };
    // White floor slab (top surface at y = 0).
    scene.instances.push(MeshInstance::lit(
        DVec3::new(0.0, -0.25, 0.0),
        Quat::IDENTITY,
        Vec3::new(14.0, 0.5, 14.0),
        [0.92, 0.92, 0.92, 1.0],
        1,
    ));
    // A near-black bar that GLOWS green, floating over the floor.
    scene.instances.push(MeshInstance {
        emissive: [0.10, 3.6, 0.10],
        ..MeshInstance::lit(
            DVec3::new(0.0, 1.5, -2.2),
            Quat::IDENTITY,
            Vec3::new(7.0, 0.5, 0.5),
            [0.02, 0.02, 0.02, 1.0],
            2,
        )
    });
    // The whole point: a directional light with ZERO radiance. Present so the
    // shaders take the light-loop path (not the fallback editor sun), contributing
    // nothing — so the bar is the only source in the scene.
    scene.lights.push(RenderLight {
        kind: LightKind::Directional,
        color: [0.0, 0.0, 0.0],
        intensity: 0.0,
        direction: Vec3::Y,
        ..RenderLight::default()
    });
    scene.mark_dirty();

    let view = look_view(DVec3::new(0.0, 3.2, 6.0), DVec3::new(0.0, 0.2, -1.6));
    let gi_on = gi_settings(24.0, 64, 2.0);

    let img = check_golden_with(&gpu, "gi_emissive", &scene, &view, gi_on);
    let off = render_with(&gpu, &scene, &view, RenderSettings::default());

    // The floor, well below the bar.
    let (fx, fy) = (
        (W * 30 / 100)..(W * 70 / 100),
        (H * 72 / 100)..(H * 95 / 100),
    );
    let green_on = region_channel_ratio(&img, fx.clone(), fy.clone(), 1, 0);
    let green_off = region_channel_ratio(&off, fx.clone(), fy.clone(), 1, 0);
    eprintln!("gi_emissive floor green/red: GI on {green_on:.3}, off {green_off:.3}");
    assert!(
        green_on > green_off + 0.08,
        "the emissive bar did not bleed green onto the floor \
         (on {green_on:.3} vs off {green_off:.3})"
    );
    assert!(
        green_on > 1.08,
        "floor is not green at all (ratio {green_on:.3}) — emissive injection is dead"
    );
    // The floor is genuinely lit by it, not merely tinted noise.
    assert!(
        region_mean(&img, fx.clone(), fy.clone()) > region_mean(&off, fx, fy) + 1.0,
        "an emissive source lit nothing"
    );
}

/// **THE VENUE INTERIOR** (wave VEN1a) — `venue_interior`.
///
/// The wave's visual claim, made as one frame: a near-black room whose every
/// photon comes from a practical.
///
/// What is in it, and which reference frame each part answers:
///
/// * a **wood floor and a raised stage** with the plank tint the module table
///   gives `ModuleShape::Stage` (`venues/0028`, `0036`);
/// * a **chrome pole** on the stage — a scatter batch at `metallic 1.0,
///   roughness 0.12`, which is what turns a red wash into one bright vertical
///   streak (`0044`);
/// * a **red stage wash** and a **blue rim**: two real spot lights, cones
///   crossing over the middle of the stage, the shape a `StageRig` produces;
/// * a **magenta neon plate** and a run of **festoon bulbs** as *scattered
///   emission* — carrying no light slot at all, and reaching the bounce through
///   `passes::gi`'s scatter staging (`0004`, `0060`);
/// * **near-zero ambient**: the sun is present and BLACK, so the light-loop
///   path runs and contributes nothing. Everything visible is a practical.
///
/// The arms are about *where the light is*, which is the whole content of the
/// reference's lighting recipe. A frame that drew the same objects under a flat
/// fill would satisfy "not black" and fail every one of them.
#[test]
fn golden_venue_interior() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mut scene = RenderScene {
        grid_enabled: false,
        ..Default::default()
    };
    // The room: a dark plank floor and a dark back wall.
    scene.instances.push(MeshInstance::lit(
        DVec3::new(0.0, -0.15, 0.0),
        Quat::IDENTITY,
        Vec3::new(18.0, 0.3, 16.0),
        [0.19, 0.13, 0.08, 1.0],
        1,
    ));
    // **The room is CLOSED**, and that is not scene dressing. A club has a
    // roof: with the box open the GI volume's sky ray reaches the floor from
    // every direction and the frame is lit by daylight with some neon in it,
    // which is the opposite of the thing this golden claims. Measured on the
    // first cut: the far corner sat at 40 of 255 with the walls off and at a
    // third of that with them on.
    for (c, h, id) in [
        (DVec3::new(0.0, 2.4, -6.0), Vec3::new(18.0, 5.0, 0.4), 2),
        (DVec3::new(-8.2, 2.4, 0.0), Vec3::new(0.4, 5.0, 16.0), 4),
        (DVec3::new(8.2, 2.4, 0.0), Vec3::new(0.4, 5.0, 16.0), 5),
        (DVec3::new(0.0, 4.9, 0.0), Vec3::new(18.0, 0.4, 16.0), 6),
        (DVec3::new(0.0, 2.4, 8.0), Vec3::new(18.0, 5.0, 0.4), 7),
    ] {
        scene.instances.push(MeshInstance::lit(
            c,
            Quat::IDENTITY,
            h,
            [0.05, 0.045, 0.05, 1.0],
            id,
        ));
    }
    // The stage: a raised plank platform, `ModuleShape::Stage`'s own tint.
    scene.instances.push(MeshInstance {
        roughness: 0.55,
        ..MeshInstance::lit(
            DVec3::new(0.0, 0.22, -1.6),
            Quat::IDENTITY,
            Vec3::new(6.4, 0.45, 3.4),
            [0.38, 0.26, 0.16, 1.0],
            3,
        )
    });
    // **The chrome pole**, as the scatter path draws it: a batch at the
    // material `ModuleShape::Pole` states. One instance, because a pole is one
    // pole and the claim is about its SHADING.
    scene.scatter.push(ScatterBatch {
        metallic: 1.0,
        roughness: 0.12,
        ..ScatterBatch::lit(
            Arc::new(ScatterData::build(
                PrimMesh::Cube,
                DVec3::ZERO,
                vec![ScatterInstance {
                    position: DVec3::new(0.0, 1.85, -1.6),
                    rotation: Quat::IDENTITY,
                    scale: Vec3::new(0.09, 2.8, 0.09),
                    color: [0.90, 0.91, 0.94, 1.0],
                }],
            )),
            DVec3::ZERO,
            0.12,
            10,
        )
    });
    // **The neon plate** — scattered emission, no light slot.
    scene.scatter.push(ScatterBatch {
        emissive: [3.4, 0.35, 3.0],
        ..ScatterBatch::lit(
            Arc::new(ScatterData::build(
                PrimMesh::Cube,
                DVec3::ZERO,
                vec![ScatterInstance {
                    position: DVec3::new(-4.6, 2.6, -5.6),
                    rotation: Quat::IDENTITY,
                    scale: Vec3::new(1.6, 0.8, 0.14),
                    color: [0.05, 0.05, 0.06, 1.0],
                }],
            )),
            DVec3::ZERO,
            0.3,
            11,
        )
    });
    // **The festoon** — a chain of small warm bulbs along the back wall.
    scene.scatter.push(ScatterBatch {
        emissive: [1.5, 1.15, 0.75],
        ..ScatterBatch::lit(
            Arc::new(ScatterData::build(
                PrimMesh::Cube,
                DVec3::ZERO,
                (0..9)
                    .map(|i| ScatterInstance {
                        position: DVec3::new(f64::from(i) * 1.5 - 6.0, 3.5, -5.7),
                        rotation: Quat::IDENTITY,
                        scale: Vec3::splat(0.16),
                        color: [0.06, 0.06, 0.06, 1.0],
                    })
                    .collect::<Vec<_>>(),
            )),
            DVec3::ZERO,
            0.4,
            12,
        )
    });
    // **The sun, and it is BLACK.** Present so the lit passes take the
    // light-loop path rather than the fallback editor sun; contributing
    // nothing, so every photon below is a practical.
    scene.lights.push(RenderLight {
        kind: LightKind::Directional,
        color: [0.0, 0.0, 0.0],
        intensity: 0.0,
        direction: Vec3::Y,
        ..RenderLight::default()
    });
    // **The rig**: a red key over the stage and a blue rim off to one side,
    // toed in so the cones cross — `StageRig`'s own shape.
    for (x, colour, intensity) in [
        (-1.7_f64, [3.0, 0.12, 0.22_f32], 26.0_f32),
        (1.7, [0.28, 0.5, 2.8], 22.0),
    ] {
        let at = DVec3::new(x, 4.2, -1.6);
        let aim = Vec3::new(-(x as f32) * 0.35, -1.0, 0.0).normalize();
        scene.lights.push(RenderLight {
            kind: LightKind::Spot,
            color: colour,
            intensity,
            // Toward the light: the negated emission direction.
            direction: -aim,
            position: at,
            range: 8.4,
            inner_cos: 20.0_f32.to_radians().cos(),
            outer_cos: 36.0_f32.to_radians().cos(),
            cast_shadows: false,
        });
    }
    scene.mark_dirty();

    let view = look_view(DVec3::new(0.0, 2.6, 7.0), DVec3::new(0.0, 1.0, -2.0));
    let lit = gi_settings(28.0, 64, 2.0);
    let img = check_golden_with(&gpu, "venue_interior", &scene, &view, lit);

    // ── (a) NEAR-BLACK AMBIENT. The room's far corners are dark; a flat fill
    // would light them as brightly as the stage.
    let stage = region_mean(
        &img,
        (W * 40 / 100)..(W * 60 / 100),
        (H * 58 / 100)..(H * 72 / 100),
    );
    let corner = region_mean(
        &img,
        (W * 2 / 100)..(W * 14 / 100),
        (H * 80 / 100)..(H * 98 / 100),
    );
    eprintln!("venue_interior: stage {stage:.2}, dark corner {corner:.2}");
    assert!(
        stage > corner * 2.0 + 8.0,
        "the stage ({stage:.2}) is not markedly brighter than the room's corner \
         ({corner:.2}) — this frame is lit by a fill, not by practicals"
    );
    // **42 was ABOVE the defect it names** (VEN1a audit). The roofless first cut
    // measured 40.06, so this clause passed on the very frame it exists to
    // reject and the mutation was caught by the *ratio* clause above it instead
    // — an accident, and one that evaporates the day the stage gets brighter.
    // Mutation-verified at the audit: with the roof and three walls removed the
    // corner reads **40.06** and the closed room reads **4.2**, so 15 falsifies
    // the named defect with 2.7× under it and 3.6× of headroom over the frame
    // that passes. (A near-black region mean is where an adapter would differ
    // most in relative terms and least in absolute ones; the headroom is 10.8
    // of 255.)
    assert!(
        corner < 15.0,
        "the room's far corner is at {corner:.2} — a venue interior's ambient is \
         near black and everything else in this arm rests on it"
    );

    // ── (b) THE WASH IS COLOURED, AND THE TWO SIDES DIFFER. A red key on one
    // side of the stage and a blue rim on the other is the reference's whole
    // lighting signature, and one white spot would fail both halves.
    let left = region_channel_ratio(
        &img,
        (W * 30 / 100)..(W * 46 / 100),
        (H * 60 / 100)..(H * 72 / 100),
        0,
        2,
    );
    let right = region_channel_ratio(
        &img,
        (W * 54 / 100)..(W * 70 / 100),
        (H * 60 / 100)..(H * 72 / 100),
        0,
        2,
    );
    eprintln!("venue_interior: stage red/blue left {left:.3}, right {right:.3}");
    // Named by MEASUREMENT and not by which side of the screen each lamp lands
    // on: a view's handedness is the harness's business and this arm's claim is
    // about the two lamps.
    let (warm, cool) = (left.max(right), left.min(right));
    assert!(
        warm > cool + 0.5,
        "the two stage lamps are the same colour (r/b {left:.3} and {right:.3})"
    );
    assert!(
        warm > 1.6,
        "neither lamp lays a RED wash on the planks ({warm:.3})"
    );
    assert!(
        cool < 1.4,
        "neither lamp lays a cool rim on the planks ({cool:.3}) — one red key and \
         one blue rim is the reference's whole stage signature"
    );

    // ── (c) THE POLE IS BRIGHT. A chrome cylinder in a coloured wash is one
    // vertical specular streak; a dielectric at roughness 0.75 is a grey stick.
    let pole = region_mean(
        &img,
        (W * 48 / 100)..(W * 52 / 100),
        (H * 34 / 100)..(H * 56 / 100),
    );
    let beside = region_mean(
        &img,
        (W * 62 / 100)..(W * 68 / 100),
        (H * 34 / 100)..(H * 56 / 100),
    );
    eprintln!("venue_interior: pole {pole:.2}, wall beside it {beside:.2}");
    assert!(
        pole > beside + 6.0,
        "the chrome pole ({pole:.2}) does not stand out from the wall behind it \
         ({beside:.2})"
    );

    // ── (d) THE NEON IS THE BRIGHTEST THING IN THE FRAME, and it is magenta.
    let (nx, ny) = (
        (W * 29 / 100)..(W * 37 / 100),
        (H * 31 / 100)..(H * 40 / 100),
    );
    let neon = region_mean(&img, nx.clone(), ny.clone());
    // The HALO the plate throws on the boards around it — the thing the sign's
    // standoffs exist for, and the half of it that is not simply "a bright
    // rectangle".
    let halo = region_channel_ratio(
        &img,
        (W * 20 / 100)..(W * 28 / 100),
        (H * 34 / 100)..(H * 46 / 100),
        2,
        1,
    );
    eprintln!("venue_interior: neon {neon:.2}, halo blue/green {halo:.3}");
    assert!(
        neon > stage + 40.0,
        "the neon ({neon:.2}) is not the brightest thing in the room (stage {stage:.2})"
    );
    assert!(
        halo > 1.25,
        "the wall beside the neon plate is not magenta (blue/green {halo:.3}); a sign bolted flat to a wall has nowhere for its glow to spill"
    );
}

/// **Scattered emission reaches the bounce** (wave VEN1a) —
/// `golden_gi_scatter_neon`.
///
/// The venue substrate's first claim, made falsifiable. A white floor, a
/// **scatter batch** of small magenta plates on a wall line, and *no analytic
/// light at all*: the scene's one directional light has zero radiance, so a
/// voxel's bounce term is `albedo × 0 + emissive` and every magenta photon on
/// that floor arrived through `passes::gi`'s scatter staging.
///
/// It is deliberately a scatter batch and not a `MeshInstance`, because that is
/// the whole point: `golden_gi_emissive` beside it has proved instance emission
/// bounces since P18.4, and a grammar-built venue's neon, string lights and lit
/// panes are **none of them instances**. Before this wave they were drawn and
/// lit nothing.
///
/// Two control arms, because one is not enough to name the path:
///
/// * the same scene with the batch's `emissive` **zeroed** — if the floor is
///   magenta in both, the emission is not what put it there;
/// * the same *emissive* scene with **GI off** — if the floor is magenta in
///   both, the emission reached it through the raster rather than through the
///   bounce, and this arm would pass on an engine where nothing was staged.
///
/// The GI-on control's floor is not dark: it carries the volume's **sky** term.
/// That is why the claim is made about the red/green *ratio* and not about
/// brightness alone — a bluish sky bounce and a magenta neon bounce are the
/// same number of lumens and opposite hues.
#[test]
fn golden_gi_scatter_neon() {
    let Some(gpu) = gpu_or_skip() else { return };
    let neon = |emissive: [f32; 3]| {
        let mut scene = RenderScene {
            grid_enabled: false,
            ..Default::default()
        };
        // White floor slab (top surface at y = 0).
        scene.instances.push(MeshInstance::lit(
            DVec3::new(0.0, -0.25, 0.0),
            Quat::IDENTITY,
            Vec3::new(14.0, 0.5, 14.0),
            [0.92, 0.92, 0.92, 1.0],
            1,
        ));
        // A run of near-black plates on a line, ALL of them scattered. Each is
        // 0.8 m across — above the volume's half-voxel dust floor, so the arm
        // is about emission and not about the size reject.
        let plates: Vec<ScatterInstance> = (0..7)
            .map(|i| ScatterInstance {
                position: DVec3::new(f64::from(i) * 1.1 - 3.3, 1.5, -2.2),
                rotation: Quat::IDENTITY,
                scale: Vec3::new(0.8, 0.8, 0.4),
                color: [0.02, 0.02, 0.02, 1.0],
            })
            .collect();
        scene.scatter.push(ScatterBatch {
            emissive,
            ..ScatterBatch::lit(
                Arc::new(ScatterData::build(PrimMesh::Cube, DVec3::ZERO, plates)),
                DVec3::ZERO,
                0.6,
                2,
            )
        });
        // Present so the shaders take the light-loop path rather than the
        // fallback editor sun, and contributing exactly nothing.
        scene.lights.push(RenderLight {
            kind: LightKind::Directional,
            color: [0.0, 0.0, 0.0],
            intensity: 0.0,
            direction: Vec3::Y,
            ..RenderLight::default()
        });
        scene.mark_dirty();
        scene
    };

    let scene = neon([3.4, 0.20, 2.6]);
    let dark = neon([0.0, 0.0, 0.0]);
    let view = look_view(DVec3::new(0.0, 3.2, 6.0), DVec3::new(0.0, 0.2, -1.6));
    let gi_on = gi_settings(24.0, 64, 2.0);

    let img = check_golden_with(&gpu, "gi_scatter_neon", &scene, &view, gi_on);
    let unlit_batch = render_with(&gpu, &dark, &view, gi_on);
    let no_gi = render_with(&gpu, &scene, &view, RenderSettings::default());

    // The floor, well below the plates.
    let (fx, fy) = (
        (W * 30 / 100)..(W * 70 / 100),
        (H * 72 / 100)..(H * 95 / 100),
    );
    let ratio = |img: &[u8]| region_channel_ratio(img, fx.clone(), fy.clone(), 0, 1);
    let mean = |img: &[u8]| region_mean(img, fx.clone(), fy.clone());
    let (r_on, r_dark, r_nogi) = (ratio(&img), ratio(&unlit_batch), ratio(&no_gi));
    let (m_on, m_dark) = (mean(&img), mean(&unlit_batch));
    eprintln!(
        "gi_scatter_neon floor red/green: neon+GI {r_on:.3} (mean {m_on:.2}), \
         unlit batch+GI {r_dark:.3} (mean {m_dark:.2}), neon no-GI {r_nogi:.3}"
    );
    assert!(
        r_on > r_dark + 0.08,
        "the scattered neon did not bleed magenta onto the floor \
         (neon {r_on:.3} vs unlit batch {r_dark:.3}) — scatter is not staged into GI"
    );
    assert!(
        r_on > r_nogi + 0.08,
        "the floor is as magenta with GI off ({r_nogi:.3}) as with it on ({r_on:.3}) — \
         whatever tinted it, it was not the bounce"
    );
    assert!(
        m_on > m_dark + 1.0,
        "a scattered emissive source lit nothing (mean {m_on:.2} vs {m_dark:.2})"
    );
    // The GI-on control's floor IS lit — by the sky term — so the arm above is
    // about hue, and this pins that the sky's hue is the opposite one. Without
    // it a scene whose sky happened to be magenta would satisfy everything.
    assert!(
        r_dark < 1.0,
        "the sky bounce alone is already red-dominant ({r_dark:.3}); the ratio \
         arms above cannot then name the neon as the source"
    );
}

/// **Specular** (P18.4 deliverable 4a) — `golden_gi_specular`.
///
/// A smooth, semi-metallic floor under a bright red wall. With the SH specular on,
/// the floor reconstructs radiance along its **reflection vector** (which points at
/// the wall) instead of the cosine-weighted hemisphere average, so the reflected
/// red is stronger and falls off with the reflection geometry rather than with the
/// diffuse lobe.
#[test]
fn golden_gi_specular() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mut scene = RenderScene {
        grid_enabled: false,
        ..Default::default()
    };
    // A SMOOTH, semi-metallic floor: metallic kills the diffuse term, so what is
    // left on it is (almost) purely the specular path this golden is about.
    scene.instances.push(MeshInstance {
        metallic: 0.85,
        roughness: 0.18,
        ..MeshInstance::lit(
            DVec3::new(0.0, -0.25, 0.5),
            Quat::IDENTITY,
            Vec3::new(12.0, 0.5, 11.0),
            [0.85, 0.85, 0.88, 1.0],
            1,
        )
    });
    // A tall RED wall along the far edge.
    scene.instances.push(MeshInstance::lit(
        DVec3::new(0.0, 1.5, -4.0),
        Quat::IDENTITY,
        Vec3::new(11.0, 3.0, 0.5),
        [0.92, 0.05, 0.05, 1.0],
        2,
    ));
    scene.lights.push(RenderLight {
        kind: LightKind::Directional,
        color: [1.0, 0.98, 0.95],
        intensity: 2.0,
        direction: Vec3::new(0.0, 0.5, 1.0).normalize(),
        position: DVec3::ZERO,
        range: 0.0,
        ..RenderLight::default()
    });
    scene.mark_dirty();

    let view = look_view(DVec3::new(0.0, 3.2, 7.0), DVec3::new(0.0, 0.0, -1.5));
    let mut spec_on = gi_settings(40.0, 48, 2.0);
    spec_on.gi.specular = true;
    let mut spec_off = spec_on;
    spec_off.gi.specular = false;

    let img = check_golden_with(&gpu, "gi_specular", &scene, &view, spec_on);
    let flat = render_with(&gpu, &scene, &view, spec_off);

    assert_ne!(img, flat, "the SH specular term changed nothing");

    let cols = (W * 25 / 100)..(W * 75 / 100);
    let near = (H * 40 / 100)..(H * 50 / 100); // floor close to the wall
    let far = (H * 90 / 100)..(H * 100 / 100); // floor close to the camera

    // (a) A smooth metal floor reflects MORE than the flat `f0 × 0.5` constant it
    // replaced — the grazing-angle Fresnel the split-sum term carries and the
    // constant could not.
    let lit_spec = region_mean(&img, cols.clone(), far.clone());
    let lit_flat = region_mean(&flat, cols.clone(), far.clone());
    eprintln!("gi_specular grazing floor: specular {lit_spec:.1}, flat {lit_flat:.1}");
    assert!(
        lit_spec > lit_flat + 8.0,
        "the grazing floor did not brighten (specular {lit_spec:.1} vs flat {lit_flat:.1})"
    );

    // (b) Directionality: the floor near the wall reflects it and the floor under
    // the camera does not, because the reflection vector points somewhere else.
    let red_near = region_channel_ratio(&img, cols.clone(), near.clone(), 0, 1);
    let red_far = region_channel_ratio(&img, cols.clone(), far.clone(), 0, 1);
    eprintln!("gi_specular red/green: near-wall {red_near:.3}, far {red_far:.3}");
    assert!(
        red_near > red_far + 0.08,
        "the reflection is not direction-dependent (near {red_near:.3}, far {red_far:.3})"
    );

    // (c) **Roughness reads.** The lobe sharpening (`1 − roughness`) is what makes
    // this a reflection rather than a second ambient constant, so a smooth floor
    // must depart from the term it replaced further than a matte one does. (The
    // exact reduction — that a *uniform* radiance field at full roughness gives
    // back the retired `f0 × 0.5` — is a statement about the arithmetic, and is
    // pinned in the pure half where a uniform field can actually be constructed:
    // `gi::tests::specular_reduces_to_the_retired_ambient_constant`.)
    let diff_at = |r: f32| {
        let mut sc = scene.clone();
        sc.instances[0].roughness = r;
        sc.mark_dirty();
        let on = render_with(&gpu, &sc, &view, spec_on);
        let off = render_with(&gpu, &sc, &view, spec_off);
        image_diff(&on, &off, W, H).0
    };
    let (smooth, matte) = (diff_at(0.18), diff_at(1.0));
    eprintln!("gi_specular roughness: mean diff smooth {smooth:.4}, matte {matte:.4}");
    assert!(
        smooth > matte * 1.2,
        "roughness does not read: smooth {smooth:.4} vs matte {matte:.4}"
    );

    // SSR (deliverable 4b, rewritten in wave VIS1a): deterministic, and it
    // actually reaches the shading.
    //
    // **Two frames, not one.** SSR v2 samples the PREVIOUS frame's resolved
    // colour; on frame 0 that texture is a zero-initialized allocation and the
    // `ssr.w` history flag holds the march off entirely, so a single-frame render
    // can never show a reflection. That is the same shape `taa_multiframe_stable`
    // has, for the same reason, and it is why SSR is opt-in.
    let mut ssr = spec_on;
    ssr.ssr.enabled = true;
    let a = render_frames_with(&gpu, &scene, &view, ssr, 2).0;
    let b = render_frames_with(&gpu, &scene, &view, ssr, 2).0;
    assert_eq!(a, b, "SSR is not deterministic");
    let base2 = render_frames_with(&gpu, &scene, &view, spec_on, 2).0;
    assert_ne!(
        a, base2,
        "SSR changed nothing — the screen-space march never found a hit"
    );
    // ...and the first frame really is the one with no history, which is what
    // makes the two-frame shape necessary rather than superstitious.
    let one = render_frames_with(&gpu, &scene, &view, ssr, 1).0;
    let one_base = render_frames_with(&gpu, &scene, &view, spec_on, 1).0;
    assert_eq!(
        one, one_base,
        "SSR reflected something on frame 0, where the colour history is a \
         zero-initialized allocation"
    );
}

/// **Terrain voxelization** (P18.4 deliverable 1) — `golden_gi_terrain`.
///
/// A **red** heightfield with a tall white wall standing on it. The wall's lit face
/// picks up a red bounce off the sunlit ground — the same proof `golden_gi_bleed`
/// makes for a rigid box, except the bouncing surface is *terrain*, which before
/// P18.4 was invisible to the voxelizer: a landscape neither occluded a bounce nor
/// contributed one, so GI over open ground was GI over a void.
///
/// The control is the identical scene with the terrain projection **removed**. The
/// measured band is wall in both renders, so the only thing that can move it is the
/// probe field.
#[test]
fn golden_gi_terrain() {
    let Some(gpu) = gpu_or_skip() else { return };
    // A flat, RED terrain: one 2×2-tile patch centred near the origin.
    let res = 33u32;
    let mps = 1.0f64;
    let span = (res - 1) as f64 * mps;
    let mut tiles = Vec::new();
    for tz in 0..2 {
        for tx in 0..2 {
            tiles.push(RenderTerrainTile {
                key: TerrainTileKey::lod0((tx, tz)),
                origin: DVec3::new(tx as f64 * span - span, 0.0, tz as f64 * span - span),
                heights: vec![0.0; (res * res) as usize],
                weights: Vec::new(),
                biomes: Vec::new(),
                height_bounds: (0.0, 0.0),
                holes: Vec::new(),
                version: 1,
            });
        }
    }
    let terrain = RenderTerrain {
        id: 7,
        tile_resolution: res,
        meters_per_sample: mps,
        tiles,
        layers: [
            RenderTerrainLayer {
                albedo: [0.95, 0.06, 0.06, 1.0],
                roughness: 0.9,
                tex_scale: 8.0,
                vt: Default::default(),
            },
            RenderTerrainLayer::default(),
            RenderTerrainLayer::default(),
            RenderTerrainLayer::default(),
        ],
        macro_variation: 0.0,
        biome_palette: Vec::new(),
    };

    let mut scene = RenderScene {
        grid_enabled: false,
        terrains: vec![terrain],
        ..Default::default()
    };
    // A tall WHITE wall standing on the ground, facing the camera.
    scene.instances.push(MeshInstance::lit(
        DVec3::new(0.0, 2.5, -4.0),
        Quat::IDENTITY,
        Vec3::new(12.0, 5.0, 0.5),
        [0.94, 0.94, 0.94, 1.0],
        1,
    ));
    // Sun from +Z and above: the wall's +Z face is lit, and so is the ground in
    // front of it (the ground the bounce has to come from).
    scene.lights.push(RenderLight {
        kind: LightKind::Directional,
        color: [1.0, 0.98, 0.94],
        intensity: 1.2,
        direction: Vec3::new(0.0, 0.55, 0.84).normalize(),
        position: DVec3::ZERO,
        range: 0.0,
        ..RenderLight::default()
    });
    scene.mark_dirty();

    let view = look_view(DVec3::new(0.0, 2.6, 6.0), DVec3::new(0.0, 1.6, -4.0));
    let gi_on = gi_settings(40.0, 64, 1.0);

    let img = check_golden_with(&gpu, "gi_terrain", &scene, &view, gi_on);

    // The same scene with the terrain projection removed — everything drawn in the
    // measured band is identical, so any difference there is GI.
    let mut no_terrain = scene.clone();
    no_terrain.terrains.clear();
    no_terrain.mark_dirty();
    let bare = render_with(&gpu, &no_terrain, &view, gi_on);

    // The wall's mid-height band: wall pixels in BOTH renders (the ground line is
    // below it and the sky above), so nothing but the probe field can move it.
    let band = (
        (W * 35 / 100)..(W * 65 / 100),
        (H * 30 / 100)..(H * 55 / 100),
    );
    let red_with = region_channel_ratio(&img, band.0.clone(), band.1.clone(), 0, 1);
    let red_without = region_channel_ratio(&bare, band.0.clone(), band.1.clone(), 0, 1);
    eprintln!("gi_terrain wall red/green: with terrain {red_with:.3}, without {red_without:.3}");
    assert!(
        region_mean(&img, band.0.clone(), band.1.clone()) > 100.0,
        "the measured band is not the lit wall"
    );

    // **The two things a red floor does to a white wall**, measured per channel
    // rather than as one ratio — wave FIX3.
    //
    // The threshold here used to be a ratio lift of 0.10, and the lift measured
    // at `e8451338` was **+0.257** (1.226 against 0.969). It is **+0.027** now,
    // and the cause is a physics fix rather than a regression: the probe march's
    // bounce carries a **cosine** since FIX3 (the ray's own direction is the hit
    // face's normal proxy), where before it handed every voxel full
    // normal-incidence sun — EDIT1's carried "GI bounce has no n·l", closed.
    // Measured on this very fixture by putting `ndl = 1.0` back into
    // `gi_probes.wgsl` and changing nothing else: the lift goes straight to
    // **+0.331**. The old number was the over-estimate, not the signal.
    //
    // So the arm reads the two channels instead, where the effect is not one
    // small ratio but two large, independent numbers:
    //
    //   red   182.30 -> 177.23   (a drop of 5.07)
    //   green 184.35 -> 174.51   (a drop of 9.85)
    //
    // Adding the ground makes the wall DARKER, because a wall standing on
    // something receives less sky than one floating in an open sphere of it —
    // that is the `-sky` term the FIX3 probe march subtracts, and it is the
    // occlusion the sky-irradiance term would otherwise have handed out for
    // free. And it makes it darker **twice as fast in green as in red**, which
    // is the red albedo coming back. A terrain that reached the voxelizer with
    // no albedo would fail the second assert; one that never reached it at all
    // would fail the first.
    let px_band = (band.0.start, band.1.start, band.0.end, band.1.end);
    let ch =
        |img: &[u8], c: usize| mean_channel(img, c, px_band.0, px_band.1, px_band.2, px_band.3);
    let red_drop = ch(&bare, 0) - ch(&img, 0);
    let green_drop = ch(&bare, 1) - ch(&img, 1);
    eprintln!(
        "gi_terrain wall channels: red {:.2} -> {:.2} ({red_drop:+.2}), green {:.2} -> {:.2} ({green_drop:+.2})",
        ch(&bare, 0),
        ch(&img, 0),
        ch(&bare, 1),
        ch(&img, 1)
    );
    assert!(
        green_drop > 3.0,
        "the terrain did not occlude any of the sky the wall was receiving \
         (green {green_drop:+.2} of 255) — terrain is invisible to GI"
    );
    assert!(
        green_drop > red_drop * 1.4,
        "the terrain blocked sky without returning any red (green {green_drop:+.2} \
         against red {red_drop:+.2}) — its ALBEDO is invisible to GI"
    );
    assert!(
        red_with > red_without + 0.02,
        "the terrain did not bounce red onto the wall \
         (with {red_with:.3} vs without {red_without:.3}) — terrain is invisible to GI"
    );

    // ...and the audit says the columns were actually sampled.
    let (_, audit) = render_frames_with(&gpu, &scene, &view, gi_on, 1);
    assert!(
        audit.terrain_columns > 0,
        "no terrain column reached the voxelizer"
    );
}

/// **The GI volume does not depend on terrain residency** (P18.4, audit finding).
///
/// `RenderTerrain::tiles` is the streamer's **camera-driven** working set, so a
/// voxelizer that sampled "the finest resident tile" would make GI occupancy and
/// albedo a function of where the camera has *been* — and no golden would notice,
/// because every golden renders a fully-resident terrain. The voxelizer therefore
/// reads only the projection's coarsest asset level (`gi::voxelization_tiles`), the
/// terrain analogue of vgeom's always-resident root page.
///
/// This drives the same scene through two genuinely different residency states —
/// fully resident (level 0 + 1 + 2) versus punched out (the coarse pyramid alone,
/// what the streamer publishes with the camera far away) — and byte-compares the
/// **GI volume and probe buffers**, not the pixels. Pixels would be the wrong
/// instrument: the two states legitimately *draw* different terrain detail, and
/// only the voxel volume can say whether GI saw the same world.
#[test]
fn gi_terrain_voxelization_is_independent_of_residency() {
    let Some(gpu) = gpu_or_skip() else { return };
    let res = 33;
    let full = streamed_terrain(res, 1.0);
    // The far-camera residency: the coarse pyramid alone. `streamed_terrain`
    // deliberately omits level-2 page (1,1), so this also covers a hole.
    let coarse_only = RenderTerrain {
        tiles: full
            .tiles
            .iter()
            .filter(|t| t.key.lod == 2)
            .cloned()
            .collect(),
        ..full.clone()
    };
    assert_eq!(full.max_lod(), 2);
    assert_eq!(coarse_only.max_lod(), 2);
    assert!(
        coarse_only.tiles.len() < full.tiles.len(),
        "the punched-out state must actually be smaller"
    );

    let scene_with = |terrain: RenderTerrain| {
        let mut s = RenderScene {
            grid_enabled: false,
            terrains: vec![terrain],
            ..Default::default()
        };
        s.instances.push(MeshInstance::lit(
            DVec3::new(40.0, 12.0, 40.0),
            Quat::IDENTITY,
            Vec3::new(6.0, 4.0, 0.5),
            [0.9, 0.9, 0.9, 1.0],
            1,
        ));
        s.lights.push(RenderLight {
            kind: LightKind::Directional,
            color: [1.0, 0.98, 0.94],
            intensity: 1.4,
            direction: Vec3::new(0.3, 0.7, 0.6).normalize(),
            ..RenderLight::default()
        });
        s.mark_dirty();
        s
    };
    // Sitting on the terrain, where a residency difference is at its loudest.
    let view = look_view(DVec3::new(40.0, 14.0, 60.0), DVec3::new(40.0, 6.0, 20.0));
    let gi_on = gi_settings(40.0, 48, 1.0);

    let gi_state = |terrain: RenderTerrain| -> (Vec<u8>, Vec<u8>, GiAudit) {
        let scene = scene_with(terrain);
        let target = HeadlessTarget::new(&gpu, W, H);
        let mut renderer = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
        renderer.set_settings(gi_on);
        renderer.render(&gpu, &scene, &view, &target.view, (W, H));
        let _ = target.read_rgba(&gpu).expect("readback");
        let r = renderer.gi_resources();
        (r.read_voxels(&gpu), r.read_sh(&gpu), renderer.gi_audit())
    };

    let (vox_full, sh_full, audit_full) = gi_state(full.clone());
    let (vox_coarse, sh_coarse, audit_coarse) = gi_state(coarse_only.clone());

    // The terrain has to actually be reaching the voxelizer, or this test would
    // pass on two empty volumes.
    assert!(
        audit_full.terrain_columns > 0,
        "no terrain column reached the voxelizer ({audit_full:?})"
    );
    assert_eq!(audit_full.terrain_columns, audit_coarse.terrain_columns);
    assert!(
        vox_full.iter().any(|&b| b != 0),
        "the voxel volume is empty — nothing was voxelized at all"
    );

    // Summarize rather than `assert_eq!` on the raw buffers: the volume is 2 MB at
    // High, and a failing `assert_eq!` would print all of it twice.
    let mismatch = |a: &[u8], b: &[u8]| -> Option<(usize, usize)> {
        if a.len() != b.len() {
            return Some((0, a.len().max(b.len())));
        }
        let differing = a.iter().zip(b).filter(|(x, y)| x != y).count();
        let first = a.iter().zip(b).position(|(x, y)| x != y);
        first.map(|i| (i, differing))
    };
    assert!(
        mismatch(&vox_full, &vox_coarse).is_none(),
        "the GI voxel volume changed with terrain residency — GI occupancy is a \
         function of camera history ({:?} = (first differing byte, count of {}))",
        mismatch(&vox_full, &vox_coarse),
        vox_full.len()
    );
    assert!(
        mismatch(&sh_full, &sh_coarse).is_none(),
        "the GI probe buffer changed with terrain residency ({:?} of {})",
        mismatch(&sh_full, &sh_coarse),
        sh_full.len()
    );

    // Sanity that the two states are genuinely different inputs: they DRAW
    // differently (fine pages carry detail the coarse ones decimated away), which
    // is exactly why the comparison above is on the volume and not on pixels.
    assert_ne!(
        render_with(&gpu, &scene_with(full), &view, gi_on),
        render_with(&gpu, &scene_with(coarse_only), &view, gi_on),
        "the two residency states rendered identically — the fixture is not \
         exercising a residency difference at all"
    );
}

/// **A running time-of-day clock must not starve the amortization sweep** (P18.4,
/// audit finding). The sweep key holds a *bucketed* sun (`gi::sun_bucket`, ≈0.50°);
/// with raw `f32` bits it would reset every frame under a live clock, pinning the
/// cursor in its first slice forever — amortization paying a full update's CPU
/// cost for one slice of freshness, precisely where it was meant to help.
///
/// The cursor is otherwise unobservable from outside, which is why `GiAudit`
/// carries it.
#[test]
fn gi_amortization_survives_a_running_time_of_day_clock() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mut base = RenderScene {
        grid_enabled: false,
        ..Default::default()
    };
    base.instances.push(MeshInstance::lit(
        DVec3::new(0.0, -0.25, 0.0),
        Quat::IDENTITY,
        Vec3::new(12.0, 0.5, 12.0),
        [0.9, 0.9, 0.9, 1.0],
        1,
    ));
    base.mark_dirty();
    let view = look_view(DVec3::new(0.0, 4.0, 7.0), DVec3::ZERO);
    let mut settings = gi_settings(40.0, 32, 1.0);
    settings.gi.probe_budget = 256; // 2048 probes ⇒ an 8-frame sweep

    // Drive the sun through `scene.sun` (the P17.1 projected sun, which is what a
    // live TimeOfDay clock moves), stepping it by `deg_per_frame`.
    let run = |deg_per_frame: f64, frames: u32| -> Vec<GiAudit> {
        let target = HeadlessTarget::new(&gpu, W, H);
        let mut renderer = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
        renderer.set_settings(settings);
        let mut out = Vec::new();
        for f in 0..frames {
            let deg = 40.0 + deg_per_frame * f as f64;
            let r = deg.to_radians();
            let mut scene = base.clone();
            scene.sun = SunParams {
                direction: Vec3::new(r.cos() as f32, r.sin() as f32, 0.3),
                ..SunParams::default()
            };
            // `scene.version` churns here exactly as the shipped player's does
            // (`project_scene` re-projects and marks dirty every frame). It used to
            // be part of the sweep key, so this test had to pin it to isolate the
            // sun; since that reset was removed it does not, and leaving the churn
            // in makes this a second witness for the fix — see
            // `gi_amortization_survives_the_shipped_players_scene_version_churn`.
            scene.mark_dirty();
            renderer.render(&gpu, &scene, &view, &target.view, (W, H));
            let _ = target.read_rgba(&gpu).expect("readback");
            out.push(renderer.gi_audit());
        }
        out
    };

    // A real-time clock at rate = 1: 15°/hour, i.e. 1/14400° per frame at 60 fps.
    // Sub-bucket by three orders of magnitude — the cursor must sweep freely.
    let slow = run(1.0 / 14_400.0, 10);
    let cursors: Vec<u32> = slow.iter().map(|a| a.probe_cursor).collect();
    eprintln!("gi TOD sweep (rate 1): cursors {cursors:?}");
    assert_eq!(
        cursors[0], 256,
        "the first frame should have taken one slice"
    );
    assert!(
        cursors.iter().any(|&c| c > 256),
        "the cursor never advanced past its first slice — a running clock is \
         resetting the sweep every frame ({cursors:?})"
    );
    // Eight frames is a full sweep at this budget, so the cursor must wrap.
    assert_eq!(
        cursors[7], 0,
        "the sweep did not complete in 8 frames ({cursors:?})"
    );

    // A sun that crosses a bucket every frame DOES reset — the key still does its
    // job, it is only the resolution that changed.
    let fast = run(2.0, 6);
    let fast_cursors: Vec<u32> = fast.iter().map(|a| a.probe_cursor).collect();
    eprintln!("gi TOD sweep (2°/frame): cursors {fast_cursors:?}");
    assert!(
        fast_cursors.iter().all(|&c| c == 256),
        "a bucket-crossing sun did not restart the sweep ({fast_cursors:?})"
    );
}

/// **THE SHIPPED PLAYER'S `scene.version` CHURN MUST NOT STARVE THE SWEEP.**
///
/// `inf_player::render::project_scene` re-projects the whole scene every frame and
/// ends with `RenderScene::mark_dirty()`, so `scene.version` increments every
/// frame in a shipped build. While that version was part of `GiSweepKey`, the
/// sweep reset every frame and the cursor never left its first slice —
/// amortization paying a full update's CPU cost for one slice of freshness, in the
/// *shipped* build only. The editor viewport hid it, because `sync_from_doc` is
/// version-gated there and a static document holds its version still: a
/// PIE-vs-shipping divergence in everything but name.
///
/// This drives the renderer the way the player drives it — nothing moving but the
/// version — and asserts the cursor sweeps and wraps. The anti-vacuity arm below
/// keeps the reset honest: the things that genuinely invalidate the integration
/// still restart it.
#[test]
fn gi_amortization_survives_the_shipped_players_scene_version_churn() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mut base = RenderScene {
        grid_enabled: false,
        ..Default::default()
    };
    base.instances.push(MeshInstance::lit(
        DVec3::new(0.0, -0.25, 0.0),
        Quat::IDENTITY,
        Vec3::new(12.0, 0.5, 12.0),
        [0.9, 0.9, 0.9, 1.0],
        1,
    ));
    base.mark_dirty();
    let view = look_view(DVec3::new(0.0, 4.0, 7.0), DVec3::ZERO);
    let mut settings = gi_settings(40.0, 32, 1.0);
    settings.gi.probe_budget = 256; // 2048 probes ⇒ an 8-frame sweep

    // `extent` per frame: `None` keeps the GI volume (and so the probe geometry)
    // fixed, which is the case under test; `Some(f)` perturbs it, which must reset.
    let run = |extent: Option<&dyn Fn(u32) -> f32>, frames: u32| -> Vec<GiAudit> {
        let target = HeadlessTarget::new(&gpu, W, H);
        let mut renderer = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
        renderer.set_settings(settings);
        let mut out = Vec::new();
        for f in 0..frames {
            if let Some(ext) = extent {
                let mut s = settings;
                s.gi.extent = ext(f);
                renderer.set_settings(s);
            }
            let mut scene = base.clone();
            // THE PLAYER'S OWN BEHAVIOUR: a fresh projection, marked dirty, every
            // frame — identical content, a different version.
            scene.mark_dirty();
            renderer.render(&gpu, &scene, &view, &target.view, (W, H));
            let _ = target.read_rgba(&gpu).expect("readback");
            out.push(renderer.gi_audit());
        }
        out
    };

    let churn = run(None, 10);
    let cursors: Vec<u32> = churn.iter().map(|a| a.probe_cursor).collect();
    eprintln!("gi version-churn sweep: cursors {cursors:?}");
    // The version really is moving — otherwise this proves nothing.
    assert_eq!(cursors[0], 256, "the first frame should take one slice");
    assert!(
        cursors.iter().any(|&c| c > 256),
        "the cursor never advanced past its first slice — the shipped player's \
         per-frame `scene.version` bump is resetting the sweep ({cursors:?})"
    );
    assert_eq!(
        cursors[7], 0,
        "the sweep did not complete in 8 frames ({cursors:?})"
    );
    // Strictly monotone through the first sweep: 256, 512, … 1792, 0.
    for (i, w) in cursors[..8].windows(2).enumerate() {
        if w[1] != 0 {
            assert_eq!(w[1], w[0] + 256, "slice {i} did not advance ({cursors:?})");
        }
    }

    // ANTI-VACUITY: a change that really does invalidate the integration — the GI
    // volume extent, which moves every probe — still restarts the sweep. Without
    // this the assertions above would be satisfied by a key that never resets.
    let moved = run(Some(&|f: u32| 40.0 + f as f32 * 4.0), 6);
    let moved_cursors: Vec<u32> = moved.iter().map(|a| a.probe_cursor).collect();
    eprintln!("gi extent-change sweep: cursors {moved_cursors:?}");
    assert!(
        moved_cursors.iter().all(|&c| c == 256),
        "a moving GI volume did not restart the sweep ({moved_cursors:?})"
    );
}

/// **The instance cap is gone** (P18.4 deliverable 1) — the regression this replaces
/// is `MAX_GI_INSTANCES = 256`, which silently ignored everything past the 257th
/// instance *in scene order*, so which geometry lit a room depended on the outliner.
#[test]
fn gi_lifts_the_instance_cap_and_reports_overflow() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mut scene = RenderScene {
        grid_enabled: false,
        ..Default::default()
    };
    // 600 small boxes on a lattice inside a 24 m volume — comfortably past the
    // retired 256 cap.
    let mut id = 1u32;
    for iz in 0..10 {
        for iy in 0..6 {
            for ix in 0..10 {
                scene.instances.push(MeshInstance::lit(
                    DVec3::new(ix as f64 - 4.5, iy as f64 - 2.5, iz as f64 - 4.5),
                    Quat::IDENTITY,
                    Vec3::splat(0.4),
                    [0.7, 0.4, 0.3, 1.0],
                    id,
                ));
                id += 1;
            }
        }
    }
    assert_eq!(scene.instances.len(), 600);
    scene.lights.push(RenderLight {
        kind: LightKind::Directional,
        color: [1.0; 3],
        intensity: 2.0,
        direction: Vec3::new(0.3, 0.9, 0.3).normalize(),
        ..RenderLight::default()
    });
    scene.mark_dirty();
    // The camera centres the volume, so a 48 m extent covers the whole lattice —
    // otherwise the volume clip, not the budget, would be what limits the count.
    let view = look_view(DVec3::new(0.0, 3.0, 12.0), DVec3::ZERO);

    // Generous budget: everything inside the volume is voxelized, nothing dropped.
    let generous = gi_settings(48.0, 32, 1.0);
    let (_, audit) = render_frames_with(&gpu, &scene, &view, generous, 1);
    eprintln!("gi cap-lift audit: {audit:?}");
    assert!(
        audit.candidates > 256,
        "the test scene did not exceed the retired cap ({} candidates)",
        audit.candidates
    );
    assert_eq!(
        audit.voxelized, audit.candidates,
        "a generous budget still dropped geometry"
    );
    assert_eq!(audit.dropped, 0);
    assert!(
        audit.cell_entries >= audit.voxelized,
        "every voxelized primitive must land in at least one macro cell"
    );

    // A deliberately tight budget REPORTS the overflow instead of swallowing it.
    let mut tight = generous;
    tight.gi.instance_budget = 100;
    let (_, tight_audit) = render_frames_with(&gpu, &scene, &view, tight, 1);
    assert_eq!(tight_audit.voxelized, 100);
    assert_eq!(
        tight_audit.dropped,
        tight_audit.candidates - 100,
        "the budget overflow was not reported"
    );
    assert!(tight_audit.dropped > 0);

    // The bigger budget genuinely changes the lighting — proof that the extra
    // primitives are doing work rather than being uploaded and ignored.
    let (full_img, _) = render_frames_with(&gpu, &scene, &view, generous, 1);
    let (tight_img, _) = render_frames_with(&gpu, &scene, &view, tight, 1);
    assert_ne!(
        full_img, tight_img,
        "voxelizing 600 primitives lit the scene identically to voxelizing 100"
    );
}

/// **Skinned and vgeom geometry reach the voxelizer** (P18.4 deliverable 1).
///
/// Before this, a character cast no bounce and a Nanite-class mesh might as well
/// not have been in the room: the voxelizer saw rigid `MeshInstance` boxes and
/// nothing else. The control in both halves is the **same geometry recoloured** —
/// red versus grey — so the rasterized silhouette, the depth, and the direct
/// lighting of the measured floor band are all identical and the only thing that
/// can move it is what the probes saw.
#[test]
fn gi_sees_skinned_and_vgeom_geometry() {
    let Some(gpu) = gpu_or_skip() else { return };
    let gi_on = gi_settings(40.0, 48, 1.6);
    let view = look_view(DVec3::new(0.0, 3.0, 7.0), DVec3::new(0.0, 0.6, -2.0));
    let sun = RenderLight {
        kind: LightKind::Directional,
        color: [1.0, 0.98, 0.95],
        intensity: 1.6,
        direction: Vec3::new(0.0, 0.55, 0.84).normalize(),
        position: DVec3::ZERO,
        range: 0.0,
        ..RenderLight::default()
    };
    let floor = MeshInstance::lit(
        DVec3::new(0.0, -0.25, 0.0),
        Quat::IDENTITY,
        Vec3::new(18.0, 0.5, 18.0),
        [0.92, 0.92, 0.92, 1.0],
        1,
    );
    // A floor band to the RIGHT of the occluder (which stands left of centre), so
    // the band is bare floor in both renders.
    let band = (
        (W * 55 / 100)..(W * 85 / 100),
        (H * 62 / 100)..(H * 88 / 100),
    );

    // ── skinned: per-joint boxes carried by the live palette ──
    let (sk, clip, mesh) = skinned_cylinder();
    let mesh = std::sync::Arc::new(mesh);
    let skinned_scene = |color: [f32; 4]| {
        let mut s = RenderScene {
            grid_enabled: false,
            skinned_meshes: vec![mesh.clone()],
            ..Default::default()
        };
        s.instances.push(floor);
        s.skinned.push(SkinnedInstance {
            vt: Default::default(),
            translation: DVec3::new(-2.2, 0.0, -1.5),
            rotation: Quat::IDENTITY,
            scale: Vec3::splat(2.4),
            color,
            metallic: 0.0,
            roughness: 0.6,
            emissive: [0.0; 3],
            id: 2,
            mesh: 0,
            palette: palette_at(&sk, &clip, 0.0),
            shadow: inf_render::SkinnedShadow::BindSphere,
        });
        s.lights.push(sun);
        s.mark_dirty();
        s
    };
    let red = skinned_scene([0.95, 0.05, 0.05, 1.0]);
    let grey = skinned_scene([0.5, 0.5, 0.5, 1.0]);
    let (_, audit) = render_frames_with(&gpu, &red, &view, gi_on, 1);
    assert!(
        audit.candidates >= 2,
        "the skinned instance staged no joint boxes (audit {audit:?})"
    );
    let red_img = render_with(&gpu, &red, &view, gi_on);
    let grey_img = render_with(&gpu, &grey, &view, gi_on);
    let r_red = region_channel_ratio(&red_img, band.0.clone(), band.1.clone(), 0, 1);
    let r_grey = region_channel_ratio(&grey_img, band.0.clone(), band.1.clone(), 0, 1);
    eprintln!("gi skinned bounce: red {r_red:.4}, grey {r_grey:.4} (audit {audit:?})");
    assert!(
        r_red > r_grey + 0.002,
        "a red skinned character bounced no red onto the floor \
         (red {r_red:.4} vs grey {r_grey:.4}) — skinned geometry is invisible to GI"
    );

    // ── vgeom: the root page's meshlet spheres ──
    let vmesh = Arc::new(dense_grid_mesh(24));
    let vgeom_scene_at = |color: [f32; 4]| {
        let mut s = RenderScene {
            grid_enabled: false,
            vgeom_assets: vec![VgeomAsset::from_mesh(VGEOM_ASSET, &vmesh).expect("index the vmesh")],
            ..Default::default()
        };
        s.instances.push(floor);
        // The displaced grid stood on edge: a wall facing the camera.
        s.vgeom_instances.push(VgeomInstance::lit(
            VGEOM_ASSET,
            DVec3::new(-2.4, 1.6, -1.6),
            Quat::from_rotation_x(std::f32::consts::FRAC_PI_2),
            Vec3::splat(1.8),
            color,
            2,
        ));
        s.lights.push(sun);
        s.mark_dirty();
        s
    };
    let vred = vgeom_scene_at([0.95, 0.05, 0.05, 1.0]);
    let vgrey = vgeom_scene_at([0.5, 0.5, 0.5, 1.0]);
    let (_, vaudit) = render_frames_with(&gpu, &vred, &view, gi_on, 1);
    assert!(
        vaudit.candidates >= 2,
        "the vgeom instance staged no meshlet spheres (audit {vaudit:?})"
    );
    let vred_img = render_with(&gpu, &vred, &view, gi_on);
    let vgrey_img = render_with(&gpu, &vgrey, &view, gi_on);
    let v_red = region_channel_ratio(&vred_img, band.0.clone(), band.1.clone(), 0, 1);
    let v_grey = region_channel_ratio(&vgrey_img, band.0.clone(), band.1.clone(), 0, 1);
    eprintln!("gi vgeom bounce: red {v_red:.4}, grey {v_grey:.4} (audit {vaudit:?})");
    assert!(
        v_red > v_grey + 0.002,
        "a red vgeom mesh bounced no red onto the floor \
         (red {v_red:.4} vs grey {v_grey:.4}) — meshlet geometry is invisible to GI"
    );
}

/// **Temporal amortization determinism** (P18.4 deliverable 3).
///
/// Three properties, which together are what "deterministic amortization" means:
///
/// 1. **cold == cold** — two fresh renderers running the same frame count agree
///    (the schedule comes from a renderer-side cursor starting at 0, never from a
///    frame index, so a warm-up frame cannot desync them);
/// 2. **converged == converged** — once a full sweep has completed, a static scene
///    reproduces across runs;
/// 3. **converged == full update** — and it reproduces the *non*-amortized frame
///    byte for byte, which is the strong statement: amortization is a schedule, not
///    an approximation.
#[test]
fn gi_amortization_is_deterministic_and_converges() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mut scene = RenderScene {
        grid_enabled: false,
        ..Default::default()
    };
    scene.instances.push(MeshInstance::lit(
        DVec3::new(0.0, -0.25, 0.5),
        Quat::IDENTITY,
        Vec3::new(12.0, 0.5, 11.0),
        [0.90, 0.90, 0.90, 1.0],
        1,
    ));
    scene.instances.push(MeshInstance::lit(
        DVec3::new(0.0, 1.5, -4.0),
        Quat::IDENTITY,
        Vec3::new(11.0, 3.0, 0.5),
        [0.90, 0.05, 0.05, 1.0],
        2,
    ));
    scene.lights.push(RenderLight {
        kind: LightKind::Directional,
        color: [1.0, 0.98, 0.95],
        intensity: 2.0,
        direction: Vec3::new(0.0, 0.5, 1.0).normalize(),
        ..RenderLight::default()
    });
    scene.mark_dirty();
    let view = look_view(DVec3::new(0.0, 4.5, 7.0), DVec3::new(0.0, 0.0, -1.5));

    let full = gi_settings(40.0, 48, 2.5);
    let mut amortized = full;
    // 2048 probes at High / 256 per frame → a sweep completes in 8 frames.
    amortized.gi.probe_budget = 256;

    // (1) cold == cold, at a frame count that has NOT yet converged (3 of 8).
    let (cold_a, audit_a) = render_frames_with(&gpu, &scene, &view, amortized, 3);
    let (cold_b, audit_b) = render_frames_with(&gpu, &scene, &view, amortized, 3);
    assert_eq!(cold_a, cold_b, "two cold amortized renders diverged");
    assert_eq!(audit_a.probes_updated, 256);
    assert_eq!(audit_a.probes_updated, audit_b.probes_updated);

    // (2) converged == converged.
    let (conv_a, _) = render_frames_with(&gpu, &scene, &view, amortized, 12);
    let (conv_b, _) = render_frames_with(&gpu, &scene, &view, amortized, 12);
    assert_eq!(conv_a, conv_b, "two converged amortized renders diverged");

    // The sweep really is a transient: an unconverged frame differs from a
    // converged one, or this test would prove nothing about convergence.
    assert_ne!(
        cold_a, conv_a,
        "the amortized sweep had already converged at frame 3 — pick a smaller budget"
    );

    // (3) converged == full update, byte for byte.
    let (full_img, full_audit) = render_frames_with(&gpu, &scene, &view, full, 12);
    assert_eq!(full_audit.probes_updated, 16 * 8 * 16, "full update");
    let (mean, max) = image_diff(&conv_a, &full_img, W, H);
    assert!(
        conv_a == full_img,
        "a converged amortized frame is not the full-update frame \
         (mean {mean}, max {max})"
    );
}

/// **SSR off-path neutrality + the whole P18.4 knob set being inert while GI is
/// off** (deliverables 4b and the off-path discipline the golden suite depends on).
///
/// A non-GI scene rendered with every new knob wound to a non-default value must be
/// **byte-identical** to the same scene at the defaults. This is the property that
/// lets 33 of the 36 pre-P18.4 goldens stay untouched.
#[test]
fn gi_v2_off_path_is_byte_identical() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mut scene = RenderScene {
        grid_enabled: true,
        ..Default::default()
    };
    for (i, x) in [-2.0f64, 0.0, 2.4].into_iter().enumerate() {
        scene.instances.push(MeshInstance {
            metallic: 0.4,
            roughness: 0.25,
            ..MeshInstance::lit(
                DVec3::new(x, 0.5, 0.0),
                Quat::from_rotation_y(0.4),
                Vec3::splat(1.0),
                [0.7, 0.5, 0.4, 1.0],
                i as u32 + 1,
            )
        });
    }
    scene.mark_dirty();
    let view = overlook_view();

    let base = render_with(&gpu, &scene, &view, RenderSettings::default());
    let mut fiddled = RenderSettings::default();
    fiddled.gi.specular = false;
    fiddled.gi.probe_budget = 37;
    fiddled.gi.instance_budget = 3;
    fiddled.gi.quality = GiQuality::Low;
    fiddled.shadows.cascade_blend = 0.5;
    assert!(!fiddled.gi.enabled && !fiddled.shadows.enabled);
    let same = render_with(&gpu, &scene, &view, fiddled);
    assert_eq!(
        base, same,
        "a GI-off scene moved when the P18.4 knobs changed — the off path is not neutral"
    );

    // **The SSR knobs left `GiSettings` in wave VIS1a**, so their off-path claim
    // is now about their own block: every tuning field moved, `enabled` left
    // alone, and not a pixel with it. Measured over TWO frames, because a
    // one-frame render has no colour history and would satisfy this arm even if
    // the knobs did reach the shading.
    let two = render_frames_with(&gpu, &scene, &view, RenderSettings::default(), 2).0;
    let mut tuned = RenderSettings::default();
    tuned.ssr.max_distance = 64.0;
    tuned.ssr.thickness = 0.9;
    tuned.ssr.quality = inf_render::SsrQuality::High;
    tuned.ssr.intensity = 0.25;
    tuned.ssr.roughness_cutoff = 1.0;
    assert!(!tuned.ssr.enabled);
    let tuned_two = render_frames_with(&gpu, &scene, &view, tuned, 2).0;
    assert_eq!(two, tuned_two, "SSR tuning moved pixels while SSR is off");

    // ...and with GI ON but SSR off, the same knobs are still inert.
    let gi_on = gi_settings(40.0, 32, 1.0);
    let a = render_frames_with(&gpu, &scene, &view, gi_on, 2).0;
    let mut gi_tuned = gi_on;
    gi_tuned.ssr.max_distance = 64.0;
    gi_tuned.ssr.thickness = 0.9;
    let b = render_frames_with(&gpu, &scene, &view, gi_tuned, 2).0;
    assert_eq!(a, b, "SSR tuning moved pixels while SSR is off, with GI on");
}

/// **SSR NO LONGER NEEDS PROBES** (wave VIS1a) — the decoupling, measured.
///
/// Until this wave SSR was a field on `GiSettings` and did one thing: move the GI
/// probe fetch point. With GI off there were no probes, so the feature was
/// unreachable — which is why `gi_v2_off_path_is_byte_identical` could assert that
/// SSR requested without GI produced no depth prepass and changed nothing.
///
/// It has its own colour source now. This is the same scene with **GI off**, and
/// the reflection has to arrive anyway.
#[test]
fn ssr_reflects_without_gi() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mut scene = RenderScene {
        grid_enabled: false,
        ..Default::default()
    };
    // A mirror-smooth floor and a scarlet block standing on it: the block's
    // reflection is the only red the floor can acquire.
    scene.instances.push(MeshInstance {
        metallic: 1.0,
        roughness: 0.05,
        ..MeshInstance::lit(
            DVec3::new(0.0, -0.25, 0.0),
            Quat::IDENTITY,
            Vec3::new(14.0, 0.5, 14.0),
            [0.55, 0.55, 0.58, 1.0],
            1,
        )
    });
    scene.instances.push(MeshInstance::lit(
        DVec3::new(0.0, 1.0, -1.5),
        Quat::IDENTITY,
        Vec3::new(2.0, 2.0, 2.0),
        [0.95, 0.05, 0.05, 1.0],
        2,
    ));
    scene.lights.push(RenderLight {
        kind: LightKind::Directional,
        color: [1.0, 0.98, 0.95],
        intensity: 2.0,
        direction: Vec3::new(0.2, 0.9, 0.35).normalize(),
        ..RenderLight::default()
    });
    scene.mark_dirty();
    let view = look_view(DVec3::new(0.0, 2.2, 6.5), DVec3::new(0.0, 0.2, -1.0));

    let mut on = RenderSettings::default();
    on.ssr.enabled = true;
    assert!(!on.gi.enabled, "the whole point: no probes");
    assert!(
        on.needs_depth_prepass(),
        "SSR must force the prepass it marches against"
    );

    let off_img = render_frames_with(&gpu, &scene, &view, RenderSettings::default(), 3).0;
    let on_img = render_frames_with(&gpu, &scene, &view, on, 3).0;
    let redness = |img: &[u8]| -> f64 {
        img.chunks(4)
            .map(|p| p[0] as f64 - 0.5 * (p[1] as f64 + p[2] as f64))
            .sum::<f64>()
            / (W * H) as f64
    };
    let (r_off, r_on) = (redness(&off_img), redness(&on_img));
    eprintln!("ssr without gi: redness {r_off:.3} -> {r_on:.3}");
    assert!(
        r_on > r_off + 0.5,
        "the block did not appear in the mirror floor with GI off (redness \
         {r_off:.3} -> {r_on:.3}) — SSR is still riding the probe field"
    );

    // A rough floor must NOT reflect: the roughness cutoff is the cost knob, and
    // a knob that changes nothing is not one.
    let mut rough_scene = scene.clone();
    rough_scene.instances[0].roughness = 0.95;
    rough_scene.mark_dirty();
    let rough_off = render_frames_with(&gpu, &rough_scene, &view, RenderSettings::default(), 3).0;
    let rough_on = render_frames_with(&gpu, &rough_scene, &view, on, 3).0;
    let rough_delta = redness(&rough_on) - redness(&rough_off);
    eprintln!(
        "ssr roughness cutoff: rough floor moved {rough_delta:.3} against {:.3} smooth",
        r_on - r_off
    );
    assert!(
        rough_delta < (r_on - r_off) * 0.25,
        "a floor at roughness 0.95 reflected {rough_delta:.3} against the smooth \
         floor's {:.3} — the roughness cutoff is not holding the march off",
        r_on - r_off
    );
}

/// **GI resources joined the `ResourceKey`** (P18.4 deliverable 7) — the GPU-side
/// twin of `passes::gen_cache_tests::pointer_identity_changes_only_when_the_key_does`.
///
/// A [`GiQuality`] change recreates the voxel + SH buffers. If `EnvBinding` did not
/// key on the GI generation, the lit passes would keep sampling the **previous**
/// tier's SH buffer while the GI node wrote the new one — no validation error, no
/// black frame (the old buffer is larger and stays alive), just a Low-quality frame
/// that is byte-identical to the High one. Which is exactly what this asserts is not
/// the case.
#[test]
fn gi_quality_switch_rebuilds_the_env_bind() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mut scene = RenderScene {
        grid_enabled: false,
        ..Default::default()
    };
    scene.instances.push(MeshInstance::lit(
        DVec3::new(0.0, -0.25, 0.5),
        Quat::IDENTITY,
        Vec3::new(12.0, 0.5, 11.0),
        [0.90, 0.90, 0.90, 1.0],
        1,
    ));
    scene.instances.push(MeshInstance::lit(
        DVec3::new(0.0, 1.5, -4.0),
        Quat::IDENTITY,
        Vec3::new(11.0, 3.0, 0.5),
        [0.90, 0.05, 0.05, 1.0],
        2,
    ));
    scene.lights.push(RenderLight {
        kind: LightKind::Directional,
        color: [1.0, 0.98, 0.95],
        intensity: 2.0,
        direction: Vec3::new(0.0, 0.5, 1.0).normalize(),
        ..RenderLight::default()
    });
    scene.mark_dirty();
    let view = look_view(DVec3::new(0.0, 4.5, 7.0), DVec3::new(0.0, 0.0, -1.5));

    let target = HeadlessTarget::new(&gpu, W, H);
    let mut renderer = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
    let mut settings = gi_settings(40.0, 48, 2.5);
    let mut frames = Vec::new();
    for q in [
        GiQuality::High,
        GiQuality::Low,
        GiQuality::Medium,
        GiQuality::High,
    ] {
        settings.gi.quality = q;
        renderer.set_settings(settings);
        renderer.render(&gpu, &scene, &view, &target.view, (W, H));
        frames.push(target.read_rgba(&gpu).expect("readback"));
    }
    assert_ne!(
        frames[1], frames[0],
        "Low and High GI rendered identically — the env bind group is still \
         holding the previous quality's SH buffer"
    );
    assert_ne!(
        frames[2], frames[1],
        "Medium and Low GI rendered identically"
    );
    assert_eq!(
        frames[3], frames[0],
        "returning to High did not reproduce the High frame"
    );
}

/// **The GI sky comes from the P17.2 atmosphere** (P18.4 deliverable 2) — closing
/// the tracked P17 deferral: the probes' ray-miss term was two authored gradient
/// constants, so a heavy overcast dimmed the sun on the ground but not the bounce,
/// and dawn and noon bounced the same colour.
///
/// Measured on a NEAR receiver (~3 m), where aerial perspective and height fog are
/// numerically nothing, so what moves the floor is the probe field.
#[test]
fn gi_sky_radiance_comes_from_the_atmosphere() {
    let Some(gpu) = gpu_or_skip() else { return };
    let noon = 43_200.0;
    let (mut scene, _) = tod_scene(noon);
    // A near white floor + a small box, with NO analytic light: the only thing
    // lighting the floor is the probe field, whose miss term is the sky.
    scene.instances.push(MeshInstance::lit(
        DVec3::new(0.0, -0.25, 0.0),
        Quat::IDENTITY,
        Vec3::new(10.0, 0.5, 10.0),
        [0.9, 0.9, 0.9, 1.0],
        1,
    ));
    scene.lights.push(RenderLight {
        kind: LightKind::Directional,
        color: [0.0; 3],
        intensity: 0.0,
        direction: Vec3::Y,
        ..RenderLight::default()
    });
    scene.mark_dirty();
    let view = look_view(DVec3::new(0.0, 1.6, 3.0), DVec3::new(0.0, 0.0, -1.0));
    let gi_on = gi_settings(24.0, 64, 1.0);

    let floor = (
        (W * 30 / 100)..(W * 70 / 100),
        (H * 65 / 100)..(H * 92 / 100),
    );
    let with_atmos = render_with(&gpu, &scene, &view, gi_on);

    // The identical scene with the atmosphere switched off: the probes fall back
    // to the authored gradient (the pre-P18.4 behaviour).
    let mut gradient = scene.clone();
    gradient.atmosphere.enabled = false;
    gradient.mark_dirty();
    let with_gradient = render_with(&gpu, &gradient, &view, gi_on);

    let lit_sky = region_mean(&with_atmos, floor.0.clone(), floor.1.clone());
    let lit_grad = region_mean(&with_gradient, floor.0.clone(), floor.1.clone());
    eprintln!("gi sky source: atmosphere {lit_sky:.2}, gradient {lit_grad:.2}");
    assert!(
        lit_sky > lit_grad + 2.0,
        "a real noon sky did not brighten the bounce over the dark authored \
         gradient (atmosphere {lit_sky:.2} vs gradient {lit_grad:.2})"
    );

    // ...and the time of day propagates: a dusk sky bounces a different colour.
    let dusk = 68_400.0; // 19:00 UTC
    let (mut dusk_scene, _) = tod_scene(dusk);
    dusk_scene.instances = scene.instances.clone();
    dusk_scene.lights = scene.lights.clone();
    dusk_scene.mark_dirty();
    let at_dusk = render_with(&gpu, &dusk_scene, &view, gi_on);
    let blue_noon = region_channel_ratio(&with_atmos, floor.0.clone(), floor.1.clone(), 2, 0);
    let blue_dusk = region_channel_ratio(&at_dusk, floor.0.clone(), floor.1.clone(), 2, 0);
    eprintln!("gi sky TOD: noon blue/red {blue_noon:.3}, dusk {blue_dusk:.3}");
    assert!(
        (blue_noon - blue_dusk).abs() > 0.02,
        "the bounce colour did not track the time of day \
         (noon {blue_noon:.3}, dusk {blue_dusk:.3})"
    );

    // ...and at MIDNIGHT the bounce is not sunset-coloured (wave GTA1).
    //
    // The probe march's miss term is `atmos_sample_skyview` — the same LUT the
    // sky pass draws — so whatever that table says at 23:30 is what the floor
    // bounces. What it said before the wave is measured, on the horizon band, by
    // `golden_sky_night_horizon`: `[0.1174, 0.0281, 0.0318]`, a **blue/red of
    // 0.27**. The floor asserted here is 0.6, so any of that band reaching the
    // bounce fails this arm; it reads 1.000 (colourless) now.
    //
    // A RATIO rather than a level, because the honest post-fix answer is "there
    // is almost no bounce at midnight" and a level assertion would be measuring
    // the tonemap's floor.
    let (mut night_scene, night_bodies) = tod_scene(84_600.0); // 23:30 UTC
    assert!(night_bodies.sun.y < -0.2, "23:30 should be deep night");
    night_scene.instances = scene.instances.clone();
    night_scene.lights = scene.lights.clone();
    night_scene.mark_dirty();
    let at_night = render_with(&gpu, &night_scene, &view, gi_on);
    let blue_night = region_channel_ratio(&at_night, floor.0.clone(), floor.1.clone(), 2, 0);
    eprintln!("gi sky TOD: midnight blue/red {blue_night:.3}");
    assert!(
        blue_night > 0.6,
        "the midnight bounce is red-dominated (blue/red {blue_night:.3}) — the \
         probe march is reading the transmittance LUT's horizon-tangent texel"
    );
}

// ── P17.2 physical atmosphere ────────────────────────────────────────────────
//
// The time-of-day sweep. These are the FIRST goldens in the suite that carry an
// atmosphere at all — every scene above renders with `AtmosphereParams::default()`
// (disabled), which is what keeps all 23 pre-P17.2 goldens byte-identical.
//
// The scenes are built from `inf_math::solar` exactly as both scene projectors
// build them, at `SkyAtmosphere`'s defaults, so what these images show is what a
// new level actually looks like — not a hand-tuned demo of the shader.

/// The default sky authority a new level gets: day 172 (June solstice), 48.9° N,
/// prime meridian — `TimeOfDay::default()`'s place, at `seconds` UTC.
fn tod_scene(seconds: f64) -> (RenderScene, inf_math::solar::SkyBodies) {
    let bodies = inf_math::solar::bodies(&inf_math::solar::SolarInput {
        seconds,
        day_of_year: 172,
        latitude_deg: 48.9,
        longitude_deg: 0.0,
    });
    // The `SkyAtmosphere::default()` values, mapped the way `project_sky` maps
    // them in both hosts.
    let scene = RenderScene {
        sun: SunParams {
            direction: bodies.sun.as_vec3(),
            color: [1.0, 0.98, 0.95],
            intensity: 3.0,
            moon_direction: bodies.moon.as_vec3(),
            moon_color: [0.62, 0.72, 1.0],
            moon_intensity: 0.15,
            moon_phase: bodies.moon_phase as f32,
        },
        atmosphere: AtmosphereParams {
            enabled: true,
            moon_phase: bodies.moon_phase as f32,
            ..AtmosphereParams::default()
        },
        ..Default::default()
    };
    (scene, bodies)
}

/// A ground-level camera looking along the horizontal azimuth of `toward`,
/// pitched up by `pitch_deg` so the horizon sits low in frame. Aiming at a body's
/// azimuth (rather than at a fixed compass point) keeps the disc in shot whatever
/// the date and latitude, so these goldens do not silently become pictures of
/// empty sky if the solar model is ever refined.
fn horizon_view(toward: DVec3, pitch_deg: f64) -> RenderView {
    let flat = DVec3::new(toward.x, 0.0, toward.z);
    let flat = if flat.length_squared() > 1e-9 {
        flat.normalize()
    } else {
        DVec3::X
    };
    let p = pitch_deg.to_radians();
    let forward = (flat * p.cos() + DVec3::Y * p.sin()).normalize();
    RenderView {
        origin: FloatingOrigin::new(DVec3::ZERO),
        eye_world: DVec3::new(0.0, 2.0, 0.0),
        forward: forward.as_vec3(),
        up: Vec3::Y,
        fov_y: 60f32.to_radians(),
        near: 0.05,
        width: W,
        height: H,
        ortho: None,
    }
}

/// Mean sRGB-encoded RGB of a screen rectangle (0..1 per channel). Ratios between
/// two such means are what the structural assertions below compare, which is
/// adapter-robust in a way absolute pixel values are not.
fn mean_rgb(img: &[u8], x0: u32, y0: u32, x1: u32, y1: u32) -> [f32; 3] {
    let mut acc = [0.0f32; 3];
    let mut n = 0.0;
    for y in y0..y1 {
        for x in x0..x1 {
            let p = px(img, x, y);
            for c in 0..3 {
                acc[c] += p[c] as f32 / 255.0;
            }
            n += 1.0;
        }
    }
    [acc[0] / n, acc[1] / n, acc[2] / n]
}

fn luma(c: [f32; 3]) -> f32 {
    c[0] * 0.2126 + c[1] * 0.7152 + c[2] * 0.0722
}

/// The brightest single pixel in a screen rectangle, as a 0..1 mean of channels.
fn brightest(img: &[u8], x0: u32, y0: u32, x1: u32, y1: u32) -> f32 {
    let mut best = 0.0f32;
    for y in y0..y1 {
        for x in x0..x1 {
            let p = px(img, x, y);
            let v = (p[0] as f32 + p[1] as f32 + p[2] as f32) / (3.0 * 255.0);
            best = best.max(v);
        }
    }
    best
}

/// Sky brightness gradient + colour at high noon: deep blue overhead, brighter
/// and less saturated toward the horizon (Rayleigh optical depth grows with the
/// path length). This is the shape a three-colour gradient cannot fake — it falls
/// out of the LUT parameterization, and is wrong the moment that is.
#[test]
fn golden_sky_noon() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (scene, bodies) = tod_scene(43_200.0); // 12:00 UTC
    assert!(bodies.sun.y > 0.85, "12:00 at the solstice should be high");
    let view = horizon_view(bodies.sun, 25.0);
    let img = check_golden(&gpu, "sky_noon", &scene, &view);

    // The camera is pitched +25° with a 60° vertical FOV, so the horizon LINE
    // sits ~75 px below centre (y ≈ 165) and everything below it is the sky
    // pass's ground. Both bands are sampled in sky, above that line.
    let top = mean_rgb(&img, 0, 0, W, H / 8);
    let horizon = mean_rgb(&img, 0, H * 80 / 100, W, H * 90 / 100);
    eprintln!("sky_noon top {top:?} horizon {horizon:?}");
    assert!(top[2] > top[0] + 0.08, "zenith not blue: {top:?}");
    // A real daytime sky, not a dim one.
    assert!(top[2] > 0.35, "zenith too dark for noon: {top:?}");
    assert!(
        luma(horizon) > luma(top),
        "horizon should out-brighten the zenith: {horizon:?} vs {top:?}"
    );
    assert!(
        horizon[0] / horizon[2] > top[0] / top[2] + 0.05,
        "horizon should be less blue than the zenith: {horizon:?} vs {top:?}"
    );
}

/// Dawn: the sun is a few degrees up, so its light has crossed a long slab of
/// air. The band around it must be markedly redder than the zenith — the single
/// assertion that catches a swapped Rayleigh triple, and the GPU sibling of the
/// CPU `sunset_is_redder_than_noon` unit test.
#[test]
fn golden_sky_dawn() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (scene, bodies) = tod_scene(16_200.0); // 04:30 UTC
    assert!(
        bodies.sun.y > 0.0 && bodies.sun.y < 0.2,
        "04:30 should be just after sunrise, got y {}",
        bodies.sun.y
    );
    let view = horizon_view(bodies.sun, 6.0);
    let img = check_golden(&gpu, "sky_dawn", &scene, &view);

    // The band just above the horizon line (the camera is pitched +6°, so the
    // horizon sits ~18 px below centre) — not the bottom of the frame, which is
    // the sky pass's ground.
    let top = mean_rgb(&img, 0, 0, W, H / 6);
    let low = mean_rgb(&img, 0, H * 50 / 100, H, H * 58 / 100);
    eprintln!("sky_dawn top {top:?} low {low:?}");
    assert!(
        low[0] / low[2].max(1e-4) > top[0] / top[2].max(1e-4) + 0.15,
        "the horizon band should be redder than the zenith: {low:?} vs {top:?}"
    );
    // The sun disc is in frame and clips to (near) white.
    let peak = brightest(&img, 0, 0, W, H);
    assert!(peak > 0.94, "no sun disc in frame (brightest {peak:.3})");
}

/// Dusk, on the other side of the sky. A separate golden from dawn because the
/// sun's azimuth differs by ~100° and so does the ozone-shaped blue of the
/// opposite horizon — a sweep with only one twilight would not notice a model
/// that made both ends identical.
#[test]
fn golden_sky_dusk() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (scene, bodies) = tod_scene(71_100.0); // 19:45 UTC
    assert!(
        bodies.sun.y > 0.0 && bodies.sun.y < 0.2,
        "19:45 should be just before sunset, got y {}",
        bodies.sun.y
    );
    // Dawn and dusk must not be the same picture.
    let (_, dawn) = tod_scene(16_200.0);
    assert!(
        bodies.sun.x * dawn.sun.x + bodies.sun.z * dawn.sun.z < 0.5,
        "dawn and dusk azimuths are too close to be distinct goldens"
    );
    let view = horizon_view(bodies.sun, 6.0);
    let img = check_golden(&gpu, "sky_dusk", &scene, &view);

    let top = mean_rgb(&img, 0, 0, W, H / 6);
    let low = mean_rgb(&img, 0, H * 50 / 100, H, H * 58 / 100);
    eprintln!("sky_dusk top {top:?} low {low:?}");
    assert!(
        low[0] / low[2].max(1e-4) > top[0] / top[2].max(1e-4) + 0.15,
        "the horizon band should be redder than the zenith: {low:?} vs {top:?}"
    );
}

/// Night: the sky collapses and the procedural starfield appears. The star
/// assertion is a *local-contrast* one (a bright isolated texel against a dark
/// field) rather than a mean, because a mean would also pass for a uniformly-
/// raised black level.
///
/// **What it collapses TO moved in wave GTA1**, and the frame was re-blessed for
/// it: this used to be "the multiple-scattering floor", which sounds like a
/// physical quantity and was in fact the horizon-tangent transmittance texel
/// leaking under the ground (`atmos_horizon_visibility`). With the planet's own
/// shadow applied there is no sun term at midnight at all, so the sky here is
/// black plus stars — the mean went from `[0.0098, 0.0080, 0.0094]` (a red-biased
/// black) to `[0.0067, 0.0067, 0.0067]` (a colourless one). Every scattering
/// model this engine has is sun-driven; moonlight scattering is not modelled, and
/// a black night sky is the honest depiction of that rather than a red one.
/// [`golden_sky_night_horizon`] is the arm that measures the band this frame
/// cannot see.
#[test]
fn golden_sky_night() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (scene, bodies) = tod_scene(84_600.0); // 23:30 UTC
    assert!(bodies.sun.y < -0.2, "23:30 should be deep night");
    // Look away from the sun and well up, where the stars are.
    let view = horizon_view(-bodies.sun, 35.0);
    let img = check_golden(&gpu, "sky_night", &scene, &view);

    let sky = mean_rgb(&img, 0, 0, W, H / 2);
    eprintln!("sky_night mean {sky:?}");
    assert!(sky[2] < 0.30, "night sky is not dark: {sky:?}");
    let field = (sky[0] + sky[1] + sky[2]) / 3.0;
    let peak = brightest(&img, 0, 0, W, H / 2);
    assert!(
        peak > field + 0.12,
        "no starfield contrast (brightest {peak:.3} vs field {field:.3})"
    );
}

/// **The night horizon, facing the sun** (wave GTA1) — the arm
/// [`golden_sky_night`] could not have: it pitches 35° up and looks *away* from
/// the sun, so the one place a below-horizon sun can still be seen is outside its
/// frame entirely, and it bounds only blue.
///
/// This one stands where the defect lived. The transmittance LUT's `u` axis ends
/// at the horizon-tangent ray, so before the fix every below-horizon sun cosine
/// clamped onto that last texel — the reddest one in the table (the longest air
/// path that still misses the planet) — and the sky-view march multiplied it into
/// a band of sunset red sitting on the midnight horizon. Measured on this exact
/// frame at 23:30 UTC, before the fix and after it:
///
/// | | band | R/G | band red ÷ zenith red |
/// |---|---|---|---|
/// | before | `[0.1174, 0.0281, 0.0318]` | **4.18** | **10.7×** |
/// | after  | `[0.0068, 0.0068, 0.0068]` | 1.00 | 1.01× |
///
/// The second half of the test is the one that says which fix this is: at civil
/// twilight the same band measures `[0.398, 0.223, 0.079]` — 59× the midnight
/// red, and warm — because the gate is on each sample's own horizon rather than
/// on the sun's elevation. A global fade would zero both.
///
/// So the assertions are on RED, not blue: an absolute ceiling (a night horizon
/// is dark) and a ratio ceiling (whatever light is left must not be sunset
/// coloured). Both are what the reference footage shows — a deep blue-black
/// horizon with no warm band anywhere on it.
#[test]
fn golden_sky_night_horizon() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (scene, bodies) = tod_scene(84_600.0); // 23:30 UTC
    assert!(bodies.sun.y < -0.2, "23:30 should be deep night");
    // Facing the sun's azimuth with the horizon low in frame — the worst case,
    // and the one a player standing on the island at midnight actually looks at.
    let view = horizon_view(bodies.sun, 6.0);
    let img = check_golden(&gpu, "sky_night_horizon", &scene, &view);

    let top = mean_rgb(&img, 0, 0, W, H / 6);
    let band = mean_rgb(&img, 0, H * 50 / 100, W, H * 58 / 100);
    eprintln!("sky_night_horizon top {top:?} band {band:?}");
    assert!(
        band[0] < 0.05,
        "the midnight horizon is glowing red: {band:?}"
    );
    assert!(
        band[0] / band[1].max(1e-4) < 1.6,
        "the midnight horizon is sunset-coloured: {band:?}"
    );
    assert!(
        band[0] < top[0] + 0.02,
        "the midnight horizon out-reds the sky above it: band {band:?} vs top {top:?}"
    );

    // ── and the arm that says WHICH fix this is ──
    //
    // The cheap way to kill the red band is a gate on the sun's own elevation:
    // fade the whole sun term out as `sun.y` crosses zero. It would pass every
    // assertion above and it would also delete **civil twilight**, because the
    // sky after sunset is lit by air the sun has not set on yet — a parcel 60 km
    // up sees it until it is 7.8° under the ground's horizon.
    //
    // So the gate is per-SAMPLE (each sample's own local horizon,
    // `atmos_horizon_visibility`), and this is the arm that proves it: with the
    // sun a few degrees down the same band must still be lit and still be warm.
    // A global elevation fade renders it black and fails here.
    let mut seconds = 71_100.0;
    let twilight = loop {
        let (s, b) = tod_scene(seconds);
        // −1.7° .. −5.2°: after sunset, before the end of civil twilight.
        if b.sun.y < -0.03 && b.sun.y > -0.09 {
            break (s, b);
        }
        seconds += 60.0;
        assert!(
            seconds < 86_400.0,
            "no civil twilight found on this date/latitude"
        );
    };
    let twi = render(&gpu, &twilight.0, &horizon_view(twilight.1.sun, 6.0));
    let twi_band = mean_rgb(&twi, 0, H * 50 / 100, W, H * 58 / 100);
    eprintln!(
        "sky_night_horizon twilight (sun.y {:.3}) band {twi_band:?}",
        twilight.1.sun.y
    );
    assert!(
        twi_band[0] > band[0] * 4.0,
        "civil twilight is as dark as midnight — the horizon gate is on the sun's \
         elevation rather than each sample's own horizon: {twi_band:?} vs {band:?}"
    );
    assert!(
        twi_band[0] / twi_band[1].max(1e-4) > 1.15,
        "civil twilight lost its warmth: {twi_band:?}"
    );
}

/// The starfield is a pure function of the view direction: two renders of the
/// same night sky must be byte-identical (the hash is integer-only, per the
/// psin/pcos law's spirit — no trig anywhere in it), and a *rotated* camera must
/// see a different patch of sky rather than a field pinned to the screen.
#[test]
fn stars_are_deterministic_and_world_locked() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (scene, bodies) = tod_scene(84_600.0);
    let view = horizon_view(-bodies.sun, 35.0);
    let a = render(&gpu, &scene, &view);
    let b = render(&gpu, &scene, &view);
    assert_eq!(a, b, "the starfield is not deterministic");

    // Yaw the camera 40°: the same screen pixels must now show different sky.
    let f = view.forward;
    let (s, c) = 40f32.to_radians().sin_cos();
    let rotated = RenderView {
        forward: Vec3::new(f.x * c + f.z * s, f.y, -f.x * s + f.z * c).normalize(),
        ..view
    };
    let r = render(&gpu, &scene, &rotated);
    assert_ne!(a, r, "the starfield followed the camera instead of the sky");
}

/// Aerial perspective + height fog on lit geometry, as a **controlled
/// experiment** rather than a pretty picture: two walls with identical albedo,
/// identical orientation and identical screen size, one at 50 m and one at
/// 1500 m — the far one is the near one scaled 30× about the eye, so it projects
/// to the same rectangle mirrored across the frame. Every pixel-level difference
/// between them is therefore the atmosphere and nothing else.
///
/// Also carries the off-path proof: the same scene with the atmosphere disabled
/// is the pre-P17.2 render.
#[test]
fn golden_aerial_fog() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (mut scene, bodies) = tod_scene(43_200.0);
    scene.atmosphere.fog = HeightFog {
        density: 1.5e-3, // ≈ 2 km visibility — a properly foggy morning
        falloff: 0.002,  // 500 m e-folding height
        height: 0.0,
        color: [1.0, 1.0, 1.0],
    };
    // Ground, deliberately DARK: a bright albedo is already near white before any
    // scattering touches it, so the wash toward the sky would be invisible.
    scene.instances.push(MeshInstance::lit(
        DVec3::new(0.0, -0.5, -6000.0),
        Quat::IDENTITY,
        Vec3::new(8000.0, 1.0, 16000.0),
        [0.10, 0.11, 0.12, 1.0],
        1,
    ));
    // The matched pair. `NEAR` is at −14 m across at 50 m out; `FAR` is that
    // exact vector × 30, mirrored in x, so both subtend the same angle.
    const WALL: [f32; 4] = [0.16, 0.17, 0.18, 1.0];
    scene.instances.push(MeshInstance::lit(
        DVec3::new(-14.0, 8.0, -50.0),
        Quat::IDENTITY,
        Vec3::new(16.0, 16.0, 1.0),
        WALL,
        2,
    ));
    scene.instances.push(MeshInstance::lit(
        DVec3::new(420.0, 8.0, -1500.0),
        Quat::IDENTITY,
        Vec3::new(480.0, 480.0, 30.0),
        WALL,
        3,
    ));
    // A few pillars down the centre — not measured, but they are what makes the
    // golden readable as a picture of depth rather than two grey squares.
    for (i, d) in [15.0f64, 45.0, 140.0, 420.0].into_iter().enumerate() {
        scene.instances.push(MeshInstance::lit(
            DVec3::new(0.0, 6.0, -d),
            Quat::IDENTITY,
            Vec3::new(2.0, 12.0, 2.0),
            [0.13, 0.14, 0.15, 1.0],
            i as u32 + 4,
        ));
    }
    scene.lights.push(RenderLight {
        kind: LightKind::Directional,
        color: [1.0, 0.98, 0.95],
        intensity: 3.0,
        direction: bodies.sun.as_vec3(),
        position: DVec3::ZERO,
        range: 0.0,
        ..RenderLight::default()
    });
    scene.mark_dirty();

    // Eye at the walls' centre height looking dead ahead, so the two rectangles
    // land symmetrically about the frame centre.
    let view = RenderView {
        origin: FloatingOrigin::new(DVec3::ZERO),
        eye_world: DVec3::new(0.0, 8.0, 0.0),
        forward: Vec3::NEG_Z,
        up: Vec3::Y,
        fov_y: 45f32.to_radians(),
        near: 0.05,
        width: W,
        height: H,
        ortho: None,
    };
    let img = check_golden(&gpu, "aerial_fog", &scene, &view);

    // Both walls are sampled 20 px above centre — inside each rectangle, and
    // above the horizon line so no ground creeps into either box.
    let near = mean_rgb(&img, 91, 62, 107, 78);
    let far = mean_rgb(&img, 213, 62, 229, 78);
    // The sky is sampled at the SAME screen height as the walls, off to the right
    // of the far one: the in-scattered light a horizontal ray picks up is the
    // horizon's air column, which is markedly whiter than the deep blue overhead.
    let sky = mean_rgb(&img, 276, 62, 316, 78);
    let gap = |a: [f32; 3], b: [f32; 3]| {
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
    };
    eprintln!(
        "aerial_fog near {near:?} far {far:?} sky {sky:?} gap {:.3} / {:.3}",
        gap(near, sky),
        gap(far, sky)
    );
    assert!(
        luma(far) > luma(near) + 0.10,
        "the far wall did not lighten: near {near:?} far {far:?}"
    );
    // ...and it converges ON THE SKY, which is the assertion that actually pins
    // the model. "Gets bluer" would be wrong here and would rightly fail: a hazy
    // noon horizon is whiter than the blue hemispheric ambient, so the far wall
    // gets *less* blue while still becoming more sky-coloured. Distance in RGB
    // says what was meant.
    assert!(
        gap(far, sky) < gap(near, sky) * 0.6,
        "the far wall did not converge on the sky: near {near:?} far {far:?} sky {sky:?}"
    );

    // Off path, in its strongest form: the SAME geometry with no atmosphere is
    // the pre-P17.2 render, and it must be deterministic and different.
    let mut plain = scene.clone();
    plain.atmosphere = AtmosphereParams::default();
    plain.sun = SunParams::default();
    let a = render(&gpu, &plain, &view);
    let b = render(&gpu, &plain, &view);
    assert_eq!(a, b, "the no-atmosphere path must stay deterministic");
    assert_ne!(a, img, "the atmosphere changed nothing about the scene");
    // With no atmosphere the two walls are the SAME colour (only the old fixed
    // distance haze separates them) — which is exactly what P17.2 replaced.
    let plain_near = mean_rgb(&a, 91, 62, 107, 78);
    let plain_far = mean_rgb(&a, 213, 62, 229, 78);
    assert!(
        (luma(plain_far) - luma(plain_near)).abs() < luma(far) - luma(near),
        "the old haze separated the walls more than the atmosphere does: \
         plain {plain_near:?}/{plain_far:?} vs atmos {near:?}/{far:?}"
    );
}

/// The LUT determinism gate: two independent renderers bake the same LUTs from
/// the same inputs, and the texels must match **byte for byte**. This is the
/// atmosphere's version of the double-render gate every golden runs, one level
/// lower — on the intermediate the sky is a lookup into — so a nondeterministic
/// march surfaces here rather than as a flaky pixel three passes downstream.
#[test]
fn atmosphere_luts_are_deterministic() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (scene, bodies) = tod_scene(43_200.0);
    let view = horizon_view(bodies.sun, 20.0);

    let bake = || {
        let target = HeadlessTarget::new(&gpu, W, H);
        let mut renderer = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
        renderer.render(&gpu, &scene, &view, &target.view, (W, H));
        let a = renderer.atmosphere();
        (
            a.read_transmittance(&gpu).expect("transmittance readback"),
            a.read_sky_view(&gpu).expect("sky-view readback"),
        )
    };
    let (t1, s1) = bake();
    let (t2, s2) = bake();
    assert_eq!(t1, t2, "transmittance LUT is not deterministic");
    assert_eq!(s1, s2, "sky-view LUT is not deterministic");
    // ...and not trivially empty (a bake that never dispatched would also match).
    assert!(
        t1.iter().any(|&b| b != 0),
        "transmittance LUT is empty — did the bake dispatch?"
    );
    assert!(
        s1.iter().any(|&b| b != 0),
        "sky-view LUT is empty — did the bake dispatch?"
    );
    let (tw, th) = AtmosphereQuality::High.transmittance_size();
    let (sw, sh) = AtmosphereQuality::High.skyview_size();
    assert_eq!(t1.len(), (tw * th * 8) as usize);
    assert_eq!(s1.len(), (sw * sh * 8) as usize);
}

/// A quality change RESIZES the LUTs — exactly the case the `EnvBinding` cache
/// invariant guards. A stale key does **not** validate-error and does not blank
/// the frame: wgpu keeps the old texture alive as long as a bind group references
/// it, so the pass just silently samples the previous quality's LUT. So the
/// assertions here are frame **differences**, not liveness:
///
/// * `frames[1] != frames[0]` — Low really does produce a different image from
///   High (if it did not, every other assertion would be vacuous);
/// * `frames[3] == frames[0]` — coming back to High reproduces High **exactly**.
///   With a stale bind group the last frame would keep Medium's LUTs and this is
///   the assertion that catches it.
///
/// The cube is placed along the view direction, so the lit pass — the one that
/// binds `EnvBinding` — actually covers pixels; that it does is asserted rather
/// than assumed. The adapter-free half of this gate is
/// `passes::gen_cache_tests::pointer_identity_changes_only_when_the_key_does`.
#[test]
fn atmosphere_quality_switch_rebuilds_the_env_bind() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (mut scene, bodies) = tod_scene(43_200.0);
    let view = horizon_view(bodies.sun, 10.0);
    // Down the view axis, not down −Z: `horizon_view` aims at the sun's azimuth,
    // which at noon is due south (+Z), so a cube at −Z would be behind the camera
    // and the env bind group would never be sampled at all.
    let ahead = DVec3::new(bodies.sun.x, 0.0, bodies.sun.z).normalize();
    // A DISTANT wall filling the middle of the frame, under thick fog. NEAR
    // geometry is not enough: at a few metres the aerial/fog term is ~nothing, so
    // the lit pass barely reads the LUT and a stale `EnvBinding` is invisible
    // (verified by mutation — with a near cube, dropping the atmosphere
    // generation from the key left this test green). At 1500 m under 2 km
    // visibility the wall's colour is mostly in-scattered sky sampled *through
    // the env bind*, so a stale LUT moves it.
    scene.atmosphere.fog = HeightFog {
        density: 1.5e-3,
        falloff: 0.0,
        height: 0.0,
        color: [1.0, 1.0, 1.0],
    };
    scene.instances.push(MeshInstance::lit(
        ahead * 1500.0 + DVec3::new(0.0, 266.0, 0.0),
        Quat::IDENTITY,
        Vec3::new(1400.0, 540.0, 5.0),
        [0.06, 0.06, 0.07, 1.0],
        1,
    ));
    scene.mark_dirty();

    // Prove the cube covers pixels: the same frame without it must differ.
    let mut sky_only = scene.clone();
    sky_only.instances.clear();
    sky_only.mark_dirty();
    let bare = render(&gpu, &sky_only, &view);

    let target = HeadlessTarget::new(&gpu, W, H);
    let mut renderer = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
    let mut settings = RenderSettings::default();
    let mut frames = Vec::new();
    let mut seen = Vec::new();
    for q in [
        AtmosphereQuality::High,
        AtmosphereQuality::Low,
        AtmosphereQuality::Medium,
        AtmosphereQuality::High,
    ] {
        settings.atmosphere.quality = q;
        renderer.set_settings(settings);
        renderer.render(&gpu, &scene, &view, &target.view, (W, H));
        frames.push(target.read_rgba(&gpu).expect("readback"));
        seen.push(renderer.atmosphere().quality);
    }

    let differing = |a: &[u8], b: &[u8]| a.iter().zip(b).filter(|(x, y)| x != y).count();
    // The wall's interior, well inside its screen rectangle: LIT pixels, which is
    // the only place the env bind group is read at all.
    let wall = |img: &[u8]| -> Vec<u8> {
        (70..110)
            .flat_map(|y| (100..220).map(move |x| (x, y)))
            .flat_map(|(x, y)| px(img, x, y))
            .collect()
    };
    let covered = differing(&frames[0], &bare);
    eprintln!(
        "quality switch: wall covers {covered} bytes; whole frame Low-vs-High {}; \
         wall region Low-vs-High {}",
        differing(&frames[1], &frames[0]),
        differing(&wall(&frames[1]), &wall(&frames[0]))
    );
    assert!(
        covered > 20_000,
        "the lit wall covers almost nothing ({covered} bytes) — the env bind group \
         is not being sampled, so this test would pass with a stale key"
    );

    // The env-bind assertion: the LIT region must change with the LUT. A stale
    // `EnvBinding` keeps High's views for every frame, so this region would come
    // back byte-identical while the (separately keyed) sky around it changed —
    // which is exactly what a whole-frame comparison would fail to notice.
    assert_ne!(
        wall(&frames[1]),
        wall(&frames[0]),
        "the lit wall is byte-identical at Low and High — the env bind group is \
         still holding the previous quality's LUT views"
    );
    // And the whole frame differs too (the sky path, separately keyed).
    assert_ne!(
        frames[1], frames[0],
        "Low and High rendered identically — the LUT resize had no visible effect"
    );
    // Round trip: back at High, the frame must reproduce the first High frame
    // byte for byte.
    assert_eq!(
        frames[3], frames[0],
        "returning to High did not reproduce the High frame"
    );

    assert_eq!(
        seen,
        vec![
            AtmosphereQuality::High,
            AtmosphereQuality::Low,
            AtmosphereQuality::Medium,
            AtmosphereQuality::High,
        ],
        "the resources did not follow the settings"
    );
}

/// **The editor default look** (P17.2's "gorgeous default sky" deliverable): the
/// `TimeOfDay::default()` clock — 10:00 UTC on day 172 at 48.9° N — over a
/// primitive scene of the same shape a new level is built with.
///
/// This is deliberately *not* a mirror of `inf_editor_core::scene::demo::build`,
/// which lives in another ring and would silently drift from a copy here. It is
/// the same **sky** over representative geometry: what this golden pins is the
/// default clock's look, which is the thing a change to the defaults would move.
///
/// Why 10:00 rather than noon: the sun lands ≈ 55° up, which keeps a real
/// direction — long enough shadows and a clear light/shade split to read shape —
/// while still giving a saturated blue zenith. A noon sun lights everything from
/// straight overhead and flattens exactly the geometry a default scene exists to
/// show off.
#[test]
fn golden_editor_default() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (mut scene, bodies) = tod_scene(36_000.0); // TimeOfDay::default(): 10:00 UTC
    let elevation = bodies.sun.y.asin().to_degrees();
    assert!(
        (50.0..60.0).contains(&elevation),
        "the default clock should put the sun ~55° up, got {elevation:.1}°"
    );
    scene.grid_enabled = true;
    // P17.3: a new level now boots with CLOUDS. This is the one golden P17.3
    // re-blessed, and this line is the reason — `inf_editor_core::scene::demo`
    // sets `clouds_enabled = true` on the default scene's `SkyAtmosphere` while
    // the *component* default stays false (which is what every existing v12 level
    // lifts to). Everything else stays at `CloudParams::default()`, so what this
    // pictures is the documented defaults rather than a private tuning.
    scene.atmosphere.clouds = CloudParams {
        enabled: true,
        ..CloudParams::default()
    };

    // Ground plane + the three props, at the default scene's placements/colours.
    let mut push = |mesh, t: DVec3, s: Vec3, c: [f32; 3], id| {
        scene.instances.push(MeshInstance {
            vt: Default::default(),
            translation: t,
            rotation: Quat::IDENTITY,
            scale: s,
            color: [c[0], c[1], c[2], 1.0],
            metallic: 0.0,
            roughness: 0.6,
            emissive: [0.0; 3],
            id,
            mesh,
            blend: 0,
            cutoff: 0.5,
        })
    };
    push(
        PrimMesh::Plane,
        DVec3::ZERO,
        Vec3::new(20.0, 1.0, 20.0),
        [0.30, 0.32, 0.35],
        1,
    );
    push(
        PrimMesh::Cube,
        DVec3::new(-2.0, 0.5, 0.0),
        Vec3::ONE,
        [0.80, 0.25, 0.22],
        2,
    );
    push(
        PrimMesh::Sphere,
        DVec3::new(0.0, 0.6, -1.5),
        Vec3::ONE,
        [0.25, 0.55, 0.85],
        3,
    );
    push(
        PrimMesh::Cylinder,
        DVec3::new(2.0, 0.75, 0.5),
        Vec3::ONE,
        [0.30, 0.70, 0.35],
        4,
    );
    // The sky's own key light, exactly as `project_sky` pushes it, plus the
    // default scene's point fill.
    scene.lights.push(RenderLight {
        kind: LightKind::Directional,
        color: [1.0, 0.98, 0.95],
        intensity: 3.0,
        direction: bodies.sun.as_vec3(),
        position: DVec3::ZERO,
        range: 0.0,
        cast_shadows: true,
        ..RenderLight::default()
    });
    // The default scene's point fill, at `Light::default()`'s intensity 1.0 —
    // not a number invented here.
    scene.lights.push(RenderLight {
        kind: LightKind::Point,
        color: [1.0, 1.0, 1.0],
        intensity: 1.0,
        direction: Vec3::Y,
        position: DVec3::new(4.0, 3.0, 4.0),
        range: 0.0,
        ..RenderLight::default()
    });
    scene.mark_dirty();

    // A near-horizontal camera rather than the suite's usual look-down overlook:
    // the point of this golden is the SKY, and an overlook shows almost none.
    let view = look_view(DVec3::new(7.0, 2.4, 9.5), DVec3::new(0.0, 1.6, 0.0));
    let img = check_golden(&gpu, "editor_default", &scene, &view);

    // The sky above the horizon is a believable daytime blue: bright, clearly
    // blue-dominant, and NOT the near-black editor gradient this replaced (whose
    // zenith was linear 0.038 — about 0.07 sRGB).
    let sky = mean_rgb(&img, 0, 0, W, H / 6);
    eprintln!("editor_default sky {sky:?}");
    assert!(sky[2] > 0.45, "the default sky is not bright: {sky:?}");
    assert!(
        sky[2] > sky[0] + 0.08,
        "the default sky is not blue: {sky:?}"
    );
    assert!(sky[2] < 0.98, "the default sky is blown out: {sky:?}");
    // The props are lit and readable against it.
    assert!(
        luma(mean_rgb(
            &img,
            W / 2 - 60,
            H * 70 / 100,
            W / 2 + 60,
            H * 92 / 100
        )) > 0.15,
        "the ground and props are too dark under the default sun"
    );

    // P17.3: and the default level really does have clouds in it. Without this,
    // the golden's one re-bless would be justified by a code comment alone — this
    // is what makes "the new-level look changed" a measured claim. (Declared
    // after the cloud helpers below, which is fine: Rust does not care, and
    // keeping this assertion beside the rest of `editor_default` does.)
    let mut cloudless = scene.clone();
    cloudless.atmosphere.clouds = CloudParams::default();
    cloudless.mark_dirty();
    let bare = render(&gpu, &cloudless, &view);
    let covered = changed_fraction(&img, &bare, H / 2);
    eprintln!("editor_default cloud cover {covered:.3}");
    assert!(
        covered > 0.05,
        "the default level's sky has no clouds in it ({covered:.3}) — the \
         `editor_default` re-bless would then have no reason"
    );
}

// ── P17.3 volumetric clouds ──────────────────────────────────────────────────
//
// The cloud goldens extend the P17.2 time-of-day sweep rather than inventing a
// scene: `tod_scene` builds the sun from `inf_math::solar` at the component
// defaults, and each test flips ONLY the cloud fields. What the images show is
// therefore what a level actually gets when it ticks the Clouds box, not a
// hand-tuned demo of the raymarch.
//
// None of the 29 pre-P17.3 goldens moves: clouds default to disabled, the bake
// and raymarch nodes dispatch nothing, and the lit shaders' cloud-shadow multiply
// sits inside a guarded branch. That was verified the P17.2 way — running the
// whole suite under `INF_BLESS_GOLDENS=1` and confirming `git status` reports
// zero changed PNGs — not merely asserted.

/// A cloud layer over the P17.2 sky. `coverage`/`cloud_type` are the two knobs a
/// level actually reaches for; everything else stays at `CloudParams::default()`.
fn cloud_scene(
    seconds: f64,
    coverage: f32,
    cloud_type: f32,
) -> (RenderScene, inf_math::solar::SkyBodies) {
    let (mut scene, bodies) = tod_scene(seconds);
    scene.atmosphere.clouds = CloudParams {
        enabled: true,
        coverage,
        cloud_type,
        ..CloudParams::default()
    };
    (scene, bodies)
}

/// The same scene with clouds switched back off — the off-path control every
/// cloud golden compares against, so "the feature drew something" is measured
/// rather than assumed.
fn without_clouds(scene: &RenderScene) -> RenderScene {
    let mut s = scene.clone();
    s.atmosphere.clouds = CloudParams::default();
    s.mark_dirty();
    s
}

/// Fraction of the given screen band that a cloud layer **perceptibly** changed.
///
/// Perceptibly, not at all: with premultiplied compositing plus aerial
/// perspective, an alpha of a thousandth still moves the low bit of a pixel, so
/// an exact-inequality count reports 100 % coverage for any sky that has a wisp
/// anywhere. The threshold (8/255 summed over RGB, ~1 % of range) is what makes
/// "covered" mean what an author means by it.
fn changed_fraction(a: &[u8], b: &[u8], rows: u32) -> f32 {
    let mut n = 0u32;
    for y in 0..rows {
        for x in 0..W {
            let p = px(a, x, y);
            let q = px(b, x, y);
            let d: i32 = (0..3).map(|c| (p[c] as i32 - q[c] as i32).abs()).sum();
            if d > 8 {
                n += 1;
            }
        }
    }
    n as f32 / (W * rows) as f32
}

/// Standard deviation of luma over the top `rows` of the frame — the measure that
/// tells broken cloud apart from a flat wash.
fn luma_spread(img: &[u8], rows: u32) -> f32 {
    let n = (W * rows) as f32;
    let mut sum = 0.0;
    let mut sum2 = 0.0;
    for y in 0..rows {
        for x in 0..W {
            let p = px(img, x, y);
            let l = luma([
                p[0] as f32 / 255.0,
                p[1] as f32 / 255.0,
                p[2] as f32 / 255.0,
            ]);
            sum += l;
            sum2 += l * l;
        }
    }
    (sum2 / n - (sum / n) * (sum / n)).max(0.0).sqrt()
}

/// **Overcast noon.** Solid coverage of a low stratus sheet: the sky must be
/// mostly cloud, and the cloud must be *bright* — an overcast sky is the single
/// hardest case for a single-scattering march, which renders it as soot. The
/// assertion on absolute luminance is the one that catches that.
#[test]
fn golden_clouds_overcast() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (mut scene, bodies) = cloud_scene(43_200.0, 1.0, 0.15);
    // Stratus geometry: a thinner sheet, lower down.
    scene.atmosphere.clouds.bottom = 900.0;
    scene.atmosphere.clouds.top = 2200.0;
    scene.mark_dirty();
    let view = horizon_view(bodies.sun, 30.0);
    let img = check_golden(&gpu, "clouds_overcast", &scene, &view);

    let bare = render(&gpu, &without_clouds(&scene), &view);
    let sky = mean_rgb(&img, 0, 0, W, H / 2);
    let clear = mean_rgb(&bare, 0, 0, W, H / 2);
    let covered = changed_fraction(&img, &bare, H / 2);
    eprintln!("clouds_overcast sky {sky:?} vs clear {clear:?}; covered {covered:.3}");

    // An overcast sky is bright, and grey rather than blue: the droplets' albedo
    // is neutral, so the sky's blue excess must collapse.
    assert!(luma(sky) > 0.30, "overcast sky is soot: {sky:?}");
    assert!(
        sky[2] - sky[0] < (clear[2] - clear[0]) * 0.7,
        "overcast did not de-blue the sky: {sky:?} vs {clear:?}"
    );
    assert!(
        covered > 0.9,
        "coverage 1.0 left {:.1}% of the sky untouched",
        100.0 * (1.0 - covered)
    );
}

/// **Scattered cumulus at noon** — the default look, and the one that proves the
/// field has *structure*: broken cloud with real gaps, not a uniform haze. The
/// assertion is on the spread of luma across the sky band, which a flat wash
/// cannot pass, plus a floor on how much clear sky survives.
#[test]
fn golden_clouds_scattered() {
    let Some(gpu) = gpu_or_skip() else { return };
    // The *component default* coverage, so this golden pictures what a level
    // actually gets rather than a tuned demo — the P17.2 doctrine.
    let (scene, bodies) = cloud_scene(43_200.0, CloudParams::default().coverage, 0.9);
    let view = horizon_view(bodies.sun, 28.0);
    let img = check_golden(&gpu, "clouds_scattered", &scene, &view);

    let bare = render(&gpu, &without_clouds(&scene), &view);
    let rows = H / 3;
    let cloudy = luma_spread(&img, rows);
    let clear = luma_spread(&bare, rows);
    let covered = changed_fraction(&img, &bare, rows);
    eprintln!("clouds_scattered luma sd {cloudy:.4} vs clear {clear:.4}; covered {covered:.3}");
    assert!(
        cloudy > clear * 1.8,
        "scattered clouds have no structure: sd {cloudy:.4} vs clear sky {clear:.4}"
    );
    // Both ends, which is the whole meaning of "scattered": the clouds are really
    // there, and so are the gaps.
    assert!(
        covered > 0.2,
        "the default coverage drew almost nothing ({covered:.3})"
    );
    assert!(
        covered < 0.9,
        "no clear sky survives at the default coverage ({covered:.3}) — that is \
         overcast, and the default is meant to be broken cumulus"
    );
}

/// **Dusk clouds.** A cloud's lit top is lit by *the sun's transmittance through
/// the atmosphere*, so at 19:45 it must be measurably warmer than the same cloud
/// at noon. This is the single assertion that would catch clouds being lit by a
/// hard-coded white sun instead of by the transmittance LUT.
#[test]
fn golden_clouds_dusk() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (scene, bodies) = cloud_scene(71_100.0, 0.6, 0.85);
    assert!(
        bodies.sun.y > 0.0 && bodies.sun.y < 0.2,
        "19:45 should put the sun just above the horizon"
    );
    let view = horizon_view(bodies.sun, 14.0);
    let img = check_golden(&gpu, "clouds_dusk", &scene, &view);

    // The same clouds under a noon sun, from the same relative viewpoint.
    let (noon_scene, noon_bodies) = cloud_scene(43_200.0, 0.6, 0.85);
    let noon = render(&gpu, &noon_scene, &horizon_view(noon_bodies.sun, 14.0));

    let dusk_rgb = mean_rgb(&img, 0, 0, W, H / 2);
    let noon_rgb = mean_rgb(&noon, 0, 0, W, H / 2);
    let warm = |c: [f32; 3]| c[0] / c[2].max(1e-4);
    eprintln!(
        "clouds_dusk {dusk_rgb:?} (r/b {:.3}) vs noon {noon_rgb:?} (r/b {:.3})",
        warm(dusk_rgb),
        warm(noon_rgb)
    );
    assert!(
        warm(dusk_rgb) > warm(noon_rgb) + 0.15,
        "dusk clouds are not warmer than noon clouds: {dusk_rgb:?} vs {noon_rgb:?}"
    );
}

/// **Night clouds.** Stars stay visible through the gaps while being occluded
/// where a cloud is. The cloud pass composites over the sky pass, so this is the
/// test that the premultiplied alpha is doing its job rather than the clouds
/// being drawn behind everything.
#[test]
fn golden_clouds_night() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (scene, bodies) = cloud_scene(84_600.0, 0.45, 0.8);
    assert!(bodies.sun.y < -0.2, "23:30 should be deep night");
    let view = horizon_view(-bodies.sun, 35.0);
    let img = check_golden(&gpu, "clouds_night", &scene, &view);

    let bare = render(&gpu, &without_clouds(&scene), &view);

    // Stars survive the gaps. Asserted by REMOVING them: a contrast-against-the-
    // mean test would also be satisfied by a bright cloud edge, whereas the only
    // thing that can make the peak drop when `star_intensity` goes to zero is a
    // star that was visible. This is the same reasoning `sky_night` uses, taken
    // one step further because there is now something else bright in frame.
    let mut starless = scene.clone();
    starless.atmosphere.star_intensity = 0.0;
    starless.mark_dirty();
    let no_stars = render(&gpu, &starless, &view);

    let m = mean_rgb(&img, 0, 0, W, H / 2);
    let field = (m[0] + m[1] + m[2]) / 3.0;
    let peak = brightest(&img, 0, 0, W, H / 2);
    let peak_starless = brightest(&no_stars, 0, 0, W, H / 2);
    eprintln!("clouds_night field {field:.3} peak {peak:.3} (starless {peak_starless:.3})");
    assert!(
        peak > peak_starless + 0.04,
        "no stars survive the gaps: the brightest pixel barely moves when the \
         starfield is switched off ({peak:.3} vs {peak_starless:.3})"
    );
    assert!(peak > field + 0.05, "no star contrast at all");
    assert_ne!(img, bare, "night clouds drew nothing");

    // ...and the clouds really do occlude: somewhere the frame got DARKER than
    // the starfield alone, which only happens where a dim night cloud covered a
    // star.
    let mut occluded = 0u32;
    for y in 0..H / 2 {
        for x in 0..W {
            let a = px(&img, x, y);
            let b = px(&bare, x, y);
            let sa = a[0] as i16 + a[1] as i16 + a[2] as i16;
            let sb = b[0] as i16 + b[1] as i16 + b[2] as i16;
            if sa < sb - 12 {
                occluded += 1;
            }
        }
    }
    eprintln!("clouds_night occluded {occluded} px");
    assert!(
        occluded > 0,
        "clouds never occluded a single star — is the alpha compositing backwards?"
    );
}

/// The **depth** contract: geometry in front of the cloud layer occludes it.
/// Without it the raymarch would hang in front of the world, which is exactly
/// what drawing clouds inside the sky pass would have produced (that pass clears
/// depth, so there is nothing to test against yet).
#[test]
fn clouds_are_occluded_by_geometry() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (mut scene, bodies) = cloud_scene(43_200.0, 1.0, 0.4);
    let view = horizon_view(bodies.sun, 20.0);
    let sky_only = render(&gpu, &scene, &view);

    // A wall a few metres in front of the camera, filling the middle of the
    // frame. Everything behind it — including a kilometre of overcast — must go.
    let ahead = DVec3::new(bodies.sun.x, 0.36, bodies.sun.z).normalize();
    scene.instances.push(MeshInstance::lit(
        ahead * 12.0,
        Quat::IDENTITY,
        Vec3::new(60.0, 60.0, 0.5),
        [0.5, 0.1, 0.1, 1.0],
        1,
    ));
    scene.mark_dirty();
    let walled = render(&gpu, &scene, &view);
    let walled_clear = render(&gpu, &without_clouds(&scene), &view);

    let centre = |img: &[u8]| -> Vec<u8> {
        (H * 42 / 100..H * 58 / 100)
            .flat_map(|y| (W * 42 / 100..W * 58 / 100).map(move |x| (x, y)))
            .flat_map(|(x, y)| px(img, x, y))
            .collect()
    };
    assert_ne!(
        centre(&sky_only),
        centre(&walled),
        "the wall covers nothing in the sampled region — this test would pass vacuously"
    );
    assert_eq!(
        centre(&walled),
        centre(&walled_clear),
        "clouds bled through solid geometry — the depth test is not rejecting them"
    );
}

/// The **intersecting**-geometry contract, which the entry-depth test alone
/// cannot satisfy: a summit that pokes *into* the cloud deck must not be veiled
/// by the cloud physically behind it.
///
/// This is the case `clouds_are_occluded_by_geometry` does not reach. There the
/// wall is entirely in front of the slab, so its fragments' depth beats the
/// slab's entry plane and the hardware test rejects the cloud outright. A mesa
/// whose top is 500 m inside a 1.5–4 km deck sits *beyond* that entry plane, so
/// it passes a `Greater` test — and without the `t_far` clamp the shader would
/// composite the whole marched span over it, including the five kilometres of
/// cloud behind the mountain. On an 8 km terrain that is not an exotic case, it
/// is Tuesday.
///
/// Measured as a **reduction**, not as an absence, because the correct answer is
/// not zero: there really is ~1 km of cloud between the eye and the mesa's face,
/// and it should still be visible. The reference for "the whole veil" is the same
/// scene with the mesa removed, so the two numbers are produced by the same
/// shader on the same pixels and the comparison needs no second build.
///
/// Mutation-verified: disabling the `t_far` clamp in `cloud.wgsl` moves the
/// measured alpha over the mesa from **0.275 to 0.588** and fails the assertion.
/// (It does not go to 1.0 because only ~1.4 km of this thin deck lies behind the
/// mesa along the ray, and ACES compresses the top end — the veil is a doubling,
/// not a wipe, which is exactly the sort of wrongness that ships unnoticed.)
#[test]
fn clouds_do_not_veil_geometry_inside_the_slab() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (mut open, bodies) = cloud_scene(43_200.0, 1.0, 0.3);
    // A LOW, THIN deck rather than the default 1.5–4 km one, for a reason worth
    // stating: at the default extinction a 2.5 km column saturates within the
    // first kilometre, so the cloud *behind* a mountain contributes almost
    // nothing and the bug hides itself. A 100–900 m deck at an optical depth
    // around 1 is both real content (a valley stratus deck) and the regime where
    // the veil is actually visible — which is exactly where a player would see it.
    open.atmosphere.clouds.bottom = 100.0;
    open.atmosphere.clouds.top = 900.0;
    open.atmosphere.clouds.density = 0.0012;
    open.mark_dirty();
    // Looking up at ~10°, so the ray enters the deck ~580 m out and the mesa's
    // face sits just past that — the correct span is short and the naive one is
    // the rest of the deck.
    let view = horizon_view(bodies.sun, 10.0);

    // A mesa 800 m away rising to 250 m: its top 150 m are inside the deck.
    let ahead = DVec3::new(bodies.sun.x, 0.0, bodies.sun.z).normalize();
    let mut walled = open.clone();
    walled.instances.push(MeshInstance::lit(
        ahead * 800.0 + DVec3::new(0.0, 125.0, 0.0),
        Quat::IDENTITY,
        Vec3::new(3000.0, 250.0, 200.0),
        [0.42, 0.36, 0.30, 1.0],
        1,
    ));
    walled.mark_dirty();

    let open_clouds = render(&gpu, &open, &view);
    let open_clear = render(&gpu, &without_clouds(&open), &view);
    let walled_clouds = render(&gpu, &walled, &view);
    let walled_clear = render(&gpu, &without_clouds(&walled), &view);

    // The mesa's silhouette, derived rather than hard-coded: the pixels the mesa
    // changed in the cloudless pair are exactly the ones it covers.
    let mask: Vec<(u32, u32)> = (0..H)
        .flat_map(|y| (0..W).map(move |x| (x, y)))
        .filter(|&(x, y)| px(&walled_clear, x, y) != px(&open_clear, x, y))
        .collect();
    assert!(
        mask.len() > (W * H) as usize / 20,
        "the mesa covers only {} px — this test would prove nothing",
        mask.len()
    );

    // Mean luminance over the mesa's pixels. Comparing RGB *deltas* would be a
    // mistake here, and was the first thing tried: the cloud sits over a dark
    // mesa in one frame and over a bright sky in the other, so the same alpha
    // produces wildly different deltas and the metric reported 97 % either way.
    // What is comparable is the composited **alpha**, and over a near-black
    // surface that is directly recoverable.
    let lum = |img: &[u8]| -> f32 {
        let sum: f32 = mask
            .iter()
            .map(|&(x, y)| {
                let p = px(img, x, y);
                luma([
                    p[0] as f32 / 255.0,
                    p[1] as f32 / 255.0,
                    p[2] as f32 / 255.0,
                ])
            })
            .sum();
        sum / mask.len() as f32
    };
    // `open_clouds` at these pixels is the same cloud seen at ~full alpha against
    // the sky, so it stands in for the cloud's own radiance, and the dark mesa
    // stands in for zero:
    //   alpha = (L_composited − L_background) / (L_cloud − L_background)
    let bg = lum(&walled_clear);
    let cloud = lum(&open_clouds);
    let composited = lum(&walled_clouds);
    let alpha = (composited - bg) / (cloud - bg).max(1e-4);
    eprintln!(
        "veil over {} mesa px: background {bg:.4}, cloud {cloud:.4}, composited \
         {composited:.4} => alpha {alpha:.3}",
        mask.len()
    );

    assert!(
        cloud > bg + 0.05,
        "the cloud is not brighter than the mesa ({cloud:.4} vs {bg:.4}) — the \
         alpha estimate below would be meaningless"
    );
    assert!(
        alpha < 0.45,
        "geometry inside the slab is still veiled at alpha {alpha:.3} — the whole \
         deck behind the mesa is being composited over it, so the `t_far` depth \
         clamp is not doing its job"
    );
    // ...and the correct answer is not zero: ~120 m of real cloud sits between the
    // eye and the mesa's face and must still be visible. An over-eager clamp (one
    // that stopped at the slab entry, say) would fail here.
    assert!(
        alpha > 0.02,
        "the mesa shows no cloud at all (alpha {alpha:.3}) — the clamp went too far \
         and removed the cloud that is genuinely in front of it"
    );
}

/// Cloud **shadows on the world**: the layer darkens lit geometry softly and at a
/// large scale, and is byte-neutral when off.
#[test]
fn cloud_shadows_darken_lit_geometry() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (mut scene, bodies) = cloud_scene(43_200.0, 1.0, 0.2);
    // A big bright ground plane plus the sky's own key light, so what the ground
    // band measures is dominated by the DIRECT term rather than by the sky.
    scene.instances.push(MeshInstance::lit(
        DVec3::new(0.0, -1.0, 0.0),
        Quat::IDENTITY,
        Vec3::new(4000.0, 1.0, 4000.0),
        [0.8, 0.8, 0.8, 1.0],
        1,
    ));
    scene.lights.push(RenderLight {
        kind: LightKind::Directional,
        position: DVec3::ZERO,
        direction: bodies.sun.as_vec3(),
        color: [1.0, 0.98, 0.95],
        intensity: 3.0,
        range: 0.0,
        inner_cos: 1.0,
        outer_cos: 0.0,
        cast_shadows: false,
    });
    scene.mark_dirty();
    let view = look_view(DVec3::new(0.0, 3.0, 0.0), DVec3::new(0.0, 0.5, 60.0));

    let shaded = render(&gpu, &scene, &view);
    let mut unshadowed = scene.clone();
    unshadowed.atmosphere.clouds.shadow_strength = 0.0;
    unshadowed.mark_dirty();
    let lit = render(&gpu, &unshadowed, &view);

    let ground = |img: &[u8]| mean_rgb(img, 0, H * 78 / 100, W, H * 96 / 100);
    let a = ground(&shaded);
    let b = ground(&lit);
    eprintln!("cloud shadow: ground {a:?} vs unshadowed {b:?}");
    assert!(
        luma(a) < luma(b) - 0.01,
        "a solid overcast layer did not darken the ground: {a:?} vs {b:?}"
    );

    // Off ⇒ byte-identical to the same scene with clouds entirely absent, over
    // the GROUND band (the sky above still has clouds in it either way). This is
    // what the lit shaders' guarded branch exists for.
    let mut no_clouds = scene.clone();
    no_clouds.atmosphere.clouds = CloudParams::default();
    no_clouds.mark_dirty();
    let bare = render(&gpu, &no_clouds, &view);
    let band = |img: &[u8]| -> Vec<u8> {
        (H * 78 / 100..H * 96 / 100)
            .flat_map(|y| (0..W).map(move |x| (x, y)))
            .flat_map(|(x, y)| px(img, x, y))
            .collect()
    };
    assert_eq!(
        band(&lit),
        band(&bare),
        "shadow_strength = 0 is not byte-identical to no clouds at all — the \
         off path is not off"
    );
}

/// The bake-determinism gate, one level below the frame: two independent
/// renderers must write byte-identical noise volumes and shadow maps. A
/// nondeterministic bake surfaces here rather than as a flaky pixel three passes
/// downstream — the same shape as `atmosphere_luts_are_deterministic`.
#[test]
fn cloud_bakes_are_deterministic() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (scene, bodies) = cloud_scene(43_200.0, 0.7, 0.8);
    let view = horizon_view(bodies.sun, 20.0);

    let bake = || {
        let target = HeadlessTarget::new(&gpu, W, H);
        let mut renderer = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
        renderer.render(&gpu, &scene, &view, &target.view, (W, H));
        let a = renderer.atmosphere();
        (
            a.read_cloud_shape(&gpu).expect("shape readback"),
            a.read_cloud_detail(&gpu).expect("detail readback"),
            a.read_cloud_shadow(&gpu).expect("shadow readback"),
        )
    };
    let (s1, d1, h1) = bake();
    let (s2, d2, h2) = bake();
    assert_eq!(s1, s2, "cloud shape volume is not deterministic");
    assert_eq!(d1, d2, "cloud detail volume is not deterministic");
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        bits(&h1),
        bits(&h2),
        "cloud shadow map is not deterministic"
    );

    // ...and not trivially empty (an undispatched bake would also compare equal).
    assert!(s1.iter().any(|&b| b != 0), "shape volume is empty");
    assert!(d1.iter().any(|&b| b != 0), "detail volume is empty");
    assert!(
        h1.iter().any(|&v| v < 0.99),
        "the shadow map is uniformly transparent — did the bake dispatch?"
    );
    let q = CloudQuality::High;
    // Eight bytes a texel: `Rgba16Float` since SKY2 (four binary16 channels).
    let r = q.shape_res() as usize;
    assert_eq!(s1.len(), r * r * r * 8);
    let r = q.detail_res() as usize;
    assert_eq!(d1.len(), r * r * r * 8);
    let r = q.shadow_res() as usize;
    assert_eq!(h1.len(), r * r);
}

/// Mean absolute difference between horizontally adjacent pixels over the top
/// `rows` of a frame — a measure of how much **high-frequency** content the sky
/// band carries.
///
/// This is the metric the jitter trades banding for and the temporal pass buys
/// back: a coherent sampling lattice puts its error in low frequencies (bands),
/// a blue-noise-jittered one puts it here, and an accumulation over several
/// jitter offsets removes it. Cloud structure contributes to this number too,
/// which is why it is only ever compared between two renders of the *same*
/// clouds.
fn adjacent_spread(img: &[u8], rows: u32) -> f32 {
    let mut sum = 0.0f32;
    let mut n = 0u32;
    for y in 0..rows {
        for x in 1..W {
            let a = px(img, x - 1, y);
            let b = px(img, x, y);
            for c in 0..3 {
                sum += (a[c] as f32 - b[c] as f32).abs() / 255.0;
            }
            n += 3;
        }
    }
    sum / n.max(1) as f32
}

// ── POWDER'S RENDERED CONSEQUENCE — measured, and deliberately not asserted ──
//
// The SKY2 audit set out to close the wave's first carried item by adding an
// away-from-sun structural arm, on the reasoning that all four cloud goldens
// look TOWARD the sun where `facing` is ~0 and the term is gated off by
// geometry. It was built, and then it was measured by mutation — `single`
// forced to 1.0 in `cloud.wgsl`, everything else identical, sixteen renders
// over twelve configurations (noon and dusk; anti-sun, side-on and sunward;
// coverage 0.30 to 0.75; pitch 12° to 45°) — and the arm was thrown away,
// because what the mutation says is:
//
//   * at NOON the powder term moves at most **14 of 255** on any single pixel
//     and **≤0.9 %** on every aggregate (mean cloud luma, brightest decile,
//     anti-sun/sunward ratio) in every configuration;
//   * at DUSK, where the sun term is a larger share of a cloud's radiance than
//     the ambient, it reaches **44 of 255** on one pixel — and still moves the
//     mean anti-sun cloud luma by **1.0 %** (0.528 → 0.523).
//
// A bound that has to separate 0.528 from 0.523 is not a gate, it is a byte pin
// with extra steps, and the golden set already holds byte pins. So the honest
// statement is the measurement rather than an arm, and it is stronger and less
// flattering than the one the wave carried: powder's rendered consequence is not
// missing from the goldens because of where their cameras point, it is ≤1 % of
// the image *everywhere*. The arithmetic and its three gates are pinned by
// `clouds::tests::powder_darkens_the_thin_sunward_side_only` and
// `the_powder_gate_ramps_instead_of_stepping_at_sunrise`; the picture is not,
// because there is not enough of it to pin.

/// **The cloud pass's own temporal history** (wave SKY2), measured rather than
/// asserted into existence.
///
/// Three claims, one test, because they are the same mechanism seen three ways:
///
/// 1. **Off is inert.** With `cloud_temporal` off, a renderer that has drawn ten
///    frames produces byte-for-byte what a fresh one produces at the same level
///    clock. No accumulation, no dependence on how long the process has been up
///    — which is what lets every golden render with the jitter ON.
/// 2. **On accumulates.** With it on, the tenth frame is a different image, and
///    the difference is in the direction the wave claims: LESS high-frequency
///    content in the sky band, because the jittered march's per-pixel error is
///    what the history averages away.
/// 3. **On is still deterministic as a sequence.** Ten frames rendered twice
///    give the same tenth frame. The accumulation is a function of the level
///    clocks visited, not of the wall clock — the distinction the whole jitter
///    design rests on.
#[test]
fn the_cloud_temporal_pass_accumulates_and_stays_a_function_of_the_clock() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (base, bodies) = cloud_scene(43_200.0, 0.55, 0.85);
    let view = horizon_view(bodies.sun, 25.0);
    let target = HeadlessTarget::new(&gpu, W, H);

    // The level's clock advances a frame at a time, which is what walks the
    // jitter sequence. A static clock would freeze the pattern and the
    // accumulation would converge onto the single frame it started from — true,
    // deterministic, and no evidence of anything.
    const FRAMES: u32 = 10;
    const DT: f64 = 1.0 / 60.0;
    let at = |f: u32| -> RenderScene {
        let mut s = base.clone();
        s.atmosphere.clouds.time_s = 43_200.0 + f64::from(f) * DT;
        s.mark_dirty();
        s
    };

    // The last TWO frames of a run, which is what makes the flicker measurable:
    // consecutive level clocks are consecutive jitter offsets, so the difference
    // between these two IS the march's per-pixel error, isolated from the cloud
    // structure that swamps it in any single-frame statistic.
    let run2 = |temporal: bool| -> (Vec<u8>, Vec<u8>) {
        let mut r = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
        r.set_settings(RenderSettings {
            cloud_temporal: temporal,
            ..RenderSettings::default()
        });
        for f in 0..FRAMES - 1 {
            r.render(&gpu, &at(f), &view, &target.view, (W, H));
        }
        let penultimate = target.read_rgba(&gpu).expect("readback");
        r.render(&gpu, &at(FRAMES - 1), &view, &target.view, (W, H));
        (penultimate, target.read_rgba(&gpu).expect("readback"))
    };
    let run = |temporal: bool| -> Vec<u8> { run2(temporal).1 };
    // A fresh renderer at the LAST clock only — no history behind it.
    let single = {
        let mut r = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
        r.set_settings(RenderSettings::default());
        r.render(&gpu, &at(FRAMES - 1), &view, &target.view, (W, H));
        target.read_rgba(&gpu).expect("readback")
    };

    // (1) Off is inert — byte-identical, not "close".
    let off = run(false);
    assert_eq!(
        off, single,
        "with cloud_temporal OFF the tenth frame differs from a fresh one at the \
         same clock — something is accumulating that should not be"
    );

    // (3) On is a function of the clocks visited.
    let on = run(true);
    assert_eq!(
        on,
        run(true),
        "the temporal accumulation is not reproducible across runs"
    );

    // (2) On accumulates, in the direction claimed: FLICKER between consecutive
    // jitter offsets is what the history is for, so that is what is measured.
    let rows = H / 3;
    let (a_off, b_off) = run2(false);
    let (a_on, b_on) = run2(true);
    let flicker_off = image_diff(&a_off, &b_off, W, H).0;
    let flicker_on = image_diff(&a_on, &b_on, W, H).0;
    let (mean, max) = image_diff(&on, &single, W, H);
    let spread_on = adjacent_spread(&on, rows);
    let spread_off = adjacent_spread(&single, rows);
    eprintln!(
        "cloud temporal: frame-to-frame flicker {flicker_on:.6} accumulated vs \
         {flicker_off:.6} raw; vs single mean {mean:.5} max {max:.5}; adjacent \
         spread {spread_on:.5} vs {spread_off:.5}"
    );
    assert!(
        mean > 1e-4,
        "the temporal pass changed nothing (mean {mean:.6}) — it is a no-op"
    );
    assert!(
        flicker_off > 1e-5,
        "the raw march does not flicker between jitter offsets ({flicker_off:.7}) \
         — the jitter is not reaching the image and this test measures nothing"
    );
    assert!(
        flicker_on < flicker_off * 0.6,
        "the history did not damp the jitter: {flicker_on:.6} against a raw \
         {flicker_off:.6}"
    );
    // ...and it did not do it by going flat, which a blend that returned the
    // history unclamped would also achieve.
    assert!(
        spread_on > spread_off * 0.5,
        "the accumulation flattened the sky ({spread_on:.5} against \
         {spread_off:.5}) — that is a smear, not a convergence"
    );
}

/// The gap between binary16 `h` and its successor, in the field's units — one
/// "step" measured as a value rather than as a bit pattern.
///
/// Everything here is positive and finite (the cloud field lives in `[0, 1]`), so
/// the bit patterns are monotone in magnitude and this is well defined.
fn half_step_value(h: u16) -> f32 {
    (half_to_f32(h.saturating_add(1)) - half_to_f32(h)).abs()
}

/// One baked binary16 channel against its CPU reference, **in both units the
/// disagreement comes in**: steps of the bit pattern, and units of the field.
fn half_channel_delta(got: u16, want: u16) -> (u16, f32) {
    (
        got.abs_diff(want),
        (half_to_f32(got) - half_to_f32(want)).abs(),
    )
}

/// The envelope the two bake gates share (`cloud_noise_bake_matches_the_cpu_reference`
/// and `cloud_quality_switch_rebuilds_the_cloud_binds`) — one door, so a bound
/// re-derived for one of them cannot leave the other carrying the old exposure.
///
/// A channel passes on **either** ruler. See `clouds::CPU_GPU_VALUE_TOLERANCE`
/// for why there are two: in the Worley channels' near-zero tail a step of the
/// bit pattern is as fine as an f32 last place of the `√best ≈ 1` the value was
/// subtracted from, so the step ruler asks there for an agreement no two
/// compilers owe each other.
fn baked_half_agrees(got: u16, want: u16) -> bool {
    let (steps, dv) = half_channel_delta(got, want);
    steps <= CPU_GPU_STEP_TOLERANCE || dv <= CPU_GPU_VALUE_TOLERANCE
}

/// **CPU/GPU parity of the noise bake.** The GPU volumes must reproduce
/// `inf_render::shape_texel` / `detail_texel` to within the documented envelope:
/// on every channel either at most `CPU_GPU_STEP_TOLERANCE` steps of the binary16
/// bit pattern or at most `CPU_GPU_VALUE_TOLERANCE` of value, with the second
/// route taken by at most `CPU_GPU_VALUE_ESCAPE_FRACTION` of channels; and exact
/// equality on at least `CPU_GPU_EXACT_CHANNEL_FRACTION` of channels.
///
/// **Both units changed at SKY2** and this comment is the third place that had to
/// say so: the volumes are `Rgba16Float`, so a texel is four little-endian halves
/// rather than four bytes, "one step" is the adjacent representable value rather
/// than `1/255` of the range, and the floor counts channels rather than whole
/// texels. The reason for the last of those is the finding SKY2 landed: the
/// disagreement here is **one-sided** — 49.98 % of channels exactly one step low
/// against 0.00 % high — because WGSL does not pin the rounding mode of the
/// `textureStore` conversion and this adapter truncates where `f32_to_half`
/// rounds to nearest. Four independent coin-flips a texel puts *whole-texel*
/// agreement at 0.5⁴ = 6.25 %, which no honest floor over texels could survive.
///
/// **The value ruler is the macOS CI red's answer**, and it is the interesting
/// half of this gate now. SKY2's envelope was one step everywhere, on the
/// argument that an f32 last place is 2¹³ times finer than an f16 step and so
/// nothing short of a port error could cross one. That argument holds for a
/// well-scaled value and fails completely in the Worley channels' cancellation
/// tail, where `1 − min(√best, 1)` lands in binary16's *subnormals* and one step
/// is 2⁻²⁴ — exactly one last place of the near-1.0 quantity it came from.
/// Windows/Vulkan happened to land inside one step there; the Apple-silicon
/// runner landed two out, and SKY2's panic message called that "a port error, not
/// FMA contraction and not a rounding-mode difference" — a claim about a platform
/// the gate had never run on. It is neither: it is one last place of arithmetic
/// on top of one step of store rounding, in the one corner where those are the
/// same size. `clouds::CPU_GPU_VALUE_TOLERANCE` carries the derivation.
///
/// What the pair still catches without appeal to any platform: everything the
/// field is built on that *could* diverge structurally — the hash, the gradient
/// table, the lattice wrap — is pure integer arithmetic, and a mistake in any of
/// those moves whole texels by O(0.1–1). That is six orders of magnitude over the
/// value tolerance and hundreds of steps over the step tolerance, so it fails on
/// both rulers at once, on essentially every channel.
#[test]
fn cloud_noise_bake_matches_the_cpu_reference() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (scene, bodies) = cloud_scene(43_200.0, 0.7, 0.8);
    let view = horizon_view(bodies.sun, 20.0);
    let target = HeadlessTarget::new(&gpu, W, H);
    let mut renderer = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
    renderer.render(&gpu, &scene, &view, &target.view, (W, H));
    let res = renderer.atmosphere();
    let q = res.cloud_quality;
    let seed = scene.atmosphere.clouds.seed;

    // Since SKY2 the volumes are `Rgba16Float`, so a texel is four little-endian
    // binary16s and "one LSB" means one step of the bit pattern — i.e. the
    // adjacent representable value. Because the halves here are all positive and
    // finite, the bit patterns are monotone in magnitude and a difference of the
    // patterns IS the number of representable values between them.
    let compare =
        |what: &str, data: &[u8], edge: u32, reference: &dyn Fn(u32, u32, u32) -> [u16; 4]| {
            let mut exact_ch = 0u64;
            let mut exact_texel = 0u64;
            let mut low = 0u64;
            let mut high = 0u64;
            let mut total = 0u64;
            // The distribution, which is the thing the macOS red could not be
            // read from: one coordinate and no counts says nothing about whether
            // two steps is a corner or a collapse.
            let mut steps = [0u64; 5]; // 0, 1, 2, 3, >=4
            let mut escaped = 0u64; // > 1 step, inside the value tolerance
            let mut over = Vec::new(); // > 1 step and outside it — the failures
            let mut tail = 0u64; // channels whose f16 grid is finer than that
            let mut worst = 0u16;
            let mut worst_at = (0, 0, 0, 0usize);
            let mut worst_dv = 0.0f32;
            for z in 0..edge {
                for y in 0..edge {
                    for x in 0..edge {
                        let i = (((z * edge + y) * edge + x) * 8) as usize;
                        let mut got = [0u16; 4];
                        for (c, g) in got.iter_mut().enumerate() {
                            *g = u16::from_le_bytes([data[i + c * 2], data[i + c * 2 + 1]]);
                        }
                        let want = reference(x, y, z);
                        total += 1;
                        if got == want {
                            exact_texel += 1;
                        }
                        for c in 0..4 {
                            match got[c].cmp(&want[c]) {
                                std::cmp::Ordering::Equal => exact_ch += 1,
                                std::cmp::Ordering::Less => low += 1,
                                std::cmp::Ordering::Greater => high += 1,
                            }
                            let (d, dv) = half_channel_delta(got[c], want[c]);
                            steps[(d as usize).min(4)] += 1;
                            if half_step_value(want[c]) <= CPU_GPU_VALUE_TOLERANCE {
                                tail += 1;
                            }
                            if d > CPU_GPU_STEP_TOLERANCE {
                                if baked_half_agrees(got[c], want[c]) {
                                    escaped += 1;
                                } else if over.len() < 8 {
                                    over.push((x, y, z, c, got[c], want[c], d, dv));
                                }
                            }
                            if d > worst {
                                worst = d;
                                worst_at = (x, y, z, c);
                                worst_dv = dv;
                            }
                        }
                    }
                }
            }
            let channels = total * 4;
            let per_channel = exact_ch as f64 / channels as f64;
            let escaped_frac = escaped as f64 / channels as f64;
            // Printed unconditionally AND quoted into every panic below: a
            // divergence on a platform nobody here can run has to be diagnosable
            // from the CI log alone.
            let dist = format!(
                "{what}: {channels} channels over {total} texels — {:.4}% exact \
                 ({:.4}% of texels), {:.2}% low / {:.2}% high; |d| in f16 steps = \
                 [0]{} [1]{} [2]{} [3]{} [>=4]{}; {escaped} channels ({escaped_frac:.3e}) \
                 passed on value not steps, against a near-zero tail of {tail} \
                 ({:.3e}); worst |d| = {worst} steps = {worst_dv:e} at (x,y,z,ch) \
                 {worst_at:?}",
                per_channel * 100.0,
                exact_texel as f64 / total as f64 * 100.0,
                low as f64 / channels as f64 * 100.0,
                high as f64 / channels as f64 * 100.0,
                steps[0],
                steps[1],
                steps[2],
                steps[3],
                steps[4],
                tail as f64 / channels as f64,
            );
            eprintln!("{dist}");
            // What each regime means, so the next reader of a red log does not
            // have to re-derive it (SKY2's message asserted the wrong one):
            //
            //   every channel at 0 or 1 step   an adapter that rounds to nearest,
            //                                  or one that truncates; both legal,
            //                                  and the low/high split tells them
            //                                  apart.
            //   a handful at 2+ steps, all
            //   inside CPU_GPU_VALUE_TOLERANCE the Worley cancellation tail: an
            //                                  f32 last place stacked on the
            //                                  store's rounding, in the corner
            //                                  where those are the same size.
            //   anything outside that value    a real disagreement in the field.
            //   floor, or a tail that is a     Not a rounding mode: no rounding
            //   large share of the volume      of the same number can do it.
            assert!(
                over.is_empty(),
                "{dist}\n{} channel(s) are more than {CPU_GPU_STEP_TOLERANCE} step(s) \
                 out AND outside the {CPU_GPU_VALUE_TOLERANCE:e} value envelope — that \
                 is a real disagreement in the field, not a rounding mode and not the \
                 near-zero tail (which is bounded by the value envelope by \
                 construction). First offenders as (x, y, z, channel, got, want, steps, \
                 |dvalue|): {over:?}",
                over.len()
            );
            assert!(
                escaped_frac <= CPU_GPU_VALUE_ESCAPE_FRACTION,
                "{dist}\n{escaped_frac:.3e} of channels needed the value envelope rather \
                 than the step envelope, over a ceiling of {CPU_GPU_VALUE_ESCAPE_FRACTION:e} \
                 — the escape is only legitimate while the cancellation tail is a \
                 thousandth of the field, and this says the field has collapsed toward \
                 zero somewhere it should not have"
            );
            assert!(
                per_channel >= CPU_GPU_EXACT_CHANNEL_FRACTION,
                "{dist}\nonly {:.2}% of channels are bit-exact (the envelope requires \
                 {:.0}%) — a last-bit rounding-mode difference cannot get below half, \
                 so this is a computation difference",
                per_channel * 100.0,
                CPU_GPU_EXACT_CHANNEL_FRACTION * 100.0
            );
        };

    let shape = res.read_cloud_shape(&gpu).expect("shape readback");
    compare("cloud shape", &shape, q.shape_res(), &|x, y, z| {
        shape_texel(seed, x, y, z, q.shape_res())
    });
    let detail = res.read_cloud_detail(&gpu).expect("detail readback");
    compare("cloud detail", &detail, q.detail_res(), &|x, y, z| {
        detail_texel(seed, x, y, z, q.detail_res())
    });
}

/// **CPU/GPU parity of the density function**, measured end-to-end through the
/// cloud-shadow map.
///
/// The shadow map is the right probe: every texel is a Beer–Lambert march of
/// `cloud_density` along the sun, so agreeing on it means agreeing on the whole
/// density function — the weather bias, the height gradient, the Perlin–Worley
/// remap, the coverage dissolve and the erosion, in the right order. The CPU
/// reference evaluates against the **read-back** volumes rather than re-baking
/// them, so any disagreement is attributable to the density function itself and
/// not to the (separately gated) bake.
///
/// The envelope is relative and much looser than the bake's, for a stated reason:
/// hardware trilinear filtering carries only ~8 bits of sub-texel precision while
/// the reference filters in full f32, so exact agreement is not available at any
/// price. `CPU_GPU_SHADOW_TOLERANCE` is far tighter than what a genuinely wrong
/// march produces.
#[test]
fn cloud_density_matches_the_cpu_reference() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (scene, bodies) = cloud_scene(43_200.0, 0.75, 0.8);
    let view = horizon_view(bodies.sun, 20.0);
    let target = HeadlessTarget::new(&gpu, W, H);
    let mut renderer = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
    renderer.render(&gpu, &scene, &view, &target.view, (W, H));
    let res = renderer.atmosphere();
    let q = res.cloud_quality;

    let volumes = CloudVolumes {
        shape: res.read_cloud_shape(&gpu).expect("shape readback"),
        shape_res: q.shape_res(),
        detail: res.read_cloud_detail(&gpu).expect("detail readback"),
        detail_res: q.detail_res(),
    };
    let gpu_map = res.read_cloud_shadow(&gpu).expect("shadow readback");
    let params = scene.atmosphere.clouds;
    let sun = bodies.sun.as_vec3().normalize();

    // The map's parameterization, mirrored from `cs_cloud_shadow`.
    let edge = q.shadow_res();
    let extent = inf_render::passes::sky_lut::CLOUD_SHADOW_EXTENT_M;
    let centre = inf_render::passes::sky_lut::AtmosphereGpu::cloud_shadow_centre(
        [view.eye_world.x as f32, view.eye_world.z as f32],
        q,
    );

    // A deterministic scatter of taps rather than every texel: the CPU march is
    // orders of magnitude slower than the GPU's, and a few thousand taps is
    // plenty to catch a structural disagreement.
    let stride = (edge / 48).max(1);
    let mut worst = 0.0f32;
    let mut worst_at = (0, 0);
    let mut sum = 0.0f64;
    let mut n = 0u64;
    let mut shadowed = 0u64;
    for iy in (0..edge).step_by(stride as usize) {
        for ix in (0..edge).step_by(stride as usize) {
            let u = (ix as f32 + 0.5) / edge as f32;
            let v = (iy as f32 + 0.5) / edge as f32;
            let p = [
                centre[0] + (u - 0.5) * extent,
                params.bottom,
                centre[1] + (v - 0.5) * extent,
            ];
            let want =
                volumes.sun_transmittance(&params, p, [sun.x, sun.y, sun.z], q.shadow_steps());
            let got = gpu_map[(iy * edge + ix) as usize];
            let d = (got - want).abs();
            if d > worst {
                worst = d;
                worst_at = (ix, iy);
            }
            sum += d as f64;
            n += 1;
            if want < 0.99 {
                shadowed += 1;
            }
        }
    }
    let mean = sum / n as f64;
    eprintln!(
        "cloud density parity: {n} taps, mean |d| = {mean:.5}, worst = {worst:.5} at \
         {worst_at:?}, {shadowed} taps genuinely shadowed"
    );
    // The gate must not pass by both sides finding an empty sky.
    assert!(
        shadowed > n / 10,
        "only {shadowed}/{n} taps are shadowed at all — the fixture is too clear to \
         test anything"
    );
    assert!(
        worst <= CPU_GPU_SHADOW_TOLERANCE,
        "worst |d| = {worst:.5} at {worst_at:?} exceeds the documented \
         {CPU_GPU_SHADOW_TOLERANCE} envelope"
    );
    assert!(
        (mean as f32) < CPU_GPU_SHADOW_TOLERANCE * 0.25,
        "mean |d| = {mean:.5} is too large even if the worst case fits"
    );
}

/// The cloud field drifts with the **level's clock**, not with a wall clock — the
/// deterministic-wind law. Two renders at the same time of day are byte-identical;
/// advancing the clock moves the sky; and a whole number of tile wraps is a no-op,
/// which is what keeps an all-day session from quantizing into stair-steps.
#[test]
fn cloud_wind_follows_the_level_clock() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (mut scene, bodies) = cloud_scene(43_200.0, 0.6, 0.8);
    let view = horizon_view(bodies.sun, 25.0);

    scene.atmosphere.clouds.time_s = 600.0;
    scene.mark_dirty();
    let a = render(&gpu, &scene, &view);
    let b = render(&gpu, &scene, &view);
    assert_eq!(a, b, "the same clock rendered two different skies");

    scene.atmosphere.clouds.time_s = 1200.0;
    scene.mark_dirty();
    let later = render(&gpu, &scene, &view);
    assert_ne!(a, later, "ten minutes of wind moved nothing");

    // One whole tile of drift is exactly a no-op, because the volumes tile. Both
    // wind components are set to the same speed so they wrap in the same breath.
    scene.atmosphere.clouds.wind_x = 8.0;
    scene.atmosphere.clouds.wind_z = 8.0;
    scene.atmosphere.clouds.time_s = 0.0;
    scene.mark_dirty();
    let t0 = render(&gpu, &scene, &view);
    scene.atmosphere.clouds.time_s = (inf_render::clouds::SHAPE_TILE_M / 8.0) as f64;
    scene.mark_dirty();
    let wrapped = render(&gpu, &scene, &view);
    let (mean, max) = image_diff(&t0, &wrapped, W, H);
    eprintln!("one-tile wrap: mean {mean:.5} max {max:.5}");
    assert!(
        mean < 0.01 && max < 0.08,
        "a whole tile of wind drift was not a no-op (mean {mean}, max {max}) — the \
         field is not tiling"
    );
}

/// The quality-switch seam, extended to the cloud resources. The three cloud
/// textures live in `AtmosphereResources` and are recreated with the LUTs, so a
/// bind group that missed the generation would keep sampling the previous tier's
/// volumes — silently, exactly as the P17.2 `EnvBinding` comment warns.
#[test]
fn cloud_quality_switch_rebuilds_the_cloud_binds() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (scene, bodies) = cloud_scene(43_200.0, 0.7, 0.85);
    let view = horizon_view(bodies.sun, 25.0);

    let target = HeadlessTarget::new(&gpu, W, H);
    let mut renderer = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
    let mut settings = RenderSettings::default();
    let mut frames = Vec::new();
    let mut seen = Vec::new();
    let mut sizes = Vec::new();
    let seed = scene.atmosphere.clouds.seed;
    for q in [
        AtmosphereQuality::High,
        AtmosphereQuality::Low,
        AtmosphereQuality::Medium,
        AtmosphereQuality::High,
    ] {
        settings.atmosphere.quality = q;
        renderer.set_settings(settings);
        renderer.render(&gpu, &scene, &view, &target.view, (W, H));
        frames.push(target.read_rgba(&gpu).expect("readback"));
        let a = renderer.atmosphere();
        seen.push(a.cloud_quality);

        // ── the assertion that actually bites ──
        //
        // Every whole-frame comparison below can be satisfied by a STALE bind
        // group, and that was the original version of this test's flaw: with the
        // generation dropped from the bake's `GenCache` key, the bake keeps
        // writing into the *previous* tier's texture views, the newly created
        // ones stay at their zeroed initial contents, and the frames still differ
        // by tier because the march step counts come from the uniform rather than
        // from the bind group. So: read the freshly-created volume back and
        // require it to carry the field. Zeros mean the bake wrote somewhere else.
        //
        // Mutation-verified: dropping `res.generation` from `CloudBakeNode`'s
        // `noise_bg` key makes this fail on the second tier with an all-zero
        // volume, while every other assertion in this test still passes.
        let cq = a.cloud_quality;
        let shape = a.read_cloud_shape(&gpu).expect("shape readback");
        assert!(
            shape.iter().any(|&b| b != 0),
            "{q:?}: the volume is all zeros after the switch — the bake wrote into \
             a previous tier's texture, so a cloud bind group is stale"
        );
        // Stronger than "not zero": it must be the field this tier should hold, at
        // this tier's resolution. A stale *render* bind group cannot be caught by
        // a readback, but a stale bake one cannot survive this.
        let res = cq.shape_res();
        for &(x, y, z) in &[
            (0u32, 0u32, 0u32),
            (res / 3, res / 2, res / 5),
            (res - 1, res - 1, res - 1),
        ] {
            // Eight bytes a texel: `Rgba16Float` since SKY2.
            let i = (((z * res + y) * res + x) * 8) as usize;
            let mut got = [0u16; 4];
            for (c, g) in got.iter_mut().enumerate() {
                *g = u16::from_le_bytes([shape[i + c * 2], shape[i + c * 2 + 1]]);
            }
            let want = shape_texel(seed, x, y, z, res);
            for c in 0..4 {
                // The same two-ruler envelope the parity gate uses, through the
                // same door: three texels is 12 channels, the cancellation tail
                // is ~6.6e-4 of channels, so a one-step-only bound here had a
                // few-percent chance per run of red-CIing on an adapter this
                // machine cannot see — the P25 one-platform-bounds law, met
                // where it is cheapest to miss.
                let (d, dv) = half_channel_delta(got[c], want[c]);
                assert!(
                    baked_half_agrees(got[c], want[c]),
                    "{q:?}: texel ({x},{y},{z}) channel {c} is {} not {} ({d} f16 \
                     steps = {dv:e}) — the volume does not hold this tier's field",
                    got[c],
                    want[c]
                );
            }
        }
        sizes.push(shape.len());
    }

    // The tier followed the setting and the volumes really did resize.
    assert_eq!(
        seen,
        vec![
            CloudQuality::High,
            CloudQuality::Low,
            CloudQuality::Medium,
            CloudQuality::High,
        ],
        "the cloud resources did not follow the atmosphere quality"
    );
    assert!(
        sizes[1] < sizes[0],
        "the Low volume is not smaller than High's"
    );

    assert_ne!(
        frames[1], frames[0],
        "Low and High rendered identically — the cloud tier had no visible effect"
    );
    assert_eq!(
        frames[3], frames[0],
        "returning to High did not reproduce the High frame — a cloud bind group is \
         still holding a previous tier's volume"
    );
}

// ── P17.4 weather states + precipitation ─────────────────────────────────────
//
// Three more sweep goldens, built the same way the P17.2/P17.3 ones are: from
// `inf_math::solar` at the component defaults, with the weather block's values
// taken straight from `inf_ecs::components::WeatherPreset::params()`. So what
// these picture is what a level gets when it clicks a preset button, not a tuned
// demo of the shader.
//
// The preset numbers are LITERALS here rather than reached for through `inf-ecs`,
// for the reason `tod_scene` spells the `SkyAtmosphere` defaults out: `inf-render`
// does not depend on `inf-ecs` and must not start doing so for a test. The Ring-0
// side pins the same table (`preset_names_round_trip_and_reject_typos` asserts the
// presets are distinct; the phase-17 gate asserts the projected values), and the
// frontend mirror is pinned by `todModel.test.ts` — three copies, each with a test
// naming the other.
//
// None of the 33 pre-P17.4 goldens moves: weather is disabled by default, the
// precipitation node dispatches nothing at all, and the projection's
// `precip.enabled` is false unless a weather block asks for rain. Verified the
// P17.2/P17.3 way — running the whole suite under `INF_BLESS_GOLDENS=1` and
// confirming `git status` reports zero changed PNGs — not merely asserted.

/// A weather state over the P17.3 clouds and the P17.2 sky.
///
/// `(coverage, cloud_type, wind_x, wind_z, fog_density, precipitation, snowiness)`
/// — the seven fields of `WeatherParams`, in declaration order.
#[allow(clippy::too_many_arguments)]
fn weather_scene(
    seconds: f64,
    coverage: f32,
    cloud_type: f32,
    wind_x: f32,
    wind_z: f32,
    fog_density: f32,
    precipitation: f32,
    snowiness: f32,
) -> (RenderScene, inf_math::solar::SkyBodies) {
    let (mut scene, bodies) = tod_scene(seconds);
    scene.atmosphere.clouds = CloudParams {
        enabled: true,
        coverage,
        cloud_type,
        wind_x,
        wind_z,
        ..CloudParams::default()
    };
    scene.atmosphere.fog = HeightFog {
        density: fog_density,
        ..HeightFog::default()
    };
    scene.atmosphere.precip = PrecipParams {
        enabled: precipitation > 0.0,
        intensity: precipitation,
        snowiness,
        wind_x,
        wind_z,
        // A fixed clock reading: the golden must picture ONE instant, and the
        // drift is a pure function of this number (`PrecipParams::offsets`), so
        // pinning it is what makes the image reproducible at all.
        time_s: 1_234.5,
        ..PrecipParams::default()
    };
    (scene, bodies)
}

/// The Storm preset — `WeatherPreset::Storm.params()`, field for field — over a
/// low, thick storm deck. The slab geometry is *not* part of a preset (a preset
/// drives coverage and type, not altitude), so it is authored here the way a
/// level would author it, and for the reason `clouds_overcast` lowers its stratus
/// sheet: a 1.5-4 km fair-weather base seen from the ground is mostly horizon,
/// and a storm is a ceiling.
fn storm_scene(seconds: f64) -> (RenderScene, inf_math::solar::SkyBodies) {
    let (mut scene, bodies) = weather_scene(seconds, 1.0, 0.35, 22.0, 9.0, 6.0e-4, 1.0, 0.0);
    scene.atmosphere.clouds.bottom = 600.0;
    scene.atmosphere.clouds.top = 2800.0;
    scene.mark_dirty();
    (scene, bodies)
}
/// The Fog preset.
fn fog_scene(seconds: f64) -> (RenderScene, inf_math::solar::SkyBodies) {
    weather_scene(seconds, 0.5, 0.1, 1.5, 0.5, 6.0e-3, 0.0, 0.0)
}
/// The Snow preset.
fn snow_scene(seconds: f64) -> (RenderScene, inf_math::solar::SkyBodies) {
    weather_scene(seconds, 0.9, 0.3, 5.0, 2.0, 1.2e-3, 0.7, 1.0)
}

/// The same scene with the precipitation switched off — the control every
/// precipitation assertion compares against, so "it drew something" is measured
/// rather than assumed.
fn without_precip(scene: &RenderScene) -> RenderScene {
    let mut s = scene.clone();
    s.atmosphere.precip = PrecipParams::default();
    s.mark_dirty();
    s
}

/// A few metres of ground under the camera, so the frame is not all sky and the
/// depth buffer has something in it for the precipitation to be tested against.
fn ground_plane(scene: &mut RenderScene) {
    scene.instances.push(MeshInstance::lit(
        DVec3::new(0.0, -0.5, 0.0),
        Quat::IDENTITY,
        Vec3::new(400.0, 1.0, 400.0),
        [0.22, 0.24, 0.26, 1.0],
        1,
    ));
    scene.mark_dirty();
}

/// **Storm at noon.** Full coverage, a hard wind and heavy rain. The assertions
/// are the two things a weather state has to get right at once: the *sky* went
/// overcast (the cloud half), and the *air* filled with rain (the precipitation
/// half), measured against the same frame with each switched off in turn.
#[test]
fn golden_weather_storm_noon() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (mut scene, bodies) = storm_scene(43_200.0);
    ground_plane(&mut scene);
    let view = horizon_view(bodies.sun, 14.0);
    let img = check_golden(&gpu, "weather_storm_noon", &scene, &view);

    let dry = render(&gpu, &without_precip(&scene), &view);
    let clear = render(&gpu, &without_clouds(&without_precip(&scene)), &view);

    // The cloud half. `clouds_overcast` already owns the de-blueing claim; what a
    // *weather preset* has to show is that the whole coherent state took effect,
    // so this measures the two things a storm ceiling does to the sky: it covers
    // it, and it darkens it.
    let covered = changed_fraction(&dry, &clear, H / 3);
    let sky = mean_rgb(&dry, 0, 0, W, H / 3);
    let bare = mean_rgb(&clear, 0, 0, W, H / 3);
    eprintln!("storm sky {sky:?} vs clear {bare:?}; covered {covered:.3}");
    assert!(
        covered > 0.9,
        "the storm deck left {:.1}% of the sky open",
        100.0 * (1.0 - covered)
    );
    assert!(
        luma(sky) < luma(bare) * 0.9,
        "the storm deck did not darken the sky: {sky:?} vs {bare:?}"
    );

    // The precipitation half: the rain perceptibly changed a real fraction of the
    // frame. The threshold is low on purpose — a drop is a faint mark by design
    // (see PRECIP_ALPHA), and a sheet of rain is a *density* of them, so demanding
    // heavy per-pixel deltas would be demanding the wrong look.
    let wet = changed_fraction(&img, &dry, H);
    eprintln!("storm rain covered {wet:.4} of the frame");
    assert!(wet > 0.01, "heavy rain changed almost nothing ({wet:.4})");

    // …and it is DISTRIBUTED rather than a blob: split the frame into eight
    // vertical bands and require rain in most of them. A single bright artefact
    // (a degenerate quad, a NaN centre) would pass the fraction test above and
    // fail this one.
    let mut bands = 0;
    for b in 0..8u32 {
        let (x0, x1) = (b * W / 8, (b + 1) * W / 8);
        let mut n = 0u32;
        for y in 0..H {
            for x in x0..x1 {
                let a = px(&img, x, y);
                let c = px(&dry, x, y);
                let d: i32 = (0..3).map(|i| (a[i] as i32 - c[i] as i32).abs()).sum();
                if d > 4 {
                    n += 1;
                }
            }
        }
        if n > 4 {
            bands += 1;
        }
    }
    eprintln!("storm rain reached {bands}/8 bands");
    assert!(bands >= 6, "the rain is not distributed across the frame");
}

/// **Fog at dawn.** The Fog preset's 6e-3 m⁻¹ is a Koschmieder visibility of
/// ~500 m, so the assertion is the one thing fog must do: a distant wall loses
/// its contrast against the sky while a near one keeps it.
#[test]
fn golden_weather_fog_dawn() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (mut scene, bodies) = fog_scene(18_000.0);
    assert!(
        bodies.sun.y > -0.1 && bodies.sun.y < 0.25,
        "05:00 should put the sun near the horizon"
    );
    ground_plane(&mut scene);
    // Two dark walls of identical albedo and identical SCREEN size — the far one
    // is the near one scaled 30x about the eye — so every pixel of difference
    // between them is the fog. (The `aerial_fog` construction, reused because it
    // is the only way to compare two distances without also comparing two sizes.)
    let dir = DVec3::new(bodies.sun.x, 0.0, bodies.sun.z).normalize();
    for (dist, scale) in [(30.0f64, 1.0f32), (900.0, 30.0)] {
        let across = DVec3::new(-dir.z, 0.0, dir.x);
        scene.instances.push(MeshInstance::lit(
            dir * dist + across * (dist * 0.25),
            Quat::IDENTITY,
            Vec3::new(12.0 * scale, 12.0 * scale, 0.5 * scale),
            [0.08, 0.08, 0.09, 1.0],
            1,
        ));
    }
    scene.mark_dirty();
    let view = horizon_view(bodies.sun, 2.0);
    let img = check_golden(&gpu, "weather_fog_dawn", &scene, &view);

    // The control: the same scene with the fog density back at zero.
    let mut clear = scene.clone();
    clear.atmosphere.fog = HeightFog::default();
    clear.mark_dirty();
    let dry = render(&gpu, &clear, &view);

    // Fog raises the darkest thing in frame toward the sky: the walls are much
    // darker than the air, so the frame's minimum luma must climb.
    let darkest = |img: &[u8]| {
        let mut lo = 1.0f32;
        for y in 0..H {
            for x in 0..W {
                let c = px(img, x, y);
                let l = luma([
                    c[0] as f32 / 255.0,
                    c[1] as f32 / 255.0,
                    c[2] as f32 / 255.0,
                ]);
                lo = lo.min(l);
            }
        }
        lo
    };
    let foggy = darkest(&img);
    let clean = darkest(&dry);
    eprintln!("fog_dawn darkest {foggy:.4} vs clear {clean:.4}");
    assert!(
        foggy > clean + 0.01,
        "the fog preset did not lift the shadows: {foggy:.4} vs {clean:.4}"
    );
    // …and it did it *with distance*: the whole frame is not merely brighter.
    assert!(
        changed_fraction(&img, &dry, H) > 0.1,
        "the fog changed almost nothing"
    );
}

/// **Snow at dusk.** Two claims, and the second is the interesting one: the
/// flakes are lit by the *sky* rather than by a hard-coded white, so the same
/// snow is measurably warmer at dusk than at noon. That is the single assertion
/// that would catch precipitation being shaded by a constant — the P17.3
/// `clouds_dusk` argument, one layer down.
#[test]
fn golden_weather_snow_dusk() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (mut scene, bodies) = snow_scene(71_100.0);
    assert!(
        bodies.sun.y > 0.0 && bodies.sun.y < 0.2,
        "19:45 should put the sun just above the horizon"
    );
    ground_plane(&mut scene);
    let view = horizon_view(bodies.sun, 12.0);
    let img = check_golden(&gpu, "weather_snow_dusk", &scene, &view);

    let dry = render(&gpu, &without_precip(&scene), &view);
    let snowing = changed_fraction(&img, &dry, H);
    eprintln!("snow_dusk covered {snowing:.4}");
    assert!(
        snowing > 0.01,
        "the snow drew almost nothing ({snowing:.4})"
    );

    // Snow is not rain: the same intensity at `snowiness = 0` produces a
    // different image, because the fall speed, the streak length and the flake
    // radius all move with the phase.
    let mut as_rain = scene.clone();
    as_rain.atmosphere.precip.snowiness = 0.0;
    as_rain.mark_dirty();
    let rain = render(&gpu, &as_rain, &view);
    assert_ne!(
        img, rain,
        "snowiness changed nothing — the phase is ignored"
    );

    // The colour claim. Measure only where the precipitation actually IS (pixels
    // the dry control differs from), so the sky behind it cannot carry the test.
    let warmth = |img: &[u8], base: &[u8]| {
        let (mut r, mut b, mut n) = (0.0f32, 0.0f32, 0.0f32);
        for y in 0..H {
            for x in 0..W {
                let a = px(img, x, y);
                let c = px(base, x, y);
                let d: i32 = (0..3).map(|i| (a[i] as i32 - c[i] as i32).abs()).sum();
                if d > 4 {
                    r += a[0] as f32;
                    b += a[2] as f32;
                    n += 1.0;
                }
            }
        }
        (r / n.max(1.0)) / (b / n.max(1.0)).max(1e-4)
    };
    let (mut noon_scene, noon_bodies) = snow_scene(43_200.0);
    ground_plane(&mut noon_scene);
    let noon_view = horizon_view(noon_bodies.sun, 12.0);
    let noon = render(&gpu, &noon_scene, &noon_view);
    let noon_dry = render(&gpu, &without_precip(&noon_scene), &noon_view);

    let dusk_warm = warmth(&img, &dry);
    let noon_warm = warmth(&noon, &noon_dry);
    eprintln!("snow_dusk r/b {dusk_warm:.3} vs noon {noon_warm:.3}");
    assert!(
        dusk_warm > noon_warm + 0.05,
        "dusk snow is not warmer than noon snow ({dusk_warm:.3} vs {noon_warm:.3}) \
         — the flakes are being lit by a constant, not by the sky"
    );
}

/// The **off path**, measured rather than asserted: a scene whose precipitation
/// is disabled renders **byte-identically** to one that never had a
/// `PrecipParams` at all. That is what keeps all 33 pre-P17.4 goldens intact —
/// the node returns before touching the encoder, so the command stream is the
/// one it always was.
#[test]
fn precipitation_off_is_byte_identical() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (mut scene, bodies) = storm_scene(43_200.0);
    ground_plane(&mut scene);
    let view = horizon_view(bodies.sun, 8.0);

    let mut disabled = scene.clone();
    disabled.atmosphere.precip.enabled = false;
    disabled.mark_dirty();

    // Three ways to say "no rain", all of which must produce the same pixels:
    // never configured, explicitly disabled, and zero intensity.
    let bare = render(&gpu, &without_precip(&scene), &view);
    assert_eq!(render(&gpu, &disabled, &view), bare, "disabled != absent");
    let mut zero = scene.clone();
    zero.atmosphere.precip.intensity = 0.0;
    zero.mark_dirty();
    assert_eq!(render(&gpu, &zero, &view), bare, "zero intensity != absent");

    // …and the enabled one really is different, or all three compare nothing.
    assert_ne!(render(&gpu, &scene, &view), bare);
}

/// The **depth** contract: geometry in front of the camera occludes the
/// precipitation behind it. Without a depth test the drops would hang in front
/// of the world, which is the most visible way a particle layer goes wrong.
#[test]
fn precipitation_is_occluded_by_geometry() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (mut scene, bodies) = storm_scene(43_200.0);
    ground_plane(&mut scene);
    let view = horizon_view(bodies.sun, 8.0);

    // A wall a couple of metres ahead, filling the middle of the frame. Every
    // drop beyond it must go.
    let ahead = DVec3::new(bodies.sun.x, 0.05, bodies.sun.z).normalize();
    scene.instances.push(MeshInstance::lit(
        ahead * 2.5,
        Quat::IDENTITY,
        Vec3::new(20.0, 20.0, 0.4),
        [0.5, 0.1, 0.1, 1.0],
        1,
    ));
    scene.mark_dirty();
    let walled = render(&gpu, &scene, &view);
    let walled_dry = render(&gpu, &without_precip(&scene), &view);
    // The control: the identical scene with the wall removed, so the SAME screen
    // region is measured against itself with and without an occluder. An absolute
    // threshold would only be measuring how much of the box lies in front of
    // 2.5 m; the ratio is the depth test.
    let mut open_scene = scene.clone();
    open_scene.instances.pop();
    open_scene.mark_dirty();
    let open = render(&gpu, &open_scene, &view);
    let open_dry = render(&gpu, &without_precip(&open_scene), &view);

    let centre_changed = |a: &[u8], b: &[u8]| {
        let (x0, x1) = (W * 3 / 8, W * 5 / 8);
        let (y0, y1) = (H * 3 / 8, H * 5 / 8);
        let mut n = 0u32;
        for y in y0..y1 {
            for x in x0..x1 {
                let p = px(a, x, y);
                let q = px(b, x, y);
                let d: i32 = (0..3).map(|i| (p[i] as i32 - q[i] as i32).abs()).sum();
                if d > 4 {
                    n += 1;
                }
            }
        }
        n as f32 / ((x1 - x0) * (y1 - y0)) as f32
    };
    let behind_wall = centre_changed(&walled, &walled_dry);
    let open_air = centre_changed(&open, &open_dry);
    eprintln!("precip over a 2.5 m wall: {behind_wall:.4} vs open air {open_air:.4}");
    // A wall 2.5 m into a 40 m box hides ~94 % of the depth the rain occupies, so
    // what survives is the sliver of drops genuinely in front of it. Without the
    // depth test this region would be as rained-on as the open air beside it —
    // mutation-verified by removing `depth_compare` from the pipeline, which
    // takes the ratio from ~0.2 to ~1.0.
    assert!(
        behind_wall < open_air * 0.35,
        "rain is drawing through a wall 2.5 m away ({behind_wall:.4} vs {open_air:.4} in open air)"
    );
    assert!(open_air > 0.2, "the open-air control is not raining");
}

/// The precipitation field is a pure function of the level's clock, exactly like
/// the cloud wind: two scenes at the same `time_s` are byte-identical, and one a
/// tenth of a second later is not.
///
/// Adapter-free where it can be (the offsets), on the GPU where it must be (the
/// frame), so a nondeterministic placement surfaces at whichever level it is
/// introduced.
#[test]
fn precipitation_follows_the_level_clock() {
    // The CPU half runs everywhere, including CI legs with no adapter.
    let at = |t: f64| {
        PrecipParams {
            enabled: true,
            intensity: 1.0,
            wind_x: 22.0,
            wind_z: 9.0,
            time_s: t,
            ..PrecipParams::default()
        }
        .offsets()
    };
    assert_eq!(
        at(1_234.5),
        at(1_234.5),
        "the offsets are not a pure function"
    );
    assert_ne!(at(1_234.5), at(1_234.6));

    let Some(gpu) = gpu_or_skip() else { return };
    let (mut scene, bodies) = storm_scene(43_200.0);
    ground_plane(&mut scene);
    let view = horizon_view(bodies.sun, 8.0);
    let a = render(&gpu, &scene, &view);
    let b = render(&gpu, &scene, &view);
    assert_eq!(a, b, "the precipitation pass is not deterministic");

    let mut later = scene.clone();
    later.atmosphere.precip.time_s += 0.1;
    later.mark_dirty();
    assert_ne!(render(&gpu, &later, &view), a, "the rain never fell");
}

/// The **tier** clamp reaches precipitation: a lower atmosphere quality draws
/// fewer particles, and never more. Asserted on the count (which is what the tier
/// actually controls) and confirmed on the frame, so "the tier is wired" and "the
/// tier is honoured" are separate claims.
#[test]
fn precipitation_density_follows_the_render_tier() {
    let p = PrecipParams {
        enabled: true,
        intensity: 1.0,
        ..PrecipParams::default()
    };
    let n = |q| p.count(PrecipQuality::from_atmosphere(q));
    assert!(n(AtmosphereQuality::Medium) < n(AtmosphereQuality::High));
    assert!(n(AtmosphereQuality::Low) < n(AtmosphereQuality::Medium));

    let Some(gpu) = gpu_or_skip() else { return };
    let (mut scene, bodies) = storm_scene(43_200.0);
    ground_plane(&mut scene);
    let view = horizon_view(bodies.sun, 8.0);
    // The control must be rendered at the SAME tier: a lower atmosphere quality
    // also shrinks the sky LUTs, so comparing a Low frame against a High dry one
    // measures the whole sky rather than the rain. Not hypothetical — the first
    // draft of this test reported Low drawing *thirty times* the rain of High,
    // which was the LUT difference in its entirety.
    let with = |q, precip: bool| {
        let mut s = RenderSettings::default();
        s.atmosphere.quality = q;
        let sc = if precip {
            scene.clone()
        } else {
            without_precip(&scene)
        };
        render_with(&gpu, &sc, &view, s)
    };
    let high = changed_fraction(
        &with(AtmosphereQuality::High, true),
        &with(AtmosphereQuality::High, false),
        H,
    );
    let low = changed_fraction(
        &with(AtmosphereQuality::Low, true),
        &with(AtmosphereQuality::Low, false),
        H,
    );
    eprintln!("precip coverage High {high:.4} vs Low {low:.4}");
    assert!(
        low < high,
        "the Low tier drew as much rain as High ({low:.4} vs {high:.4})"
    );
}

// ── GPU-instanced scatter (P18.5) ────────────────────────────────────────────

/// A deterministic field of scattered cubes for the scatter goldens: an `n×n`
/// jittered grid on the ground, spanning `span` metres around the origin.
///
/// Integer-hashed jitter, no `std` trig — committed golden pixels may not depend
/// on a platform's libm (the P14 LAW). Irregular rather than a lattice, because a
/// lattice lets a whole row cross an LOD boundary together and would hide exactly
/// the off-by-one an LOD golden exists to catch.
fn scatter_field(n: u32, span: f64, scale: f32, color: [f32; 4]) -> ScatterBatch {
    let step = span / n as f64;
    let mut out = Vec::with_capacity((n * n) as usize);
    for i in 0..n * n {
        let (gx, gz) = ((i % n) as f64, (i / n) as f64);
        let mut h = i.wrapping_mul(2_654_435_761);
        h ^= h >> 15;
        h = h.wrapping_mul(0x27d4_eb2d);
        let jx = ((h & 0xFFFF) as f64 / 65535.0) - 0.5;
        let jz = (((h >> 16) & 0xFFFF) as f64 / 65535.0) - 0.5;
        out.push(ScatterInstance {
            position: DVec3::new(
                (gx - (n as f64 - 1.0) * 0.5 + jx * 0.7) * step,
                scale as f64 * 0.5,
                (gz - (n as f64 - 1.0) * 0.5 + jz * 0.7) * step,
            ),
            rotation: Quat::IDENTITY,
            scale: Vec3::splat(scale),
            color,
        });
    }
    ScatterBatch::lit(
        Arc::new(ScatterData::build(PrimMesh::Cube, DVec3::ZERO, out)),
        DVec3::ZERO,
        0.85,
        90,
    )
}

fn scatter_scene() -> RenderScene {
    let mut scene = RenderScene {
        scatter: vec![scatter_field(28, 44.0, 0.9, [0.24, 0.52, 0.20, 1.0])],
        lights: vec![RenderLight {
            kind: LightKind::Directional,
            direction: Vec3::new(-0.4, 0.78, -0.48).normalize(),
            color: [1.0, 0.96, 0.88],
            intensity: 3.2,
            ..Default::default()
        }],
        ..Default::default()
    };
    // A ground plane so the field reads as ground cover rather than as floating
    // boxes, and so the frame has something for the scatter to occlude against.
    scene.instances.push(MeshInstance::lit(
        DVec3::new(0.0, -0.05, 0.0),
        Quat::IDENTITY,
        Vec3::new(120.0, 0.1, 120.0),
        [0.20, 0.17, 0.13, 1.0],
        1,
    ));
    scene.mark_dirty();
    scene
}

/// **Scatter golden** (P18.5): a GPU-instanced field drawn entirely from the
/// compacted visible list. The camera sits inside the full-mesh band, so this
/// golden pins the mesh half of the path — the cull, the prefix-sum compaction,
/// the vertex-pulled indirect draw and the PBR shading.
#[test]
fn golden_scatter() {
    let Some(gpu) = gpu_or_skip() else { return };
    let scene = scatter_scene();
    let view = look_view(DVec3::new(0.0, 7.0, -26.0), DVec3::new(0.0, 0.5, 4.0));
    let mut settings = RenderSettings::default();
    settings.scatter.mesh_distance_m = 200.0;
    settings.scatter.cull_distance_m = 400.0;

    let img = check_golden_with(&gpu, "scatter", &scene, &view, settings);
    // Green ground cover, and a lot of it.
    let green = img
        .chunks(4)
        .filter(|p| p[1] > p[0] + 4 && p[1] > p[2] + 4)
        .count();
    assert!(
        green > 1_500,
        "expected the scatter field to cover the frame ({green} px)"
    );
}

/// **Scatter impostor golden** (P18.5): the same field with the mesh band pulled
/// in so the far half resolves to camera-facing impostor discs, with a dithered
/// cross-fade between the two.
///
/// It is a separate golden rather than a variant assertion because the impostor is
/// *different geometry* — one 6-vertex billboard per instance out of the second
/// indirect draw — and a golden that never rendered one would leave the whole
/// second half of the path unpinned.
#[test]
fn golden_scatter_impostors() {
    let Some(gpu) = gpu_or_skip() else { return };
    let scene = scatter_scene();
    let view = look_view(DVec3::new(0.0, 7.0, -26.0), DVec3::new(0.0, 0.5, 4.0));
    let mut settings = RenderSettings::default();
    settings.scatter.mesh_distance_m = 22.0;
    settings.scatter.cull_distance_m = 90.0;
    settings.scatter.fade_band_m = 8.0;

    let img = check_golden_with(&gpu, "scatter_impostors", &scene, &view, settings);
    let green = img
        .chunks(4)
        .filter(|p| p[1] > p[0] + 4 && p[1] > p[2] + 4)
        .count();
    assert!(
        green > 800,
        "expected impostors to cover the far field ({green} px)"
    );

    // The banding is real: the same scene with impostors reaching further draws a
    // measurably different frame. (What the golden pins is the *look*; this is the
    // anti-vacuity claim that the band setting reached the GPU at all.)
    let mut wide = settings;
    wide.scatter.mesh_distance_m = 200.0;
    let far = render_with(&gpu, &scene, &view, wide);
    assert!(
        changed_fraction(&img, &far, H) > 0.01,
        "the impostor band made no difference to the frame"
    );
}

/// The scatter path is **inert** on a scene that carries none: every knob wound to
/// a non-default value must leave a scatter-free frame byte-identical.
///
/// This is the machine-checked half of "all 39 goldens are byte-identical" — the
/// off-path discipline P18.4's `gi_v2_off_path_is_byte_identical` established,
/// applied to the pass this batch adds.
#[test]
fn scatter_off_path_is_byte_identical() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mut scene = RenderScene {
        grid_enabled: true,
        ..Default::default()
    };
    scene.instances.push(MeshInstance::lit(
        DVec3::ZERO,
        Quat::IDENTITY,
        Vec3::splat(1.0),
        [0.80, 0.20, 0.20, 1.0],
        1,
    ));
    scene.mark_dirty();
    let view = overlook_view();
    let base = render_with(&gpu, &scene, &view, RenderSettings::default());

    let mut wound = RenderSettings::default();
    wound.scatter.gpu = false;
    wound.scatter.impostors = false;
    wound.scatter.occlusion = false;
    wound.scatter.frustum_cull = false;
    wound.scatter.mesh_distance_m = 1.0;
    wound.scatter.cull_distance_m = 2.0;
    wound.scatter.fade_band_m = 0.5;
    let other = render_with(&gpu, &scene, &view, wound);
    assert_eq!(
        base, other,
        "a scene with no scatter batches must be untouched by every scatter knob"
    );
}

// ── P20.1 water (oceans, lakes, spline rivers) ───────────────────────────────
//
// Three scenes, one per body kind, each built from `WaterBody` component
// defaults mapped the way both projectors map them — so what these images show
// is what a level actually gets when you add water, not a hand-tuned demo of the
// shader.
//
// The clock is a FIXED `time_s` in every one of them. Everything the surface
// does is a pure function of that number (the phase each wave arrives with is
// `wrap(φ − ωt + k·(d·origin))`, computed in f64 on the CPU), so a golden
// pictures one instant of a moving sea and is exactly reproducible — the same
// choice `weather_scene` makes for cloud drift.
//
// **None of the 42 pre-P20.1 goldens moves.** The water node returns before
// touching the encoder on a scene with no `waters`, so those frames record the
// command stream they always did; verified the house way — the whole suite under
// `INF_BLESS_GOLDENS=1`, `git status` reporting only the three new PNGs — and
// pinned from the inside by `water_off_path_is_byte_identical`.

/// The sea state a `WaterBody::default()` produces, mapped exactly as
/// `project_water` maps it: amplitude 0.6 m over a 40 m swell, steepness 0.5,
/// four components, a 45° spread about a 6 m/s wind from +X, seed 0.
fn default_ocean_waves() -> WaveField {
    WaveField::from_spec(&WaveSpec {
        amplitude_m: 0.6,
        wavelength_m: 40.0,
        steepness: 0.5,
        wind_x: 6.0,
        wind_z: 0.0,
        spread_rad: 45f64.to_radians(),
        seed: 0,
        count: 4,
    })
}

/// A water body at the component defaults, with the fixed clock every water
/// golden uses.
fn water_body(kind: WaterKindGpu, waves: WaveField) -> RenderWater {
    RenderWater {
        id: 1,
        kind,
        waves,
        // The one instant these goldens picture. A pure function of this number.
        time_s: 1_234.5,
        ..RenderWater::default()
    }
}

/// The ocean scene: a default sea at `level_m`, under the default sky authority
/// at `seconds`, with a hill terrain rising out of it so the shore blend, the
/// absorption and the shallow-water foam all have something to act against. An
/// ocean with nothing in it is a picture of a Fresnel term.
fn ocean_scene(seconds: f64, level_m: f64) -> RenderScene {
    let (mut scene, _) = tod_scene(seconds);
    scene.grid_enabled = false;
    scene.terrains = vec![hill_terrain(33, 4.0, 2, 2)];
    scene.waters = vec![RenderWater {
        level_m,
        ..water_body(WaterKindGpu::Ocean, default_ocean_waves())
    }];
    scene
}

/// The same scene with the water removed — the control every water golden is
/// measured against, and the frame the off-path test asserts is byte-identical
/// to the pre-P20.1 renderer.
fn without_water(scene: &RenderScene) -> RenderScene {
    let mut s = scene.clone();
    s.waters.clear();
    s
}

/// How much of the frame two renders disagree about, as a fraction of pixels
/// whose sRGB byte differs by more than 6/255 in any channel. Adapter-robust in
/// a way an absolute pixel comparison is not.
fn water_changed_fraction(a: &[u8], b: &[u8]) -> f64 {
    let mut n = 0usize;
    for i in 0..(W * H) as usize {
        let (pa, pb) = (&a[i * 4..i * 4 + 3], &b[i * 4..i * 4 + 3]);
        if pa
            .iter()
            .zip(pb)
            .any(|(x, y)| (*x as i32 - *y as i32).abs() > 6)
        {
            n += 1;
        }
    }
    n as f64 / (W * H) as f64
}

/// **GOLDEN — the boat reflects itself** (wave VIS1a audit): a scarlet block on
/// an open sea with screen-space reflections on, from five metres above the
/// surface.
///
/// **This golden exists because the wave said it could not.** VIS1a's ledger
/// carried "no golden can ever capture SSR — a golden renders one frame from a
/// fresh renderer", which is true of the **opaque** path and only of it: that one
/// samples the previous frame's resolve, so `scene_history_valid` holds its march
/// off on frame 0. **Water's does not.** The water pass runs its own
/// `color_msaa → scene_hdr` resolve before it draws, so `water_ssr` marches
/// against *this* frame's colour and depth and needs no history at all — which is
/// exactly what `water_reflects_the_scene_from_above_and_defers_to_the_sky_at_grazing`
/// demonstrates by measuring it with a single-frame `render_with`. A feature that
/// can be rendered in one deterministic frame can be pinned as pixels, and the
/// wave's signature effect had no pixel pin.
///
/// The scene is the sibling arm's, deliberately: that one measures *how much* red
/// arrives and *where it must not*, and this one fixes what the frame looks like
/// while it does.
#[test]
fn golden_water_ssr() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mut scene = ocean_scene(12.0 * 3600.0, 3.0);
    scene.terrains.clear();
    scene.instances.push(MeshInstance::lit(
        DVec3::new(0.0, 4.5, 0.0),
        Quat::IDENTITY,
        Vec3::new(6.0, 3.0, 6.0),
        [0.95, 0.05, 0.05, 1.0],
        1,
    ));
    scene.mark_dirty();
    let ssr_on = RenderSettings {
        ssr: inf_render::SsrSettings {
            enabled: true,
            ..inf_render::SsrSettings::default()
        },
        ..RenderSettings::default()
    };
    let view = look_view(DVec3::new(0.0, 8.0, 24.0), DVec3::new(0.0, 3.0, -2.0));
    let img = check_golden_with(&gpu, "water_ssr", &scene, &view, ssr_on);

    // The structural half, for the CI legs that do not compare pixels strictly:
    // the reflection is really in THIS frame, not merely in the committed one.
    // A ratio against the same frame with SSR off, so it survives a different
    // rasterizer.
    let off = render_with(&gpu, &scene, &view, RenderSettings::default());
    let redness = |i: &[u8]| -> f64 {
        i.chunks(4)
            .map(|p| p[0] as f64 - 0.5 * (p[1] as f64 + p[2] as f64))
            .sum::<f64>()
            / (W * H) as f64
    };
    let (r_off, r_on) = (redness(&off), redness(&img));
    eprintln!("golden water_ssr: redness {r_off:.3} -> {r_on:.3}");
    assert!(
        r_on > r_off + 0.5,
        "the committed frame carries no reflection (redness {r_off:.3} -> \
         {r_on:.3}) — the golden would be pinning the sky term alone"
    );
}

/// **THE BOAT REFLECTS ITSELF** (wave VIS1a) — the P20.3 routed item, closed,
/// and the measurement that makes closing it safe.
///
/// P20.1 ruled water's reflection to be the sky and only the sky: *"a
/// wave-perturbed normal at the grazing angles that dominate a water surface
/// reflects toward the horizon, which is exactly where a screen-space march has
/// nothing to hit."* Every clause of that is still true. What it could not
/// distinguish is the pixel looking *down* into the water — the one under a hull,
/// against a jetty, with a cliff standing in it — from the grazing one, and
/// `water_grazing_weight(cos_v)` is that distinction.
///
/// So this arm is two measurements, not one:
///
/// * **from above**, a scarlet block floating on the sea must appear in the water
///   beside it — the change SSR makes, and the direction it makes it in;
/// * **from a grazing angle**, the SAME scene must change far less, because that
///   is the regime the original ruling describes and the regime the fade defers
///   to the sky in. Without this half, "SSR on water" would be a claim that the
///   ruling was simply wrong.
#[test]
fn water_reflects_the_scene_from_above_and_defers_to_the_sky_at_grazing() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mut scene = ocean_scene(12.0 * 3600.0, 3.0);
    scene.terrains.clear(); // open water, so nothing but the block can reflect
                            // A scarlet block sitting in the sea. Deliberately a colour nothing else in
                            // the frame carries: the sky is blue, the water is blue-green, and the
                            // measurement below is "did red arrive".
    scene.instances.push(MeshInstance::lit(
        DVec3::new(0.0, 4.5, 0.0),
        Quat::IDENTITY,
        Vec3::new(6.0, 3.0, 6.0),
        [0.95, 0.05, 0.05, 1.0],
        1,
    ));
    scene.mark_dirty();

    let ssr_on = RenderSettings {
        ssr: inf_render::SsrSettings {
            enabled: true,
            ..inf_render::SsrSettings::default()
        },
        ..RenderSettings::default()
    };

    // The frame's overall redness. Measured over the WHOLE image on purpose: the
    // block itself renders identically in both, so anything that moves this
    // number moved on the water.
    let redness = |img: &[u8]| -> f64 {
        let mut sum = 0.0f64;
        for p in img.chunks(4) {
            sum += p[0] as f64 - 0.5 * (p[1] as f64 + p[2] as f64);
        }
        sum / (W * H) as f64
    };

    // ── the reflection shot ──────────────────────────────────────────────────
    // An eye five metres above the surface, twenty-four out: the water between
    // the camera and the block reflects it, which is the picture the P20.3 item
    // names. `cos_v` there is about 0.3 — well inside the ordinary-viewing
    // regime, and well away from the horizon case the ruling was about.
    let above = look_view(DVec3::new(0.0, 8.0, 24.0), DVec3::new(0.0, 3.0, -2.0));
    let off = render_with(&gpu, &scene, &above, RenderSettings::default());
    let on = render_with(&gpu, &scene, &above, ssr_on);
    let (r_off, r_on) = (redness(&off), redness(&on));
    let above_change = water_changed_fraction(&off, &on);
    eprintln!(
        "water SSR, eye 5 m above the surface: redness {r_off:.3} -> {r_on:.3}, \
         {:.3} % of the frame moved",
        above_change * 100.0
    );
    assert!(
        r_on > r_off + 0.5,
        "the block did not arrive in the water in front of it (redness {r_off:.3} \
         -> {r_on:.3}) — the march found nothing, or the grazing fade is holding \
         it off at an angle it should not"
    );

    // ── the grazing control ──────────────────────────────────────────────────
    // The same sea, the same block, at the same distance, with the eye 60 cm
    // above the waterline instead of five metres. This is the regime P20.1
    // described — `cos_v` in the hundredths — and it must still be the sky's.
    let grazing = look_view(DVec3::new(0.0, 3.6, 24.0), DVec3::new(0.0, 3.4, -20.0));
    let g_off = render_with(&gpu, &scene, &grazing, RenderSettings::default());
    let g_on = render_with(&gpu, &scene, &grazing, ssr_on);
    let grazing_change = water_changed_fraction(&g_off, &g_on);
    let (g_off_r, g_on_r) = (redness(&g_off), redness(&g_on));
    eprintln!(
        "water SSR at grazing: redness {g_off_r:.3} -> {g_on_r:.3} (delta {:.3} \
         against {:.3} from above), {:.3} % of the frame moved (against {:.3} %)",
        g_on_r - g_off_r,
        r_on - r_off,
        grazing_change * 100.0,
        above_change * 100.0
    );
    // **What the control is, precisely.** P20.1's objection was that at grazing
    // angles the march is *futile* — every ray leaves the frame and every pixel
    // takes the miss path, which is the sky. It was never that SSR would be
    // wrong there. So the claim to check is that the block does not arrive: the
    // scarlet that floods the reflection shot must stay out of the horizon shot.
    //
    // It is not zero, and that is honest rather than a slack threshold: a wave
    // facet tilted toward the camera has a `cos_v` of its own, and a facet facing
    // you really does reflect what is in front of it. That is the same
    // wave-perturbed normal P20.1's sentence names, read the other way round.
    assert!(
        g_on_r - g_off_r < (r_on - r_off) * 0.5,
        "the block reflected as strongly at grazing ({:.3}) as from above ({:.3}) \
         — the angle fade is not deferring to the sky where P20.1's argument holds",
        g_on_r - g_off_r,
        r_on - r_off
    );
    assert!(
        grazing_change < above_change,
        "a grazing sea moved {grazing_change:.4} of the frame against \
         {above_change:.4} from above"
    );
}

/// **GOLDEN — ocean at noon.** A Gerstner sea against a hill coast, with the sun
/// high: the frame where absorption, the shore blend and the sky reflection all
/// read at once.
#[test]
fn golden_water_ocean_noon() {
    let Some(gpu) = gpu_or_skip() else { return };
    let scene = ocean_scene(12.0 * 3600.0, 3.0);
    let view = look_view(DVec3::new(-30.0, 14.0, -30.0), DVec3::new(60.0, 0.0, 60.0));
    let img = check_golden(&gpu, "water_ocean_noon", &scene, &view);

    // The water really is in the frame: removing it moves a large part of it.
    let dry = render(&gpu, &without_water(&scene), &view);
    let moved = water_changed_fraction(&img, &dry);
    assert!(
        moved > 0.10,
        "the ocean changed only {:.1}% of the frame — is it drawing at all?",
        moved * 100.0
    );

    // …and it is WATER, not a grey plane: absorption removes red an order of
    // magnitude faster than blue, so the surface must read blue-dominant. Sampled
    // low in the frame, where the sea is, and compared as a ratio (adapter-robust).
    let c = mean_rgb(&img, W / 8, H * 5 / 8, W * 7 / 8, H);
    let (r, b) = (c[0], c[2]);
    assert!(
        b > r * 1.15,
        "the sea is not blue-dominant (r {r:.3}, b {b:.3}) — the Beer-Lambert \
         absorption is not reaching the surface"
    );
}

/// **GOLDEN — a lake at dusk.** A bounded rectangle with a gentle ripple, lit by
/// a low sun: the frame where the Fresnel term dominates and the sky reflection
/// is the whole appearance.
#[test]
fn golden_water_lake_dusk() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (mut scene, bodies) = tod_scene(20.0 * 3600.0);
    scene.grid_enabled = false;
    scene.terrains = vec![hill_terrain(33, 4.0, 2, 2)];
    // The `WaterBody::lake` preset, mapped as `project_water` maps it: a small,
    // long, low-steepness ripple rather than a sea.
    let ripple = WaveField::from_spec(&WaveSpec {
        amplitude_m: 0.05,
        wavelength_m: 7.0,
        steepness: 0.12,
        wind_x: 1.0,
        wind_z: 0.0,
        spread_rad: 45f64.to_radians(),
        seed: 0,
        count: 3,
    });
    // Level 5 m against a terrain whose hills span [-0.5, 7.5] m: the lake fills
    // the basins and the ridges stand out of it, which is the frame a shore blend
    // is actually about.
    scene.waters = vec![RenderWater {
        level_m: 5.0,
        center: glam::DVec2::new(60.0, 60.0),
        half_extent: glam::DVec2::new(55.0, 55.0),
        ..water_body(WaterKindGpu::Lake, ripple)
    }];
    let view = look_view(DVec3::new(6.0, 16.0, 6.0), DVec3::new(70.0, 3.0, 70.0));
    let img = check_golden(&gpu, "water_lake_dusk", &scene, &view);

    let dry = render(&gpu, &without_water(&scene), &view);
    let moved = water_changed_fraction(&img, &dry);
    assert!(
        moved > 0.02,
        "the lake changed only {:.1}% of the frame",
        moved * 100.0
    );

    // A lake is BOUNDED: shrinking it to nothing must return the frame to the dry
    // control exactly. That is the property an unbounded plane would fail, and it
    // is what distinguishes a lake from an ocean in one assertion.
    let mut tiny = scene.clone();
    tiny.waters[0].half_extent = glam::DVec2::ZERO;
    let none = render(&gpu, &tiny, &view);
    assert_eq!(
        none, dry,
        "a zero-extent lake still drew something — `drawable()` is not gating it"
    );

    // The sun is low, so this is the Fresnel-dominated frame by construction.
    assert!(
        bodies.sun.y < 0.35,
        "20:00 is not a low sun: {}",
        bodies.sun.y
    );
}

/// **GOLDEN — a spline river.** A ribbon following a Catmull-Rom centreline down
/// a hillside, with a width profile and flow foam. The frame that proves a river
/// is geometry derived from a spline rather than a rectangle.
#[test]
fn golden_water_river() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (mut scene, _) = tod_scene(15.0 * 3600.0);
    scene.grid_enabled = false;
    scene.terrains = vec![hill_terrain(33, 4.0, 2, 2)];

    // The `WaterBody::river` preset's ripple, and a centreline the same
    // `inf_water::RiverPath` both projectors build.
    let ripple = WaveField::from_spec(&WaveSpec {
        amplitude_m: 0.06,
        wavelength_m: 4.0,
        steepness: 0.12,
        wind_x: 1.0,
        wind_z: 0.0,
        spread_rad: 60f64.to_radians(),
        seed: 0,
        count: 3,
    });
    let points = [
        DVec3::new(4.0, 9.0, 4.0),
        DVec3::new(40.0, 7.0, 24.0),
        DVec3::new(70.0, 5.0, 60.0),
        DVec3::new(110.0, 3.0, 90.0),
    ];
    let path = inf_render::RiverPath::from_points(
        &points,
        false,
        inf_math::spline::SplineInterp::CatmullRom,
        &inf_render::RiverProfile {
            width_start_m: 6.0,
            width_end_m: 14.0,
            depth_start_m: 1.0,
            depth_end_m: 2.5,
            flow_speed_m_s: 3.5,
        },
    );
    scene.waters = vec![RenderWater {
        level_m: 9.0,
        flow_speed_m_s: path.flow_speed_m_s,
        frames: path
            .frames
            .iter()
            .map(inf_render::WaterFrame::from)
            .collect(),
        ..water_body(WaterKindGpu::River, ripple)
    }];
    let view = look_view(DVec3::new(-10.0, 30.0, 10.0), DVec3::new(70.0, 0.0, 60.0));
    let img = check_golden(&gpu, "water_river", &scene, &view);

    let dry = render(&gpu, &without_water(&scene), &view);
    let moved = water_changed_fraction(&img, &dry);
    assert!(
        moved > 0.01,
        "the river changed only {:.2}% of the frame",
        moved * 100.0
    );

    // A river is a RIBBON: widening its profile must widen what it covers. That
    // separates "the spline was used" from "a rectangle was drawn somewhere".
    let mut wide = scene.clone();
    let wide_path = inf_render::RiverPath::from_points(
        &points,
        false,
        inf_math::spline::SplineInterp::CatmullRom,
        &inf_render::RiverProfile {
            width_start_m: 18.0,
            width_end_m: 30.0,
            depth_start_m: 1.0,
            depth_end_m: 2.5,
            flow_speed_m_s: 3.5,
        },
    );
    wide.waters[0].frames = wide_path
        .frames
        .iter()
        .map(inf_render::WaterFrame::from)
        .collect();
    let wider = render(&gpu, &wide, &view);
    let a = water_changed_fraction(&img, &dry);
    let b = water_changed_fraction(&wider, &dry);
    assert!(
        b > a * 1.3,
        "widening the river's profile barely changed its coverage ({:.2}% → \
         {:.2}%) — the ribbon is not following the width profile",
        a * 100.0,
        b * 100.0
    );

    // …and a river with no frames draws nothing at all (the no-`Spline`
    // authoring state), returning the frame to the dry control exactly.
    let mut unrouted = scene.clone();
    unrouted.waters[0].frames.clear();
    assert_eq!(render(&gpu, &unrouted, &view), dry);
}

/// **The off-path gate** (the P18.4 `gi_v2_off_path_is_byte_identical` shape,
/// applied to water).
///
/// This is the machine-checked half of "all 42 pre-P20.1 goldens are
/// byte-identical": a scene with no `waters` must render the exact same bytes
/// however hard the new knobs are wound, because the node returns before it
/// records a resolve, a render pass, a pipeline bind or a draw.
///
/// Both directions are checked. Winding the water QUALITY on a water-free scene
/// must be inert (the settings half), and so must every per-body knob on a scene
/// whose only body is not drawable (the content half) — the second is what stops
/// a future refactor from "helpfully" drawing a zero-extent lake.
#[test]
fn water_off_path_is_byte_identical() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (mut scene, _) = tod_scene(12.0 * 3600.0);
    scene.grid_enabled = false;
    scene.terrains = vec![hill_terrain(33, 4.0, 2, 2)];
    let view = look_view(DVec3::new(-30.0, 14.0, -30.0), DVec3::new(60.0, 0.0, 60.0));

    let base = render_with(&gpu, &scene, &view, RenderSettings::default());

    // (1) Every water setting wound, on a scene that carries no water.
    for quality in [WaterQuality::Low, WaterQuality::Medium, WaterQuality::High] {
        let mut fiddled = RenderSettings::default();
        fiddled.water.quality = quality;
        assert_eq!(
            base,
            render_with(&gpu, &scene, &view, fiddled),
            "a water-free scene moved when the P20.1 quality knob changed to \
             {quality:?} — the off path is not neutral"
        );
    }

    // (2) A scene carrying a body that is NOT drawable, with every per-body knob
    // wound to a non-default value. Nothing may reach the frame.
    let mut lake = scene.clone();
    lake.waters = vec![RenderWater {
        kind: WaterKindGpu::Lake,
        half_extent: glam::DVec2::ZERO, // ⇒ !drawable()
        level_m: 5.0,
        waves: default_ocean_waves(),
        time_s: 999.0,
        flow_speed_m_s: 4.0,
        shallow_color: [1.0, 0.0, 0.0],
        deep_color: [0.0, 1.0, 0.0],
        absorption: [0.0, 0.0, 0.0],
        roughness: 1.0,
        refraction_m: 5.0,
        shore_fade_m: 20.0,
        opacity: 1.0,
        foam_color: [1.0, 0.0, 1.0],
        foam_crest_threshold: 0.0,
        foam_shore_m: 50.0,
        foam_flow_m_s: 0.1,
        ..RenderWater::default()
    }];
    assert!(!lake.waters[0].drawable(), "the fixture must be undrawable");
    assert_eq!(
        base,
        render_with(&gpu, &lake, &view, RenderSettings::default()),
        "an undrawable water body reached the frame"
    );

    // …and a river with a single frame is the other undrawable shape.
    let mut river = scene.clone();
    river.waters = vec![RenderWater {
        kind: WaterKindGpu::River,
        frames: vec![inf_render::WaterFrame {
            center: DVec3::new(10.0, 5.0, 10.0),
            tangent: DVec3::X,
            right: DVec3::Z,
            s: 0.0,
            width_m: 8.0,
            depth_m: 2.0,
            flow_gain: 1.0,
        }],
        ..RenderWater::default()
    }];
    assert!(!river.waters[0].drawable());
    assert_eq!(
        base,
        render_with(&gpu, &river, &view, RenderSettings::default())
    );

    // A guard on the guard: with a REAL body the frame does move, so the
    // assertions above are not passing vacuously.
    let mut wet = scene.clone();
    wet.waters = vec![RenderWater {
        level_m: 3.0,
        ..water_body(WaterKindGpu::Ocean, default_ocean_waves())
    }];
    assert_ne!(
        base,
        render_with(&gpu, &wet, &view, RenderSettings::default()),
        "water that IS drawable changed nothing — every assertion above is vacuous"
    );
}

/// The quality tier really reaches the GPU: the same sea rendered at three tiers
/// must differ, and the Low tier must differ *most* (it drops both the
/// tessellation and screen-space refraction).
///
/// The inverse of the off-path test, and needed for the same reason
/// `gi_quality_switch_rebuilds_the_env_bind` is: a tier that silently did nothing
/// would satisfy every neutrality assertion ever written.
#[test]
fn water_quality_reaches_the_gpu() {
    let Some(gpu) = gpu_or_skip() else { return };
    let scene = ocean_scene(12.0 * 3600.0, 3.0);
    let view = look_view(DVec3::new(-30.0, 14.0, -30.0), DVec3::new(60.0, 0.0, 60.0));

    let at = |q: WaterQuality| {
        let mut s = RenderSettings::default();
        s.water.quality = q;
        render_with(&gpu, &scene, &view, s)
    };
    let high = at(WaterQuality::High);
    let medium = at(WaterQuality::Medium);
    let low = at(WaterQuality::Low);

    assert_ne!(high, medium, "Medium rendered identically to High");
    assert_ne!(medium, low, "Low rendered identically to Medium");
    // Low drops refraction as well as tessellation, so it is the furthest from
    // High — a tier that only halved the grid would fail this.
    let d_med = water_changed_fraction(&high, &medium);
    let d_low = water_changed_fraction(&high, &low);
    assert!(
        d_low > d_med,
        "Low ({:.2}%) is not further from High than Medium ({:.2}%) — is the \
         refraction clamp reaching the shader?",
        d_low * 100.0,
        d_med * 100.0
    );
}

/// **Determinism, and the clock.** The sea is a pure function of `time_s`: two
/// renders at the same clock are identical, and two at different clocks are not.
///
/// This is the property that lets a golden picture one instant of a moving
/// surface, and it is the same property P20.2's replay gate will rest on.
#[test]
fn the_sea_is_a_pure_function_of_the_level_clock() {
    let Some(gpu) = gpu_or_skip() else { return };
    let scene = ocean_scene(12.0 * 3600.0, 3.0);
    let view = look_view(DVec3::new(-30.0, 14.0, -30.0), DVec3::new(60.0, 0.0, 60.0));

    let a = render(&gpu, &scene, &view);
    let b = render(&gpu, &scene, &view);
    assert_eq!(a, b, "the same clock rendered two different seas");

    let mut later = scene.clone();
    later.waters[0].time_s += 3.0;
    assert_ne!(
        a,
        render(&gpu, &later, &view),
        "advancing the level clock moved no water — the wave phase is not \
         reaching the shader"
    );

    // …and it really is the CLOCK, not the frame counter: rendering the same
    // clock twice in a row after a different one still reproduces the original.
    assert_eq!(a, render(&gpu, &scene, &view));
}

/// The **wind response**, end to end: a stronger wind raises a bigger sea, and it
/// does so through the same `WaveField::from_spec` the sim will sample.
#[test]
fn a_stronger_wind_raises_a_bigger_sea() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mut scene = ocean_scene(12.0 * 3600.0, 3.0);
    let view = look_view(DVec3::new(-30.0, 8.0, -30.0), DVec3::new(60.0, 0.0, 60.0));

    let calm = WaveField::from_spec(&WaveSpec {
        wind_x: 0.0,
        wind_z: 0.0,
        ..WaveSpec {
            amplitude_m: 0.6,
            wavelength_m: 40.0,
            steepness: 0.5,
            wind_x: 0.0,
            wind_z: 0.0,
            spread_rad: 45f64.to_radians(),
            seed: 0,
            count: 4,
        }
    });
    let gale = WaveField::from_spec(&WaveSpec {
        amplitude_m: 0.6,
        wavelength_m: 40.0,
        steepness: 0.5,
        wind_x: 20.0,
        wind_z: 0.0,
        spread_rad: 45f64.to_radians(),
        seed: 0,
        count: 4,
    });
    // The Ring-0 claim, restated here so a shader change cannot quietly decouple
    // the picture from the model.
    assert!(gale.max_amplitude_m() > calm.max_amplitude_m() * 2.0);

    scene.waters[0].waves = calm;
    let calm_img = render(&gpu, &scene, &view);
    scene.waters[0].waves = gale;
    let gale_img = render(&gpu, &scene, &view);
    assert_ne!(calm_img, gale_img, "the wind moved nothing");
}

// ── P20.3 underwater post & shoreline wetness ────────────────────────────────
//
// Two new goldens (a submerged camera, a wet shoreline) and the tests that carry
// the claims a picture cannot: that the fog is the SAME absorption the surface
// applies from above, that the light shafts really reach the frame, and that the
// wet band is terrain shading driven by the water rather than a second way of
// drawing water.
//
// **The three P20.1 water goldens moved, deliberately.** Wetness is default-on
// and all three carry `hill_terrain`, so the ground at and below their water
// levels is now darker and glossier — which is the feature. They were re-blessed
// in the same single-package pass that wrote the two new PNGs; the delta is
// described in the P20.3 commit and in the ROADMAP block. Every other golden is
// byte-identical: the underwater node returns before touching the encoder above
// the waterline, and `wet.dims.x` is 0 on a scene with no water.

/// A basin floor of `hill_terrain(33, 4.0, …)`. Its height is
/// `4·sin(0.15x)·cos(0.15z) + 3.5`, which bottoms out at −0.5 m where
/// `sin(0.15x) = −1` and `cos(0.15z) = 1` — i.e. at `x = 31.416 + 41.888k`,
/// `z = 41.888k`. Both scenes below are built around a known-height point so the
/// camera's submersion is arithmetic rather than a guess.
const BASIN: (f64, f64) = (198.97, 125.66);

/// A sea deep enough to put a camera under, over the same hill coast the P20.1
/// water goldens use, plus the view that looks up at the sun through it.
///
/// The sun at 17:00 (day 172, 48.9°N) sits 27° above the horizon on a bearing of
/// roughly −X; flattening its elevation by half puts the camera at a 14.5° pitch,
/// which keeps the disc on screen (so the shafts have somewhere to converge) while
/// the sea floor still fills the bottom of the frame.
fn underwater_scene(level_m: f64, eye_y: f64) -> (RenderScene, RenderView) {
    let scene = ocean_scene(17.0 * 3600.0, level_m);
    let sun = scene.sun.unit_direction().as_dvec3();
    let eye = DVec3::new(BASIN.0, eye_y, BASIN.1);
    let target = eye + DVec3::new(sun.x, sun.y * 0.5, sun.z) * 60.0;
    (scene, look_view(eye, target))
}

/// Frame rows showing the NEAR sea floor in [`underwater_scene`]'s shot, and the
/// FAR one.
///
/// The view is pitched 14.5° up in a 60° vertical FOV, so the horizon sits at row
/// ~134 and everything below it looks down at the floor: rows 164–180 are ~20 m
/// of water away, rows 138–150 hundreds. Everything ABOVE row 134 is water
/// surface — which is why neither band goes there. The surface is what the
/// quality tier changes, and these bands exist to be free of it.
const FLOOR_NEAR: (u32, u32) = (164, 180);
const FLOOR_FAR: (u32, u32) = (138, 150);

/// Frame rows looking UP through the surface — Snell's window. Rows 90–120 rise
/// at 4.5°–14.5°, so they leave the water after 16–50 m, where [`FLOOR_FAR`] just
/// below them crosses hundreds of metres of it.
const WINDOW: (u32, u32) = (90, 120);

/// A 60°-down look into the basin at [`BASIN`], from 26 m up and 15 m back.
///
/// The steep pitch is load-bearing for the two wetness isolation tests, not a
/// composition choice: the top edge of the frame is 30° BELOW the horizontal, so
/// the furthest ground in shot is ~46 m away. That keeps the whole frame inside
/// the 256 m terrain — no sky, no horizon, and (the point) **no view past the
/// terrain's edge to the open ocean beyond it**, which is what a shallower angle
/// gives you and what would put water pixels in a frame that is supposed to have
/// none.
fn basin_view() -> RenderView {
    look_view(
        DVec3::new(BASIN.0, 26.0, BASIN.1 + 15.0),
        DVec3::new(BASIN.0, 0.0, BASIN.1),
    )
}

/// Mean relative luminance of a rect, `[0, 1]` from sRGB bytes.
fn mean_luma(img: &[u8], x0: u32, y0: u32, x1: u32, y1: u32) -> f32 {
    let c = mean_rgb(img, x0, y0, x1, y1);
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

/// **GOLDEN — a camera under an ocean.** Four metres down in a hill-coast basin
/// at 17:00, looking up at the sun through the surface: the frame where the
/// depth-graded absorption fog and the v1 surface light shafts both read.
#[test]
fn golden_water_underwater_ocean() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (scene, view) = underwater_scene(15.0, 11.0);

    // The camera really is submerged — and the thing that says so is the Ring-0
    // evaluator P20.2's buoyancy samples, not a second surface implementation.
    let under = inf_render::camera_underwater(&scene.waters, view.eye_world)
        .expect("the fixture camera must be under the sea");
    assert!(
        (under.depth_m - 4.0).abs() < 1.0,
        "the fixture is {} m down, not ~4",
        under.depth_m
    );
    assert_eq!(under.strength, 1.0, "well past the waterline ramp");

    let img = check_golden(&gpu, "water_underwater_ocean", &scene, &view);

    // (1) The whole frame is affected, which no surface-only pass can do: the
    // P20.1 ocean golden, seen from above, moves 10–40 % of its frame. Part of
    // this is the P20.3 wet band — under a 15 m sea every terrain sample here is
    // submerged — which is exactly why the sharper claim is (2).
    let dry = render(&gpu, &without_water(&scene), &view);
    let moved = water_changed_fraction(&img, &dry);
    assert!(
        moved > 0.85,
        "the underwater treatment covered only {:.1}% of the frame",
        moved * 100.0
    );

    // (2) THE FOG IS RUNNING, isolated from the wet band. Wetness is a scalar
    // multiply on albedo, so it cannot move a HUE; absorption is per-channel and
    // removes red an order of magnitude faster than blue, so hue is exactly what
    // it moves. Measured on the near sea floor — opaque geometry in both frames,
    // and the one part of this shot that carries no water surface.
    let ratio = |i: &[u8]| {
        let c = mean_rgb(i, 0, FLOOR_NEAR.0, W, FLOOR_NEAR.1);
        c[2] / c[0].max(1e-4)
    };
    assert!(
        ratio(&img) > ratio(&dry) * 1.5,
        "the sea floor's blue/red ratio is {:.2} submerged and {:.2} dry — the \
         Beer-Lambert absorption is not reaching the fog",
        ratio(&img),
        ratio(&dry)
    );

    // (3) THE SURFACE CAPS THE COLUMN. A ray that rises out of the medium stops
    // absorbing at the surface, so rows looking UP through it (16–50 m of water)
    // must carry visibly less than the rows just below them looking level and
    // down (hundreds). Without the cap both bands take the far-field column and
    // this margin collapses — which is what would fog away the bright disc
    // overhead and leave the shafts with nothing to come out of.
    let blue = |rows: (u32, u32)| mean_rgb(&img, 0, rows.0, W, rows.1)[2];
    assert!(
        blue(WINDOW) > blue(FLOOR_FAR) * 1.25,
        "looking up carries {:.3} of blue and looking down {:.3} — the water \
         surface is not capping the column",
        blue(WINDOW),
        blue(FLOOR_FAR)
    );
}

/// **The one-absorption-story gate, in pixels.** The fog's inscattered term is
/// weighted by `1 − exp(−a·column)`, so it must grow with the length of water in
/// front of a pixel — the far sea floor carries more medium than the near one.
///
/// Isolated by *removing the medium's colour* rather than by comparing two bands
/// directly. Two renders that differ only in `deep_color` are bit-identical in
/// their terrain, haze, sun, shafts and wet band, so the difference at a pixel is
/// exactly the medium's contribution there — a function of the column and of
/// nothing else. Comparing two bands of one frame would instead be comparing two
/// pieces of terrain.
#[test]
fn the_underwater_fog_is_graded_by_the_water_column() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (scene, view) = underwater_scene(15.0, 11.0);
    let base = render(&gpu, &scene, &view);
    let mut colourless = scene.clone();
    colourless.waters[0].deep_color = [0.0, 0.0, 0.0];
    let dark = render(&gpu, &colourless, &view);

    // Blue is the channel that can tell the two bands apart: at 0.035 m⁻¹ the
    // near floor (~20 m off) is still half transmissive while the far field has
    // saturated. Red (0.45 m⁻¹) has saturated in both and would compare equal —
    // a fact about the water, not a weakness of the test.
    let blue = |i: &[u8], rows: (u32, u32)| mean_rgb(i, 0, rows.0, W, rows.1)[2];
    let d_near = blue(&base, FLOOR_NEAR) - blue(&dark, FLOOR_NEAR);
    let d_far = blue(&base, FLOOR_FAR) - blue(&dark, FLOOR_FAR);
    assert!(
        d_near > 0.005,
        "the medium contributes nothing even close to the camera ({d_near:.4}) — \
         the comparison is vacuous"
    );
    assert!(
        d_far > d_near * 1.15,
        "the far water carries {d_far:.4} of medium and the near water \
         {d_near:.4} — the fog is a flat tint, not a column"
    );
}

/// The waterline is the switch, and crossing it is the only thing that engages
/// the pass.
///
/// Above the surface, water is a *surface*: it changes the part of the frame it
/// covers. Below it, water is a *medium*: it changes all of it. The two fractions
/// must therefore differ by a wide margin, and the Ring-0 evaluator must agree
/// with the pixels about which side of the line the camera is on.
#[test]
fn the_underwater_pass_engages_only_below_the_waterline() {
    let Some(gpu) = gpu_or_skip() else { return };
    let level = 15.0;
    let (scene, below_view) = underwater_scene(level, 11.0);
    // The same shot from above the highest crest.
    let crest = scene.waters[0].waves.max_amplitude_m();
    let (_, above_view) = underwater_scene(level, level + crest + 2.0);

    assert!(inf_render::camera_underwater(&scene.waters, below_view.eye_world).is_some());
    assert!(
        inf_render::camera_underwater(&scene.waters, above_view.eye_world).is_none(),
        "a camera above every crest was reported submerged"
    );

    let dry = without_water(&scene);
    let below = water_changed_fraction(
        &render(&gpu, &scene, &below_view),
        &render(&gpu, &dry, &below_view),
    );
    let above = water_changed_fraction(
        &render(&gpu, &scene, &above_view),
        &render(&gpu, &dry, &above_view),
    );
    assert!(
        below > 0.85,
        "submerged, the water changed only {:.1}% of the frame",
        below * 100.0
    );
    assert!(
        above < below * 0.9,
        "from above the waterline the water changed {:.1}% of the frame and from \
         below {:.1}% — the post pass is running on the wrong side of the line",
        above * 100.0,
        below * 100.0
    );
}

/// **The light shafts reach the frame**, isolated from everything else the
/// quality tier does.
///
/// [`WaterQuality::Low`] drops the shafts — and also the surface tessellation and
/// the screen-space refraction. Both of those only ever touch *water-surface*
/// pixels, and this camera sits four metres BELOW its ocean, so no downward ray
/// can reach the surface: the bottom band of the frame contains no surface pixels
/// at all. A difference there is the shafts and nothing else, and because they
/// are additive it must be a brightening.
#[test]
fn underwater_light_shafts_reach_the_frame() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (scene, view) = underwater_scene(15.0, 11.0);
    let at = |q: WaterQuality| {
        let mut s = RenderSettings::default();
        s.water.quality = q;
        render_with(&gpu, &scene, &view, s)
    };
    let high = at(WaterQuality::High);
    let low = at(WaterQuality::Low);
    assert_ne!(high, low, "the quality tier reached nothing");

    let band = |i: &[u8]| mean_luma(i, 0, FLOOR_NEAR.0, W, FLOOR_NEAR.1);
    assert!(
        band(&high) > band(&low),
        "the sea floor is no brighter with the shafts on ({:.4} vs {:.4}) — the \
         shaft term is not reaching the frame",
        band(&high),
        band(&low)
    );
}

/// **THE P20.3 OFF-PATH GATE.** Drawable water in the scene, camera above it ⇒
/// the underwater node records **nothing at all**.
///
/// This is a claim about the *command stream*, and the reason it is asserted with
/// a counter rather than with pixels is that pixels cannot make it: a pass that
/// engaged and wrote the scene back unchanged — `strength` 0, or a fog that
/// happened to cancel — is byte-identical from outside. `UnderwaterReport` is
/// bumped at the point in `run` past which the encoder *will* be touched, so an
/// unchanged count is the property the module docs actually claim.
///
/// Scope, stated because it is easy to over-read: this is about the **underwater
/// post pass**. Shoreline wetness is a different feature and it *does* run above
/// water — that is what a wet shoreline is — so the frames below are not claimed
/// to equal a water-free scene.
#[test]
fn underwater_off_path_never_engages() {
    let Some(gpu) = gpu_or_skip() else { return };
    let level = 15.0;
    let (scene, _) = underwater_scene(level, 11.0);
    assert!(
        scene.waters[0].drawable(),
        "the fixture must carry DRAWABLE water, or this proves nothing"
    );
    let crest = scene.waters[0].waves.max_amplitude_m();

    // Above every crest, looking down at the sea — the ordinary P20.1 shot.
    let above = look_view(
        DVec3::new(BASIN.0, level + crest + 6.0, BASIN.1 - 40.0),
        DVec3::new(BASIN.0, level, BASIN.1 + 30.0),
    );
    assert!(
        inf_render::camera_underwater(&scene.waters, above.eye_world).is_none(),
        "the fixture camera is not above the water"
    );

    // Render several frames, at every quality tier, with the camera above: the
    // node must never engage. A tier loop because `light_shafts()` is the one
    // setting it reads, and a node that consulted it *before* the submersion test
    // would engage here.
    let target = HeadlessTarget::new(&gpu, W, H);
    let mut renderer = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
    assert_eq!(renderer.underwater_engaged_frames(), 0);
    for quality in [WaterQuality::Low, WaterQuality::Medium, WaterQuality::High] {
        let mut settings = RenderSettings::default();
        settings.water.quality = quality;
        renderer.set_settings(settings);
        renderer.render(&gpu, &scene, &above, &target.view, (W, H));
        assert_eq!(
            renderer.underwater_engaged_frames(),
            0,
            "the underwater pass engaged above the waterline at {quality:?}"
        );
    }

    // THE GUARD ON THE GUARD: the same renderer, the same scene, the camera moved
    // below the surface — the counter must move. Without this the assertions
    // above would pass on a node that never runs at all.
    let (_, below) = underwater_scene(level, 11.0);
    renderer.render(&gpu, &scene, &below, &target.view, (W, H));
    assert_eq!(
        renderer.underwater_engaged_frames(),
        1,
        "the underwater pass did not engage BELOW the waterline — the off-path \
         assertions above are vacuous"
    );
}

/// **GOLDEN — a wet shoreline.** The P20.1 hill coast under a noon sun with the
/// sea at 3 m: every island and channel now carries a darkened, glossier band at
/// the waterline. The wetness-*specific* claim — that this is terrain shading
/// driven by the water level and not another way of drawing water — is carried by
/// `shoreline_wetness_is_terrain_shading_not_water` below, which measures a frame
/// containing no water pixels at all.
#[test]
fn golden_water_wetness_shore() {
    let Some(gpu) = gpu_or_skip() else { return };
    let scene = ocean_scene(12.0 * 3600.0, 3.0);
    let view = look_view(DVec3::new(90.0, 12.0, 40.0), DVec3::new(150.0, 2.0, 100.0));
    let img = check_golden(&gpu, "water_wetness_shore", &scene, &view);

    // The shot contains both a sea and a shore: the lower band is blue-dominant
    // water, and moving the sea moves the shoreline — so the picture is of a
    // water LEVEL rather than of a fixed piece of terrain.
    let sea = mean_rgb(&img, 0, H * 3 / 4, W, H);
    assert!(
        sea[2] > sea[0] * 1.1,
        "the lower band is not water (r {:.3}, b {:.3})",
        sea[0],
        sea[2]
    );
    let mut higher = scene.clone();
    higher.waters[0].level_m = 5.0;
    assert!(
        water_changed_fraction(&img, &render(&gpu, &higher, &view)) > 0.05,
        "raising the sea by 2 m moved nothing"
    );
}

/// **The wetness isolation gate.** A frame with a live wet band and *zero water
/// pixels*, so the only thing that can differ is the terrain shading.
///
/// The ocean sits at −0.6 m. The terrain's lowest sample is −0.5 m and the
/// clipmap interpolates between samples (so the rendered ground never dips below
/// the sample minimum): the surface is depth-tested away **everywhere** and
/// contributes nothing. The band, meanwhile, reaches from −0.6 m up to
/// `−0.6 + WET_BAND_M`, which covers the basin floors this camera points straight
/// down at.
///
/// The control drops the same ocean to −60 m: still no water pixels, and now
/// nothing within a band of the ground either. The guard below *proves* the "no
/// water pixels" half rather than assuming it — an occluded ocean must render
/// byte-identically to no ocean at all.
#[test]
fn shoreline_wetness_is_terrain_shading_not_water() {
    let Some(gpu) = gpu_or_skip() else { return };
    let view = basin_view();

    let at = |level_m: f64| {
        let mut s = ocean_scene(12.0 * 3600.0, level_m);
        // A flat calm, so no crest can displace the surface above the ground and
        // put a water pixel in the frame after all.
        s.waters[0].waves = WaveField::default();
        s
    };
    let wet_scene = at(-0.6);
    let dry_scene = at(-60.0);

    // THE GUARD: an ocean below the ground draws nothing. Without it this test
    // would be comparing two water surfaces and calling the difference wetness.
    let none = render(&gpu, &without_water(&dry_scene), &view);
    let dry = render(&gpu, &dry_scene, &view);
    assert_eq!(
        dry, none,
        "an ocean 60 m under the terrain still put pixels in the frame — the \
         isolation this test rests on does not hold"
    );

    let wet = render(&gpu, &wet_scene, &view);
    assert_ne!(
        wet, dry,
        "raising the ocean to just under the ground wet nothing — the band is not \
         reaching the terrain shader"
    );
    // Wet ground is DARKER ground: the film scatters light into the substrate
    // rather than back out of it.
    let whole = |i: &[u8]| mean_luma(i, 0, 0, W, H);
    assert!(
        whole(&wet) < whole(&dry),
        "the wet frame ({:.4}) is not darker than the dry one ({:.4})",
        whole(&wet),
        whole(&dry)
    );
    // …and it is a BAND, not a global tint: it must reach a real part of the
    // frame without reaching all of it.
    let changed = water_changed_fraction(&wet, &dry);
    assert!(
        changed > 0.01 && changed < 0.9,
        "the wet band covered {:.1}% of the frame — that is a global tint, not a \
         shoreline",
        changed * 100.0
    );
}

/// Wetness is **camera-independent**: the same ground, seen from the same place,
/// carries the same band however the camera got there.
///
/// The packing half is pinned exactly and GPU-free by
/// `wetness::tests::wetness_is_a_pure_function_of_the_water`; this is the pixel
/// half. An ocean's *drawn* patch is snapped to the camera, so a band derived
/// from that patch instead of from the body's level would slide under a moving
/// player — the P18.2 camera-residency law, in the one place P20.3 made it easy
/// to break.
#[test]
fn the_wet_band_does_not_follow_the_camera() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mut scene = ocean_scene(12.0 * 3600.0, -0.6);
    scene.waters[0].waves = WaveField::default();

    let view = basin_view();
    let a = render(&gpu, &scene, &view);
    // …after the camera has visited somewhere 900 m away, which moves the ocean's
    // snapped patch by hundreds of cells.
    let elsewhere = look_view(
        DVec3::new(BASIN.0 + 900.0, 40.0, BASIN.1 + 900.0),
        DVec3::new(BASIN.0, 0.0, BASIN.1),
    );
    let _ = render(&gpu, &scene, &elsewhere);
    let b = render(&gpu, &scene, &view);
    assert_eq!(a, b, "the wet band moved with a camera that came back");

    // The guard on the guard: the band IS in this frame, so the equality above is
    // not a statement about an empty effect.
    let dry = {
        let mut s = scene.clone();
        s.waters[0].level_m = -60.0;
        render(&gpu, &s, &view)
    };
    assert_ne!(a, dry, "there was no band to be camera-independent about");
}

// ── P21.1: volumetric terrain ────────────────────────────────────────────────

/// The carved volume the voxel golden draws: a ground slab with a **tunnel bored
/// straight through it** and a dome sitting on top — an overhang, a roof and a
/// cave mouth, none of which a heightfield can represent, which is the entire
/// point of the phase.
///
/// Built from the real `inf-voxel` field and meshed by the real Surface-Nets
/// mesher (a dev-dependency, so Ring 0 stays clean and the golden still exercises
/// the shipped code path rather than a hand-written triangle soup that could agree
/// with nothing).
fn carved_volume() -> (inf_voxel::VoxelData, inf_voxel::VoxelMeshCache) {
    use inf_voxel::{ChunkKey, VoxelChunk, VoxelData, VoxelMeshCache};

    let mut data = VoxelData::new(0.5);
    for key in inf_voxel::chunk_range(ChunkKey::new(0, 0, 0), ChunkKey::new(1, 1, 1)) {
        let b = key.base_sample();
        let mut chunk = VoxelChunk::from_fn(|i, j, k| {
            let (x, y, z) = (
                (b[0] + i as i32) as f64,
                (b[1] + j as i32) as f64,
                (b[2] + k as i32) as f64,
            );
            // A slab of ground: solid below the sample plane y = 15.
            let mut d = y - 15.0;
            // A dome on top of it (CSG union), so the silhouette is not a plane.
            d = d.min(((x - 16.0).powi(2) + (y - 15.0).powi(2) + (z - 16.0).powi(2)).sqrt() - 9.0);
            // A tunnel bored along x at (y, z) = (11, 16) (CSG difference) — the
            // overhang. Its roof is ground the dome and the slab both sit on.
            let tube = ((y - 11.0).powi(2) + (z - 16.0).powi(2)).sqrt() - 4.0;
            d.max(-tube)
        });
        // Three materials, banded by height, so the flat 4-layer palette is
        // exercised rather than asserted about: bedrock below, rock in the slab,
        // a paler crust on the dome.
        for k in 0..inf_voxel::CHUNK_DIM {
            for j in 0..inf_voxel::CHUNK_DIM {
                for i in 0..inf_voxel::CHUNK_DIM {
                    let y = b[1] + j as i32;
                    chunk.set_material(
                        i,
                        j,
                        k,
                        if y > 16 {
                            2
                        } else if y > 9 {
                            1
                        } else {
                            0
                        },
                    );
                }
            }
        }
        data.insert_chunk(key, chunk);
    }
    let mut meshes = VoxelMeshCache::new();
    meshes.sync(&data);
    (data, meshes)
}

/// Project the carved volume into the renderer DTO — the same mapping the two
/// hosts' `project_voxel` performs, written out here because the golden harness
/// has neither a `SceneDoc` nor an ECS world (the same reason
/// `streamed_terrain_renders_partial_residency` builds its `RenderTerrain` by
/// hand, through the `streamed_terrain` helper above).
fn voxel_scene() -> RenderScene {
    let (data, meshes) = carved_volume();
    let voxel_size_m = data.voxel_size_m();
    let chunks: Vec<RenderVoxelChunk> = meshes
        .meshes()
        .map(|(&key, mesh)| RenderVoxelChunk {
            key: VoxelChunkKey::new(key.x, key.y, key.z),
            origin: data.chunk_origin_world(key),
            vertices: mesh
                .local_positions_m(voxel_size_m)
                .into_iter()
                .enumerate()
                .map(|(i, pos)| RenderVoxelVertex {
                    pos,
                    normal: mesh.normals[i],
                    material: mesh.materials[i] as u32,
                    seam_nh: RenderVoxelVertex::NO_SEAM,
                    seam_albedo: [0.0; 4],
                })
                .collect(),
            indices: mesh.indices.clone(),
            bounds: mesh.local_bounds_m(voxel_size_m),
            version: meshes.version(key),
        })
        .collect();
    assert!(!chunks.is_empty(), "the carved fixture produced no surface");

    let mut scene = RenderScene {
        voxels: vec![RenderVoxelVolume {
            id: 1,
            chunks,
            layers: [
                // Three visibly different bands, so the per-vertex material
                // index is proven by the image and not only by the assertion
                // above: dark bedrock, red-brown rock, pale crust.
                RenderTerrainLayer {
                    albedo: [0.16, 0.14, 0.13, 1.0],
                    roughness: 0.95,
                    tex_scale: 4.0,
                    vt: Default::default(),
                },
                RenderTerrainLayer {
                    albedo: [0.46, 0.24, 0.15, 1.0],
                    roughness: 0.85,
                    tex_scale: 4.0,
                    vt: Default::default(),
                },
                RenderTerrainLayer {
                    albedo: [0.62, 0.60, 0.52, 1.0],
                    roughness: 0.70,
                    tex_scale: 4.0,
                    vt: Default::default(),
                },
                RenderTerrainLayer::default(),
            ],
            seam_band_m: 0.0,
        }],
        ..Default::default()
    };
    scene.lights.push(RenderLight {
        kind: LightKind::Directional,
        direction: Vec3::new(0.4, 0.8, 0.45).normalize(),
        color: [1.0, 0.97, 0.9],
        intensity: 1.8,
        ..RenderLight::default()
    });
    scene
}

/// A three-quarter view of the carved volume, framed so the tunnel mouth and the
/// dome above it are both on screen.
fn voxel_view() -> RenderView {
    let eye = DVec3::new(24.0, 12.5, 26.0);
    let target = DVec3::new(8.0, 5.5, 8.0);
    RenderView {
        origin: FloatingOrigin::new(DVec3::ZERO),
        eye_world: eye,
        forward: (target - eye).as_vec3().normalize(),
        up: Vec3::Y,
        fov_y: 55f32.to_radians(),
        near: 0.05,
        width: W,
        height: H,
        ortho: None,
    }
}

/// **The P21.1 render gate.** A carved SDF volume — slab, bored tunnel, dome —
/// reaches the screen through the real mesher, the real DTO and the real voxel
/// pass, deterministically.
#[test]
fn golden_voxel() {
    let Some(gpu) = gpu_or_skip() else { return };
    let scene = voxel_scene();

    // Structural, on the CPU side of the pass, before a pixel is drawn: the
    // fixture really is geometry a heightfield cannot hold, and it really does
    // carry more than one material. Asserted here rather than trusted, because a
    // golden of a flat slab would still be a golden.
    let volume = &scene.voxels[0];
    let mut up = 0usize;
    let mut down = 0usize;
    let mut materials = std::collections::BTreeSet::new();
    let mut tris = 0usize;
    for c in &volume.chunks {
        assert_eq!(c.indices.len() % 3, 0, "{:?}", c.key);
        tris += c.indices.len() / 3;
        for v in &c.vertices {
            if v.normal[1] > 0.7 {
                up += 1;
            }
            if v.normal[1] < -0.7 {
                down += 1;
            }
            materials.insert(v.material);
        }
    }
    assert!(
        tris > 2_000,
        "only {tris} triangles — is the fixture carved?"
    );
    assert!(
        up > 50 && down > 50,
        "a bored tunnel must have a floor ({up} up-facing) AND a ROOF ({down} \
         down-facing) — a heightfield has only the former, and that difference is \
         the whole phase"
    );
    assert!(
        materials.len() >= 3,
        "the palette is exercised by {} material(s)",
        materials.len()
    );

    let img = check_golden(&gpu, "voxel", &scene, &voxel_view());

    // Structural, on the pixels: the top of the frame is sky, the middle is lit
    // rock, and the two are not the same colour.
    let sky = px(&img, W / 2, 3);
    assert!(
        sky[2] as u16 + 4 >= sky[0] as u16,
        "sky not bluish: {sky:?}"
    );
    let mut solid = 0usize;
    for y in (H / 3)..(H - 4) {
        for x in (W / 4)..(3 * W / 4) {
            let p = px(&img, x, y);
            // Rock is warmer than the sky (the palette is a brown/grey ramp), so
            // "red at least matches blue" separates surface from background
            // without pinning an exact colour.
            if p[0] as u16 >= p[2] as u16 + 6 {
                solid += 1;
            }
        }
    }
    assert!(
        solid > 1_500,
        "only {solid} rock pixels — the voxel pass drew (almost) nothing"
    );
}

/// **THE P21.1 OFF-PATH GATE.** A scene with no voxel volumes ⇒ the voxel node
/// records **nothing at all**, which is the byte-stability guarantee for every
/// golden that predates volumetric terrain.
///
/// Asserted with the node's own engagement counter and **not** with a pixel
/// comparison, because a pixel comparison cannot make this claim. The first cut of
/// this gate rendered a volume-free scene twice — once with `voxels` already empty
/// and once after calling `.clear()` on the same empty list — and compared the two
/// images. That is `render(X) == render(X)`: it passes on a node that opens a
/// render pass, binds a pipeline and draws nothing on every single frame, which is
/// precisely the failure the early-out exists to prevent. The house precedent for
/// counting instead is `underwater_off_path_never_engages` (P20.3).
///
/// Three assertions, each catching a different way to get this wrong:
///  1. the counter does not move over several frames of a volume-free scene;
///  2. it moves for a scene that *does* carry volumes (without which 1 would pass
///     on a node that never runs at all);
///  3. the two frames genuinely **differ**, so the pass is not merely engaging
///     but contributing — a node that engaged and drew nothing would satisfy 1
///     and 2 and still be broken.
///
/// # What it does and does not pin, honestly
///
/// `VoxelNode::run` has **defence in depth**: the `scene.voxels.is_empty()`
/// early-out, then a no-drawable-chunks guard, then the instance-buffer check. The
/// counter sits past all three, so this test fails the moment *the conjunction*
/// stops holding — mutation-verified by deleting the guards, which flips assertion
/// 1 to `left: 1`.
///
/// Deleting **only** the first early-out is not observable here, and that is a
/// true statement about the node rather than a weakness in the test: on an empty
/// list the downstream guards already return before any encoder touch or buffer
/// write. The first early-out earns its place on a different claim — releasing the
/// GPU cache on the last-volume-leaves transition — which is stated in `run` and
/// is not what this gate is about.
#[test]
fn voxel_off_path_never_engages() {
    let Some(gpu) = gpu_or_skip() else { return };

    // A volume-free scene with real content, so "nothing drew" cannot be the
    // reason the counter stays put.
    let mut bare = RenderScene {
        grid_enabled: true,
        ..Default::default()
    };
    bare.instances.push(MeshInstance::lit(
        DVec3::new(0.0, 0.5, 0.0),
        Quat::IDENTITY,
        Vec3::ONE,
        [0.8, 0.3, 0.2, 1.0],
        1,
    ));
    assert!(bare.voxels.is_empty());

    let target = HeadlessTarget::new(&gpu, W, H);
    let mut renderer = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
    assert_eq!(renderer.voxel_engaged_frames(), 0);
    for _ in 0..3 {
        renderer.render(&gpu, &bare, &overlook_view(), &target.view, (W, H));
        assert_eq!(
            renderer.voxel_engaged_frames(),
            0,
            "the voxel pass touched the encoder on a scene with no volumes — every \
             golden that predates P21.1 depends on it not doing that"
        );
    }
    let without = target.read_rgba(&gpu).expect("readback");

    // THE GUARD ON THE GUARD: the same renderer, a scene that DOES carry volumes.
    let carved = voxel_scene();
    assert!(!carved.voxels.is_empty());
    renderer.render(&gpu, &carved, &voxel_view(), &target.view, (W, H));
    assert_eq!(
        renderer.voxel_engaged_frames(),
        1,
        "the voxel pass did not engage on a scene full of caves — the off-path \
         assertions above are vacuous"
    );
    let with = target.read_rgba(&gpu).expect("readback");

    // …and it did not merely engage, it CONTRIBUTED: a node that opened a pass and
    // drew nothing would satisfy both counters above.
    let (mean, max) = image_diff(&without, &with, W, H);
    assert!(
        mean > 0.02 && max > 0.2,
        "the engaged frame is indistinguishable from the volume-free one (mean \
         {mean}, max {max}) — the pass ran but drew nothing"
    );

    // A volume list that is present but carries no DRAWABLE chunk must also stay
    // off the encoder: `run` returns after the cache sync and before the pass.
    let empty_volume = RenderScene {
        voxels: vec![RenderVoxelVolume {
            id: 7,
            chunks: Vec::new(),
            layers: [RenderTerrainLayer::default(); 4],
            seam_band_m: 0.0,
        }],
        ..bare.clone()
    };
    let before = renderer.voxel_engaged_frames();
    renderer.render(&gpu, &empty_volume, &overlook_view(), &target.view, (W, H));
    assert_eq!(
        renderer.voxel_engaged_frames(),
        before,
        "a volume with no chunks engaged the encoder"
    );
}

// ── P21.2: the cave mouth ───────────────────────────────────────────

/// Row-pack a tile's holes into the [`RenderTerrainTile::holes`] layout: bit
/// `i & 31` of word `(i >> 5) + j * ceil(res/32)`.
///
/// The same repack a host projector performs, written out here for the same
/// reason `voxel_scene` writes out `project_voxel`: the golden harness has no
/// `SceneDoc` and no ECS world.
fn pack_holes(res: u32, holed: impl Fn(u32, u32) -> bool) -> Vec<u32> {
    let words = RenderTerrainTile::hole_words_per_row(res) as usize;
    let mut out = vec![0u32; words * res as usize];
    for j in 0..res {
        for i in 0..res {
            if holed(i, j) {
                out[j as usize * words + (i >> 5) as usize] |= 1u32 << (i & 31);
            }
        }
    }
    out
}

/// A flat heightfield at y = 15 with a **round hole** punched through the middle
/// of it, and — through that hole — the P21.1 carved volume's tunnel underneath.
///
/// The two are deliberately built from the same numbers: the volume's slab top is
/// y = 15 and its bore runs along x at (y, z) = (11, 16), so the hole opens
/// directly onto four metres of cave and the camera can see into it. That is the
/// whole claim of P21.2 in one frame — the clipmap stops where the mask says, and
/// what is behind it is volumetric terrain, not sky.
fn cave_mouth_scene() -> RenderScene {
    const RES: u32 = 33;
    const MPS: f64 = 1.0;
    // The hole: every sample within 5 m of (16, 16), which is where the bore is.
    let holed = |i: u32, j: u32| {
        let dx = i as f64 - 16.0;
        let dz = j as f64 - 16.0;
        (dx * dx + dz * dz).sqrt() <= 5.0
    };

    let mut scene = voxel_scene();
    scene.terrains.push(RenderTerrain {
        id: 2,
        tile_resolution: RES,
        meters_per_sample: MPS,
        tiles: vec![RenderTerrainTile {
            key: TerrainTileKey::lod0((0, 0)),
            origin: DVec3::new(0.0, 15.0, 0.0),
            heights: vec![0.0; (RES * RES) as usize],
            weights: vec![[255, 0, 0, 0]; (RES * RES) as usize],
            biomes: vec![0; (RES * RES) as usize],
            holes: pack_holes(RES, holed),
            height_bounds: (0.0, 0.0),
            version: 1,
        }],
        layers: [
            // A grassy top layer, deliberately unlike the volume's rock ramp, so
            // "which surface am I looking at" is answerable from the pixels.
            RenderTerrainLayer {
                albedo: [0.22, 0.34, 0.16, 1.0],
                roughness: 0.9,
                tex_scale: 6.0,
                vt: Default::default(),
            },
            RenderTerrainLayer::default(),
            RenderTerrainLayer::default(),
            RenderTerrainLayer::default(),
        ],
        macro_variation: 0.0,
        biome_palette: Vec::new(),
    });

    // Turn the seam blend on for the volume, and fill the per-vertex seam terms
    // from the terrain that is now over it — exactly what a projector does.
    let terrain = scene.terrains[0].clone();
    let volume = &mut scene.voxels[0];
    volume.seam_band_m = inf_render::DEFAULT_SEAM_BAND_M;
    for chunk in &mut volume.chunks {
        let base = chunk.origin;
        for v in &mut chunk.vertices {
            let wx = base.x + v.pos[0] as f64;
            let wz = base.z + v.pos[2] as f64;
            if let Some(sample) = terrain.seam_sample(wx, wz) {
                let (nh, albedo) = sample.pack(0.0);
                v.seam_nh = nh;
                v.seam_albedo = albedo;
            }
        }
    }
    scene
}

/// **The P21.2 render gate.** A hole-punched heightfield with a voxel cave
/// visible through it: the clipmap discards where the mask says, and what shows
/// through the gap is the volume, not the sky.
///
/// The structural assertions are what make this a gate rather than a picture.
/// The mask has to actually be sparse-and-nonzero (a golden of an un-carved tile
/// would still be a golden), the seam has to actually have been sampled at some
/// vertices and NOT at the ones inside the hole, and the pixels through the
/// mouth have to be lit rock rather than sky — which is the one thing a discard
/// that fired in the wrong place would break.
#[test]
fn golden_cave_mouth() {
    let Some(gpu) = gpu_or_skip() else { return };
    let scene = cave_mouth_scene();

    // ── CPU-side, before a pixel is drawn ───────────────────────────────
    let tile = &scene.terrains[0].tiles[0];
    let res = scene.terrains[0].tile_resolution;
    assert!(tile.has_holes(), "the fixture is not carved");
    let holed = (0..res)
        .flat_map(|j| (0..res).map(move |i| (i, j)))
        .filter(|&(i, j)| tile.is_hole(res, i, j))
        .count();
    assert!(
        (60..400).contains(&holed),
        "{holed} holed samples — the mouth must be a real opening, and not the \
         whole tile"
    );
    // The packed mask is bits, not bytes: a 33-sample row costs two words.
    assert_eq!(tile.holes.len(), 2 * res as usize);

    // The seam was sampled outside the hole and refused inside it — the poison
    // rule, on the projector's side of the boundary.
    let volume = &scene.voxels[0];
    let seamed = volume
        .chunks
        .iter()
        .flat_map(|c| &c.vertices)
        .filter(|v| v.seam_nh[1] > 0.0)
        .count();
    assert!(seamed > 0, "no vertex picked up a seam sample");
    let under_hole = volume
        .chunks
        .iter()
        .flat_map(|c| c.vertices.iter().map(move |v| (c.origin, v)))
        .filter(|(o, v)| {
            let dx = o.x + v.pos[0] as f64 - 16.0;
            let dz = o.z + v.pos[2] as f64 - 16.0;
            (dx * dx + dz * dz).sqrt() < 3.0
        })
        .collect::<Vec<_>>();
    assert!(!under_hole.is_empty(), "no cave geometry under the mouth");
    assert!(
        under_hole.iter().all(|(_, v)| v.seam_nh[1] == 0.0),
        "a vertex under the hole picked up a seam — the poison rule did not fire"
    );

    let img = check_golden(&gpu, "cave_mouth", &scene, &cave_mouth_view());

    // ── pixels ────────────────────────────────────────────────
    // The camera looks straight down at the ground, so the frame is grass with a
    // hole in it. The claim is the CONTRAST between the two: the surround is the
    // terrain's green layer, and the middle — seen through the mouth — is the
    // volume's rock ramp. If the discard had not fired the middle would be green
    // too; if it had over-fired there would be nothing green at all.
    let surround = px(&img, W / 2, H / 8);
    let mouth = px(&img, W / 2, H / 2);
    let greener = |p: [u8; 4]| p[1] as i32 - p[0] as i32;
    assert!(
        greener(surround) > 8,
        "the surround is not the terrain's green layer: {surround:?}"
    );
    assert!(
        greener(mouth) + 8 < greener(surround),
        "the mouth is as green as the ground around it — the hole did not \
         discard: mouth {mouth:?} vs surround {surround:?}"
    );
    // … and both surfaces are actually present in quantity, so neither claim is
    // resting on a single lucky texel.
    let mut green = 0usize;
    let mut rock = 0usize;
    for y in 0..H {
        for x in 0..W {
            let p = px(&img, x, y);
            if greener(p) > 8 {
                green += 1;
            } else if p[0] > 24 {
                rock += 1;
            }
        }
    }
    assert!(
        green > 5_000,
        "only {green} grass pixels — the clipmap is gone"
    );
    assert!(
        rock > 500,
        "only {rock} non-grass lit pixels — nothing shows through the mouth"
    );
}

/// The cave-mouth heightfield as a **streamed** terrain, in the two residency
/// states a camera produces: `(near, far)`.
///
/// `near` is the authored level-0 page (holed, painted) *plus* the coarse pyramid
/// page `TerrainStreamer` pins as its residency floor. `far` is what the streamer
/// publishes once the camera has walked away — the floor alone. Both are valid
/// projections of one asset, they differ only in what the camera dragged in, and
/// `max_lod()` is `1` in both, which is the property the floor exists to give.
///
/// The coarse page is flat at the same surface height as the fine one but carries
/// **no weights and no hole mask** — what `inf_terrain::pyramid::downsample_block`
/// actually produces.
fn streamed_cave_terrain() -> (RenderTerrain, RenderTerrain) {
    let mut near = cave_mouth_scene().terrains.remove(0);
    let fine = near.tiles[0].clone();
    let res = near.tile_resolution;
    near.tiles.push(RenderTerrainTile {
        key: TerrainTileKey::new(1, (0, 0)),
        origin: fine.origin,
        heights: vec![0.0; (res * res) as usize],
        weights: Vec::new(),
        biomes: Vec::new(),
        holes: Vec::new(),
        height_bounds: (0.0, 0.0),
        version: 2,
    });
    let far = RenderTerrain {
        tiles: near
            .tiles
            .iter()
            .filter(|t| t.key.lod == 1)
            .cloned()
            .collect(),
        ..near.clone()
    };
    (near, far)
}

/// **THE B1 GATE: camera-driven residency must never feed voxel lighting.**
///
/// The P18 law (`gi::voxelization_tiles`) with a second consumer. P21.2's seam
/// blend hands a voxel fragment the heightfield's albedo, roughness and shading
/// normal, and the first cut resolved them against the *level-0* pages —
/// precisely the part of the published cut that pages in and out. A cave mouth
/// therefore lit one way with the fine page resident and another without it, and
/// no golden would have noticed: every golden renders a fully-resident terrain.
///
/// This drives one asset through two genuinely different residency histories and
/// **byte-compares the rendered voxel surface**. The terrain itself is removed
/// from the frame before rendering — deliberately, and it is the whole design of
/// the instrument: the two states legitimately *draw* different heightfield
/// detail (asserted below), so an image that contained the terrain could not
/// separate "the clipmap drew a coarser hill" from "the cave was lit differently".
/// The seam terms are baked into the vertices before the terrain leaves, so what
/// is compared is exactly the voxel pass's own output.
///
/// Structural + bitwise, so no golden PNG is added (the count stays at 49): the
/// claim is an equality between two renders, not the appearance of either.
#[test]
fn voxel_lighting_is_independent_of_terrain_residency() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (near, far) = streamed_cave_terrain();
    assert_eq!(near.max_lod(), 1);
    assert_eq!(far.max_lod(), 1);
    assert!(
        far.tiles.len() < near.tiles.len(),
        "the paged-out state must actually be a smaller residency set"
    );

    // Bake the seam from `terrain`, then drop the terrain and draw the volume.
    let lit = |terrain: &RenderTerrain, band: f32| -> (Vec<u8>, usize) {
        let mut scene = voxel_scene();
        inf_render::apply_seam(&mut scene.voxels, std::slice::from_ref(terrain), band);
        let seamed = scene.voxels[0]
            .chunks
            .iter()
            .flat_map(|c| &c.vertices)
            .filter(|v| v.seam_nh[1] > 0.0)
            .count();
        scene.terrains.clear();
        scene.mark_dirty();
        (
            render_with(&gpu, &scene, &cave_mouth_view(), RenderSettings::default()),
            seamed,
        )
    };

    let (with_fine, seamed_near) = lit(&near, inf_render::DEFAULT_SEAM_BAND_M);
    let (without_fine, seamed_far) = lit(&far, inf_render::DEFAULT_SEAM_BAND_M);

    // Not vacuous: the seam fired, on the same vertices, in both states.
    assert!(
        seamed_near > 0,
        "no vertex picked up a seam — the gate would pass on two unseamed renders"
    );
    assert_eq!(
        seamed_near, seamed_far,
        "a different number of vertices seamed depending on residency"
    );

    // Summarize rather than assert_eq! on the buffers: a failure would otherwise
    // print two 320×180 RGBA images.
    let mismatch = |a: &[u8], b: &[u8]| -> Option<(usize, usize)> {
        if a.len() != b.len() {
            return Some((0, a.len().max(b.len())));
        }
        let differing = a.iter().zip(b).filter(|(x, y)| x != y).count();
        a.iter()
            .zip(b)
            .position(|(x, y)| x != y)
            .map(|i| (i, differing))
    };
    assert!(
        mismatch(&with_fine, &without_fine).is_none(),
        "the voxel surface was lit differently across two residency histories \
         ({:?} = (first differing byte, count of {})) — camera history is feeding \
         lighting",
        mismatch(&with_fine, &without_fine),
        with_fine.len()
    );

    // The comparison above is only worth making if the seam reaches the pixels at
    // all: the same volume with the band disarmed must render differently.
    let (unseamed, none) = lit(&near, 0.0);
    assert_eq!(
        none, 0,
        "a disarmed band must leave every vertex at NO_SEAM"
    );
    // Counted, not averaged: the band is a two-metre ring around one surface
    // height by construction, so it covers a small fraction of a frame that is
    // mostly dome. A whole-image mean would be a threshold tuned to this
    // fixture's framing rather than a statement about the seam.
    let moved = with_fine
        .chunks_exact(4)
        .zip(unseamed.chunks_exact(4))
        .filter(|(a, b)| (0..3).any(|c| (a[c] as i32 - b[c] as i32).abs() >= 8))
        .count();
    assert!(
        moved > 200,
        "only {moved} pixel(s) moved when the band was armed, so byte-equality \
         across residency proves nothing about lighting"
    );

    // …and the two residency states really are different inputs: with the terrain
    // left in the frame they draw differently, which is exactly why the gate
    // takes it out.
    let drawn = |terrain: &RenderTerrain| {
        let mut scene = voxel_scene();
        scene.terrains.push(terrain.clone());
        scene.mark_dirty();
        render_with(&gpu, &scene, &cave_mouth_view(), RenderSettings::default())
    };
    assert_ne!(
        drawn(&near),
        drawn(&far),
        "the two residency states rendered identically — the fixture is not \
         exercising a residency difference at all"
    );
}

/// Looking down into the mouth from above and to the side, close enough that the
/// opening fills the middle of the frame.
fn cave_mouth_view() -> RenderView {
    let eye = DVec3::new(16.0, 30.0, 30.0);
    let target = DVec3::new(16.0, 11.0, 16.0);
    RenderView {
        origin: FloatingOrigin::new(DVec3::ZERO),
        eye_world: eye,
        forward: (target - eye).as_vec3().normalize(),
        up: Vec3::Y,
        fov_y: 55f32.to_radians(),
        near: 0.05,
        width: W,
        height: H,
        ortho: None,
    }
}

// ── surface deformation (P22.1) ──────────────────────────────────────────────

/// Snow-layer terrain samples per tile edge, and metres between them.
///
/// 0.25 m/sample is deliberately **exactly** the deformation lattice pitch, so
/// the fragment's central-difference normal (`span / (res - 1)`) steps one field
/// sample at a time and a footprint reads as the shape it is rather than as a
/// blur across four of them.
const DFM_RES: u32 = 65;
const DFM_MPS: f64 = 0.25;

/// A flat 2 x 2-tile snow field — the ground the deformation golden presses.
///
/// Flat on purpose: relief of its own would make it impossible to say whether a
/// dent in the frame came from the heightfield or from the deformation window,
/// which is the whole thing this golden exists to pin.
fn snow_flat_terrain() -> RenderTerrain {
    let span = (DFM_RES - 1) as f64 * DFM_MPS;
    let n = (DFM_RES * DFM_RES) as usize;
    let mut tiles = Vec::new();
    for tx in 0..2 {
        for tz in 0..2 {
            tiles.push(RenderTerrainTile {
                key: TerrainTileKey::lod0((tx, tz)),
                origin: DVec3::new(tx as f64 * span, 0.0, tz as f64 * span),
                heights: vec![0.0; n],
                // Pure layer 3 = snow, which is the deep/soft archetype in
                // `inf_terrain::deform::LAYER_RESPONSE` and the one a boot
                // actually sinks into.
                weights: vec![[0, 0, 0, 255]; n],
                biomes: Vec::new(),
                height_bounds: (0.0, 0.0),
                holes: Vec::new(),
                version: 1,
            });
        }
    }
    RenderTerrain {
        id: 0,
        tile_resolution: DFM_RES,
        meters_per_sample: DFM_MPS,
        tiles,
        layers: default_layers(),
        macro_variation: 0.0,
        biome_palette: Vec::new(),
    }
}

/// Press a walked trail and a pair of vehicle ruts into a **real**
/// `inf_terrain::deform::DeformField`, then project it exactly the way both
/// hosts' `project_deform` does.
///
/// The real field rather than a hand-written buffer, for the same reason the P21
/// voxel golden meshes with the real mesher: a golden built from a fixture the
/// shipped code never produces pins a picture of nothing. Everything here — the
/// falloff profile, the per-class depth fraction, the snow saturation — comes out
/// of the Ring-0 response table.
fn walked_deform() -> RenderDeform {
    use inf_terrain::deform::{
        DeformField, PressureClass, DEFORM_CELL_SAMPLES, DEFORM_SAMPLE_PITCH_M,
    };
    const SNOW: u8 = 3;
    const DT: f64 = 1.0 / 60.0;
    let mut field = DeformField::new();

    // A gait: a footfall every 0.7 m, alternating 25 cm either side of the line,
    // each rested on long enough to reach its class depth.
    for i in 0..18u32 {
        let x = 3.0 + i as f64 * 0.7;
        let z = 5.0 + if i % 2 == 0 { -0.25 } else { 0.25 };
        // relax + stamp per iteration: the field advances a sample once per
        // fixed STEP (that is what makes a step's contacts commute), so a burst
        // of stamps with no relax between them is one instant and one advance.
        for _ in 0..24 {
            field.relax(DT, 0.0);
            field.stamp_contact(glam::DVec2::new(x, z), 0.22, PressureClass::Foot, SNOW, DT);
        }
    }
    // Two parallel ruts across bare snow, and a third driven straight through the
    // grass band so the foliage bend has tracks to lie down in.
    for lane in [8.0f64, 9.2, 12.0] {
        let mut x = 2.0;
        while x < 17.0 {
            // 20 steps at ~3 cm/step is past snow's 0.4 m saturation, so a rut
            // really does bottom out at the table's `max_depth_m` — which is what
            // the CPU-side assertion in `golden_deform` checks against.
            for _ in 0..20 {
                field.relax(DT, 0.0);
                field.stamp_contact(
                    glam::DVec2::new(x, lane),
                    0.34,
                    PressureClass::Heavy,
                    SNOW,
                    DT,
                );
            }
            x += 0.12;
        }
    }
    assert!(
        !field.is_empty(),
        "the fixture must actually press something"
    );

    RenderDeform {
        cell_samples: DEFORM_CELL_SAMPLES,
        texel_m: DEFORM_SAMPLE_PITCH_M,
        epoch: field.epoch(),
        cells: field
            .cells()
            .map(|(coord, cell)| RenderDeformCell {
                coord: *coord,
                depths: cell.depths().to_vec(),
            })
            .collect(),
    }
}

/// Low grass over the same ground, so the scatter bend has something to bend.
fn grass_over(n: u32, span: f64, center: DVec3) -> ScatterBatch {
    let step = span / n as f64;
    let scale = 0.24f32;
    let mut out = Vec::with_capacity((n * n) as usize);
    for i in 0..n * n {
        let (gx, gz) = ((i % n) as f64, (i / n) as f64);
        // The scatter goldens' integer hash — no `std` trig anywhere near
        // committed pixels (the P14 LAW).
        let mut h = i.wrapping_mul(2_654_435_761);
        h ^= h >> 15;
        h = h.wrapping_mul(0x27d4_eb2d);
        let jx = ((h & 0xFFFF) as f64 / 65535.0) - 0.5;
        let jz = (((h >> 16) & 0xFFFF) as f64 / 65535.0) - 0.5;
        out.push(ScatterInstance {
            position: DVec3::new(
                center.x + (gx - (n as f64 - 1.0) * 0.5 + jx * 0.7) * step,
                scale as f64 * 0.5,
                center.z + (gz - (n as f64 - 1.0) * 0.5 + jz * 0.7) * step,
            ),
            rotation: Quat::IDENTITY,
            scale: Vec3::splat(scale),
            color: [0.20, 0.42, 0.16, 1.0],
        });
    }
    ScatterBatch::lit(
        Arc::new(ScatterData::build(PrimMesh::Cube, center, out)),
        center,
        0.9,
        91,
    )
}

fn deform_scene() -> RenderScene {
    let mut scene = RenderScene {
        terrains: vec![snow_flat_terrain()],
        scatter: vec![grass_over(44, 8.0, DVec3::new(11.0, 0.0, 12.0))],
        deform: Some(Arc::new(walked_deform())),
        lights: vec![RenderLight {
            kind: LightKind::Directional,
            // Low and across the trail: a footprint is a *shading* feature, and a
            // sun overhead would light both walls of every dent equally.
            direction: Vec3::new(-0.62, 0.34, -0.71).normalize(),
            color: [1.0, 0.97, 0.92],
            // Low intensity as well as a low angle: snow's albedo is ~0.9, and a
            // brighter sun clips the whole field to white in the tonemap — which
            // is exactly where a shading feature stops being visible.
            intensity: 1.5,
            ..Default::default()
        }],
        ..Default::default()
    };
    // The wind phase is the level's clock, published on the cloud block by both
    // projectors (`ResolvedSky::cloud_time_s`) — no frame index, so two renders
    // of this scene are the same frame.
    scene.atmosphere.clouds.wind_x = 1.0;
    scene.atmosphere.clouds.wind_z = 0.35;
    scene.atmosphere.clouds.time_s = 12.5;
    scene.mark_dirty();
    scene
}

fn deform_view() -> RenderView {
    look_view(DVec3::new(2.5, 8.0, -1.5), DVec3::new(11.0, 0.0, 11.0))
}

/// **Surface deformation golden** (P22.1): a walked footprint trail and a pair of
/// vehicle ruts pressed into a snow-layer terrain patch, with grass scatter bent
/// out of the tracks.
///
/// What it pins, and why each half needs the other: the vertex stage's
/// displacement (the ground physically dips), the fragment stage's
/// central-difference normal through the SAME `ground_height` wrapper (the dip is
/// *shaded* — a vertex-only offset would leave the dent invisible under this
/// grazing sun), the compaction darkening, and the scatter shear that lays the
/// foliage down where the tracks ran.
#[test]
fn golden_deform() {
    let Some(gpu) = gpu_or_skip() else { return };
    let scene = deform_scene();
    let view = deform_view();

    // CPU-side, before a pixel is drawn: the fixture really pressed snow, and it
    // pressed it to the depth the Ring-0 table says — not to a number this test
    // invented.
    let d = scene.deform.as_ref().expect("the fixture projects a field");
    assert!(d.drawable());
    let deepest = d
        .cells
        .iter()
        .flat_map(|c| c.depths.iter().copied())
        .fold(0.0f32, f32::max);
    let snow_max = inf_terrain::deform::LAYER_RESPONSE[3].max_depth_m;
    assert!(
        (deepest - snow_max).abs() < 1e-6,
        "a Heavy rut in snow must saturate at {snow_max} m, reached {deepest}"
    );
    assert!(
        deepest <= inf_render::DEFORM_MAX_DEPTH_M,
        "the skirt coupling: nothing may press past {} m",
        inf_render::DEFORM_MAX_DEPTH_M
    );

    let img = check_golden(&gpu, "deform", &scene, &view);

    // Structural, not just perceptual: the frame really contains snow, grass, and
    // ground darker than the snow around it (the tracks).
    let bright = img.chunks(4).filter(|p| p[0] > 200 && p[2] > 200).count();
    let green = img
        .chunks(4)
        .filter(|p| p[1] > p[0] + 6 && p[1] > p[2] + 6)
        .count();
    assert!(bright > 8_000, "expected a snowfield ({bright} px)");
    assert!(green > 1_000, "expected the grass band ({green} px)");

    // The tracks are visible as *shading*, and the honest way to say so is against
    // the same geometry with the field removed rather than against an absolute
    // brightness threshold that snow's own albedo would decide.
    let mut flat = deform_scene();
    flat.deform = None;
    flat.mark_dirty();
    let without = render(&gpu, &flat, &view);
    let darker = img
        .chunks(4)
        .zip(without.chunks(4))
        .filter(|(a, b)| b[0] as i32 - a[0] as i32 > 6)
        .count();
    assert!(
        darker > 400,
        "only {darker} px darkened — the tracks are not reaching the terrain shader"
    );
    let moved = changed_fraction(&img, &without, H);
    assert!(
        moved > 0.02,
        "the deformation moved only {moved:.4} of the frame — the window is not \
         reaching the terrain shader"
    );
}

/// **The off-path gate** (the P20.3 engagement-counter law): a scene with no
/// deformation field must record the command stream it always did.
///
/// A pixel comparison cannot make this claim — a frame that uploaded a megabyte
/// of zeroes looks exactly like one that uploaded nothing — so the assertion is
/// on the upload counter, with a guard on the guard so it cannot pass vacuously.
#[test]
fn deform_off_path_never_engages() {
    let Some(gpu) = gpu_or_skip() else { return };
    let target = HeadlessTarget::new(&gpu, W, H);
    let mut renderer = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
    let view = deform_view();

    // Arm 1: a scene with terrain and scatter but no field — many frames, zero
    // uploads.
    let mut flat = deform_scene();
    flat.deform = None;
    flat.mark_dirty();
    assert_eq!(renderer.deform_uploads(), 0);
    for f in 0..4 {
        renderer.render(&gpu, &flat, &view, &target.view, (W, H));
        assert_eq!(
            renderer.deform_uploads(),
            0,
            "frame {f} uploaded a deformation window for a scene that has none"
        );
    }

    // Arm 2 — THE GUARD ON THE GUARD. The same renderer, given a field, must
    // upload. Without this, arm 1 would pass just as happily if the counter were
    // never incremented at all.
    let scene = deform_scene();
    renderer.render(&gpu, &scene, &view, &target.view, (W, H));
    let after_first = renderer.deform_uploads();
    assert!(
        after_first >= 1,
        "a scene WITH a deformation field uploaded nothing"
    );

    // Arm 3 — the dirty gate. Re-rendering the identical scene from the identical
    // camera must upload nothing further: the window already holds the answer,
    // and the epoch says so.
    for f in 0..4 {
        renderer.render(&gpu, &scene, &view, &target.view, (W, H));
        assert_eq!(
            renderer.deform_uploads(),
            after_first,
            "frame {f} re-uploaded an unchanged window"
        );
    }

    // Arm 4 — a moved camera re-windows (and is allowed to). This is what makes
    // arm 3 a statement about the *gate* rather than about a counter that had
    // simply stopped moving.
    let panned = look_view(DVec3::new(41.0, 5.0, 39.0), DVec3::new(51.0, 0.0, 50.0));
    renderer.render(&gpu, &scene, &panned, &target.view, (W, H));
    assert!(
        renderer.deform_uploads() > after_first,
        "panning the camera off the window did not re-upload it"
    );

    // Arm 5 — dropping the field forgets the window, so the next real projection
    // cannot be diffed against a stale one.
    let before_drop = renderer.deform_uploads();
    renderer.render(&gpu, &flat, &view, &target.view, (W, H));
    assert_eq!(
        renderer.deform_uploads(),
        before_drop,
        "a scene with no field uploaded on the way back out"
    );
    renderer.render(&gpu, &scene, &view, &target.view, (W, H));
    assert!(
        renderer.deform_uploads() > before_drop,
        "the window was not re-uploaded after the field had been dropped"
    );
}

// ── the vertex half of the one-wrapper claim (P22.1 audit B2) ────────────────

/// Ruts driven right along the terrain's far boundary, so the boundary itself is
/// what the deformation moves.
fn edge_deform() -> RenderDeform {
    use inf_terrain::deform::{
        DeformField, PressureClass, DEFORM_CELL_SAMPLES, DEFORM_SAMPLE_PITCH_M,
    };
    const DT: f64 = 1.0 / 60.0;
    let span = (DFM_RES - 1) as f64 * DFM_MPS; // 16 m
    let far = span * 2.0; // the far edge of the 2 x 2 patch
    let mut field = DeformField::new();
    // A WIDE (6 m) saturated band along the far edge. Wide on purpose: a narrow
    // trench's near rim occludes its own floor at a shallow view angle, and then
    // the silhouette never moves however deep the trench is.
    for lane in 0..24 {
        let z = far - 0.2 - lane as f64 * 0.25;
        let mut x = 0.5;
        while x < span * 2.0 - 0.5 {
            for _ in 0..20 {
                field.relax(DT, 0.0);
                field.stamp_contact(glam::DVec2::new(x, z), 0.6, PressureClass::Heavy, 3, DT);
            }
            x += 0.5;
        }
    }
    assert!(!field.is_empty());
    RenderDeform {
        cell_samples: DEFORM_CELL_SAMPLES,
        texel_m: DEFORM_SAMPLE_PITCH_M,
        epoch: field.epoch(),
        cells: field
            .cells()
            .map(|(coord, cell)| RenderDeformCell {
                coord: *coord,
                depths: cell.depths().to_vec(),
            })
            .collect(),
    }
}

/// A **grazing-incidence** view of the same snow patch: the eye sits a few
/// centimetres above the surface, so the patch's far boundary is a silhouette
/// against the sky and a change in the boundary's *height* is a change in where
/// that silhouette sits on screen.
fn grazing_scene(deform: Option<RenderDeform>) -> RenderScene {
    let mut scene = RenderScene {
        terrains: vec![snow_flat_terrain()],
        deform: deform.map(Arc::new),
        lights: vec![RenderLight {
            kind: LightKind::Directional,
            direction: Vec3::new(-0.2, 0.9, -0.35).normalize(),
            color: [1.0, 0.97, 0.92],
            intensity: 1.5,
            ..Default::default()
        }],
        ..Default::default()
    };
    scene.mark_dirty();
    scene
}

fn grazing_view() -> RenderView {
    let span = (DFM_RES - 1) as f64 * DFM_MPS;
    let far = span * 2.0;
    // A low, oblique look at the patch's FAR edge: shallow enough that 40 cm of
    // vertical displacement is a large screen-space move, steep enough that the
    // depressed band's floor is not occluded by its own near rim.
    look_view(
        DVec3::new(span, 3.2, far - 14.0),
        DVec3::new(span, -0.2, far + 1.0),
    )
}

/// Rows of "ground" (bright) pixels per column — the ground/sky split the
/// silhouette assertion measures.
fn bright_pixels(img: &[u8]) -> usize {
    img.chunks(4)
        .filter(|p| p[0] as u32 + p[1] as u32 + p[2] as u32 > 480)
        .count()
}

/// **B2 — the vertex half of `ground_height` is pinned.**
///
/// The fragment half was already defended: diverting the central difference away
/// from the wrapper fails `golden_deform`'s "darker px" arm. The vertex half was
/// not, and an audit proved it — bypassing the wrapper in the vertex stage (so
/// the ground never dips and only the shading pretends) left **93/93 tests green
/// under `INF_GOLDEN_STRICT` on real hardware**.
///
/// This is the arm that catches it. At grazing incidence the terrain's far
/// boundary is a silhouette against the sky, so its *height* is directly visible
/// as the position of the ground/sky split. A vertex displacement moves the
/// silhouette. A fragment-only change — the mutation that survived — cannot move
/// it by a single pixel, because shading does not decide which pixels the
/// geometry covers.
///
/// A structural pixel-split assertion on the existing render, deliberately, and
/// not a new golden: what is being asserted here is a *difference between two
/// frames*, which a golden image is the wrong instrument for.
#[test]
fn deform_moves_the_silhouette_at_grazing_incidence() {
    let Some(gpu) = gpu_or_skip() else { return };
    let view = grazing_view();
    let flat = render(&gpu, &grazing_scene(None), &view);
    let dented = render(&gpu, &grazing_scene(Some(edge_deform())), &view);

    let flat_ground = bright_pixels(&flat);
    let dented_ground = bright_pixels(&dented);
    // The fixture must actually be a grazing view of ground against sky, or the
    // comparison below is measuring nothing.
    assert!(
        flat_ground > 2_000 && flat_ground < (W * H) as usize * 3 / 4,
        "the grazing fixture does not show ground against sky ({flat_ground} px)"
    );
    let moved = flat_ground.abs_diff(dented_ground);
    assert!(
        moved > 300,
        "pressing the terrain's boundary down 0.4 m moved the ground/sky split by \
         only {moved} px ({flat_ground} → {dented_ground}). The VERTEX half of \
         `ground_height` is not displacing geometry — shading alone cannot move a \
         silhouette."
    );
}

// ── byte identity, in the house pattern (P22.1 audit B3) ────────────────────

/// **B3 — the off path is byte-identical, proven by exact buffer equality.**
///
/// `INF_GOLDEN_STRICT` is a *perceptual* compare (`image_diff` downscales to
/// 5 × 5 boxes and allows 6 % mean / 35 % max), which is precisely why the B2
/// mutation above sailed through it. "All 49 goldens byte-identical" was a claim
/// the cited gate could not make. This is the gate that can — the same exact
/// `assert_eq!` on the whole framebuffer that
/// `water_off_path_is_byte_identical` has used since P20.3.
#[test]
fn deform_off_path_is_byte_identical() {
    let Some(gpu) = gpu_or_skip() else { return };
    let view = deform_view();

    // The reference: the deformation scene with no field at all.
    let mut base_scene = deform_scene();
    base_scene.deform = None;
    base_scene.mark_dirty();
    let base = render_with(&gpu, &base_scene, &view, RenderSettings::default());

    // (1) A projection that carries no cells is **undrawable**, and must reach
    //     the frame exactly as `None` does — not "almost".
    let mut empty = base_scene.clone();
    empty.deform = Some(Arc::new(RenderDeform {
        cell_samples: inf_terrain::deform::DEFORM_CELL_SAMPLES,
        texel_m: inf_terrain::deform::DEFORM_SAMPLE_PITCH_M,
        epoch: 7,
        cells: Vec::new(),
    }));
    empty.mark_dirty();
    assert!(!empty.deform.as_ref().unwrap().drawable());
    assert_eq!(
        base,
        render_with(&gpu, &empty, &view, RenderSettings::default()),
        "an EMPTY deformation projection changed the frame — the off path is not \
         neutral, and every pre-P22.1 golden depends on it being"
    );

    // (2) The foliage-wind switch is off by default and must be inert with it
    //     off, whatever the level's wind says.
    let mut windy = base_scene.clone();
    windy.atmosphere.clouds.wind_x = 9.0;
    windy.atmosphere.clouds.wind_z = -4.0;
    windy.atmosphere.clouds.time_s = 4_321.5;
    windy.mark_dirty();
    assert!(!RenderSettings::default().scatter.foliage_wind);
    assert_eq!(
        base,
        render_with(&gpu, &windy, &view, RenderSettings::default()),
        "the level's wind moved foliage with `foliage_wind` off — the switch is \
         not the switch"
    );

    // GUARDS ON THE GUARDS: a real field, and the wind switch on, each move the
    // frame — so the equalities above are not passing because nothing draws.
    assert_ne!(
        base,
        render_with(&gpu, &deform_scene(), &view, RenderSettings::default()),
        "a real deformation field did not change the frame"
    );
    let mut on = RenderSettings::default();
    on.scatter.foliage_wind = true;
    assert_ne!(
        base,
        render_with(&gpu, &windy, &view, on),
        "turning `foliage_wind` on did not change the frame"
    );
}

// ── the window's upload paths (P22.1 audit majors) ──────────────────────────

/// A projection holding the first `keep` cells of the walked fixture, stamped
/// with `epoch`.
fn deform_subset(keep: usize, epoch: u64) -> RenderDeform {
    let full = walked_deform();
    RenderDeform {
        epoch,
        cells: full.cells.into_iter().take(keep).collect(),
        ..full
    }
}

/// **The partial-upload path lands the same texture a full upload would, and a
/// shrinking field clears the texels it vacated.**
///
/// Both are asserted the same way, and it is the strongest available: a renderer
/// that reached state *B* **incrementally** must produce a frame byte-identical
/// to a fresh renderer that only ever saw *B*. Any texel the incremental path
/// failed to write — or wrote in the wrong place — shows up immediately.
///
/// The vacated-texel case is the one the first cut got wrong: the dirty rect
/// covers what the cells *wrote*, not what they stopped writing, so a field that
/// shrank with the camera parked produced a smaller rect (or `None`) and last
/// frame's ruts stayed burned into the window for ever.
#[test]
fn the_incremental_window_matches_a_cold_one() {
    let Some(gpu) = gpu_or_skip() else { return };
    let target = HeadlessTarget::new(&gpu, W, H);
    let view = deform_view();

    let big = deform_subset(usize::MAX, 1);
    let small = deform_subset(3, 2);
    let empty = deform_subset(0, 3);
    assert!(big.cells.len() > small.cells.len() && small.drawable());

    let frame = |scene: &RenderScene, renderer: &mut EngineRenderer| {
        renderer.render(&gpu, scene, &view, &target.view, (W, H));
        target.read_rgba(&gpu).expect("readback")
    };
    let scene_of = |d: &RenderDeform| {
        let mut s = deform_scene();
        s.deform = Some(Arc::new(d.clone()));
        s.mark_dirty();
        s
    };

    // (1) GROWING then SHRINKING, all from one renderer and one camera — so
    //     every step after the first takes the PARTIAL path.
    let mut incremental = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
    let _ = frame(&scene_of(&big), &mut incremental);
    let after_small = frame(&scene_of(&small), &mut incremental);
    let after_empty = frame(&scene_of(&empty), &mut incremental);

    // (2) The same two states, each from a renderer that never saw the others.
    let mut cold_small = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
    let small_cold = frame(&scene_of(&small), &mut cold_small);
    let mut cold_empty = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
    let empty_cold = frame(&scene_of(&empty), &mut cold_empty);

    assert_eq!(
        after_small, small_cold,
        "a window that shrank incrementally does not match a cold one — texels \
         the vanished cells vacated were left holding their old depths"
    );
    assert_eq!(
        after_empty, empty_cold,
        "the last cell disappearing left its ruts burned into the window"
    );

    // ANTI-VACUITY: the three states really are different pictures, so the
    // equalities above are comparing something.
    let big_cold = {
        let mut r = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
        frame(&scene_of(&big), &mut r)
    };
    assert_ne!(big_cold, small_cold);
    assert_ne!(small_cold, empty_cold);

    // …and the shrink really did take the PARTIAL path: the camera never moved,
    // so the only reason to upload was the field changing, and the union with
    // the previous rect is what made that upload cover the vacated texels.
    let mut counted = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
    let _ = frame(&scene_of(&big), &mut counted);
    let after_big = counted.deform_uploads();
    assert!(after_big >= 1);
    let _ = frame(&scene_of(&small), &mut counted);
    assert!(
        counted.deform_uploads() > after_big,
        "shrinking the field with a parked camera uploaded nothing — the vacated \
         texels would still be holding their old depths"
    );

    // The empty state is the OTHER path and is worth saying out loud: a
    // projection with no cells is undrawable, so the uniform is disabled and the
    // window is not consulted at all — no upload is needed to stop drawing it.
    let before_empty = counted.deform_uploads();
    let _ = frame(&scene_of(&empty), &mut counted);
    assert_eq!(counted.deform_uploads(), before_empty);
}

// ── P27.4: virtual shadow maps reach pixels ─────────────────────────────────
//
// **Four NEW goldens, and not one existing frame moves.** `VsmSettings::enabled`
// is `false` by default, so every other scene in this file binds the renderer's
// permanent EMPTY shadow surface — a 1 × 1 depth texture and a four-byte table
// whose first word is not the magic — and `vsm_active()` is false, which makes
// `shadow_factor` take the cascaded path it always took and every analytic
// light's term an exact `* 1.0`. Verified the P17.2/P17.3/P22.1 way: the whole
// suite under `INF_BLESS_GOLDENS=1`, with `git status` reporting only the four
// new PNGs.
//
// What each one pins:
//
// * `vsm_directional` — the clipmap through the page table, on the CSM golden's
//   own caster/receiver arrangement, so the two paths can be looked at side by
//   side;
// * `vsm_spot` and `vsm_point` — the first spot and point shadows this engine
//   has ever drawn. `vsm_point` is the one that pins the cube-face seam: four
//   casters around one light send their shadows out through four different
//   faces;
// * `vsm_bias_grazing` — the two derived bias terms at the angle that breaks a
//   flat constant. A slope bias that is too small stripes the floor with acne
//   and a bias that is too large detaches every shadow from its caster; both
//   are visible in this frame and neither is present in it.
//
// The residency is fed by a readback ring pinned at `frame − 2`, so a VSM
// golden has to render a few frames before its shadows exist at all — which is
// why these four go through `check_vsm_golden` rather than `check_golden`.
// Six frames, on the same renderer, with the device pumped between them so
// arrival is a constant rather than a timing fact.

/// The settings the four virtual-shadow goldens render with: half the shipped
/// clipmap (32 pages a side over a 24 m level 0, a **5.86 mm** level-0 texel),
/// so a golden's atlas is 16 MiB rather than 64 and its levels are still real.
fn vsm_golden_settings() -> inf_render::VsmSettings {
    inf_render::VsmSettings {
        enabled: true,
        budget_bytes: 256 * 64 * 1024,
        clipmap_pages_per_side: 32,
        clipmap_levels: 8,
        first_level_extent_m: 12.0,
        spot_levels: 7,
        point_levels: 6,
        ..Default::default()
    }
}

/// [`check_golden_with`] over a **warmed** renderer: the same scene rendered
/// `VSM_GOLDEN_FRAMES` times before the frame that is compared, twice, so the
/// determinism gate is over the whole warm-up rather than over one cold frame.
fn check_vsm_golden(
    gpu: &GpuContext,
    name: &str,
    scene: &RenderScene,
    view: &RenderView,
    settings: RenderSettings,
) -> Vec<u8> {
    let a = render_warm(gpu, scene, view, settings);
    let b = render_warm(gpu, scene, view, settings);
    let (mean, max) = image_diff(&a, &b, W, H);
    assert!(
        mean < 0.005 && max < 0.05,
        "{name}: the virtual-shadow path is not deterministic (mean {mean}, max {max})"
    );
    let path = goldens_dir().join(format!("{name}.png"));
    if std::env::var("INF_BLESS_GOLDENS").is_ok() || read_png(&path).is_none() {
        write_png(&path, &a);
        eprintln!("golden {name}: wrote {}", path.display());
    } else if std::env::var("INF_GOLDEN_STRICT").is_ok() {
        let golden = read_png(&path).expect("golden png");
        let (mean, max) = image_diff(&a, &golden, W, H);
        assert!(
            within_tolerance(mean, max),
            "{name}: differs from golden (mean {mean}, max {max})"
        );
    }
    a
}

/// Frames a virtual-shadow golden warms for. The marking ring is pinned at
/// `frame − 2`, so nothing is resident before frame 2 and nothing is rasterized
/// before it; six leaves the residency a settled frame either side.
const VSM_GOLDEN_FRAMES: u64 = 6;

fn render_warm(
    gpu: &GpuContext,
    scene: &RenderScene,
    view: &RenderView,
    settings: RenderSettings,
) -> Vec<u8> {
    let target = HeadlessTarget::new(gpu, W, H);
    let mut renderer = EngineRenderer::new(gpu, HEADLESS_FORMAT);
    renderer.set_settings(settings);
    for _ in 0..VSM_GOLDEN_FRAMES {
        renderer.render(gpu, scene, view, &target.view, (W, H));
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    }
    target.read_rgba(gpu).expect("readback")
}

/// A white floor slab with its top face at `y = 0`, and boxes standing on it.
fn vsm_floor(casters: &[(f64, f64)]) -> RenderScene {
    let mut scene = RenderScene {
        grid_enabled: false,
        ..Default::default()
    };
    scene.instances.push(MeshInstance::lit(
        DVec3::new(0.0, -0.5, 0.0),
        Quat::IDENTITY,
        Vec3::new(60.0, 1.0, 60.0),
        [0.82, 0.82, 0.84, 1.0],
        1,
    ));
    for (i, (x, z)) in casters.iter().enumerate() {
        scene.instances.push(MeshInstance::lit(
            DVec3::new(*x, 0.75, *z),
            Quat::from_rotation_y(0.25),
            Vec3::new(1.4, 1.5, 1.4),
            [0.78, 0.42, 0.32, 1.0],
            i as u32 + 2,
        ));
    }
    scene.mark_dirty();
    scene
}

fn vsm_settings_on() -> RenderSettings {
    RenderSettings {
        vsm: vsm_golden_settings(),
        ..RenderSettings::default()
    }
}

/// **Virtual directional shadows** (P27.4): the clipmap read through the page
/// table, on `golden_csm`'s own caster/receiver arrangement.
///
/// Structural gate: the virtual path DARKENS the frame against the same scene
/// with no shadows at all, the scene stays lit, and the receiver's engagement
/// counter moved — which is the claim no pixel comparison can make.
#[test]
fn golden_vsm_directional() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mut scene = vsm_floor(&[(-2.2, 0.6), (1.2, -1.4), (2.8, 1.4)]);
    scene.lights.push(RenderLight {
        kind: LightKind::Directional,
        color: [1.0, 0.97, 0.9],
        intensity: 3.0,
        direction: Vec3::new(0.35, 0.86, -0.37).normalize(),
        position: DVec3::ZERO,
        range: 0.0,
        cast_shadows: true,
        ..RenderLight::default()
    });
    scene.mark_dirty();
    let view = look_view(DVec3::new(0.0, 5.0, 9.5), DVec3::new(0.0, 0.4, 0.0));

    let img = check_vsm_golden(&gpu, "vsm_directional", &scene, &view, vsm_settings_on());
    let off = render_warm(&gpu, &scene, &view, RenderSettings::default());
    let sum = |img: &[u8]| -> u64 {
        img.chunks(4)
            .map(|p| p[0] as u64 + p[1] as u64 + p[2] as u64)
            .sum()
    };
    assert!(
        sum(&img) < sum(&off),
        "virtual shadows should darken the receiving floor (on {} vs off {})",
        sum(&img),
        sum(&off)
    );
    assert!(
        img.chunks(4)
            .any(|p| p[0] as u16 + p[1] as u16 + p[2] as u16 > 200),
        "expected the scene to stay lit"
    );
}

/// **The engine's first spot shadow** (P27.4), through a single quadtree.
#[test]
fn golden_vsm_spot() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mut scene = vsm_floor(&[(0.0, 0.0), (2.6, 1.0)]);
    scene.lights.push(RenderLight {
        kind: LightKind::Spot,
        color: [1.0, 0.95, 0.88],
        intensity: 260.0,
        position: DVec3::new(0.0, 7.5, 5.0),
        direction: Vec3::new(0.0, 0.83, 0.55).normalize(),
        range: 45.0,
        inner_cos: 30f32.to_radians().cos(),
        outer_cos: 42f32.to_radians().cos(),
        cast_shadows: true,
    });
    scene.mark_dirty();
    let view = look_view(DVec3::new(0.0, 5.0, 9.0), DVec3::new(0.0, 0.3, -1.0));

    let img = check_vsm_golden(&gpu, "vsm_spot", &scene, &view, vsm_settings_on());
    let off = render_warm(&gpu, &scene, &view, RenderSettings::default());
    assert_ne!(img, off, "the spot cast no shadow");
}

/// **The engine's first point shadow** (P27.4), through the cube-face
/// quadtrees — four casters around one light, so four different faces answer.
#[test]
fn golden_vsm_point() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mut scene = vsm_floor(&[(-2.6, 0.0), (2.6, 0.0), (0.0, -2.6), (0.0, 2.6)]);
    scene.lights.push(RenderLight {
        kind: LightKind::Point,
        color: [1.0, 0.94, 0.88],
        intensity: 340.0,
        position: DVec3::new(0.0, 3.4, 0.0),
        range: 45.0,
        cast_shadows: true,
        ..RenderLight::default()
    });
    scene.mark_dirty();
    let view = look_view(DVec3::new(0.0, 8.5, 8.5), DVec3::new(0.0, 0.2, 0.0));

    let img = check_vsm_golden(&gpu, "vsm_point", &scene, &view, vsm_settings_on());
    let off = render_warm(&gpu, &scene, &view, RenderSettings::default());
    assert_ne!(img, off, "the point light cast no shadow");
}

/// **The bias, at the angle that breaks a flat constant** (P27.4): a low sun
/// over a large flat receiver, where a slope term that is too small stripes the
/// floor and one that is too large lifts every shadow off its caster.
#[test]
fn golden_vsm_bias_grazing() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mut scene = vsm_floor(&[(0.0, 0.0), (3.2, -1.6)]);
    scene.lights.push(RenderLight {
        kind: LightKind::Directional,
        color: [1.0, 0.95, 0.86],
        intensity: 3.2,
        // 34 degrees above the horizon, and the light is on the FAR side of the
        // casters so their shadows fall toward the camera rather than behind
        // them.
        direction: Vec3::new(0.18, 34f32.to_radians().sin(), -34f32.to_radians().cos()).normalize(),
        position: DVec3::ZERO,
        range: 0.0,
        cast_shadows: true,
        ..RenderLight::default()
    });
    scene.mark_dirty();
    let view = look_view(DVec3::new(0.0, 3.6, 9.0), DVec3::new(0.0, 0.3, 1.0));

    let img = check_vsm_golden(&gpu, "vsm_bias_grazing", &scene, &view, vsm_settings_on());
    let off = render_warm(&gpu, &scene, &view, RenderSettings::default());
    assert_ne!(img, off, "the grazing sun cast no shadow");
}

// ── the near-ground golden (wave TER2a, clause 3's gate) ────────────────────

/// **Render one frame with live virtual-texture pools installed.**
///
/// The harness's [`render_with`] cannot do this: `EngineRenderer::set_vt_pools`
/// takes the pools by value, and the determinism gate renders twice — so a
/// caller has to be able to build a *fresh* set for each frame. Hence a closure
/// rather than a value.
fn render_with_vt(
    gpu: &GpuContext,
    view: &RenderView,
    settings: RenderSettings,
    build: &dyn Fn(
        &GpuContext,
    ) -> (
        RenderScene,
        inf_render::vt_library::VtTextures,
        inf_render::vt::VtPools,
    ),
) -> Vec<u8> {
    let target = HeadlessTarget::new(gpu, W, H);
    let mut renderer = EngineRenderer::new(gpu, HEADLESS_FORMAT);
    renderer.set_settings(settings);
    let (scene, _lib, pools) = build(gpu);
    // `_lib` is held for the whole render deliberately: it owns the residency
    // the pools' geometry was built from, and dropping it early would leave the
    // atlas describing a table that no longer exists.
    renderer.set_vt_pools(Some(pools));
    renderer.render(gpu, &scene, view, &target.view, (W, H));
    target.read_rgba(gpu).expect("readback")
}

/// [`check_golden_with`] for a scene that needs virtual textures — the same
/// determinism gate, the same bless/strict rules, over a scene built twice.
fn check_golden_vt(
    gpu: &GpuContext,
    name: &str,
    view: &RenderView,
    settings: RenderSettings,
    build: &dyn Fn(
        &GpuContext,
    ) -> (
        RenderScene,
        inf_render::vt_library::VtTextures,
        inf_render::vt::VtPools,
    ),
) -> Vec<u8> {
    let a = render_with_vt(gpu, view, settings, build);
    let b = render_with_vt(gpu, view, settings, build);
    let (mean, max) = image_diff(&a, &b, W, H);
    assert!(
        mean < 0.005 && max < 0.05,
        "{name}: renderer not deterministic (mean {mean}, max {max})"
    );
    let path = goldens_dir().join(format!("{name}.png"));
    let bless = std::env::var("INF_BLESS_GOLDENS").is_ok();
    let strict = std::env::var("INF_GOLDEN_STRICT").is_ok();
    if bless || read_png(&path).is_none() {
        write_png(&path, &a);
        eprintln!("golden {name}: wrote {}", path.display());
    } else if strict {
        let golden = read_png(&path).expect("golden png");
        let (mean, max) = image_diff(&a, &golden, W, H);
        eprintln!("golden {name}: mean {mean:.6}, max {max:.6} against the committed frame");
        assert!(
            within_tolerance(mean, max),
            "{name}: differs from golden (mean {mean}, max {max})"
        );
    }
    a
}

/// The island's own four ground sets, in its own splat order.
const GROUND_ORDER: [inf_material::ground::GroundKind; 4] = [
    inf_material::ground::GroundKind::Grass,
    inf_material::ground::GroundKind::Rock,
    inf_material::ground::GroundKind::ForestFloor,
    inf_material::ground::GroundKind::Sand,
];

/// The four ground sets the island binds, registered as real virtual textures.
///
/// **The real committed content**, synthesised through the same
/// `inf_material::ground` door `samples/ground/` is written from — not a fixture
/// that resembles it. A golden over a stand-in would depict a surface the engine
/// never ships.
///
/// # Two deliberate departures from the shipped configuration, both stated
///
/// * **The pool is `Rgba8`.** The shipped atlas is BC1 + BC5 and the containers
///   here are BC1 + BC5, but a headless CI adapter need not expose
///   `TEXTURE_COMPRESSION_BC`; the residency door transcodes on the way in
///   (`TiledTextureReader::tile_rgba8`), which is the same path a mobile tier
///   takes and is adapter-robust. What reaches the frame is the same texels.
///   One arm, therefore, and not two — the transcode tier collapses every block
///   format onto RGBA8, which is exactly what `build_vt_level` does on such an
///   adapter.
/// * **Mip 0 is not requested.** The wants below start at mip 1, so a 1 024²
///   albedo contributes 27 pages rather than 92 and the four sets fit inside a
///   320-page pool. At 320×180 with a one-metre eye a screen pixel covers
///   several millimetres of ground and mip 1 (3.9 mm a texel over a 2 m tile) is
///   at or finer than that — so nothing the frame can show is lost, and the test
///   does not allocate 37 MB on a software adapter.
fn ground_vt_library(
    gpu: &GpuContext,
) -> (
    inf_render::vt_library::VtTextures,
    inf_render::vt::VtPools,
    [inf_render::VtTextureSet; 4],
) {
    let mut lib = inf_render::vt_library::VtTextures::new(inf_render::VtPoolConfig {
        format: inf_render::PageFormat::Rgba8,
        stored_tile_size: inf_render::STORED_TILE_SIZE,
        budget_bytes: inf_render::PageFormat::Rgba8.page_bytes(inf_render::STORED_TILE_SIZE) * 320,
        max_texture_dim: 8192,
        trilinear: false,
        // Unthrottled: this is a still frame, and a deferred page would make the
        // picture a fact about the upload budget rather than about the content.
        upload_budget_bytes: 0,
    })
    .0;
    // **The library's own settings, per slot** (wave IASSET2). This used to
    // hard-code `TextureCompression::Bc1` for all four maps, which was right
    // while `inf_material::ground` authored all four that way and became a
    // fixture that depicts a surface the engine no longer ships the moment the
    // normal maps moved to BC5. Calling the library's own functions is what
    // keeps "the real committed content" true rather than remembered.
    let tile = |rgba: Vec<u8>, n: u32, settings| {
        inf_material::build_tiled_texture(rgba, n, n, settings)
            .expect("the ground set tiles")
            .into_bytes()
    };
    let albedo_s = inf_material::ground::albedo_settings();
    let normal_s = inf_material::ground::normal_settings();
    let data_s = inf_material::ground::data_settings();
    let albedo_n = inf_material::ground::GROUND_ALBEDO_EXTENT;
    let map_n = inf_material::ground::GROUND_MAP_EXTENT;
    let mut guid: u128 = 0x9E20_0000;
    let mut maps = Vec::new();
    for kind in GROUND_ORDER {
        let g = inf_material::ground::synthesize(kind);
        let (a, n, o, d) = (guid + 1, guid + 2, guid + 3, guid + 4);
        lib.register_or_record(a, Arc::new(tile(g.albedo, albedo_n, albedo_s)))
            .expect("albedo registers");
        lib.register_or_record(n, Arc::new(tile(g.normal, map_n, normal_s)))
            .expect("normal registers");
        lib.register_or_record(o, Arc::new(tile(g.orm, map_n, data_s)))
            .expect("orm registers");
        let detail = g.detail.map(|rgba| {
            lib.register_or_record(d, Arc::new(tile(rgba, map_n, normal_s)))
                .expect("detail registers");
            d
        });
        maps.push(inf_render::vt_library::VtMaterialMaps {
            albedo: Some(a),
            normal: Some(n),
            orm: Some(o),
            detail,
            detail_scale_q8: (kind.detail_scale() * 256.0) as u16,
            // Wave ROAD1: the library's own rate, read from the library rather
            // than spelled — and every one of the four splat kinds answers 0.0,
            // because terrain has already divided world metres by its layer's
            // `tex_scale`. So this frame is byte-identical to the pre-ROAD1 one
            // BY THE LIBRARY'S OWN RULE rather than by a literal zero here.
            uv_tiling_q8: inf_render::scene::uv_tiling_q8(kind.mesh_uv_tiling_m() as f32),
        });
        guid += 6;
    }
    let mut pools = inf_render::vt::VtPools::new(&gpu.device, &gpu.queue, lib.residency(), false);
    // Everything from mip 1 down — see the note above.
    let mut wants = Vec::new();
    for h in 0..maps.len() as u32 * 4 {
        let handle = inf_vt::VtTextureHandle(h);
        let Some(desc) = lib.residency().desc(handle) else {
            break;
        };
        for m in 1..desc.mip_count() {
            let g = desc.mips[m as usize];
            for y in 0..g.tiles_y {
                for x in 0..g.tiles_x {
                    wants.push(inf_vt::VtWant::new(handle, inf_vt::TileCoord::new(m, x, y)));
                }
            }
        }
    }
    let (txn, report) = lib.sync(&gpu.device, &gpu.queue, &mut pools, &wants);
    assert_eq!(
        txn.deferred, 0,
        "the ground pyramid did not fit the fixture pool"
    );
    assert!(
        report.missing.is_empty(),
        "{} pages missing",
        report.missing.len()
    );
    let sets = std::array::from_fn(|k| {
        let s = lib.set_for_maps(&maps[k]);
        assert!(!s.is_none(), "ground set {k} never went warm");
        s
    });
    (lib, pools, sets)
}

/// A patch of the island's own ground, seen from **one metre up** — the height
/// of an eye.
///
/// Sixteen metres square at half-metre samples, rising gently to the north with
/// a bank on the east side, and splat weights that put all four layers on it:
/// sand at the water's edge, grass over the flat, forest floor inland, rock on
/// the bank. The camera stands on it and looks along it, which is the shot every
/// previous terrain golden could not take — before TER2a, ground below about
/// three metres had no colour signal at all and this frame would have been one
/// flat green with a shading normal on it.
fn ground_close_terrain(sets: [inf_render::VtTextureSet; 4]) -> RenderTerrain {
    let res = 33u32;
    let mps = 0.5f64;
    let span = (res as f64 - 1.0) * mps;
    let height = |x: f64, z: f64| -> f64 {
        let bank = ((x - 10.0) / 4.0).clamp(0.0, 1.0);
        0.35 * z + 1.6 * bank * bank
    };
    // Four bands across +X, feathered, in the island's own layer order:
    // grass, rock, forest floor, sand.
    let weight = |x: f64| -> [u8; 4] {
        // The four bands are packed into the near half of the patch so all four
        // are in one frame at a metre's eye height — a golden that showed two of
        // them would prove the branch runs and not that the four blend.
        let u = (x / (2.0 * span)).clamp(0.0, 1.0);
        let tent = |c: f64| (1.0 - (u - c).abs() * 5.0).max(0.0);
        let raw = [tent(0.22), tent(0.58), tent(0.40), tent(0.05)];
        let s: f64 = raw.iter().sum::<f64>().max(1e-6);
        let mut out = [0u8; 4];
        let mut acc = 0i32;
        for k in 0..4 {
            out[k] = (raw[k] / s * 255.0).round() as u8;
            acc += out[k] as i32;
        }
        out[0] = (out[0] as i32 + (255 - acc)).clamp(0, 255) as u8;
        out
    };
    let mut tiles = Vec::new();
    for tx in 0..2 {
        for tz in 0..2 {
            let (ox, oz) = (tx as f64 * span, tz as f64 * span);
            let mut heights = vec![0f32; (res * res) as usize];
            let mut weights = vec![[0u8; 4]; (res * res) as usize];
            let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
            for j in 0..res {
                for i in 0..res {
                    let (wx, wz) = (ox + i as f64 * mps, oz + j as f64 * mps);
                    let h = height(wx, wz) as f32;
                    heights[(j * res + i) as usize] = h;
                    weights[(j * res + i) as usize] = weight(wx);
                    lo = lo.min(h);
                    hi = hi.max(h);
                }
            }
            tiles.push(RenderTerrainTile {
                key: TerrainTileKey::lod0((tx, tz)),
                origin: DVec3::new(ox, 0.0, oz),
                heights,
                weights,
                biomes: Vec::new(),
                height_bounds: (lo, hi),
                holes: Vec::new(),
                version: 1,
            });
        }
    }
    let layers = std::array::from_fn(|k| RenderTerrainLayer {
        albedo: GROUND_ORDER[k].base_color(),
        roughness: GROUND_ORDER[k].roughness(),
        tex_scale: GROUND_ORDER[k].tex_scale_m() as f32,
        vt: sets[k],
    });
    RenderTerrain {
        id: 0,
        tile_resolution: res,
        meters_per_sample: mps,
        tiles,
        layers,
        macro_variation: 0.15,
        biome_palette: Vec::new(),
    }
}

/// **The near-ground golden** (wave TER2a) — the 58th, on the additive branch.
///
/// What it depicts is the thing this wave exists to make exist: real PBR ground
/// materials, blended by real splat weights, at the range a player's eye is at.
/// Every terrain golden before it is an overlook, because below about three
/// metres there was nothing to look at — no colour signal and no height signal,
/// just a 1 m central-difference shading normal on a flat green.
///
/// The structural arm beside the pixels is a **difference**, on
/// `terrain_vt_fragment`'s own reasoning: a frame that is merely "not blank"
/// proves nothing, so this measures the textured frame against the identical
/// frame with **no pools installed** — same scene, same weights, same camera,
/// same layer colours, and the branch `vt_active()` guards simply not taken.
#[test]
fn golden_ground_close() {
    let Some(gpu) = gpu_or_skip() else { return };
    // A metre above the ground under the camera, looking along the bands.
    let eye = DVec3::new(1.0, 0.35 * 8.0 + 1.0, 8.0);
    let view = look_view(eye, DVec3::new(21.0, 3.2, 12.0));
    let build = |gpu: &GpuContext| {
        let (lib, pools, sets) = ground_vt_library(gpu);
        let scene = RenderScene {
            grid_enabled: false,
            terrains: vec![ground_close_terrain(sets)],
            ..Default::default()
        };
        (scene, lib, pools)
    };
    let img = check_golden_vt(
        &gpu,
        "ground_close",
        &view,
        RenderSettings::default(),
        &build,
    );

    // The control: the identical scene with no pools, so `vt_active()` is false
    // and the four layers shade off their scalar colours alone.
    let flat = {
        let target = HeadlessTarget::new(&gpu, W, H);
        let mut renderer = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
        let (_lib, _pools, sets) = ground_vt_library(&gpu);
        let scene = RenderScene {
            grid_enabled: false,
            terrains: vec![ground_close_terrain(sets)],
            ..Default::default()
        };
        renderer.set_vt_pools(None);
        renderer.render(&gpu, &scene, &view, &target.view, (W, H));
        target.read_rgba(&gpu).expect("readback")
    };

    // **Detail, measured.** A textured ground has high-frequency variation a
    // flat-albedo one cannot have: count horizontally adjacent pixel pairs whose
    // luminance differs by more than four levels. The scalar frame's only
    // sources of that are the procedural grain and the shading normal, both of
    // which are in BOTH frames — so the difference is the material's own texels.
    let steps = |img: &[u8]| -> u64 {
        let mut n = 0u64;
        for y in 0..H {
            for x in 1..W {
                let a = px(img, x - 1, y);
                let b = px(img, x, y);
                let la = 0.2126 * a[0] as f64 + 0.7152 * a[1] as f64 + 0.0722 * a[2] as f64;
                let lb = 0.2126 * b[0] as f64 + 0.7152 * b[1] as f64 + 0.0722 * b[2] as f64;
                if (la - lb).abs() > 4.0 {
                    n += 1;
                }
            }
        }
        n
    };
    let (on, off) = (steps(&img), steps(&flat));
    eprintln!(
        "ground_close: {on} textured luminance steps against {off} untextured ({:.2}x)",
        on as f64 / off.max(1) as f64
    );
    assert!(
        on > off * 2,
        "the ground is no busier textured ({on}) than untextured ({off}) — the \
         virtual-texture branch did not run"
    );
    // …and it is not merely noise: the frame must still be recognisably ground
    // rather than a spray of texels, so the two frames' MEAN luminance stays
    // close. A material that resolved to garbage would move it — and so would
    // the wave's own second finding, which is what this tolerance is set from.
    //
    // **THE TOLERANCE IS 20 %, AND THAT NUMBER IS A MUTATION** (TER2a audit).
    // It was 50 %, and at 50 % this arm does not catch the defect it is cited
    // for. Restoring the `vt_surface` factor bug — a terrain layer's scalar
    // passed as a glTF `baseColorFactor` instead of as the no-texture fallback —
    // renders this frame at a mean luminance of **48.6 against 95.5**, a gap of
    // 46.9 against a 50 % allowance of 47.75. It passed **by 0.85 of a level**,
    // and the `steps` arm above does not see it at all (a multiply darkens the
    // texel variation without removing it: 13 214 steps against 175, still 75×).
    // The only thing that caught it was the committed image under
    // `INF_GOLDEN_STRICT=1` — which **CI never sets**, so on every leg CI
    // actually runs, the wave's own finding was unpinned.
    //
    // 20 % separates the two by an order of magnitude in both directions: the
    // real signal is **1.0 level, 1.1 %** (94.5 textured against 95.5), and the
    // defect is 49 %. Mutation-verified in both directions.
    let mean = |img: &[u8]| -> f64 {
        let s: f64 = img
            .chunks_exact(4)
            .map(|p| 0.2126 * p[0] as f64 + 0.7152 * p[1] as f64 + 0.0722 * p[2] as f64)
            .sum();
        s / f64::from(W * H)
    };
    let (ma, mb) = (mean(&img), mean(&flat));
    eprintln!(
        "ground_close: mean luma {ma:.1} textured against {mb:.1} untextured \
         ({:+.1} %, allowance ±20 %)",
        (ma - mb) / mb * 100.0
    );
    assert!(
        (ma - mb).abs() < mb * 0.20,
        "the textured frame's mean luminance ({ma:.1}) is {:.0} % from the \
         untextured one's ({mb:.1}). A terrain layer's albedo is the NO-TEXTURE \
         FALLBACK, not a glTF `baseColorFactor`: if `terrain_layers` passes it \
         into `vt_surface` as a factor, a bound ground shades at the product of \
         its own colour and itself and the frame comes out about twice too dark. \
         That is the wave-TER2a finding this tolerance is set from, and at the \
         50 % it was first written with this arm passed with the defect in place.",
        (ma - mb).abs() / mb * 100.0
    );
}

/// **How bright a river is against its own bank, at two hundred metres** (wave
/// ROAD1b, clause 5) — the number the sparkle clause asked for, measured
/// instead of described.
///
/// # The user's sentence, and what a number can say about it
///
/// The ROAD1 audit's honest close was that a 1.5–4 m creek "reads as a dark line
/// in gravel". That is a **contrast** claim, so this renders the river scene from
/// 200 m, finds the river's pixels by differencing against the same frame with
/// the water removed, and compares their mean luminance with the mean of the
/// pixels immediately beside them. A dark line is a ratio below one; a river you
/// can see is one above it, and a sparkling river is one whose brightest pixels
/// are far above its own mean.
///
/// # What it does NOT do
///
/// It does not assert a target. The sparkle itself — a normal scroll advected at
/// `river_flow_m_s` and a sun glint that is lit by the SUN — is routed to PAR3/
/// PAR4 with these numbers, because `water.wgsl` is built on time NOT reaching
/// the GPU (each wave's phase arrives already reduced in f64 on the CPU) and an
/// advection term is a uniform lane and a design decision, not an edit. What is
/// asserted is that the instrument is not vacuous: the river was found, and it is
/// not the same colour as the ground.
#[test]
fn road1b_a_rivers_contrast_against_its_bank_at_200m() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (mut scene, _) = tod_scene(15.0 * 3600.0);
    scene.grid_enabled = false;
    scene.terrains = vec![hill_terrain(33, 4.0, 2, 2)];
    let ripple = WaveField::from_spec(&WaveSpec {
        amplitude_m: 0.06,
        wavelength_m: 4.0,
        steepness: 0.12,
        wind_x: 1.0,
        wind_z: 0.0,
        spread_rad: 60f64.to_radians(),
        seed: 0,
        count: 3,
    });
    let points = [
        DVec3::new(4.0, 9.0, 4.0),
        DVec3::new(40.0, 7.0, 24.0),
        DVec3::new(70.0, 5.0, 60.0),
        DVec3::new(110.0, 3.0, 90.0),
    ];
    let path = inf_render::RiverPath::from_points(
        &points,
        false,
        inf_math::spline::SplineInterp::CatmullRom,
        &inf_render::RiverProfile {
            width_start_m: 6.0,
            width_end_m: 14.0,
            depth_start_m: 1.0,
            depth_end_m: 2.5,
            flow_speed_m_s: 3.5,
        },
    );
    scene.waters = vec![RenderWater {
        level_m: 9.0,
        flow_speed_m_s: path.flow_speed_m_s,
        frames: path
            .frames
            .iter()
            .map(inf_render::WaterFrame::from)
            .collect(),
        ..water_body(WaterKindGpu::River, ripple)
    }];

    // **Two hundred metres**, and stated as a distance rather than picked as a
    // camera: the target is the reach at (70, 60) and the eye is placed on a
    // shallow descent 200 m from it, which is the range the ROAD1 audit's
    // sentence is about.
    let target = DVec3::new(70.0, 5.0, 60.0);
    let dir = DVec3::new(-0.72, 0.30, -0.62).normalize();
    let eye = target + dir * 200.0;
    let view = look_view(eye, target);
    assert!(
        ((eye - target).length() - 200.0).abs() < 1.0e-9,
        "the eye is not 200 m from the reach"
    );

    let wet = render(&gpu, &scene, &view);
    let dry = render(&gpu, &without_water(&scene), &view);

    // The river's pixels: the ones the water changed. The bank's: the ones
    // beside them that it did not.
    let mut is_river = vec![false; (W * H) as usize];
    for i in 0..(W * H) as usize {
        let (pa, pb) = (&wet[i * 4..i * 4 + 3], &dry[i * 4..i * 4 + 3]);
        is_river[i] = pa
            .iter()
            .zip(pb)
            .any(|(x, y)| (*x as i32 - *y as i32).abs() > 6);
    }
    let mut river: Vec<f32> = Vec::new();
    let mut bank: Vec<f32> = Vec::new();
    for y in 1..H - 1 {
        for x in 1..W - 1 {
            let i = (y * W + x) as usize;
            let p = px(&wet, x, y);
            let l = (p[0] as f32 + p[1] as f32 + p[2] as f32) / 3.0;
            if is_river[i] {
                river.push(l);
            } else if [
                ((y - 1) * W + x) as usize,
                ((y + 1) * W + x) as usize,
                (y * W + x - 1) as usize,
                (y * W + x + 1) as usize,
            ]
            .iter()
            .any(|k| is_river[*k])
            {
                bank.push(l);
            }
        }
    }
    let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len().max(1) as f32;
    let (rm, bm) = (mean(&river), mean(&bank));
    let brightest = river.iter().copied().fold(0.0f32, f32::max);
    println!(
        "ROAD1b RIVER | at 200 m: {} river px (mean {rm:.1}/255), {} bank px (mean {bm:.1}/255); contrast {:.3}x, brightest river px {brightest:.1} = {:.2}x its own mean",
        river.len(),
        bank.len(),
        rm / bm.max(1.0e-6),
        brightest / rm.max(1.0e-6)
    );
    assert!(
        river.len() > 50 && bank.len() > 50,
        "the instrument found {} river px and {} bank px — it is measuring \
         nothing",
        river.len(),
        bank.len()
    );
    assert!(
        (rm - bm).abs() > 1.0,
        "the river and its bank are the same colour to within a level, so this \
         frame cannot say anything about contrast"
    );
}
