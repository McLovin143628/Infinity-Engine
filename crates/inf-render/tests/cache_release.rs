//! **The caches release what they cache** (Hardening D).
//!
//! Every pass in this tree that materializes GPU resources per unit of content
//! reconciles its cache with the projection *inside* the body its early-out
//! guards. That is correct on every frame the node runs — and invisible on the
//! one transition that matters: the frame the last unit of content leaves the
//! scene, which is exactly when the guard fires and the reconciliation never
//! runs. `VoxelNode` and `FractureNode` have carried the release on the early-out
//! since P21/P22 ("the one transition the early-out would otherwise hide: the
//! last volume leaving the scene"); `ClassicVgeomNode` has carried `retain_live`
//! *before* its early-out since P18.3. The terrain and meshlet nodes did not.
//!
//! # Why these arms exist at all
//!
//! **A stranded cache renders identically to a released one.** There is no pixel,
//! no golden and no command-stream counter that can tell the two apart — which is
//! why the defect survived every gate in the tree. So the assertions here read the
//! *maps*: after the content leaves, the cache count is zero and the streamer's
//! resident bytes are zero. Assert the WORLD, not the report (P21).
//!
//! Skips cleanly with no GPU adapter, like every GPU path in this repo.

use std::sync::Arc;

use glam::{DVec3, Quat, Vec3};
use inf_math::FloatingOrigin;
use inf_render::{
    EngineRenderer, GiSettings, GpuContext, HeadlessTarget, RenderScene, RenderSettings,
    RenderTerrain, RenderTerrainTile, RenderView, SkinnedInstance, SkinnedMeshData, SkinnedVertex,
    TerrainTileKey, VgeomAsset, VgeomInstance, VgeomMesh, VgeomSettings, HEADLESS_FORMAT,
};
use inf_vgeom::test_support::dense_grid_mesh;

const W: u32 = 160;
const H: u32 = 120;
const RES: u32 = 16;
const MPS: f64 = 1.0;
const ASSET: u128 = 0x00CA_0FEE_0001;

fn gpu_or_skip(name: &str) -> Option<GpuContext> {
    match GpuContext::headless() {
        Ok(gpu) => Some(gpu),
        Err(e) => {
            eprintln!("{name}: no GPU adapter ({e}) — skipping");
            None
        }
    }
}

fn view() -> RenderView {
    RenderView {
        origin: FloatingOrigin::new(DVec3::ZERO),
        eye_world: DVec3::new(0.0, 12.0, 24.0),
        forward: Vec3::new(0.0, -0.4, -1.0).normalize(),
        up: Vec3::Y,
        fov_y: 60f32.to_radians(),
        near: 0.05,
        width: W,
        height: H,
        ortho: None,
    }
}

/// One flat level-0 tile at `coord`.
fn tile_at(coord: (i32, i32)) -> RenderTerrainTile {
    let span = (RES - 1) as f64 * MPS;
    RenderTerrainTile {
        key: TerrainTileKey::lod0(coord),
        origin: DVec3::new(coord.0 as f64 * span, 0.0, coord.1 as f64 * span),
        heights: vec![0.0; (RES * RES) as usize],
        weights: Vec::new(),
        biomes: Vec::new(),
        height_bounds: (0.0, 0.0),
        holes: Vec::new(),
        version: 1,
    }
}

fn terrain_scene(tiles: usize) -> RenderScene {
    let mut scene = RenderScene {
        grid_enabled: false,
        ..Default::default()
    };
    if tiles > 0 {
        scene.terrains.push(RenderTerrain {
            id: 7,
            tile_resolution: RES,
            meters_per_sample: MPS,
            tiles: (0..tiles as i32).map(|x| tile_at((x, 0))).collect(),
            ..Default::default()
        });
    }
    scene.mark_dirty();
    scene
}

fn vgeom_scene(mesh: &Arc<VgeomMesh>, instances: bool) -> RenderScene {
    let mut scene = RenderScene {
        grid_enabled: false,
        vgeom_assets: vec![VgeomAsset::from_mesh(ASSET, mesh).expect("index the vmesh")],
        ..Default::default()
    };
    if instances {
        scene.vgeom_instances.push(VgeomInstance::lit(
            ASSET,
            DVec3::ZERO,
            Quat::IDENTITY,
            Vec3::splat(4.0),
            [0.7, 0.5, 0.3, 1.0],
            1,
        ));
    }
    scene.mark_dirty();
    scene
}

fn vgeom_settings(enabled: bool) -> RenderSettings {
    RenderSettings {
        vgeom: VgeomSettings {
            enabled,
            occlusion: false,
            two_pass: false,
            visbuffer: false,
            ..VgeomSettings::default()
        },
        ..RenderSettings::default()
    }
}

/// **The terrain finding (lens 2 H1).** `TerrainNode::run` returns before
/// `sync_textures` when no terrain has a resident tile, and `sync_textures` owns
/// the only two eviction paths there are. So a level switch (or every tile
/// streaming out) used to strand the whole per-tile texture cache — four textures
/// per tile — plus one splat-material slot per terrain, for the renderer's life.
#[test]
fn terrain_releases_its_tile_cache_when_the_last_tile_leaves() {
    let Some(gpu) = gpu_or_skip("terrain_releases_its_tile_cache_when_the_last_tile_leaves") else {
        return;
    };
    let target = HeadlessTarget::new(&gpu, W, H);
    let mut renderer = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
    let view = view();

    let populated = terrain_scene(4);
    renderer.render(&gpu, &populated, &view, &target.view, (W, H));
    let (tiles, materials) = renderer.terrain_cache_counts();
    assert_eq!(
        (tiles, materials),
        (4, 1),
        "a four-tile terrain should cache four tiles and one material slot"
    );

    // The transition: the level's terrain content is gone. A scene with the
    // terrain still listed but nothing resident takes the same branch, which is
    // the streamed-out case.
    let empty = terrain_scene(0);
    renderer.render(&gpu, &empty, &view, &target.view, (W, H));
    assert_eq!(
        renderer.terrain_cache_counts(),
        (0, 0),
        "the tile textures and the splat-material slot must not outlive the content"
    );

    // And it comes back: the release is not a one-way door.
    renderer.render(&gpu, &populated, &view, &target.view, (W, H));
    assert_eq!(
        renderer.terrain_cache_counts(),
        (4, 1),
        "a terrain that returns is re-cached"
    );
}

/// The other half of the same branch: a terrain that is *listed* but has streamed
/// every tile out is the transition the guard reads as "no terrain".
#[test]
fn a_terrain_that_streams_every_tile_out_releases_too() {
    let Some(gpu) = gpu_or_skip("a_terrain_that_streams_every_tile_out_releases_too") else {
        return;
    };
    let target = HeadlessTarget::new(&gpu, W, H);
    let mut renderer = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
    let view = view();

    renderer.render(&gpu, &terrain_scene(2), &view, &target.view, (W, H));
    assert_eq!(renderer.terrain_cache_counts().0, 2);

    // Same terrain id, zero resident tiles — `tiles.is_empty()` for every terrain.
    let mut streamed_out = RenderScene {
        grid_enabled: false,
        ..Default::default()
    };
    streamed_out.terrains.push(RenderTerrain {
        id: 7,
        tile_resolution: RES,
        meters_per_sample: MPS,
        tiles: Vec::new(),
        ..Default::default()
    });
    streamed_out.mark_dirty();
    renderer.render(&gpu, &streamed_out, &view, &target.view, (W, H));
    assert_eq!(
        renderer.terrain_cache_counts(),
        (0, 0),
        "residency reaching zero must release, not park"
    );
}

/// **The meshlet finding (lens 2 H2).** `plan_cluster_pages` returns before the
/// streamer's plan when vgeom is off or the scene carries none, and `plan` →
/// `plan.dropped` → `draws.remove` is the only eviction chain there is. Two
/// transitions were therefore invisible: the content leaving, and the setting
/// being switched off. `stream_report` kept publishing the stale floor either way,
/// which is what the unified arbiter reserves against.
#[test]
fn vgeom_releases_residency_when_the_content_leaves() {
    let Some(gpu) = gpu_or_skip("vgeom_releases_residency_when_the_content_leaves") else {
        return;
    };
    let target = HeadlessTarget::new(&gpu, W, H);
    let mut renderer = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
    renderer.set_settings(vgeom_settings(true));
    let view = view();
    let mesh = Arc::new(dense_grid_mesh(24));

    let populated = vgeom_scene(&mesh, true);
    for _ in 0..3 {
        renderer.render(&gpu, &populated, &view, &target.view, (W, H));
    }
    let hot = renderer.vgeom_stream_report();
    assert!(
        hot.stats.resident_bytes > 0 && hot.stats.assets == 1,
        "the fixture must actually be resident before a release can mean anything \
         (assets {}, bytes {})",
        hot.stats.assets,
        hot.stats.resident_bytes
    );
    assert!(
        hot.floor_bytes > 0,
        "the arbiter floor must be non-zero first"
    );

    // The transition: the scene stops carrying vgeom.
    let empty = RenderScene {
        grid_enabled: false,
        ..Default::default()
    };
    renderer.render(&gpu, &empty, &view, &target.view, (W, H));
    let cold = renderer.vgeom_stream_report();
    assert_eq!(cold.stats.assets, 0, "every asset's residency is released");
    assert_eq!(cold.stats.resident_bytes, 0, "and its pool blocks with it");
    assert_eq!(
        cold.floor_bytes, 0,
        "and the arbiter stops reserving a floor for content nothing draws"
    );
    assert!(
        cold.pages.is_empty() && cold.floor_lod.is_empty(),
        "the per-asset report is emptied, not frozen at its last value"
    );
    assert!(
        cold.stats.evictions >= hot.stats.resident_pages as u64,
        "the release is an eviction of every resident page, counted"
    );
}

/// The setting's own off-transition — the second door `plan_cluster_pages`'s guard
/// hides. The scene still carries the asset; nothing is allowed to draw it.
#[test]
fn vgeom_releases_residency_when_the_setting_goes_off() {
    let Some(gpu) = gpu_or_skip("vgeom_releases_residency_when_the_setting_goes_off") else {
        return;
    };
    let target = HeadlessTarget::new(&gpu, W, H);
    let mut renderer = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
    renderer.set_settings(vgeom_settings(true));
    let view = view();
    let mesh = Arc::new(dense_grid_mesh(24));
    let scene = vgeom_scene(&mesh, true);

    for _ in 0..3 {
        renderer.render(&gpu, &scene, &view, &target.view, (W, H));
    }
    assert!(renderer.vgeom_stream_report().stats.resident_bytes > 0);

    renderer.set_settings(vgeom_settings(false));
    renderer.render(&gpu, &scene, &view, &target.view, (W, H));
    let cold = renderer.vgeom_stream_report();
    assert_eq!(
        (
            cold.stats.assets,
            cold.stats.resident_bytes,
            cold.floor_bytes
        ),
        (0, 0, 0),
        "switching the meshlet path off releases what it was streaming"
    );

    // Back on: the streamer re-pages rather than staying dead.
    renderer.set_settings(vgeom_settings(true));
    for _ in 0..3 {
        renderer.render(&gpu, &scene, &view, &target.view, (W, H));
    }
    assert!(
        renderer.vgeom_stream_report().stats.resident_bytes > 0,
        "the release must not be a one-way door"
    );
}

/// A one-triangle skinned mesh whose every vertex is dominated by joint 0.
fn skinned_mesh() -> Arc<SkinnedMeshData> {
    let v = |x: f32, y: f32| SkinnedVertex {
        pos: [x, y, 0.0],
        normal: [0.0, 0.0, 1.0],
        uv: [0.0, 0.0],
        joints: [0, 0, 0, 0],
        weights: [1.0, 0.0, 0.0, 0.0],
    };
    Arc::new(SkinnedMeshData {
        vertices: vec![v(-0.5, 0.0), v(0.5, 0.0), v(0.0, 1.0)],
        indices: vec![0, 1, 2],
    })
}

fn gi_settings(enabled: bool) -> RenderSettings {
    RenderSettings {
        gi: GiSettings {
            enabled,
            ..GiSettings::default()
        },
        vgeom: VgeomSettings {
            enabled: true,
            occlusion: false,
            two_pass: false,
            visbuffer: false,
            ..VgeomSettings::default()
        },
        ..RenderSettings::default()
    }
}

/// **The GI finding (lens 1 M1 / m8, lens 2 H4 / M1).** `joint_boxes` was keyed on
/// `Arc::as_ptr` **without holding the `Arc`** — an address is an identity only
/// while its allocation lives — and neither it nor `meshlet_spheres` was ever
/// evicted. Both are now retained to what the frame's scene names, and the
/// joint-box entry holds the mesh, which is what makes the pointer key sound.
///
/// The `ptr_eq` guard itself is unarmed **by construction**, exactly as
/// `vsm_raster::sync_skinned` records of its own: with the `Arc` held it can never
/// be false for a live key. It is there so that dropping the field breaks a
/// compile-visible invariant instead of silently lighting with a dead mesh.
#[test]
fn gi_holds_only_the_content_the_scene_still_names() {
    let Some(gpu) = gpu_or_skip("gi_holds_only_the_content_the_scene_still_names") else {
        return;
    };
    let target = HeadlessTarget::new(&gpu, W, H);
    let mut renderer = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
    renderer.set_settings(gi_settings(true));
    let view = view();
    let mesh = dense_grid_mesh(24);
    let mesh = Arc::new(mesh);

    let mut scene = vgeom_scene(&mesh, true);
    scene.skinned_meshes.push(skinned_mesh());
    scene.skinned.push(SkinnedInstance {
        blend: 0,
        cutoff: 0.5,
        vt: Default::default(),
        translation: DVec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
        color: [0.7, 0.6, 0.5, 1.0],
        metallic: 0.0,
        roughness: 0.6,
        emissive: [0.0; 3],
        id: 2,
        mesh: 0,
        palette: inf_render::identity_palette(),
        shadow: inf_render::SkinnedShadow::BindSphere,
        sections: Vec::new(),
    });
    scene.mark_dirty();
    renderer.render(&gpu, &scene, &view, &target.view, (W, H));
    assert_eq!(
        renderer.gi_cache_counts(),
        (1, 1),
        "one skinned mesh and one vgeom asset should be cached"
    );

    // A *different* skinned mesh and a *different* asset id: the editor's
    // content-addressed re-import, in miniature. The superseded entries must not
    // accrete.
    let mut swapped = vgeom_scene(&mesh, true);
    swapped.vgeom_assets[0].id = ASSET ^ 0xFF;
    swapped.vgeom_instances[0].asset = ASSET ^ 0xFF;
    swapped.skinned_meshes.push(skinned_mesh());
    swapped.skinned.push(SkinnedInstance {
        blend: 0,
        cutoff: 0.5,
        vt: Default::default(),
        translation: DVec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
        color: [0.7, 0.6, 0.5, 1.0],
        metallic: 0.0,
        roughness: 0.6,
        emissive: [0.0; 3],
        id: 2,
        mesh: 0,
        palette: inf_render::identity_palette(),
        shadow: inf_render::SkinnedShadow::BindSphere,
        sections: Vec::new(),
    });
    swapped.mark_dirty();
    renderer.render(&gpu, &swapped, &view, &target.view, (W, H));
    assert_eq!(
        renderer.gi_cache_counts(),
        (1, 1),
        "re-imported content replaces its predecessor rather than accreting beside it"
    );

    // And the content leaving empties both.
    let empty = RenderScene {
        grid_enabled: false,
        ..Default::default()
    };
    renderer.render(&gpu, &empty, &view, &target.view, (W, H));
    assert_eq!(
        renderer.gi_cache_counts(),
        (0, 0),
        "a cache must not outlive every last piece of the content it caches"
    );
}

/// GI's own off-transition: the node returns before it stages anything, so the
/// caches — which hold whole skinned meshes — are released in that branch.
#[test]
fn gi_releases_its_caches_when_the_setting_goes_off() {
    let Some(gpu) = gpu_or_skip("gi_releases_its_caches_when_the_setting_goes_off") else {
        return;
    };
    let target = HeadlessTarget::new(&gpu, W, H);
    let mut renderer = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
    renderer.set_settings(gi_settings(true));
    let view = view();
    let mesh = Arc::new(dense_grid_mesh(24));

    let mut scene = vgeom_scene(&mesh, true);
    scene.skinned_meshes.push(skinned_mesh());
    scene.skinned.push(SkinnedInstance {
        blend: 0,
        cutoff: 0.5,
        vt: Default::default(),
        translation: DVec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
        color: [0.7, 0.6, 0.5, 1.0],
        metallic: 0.0,
        roughness: 0.6,
        emissive: [0.0; 3],
        id: 2,
        mesh: 0,
        palette: inf_render::identity_palette(),
        shadow: inf_render::SkinnedShadow::BindSphere,
        sections: Vec::new(),
    });
    scene.mark_dirty();
    renderer.render(&gpu, &scene, &view, &target.view, (W, H));
    assert_eq!(renderer.gi_cache_counts(), (1, 1));

    renderer.set_settings(gi_settings(false));
    renderer.render(&gpu, &scene, &view, &target.view, (W, H));
    assert_eq!(
        renderer.gi_cache_counts(),
        (0, 0),
        "GI off means nothing reads either cache"
    );
}

/// A 4×4 solid-colour RGBA8 upload under `handle`.
fn solid_upload(handle: u64, rgba: [u8; 4]) -> inf_render::SpriteTextureUpload {
    inf_render::SpriteTextureUpload {
        handle,
        width: 4,
        height: 4,
        rgba8: rgba.iter().copied().cycle().take(4 * 4 * 4).collect(),
    }
}

/// A scene with one full-screen-ish sprite sampling `handle`, plus the upload.
fn sprite_scene(handle: u64, rgba: [u8; 4], with_sprite: bool) -> RenderScene {
    let mut scene = RenderScene {
        grid_enabled: false,
        pending_texture_uploads: vec![solid_upload(handle, rgba)],
        ..Default::default()
    };
    if with_sprite {
        scene.sprites.push(inf_render::SpriteInstance {
            position: DVec3::ZERO,
            size: glam::Vec2::new(6.0, 6.0),
            color: [1.0, 1.0, 1.0, 1.0],
            texture: handle,
            ..Default::default()
        });
    }
    scene.mark_dirty();
    scene
}

/// **The sprite finding (lens 2 M2), both halves.** The cache was keyed on a
/// GUID-derived handle with no content in the identity, and `ingest` skipped on
/// `contains_key` — so re-importing a sprite texture in place never reached the
/// screen again for the rest of the session. And it was eviction-free, so every
/// texture a session ever drew stayed in VRAM across every level switch.
///
/// The first half is asserted **in pixels**, because that is where the defect
/// lives: same handle, different bytes, must render differently.
#[test]
fn a_re_imported_sprite_texture_reaches_the_screen() {
    let Some(gpu) = gpu_or_skip("a_re_imported_sprite_texture_reaches_the_screen") else {
        return;
    };
    let target = HeadlessTarget::new(&gpu, W, H);
    let mut renderer = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
    let view = RenderView {
        origin: FloatingOrigin::new(DVec3::ZERO),
        eye_world: DVec3::new(0.0, 0.0, 6.0),
        forward: Vec3::new(0.0, 0.0, -1.0),
        up: Vec3::Y,
        fov_y: 60f32.to_radians(),
        near: 0.05,
        width: W,
        height: H,
        ortho: None,
    };
    const HANDLE: u64 = 0x5F31_7E00_0001;

    let red = sprite_scene(HANDLE, [220, 20, 20, 255], true);
    renderer.render(&gpu, &red, &view, &target.view, (W, H));
    let red_px = target.read_rgba(&gpu).expect("readback");

    // The SAME handle, different bytes — the in-place re-import.
    let mut blue = sprite_scene(HANDLE, [20, 20, 220, 255], true);
    blue.mark_dirty();
    renderer.render(&gpu, &blue, &view, &target.view, (W, H));
    let blue_px = target.read_rgba(&gpu).expect("readback");

    assert_ne!(
        red_px, blue_px,
        "a texture re-imported under its existing GUID must update on screen"
    );
}

/// The eviction half: a texture the level stopped referencing is released, and
/// the always-resident built-in font is what remains.
#[test]
fn sprite_textures_do_not_outlive_the_level_that_referenced_them() {
    let Some(gpu) = gpu_or_skip("sprite_textures_do_not_outlive_the_level_that_referenced_them")
    else {
        return;
    };
    let target = HeadlessTarget::new(&gpu, W, H);
    let mut renderer = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
    let view = view();

    // The built-in font is uploaded at construction and is always live.
    let floor = renderer.sprite_cache_len();

    let mut scene = RenderScene {
        grid_enabled: false,
        pending_texture_uploads: (0..4)
            .map(|i| solid_upload(0x7000 + i, [10 * i as u8, 40, 90, 255]))
            .collect(),
        ..Default::default()
    };
    for i in 0..4u64 {
        scene.sprites.push(inf_render::SpriteInstance {
            position: DVec3::new(i as f64, 0.0, 0.0),
            size: glam::Vec2::splat(1.0),
            color: [1.0, 1.0, 1.0, 1.0],
            texture: 0x7000 + i,
            ..Default::default()
        });
    }
    scene.mark_dirty();
    renderer.render(&gpu, &scene, &view, &target.view, (W, H));
    assert_eq!(
        renderer.sprite_cache_len(),
        floor + 4,
        "four referenced textures should be cached"
    );

    // The level switches: nothing references them any more.
    let empty = RenderScene {
        grid_enabled: false,
        ..Default::default()
    };
    renderer.render(&gpu, &empty, &view, &target.view, (W, H));
    assert_eq!(
        renderer.sprite_cache_len(),
        floor,
        "a level switch releases the textures the old level referenced"
    );
}
