//! **P18.3 end-to-end gate**: an imported glTF, placed in a scene, projects as
//! real geometry in an editor-shaped projection.
//!
//! The unit tests in `assets::vmesh` and `render_assets` pin the derivation and
//! the store in isolation. This one runs the whole chain the user actually
//! travels — *import a file → the DAG is derived → the entity resolves it → the
//! projection carries drawable geometry* — because every link in it is owned by a
//! different module and the interesting failures live in the joins.
//!
//! # Why the projection is re-stated here instead of called
//!
//! The real projection is `inf_viewport::host::rebuild_scene`, which is
//! `#[cfg(any(windows, target_os = "macos"))]` and needs a GPU-backed
//! `EngineHost`. Neither is available to Linux CI. What this file drives is the
//! *substance* of that branch — the Ring-1 resolution and the `RenderScene` DTO it
//! fills — while `tests/projector_mirror.rs` separately pins, field for field,
//! that the host's branch and the shipped player's are the same projection. The
//! two together are the gate: this proves the pipeline produces drawable geometry,
//! that proves both hosts draw it the same way.

use std::path::Path;

use inf_asset::{AssetId, AssetKind};
use inf_ecs::components::{GlobalTransform, Material, MeshRef, Primitive, SkeletalMesh};
use inf_editor_core::assets::AssetProject;
use inf_editor_core::render_assets::EditorRenderAssets;
use inf_render::{MeshInstance, PrimMesh, RenderScene, VgeomAsset, VgeomInstance};
use uuid::Uuid;

/// A glTF cube with an external buffer — 12 triangles, small on purpose: the
/// *cook* would decline to virtualize it (`min_triangles = 2048`) and the editor
/// must not, or the mesh a user actually imports keeps drawing a placeholder.
fn write_cube_gltf(dir: &Path) -> std::path::PathBuf {
    const P: [[f32; 3]; 8] = [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ];
    const TRIS: [u16; 36] = [
        0, 1, 2, 0, 2, 3, // -Z
        4, 6, 5, 4, 7, 6, // +Z
        0, 4, 5, 0, 5, 1, // -Y
        3, 2, 6, 3, 6, 7, // +Y
        0, 3, 7, 0, 7, 4, // -X
        1, 5, 6, 1, 6, 2, // +X
    ];
    let mut buf = Vec::new();
    for p in P {
        for f in p {
            buf.extend_from_slice(&f.to_le_bytes());
        }
    }
    let idx_off = buf.len();
    for i in TRIS {
        buf.extend_from_slice(&i.to_le_bytes());
    }
    std::fs::write(dir.join("cube.bin"), &buf).unwrap();
    let gltf = format!(
        r#"{{"asset":{{"version":"2.0"}},"scene":0,"scenes":[{{"nodes":[0]}}],"nodes":[{{"mesh":0}}],"meshes":[{{"name":"Cube","primitives":[{{"attributes":{{"POSITION":0}},"indices":1,"material":0}}]}}],"materials":[{{"name":"Paint","pbrMetallicRoughness":{{"baseColorFactor":[0.2,0.4,0.8,1],"metallicFactor":0.25,"roughnessFactor":0.6}}}}],"buffers":[{{"uri":"cube.bin","byteLength":{total}}}],"bufferViews":[{{"buffer":0,"byteOffset":0,"byteLength":{pl},"target":34962}},{{"buffer":0,"byteOffset":{io},"byteLength":72,"target":34963}}],"accessors":[{{"bufferView":0,"componentType":5126,"count":8,"type":"VEC3","min":[-1,-1,-1],"max":[1,1,1]}},{{"bufferView":1,"componentType":5123,"count":36,"type":"SCALAR"}}]}}"#,
        total = buf.len(),
        pl = idx_off,
        io = idx_off,
    );
    let path = dir.join("cube.gltf");
    std::fs::write(&path, gltf).unwrap();
    path
}

/// One entity's projection input — the fields the real `rebuild_scene` reads off
/// the world before it branches.
struct Placed {
    mesh_ref: MeshRef,
    material: Material,
    transform: GlobalTransform,
}

/// The **editor-shaped** `MeshRef` projection: the substance of
/// `inf_viewport::host::rebuild_scene`'s branch, in a form Linux CI can run.
/// Field-for-field agreement with the shipped player's copy is gated separately
/// by `tests/projector_mirror.rs`.
fn project(scene: &mut RenderScene, store: &mut EditorRenderAssets, placed: &[Placed]) {
    scene.instances.clear();
    scene.vgeom_assets.clear();
    scene.vgeom_instances.clear();
    let mut seen: std::collections::BTreeSet<u128> = std::collections::BTreeSet::new();
    // Ids start at 1 (`ID_NONE` is 0) and advance per projected entity, exactly as
    // the host's `next_id` does.
    for (id, p) in (1u32..).zip(placed) {
        let (scale, rot, translation) = p.transform.0.to_scale_rotation_translation();
        let e = p.material.emissive.to_array();
        let (color, metallic, roughness, emissive) = (
            p.material.base_color.to_array(),
            p.material.metallic,
            p.material.roughness,
            [e[0], e[1], e[2]],
        );
        match p.mesh_ref.asset.and_then(|m| store.resolve_vgeom(m)) {
            Some(loaded) => {
                if seen.insert(loaded.id) {
                    scene
                        .vgeom_assets
                        .push(VgeomAsset::new(loaded.id, loaded.source));
                }
                scene.vgeom_instances.push(VgeomInstance {
                    vt: Default::default(),
                    asset: loaded.id,
                    translation,
                    rotation: rot.as_quat(),
                    scale: scale.as_vec3(),
                    color,
                    metallic,
                    roughness,
                    emissive,
                    id,
                });
            }
            None => scene.instances.push(MeshInstance {
                vt: Default::default(),
                translation,
                rotation: rot.as_quat(),
                scale: scale.as_vec3(),
                color,
                metallic,
                roughness,
                emissive,
                id,
                mesh: match p.mesh_ref.primitive {
                    Primitive::Cube => PrimMesh::Cube,
                    Primitive::Sphere => PrimMesh::Sphere,
                    Primitive::Plane => PrimMesh::Plane,
                    Primitive::Cylinder => PrimMesh::Cylinder,
                    Primitive::Cone => PrimMesh::Cone,
                },
                blend: 0,
                cutoff: 0.5,
            }),
        }
    }
    scene.mark_dirty();
}

fn at(x: f64) -> GlobalTransform {
    GlobalTransform(glam::DAffine3::from_translation(glam::DVec3::new(
        x, 0.0, 0.0,
    )))
}

/// Import the fixture and return `(project dir, content root, mesh asset id)`.
fn import_cube() -> (tempfile::TempDir, tempfile::TempDir, AssetId) {
    let src = tempfile::tempdir().unwrap();
    let root = tempfile::tempdir().unwrap();
    let gltf = write_cube_gltf(src.path());
    let mut proj = AssetProject::open(root.path()).unwrap();
    let dest = proj.content_dir("Imported").unwrap();
    let out = proj.import_file(&gltf, &dest).unwrap();
    let mesh = out.primary.unwrap();
    assert_eq!(proj.db().get(mesh).unwrap().kind(), AssetKind::Mesh);
    (src, root, mesh)
}

/// **The headline.** An imported glTF placed in a scene projects as virtualized
/// geometry rather than a placeholder — the gap open since P4, closed.
#[test]
fn an_imported_gltf_projects_as_real_geometry() {
    let (_src, root, mesh) = import_cube();

    // The import derived the DAG (small mesh included — the editor's threshold is
    // one triangle, not the cook's 2048).
    let derived = inf_vgeom::derived_vmesh_id(mesh);
    {
        let proj = AssetProject::open(root.path()).unwrap();
        let e = proj
            .db()
            .get(derived)
            .expect("the import derived a .inf_vmesh");
        assert_eq!(e.kind(), AssetKind::MeshletMesh);
    }

    let mut store = EditorRenderAssets::new();
    store.set_content_root(Some(root.path().to_path_buf()));
    let mut scene = RenderScene::default();
    project(
        &mut scene,
        &mut store,
        &[Placed {
            mesh_ref: MeshRef {
                primitive: Primitive::Cube,
                asset: Some(mesh.uuid()),
            },
            material: Material {
                metallic: 0.25,
                roughness: 0.6,
                ..Material::default()
            },
            transform: at(4.0),
        }],
    );

    assert_eq!(scene.vgeom_assets.len(), 1, "one asset listed");
    assert_eq!(scene.vgeom_instances.len(), 1, "one instance placed");
    assert!(
        scene.instances.is_empty(),
        "no placeholder cube — that is the whole point of P18.3"
    );

    // The listed asset is drawable: real meshlets, a real page directory, a real
    // bounding sphere (the LOD projection reads it every frame).
    let a = &scene.vgeom_assets[0];
    assert!(a.source.meshlet_count() > 0);
    assert!(!a.source.pages().is_empty());
    assert!(a.bounds().1 > 0.0);

    // …and the instance carries the entity's placement and PBR params (deliverable
    // 4: materials on real meshes, to the extent the player already does).
    let i = &scene.vgeom_instances[0];
    assert_eq!(i.asset, a.id);
    assert!((i.translation.x - 4.0).abs() < 1e-9);
    assert_eq!(i.scale, glam::Vec3::ONE);
    assert!((i.metallic - 0.25).abs() < 1e-6);
    assert!((i.roughness - 0.6).abs() < 1e-6);
    assert_eq!(i.color, Material::default().base_color.to_array());
}

/// Determinism: two independent runs of the whole chain — import, derive,
/// resolve, project — produce the same asset key and the same instance stream.
#[test]
fn the_whole_chain_is_deterministic() {
    let fingerprint = || {
        let (_src, root, mesh) = import_cube();
        let mut store = EditorRenderAssets::new();
        store.set_content_root(Some(root.path().to_path_buf()));
        let mut scene = RenderScene::default();
        project(
            &mut scene,
            &mut store,
            &(0..3)
                .map(|k| Placed {
                    mesh_ref: MeshRef {
                        primitive: Primitive::Cube,
                        asset: Some(mesh.uuid()),
                    },
                    material: Material::default(),
                    transform: at(k as f64),
                })
                .collect::<Vec<_>>(),
        );
        // The asset ids + the per-instance stream. Deliberately NOT the residency
        // generation or anything else drawn from a process-global counter.
        let assets: Vec<(u128, u32, usize)> = scene
            .vgeom_assets
            .iter()
            .map(|a| (a.id, a.source.meshlet_count(), a.source.pages().len()))
            .collect();
        let insts: Vec<(u128, [f64; 3], u32)> = scene
            .vgeom_instances
            .iter()
            .map(|i| (i.asset, i.translation.to_array(), i.id))
            .collect();
        // Keep `root` alive until after the projection reads it.
        drop(store);
        drop(root);
        (assets, insts)
    };
    let a = fingerprint();
    let b = fingerprint();
    assert_eq!(
        a, b,
        "the editor's real-mesh projection must be reproducible"
    );
    assert_eq!(a.0.len(), 1, "three instances share one asset entry");
    assert_eq!(a.1.len(), 3);
}

/// Deliverable 5, end to end: deleting the mesh asset degrades the entity to its
/// primitive placeholder — no panic, no stale geometry.
#[test]
fn deleting_the_mesh_degrades_the_entity_to_its_primitive() {
    let (_src, root, mesh) = import_cube();
    let mut store = EditorRenderAssets::new();
    store.set_content_root(Some(root.path().to_path_buf()));
    let placed = [Placed {
        mesh_ref: MeshRef {
            primitive: Primitive::Sphere,
            asset: Some(mesh.uuid()),
        },
        material: Material::default(),
        transform: at(0.0),
    }];

    let mut scene = RenderScene::default();
    project(&mut scene, &mut store, &placed);
    assert_eq!(scene.vgeom_instances.len(), 1);

    {
        let mut proj = AssetProject::open(root.path()).unwrap();
        // Force: the imported material references nothing here, but the guard is
        // about referrers, and this is the "user deleted it anyway" path.
        proj.delete(mesh, true).unwrap();
    }
    store.refresh_index();

    project(&mut scene, &mut store, &placed);
    assert!(scene.vgeom_instances.is_empty(), "no stale geometry");
    assert!(scene.vgeom_assets.is_empty());
    assert_eq!(scene.instances.len(), 1, "degraded to a placeholder");
    assert!(
        matches!(scene.instances[0].mesh, PrimMesh::Sphere),
        "…and to the MeshRef's OWN primitive kind, not always a cube"
    );
}

/// A primitive-only `MeshRef` is untouched by any of this: it is legitimate
/// authored content, not a placeholder, and it must keep drawing through the
/// rigid mesh path exactly as it did before P18.3. (This is what keeps all 36
/// goldens byte-identical — every golden scene is primitives.)
#[test]
fn primitive_mesh_refs_are_unaffected() {
    let root = tempfile::tempdir().unwrap();
    let mut store = EditorRenderAssets::new();
    store.set_content_root(Some(root.path().to_path_buf()));
    let mut scene = RenderScene::default();
    project(
        &mut scene,
        &mut store,
        &[
            Placed {
                mesh_ref: MeshRef {
                    primitive: Primitive::Cone,
                    asset: None,
                },
                material: Material::default(),
                transform: at(0.0),
            },
            // A dangling asset reference degrades the same way.
            Placed {
                mesh_ref: MeshRef {
                    primitive: Primitive::Cylinder,
                    asset: Some(Uuid::from_u128(0xDEAD)),
                },
                material: Material::default(),
                transform: at(1.0),
            },
        ],
    );
    assert!(scene.vgeom_instances.is_empty());
    assert!(scene.vgeom_assets.is_empty());
    assert_eq!(scene.instances.len(), 2);
    assert!(matches!(scene.instances[0].mesh, PrimMesh::Cone));
    assert!(matches!(scene.instances[1].mesh, PrimMesh::Cylinder));
}

/// A `SkeletalMesh` whose assets are not (yet) bound keeps its placeholder — the
/// authoring affordance P11.1 introduced must survive P18.3.
#[test]
fn an_unbound_skeletal_mesh_still_has_a_placeholder_path() {
    let root = tempfile::tempdir().unwrap();
    let mut store = EditorRenderAssets::new();
    store.set_content_root(Some(root.path().to_path_buf()));
    assert!(store
        .resolve_skinned(&SkeletalMesh::default(), None, None, None)
        .is_none());
}
