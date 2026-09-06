//! glTF/GLB import → geometry + material/texture descriptors.
//!
//! This reads a glTF document and produces a Ring-0-friendly intermediate:
//!   * one [`MeshAsset`] per glTF mesh (primitives → submeshes, optimized);
//!   * [`ImportedMaterial`] descriptors (PBR factors + which image each map is);
//!   * [`RawImage`]s decoded to RGBA8 (glTF decodes via the `image` crate).
//!
//! The editor's import orchestrator turns this into real `.inf_mesh` /
//! `.inf_tex` / `.inf_mat` assets with GUIDs and dependency edges. Missing
//! normals are generated (flat per-face); missing tangents default (the material
//! pass can regenerate from UVs later).

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use glam::{Mat4, Vec3};
use inf_anim::{
    AnimClip, Interpolation, Joint, JointTrack, JointTransform, QuatTrack, Skeleton, Vec3Track,
};

use crate::asset::{MeshAsset, MeshVertex, SubMesh, VertexSkin};
use crate::error::MeshError;
use crate::optimize::optimize;
use crate::validate::{
    reject_joint_indices, reject_length_mismatch, reject_non_finite, reject_non_increasing,
    reject_out_of_range, reject_partial_triangle,
};

/// A decoded texture source, RGBA8, straight from the glTF image list.
#[derive(Debug, Clone, PartialEq)]
pub struct RawImage {
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub rgba8: Vec<u8>,
}

/// A glTF material's PBR parameters + the image indices of its maps.
#[derive(Debug, Clone, PartialEq)]
pub struct ImportedMaterial {
    pub name: String,
    pub base_color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub emissive: [f32; 3],
    /// Indices into [`GltfImport::images`], if the map is present.
    pub base_color_image: Option<usize>,
    pub normal_image: Option<usize>,
    pub metallic_roughness_image: Option<usize>,
}

/// One imported mesh with its display name.
#[derive(Debug, Clone, PartialEq)]
pub struct ImportedMesh {
    pub name: String,
    pub mesh: MeshAsset,
    /// Index into [`GltfImport::skeletons`] this mesh is skinned to, if any
    /// (resolved from the glTF node that pairs this mesh with a skin).
    pub skin: Option<usize>,
}

/// One imported skinning skeleton (from a glTF `skin`).
#[derive(Debug, Clone, PartialEq)]
pub struct ImportedSkeleton {
    pub name: String,
    pub skeleton: Skeleton,
}

/// One imported animation clip (from a glTF `animation`).
#[derive(Debug, Clone, PartialEq)]
pub struct ImportedClip {
    pub name: String,
    pub clip: AnimClip,
    /// Index into [`GltfImport::skeletons`] this clip's tracks target, if the
    /// channels resolved to a skin's joints.
    pub skeleton: Option<usize>,
}

/// The full result of importing a glTF file.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct GltfImport {
    pub meshes: Vec<ImportedMesh>,
    pub materials: Vec<ImportedMaterial>,
    pub images: Vec<RawImage>,
    /// Skinning skeletons (P11.1), one per glTF `skin`, in `skin.index()` order.
    pub skeletons: Vec<ImportedSkeleton>,
    /// Skeletal animation clips (P11.1), one per glTF `animation`.
    pub clips: Vec<ImportedClip>,
}

/// Import a glTF/GLB file.
pub fn import_gltf(path: &Path) -> Result<GltfImport, MeshError> {
    let (doc, buffers, images) =
        gltf::import(path).map_err(|e| MeshError::Gltf(format!("{path:?}: {e}")))?;

    let material_names: Vec<String> = doc
        .materials()
        .map(|m| m.name().unwrap_or("Material").to_string())
        .collect();

    let mut out = GltfImport::default();

    // Skins → skeletons (P11.1). `node_to_joint[skin_index]` maps a glTF node
    // index to its joint index within that skeleton (used to bind clip channels).
    let mut node_to_joint: Vec<HashMap<usize, u16>> = Vec::new();
    {
        // Child → parent map (glTF nodes have no back-pointer to their parent).
        let mut parent_of: HashMap<usize, usize> = HashMap::new();
        for node in doc.nodes() {
            for child in node.children() {
                parent_of.insert(child.index(), node.index());
            }
        }
        for skin in doc.skins() {
            let joint_nodes: Vec<gltf::Node> = skin.joints().collect();
            let map: HashMap<usize, u16> = joint_nodes
                .iter()
                .enumerate()
                .map(|(i, n)| (n.index(), i as u16))
                .collect();
            let reader = skin.reader(|b| Some(&buffers[b.index()]));
            let ibms: Vec<[[f32; 4]; 4]> = reader
                .read_inverse_bind_matrices()
                .map(|it| it.collect())
                .unwrap_or_default();
            // The import door (P-hardening B / C4-8): an inverse-bind matrix is
            // raw IEEE-754 out of the binary buffer and rides into `.inf_skel`
            // and from there into every skinning matrix.
            reject_non_finite(&ibms, "inverseBindMatrices")?;
            let mut joints = Vec::with_capacity(joint_nodes.len());
            for (i, node) in joint_nodes.iter().enumerate() {
                let parent = parent_of
                    .get(&node.index())
                    .and_then(|p| map.get(p))
                    .copied();
                let (t, r, s) = node.transform().decomposed();
                reject_non_finite(&[t, s], "node translation/scale")?;
                reject_non_finite(&[r], "node rotation")?;
                let inverse_bind = ibms
                    .get(i)
                    .map(|m| Mat4::from_cols_array_2d(m).to_cols_array())
                    .unwrap_or_else(|| Mat4::IDENTITY.to_cols_array());
                joints.push(Joint {
                    name: node.name().unwrap_or("joint").to_string(),
                    parent,
                    inverse_bind,
                    local_bind: JointTransform {
                        translation: t,
                        rotation: r,
                        scale: s,
                    },
                });
            }
            // v1: trust the glTF joint order (parents-before-children in every
            // exporter we target). A non-topological skin errors here; a stable
            // topological re-sort + JOINTS_0/channel remap is a documented P11.x
            // follow-up.
            let skeleton = Skeleton::new(joints)
                .map_err(|e| MeshError::Gltf(format!("skin {}: {e}", skin.index())))?;
            out.skeletons.push(ImportedSkeleton {
                name: skin.name().unwrap_or("Skeleton").to_string(),
                skeleton,
            });
            node_to_joint.push(map);
        }
    }

    // Node → skin index, so a mesh drawn by a skinned node binds to its skeleton.
    let mut mesh_to_skin: HashMap<usize, usize> = HashMap::new();
    for node in doc.nodes() {
        if let (Some(mesh), Some(skin)) = (node.mesh(), node.skin()) {
            mesh_to_skin.insert(mesh.index(), skin.index());
        }
    }

    // Animations → clips (P11.1).
    for anim in doc.animations() {
        let mut tracks_by_joint: BTreeMap<u16, JointTrack> = BTreeMap::new();
        let mut skel_idx: Option<usize> = None;
        for channel in anim.channels() {
            let target = channel.target().node().index();
            let Some((si, joint)) = find_joint(&node_to_joint, target) else {
                continue; // a channel targeting a non-joint node (v1 skips it)
            };
            // v1: a clip binds to a single skeleton (the first it touches).
            if *skel_idx.get_or_insert(si) != si {
                continue;
            }
            let sampler = channel.sampler();
            let cubic = matches!(
                sampler.interpolation(),
                gltf::animation::Interpolation::CubicSpline
            );
            let interp = match sampler.interpolation() {
                gltf::animation::Interpolation::Step => Interpolation::Step,
                // LINEAR and (cubic-resampled-to-linear) CUBICSPLINE both land here.
                _ => Interpolation::Linear,
            };
            let reader = channel.reader(|b| Some(&buffers[b.index()]));
            let times: Vec<f32> = match reader.read_inputs() {
                Some(it) => it.collect(),
                None => continue,
            };
            // The import door (C4-7). `inf_anim::clip::locate` binary-searches
            // these, so a NaN or an unsorted list is not a cosmetic defect: it
            // is a wrong bracket at best and, before the guard beside it, an
            // index underflow in the shipped player.
            let what = format!("animation {:?} sampler input", anim.name().unwrap_or(""));
            reject_non_increasing(&times, &what)?;
            let entry = tracks_by_joint
                .entry(joint)
                .or_insert_with(|| JointTrack::new(joint));
            match reader.read_outputs() {
                Some(gltf::animation::util::ReadOutputs::Translations(it)) => {
                    let vals = resample_cubic(it.collect(), cubic);
                    check_track(&times, &vals, "translation", &what)?;
                    entry.translation = Some(Vec3Track::new(times, vals, interp));
                }
                Some(gltf::animation::util::ReadOutputs::Scales(it)) => {
                    let vals = resample_cubic(it.collect(), cubic);
                    check_track(&times, &vals, "scale", &what)?;
                    entry.scale = Some(Vec3Track::new(times, vals, interp));
                }
                Some(gltf::animation::util::ReadOutputs::Rotations(rot)) => {
                    let vals = resample_cubic(rot.into_f32().collect(), cubic);
                    check_track(&times, &vals, "rotation", &what)?;
                    entry.rotation = Some(QuatTrack::new(times, vals, interp));
                }
                _ => {}
            }
        }
        let tracks: Vec<JointTrack> = tracks_by_joint.into_values().collect();
        out.clips.push(ImportedClip {
            name: anim.name().unwrap_or("Animation").to_string(),
            clip: AnimClip::new(anim.name().unwrap_or("Animation").to_string(), tracks),
            skeleton: skel_idx,
        });
    }

    // Images → RGBA8.
    for (i, img) in images.iter().enumerate() {
        out.images.push(RawImage {
            name: format!("image_{i}"),
            width: img.width,
            height: img.height,
            rgba8: to_rgba8(img),
        });
    }

    // Materials.
    for mat in doc.materials() {
        let pbr = mat.pbr_metallic_roughness();
        out.materials.push(ImportedMaterial {
            name: mat.name().unwrap_or("Material").to_string(),
            base_color: pbr.base_color_factor(),
            metallic: pbr.metallic_factor(),
            roughness: pbr.roughness_factor(),
            emissive: mat.emissive_factor(),
            base_color_image: pbr
                .base_color_texture()
                .map(|t| t.texture().source().index()),
            normal_image: mat.normal_texture().map(|t| t.texture().source().index()),
            metallic_roughness_image: pbr
                .metallic_roughness_texture()
                .map(|t| t.texture().source().index()),
        });
    }

    // Meshes → submeshes.
    for mesh in doc.meshes() {
        let name = mesh.name().unwrap_or("Mesh").to_string();
        // How many joints this mesh's skin actually has, so `JOINTS_0` can be
        // bounded against it (C4-20). Zero when no skin resolved, in which case
        // the influences are dropped rather than bound.
        let skin_joints = mesh_to_skin
            .get(&mesh.index())
            .and_then(|si| out.skeletons.get(*si))
            .map(|s| s.skeleton.len())
            .unwrap_or(0);
        let mut submeshes = Vec::new();
        for (pi, prim) in mesh.primitives().enumerate() {
            if prim.mode() != gltf::mesh::Mode::Triangles {
                // Only triangle lists are supported; skip lines/points.
                continue;
            }
            let reader = prim.reader(|b| Some(&buffers[b.index()]));
            let positions: Vec<[f32; 3]> = match reader.read_positions() {
                Some(p) => p.collect(),
                None => continue, // a primitive with no positions is unusable
            };
            if positions.is_empty() {
                // A zero-count POSITION accessor. Skipping it here is what stops
                // `SubMesh { vertices: [], indices: [..] }` from being persisted
                // as a valid-looking `.inf_mesh` (C4-9).
                continue;
            }
            let indices: Vec<u32> = match reader.read_indices() {
                Some(idx) => idx.into_u32().collect(),
                None => (0..positions.len() as u32).collect(),
            };
            let normals: Option<Vec<[f32; 3]>> = reader.read_normals().map(|n| n.collect());
            let uvs: Option<Vec<[f32; 2]>> = reader
                .read_tex_coords(0)
                .map(|t| t.into_f32().collect::<Vec<_>>());
            let tangents: Option<Vec<[f32; 4]>> = reader.read_tangents().map(|t| t.collect());

            // Skinning influences (P11.1): JOINTS_0 (widened to u16) + WEIGHTS_0
            // (normalized). Present together on a skinned primitive.
            let joints0: Option<Vec<[u16; 4]>> =
                reader.read_joints(0).map(|j| j.into_u16().collect());
            let weights0: Option<Vec<[f32; 4]>> =
                reader.read_weights(0).map(|w| w.into_f32().collect());
            // **The SECOND influence set** (wave CHAR1a). glTF allows any number
            // of `JOINTS_n`/`WEIGHTS_n` pairs and Unreal's exporter writes two —
            // EIGHT influences a vertex — for every skeletal mesh measured here
            // (SKM_Manny and SKM_Quinn, LOD 0 through 2).
            //
            // `.inf_mesh`'s `VertexSkin` holds four, and that is not changing in
            // this wave: widening it moves a `#[repr(C)] Pod` struct that is
            // uploaded verbatim to a GPU vertex buffer and stored in every
            // committed `.inf_mesh`, for a fourth and fifth bone whose weight on
            // this content is small. What DID have to change is what happens to
            // set 1. Reading set 0 alone keeps *the first four written*, which
            // the glTF specification does not require to be the heaviest four —
            // so a vertex whose dominant bone the exporter happened to write
            // fifth was skinned to the wrong bones and then renormalized to make
            // the error total 1.0, which hides it perfectly.
            //
            // So both sets are read and the four HEAVIEST are kept.
            //
            // **Measured on SKM_Manny LOD 0** (48 779 vertices across two
            // primitives, WEIGHTS as normalized u8):
            //
            // * **15 431 vertices (31.63%)** carry a non-zero influence in set
            //   1 — the eight-bone budget is really used, on about a third of
            //   the body.
            // * the weight mass those vertices lose to the four-influence cap is
            //   **mean 0.0596, median 0.0510, p95 0.1333, max 0.2824** (first
            //   primitive, 9 561 of 28 678 vertices). `normalized()` divides it
            //   back out, so the kept bones absorb up to 28% more motion than
            //   the author gave them at the worst vertex.
            // * **zero** of the 48 779 set-0 quadruples arrive out of order:
            //   Unreal writes its influences heaviest-first already. So the sort
            //   below rescues nothing *on this content* and is a guard rather
            //   than a fix — the glTF specification does not promise the order,
            //   and an exporter that does not sort would otherwise skin a
            //   vertex to its four LIGHTEST bones and renormalize the error to
            //   1.0, which looks perfectly healthy.
            let joints1: Option<Vec<[u16; 4]>> =
                reader.read_joints(1).map(|j| j.into_u16().collect());
            let weights1: Option<Vec<[f32; 4]>> =
                reader.read_weights(1).map(|w| w.into_f32().collect());

            // ── The import door ─────────────────────────────────────────────
            //
            // Everything below this point either hands the index buffer to
            // meshopt's raw FFI (C4-1), indexes a parallel stream by vertex
            // ordinal (C4-6), or writes the numbers straight into `.inf_mesh`
            // (C4-8). All three are checked here, once, and named for the
            // attribute the exporter wrote.
            let at = format!("{name} primitive {pi}");
            reject_non_finite(&positions, &format!("{at} POSITION"))?;
            reject_out_of_range(&indices, positions.len(), &format!("{at} indices"))?;
            // A count that is not a multiple of three: `meshopt` asserts against
            // it and the assert is compiled out, so the partial triangle becomes
            // an out-of-bounds heap write inside the vertex-cache optimizer four
            // lines of ours later. Named here, where the file and the primitive
            // can be named with it.
            reject_partial_triangle(indices.len(), &format!("{at} indices"))?;
            for (stream, label) in [
                (normals.as_ref().map(Vec::len), "NORMAL"),
                (uvs.as_ref().map(Vec::len), "TEXCOORD_0"),
                (tangents.as_ref().map(Vec::len), "TANGENT"),
                (joints0.as_ref().map(Vec::len), "JOINTS_0"),
                (weights0.as_ref().map(Vec::len), "WEIGHTS_0"),
            ] {
                if let Some(len) = stream {
                    reject_length_mismatch(
                        len,
                        positions.len(),
                        &format!("{at} {label}"),
                        "POSITION",
                    )?;
                }
            }
            if let Some(n) = normals.as_ref() {
                reject_non_finite(n, &format!("{at} NORMAL"))?;
            }
            if let Some(u) = uvs.as_ref() {
                reject_non_finite(u, &format!("{at} TEXCOORD_0"))?;
            }
            if let Some(t) = tangents.as_ref() {
                reject_non_finite(t, &format!("{at} TANGENT"))?;
            }
            if let Some(w) = weights0.as_ref() {
                // `VertexSkin::normalized` divides by the sum of these, so an
                // infinity here manufactures a NaN out of finite-looking input
                // (C4-37) and uploads it to a GPU vertex buffer.
                reject_non_finite(w, &format!("{at} WEIGHTS_0"))?;
            }
            if let Some(j) = joints0.as_ref() {
                reject_joint_indices(j, skin_joints, &format!("{at} JOINTS_0"))?;
            }
            if let Some(w) = weights1.as_ref() {
                reject_non_finite(w, &format!("{at} WEIGHTS_1"))?;
            }
            if let Some(j) = joints1.as_ref() {
                reject_joint_indices(j, skin_joints, &format!("{at} JOINTS_1"))?;
            }

            let normals = normals.unwrap_or_else(|| compute_normals(&positions, &indices));

            let mut verts = Vec::with_capacity(positions.len());
            for (i, &position) in positions.iter().enumerate() {
                verts.push(MeshVertex {
                    position,
                    normal: *normals.get(i).unwrap_or(&[0.0, 1.0, 0.0]),
                    uv: uvs
                        .as_ref()
                        .and_then(|u| u.get(i).copied())
                        .unwrap_or([0.0, 0.0]),
                    tangent: tangents
                        .as_ref()
                        .and_then(|t| t.get(i).copied())
                        .unwrap_or(crate::TANGENT_PLACEHOLDER),
                });
            }

            let skin: Vec<VertexSkin> = match (joints0, weights0) {
                (Some(j), Some(w)) => (0..positions.len())
                    .map(|i| {
                        let mut pairs: [(u16, f32); 8] = [(0, 0.0); 8];
                        let a = *j.get(i).unwrap_or(&[0; 4]);
                        let aw = *w.get(i).unwrap_or(&[1.0, 0.0, 0.0, 0.0]);
                        for k in 0..4 {
                            pairs[k] = (a[k], aw[k]);
                        }
                        if let (Some(j1), Some(w1)) = (joints1.as_ref(), weights1.as_ref()) {
                            let b = *j1.get(i).unwrap_or(&[0; 4]);
                            let bw = *w1.get(i).unwrap_or(&[0.0; 4]);
                            for k in 0..4 {
                                pairs[4 + k] = (b[k], bw[k]);
                            }
                        }
                        // Heaviest first. A STABLE sort on the descending
                        // weight, so two influences of equal weight keep the
                        // order the exporter wrote them in and the same glTF
                        // imports to the same bytes on every machine — the
                        // determinism law, on a path that lands in a pack.
                        pairs.sort_by(|x, y| {
                            y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal)
                        });
                        VertexSkin {
                            joints: [pairs[0].0, pairs[1].0, pairs[2].0, pairs[3].0],
                            weights: [pairs[0].1, pairs[1].1, pairs[2].1, pairs[3].1],
                        }
                        .normalized()
                    })
                    .collect(),
                _ => Vec::new(),
            };

            let material_slot = prim.material().index().map(|i| i as u32);
            let sub = if skin.is_empty() {
                // Rigid submesh: the meshopt weld/cache/fetch pass reorders verts.
                let (verts, idx) = optimize(verts, indices);
                SubMesh {
                    name: format!("{name}_{pi}"),
                    vertices: verts,
                    indices: idx,
                    material_slot,
                    skin: Vec::new(),
                }
            } else {
                // Skinned submesh: skip meshopt so the parallel `skin` stream stays
                // index-aligned to `vertices` (a skin-aware remap is a follow-up).
                SubMesh {
                    name: format!("{name}_{pi}"),
                    vertices: verts,
                    indices,
                    material_slot,
                    skin,
                }
            };
            submeshes.push(sub);
        }
        if submeshes.is_empty() {
            continue;
        }
        out.meshes.push(ImportedMesh {
            name: name.clone(),
            mesh: MeshAsset::new(submeshes, material_names.clone()),
            skin: mesh_to_skin.get(&mesh.index()).copied(),
        });
    }

    if out.meshes.is_empty() {
        return Err(MeshError::Gltf(format!("{path:?}: no triangle meshes")));
    }
    Ok(out)
}

/// Find the `(skeleton_index, joint_index)` a glTF node maps to, scanning each
/// skin's node→joint map.
fn find_joint(node_to_joint: &[HashMap<usize, u16>], node: usize) -> Option<(usize, u16)> {
    node_to_joint
        .iter()
        .enumerate()
        .find_map(|(si, map)| map.get(&node).map(|&j| (si, j)))
}

/// Check one resampled animation channel against the keyframe times it is
/// indexed by, and against finiteness.
///
/// [`Vec3Track::new`] and [`QuatTrack::new`] only `debug_assert` the length
/// agreement — compiled out in release — and `sample()` then indexes
/// `self.values[i0]` with an index derived from `times`. So a sampler whose
/// output count differs from its input count is a panic in the shipped player,
/// deferred until somebody plays the clip. It is also how a `CUBICSPLINE`
/// stream with a remainder shows up: [`resample_cubic`]'s `chunks_exact(3)`
/// drops it, and the shortfall lands here (C4-45).
fn check_track<T: crate::validate::AllFinite>(
    times: &[f32],
    values: &[T],
    channel: &str,
    sampler: &str,
) -> Result<(), MeshError> {
    let what = format!("{sampler} {channel} output");
    reject_length_mismatch(values.len(), times.len(), &what, "its sampler input")?;
    reject_non_finite(values, &what)
}

/// Resample a glTF animation output stream to one value per keyframe.
///
/// A `CUBICSPLINE` sampler stores three values per key — `[in_tangent, value,
/// out_tangent]` — so its output length is `3 · keys`; v1 keeps the middle
/// **value** of each triple and drops the tangents (the curve is treated as
/// linear between the preserved sample points). A `STEP`/`LINEAR` stream is one
/// value per key and passes through unchanged.
///
/// A stream whose length is **not** a multiple of three loses its remainder
/// here, silently — which is exactly why [`check_track`] compares the result
/// against the input count rather than trusting it.
fn resample_cubic<T: Copy>(values: Vec<T>, cubic: bool) -> Vec<T> {
    if cubic {
        values.chunks_exact(3).map(|c| c[1]).collect()
    } else {
        values
    }
}

/// Flat per-face normals for geometry that ships without them.
fn compute_normals(positions: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    let mut acc = vec![Vec3::ZERO; positions.len()];
    for tri in indices.chunks_exact(3) {
        let (a, b, c) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        let pa = Vec3::from(positions[a]);
        let pb = Vec3::from(positions[b]);
        let pc = Vec3::from(positions[c]);
        let n = (pb - pa).cross(pc - pa);
        acc[a] += n;
        acc[b] += n;
        acc[c] += n;
    }
    acc.into_iter()
        .map(|v| v.normalize_or_zero().to_array())
        .collect()
}

/// Expand a glTF image (any channel layout / bit depth we recognize) to RGBA8.
fn to_rgba8(img: &gltf::image::Data) -> Vec<u8> {
    use gltf::image::Format;
    // **usize, not u32** (C4-16). `width * height` in u32 wraps to 0 at 2³⁰
    // pixels and the release profile has no `overflow-checks`, so a hostile
    // 65 536² image used to hand back an EMPTY buffer while `RawImage.width` /
    // `.height` went on claiming the full size — and the downstream guard that
    // was supposed to notice (`rgba.len() < (width * height * 4) as usize`)
    // wrapped identically, so `0 < 0` was false and the truncation sailed
    // through. Two wraps that cancelled into silence.
    let n = img.width as usize * img.height as usize;
    let px = &img.pixels;
    let mut out = Vec::with_capacity(n * 4);
    match img.format {
        Format::R8 => {
            for &r in px.iter().take(n) {
                out.extend_from_slice(&[r, r, r, 255]);
            }
        }
        Format::R8G8 => {
            for c in px.chunks_exact(2).take(n) {
                out.extend_from_slice(&[c[0], c[0], c[0], c[1]]);
            }
        }
        Format::R8G8B8 => {
            for c in px.chunks_exact(3).take(n) {
                out.extend_from_slice(&[c[0], c[1], c[2], 255]);
            }
        }
        Format::R8G8B8A8 => {
            out.extend_from_slice(&px[..(n * 4).min(px.len())]);
        }
        Format::R16 | Format::R16G16 | Format::R16G16B16 | Format::R16G16B16A16 => {
            // 16-bit: take the high byte of each channel, expand to RGBA8.
            let channels = match img.format {
                Format::R16 => 1,
                Format::R16G16 => 2,
                Format::R16G16B16 => 3,
                _ => 4,
            };
            for texel in px.chunks_exact(channels * 2).take(n) {
                let mut rgba = [0u8, 0, 0, 255];
                for ch in 0..channels {
                    rgba[ch.min(3)] = texel[ch * 2 + 1]; // high byte (LE)
                }
                if channels == 1 {
                    rgba = [rgba[0], rgba[0], rgba[0], 255];
                }
                out.extend_from_slice(&rgba);
            }
        }
        Format::R32G32B32FLOAT | Format::R32G32B32A32FLOAT => {
            let channels = if matches!(img.format, Format::R32G32B32FLOAT) {
                3
            } else {
                4
            };
            for texel in px.chunks_exact(channels * 4).take(n) {
                let mut rgba = [0u8, 0, 0, 255];
                for ch in 0..channels {
                    let bytes = [
                        texel[ch * 4],
                        texel[ch * 4 + 1],
                        texel[ch * 4 + 2],
                        texel[ch * 4 + 3],
                    ];
                    let f = f32::from_le_bytes(bytes).clamp(0.0, 1.0);
                    rgba[ch.min(3)] = (f * 255.0) as u8;
                }
                out.extend_from_slice(&rgba);
            }
        }
    }
    // Pad if a truncated source left us short.
    out.resize(n * 4, 0);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Write a minimal single-triangle glTF (external .bin buffer) and import it.
    #[test]
    fn imports_a_minimal_triangle_gltf() {
        let dir = tempfile::tempdir().unwrap();
        // Buffer: 3 positions (VEC3 f32) then 3 indices (u16).
        let positions: [[f32; 3]; 3] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let mut buf = Vec::new();
        for p in positions {
            for f in p {
                buf.extend_from_slice(&f.to_le_bytes());
            }
        }
        let idx_offset = buf.len();
        for i in [0u16, 1, 2] {
            buf.extend_from_slice(&i.to_le_bytes());
        }
        std::fs::write(dir.path().join("tri.bin"), &buf).unwrap();

        let gltf = format!(
            r#"{{
  "asset": {{ "version": "2.0" }},
  "scene": 0,
  "scenes": [{{ "nodes": [0] }}],
  "nodes": [{{ "mesh": 0 }}],
  "meshes": [{{ "name": "Tri", "primitives": [{{ "attributes": {{ "POSITION": 0 }}, "indices": 1 }}] }}],
  "buffers": [{{ "uri": "tri.bin", "byteLength": {total} }}],
  "bufferViews": [
    {{ "buffer": 0, "byteOffset": 0, "byteLength": {pos_len}, "target": 34962 }},
    {{ "buffer": 0, "byteOffset": {idx_offset}, "byteLength": 6, "target": 34963 }}
  ],
  "accessors": [
    {{ "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "min": [0,0,0], "max": [1,1,0] }},
    {{ "bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR" }}
  ]
}}"#,
            total = buf.len(),
            pos_len = idx_offset,
            idx_offset = idx_offset,
        );
        let gltf_path = dir.path().join("tri.gltf");
        std::fs::write(&gltf_path, gltf).unwrap();

        let imported = import_gltf(&gltf_path).unwrap();
        assert_eq!(imported.meshes.len(), 1);
        let m = &imported.meshes[0].mesh;
        assert_eq!(m.triangle_count(), 1);
        assert_eq!(m.vertex_count(), 3);
        // Normals were generated (the source had none): +Z face.
        let n = m.submeshes[0].vertices[0].normal;
        assert!(n[2] > 0.9, "generated normal faces +Z, got {n:?}");
        // A plain triangle has no skin / skeleton / clips.
        assert!(imported.skeletons.is_empty());
        assert!(imported.clips.is_empty());
        assert!(!m.submeshes[0].is_skinned());
        assert!(imported.meshes[0].skin.is_none());
    }

    /// Append `f32`s to `buf` and return the starting byte offset.
    fn push_f32s(buf: &mut Vec<u8>, vals: &[f32]) -> usize {
        let off = buf.len();
        for &v in vals {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        off
    }

    /// A programmatically built skinned glTF: a 3-joint arm (chain along +Y), a
    /// one-triangle skinned mesh, and a 2-keyframe rotation clip on the middle
    /// joint. No binary fixtures — the buffer + JSON are constructed in-test.
    #[test]
    fn imports_a_skinned_arm_with_a_rotation_clip() {
        let dir = tempfile::tempdir().unwrap();
        let mut buf: Vec<u8> = Vec::new();

        // 0: POSITION — 3 verts (VEC3 f32).
        let pos_off = push_f32s(&mut buf, &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0]);
        // 1: WEIGHTS_0 — 3 verts (VEC4 f32), each fully weighted to one joint.
        let wgt_off = push_f32s(
            &mut buf,
            &[
                1.0, 0.0, 0.0, 0.0, // v0 → joint 0
                1.0, 0.0, 0.0, 0.0, // v1 → joint 0
                1.0, 0.0, 0.0, 0.0, // v2 → joint 1
            ],
        );
        // 2: inverseBindMatrices — 3 × MAT4 f32 (identity).
        let ibm_off = {
            let off = buf.len();
            for _ in 0..3 {
                let m = Mat4::IDENTITY.to_cols_array();
                for v in m {
                    buf.extend_from_slice(&v.to_le_bytes());
                }
            }
            off
        };
        // 3: animation input times — 2 (SCALAR f32).
        let time_off = push_f32s(&mut buf, &[0.0, 1.0]);
        // 4: animation output rotations — 2 × VEC4 f32 (identity, then 90° about Z).
        let rot_off = push_f32s(
            &mut buf,
            &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.707_106_77, 0.707_106_77],
        );
        // 5: indices — 3 × u16, 4-byte aligned.
        while !buf.len().is_multiple_of(4) {
            buf.push(0);
        }
        let idx_off = buf.len();
        for i in [0u16, 1, 2] {
            buf.extend_from_slice(&i.to_le_bytes());
        }
        while !buf.len().is_multiple_of(4) {
            buf.push(0);
        }
        // 6: JOINTS_0 — 3 verts × VEC4 u16.
        let joint_off = buf.len();
        for j in [[0u16, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]] {
            for c in j {
                buf.extend_from_slice(&c.to_le_bytes());
            }
        }

        std::fs::write(dir.path().join("arm.bin"), &buf).unwrap();

        let gltf = format!(
            r#"{{
  "asset": {{ "version": "2.0" }},
  "scene": 0,
  "scenes": [{{ "nodes": [0, 3] }}],
  "nodes": [
    {{ "name": "joint0", "translation": [0,0,0], "children": [1] }},
    {{ "name": "joint1", "translation": [0,1,0], "children": [2] }},
    {{ "name": "joint2", "translation": [0,1,0] }},
    {{ "name": "armMeshNode", "mesh": 0, "skin": 0 }}
  ],
  "meshes": [{{ "name": "Arm", "primitives": [{{
    "attributes": {{ "POSITION": 0, "JOINTS_0": 6, "WEIGHTS_0": 1 }},
    "indices": 5
  }}] }}],
  "skins": [{{ "name": "ArmSkel", "joints": [0,1,2], "inverseBindMatrices": 2 }}],
  "animations": [{{
    "name": "Wave",
    "channels": [{{ "sampler": 0, "target": {{ "node": 1, "path": "rotation" }} }}],
    "samplers": [{{ "input": 3, "output": 4, "interpolation": "LINEAR" }}]
  }}],
  "buffers": [{{ "uri": "arm.bin", "byteLength": {total} }}],
  "bufferViews": [
    {{ "buffer": 0, "byteOffset": {pos_off}, "byteLength": 36, "target": 34962 }},
    {{ "buffer": 0, "byteOffset": {wgt_off}, "byteLength": 48, "target": 34962 }},
    {{ "buffer": 0, "byteOffset": {ibm_off}, "byteLength": 192 }},
    {{ "buffer": 0, "byteOffset": {time_off}, "byteLength": 8 }},
    {{ "buffer": 0, "byteOffset": {rot_off}, "byteLength": 32 }},
    {{ "buffer": 0, "byteOffset": {idx_off}, "byteLength": 6, "target": 34963 }},
    {{ "buffer": 0, "byteOffset": {joint_off}, "byteLength": 24, "target": 34962 }}
  ],
  "accessors": [
    {{ "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "min": [0,0,0], "max": [1,2,0] }},
    {{ "bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC4" }},
    {{ "bufferView": 2, "componentType": 5126, "count": 3, "type": "MAT4" }},
    {{ "bufferView": 3, "componentType": 5126, "count": 2, "type": "SCALAR", "min": [0.0], "max": [1.0] }},
    {{ "bufferView": 4, "componentType": 5126, "count": 2, "type": "VEC4" }},
    {{ "bufferView": 5, "componentType": 5123, "count": 3, "type": "SCALAR" }},
    {{ "bufferView": 6, "componentType": 5123, "count": 3, "type": "VEC4" }}
  ]
}}"#,
            total = buf.len(),
            pos_off = pos_off,
            wgt_off = wgt_off,
            ibm_off = ibm_off,
            time_off = time_off,
            rot_off = rot_off,
            idx_off = idx_off,
            joint_off = joint_off,
        );
        let path = dir.path().join("arm.gltf");
        std::fs::write(&path, gltf).unwrap();

        let imported = import_gltf(&path).unwrap();

        // One skeleton: a 3-joint chain, parents preceding children.
        assert_eq!(imported.skeletons.len(), 1);
        let sk = &imported.skeletons[0].skeleton;
        assert_eq!(sk.len(), 3);
        assert_eq!(sk.joint(0).unwrap().parent, None);
        assert_eq!(sk.joint(1).unwrap().parent, Some(0));
        assert_eq!(sk.joint(2).unwrap().parent, Some(1));
        // Joint local binds carry the node translations.
        assert_eq!(sk.joint(1).unwrap().local_bind.translation, [0.0, 1.0, 0.0]);

        // The mesh is skinned to skeleton 0 with index-aligned, normalized weights.
        let im = &imported.meshes[0];
        assert_eq!(im.skin, Some(0));
        let sub = &im.mesh.submeshes[0];
        assert!(sub.is_skinned());
        assert_eq!(sub.skin.len(), sub.vertices.len());
        assert_eq!(sub.skin[0].joints, [0, 0, 0, 0]);
        assert_eq!(sub.skin[2].joints, [1, 0, 0, 0]);
        assert!((sub.skin[0].weights.iter().sum::<f32>() - 1.0).abs() < 1e-6);

        // One clip: a 2-key rotation track on joint 1, bound to skeleton 0.
        assert_eq!(imported.clips.len(), 1);
        let clip = &imported.clips[0];
        assert_eq!(clip.skeleton, Some(0));
        assert_eq!(clip.clip.duration, 1.0);
        assert_eq!(clip.clip.tracks.len(), 1);
        let track = &clip.clip.tracks[0];
        assert_eq!(track.joint, 1);
        let rot = track.rotation.as_ref().unwrap();
        assert_eq!(rot.times, vec![0.0, 1.0]);
        assert_eq!(rot.values.len(), 2);
    }
}
