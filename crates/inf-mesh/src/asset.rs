//! The `.inf_mesh` payload schema.
//!
//! A mesh asset is one or more submeshes of interleaved vertices + a 32-bit
//! index buffer, plus a local-space bounding box. Vertices are interleaved
//! (position/normal/uv/tangent) because that is the layout the renderer uploads
//! directly and the layout `meshopt`'s vertex-fetch optimization assumes.

use bytemuck::{Pod, Zeroable};
use inf_asset::{AssetKind, AssetPayload};
use serde::{Deserialize, Serialize};

/// **The tangent an importer writes when the source file has none** — `[1, 0, 0, 1]`.
///
/// Named once, here, because it is not a direction: it is the absence of one,
/// spelled as a value so [`MeshVertex`] can stay `Pod`. Every producer in the
/// tree writes exactly this (`gltf_import`'s `unwrap_or`, `obj_import`
/// unconditionally, `inf_dcc::TANGENT_FALLBACK` for a corner with no usable
/// accumulation, and [`MeshVertex::default`]), and until P28.2 nothing on the
/// meshlet path read the field, so the placeholder cost nothing.
///
/// It costs something now, which is why it has a name (P28.2 audit): a v3 cook
/// packs whatever is in this field into `VgeomVertex::tangent`, and
/// `pack_tangent` returns its `NO_TANGENT` sentinel only for a *non-finite or
/// zero-length* input. `[1, 0, 0]` is neither. So an OBJ mesh, a glTF with no
/// `TANGENT` attribute, or a UV-less DCC export would shade through a constant
/// object-space +X tangent instead of the per-fragment derivative frame it used
/// before the channel existed — a visible change to a surface whose author
/// supplied nothing. [`MeshAsset::vgeom_streams`] is where that is stopped.
pub const TANGENT_PLACEHOLDER: [f32; 4] = [1.0, 0.0, 0.0, 1.0];

/// One interleaved vertex. `#[repr(C)]` + `Pod` so it uploads to a GPU buffer
/// and feeds `meshopt` without a copy. 48 bytes, naturally aligned (no padding).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Pod, Zeroable)]
pub struct MeshVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    /// xyz = tangent, w = handedness sign (±1) for the bitangent.
    pub tangent: [f32; 4],
}

impl Default for MeshVertex {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            normal: [0.0, 1.0, 0.0],
            uv: [0.0; 2],
            tangent: TANGENT_PLACEHOLDER,
        }
    }
}

/// Per-vertex skinning influences (P11.1): the four joints that deform a vertex
/// and their **normalized** weights (`Σ = 1`). Stored as a **parallel stream**
/// to [`SubMesh::vertices`] (index-aligned) rather than folded into
/// [`MeshVertex`] so the existing 48-byte interleaved vertex — and every
/// `.inf_mesh` payload written before skinning — is untouched. `#[repr(C)]` +
/// `Pod` so the renderer uploads it straight to a GPU vertex buffer. 24 bytes,
/// naturally aligned (no padding).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Pod, Zeroable)]
pub struct VertexSkin {
    /// Joint indices into the mesh's skeleton (glTF `JOINTS_0`, widened to u16).
    pub joints: [u16; 4],
    /// Influence weights (glTF `WEIGHTS_0`), normalized so they sum to 1.
    pub weights: [f32; 4],
}

impl Default for VertexSkin {
    fn default() -> Self {
        Self {
            joints: [0; 4],
            weights: [1.0, 0.0, 0.0, 0.0],
        }
    }
}

impl VertexSkin {
    /// Normalize the weights so they sum to 1 (falls back to "all joint 0" when
    /// the weights are degenerate / zero).
    ///
    /// # `sum > 1e-6` catches NaN and lets `+inf` through (C4-37)
    ///
    /// The guard reads as though it rejects everything unusual, and it half
    /// does: `NaN > 1e-6` is false, so a NaN weight takes the safe fallback.
    /// `inf > 1e-6` is **true**, so `weights = [inf, 0, 0, 0]` passes and each
    /// division is `inf / inf` — NaN, manufactured out of an input that never
    /// held one. It is then serialized into the `.inf_mesh` skin stream, which
    /// is `#[repr(C)] Pod` and uploaded straight to a GPU vertex buffer.
    ///
    /// The importers refuse a non-finite `WEIGHTS_0` at the door
    /// ([`crate::validate`]), so this is the second line rather than the first;
    /// it stays because `VertexSkin` is public and a caller who built one by
    /// hand never crossed that door.
    pub fn normalized(mut self) -> Self {
        let sum: f32 = self.weights.iter().sum();
        if sum.is_finite() && sum > 1e-6 && self.weights.iter().all(|w| w.is_finite()) {
            for w in &mut self.weights {
                *w /= sum;
            }
        } else {
            self.weights = [1.0, 0.0, 0.0, 0.0];
        }
        self
    }
}

/// An axis-aligned bounding box in the mesh's local space (render f32).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Aabb {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

impl Aabb {
    /// An empty box that grows to fit inserted points.
    pub fn empty() -> Self {
        Self {
            min: [f32::INFINITY; 3],
            max: [f32::NEG_INFINITY; 3],
        }
    }

    pub fn grow(&mut self, p: [f32; 3]) {
        for ((mn, mx), &pv) in self.min.iter_mut().zip(self.max.iter_mut()).zip(p.iter()) {
            *mn = mn.min(pv);
            *mx = mx.max(pv);
        }
    }

    pub fn from_points(points: impl IntoIterator<Item = [f32; 3]>) -> Self {
        let mut b = Self::empty();
        for p in points {
            b.grow(p);
        }
        if b.min[0] > b.max[0] {
            // No points: collapse to origin.
            b = Aabb {
                min: [0.0; 3],
                max: [0.0; 3],
            };
        }
        b
    }

    pub fn center(&self) -> [f32; 3] {
        [
            (self.min[0] + self.max[0]) * 0.5,
            (self.min[1] + self.max[1]) * 0.5,
            (self.min[2] + self.max[2]) * 0.5,
        ]
    }

    /// Radius of the bounding sphere around the box center — used to frame the
    /// mesh for thumbnails and F-focus.
    pub fn radius(&self) -> f32 {
        let c = self.center();
        let dx = self.max[0] - c[0];
        let dy = self.max[1] - c[1];
        let dz = self.max[2] - c[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

/// One drawable submesh: an interleaved vertex buffer + indices, tagged with the
/// material slot it should draw with (an index into the mesh's material slot
/// list, resolved to a real material asset by the importer).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SubMesh {
    pub name: String,
    pub vertices: Vec<MeshVertex>,
    pub indices: Vec<u32>,
    /// Material slot index (glTF primitive material), or `None` for default.
    pub material_slot: Option<u32>,
    /// Per-vertex skinning influences (P11.1), index-aligned to `vertices`.
    /// **Additive field:** `#[serde(default)]` → an empty vec for every pre-P11
    /// payload and every static (unskinned) primitive, so this submesh is a rigid
    /// mesh unless a skin stream is present. When non-empty, `skin.len() ==
    /// vertices.len()`.
    #[serde(default)]
    pub skin: Vec<VertexSkin>,
}

impl SubMesh {
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }
    /// Whether this submesh carries per-vertex skinning influences.
    pub fn is_skinned(&self) -> bool {
        !self.skin.is_empty()
    }
}

/// The `.inf_mesh` payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshAsset {
    pub schema_version: u32,
    pub submeshes: Vec<SubMesh>,
    pub bounds: Aabb,
    /// Names of the material slots this mesh expects, in slot order. The
    /// importer maps these to material asset GUIDs (stored as dependencies).
    pub material_slots: Vec<String>,
    /// **The material each slot resolves to** — v3, wave CHAR1a.3 — index-aligned
    /// to [`material_slots`](Self::material_slots), `None` where the importer
    /// bound nothing.
    ///
    /// # Why the names were not enough
    ///
    /// [`material_slots`](Self::material_slots) has carried slot NAMES since P4
    /// and [`SubMesh::material_slot`] has carried the index into them, so the
    /// geometry has always known which of its triangles wanted which slot. What
    /// nothing carried was **which asset that slot is** — the importer resolved
    /// it, bound slot 0 to the whole body, and dropped the rest. A skinned mesh
    /// therefore drew with one material however many slots it had: measured on
    /// `SKM_Manny`, **2** slots and one drawn; on a MetaHuman face, **12**.
    ///
    /// It could not live in the sidecar. The sidecar is the *loose file* half of
    /// an asset and neither a cooked `.ipack` nor a PIE `ScenePayload` carries
    /// one — so a slot table there would make the editor draw a face correctly
    /// and the shipped build draw it in one colour, which is the exact divergence
    /// the PIE == shipping law exists to prevent. It is payload data because both
    /// hosts have to read it.
    ///
    /// A shorter list than [`material_slots`](Self::material_slots) is legal and
    /// means "the rest are unbound": every reader indexes with `get`.
    pub material_slot_assets: Vec<Option<inf_asset::AssetId>>,
}

impl MeshAsset {
    /// Schema v2 (P11.1): [`SubMesh`] gained the additive, `#[serde(default)]`
    /// `skin` stream. A v1 payload has no skinned submeshes; new payloads always
    /// write the (possibly empty) stream.
    ///
    /// **v3 (wave CHAR1a.3)** appends
    /// [`material_slot_assets`](Self::material_slot_assets). bincode is
    /// POSITIONAL — the standing law of this tree, caught three times — so
    /// `#[serde(default)]` buys a v2 payload nothing: the decoder simply runs off
    /// the end. The rung is a real one, with a frozen v2 record and a
    /// [`AssetPayload::decode_wire`] branch, so **every `.inf_mesh` written
    /// before this wave keeps reading** and comes back with an empty slot table,
    /// which is exactly what it meant.
    pub const CURRENT_VERSION: u32 = 3;

    /// Assemble a mesh from submeshes, computing the overall bounds.
    pub fn new(submeshes: Vec<SubMesh>, material_slots: Vec<String>) -> Self {
        let bounds = Aabb::from_points(
            submeshes
                .iter()
                .flat_map(|s| s.vertices.iter().map(|v| v.position)),
        );
        Self {
            schema_version: Self::CURRENT_VERSION,
            submeshes,
            bounds,
            material_slots,
            material_slot_assets: Vec::new(),
        }
    }

    /// **Bind the slot table** (v3): `(slot index, material)` pairs, in any
    /// order, written into a list index-aligned to
    /// [`material_slots`](Self::material_slots).
    ///
    /// A pair naming a slot the mesh does not have is IGNORED rather than
    /// growing the list past the slots: a table longer than the slots is a table
    /// a reader can index past the mesh's own submeshes, which is how a face
    /// ends up wearing a body's skin.
    pub fn bind_material_slots(
        &mut self,
        pairs: impl IntoIterator<Item = (u32, inf_asset::AssetId)>,
    ) {
        if self.material_slots.is_empty() {
            return;
        }
        self.material_slot_assets
            .resize(self.material_slots.len(), None);
        for (slot, id) in pairs {
            if let Some(entry) = self.material_slot_assets.get_mut(slot as usize) {
                *entry = Some(id);
            }
        }
    }

    /// The material bound to one slot, or `None`.
    pub fn material_for_slot(&self, slot: u32) -> Option<inf_asset::AssetId> {
        self.material_slot_assets
            .get(slot as usize)
            .copied()
            .flatten()
    }

    /// **The DRAWN sections of this mesh's skinned geometry**, as
    /// `(first index, index count, material slot)` over the ONE concatenated
    /// index buffer both hosts build with `skinned_mesh_data`.
    ///
    /// The concatenation rule is the projectors': submeshes in payload order,
    /// each rebased onto the running vertex count, **including unskinned ones**
    /// (a rigid part welded to a skeleton's root is kept and pinned to joint 0).
    /// This function walks the same list in the same order, so the ranges it
    /// returns address the buffer that function produced — which is why it lives
    /// here, beside the payload, rather than being a third copy on each side.
    ///
    /// Returns an EMPTY vector for a mesh whose submeshes all want one slot:
    /// "one section" is what every reader already does, so saying it costs a
    /// draw call's worth of bookkeeping for no decision.
    pub fn skinned_sections(&self) -> Vec<(u32, u32, u32)> {
        let mut out: Vec<(u32, u32, u32)> = Vec::new();
        let mut first = 0u32;
        for sm in &self.submeshes {
            let count = sm.indices.len() as u32;
            let slot = sm.material_slot.unwrap_or(0);
            match out.last_mut() {
                // Adjacent submeshes wanting one slot are ONE range: a mesh split
                // into parts for authoring reasons must not become one draw call
                // per part.
                Some(last) if last.2 == slot && last.0 + last.1 == first => last.1 += count,
                _ => out.push((first, count, slot)),
            }
            first += count;
        }
        if out.len() < 2 {
            return Vec::new();
        }
        out
    }

    pub fn triangle_count(&self) -> usize {
        self.submeshes.iter().map(SubMesh::triangle_count).sum()
    }
    pub fn vertex_count(&self) -> usize {
        self.submeshes.iter().map(SubMesh::vertex_count).sum()
    }

    /// Flatten every submesh into one combined `(positions, normals, uvs,
    /// tangents, indices)` geometry — the raw streams the virtualized-geometry
    /// builder (`inf-vgeom`) consumes at cook time. Submesh index buffers are
    /// rebased onto the concatenated vertex buffer.
    ///
    /// v1 fuses all material slots into one geometry (virtualized geometry treats
    /// the whole mesh as one clusterizable surface); per-slot meshlet tagging is a
    /// documented follow-up. Additive: does not change any payload.
    ///
    /// **The tangent stream joined in P28.2**, where `.inf_vmesh` v3 gave
    /// `VgeomVertex` somewhere to put one. It has been sitting in
    /// [`MeshVertex::tangent`] since P4 — read by nothing that reaches the meshlet
    /// path, which is exactly the gap `docs/memos/p26-5-vertex-streams.md`
    /// recorded and routed here.
    ///
    /// **A mesh whose every vertex carries [`TANGENT_PLACEHOLDER`] hands back an
    /// EMPTY tangent stream** (P28.2 audit), which `build_vgeom` writes as
    /// `NO_TANGENT` and the shader reads as "use the derivative frame". The
    /// alternative was measured and is a regression: the importers substitute
    /// `[1, 0, 0, 1]` when a source file has no tangents, `pack_tangent` refuses
    /// only a non-finite or zero-length input, so every OBJ mesh and every
    /// untangented glTF would have started shading through a constant +X tangent
    /// the moment the container went to v3. The test is over the WHOLE mesh and
    /// exact — never per vertex and never a tolerance — and it cannot misfire: a
    /// surface whose authored tangent really is uniformly `+X` has axis-aligned
    /// uvs, and the derivative frame derives `+X` for it too.
    #[allow(clippy::type_complexity)]
    pub fn vgeom_streams(
        &self,
    ) -> (
        Vec<[f32; 3]>,
        Vec<[f32; 3]>,
        Vec<[f32; 2]>,
        Vec<[f32; 4]>,
        Vec<u32>,
    ) {
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut uvs = Vec::new();
        let mut tangents = Vec::new();
        let mut indices = Vec::new();
        for sm in &self.submeshes {
            let base = positions.len() as u32;
            for v in &sm.vertices {
                positions.push(v.position);
                normals.push(v.normal);
                uvs.push(v.uv);
                tangents.push(v.tangent);
            }
            indices.extend(sm.indices.iter().map(|&i| i + base));
        }
        if tangents.iter().all(|t| *t == TANGENT_PLACEHOLDER) {
            tangents.clear();
        }
        (positions, normals, uvs, tangents, indices)
    }
}

impl MeshAsset {
    /// The three [`crate::validate`] questions, asked of a **decoded payload**
    /// rather than of an imported file (round-2 finding B2).
    ///
    /// Separate from [`AssetPayload::migrate`] so a caller holding a
    /// hand-assembled `MeshAsset` — the DCC writer, the grammar bake, a test —
    /// can ask the same question without a round trip through bincode.
    pub fn validate(&self) -> Result<(), crate::MeshError> {
        use crate::validate::AllFinite;
        let mut total: usize = 0;
        for (s, sm) in self.submeshes.iter().enumerate() {
            let what = format!("submesh {s} ({})", sm.name);
            // The vertex streams. `MeshVertex` is `#[repr(C)] Pod` and is
            // uploaded verbatim; a NaN position also poisons every cull bound
            // derived from it, and `f32::min`/`max` ignore NaN so the bounds
            // still look healthy.
            for v in &sm.vertices {
                if !(v.position.all_finite()
                    && v.normal.all_finite()
                    && v.uv.all_finite()
                    && v.tangent.all_finite())
                {
                    return Err(crate::MeshError::Malformed(format!(
                        "{what}: a vertex attribute is not a finite number (NaN or infinity)"
                    )));
                }
            }
            // The index buffer, against ITS OWN vertex buffer — this is the one
            // that reaches `meshopt` through raw FFI.
            crate::validate::reject_out_of_range(&sm.indices, sm.vertices.len(), &what)?;
            // …and it must be a whole number of TRIANGLES.
            crate::validate::reject_partial_triangle(sm.indices.len(), &what)?;
            // The parallel skin stream. `SubMesh`'s own doc states the
            // invariant ("when non-empty, `skin.len() == vertices.len()`") and
            // until now nothing enforced it on the decode path; the GPU
            // uploads both as one interleaved draw.
            if !sm.skin.is_empty() {
                crate::validate::reject_length_mismatch(
                    sm.skin.len(),
                    sm.vertices.len(),
                    &format!("{what}: skin"),
                    "vertices",
                )?;
            }
            for sk in &sm.skin {
                if !sk.weights.all_finite() {
                    return Err(crate::MeshError::Malformed(format!(
                        "{what}: a skin weight is not a finite number"
                    )));
                }
            }
            // `vgeom_streams` rebases every submesh onto one concatenated
            // buffer with `i + base` in bare `u32`, and the release profile has
            // no overflow checks. Refuse a payload whose concatenation cannot
            // be addressed rather than wrap into a valid-looking index.
            total = total
                .checked_add(sm.vertices.len())
                .filter(|t| u32::try_from(*t).is_ok())
                .ok_or_else(|| {
                    crate::MeshError::Malformed(format!(
                        "{what}: the mesh's combined vertex count exceeds a 32-bit index buffer"
                    ))
                })?;
        }
        Ok(())
    }
}

impl AssetPayload for MeshAsset {
    const KIND: AssetKind = AssetKind::Mesh;
    const SCHEMA_VERSION: u32 = Self::CURRENT_VERSION;
    fn schema_version(&self) -> u32 {
        self.schema_version
    }

    /// Reject newer-than-current (the default rule), then run
    /// [`MeshAsset::validate`] — round-2 finding **B2**, and the same
    /// `AnimClipAsset`/`TextureAsset`/`BiomeSet` shape.
    ///
    /// # The import door has a second entrance
    ///
    /// [`crate::validate`]'s module doc says a mesh from outside is checked
    /// "before any of it becomes engine data", and for `.gltf`/`.glb`/`.obj`
    /// that is true. But a `.inf_mesh` **on disk** is also bytes somebody else
    /// wrote — the Content Drawer scans every loose payload under the project
    /// root, an asset arrives in a zip, a pack is hand-edited — and this type
    /// used the default `migrate`, which reads one integer and asks nothing
    /// about the buffers behind it.
    ///
    /// Both production consumers of a decoded `.inf_mesh` hand its index buffer
    /// straight to `meshopt::generate_vertex_remap` through raw FFI:
    /// `inf_editor_core::assets::vmesh` and the cook's `build_vgeom`. That is
    /// C4-1's out-of-bounds heap **write**, verbatim, at a door the importer's
    /// validator never sees. `crate::optimize()` got its own backstop in Wave
    /// B; `inf_vgeom::build_vgeom` did not, and now has one too — but the
    /// place to stop this is the decode, because by the time a builder sees the
    /// streams the payload has already been trusted by a dozen other readers.
    fn migrate(self) -> inf_asset::Result<Self> {
        let found = self.schema_version;
        if found > Self::SCHEMA_VERSION {
            return Err(inf_asset::AssetError::SchemaTooNew {
                kind: Self::KIND.slug(),
                found,
                current: Self::SCHEMA_VERSION,
            });
        }
        self.validate()
            .map_err(|e| inf_asset::AssetError::Decode(format!("invalid mesh: {e}")))?;
        Ok(self)
    }

    /// **The v2 → v3 rung.** A v2 payload has no slot table, and "no slot table"
    /// is exactly what an empty one means — so this is a pure default-fill and
    /// an honest migration rather than a guess.
    ///
    /// v1 is deliberately absent: it predates the `skin` stream and falls through
    /// to the current shape, fails, and `decode` turns that into `SchemaTooOld`
    /// with this type's own remedy — the behaviour every `.inf_mesh` in the tree
    /// has had since P11.1, which this rung does not change.
    fn migrates_from(v: u32) -> bool {
        v == 2
    }

    fn decode_wire(bytes: &[u8], found: Option<u32>) -> inf_asset::Result<Self> {
        if found == Some(2) {
            let old: mesh_v2::MeshAsset = inf_asset::decode_shape(bytes)
                .map_err(|e| inf_asset::AssetError::Decode(format!("v2 mesh: {e}")))?;
            return Ok(old.into_current());
        }
        inf_asset::decode_shape(bytes)
    }
}

/// **The frozen schema-v2 `.inf_mesh` record** — the shape before CHAR1a.3
/// appended the material-slot table.
///
/// Ladder-local and declared field-for-field rather than derived from the live
/// types, which is the whole point (`inf_anim`'s `skel_v2` says it first): a
/// shape built by asking today's encoder what it emits reproduces the right
/// bytes and pins nothing, because it moves whenever the encoder does. This says
/// what v2 *was*, so the day someone appends a second table without a bump, the
/// arms below stop matching real bytes.
///
/// [`SubMesh`] itself did not move, so it is named rather than re-declared — a
/// copy of a 5-field struct that has not changed would be a second thing to keep
/// in step, and the arms below write v2 bytes THROUGH this record, so a change to
/// `SubMesh` breaks them loudly rather than silently.
pub(crate) mod mesh_v2 {
    use serde::{Deserialize, Serialize};

    /// v2's payload: four fields, no slot table.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct MeshAsset {
        pub schema_version: u32,
        pub submeshes: Vec<super::SubMesh>,
        pub bounds: super::Aabb,
        pub material_slots: Vec<String>,
    }

    impl MeshAsset {
        pub fn into_current(self) -> super::MeshAsset {
            super::MeshAsset {
                // The version is RAISED here, not carried: the value in hand is
                // the current shape, and a payload that decoded through the v2
                // rung and kept saying "2" would be re-migrated on every load and
                // would re-encode as a v2 file the moment anything saved it.
                schema_version: super::MeshAsset::CURRENT_VERSION,
                submeshes: self.submeshes,
                bounds: self.bounds,
                material_slots: self.material_slots,
                material_slot_assets: Vec::new(),
            }
        }

        /// The v2 projection of a current payload — the encoder half, so the arms
        /// can write real v2 bytes from a v3 value.
        #[cfg_attr(not(test), allow(dead_code))]
        pub fn from_current(m: &super::MeshAsset) -> Self {
            Self {
                schema_version: 2,
                submeshes: m.submeshes.clone(),
                bounds: m.bounds,
                material_slots: m.material_slots.clone(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inf_asset::{decode, encode};

    /// **The placeholder is not a direction, and `vgeom_streams` must not ship it
    /// as one** (P28.2 audit). Exact and whole-mesh: one authored tangent
    /// anywhere keeps the whole stream, because the placeholders beside it are
    /// then corners of a real field rather than the absence of one.
    #[test]
    fn an_untangented_mesh_hands_the_vgeom_builder_no_tangent_stream() {
        let quad = |tangent: Option<[f32; 4]>| {
            let mut v = vec![MeshVertex::default(); 4];
            if let Some(t) = tangent {
                v[2].tangent = t;
            }
            MeshAsset::new(
                vec![SubMesh {
                    name: "q".into(),
                    vertices: v,
                    indices: vec![0, 1, 2, 0, 2, 3],
                    material_slot: None,
                    skin: Vec::new(),
                }],
                Vec::new(),
            )
        };

        let (_, _, _, tangents, _) = quad(None).vgeom_streams();
        assert!(
            tangents.is_empty(),
            "a mesh of nothing but placeholders offered the builder a direction"
        );

        let authored = [0.0, 0.0, 1.0, -1.0];
        let (_, _, _, tangents, _) = quad(Some(authored)).vgeom_streams();
        assert_eq!(
            tangents.len(),
            4,
            "one authored tangent must keep the whole stream"
        );
        assert_eq!(tangents[2], authored);
        assert_eq!(tangents[0], TANGENT_PLACEHOLDER);
    }

    fn quad() -> SubMesh {
        let v = |x: f32, y: f32| MeshVertex {
            position: [x, y, 0.0],
            ..Default::default()
        };
        SubMesh {
            name: "quad".into(),
            vertices: vec![v(0.0, 0.0), v(1.0, 0.0), v(1.0, 1.0), v(0.0, 1.0)],
            indices: vec![0, 1, 2, 0, 2, 3],
            material_slot: Some(0),
            skin: Vec::new(),
        }
    }

    #[test]
    fn bounds_and_counts() {
        let m = MeshAsset::new(vec![quad()], vec!["Default".into()]);
        assert_eq!(m.triangle_count(), 2);
        assert_eq!(m.vertex_count(), 4);
        assert_eq!(m.bounds.min, [0.0, 0.0, 0.0]);
        assert_eq!(m.bounds.max, [1.0, 1.0, 0.0]);
    }

    #[test]
    fn payload_round_trips_deterministically() {
        let m = MeshAsset::new(vec![quad()], vec!["Default".into()]);
        let a = encode(&m).unwrap();
        let b = encode(&m).unwrap();
        assert_eq!(a, b);
        let back: MeshAsset = decode(&a).unwrap();
        assert_eq!(back, m);
    }
}
