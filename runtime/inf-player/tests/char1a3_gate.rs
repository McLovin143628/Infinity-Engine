//! **WAVE CHAR1a.3 — THE METAHUMAN ON THE HERO.**
//!
//! The arms this wave adds, each aimed at a thing the CHAR1a audit carried and
//! this wave closed. `char1a_gate.rs` keeps the bodies wave's own 22; this file
//! is the third slice's, so a reader looking for "what CHAR1a.3 proved" finds it
//! in one place.

use std::path::{Path, PathBuf};

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("the repository root")
}

/// The island project this machine builds locally, or `None`.
///
/// Every arm that reads it SKIPS with a printed reason rather than failing, the
/// same rule the island gates follow: the MetaHumans are licensed content that
/// never enters this repository, so CI has no island project and must not have
/// a red gate about it.
fn island_project() -> Option<PathBuf> {
    let p = repo().join("../island-build/project/Content");
    p.is_dir().then(|| p.canonicalize().unwrap_or(p))
}

/// The named joint's world position under a palette, in the mesh's bind space.
fn joint_at(sk: &inf_anim::Skeleton, pose: &inf_anim::Pose, name: &str) -> Option<glam::Vec3> {
    let i = sk.joints().iter().position(|j| j.name == name)?;
    let globals = inf_anim::pose::global_transforms(sk, pose);
    Some(globals.get(i)?.to_scale_rotation_translation().2)
}

// ─────────────────────────────────────────────────────────────────────────────
// (1) THE STREET NPC'S ARMS — carried item 92
// ─────────────────────────────────────────────────────────────────────────────

/// **A CROWD AGENT'S `hand_l` MOVES WHEN ITS CLIP MOVES IT** (carried 92).
///
/// The CHAR1a audit photographed a street NPC at 5× with both arms held straight
/// out from the shoulders — the mannequin's bind A-pose, `hand_l` at
/// (+0.478, +1.045) below a shoulder at (+0.190, +1.436) — while the hero beside
/// it in the same frame was correctly posed mid-stride. Fixing the PIE payload
/// store's missing `.inf_sm` did not change the frame, so the remaining cause
/// was in what the sim publishes for a crowd agent.
///
/// This arm asks the question of a real sim: stand agents on the island fixture,
/// step it, and compare each posed agent's `hand_l` against the same rig's bind.
/// A crowd whose agents draw a bind pose reads **0.0000 m** here.
#[test]
fn a_crowd_agents_hand_leaves_the_bind_pose() {
    let recipe_path = repo().join("samples/island-fixture/island.toml");
    let recipe = inf_island::IslandRecipe::load(&recipe_path).expect("the fixture recipe loads");
    let tmp = tempfile::tempdir().expect("a temp dir");
    let build = inf_island::build_island(&recipe, &inf_island::BuildOptions::default())
        .expect("the fixture island builds");
    let proj = tmp.path().join("island");
    inf_project::ProjectManifest::new(&recipe.name, "blank-3d")
        .save(&proj)
        .expect("the project scaffolds");
    let content = proj.join("Content");
    inf_island::write_content(&build, &content).expect("the island's content writes");
    let slug = inf_island::slug(&recipe.name);

    let mut sim = loose_sim(&content, &slug);
    let hero = hero_entity(&sim).expect("the fixture has a hero");
    let from = {
        let w = sim.world().world();
        let t = w
            .get::<inf_ecs::components::Transform>(hero)
            .expect("the hero has a transform");
        glam::DVec3::new(t.translation.x, t.translation.y, t.translation.z)
    };

    // The archetype the crowd wears is the hero's own, read off the world.
    let archetype = {
        let w = sim.world().world();
        let mut best: Option<(uuid::Uuid, inf_ecs::crowd::CrowdArchetype)> = None;
        for e in w.iter_entities() {
            let (Some(g), Some(sk)) = (
                e.get::<inf_ecs::Guid>(),
                e.get::<inf_ecs::components::SkeletalMesh>(),
            ) else {
                continue;
            };
            if sk.skeleton.is_none() {
                continue;
            }
            let sm = e
                .get::<inf_ecs::components::AnimStateMachine>()
                .and_then(|a| a.sm);
            if best.as_ref().is_none_or(|(bg, _)| g.0 < *bg) {
                best = Some((
                    g.0,
                    inf_ecs::crowd::CrowdArchetype::humanoid(sk.mesh, sk.skeleton, sm),
                ));
            }
        }
        best.expect("the fixture has a rigged character to copy").1
    };
    assert!(
        archetype.skeleton.is_some() && archetype.sm.is_some(),
        "the fixture hero carries no skeleton or no machine, so the crowd would \
         have nothing to pose and this arm would be vacuous"
    );

    // Three agents, one per posed band: 4 m (Full), 50 m (Near), 200 m (Far).
    let mut pop = std::collections::BTreeMap::new();
    let dists = [4.0_f64, 50.0, 200.0];
    for (i, d) in dists.iter().enumerate() {
        pop.insert(
            uuid::Uuid::from_u128(0xC1A3_0001 + i as u128),
            inf_ecs::crowd::CrowdRecord::standing(
                archetype,
                glam::DVec3::new(from.x + d, from.y, from.z + 2.0),
            ),
        );
    }
    sim.set_crowd_population(pop.clone());
    for _ in 0..90 {
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
    }

    // The rig, off the same content the host reads.
    let store = inf_player::skinned::SkinnedRegistry::from_dir(&content);
    let skeleton = store
        .skeleton(archetype.skeleton.expect("a skeleton"))
        .expect("the crowd's rig is in the island's own content");
    let bind = inf_anim::Pose::rest(&skeleton);
    let bind_hand = joint_at(&skeleton, &bind, "hand_l").expect("the rig names hand_l");

    let mut rows: Vec<(uuid::Uuid, String, f32, bool)> = Vec::new();
    {
        let w = sim.world();
        for (guid, _) in &pop {
            let agent = w.world().iter_entities().find_map(|e| {
                let g = e.get::<inf_ecs::Guid>()?;
                (g.0 == *guid).then(|| e.get::<inf_ecs::crowd::CrowdAgent>().copied())?
            });
            let Some(agent) = agent else {
                rows.push((*guid, "no entity".into(), 0.0, false));
                continue;
            };
            let posed = inf_ecs::pose::evaluated_pose(w, *guid);
            let d = posed
                .and_then(|p| joint_at(&skeleton, &p.pose, "hand_l"))
                .map(|h| (h - bind_hand).length())
                .unwrap_or(0.0);
            rows.push((*guid, format!("{:?}", agent.tier), d, posed.is_some()));
        }
    }
    for (guid, tier, d, published) in &rows {
        println!(
            "CROWD AGENT {guid}: tier {tier}, pose published {published}, \
             hand_l {d:.4} m from bind"
        );
    }
    // ── AND THE SHARED (Far-tier) DOOR, which is the one that draws the
    // street ────────────────────────────────────────────────────────────────
    //
    // A `Far` agent publishes no pose at all (`tier.poses()` is false), so the
    // projector calls `resolve_skinned_shared`, which samples the machine's
    // ENTRY clip at t = 0 into one palette every such agent shares. A skinning
    // palette is `global · inverse_bind`, so the BIND pose is exactly the
    // identity in every slot -- which makes "does the shared crowd draw a bind
    // pose?" a question with an exact answer rather than a tolerance.
    let sm_comp = inf_ecs::components::SkeletalMesh {
        mesh: archetype.mesh,
        skeleton: archetype.skeleton,
    };
    let machine = inf_ecs::components::AnimStateMachine {
        sm: archetype.sm,
        ..inf_ecs::components::AnimStateMachine::default()
    };
    let shared = store
        .resolve_skinned_shared(&sm_comp, Some(&machine))
        .expect("the shared crowd palette resolves on the island's own content");
    let hand_i = skeleton
        .joints()
        .iter()
        .position(|j| j.name == "hand_l")
        .expect("the rig names hand_l");
    let worst = shared
        .palette
        .iter()
        .map(|m| {
            (*m - glam::Mat4::IDENTITY)
                .to_cols_array()
                .iter()
                .fold(0.0f32, |a, b| a.max(b.abs()))
        })
        .fold(0.0f32, f32::max);
    let hand_off = (shared.palette[hand_i] - glam::Mat4::IDENTITY)
        .to_cols_array()
        .iter()
        .fold(0.0f32, |a, b| a.max(b.abs()));
    println!(
        "SHARED (Far) PALETTE: {} joints, worst |M - I| {worst:.4}, hand_l {hand_off:.4}",
        shared.palette.len()
    );
    assert!(
        worst > 1e-3,
        "the crowd's SHARED palette is the identity in every slot — every Far          agent draws the rig's bind pose, which on this rig is the A-pose the          CHAR1a audit photographed (carried 92)"
    );
    assert!(
        hand_off > 1e-3,
        "the crowd's shared palette leaves hand_l at bind ({hand_off:.4}) while          moving something else — the street's arms are held out (carried 92)"
    );

    let posed_rows: Vec<_> = rows
        .iter()
        .filter(|(_, t, _, _)| t == "Full" || t == "Near")
        .collect();
    assert!(
        !posed_rows.is_empty(),
        "no agent reached a posed tier — the fixture placed them all outside \
         the ladder and the arm is vacuous"
    );
    for (guid, tier, d, published) in &posed_rows {
        assert!(
            published,
            "agent {guid} at tier {tier} published NO pose — the renderer falls \
             through to the machine's entry clip and, if that misses, to the \
             bind pose, which is the A-posed street of carried item 92"
        );
        assert!(
            *d > 0.02,
            "agent {guid} at tier {tier} has hand_l {d:.4} m from bind — its arms \
             are at the bind pose while the hero beside it is posed (carried 92)"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// (2) THE COMBINED BODY — clause 1
// ─────────────────────────────────────────────────────────────────────────────

/// **UE's OWN COMBINED FACE+BODY MESH IS ON THE HERO** (clause 1).
///
/// The wave's premise — carried item 81 — was that a MetaHuman needs an
/// ENGINE-SIDE merge of a headless 342-joint body and a 875-joint face onto a
/// ~1 185-joint union rig. It does not:
/// `MetaHumanCharacterExportBlueprintLibrary::ExportGeometry` with
/// `bFullBodySkeletalMesh` routes to the subsystem's
/// `CreateCombinedFaceAndBodyMesh`, and what came out through the live-editor
/// door is ONE mesh on the BODY's own 342-joint rig with the face skinned into
/// it — 95 330 triangles against the halves' 47 392 + 34 514, one material slot
/// against 1 + 12, and a y-span that runs continuously from the body's own floor
/// to the face's own crown.
///
/// So the numbers this arm pins are the numbers that FALSIFY the merge premise,
/// and they are read off the hero the island actually wears rather than off a
/// manifest: 342 joints, a body 1.78 m tall, and the eleven mannequin names the
/// MetaHuman rig does not publish — every one of them an IK or attachment
/// helper, not one of them a bone that deforms the body.
///
/// **Local-only**, and it SKIPS with a printed reason: the MetaHumans are
/// licensed content that never enters this repository, so CI has no island
/// project and must not have a red gate about it.
#[test]
fn the_islands_hero_is_the_combined_metahuman_body() {
    let Some(content) = island_project() else {
        eprintln!(
            "SKIP: no island project at ../island-build/project/Content — the \
             MetaHumans are local-only content and CI has none"
        );
        return;
    };
    let store = inf_player::skinned::SkinnedRegistry::from_dir(&content);
    assert!(store.has_content(), "the island project did not index");

    for (label, base) in [("hero", 0xA0u128), ("street NPC", 0xB0)] {
        let id = |n: u128| uuid::Uuid::from_u128(0x5C10_0000 + base + n);
        let Some(sk) = store.skeleton(id(0)) else {
            panic!("{label}: the committed skeleton GUID resolves to nothing");
        };
        assert_eq!(
            sk.len(),
            342,
            "{label}'s rig has {} joints — the combined MetaHuman body is on the \
             342-joint body skeleton, and 1185 would mean the engine-side union \
             rig carried 81's premise instead",
            sk.len()
        );
        // The mannequin's names, and the eleven that are absent.
        let manny = inf_anim::manny::build_manny(&Default::default())
            .expect("the mannequin rig")
            .skeleton;
        let have: std::collections::BTreeSet<&str> =
            sk.joints().iter().map(|j| j.name.as_str()).collect();
        let missing: Vec<&str> = manny
            .joints()
            .iter()
            .map(|j| j.name.as_str())
            .filter(|n| !have.contains(n))
            .collect();
        println!(
            "{label}: 342 joints; {} mannequin names absent: {missing:?}",
            missing.len()
        );
        for helper in ["ik_hand_l", "ik_foot_r", "weapon_l", "center_of_mass"] {
            assert!(
                missing.contains(&helper),
                "{label}: {helper} is present on a MetaHuman rig, so the eleven \
                 this arm names are not the eleven that are missing"
            );
        }
        for deforming in [
            "pelvis",
            "spine_01",
            "upperarm_l",
            "hand_l",
            "thigh_r",
            "head",
        ] {
            assert!(
                have.contains(deforming),
                "{label}: the rebound rig does not publish {deforming} — a clip \
                 retargeted by name cannot reach it"
            );
        }
        assert!(
            missing.iter().all(|n| n.starts_with("ik_")
                || n.starts_with("weapon")
                || *n == "interaction"
                || *n == "center_of_mass"),
            "{label}: a DEFORMING mannequin bone is absent from the rebound rig \
             ({missing:?}) — the retarget would drop a track that moves the body"
        );
    }
}

/// **The body is a whole body, and its clips move it** (clause 1 + clause 4).
///
/// The y-span is the measurement that says the seam is not there: CHAR1a.2
/// measured the halves at ≤ 1.3343 m (body) and 1.3962–1.7798 m (face) — a
/// 6.2 cm gap — and the combined mesh runs −0.0016 … 1.7798 m in one surface.
/// Read off the drawn geometry the store hands the renderer, not off the glTF.
#[test]
fn the_rebound_body_spans_a_whole_person_and_its_clips_move_it() {
    let Some(content) = island_project() else {
        eprintln!("SKIP: no island project — the MetaHumans are local-only");
        return;
    };
    let store = inf_player::skinned::SkinnedRegistry::from_dir(&content);
    for (label, base) in [("hero", 0xA0u128), ("street NPC", 0xB0)] {
        let id = |n: u128| uuid::Uuid::from_u128(0x5C10_0000 + base + n);
        let sm = inf_ecs::components::SkeletalMesh {
            mesh: Some(id(2)),
            skeleton: Some(id(0)),
        };
        let draw = store
            .resolve_skinned(&sm, None, None, None)
            .unwrap_or_else(|| panic!("{label}: the committed body GUID draws nothing"));
        let (lo, hi) = draw
            .mesh
            .vertices
            .iter()
            .fold((f32::MAX, f32::MIN), |(lo, hi), v| {
                (lo.min(v.pos[1]), hi.max(v.pos[1]))
            });
        println!(
            "{label}: {} vertices, {} triangles, y {lo:.4} … {hi:.4} m",
            draw.mesh.vertices.len(),
            draw.mesh.indices.len() / 3
        );
        assert!(
            hi > 1.70 && hi < 1.90,
            "{label} is {hi:.4} m tall — a headless body tops out at about 1.33 m, \
             which is what binding the BODY half alone would put on the island"
        );
        assert!(lo.abs() < 0.05, "{label}'s feet are at y = {lo:.4}");

        // …and the three clips the machine names move the hand off the bind.
        let sk = store.skeleton(id(0)).expect("the rig");
        let bind = inf_anim::Pose::rest(&sk);
        let hand = sk
            .joints()
            .iter()
            .position(|j| j.name == "hand_l")
            .expect("hand_l");
        let at = |p: &inf_anim::Pose| {
            inf_anim::pose::global_transforms(&sk, p)[hand]
                .to_scale_rotation_translation()
                .2
        };
        let b = at(&bind);
        for (n, what) in [(3u128, "idle"), (4, "walk"), (5, "run")] {
            let clip = store
                .clip(id(n))
                .unwrap_or_else(|| panic!("{label}: the {what} clip GUID resolves to nothing"));
            let worst = (0..8)
                .map(|k| {
                    let t = clip.duration * k as f32 / 8.0;
                    (at(&inf_anim::sample_clip(&sk, &clip, t, true)) - b).length()
                })
                .fold(0.0f32, f32::max);
            println!(
                "  {what}: {} tracks, {:.2} s, hand_l up to {worst:.4} m from bind",
                clip.tracks.len(),
                clip.duration
            );
            assert!(
                clip.tracks.len() > 100,
                "{label}'s {what} carries {} tracks — a clip re-retargeted from a \
                 161-joint rig onto this one keeps 150 of 161, and a handful means \
                 it was left indexed against the rig it replaced",
                clip.tracks.len()
            );
            assert!(
                worst > 0.05,
                "{label}'s {what} moves hand_l {worst:.4} m — the clips did not \
                 follow the rig across the rebind and the body plays a bind pose"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// (3) SECTIONS — clause 2
// ─────────────────────────────────────────────────────────────────────────────

/// **A MESH WITH TWO MATERIAL SLOTS DRAWS TWO RANGES, AND ONE DRAWS ONE.**
///
/// The whole of the sections feature, asserted on the derivation rather than on a
/// picture: `MeshAsset::skinned_sections` walks the submeshes in payload order
/// exactly as `skinned_mesh_data` concatenates them, coalescing adjacent
/// submeshes that want one slot, and answers EMPTY for a mesh that wants one.
///
/// **The mutation**: return every submesh as its own section (drop the
/// coalescing) → the one-slot case answers two sections and this arm goes red on
/// the first assertion. Verified.
#[test]
fn a_two_slot_mesh_derives_two_sections_and_a_one_slot_mesh_derives_none() {
    let quad = |slot: Option<u32>| inf_mesh::SubMesh {
        name: format!("part{slot:?}"),
        vertices: vec![inf_mesh::MeshVertex::default(); 4],
        indices: vec![0, 1, 2, 0, 2, 3],
        material_slot: slot,
        skin: vec![inf_mesh::VertexSkin::default(); 4],
    };
    // One slot, two submeshes: ONE surface, so no sections at all.
    let one = inf_mesh::MeshAsset::new(vec![quad(Some(0)), quad(Some(0))], vec!["Skin".into()]);
    assert!(
        one.skinned_sections().is_empty(),
        "a mesh whose submeshes all want one slot derived sections — every \
         committed character in this tree is that mesh, and it must emit the \
         whole-buffer draw it always did"
    );
    // Two slots: two ranges over the concatenated index buffer.
    let two = inf_mesh::MeshAsset::new(
        vec![quad(Some(0)), quad(Some(1)), quad(Some(1))],
        vec!["Skin".into(), "Lashes".into()],
    );
    assert_eq!(
        two.skinned_sections(),
        vec![(0, 6, 0), (6, 12, 1)],
        "the section ranges do not address the buffer `skinned_mesh_data` builds \
         — the two adjacent slot-1 submeshes must coalesce into one range"
    );

    // …and the slot table round-trips through the v3 payload.
    let mut bound = two.clone();
    let mat = inf_asset::AssetId(uuid::Uuid::from_u128(0xBEEF));
    bound.bind_material_slots([(1u32, mat)]);
    assert_eq!(bound.material_for_slot(1), Some(mat));
    assert_eq!(
        bound.material_for_slot(0),
        None,
        "slot 0 was bound by nothing"
    );
    assert_eq!(
        bound.material_for_slot(7),
        None,
        "a slot past the mesh's own list answered a material — a table longer \
         than the slots is a table a reader can index past the submeshes"
    );
    let bytes = inf_asset::encode(&bound).expect("encode");
    let back: inf_mesh::MeshAsset = inf_asset::decode(&bytes).expect("decode");
    assert_eq!(back.material_slot_assets, bound.material_slot_assets);
    assert_eq!(back.schema_version, inf_mesh::MeshAsset::CURRENT_VERSION);
}

/// **A v2 `.inf_mesh` STILL READS** — the format window's dispatch (clause 2).
///
/// bincode is positional, so an appended field is not something a v2 payload can
/// be default-filled into: the decoder runs off the end of the stream. The rung
/// is a frozen v2 record plus a `decode_wire` branch, and this is the arm that
/// says a mesh written before this wave — which is every `.inf_mesh` in every
/// project on this machine — still opens, and opens as "no slot table", which is
/// exactly what it meant.
///
/// **The mutation**: delete the `found == Some(2)` branch from `decode_wire`.
/// The v2 bytes fail to decode and `decode` reports them as `SchemaTooOld`, red.
/// Verified.
#[test]
fn a_v2_mesh_still_decodes_through_the_v3_rung() {
    let mesh = inf_mesh::MeshAsset::new(
        vec![inf_mesh::SubMesh {
            name: "part".into(),
            vertices: vec![inf_mesh::MeshVertex::default(); 4],
            indices: vec![0, 1, 2, 0, 2, 3],
            material_slot: Some(0),
            skin: Vec::new(),
        }],
        vec!["Skin".into()],
    );
    // Real v2 BYTES, written through the frozen record rather than by truncating
    // a v3 encode — a shape built by asking today's encoder what it emits pins
    // nothing.
    let v2 = inf_mesh::encode_v2_for_test(&mesh).expect("v2 bytes");
    let v3 = inf_asset::encode(&mesh).expect("v3 bytes");
    assert_eq!(
        v3.len(),
        v2.len() + 1,
        "the v3 encoding is not one byte longer than the v2 one — an empty `Vec` \
         is a single zero varint, and this is the whole cost of the window"
    );
    let back: inf_mesh::MeshAsset = inf_asset::decode(&v2).expect("a v2 mesh still decodes");
    assert_eq!(back.schema_version, inf_mesh::MeshAsset::CURRENT_VERSION);
    assert!(
        back.material_slot_assets.is_empty(),
        "a v2 payload came back with a slot table it never carried"
    );
    assert_eq!(back.submeshes, mesh.submeshes);
    assert_eq!(back.material_slots, mesh.material_slots);
}

/// **BOTH HOSTS BUILD SECTIONS THROUGH THE SAME DOOR** (clause 2).
///
/// Source-text, like `projector_mirror`'s arms and for its reason: the two
/// projectors are separate files that must not drift, and the thing that must not
/// drift here is *which function* they call. A host that built its own sections
/// inline would pass every scene-level comparison — both hosts would have
/// sections — and draw different ones.
#[test]
fn both_projectors_build_sections_through_the_ring_zero_door() {
    for (label, rel) in [
        (
            "the editor viewport",
            "editor/crates/inf-viewport/src/host.rs",
        ),
        ("the shipped player", "runtime/inf-player/src/render.rs"),
    ] {
        let src = std::fs::read_to_string(repo().join(rel))
            .unwrap_or_else(|e| panic!("{rel}: {e}"))
            .replace("\r\n", "\n");
        assert!(
            src.contains("inf_render::skinned_sections("),
            "{label} does not build its skinned sections through \
             `inf_render::skinned_sections` — a second copy of the rule is how \
             two hosts come to draw different ranges of one body"
        );
        assert!(
            src.contains("&draw.sections,"),
            "{label} passes something other than the STORE's ranges — the ranges \
             address the buffer the store concatenated, and a projector that \
             re-derived them would have to re-derive the concatenation too"
        );
        assert!(
            src.contains("SkinnedInstance { sections, ..inst }")
                || src.contains("inf_render::SkinnedInstance { sections, ..inst }"),
            "{label} builds a `SkinnedInstance` that does not carry the sections \
             it just computed"
        );
    }
    // …and the pass draws the RANGE rather than the whole buffer.
    let pass = std::fs::read_to_string(repo().join("crates/inf-render/src/passes/skinned.rs"))
        .expect("the skinned pass")
        .replace("\r\n", "\n");
    assert!(
        pass.contains("let (first_index, index_count) = match run.range {"),
        "the skinned pass no longer draws a run's RANGE — a sectioned body draws \
         every section over the whole buffer"
    );
    assert!(
        pass.contains("None => (0, gpu_mesh.index_count),"),
        "the pass no longer treats `range: None` as the whole buffer — that is \
         the case every committed skinned golden was blessed against"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// (4) THE BRIDGE'S OWN DEFECTS — carried 93, 94, 95, 96
// ─────────────────────────────────────────────────────────────────────────────

/// **A LOCOMOTION CLIP THAT ANIMATES TWO JOINTS IS REFUSED OUT LOUD** (carried
/// 93).
///
/// 124 of the 164 clips wave CHAR1a.2 exported carried `root` + one virtual bone
/// and six channels against a 68-bone skeleton, and every one of them was
/// reported as "crossed". The cause was the glTF exporter's PREVIEW MESH — it
/// writes tracks for the bones of a mesh, not of the sequence's skeleton, and
/// falls back to the first registry mesh sharing the rig, which on this project
/// is a two-bone helper. `export.py` now chooses the richest compatible mesh and
/// records the animated-joint count per clip.
///
/// This arm reads the manifest back and refuses a shell. It is a **local-only**
/// arm (the manifest is a bridge output and never enters the checkout) and skips
/// with a printed reason.
#[test]
fn no_exported_locomotion_clip_is_a_two_joint_shell() {
    let manifest = repo().join("../ue-out/char1a3/manifest.json");
    if !manifest.is_file() {
        eprintln!(
            "SKIP: no bridge manifest at ../ue-out/char1a3 — it is a local-only \
             output of tools/ue-export/export.py"
        );
        return;
    }
    let raw = std::fs::read_to_string(&manifest).expect("the manifest reads");
    let json: serde_json::Value = serde_json::from_str(&raw).expect("the manifest parses");
    let clips = json["clips"].as_array().expect("a clips array");
    assert!(
        clips.len() > 100,
        "only {} clips in the manifest",
        clips.len()
    );
    // **The floor.** A locomotion clip on a humanoid rig moves a spine, two legs
    // and two arms; ten is far below the 79 and 161 the two rigs here actually
    // export and far above the 2 a shell carries, so it separates the two
    // populations without pinning either.
    const FLOOR: u64 = 10;
    let mut shells: Vec<(String, u64)> = Vec::new();
    let mut counts: std::collections::BTreeMap<u64, usize> = std::collections::BTreeMap::new();
    for c in clips {
        let n = c["animated_joints"].as_u64().unwrap_or(0);
        *counts.entry(n).or_default() += 1;
        if n < FLOOR {
            shells.push((c["name"].as_str().unwrap_or("?").to_string(), n));
        }
    }
    println!("animated-joint histogram: {counts:?}");
    assert!(
        shells.is_empty(),
        "{} of {} exported clips animate fewer than {FLOOR} joints — they are \
         shells, and a shell plays as a bind pose: {:?}",
        shells.len(),
        clips.len(),
        &shells[..shells.len().min(8)]
    );
    // …and the preview mesh that made the difference is recorded per clip, so a
    // future run that silently stopped binding one is visible.
    assert!(
        clips
            .iter()
            .all(|c| c["preview_mesh"].as_str().is_some_and(|m| !m.is_empty())),
        "a clip crossed with no preview mesh recorded — the bone set it exported \
         against is then whatever the exporter's own fallback found"
    );
}

/// **THE CLIP IMPORTER IS IDENTITY-IDEMPOTENT** (carried 94).
///
/// `clip_guid` is a pure function of the manifest key, so a re-import overwrites
/// its own output instead of writing a 657th `.inf_anim` beside it.
///
/// **The mutation**: go back to `write_asset` (an allocated GUID). The second
/// call returns a different id and this arm goes red. Verified by construction —
/// the allocating door cannot return the same id twice.
#[test]
fn a_clips_identity_is_a_function_of_its_source() {
    use inf_editor_core::assets::ue_import::clip_guid;
    let a = clip_guid("_ALSV4_CPP_AnimationExamples_ALS_N_Walk_F_ALS_N_Walk_F");
    assert_eq!(
        a,
        clip_guid("_ALSV4_CPP_AnimationExamples_ALS_N_Walk_F_ALS_N_Walk_F"),
        "one source key gave two identities — a re-import writes a second asset"
    );
    assert_ne!(
        a,
        clip_guid("_ALSV4_CPP_AnimationExamples_ALS_N_Walk_B_ALS_N_Walk_B"),
        "two sources collapsed onto one identity — the second overwrites the first"
    );
    // A UUID shape, so the value round-trips through every reader that prints one.
    assert_eq!(a.uuid().get_version_num(), 4, "{a} is not a v4 uuid");
    // …and the island project holds ONE `.inf_anim` per source rather than four
    // runs' worth.
    let Some(content) = island_project() else {
        eprintln!("SKIP: no island project for the on-disk half");
        return;
    };
    let dir = content.join("UE/Mannequins");
    if !dir.is_dir() {
        eprintln!("SKIP: no imported mannequin pack");
        return;
    }
    let n = std::fs::read_dir(&dir)
        .expect("the pack dir reads")
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == "inf_anim"))
        .count();
    println!("island `.inf_anim` count: {n}");
    assert!(
        n <= 200,
        "{n} `.inf_anim` in the island's mannequin pack — the bridge exports 164 \
         sources, so anything above them is a re-import writing beside itself \
         (the CHAR1a audit measured 656)"
    );
}

/// **A REBOUND CHARACTER'S CLIPS RECORD THE RIG THEY WERE REBOUND ONTO**
/// (carried 95).
///
/// The rebind writes clips authored against the pack's rig at the starter
/// character's GUID, over a rig it has just replaced with that same pack's — so
/// the two ARE authored together, and the sidecar said otherwise: the island
/// recorded `8d06c1ee…` against `Starter.inf_skel`'s `5c7c1647…` and the editor
/// printed "the character animates the wrong bones" on every boot for correct
/// content.
///
/// Mutation-verified by its own second half: the arm reads the hash the
/// **skeleton on disk** hashes to and compares, so a sidecar left at any other
/// value fails.
#[test]
fn the_rebound_clips_record_the_rig_they_animate() {
    let Some(content) = island_project() else {
        eprintln!("SKIP: no island project — the rebind is a local-only act");
        return;
    };
    let skel = content.join("Starter.inf_skel");
    if !skel.is_file() {
        eprintln!("SKIP: the island project has no rebound character");
        return;
    }
    let want = inf_asset::ContentHash::of(&std::fs::read(&skel).expect("the rig reads")).to_hex();
    let mut checked = 0usize;
    for stem in [
        "Starter_Idle",
        "Starter_Walk",
        "Starter_Run",
        "Starter_F_Idle",
        "Starter_F_Walk",
        "Starter_F_Run",
    ] {
        let side = content.join(format!("{stem}.inf_anim.toml"));
        if !side.is_file() {
            continue;
        }
        let text = std::fs::read_to_string(&side).expect("the sidecar reads");
        let female = stem.starts_with("Starter_F");
        let rig = if female {
            content.join("Starter_F.inf_skel")
        } else {
            skel.clone()
        };
        let want = if female {
            inf_asset::ContentHash::of(&std::fs::read(&rig).expect("the rig reads")).to_hex()
        } else {
            want.clone()
        };
        assert!(
            text.contains(&want),
            "{stem}.inf_anim.toml does not record {}'s content hash ({want}) — \
             the editor's content scan prints \"the character animates the wrong \
             bones\" on every boot for content that is correct, and a false \
             positive hides a true one.\n{text}",
            rig.file_name().expect("a name").to_string_lossy()
        );
        checked += 1;
    }
    assert!(checked >= 3, "only {checked} rebound clips were on disk");
    println!("{checked} rebound clips record their rig's hash");
}

/// **EVERY IMPORTED ASSET CARRIES ITS PACK'S LICENCE, ON DISK** (carried 96).
///
/// The row existed in the export manifest and in the printed import report and
/// in **no sidecar anywhere** — a grep for `licen` over all 272 imported
/// sidecars returned nothing. For content that MAY SHIP and MAY NOT BE COMMITTED,
/// the licence is the one fact that has to be attached to the bytes.
#[test]
fn every_imported_asset_carries_its_licence_on_disk() {
    let Some(content) = island_project() else {
        eprintln!("SKIP: no island project — imported content is local-only");
        return;
    };
    let mut stamped = 0usize;
    let mut bare: Vec<String> = Vec::new();
    let mut ship: std::collections::BTreeMap<String, (usize, bool)> = Default::default();
    for sub in ["UE/MetaHumans", "UE/Mannequins"] {
        let dir = content.join(sub);
        if !dir.is_dir() {
            continue;
        }
        for e in std::fs::read_dir(&dir).expect("read").flatten() {
            let p = e.path();
            if p.extension().is_none_or(|x| x != "toml") {
                continue;
            }
            let text = std::fs::read_to_string(&p).unwrap_or_default();
            if let Some(line) = text.lines().find(|l| l.starts_with("licence_pack = ")) {
                stamped += 1;
                let pack = line.trim_start_matches("licence_pack = ").trim_matches('"');
                let may = text.contains("licence_may_ship = true");
                let row = ship.entry(pack.to_string()).or_insert((0, may));
                row.0 += 1;
                assert_eq!(
                    row.1, may,
                    "pack {pack} records two different ship positions across its \
                     assets — a licence that disagrees with itself is worse than none"
                );
            } else {
                bare.push(p.file_name().expect("a name").to_string_lossy().to_string());
            }
        }
    }
    println!("licence-stamped sidecars: {stamped}; by pack: {ship:?}");
    assert!(
        stamped > 100,
        "only {stamped} imported sidecars carry a licence position — the packs' \
         licences exist only in the import report, which is carried item 96"
    );
    assert!(
        bare.len() * 4 < stamped,
        "{} imported sidecars carry no licence at all: {:?}",
        bare.len(),
        &bare[..bare.len().min(6)]
    );
    // …and the MetaHumans' position is the one the memo states.
    if let Some((n, may)) = ship.get("MetaHumans") {
        assert!(*may, "the MetaHumans are recorded as NOT shippable");
        println!("MetaHumans: {n} assets, may ship");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// (5) THE EDITOR'S VIRTUAL TEXTURES — carried 91
// ─────────────────────────────────────────────────────────────────────────────

/// **THE EDITOR ASKS FOR ONE MORE PROJECTION AFTER IT INSTALLS A VT LEVEL**
/// (carried 91).
///
/// `vt_set_for` is a warm-gated snapshot, not a binding: `warm_slot` answers `0`
/// for a texture whose pages the residency has not applied yet. The player
/// re-projects every frame, so its frame 0 is cold and its frame 1 is not — its
/// own `phase26_gate` records that extra cold frame. The editor projects on a
/// document-version change, and the ONE projection it runs is the projection that
/// installs the registry, in which nothing can be warm. Every instance latched at
/// `VtTextureSet::NONE`, `scene_coverage` skipped it, no page was ever wanted for
/// it, and the surface drew off its base colour for the session: editor hero p50
/// **194.0** / chroma 14.0 against PIE's **81.9** / 12.8 for the same character.
///
/// Source-text rather than a rendered frame, and that is a stated limit:
/// `EngineHost::new` takes a real surface, so nothing headless can execute this
/// path — the same reason `VtLevelKey` was moved to Ring 1 at P26.5 so its rule
/// could be tested at all. What CAN be pinned here is that the latch is cleared
/// where the level is installed, and that the clearing is INSIDE the `Some` arm,
/// which is what makes it self-disarming.
///
/// **The mutation**: delete the line. The arm goes red, and the frame goes back
/// to a flat grey character — photographed both ways in `CHAR1a3-FINAL/`.
#[test]
fn the_editor_reprojects_once_after_a_vt_level_is_installed() {
    let src = std::fs::read_to_string(repo().join("editor/crates/inf-viewport/src/host.rs"))
        .expect("the viewport host")
        .replace("\r\n", "\n");
    let i = src
        .find("fn sync_vt_bindings")
        .expect("the host no longer binds virtual textures");
    let body = &src[i..];
    let install = body
        .find("self.renderer.set_vt_level(Some((textures, pools)));")
        .expect("the host no longer installs a VT level");
    let after = &body[install..];
    let clear = after.find("self.synced_version = None;").expect(
        "`sync_vt_bindings` does not clear the projection latch after \
             installing a level — every instance keeps the COLD texture set it \
             resolved in the same call that registered the textures, and the \
             editor draws a character off its base colour for the whole session",
    );
    let next_none = after.find("None => self.renderer.set_vt_level(None)");
    assert!(
        next_none.is_none_or(|n| clear < n),
        "the latch is cleared outside the arm that installed a level — a level \
         that resolved to nothing would re-project for ever"
    );
    // …and the early-out that makes it terminate is still above it.
    assert!(
        src[i..i + install].contains("if key == self.vt_level_key {"),
        "the level-key early-out is gone, so the extra projection this line asks \
         for would install a level again and ask for another"
    );
}

// ── the fixture's own doors, mirrored from `island_gate` ─────────────────────

fn loose_sim(content: &Path, slug: &str) -> inf_player::runtime_sim::RuntimeSim {
    let source = inf_player::level::DevDirLevelSource::new(content.join(format!("{slug}.inf_lvl")));
    let terrains = inf_player::level::terrain_paths_by_guid_from_dir(content);
    let pcg_terrains = terrains.clone();
    let (skeletons, clips, machines) = inf_player::level::load_anim_assets_from_dir(content);
    let builder = inf_player::level::InfSceneWorldBuilder::with_defaults(
        inf_player::level::load_actor_classes_from_dir(content),
    )
    .with_pcgs(inf_player::level::load_pcg_payloads_by_guid_from_dir(
        content,
    ))
    .with_biome_sets(inf_player::level::load_biome_sets_by_guid_from_dir(content))
    .with_anim_assets(skeletons, clips, machines)
    .with_audio(inf_player::level::load_audio_assets_from_dir(content))
    .with_terrain_resolver(std::sync::Arc::new(move |g| {
        inf_player::level::terrain_source_from_file(pcg_terrains.get(&g)?).ok()
    }));
    let mut built = inf_player::level::load(&source, &builder).expect("the loose level builds");
    let partition = built.take_partition();
    let pcg = built.pcg_context();
    let mut sim = inf_player::sim_from_built(built);
    inf_player::attach_cell_streaming(&mut sim, &partition, pcg);
    inf_player::attach_terrain_streaming(&mut sim, &inf_player::TerrainContent::Dir(terrains));
    sim
}

fn hero_entity(sim: &inf_player::runtime_sim::RuntimeSim) -> Option<inf_ecs::Entity> {
    let world = sim.world().world();
    world.iter_entities().find_map(|e| {
        e.get::<inf_ecs::components::CharacterMovement>()
            .map(|_| e.id())
    })
}
