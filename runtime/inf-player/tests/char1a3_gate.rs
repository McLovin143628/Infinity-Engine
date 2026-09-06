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
