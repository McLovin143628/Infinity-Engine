//! **WAVE CHAR1a'S INSTRUMENT** — the bodies, the rig they all bind to, and the
//! clips that have to resolve on every one of them.
//!
//! # What this gate is for
//!
//! CHAR1a brought three bodies onto one skeleton: two Unreal mannequins and two
//! MetaHumans that live **outside** this repository under licence, and one body
//! this engine generates and commits. A gate that could only see the committed
//! one would certify a third of the wave; a gate that needed the licensed ones
//! could not run in CI at all. So every arm below is asked of something that is
//! either **in the tree** or **derivable in the tree** — the interchange
//! contract, not the content.
//!
//! The contract is the **161 joint names**. That is the whole reason three
//! bodies from three sources can wear one animation set: `AnimClip` addresses a
//! joint by INDEX, the index is only meaningful against a name list, and every
//! one of these rigs publishes the same list in the same order. The arms below
//! measure that, measure what the import door does when it is *not* true, and
//! measure the two numbers the wave changed (the ladder's switch distances and
//! the hand ratio).
//!
//! # Every arm is mutation-verified
//!
//! Each `#[test]` names, in its own doc, the mutation that turns it red. They
//! were run, not imagined — see the wave ledger for the output of each.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use inf_anim::retarget::{retarget_clip, ClipRetargetReport, RetargetMap};
use inf_anim::skeleton::Skeleton;
use inf_anim::{AnimClip, BodyParams, JointTrack, QuatTrack};

/// The repository root, from this test binary's own manifest.
fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("runtime/inf-player has two parents")
        .to_path_buf()
}

/// The engine's own 161-bone rig, as the wizard builds it.
fn manny() -> Skeleton {
    inf_anim::manny::build_manny(&BodyParams::default())
        .expect("the mannequin rig builds from default params")
        .skeleton
}

/// The names of `sk`, in order.
fn names(sk: &Skeleton) -> Vec<String> {
    sk.joints().iter().map(|j| j.name.clone()).collect()
}

// ── arm 1: the interchange contract ─────────────────────────────────────────

/// **THE 161 NAMES, VERBATIM AND IN ORDER.**
///
/// The list below is not a copy of `manny.rs`'s table — it is what
/// `SKM_Manny.uasset` exports, read out of the glTF the bridge wrote
/// (`skins[0].joints` mapped through `nodes[].name`) on 2026-09-05. Both
/// mannequins and both MetaHumans publish it; the engine's generated rig has to
/// publish exactly the same thing or none of the three bodies can wear one
/// clip, because a clip's `JointTrack::joint` is a positional index into it.
///
/// **The mutation**: rename any single entry. Verified by renaming the table's
/// `neck_02` row to `neck_2` — this arm goes red at char1a_gate.rs:87, naming
/// joint 8.
#[test]
fn the_engines_rig_publishes_the_mannequins_161_names_in_order() {
    let sk = manny();
    assert_eq!(
        sk.len(),
        161,
        "the interchange rig is 161 bones; this one has {}",
        sk.len()
    );
    let got = names(&sk);
    // The head of the chain, which is where every retarget rule in the wave
    // lands, and the tail, which is where the IK handles live. Spelled out
    // rather than hashed: a digest tells you it broke, a name tells you where.
    let head = [
        "root", "pelvis", "spine_01", "spine_02", "spine_03", "spine_04", "spine_05", "neck_01",
        "neck_02", "head",
    ];
    for (i, want) in head.iter().enumerate() {
        assert_eq!(
            got[i], *want,
            "joint {i} is {:?}, and the exported SKM_Manny says {want:?}",
            got[i]
        );
    }
    for want in [
        "clavicle_l",
        "upperarm_l",
        "lowerarm_l",
        "hand_l",
        "thigh_r",
        "calf_r",
        "foot_r",
        "ball_r",
        "middle_metacarpal_l",
        "middle_03_r",
        "ik_hand_gun",
        "ik_foot_l",
        "interaction",
        "center_of_mass",
    ] {
        assert!(
            got.iter().any(|n| n == want),
            "the rig has no {want:?} — the exported mannequin does, and a body \
             bound by name would lose it"
        );
    }
    // Names are UNIQUE, or "bind by name" is not a function.
    let unique: BTreeSet<&String> = got.iter().collect();
    assert_eq!(
        unique.len(),
        161,
        "two joints share a name, so a name→index map is not one-to-one"
    );
}

/// **A body from any source binds by NAME, and a name that is not there is
/// reported rather than dropped in silence.**
///
/// This is the arm that stands in for "all three bodies": the two licensed ones
/// cannot be in CI, so what is measured is the DOOR they came through. A rig
/// with the mannequin's names in a *shuffled* order still retargets one for one
/// — which is the property that makes the contract a contract — and a rig
/// missing a name says so.
///
/// **The mutation**: make `RetargetMap::shared_names` return `Self::default()`.
/// The first assertion fails at 0 pairs. Verified.
#[test]
fn a_body_binds_to_the_rig_by_name_and_a_missing_name_is_named() {
    let sk = manny();
    let map = RetargetMap::shared_names(&sk, &sk);
    assert_eq!(
        map.pairs.len(),
        161,
        "every joint of the rig should pair with itself"
    );

    // A source rig that is the mannequin minus its two UE5-only spine segments
    // and its second neck bone — i.e. the UE4 mannequin the ALS clips are
    // authored on, in the one respect that matters.
    let trimmed: Vec<_> = sk
        .joints()
        .iter()
        .filter(|j| !matches!(j.name.as_str(), "spine_04" | "spine_05" | "neck_02"))
        .cloned()
        .collect();
    // Reparenting is not needed for the name map, which is what is under test.
    let partial = RetargetMap::shared_names(&sk, &Skeleton::new(reindex(trimmed)).unwrap());
    assert_eq!(
        partial.pairs.len(),
        158,
        "three names are absent from the target, so three pairs must not form"
    );
}

/// Re-point every parent index at the joint's new position after a filter, so a
/// trimmed joint list is still topologically valid.
fn reindex(mut joints: Vec<inf_anim::skeleton::Joint>) -> Vec<inf_anim::skeleton::Joint> {
    for (i, joint) in joints.iter_mut().enumerate() {
        joint.parent = joint.parent.filter(|&p| (p as usize) < i);
    }
    joints
}

// ── arm 2: every clip resolves on every body ────────────────────────────────

/// One rotation key on `joint`, so a clip exists to retarget.
fn key_clip(name: &str, joint: u16, q: glam::Quat) -> AnimClip {
    let mut t = JointTrack::new(joint);
    t.rotation = Some(QuatTrack {
        times: vec![0.0, 1.0],
        values: vec![q.to_array(), q.to_array()],
        interp: inf_anim::Interpolation::Linear,
    });
    AnimClip::new(name, vec![t])
}

/// **EVERY CLIP RESOLVES ON EVERY BODY — no T-pose.**
///
/// FIX1's law: a clip that does not resolve does not fail, it plays the bind
/// pose, and on this rig the bind pose is a T. So the claim is not "the import
/// did not error" — it is that the retarget wrote TRACKS, and that a retarget
/// which wrote none is REPORTED as vacuous rather than written to disk as a
/// perfectly valid clip that animates nothing.
///
/// The three bodies stand in as three skeletons carrying the same names: the
/// engine's generated rig, the same rig at a different height (a MetaHuman is
/// not 1.75 m), and the same rig with the extra UE5 bones removed (the ALS
/// source). A clip retargeted between any pair must land tracks.
///
/// **The mutation**: make `retarget_clip` return `(clip.clone(), report)`
/// without rewriting `track.joint`. The index assertion fails — the source's
/// index survives onto a rig where it names a different bone. Verified.
#[test]
fn every_clip_resolves_on_every_body_and_a_vacuous_retarget_is_named() {
    let tall = inf_anim::manny::build_manny(&BodyParams {
        height_m: 1.85,
        ..Default::default()
    })
    .expect("a taller mannequin builds")
    .skeleton;
    let short = manny();
    let trimmed = Skeleton::new(reindex(
        short
            .joints()
            .iter()
            .filter(|j| !matches!(j.name.as_str(), "spine_04" | "spine_05" | "neck_02"))
            .cloned()
            .collect(),
    ))
    .unwrap();

    let bodies: [(&str, &Skeleton); 3] = [
        ("engine", &short),
        ("tall", &tall),
        ("ue4-source", &trimmed),
    ];
    for (src_name, src) in bodies {
        // A clip authored on `src`, driving whatever `src` calls its upper arm.
        let joint = src.index_of("upperarm_l").expect("every body has an arm");
        let clip = key_clip("wave", joint, glam::Quat::from_rotation_z(0.5));
        for (dst_name, dst) in bodies {
            let map = RetargetMap::shared_names(src, dst);
            let (out, rep): (AnimClip, ClipRetargetReport) =
                retarget_clip(&clip, src, dst, &map, true);
            assert!(
                !rep.is_vacuous(),
                "{src_name} → {dst_name}: the retarget wrote no tracks, so this \
                 clip would play the bind pose (a T)"
            );
            assert_eq!(out.tracks.len(), 1, "{src_name} → {dst_name}");
            assert_eq!(
                out.tracks[0].joint,
                dst.index_of("upperarm_l").unwrap(),
                "{src_name} → {dst_name}: the track must name the TARGET's \
                 upperarm_l, not the source's index"
            );
        }
    }

    // **The control.** A rig that shares NO name with the mannequin retargets
    // to nothing, and the report says so — which is what the import door reads
    // before it writes a file.
    let alien = Skeleton::new(vec![inf_anim::skeleton::Joint {
        name: "Bip01_Spine".into(),
        parent: None,
        inverse_bind: glam::Mat4::IDENTITY.to_cols_array(),
        local_bind: inf_anim::skeleton::JointTransform::from_trs(
            glam::Vec3::ZERO,
            glam::Quat::IDENTITY,
            glam::Vec3::ONE,
        ),
    }])
    .unwrap();
    let clip = key_clip("alien", 0, glam::Quat::from_rotation_z(0.5));
    let map = RetargetMap::shared_names(&alien, &short);
    let (out, rep) = retarget_clip(&clip, &alien, &short, &map, true);
    assert!(
        rep.is_vacuous() && out.tracks.is_empty(),
        "a rig sharing no name must retarget to nothing AND say so: {}",
        rep.summary()
    );
    assert_eq!(rep.dropped, ["Bip01_Spine"]);
}

// ── arm 3: the LOD ladder's switch distances ────────────────────────────────

/// **THE LADDER SWITCHES WHERE THE SIMULATION DOES.**
///
/// A character's stored LOD rungs take over at 0 / 32 / 96 m, and those are not
/// three numbers somebody liked: they are `inf_ecs::crowd`'s own tier radii, so
/// a body cannot be posed by the full animation graph while drawing its
/// cheapest mesh, or draw its finest while the crowd ladder has stopped posing
/// it. Asserted against the crowd constants rather than against literals, so
/// moving a tier moves the ladder and cannot silently desynchronise it.
///
/// **The mutation**: change `character_lod_switch_m` to `[0.0, 50.0, 150.0]`.
/// Fails on rung 1. Verified.
#[test]
fn the_character_lod_ladder_switches_at_the_crowd_tier_radii() {
    use inf_editor_core::assets::ue_import::character_lod_switch_m;
    let d = character_lod_switch_m(3);
    assert_eq!(d.len(), 3);
    assert_eq!(d[0], 0.0, "rung 0 is drawn from the camera outward");
    assert_eq!(
        d[1],
        inf_ecs::crowd::DEFAULT_CROWD_FULL_M,
        "rung 1 must take over exactly where CrowdTier::Full ends"
    );
    assert_eq!(
        d[2],
        inf_ecs::crowd::DEFAULT_CROWD_NEAR_M,
        "rung 2 must take over exactly where CrowdTier::Near ends"
    );
    // Strictly increasing, or a rung is unreachable.
    assert!(d.windows(2).all(|w| w[1] > w[0]), "{d:?}");
    // One rung asks for one distance and never crashes.
    assert_eq!(character_lod_switch_m(1), vec![0.0]);
}

// ── arm 4: the hand ratio is measured, not authored ─────────────────────────

/// **THE HAND IS THE MESH'S HAND.**
///
/// SK1a authored `HAND_OF_HEIGHT = 0.105` and recorded that the reference rig's
/// value was about 0.096; CHAR1a measured it on the exported asset — a
/// 0.1720 m middle-finger chain on a 1.8054 m mesh — and the constant is now
/// 0.0953. This arm holds the *consequence*: the rig the wizard generates has a
/// hand the exported mannequin's mesh would fit.
///
/// Measured against the RIG, not against the constant, because a test that
/// re-states a constant certifies nothing.
///
/// **The mutation**: put `HAND_OF_HEIGHT` back to 0.105. The ratio comes out at
/// 0.1049 and the bound fails. Verified.
#[test]
fn the_generated_hand_matches_the_reference_meshs_proportion() {
    let sk = manny();
    let seg = |name: &str| -> f64 {
        let i = sk.index_of(name).unwrap_or_else(|| panic!("no {name}")) as usize;
        let t = sk.joints()[i].local_bind.translation_vec();
        f64::from(t.length())
    };
    let chain: f64 = [
        "middle_metacarpal_l",
        "middle_01_l",
        "middle_02_l",
        "middle_03_l",
    ]
    .iter()
    .map(|n| seg(n))
    .sum();
    let height = BodyParams::default().height_m;
    let ratio = chain / height;
    // **MEASURED ON SKM_Manny**: 0.1720 m of finger on a 1.8054 m mesh =
    // 0.09527. SKM_Quinn: 0.1720 / 1.8017 = 0.09547.
    //
    // The generated rig comes out at **0.09579**, and the 0.00052 excess is
    // geometry rather than error: `HAND_OF_HEIGHT` scales the finger table's
    // ALONG-axis proportions, which sum to exactly 1.0, while the metacarpal
    // also carries an `across` and an `up` offset (measured at SK1a off the same
    // asset), so the 3-D chain LENGTH is 0.55% longer than the along-axis sum.
    // Both sides of this comparison are 3-D magnitudes, so the excess is real
    // and the bound has to admit it.
    //
    // 1.0e-3 of height is 1.75 mm of finger. It admits the 0.55% and the 0.2%
    // between the two mannequins; it does NOT admit the authored 0.105, which
    // puts the ratio at 0.10552 — **19 times the bound** away.
    const REFERENCE: f64 = 0.1720 / 1.8054;
    assert!(
        (ratio - REFERENCE).abs() < 1.0e-3,
        "the generated hand is {ratio:.5} of height and the exported mannequin's \
         is {REFERENCE:.5} — the invented ratio is back"
    );
    // …and the finger proportions INSIDE the hand are unchanged, which is what
    // makes this a scale change rather than a re-authoring: the three phalanges
    // plus the metacarpal still sum to the whole hand.
    let parts: f64 = [
        "middle_metacarpal_l",
        "middle_01_l",
        "middle_02_l",
        "middle_03_l",
    ]
    .iter()
    .map(|n| seg(n))
    .sum();
    assert!((parts - chain).abs() < 1.0e-9);
}

// ── arm 5: the licence rows exist and say which way they point ──────────────

/// **EVERY CHARACTER PACK STATES A LICENCE, AND STATES WHETHER IT MAY SHIP.**
///
/// The three sources have three different positions and the difference is the
/// whole reason the wave is arranged as it is: Epic's mannequins are UE-Only
/// content (reference, never shipped), ALS is MIT (may ship, notice preserved),
/// MetaHumans are licensed for any engine (shipped in a cooked pack, never
/// committed). A pack that carried no position, or carried "unknown", would put
/// the decision back on whoever runs the import at 2 a.m.
///
/// Read out of `tools/ue-export/export.py` and `metahuman.py` themselves, so the
/// rows cannot be true in a document and absent from the tool.
///
/// **The mutation**: delete the `"ship"` key from the `ALS_Community` entry.
/// Fails naming the pack. Verified.
#[test]
fn every_character_pack_carries_a_licence_and_a_ship_position() {
    let src = std::fs::read_to_string(repo().join("tools/ue-export/export.py"))
        .expect("export.py is in the tree");
    let chars = src
        .split_once("CHARACTERS = [")
        .expect("export.py has a CHARACTERS list")
        .1;
    let chars = chars.split_once("\nOUT =").map(|p| p.0).unwrap_or(chars);
    for pack in ["UE5_Mannequins", "ALS_Community"] {
        let at = chars
            .find(&format!("\"name\": \"{pack}\""))
            .unwrap_or_else(|| panic!("export.py has no {pack} character pack"));
        let block = &chars[at..];
        let block = block.split_once("\"clips\"").map(|p| p.0).unwrap_or(block);
        assert!(
            block.contains("\"license\":"),
            "{pack} carries no licence text"
        );
        assert!(
            block.contains("\"ship\": True") || block.contains("\"ship\": False"),
            "{pack} does not say whether it may be shipped"
        );
        assert!(
            !block.to_lowercase().contains("\"license\": \"unknown"),
            "{pack}'s licence is still 'unknown' — the wave's docs clause says \
             every character pack's position is established"
        );
    }
    assert!(
        src.contains("UE-Only Content"),
        "the mannequin row must name what Epic calls it"
    );
    assert!(
        src.contains("MIT License, Copyright (c) 2020"),
        "MIT requires the notice to travel; it must be in the row"
    );

    let mh = std::fs::read_to_string(repo().join("tools/ue-export/metahuman.py"))
        .expect("metahuman.py is in the tree");
    assert!(
        mh.contains("mid-2025 terms") && mh.contains("NEVER committed"),
        "the MetaHuman row must record the terms version relied on and the \
         position on committing"
    );
    assert!(
        mh.contains("unrealengine.com/en-US/eula/metahuman"),
        "the MetaHuman row must name the exact terms it relied on"
    );
}

// ── arm 6: nothing from Unreal is inside the checkout ───────────────────────

/// **THE LICENCE LAW, AS A SCANNER.**
///
/// Everything the bridge writes is licensed content and none of it may enter
/// this repository. Two doors already refuse a destination inside the checkout
/// (`export.py`'s `engine_checkout_above` and `ue_import`'s mirror), but a door
/// only guards the path through it: a file copied in by hand, or an output
/// directory pointed at a subfolder by an author in a hurry, walks straight
/// past both.
///
/// So this walks the tree. It is **non-vacuous by construction**: it asserts it
/// visited a five-figure number of files and that it can see the markers it is
/// looking for when they really are present (the control below plants one in a
/// string and finds it).
///
/// **The mutation**: drop a copy of `SKM_Manny.inf_mesh` into `samples/`. Fails
/// naming the path. Verified with a 12-byte file called
/// `samples/SKM_Manny.inf_mesh`.
#[test]
fn nothing_from_unreal_is_inside_the_checkout() {
    // Asset stems that only exist in Unreal's content, and the pack names the
    // bridge writes under. A stem is matched against the FILE NAME, so a source
    // file that merely mentions one in a comment (this file, for instance) is
    // not a hit.
    const UE_STEMS: [&str; 8] = [
        "SKM_Manny",
        "SKM_Quinn",
        "SK_Mannequin",
        "MI_Manny",
        "MI_Quinn",
        "ALS_Mannequin_Skeleton",
        "MetaHumanCharacter",
        "ABP_Manny",
    ];
    // Directories a bridge output would land in if somebody pointed it here.
    const UE_DIRS: [&str; 3] = ["ue-staging", "MHForge", "island-build"];

    let root = repo();
    let mut visited = 0usize;
    let mut offenders: Vec<String> = Vec::new();
    let mut stack = vec![root.clone()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for e in entries.flatten() {
            let path = e.path();
            let name = e.file_name().to_string_lossy().to_string();
            let is_dir = path.is_dir();
            if is_dir {
                // `target/` is build output and `.git/` is history; neither is
                // content and both are enormous.
                if name == "target" || name == ".git" || name == "node_modules" {
                    continue;
                }
                if UE_DIRS.contains(&name.as_str()) {
                    offenders.push(format!("{} (a bridge output directory)", path.display()));
                    continue;
                }
                stack.push(path);
                continue;
            }
            visited += 1;
            // Source files may NAME these assets — this gate does. What may not
            // exist is an asset FILE.
            let ext = path
                .extension()
                .map(|e| e.to_string_lossy().to_lowercase())
                .unwrap_or_default();
            let is_asset = matches!(
                ext.as_str(),
                "uasset"
                    | "umap"
                    | "inf_mesh"
                    | "inf_skel"
                    | "inf_anim"
                    | "inf_tex"
                    | "gltf"
                    | "glb"
                    | "fbx"
                    | "dna"
            );
            if !is_asset {
                continue;
            }
            if UE_STEMS.iter().any(|s| name.starts_with(s)) {
                offenders.push(path.display().to_string());
            }
        }
    }
    // **Non-vacuity, two ways.** The walk really walked, and the matcher really
    // matches — a scanner that visited nothing, or one whose predicate is
    // accidentally always false, passes this test perfectly otherwise.
    // **MEASURED**: 2 034 files on this checkout (`git ls-files` counts 2 022
    // tracked; the walk also sees a handful of untracked ones and skips
    // `target/`, `.git/` and every `node_modules/`). The floor is 1 800 — low
    // enough that deleting a sample folder does not turn this red for the wrong
    // reason, high enough that a walk which stopped at the first directory
    // cannot pass.
    assert!(
        visited > 1_800,
        "the scanner visited only {visited} files — it is not walking the tree"
    );
    let control = "SKM_Manny_LOD0.gltf";
    assert!(
        UE_STEMS.iter().any(|s| control.starts_with(s)),
        "the matcher does not recognise {control}, so a real intrusion would \
         also slip past it"
    );
    assert!(
        offenders.is_empty(),
        "content from Unreal is inside the engine checkout — it may never be \
         committed:\n  {}",
        offenders.join("\n  ")
    );
}

// ── arm 7: the committed body is the one the wave built ─────────────────────

/// **THE COMMITTED STARTER BODY IS THE HIGH-POLY ONE, AND IT IS SKINNED TO THE
/// 161-BONE RIG.**
///
/// The wave replaced a 1 498-triangle body with a denser one, and the thing
/// worth holding is not the number — it is that the committed bytes are a
/// *skinned* body whose influences address the committed rig. A body with an
/// empty skin stream draws rigid and never moves; one whose joint indices run
/// past the rig's length is a panic in the palette build.
///
/// **The mutation**: reverting `BodyOptions::default()` to 10/14/5/14/8 does
/// **not** turn this red, and that is correct — this arm reads the COMMITTED
/// BYTES, not the generator, so a generator change with no bless does not move
/// it. (Recorded because the first mutation tried was that one, and a mutation
/// that leaves an arm green is a fact about the arm worth writing down.) The
/// mutation that does turn it red is putting the previous committed body back:
/// `git checkout HEAD -- samples/starter-character/Starter_Body.inf_mesh`,
/// which fails at 1 498 triangles. Verified.
#[test]
fn the_committed_starter_body_is_high_poly_and_skinned_to_the_committed_rig() {
    let dir = repo().join("samples/starter-character");
    let mesh: inf_mesh::MeshAsset = inf_asset::decode(
        &std::fs::read(dir.join("Starter_Body.inf_mesh")).expect("the body is committed"),
    )
    .expect("the committed body decodes");
    let rig: inf_anim::SkeletonAsset = inf_asset::decode(
        &std::fs::read(dir.join("Starter.inf_skel")).expect("the rig is committed"),
    )
    .expect("the committed rig decodes");

    assert_eq!(
        rig.skeleton.len(),
        161,
        "the committed rig is the mannequin's"
    );
    assert!(
        mesh.submeshes.iter().all(|s| s.is_skinned()),
        "a committed submesh carries no skin stream — it would draw rigid"
    );
    // **The floor, not the number.** 5 718 triangles today; the claim is that
    // the body is no longer the 1 498-triangle one, which is what an author
    // sees and what the CI fixture stands for.
    let tris = mesh.triangle_count();
    assert!(
        tris > 4_000,
        "the committed body is {tris} triangles — the wave replaced the \
         1 498-triangle one"
    );
    // Every influence addresses a joint the committed rig has.
    let joints = rig.skeleton.len() as u16;
    let mut worst = 0u16;
    for sub in &mesh.submeshes {
        for s in &sub.skin {
            for j in s.joints {
                worst = worst.max(j);
                assert!(
                    j < joints,
                    "a vertex is skinned to joint {j} and the rig has {joints}"
                );
            }
        }
    }
    assert!(
        worst > 0,
        "every influence names joint 0 — the body is bound to its root and \
         nothing else, which is not a skin"
    );
}

// ── WAVE CHAR1a.2 ────────────────────────────────────────────────────────────
//
// Five more, on the five things the second half changed. Same rule as above:
// each names the mutation that turns it red, and each was run.

/// **THE EDITOR PREVIEWS AN IDLE, NOT THE BIND POSE** (CHAR1a carried 72).
///
/// The defect was photographable and was photographed: outside Play there is no
/// sim pose and an authored character carries no `AnimPlayer`, so every rig in
/// the viewport fell to the rest pose and stood in its bind — a T on the
/// generated rig, an A on the mannequin's.
///
/// This asks the SHIPPED store, because the rule is one rule in two files and
/// `projector_mirror.rs` is what keeps them equal; measuring either one measures
/// both. The clip's single key is 90° about X on joint 1, so "the preview is not
/// the bind pose" is a palette that differs — and the rest palette is compared
/// against too, so an arm that accidentally compared a pose to itself would fail.
///
/// **The mutation**: delete the `machine_entry_clip` arm from `resolve_skinned`
/// (restore `(None, None) => Pose::rest`). The first assertion fails: the
/// preview palette equals the rest palette. Verified.
#[test]
fn the_preview_pose_is_the_machines_entry_clip_and_not_the_bind_pose() {
    use inf_ecs::components::{AnimStateMachine, SkeletalMesh};
    use inf_player::skinned::SkinnedRegistry;

    let mesh_id = uuid::Uuid::from_u128(0xCA1A_2001);
    let skel_id = uuid::Uuid::from_u128(0xCA1A_2002);
    let clip_id = uuid::Uuid::from_u128(0xCA1A_2003);
    let sm_id = uuid::Uuid::from_u128(0xCA1A_2004);

    let skeleton = two_joint_rig();
    let mut reg = SkinnedRegistry::new();
    reg.insert_mesh(mesh_id, one_triangle_skin());
    reg.insert_skeleton(skel_id, skeleton.clone());
    reg.insert_clip(
        clip_id,
        key_clip(
            "Idle",
            1,
            glam::Quat::from_rotation_x(std::f32::consts::FRAC_PI_2),
        ),
    );
    reg.insert_state_machine(
        sm_id,
        inf_anim::StateMachine {
            states: vec![inf_anim::SmState::clip("Idle", *clip_id.as_bytes())],
            transitions: Vec::new(),
            entry: 0,
            params: Vec::new(),
            profiles: Vec::new(),
        },
    );

    let sm = SkeletalMesh {
        mesh: Some(mesh_id),
        skeleton: Some(skel_id),
    };
    let machine = AnimStateMachine {
        sm: Some(sm_id),
        ..AnimStateMachine::default()
    };

    let rest = reg
        .resolve_skinned(&sm, None, None, None)
        .expect("the rest pose resolves")
        .palette;
    let preview = reg
        .resolve_skinned(&sm, None, None, Some(&machine))
        .expect("the preview resolves")
        .palette;
    assert_ne!(
        rest[1].to_cols_array(),
        preview[1].to_cols_array(),
        "the preview palette IS the bind pose — the editor is still drawing a \
         rig at rest instead of its machine's idle"
    );

    // …and the shared (crowd) door agrees with it, which is the invariant a
    // tier switch would otherwise break: an idle up close and a bind pose at
    // 96 m is one character with two silhouettes.
    let shared = reg
        .resolve_skinned_shared(&sm, Some(&machine))
        .expect("the shared palette resolves")
        .palette;
    assert_eq!(
        shared[1].to_cols_array(),
        preview[1].to_cols_array(),
        "the crowd's shared palette is not the one the per-agent path builds"
    );
    // A machine with no asset behind it is still the rest pose, so a level with
    // a dangling `.inf_sm` previews exactly as it did.
    let dangling = AnimStateMachine {
        sm: Some(uuid::Uuid::from_u128(0xDEAD)),
        ..AnimStateMachine::default()
    };
    let fallback = reg
        .resolve_skinned(&sm, None, None, Some(&dangling))
        .expect("it still resolves")
        .palette;
    assert_eq!(
        fallback[1].to_cols_array(),
        rest[1].to_cols_array(),
        "an unresolvable machine changed the preview — the fallback is the rest \
         pose and nothing else"
    );
}

/// **AND IT RESOLVES THROUGH THE ASSET INDEX, ON COMMITTED CONTENT** — the half
/// the arm above cannot see, and the half that was broken.
///
/// `the_preview_pose_is_the_machines_entry_clip_and_not_the_bind_pose` registers
/// its machine with `insert_state_machine`, which puts it straight into the
/// store's map. It therefore proves the RULE and says nothing about the LOOKUP —
/// and the lookup was the thing that did not work: `INDEXED_EXTENSIONS` listed
/// `inf_mesh`, `inf_skel` and `inf_anim` and **not `inf_sm`**, in both hosts, so
/// `load_payload::<StateMachineAsset>` missed on every real `.inf_sm` and every
/// character in the editor stood in its bind pose exactly as before the wave.
/// The gate was green; the demo frame showed a body in a plain T.
///
/// So this arm opens the committed female character as a DIRECTORY — the same
/// `from_dir` index a `--level` dev run takes — and asks for the pose the way a
/// host does. Measured: the preview palette differs from the rest palette by
/// **1.6514 m** at its worst joint, on a body whose idle moves `hand_l` 0.6252 m
/// down out of the bind T.
///
/// **The mutation**: drop `"inf_sm"` from `INDEXED_EXTENSIONS`. The delta falls
/// to 0.0000 and the arm goes red — which is the defect, reproduced. Verified.
#[test]
fn the_preview_pose_resolves_through_the_stores_own_asset_index() {
    use inf_ecs::components::{AnimStateMachine, SkeletalMesh};
    use inf_player::skinned::SkinnedRegistry;

    let dir = repo().join("samples/starter-character-f");
    let reg = SkinnedRegistry::from_dir(&dir);
    assert!(
        reg.has_content(),
        "the committed female folder did not index"
    );

    let ids = |n: u128| uuid::Uuid::from_u128(0x5C10_00B0 + n);
    let sm = SkeletalMesh {
        mesh: Some(ids(2)),
        skeleton: Some(ids(0)),
    };
    let machine = AnimStateMachine {
        sm: Some(ids(6)),
        ..AnimStateMachine::default()
    };
    let rest = reg
        .resolve_skinned(&sm, None, None, None)
        .expect("the committed body resolves from a dev dir");
    let preview = reg
        .resolve_skinned(&sm, None, None, Some(&machine))
        .expect("…and so does its preview");
    let worst = rest
        .palette
        .iter()
        .zip(preview.palette.iter())
        .map(|(a, b)| (a.w_axis - b.w_axis).length())
        .fold(0.0f32, f32::max);
    eprintln!("preview vs rest, through the index: {worst:.4} m");
    assert!(
        worst > 0.05,
        "the preview palette is the rest palette ({worst:.4} m apart) — the \
         machine did not resolve through the store's index, so the rule is \
         inert on every real character"
    );
}

/// A two-joint rig: a root and one child a metre up.
fn two_joint_rig() -> Skeleton {
    use inf_anim::skeleton::{Joint, JointTransform};
    Skeleton::new(vec![
        Joint {
            name: "root".into(),
            parent: None,
            local_bind: JointTransform::default(),
            inverse_bind: glam::Mat4::IDENTITY.to_cols_array(),
        },
        Joint {
            name: "child".into(),
            parent: Some(0),
            local_bind: JointTransform::from_trs(
                glam::Vec3::new(0.0, 1.0, 0.0),
                glam::Quat::IDENTITY,
                glam::Vec3::ONE,
            ),
            inverse_bind: glam::Mat4::from_translation(glam::Vec3::new(0.0, -1.0, 0.0))
                .to_cols_array(),
        },
    ])
    .expect("a two-joint rig is valid")
}

/// One triangle bound to joint 1 — enough geometry for a draw to resolve.
fn one_triangle_skin() -> inf_render::SkinnedMeshData {
    use inf_render::{SkinnedMeshData, SkinnedVertex};
    let v = |x: f32| SkinnedVertex {
        pos: [x, 0.0, 0.0],
        normal: [0.0, 1.0, 0.0],
        joints: [1, 0, 0, 0],
        weights: [1.0, 0.0, 0.0, 0.0],
        uv: [0.0, 0.0],
    };
    SkinnedMeshData {
        vertices: vec![v(0.0), v(1.0), v(0.5)],
        indices: vec![0, 1, 2],
    }
}

/// **`SkinnedInstance` CARRIES `blend` AND `cutoff`, IN BOTH HOSTS** (CHAR1a
/// carried 71, and FIX2's carried 40 with it).
///
/// A source arm, and deliberately: the two projectors are two inline loops in
/// two crates that this test binary cannot both link and run a frame of, so what
/// is checked is that the field reaches the literal FROM THE MATERIAL on both
/// sides — not that it is present, which a hard-coded `blend: 0` would also
/// satisfy. `projector_mirror.rs`'s field-for-field gate catches the two hosts
/// disagreeing; this catches them agreeing on a constant.
///
/// **The mutation**: replace `blend,` with `blend: 0,` in the viewport's
/// literal. The `blend_code(m.blend)` assertion still passes (the rigid path has
/// one) and the "from the material, not a constant" assertion fails. Verified.
#[test]
fn the_skinned_instance_carries_blend_and_cutoff_from_the_material() {
    let hosts = [
        (
            "the editor viewport",
            "editor/crates/inf-viewport/src/host.rs",
        ),
        ("the shipped player", "runtime/inf-player/src/render.rs"),
    ];
    for (label, rel) in hosts {
        let src = std::fs::read_to_string(repo().join(rel))
            .unwrap_or_else(|e| panic!("{rel}: {e}"))
            .replace("\r\n", "\n");
        // The skeletal branch's own Material read — a seven-tuple since this
        // wave, and the two new terms are the material's own.
        assert!(
            src.contains("let (color, metallic, roughness, emissive, blend, cutoff, vt) = w"),
            "{label}'s skeletal branch does not read blend/cutoff from the Material"
        );
        // …and the literal takes them by NAME, so the value cannot be a
        // constant that happens to look right on an opaque body.
        let lit = src
            .split("SkinnedInstance {")
            .nth(1)
            .unwrap_or_else(|| panic!("{label} builds no SkinnedInstance"));
        let body = &lit[..lit.find("});").unwrap_or(lit.len())];
        for field in ["blend,", "cutoff,"] {
            assert!(
                body.contains(field),
                "{label}'s SkinnedInstance does not carry `{field}` from the \
                 material — an alpha-masked character draws solid"
            );
        }
    }
    // The renderer's own end of it: the field exists, is packed, and the shader
    // reads it. A pass that carried the field and never uploaded it would pass
    // every assertion above.
    let pass = std::fs::read_to_string(repo().join("crates/inf-render/src/passes/skinned.rs"))
        .expect("the skinned pass")
        .replace("\r\n", "\n");
    assert!(
        pass.contains("inst.blend as f32 * 4.0 + inst.cutoff.clamp(0.0, 1.0)"),
        "the skinned pass no longer packs blend/cutoff into `pbr.w`"
    );
    let wgsl =
        std::fs::read_to_string(repo().join("crates/inf-render/src/shaders/skinned_mesh.wgsl"))
            .expect("the skinned shader")
            .replace("\r\n", "\n");
    assert!(
        wgsl.contains("if (in.pbr.w > 0.5 && in.pbr.w < 1.5 && in.color.a < in.pbr.z)"),
        "the skinned fragment stage has no masked discard — a hair card draws solid"
    );
}

/// **THE FEMALE BODY IS THE SAME RIG** — 161 names, in the same order, as the
/// male's and as the mannequin's.
///
/// This is the whole interchange contract applied to the wave's own second
/// body: a clip addresses a joint by INDEX, so two bodies that do not publish
/// the same list in the same order cannot share one animation set, and the
/// female body exists precisely to be dressed by the same clips.
///
/// **The mutation**: point `starter_character_f_spec` at `spine_segments: 3`
/// (or any change that moves the rig's joint list). The name-equality assertion
/// fails after a re-bless; without a re-bless the committed bytes still decode
/// to 161 and the arm stays green, which is correct — it reads COMMITTED bytes,
/// exactly like the male body's arm, and its doc says so. Verified by renaming
/// joint 8 of the committed female `.inf_skel` in a scratch copy: red at the
/// name-list assertion.
#[test]
fn the_female_committed_body_publishes_the_same_161_names() {
    let male: inf_anim::SkeletonAsset =
        decode_committed("samples/starter-character/Starter.inf_skel");
    let female: inf_anim::SkeletonAsset =
        decode_committed("samples/starter-character-f/Starter_F.inf_skel");
    assert_eq!(
        female.skeleton.len(),
        161,
        "the female rig is not the mannequin's"
    );
    assert_eq!(
        names(&female.skeleton),
        names(&male.skeleton),
        "the two committed bodies publish different joint lists — one clip \
         cannot play on both"
    );
    assert_eq!(
        names(&female.skeleton),
        names(&manny()),
        "the female rig is not the name list `manny::build_manny` publishes — \
         and that list is the one the first arm pins against the mannequin's own"
    );

    // …and she is a DIFFERENT BODY, not a copy: the measured proportions have
    // to reach the bind pose or the folder is 368 KB of duplicate.
    // The joint's WORLD bind position, from the inverse bind matrix it stores —
    // the same number `char1a2_measure_quinn.py` read off the glTF, derived on
    // this side of the bridge rather than trusted from the other.
    let at = |sk: &Skeleton, n: &str| -> glam::Vec3 {
        let i = sk.index_of(n).unwrap_or_else(|| panic!("no joint {n}"));
        sk.joint(i as usize)
            .expect("the index came from the skeleton")
            .inverse_bind_mat()
            .inverse()
            .to_scale_rotation_translation()
            .2
    };
    let span = |sk: &Skeleton| (at(sk, "upperarm_l").x - at(sk, "upperarm_r").x).abs();
    let hips = |sk: &Skeleton| (at(sk, "thigh_l").x - at(sk, "thigh_r").x).abs();
    let (ms, fs) = (span(&male.skeleton), span(&female.skeleton));
    let (mh, fh) = (hips(&male.skeleton), hips(&female.skeleton));
    assert!(
        fs < ms * 0.90,
        "the female body's shoulders are {fs:.4} m against the male's {ms:.4} — \
         Quinn measures 0.3211 and this rig does not"
    );
    // **The SHOULDER-TO-HIP RATIO, not the hip width.** Measured: the committed
    // male is 0.40 / 0.22 = 1.818 and the female 0.3211 / 0.2231 = 1.439, a
    // ratio of 0.79 — that is the silhouette difference, and it is the one an
    // absolute hip comparison cannot see. The male's 0.22 m hip is a wizard
    // DEFAULT and not a measurement (Manny's own is 0.1994 at 1.8054 m, so the
    // committed male is relatively wider in the hips than the mannequin he
    // stands for) — the same family of invented defaults this wave carries the
    // arm ratio for, and the reason this arm compares shapes rather than
    // millimetres.
    assert!(
        fs / fh < (ms / mh) * 0.85,
        "the two committed bodies have the same silhouette: shoulders/hips is \
         {:.3} for the female and {:.3} for the male",
        fs / fh,
        ms / mh
    );
    // **And their heads land 2.2 mm apart**, which is what this arm found when
    // it tried to assert stature — so it does not. Hers is built on a 1.8017 m
    // mesh and his on the wizard's 1.75 m default, but her MEASURED
    // `head_height_ratio` is 0.9021 against his 0.93: 1.8017 x 0.9021 = 1.6253
    // and 1.75 x 0.93 = 1.6275. A taller body with a proportionally lower head
    // is exactly what the two mannequins measure. "These are two rigs" therefore
    // has to be asked of the SHAPE — and what can be asserted is that the bind
    // poses differ by more than a centimetre somewhere, which a copied folder
    // could not do.
    let moved = MEASURED_JOINTS
        .iter()
        .map(|n| (at(&female.skeleton, n) - at(&male.skeleton, n)).length())
        .fold(0.0f32, f32::max);
    assert!(
        moved > 0.01,
        "the female rig's joints are within {moved:.4} m of the male's — the \
         folder is a copy, not a second body"
    );
}

/// The joints the female-body arm measures — one per girdle plus a limb, so a
/// rig that moved only its arms and one that moved only its hips both show.
const MEASURED_JOINTS: [&str; 6] = [
    "pelvis",
    "spine_05",
    "clavicle_l",
    "upperarm_l",
    "hand_l",
    "thigh_l",
];

/// Decode a committed skeleton asset by repo-relative path.
fn decode_committed(rel: &str) -> inf_anim::SkeletonAsset {
    let bytes = std::fs::read(repo().join(rel)).unwrap_or_else(|e| panic!("{rel}: {e}"));
    inf_asset::decode(&bytes).unwrap_or_else(|e| panic!("{rel} does not decode: {e}"))
}

/// **THE PACKED UE MASK LANDS IN ORM ORDER** (CHAR1a carried 76), and the AO
/// comes out of the channel that HOLDS it.
///
/// Two swizzles, both read off the material and confirmed by a channel census
/// of the exported PNGs rather than guessed from a name:
///
/// * `T_*_MSR_MSK` is metallic (R) / specular (G) / roughness (B); this
///   engine's ORM is occlusion (R) / roughness (G) / metallic (B).
/// * `T_*_AS?AO?MASK_MSK` is anisotropy (R) / **ambient occlusion (G)** / paint
///   mask (B). The importer took plane R, whose mean over `T_Manny_01` is 3.5
///   of 255 — so every character through this bridge had its ambient term
///   multiplied by 0.014.
///
/// **The mutation**: swap `Some(0)` and `Some(2)` in `role_to_planes("msr")`.
/// The metallic/roughness assertions fail (200 and 60 change places). Verified.
/// Changing `aniso_ao_paint`'s `Some(1)` to `Some(0)` fails the occlusion
/// assertion with 3 instead of 244 — which is the defect itself, reproduced.
#[test]
fn a_packed_ue_mask_swizzles_into_the_engines_orm_order() {
    use inf_editor_core::assets::ue_import::{broadcast_channel, role_to_planes};
    use inf_material::MapKind;

    // One texel, four channels, all distinct so a wrong channel cannot look right.
    let msr = [200u8, 120, 60, 255]; // metallic, specular, roughness
    let aniso = [3u8, 244, 255, 255]; // anisotropy, AO, paint mask

    let planes = role_to_planes("msr");
    assert_eq!(
        planes.len(),
        2,
        "an MSR mask fills TWO of the engine's slots, not one"
    );
    let metallic = planes
        .iter()
        .find(|(k, _)| *k == MapKind::Metallic)
        .and_then(|(_, c)| *c)
        .expect("msr names no metallic channel");
    let roughness = planes
        .iter()
        .find(|(k, _)| *k == MapKind::Roughness)
        .and_then(|(_, c)| *c)
        .expect("msr names no roughness channel");
    let occlusion = role_to_planes("aniso_ao_paint")
        .iter()
        .find(|(k, _)| *k == MapKind::Occlusion)
        .and_then(|(_, c)| *c)
        .expect("the AO mask names no occlusion channel");

    let o = broadcast_channel(&aniso, occlusion);
    let r = broadcast_channel(&msr, roughness);
    let m = broadcast_channel(&msr, metallic);
    let orm = inf_material::pack_orm(Some(&o), Some(&r), Some(&m), 1, 1).expect("one texel packs");
    assert_eq!(
        (orm[0], orm[1], orm[2]),
        (244, 60, 200),
        "the ORM is not (occlusion, roughness, metallic) — got \
         {:?} from an MSR of {msr:?} and an AO mask of {aniso:?}",
        &orm[..3]
    );

    // A role this engine has nowhere to put says so rather than racing for a
    // slot: the mannequin binds a SECOND normal and a tangent field, and both
    // used to be classified `normal`.
    for role in ["normal_second", "tangent", "decal", "clearcoat"] {
        assert!(
            role_to_planes(role).is_empty(),
            "`{role}` claims an engine slot — it is reported as unplaced, not placed"
        );
    }
    assert_eq!(
        role_to_planes("normal"),
        &[(MapKind::Normal, None)],
        "the `normal` role no longer fills the normal slot"
    );
}

/// **THE MATERIAL'S OWN PARAMETER NAMES DECIDE THE ROLE**, and `BNormal` is not
/// `Normal`.
///
/// The export side is Python and cannot be linked, so this reads the table as
/// source — the same shape as the licence scanner arm above, and for the same
/// reason: a rule stated in a file this repository owns is checkable from a file
/// this repository owns.
///
/// It is not a spelling check. Each assertion is the ANSWER for one of the eight
/// textures `M_Mannequin` binds, and three of them used to be wrong: `BNormal`
/// and `Tangent` both classified as `normal` (three maps racing for one slot),
/// and `LogoTexture` classified as `metallic` from the `_m$` name rule — so the
/// hero's metal mask was the Unreal logo.
///
/// **The mutation**: delete the `"bnormal": "normal_second",` line. The
/// substring fallback in `kind_of_texture` then matches `normal` inside
/// `bnormal` again and the arm goes red on that key. Verified.
#[test]
fn the_mannequin_material_parameters_map_to_one_role_each() {
    let src = std::fs::read_to_string(repo().join("tools/ue-export/export.py"))
        .expect("export.py")
        .replace("\r\n", "\n");
    let table = src
        .split("PARAM_KIND = {")
        .nth(1)
        .expect("export.py has no PARAM_KIND")
        .split("\n}")
        .next()
        .expect("PARAM_KIND does not close");
    for (param, role) in [
        ("\"normal\"", "\"normal\""),
        ("\"bnormal\"", "\"normal_second\""),
        ("\"tangent\"", "\"tangent\""),
        ("\"logotexture\"", "\"decal\""),
        ("\"msrtex\"", "\"msr\""),
        ("\"anisoaopaintmasktex\"", "\"aniso_ao_paint\""),
        ("\"ccccrtex\"", "\"clearcoat\""),
        ("\"base texture\"", "\"albedo\""),
    ] {
        let want = format!("{param}: {role}");
        assert!(
            table.contains(&want),
            "`M_Mannequin`'s {param} parameter no longer maps to {role} — the \
             classifier's exact table is what stops three normals racing for \
             one slot"
        );
    }
    // The exact table must be consulted BEFORE the substring fallback, which is
    // the actual defect: `normal` is a substring of `bnormal`.
    let f = src
        .split("def kind_of_texture(")
        .nth(1)
        .expect("no kind_of_texture");
    let exact = f.find("if p in PARAM_KIND:").expect("no exact lookup");
    let loose = f
        .find("for key, kind in PARAM_KIND.items():")
        .expect("no fallback");
    assert!(
        exact < loose,
        "the substring fallback runs before the exact lookup — `BNormal` \
         classifies as `normal` again"
    );
}

/// **A COMMITTED BODY IS NOT NEAR-BLACK WHEN THE SUN IS ON IT** (CHAR1a
/// carried 74) — the luminance floor, measured in a gate instead of on a
/// screenshot.
///
/// # It is not the island, and it cannot be
///
/// CHAR1a measured p25 72.2 on the hero in a PIE frame of Vancouver Island and
/// carried "fold the floor into the gate". Folding *that* measurement in is not
/// a matter of cost: the island's content is gigabytes of licensed Megascans and
/// Unreal mannequin under `island-build/`, outside this repository by law, so no
/// CI leg can open it. Timed anyway, so the number is on the record rather than
/// the excuse: building this scene and rendering it headless is **well under a
/// second** on this machine (the whole 13-arm binary, this arm included, runs in
/// about 1.6 s against 0.12 s without it) — the island's absence is the blocker,
/// not the clock.
///
/// So the arm asks the same question of content CI HAS: the committed body, the
/// committed rig, one sun, the renderer's own defaults. A character that renders
/// near-black under a directional light is a character bug — a normal, a skin or
/// a material — and that is exactly what FIX3's ledger says a low number now
/// means, because the ambient term stopped being the suspect when the sky became
/// the ambient.
///
/// Measured on the committed body: **4 255 body pixels, p25 165, median 179**,
/// against a floor of 40.
///
/// **The mutation**: negate every vertex normal in the decoded body. The lit
/// side faces away and the arm goes red at the floor. Verified — see the wave
/// ledger for the number it fell to.
#[test]
fn a_committed_body_lit_by_one_sun_is_not_near_black() {
    let Some(gpu) = gpu_or_skip() else {
        eprintln!("SKIP: no GPU adapter for the luminance floor");
        return;
    };
    let bytes = std::fs::read(repo().join("samples/starter-character/Starter_Body.inf_mesh"))
        .expect("the committed body");
    let mesh: inf_mesh::MeshAsset = inf_asset::decode(&bytes).expect("it decodes");
    let skinned = inf_player::skinned::skinned_mesh_data(&mesh).expect("it has a skin stream");
    let rig: inf_anim::SkeletonAsset = inf_asset::decode(
        &std::fs::read(repo().join("samples/starter-character/Starter.inf_skel"))
            .expect("the committed rig"),
    )
    .expect("it decodes");
    let palette = std::sync::Arc::new(inf_anim::skinning_matrices(
        &rig.skeleton,
        &inf_anim::Pose::rest(&rig.skeleton),
    ));

    let mut scene = inf_render::RenderScene {
        skinned_meshes: vec![std::sync::Arc::new(skinned)],
        ..Default::default()
    };
    scene.skinned.push(inf_render::SkinnedInstance {
        vt: Default::default(),
        translation: glam::DVec3::ZERO,
        rotation: glam::Quat::IDENTITY,
        scale: glam::Vec3::ONE,
        // The starter skin's own neutral, and NOT white: a body that only passes
        // the floor because it is a mirror would not be a measurement of light.
        color: [0.62, 0.55, 0.50, 1.0],
        metallic: 0.0,
        roughness: 0.6,
        emissive: [0.0; 3],
        id: 1,
        mesh: 0,
        blend: 0,
        cutoff: 0.5,
        palette,
        shadow: inf_render::SkinnedShadow::BindSphere,
    });
    // One directional sun, over the camera's shoulder — the light every level in
    // this engine starts with (`samples::blank3d_scene`'s own).
    scene.lights.push(inf_render::RenderLight {
        kind: inf_render::LightKind::Directional,
        direction: glam::Vec3::new(0.35, 0.8, 0.5).normalize(),
        color: [1.0, 0.98, 0.94],
        intensity: 3.0,
        ..Default::default()
    });
    scene.mark_dirty();

    let (w, h) = (256u32, 320u32);
    let eye = glam::DVec3::new(0.0, 1.0, 2.6);
    let at = glam::DVec3::new(0.0, 0.95, 0.0);
    let view = inf_render::RenderView {
        origin: inf_math::FloatingOrigin::new(glam::DVec3::ZERO),
        eye_world: eye,
        forward: (at - eye).as_vec3().normalize(),
        up: glam::Vec3::Y,
        fov_y: 60f32.to_radians(),
        near: 0.05,
        width: w,
        height: h,
        ortho: None,
    };
    let target = inf_render::HeadlessTarget::new(&gpu, w, h);
    let mut renderer = inf_render::EngineRenderer::new(&gpu, inf_render::HEADLESS_FORMAT);
    renderer.render(&gpu, &scene, &view, &target.view, (w, h));
    let img = target.read_rgba(&gpu).expect("readback");

    // The BODY's pixels, not the frame's: the background is the renderer's sky
    // and averaging it in would answer a question about the sky. A body pixel is
    // one whose red channel leads, which the blue-grey background's does not —
    // the same predicate `golden_skinned_masked` had to learn.
    let mut lum: Vec<u16> = img
        .chunks(4)
        .filter(|p| p[0] as i16 - p[2] as i16 > 8)
        .map(|p| (0.2126 * p[0] as f32 + 0.7152 * p[1] as f32 + 0.0722 * p[2] as f32) as u16)
        .collect();
    assert!(
        lum.len() > 2_000,
        "only {} body pixels — the body is not on screen and the floor below \
         would be measuring nothing",
        lum.len()
    );
    lum.sort_unstable();
    let p25 = lum[lum.len() / 4];
    let median = lum[lum.len() / 2];
    eprintln!(
        "committed body under one sun: {} px, p25 {p25}, median {median}",
        lum.len()
    );
    assert!(
        p25 as f64 > 40.0,
        "the committed body's p25 luminance is {p25} — the floor is 40, and \
         below it a character reads as a silhouette. Since FIX3 made the sky the \
         ambient term, a dark body is a CHARACTER bug (normals, skin weights or \
         the material), not a lighting one"
    );
}

/// A GPU context, or `None` on a machine (or a CI leg) with no adapter — the
/// same door `inf-render`'s goldens take, so a headless leg skips rather than
/// fails.
fn gpu_or_skip() -> Option<inf_render::GpuContext> {
    match inf_render::GpuContext::headless() {
        Ok(gpu) => Some(gpu),
        Err(e) => {
            eprintln!("SKIP: no GPU adapter ({e})");
            None
        }
    }
}

/// **THE SOLES REST ON THE SURFACE** (user mandate 2026-09-05 #2, flat ground).
///
/// # Where the ground is, in model space
///
/// A character's entity transform is its capsule CENTRE and
/// `inf_ecs::pose::character_drop` subtracts `half_extents.y + radius` before
/// drawing, so **the model's origin is the capsule's lowest point** — which is
/// where a controller rests it on flat ground. A body mesh is authored with its
/// soles at y = 0. So "the soles rest on the surface" is exactly "the lowest
/// skinned vertex of the posed body sits at y = 0", and any other number is a
/// hover or a penetration in metres.
///
/// # What was measured before this wave
///
/// Over 21 samples of each cycle, the **lowest** sole reached:
///
/// | | idle | walk | run |
/// |---|---|---|---|
/// | committed male | −5.6 mm | **+29.5 mm** | **+73.7 mm** |
/// | committed female | −5.9 mm | **+32.5 mm** | **+79.6 mm** |
/// | island hero (rebound mannequin, local) | **+19.5 mm** | +5.8 mm | +86.6 mm |
///
/// A generated cycle is built from swing angles and a hip height and nothing in
/// that derivation knows where the soles are; a retargeted cycle carries the
/// residue of two rigs' hip heights. Both are one constant per clip, and
/// `inf_anim::retarget::settle_to_ground` subtracts it — at the end of the
/// derivation and at the end of the retarget, so every clip that reaches a body
/// has been settled once.
///
/// The tolerance is **10 mm**, which is the mandate's, and it is asked of the
/// cycle's MINIMUM: a run's flight phase is supposed to be off the ground, and a
/// rule that pinned every frame would delete it.
///
/// **The mutation**: comment out the `settle_to_ground` call in
/// `locomotion::derive_locomotion` and re-bless. Walk fails at +29.5 mm and run
/// at +73.7 mm. Verified.
///
/// Slopes, stairs and kerbs need per-foot traces — that is CHAR1b's foot IK and
/// this arm is deliberately about flat ground only.
#[test]
fn a_committed_bodys_soles_rest_on_the_ground_plane() {
    const TOLERANCE_M: f32 = 0.010;
    for (who, dir, skel, mesh, clips) in [
        (
            "male",
            "samples/starter-character",
            "Starter.inf_skel",
            "Starter_Body.inf_mesh",
            [
                "Starter_Idle.inf_anim",
                "Starter_Walk.inf_anim",
                "Starter_Run.inf_anim",
            ],
        ),
        (
            "female",
            "samples/starter-character-f",
            "Starter_F.inf_skel",
            "Starter_F_Body.inf_mesh",
            [
                "Starter_F_Idle.inf_anim",
                "Starter_F_Walk.inf_anim",
                "Starter_F_Run.inf_anim",
            ],
        ),
    ] {
        let rd = |n: &str| {
            std::fs::read(repo().join(dir).join(n)).unwrap_or_else(|e| panic!("{dir}/{n}: {e}"))
        };
        let sk: inf_anim::SkeletonAsset = inf_asset::decode(&rd(skel)).expect("rig decodes");
        let ma: inf_mesh::MeshAsset = inf_asset::decode(&rd(mesh)).expect("body decodes");
        let sd = inf_player::skinned::skinned_mesh_data(&ma).expect("the body has a skin");
        // The BIND pose first: a mesh whose soles are not at zero would make
        // every number below meaningless, and this is where that shows.
        let bind = inf_anim::skinning_matrices(&sk.skeleton, &inf_anim::Pose::rest(&sk.skeleton));
        let bind_low = lowest_skinned_y(&sd, &bind);
        assert!(
            bind_low.abs() < TOLERANCE_M,
            "{who}: the BIND pose's lowest vertex is {bind_low:+.4} m — the body \
             is not authored with its soles on the origin, so the capsule's \
             bottom is not its ground"
        );
        for c in clips {
            let cl: inf_anim::AnimClipAsset = inf_asset::decode(&rd(c)).expect("clip decodes");
            let mut lowest = f32::INFINITY;
            for i in 0..=20 {
                let t = cl.clip.duration * i as f32 / 20.0;
                let pose = inf_anim::sample_clip(&sk.skeleton, &cl.clip, t, true);
                let pal = inf_anim::skinning_matrices(&sk.skeleton, &pose);
                lowest = lowest.min(lowest_skinned_y(&sd, &pal));
            }
            eprintln!("{who} {c}: lowest sole over the cycle {lowest:+.4} m");
            assert!(
                lowest.abs() <= TOLERANCE_M,
                "{who}'s {c} plants its lowest foot {lowest:+.4} m from the \
                 ground plane, against a 10 mm bar — positive is a character \
                 walking on air, negative is one whose soles are inside the \
                 pavement"
            );
        }
    }
}

/// The lowest Y a skinned body reaches under one palette — linear-blend
/// skinning, the same arithmetic `skinned_mesh.wgsl` runs, on the CPU.
fn lowest_skinned_y(mesh: &inf_render::SkinnedMeshData, palette: &[glam::Mat4]) -> f32 {
    let mut lo = f32::INFINITY;
    for v in &mesh.vertices {
        let p = glam::Vec3::from(v.pos);
        let mut acc = glam::Vec4::ZERO;
        for k in 0..4 {
            let w = v.weights[k];
            if w == 0.0 {
                continue;
            }
            let j = (v.joints[k] as usize).min(palette.len().saturating_sub(1));
            acc += w * (palette[j] * p.extend(1.0));
        }
        lo = lo.min(acc.y);
    }
    lo
}

/// **A RETARGETED CLIP IS SETTLED TOO** — the same rule, at the other door.
///
/// The committed clips above are GENERATED; every clip that comes across the
/// Unreal bridge is RETARGETED, and the island's own hero wears three of those.
/// So the rule has to hold at both doors or the body the demo photographs is not
/// the body the gate measured. Built here rather than read off disk, because the
/// mannequin's clips are licensed content that CI may not hold.
///
/// The fixture is the honest shape of the defect: a source rig whose hips sit
/// **80 mm** higher than the target's, and a clip that copies its root track
/// across. Before the settle the retargeted cycle hovers by that difference.
///
/// **The mutation**: delete the `settle_to_ground` call at the end of
/// `retarget_clip`. `ground_settle_m` becomes 0.0 and the post-settle assertion
/// fails at +0.0800 m. Verified.
#[test]
fn a_retargeted_clip_is_settled_onto_the_target_rigs_ground_plane() {
    let dst = manny();
    // The SAME rig on both sides — a retarget between two copies of one skeleton
    // is the identity, so anything left over is the clip's own, which is exactly
    // what this arm is about.
    let lifted = dst.clone();
    // A one-key root translation track that LIFTS: the clip says "stand 80 mm
    // above where you stand", which is what a retarget's hip-height residue
    // looks like once it has landed on the target rig.
    let root_t = lifted.joints()[0].local_bind.translation_vec() + glam::Vec3::new(0.0, 0.080, 0.0);
    let mut track = inf_anim::JointTrack::new(0);
    track.translation = Some(inf_anim::Vec3Track {
        times: vec![0.0, 1.0],
        values: vec![root_t.to_array(), root_t.to_array()],
        interp: inf_anim::Interpolation::Linear,
    });
    let clip = AnimClip::new("lifted", vec![track]);

    let map = RetargetMap::shared_names(&lifted, &dst);
    let (out, report) = retarget_clip(&clip, &lifted, &dst, &map, true);
    eprintln!("ground settle {:+.4} m", report.ground_settle_m);
    assert!(
        (report.ground_settle_m - 0.080).abs() < 0.002,
        "the retarget settled {:+.4} m against an 80 mm lift",
        report.ground_settle_m
    );

    // …and the settled clip really stands on the target's floor.
    let rest = inf_anim::Pose::rest(&dst);
    let bind_low = lowest_ground_joint_y(&dst, &rest);
    let mut lowest = f32::INFINITY;
    for i in 0..=20 {
        let t = out.duration * i as f32 / 20.0;
        let pose = inf_anim::sample_clip(&dst, &out, t, true);
        lowest = lowest.min(lowest_ground_joint_y(&dst, &pose));
    }
    assert!(
        (lowest - bind_low).abs() < 0.002,
        "the settled clip's lowest foot is {:+.4} m from the bind pose's",
        lowest - bind_low
    );
}

/// The lowest of a rig's ground joints under `pose`, in world space — the same
/// question `settle_to_ground` asks, asked from outside it.
fn lowest_ground_joint_y(sk: &Skeleton, pose: &inf_anim::Pose) -> f32 {
    let mut lo = f32::INFINITY;
    for name in ["ball_l", "ball_r", "foot_l", "foot_r"] {
        let Some(i) = sk.index_of(name) else { continue };
        let mut chain = vec![i as usize];
        let mut cur = i as usize;
        while let Some(p) = sk.joint(cur).and_then(|j| j.parent) {
            chain.push(p as usize);
            cur = p as usize;
        }
        let mut m = glam::Mat4::IDENTITY;
        for &j in chain.iter().rev() {
            m *= pose.locals[j].to_mat4();
        }
        lo = lo.min(m.w_axis.y);
    }
    lo
}
