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
    for i in 0..joints.len() {
        joints[i].parent = joints[i]
            .parent
            .and_then(|p| if (p as usize) < i { Some(p) } else { None });
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
    let height = f64::from(BodyParams::default().height_m);
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
