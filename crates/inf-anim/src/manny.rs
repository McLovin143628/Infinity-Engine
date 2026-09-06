//! **The 161-bone UE5 mannequin hierarchy** (SK1a) — this engine's default biped.
//!
//! [`BodyPlan::Biped`](crate::BodyPlan::Biped) emits the bone list, the parent
//! links and the emission order of Epic's `SK_Mannequin`, verbatim, because those
//! names are the interchange vocabulary the whole character ecosystem is written
//! against: a Mixamo clip, a MetaHuman body, an ALS animation blueprint and every
//! marketplace retarget chain address `thigh_l`, `spine_03`, `ik_foot_root` and
//! `index_metacarpal_r` by those spellings. A rig that calls its shin `lower_leg_l`
//! is a rig nobody else's content fits.
//!
//! # Where this list came from — and the correction it carries
//!
//! **Measured, not transcribed.** The briefing document this table was asked for
//! (`character-skeleton-rig-outline.md`, which lives outside the repository with
//! the rest of the reference material) prints a tree under the heading "the
//! complete hierarchy for the 161 bones"; that tree is
//! **89 bones** — exactly the figure the same document's own table gives for
//! `SKM_Manny_Simple` — and it also states parents the shipped asset does not have
//! (it hangs the IK subtrees under `spine_03` and invents `ik_head`, `ik_pelvis`
//! and `ik_spine`). The table below was read instead out of the
//! `FReferenceSkeleton` in a shipped `SK_Mannequin.uasset`: 161 bones, one root,
//! every parent
//! preceding its child. Where the document and the asset disagree, the asset wins,
//! and the disagreements are named here rather than quietly resolved:
//!
//! **How to re-derive it** (SK1a audit — the first write-up said "measured off
//! the asset" without saying how, which is a claim a successor cannot check). The
//! package is `Content/.../Characters/Mannequins/Meshes/SK_Mannequin.uasset` in
//! the UE5 reference project named in the anim/island mandate memo — a
//! **non-cooked** editor package, so `FMeshBoneInfo::ExportName` is present and
//! the names are plain ASCII in the file. Scan the package body for an `i32`
//! equal to 161 followed by 161 records of
//! `(i32 name-index, i32 name-number, i32 parent, i32 length, ASCII+NUL)`, and
//! accept only the offset where the first parent is `-1` and every later one is
//! in `0..i`. There is exactly one such offset. `RawRefBonePose` follows
//! immediately: another `i32` count of 161, then 161 `FTransform`s of ten `f64`
//! (quat *xyzw*, translation, scale). The twist thirds and the hand proportions
//! below come from that second array.
//!
//! | | the document | the asset |
//! |---|---|---|
//! | bone count of the printed tree | claimed 161 | **89** |
//! | neck | `neck_01` → `head` | `neck_01` → **`neck_02`** → `head` |
//! | IK subtree parent | `spine_03` | **`root`** |
//! | `ik_hand_l` / `ik_hand_r` parent | `ik_hand_root` | **`ik_hand_gun`** |
//! | `ik_head`, `ik_pelvis`, `ik_spine` | present | **absent** |
//! | corrective / helper bones | absent | **74**, `weapon_l/r`, `interaction` and `center_of_mass` among them |
//! | census | — | 63 deform + 16 twists + 7 handles + 74 helpers + 1 root = **161** |
//!
//! # What is Epic's and what is ours
//!
//! The **names, the parent links and the emission order** are the interchange
//! contract and are reproduced. **Every offset is a proportion multiplied by a
//! [`BodyParams`] length** — nothing here is a copied absolute — so the rig is
//! proportionally *whatever height it was asked for* rather than a scaled copy of
//! somebody's mannequin. Where those proportions come from is three different
//! answers, and the SK1a audit's correction is that the first write-up gave only
//! the first two:
//!
//! 1. **Invented** — the torso, the limbs and the girdles. `CLAVICLE_OF_SPAN`,
//!    `SHOULDER_OF_SPAN`, `HAND_OF_HEIGHT`, `BALL_OF_HEIGHT` and every ratio on
//!    [`BodyParams`] are this engine's own numbers, shared with
//!    [`crate::template`], and they do **not** reproduce the shipped rig's
//!    measurements (the asset's hand is 0.096 of its height; this module's is
//!    0.105).
//! 2. **Rules read off the asset** — the twist bones sit at *exactly one third
//!    and two thirds* of their segment and `_01` is always the one nearest the
//!    joint that drives it. Both are load-bearing for [`crate::drive`]'s law, and
//!    both are structure rather than geometry.
//! 3. **Measured proportions** — the nineteen bones of the hand. Their offsets in
//!    `Place::Hand` (not a link: the type is private) are the reference
//!    skeleton's own local bind translations,
//!    **normalized so the middle-finger chain sums to 1.0 of hand length**, so
//!    they are ratios of a length this module chooses and carry no absolute
//!    dimension. There is no *rule* for where a pinky metacarpal sits relative to
//!    a ring one; it is anatomy, and inventing it would make a hand nobody's
//!    glove fits. Fifty-seven numbers, named here rather than left to read as
//!    though they were derived.
//!
//! # The bind pose is a T-pose of pure translations
//!
//! The template law holds here too: **identity rotation and unit scale on every
//! joint**, so an inverse bind is the exact negation of an accumulated translation
//! and no `sin`/`cos` appears anywhere (the P14 portability law). That forces the
//! arms out along ±X rather than into the A-pose the shipped mannequin stands in —
//! an A-pose is rotation, and rotation in a bind pose is the thing this generator
//! does not do. The consequence is honest and small: every limb axis is an exact
//! unit basis vector, which is what makes [`crate::drive`]'s twist axes exact.
//!
//! # The helper bones exist and are not placed
//!
//! Seventy-four of the 161 are correctives, bend-assists and markers
//! (`upperarm_correctiveRoot_l`, `thigh_bck_lwr_r`, `wrist_inner_l`, `weapon_r`,
//! `interaction`, `center_of_mass`).
//! They are emitted **at their parent's origin** and carry
//! [`BoneRoleKind::Helper`], because their reason to exist here is that an
//! externally authored clip or a retarget finds every bone it names. Nothing in
//! this engine drives them; a bone at its parent's origin is a bone with no
//! influence, and the role table is what keeps the mannequin generator, the
//! ragdoll builder and the weight solver from tripping over them.

use glam::{Quat, Vec3};

use crate::asset::SkeletonAsset;
use crate::roles::{BoneRole, BoneRoleKind, BoneSide, IkFollow, TwistDriver};
use crate::skeleton::{Joint, JointTransform, Skeleton};
use crate::sockets::Socket;
use crate::template::{BodyParams, ConeLimit, JointLimit, TemplateError};

/// How many bones the mannequin hierarchy has. Asserted against the emitted rig by
/// `the_manny_rig_is_one_hundred_and_sixty_one_bones`, which is the arm that would
/// notice a row being dropped out of the table below.
pub const MANNY_JOINT_COUNT: usize = 161;

/// Where a clavicle sits, as a fraction of `shoulder_width_m` out from the spine.
const CLAVICLE_OF_SPAN: f32 = 0.15;
/// The rest of the way out to the shoulder joint, from the clavicle.
const SHOULDER_OF_SPAN: f32 = 0.35;
/// Hand length — wrist to the tip of the middle finger — as a fraction of height.
/// The whole finger table below is expressed in units of this.
///
/// **MEASURED, since wave CHAR1a.** SK1a authored 0.105 and said so, because the
/// only number it had was read off a `.uasset` and estimated against a rig
/// "~178 cm tall" — it recorded 0.096 as the reference's value and shipped a
/// bigger hand anyway. The bridge now exports `SKM_Manny` as glTF, so both
/// halves of that ratio are measurable on the shipped asset instead of
/// estimated:
///
/// * the middle-finger chain (`middle_metacarpal_l` + `_01` + `_02` + `_03`,
///   summed as the magnitudes of their local bind translations, which are
///   rotation-independent) is **0.1720 m**;
/// * the mesh's own vertex bounds are **1.8054 m** tall (`SKM_Quinn`: 0.1720 m
///   on 1.8017 m).
///
/// 0.1720 / 1.8054 = **0.0953**, and that is what this constant now is. The old
/// value made every rig this engine generates 10% big in the hand — visible the
/// moment a character holds a weapon, which is why the grip catalogue's
/// affordances were being authored against a hand that did not match the mesh
/// they would be shown on. The *proportions inside* the hand were measured at
/// SK1a and are unchanged; this is only the scale they are expressed in.
const HAND_OF_HEIGHT: f64 = 0.0953;
/// How far forward of the ankle the ball of the foot sits, as a fraction of height.
const BALL_OF_HEIGHT: f64 = 0.085;
/// A knee's flexion range, degrees.
const KNEE_RANGE_DEG: (f32, f32) = (-150.0, 0.0);
/// An elbow's flexion range, degrees.
const ELBOW_RANGE_DEG: (f32, f32) = (0.0, 150.0);
/// How much wider than the solver's own maximum a finger's cone is authored,
/// degrees — see the cone loop in [`build_manny`] for why it is not zero.
const CONE_MARGIN_DEG: f32 = 0.5;

/// How a bone's bind offset is derived from [`BodyParams`].
///
/// A rule per *class* of bone rather than a number per bone — with one honest
/// exception, [`Place::Hand`], whose numbers are per-bone measured proportions.
/// See the module docs' three-way split.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Place {
    /// At its parent's origin — the helpers, the corrective roots, the IK anchors.
    Origin,
    /// The hip girdle, at `hip_height_ratio` of the height.
    Pelvis,
    /// One fifth of the pelvis→shoulder rise.
    SpineRise,
    /// One third of the shoulder→head rise.
    NeckRise,
    /// Out along ±X by a fraction of `shoulder_width_m`.
    Span(f32),
    /// Out along ±X by a fraction of the upper-arm length.
    UpperArm(f32),
    /// Out along ±X by a fraction of the forearm length.
    LowerArm(f32),
    /// Out along ±X by half of `hip_width_m` (a thigh root).
    HipSpan,
    /// Down along −Y by a fraction of the upper-leg length.
    UpperLeg(f32),
    /// Down along −Y by a fraction of the lower-leg length.
    LowerLeg(f32),
    /// Forward along +Z to the ball of the foot.
    Ball,
    /// A hand bone, in units of hand length: `along` the arm axis, `across` the
    /// palm (+Z), `up` its normal (+Y).
    ///
    /// **These are measured, not derived** — the reference skeleton's own local
    /// bind translations, normalized so the middle-finger chain's `along`
    /// components sum to exactly 1.0. The two sides are an exact mirror (`along`
    /// takes the side's sign; `across` and `up` do not, because a mirror about
    /// the X plane leaves Y and Z alone). See the module docs.
    Hand { along: f32, across: f32, up: f32 },
    /// Wherever the named joint's **global** bind is — an IK handle's placement,
    /// resolved against this bone's own parent.
    Mirrors(&'static str),
}

/// One row of the hierarchy.
struct MannyBone {
    name: &'static str,
    parent: Option<u16>,
    place: Place,
    kind: BoneRoleKind,
    side: BoneSide,
}

const fn b(
    name: &'static str,
    parent: Option<u16>,
    place: Place,
    kind: BoneRoleKind,
    side: BoneSide,
) -> MannyBone {
    MannyBone {
        name,
        parent,
        place,
        kind,
        side,
    }
}

use BoneRoleKind as Kind;
use BoneSide as Side;

/// **The hierarchy**, in the shipped asset's own index order.
///
/// The order is not decoration. It puts every deform bone ahead of the `ik_*`
/// subtrees, which is the belt beside the role table's suspenders: the last site in
/// the engine that still matches `contains("foot")` and takes the first hit finds
/// `foot_l` before `ik_foot_l` **because of this ordering**, and a rig that emitted
/// its handles first would drive a marker instead of a leg.
static BONES: [MannyBone; MANNY_JOINT_COUNT] = [
    b("root", None, Place::Origin, Kind::Root, Side::Center),
    b("pelvis", Some(0), Place::Pelvis, Kind::Pelvis, Side::Center),
    b(
        "spine_01",
        Some(1),
        Place::SpineRise,
        Kind::Spine,
        Side::Center,
    ),
    b(
        "spine_02",
        Some(2),
        Place::SpineRise,
        Kind::Spine,
        Side::Center,
    ),
    b(
        "spine_03",
        Some(3),
        Place::SpineRise,
        Kind::Spine,
        Side::Center,
    ),
    b(
        "spine_04",
        Some(4),
        Place::SpineRise,
        Kind::Spine,
        Side::Center,
    ),
    b(
        "spine_05",
        Some(5),
        Place::SpineRise,
        Kind::Spine,
        Side::Center,
    ),
    b(
        "neck_01",
        Some(6),
        Place::NeckRise,
        Kind::Neck,
        Side::Center,
    ),
    b(
        "neck_02",
        Some(7),
        Place::NeckRise,
        Kind::Neck,
        Side::Center,
    ),
    b("head", Some(8), Place::NeckRise, Kind::Head, Side::Center),
    b(
        "clavicle_l",
        Some(6),
        Place::Span(CLAVICLE_OF_SPAN),
        Kind::Clavicle,
        Side::Left,
    ),
    b(
        "upperarm_l",
        Some(10),
        Place::Span(SHOULDER_OF_SPAN),
        Kind::UpperArm,
        Side::Left,
    ),
    b(
        "lowerarm_l",
        Some(11),
        Place::UpperArm(1.0),
        Kind::LowerArm,
        Side::Left,
    ),
    b(
        "lowerarm_twist_02_l",
        Some(12),
        Place::LowerArm(1.0 / 3.0),
        Kind::Twist,
        Side::Left,
    ),
    b(
        "lowerarm_twist_01_l",
        Some(12),
        Place::LowerArm(2.0 / 3.0),
        Kind::Twist,
        Side::Left,
    ),
    b(
        "lowerarm_correctiveRoot_l",
        Some(12),
        Place::Origin,
        Kind::Helper,
        Side::Left,
    ),
    b(
        "lowerarm_in_l",
        Some(15),
        Place::Origin,
        Kind::Helper,
        Side::Left,
    ),
    b(
        "lowerarm_out_l",
        Some(15),
        Place::Origin,
        Kind::Helper,
        Side::Left,
    ),
    b(
        "lowerarm_fwd_l",
        Some(15),
        Place::Origin,
        Kind::Helper,
        Side::Left,
    ),
    b(
        "lowerarm_bck_l",
        Some(15),
        Place::Origin,
        Kind::Helper,
        Side::Left,
    ),
    b(
        "hand_l",
        Some(12),
        Place::LowerArm(1.0),
        Kind::Hand,
        Side::Left,
    ),
    b(
        "middle_metacarpal_l",
        Some(20),
        Place::Hand {
            along: 0.1972,
            across: -0.0107,
            up: 0.0440,
        },
        Kind::Finger,
        Side::Left,
    ),
    b(
        "middle_01_l",
        Some(21),
        Place::Hand {
            along: 0.3563,
            across: 0.0,
            up: 0.0,
        },
        Kind::Finger,
        Side::Left,
    ),
    b(
        "middle_02_l",
        Some(22),
        Place::Hand {
            along: 0.3020,
            across: 0.0,
            up: 0.0,
        },
        Kind::Finger,
        Side::Left,
    ),
    b(
        "middle_03_l",
        Some(23),
        Place::Hand {
            along: 0.1445,
            across: 0.0,
            up: 0.0,
        },
        Kind::Finger,
        Side::Left,
    ),
    b(
        "pinky_metacarpal_l",
        Some(20),
        Place::Hand {
            along: 0.1936,
            across: 0.1397,
            up: 0.0179,
        },
        Kind::Finger,
        Side::Left,
    ),
    b(
        "pinky_01_l",
        Some(25),
        Place::Hand {
            along: 0.2896,
            across: 0.0,
            up: 0.0,
        },
        Kind::Finger,
        Side::Left,
    ),
    b(
        "pinky_02_l",
        Some(26),
        Place::Hand {
            along: 0.2229,
            across: 0.0,
            up: 0.0,
        },
        Kind::Finger,
        Side::Left,
    ),
    b(
        "pinky_03_l",
        Some(27),
        Place::Hand {
            along: 0.1192,
            across: 0.0,
            up: 0.0,
        },
        Kind::Finger,
        Side::Left,
    ),
    b(
        "ring_metacarpal_l",
        Some(20),
        Place::Hand {
            along: 0.1971,
            across: 0.0638,
            up: 0.0317,
        },
        Kind::Finger,
        Side::Left,
    ),
    b(
        "ring_01_l",
        Some(29),
        Place::Hand {
            along: 0.3299,
            across: 0.0,
            up: 0.0,
        },
        Kind::Finger,
        Side::Left,
    ),
    b(
        "ring_02_l",
        Some(30),
        Place::Hand {
            along: 0.2907,
            across: 0.0,
            up: 0.0,
        },
        Kind::Finger,
        Side::Left,
    ),
    b(
        "ring_03_l",
        Some(31),
        Place::Hand {
            along: 0.1323,
            across: 0.0,
            up: 0.0,
        },
        Kind::Finger,
        Side::Left,
    ),
    b(
        "thumb_01_l",
        Some(20),
        Place::Hand {
            along: 0.1164,
            across: -0.1508,
            up: -0.0793,
        },
        Kind::Thumb,
        Side::Left,
    ),
    b(
        "thumb_02_l",
        Some(33),
        Place::Hand {
            along: 0.2558,
            across: 0.0,
            up: 0.0,
        },
        Kind::Thumb,
        Side::Left,
    ),
    b(
        "thumb_03_l",
        Some(34),
        Place::Hand {
            along: 0.1803,
            across: 0.0,
            up: 0.0,
        },
        Kind::Thumb,
        Side::Left,
    ),
    b(
        "index_metacarpal_l",
        Some(20),
        Place::Hand {
            along: 0.2013,
            across: -0.1390,
            up: 0.0225,
        },
        Kind::Finger,
        Side::Left,
    ),
    b(
        "index_01_l",
        Some(36),
        Place::Hand {
            along: 0.3434,
            across: 0.0,
            up: 0.0,
        },
        Kind::Finger,
        Side::Left,
    ),
    b(
        "index_02_l",
        Some(37),
        Place::Hand {
            along: 0.2384,
            across: 0.0,
            up: 0.0,
        },
        Kind::Finger,
        Side::Left,
    ),
    b(
        "index_03_l",
        Some(38),
        Place::Hand {
            along: 0.1516,
            across: 0.0,
            up: 0.0,
        },
        Kind::Finger,
        Side::Left,
    ),
    b(
        "wrist_inner_l",
        Some(20),
        Place::Origin,
        Kind::Helper,
        Side::Left,
    ),
    b(
        "wrist_outer_l",
        Some(20),
        Place::Origin,
        Kind::Helper,
        Side::Left,
    ),
    b(
        "weapon_l",
        Some(20),
        Place::Origin,
        Kind::Helper,
        Side::Left,
    ),
    b(
        "upperarm_twist_01_l",
        Some(11),
        Place::UpperArm(1.0 / 3.0),
        Kind::Twist,
        Side::Left,
    ),
    b(
        "upperarm_twistCor_01_l",
        Some(43),
        Place::Origin,
        Kind::Helper,
        Side::Left,
    ),
    b(
        "upperarm_twist_02_l",
        Some(11),
        Place::UpperArm(2.0 / 3.0),
        Kind::Twist,
        Side::Left,
    ),
    b(
        "upperarm_tricep_l",
        Some(45),
        Place::Origin,
        Kind::Helper,
        Side::Left,
    ),
    b(
        "upperarm_bicep_l",
        Some(45),
        Place::Origin,
        Kind::Helper,
        Side::Left,
    ),
    b(
        "upperarm_twistCor_02_l",
        Some(45),
        Place::Origin,
        Kind::Helper,
        Side::Left,
    ),
    b(
        "upperarm_correctiveRoot_l",
        Some(11),
        Place::Origin,
        Kind::Helper,
        Side::Left,
    ),
    b(
        "upperarm_bck_l",
        Some(49),
        Place::Origin,
        Kind::Helper,
        Side::Left,
    ),
    b(
        "upperarm_fwd_l",
        Some(49),
        Place::Origin,
        Kind::Helper,
        Side::Left,
    ),
    b(
        "upperarm_in_l",
        Some(49),
        Place::Origin,
        Kind::Helper,
        Side::Left,
    ),
    b(
        "upperarm_out_l",
        Some(49),
        Place::Origin,
        Kind::Helper,
        Side::Left,
    ),
    b(
        "clavicle_out_l",
        Some(10),
        Place::Origin,
        Kind::Helper,
        Side::Left,
    ),
    b(
        "clavicle_scap_l",
        Some(10),
        Place::Origin,
        Kind::Helper,
        Side::Left,
    ),
    b(
        "clavicle_r",
        Some(6),
        Place::Span(CLAVICLE_OF_SPAN),
        Kind::Clavicle,
        Side::Right,
    ),
    b(
        "upperarm_r",
        Some(56),
        Place::Span(SHOULDER_OF_SPAN),
        Kind::UpperArm,
        Side::Right,
    ),
    b(
        "lowerarm_r",
        Some(57),
        Place::UpperArm(1.0),
        Kind::LowerArm,
        Side::Right,
    ),
    b(
        "lowerarm_twist_02_r",
        Some(58),
        Place::LowerArm(1.0 / 3.0),
        Kind::Twist,
        Side::Right,
    ),
    b(
        "lowerarm_twist_01_r",
        Some(58),
        Place::LowerArm(2.0 / 3.0),
        Kind::Twist,
        Side::Right,
    ),
    b(
        "lowerarm_correctiveRoot_r",
        Some(58),
        Place::Origin,
        Kind::Helper,
        Side::Right,
    ),
    b(
        "lowerarm_out_r",
        Some(61),
        Place::Origin,
        Kind::Helper,
        Side::Right,
    ),
    b(
        "lowerarm_in_r",
        Some(61),
        Place::Origin,
        Kind::Helper,
        Side::Right,
    ),
    b(
        "lowerarm_fwd_r",
        Some(61),
        Place::Origin,
        Kind::Helper,
        Side::Right,
    ),
    b(
        "lowerarm_bck_r",
        Some(61),
        Place::Origin,
        Kind::Helper,
        Side::Right,
    ),
    b(
        "hand_r",
        Some(58),
        Place::LowerArm(1.0),
        Kind::Hand,
        Side::Right,
    ),
    b(
        "middle_metacarpal_r",
        Some(66),
        Place::Hand {
            along: 0.1972,
            across: -0.0107,
            up: 0.0440,
        },
        Kind::Finger,
        Side::Right,
    ),
    b(
        "middle_01_r",
        Some(67),
        Place::Hand {
            along: 0.3563,
            across: 0.0,
            up: 0.0,
        },
        Kind::Finger,
        Side::Right,
    ),
    b(
        "middle_02_r",
        Some(68),
        Place::Hand {
            along: 0.3020,
            across: 0.0,
            up: 0.0,
        },
        Kind::Finger,
        Side::Right,
    ),
    b(
        "middle_03_r",
        Some(69),
        Place::Hand {
            along: 0.1445,
            across: 0.0,
            up: 0.0,
        },
        Kind::Finger,
        Side::Right,
    ),
    b(
        "pinky_metacarpal_r",
        Some(66),
        Place::Hand {
            along: 0.1936,
            across: 0.1397,
            up: 0.0179,
        },
        Kind::Finger,
        Side::Right,
    ),
    b(
        "pinky_01_r",
        Some(71),
        Place::Hand {
            along: 0.2896,
            across: 0.0,
            up: 0.0,
        },
        Kind::Finger,
        Side::Right,
    ),
    b(
        "pinky_02_r",
        Some(72),
        Place::Hand {
            along: 0.2229,
            across: 0.0,
            up: 0.0,
        },
        Kind::Finger,
        Side::Right,
    ),
    b(
        "pinky_03_r",
        Some(73),
        Place::Hand {
            along: 0.1192,
            across: 0.0,
            up: 0.0,
        },
        Kind::Finger,
        Side::Right,
    ),
    b(
        "ring_metacarpal_r",
        Some(66),
        Place::Hand {
            along: 0.1971,
            across: 0.0638,
            up: 0.0317,
        },
        Kind::Finger,
        Side::Right,
    ),
    b(
        "ring_01_r",
        Some(75),
        Place::Hand {
            along: 0.3299,
            across: 0.0,
            up: 0.0,
        },
        Kind::Finger,
        Side::Right,
    ),
    b(
        "ring_02_r",
        Some(76),
        Place::Hand {
            along: 0.2907,
            across: 0.0,
            up: 0.0,
        },
        Kind::Finger,
        Side::Right,
    ),
    b(
        "ring_03_r",
        Some(77),
        Place::Hand {
            along: 0.1323,
            across: 0.0,
            up: 0.0,
        },
        Kind::Finger,
        Side::Right,
    ),
    b(
        "thumb_01_r",
        Some(66),
        Place::Hand {
            along: 0.1164,
            across: -0.1508,
            up: -0.0793,
        },
        Kind::Thumb,
        Side::Right,
    ),
    b(
        "thumb_02_r",
        Some(79),
        Place::Hand {
            along: 0.2558,
            across: 0.0,
            up: 0.0,
        },
        Kind::Thumb,
        Side::Right,
    ),
    b(
        "thumb_03_r",
        Some(80),
        Place::Hand {
            along: 0.1803,
            across: 0.0,
            up: 0.0,
        },
        Kind::Thumb,
        Side::Right,
    ),
    b(
        "index_metacarpal_r",
        Some(66),
        Place::Hand {
            along: 0.2013,
            across: -0.1390,
            up: 0.0225,
        },
        Kind::Finger,
        Side::Right,
    ),
    b(
        "index_01_r",
        Some(82),
        Place::Hand {
            along: 0.3434,
            across: 0.0,
            up: 0.0,
        },
        Kind::Finger,
        Side::Right,
    ),
    b(
        "index_02_r",
        Some(83),
        Place::Hand {
            along: 0.2384,
            across: 0.0,
            up: 0.0,
        },
        Kind::Finger,
        Side::Right,
    ),
    b(
        "index_03_r",
        Some(84),
        Place::Hand {
            along: 0.1516,
            across: 0.0,
            up: 0.0,
        },
        Kind::Finger,
        Side::Right,
    ),
    b(
        "wrist_inner_r",
        Some(66),
        Place::Origin,
        Kind::Helper,
        Side::Right,
    ),
    b(
        "wrist_outer_r",
        Some(66),
        Place::Origin,
        Kind::Helper,
        Side::Right,
    ),
    b(
        "weapon_r",
        Some(66),
        Place::Origin,
        Kind::Helper,
        Side::Right,
    ),
    b(
        "upperarm_twist_01_r",
        Some(57),
        Place::UpperArm(1.0 / 3.0),
        Kind::Twist,
        Side::Right,
    ),
    b(
        "upperarm_twistCor_01_r",
        Some(89),
        Place::Origin,
        Kind::Helper,
        Side::Right,
    ),
    b(
        "upperarm_twist_02_r",
        Some(57),
        Place::UpperArm(2.0 / 3.0),
        Kind::Twist,
        Side::Right,
    ),
    b(
        "upperarm_tricep_r",
        Some(91),
        Place::Origin,
        Kind::Helper,
        Side::Right,
    ),
    b(
        "upperarm_bicep_r",
        Some(91),
        Place::Origin,
        Kind::Helper,
        Side::Right,
    ),
    b(
        "upperarm_twistCor_02_r",
        Some(91),
        Place::Origin,
        Kind::Helper,
        Side::Right,
    ),
    b(
        "upperarm_correctiveRoot_r",
        Some(57),
        Place::Origin,
        Kind::Helper,
        Side::Right,
    ),
    b(
        "upperarm_bck_r",
        Some(95),
        Place::Origin,
        Kind::Helper,
        Side::Right,
    ),
    b(
        "upperarm_in_r",
        Some(95),
        Place::Origin,
        Kind::Helper,
        Side::Right,
    ),
    b(
        "upperarm_fwd_r",
        Some(95),
        Place::Origin,
        Kind::Helper,
        Side::Right,
    ),
    b(
        "upperarm_out_r",
        Some(95),
        Place::Origin,
        Kind::Helper,
        Side::Right,
    ),
    b(
        "clavicle_out_r",
        Some(56),
        Place::Origin,
        Kind::Helper,
        Side::Right,
    ),
    b(
        "clavicle_scap_r",
        Some(56),
        Place::Origin,
        Kind::Helper,
        Side::Right,
    ),
    b(
        "clavicle_pec_r",
        Some(6),
        Place::Origin,
        Kind::Helper,
        Side::Right,
    ),
    b(
        "spine_04_latissimus_l",
        Some(6),
        Place::Origin,
        Kind::Helper,
        Side::Left,
    ),
    b(
        "spine_04_latissimus_r",
        Some(6),
        Place::Origin,
        Kind::Helper,
        Side::Right,
    ),
    b(
        "clavicle_pec_l",
        Some(6),
        Place::Origin,
        Kind::Helper,
        Side::Left,
    ),
    b("thigh_r", Some(1), Place::HipSpan, Kind::Thigh, Side::Right),
    b(
        "calf_r",
        Some(106),
        Place::UpperLeg(1.0),
        Kind::Calf,
        Side::Right,
    ),
    b(
        "foot_r",
        Some(107),
        Place::LowerLeg(1.0),
        Kind::Foot,
        Side::Right,
    ),
    b("ball_r", Some(108), Place::Ball, Kind::Ball, Side::Right),
    b(
        "ankle_fwd_r",
        Some(108),
        Place::Origin,
        Kind::Helper,
        Side::Right,
    ),
    b(
        "ankle_bck_r",
        Some(108),
        Place::Origin,
        Kind::Helper,
        Side::Right,
    ),
    b(
        "calf_twist_02_r",
        Some(107),
        Place::LowerLeg(1.0 / 3.0),
        Kind::Twist,
        Side::Right,
    ),
    b(
        "calf_twistCor_02_r",
        Some(112),
        Place::Origin,
        Kind::Helper,
        Side::Right,
    ),
    b(
        "calf_twist_01_r",
        Some(107),
        Place::LowerLeg(2.0 / 3.0),
        Kind::Twist,
        Side::Right,
    ),
    b(
        "calf_correctiveRoot_r",
        Some(107),
        Place::Origin,
        Kind::Helper,
        Side::Right,
    ),
    b(
        "calf_kneeBack_r",
        Some(115),
        Place::Origin,
        Kind::Helper,
        Side::Right,
    ),
    b(
        "calf_knee_r",
        Some(115),
        Place::Origin,
        Kind::Helper,
        Side::Right,
    ),
    b(
        "thigh_twist_01_r",
        Some(106),
        Place::UpperLeg(1.0 / 3.0),
        Kind::Twist,
        Side::Right,
    ),
    b(
        "thigh_twistCor_01_r",
        Some(118),
        Place::Origin,
        Kind::Helper,
        Side::Right,
    ),
    b(
        "thigh_twist_02_r",
        Some(106),
        Place::UpperLeg(2.0 / 3.0),
        Kind::Twist,
        Side::Right,
    ),
    b(
        "thigh_twistCor_02_r",
        Some(120),
        Place::Origin,
        Kind::Helper,
        Side::Right,
    ),
    b(
        "thigh_correctiveRoot_r",
        Some(106),
        Place::Origin,
        Kind::Helper,
        Side::Right,
    ),
    b(
        "thigh_fwd_r",
        Some(122),
        Place::Origin,
        Kind::Helper,
        Side::Right,
    ),
    b(
        "thigh_bck_r",
        Some(122),
        Place::Origin,
        Kind::Helper,
        Side::Right,
    ),
    b(
        "thigh_out_r",
        Some(122),
        Place::Origin,
        Kind::Helper,
        Side::Right,
    ),
    b(
        "thigh_in_r",
        Some(122),
        Place::Origin,
        Kind::Helper,
        Side::Right,
    ),
    b(
        "thigh_bck_lwr_r",
        Some(122),
        Place::Origin,
        Kind::Helper,
        Side::Right,
    ),
    b(
        "thigh_fwd_lwr_r",
        Some(122),
        Place::Origin,
        Kind::Helper,
        Side::Right,
    ),
    b("thigh_l", Some(1), Place::HipSpan, Kind::Thigh, Side::Left),
    b(
        "calf_l",
        Some(129),
        Place::UpperLeg(1.0),
        Kind::Calf,
        Side::Left,
    ),
    b(
        "foot_l",
        Some(130),
        Place::LowerLeg(1.0),
        Kind::Foot,
        Side::Left,
    ),
    b("ball_l", Some(131), Place::Ball, Kind::Ball, Side::Left),
    b(
        "ankle_bck_l",
        Some(131),
        Place::Origin,
        Kind::Helper,
        Side::Left,
    ),
    b(
        "ankle_fwd_l",
        Some(131),
        Place::Origin,
        Kind::Helper,
        Side::Left,
    ),
    b(
        "calf_twist_02_l",
        Some(130),
        Place::LowerLeg(1.0 / 3.0),
        Kind::Twist,
        Side::Left,
    ),
    b(
        "calf_twistCor_02_l",
        Some(135),
        Place::Origin,
        Kind::Helper,
        Side::Left,
    ),
    b(
        "calf_twist_01_l",
        Some(130),
        Place::LowerLeg(2.0 / 3.0),
        Kind::Twist,
        Side::Left,
    ),
    b(
        "calf_correctiveRoot_l",
        Some(130),
        Place::Origin,
        Kind::Helper,
        Side::Left,
    ),
    b(
        "calf_kneeBack_l",
        Some(138),
        Place::Origin,
        Kind::Helper,
        Side::Left,
    ),
    b(
        "calf_knee_l",
        Some(138),
        Place::Origin,
        Kind::Helper,
        Side::Left,
    ),
    b(
        "thigh_twist_01_l",
        Some(129),
        Place::UpperLeg(1.0 / 3.0),
        Kind::Twist,
        Side::Left,
    ),
    b(
        "thigh_twistCor_01_l",
        Some(141),
        Place::Origin,
        Kind::Helper,
        Side::Left,
    ),
    b(
        "thigh_twist_02_l",
        Some(129),
        Place::UpperLeg(2.0 / 3.0),
        Kind::Twist,
        Side::Left,
    ),
    b(
        "thigh_twistCor_02_l",
        Some(143),
        Place::Origin,
        Kind::Helper,
        Side::Left,
    ),
    b(
        "thigh_correctiveRoot_l",
        Some(129),
        Place::Origin,
        Kind::Helper,
        Side::Left,
    ),
    b(
        "thigh_bck_l",
        Some(145),
        Place::Origin,
        Kind::Helper,
        Side::Left,
    ),
    b(
        "thigh_fwd_l",
        Some(145),
        Place::Origin,
        Kind::Helper,
        Side::Left,
    ),
    b(
        "thigh_out_l",
        Some(145),
        Place::Origin,
        Kind::Helper,
        Side::Left,
    ),
    b(
        "thigh_bck_lwr_l",
        Some(145),
        Place::Origin,
        Kind::Helper,
        Side::Left,
    ),
    b(
        "thigh_in_l",
        Some(145),
        Place::Origin,
        Kind::Helper,
        Side::Left,
    ),
    b(
        "thigh_fwd_lwr_l",
        Some(145),
        Place::Origin,
        Kind::Helper,
        Side::Left,
    ),
    b(
        "ik_hand_root",
        Some(0),
        Place::Origin,
        Kind::IkTarget,
        Side::Center,
    ),
    b(
        "ik_hand_gun",
        Some(152),
        Place::Mirrors("hand_r"),
        Kind::IkTarget,
        Side::Center,
    ),
    b(
        "ik_hand_r",
        Some(153),
        Place::Mirrors("hand_r"),
        Kind::IkTarget,
        Side::Right,
    ),
    b(
        "ik_hand_l",
        Some(153),
        Place::Mirrors("hand_l"),
        Kind::IkTarget,
        Side::Left,
    ),
    b(
        "ik_foot_root",
        Some(0),
        Place::Origin,
        Kind::IkTarget,
        Side::Center,
    ),
    b(
        "ik_foot_l",
        Some(156),
        Place::Mirrors("foot_l"),
        Kind::IkTarget,
        Side::Left,
    ),
    b(
        "ik_foot_r",
        Some(156),
        Place::Mirrors("foot_r"),
        Kind::IkTarget,
        Side::Right,
    ),
    b(
        "interaction",
        Some(0),
        Place::Origin,
        Kind::Helper,
        Side::Center,
    ),
    b(
        "center_of_mass",
        Some(0),
        Place::Origin,
        Kind::Helper,
        Side::Center,
    ),
];

/// The sockets a mannequin rig publishes: this engine's own six, plus the ALS set
/// under its own spellings.
///
/// Both families, deliberately. The engine's `hand_l` / `foot_r` / `head` / `back`
/// are what [`crate::template`] has always emitted and what
/// `inf_ecs::attach` composes against; the ALS six (`hand_l_socket`,
/// `hand_r_socket`, `FX_Foot_L`, `FX_Foot_R`, `head_socket`, `root_socket`) are what
/// a ported ALS animation blueprint and its footstep effects address. Publishing
/// one and not the other would make every port a rename.
static SOCKETS: [(&str, &str); 12] = [
    ("hand_l", "hand_l"),
    ("hand_r", "hand_r"),
    ("foot_l", "foot_l"),
    ("foot_r", "foot_r"),
    ("head", "head"),
    ("back", "spine_05"),
    ("hand_l_socket", "hand_l"),
    ("hand_r_socket", "hand_r"),
    ("FX_Foot_L", "foot_l"),
    ("FX_Foot_R", "foot_r"),
    ("head_socket", "head"),
    ("root_socket", "root"),
];

/// Which FK joint each IK handle follows when nothing is driving it.
///
/// `ik_hand_root` and `ik_foot_root` are deliberately absent: they are the subtree
/// anchors, and an anchor that chases a hand is not an anchor.
static IK_FOLLOW: [(&str, &str); 5] = [
    ("ik_hand_gun", "hand_r"),
    ("ik_hand_l", "hand_l"),
    ("ik_hand_r", "hand_r"),
    ("ik_foot_l", "foot_l"),
    ("ik_foot_r", "foot_r"),
];

/// Index of the bone named `name` in [`BONES`], if there is one.
fn index_of(name: &str) -> Option<u16> {
    BONES.iter().position(|b| b.name == name).map(|i| i as u16)
}

/// The joint a twist bone reads, the axis it reads about, and how much of the roll
/// it takes — [`crate::drive::TwistDriver`]'s law, applied to this rig's names.
///
/// See [`TwistDriver`] for why an upper segment's fractions are negative and a
/// lower segment's are positive. The magnitudes come straight out of the placement:
/// a bone at `p` along its segment shows `p` of the roll, so it adds `p` (lower) or
/// gives back `1 − p` (upper).
fn twist_rule(name: &str) -> Option<(String, [f32; 3], f32)> {
    const X: [f32; 3] = [1.0, 0.0, 0.0];
    const Y: [f32; 3] = [0.0, 1.0, 0.0];
    let side = if name.ends_with("_l") {
        "l"
    } else if name.ends_with("_r") {
        "r"
    } else {
        return None;
    };
    // `_01` is the bone nearest the joint that drives it and `_02` the far one;
    // which END of the segment that is depends on the segment, which is what the
    // source table below says.
    let (stem, near) = match name.strip_suffix(&format!("_twist_01_{side}")) {
        Some(s) => (s, true),
        None => (name.strip_suffix(&format!("_twist_02_{side}"))?, false),
    };
    let (source, axis) = match stem {
        "upperarm" => ("upperarm", X),
        "thigh" => ("thigh", Y),
        "lowerarm" => ("hand", X),
        "calf" => ("foot", Y),
        _ => return None,
    };
    // The bone nearest the driver carries two thirds and the far one a third,
    // which is exactly its own position along the segment. An UPPER segment is
    // rolled by its own joint and its twists must give that roll back, so their
    // fraction is negative; a LOWER segment is rolled by its distal child and its
    // twists add. See `TwistDriver` for the one sentence both fall out of.
    let magnitude = if near { 2.0 / 3.0 } else { 1.0 / 3.0 };
    let counters = matches!(stem, "upperarm" | "thigh");
    let fraction = if counters { -magnitude } else { magnitude };
    Some((format!("{source}_{side}"), axis, fraction))
}

/// **Build the mannequin rig** at `params`' proportions.
///
/// The caller ([`crate::build_template`]) has already refused degenerate
/// parameters; the only failure left is [`Skeleton::new`] rejecting what this
/// generator produced, which is a generator bug kept as a value so it can never be
/// a panic in a user's editor.
pub fn build_manny(params: &BodyParams) -> Result<SkeletonAsset, TemplateError> {
    let h = params.height_m;
    let hip_y = h * params.hip_height_ratio;
    let shoulder_y = h * params.shoulder_height_ratio;
    let head_y = h * params.head_height_ratio;
    let spine_rise = ((shoulder_y - hip_y) / 5.0) as f32;
    let neck_rise = ((head_y - shoulder_y) / 3.0) as f32;
    let arm_len = h * params.arm_length_ratio;
    let upper_arm = (arm_len * params.upper_limb_ratio) as f32;
    let lower_arm = (arm_len - arm_len * params.upper_limb_ratio) as f32;
    let upper_leg = (hip_y * params.upper_limb_ratio) as f32;
    let lower_leg = (hip_y - hip_y * params.upper_limb_ratio) as f32;
    let hand_len = (h * HAND_OF_HEIGHT) as f32;
    let ball_fwd = (h * BALL_OF_HEIGHT) as f32;
    let span = params.shoulder_width_m as f32;
    let hip_span = (params.hip_width_m * 0.5) as f32;

    let mut joints: Vec<Joint> = Vec::with_capacity(MANNY_JOINT_COUNT);
    let mut globals: Vec<Vec3> = Vec::with_capacity(MANNY_JOINT_COUNT);
    let mut roles: Vec<BoneRole> = Vec::with_capacity(MANNY_JOINT_COUNT);
    let mut twists: Vec<TwistDriver> = Vec::new();

    for (i, bone) in BONES.iter().enumerate() {
        // Left is −X and right is +X — the convention `template.rs` set and every
        // mirror rule in the engine reads.
        let sx = match bone.side {
            Side::Left => -1.0f32,
            Side::Right => 1.0,
            Side::Center => 0.0,
        };
        let base = bone
            .parent
            .and_then(|p| globals.get(p as usize).copied())
            .unwrap_or(Vec3::ZERO);
        let local = match bone.place {
            Place::Origin => Vec3::ZERO,
            Place::Pelvis => Vec3::new(0.0, hip_y as f32, 0.0),
            Place::SpineRise => Vec3::new(0.0, spine_rise, 0.0),
            Place::NeckRise => Vec3::new(0.0, neck_rise, 0.0),
            Place::Span(f) => Vec3::new(sx * f * span, 0.0, 0.0),
            Place::UpperArm(f) => Vec3::new(sx * f * upper_arm, 0.0, 0.0),
            Place::LowerArm(f) => Vec3::new(sx * f * lower_arm, 0.0, 0.0),
            Place::HipSpan => Vec3::new(sx * hip_span, 0.0, 0.0),
            Place::UpperLeg(f) => Vec3::new(0.0, -f * upper_leg, 0.0),
            Place::LowerLeg(f) => Vec3::new(0.0, -f * lower_leg, 0.0),
            Place::Ball => Vec3::new(0.0, 0.0, ball_fwd),
            Place::Hand { along, across, up } => {
                Vec3::new(sx * along * hand_len, up * hand_len, across * hand_len)
            }
            // Resolved against this bone's own parent, so a handle lands ON the
            // joint it marks however deep in the hierarchy that joint is.
            Place::Mirrors(target) => index_of(target)
                .and_then(|t| globals.get(t as usize).copied())
                .map(|g| g - base)
                .unwrap_or(Vec3::ZERO),
        };
        let global = base + local;
        joints.push(Joint {
            name: bone.name.to_string(),
            parent: bone.parent,
            // Identity rotation, unit scale: the inverse bind is the exact
            // negated translation and no matrix is ever inverted.
            inverse_bind: glam::Mat4::from_translation(-global).to_cols_array(),
            local_bind: JointTransform::from_trs(local, Quat::IDENTITY, Vec3::ONE),
        });
        globals.push(global);
        roles.push(BoneRole::new(i as u16, bone.kind, bone.side));
        if bone.kind == Kind::Twist {
            if let Some((source, axis, fraction)) = twist_rule(bone.name) {
                if let Some(s) = index_of(&source) {
                    twists.push(TwistDriver::new(i as u16, s, axis, fraction));
                }
            }
        }
    }

    let sockets: Vec<Socket> = SOCKETS
        .iter()
        .filter_map(|(name, joint)| index_of(joint).map(|j| Socket::new(*name, j)))
        .collect();
    let mut limits: Vec<JointLimit> = Vec::with_capacity(4);
    // **The elbows hinge about Y and the knees about X**, and the difference is
    // this rig's own bind pose: a leg runs down `−Y` so its bend axis is `X`, and
    // a T-posed arm runs out along `±X` so its bend axis is `Y`. See
    // `JointLimit::hinge_y` for the measurement that found it, and for how long an
    // elbow "hinge" about the forearm's own roll axis had been straightening arms
    // rather than limiting them. The elbow ranges are mirrored; a knee's are not.
    for (name, range) in [
        ("lowerarm_l", (ELBOW_RANGE_DEG.0, ELBOW_RANGE_DEG.1)),
        ("lowerarm_r", (-ELBOW_RANGE_DEG.1, -ELBOW_RANGE_DEG.0)),
    ] {
        if let Some(j) = index_of(name) {
            limits.push(JointLimit::hinge_y(j, range.0, range.1));
        }
    }
    for name in ["calf_l", "calf_r"] {
        if let Some(j) = index_of(name) {
            limits.push(JointLimit::hinge_x(j, KNEE_RANGE_DEG.0, KNEE_RANGE_DEG.1));
        }
    }
    // ── SK1b: a swing-twist cone on every finger bone ──
    //
    // The first *producer* of `ConeLimit`, whose slot SK1a spent a bump on and
    // which the SK1a audit then recorded as "authored and enforced by nothing".
    // A hand is the reason the type exists: three independent per-axis ranges
    // either forbid a legal finger pose at the corners or admit an illegal one at
    // the diagonals, and a box cannot say "this knuckle bends ninety degrees, in
    // one plane, and rolls almost not at all".
    //
    // The axis is the **bone's own direction**, so a curl — which is about an axis
    // perpendicular to the bone — registers as pure *swing* and is clamped by
    // `swing_deg`, while a roll registers as *twist* and is clamped to almost
    // nothing. See `crate::grip` for where the ranges come from.
    for (i, bone) in BONES.iter().enumerate() {
        if !matches!(bone.kind, Kind::Finger | Kind::Thumb) {
            continue;
        }
        // How far down its own digit this bone sits: metacarpal 0, knuckle 1, and
        // so on. Walked rather than parsed out of the name, because `_01` is a
        // spelling and the hierarchy is a fact.
        let mut depth = 0usize;
        let mut up = bone.parent;
        while let Some(p) = up {
            let parent = &BONES[p as usize];
            if !matches!(parent.kind, Kind::Finger | Kind::Thumb) {
                break;
            }
            depth += 1;
            up = parent.parent;
        }
        let flex = if bone.kind == Kind::Thumb {
            crate::grip::THUMB_FLEX_DEG
        } else {
            crate::grip::FINGER_FLEX_DEG
        };
        // The direction to this bone's own child, or — for a fingertip, which has
        // none — the direction that carried it here. Both are the bone's axis.
        let child = BONES.iter().position(|b| {
            b.parent == Some(i as u16) && matches!(b.kind, Kind::Finger | Kind::Thumb)
        });
        let dir = match child {
            Some(c) => joints[c].local_bind.translation_vec(),
            None => joints[i].local_bind.translation_vec(),
        };
        // **Normalized in `f64` and narrowed once**, not normalized in `f32`.
        // A bone that runs along one axis has direction `(-x, 0, 0)`, and
        // `-x / sqrt(x*x)` in `f32` is `-1.0` for some `x` and `-0.99999994` for
        // others — so two rigs of *different heights* would carry cone axes that
        // differ by an ulp, and `a_fitted_mannequin_keeps_every_table_it_arrived_with`
        // caught exactly that. In `f64` the same quotient rounds to `-1.0` on the
        // narrowing cast for every length a hand bone has.
        let dir = glam::DVec3::new(dir.x as f64, dir.y as f64, dir.z as f64);
        let len2 = dir.length_squared();
        if len2 <= 1.0e-24 {
            continue;
        }
        let d = dir / len2.sqrt();
        let axis = Vec3::new(d.x as f32, d.y as f32, d.z as f32);
        limits.push(JointLimit::cone_only(
            i as u16,
            ConeLimit {
                axis: axis.to_array(),
                // **Half a degree of margin**, deliberately: the solver's own
                // maximum is this same number, and a clamp applied exactly at the
                // boundary is a quaternion rebuild whose result is not bit-identical
                // to its input — so `GripReport::clamped` would count every fully
                // closed finger and stop meaning anything.
                swing_deg: flex[depth.min(flex.len() - 1)] + CONE_MARGIN_DEG,
                twist_deg: [
                    -crate::grip::FINGER_TWIST_DEG,
                    crate::grip::FINGER_TWIST_DEG,
                ],
            },
        ));
    }

    let mut ik_follow: Vec<IkFollow> = IK_FOLLOW
        .iter()
        .filter_map(|(handle, source)| Some(IkFollow::new(index_of(handle)?, index_of(source)?)))
        .collect();
    // Ascending by joint, because that is the table's invariant and the drive pass
    // walks it as it is (see `crate::drive::drive_ik_follow`). The list above is
    // written in the order a person reads it, not in the order the rig indexes it:
    // `ik_hand_r` is 154 and `ik_hand_l` is 155.
    ik_follow.sort_by_key(|f| f.joint);

    let skeleton = Skeleton::new(joints)?;
    let mut asset = SkeletonAsset::with_sockets(skeleton, sockets);
    asset.limits = limits;
    asset.roles = roles;
    asset.twists = twists;
    asset.ik_follow = ik_follow;
    // **The grip catalogue** (SK1c). SK1a shipped `SkeletonAsset::grips` empty on
    // every rig and said so; SK1b's finger solver read it and the only table in
    // the tree was a test fixture. It is generated now, off this rig's own hand
    // roles — see `crate::grip::grip_catalogue` for why a catalogue is a property
    // of the hand and can therefore be derived at all.
    //
    // After `roles` is set, because the catalogue is built from the role index.
    asset.grips = crate::grip::grip_catalogue(asset.role_index());
    Ok(asset)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pose::{global_transforms, Pose};
    use crate::roles::RoleIndex;

    fn manny() -> SkeletonAsset {
        build_manny(&BodyParams::default()).expect("the mannequin builds")
    }

    /// **161, and the invariant that makes the number mean something.**
    ///
    /// The count alone is satisfied by any 161 rows; what this asserts with it is
    /// that every parent precedes its child, that there is exactly one root, and
    /// that no name is used twice — the three things a hand-maintained table of
    /// this size actually gets wrong.
    #[test]
    fn the_manny_rig_is_one_hundred_and_sixty_one_bones() {
        let asset = manny();
        let sk = &asset.skeleton;
        assert_eq!(sk.len(), MANNY_JOINT_COUNT, "the mannequin is 161 bones");
        assert_eq!(sk.len(), asset.roles.len(), "one role row per joint");
        // Re-validated through the public door rather than trusted from the builder.
        Skeleton::new(sk.joints().to_vec()).expect("re-validates");
        assert_eq!(
            sk.joints().iter().filter(|j| j.parent.is_none()).count(),
            1,
            "more than one root"
        );
        for (i, j) in sk.joints().iter().enumerate() {
            if let Some(p) = j.parent {
                assert!(
                    (p as usize) < i,
                    "joint {i} `{}` names parent {p}, which does not precede it",
                    j.name
                );
            }
        }
        let mut names: Vec<&str> = sk.joints().iter().map(|j| j.name.as_str()).collect();
        names.sort_unstable();
        let unique = names.len();
        names.dedup();
        assert_eq!(names.len(), unique, "a bone name is used twice");
        assert_eq!(sk.joints()[0].name, "root");
    }

    /// **The hierarchy is the shipped asset's**, spot-checked at every place the
    /// source document and the asset disagree — so a future edit that "corrects"
    /// this table back towards the document fails here.
    #[test]
    fn the_parents_are_the_ones_the_asset_has_and_not_the_ones_the_document_claims() {
        let asset = manny();
        let sk = &asset.skeleton;
        let parent_of = |name: &str| -> Option<String> {
            let i = sk.index_of(name)? as usize;
            let p = sk.joints()[i].parent?;
            Some(sk.joints()[p as usize].name.clone())
        };
        for (child, want) in [
            ("pelvis", "root"),
            ("spine_01", "pelvis"),
            ("spine_05", "spine_04"),
            // The document omits `neck_02` entirely.
            ("neck_02", "neck_01"),
            ("head", "neck_02"),
            ("clavicle_l", "spine_05"),
            ("upperarm_l", "clavicle_l"),
            ("hand_l", "lowerarm_l"),
            ("index_01_l", "index_metacarpal_l"),
            ("thumb_01_r", "hand_r"),
            ("thigh_l", "pelvis"),
            ("ball_r", "foot_r"),
            ("calf_twist_01_l", "calf_l"),
            // The document hangs these off `spine_03`; the asset hangs them off
            // the root, and `ik_hand_l` off `ik_hand_gun` rather than the anchor.
            ("ik_hand_root", "root"),
            ("ik_hand_gun", "ik_hand_root"),
            ("ik_hand_l", "ik_hand_gun"),
            ("ik_foot_root", "root"),
            ("ik_foot_l", "ik_foot_root"),
            ("interaction", "root"),
            ("center_of_mass", "root"),
        ] {
            assert_eq!(
                parent_of(child).as_deref(),
                Some(want),
                "`{child}`'s parent"
            );
        }
        // …and the three bones the document invents are NOT here.
        for absent in ["ik_head", "ik_pelvis", "ik_spine"] {
            assert!(
                sk.index_of(absent).is_none(),
                "`{absent}` is in the document and not in the asset"
            );
        }
    }

    /// **Deform bones come before the `ik_*` subtrees** — the belt beside the role
    /// table's suspenders.
    ///
    /// `inf_ecs::pose::foot_joints` matches `contains("foot")` and takes the first
    /// hit per side. If a handle sorted first, foot IK would drive `ik_foot_l` and
    /// the "leg" it derived would be `[root, ik_foot_root, ik_foot_l]` — which
    /// solves perfectly and animates nothing.
    #[test]
    fn every_deform_bone_precedes_every_ik_handle() {
        let asset = manny();
        let sk = &asset.skeleton;
        let idx = RoleIndex::new(&asset.roles);
        let first_handle = sk
            .joints()
            .iter()
            .enumerate()
            .find(|(i, _)| idx.kind_of(*i as u16) == Some(BoneRoleKind::IkTarget))
            .map(|(i, _)| i)
            .expect("the rig has IK handles");
        let last_deform = sk
            .joints()
            .iter()
            .enumerate()
            .rev()
            .find(|(i, _)| idx.is_deform(*i as u16))
            .map(|(i, _)| i)
            .expect("the rig has deform bones");
        assert!(
            last_deform < first_handle,
            "`{}` (a deform bone at {last_deform}) sorts after the first handle \
             `{}` at {first_handle}",
            sk.joints()[last_deform].name,
            sk.joints()[first_handle].name
        );
        // The consequence, measured rather than argued: the first `foot`-ish bone
        // on each side is the ankle and not the handle.
        for (side, ankle) in [("_l", "foot_l"), ("_r", "foot_r")] {
            let first = sk
                .joints()
                .iter()
                .find(|j| j.name.contains("foot") && j.name.ends_with(side))
                .map(|j| j.name.as_str());
            assert_eq!(first, Some(ankle), "the first `foot` bone on {side}");
        }
    }

    /// The bind pose has the proportions it was asked for, and it is a **pure
    /// translation** pose — the template law, re-asserted on this generator.
    #[test]
    fn the_bind_pose_is_the_shape_the_parameters_asked_for() {
        let p = BodyParams::default();
        let asset = manny();
        let sk = &asset.skeleton;
        for j in sk.joints() {
            assert_eq!(
                j.local_bind.rotation,
                Quat::IDENTITY.to_array(),
                "{}",
                j.name
            );
            assert_eq!(j.local_bind.scale, [1.0, 1.0, 1.0], "{}", j.name);
        }
        let globals = global_transforms(sk, &Pose::rest(sk));
        let at =
            |name: &str| globals[sk.index_of(name).unwrap() as usize].transform_point3(Vec3::ZERO);
        assert!(
            at("foot_l").y.abs() < 1e-4,
            "left ankle at {}",
            at("foot_l").y
        );
        assert!(at("foot_r").y.abs() < 1e-4);
        assert!(at("ball_l").y.abs() < 1e-4, "the ball is on the ground too");
        assert!(at("ball_l").z > 0.0, "the ball is forward of the ankle");
        assert!((at("head").y as f64 - p.height_m * p.head_height_ratio).abs() < 1e-4);
        assert!((at("pelvis").y as f64 - p.height_m * p.hip_height_ratio).abs() < 1e-4);
        assert!((at("spine_05").y as f64 - p.height_m * p.shoulder_height_ratio).abs() < 1e-4);
        // Left is −X, right is +X, and the shoulders are a shoulder-width apart.
        assert!(at("upperarm_l").x < 0.0 && at("upperarm_r").x > 0.0);
        assert!(
            ((at("upperarm_r").x - at("upperarm_l").x) as f64 - p.shoulder_width_m).abs() < 1e-4
        );
        // The rest pose's skinning palette is the identity, which is what makes a
        // freshly generated rig draw its mesh unchanged.
        for m in crate::pose::skinning_matrices(sk, &Pose::rest(sk)) {
            assert!(m.abs_diff_eq(glam::Mat4::IDENTITY, 1e-5), "{m:?}");
        }
    }

    /// **The hand table is an exact mirror** (SK1a audit).
    ///
    /// Fifty-seven hand-maintained numbers — measured proportions, per the module
    /// docs, and the one place in this generator where a row is a number rather
    /// than a rule. A typo in one of them makes one hand subtly wrong and nothing
    /// else, which is precisely the defect no other arm here can see. The mirror
    /// is about the X plane, so `x` flips sign and `y`/`z` do not.
    #[test]
    fn the_two_hands_are_an_exact_mirror_of_each_other() {
        let asset = manny();
        let sk = &asset.skeleton;
        let local = |name: &str| sk.joints()[sk.index_of(name).unwrap() as usize].local_bind;
        let mut checked = 0usize;
        for j in sk.joints() {
            let Some(stem) = j.name.strip_suffix("_l") else {
                continue;
            };
            let Some(kind) = RoleIndex::new(&asset.roles).kind_of(sk.index_of(&j.name).unwrap())
            else {
                continue;
            };
            if !matches!(kind, BoneRoleKind::Finger | BoneRoleKind::Thumb) {
                continue;
            }
            let r = local(&format!("{stem}_r"));
            let l = j.local_bind;
            assert_eq!(l.translation[0], -r.translation[0], "`{}` x", j.name);
            assert_eq!(l.translation[1], r.translation[1], "`{}` y", j.name);
            assert_eq!(l.translation[2], r.translation[2], "`{}` z", j.name);
            checked += 1;
        }
        assert_eq!(checked, 19, "the hand has nineteen bones per side");
        // …and the middle chain really is one hand length, which is the
        // normalization the whole table is expressed against.
        let p = BodyParams::default();
        let chain: f32 = [
            "middle_metacarpal_l",
            "middle_01_l",
            "middle_02_l",
            "middle_03_l",
        ]
        .iter()
        .map(|n| local(n).translation[0].abs())
        .sum();
        let hand_len = (p.height_m * HAND_OF_HEIGHT) as f32;
        assert!(
            (chain - hand_len).abs() < 1.0e-5,
            "the middle finger spans {chain} against a hand length of {hand_len}"
        );
    }

    /// **The twist bones sit at thirds and are driven from the right end**, which
    /// is the structural fact the drive law rests on.
    #[test]
    fn the_twist_bones_sit_at_thirds_and_read_the_joint_nearest_them() {
        let asset = manny();
        let sk = &asset.skeleton;
        assert_eq!(
            asset.twists.len(),
            16,
            "two per segment, four segments, two sides"
        );
        let globals = global_transforms(sk, &Pose::rest(sk));
        let at =
            |name: &str| globals[sk.index_of(name).unwrap() as usize].transform_point3(Vec3::ZERO);
        // A forearm: the twists divide it into thirds, `_02` nearest the elbow.
        let (elbow, wrist) = (at("lowerarm_l"), at("hand_l"));
        let span = (wrist - elbow).length();
        for (name, want) in [
            ("lowerarm_twist_02_l", 1.0 / 3.0),
            ("lowerarm_twist_01_l", 2.0 / 3.0),
        ] {
            let f = (at(name) - elbow).length() / span;
            assert!((f - want).abs() < 1e-4, "`{name}` is at {f} of the forearm");
        }
        // An upper arm: `_01` nearest the shoulder, which is the driving end.
        let (shoulder, elbow) = (at("upperarm_l"), at("lowerarm_l"));
        let span = (elbow - shoulder).length();
        for (name, want) in [
            ("upperarm_twist_01_l", 1.0 / 3.0),
            ("upperarm_twist_02_l", 2.0 / 3.0),
        ] {
            let f = (at(name) - shoulder).length() / span;
            assert!(
                (f - want).abs() < 1e-4,
                "`{name}` is at {f} of the upper arm"
            );
        }
        // The drivers: an upper segment reads ITSELF and counters; a lower segment
        // reads its distal child and adds.
        let name_of = |j: u16| sk.joints()[j as usize].name.as_str();
        let driver = |joint: &str| {
            let j = sk.index_of(joint).unwrap();
            *asset.twists.iter().find(|d| d.joint == j).expect(joint)
        };
        let d = driver("upperarm_twist_01_l");
        assert_eq!(name_of(d.source), "upperarm_l");
        assert!((d.fraction + 2.0 / 3.0).abs() < 1e-6, "counters two thirds");
        let d = driver("lowerarm_twist_01_l");
        assert_eq!(name_of(d.source), "hand_l");
        assert!((d.fraction - 2.0 / 3.0).abs() < 1e-6, "adds two thirds");
        let d = driver("thigh_twist_02_r");
        assert_eq!(name_of(d.source), "thigh_r");
        assert!(d.fraction < 0.0);
        let d = driver("calf_twist_02_r");
        assert_eq!(name_of(d.source), "foot_r");
        assert!(d.fraction > 0.0);
        // Every driver names a real joint and a bone the table calls a twist.
        let idx = RoleIndex::new(&asset.roles);
        for d in &asset.twists {
            assert_eq!(idx.kind_of(d.joint), Some(BoneRoleKind::Twist), "{d:?}");
            assert!((d.source as usize) < sk.len());
            assert!(d.fraction.abs() > 0.0 && d.fraction.abs() <= 1.0);
        }
    }

    /// The IK handles start **on** the joints they mark, and follow them.
    #[test]
    fn the_ik_handles_are_authored_onto_their_sources() {
        let asset = manny();
        let sk = &asset.skeleton;
        assert_eq!(asset.ik_follow.len(), 5);
        let globals = global_transforms(sk, &Pose::rest(sk));
        for f in &asset.ik_follow {
            let a = globals[f.joint as usize].transform_point3(Vec3::ZERO);
            let b = globals[f.source as usize].transform_point3(Vec3::ZERO);
            assert!(
                (a - b).length() < 1e-5,
                "`{}` binds at {a:?} and `{}` at {b:?}",
                sk.joints()[f.joint as usize].name,
                sk.joints()[f.source as usize].name
            );
        }
        // The anchors are NOT followers.
        for anchor in ["ik_hand_root", "ik_foot_root"] {
            let j = sk.index_of(anchor).unwrap();
            assert!(asset.ik_follow.iter().all(|f| f.joint != j), "{anchor}");
        }
    }

    /// The role table covers the whole rig and puts each family where it belongs.
    #[test]
    fn every_bone_carries_a_role_and_the_helpers_are_marked_as_such() {
        let asset = manny();
        let idx = RoleIndex::new(&asset.roles);
        let sk = &asset.skeleton;
        for i in 0..sk.len() as u16 {
            assert!(
                idx.kind_of(i).is_some(),
                "`{}` has no role",
                sk.joints()[i as usize].name
            );
        }
        let kind = |name: &str| idx.kind_of(sk.index_of(name).unwrap()).unwrap();
        assert_eq!(kind("root"), BoneRoleKind::Root);
        assert_eq!(kind("pelvis"), BoneRoleKind::Pelvis);
        assert_eq!(kind("spine_03"), BoneRoleKind::Spine);
        assert_eq!(kind("neck_02"), BoneRoleKind::Neck);
        assert_eq!(kind("head"), BoneRoleKind::Head);
        assert_eq!(kind("clavicle_r"), BoneRoleKind::Clavicle);
        assert_eq!(kind("upperarm_l"), BoneRoleKind::UpperArm);
        assert_eq!(kind("lowerarm_l"), BoneRoleKind::LowerArm);
        assert_eq!(kind("hand_l"), BoneRoleKind::Hand);
        assert_eq!(kind("index_02_l"), BoneRoleKind::Finger);
        assert_eq!(kind("index_metacarpal_l"), BoneRoleKind::Finger);
        assert_eq!(kind("thumb_02_l"), BoneRoleKind::Thumb);
        assert_eq!(kind("thigh_r"), BoneRoleKind::Thigh);
        assert_eq!(kind("calf_r"), BoneRoleKind::Calf);
        assert_eq!(kind("foot_r"), BoneRoleKind::Foot);
        assert_eq!(kind("ball_r"), BoneRoleKind::Ball);
        assert_eq!(kind("calf_twist_01_r"), BoneRoleKind::Twist);
        assert_eq!(kind("ik_foot_root"), BoneRoleKind::IkTarget);
        assert_eq!(kind("weapon_r"), BoneRoleKind::Helper);
        assert_eq!(kind("thigh_correctiveRoot_l"), BoneRoleKind::Helper);
        assert_eq!(kind("center_of_mass"), BoneRoleKind::Helper);
        // Sides, on the three cases a suffix rule gets wrong.
        let side = |name: &str| idx.role_of(sk.index_of(name).unwrap()).unwrap().side;
        assert_eq!(side("spine_01"), BoneSide::Center);
        assert_eq!(side("ik_hand_gun"), BoneSide::Center);
        assert_eq!(side("thigh_bck_lwr_l"), BoneSide::Left);
        assert_eq!(side("upperarm_twist_01_r"), BoneSide::Right);
        // The census: **87** deform-or-driven bones and **74** helpers is what
        // the asset has, and a table that quietly re-labelled a family moves it.
        let helpers = (0..sk.len() as u16)
            .filter(|i| idx.kind_of(*i) == Some(BoneRoleKind::Helper))
            .count();
        assert_eq!(helpers, 74, "the corrective/helper census");
        let deform = (0..sk.len() as u16).filter(|i| idx.is_deform(*i)).count();
        assert_eq!(deform, 63, "the deform census");
        // **The WHOLE census, kind by kind** (SK1a audit). This comment already
        // claimed "a family cannot be re-labelled without moving a number here",
        // and three kinds were pinned. Measured: re-labelling `thigh_l` from
        // `Thigh` to `Spine` left every assertion in this arm green — the deform
        // total does not move (both kinds deform), the helper total does not
        // move, and neither `Thigh` nor `Spine` was counted. It was caught two
        // crates away, by `build_locomotion` failing to find a leg, which is a
        // long way from the table that lied. Every kind is counted now, and the
        // sum is asserted against the count so a row cannot be added to one
        // family and taken from another.
        let count = |k: BoneRoleKind| {
            (0..sk.len() as u16)
                .filter(|i| idx.kind_of(*i) == Some(k))
                .count()
        };
        use BoneRoleKind::*;
        let census: [(BoneRoleKind, usize); 18] = [
            (Root, 1),
            (Pelvis, 1),
            (Spine, 5),
            (Neck, 2),
            (Head, 1),
            (Clavicle, 2),
            (UpperArm, 2),
            (LowerArm, 2),
            (Hand, 2),
            (Finger, 32),
            (Thumb, 6),
            (Thigh, 2),
            (Calf, 2),
            (Foot, 2),
            (Ball, 2),
            (Twist, 16),
            (IkTarget, 7),
            (Helper, 74),
        ];
        for (kind, want) in census {
            assert_eq!(count(kind), want, "the {kind:?} census");
        }
        assert_eq!(
            census.iter().map(|(_, n)| n).sum::<usize>(),
            MANNY_JOINT_COUNT,
            "the census does not add up to the rig"
        );
        assert_eq!(63 + 16 + 7 + 74 + 1, MANNY_JOINT_COUNT);
    }

    /// Sockets ride real joints, both families are present, and a socket that
    /// shares a joint's name rides that joint.
    #[test]
    fn both_socket_families_are_published_and_ride_real_joints() {
        let asset = manny();
        let names: Vec<&str> = asset.sockets.iter().map(|s| s.name.as_str()).collect();
        for want in [
            "hand_l",
            "hand_r",
            "foot_l",
            "foot_r",
            "head",
            "back",
            "hand_l_socket",
            "hand_r_socket",
            "FX_Foot_L",
            "FX_Foot_R",
            "head_socket",
            "root_socket",
        ] {
            assert!(
                names.contains(&want),
                "socket `{want}` missing from {names:?}"
            );
        }
        for s in &asset.sockets {
            assert!(
                (s.joint as usize) < asset.skeleton.len(),
                "`{}` out of range",
                s.name
            );
            if let Some(j) = asset.skeleton.index_of(&s.name) {
                assert_eq!(s.joint, j, "socket `{}` rides the wrong joint", s.name);
            }
        }
        // The ALS spellings ride the joints their UE originals ride.
        let joint_of = |n: &str| asset.sockets.iter().find(|s| s.name == n).unwrap().joint;
        assert_eq!(
            joint_of("FX_Foot_L"),
            asset.skeleton.index_of("foot_l").unwrap()
        );
        assert_eq!(
            joint_of("root_socket"),
            asset.skeleton.index_of("root").unwrap()
        );
        assert_eq!(
            joint_of("back"),
            asset.skeleton.index_of("spine_05").unwrap()
        );
    }

    /// **Two families of limit and nothing else**: hinges on the four joints that
    /// are hinges, cones on every finger bone (SK1b), and no third kind.
    ///
    /// The census is asserted as an identity rather than a count, because a count
    /// is satisfied by the wrong forty-two rows as happily as by the right ones —
    /// the M3 lesson, on the table this wave grew.
    #[test]
    fn the_hinges_are_the_elbows_and_the_knees_and_the_cones_are_the_fingers() {
        let asset = manny();
        let sk = &asset.skeleton;
        let roles = asset.role_index();
        let name = |j: u16| sk.joints()[j as usize].name.as_str();

        let hinges: Vec<&str> = asset
            .limits
            .iter()
            .filter(|l| l.cone.is_none())
            .map(|l| name(l.joint))
            .collect();
        assert_eq!(
            hinges,
            ["lowerarm_l", "lowerarm_r", "calf_l", "calf_r"],
            "the hinge set moved"
        );
        // **The elbows hinge about Y and the knees about X** — the SK1b
        // correction. An elbow hinged about X on a T-posed arm names the
        // forearm's own roll axis and straightens the arm instead of limiting it
        // (0.484 m of missed reach, measured); the knees are unaffected, because
        // a leg runs down `-Y`.
        for (name, free, mirrored) in [
            ("lowerarm_l", 1usize, false),
            ("lowerarm_r", 1, true),
            ("calf_l", 0, true),
            ("calf_r", 0, true),
        ] {
            let j = sk.index_of(name).unwrap();
            let l = asset.limits.iter().find(|l| l.joint == j).unwrap();
            for a in 0..3 {
                assert_eq!(
                    l.is_free(a),
                    a == free,
                    "{name}: axis {a} freedom is wrong — a hinge has exactly one"
                );
            }
            // Which SIDE of zero the range sits on decides which way the joint
            // folds, and an elbow's is mirrored while a knee's is not.
            assert_eq!(
                l.min_deg[free] < 0.0,
                mirrored,
                "{name}: the range is on the wrong side of zero ({:?}..{:?})",
                l.min_deg[free],
                l.max_deg[free]
            );
            assert!(l.cone.is_none(), "{name}: no cone is authored on a hinge");
        }

        // The cones are EXACTLY the digit bones — every one of them, and nothing
        // else. Mutation: dropping the `Kind::Thumb` arm of the emitter leaves
        // six names on the left that are not on the right.
        let coned: Vec<u16> = asset
            .limits
            .iter()
            .filter(|l| l.cone.is_some())
            .map(|l| l.joint)
            .collect();
        let digits: Vec<u16> = (0..sk.len() as u16)
            .filter(|j| {
                matches!(
                    roles.kind_of(*j),
                    Some(BoneRoleKind::Finger | BoneRoleKind::Thumb)
                )
            })
            .collect();
        let mut sorted = coned.clone();
        sorted.sort_unstable();
        assert_eq!(
            sorted,
            digits,
            "the cone set is not the digit set: {:?} against {:?}",
            coned.iter().map(|j| name(*j)).collect::<Vec<_>>(),
            digits.iter().map(|j| name(*j)).collect::<Vec<_>>()
        );
        assert_eq!(digits.len(), 38, "nineteen digit bones per hand");

        // A cone's axis is the BONE's direction, which is what makes a curl read
        // as swing and a roll read as twist. Asserted where it is checkable: the
        // knuckle's cone axis points at its own child.
        let knuckle = sk.index_of("middle_01_r").unwrap();
        let phalanx = sk.index_of("middle_02_r").unwrap();
        let cone = asset
            .limits
            .iter()
            .find(|l| l.joint == knuckle)
            .and_then(|l| l.cone)
            .expect("a knuckle carries a cone");
        let want = sk.joints()[phalanx as usize]
            .local_bind
            .translation_vec()
            .normalize();
        assert!(
            (Vec3::from_array(cone.axis) - want).length() < 1.0e-5,
            "the cone axis is {:?}, the bone points {want:?}",
            cone.axis
        );
        assert!(
            (cone.swing_deg - (crate::grip::FINGER_FLEX_DEG[1] + CONE_MARGIN_DEG)).abs() < 1.0e-4,
            "a knuckle's cone is {} deg",
            cone.swing_deg
        );
        assert_eq!(
            cone.twist_deg,
            [
                -crate::grip::FINGER_TWIST_DEG,
                crate::grip::FINGER_TWIST_DEG
            ]
        );
    }

    /// Deterministic, and a function of its input: same params, same bytes;
    /// different params, different bytes.
    #[test]
    fn the_same_params_generate_the_same_bytes() {
        let a = build_manny(&BodyParams::default()).unwrap();
        let b = build_manny(&BodyParams::default()).unwrap();
        assert_eq!(a, b);
        let ea = inf_asset::encode(&a).unwrap();
        assert_eq!(ea, inf_asset::encode(&b).unwrap(), "not byte-reproducible");
        let back: SkeletonAsset = inf_asset::decode(&ea).unwrap();
        assert_eq!(back, a, "the tail tables survived the wire");
        assert_eq!(back.roles.len(), MANNY_JOINT_COUNT);
        assert_eq!(back.twists.len(), 16);
        assert_eq!(back.ik_follow.len(), 5);
        let tall = BodyParams {
            height_m: 2.4,
            ..BodyParams::default()
        };
        assert_ne!(
            inf_asset::encode(&build_manny(&tall).unwrap()).unwrap(),
            ea,
            "the generator ignored its input"
        );
    }
}
