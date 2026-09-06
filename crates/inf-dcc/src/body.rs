//! **The procedural body mesh** (SK1b): a humanoid, generated from its own rig.
//!
//! # What this replaces, and why a second generator exists
//!
//! `inf_editor_core::character::block_body_mesh` puts **one box on every deform
//! bone** and the boxes do not touch: a mannequin is a pile of disconnected
//! cuboids, and at 161 bones the SK1a wave had to add a role filter just to stop
//! it drawing a box on every corrective. It is the right shape for a *preview*
//! and it is not a character. This is:
//!
//! * limbs are **welded tapered tubes** swept along a whole chain — the shoulder,
//!   elbow and wrist are one surface with a continuous silhouette, not three
//!   boxes;
//! * hands are **real hands** — a palm slab and five tapered digits, each one
//!   following its own metacarpal and phalanges;
//! * the head is a **tapered cranium with a jaw**, a nose and two ears rather
//!   than a cube.
//!
//! # The honest bound: shells, not one skin
//!
//! Each *chain* is welded along its own length and is a closed manifold shell.
//! The shells **interpenetrate** at the girdles — an arm's tube starts inside the
//! torso's — rather than being stitched into a single surface, because stitching
//! them is a boolean union and this kernel does not have one (the same reason
//! `P23`'s bake merges rather than welds). A silhouette reads correctly and a
//! cross-section at the shoulder would show two surfaces. Stated here rather
//! than discovered later.
//!
//! # Everything is a proportion
//!
//! Not one absolute dimension appears below. Every radius is a fraction of the
//! rig's own **measured** height and every centre is a joint the rig already
//! placed, so a 1.2 m character and a 2.4 m one get the same body at their own
//! scale — the [`crate::autofit`] discipline, and [`inf_anim::manny`]'s.
//!
//! # Determinism
//!
//! `f64` throughout, `inf_math::psin64`/`pcos64` for the ring angles, `BTreeSet`
//! for the patch sets, and **every face's winding is measured rather than
//! assumed** — a loop whose Newell normal points at the shell's interior is
//! reversed before it is added. That is the `block_body_mesh` rule, kept, because
//! a swept frame's handedness flips with the direction it is swept in and a
//! generator that gets it right by construction is a generator that gets it wrong
//! the first time a limb points the other way.

use std::collections::BTreeSet;

use glam::DVec3;
use inf_anim::{BoneRoleKind, BoneSide, RoleIndex, Skeleton, SkeletonAsset};
use inf_math::{pcos64, psin64};

use crate::ops::OpError;
use crate::topo::{CornerData, Mesh, VertId};

/// How finely the body is tessellated.
///
/// Defaults chosen so the whole character lands in the low thousands of
/// triangles — a starter body is a thing you *replace*, and a heavy one is a
/// heavy heat solve and a heavy skinning palette upload for geometry nobody
/// ships.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BodyOptions {
    /// Sides on a limb's cross-section.
    pub limb_segments: usize,
    /// Sides on the torso's cross-section.
    pub torso_segments: usize,
    /// Sides on a finger's cross-section.
    pub finger_segments: usize,
    /// Sides on the head's cross-section.
    pub head_segments: usize,
    /// Rings between the head's two poles.
    pub head_rings: usize,
}

impl Default for BodyOptions {
    fn default() -> Self {
        Self {
            limb_segments: 32,
            torso_segments: 48,
            finger_segments: 10,
            head_segments: 44,
            head_rings: 26,
        }
    }
}

/// Why a body could not be generated.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum BodyError {
    /// The rig carries no [`inf_anim::BoneRole`] table, so nothing here knows
    /// which bone is a thigh.
    ///
    /// Refused rather than guessed at: this generator is for a rig **this
    /// engine** produced, every one of which carries a table since SK1a, and a
    /// name heuristic that placed a torso on the wrong chain would make a body
    /// that is wrong rather than absent.
    #[error(
        "this rig carries no bone-role table, so a body cannot be built from it \
         (generate it with `BodyPlan::Biped`, or use the block mannequin)"
    )]
    NoRoles,
    /// A part the body cannot be built without is missing from the rig.
    #[error("this rig has no {0}, so a humanoid body cannot be built from it")]
    MissingPart(&'static str),
    /// The kernel refused a face or a patch — a generator bug, kept as a value
    /// so it can never be a panic in a user's editor.
    #[error("the generated topology is not manifold: {0}")]
    Topology(#[from] OpError),
}

/// How far the crown of the head sits **above** the rig's `head` joint, as a
/// fraction of height.
///
/// The rig's own arithmetic makes this the number that decides how tall the
/// finished body is: [`inf_anim::manny`] places `head` at `head_height_ratio` of
/// the requested height (0.93 by default) and the ankle at exactly `y = 0`, so
/// the body spans `head_y + CROWN_ABOVE_HEAD·h`. At 0.070 that is exactly `h`.
const CROWN_ABOVE_HEAD: f64 = 0.070;

/// How far the chin sits **below** the `head` joint. With the constant above it
/// makes a head 0.115 of a standing height from chin to crown — 20 cm on a 1.75 m
/// adult, which is what one is.
const CHIN_BELOW_HEAD: f64 = 0.045;

/// `v > 0.0`, written once so the **NaN-rejecting** form reads as intent rather
/// than as a negated comparison.
///
/// `inf_anim::positive`'s discipline, and its reason: `v <= 0.0` is **false** for
/// a NaN and would let one through to arithmetic that manufactures a body of
/// NaNs, and `!(v > 0.0)` is the only spelling that rejects it — which clippy
/// distrusts for exactly the reason it is right here.
#[inline]
fn positive(v: f64) -> bool {
    v > 0.0
}

/// A cross-section of a swept shell: where it is, and how wide and deep.
///
/// Two radii rather than one, because nothing on a body is round: a torso is
/// wider than it is deep, a palm is a slab, and a foot is flatter still.
#[derive(Clone, Copy, Debug)]
struct Ring {
    at: DVec3,
    half_width: f64,
    half_depth: f64,
    /// **Which bone made this ring** — the seed weight for every vertex on it.
    ///
    /// Not decoration and not a shortcut past the weight solver: it is the
    /// *prior*. `crate::heat`'s solve is documented as "additive evidence rather
    /// than a reset — a vertex no bone can see keeps its current weights", and a
    /// generated body is the one mesh in the world where a correct answer is
    /// known before the solve runs. Without it the 4 % of vertices the visibility
    /// oracle cannot reach keep `VertWeights::RIGID`, which is *all of joint 0* —
    /// the rig's root — and a handful of triangles stay behind when the character
    /// walks away.
    joint: u16,
}

/// What closes the end of a swept shell.
#[derive(Clone, Copy, Debug)]
enum Cap {
    /// A flat n-gon across the last ring — a shoulder that starts inside a torso.
    Flat,
    /// A fan to a single point — a fingertip, the crown of a head.
    Point(DVec3),
}

/// What one generated body is made of.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct BodyReport {
    /// **Which bone made each vertex** — the rigid prior a weight solve refines.
    /// Ascending by vertex, one entry per vertex the generator emitted.
    pub seed: Vec<(VertId, u16)>,
    /// How many welded shells the body is (see the module docs' honest bound).
    pub shells: usize,
}

/// **Build a humanoid body mesh for `rig`.**
///
/// The rig's own bind pose decides every position and its role table decides
/// every part; the only free parameters are how finely it is tessellated.
///
/// The result is **unbound**, and it carries a [`BodyReport::seed`] rather than
/// weights: binding and solving are [`crate::heat`]'s job, and keeping them apart
/// is what lets the same mesh be re-weighted onto a fitted rig without being
/// regenerated.
pub fn body_mesh(rig: &SkeletonAsset, opts: &BodyOptions) -> Result<(Mesh, BodyReport), BodyError> {
    let roles = rig.role_index();
    if roles.is_empty() {
        return Err(BodyError::NoRoles);
    }
    let skeleton = &rig.skeleton;
    let at = bind_globals(skeleton);
    let h = reference_height(&at, roles);
    if !positive(h) {
        return Err(BodyError::MissingPart("a body with any height at all"));
    }

    let mut mesh = Mesh::new();
    mesh.set_material_slots(vec!["Skin".to_string()]);
    mesh.set_slot_name(Some(0), "Skin".to_string());
    // Every shell takes its own horizontal band of the atlas, so two parts never
    // share a texel. Not a packed atlas — a `u` band per shell, which is the
    // cheapest thing that is not simply "everything on top of everything".
    let mut b = Build {
        band: BandAllocator::default(),
        seed: Vec::new(),
        shells: 0,
    };

    let shape = Shape {
        skeleton,
        roles,
        at: &at,
        h,
        opts,
    };
    torso(&mut mesh, &mut b, &shape)?;
    head(&mut mesh, &mut b, &shape)?;
    for side in [BoneSide::Left, BoneSide::Right] {
        arm(&mut mesh, &mut b, &shape, side)?;
        hand(&mut mesh, &mut b, &shape, side)?;
        leg(&mut mesh, &mut b, &shape, side)?;
        foot(&mut mesh, &mut b, &shape, side)?;
    }
    b.seed.sort_unstable();
    Ok((
        mesh,
        BodyReport {
            seed: b.seed,
            shells: b.shells,
        },
    ))
}

/// **What every part of the body is built against** — the rig, read once.
///
/// A struct rather than five parameters threaded through eight functions, and
/// not only because clippy counts: `roles`, `at` and `h` are three readings of
/// one rig and passing them separately is what lets a caller pass two rigs'
/// worth by accident.
struct Shape<'a> {
    skeleton: &'a Skeleton,
    roles: RoleIndex<'a>,
    /// Every joint's bind-pose position, in model space.
    at: &'a [DVec3],
    /// The rig's measured standing height — every proportion below is a fraction
    /// of it.
    h: f64,
    opts: &'a BodyOptions,
}

/// The state a body build carries between its parts.
struct Build {
    band: BandAllocator,
    seed: Vec<(VertId, u16)>,
    shells: usize,
}

/// Hands out a `u` band per shell so no two parts share a texel.
#[derive(Default)]
struct BandAllocator {
    next: usize,
}

impl BandAllocator {
    /// The number of bands the body is divided into. A constant rather than a
    /// count, because the bands are handed out while the body is being built and
    /// the total is not known until it is finished — and a `u` that depends on
    /// how many parts a rig happened to have would make two rigs' atlases
    /// incomparable.
    const BANDS: usize = 24;

    fn take(&mut self) -> (f64, f64) {
        let i = self.next.min(Self::BANDS - 1);
        self.next += 1;
        let w = 1.0 / Self::BANDS as f64;
        (i as f64 * w, (i + 1) as f64 * w)
    }
}

/// Every joint's bind-pose position, in model space.
fn bind_globals(skeleton: &Skeleton) -> Vec<DVec3> {
    let joints = skeleton.joints();
    let mut out: Vec<DVec3> = Vec::with_capacity(joints.len());
    for j in joints {
        let l = DVec3::new(
            j.local_bind.translation[0] as f64,
            j.local_bind.translation[1] as f64,
            j.local_bind.translation[2] as f64,
        );
        let base = j.parent.map(|p| out[p as usize]).unwrap_or(DVec3::ZERO);
        out.push(base + l);
    }
    out
}

/// The rig's **measured** height: the vertical span of its deform bones.
///
/// Measured and not asked for, because this generator is handed a rig and not a
/// [`inf_anim::BodyParams`] — and because a *fitted* rig's proportions are no
/// longer the ones its parameters named ([`crate::autofit`] moves joints).
fn reference_height(at: &[DVec3], roles: RoleIndex<'_>) -> f64 {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for (i, p) in at.iter().enumerate() {
        if roles.is_deform(i as u16) {
            lo = lo.min(p.y);
            hi = hi.max(p.y);
        }
    }
    if hi <= lo {
        return 0.0;
    }
    // The head JOINT is not the top of the head, and the ankle is not the sole.
    // A rig's deform span is about 0.93 of a standing height on every plan this
    // engine generates; the factor is here so every proportion below reads as a
    // fraction of a height rather than of a joint span.
    (hi - lo) / 0.93
}

// ── the parts ───────────────────────────────────────────────────────────────

fn chain_of(roles: RoleIndex<'_>, kind: BoneRoleKind, side: BoneSide) -> Vec<u16> {
    roles.all(kind, side)
}

/// Pelvis → the whole spine → the neck: one welded tube.
fn torso(mesh: &mut Mesh, b: &mut Build, s: &Shape<'_>) -> Result<(), BodyError> {
    let (roles, at, h, opts) = (s.roles, s.at, s.h, s.opts);
    let pelvis = roles
        .first(BoneRoleKind::Pelvis, BoneSide::Center)
        .ok_or(BodyError::MissingPart("a pelvis"))?;
    let spine = chain_of(roles, BoneRoleKind::Spine, BoneSide::Center);
    if spine.is_empty() {
        return Err(BodyError::MissingPart("a spine"));
    }
    let neck = chain_of(roles, BoneRoleKind::Neck, BoneSide::Center);

    // The profile, from the hips up: a waist that narrows and a chest that
    // widens to carry the shoulders. Sampled along the chain by fraction, so a
    // five-segment spine and a two-segment one get the same silhouette.
    //
    // The numbers are half-widths and half-depths as fractions of height, and
    // they are anthropometry rather than taste: a 1.75 m adult is about 36 cm
    // across the hips and 37 cm across the chest, which is where 0.075 and 0.082
    // come from. They were **measured against the generated body**, not guessed:
    // the first pass used numbers half again as large and produced a 34 cm-wide
    // neck on a 25 cm-wide head.
    const PROFILE: [(f64, f64); 5] = [
        (0.076, 0.056), // hips
        (0.063, 0.047), // waist
        (0.071, 0.052),
        (0.082, 0.058), // chest
        (0.058, 0.046), // the shoulders' top, tapering into the neck
    ];
    let mut rings: Vec<Ring> = Vec::with_capacity(1 + spine.len() + neck.len());
    // The pelvis ring sits BELOW the pelvis joint, at the hip girdle, so the
    // torso closes over the tops of the legs instead of ending at a joint that
    // is halfway up the body.
    let hip_drop = h * 0.030;
    rings.push(Ring {
        at: at[pelvis as usize] - DVec3::Y * hip_drop,
        half_width: h * PROFILE[0].0,
        half_depth: h * PROFILE[0].1,
        joint: pelvis,
    });
    let n = spine.len().max(1);
    for (i, &j) in spine.iter().enumerate() {
        let t = if n == 1 {
            0.0
        } else {
            i as f64 / (n - 1) as f64
        };
        let (w, d) = sample_profile(&PROFILE, t);
        rings.push(Ring {
            at: at[j as usize],
            half_width: h * w,
            half_depth: h * d,
            joint: j,
        });
    }
    for &j in &neck {
        rings.push(Ring {
            at: at[j as usize],
            half_width: h * 0.030,
            half_depth: h * 0.029,
            joint: j,
        });
    }
    // …and a stub above the last neck joint, so the neck does not end in a flat
    // disc exactly where the head shell begins.
    if let Some(&last) = neck.last() {
        let up = (at[last as usize] - rings[rings.len() - 2].at).normalize_or_zero();
        rings.push(Ring {
            at: at[last as usize] + up * (h * 0.02),
            half_width: h * 0.029,
            half_depth: h * 0.028,
            joint: last,
        });
    }
    sweep(mesh, b, &rings, opts.torso_segments, Cap::Flat, Cap::Flat)
}

/// The head: a tapered cranium, a nose, two ears.
fn head(mesh: &mut Mesh, b: &mut Build, s: &Shape<'_>) -> Result<(), BodyError> {
    let (roles, at, h, opts) = (s.roles, s.at, s.h, s.opts);
    let head = roles
        .first(BoneRoleKind::Head, BoneSide::Center)
        .ok_or(BodyError::MissingPart("a head"))?;
    let base = at[head as usize];
    // A head is about 15.5 cm across and 20 cm from chin to crown on a 1.75 m
    // adult. The vertical numbers are **load-bearing**, not decoration: this rig
    // puts its `head` joint at 0.93 of the standing height and its ankle exactly
    // on the ground, so `CROWN_ABOVE_HEAD` is what decides whether a 1.75 m
    // character is 1.75 m tall. `the_body_stands_as_tall_as_its_rig_asked_for`
    // is the arm that keeps it honest.
    let r = h * 0.045;
    let height = h * (CROWN_ABOVE_HEAD + CHIN_BELOW_HEAD);
    // The cranium's radius profile from the jaw to the crown: an ellipse, which
    // is a `sqrt`, narrowed at the bottom so the head has a jaw rather than a
    // chin the width of its skull.
    let rings_n = opts.head_rings.max(3);
    let mut rings: Vec<Ring> = Vec::with_capacity(rings_n);
    for i in 0..rings_n {
        let t = (i + 1) as f64 / (rings_n + 1) as f64;
        // `t` from 0 (jaw) to 1 (crown); the ellipse gives the round part and the
        // jaw factor takes the bottom third in.
        let e = (1.0 - (2.0 * t - 1.0) * (2.0 * t - 1.0)).max(0.0).sqrt();
        let jaw = 0.62 + 0.38 * t.min(0.5) * 2.0;
        rings.push(Ring {
            at: base + DVec3::Y * (t * height - h * CHIN_BELOW_HEAD),
            half_width: r * e * jaw,
            half_depth: r * 1.15 * e * jaw,
            joint: head,
        });
    }
    let chin = base - DVec3::Y * (h * CHIN_BELOW_HEAD);
    let crown = base + DVec3::Y * (h * CROWN_ABOVE_HEAD);
    sweep(
        mesh,
        b,
        &rings,
        opts.head_segments,
        Cap::Point(chin),
        Cap::Point(crown),
    )?;

    // The nose: a short taper forward from the middle of the face. `+Z` is
    // forward in this engine's model space (`manny::Place::Ball`).
    let face_y = base.y + h * (CROWN_ABOVE_HEAD * 0.10);
    let nose_root = DVec3::new(base.x, face_y, base.z + r * 0.95);
    let nose_tip = DVec3::new(base.x, face_y - h * 0.012, base.z + r * 1.30);
    sweep(
        mesh,
        b,
        &[
            Ring {
                at: nose_root,
                half_width: h * 0.011,
                half_depth: h * 0.008,
                joint: head,
            },
            Ring {
                at: nose_tip,
                half_width: h * 0.009,
                half_depth: h * 0.008,
                joint: head,
            },
        ],
        6,
        Cap::Flat,
        Cap::Point(nose_tip + DVec3::new(0.0, -h * 0.004, h * 0.006)),
    )?;

    // Two ears, flattened against the sides of the head.
    for sx in [-1.0f64, 1.0] {
        let ear = DVec3::new(
            base.x + sx * r * 0.90,
            face_y + h * 0.008,
            base.z - r * 0.10,
        );
        sweep(
            mesh,
            b,
            &[
                Ring {
                    at: ear,
                    half_width: h * 0.004,
                    half_depth: h * 0.015,
                    joint: head,
                },
                Ring {
                    at: ear + DVec3::X * (sx * h * 0.010),
                    half_width: h * 0.003,
                    half_depth: h * 0.011,
                    joint: head,
                },
            ],
            6,
            Cap::Flat,
            Cap::Flat,
        )?;
    }
    Ok(())
}

/// Shoulder → elbow → wrist, one welded tube.
fn arm(mesh: &mut Mesh, b: &mut Build, s: &Shape<'_>, side: BoneSide) -> Result<(), BodyError> {
    let (roles, at, h, opts) = (s.roles, s.at, s.h, s.opts);
    let (Some(upper), Some(lower), Some(hand)) = (
        roles.first(BoneRoleKind::UpperArm, side),
        roles.first(BoneRoleKind::LowerArm, side),
        roles.first(BoneRoleKind::Hand, side),
    ) else {
        return Err(BodyError::MissingPart("an arm"));
    };
    // A shoulder cap that starts inside the torso, so the two shells overlap
    // rather than leaving a gap at the joint.
    let shoulder = at[upper as usize];
    let inward = (at[upper as usize] - at[lower as usize]).normalize_or_zero();
    let rings = [
        Ring {
            at: shoulder + inward * (h * 0.030),
            half_width: h * 0.037,
            half_depth: h * 0.037,
            joint: upper,
        },
        Ring {
            at: shoulder,
            half_width: h * 0.035,
            half_depth: h * 0.035,
            joint: upper,
        },
        Ring {
            at: at[lower as usize],
            half_width: h * 0.027,
            half_depth: h * 0.027,
            joint: lower,
        },
        Ring {
            at: at[hand as usize],
            half_width: h * 0.021,
            half_depth: h * 0.019,
            joint: hand,
        },
    ];
    sweep(mesh, b, &rings, opts.limb_segments, Cap::Flat, Cap::Flat)
}

/// A palm slab and five tapered digits.
fn hand(mesh: &mut Mesh, b: &mut Build, s: &Shape<'_>, side: BoneSide) -> Result<(), BodyError> {
    let (skeleton, roles, at, h, opts) = (s.skeleton, s.roles, s.at, s.h, s.opts);
    let wrist = roles
        .first(BoneRoleKind::Hand, side)
        .ok_or(BodyError::MissingPart("a hand"))?;
    let Some(geometry) = inf_anim::hand_of(skeleton, roles, wrist) else {
        // A rig with a hand joint and no digits gets an arm and no hand, which
        // is the honest answer rather than a guessed-at mitten.
        return Ok(());
    };
    // The palm: from the wrist to the far end of the metacarpals, flattened
    // along the palm normal. The knuckle line's own extent decides its width,
    // so a broad hand gets a broad palm.
    let mut knuckle_lo = f64::INFINITY;
    let mut knuckle_hi = f64::NEG_INFINITY;
    let mut knuckle_end = DVec3::ZERO;
    let mut counted = 0usize;
    let spread = DVec3::new(
        geometry.spread[0] as f64,
        geometry.spread[1] as f64,
        geometry.spread[2] as f64,
    );
    for digit in inf_anim::Digit::ALL {
        let Some(chain) = geometry.finger(digit) else {
            continue;
        };
        if digit == inf_anim::Digit::Thumb {
            continue;
        }
        // The knuckle is where a finger's second bone starts (past the
        // metacarpal), or the finger's own root on a rig with no metacarpals.
        let k = chain.joints[1.min(chain.joints.len() - 1)] as usize;
        let s = (at[k] - at[wrist as usize]).dot(spread);
        knuckle_lo = knuckle_lo.min(s);
        knuckle_hi = knuckle_hi.max(s);
        knuckle_end += at[k];
        counted += 1;
    }
    if counted > 0 {
        let knuckles = knuckle_end / counted as f64;
        let half_span = ((knuckle_hi - knuckle_lo) * 0.5 + h * 0.006).max(h * 0.010);
        let across = (knuckles - at[wrist as usize]).normalize_or_zero();
        sweep(
            mesh,
            b,
            &[
                Ring {
                    at: at[wrist as usize] - across * (h * 0.010),
                    half_width: h * 0.019,
                    half_depth: h * 0.010,
                    joint: wrist,
                },
                Ring {
                    at: at[wrist as usize] + (knuckles - at[wrist as usize]) * 0.5,
                    half_width: half_span,
                    half_depth: h * 0.010,
                    joint: wrist,
                },
                Ring {
                    at: knuckles,
                    half_width: half_span * 0.94,
                    half_depth: h * 0.009,
                    joint: wrist,
                },
            ],
            8,
            Cap::Flat,
            Cap::Flat,
        )?;
    }

    // The digits, each swept through its own bones.
    for digit in inf_anim::Digit::ALL {
        let Some(chain) = geometry.finger(digit) else {
            continue;
        };
        // The metacarpals are inside the palm slab; a finger's own tube starts at
        // the knuckle. A thumb has no metacarpal in this vocabulary, so its whole
        // chain is drawn.
        let start = if digit == inf_anim::Digit::Thumb || chain.joints.len() < 3 {
            0
        } else {
            1
        };
        let bones = &chain.joints[start..];
        if bones.len() < 2 {
            continue;
        }
        let thick = if digit == inf_anim::Digit::Thumb {
            0.0058
        } else {
            0.0046
        };
        let mut rings: Vec<Ring> = Vec::with_capacity(bones.len() + 1);
        for (i, &j) in bones.iter().enumerate() {
            let t = i as f64 / (bones.len() - 1) as f64;
            let r = h * thick * (1.0 - 0.30 * t);
            rings.push(Ring {
                at: at[j as usize],
                half_width: r,
                half_depth: r * 0.92,
                joint: j,
            });
        }
        // The tip: past the last joint by the length of the bone before it, so a
        // finger has a fingertip rather than ending at a knuckle.
        let last = *bones.last().expect("checked");
        let prev = bones[bones.len() - 2];
        let out = at[last as usize] - at[prev as usize];
        let tip = at[last as usize] + out * 0.85;
        rings.push(Ring {
            at: at[last as usize] + out * 0.45,
            half_width: h * thick * 0.62,
            half_depth: h * thick * 0.58,
            joint: last,
        });
        sweep(
            mesh,
            b,
            &rings,
            opts.finger_segments,
            Cap::Flat,
            Cap::Point(tip),
        )?;
    }
    Ok(())
}

/// Hip → knee → ankle, one welded tube.
fn leg(mesh: &mut Mesh, b: &mut Build, s: &Shape<'_>, side: BoneSide) -> Result<(), BodyError> {
    let (roles, at, h, opts) = (s.roles, s.at, s.h, s.opts);
    let (Some(thigh), Some(calf), Some(foot)) = (
        roles.first(BoneRoleKind::Thigh, side),
        roles.first(BoneRoleKind::Calf, side),
        roles.first(BoneRoleKind::Foot, side),
    ) else {
        return Err(BodyError::MissingPart("a leg"));
    };
    let hip = at[thigh as usize];
    let up = (at[thigh as usize] - at[calf as usize]).normalize_or_zero();
    let rings = [
        Ring {
            at: hip + up * (h * 0.030),
            half_width: h * 0.054,
            half_depth: h * 0.054,
            joint: thigh,
        },
        Ring {
            at: hip,
            half_width: h * 0.052,
            half_depth: h * 0.052,
            joint: thigh,
        },
        Ring {
            // The calf's belly, a third of the way down the shin.
            at: at[calf as usize] + (at[foot as usize] - at[calf as usize]) * 0.30,
            half_width: h * 0.038,
            half_depth: h * 0.040,
            joint: calf,
        },
        Ring {
            at: at[foot as usize]
                + (at[calf as usize] - at[foot as usize]).normalize_or_zero() * (h * 0.030),
            half_width: h * 0.024,
            half_depth: h * 0.024,
            joint: calf,
        },
    ];
    sweep(mesh, b, &rings, opts.limb_segments, Cap::Flat, Cap::Flat)
}

/// The foot: a flattened slab from the ankle forward over the ball to the toe.
fn foot(mesh: &mut Mesh, b: &mut Build, s: &Shape<'_>, side: BoneSide) -> Result<(), BodyError> {
    let (roles, at, h, opts) = (s.roles, s.at, s.h, s.opts);
    let ankle = roles
        .first(BoneRoleKind::Foot, side)
        .ok_or(BodyError::MissingPart("a foot"))?;
    // A rig with a ball joint says where the foot bends; one without gets a
    // default forward reach, because a leg that ends at an ankle has no foot at
    // all and that reads worse than a plain one.
    let ball = roles
        .first(BoneRoleKind::Ball, side)
        .map(|b| at[b as usize])
        .unwrap_or(at[ankle as usize] + DVec3::Z * (h * 0.085));
    // **The sole is the ankle's own height**, because on this rig it already is:
    // `manny` places `foot_l` at exactly `y = 0`, which is what makes foot IK's
    // plant mean "the ground". A foot modelled *below* its ankle would put the
    // character's mesh through the floor by its own thickness, and every gate
    // that measures a plant would be measuring a different surface from the one
    // the renderer draws.
    let sole = at[ankle as usize].y;
    // **The foot sweeps forward**, heel to toe, and every ring is centred exactly
    // its own half-depth above the sole — so the bottom of the shell is the
    // ground plane at every station rather than at one of them. Swept from the
    // ankle *down* to the heel first (the shape it was written as) the rings tilt
    // with the direction and the heel dips through the floor: measured 1.8 cm
    // below on a 1.2 m character, which is a heel buried in the ground.
    // The heel and the arch belong to the ankle; everything past the ball belongs
    // to the toe joint, which is the one that bends.
    let toe_joint = roles.first(BoneRoleKind::Ball, side).unwrap_or(ankle);
    let ring = |z: f64, x: f64, half_width: f64, half_depth: f64, joint: u16| Ring {
        at: DVec3::new(x, sole + half_depth, z),
        half_width,
        half_depth,
        joint,
    };
    let ax = at[ankle as usize].x;
    sweep(
        mesh,
        b,
        &[
            ring(
                at[ankle as usize].z - h * 0.032,
                ax,
                h * 0.021,
                h * 0.017,
                ankle,
            ),
            ring(at[ankle as usize].z, ax, h * 0.026, h * 0.026, ankle),
            ring(
                (at[ankle as usize].z + ball.z) * 0.5,
                (ax + ball.x) * 0.5,
                h * 0.027,
                h * 0.017,
                ankle,
            ),
            ring(ball.z, ball.x, h * 0.026, h * 0.012, toe_joint),
            ring(ball.z + h * 0.030, ball.x, h * 0.021, h * 0.008, toe_joint),
        ],
        opts.limb_segments.max(6),
        Cap::Flat,
        Cap::Flat,
    )
}

// ── the sweep ───────────────────────────────────────────────────────────────

/// Sample a `(half_width, half_depth)` profile at `t` in `[0, 1]`, linearly.
fn sample_profile(profile: &[(f64, f64)], t: f64) -> (f64, f64) {
    if profile.is_empty() {
        return (0.0, 0.0);
    }
    if profile.len() == 1 {
        return profile[0];
    }
    let x = t.clamp(0.0, 1.0) * (profile.len() - 1) as f64;
    let i = (x as usize).min(profile.len() - 2);
    let f = x - i as f64;
    let (a, b) = (profile[i], profile[i + 1]);
    (a.0 + (b.0 - a.0) * f, a.1 + (b.1 - a.1) * f)
}

/// The **lateral** direction for a shell sweeping along `d`, unit length.
///
/// # Why this is not "any perpendicular"
///
/// A ring is an *ellipse*, and [`Ring`]'s two radii are named `half_width` and
/// `half_depth` because a torso is wider than it is deep. That is only true if
/// `half_width` lands on the body's left-right axis — and a frame seeded from
/// "whichever cardinal axis the direction leans on least" does not: swept up `+Y`
/// it produces a `right` of `−Z`, which puts the *width* along the body's depth
/// and builds a torso 30 cm deep and 22 cm across.
///
/// So the seed is `X` — the body's own lateral axis — projected perpendicular to
/// the sweep, except for a shell that sweeps *along* `X` (an arm, a finger),
/// where `X` carries no lateral information and `Z` does.
fn lateral_seed(d: DVec3) -> DVec3 {
    let prefer = if d.x.abs() < 0.8 { DVec3::X } else { DVec3::Z };
    let projected = prefer - d * prefer.dot(d);
    if projected.length_squared() > 1.0e-12 {
        return projected.normalize();
    }
    any_perpendicular(d)
}

/// A unit vector perpendicular to `v`, chosen deterministically — crossed with
/// whichever cardinal axis `v` leans on least. The last-resort seed.
fn any_perpendicular(v: DVec3) -> DVec3 {
    let a = v.abs();
    let axis = if a.x <= a.y && a.x <= a.z {
        DVec3::X
    } else if a.y <= a.z {
        DVec3::Y
    } else {
        DVec3::Z
    };
    v.cross(axis).normalize_or_zero()
}

/// **Sweep a closed shell through `rings`.**
///
/// The cross-section frame is carried forward by **projection** rather than by a
/// rotation: each ring's `right` is the previous ring's with the new direction's
/// component taken out. That is parallel transport in the only form this needs,
/// it costs a `sqrt`, and it means a limb's tessellation does not spiral round
/// itself as the chain bends.
fn sweep(
    mesh: &mut Mesh,
    b: &mut Build,
    rings: &[Ring],
    segments: usize,
    start: Cap,
    end: Cap,
) -> Result<(), BodyError> {
    if rings.len() < 2 {
        return Ok(());
    }
    let band = b.band.take();
    let n = segments.max(3);
    let m = rings.len();

    // Directions: central differences inside, one-sided at the ends.
    let mut dirs: Vec<DVec3> = Vec::with_capacity(m);
    for i in 0..m {
        let d = if i == 0 {
            rings[1].at - rings[0].at
        } else if i == m - 1 {
            rings[m - 1].at - rings[m - 2].at
        } else {
            rings[i + 1].at - rings[i - 1].at
        };
        dirs.push(d.normalize_or_zero());
    }
    // A degenerate chain (two coincident rings) carries no direction and is
    // dropped rather than producing a shell of zero volume.
    if dirs.iter().any(|d| d.length_squared() < 0.5) {
        return Ok(());
    }

    let mut right = lateral_seed(dirs[0]);
    if right.length_squared() < 0.5 {
        return Ok(());
    }
    let mut frames: Vec<(DVec3, DVec3)> = Vec::with_capacity(m);
    for d in &dirs {
        let projected = right - *d * right.dot(*d);
        right = if projected.length_squared() > 1.0e-12 {
            projected.normalize()
        } else {
            any_perpendicular(*d)
        };
        let up = d.cross(right).normalize_or_zero();
        frames.push((right, up));
    }

    // The ring vertices.
    let mut ids: Vec<Vec<VertId>> = Vec::with_capacity(m);
    let mut touched: BTreeSet<VertId> = BTreeSet::new();
    for (i, ring) in rings.iter().enumerate() {
        let (r, u) = frames[i];
        let mut row = Vec::with_capacity(n);
        for k in 0..n {
            let theta = std::f64::consts::TAU * (k as f64) / (n as f64);
            let p = ring.at
                + r * (ring.half_width * pcos64(theta))
                + u * (ring.half_depth * psin64(theta));
            let v = mesh.alloc_vert(p.to_array());
            touched.insert(v);
            b.seed.push((v, ring.joint));
            row.push(v);
        }
        ids.push(row);
    }

    let (u0, u1) = band;
    let uu = |k: usize| u0 + (u1 - u0) * (k as f64 / n as f64);
    let vv = |i: usize| i as f64 / (m - 1) as f64;

    let mut faces: Vec<(Vec<VertId>, Vec<[f64; 2]>, DVec3)> = Vec::new();
    for i in 0..m - 1 {
        for k in 0..n {
            let k1 = (k + 1) % n;
            faces.push((
                vec![ids[i][k], ids[i][k1], ids[i + 1][k1], ids[i + 1][k]],
                vec![
                    [uu(k), vv(i)],
                    [uu(k + 1), vv(i)],
                    [uu(k + 1), vv(i + 1)],
                    [uu(k), vv(i + 1)],
                ],
                (rings[i].at + rings[i + 1].at) * 0.5,
            ));
        }
    }
    // The two ends.
    for (which, cap) in [(0usize, start), (m - 1, end)] {
        let inside = rings[which].at;
        match cap {
            Cap::Flat => {
                let loop_verts: Vec<VertId> = (0..n).map(|k| ids[which][k]).collect();
                let uvs: Vec<[f64; 2]> = (0..n)
                    .map(|k| {
                        let theta = std::f64::consts::TAU * (k as f64) / (n as f64);
                        [
                            u0 + (u1 - u0) * (0.5 + 0.5 * pcos64(theta)),
                            0.5 + 0.5 * psin64(theta),
                        ]
                    })
                    .collect();
                // The interior reference is the NEXT ring in, so the cap's normal
                // is judged against the body of the tube rather than against its
                // own plane (where the dot product is zero and the sign is noise).
                let inward = if which == 0 {
                    rings[1].at
                } else {
                    rings[m - 2].at
                };
                faces.push((loop_verts, uvs, inward));
            }
            Cap::Point(p) => {
                let apex = mesh.alloc_vert(p.to_array());
                touched.insert(apex);
                b.seed.push((apex, rings[which].joint));
                for k in 0..n {
                    let k1 = (k + 1) % n;
                    faces.push((
                        vec![apex, ids[which][k], ids[which][k1]],
                        vec![
                            [(u0 + u1) * 0.5, if which == 0 { 0.0 } else { 1.0 }],
                            [uu(k), vv(which)],
                            [uu(k + 1), vv(which)],
                        ],
                        inside,
                    ));
                }
            }
        }
    }

    for (verts, uvs, inside) in faces {
        push_face(mesh, &verts, &uvs, inside)?;
    }
    mesh.finish_patch(&touched)?;
    b.shells += 1;
    Ok(())
}

/// Add one face, **winding it outward by measurement**.
///
/// A swept frame's handedness depends on the direction it was swept in, and a
/// generator that assumes one is a generator that produces an inside-out limb the
/// first time a chain points the other way. The Newell normal is compared against
/// the direction away from a point known to be inside the shell, and the loop is
/// reversed if it points the wrong way — which is exactly what
/// `character::block_body_mesh`'s `push_quad` already does, for the same reason.
fn push_face(
    mesh: &mut Mesh,
    verts: &[VertId],
    uvs: &[[f64; 2]],
    inside: DVec3,
) -> Result<(), BodyError> {
    let positions: Vec<DVec3> = verts
        .iter()
        .map(|&v| mesh.position(v).unwrap_or(DVec3::ZERO))
        .collect();
    let mut normal = DVec3::ZERO;
    let mut centroid = DVec3::ZERO;
    for i in 0..positions.len() {
        let a = positions[i];
        let b = positions[(i + 1) % positions.len()];
        normal += a.cross(b);
        centroid += a;
    }
    centroid /= positions.len() as f64;
    let outward = centroid - inside;
    let (loop_verts, corners): (Vec<VertId>, Vec<CornerData>) = if normal.dot(outward) < 0.0 {
        (
            verts.iter().rev().copied().collect(),
            uvs.iter()
                .rev()
                .map(|&uv| CornerData { uv, normal: None })
                .collect(),
        )
    } else {
        (
            verts.to_vec(),
            uvs.iter()
                .map(|&uv| CornerData { uv, normal: None })
                .collect(),
        )
    };
    mesh.add_face_raw(&loop_verts, &corners, Some(0))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manny() -> SkeletonAsset {
        inf_anim::build_manny(&inf_anim::BodyParams::default()).expect("the mannequin builds")
    }

    /// **The body is a body**: it has the parts, they are where the rig says, and
    /// the whole thing is a valid manifold mesh the exporter accepts.
    #[test]
    fn a_generated_body_is_manifold_and_the_exporter_takes_it() {
        let rig = manny();
        let (mesh, seeded) = body_mesh(&rig, &BodyOptions::default()).expect("a body");
        crate::validate::validate(&mesh).expect("the generated body is a valid mesh");
        // **Every vertex knows which bone made it**, and every one of those is a
        // deform bone — the prior a weight solve refines, and the thing that
        // stops the 4 % of geometry no ray can reach from staying behind on the
        // root. Mutation: dropping the `b.seed.push` in the cap arm leaves the
        // fingertips and the head's two poles without one.
        assert_eq!(
            seeded.seed.len(),
            mesh.vert_ids().count(),
            "a vertex was emitted without a seed bone"
        );
        let roles = rig.role_index();
        for (v, j) in &seeded.seed {
            assert!(
                roles.is_deform(*j),
                "vertex {v:?} was seeded onto `{}`, which does not deform anything",
                rig.skeleton.joints()[*j as usize].name
            );
        }
        println!("{} welded shells", seeded.shells);
        assert!(seeded.shells >= 16, "{} shells", seeded.shells);
        let (asset, report) = crate::export::to_mesh_asset(&mesh, &crate::ExportOptions::default());
        println!(
            "starter body: {} vertices, {} faces -> {} exported vertices, {} triangles",
            mesh.vert_ids().count(),
            mesh.face_ids().count(),
            report.vertices,
            report.triangles
        );
        assert_eq!(report.non_finite_written, 0);
        assert_eq!(
            report.coincident_vertices, 0,
            "the sweep welded two rings onto each other"
        );
        assert_eq!(asset.submeshes.len(), 1, "one material slot, one submesh");
        // A body, not a pile of boxes: enough geometry to read as a character and
        // little enough to be a starter.
        assert!(
            (1_000..40_000).contains(&report.triangles),
            "{} triangles",
            report.triangles
        );
    }

    /// **Every face points OUT.** The signed volume of a closed shell is positive
    /// exactly when its faces are wound outward, and this generator sweeps limbs
    /// in four different directions — so one assumed handedness would flip half
    /// the body inside out.
    #[test]
    fn every_shell_is_wound_outward() {
        let rig = manny();
        let (mesh, _) = body_mesh(&rig, &BodyOptions::default()).expect("a body");
        let (asset, _) = crate::export::to_mesh_asset(&mesh, &crate::ExportOptions::default());
        let sub = &asset.submeshes[0];
        let mut volume = 0.0f64;
        for tri in sub.indices.chunks_exact(3) {
            let p = |i: usize| {
                let v = sub.vertices[tri[i] as usize].position;
                DVec3::new(v[0] as f64, v[1] as f64, v[2] as f64)
            };
            volume += p(0).dot(p(1).cross(p(2))) / 6.0;
        }
        println!("the body encloses {volume:.5} m^3");
        assert!(
            volume > 0.0,
            "the body's faces are wound INWARD (signed volume {volume})"
        );
        // A 1.75 m human is somewhere near 0.06 m^3 of water. Interpenetrating
        // shells double-count where they overlap, so this is a band and not a
        // number — but it is a band a body is in and a pile of boxes is not.
        assert!(
            (0.02..0.20).contains(&volume),
            "a 1.75 m body enclosing {volume} m^3 is not a body"
        );
    }

    /// **It is a function of the rig**, not of a constant: a taller rig gets a
    /// bigger body, and two runs of the same rig are identical.
    #[test]
    fn the_body_follows_its_rig_and_is_reproducible() {
        let rig = manny();
        let (a, _) = body_mesh(&rig, &BodyOptions::default()).expect("a body");
        let (b, _) = body_mesh(&rig, &BodyOptions::default()).expect("a body");
        assert_eq!(a.canonical(), b.canonical(), "not reproducible");

        let tall = inf_anim::build_manny(&inf_anim::BodyParams {
            height_m: 2.4,
            ..inf_anim::BodyParams::default()
        })
        .expect("a tall mannequin");
        let (big, _) = body_mesh(&tall, &BodyOptions::default()).expect("a body");
        let extent = |m: &Mesh| {
            let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
            for v in m.vert_ids() {
                let p = m.position(v).expect("live");
                lo = lo.min(p.y);
                hi = hi.max(p.y);
            }
            hi - lo
        };
        let (small, large) = (extent(&a), extent(&big));
        println!("the body is {small:.3} m tall at 1.75 and {large:.3} m at 2.4");
        assert!(
            large > small * 1.25,
            "a 2.4 m rig produced a {large} m body against {small} m at 1.75 m"
        );
        // The counts do NOT change with height — the tessellation is a parameter
        // and the proportions are the rig's.
        assert_eq!(a.vert_ids().count(), big.vert_ids().count());
    }

    /// **A rig with no role table is refused**, by name, rather than producing a
    /// body built on a guess.
    #[test]
    fn a_rig_with_no_roles_is_refused_and_a_rig_with_no_arms_says_which_part() {
        let canonical = inf_anim::build_template(
            inf_anim::BodyPlan::BipedCanonical,
            &inf_anim::BodyParams::default(),
        )
        .expect("the canonical biped builds");
        assert!(canonical.roles.is_empty(), "the fixture must be table-less");
        assert_eq!(
            body_mesh(&canonical, &BodyOptions::default()).err(),
            Some(BodyError::NoRoles)
        );
        // A rig that HAS a table but is not a humanoid names the part it lacks.
        let mut legless = manny();
        legless.roles.retain(|r| r.kind != BoneRoleKind::Thigh);
        assert_eq!(
            body_mesh(&legless, &BodyOptions::default()).err(),
            Some(BodyError::MissingPart("a leg"))
        );
    }

    /// **The hands are hands**: five digits per side, each with a tip past its
    /// last joint, and the two of them mirrored.
    #[test]
    fn the_body_has_two_five_fingered_hands() {
        let rig = manny();
        let (mesh, _) = body_mesh(&rig, &BodyOptions::default()).expect("a body");
        let at = bind_globals(&rig.skeleton);
        let roles = rig.role_index();
        // Every fingertip has geometry within a finger's width of it.
        let mut checked = 0;
        for side in [BoneSide::Left, BoneSide::Right] {
            let wrist = roles.first(BoneRoleKind::Hand, side).expect("a hand");
            let hand = inf_anim::hand_of(&rig.skeleton, roles, wrist).expect("a hand's geometry");
            for digit in inf_anim::Digit::ALL {
                let chain = hand.finger(digit).expect("a digit");
                let tip = at[*chain.joints.last().expect("a tip") as usize];
                let near = mesh
                    .vert_ids()
                    .filter_map(|v| mesh.position(v))
                    .filter(|p| (*p - tip).length() < 0.03)
                    .count();
                assert!(
                    near >= 3,
                    "{side:?} {digit:?}: only {near} vertices within 3 cm of its tip"
                );
                checked += 1;
            }
        }
        assert_eq!(checked, 10, "five digits on each of two hands");
    }

    /// **A 1.75 m character is 1.75 m tall.**
    ///
    /// The claim `CROWN_ABOVE_HEAD` exists for, and it is not obvious: this rig
    /// puts its ankle at exactly `y = 0` and its `head` joint at 0.93 of the
    /// standing height, so the body's height is decided by how far this generator
    /// puts the crown above a joint — nothing in the rig says. The first pass got
    /// it wrong by **11 %** (a 1.943 m body on a 1.75 m rig) and every count-based
    /// assertion in this file was perfectly happy.
    #[test]
    fn the_body_stands_as_tall_as_its_rig_asked_for() {
        for height_m in [1.2f64, 1.75, 2.4] {
            let rig = inf_anim::build_manny(&inf_anim::BodyParams {
                height_m,
                ..inf_anim::BodyParams::default()
            })
            .expect("a mannequin");
            let (mesh, _) = body_mesh(&rig, &BodyOptions::default()).expect("a body");
            let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
            for v in mesh.vert_ids() {
                let p = mesh.position(v).expect("live");
                lo = lo.min(p.y);
                hi = hi.max(p.y);
            }
            println!(
                "a {height_m} m rig makes a body {:.3} m tall, sole at {lo:.4}",
                hi - lo
            );
            assert!(
                ((hi - lo) - height_m).abs() < height_m * 0.04,
                "a {height_m} m rig produced a {} m body",
                hi - lo
            );
            // …and it STANDS on the ground rather than through it: the sole is at
            // the ankle's own height, which is what foot IK plants.
            assert!(
                lo.abs() < height_m * 0.005,
                "the sole is at {lo}, not on the ground the ankle is on"
            );
        }
    }

    /// **A VISIBILITY ORACLE MUST BE BUILT IN THE SPACE ITS RAYS ARE CAST IN** —
    /// SK1b's fifth decision, measured in the crate that owns both halves (SK1b
    /// audit).
    ///
    /// The wave found this and fixed it at one call site: `starter_body` builds
    /// its BVH from [`crate::mesh_soup`] — the kernel's own `f64` triangles —
    /// rather than from the tessellated, `f32`-narrowed ones. The evidence for it
    /// was two numbers in a ledger and an `unreached < 60` bound two crates away.
    /// Numbers in a ledger are numbers a test prints (SK1a audit, decision 2), so
    /// both are printed and both are asserted here, on the mesh that surfaced it.
    ///
    /// The mechanism: a ray cast from an **exact kernel vertex** starts up to an
    /// ulp outside the surface a narrowed copy describes, and hits its own face
    /// at `t ~ 0`. An *imported* mesh is immune — its kernel positions are widened
    /// `f32` and the round trip is exact — so this could not appear until the
    /// engine generated a mesh in `f64`, which is what this module is.
    #[test]
    fn the_narrowed_oracle_cannot_see_a_third_of_a_generated_body() {
        let rig = manny();
        let (mesh, _) = body_mesh(&rig, &BodyOptions::default()).expect("a body");
        let verts = mesh.vert_ids().count();

        // The exporter's triangles, read back at `f32` and widened again — which
        // is exactly the soup `inf_editor_core::dcc::triangle_soup` hands the
        // BVH, and what this call site used to be given.
        let (asset, _) = crate::export::to_mesh_asset(&mesh, &crate::ExportOptions::default());
        let mut narrowed: Vec<crate::Tri> = Vec::new();
        for sub in &asset.submeshes {
            for t in sub.indices.chunks_exact(3) {
                let p = |i: u32| {
                    let v = sub.vertices[i as usize].position;
                    DVec3::new(v[0] as f64, v[1] as f64, v[2] as f64)
                };
                narrowed.push(crate::Tri {
                    a: p(t[0]),
                    b: p(t[1]),
                    c: p(t[2]),
                });
            }
        }
        // Bind once; the two solves differ only in the oracle they are given.
        let mut bound = mesh.clone();
        crate::ops::apply(
            &mut bound,
            &crate::Op::BindSkin {
                skeleton: None,
                joints: rig.skeleton.len() as u32,
            },
        )
        .expect("a generated body binds");
        let roles = rig.role_index();
        let deform: Vec<bool> = (0..rig.skeleton.len() as u16)
            .map(|j| roles.is_deform(j))
            .collect();
        let unreached = |tris: Vec<crate::Tri>| -> usize {
            crate::heat::solve_heat_weights_for(
                &bound,
                &crate::Bvh::new(tris),
                &rig.skeleton,
                Some(&deform),
            )
            .expect("a solve")
            .1
            .unreached
        };
        let exact = unreached(crate::mesh_soup(&mesh));
        let rounded = unreached(narrowed);
        println!(
            "the visibility oracle over {verts} generated vertices: {exact} unreached in f64, \
             {rounded} through an f32 round trip"
        );
        assert_eq!(
            (exact, rounded, verts),
            (35, 349, 795),
            "the f32/f64 oracle seam moved — if that is deliberate, the numbers in \
             `docs/memos/island-progress.md` move with it"
        );
        // The claim is the RATIO, not the two numbers: a narrowed oracle cannot
        // see ten times as much of a generated body as an exact one.
        assert!(
            rounded > exact * 5,
            "the narrowed oracle lost {rounded} against {exact} — this arm no longer \
             demonstrates the seam it is named for"
        );
    }
}
