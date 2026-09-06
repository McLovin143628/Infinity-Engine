//! **THE SIM-LOD TIER SYSTEM** (wave NPC1a) — what a crowd NPC costs, decided
//! once, from sim state, in a door both hosts call.
//!
//! # The measurement this exists for
//!
//! Before this module the engine had **zero animation LOD** (grep-verified): the
//! whole pose pipeline — drive → pelvis → foot IK → hand IK → goals → ragdoll —
//! ran for every [`AnimStateMachine`] entity on every fixed step regardless of
//! where it stood, every character got a rapier controller and a capsule, and
//! every posed character contributed `36 + 40 · joints` bytes to the sim trace:
//! **6 476 B per NPC per step** at the starter character's 161 bones
//! ([`crate::pose::pose_state_bytes`]). A thousand of those is 5.8 GB of retained
//! trace in one gate process and a fixed step three orders past its 6.0 ms
//! ratchet. None of that improves by making any one pass faster.
//!
//! So the crowd pays by **tier**, and the tier is one decision:
//!
//! | tier | rapier | controller | pose | hand IK | position |
//! |---|---|---|---|---|---|
//! | [`Full`](CrowdTier::Full) | capsule | yes | full | yes | **steered** |
//! | [`Near`](CrowdTier::Near) | capsule | **no** | full | **no** | route(clock) |
//! | [`Far`](CrowdTier::Far) | **none** | **none** | **none** | no | route(clock) |
//! | [`Dormant`](CrowdTier::Dormant) | — | — | — | — | route(clock), with no entity to write it to |
//!
//! # The tier owns the COMPONENTS (wave NPC1c)
//!
//! NPC1a gave every materialized agent a body, a capsule and a controller at
//! every tier and had the 3D bridge decline to *mirror* the ones the tier called
//! bodiless. That is one system holding an opinion about another system's data,
//! and it cost what this module's own founding law predicts:
//! `terrain_stream::observes_terrain` picks its subjects **by component**, so a
//! thousand NPCs were a thousand terrain observers — 712 of them `Far`, with no
//! rapier body and no door by which to query a height — each pulling a
//! `SIM_MARGIN_TILES` neighbourhood of level-0 pages into a want set that is
//! never clamped. Measured by the NPC1b audit at **+1.16 ms of "crowd" cost that
//! was not the crowd**.
//!
//! So `set_tier_components` puts the physical set on and takes it off with the
//! tier. A thing with no body has no body; nothing downstream has to be told,
//! because there is nothing to see.
//!
//! **Every agent is somewhere on its own route at every tier**, and that is
//! deliberate rather than unfinished. A record that froze where it
//! dematerialized was then *tiered* from that frozen point, so a walking agent
//! whose route carried it home could never come back — the NPC1a audit's
//! `a_dormant_agent_keeps_walking_its_route_and_can_come_back` is why the law
//! reads this way.
//!
//! What NPC1c splits is **who advances the agent along the route**:
//!
//! * a tier with **no controller** ([`Near`](CrowdTier::Near),
//!   [`Far`](CrowdTier::Far), [`Dormant`](CrowdTier::Dormant)) is `route(clock)`
//!   — a pure function of the record, the step count and one phase, costing an
//!   interpolation;
//! * the one **with** it ([`Full`](CrowdTier::Full), and see
//!   [`CrowdTier::steers`] for the 92.756 ms that decided where the line goes)
//!   is moved by `inf_physics::d3::step_character_movement`, five phases later
//!   in the same fixed step, through `move_and_slide` — the one door a character
//!   in this engine moves through. What the crowd writes for those is an
//!   **intent** (`steer_agent`), never a transform, because two writers on one
//!   transform is precisely what NPC1a refused a controller to avoid.
//!
//! The two meet at the transitions, and neither is allowed to pop:
//! a **promotion** places the body at `route(clock)`, which is exactly where the
//! tier below was already drawing it; a **demotion** hands the clock back the
//! metre the *body* reached ([`CrowdRoute::rephase_delta`],
//! [`CrowdRecord::rephase_m`]), because a body that was blocked, or slowed on a
//! slope, or simply walking at its own gait, is not where the clock ran on to.
//! Measured: a body 4.05 m behind its own clock moves **0.0000 m** across the
//! transition.
//!
//! # THE VISIBILITY LAW: the tier never reads a camera
//!
//! The standing law is *visibility filters what is DRAWN, never what is
//! SIMULATED*. A tier that read the camera would break it outright — two players
//! looking in different directions would simulate different crowds, and PIE would
//! stop equalling shipping the moment the editor's free camera moved.
//!
//! So [`CrowdBand`] is [`crate::band::SimBand`]'s shape, one radius wider:
//! anchors are the entities carrying [`StreamingSource`] (the set P16's cell
//! activation already reads to decide which parts of the world exist at all),
//! snapped to the same [`BAND_LATTICE_M`] lattice, ordered, deduplicated. There
//! is **no camera argument** to [`step_crowd`], so a caller cannot pass one by
//! accident.
//!
//! **The lattice's slop is inherited too, and it is worth stating in metres.**
//! `SimBand`'s own module measures it: snapping an anchor to a 16 m cell centre
//! moves it by at most `BAND_LATTICE_M · √2 / 2` ≈ **11.3 m**, so every radius
//! below is really "that radius ± 11.3 m" — a third of the 32 m `Full` ring and
//! a fiftieth of the 512 m one. The radii are chosen with that in mind (`Full`
//! at half of `DEFAULT_COLLIDER_NEAR_M` still sits inside the solid world at
//! its worst case), and the arm that bounds it is `SimBand`'s
//! `the_lattice_slop_is_bounded_by_half_a_cell_diagonal` — one lattice, one
//! bound, not two copies of it.
//!
//! # Hysteresis is REFUSED, and it is refused for the same reason
//!
//! A tier is a *function of sim state*, not of the history of sim states. A
//! hysteretic tier would agree with itself inside each host and diverge between
//! them the first time one of them started mid-trace — which is the whole
//! property the island gate exists to protect. The cost is the one
//! `SimBand`'s own module states and measures: an agent parked on a lattice line
//! re-tiers every step, alternating between exactly **two** tiers, never
//! wandering. See `an_agent_parked_on_a_tier_boundary_alternates_between_two`.
//!
//! # It fails toward FULL
//!
//! A world with no streaming source at all is [`CrowdBand::unbounded`]: every
//! agent is [`Full`](CrowdTier::Full), which is the pre-NPC1a behaviour of a
//! character in every fixture this tree already has. Same for a world whose only
//! sources carry non-finite positions, and same for an agent whose own position
//! is not finite. Dropping a tier is the dangerous direction (an NPC stops
//! animating, stops colliding, or stops existing); keeping it is merely slow.
//!
//! # No schema moves
//!
//! [`CrowdPopulationRes`] is a bevy **resource** and [`CrowdAgent`] is a
//! component the scene serializer does not know about, exactly as
//! [`crate::deform::DeformFieldRes`] is: the `.inf_lvl` walk writes
//! `RuntimeEntity` fields and never a resource, so nothing here can be saved and
//! **scene v26 does not move**. That is correct for NPC1a — a test population is
//! transient — and it is the shape NPC1d inherits: a population is *data in the
//! recipe*, and bodies materialize by tier.
//!
//! [`StreamingSource`]: crate::components::StreamingSource

use std::collections::{BTreeMap, BTreeSet};

use bevy_ecs::prelude::{Component, Entity, Resource};
use glam::DVec3;
use uuid::Uuid;

use crate::band::{streaming_sources, BAND_LATTICE_M};
use crate::components::{
    AnimStateMachine, BodyKind3D, CharacterController3D, CharacterMovement, Collider3D,
    ColliderShape3DKind, Gait, RigidBody3D, RotationMode, SkeletalMesh, Transform,
};
use crate::math::Vec3d;
use crate::world::EcsWorld;

// ── the tier ────────────────────────────────────────────────────────────────

/// What one crowd NPC costs this step.
///
/// Ordered cheapest-last: `Full < Near < Far < Dormant`, so `max`/`min` over a
/// set of tiers mean "the dearest" / "the cheapest" without a lookup table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Hash)]
pub enum CrowdTier {
    /// Today's whole pipeline: a capsule in rapier, the full pose evaluation and
    /// the hand IK pass. What every character in this tree got unconditionally
    /// before NPC1a, and what a hero-class actor still gets — see
    /// [`step_crowd`]'s "the hero is untouched" note.
    #[default]
    Full,
    /// A capsule and a full pose, without the hand pass. The reach and the
    /// finger closure are the passes a viewer cannot resolve at distance and the
    /// ones that need a `HandIk` request to do anything at all, so they are the
    /// first thing off the ladder.
    Near,
    /// **Kinematic.** No rapier body, no pose evaluation, no machine advance:
    /// the transform is `route(clock)` and the trace carries a cached digest of
    /// the last pose the agent published instead of 161 joints.
    Far,
    /// **Data only.** The entity is despawned; [`CrowdRecord`] remembers what it
    /// was doing and what it looked like, keeps walking its route as a pure
    /// function of the clock (there is just nothing to write the transform
    /// onto), and re-materializes the step its tier comes back.
    Dormant,
}

impl CrowdTier {
    /// Whether the pose pipeline runs for this tier (`Full`/`Near`).
    ///
    /// Read by [`crate::pose::step_pose_evaluation`], which is the one place a
    /// pose is evaluated in this engine, so a tier cannot mean one thing in the
    /// editor and another in the player.
    #[inline]
    pub fn poses(self) -> bool {
        matches!(self, CrowdTier::Full | CrowdTier::Near)
    }

    /// Whether the SK1b hand pass runs (`Full` only).
    #[inline]
    pub fn hand_ik(self) -> bool {
        matches!(self, CrowdTier::Full)
    }

    /// Whether this tier is solid — i.e. whether the 3D bridge gives it a rapier
    /// body and collider (`Full`/`Near`).
    #[inline]
    pub fn has_body(self) -> bool {
        matches!(self, CrowdTier::Full | CrowdTier::Near)
    }

    /// **Whether this tier is STEERED** — carries a `CharacterController3D` and
    /// a [`CharacterMovement`], and is moved by
    /// `inf_physics::d3::step_character_movement` rather than by its route
    /// clock. `Full` only, and the radius is 32 m.
    ///
    /// # It was `Full`/`Near` for one measurement
    ///
    /// NPC1c's first cut gave both near tiers a controller, on the reasoning
    /// that the `Near` rung would then finally *save* something (a near agent
    /// can carry a hand request the tier refuses). The island's own N = 1 000
    /// row priced it: `character move` went from **0.132 ms to 92.756 ms** —
    /// 83 % of a 111 ms fixed step, for 291 controllers, which is a quarter of a
    /// millisecond each and **super-linear** against the hero's own 0.13.
    ///
    /// A character in this engine is moved by collide-and-slide shape casts
    /// against a city of 17 823 bodies, and that is not a thing 291 of anything
    /// can afford. So the ladder splits where the measurement says rather than
    /// where the design hoped:
    ///
    /// | tier | solid | steered | posed |
    /// |---|---|---|---|
    /// | `Full` | yes | **yes** | yes + hands |
    /// | `Near` | yes | **no** — `route(clock)` | yes |
    /// | `Far` | no | no | no |
    ///
    /// A `Near` agent is still something a player can walk into and still poses
    /// every joint; what it does not do is negotiate its own way past a wall.
    /// At 32 m and out that is a difference a viewer can see only when the route
    /// runs through something, which is the trade this rung is for.
    #[inline]
    pub fn steers(self) -> bool {
        matches!(self, CrowdTier::Full)
    }

    /// Whether an entity exists for this tier at all (everything but `Dormant`).
    #[inline]
    pub fn materialized(self) -> bool {
        !matches!(self, CrowdTier::Dormant)
    }

    /// **Whether this tier casts a shadow of its own shape** (`Full` only), or
    /// through the crowd's shared proxy (wave NPC1b).
    ///
    /// A skinned caster is one *geometry group* in the virtual shadow map, and
    /// `inf_render::VSM_MAX_GROUPS` is 1 024 — so a thousand NPCs each casting
    /// their own silhouette is a thousand groups, past the ceiling, and the
    /// overflow is refused. `Full` is 32 m, which is where a viewer can read that
    /// an arm moved; past it a box the agent's own size is the same handful of
    /// page texels and costs ONE group for the whole crowd.
    ///
    /// The predicate lives here, beside [`poses`](Self::poses) and
    /// [`has_body`](Self::has_body), so the tier means one thing in the editor
    /// and the player — both projectors read it through the same door.
    #[inline]
    pub fn skinned_caster(self) -> bool {
        matches!(self, CrowdTier::Full)
    }

    /// **Whether this tier casts a shadow AT ALL** — the crowd's shadow LOD
    /// (island wave NPC1e).
    ///
    /// [`skinned_caster`](Self::skinned_caster) answered *"its own silhouette or
    /// the shared proxy"*, and NPC1b's carried item 4 measured what the proxy did
    /// not fix: 968 boxes walking through Harbour City scattered shadow-page
    /// invalidation over **168.6 pages a frame against the island's own 56.3**,
    /// at 1 236 page draws against 328, and the NPC1b audit added that the
    /// *deferred* pages doubled with them — so the crowd was also serving more
    /// stale shadow. One group is the right answer to `VSM_MAX_GROUPS` and no
    /// answer at all to how many pages a moving crowd dirties. That item named
    /// the lever as *"proxies that stop casting past a radius"*, and this is the
    /// radius: the ladder's own [`Near`](CrowdTier::Near), 96 m.
    ///
    /// **The cost is visible and is stated rather than hidden**, which is the
    /// same bargain the proxy already struck one rung in: past 96 m an NPC has no
    /// shadow. It is chosen at the tier rather than as a render setting because
    /// the tier is the one place a distance decision about a crowd is made, and
    /// both projectors read it through this door.
    #[inline]
    pub fn casts_shadow(self) -> bool {
        matches!(self, CrowdTier::Full | CrowdTier::Near)
    }

    /// The byte the trace folds. Frozen: these discriminants are compared
    /// between two hosts and, through the replay path, two machines.
    #[inline]
    pub fn as_u8(self) -> u8 {
        match self {
            CrowdTier::Full => 0,
            CrowdTier::Near => 1,
            CrowdTier::Far => 2,
            CrowdTier::Dormant => 3,
        }
    }

    /// The label the instruments print.
    #[inline]
    pub fn name(self) -> &'static str {
        match self {
            CrowdTier::Full => "full",
            CrowdTier::Near => "near",
            CrowdTier::Far => "far",
            CrowdTier::Dormant => "dormant",
        }
    }
}

// ── the radii ───────────────────────────────────────────────────────────────

/// Metres inside which an agent is [`Full`](CrowdTier::Full).
///
/// **Chosen against the collider band, not invented.**
/// [`crate::band::DEFAULT_COLLIDER_NEAR_M`] is 64 m — the radius inside which a
/// building is solid — and an NPC you can walk into has to be inside a world you
/// can walk into. 32 m is half of it, which keeps the dearest tier to the
/// quarter-area a viewer can actually read a finger pose at, and is still nine
/// times a fixed step's travel at a sprint.
pub const DEFAULT_CROWD_FULL_M: f64 = 32.0;

/// Metres inside which an agent is at worst [`Near`](CrowdTier::Near).
///
/// Wider than [`crate::band::DEFAULT_COLLIDER_NEAR_M`] on purpose: an NPC is one
/// capsule where a grammar building is dozens of solids, so the tier that keeps
/// a body can afford to reach past the tier that keeps a *building's* body.
pub const DEFAULT_CROWD_NEAR_M: f64 = 96.0;

/// Metres past which an agent is [`Dormant`](CrowdTier::Dormant).
///
/// [`crate::band::DEFAULT_COLLIDER_FAR_M`] is 1 024 m, and this is half of it,
/// because a dormant agent is *gone* rather than cheap: the radius has to be
/// comfortably inside the cell-activation neighbourhood so an agent does not
/// dematerialize inside the world the player can see. NPC1b's impostors are what
/// moves it out.
pub const DEFAULT_CROWD_FAR_M: f64 = 512.0;

/// The three radii, in metres, ascending — the shape [`CrowdBand`] is built from.
pub const DEFAULT_CROWD_RADII: (f64, f64, f64) = (
    DEFAULT_CROWD_FULL_M,
    DEFAULT_CROWD_NEAR_M,
    DEFAULT_CROWD_FAR_M,
);

// ── the band ────────────────────────────────────────────────────────────────

/// **The one door**: which tier an agent takes this step.
///
/// [`crate::band::SimBand`] with three radii instead of two, and every one of
/// that type's rules restated in code rather than in prose: anchors are
/// [`StreamingSource`] entities, snapped to [`BAND_LATTICE_M`], sorted,
/// deduplicated, non-finite ones dropped; an empty anchor set is
/// [`unbounded`](Self::unbounded); the [`stamp`](Self::stamp) is a membership
/// hash and the only legal operation on it is `==`.
///
/// [`StreamingSource`]: crate::components::StreamingSource
#[derive(Debug, Clone, PartialEq)]
pub struct CrowdBand {
    anchors: Vec<DVec3>,
    full_m: f64,
    near_m: f64,
    far_m: f64,
    unbounded: bool,
    stamp: u64,
}

impl Default for CrowdBand {
    fn default() -> Self {
        Self::unbounded()
    }
}

impl CrowdBand {
    /// Everything is [`Full`](CrowdTier::Full) — the answer for a world with no
    /// streaming source, and the pre-NPC1a behaviour of every fixture in this
    /// tree.
    pub fn unbounded() -> Self {
        Self {
            anchors: Vec::new(),
            full_m: f64::INFINITY,
            near_m: f64::INFINITY,
            far_m: f64::INFINITY,
            unbounded: true,
            stamp: 0,
        }
    }

    /// The band a world's own streaming sources define.
    pub fn from_world(world: &EcsWorld, radii: (f64, f64, f64)) -> Self {
        Self::from_anchors(streaming_sources(world).into_iter().map(|(p, _)| p), radii)
    }

    /// The band a set of anchor positions defines.
    ///
    /// Non-finite anchors are dropped; if that leaves none — or if any radius is
    /// not finite, or they are not ascending — the band is
    /// [`unbounded`](Self::unbounded), failing toward `Full` per the module docs.
    pub fn from_anchors(anchors: impl IntoIterator<Item = DVec3>, radii: (f64, f64, f64)) -> Self {
        let (full_m, near_m, far_m) = radii;
        let mut snapped: Vec<[i64; 2]> = anchors
            .into_iter()
            .filter(|p| p.x.is_finite() && p.z.is_finite())
            .map(|p| [lattice(p.x), lattice(p.z)])
            .collect();
        let ordered = full_m <= near_m && near_m <= far_m;
        let finite = full_m.is_finite() && near_m.is_finite() && far_m.is_finite();
        if snapped.is_empty() || !finite || !ordered {
            return Self::unbounded();
        }
        snapped.sort_unstable();
        snapped.dedup();

        // FNV-1a over the lattice coordinates and all three radii — `SimBand`'s
        // mixer, spelled the same way, because two stamps that meant the same
        // thing and hashed differently would be worse than no stamp at all.
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut fold = |v: u64| {
            for b in v.to_le_bytes() {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        };
        for c in &snapped {
            fold(c[0] as u64);
            fold(c[1] as u64);
        }
        fold(full_m.to_bits());
        fold(near_m.to_bits());
        fold(far_m.to_bits());

        Self {
            anchors: snapped
                .iter()
                .map(|c| DVec3::new(unlattice(c[0]), 0.0, unlattice(c[1])))
                .collect(),
            full_m,
            near_m,
            far_m,
            unbounded: false,
            stamp: h,
        }
    }

    /// `true` when nothing is banded — every agent is `Full`.
    #[inline]
    pub fn is_unbounded(&self) -> bool {
        self.unbounded
    }

    /// The band's membership stamp. `0` for an unbounded band.
    #[inline]
    pub fn stamp(&self) -> u64 {
        self.stamp
    }

    /// The lattice-snapped anchors the band is measured about.
    #[inline]
    pub fn anchors(&self) -> &[DVec3] {
        &self.anchors
    }

    /// The three radii, in metres.
    #[inline]
    pub fn radii(&self) -> (f64, f64, f64) {
        (self.full_m, self.near_m, self.far_m)
    }

    /// **The tier a point takes in this band** — the decision, in one place.
    ///
    /// Measured in the XZ plane from the nearest anchor, exactly as the collider
    /// band measures a building: an NPC on a roof and an NPC in the street below
    /// are the same distance away for the purpose of what they cost, and folding
    /// height in would make a tier depend on terrain the agent is standing on.
    ///
    /// A non-finite point is `Full` in an unbounded band (refusing it would
    /// silently change a fixture's pre-NPC1a behaviour) and `Dormant` in a
    /// banded one, because a NaN distance compares false against every radius
    /// and the fall-through is the cheapest tier.
    #[inline]
    pub fn tier(&self, p: DVec3) -> CrowdTier {
        if self.unbounded {
            return CrowdTier::Full;
        }
        let mut best = f64::INFINITY;
        for a in &self.anchors {
            let (dx, dz) = (p.x - a.x, p.z - a.z);
            let d2 = dx * dx + dz * dz;
            if d2 < best {
                best = d2;
            }
        }
        // `sqrt` and not a squared comparison: IEEE-754 specifies `sqrt`
        // exactly, so the metre-space compare is portable, and the radii are
        // metres everywhere else in this file. NaN falls through every branch.
        let d = best.sqrt();
        if d <= self.full_m {
            CrowdTier::Full
        } else if d <= self.near_m {
            CrowdTier::Near
        } else if d <= self.far_m {
            CrowdTier::Far
        } else {
            CrowdTier::Dormant
        }
    }
}

#[inline]
fn lattice(v: f64) -> i64 {
    let q = (v / BAND_LATTICE_M).floor();
    if q <= -(i64::MAX as f64) {
        i64::MIN + 1
    } else if q >= i64::MAX as f64 {
        i64::MAX
    } else {
        q as i64
    }
}

#[inline]
fn unlattice(c: i64) -> f64 {
    (c as f64 + 0.5) * BAND_LATTICE_M
}

// ── per-agent randomness ────────────────────────────────────────────────────

/// The salt an agent's route speed multiplier is drawn with.
pub const SALT_SPEED: u64 = 0x5350_4545_4400_0001;

/// The salt an agent's route phase offset is drawn with.
pub const SALT_PHASE: u64 = 0x5048_4153_4500_0002;

/// **`mix64(guid ^ tick ^ salt)`** — the house RNG doctrine, as a function.
///
/// There is no engine RNG (no `rand` dependency anywhere in this tree) and there
/// must not be one on a sim path: a stateful generator is state, and state that
/// is not folded into `state_bytes` breaks parity the first time one host starts
/// mid-trace. So every per-agent draw is a **pure function of sim state** — the
/// agent's stable `Guid`, the fixed step it is drawn on, and a compile-time salt
/// naming what it is for.
///
/// The mixer is the SplitMix64 finalizer, the same *specification*
/// `inf_pcg::hash`, `inf_mesh::fracture` and `inf_photo::hash` each spell out;
/// `the_mixer_is_the_splitmix64_finalizer` pins it against the constants rather
/// than against one of those copies.
#[inline]
pub fn agent_rand(guid: Uuid, tick: u64, salt: u64) -> u64 {
    const GOLDEN: u64 = 0x9e37_79b9_7f4a_7c15;
    let bits = guid.as_u128();
    let lo = bits as u64;
    let hi = (bits >> 64) as u64;
    let mut x = lo ^ hi.wrapping_mul(GOLDEN) ^ tick.wrapping_mul(GOLDEN) ^ salt;
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// [`agent_rand`] as a uniform in `[0, 1)`.
///
/// The top 53 bits over `2^53` — exact in IEEE-754 double, and therefore the
/// same on every target, which `powf`-based scalings are not.
#[inline]
pub fn agent_unit(guid: Uuid, tick: u64, salt: u64) -> f64 {
    (agent_rand(guid, tick, salt) >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
}

// ── what an agent looks like (wave NPC1b) ───────────────────────────────────

/// The salt an agent's palette-swap look is drawn with.
pub const SALT_LOOK: u64 = 0x4c4f_4f4b_0000_0003;

/// The salt an agent's build (its drawn height and girth) is drawn with.
pub const SALT_BUILD: u64 = 0x4255_494c_4400_0004;

/// **Which committed body an agent wears** (wave CHAR1a.2) — see
/// [`crate::society::level_archetype_for`]. Its own salt, so a level that grew a
/// second body could not move any existing agent's speed, phase, schedule, look
/// or build: every one of those is drawn from the same generator with a salt of
/// its own, which is the whole reason the salts are named constants.
pub const SALT_BODY: u64 = 0x424f_4459_0000_0006;

/// **The crowd's palette swaps** — linear-space multipliers over whatever base
/// colour the archetype's material resolves to.
///
/// Eight, because a crowd wants to stop reading as clones and does not want to
/// read as a paint chart: at eight looks a group of six is very unlikely to be
/// uniform and no look is rare enough to feel like a special character. They are
/// *multipliers*, not colours, so a character with an authored material keeps its
/// own material and takes the variation on top — which is what makes this work
/// for content that does not exist yet.
///
/// **Body variation without rig variation**, which is the shape
/// [`CrowdArchetype`]'s own doc names: N NPCs share one mesh, one skeleton, one
/// `.inf_sm` and every clip, and what makes them different is per-instance data
/// the renderer already carries. NPC1a deliberately did not smuggle it in here;
/// this is where it lands.
pub const CROWD_LOOKS: [[f32; 3]; 8] = [
    [1.00, 0.98, 0.94], // bone
    [0.52, 0.60, 0.82], // denim
    [0.58, 0.62, 0.42], // olive
    [0.86, 0.52, 0.36], // rust
    [0.38, 0.40, 0.44], // charcoal
    [0.92, 0.82, 0.62], // sand
    [0.40, 0.70, 0.70], // teal
    [0.66, 0.34, 0.40], // maroon
];

/// The narrowest and widest an agent is drawn, as a multiplier on its archetype's
/// proportions.
///
/// **±8 %, and the bound is a physics bound rather than a taste one.** A crowd
/// agent's *collider* is its archetype's capsule and this multiplier does not
/// reach it (see [`agent_look`]), so every centimetre of it is a centimetre of
/// disagreement between what a player sees and what they can walk into. Eight
/// per cent of a 1.8 m adult is 14 cm of height and 2.4 cm of radius — inside the
/// slack a capsule already has around a humanoid mesh, and visible enough that a
/// row of six does not read as one model repeated.
pub const CROWD_BUILD_RANGE: (f32, f32) = (0.92, 1.08);

/// **What one crowd agent looks like** — derived from its `Guid` and nothing
/// else, and therefore never stored.
///
/// A stored look would be a second copy of a pure function, and the copy is what
/// drifts (NPC1a's ruling about the speed multiplier and the route phase, met
/// again). It is also why this is not a component: nothing here is sim state, so
/// nothing here reaches `state_bytes`, and **PIE equals shipping for the same
/// reason it did before the crowd had a face** — both projectors call this one
/// door with the same `Guid` and get the same answer.
///
/// Drawn at `tick = 0`, so an agent's look does not change as it walks.
#[inline]
pub fn agent_look(guid: Uuid) -> CrowdLook {
    // Through `derived_outfit` (wave EMS3), so the default an absent
    // `AppearanceRes` falls back to and the draw this function makes are ONE
    // expression: a second spelling of `% CROWD_LOOKS.len()` is a second answer
    // to "what is this person wearing before anybody dressed them", and the
    // whole appearance channel rests on the two being the same number.
    let swap = derived_outfit(guid) as usize;
    let u = agent_unit(guid, 0, SALT_BUILD) as f32;
    let (lo, hi) = CROWD_BUILD_RANGE;
    CrowdLook {
        tint: CROWD_LOOKS[swap],
        build: lo + (hi - lo) * u,
    }
}

/// **What somebody is WEARING** (wave EMS3) — the appearance channel, and the
/// one thing about a person a witness can actually describe.
///
/// # Why this type exists at all, which is the wave's headline
///
/// Wave WPN1 recorded a description as `crate::witness::look_digest`, which was
/// `agent_rand(guid, 0, SALT_LOOK)` — **the agent's identity, hashed**. Two
/// people in the same coat got different numbers and one person who changed
/// coats kept theirs, so a police force keyed on it would have been recognising
/// *who you are* through a channel dressed up as *what you look like*. That is
/// omniscient tagging: no clothes change could ever have defeated it, and the
/// mandate's whole sentence — *"the user will realistically have to ditch their
/// car and/or their clothes to evade the police"* — was unimplementable while
/// it stood.
///
/// So an appearance is a **value**, not an identity: the index of the palette
/// swap the person is drawn in. Two people in the same swap describe
/// identically, and one person who changes swap describes differently (which is
/// what makes a wardrobe worth walking into).
///
/// The first half of that is a property of the **channel** and not yet of the
/// world (EMS3 audit): the recognition pass scores every suspect against their
/// own file and never scores somebody who has no file, so a civilian in a wanted
/// man's coat collides here and is never looked at. See
/// `inf_physics::d3::crime::look`, which carries the cost argument and the arm
/// that pins it.
///
/// # It defaults to the DERIVED draw, and that is what keeps the tree still
///
/// [`agent_look`] has drawn every crowd agent's tint from its `Guid` since
/// NPC1b, and both projectors read it. An `Appearance` is therefore **absent**
/// for everybody until something dresses them, and absent means
/// [`derived_outfit`] — the exact swap `agent_look` already chose. Every level
/// committed before this wave draws the same pixels and folds the same bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct Appearance {
    /// Which of [`CROWD_LOOKS`] this person is wearing. Out-of-range values are
    /// wrapped by [`CrowdLook::of`] rather than refused, because an appearance
    /// is content and content that is out of range should be visible rather
    /// than fatal.
    pub outfit: u8,
}

impl Appearance {
    /// **This appearance as one number** — the description a witness gives.
    ///
    /// `mix64` over the outfit and nothing else. A digest rather than the raw
    /// index because it is compared against a *record* of what somebody was
    /// wearing and a record wants a fixed-width opaque token, and because the
    /// day a second channel joins the outfit (a build, a hat) it folds in here
    /// without changing a single caller.
    ///
    /// **No `Guid` reaches this function**, which is the police-don't-cheat law
    /// in its compile-checked form: a description cannot accidentally become an
    /// identity if the identity is not in scope.
    #[inline]
    pub fn digest(self) -> u64 {
        // The mixer is `agent_rand`'s, asked of a VALUE rather than of a guid —
        // one `Uuid::from_u64_pair(0, outfit)` would work and would put a guid
        // shape in a function whose whole point is that there is no guid in it.
        let mut x = u64::from(self.outfit) ^ SALT_LOOK;
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        x ^ (x >> 31)
    }
}

/// **The swap this guid is drawn in when nothing has dressed it** — NPC1b's own
/// draw, hoisted so the default and the override cannot disagree.
#[inline]
pub fn derived_outfit(guid: Uuid) -> u8 {
    (agent_rand(guid, 0, SALT_LOOK) % CROWD_LOOKS.len() as u64) as u8
}

/// **Who is wearing what** (wave EMS3) — the appearance channel, per guid.
///
/// [`crate::item::ItemDefs`]' shape, a fourth time: derived at run time, nothing
/// can save it, and **no schema moves** — scene v27 and `ScenePayload` v13
/// stand. A body's clothes are not a component for [`PanickedRes`]' reason
/// exactly: a `Dormant` crowd agent has no entity at all and is still a person
/// wearing something, and a marker component would have been silently absent on
/// every one of them.
///
/// **Sparse on purpose.** Only somebody who has *changed* has an entry, so a
/// city of four hundred residents costs nothing until one of them opens a
/// wardrobe.
#[derive(Resource, Default, Debug, Clone, PartialEq, Eq)]
pub struct AppearanceRes {
    /// The overrides, in `Guid` order.
    pub worn: std::collections::BTreeMap<Uuid, Appearance>,
}

/// **What this person is wearing** — the override if there is one, the derived
/// draw otherwise. The ONE reader.
pub fn appearance_of(world: &EcsWorld, guid: Uuid) -> Appearance {
    world
        .world()
        .get_resource::<AppearanceRes>()
        .and_then(|r| r.worn.get(&guid).copied())
        .unwrap_or(Appearance {
            outfit: derived_outfit(guid),
        })
}

/// **Dress somebody.** The one door, so a second producer cannot invent a second
/// shape of the channel.
///
/// Returns whether it changed anything — an engagement counter, and the thing a
/// wardrobe press reports: pressing E on a wardrobe you are already dressed out
/// of is a refusal, and a refusal is a value.
pub fn set_appearance(world: &mut EcsWorld, guid: Uuid, look: Appearance) -> bool {
    if appearance_of(world, guid) == look {
        return false;
    }
    let mut res = world
        .world_mut()
        .remove_resource::<AppearanceRes>()
        .unwrap_or_default();
    res.worn.insert(guid, look);
    world.world_mut().insert_resource(res);
    true
}

/// **Forget who changed their clothes** — [`clear_crowd`]'s twin, for its
/// reason: an editor Simulate session must leave nothing behind in the author's
/// document, and a resource is outside the `ScenePersist::Memory` snapshot by
/// construction.
pub fn clear_appearance(world: &mut EcsWorld) {
    world.world_mut().remove_resource::<AppearanceRes>();
}

/// **What one agent looks like RIGHT NOW** (wave EMS3) — [`agent_look`] with the
/// appearance channel read.
///
/// The door both projectors call. It is a *different function* from
/// [`agent_look`] rather than a changed one because the two answer different
/// questions: `agent_look` is the draw (a pure function of the guid, and still
/// the default), and this is the draw **as worn**. The build is untouched —
/// nobody changes their height at a wardrobe.
#[inline]
pub fn agent_look_in(world: &EcsWorld, guid: Uuid) -> CrowdLook {
    CrowdLook {
        tint: CrowdLook::of(appearance_of(world, guid).outfit).tint,
        build: agent_look(guid).build,
    }
}

/// One agent's drawn variation — see [`agent_look`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CrowdLook {
    /// A linear-space multiplier over the base colour, from [`CROWD_LOOKS`].
    pub tint: [f32; 3],
    /// A uniform multiplier on the agent's **drawn** scale, inside
    /// [`CROWD_BUILD_RANGE`].
    ///
    /// Drawn and not simulated: it multiplies the render instance's scale and
    /// leaves the rapier capsule at the archetype's own proportions. That is the
    /// honest v1 — a per-agent collider is a per-agent `Collider3D`, which is sim
    /// state, which is a schema question and a trace question and belongs with the
    /// wave that gives a near agent a controller.
    pub build: f32,
}

impl CrowdLook {
    /// **The look one outfit index draws**, wrapping out-of-range values — the
    /// one place [`CROWD_LOOKS`] is indexed, so an [`Appearance`] a wardrobe or
    /// a mod handed a `200` tints rather than panics.
    ///
    /// The build is [`CROWD_BUILD_RANGE`]'s midpoint, because an outfit says
    /// nothing about a body; [`agent_look_in`] overwrites it with the wearer's
    /// own draw.
    #[inline]
    pub fn of(outfit: u8) -> Self {
        let (lo, hi) = CROWD_BUILD_RANGE;
        Self {
            tint: CROWD_LOOKS[outfit as usize % CROWD_LOOKS.len()],
            build: (lo + hi) * 0.5,
        }
    }

    /// This look applied to a base colour, alpha untouched.
    #[inline]
    pub fn over(self, base: [f32; 4]) -> [f32; 4] {
        [
            base[0] * self.tint[0],
            base[1] * self.tint[1],
            base[2] * self.tint[2],
            base[3],
        ]
    }
}

// ── the route ───────────────────────────────────────────────────────────────

/// What an agent does when it runs out of route (wave NPC1c).
///
/// Not a wire enum and not persisted — a route is runtime state on a resource,
/// so this costs no schema ladder and no freeze pin.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RouteMode {
    /// There and back for ever. NPC1a's only behaviour, and what a patrol,
    /// a fixture and a stand all are.
    #[default]
    PingPong,
    /// Walk it once and **arrive** — the shape a real errand takes, and the one
    /// [`RouteProgress::arrived`] is about.
    Once,
    /// Round and round: a chain whose ends coincide, walked for ever forwards.
    Loop,
}

/// **Where an agent is along its route** — [`CrowdRoute::progress_at`]'s answer,
/// in metres of arc length rather than in a normalized `t`.
///
/// Arc length is what makes a *speed* in m/s mean what it says on an uneven
/// polyline, and it is the currency the near tiers' pursuit and the demotion
/// re-phase are both expressed in.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RouteProgress {
    /// Metres along the path, clamped into `[0, length]`.
    pub s_m: f64,
    /// Whether arc length is increasing this step. Always `true` for
    /// [`RouteMode::Once`] and [`RouteMode::Loop`]; a ping-pong on its way home
    /// answers `false`, and that is what tells the near tiers' pursuit which way
    /// to look.
    pub forward: bool,
    /// Whether a [`RouteMode::Once`] route has run out. Never `true` for the
    /// other two — they do not end.
    pub arrived: bool,
}

/// **Where an agent is at a given sim time** — a pure function of the record, the
/// clock and one phase, over an [`inf_nav::NavPath`].
///
/// # What NPC1c changed, and what it deliberately did not
///
/// NPC1a's route was a straight there-and-back between two `DVec3`s, chosen so
/// that a tier measurement measured the *tier* and not a graph search. NPC1c
/// replaces the two points with a **path** — a shared, arc-length-parameterized
/// polyline that a Dijkstra over the road network, a settlement's street grid or
/// a building's own corridors produced — and adds a [`RouteMode`] so a route can
/// end.
///
/// What did not change is the arithmetic. A two-point [`RouteMode::PingPong`]
/// route computes the **same bits** it computed before this wave:
/// `inf_nav::NavPath::position_at` interpolates `from + (to - from) · (s / len)`
/// with `len` built out of the same three multiplies and one `sqrt`, and the
/// fold below is the same `rem_euclid`. `a_two_point_path_interpolates_exactly_as_the_npc1a_route_did`
/// pins it by `to_bits()`. That is what let a population that was already
/// walking move onto this substrate without moving.
///
/// The path is an `Arc`, so a thousand agents sent down one street hold one copy
/// of it and cloning a record is a refcount bump.
///
/// Every operation is IEEE-exact (`+ - * / sqrt %`), so two machines derive the
/// same metre — the P14 portable-math law, which binds here because this output
/// reaches a `Transform` and therefore the sim trace.
#[derive(Debug, Clone, PartialEq)]
pub struct CrowdRoute {
    /// The spine, in world metres, ground-snapped **when it was built**.
    pub path: inf_nav::NavPath,
    /// Metres per second along it. Non-positive is a stand.
    ///
    /// This is the speed the tiers with **no controller** walk at. A `Full` or
    /// `Near` agent walks at its [`crate::components::CharacterMovement`]'s own
    /// gait speed instead, because a body integrated by the movement model is
    /// the movement model's to move — and the difference between the two is
    /// exactly what [`CrowdRecord::rephase_m`] absorbs on the way back down.
    pub speed_mps: f64,
    /// What happens at the end.
    pub mode: RouteMode,
}

impl CrowdRoute {
    /// A route that stands still at `p`.
    pub fn standing(p: DVec3) -> Self {
        Self {
            path: inf_nav::NavPath::single(p),
            speed_mps: 0.0,
            mode: RouteMode::PingPong,
        }
    }

    /// NPC1a's route: a straight there-and-back between two points.
    ///
    /// Kept as a named constructor rather than as the only shape, because it is
    /// what every fixture and every tier measurement in this tree is written
    /// against and its bytes must not move.
    pub fn between(from: DVec3, to: DVec3, speed_mps: f64) -> Self {
        Self {
            path: inf_nav::NavPath::new([from, to]),
            speed_mps,
            mode: RouteMode::PingPong,
        }
    }

    /// A route along a planned path — what [`inf_nav::route()`] hands back.
    pub fn along(path: inf_nav::NavPath, speed_mps: f64, mode: RouteMode) -> Self {
        Self {
            path,
            speed_mps,
            mode,
        }
    }

    /// Where the route starts. A stand's own place.
    pub fn origin(&self) -> DVec3 {
        self.path.points()[0]
    }

    /// Where the route ends.
    pub fn destination(&self) -> DVec3 {
        let pts = self.path.points();
        pts[pts.len() - 1]
    }

    /// Total arc length, metres.
    pub fn length_m(&self) -> f64 {
        self.path.length_m()
    }

    /// Whether this route goes anywhere at all — a finite positive speed over a
    /// path with length in it.
    ///
    /// Spelled as one conjunction rather than as negated comparisons: `!(len >
    /// 0.0)` reads NaN correctly and reads as a clippy lint, and a route with a
    /// NaN end has to answer its origin rather than wander.
    pub fn is_walkable(&self) -> bool {
        let len = self.length_m();
        len.is_finite() && len > 0.0 && self.speed_mps.is_finite() && self.speed_mps > 0.0
    }

    /// Metres travelled by sim time `t_s`, before the fold — the scalar every
    /// other question here is asked of.
    pub fn travelled_at(&self, t_s: f64, phase_m: f64) -> f64 {
        self.speed_mps * t_s + phase_m
    }

    /// **How far along the path the clock says the agent is**, folded by the
    /// route's own [`RouteMode`].
    pub fn progress_at(&self, t_s: f64, phase_m: f64) -> RouteProgress {
        let len = self.length_m();
        if !self.is_walkable() || !t_s.is_finite() || !phase_m.is_finite() {
            return RouteProgress {
                s_m: 0.0,
                forward: true,
                // A stand has nowhere to get to, so it is always there. Reading
                // it the other way would make every idle NPC in a town
                // permanently *en route*.
                arrived: true,
            };
        }
        let travelled = self.travelled_at(t_s, phase_m);
        match self.mode {
            RouteMode::PingPong => {
                let period = 2.0 * len;
                let u = travelled.rem_euclid(period);
                let forward = u <= len;
                RouteProgress {
                    s_m: if forward { u } else { period - u },
                    forward,
                    arrived: false,
                }
            }
            RouteMode::Loop => RouteProgress {
                s_m: travelled.rem_euclid(len),
                forward: true,
                arrived: false,
            },
            RouteMode::Once => RouteProgress {
                s_m: travelled.clamp(0.0, len),
                forward: true,
                arrived: travelled >= len,
            },
        }
    }

    /// The agent's position at sim time `t_s`.
    pub fn position_at(&self, t_s: f64, phase_m: f64) -> DVec3 {
        self.path.position_at(self.progress_at(t_s, phase_m).s_m)
    }

    /// **The phase change that puts `route(clock)` at `s_m`** — the arithmetic
    /// that makes a demotion continuous.
    ///
    /// A `Near` agent's body is moved by the movement model, so by the time it
    /// drops to `Far` the clock and the body disagree: the body was blocked by a
    /// wall, or slowed on a slope, or simply walks at its own gait rather than
    /// at [`speed_mps`](Self::speed_mps). Snapping to `route(clock)` would
    /// teleport it by however much it had lagged — measured on a blocked agent,
    /// metres. So the *clock* is moved onto the body instead, once, at the
    /// transition, and everything downstream stays a pure function of the route
    /// and the clock.
    ///
    /// The branch matters: on a ping-pong's return leg the same `s_m` is a
    /// different `travelled`, so the leg the clock is *currently* on is what the
    /// solution is taken on.
    pub fn rephase_delta(&self, travelled_now: f64, s_m: f64) -> f64 {
        let len = self.length_m();
        if !self.is_walkable() || !travelled_now.is_finite() || !s_m.is_finite() {
            return 0.0;
        }
        let s = s_m.clamp(0.0, len);
        let want = match self.mode {
            RouteMode::Once => s,
            RouteMode::Loop => {
                let k = (travelled_now / len).floor();
                k * len + s
            }
            RouteMode::PingPong => {
                let period = 2.0 * len;
                let k = (travelled_now / period).floor();
                let u = travelled_now - k * period;
                if u <= len {
                    k * period + s
                } else {
                    k * period + (period - s)
                }
            }
        };
        want - travelled_now
    }
}

// ── the day ─────────────────────────────────────────────────────────────────

/// **What the crowd is told about time** (NPC1d) — the sim clock and the level
/// clock, taken together in one place.
///
/// Two numbers rather than one because a crowd answers two different questions.
/// `t_s` is *how long the simulation has run*, which is what an unscheduled
/// route ping-pongs on and what every NPC1a/b/c fixture in this tree is written
/// against. `hour` is *what time of day it is on this level*, which is the only
/// thing a [`CrowdSchedule`] reads: a society goes to work at eight o'clock, not
/// at the four-thousandth fixed step.
///
/// Keeping them apart is what lets the island run its day at any rate and get
/// the same society: a leg's position is a fraction of its own clock WINDOW, so
/// every agent is in the same place at the same hour whether the day takes
/// forty-eight minutes or four. What a faster clock does change is the metres
/// per second a walk implies — see [`ScheduleLeg::implied_speed_mps`].
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct CrowdClock {
    /// Sim seconds since the population was installed — `steps` times `dt`,
    /// never an accumulated `+= dt`.
    pub t_s: f64,
    /// The **local** hour of day, `[0, 24)`. `0.0` on a level with no clock,
    /// which is [`crate::sky::local_hour`]'s own answer.
    pub hour: f64,
}

impl CrowdClock {
    /// A clock with no day in it — sim seconds alone.
    ///
    /// The right constructor for every fixture written before this wave and for
    /// every record with no schedule, and it is spelled out rather than
    /// defaulted so a caller that *does* have an hour cannot lose it by
    /// accident.
    pub fn at(t_s: f64) -> Self {
        Self { t_s, hour: 0.0 }
    }

    /// Both halves.
    pub fn new(t_s: f64, hour: f64) -> Self {
        Self { t_s, hour }
    }

    /// The clock this world is on at sim time `t_s`.
    pub fn from_world(world: &EcsWorld, t_s: f64) -> Self {
        Self {
            t_s,
            hour: crate::sky::local_hour(world),
        }
    }
}

/// Salts the per-agent schedule jitter. See [`SCHEDULE_JITTER_H`].
pub const SALT_SCHEDULE: u64 = 0x5343_4845_4400_0005;

/// **How much of an hour an agent's own day is shifted by**, either way.
///
/// A town whose four hundred residents all leave at exactly eight o'clock is a
/// tide rather than a street, and the pursuit at `Full` would put every one of
/// them on the same waypoint on the same step. Half an hour each way is drawn
/// once per agent from [`SALT_SCHEDULE`] at `tick = 0` — derived and never
/// stored, on [`CrowdRecord::speed_of`]'s own terms, because a jitter written
/// into the record at spawn time is a second copy of a pure function and the
/// copy is what drifts.
pub const SCHEDULE_JITTER_H: f64 = 0.5;

/// **The most legs one day may have** (NPC1d audit) — 256, the range of the
/// one byte of a schedule that reaches the replay trace.
///
/// [`CrowdRecord::leg`] is a `u8` because a day is four legs and a byte holds
/// two hundred and fifty-six of them. That sentence was a doc comment and
/// nothing else until this audit: a 300-leg schedule would have aliased leg 0
/// with leg 256, and the step's "a new leg starts clean" comparison would have
/// read *no change* across the wrap and carried a phase it exists to drop.
/// [`CrowdSchedule::new`] refuses past it.
pub const MAX_SCHEDULE_LEGS: usize = 256;

/// **One leg of an agent's day**: leave at an hour, walk a route, and stand at
/// the far end until the next leg begins.
///
/// # The walk is a fraction of its WINDOW, not a speed
///
/// A leg's position is `length` times `(elapsed / travel_h)`, clamped — so an
/// agent is at the same *place* at the same *hour* whatever rate the level's
/// clock runs at. That is what makes a day in the life provable in a test
/// process at all: twenty-four hours of the island's own authored rate — 18, an
/// eighty-minute day — is **288 000** fixed steps a host, and the same day
/// compressed is the same day, arriving sooner. *(172 800 until the NPC1d
/// audit, which is the number for the rate-30 first draft the wave measured its
/// way off.)*
///
/// The price is stated rather than hidden: at a compressed rate the implied
/// walking speed is not a walking speed, and a [`Full`](CrowdTier::Full) body —
/// which is moved by its own gait through `move_and_slide` — falls behind its
/// clock and reads `blocked`. That is the NPC1c steady state already (96 % of
/// the town walk's steps), and [`implied_speed_mps`](Self::implied_speed_mps) is
/// what an arm holds against the rate a level actually authors.
#[derive(Debug, Clone, PartialEq)]
pub struct ScheduleLeg {
    /// The local hour this leg begins, `[0, 24)`.
    pub start_h: f64,
    /// How long the walk itself takes, in hours of the same clock. The agent
    /// stands at the destination from `start_h + travel_h` until the next leg.
    pub travel_h: f64,
    /// The route walked, in world metres, ground-snapped when it was built.
    pub path: inf_nav::NavPath,
    /// **What the body does once it gets there** (wave VEN1b).
    ///
    /// The leg carries it rather than the record, because a day is a sequence
    /// of *arrivals*: the same agent stands at a desk at ten, sits on a bar
    /// stool at eleven at night and lies in bed at three, and a posture stored
    /// on the record could only ever describe one of them.
    ///
    /// [`SlotArrival::STANDING`] for every leg planned before this wave, which
    /// is what keeps a pre-VEN1b level's trace byte-identical: a `Stand`
    /// arrival with no facing is exactly the behaviour that had no name.
    pub arrival: SlotArrival,
}

/// **What a body does at the far end of a leg** (wave VEN1b) — a posture and,
/// optionally, a direction to face.
///
/// A pair rather than two fields on [`ScheduleLeg`], because they are answered
/// together by one `SocietyPlace` and a leg that got one without the other
/// would be a seated body facing wherever it happened to walk in from.
///
/// Derived at plan time and never mutated, so it costs the step nothing: the
/// crowd reads it off the leg the clock already named
/// ([`CrowdRecord::arrival_on`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SlotArrival {
    /// **What kind of place this leg ends at**, or `None` for a leg with no
    /// destination — which is every record with no schedule.
    ///
    /// # This is the only thing that can tell a shift from a walk home
    ///
    /// `HOME_H` and `NIGHT_WORK_START_H` are **both eighteen hundred**: the
    /// town comes home at the hour a bar's keeper leaves for work, which is
    /// exactly right and makes a leg's `start_h` useless as an identity.
    /// Measured, on a real nightclub: a classifier that read "the active leg
    /// started at `NIGHT_WORK_START_H`" counted **155 night workers** in a town
    /// with **31 night jobs** — every ordinary resident standing at its own
    /// front door.
    ///
    /// So an arrival names the kind of place it is. Nothing about a leg's
    /// *hour* is load-bearing any more, which is the property that lets the
    /// hour table be re-tuned without silently re-labelling the population.
    pub role: Option<crate::components::SlotRole>,
    /// The posture — see [`crate::components::SlotPosture`].
    pub posture: crate::components::SlotPosture,
    /// A unit XZ direction to face, or `DVec3::ZERO` to keep whatever facing
    /// the walk left the body with.
    pub face: DVec3,
}

impl SlotArrival {
    /// **On both feet, going nowhere in particular** — the arrival every record
    /// with no schedule wears, and every fixture written before wave VEN1b.
    pub const STANDING: Self = Self {
        role: None,
        posture: crate::components::SlotPosture::Stand,
        face: DVec3::ZERO,
    };
}

impl Default for SlotArrival {
    fn default() -> Self {
        Self::STANDING
    }
}

impl ScheduleLeg {
    /// **The metres per second this leg implies at a given clock rate** — the
    /// bridge between a schedule written in hours and a body that walks.
    ///
    /// `rate` is [`crate::components::TimeOfDay::rate`], clock-seconds per sim
    /// second. A frozen clock (`0`) has no answer and returns `0.0`.
    pub fn implied_speed_mps(&self, rate: f64) -> f64 {
        let window_s = self.travel_h * 3600.0 / rate;
        if !(window_s.is_finite() && window_s > 0.0) {
            return 0.0;
        }
        self.path.length_m() / window_s
    }
}

/// **Which leg an agent is on and how far through that leg's walk the clock
/// is** — `None` for a record with no schedule (island wave NPC1e).
///
/// # Why this is a type and not four calls to [`CrowdSchedule::at`]
///
/// Resolving it is not free: [`CrowdRecord::hour_of`] is a `mix64` and a
/// `rem_euclid`, and [`CrowdSchedule::at`] is another `rem_euclid` per leg. The
/// step used to ask for it **four times an agent** (the leg-change check, then
/// `progress_at`, then `position_at` → `feet_at` → `path_at` → `progress_at`
/// again), and a *steered* agent a fifth time inside the pursuit. NPC1d measured
/// what that costs and carried it as item 11: the island's 329 scheduled
/// residents read `crowd` **0.353 ms — 1.07 µs an agent —** against the
/// instrument's thousand ping-pong walkers at **0.164 ms, 0.16 µs** — **6.5×**.
///
/// The lever that item names and refuses is *"cache the leg index on the
/// record"*, because a stored index is a second copy of a pure function and this
/// module has refused two of those already (the speed multiplier and the route
/// phase). **This is the other lever**: resolve it once and pass it down, which
/// is the P29.4-A8 shape — `step_character_movement` walks its character list
/// once and hands the slice to `try_mantle` rather than letting it ask again.
/// Nothing is stored, nothing can drift, and every `*_on` door below answers
/// exactly what its `*_at` sibling answers.
pub type ActiveLeg = Option<(usize, f64)>;

/// **An agent's whole day**, as legs on the level clock.
///
/// A schedule is a pure function of `(agent seed, clock)` and holds no
/// progression state: ask it what hour it is and it answers which leg is running
/// and how far through that leg's walk the clock has got. Nothing here counts,
/// accumulates or remembers, which is what lets two hosts that started at
/// different steps agree — and what lets a `Dormant` agent, which has no entity
/// at all, still be *at work at two in the afternoon*.
///
/// That last sentence is NPC1c's inherited item about `Dormant` re-materializing
/// at `last`, closed: a dormant agent's place has been `route(clock)` since the
/// NPC1a audit, and this makes the route a *day* rather than a ping-pong.
#[derive(Debug, Clone, PartialEq)]
pub struct CrowdSchedule {
    legs: Vec<ScheduleLeg>,
}

impl CrowdSchedule {
    /// A schedule over `legs`, or `None` if there is nothing walkable in it.
    ///
    /// Legs with a non-finite or non-positive `travel_h`, a non-finite
    /// `start_h`, or a non-finite path length are dropped — a refusal is a
    /// value (P21.4), and a leg that cannot be walked is not a leg. The order
    /// the caller gives is kept: [`at`](Self::at) picks by clock rather than by
    /// index, so nothing here needs sorting.
    ///
    /// **And more than [`MAX_SCHEDULE_LEGS`] walkable legs is refused whole**
    /// (NPC1d audit) rather than truncated. [`CrowdRecord::leg`] is the one
    /// byte of this that reaches the replay trace, so a schedule past the byte
    /// would alias leg 0 with leg 256 — the step would read "no change" across
    /// that boundary and keep a phase it is meant to drop. The doc on
    /// [`AGENT_TRACE_BYTES`] said "a schedule with more than 255 legs is not a
    /// day"; a hazard written into a doc is not a hazard handled, so it is a
    /// value now. Refused rather than clipped because a caller silently given
    /// the first 256 of its 300 legs has a day that ends at teatime.
    pub fn new(legs: Vec<ScheduleLeg>) -> Option<Self> {
        let legs: Vec<ScheduleLeg> = legs
            .into_iter()
            .filter(|l| {
                l.start_h.is_finite()
                    && l.travel_h.is_finite()
                    && l.travel_h > 0.0
                    && l.path.length_m().is_finite()
            })
            .collect();
        (!legs.is_empty() && legs.len() <= MAX_SCHEDULE_LEGS).then_some(Self { legs })
    }

    /// The legs, in the order they were given.
    pub fn legs(&self) -> &[ScheduleLeg] {
        &self.legs
    }

    /// **Which leg is running at `hour`, and how far through its walk the clock
    /// is** — `0.0` at the moment it starts, `1.0` once the walk is done and
    /// the agent is standing at the far end.
    ///
    /// The active leg is the one that started **most recently**, measured as
    /// `(hour - start_h) mod 24`: the smallest such gap wins, ties keep the
    /// lower index. Written that way rather than as a sorted search with a
    /// midnight special case because midnight is not special — a day is a
    /// circle, and the modulo is the circle.
    pub fn at(&self, hour: f64) -> (usize, f64) {
        if !hour.is_finite() {
            return (0, 1.0);
        }
        let mut best = (0usize, f64::INFINITY);
        for (i, leg) in self.legs.iter().enumerate() {
            let d = (hour - leg.start_h).rem_euclid(24.0);
            if d < best.1 {
                best = (i, d);
            }
        }
        let leg = &self.legs[best.0];
        (best.0, (best.1 / leg.travel_h).clamp(0.0, 1.0))
    }
}

// ── the population ──────────────────────────────────────────────────────────

/// What a crowd NPC is made of, so a [`Dormant`](CrowdTier::Dormant) record can
/// build one back.
///
/// Assets are GUIDs and nothing else: N NPCs on one mannequin share every
/// buffer, every clip, every `.inf_sm` and every `.inf_skel` (the renderer
/// Arc-dedupes by `(mesh, skeleton)`), so a thousand records naming one
/// archetype cost one of each. Body variation without rig variation is NPC1b's
/// packed-channel work and is deliberately not smuggled in here.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct CrowdArchetype {
    /// `.inf_mesh` with per-vertex skin.
    pub mesh: Option<Uuid>,
    /// `.inf_skel` binding its joint indices.
    pub skeleton: Option<Uuid>,
    /// `.inf_sm` the agent's machine plays.
    pub sm: Option<Uuid>,
    /// Capsule half-height, metres (the segment half-length, `Collider3D`'s
    /// convention).
    pub half_height_m: f64,
    /// Capsule radius, metres.
    pub radius_m: f64,
}

impl CrowdArchetype {
    /// **How far below its transform this archetype's feet are**, metres.
    ///
    /// P29.6's ruling, applied to a crowd: a rig is authored with its feet at the
    /// origin and a character's entity transform is its capsule **centre**, so
    /// the two differ by `half_height + radius` — 1.20 m on the humanoid. It is
    /// spelled here rather than only in `movement::feet_offset_m` because a
    /// `Far` agent has no `CharacterMovement` and no `Collider3D` for that
    /// function to read, and it still has feet.
    ///
    /// The two must agree for the tiers that have both, and
    /// `the_crowds_foot_offset_is_the_one_the_movement_model_computes` is the
    /// arm that says so.
    #[inline]
    pub fn feet_offset_m(&self) -> f64 {
        self.half_height_m + self.radius_m
    }

    /// **The starter character's proportions — a 1.8 m adult**: 0.6 m
    /// half-height, 0.3 m radius, which is
    /// [`CharacterMovement::stand_half_height_m`]'s own default for the same
    /// body.
    ///
    /// # It read 0.9 until NPC1c, and nothing could see it
    ///
    /// A capsule's `half_height` in this engine is the segment half-length,
    /// **excluding the radius** — the component's own doc says so — so
    /// `2 (0.9 + 0.3)` is a **2.4 m** capsule wearing a comment that says 1.8.
    /// Every arm NPC1a and NPC1b wrote passed on it: a tier decision, a trace
    /// byte and a palette upload do not care how tall a capsule is, and the
    /// crowd never met anything with a ceiling.
    ///
    /// The first thing that did was a doorway. `DEFAULT_DOOR_HEIGHT_M` is 2.1 m,
    /// so a 2.4 m NPC does not fit through any door this engine builds, and the
    /// town walk found it by standing in an open doorway for 3 400 steps with
    /// the leaf swung to its full 95 degrees. It is also the other half of the
    /// NPC1b audit's **1.20 m** character-space gap: that number was
    /// `half_height + radius` on this capsule, and on the corrected one the
    /// drop is **0.90 m**.
    ///
    /// [`CharacterMovement::stand_half_height_m`]: crate::components::CharacterMovement::stand_half_height_m
    pub fn humanoid(mesh: Option<Uuid>, skeleton: Option<Uuid>, sm: Option<Uuid>) -> Self {
        Self {
            mesh,
            skeleton,
            sm,
            half_height_m: 0.6,
            radius_m: 0.3,
        }
    }
}

/// One member of the population.
///
/// `last` and `pose_digest` are the two fields that make [`Dormant`] and
/// [`Far`] honest rather than lossy: the first is where the agent stood when it
/// stopped having an entity, the second is a fold of the last pose it published
/// before it stopped evaluating one. Both are sim state, both are folded by
/// [`crowd_state_bytes`], and neither is ever written to a file.
///
/// [`Dormant`]: CrowdTier::Dormant
/// [`Far`]: CrowdTier::Far
#[derive(Debug, Clone, PartialEq)]
pub struct CrowdRecord {
    /// What it is made of.
    pub archetype: CrowdArchetype,
    /// Where it walks.
    pub route: CrowdRoute,
    /// The tier it took on the last [`step_crowd`].
    pub tier: CrowdTier,
    /// Where it stood then.
    pub last: DVec3,
    /// A fold of the last pose it published, carried while it is not evaluating
    /// one. `0` for an agent that has never posed.
    pub pose_digest: u64,
    /// **What the near tiers did to the clock** (wave NPC1c), in metres of
    /// phase.
    ///
    /// Zero for an agent that has never carried a controller, which is what
    /// makes every NPC1a fixture reproduce exactly. It moves once per demotion,
    /// by [`CrowdRoute::rephase_delta`], so the route clock lands where the
    /// *body* actually got to — see that function for why the alternative is a
    /// teleport.
    ///
    /// This is genuine sim state and not a cached pure function, which is why it
    /// is stored where the derived speed and phase draws deliberately are not:
    /// it is produced by the simulation — by a body being blocked, or slowed on
    /// a slope, or simply walking at its own gait — and two hosts that produced
    /// different values have diverged. It is folded into [`crowd_state_bytes`]
    /// for exactly that reason.
    pub rephase_m: f64,
    /// **The agent's day** (NPC1d) — `None` for a record that simply walks its
    /// [`route`](Self::route), which is every NPC1a/b/c fixture in this tree and
    /// every patrol.
    ///
    /// A schedule OVERRIDES the route: where it is present,
    /// [`progress_at`](Self::progress_at) reads the active leg and the level
    /// clock, and `route` is left as the thing the record was built with. Two
    /// fields rather than an enum because the route is what a caller hands in
    /// and the schedule is what a society derives, and a record can be promoted
    /// from one to the other without being rebuilt.
    pub schedule: Option<CrowdSchedule>,
    /// **Which leg of its schedule this agent was on last step.** Sim state, and
    /// folded into [`crowd_state_bytes`] for the reason
    /// [`rephase_m`](Self::rephase_m) is: it is what the *step* decides a
    /// re-phase against, so two hosts that disagreed about it have diverged.
    ///
    /// `0` for an unscheduled record, which never moves it.
    pub leg: u8,
}

impl CrowdRecord {
    /// A record standing at `p`.
    pub fn standing(archetype: CrowdArchetype, p: DVec3) -> Self {
        Self {
            archetype,
            route: CrowdRoute::standing(p),
            tier: CrowdTier::Dormant,
            last: p,
            pose_digest: 0,
            rephase_m: 0.0,
            schedule: None,
            leg: 0,
        }
    }

    /// A record walking `route`, starting at its origin.
    pub fn walking(archetype: CrowdArchetype, route: CrowdRoute) -> Self {
        let last = route.origin();
        Self {
            archetype,
            route,
            tier: CrowdTier::Dormant,
            last,
            pose_digest: 0,
            rephase_m: 0.0,
            schedule: None,
            leg: 0,
        }
    }

    /// **The agent's own speed**: `0.85 … 1.15` of its route's, so a population
    /// sharing one route does not march in lockstep.
    ///
    /// DERIVED and never stored, and drawn at `tick = 0` so it is a constant of
    /// the agent rather than of the step: a jitter written into the record at
    /// spawn time would be a second copy of a pure function, and the copy is the
    /// thing that drifts. A per-step draw uses the same door with the live tick.
    pub fn speed_of(&self, guid: Uuid) -> f64 {
        self.route.speed_mps * (0.85 + 0.3 * agent_unit(guid, 0, SALT_SPEED))
    }

    /// **The agent's own phase**: up to 8 m of derived head start along the
    /// path, so they do not all turn round together, plus whatever the near
    /// tiers have put into [`rephase_m`](Self::rephase_m).
    pub fn phase_of(&self, guid: Uuid) -> f64 {
        agent_unit(guid, 0, SALT_PHASE) * 8.0 + self.rephase_m
    }

    /// The record's route wearing this agent's own speed — the one place the
    /// draws are applied, so nothing downstream can apply them twice.
    fn walked(&self, guid: Uuid) -> CrowdRoute {
        CrowdRoute {
            path: self.route.path.clone(),
            speed_mps: self.speed_of(guid),
            mode: self.route.mode,
        }
    }

    /// **A record walking `schedule`** — a society's shape, and the one
    /// [`inf_ecs::society`](crate::society) mints.
    ///
    /// The `route` it carries is the first leg's path, so a caller that reads
    /// `route` (a diagnostic, a bound, an unscheduled fall-through) sees
    /// something true rather than an empty stand.
    pub fn scheduled(archetype: CrowdArchetype, schedule: CrowdSchedule) -> Self {
        let first = schedule.legs()[0].path.clone();
        let last = first.points()[0];
        Self {
            archetype,
            route: CrowdRoute::along(first, 0.0, RouteMode::Once),
            tier: CrowdTier::Dormant,
            last,
            pose_digest: 0,
            rephase_m: 0.0,
            schedule: Some(schedule),
            leg: 0,
        }
    }

    /// **The hour this agent's OWN day is on** — the level clock shifted by its
    /// derived jitter, wrapped into `[0, 24)`.
    ///
    /// Drawn at `tick = 0`, so it is a constant of the agent rather than of the
    /// step (see [`speed_of`](Self::speed_of) for why the draw is never stored).
    pub fn hour_of(&self, guid: Uuid, hour: f64) -> f64 {
        let jitter = SCHEDULE_JITTER_H * (2.0 * agent_unit(guid, 0, SALT_SCHEDULE) - 1.0);
        (hour - jitter).rem_euclid(24.0)
    }

    /// **Which leg this agent is on and how far through its walk**, or `None`
    /// for an unscheduled record.
    pub fn leg_at(&self, guid: Uuid, clock: CrowdClock) -> ActiveLeg {
        self.schedule
            .as_ref()
            .map(|s| s.at(self.hour_of(guid, clock.hour)))
    }

    /// **The path this agent is walking right now** — its active leg's, or its
    /// route's.
    pub fn path_at(&self, guid: Uuid, clock: CrowdClock) -> &inf_nav::NavPath {
        self.path_on(self.leg_at(guid, clock))
    }

    /// [`path_at`](Self::path_at) with the leg already resolved — see
    /// [`ActiveLeg`].
    pub fn path_on(&self, leg: ActiveLeg) -> &inf_nav::NavPath {
        match (&self.schedule, leg) {
            (Some(s), Some((i, _))) => &s.legs()[i].path,
            _ => &self.route.path,
        }
    }

    /// **What this agent is DOING right now** (wave VEN1b) — the active leg's
    /// [`arrival`](ScheduleLeg::arrival), once it has finished walking it.
    ///
    /// # A posture is a place you have GOT to
    ///
    /// `u >= 1.0` is the same test [`RouteProgress::arrived`] is written from,
    /// and using it is the whole of the rule: an agent halfway to the club is
    /// walking, and an agent whose leg's clock window has run out is sitting on
    /// the stool at the end of it. Nothing is stored — this is a pure function
    /// of `(schedule, clock)`, so two hosts sit the same people down without
    /// exchanging a byte, and a `Dormant` agent with no entity at all is still
    /// *seated at the bar at eleven*.
    ///
    /// [`SlotArrival::STANDING`] for an unscheduled record, which is every
    /// fixture written before this wave.
    pub fn arrival_on(&self, leg: ActiveLeg) -> SlotArrival {
        match (&self.schedule, leg) {
            (Some(s), Some((i, u))) if u >= 1.0 => s.legs()[i].arrival,
            _ => SlotArrival::STANDING,
        }
    }

    /// [`arrival_on`](Self::arrival_on) with the leg resolved from the clock —
    /// the ad-hoc door, for a reader that holds a record and a question.
    pub fn arrival_at(&self, guid: Uuid, clock: CrowdClock) -> SlotArrival {
        self.arrival_on(self.leg_at(guid, clock))
    }

    /// **How far along its route this agent is** at `clock`.
    pub fn progress_at(&self, guid: Uuid, clock: CrowdClock) -> RouteProgress {
        self.progress_on(self.leg_at(guid, clock), guid, clock)
    }

    /// [`progress_at`](Self::progress_at) with the leg already resolved.
    pub fn progress_on(&self, leg: ActiveLeg, guid: Uuid, clock: CrowdClock) -> RouteProgress {
        match (&self.schedule, leg) {
            (Some(s), Some((i, u))) => {
                let len = s.legs()[i].path.length_m();
                let raw = len * u + self.rephase_m;
                RouteProgress {
                    s_m: if raw.is_finite() {
                        raw.clamp(0.0, len)
                    } else {
                        0.0
                    },
                    // A day only ever runs forwards. The ping-pong's return leg
                    // is a patrol's idea, and a commute home is its own leg with
                    // its own path.
                    forward: true,
                    arrived: u >= 1.0,
                }
            }
            _ => self
                .walked(guid)
                .progress_at(clock.t_s, self.phase_of(guid)),
        }
    }

    /// **Where this agent's FEET are at sim time `t_s`** — the point on the
    /// route itself, with no body geometry in it.
    pub fn feet_at(&self, guid: Uuid, clock: CrowdClock) -> DVec3 {
        self.feet_on(self.leg_at(guid, clock), guid, clock)
    }

    /// [`feet_at`](Self::feet_at) with the leg already resolved.
    pub fn feet_on(&self, leg: ActiveLeg, guid: Uuid, clock: CrowdClock) -> DVec3 {
        self.path_on(leg)
            .position_at(self.progress_on(leg, guid, clock).s_m)
    }

    /// **Where this agent's TRANSFORM is at sim time `t_s`** — the record's
    /// route plus the two per-agent draws plus its own capsule, in the one place
    /// that decides them.
    ///
    /// The lift is here and nowhere else. A nav path is the ground a body walks
    /// on and an entity's transform is its capsule's centre (P29.6), so the two
    /// differ by [`CrowdArchetype::feet_offset_m`] — and a wave that computed
    /// that difference in two places got it 1.20 m wrong in one of them, which
    /// is what `a_demotion_hands_the_clock_the_metre_the_body_reached` caught.
    pub fn position_at(&self, guid: Uuid, clock: CrowdClock) -> DVec3 {
        self.position_on(self.leg_at(guid, clock), guid, clock)
    }

    /// [`position_at`](Self::position_at) with the leg already resolved.
    pub fn position_on(&self, leg: ActiveLeg, guid: Uuid, clock: CrowdClock) -> DVec3 {
        self.feet_on(leg, guid, clock) + DVec3::new(0.0, self.archetype.feet_offset_m(), 0.0)
    }

    /// Metres of route travelled by `t_s`, before the fold — what a re-phase is
    /// computed against.
    pub fn travelled_at(&self, guid: Uuid, clock: CrowdClock) -> f64 {
        self.walked(guid)
            .travelled_at(clock.t_s, self.phase_of(guid))
    }

    /// **The phase change that puts this agent's clock on `s_m`** — one door for
    /// both kinds of record, so the step below has no branch of its own.
    ///
    /// For a schedule the arithmetic is simpler than a route's, because a leg
    /// runs forwards once: the clock's own metre is `length x u`, so the phase
    /// that lands it on the body is `s_m - length x u`, and the delta is that
    /// minus whatever the phase already holds. There is no ping-pong leg to pick
    /// and no lap to count.
    pub fn rephase_delta_at(&self, guid: Uuid, clock: CrowdClock, s_m: f64) -> f64 {
        self.rephase_delta_on(self.leg_at(guid, clock), guid, clock, s_m)
    }

    /// [`rephase_delta_at`](Self::rephase_delta_at) with the leg already
    /// resolved.
    pub fn rephase_delta_on(&self, leg: ActiveLeg, guid: Uuid, clock: CrowdClock, s_m: f64) -> f64 {
        match (&self.schedule, leg) {
            (Some(s), Some((i, u))) => {
                let len = s.legs()[i].path.length_m();
                if !(len.is_finite() && s_m.is_finite() && self.rephase_m.is_finite()) {
                    return 0.0;
                }
                (s_m.clamp(0.0, len) - len * u) - self.rephase_m
            }
            _ => {
                let travelled = self.travelled_at(guid, clock);
                self.route.rephase_delta(travelled, s_m)
            }
        }
    }
}

/// **The population** — every crowd NPC a level has, whether or not it currently
/// has an entity.
///
/// A resource, so no schema moves (see the module docs). Absent until something
/// installs one, so a level with no crowd pays exactly one `get_resource` per
/// fixed step and allocates nothing — the "absent costs nothing" discipline
/// [`crate::deform`] and [`crate::cloth`] already follow.
#[derive(Resource, Debug, Clone, Default, PartialEq)]
pub struct CrowdPopulationRes {
    /// The records, in `Guid` order — so every walk over the population is a
    /// function of the level's contents and not of a hash seed.
    pub records: BTreeMap<Uuid, CrowdRecord>,
    /// Fixed steps since the population was installed. The route clock is
    /// `steps · dt` rather than an accumulated `+= dt`, so a long run cannot
    /// drift and two hosts that started at the same step agree exactly.
    pub steps: u64,
    /// **Somebody installed this population by hand** (NPC1d) — through
    /// [`set_population`], which REPLACES.
    ///
    /// The rule it carries is one sentence: *a caller that installs a population
    /// by hand owns it*, so [`crate::society`] stops deriving one. Without it an
    /// instrument that installs a measured N and then clears it would find the
    /// level's own residents walking back in on the next step, and every number
    /// it printed would be about a population it did not choose.
    ///
    /// Not folded into [`crowd_state_bytes`]: it decides whether a *derivation*
    /// runs, and two hosts that disagreed about it would disagree about the
    /// records a step later, which is where the trace already looks.
    pub hand_installed: bool,
}

/// The tier an entity's agent took this step — the component
/// [`crate::pose::step_pose_evaluation`] and the 3D bridge read.
///
/// Written only by [`step_crowd`]. Not reflected and not serialized: it is a
/// *published verdict*, like `AnimStateMachine::runtime`, and an authored one
/// would be a second opinion about a thing that has one door.
#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct CrowdAgent {
    /// This step's tier.
    pub tier: CrowdTier,
    /// The agent's own draw seed — `agent_rand(guid, tick, salt)`'s first
    /// argument, cached on the entity so a consumer does not have to look the
    /// `Guid` up again.
    pub guid: Uuid,
    /// **How far below this entity's transform its feet are**, metres — the
    /// archetype's [`CrowdArchetype::feet_offset_m`], published so
    /// [`crate::pose::model_to_world`] can answer for a tier that carries no
    /// `CharacterMovement`.
    ///
    /// Without it a `Far` agent is drawn with its feet at its capsule's centre:
    /// **1.20 m** into the air on the humanoid archetype, measured by the NPC1b
    /// audit's own tripwire the step this wave gave the near tiers a controller.
    pub feet_offset_m: f64,
    /// **This agent is further than [`BLOCKED_LAG_M`] behind its own route
    /// clock** — its body has stopped making progress along the path while the
    /// clock has run on.
    ///
    /// Published rather than kept in [`CrowdStats`] alone because it has a
    /// *reader*: `inf_physics::d3::gameplay::step_crowd_doors` opens the door a
    /// blocked agent is standing against, and it needs to know **which** agent
    /// rather than how many. Reading it that way is also what keeps that pass
    /// cheap — a doorway scan per agent would be `O(agents x doorways)` over a
    /// city that plans 19 790 of them, and the blocked set is a handful.
    ///
    /// Always `false` on a tier with no controller: an agent the clock moves
    /// cannot fall behind the clock.
    pub blocked: bool,
    /// **What this agent is doing right now** (wave VEN1b) — its active leg's
    /// [`SlotArrival::posture`], once it has arrived.
    ///
    /// Published here for [`crate::pose::step_pose_evaluation`]'s benefit, on
    /// exactly [`feet_offset_m`](Self::feet_offset_m)'s terms: the pose step's
    /// query already reads this component, so a posture on it costs the pose
    /// pass **nothing** — no second lookup, no resource, no map. It is a
    /// re-publication of a pure function rather than sim state, which is why it
    /// is not folded into `crowd_state_bytes`: both hosts derive it from the
    /// same schedule and the same clock, and what a divergence would change is
    /// the *pose*, which the trace already carries joint by joint.
    ///
    /// [`Stand`](crate::components::SlotPosture::Stand) for every agent that
    /// is walking, for every unscheduled record, and for every level that
    /// predates the venues.
    pub posture: crate::components::SlotPosture,
    /// **A unit XZ direction this agent should face**, or `DVec3::ZERO` for one
    /// with no opinion (wave VEN1b) — its active leg's
    /// [`SlotArrival::face`].
    ///
    /// Published beside the posture because they are answered together and a
    /// seated body facing the wall it walked in past is not a seated body.
    pub face: DVec3,
    /// **Where in its posture's own loop this agent is**, seconds (wave VEN1b).
    ///
    /// Already wrapped into `[0, duration)` and already carrying this agent's
    /// own phase and tempo — see [`posture_time`], and see it in particular for
    /// why the wrap happens in `f64` *before* the cast: at a level's fourth
    /// hour `t_s` is 14 400, where an `f32` has about a millisecond of
    /// resolution left, and a 1.2-second loop sampled in `f32` at that scale
    /// steps rather than moves. (The same arithmetic VEN1a's `pulse_tick`
    /// carries, one system over.)
    ///
    /// `0.0` for [`SlotPosture::Stand`](crate::components::SlotPosture::Stand),
    /// which plays no clip.
    pub posture_t: f32,
}

/// What one [`step_crowd`] did — the instrument's read, and the gate's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CrowdStats {
    /// Records at each tier, indexed by [`CrowdTier::as_u8`].
    pub per_tier: [usize; 4],
    /// **Agents turned to face something this step** (wave VEN1b) — one per
    /// materialized agent whose active leg's arrival names a direction.
    ///
    /// Counted because `face_body` — the private door both branches of the
    /// ladder turn a body through — writes through two different doors
    /// depending on the tier and a write to the wrong one is *invisible*: a
    /// steered body would simply keep the yaw it walked in with. A gate can
    /// hold this against the number of agents it knows are sitting down.
    pub faced: u64,
    /// **Agents in a posture other than standing this step** (wave VEN1b) —
    /// the town's seated and dancing population, as one number.
    pub posed_apart: u64,
    /// Entities materialized this step (`Dormant` → anything else).
    pub spawned: u64,
    /// Entities dematerialized this step (anything else → `Dormant`).
    pub despawned: u64,
    /// Records whose tier changed this step — the number the transition arm
    /// asserts is non-zero over a drive.
    pub retiered: u64,
    /// **Poses folded into a cached digest this step** (NPC1a audit) — one per
    /// agent that left a posing tier and had an entry in the store.
    ///
    /// A counter and not a clock, because the property it exists to hold is a
    /// *shape*: the crowd phase's work must be a function of the CROWD. The
    /// first cut folded every entry in the pose store on every step to serve
    /// this, so the phase cost grew with the level's hero count and the wave's
    /// budget was minted against a number that was mostly other systems' poses
    /// (0.282 ms banded against 0.759 all-`Full`, at one thousand agents doing
    /// identical work). This reads 0 on a settled step whatever the level's
    /// character count, which is the machine-independent way to say so.
    pub digests_folded: u64,
    /// **Clocks re-phased onto a body this step** (NPC1c) — one per agent that
    /// left a tier with a controller, so this is a *transition* counter beside
    /// [`digests_folded`](Self::digests_folded) and for the same reason: both
    /// are the work a hand-off costs, and both must read 0 on a settled step.
    pub rephased: u64,
    /// **Agents steered this step**: on a tier with a controller, wanting to
    /// move, with somewhere to go. The falsifier for the near tiers' whole
    /// clause — a crowd that walked by teleporting would read 0 here.
    pub steered: u64,
    /// **Agents that have arrived** — on a [`RouteMode::Once`] route, within
    /// [`ARRIVE_M`] of its end. Counted every step they stand there, not once,
    /// so it is a census and not an event.
    pub arrived: u64,
    /// **Agents further than [`BLOCKED_LAG_M`] behind their own route clock** —
    /// a body against a shut door, or wedged. Reported rather than corrected:
    /// what to *do* about one is the door pass's job and NPC1d's, and a per-step
    /// correction would be the second authority over a transform this design
    /// exists without.
    pub blocked: u64,
    /// **Agents that started a new leg of their day this step** (NPC1d). Zero
    /// on a level with no schedules, and on most steps of one that has them:
    /// a four-leg day changes leg four times per agent per day.
    pub legs_changed: u64,
    /// The band's membership stamp (`0` = unbounded).
    pub band_stamp: u64,
}

impl CrowdStats {
    /// Records at `tier`.
    #[inline]
    pub fn at(&self, tier: CrowdTier) -> usize {
        self.per_tier[tier.as_u8() as usize]
    }

    /// Records in total.
    #[inline]
    pub fn total(&self) -> usize {
        self.per_tier.iter().sum()
    }

    /// A one-line summary for the diagnostics log and the instruments.
    pub fn summary(&self) -> String {
        format!(
            "crowd: {} agent(s) — {} full / {} near / {} far / {} dormant, \
             {} spawned / {} despawned / {} re-tiered, {} pose digest(s) folded, \
             {} re-phased, {} steered / {} arrived / {} blocked, band {:#018x}",
            self.total(),
            self.at(CrowdTier::Full),
            self.at(CrowdTier::Near),
            self.at(CrowdTier::Far),
            self.at(CrowdTier::Dormant),
            self.spawned,
            self.despawned,
            self.retiered,
            self.digests_folded,
            self.rephased,
            self.steered,
            self.arrived,
            self.blocked,
            self.band_stamp,
        )
    }
}

// ── the step ────────────────────────────────────────────────────────────────

/// **Install a population** on `world`, replacing any it already had.
///
/// Records arrive tier-less (`Dormant`, no entity); the first [`step_crowd`]
/// materializes the ones the band wants. That ordering is the point: a spawner
/// that decided tiers itself would be a second copy of the decision.
///
/// # The `Guid`s are the CALLER's to keep distinct
///
/// A record's key becomes its entity's `Guid` when it materializes, and
/// `EcsWorld::spawn_with_guid` does not refuse a key the world already uses — it
/// would overwrite the index entry, and the level's own entity would become
/// unreachable by `Guid` while still existing. Every caller in this tree draws
/// from a fixed namespace of its own for exactly that reason. A checked door
/// belongs on `spawn_with_guid` rather than here (it is the one place that could
/// answer for *every* spawner), and it is not this wave's to build; NPC1d, which
/// derives a population from a level's own buildings, is the wave that needs it.
///
/// # "Replacing" includes the BODIES (NPC1a audit)
///
/// A population is not only its records: a materialized agent is a real entity
/// carrying a skeletal mesh, a machine, a capsule and a [`CrowdAgent`]. Dropping
/// the resource alone left those standing with a tier frozen at whatever the
/// last step decided and **no record behind them** — so [`crowd_colliders`],
/// which reads records, and the [`CrowdAgent`] component, which the pose door
/// and [`crate::deform::ground_contacts`] read, would answer differently about
/// one entity. That is the same two-opinions shape as this wave's own deform
/// finding, so the old crowd goes through [`clear_crowd`] first.
pub fn set_population(world: &mut EcsWorld, records: BTreeMap<Uuid, CrowdRecord>) {
    clear_crowd(world);
    world.world_mut().insert_resource(CrowdPopulationRes {
        records,
        steps: 0,
        hand_installed: true,
    });
}

/// **Add agents to the population without disturbing the ones already in it**
/// (NPC1d) — the door a society grows a level's crowd through.
///
/// [`set_population`] REPLACES: it despawns every materialized body and restarts
/// the route clock at zero, which is right for an instrument installing a
/// measured crowd and wrong for a settlement whose blocks stream in over a
/// hundred steps. This inserts, keeps `steps` where it is, and leaves every
/// existing record's sim state alone.
///
/// # It refuses a `Guid` the world already holds
///
/// NPC1a's carried item 9: `EcsWorld::spawn_with_guid` does not refuse a key the
/// world already has — it overwrites the index entry and leaves the level's own
/// entity unreachable by `Guid` while it still exists. Every caller so far drew
/// from a namespace of its own and was trusted to; a society derives its agents'
/// ids from a level's own content, so it is the first caller that *could* meet
/// one. The check is here rather than on `spawn_with_guid` because that door
/// answers for every spawner in the engine and moving it is a decision of its
/// own; this closes the hazard for the one caller that has it. Returns how many
/// were refused, which is a number a gate can assert is zero.
pub fn add_agents(world: &mut EcsWorld, records: BTreeMap<Uuid, CrowdRecord>) -> usize {
    let mut refused = 0usize;
    let mut pop = world
        .world_mut()
        .remove_resource::<CrowdPopulationRes>()
        .unwrap_or_default();
    for (guid, rec) in records {
        if pop.records.contains_key(&guid) || world.entity_of(guid).is_some() {
            refused += 1;
            continue;
        }
        pop.records.insert(guid, rec);
    }
    world.world_mut().insert_resource(pop);
    refused
}

/// **Forget the crowd**: despawn every materialized agent and remove the
/// resource, so the world is byte-for-byte one that never had a population.
///
/// The editor calls this at both ends of a Simulate session for the reason
/// [`crate::pose::clear_poses`] documents — a `SceneDoc` snapshot carries
/// entities and components and `EcsWorld::clear` despawns entities, and neither
/// touches a resource, so without this a stopped session's crowd would keep
/// standing in the author's document.
pub fn clear_crowd(world: &mut EcsWorld) {
    let materialized: Vec<Uuid> = world
        .world()
        .get_resource::<CrowdPopulationRes>()
        .map(|p| {
            p.records
                .iter()
                .filter(|(_, r)| r.tier.materialized())
                .map(|(g, _)| *g)
                .collect()
        })
        .unwrap_or_default();
    for guid in materialized {
        if let Some(e) = world.entity_of(guid) {
            world.despawn(e);
        }
    }
    world.world_mut().remove_resource::<CrowdPopulationRes>();
}

/// **THE FIXED-STEP CROWD SLOT**: decide every agent's tier, materialize or
/// dematerialize it, and put it where its route says it is.
///
/// ONE function, called from both hosts' fixed steps — the strongest form the
/// MIRROR rule takes, and the same shape [`crate::deform::step_deformation`],
/// [`crate::sky::advance_weather`] and [`crate::pose::step_pose_evaluation`]
/// use. It takes a world and a `dt` and **no camera, no host state and no
/// registries**: everything it reads is in the world.
///
/// The sequence, all of it a pure function of sim state:
///
/// 1. read the band off the world's [`StreamingSource`] entities;
/// 2. walk the records in `Guid` order; for each, the tier of where it *is*
///    (its live transform if it has an entity, its remembered `last` if not);
/// 3. a record leaving [`Full`]/[`Near`] folds its published pose into
///    `pose_digest` — **before** the pose store is rebuilt this step, which is
///    why this phase runs early;
/// 4. `Dormant` → despawn; anything else → spawn if absent;
/// 5. write the route position onto the transform, and the tier onto the
///    [`CrowdAgent`].
///
/// # Where it runs, and why there
///
/// After cell + terrain streaming and the sky, **before** the physics sync, the
/// character step and the animation: the bridge has to see this step's bodies,
/// and [`crate::pose::step_pose_evaluation`] has to see this step's tiers. Its
/// own phase (`inf_player::step_profile::STEP_PHASES` 24 to 25) rather than a
/// corner of an existing one, because a step that cannot say where its
/// milliseconds went is the defect wave I4b existed to remove.
///
/// # The hero is untouched
///
/// A character carrying no [`CrowdAgent`] — every hero, every authored NPC,
/// every fixture in this tree — is not in the population, is never walked here,
/// and gets exactly the pipeline it got before NPC1a. The tier system is
/// **opt-in by record**, which is what makes "zero cost when absent" a structural
/// claim rather than a benchmark.
///
/// [`Full`]: CrowdTier::Full
/// [`Near`]: CrowdTier::Near
/// [`StreamingSource`]: crate::components::StreamingSource
pub fn step_crowd(world: &mut EcsWorld, dt: f64) -> CrowdStats {
    step_crowd_banded(world, dt, DEFAULT_CROWD_RADII)
}

/// **What one agent's tier and place would be** — the step's PURE half, split
/// out so it can be measured and, if it ever pays, parallelized.
///
/// # Why this is a separate function
///
/// The step below is three things in a trench coat: a world read (where is this
/// agent), a decision (`band.tier` + `route(clock)`), and a world write
/// (spawn/despawn/transform). Only the middle one is a pure function, and only a
/// pure function can go through `inf_core::job`'s deterministic in-order map —
/// the ECS mutation cannot, it needs `&mut World`. (Named in prose rather than
/// linked: `inf-ecs` does not depend on `inf-core`, and this wave's measurement
/// says it should not start.)
///
/// So the decision lives here, the step calls it, and the sweep instrument times
/// **this exact function** serially and in parallel at N ∈ {1, 10, 100, 1 000}.
/// A benchmark of a private copy would be a benchmark of something the engine
/// does not run, which is this repository's own "a gate must aim at the thing it
/// names".
///
/// `here` is where the agent is *now* — its live transform if it has an entity,
/// its remembered [`CrowdRecord::last`] if it does not.
#[inline]
pub fn plan_agent(
    band: &CrowdBand,
    guid: Uuid,
    rec: &CrowdRecord,
    here: DVec3,
    clock: CrowdClock,
    leg: ActiveLeg,
) -> AgentPlan {
    // **The position law does not vary with the tier, `Dormant` included**
    // (NPC1a audit). It used to: a dematerialized record froze at `last`, the
    // tier was then decided from that frozen point, and a walking agent whose
    // route carried it home could never be re-admitted — it was judged for ever
    // at the metre where it went out of range, while the next materialization
    // would have placed it at `route(now)` somewhere else entirely. Frozen for
    // the decision and live for the placement is two authorities over one thing,
    // which is what this module exists to avoid.
    //
    // A route is a pure function of the clock and costs a handful of flops, so
    // keeping it live while an agent has no entity costs nothing and is the only
    // reading under which `Dormant` is a *cost* tier rather than a one-way door.
    // The day NPC1d gives an off-screen agent a schedule is the day this line
    // reads the schedule instead — for every tier at once.
    //
    // NPC1c adds the second half of the same sentence: for a tier that carries a
    // controller the clock decides the agent's *heading*, and the body decides
    // where it is. The plan still answers `route(clock)` for both, because the
    // near tiers' pursuit is a lookahead ALONG this path and the demotion
    // re-phase is what keeps the two answers from parting company.
    //
    // **The path is a WALKING SURFACE and a transform is a capsule centre**, so
    // the agent's own `feet_offset_m` is added here — once, in the one place the
    // place is decided. A nav path is the ground a body walks on (a road's
    // surveyed spine, a street's centreline, a room's floor), and an entity's
    // transform is its capsule's middle: P29.6's ruling, met by a tier that has
    // no `CharacterMovement` for `movement::feet_offset_m` to read. Keeping the
    // lift out of the path is also what lets a tall agent and a short one share
    // one `Arc`.
    //
    // **And the leg is handed IN** (island wave NPC1e). This used to resolve it
    // three times inside one call — once for `progress_at` and twice more down
    // `position_at → feet_at → {path_at, progress_at}` — each a `mix64` and a
    // `rem_euclid` a leg, for an answer that cannot change inside a step. See
    // [`ActiveLeg`] for the measurement that made it worth threading.
    let progress = rec.progress_on(leg, guid, clock);
    AgentPlan {
        tier: band.tier(here),
        at: rec.position_on(leg, guid, clock),
        progress,
    }
}

/// One agent's decided tier and place — [`plan_agent`]'s answer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AgentPlan {
    /// What it costs this step.
    pub tier: CrowdTier,
    /// Where the route's clock says it is this step.
    pub at: DVec3,
    /// How far along the route that is, which way it is going, and whether it
    /// has run out.
    pub progress: RouteProgress,
}

/// [`step_crowd`] with explicit radii — the seam the sweep instrument drives to
/// price a tier ladder, and the one a level's own crowd settings will use.
pub fn step_crowd_banded(world: &mut EcsWorld, dt: f64, radii: (f64, f64, f64)) -> CrowdStats {
    // Absent costs nothing: one `contains_resource` on every level with no crowd.
    if !world.world().contains_resource::<CrowdPopulationRes>() {
        return CrowdStats::default();
    }
    let band = CrowdBand::from_world(world, radii);
    let mut stats = CrowdStats {
        band_stamp: band.stamp(),
        ..CrowdStats::default()
    };
    // Lifted out of the world, because the materialization below needs
    // `&mut EcsWorld` and a borrow of the resource would outlive it — the shape
    // `step_pose_evaluation` lifts its goals and blenders with.
    let mut pop = world
        .world_mut()
        .remove_resource::<CrowdPopulationRes>()
        .unwrap_or_default();
    // **The clock the crowd is on** (NPC1d): sim seconds AND the level's own
    // local hour, read once for the whole step. The sky phase advanced it four
    // phases ago (`STEP_PHASES` index 1 against the crowd's 4), which is what
    // makes "the town leaves for work at eight" a statement about this step
    // rather than about the last one.
    let clock = CrowdClock::from_world(world, pop.steps as f64 * dt);

    for (guid, rec) in pop.records.iter_mut() {
        let guid = *guid;
        let entity = world.entity_of(guid);
        let here = match entity.and_then(|e| world.world().get::<Transform>(e)) {
            Some(t) => t.translation.to_dvec3(),
            None => rec.last,
        };
        // 0. **A new leg starts clean** (NPC1d). `rephase_m` is the metre a body
        //    fell behind ITS OWN leg's clock; carrying it into the next leg
        //    would start a commute home already two metres late on a different
        //    path. The leg index is sim state and is folded into the trace for
        //    exactly that reason.
        //
        //    **Resolved ONCE, here, and handed down** (island wave NPC1e): the
        //    leg-change check, the plan and the pursuit all asked for it
        //    separately, which is five `mix64`-plus-`rem_euclid` resolutions an
        //    agent for an answer that is constant across the step. See
        //    [`ActiveLeg`].
        let leg = rec.leg_at(guid, clock);
        if let Some((i, _)) = leg {
            let i = i as u8;
            if i != rec.leg {
                rec.leg = i;
                rec.rephase_m = 0.0;
                stats.legs_changed += 1;
            }
        }
        let plan = plan_agent(&band, guid, rec, here, clock, leg);
        let tier = plan.tier;
        let was = rec.tier;
        if tier != was {
            stats.retiered += 1;
        }
        // 3. The cached digest, taken on the way DOWN out of a posing tier —
        //    from the store the PREVIOUS step published, and **only for the
        //    agent that is demoting** (NPC1a audit).
        //
        //    The first cut folded every entry in the pose store into a map up
        //    front, which made this phase `O(posed characters × joints)` a step
        //    to serve a demotion that happens to one agent every few hundred.
        //    The wave's own sweep table is what says so: at N = 1 000 the crowd
        //    phase read **0.282 ms banded and 0.759 ms all-`Full`** — the same
        //    thousand agents doing the same work, differing only in how many
        //    characters were in the store — so most of a number minted as "what
        //    a thousand agents cost" was a fold over other systems' poses, and
        //    it grew with the level's hero count rather than with the crowd.
        if was.poses() && !tier.poses() {
            if let Some(d) = published_pose_digest(world, guid) {
                rec.pose_digest = d;
                stats.digests_folded += 1;
            }
        }
        rec.tier = tier;
        stats.per_tier[tier.as_u8() as usize] += 1;

        // 3a. **THE HAND-OFF DOWN** (NPC1c). An agent leaving a tier that had a
        //     controller hands the clock back the metre its BODY reached, not
        //     the metre the clock had run on to. Without it, an agent that had
        //     been blocked by a wall for ten seconds teleports 16 m the step it
        //     drops to `Far`; with it, the transition is invisible in the
        //     transform, which is the property NPC1a's whole gate rests on.
        //
        //     `project` is `O(path points)` and runs on a *transition* only —
        //     one agent every few hundred steps — which is the same budget the
        //     pose digest above is taken on.
        let mut at = plan.at;
        if was.steers() && !tier.steers() {
            let feet = here - DVec3::new(0.0, rec.archetype.feet_offset_m(), 0.0);
            let s_body = rec.path_on(leg).project(feet).s_m;
            let delta = rec.rephase_delta_on(leg, guid, clock, s_body);
            if delta != 0.0 {
                rec.rephase_m += delta;
                stats.rephased += 1;
                at = rec.position_on(leg, guid, clock);
            }
        }

        if !tier.materialized() {
            rec.last = at;
            if let Some(e) = entity {
                world.despawn(e);
                stats.despawned += 1;
            }
            continue;
        }
        let entity = match entity {
            Some(e) => e,
            None => {
                stats.spawned += 1;
                // **THE HAND-OFF UP**: a materializing agent is placed at
                // `route(clock)`, which is exactly where the tier below it was
                // already drawn — so a promotion moves nothing. (NPC1a's
                // promotion-teleport arm class; NPC1c keeps the property while
                // adding a controller under it.)
                materialize(world, guid, rec, at)
            }
        };
        // 4a. **THE COMPONENTS ARE THE TIER'S** (NPC1c). A `Far` agent carries
        //     no body, no collider, no controller and no movement model — not
        //     "a body the bridge declines to mirror", which is what NPC1a
        //     shipped and what made every one of a thousand agents a terrain
        //     observer at every tier (`terrain_stream::observes_terrain` picks
        //     its subjects by COMPONENT). One authority for "is this thing
        //     physically here", which is this module's own founding law.
        set_tier_components(world, entity, tier, &rec.archetype);

        // 5. Where the route says it is — for the tiers that have no controller.
        //    A tier that HAS one is moved by `step_character_movement` five
        //    phases later, and writing a transform here as well would be the two
        //    authorities NPC1a refused a controller to avoid. What the crowd
        //    writes for those is an INTENT, and the body follows it.
        if tier.steers() {
            rec.last = here;
            steer_agent(
                world,
                entity,
                SteerSubject { rec, here, leg },
                &plan,
                &mut stats,
            );
        } else {
            rec.last = at;
            if let Some(mut t) = world.world_mut().get_mut::<Transform>(entity) {
                t.translation = Vec3d::new(at.x, at.y, at.z);
            }
        }
        // **What the agent is DOING, and which way it is doing it** (VEN1b).
        // Resolved off the leg the step already named, so this costs one match
        // on a value in hand; `face_body` is the one door both branches of the
        // ladder turn a body through.
        let arrival = rec.arrival_on(leg);
        if arrival.face != DVec3::ZERO {
            face_body(world, entity, tier.steers(), arrival.face);
            stats.faced += 1;
        }
        let posture_t = posture_time(guid, clock, arrival.posture);
        if let Some(mut a) = world.world_mut().get_mut::<CrowdAgent>(entity) {
            a.tier = tier;
            a.posture = arrival.posture;
            a.face = arrival.face;
            a.posture_t = posture_t;
            if !tier.steers() {
                // An agent the clock moves cannot fall behind the clock, and a
                // stale `true` here would send the door pass looking for a body
                // that is no longer standing against anything.
                a.blocked = false;
            }
        }
        if arrival.posture != crate::components::SlotPosture::Stand {
            stats.posed_apart += 1;
        }
    }

    pop.steps += 1;
    world.world_mut().insert_resource(pop);
    stats
}

/// How far ahead along its own path a steered agent aims, metres.
///
/// One stride. Pure pursuit with too short a lookahead oscillates about the
/// spine (the target is behind the character's own turn rate); with too long a
/// one it cuts corners, which on a street grid means walking through the
/// building on the inside of the turn. 1.5 m is a little under the 1.65 m/s the
/// movement model's `walk_speed_mps` covers in a second, so at a walk the agent
/// is aiming about where it will be next second.
pub const PURSUIT_LOOKAHEAD_M: f64 = 1.5;

/// How far the pursuit lookahead may grow looking for a horizontal lead,
/// metres.
///
/// Six metres is four strides, which is longer than any flight of stairs this
/// engine generates and shorter than a city block, so a follower that has run
/// out of lead has really run out of path rather than merely met a steep bit.
pub const PURSUIT_LOOKAHEAD_MAX_M: f64 = 6.0;

/// The smallest horizontal separation a pursuit target may have from the agent's
/// own feet, metres.
///
/// A quarter of a metre is under one capsule radius, so a target this close is
/// somewhere the body is already standing — which on a stair means *directly
/// above it*.
pub const MIN_LEAD_M: f64 = 0.25;

/// How close a steered agent has to get to the end of a [`RouteMode::Once`]
/// route to have arrived, metres.
///
/// Half a metre is one capsule radius plus the mover's own skin, which is as
/// close as a body with a 0.3 m radius can get to a point without standing on
/// it.
pub const ARRIVE_M: f64 = 0.5;

/// How far a steered agent may fall behind its own route clock before it is
/// counted **blocked**, metres.
///
/// Two metres is a little over one second of walking, so a body slowed by a
/// slope or squeezing past another agent is not blocked and a body standing
/// against a shut door is. Reported in [`CrowdStats::blocked`] rather than
/// acted on here: what to *do* about a blocked agent is the door pass's job
/// (`inf_physics::d3::gameplay`, which can open the door in its way) and,
/// beyond that, NPC1d's.
pub const BLOCKED_LAG_M: f64 = 2.0;

/// Salts the per-agent posture phase — where in a dance loop an agent starts.
pub const SALT_POSTURE_PHASE: u64 = 0x504f_5354_5552_4501;

/// Salts the per-agent posture tempo.
pub const SALT_POSTURE_RATE: u64 = 0x504f_5354_5552_4502;

/// **How far either side of the authored tempo an agent dances**, as a
/// fraction.
///
/// A tenth. `CrowdRecord::speed_of` spreads a walk by 0.15 for the same reason
/// and the same amount of it is right here: twenty agents on one dance floor
/// playing one 1.2-second loop at one rate are twenty copies of one body, and
/// the *phase* spread alone does not fix that — two agents a beat apart are
/// still in lockstep, for ever.
pub const POSTURE_RATE_SPREAD: f64 = 0.1;

/// **Where in its own posture loop an agent is**, seconds — the value
/// [`CrowdAgent::posture_t`] carries.
///
/// A pure function of `(guid, clock, posture)` with the two per-agent draws
/// taken at `tick = 0`, on [`CrowdRecord::speed_of`]'s own terms: derived,
/// never stored, and therefore incapable of drifting from the thing it is
/// derived from.
///
/// **Wrapped in `f64` and cast once.** `t_s` is a level's whole run in seconds,
/// and the clip is a second and a bit; doing the modulo after the cast would
/// quantize a dance floor's phase to whatever an `f32` has left at four hours,
/// which is about a millisecond. This is VEN1a's `pulse_tick` argument in a
/// different system.
pub fn posture_time(guid: Uuid, clock: CrowdClock, posture: crate::components::SlotPosture) -> f32 {
    let Some(p) = posture.anim() else {
        return 0.0;
    };
    let d = f64::from(p.clip().duration_s);
    if !(d.is_finite() && d > 0.0 && clock.t_s.is_finite()) {
        return 0.0;
    }
    let rate = 1.0 + POSTURE_RATE_SPREAD * (2.0 * agent_unit(guid, 0, SALT_POSTURE_RATE) - 1.0);
    let offset = agent_unit(guid, 0, SALT_POSTURE_PHASE) * d;
    ((clock.t_s * rate + offset).rem_euclid(d)) as f32
}

/// **Turn a body to face `face`** (wave VEN1b) — the one door, because the
/// ladder has two ways to own a rotation and writing to the wrong one is
/// invisible.
///
/// A **steered** agent's rotation belongs to the movement model:
/// `step_character_movement` writes `runtime.body_yaw_deg` back onto
/// `Transform::rotation.y` every step, five phases after this one, so a
/// transform written here would be overwritten before anything drew it. The
/// model is told instead — both the current yaw and the target, or the model's
/// own turn-rate would spend the next half second walking it back.
///
/// Every other tier has no model, so its transform IS the answer.
///
/// The yaw comes from [`inf_math::patan2_64`] and never `f64::atan2`: it lands
/// on a `Transform` and therefore in the replay trace (P14), and it is the same
/// spelling `traffic::yaw_deg_of` uses for the same reason.
fn face_body(world: &mut EcsWorld, entity: Entity, steers: bool, face: DVec3) {
    let yaw = inf_math::patan2_64(face.x, face.z).to_degrees();
    if !yaw.is_finite() {
        return;
    }
    if steers {
        if let Some(mut cm) = world.world_mut().get_mut::<CharacterMovement>(entity) {
            cm.runtime.body_yaw_deg = yaw;
            cm.runtime.target_yaw_deg = yaw;
            return;
        }
    }
    if let Some(mut t) = world.world_mut().get_mut::<Transform>(entity) {
        t.rotation.y = yaw;
    }
}

/// Build the entity a record describes, at `at`.
///
/// The component set is fixed and small: a skeletal mesh, a machine, the
/// [`CrowdAgent`] verdict — and then whatever the tier says
/// ([`set_tier_components`]), which is where the body, the capsule, the
/// controller and the movement model arrive.
///
/// Placed at `route(clock)` rather than at the record's remembered `last`,
/// because that is exactly where the tier below was already drawing it: a
/// promotion must move nothing.
fn materialize(world: &mut EcsWorld, guid: Uuid, rec: &CrowdRecord, at: DVec3) -> Entity {
    let e = world.spawn_with_guid(guid, "Crowd NPC", None);
    let a = rec.archetype;
    world.world_mut().entity_mut(e).insert((
        Transform {
            translation: Vec3d::new(at.x, at.y, at.z),
            ..Transform::IDENTITY
        },
        SkeletalMesh {
            mesh: a.mesh,
            skeleton: a.skeleton,
        },
        AnimStateMachine {
            sm: a.sm,
            ..AnimStateMachine::default()
        },
        CrowdAgent {
            tier: rec.tier,
            guid,
            feet_offset_m: a.feet_offset_m(),
            blocked: false,
            posture: crate::components::SlotPosture::Stand,
            face: DVec3::ZERO,
            posture_t: 0.0,
        },
    ));
    set_tier_components(world, e, rec.tier, &a);
    e
}

/// **The physical half of the tier, as COMPONENTS** (wave NPC1c).
///
/// # Why this is not the bridge's job
///
/// NPC1a gave every materialized agent a [`RigidBody3D`], a [`Collider3D`] and a
/// [`CharacterController3D`] at every tier, and had the 3D bridge decline to
/// mirror the ones the tier said were bodiless. That is one system holding an
/// opinion about another system's data, and it cost exactly what this module's
/// own founding law predicts: `terrain_stream::observes_terrain` picks its
/// subjects **by component**, so a thousand NPCs were a thousand terrain
/// observers — including the 712 `Far` ones that had no rapier body and no door
/// by which to query a height — and each pulled a `SIM_MARGIN_TILES`
/// neighbourhood of level-0 pages into a want set that is never clamped.
/// Measured by the NPC1b audit at **+1.16 ms of "crowd" cost that was not the
/// crowd**.
///
/// So the tier owns the components. A thing with no body has no body: nothing
/// downstream has to be told, because there is nothing to see.
///
/// # What each tier carries
///
/// | tier | body + capsule + controller | `CharacterMovement` |
/// |---|---|---|
/// | [`Full`](CrowdTier::Full) | yes | **yes** |
/// | [`Near`](CrowdTier::Near) | yes | **no** |
/// | [`Far`](CrowdTier::Far) | **no** | **no** |
///
/// *(The NPC1c audit's correction. This table read "`Full` / `Near` | yes |
/// yes", which is the ladder this wave **started** with: the brief put a
/// controller on both near tiers, the island priced 291 of them at 92.756 ms,
/// and [`steers`](CrowdTier::steers) moved to `Full` alone — while this table
/// and the two paragraphs under it stayed in the first draft. A doc that
/// describes the component ladder wrongly, in the function that **is** the
/// component ladder, is the worst place in the crate for it to be wrong.)*
///
/// Two consequences, as they actually stand:
///
/// * **the `Near` rung still saves nothing that can be falsified** — the NPC1b
///   audit's carried item 10, and this wave's own carried item 7.
///   `movement_targets` is the set carrying [`CharacterMovement`] and
///   `step_hand_ik` composes its requests over that set, so a `Near` agent —
///   which has none — can no more hold a hand request than it could before this
///   wave, and severing [`hand_ik`](CrowdTier::hand_ik) still changes nothing
///   any arm can see. The rung becomes falsifiable the day a near agent carries
///   a controller, and 291 of those is 92.756 ms;
/// * **the 1.20 m gap is closed by the POSE door, not by this ladder.**
///   `pose::model_to_world` was gated on `CharacterMovement`, so an NPC without
///   one was drawn with its feet at its capsule's centre; it now falls back to
///   [`CrowdAgent::feet_offset_m`] through `pose::character_drop`, which is what
///   covers `Near` and `Far` — the two tiers that deliberately carry no
///   movement model and still have feet.
fn set_tier_components(world: &mut EcsWorld, entity: Entity, tier: CrowdTier, a: &CrowdArchetype) {
    let has_body = world.world().get::<RigidBody3D>(entity).is_some();
    let steers = world.world().get::<CharacterMovement>(entity).is_some();
    if tier.has_body() == has_body && tier.steers() == steers {
        return;
    }
    let mut em = world.world_mut().entity_mut(entity);
    if tier.has_body() != has_body {
        if tier.has_body() {
            em.insert((
                RigidBody3D {
                    kind: BodyKind3D::Kinematic,
                    fixed_rotation: true,
                    ..RigidBody3D::default()
                },
                Collider3D {
                    shape_kind: ColliderShape3DKind::Capsule,
                    half_extents: Vec3d::new(a.radius_m, a.half_height_m, a.radius_m),
                    radius: a.radius_m,
                    ..Collider3D::default()
                },
            ));
        } else {
            em.remove::<RigidBody3D>();
            em.remove::<Collider3D>();
        }
    }
    if tier.steers() != steers {
        if tier.steers() {
            em.insert((CharacterController3D::default(), crowd_movement(a)));
        } else {
            em.remove::<CharacterController3D>();
            em.remove::<CharacterMovement>();
        }
    }
}

/// **Build one crowd-archetype body that is NOT in the population** (wave
/// VEH2b) — the door a traffic car's driver comes through.
///
/// A traffic driver is a person made of a level's own archetype, wearing the
/// same capsule, the same mesh and the same state machine as every resident,
/// and it must be: a carjack pulls one out onto the pavement and it walks away
/// as an ordinary NPC. What it is *not* is a [`CrowdRecord`] — it has no route,
/// no schedule and no tier of its own, because the thing that decides where it
/// is is the car it is sitting in.
///
/// So it goes through this door rather than through [`materialize`]: the same
/// components, minus the [`CrowdAgent`] verdict that would make
/// [`step_crowd`] and [`crate::pose::step_pose_evaluation`] answer for an agent
/// the population has never heard of. It is always a **steered** body
/// ([`CrowdTier::Full`]'s components), because a driver only exists at a car's
/// own `Full` tier and a body the movement step does not walk cannot sit in a
/// seat.
pub fn spawn_body(world: &mut EcsWorld, guid: Uuid, a: &CrowdArchetype, at: DVec3) -> Entity {
    if let Some(e) = world.entity_of(guid) {
        return e;
    }
    let e = world.spawn_with_guid(guid, "Driver", None);
    world.world_mut().entity_mut(e).insert((
        Transform {
            translation: Vec3d::new(at.x, at.y, at.z),
            ..Transform::IDENTITY
        },
        SkeletalMesh {
            mesh: a.mesh,
            skeleton: a.skeleton,
        },
        AnimStateMachine {
            sm: a.sm,
            ..AnimStateMachine::default()
        },
        // The verdict goes on even though the population has never heard of
        // this body, because `step_pose_evaluation` picks its subjects by this
        // component and a driver with no tier is a driver with no pose. `Full`
        // is the honest answer and not a convenience: a driver exists only
        // while its car is Full, which is 64 m.
        CrowdAgent {
            tier: CrowdTier::Full,
            guid,
            feet_offset_m: a.feet_offset_m(),
            blocked: false,
            posture: crate::components::SlotPosture::Stand,
            face: DVec3::ZERO,
            posture_t: 0.0,
        },
    ));
    set_tier_components(world, e, CrowdTier::Full, a);
    e
}

/// **How many fixed steps the crowd has run** — the `t_s` half of a
/// [`CrowdClock`], for a caller that has to phase a route onto NOW.
///
/// `0` for a level with no population, which is the value a fresh one starts
/// at, so a caller cannot tell the two apart and does not need to.
pub fn population_steps(world: &EcsWorld) -> u64 {
    world
        .world()
        .get_resource::<CrowdPopulationRes>()
        .map(|p| p.steps)
        .unwrap_or(0)
}

/// **Give an existing body a route** (wave VEH2b) — the door a carjacked driver
/// walks out of.
///
/// [`add_agents`] refuses a `Guid` the world already holds, and it is right to:
/// every caller before this one drew from a namespace of its own, and
/// `spawn_with_guid` would leave a level's own entity unreachable. This caller
/// is different in exactly one way — **the body is one this module built**,
/// through [`spawn_body`], and what it needs is not a body but a *record*: a
/// person who was driving a car a moment ago and is now standing in the road
/// with somewhere to be.
///
/// Returns `false` if the population already holds the guid, which is the
/// refusal that matters: adopting twice would replace a walking agent's route
/// with a fresh one and restart its day.
pub fn adopt(world: &mut EcsWorld, guid: Uuid, rec: CrowdRecord) -> bool {
    let mut pop = world
        .world_mut()
        .remove_resource::<CrowdPopulationRes>()
        .unwrap_or_default();
    let fresh = !pop.records.contains_key(&guid);
    if fresh {
        pop.records.insert(guid, rec);
    }
    world.world_mut().insert_resource(pop);
    fresh
}

/// **How fast somebody who wants to be somewhere else walks**, m/s.
///
/// `CharacterMovement::run_speed_mps`'s own default. A person who has just been
/// dragged out of their car — or heard a shot — is not strolling, and they are
/// not sprinting either: the crowd's `Gait::Walk` is what the body will actually
/// manage through `move_and_slide`, and the route speed is what the clock's tiers
/// use.
///
/// Hoisted out of `inf_physics::d3::carjack` at wave WPN1, when the flee gained
/// its second caller.
pub const FLEE_MPS: f64 = 3.75;

/// **Everybody who is running from something** (wave WPN1) — the latch that
/// stops [`flee_from`] restarting the same route every step.
///
/// [`crate::weapon::Downed`]'s argument verbatim: without it a crowd standing in
/// a firefight is re-routed on every step the gunfire continues, and a route that
/// restarts sixty times a second never gets anywhere — the clock's progress is
/// always about zero, so the body never leaves and the panic reads as a bug in
/// the steering.
///
/// # A resource and not a component, and the reason is the TIER
///
/// `Downed` is a component because a body that can be shot is a body that
/// exists. A crowd agent is not: a [`Dormant`](CrowdTier::Dormant) one has **no
/// entity at all** and its record is still in the population, still steps, still
/// has a route — and it is exactly the agent a shot at the far edge of the panic
/// radius reaches. A component latch would have been silently absent on every one
/// of them, which is the tier-dependent-state trap this module's own
/// `crowd_state_bytes` doc names.
///
/// Not a field on [`CrowdRecord`] either, because [`AGENT_TRACE_BYTES`] is
/// pinned and quoted — the crowd's whole trace-reshape claim is a ratio with that
/// number as its numerator. What the trace *does* see is the effect: the
/// re-phase, the position and the tier all move when an agent flees, so two hosts
/// that disagreed about who panicked part company there on the next step.
///
/// Derived, never saved and no schema moves — [`crate::item::ItemDefs`]' shape.
///
/// **A person panics ONCE.** The latch is never cleared, so somebody who has run
/// forty metres and stopped is not frightened again by a later shot. That is the
/// carjack's own bound (*"they arrive, and then they stand; they do not resume
/// their day"*) rather than a design, and it is on this wave's carried list.
#[derive(bevy_ecs::prelude::Resource, Default, Debug, Clone, PartialEq, Eq)]
pub struct PanickedRes {
    /// Who is running, in `Guid` order.
    pub fleeing: BTreeSet<Uuid>,
}

/// **Whether the crowd knows about this guid at all** — whether it has a record.
///
/// The question a caller asks *before* [`flee_from`], so that "give this person
/// somewhere to be" is `O(1)` for everybody the crowd does not own: the hero, a
/// scripted actor, a shopkeeper the level authored by hand. `flee_from` refuses
/// them anyway; this is what lets a caller not pay for the refusal.
pub fn is_in_population(world: &EcsWorld, guid: Uuid) -> bool {
    world
        .world()
        .get_resource::<CrowdPopulationRes>()
        .is_some_and(|p| p.records.contains_key(&guid))
}

/// Whether this agent is already running from something.
pub fn is_panicked(world: &EcsWorld, guid: Uuid) -> bool {
    world
        .world()
        .get_resource::<PanickedRes>()
        .is_some_and(|p| p.fleeing.contains(&guid))
}

/// **Forget who was running** — [`clear_crowd`]'s twin, for its reason: an editor
/// Simulate session must leave nothing behind in the author's document.
pub fn clear_panic(world: &mut EcsWorld) {
    world.world_mut().remove_resource::<PanickedRes>();
}

/// **THE ONE FLEE DOOR** (wave WPN1) — give somebody a reason to be somewhere
/// else, and the route to get there.
///
/// # It was `carjack::flee`, and hoisting it is the point
///
/// Wave VEH2b wrote this to walk a pulled-out driver away from the car. Wave
/// WPN1 needs the identical behaviour for a crowd that has heard a gunshot, and
/// a second copy would have been a second answer to *"what does a frightened
/// person in this engine do"* — including a second copy of the re-phase below,
/// which is the half that is easy to leave out and impossible to see.
///
/// `away_from` is the place being run from; `distance_m` is how far.
///
/// # THE RE-PHASE, and why it is not optional
///
/// [`CrowdRoute::progress_at`] is `speed × t_s + phase`, and `t_s` is the
/// crowd's own elapsed **session** time — so a fresh forty-metre
/// [`RouteMode::Once`] route handed to a population that has been running for a
/// minute reads as **finished before the agent has taken a step**. Two things go
/// wrong when it does: the agent is permanently `blocked` (which the door pass
/// reads, so it stands at the nearest door turning the handle for ever), and the
/// moment it drops off the steered tier its transform is written at the route's
/// END — forty metres away, instantly.
///
/// [`CrowdRecord::rephase_m`] exists for exactly this, so it is set once, here,
/// to the negative of whatever the clock has already run up plus the agent's own
/// derived phase.
///
/// # A scheduled agent's day is RELEASED, not paused
///
/// An agent with a [`CrowdSchedule`] is standing at a slot it was planned into —
/// a bar stool, a desk, a bed — and its [`SlotArrival`] is what holds it in that
/// posture. Fleeing **clears the schedule**, which releases that claim: the body
/// stops being at its venue and becomes a person walking. It does not resume its
/// day afterwards, exactly as a carjacked driver does not, and that is on this
/// wave's carried list rather than hidden here.
///
/// Answers `false` for a guid that is neither a record nor a body — there is
/// nobody to frighten — and for one that is already running ([`PanickedRes`]).
///
/// **A `Dormant` agent has no entity and is still a legitimate subject**: its
/// record is in the population, it still steps, and it is exactly the agent a
/// shot at the far edge of a panic radius reaches. So the body is optional here
/// and the record is not.
///
/// # AN OFFICER UNDER FIRE DOES NOT ROUT (wave EMS2)
///
/// The third refusal, and the one that is a **rule about the world** rather than
/// a fact about the argument: somebody
/// [`crate::dispatch::is_responder`] answers `true` for is on duty at
/// something, and this door refuses them.
///
/// It is here — at the one flee door — rather than inside the crowd panic,
/// because every caller needs it: the panic pass frightens a whole street, the
/// carjack frightens one driver, and an officer securing a crime scene while a
/// second shot goes off is reached by the first. Putting it in one caller would
/// have left the others, and a *second* copy of the test is the thing this
/// module's own hoist of `carjack::flee` existed to remove.
///
/// The consequence is stated rather than hidden. Fleeing **clears the schedule**
/// and [`PanickedRes`] is never released, so a responder that routed once would
/// have dropped its post *permanently* — for the rest of the session, on both
/// hosts, with its incident still open and no unit on the way. That is the trap
/// [`PanickedRes`]' own doc names, met by a wave that put people at gunfights on
/// purpose.
///
/// `inf_physics::d3::gameplay::PanicReport::exempt` counts the times this arm
/// fired, so a gate can tell *the officers held* from *no officer was in the
/// radius*.
pub fn flee_from(
    world: &mut EcsWorld,
    guid: Uuid,
    from: DVec3,
    away_from: DVec3,
    dt: f64,
    distance_m: f64,
) -> bool {
    let known = world
        .world()
        .get_resource::<CrowdPopulationRes>()
        .is_some_and(|p| p.records.contains_key(&guid));
    if (!known && world.entity_of(guid).is_none()) || is_panicked(world, guid) {
        return false;
    }
    if crate::dispatch::is_responder(world, guid) {
        return false;
    }
    let away = from - away_from;
    let len = (away.x * away.x + away.z * away.z).sqrt();
    let dir = if len > 1.0e-6 {
        DVec3::new(away.x / len, 0.0, away.z / len)
    } else {
        // Standing exactly on the thing being run from has no direction; `+Z` is
        // the engine's own forward and is a value rather than a NaN.
        DVec3::Z
    };
    let distance = if distance_m.is_finite() {
        distance_m.clamp(1.0, 1000.0)
    } else {
        FLEE_M
    };
    let route = CrowdRoute::along(
        inf_nav::NavPath::new([from, from + dir * distance]),
        FLEE_MPS,
        RouteMode::Once,
    );
    let t_s = population_steps(world) as f64 * dt;
    let mut pop = world
        .world_mut()
        .remove_resource::<CrowdPopulationRes>()
        .unwrap_or_default();
    if let Some(rec) = pop.records.get_mut(&guid) {
        // **An agent the population already holds keeps its identity and takes a
        // new route.** `adopt` refuses this case and is right to for its own
        // caller — adopting twice would restart a walking agent's day — but the
        // whole point of a panic is that it interrupts somebody who was already
        // doing something.
        rec.route = route;
        rec.schedule = None;
        rec.leg = 0;
        rec.rephase_m = -(rec.speed_of(guid) * t_s + agent_unit(guid, 0, SALT_PHASE) * 8.0);
    } else {
        // **`level_archetype` is asked for HERE and nowhere above**, and the
        // difference is measured in walks over the world: it is `O(entities)`
        // over a furnished town, and this door's *hot* caller is the crowd
        // panic, which reaches it once per frightened agent — every one of
        // which already has a record and takes the arm above. Computed
        // unconditionally it would have been one walk over every entity in the
        // level **per fleeing bystander**, on the step a shot goes off in a
        // square. Only a body with no record at all needs one, which is the
        // carjack's victim and nobody else.
        let archetype = crate::society::level_archetype(world);
        let mut rec = CrowdRecord::walking(archetype, route);
        rec.rephase_m = -(rec.speed_of(guid) * t_s + agent_unit(guid, 0, SALT_PHASE) * 8.0);
        pop.records.insert(guid, rec);
    }
    world.world_mut().insert_resource(pop);
    let mut latch = world
        .world_mut()
        .remove_resource::<PanickedRes>()
        .unwrap_or_default();
    latch.fleeing.insert(guid);
    world.world_mut().insert_resource(latch);
    true
}

/// How far somebody who wants to be somewhere else walks, metres.
///
/// Forty is over a city block: far enough that a player who turns round has
/// genuinely lost them, and short enough that the route is one straight leg
/// rather than a plan. Hoisted out of `inf_physics::d3::carjack` at wave WPN1.
pub const FLEE_M: f64 = 40.0;

/// The movement model a crowd agent walks with.
///
/// Defaults everywhere except the four things a *pedestrian* is:
///
/// * **`player_controlled: false`** — so `movement::apply_intent`, which is the
///   local player's path into the runtime intent, walks past it. An NPC's intent
///   is written by [`step_crowd`], five phases earlier in the same step;
/// * **`Gait::Walk`** — a town is people walking, and the gait is the difference
///   between 1.65 m/s and 3.75;
/// * **`RotationMode::VelocityDirection`** (the default, stated) — the body
///   faces where it is going, because a crowd agent has no camera to face;
/// * the archetype's own **capsule**, so the drawn body and the thing you can
///   walk into are one shape.
fn crowd_movement(a: &CrowdArchetype) -> CharacterMovement {
    CharacterMovement {
        player_controlled: false,
        gait: Gait::Walk,
        rotation_mode: RotationMode::VelocityDirection,
        stand_half_height_m: a.half_height_m,
        ..CharacterMovement::default()
    }
}

/// **Who is being steered, and when** — the four things [`steer_agent`] needs
/// about the *agent* rather than about the world.
///
/// A struct rather than four parameters because `clippy::too_many_arguments` is
/// right: `(guid, rec, clock, here)` are one subject read from one record on one
/// step, and a call site that got `here` and `clock` the wrong way round would
/// still compile.
struct SteerSubject<'a> {
    rec: &'a CrowdRecord,
    /// Where the body is **now** — its live transform, not the clock's answer.
    here: DVec3,
    /// The leg the step already resolved (island wave NPC1e; see
    /// [`ActiveLeg`]).
    ///
    /// It replaced the `guid` and the `clock` this struct used to carry, and
    /// that is the point rather than tidying: those two existed only so the
    /// pursuit could re-derive the leg for itself, which is the fifth
    /// resolution an agent the wave removed. A field nothing reads is a door
    /// back to asking.
    leg: ActiveLeg,
}

/// **Steer one agent along its own path** — pure pursuit, written as an intent.
///
/// The body is the movement model's to move, so what the crowd writes is
/// `intent_move`: the planar direction of a point [`PURSUIT_LOOKAHEAD_M`] ahead
/// of the agent's own projection on the spine, expressed in the aim frame the
/// model reads it in. `step_character_movement` consumes it five phases later,
/// through `move_and_slide`, which is the one door a character in this engine
/// moves through — so an NPC is stopped by the same wall the player is, steps up
/// the same stair with the same `step_height_m`, and slides along the same
/// slope.
///
/// **The clock decides the direction and the body decides the place.** The
/// lookahead is taken from `project(here)` rather than from the clock's own
/// `s_m`, so an agent that was held up follows the spine from where it actually
/// is instead of cutting across the block towards where it should have been;
/// the clock contributes [`RouteProgress::forward`], which is what tells a
/// ping-pong agent on its way home to look the other way.
///
/// # The path is the one the CLOCK names (NPC1d audit)
///
/// [`CrowdRecord::path_at`], not `rec.route.path`. For an unscheduled record
/// those are the same object and this reads identically; for a **scheduled**
/// one `route` is the *first leg*, carried as a diagnostic so a reader of
/// `route` sees something true, and steering along it walks a body towards its
/// office at six in the evening and never home.
///
/// The walkability test moved with it, for the same reason and with a sharper
/// edge: [`CrowdRoute::is_walkable`] asks the route for a **speed**, and a
/// schedule has none — a leg is a fraction of its own clock window, so
/// `CrowdRecord::scheduled` builds its diagnostic route at `0.0` m/s. Every
/// scheduled agent that reached [`Full`](CrowdTier::Full) therefore wished
/// `ZERO` on every step and stood exactly still while its clock walked on
/// without it. What a schedule has instead of a speed is a leg with *length* in
/// it, which is what is asked here.
fn steer_agent(
    world: &mut EcsWorld,
    entity: Entity,
    who: SteerSubject<'_>,
    plan: &AgentPlan,
    stats: &mut CrowdStats,
) {
    let SteerSubject { rec, here, leg } = who;
    let path = rec.path_on(leg);
    let len = path.length_m();
    // The path is the ground; the transform is the capsule's centre. Projecting
    // the centre would work on the flat and would find the wrong leg on a stair,
    // where 1.2 m of Y is most of a storey.
    let feet = here - DVec3::new(0.0, rec.archetype.feet_offset_m(), 0.0);
    let on = path.project(feet);
    // How far behind (or ahead of) its own clock the body is. Reported, not
    // corrected: the correction is the demotion re-phase, once, and a per-step
    // one would be the second authority this whole design exists without.
    let blocked = (plan.progress.s_m - on.s_m).abs() > BLOCKED_LAG_M;
    if blocked {
        stats.blocked += 1;
    }
    if let Some(mut agent) = world.world_mut().get_mut::<CrowdAgent>(entity) {
        agent.blocked = blocked;
    }
    // A scheduled agent has arrived when its leg's own clock window has run out
    // (it stands at the far end until the next leg begins) or when its body has
    // reached the end of the leg, whichever comes first. An unscheduled one
    // keeps NPC1c's rule to the letter, so nothing written before this wave
    // moves by a bit.
    let (arrived, walkable) = match &rec.schedule {
        Some(_) => (
            plan.progress.arrived || on.s_m >= len - ARRIVE_M,
            len.is_finite() && len > 0.0,
        ),
        None => (
            rec.route.mode == RouteMode::Once && on.s_m >= len - ARRIVE_M,
            rec.route.is_walkable(),
        ),
    };
    if arrived {
        stats.arrived += 1;
    }
    let wish = if arrived || !walkable {
        DVec3::ZERO
    } else {
        // **The lookahead grows until it has a HORIZONTAL lead** (NPC1c).
        //
        // A stair edge is nearly vertical: both ends of a flight sit inside one
        // core, and a segment that climbs 3.6 m over 1.5 m of run puts the
        // ordinary lookahead almost directly above the agent's own feet. A
        // planar wish computed from that is zero, the agent stops at the bottom
        // of the stairs, and the whole walk stalls one metre from the door it
        // came in through. Measured exactly that way on the island's own gate.
        //
        // So the lookahead steps forward in whole strides until the target is at
        // least [`MIN_LEAD_M`] away in plan, up to
        // [`PURSUIT_LOOKAHEAD_MAX_M`]. On flat ground the first try answers and
        // this costs one comparison.
        let dir = if plan.progress.forward { 1.0 } else { -1.0 };
        let mut wish = DVec3::ZERO;
        let mut reach = PURSUIT_LOOKAHEAD_M;
        while reach <= PURSUIT_LOOKAHEAD_MAX_M {
            let ahead = (on.s_m + dir * reach).clamp(0.0, len);
            let target = path.position_at(ahead);
            let d = crate::math::Vec2d::new(target.x - feet.x, target.z - feet.z);
            let m = (d.x * d.x + d.y * d.y).sqrt();
            if m >= MIN_LEAD_M {
                wish = DVec3::new(d.x / m, 0.0, d.y / m);
                break;
            }
            // The end of the path is as far as a lookahead can reach; growing it
            // past there would spin.
            if ahead <= 0.0 || ahead >= len {
                break;
            }
            reach += PURSUIT_LOOKAHEAD_M;
        }
        wish
    };
    let Some(mut cm) = world.world_mut().get_mut::<CharacterMovement>(entity) else {
        return;
    };
    // Into the AIM frame, through the model's own door. A crowd agent never
    // looks anywhere, so its aim yaw stays where `step_one` seeded it and this
    // is the identity today — spelled through `rotate_into_frame` anyway,
    // because the day an NPC turns its head is the day an inlined `(x, z)`
    // would send it walking sideways.
    let aim = cm.runtime.aim_yaw_deg;
    cm.runtime.intent_move =
        crate::movement::rotate_into_frame(crate::math::Vec2d::new(wish.x, wish.z), aim);
    if wish != DVec3::ZERO {
        stats.steered += 1;
    }
}

/// A fold of **one** entity's published pose — the source of a record's
/// [`CrowdRecord::pose_digest`], and `None` for an entity the store does not
/// hold (which is every entity, on a level with no character at all).
///
/// FNV-1a over the same bytes [`crate::pose::pose_state_bytes`] emits for that
/// entity, so "the digest of the pose the trace would have carried" is exactly
/// what it says.
///
/// # Per agent, on demand, and not a map (NPC1a audit)
///
/// The first cut built a `BTreeMap` of **every** entry in the store at the top
/// of [`step_crowd_banded`], which is `O(posed characters × joints)` a step to
/// serve a demotion that happens to one agent every few hundred steps — and it
/// scaled with the *level's* posed characters rather than with the crowd, so a
/// hero-heavy level paid the crowd system for poses no agent owns. Called here
/// it is `O(demotions × joints)`, and a step on which nobody leaves a posing
/// tier — almost all of them — does not touch the store at all.
fn published_pose_digest(world: &EcsWorld, guid: Uuid) -> Option<u64> {
    let store = world.world().get_resource::<crate::pose::PoseStoreRes>()?;
    let ep = store.0.get(&guid)?;
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut fold = |bytes: &[u8]| {
        for b in bytes {
            h ^= u64::from(*b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    fold(ep.skeleton.as_bytes());
    for l in &ep.pose.locals {
        for v in l
            .translation
            .iter()
            .chain(l.rotation.iter())
            .chain(l.scale.iter())
        {
            fold(&v.to_le_bytes());
        }
    }
    Some(h)
}

// ── what the rest of the engine reads ───────────────────────────────────────

/// The tier `entity` took this step, or `None` for anything that is not a crowd
/// agent — which is every hero and every authored character.
///
/// **No production caller** (the `set_debris_budget` seam, stated the way this
/// tree states them): the two in-engine readers of the verdict both had a
/// cheaper door already open — the pose evaluation queries `CrowdAgent`
/// alongside its other components, and `deform::ground_contacts` reads it off
/// the `EntityRef` its walk already holds. This is the *ad-hoc* reader, used by
/// the tests and by anything that has an [`Entity`] and one question.
#[inline]
pub fn agent_tier(world: &EcsWorld, entity: Entity) -> Option<CrowdTier> {
    world.world().get::<CrowdAgent>(entity).map(|a| a.tier)
}

/// **Every agent that has a collider in rapier this step** — the `Guid`s whose
/// tier [`has_body`](CrowdTier::has_body).
///
/// # What this replaced, and why the replacement is the falsifiable one
///
/// NPC1a's door here was `bodiless_agents`: the set the 3D bridge used to
/// *decline* to mirror a `Far` agent's `RigidBody3D`. The NPC1a audit's finding
/// 7 recorded that it had no arm and could not easily get one — a crowd agent is
/// kinematic, so severing the bridge's read changed no trace and no count this
/// tree measures — and NPC1c retires it outright: a `Far` agent now carries no
/// body **component** (`set_tier_components`), so there is nothing for the
/// bridge to decline. One authority for "is this thing physically here", which
/// is this module's own founding law, and it is the fix for the terrain-observer
/// finding as well.
///
/// What the bridge reads instead is this: which colliders belong to a crowd, so
/// they can be given the **narrowed pairing** the NPC1c solver measurement
/// named. Unlike its predecessor, that read is measurable in milliseconds —
/// 288 moving kinematic capsules over a heightfield were **+0.773 ms** of
/// `world.step` against **+0.023** for the same capsules standing still, and
/// narrowing both sides of those pairs recovers 84 % of it.
///
/// Returned as a set rather than read per entity inside the bridge's walk
/// because that walk is `O(entities)` over a furnished town and this is
/// `O(agents)`. Empty — and allocation-free — when there is no population at
/// all, which is every level committed before NPC1a.
pub fn crowd_colliders(world: &EcsWorld) -> BTreeSet<Uuid> {
    let Some(pop) = world.world().get_resource::<CrowdPopulationRes>() else {
        return BTreeSet::new();
    };
    pop.records
        .iter()
        .filter(|(_, r)| r.tier.has_body())
        .map(|(g, _)| *g)
        .collect()
}

/// **Every agent that is standing against something**, in `Guid` order — the
/// `Guid`s whose [`CrowdAgent::blocked`] verdict is set.
///
/// The reader is `inf_physics::d3::gameplay::step_crowd_doors`, which opens the
/// door a blocked agent is leaning on. It reads this rather than scanning every
/// agent for a doorway because the alternative is `O(agents x doorways)` over a
/// city that plans 19 790 of them, and the blocked set on a settled town is a
/// handful.
///
/// Off the records rather than off a component query, for
/// [`crowd_colliders`]'s reason: `O(agents)` rather than `O(entities)`, and
/// allocation-free on a level with no crowd.
pub fn blocked_agents(world: &EcsWorld) -> Vec<Uuid> {
    let Some(pop) = world.world().get_resource::<CrowdPopulationRes>() else {
        return Vec::new();
    };
    let mut out: Vec<Uuid> = Vec::new();
    for (guid, rec) in &pop.records {
        if !rec.tier.steers() {
            continue;
        }
        let Some(e) = world.entity_of(*guid) else {
            continue;
        };
        if world
            .world()
            .get::<CrowdAgent>(e)
            .map(|a| a.blocked)
            .unwrap_or(false)
        {
            out.push(*guid);
        }
    }
    out
}

/// This step's crowd counters, or the zero stats on a world with no population.
pub fn crowd_stats(world: &EcsWorld) -> CrowdStats {
    let Some(pop) = world.world().get_resource::<CrowdPopulationRes>() else {
        return CrowdStats::default();
    };
    let mut s = CrowdStats::default();
    for r in pop.records.values() {
        s.per_tier[r.tier.as_u8() as usize] += 1;
    }
    s
}

/// **The crowd's canonical bytes** — the shape a replay / PIE trace folds,
/// exactly like [`crate::deform::deform_state_bytes`] and
/// [`crate::pose::pose_state_bytes`].
///
/// # The trace re-shape, stated as arithmetic
///
/// A posed character contributes `36 + 40 · joints` bytes to
/// [`crate::pose::pose_state_bytes`] — **6 476 B** at the starter character's
/// 161 bones. A [`Far`](CrowdTier::Far) agent evaluates no pose, so it
/// contributes **nothing** there; a [`Dormant`](CrowdTier::Dormant) one has no
/// entity, so it contributes nothing to the sim snapshot either. What would then
/// be invisible is the thing that decided it, so this section carries **57 bytes
/// an agent at every tier**: the `Guid` (16), the tier (1), where it stands (24),
/// the cached digest of the pose it last published (8), since NPC1c the phase
/// the near tiers handed back to the clock (8), and since NPC1d the leg of its
/// day it is on (1).
///
/// That is the whole re-shape: at N = 100 agents all Far the crowd costs
/// `100 × 58 = 5 800` B a step against `100 × 6 476 = 647 600` B — **112×**.
///
/// # Why the RE-PHASE is folded (NPC1c)
///
/// [`CrowdRecord::rephase_m`] is the one field here that a *body* wrote. It is
/// set when an agent leaves a tier with a controller, from where that body
/// actually got to — which is a function of everything the body met on the way:
/// a wall, a slope, a shut door, another agent. Two hosts whose bodies met
/// different worlds diverge here first and in the position a step later, and the
/// step a reader needs is the first one. Eight bytes for that is the same trade
/// the pose digest above makes.
///
/// # What the DIGEST buys, precisely
///
/// A demoted agent's pose is not merely small in the trace; it is **gone** —
/// [`crate::pose::step_pose_evaluation`] rebuilds its store from scratch each
/// step, so an agent that stopped posing has no entry and no current pose to
/// describe. The digest is therefore not a summary of live state; it is a fold of
/// the last pose the agent published, carried as **history**.
///
/// That is what makes it worth eight bytes: it puts the *step at which an agent
/// left the pose path* into the trace. Without it, a host that demoted one agent
/// a single step early would produce identical bytes until the two hosts happened
/// to run a pose that differed — which on a stationary crowd could be never. With
/// it, the two diverge on the step they disagreed, which is the step a reader
/// needs.
///
/// A section that emitted only the tier would nearly do the same job and would
/// lose one case: two runs that demoted the same agent on the same step from
/// *different* poses. That is the case a mid-trace start produces.
///
/// The position is folded even for materialized agents, where the sim snapshot
/// already carries it. That is 24 duplicated bytes an agent, spent on purpose: a
/// **Dormant** agent has no snapshot entry at all, and a section whose meaning
/// changed with the tier would be a section a reader has to case-split.
///
/// Appended to the sim's `state_bytes`, which is **hashed and never decoded**,
/// so this needs no version and no reader. A level with no population produces
/// an empty vec and every pre-NPC1a trace is byte-identical.
pub fn crowd_state_bytes(world: &EcsWorld) -> Vec<u8> {
    let Some(pop) = world.world().get_resource::<CrowdPopulationRes>() else {
        return Vec::new();
    };
    let mut out = Vec::with_capacity(pop.records.len() * AGENT_TRACE_BYTES);
    // BTreeMap: `Guid` order, so the bytes are a property of the level and not
    // of bevy's archetype layout.
    for (guid, rec) in &pop.records {
        out.extend_from_slice(guid.as_bytes());
        out.push(rec.tier.as_u8());
        out.extend_from_slice(&rec.last.x.to_le_bytes());
        out.extend_from_slice(&rec.last.y.to_le_bytes());
        out.extend_from_slice(&rec.last.z.to_le_bytes());
        out.extend_from_slice(&rec.pose_digest.to_le_bytes());
        out.extend_from_slice(&rec.rephase_m.to_le_bytes());
        out.push(rec.leg);
    }
    out
}

/// Bytes one agent contributes to [`crowd_state_bytes`]: 16 `Guid` + 1 tier +
/// 24 position + 8 digest + 8 re-phase + 1 leg.
///
/// Pinned as a constant because the gates quote it: the re-shape's whole claim
/// is a ratio between this number and a posed character's 6 476, and a ratio
/// with a drifting denominator is not a claim.
///
/// # Why the LEG is folded (NPC1d)
///
/// [`CrowdRecord::leg`] is what the step compares a re-phase against: a body
/// that starts a new leg drops the metre it had fallen behind the old one. Two
/// hosts that disagreed about which leg an agent is on would carry different
/// phases into the same walk and part company a step later, in the position —
/// so the byte that decides it goes in the trace, on `rephase_m`'s own terms.
/// One byte holds a day, and [`CrowdSchedule::new`] refuses a schedule longer
/// than [`MAX_SCHEDULE_LEGS`] rather than letting leg 256 wear leg 0's number
/// (NPC1d audit — the sentence used to live here and nowhere else).
pub const AGENT_TRACE_BYTES: usize = 16 + 1 + 24 + 8 + 8 + 1;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::StreamingSource;

    fn guid(n: u128) -> Uuid {
        Uuid::from_u128(n)
    }

    fn band_at(x: f64) -> CrowdBand {
        CrowdBand::from_anchors([DVec3::new(x, 0.0, 0.0)], DEFAULT_CROWD_RADII)
    }

    /// **It fails toward FULL.** No sources means no banding, which is what keeps
    /// every pre-NPC1a fixture and every committed level behaving exactly as
    /// before.
    #[test]
    fn a_world_with_no_streaming_source_tiers_everything_full() {
        let b = CrowdBand::from_anchors(Vec::<DVec3>::new(), DEFAULT_CROWD_RADII);
        assert!(b.is_unbounded());
        assert_eq!(b.stamp(), 0);
        assert_eq!(b.tier(DVec3::new(1e9, 0.0, -1e9)), CrowdTier::Full);
        // …and a NaN point in an unbounded band is Full too: refusing it would
        // silently change a fixture that has one.
        assert_eq!(b.tier(DVec3::new(f64::NAN, 0.0, 0.0)), CrowdTier::Full);

        // A source at a non-finite position leaves no anchors, so the band fails
        // open the same way.
        let nan = CrowdBand::from_anchors([DVec3::new(f64::NAN, 0.0, 0.0)], DEFAULT_CROWD_RADII);
        assert!(nan.is_unbounded());
        // A non-finite radius, and radii out of order, are both refused.
        assert!(CrowdBand::from_anchors([DVec3::ZERO], (32.0, f64::NAN, 512.0)).is_unbounded());
        assert!(CrowdBand::from_anchors([DVec3::ZERO], (96.0, 32.0, 512.0)).is_unbounded());
    }

    /// The four tiers over a real anchor, at their own boundaries.
    #[test]
    fn the_band_tiers_a_point_by_its_nearest_anchor() {
        let b = band_at(0.0);
        assert!(!b.is_unbounded());
        // Snapped to the lattice cell centre, exactly as `SimBand` does.
        assert_eq!(b.anchors(), [DVec3::new(8.0, 0.0, 8.0)]);
        let at = |x: f64| b.tier(DVec3::new(x, 0.0, 8.0));
        assert_eq!(at(8.0), CrowdTier::Full);
        assert_eq!(at(39.0), CrowdTier::Full); //  31 m
        assert_eq!(at(41.0), CrowdTier::Near); //  33 m
        assert_eq!(at(103.0), CrowdTier::Near); //  95 m
        assert_eq!(at(105.0), CrowdTier::Far); //  97 m
        assert_eq!(at(519.0), CrowdTier::Far); // 511 m
        assert_eq!(at(521.0), CrowdTier::Dormant); // 513 m

        // A NaN point in a BANDED world falls through to the cheapest tier.
        assert_eq!(b.tier(DVec3::new(f64::NAN, 0.0, 0.0)), CrowdTier::Dormant);

        // The NEAREST anchor wins: a second source next to the far point
        // promotes it all the way back to Full.
        let two = CrowdBand::from_anchors(
            [DVec3::ZERO, DVec3::new(520.0, 0.0, 8.0)],
            DEFAULT_CROWD_RADII,
        );
        assert_eq!(two.tier(DVec3::new(521.0, 0.0, 8.0)), CrowdTier::Full);
        assert_ne!(two.stamp(), b.stamp(), "a new anchor is a new membership");
    }

    /// **The tier is a function of the SET of source positions**, not of the
    /// order a world walk produced them in, nor of duplicates, nor of height.
    #[test]
    fn the_band_is_a_function_of_the_source_set() {
        let a = CrowdBand::from_anchors(
            [DVec3::new(0.0, 0.0, 0.0), DVec3::new(300.0, 0.0, 40.0)],
            DEFAULT_CROWD_RADII,
        );
        let b = CrowdBand::from_anchors(
            [
                DVec3::new(300.0, 0.0, 40.0),
                DVec3::new(1.0, 5.0, 1.0),
                DVec3::new(0.0, 0.0, 0.0),
            ],
            DEFAULT_CROWD_RADII,
        );
        assert_eq!(a, b, "order, height and duplicates must not move the band");
    }

    /// **Hysteresis is refused, and this is what that costs** — the
    /// `SimBand::a_source_parked_on_a_lattice_line_rebands_every_step` arm, one
    /// system over.
    ///
    /// An agent parked on a tier boundary re-tiers on every step, and the bound
    /// this holds is that the thrash alternates between exactly **two** tiers —
    /// the two it is between — rather than wandering. A stateful tier would fix
    /// it and would stop being a pure function of sim state, which is the whole
    /// reason PIE equals shipping here.
    #[test]
    fn an_agent_parked_on_a_tier_boundary_alternates_between_two() {
        let b = band_at(0.0);
        // The Full/Near boundary is 32 m from the snapped anchor at (8, 8).
        let mut tiers = Vec::new();
        for step in 0..60 {
            let x = 8.0 + DEFAULT_CROWD_FULL_M + if step % 2 == 0 { -0.001 } else { 0.001 };
            tiers.push(b.tier(DVec3::new(x, 0.0, 8.0)));
        }
        let distinct: BTreeSet<CrowdTier> = tiers.iter().copied().collect();
        let changes = tiers.windows(2).filter(|w| w[0] != w[1]).count();
        println!(
            "NPC1a tier edge: an agent jittering +/-1 mm across the Full/Near \
             boundary re-tiers {changes} times in {} steps, over {} distinct tiers",
            tiers.len(),
            distinct.len()
        );
        assert_eq!(
            distinct.len(),
            2,
            "a parked agent produced {} tiers — the thrash is a wander, not the \
             two it is between",
            distinct.len()
        );
        assert_eq!(
            changes,
            tiers.len() - 1,
            "the arm is not measuring the edge"
        );

        // THE CONTROL: the same jitter a metre inside the boundary never moves.
        let inside: Vec<CrowdTier> = (0..60)
            .map(|step| {
                let x =
                    8.0 + DEFAULT_CROWD_FULL_M - 1.0 + if step % 2 == 0 { -0.001 } else { 0.001 };
                b.tier(DVec3::new(x, 0.0, 8.0))
            })
            .collect();
        assert!(
            inside.windows(2).all(|w| w[0] == w[1]),
            "the same jitter away from the boundary re-tiered — the band is \
             buying nothing"
        );
    }

    /// **WHAT REFUSING HYSTERESIS ACTUALLY COSTS** (NPC1a audit) — the
    /// boundary-thrash arm one rung further down, in the currency the thrash is
    /// paid in.
    ///
    /// `an_agent_parked_on_a_tier_boundary_alternates_between_two` measures the
    /// *bound* on the thrash: two tiers, never a wander. That is the right thing
    /// to hold and it is not a cost, and the wave's ledger read it as one ("the
    /// cost of refusing it is measured rather than argued"). On the `Full`/`Near`
    /// line the cost really is nothing — the two tiers differ by a hand pass. On
    /// the **`Far`/`Dormant`** line it is an entity **spawned and despawned every
    /// step**: six components built and a subtree torn down, sixty times a second
    /// per parked agent, plus a `Guid` index insert and remove. A crowd standing
    /// near the edge of the world is the ordinary outcome rather than the exotic
    /// one, so the number belongs in the record NPC1b and NPC1d read.
    ///
    /// The arm asserts the shape (every step materializes and dematerializes)
    /// rather than a millisecond, and PRINTS the count. Fixing it is not this
    /// audit's to do and is not hysteresis: the fixes that keep the tier a pure
    /// function of sim state are a pooled entity or a quantized agent position,
    /// and both belong with the wave that has a renderer in it.
    /// # Where the thrash actually lives
    ///
    /// Not under a jittering source: the lattice exists precisely so that
    /// sub-cell movement cannot move an anchor, and a source wandering inside
    /// one 16 m cell re-tiers nothing. It lives on the **lattice line**, where a
    /// millimetre of solver residue moves the snapped anchor a whole 16 m — the
    /// `SimBand` arm's own mechanism, one system over, and here it is worth an
    /// entity a step rather than a re-stamp.
    #[test]
    fn a_parked_agent_on_the_dormant_edge_spawns_and_despawns_every_step() {
        let mut records = BTreeMap::new();
        records.insert(
            guid(0xE30),
            CrowdRecord::standing(CrowdArchetype::default(), DVec3::ZERO),
        );
        let mut world = crowd_world(records);

        // The line at 32 cells snaps to a centre 504 m from the agent on one
        // side (`Far` — it exists) and 520 m on the other (`Dormant` — it does
        // not), across the 512 m `DEFAULT_CROWD_FAR_M` boundary.
        let line = BAND_LATTICE_M * 32.0;
        let (mut spawned, mut despawned) = (0u64, 0u64);
        for step in 0..60 {
            let d = if step % 2 == 0 { -0.001 } else { 0.001 };
            move_source(&mut world, line + d);
            let s = step_crowd(&mut world, 1.0 / 60.0);
            spawned += s.spawned;
            despawned += s.despawned;
        }
        println!(
            "NPC1a hysteresis cost: one agent whose anchor sits on a lattice \
             line across the Far/Dormant boundary cost {spawned} entity \
             spawn(s) and {despawned} despawn(s) in 60 steps — one second of \
             wall clock, for ONE NPC"
        );
        assert!(
            spawned >= 29 && despawned >= 29,
            "the fixture is not straddling the Dormant edge: {spawned} spawn(s) \
             / {despawned} despawn(s) in 60 steps — the arm is measuring a \
             steady state and the cost it names is unrecorded"
        );
    }

    /// **AND ON THE `Full`/`Near` LINE IT IS A MOVEMENT MODEL A STEP** (NPC1c
    /// audit) — the arm above's own sentence, falsified by the clause this wave
    /// is proudest of.
    ///
    /// The NPC1a audit wrote *"On the `Full`/`Near` line the cost really is
    /// nothing — the two tiers differ by a hand pass"*, and it was true: the two
    /// rungs differed by [`CrowdTier::hand_ik`] and nothing else. NPC1c made the
    /// tier own the **components** and put [`CrowdTier::steers`] on `Full`
    /// alone, so crossing that line now removes
    /// [`CharacterMovement`](crate::components::CharacterMovement) and
    /// `CharacterController3D` and inserts **fresh defaults** on the way back —
    /// and every field of `MovementRuntime` goes with them: the velocity, the
    /// mode, the smoothed body yaw, `grounded`, the landing clocks, and the
    /// `seeded` latch that takes the authored facing and runs `settle_on_spawn`
    /// exactly once.
    ///
    /// Hysteresis is refused by design and the thrash is not a jitter: it lives
    /// on the 16 m lattice line, where a millimetre of solver residue moves the
    /// snapped anchor a whole cell (`SimBand`'s own mechanism). So an agent
    /// parked there is a character whose mover starts from nothing sixty times a
    /// second — it can never accumulate a fall, never finish a gait ramp, and
    /// re-settles its feet every step.
    ///
    /// **The arm reads the STATE, not the component count**, because "the
    /// component was replaced" and "the state was lost" are different facts and
    /// only the second one costs anything: a marker is written into the runtime
    /// on every step the model exists and read back on the next one.
    ///
    /// Not fixed here, and it is not hysteresis' to fix — the remedies that keep
    /// the tier a pure function of sim state are the same two the arm above
    /// names (a quantized agent position, or carrying the runtime across the
    /// rung) and both are a policy decision about what a demoted body remembers.
    #[test]
    fn a_parked_agent_on_the_full_edge_gets_a_new_movement_model_every_step() {
        /// A yaw no default ever holds, written into the runtime and looked for
        /// again on the next step the model exists.
        const MARK: f64 = 123.456;

        let mut records = BTreeMap::new();
        records.insert(
            guid(0xE31),
            CrowdRecord::standing(CrowdArchetype::humanoid(None, None, None), DVec3::ZERO),
        );
        let mut world = crowd_world(records);

        // The line at two cells: cell 1's centre is 24 m from the agent (`Full`,
        // inside the 32 m radius) and cell 2's is 40 m (`Near`).
        let line = BAND_LATTICE_M * 2.0;
        let (mut steered, mut bodiless_of_model, mut wiped) = (0u64, 0u64, 0u64);
        let mut tiers: BTreeSet<CrowdTier> = BTreeSet::new();
        for step in 0..60 {
            let d = if step % 2 == 0 { -0.001 } else { 0.001 };
            move_source(&mut world, line + d);
            step_crowd(&mut world, 1.0 / 60.0);
            let e = world
                .entity_of(guid(0xE31))
                .expect("both tiers are materialized");
            tiers.insert(
                world
                    .world()
                    .get::<CrowdAgent>(e)
                    .expect("the tier verdict")
                    .tier,
            );
            match world
                .world_mut()
                .get_mut::<crate::components::CharacterMovement>(e)
            {
                Some(mut cm) => {
                    steered += 1;
                    if cm.runtime.body_yaw_deg != MARK {
                        wiped += 1;
                    }
                    cm.runtime.body_yaw_deg = MARK;
                }
                None => bodiless_of_model += 1,
            }
        }

        println!(
            "NPC1c hysteresis cost: one agent whose anchor sits on a lattice \
             line across the Full/Near boundary carried a `CharacterMovement` on \
             {steered} of 60 steps and lost its runtime {wiped} times — the \
             NPC1a audit's \"the two tiers differ by a hand pass\" is retired"
        );
        assert_eq!(
            tiers.len(),
            2,
            "the fixture is not straddling the Full/Near edge: {tiers:?}"
        );
        assert!(
            steered >= 29 && bodiless_of_model >= 29,
            "the model is not being taken off and put back: {steered} step(s) \
             with one, {bodiless_of_model} without"
        );
        // Every step the model exists it is a fresh default, so the marker
        // written on the previous such step is never there. The first one has no
        // marker to lose, which is the one this bound allows for.
        assert!(
            wiped >= steered - 1,
            "the runtime survived {} of {steered} promotions — if a demoted \
             agent now KEEPS its movement state, this cost is paid and the \
             ledger sentence has to say so",
            steered - wiped
        );
    }

    /// **THE 1.20 m GAP, CLOSED** — the NPC1b audit's tripwire, met.
    ///
    /// That audit's finding 5 measured a crowd agent's drawn body against its
    /// capsule and found them **1.20 m** apart: P29.6's ruling is that a rig's
    /// origin is its **feet** and a character's entity transform is its capsule
    /// **centre**, both projectors lift the pose through
    /// [`crate::pose::model_to_world`], and that door was gated on
    /// [`CharacterMovement`](crate::components::CharacterMovement) — which a
    /// crowd agent deliberately did not carry. So the drop was zero and an NPC
    /// was drawn with its feet where its capsule's middle is. The arm it left
    /// behind asserted the zero and said in its own message that it would fail
    /// the day NPC1c gave a near agent a controller.
    ///
    /// It did, at `DVec3(0.0, -1.2, 0.0)`, and this is its successor. Two
    /// halves, because the fix has two halves:
    ///
    /// * a `Full`/`Near` agent carries a `CharacterMovement` and a capsule, so
    ///   the **general** rule measures it, exactly as it measures the hero;
    /// * a `Far` agent carries neither and its feet are in the same place, so it
    ///   publishes [`CrowdAgent::feet_offset_m`] instead — and the two numbers
    ///   are asserted **equal**, which is what stops the second source being a
    ///   second opinion.
    #[test]
    fn the_crowds_foot_offset_is_the_one_the_movement_model_computes() {
        let mut records = BTreeMap::new();
        records.insert(
            guid(0xC5A),
            CrowdRecord::standing(
                CrowdArchetype::humanoid(None, None, None),
                DVec3::new(4.0, 0.0, 0.0),
            ),
        );
        let mut world = crowd_world(records);
        move_source(&mut world, 4.0);
        let stats = step_crowd(&mut world, 1.0 / 60.0);
        assert_eq!(stats.per_tier[0], 1, "the fixture agent is not `Full`");

        let e = world
            .entity_of(guid(0xC5A))
            .expect("the agent materialized");
        let placed = world
            .world()
            .get::<crate::components::GlobalTransform>(e)
            .expect("a transform")
            .translation();
        let drawn = crate::pose::model_to_world(&world, e).translation;
        let gap = placed - drawn;
        assert!(
            (gap.y - 0.9).abs() < 1e-6,
            "a `Full` agent must publish the humanoid capsule's 0.6 + 0.3: {gap:?}"
        );

        // The two sources, held to one number. `feet_offset_m` off the live
        // collider, and the archetype's own published offset.
        let col = *world.world().get::<Collider3D>(e).expect("a capsule");
        let cm = world
            .world()
            .get::<crate::components::CharacterMovement>(e)
            .expect("a `Full` agent carries the movement model")
            .clone();
        let measured = crate::movement::feet_offset_m(&cm, Some(&col));
        let published = world
            .world()
            .get::<CrowdAgent>(e)
            .expect("a crowd agent")
            .feet_offset_m;
        assert_eq!(
            measured, published,
            "the published foot offset and the measured one have parted company"
        );

        // …and the `Far` tier, which has neither component, publishes the same
        // drop through the other half of the door.
        move_source(&mut world, 4.0 + DEFAULT_CROWD_NEAR_M + 64.0);
        let stats = step_crowd(&mut world, 1.0 / 60.0);
        assert_eq!(stats.per_tier[2], 1, "the fixture agent is not `Far`");
        let e = world.entity_of(guid(0xC5A)).expect("still materialized");
        assert!(
            world
                .world()
                .get::<crate::components::CharacterMovement>(e)
                .is_none(),
            "a `Far` agent must carry no movement model"
        );
        let placed = world
            .world()
            .get::<crate::components::GlobalTransform>(e)
            .expect("a transform")
            .translation();
        let drawn = crate::pose::model_to_world(&world, e).translation;
        assert!(
            ((placed - drawn).y - 0.9).abs() < 1e-6,
            "a `Far` agent must publish a `Full` one's drop: {:?}",
            placed - drawn
        );
        println!("NPC1c: the character-space gap is closed at every tier: {measured:.2} m.");
        println!("NPC1c: the archetype publishes {published:.2} m for the tiers with no capsule.");
    }

    /// The cost ladder is monotone: every tier is cheaper than the one above it,
    /// in every dimension, and nothing costs more as it gets further away.
    #[test]
    fn the_tier_ladder_is_monotone() {
        let ladder = [
            CrowdTier::Full,
            CrowdTier::Near,
            CrowdTier::Far,
            CrowdTier::Dormant,
        ];
        for w in ladder.windows(2) {
            let (a, b) = (w[0], w[1]);
            assert!(a < b, "{a:?} must order before {b:?}");
            assert!(a.hand_ik() >= b.hand_ik(), "{a:?} vs {b:?}: hand IK grew");
            assert!(a.poses() >= b.poses(), "{a:?} vs {b:?}: pose grew");
            assert!(a.has_body() >= b.has_body(), "{a:?} vs {b:?}: body grew");
            assert!(
                a.materialized() >= b.materialized(),
                "{a:?} vs {b:?}: an entity appeared"
            );
        }
        // …and each rung really is a rung: the four tiers are four distinct
        // cost vectors, so no two of them are the same tier under two names.
        let vectors: BTreeSet<(bool, bool, bool, bool)> = ladder
            .iter()
            .map(|t| (t.hand_ik(), t.poses(), t.has_body(), t.materialized()))
            .collect();
        assert_eq!(vectors.len(), 4, "two tiers cost the same thing");
    }

    /// The trace byte count is what the doc claims, and the re-shape's ratio
    /// with it.
    ///
    /// *(NPC1c audit: this was still named `…_forty_nine_bytes` while asserting
    /// **57**. A test's name is the sentence a reader takes away from a green
    /// run, and this tree's own law is that a gate must aim at the thing it
    /// names.)*
    fn leg(start_h: f64, travel_h: f64, from: DVec3, to: DVec3) -> ScheduleLeg {
        ScheduleLeg {
            start_h,
            travel_h,
            path: inf_nav::NavPath::new([from, to]),
            arrival: crate::crowd::SlotArrival::STANDING,
        }
    }

    /// A day of four legs: home to work, work to shop, shop to work, work home.
    fn day() -> CrowdSchedule {
        let home = DVec3::new(0.0, 0.0, 0.0);
        let work = DVec3::new(100.0, 0.0, 0.0);
        let shop = DVec3::new(100.0, 0.0, 40.0);
        CrowdSchedule::new(vec![
            leg(8.0, 1.0, home, work),
            leg(12.0, 0.5, work, shop),
            leg(13.0, 0.5, shop, work),
            leg(18.0, 1.0, work, home),
        ])
        .expect("four walkable legs")
    }

    /// **The schedule is a pure function of the clock**, and midnight is not a
    /// special case: the leg that started most recently is the one running,
    /// measured round the circle.
    #[test]
    fn a_day_picks_the_leg_that_started_most_recently() {
        let d = day();
        // Mid-commute.
        assert_eq!(d.at(8.5), (0, 0.5));
        // The walk is done; the agent stands at work all afternoon.
        assert_eq!(d.at(9.0), (0, 1.0));
        assert_eq!(d.at(11.0), (0, 1.0));
        // Lunch, and back.
        assert_eq!(d.at(12.25), (1, 0.5));
        assert_eq!(d.at(13.25), (2, 0.5));
        // Home, and then all night on the last leg, THROUGH midnight.
        assert_eq!(d.at(18.5), (3, 0.5));
        assert_eq!(d.at(19.0), (3, 1.0));
        assert_eq!(d.at(23.9), (3, 1.0));
        assert_eq!(d.at(0.1), (3, 1.0));
        assert_eq!(d.at(7.9), (3, 1.0));
        // And it wraps: the same hour tomorrow is the same answer.
        for h in [0.0_f64, 3.0, 8.5, 12.25, 23.5] {
            assert_eq!(d.at(h), d.at(h + 24.0), "hour {h} differs a day later");
            assert_eq!(d.at(h), d.at(h - 24.0));
        }
        // A non-finite hour answers a stand rather than panicking.
        assert_eq!(d.at(f64::NAN), (0, 1.0));
    }

    /// A leg that cannot be walked is not a leg, and a schedule of nothing is
    /// `None` rather than an empty one nothing can index.
    #[test]
    fn a_schedule_refuses_what_it_cannot_walk() {
        let a = DVec3::ZERO;
        let b = DVec3::new(10.0, 0.0, 0.0);
        assert!(CrowdSchedule::new(vec![]).is_none());
        assert!(CrowdSchedule::new(vec![leg(8.0, 0.0, a, b)]).is_none());
        assert!(CrowdSchedule::new(vec![leg(8.0, -1.0, a, b)]).is_none());
        assert!(CrowdSchedule::new(vec![leg(8.0, f64::NAN, a, b)]).is_none());
        assert!(CrowdSchedule::new(vec![leg(f64::NAN, 1.0, a, b)]).is_none());
        // One good leg among bad ones survives, and is the only one.
        let mixed = CrowdSchedule::new(vec![
            leg(8.0, 0.0, a, b),
            leg(9.0, 1.0, a, b),
            leg(f64::NAN, 1.0, a, b),
        ])
        .expect("one leg survives");
        assert_eq!(mixed.legs().len(), 1);
        assert_eq!(mixed.legs()[0].start_h, 9.0);
        // **And a day longer than the trace byte is refused whole** (NPC1d
        // audit): the byte is what the step compares to decide "a new leg
        // starts clean", so leg 256 wearing leg 0's number is a phase that is
        // never dropped.
        let many = |n: usize| -> Vec<ScheduleLeg> {
            (0..n)
                .map(|i| leg(24.0 * i as f64 / n as f64, 0.1, a, b))
                .collect()
        };
        assert!(CrowdSchedule::new(many(MAX_SCHEDULE_LEGS)).is_some());
        assert!(
            CrowdSchedule::new(many(MAX_SCHEDULE_LEGS + 1)).is_none(),
            "a {}-leg schedule was accepted against a one-byte leg index",
            MAX_SCHEDULE_LEGS + 1
        );
    }

    /// **The whole reason a leg is a fraction of its window**: the day has the
    /// same SHAPE at any clock rate, so a gate can run one in a test process.
    #[test]
    fn the_same_hour_is_the_same_place_at_any_clock_rate() {
        let g = guid(0xDA1);
        let rec = CrowdRecord::scheduled(CrowdArchetype::humanoid(None, None, None), day());
        // Two "runs" that reached 09:30 after wildly different numbers of steps.
        let slow = rec.position_at(g, CrowdClock::new(120.0 * 3600.0, 9.5));
        let fast = rec.position_at(g, CrowdClock::new(3.0, 9.5));
        assert_eq!(
            slow.to_array().map(f64::to_bits),
            fast.to_array().map(f64::to_bits),
            "the same hour gave two places: {slow:?} against {fast:?}"
        );
        // And the sim clock alone moves it nowhere, which is what "a schedule
        // reads the DAY" means.
        let noon_a = rec.position_at(g, CrowdClock::new(0.0, 12.0));
        let noon_b = rec.position_at(g, CrowdClock::new(1.0e6, 12.0));
        assert_eq!(noon_a, noon_b);
    }

    /// The implied speed is a walking pace at the rate the island authors, and
    /// that is the arithmetic the rate was CHOSEN by.
    #[test]
    fn a_leg_walked_over_its_window_is_a_walking_pace_at_the_islands_rate() {
        // A 48-minute day: 86 400 clock-seconds in 2 880 sim seconds.
        const ISLAND_RATE: f64 = 30.0;
        let d = day();
        let commute = &d.legs()[0];
        assert_eq!(commute.path.length_m(), 100.0);
        let v = commute.implied_speed_mps(ISLAND_RATE);
        // 1 h of clock at rate 30 is 120 sim seconds; 100 m over 120 s.
        assert!(
            (v - 100.0 / 120.0).abs() < 1e-12,
            "a 100 m commute over an hour reads {v:.4} m/s"
        );
        assert!(
            (0.4..=2.0).contains(&v),
            "the island's authored rate makes a commute {v:.2} m/s, which is not \
             a walk"
        );
        // A frozen clock has no answer rather than an infinite one.
        assert_eq!(commute.implied_speed_mps(0.0), 0.0);
        assert_eq!(commute.implied_speed_mps(f64::NAN), 0.0);
    }

    /// **A scheduled agent walks each leg's OWN path**, and the record's
    /// unscheduled arithmetic is untouched beside it.
    #[test]
    fn a_scheduled_agent_walks_the_leg_the_clock_names() {
        let g = guid(0xDA2);
        let rec = CrowdRecord::scheduled(CrowdArchetype::humanoid(None, None, None), day());
        let lift = rec.archetype.feet_offset_m();
        // The jitter shifts this agent's own day, so ask about ITS hour.
        let own = |h: f64| {
            // Invert `hour_of`: feed the level hour that puts this agent at `h`.
            let j = SCHEDULE_JITTER_H * (2.0 * agent_unit(g, 0, SALT_SCHEDULE) - 1.0);
            (h + j).rem_euclid(24.0)
        };
        let at = |h: f64| rec.position_at(g, CrowdClock::new(0.0, own(h)));
        // Halfway to work is halfway along leg 0.
        let half = at(8.5);
        assert!((half.x - 50.0).abs() < 1e-9, "{half:?}");
        assert!((half.y - lift).abs() < 1e-9);
        // At work, and still there at eleven.
        assert!((at(9.0).x - 100.0).abs() < 1e-9);
        assert_eq!(at(9.0), at(11.0));
        // Lunch is on leg 1, which runs in Z rather than X.
        let lunch = at(12.25);
        assert!(
            (lunch.x - 100.0).abs() < 1e-9 && (lunch.z - 20.0).abs() < 1e-9,
            "{lunch:?}"
        );
        // Home overnight.
        assert!(at(2.0).x.abs() < 1e-9);
        assert!(rec.leg_at(g, CrowdClock::new(0.0, own(2.0))) == Some((3, 1.0)));
    }

    /// The jitter is real, derived, and bounded — a town does not commute in
    /// lockstep, and no agent's day is shifted by more than half an hour.
    #[test]
    fn every_agent_keeps_its_own_hour_and_the_shift_is_bounded() {
        let rec = CrowdRecord::scheduled(CrowdArchetype::humanoid(None, None, None), day());
        let mut seen = std::collections::BTreeSet::new();
        for n in 0..64u128 {
            let g = guid(0x5000 + n);
            let own = rec.hour_of(g, 8.0);
            let shift = {
                let d = own - 8.0;
                if d > 12.0 {
                    d - 24.0
                } else if d < -12.0 {
                    d + 24.0
                } else {
                    d
                }
            };
            assert!(
                shift.abs() <= SCHEDULE_JITTER_H + 1e-12,
                "agent {n} is shifted {shift:.4} h"
            );
            seen.insert(own.to_bits());
        }
        assert!(
            seen.len() > 50,
            "64 agents produced only {} distinct hours -- the jitter is not \
             per-agent",
            seen.len()
        );
        // And it is a constant of the agent: the same guid always answers the
        // same shift, whatever the level hour.
        let g = guid(0x5001);
        assert_eq!(rec.hour_of(g, 8.0) - 8.0, rec.hour_of(g, 15.0) - 15.0);
    }

    /// **A SCHEDULED agent at `Full` actually steers, and steers along the leg
    /// the clock names** (NPC1d audit).
    ///
    /// Two defects in one arm, and the second is the severe one.
    ///
    /// `steer_agent` read `rec.route.path`, which for a scheduled record is the
    /// **first leg** — carried as a diagnostic so a reader of `route` sees
    /// something true — so a body walked towards its office at six in the
    /// evening and never home. And it gated the whole wish on
    /// `rec.route.is_walkable()`, which asks the route for a **speed**: a
    /// schedule has none (a leg is a fraction of its own clock window, and
    /// `CrowdRecord::scheduled` builds its diagnostic route at `0.0` m/s), so
    /// **every scheduled agent that reached `Full` wished `ZERO` on every step
    /// and stood exactly still** while its clock walked on without it.
    ///
    /// Neither was visible to the island gate, because from the town's edge
    /// nothing steers at all (`+0 steered` at every one of its six hours). This
    /// is the arm that sees them: an unbounded band, so the agent is `Full`; a
    /// day whose second leg runs the other way, so "the active leg" and "leg 0"
    /// are different directions rather than different distances.
    #[test]
    fn a_scheduled_agent_at_full_steers_along_the_leg_the_clock_names() {
        let g = guid(0xDA4);
        // Leg 0 runs +X from the origin; leg 1 runs +Z from where leg 0 ended.
        // A body steering along leg 0 while the clock is on leg 1 wishes +X;
        // one steering along the leg the clock names wishes +Z.
        let home = DVec3::new(0.0, 0.0, 0.0);
        let work = DVec3::new(100.0, 0.0, 0.0);
        let shop = DVec3::new(100.0, 0.0, 100.0);
        let sched = CrowdSchedule::new(vec![leg(8.0, 1.0, home, work), leg(12.0, 1.0, work, shop)])
            .expect("two walkable legs");
        let mut records = BTreeMap::new();
        records.insert(
            g,
            CrowdRecord::scheduled(CrowdArchetype::humanoid(None, None, None), sched),
        );

        let mut world = EcsWorld::new();
        set_population(&mut world, records);
        // A clock parked in the middle of leg 1, at THIS agent's own hour.
        let own = {
            let j = SCHEDULE_JITTER_H * (2.0 * agent_unit(g, 0, SALT_SCHEDULE) - 1.0);
            (12.5 + j).rem_euclid(24.0)
        };
        world.spawn_with_guid(guid(0xC10CD), "clock", None);
        let e = world.entity_of(guid(0xC10CD)).expect("a clock");
        world
            .world_mut()
            .entity_mut(e)
            .insert(crate::components::TimeOfDay {
                seconds: own * 3600.0,
                rate: 0.0,
                longitude_deg: 0.0,
                ..Default::default()
            });

        // No streaming source anywhere, so the band is unbounded and the agent
        // materializes `Full` — which is the tier that steers.
        let first = step_crowd(&mut world, 1.0 / 60.0);
        assert_eq!(first.per_tier[0], 1, "the agent is not `Full`: {first:?}");
        // The materialization step places the body; the NEXT one steers it.
        let stats = step_crowd(&mut world, 1.0 / 60.0);
        assert_eq!(
            stats.steered, 1,
            "a scheduled `Full` agent wrote no steering intent — it is standing \
             still while its clock walks on ({stats:?})"
        );

        // …and the intent points along leg 1 (+Z), not leg 0 (+X).
        let ent = world.entity_of(g).expect("a materialized agent");
        let cm = world
            .world()
            .get::<CharacterMovement>(ent)
            .expect("a `Full` agent carries the movement model");
        let m = cm.runtime.intent_move;
        assert!(
            m.y.abs() > m.x.abs(),
            "the agent wishes ({:.3}, {:.3}) — mostly along leg 0's +X, which is \
             the leg its clock is NOT on",
            m.x,
            m.y
        );
        assert!(m.y > 0.0, "the agent walks away from its own destination");
    }

    /// **A new leg starts clean**: the metre a body fell behind one leg is not
    /// carried into the next, and the leg byte is what decides it.
    #[test]
    fn a_leg_change_drops_the_phase_and_moves_the_trace_byte() {
        let mut world = EcsWorld::new();
        let g = guid(0xDA3);
        let mut records = BTreeMap::new();
        let mut rec = CrowdRecord::scheduled(CrowdArchetype::humanoid(None, None, None), day());
        rec.rephase_m = -7.5;
        rec.leg = 0;
        records.insert(g, rec);
        set_population(&mut world, records);
        // Park the clock in the middle of leg 1 and step once.
        world.spawn_with_guid(guid(0xC10CC), "clock", None);
        let e = world.entity_of(guid(0xC10CC)).expect("a clock");
        world
            .world_mut()
            .entity_mut(e)
            .insert(crate::components::TimeOfDay {
                seconds: 12.25 * 3600.0,
                rate: 0.0,
                longitude_deg: 0.0,
                ..Default::default()
            });
        // No streaming source anywhere, so the band is unbounded and the agent
        // materializes `Full` -- the tier is not what this arm is about.
        step_crowd(&mut world, 1.0 / 60.0);
        let pop = world
            .world()
            .get_resource::<CrowdPopulationRes>()
            .expect("the population");
        let rec = &pop.records[&g];
        assert_ne!(rec.leg, 0, "the agent never left leg 0 at 12:15");
        assert_eq!(
            rec.rephase_m, 0.0,
            "leg {} kept leg 0's phase of -7.5 m",
            rec.leg
        );
        // The byte is in the trace, and it is the LAST one of the agent's run.
        let bytes = crowd_state_bytes(&world);
        assert_eq!(bytes.len(), AGENT_TRACE_BYTES);
        assert_eq!(bytes[AGENT_TRACE_BYTES - 1], rec.leg);
    }

    #[test]
    fn the_agent_trace_section_is_fifty_eight_bytes() {
        let mut records = BTreeMap::new();
        for i in 0..7u128 {
            records.insert(
                guid(0x900 + i),
                CrowdRecord::standing(CrowdArchetype::default(), DVec3::ZERO),
            );
        }
        let mut world = EcsWorld::new();
        set_population(&mut world, records);
        let bytes = crowd_state_bytes(&world);
        assert_eq!(bytes.len(), 7 * AGENT_TRACE_BYTES);
        // 49 at NPC1a; NPC1c adds the eight bytes of `rephase_m`, which is the
        // one field here a BODY wrote and therefore the one a divergence shows
        // up in first.
        assert_eq!(AGENT_TRACE_BYTES, 58);
        // The claim the ledger quotes: a 161-bone posed character is 6 476 B.
        const POSED: usize = 36 + 161 * 40;
        assert_eq!(POSED / AGENT_TRACE_BYTES, 111);
        println!(
            "NPC1d trace: {AGENT_TRACE_BYTES} B an agent against {POSED} B a posed \
             character — {}x (49 B and 132x at NPC1a, before the re-phase)",
            POSED / AGENT_TRACE_BYTES
        );

        // A world with no population folds nothing at all, so every pre-NPC1a
        // trace is byte-identical.
        assert!(crowd_state_bytes(&EcsWorld::new()).is_empty());
    }

    /// The mixer is the SplitMix64 finalizer, pinned against the spec rather
    /// than against one of the tree's four copies of it.
    #[test]
    fn the_mixer_is_the_splitmix64_finalizer() {
        fn reference(mut x: u64) -> u64 {
            x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            x ^ (x >> 31)
        }
        const GOLDEN: u64 = 0x9e37_79b9_7f4a_7c15;
        let g = guid(0x1234_5678_9abc_def0);
        for tick in [0u64, 1, 7, u64::MAX] {
            let bits = g.as_u128();
            let want = reference(
                (bits as u64)
                    ^ ((bits >> 64) as u64).wrapping_mul(GOLDEN)
                    ^ tick.wrapping_mul(GOLDEN)
                    ^ SALT_SPEED,
            );
            assert_eq!(agent_rand(g, tick, SALT_SPEED), want);
        }
        // The three arguments each separate: two agents, two ticks and two
        // salts all give different streams.
        assert_ne!(
            agent_rand(g, 0, SALT_SPEED),
            agent_rand(guid(2), 0, SALT_SPEED)
        );
        assert_ne!(agent_rand(g, 0, SALT_SPEED), agent_rand(g, 1, SALT_SPEED));
        assert_ne!(agent_rand(g, 0, SALT_SPEED), agent_rand(g, 0, SALT_PHASE));
        // …and the uniform really is in [0, 1).
        for i in 0..1000u64 {
            let u = agent_unit(guid(i as u128), i, SALT_PHASE);
            assert!((0.0..1.0).contains(&u), "draw {i} was {u}");
        }
    }

    /// **AN UNDRESSED WORLD DRAWS EXACTLY WHAT IT DREW BEFORE THE WAVE** (wave
    /// EMS3 audit) — the byte-stability claim, as an assertion.
    ///
    /// EMS3 moved both projectors from [`agent_look`] to [`agent_look_in`] and
    /// says every level committed before the wave draws identical pixels,
    /// because an absent [`AppearanceRes`] falls back to [`derived_outfit`] —
    /// the exact swap `agent_look` chooses. `projector_mirror` pins that the two
    /// hosts call the same door; nothing pinned that the door answers the same
    /// thing, and a wrong `agent_look_in` would be wrong on BOTH hosts
    /// identically, so PIE == shipping would go on passing.
    ///
    /// The build is the half that would go quietly: `agent_look_in` takes its
    /// tint from [`CrowdLook::of`] and its build from [`agent_look`], and
    /// `CrowdLook::of`'s own build is the range's **midpoint** — so an
    /// `agent_look_in` "simplified" to one `CrowdLook::of` call would resize
    /// every crowd agent in the engine to one body and change no test.
    #[test]
    fn an_undressed_agent_draws_exactly_what_the_derived_look_draws() {
        let bare = EcsWorld::new();
        for i in 0..256u128 {
            let g = guid(i);
            assert_eq!(
                agent_look_in(&bare, g),
                agent_look(g),
                "an undressed agent is drawn differently since wave EMS3 — every \
                 level committed before it moves"
            );
        }
        // …and dressing somebody moves the TINT and leaves the BUILD alone,
        // which is what "nobody changes their height at a wardrobe" means.
        let mut w = EcsWorld::new();
        let g = guid(0xc0a7);
        let before = agent_look_in(&w, g);
        let other = (derived_outfit(g) as usize + 3) % CROWD_LOOKS.len();
        assert!(set_appearance(
            &mut w,
            g,
            Appearance {
                outfit: other as u8
            }
        ));
        let after = agent_look_in(&w, g);
        assert_ne!(before.tint, after.tint, "the wardrobe changed no pixels");
        assert_eq!(
            before.build, after.build,
            "a coat changed somebody's height"
        );
        assert_eq!(after.tint, CROWD_LOOKS[other]);
        // An out-of-range appearance wraps rather than panicking — content that
        // is wrong should be visible, and `CROWD_LOOKS` is indexed in one place.
        assert!(set_appearance(&mut w, g, Appearance { outfit: 200 }));
        assert_eq!(
            agent_look_in(&w, g).tint,
            CROWD_LOOKS[200 % CROWD_LOOKS.len()]
        );
        // …and clearing it returns the derived draw exactly.
        clear_appearance(&mut w);
        assert_eq!(agent_look_in(&w, g), agent_look(g));
    }

    /// **A crowd is not a thousand clones, and the look is derived rather than
    /// stored** (wave NPC1b).
    ///
    /// The three claims that make the variation legal at all: it is a pure
    /// function of the `Guid` (so both hosts agree and nothing is folded into a
    /// trace), it does not vary with the clock (so an agent does not change
    /// clothes as it walks), and it actually spreads (so the crowd looks like a
    /// crowd rather than like one model repeated).
    #[test]
    fn an_agents_look_is_derived_from_its_guid_and_spreads() {
        // Pure, and constant over the agent's life — `agent_look` takes no tick
        // at all, which is what makes the second half structural.
        let g = guid(0xfeed_face);
        assert_eq!(agent_look(g), agent_look(g));
        assert_ne!(agent_look(g).tint, agent_look(guid(0xfeed_fade)).tint);

        // Every look is a member of the table and every build is in range.
        let mut seen = [0usize; CROWD_LOOKS.len()];
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        const N: u128 = 4_000;
        for i in 0..N {
            let l = agent_look(guid(i));
            let at = CROWD_LOOKS
                .iter()
                .position(|t| *t == l.tint)
                .expect("a tint that is not one of the palette swaps");
            seen[at] += 1;
            lo = lo.min(l.build);
            hi = hi.max(l.build);
            assert!(
                (CROWD_BUILD_RANGE.0..=CROWD_BUILD_RANGE.1).contains(&l.build),
                "build {} is outside {:?}",
                l.build,
                CROWD_BUILD_RANGE
            );
        }
        // **The anti-clone arm.** A table nobody indexes into, or a mixer that
        // collapsed, would put everything in one bucket and satisfy every
        // assertion above. Four thousand draws over eight looks is 500 each; a
        // fifth of that is a bound no working mixer comes near and no broken one
        // clears.
        let floor = N as usize / CROWD_LOOKS.len() / 5;
        for (i, n) in seen.iter().enumerate() {
            assert!(
                *n > floor,
                "look {i} was drawn {n} times of {N}, floor {floor}"
            );
        }
        // …and the build really does span its range rather than sitting at one
        // end of it.
        assert!(
            lo < CROWD_BUILD_RANGE.0 + 0.01 && hi > CROWD_BUILD_RANGE.1 - 0.01,
            "the build spread only {lo}..{hi}"
        );

        // The tint MULTIPLIES rather than replaces, and it leaves alpha alone —
        // so an authored material still shows through its own variation.
        let l = CrowdLook {
            tint: [0.5, 0.25, 2.0],
            build: 1.0,
        };
        assert_eq!(l.over([0.8, 0.8, 0.8, 0.5]), [0.4, 0.2, 1.6, 0.5]);
    }

    /// The caster rung: `Full` casts its own silhouette, everything else casts
    /// through the crowd's shared proxy. Held here rather than only at the two
    /// projectors, because a tier predicate that agreed with itself in one host
    /// is the failure mode this whole module is shaped against.
    #[test]
    fn only_the_full_tier_casts_a_skinned_shadow() {
        assert!(CrowdTier::Full.skinned_caster());
        for t in [CrowdTier::Near, CrowdTier::Far, CrowdTier::Dormant] {
            assert!(!t.skinned_caster(), "{t:?} asked for a skinned caster");
        }
        // …and it is the same rung as the collider, not a second opinion: a tier
        // that keeps a body is not automatically a tier that keeps a silhouette.
        assert!(CrowdTier::Near.has_body() && !CrowdTier::Near.skinned_caster());
    }

    /// **The shadow LOD is a THIRD rung, and it lands where the ladder already
    /// splits** (island wave NPC1e).
    ///
    /// `skinned_caster` chooses between a silhouette and the shared proxy;
    /// `casts_shadow` chooses whether there is a caster at all. `Near` keeps a
    /// proxy — it is inside 96 m, where a person's shadow on the pavement is
    /// something a player reads — and `Far` casts nothing, which is NPC1b's
    /// carried item 4 ("proxies that stop casting past a radius").
    #[test]
    fn the_crowds_shadow_lod_stops_at_the_near_rung() {
        assert!(CrowdTier::Full.casts_shadow() && CrowdTier::Near.casts_shadow());
        for t in [CrowdTier::Far, CrowdTier::Dormant] {
            assert!(!t.casts_shadow(), "{t:?} still casts a shadow");
        }
        // The two predicates are ordered, not independent: anything that casts
        // its own silhouette casts *something*. A ladder where they crossed
        // would ask the raster for a skinned caster it had already been told to
        // drop.
        for t in [
            CrowdTier::Full,
            CrowdTier::Near,
            CrowdTier::Far,
            CrowdTier::Dormant,
        ] {
            assert!(
                !t.skinned_caster() || t.casts_shadow(),
                "{t:?} casts its own silhouette and no shadow at once"
            );
            // …and a tier with no entity cannot cast: `Dormant` is not drawn at
            // all, so a `true` here would be an opinion about something that is
            // not in the world.
            assert!(t.materialized() || !t.casts_shadow(), "{t:?}");
        }
    }

    /// A route is a pure function of the clock, ping-pongs, and stands still
    /// when it has nowhere to go.
    #[test]
    fn a_route_is_a_pure_function_of_route_and_clock() {
        let r = CrowdRoute::between(DVec3::new(0.0, 0.0, 0.0), DVec3::new(10.0, 0.0, 0.0), 1.0);
        assert_eq!(r.position_at(0.0, 0.0).x, 0.0);
        assert_eq!(r.position_at(5.0, 0.0).x, 5.0);
        assert_eq!(r.position_at(10.0, 0.0).x, 10.0);
        assert_eq!(r.position_at(15.0, 0.0).x, 5.0, "it did not turn round");
        assert_eq!(r.position_at(20.0, 0.0).x, 0.0);
        assert_eq!(r.position_at(21.0, 0.0).x, 1.0, "the period is wrong");
        // Same input, same output — twice, because "pure function" is the claim.
        assert_eq!(r.position_at(7.5, 1.25), r.position_at(7.5, 1.25));
        // A stand stands, whatever the clock says.
        let s = CrowdRoute::standing(DVec3::new(3.0, 4.0, 5.0));
        assert_eq!(s.position_at(1e6, 3.0), DVec3::new(3.0, 4.0, 5.0));
        // …and so does a route with no speed, or a non-finite clock.
        assert_eq!(
            CrowdRoute {
                speed_mps: 0.0,
                ..r.clone()
            }
            .position_at(9.0, 0.0),
            r.origin()
        );
        assert_eq!(r.position_at(f64::NAN, 0.0), r.origin());
    }

    /// A world holding a source and a population, with the source at `x`.
    fn crowd_world(records: BTreeMap<Uuid, CrowdRecord>) -> EcsWorld {
        let mut world = EcsWorld::new();
        let src = world.spawn_with_guid(guid(0xF1), "Player", None);
        world
            .world_mut()
            .entity_mut(src)
            .insert(StreamingSource { radius_m: 0.0 });
        world.propagate();
        set_population(&mut world, records);
        world
    }

    fn move_source(world: &mut EcsWorld, x: f64) {
        let e = world.entity_of(guid(0xF1)).expect("the source");
        crate::sim::set_translation(world, e, Vec3d::new(x, 0.0, 0.0));
        world.propagate();
    }

    // -- NPC1c: the route, the components and the two hand-offs --------------

    /// **A route mode is a rule about the end of a path**, and each of the three
    /// is a different answer.
    #[test]
    fn the_three_route_modes_fold_the_clock_three_ways() {
        let path = inf_nav::NavPath::new([DVec3::ZERO, DVec3::new(10.0, 0.0, 0.0)]);
        let ping = CrowdRoute::along(path.clone(), 1.0, RouteMode::PingPong);
        let once = CrowdRoute::along(path.clone(), 1.0, RouteMode::Once);
        let round = CrowdRoute::along(path, 1.0, RouteMode::Loop);

        // Ping-pong turns round, and says which way it is going -- which is what
        // the near tiers' pursuit reads.
        assert_eq!(ping.progress_at(4.0, 0.0).s_m, 4.0);
        assert!(ping.progress_at(4.0, 0.0).forward);
        assert_eq!(ping.progress_at(15.0, 0.0).s_m, 5.0);
        assert!(!ping.progress_at(15.0, 0.0).forward);
        assert!(
            !ping.progress_at(1e6, 0.0).arrived,
            "a patrol never arrives"
        );

        // Once stops at the end and SAYS it stopped.
        assert_eq!(once.progress_at(4.0, 0.0).s_m, 4.0);
        assert!(!once.progress_at(4.0, 0.0).arrived);
        assert_eq!(once.progress_at(15.0, 0.0).s_m, 10.0);
        assert!(once.progress_at(15.0, 0.0).arrived);
        assert_eq!(once.position_at(1e6, 0.0), once.destination());

        // A loop wraps and never turns round.
        assert_eq!(round.progress_at(15.0, 0.0).s_m, 5.0);
        assert!(round.progress_at(15.0, 0.0).forward);

        // A stand is ALWAYS arrived: reading it the other way would make every
        // idle NPC in a town permanently en route.
        assert!(
            CrowdRoute::standing(DVec3::ZERO)
                .progress_at(9.0, 0.0)
                .arrived
        );
    }

    /// **The re-phase is an inverse**: after it, the clock is where the body is,
    /// on the leg the clock was already on.
    #[test]
    fn a_rephase_puts_the_clock_exactly_where_the_body_is() {
        let path = inf_nav::NavPath::new([DVec3::ZERO, DVec3::new(10.0, 0.0, 0.0)]);
        for mode in [RouteMode::PingPong, RouteMode::Once, RouteMode::Loop] {
            let r = CrowdRoute::along(path.clone(), 1.0, mode);
            // Three clocks: one on each leg of a ping-pong, one well past both.
            for t in [4.0, 15.0, 33.0] {
                // Not the far END for a `Loop`: `rem_euclid(len)` folds it to
                // zero, which is the same PLACE and a different arc length, and
                // an arm that demanded 10 would be demanding a representation
                // rather than a position.
                let ends: &[f64] = match mode {
                    RouteMode::Loop => &[0.0, 2.5, 7.25],
                    _ => &[0.0, 2.5, 7.25, 10.0],
                };
                for &want in ends {
                    let travelled = r.travelled_at(t, 0.0);
                    let d = r.rephase_delta(travelled, want);
                    let after = r.progress_at(t, d);
                    assert!(
                        (after.s_m - want).abs() < 1e-9,
                        "{mode:?} at t = {t}: re-phasing to {want} landed on {}",
                        after.s_m
                    );
                }
            }
        }
        // A stand cannot be re-phased, and answers a zero rather than a NaN.
        assert_eq!(
            CrowdRoute::standing(DVec3::ZERO).rephase_delta(3.0, 1.0),
            0.0
        );
    }

    /// **A `Far` agent carries NO body, NO collider, NO controller and NO
    /// movement model** -- the NPC1b audit's finding 4, closed at the source.
    ///
    /// Its tripwire, `every_materialized_crowd_agent_observes_the_terrain_at_every_tier`,
    /// is the other half of this and lives in `inf-player` where the predicate
    /// does. Here is where the components go on and come off, and the transition
    /// is asserted **both ways**: a tier that gated only materialization would
    /// leave every agent that was ever close solid for ever, which is the shape
    /// the NPC1a audit's finding 7 found in the bridge.
    #[test]
    fn a_far_agent_carries_no_body_no_controller_and_no_movement_model() {
        let mut records = BTreeMap::new();
        records.insert(
            guid(0xD10),
            CrowdRecord::standing(
                CrowdArchetype::humanoid(None, None, None),
                DVec3::new(4.0, 0.0, 0.0),
            ),
        );
        let mut world = crowd_world(records);

        let has = |w: &EcsWorld| {
            let e = w.entity_of(guid(0xD10)).expect("materialized");
            let ww = w.world();
            (
                ww.get::<RigidBody3D>(e).is_some(),
                ww.get::<Collider3D>(e).is_some(),
                ww.get::<CharacterController3D>(e).is_some(),
                ww.get::<CharacterMovement>(e).is_some(),
                ww.get::<SkeletalMesh>(e).is_some(),
            )
        };

        move_source(&mut world, 4.0);
        step_crowd(&mut world, 1.0 / 60.0);
        assert_eq!(
            has(&world),
            (true, true, true, true, true),
            "a `Full` agent must carry the whole physical set"
        );

        // **The `Near` rung, which is where the ladder splits.** A near agent
        // keeps its body and its capsule -- it is something a player can walk
        // into -- and loses the controller and the movement model, because the
        // island's own N = 1 000 row priced 291 of those at 92.756 ms of
        // `character move`. See `CrowdTier::steers`.
        move_source(&mut world, 4.0 + DEFAULT_CROWD_FULL_M + 16.0);
        let s = step_crowd(&mut world, 1.0 / 60.0);
        assert_eq!(s.at(CrowdTier::Near), 1, "the fixture is not `Near`");
        assert_eq!(
            has(&world),
            (true, true, false, false, true),
            "a `Near` agent must keep its body and lose its controller"
        );

        move_source(&mut world, 4.0 + DEFAULT_CROWD_NEAR_M + 64.0);
        let s = step_crowd(&mut world, 1.0 / 60.0);
        assert_eq!(s.at(CrowdTier::Far), 1);
        assert_eq!(
            has(&world),
            (false, false, false, false, true),
            "a `Far` agent must carry NONE of the physical set -- and keep its \
             mesh, because the renderer still draws it"
        );

        // ...and back. The transition is the half a materialization-only gate
        // would miss.
        move_source(&mut world, 4.0);
        step_crowd(&mut world, 1.0 / 60.0);
        assert_eq!(
            has(&world),
            (true, true, true, true, true),
            "a promoted agent did not get its body back"
        );
    }

    /// **The near tiers are STEERED, not teleported.** The crowd writes an
    /// intent and the movement step owns the transform, which is the
    /// one-authority rule NPC1a refused a controller to keep.
    ///
    /// The falsifier is the pair: the intent points along the route AND the
    /// transform did not move to where the clock says. An implementation that
    /// wrote both would pass the first assertion and fail the second, and one
    /// that wrote neither would fail the first.
    #[test]
    fn a_near_agent_is_steered_rather_than_teleported() {
        let mut records = BTreeMap::new();
        records.insert(
            guid(0xD20),
            CrowdRecord::walking(
                CrowdArchetype::humanoid(None, None, None),
                CrowdRoute::between(DVec3::ZERO, DVec3::new(100.0, 0.0, 0.0), 1.4),
            ),
        );
        let mut world = crowd_world(records);
        move_source(&mut world, 0.0);

        let s = step_crowd(&mut world, 1.0 / 60.0);
        assert_eq!(s.at(CrowdTier::Full), 1);
        assert_eq!(s.steered, 1, "a `Full` agent was not steered");
        let e = world.entity_of(guid(0xD20)).expect("materialized");
        let placed = world
            .world()
            .get::<Transform>(e)
            .expect("a transform")
            .translation
            .to_dvec3();

        // The route runs +X, which at yaw 0 is the aim frame's own right --
        // through `rotate_into_frame`, so this is a claim about the door and not
        // about a coincidence of signs.
        let cm = world
            .world()
            .get::<CharacterMovement>(e)
            .expect("the movement model")
            .clone();
        assert!(
            cm.runtime.intent_move.x > 0.99,
            "the intent does not point down the route: {:?}",
            cm.runtime.intent_move
        );
        assert!(!cm.player_controlled, "a crowd agent is not the player");

        // Fifty steps later the CLOCK has moved metres and the transform has
        // not, because nothing in a unit world runs the mover.
        for _ in 0..50 {
            step_crowd(&mut world, 1.0 / 60.0);
        }
        let after = world
            .world()
            .get::<Transform>(e)
            .expect("a transform")
            .translation
            .to_dvec3();
        assert_eq!(
            placed, after,
            "the crowd step wrote a near agent's transform -- that is the second \
             authority this design exists without"
        );
        let pop = world
            .world()
            .get_resource::<CrowdPopulationRes>()
            .expect("the population");
        let rec = &pop.records[&guid(0xD20)];
        let clock = rec.position_at(guid(0xD20), CrowdClock::at((pop.steps - 1) as f64 / 60.0));
        assert!(
            (clock.x - after.x) > 0.5,
            "the route clock did not run on while the body stood still, so this \
             arm is not posing the problem"
        );
    }

    /// **A demotion hands the clock the metre the body reached** -- clause 4's
    /// continuity, measured as the size of the pop it removes.
    #[test]
    fn a_demotion_hands_the_clock_the_metre_the_body_reached() {
        let mut records = BTreeMap::new();
        records.insert(
            guid(0xD30),
            CrowdRecord::walking(
                CrowdArchetype::humanoid(None, None, None),
                CrowdRoute::between(DVec3::ZERO, DVec3::new(400.0, 0.0, 0.0), 4.0),
            ),
        );
        let mut world = crowd_world(records);
        move_source(&mut world, 0.0);
        step_crowd(&mut world, 1.0 / 60.0);
        let e = world.entity_of(guid(0xD30)).expect("materialized");

        // Sixty steps of clock while the body -- which nothing moves in a unit
        // world -- stands where it was placed. That is the worst case a blocked
        // agent produces, and it is exactly what the re-phase is for.
        for _ in 0..60 {
            step_crowd(&mut world, 1.0 / 60.0);
        }
        let body = world
            .world()
            .get::<Transform>(e)
            .expect("a transform")
            .translation
            .to_dvec3();
        let lag = {
            let pop = world
                .world()
                .get_resource::<CrowdPopulationRes>()
                .expect("the population");
            let rec = &pop.records[&guid(0xD30)];
            rec.position_at(guid(0xD30), CrowdClock::at(pop.steps as f64 / 60.0))
                .x
                - body.x
        };
        assert!(
            lag > 3.0,
            "the fixture has not built a lag to close: {lag:.3} m"
        );

        // Now demote it. Without the re-phase the transform would jump `lag`
        // metres this step.
        move_source(&mut world, DEFAULT_CROWD_NEAR_M + 64.0);
        let s = step_crowd(&mut world, 1.0 / 60.0);
        assert_eq!(s.at(CrowdTier::Far), 1);
        assert_eq!(s.rephased, 1, "the demotion did not re-phase the clock");
        let after = world
            .world()
            .get::<Transform>(e)
            .expect("a transform")
            .translation
            .to_dvec3();
        let jump = (after - body).length();
        assert!(
            jump < 0.2,
            "a demotion moved the body {jump:.3} m -- the clock was not handed \
             the metre the body reached (the lag it had built was {lag:.3} m)"
        );
        println!(
            "NPC1c demotion continuity: a body {lag:.2} m behind its own clock \
             moved {jump:.4} m across the `Near` -> `Far` transition"
        );
    }

    /// **A materialization moves nothing**: an agent's first entity is placed
    /// exactly where the tier below it was already being drawn.
    #[test]
    fn a_promotion_places_the_agent_where_the_clock_already_had_it() {
        let mut records = BTreeMap::new();
        records.insert(
            guid(0xD40),
            CrowdRecord::walking(
                CrowdArchetype::humanoid(None, None, None),
                CrowdRoute::between(
                    DVec3::new(900.0, 0.0, 0.0),
                    DVec3::new(1000.0, 0.0, 0.0),
                    1.4,
                ),
            ),
        );
        let mut world = crowd_world(records);
        // Far enough away to be Dormant: no entity at all.
        move_source(&mut world, 0.0);
        for _ in 0..30 {
            step_crowd(&mut world, 1.0 / 60.0);
        }
        let drawn = {
            let pop = world
                .world()
                .get_resource::<CrowdPopulationRes>()
                .expect("the population");
            assert_eq!(pop.records[&guid(0xD40)].tier, CrowdTier::Dormant);
            pop.records[&guid(0xD40)].last
        };

        move_source(&mut world, 900.0);
        let s = step_crowd(&mut world, 1.0 / 60.0);
        assert_eq!(s.spawned, 1);
        let e = world.entity_of(guid(0xD40)).expect("materialized");
        let placed = world
            .world()
            .get::<Transform>(e)
            .expect("a transform")
            .translation
            .to_dvec3();
        // One step of clock is 1.4 / 60 = 23 mm; the claim is that the promotion
        // itself moved nothing, not that time stopped.
        assert!(
            (placed - drawn).length() < 0.05,
            "a promotion teleported the agent {:.3} m from where the tier below \
             was drawing it",
            (placed - drawn).length()
        );
    }

    /// **The step materializes what the band wants and dematerializes what it
    /// does not** — and the walk back proves it is a function of *where the
    /// source is* rather than of what happened first.
    #[test]
    fn the_step_materializes_by_tier_and_dematerializes_by_tier() {
        let mut records = BTreeMap::new();
        // Four agents on a line: 10 m, 50 m, 200 m and 2 km from the origin.
        for (i, x) in [10.0f64, 50.0, 200.0, 2000.0].iter().enumerate() {
            records.insert(
                guid(0xA00 + i as u128),
                CrowdRecord::standing(CrowdArchetype::default(), DVec3::new(*x, 0.0, 0.0)),
            );
        }
        let mut world = crowd_world(records);

        let s = step_crowd(&mut world, 1.0 / 60.0);
        assert_eq!(
            (
                s.at(CrowdTier::Full),
                s.at(CrowdTier::Near),
                s.at(CrowdTier::Far),
                s.at(CrowdTier::Dormant)
            ),
            (1, 1, 1, 1),
            "the fixture is not posing the problem: {}",
            s.summary()
        );
        assert_eq!(s.spawned, 3, "three tiers materialize, one does not");
        assert_eq!(s.despawned, 0);
        assert!(world.entity_of(guid(0xA00)).is_some());
        assert!(
            world.entity_of(guid(0xA03)).is_none(),
            "a dormant agent has an entity"
        );

        // Walk the source out to the far agent: the near ones dematerialize and
        // the far one comes to life.
        move_source(&mut world, 2000.0);
        let s = step_crowd(&mut world, 1.0 / 60.0);
        assert_eq!(s.at(CrowdTier::Full), 1, "{}", s.summary());
        assert_eq!(s.at(CrowdTier::Dormant), 3, "{}", s.summary());
        assert_eq!(s.spawned, 1);
        assert_eq!(s.despawned, 3);
        assert!(world.entity_of(guid(0xA03)).is_some());
        assert!(world.entity_of(guid(0xA00)).is_none());

        // …and back. A record that dematerialized comes back where it stood,
        // which is what `last` is for.
        move_source(&mut world, 0.0);
        step_crowd(&mut world, 1.0 / 60.0);
        let e = world.entity_of(guid(0xA00)).expect("it came back");
        let t = world.world().get::<Transform>(e).expect("with a transform");
        assert_eq!(t.translation.to_dvec3(), DVec3::new(10.0, 0.0, 0.0));
    }

    /// **THE CROWD PHASE'S WORK IS A FUNCTION OF THE CROWD** (NPC1a audit).
    ///
    /// The cached pose digest is the one thing this phase reads outside its own
    /// population, and the first cut read **all** of it: a `BTreeMap` of every
    /// entry in [`crate::pose::PoseStoreRes`], folded joint by joint, on every
    /// step, to serve a demotion that happens to one agent every few hundred.
    /// So the phase's cost scaled with the *level's* posed characters rather
    /// than with the population, and the budget minted from it was mostly a
    /// measurement of other systems' poses.
    ///
    /// The wave's own sweep table said so and nobody read it that way: at
    /// N = 1 000 the phase charged **0.282 ms banded and 0.759 ms all-`Full`** —
    /// the same thousand agents doing identical work, differing only in how many
    /// characters were in the store. After the fix the two agree (0.103 and
    /// 0.109), which is what a per-agent phase has to look like.
    ///
    /// This arm is a **counter, not a clock**, so it holds on any machine — and
    /// it is worth being exact about which half of the defect it holds. It pins
    /// the **semantics**: a digest is taken on the TRANSITION out of a posing
    /// tier and not once per step per Far agent, and a step on which nobody
    /// leaves a posing tier reads the store zero times however full it is.
    /// It cannot see an implementation that computed a hundred digests and threw
    /// ninety-nine away, because that is a cost and not a behaviour; the arm for
    /// *that* is `crowd_sweep.rs`'s banded-vs-all-`Full` crowd-phase comparison,
    /// which is a wall clock and is asserted where this tree asserts wall clocks
    /// (release, off CI) and reported everywhere else.
    #[test]
    fn a_settled_crowd_folds_no_pose_digests_however_many_characters_pose() {
        use crate::pose::{EvaluatedPose, PoseStoreRes};
        use inf_anim::{JointTransform, Pose};

        let agent = guid(0xF00);
        let mut records = BTreeMap::new();
        records.insert(
            agent,
            CrowdRecord::standing(CrowdArchetype::default(), DVec3::new(10.0, 0.0, 0.0)),
        );
        let mut world = crowd_world(records);

        // A store with a hundred posed characters in it, one of which is the
        // agent. Fabricated rather than evaluated: this arm is about how many
        // of them the crowd phase touches, not about what a pose contains.
        let pose = Pose {
            locals: vec![JointTransform::IDENTITY; 32],
        };
        let mut store = BTreeMap::new();
        for i in 0..100u128 {
            store.insert(
                guid(0x9000 + i),
                EvaluatedPose {
                    skeleton: guid(0x77),
                    pose: pose.clone(),
                    sockets: Vec::new(),
                },
            );
        }
        store.insert(
            agent,
            EvaluatedPose {
                skeleton: guid(0x77),
                pose,
                sockets: Vec::new(),
            },
        );
        world.world_mut().insert_resource(PoseStoreRes(store));

        // Step 1: the agent classifies Full. Nothing leaves a posing tier.
        let s = step_crowd(&mut world, 1.0 / 60.0);
        assert_eq!(s.at(CrowdTier::Full), 1, "{}", s.summary());
        assert_eq!(
            s.digests_folded, 0,
            "a crowd that promoted nobody folded {} digest(s) out of a \
             101-entry store — the phase is walking the level's poses rather \
             than its own demotions",
            s.digests_folded
        );

        // Step 2: the source walks away and the agent demotes. Exactly one.
        move_source(&mut world, 400.0);
        let s = step_crowd(&mut world, 1.0 / 60.0);
        assert_eq!(s.at(CrowdTier::Far), 1, "{}", s.summary());
        assert_eq!(s.digests_folded, 1, "{}", s.summary());

        // Step 3: it is still Far, so there is nothing to fold — the store is
        // as full as it was and the phase does not look at it.
        let s = step_crowd(&mut world, 1.0 / 60.0);
        assert_eq!(
            s.digests_folded, 0,
            "an agent that was already Far folded another digest — the capture \
             is on the tier rather than on the TRANSITION"
        );

        // …and the digest it took is the pose that was published, not a zero.
        let pop = world
            .world()
            .get_resource::<CrowdPopulationRes>()
            .expect("the population");
        assert_ne!(
            pop.records[&agent].pose_digest, 0,
            "the demotion folded a digest and stored nothing"
        );
    }

    /// **A DORMANT AGENT IS STILL ON ITS ROUTE** (NPC1a audit).
    ///
    /// The module's headline is that *the position law does not vary with the
    /// tier* — an agent's place is `route(clock)` at `Full` exactly as at `Far`.
    /// `Dormant` was the exception, and it was the exception in the one
    /// direction that costs something: a dematerialized record froze at `last`,
    /// **the tier was then decided from that frozen point**, and a walking agent
    /// whose route carried it home could never be re-admitted — it was judged
    /// for ever at the metre where it went out of range, while the *next*
    /// materialization would have placed it at `route(now)` somewhere else
    /// entirely. Frozen for the decision and live for the placement is the two
    /// authorities this module exists to avoid.
    ///
    /// A route is a pure function of the clock and costs a handful of flops, so
    /// keeping it live while an agent has no entity costs nothing and is the
    /// only reading under which `Dormant` is a *cost* tier rather than a
    /// one-way door.
    #[test]
    fn a_dormant_agent_keeps_walking_its_route_and_can_come_back() {
        let mut records = BTreeMap::new();
        // Out to 2 km and back, fast enough that the round trip fits in the step
        // budget of a unit test — and starting at 150 m, **outside the `Near`
        // ring**, which is NPC1c's constraint on this fixture rather than a
        // detail. From this wave a `Full`/`Near` agent's transform belongs to
        // `step_character_movement`, five phases later in a real host's fixed
        // step; there is no such step in a unit world, so an agent that came
        // within 96 m of the anchor here would simply stand. The tiers this arm
        // is about — `Far` and `Dormant` — are the clock's, and it stays in
        // them. `a_near_agent_is_steered_rather_than_teleported` is the arm for
        // the other half.
        records.insert(
            guid(0xE00),
            CrowdRecord::walking(
                CrowdArchetype::default(),
                CrowdRoute::between(
                    DVec3::new(150.0, 0.0, 0.0),
                    DVec3::new(2000.0, 0.0, 0.0),
                    1000.0,
                ),
            ),
        );
        // The anchor never moves: everything below is the AGENT walking.
        let mut world = crowd_world(records);

        let mut went_dormant = None;
        let mut came_back = None;
        for step in 0..400u64 {
            let s = step_crowd(&mut world, 1.0 / 60.0);
            let dormant = s.at(CrowdTier::Dormant) == 1;
            if dormant && went_dormant.is_none() {
                went_dormant = Some(step);
            }
            if went_dormant.is_some() && !dormant && came_back.is_none() {
                came_back = Some(step);
            }
        }
        let out = went_dormant.expect(
            "the agent never left the band, so this arm is not posing the problem — \
             the route must reach past DEFAULT_CROWD_FAR_M",
        );
        let back = came_back.unwrap_or_else(|| {
            panic!(
                "the agent went Dormant at step {out} and never came back over 400 \
                 steps of a route that returns to its anchor — a dematerialized \
                 agent is being tiered from where it froze rather than from where \
                 its route says it is"
            )
        });
        println!(
            "NPC1a dormancy: an agent walking a 2 km route went Dormant at step \
             {out} and re-materialized at step {back}"
        );

        // …and the record's remembered place tracked the route the whole way,
        // which is what makes the tier above a decision about the present.
        let pop = world
            .world()
            .get_resource::<CrowdPopulationRes>()
            .expect("the population");
        let rec = &pop.records[&guid(0xE00)];
        let want = rec.position_at(
            guid(0xE00),
            CrowdClock::at((pop.steps - 1) as f64 * (1.0 / 60.0)),
        );
        assert_eq!(
            rec.last, want,
            "a record's `last` diverged from its own route — the trace's 24 \
             position bytes and the tier decision are reading different places"
        );
    }

    /// **A Far agent gets no body and no pose, and the trace says which.**
    ///
    /// The anti-vacuity half matters more than the assertion: an arm that only
    /// checked "the Far agent has no rapier body" would pass on a world where
    /// nothing has one.
    #[test]
    fn the_far_tier_drops_the_body_and_the_pose_and_the_trace_records_it() {
        let mut records = BTreeMap::new();
        records.insert(
            guid(0xB00),
            CrowdRecord::standing(CrowdArchetype::default(), DVec3::new(10.0, 0.0, 0.0)),
        );
        records.insert(
            guid(0xB01),
            CrowdRecord::standing(CrowdArchetype::default(), DVec3::new(200.0, 0.0, 0.0)),
        );
        let mut world = crowd_world(records);
        step_crowd(&mut world, 1.0 / 60.0);

        let solid = crowd_colliders(&world);
        assert_eq!(
            solid.len(),
            1,
            "exactly the near agent keeps its body — {solid:?}"
        );
        assert!(solid.contains(&guid(0xB00)));
        assert!(
            !solid.contains(&guid(0xB01)),
            "the far agent kept its body, so the tier is doing nothing"
        );

        // The published verdict is on the entity, which is what the pose door
        // and the physics bridge read.
        let near = world.entity_of(guid(0xB00)).expect("near");
        let far = world.entity_of(guid(0xB01)).expect("far");
        assert_eq!(agent_tier(&world, near), Some(CrowdTier::Full));
        assert_eq!(agent_tier(&world, far), Some(CrowdTier::Far));
        assert!(
            agent_tier(&world, world.entity_of(guid(0xF1)).unwrap()).is_none(),
            "the streaming source is not a crowd agent and must have no tier"
        );

        // The trace carries the tier byte at the agent's own offset.
        let bytes = crowd_state_bytes(&world);
        assert_eq!(bytes.len(), 2 * AGENT_TRACE_BYTES);
        assert_eq!(bytes[16], CrowdTier::Full.as_u8());
        assert_eq!(bytes[AGENT_TRACE_BYTES + 16], CrowdTier::Far.as_u8());
    }

    /// **The whole step is a pure function of sim state**: two worlds built the
    /// same way and stepped the same way produce the same bytes, and a world
    /// whose source moved produces different ones.
    #[test]
    fn two_identical_worlds_produce_identical_crowd_traces() {
        let build = || {
            let mut records = BTreeMap::new();
            for i in 0..12u128 {
                records.insert(
                    guid(0xC00 + i),
                    CrowdRecord::walking(
                        CrowdArchetype::default(),
                        CrowdRoute::between(
                            DVec3::new(i as f64 * 20.0, 0.0, 0.0),
                            DVec3::new(i as f64 * 20.0 + 40.0, 0.0, 0.0),
                            1.4,
                        ),
                    ),
                );
            }
            crowd_world(records)
        };
        let (mut a, mut b) = (build(), build());
        let mut trace_a = Vec::new();
        let mut trace_b = Vec::new();
        for step in 0..90u64 {
            move_source(&mut a, step as f64 * 2.0);
            move_source(&mut b, step as f64 * 2.0);
            step_crowd(&mut a, 1.0 / 60.0);
            step_crowd(&mut b, 1.0 / 60.0);
            trace_a.push(crowd_state_bytes(&a));
            trace_b.push(crowd_state_bytes(&b));
        }
        assert_eq!(trace_a, trace_b, "two identical runs diverged");
        // …and the trace is not a constant, or the comparison is between two
        // recordings of nothing happening.
        let distinct: BTreeSet<&Vec<u8>> = trace_a.iter().collect();
        assert!(
            distinct.len() > 45,
            "only {} of 90 crowd states differ — the agents are not moving",
            distinct.len()
        );
        println!(
            "NPC1a determinism: 90 steps, {} distinct crowd states, {} B a state",
            distinct.len(),
            trace_a[0].len()
        );
    }

    /// **THE ONE FLEE DOOR, and the re-phase trap it exists to hold** (wave
    /// WPN1).
    ///
    /// Four claims:
    ///
    /// 1. a **dormant** agent — no entity at all — can be frightened, because
    ///    that is exactly the agent a shot at the edge of a panic radius
    ///    reaches, and a door that needed a body would have been silently
    ///    inert on every one of them;
    /// 2. the route really starts **now**: without the re-phase a fresh `Once`
    ///    route handed to a population that has already been running reads as
    ///    finished before the agent has taken a step, which puts its body at the
    ///    route's far end the moment it drops off the steered tier;
    /// 3. a **scheduled** agent's day is released, so its slot posture stops
    ///    holding it;
    /// 4. the **latch** holds: a second call does nothing.
    #[test]
    fn the_flee_door_starts_the_route_now_and_latches() {
        let dt = 1.0 / 60.0;
        let a = CrowdArchetype::humanoid(None, None, None);
        let mut w = EcsWorld::new();
        let who = guid(0xF1EE);
        // A record with a schedule, standing at the origin — and NO entity: the
        // dormant case, which is the one a component latch could not see.
        let legs = vec![ScheduleLeg {
            start_h: 8.0,
            travel_h: 1.0,
            path: inf_nav::NavPath::new([DVec3::ZERO, DVec3::new(10.0, 0.0, 0.0)]),
            arrival: SlotArrival {
                role: Some(crate::components::SlotRole::Work),
                posture: crate::components::SlotPosture::Sit,
                face: DVec3::Z,
            },
        }];
        let rec = CrowdRecord::scheduled(a, CrowdSchedule::new(legs).expect("a day"));
        set_population(&mut w, BTreeMap::from([(who, rec)]));
        assert!(w.entity_of(who).is_none(), "the fixture materialized it");
        // Run the population's clock on, so the session time is not zero — which
        // is the whole reason the re-phase exists.
        {
            let mut pop = w
                .world_mut()
                .remove_resource::<CrowdPopulationRes>()
                .expect("a population");
            pop.steps = 3600;
            w.world_mut().insert_resource(pop);
        }
        let here = DVec3::new(2.0, 0.0, 0.0);
        assert!(!is_panicked(&w, who));
        assert!(
            flee_from(&mut w, who, here, DVec3::ZERO, dt, 60.0),
            "a dormant agent could not be frightened"
        );
        assert!(is_panicked(&w, who));

        let pop = w
            .world()
            .get_resource::<CrowdPopulationRes>()
            .expect("a population");
        let rec = &pop.records[&who];
        assert!(
            rec.schedule.is_none(),
            "the agent kept its day, so its slot posture still holds it at a \
             desk it is supposed to be running from"
        );
        assert_eq!(rec.route.mode, RouteMode::Once);
        // It runs AWAY: the destination is further from the source than the
        // start was.
        let dest = rec.route.destination();
        println!(
            "fleeing from {:?}: {here:?} -> {dest:?} ({:.1} m)",
            DVec3::ZERO,
            (dest - here).length()
        );
        assert!(dest.length() > here.length() + 50.0, "{dest:?}");
        // **THE RE-PHASE.** `t_s` is 3600 steps of session time, so an
        // un-rephased route is finished before it starts. Measured as the
        // progress the clock answers on the step it was handed the route.
        let clock = CrowdClock::at(pop.steps as f64 * dt);
        let p = rec.progress_at(who, clock);
        println!(
            "the clock says the agent is {:.3} m along a {:.1} m route ({} steps of session time)",
            p.s_m,
            rec.route.length_m(),
            pop.steps
        );
        assert!(
            p.s_m.abs() < 1.0,
            "the route is {:.1} m in on the step it was handed over — the \
             re-phase is missing and the body will be written at the far end",
            p.s_m
        );
        // …and the control: the SAME record without the re-phase really is
        // finished, so the arm above is measuring something.
        let mut naive = rec.clone();
        naive.rephase_m = 0.0;
        let q = naive.progress_at(who, clock);
        println!("without the re-phase the same clock says {:.1} m", q.s_m);
        assert!(
            q.s_m >= naive.route.length_m(),
            "the un-rephased control is not finished, so the trap this arm \
             names cannot happen and the arm proves nothing"
        );

        // **THE LATCH.**
        assert!(
            !flee_from(&mut w, who, here, DVec3::ZERO, dt, 60.0),
            "a second call re-routed somebody who is already running"
        );
        clear_panic(&mut w);
        assert!(!is_panicked(&w, who));
    }

    /// **Clearing the crowd leaves a world that never had one.**
    #[test]
    fn clearing_the_crowd_despawns_every_agent_and_removes_the_resource() {
        let mut records = BTreeMap::new();
        for i in 0..5u128 {
            records.insert(
                guid(0xD00 + i),
                CrowdRecord::standing(CrowdArchetype::default(), DVec3::new(i as f64, 0.0, 0.0)),
            );
        }
        let mut world = crowd_world(records);
        step_crowd(&mut world, 1.0 / 60.0);
        assert_eq!(crowd_stats(&world).at(CrowdTier::Full), 5);

        clear_crowd(&mut world);
        for i in 0..5u128 {
            assert!(
                world.entity_of(guid(0xD00 + i)).is_none(),
                "agent {i} survived"
            );
        }
        assert!(crowd_state_bytes(&world).is_empty());
        assert_eq!(crowd_stats(&world), CrowdStats::default());
        // Idempotent, and a no-op on a world that never had a population.
        clear_crowd(&mut world);
        clear_crowd(&mut EcsWorld::new());
    }

    /// **INSTALLING A POPULATION TAKES THE OLD ONE'S BODIES WITH IT** (NPC1a
    /// audit).
    ///
    /// [`set_population`] says it replaces the population a world already had,
    /// and a population is not only its records: a materialized agent is a real
    /// entity carrying a skeletal mesh, a machine, a capsule and a
    /// [`CrowdAgent`]. Dropping the resource alone left those standing, with a
    /// tier frozen at whatever the last step decided and **no record behind
    /// them** — so [`crowd_colliders`] (which reads records) and the
    /// [`CrowdAgent`] component (which the pose door and the deform pass read)
    /// would answer differently about the same entity, which is the two-opinions
    /// defect this wave's own deform finding is about.
    #[test]
    fn installing_a_second_population_does_not_leave_the_first_standing() {
        let mut first = BTreeMap::new();
        for i in 0..4u128 {
            first.insert(
                guid(0xE10 + i),
                CrowdRecord::standing(CrowdArchetype::default(), DVec3::new(i as f64, 0.0, 0.0)),
            );
        }
        let mut world = crowd_world(first);
        step_crowd(&mut world, 1.0 / 60.0);
        assert_eq!(crowd_stats(&world).at(CrowdTier::Full), 4);

        let mut second = BTreeMap::new();
        second.insert(
            guid(0xE20),
            CrowdRecord::standing(CrowdArchetype::default(), DVec3::ZERO),
        );
        set_population(&mut world, second);
        step_crowd(&mut world, 1.0 / 60.0);

        for i in 0..4u128 {
            assert!(
                world.entity_of(guid(0xE10 + i)).is_none(),
                "agent {i} of the replaced population is still standing in the \
                 world with no record behind it — the crowd door and the tier \
                 component now disagree about whether it exists"
            );
        }
        assert_eq!(crowd_state_bytes(&world).len(), AGENT_TRACE_BYTES);
        assert_eq!(crowd_stats(&world).total(), 1);
    }

    /// **A world with no population pays nothing** — the anti-vacuity control
    /// for every arm above, and the structural half of "zero cost when absent".
    #[test]
    fn a_world_with_no_population_steps_to_the_zero_stats() {
        let mut world = EcsWorld::new();
        let s = step_crowd(&mut world, 1.0 / 60.0);
        assert_eq!(s, CrowdStats::default());
        assert!(crowd_colliders(&world).is_empty());
        assert!(blocked_agents(&world).is_empty());
        assert!(crowd_state_bytes(&world).is_empty());
        assert!(
            !world.world().contains_resource::<CrowdPopulationRes>(),
            "a step over a world with no crowd installed one"
        );
    }

    /// **ISLAND WAVE NPC1e: resolving the leg once answers what resolving it
    /// five times answered — to the BIT.**
    ///
    /// The step used to ask `CrowdRecord::leg_at` four times an agent and a
    /// steered one five (see [`ActiveLeg`] for the measurement). The `*_on`
    /// doors take the answer instead of re-deriving it, and the whole
    /// correctness claim is that they are the same functions: this walks a
    /// scheduled record round its own day, at the hour boundaries and between
    /// them, and compares `to_bits()` rather than an epsilon, because "the same
    /// arithmetic in the same order" is what is being claimed and a tolerance
    /// would admit a different one.
    ///
    /// # What that half is worth, honestly (the NPC1e audit)
    ///
    /// Each `*_at` door is now **implemented as** `*_on(self.leg_at(..))`, so
    /// the equality sweep below is a **tautology up to `leg_at` being a pure
    /// function of its arguments** — it fails only if somebody re-writes an
    /// `*_at` door to stop delegating. It is kept for exactly that, and the
    /// claim the refactor really makes is the one under it: *the leg cannot
    /// change inside a step*, so the answer resolved at the top of
    /// [`step_crowd_banded`] is still the right answer at the fifth site that
    /// reads it.
    ///
    /// That is falsifiable, and the second half of this arm holds it: the step
    /// writes exactly two record fields between the resolution and the last use
    /// — `leg` (the leg-change check) and `rephase_m` (the demotion re-phase) —
    /// and `leg_at` must not read either. A `leg_at` that consulted `self.leg`
    /// as a hint, or that folded `rephase_m` into the hour, would hand every
    /// downstream `*_on` call a stale answer the pre-wave code could never have
    /// produced, and nothing else in this tree would notice.
    #[test]
    fn resolving_the_leg_once_answers_what_resolving_it_every_time_answered() {
        let guid = Uuid::from_u128(0x1e9);
        let mut rec = CrowdRecord::scheduled(CrowdArchetype::humanoid(None, None, None), day());
        // A body that fell behind its clock, so `rephase_m` is not zero and the
        // two doors have something to disagree about.
        rec.rephase_m = -3.75;
        let mut checked = 0_u32;
        for step in 0..=240 {
            let hour = f64::from(step) * 0.1;
            let clock = CrowdClock {
                t_s: f64::from(step) * 1.7,
                hour,
            };
            let leg = rec.leg_at(guid, clock);
            assert_eq!(
                rec.progress_on(leg, guid, clock),
                rec.progress_at(guid, clock),
                "progress parted company at {hour} h"
            );
            let a = rec.position_on(leg, guid, clock);
            let b = rec.position_at(guid, clock);
            assert_eq!(
                (a.x.to_bits(), a.y.to_bits(), a.z.to_bits()),
                (b.x.to_bits(), b.y.to_bits(), b.z.to_bits()),
                "position parted company at {hour} h: {a:?} against {b:?}"
            );
            assert!(
                std::ptr::eq(rec.path_on(leg), rec.path_at(guid, clock)),
                "the two doors named different paths at {hour} h"
            );
            for s_m in [0.0, 12.5, 99.0] {
                assert_eq!(
                    rec.rephase_delta_on(leg, guid, clock, s_m).to_bits(),
                    rec.rephase_delta_at(guid, clock, s_m).to_bits(),
                    "the re-phase parted company at {hour} h, s = {s_m}"
                );
            }
            checked += 1;
        }
        // ANTI-VACUITY: the walk really crossed legs rather than sitting on one.
        let legs: std::collections::BTreeSet<usize> = (0..=240)
            .filter_map(|step| {
                rec.leg_at(
                    guid,
                    CrowdClock {
                        t_s: 0.0,
                        hour: f64::from(step) * 0.1,
                    },
                )
                .map(|(i, _)| i)
            })
            .collect();
        assert_eq!(
            legs.len(),
            4,
            "the sweep only ever saw legs {legs:?}, so it compared one leg 241 times"
        );
        assert_eq!(checked, 241);

        // **THE HALF THE DELEGATION CANNOT SHOW** (the NPC1e audit): the leg
        // does not move under the two fields the step writes while it is being
        // used. `step_crowd_banded` resolves it once, then writes `rec.leg` in
        // the leg-change check and `rec.rephase_m` in the demotion re-phase, and
        // hands the SAME resolved value to the plan, the re-phase and the
        // pursuit after both writes. A `leg_at` that read either — as a hint, or
        // folded into the hour — would make the handed-down answer stale in a
        // way the pre-wave code could not produce, and no other arm in this tree
        // would see it.
        for hour in [0.0_f64, 5.5, 9.25, 13.0, 18.75, 23.9] {
            let clock = CrowdClock { t_s: 41.0, hour };
            let before = rec.leg_at(guid, clock);
            let mut moved = rec.clone();
            moved.leg = moved.leg.wrapping_add(3);
            moved.rephase_m += 137.5;
            moved.tier = CrowdTier::Full;
            moved.last = DVec3::new(11.0, 2.0, -7.0);
            assert_eq!(
                moved.leg_at(guid, clock),
                before,
                "the leg moved under the step's own writes at {hour} h, so a leg \
                 resolved at the top of a step is not the leg its fifth reader \
                 needs"
            );
        }

        // And an UNSCHEDULED record, where the leg is `None` and the `*_on`
        // doors must fall through to the route exactly as the `*_at` ones do.
        let ping = CrowdRecord::walking(
            CrowdArchetype::humanoid(None, None, None),
            CrowdRoute::between(DVec3::ZERO, DVec3::new(0.0, 0.0, 6.0), 1.4),
        );
        for step in 0..64 {
            let clock = CrowdClock::at(f64::from(step) * 0.37);
            let leg = ping.leg_at(guid, clock);
            assert!(leg.is_none(), "an unscheduled record produced a leg");
            let a = ping.position_on(leg, guid, clock);
            let b = ping.position_at(guid, clock);
            assert_eq!(
                (a.x.to_bits(), a.y.to_bits(), a.z.to_bits()),
                (b.x.to_bits(), b.y.to_bits(), b.z.to_bits())
            );
        }
    }
}
