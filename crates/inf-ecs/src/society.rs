//! **The society** (island wave NPC1d): a level's own buildings, turned into
//! people with days.
//!
//! `inf_pcg` decides who a *building* holds — bedrooms are homes, offices and
//! shops are workplaces, a shop is also somewhere to go (`inf_pcg::building::
//! society`). This module is the half that needs the whole **level**: it pairs a
//! home with a workplace, lays a network a body can walk between them, plans the
//! four legs of a day over it, and installs the result through
//! [`crate::crowd::add_agents`].
//!
//! ```text
//!   PcgVolume::residents ─┐
//!   PcgVolume::interior_nav ─┼─▶ SocietyRes { network, places } ─▶ CrowdSchedule
//!   PcgVolume centre+extent ─┘                                        │
//!                                                                     ▼
//!                                                      CrowdPopulationRes
//! ```
//!
//! # The street network is the PAVEMENT round the blocks
//!
//! A settlement's street plan lives in Ring 1, in the recipe, and is not in a
//! cooked pack — so a shipped player has no grid to read. What every host *does*
//! have is the blocks themselves: a `PcgVolume` is a centre and an axis-aligned
//! half-extent, and the streets are the gaps between them. So each volume lays a
//! **ring of eight nodes** [`PAVEMENT_M`] outside its own rectangle — four
//! corners and four edge midpoints — and two rings within [`BLOCK_LINK_MAX_M`]
//! of each other are joined at their nearest pair. That is a street crossing,
//! and it is derived from the level's own contents rather than from a plan the
//! level does not carry.
//!
//! It also closes NPC1c's defect 5 by construction. A settlement's grid runs out
//! to its whole reservation radius while the *levelled pad* is smaller, so the
//! outer lines lie on raw hillside and a route down one walks a body into the cut
//! face. A pavement hugs its own block, and a block stands on the pad — so every
//! node of this network is on ground a body can walk, without a ground profile
//! and without a terrain query.
//!
//! # THE SEARCH IS TWO LEVELS, and that is a measurement rather than a taste
//!
//! The obvious network is one graph with every building's whole interior in it.
//! It was built that way first and priced: a settlement's own buildings are
//! about **25 000 nodes**, and `inf-nav`'s own measured 743 µs over a 1 600-node
//! grid extrapolates to roughly **11 ms a search** — four legs an agent, four
//! hundred agents, inside a fixed step. That is not a slow gate, it is a
//! simulation that stops.
//!
//! So the level network holds the **streets and the front doors** — about
//! 1 600 nodes, which is exactly the size `inf-nav` measured — and a building's
//! interior is searched *inside the building*, over the hundred-odd nodes it has
//! of its own. A leg is `home → front door` (inside), `front door → front door`
//! (outside), `front door → work` (inside), joined. The outer half is memoized
//! on its endpoint pair, because a hundred residents of one block commuting to
//! one office is a hundred agents sharing one street route.
//!
//! # A building joins at its own front door
//!
//! Each volume's [`PcgVolume::interior_nav`] is already salted per building
//! (`inf_pcg::building::society::building_salt`), so absorbing them all is a
//! union rather than a collision. A building's **exterior** door is the one
//! doorway node with a single edge — a leaf, because the wall it stands in has no
//! room on the far side — and it is linked to the nearest node of its own block's
//! ring. So a route reads *room → doorway → pavement → pavement → doorway →
//! room*, which is the building↔street↔building crossing this wave exists to
//! make possible.
//!
//! # Nobody plans a day while the town is still being built
//!
//! Volumes stream. A resident of the first block to arrive would be paired with
//! the only workplace it could see, which would make a level's society a function
//! of the order its cells activated. So [`sync_society`] plans **only on a step
//! that folded no new volume**, and plans at most [`SOCIETY_PLANS_PER_STEP`]
//! agents on it — a bounded spike rather than a load-time cliff. Both hosts
//! stream identically, so both derive the same society; what is carried honestly
//! is that an agent plans **once**, and a workplace that arrives afterwards does
//! not re-open its day.
//!
//! # Everything here is derived
//!
//! `SocietyRes` is a bevy resource, exactly as `CrowdPopulationRes` and
//! `DeformFieldRes` are, so **no schema moves**. The slots it reads are
//! `#[serde(skip)]` on the volume. An agent's `Guid` is a hash of the level's own
//! content, so two hosts mint the same one without talking.
//!
//! # Portable math
//!
//! Distances are `sqrt` of sums of products, a quantization is a `floor`, and a
//! pairing is a comparison — the P14 ban list binds this module because a slot's
//! metres land on an NPC's `Transform` and therefore in the replay trace.

use std::collections::{BTreeMap, BTreeSet};

use bevy_ecs::prelude::Resource;
use glam::{DVec2, DVec3};
use inf_nav::{NavGraph, NavKind, NavNodeId};
use uuid::Uuid;

use crate::components::{GlobalTransform, Guid, PcgVolume, SlotRole, SlotShift};
use crate::crowd::{CrowdArchetype, CrowdRecord, CrowdSchedule, ScheduleLeg};
use crate::world::EcsWorld;

/// How far outside a block's own rectangle its pavement ring runs, metres.
///
/// Two metres is a pavement: far enough out that the ring is not inside the
/// building line, close enough that a crossing to the next block is the width of
/// the street and not a diagonal across it. The settlement generator reserves
/// about eight metres between blocks, so two rings sit about four metres apart.
pub const PAVEMENT_M: f64 = 2.0;

/// The furthest two blocks' pavements are joined across the street, metres.
///
/// Sized against the settlement grid it is derived from rather than for looks:
/// blocks on one grid are a street reserve apart (about 8 m) and the *next* block
/// along is a whole pitch away (about 120 m), so forty metres joins neighbours
/// and never reaches past one. Two settlements are kilometres apart and stay
/// separate components, which is correct — they are separate towns.
pub const BLOCK_LINK_MAX_M: f64 = 40.0;

/// The furthest a front door is joined to its own block's pavement, metres.
///
/// A door that is further than this from its own block's ring is a door on a
/// building that is not on that block, and linking it would cut a route through
/// whatever stands between. Refused rather than stretched; the count is in
/// [`SocietyStats::frontages_refused`].
pub const FRONTAGE_MAX_M: f64 = 40.0;

/// The lattice a pavement node's id is quantized onto, metres.
///
/// One centimetre. Two blocks' rings are laid from their own centres and
/// half-extents, so a shared corner is the same point *arithmetically* — the
/// quantization is what makes it the same point after the arithmetic, and it
/// is deliberately far finer than any distance that means anything here.
pub const PAVEMENT_LATTICE_M: f64 = 0.01;

/// How many agents' days [`sync_society`] plans on one fixed step.
///
/// A day is four Dijkstras over the level's network, so a settlement of four
/// hundred residents is sixteen hundred searches. Doing them all on the step the
/// last block arrived would be a load-time cliff inside a *fixed* step, which is
/// the shape wave I4b's budgets exist to refuse. Eight a step fills a settlement
/// in about a second of sim and is measured rather than assumed — see the wave
/// ledger's planning row.
pub const SOCIETY_PLANS_PER_STEP: usize = 8;

/// **The most people one level's society will ever be.**
///
/// A settlement's own buildings imply far more residents than a fixed step can
/// carry. Measured on the CI island's Harbour City: **four blocks alone offer
/// 329 homes**, so its hundred and seventy imply something like fourteen
/// thousand — a real number for a real town and about fourteen times the
/// population this arc has ever measured.
///
/// A thousand, and the number is chosen so the island's society is directly
/// comparable to the arc's own ledger: N = 1 000 is what NPC1a's sweep, NPC1b's
/// crowd row and NPC1c's re-measurement are all quoted at. A level that wants
/// all fourteen thousand needs the cheaper character mover the NPC1c audit
/// routed, and until it exists a ceiling is more honest than a fixed step that
/// silently takes four hundred milliseconds.
///
/// Homes past it are **declined, in `Guid` order**, and counted in
/// [`SocietyStats::homes_declined`] — so which thousand a level gets is a
/// function of its own content and not of the order its cells activated.
pub const SOCIETY_MAX_AGENTS: usize = 1_000;

/// The hour the working day begins, and the hours the other three legs do.
///
/// One table, in one place, so "the town populates at morning and empties at
/// night" is a statement about these six numbers. The commute is an hour and the
/// errand half of one, which at the island's authored rate is a walking pace
/// over a settlement-sized route (`ScheduleLeg::implied_speed_mps`).
pub const WORK_START_H: f64 = 8.0;
/// How long a commute leg takes, in hours of the level clock.
pub const COMMUTE_H: f64 = 1.0;
/// The hour the errand out of work begins.
pub const ERRAND_OUT_H: f64 = 12.0;
/// How long an errand leg takes, in hours of the level clock.
pub const ERRAND_H: f64 = 0.5;
/// The hour the walk back to work begins.
pub const ERRAND_BACK_H: f64 = 13.0;
/// The hour the walk home begins.
pub const HOME_H: f64 = 18.0;

// ── the night (wave VEN1b) ──────────────────────────────────────────────────
//
// Before this wave the table above WAS the day, and the day ended: every agent
// walked home at eighteen hundred and stood there until eight the next morning.
// A town with a nightclub in it that empties before the nightclub opens is the
// thing this block exists to stop.

/// The hour a reveller leaves for a venue.
///
/// Eight in the evening: an hour after the last commute home lands, so a body
/// is at its own front door before it turns round, and late enough that the
/// island's own night-glow ramp is at full by the time it arrives.
pub const EVENING_OUT_H: f64 = 20.0;

/// How long the walk to a venue takes, in hours of the level clock.
///
/// The same hour a commute takes, because it is the same walk across the same
/// settlement: the venues sit one ring out of a city's office core, which is
/// exactly the distance a resident already walks to work.
pub const EVENING_H: f64 = 1.0;

/// The hour a reveller starts walking home.
///
/// Two in the morning, which is the other end of the mandate's own
/// **18:00–02:00** window. It is past midnight on purpose: `CrowdSchedule::at`
/// picks the leg that started most recently as `(hour - start_h) mod 24`, so a
/// day is a circle and a leg that begins after midnight is not a special case —
/// it is the arm `a_night_out_wraps_past_midnight` exists to prove.
pub const NIGHT_HOME_H: f64 = 2.0;

/// The hour a night-shift worker leaves for the venue they work at.
///
/// Six in the evening. A bar's keeper is not an office worker who happens to
/// work in a bar: they are behind the counter before the first patron arrives
/// at nine, and asleep while the town commutes.
pub const NIGHT_WORK_START_H: f64 = 18.0;

/// The hour a night-shift worker starts walking home — an hour after the last
/// patron leaves.
pub const NIGHT_WORK_END_H: f64 = 3.0;

/// **What share of a town goes out on a given night.**
///
/// A third, drawn per agent from [`SALT_NIGHTLIFE`] at `tick = 0`, on
/// `CrowdRecord::speed_of`'s own terms: derived and never stored, so it is a
/// constant of the agent rather than of the step.
///
/// The number is deliberately larger than the venues can hold, and that is the
/// design rather than a slip. **The venue's own seat count is the cap** — a
/// leisure slot is CLAIMED, so a settlement with one bar seats eleven people
/// whatever this is, and the share only decides *which* eleven of the willing
/// get there first. Making the share the cap instead would have put the
/// occupancy of a club in a constant rather than in the club.
pub const NIGHTLIFE_SHARE: f64 = 0.34;

/// Salts the per-agent nightlife draw. See [`NIGHTLIFE_SHARE`].
pub const SALT_NIGHTLIFE: u64 = 0x4e49_4748_5400_0001;

/// How far an agent will walk to a night job, metres in plan.
///
/// A night job is scarce — three per venue — and is claimed by the nearest
/// unclaimed home in `Guid` order, so without a reach the first agent planned
/// would take a job on the far side of the island. Six hundred metres is about
/// twice a settlement's own block span, which keeps a bar staffed by people who
/// live in the same town as the bar.
pub const NIGHT_JOB_MAX_M: f64 = 600.0;

/// How far an agent will walk for a night out, metres in plan.
///
/// Wider than [`NIGHT_JOB_MAX_M`], because going out is a choice and going to
/// work is a routine: a kilometre is the whole of one of the island's cities,
/// and a resident who would cross it for a nightclub is a resident who wants a
/// nightclub. Two settlements are kilometres apart and stay separate towns,
/// which the network already enforces — an unroutable pair is refused by the
/// leg search whatever this says.
pub const NIGHT_OUT_MAX_M: f64 = 1_000.0;

/// Salts an agent's derived `Guid`. See [`agent_guid`].
const SALT_AGENT: u64 = 0x4147_454E_5400_0001;

/// The tag every derived agent `Guid` carries in its top sixteen bits — `"NP"`.
///
/// Not a namespace guarantee and not pretended to be one: it is there so a guid
/// in a trace or a log is recognizable as a crowd agent's rather than a level
/// entity's. The guarantee that an agent never overwrites a level entity is
/// `crate::crowd::add_agents`' own refusal, which asks the world.
const AGENT_TAG: u128 = 0x4E50;

/// **A block's pavement ring**: where the block is, how big it is, and the
/// eight nodes laid round it.
///
/// Named because it is three things a reader has to keep together and because
/// `clippy::type_complexity` is right that a bare tuple in a public map is a
/// puzzle.
pub type BlockRing = (DVec3, DVec2, Vec<(NavNodeId, DVec3)>);

/// **One place a level offers**, with the node a route reaches it by.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SocietyPlace {
    /// What it is for.
    pub role: SlotRole,
    /// **What a body does when it gets here** (wave VEN1b), carried onto the
    /// leg that ends here so the crowd step and the pose step can read it
    /// without asking the society a second time.
    pub posture: crate::components::SlotPosture,
    /// **Which way the body faces here** (wave VEN1b), a unit XZ direction, or
    /// `DVec3::ZERO` for a place with no opinion.
    pub face: DVec3,
    /// Where it is, world metres.
    pub at: DVec3,
    /// The node of its own building's interior it stands on — a node of
    /// [`SocietyRes::interiors`]`[volume]`, **not** of the level network.
    pub node: NavNodeId,
    /// The volume whose interior graph holds [`node`](Self::node).
    pub volume: Uuid,
    /// Its building's front door, which IS a node of the level network — the
    /// join between the two levels of the search.
    pub door: NavNodeId,
}

/// What one [`sync_society`] did, and what the society holds after it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SocietyStats {
    /// Volumes folded into the network so far.
    pub volumes: usize,
    /// Volumes folded on THIS step. Non-zero means the town is still building,
    /// and nothing plans a day on such a step.
    pub folded_now: usize,
    /// Homes the level has offered — the people it HAS, plus the ones still
    /// waiting for a day, plus the ones the ceiling declined.
    pub homes: usize,
    /// Workplaces the level has offered.
    pub works: usize,
    /// Errands the level has offered.
    pub errands: usize,
    /// **Night-shift jobs the level has offered** (wave VEN1b) — three per
    /// venue: a counter, a deck and a door.
    pub night_jobs: usize,
    /// **Leisure places the level has offered** — every seat, bench place and
    /// patch of dance floor its venues hold. The occupancy ceiling of the whole
    /// town's nightlife, and it is the CONTENT's number rather than a constant.
    pub leisure_places: usize,
    /// Agents who took a night job. At most [`night_jobs`](Self::night_jobs),
    /// and less when a venue stands further than [`NIGHT_JOB_MAX_M`] from
    /// anybody's home.
    pub night_workers: usize,
    /// **Agents whose day ends at a venue rather than at home** — the town's
    /// nightlife, as a count.
    ///
    /// At most [`leisure_places`](Self::leisure_places) by construction (a
    /// place is claimed), and less when [`NIGHTLIFE_SHARE`] of the town is
    /// smaller than the venues are, or when a venue is unroutable from a home.
    pub revellers: usize,
    /// **Agents who wanted a night out and found nowhere to have one** — the
    /// share that drew in and then met either a full town or an unroutable one.
    ///
    /// Not a defect and deliberately counted: a bar seats eleven and a town
    /// holds three hundred, so this is *large* and is what says the cap is the
    /// venue rather than the constant.
    pub turned_away: usize,
    /// Agents installed into the population.
    pub agents: usize,
    /// Homes still waiting for a day.
    pub pending: usize,
    /// Days planned on this step.
    pub planned_now: usize,
    /// Agents whose home routes to no workplace, and who therefore keep a
    /// stay-at-home day. A number to watch: a large one means the network is not
    /// joined up.
    pub homebound: usize,
    /// Agents with nowhere at all to go — no reachable work and no reachable
    /// errand. They stand at home.
    pub housebound: usize,
    /// **Agents with a workplace whose walk HOME did not route** (NPC1d audit).
    ///
    /// A day is not a day if it only goes one way: such an agent leaves at
    /// eight, arrives at nine and stands at its desk through the night, because
    /// the leg that would take it home is not in its schedule. It used to be
    /// counted as a full day (the private `DayKind::Full` is returned the
    /// moment the outbound commute routes) — so the one number that would have
    /// said so was invisible. Zero on the island, and the gate asserts it.
    pub no_return: usize,
    /// **Agents with a workplace and no errand** (NPC1d audit) — the difference
    /// between "scheduled" and "a full four-leg day" in the wave ledger, given
    /// a name.
    ///
    /// Not a defect: it is carried item 2 in the units the report is written
    /// in. An agent is planned on the first step that folds nothing, which in a
    /// streaming town is a *gap* rather than the end, so an agent planned early
    /// chooses its shop from the shops that had folded by then and never
    /// re-opens the question.
    pub errandless: usize,
    /// Nodes in the network.
    pub nodes: usize,
    /// Directed edges in the network.
    pub edges: usize,
    /// Front doors joined to a pavement.
    pub frontages: usize,
    /// Front doors further than [`FRONTAGE_MAX_M`] from their own block's ring,
    /// and therefore not joined. Should be zero on a settlement.
    pub frontages_refused: usize,
    /// Pavement rings joined to a neighbour's.
    pub crossings: usize,
    /// **Interior node ids that were already in the network when their building
    /// was absorbed** — a building salt collision, which welds one building's
    /// room to another's. Zero is the expectation and the arm.
    ///
    /// # What it can and cannot see (NPC1d audit)
    ///
    /// It sees a collision **between two volumes**: the second volume's front
    /// door is already a node of the level network, and that is the check
    /// below. It does **not** see one between two buildings *of the same
    /// volume*, because their interiors are folded into one graph by
    /// `inf_pcg::GrammarOutput::absorb` — whose own rule is that the first
    /// record wins — long before this module is handed the result. Such a pair
    /// arrives here as one building wearing both sets of doors and is
    /// indistinguishable from a building that always had them.
    ///
    /// The width is what makes that acceptable rather than the counter: at 2³⁹
    /// over an island's thousand buildings the birthday probability is about one
    /// in a million, and the number is sized for the intra-volume case as much
    /// as for the other one. Making it visible needs a counter minted where the
    /// salts are — `inf_pcg::building::evaluate_buildings_in` — and carried out
    /// through `GrammarOutput`, which is the mirror-fenced path two hosts
    /// compare character for character. Carried, named, and not built.
    pub salt_collisions: usize,
    /// **Homes the level offered and the society declined**, because it had
    /// already reached [`SOCIETY_MAX_AGENTS`]. Non-zero is not a defect — it is
    /// the ceiling doing its job, and the number is how far past it a settlement
    /// reached.
    pub homes_declined: usize,
    /// Slots in a building with no exterior door — nobody can walk in, so
    /// nobody living there gets a day. Zero on a settlement.
    pub doorless: usize,
    /// Agent `Guid`s `add_agents` refused because the world already held them.
    pub guid_refusals: usize,
    /// Street routes searched over the level network so far — the outer halves
    /// that were NOT served by the memo.
    pub outer_searches: usize,
    /// Outer halves served by the memo. The ratio of this to
    /// [`outer_searches`](Self::outer_searches) is what the two-level split buys
    /// on a real settlement.
    pub outer_cached: usize,
}

/// **The level's society** — its walkable network, the places it offers, and the
/// homes that have not been given a day yet.
///
/// A bevy resource, so nothing here is serialized and no schema moves. Absent
/// until [`sync_society`] finds a volume with residents on it, which is what
/// makes "a level with no population costs one `contains_resource`" structural.
#[derive(Resource, Debug, Clone, Default)]
pub struct SocietyRes {
    /// **The level network**: every block's pavement, every slot-bearing
    /// building's front door, the frontage links between them and the crossings
    /// between blocks. About 1 600 nodes on a settlement — see the module docs
    /// for why the interiors are not in it.
    pub network: NavGraph,
    /// **Each volume's own interior**, searched inside the building.
    pub interiors: BTreeMap<Uuid, NavGraph>,
    /// **Every block's pavement ring, kept**: its rectangle and its eight nodes.
    ///
    /// Kept rather than re-derived from the resident volumes, because a volume
    /// that has streamed OUT still has its ring in the network and a block
    /// folded later must still be able to cross to it. Deriving the crossing
    /// candidates from residency alone would make the network a function of what
    /// had paged in when — the hazard NPC1c named about positions, one level up.
    pub rings: BTreeMap<Uuid, BlockRing>,
    /// **The outer half of a leg, memoized on its endpoint pair.** A hundred
    /// residents of one block commuting to one office share one street route,
    /// and this is what makes them pay for it once. `None` records a pair the
    /// network cannot join, so a refusal is not re-searched either — **and the
    /// `None`s are dropped on any step that folds a volume** (NPC1d audit),
    /// because the network only ever grows and a pair that could not be joined
    /// before the bridging block arrived can be joined after it.
    pub legs: BTreeMap<(NavNodeId, NavNodeId), Option<inf_nav::NavPath>>,
    /// Volumes already folded in, by `Guid`.
    pub folded: BTreeSet<Uuid>,
    /// Every **day-shift** workplace, in the order the volumes were folded.
    pub work: Vec<SocietyPlace>,
    /// Every errand.
    pub errand: Vec<SocietyPlace>,
    /// **Every night-shift job** (wave VEN1b) — a venue's keeper, its act, its
    /// door — in the order the volumes were folded.
    ///
    /// Kept apart from [`work`](Self::work) rather than flagged inside it,
    /// because the two are searched for by different agents at different hours
    /// and a `nearest()` over one list would offer a bar's counter to somebody
    /// looking for a desk at eight in the morning.
    pub night_work: Vec<SocietyPlace>,
    /// **Every place the town can spend an evening** (wave VEN1b) — a seat, a
    /// bench at a stage edge, a patch of dance floor.
    pub leisure: Vec<SocietyPlace>,
    /// **Which [`night_work`](Self::night_work) entries are already somebody's
    /// job**, by index (wave VEN1b).
    ///
    /// A night job is CLAIMED and a day job is not, and the difference is
    /// deliberate: an office slot is one of `area / 12` identical desks in one
    /// room and every worker stands on the room's own node anyway, while a
    /// venue's three jobs are three *distinct places* — behind that counter, on
    /// that deck, at that door — and two bodies claiming one of them stand
    /// inside each other. See [`SocietyStats::night_workers`].
    ///
    /// Indices rather than a `Vec<bool>` beside the list, because the list only
    /// ever grows (a volume folds, its jobs are appended) and an index is
    /// therefore stable; a parallel vector would have to be grown in lockstep
    /// at every append.
    pub taken_night: BTreeSet<usize>,
    /// **Which [`leisure`](Self::leisure) places are already somebody's seat**,
    /// by index (wave VEN1b).
    ///
    /// **THIS IS THE OCCUPANCY CAP.** A dance floor is the densest interior
    /// this engine has and a kinematic `Full` crowd agent does not part for
    /// another; the stations are laid at least
    /// `inf_pcg::building::station::MINGLE_PITCH_M` apart, and claiming one
    /// exclusively is what turns that spacing into a promise about bodies
    /// rather than about points.
    pub taken_leisure: BTreeSet<usize>,
    /// Homes with no day yet, by the agent `Guid` that will live there.
    pub pending: BTreeMap<Uuid, SocietyPlace>,
    /// The counters after the last sync.
    pub stats: SocietyStats,
}

/// **The `Guid` of the agent who lives in one home slot** — a hash of the
/// level's own content, so two hosts mint the same one without talking.
///
/// `(volume, building, room, index)` is the level's name for that bed. Nothing
/// about it depends on iteration order, on when the volume streamed in, or on
/// how many agents have been minted already, which is what makes a society
/// re-derivable from a level rather than a thing a save file has to carry.
pub fn agent_guid(volume: Uuid, building: u32, room: u32, index: u32) -> Uuid {
    let b = volume.as_u128();
    let hi = mix64((b as u64) ^ SALT_AGENT ^ u64::from(building));
    let lo = mix64(((b >> 64) as u64) ^ (u64::from(room) << 32) ^ u64::from(index) ^ hi);
    let raw = (u128::from(hi) << 64) | u128::from(lo);
    Uuid::from_u128((AGENT_TAG << 112) | (raw & ((1u128 << 112) - 1)))
}

/// The SplitMix64 finalizer, the house mixer.
///
/// The same constants `crate::crowd::agent_rand` pins, spelled out here rather
/// than borrowed so this module's ids do not move if that one's salts ever do.
pub(crate) fn mix64(x: u64) -> u64 {
    let mut x = x ^ 0x9e37_79b9_7f4a_7c15;
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// The id of a pavement node standing at `p` — a hash of its quantized XZ.
///
/// Position rather than an index, so two blocks that lay a node at the same
/// corner lay the **same** node and the rings weld themselves. XZ and not Y,
/// because two neighbouring pads may differ by a few centimetres and a corner is
/// a corner; the last ring to write it wins, which is deterministic because
/// volumes are folded in `Guid` order.
pub fn pavement_node_id(p: DVec2) -> NavNodeId {
    let q = |v: f64| -> i64 {
        if !v.is_finite() {
            return 0;
        }
        (v / PAVEMENT_LATTICE_M).floor() as i64
    };
    let h = mix64((q(p.x) as u64) ^ mix64(q(p.y) as u64));
    inf_nav::domain::PAVEMENT | (h & inf_nav::domain::LOCAL_MASK)
}

/// **Which building an interior node belongs to** — the salt out of its id.
///
/// The mirror of `inf_pcg::building::node_salt`, spelled here because `inf-ecs`
/// does not depend on `inf-pcg` (the P19.5 dependency-light mirror ruling, which
/// is also why [`crate::components::ResidentSlot`] exists at all). The layout is
/// frozen by `inf_pcg::building::interior_nav_in`'s own doc table: domain in
/// 60–63, class in 59, salt in 20–58, index in 0–19.
fn node_salt(id: NavNodeId) -> u64 {
    (id >> 20) & ((1 << 39) - 1)
}

/// One volume's own eight pavement points, world metres, in a walking order
/// round the ring.
fn ring_points(centre: DVec3, extent: DVec2, y: f64) -> [DVec3; 8] {
    let (x, z) = (extent.x + PAVEMENT_M, extent.y + PAVEMENT_M);
    let (cx, cz) = (centre.x, centre.z);
    let p = |dx: f64, dz: f64| DVec3::new(cx + dx, y, cz + dz);
    [
        p(-x, -z),
        p(0.0, -z),
        p(x, -z),
        p(x, 0.0),
        p(x, z),
        p(0.0, z),
        p(-x, z),
        p(-x, 0.0),
    ]
}

/// The plan distance between two axis-aligned block rectangles, metres — `0.0`
/// when they overlap.
pub(crate) fn rect_gap(a_c: DVec3, a_e: DVec2, b_c: DVec3, b_e: DVec2) -> f64 {
    rect_gap_2d(DVec2::new(a_c.x, a_c.z), a_e, DVec2::new(b_c.x, b_c.z), b_e)
}

/// [`rect_gap`] over rectangles that are already **plan** (wave ROAD1b).
///
/// The 3D spelling above is the one every caller with a `GlobalTransform` in
/// hand wants; this is the one `traffic::streets_of_blocks` wants, because a
/// block's plan is all that derivation ever reads and carrying a Y through it
/// would invite someone to use it.
pub(crate) fn rect_gap_2d(a_c: DVec2, a_e: DVec2, b_c: DVec2, b_e: DVec2) -> f64 {
    let dx = ((a_c.x - b_c.x).abs() - (a_e.x + b_e.x)).max(0.0);
    let dz = ((a_c.y - b_c.y).abs() - (a_e.y + b_e.y)).max(0.0);
    (dx * dx + dz * dz).sqrt()
}

/// **Where a volume is** — the half every sync needs, and the half that costs
/// nothing to read.
///
/// Split from the heavy half deliberately. A sync runs on **every fixed step**
/// and most of them fold nothing, so cloning every resident list and every
/// interior graph each time would be a per-step copy of a settlement — hundreds
/// of graphs of a hundred nodes, sixty times a second, to discover that nothing
/// had changed. This is a walk and four scalars; [`heavy_facts`] is what is paid
/// once, for a volume that is actually being folded.
pub(crate) struct VolumeSite {
    pub(crate) guid: Uuid,
    pub(crate) centre: DVec3,
    pub(crate) extent: DVec2,
    pub(crate) pad_y: f64,
}

/// What one volume contributes the once, read out of the world before anything
/// is mutated. Keyed by `Guid` where it is held, so it carries no id of its own.
struct VolumeFacts {
    residents: Vec<crate::components::ResidentSlot>,
    interior: NavGraph,
}

/// Every volume that offers a resident, in `Guid` order — positions only.
pub(crate) fn volume_sites(world: &EcsWorld) -> Vec<VolumeSite> {
    let mut out: Vec<VolumeSite> = Vec::new();
    for e in world.world().iter_entities() {
        let (Some(g), Some(v)) = (e.get::<Guid>(), e.get::<PcgVolume>()) else {
            continue;
        };
        if v.residents.is_empty() {
            continue;
        }
        let centre = e
            .get::<GlobalTransform>()
            .map(|t| t.translation())
            .unwrap_or(DVec3::ZERO);
        if !centre.is_finite() {
            continue;
        }
        // **The pad, from the level's own front doors.** A block's volume
        // entity sits at the block centre, whose Y is whatever the level put
        // there; a ground-floor exterior doorway's sill is the walking surface a
        // body actually stands on. Taking the pad from the doors is what keeps a
        // pavement out of the hillside without a terrain query — NPC1c's defect
        // 5, closed by construction rather than by a profile.
        let pad_y = v
            .doorways
            .iter()
            .filter(|d| d.exterior && d.floor == 0 && d.hinge.is_finite())
            .map(|d| d.hinge.y - d.height_m * 0.5)
            .fold(f64::INFINITY, f64::min);
        out.push(VolumeSite {
            guid: g.0,
            centre,
            extent: DVec2::new(v.extent.x, v.extent.y),
            pad_y: if pad_y.is_finite() { pad_y } else { centre.y },
        });
    }
    out.sort_by_key(|f| f.guid);
    out
}

/// The residents and the interior of the volumes in `want` — the copy that is
/// paid once, when a volume is folded, and never again.
fn heavy_facts(world: &EcsWorld, want: &BTreeSet<Uuid>) -> BTreeMap<Uuid, VolumeFacts> {
    let mut out = BTreeMap::new();
    if want.is_empty() {
        return out;
    }
    for e in world.world().iter_entities() {
        let (Some(g), Some(v)) = (e.get::<Guid>(), e.get::<PcgVolume>()) else {
            continue;
        };
        if !want.contains(&g.0) {
            continue;
        }
        out.insert(
            g.0,
            VolumeFacts {
                residents: v.residents.clone(),
                interior: v.interior_nav.clone(),
            },
        );
    }
    out
}

/// **The body a crowd wears on this level** (NPC1d) — the lowest-`Guid` entity
/// that carries a rigged [`SkeletalMesh`](crate::components::SkeletalMesh), with
/// its skeleton and its state machine.
///
/// Derived rather than configured, and derived HERE rather than by each host, so
/// a crowd installed by the editor's Simulate and one installed by the shipped
/// player are made of the same asset without either being told which. The
/// lowest `Guid` because a rule that says "the first one" over a bevy world is a
/// rule about archetype layout.
///
/// A level with no rigged character answers a bodiless humanoid: an NPC with the
/// right capsule, the right feet offset and no mesh. It still walks, still
/// collides and still traces; it is simply not drawn. That is the honest answer
/// for a level that has nothing to draw it with, and it is why this returns a
/// value rather than an `Option` a caller would have to case-split.
pub fn level_archetype(world: &EcsWorld) -> CrowdArchetype {
    level_archetypes(world)
        .into_iter()
        .next()
        .unwrap_or_else(|| CrowdArchetype::humanoid(None, None, None))
}

/// **Every body the level offers**, distinct by `(mesh, skeleton, machine)`, in
/// `Guid` order (wave CHAR1a.2).
///
/// [`level_archetype`] answers with the first of these, which is what it always
/// answered and what a single-body caller still wants. This is the door a CROWD
/// takes, because a street in which every person is the same person is a street
/// nobody believes — and until this wave that was not a content problem, it was
/// a rule: the body was "the lowest-`Guid` rigged entity", singular, so a level
/// carrying a man and a woman dressed all thousand of its pedestrians as the
/// man.
///
/// Distinctness is by the three asset ids and nothing else. Two entities wearing
/// the same body are one body, however many of them there are, so a level that
/// spawns ten copies of its hero still offers one archetype and its crowd is
/// unchanged — the property that keeps every committed level's crowd exactly
/// where it was.
///
/// **Crowd agents are excluded from the survey**, and that is load-bearing: a
/// materialized agent *is* an entity carrying a rigged `SkeletalMesh`, so a
/// survey that counted them would feed the crowd's own output back into its
/// input and the set could grow on the second step. The predicate is the
/// [`CrowdAgent`](crate::crowd::CrowdAgent) component, which is exactly "this
/// entity is one of ours".
///
/// Returns an EMPTY vector for a level with no rigged character; the callers'
/// own fallback (a bodiless humanoid, which still walks and collides and traces)
/// belongs to them.
pub fn level_archetypes(world: &EcsWorld) -> Vec<CrowdArchetype> {
    let mut found: Vec<(Uuid, CrowdArchetype)> = Vec::new();
    for e in world.world().iter_entities() {
        let (Some(g), Some(sk)) = (e.get::<Guid>(), e.get::<crate::components::SkeletalMesh>())
        else {
            continue;
        };
        if sk.skeleton.is_none() || e.get::<crate::crowd::CrowdAgent>().is_some() {
            continue;
        }
        let sm = e
            .get::<crate::components::AnimStateMachine>()
            .and_then(|a| a.sm);
        let a = CrowdArchetype::humanoid(sk.mesh, sk.skeleton, sm);
        if found
            .iter()
            .any(|(_, b)| (b.mesh, b.skeleton, b.sm) == (a.mesh, a.skeleton, a.sm))
        {
            continue;
        }
        found.push((g.0, a));
    }
    // `iter_entities` walks bevy's archetype layout, which is not an order this
    // engine may depend on — the same rule `level_archetype` stated by taking a
    // minimum rather than a first.
    found.sort_by_key(|(g, _)| *g);
    found.into_iter().map(|(_, a)| a).collect()
}

/// **The body one agent wears**, chosen from [`level_archetypes`] by that
/// agent's own `Guid` (wave CHAR1a.2).
///
/// Deterministic and stateless: the same agent picks the same body on every
/// host, every run and every reload, because the only input is the id it already
/// carries. A one-body level answers that body for every agent, so nothing a
/// committed level does changes.
///
/// The salt is [`crate::crowd::SALT_BODY`], its own, so adding this could not
/// move any existing agent's speed, phase, schedule, look or build — all of
/// which are drawn from the same generator with salts of their own.
pub fn level_archetype_for(bodies: &[CrowdArchetype], guid: Uuid) -> CrowdArchetype {
    if bodies.is_empty() {
        return CrowdArchetype::humanoid(None, None, None);
    }
    let i = crate::crowd::agent_rand(guid, 0, crate::crowd::SALT_BODY) as usize % bodies.len();
    bodies[i]
}

/// **Grow the level's society, and install any day it can plan** (NPC1d) — the
/// one Ring-0 door both hosts call, once per fixed step, inside the crowd phase.
///
/// Cheap when nothing changed: one walk over the entities to see whether a
/// volume with residents has appeared, and nothing else on a step that neither
/// folded a volume nor had a home waiting.
///
/// Returns this step's counters; they are also left on
/// [`SocietyRes::stats`].
pub fn sync_society(world: &mut EcsWorld) -> SocietyStats {
    let sites = volume_sites(world);
    if sites.is_empty() && !world.world().contains_resource::<SocietyRes>() {
        // Absent costs nothing.
        return SocietyStats::default();
    }
    let mut soc = world
        .world_mut()
        .remove_resource::<SocietyRes>()
        .unwrap_or_default();
    // **Every counter here is a TOTAL except the two that say "this step"**
    // (`folded_now`, `planned_now`). The first cut reset the fold's own counters
    // each sync, so a gate that read them after the town had settled saw
    // `0 frontages, 0 crossings` over a network that plainly had both — a report
    // that says a system did nothing, about a system that did it on an earlier
    // step.
    let mut stats = SocietyStats {
        agents: soc.stats.agents,
        guid_refusals: soc.stats.guid_refusals,
        homebound: soc.stats.homebound,
        housebound: soc.stats.housebound,
        no_return: soc.stats.no_return,
        errandless: soc.stats.errandless,
        doorless: soc.stats.doorless,
        outer_searches: soc.stats.outer_searches,
        outer_cached: soc.stats.outer_cached,
        frontages: soc.stats.frontages,
        frontages_refused: soc.stats.frontages_refused,
        crossings: soc.stats.crossings,
        salt_collisions: soc.stats.salt_collisions,
        homes_declined: soc.stats.homes_declined,
        // …and the night's four (VEN1b), on the same terms: they accumulate
        // over the whole run, and a gate that read them after the town settled
        // would otherwise see a nightlife of zero over a town that plainly has
        // one.
        night_workers: soc.stats.night_workers,
        revellers: soc.stats.revellers,
        turned_away: soc.stats.turned_away,
        ..SocietyStats::default()
    };

    // ── 1. fold every volume the network has not seen ──────────────────────
    // Rings are laid FIRST, all of them, so a crossing can never miss a
    // neighbour that happens to be folded later in the same step.
    let want: BTreeSet<Uuid> = sites
        .iter()
        .filter(|f| !soc.folded.contains(&f.guid))
        .map(|f| f.guid)
        .collect();
    let heavy = heavy_facts(world, &want);
    let fresh: Vec<&VolumeSite> = sites.iter().filter(|f| want.contains(&f.guid)).collect();
    let mut rings: BTreeMap<Uuid, Vec<(NavNodeId, DVec3)>> = BTreeMap::new();
    for f in &fresh {
        let pts = ring_points(f.centre, f.extent, f.pad_y);
        let ids: Vec<(NavNodeId, DVec3)> = pts
            .iter()
            .map(|p| (pavement_node_id(DVec2::new(p.x, p.z)), *p))
            .collect();
        for (id, p) in &ids {
            soc.network.add_node(*id, *p, NavKind::Street);
        }
        for i in 0..ids.len() {
            let j = (i + 1) % ids.len();
            soc.network
                .link(ids[i].0, ids[j].0, NavKind::Street, Vec::new());
        }
        rings.insert(f.guid, ids);
    }
    for f in &fresh {
        let ids = rings[&f.guid].clone();
        soc.rings.insert(f.guid, (f.centre, f.extent, ids));
    }
    // The rings a crossing may reach: **every ring this society has ever laid**,
    // whether or not its volume is resident now.
    let known: BTreeMap<Uuid, BlockRing> = std::mem::take(&mut soc.rings);
    for f in &fresh {
        let (ac, ae, a_ids) = &known[&f.guid];
        for (other, (bc, be, b_ids)) in &known {
            if other == &f.guid || rect_gap(*ac, *ae, *bc, *be) > BLOCK_LINK_MAX_M {
                continue;
            }
            // Only one direction per pair: a fresh ring links to everything, and
            // two fresh rings link once because `link` is symmetric and
            // `push_edge` deduplicates on `(to, cost)`.
            let mut best: Option<(f64, NavNodeId, NavNodeId)> = None;
            for (ai, ap) in a_ids {
                for (bi, bp) in b_ids {
                    let d = (*ap - *bp).length();
                    if best.map(|(bd, _, _)| d < bd).unwrap_or(true) {
                        best = Some((d, *ai, *bi));
                    }
                }
            }
            if let Some((_, ai, bi)) = best {
                soc.network.link(ai, bi, NavKind::Street, Vec::new());
                stats.crossings += 1;
            }
        }
    }

    // ── 2. keep each fresh volume's interior, and put its FRONT DOORS on the
    //       level network ───────────────────────────────────────────────────
    soc.rings = known;
    let known = &soc.rings;
    let mut doors: BTreeMap<Uuid, BTreeMap<u64, NavNodeId>> = BTreeMap::new();
    for f in &fresh {
        let Some(hf) = heavy.get(&f.guid) else {
            continue;
        };
        let ring = &known[&f.guid].2;
        let mut mine: BTreeMap<u64, NavNodeId> = BTreeMap::new();
        for n in hf.interior.nodes() {
            // A building's EXTERIOR door is the doorway with one edge: the wall
            // it stands in has no room on the far side, so `interior_nav` links
            // it to exactly one room. Every internal door has two.
            if n.kind != NavKind::Doorway || hf.interior.edges_from(n.id).len() != 1 {
                continue;
            }
            // **A salt collision is two buildings claiming one id.** Checked on
            // the door, which is the node that reaches the shared network, and
            // counted rather than papered over: a collision welds one
            // building's front door to another's, which is a route that walks
            // into the wrong house.
            if soc.network.contains(n.id) {
                stats.salt_collisions += 1;
            }
            soc.network.add_node(n.id, n.position, NavKind::Doorway);
            mine.insert(node_salt(n.id), n.id);
            let mut best: Option<(f64, NavNodeId)> = None;
            for (id, p) in ring {
                let d = (*p - n.position).length();
                if best.map(|(bd, _)| d < bd).unwrap_or(true) {
                    best = Some((d, *id));
                }
            }
            match best {
                Some((d, id)) if d <= FRONTAGE_MAX_M => {
                    soc.network.link(n.id, id, NavKind::Doorway, Vec::new());
                    stats.frontages += 1;
                }
                _ => stats.frontages_refused += 1,
            }
        }
        doors.insert(f.guid, mine);
        soc.interiors.insert(f.guid, hf.interior.clone());
    }

    // ── 3. register the fresh volumes' slots ───────────────────────────────
    for f in &fresh {
        let Some(hf) = heavy.get(&f.guid) else {
            soc.folded.insert(f.guid);
            continue;
        };
        for s in &hf.residents {
            if !s.at.is_finite() {
                continue;
            }
            // The building this slot is in, by the salt its own node carries —
            // the same word `interior_nav_in` wrote and `building_salt` minted.
            let Some(door) = doors
                .get(&f.guid)
                .and_then(|m| m.get(&node_salt(s.node)))
                .copied()
            else {
                // A building with no exterior door is one nobody can walk into.
                // Its people are not people this society can give a day to, and
                // saying so is better than giving them one that starts inside a
                // sealed box.
                stats.doorless += 1;
                continue;
            };
            let place = SocietyPlace {
                role: s.role,
                posture: s.posture,
                face: s.face,
                at: s.at,
                node: s.node,
                volume: f.guid,
                door,
            };
            // **The role AND the shift** (wave VEN1b). A venue's counter is a
            // `Work` slot and so is an office desk; what tells them apart is
            // *when*, and putting a night job in `work` would have the town's
            // residents commuting to a shut bar at eight in the morning.
            match (s.role, s.shift) {
                (SlotRole::Home, _) => {
                    let g = agent_guid(f.guid, s.building, s.room, s.index);
                    soc.pending.entry(g).or_insert(place);
                }
                (SlotRole::Work, SlotShift::Day) => soc.work.push(place),
                (SlotRole::Work, SlotShift::Night) => soc.night_work.push(place),
                (SlotRole::Errand, _) => soc.errand.push(place),
                (SlotRole::Leisure, _) => soc.leisure.push(place),
            }
        }
        soc.folded.insert(f.guid);
    }
    // **A refusal the network has outgrown is not a refusal** (NPC1d audit).
    // [`SocietyRes::legs`] caches `None` for an endpoint pair the level network
    // could not join, so a refusal is not re-searched — and the network only
    // ever GROWS, so a pair refused before the block that bridges it had folded
    // would stay refused for ever and every agent planned afterwards would
    // inherit a verdict about a town that no longer exists. Cleared on a step
    // that folded something, which is rare and bounded. The POSITIVE half is
    // kept: nothing is ever removed from the network, so a path found over it
    // is still walkable, and keeping those is the whole measurement the
    // two-level split rests on.
    if !fresh.is_empty() {
        soc.legs.retain(|_, v| v.is_some());
    }
    stats.folded_now = fresh.len();
    stats.volumes = soc.folded.len();
    stats.works = soc.work.len();
    stats.errands = soc.errand.len();
    stats.night_jobs = soc.night_work.len();
    stats.leisure_places = soc.leisure.len();

    // ── 4. plan days, but never while the town is still arriving ───────────
    // ...and never at all once somebody has installed a population by hand: a
    // caller that says `set_crowd_population` owns the crowd, and a level that
    // walked its own residents back in on the next step would make every number
    // an instrument printed a number about a population it did not choose. The
    // network is still folded above, because that is a property of the level
    // rather than of anybody's crowd.
    let hand = world
        .world()
        .get_resource::<crate::crowd::CrowdPopulationRes>()
        .is_some_and(|p| p.hand_installed);
    if hand {
        soc.pending.clear();
    }
    if stats.folded_now == 0 && !soc.pending.is_empty() {
        let archetype = level_archetype(world);
        // The ceiling. Homes past it are declined in `Guid` order, so which
        // thousand a level gets is a function of its own content.
        let room = SOCIETY_MAX_AGENTS.saturating_sub(stats.agents);
        if room == 0 {
            stats.homes_declined += soc.pending.len();
            soc.pending.clear();
        }
        let batch: Vec<Uuid> = soc
            .pending
            .keys()
            .copied()
            .take(SOCIETY_PLANS_PER_STEP.min(room))
            .collect();
        let mut records: BTreeMap<Uuid, CrowdRecord> = BTreeMap::new();
        for g in batch {
            let home = soc.pending.remove(&g).expect("a key we just read");
            let (rec, kind) = plan_day(&mut soc, archetype, g, home, &mut stats);
            match kind {
                DayKind::Full | DayKind::NightShift => {}
                DayKind::Homebound => stats.homebound += 1,
                DayKind::Housebound => stats.housebound += 1,
            }
            records.insert(g, rec);
            stats.planned_now += 1;
        }
        stats.agents += records.len();
        let refused = crate::crowd::add_agents(world, records);
        stats.agents -= refused;
        stats.guid_refusals += refused;
    }

    // Every home the level has OFFERED, which is not the same as the people
    // it has: the ceiling declines some and a doorless building's are refused.
    stats.homes = stats.agents + soc.pending.len() + stats.homes_declined;
    stats.pending = soc.pending.len();
    stats.nodes = soc.network.len();
    stats.edges = soc.network.edge_count();
    soc.stats = stats;
    world.world_mut().insert_resource(soc);
    stats
}

/// What kind of day one agent got.
enum DayKind {
    /// **The agent has a workplace it can reach**, which is what this kind
    /// names and all it names (NPC1d audit — it used to read "home, work, an
    /// errand and home again", which is the *best* case rather than the test).
    /// Whether the errand and the walk home are in the schedule as well is
    /// [`SocietyStats::errandless`] and [`SocietyStats::no_return`].
    Full,
    /// No reachable workplace — an errand out and back, and home the rest of the
    /// day.
    Homebound,
    /// Nowhere reachable at all. The agent stands at home, which is what a
    /// record with no schedule does.
    Housebound,
    /// **The agent works a night shift** (wave VEN1b) — out at six in the
    /// evening, behind a counter or on a deck until three in the morning, home
    /// and asleep while the town commutes.
    NightShift,
}

/// The nearest place of a role to `from`, ties broken on the node id.
fn nearest(places: &[SocietyPlace], from: DVec3) -> Option<SocietyPlace> {
    let mut best: Option<(f64, SocietyPlace)> = None;
    for p in places {
        let d = (p.at - from).length();
        if !d.is_finite() {
            continue;
        }
        let better = match &best {
            None => true,
            Some((bd, bp)) => d < *bd || (d == *bd && p.node < bp.node),
        };
        if better {
            best = Some((d, *p));
        }
    }
    best.map(|(_, p)| p)
}

/// **The nearest UNCLAIMED place to `from` within `reach`**, by index (wave
/// VEN1b) — the door a scarce place is taken through.
///
/// An index and not a place, because the caller has to mark it taken and a
/// `SocietyPlace` is `Copy` and carries no identity of its own. Ties are broken
/// on the index, which is the list's own append order, so nothing here depends
/// on a hash or on a float comparison beyond `<`.
///
/// The reach is a *plan* distance and is a filter rather than a verdict: a
/// place inside it may still be unroutable, and [`leg`] is what says so.
fn nearest_unclaimed(
    places: &[SocietyPlace],
    taken: &BTreeSet<usize>,
    from: DVec3,
    reach_m: f64,
) -> Option<usize> {
    let mut best: Option<(f64, usize)> = None;
    for (i, p) in places.iter().enumerate() {
        if taken.contains(&i) {
            continue;
        }
        let d = DVec2::new(p.at.x - from.x, p.at.z - from.z).length();
        if !d.is_finite() || d > reach_m {
            continue;
        }
        if best.is_none_or(|(bd, _)| d < bd) {
            best = Some((d, i));
        }
    }
    best.map(|(_, i)| i)
}

/// **Whether this agent goes out tonight** — [`NIGHTLIFE_SHARE`] of the town,
/// drawn once per agent at `tick = 0`.
///
/// `crate::crowd::agent_unit`'s own door, so the draw is the house mixer over
/// `(guid, tick, salt)` and is a constant of the AGENT rather than of the step
/// — the property `CrowdRecord::speed_of` and `hour_of` are both written
/// against, and the reason none of the three is ever stored.
fn goes_out(guid: Uuid) -> bool {
    crate::crowd::agent_unit(guid, 0, SALT_NIGHTLIFE) < NIGHTLIFE_SHARE
}

/// **The arrival a place implies** — what kind of place it is, what a body does
/// there, and which way it faces.
///
/// Every leg's arrival is its DESTINATION's, without exception. There is no
/// "plain" leg helper beside this one and there deliberately is not: a slot a
/// *plan* implies already carries `Stand` and no facing (`slots_of` writes
/// exactly that), so the ordinary four-leg day comes out of this door
/// byte-identical to the day NPC1d planned — and the day it does not, the
/// reason is a change to the SLOT rather than a second table of arrivals to
/// keep in step.
fn arrival_of(p: &SocietyPlace) -> crate::crowd::SlotArrival {
    crate::crowd::SlotArrival {
        role: Some(p.role),
        posture: p.posture,
        face: p.face,
    }
}

/// One leg of a day: leave at an hour, walk `path`, and arrive at `to`.
fn leg_to(to: &SocietyPlace, start_h: f64, travel_h: f64, path: inf_nav::NavPath) -> ScheduleLeg {
    ScheduleLeg {
        start_h,
        travel_h,
        path,
        arrival: arrival_of(to),
    }
}

/// A route between two nodes of one graph, or `None`. A search whose two ends
/// are the same node answers an empty contribution rather than a refusal.
fn hop(graph: &NavGraph, from: NavNodeId, to: NavNodeId) -> Option<Vec<DVec3>> {
    if from == to {
        return graph.node(from).map(|n| vec![n.position]);
    }
    match inf_nav::route(graph, from, to) {
        inf_nav::NavVerdict::Found(r) => Some(r.path.points().to_vec()),
        _ => None,
    }
}

/// **A leg from one place to another**, over the two levels of the search — the
/// one door every leg of every day goes through.
///
/// Same building: one search inside it. Different buildings: out to the front
/// door, along the street (memoized), and in through the other front door. The
/// three point lists are joined by `NavPath::new`, which drops the coincident
/// ends for us.
///
/// # THE LAST FEW METRES (wave VEN1b)
///
/// The nav graph is room centres and doorways — it "knows nothing about the
/// furniture the same plan scatters into the room", as `interior_nav_in`'s own
/// doc says. Every slot a *plan* implies stands exactly on its room's node, so
/// a route that ended at the node ended at the slot and the two were the same
/// point. A **station** does not: a stool is against the wall, a counter's
/// service side is behind the counter, and a body that walked to the room
/// centre and sat down would be sitting in the middle of the dance floor.
///
/// So a leg is extended by the endpoints' own metres — the origin's prepended,
/// the destination's appended — whenever they differ from the node. That is
/// **exactly a no-op for every leg planned before this wave**, because
/// `slots_of` puts every plan-derived slot's `at` on its room node and
/// `NavPath::new` drops a coincident end. Measured that way rather than
/// asserted: `a_plan_slots_leg_is_the_route_it_always_was` compares the two
/// spellings point for point.
///
/// The last segment crosses a room and may pass through the furniture the graph
/// cannot see. That is the same honest limit the graph already carries, one
/// scale smaller, and it is why the seat is *claimed* — two bodies never walk
/// to the same one.
fn leg(
    soc: &mut SocietyRes,
    from: &SocietyPlace,
    to: &SocietyPlace,
    stats: &mut SocietyStats,
) -> Option<inf_nav::NavPath> {
    let finish = |soc: &SocietyRes, mut pts: Vec<DVec3>| -> Vec<DVec3> {
        if node_offset(soc, from) {
            pts.insert(0, from.at);
        }
        if node_offset(soc, to) {
            pts.push(to.at);
        }
        pts
    };
    if from.volume == to.volume {
        let g = soc.interiors.get(&from.volume)?;
        let pts = hop(g, from.node, to.node)?;
        let pts = finish(soc, pts);
        return (pts.len() > 1).then(|| inf_nav::NavPath::new(pts));
    }
    let inside_a = hop(soc.interiors.get(&from.volume)?, from.node, from.door)?;
    let inside_b = hop(soc.interiors.get(&to.volume)?, to.door, to.node)?;
    let key = (from.door, to.door);
    let street = match soc.legs.get(&key) {
        Some(hit) => {
            stats.outer_cached += 1;
            hit.clone()
        }
        None => {
            stats.outer_searches += 1;
            let found = match inf_nav::route(&soc.network, key.0, key.1) {
                inf_nav::NavVerdict::Found(r) => Some(r.path),
                _ => None,
            };
            soc.legs.insert(key, found.clone());
            found
        }
    }?;
    let mut pts = inside_a;
    pts.extend_from_slice(street.points());
    pts.extend_from_slice(&inside_b);
    let pts = finish(soc, pts);
    let path = inf_nav::NavPath::new(pts);
    (path.length_m() > 0.0).then_some(path)
}

/// **Does this place stand somewhere other than its own room's node?**
///
/// The test [`leg`]'s extension is gated on, and it is a comparison against the
/// graph rather than a flag on the place, because the graph is the authority on
/// where a node is and a second copy of that answer is the thing that drifts.
/// `false` when the node is not in the volume's interior at all, which is the
/// conservative answer: a leg that cannot find its own endpoint's node is a leg
/// that should not be lengthened by a metre nobody can check.
fn node_offset(soc: &SocietyRes, p: &SocietyPlace) -> bool {
    let Some(g) = soc.interiors.get(&p.volume) else {
        return false;
    };
    let Some(n) = g.node(p.node) else {
        return false;
    };
    // Half a centimetre — far below any distance that means anything here, and
    // far above the arithmetic difference between `frame.to_world(centre)`
    // computed twice.
    (n.position - p.at).length() > 5e-3
}

/// **Plan one agent's day** — the four legs, the night that follows them, and
/// the three honest fall-backs.
///
/// # The order the three kinds of day are tried in, and why
///
/// 1. **A night job first.** They are scarce — three per venue, claimed
///    exclusively — and a bar with nobody behind the counter is a bar that is
///    shut. Whoever is planned first and lives near enough takes one.
/// 2. **Then the working day**, unchanged from NPC1d, followed by the evening.
/// 3. **Then the fall-backs**, unchanged.
///
/// That order is a decision about scarcity and not about importance: a day job
/// is `area / 12` identical desks and there is always another one, while a
/// venue's deck is one deck.
fn plan_day(
    soc: &mut SocietyRes,
    archetype: CrowdArchetype,
    guid: Uuid,
    home: SocietyPlace,
    stats: &mut SocietyStats,
) -> (CrowdRecord, DayKind) {
    // ── the night shift ────────────────────────────────────────────────────
    if let Some(i) = nearest_unclaimed(&soc.night_work, &soc.taken_night, home.at, NIGHT_JOB_MAX_M)
    {
        let job = soc.night_work[i];
        if let (Some(out), Some(back)) =
            (leg(soc, &home, &job, stats), leg(soc, &job, &home, stats))
        {
            // Claimed only once the walk is known to route BOTH ways: a job
            // claimed and then refused would be a counter nobody could ever
            // stand behind again.
            soc.taken_night.insert(i);
            stats.night_workers += 1;
            if let Some(sched) = CrowdSchedule::new(vec![
                leg_to(&job, NIGHT_WORK_START_H, COMMUTE_H, out),
                leg_to(&home, NIGHT_WORK_END_H, COMMUTE_H, back),
            ]) {
                return (
                    CrowdRecord::scheduled(archetype, sched),
                    DayKind::NightShift,
                );
            }
        }
    }
    // **The evening is planned at most once.** Both day branches below append
    // one, and the first falls through to the second if `CrowdSchedule::new`
    // refuses its legs — which today it cannot (leg 0 is walkable by
    // construction) and which would otherwise claim a second leisure place for
    // one agent and count it twice. A flag rather than a comment, because
    // "unreachable in practice" is the shape this tree has a law about.
    let mut evening_done = false;
    if let Some(w) = nearest(&soc.work, home.at) {
        if let Some(out) = leg(soc, &home, &w, stats) {
            let back = leg(soc, &w, &home, stats);
            let mut legs = vec![leg_to(&w, WORK_START_H, COMMUTE_H, out)];
            // The errand is nearest the WORKPLACE, because that is where the
            // agent is standing at noon.
            let mut got_errand = false;
            if let Some(e) = nearest(&soc.errand, w.at) {
                if let (Some(to_shop), Some(to_work)) =
                    (leg(soc, &w, &e, stats), leg(soc, &e, &w, stats))
                {
                    legs.push(leg_to(&e, ERRAND_OUT_H, ERRAND_H, to_shop));
                    legs.push(leg_to(&w, ERRAND_BACK_H, ERRAND_H, to_work));
                    got_errand = true;
                }
            }
            if !got_errand {
                stats.errandless += 1;
            }
            // **A day that only goes one way is counted** (NPC1d audit). Before
            // it was, an agent whose walk home did not route left at eight and
            // stood at its desk for ever, and the report called it a full day.
            let came_home = back.is_some();
            match back {
                Some(back) => legs.push(leg_to(&home, HOME_H, COMMUTE_H, back)),
                None => stats.no_return += 1,
            }
            // ── and the evening (wave VEN1b) ────────────────────────────────
            //
            // Only for an agent that got home: a body still standing at its
            // desk at six has no front door to leave from, and giving it a
            // night out would teleport it across the town at eight.
            if came_home {
                plan_evening(soc, guid, &home, &mut legs, stats, &mut evening_done);
            }
            if let Some(sched) = CrowdSchedule::new(legs) {
                return (CrowdRecord::scheduled(archetype, sched), DayKind::Full);
            }
        }
    }
    // No workplace. An errand out and back is still a day.
    if let Some(e) = nearest(&soc.errand, home.at) {
        if let (Some(out), Some(back)) = (leg(soc, &home, &e, stats), leg(soc, &e, &home, stats)) {
            let mut legs = vec![
                leg_to(&e, 10.0, ERRAND_H, out),
                leg_to(&home, 16.0, ERRAND_H, back),
            ];
            // A homebound agent has an evening too — it is the *day* it has
            // nowhere to be, and the venues do not care where somebody works.
            plan_evening(soc, guid, &home, &mut legs, stats, &mut evening_done);
            if let Some(sched) = CrowdSchedule::new(legs) {
                return (CrowdRecord::scheduled(archetype, sched), DayKind::Homebound);
            }
        }
    }
    (
        CrowdRecord::standing(archetype, home.at),
        DayKind::Housebound,
    )
}

/// **Append a night out to a day that has one** (wave VEN1b) — out at eight,
/// back at two.
///
/// Nothing is appended when the agent did not draw in
/// ([`NIGHTLIFE_SHARE`]), when every place in reach is already somebody's, or
/// when the one it wanted does not route both ways; the last two are counted in
/// [`SocietyStats::turned_away`], because a town whose venues are full is a
/// different fact from a town that does not go out.
///
/// **The place is CLAIMED**, which is the whole of the crowd-density answer:
/// the stations are laid at least a body's width apart by
/// `inf_pcg::building::station::mingle_points`, and claiming makes that a
/// promise about people rather than about points.
fn plan_evening(
    soc: &mut SocietyRes,
    guid: Uuid,
    home: &SocietyPlace,
    legs: &mut Vec<ScheduleLeg>,
    stats: &mut SocietyStats,
    done: &mut bool,
) {
    if *done || !goes_out(guid) {
        return;
    }
    *done = true;
    let Some(i) = nearest_unclaimed(&soc.leisure, &soc.taken_leisure, home.at, NIGHT_OUT_MAX_M)
    else {
        stats.turned_away += 1;
        return;
    };
    let spot = soc.leisure[i];
    let (Some(out), Some(back)) = (leg(soc, home, &spot, stats), leg(soc, &spot, home, stats))
    else {
        stats.turned_away += 1;
        return;
    };
    soc.taken_leisure.insert(i);
    stats.revellers += 1;
    legs.push(leg_to(&spot, EVENING_OUT_H, EVENING_H, out));
    legs.push(leg_to(home, NIGHT_HOME_H, COMMUTE_H, back));
}

/// The society's counters, or all zeroes on a level that has none.
pub fn society_stats(world: &EcsWorld) -> SocietyStats {
    world
        .world()
        .get_resource::<SocietyRes>()
        .map(|s| s.stats)
        .unwrap_or_default()
}

/// Forget the society — the twin of [`crate::crowd::clear_crowd`], and called
/// beside it for the same reason: a `SceneDoc` snapshot restores entities and
/// components and touches no resource, so without this a stopped Simulate
/// session's network would outlive the run that built it.
pub fn clear_society(world: &mut EcsWorld) {
    world.world_mut().remove_resource::<SocietyRes>();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{DoorwaySlot, ResidentSlot, Transform};
    use inf_nav::domain;

    /// A block with one building on it: a bedroom, an office and a shop, joined
    /// by a corridor, with one exterior door.
    fn block(
        world: &mut EcsWorld,
        guid: Uuid,
        centre: DVec3,
        half: f64,
        salt: u64,
        roles: &[SlotRole],
    ) {
        let day: Vec<VenueSlot> = roles
            .iter()
            .map(|r| VenueSlot {
                role: *r,
                shift: SlotShift::Day,
                posture: crate::components::SlotPosture::Stand,
                offset: DVec3::ZERO,
            })
            .collect();
        block_of(world, guid, centre, half, salt, &day)
    }

    /// One slot a fixture block offers, as the fixture spells it.
    #[derive(Clone, Copy)]
    struct VenueSlot {
        role: SlotRole,
        shift: SlotShift,
        posture: crate::components::SlotPosture,
        /// How far the slot stands from its own room's node — zero for a slot a
        /// PLAN implies, non-zero for a station.
        offset: DVec3,
    }

    /// [`block`], with a shift, a posture and an offset on every slot — the
    /// shape a venue's own rooms have (wave VEN1b).
    fn block_of(
        world: &mut EcsWorld,
        guid: Uuid,
        centre: DVec3,
        half: f64,
        salt: u64,
        slots: &[VenueSlot],
    ) {
        let roles: Vec<SlotRole> = slots.iter().map(|s| s.role).collect();
        let roles = &roles[..];
        world.spawn_with_guid(guid, "block", None);
        let e = world.entity_of(guid).expect("the block");
        let room = |i: usize| domain::BUILDING | (i as u64 & 0xF_FFFF) | (salt << 20);
        let door =
            |i: usize| domain::BUILDING | (1u64 << 59) | (i as u64 & 0xF_FFFF) | (salt << 20);
        let mut g = NavGraph::new();
        // room 0 is the corridor, at the block centre; the others hang off it.
        g.add_node(room(0), centre, NavKind::Room);
        let mut residents = Vec::new();
        for (i, role) in roles.iter().enumerate() {
            let at = centre + DVec3::new(0.0, 0.0, (i as f64 + 1.0) * 2.0);
            g.add_node(room(i + 1), at, NavKind::Room);
            g.add_node(door(i + 1), (at + centre) * 0.5, NavKind::Doorway);
            g.link(room(0), door(i + 1), NavKind::Doorway, Vec::new());
            g.link(door(i + 1), room(i + 1), NavKind::Doorway, Vec::new());
            residents.push(ResidentSlot {
                role: *role,
                at: at + slots[i].offset,
                room: (i + 1) as u32,
                building: 0,
                floor: 0,
                index: i as u32,
                node: room(i + 1),
                posture: slots[i].posture,
                shift: slots[i].shift,
                face: if slots[i].offset == DVec3::ZERO {
                    DVec3::ZERO
                } else {
                    DVec3::Z
                },
            });
        }
        // The exterior door: one edge, at the block's own edge.
        g.add_node(
            door(0),
            centre + DVec3::new(half, 0.0, 0.0),
            NavKind::Doorway,
        );
        g.link(room(0), door(0), NavKind::Doorway, Vec::new());

        let mut vol = PcgVolume {
            extent: crate::math::Vec2d::new(half, half),
            ..Default::default()
        };
        vol.set_population(
            Vec::new(),
            Vec::new(),
            Vec::new(),
            vec![DoorwaySlot {
                hinge: centre + DVec3::new(half, 1.05, 0.0),
                closed_yaw_deg: 0.0,
                width_m: 0.9,
                height_m: 2.1,
                thickness_m: 0.2,
                inside_yaw_deg: 180.0,
                exterior: true,
                floor: 0,
            }],
            residents,
            g,
            Vec::new(),
            Vec::new(),
        );
        world.world_mut().entity_mut(e).insert((
            Transform {
                translation: crate::math::Vec3d::new(centre.x, centre.y, centre.z),
                ..Default::default()
            },
            GlobalTransform(glam::DAffine3::from_translation(centre)),
            vol,
        ));
    }

    /// **The clause: a route crosses building, street and building.** Two blocks
    /// twenty metres apart, one holding a home and one a workplace; the agent's
    /// commute must leave its own building, walk the pavement, and enter the
    /// other one.
    #[test]
    fn a_commute_crosses_building_street_and_building() {
        let mut world = EcsWorld::new();
        block(
            &mut world,
            Uuid::from_u128(1),
            DVec3::new(0.0, 0.0, 0.0),
            10.0,
            0x11,
            &[SlotRole::Home],
        );
        block(
            &mut world,
            Uuid::from_u128(2),
            DVec3::new(28.0, 0.0, 0.0),
            10.0,
            0x22,
            &[SlotRole::Work, SlotRole::Errand],
        );
        // First sync folds; the second plans (nobody plans while a town builds).
        let a = sync_society(&mut world);
        assert_eq!(a.folded_now, 2);
        assert_eq!(a.planned_now, 0, "a day was planned on a folding step");
        assert_eq!(a.salt_collisions, 0);
        assert_eq!(a.frontages, 2, "the two front doors did not both join");
        assert!(a.crossings > 0, "the two pavements were never joined");

        let b = sync_society(&mut world);
        assert_eq!(b.folded_now, 0);
        assert_eq!(b.planned_now, 1, "the one resident got no day");
        assert_eq!(b.agents, 1);
        assert_eq!(b.homebound, 0, "the resident could not reach the workplace");
        assert_eq!(b.housebound, 0);
        assert_eq!(b.guid_refusals, 0);

        // And the commute really crosses all three.
        let pop = world
            .world()
            .get_resource::<crate::crowd::CrowdPopulationRes>()
            .expect("a population");
        let (_, rec) = pop.records.iter().next().expect("one agent");
        let sched = rec.schedule.as_ref().expect("a schedule");
        assert_eq!(sched.legs().len(), 4, "a full day is four legs");
        let commute = &sched.legs()[0];
        assert!(
            commute.path.length_m() > 20.0,
            "the commute is {:.1} m, which is not across a street",
            commute.path.length_m()
        );
        // **The commute crosses all three, asserted on the PATH the agent
        // walks** rather than on a search a reader could re-run differently.
        // The network holds streets and front doors; a leg is joined from an
        // inside hop, a street route and another inside hop, so the claim has to
        // be made of the joined thing.
        let soc = world
            .world()
            .get_resource::<SocietyRes>()
            .expect("a society");
        let home_at = DVec3::new(0.0, 0.0, 2.0);
        let work_at = DVec3::new(28.0, 0.0, 2.0);
        let pts = commute.path.points();
        assert!(
            (pts[0] - home_at).length() < 1e-9,
            "the commute starts at {:?} and the home is at {home_at:?}",
            pts[0]
        );
        assert!(
            (pts[pts.len() - 1] - work_at).length() < 1e-9,
            "the commute ends at {:?} and the work is at {work_at:?}",
            pts[pts.len() - 1]
        );
        // It goes out through ONE block's pavement and in through the other's.
        let mut on_pavement: BTreeSet<NavNodeId> = BTreeSet::new();
        for n in soc.network.nodes() {
            if domain::of(n.id) != domain::PAVEMENT {
                continue;
            }
            if pts.iter().any(|p| (*p - n.position).length() < 1e-6) {
                on_pavement.insert(n.id);
            }
        }
        assert!(
            on_pavement.len() >= 2,
            "the commute touches {} pavement node(s) -- it never crossed a \
             street",
            on_pavement.len()
        );
        // And the two ends are in two DIFFERENT buildings, which is the half a
        // single-namespace network could never make true. The salt is what says
        // so, read off the nodes the two places name.
        let salts: BTreeSet<u64> = soc
            .work
            .iter()
            .map(|w| super::node_salt(w.node))
            .chain(soc.pending.values().map(|h| super::node_salt(h.node)))
            .chain(std::iter::once(super::node_salt(
                soc.interiors[&Uuid::from_u128(1)]
                    .nodes()
                    .next()
                    .expect("a node")
                    .id,
            )))
            .collect();
        assert!(
            salts.contains(&0x11) && salts.contains(&0x22),
            "the two buildings' salts are {salts:?}"
        );
        // The memo did its job: four legs, and the second commute of the day is
        // a different pair, so both searched once and neither twice.
        assert!(b.outer_searches > 0, "no street route was ever searched");
    }

    /// **THE NIGHT** (wave VEN1b): a town with a venue in it stops emptying at
    /// six.
    ///
    /// Eight homes and one venue offering a night job, two seats and a patch of
    /// dance floor. Every claim is a count with a floor, because a town with no
    /// nightlife and a town whose nightlife nobody can reach produce the same
    /// silence:
    ///
    /// * somebody works the night shift, and their day is the *two*-leg one —
    ///   out at eighteen, home at three — rather than the four-leg commute;
    /// * somebody goes out, and their day is **six** legs;
    /// * the evening leg's arrival carries a posture that is not standing;
    /// * the venue's places are **claimed**, so two agents never take one;
    /// * and the ones who wanted a night out and found the venue full are
    ///   counted rather than dropped.
    #[test]
    fn a_town_with_a_venue_in_it_has_a_night() {
        use crate::components::SlotPosture;
        let home = VenueSlot {
            role: SlotRole::Home,
            shift: SlotShift::Day,
            posture: SlotPosture::Stand,
            offset: DVec3::ZERO,
        };
        let mut world = EcsWorld::new();
        // Eight homes, a day job and a shop — so the ordinary day really is the
        // FOUR-leg one and a night shift is the only two-leg day in the town.
        // (Without the shop every day is two legs, `errandless`, and the arm
        // below cannot tell a bar's keeper from an office worker.)
        let mut homes = vec![home; 8];
        for role in [SlotRole::Work, SlotRole::Errand] {
            homes.push(VenueSlot {
                role,
                shift: SlotShift::Day,
                posture: SlotPosture::Stand,
                offset: DVec3::ZERO,
            });
        }
        block_of(
            &mut world,
            Uuid::from_u128(1),
            DVec3::ZERO,
            10.0,
            0x11,
            &homes,
        );
        // The venue: a counter to work behind, two seats and a dance spot —
        // every one of them OFFSET from its own room's node, which is what a
        // station is and what the plan-derived slots are not.
        let off = DVec3::new(1.5, 0.0, 0.0);
        block_of(
            &mut world,
            Uuid::from_u128(2),
            DVec3::new(28.0, 0.0, 0.0),
            10.0,
            0x22,
            &[
                VenueSlot {
                    role: SlotRole::Work,
                    shift: SlotShift::Night,
                    posture: SlotPosture::Stand,
                    offset: off,
                },
                VenueSlot {
                    role: SlotRole::Leisure,
                    shift: SlotShift::Night,
                    posture: SlotPosture::Sit,
                    offset: off,
                },
                VenueSlot {
                    role: SlotRole::Leisure,
                    shift: SlotShift::Night,
                    posture: SlotPosture::Sit,
                    offset: off,
                },
                VenueSlot {
                    role: SlotRole::Leisure,
                    shift: SlotShift::Night,
                    posture: SlotPosture::Dance,
                    offset: off,
                },
            ],
        );
        sync_society(&mut world);
        let mut last = SocietyStats::default();
        for _ in 0..8 {
            last = sync_society(&mut world);
            if last.pending == 0 && last.planned_now == 0 {
                break;
            }
        }
        println!(
            "VEN1b: {} agent(s); {} night job(s) -> {} worker(s); {} leisure \
             place(s) -> {} reveller(s), {} turned away",
            last.agents,
            last.night_jobs,
            last.night_workers,
            last.leisure_places,
            last.revellers,
            last.turned_away
        );
        assert_eq!(last.night_jobs, 1, "the venue offered no night job");
        assert_eq!(last.leisure_places, 3);
        assert_eq!(
            last.night_workers, 1,
            "the venue's counter has nobody behind it"
        );
        assert!(last.revellers > 0, "not one of eight went out");
        assert!(
            last.revellers <= last.leisure_places,
            "{} revellers claimed {} places",
            last.revellers,
            last.leisure_places
        );

        let pop = world
            .world()
            .get_resource::<crate::crowd::CrowdPopulationRes>()
            .expect("a population");
        let mut nights = 0usize;
        let mut outs = 0usize;
        let mut postures: BTreeSet<u8> = BTreeSet::new();
        let mut spots: Vec<[u64; 3]> = Vec::new();
        for (g, rec) in &pop.records {
            let Some(s) = rec.schedule.as_ref() else {
                continue;
            };
            match s.legs().len() {
                // The night shift: out at eighteen, home at three.
                2 if s.legs()[0].start_h == NIGHT_WORK_START_H => {
                    nights += 1;
                    assert_eq!(s.legs()[1].start_h, NIGHT_WORK_END_H);
                }
                6 => {
                    outs += 1;
                    let evening = &s.legs()[4];
                    assert_eq!(evening.start_h, EVENING_OUT_H);
                    assert_eq!(s.legs()[5].start_h, NIGHT_HOME_H);
                    postures.insert(evening.arrival.posture.as_u8());
                    assert_ne!(
                        evening.arrival.face,
                        DVec3::ZERO,
                        "a reveller arrives facing nowhere"
                    );
                    // …and the walk really ENDS at the seat, not at the room's
                    // node in the middle of the floor.
                    let end = evening.path.points()[evening.path.points().len() - 1];
                    spots.push(end.to_array().map(f64::to_bits));
                }
                // The four-leg commute, unchanged: an agent that did not draw
                // in stays the agent NPC1d planned.
                4 => {}
                n => panic!("agent {g} got a {n}-leg day"),
            }
            // Whatever the day, the agent is at HOME at ten in the morning.
            let (i, u) = s.at(10.0);
            assert!(
                u >= 1.0 || i == 0,
                "agent {g} is mid-walk at ten in the morning"
            );
        }
        assert_eq!(nights, 1);
        assert_eq!(outs, last.revellers);
        assert!(
            postures.contains(&SlotPosture::Sit.as_u8())
                || postures.contains(&SlotPosture::Dance.as_u8()),
            "nobody in the club is sitting or dancing: {postures:?}"
        );
        // **THE CLAIM.** Two revellers never walk to one seat, which is the
        // whole of the crowd-density answer expressed as a count.
        let unique: BTreeSet<[u64; 3]> = spots.iter().copied().collect();
        assert_eq!(
            unique.len(),
            spots.len(),
            "{} revellers arrived at {} distinct places",
            spots.len(),
            unique.len()
        );
    }

    /// **The nightlife draw really is [`NIGHTLIFE_SHARE`] of a town**, over a
    /// population big enough for the answer to mean something.
    ///
    /// Eight agents is a coin toss and the arm above deliberately does not test
    /// this; a thousand is the size the arc's own sweeps are quoted at. Written
    /// because a `goes_out` that answered `false` for everybody would make the
    /// night arm's "somebody went out" pass on one lucky draw and the whole
    /// clause vacuous on a real settlement.
    #[test]
    fn a_third_of_a_town_goes_out() {
        const N: u128 = 1_000;
        let out = (0..N)
            .filter(|i| goes_out(Uuid::from_u128(0x4e50_0000_0000_0000 + i)))
            .count();
        let share = out as f64 / N as f64;
        println!(
            "VEN1b: {out} of {N} go out — {:.3} against {NIGHTLIFE_SHARE}",
            share
        );
        assert!(
            (share - NIGHTLIFE_SHARE).abs() < 0.05,
            "the draw put {share:.3} of a thousand agents in a venue against a \
             share of {NIGHTLIFE_SHARE}"
        );
        // …and it is a constant of the agent, not of the call.
        let g = Uuid::from_u128(7);
        assert_eq!(goes_out(g), goes_out(g));
    }

    /// **A night out wraps past midnight**, and the schedule needs no special
    /// case for it: `CrowdSchedule::at` is `(hour - start) mod 24`, so a leg
    /// that begins at two in the morning is the active leg from two until the
    /// next one starts.
    #[test]
    fn a_night_out_wraps_past_midnight() {
        let path = |a: DVec3, b: DVec3| inf_nav::NavPath::new(vec![a, b]);
        let p = path(DVec3::ZERO, DVec3::new(10.0, 0.0, 0.0));
        let sched = CrowdSchedule::new(vec![
            ScheduleLeg {
                start_h: EVENING_OUT_H,
                travel_h: EVENING_H,
                path: p.clone(),
                arrival: crate::crowd::SlotArrival {
                    role: Some(SlotRole::Leisure),
                    posture: crate::components::SlotPosture::Dance,
                    face: DVec3::Z,
                },
            },
            ScheduleLeg {
                start_h: NIGHT_HOME_H,
                travel_h: COMMUTE_H,
                path: p,
                arrival: crate::crowd::SlotArrival::STANDING,
            },
        ])
        .expect("two walkable legs");
        // At eleven at night the reveller is at the club, dancing.
        let (i, u) = sched.at(23.0);
        assert_eq!(i, 0);
        assert!(u >= 1.0, "still walking at eleven ({u})");
        // At half past two it is walking home, and at four it is there.
        let (i, u) = sched.at(2.5);
        assert_eq!(i, 1, "the small-hours leg never became active");
        assert!(u < 1.0, "home already at half past two ({u})");
        assert!(sched.at(4.0).1 >= 1.0);
        // Nothing is standing where the club is: the arrival is the leg's.
        assert_eq!(
            sched.legs()[0].arrival.posture,
            crate::components::SlotPosture::Dance
        );
    }

    /// **A slot a PLAN implies stands on its own room's node, so the leg it
    /// ends is the route it always was** — the arm that says wave VEN1b's
    /// endpoint extension is a no-op for every level that predates it.
    #[test]
    fn a_plan_slots_leg_is_the_route_it_always_was() {
        let mut world = EcsWorld::new();
        block(
            &mut world,
            Uuid::from_u128(1),
            DVec3::ZERO,
            10.0,
            0x11,
            &[SlotRole::Home],
        );
        block(
            &mut world,
            Uuid::from_u128(2),
            DVec3::new(28.0, 0.0, 0.0),
            10.0,
            0x22,
            &[SlotRole::Work],
        );
        sync_society(&mut world);
        sync_society(&mut world);
        let soc = world
            .world()
            .get_resource::<SocietyRes>()
            .expect("a society");
        // Not one place in this level stands off its own node…
        for p in soc.work.iter().chain(soc.pending.values()) {
            assert!(
                !super::node_offset(soc, p),
                "a plan-derived slot at {:?} stands off its room's node",
                p.at
            );
        }
        // …so the commute's ends are the graph's own points, and the extension
        // added nothing.
        let pop = world
            .world()
            .get_resource::<crate::crowd::CrowdPopulationRes>()
            .expect("a population");
        let rec = pop.records.values().next().expect("one agent");
        let commute = &rec.schedule.as_ref().expect("a day").legs()[0];
        let pts = commute.path.points();
        assert_eq!(
            pts[0],
            DVec3::new(0.0, 0.0, 2.0),
            "the commute grew a metre at its start"
        );
        assert_eq!(
            pts[pts.len() - 1],
            DVec3::new(28.0, 0.0, 2.0),
            "the commute grew a metre at its end"
        );
    }

    /// A resident with no reachable workplace still gets a day, and the counter
    /// says which kind — a refusal is a value.
    #[test]
    fn a_resident_with_nowhere_to_work_keeps_a_stay_at_home_day() {
        let mut world = EcsWorld::new();
        block(
            &mut world,
            Uuid::from_u128(1),
            DVec3::ZERO,
            10.0,
            0x11,
            &[SlotRole::Home, SlotRole::Errand],
        );
        sync_society(&mut world);
        let s = sync_society(&mut world);
        assert_eq!(s.agents, 1);
        assert_eq!(
            s.homebound, 1,
            "the resident was given a job it has not got"
        );
        let pop = world
            .world()
            .get_resource::<crate::crowd::CrowdPopulationRes>()
            .expect("a population");
        let sched = pop
            .records
            .values()
            .next()
            .expect("one agent")
            .schedule
            .as_ref()
            .expect("a stay-at-home day is still a day");
        assert_eq!(sched.legs().len(), 2);
    }

    /// **The ceiling fires, and the homes past it are counted rather than
    /// dropped on the floor.**
    ///
    /// `SOCIETY_MAX_AGENTS` is a decision about what a fixed step can carry, and
    /// on the island it has never fired — a stationary hero pages in four blocks
    /// and they offer 329 homes against a ceiling of a thousand. A ceiling with
    /// no arm is the `set_debris_budget` shape this arc has named four times, so
    /// this is the arm: enough blocks to go past it, driven until the society
    /// stops planning, with the declined count asserted rather than the
    /// remainder assumed.
    #[test]
    fn the_population_ceiling_fires_and_says_how_far_past_it_the_level_went() {
        let mut world = EcsWorld::new();
        // Each block offers 8 homes and one workplace; 200 of them is 1 600
        // homes against a thousand-agent ceiling.
        const BLOCKS: usize = 200;
        let roles: Vec<SlotRole> = std::iter::repeat_n(SlotRole::Home, 8)
            .chain(std::iter::once(SlotRole::Work))
            .collect();
        for i in 0..BLOCKS {
            block(
                &mut world,
                Uuid::from_u128(100 + i as u128),
                DVec3::new((i % 20) as f64 * 28.0, 0.0, (i / 20) as f64 * 28.0),
                10.0,
                0x1000 + i as u64,
                &roles,
            );
        }
        let mut last = sync_society(&mut world);
        for _ in 0..(BLOCKS * 8 / SOCIETY_PLANS_PER_STEP + 8) {
            last = sync_society(&mut world);
            if last.pending == 0 && last.planned_now == 0 {
                break;
            }
        }
        assert_eq!(
            last.agents, SOCIETY_MAX_AGENTS,
            "the society took {} people against a {SOCIETY_MAX_AGENTS} ceiling",
            last.agents
        );
        assert_eq!(
            last.homes_declined,
            BLOCKS * 8 - SOCIETY_MAX_AGENTS,
            "{} of {} homes were declined",
            last.homes_declined,
            BLOCKS * 8
        );
        assert_eq!(
            last.homes,
            BLOCKS * 8,
            "the level offered {} homes and the report says {}",
            BLOCKS * 8,
            last.homes
        );
        // And the population really is that size, in the world rather than in
        // the report.
        let pop = world
            .world()
            .get_resource::<crate::crowd::CrowdPopulationRes>()
            .expect("a population");
        assert_eq!(pop.records.len(), SOCIETY_MAX_AGENTS);
    }

    /// A block a kilometre away is a different town, and the pavements do not
    /// pretend otherwise.
    #[test]
    fn two_towns_are_two_components() {
        let mut world = EcsWorld::new();
        block(
            &mut world,
            Uuid::from_u128(1),
            DVec3::ZERO,
            10.0,
            0x11,
            &[SlotRole::Home],
        );
        block(
            &mut world,
            Uuid::from_u128(2),
            DVec3::new(1000.0, 0.0, 0.0),
            10.0,
            0x22,
            &[SlotRole::Work],
        );
        let a = sync_society(&mut world);
        assert_eq!(a.crossings, 0, "a kilometre of sea got a zebra crossing");
        let b = sync_society(&mut world);
        assert_eq!(
            b.housebound, 1,
            "the resident commuted a kilometre with no road"
        );
    }

    /// **Nothing is a function of streaming order that can be**: a level's agent
    /// `Guid`s are a hash of its own content, so the same block folded in a
    /// different order mints the same people.
    #[test]
    fn the_agents_are_the_levels_own_and_not_the_orders() {
        let mut a = EcsWorld::new();
        block(
            &mut a,
            Uuid::from_u128(1),
            DVec3::ZERO,
            10.0,
            0x11,
            &[SlotRole::Home],
        );
        block(
            &mut a,
            Uuid::from_u128(2),
            DVec3::new(28.0, 0.0, 0.0),
            10.0,
            0x22,
            &[SlotRole::Home, SlotRole::Work],
        );
        sync_society(&mut a);
        sync_society(&mut a);
        sync_society(&mut a);

        let mut b = EcsWorld::new();
        // The other order.
        block(
            &mut b,
            Uuid::from_u128(2),
            DVec3::new(28.0, 0.0, 0.0),
            10.0,
            0x22,
            &[SlotRole::Home, SlotRole::Work],
        );
        block(
            &mut b,
            Uuid::from_u128(1),
            DVec3::ZERO,
            10.0,
            0x11,
            &[SlotRole::Home],
        );
        sync_society(&mut b);
        sync_society(&mut b);
        sync_society(&mut b);

        let keys = |w: &EcsWorld| -> Vec<Uuid> {
            w.world()
                .get_resource::<crate::crowd::CrowdPopulationRes>()
                .map(|p| p.records.keys().copied().collect())
                .unwrap_or_default()
        };
        let (ka, kb) = (keys(&a), keys(&b));
        assert_eq!(ka.len(), 2, "two homes made {} agents", ka.len());
        assert_eq!(ka, kb, "two orders minted two different populations");
        assert_eq!(
            crowd_bytes(&a),
            crowd_bytes(&b),
            "two orders produced different crowd traces"
        );
    }

    fn crowd_bytes(w: &EcsWorld) -> Vec<u8> {
        crate::crowd::crowd_state_bytes(w)
    }

    /// A level with no residents installs nothing at all.
    #[test]
    fn a_level_with_no_residents_has_no_society() {
        let mut world = EcsWorld::new();
        let s = sync_society(&mut world);
        assert_eq!(s, SocietyStats::default());
        assert!(world.world().get_resource::<SocietyRes>().is_none());
        assert!(world
            .world()
            .get_resource::<crate::crowd::CrowdPopulationRes>()
            .is_none());
    }

    /// A pavement node's id is its own place, so two blocks laying a node at one
    /// corner lay one node.
    #[test]
    fn a_pavement_node_is_named_by_where_it_stands() {
        let a = pavement_node_id(DVec2::new(12.0, -4.0));
        assert_eq!(a, pavement_node_id(DVec2::new(12.0, -4.0)));
        assert_eq!(a, pavement_node_id(DVec2::new(12.004, -3.998)));
        assert_ne!(a, pavement_node_id(DVec2::new(12.5, -4.0)));
        assert_eq!(domain::of(a), domain::PAVEMENT);
        // A non-finite point does not corrupt the tag.
        assert_eq!(
            domain::of(pavement_node_id(DVec2::new(f64::NAN, 0.0))),
            domain::PAVEMENT
        );
    }
}
