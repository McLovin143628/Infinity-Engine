//! The property battery: random op sequences against every promise the kernel
//! makes.
//!
//! The generator does not produce `Op`s directly — it produces *choices*, which
//! are resolved against the mesh as it stands (pick the `i`-th live half-edge,
//! and so on). Two consequences, both wanted:
//!
//! * The sequences are **reachable**: a random `HalfId` would be dead almost
//!   always and every property would degenerate into "refusals refuse".
//! * Refusals still happen constantly — a fallback id of `u32::MAX` is generated
//!   whenever a pick has nothing to pick from — so the *inertness* property gets
//!   hammered too.
//!
//! Each property is written so a specific defect makes it fail. The three most
//! load-bearing, and the mutation each was verified against, are named in the
//! batch report: link surgery that forgets a `prev` (caught by
//! `validity_holds_after_every_op`), an op that mutates without journalling
//! (caught by `replay_is_a_pure_function_of_the_ops`), and a seam reconstruction
//! that averages corner attributes (caught by `export_is_a_fixed_point`).
//!
//! Two of the properties are paired with **deterministic pins** rather than left
//! to the generator, and the reason is the same in both cases: the input that
//! found the defect turns up in roughly one script in three thousand, so a
//! 256-case run on a random seed is not a re-test of it. See
//! `a_double_sided_sheet_saves_as_two_shells_and_then_holds_still` and
//! `a_repaired_winding_settles_on_the_second_save_not_the_first`. A seed in
//! `property.proptest-regressions` says "this broke once"; a named pin says what
//! broke and holds the numbers.

use proptest::prelude::*;

use inf_dcc::{
    cube, cylinder, from_mesh_asset, op_preserves_ids, plane, to_mesh_asset, torus, validate,
    CornerData, ExportOptions, FaceId, HalfId, ImportError, KnifePoint, MergeTarget, Mesh,
    MeshSession, MirrorAxis, Op, SculptFalloff, SculptMode, SelectMode, SelectionSet, VertId,
};

/// A generated op, before it is resolved against a mesh.
#[derive(Debug, Clone, Copy)]
struct Choice {
    kind: u8,
    a: u16,
    b: u16,
    p: u8,
}

fn choice() -> impl Strategy<Value = Choice> {
    (0u8..32, any::<u16>(), any::<u16>(), any::<u8>()).prop_map(|(kind, a, b, p)| Choice {
        kind,
        a,
        b,
        p,
    })
}

/// A shrunk proptest script, written out as `(kind, a, b, p)` tuples.
///
/// Purely for legibility, and it earns its place: rustfmt puts each field of a
/// four-field struct literal on its own line, so an eighteen-op pin becomes a
/// hundred and ten lines in which the *sequence* — the only thing a reader of a
/// regression fixture cares about — is invisible. The tuple order is the field
/// order, checked by the compiler.
fn choices(raw: &[(u8, u16, u16, u8)]) -> Vec<Choice> {
    raw.iter()
        .map(|&(kind, a, b, p)| Choice { kind, a, b, p })
        .collect()
}

fn base_mesh() -> impl Strategy<Value = Mesh> {
    prop_oneof![
        Just(plane(2.0)),
        Just(cube(1.0)),
        Just(cylinder(0.5, 2.0, 6)),
        Just(torus(1.0, 0.3, 6, 4)),
    ]
}

fn pick<T: Copy>(items: &[T], i: u16, fallback: T) -> T {
    if items.is_empty() {
        fallback
    } else {
        items[i as usize % items.len()]
    }
}

/// Resolve a choice against the current mesh. Deliberately allowed to produce
/// ops that will refuse.
fn make_op(mesh: &Mesh, c: Choice) -> Op {
    let verts: Vec<VertId> = mesh.vert_ids().collect();
    let halfs: Vec<HalfId> = mesh.half_ids().collect();
    let faces: Vec<FaceId> = mesh.face_ids().collect();
    let corners: Vec<HalfId> = halfs
        .iter()
        .copied()
        .filter(|&h| mesh.is_boundary(h) == Some(false))
        .collect();
    let dead_v = VertId(u32::MAX);
    let dead_h = HalfId(u32::MAX);
    let dead_f = FaceId(u32::MAX);
    let scale = |x: u16| (x as f64 / 65_535.0) * 2.0 - 1.0;

    match c.kind {
        0 => Op::AddVertex {
            position: [scale(c.a), scale(c.b), c.p as f64 / 255.0],
        },
        1 => Op::RemoveVertex {
            vert: pick(&verts, c.a, dead_v),
        },
        2 => Op::AddFace {
            verts: vec![
                pick(&verts, c.a, dead_v),
                pick(&verts, c.b, dead_v),
                pick(&verts, c.a.wrapping_add(c.b), dead_v),
            ],
            corners: vec![CornerData::default(); 3],
            slot: None,
        },
        3 => Op::RemoveFace {
            face: pick(&faces, c.a, dead_f),
        },
        4 => Op::SplitEdge {
            half: pick(&halfs, c.a, dead_h),
            t: 0.1 + 0.8 * (c.p as f64 / 255.0),
        },
        5 => Op::CollapseEdge {
            half: pick(&halfs, c.a, dead_h),
        },
        6 => Op::SplitFace {
            from: pick(&corners, c.a, dead_h),
            to: pick(&corners, c.b, dead_h),
        },
        7 => Op::WeldVerts {
            keep: pick(&verts, c.a, dead_v),
            merge: pick(&verts, c.b, dead_v),
        },
        8 => Op::TranslateVerts {
            verts: vec![pick(&verts, c.a, dead_v), pick(&verts, c.b, dead_v)],
            delta: [scale(c.a) * 0.1, scale(c.b) * 0.1, 0.0],
        },
        9 => Op::SetCornerUv {
            half: pick(&corners, c.a, dead_h),
            uv: [scale(c.a), scale(c.b)],
        },
        10 => Op::SetCornerNormal {
            half: pick(&corners, c.a, dead_h),
            normal: if c.p.is_multiple_of(2) {
                None
            } else {
                Some([0.0, 1.0, 0.0])
            },
        },
        11 => Op::SetEdgeSharp {
            half: pick(&halfs, c.a, dead_h),
            sharp: c.p.is_multiple_of(2),
        },
        12 => Op::SetFaceSlot {
            face: pick(&faces, c.a, dead_f),
            slot: None,
        },

        // ── the P23.4 modelling set ────────────────────────────────────────
        //
        // Same rule as above: resolve against the mesh as it stands, so the ops
        // are REACHABLE, and let the fallback ids keep the inertness property
        // fed. Sizes are small and signed so an extrude can go inward, which is
        // where a winding mistake shows up.
        13 => Op::ExtrudeFaces {
            faces: two_faces(&faces, c),
            distance: scale(c.a) * 0.5,
        },
        14 => Op::ExtrudeEdges {
            edges: vec![pick(&halfs, c.a, dead_h)],
            delta: [scale(c.a) * 0.2, scale(c.b) * 0.2, 0.1],
        },
        15 => Op::InsetFaces {
            faces: two_faces(&faces, c),
            amount: 0.05 + 0.2 * (c.p as f64 / 255.0),
            individual: c.p.is_multiple_of(2),
        },
        16 => Op::BevelEdges {
            edges: vec![pick(&halfs, c.a, dead_h)],
            amount: 0.01 + 0.1 * (c.p as f64 / 255.0),
            // Wave D: the segment count VARIES, so the multi-strip construction
            // is inside the determinism law rather than beside it.
            segments: 1 + (c.p % 4) as u32,
        },
        17 => Op::LoopCut {
            half: pick(&halfs, c.a, dead_h),
            cuts: 1 + (c.p % 3) as u32,
        },
        18 => Op::Knife {
            path: vec![
                KnifePoint::Vertex(pick(&verts, c.a, dead_v)),
                KnifePoint::Vertex(pick(&verts, c.b, dead_v)),
            ],
        },
        19 => Op::MergeVerts {
            verts: vec![pick(&verts, c.a, dead_v), pick(&verts, c.b, dead_v)],
            target: if c.p.is_multiple_of(2) {
                MergeTarget::Center
            } else {
                MergeTarget::Last
            },
        },
        20 => Op::SubdivideFaces {
            faces: two_faces(&faces, c),
        },
        21 => Op::Mirror {
            axis: match c.p % 3 {
                0 => MirrorAxis::X,
                1 => MirrorAxis::Y,
                _ => MirrorAxis::Z,
            },
            // A plane through a vertex the mesh actually has, so the exact-zero
            // seam weld is genuinely exercised rather than always missing.
            coord: 0.0,
        },

        // ── the P23.5 sculpt / gizmo set ───────────────────────────────────
        //
        // A stroke is generated with a REAL path (several dabs, resampled by the
        // product's own `stroke_dabs`) rather than a single point, because the
        // whole claim of the op is that a multi-dab gesture replays byte for
        // byte — a one-dab generator would test the easy half only.
        22 => {
            let seed = pick(&verts, c.a, dead_v);
            let start = mesh.position(seed).unwrap_or(glam::DVec3::ZERO);
            // **The radius is a fraction of the model, not an absolute.** The
            // first version used 0.4 m, and a script that happened to apply a few
            // shrinking `ScaleVerts` left every dab covering the WHOLE mesh — a
            // Dijkstra plus a normal fan over every vertex, per dab, which took
            // the reachability battery from 0.7 s to 95 s. A brush sized to the
            // model is also the more honest generator: it exercises the same
            // fraction of the surface whatever the script did to the scale.
            let extent = model_extent(mesh);
            let radius = (extent * 0.12).max(1e-9);
            let path: Vec<glam::DVec3> = (0..3)
                .map(|i| {
                    start
                        + glam::DVec3::new(
                            scale(c.b) * radius * i as f64,
                            radius * 0.1 * i as f64,
                            0.0,
                        )
                })
                .collect();
            Op::Sculpt {
                mode: match c.p % 4 {
                    0 => SculptMode::Draw,
                    1 => SculptMode::Smooth,
                    2 => SculptMode::Flatten,
                    _ => SculptMode::Grab,
                },
                dabs: inf_dcc::stroke_dabs(&path, radius)
                    .into_iter()
                    .map(|d| d.to_array())
                    .collect(),
                radius,
                strength: scale(c.a) * radius * 0.3,
                falloff: match c.p % 3 {
                    0 => SculptFalloff::Smooth,
                    1 => SculptFalloff::Linear,
                    _ => SculptFalloff::Sharp,
                },
            }
        }
        23 => Op::RotateVerts {
            verts: vec![pick(&verts, c.a, dead_v), pick(&verts, c.b, dead_v)],
            pivot: [0.0, 0.0, 0.0],
            axis: [0.0, 1.0, 0.0],
            radians: scale(c.a) * 0.5,
        },
        24 => Op::ScaleVerts {
            verts: vec![pick(&verts, c.a, dead_v), pick(&verts, c.b, dead_v)],
            pivot: [0.0, 0.0, 0.0],
            factor: [
                1.0 + scale(c.a) * 0.2,
                1.0 + scale(c.b) * 0.2,
                1.0 + scale(c.a) * 0.1,
            ],
        },

        // ── the Wave-D set ─────────────────────────────────────────────────
        //
        // Reachability is the point (the P19 vacuous-check law), so each of
        // these is resolved against the mesh as it stands and shaped so that the
        // *applied* outcome is genuinely reachable — a generator that only ever
        // produces refusals leaves every property below vacuous for its kind.
        25 => {
            // Two DISTINCT vertices with different deltas, so the "each by its
            // own" half of the op is what is being replayed. The fallback pair
            // collapses to one id, which is the `SameVertex` refusal.
            let mut ids = vec![pick(&verts, c.a, dead_v), pick(&verts, c.b, dead_v)];
            ids.sort_unstable();
            ids.dedup();
            Op::MoveVerts {
                moves: ids
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| {
                        (
                            v,
                            [
                                scale(c.a) * 0.1 * (i + 1) as f64,
                                scale(c.b) * 0.1,
                                0.05 * i as f64,
                            ],
                        )
                    })
                    .collect(),
            }
        }
        26 => Op::SetEdgesSharp {
            halfs: vec![pick(&halfs, c.a, dead_h), pick(&halfs, c.b, dead_h)],
            sharp: c.p.is_multiple_of(2),
        },
        27 => {
            // Flipping ONE face of a closed shell always refuses, so half the
            // draws flip the WHOLE mesh — otherwise this kind would be
            // unreachable on three of the four base meshes.
            if c.p.is_multiple_of(2) {
                Op::FlipFaces {
                    faces: faces.clone(),
                }
            } else {
                Op::FlipFaces {
                    faces: two_faces(&faces, c),
                }
            }
        }
        // Audit fix: variants 36 and 37 were shipped in this wave and the
        // generator stopped at 29, so `every_modelling_op_applies_at_least_once`
        // proved nothing about either — two of the seven new ops sat outside the
        // determinism, inertness and replay properties entirely.
        29 => Op::SetEdgesSeam {
            halfs: vec![pick(&halfs, c.a, dead_h), pick(&halfs, c.b, dead_h)],
            seam: c.p.is_multiple_of(2),
        },
        30 => Op::MoveUvs {
            // A boundary half-edge has no corner (`BoundaryCorner`), so the
            // applied path needs `corners`; `dead_h` still reaches the refusal.
            corners: vec![
                (
                    pick(&corners, c.a, dead_h),
                    [scale(c.a) * 0.5, scale(c.b) * 0.5],
                ),
                (
                    pick(&corners, c.b, dead_h),
                    [scale(c.b) * 0.5, scale(c.a) * 0.5],
                ),
            ],
        },
        28 => Op::DissolveEdges {
            // `corners` is the interior half-edges; a boundary one is the
            // `DissolveBoundaryEdge` refusal and `dead_h` is `NoSuchHalf`.
            edges: vec![pick(&corners, c.a, dead_h)],
        },
        _ => {
            // A bridge needs two boundary half-edges that are not neighbours;
            // pairing across the loop is the shape that closes an open mesh.
            let border: Vec<HalfId> = halfs
                .iter()
                .copied()
                .filter(|&h| mesh.is_boundary(h) == Some(true))
                .collect();
            if border.len() < 4 {
                return Op::BridgeLoops {
                    pairs: vec![(dead_h, dead_h)],
                };
            }
            let i = c.a as usize % border.len();
            let j = (i + border.len() / 2) % border.len();
            Op::BridgeLoops {
                pairs: vec![(border[i], border[j])],
            }
        }
    }
}

/// The longest side of the mesh's bounding box, or `0` when it has no vertices.
/// Used to size a generated brush against the model rather than against nothing.
fn model_extent(mesh: &Mesh) -> f64 {
    let (mut lo, mut hi) = (glam::DVec3::splat(f64::MAX), glam::DVec3::splat(f64::MIN));
    let mut any = false;
    for v in mesh.vert_ids() {
        if let Some(p) = mesh.position(v) {
            if p.is_finite() {
                lo = lo.min(p);
                hi = hi.max(p);
                any = true;
            }
        }
    }
    if any {
        (hi - lo).max_element().max(0.0)
    } else {
        0.0
    }
}

/// One or two faces — the region form of an op has to be reached with a set that
/// is sometimes bigger than one, or the border-detection rule is never tested.
fn two_faces(faces: &[FaceId], c: Choice) -> Vec<FaceId> {
    if faces.is_empty() {
        return vec![FaceId(u32::MAX)];
    }
    let first = faces[c.a as usize % faces.len()];
    if c.p.is_multiple_of(3) {
        vec![first]
    } else {
        vec![first, faces[c.b as usize % faces.len()]]
    }
}

/// Drive a session through a choice list, returning how many ops applied and how
/// many refused.
fn drive(session: &mut MeshSession, script: &[Choice]) -> (usize, usize) {
    let (mut applied, mut refused) = (0, 0);
    for &c in script {
        let op = make_op(session.mesh(), c);
        let before = session.mesh().encoded();
        match session.apply(op.clone()) {
            Ok(_) => {
                applied += 1;
                assert_eq!(validate(session.mesh()), Ok(()), "invalid after {op:?}");
            }
            Err(_) => {
                refused += 1;
                assert_eq!(
                    session.mesh().encoded(),
                    before,
                    "a refused {op:?} must leave the mesh byte-identical"
                );
            }
        }
    }
    (applied, refused)
}

/// A deterministic 200-op script over every base mesh, asserting that the
/// generator actually **reaches** both outcomes.
///
/// Without this the whole file could be passing because every generated op
/// refuses — six properties about "the mesh after an op" holding vacuously over
/// a mesh no op ever touched. Vacuous checks hide real intrusions (the P19 law),
/// and a generator is exactly the kind of thing that degrades into one silently
/// when an id-picking rule changes.
#[test]
fn the_generator_reaches_both_applied_and_refused_ops() {
    let bases = [
        ("plane", plane(2.0)),
        ("cube", cube(1.0)),
        ("cylinder", cylinder(0.5, 2.0, 6)),
        ("torus", torus(1.0, 0.3, 6, 4)),
    ];
    for (name, base) in bases {
        // A fixed LCG, so this test says the same thing on every run and machine.
        let mut state: u64 = 0x2545_F491_4F6C_DD1D;
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) as u32
        };
        let script: Vec<Choice> = (0..200)
            .map(|_| Choice {
                kind: (next() % 32) as u8,
                a: next() as u16,
                b: next() as u16,
                p: next() as u8,
            })
            .collect();
        let mut session = MeshSession::new(base);
        let (applied, refused) = drive(&mut session, &script);
        assert!(
            applied >= 40,
            "{name}: only {applied}/200 ops applied — the generator has gone vacuous"
        );
        assert!(
            refused >= 20,
            "{name}: only {refused}/200 ops refused — the inertness property is untested"
        );
        assert_eq!(validate(session.mesh()), Ok(()));
    }
}

/// Every one of the twelve modelling / sculpt ops must actually APPLY somewhere
/// in the battery, not merely be generated and refused.
///
/// The P19 vacuity law, aimed at the exact way this file could rot: `make_op`
/// resolves ids against the live mesh, so a change to a picking rule (or an op
/// whose preconditions are tighter than the generator can satisfy) turns a
/// property into a very fast test of nothing. `the_generator_reaches_both_...`
/// counts applications in bulk and would still pass with all nine dead.
#[test]
fn every_modelling_op_applies_at_least_once_somewhere_in_the_battery() {
    let bases = [
        ("plane", plane(2.0)),
        ("cube", cube(1.0)),
        ("cylinder", cylinder(0.5, 2.0, 6)),
        ("torus", torus(1.0, 0.3, 6, 4)),
    ];
    let mut applied = [0usize; 32];
    for (_, base) in bases {
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) as u32
        };
        let seed_mesh = base.clone();
        let mut session = MeshSession::new(base);
        for _ in 0..600 {
            let c = Choice {
                kind: (next() % 32) as u8,
                a: next() as u16,
                b: next() as u16,
                p: next() as u8,
            };
            let op = make_op(session.mesh(), c);
            if session.apply(op).is_ok() {
                applied[c.kind as usize] += 1;
            }
            // **A bounded battery.** `Mirror` doubles the mesh, and once the
            // transform ops have pushed geometry off the mirror plane the seam
            // weld stops collapsing anything — so a script that reaches it a few
            // dozen times grows exponentially. The P23.5 generator found exactly
            // that: 6.6 million vertices on the plane and 93 seconds of CI, in a
            // battery whose point is *reachability*, not size. Restarting from
            // the base past the cap keeps every kind reachable (the counts are
            // cumulative across restarts) and keeps the runtime a constant.
            if session.mesh().vert_count() > 4_000 {
                session = MeshSession::new(seed_mesh.clone());
            }
        }
        assert_eq!(validate(session.mesh()), Ok(()));
    }
    for kind in 13..32 {
        assert!(
            applied[kind] > 0,
            "op kind {kind} never applied — the generator cannot reach it, so \
             every property below is vacuous for it. Counts: {applied:?}"
        );
    }
}

/// The mesh the asset format cannot carry, held still: **a two-sided coincident
/// sheet saves as two shells, and then it stops moving.**
///
/// This is `export_is_a_fixed_point`'s counter-example, made deterministic. The
/// fuzzer found it on Windows CI at wave I2 and proptest's shrunk input lives on
/// a runner that has since been recycled, so the script is written out here
/// rather than left to a seed — the P21 law that a pin nobody can re-run is not
/// a pin. It is also the *only* fixture in the tree where the round trip moves
/// with `coincident_vertices == 0`: every other counter-example in this crate is
/// the f32-coincidence hazard, and that is precisely why the property's evidence
/// clause had no word for this one.
///
/// # What the mesh is, and why one save cannot keep it
///
/// `BridgeLoops` closes the two boundary loops of a dart-shaped quad onto each
/// other, leaving **two coincident faces with opposite windings on the same four
/// vertices** — a flat balloon. That is legal in the kernel and `validate`
/// agrees: no directed edge repeats, every edge has exactly two half-edges.
///
/// It cannot survive a write, and the reason is arithmetic rather than a bug.
/// The quad is non-convex, so exactly **one** of its two diagonals lies inside
/// it; the second sheet is the same polygon reversed, so its only legal diagonal
/// is the same one. Four triangles therefore meet on that one edge, which is not
/// a surface — and no reader can tell that soup apart from a genuine interior
/// partition. The writer says so (`reused_diagonals`), the reader detaches the
/// second sheet and says so (`non_manifold_splits`), and the mesh comes back as
/// two shells at the same coordinates with private vertices.
///
/// The claim that survives, and the one this test exists to hold, is the last
/// paragraph: **the repair is idempotent**. The second save reproduces the first
/// one's bytes, so an author who opens and saves a file they never edited does
/// not watch it grow a vertex buffer every time.
#[test]
fn a_double_sided_sheet_saves_as_two_shells_and_then_holds_still() {
    // The shrunk input from the I2 run, verbatim: `plane(2.0)` and five choices.
    let script = choices(&[
        (4, 26429, 0, 106),  // SplitEdge
        (3, 0, 0, 0),        // RemoveFace
        (2, 41239, 4801, 0), // AddFace
        (4, 21971, 0, 79),   // SplitEdge
        (31, 35761, 0, 0),   // BridgeLoops
    ]);
    let mut session = MeshSession::new(plane(2.0));
    let (applied, refused) = drive(&mut session, &script);
    assert_eq!(
        (applied, refused),
        (5, 0),
        "every op in the pin must still apply"
    );

    // ── the world, not the report ──────────────────────────────────────────
    //
    // Asserted on the face loops rather than on a count, because "2 faces" is
    // also what an ordinary two-triangle mesh has and the whole fixture rests on
    // these two being the SAME polygon wound both ways.
    let faces: Vec<Vec<VertId>> = session
        .mesh()
        .face_ids()
        .map(|f| session.mesh().face_verts(f).expect("live face"))
        .collect();
    assert_eq!(faces.len(), 2, "the bridge left two faces: {faces:?}");
    let mut reversed = faces[0].clone();
    reversed.reverse();
    let start = reversed
        .iter()
        .position(|v| *v == faces[1][0])
        .expect("the second face is drawn on the first one's vertices");
    let rotated: Vec<VertId> = reversed[start..]
        .iter()
        .chain(&reversed[..start])
        .copied()
        .collect();
    assert_eq!(
        rotated, faces[1],
        "the second face is the first one reversed"
    );
    assert_eq!(
        validate(session.mesh()),
        Ok(()),
        "and the kernel calls it legal"
    );

    // ── the write: entitled by a diagonal, and by nothing else ─────────────
    let opts = ExportOptions::default();
    let (a1, w1) = to_mesh_asset(session.mesh(), &opts);
    assert_eq!(
        (w1.coincident_vertices, w1.reused_diagonals),
        (0, 2),
        "the ONLY advisory here is the reused diagonal — if coincidence has \
         appeared, the fixture has drifted onto the hazard it was written to \
         exclude: {w1:?}"
    );
    assert_eq!(
        w1.fan_fallbacks, 0,
        "the clipper found real ears; it did not give up"
    );

    // ── the read: a detach, and pointedly not a fusion ─────────────────────
    let read = from_mesh_asset(&a1).expect("Wave D repairs this rather than refusing it");
    assert_eq!(validate(&read.mesh), Ok(()));
    assert_eq!(
        read.report.non_manifold_splits, 2,
        "both sheets' triangles detach"
    );
    assert_eq!(
        read.report.detach_severity,
        inf_dcc::DetachSeverity::Pervasive
    );
    assert_eq!(
        read.report.degenerate_triangles_skipped, 0,
        "nothing collapsed"
    );
    assert_eq!(
        read.report.welded_positions, w1.vertices,
        "and nothing fused — the exact terms the old evidence clause tested, \
         all false, on a round trip that was entitled to move"
    );

    // ── and then it holds still ────────────────────────────────────────────
    let (a2, _) = to_mesh_asset(&read.mesh, &opts);
    let e1 = inf_asset::encode(&a1).expect("encodable");
    let e2 = inf_asset::encode(&a2).expect("encodable");
    assert_ne!(e1, e2, "the first save is where the sheet is lost");
    let read2 = from_mesh_asset(&a2).expect("a2 reads back");
    let (a3, _) = to_mesh_asset(&read2.mesh, &opts);
    assert_eq!(
        e2,
        inf_asset::encode(&a3).expect("encodable"),
        "the repair must be idempotent: save, open, save again is where an \
         author finds out whether their file grows without being edited"
    );
    assert_eq!(
        read.mesh.canonical(),
        read2.mesh.canonical(),
        "and the mesh it settled on is the same mesh, not merely the same size"
    );
}

/// **The winding repair settles on the second save, not the first** — the
/// measured witness for `export_is_a_fixed_point`'s `SETTLE_CAP`.
///
/// The intuitive form of that property's last clause is "the round trip moves at
/// most once and then holds still", and it is false. This is the fixture that
/// says so, and it is here because the property will almost never find it again:
/// at 20 000 random scripts — 78× what CI runs per push — the *observed* rate of
/// scripts needing a second pass was about one in three thousand, and two
/// separate 20 000-case runs produced **zero**. A cap justified only by a number
/// nobody can re-derive is the P25 law about unmeasured prescriptions; this is
/// the derivation.
///
/// # Why a second pass is needed, and why it is not a defect
///
/// The reader's winding repair walks each component from its lowest-indexed face
/// and flips whatever disagrees, then applies the **minority rule** — if it
/// flipped more than half the component, it flips the whole component back, so
/// the source's own majority winding is what survives. That rule is a decision
/// about the component *as the reader found it*.
///
/// But the writer emits triangles in the winding the mesh now has, so a repaired
/// component comes back out re-wound — and the next read's walk is looking at a
/// different arrangement, from a different seed, with a different majority. It
/// can still find one face to agree. Here the counts are legible: **39
/// reoriented on the first read, 1 on the second, 0 on the third.** Monotone,
/// converging, and finished — the repair is a fixed point reached by iteration
/// rather than a projection reached in one step.
///
/// Making it a projection is a change to the repair rule and not a test's to
/// make; what is pinned here is the behaviour as it stands, so that a future
/// change to `repair_non_manifold` has to come past these three numbers on
/// purpose.
#[test]
fn a_repaired_winding_settles_on_the_second_save_not_the_first() {
    // `cube(1.0)` under the shrunk 18-op script from the I2 measurement run.
    let script = choices(&[
        (27, 0, 0, 14),
        (13, 40748, 48368, 0),
        (4, 11148, 0, 0),
        (3, 15760, 0, 0),
        (28, 57108, 0, 0),
        (3, 37691, 0, 0),
        (22, 12653, 0, 59),
        (0, 0, 0, 1),
        (13, 11546, 1448, 31),
        (4, 55618, 0, 0),
        (0, 0, 0, 1),
        (21, 0, 0, 68),
        (28, 32504, 0, 0),
        (4, 31354, 0, 0),
        (31, 26126, 0, 0),
        (25, 1387, 7398, 0),
        (6, 0, 0, 0),
        (28, 37058, 0, 0),
    ]);
    let mut session = MeshSession::new(cube(1.0));
    drive(&mut session, &script);
    let opts = ExportOptions::default();

    // Four saves and the three reads between them, kept as a sequence so the
    // claim is about the SERIES and not about one comparison.
    let mut asset = to_mesh_asset(session.mesh(), &opts).0;
    let mut bytes = vec![inf_asset::encode(&asset).expect("encodable")];
    let mut reoriented = Vec::new();
    for _ in 0..3 {
        let read = from_mesh_asset(&asset).expect("Wave D repairs rather than refusing");
        assert_eq!(
            validate(&read.mesh),
            Ok(()),
            "every mesh in the chain is legal"
        );
        reoriented.push(read.report.faces_reoriented);
        asset = to_mesh_asset(&read.mesh, &opts).0;
        bytes.push(inf_asset::encode(&asset).expect("encodable"));
    }

    assert_eq!(
        reoriented,
        vec![39, 1, 0],
        "the winding repair converges by iteration: a big first agreement, one \
         straggler the re-written winding exposed, then nothing"
    );
    // The world, not the report: the bytes are what an author's file is.
    assert_ne!(bytes[0], bytes[1], "the first save is repaired");
    assert_ne!(
        bytes[1], bytes[2],
        "and one pass was NOT enough — the point"
    );
    assert_eq!(bytes[2], bytes[3], "the second pass is the fixed point");
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 256, max_shrink_iters: 4_000, ..ProptestConfig::default() })]

    /// A selection is only ever read at the generation it was stamped for.
    ///
    /// The contract the whole selection model rests on, hammered against random
    /// edits: after any op, EITHER the stamp still matches (and every id is still
    /// live) OR the consumer is told to drop. There is no third state in which a
    /// set silently survives a renumbering — and the ops that
    /// `op_preserves_ids` lets through `carry` really do leave every kept id
    /// naming the same polygon.
    #[test]
    fn a_selection_never_outlives_the_generation_it_was_stamped_for(
        base in base_mesh(),
        script in prop::collection::vec(choice(), 1..24),
    ) {
        let mut session = MeshSession::new(base);
        let mut sel = SelectionSet::new(session.generation());
        for &c in &script {
            // Select EVERY face, then edit.
            //
            // Selecting only the first was a gate that did not fire: a structural
            // op rebuilds two or three faces out of dozens, so a one-face
            // selection usually missed them and the property passed with the
            // id-preservation table lying about `SplitEdge` (measured). The set
            // has to contain what the op will touch, whatever it turns out to
            // touch.
            for f in session.mesh().face_ids() {
                sel.set_face(f, true);
            }
            let before_faces: Vec<(FaceId, Vec<VertId>)> = sel
                .faces()
                .iter()
                .map(|&f| (f, session.mesh().face_verts(f).unwrap_or_default()))
                .collect();
            let op = make_op(session.mesh(), c);
            let preserves = op_preserves_ids(&op);
            let Ok(outcome) = session.apply(op) else { continue };
            if preserves {
                sel.carry(session.generation(), session.mesh());
                for (f, loop_verts) in before_faces {
                    if sel.contains_face(f) {
                        prop_assert_eq!(
                            session.mesh().face_verts(f).unwrap_or_default(),
                            loop_verts,
                            "carry kept a face id that changed meaning"
                        );
                    }
                }
            } else {
                sel.adopt(session.generation(), &outcome, session.mesh());
            }
            prop_assert_eq!(sel.generation(), session.generation());
            for &f in sel.faces() {
                prop_assert!(session.mesh().has_face(f), "a dead face is selected");
            }
            for &v in sel.verts() {
                prop_assert!(session.mesh().has_vert(v), "a dead vertex is selected");
            }
            for &h in sel.edges() {
                prop_assert!(session.mesh().has_half(h), "a dead edge is selected");
            }
            prop_assert!(!sel.sync(session.generation()), "already in sync");
        }
    }

    /// Soft-select weights are bounded, seed-anchored and order-independent.
    #[test]
    fn soft_select_weights_are_bounded_and_deterministic(
        base in base_mesh(),
        radius in 0.05f64..3.0,
    ) {
        let seeds: Vec<VertId> = base.vert_ids().step_by(3).collect();
        let mut sel = SelectionSet::new(7);
        for v in &seeds { sel.set_vert(*v, true); }
        let a = sel.soft_weights(&base, SelectMode::Vert, radius, inf_terrain::Falloff::Smooth);
        let b = sel.soft_weights(&base, SelectMode::Vert, radius, inf_terrain::Falloff::Smooth);
        prop_assert_eq!(&a, &b);
        for (&v, &w) in &a {
            prop_assert!((0.0..=1.0).contains(&w), "weight {} at {}", w, v);
            prop_assert!(base.has_vert(v));
        }
        for v in &seeds {
            prop_assert_eq!(a.get(v), Some(&1.0), "a seed is full weight");
        }
    }

    /// The kernel's headline promise: whatever the op, the mesh is still a mesh.
    /// Mutation-verified — dropping the `prev` fix-up in `add_face_raw` makes
    /// this fail with `NextPrevMismatch`.
    #[test]
    fn validity_holds_after_every_op(base in base_mesh(), script in prop::collection::vec(choice(), 1..40)) {
        prop_assert_eq!(validate(&base), Ok(()));
        let mut session = MeshSession::new(base);
        let (applied, refused) = drive(&mut session, &script);
        prop_assert!(applied + refused == script.len());
    }

    /// `replay(base, ops)` is the mesh, byte for byte. This is what makes undo
    /// "truncate and replay" sound, and it is what an op that mutates without
    /// journalling breaks.
    #[test]
    fn replay_is_a_pure_function_of_the_ops(base in base_mesh(), script in prop::collection::vec(choice(), 1..40)) {
        let mut session = MeshSession::new(base);
        drive(&mut session, &script);
        let replayed = MeshSession::replay(session.base(), &session.ops()[..session.cursor()])
            .expect("journalled ops replay");
        prop_assert_eq!(replayed.encoded(), session.mesh().encoded());
    }

    /// Two runs of the same script agree byte for byte. In one process this
    /// catches order-dependence (an iteration over a hash container, an
    /// allocation that depends on anything but the op sequence); the
    /// cross-machine half of the claim is structural and is pinned by
    /// `tests/determinism_law.rs`.
    #[test]
    fn two_runs_of_a_script_agree(base in base_mesh(), script in prop::collection::vec(choice(), 1..40)) {
        let mut a = MeshSession::new(base.clone());
        let mut b = MeshSession::new(base);
        drive(&mut a, &script);
        drive(&mut b, &script);
        prop_assert_eq!(a.mesh().encoded(), b.mesh().encoded());
        prop_assert_eq!(a.ops(), b.ops());
    }

    /// Undo to the base and redo to the head, both ending on the exact bytes
    /// they started from — over checkpoint boundaries and evictions.
    #[test]
    fn undo_and_redo_are_inverses(base in base_mesh(), script in prop::collection::vec(choice(), 1..40)) {
        let mut session = MeshSession::new(base.clone());
        drive(&mut session, &script);
        let head = session.mesh().encoded();
        let steps = session.cursor();
        while session.undo() {
            prop_assert_eq!(validate(session.mesh()), Ok(()));
        }
        prop_assert_eq!(session.mesh().encoded(), base.encoded());
        for _ in 0..steps {
            prop_assert!(session.redo());
        }
        prop_assert_eq!(session.mesh().encoded(), head);
    }

    /// The asset round trip: an exported mesh read back and written again is
    /// byte-identical, and the mesh that came back is valid.
    ///
    /// Where it is *not* byte-identical, both halves have to account for
    /// themselves — the writer's report names a reason, the reader's report
    /// names what it did about it — **and one pass is all it gets**: the second
    /// round trip must reproduce the first one's bytes exactly. That is the
    /// surviving form of "open, save, open, save is a no-op" for a mesh the
    /// asset format cannot carry, and it is a stronger claim than the
    /// unconditional one it replaces on those inputs, not a weaker one.
    ///
    /// Exactly two refusals are permitted, and each one has to *prove* it was
    /// entitled: `NoGeometry` only when the mesh has no faces, and
    /// `NonManifoldEdge` only when the writer's own report says why — coincident
    /// distinct vertices (which the reader's exact weld fuses) or a triangulation
    /// diagonal that had to repeat an existing edge. Anything else means the
    /// writer emitted a soup its own reader calls illegal, with nothing to blame.
    /// (`NonManifoldEdge` is a **convergence guard** since Wave D and is not
    /// expected to be reached; the case that used to reach it now arrives in the
    /// `Ok` arm as a repair, which is exactly what this property missed.)
    #[test]
    fn export_is_a_fixed_point(base in base_mesh(), script in prop::collection::vec(choice(), 1..40)) {
        let mut session = MeshSession::new(base);
        drive(&mut session, &script);
        let opts = ExportOptions::default();
        let (a1, report) = to_mesh_asset(session.mesh(), &opts);
        match from_mesh_asset(&a1) {
            Ok(read) => {
                prop_assert_eq!(validate(&read.mesh), Ok(()));
                let (a2, _) = to_mesh_asset(&read.mesh, &opts);
                let e1 = inf_asset::encode(&a1).expect("encodable");
                let e2 = inf_asset::encode(&a2).expect("encodable");
                if e1 == e2 {
                    // The second read is the same mesh as the first, up to
                    // labelling — the canonical-form claim, on real edits.
                    let read2 = from_mesh_asset(&a2).expect("a2 reads back");
                    prop_assert_eq!(read.mesh.canonical(), read2.mesh.canonical());
                } else {
                    // **The third face of the coincidence hazard, found by
                    // P23.4's ops** (P23.3 documented the first two: the read is
                    // refused, or a diagonal repeats an edge). Two kernel
                    // vertices that round to the same `f32` are not a refusal at
                    // all — the reader's exact weld fuses them, the triangles
                    // that used both become degenerate and are *skipped and
                    // counted*, and the mesh comes back legal and smaller.
                    //
                    // The extrude/inset/bevel set makes this ordinary, because
                    // they place new vertices a parameter away from existing
                    // ones and a small enough parameter is nothing in `f32`. It
                    // stays a documented advisory rather than a fix for the
                    // reasons already recorded (nudging geometry falsifies the
                    // model; refusing the export makes a legal intermediate
                    // unsaveable) — but the writer must still have SAID so.
                    prop_assert!(
                        report.coincident_vertices > 0 || report.reused_diagonals > 0,
                        "the round trip moved and the writer's report has nothing \
                         to blame: {:?}",
                        report
                    );
                    // **The fourth face, and it is Wave D's** — found by the I2
                    // fuzzer, not by the wave that opened it. This clause read
                    // `degenerate_triangles_skipped > 0 || welded_positions <
                    // report.vertices` from P23.4 until now: *fusion*, which is
                    // the reader's answer to a COINCIDENCE and to nothing else.
                    // Wave D gave the reader a second answer. A reused diagonal
                    // used to make the read fail with `NonManifoldEdge` — the
                    // arm below, which accepts `reused_diagonals` on its own —
                    // and Wave D turned that refusal into a repair, so those
                    // cases moved up here into `Ok` and landed on an evidence
                    // clause that has no word for what the reader did. Measured
                    // on the case that caught it (a two-sided coincident sheet,
                    // pinned deterministically as
                    // `a_double_sided_sheet_saves_as_two_shells_and_then_holds_still`):
                    // writer `coincident_vertices: 0, reused_diagonals: 2`,
                    // reader `welded_positions: 4 == vertices, degenerate: 0,
                    // non_manifold_splits: 2` — every term false, nothing wrong.
                    // Same class as the Wave-D audit's A11, which found a
                    // *different* gate still aimed at the refusal Wave D had
                    // made impossible.
                    //
                    // The clause the wave should have written, and it is a
                    // PAIRING rather than a longer `||` chain. Simply adding the
                    // three Wave-D counters to the old list would have gone
                    // green — and would have been unfalsifiable: measured, a
                    // reader that detaches sheets *without counting them* still
                    // passes a flat five-way `||` on 256 random cases, because
                    // the cases that move are nearly all the fusion kind and one
                    // of the fusion terms carries the clause. So each advisory is
                    // matched to the symptom it predicts, and neither half can
                    // stand in for the other:
                    //
                    // * `coincident_vertices` → the reader **fused** (positions
                    //   welded away, or a triangle collapsed and was skipped).
                    // * `reused_diagonals` → the reader **repaired an edge** it
                    //   could not hold: a detach, a dropped duplicate, or a
                    //   winding flip.
                    let fused = read.report.degenerate_triangles_skipped > 0
                        || read.report.welded_positions < report.vertices;
                    let edge_repaired = read.report.non_manifold_splits > 0
                        || read.report.duplicate_faces_dropped > 0
                        || read.report.faces_reoriented > 0;
                    prop_assert!(
                        (report.coincident_vertices > 0 && fused)
                            || (report.reused_diagonals > 0 && edge_repaired),
                        "…and the reader did not do what the writer's advisory \
                         predicted: writer {:?}, reader {:?}",
                        report,
                        read.report
                    );
                    // **The arming.** Widening the clause above costs a claim,
                    // so it is replaced here by one the old property never made
                    // at all: whatever the repair, **the round trip settles**.
                    // Iterating `export ∘ import` from `a2` reaches a fixed
                    // point, and every mesh along the way is valid.
                    //
                    // The first draft of this said "settles in ONE pass", which
                    // is the intuitive claim and is false — measured, not
                    // reasoned: a `cube(1.0)` under an 18-op script needs
                    // **three**, and the reason is legible in the reports
                    // (`faces_reoriented` 39 → 1 → 0). The winding repair is not
                    // a projection. Flipping a component changes the winding the
                    // *writer* then emits, which changes what the next read's
                    // walk sees, so a second look can still find one face to
                    // agree — and only then does it stop. That is a fixed point
                    // reached by iteration rather than in one step, and pinning
                    // the wrong one of those would have red-flagged CI on about
                    // one script in three thousand: measured at 20 000 cases,
                    // which is 78× what CI runs per push, so the one-pass version
                    // would have looked green here and gone off in someone
                    // else's wave.
                    //
                    // What the cap still falsifies is everything worse than slow:
                    // a repair that oscillates between two forms, or that mints a
                    // vertex on every pass, never reaches a fixed point and never
                    // will — and it satisfies every clause above while doing it.
                    const SETTLE_CAP: usize = 8;
                    let mut asset = a2;
                    let mut bytes = e2;
                    let mut passes = 0usize;
                    let settled = loop {
                        if passes == SETTLE_CAP {
                            break false;
                        }
                        let again = from_mesh_asset(&asset).map_err(|e| {
                            TestCaseError::fail(format!("pass {passes} does not read back: {e}"))
                        })?;
                        prop_assert_eq!(validate(&again.mesh), Ok(()));
                        let (next, _) = to_mesh_asset(&again.mesh, &opts);
                        let next_bytes = inf_asset::encode(&next).expect("encodable");
                        passes += 1;
                        if next_bytes == bytes {
                            break true;
                        }
                        asset = next;
                        bytes = next_bytes;
                    };

                    prop_assert!(
                        settled,
                        "the round trip never settled: {} passes of \
                         export∘import and the bytes were still moving ({} at \
                         the last one). Writer {:?}",
                        SETTLE_CAP,
                        bytes.len(),
                        report
                    );
                }
            }
            Err(ImportError::NoGeometry) => {
                // Either there was nothing to write, or **every** triangle
                // written collapsed in `f32` and the reader skipped the lot —
                // the coincidence hazard again, in its third symptom. The
                // entitlement is the same one: the writer must have said so.
                prop_assert!(
                    session.mesh().face_count() == 0
                        || report.coincident_vertices > 0
                        || report.reused_diagonals > 0,
                    "the reader found no geometry in an asset written from {} \
                     face(s), with nothing in the report to blame: {:?}",
                    session.mesh().face_count(),
                    report
                );
            }
            Err(ImportError::NonManifoldEdge { .. }) => {
                prop_assert!(
                    report.coincident_vertices > 0 || report.reused_diagonals > 0,
                    "the reader refused an asset with neither coincident vertices \
                     nor a reused diagonal to blame"
                );
            }
            Err(other) => prop_assert!(false, "the writer produced an unreadable asset: {other}"),
        }
    }

    /// Every written vertex is finite, indices are in range, and the bounds
    /// contain the geometry — the properties a consumer that never heard of this
    /// crate (the renderer, the cook, `fracture_mesh`) relies on.
    #[test]
    fn every_written_asset_is_well_formed(base in base_mesh(), script in prop::collection::vec(choice(), 1..40)) {
        let mut session = MeshSession::new(base);
        drive(&mut session, &script);
        let (asset, report) = to_mesh_asset(session.mesh(), &ExportOptions::default());
        // **The CURRENT version, not a literal 2** (wave CHAR1a.3). The number
        // was 2 and `.inf_mesh` went to v3 for the material-slot table; a
        // literal here pins the version this crate happened to write on the day
        // it was authored, which is not the property this arm is about. What it
        // is about is that a DCC export writes a payload the current readers
        // accept, and that is exactly the constant.
        prop_assert_eq!(asset.schema_version, inf_mesh::MeshAsset::CURRENT_VERSION);
        prop_assert_eq!(report.submeshes, asset.submeshes.len());
        for sm in &asset.submeshes {
            prop_assert_eq!(sm.indices.len() % 3, 0);
            prop_assert!(sm.skin.is_empty());
            for &i in &sm.indices {
                prop_assert!((i as usize) < sm.vertices.len());
            }
            for v in &sm.vertices {
                for k in 0..3 {
                    prop_assert!(v.position[k].is_finite());
                    prop_assert!(v.normal[k].is_finite());
                    prop_assert!(v.tangent[k].is_finite());
                    prop_assert!(v.position[k] >= asset.bounds.min[k]);
                    prop_assert!(v.position[k] <= asset.bounds.max[k]);
                }
                prop_assert!(v.tangent[3] == 1.0 || v.tangent[3] == -1.0);
                let n2: f32 = v.normal.iter().map(|c| c * c).sum();
                prop_assert!((n2 - 1.0).abs() < 1e-4, "normal not unit: {:?}", v.normal);
            }
        }
    }

    /// **The amendment law** (Wave D), and it is the journal's determinism law
    /// pointed at the new door: after re-parameterizing an op somewhere in the
    /// middle of a random history, the session is *still* nothing but its ops.
    ///
    /// Two outcomes, no third:
    ///
    /// * **`Ok`** — the mesh must equal what `MeshSession::restore` builds from
    ///   the saved history. That is a genuinely independent check: `restore`
    ///   replays from the base through `crate::ops::apply`, never touching a
    ///   line of `amend`. An `amend` that patched the live mesh without writing
    ///   the op back — or wrote the op back and left a stale mesh — fails here
    ///   and cannot fail anywhere else.
    /// * **`Err`** — the session is byte-identical, ops included. The refusals-
    ///   are-inert law, which the ops have obeyed since P23.3 and which a
    ///   history rewrite is the easiest thing in the crate to break.
    #[test]
    fn an_amendment_either_refuses_inertly_or_leaves_a_pure_function_of_the_ops(
        base in base_mesh(),
        script in prop::collection::vec(choice(), 4..40),
        pick_at in any::<u16>(),
    ) {
        let mut session = MeshSession::new(base);
        drive(&mut session, &script);
        if session.cursor() == 0 {
            return Ok(());
        }
        // Walk from a random start so every op in the history gets reached
        // across the battery, and take the first one that has a perturbation.
        let n = session.cursor();
        let start = pick_at as usize % n;
        let Some((index, amended)) = (0..n).find_map(|k| {
            let i = (start + k) % n;
            perturb(&session.ops()[i]).map(|op| (i, op))
        }) else {
            return Ok(());
        };

        let before_mesh = session.mesh().encoded();
        let before_ops = session.ops().to_vec();
        let before_cursor = session.cursor();
        match session.amend(index, amended) {
            Ok(()) => {
                prop_assert_eq!(validate(session.mesh()), Ok(()));
                prop_assert_eq!(session.cursor(), before_cursor, "the cursor travelled");
                prop_assert_eq!(session.ops().len(), before_ops.len(), "an op appeared or vanished");
                let back = MeshSession::restore(session.save())
                    .map_err(|e| TestCaseError::fail(format!("the amended save does not restore: {e}")))?;
                prop_assert_eq!(back.mesh().encoded(), session.mesh().encoded());
            }
            Err(_) => {
                prop_assert_eq!(session.mesh().encoded(), before_mesh);
                prop_assert_eq!(session.ops(), &before_ops[..]);
                prop_assert_eq!(session.cursor(), before_cursor);
            }
        }
    }

    /// The classifier is not decoration: an op it calls
    /// [`inf_dcc::Amendability::Never`] is refused **before** anything is
    /// replayed, and an op it calls amendable never comes back as
    /// `DifferentKind` or `StructureChanged` for a perturbation that only moved
    /// a parameter.
    #[test]
    fn the_classifier_and_the_perturbation_agree(
        base in base_mesh(),
        script in prop::collection::vec(choice(), 1..24),
    ) {
        let mut session = MeshSession::new(base);
        drive(&mut session, &script);
        for i in 0..session.cursor() {
            let op = session.ops()[i].clone();
            let Some(amended) = perturb(&op) else { continue };
            match inf_dcc::op_amendable(&op) {
                inf_dcc::Amendability::Never => {
                    prop_assert!(
                        inf_dcc::amend_shape_ok(&op, &amended).is_err(),
                        "{} is Never but its perturbation passed gate 1",
                        inf_dcc::op_kind(&op)
                    );
                }
                _ => {
                    prop_assert!(
                        inf_dcc::amend_shape_ok(&op, &amended).is_ok(),
                        "{} refused a parameter-only perturbation at gate 1",
                        inf_dcc::op_kind(&op)
                    );
                }
            }
        }
    }
}

/// Move an op's **parameters** and nothing else, or `None` for an op with no
/// parameter to move.
///
/// Deliberately covers only the ops an author would reach for a slider on. The
/// perturbation is a pure function of the op, so a shrunk proptest case is
/// reproducible.
fn perturb(op: &Op) -> Option<Op> {
    Some(match op {
        Op::AddVertex { position } => Op::AddVertex {
            position: [position[0] + 0.1, position[1], position[2]],
        },
        Op::TranslateVerts { verts, delta } => Op::TranslateVerts {
            verts: verts.clone(),
            delta: [delta[0] + 0.05, delta[1] - 0.02, delta[2]],
        },
        Op::MoveVerts { moves } => Op::MoveVerts {
            moves: moves
                .iter()
                .map(|&(v, d)| (v, [d[0] * 0.5, d[1] * 0.5, d[2] * 0.5]))
                .collect(),
        },
        Op::SplitEdge { half, t } => Op::SplitEdge {
            half: *half,
            // Still strictly inside (0, 1), so the perturbation tests the
            // parameter rather than the range check.
            t: (t * 0.5 + 0.25).clamp(0.05, 0.95),
        },
        Op::ExtrudeFaces { faces, distance } => Op::ExtrudeFaces {
            faces: faces.clone(),
            distance: distance * 0.5,
        },
        Op::ExtrudeEdges { edges, delta } => Op::ExtrudeEdges {
            edges: edges.clone(),
            delta: [delta[0] * 0.5, delta[1] * 0.5, delta[2] * 0.5],
        },
        Op::InsetFaces {
            faces,
            amount,
            individual,
        } => Op::InsetFaces {
            faces: faces.clone(),
            amount: amount * 0.5,
            individual: *individual,
        },
        Op::BevelEdges {
            edges,
            amount,
            segments,
        } => Op::BevelEdges {
            edges: edges.clone(),
            amount: amount * 0.5,
            segments: *segments,
        },
        Op::Mirror { axis, coord } => Op::Mirror {
            axis: *axis,
            coord: coord + 0.3,
        },
        Op::ScaleVerts {
            verts,
            pivot,
            factor,
        } => Op::ScaleVerts {
            verts: verts.clone(),
            pivot: *pivot,
            factor: [factor[0] * 0.9, factor[1], factor[2]],
        },
        Op::RotateVerts {
            verts,
            pivot,
            axis,
            radians,
        } => Op::RotateVerts {
            verts: verts.clone(),
            pivot: *pivot,
            axis: *axis,
            radians: radians * 0.5,
        },
        // …and one that the classifier must REFUSE, so the battery reaches the
        // `Never` arm of `the_classifier_and_the_perturbation_agree` rather than
        // only ever generating amendable ops.
        Op::LoopCut { half, cuts } => Op::LoopCut {
            half: *half,
            cuts: cuts + 1,
        },
        _ => return None,
    })
}
