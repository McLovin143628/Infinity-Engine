//! **The Phase 28 gate** (P28.5, the last batch) — Nanite-native unification, in
//! one file, against the phase's own "Done when".
//!
//! Phase 28's clause, verbatim:
//!
//! > the VisBuffer path shades meshlets with materials resolved from per-pixel
//! > primitive IDs at golden-parity with the forward path, a cluster page carries
//! > its texture tiles in one aligned pack page and a single transaction admits
//! > both (the "high-poly mesh with a blurry texture" artifact is asserted
//! > impossible as a state invariant), one streamer arbitrates vgeom + SVT + VSM
//! > under one budget with one feedback ring, and a scripted 360° whip-pan shows
//! > measurably fewer fallback-frames with the predictor on than off (A/B inside
//! > the gate, counters not pixels).
//!
//! The seven arms, and what each is built to FALSIFY:
//!
//! * **(a) golden-parity, per THE RECORDED CRITERION.** The P28.1 audit recorded
//!   what the parity claim is (`docs/ROADMAP.md`, "THE GATE CRITERION, recorded
//!   for P28.5") and this executes it rather than restating it: the criterion
//!   lives in [`inf_render::visbuffer`] as [`inf_render::parity_verdict`] +
//!   [`inf_render::parity_ok`], the P28.1 nucleus measures with it, and so does
//!   this. **Byte-exact to [`inf_render::PARITY_MAX_STEP`] on the rows a
//!   derivative cannot reach; bounded AND classified on the textured row.** The
//!   thing it falsifies is a resolve that shades from a wrong barycentric or a
//!   wrong gradient — which the population bound alone cannot see, because a
//!   fraction is satisfied equally by a one-pixel mip boundary and by a smear.
//!   The classifier's own falsifier is GPU-free and lives beside it
//!   (`the_classifier_separates_a_mip_boundary_from_a_smear`).
//! * **(b) one transaction admits geometry AND its tiles.** The forbidden state
//!   is *reached* by the coupling-off control and *unreachable* with it on, over
//!   a churn that admits, evicts and re-admits. The oracle re-derives the pairing
//!   by point-sampling triangles, so it shares no code with the cook's uv-bound
//!   derivation — a gate cannot see an error the subsystems share.
//! * **(c) ONE streamer, one budget, one ring.** Want-set conservation against
//!   the want set as this file offered it; floor protection against a set
//!   recorded *before* the transaction; byte conservation against the three
//!   residencies' own `resident_bytes` summed **here** and held against the
//!   arbiter's grant through [`inf_render::StreamReport::within_grant`]; one
//!   stamp domain observed across three crates; one ring, with three consumers
//!   agreeing on the source frame.
//! * **(d) the A/B whip-pan**, through the production doors
//!   ([`inf_render::analytic_floor`], [`inf_render::speculative_wants`],
//!   [`inf_math::dead_reckon`]) and a real residency. Counters, not pixels.
//!   Bit-exact per arm, the two arms diverge, the blur counters strictly fall —
//!   with the OFF leg's anti-vacuity asserted in BOTH directions first, because
//!   "never blurry" makes the reduction a bound on zero and "always blurry" is a
//!   budget problem no predictor may be credited for.
//! * **(e) budgets in the three classes** P26.5 settled: **LOAD** once, against
//!   [`inf_player::budget::LOAD_BUDGET_MS`]; **WORLD** in bytes and counters,
//!   asserted unconditionally because a byte is a byte on every adapter; and
//!   **CLOCK** in milliseconds, only where a millisecond represents a frame and
//!   only on a real device.
//! * **(f) the golden set is pinned by count AND by content digest**, additively.
//!   Phase 28 re-blessed nothing: every claim it makes is state or a counter.
//! * **(g) the ray-query experiment is never load-bearing.** Off by default, no
//!   tier and no preset turns it on, the capability clamp only ever *clears*, and
//!   virtual shadow maps remain the shipped shadow path on High and Medium while
//!   Low keeps CSM.
//!
//! # Device arms and GPU-free arms
//!
//! Only (a) needs an adapter, and (e)'s CLOCK section. Everything else is a pure
//! function of `(committed scene, camera path, budget)` and runs on every CI leg
//! — which is deliberate: the phase's own clause says *counters, not pixels*, and
//! a counter that only runs where there is a GPU is a counter that does not run.
//!
//! # Duplication is the house pattern
//!
//! There is no shared test module in this tree, so every fixture below is this
//! file's own copy — the same rule `phase26_gate` and `phase27_gate` follow. What
//! is emphatically **not** duplicated is any threshold: the parity criterion, the
//! predictor's horizon, the budget ceilings and the lane order are each read from
//! their one definition in the library.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use glam::{DVec3, Quat, Vec3};
use inf_asset::ContentHash;
use inf_core::FRAME_BUDGET_MS;
use inf_player::budget::LOAD_BUDGET_MS;
use inf_render::caps::AdapterCaps;
use inf_render::{
    analytic_floor, arbitrate, justified_mip, ndc_margin, on_screen, parity_ok, parity_verdict,
    projection_scale, screen_diameter_px, speculative_wants, BudgetRequest, Consumer, Coupling,
    EngineRenderer, GpuContext, HeadlessTarget, LightKind, ParityVerdict, RaytraceSettings,
    RenderLight, RenderScene, RenderSettings, RenderTier, RenderView, RingLedger, StreamReport,
    VgeomAsset, VgeomInstance, VgeomMesh, VgeomSettings, VsmAtlasConfig, VsmLightDesc,
    VsmLightHandle, VsmPage, VsmResidency, VsmSettings, VsmWant, VtCoverage, VtTextureSet,
    VtTextures, VtWant, DEFAULT_PREDICT_HORIZON_TICKS, HEADLESS_FORMAT, LANE_FLOOR,
    PARITY_MAX_STEP, PARITY_TEXTURED_MAX_FRACTION, PARITY_TEXTURED_MAX_SOLID_CENTRES,
    PARITY_TEXTURED_MIN_BORDERING, PARITY_UNTEXTURED_MAX_FRACTION, READBACK_LATENCY_FRAMES,
    VT_FEEDBACK_MAX_TILES,
};
use inf_vgeom::{ClusterTexture, ClusterTextureSet, VgeomSource, VgeomStreamBudget, VgeomStreamer};
use inf_vt::{
    full_pyramid, TileCoord, VtPoolConfig, VtResidency, VtTextureDesc, VtTextureHandle,
    VT_PRIORITY_FLOOR,
};

// ══ shared vocabulary ═══════════════════════════════════════════════════════

/// The asset id the paired fixtures stream under (the streamer keys on a `u128`).
const PAIRED_ASSET: u128 = 0x2805_0000_0000_0001;
/// Texture GUIDs, distinct so a mixed-up pairing is visible rather than lucky.
const TEX_A: u128 = 0xC0C0_0000_0000_0000_0000_0000_0000_0001;
const TEX_B: u128 = 0xC0C0_0000_0000_0000_0000_0000_0000_0002;

fn asset_id(v: u128) -> inf_asset::AssetId {
    inf_asset::AssetId(uuid::Uuid::from_u128(v))
}

/// The two virtual textures the paired fixture's material samples: a 2 048²
/// albedo and a 1 024² second map, so the pairing has to get *two* different
/// pyramids right and a single-texture bug cannot pass by symmetry.
fn descs() -> Vec<(u128, VtTextureDesc)> {
    vec![
        (TEX_A, full_pyramid(2048, 2048, 128, 4, true)),
        (TEX_B, full_pyramid(1024, 1024, 128, 4, false)),
    ]
}

/// A paired `.inf_vmesh` — the shape a cook produces, tangent channel and all,
/// not a reduced stand-in for one.
fn paired_source() -> (VgeomSource, inf_vgeom::VgeomMesh) {
    let mesh = inf_vgeom::test_support::build_grid_tangented(
        24,
        0.3,
        inf_vgeom::test_support::GridNormals::Analytic,
        true,
    );
    let set = ClusterTextureSet {
        textures: descs()
            .iter()
            .map(|(g, d)| ClusterTexture::from_desc(asset_id(*g), d))
            .collect(),
    };
    let src = VgeomSource::from_mesh_paired(&mesh, &set).expect("build the paired image");
    (src, mesh)
}

/// A texture residency holding both of [`descs`], at `budget_bytes`.
fn texture_residency(budget_bytes: u64) -> (VtResidency, BTreeMap<u128, VtTextureHandle>) {
    let (mut res, _adv) = VtResidency::new(VtPoolConfig {
        budget_bytes,
        ..Default::default()
    });
    let mut by_guid = BTreeMap::new();
    for (guid, desc) in descs() {
        by_guid.insert(guid, res.register_texture(desc).expect("the floor fits"));
    }
    (res, by_guid)
}

// ══ (b)'s oracle — the pairing, re-derived a different way ══════════════════
//
// The cook's pairing is a **uv bound per page** turned into a tile rectangle;
// this file's oracle **samples the actual triangles** and places each sample
// with plain arithmetic, reading the materialized mesh and the texture
// descriptor and never the container's tiles section. The two derivations share
// no code, which is the only reason the invariant arm can see a pairing bug at
// all (P27.5's law: a gate cannot see an error the subsystems share).

/// Place a uv on a tile grid. Written out rather than called, so this file's
/// arithmetic is its own.
fn tile_of(u: f32, n: u32) -> u32 {
    let w = u - u.floor(); // the sampler's wrap, spelled here
    ((w * n as f32) as u32).min(n - 1)
}

/// **The mip rule, re-derived**: a LOD level halves a page's triangles and a mip
/// level quarters its texels, so one mip is worth two LOD levels; the root page
/// (`lod == u32::MAX`, spanning every level) takes the coarsest level there is.
fn oracle_mip(lod: u32, mip_count: u32) -> u32 {
    let coarsest = mip_count - 1;
    if lod == u32::MAX {
        coarsest
    } else {
        (lod / 2).min(coarsest)
    }
}

/// Every tile a page's geometry actually samples, derived by walking its
/// triangles — never by reading the page's tiles section.
fn oracle_tiles(
    src: &VgeomSource,
    mesh: &inf_vgeom::VgeomMesh,
    page: usize,
) -> BTreeSet<(u128, TileCoord)> {
    let mut out = BTreeSet::new();
    let entry = src.pages()[page];
    // The page's global meshlet indices, decoded here rather than cast: the
    // section is `&[u8]` and this file has no `bytemuck`, which is no loss — a
    // little-endian `u32` walk is one line and is the oracle's own arithmetic.
    let Some(globals) = src.with_page_sections(page, |s| {
        s.indices
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect::<Vec<u32>>()
    }) else {
        return out;
    };
    for g in globals {
        let ml = &mesh.meshlets[g as usize];
        for t in 0..ml.triangle_count as usize {
            let tri = mesh.triangle(g as usize, t);
            let uvs: Vec<[f32; 2]> = tri.iter().map(|&v| mesh.vertices[v as usize].uv).collect();
            let centroid = [
                (uvs[0][0] + uvs[1][0] + uvs[2][0]) / 3.0,
                (uvs[0][1] + uvs[1][1] + uvs[2][1]) / 3.0,
            ];
            for uv in uvs.iter().chain(std::iter::once(&centroid)) {
                for (guid, desc) in descs() {
                    let mip = oracle_mip(entry.lod, desc.mips.len() as u32);
                    let m = &desc.mips[mip as usize];
                    out.insert((
                        guid,
                        TileCoord::new(mip, tile_of(uv[0], m.tiles_x), tile_of(uv[1], m.tiles_y)),
                    ));
                }
            }
        }
    }
    out
}

/// One frame of the page-in, with the virtual-texture transaction in the middle
/// — the order the renderer runs and the only order under which a resident
/// page's tiles are protected *between* transactions.
///
/// `couple` off is the pre-P28.2 arrangement: geometry streams, textures stream,
/// nothing joins them, and the seat is unconditional. It is the control arm (b)
/// requires to reach the forbidden state.
fn couple_step(
    streamer: &mut VgeomStreamer,
    res: &mut VtResidency,
    by_guid: &BTreeMap<u128, VtTextureHandle>,
    src: &VgeomSource,
    threshold: f32,
    couple: bool,
) -> u32 {
    // 1. the geometry half.
    let plan = streamer.plan(&[inf_vgeom::VgeomWant {
        asset: PAIRED_ASSET,
        source: src,
        threshold,
    }]);

    // 2. the tiles every resident page samples, read off the container — through
    //    the renderer's own staleness filter, so an address the registered image
    //    does not have is uncoupled rather than fatal.
    let mut page_tiles: BTreeMap<usize, Vec<(u128, TileCoord)>> = BTreeMap::new();
    let resident_now = streamer
        .residency(PAIRED_ASSET)
        .map_or(0, |r| r.resident_pages());
    for page in 0..resident_now {
        let refs = src
            .with_page_sections(page, |s| {
                s.tile_refs()
                    .iter()
                    .map(|t| (t.texture().uuid().as_u128(), t.coord()))
                    .filter(|(g, t)| by_guid.get(g).is_some_and(|h| res.can_address(*h, *t)))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        page_tiles.insert(page, refs);
    }

    // 3. ONE virtual-texture transaction, floor priority, in one want set.
    let mut wants: Vec<VtWant> = Vec::new();
    if couple {
        for refs in page_tiles.values() {
            for (guid, tile) in refs {
                if let Some(h) = by_guid.get(guid) {
                    wants.push(VtWant::new(*h, *tile));
                }
            }
        }
    }
    let _txn = res.apply_wants(&wants);

    // 4. pair, or hand the page back.
    let page_in = streamer.pair(plan, |_asset, page| {
        if !couple {
            return Some(Vec::new());
        }
        let refs = page_tiles.get(&page)?;
        let mut seated = Vec::with_capacity(refs.len());
        for (guid, tile) in refs {
            let h = by_guid.get(guid)?;
            if !res.is_resident(*h, *tile) {
                return None;
            }
            seated.push((*guid, *tile));
        }
        Some(seated)
    });
    // Every page handed back carries both halves, and the halves belong to each
    // other — the type makes the pair inseparable; this reads it back, so a
    // `pair` that handed out somebody else's tiles is visible here.
    if couple {
        for p in page_in.pages() {
            let want = page_tiles
                .get(&p.geometry().page)
                .expect("a page the want pass never saw");
            assert_eq!(
                p.tiles(),
                want.as_slice(),
                "page {} carries tiles that are not its own",
                p.geometry().page
            );
        }
    }
    page_in.retracted
}

/// The invariant, checked against the WORLD: for every page the streamer says is
/// resident, every tile the geometry samples is in the atlas. Returns the number
/// of `(page, tile)` pairs it checked — the anti-vacuity number.
fn assert_invariant(
    streamer: &VgeomStreamer,
    res: &VtResidency,
    by_guid: &BTreeMap<u128, VtTextureHandle>,
    src: &VgeomSource,
    mesh: &inf_vgeom::VgeomMesh,
    at: &str,
) -> usize {
    let mut checked = 0usize;
    let resident = streamer
        .residency(PAIRED_ASSET)
        .map_or(0, |r| r.resident_pages());
    for page in 0..resident {
        for (guid, tile) in oracle_tiles(src, mesh, page) {
            assert!(
                res.is_resident(by_guid[&guid], tile),
                "{at}: page {page} is resident and the tile it samples is not — \
                 texture {guid:#x}, mip {} ({}, {}). This is the \"high-poly mesh, \
                 blurry texture\" state, and it is supposed to be unreachable.",
                tile.mip,
                tile.x,
                tile.y
            );
            checked += 1;
        }
    }
    checked
}

// ══ (a) the VisBuffer parity fixture ════════════════════════════════════════

const W: u32 = 320;
const H: u32 = 180;
const VIS_ASSET: u128 = 0x2805_0000_0715_0000;

fn gpu_or_skip(what: &str) -> Option<GpuContext> {
    match GpuContext::headless() {
        Ok(gpu) => Some(gpu),
        Err(e) => {
            eprintln!("SKIP phase28_gate ({what}): no GPU adapter ({e})");
            None
        }
    }
}

/// A 16 × 16 displaced grid — 512 triangles, several meshlets, a real DAG, and
/// the coarseness the P28.1 nucleus settled on: a 48 × 48 grid at this viewport
/// puts ~10 pixels under a triangle of which 81.6 % are rim, so the interior
/// population the criterion is measured over would be a tenth of what it should
/// be.
fn vis_mesh() -> Arc<VgeomMesh> {
    Arc::new(inf_vgeom::test_support::dense_grid_mesh(16))
}

/// Instances of one asset stood up in world XY, spread in x and stepped back in
/// z — the z step is load-bearing: near-coplanar surfaces at one depth z-fight,
/// and a 4× MSAA depth buffer and a 1× one resolve a z-fight differently, which
/// reads exactly like a resolve defect and is a fixture defect.
fn vis_scene(variants: usize, lights: bool, vt: VtTextureSet) -> RenderScene {
    let m = vis_mesh();
    let mut sc = RenderScene {
        grid_enabled: false,
        vgeom_assets: vec![VgeomAsset::from_mesh(VIS_ASSET, &m).expect("index the vmesh")],
        ..Default::default()
    };
    let standing = Quat::from_rotation_x(std::f32::consts::FRAC_PI_2);
    for i in 0..variants {
        let t = i as f32 / variants.max(1) as f32;
        let mut inst = VgeomInstance::lit(
            VIS_ASSET,
            DVec3::new(
                (i as f64 - (variants as f64 - 1.0) * 0.5) * 2.4,
                0.0,
                -(i as f64) * 2.0,
            ),
            standing,
            Vec3::splat(2.2),
            [0.2 + 0.7 * t, 0.65 - 0.4 * t, 0.30 + 0.5 * t, 1.0],
            i as u32 + 1,
        );
        inst.metallic = 0.1 + 0.8 * t;
        inst.roughness = 0.15 + 0.7 * (1.0 - t);
        inst.vt = vt;
        sc.vgeom_instances.push(inst);
    }
    if lights {
        sc.lights.push(RenderLight {
            kind: LightKind::Directional,
            color: [1.0, 0.97, 0.9],
            intensity: 3.0,
            direction: Vec3::new(0.35, 0.55, 0.75).normalize(),
            position: DVec3::ZERO,
            range: 0.0,
            ..RenderLight::default()
        });
        sc.lights.push(RenderLight {
            kind: LightKind::Point,
            color: [0.4, 0.7, 1.0],
            intensity: 8.0,
            position: DVec3::new(2.0, 1.5, 3.0),
            range: 20.0,
            ..RenderLight::default()
        });
        sc.lights.push(RenderLight {
            kind: LightKind::Spot,
            color: [1.0, 0.6, 0.3],
            intensity: 10.0,
            position: DVec3::new(-2.0, 2.0, 4.0),
            direction: Vec3::new(0.3, -0.4, -0.8).normalize(),
            range: 25.0,
            inner_cos: 30f32.to_radians().cos(),
            outer_cos: 40f32.to_radians().cos(),
            ..RenderLight::default()
        });
    }
    sc.mark_dirty();
    sc
}

fn vis_view() -> RenderView {
    RenderView {
        origin: inf_math::FloatingOrigin::new(DVec3::ZERO),
        eye_world: DVec3::new(0.0, 0.0, 7.0),
        forward: Vec3::NEG_Z,
        up: Vec3::Y,
        fov_y: 60f32.to_radians(),
        near: 0.05,
        width: W,
        height: H,
        ortho: None,
    }
}

/// Meshlet settings with occlusion off — two-pass HZB is a temporal feature and
/// this arm is about one frame's shading. Its contract is purely subtractive, so
/// it cannot change what either path shades.
fn vis_settings(visbuffer: bool) -> RenderSettings {
    RenderSettings {
        vgeom: VgeomSettings {
            enabled: true,
            occlusion: false,
            two_pass: false,
            visbuffer,
            ..VgeomSettings::default()
        },
        ..RenderSettings::default()
    }
}

struct VisFrame {
    rgba: Vec<u8>,
    ids: Option<inf_render::passes::visbuffer::VisImage>,
    audit: inf_render::passes::visbuffer::VisAudit,
}

fn vis_render(gpu: &GpuContext, sc: &RenderScene, st: RenderSettings, ids: bool) -> VisFrame {
    let target = HeadlessTarget::new(gpu, W, H);
    let mut r = EngineRenderer::new(gpu, HEADLESS_FORMAT);
    r.set_settings(st);
    r.set_visbuffer_readback(ids);
    let v = vis_view();
    r.render(gpu, sc, &v, &target.view, (W, H));
    VisFrame {
        rgba: target.read_rgba(gpu).expect("readback"),
        ids: if ids { r.read_visbuffer(gpu) } else { None },
        audit: r.vis_audit(),
    }
}

/// One row of the criterion: render both paths, check the visibility path was
/// actually taken, and hold the comparison to [`parity_ok`].
///
/// Every guard here is an anti-vacuity guard and they come first: a refused
/// frame renders through the forward path (so the row would compare the forward
/// path with itself), an off-screen fixture makes the comparison about the sky,
/// and a thin interior population is a parity claim over a handful of pixels.
fn parity_row(gpu: &GpuContext, label: &str, sc: &RenderScene) -> ParityVerdict {
    let fwd = vis_render(gpu, sc, vis_settings(false), false);
    let res = vis_render(gpu, sc, vis_settings(true), true);
    assert_eq!(
        res.audit.refused(),
        0,
        "{label}: the visibility path was refused ({:?})",
        res.audit
    );
    assert!(
        res.audit.frames > 0,
        "{label}: no frame rasterized a visibility buffer, so this row compared \
         the forward path with itself"
    );
    let ids = res.ids.as_ref().expect("readback was on");
    let covered = ids.covered();
    assert!(
        covered > (W * H / 20) as usize,
        "{label}: the visibility buffer covers only {covered} of {} pixels — the \
         fixture is not on screen and the comparison is about the sky",
        W * H
    );
    let interior = ids.interior();
    let v = parity_verdict(&fwd.rgba, &res.rgba, ids.width, &interior);
    assert!(
        v.interior > (W * H / 100) as usize,
        "{label}: only {} interior pixels; a parity claim over a handful of \
         pixels is not one",
        v.interior
    );
    println!("phase28 parity {label}: {}", v.summary());
    parity_ok(&v, false).unwrap_or_else(|e| panic!("{label}: {e}"));
    v
}

// ══ (a) ═════════════════════════════════════════════════════════════════════

/// **(a) The VisBuffer path shades at golden-parity with the forward path, per
/// the criterion the P28.1 audit RECORDED.**
///
/// The criterion is not restated here. It is
/// [`inf_render::parity_verdict`] + [`inf_render::parity_ok`], one definition,
/// which `crates/inf-render/tests/visbuffer_parity.rs` measures with and which
/// this gate executes — so the phase-level claim and the nucleus cannot drift
/// apart, and a loosened bound fails in both places at once.
///
/// Two classes of row, because the criterion has two halves:
///
/// * **rows a derivative cannot reach** — an unlit scalar material and the full
///   three-kind analytic light loop. These shade from per-instance constants, and
///   a constant does not care what its screen derivative is, so the bound is
///   byte-exactness to [`PARITY_MAX_STEP`]: the two paths agree to within 8-bit
///   rounding or they do not agree.
/// * **the textured row** — the one consumer of the gradients, where a wrong
///   derivative becomes a wrong mip and a visibly wrong texel. Here the two are
///   *not* expected to agree exactly (the forward path's `dpdx` is a first-order
///   quad difference, the resolve's is the exact analytic derivative), so the
///   claim is bounded AND classified: the disagreement must be a **curve**, not a
///   region. A fraction bound alone is satisfied by both.
///
/// What it falsifies, measured by the P28.1 ledger's own mutation matrix:
/// transposing the resolve's `d_dx`/`d_dy` puts the worst delta at 60+ of 255 and
/// the differing fraction past a third; scaling the uv x-gradient 4× takes the
/// bordering class to **0 %** and the solid-centre count to **1 320**.
///
/// Device-gated: skips cleanly with no adapter, like every GPU path in this tree.
#[test]
fn the_visbuffer_path_shades_at_parity_with_the_forward_path() {
    let Some(gpu) = gpu_or_skip("the parity arm") else {
        return;
    };
    println!(
        "phase28 parity criterion: step <= {PARITY_MAX_STEP}, untextured population \
         <= {:.0} %, textured population < {:.0} %, bordering >= {:.0} %, solid \
         centres <= {:.0} %",
        100.0 * PARITY_UNTEXTURED_MAX_FRACTION,
        100.0 * PARITY_TEXTURED_MAX_FRACTION,
        100.0 * PARITY_TEXTURED_MIN_BORDERING,
        100.0 * PARITY_TEXTURED_MAX_SOLID_CENTRES,
    );

    // **THE CRITERION IS PINNED, not merely printed** (P28.5 audit).
    //
    // Hoisting the five constants made the criterion one door. It did not make
    // the door hard to move: mutation-measured at this audit, taking
    // `PARITY_MAX_STEP` from 1 to **60** leaves all seven gate arms and all
    // twelve nucleus arms green, because a *loosened* bound is satisfied by
    // every measurement that satisfied the tight one. The phase's headline claim
    // would then be "the two paths agree to within 60 of 255", which is not the
    // claim, and nothing in the tree would have said so.
    //
    // These five numbers ARE the recorded criterion (`docs/ROADMAP.md`, "THE
    // GATE CRITERION, recorded for P28.5"). Changing one is a deliberate act
    // that re-states what Phase 28 measured, so it fails here first and gets
    // argued for in a ledger.
    assert_eq!(
        PARITY_MAX_STEP, 1,
        "the recorded criterion's step bound moved"
    );
    assert_eq!(
        PARITY_UNTEXTURED_MAX_FRACTION, 0.02,
        "the recorded criterion's untextured population bound moved"
    );
    assert_eq!(
        PARITY_TEXTURED_MAX_FRACTION, 0.12,
        "the recorded criterion's textured population bound moved"
    );
    assert_eq!(
        PARITY_TEXTURED_MIN_BORDERING, 0.80,
        "the recorded criterion's bordering bound moved"
    );
    assert_eq!(
        PARITY_TEXTURED_MAX_SOLID_CENTRES, 0.10,
        "the recorded criterion's solid-centre bound moved"
    );

    // ── the rows a derivative cannot reach ──────────────────────────────────
    parity_row(
        &gpu,
        "untextured/scalar",
        &vis_scene(1, false, VtTextureSet::NONE),
    );
    // Four instances of ONE asset differing in colour, metallic and roughness:
    // the resolve reads its material from the flat instance table through the id
    // it unpacked, so an off-by-one there shades every pixel with a neighbour's
    // material and only a multi-instance row sees it.
    let lit = vis_scene(3, true, VtTextureSet::NONE);
    let mats: BTreeSet<_> = lit
        .vgeom_instances
        .iter()
        .map(|i| {
            (
                i.metallic.to_bits(),
                i.roughness.to_bits(),
                i.color[0].to_bits(),
            )
        })
        .collect();
    assert_eq!(
        mats.len(),
        lit.vgeom_instances.len(),
        "the instances share a material, so this row would pass with the instance \
         field decoded wrong"
    );
    parity_row(&gpu, "dir+point+spot", &lit);

    // ── the textured row ────────────────────────────────────────────────────
    // A prime-period diagonal ramp: a transposed lookup, a mirrored row order or
    // an off-by-a-tile address all read a different value (the fixture-asymmetry
    // law `vt_sampling.rs` states).
    const N: u32 = 256;
    let mut rgba = Vec::with_capacity((N * N * 4) as usize);
    for y in 0..N {
        for x in 0..N {
            rgba.extend_from_slice(&[
                ((x * 7 + y * 3) % 251) as u8,
                ((x * 11 + y * 29) % 241) as u8,
                ((x * 5 + y * 17) % 239) as u8,
                255,
            ]);
        }
    }
    let bytes = inf_material::build_tiled_texture(
        rgba,
        N,
        N,
        inf_material::TextureImportSettings {
            srgb: false,
            generate_mips: true,
            compression: inf_material::TextureCompression::None,
            hdr: false,
        },
    )
    .expect("the fixture tiles")
    .into_bytes();

    let run = |visbuffer: bool| {
        let (mut lib, _) = VtTextures::new(VtPoolConfig {
            format: inf_vt::PageFormat::Rgba8,
            stored_tile_size: inf_vt::STORED_TILE_SIZE,
            budget_bytes: inf_vt::PageFormat::Rgba8.page_bytes(inf_vt::STORED_TILE_SIZE) * 256,
            max_texture_dim: 8192,
            trilinear: false,
            // **Unthrottled** (IB-16): this arm compares the visbuffer feedback
            // producer against the per-surface one, and a deferred page would
            // change both arms' residency for a reason neither is about.
            upload_budget_bytes: 0,
        });
        lib.register_or_record(1, Arc::new(bytes.clone()))
            .unwrap_or_else(|| panic!("the fixture registers: {:?}", lib.refusals()));
        let pools = inf_render::vt::VtPools::new(&gpu.device, &gpu.queue, lib.residency(), false);
        let target = HeadlessTarget::new(&gpu, W, H);
        let mut r = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
        r.set_settings(vis_settings(visbuffer));
        r.set_visbuffer_readback(visbuffer);
        r.set_vt_level(Some((lib, pools)));
        let v = vis_view();
        // Three frames: the registry warms, the floor pages in, and the third
        // draws through a resident set. Both paths get the same three.
        let mut rgba = Vec::new();
        for _ in 0..3 {
            let set = r
                .vt_textures()
                .map(|l| l.set_for(Some(1), None, None))
                .unwrap_or(VtTextureSet::NONE);
            let sc = vis_scene(2, true, set);
            r.render(&gpu, &sc, &v, &target.view, (W, H));
            rgba = target.read_rgba(&gpu).expect("readback");
        }
        (
            rgba,
            if visbuffer {
                r.read_visbuffer(&gpu)
            } else {
                None
            },
            r.vt_pop_in().admits,
        )
    };
    let (fwd, _, fwd_admits) = run(false);
    let (res, ids, res_admits) = run(true);

    // ANTI-VACUITY, before the bound is read: something was actually paged in,
    // and the texture actually modulates the surface. Without both, this row
    // compares two untextured frames and calls it parity.
    assert!(
        fwd_admits > 0 && res_admits > 0,
        "nothing was paged in ({fwd_admits} / {res_admits})"
    );
    let flat = {
        let target = HeadlessTarget::new(&gpu, W, H);
        let mut r = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
        r.set_settings(vis_settings(true));
        r.render(
            &gpu,
            &vis_scene(2, true, VtTextureSet::NONE),
            &vis_view(),
            &target.view,
            (W, H),
        );
        target.read_rgba(&gpu).expect("readback")
    };
    assert_ne!(
        flat, res,
        "the textured frame is byte-identical to the untextured one, so nothing \
         was sampled and this row is about nothing"
    );

    let ids = ids.expect("readback on");
    let v = parity_verdict(&fwd, &res, ids.width, &ids.interior());
    assert!(
        v.interior > 1000,
        "textured: only {} interior pixels",
        v.interior
    );
    println!(
        "phase28 parity textured: {} ({} admits fwd / {} admits res)",
        v.summary(),
        fwd_admits,
        res_admits
    );
    parity_ok(&v, true).unwrap_or_else(|e| panic!("textured: {e}"));

    // The differing count is **printed and not asserted non-zero**, deliberately:
    // zero would mean the two gradient constructions chose the same mip on every
    // interior pixel, which is strictly better than the P28.1 measurement and
    // must not fail a gate. What IS asserted non-zero above is the evidence
    // population — interior pixels, VT admits, and the textured-vs-flat
    // difference — because those are what make the agreement a claim.
    if v.differing == 0 {
        println!(
            "phase28 parity textured: the two gradient constructions agreed on \
             every interior pixel; the class bounds are vacuous on this frame"
        );
    }
}

// ══ (b) ═════════════════════════════════════════════════════════════════════

/// **(b) One transaction admits a cluster page AND its texture tiles, and the
/// "high-poly mesh with a blurry texture" state is unreachable.**
///
/// The claim is not "the code calls both consumers"; it is that **there is no
/// reachable state** in which a meshlet page is in the pools and a tile its
/// materials sample at that page's detail level is not in the atlas. So the arm
/// drives the real machinery — `inf_vgeom::VgeomStreamer`, `inf_vt::VtResidency`
/// and the `pair` door between them — through a churn of camera thresholds, and
/// after **every** transaction re-derives from the geometry what each resident
/// page must hold and asserts the atlas holds it.
///
/// The churn goes coarse → fine → coarse → fine, so pages are admitted, evicted
/// and re-admitted rather than only ever added.
///
/// **The control must REACH the forbidden state.** Without it the invariant is
/// satisfied by a fixture whose analytic floor happens to cover every tile the
/// geometry samples, and would say nothing at all about the coupling. With the
/// cluster tiles removed from the want set — precisely the pre-P28.2 arrangement,
/// two systems with no edge between them — more than half of the sampled tiles
/// must be missing, which is not a rounding-scale miss.
#[test]
fn one_transaction_admits_a_cluster_page_and_its_tiles() {
    let (src, mesh) = paired_source();
    let (mut res, by_guid) = texture_residency(64 * 1024 * 1024);
    let mut streamer = VgeomStreamer::new(VgeomStreamBudget::default());

    let thresholds: [f32; 14] = [
        4.0, 2.0, 1.0, 0.5, 0.25, 0.1, 0.02, 0.005, 0.02, 0.25, 2.0, 8.0, 0.5, 0.001,
    ];
    let mut peak = 0usize;
    let mut checks = 0usize;
    for (i, t) in thresholds.iter().enumerate() {
        let retracted = couple_step(&mut streamer, &mut res, &by_guid, &src, *t, true);
        assert_eq!(
            retracted, 0,
            "step {i}: a comfortable budget retracted a page"
        );
        checks += assert_invariant(
            &streamer,
            &res,
            &by_guid,
            &src,
            &mesh,
            &format!("step {i} (threshold {t})"),
        );
        peak = peak.max(
            streamer
                .residency(PAIRED_ASSET)
                .map_or(0, |r| r.resident_pages()),
        );
    }

    // ANTI-VACUITY, both directions: the churn really did stream, and the oracle
    // really did have tiles to check.
    assert!(
        peak >= 3,
        "the churn never got past {peak} resident pages — it is not exercising \
         residency"
    );
    assert_eq!(
        streamer.residency(PAIRED_ASSET).map(|r| r.resident_pages()),
        Some(src.pages().len()),
        "the finest threshold must reach full residency"
    );
    assert!(
        checks > 1000,
        "the oracle only checked {checks} (page, tile) pairs"
    );

    // ── THE CONTROL: it must REACH the forbidden state ──────────────────────
    let (mut res, by_guid) = texture_residency(64 * 1024 * 1024);
    let mut streamer = VgeomStreamer::new(VgeomStreamBudget::default());
    for t in [4.0f32, 0.5, 0.02, 0.001] {
        couple_step(&mut streamer, &mut res, &by_guid, &src, t, false);
    }
    let resident = streamer
        .residency(PAIRED_ASSET)
        .map_or(0, |r| r.resident_pages());
    assert!(
        resident > 0,
        "nothing streamed — the control proves nothing"
    );
    let (mut missing, mut total) = (0usize, 0usize);
    for page in 0..resident {
        for (guid, tile) in oracle_tiles(&src, &mesh, page) {
            total += 1;
            missing += usize::from(!res.is_resident(by_guid[&guid], tile));
        }
    }
    println!(
        "phase28 coupling: {checks} (page, tile) pairs held over a {}-step churn \
         at peak {peak} pages; uncoupled, {missing} of {total} sampled tiles are \
         MISSING ({:.1} %)",
        thresholds.len(),
        100.0 * missing as f64 / total.max(1) as f64,
    );
    assert!(
        missing * 2 > total,
        "only {missing} of {total} sampled tiles were missing without the \
         coupling — the invariant above is measuring a fixture, not a mechanism"
    );
}

// ══ (c) the unified streamer ════════════════════════════════════════════════

/// The three residencies plus the coupling and the ring — the world the arbiter
/// divides.
struct Unified {
    geometry: VgeomStreamer,
    texture: VtResidency,
    shadow: VsmResidency,
    by_guid: BTreeMap<u128, VtTextureHandle>,
    coupling: Coupling<(u128, usize), (u128, TileCoord)>,
    ring: RingLedger,
    /// A 2 048² texture the mesh never samples — the competing **refinement**
    /// class. Without one the only wants in the pool are floor wants, every miss
    /// is deferred rather than seated, and "nothing a floor protects was evicted"
    /// is a bound on zero evictions.
    decoy: Option<VtTextureHandle>,
    dropped: u64,
    peak_bytes: u64,
    contested: u32,
}

/// What one unified step offered and what came back — this file's own record,
/// kept beside the transaction so the oracle never asks the residency what it
/// was asked for.
struct UnifiedStep {
    offered: BTreeSet<(u32, TileCoord)>,
    floor_before: BTreeSet<(u32, TileCoord)>,
    admits: Vec<(u32, u32, TileCoord)>,
    evicts: Vec<(u32, u32, TileCoord)>,
    shadow_admits: usize,
}

impl Unified {
    fn new(texture_budget: u64, atlas_pages: u64, decoy: bool) -> Self {
        let (mut texture, by_guid) = texture_residency(texture_budget);
        let (mut shadow, _adv) = VsmResidency::new(VsmAtlasConfig {
            budget_bytes: VsmAtlasConfig::default().page_bytes() * atlas_pages,
            ..Default::default()
        });
        shadow
            .register_light(VsmLightDesc::clipmap(3, 8))
            .expect("a three-level clipmap");
        let decoy = decoy.then(|| {
            texture
                .register_texture(full_pyramid(2048, 2048, 128, 4, true))
                .expect("the floor fits")
        });
        Self {
            geometry: VgeomStreamer::new(VgeomStreamBudget::default()),
            texture,
            shadow,
            by_guid,
            coupling: Coupling::new(),
            ring: RingLedger::default(),
            decoy,
            dropped: 0,
            peak_bytes: 0,
            contested: 0,
        }
    }

    fn decoy_wants(&self) -> Vec<VtWant> {
        let Some(h) = self.decoy else {
            return Vec::new();
        };
        let m = self.texture.desc(h).expect("registered").mips[0];
        (0..m.tiles_y)
            .flat_map(|y| (0..m.tiles_x).map(move |x| (x, y)))
            .map(|(x, y)| VtWant::refine(h, TileCoord::new(0, x, y)))
            .collect()
    }

    /// Bytes the three hold between them — summed **here** rather than read off
    /// a report, so the conservation claim is this file's own measurement.
    fn resident_bytes(&self) -> [u64; 3] {
        [
            self.geometry.stats().resident_bytes,
            self.texture.resident_bytes(),
            self.shadow.resident_bytes(),
        ]
    }

    /// The live floors: the meshlet streamer's always-resident page 0s and the
    /// virtual texture's pinned roots. The shadow atlas has none by design — a
    /// page nothing marked is a page nothing reads.
    fn floors(&self) -> [u64; 3] {
        [
            self.geometry
                .assets()
                .map(|(_, r)| r.pages().first().map_or(0, |p| p.resident_bytes()))
                .sum(),
            self.texture.floor_bytes(),
            0,
        ]
    }

    fn step(&mut self, src: &VgeomSource, threshold: f32, frame: u64, couple: bool) -> UnifiedStep {
        let plan = self.geometry.plan(&[inf_vgeom::VgeomWant {
            asset: PAIRED_ASSET,
            source: src,
            threshold,
        }]);

        // The coupling, rebuilt from the container for the pages resident now.
        self.coupling.clear();
        let resident = self
            .geometry
            .residency(PAIRED_ASSET)
            .map_or(0, |r| r.resident_pages());
        for page in 0..resident {
            let refs = src
                .with_page_sections(page, |s| {
                    s.tile_refs()
                        .iter()
                        .map(|t| (t.texture().uuid().as_u128(), t.coord()))
                        .filter(|(g, t)| {
                            self.by_guid
                                .get(g)
                                .is_some_and(|h| self.texture.can_address(*h, *t))
                        })
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            self.coupling.couple((PAIRED_ASSET, page), refs);
        }

        // ONE texture transaction: the coupling's wants at the floor lane, plus
        // whatever the feedback class is asking for.
        let mut offered: Vec<VtWant> = if couple {
            self.coupling
                .wants(LANE_FLOOR)
                .into_iter()
                .filter_map(|(_, (g, t))| self.by_guid.get(&g).map(|h| VtWant::new(*h, t)))
                .collect()
        } else {
            Vec::new()
        };
        offered.extend(self.decoy_wants());
        // **The FLOOR lane only.** A resident *refinement* is exactly what a
        // floor miss is now allowed to take (P28.3's lane walk), so folding one
        // in here would make this arm assert the defect that batch removed.
        let floor_before: BTreeSet<(u32, TileCoord)> = offered
            .iter()
            .filter(|w| w.priority == VT_PRIORITY_FLOOR)
            .filter(|w| self.texture.is_resident(w.texture, w.tile))
            .map(|w| (w.texture.0, w.tile))
            .collect();
        let txn = self.texture.apply_wants(&offered);
        self.contested += u32::from(txn.deferred > 0);

        // Pair, or hand the page back.
        let coupling = &self.coupling;
        let texture = &self.texture;
        let by_guid = &self.by_guid;
        let page_in = self.geometry.pair(plan, |asset, page| {
            if !couple {
                return Some(Vec::new());
            }
            if !coupling.has_group(&(asset, page)) {
                return None;
            }
            let mut seated = Vec::new();
            for &(g, t) in coupling.members(&(asset, page)) {
                let h = by_guid.get(&g)?;
                if !texture.is_resident(*h, t) {
                    return None;
                }
                seated.push((g, t));
            }
            Some(seated)
        });
        let _ = page_in;

        // Cross-system aging: a group whose page left stops wanting its members
        // in this same call.
        let live: BTreeSet<(u128, usize)> = self
            .geometry
            .assets()
            .flat_map(|(a, r)| (0..r.resident_pages()).map(move |p| (a, p)))
            .collect();
        self.dropped += self.coupling.retain(|g| live.contains(g)) as u64;

        // The shadow half runs its own transaction against the same stamp
        // domain — a consumer of the budget and of the lane order, not of the
        // coupling.
        let marks: Vec<VsmPage> = (0..6)
            .map(|k| VsmPage::flat(0, ((frame as usize + k) % 8) as u32, (k % 8) as u32))
            .collect();
        let shadow_txn = self.shadow.apply_wants(
            &marks
                .iter()
                .map(|p| VsmWant::new(VsmLightHandle(0), *p))
                .collect::<Vec<_>>(),
        );

        // **ONE RING.** All three consumers read the same ledger at the same
        // frame, so `readers_agree` is a statement about one domain rather than
        // three sequences that happen to line up.
        for c in Consumer::ALL {
            self.ring.read(c, frame, |_| true);
        }

        let bytes = self.resident_bytes();
        self.peak_bytes = self.peak_bytes.max(bytes.iter().sum::<u64>());
        UnifiedStep {
            offered: offered.iter().map(|w| (w.texture.0, w.tile)).collect(),
            floor_before,
            admits: txn
                .admits
                .iter()
                .map(|a| (a.slot, a.texture.0, a.tile))
                .collect(),
            evicts: txn
                .evicts
                .iter()
                .map(|e| (e.slot, e.texture.0, e.tile))
                .collect(),
            shadow_admits: shadow_txn.admits.len(),
        }
    }

    /// The renderer's own audit line, assembled from the live numbers.
    fn report(&self, budget_bytes: u64) -> StreamReport {
        let floors = self.floors();
        let resident = self.resident_bytes();
        StreamReport {
            budget_bytes,
            grant: arbitrate(
                budget_bytes,
                &[
                    BudgetRequest {
                        floor_bytes: floors[0],
                        want_bytes: VgeomStreamBudget::default().budget_bytes,
                    },
                    BudgetRequest {
                        floor_bytes: floors[1],
                        want_bytes: self.texture.capacity_bytes(),
                    },
                    BudgetRequest::want(self.shadow.capacity_bytes()),
                ],
            ),
            resident,
            floors,
            coupled_groups: self.coupling.len(),
            coupled_tiles: self.coupling.member_count(),
            dropped_groups: self.dropped,
            retracted: 0,
            stale_tiles: 0,
            mismatched_textures: 0,
            ring: self.ring,
        }
    }
}

/// A ladder of strictly refining thresholds, so "residency went backwards" can
/// only ever mean the arbiter took ground back.
const LADDER: [f32; 8] = [4.0, 0.5, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001];

/// **(c) ONE streamer arbitrates vgeom + SVT + VSM under one budget with one
/// feedback ring.**
///
/// Four conservation claims, each against a quantity the arbiter does not
/// compute — because after P28.3 the three subsystems share a stamp domain, an
/// admission walk and a budget, and a gate cannot see an error its subjects
/// share:
///
/// | claim | the arbiter's mechanism | this arm's oracle |
/// |---|---|---|
/// | want-set conservation | `admit_by_lane`'s misses | the want set **as this file offered it**, kept beside the transaction |
/// | floor protection | lane order + `protected` | a floor set recorded *before* the transaction, checked against the evicts *after* |
/// | byte conservation | `arbitrate`'s water-fill | the three residencies' own `resident_bytes`, summed here |
/// | one stamp domain | one `static` | three crates' generations, compared for strict interleaving |
/// | one ring | one [`RingLedger`] | three consumers reading it, required to agree on the source frame |
///
/// Every bound carries its anti-vacuity counter FIRST: nothing may be read off
/// `evicts` until the pool has been proved to run out, off `admits` until
/// something was admitted, or off the grant until all three consumers held bytes.
#[test]
fn one_streamer_arbitrates_the_three_under_one_budget_with_one_ring() {
    let (src, _mesh) = paired_source();

    // A pool tight enough that the pairing genuinely contests slots, plus the
    // competing refinement class that gives it something evictable to take.
    let mut s = Unified::new(768 * 1024, 8, true);
    let (mut admitted, mut evicted, mut shadow_admitted) = (0usize, 0usize, 0usize);
    let mut coupled_checked = 0usize;
    for (i, t) in LADDER.iter().enumerate() {
        let step = s.step(&src, *t, i as u64, true);

        // ── WANT-SET CONSERVATION ───────────────────────────────────────────
        for (slot, tex, tile) in &step.admits {
            // A pinned root is admitted by REGISTRATION and never by a want, so
            // it is excluded by NAMING what it is — not by a blanket escape.
            assert!(
                step.offered.contains(&(*tex, *tile)) || s.texture.slot_is_root(*slot),
                "step {i}: slot {slot} took ({tex}, {tile:?}) and nothing wanted it"
            );
        }
        // ── FLOOR PROTECTION ────────────────────────────────────────────────
        for (slot, tex, tile) in &step.evicts {
            assert!(
                !step.floor_before.contains(&(*tex, *tile)),
                "step {i}: slot {slot} held a resident FLOOR want ({tex}, {tile:?}) \
                 and the transaction took it"
            );
        }
        admitted += step.admits.len();
        evicted += step.evicts.len();
        shadow_admitted += step.shadow_admits;
        assert!(
            step.shadow_admits <= 6,
            "step {i}: the shadow atlas admitted more pages than were marked"
        );

        // ── THE COUPLING HOLDS, through the library's own predicate ─────────
        let by_guid = &s.by_guid;
        let texture = &s.texture;
        let broken = inf_render::stream::breaches(&s.coupling, |(g, tile)| {
            by_guid
                .get(g)
                .is_some_and(|h| texture.is_resident(*h, *tile))
        });
        assert!(
            broken.is_empty(),
            "step {i}: {} coupled (page, tile) pairs are not resident — the first \
             is {:?}",
            broken.len(),
            broken.first()
        );
        coupled_checked += s.coupling.member_count();
    }

    // ── ANTI-VACUITY, before any bound is believed ──────────────────────────
    assert!(admitted > 0, "nothing was ever admitted");
    assert!(shadow_admitted > 0, "the shadow atlas never took a page");
    assert!(
        evicted > 0 && s.contested > 0,
        "the pool never ran out ({evicted} evictions, {} contested steps) — the \
         floor-protection bound above is a bound on nothing",
        s.contested
    );
    assert!(
        s.dropped > 0,
        "no coupled group was ever dropped, so nothing aged out and the retain is \
         untested"
    );
    // Summed over the ladder, not read at the end: the coupling is rebuilt from
    // the resident set every step, so its final size is the size of ONE frame's
    // pairing and a bound on it would be a bound on the last threshold.
    assert!(
        coupled_checked > 100,
        "only {coupled_checked} (page, tile) pairs were ever coupled and checked \
         — too few for the invariant above to mean anything"
    );

    // ── BYTE-BUDGET CONSERVATION, through the shipped audit ─────────────────
    // The unified ceiling is the sum of the three requests, so the arbiter is an
    // identity here and the RESIDENCY bound is the load-bearing half.
    let total = VgeomStreamBudget::default().budget_bytes
        + s.texture.capacity_bytes()
        + s.shadow.capacity_bytes();
    let report = s.report(total);
    println!("phase28 {}", report.summary());
    let grant = report
        .grant
        .as_ref()
        .expect("the live floors fit the sum of the three requests");
    assert_eq!(
        grant.total(),
        total,
        "the arbiter did not hand out the whole ceiling it was an identity on"
    );
    for c in Consumer::ALL {
        assert!(
            report.resident_of(c) > 0,
            "{} held nothing all run — its bound is a bound on zero",
            c.label()
        );
        assert!(
            report.resident_of(c) >= report.floors[c.index()],
            "{} fell below its own floor",
            c.label()
        );
    }
    assert!(
        report.within_grant(),
        "a consumer is outside the grant the live floors justify: {:?} against \
         {:?}",
        report.resident,
        grant.bytes
    );
    assert_eq!(
        report.resident_bytes(),
        report.resident.iter().sum::<u64>(),
        "the combined number is not the sum of its three readable parts"
    );

    // ── ONE FEEDBACK RING ───────────────────────────────────────────────────
    // Three consumers, one ledger, one source frame. Both counters are asserted:
    // the hits prove the ring delivered, and the misses prove it is a ring with a
    // latency rather than a passthrough that always says yes.
    assert!(
        report.ring.hits() > 0,
        "no consumer ever read a frame off the ring"
    );
    assert!(
        report.ring.misses() > 0,
        "the ring never missed, so it has no latency and is not a ring"
    );
    assert!(
        report.ring.readers_agree(),
        "the three consumers last read DIFFERENT source frames — that is three \
         rings wearing one name: {}",
        report.ring.summary()
    );
    // **THE CONTROL, because `readers_agree` over three consumers walked in
    // lockstep is close to true by construction.** Take a copy of the live
    // ledger, let ONE consumer read a frame ahead of the others, and the
    // predicate must go false. Without this the assertion above is satisfied by a
    // predicate that cannot fail, which is the vacuous shape this gate exists to
    // refuse.
    let mut skewed = report.ring;
    skewed.read(Consumer::Texture, LADDER.len() as u64 + 4, |_| true);
    assert!(
        !skewed.readers_agree(),
        "one consumer read a LATER source frame than the other two and the ring \
         still called them agreed — the predicate cannot fail, so the assertion \
         above measures nothing: {}",
        skewed.summary()
    );

    // ── ONE STAMP DOMAIN, observed across three crates ──────────────────────
    // Interleave: texture, shadow, geometry, texture — and the four generations
    // must come out in that order. With three domains the sequences are
    // independent and a generation minted by one crate says nothing about
    // another's; with one, a generation minted *after* another is strictly
    // greater, which is the only thing about a stamp that may be compared.
    let (texture, _) = texture_residency(4 * 1024 * 1024);
    let t0 = texture.layout_generation();
    let (mut shadow, _adv) = VsmResidency::new(VsmAtlasConfig::default());
    shadow
        .register_light(VsmLightDesc::clipmap(3, 8))
        .expect("registers");
    let s0 = shadow.layout_generation();
    let mut geometry = VgeomStreamer::new(VgeomStreamBudget::default());
    geometry.plan(&[inf_vgeom::VgeomWant {
        asset: PAIRED_ASSET,
        source: &src,
        threshold: 0.01,
    }]);
    let g0 = geometry
        .residency(PAIRED_ASSET)
        .expect("planned")
        .generation();
    let (texture2, _) = texture_residency(4 * 1024 * 1024);
    let t1 = texture2.layout_generation();
    assert!(
        t0 < s0 && s0 < g0 && g0 < t1,
        "the three crates' stamps do not interleave — three domains, not one: \
         texture {t0}, shadow {s0}, geometry {g0}, texture again {t1}"
    );
    println!(
        "phase28 arbiter: {admitted} admits / {evicted} evicts over {} steps \
         ({} contested), {shadow_admitted} shadow pages, {coupled_checked} coupled \
         pairs checked, {} groups dropped; stamps {t0} < {s0} < {g0} < {t1}",
        LADDER.len(),
        s.contested,
        s.dropped,
    );
}

// ══ (d) the A/B whip-pan ════════════════════════════════════════════════════

/// Ticks the whole run lasts — 260 at the shipped 60 Hz step is 4.3 s.
const TICKS: u64 = 260;
/// Ticks the camera stands still before the whip, so the history is full and
/// residency has settled. A predictor measured from a cold pool would be
/// measuring the warm-up.
const WARM: u64 = 40;
/// Ticks the 360° sweep takes: 3 s, i.e. **120°/s** average.
const SWEEP: u64 = 180;
/// Ticks of the sweep spent accelerating, and the same again decelerating. Not
/// decoration: a constant-rate turn is the case dead reckoning is exactly right
/// about, and a fixture made only of those would measure a tautology.
const RAMP: u64 = 30;
/// **The shipped horizon**, read from the library and never restated — **0**
/// since P28.5's lead-time ruling, i.e. the committed pose with the speculative
/// LANE still running. The lane is the win, not the lead.
const HORIZON: u32 = DEFAULT_PREDICT_HORIZON_TICKS;
/// The pool the fixture runs against — comfortably over the steady demand on
/// purpose. A pool *under* the demand makes every frame a fallback frame in both
/// arms, for a reason no predictor can touch; this fixture measures **latency**,
/// so it is deliberately not a budget fixture.
const PAGES: u64 = 800;

/// Surfaces on the ring — **three clusters of four**, not twelve evenly spaced.
/// An even ring keeps the arrival rate constant; clusters make it bursty, which
/// is the shape a prefetcher is for and the shape a fixed lag hurts most.
const CLUSTERS: [f64; 3] = [0.0, 2.0943951023931953, 4.1887902047863905];
const PER_CLUSTER: usize = 4;
const RING_RADIUS: f64 = 14.0;
/// Big enough that the surface's footprint justifies a level with more tiles than
/// the floor's cap, which is the only regime in which the floor and the
/// refinement settle on different mips and the two classes are separable at all.
const SURFACE_RADIUS: f32 = 12.0;

/// **The whip-pan's yaw at `tick`**, radians — a closed form, so the camera at
/// tick `t` does not depend on the camera at `t − 1` and the oracle can seek to
/// any tick without replaying the path.
///
/// Piecewise-constant angular acceleration: still, ramp up, constant, ramp down,
/// still. The constant rate is set so the phases integrate to exactly one
/// revolution.
fn yaw_at(tick: u64) -> f64 {
    let hold = (SWEEP - 2 * RAMP) as f64;
    let w = std::f64::consts::TAU / (RAMP as f64 + hold);
    if tick <= WARM {
        return 0.0;
    }
    let u = (tick - WARM).min(SWEEP) as f64;
    let r = RAMP as f64;
    if u < r {
        w * u * u / (2.0 * r)
    } else if u < r + hold {
        w * r * 0.5 + w * (u - r)
    } else {
        let v = u - r - hold;
        w * r * 0.5 + w * hold + w * (v - v * v / (2.0 * r))
    }
}

/// The camera at `tick`, with a slow drift so the path is not a pure rotation
/// about one point. Portable trig, because the pose feeds a want set two hosts
/// are claimed to agree about (the P14 law).
fn whip_view(tick: u64) -> RenderView {
    let a = yaw_at(tick);
    RenderView {
        origin: inf_math::FloatingOrigin::new(DVec3::ZERO),
        eye_world: DVec3::new(0.012 * tick as f64, 1.6, 0.0),
        forward: Vec3::new(inf_math::psin64(a) as f32, 0.0, -inf_math::pcos64(a) as f32)
            .normalize(),
        up: Vec3::Y,
        fov_y: 60_f32.to_radians(),
        near: 0.1,
        width: 1280,
        height: 720,
        ortho: None,
    }
}

/// One 2 048² texture per ring surface, in a real [`VtTextures`] over a real
/// [`VtResidency`].
fn whip_library(pages: u64) -> (VtTextures, Vec<VtTextureHandle>) {
    let (mut lib, _adv) = VtTextures::new(VtPoolConfig {
        budget_bytes: pages * 8192,
        ..Default::default()
    });
    let handles = (0..CLUSTERS.len() * PER_CLUSTER)
        .map(|_| {
            lib.residency_mut()
                .register_texture(full_pyramid(2048, 2048, 128, 4, true))
                .expect("the mandatory floor fits")
        })
        .collect();
    (lib, handles)
}

/// The coverage list — **camera-free**, exactly as `scene_coverage` is, so the
/// committed and predicted cameras are handed the same scene.
fn whip_coverage(handles: &[VtTextureHandle]) -> Vec<VtCoverage> {
    let mut out = Vec::new();
    for (c, base) in CLUSTERS.iter().enumerate() {
        for k in 0..PER_CLUSTER {
            let a = base + (k as f64 - 1.5) * 0.11;
            let h = handles[c * PER_CLUSTER + k];
            out.push(VtCoverage {
                centre: Vec3::new(
                    (RING_RADIUS * inf_math::psin64(a)) as f32,
                    1.6,
                    (-RING_RADIUS * inf_math::pcos64(a)) as f32,
                ),
                radius: SURFACE_RADIUS,
                set: VtTextureSet {
                    albedo: h.0 + 1,
                    ..VtTextureSet::NONE
                },
                vgeom: false,
            });
        }
    }
    out
}

/// **The level every visible surface's footprint justifies, and its tiles** —
/// this file's own derivation, not the producer's want set.
///
/// Written out rather than borrowed because it is the measurement the whole arm
/// rests on: a surface is *blurry* exactly when the level its screen footprint
/// justifies is not resident, whichever class did or did not ask for it. The rule
/// is [`justified_mip`] walked coarser until the level fits the refinement's cap
/// — the shipped rule re-derived, so a one-sided edit fails here.
fn justified_tiles(
    res: &VtResidency,
    view: &RenderView,
    cov: &[VtCoverage],
) -> Vec<(VtTextureHandle, TileCoord)> {
    let view_proj = view.view_proj();
    let scale = projection_scale(view);
    let eye = view.eye_local();
    let half_h = (view.height as f32 * 0.5).max(1.0);
    let mut out = Vec::new();
    for c in cov {
        let px = screen_diameter_px(c.centre, c.radius, eye, scale);
        if !on_screen(&view_proj, c.centre, c.radius, ndc_margin(px, half_h)) {
            continue;
        }
        // `handles`, not `slots` — the twin follows the shipped function, and
        // the shipped function stopped reading packed instance words in the
        // Wave-T audit (see `VtTextureSet::handles`).
        for slot in c.set.handles() {
            if slot == 0 {
                continue;
            }
            let h = VtTextureHandle(slot - 1);
            let Some(desc) = res.desc(h) else { continue };
            let extent = desc.mips[0].width.max(desc.mips[0].height);
            let mut lv = justified_mip(extent, px, desc.mip_count());
            while lv + 1 < desc.mip_count()
                && desc.mips[lv as usize].tile_count() > VT_FEEDBACK_MAX_TILES
            {
                lv += 1;
            }
            let m = desc.mips[lv as usize];
            for y in 0..m.tiles_y {
                for x in 0..m.tiles_x {
                    out.push((h, TileCoord::new(lv, x, y)));
                }
            }
        }
    }
    out
}

/// One arm of the A/B.
#[derive(Debug, Default)]
struct WhipArm {
    /// One line a tick, **by address and never by slot** (the P27 rule): a slot
    /// index is an allocation detail two arms may legitimately disagree about.
    trace: Vec<String>,
    /// **THE COUNTER**: tiles a visible surface's footprint justified and the
    /// pool did not hold, summed over frames.
    blur: u64,
    /// …and the frames at least one of them fell in.
    blur_frames: u64,
    /// Justified tiles offered to the measurement at all — the anti-vacuity
    /// number for the two above.
    justified: u64,
    floor_fallback_frames: u64,
    predict_wants: u64,
    admits: u64,
    evicts: u64,
    /// **Floor** wants resident before a transaction and evicted by it — the
    /// world invariant's counter. The floor and not the whole proved set: a
    /// floor miss taking a resident *refinement*'s slot is exactly what P28.3's
    /// lane walk was built to allow.
    floor_breaches: u64,
}

fn whip_run(predict: Option<u32>, pages: u64) -> WhipArm {
    let (mut lib, handles) = whip_library(pages);
    let cov = whip_coverage(&handles);
    let mut hist = inf_math::CameraHistory::new();
    let mut arm = WhipArm::default();

    for tick in 0..TICKS {
        let view = whip_view(tick);
        assert!(
            hist.commit(inf_math::CameraSample {
                tick,
                eye: view.eye_world,
                forward: view.forward.as_dvec3(),
                up: view.up.as_dvec3(),
            }),
            "tick {tick} was refused by the history"
        );

        let mut wants = analytic_floor(&lib, &view, &cov);
        let floor_end = wants.len();

        // **The refinement lane, at the ring's own latency.** The mask the CPU
        // reads at tick `t` was written at `t − READBACK_LATENCY_FRAMES` and
        // could only mark surfaces visible then. Both halves of that are the lag
        // the predictor exists to close, and a model without it would make this
        // arm a tautology in the other direction.
        if tick >= READBACK_LATENCY_FRAMES {
            let past = whip_view(tick - READBACK_LATENCY_FRAMES);
            wants.extend(
                justified_tiles(lib.residency(), &past, &cov)
                    .into_iter()
                    .map(|(h, t)| VtWant::refine(h, t)),
            );
        }

        if let Some(p) = predict.and_then(|h| inf_math::dead_reckon(&hist, h)) {
            let spec = speculative_wants(lib.residency(), &view, &cov, &p);
            arm.predict_wants += spec.len() as u64;
            wants.extend(spec);
        }

        // The floor's resident set **as offered**, kept beside the transaction —
        // the arbiter does not compute this and cannot agree with it by accident.
        let floor_before: BTreeSet<(u32, u32, u32, u32)> = wants[..floor_end]
            .iter()
            .filter(|w| lib.residency().is_resident(w.texture, w.tile))
            .map(|w| (w.texture.0, w.tile.mip, w.tile.x, w.tile.y))
            .collect();

        let txn = lib.residency_mut().apply_wants(&wants);
        for e in &txn.evicts {
            if floor_before.contains(&(e.texture.0, e.tile.mip, e.tile.x, e.tile.y)) {
                arm.floor_breaches += 1;
            }
        }

        // **This file's own measurement**: what a surface visible NOW needs, and
        // whether it is there. No want class is consulted.
        let need = justified_tiles(lib.residency(), &view, &cov);
        let mut blur = 0u64;
        for (h, t) in &need {
            blur += u64::from(!lib.residency().is_resident(*h, *t));
        }
        arm.justified += need.len() as u64;
        arm.blur += blur;
        arm.blur_frames += u64::from(blur > 0);

        let floor_miss = wants[..floor_end]
            .iter()
            .filter(|w| !lib.residency().is_resident(w.texture, w.tile))
            .count() as u64;
        arm.floor_fallback_frames += u64::from(floor_miss > 0);
        arm.admits += txn.admits.len() as u64;
        arm.evicts += txn.evicts.len() as u64;
        arm.trace.push(format!(
            "{tick} b{blur}/{} f{floor_miss} a{} e{} d{}",
            need.len(),
            txn.admits.len(),
            txn.evicts.len(),
            txn.deferred
        ));
    }
    arm
}

/// **(d) A scripted 360° whip-pan shows measurably fewer fallback frames with the
/// predictor ON than OFF — A/B inside the gate, counters not pixels.**
///
/// The phase's own last clause, through the production doors:
/// [`analytic_floor`] for the floor, [`speculative_wants`] for the lane,
/// [`inf_math::dead_reckon`] over an [`inf_math::CameraHistory`] for the pose,
/// and a real [`VtTextures`] over a real [`VtResidency`] for the pool. Nothing
/// here is a model of the streamer; what *is* modelled is the feedback lane's
/// latency, because a CPU cannot read a mask the GPU has not written yet — and
/// that lag is the entire thing the predictor exists to close.
///
/// The counter is **re-derived**: for every surface visible at tick `t` this file
/// computes the level its footprint justifies and asks the residency whether
/// those tiles are seated. It never asks which class wanted a tile — what a
/// player sees is whether the level is there.
///
/// Three claims, and each is the others' control:
///
/// * **each arm replays to itself.** Determinism alone is satisfied by a
///   predictor that does nothing.
/// * **the two arms differ.** Difference alone is satisfied by one that is
///   random. This is what makes the determinism a statement about this batch.
/// * **the blur counters strictly fall.** With the OFF leg's anti-vacuity
///   asserted FIRST and in **both** directions: it must be blurry on some frame
///   (or the reduction is a bound on zero) and not on every frame (or the pool is
///   under the steady demand, which is a budget problem the predictor cannot fix
///   and must not be credited for).
///
/// The ON arm uses the **shipped default** [`DEFAULT_PREDICT_HORIZON_TICKS`],
/// which is **0** as of P28.5: the speculative LANE is the measured win and the
/// lead time is a cost on this loop, because `apply_wants` seats a miss the frame
/// it is offered. The ruling and its sweep live in `whip_pan.rs`; what this gate
/// asserts is the ROADMAP's clause against whatever the shipped default is.
#[test]
fn the_predictor_strictly_reduces_a_scripted_whip_pans_fallback_frames() {
    let off = whip_run(None, PAGES);
    let on = whip_run(Some(HORIZON), PAGES);

    for (name, a) in [("OFF", &off), ("ON ", &on)] {
        println!(
            "phase28 whip-pan {name}: blur {}/{} over {} frames | floor fallback \
             over {} frames | admits {} evicts {} | speculation {} wants",
            a.blur,
            a.justified,
            a.blur_frames,
            a.floor_fallback_frames,
            a.admits,
            a.evicts,
            a.predict_wants
        );
    }

    // ── bit-exact per arm, and the two arms diverge ─────────────────────────
    assert_eq!(off.trace.len(), TICKS as usize);
    assert_eq!(
        off.trace,
        whip_run(None, PAGES).trace,
        "the OFF arm is not reproducible"
    );
    assert_eq!(
        on.trace,
        whip_run(Some(HORIZON), PAGES).trace,
        "the ON arm is not reproducible"
    );
    assert_ne!(
        off.trace, on.trace,
        "the predictor changed nothing — the A/B below would be a comparison of \
         one arm with itself"
    );

    // ── ANTI-VACUITY ON THE OFF LEG, both directions, before anything is read
    assert!(
        off.blur_frames > 0,
        "the OFF arm was never blurry — the fixture cannot miss and the reduction \
         below would be a bound on zero"
    );
    assert!(
        off.blur_frames < TICKS,
        "every frame is blurry — the pool is under the steady demand, which is a \
         budget problem the predictor cannot fix and must not be credited for"
    );
    assert!(
        off.justified > 0 && on.predict_wants > 0,
        "nothing was justified ({}) or the predictor never ran ({})",
        off.justified,
        on.predict_wants
    );
    assert!(
        on.evicts > 0 && off.evicts > 0,
        "the pool was never contested, so the floor bound below is a bound on an \
         empty list"
    );

    // ── THE ROADMAP'S CLAUSE ────────────────────────────────────────────────
    assert!(
        on.blur_frames < off.blur_frames,
        "fallback frames did not strictly fall: {} ON against {} OFF",
        on.blur_frames,
        off.blur_frames
    );
    assert!(
        on.blur < off.blur,
        "fallback tiles did not strictly fall: {} ON against {} OFF",
        on.blur,
        off.blur
    );

    // …and the win was not bought out of the classes above it: speculation may
    // never cost a resident FLOOR tile its slot (the P26.4 floor law extended to
    // the lane P28.4 added).
    assert_eq!(
        off.floor_breaches, 0,
        "the OFF arm evicted a resident FLOOR tile"
    );
    assert_eq!(
        on.floor_breaches, 0,
        "speculation cost a resident FLOOR tile its slot"
    );
}

// ══ (e) budgets, in the three classes ═══════════════════════════════════════

/// **The combined-residency ratchet.** Measured on this fixture at ~3 MiB peak
/// (geometry ~0.01, texture ~2, shadow ~1); the ceiling is well clear of it, so
/// crossing it means residency grew without bound rather than that a number
/// drifted. **RATCHET RULE: this constant may only ever DECREASE.**
const COMBINED_CEILING: u64 = 8 * 1024 * 1024;

/// **(e) The three budget classes, kept apart** (P26.5's ruling, applied to the
/// unified streamer).
///
/// * **LOAD** — once, against [`LOAD_BUDGET_MS`]: deriving a paired `.inf_vmesh`,
///   registering its two virtual textures, and streaming it to **full** paired
///   residency. That is the load-class work Phase 28 added — a level's geometry
///   and its tiles arriving together — and it is timed end to end rather than
///   sampled.
/// * **WORLD** — bytes and counters, asserted **unconditionally**. A byte is a
///   byte on a discrete card, a CPU rasterizer and a paravirtualized runner
///   alike, and arm (b) proves the sequence is a pure function of committed
///   input, so there is nothing here for an adapter to change. This is the half
///   with teeth in CI.
/// * **CLOCK** — milliseconds, only where a millisecond represents a frame:
///   the visibility path's steady-state frame cost against
///   [`FRAME_BUDGET_MS`], on a real device that is not a software or
///   paravirtualized adapter. Printed always, asserted there.
#[test]
fn the_unified_streamer_stays_inside_its_budgets() {
    // ── LOAD class ──────────────────────────────────────────────────────────
    let t0 = std::time::Instant::now();
    let (src, mesh) = paired_source();
    let (mut res, by_guid) = texture_residency(64 * 1024 * 1024);
    let mut streamer = VgeomStreamer::new(VgeomStreamBudget::default());
    for t in [4.0f32, 0.5, 0.05, 0.005, 0.001] {
        couple_step(&mut streamer, &mut res, &by_guid, &src, t, true);
    }
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
    // ANTI-VACUITY: the timing is of a load that actually happened.
    let pages = streamer
        .residency(PAIRED_ASSET)
        .map_or(0, |r| r.resident_pages());
    assert_eq!(
        pages,
        src.pages().len(),
        "the load did not reach full residency ({pages} of {}), so the time below \
         is the time of a load that did not finish",
        src.pages().len()
    );
    let checked = assert_invariant(&streamer, &res, &by_guid, &src, &mesh, "after the load");
    // The whole asset's sampled-tile set, at full residency — one snapshot, not
    // the churn's accumulation, so the bound is a fraction of arm (b)'s.
    assert!(
        checked > 100,
        "the loaded asset samples only {checked} tiles, so 'it loaded' is a claim \
         about a pairing that is barely there"
    );
    assert!(
        load_ms < LOAD_BUDGET_MS,
        "deriving and streaming a paired asset to full residency took \
         {load_ms:.2} ms, over the {LOAD_BUDGET_MS} ms load budget {}",
        inf_player::budget::RATCHET_NOTE
    );

    // ── WORLD class: everywhere ─────────────────────────────────────────────
    let (src, _mesh) = paired_source();
    let mut s = Unified::new(2 * 1024 * 1024, 16, false);
    for (i, t) in LADDER.iter().enumerate() {
        s.step(&src, *t, i as u64, true);
    }
    let bytes = s.resident_bytes();
    let total = VgeomStreamBudget::default().budget_bytes
        + s.texture.capacity_bytes()
        + s.shadow.capacity_bytes();
    let report = s.report(total);
    println!(
        "phase28 budgets:\n  \
         LOAD  paired derive + register + stream to full residency: {load_ms:.2} ms \
         of {LOAD_BUDGET_MS} ms ({pages} pages, {checked} sampled tiles)\n  \
         WORLD peak combined residency {:.2} MiB of a {:.1} MiB ratchet — \
         geometry {:.2}, texture {:.2}, shadow {:.2}\n  \
         WORLD {}",
        s.peak_bytes as f64 / (1024.0 * 1024.0),
        COMBINED_CEILING as f64 / (1024.0 * 1024.0),
        bytes[0] as f64 / (1024.0 * 1024.0),
        bytes[1] as f64 / (1024.0 * 1024.0),
        bytes[2] as f64 / (1024.0 * 1024.0),
        report.summary(),
    );
    // ANTI-VACUITY FIRST: all three consumers held something, or the ratchet is a
    // ratchet on zero.
    for c in Consumer::ALL {
        assert!(
            report.resident_of(c) > 0,
            "{} held nothing over the ladder",
            c.label()
        );
    }
    assert!(s.peak_bytes > 0);
    assert!(
        s.peak_bytes <= COMBINED_CEILING,
        "the three page systems held {} B between them, over the \
         {COMBINED_CEILING} B ceiling — RATCHET RULE: this constant may only ever \
         DECREASE, so a value above it is a regression report and not a settings \
         change",
        s.peak_bytes
    );
    assert!(
        report.within_grant(),
        "a consumer is outside the grant its live floors justify"
    );

    // ── CLOCK class: only where a millisecond represents a frame ────────────
    let Some(gpu) = gpu_or_skip("the budget arm's CLOCK class") else {
        println!("phase28 budgets: CLOCK class not measured — no adapter");
        return;
    };
    let info = gpu.adapter.get_info();
    let virtualized = {
        let n = info.name.to_ascii_lowercase();
        n.contains("paravirtual") || n.contains("virtualbox") || n.contains("vmware")
    };
    let software = info.device_type == wgpu::DeviceType::Cpu || virtualized;

    let sc = vis_scene(3, true, VtTextureSet::NONE);
    let target = HeadlessTarget::new(&gpu, W, H);
    let mut r = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
    r.set_settings(vis_settings(true));
    let v = vis_view();
    let mut frame_ms = Vec::new();
    for _ in 0..8 {
        let t = std::time::Instant::now();
        r.render(&gpu, &sc, &v, &target.view, (W, H));
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        frame_ms.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    // The first frame builds pipelines and pools; the steady state is the tail.
    let steady: f64 = frame_ms[1..].iter().sum::<f64>() / (frame_ms.len() - 1) as f64;
    let audit = r.vis_audit();
    println!(
        "phase28 budgets:\n  \
         CLOCK visibility-path steady frame {steady:.2} ms of {FRAME_BUDGET_MS} ms \
         on {} ({:?}){} — {} visibility frames, {} refusals",
        info.name,
        info.device_type,
        if software {
            " [virtual/software — the clock is reported, not asserted]"
        } else {
            ""
        },
        audit.frames,
        audit.refused(),
    );
    // ANTI-VACUITY: the frames timed were visibility frames.
    assert_eq!(
        audit.refused(),
        0,
        "the visibility path was refused: {audit:?}"
    );
    assert!(
        audit.frames >= 8,
        "only {} of 8 frames rasterized a visibility buffer, so the clock above is \
         the forward path's",
        audit.frames
    );
    if software {
        return;
    }
    assert!(
        steady < FRAME_BUDGET_MS,
        "a visibility-buffer frame cost {steady:.2} ms on {}, over the \
         {FRAME_BUDGET_MS} ms frame budget {}",
        info.name,
        inf_player::budget::RATCHET_NOTE
    );
}

// ══ (f) the golden set ══════════════════════════════════════════════════════

/// **(f) The golden set is pinned, and it is additive only.**
///
/// Phase 28 shades meshlets through a second path for the first time and it
/// re-blessed nothing: `VgeomSettings::visbuffer` is `false` by default, so a
/// default frame takes the forward meshlet raster and every golden is the picture
/// it always was. P28.1–P28.5 added **none** — every claim the phase makes is
/// asserted as state or as a counter, which is the ROADMAP's own instruction
/// ("A/B inside the gate, counters not pixels").
///
/// Pinned by CONTENT as well as by count, for the reason `phase26_gate` measured
/// (both halves of "a count is enough" are backwards): `check_golden`'s pixel
/// comparison is opt-in, and the harness WRITES any golden it cannot read, so a
/// deletion is a silent re-bless. The digest fails on an edited frame, on a
/// re-bless, and on a delete-then-regenerate whenever the regenerated frame
/// differs.
///
/// Deliberately a **third** copy of the two constants rather than a shared
/// helper: three gates are three phases' claims about one directory, and a shared
/// constant would let one phase's re-bless satisfy the others' pin.
#[test]
fn the_golden_set_is_pinned_and_additive_after_phase_28() {
    /// 50 through P26.5, plus P27.4's four virtual-shadow frames. **Phase 28
    /// adds none.** Wave VIS1a's **audit** adds the fifty-fifth, `water_ssr.png`
    /// — additive, so this pin and the digest below move together and no
    /// committed image changed; `phase26_gate`'s twin carries the reason.
    const GOLDENS: usize = 62;
    /// `xxh3_128` over `"{file_name} {hex}\n"` for every golden, name-sorted.
    /// **RULE: this may change only in a commit that adds a golden, or in one
    /// whose stated purpose is to change what the engine LOOKS like.**
    ///
    /// **Moved once, at wave SKY2** (from `23d41a61c31c28a17a20871b6c875707`) —
    /// the volumetric-cloud overhaul, eight cloud-carrying frames re-blessed on
    /// purpose, count unchanged. `phase26_gate`'s twin carries the full reason;
    /// all three pins had to be moved by hand, which is what having three of them
    /// buys.
    ///
    /// **Wave NPC1b adds the fifty-ninth**: `crowd_variation.png` — eight bodies
    /// on ONE skinned mesh sharing ONE joint palette, each with its own tint and
    /// its own build, drawn in a single instanced call from a single atlas block.
    /// Additive: `GOLDENS` moves 58 -> 59, the digest moves with it, and **not one
    /// committed image changed** — the instanced path draws a one-mesh scene in
    /// the order it drew it before, which is what the stable sort in
    /// `plan_skinned_batches` is for.
    ///
    /// **Wave GTA1 moves it a seventh time, and this move is BOTH branches of the
    /// rule at once** (from `1db3dd0bf5961ddbfb53aeb2b40a1697`), which is why the
    /// commits are separate and each says which it is:
    ///
    ///  * **additive** — `sky_night_horizon.png`, the frame the night sky's own
    ///    defect lived in. `golden_sky_night` pitches 35 deg up and looks AWAY
    ///    from the sun and bounds only blue, so the horizon band facing a
    ///    below-horizon sun was outside every committed frame in this set.
    ///    `GOLDENS` moves 59 -> 60 beside it.
    ///  * **the look** — `clouds_night.png` and `sky_night.png`, re-blessed with
    ///    the stated purpose that the committed frames depicted the defect: the
    ///    transmittance LUT's horizon-tangent texel leaking under the ground made
    ///    a midnight cloud deck glow at R/G **20.88** (top-half mean
    ///    `[0.27609, 0.01323, 0.01039]` -> `[0.00656, 0.00656, 0.00656]`, diff
    ///    mean 0.0835 / max 0.5064) and the night sky red-biased (`[0.00975,
    ///    0.00802, 0.00941]` -> `[0.00672, ...]`, diff mean 0.0057 / max 0.0500 —
    ///    inside tolerance, and re-blessed anyway, because a frame that quietly
    ///    depicts an engine that no longer exists is one nobody can read a
    ///    regression off).
    ///
    /// **Day is untouched and it is measured**: `sky_noon`, `sky_dawn` and
    /// `sky_dusk` compare at mean 0.000000 / max 0.000000, because the new
    /// visibility factor is exactly 1.0 while the sun is up.
    ///
    /// # Wave IASSET2: `ground_close.png`, re-blessed on the same precedent
    ///
    /// The ground library's normal and detail maps moved from **BC1 to BC5**,
    /// which is the wave's content clause and is measured on those exact bytes:
    /// BC1's worst per-channel error on the two channels a tangent-space normal
    /// lives in is **122 of 255** against BC5's **17** (mean 10.4 -> 1.7). A
    /// normal that wrong does not shade approximately wrong, it faces somewhere
    /// else.
    ///
    /// `ground_close.png` is the one committed frame that depicts those maps.
    /// Its fixture called `TextureCompression::Bc1` for all four slots by hand;
    /// it now calls `inf_material::ground`'s own per-slot settings, so "the real
    /// committed content" stays true rather than remembered. The frame moved by
    /// **mean 0.001212 / max 0.024941** — inside the harness's tolerance, and
    /// re-blessed anyway, for the reason the paragraph above gives verbatim: a
    /// frame that quietly depicts an engine that no longer exists is one nobody
    /// can read a regression off.
    ///
    /// **The albedo and ORM maps did NOT move**, and that is measured too: BC7
    /// would take the albedo's worst error from 11 of 255 to 5 and the ORM's
    /// from 45 to 33, for twice the page bytes — half of what a 24 MiB atlas arm
    /// holds. `GOLDENS` does not move; the set is still 60 files.
    /// **Wave VEN1a (2026-09-02) — the ADDITIVE branch, twice.** Two goldens
    /// join the set and **no existing frame was re-blessed**.
    ///
    /// `gi_scatter_neon.png`: a scatter batch of emissive plates lighting a
    /// white floor through the GI volume under a zero-radiance sun. It exists
    /// because `passes::gi` staged instances, skinned meshes and vgeom and never
    /// **scatter** — so a grammar-built venue's neon, string lights and lit
    /// panes were drawn and bounced nothing. Nothing could have moved with it:
    /// the four `gi_*` goldens hold no scatter and the two `scatter_*` goldens
    /// run with GI off, measured under `INF_GOLDEN_STRICT=1` before the pin did.
    ///
    /// `venue_interior.png`: the wave's visual claim as one frame — a **closed**
    /// near-black room lit only by practicals, with a red key and a blue rim
    /// over a plank stage, a chrome pole (`metallic 1.0, roughness 0.12`)
    /// standing in the wash, a magenta neon plate throwing a halo onto the
    /// boards, and a festoon chain. Its far corner sits at **4.2 of 255** and
    /// its stage at 45; with the room's roof and side walls removed the corner
    /// was 40, which is a frame lit by daylight with some neon in it.
    ///
    /// `GOLDENS` moves 60 -> 62.
    // Wave EDIT1 moved this ONCE, with a stated purpose and a described
    // difference: `gi_specular.png` was re-blessed when the GI ambient stopped
    // being irradiance spent as radiance. The other 61 frames are byte-identical
    // -- four of the five GI goldens hold their images through the
    // `GI_LAMBERT_PI` compensation in `golden.rs`, and `gi_specular` is the one
    // a single multiplier cannot hold because EDIT1 removed a pi from two
    // places and its subject rides both. Mean 0.072099, max 0.242510; every
    // structural assertion it exists for passes on the new frame.
    //
    // **Wave FIX3 moves it again** (from `1a03e55866b2b76958573b4050651d98`), on
    // the re-bless branch of the rule, and the wave's stated purpose IS the look:
    // the ambient term every shaded surface in the engine spends stops being the
    // GI probe field alone and becomes the sky's own irradiance, projected from
    // the P17 medium, with the probe field added as its signed difference from an
    // open sky. Measured before it: a white wall's shaded face read 0.0129 of its
    // sunlit face under a clear noon sky, against the 0.1-0.2 physics gives; the
    // island's Play frame had a third of the hero below luminance 8.
    //
    // **SIX frames move and all six are GI-on scenes**, which is the gate -- the
    // term is reached only through `GiSettings::enabled`, so the other 56 run the
    // instruction stream they always did and are byte-identical:
    //
    //   gi_bleed         mean 0.1558 / max 0.4919   more colour bleed: the red
    //                    wall is lit by the sky now, so it has light to bleed
    //   gi_terrain       0.0715 / 0.2930            the bounce carries a cosine
    //   gi_specular      0.0536 / 0.2940            the sky reflects again
    //   gi_emissive      0.0495 / 0.3170
    //   venue_interior   0.0426 / 0.3347            the room got DARKER (its far
    //                    corner 4.2 -> 2.0): the probe field's `-sky` term takes
    //                    back exactly the sky a closed room blocks
    //   gi_scatter_neon  0.0261 / 0.2783
    //
    // The last four are inside the harness's perceptual tolerance and are
    // re-blessed anyway, for the reason `golden.rs` states: a frame that no
    // longer depicts what the engine draws is a frame nobody can read a
    // regression off. `GOLDENS` does not move. The whole table, with the arms'
    // own before/after numbers, is in `docs/memos/island-progress.md` under
    // *Wave FIX3*.
    //
    // **The FIX3 AUDIT moves it once more** (from
    // `f7310588fca598ba368e4d2aaa89dc83`), on the same re-bless branch of the
    // same rule, and again the stated purpose IS the look. The paragraph above
    // says the `-sky` term "takes back exactly the sky a closed room blocks".
    // Measured, it did not: `inf-render/tests/interior_ambient.rs` photographs a
    // SEALED 24x24x12 m hall -- no door, no window, the sun outside -- and its
    // three faces summed to **0.4307** against a DOORED hall's 0.4348, i.e.
    // **99 %** of the light of a room that has an opening. Two mechanisms, both
    // closed by the audit:
    //
    //   * the probe march lit every surface it hit with the UNOCCLUDED sky, so
    //     an interior wall bounced daylight it cannot see. The sky half of that
    //     bounce is now scaled by the probe's own upward sky-view factor, which
    //     is 0 in a sealed room and 1 on open ground.
    //   * a probe standing INSIDE a wall or a floor slab still voted in the
    //     trilinear blend -- 87 % of the weight for the shaded face
    //     `sky_ambient.rs` is about -- and the blend reached through ceilings.
    //     Buried probes no longer vote and the blend carries the receiver's own
    //     normal.
    //
    // The sealed hall now reads **0.0004** against the doored hall's 0.0059, and
    // the outdoor reading the wave was written for holds and improves:
    // shaded:sunlit **0.1838 -> 0.2151** against a physical 0.1-0.35, with the
    // far row unchanged at 0.2867 and the furnace unchanged at 1.0096.
    //
    // **THREE frames move**, all of them GI-on:
    //
    //   venue_interior   mean 0.0189 / max 0.4381   stage 45.50 -> 40.35, and
    //                    the far corner is UNMOVED at 1.97
    //   gi_emissive      0.0629 / 0.1567            the bar's green bleed nearly
    //                    doubles, green/red 1.996 -> 3.371
    //   gi_scatter_neon  0.0369 / 0.3564
    //
    // …and THREE MORE, on a second pass, because the first write-up of this
    // block said `gi_bleed`, `gi_terrain` and `gi_specular` were "byte-identical
    // through it" and they were not. Their committed PNGs had not moved, which
    // is a different sentence: the RENDER moved, by mean 0.0083 / 0.0095 /
    // 0.0060 — under the harness's perceptual tolerance, so the strict run was
    // green and the claim went unmeasured. Measured on this adapter, all six
    // GI goldens read **0.000000** against their frames at `07f862f9` and three
    // of them did not any more. They are re-blessed on the same branch of the
    // same rule, for the reason `golden.rs` states: a frame that no longer
    // depicts what the engine draws is a frame nobody can read a regression off.
    //
    //   gi_bleed         mean 0.0083 / max 0.0452   near-wall red/green 1.462
    //                    -> 1.479
    //   gi_terrain       0.0095 / 0.0380            green drop 9.85 held; the
    //                    RED drop went +5.07 -> -2.46, i.e. the red ground now
    //                    brightens the wall's red instead of darkening it less
    //   gi_specular      0.0060 / 0.0478            grazing floor 194.5 -> 196.5
    //                    against a flat control 59.3 -> 57.8
    //
    // And the count above it: **ELEVEN** frames of the other 56 carry a
    // pre-existing adapter diff, not thirteen. Measured at `e8451338`,
    // `07f862f9` and here, the same eleven appear with the same means to six
    // decimals (cave_mouth, deform, ground_close, terrain, terrain_lod,
    // terrain_splat and the five water frames). The wave's thirteen counted
    // `gi_bleed` (0.044445) and `gi_terrain` (0.028975) among them, which are
    // not adapter diffs at all: they are the wave's own move, and they are the
    // very numbers its own table reports as the "before" of those two frames.
    const GOLDEN_SET_DIGEST: &str = "cb6c4704b2298cbe0d18729a3d251e29";
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("crates")
        .join("inf-render")
        .join("tests")
        .join("goldens");
    let mut pngs: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("read {}: {e}", dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "png"))
        .collect();
    pngs.sort();
    assert_eq!(
        pngs.len(),
        GOLDENS,
        "the golden set is {} files, not {GOLDENS}. ADDING one is allowed and \
         means moving both constants in the same commit; losing one is not.",
        pngs.len()
    );
    let mut manifest = String::new();
    for p in &pngs {
        let bytes = std::fs::read(p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
        assert!(bytes.len() > 256, "{} is not a golden", p.display());
        manifest.push_str(&format!(
            "{} {}\n",
            p.file_name().expect("a file name").to_string_lossy(),
            ContentHash::of(&bytes).to_hex()
        ));
    }
    assert_eq!(
        ContentHash::of(manifest.as_bytes()).to_hex(),
        GOLDEN_SET_DIGEST,
        "the golden set's CONTENT moved during Phase 28. A golden was re-blessed, \
         deleted (the harness regenerates any golden it cannot read, so the count \
         above stays {GOLDENS}), or replaced.\n{manifest}"
    );
    println!("phase28 goldens: {GOLDENS} files, digest {GOLDEN_SET_DIGEST}");
}

// ══ (g) the ray-query experiment ════════════════════════════════════════════

/// **(g) The ray-query experiment is never load-bearing.**
///
/// P28.5's own clause ends *"VSM remains the shipped path on every tier; the
/// experiment lands behind a default-off setting with a `caps.rs` clamp"*, and
/// this is the phase-level statement of it. `crates/inf-render/tests/ray_query.rs`
/// owns the experiment's correctness — the BLAS/TLAS against a CPU ray caster,
/// the surface offset, the comparison against the shipped shadow map — and none
/// of that is repeated here. What is asserted here is the **structural** property
/// that has to hold on every machine, including the overwhelming majority that
/// cannot run a ray query at all:
///
/// * the setting is off in a default `RaytraceSettings` and in a default
///   `RenderSettings`;
/// * **no tier and no preset turns it on** — every tier is applied to a settings
///   block with the experiment on and must leave it alone, and to one with it off
///   and must not set it. Both halves, because "a tier has no opinion" is two
///   claims and only one of them is about the default;
/// * [`AdapterCaps::clamp_ray_query`] **only ever clears**: an adapter that can
///   trace does not enable an experiment nobody asked for;
/// * virtual shadow maps are still the shipped shadow path on High and Medium,
///   and Low still keeps CSM.
///
/// It fails the day the experiment becomes reachable without a host explicitly
/// asking for it, which is the only way this clause can be broken.
#[test]
fn the_ray_query_experiment_is_never_load_bearing() {
    // ── off by default ──────────────────────────────────────────────────────
    assert!(
        !RaytraceSettings::default().sun_shadows,
        "the experiment shipped ON"
    );
    assert!(!RenderSettings::default().raytrace.sun_shadows);

    // ── no tier and no preset turns it on, and none of them clears it ───────
    let off = RenderSettings::default();
    let mut on = RenderSettings::default();
    on.raytrace.sun_shadows = true;
    for tier in [RenderTier::High, RenderTier::Medium, RenderTier::Low] {
        assert!(
            !tier.apply(off).raytrace.sun_shadows,
            "{tier:?} ENABLED the ray-query experiment"
        );
        assert!(
            tier.apply(on).raytrace.sun_shadows,
            "{tier:?} took an opinion about an experiment — only the adapter clamp \
             may clear it, so a host that deliberately turned it on to measure it \
             does not silently lose it to a tier decision"
        );
    }
    assert!(!RenderTier::mobile_default().raytrace.sun_shadows);
    assert!(!RenderTier::clamp_mobile(off).raytrace.sun_shadows);
    // **STATED HONESTLY RATHER THAN ASSERTED THE OTHER WAY**: the mobile preset
    // does NOT clear an experiment a host explicitly asked for. It is a preset,
    // and the rule the clause states is that no preset has an *opinion* — the one
    // thing that clears the experiment is the adapter capability clamp below,
    // which is the door that knows whether the machine can trace at all. Measured
    // here rather than assumed, so a preset that starts clearing it (or setting
    // it) fails this arm either way.
    assert!(
        RenderTier::clamp_mobile(on).raytrace.sun_shadows,
        "the mobile preset took an opinion about an experiment — only the adapter \
         clamp may decide, because only it knows whether the device can trace"
    );

    // ── the capability clamp only ever CLEARS ───────────────────────────────
    fn caps(ray_query: bool) -> AdapterCaps {
        AdapterCaps {
            compute_shaders: true,
            indirect_execution: true,
            max_storage_buffers_per_stage: 8,
            max_storage_buffer_binding_size: 128 << 20,
            max_compute_workgroups_per_dim: 65535,
            max_storage_textures_per_stage: 1,
            is_cpu: false,
            texture_compression_bc: true,
            ray_query,
            polygon_mode_line: true,
        }
    }
    assert!(!caps(false).clamp_ray_query(on).raytrace.sun_shadows);
    assert!(!caps(false).clamp_ray_query(off).raytrace.sun_shadows);
    assert!(
        caps(true).clamp_ray_query(on).raytrace.sun_shadows,
        "an adapter that can trace refused a caller who asked"
    );
    assert!(
        !caps(true).clamp_ray_query(off).raytrace.sun_shadows,
        "an adapter that CAN trace enabled an experiment nobody asked for"
    );
    // …and the capability is orthogonal to the tier, like BC and line raster: a
    // machine does not become a High machine by growing a ray-query unit.
    assert_eq!(
        inf_render::caps::choose_tier(&caps(true)),
        inf_render::caps::choose_tier(&caps(false)),
        "ray queries moved the render tier"
    );

    // ── VSM is still the shipped shadow path ────────────────────────────────
    let vsm_on = RenderSettings {
        vsm: VsmSettings {
            enabled: true,
            ..VsmSettings::default()
        },
        ..RenderSettings::default()
    };
    assert!(
        RenderTier::High.apply(vsm_on).vsm.enabled,
        "High lost virtual shadow maps"
    );
    assert!(
        RenderTier::Medium.apply(vsm_on).vsm.enabled,
        "Medium lost virtual shadow maps"
    );
    assert!(
        !RenderTier::Low.apply(vsm_on).vsm.enabled,
        "Low keeps CSM — the P27.5 clause"
    );
    println!(
        "phase28 ray query: off by default, unchanged by 3 tiers and 2 presets, \
         cleared by the clamp on a non-tracing adapter and never set on a tracing \
         one; VSM shipped on High/Medium, CSM on Low"
    );
}
