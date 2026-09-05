//! **WHAT A COMMITTED LEVEL ACTUALLY DRAWS** (wave CERT1, CP-A1 / CP-B5).
//!
//! # The hole this file closes
//!
//! `fps_instrument.rs` has documented, since island wave I4, that a level which
//! authors no render block ships with *shadows, GI, VSM, TAA, SSAO, bloom and
//! the visbuffer all off* — `RenderSettingsRecord::default()`. What no arm
//! anywhere asserted was **which levels those are**. Every committed level was
//! one, including the showcase island the application now boots on, and nothing
//! in the tree would have said so.
//!
//! So the certification's answer is two arms and a source gate, and each one
//! fails for a different reason:
//!
//! 1. the levels that are meant to be looked at DECODE to a lit record, read
//!    field by field — a future re-bless that drops the block goes red here;
//! 2. the levels that are not meant to be looked at still decode to the
//!    default, so arm 1 is a statement about a DECISION and not about a
//!    constant that happens to be true everywhere;
//! 3. a PIE payload carries the level's own block rather than the default, and
//!    the source gate keeps the substitution from coming back.
//!
//! # The CP-A1 ruling, and why it is this one
//!
//! The brief offered two routes and asked for both to be priced. They are not
//! equivalent, and the measurement is in arm 1's own subject:
//!
//! * **"the levels author the lit stack"** — five committed `.inf_lvl` files
//!   move, each by a handful of flag bytes, byte length unchanged. Taken.
//! * **"the default becomes lit for 3D levels"** — *reaches nothing that
//!   exists.* `RenderSettingsRecord` is persisted POSITIONALLY inside
//!   `RuntimeSettings`, so every already-committed level carries the values
//!   that were current when it was written; changing `Default` relights only
//!   levels created afterwards, and it breaks the standing both-hosts pin
//!   `apply_record(&default()) == RenderSettings::default()`. It would have
//!   left the island exactly as dark as it was.
//!
//! # What this file does NOT claim
//!
//! Nothing here renders. It asserts what the AUTHORED record is and what the
//! PIE seam carries; the tier clamp and the adapter clamp sit downstream in
//! `shipped_settings`, and on a Medium or Low adapter they will turn some of
//! this back off. That is `fps_instrument.rs`'s subject and it says so.

use std::path::PathBuf;

use inf_render::{detect_tier, AdapterCaps, GpuContext, RenderSettings, RenderTier, VgeomSettings};
use inf_scene::{RenderSettingsRecord, RuntimeLevel};

/// How many `.inf_lvl` files this repository commits — the same number
/// `editor/crates/inf-editor-core/tests/committed_level_sidecars.rs` pins as
/// `EXPECTED_LEVELS`, restated here because the census below is only a census if
/// it walked all of them.
const EXPECTED_LEVELS: usize = 24;

/// The repo root, from this crate's manifest directory.
fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Decode a committed `.inf_lvl` and hand back its render block.
fn record_of(rel: &str) -> RenderSettingsRecord {
    let path = repo().join(rel);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    RuntimeLevel::decode(&bytes)
        .unwrap_or_else(|e| panic!("decode {}: {e}", path.display()))
        .settings
        .render
}

/// The levels this wave ruled LIT: the showcase, the fixture the island gate
/// drives, and the three 3D starter templates `inf new` scaffolds.
///
/// The 2D platformer is deliberately absent — CSM and a GI probe march are not
/// what a sprite level wants, and its own control arm below says so.
const LIT_LEVELS: [&str; 5] = [
    "samples/island/VancouverIsland.inf_lvl",
    "samples/island-fixture/IslandFixture.inf_lvl",
    "templates/blank-3d/Blank.inf_lvl",
    "templates/first-person/FirstPerson.inf_lvl",
    "templates/hybrid-2.5d/Hybrid.inf_lvl",
];

/// The six authored bits `RenderSettingsRecord::lit_showcase` turns on, named
/// one at a time so a failure says WHICH one went missing rather than
/// "the records differ".
fn assert_lit(rel: &str, r: &RenderSettingsRecord) {
    for (name, on) in [
        ("shadows", r.shadows_enabled),
        ("gi", r.gi_enabled),
        ("bloom", r.bloom_enabled),
        ("ssao", r.ssao_enabled),
        ("taa", r.taa),
        ("flare", r.flare_enabled),
    ] {
        assert!(
            on,
            "{rel} ships with {name} OFF — the lit stack was dropped from a \
             committed level (see `RenderSettingsRecord::lit_showcase`)"
        );
    }
}

#[test]
fn every_showcase_level_ships_the_lit_stack() {
    for rel in LIT_LEVELS {
        let r = record_of(rel);
        assert_lit(rel, &r);
        println!(
            "{rel}: shadows on to {:.1} m, gi {}, bloom {}, ssao {}, taa {}, flare {}",
            r.shadows_max_distance,
            r.gi_enabled,
            r.bloom_enabled,
            r.ssao_enabled,
            r.taa,
            r.flare_enabled
        );
    }

    // The island's one knob beyond the shared constructor, read BY NAME so the
    // day the number moves the ledger and the arm move together.
    let island = record_of(LIT_LEVELS[0]);
    assert_eq!(
        island.shadows_max_distance,
        inf_editor_core::island::ISLAND_SHADOW_DISTANCE_M,
        "the island's cascade range is not the constant that documents it"
    );
    // …and it is not merely the default wearing a name.
    assert!(
        island.shadows_max_distance > RenderSettingsRecord::default().shadows_max_distance,
        "the island's cascade range is the engine default ({} m), so the \
         building across a 100 m street still casts nothing",
        RenderSettingsRecord::default().shadows_max_distance
    );
}

/// Every committed `.inf_lvl`, in the two directories `committed_level_sidecars`
/// enumerates — `samples/*` and `templates/*`, one level deep, sorted.
fn committed_levels() -> Vec<PathBuf> {
    let root = repo();
    let mut out = Vec::new();
    for dir in ["samples", "templates"] {
        let Ok(entries) = std::fs::read_dir(root.join(dir)) else {
            continue;
        };
        let mut subs: Vec<PathBuf> = entries.filter_map(|e| e.ok().map(|e| e.path())).collect();
        subs.sort();
        for sub in subs {
            let Ok(files) = std::fs::read_dir(&sub) else {
                continue;
            };
            let mut levels: Vec<PathBuf> = files
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|e| e == "inf_lvl"))
                .collect();
            levels.sort();
            out.extend(levels);
        }
    }
    out.sort();
    out
}

/// **THE CENSUS THE CERTIFICATION'S OPENING SENTENCE RESTS ON.**
///
/// The memo says every committed level shipped `RenderSettingsRecord::default()`
/// before this wave and that five of them do not now. That is a claim about
/// twenty-four files, and the two arms above check eight — so this one walks all
/// of them and prints the table, because prose is never ahead of its arms and
/// the verdict's first paragraph is the prose most likely to be quoted.
#[test]
fn the_whole_committed_corpus_is_censused_lit_or_not() {
    let levels = committed_levels();
    let mut lit = Vec::new();
    let mut plain = Vec::new();
    for path in &levels {
        let rel = path
            .strip_prefix(repo())
            .unwrap_or(path)
            .to_string_lossy()
            .replace(std::path::MAIN_SEPARATOR, "/");
        let bytes = std::fs::read(path).expect("a committed level reads");
        let r = RuntimeLevel::decode(&bytes)
            .unwrap_or_else(|e| panic!("decode {rel}: {e}"))
            .settings
            .render;
        match r == RenderSettingsRecord::default() {
            true => plain.push(rel),
            false => lit.push(rel),
        }
    }
    println!(
        "CP-A1 the committed corpus: {} levels, {} authoring a render block, {} at the default",
        levels.len(),
        lit.len(),
        plain.len()
    );
    for rel in &lit {
        println!("  LIT     {rel}");
    }
    for rel in &plain {
        println!("  default {rel}");
    }

    assert_eq!(
        levels.len(),
        EXPECTED_LEVELS,
        "the committed level count moved; `committed_level_sidecars::EXPECTED_LEVELS` is the other half of this claim and they must move together"
    );
    // The five this wave ruled lit, and NOBODY ELSE. A future bless that lights
    // a sixth by accident is as much a defect as one that darkens the island.
    assert_eq!(
        lit.len(),
        LIT_LEVELS.len(),
        "{} committed levels author a render block, not the {} this wave ruled: {lit:?}",
        lit.len(),
        LIT_LEVELS.len()
    );
    for rel in LIT_LEVELS {
        assert!(
            lit.iter().any(|l| l == rel),
            "{rel} was ruled lit and does not author a block"
        );
    }
}

#[test]
fn a_level_that_did_not_ask_for_the_stack_still_ships_without_it() {
    // The control. Without it, arm 1 would pass on a build where the record's
    // DEFAULT had been flipped lit — which is precisely the route this wave
    // measured and refused, and the route somebody will reach for again.
    for rel in [
        "templates/2d-platformer/Platformer.inf_lvl",
        "samples/platformer-2d/Platformer.inf_lvl",
        "samples/physics-playground/Playground.inf_lvl",
    ] {
        let r = record_of(rel);
        assert_eq!(
            r,
            RenderSettingsRecord::default(),
            "{rel} is not one of this wave's lit levels but no longer decodes \
             to the default record — either the default moved (which reaches \
             nothing already committed and breaks the both-hosts \
             apply_record pin) or a bless leaked"
        );
        println!("{rel}: default record, as ruled");
    }
}

#[test]
fn a_pie_payload_carries_the_levels_own_render_block_and_not_the_default() {
    // The defect: `window::run_pie` passed `RenderSettingsRecord::default()`
    // with a comment saying the payload carried no settings — and `level_bytes`
    // IS the live document's `.inf_lvl`, whose `RuntimeSettings` holds the
    // record, decoded into `BuiltWorld::render` since R-P4 and thrown away by
    // every PIE boot path since. The editor's Play button therefore previewed a
    // level unlit while the shipped build of the same level rendered it lit:
    // PIE != shipping on the one half of the frame no `state_bytes` fold can
    // see, and so the one half no determinism gate could ever have caught.
    let bytes = std::fs::read(repo().join(LIT_LEVELS[2])).expect("the template reads");
    let authored = RuntimeLevel::decode(&bytes)
        .expect("it decodes")
        .settings
        .render;
    assert_ne!(
        authored,
        RenderSettingsRecord::default(),
        "the fixture level is not lit, so this arm could not tell the level's \
         block from the default and would pass on the defect"
    );

    let payload = inf_runtime::pie::ScenePayload::new("lit-stack", bytes, Vec::new(), 60, false);
    let built = inf_player::sim_from_payload(&payload).expect("the payload builds");
    assert_eq!(
        built.render, authored,
        "a PIE session starts from a render block that is not the level's"
    );
    println!(
        "PIE payload render block: shadows {} / gi {} / bloom {} / ssao {} / taa {} / flare {}",
        built.render.shadows_enabled,
        built.render.gi_enabled,
        built.render.bloom_enabled,
        built.render.ssao_enabled,
        built.render.taa,
        built.render.flare_enabled
    );

    // The complement: a level that authors nothing still gets nothing, so the
    // seam carries the AUTHOR's decision rather than a constant either way.
    let plain = std::fs::read(repo().join("templates/2d-platformer/Platformer.inf_lvl"))
        .expect("the 2D template reads");
    let plain_payload = inf_runtime::pie::ScenePayload::new("plain", plain, Vec::new(), 60, false);
    let plain_built = inf_player::sim_from_payload(&plain_payload).expect("it builds");
    assert_eq!(plain_built.render, RenderSettingsRecord::default());
}

#[test]
fn no_windowed_boot_path_substitutes_a_default_render_block() {
    // A SOURCE gate, because the thing it guards cannot be reached without a
    // window and a GPU: `run_pie`, `run_android` and `run_web` each built their
    // `PlayerApp` with `RenderSettingsRecord::default()`, and the arm above can
    // only see as far as `sim_from_payload`. The precedent is the tree's other
    // source gates (`portable_math_law`, the `SHADER_TABLE` naga gate): when the
    // property is about which expression a call site names, read the call site.
    let src = std::fs::read_to_string(repo().join("runtime/inf-player/src/window.rs"))
        .expect("window.rs reads");
    // COMMENTS ARE STRIPPED FIRST, and that is not a convenience: the fix's own
    // doc comment quotes the expression it removed, which is exactly the shape a
    // naive `contains` gate reports as the defect it just closed. A gate that
    // cannot tell a call site from a sentence about a call site is a gate that
    // gets silenced rather than read. (It did, on this file's first run.)
    let code: String = src
        .lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n");
    let hits = code.matches("RenderSettingsRecord::default()").count();
    assert_eq!(
        hits, 0,
        "runtime/inf-player/src/window.rs names \
         `RenderSettingsRecord::default()` {hits} time(s) — a windowed boot \
         path is substituting a default render block for the level's own again"
    );
    // Non-vacuous: the file must still be the one that builds the app, or the
    // count above would be zero for the wrong reason.
    assert!(
        code.contains("PlayerApp::new"),
        "the source gate is reading the wrong file"
    );
    // …and the stripper must not have eaten the code with the comments: a
    // filter that dropped everything would satisfy the count above for the
    // wrong reason, which is the vacuity this repository has caught five times.
    assert!(
        code.len() > src.len() / 3,
        "comment stripping removed {} of {} bytes — the gate is reading almost nothing",
        src.len() - code.len(),
        src.len()
    );
    println!(
        "window.rs: 0 default-record substitutions over {} bytes of code, {} of comment",
        code.len(),
        src.len() - code.len()
    );
}

// ── the editor viewport and the shipped player, on one adapter (wave FIX3) ──

/// The editor viewport's settings decision, transcribed from
/// `inf_viewport::host::apply_render_settings` (host.rs) plus
/// `requested_render_settings`.
///
/// **A transcription, and pinned as one.** No crate in this repository depends
/// on both `inf-viewport` and `inf-player` — the editor's host is Ring 1 and the
/// player is Ring 2 — so a runtime comparison of the two chains has to restate
/// one of them. `shipped_settings`' own doc says why that is dangerous: *"a
/// harness that rebuilt this chain by hand would measure a configuration nobody
/// ships"*. So the restatement below is checked against the source it restates
/// by `the_transcribed_editor_chain_is_the_editor_viewports_own` — the same
/// `include_str!` instrument `apply_record_mirror.rs` uses on the two
/// `apply_record` bodies, and for the same reason.
///
/// The two chains are documented as differing in exactly two places:
///
/// * `detect_tier`'s second argument — the editor passes
///   `RenderSettings::default()`, the player passes the mapped record. It reads
///   only `tier_override`, which `apply_record`'s `..d` always leaves `None`, so
///   the two arguments cannot produce different tiers. Asserted below by
///   comparing the tier too, on both records.
/// * `clamp_mobile`, which the player applies under `cfg(wasm32/android)` and
///   the editor deliberately does not (there is no mobile editor). Unreachable
///   on every desktop target this arm runs on.
fn editor_settings(gpu: &GpuContext, record: RenderSettingsRecord) -> (RenderSettings, RenderTier) {
    let base = inf_player::render::apply_record(&record);
    let tier = detect_tier(gpu, &RenderSettings::default());
    let caps = AdapterCaps::probe(gpu);
    let requested = RenderSettings {
        vgeom: VgeomSettings {
            enabled: true,
            ..base.vgeom
        },
        ..base
    };
    (caps.clamp_occlusion(tier.apply(requested)), tier)
}

/// **THE EDITOR VIEWPORT AND THE SHIPPED PLAYER LIGHT A LEVEL IDENTICALLY**
/// (wave FIX3, clause 2).
///
/// The FIX2 audit measured the island's hero at 232/255 in the editor viewport
/// and 1.8 in Play, and named the editor the outlier. Wave FIX3 measured the
/// ambient term instead and found the two hosts were never configured
/// differently at all — what differed was where the camera was standing, and
/// the defect was in the term both of them spent (`crates/inf-render/tests/
/// sky_ambient.rs`).
///
/// That is a claim about a tree, and it was true by inspection and by nothing
/// else. This is the arm: every committed level's authored record, through both
/// chains, on this machine's real adapter, compared **field for field** —
/// `RenderSettings` is `PartialEq`, so a field added to it in a future wave and
/// mapped by only one host fails here without anybody remembering to extend a
/// list.
#[test]
fn the_editor_viewport_and_the_shipped_player_light_a_level_identically() {
    let Ok(gpu) = inf_render::GpuContext::headless() else {
        eprintln!("SKIP: no GPU adapter");
        return;
    };
    let mut checked = 0;
    for path in committed_levels() {
        let rel = path
            .strip_prefix(repo())
            .unwrap_or(&path)
            .to_string_lossy()
            .replace(std::path::MAIN_SEPARATOR, "/");
        let bytes = std::fs::read(&path).expect("a committed level reads");
        let record = RuntimeLevel::decode(&bytes)
            .unwrap_or_else(|e| panic!("decode {rel}: {e}"))
            .settings
            .render;
        let (shipped, ship_tier) = inf_player::render::shipped_settings(&gpu, record);
        let (editor, edit_tier) = editor_settings(&gpu, record);
        assert_eq!(
            edit_tier, ship_tier,
            "{rel}: the editor viewport detects tier {edit_tier:?} and the shipped \
             player {ship_tier:?} on one adapter"
        );
        assert_eq!(
            editor, shipped,
            "{rel}: the editor viewport and the shipped player would render this \
             level with different settings.\n\neditor:\n{editor:#?}\n\nshipped:\n{shipped:#?}"
        );
        checked += 1;
    }
    assert_eq!(
        checked, EXPECTED_LEVELS,
        "the census walked {checked} levels, not {EXPECTED_LEVELS}"
    );
    // Anti-vacuity: the lit levels must not be comparing two copies of the
    // DEFAULT settings, which would pass however wrong both hosts were.
    let island = record_of(LIT_LEVELS[0]);
    let (lit, _) = inf_player::render::shipped_settings(&gpu, island);
    assert!(
        lit.gi.enabled && lit.shadows.enabled,
        "the island's settings reached neither host with GI or shadows on \
         (tier {:?}) — this arm would then be comparing two defaults",
        detect_tier(&gpu, &RenderSettings::default())
    );
    println!(
        "FIX3 clause 2: {checked} committed levels, editor viewport == shipped player, \
         field for field, on tier {:?}",
        detect_tier(&gpu, &RenderSettings::default())
    );
}

/// **The transcription above is the editor viewport's own chain**, verbatim.
///
/// A source gate, because the thing it guards cannot be reached without the
/// editor crate: `inf-viewport` is Ring 1 and `inf-player` is Ring 2, and
/// neither depends on the other. Read as text — the instrument
/// `apply_record_mirror.rs` established for exactly this shape of claim.
///
/// Mutation-verified: dropping `caps.clamp_occlusion` from either the editor's
/// source or the transcription fails this arm.
#[test]
fn the_transcribed_editor_chain_is_the_editor_viewports_own() {
    const VIEWPORT: &str = include_str!("../../../editor/crates/inf-viewport/src/host.rs");
    // The four load-bearing statements of `apply_render_settings`, in the order
    // the transcription performs them. Whitespace-collapsed so a rustfmt run
    // that rewraps a line does not fail a gate about a decision.
    let code: String = VIEWPORT.split_whitespace().collect::<Vec<_>>().join(" ");
    for needle in [
        "detect_tier(&self.gpu, &RenderSettings::default())",
        "let requested = requested_render_settings(&doc.settings().render);",
        "let mapped = caps.clamp_occlusion(tier.apply(requested));",
        "AdapterCaps::probe(&self.gpu)",
    ] {
        let want: String = needle.split_whitespace().collect::<Vec<_>>().join(" ");
        assert!(
            code.contains(&want),
            "the editor viewport no longer does `{needle}`, so `editor_settings` in \
             this file is a transcription of something that is not there any more"
        );
    }
    // …and the request itself: vgeom on, nothing else touched.
    for needle in [
        "fn requested_render_settings(record: &RenderSettingsRecord) -> RenderSettings {",
        "let base = apply_record(record);",
        "vgeom: inf_render::VgeomSettings { enabled: true, ..base.vgeom },",
    ] {
        let want: String = needle.split_whitespace().collect::<Vec<_>>().join(" ");
        assert!(
            code.contains(&want),
            "the editor viewport's request changed shape: `{needle}` is gone"
        );
    }
    // **AND THE OTHER HALF OF THE SENTENCE** (FIX3 audit). The doc above says
    // this arm is mutation-verified against "dropping `caps.clamp_occlusion`
    // from either the editor's source or the transcription". Only the first was
    // true: everything up to here reads `host.rs`, and nothing read
    // `editor_settings`. Measured at the audit — dropping `clamp_occlusion`
    // from the transcription alone left this whole binary green, because on an
    // adapter that supports everything `clamp_occlusion` clamps nothing, so the
    // field-for-field arm could not see it either (and on an adapterless CI leg
    // that arm SKIPs).
    //
    // So the transcription is pinned to the statements it transcribes, from
    // this file's own source. `include_str!` resolves against this file's
    // directory, and the body is bounded by the next `fn` so a later helper
    // cannot satisfy the scan on `editor_settings`' behalf — the P23 ruling
    // about reading a SCOPE rather than a file.
    const SELF: &str = include_str!("lit_stack.rs");
    let at = SELF
        .find("fn editor_settings(gpu: &GpuContext, record: RenderSettingsRecord)")
        .expect("this file defines `editor_settings`");
    let rest = &SELF[at..];
    let end = rest[1..].find("\nfn ").map_or(rest.len(), |i| i + 1);
    let body: String = rest[..end].split_whitespace().collect::<Vec<_>>().join(" ");
    assert!(
        body.len() > 200,
        "the `editor_settings` extraction is {} bytes — a broken scan would pass \
         every clause below",
        body.len()
    );
    for needle in [
        "let base = inf_player::render::apply_record(&record);",
        "let tier = detect_tier(gpu, &RenderSettings::default());",
        "let caps = AdapterCaps::probe(gpu);",
        "enabled: true,",
        "(caps.clamp_occlusion(tier.apply(requested)), tier)",
    ] {
        let want: String = needle.split_whitespace().collect::<Vec<_>>().join(" ");
        assert!(
            body.contains(&want),
            "`editor_settings` no longer does `{needle}`, so this file's copy of \
             the editor viewport's chain has drifted from the chain it is pinned \
             against above — and the field-for-field arm can only notice on an \
             adapter where the missing step actually clamps something"
        );
    }
    println!("FIX3 clause 2: the editor's four-statement chain is transcribed verbatim");
}
