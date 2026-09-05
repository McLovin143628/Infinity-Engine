//! **THE SEAMS BETWEEN PREVIEW AND SHIPPING ARE ONE FUNCTION** (wave VIS1a, with
//! two arms added by its audit).
//!
//! A level's render block reaches the renderer through two copies of the same
//! mapping: `inf_player::render::apply_record` over `inf_scene::RenderSettingsRecord`,
//! and `inf_viewport::host::apply_record` over the editor codec's mirror of it.
//! Both have carried a `MIRROR: keep identical to …` doc comment since R-P4 and
//! **nothing has ever checked it**. A comment is not a pin: the two are in
//! different crates, in different rings, and the failure mode is silent —
//! preview and shipping applying a level's block differently, which is the exact
//! claim `PIE == shipping` rests on.
//!
//! Wave VIS1a doubled the size of that mapping (twenty-two appended fields), so
//! it is also the wave that has to stop trusting the sentence.
//!
//! **What is compared**: the *body* of each `fn apply_record`, with comments and
//! whitespace stripped — and nothing else normalised, deliberately. The two
//! bodies name the same types by the same identifiers (`RenderSettingsRecord`,
//! `SsrSettings`, …); which crate each identifier resolves to is settled by the
//! `use` lines at the top of each file, which are outside the extracted body. So
//! a *character-for-character* comparison is the strictest available and needs no
//! substitution table that could be widened later to hide a drift. What is
//! deliberately NOT compared is the doc comment above each — they say different
//! things about which copy is which, and they should.
//!
//! **Both files are read as text at compile time** (`include_str!`), so this arm
//! has to be built and run in the same worktree — the P23 `determinism_law`
//! constraint, documented on the gate rather than worked around. `.rs` is
//! `text eol=lf` in `.gitattributes`, which is what makes the substring search
//! below survive a Windows checkout (the P22 CRLF law).
//!
//! **The audit added two more arms and a third source.** `apply_record` was never
//! the only `MIRROR:` pair in these two files — `prim_mesh` carries the identical
//! sentence and was equally unchecked — and a character-for-character comparison
//! cannot see a field that *both* copies forget, which is the one failure the
//! wave's own "an unread wire field is not cheap" ruling is about. So the record's
//! own declaration is read too, and every field of it has to be consumed by both
//! seams.

/// The player's copy.
const PLAYER: &str = include_str!("../src/render.rs");
/// The editor viewport's copy.
const VIEWPORT: &str = include_str!("../../../editor/crates/inf-viewport/src/host.rs");
/// The record itself — read as text so the field LIST can be compared with what
/// the two seams actually consume (wave VIS1a audit).
const RECORD: &str = include_str!("../../../crates/inf-scene/src/lib.rs");

/// The body of the function whose signature starts with `sig` — from the opening
/// brace to the matching close — with comments stripped and whitespace collapsed.
fn body(src: &str, sig: &str, what: &str) -> String {
    let at = src
        .find(sig)
        .unwrap_or_else(|| panic!("{what}: no `{sig}`"));
    let open = src[at..]
        .find('{')
        .expect("a function signature is followed by a brace")
        + at;
    let mut depth = 0usize;
    let mut end = open;
    for (i, c) in src[open..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    end = open + i;
                    break;
                }
            }
            _ => {}
        }
    }
    assert!(end > open, "{what}: unbalanced braces in `apply_record`");
    let mut out = String::new();
    for line in src[open + 1..end].lines() {
        let code = match line.find("//") {
            Some(i) => &line[..i],
            None => line,
        };
        let t = code.trim();
        if t.is_empty() {
            continue;
        }
        out.push_str(t);
        out.push('\n');
    }
    out
}

/// **The pin.** Two crates, two rings, one mapping.
///
/// Mutation-verified while it was written: changing a single default on one side
/// (`..d.gi` to `..GiSettings::default()`) fails it, and so does dropping any one
/// of the twenty-two v26 lines from either copy.
#[test]
fn the_two_apply_record_seams_are_character_for_character_the_same() {
    let sig = "fn apply_record(r: &RenderSettingsRecord) -> RenderSettings {";
    let a = body(PLAYER, sig, "inf_player::render");
    let b = body(VIEWPORT, sig, "inf_viewport::host");

    // Anti-vacuity: a bad extraction that returned nothing would compare equal.
    assert!(
        a.lines().count() > 30,
        "the player's `apply_record` body extracted as {} lines — the extraction \
         is broken, and an empty comparison passes for the wrong reason",
        a.lines().count()
    );
    assert!(
        a.contains("ssr: SsrSettings {") && a.contains("film: FilmSettings {"),
        "the extracted body does not contain the v26 block — either the seam \
         moved or the extraction did"
    );

    assert_eq!(
        a, b,
        "the two `apply_record` seams have drifted. The editor viewport and the \
         shipped player would apply a level's render block differently, which is \
         the claim `PIE == shipping` rests on — and the only thing that has ever \
         kept them together is a doc comment saying they are the same.\n\n\
         player:\n{a}\n\nviewport:\n{b}"
    );
}

/// **THE OTHER MIRROR THE SAME COMMENT GUARDS** (wave VIS1a audit).
///
/// `apply_record` is not the only `MIRROR: keep identical to …` pair in these two
/// files: `prim_mesh` — the mapping from an authored `Primitive` to the geometry
/// the renderer builds — carries the identical sentence and was equally unchecked.
/// It is the same claim in a different currency: if the two drift, the editor
/// viewport draws a different *shape* from the shipped player, which `PIE ==
/// shipping` rests on exactly as much as the render block does.
///
/// Pinning it costs one arm because the wave already built the extractor. The two
/// bodies were verified identical before this was added, so it lands green rather
/// than as a discovered drift.
#[test]
fn the_two_prim_mesh_seams_are_character_for_character_the_same() {
    let sig = "fn prim_mesh(";
    let a = body(PLAYER, sig, "inf_player::render");
    let b = body(VIEWPORT, sig, "inf_viewport::host");
    assert!(
        a.lines().count() >= 4 && a.contains("Primitive::"),
        "the player's `prim_mesh` extracted as {} lines — the extraction is broken",
        a.lines().count()
    );
    assert_eq!(
        a, b,
        "the two `prim_mesh` seams have drifted: the editor viewport and the shipped \
         player would build different geometry for the same authored primitive.\n\n\
         player:\n{a}\n\nviewport:\n{b}"
    );
}

/// **EVERY PERSISTED FIELD IS ACTUALLY APPLIED** (wave VIS1a audit).
///
/// The wave's ruling was *"an unread field is cheap where an unread **wire** field
/// is not"*, and it is the whole justification for landing twenty-two fields five
/// of whose blocks have no consumer until VIS1b. What enforced it was two
/// `contains` checks for the strings `ssr: SsrSettings {` and
/// `film: FilmSettings {` — which pass with any number of the fields inside those
/// literals missing.
///
/// This reads the field names out of `pub struct RenderSettingsRecord` in
/// `inf_scene` and requires each one to appear as `r.<name>` in **both** seams. A
/// field that is persisted, crosses the IPC, is drawn in a panel and then reaches
/// no renderer setting is exactly the promise the ruling says the engine must not
/// make, and it is now the arm rather than the sentence.
///
/// Mutation-verified: deleting `intensity: r.ssr_intensity,` from either copy
/// fails it — which the character-for-character comparison above does **not**,
/// because deleting it from *both* leaves them identical.
#[test]
fn every_field_of_the_render_record_is_read_by_both_seams() {
    let at = RECORD
        .find("pub struct RenderSettingsRecord {")
        .expect("inf_scene declares the record");
    let end = RECORD[at..].find("\n}").expect("the struct closes") + at;
    let fields: Vec<&str> = RECORD[at..end]
        .lines()
        .filter_map(|l| l.trim().strip_prefix("pub "))
        .filter_map(|l| l.split(':').next())
        .filter(|n| !n.is_empty() && n.chars().all(|c| c.is_ascii_lowercase() || c == '_'))
        .collect();
    // Anti-vacuity: v26 has thirty-seven fields; an extractor that found a
    // handful would pass while proving nothing.
    assert!(
        fields.len() >= 35,
        "extracted only {} field names from `RenderSettingsRecord` — the extraction \
         is broken, not the seams: {fields:?}",
        fields.len()
    );

    let sig = "fn apply_record(r: &RenderSettingsRecord) -> RenderSettings {";
    for (what, src) in [
        ("inf_player::render", PLAYER),
        ("inf_viewport::host", VIEWPORT),
    ] {
        let seam = body(src, sig, what);
        let missing: Vec<&&str> = fields
            .iter()
            .filter(|f| !seam.contains(&format!("r.{f}")))
            .collect();
        assert!(
            missing.is_empty(),
            "{what}'s `apply_record` never reads {} of the record's {} fields: \
             {missing:?}. A value an author can type, the codec persists and no \
             seam consumes is a promise the engine is not keeping — which is the \
             ruling this wave landed twenty-two fields under.",
            missing.len(),
            fields.len()
        );
    }
}

// ── the other half of "the two hosts light one world" (wave FIX3) ───────────

/// **A HIDDEN ACTOR MUST BE DARK IN BOTH HOSTS.**
///
/// The player's projection loop is gated once, at the top:
/// `if !visible { continue; }`, so hiding anything hides everything it
/// contributes — its mesh, its scatter and its LIGHTS. The editor viewport
/// applies visibility per COMPONENT instead, and until this wave the venue rig
/// (`PcgVolume::lights`, wave VEN1a) sat outside every guard: hiding a venue
/// volume in the viewport left its stage lamps burning, and the same level in
/// PIE went dark.
///
/// That is a lighting divergence between preview and shipping on a level the
/// island actually contains, and it is exactly the class of thing wave FIX3 was
/// sent to find. It could not be caught by a runtime arm — no crate depends on
/// both hosts — so it is caught here, in the file the repository already uses to
/// hold the two projections together, and by reading the source.
///
/// Mutation-verified: dropping `.filter(|_| visible)` from `host.rs` fails this.
#[test]
fn every_light_the_editor_pushes_is_behind_a_visibility_check() {
    let code: String = VIEWPORT.split_whitespace().collect::<Vec<_>>().join(" ");
    assert!(
        code.contains("if let Some(vol) = w.get::<PcgVolume>(entity).filter(|_| visible) {"),
        "the editor viewport's `PcgVolume` branch is no longer visibility-gated,          so a hidden venue volume keeps its rig lit in the viewport and loses it          in PIE"
    );
    assert!(
        code.contains("if let Some(light) = w.get::<Light>(entity) { if visible {"),
        "the editor viewport's `Light` branch is no longer visibility-gated"
    );
    // …and the player's single gate, which is what the editor is being held to.
    let player: String = PLAYER.split_whitespace().collect::<Vec<_>>().join(" ");
    assert!(
        player.contains("if !visible { continue; }"),
        "the shipped player's projection loop no longer skips invisible entities,          so the claim the editor is measured against is gone"
    );
}
