//! The **projector MIRROR gate** (P17.1, extended P18.3 and P18.5): the editor
//! viewport's ECS→`RenderScene` projection and the shipped player's must not
//! drift.
//!
//! Three things are pinned here. `project_sky` is compared **character for
//! character** — it is a self-contained function on both sides. The `MeshRef`
//! branch that projects **real geometry** (P18.3) cannot be: it is inline in two
//! loops with different iteration orders, different id bookkeeping and different
//! asset stores. So it is pinned **field for field** instead — every field of the
//! `VgeomInstance` both sides construct, in order, with identical value
//! expressions for everything except the two host-local ones. That is exactly as
//! strong where it matters: the failure this exists to catch is "the editor
//! forgot to project `emissive`", or "the player gained a field the editor never
//! fills", either of which reads as *the shipped game looks different from the
//! preview* and is found by a player, not by a compiler.
//!
//! The third is P18.5's **GPU-instanced scatter** — PCG volumes and painted
//! foliage. It had no gate at all before this batch, which is how the two hosts
//! came to disagree about `PcgVolume::draw_distance` for two phases: the editor
//! culled against its own camera eye on the CPU and the player ignored the field
//! entirely, so a shipped build drew strictly more scatter than its preview. Now
//! the field rides on the batch, the GPU cull honours it for both hosts, and the
//! `ScatterBatch` literal is pinned field for field like the vgeom one.
//!
//! # Why it lives here and not next to either projector
//!
//! `inf_viewport::host` is `#[cfg(any(windows, target_os = "macos"))]` — a test
//! inside it is invisible to the Linux CI leg, which is exactly the leg most
//! likely to be the one a contributor's PR runs first. `inf-editor-core` compiles
//! on all three platforms and sits in the same workspace, so the comparison runs
//! everywhere. Nothing here links either crate; it reads their **source text**,
//! which is the whole point: the duplication is deliberate and the gate is that
//! the duplicate has not drifted.
//!
//! # Why the duplication is deliberate
//!
//! The part that could *silently* diverge — which entity is the sky authority,
//! given that the editor walks document order and the player walks `Guid` order —
//! lives in `inf_ecs::sky` and is shared. What is left is a ~30-line mapping from
//! `inf_ecs` types into `inf_render` types, and **neither Ring-0 crate can host
//! it**: `inf-render` does not depend on `inf-ecs`, and `inf-ecs` must not depend
//! on `inf-render`. So it is written twice on purpose — and compared here.
//!
//! The classic bug this exists to catch surfaces only as "the shipped game lights
//! differently from the preview", which is precisely the class of thing that is
//! discovered by a player, not by a test.

mod support;

use std::path::{Path, PathBuf};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("..")
}

// The positional anchor lives in `support` since the P24.1 re-audit's F2: it was
// re-implemented naively in a second test binary the same day it was fixed here,
// because a helper another binary cannot see is a helper that gets written twice.
// The reasoning that made it positional — a backticked self-reference inside
// `project_voxel`'s own doc comment, which drifted two blocks to 23 lines against
// 17 with this file green — is recorded on `support::item_start`.
use support::item_start;

/// The **item text** — the signature line (qualifiers included) through the
/// closing brace at column 0 — with line endings normalized (the two files can be
/// checked out with different EOLs, which says nothing about whether the code
/// drifted).
///
/// Used by the "not a stub" guards, which must read *code* and not prose: a
/// fragment assertion that can be satisfied by a sentence in a doc comment is a
/// phantom guard.
fn extract_item(source: &str, name: &str) -> String {
    let source = source.replace("\r\n", "\n");
    let start = item_start(&source, name);
    let line_start = source[..start].rfind('\n').map_or(0, |i| i + 1);
    let rest = &source[line_start..];
    let end = rest
        .find("\n}\n")
        .unwrap_or_else(|| panic!("`fn {name}(` does not terminate at column 0"))
        + 3;
    rest[..end].to_string()
}

/// **What every mirror-equality gate below compares: the doc block AND the item.**
///
/// The doc block is not decoration on a mirrored pair. Every rule the two copies
/// share — which of two values wins, what an empty projection means, why a field is
/// host-local — is written in the `///` lines and nowhere else, so a rule corrected
/// on one side only is the same defect as a line of code changed on one side only:
/// the next person to read the stale copy implements the stale rule.
///
/// It went uncompared for two batches. The anchor used to be the first
/// `fn <name>(` **substring** in the file, which for `project_voxel` was a
/// backticked self-reference inside its own doc comment — so extraction began
/// mid-comment, every line above it was invisible, and the two blocks drifted to 23
/// lines against 17 with this assertion green the whole time. The sentence that
/// drifted was the one *claiming* the doc block was uncovered, so the gate could
/// not catch the defect that motivated it. [`item_start`] is what closes it.
///
/// Attribute lines between the doc block and the item are **stepped over, not
/// compared**: `#[cfg(not(target_arch = "wasm32"))]` above the player's
/// `skinned_mesh_data` is a fact about which host builds for wasm, not about
/// whether the two hosts agree.
fn extract_fn(source: &str, name: &str) -> String {
    let source = source.replace("\r\n", "\n");
    let start = item_start(&source, name);
    let item_line = source[..start].rfind('\n').map_or(0, |i| i + 1);
    // Walk back over the item's attributes and doc comment, keeping the doc lines.
    let mut doc: Vec<&str> = Vec::new();
    let mut cursor = item_line;
    while cursor > 0 {
        let prev_start = source[..cursor - 1].rfind('\n').map_or(0, |i| i + 1);
        let line = &source[prev_start..cursor - 1];
        let trimmed = line.trim_start();
        if trimmed.starts_with("///") {
            doc.push(line);
        } else if !trimmed.starts_with("#[") {
            break;
        }
        cursor = prev_start;
    }
    // The guard on the guard: an undocumented item would quietly reduce this back
    // to `extract_item`, which is precisely the hole that was just closed.
    assert!(
        !doc.is_empty(),
        "`fn {name}` carries no doc comment, so this gate would compare only its \
         body — the exact hole the anchor fix closed. Document it on both sides, or \
         compare it with `extract_item`."
    );
    doc.reverse();
    let mut out = String::new();
    for line in doc {
        out.push_str(line);
        out.push('\n');
    }
    out.push_str(&extract_item(&source, name));
    out
}

/// The text of `fn <name>(` through its **brace-balanced** end — the indented
/// twin of [`extract_fn`], which can only find a free function that terminates at
/// column 0. Needed for the skeletal stores, whose shared rules are `impl`
/// methods on each host's own store type.
fn extract_method(source: &str, name: &str) -> String {
    let source = source.replace("\r\n", "\n");
    // Positionally anchored, exactly like the free-function extractor: a bare
    // `source.find("fn <name>(")` matches the first MENTION, and for
    // `project_voxel` that was a backticked self-reference inside the function's
    // own doc comment (the P21.2-F1 defect this file documents at length).
    // `item_start`'s rule — everything between the line start and the `fn` keyword
    // must be item qualifiers — holds for an indented `impl` method too:
    // `    pub fn foo(` leaves `pub`, and a `///` or a backtick fails it.
    let start = item_start(&source, name);
    let mut depth = 0usize;
    for (i, c) in source[start..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return source[start..start + i + 1].to_string();
                }
            }
            _ => {}
        }
    }
    panic!("`fn {name}(` does not terminate");
}

/// [`extract_method`] **plus its doc block** — the indented twin of
/// [`extract_fn`], and what a mirrored `impl` method must actually be compared
/// with.
///
/// P24.1 closed this hole. `extract_method` reads the item only, so for two
/// batches the skeletal stores' `resolve_skinned` doc blocks were free to
/// diverge — and they had: one side carried a paragraph naming the other file
/// that the other side could not carry. That is exactly the defect
/// [`extract_fn`]'s own doc argues against: **every rule the two copies share is
/// written in the `///` lines and nowhere else**, so a rule corrected on one side
/// only is the same defect as a line of code changed on one side only. The
/// remedy is the one `skinned_mesh_data` already uses — side-neutral wording,
/// with any "the twin lives at …" note in the module docs, which are not a
/// mirrored pair.
fn extract_method_with_doc(source: &str, name: &str) -> String {
    let source = source.replace("\r\n", "\n");
    // See `extract_method`: the anchor is positional, never the first mention.
    // Getting this wrong matters MORE here — a doc-comment anchor makes the
    // backward walk below start inside the doc block and silently compare a
    // fragment of it against a whole one.
    let start = item_start(&source, name);
    let item_line = source[..start].rfind('\n').map_or(0, |i| i + 1);
    let mut doc: Vec<&str> = Vec::new();
    let mut cursor = item_line;
    while cursor > 0 {
        let prev_start = source[..cursor - 1].rfind('\n').map_or(0, |i| i + 1);
        let line = &source[prev_start..cursor - 1];
        let trimmed = line.trim_start();
        if trimmed.starts_with("///") {
            doc.push(line);
        } else if !trimmed.starts_with("#[") {
            break;
        }
        cursor = prev_start;
    }
    assert!(
        !doc.is_empty(),
        "`fn {name}` carries no doc comment, so this gate would compare only its \
         body — the same hole the free-function anchor fix closed."
    );
    doc.reverse();
    let mut out = String::new();
    for line in doc {
        out.push_str(line);
        out.push('\n');
    }
    out.push_str(&extract_method(&source, name));
    out
}

fn read(rel: &str) -> String {
    let path = workspace_root().join(rel);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

const VIEWPORT: &str = "editor/crates/inf-viewport/src/host.rs";
const PLAYER: &str = "runtime/inf-player/src/render.rs";

/// One `// MIRROR-BEGIN <tag>` … `// MIRROR-END <tag>` region, with every
/// whitespace character filtered out.
///
/// The fences are **counted**, not just found: a `contains` needle that is a
/// prefix of a declaration can never fail, and neither can a second fence (the
/// I1 audit's law, applied to a delimiter).
///
/// Hoisted out of `both_projectors_band_a_structure_lod_the_same_way` when the
/// scatter-memo arm below joined it (island wave I8a audit) — two nested copies
/// of a delimiter reader is how two fences come to be read by two different
/// rules.
fn fenced(src: &str, tag: &str, who: &str) -> String {
    let (b, e) = (
        format!("// MIRROR-BEGIN {tag}"),
        format!("// MIRROR-END {tag}"),
    );
    assert_eq!(src.matches(&b).count(), 1, "{who}: {tag} begin fences");
    assert_eq!(src.matches(&e).count(), 1, "{who}: {tag} end fences");
    let (i, j) = (
        src.find(&b).expect("checked"),
        src.find(&e).expect("checked"),
    );
    assert!(j > i, "{who}: the {tag} fence is inverted");
    src[i..j].chars().filter(|c| !c.is_whitespace()).collect()
}

/// The editor's loose-file render-asset store (P18.3) — where the *skeletal*
/// resolution + pose rule lives on the editor side.
const EDITOR_ASSETS: &str = "editor/crates/inf-editor-core/src/render_assets.rs";
/// The shipped player's skeletal render-asset store — the same rules over a
/// cooked pack / dev dir.
const PLAYER_ASSETS: &str = "runtime/inf-player/src/skinned.rs";

#[test]
fn project_sky_is_identical_in_both_projectors() {
    let mine = extract_fn(&read(VIEWPORT), "project_sky");
    let theirs = extract_fn(&read(PLAYER), "project_sky");
    assert_eq!(
        mine, theirs,
        "the two `project_sky` projectors have drifted — PIE would stop matching \
         shipping. Keep them byte-identical, or move the shared part into \
         `inf_ecs::sky` (which is where the authority-resolution rule already lives)."
    );
}

/// A guard on the guard: if either projector's `project_sky` were reduced to a
/// stub, the identity check above would still pass. Assert the shared body
/// actually does the work — it must read the resolved sky, write both renderer
/// blocks, and publish the key light.
#[test]
fn the_shared_projector_body_is_not_a_stub() {
    let body = extract_item(&read(VIEWPORT), "project_sky");
    for fragment in [
        "inf_ecs::sky::resolve_sky",
        "scene.sun = SunParams",
        "scene.sky = SkyParams",
        "sky.sky_gradient()",
        "sky.key_light()",
        "scene.lights.push",
        "SunParams::default()",
        "SkyParams::default()",
        // P17.2: the physical atmosphere rides the same projection. Both the
        // authored block and the no-authority reset must be present, or a level
        // with a clock would render an atmosphere-less sky in one host and a
        // physical one in the other.
        "scene.atmosphere = AtmosphereParams",
        "AtmosphereParams::default()",
        "enabled: a.physical",
        "fog: HeightFog",
        "moon_phase: phase",
        // P17.3: the volumetric-cloud block. `time_s` is the fragment that
        // matters most — it is the only *derived* field in the projection, and a
        // host that fed it a frame counter or a wall clock instead of
        // `ResolvedSky::cloud_time_s()` would drift the two skies apart while
        // every other assertion here still passed.
        "clouds: CloudParams",
        "enabled: a.clouds_enabled",
        "time_s: sky.cloud_time_s()",
        "seed: a.cloud_seed",
        "shadow_strength: a.cloud_shadow",
        // P17.4: the weather block. `let w = sky.weather()` is the fragment that
        // matters — it is the Ring-0 decision about which of two parameter sets
        // is in force, and a host that inlined its own `if weather_enabled`
        // would be exactly the divergence this gate exists to stop. The three
        // *driven* assignments are here too, because a host could call
        // `sky.weather()` and then keep reading the authored fields, which would
        // pass a fragment check that only looked for the call.
        "let w = sky.weather();",
        "density: w.fog_density",
        "coverage: w.cloud_coverage",
        "cloud_type: w.cloud_type",
        "wind_x: w.wind_x",
        "precip: PrecipParams",
        "intensity: w.precipitation",
        "snowiness: w.snowiness",
    ] {
        assert!(
            body.contains(fragment),
            "`project_sky` no longer contains `{fragment}` — either it was gutted, \
             or this gate needs updating deliberately:\n{body}"
        );
    }
}

// ── P18.3: the real-geometry (`MeshRef.asset`) projection ────────────────────

/// The ordered `(field, value)` pairs of the **first** `<ty> { … }` struct literal
/// in `source`.
///
/// Deliberately naive — it takes lines until the first one that closes the
/// literal — because the thing being compared is a flat struct literal, and a
/// parser clever enough to handle anything else would be clever enough to hide a
/// drift. Comments and blank lines are dropped; a `field,` shorthand yields a
/// value equal to the field name, so `translation,` and `translation: translation`
/// compare equal (they mean the same thing and either is idiomatic).
fn struct_literal_fields(source: &str, ty: &str) -> Vec<(String, String)> {
    let source = source.replace("\r\n", "\n");
    let open = format!("{ty} {{");
    let start = source
        .find(&open)
        .unwrap_or_else(|| panic!("no `{open}` literal — did the projection move?"))
        + open.len();
    let mut out = Vec::new();
    for line in source[start..].lines() {
        let t = line.trim();
        if t.starts_with('}') {
            return out;
        }
        if t.is_empty() || t.starts_with("//") {
            continue;
        }
        let t = t.trim_end_matches(',');
        let (name, value) = match t.split_once(':') {
            Some((n, v)) => (n.trim().to_string(), v.trim().to_string()),
            None => (t.to_string(), t.to_string()),
        };
        out.push((name, value));
    }
    panic!("the `{ty}` literal does not terminate");
}

fn vgeom_instance_fields(source: &str) -> Vec<(String, String)> {
    struct_literal_fields(source, "VgeomInstance")
}

/// Fields whose value expression is **host-local by design** and therefore
/// excluded from the value comparison (their presence and order still are not).
///
///  * `asset` — the player keys a vgeom asset by its derived **GUID** (a cooked
///    pack is immutable, so an id names one sequence of bytes forever); the editor
///    keys it by the derived payload's **content hash**, because a project's
///    content root is not immutable and both render nodes cache GPU state by this
///    id. The reasoning lives once, in `inf_editor_core::render_assets`.
///  * `id` — the pick id, allocated from each host's own counter over its own
///    iteration order (document order vs `Guid` order). It has never matched and
///    is not meant to.
const HOST_LOCAL_FIELDS: [&str; 2] = ["asset", "id"];

/// **The P18.3 mirror gate.** Both projectors build a `VgeomInstance` from the
/// same ECS state; every field must be present on both sides, in the same order,
/// carrying the same expression — except the two documented host-local ones.
#[test]
fn the_vgeom_instance_projection_matches_field_for_field() {
    let mine = vgeom_instance_fields(&read(VIEWPORT));
    let theirs = vgeom_instance_fields(&read(PLAYER));

    let names = |v: &[(String, String)]| v.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>();
    assert_eq!(
        names(&mine),
        names(&theirs),
        "the two `VgeomInstance` projections carry different fields (or in a \
         different order) — a field projected on one side and not the other means \
         the shipped game draws an imported mesh differently from the preview"
    );

    for ((n, a), (_, b)) in mine.iter().zip(&theirs) {
        if HOST_LOCAL_FIELDS.contains(&n.as_str()) {
            continue;
        }
        assert_eq!(
            a, b,
            "`VgeomInstance::{n}` is projected as `{a}` in the editor viewport and \
             `{b}` in the shipped player. Keep them identical, or — if the \
             difference is deliberate — add the field to `HOST_LOCAL_FIELDS` with \
             the reason written down."
        );
    }
    // A guard on the guard: an empty literal would satisfy everything above.
    assert!(
        mine.len() >= 9,
        "the `VgeomInstance` projection shrank to {} fields — was it gutted?",
        mine.len()
    );
}

/// The surrounding *rules* — not just the literal — must exist on both sides:
/// resolution through the derived id, per-frame asset dedup, the paged source
/// handed to the scene rather than a decoded DAG, the primitive fallback, and
/// (wave FIX2) the arm that draws NOTHING for a bound mesh whose DAG is missing.
///
/// Without this a projector could satisfy the field comparison while listing the
/// asset twice, or while never falling back to the primitive at all (which would
/// make a primitive-only `MeshRef` invisible), or while keeping the placeholder
/// cube on one host after the other deleted it — which is the preview and the
/// build disagreeing about a box, in the file that exists to stop exactly that.
#[test]
fn both_projectors_keep_the_real_geometry_rules() {
    for (label, path, fragments) in [
        (
            "editor viewport",
            VIEWPORT,
            [
                // Resolution goes through the mesh asset id, never a side index.
                "mesh_ref.asset",
                "resolve_vgeom(mesh_id)",
                // One asset entry per frame, however many instances reference it.
                "vgeom_seen.insert(",
                "vgeom_assets",
                "VgeomAsset::new(",
                // The scene carries the PAGED source (P18.2), not a decoded DAG.
                "loaded.source",
                // …and a primitive-only MeshRef still draws its built-in
                // primitive kind rather than vanishing.
                "prim_mesh(mesh_ref.primitive)",
                // …while a BOUND one whose DAG is missing draws nothing at all
                // (wave FIX2). The two arms are one decision and they must be the
                // same decision on both sides: a placeholder kept on one host is
                // a box in the preview and a hole in the build, or the reverse.
                "mesh_ref.asset.is_some()",
            ],
        ),
        (
            "shipped player",
            PLAYER,
            [
                "mesh_ref.asset",
                "vmeshes.resolve(mesh_id)",
                "vgeom_seen.insert(",
                "vgeom_assets",
                "VgeomAsset::new(",
                "source",
                "prim_mesh(mesh_ref.primitive)",
                "mesh_ref.asset.is_some()",
            ],
        ),
    ] {
        let src = read(path).replace("\r\n", "\n");
        for fragment in fragments {
            assert!(
                src.contains(fragment),
                "the {label}'s `MeshRef` projection no longer contains `{fragment}` \
                 — either the real-geometry path was changed on one side only, or \
                 this gate needs updating deliberately"
            );
        }
    }
}

/// **Both hosts must OPT IN to the meshlet path** (P18.3 audit).
///
/// `VgeomSettings::default()` is `enabled: false`, so carrying vgeom content is
/// not enough — a host that never asks draws all of it through the classic
/// discrete-LOD fallback. That failure is invisible: the fallback renders the
/// *same geometry*, so the only symptom is that none of P18.2's streaming, budget
/// or eviction is running, which no screenshot shows. The player has always asked;
/// the editor did not until this batch, and this is what keeps both honest.
///
/// The opt-in is the *request* — the tier clamp still has the last word on both
/// sides, which is why `RenderTier::apply` appears here too.
#[test]
fn both_hosts_request_the_meshlet_path() {
    for (label, path) in [("editor viewport", VIEWPORT), ("shipped player", PLAYER)] {
        // Read through the comment/string stripper, not raw: every needle below
        // is a claim about what the host *executes*, and a prose sentence about
        // a clamp satisfies a raw `contains` exactly as well as the call does.
        // Measured, not feared — the fix that made the player pass the capability
        // arm carries a comment naming `clamp_occlusion`, and with that comment
        // in place the arm went on passing after the call itself was deleted.
        let src = support::strip_comments_and_strings(&read(path).replace("\r\n", "\n"));
        // The request itself: `VgeomSettings { enabled: true, .. }` over the
        // level's authored block.
        assert!(
            src.contains("VgeomSettings {") && src.contains("enabled: true"),
            "the {label} never requests `vgeom.enabled = true`, so every imported \
             mesh it carries would draw through the classic fallback"
        );
        // …and the clamps that can still take it away, so "requesting" never
        // becomes "forcing" on an adapter that cannot run it. There are TWO of
        // them and a host owes both: the tier clamp says how much GPU there is,
        // the capability clamp says which features exist. The first spelling of
        // this arm accepted either one, and the shipped player passed it while
        // applying only the tier — so `vgeom.occlusion`, the scatter budget and
        // P26.1's `vt.bc_tiles` were granted there on adapters that cannot run
        // them, and clamped in the editor. Both are named now.
        assert!(
            src.contains(".apply(") || src.contains("detect_and_clamp"),
            "the {label} applies no tier clamp to its request"
        );
        assert!(
            src.contains("clamp_occlusion") || src.contains("detect_and_clamp"),
            "the {label} applies no adapter CAPABILITY clamp, so a feature the \
             adapter does not expose stays switched on there and is switched off \
             in the other host"
        );
    }
}

/// **Both projectors resolve a surface's virtual textures through the ONE
/// expression** (P26.4, clause 0).
///
/// The three instance literals already compare field-for-field above, and every
/// one of them now spells `vt` by shorthand — which is exactly why this arm has
/// to exist: a shorthand field comparison cannot see how the local was computed.
/// A host that read a different component, or reached past
/// `inf_render::vt_set_for` into `VtTextures::set_for` with its own slot order,
/// would satisfy the literal comparison perfectly and texture a surface
/// differently from the other side.
///
/// Read through the comment/string stripper for the reason the meshlet arm above
/// records: a prose sentence naming a call satisfies a raw `contains` as well as
/// the call does, and that has happened here before.
#[test]
fn both_projectors_resolve_a_surface_texture_set_the_same_way() {
    for (label, path) in [("editor viewport", VIEWPORT), ("shipped player", PLAYER)] {
        let src = support::strip_comments_and_strings(&read(path).replace("\r\n", "\n"));
        // The ONE door — not `set_for`, not a hand-built `VtTextureSet`.
        assert!(
            src.contains("inf_render::vt_set_for("),
            "the {label} does not resolve its texture sets through \
             `inf_render::vt_set_for`, so the two hosts have two rules for \
             whether a surface is textured"
        );
        // …fed by the SCENE's own binding (`Material.asset`, scene v22) and not
        // by anything the host resolved for itself.
        assert!(
            src.contains("m.asset.map(|a| a.as_u128())"),
            "the {label} does not feed `vt_set_for` from `Material.asset`"
        );
        // …and the absent-material fallback is the scalar surface, spelled the
        // same way on both sides.
        assert!(
            src.contains("inf_render::VtTextureSet::NONE"),
            "the {label} has no explicit no-material fallback"
        );
        // …and the level itself is built through the same door, in the same
        // file as the projection that reads it, so a host cannot texture its
        // surfaces from a registry the other one never builds.
        assert!(
            src.contains("build_vt_level("),
            "the {label} never builds a virtual-texture level, so every surface \
             it draws is textureless while the other host's are not"
        );
    }

    // …and the EDITOR's rebuild is gated on the asset-index generation as well
    // as the binding set (P26.4 audit). Only the editor: the player's material
    // content arrives once, from a pack or a payload, and cannot be re-imported
    // under it.
    //
    // **Both terms now live in one value** (P26.5): `VtLevelKey`, produced by
    // `EditorRenderAssets::vt_level_key`. This is still a source pin — the
    // behaviour needs a live `EngineHost`, which needs a window and a device —
    // but it pins ONE call rather than a three-term condition, and the rule the
    // call implements is now executed end to end, through the real registration
    // door on a real device, by `tests/vt_level_key.rs`. Without the generation
    // term, re-importing a `.inf_tex` changes neither the document version nor
    // the binding set and the viewport keeps the atlas it built the first time —
    // which is what shipped, under a doc comment saying the opposite.
    let src = support::strip_comments_and_strings(&read(VIEWPORT).replace("\r\n", "\n"));
    assert!(
        src.contains("vt_level_key(doc)"),
        "the editor viewport's virtual-texture rebuild does not take an \
         `EditorRenderAssets::vt_level_key`, so its cache key is spelled locally \
         again and a term can go missing from it without a test noticing — which \
         is exactly how the index generation came to be absent"
    );
    // …and it compares the WHOLE key, so a term added to `VtLevelKey` cannot be
    // ignored here by comparing one field of it.
    assert!(
        src.contains("if key == self.vt_level_key"),
        "the viewport does not compare the whole virtual-texture key"
    );
}

// ── P18.5: the GPU-instanced scatter (PCG + foliage) projection ──────────────

/// The source of ONE component branch in a projector's entity loop: from the
/// `w.get::<Component>(entity)` probe that opens it to the probe that opens the
/// next branch.
///
/// Both hosts walk their world in one loop of `if let Some(c) = w.get::<C>(entity)`
/// branches, so the probes delimit them exactly. The end probe is passed in per
/// host because the branch that *follows* foliage differs (the editor projects
/// colliders next, the player meshes) — and if either host reorders its loop this
/// gate fails loudly, which is the correct outcome for a deliberate change.
fn branch_region(source: &str, from: &str, to: &str) -> String {
    let source = source.replace("\r\n", "\n");
    let start = source
        .find(from)
        .unwrap_or_else(|| panic!("`{from}` not found — did the projection loop change?"));
    let rest = &source[start..];
    let end = rest
        .find(to)
        .unwrap_or_else(|| panic!("`{to}` does not follow `{from}` — the loop was reordered?"));
    rest[..end].to_string()
}

/// Fields of the `ScatterBatch` literal whose value expression is **host-local by
/// design** and therefore excluded from the value comparison (their presence and
/// order still are not).
///
///  * `id` — the pick id, allocated from each host's own counter over its own
///    iteration order (document order vs `Guid` order). It has never matched
///    across hosts and is not meant to; the editor additionally maps it back to a
///    GUID, which the player has no use for. The two projectors happen to spell it
///    with the same token today because the batch is built by a shared helper, but
///    the *value* is host-local on principle and a host that inlined its own id
///    allocation must not trip this gate.
///
/// Everything else — the payload, the anchor, the material constants and the
/// content draw distance — is the same expression on both sides, because a scatter
/// that shades or culls differently in the shipped build than in the preview is
/// exactly the bug this file exists to catch.
const SCATTER_HOST_LOCAL_FIELDS: [&str; 1] = ["id"];

/// **The P18.5 mirror gate.** Both projectors build a `ScatterBatch` from the same
/// ECS state; every field must be present on both sides, in the same order,
/// carrying the same expression — except the documented host-local one.
///
/// The literal compared is the **first** in each file, which is the shared
/// `push_scatter` body (P19.3 hoisted it out of `push_pcg_scatter` so the volume
/// path and the terrain biome population cannot drift): both hosts define
/// `push_scatter` before `push_foliage_scatter`.
#[test]
fn the_scatter_batch_projection_matches_field_for_field() {
    let mine = struct_literal_fields(&read(VIEWPORT), "ScatterBatch");
    let theirs = struct_literal_fields(&read(PLAYER), "ScatterBatch");

    let names = |v: &[(String, String)]| v.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>();
    assert_eq!(
        names(&mine),
        names(&theirs),
        "the two `ScatterBatch` projections carry different fields (or in a \
         different order) — a field projected on one side and not the other means \
         the shipped game draws PCG/foliage scatter differently from the preview"
    );

    for ((n, a), (_, b)) in mine.iter().zip(&theirs) {
        if SCATTER_HOST_LOCAL_FIELDS.contains(&n.as_str()) {
            continue;
        }
        assert_eq!(
            a, b,
            "`ScatterBatch::{n}` is projected as `{a}` in the editor viewport and \
             `{b}` in the shipped player. Keep them identical, or — if the \
             difference is deliberate — add the field to `SCATTER_HOST_LOCAL_FIELDS` \
             with the reason written down."
        );
    }
    // A guard on the guard: an empty literal would satisfy everything above.
    assert!(
        mine.len() >= 7,
        "the `ScatterBatch` projection shrank to {} fields — was it gutted?",
        mine.len()
    );
}

/// The surrounding *rules* — not just the literal — must hold on both sides: the
/// anchor convention, the placeholder palette, the deterministic per-kind
/// grouping, the content draw distance… **and the absence of the per-instance CPU
/// push**, which is the whole point of P18.5.
///
/// Without the absence check a projector could satisfy every fragment above while
/// *also* still expanding each instance into `RenderScene::instances` — the scatter
/// would then draw twice on one host and once on the other, which is precisely the
/// preview-vs-shipping divergence this file exists to stop.
#[test]
fn both_projectors_scatter_pcg_and_foliage_the_same_way() {
    // Fragments that must appear in BOTH projectors, verbatim.
    const SHARED: [&str; 13] = [
        // The Ring-0 pack + content hash, so neither host invents its own layout.
        "ScatterData::build(",
        // THE ANCHOR RULE: offsets are relative to the entity's world translation…
        "anchor: translation,",
        // …and foliage instances, being entity-LOCAL already, are packed against a
        // ZERO build-anchor — no conversion, so the same stroke content-keys the
        // same wherever it is placed.
        "ScatterData::build(PrimMesh::ALL[k], DVec3::ZERO, bucket)",
        // PCG stays a placeholder cube, coloured per kind.
        "PrimMesh::Cube,",
        "pcg_kind_color(si.kind)",
        // Foliage rotation goes through the shared euler-degrees → quat rule.
        "foliage_rot_quat(fi.rotation)",
        // The palette fallback for an unknown kind, colour included.
        "(PrimMesh::Cube, [0.28, 0.52, 0.24, 1.0])",
        // Per-kind grouping: bucket in authored order, emit in `PrimMesh::ALL`
        // order — deterministic and independent of which kinds are used.
        "buckets[mesh.index()].push(",
        "for (k, bucket) in buckets.into_iter().enumerate()",
        // The content LOD knob rides on the batch (this is what made the two hosts
        // finally agree about it) — and foliage has none. P19.3 moved the batch
        // literal into the shared `push_scatter`, so the volume's authored knob is
        // now what it *hands* that body; `draw_distance` reaching the literal is
        // pinned separately, field for field, by the test above.
        "vol.draw_distance,",
        "draw_distance: 0.0,",
        // Both branches go through the shared helpers rather than open-coding.
        "push_pcg_scatter(",
        "fn push_scatter(",
    ];
    for (label, path) in [("editor viewport", VIEWPORT), ("shipped player", PLAYER)] {
        let src = read(path).replace("\r\n", "\n");
        for fragment in SHARED {
            assert!(
                src.contains(fragment),
                "the {label}'s scatter projection no longer contains `{fragment}` — \
                 either the scatter path was changed on one side only, or this gate \
                 needs updating deliberately"
            );
        }
        assert!(src.contains("push_foliage_scatter("), "{label}");
    }

    // …and the per-instance CPU push is really gone from both PCG/Foliage branches.
    for (label, path, pcg_end, foliage_end) in [
        (
            "editor viewport",
            VIEWPORT,
            "w.get::<Foliage>(entity)",
            "w.get::<Collider2D>(entity)",
        ),
        (
            "shipped player",
            PLAYER,
            "w.get::<Foliage>(entity)",
            "w.get::<MeshRef>(entity)",
        ),
    ] {
        let src = read(path);
        for (branch, region) in [
            (
                "PcgVolume",
                branch_region(&src, "w.get::<PcgVolume>(entity)", pcg_end),
            ),
            (
                "Foliage",
                branch_region(&src, "w.get::<Foliage>(entity)", foliage_end),
            ),
        ] {
            assert!(
                !region.contains("instances.push("),
                "the {label}'s {branch} branch still pushes per-instance \
                 `MeshInstance`s — P18.5 replaced that with one `ScatterBatch`, and \
                 a host doing both draws the scatter twice:\n{region}"
            );
            assert!(
                !region.contains("MeshInstance"),
                "the {label}'s {branch} branch still builds a `MeshInstance`:\n{region}"
            );
            assert!(
                region.contains("_scatter("),
                "the {label}'s {branch} branch no longer calls its scatter helper:\n{region}"
            );
        }
    }
}

/// **The P19.3 mirror gate: the terrain's BIOME POPULATION.**
///
/// A terrain bound to biomes gains a derived, never-persisted instance population
/// — each painted biome's `.inf_pcg` graph evaluated over the region its id owns.
/// It is the terrain-level sibling of `PcgVolume::evaluated`, and it reaches the
/// GPU the same way: one `ScatterBatch`.
///
/// **The structure LOD is one selection, made twice** (IB-2b).
///
/// A volume that grew buildings projects as *three* batches with complementary
/// distance bands — ungrouped content, the parts inside the LOD distance, one
/// shell box per building outside it. Every number in that split is a number the
/// two hosts must agree on: a host whose parts band is wider than its shell band
/// draws both (a solid box standing inside a building), and one whose bands leave
/// a gap draws neither (a hole in the skyline).
///
/// So the two selections are compared **character for character** between fences,
/// and the fences are counted — a `contains` needle that is a prefix of a
/// declaration can never fail, and neither can a second fence (the I1 audit's
/// law, applied to a delimiter).
#[test]
fn both_projectors_band_a_structure_lod_the_same_way() {
    for tag in ["pcg_scatter_lod", "pcg_shell_batch"] {
        let editor = fenced(&read(VIEWPORT), tag, "the editor viewport");
        let player = fenced(&read(PLAYER), tag, "the shipped player");
        assert!(
            editor.len() > 300,
            "the `{tag}` fence is {} chars — an empty fence would make this gate \
             vacuous",
            editor.len()
        );
        assert_eq!(
            editor, player,
            "`{tag}` has drifted between the editor viewport and the shipped \
             player. The two bands must stay complementary: overlap draws a shell \
             inside a building, a gap deletes it from the skyline."
        );
    }
    // Both reach the one shared LOD distance rather than each naming a number.
    // The fenced comparison above already forbids the two from *differing*; this
    // forbids them from agreeing on a literal, which would be a third place the
    // number lives and the one nobody would think to update.
    for (label, path) in [("editor viewport", VIEWPORT), ("shipped player", PLAYER)] {
        let src = read(path);
        assert!(
            src.contains("inf_render::STRUCTURE_LOD_M"),
            "the {label} no longer reads the shared structure LOD distance"
        );
        let block = fenced(&src, "pcg_scatter_lod", label);
        assert!(
            !block.contains("192"),
            "the {label}'s LOD band names a literal distance instead of the \
             constant"
        );
        assert!(
            src.contains("near_distance"),
            "the {label} no longer bands a batch from below — without the inner \
             cut the shells draw on top of the parts"
        );
    }
    assert_eq!(
        extract_fn(&read(VIEWPORT), "push_shells"),
        extract_fn(&read(PLAYER), "push_shells"),
        "`push_shells` has drifted between the editor viewport and the shipped \
         player"
    );
}

/// **THE SCATTER CARRY-FORWARD IS ONE MEMO, KEYED THE SAME WAY IN BOTH HOSTS**
/// (island wave I8a audit).
///
/// # Why this pin exists, stated so it can be argued with
///
/// The wave I8a ledger routed a defect with its price: `push_pcg_scatter` re-packs
/// a volume's whole population every frame — on the island's 172 settlement
/// blocks, **365 545 instances and 20.2 ms of projection against a 1.5 ms
/// `PROJECTION_BUDGET_MS`** — for content that changes only when a cell
/// activates. The fix is Hardening Wave E's own terrain pattern one subsystem
/// over, and the wave declined to take it *because the projector body it edits is
/// pinned character for character by this file*. So the pin moves, and this is
/// the arm that says what it now covers.
///
/// # What a divergence here would be
///
/// A memo is a decision about **what a host draws**, so two hosts memoizing on
/// different keys is the same class of failure the file already guards: one host
/// carrying a stale population forward while the other re-packs is a preview and
/// a shipped build showing two different cities, and — unlike a band drift — it
/// would show only after a cell moved, i.e. never in a screenshot.
///
/// Three claims:
///
/// 1. the memo body is **character-identical** between fences;
/// 2. both hosts **take last projection's memo out** and stamp the mesh table,
///    rather than one of them keeping it and the other clearing it;
/// 3. neither host reaches `push_pcg_scatter` from its entity walk **except**
///    through the memo — read through the comment stripper, because a sentence
///    naming the function satisfies a raw `contains` (the `.apply(` precedent).
///
/// Mutation-verified while it was written: dropping `stamp:` from either copy's
/// key fails (1); replacing `carry_or_push_pcg_scatter(` with a direct
/// `push_pcg_scatter(` call in either walk fails (3); and deleting the
/// `std::mem::take` from either host fails (2) — while deleting it from *both*
/// leaves them identical and is caught by (2) rather than by (1), which is the
/// `apply_record_mirror` audit's own lesson about equality pins.
/// **Both hosts register the same twelve building module meshes** (island wave
/// I8b).
///
/// The shape families name no `.inf_mesh` file — the geometry is a function of
/// the module's own name — so neither host's asset scan can find them and both
/// add them to the table by hand. Two hand-written loops over one Ring-0 source
/// is exactly the shape that stops agreeing, and the consequence is the one this
/// whole file exists to prevent: a preview that draws a framed window where the
/// shipped build draws a placeholder cube.
///
/// Three claims, each separately falsifiable: the fenced body is identical; both
/// hosts CALL it; and the source it reads is the Ring-0 table rather than a
/// local list (a host that inlined twelve ids would satisfy the equality and
/// would drift the moment a family was added).
#[test]
fn both_hosts_register_the_building_module_meshes() {
    let editor = fenced(
        &read(VIEWPORT),
        "building_module_table",
        "the editor viewport",
    );
    let player = fenced(
        &read("runtime/inf-player/src/scatter_mesh.rs"),
        "building_module_table",
        "the shipped player",
    );
    assert!(
        editor.len() > 200,
        "the `building_module_table` fence is {} chars — an empty fence would \
         make this gate vacuous",
        editor.len()
    );
    assert_eq!(
        editor, player,
        "the building module table has drifted between the editor viewport and \
         the shipped player."
    );
    assert!(
        editor.contains("inf_pcg::building::modules::module_meshes()"),
        "the table is no longer built from the Ring-0 source — a host-local list \
         of ids is the drift this arm exists to prevent"
    );
    for (label, path, call) in [
        (
            "editor viewport",
            VIEWPORT,
            "add_building_modules(&mut self.scatter_meshes)",
        ),
        (
            "shipped player",
            "runtime/inf-player/src/lib.rs",
            "scatter_mesh::add_building_modules(&mut table)",
        ),
    ] {
        let raw = read(path).replace("\r\n", "\n");
        let src = support::strip_comments_and_strings(&raw);
        assert!(
            src.contains(call),
            "the {label} builds its scatter-mesh table without the building \
             modules, so every wall it draws is a placeholder"
        );
    }
}

/// **Both projectors fold an instance into a batch the same way** (island wave
/// I8b audit) — the `scatter_mesh_buckets` fence.
///
/// # A fence nothing read
///
/// Both hosts have written `// MIRROR-BEGIN scatter_mesh_buckets` around the
/// body of `push_scatter` since P18.5, and **no arm in this file compared it**.
/// The delimiter reads as a pin and was not one, which is this repository's own
/// "a gate must aim at the thing it names" wearing the shape of a comment.
///
/// It matters now because island wave I8b put its whole substrate inside that
/// fence: the `extent` → `ScatterInstance::scale` fold (the drawn box IS the
/// solid box), the `(mesh, glow)` bucket key, the per-batch `glow_emissive`, and
/// the `casts_shadows` pass-through. A host that folded any one of them
/// differently would preview a city the shipped build does not draw — a window
/// lit in one and dark in the other, or a wall at one metre in one and ten in
/// the other.
///
/// Three claims, on the `apply_record_mirror` pattern: the fence is non-empty,
/// the two bodies are character-identical, and each load-bearing field is
/// **named** — because an equality pin cannot see a field deleted from *both*
/// copies.
#[test]
fn both_projectors_fold_an_instance_into_a_batch_the_same_way() {
    let editor = fenced(
        &read(VIEWPORT),
        "scatter_mesh_buckets",
        "the editor viewport",
    );
    let player = fenced(&read(PLAYER), "scatter_mesh_buckets", "the shipped player");
    assert!(
        editor.len() > 600,
        "the `scatter_mesh_buckets` fence is {} chars — an empty fence would \
         make this gate vacuous",
        editor.len()
    );
    assert_eq!(
        editor, player,
        "the scatter bucket fold has drifted between the editor viewport and the \
         shipped player. This is the body that decides how big a module draws, \
         which batch it lands in and how brightly it emits: a drift here is a \
         preview and a shipped build drawing two different cities."
    );
    for (field, why) in [
        (
            "si.extent",
            "the drawn box stops being the solid box — a 10 m slab draws as a \
             one-metre cube again",
        ),
        (
            "si.glow.to_bits()",
            "two instances that glow differently share one batch, and \
             `ScatterBatch::emissive` is one value for the whole of it",
        ),
        (
            "inf_render::glow_emissive(",
            "the hour stops reaching the batch and the city never lights up",
        ),
        (
            "casts_shadows,",
            "the parts stop opting out and the caster pack walks a city again",
        ),
    ] {
        let needle: String = field.chars().filter(|c| !c.is_whitespace()).collect();
        assert!(
            editor.contains(&needle),
            "the scatter fold no longer carries `{field}`: {why}"
        );
    }
}

#[test]
fn both_projectors_memoize_pcg_scatter_the_same_way() {
    let editor = fenced(&read(VIEWPORT), "pcg_scatter_memo", "the editor viewport");
    let player = fenced(&read(PLAYER), "pcg_scatter_memo", "the shipped player");
    assert!(
        editor.len() > 300,
        "the `pcg_scatter_memo` fence is {} chars — an empty fence would make \
         this gate vacuous",
        editor.len()
    );
    assert_eq!(
        editor, player,
        "the scatter carry-forward has drifted between the editor viewport and \
         the shipped player. A host that carries a population its twin re-packs \
         is a preview and a shipped build drawing two different cities, and it \
         shows only after a cell moves."
    );
    // The key's own fields, named here so dropping one from BOTH copies — which
    // the equality above cannot see — still fails.
    for field in [
        "entity:",
        "stamp:",
        "draw_distance_bits:",
        "table,",
        // I8b: the quantized night-glow step. A batch's emission is not part of
        // `ScatterData`, so a carried batch keeps the hour it was packed in — a
        // city lit at noon, and invisible to every other field here.
        "glow_step,",
        "anchor:",
    ] {
        assert!(
            editor.contains(
                &field
                    .chars()
                    .filter(|c| !c.is_whitespace())
                    .collect::<String>()
            ),
            "the memo key no longer carries `{field}` — every one of the five is \
             a way the payload can change without the population stamp moving"
        );
    }
    for (label, path) in [("editor viewport", VIEWPORT), ("shipped player", PLAYER)] {
        let raw = read(path).replace("\r\n", "\n");
        let src = support::strip_comments_and_strings(&raw);
        assert!(
            src.contains("std::mem::take(&mut self.scene.scatter_memo)")
                || src.contains("std::mem::take(&mut scene.scatter_memo)"),
            "the {label} no longer takes last projection's scatter memo out — a \
             memo that is never taken never sees a removal"
        );
        assert!(
            src.contains("inf_render::scatter_table_stamp("),
            "the {label} no longer folds the resolved mesh table into the memo \
             key — a mesh swapped under an existing GUID would be memoized away"
        );
        // …and the ONLY reach into the packer is the memo's own. Four
        // occurrences, exactly: `fn push_pcg_scatter(`, the miss branch's call to
        // it, `fn carry_or_push_pcg_scatter(` and the entity walk's one call to
        // THAT (the last two match as substrings, which is why the count is four
        // and not two).
        assert_eq!(
            src.matches("push_pcg_scatter(").count(),
            4,
            "the {label} names `push_pcg_scatter(` {} times, not 4 — a second, \
             unmemoized call site is the defect this arm exists to hold shut",
            src.matches("push_pcg_scatter(").count()
        );
        assert_eq!(
            src.matches("carry_or_push_pcg_scatter(").count(),
            2,
            "the {label} no longer reaches the packer through exactly one \
             memoized door"
        );
    }
}

/// This is exactly the shape of divergence this file exists to catch. A population
/// projected by the editor and not by the player is *"the biome scatter shows in
/// the preview and is missing from the shipped build"* — a whole layer of world
/// content, discovered by a player. So both hosts must carry the branch, and both
/// must reach the batch through the SAME `push_scatter` body the volume path uses:
/// two copies of "build a batch from scattered instances" is precisely how the
/// hosts came to disagree about `draw_distance` for two phases.
#[test]
fn both_projectors_project_the_terrain_biome_population_the_same_way() {
    // Fragments that must appear in BOTH projectors, verbatim.
    const SHARED: [&str; 4] = [
        // The branch calls the population helper …
        "push_biome_population(",
        // … which reads the derived component field …
        "&terrain.biome_population",
        // … and whose body is the SAME one the volume path uses, so a population
        // cannot be packed, shaded or culled differently from a volume's scatter.
        "fn push_scatter(",
        // The empty-population guard: no content ⇒ no batch, on both sides.
        "terrain.biome_population.is_empty()",
    ];
    for (label, path) in [("editor viewport", VIEWPORT), ("shipped player", PLAYER)] {
        let src = read(path).replace("\r\n", "\n");
        for fragment in SHARED {
            assert!(
                src.contains(fragment),
                "the {label}'s terrain biome-population projection no longer contains \
                 `{fragment}` — either the population path was changed on one side \
                 only (the shipped build would then draw a different world from its \
                 preview), or this gate needs updating deliberately"
            );
        }
    }

    // The helper is shared *character for character* across the hosts — it is a
    // self-contained free function on both sides, like `project_sky`, so nothing
    // weaker than equality is called for.
    assert_eq!(
        extract_fn(&read(VIEWPORT), "push_biome_population"),
        extract_fn(&read(PLAYER), "push_biome_population"),
        "`push_biome_population` has drifted between the editor viewport and the \
         shipped player"
    );
    assert_eq!(
        extract_fn(&read(VIEWPORT), "push_scatter"),
        extract_fn(&read(PLAYER), "push_scatter"),
        "the shared `push_scatter` body has drifted between the editor viewport and \
         the shipped player — it is the one place a scattered instance becomes a \
         `ScatterBatch` on either host"
    );

    // …and the population branch is instanced, never per-instance mesh draws —
    // the same absence check the PCG branch gets, for the same reason. The branch
    // is delimited by its own guard (the `w.get::<Terrain>(entity)` probe opens
    // the *clipmap* terrain branch far earlier in the loop) and ends where the
    // foliage branch begins.
    for (label, path) in [("editor viewport", VIEWPORT), ("shipped player", PLAYER)] {
        let src = read(path);
        let region = branch_region(
            &src,
            "!terrain.biome_population.is_empty()",
            "w.get::<Foliage>(entity)",
        );
        assert!(
            !region.contains("instances.push("),
            "the {label}'s biome-population branch pushes per-instance \
             `MeshInstance`s — a population is instanced, never per-instance mesh \
             draws:\n{region}"
        );
        assert!(
            !region.contains("MeshInstance"),
            "the {label}'s biome-population branch builds a `MeshInstance`:\n{region}"
        );
    }
}

/// **The "the follow-up landed" gate** (P18.5).
///
/// Both projectors used to carry the same warning — *">50k instances — instanced-
/// draw perf path is a follow-up"* — because both expanded a scatter into one
/// `MeshInstance` per instance and there was nothing better to do about it. This
/// batch IS that follow-up: the payload uploads once per content change and the
/// GPU culls it per instance. A warning left behind would be a false alarm on
/// exactly the content the engine now handles best, and a reader would take it as
/// a live limitation.
#[test]
fn neither_projector_warns_about_fifty_thousand_instances() {
    for (label, path) in [("editor viewport", VIEWPORT), ("shipped player", PLAYER)] {
        let src = read(path).replace("\r\n", "\n");
        for stale in ["50_000", "instanced-draw perf"] {
            assert!(
                !src.contains(stale),
                "the {label} still contains `{stale}` — P18.5 IS the instanced-draw \
                 follow-up that warning pointed at, so it must not survive it"
            );
        }
    }
}

// ── the skeletal (`SkeletalMesh`) projection ─────────────────────────────────
//
// P18.3 gave the editor viewport a real `SkeletalMesh` branch and left the
// shipped player with **none at all**: a level with a skeletal character
// previewed correctly in PIE and shipped as nothing. That was a live
// PIE-vs-shipping divergence, not a missing feature, and the follow-up that
// closes it is what these gates pin.
//
// Three layers, because the skeletal path is split across four files rather than
// two: the **pose rule** and the **bind-space rebuild** live in each host's
// render-asset store (Ring 1 for the editor, `inf-player` for the player) and are
// self-contained functions, so they are compared *character for character*; the
// `SkinnedInstance` literal is inline in two loops with different iteration
// orders, so it is compared field for field, exactly like the P18.3 `VgeomInstance`
// gate above.

/// **The pose rule must be one rule.** No skeleton (or a jointless one) ⇒ the
/// caller keeps its placeholder; the pose the SIM evaluated for this entity, when
/// there is one for this skeleton (P24.1); else no `AnimPlayer` / no clip / an
/// unresolvable clip ⇒ the rest pose; else the clip sampled at the play-head,
/// honouring `looping`.
///
/// Every one of those four arms is a place the two hosts could silently drift,
/// and each drift reads as a different bug in the shipped game: arm 1 as a
/// character that vanishes instead of showing a placeholder, arm 2 as one whose
/// **state machine drives the shipped build and not the preview** (or the
/// reverse), arm 3 as one that is invisible until it plays, arm 4 as one frozen
/// in its bind pose. None of them is visible to a compiler, and only arms 2 and 4
/// would even show up in a scene-level comparison of the two hosts.
///
/// **The one permitted difference**, normalized below: the editor's store resolves
/// lazily into its own maps and is owned mutably by the viewport (`&mut self`);
/// the player's memoizes behind its own locks because the projector holds it
/// immutably (`&self`). That is a difference in *who owns the cache*, not in the
/// rule.
///
/// The **doc block is compared too** since P24.1 — see
/// [`extract_method_with_doc`] for the hole that closes.
#[test]
fn the_skinned_pose_rule_is_identical_in_both_stores() {
    let raw = extract_method_with_doc(&read(EDITOR_ASSETS), "resolve_skinned");
    // The normalization must rewrite the RECEIVER and nothing else. A second
    // `&mut self` anywhere in the body would be silently erased along with it, and
    // an interior-mutability divergence is exactly the kind of difference this gate
    // exists to catch — so the token is asserted to occur exactly once before it is
    // rewritten.
    assert_eq!(
        raw.matches("&mut self").count(),
        1,
        "`resolve_skinned` contains more than one `&mut self`; the receiver \
         normalization below would erase the others too"
    );
    let mine = raw.replace("&mut self", "&self");
    let theirs = extract_method_with_doc(&read(PLAYER_ASSETS), "resolve_skinned");
    assert_eq!(
        mine, theirs,
        "the editor's and the player's `resolve_skinned` have drifted — the two \
         hosts would pose the same character differently, so PIE would stop \
         matching shipping. Keep them identical (modulo `&mut self` / `&self`), or \
         move the rule into a Ring-0 crate both can depend on."
    );
}

/// **The tier's shared pose is one rule, not two** (wave NPC1b) — the
/// `resolve_skinned` gate above, applied to the door beside it.
///
/// It matters for the same reason and one more: `resolve_skinned_shared` is what
/// a crowd's non-posing tiers draw, and the whole claim behind it is that its
/// answer *equals* the per-agent path's. Two copies of that claim are two chances
/// for one host to derive a rest pose the other does not.
#[test]
fn the_shared_tier_pose_is_identical_in_both_stores() {
    let raw = extract_method_with_doc(&read(EDITOR_ASSETS), "resolve_skinned_shared");
    assert_eq!(
        raw.matches("&mut self").count(),
        1,
        "`resolve_skinned_shared` has a second `&mut self`; the normalization below would erase it"
    );
    let mine = raw.replace("&mut self", "&self");
    let theirs = extract_method_with_doc(&read(PLAYER_ASSETS), "resolve_skinned_shared");
    assert_eq!(
        mine, theirs,
        "the two `resolve_skinned_shared` have drifted, so the hosts draw a far crowd differently"
    );
    // …and not a stub: it resolves a real rig and goes through the cache door,
    // rather than returning `None` and quietly leaving every far agent a
    // placeholder cube.
    let body = extract_method(&read(PLAYER_ASSETS), "resolve_skinned_shared");
    for fragment in [
        "let skeleton = self.skeleton(skeleton_id)?;",
        "if skeleton.is_empty() {",
        "let mesh = self.skinned_geometry(mesh_id, skeleton_id)?;",
        "palette: self.rest_palette(&skeleton, (mesh_id, skeleton_id, entry)),",
    ] {
        assert!(
            body.contains(fragment),
            "`resolve_skinned_shared` no longer contains `{fragment}`:
{body}"
        );
    }
}

/// **The tier → caster mapping is one rule, not two** (wave NPC1b).
///
/// `crowd_shadow` is the only place either host decides whether an agent casts a
/// skinned shadow or a proxy one, and the two copies are held byte-identical the
/// way `project_cloth`'s are. A host that mapped `Near` to a skinned caster would
/// spend a group per agent against a ceiling of 1 024 — and would do it in the
/// preview only, which is the divergence class this file exists for.
#[test]
fn the_crowd_shadow_door_is_identical_in_both_projectors() {
    let mine = extract_fn(&read(VIEWPORT), "crowd_shadow");
    let theirs = extract_fn(&read(PLAYER), "crowd_shadow");
    assert_eq!(
        mine, theirs,
        "the two `crowd_shadow` doors have drifted, so a crowd casts differently in the two hosts"
    );
    // …and it reads the TIER rather than hard-coding one answer: a door that
    // returned `Proxy` for everything is identical on both sides and wrong.
    // …and since island wave NPC1e it reads the tier TWICE: `skinned_caster`
    // decides its own silhouette against the shared proxy, and `casts_shadow`
    // decides whether it casts at all. A host that dropped the second arm would
    // put 712 walking proxy boxes back into the page cache in the preview only.
    for fragment in [
        "None => inf_render::SkinnedShadow::BindSphere,",
        "Some(a) if a.tier.skinned_caster() => inf_render::SkinnedShadow::Posed,",
        "Some(a) if a.tier.casts_shadow() => inf_render::SkinnedShadow::Proxy,",
        "Some(_) => inf_render::SkinnedShadow::None,",
    ] {
        assert!(
            theirs.contains(fragment),
            "`crowd_shadow` no longer contains `{fragment}`:
{theirs}"
        );
    }
}

/// A guard on the guard: two stubs are identical too. The shared body has to
/// actually implement the three arms.
#[test]
fn the_shared_pose_rule_is_not_a_stub() {
    let body = extract_method(&read(PLAYER_ASSETS), "resolve_skinned");
    for fragment in [
        // Arm 1 — both halves of the binding, and the jointless-skeleton reject.
        "let mesh_id = sm.mesh?;",
        "let skeleton_id = sm.skeleton?;",
        "if skeleton.is_empty() {",
        // Arm 2 (P24.1) — the SIM's pose, and both of its guards. A host that
        // took `posed` without checking the skeleton would wear another rig's
        // pose; one that dropped the arm entirely would draw a machine-driven
        // character at rest while the other host animated it.
        "posed.filter(|p| p.skeleton == skeleton_id && p.pose.len() == skeleton.len())",
        "(Some(p), _) => p.pose.clone(),",
        // Arm 4 — the play-head sampled through the shared Ring-0 sampler,
        // `looping` included (a host that dropped it would loop a one-shot).
        "inf_anim::sample_clip(&skeleton, &clip, p.t as f32, p.looping)",
        // Arm 3 — rest pose for BOTH "no player/clip" and "clip did not resolve".
        // Two occurrences, and the count is asserted below.
        "inf_anim::Pose::rest(&skeleton)",
        // The palette the skinned pass consumes, and the dedup key the projection
        // allocates `skinned_meshes` slots from.
        "palette: Arc::new(inf_anim::skinning_matrices(&skeleton, &pose))",
        "key: (mesh_id, skeleton_id)",
    ] {
        assert!(
            body.contains(fragment),
            "`resolve_skinned` no longer contains `{fragment}` — either it was \
             gutted, or this gate needs updating deliberately:\n{body}"
        );
    }
    assert_eq!(
        body.matches("inf_anim::Pose::rest(&skeleton)").count(),
        2,
        "the rest-pose fallback must cover BOTH `no AnimPlayer/clip` and `the clip \
         did not resolve` — one of the two arms stopped falling back:\n{body}"
    );
}

/// **The bind-space rebuild must be one rebuild.** Both hosts turn the same
/// `.inf_mesh` into the same `SkinnedMeshData`, or they upload *different vertex
/// buffers* for the same asset — a divergence no scene-level comparison can see,
/// because both scenes would carry one mesh with the right instance count.
///
/// The load-bearing details are the submesh concatenation with rebased indices,
/// and pinning an unskinned submesh to joint 0 with weight 1 (dropping it instead
/// would silently lose geometry on one host only).
#[test]
fn the_bind_space_rebuild_is_identical_in_both_stores() {
    let mine = extract_fn(&read(EDITOR_ASSETS), "skinned_mesh_data");
    let theirs = extract_fn(&read(PLAYER_ASSETS), "skinned_mesh_data");
    assert_eq!(
        mine, theirs,
        "the editor's and the player's `skinned_mesh_data` have drifted — the two \
         hosts would build different bind-space geometry from the same `.inf_mesh`"
    );
    // Not a stub: the concatenation, the rebase, and the joint-0 pin. Read off the
    // ITEM, never the doc block — a fragment a comment could satisfy proves nothing.
    let body = extract_item(&read(PLAYER_ASSETS), "skinned_mesh_data");
    for fragment in [
        "let base = vertices.len() as u32;",
        "sm.skin.get(i).copied().unwrap_or_default().normalized()",
        "indices.extend(sm.indices.iter().map(|&i| i + base));",
    ] {
        assert!(
            body.contains(fragment),
            "`skinned_mesh_data` lost `{fragment}`"
        );
    }
}

/// Whitespace and rustfmt's trailing commas removed, so a source pin states a
/// CLAIM rather than a line width (wave CHAR1a.2).
fn squash(s: &str) -> String {
    s.chars()
        .filter(|c| !c.is_whitespace())
        .collect::<String>()
        .replace(",)", ")")
}

/// Fields of the `SkinnedInstance` literal whose value expression is **host-local
/// by design** and therefore excluded from the value comparison (their presence
/// and order still are not).
///
///  * `id` — the pick id, allocated from each host's own counter over its own
///    iteration order (document order vs `Guid` order). It has never matched
///    across hosts and is not meant to; the editor additionally maps it back to a
///    GUID, which the player has no use for.
///  * `mesh` — the index into `RenderScene::skinned_meshes`, allocated from each
///    host's own per-projection dedup map in that same iteration order. Two hosts
///    that agree perfectly about *which* `(mesh, skeleton)` pairs are drawn can
///    still number their slots differently, and the number means nothing outside
///    the scene it indexes.
///
/// Note what is **not** here, and deliberately so: unlike the `VgeomInstance`
/// gate, `asset`-style content-hash-vs-GUID keying has no analogue on this path.
/// The skinned pass caches its GPU upload by the **pointer identity** of the
/// `Arc<SkinnedMeshData>`, so changed content is a different key by construction
/// and neither host needs to content-address an id.
/// Fields of `SkinnedInstance` the two hosts are *allowed* to project differently,
/// each with the reason:
///
/// * `id` / `mesh` — each host numbers from its own counter over its own iteration
///   order (the player walks the world in `Guid` order, the viewport walks the
///   document's entity order), so the *values* differ while the rule does not. The
///   same asymmetry `VGEOM_HOST_LOCAL_FIELDS` documents.
/// * `translation` — the editor reads it straight off the entity's affine, the
///   player reads `sim.interp_translation(..)` so a character is drawn at its
///   interpolated position between fixed steps rather than snapped to the last one.
///   Both hosts' own comments already say this; it belongs here so the gate stops
///   asking about it and starts *documenting* it.
const SKINNED_HOST_LOCAL_FIELDS: [&str; 3] = ["id", "mesh", "translation"];

/// **The skeletal mirror gate.** Both projectors build a `SkinnedInstance` from
/// the same ECS state; every field must be present on both sides, in the same
/// order, carrying the same expression — except the two documented host-local
/// ones.
#[test]
fn the_skinned_instance_projection_matches_field_for_field() {
    let mine = struct_literal_fields(&read(VIEWPORT), "SkinnedInstance");
    let theirs = struct_literal_fields(&read(PLAYER), "SkinnedInstance");

    let names = |v: &[(String, String)]| v.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>();
    assert_eq!(
        names(&mine),
        names(&theirs),
        "the two `SkinnedInstance` projections carry different fields (or in a \
         different order) — a field projected on one side and not the other means \
         the shipped game draws a character differently from the preview"
    );

    for ((n, a), (_, b)) in mine.iter().zip(&theirs) {
        if SKINNED_HOST_LOCAL_FIELDS.contains(&n.as_str()) {
            continue;
        }
        assert_eq!(
            a, b,
            "`SkinnedInstance::{n}` is projected as `{a}` in the editor viewport and \
             `{b}` in the shipped player. Keep them identical, or — if the \
             difference is deliberate — add the field to \
             `SKINNED_HOST_LOCAL_FIELDS` with the reason written down."
        );
    }
    // A guard on the guard: an empty literal would satisfy everything above.
    // **Twelve since wave CHAR1a.2**, when `blend` and `cutoff` joined the
    // literal so a skinned surface could carry a hole. Raised deliberately with
    // the field count rather than left at a floor two waves below it: a guard
    // that trails the struct is a guard that would not notice the struct being
    // cut back to what it used to be.
    assert!(
        mine.len() >= 12,
        "the `SkinnedInstance` projection shrank to {} fields — was it gutted?",
        mine.len()
    );
}

/// The surrounding *rules* — not just the literal — must hold on both sides: the
/// `MeshRef`-absent gate the branch hangs off, the `AnimPlayer` lookup that feeds
/// the pose, the one-slot-per-`(mesh, skeleton)` dedup, the `Arc` handed straight
/// into the scene, and the placeholder that survives an unresolvable binding.
///
/// The `Arc` fragment is the load-bearing one. `RenderScene::skinned_meshes` is a
/// `Vec<Arc<SkinnedMeshData>>` *because* the skinned pass caches its GPU upload by
/// pointer identity; a host that pushed `draw.mesh.as_ref().clone()` — or rebuilt
/// the stream per projection — would re-upload a character's whole bind-space
/// buffer every frame while every pixel stayed identical, so nothing else in the
/// repo would notice.
#[test]
fn both_projectors_draw_skeletal_meshes_the_same_way() {
    // Fragments that must appear in BOTH projectors, verbatim.
    const SHARED: [&str; 18] = [
        // The branch is the `MeshRef`-absent arm: an entity is a rigid draw or a
        // skinned one, never both.
        "w.get::<MeshRef>(entity).is_none()",
        "w.get::<SkeletalMesh>(entity).copied()",
        // **Character space** (P29.6). A rig's origin is its feet and a
        // character's entity transform is its capsule centre, so both hosts draw
        // the pose through the ONE door that knows the difference. A host that
        // went back to the raw `GlobalTransform` would draw a character half a
        // capsule above the floor its own feet are locked to — in one host only,
        // which is exactly the PIE-versus-shipping divergence this file exists
        // for.
        "inf_ecs::pose::model_to_world(world, entity)",
        // The pose inputs, and the shared store call that applies the pose rule.
        // `evaluated_pose` is the load-bearing one (P24.1): a host that stopped
        // reading the sim's pose would draw every machine-driven character at
        // rest while the other host animated it — the divergence a player finds
        // and no pixel comparison in this repo would.
        "w.get::<inf_ecs::components::AnimPlayer>(entity).copied()",
        "inf_ecs::pose::evaluated_pose(world, guid)",
        "resolve_skinned(&sm, player.as_ref(), posed, machine.as_ref())",
        // **The preview idle reaches BOTH hosts** (wave CHAR1a.2). A host
        // that stopped reading the machine would draw every unplayed
        // character in its bind pose while the other drew an idle — the
        // exact editor-versus-shipping split this file exists to stop, and
        // the one CHAR1a photographed from the editor side.
        "w.get::<inf_ecs::components::AnimStateMachine>(entity).copied()",
        // **The tier reaches the renderer** (wave NPC1b). Four fragments, because
        // four separate things follow from `CrowdAgent` and each of them is a
        // divergence if one host drops it: an agent off the pose path resolves the
        // SHARED rest palette (drop it on one side and that host derives and
        // uploads a thousand copies of one answer), its look is derived from its
        // `Guid` through the one Ring-0 door (drop it and PIE's crowd is grey
        // while the shipped one is not), its build multiplies the drawn scale, and
        // its tier decides whether it casts a skinned shadow or a proxy.
        //
        // **The look door took a WORLD at wave EMS3**, and moving both hosts in
        // the same commit is the whole reason this line is pinned. `agent_look`
        // is the *derived* draw; `agent_look_in` is that draw with the
        // appearance channel read, which is what a wardrobe changes. A host left
        // on the old door would keep drawing a criminal in the coat they were
        // wearing when they committed the crime while the other host drew the
        // one they changed into — a PIE-vs-shipping divergence in the exact
        // pixels the wanted system is played through.
        "w.get::<inf_ecs::crowd::CrowdAgent>(entity).copied()",
        "agent.map(|a| inf_ecs::crowd::agent_look_in(world, a.guid))",
        "Some(a) if !a.tier.poses() =>",
        "resolve_skinned_shared(&sm, machine.as_ref())",
        "let shadow = crowd_shadow(agent);",
        // ONE `skinned_meshes` slot per (mesh, skeleton) pair…
        "skinned_slots.entry(draw.key).or_insert_with(",
        // …and the entry is the store's own `Arc`, pushed with no copy.
        "skinned_meshes.push(draw.mesh)",
        "skinned_meshes.len() - 1",
        // Both lists are rebuilt from scratch every projection.
        "skinned_meshes.clear()",
        ".skinned.clear()",
        // The placeholder survives, down to its slate tint, so the two hosts also
        // agree about content whose assets are missing.
        "color: [0.55, 0.60, 0.72, 1.0],",
    ];
    for (label, path) in [("editor viewport", VIEWPORT), ("shipped player", PLAYER)] {
        let src = read(path).replace("\r\n", "\n");
        // **Normalized, because rustfmt is allowed to have opinions** (wave
        // CHAR1a.2). `resolve_skinned` grew a fourth argument, which pushed the
        // viewport's call past the line width; rustfmt broke it across five
        // lines and added a trailing comma. A pin written as one line then went
        // red on a formatting change rather than on a divergence — the failure
        // mode that teaches a reader to edit the gate instead of the code. The
        // claim was never about line breaks, so both sides are squashed the same
        // way: no whitespace at all, and a trailing comma before a `)` dropped.
        let flat = squash(&src);
        for fragment in SHARED {
            let fragment = &squash(fragment);
            assert!(
                flat.contains(fragment),
                "the {label}'s `SkeletalMesh` projection no longer contains \
                 `{fragment}` — either the skeletal path was changed on one side \
                 only, or this gate needs updating deliberately"
            );
        }
        // The `Arc` must reach the scene UNCLONED-as-data. `.clone()` on the inner
        // value (or a `to_vec()` of the vertex stream) would defeat the pass's
        // pointer-identity upload cache silently.
        for banned in [
            "skinned_meshes.push(draw.mesh.as_ref().clone())",
            "draw.mesh.vertices.clone()",
        ] {
            assert!(
                !src.contains(banned),
                "the {label} copies the bind-space stream into the scene \
                 (`{banned}`) — `RenderScene::skinned_meshes` is `Vec<Arc<_>>` \
                 precisely so the skinned pass can cache its GPU upload by pointer \
                 identity, and a copy re-uploads a character every frame"
            );
        }
    }
}

// ── P20.1: the water (`WaterBody`) projection ────────────────────────────────

/// **The P20.1 mirror gate.** `project_water` is a self-contained free function
/// on both sides — like `project_sky` — so nothing weaker than character-for-
/// character equality is called for.
///
/// The divergence this catches is the one a player finds: a sea that is a
/// different colour, a different height or a different *phase* in the shipped
/// build than in the preview. Water is worse than most content for this, because
/// its whole appearance is derived rather than authored: one host reading the
/// weather wind and the other reading the body's own would produce two plausible
/// seas that never agree.
#[test]
fn project_water_is_identical_in_both_projectors() {
    let mine = extract_fn(&read(VIEWPORT), "project_water");
    let theirs = extract_fn(&read(PLAYER), "project_water");
    assert_eq!(
        mine, theirs,
        "the two `project_water` projectors have drifted — PIE would stop \
         matching shipping. Keep them byte-identical, or move the shared part \
         into `inf_ecs::sky` / `inf-water` (which is where the clock, the wind \
         rule and the wave derivation already live)."
    );
}

/// A guard on the guard: two stubs are identical too. The shared body has to do
/// the work — derive the wave field, honour the wind rule, and build a river's
/// ribbon from the spline on the same entity.
#[test]
fn the_shared_water_projector_is_not_a_stub() {
    let body = extract_item(&read(PLAYER), "project_water");
    for fragment in [
        // The wind rule is the component's, not the host's — the one thing here
        // that could silently diverge.
        "water.effective_wind(weather_wind)",
        // The wave field is DERIVED in Ring 0, never re-derived per host.
        "inf_render::WaveField::from_spec(&spec)",
        "spread_rad: water.wave_spread_deg.to_radians()",
        "seed: water.wave_seed",
        // A river's ripple travels downstream, so its wave frame is the river's
        // own — a host that fed it the world wind would show a river whose
        // wavelets cross it sideways.
        "let river = water.kind == WaterKind::River;",
        "wind_x: if river { 1.0 } else { wind_x },",
        // The centreline is the spline on the SAME entity, in world space.
        "affine.transform_point3(p.to_dvec3())",
        "inf_render::RiverPath::from_points(&points, sp.closed, interp, &profile)",
        "inf_render::WaterFrame::from(f)",
        // P20.4: the P19.1 flow map reaches a river's frames, and does so through
        // the Ring-0 rule rather than a per-host world walk. A host that gathered
        // terrains itself would be free to disagree about which one answers.
        "let mapped = flow.is_mapped();",
        "flow.foam_gain_at(glam::DVec2::new(f.center.x, f.center.z))",
        // The level clock reaches the body — a host that dropped it would render
        // a frozen sea that still passed every other assertion here.
        "time_s,",
        // The kind mapping, all three arms.
        "WaterKind::Ocean => inf_render::WaterKindGpu::Ocean,",
        "WaterKind::Lake => inf_render::WaterKindGpu::Lake,",
        "WaterKind::River => inf_render::WaterKindGpu::River,",
    ] {
        assert!(
            body.contains(fragment),
            "`project_water` no longer contains `{fragment}` — either it was \
             gutted, or this gate needs updating deliberately:\n{body}"
        );
    }
}

/// The surrounding *rules* — not just the shared body — must hold on both sides:
/// the list is rebuilt from scratch, the clock and wind are resolved ONCE per
/// projection through the Ring-0 seam, the spline comes from the same entity, and
/// a body with no geometry is skipped rather than drawn degenerate.
#[test]
fn both_projectors_project_water_the_same_way() {
    const SHARED: [&str; 8] = [
        // Rebuilt every projection, like `scatter` — a body's state is a pure
        // function of its component, its spline and the clock.
        "waters.clear()",
        // The clock + wind come from Ring 0, once, never per body and never from
        // a wall clock. A host that inlined `resolve_sky(...).weather()` here
        // would be the divergence this file exists to stop.
        "inf_ecs::sky::water_environment(world)",
        // The branch itself, and the same-entity spline.
        "w.get::<WaterBody>(entity)",
        // The flow field, likewise resolved ONCE per projection through Ring 0
        // (P20.4) — same argument as the clock and the wind above.
        "inf_ecs::hydro::terrain_flow(world)",
        "w.get::<Spline>(entity),",
        "&water_flow,",
        // Nothing degenerate reaches the renderer.
        "if body.drawable()",
        "waters.push(body)",
    ];
    for (label, path) in [("editor viewport", VIEWPORT), ("shipped player", PLAYER)] {
        let src = read(path).replace("\r\n", "\n");
        for fragment in SHARED {
            assert!(
                src.contains(fragment),
                "the {label}'s water projection no longer contains `{fragment}` — \
                 either the water path was changed on one side only, or this gate \
                 needs updating deliberately"
            );
        }
        // A river's centreline must NOT be looked up through a reference: the
        // whole point of same-entity composition is that there is nothing to
        // resolve, and a host that started resolving one would need a cook edge
        // the other side does not have.
        let region = branch_region(&src, "w.get::<WaterBody>(entity)", "project_water(");
        assert!(
            !region.contains("entity_of("),
            "the {label} resolves a river's spline through a reference:\n{region}"
        );
    }
}

// ── P21.1: the volumetric-terrain (`VoxelVolume`) projection ─────────────────

/// **The P21.1 mirror gate.** `project_voxel` is a self-contained free function on
/// both sides — like `project_sky` and `project_water` — so nothing weaker than
/// character-for-character equality is called for.
///
/// The divergence this catches is one a player finds and a compiler cannot: a cave
/// whose walls sit a fraction of a voxel from where the preview drew them, or
/// whose rock shades a different colour than the hillside it opens out of. Voxel
/// surface is worse than most content for this, because none of it is authored
/// directly — every vertex is *derived* from a field, so two plausible derivations
/// can differ everywhere while agreeing about the level.
#[test]
fn project_voxel_is_identical_in_both_projectors() {
    let mine = extract_fn(&read(VIEWPORT), "project_voxel");
    let theirs = extract_fn(&read(PLAYER), "project_voxel");
    assert_eq!(
        mine, theirs,
        "the two `project_voxel` projectors have drifted — PIE would stop matching \
         shipping. Keep them byte-identical, or move the shared part into \
         `inf-voxel` (which is where the mesher, the chunk store and the \
         chunk-local rebase already live)."
    );
}

/// A guard on the guard: two stubs are identical too. The shared body has to do
/// the work — rebase through the Ring-0 helper, take the scale from the asset,
/// take the palette from the terrain on the same entity, and refuse to emit an
/// empty volume.
#[test]
fn the_shared_voxel_projector_is_not_a_stub() {
    let body = extract_item(&read(PLAYER), "project_voxel");
    for fragment in [
        // The scale comes from the ASSET, not from the component. A host that read
        // `volume.voxel_size_m` here would scale vertices against origins derived
        // from the asset's and tear the volume apart — and it would look fine in
        // every level where the two happen to agree, which is most of them.
        "let voxel_size_m = slot.data.voxel_size_m();",
        // The chunk-local rebase is the ONE shared Ring-0 helper. A host that
        // subtracted its own base, or scaled in a different order, would draw a
        // cave a fraction of a voxel from where the other host drew it.
        ".local_positions_m(voxel_size_m)",
        "bounds: mesh.local_bounds_m(voxel_size_m),",
        // The f64 world anchor + the entity transform (the floating-origin split).
        "origin: slot.data.chunk_origin_world(key) + translation,",
        // Only the NON-EMPTY meshes reach the renderer, in the cache's ascending
        // key order (a deterministic draw order per side).
        ".meshes()",
        // The GPU cache's invalidation key, which is the mesh's neighbourhood
        // stamp — NOT the chunk's own, or a carve in the neighbour would leave a
        // stale seam on screen forever.
        "version: slot.meshes.version(key),",
        // A material index is categorical and is the TERRAIN splat index.
        "material: mesh.materials[i] as u32,",
        "albedo: t.layers[k].albedo.to_array(),",
        "None => RenderTerrainLayer::default(),",
        // Nothing empty reaches the renderer — that is what keeps the voxel pass
        // off the command encoder on every scene that has no caves.
        "if chunks.is_empty() {",
        "return None;",
    ] {
        assert!(
            body.contains(fragment),
            "`project_voxel` no longer contains `{fragment}` — either it was \
             gutted, or this gate needs updating deliberately:\n{body}"
        );
    }
}

/// The surrounding *rules* — not just the shared body — must hold on both sides:
/// the list is rebuilt from scratch, the palette comes from the `Terrain` on the
/// same entity, the volume's identity is the shared entity fold, and **neither
/// host binds, loads or meshes inside its projection** — both do it in a
/// `sync_voxels` pre-pass, whose live set is built from BOUND-NESS rather than
/// from whatever happened to produce triangles.
///
/// That last rule is not style. A volume whose chunks mesh to no surface still
/// projects `None`, so a host that built its live set from the projection would
/// release it and re-read + re-mesh it from disk on the very next document bump —
/// and a gizmo drag bumps the document per input event.
#[test]
fn both_projectors_project_voxel_volumes_the_same_way() {
    const SHARED: [&str; 12] = [
        // **Carried, not rebuilt** (Hardening Wave E). Both hosts take the
        // previous frame's list out of the scene and hand it to `project_voxel`
        // as the carry source, so a volume whose chunk stamps did not move costs
        // a `Vec` move instead of two rebases of every vertex stream. What is
        // left over at the end of the walk is exactly the volumes that left the
        // scene — which is how a REMOVAL is seen, and why the seam gate below
        // reads the leftovers rather than a count.
        "let mut prev_voxels = std::mem::take(&mut",
        "prev_voxels.is_empty()",
        // …through the ONE shared predicate. A host that wrote its own
        // "has anything changed" test would be free to disagree with the other
        // about when a cave is stale, which is the same class of drift the
        // byte-identical `project_voxel` body exists to stop.
        "inf_render::take_unchanged_voxel(",
        // The branch itself.
        "w.get::<VoxelVolume>(entity)",
        // The palette is the Terrain on THIS SAME entity — composition, not a
        // reference, so there is no cook edge and nothing to dangle.
        "w.get::<Terrain>(entity),",
        // Identity is the shared entity fold, so a PIE-vs-shipping diff matches
        // volumes up by identity rather than by position in a list.
        "inf_render::terrain_id_from_guid(guid.as_u128()),",
        // Both go through the SAME Ring-0 slot type, which is what owns parsing,
        // residency and meshing.
        "inf_voxel::VolumeSlot",
        "project_voxel(",
        // The bind PRE-PASS, on both sides…
        "fn sync_voxels(",
        // …whose live set is bound-ness, never draw-ness: the release runs over
        // what `ensure` bound, not over what the projection pushed.
        "retain_only(",
        // P21.2 — the pre-pass is THREE acts and a host that skipped either of the
        // last two fails differently but silently. Without `place` residency is
        // measured from the asset's authoring anchor instead of from where the
        // entity actually is, so a cave placed away from the origin pages the
        // chunks nobody is standing in — a hole in the world with no rendering
        // explanation. Without `sync_camera` **nothing pages at all**: `ensure`
        // binds and pages zero chunks by design, so the volume meshes to nothing
        // and the host draws an empty cave while every other assertion here passes.
        "place(",
        "sync_camera(",
    ];
    for (label, path) in [("editor viewport", VIEWPORT), ("shipped player", PLAYER)] {
        let src = read(path).replace("\r\n", "\n");
        for fragment in SHARED {
            assert!(
                src.contains(fragment),
                "the {label}'s voxel projection no longer contains `{fragment}` — \
                 either the voxel path was changed on one side only, or this gate \
                 needs updating deliberately"
            );
        }
        // Neither host may re-mesh inside the projection: meshing is a `&mut` act
        // that belongs to the store's sync step, and a host that called the mesher
        // per frame would burn a chunk's whole cell walk on geometry that did not
        // move — and could disagree with the other host about *when* it meshed,
        // which is exactly the class of drift this file exists to stop.
        let region = branch_region(&src, "w.get::<VoxelVolume>(entity)", "project_voxel(");
        for banned in [
            "mesh_chunk(",
            "VoxelMeshCache::new()",
            ".meshes.sync(",
            // …and neither may BIND inside the projection either, which is what
            // cold-meshes: `ensure` parses a payload and runs a full mesh sync.
            ".ensure(",
        ] {
            assert!(
                !region.contains(banned),
                "the {label}'s voxel branch calls `{banned}` — meshing belongs to \
                 the store's sync step, never to the projection:\n{region}"
            );
        }
    }
}

/// **The store is the shared half, and it must stay shared.** Both hosts resolve
/// bytes their own way (a loose file under the content root; a pack entry) and
/// then hand them to the *same* Ring-0 `inf_voxel::VoxelVolumes`. A host that grew
/// its own parser, its own residency or its own mesh cache would be free to
/// disagree about every vertex in the world — and, unlike a projection, nothing
/// about that would be visible in a source diff of the two projectors.
#[test]
fn both_hosts_load_voxel_volumes_through_the_ring_zero_store() {
    for (label, path, own_store) in [
        (
            "editor viewport",
            VIEWPORT,
            "inf_editor_core::voxel_store::EditorVoxelVolumes",
        ),
        ("shipped player", PLAYER, "inf_voxel::VoxelVolumes"),
    ] {
        let src = read(path).replace("\r\n", "\n");
        assert!(
            src.contains(own_store),
            "the {label} no longer holds a voxel store ({own_store})"
        );
        // …and it releases what the projection stopped drawing. A loaded volume is
        // its whole decoded chunk set plus its meshed surface — real megabytes held
        // on behalf of an entity that may have been deleted.
        assert!(
            src.contains("retain_only"),
            "the {label} never releases dead voxel volumes"
        );
    }
    // The editor's Ring-1 store is a *resolver*, not a second implementation: it
    // must delegate to the Ring-0 one rather than parse or mesh a payload itself.
    let store = read("editor/crates/inf-editor-core/src/voxel_store.rs").replace("\r\n", "\n");
    assert!(
        store.contains("VoxelVolumes"),
        "the editor store is not backed by the Ring-0 one"
    );
    for banned in ["VoxelAssetReader", "mesh_chunk", "VoxelMeshCache"] {
        assert!(
            !store.contains(banned),
            "the editor's voxel store names `{banned}` — parsing and meshing belong \
             to `inf_voxel::VoxelVolumes`, which is what makes the two hosts agree"
        );
    }
}

// ── P22.1 surface deformation ────────────────────────────────────────────────

/// **The P22.1 mirror gate.** `project_deform` is a self-contained free function
/// on both sides — like `project_sky` and `project_water` — so nothing weaker
/// than character-for-character equality is called for.
///
/// The divergence this catches is subtle and would be very hard to see: a
/// projection that shipped the field in different units, or with a different cell
/// geometry, would put the *same* footprints in *different* places in the two
/// builds. Nothing would crash and every sim gate would still pass, because the
/// field is identical — it is only the window that would be wrong.
#[test]
fn project_deform_is_identical_in_both_projectors() {
    let mine = extract_fn(&read(VIEWPORT), "project_deform");
    let theirs = extract_fn(&read(PLAYER), "project_deform");
    assert_eq!(
        mine, theirs,
        "the two `project_deform` projectors have drifted — the same footprints \
         would draw in different places in the preview and the shipped build. \
         Keep them byte-identical, or move the shared part into \
         `inf_terrain::deform` (which is where the lattice already lives)."
    );
}

/// A guard on the guard: two stubs are identical too. The shared body has to
/// carry the lattice geometry from Ring 0 rather than restate it, and it has to
/// be epoch-gated rather than rebuilt.
#[test]
fn the_shared_deform_projector_is_not_a_stub() {
    let body = extract_item(&read(PLAYER), "project_deform");
    for fragment in [
        // The geometry comes from Ring 0. A host that wrote `16` or `0.25` here
        // would have a projection that silently disagreed with the field the day
        // either constant moved.
        "inf_terrain::deform::DEFORM_CELL_SAMPLES",
        "inf_terrain::deform::DEFORM_SAMPLE_PITCH_M",
        // Epoch-gated, not rebuilt — the one thing that makes a standing
        // character free, and the same key the renderer's upload gate uses.
        "d.epoch == field.epoch()",
        // An empty field projects `None`, which is what makes every scene with no
        // deformation record the command stream it always did.
        "scene.deform = None;",
        // The whole live set crosses, in the field's own coordinates.
        "field\n            .cells()",
        "cell.depths().to_vec()",
    ] {
        assert!(
            body.contains(fragment),
            "`project_deform` no longer contains `{fragment}` — either it was \
             gutted, or this gate needs updating deliberately:\n{body}"
        );
    }
    // THE CAMERA CHECK. The field is sim-authoritative; which part of it is drawn
    // is decided in the renderer. A projector that started windowing here would
    // be shipping a camera-dependent projection, which is the P21 law's other
    // half ("may be read but never shipped").
    for probe in ["eye", "camera", "window_origin"] {
        assert!(
            !body.contains(probe),
            "`project_deform` names `{probe}` — the deformation projection must \
             never see a camera:\n{body}"
        );
    }
}

/// The surrounding *rules* must hold on both sides: the field is read through the
/// Ring-0 seam (not re-derived), and the projection is driven from the world the
/// host's fixed step actually writes.
#[test]
fn both_projectors_project_deform_the_same_way() {
    const SHARED: [&str; 2] = ["inf_ecs::deform::deform_field(", "project_deform("];
    for (label, path) in [("editor viewport", VIEWPORT), ("shipped player", PLAYER)] {
        let src = read(path).replace("\r\n", "\n");
        for fragment in SHARED {
            assert!(
                src.contains(fragment),
                "the {label}'s deformation projection no longer contains \
                 `{fragment}` — either the path was changed on one side only, or \
                 this gate needs updating deliberately"
            );
        }
        // `deform` must NOT be cleared like the other lists: it is epoch-gated,
        // and a `clear()` beside the others would silently turn the gate into a
        // per-frame rebuild of the whole live cell set.
        assert!(
            !src.contains("deform.clear()"),
            "the {label} clears the deformation projection — it is epoch-gated on \
             purpose (see `project_deform`)"
        );
    }
}

/// Both fixed steps must run the deformation slot, and must run it through the
/// **one** Ring-0 rule rather than a loop spelled twice.
///
/// This is the sim half of the P22.1 mirror, and it is the half that decides
/// whether PIE matches shipping: a preview that stamped footprints the shipped
/// build did not would diverge on the very first step a character touched ground.
#[test]
fn both_fixed_steps_run_the_deformation_slot() {
    const RUNTIME_SIM: &str = "runtime/inf-player/src/runtime_sim.rs";
    const SIMULATE: &str = "editor/crates/inf-editor-core/src/simulate.rs";
    for (label, path, call) in [
        (
            "shipped player",
            RUNTIME_SIM,
            "inf_ecs::deform::step_deformation(&mut self.world, dt);",
        ),
        (
            "editor Simulate",
            SIMULATE,
            "inf_ecs::deform::step_deformation(doc.world_mut(), dt);",
        ),
    ] {
        let src = read(path).replace("\r\n", "\n");
        assert!(
            src.contains(call),
            "the {label} fixed step does not call the P22.1 deformation slot \
             (`{call}`) — the two hosts would disagree about where footprints go"
        );
        // …and it must sit AFTER the physics write-back + propagate, because a
        // footprint's XZ is read off the transform the solver just wrote.
        let at = src.find(call).expect("call present");
        let propagate = src[..at]
            .rfind("propagate();")
            .expect("a propagate must precede the deformation slot");
        let write_back = src[..propagate]
            .rfind("write_back_into(")
            .expect("the physics write-back must precede that propagate");
        assert!(
            write_back < propagate && propagate < at,
            "the {label} runs the deformation slot before the physics write-back \
             settled — footprints would land one step behind the bodies"
        );
    }
}

// ── fracture debris (P22.3) ─────────────────────────────────────────────────

/// The two `project_fracture` projectors must be byte-identical, doc block
/// included.
#[test]
fn project_fracture_is_identical_in_both_projectors() {
    let mine = extract_fn(&read(VIEWPORT), "project_fracture");
    let theirs = extract_fn(&read(PLAYER), "project_fracture");
    assert_eq!(
        mine, theirs,
        "the two `project_fracture` projectors have drifted — a broken wall would \
         render as different rubble in the preview and the shipped build. Keep \
         them byte-identical, or move the shared part into `inf-physics` (which is \
         where the fracture state, the solve and the chunk poses already live)."
    );
}

/// A guard on the guard: two stubs are identical too.
#[test]
fn the_shared_fracture_projector_is_not_a_stub() {
    let body = extract_item(&read(PLAYER), "project_fracture");
    for fragment in [
        // THE ATOMICITY PREDICATE. The same `is_intact` the physics swap reads —
        // if this ever became a second condition, an actor could render as both
        // its mesh and its chunks, or as neither.
        "if state.is_intact() {",
        "return None;",
        // A reclaimed chunk leaves the render set with the physics world, on one
        // generation. Dropping this line makes the debris budget despawn a body
        // and keep drawing it.
        "if chunk.gone {",
        // The geometry is the COOK's, mapped by the state's own placement — not
        // re-derived and not read from the ECS transform (which a detached chunk
        // has stopped following).
        "let placement = state.placement();",
        "placement.transform_point3(DVec3::from_array(src.center_of_mass))",
        // Chunk-local against the centre of mass: the pose rides on the instance
        // so the vertex buffer can be uploaded once per break.
        "chunk.translation,",
        "chunk.rotation.as_quat(),",
        // The stamp is the actor's generation, which does NOT move for a pose.
        "let version = state.generation();",
    ] {
        assert!(
            body.contains(fragment),
            "`project_fracture` no longer contains `{fragment}` — either it was \
             gutted, or this gate needs updating deliberately:\n{body}"
        );
    }
    // THE CAMERA CHECK. What has broken is sim state; which of it is drawn is the
    // renderer's business.
    for probe in ["eye", "camera", "frustum"] {
        assert!(
            !body.contains(probe),
            "`project_fracture` names `{probe}` — the fracture projection must \
             never see a camera:\n{body}"
        );
    }
}

/// The surrounding *rules*: both hosts clear the list each frame, both gate the
/// mesh push on the same `fractured` flag, and neither re-runs the solve.
#[test]
fn both_projectors_project_fracture_the_same_way() {
    const SHARED: [&str; 4] = [
        "scene.fracture_chunks.clear();",
        "project_fracture(",
        "let fractured = ",
        "fracture_chunks.extend(chunks)",
    ];
    for (label, path) in [("editor viewport", VIEWPORT), ("shipped player", PLAYER)] {
        let src = read(path).replace("\r\n", "\n");
        for fragment in SHARED {
            assert!(
                src.contains(fragment),
                "the {label}'s fracture projection no longer contains \
                 `{fragment}` — either the path was changed on one side only, or \
                 this gate needs updating deliberately"
            );
        }
        // A projector must never STEP the fracture: the solve, the budget and the
        // detach decisions belong to the fixed step, and a renderer that ran them
        // would make what breaks a function of how often you drew.
        for banned in ["runtime_destruct(", "step_fractures(", "radial_impulse("] {
            assert!(
                !src.contains(banned),
                "the {label}'s projector names `{banned}` — a render projection \
                 must never advance a simulation"
            );
        }
    }
}

// ── sub-chunk debris (P22.4) ────────────────────────────────────────────────

/// The two `project_debris` projectors must be byte-identical, doc block
/// included — the `project_fracture` rule, one batch later.
///
/// The divergence this catches is the P18.5 shape exactly: a *whole layer* of
/// visual content present in one host and absent in the other, discovered by a
/// player looking at a collapse that reads differently in the shipped build than
/// it did in the preview.
#[test]
fn project_debris_is_identical_in_both_projectors() {
    let mine = extract_fn(&read(VIEWPORT), "project_debris");
    let theirs = extract_fn(&read(PLAYER), "project_debris");
    assert_eq!(
        mine, theirs,
        "the two `project_debris` projectors have drifted — a collapse would shed \
         different rubble in the preview and the shipped build. Keep them \
         byte-identical, or move the shared part into `inf_render::debris` (which \
         is where the placement rule and the mixer already live)."
    );
}

/// **THE DEBRIS SITE IS AN ALLOWLIST, NOT A BAN.**
///
/// The first cut of this gate banned the substrings `c.translation` and
/// `c.rotation` from `project_debris` — which is a list of the two ways the
/// auditor happened to think of. It compiled
/// `entity: id ^ translation.x.to_bits() ^ age_s.to_bits()` into **both** hosts
/// and all three debris gates stayed green: `age_s` is a per-step field, so the
/// batch would have re-keyed and re-uploaded its whole instance buffer sixty
/// times a second, and nothing in the repository would have said so.
///
/// A ban enumerates what is forbidden; the set of per-step fields on
/// `ChunkState` is open (it grew `age_s` in P22.3 and will grow again). So this
/// pins the **whole `DebrisSite` literal, field by field, expression by
/// expression** — every value is one of five exact strings, and anything else at
/// all fails, whether or not anybody thought of it.
///
/// Each expression is here because it is *frozen at the break*: the placement
/// (and therefore `chunk_rest_center`), the volume-derived radius, the chunk
/// index, and the detach order, which is minted once. None of them can move while
/// a chunk tumbles, so the batch's content key cannot either.
#[test]
fn the_debris_site_projection_reads_only_frozen_state() {
    /// `field: expression` — the ONLY values `DebrisSite` may be built from.
    const ALLOWED: [(&str, &str); 5] = [
        // The actor's render id, which the entity fold already produced.
        ("entity", "id"),
        // The chunk's index in its `.inf_fracture`.
        ("chunk", "i as u32"),
        // Minted once, at detach, and never touched again.
        ("order", "c.detach_order"),
        // The REST centre — `placement · center_of_mass`, and `placement` is
        // frozen the instant the first chunk comes off.
        ("center", "state.chunk_rest_center(i)"),
        // A pure function of the chunk's authored volume and the placement's
        // determinant.
        ("radius_m", "state.chunk_radius_m(i)"),
    ];
    for (label, path) in [("editor viewport", VIEWPORT), ("shipped player", PLAYER)] {
        let fields = struct_literal_fields(&read(path), "inf_render::DebrisSite");
        assert_eq!(
            fields.len(),
            ALLOWED.len(),
            "the {label}'s `DebrisSite` literal has {} field(s), not {} — a field \
             was added or dropped on one side: {fields:?}",
            fields.len(),
            ALLOWED.len()
        );
        for ((name, value), (want_name, want_value)) in fields.iter().zip(ALLOWED) {
            assert_eq!(
                (name.as_str(), value.as_str()),
                (want_name, want_value),
                "the {label} builds `DebrisSite::{name}` from `{value}`. Every \
                 field of a debris site must be state that is FROZEN AT THE BREAK, \
                 or the batch re-keys and re-uploads its whole instance buffer \
                 every step. If this change is deliberate, prove the new \
                 expression cannot move between two detaches and add it here."
            );
        }
    }
}

/// A guard on the guard: two stubs are identical too, and a debris projection
/// that quietly stopped being *content-keyed* would look identical to one that
/// still was.
#[test]
fn the_shared_debris_projector_is_not_a_stub() {
    let body = extract_item(&read(PLAYER), "project_debris");
    for fragment in [
        // The same atomicity predicate the chunks use — rubble is never shed by an
        // intact actor.
        "if state.is_intact() {",
        "return None;",
        // A reclaimed chunk sheds nothing: the budget's despawn takes the body,
        // the collider and the rubble out together.
        "c.detached && !c.gone",
        // The placement rule itself is Ring 0, reached through the host's memo
        // (which calls `inf_render::debris_batch` on a miss) and never
        // re-implemented.
        "cache.batch(",
        "state.generation(),",
        "inf_render::DEBRIS_RUBBLE_PER_CHUNK,",
    ] {
        assert!(
            body.contains(fragment),
            "`project_debris` no longer contains `{fragment}` — either it was \
             gutted, or this gate needs updating deliberately:\n{body}"
        );
    }
    // …and, like the chunk projection, it must never see a camera.
    for probe in ["eye", "camera", "frustum"] {
        assert!(
            !body.contains(probe),
            "`project_debris` names `{probe}`:\n{body}"
        );
    }
}

/// The surrounding *rules*: both hosts push the batch onto the same list, from
/// the same `and_then` that decided the actor was broken — so the chunks and
/// their rubble cannot disagree about which actors have come apart.
#[test]
fn both_projectors_shed_debris_the_same_way() {
    const SHARED: [&str; 4] = [
        "project_debris(",
        "scatter.extend(rubble)",
        "fracture_chunks.extend(chunks)",
        // The payload is memoized against the fracture GENERATION, not rebuilt per
        // frame — the P22.4 audit's M2. A host that dropped the cache would pack
        // 1 536 instances and hash them sixty times a second and no other test
        // would notice, because the *output* is identical either way.
        "cache.batch(",
    ];
    for (label, path) in [("editor viewport", VIEWPORT), ("shipped player", PLAYER)] {
        let src = read(path).replace("\r\n", "\n");
        for fragment in SHARED {
            assert!(
                src.contains(fragment),
                "the {label}'s debris projection no longer contains `{fragment}` — \
                 either the path was changed on one side only, or this gate needs \
                 updating deliberately"
            );
        }
        // The tier must not reach the projection. `debris_budget_for` is a HOST
        // decision made once, where a session is owned; a projector that clamped
        // its own rubble by the local adapter would make the two hosts' scenes
        // differ by machine, which is precisely what this file exists to stop.
        assert!(
            !src.contains("debris_budget_for("),
            "the {label}'s projector names `debris_budget_for` — the tier→budget \
             mapping belongs to the host that owns the session, never to a \
             projection"
        );
    }
}

/// **Both fixed steps must run the P22.3 fracture slots, in order.**
///
/// The re-audit's item 5: deleting the editor host's `follow_fractures` call left
/// **460 tests green**. Every fracture test drives the bridge directly, so the
/// only thing that notices a host forgetting the call is a human reading the
/// file — which is exactly the "a boot path that forgets an attachment does not
/// crash, it agrees with itself" law from P21.4. This is the slot-parity gate for
/// it, on the `both_fixed_steps_run_the_deformation_slot` precedent above.
#[test]
fn both_fixed_steps_run_the_fracture_slots() {
    const RUNTIME_SIM: &str = "runtime/inf-player/src/runtime_sim.rs";
    const SIMULATE: &str = "editor/crates/inf-editor-core/src/simulate.rs";
    for (label, path, follow) in [
        (
            "shipped player",
            RUNTIME_SIM,
            "PhysicsBridge3D::follow_fractures(&self.world, &mut self.fractures);",
        ),
        (
            "editor Simulate",
            SIMULATE,
            "PhysicsBridge3D::follow_fractures(doc.world(), &mut self.fractures);",
        ),
    ] {
        // **Scoped to the fixed step.** Both hosts also sync the bridge once in
        // their CONSTRUCTOR, and a whole-file search finds that one first — which
        // is how the first cut of this gate reported the shipped player following
        // "after the sync" when it does no such thing. A gate that reads the wrong
        // function is a gate on nothing.
        let whole = read(path).replace("\r\n", "\n");
        let start = whole
            .find("fn fixed_step(")
            .expect("both hosts have a `fixed_step`");
        let src = whole[start..].to_string();
        assert!(
            src.contains(follow),
            "the {label} fixed step does not FOLLOW its fractures (`{follow}`) — an \
             intact destructible moved after load would shatter where it used to be"
        );
        for call in [
            "write_back_fractures(&mut self.fractures)",
            "step_fractures(&mut self.fractures, dt, self.debris_budget)",
        ] {
            assert!(
                src.contains(call),
                "the {label} fixed step does not call `{call}` — the two hosts \
                 would disagree about when debris moves and what collapses"
            );
        }

        // ORDER. Follow before the sync (the sync reads the map this writes);
        // write-back and the advance after the solver (support is a query against
        // where bodies actually ended the step).
        let at_follow = src.find(follow).expect("follow present");
        let at_sync = src
            .find("sync_from_world_sim(")
            .expect("the sim sync must exist");
        let at_step = src[at_sync..]
            .find("bridge3d.step(dt)")
            .map(|i| i + at_sync)
            .expect("the 3D solver step must exist");
        let at_write = src
            .find("write_back_fractures(")
            .expect("the fracture write-back must exist");
        let at_advance = src
            .find("step_fractures(")
            .expect("the fracture advance must exist");
        assert!(
            at_follow < at_sync,
            "the {label} follows its fractures AFTER the sync — the colliders would \
             be described from last step's placement"
        );
        assert!(
            at_step < at_write && at_write < at_advance,
            "the {label} runs the fracture slots out of order: the poses must be \
             written back after the solver, and the solve+budget after those"
        );
    }
}

// ── cloth (P24.4) ───────────────────────────────────────────────────────────

/// Both fixed steps must run the **cloth slot**, must run it through the ONE
/// Ring-0 rule, and must run it in the same place relative to the pose and the
/// attachments.
///
/// This is the sim half of the P24.4 mirror and it is the half that decides
/// whether PIE matches shipping: a preview that folded a coat against last step's
/// arm would diverge from the shipped build on the first step a character moved.
///
/// Scoped to `fn fixed_step(` for the reason the fracture gate below records —
/// both hosts also mention their registries in setters, and a whole-file search
/// finds those first.
#[test]
fn both_fixed_steps_run_the_cloth_slot() {
    const RUNTIME_SIM: &str = "runtime/inf-player/src/runtime_sim.rs";
    const SIMULATE: &str = "editor/crates/inf-editor-core/src/simulate.rs";
    for (label, path, call) in [
        ("shipped player", RUNTIME_SIM, "self.step_cloth(dt);"),
        ("editor Simulate", SIMULATE, "self.step_cloth(doc, dt);"),
    ] {
        let whole = read(path).replace("\r\n", "\n");
        let start = whole
            .find("fn fixed_step(")
            .expect("both hosts have a `fixed_step`");
        let src = whole[start..].to_string();
        assert!(
            src.contains(call),
            "the {label} fixed step does not call the P24.4 cloth slot (`{call}`) \
             — the two hosts would disagree about how a garment falls"
        );
        // ORDER: after the attachments (which are after the pose), so the capsules
        // are read off the pose THIS step published and the model frame off a
        // settled `GlobalTransform`.
        let at = src.find(call).expect("call present");
        let attach = src[..at]
            .rfind("update_attachments(")
            .expect("the attachment pass must precede the cloth slot");
        let propagate = src[attach..at]
            .find("propagate();")
            .map(|i| i + attach)
            .expect("a propagate must sit between the attachments and the cloth");
        assert!(
            attach < propagate && propagate < at,
            "the {label} folds cloth before the pose/attachment propagate settled \
             — a coat would collide against last step's body"
        );
    }

    // …and the rule itself lives ONCE: neither host may spell the solve inline.
    for (label, path) in [
        ("shipped player", RUNTIME_SIM),
        ("editor Simulate", SIMULATE),
    ] {
        let src = read(path).replace("\r\n", "\n");
        assert!(
            src.contains("inf_ecs::cloth::step_cloth_simulation("),
            "the {label} does not reach the Ring-0 cloth rule"
        );
        assert!(
            !src.contains("inf_anim::cloth::step_cloth("),
            "the {label} calls the XPBD solver DIRECTLY — the binding (capsules, \
             model-space gravity, seeding) would then exist twice, which is the \
             shape `inf_ecs::deform` was written to retire"
        );
    }
}

// ── the weapon's muzzle (SK1b) ──────────────────────────────────────────────

/// **Both fixed steps settle the weapon AFTER the pose** — the arm behind SK1b's
/// stated muzzle latency (SK1b audit).
///
/// `inf_physics::d3::gameplay::muzzle_of` reads a shot's origin off the weapon
/// entity's `GlobalTransform`, and that entity is placed by `update_attachments`,
/// which runs *after* `step_gameplay`. So a weapon-derived muzzle is **one fixed
/// step behind the hand**, and the wave wrote down why that is acceptable:
/// *"identical in both hosts, so no trace can see it."*
///
/// That is a claim about two files and it was prose. A trace comparison cannot
/// check it — a PIE==shipping gate compares two hosts, so it is blind to exactly
/// the thing both hosts do the same way, which is the whole point of this file.
/// If one host ever moved its gameplay step below the pose, its muzzle would lead
/// the other's by 16.7 ms and every shot would start somewhere else.
///
/// So the ORDER is pinned in both, on the `both_fixed_steps_run_the_cloth_slot`
/// precedent: gameplay, then the pose, then the attachments.
#[test]
fn both_fixed_steps_settle_the_weapon_after_the_pose() {
    const RUNTIME_SIM: &str = "runtime/inf-player/src/runtime_sim.rs";
    const SIMULATE: &str = "editor/crates/inf-editor-core/src/simulate.rs";
    for (label, path) in [
        ("shipped player", RUNTIME_SIM),
        ("editor Simulate", SIMULATE),
    ] {
        let whole = read(path).replace("\r\n", "\n");
        let start = whole
            .find("fn fixed_step(")
            .expect("both hosts have a `fixed_step`");
        let src = whole[start..].to_string();
        let at = |needle: &str| -> usize {
            src.find(needle)
                .unwrap_or_else(|| panic!("the {label} fixed step does not call `{needle}`"))
        };
        let gameplay = at("step_gameplay(");
        let pose = at("advance_state_machines(");
        let attach = at("update_attachments(");
        assert!(
            gameplay < pose && pose < attach,
            "the {label} runs gameplay/pose/attachments out of order \
             ({gameplay}/{pose}/{attach}) — the muzzle's one-step lag would differ \
             between the two hosts and every shot would start somewhere else"
        );
        // …and the pose really is the ONE Ring-0 door, so "after the pose" is
        // after the thing that publishes the socket a weapon hangs from.
        assert!(
            whole.contains("inf_ecs::pose::step_pose_evaluation("),
            "the {label} does not reach the Ring-0 pose rule"
        );
    }
}

/// **Root motion moves the character BEFORE the pose is evaluated against it**
/// — in both hosts, with a propagate in between.
///
/// # The landmine this closes (SK1b audit LOW, SK1c clause 4)
///
/// The two hosts ran `apply_root_motion` and `advance_state_machines` in
/// **opposite** order. They agreed on every committed trace, which the audit
/// recorded honestly as *unmeasured* rather than as *commuting* — extending a
/// pin to a divergence nobody has measured is how a gate goes red for a reason
/// nobody can name.
///
/// It was measured, and they do not commute. `step_pose_evaluation` reads the
/// entity's `GlobalTransform` twice — `authored_ik_goals` inverts it to bring a
/// world-space goal into model space, and `model_to_world` feeds the foot pass,
/// the hand pass and the feet it publishes — so the pass order decides which
/// step's placement all of those are computed in. On
/// `pose_parity`'s root-motion fixture the two hosts produced **different pose
/// bytes on every one of eight steps**, worst component **0.060**, while the
/// transform itself agreed to the bit: a divergence entirely inside the pose,
/// which is precisely where a `RootMotion` component looks harmless.
///
/// The shipped player moved to the editor's order, because root motion is
/// *movement* and every other movement in this engine happens before the pose
/// (`step_character_movement`, `step_gameplay`, and `anim_bridge`'s documented
/// one-step latency all rest on that). The propagate is not decoration: without
/// it the pose still reads last step's `GlobalTransform` and the reorder buys
/// nothing.
///
/// The behavioural half is `pose_parity`'s
/// `both_hosts_pose_a_root_motion_driven_character_the_same_way`; this is the
/// half that survives a future fixture nobody writes.
#[test]
fn both_fixed_steps_move_the_root_before_the_pose() {
    const RUNTIME_SIM: &str = "runtime/inf-player/src/runtime_sim.rs";
    const SIMULATE: &str = "editor/crates/inf-editor-core/src/simulate.rs";
    for (label, path) in [
        ("shipped player", RUNTIME_SIM),
        ("editor Simulate", SIMULATE),
    ] {
        let whole = read(path).replace("\r\n", "\n");
        let start = whole
            .find("fn fixed_step(")
            .expect("both hosts have a `fixed_step`");
        let src = whole[start..].to_string();
        let at = |needle: &str| -> usize {
            src.find(needle)
                .unwrap_or_else(|| panic!("the {label} fixed step does not call `{needle}`"))
        };
        let advance = at("advance_anim_players(");
        let root = at("self.apply_root_motion(");
        let pose = at("advance_state_machines(");
        assert!(
            advance < root && root < pose,
            "the {label} runs play-heads/root-motion/pose out of order \
             ({advance}/{root}/{pose}) — the two hosts would convert every \
             world-space IK goal, foot publish and hand target through \
             transforms one fixed step apart, and `pose_state_bytes` forks"
        );
        // …and the propagate between them, which is what makes the reorder mean
        // anything: `apply_root_motion` writes `Transform`, and the pose reads
        // `GlobalTransform`. Without a propagate the pose still sees last step's
        // placement and the two hosts agree only because they are both wrong.
        let between = &src[root..pose];
        assert!(
            between.contains("propagate()"),
            "the {label} does not propagate between root motion and the pose — \
             `apply_root_motion` writes `Transform` and `step_pose_evaluation` \
             reads `GlobalTransform`, so without it the reorder is inert"
        );
    }
}

/// **Both hosts tier the crowd, and both tier it BEFORE the three passes that
/// read a tier** (NPC1a).
///
/// `inf_ecs::crowd::step_crowd` is one Ring-0 door, so the two hosts cannot
/// disagree about *what* a tier means. What they could still disagree about is
/// *when* it is decided, and that is not a stylistic difference: the tier gates
/// the 3D bridge's bodies (NPC1c: the tier puts the body COMPONENTS on and takes
/// them off, so a host that tiered late would hand the bridge a body the ladder
/// had already decided against), the character step and the pose evaluation. A
/// host that tiered after its physics sync would give a `Far` agent a capsule
/// for one step; a host that tiered
/// after `advance_state_machines` would pose an agent it had just decided not to
/// pose. Either is a one-step disagreement between two hosts, which is exactly
/// what the island gate's byte compare exists to catch and exactly the kind of
/// thing that would take a thousand steps to show up.
///
/// So the ordering is pinned as source text in both files, the
/// `both_fixed_steps_move_the_root_before_the_pose` shape.
#[test]
fn both_fixed_steps_tier_the_crowd_before_the_passes_that_read_a_tier() {
    for (label, path) in [
        ("shipped player", "runtime/inf-player/src/runtime_sim.rs"),
        (
            "editor Simulate",
            "editor/crates/inf-editor-core/src/simulate.rs",
        ),
    ] {
        let whole = read(path).replace("\r\n", "\n");
        let start = whole
            .find("fn fixed_step(")
            .expect("both hosts have a `fixed_step`");
        let src = whole[start..].to_string();
        let at = |needle: &str| -> usize {
            src.find(needle)
                .unwrap_or_else(|| panic!("the {label} fixed step does not call `{needle}`"))
        };
        // **The spelling is pinned again** (NPC1a audit), and with the paren.
        // The wave's own version of this arm dropped it, because the player had
        // taken the radii seam (`step_crowd_banded`) while the editor still
        // called `step_crowd` -- i.e. the two hosts were passing *different
        // arguments* to one door, which is a divergence the ordering assertion
        // below cannot see and which becomes a PIE-!=-shipping bug the day a
        // level's crowd block sets a radius. Both hosts now carry a mirrored
        // `crowd_radii` field and both call the banded door, so the needle can
        // be exact again -- and being exact is what makes it catch a host that
        // quietly reverts to the constant-radii spelling.
        let crowd = at("inf_ecs::crowd::step_crowd_banded(");
        let bridge = at("sync_from_world_sim(");
        let mover = at("step_character_movement(");
        let pose = at("advance_state_machines(");
        assert!(
            crowd < bridge && crowd < mover && crowd < pose,
            "the {label} tiers the crowd at {crowd}, after one of the passes \
             that reads a tier (bridge {bridge}, movement {mover}, pose {pose}) \
             — a `Far` agent would keep a capsule, or be posed, for the step \
             its tier said not to"
        );
        // **And the society grows BEFORE the crowd is tiered** (NPC1d). A host
        // that installed a level's residents after the tiering would leave every
        // one of them `Dormant` — the tier a `CrowdRecord` is built with — for
        // exactly one step, and the OTHER host would not. That is a one-step
        // disagreement in the crowd trace section on the step a settlement
        // finishes streaming, which is the shape this whole arm exists for.
        let society = at("inf_ecs::society::sync_society(");
        assert!(
            society < crowd,
            "the {label} grows its society at {society}, AFTER it tiers the \
             crowd at {crowd}"
        );
        // The clock is advanced before both, because a schedule reads it.
        let sky = at("advance_time_of_day(");
        assert!(
            sky < society,
            "the {label} advances its clock at {sky}, after it grows its \
             society at {society} — a day's first step would be planned against \
             yesterday's hour"
        );
    }
}

/// **The lines of a fn body a COMPILER would see** — every whole-line `//`
/// comment dropped (EMS1 audit).
///
/// The two allowlists below are substring searches over a function's source,
/// and until this existed a fold **commented out** still satisfied its own pin:
/// measured, by commenting out `traffic::traffic_state_bytes` — the very line
/// wave EMS1 added to `SECTIONS` — and watching
/// `every_trace_section_is_folded_in_its_frozen_order` go green on a player
/// that no longer folds traffic at all. A pin that reads a line the compiler
/// does not is a pin on a comment.
///
/// **Whole-line comments only**, and deliberately: stripping from the first
/// `//` on every line would cut a `//` inside a string literal, and the defect
/// on the record is a *statement* commented out rather than a fold hidden
/// behind a trailing note. A `/* … */` block still defeats both arms and is
/// carried, named here rather than discovered.
fn code_lines(body: &str) -> String {
    body.lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// **Every pose writer runs, in a frozen order** (SK1a) — the twin of the trace
/// law below, one level down.
///
/// That law pins which SECTIONS the trace folds. This pins the order of the
/// writers *inside* the section, and it needs its own arm for the same reason:
/// the pose is a sequence of passes that each overwrite part of what the one
/// before wrote, so moving one changes every committed hash — and two hosts that
/// both moved it agree perfectly, which is what makes a trace comparison blind to
/// it. SK1a inserted a pass into the middle of this list, which is exactly the
/// edit that needed a pin to exist before it, so the pin is written here now.
///
/// An **allowlist over the whole sequence**, not a check that the new one is
/// present: naming only the newest writer says nothing about the six that were
/// already there, which is the mistake the I6 audit caught in the trace law.
///
/// It reads the body through [`code_lines`], because a substring search cannot
/// tell a pass from a pass somebody commented out.
#[test]
fn every_pose_writer_runs_in_its_frozen_order() {
    let src = read("crates/inf-ecs/src/pose.rs").replace("\r\n", "\n");
    let start = src
        .find("pub fn step_pose_evaluation<")
        .expect("the fixed step's one pose door");
    let body = &src[start..];
    let end = body.find("\n/// The **foot joints**").expect("the fn ends");
    let body = code_lines(&body[..end]);
    let body = body.as_str();
    // The sequence, in the order each pass writes into the pose. A pass deleted
    // fails at its own `expect`; a pass MOVED fails the ordering assertion.
    const WRITERS: [(&str, &str); 8] = [
        (
            "pending_pose = Some(pose);",
            "the machine + the layer stack + the inertializer produce the pose",
        ),
        (
            "inf_anim::drive_pose(",
            "SK1a: the twist chains and the IK handles, which is pose CONSTRUCTION and not a correction",
        ),
        (
            "pelvis_joint(asset)",
            "P29.5: the pelvis drop, before the legs solve from a lowered hip",
        ),
        (
            "inf_anim::solve_chain(",
            "P24.2: the authored and runtime IK goals",
        ),
        (
            "apply_foot_ik(asset,",
            "P29.4: foot IK, over the same solver",
        ),
        (
            "apply_hand_ik(asset,",
            "SK1b: the arms that reach, the off hand the weapon carries, and the fingers that close — after the feet, because a stance is decided by the ground and a hand solves against the body that stance produced",
        ),
        (
            "redrive(asset,",
            "SK1b: the CORRECTION re-drive, which closes SK1a's stated ordering bound — a twist bone is a statement about the pose that is finally published, so it is computed from the corrected one and not from the authored one. It is a named fn precisely so this pin can see it: a second `inf_anim::drive_pose(` would be the same needle as the first",
        ),
        (
            "foot_states(asset,",
            "P29.4: the feet published for the NEXT step, read off the final pose",
        ),
    ];
    let at: Vec<usize> = WRITERS
        .iter()
        .map(|(needle, why)| {
            body.find(needle).unwrap_or_else(|| {
                panic!(
                    "`{needle}` does not run in `step_pose_evaluation` at all — {why}; a pass missing from BOTH hosts is invisible to every trace comparison in this repository"
                )
            })
        })
        .collect();
    println!(
        "the fixed step runs {} pose writers at {at:?}",
        WRITERS.len()
    );
    for (i, w) in at.windows(2).enumerate() {
        assert!(
            w[0] < w[1],
            "the pose writers moved: `{}` must run before `{}` — every committed pose hash was taken with them in this order",
            WRITERS[i].0,
            WRITERS[i + 1].0
        );
    }
}

/// **Every trace section is folded, in a frozen order** — the whole list, not
/// the newest one.
///
/// Every committed trace hash in the tree was taken over the concatenation
/// `snapshot ++ deform ++ pose`; P24.4 appended `cloth` and `hair` after it and
/// island wave I6 appended `door ++ item ++ weapon ++ health`. Inserting a
/// section anywhere but the tail would change every one of those hashes
/// silently, so the order is pinned here rather than trusted to a comment.
///
/// **It is an ALLOWLIST over the whole sequence, and the I6 audit is why.** The
/// arm used to name three sections and assert their order, which said nothing
/// about the five that arrived later: measured, deleting
/// `door::door_state_bytes` from the fold — and, separately,
/// `weapon::weapon_state_bytes` — left **all sixty-nine `inf-player` test
/// binaries green**, the PIE-versus-shipping gate included, because two hosts
/// that both stopped folding a section agree about it perfectly. A gate that
/// compares two traces cannot see a section missing from both; only a pin on
/// the fold itself can. (P22's own "a ban enumerates what you thought of, an
/// allowlist what is allowed", at a trace.)
#[test]
fn every_trace_section_is_folded_in_its_frozen_order() {
    let src = read("runtime/inf-player/src/runtime_sim.rs").replace("\r\n", "\n");
    let start = src
        .find("pub fn state_bytes(")
        .expect("the player folds a trace");
    let body = &src[start..];
    let end = body.find("\n    }\n").expect("the fn ends");
    // **Through [`code_lines`]**, because a substring search over raw source
    // cannot tell a fold from a fold somebody commented out — measured, on this
    // arm's own newest entry.
    let body = code_lines(&body[..end]);
    let body = body.as_str();
    // The sequence, in the order the bytes are concatenated. A section deleted
    // from the fold fails at its own `expect`; a section MOVED fails the
    // ordering assertion below.
    const SECTIONS: [&str; 13] = [
        "deform::deform_state_bytes",
        "pose::pose_state_bytes",
        "cloth::cloth_state_bytes",
        "hair::hair_state_bytes",
        "door::door_state_bytes",
        "item::item_state_bytes",
        "weapon::weapon_state_bytes",
        "weapon::health_state_bytes",
        // NPC1a. The one section whose ABSENCE is invisible in a way the eight
        // above are not: a `Far` agent evaluates no pose and a `Dormant` one
        // has no entity, so a sim-LOD tier that differed between two hosts
        // would produce two identical traces until one of them ran a pose the
        // other did not.
        "crowd::crowd_state_bytes",
        // VEH2b, pinned at EMS1 -- and the gap is the point. The fold grew a
        // tenth section on the crowd's own argument (a `Near` car is not
        // simulated and a `Dormant` one has no entity, so a tier decision that
        // differed between two hosts would be invisible), and the PIN was not
        // moved with it. For two waves, deleting or moving the traffic fold
        // would have left every `inf-player` binary green -- which is the exact
        // measurement the I6 audit made about `door` and `weapon`, unlearned in
        // the time it takes to write one line.
        "traffic::traffic_state_bytes",
        // EMS2, pinned in the SAME commit that folded it -- which is the whole
        // of what the line above learned. A unit's state (in station, en route,
        // on scene, returning) decides everything the dispatch step does with a
        // vehicle and is not a transform anything else folds, so two hosts that
        // sent different ambulances to one fire would compare equal until one
        // of them happened to solve a chassis the other did not.
        "dispatch::dispatch_state_bytes",
        // EMS3, pinned in the SAME commit that folded them. Two sections and not
        // one, because they fail for different reasons: an ACT's observer list
        // is a line-of-sight ray (so two hosts that built different colliders
        // disagree here first), and a PROFILE is heat, evidence and a last-seen
        // position (so two hosts that recognised different people disagree
        // there). Neither is a transform anything else folds, and a wanted level
        // that differed between PIE and shipping is the exact divergence a
        // player experiences as "the police behaved differently in the editor".
        "witness::witness_state_bytes",
        "crime::profile_state_bytes",
    ];
    let at: Vec<usize> = SECTIONS
        .iter()
        .map(|s| {
            body.find(s).unwrap_or_else(|| {
                panic!(
                    "`{s}` is not appended to `state_bytes` at all — a section \
                        missing from BOTH hosts is invisible to every trace comparison \
                        in this repository"
                )
            })
        })
        .collect();
    println!("the trace folds {} sections at {at:?}", SECTIONS.len());
    for (i, w) in at.windows(2).enumerate() {
        assert!(
            w[0] < w[1],
            "the trace sections moved: `{}` must be folded before `{}` — every \
             committed hash was taken over {SECTIONS:?}, in that order",
            SECTIONS[i],
            SECTIONS[i + 1]
        );
    }
}

/// **EVERY RESOURCE SIMULATE WIPES ON THE WAY IN, IT WIPES ON THE WAY OUT** —
/// the twin pin (wave EMS3).
///
/// # This gate did not exist and the wave that needed it found out
///
/// `SimSession::enter_with_gravity` and `SimSession::exit` each call a list of
/// `clear_*` doors, and the two lists have to be the same list. Nothing checked
/// that. A door added to `enter` and forgotten in `exit` leaves a **resource in
/// the author's document**: run 1's wanted level, run 1's parked traffic, run
/// 1's duty roster. It is the exact defect every one of those doors' own doc
/// comments describes, and it was enforced only by whoever was editing at the
/// time remembering to scroll four hundred lines down.
///
/// So the two blocks are read out of the source and compared as **sets**. Not as
/// sequences: the order a session forgets things in has no meaning (they are
/// independent resources), and pinning an order would fail a commit that merely
/// tidied one. What has meaning is the *membership*, and a missing member is
/// content in a level nobody authored.
///
/// It also asserts the list is not empty and holds the doors this wave depends
/// on by name, so a future refactor that replaced twelve calls with one
/// `clear_everything()` has to come back here and say so.
#[test]
fn simulate_forgets_on_exit_exactly_what_it_forgot_on_entry() {
    let src = read("editor/crates/inf-editor-core/src/simulate.rs").replace("\r\n", "\n");
    // The two blocks, each ending at the last `clear_*` call in it. `enter`
    // continues into the bridge construction and `exit` ends the function, so
    // both are bounded by finding their own calls rather than by a brace.
    let block = |anchor: &str| -> std::collections::BTreeSet<String> {
        let start = src
            .find(anchor)
            .unwrap_or_else(|| panic!("`{anchor}` is not in `simulate.rs`"));
        let body = &src[start..];
        // Far enough to hold the whole clear block and stop well before the next
        // one: the two are hundreds of lines apart.
        let window = &body[..body.len().min(6000)];
        let mut out = std::collections::BTreeSet::new();
        for line in window.lines() {
            let line = line.trim();
            // Through the compiler's eyes, exactly as `code_lines` does for the
            // trace fold: a clear somebody commented out is a clear that does
            // not run, and a pin that reads a comment is a pin on a comment.
            if line.starts_with("//") {
                continue;
            }
            let Some(rest) = line.strip_prefix("inf_ecs::") else {
                continue;
            };
            let Some(call) = rest.split('(').next() else {
                continue;
            };
            if call.contains("::clear_") {
                out.insert(call.to_string());
            }
        }
        out
    };
    let entered = block("pub fn enter_with_gravity(");
    let exited = block("    pub fn exit(");
    println!(
        "Simulate clears {} resources on entry and {} on exit",
        entered.len(),
        exited.len()
    );
    assert!(
        entered.len() >= 12,
        "only {} `clear_*` doors were found on the way in — this gate is reading \
         the wrong block, or the list moved",
        entered.len()
    );
    for door in ["crowd::clear_crowd", "witness::clear_witness"] {
        assert!(
            entered.contains(door),
            "`inf_ecs::{door}` is not in the entry block — either it moved or \
             this gate is reading the wrong window"
        );
    }
    let missing_on_exit: Vec<&String> = entered.difference(&exited).collect();
    assert!(
        missing_on_exit.is_empty(),
        "Simulate clears {missing_on_exit:?} on the way IN and never on the way \
         OUT — whatever those resources hold is left in the author's document \
         when the session stops"
    );
    let missing_on_entry: Vec<&String> = exited.difference(&entered).collect();
    assert!(
        missing_on_entry.is_empty(),
        "Simulate clears {missing_on_entry:?} on the way OUT and never on the way \
         IN — run 2 of a session begins on run 1's state, and its trace will not \
         match the shipped player's, which starts from nothing every time"
    );
}

/// The two `project_cloth` projectors must be **byte-identical, doc block
/// included**.
///
/// The rule they share is small — read the sim's cloth store, build one deformed
/// skinned mesh through the Ring-0 builder, push one instance — and that is
/// exactly the kind of rule that drifts: a tint changed on one side, a roughness
/// on the other, and PIE stops matching shipping in a way no scene-level
/// comparison can see.
#[test]
fn project_cloth_is_identical_in_both_projectors() {
    let mine = extract_fn(&read(VIEWPORT), "project_cloth");
    let theirs = extract_fn(&read(PLAYER), "project_cloth");
    assert_eq!(
        mine, theirs,
        "the two `project_cloth` projectors have drifted — a garment would be \
         drawn differently in the editor viewport than in the shipped player"
    );
}

/// …and it is **not a stub**: the shared body really does everything the doc
/// claims, through the doors it claims.
#[test]
fn the_shared_cloth_projector_is_not_a_stub() {
    let src = extract_item(&read(PLAYER), "project_cloth");
    for fragment in [
        // It reads the SIM, not an asset store.
        "inf_ecs::cloth::live_cloth(world, guid)",
        // …through the ONE Ring-0 vertex builder.
        "inf_render::deformed_skinned_mesh(&live.state.x, &live.state.indices)",
        // …onto the skinned path, with the identity palette that makes a
        // model-space garment land where the sim put it.
        "scene.skinned_meshes.push(std::sync::Arc::new(mesh))",
        // …with the SHARED one-entry identity palette (wave NPC1b). `vec![]` at
        // each call site was two allocations naming one value, and the renderer's
        // atlas deduplicates blocks by pointer identity — so two `vec![]`s are two
        // blocks and one `identity_palette()` is one, for every garment in a level.
        "palette: inf_render::identity_palette()",
        "shadow: inf_render::SkinnedShadow::BindSphere",
        "color: inf_render::CLOTH_TINT",
        "id: inf_render::ID_NONE",
        // …and it refuses an empty garment rather than emitting a draw for no
        // pixels.
        "if mesh.indices.len() < 3 {",
    ] {
        assert!(
            src.contains(fragment),
            "`project_cloth` no longer contains `{fragment}` — {src}"
        );
    }
    // A garment's geometry MOVES, so it may never be memoized behind the
    // `Arc`-pointer cache the static skinned path uses: a store lookup here would
    // hand out last step's fold for ever.
    assert!(
        !src.contains("skinned_slots"),
        "`project_cloth` deduplicates through `skinned_slots` — a garment's \
         vertices change every step, so a shared slot would freeze it"
    );
}

/// Both projectors must call it, **outside** the `MeshRef`-absent branch, so a
/// garment is drawn beside its wearer's geometry rather than instead of it.
#[test]
fn both_projectors_draw_cloth_beside_the_wearer() {
    for (label, path) in [("editor viewport", VIEWPORT), ("shipped player", PLAYER)] {
        let src = read(path);
        let call = src
            .find("project_cloth(")
            .expect("both projectors must call the cloth projector");
        // The call comes BEFORE the skeletal branch's `MeshRef` probe, which is
        // what puts it outside that branch.
        let probe = src
            .find("w.get::<MeshRef>(entity).is_none()")
            .expect("both projectors probe MeshRef");
        assert!(
            call < probe,
            "the {label} calls `project_cloth` inside (or after) the \
             `MeshRef`-absent branch — a character with a static mesh and a cloak \
             would draw the mesh and lose the cloak"
        );
    }
}

/// The two `project_hair` projectors must be **byte-identical, doc block
/// included** — the `project_cloth` gate, on the ribbons.
#[test]
fn project_hair_is_identical_in_both_projectors() {
    assert_eq!(
        extract_fn(&read(VIEWPORT), "project_hair"),
        extract_fn(&read(PLAYER), "project_hair"),
        "the two `project_hair` projectors have drifted"
    );
}

/// Both fixed steps must run the **hair slot**, through the ONE Ring-0 rule, and
/// adjacent to the cloth slot.
#[test]
fn both_fixed_steps_run_the_hair_slot() {
    const RUNTIME_SIM: &str = "runtime/inf-player/src/runtime_sim.rs";
    const SIMULATE: &str = "editor/crates/inf-editor-core/src/simulate.rs";
    for (label, path, cloth, hair) in [
        (
            "shipped player",
            RUNTIME_SIM,
            "self.step_cloth(dt);",
            "self.step_hair(dt);",
        ),
        (
            "editor Simulate",
            SIMULATE,
            "self.step_cloth(doc, dt);",
            "self.step_hair(doc, dt);",
        ),
    ] {
        let whole = read(path).replace("\r\n", "\n");
        let start = whole.find("fn fixed_step(").expect("a fixed step");
        let src = whole[start..].to_string();
        let at_cloth = src.find(cloth).expect("the cloth slot");
        let at_hair = src
            .find(hair)
            .unwrap_or_else(|| panic!("the {label} fixed step does not run the hair slot"));
        assert!(
            at_cloth < at_hair,
            "the {label} runs the hair slot before the cloth slot; the two must \
             stay adjacent and ordered, or the hosts can diverge about which of \
             the two sees the other's writes"
        );
        // The rule lives ONCE: neither host may spell the strand solve inline.
        assert!(whole.contains("inf_ecs::hair::step_hair_simulation("));
        assert!(
            !whole.contains("inf_anim::hair::step_hair("),
            "the {label} calls the strand solver DIRECTLY — the binding (roots, \
             capsules, ribbons, seeding) would then exist twice, which is the \
             shape `inf_ecs::deform` was written to retire"
        );
    }
}

/// **The movement step publishes its parameters AFTER the Blueprint Tick**, in
/// both hosts — which is the precedence, and the opposite of what P29.6's own
/// documentation claimed in two places (P29.6 audit, A11).
///
/// `publish_character_params` writes nine names into the very overlay
/// `anim.set_param` writes into, and the last writer in a step wins. Both hosts
/// dispatch `EventKind::Tick` and *then* run `step_character_movement`, so on a
/// character with a `CharacterMovement` the **engine** owns `speed`, `gait`,
/// `grounded`, `mode`, `direction`, `fall_speed`, `land_alpha`, `overlay` and
/// `flail`, and a Blueprint that sets one of them is overwritten before anything
/// reads it. That is a defensible design — one authority for one fact — and it
/// is a real constraint on what a game can do, so it must be asserted rather
/// than described in a comment that said the reverse.
///
/// `live_tuning` pins the same precedence from the tuning door's side, in one
/// host. This pins the ORDER, in both, which is where the precedence comes from.
#[test]
fn both_fixed_steps_publish_character_params_after_the_blueprint_tick() {
    const RUNTIME_SIM: &str = "runtime/inf-player/src/runtime_sim.rs";
    const SIMULATE: &str = "editor/crates/inf-editor-core/src/simulate.rs";
    for (label, path) in [
        ("shipped player", RUNTIME_SIM),
        ("editor Simulate", SIMULATE),
    ] {
        let whole = read(path).replace("\r\n", "\n");
        let start = whole.find("fn fixed_step(").expect("a fixed step");
        let src = &whole[start..];
        let at_tick = src
            .find("&EventKind::Tick")
            .unwrap_or_else(|| panic!("the {label} fixed step runs no Blueprint Tick"));
        let at_move = src
            .find("step_character_movement(")
            .unwrap_or_else(|| panic!("the {label} fixed step runs no movement step"));
        assert!(
            at_tick < at_move,
            "the {label} runs the movement step BEFORE the Blueprint Tick, so a \
             Blueprint's `anim.set_param` would now win over the character's own \
             published state — the precedence inverted, and every doc comment \
             about it is now wrong in the other direction"
        );
    }
}

/// The hair trace is appended **after** the cloth trace, and both after the pose.
#[test]
fn the_hair_trace_is_appended_after_the_cloth_trace() {
    let src = read("runtime/inf-player/src/runtime_sim.rs").replace("\r\n", "\n");
    let start = src
        .find("pub fn state_bytes(")
        .expect("the player folds a trace");
    let body = &src[start..];
    let body = &body[..body.find("\n    }\n").expect("the fn ends")];
    let pose = body
        .find("pose::pose_state_bytes")
        .expect("the pose section");
    let cloth = body
        .find("cloth::cloth_state_bytes")
        .expect("the cloth section");
    let hair = body
        .find("hair::hair_state_bytes")
        .expect("the hair section is not appended at all");
    assert!(
        pose < cloth && cloth < hair,
        "the trace sections moved: every committed hash was folded over \
         snapshot ++ deform ++ pose ++ cloth ++ hair, in that order"
    );
}

/// **Both hosts gather EVERY terrain for the ground query** — the island phase's
/// IB-15 seam, mirrored.
///
/// The Ring-0 rule (`inf_voxel::ground_height_at`) takes a slice of
/// `(terrain, origin)` pairs and returns the topmost surface that answers. The
/// **gather** that fills that slice is written twice, once per host, and it is
/// where the defect lived: both hosts used to hand the rule the lowest-`Guid`
/// terrain and nothing else, so a character walking onto a second terrain read
/// `None` and fell to the host default `0.0`.
///
/// The I1 audit measured what an unmirrored gather is worth: reverting the
/// player's to a single terrain left all 64 `inf-player` test binaries green,
/// because the only behavioural arm called the Ring-0 rule with a slice it had
/// built itself. `multi_terrain_seam::the_shipped_host_seam_answers_on_both_terrains`
/// is now that arm for the player, and this is what carries it to the editor —
/// which cannot be reached behaviourally from here at all.
///
/// Anchored on the free function's own multi-line signature rather than through
/// [`item_start`], because `runtime_sim.rs` also has a `RuntimeSim::terrain_height_at`
/// **method** that forwards to it and is the first item of that name in the file.
#[test]
fn both_hosts_gather_every_terrain_for_the_ground_query() {
    const RUNTIME_SIM: &str = "runtime/inf-player/src/runtime_sim.rs";
    const SIMULATE: &str = "editor/crates/inf-editor-core/src/simulate.rs";
    const ANCHOR: &str = "fn terrain_height_at(\n    world: &mut EcsWorld,";

    let body = |path: &str| -> String {
        let src = read(path).replace("\r\n", "\n");
        let at = src.find(ANCHOR).unwrap_or_else(|| {
            panic!("{path}: the free `terrain_height_at(world, ..)` is not there under that name")
        });
        let rest = &src[at..];
        let end = rest
            .find("\n}\n")
            .unwrap_or_else(|| panic!("{path}: `terrain_height_at` does not end at column 0"))
            + 3;
        rest[..end].to_string()
    };
    let player = body(RUNTIME_SIM);
    let editor = body(SIMULATE);
    assert_eq!(
        player, editor,
        "the two hosts resolve the ground differently — one of them will drop a \
         character to sea level at a border the other walks across"
    );

    // …and it really is the position-aware gather, not a one-terrain pick with a
    // slice wrapped round it.
    for needle in [
        "let mut found: Vec<(Uuid, Entity, DVec3)> = Vec::new();",
        "found.sort_unstable_by_key(|(g, _, _)| *g);",
        "inf_voxel::ground_height_at(&terrains, voxels, x, z)",
    ] {
        assert!(
            player.contains(needle),
            "the shared ground gather no longer contains `{needle}`"
        );
    }
    // ANTI-VACUITY: every terrain the scan found reaches the rule. The defect
    // was a *narrowing* between the scan and the call, and it is spelled as one.
    for narrowing in [".take(1)", ".first()", ".into_iter().next()", "found[0]"] {
        assert!(
            !player.contains(narrowing),
            "the ground gather narrows its terrain set with `{narrowing}` — that \
             is the IB-15 defect, which reads as a character falling to y = 0 at \
             a two-terrain border"
        );
    }
}
