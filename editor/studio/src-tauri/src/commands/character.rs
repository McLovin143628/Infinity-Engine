//! **New Character from Template** (P24.5) — the wizard's Ring-2 door.
//!
//! Two commands, and between them they are the whole wizard:
//!
//! * [`character_preview`] answers "what would this spec make?" without writing
//!   anything. The panel calls it on every edit, so it must be cheap and must
//!   never be a toast: a refusal comes back as a **value** inside the DTO,
//!   because half the intermediate states of a proportion drag are invalid (hips
//!   above shoulders while a slider passes through, for one).
//! * [`character_create`] writes the six assets and, when asked, adds the actor
//!   to the open level.
//!
//! Every rule lives in `inf_editor_core::character` (see its module docs);
//! everything here is the string↔enum hop at the wire, the two `State` lookups,
//! and one event emission per side effect — the typed-IPC law.
//!
//! # The order of the two side effects
//!
//! Assets are written first and `assets://changed` is emitted **before** the
//! actor is spawned, so the Content Drawer already knows about the `.inf_skel`
//! and the `.inf_sm` by the time an entity references them. The other order
//! would put a `world://delta` naming three GUIDs the drawer has never heard of
//! on the wire, and the Details panel resolves asset names through that snapshot.

use std::sync::Mutex;

use inf_anim::BodyPlan;
use inf_asset::{AssetId, AssetKind};
use inf_editor_core::character::{
    build_character, CharacterPreviewSession, CharacterSpec, CHARACTER_FOLDER,
};
use inf_editor_core::ipc::{
    CharacterCreateDto, CharacterPreviewDto, CharacterRigDto, CharacterSpecDto,
};
use inf_mesh::MeshAsset;
use tauri::{AppHandle, State};

use super::assets::{emit_changed, AssetState};
use super::scene::{emit_world_delta, SceneState};

/// **The wizard's warm state** (P29.5, ROADMAP §13's F5 item).
///
/// Two things are kept between previews, and neither is a rule — the rules are
/// all in Ring 1:
///
/// * the **decoded model**, keyed by `(asset id, content hash)`. `fit_source`
///   used to `read` and decode the author's `.inf_mesh` on every keystroke, and
///   that is the half of the cold path that lives up here rather than in
///   `preview_character`. Keying on the hash and not only on the id is what makes
///   it correct while the Model Editor is open beside the wizard: an edited mesh
///   is a new hash, so it is a new decode.
/// * the Ring-1 [`CharacterPreviewSession`], which keeps the BVH, the locomotion
///   summary and the mannequin's counts.
///
/// One `Mutex` and not two, because a preview reads both and they are only ever
/// touched from `character_preview`, which is serialized against itself anyway.
#[derive(Default)]
pub struct CharacterWizardState {
    inner: Mutex<Warm>,
}

#[derive(Default)]
struct Warm {
    session: CharacterPreviewSession,
    model: Option<(AssetId, inf_asset::ContentHash, MeshAsset)>,
}

/// Parse the wire name of a body plan.
///
/// The same four names `skel_create_template` accepts, and the same reasoning:
/// a stringly-typed plan that fails loudly with the list of what it knows is
/// kinder to a frontend built against an older backend than a generated union
/// that silently loses a variant. Kept as its own function rather than shared
/// with `skel.rs` because the two are three lines each and a shared parser would
/// be a module named after a `match`.
fn parse_plan(plan: &str, legs: Option<u16>) -> Result<BodyPlan, String> {
    match plan {
        "biped" => Ok(BodyPlan::Biped),
        "biped-canonical" => Ok(BodyPlan::BipedCanonical),
        "quadruped" => Ok(BodyPlan::Quadruped),
        "hexapod" => Ok(BodyPlan::Hexapod),
        "npedal" => Ok(BodyPlan::Npedal {
            legs: legs.ok_or("an `npedal` character needs a leg count")?,
        }),
        other => Err(format!(
            "unknown body plan `{other}` (expected biped, biped-canonical, quadruped, hexapod or npedal)"
        )),
    }
}

/// The Ring-1 spec a wire spec describes, plus the mesh id it names.
fn to_spec(dto: &CharacterSpecDto) -> Result<CharacterSpec, String> {
    let mesh = match dto.mesh_asset.as_deref() {
        Some(id) if !id.is_empty() => Some(
            id.parse::<AssetId>()
                .map_err(|e| format!("bad asset id: {e}"))?,
        ),
        _ => None,
    };
    Ok(CharacterSpec {
        name: dto.name.clone(),
        plan: parse_plan(&dto.plan, dto.legs)?,
        params: dto.params.to_params(),
        gait: dto.gait.to_gait(),
        mesh,
    })
}

/// The identity of the spec's fit source — id and content hash — without
/// decoding it.
///
/// A missing or wrong-kind asset is a **refusal by name** rather than a silent
/// `None`: the `resolve_rig` rule P24.4 settled, because a wizard that quietly
/// ignored the mesh you picked would generate a mannequin and look like it had
/// worked.
fn fit_source_id(
    state: &State<'_, AssetState>,
    spec: &CharacterSpec,
) -> Result<Option<(AssetId, inf_asset::ContentHash)>, String> {
    let Some(id) = spec.mesh else {
        return Ok(None);
    };
    state.with_project(|p| {
        let entry = p.db().get(id).ok_or_else(|| format!("no asset {id}"))?;
        if entry.kind() != AssetKind::Mesh {
            return Err(format!("{} is not a mesh asset", entry.name));
        }
        Ok(Some((id, entry.sidecar.content_hash)))
    })
}

/// Describe what a spec would produce. Writes nothing.
///
/// **The warm path** (P29.5): the panel calls this on every edit, so both the
/// model decode and the fit's BVH are kept between calls — see
/// [`CharacterWizardState`]. The answer is identical to the cold one; only what
/// it costs changed.
#[tauri::command]
pub async fn character_preview(
    spec: CharacterSpecDto,
    assets: State<'_, AssetState>,
    wizard: State<'_, CharacterWizardState>,
) -> Result<CharacterPreviewDto, String> {
    let spec = to_spec(&spec)?;
    let ident = fit_source_id(&assets, &spec)?;
    let mut warm = wizard
        .inner
        .lock()
        .map_err(|_| "the character wizard's state is poisoned".to_string())?;
    if let Some((id, hash)) = ident {
        // A decode only when the identity moved. The hash and not just the id:
        // the Model Editor can be open beside the wizard, and an edited mesh
        // keeps its GUID.
        let hit = matches!(&warm.model, Some((i, h, _)) if *i == id && *h == hash);
        if !hit {
            let payload = assets
                .with_project(|p| p.load_payload::<MeshAsset>(id).map_err(|e| e.to_string()))?;
            warm.model = Some((id, hash, payload));
        }
    } else {
        warm.model = None;
    }
    let Warm { session, model } = &mut *warm;
    // The stamp the Ring-1 session keys its BVH on IS the content hash, folded
    // to 64 bits — the same identity this side just used for the decode, so the
    // two caches cannot disagree about which model they are holding.
    let source = model
        .as_ref()
        .map(|(_, h, m)| (m, h.0 as u64 ^ (h.0 >> 64) as u64));
    Ok(match session.preview(&spec, source) {
        Ok(p) => CharacterPreviewDto {
            refusal: None,
            rig: Some(CharacterRigDto::from_preview(&p)),
        },
        // A VALUE. See the module docs.
        Err(e) => CharacterPreviewDto {
            refusal: Some(e.to_string()),
            rig: None,
        },
    })
}

/// **Place the committed starter character** (wave GTA1) — `Place Actor ▸
/// Starter Character`, and the button on the no-pawn Play dialog.
///
/// The distinction from [`character_create`] is the whole point of having two
/// commands: that one is a *wizard* and mints six new assets every time it runs,
/// which is the wrong verb for "give this level a player". This one writes
/// nothing at all. It spawns an actor wearing
/// [`starter_character_ids`](inf_editor_core::samples::starter_character_ids) —
/// the guids of the seventeen files every 3D project scaffolds and the editor's
/// own content root is seeded with — through the same
/// `SceneDoc::edit_create_character` door the wizard's spawn half takes, so the
/// character is a character by the same definition.
///
/// One undo step, and the placed actor is selected, exactly like a wizard spawn.
///
/// **`female` picks which committed body** (wave CHAR1a.2). There are two now —
/// `samples/starter-character/` and `samples/starter-character-f/`, the same
/// eight assets on the same 161 joint names, built by the same wizard from two
/// sets of measured proportions — and every 3D template scaffolds both, so both
/// resolve in any project this command can run in. `None`/`false` keeps the
/// male body, which is what every existing caller means.
#[tauri::command]
pub async fn character_place_starter(
    app: AppHandle,
    at: Option<[f64; 3]>,
    female: Option<bool>,
    scene: State<'_, SceneState>,
) -> Result<String, String> {
    let female = female.unwrap_or(false);
    let ids = if female {
        inf_editor_core::samples::starter_character_f_ids()
    } else {
        inf_editor_core::samples::starter_character_ids()
    };
    let asset = |id: Option<AssetId>| -> Result<uuid::Uuid, String> {
        id.map(|a| a.0)
            .ok_or_else(|| "the starter character's ids are not all fixed".to_string())
    };
    let at = at.unwrap_or([0.0, 0.0, 0.0]);
    let guid = {
        let mut doc = super::scene::lock(&scene.doc)?;
        let guid = doc.edit_create_character(
            if female {
                inf_editor_core::samples::STARTER_CHARACTER_F_NAME
            } else {
                inf_editor_core::samples::STARTER_CHARACTER_NAME
            },
            asset(ids.skeleton)?,
            asset(ids.mesh)?,
            asset(ids.machine)?,
            glam::DVec3::new(at[0], at[1], at[2]),
            Some(asset(ids.actor)?),
            if female {
                inf_editor_core::samples::starter_character_f_spec()
                    .params
                    .height_m
            } else {
                inf_editor_core::samples::starter_character_spec()
                    .params
                    .height_m
            },
        );
        doc.select(&[guid], false);
        guid
    };
    emit_world_delta(&app, &scene);
    Ok(guid.to_string())
}

/// Generate the character's assets under `Content/Characters`, and optionally
/// add a ready actor to the open level.
///
/// `at` is the actor's world position in **metres** (units doctrine); it is
/// ignored when `add_to_scene` is false.
#[tauri::command]
pub async fn character_create(
    app: AppHandle,
    spec: CharacterSpecDto,
    add_to_scene: bool,
    at: Option<[f64; 3]>,
    assets: State<'_, AssetState>,
    scene: State<'_, SceneState>,
) -> Result<CharacterCreateDto, String> {
    let spec = to_spec(&spec)?;
    let built = assets.with_project(|p| build_character(p, &spec).map_err(|e| e.to_string()))?;
    // The drawer learns about the six assets BEFORE anything references them.
    emit_changed(&app, &assets);

    let actor = if add_to_scene {
        let at = at.unwrap_or([0.0, 0.0, 0.0]);
        let guid = {
            let mut doc = super::scene::lock(&scene.doc)?;
            let guid = doc.edit_create_character(
                spec.name.trim(),
                built.skeleton.0,
                built.mesh.0,
                built.machine.0,
                glam::DVec3::new(at[0], at[1], at[2]),
                Some(built.actor.0),
                spec.params.height_m,
            );
            doc.select(&[guid], false);
            guid
        };
        emit_world_delta(&app, &scene);
        Some(guid.to_string())
    } else {
        None
    };

    Ok(CharacterCreateDto {
        skeleton: built.skeleton.to_string(),
        mesh: built.mesh.to_string(),
        idle: built.idle.to_string(),
        walk: built.walk.to_string(),
        run: built.run.to_string(),
        machine: built.machine.to_string(),
        actor,
        mannequin: built.mannequin,
        fit: built.fit.map(|f| {
            format!(
                "fitted to a {:.2} m mesh: {} of {} joints landed inside it, symmetry {:.2}",
                f.height_m, f.joints_inside, f.joints, f.symmetry_score
            )
        }),
        weights: built.weights.map(|w| {
            format!(
                "{} vertices weighted, {} unreached, worst residual {:.4}",
                w.assigned, w.unreached, w.worst_residual
            )
        }),
        warnings: built.warnings,
    })
}

/// Where a generated character's assets land, for the panel's "written to…"
/// line. A command rather than a frontend constant so the folder has one
/// spelling.
#[tauri::command]
pub async fn character_folder() -> Result<String, String> {
    Ok(CHARACTER_FOLDER.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use inf_editor_core::ipc::{BodyParamsDto, GaitParamsDto};

    fn dto(plan: &str) -> CharacterSpecDto {
        CharacterSpecDto {
            name: "Hero".into(),
            plan: plan.into(),
            legs: None,
            params: BodyParamsDto::from_params(&inf_anim::BodyParams::default()),
            gait: GaitParamsDto::from_gait(&inf_anim::locomotion::GaitParams::default()),
            mesh_asset: None,
        }
    }

    #[test]
    fn every_plan_the_panel_offers_parses() {
        assert_eq!(parse_plan("biped", None), Ok(BodyPlan::Biped));
        assert_eq!(parse_plan("quadruped", None), Ok(BodyPlan::Quadruped));
        assert_eq!(parse_plan("hexapod", None), Ok(BodyPlan::Hexapod));
        assert_eq!(
            parse_plan("npedal", Some(8)),
            Ok(BodyPlan::Npedal { legs: 8 })
        );
    }

    /// A plan this build does not know names itself **and lists what it does
    /// know** — the whole reason the field is a string.
    #[test]
    fn an_unknown_plan_names_itself_and_the_alternatives() {
        let err = parse_plan("octopod", None).unwrap_err();
        assert!(err.contains("octopod"), "{err}");
        assert!(err.contains("quadruped"), "{err}");
        // …and `npedal` without a count is refused rather than defaulted to
        // something: a silent 4 would generate the wrong creature.
        assert!(parse_plan("npedal", None).is_err());
    }

    #[test]
    fn a_wire_spec_becomes_the_ring_one_spec_it_describes() {
        let mut d = dto("quadruped");
        d.params.height_m = 0.9;
        d.gait.walk_cadence_hz = 1.4;
        let spec = to_spec(&d).expect("parses");
        assert_eq!(spec.plan, BodyPlan::Quadruped);
        assert_eq!(spec.params.height_m, 0.9);
        assert_eq!(spec.gait.walk_cadence_hz, 1.4);
        assert_eq!(spec.mesh, None);

        // An EMPTY mesh id is "no mesh", not a parse error: an unset picker
        // sends `""` and refusing it would make the common path an error.
        d.mesh_asset = Some(String::new());
        assert_eq!(to_spec(&d).unwrap().mesh, None);
        // A malformed one is still a refusal.
        d.mesh_asset = Some("not-a-guid".into());
        assert!(to_spec(&d).is_err());
    }
}
