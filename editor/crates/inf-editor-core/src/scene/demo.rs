//! The default authored scene — a small primitive+lights setup, exactly the
//! "author a primitive+lights scene" the Phase 3 gate calls for. Loaded into a
//! fresh document so the editor boots with something to select, transform, and
//! save (then round-trip byte-identically).

use glam::DVec3;
use inf_ecs::components::{Material, SkyAtmosphere, TimeOfDay, Transform};
use inf_ecs::math::{Color, Vec3d};

use crate::ipc::SpawnKind;
use crate::scene::SceneDoc;

/// Populate `doc` with the default scene and mark it clean (unsaved-but-pristine).
pub fn build(doc: &mut SceneDoc) {
    doc.set_title("Untitled");

    // Ground.
    let floor = doc.create(SpawnKind::Plane, "Floor", None);
    set_transform(
        doc,
        floor,
        Transform {
            translation: Vec3d::ZERO,
            rotation: Vec3d::ZERO,
            scale: Vec3d::new(20.0, 1.0, 20.0),
        },
    );
    set_color(doc, floor, Color::new(0.30, 0.32, 0.35, 1.0));
    // **And the collider under it** (wave GTA1 audit). `SpawnKind::Plane` inserts
    // a `MeshRef` and nothing physical, and `inf_physics::d3::ecs`'s sync skips
    // an entity with neither a body nor a collider — so this floor reached the
    // solver as nothing, which cost nothing until the pawn below arrived and
    // began falling through it at 9.81 m/s². Measured, one second of Simulate:
    // **4.9868 m of fall** without it. See `samples::ground_slab` for the shape and why
    // the offset is what it is; the plane is a unit quad scaled by 20, so ±10 m.
    set_ground_collider(doc, floor, 10.0);

    // A little cluster of primitives under a "Props" folder.
    let props = doc.create(SpawnKind::Empty, "Props", None);
    let cube = doc.create(SpawnKind::Cube, "Cube", Some(props));
    set_transform(doc, cube, translate(-2.0, 0.5, 0.0));
    set_color(doc, cube, Color::new(0.80, 0.25, 0.22, 1.0));

    let sphere = doc.create(SpawnKind::Sphere, "Sphere", Some(props));
    set_transform(doc, sphere, translate(0.0, 0.6, -1.5));
    set_color(doc, sphere, Color::new(0.25, 0.55, 0.85, 1.0));

    let cyl = doc.create(SpawnKind::Cylinder, "Cylinder", Some(props));
    set_transform(doc, cyl, translate(2.0, 0.75, 0.5));
    set_color(doc, cyl, Color::new(0.30, 0.70, 0.35, 1.0));

    // Lighting.
    let lighting = doc.create(SpawnKind::Empty, "Lighting", None);
    add_sky(doc, lighting);
    let fill = doc.create(SpawnKind::PointLight, "FillLight", Some(lighting));
    set_transform(doc, fill, translate(4.0, 3.0, 4.0));

    // ── the pawn (wave GTA1) ──
    //
    // The editor's boot document had furniture and no player, so the first thing
    // a new user does — press Play — produced an overhead camera above a floor
    // and three primitives, with nothing to move: `camera_subject` returns `None`
    // on a level with no `player_controlled` character, and the player host then
    // keeps its own view.
    //
    // Through the same door the New Character wizard and the island's hero take,
    // naming the committed starter character's asset guids. The editor's own
    // content root is seeded with those seventeen files whenever they are ABSENT
    // (`commands::assets::seed_starter_content`) rather than only on a fresh
    // root, so the references resolve to real bytes rather than to a placeholder
    // cube — on an editor that has been run before this wave too.
    let ids = crate::samples::starter_character_ids();
    let asset = |id: Option<inf_asset::AssetId>| id.expect("every starter id is fixed").0;
    doc.edit_create_character_with_guid(
        DEMO_PLAYER_GUID,
        crate::samples::STARTER_CHARACTER_NAME,
        asset(ids.skeleton),
        asset(ids.mesh),
        asset(ids.machine),
        Some(crate::scene::doc::CharacterSkin::from_material(
            asset(ids.material),
            &crate::character::starter_skin_material(),
        )),
        DVec3::new(0.0, 0.0, 3.0),
        Some(asset(ids.actor)),
        crate::samples::starter_character_spec().params.height_m,
    );

    // **A boot document has nothing to undo.** Every other spawn here goes
    // through the non-recording `create`; the character's door records, because
    // it is the same door an author's "Create Character" takes. Clearing the
    // history is what keeps "press Ctrl+Z on a fresh editor" from deleting the
    // player it just showed you — and keeps `undo_depth` at 0, which is what a
    // fresh document means everywhere else in this crate.
    doc.clear_history();
    doc.world_mut().propagate();
    doc.mark_saved();
}

/// The boot document's player entity. Fixed rather than minted so a level saved
/// out of a fresh document twice names the same character both times.
pub const DEMO_PLAYER_GUID: uuid::Uuid = uuid::Uuid::from_u128(0x0DE0_0001);

/// Give the default scene the **sky authority** — the `TimeOfDay` +
/// `SkyAtmosphere` pair that makes a new level's sky the physically-based one
/// (P17.2; the Phase-17 "done when"). Existing levels are untouched: this runs
/// only when a fresh document is built.
///
/// ## Why this REPLACED the demo's directional light rather than joining it
///
/// `project_sky` pushes the sun as a directional light in **both** hosts, so a
/// level carrying a `TimeOfDay` *and* an authored `Sun` would be lit twice, from
/// two unrelated directions, and would cast two sets of shadows. The time of day
/// is the sun now; the old `DirectionalLight` at `(-50°, -30°, 0°)` has no job
/// left. (A level that would rather author its own suns still can — that is what
/// `SkyAtmosphere::enabled = false` is for.)
///
/// ## Why the actor is called `Sky` and lives here
///
/// `SceneDoc::edit_time_of_day` — what World Settings calls when a *clockless*
/// level opts in — creates an actor with exactly this name and these two
/// components. Matching it means a user who learns the sky on a default level
/// recognises it on any other, and means `edit_time_of_day` finds this one
/// instead of conjuring a second (it resolves the authority by lowest `Guid`, so
/// there is never a second clock).
///
/// ## Why the defaults are the *component* defaults, untouched
///
/// "What a new level gets" and "what these components mean" being the same thing
/// is worth more than a tuned demo — there is then nothing to explain twice, and
/// the golden that pins this look (`inf-render`'s `sky_noon`/`editor_default`) is
/// pinning the documented defaults rather than a private tweak. They are good
/// defaults on their own terms:
///
/// * **10:00 UTC, day 172, 48.9° N** puts the sun ≈ 55° up — high enough for a
///   saturated blue zenith and a bright, clean key, low enough to keep a
///   *direction*. A noon sun lights everything from straight overhead and
///   flattens exactly the geometry the default scene exists to show.
/// * **`rate = 0`** freezes the clock. An idle editor must never move the sun,
///   because moving it would dirty the document under a user who touched nothing.
/// * **no height fog** (`fog_density = 0`). At this scene's 20 m scale any
///   density that would read at a distance is invisible here, so a nonzero
///   default would be a knob that appears to do nothing. Aerial perspective is
///   already on, already physical, and already correct at every scale.
///
/// ## The one place this scene departs from the component defaults (P17.3)
///
/// `SkyAtmosphere::clouds_enabled` defaults to **false** and this scene sets it
/// **true**. The two are not in tension, they are answering different questions.
/// The *component* default has to be false because it is also what every existing
/// v12 level lifts to, and a level must not grow a sky it was never authored
/// against — a rule the schema bump depends on. The *new-level* default is what
/// an author should see the first time they open the editor, and an empty blue
/// dome is not it. Everything else in the cloud block stays at its documented
/// default (0.35 coverage — broken cumulus with real gaps — a 1.5–4 km layer, a
/// 6 m/s westerly), so this is one boolean of divergence, not a private tuning.
/// It is the one reason `editor_default` was re-blessed in P17.3.
fn add_sky(doc: &mut SceneDoc, parent: uuid::Uuid) {
    let sky = doc.create(SpawnKind::Empty, "Sky", Some(parent));
    let Some(e) = doc.entity_of(sky) else {
        debug_assert!(false, "the Sky actor was just created");
        return;
    };
    let world = doc.world_mut();
    let mut entity = world.world_mut().entity_mut(e);
    entity.insert(TimeOfDay::default());
    entity.insert(SkyAtmosphere {
        clouds_enabled: true,
        ..SkyAtmosphere::default()
    });
    world.mark_dirty();
}

fn translate(x: f64, y: f64, z: f64) -> Transform {
    Transform::from_translation(DVec3::new(x, y, z))
}

/// Put [`crate::samples::ground_slab`] under an already-created ground plane.
fn set_ground_collider(doc: &mut SceneDoc, guid: uuid::Uuid, half_xz: f64) {
    if let Some(e) = doc.entity_of(guid) {
        doc.world_mut()
            .world_mut()
            .entity_mut(e)
            .insert(crate::samples::ground_slab(half_xz));
        doc.world_mut().mark_dirty();
    }
}

fn set_transform(doc: &mut SceneDoc, guid: uuid::Uuid, t: Transform) {
    if let Some(e) = doc.entity_of(guid) {
        doc.world_mut().world_mut().entity_mut(e).insert(t);
        doc.world_mut().mark_dirty();
    }
}

fn set_color(doc: &mut SceneDoc, guid: uuid::Uuid, color: Color) {
    if let Some(e) = doc.entity_of(guid) {
        doc.world_mut().world_mut().entity_mut(e).insert(Material {
            base_color: color,
            ..Default::default()
        });
    }
}
