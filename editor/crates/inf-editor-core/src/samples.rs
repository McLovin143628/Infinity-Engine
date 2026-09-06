//! Committed sample content (P8.4): the **2D platformer** the Phase-8 gate plays
//! in-viewport via the interpreter.
//!
//! The sample is defined by the generators here (the source of truth) and
//! committed under `samples/platformer-2d/` as the exact bytes they produce:
//!
//! * `Platformer.inf_lvl` (+ `.toml` sidecar) — a tilemap ground strip, a static
//!   ground collider that ends in a **ledge**, a floating platform, a 2D light,
//!   and the **player** (Sprite + kinematic `RigidBody2D` + capsule `Collider2D`
//!   + `CharacterController2D`).
//! * `Coyote.inf_act` (+ `.json`) — the player's [`BlueprintClass`]: the
//!   **coyote-time jump** handler. It is authored directly as [`BlueprintFn`] IR
//!   (the on-disk `.inf_act` stores the lowered IR, not the visual graph) —
//!   richer than a hand-wired graph would be, and the same IR the interpreter and
//!   the transpiler consume.
//!
//! The coyote-time rule: horizontal move from left/right input, gravity applied
//! to a `vy` velocity var, and a jump that is allowed while `is_grounded` **or**
//! within a short **coyote window** — a `coyote` float reset when grounded and
//! decremented every tick — so a jump pressed a few frames *after* walking off a
//! ledge still fires.

use std::path::PathBuf;

use glam::{DVec2, DVec3};
use uuid::Uuid;

use inf_blueprint::{
    BinOp, Binding, BlueprintClass, BlueprintFn, EventBinding, EventKind, Expr, Lit, LocalId,
    Param, Stmt, Ty, Variable,
};
use inf_ecs::components::{
    ActorClass, BillboardMode, BodyKind2D, CharacterController2D, Collider2D, ColliderShape2DKind,
    Light2D, RigidBody2D, Sprite, Tilemap, Transform,
};
use inf_ecs::math::{Color, Vec2d};

use crate::ipc::SpawnKind;
use crate::scene::SceneDoc;

// ── Stable identities (so the committed sample is byte-reproducible) ──────────
pub const GROUND_GUID: Uuid = Uuid::from_u128(0x8401_0001);
pub const GROUND_TILES_GUID: Uuid = Uuid::from_u128(0x8401_0002);
pub const PLATFORM_GUID: Uuid = Uuid::from_u128(0x8401_0003);
pub const PLAYER_GUID: Uuid = Uuid::from_u128(0x8401_0004);
pub const LIGHT_GUID: Uuid = Uuid::from_u128(0x8401_0005);
/// The fixed level GUID stamped into the committed sidecar.
pub const LEVEL_GUID: Uuid = Uuid::from_u128(0x8401_0000);
/// The coyote actor class id.
pub const COYOTE_CLASS_ID: &str = "act:coyote_player";
/// The **asset** GUID of the committed `Coyote.inf_act` (its inf_asset sidecar,
/// P9.5). Stable so the level's persisted `actor` binding (the [`ActorClass`] on
/// the player) resolves to this blueprint through the AssetDb / cooked pack.
pub const COYOTE_ASSET_GUID: Uuid = Uuid::from_u128(0x8401_00AC);

// ── Coyote-time tuning (world units, seconds) ────────────────────────────────
/// Downward acceleration applied to `vy` each tick.
pub const GRAVITY: f64 = 30.0;
/// Upward velocity a jump imparts.
pub const JUMP_SPEED: f64 = 9.0;
/// Horizontal run speed.
pub const MOVE_SPEED: f64 = 5.0;
/// How long after leaving the ground a jump is still allowed (the coyote window).
pub const COYOTE_TIME: f64 = 0.12;

// ── IR builders (keep the handler readable) ──────────────────────────────────
fn str_lit(v: &str) -> Expr {
    Expr::Lit(Lit::Str(v.to_string()))
}
fn float_lit(v: f64) -> Expr {
    Expr::Lit(Lit::Float(v))
}
fn local(id: u32) -> Expr {
    Expr::Local(LocalId(id))
}
fn call(path: &[&str], args: Vec<Expr>) -> Expr {
    Expr::Call {
        path: path.iter().map(|s| s.to_string()).collect(),
        args,
    }
}
fn get_var(name: &str) -> Expr {
    call(&["vars", "get"], vec![str_lit(name)])
}
fn set_var(name: &str, value: Expr) -> Stmt {
    Stmt::ExprStmt(call(&["vars", "set"], vec![str_lit(name), value]))
}
fn bin(op: BinOp, a: Expr, b: Expr) -> Expr {
    Expr::Binary(op, Box::new(a), Box::new(b))
}
fn let_named(id: u32, name: &str, mutable: bool, value: Expr) -> Stmt {
    Stmt::Let {
        id: LocalId(id),
        binding: Binding::Named(name.to_string()),
        ty: None,
        mutable,
        value,
    }
}
fn if_then(cond: Expr, then_body: Vec<Stmt>, else_body: Vec<Stmt>) -> Stmt {
    Stmt::If {
        cond,
        then_body,
        else_body,
    }
}

/// The coyote-time **Tick** handler as `BlueprintFn` IR. `dt` is the fixed step.
///
/// Locals: `n1=entity`, `n2=grounded`, `n3=vx`, `n4=grounded_after`.
fn coyote_tick_fn() -> BlueprintFn {
    let e = || local(1);
    let dt = || Expr::Param("dt".to_string());
    let grounded = || local(2);

    let body = vec![
        // let entity = vars::get("entity")   (seeded by the Simulate session)
        let_named(1, "entity", false, get_var("entity")),
        // let grounded = physics2d::is_grounded(entity)
        let_named(
            2,
            "grounded",
            false,
            call(&["physics2d", "is_grounded"], vec![e()]),
        ),
        // coyote timer: reset to COYOTE_TIME when grounded, else count down by dt.
        if_then(
            grounded(),
            vec![set_var("coyote", float_lit(COYOTE_TIME))],
            vec![set_var("coyote", bin(BinOp::Sub, get_var("coyote"), dt()))],
        ),
        // gravity: vy -= GRAVITY * dt
        set_var(
            "vy",
            bin(
                BinOp::Sub,
                get_var("vy"),
                bin(BinOp::Mul, float_lit(GRAVITY), dt()),
            ),
        ),
        // jump: if just_pressed("jump") && (grounded || coyote > 0) { vy = JUMP; coyote = 0 }
        if_then(
            bin(
                BinOp::And,
                call(&["input", "just_pressed"], vec![str_lit("jump")]),
                bin(
                    BinOp::Or,
                    grounded(),
                    bin(BinOp::Gt, get_var("coyote"), float_lit(0.0)),
                ),
            ),
            vec![
                set_var("vy", float_lit(JUMP_SPEED)),
                set_var("coyote", float_lit(0.0)),
            ],
            vec![],
        ),
        // horizontal: vx from left/right held state.
        let_named(3, "vx", true, float_lit(0.0)),
        if_then(
            call(&["input", "is_down"], vec![str_lit("right")]),
            vec![Stmt::Assign {
                target: LocalId(3),
                value: bin(BinOp::Add, local(3), float_lit(MOVE_SPEED)),
            }],
            vec![],
        ),
        if_then(
            call(&["input", "is_down"], vec![str_lit("left")]),
            vec![Stmt::Assign {
                target: LocalId(3),
                value: bin(BinOp::Sub, local(3), float_lit(MOVE_SPEED)),
            }],
            vec![],
        ),
        // move_and_slide(entity, vx*dt, vy*dt) → grounded_after
        let_named(
            4,
            "grounded_after",
            false,
            call(
                &["physics2d", "move_and_slide"],
                vec![
                    e(),
                    bin(BinOp::Mul, local(3), dt()),
                    bin(BinOp::Mul, get_var("vy"), dt()),
                ],
            ),
        ),
        // Landed while moving down → cancel the downward velocity.
        if_then(
            bin(
                BinOp::And,
                local(4),
                bin(BinOp::Lt, get_var("vy"), float_lit(0.0)),
            ),
            vec![set_var("vy", float_lit(0.0))],
            vec![],
        ),
    ];

    BlueprintFn {
        id: EventKind::Tick.key(),
        name: EventKind::Tick.key(),
        params: vec![Param {
            name: "dt".to_string(),
            ty: Ty::Float,
        }],
        ret: Ty::Unit,
        body,
    }
}

/// A `BeginPlay` handler that zeroes the motion state (so a re-enter is clean).
fn coyote_begin_play_fn() -> BlueprintFn {
    BlueprintFn {
        id: EventKind::BeginPlay.key(),
        name: EventKind::BeginPlay.key(),
        params: vec![],
        ret: Ty::Unit,
        body: vec![
            set_var("vy", float_lit(0.0)),
            set_var("coyote", float_lit(0.0)),
        ],
    }
}

/// The player's coyote-time [`BlueprintClass`] (the `.inf_act`).
pub fn coyote_class() -> BlueprintClass {
    let mut class = BlueprintClass::new(COYOTE_CLASS_ID, "Coyote Player");
    class.variables = vec![
        // `entity` is the opaque blueprint id; the Simulate session seeds it.
        Variable {
            name: "entity".into(),
            ty: Ty::Int,
            default: Lit::Int(0),
            exposed: false,
        },
        Variable {
            name: "vy".into(),
            ty: Ty::Float,
            default: Lit::Float(0.0),
            exposed: true,
        },
        Variable {
            name: "coyote".into(),
            ty: Ty::Float,
            default: Lit::Float(0.0),
            exposed: true,
        },
    ];
    class.events = vec![
        EventBinding {
            event: EventKind::BeginPlay,
            body: coyote_begin_play_fn(),
        },
        EventBinding {
            event: EventKind::Tick,
            body: coyote_tick_fn(),
        },
    ];
    class
}

// ── The scene ────────────────────────────────────────────────────────────────

/// Insert a bundle onto `guid`'s entity (dirties the doc), mirroring the pattern
/// the scene serialize tests use — this crate doesn't name `bevy_ecs::Bundle`.
macro_rules! insert {
    ($doc:expr, $guid:expr, $comp:expr) => {{
        if let Some(e) = $doc.entity_of($guid) {
            $doc.world_mut().world_mut().entity_mut(e).insert($comp);
            $doc.world_mut().mark_dirty();
        }
    }};
}

/// Build the committed platformer [`SceneDoc`].
///
/// Layout (side view, +Y up): a static ground box spanning world x ∈ [-3, 3] with
/// its top at y = 0, a painted tilemap strip beneath it, a floating platform to
/// the right, a soft 2D light, and the player standing on the ground at x = 1.5
/// — so running **right** walks off the ledge at x = 3.
pub fn platformer_scene() -> SceneDoc {
    let mut doc = SceneDoc::new();
    doc.set_title("Platformer 2D");

    // Ground collider (static box): centre (0,-0.5), half (3.0,0.5) → top y=0.
    doc.create_with_guid(GROUND_GUID, SpawnKind::Empty, "Ground", None);
    insert!(
        doc,
        GROUND_GUID,
        Transform::from_translation(DVec3::new(0.0, -0.5, 0.0))
    );
    insert!(
        doc,
        GROUND_GUID,
        RigidBody2D {
            kind: BodyKind2D::Static,
            ..Default::default()
        }
    );
    insert!(
        doc,
        GROUND_GUID,
        Collider2D {
            shape_kind: ColliderShape2DKind::Box,
            half_extents: Vec2d::new(3.0, 0.5),
            ..Default::default()
        }
    );

    // Visual tilemap ground strip (painted via the chunk API) under the collider.
    doc.create_with_guid(GROUND_TILES_GUID, SpawnKind::Empty, "Ground Tiles", None);
    insert!(
        doc,
        GROUND_TILES_GUID,
        Transform::from_translation(DVec3::new(-3.0, -1.0, 0.0))
    );
    let mut tm = Tilemap {
        tile_size: Vec2d::new(1.0, 1.0),
        atlas_cols: 1,
        atlas_rows: 1,
        tint: Color::new(0.35, 0.30, 0.25, 1.0),
        ..Default::default()
    };
    // A 6-wide, 1-tall row of ground tiles beneath the collider.
    for gx in 0..6i32 {
        tm.set_tile(gx, 0, 1);
    }
    insert!(doc, GROUND_TILES_GUID, tm);

    // Floating platform (static box) up and to the right.
    doc.create_with_guid(PLATFORM_GUID, SpawnKind::Empty, "Platform", None);
    insert!(
        doc,
        PLATFORM_GUID,
        Transform::from_translation(DVec3::new(5.0, 1.5, 0.0))
    );
    insert!(
        doc,
        PLATFORM_GUID,
        RigidBody2D {
            kind: BodyKind2D::Static,
            ..Default::default()
        }
    );
    insert!(
        doc,
        PLATFORM_GUID,
        Collider2D {
            shape_kind: ColliderShape2DKind::Box,
            half_extents: Vec2d::new(1.25, 0.25),
            ..Default::default()
        }
    );

    // A soft 2D light so the sprites read.
    doc.create_with_guid(LIGHT_GUID, SpawnKind::Empty, "Sun 2D", None);
    insert!(
        doc,
        LIGHT_GUID,
        Transform::from_translation(DVec3::new(0.0, 3.0, 0.0))
    );
    insert!(
        doc,
        LIGHT_GUID,
        Light2D {
            color: Color::new(1.0, 0.95, 0.8, 1.0),
            intensity: 1.2,
            radius: 12.0,
        }
    );

    // The player: sprite + kinematic body + capsule + character controller.
    doc.create_with_guid(PLAYER_GUID, SpawnKind::Empty, "Player", None);
    insert!(
        doc,
        PLAYER_GUID,
        Transform::from_translation(DVec3::new(1.5, 0.8, 0.0))
    );
    insert!(
        doc,
        PLAYER_GUID,
        Sprite {
            size: Vec2d::new(0.8, 1.2),
            color: Color::new(0.9, 0.4, 0.3, 1.0),
            billboard: BillboardMode::None,
            ..Default::default()
        }
    );
    insert!(
        doc,
        PLAYER_GUID,
        RigidBody2D {
            kind: BodyKind2D::Kinematic,
            fixed_rotation: true,
            ..Default::default()
        }
    );
    insert!(
        doc,
        PLAYER_GUID,
        Collider2D {
            shape_kind: ColliderShape2DKind::Capsule,
            half_extents: Vec2d::new(0.3, 0.35),
            radius: 0.3,
            ..Default::default()
        }
    );
    insert!(
        doc,
        PLAYER_GUID,
        CharacterController2D {
            max_slope_deg: 46.0,
            snap_to_ground: 0.3,
            offset: 0.02,
        }
    );
    // Bind the player to the Coyote blueprint class (P9.5 persisted actor link):
    // the level now carries its own gameplay binding — no CC2D heuristic needed.
    insert!(doc, PLAYER_GUID, ActorClass(COYOTE_ASSET_GUID));

    // Level settings: the platformer keeps the character-self-gravity convention
    // (2D world gravity ZERO) at 60 Hz — i.e. the schema-v3 defaults, now made
    // explicit + persisted instead of the player's old hard-coded constants.
    doc.set_settings(crate::scene::serialize::LevelSettings::default());

    doc.world_mut().propagate();
    doc.mark_saved();
    doc
}

/// The `(guid, class)` actor list for [`crate::simulate::SimSession::enter`].
pub fn platformer_actors() -> Vec<(Uuid, BlueprintClass)> {
    vec![(PLAYER_GUID, coyote_class())]
}

/// Resolve the in-editor Simulate actor list, **preferring the level's persisted
/// [`ActorClass`] bindings** (P9.5): each entity carrying an `ActorClass` is run
/// with the blueprint class its asset GUID resolves to (via `resolve`, which the
/// caller backs with the project's asset DB). Falls back to the legacy
/// [`character_actors`] heuristic when the scene carries **no** bindings at all
/// (kept for scenes authored before v3). Guid order.
pub fn bound_actors<F>(doc: &SceneDoc, mut resolve: F) -> Vec<(Uuid, BlueprintClass)>
where
    F: FnMut(Uuid) -> Option<BlueprintClass>,
{
    let mut out = Vec::new();
    let mut any_binding = false;
    let w = doc.world();
    for &guid in doc.order() {
        if let Some(e) = w.entity_of(guid) {
            if let Some(ac) = w.world().get::<ActorClass>(e) {
                any_binding = true;
                if let Some(class) = resolve(ac.0) {
                    out.push((guid, class));
                }
            }
        }
    }
    if any_binding && !out.is_empty() {
        out
    } else {
        // No bindings (or none resolved) → legacy CC2D heuristic.
        character_actors(doc)
    }
}

/// Discover controllable actors in an arbitrary scene for Simulate: every entity
/// carrying a `CharacterController2D` gets the coyote-time class (the legacy
/// pre-v3 heuristic, kept as the fallback for [`bound_actors`]). Guid order.
pub fn character_actors(doc: &SceneDoc) -> Vec<(Uuid, BlueprintClass)> {
    let mut out = Vec::new();
    let w = doc.world();
    for &guid in doc.order() {
        if let Some(e) = w.entity_of(guid) {
            if w.world().get::<CharacterController2D>(e).is_some() {
                out.push((guid, coyote_class()));
            }
        }
    }
    out
}

/// **The collider under a starter template's ground plane** (wave GTA1 audit),
/// `half_xz` metres from the origin in X and Z.
///
/// # The defect this closes
///
/// Every 3D starter template's ground was a `MeshRef` and a `Material` and
/// nothing else. `inf_physics::d3::ecs`'s sync walks the entities that carry a
/// body **or** a collider and `continue`s on the rest, so those planes reached
/// the solver as nothing at all — which cost nothing while the levels held only
/// furniture, and became the whole story the moment wave GTA1 put a
/// *gravity-driven pawn* on them: `CharacterMovement::gravity_mps2` is 9.81, and
/// a character the controller does not report grounded integrates it to a 53 m/s
/// terminal velocity. Measured, one second of Simulate at 60 Hz, with no
/// collider under the plane: **4.9868 m of fall on all four documents, still
/// accelerating**; with this slab under it, **−0.0201 m** (the controller's own
/// ground snap, upward). The template README this wave wrote — *"press Play and
/// WASD moves the Player"* — described a body in free fall.
///
/// A collider with **no** `RigidBody3D`, which is the documented way to say
/// "static world" ([`Collider3D`](inf_ecs::components::Collider3D) attaches to an
/// implicit static body), and the offset puts the slab's **top face exactly on
/// the visual plane**: `inf_render::primitives::plane_geometry` is a unit quad
/// spanning ±0.5, so a ground scaled by 20 is ±10 m at `y = 0`.
pub(crate) fn ground_slab(half_xz: f64) -> inf_ecs::components::Collider3D {
    inf_ecs::components::Collider3D {
        shape_kind: inf_ecs::components::ColliderShape3DKind::Box,
        half_extents: inf_ecs::math::Vec3d::new(half_xz, 0.5, half_xz),
        offset: inf_ecs::math::Vec3d::new(0.0, -0.5, 0.0),
        ..Default::default()
    }
}

// ── Hybrid 2.5D template scene (P8.4b) ───────────────────────────────────────
pub const HYBRID_GROUND_GUID: Uuid = Uuid::from_u128(0x8402_0001);
pub const HYBRID_SUN_GUID: Uuid = Uuid::from_u128(0x8402_0002);
pub const HYBRID_LIGHT2D_GUID: Uuid = Uuid::from_u128(0x8402_0003);
pub const HYBRID_SPRITE_SPHERE_GUID: Uuid = Uuid::from_u128(0x8402_0004);
pub const HYBRID_SPRITE_CYL_GUID: Uuid = Uuid::from_u128(0x8402_0005);
/// The starter character this template spawns (wave GTA1).
pub const HYBRID_PLAYER_GUID: Uuid = Uuid::from_u128(0x8402_0006);
pub const HYBRID_LEVEL_GUID: Uuid = Uuid::from_u128(0x8402_0000);

// ── First-person template GUIDs (0x8406 block) ──
pub const FP_LEVEL_GUID: Uuid = Uuid::from_u128(0x8406_0000);
pub const FP_GROUND_GUID: Uuid = Uuid::from_u128(0x8406_0001);
pub const FP_SUN_GUID: Uuid = Uuid::from_u128(0x8406_0002);
pub const FP_PLAYER_GUID: Uuid = Uuid::from_u128(0x8406_0003);
pub const FP_CAMERA_GUID: Uuid = Uuid::from_u128(0x8406_0004);

/// The minimal **hybrid 2.5D** starter scene: a 3D ground **plane mesh**, a
/// directional sun, a soft 2D light, and two **billboarded** sprites standing on
/// the plane (one spherical, one cylindrical) — the 2.5D idiom (2D cards in a 3D
/// world). Scaffolded by `inf new --template hybrid-2.5d`.
pub fn hybrid_scene() -> SceneDoc {
    use inf_ecs::components::{Light, LightKind, Material, MeshRef, Primitive};

    let mut doc = SceneDoc::new();
    doc.set_title("Hybrid 2.5D");

    // **THE LIT STACK, AUTHORED** (wave CERT1, CP-A1). A level that authors no
    // render block ships with shadows, GI, VSM, TAA, SSAO and bloom ALL OFF —
    // `fps_instrument.rs` has documented that about this engine for four waves,
    // and it meant the first thing anyone saw of it was unlit. A starter
    // template that plays unlit is not a certified starter.
    doc.set_settings(crate::scene::serialize::LevelSettings {
        render: crate::scene::serialize::RenderSettingsRecord::lit_showcase(),
        ..crate::scene::serialize::LevelSettings::default()
    });

    // 3D ground plane.
    doc.create_with_guid(HYBRID_GROUND_GUID, SpawnKind::Empty, "Ground", None);
    insert!(
        doc,
        HYBRID_GROUND_GUID,
        Transform {
            translation: inf_ecs::math::Vec3d::ZERO,
            rotation: inf_ecs::math::Vec3d::ZERO,
            scale: inf_ecs::math::Vec3d::new(20.0, 1.0, 20.0),
        }
    );
    insert!(
        doc,
        HYBRID_GROUND_GUID,
        MeshRef {
            primitive: Primitive::Plane,
            asset: None,
        }
    );
    insert!(
        doc,
        HYBRID_GROUND_GUID,
        Material {
            base_color: Color::new(0.30, 0.34, 0.30, 1.0),
            ..Default::default()
        }
    );
    insert!(doc, HYBRID_GROUND_GUID, ground_slab(10.0));

    // A directional sun (3D lighting).
    doc.create_with_guid(HYBRID_SUN_GUID, SpawnKind::Empty, "Sun", None);
    insert!(
        doc,
        HYBRID_SUN_GUID,
        Light {
            kind: LightKind::Directional,
            color: Color::WHITE,
            intensity: 2.0,
            ..Default::default()
        }
    );

    // A soft 2D light for the sprites.
    doc.create_with_guid(HYBRID_LIGHT2D_GUID, SpawnKind::Empty, "Fill 2D", None);
    insert!(
        doc,
        HYBRID_LIGHT2D_GUID,
        Transform::from_translation(DVec3::new(0.0, 2.0, 0.0))
    );
    insert!(
        doc,
        HYBRID_LIGHT2D_GUID,
        Light2D {
            color: Color::new(0.9, 0.9, 1.0, 1.0),
            intensity: 0.8,
            radius: 10.0,
        }
    );

    // Two billboarded sprites standing on the plane.
    for (guid, name, x, mode, tint) in [
        (
            HYBRID_SPRITE_SPHERE_GUID,
            "Billboard (Spherical)",
            -1.5,
            BillboardMode::Spherical,
            Color::new(0.9, 0.5, 0.3, 1.0),
        ),
        (
            HYBRID_SPRITE_CYL_GUID,
            "Tree (Cylindrical)",
            1.5,
            BillboardMode::Cylindrical,
            Color::new(0.4, 0.8, 0.4, 1.0),
        ),
    ] {
        doc.create_with_guid(guid, SpawnKind::Empty, name, None);
        insert!(
            doc,
            guid,
            Transform::from_translation(DVec3::new(x, 1.0, 0.0))
        );
        insert!(
            doc,
            guid,
            Sprite {
                size: Vec2d::new(1.4, 2.0),
                color: tint,
                billboard: mode,
                ..Default::default()
            }
        );
    }

    // The pawn (wave GTA1) — the same starter character the other two 3D
    // templates spawn, and this template scaffolds the same seventeen files
    // (`ProjectTemplate::Hybrid25d` takes `STARTER_CHARACTER`). Set back from the
    // billboards so it is looking at them rather than standing in one.
    let ids = starter_character_ids();
    let asset = |id: Option<inf_asset::AssetId>| id.expect("every starter id is fixed").0;
    doc.edit_create_character_with_guid(
        HYBRID_PLAYER_GUID,
        STARTER_CHARACTER_NAME,
        asset(ids.skeleton),
        asset(ids.mesh),
        asset(ids.machine),
        DVec3::new(0.0, 0.0, 4.0),
        Some(asset(ids.actor)),
        starter_character_spec().params.height_m,
    );

    doc.world_mut().propagate();
    doc.mark_saved();
    doc
}

/// The repo-root `templates/hybrid-2.5d/` directory (committed starter scene).
pub fn hybrid_template_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../templates/hybrid-2.5d")
}

/// Write the committed hybrid-2.5D template scene from [`hybrid_scene`].
pub fn write_hybrid_template() -> Result<(), String> {
    let dir = hybrid_template_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("mkdir: {e}"))?;
    let doc = hybrid_scene();
    crate::scene::serialize::save(&doc, &dir.join("Hybrid.inf_lvl"), Some(HYBRID_LEVEL_GUID))?;
    std::fs::write(dir.join("README.md"), HYBRID_README).map_err(|e| format!("write: {e}"))?;
    Ok(())
}

const HYBRID_README: &str = "# Hybrid 2.5D template\n\n\
The starter scene `inf new --template hybrid-2.5d` scaffolds: a 3D ground plane,\n\
a directional sun, a soft 2D light, and two **billboarded** sprites (spherical +\n\
cylindrical) — 2D cards standing in a 3D world.\n\n\
Generated by `inf_editor_core::samples::hybrid_scene`. Regenerate with\n\
`INF_BLESS_SAMPLES=1 cargo test -p inf-editor-core samples`.\n";

/// The minimal **first-person** starter scene: a 3D ground plane, a directional
/// sun, a kinematic **Player** capsule that is a real pawn, and a first-person
/// **Camera** at eye height. Scaffolded by `inf new --template first-person`.
///
/// # The mismatch this used to carry (wave GTA1)
///
/// The Player had a `CharacterController3D` — the *physics* half — and nothing
/// that reads input, because `CharacterMovement` is what
/// `inf_ecs::movement::{movement_targets, camera_subject}` key on. So the
/// template that exists to say "here is your player" shipped a level with no
/// player-controlled character in it: Play fell back to the overhead camera and
/// WASD moved nothing. The smallest true fix is the missing component, sized to
/// the capsule that was already there; P29's movement step does the rest, and
/// no Blueprint is needed to walk.
///
/// **What is still honest to say it lacks**: `ViewMode` is camera-side only and
/// never crosses the sim wire (`inf_ecs::camera`'s ruling 4), so nothing in a
/// level can ask for a first-person camera — the locomotion camera boots in its
/// third-person default on every host. A per-level view mode is the carried
/// follow-up, and until it exists this template is first-person in its intent
/// and its `Camera` entity rather than in what Play shows.
pub fn firstperson_scene() -> SceneDoc {
    use inf_ecs::components::{
        BodyKind3D, Camera, CharacterController3D, CharacterMovement, Collider3D,
        ColliderShape3DKind, Light, LightKind, Material, MeshRef, Primitive, RigidBody3D,
    };
    use inf_ecs::math::Vec3d;

    let mut doc = SceneDoc::new();
    doc.set_title("First Person");

    // **THE LIT STACK, AUTHORED** (wave CERT1, CP-A1). A level that authors no
    // render block ships with shadows, GI, VSM, TAA, SSAO and bloom ALL OFF —
    // `fps_instrument.rs` has documented that about this engine for four waves,
    // and it meant the first thing anyone saw of it was unlit. A starter
    // template that plays unlit is not a certified starter.
    doc.set_settings(crate::scene::serialize::LevelSettings {
        render: crate::scene::serialize::RenderSettingsRecord::lit_showcase(),
        ..crate::scene::serialize::LevelSettings::default()
    });

    // 3D ground plane.
    doc.create_with_guid(FP_GROUND_GUID, SpawnKind::Empty, "Ground", None);
    insert!(
        doc,
        FP_GROUND_GUID,
        Transform {
            translation: Vec3d::ZERO,
            rotation: Vec3d::ZERO,
            scale: Vec3d::new(40.0, 1.0, 40.0),
        }
    );
    insert!(
        doc,
        FP_GROUND_GUID,
        MeshRef {
            primitive: Primitive::Plane,
            asset: None,
        }
    );
    insert!(
        doc,
        FP_GROUND_GUID,
        Material {
            base_color: Color::new(0.28, 0.30, 0.34, 1.0),
            ..Default::default()
        }
    );
    insert!(doc, FP_GROUND_GUID, ground_slab(20.0));

    // A directional sun.
    doc.create_with_guid(FP_SUN_GUID, SpawnKind::Empty, "Sun", None);
    insert!(
        doc,
        FP_SUN_GUID,
        Transform {
            translation: Vec3d::new(0.0, 30.0, 0.0),
            rotation: Vec3d::new(-50.0, -30.0, 0.0),
            scale: Vec3d::new(1.0, 1.0, 1.0),
        }
    );
    insert!(
        doc,
        FP_SUN_GUID,
        Light {
            kind: LightKind::Directional,
            color: Color::new(1.0, 0.97, 0.9, 1.0),
            intensity: 2.5,
            ..Default::default()
        }
    );

    // The player: a kinematic capsule with a 3D character controller.
    //
    // **y = 1.2, not 1.0** (wave GTA1 audit). A character's transform is its
    // capsule CENTRE and its feet are `inf_ecs::movement::feet_offset_m` below —
    // `half_extents.y + radius`, so 0.9 + 0.3 = 1.2 for this capsule. Authored
    // at 1.0 the feet sat at −0.2 m, which cost nothing while the capsule was
    // furniture and is 20 cm inside the ground the moment it became a pawn with
    // a collider under it. `edit_create_character_with_guid` applies exactly this
    // offset for the other three templates; this one is hand-authored, so it says
    // so here.
    doc.create_with_guid(FP_PLAYER_GUID, SpawnKind::Empty, "Player", None);
    insert!(
        doc,
        FP_PLAYER_GUID,
        Transform::from_translation(DVec3::new(0.0, 1.2, 0.0))
    );
    insert!(
        doc,
        FP_PLAYER_GUID,
        RigidBody3D {
            kind: BodyKind3D::Kinematic,
            ..Default::default()
        }
    );
    insert!(
        doc,
        FP_PLAYER_GUID,
        Collider3D {
            shape_kind: ColliderShape3DKind::Capsule,
            half_extents: Vec3d::new(0.3, 0.9, 0.3),
            radius: 0.3,
            ..Default::default()
        }
    );
    insert!(doc, FP_PLAYER_GUID, CharacterController3D::default());
    // **The component that makes the capsule a PAWN** (wave GTA1).
    //
    // The scene had a `CharacterController3D` and a capsule and nothing that
    // reads input: `inf_ecs::movement::movement_targets` and `camera_subject`
    // both key on `CharacterMovement { player_controlled }`, so this level
    // pressed Play into an overhead camera over an object nothing could move.
    // The controller was the *physics* half all along; this is the other one.
    //
    // The half-heights are the capsule's own, not the component defaults, so the
    // body the physics moves and the body the movement step thinks it is
    // crouching are the same body: `half_extents.y` is 0.9 and the radius 0.3.
    // Nothing here is a skeleton — a first-person template has no visible body,
    // and a pawn does not need one.
    insert!(
        doc,
        FP_PLAYER_GUID,
        CharacterMovement {
            player_controlled: true,
            stand_half_height_m: 0.9,
            crouch_half_height_m: 0.45,
            prone_half_height_m: 0.18,
            ..Default::default()
        }
    );

    // A first-person camera at eye height.
    doc.create_with_guid(
        FP_CAMERA_GUID,
        SpawnKind::Empty,
        "First-Person Camera",
        None,
    );
    insert!(
        doc,
        FP_CAMERA_GUID,
        Transform::from_translation(DVec3::new(0.0, 1.7, 0.0))
    );
    insert!(doc, FP_CAMERA_GUID, Camera::default());

    doc.world_mut().propagate();
    doc.mark_saved();
    doc
}

/// The repo-root `templates/first-person/` directory (committed starter scene).
pub fn firstperson_template_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../templates/first-person")
}

/// Write the committed first-person template scene from [`firstperson_scene`].
pub fn write_firstperson_template() -> Result<(), String> {
    let dir = firstperson_template_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("mkdir: {e}"))?;
    let doc = firstperson_scene();
    crate::scene::serialize::save(&doc, &dir.join("FirstPerson.inf_lvl"), Some(FP_LEVEL_GUID))?;
    std::fs::write(dir.join("README.md"), FIRSTPERSON_README).map_err(|e| format!("write: {e}"))?;
    Ok(())
}

const FIRSTPERSON_README: &str = "# First Person template\n\n\
The starter scene `inf new --template first-person` scaffolds: a 3D ground\n\
plane, a directional sun, a kinematic **Player** capsule that is a real pawn\n\
(`CharacterController3D` + `CharacterMovement { player_controlled }`), and a\n\
first-person **Camera** at eye height.\n\n\
Press Play and WASD moves the Player — the engine's own movement step drives\n\
it, so no Blueprint is required to walk. Before wave GTA1 the capsule had the\n\
controller and not the movement component, which meant no player-controlled\n\
character existed: Play fell back to an overhead camera and nothing moved.\n\n\
Generated by `inf_editor_core::samples::firstperson_scene`. Regenerate with\n\
`INF_BLESS_SAMPLES=1 cargo test -p inf-editor-core samples`.\n\n\
## What it still does not do (follow-up)\n\n\
The camera is the engine's **third-person** locomotion camera. `ViewMode` is\n\
camera-side only and never crosses the sim wire, so a level cannot ask for the\n\
first-person one yet; that is the tracked follow-up for a fully first-person\n\
starter, and the `Camera` entity in the scene records the intent meanwhile.\n";

// ── Blank-3D template scene (island I1 / IB-7) ───────────────────────────────
//
// The fourth template's starter scene. It exists because of the IB-7 ruling: a
// template that scaffolds no level produces a project `inf cook` refuses, and
// "blank" was the reason two of the four templates had nothing to scaffold. A
// blank project is still a project that must boot, so the floor here is exactly
// the floor of a shippable build — one level, something to stand on, something
// to see it by, and somewhere to see it from.

/// The blank-3D level's own asset GUID.
pub const BLANK3D_LEVEL_GUID: Uuid = Uuid::from_u128(0x8407_0000);
pub const BLANK3D_GROUND_GUID: Uuid = Uuid::from_u128(0x8407_0001);
pub const BLANK3D_SUN_GUID: Uuid = Uuid::from_u128(0x8407_0002);
pub const BLANK3D_CAMERA_GUID: Uuid = Uuid::from_u128(0x8407_0003);
/// The starter character the boot level spawns (wave GTA1). Fixed like every
/// other guid here so the committed bytes are reproducible.
pub const BLANK3D_PLAYER_GUID: Uuid = Uuid::from_u128(0x8407_0004);

/// The minimal **blank 3D** starter scene: a ground plane, a directional sun, a
/// camera looking at the origin — and, since wave GTA1, **the starter
/// character**. Scaffolded by `inf new --template blank-3d` (the default
/// template).
///
/// Four entities. It was three, and the fourth is the one that makes Play mean
/// anything: without a `player_controlled` character `camera_subject` returns
/// `None`, the player keeps its own overhead view, and pressing Play showed an
/// author their furniture from above with no way to move. The template already
/// scaffolded the seventeen files of `samples/starter-character` into
/// `Content/Characters/` (`inf_project::template::STARTER_CHARACTER`); nothing
/// spawned them.
///
/// Placed through `SceneDoc::edit_create_character_with_guid` — the same door
/// the New Character wizard and the island's hero take — so there is one
/// definition of what a character is, and the guids are
/// [`starter_character_ids`]'s committed ones so the level's references resolve
/// against the files the template ships.
///
/// The property it must keep is the one IB-7 was about: `inf new` → `inf cook` →
/// `inf-player --pack` runs it.
pub fn blank3d_scene() -> SceneDoc {
    use inf_ecs::components::{Camera, Light, LightKind, Material, MeshRef, Primitive};
    use inf_ecs::math::Vec3d;

    let mut doc = SceneDoc::new();
    doc.set_title("Blank 3D");

    // **THE LIT STACK, AUTHORED** (wave CERT1, CP-A1). A level that authors no
    // render block ships with shadows, GI, VSM, TAA, SSAO and bloom ALL OFF —
    // `fps_instrument.rs` has documented that about this engine for four waves,
    // and it meant the first thing anyone saw of it was unlit. A starter
    // template that plays unlit is not a certified starter.
    doc.set_settings(crate::scene::serialize::LevelSettings {
        render: crate::scene::serialize::RenderSettingsRecord::lit_showcase(),
        ..crate::scene::serialize::LevelSettings::default()
    });

    // A 20 × 20 m ground plane at the origin.
    doc.create_with_guid(BLANK3D_GROUND_GUID, SpawnKind::Empty, "Ground", None);
    insert!(
        doc,
        BLANK3D_GROUND_GUID,
        Transform {
            translation: Vec3d::ZERO,
            rotation: Vec3d::ZERO,
            scale: Vec3d::new(20.0, 1.0, 20.0),
        }
    );
    insert!(
        doc,
        BLANK3D_GROUND_GUID,
        MeshRef {
            primitive: Primitive::Plane,
            asset: None,
        }
    );
    insert!(
        doc,
        BLANK3D_GROUND_GUID,
        Material {
            base_color: Color::new(0.32, 0.33, 0.35, 1.0),
            ..Default::default()
        }
    );
    insert!(doc, BLANK3D_GROUND_GUID, ground_slab(10.0));

    // A directional sun, tilted so the plane is shaded rather than flat-lit.
    doc.create_with_guid(BLANK3D_SUN_GUID, SpawnKind::Empty, "Sun", None);
    insert!(
        doc,
        BLANK3D_SUN_GUID,
        Transform {
            translation: Vec3d::new(0.0, 10.0, 0.0),
            rotation: Vec3d::new(-50.0, -30.0, 0.0),
            scale: Vec3d::ONE,
        }
    );
    insert!(
        doc,
        BLANK3D_SUN_GUID,
        Light {
            kind: LightKind::Directional,
            color: Color::WHITE,
            intensity: 2.0,
            ..Default::default()
        }
    );

    // A camera set back from the origin, looking down at it.
    doc.create_with_guid(BLANK3D_CAMERA_GUID, SpawnKind::Empty, "Camera", None);
    insert!(
        doc,
        BLANK3D_CAMERA_GUID,
        Transform {
            translation: Vec3d::new(0.0, 4.0, -8.0),
            rotation: Vec3d::new(-20.0, 0.0, 0.0),
            scale: Vec3d::ONE,
        }
    );
    insert!(doc, BLANK3D_CAMERA_GUID, Camera::default());

    // The pawn (wave GTA1). Feet on the plane at the origin; the door lifts the
    // transform to the capsule's centre itself.
    let ids = starter_character_ids();
    let asset = |id: Option<inf_asset::AssetId>| id.expect("every starter id is fixed").0;
    doc.edit_create_character_with_guid(
        BLANK3D_PLAYER_GUID,
        STARTER_CHARACTER_NAME,
        asset(ids.skeleton),
        asset(ids.mesh),
        asset(ids.machine),
        DVec3::ZERO,
        Some(asset(ids.actor)),
        starter_character_spec().params.height_m,
    );

    doc.world_mut().propagate();
    doc.mark_saved();
    doc
}

/// The repo-root `templates/blank-3d/` directory (committed starter scene).
pub fn blank3d_template_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../templates/blank-3d")
}

/// Write the committed blank-3D template scene from [`blank3d_scene`].
pub fn write_blank3d_template() -> Result<(), String> {
    let dir = blank3d_template_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("mkdir: {e}"))?;
    let doc = blank3d_scene();
    crate::scene::serialize::save(&doc, &dir.join("Blank.inf_lvl"), Some(BLANK3D_LEVEL_GUID))?;
    std::fs::write(dir.join("README.md"), BLANK3D_README).map_err(|e| format!("write: {e}"))?;
    Ok(())
}

const BLANK3D_README: &str = "# Blank 3D template\n\n\
The starter scene `inf new --template blank-3d` scaffolds: a ground plane, a\n\
directional sun and a camera. Three entities — a starting point, not a demo.\n\n\
It exists so the default template produces a project that **cooks and runs**:\n\
`inf new My Game` then `inf cook --project \"My Game\"` then\n\
`inf-player --pack \"My Game/Build\" --headless --run-frames 300 --assert-exit`.\n\n\
Generated by `inf_editor_core::samples::blank3d_scene`. Regenerate with\n\
`INF_BLESS_SAMPLES=1 cargo test -p inf-editor-core samples`.\n";

// ── 2D-platformer template (island I1 / IB-7) ────────────────────────────────

/// The repo-root `templates/2d-platformer/` directory (committed starter scene).
pub fn platformer_template_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../templates/2d-platformer")
}

/// Write the committed 2D-platformer template from the **same** generators the
/// `samples/platformer-2d/` fixture is written from — [`platformer_scene`] and
/// [`coyote_class`].
///
/// # Why the bytes are duplicated rather than the sample being pointed at
///
/// A template and a sample are two different promises. The sample is the
/// engine's own fixture: dozens of gates read it, and it is re-tuned whenever one
/// of them needs it to be. The template is what an author's project starts as,
/// and it must not change under them because a gate wanted a different ledge
/// height. Duplicating the *bytes* while sharing the *generator* is what makes
/// both true at once — there is one definition of a platformer in this
/// repository, and two blessed copies of what it produced, each locked to it.
pub fn write_platformer_template() -> Result<(), String> {
    let dir = platformer_template_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("mkdir: {e}"))?;

    let doc = platformer_scene();
    crate::scene::serialize::save(
        &doc,
        &dir.join("Platformer.inf_lvl"),
        Some(PLATFORMER_TEMPLATE_LEVEL_GUID),
    )?;

    // The gameplay class the level's `ActorClass` binds, with the SAME asset
    // GUID the binding names — a template whose actor did not resolve would
    // scaffold a player that stands still.
    let act_bytes = encode_actor(&coyote_class())?;
    let act_path = dir.join("Coyote.inf_act");
    std::fs::write(&act_path, &act_bytes).map_err(|e| format!("write actor: {e}"))?;
    let side = inf_asset::AssetSidecar::new(
        inf_asset::AssetId(COYOTE_ASSET_GUID),
        inf_asset::AssetKind::Blueprint,
        inf_asset::ContentHash::of(&act_bytes),
    );
    side.save(&act_path)
        .map_err(|e| format!("write actor sidecar: {e}"))?;

    std::fs::write(dir.join("README.md"), PLATFORMER_TEMPLATE_README)
        .map_err(|e| format!("write: {e}"))?;
    Ok(())
}

/// The template platformer level's own asset GUID — **distinct from the
/// sample's** [`LEVEL_GUID`], because two levels with one GUID in one asset
/// database is a collision, and an author who imports the sample into a project
/// scaffolded from this template would hit exactly that.
pub const PLATFORMER_TEMPLATE_LEVEL_GUID: Uuid = Uuid::from_u128(0x8408_0000);

const PLATFORMER_TEMPLATE_README: &str = "# 2D Platformer template\n\n\
The starter project `inf new --template 2d-platformer` scaffolds: a ground\n\
collider with a painted tilemap strip, a floating platform, a soft 2D light,\n\
and a player capsule bound to the **Coyote** blueprint (coyote-time jumping).\n\n\
`Coyote.inf_act` is scaffolded beside the level because the level's\n\
`ActorClass` binding names it; without it the player would not move.\n\n\
Generated by `inf_editor_core::samples::platformer_scene` +\n\
`coyote_class`. Regenerate with\n\
`INF_BLESS_SAMPLES=1 cargo test -p inf-editor-core samples`.\n";

// ── Committed files (fixture discipline) ─────────────────────────────────────

/// The repo-root `samples/platformer-2d/` directory.
pub fn sample_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../samples/platformer-2d")
}

/// Encode a [`BlueprintClass`] to the deterministic `.inf_act` payload.
///
/// The `.inf_act` is stored as **pretty JSON**, not bincode: `BlueprintClass`
/// carries `#[serde(skip_serializing_if)]` fields (e.g. `parent`) that a
/// non-self-describing bincode stream cannot round-trip (the omitted field
/// desynchronizes the decoder). JSON is self-describing, deterministic (ordered
/// `Vec`s + `BTreeMap`s), and human-diffable — the right fit for a blueprint
/// document. (A future asset-DB integration for blueprints would either drop the
/// `skip_serializing_if` or keep JSON; see the P8.4 notes.)
pub fn encode_actor(class: &BlueprintClass) -> Result<Vec<u8>, String> {
    serde_json::to_vec_pretty(class).map_err(|e| format!("encode actor: {e}"))
}

/// Decode a `.inf_act` JSON payload.
pub fn decode_actor(bytes: &[u8]) -> Result<BlueprintClass, String> {
    serde_json::from_slice(bytes).map_err(|e| format!("decode actor: {e}"))
}

/// Write the committed sample files from the generators (regeneration path).
/// Used by the blessed regeneration test; also handy for tooling.
pub fn write_sample() -> Result<(), String> {
    let dir = sample_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("mkdir: {e}"))?;

    // Scene (payload + sidecar) with a fixed level GUID.
    let doc = platformer_scene();
    crate::scene::serialize::save(&doc, &dir.join("Platformer.inf_lvl"), Some(LEVEL_GUID))?;

    // Actor: JSON payload (see `encode_actor` — bincode can't round-trip it).
    let class = coyote_class();
    let act_bytes = encode_actor(&class)?;
    let act_path = dir.join("Coyote.inf_act");
    std::fs::write(&act_path, &act_bytes).map_err(|e| format!("write actor: {e}"))?;

    // Its inf_asset sidecar with the **stable** [`COYOTE_ASSET_GUID`], so the
    // level's persisted `actor` binding resolves to this blueprint through the
    // AssetDb + cooked pack (P9.5 dependency edge level→blueprint).
    let side = inf_asset::AssetSidecar::new(
        inf_asset::AssetId(COYOTE_ASSET_GUID),
        inf_asset::AssetKind::Blueprint,
        inf_asset::ContentHash::of(&act_bytes),
    );
    side.save(&act_path)
        .map_err(|e| format!("write actor sidecar: {e}"))?;

    std::fs::write(dir.join("README.md"), SAMPLE_README)
        .map_err(|e| format!("write readme: {e}"))?;
    Ok(())
}

// ── Terrain-demo gate scene (P10.6) ──────────────────────────────────────────
//
// The Phase-10-closing gate scene: a multi-tile sculpted + splat-painted terrain,
// a PCG scatter volume (noise+slope rule) referencing a committed `.inf_pcg`
// graph, a directional sun, and a camera. Committed as v4 `.inf_lvl` bytes + the
// `.inf_pcg` sidecar under `samples/terrain-demo/`. The terrain heights come from
// [`terrain_demo_height`] so the runtime gate can probe `height_at` against the
// exact generator function.

pub const TERRAIN_DEMO_LEVEL_GUID: Uuid = Uuid::from_u128(0x8403_0000);
pub const TERRAIN_DEMO_TERRAIN_GUID: Uuid = Uuid::from_u128(0x8403_0001);
pub const TERRAIN_DEMO_PCG_GUID: Uuid = Uuid::from_u128(0x8403_0002);
pub const TERRAIN_DEMO_SUN_GUID: Uuid = Uuid::from_u128(0x8403_0003);
pub const TERRAIN_DEMO_CAMERA_GUID: Uuid = Uuid::from_u128(0x8403_0004);
/// The **asset** GUID of the committed `Scatter.inf_pcg` (its inf_asset sidecar).
/// Stable so the level's `PcgVolume.graph` ref resolves through the AssetDb /
/// cooked pack, and the cook's level→pcg dep edge ships the graph.
pub const TERRAIN_DEMO_PCG_ASSET_GUID: Uuid = Uuid::from_u128(0x8403_00AA);

/// Terrain samples-per-tile side + world spacing for the demo (small, so the
/// committed payload stays compact while genuinely multi-tile).
pub const TERRAIN_DEMO_RESOLUTION: u32 = 16;
pub const TERRAIN_DEMO_MPS: f64 = 2.0;
/// World XZ span authored (a 2×2 tile block at the above resolution/spacing).
pub const TERRAIN_DEMO_SPAN: f64 = 64.0;

/// The demo's analytic terrain height at world `(x, z)` — a gentle sine hill. The
/// runtime gate probes `TerrainData::height_at` at a grid point against this.
///
/// **Portable trig, because this generator is byte-locked** (the P14 law).
/// `committed_sample_matches_generators` re-runs this function and asserts the
/// result is byte-equal to the committed `TerrainDemo.inf_lvl`. With `std`'s
/// `sin`/`cos` that is an assertion about *the machine running the test*: the
/// platform libm is entitled to disagree in the last ulp, so the byte-lock was
/// green where it was blessed and a latent red on any target whose libm rounds
/// differently. `psin64`/`pcos64` use only IEEE add/mul/floor, which makes the
/// lock mean what it says. (~1e-7 accuracy here against the gate's 1e-3 probe
/// tolerance — four orders of margin.)
pub fn terrain_demo_height(x: f64, z: f64) -> f64 {
    6.0 * inf_math::psin64(x * 0.08) * inf_math::pcos64(z * 0.08)
}

/// Build the demo's [`inf_pcg::PcgDocument`]: one layer, one rule scattering on a
/// noise-modulated gentle-slope band — a few hundred instances over the terrain.
/// Two weighted kinds so a multi-kind scatter reads as varied placeholder content.
pub fn terrain_demo_pcg_document() -> inf_pcg::PcgDocument {
    use inf_pcg::{PcgKind, PcgRule, SamplerDef};
    let sampler = SamplerDef::Multiply(
        Box::new(SamplerDef::Noise(inf_pcg::ValueNoise {
            seed: 1337,
            frequency: 0.05,
            octaves: 3,
            lacunarity: 2.0,
            gain: 0.5,
        })),
        Box::new(SamplerDef::Slope {
            min_deg: 0.0,
            max_deg: 32.0,
            feather_deg: 6.0,
        }),
    );
    let rule = PcgRule {
        name: "vegetation".into(),
        sampler,
        scatter: inf_pcg::ScatterParams {
            seed: 2026_0721,
            cell_size: 8.0,
            base_density: 0.5,
            jitter: 1.0,
            align_to_normal: false,
            scale_range: (0.8, 1.4),
            rotation: inf_pcg::RotationMode::RandomYaw,
            altitude_offset: 0.0,
        },
        kinds: vec![
            PcgKind {
                mesh: None,
                weight: 3.0,
            },
            PcgKind {
                mesh: None,
                weight: 1.0,
            },
        ],
    };
    inf_pcg::PcgDocument::single_layer("ground", vec![rule])
}

/// The committed `.inf_pcg` payload for the demo (document-only envelope — the
/// player evaluates from its stored lowered document).
pub fn terrain_demo_pcg_payload() -> inf_pcg::PcgAssetPayload {
    inf_pcg::PcgAssetPayload::new(terrain_demo_pcg_document())
}

/// Build the terrain-demo [`SceneDoc`]: a sculpted + painted heightfield terrain,
/// a PCG scatter volume referencing the committed graph, a directional sun, and a
/// camera framing the terrain.
pub fn terrain_demo_scene() -> SceneDoc {
    use inf_ecs::components::{Camera, Light, LightKind, PcgVolume, Terrain};

    let mut doc = SceneDoc::new();
    doc.set_title("Terrain Demo");

    // ── Terrain: a multi-tile sine hill (sculpt-level detail via write_region) +
    //    two splat-painted bands (materialized weights on some tiles, defaults on
    //    others — the sparse/materialized mix). ──
    doc.create_with_guid(TERRAIN_DEMO_TERRAIN_GUID, SpawnKind::Empty, "Terrain", None);
    // Terrain entity sits at the origin, so world XZ == terrain-local XZ (the
    // height probe + PCG height seam are then the bare generator function).
    insert!(
        doc,
        TERRAIN_DEMO_TERRAIN_GUID,
        Transform::from_translation(DVec3::ZERO)
    );
    {
        let mut terrain = Terrain::configured(TERRAIN_DEMO_RESOLUTION, TERRAIN_DEMO_MPS);
        terrain.data.write_region(
            glam::DVec2::ZERO,
            glam::DVec2::splat(TERRAIN_DEMO_SPAN),
            terrain_demo_height,
        );
        // Splat band A: rock (layer 1) over the low-left quadrant.
        let _ = inf_terrain::apply_paint(
            &mut terrain.data,
            1,
            inf_terrain::BrushParams {
                center: glam::DVec2::new(16.0, 16.0),
                radius: 14.0,
                strength: 1.0,
                falloff: inf_terrain::Falloff::Plateau(0.5),
            },
        );
        // Splat band B: dirt (layer 2) over an upper strip.
        let _ = inf_terrain::apply_paint(
            &mut terrain.data,
            2,
            inf_terrain::BrushParams {
                center: glam::DVec2::new(16.0, 48.0),
                radius: 12.0,
                strength: 1.0,
                falloff: inf_terrain::Falloff::Smooth,
            },
        );
        terrain.macro_variation = 0.2;
        insert!(doc, TERRAIN_DEMO_TERRAIN_GUID, terrain);
    }

    // ── PCG scatter volume: references the committed graph; centered over the
    //    terrain so its region covers the authored footprint. ──
    doc.create_with_guid(
        TERRAIN_DEMO_PCG_GUID,
        SpawnKind::Empty,
        "Scatter Volume",
        None,
    );
    insert!(
        doc,
        TERRAIN_DEMO_PCG_GUID,
        Transform::from_translation(DVec3::new(
            TERRAIN_DEMO_SPAN * 0.5,
            0.0,
            TERRAIN_DEMO_SPAN * 0.5
        ))
    );
    insert!(
        doc,
        TERRAIN_DEMO_PCG_GUID,
        PcgVolume {
            graph: Some(TERRAIN_DEMO_PCG_ASSET_GUID),
            extent: Vec2d::new(TERRAIN_DEMO_SPAN * 0.5, TERRAIN_DEMO_SPAN * 0.5),
            seed: 0,
            ..Default::default()
        }
    );

    // ── A directional sun. ──
    doc.create_with_guid(TERRAIN_DEMO_SUN_GUID, SpawnKind::Empty, "Sun", None);
    insert!(
        doc,
        TERRAIN_DEMO_SUN_GUID,
        Transform {
            translation: inf_ecs::math::Vec3d::new(0.0, 40.0, 0.0),
            // Angle the sun so the hills cast readable shading.
            rotation: inf_ecs::math::Vec3d::new(-50.0, -30.0, 0.0),
            scale: inf_ecs::math::Vec3d::new(1.0, 1.0, 1.0),
        }
    );
    insert!(
        doc,
        TERRAIN_DEMO_SUN_GUID,
        Light {
            kind: LightKind::Directional,
            color: Color::new(1.0, 0.97, 0.9, 1.0),
            intensity: 2.5,
            ..Default::default()
        }
    );

    // ── A camera framing the terrain (the "camera note"). ──
    doc.create_with_guid(TERRAIN_DEMO_CAMERA_GUID, SpawnKind::Empty, "Camera", None);
    insert!(
        doc,
        TERRAIN_DEMO_CAMERA_GUID,
        Transform::from_translation(DVec3::new(TERRAIN_DEMO_SPAN * 0.5, 30.0, -20.0))
    );
    insert!(doc, TERRAIN_DEMO_CAMERA_GUID, Camera::default());

    doc.world_mut().propagate();
    doc.mark_saved();
    doc
}

/// The repo-root `samples/terrain-demo/` directory.
pub fn terrain_demo_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../samples/terrain-demo")
}

/// Write the committed terrain-demo files from the generators (regeneration path):
/// the v4 `.inf_lvl` (+ sidecar), the `.inf_pcg` graph (+ its inf_asset sidecar so
/// the `PcgVolume.graph` ref resolves through the AssetDb / cooked pack), + README.
pub fn write_terrain_demo() -> Result<(), String> {
    let dir = terrain_demo_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("mkdir: {e}"))?;

    // Level (payload + sidecar) with a fixed level GUID.
    let doc = terrain_demo_scene();
    crate::scene::serialize::save(
        &doc,
        &dir.join("TerrainDemo.inf_lvl"),
        Some(TERRAIN_DEMO_LEVEL_GUID),
    )?;

    // The `.inf_pcg` graph payload + its inf_asset sidecar with the STABLE asset
    // GUID the level's PcgVolume.graph points at.
    let pcg_bytes = terrain_demo_pcg_payload()
        .encode()
        .map_err(|e| format!("encode pcg: {e}"))?;
    let pcg_path = dir.join("Scatter.inf_pcg");
    std::fs::write(&pcg_path, &pcg_bytes).map_err(|e| format!("write pcg: {e}"))?;
    let side = inf_asset::AssetSidecar::new(
        inf_asset::AssetId(TERRAIN_DEMO_PCG_ASSET_GUID),
        inf_asset::AssetKind::Pcg,
        inf_asset::ContentHash::of(&pcg_bytes),
    );
    side.save(&pcg_path)
        .map_err(|e| format!("write pcg sidecar: {e}"))?;

    std::fs::write(dir.join("README.md"), TERRAIN_DEMO_README)
        .map_err(|e| format!("write readme: {e}"))?;
    Ok(())
}

const TERRAIN_DEMO_README: &str = "# Terrain Demo (Phase-10 gate scene)\n\n\
Generated by `inf_editor_core::samples::terrain_demo_scene` — the P10.6 gate\n\
scene: a multi-tile **sculpted + splat-painted** heightfield terrain, a **PCG\n\
scatter volume** (noise+slope rule) referencing `Scatter.inf_pcg`, a directional\n\
sun, and a camera.\n\n\
- `TerrainDemo.inf_lvl` — the scene as schema-v4 `.inf_lvl` bytes (terrain +\n\
  PcgVolume persist).\n\
- `Scatter.inf_pcg` — the scatter graph the volume evaluates on load (its\n\
  instances are a derived cache, never persisted in the level).\n\n\
The terrain heights are `terrain_demo_height(x, z)`; the runtime gate probes\n\
`TerrainData::height_at` against it. The PCG volume's `evaluated` cache is\n\
re-computed on load (editor `pcg_evaluate`, shipped/PIE player `evaluate_pcg_volumes`).\n\n\
Regenerate with `INF_BLESS_SAMPLES=1 cargo test -p inf-editor-core samples`.\n";

const SAMPLE_README: &str = "# 2D Platformer sample\n\n\
Generated by `inf_editor_core::samples` — the Phase-8 gate scene. A small\n\
platformer with a **Blueprint coyote-time jump** that plays in-viewport via the\n\
interpreter.\n\n\
- `Platformer.inf_lvl` — the scene (tilemap ground + collider ledge + platform +\n\
  a kinematic character player).\n\
- `Coyote.inf_act` — the player's blueprint class (BeginPlay + Tick coyote-time\n\
  handler), stored as JSON.\n\n\
Regenerate with `INF_BLESS_SAMPLES=1 cargo test -p inf-editor-core samples`.\n";

// ── Character-demo gate scene (P11.4) ────────────────────────────────────────
//
// The Phase-11-closing gate scene: a sine-hill terrain (P10-style) and a
// **character** driven by a Blueprint across it. The character carries the full
// P11 animation/character component set — `SkeletalMesh` + `AnimStateMachine` +
// `RootMotion` + a 3D `CharacterController3D`/`Collider3D`/`RigidBody3D` — plus an
// actor blueprint that reads left/right input, moves via `physics3d.move_and_slide`
// and tracks terrain height via the `terrain.height_at` host seam, jumping on the
// `jump` action. A procedural 6-joint humanoid-ish skeleton, three programmatic
// clips (idle bob / run forward via `straight_line_clip` / jump arc), and a
// state machine (idle→run on speed>0.1, run→idle on ≤0.1, any→jump on
// jump_pressed>0.5 with exit back) are committed as `.inf_skel` / `.inf_anim` /
// `.inf_sm` sidecars. Committed as v5 `.inf_lvl` bytes under
// `samples/character-demo/`.
//
// Root motion note: the run clip is authored with forward root translation (via
// `straight_line_clip`, honest placeholder), but the entity is **state-machine
// driven** (no `AnimPlayer`), so locomotion comes from the Blueprint's
// `physics3d.move_and_slide` — root-motion *extraction* (which reads `AnimPlayer`)
// is inert this session; the `RootMotion` component persists for future use.

pub const CHARACTER_DEMO_LEVEL_GUID: Uuid = Uuid::from_u128(0x8404_0000);
pub const CHARACTER_DEMO_TERRAIN_GUID: Uuid = Uuid::from_u128(0x8404_0001);
pub const CHARACTER_DEMO_CHARACTER_GUID: Uuid = Uuid::from_u128(0x8404_0002);
pub const CHARACTER_DEMO_SUN_GUID: Uuid = Uuid::from_u128(0x8404_0003);
pub const CHARACTER_DEMO_CAMERA_GUID: Uuid = Uuid::from_u128(0x8404_0004);
/// The committed anim asset GUIDs the level's components reference (stable so the
/// refs resolve through the AssetDb / cooked pack and the cook's dep edges ship them).
pub const CHARACTER_DEMO_SKELETON_GUID: Uuid = Uuid::from_u128(0x8404_00A0);
pub const CHARACTER_DEMO_IDLE_CLIP_GUID: Uuid = Uuid::from_u128(0x8404_00A1);
pub const CHARACTER_DEMO_RUN_CLIP_GUID: Uuid = Uuid::from_u128(0x8404_00A2);
pub const CHARACTER_DEMO_JUMP_CLIP_GUID: Uuid = Uuid::from_u128(0x8404_00A3);
pub const CHARACTER_DEMO_SM_GUID: Uuid = Uuid::from_u128(0x8404_00A4);
pub const CHARACTER_DEMO_ACTOR_GUID: Uuid = Uuid::from_u128(0x8404_00AC);

/// The actor class id string.
pub const CHARACTER_DEMO_CLASS_ID: &str = "act:character_demo";

// Terrain + tuning (world units, seconds).
pub const CHARACTER_DEMO_RESOLUTION: u32 = 16;
pub const CHARACTER_DEMO_MPS: f64 = 2.0;
/// Horizontal run speed (m/s) from left/right input.
pub const CHAR_MOVE_SPEED: f64 = 4.0;
/// Downward acceleration applied to `vy` each tick.
pub const CHAR_GRAVITY: f64 = 20.0;
/// Upward velocity a jump imparts.
pub const CHAR_JUMP_SPEED: f64 = 7.0;
/// Height the capsule centre stands above the sampled terrain height.
pub const CHAR_STAND: f64 = 0.9;
/// Start position (world X); the character begins at the terrain origin where the
/// height is exactly `0`, so its start Y is exactly `CHAR_STAND` (grounded).
pub const CHAR_START_X: f64 = 0.0;
/// Start Y = terrain height at the start (0) + the stand offset. Kept a const so
/// the spawn `Transform` and the Blueprint's `BeginPlay` position seed agree
/// exactly — the invariant that makes `transform == tracked position` hold each
/// tick (`move_and_slide` deltas telescope).
pub const CHAR_START_Y: f64 = CHAR_STAND;

/// The demo's analytic terrain height at world `(x, z)` — a gentle sine hill along
/// X (flat in Z along the character's z=0 path). `height(0,0) == 0`, so the
/// character starts grounded at `CHAR_START_Y`. The runtime gate probes
/// `TerrainData::height_at` + the character's Y against this.
///
/// **Portable trig** for the same reason as [`terrain_demo_height`]: the
/// committed `Character.inf_lvl` is byte-locked against this function, and the
/// character-demo gate compares a PIE trace to a shipping one on ground heights
/// derived from it. Both claims are about two machines agreeing, which is
/// exactly what `std`'s libm does not promise.
pub fn character_demo_height(x: f64, z: f64) -> f64 {
    3.0 * inf_math::psin64(x * 0.08) * inf_math::pcos64(z * 0.08)
}

/// A procedural 6-joint humanoid-ish skeleton (honest placeholder): hips (root) →
/// spine → head, plus two upper arms and a foot. Bind transforms are simple local
/// offsets; inverse-binds are identity (a placeholder rig, not an imported mesh).
pub fn character_demo_skeleton() -> inf_anim::Skeleton {
    use glam::{Mat4, Quat, Vec3};
    use inf_anim::{Joint, JointTransform};
    let joint = |name: &str, parent: Option<u16>, t: Vec3| Joint {
        name: name.into(),
        parent,
        inverse_bind: Mat4::IDENTITY.to_cols_array(),
        local_bind: JointTransform::from_trs(t, Quat::IDENTITY, Vec3::ONE),
    };
    inf_anim::Skeleton::new(vec![
        joint("hips", None, Vec3::new(0.0, 1.0, 0.0)),
        joint("spine", Some(0), Vec3::new(0.0, 0.4, 0.0)),
        joint("head", Some(1), Vec3::new(0.0, 0.4, 0.0)),
        joint("upper_arm_l", Some(1), Vec3::new(-0.25, 0.3, 0.0)),
        joint("upper_arm_r", Some(1), Vec3::new(0.25, 0.3, 0.0)),
        joint("foot", Some(0), Vec3::new(0.0, -0.9, 0.0)),
    ])
    .expect("valid procedural skeleton")
}

/// The **idle** clip: a subtle vertical bob of the hips (joint 0) over 2 s, looping.
pub fn character_demo_idle_clip() -> inf_anim::AnimClip {
    use inf_anim::{AnimClip, Interpolation, JointTrack, Vec3Track};
    let mut jt = JointTrack::new(0);
    jt.translation = Some(Vec3Track::new(
        vec![0.0, 1.0, 2.0],
        vec![[0.0, 1.0, 0.0], [0.0, 1.05, 0.0], [0.0, 1.0, 0.0]],
        Interpolation::Linear,
    ));
    AnimClip::new("idle", vec![jt])
}

/// The **run** clip: forward root motion via the `straight_line_clip` helper
/// (hips translate +X over the loop). Authored honestly even though locomotion is
/// Blueprint-driven this session (see the module note).
pub fn character_demo_run_clip() -> inf_anim::AnimClip {
    inf_anim::root_motion::straight_line_clip("run", glam::Vec3::X, 2.0, 0.6)
}

/// The **jump** clip: a vertical arc on the hips (joint 0) over 0.5 s, non-looping.
pub fn character_demo_jump_clip() -> inf_anim::AnimClip {
    use inf_anim::{AnimClip, Interpolation, JointTrack, Vec3Track};
    let mut jt = JointTrack::new(0);
    jt.translation = Some(Vec3Track::new(
        vec![0.0, 0.25, 0.5],
        vec![[0.0, 1.0, 0.0], [0.0, 1.6, 0.0], [0.0, 1.0, 0.0]],
        Interpolation::Linear,
    ));
    AnimClip::new("jump", vec![jt])
}

/// The state machine: idle(0) / run(1) / jump(2). Reads the actor's `speed` +
/// `jump` Blueprint variables via the [`SmContext`] seam.
///
/// # Re-authored for `.inf_sm` v2 (P29.1), behaviour unchanged
///
/// Two of the three v1 workarounds this table carried are gone, and the third
/// never was one:
///
/// * "jump wins over run" used to be expressed by *where the two jump edges sat
///   in a `Vec`* — a comment saying `declared first`, invisible in the editor and
///   destroyed by any re-ordering. It is a `priority` now, so the property
///   survives the list being sorted.
/// * the two jump edges (`idle→jump`, `run→jump`) were the same edge written
///   twice because v1 had no **any-state** source. They are one edge, with
///   `exclude_self` doing what the missing `jump→jump` row's absence did.
/// * both parameters are **declared**, which costs nothing behaviourally (a
///   `Float` is what an undeclared name already read as) and is what lets a
///   reader — and P29.5's rule builder — know what this machine expects from
///   gameplay without grepping its conditions.
///
/// The evaluated behaviour is identical, which is the point: `phase24_gate` and
/// `character_demo`'s PIE == shipping trace are the check on that claim, not this
/// paragraph.
pub fn character_demo_state_machine() -> inf_anim::StateMachine {
    use inf_anim::state_machine::{CmpOp, SmParam, SmState, SmTransition};
    let clip_ref = |g: Uuid| *g.as_bytes();
    let state = |name: &str, g: Uuid, looping: bool, position: (f32, f32)| {
        let mut s = SmState::clip_at(name, clip_ref(g), position);
        s.looping = looping;
        s
    };
    // The cross-fade every edge in this machine uses.
    let tr = |from: usize, to: usize, var: &str, op: CmpOp, value: f64| {
        SmTransition::on(from, to, 0.15, var, op, value)
    };
    inf_anim::StateMachine {
        states: vec![
            state("idle", CHARACTER_DEMO_IDLE_CLIP_GUID, true, (0.0, 0.0)),
            state("run", CHARACTER_DEMO_RUN_CLIP_GUID, true, (240.0, 0.0)),
            state(
                "jump",
                CHARACTER_DEMO_JUMP_CLIP_GUID,
                false,
                (120.0, -160.0),
            ),
        ],
        transitions: vec![
            // ANY → jump, at a priority that says what the v1 comment said.
            SmTransition::any(2, 0.15)
                .when(inf_anim::SmCond::float("jump", CmpOp::Gt, 0.5))
                .with_priority(10),
            // locomotion.
            tr(0, 1, "speed", CmpOp::Gt, 0.1),
            tr(1, 0, "speed", CmpOp::Le, 0.1),
            // exit jump back to run (moving) or idle (stopped).
            tr(2, 1, "speed", CmpOp::Gt, 0.1),
            tr(2, 0, "speed", CmpOp::Le, 0.1),
        ],
        entry: 0,
        params: vec![SmParam::float("speed"), SmParam::float("jump")],
        profiles: Vec::new(),
    }
}

/// The character's **Tick** handler as `BlueprintFn` IR. Reads left/right + jump
/// input, integrates a var-tracked position, clamps Y to `terrain.height_at + STAND`
/// (gravity + grounding), sets the `speed`/`jump` vars the state machine reads, and
/// drives the entity with `physics3d.move_and_slide` deltas.
///
/// Locals: `n1=entity`, `n2=vx` (mut), `n3=old_py`, `n4=ground`.
fn character_tick_fn() -> BlueprintFn {
    let e = || local(1);
    let dt = || Expr::Param("dt".to_string());
    let vx = || local(2);
    let old_py = || local(3);
    let ground = || local(4);

    let body = vec![
        let_named(1, "entity", false, get_var("entity")),
        // Horizontal velocity from held input.
        let_named(2, "vx", true, float_lit(0.0)),
        if_then(
            call(&["input", "is_down"], vec![str_lit("right")]),
            vec![Stmt::Assign {
                target: LocalId(2),
                value: bin(BinOp::Add, vx(), float_lit(CHAR_MOVE_SPEED)),
            }],
            vec![],
        ),
        if_then(
            call(&["input", "is_down"], vec![str_lit("left")]),
            vec![Stmt::Assign {
                target: LocalId(2),
                value: bin(BinOp::Sub, vx(), float_lit(CHAR_MOVE_SPEED)),
            }],
            vec![],
        ),
        // speed = |vx| → drives idle↔run.
        if_then(
            bin(BinOp::Lt, vx(), float_lit(0.0)),
            vec![set_var("speed", bin(BinOp::Sub, float_lit(0.0), vx()))],
            vec![set_var("speed", vx())],
        ),
        // Jump on the rising edge while grounded → seed vy + the jump var.
        if_then(
            bin(
                BinOp::And,
                call(&["input", "just_pressed"], vec![str_lit("jump")]),
                bin(BinOp::Gt, get_var("grounded"), float_lit(0.5)),
            ),
            vec![
                set_var("vy", float_lit(CHAR_JUMP_SPEED)),
                set_var("jump", float_lit(1.0)),
            ],
            vec![set_var("jump", float_lit(0.0))],
        ),
        // Gravity.
        set_var(
            "vy",
            bin(
                BinOp::Sub,
                get_var("vy"),
                bin(BinOp::Mul, float_lit(CHAR_GRAVITY), dt()),
            ),
        ),
        // Integrate the var-tracked position.
        let_named(3, "old_py", false, get_var("py")),
        set_var(
            "px",
            bin(BinOp::Add, get_var("px"), bin(BinOp::Mul, vx(), dt())),
        ),
        set_var(
            "py",
            bin(BinOp::Add, old_py(), bin(BinOp::Mul, get_var("vy"), dt())),
        ),
        // Ground = terrain height under the character + the stand offset.
        let_named(
            4,
            "ground",
            false,
            bin(
                BinOp::Add,
                call(
                    &["terrain", "height_at"],
                    vec![get_var("px"), get_var("pz")],
                ),
                float_lit(CHAR_STAND),
            ),
        ),
        // Clamp to the ground when at/under it; else airborne.
        if_then(
            bin(BinOp::Le, get_var("py"), ground()),
            vec![
                set_var("py", ground()),
                set_var("vy", float_lit(0.0)),
                set_var("grounded", float_lit(1.0)),
            ],
            vec![set_var("grounded", float_lit(0.0))],
        ),
        // Move the entity by this tick's delta (x + the y delta toward the target).
        Stmt::ExprStmt(call(
            &["physics3d", "move_and_slide"],
            vec![
                e(),
                bin(BinOp::Mul, vx(), dt()),
                bin(BinOp::Sub, get_var("py"), old_py()),
                float_lit(0.0),
            ],
        )),
    ];

    BlueprintFn {
        id: EventKind::Tick.key(),
        name: EventKind::Tick.key(),
        params: vec![Param {
            name: "dt".to_string(),
            ty: Ty::Float,
        }],
        ret: Ty::Unit,
        body,
    }
}

/// A `BeginPlay` handler seeding the var-tracked position to the spawn position
/// (so `transform == tracked position` holds), grounded and at rest.
fn character_begin_play_fn() -> BlueprintFn {
    BlueprintFn {
        id: EventKind::BeginPlay.key(),
        name: EventKind::BeginPlay.key(),
        params: vec![],
        ret: Ty::Unit,
        body: vec![
            set_var("px", float_lit(CHAR_START_X)),
            set_var("py", float_lit(CHAR_START_Y)),
            set_var("pz", float_lit(0.0)),
            set_var("vy", float_lit(0.0)),
            set_var("speed", float_lit(0.0)),
            set_var("jump", float_lit(0.0)),
            set_var("grounded", float_lit(1.0)),
        ],
    }
}

/// The character's [`BlueprintClass`] (the `.inf_act`).
pub fn character_demo_class() -> BlueprintClass {
    let mut class = BlueprintClass::new(CHARACTER_DEMO_CLASS_ID, "Character Demo");
    let fvar = |name: &str| Variable {
        name: name.into(),
        ty: Ty::Float,
        default: Lit::Float(0.0),
        exposed: true,
    };
    class.variables = vec![
        Variable {
            name: "entity".into(),
            ty: Ty::Int,
            default: Lit::Int(0),
            exposed: false,
        },
        fvar("px"),
        fvar("py"),
        fvar("pz"),
        fvar("vy"),
        fvar("speed"),
        fvar("jump"),
        fvar("grounded"),
    ];
    class.events = vec![
        EventBinding {
            event: EventKind::BeginPlay,
            body: character_begin_play_fn(),
        },
        EventBinding {
            event: EventKind::Tick,
            body: character_tick_fn(),
        },
    ];
    class
}

/// Build the character-demo [`SceneDoc`]: a sine-hill terrain, a character entity
/// carrying the full P11 animation/character component set + the actor binding, a
/// directional sun, and a camera.
pub fn character_demo_scene() -> SceneDoc {
    use inf_ecs::components::{
        AnimStateMachine, BodyKind3D, Camera, CharacterController3D, Collider3D,
        ColliderShape3DKind, Light, LightKind, RigidBody3D, RootMotion, SkeletalMesh, Terrain,
    };

    let mut doc = SceneDoc::new();
    doc.set_title("Character Demo");

    // ── Terrain: a sine hill at the origin (world XZ == terrain-local XZ, so the
    //    height probe + the character's terrain.height_at seam are the bare
    //    generator function). Authored over the character's path. ──
    doc.create_with_guid(
        CHARACTER_DEMO_TERRAIN_GUID,
        SpawnKind::Empty,
        "Terrain",
        None,
    );
    insert!(
        doc,
        CHARACTER_DEMO_TERRAIN_GUID,
        Transform::from_translation(DVec3::ZERO)
    );
    {
        let mut terrain = Terrain::configured(CHARACTER_DEMO_RESOLUTION, CHARACTER_DEMO_MPS);
        terrain.data.write_region(
            glam::DVec2::new(-16.0, -16.0),
            glam::DVec2::new(48.0, 16.0),
            character_demo_height,
        );
        terrain.macro_variation = 0.15;
        insert!(doc, CHARACTER_DEMO_TERRAIN_GUID, terrain);
    }

    // ── The character: SkeletalMesh + AnimStateMachine + RootMotion + a 3D
    //    kinematic character controller, standing at the origin (grounded). ──
    doc.create_with_guid(
        CHARACTER_DEMO_CHARACTER_GUID,
        SpawnKind::Empty,
        "Character",
        None,
    );
    insert!(
        doc,
        CHARACTER_DEMO_CHARACTER_GUID,
        Transform::from_translation(DVec3::new(CHAR_START_X, CHAR_START_Y, 0.0))
    );
    insert!(
        doc,
        CHARACTER_DEMO_CHARACTER_GUID,
        SkeletalMesh {
            mesh: None,
            skeleton: Some(CHARACTER_DEMO_SKELETON_GUID),
        }
    );
    insert!(
        doc,
        CHARACTER_DEMO_CHARACTER_GUID,
        AnimStateMachine {
            sm: Some(CHARACTER_DEMO_SM_GUID),
            params_from_vars: true,
            ..Default::default()
        }
    );
    insert!(doc, CHARACTER_DEMO_CHARACTER_GUID, RootMotion::apply());
    insert!(
        doc,
        CHARACTER_DEMO_CHARACTER_GUID,
        RigidBody3D {
            kind: BodyKind3D::Kinematic,
            ..Default::default()
        }
    );
    insert!(
        doc,
        CHARACTER_DEMO_CHARACTER_GUID,
        Collider3D {
            shape_kind: ColliderShape3DKind::Capsule,
            half_extents: inf_ecs::math::Vec3d::new(0.3, 0.5, 0.3),
            radius: 0.3,
            ..Default::default()
        }
    );
    insert!(
        doc,
        CHARACTER_DEMO_CHARACTER_GUID,
        CharacterController3D::default()
    );
    insert!(
        doc,
        CHARACTER_DEMO_CHARACTER_GUID,
        ActorClass(CHARACTER_DEMO_ACTOR_GUID)
    );

    // ── A directional sun. ──
    doc.create_with_guid(CHARACTER_DEMO_SUN_GUID, SpawnKind::Empty, "Sun", None);
    insert!(
        doc,
        CHARACTER_DEMO_SUN_GUID,
        Transform {
            translation: inf_ecs::math::Vec3d::new(0.0, 30.0, 0.0),
            rotation: inf_ecs::math::Vec3d::new(-50.0, -30.0, 0.0),
            scale: inf_ecs::math::Vec3d::new(1.0, 1.0, 1.0),
        }
    );
    insert!(
        doc,
        CHARACTER_DEMO_SUN_GUID,
        Light {
            kind: LightKind::Directional,
            color: Color::new(1.0, 0.97, 0.9, 1.0),
            intensity: 2.5,
            ..Default::default()
        }
    );

    // ── A camera framing the character. ──
    doc.create_with_guid(CHARACTER_DEMO_CAMERA_GUID, SpawnKind::Empty, "Camera", None);
    insert!(
        doc,
        CHARACTER_DEMO_CAMERA_GUID,
        Transform::from_translation(DVec3::new(6.0, 4.0, -8.0))
    );
    insert!(doc, CHARACTER_DEMO_CAMERA_GUID, Camera::default());

    doc.world_mut().propagate();
    doc.mark_saved();
    doc
}

/// The `(guid, class)` actor list for a headless Simulate of the character demo.
pub fn character_demo_actors() -> Vec<(Uuid, BlueprintClass)> {
    vec![(CHARACTER_DEMO_CHARACTER_GUID, character_demo_class())]
}

/// The repo-root `samples/character-demo/` directory.
pub fn character_demo_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../samples/character-demo")
}

/// Encode an [`AssetPayload`](inf_asset::AssetPayload) + write it with an inf_asset
/// sidecar stamped with a stable GUID (the character-demo asset-writing helper).
fn write_anim_asset<T: inf_asset::AssetPayload>(
    dir: &std::path::Path,
    file: &str,
    guid: Uuid,
    kind: inf_asset::AssetKind,
    payload: &T,
) -> Result<(), String> {
    let bytes = inf_asset::encode(payload).map_err(|e| format!("encode {file}: {e}"))?;
    let path = dir.join(file);
    std::fs::write(&path, &bytes).map_err(|e| format!("write {file}: {e}"))?;
    inf_asset::AssetSidecar::new(
        inf_asset::AssetId(guid),
        kind,
        inf_asset::ContentHash::of(&bytes),
    )
    .save(&path)
    .map_err(|e| format!("write {file} sidecar: {e}"))
}

/// Write the committed character-demo files from the generators (regeneration
/// path): the v5 `.inf_lvl` (+ sidecar), the `.inf_skel` / three `.inf_anim` /
/// `.inf_sm` anim assets (+ sidecars with their stable GUIDs), the `.inf_act`
/// actor (+ sidecar), and a README.
pub fn write_character_demo() -> Result<(), String> {
    use inf_anim::{AnimClipAsset, SkeletonAsset, StateMachineAsset};
    use inf_asset::AssetKind;

    let dir = character_demo_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("mkdir: {e}"))?;

    // Level (payload + sidecar).
    let doc = character_demo_scene();
    crate::scene::serialize::save(
        &doc,
        &dir.join("Character.inf_lvl"),
        Some(CHARACTER_DEMO_LEVEL_GUID),
    )?;

    // Skeleton.
    let skel_bytes_ref = *CHARACTER_DEMO_SKELETON_GUID.as_bytes();
    write_anim_asset(
        &dir,
        "Character.inf_skel",
        CHARACTER_DEMO_SKELETON_GUID,
        inf_asset::AssetKind::Skeleton,
        &SkeletonAsset::new(character_demo_skeleton()),
    )?;

    // Three clips (each bound to the skeleton GUID — a dep edge).
    for (file, guid, clip) in [
        (
            "Idle.inf_anim",
            CHARACTER_DEMO_IDLE_CLIP_GUID,
            character_demo_idle_clip(),
        ),
        (
            "Run.inf_anim",
            CHARACTER_DEMO_RUN_CLIP_GUID,
            character_demo_run_clip(),
        ),
        (
            "Jump.inf_anim",
            CHARACTER_DEMO_JUMP_CLIP_GUID,
            character_demo_jump_clip(),
        ),
    ] {
        write_anim_asset(
            &dir,
            file,
            guid,
            inf_asset::AssetKind::AnimClip,
            &AnimClipAsset::new(clip, Some(skel_bytes_ref)),
        )?;
    }

    // State machine (bound to the skeleton GUID; references the clip GUIDs).
    write_anim_asset(
        &dir,
        "Locomotion.inf_sm",
        CHARACTER_DEMO_SM_GUID,
        inf_asset::AssetKind::StateMachine,
        &StateMachineAsset::new(character_demo_state_machine(), Some(skel_bytes_ref)),
    )?;

    // Actor blueprint (JSON, like Coyote) + its inf_asset sidecar.
    let class = character_demo_class();
    let act_bytes = encode_actor(&class)?;
    let act_path = dir.join("Character.inf_act");
    std::fs::write(&act_path, &act_bytes).map_err(|e| format!("write actor: {e}"))?;
    inf_asset::AssetSidecar::new(
        inf_asset::AssetId(CHARACTER_DEMO_ACTOR_GUID),
        AssetKind::Blueprint,
        inf_asset::ContentHash::of(&act_bytes),
    )
    .save(&act_path)
    .map_err(|e| format!("write actor sidecar: {e}"))?;

    std::fs::write(dir.join("README.md"), CHARACTER_DEMO_README)
        .map_err(|e| format!("write readme: {e}"))?;
    Ok(())
}

const CHARACTER_DEMO_README: &str = "# Character Demo (Phase-11 gate scene)\n\n\
Generated by `inf_editor_core::samples::character_demo_scene` — the P11.4 gate\n\
scene: an idle/run/jump **state-machine character** driven by a Blueprint across a\n\
sine-hill terrain (the P10→P11 capstone).\n\n\
- `Character.inf_lvl` — the scene as schema-v5 `.inf_lvl` bytes (SkeletalMesh +\n\
  AnimStateMachine + RootMotion + a 3D character controller persist).\n\
- `Character.inf_skel` — a procedural 6-joint humanoid-ish skeleton.\n\
- `Idle/Run/Jump.inf_anim` — three programmatic clips (bob / forward root motion /\n\
  vertical arc).\n\
- `Locomotion.inf_sm` — the state machine (idle→run on speed>0.1, run→idle on ≤0.1,\n\
  any→jump on jump>0.5 with exit back).\n\
- `Character.inf_act` — the actor blueprint (left/right → move_and_slide across the\n\
  terrain via `terrain.height_at`; jump on the rising edge).\n\n\
The terrain heights are `character_demo_height(x, z)`; the runtime gate scripts\n\
input and asserts the character crosses the terrain (x advances, Y tracks the\n\
height), jumps (Y rises then returns), and its state machine transitions\n\
idle→run→jump. PIE == shipping (identical trace/probes on both paths).\n\n\
Regenerate with `INF_BLESS_SAMPLES=1 cargo test -p inf-editor-core samples`.\n";

// ── Physics-playground gate scene (P12.4) ────────────────────────────────────
//
// The Phase-12-closing gate scene: a committed 3D physics playground composing
// every P12 feature at once — box stacks, a motorized revolute spinner, a
// distance-rope pendulum, a prismatic slider, a CCD bullet vs a thin wall, a
// collision-layer ghost pair, a sensor plate, and a small **ragdoll**
// (`inf_physics::ragdoll::build_ragdoll` output, its descs mapped to `Joint3D`
// components) — plus TWO spatial `AudioSource`s (one autoplay-looping on the
// spinner with occlusion, one on the sensor plate) and an `AudioListener` on a
// camera. All persisted as schema-v6 `.inf_lvl` bytes under
// `samples/physics-playground/`, with the two `.inf_audio` clips beside it.
//
// The in-code determinism guarantee is `inf-physics`'s
// `playground_determinism.rs`; this is the same composition as **committed
// content**, run through the cook/PIE pipeline by the runtime gate
// (`runtime/inf-player/tests/physics_demo.rs`): 300 fixed steps twice yields a
// byte-identical pose trace + identical audio command stream, PIE == shipping.
//
// Joints/audio persist because P12.4 bumped the `.inf_lvl` schema to v6 (the
// `joint_2d`/`joint_3d`/`audio_source`/`audio_listener` slots); the physics bridge
// reconciles the joints from the ECS components each step, so authoring
// `RigidBody3D` + `Collider3D` + `Joint3D` is sufficient to simulate them.

pub const PLAYGROUND_LEVEL_GUID: Uuid = Uuid::from_u128(0x8405_0000);
pub const PLAYGROUND_GROUND_GUID: Uuid = Uuid::from_u128(0x8405_0001);
pub const PLAYGROUND_SPINNER_HUB_GUID: Uuid = Uuid::from_u128(0x8405_0020);
pub const PLAYGROUND_SPINNER_WHEEL_GUID: Uuid = Uuid::from_u128(0x8405_0021);
pub const PLAYGROUND_PENDULUM_ANCHOR_GUID: Uuid = Uuid::from_u128(0x8405_0030);
pub const PLAYGROUND_PENDULUM_BOB_GUID: Uuid = Uuid::from_u128(0x8405_0031);
pub const PLAYGROUND_SLIDER_RAIL_GUID: Uuid = Uuid::from_u128(0x8405_0040);
pub const PLAYGROUND_SLIDER_GUID: Uuid = Uuid::from_u128(0x8405_0041);
pub const PLAYGROUND_BULLET_WALL_GUID: Uuid = Uuid::from_u128(0x8405_0050);
pub const PLAYGROUND_BULLET_GUID: Uuid = Uuid::from_u128(0x8405_0051);
pub const PLAYGROUND_GHOST_A_GUID: Uuid = Uuid::from_u128(0x8405_0060);
pub const PLAYGROUND_GHOST_B_GUID: Uuid = Uuid::from_u128(0x8405_0061);
pub const PLAYGROUND_SENSOR_GUID: Uuid = Uuid::from_u128(0x8405_0070);
pub const PLAYGROUND_SENSOR_PROBE_GUID: Uuid = Uuid::from_u128(0x8405_0071);
pub const PLAYGROUND_CAMERA_GUID: Uuid = Uuid::from_u128(0x8405_0090);
/// First box of the stack (subsequent boxes are `+ i`).
pub const PLAYGROUND_STACK_BASE_GUID: u128 = 0x8405_0010;
/// First ragdoll part (subsequent parts are `+ i`, in `build_ragdoll` order).
pub const PLAYGROUND_RAGDOLL_BASE_GUID: u128 = 0x8405_0080;
/// The two committed `.inf_audio` clip asset GUIDs (stable so the AudioSource
/// `clip` refs resolve through the AssetDb / cooked pack, and the cook's
/// level→audio dep edge ships them).
pub const PLAYGROUND_SPINNER_CLIP_GUID: Uuid = Uuid::from_u128(0x8405_00A0);
pub const PLAYGROUND_SENSOR_CLIP_GUID: Uuid = Uuid::from_u128(0x8405_00A1);

/// The number of dynamic boxes in the settling stack.
pub const PLAYGROUND_STACK_COUNT: usize = 5;

/// Map a ragdoll [`inf_physics::d3::JointDesc3D`] onto a persisted [`Joint3D`]
/// component (the "descs mapped to components" step). The other body is `other`.
fn joint3d_from_ragdoll(
    other: Uuid,
    desc: inf_physics::d3::JointDesc3D,
) -> inf_ecs::components::Joint3D {
    use inf_ecs::components::{Joint3D, JointKind3D as EK};
    use inf_physics::d3::JointKind3D as PK;
    let mut j = Joint3D {
        other: inf_ecs::EntityRef::new(other),
        local_anchor: desc.local_anchor1.into(),
        other_anchor: desc.local_anchor2.into(),
        ..Default::default()
    };
    match desc.kind {
        PK::Fixed => j.kind = EK::Fixed,
        PK::Spherical => j.kind = EK::Spherical,
        PK::Distance { max_distance } => {
            j.kind = EK::Distance;
            j.max_distance = max_distance;
        }
        PK::Revolute {
            axis,
            limits,
            motor,
        } => {
            j.kind = EK::Revolute;
            j.axis = axis.into();
            if let Some([lo, hi]) = limits {
                j.limits_enabled = true;
                j.limit_min = lo;
                j.limit_max = hi;
            }
            if let Some(m) = motor {
                j.motor_enabled = true;
                j.motor_target_pos = m.target_pos;
                j.motor_target_vel = m.target_vel;
                j.motor_stiffness = m.stiffness;
                j.motor_damping = m.damping;
                j.motor_max_force = m.max_force;
            }
        }
        PK::Prismatic {
            axis,
            limits,
            motor,
        } => {
            j.kind = EK::Prismatic;
            j.axis = axis.into();
            if let Some([lo, hi]) = limits {
                j.limits_enabled = true;
                j.limit_min = lo;
                j.limit_max = hi;
            }
            if let Some(m) = motor {
                j.motor_enabled = true;
                j.motor_target_pos = m.target_pos;
                j.motor_target_vel = m.target_vel;
                j.motor_stiffness = m.stiffness;
                j.motor_damping = m.damping;
                j.motor_max_force = m.max_force;
            }
        }
    }
    j
}

/// The small humanoid skeleton fed to [`build_ragdoll`] (world-space bone
/// endpoints of a figure standing at world x = 30). Names classify to Hips /
/// Spine / Chest / Head / UpperArm{L,R} / Thigh{L,R} → 8 bodies + 7 joints.
pub fn playground_ragdoll_skeleton() -> Vec<inf_physics::ragdoll::RagdollBone> {
    use inf_physics::ragdoll::RagdollBone;
    let x = 30.0;
    vec![
        RagdollBone::new("hips", DVec3::new(x, 2.0, 0.0), DVec3::new(x, 2.3, 0.0)),
        RagdollBone::new("spine", DVec3::new(x, 2.3, 0.0), DVec3::new(x, 2.7, 0.0)),
        RagdollBone::new("chest", DVec3::new(x, 2.7, 0.0), DVec3::new(x, 3.1, 0.0)),
        RagdollBone::new("head", DVec3::new(x, 3.1, 0.0), DVec3::new(x, 3.5, 0.0)),
        RagdollBone::new(
            "upperarm_l",
            DVec3::new(x - 0.1, 3.0, 0.0),
            DVec3::new(x - 0.6, 3.0, 0.0),
        ),
        RagdollBone::new(
            "upperarm_r",
            DVec3::new(x + 0.1, 3.0, 0.0),
            DVec3::new(x + 0.6, 3.0, 0.0),
        ),
        RagdollBone::new(
            "thigh_l",
            DVec3::new(x - 0.15, 2.0, 0.0),
            DVec3::new(x - 0.15, 1.3, 0.0),
        ),
        RagdollBone::new(
            "thigh_r",
            DVec3::new(x + 0.15, 2.0, 0.0),
            DVec3::new(x + 0.15, 1.3, 0.0),
        ),
    ]
}

/// Build the physics-playground [`SceneDoc`]. See the module note for the layout.
pub fn physics_playground_scene() -> SceneDoc {
    use inf_ecs::components::{
        AudioListener, AudioSource, BodyKind3D, Camera, Collider3D, ColliderShape3DKind,
        DistanceModel, Joint3D, JointKind3D, Light, LightKind, RigidBody3D,
    };
    use inf_ecs::math::Vec3d;
    use inf_physics::ragdoll::{build_ragdoll, RagdollConfig};

    let mut doc = SceneDoc::new();
    doc.set_title("Physics Playground");

    // Helpers to cut the boilerplate.
    let static_body = || RigidBody3D {
        kind: BodyKind3D::Static,
        ..Default::default()
    };
    let box_collider = |half: Vec3d| Collider3D {
        shape_kind: ColliderShape3DKind::Box,
        half_extents: half,
        ..Default::default()
    };

    // ── Ground slab (static box; top at y = 0). ──
    doc.create_with_guid(PLAYGROUND_GROUND_GUID, SpawnKind::Empty, "Ground", None);
    insert!(
        doc,
        PLAYGROUND_GROUND_GUID,
        Transform::from_translation(DVec3::new(0.0, -0.5, 0.0))
    );
    insert!(doc, PLAYGROUND_GROUND_GUID, static_body());
    insert!(
        doc,
        PLAYGROUND_GROUND_GUID,
        box_collider(Vec3d::new(48.0, 0.5, 48.0))
    );

    // ── A settling box stack (5 dynamic boxes at x = 0). ──
    for i in 0..PLAYGROUND_STACK_COUNT {
        let guid = Uuid::from_u128(PLAYGROUND_STACK_BASE_GUID + i as u128);
        doc.create_with_guid(guid, SpawnKind::Empty, &format!("Box {i}"), None);
        insert!(
            doc,
            guid,
            Transform::from_translation(DVec3::new(0.0, 0.5 + i as f64 * 1.02, 0.0))
        );
        insert!(
            doc,
            guid,
            RigidBody3D {
                kind: BodyKind3D::Dynamic,
                ..Default::default()
            }
        );
        insert!(
            doc,
            guid,
            Collider3D {
                shape_kind: ColliderShape3DKind::Box,
                half_extents: Vec3d::splat(0.5),
                friction: 0.7,
                ..Default::default()
            }
        );
    }

    // ── A motorized revolute spinner (hub static, wheel dynamic, at x = 8). The
    //    wheel also carries the autoplay-looping, occluded spatial AudioSource. ──
    doc.create_with_guid(
        PLAYGROUND_SPINNER_HUB_GUID,
        SpawnKind::Empty,
        "Spinner Hub",
        None,
    );
    insert!(
        doc,
        PLAYGROUND_SPINNER_HUB_GUID,
        Transform::from_translation(DVec3::new(8.0, 4.0, 0.0))
    );
    insert!(
        doc,
        PLAYGROUND_SPINNER_HUB_GUID,
        RigidBody3D {
            kind: BodyKind3D::Static,
            ..Default::default()
        }
    );
    doc.create_with_guid(
        PLAYGROUND_SPINNER_WHEEL_GUID,
        SpawnKind::Empty,
        "Spinner Wheel",
        None,
    );
    insert!(
        doc,
        PLAYGROUND_SPINNER_WHEEL_GUID,
        Transform::from_translation(DVec3::new(8.0, 4.0, 0.0))
    );
    insert!(
        doc,
        PLAYGROUND_SPINNER_WHEEL_GUID,
        RigidBody3D {
            kind: BodyKind3D::Dynamic,
            ..Default::default()
        }
    );
    insert!(
        doc,
        PLAYGROUND_SPINNER_WHEEL_GUID,
        Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: Vec3d::splat(0.4),
            ..Default::default()
        }
    );
    insert!(
        doc,
        PLAYGROUND_SPINNER_WHEEL_GUID,
        Joint3D {
            other: inf_ecs::EntityRef::new(PLAYGROUND_SPINNER_HUB_GUID),
            kind: JointKind3D::Revolute,
            axis: Vec3d::new(0.0, 0.0, 1.0),
            motor_enabled: true,
            motor_target_vel: 8.0,
            motor_damping: 1.0,
            ..Default::default()
        }
    );
    insert!(
        doc,
        PLAYGROUND_SPINNER_WHEEL_GUID,
        AudioSource {
            clip: Some(PLAYGROUND_SPINNER_CLIP_GUID),
            bus: "sfx".to_string(),
            volume: 0.8,
            pitch: 1.0,
            looping: true,
            spatial: true,
            min_distance: 2.0,
            max_distance: 40.0,
            distance_model: DistanceModel::Inverse,
            rolloff: 1.0,
            occlusion: true,
            autoplay: true,
        }
    );

    // ── A distance-rope pendulum (anchor static, bob dynamic, at x = -8). ──
    doc.create_with_guid(
        PLAYGROUND_PENDULUM_ANCHOR_GUID,
        SpawnKind::Empty,
        "Rope Anchor",
        None,
    );
    insert!(
        doc,
        PLAYGROUND_PENDULUM_ANCHOR_GUID,
        Transform::from_translation(DVec3::new(-8.0, 6.0, 0.0))
    );
    insert!(
        doc,
        PLAYGROUND_PENDULUM_ANCHOR_GUID,
        RigidBody3D {
            kind: BodyKind3D::Static,
            ..Default::default()
        }
    );
    doc.create_with_guid(
        PLAYGROUND_PENDULUM_BOB_GUID,
        SpawnKind::Empty,
        "Rope Bob",
        None,
    );
    insert!(
        doc,
        PLAYGROUND_PENDULUM_BOB_GUID,
        // Offset horizontally so the taut rope swings (a real pendulum).
        Transform::from_translation(DVec3::new(-7.0, 4.8, 0.0))
    );
    insert!(
        doc,
        PLAYGROUND_PENDULUM_BOB_GUID,
        RigidBody3D {
            kind: BodyKind3D::Dynamic,
            ..Default::default()
        }
    );
    insert!(
        doc,
        PLAYGROUND_PENDULUM_BOB_GUID,
        Collider3D {
            shape_kind: ColliderShape3DKind::Sphere,
            radius: 0.3,
            ..Default::default()
        }
    );
    insert!(
        doc,
        PLAYGROUND_PENDULUM_BOB_GUID,
        Joint3D {
            other: inf_ecs::EntityRef::new(PLAYGROUND_PENDULUM_ANCHOR_GUID),
            kind: JointKind3D::Distance,
            max_distance: 1.5,
            ..Default::default()
        }
    );

    // ── A prismatic slider under gravity with limits (at x = 12). ──
    doc.create_with_guid(
        PLAYGROUND_SLIDER_RAIL_GUID,
        SpawnKind::Empty,
        "Slider Rail",
        None,
    );
    insert!(
        doc,
        PLAYGROUND_SLIDER_RAIL_GUID,
        Transform::from_translation(DVec3::new(12.0, 6.0, 0.0))
    );
    insert!(
        doc,
        PLAYGROUND_SLIDER_RAIL_GUID,
        RigidBody3D {
            kind: BodyKind3D::Static,
            ..Default::default()
        }
    );
    doc.create_with_guid(PLAYGROUND_SLIDER_GUID, SpawnKind::Empty, "Slider", None);
    insert!(
        doc,
        PLAYGROUND_SLIDER_GUID,
        Transform::from_translation(DVec3::new(12.0, 6.0, 0.0))
    );
    insert!(
        doc,
        PLAYGROUND_SLIDER_GUID,
        RigidBody3D {
            kind: BodyKind3D::Dynamic,
            ..Default::default()
        }
    );
    insert!(
        doc,
        PLAYGROUND_SLIDER_GUID,
        Collider3D {
            shape_kind: ColliderShape3DKind::Sphere,
            radius: 0.3,
            ..Default::default()
        }
    );
    insert!(
        doc,
        PLAYGROUND_SLIDER_GUID,
        Joint3D {
            other: inf_ecs::EntityRef::new(PLAYGROUND_SLIDER_RAIL_GUID),
            kind: JointKind3D::Prismatic,
            axis: Vec3d::new(0.0, 1.0, 0.0),
            limits_enabled: true,
            limit_min: -2.0,
            limit_max: 0.0,
            ..Default::default()
        }
    );

    // ── A CCD bullet aimed (by fast gravity) at a thin horizontal wall (x = 24).
    //    Without CCD the fast body tunnels the 0.04-thick plate; with it, it stops. ──
    doc.create_with_guid(
        PLAYGROUND_BULLET_WALL_GUID,
        SpawnKind::Empty,
        "Thin Wall",
        None,
    );
    insert!(
        doc,
        PLAYGROUND_BULLET_WALL_GUID,
        Transform::from_translation(DVec3::new(24.0, 5.0, 0.0))
    );
    insert!(
        doc,
        PLAYGROUND_BULLET_WALL_GUID,
        RigidBody3D {
            kind: BodyKind3D::Static,
            ..Default::default()
        }
    );
    insert!(
        doc,
        PLAYGROUND_BULLET_WALL_GUID,
        box_collider(Vec3d::new(2.0, 0.02, 2.0))
    );
    doc.create_with_guid(PLAYGROUND_BULLET_GUID, SpawnKind::Empty, "CCD Bullet", None);
    insert!(
        doc,
        PLAYGROUND_BULLET_GUID,
        Transform::from_translation(DVec3::new(24.0, 22.0, 0.0))
    );
    insert!(
        doc,
        PLAYGROUND_BULLET_GUID,
        RigidBody3D {
            kind: BodyKind3D::Dynamic,
            // A heavy gravity scale accelerates it to a tunnelling speed fast.
            gravity_scale: 12.0,
            ccd_enabled: true,
            ..Default::default()
        }
    );
    insert!(
        doc,
        PLAYGROUND_BULLET_GUID,
        Collider3D {
            shape_kind: ColliderShape3DKind::Sphere,
            radius: 0.15,
            ..Default::default()
        }
    );

    // ── A collision-layer ghost PAIR: two dynamic spheres at the SAME point, each
    //    with an empty collision filter (interact with nothing) → they free-fall in
    //    lockstep, perfectly co-located, interpenetrating unimpeded (the layer
    //    proof: were the filters non-empty the contact solver would shove them
    //    apart, and the floor would stop them). ──
    for (guid, name) in [
        (PLAYGROUND_GHOST_A_GUID, "Ghost A"),
        (PLAYGROUND_GHOST_B_GUID, "Ghost B"),
    ] {
        doc.create_with_guid(guid, SpawnKind::Empty, name, None);
        insert!(
            doc,
            guid,
            Transform::from_translation(DVec3::new(-16.0, 8.0, 0.0))
        );
        insert!(
            doc,
            guid,
            RigidBody3D {
                kind: BodyKind3D::Dynamic,
                ..Default::default()
            }
        );
        insert!(
            doc,
            guid,
            Collider3D {
                shape_kind: ColliderShape3DKind::Sphere,
                radius: 0.4,
                // Membership present but an EMPTY filter → collides with nothing.
                collision_memberships: 0b10,
                collision_filter: 0,
                ..Default::default()
            }
        );
    }

    // ── A sensor plate (static trigger volume, x = 16) with the second AudioSource,
    //    plus a probe ball that falls THROUGH it (a sensor generates no force) and
    //    lands on the ground — proving the plate is non-blocking. ──
    doc.create_with_guid(
        PLAYGROUND_SENSOR_GUID,
        SpawnKind::Empty,
        "Sensor Plate",
        None,
    );
    insert!(
        doc,
        PLAYGROUND_SENSOR_GUID,
        Transform::from_translation(DVec3::new(16.0, 1.0, 0.0))
    );
    insert!(
        doc,
        PLAYGROUND_SENSOR_GUID,
        RigidBody3D {
            kind: BodyKind3D::Static,
            ..Default::default()
        }
    );
    insert!(
        doc,
        PLAYGROUND_SENSOR_GUID,
        Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: Vec3d::new(1.5, 0.5, 1.5),
            sensor: true,
            ..Default::default()
        }
    );
    insert!(
        doc,
        PLAYGROUND_SENSOR_GUID,
        AudioSource {
            clip: Some(PLAYGROUND_SENSOR_CLIP_GUID),
            bus: "sfx".to_string(),
            volume: 0.6,
            pitch: 1.0,
            looping: false,
            spatial: true,
            min_distance: 1.0,
            max_distance: 30.0,
            distance_model: DistanceModel::Inverse,
            rolloff: 1.0,
            occlusion: false,
            autoplay: true,
        }
    );
    doc.create_with_guid(
        PLAYGROUND_SENSOR_PROBE_GUID,
        SpawnKind::Empty,
        "Sensor Probe",
        None,
    );
    insert!(
        doc,
        PLAYGROUND_SENSOR_PROBE_GUID,
        Transform::from_translation(DVec3::new(16.0, 5.0, 0.0))
    );
    insert!(
        doc,
        PLAYGROUND_SENSOR_PROBE_GUID,
        RigidBody3D {
            kind: BodyKind3D::Dynamic,
            ..Default::default()
        }
    );
    insert!(
        doc,
        PLAYGROUND_SENSOR_PROBE_GUID,
        Collider3D {
            shape_kind: ColliderShape3DKind::Sphere,
            radius: 0.25,
            ..Default::default()
        }
    );

    // ── A small ragdoll from `build_ragdoll` (its descs mapped to components). ──
    let parts = build_ragdoll(&playground_ragdoll_skeleton(), RagdollConfig::default());
    // Stable per-part GUIDs (index order == build_ragdoll's parents-first order).
    let part_guid = |i: usize| Uuid::from_u128(PLAYGROUND_RAGDOLL_BASE_GUID + i as u128);
    for (i, part) in parts.iter().enumerate() {
        let guid = part_guid(i);
        doc.create_with_guid(
            guid,
            SpawnKind::Empty,
            &format!("Ragdoll {}", part.name),
            None,
        );
        let mut t = Transform::from_translation(part.position);
        t.set_quat(part.rotation);
        insert!(doc, guid, t);
        insert!(
            doc,
            guid,
            RigidBody3D {
                kind: BodyKind3D::Dynamic,
                ..Default::default()
            }
        );
        // The capsule spanning the bone (build_ragdoll always emits a Capsule).
        //
        // **The collision layers ride along** (island phase, IB-10). They did
        // not, and `..Default::default()` quietly reset them to "member of every
        // layer, collides with every layer" — so the *persisted* ragdoll got
        // P29.6's anchor and density fixes and neither of its collision ones. A
        // ragdoll's limb capsules overlap by construction (adjacent bones share
        // an endpoint), so limbs that push each other apart are a permanent
        // depenetration force inside the body: measured on the P29.6 course, a
        // settled pelvis climbed 14 cm per fixed step and rose ten metres.
        //
        // MEASURED HERE, and it decided this wave's schema window:
        // `inf-physics`'s `the_persisted_ragdolls_two_missing_fixes_are_not_equally_load_bearing`
        // reports **1.339 m** of divergence from dropping the layer mask and
        // **0.000000 m** from dropping `JointDesc3D::contacts` while the mask is
        // kept. The mask subsumes the flag — limbs that cannot collide at all do
        // not need contacts disabled between the jointed pairs — so `Joint3D`
        // does NOT owe a `contacts` field, and the whole of this gap closes in
        // the generator with no schema move.
        if let inf_physics::ColliderShape3D::Capsule {
            half_height,
            radius,
        } = part.collider.shape
        {
            insert!(
                doc,
                guid,
                Collider3D {
                    shape_kind: ColliderShape3DKind::Capsule,
                    half_extents: Vec3d::new(radius, half_height, radius),
                    radius,
                    density: part.collider.density,
                    friction: part.collider.friction,
                    collision_memberships: part.collider.layers.memberships,
                    collision_filter: part.collider.layers.filter,
                    ..Default::default()
                }
            );
        }
        // The joint to the parent part (root has none).
        if let Some(rj) = &part.joint {
            let other = part_guid(rj.parent);
            insert!(doc, guid, joint3d_from_ragdoll(other, rj.desc));
        }
    }

    // ── A camera carrying the active AudioListener (the sim reads its pose). ──
    doc.create_with_guid(PLAYGROUND_CAMERA_GUID, SpawnKind::Empty, "Camera", None);
    insert!(
        doc,
        PLAYGROUND_CAMERA_GUID,
        Transform::from_translation(DVec3::new(0.0, 6.0, -15.0))
    );
    insert!(doc, PLAYGROUND_CAMERA_GUID, Camera::default());
    insert!(doc, PLAYGROUND_CAMERA_GUID, AudioListener { active: true });

    // A directional sun so the playground reads (rendering is human-verified).
    let sun = Uuid::from_u128(0x8405_0002);
    doc.create_with_guid(sun, SpawnKind::Empty, "Sun", None);
    insert!(
        doc,
        sun,
        Transform {
            translation: Vec3d::new(0.0, 30.0, 0.0),
            rotation: Vec3d::new(-50.0, -30.0, 0.0),
            scale: Vec3d::ONE,
        }
    );
    insert!(
        doc,
        sun,
        Light {
            kind: LightKind::Directional,
            color: Color::new(1.0, 0.97, 0.9, 1.0),
            intensity: 2.5,
            ..Default::default()
        }
    );

    // The 3D physics gravity flows from `gravity_2d.y` (the runtime sim wires the
    // 3D bridge to it), so the playground makes it explicit real-world down.
    doc.set_settings(crate::scene::serialize::LevelSettings {
        gravity_2d: Vec2d::new(0.0, -9.81),
        gravity_3d: Vec3d::new(0.0, -9.81, 0.0),
        sim_hz: 60.0,
        ..Default::default()
    });

    doc.world_mut().propagate();
    doc.mark_saved();
    doc
}

/// The repo-root `samples/physics-playground/` directory.
pub fn physics_playground_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../samples/physics-playground")
}

/// A minimal valid 16-bit mono PCM WAV (decodable by kira headlessly), so the
/// committed `.inf_audio` clips need no binary fixture. Ported from the
/// `inf-audio` payload test's `tone_wav`.
fn tone_wav(samples: usize, sample_rate: u32) -> Vec<u8> {
    let bits = 16u16;
    let channels = 1u16;
    let block_align = channels * bits / 8;
    let byte_rate = sample_rate * block_align as u32;
    let data_len = samples as u32 * block_align as u32;
    let mut w = Vec::new();
    w.extend_from_slice(b"RIFF");
    w.extend_from_slice(&(36 + data_len).to_le_bytes());
    w.extend_from_slice(b"WAVE");
    w.extend_from_slice(b"fmt ");
    w.extend_from_slice(&16u32.to_le_bytes());
    w.extend_from_slice(&1u16.to_le_bytes());
    w.extend_from_slice(&channels.to_le_bytes());
    w.extend_from_slice(&sample_rate.to_le_bytes());
    w.extend_from_slice(&byte_rate.to_le_bytes());
    w.extend_from_slice(&block_align.to_le_bytes());
    w.extend_from_slice(&bits.to_le_bytes());
    w.extend_from_slice(b"data");
    w.extend_from_slice(&data_len.to_le_bytes());
    for i in 0..samples {
        let s = (i as i16).wrapping_mul(64);
        w.extend_from_slice(&s.to_le_bytes());
    }
    w
}

/// The committed `AudioAsset` for the given clip (a short deterministic tone).
pub fn playground_audio_asset() -> inf_audio::AudioAsset {
    inf_audio::AudioAsset::from_encoded(tone_wav(4000, 8000), inf_audio::AudioFormat::Wav)
        .expect("tone wav decodes")
}

/// Write the committed physics-playground files from the generators (regeneration
/// path): the v6 `.inf_lvl` (+ sidecar), the two `.inf_audio` clips (+ inf_asset
/// sidecars with their stable GUIDs so the AudioSource `clip` refs resolve through
/// the AssetDb / cooked pack), + a README.
pub fn write_physics_playground() -> Result<(), String> {
    let dir = physics_playground_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("mkdir: {e}"))?;

    // Level (payload + sidecar) with a fixed level GUID.
    let doc = physics_playground_scene();
    crate::scene::serialize::save(
        &doc,
        &dir.join("Playground.inf_lvl"),
        Some(PLAYGROUND_LEVEL_GUID),
    )?;

    // The two committed `.inf_audio` clips (same tone; distinct stable GUIDs).
    let audio = playground_audio_asset();
    for (file, guid) in [
        ("Spinner.inf_audio", PLAYGROUND_SPINNER_CLIP_GUID),
        ("Sensor.inf_audio", PLAYGROUND_SENSOR_CLIP_GUID),
    ] {
        write_anim_asset(&dir, file, guid, inf_asset::AssetKind::Audio, &audio)?;
    }

    std::fs::write(dir.join("README.md"), PLAYGROUND_README)
        .map_err(|e| format!("write readme: {e}"))?;
    Ok(())
}

const PLAYGROUND_README: &str = "# Physics Playground (Phase-12 gate scene)\n\n\
Generated by `inf_editor_core::samples::physics_playground_scene` — the P12.4 gate\n\
scene: a committed 3D physics playground composing every P12 feature at once — a\n\
settling **box stack**, a **motorized revolute spinner**, a **distance-rope\n\
pendulum**, a **prismatic slider**, a **CCD bullet** vs a thin wall, a\n\
**collision-layer ghost pair** (overlapping, non-interacting via filters), a\n\
**sensor plate**, and a small **ragdoll** (`inf_physics::ragdoll::build_ragdoll`\n\
output, its joint descs mapped to `Joint3D` components) — plus two spatial\n\
**AudioSource**s (one autoplay-looping on the spinner with occlusion, one on the\n\
sensor) and an **AudioListener** on a camera.\n\n\
- `Playground.inf_lvl` — the scene as schema-**v6** `.inf_lvl` bytes (joints +\n\
  audio persist).\n\
- `Spinner.inf_audio` / `Sensor.inf_audio` — the two clips the AudioSources\n\
  reference (a short deterministic tone; shipped via the cook's level→audio edge).\n\n\
Determinism is asserted by `runtime/inf-player/tests/physics_demo.rs`: 300 fixed\n\
steps twice yield a byte-identical pose trace (xxh3 over Guid-sorted transforms)\n\
AND an identical audio command stream; the ragdoll settles bounded, the CCD bullet\n\
is stopped, the ghost pair interpenetrates unimpeded, PIE == shipping.\n\n\
Regenerate with `INF_BLESS_SAMPLES=1 cargo test -p inf-editor-core samples`.\n";

// ── Phase 13 gate: the virtualized-geometry demo (P13.4) ─────────────────────
//
// The generator builds ONE dense displaced mesh asset (~33k triangles) and
// instances it `GRID × GRID` times across an XZ plane, so the **source** triangle
// count across instances exceeds 10M while the committed `.inf_lvl` stays tiny
// (every instance references the single `.inf_mesh` by GUID). The cook derives the
// `.inf_vmesh` meshlet DAG from that mesh (via the P13.4 `MeshRef.asset` →
// dependency-closure edge); the player renders it through the GPU meshlet path
// (vgeom on) or the classic discrete-LOD fallback (vgeom off / a lower render
// tier). The gate lives in `runtime/inf-player/tests/vgeom_gate.rs`.

/// Stable GUID of the vgeom-demo level.
pub const VGEOM_DEMO_LEVEL_GUID: Uuid = Uuid::from_u128(0x8406_0001_0000_0000_0000_0000_0000_0001);
/// Stable GUID of the shared dense `.inf_mesh` asset every instance references.
pub const VGEOM_DEMO_MESH_GUID: Uuid = Uuid::from_u128(0x8406_0002_0000_0000_0000_0000_0000_0002);
/// Stable GUID of the directional sun.
pub const VGEOM_DEMO_SUN_GUID: Uuid = Uuid::from_u128(0x8406_0003_0000_0000_0000_0000_0000_0003);
/// Base GUID for the grid instance entities (add the flat index).
const VGEOM_DEMO_INSTANCE_BASE: u128 = 0x8406_0100_0000_0000_0000_0000_0000_0000;

/// Grid subdivisions of the dense mesh — `2·N²` triangles (128 → 32 768 tris,
/// well over the cook's 2048 `min_triangles` vmesh threshold).
pub const VGEOM_DEMO_MESH_N: usize = 128;
/// Instance grid side: `GRID × GRID` placed copies of the dense mesh.
pub const VGEOM_DEMO_GRID: usize = 18;

/// Total **source** triangles across every instance (`2·N² · GRID²`). Exceeds the
/// phase gate's 10M requirement (128/18 → 10 616 832).
pub const fn vgeom_demo_source_triangles() -> u64 {
    (2 * VGEOM_DEMO_MESH_N * VGEOM_DEMO_MESH_N * VGEOM_DEMO_GRID * VGEOM_DEMO_GRID) as u64
}

/// A byte-portable sine: std `f32::sin` is NOT bit-identical across libms (MSVC
/// vs glibc diverged on the Ubuntu CI runner — the generator-lock caught it), so
/// the displacement uses this pure-arithmetic minimax polynomial instead. IEEE
/// f32 add/mul/floor are exactly specified, so the committed mesh bytes are
/// identical on every platform.
fn psin(x: f32) -> f32 {
    use std::f32::consts::TAU;
    // Range-reduce to [-π, π] (floor is exact; inputs here are small, no
    // catastrophic cancellation at the scales the generator uses).
    let x = x - (x / TAU + 0.5).floor() * TAU;
    // Odd 7th-order minimax on [-π, π] (~1e-4 abs error — far below visual or
    // meshlet-build significance, and perfectly reproducible).
    let x2 = x * x;
    x * (0.987_862 + x2 * (-0.155_271 + x2 * (0.005_641_12 - x2 * 0.000_060_461_2)))
}

/// Byte-portable cosine via [`psin`].
fn pcos(x: f32) -> f32 {
    psin(x + std::f32::consts::FRAC_PI_2)
}

/// One dense displaced-grid [`inf_mesh::MeshAsset`] (`2·N²` triangles) — the shared
/// asset every instance references. A deterministic function of `N` built on
/// byte-portable arithmetic ([`psin`]/[`pcos`]), so the committed `.inf_mesh` is
/// reproducible on every platform.
pub fn vgeom_demo_mesh() -> inf_mesh::MeshAsset {
    let n = VGEOM_DEMO_MESH_N;
    let mut vertices = Vec::with_capacity((n + 1) * (n + 1));
    for j in 0..=n {
        for i in 0..=n {
            let u = i as f32 / n as f32;
            let v = j as f32 / n as f32;
            let x = (u - 0.5) * 2.0;
            let z = (v - 0.5) * 2.0;
            let y = 0.3 * psin(x * 3.0) * pcos(z * 3.0);
            let nrm = glam::Vec3::new(
                -0.9 * pcos(x * 3.0) * pcos(z * 3.0),
                1.0,
                0.9 * psin(x * 3.0) * psin(z * 3.0),
            )
            .normalize();
            vertices.push(inf_mesh::MeshVertex {
                position: [x, y, z],
                normal: nrm.to_array(),
                uv: [u, v],
                tangent: [1.0, 0.0, 0.0, 1.0],
            });
        }
    }
    let stride = (n + 1) as u32;
    let mut indices = Vec::with_capacity(n * n * 6);
    for j in 0..n as u32 {
        for i in 0..n as u32 {
            let a = j * stride + i;
            indices.extend_from_slice(&[a, a + stride, a + 1, a + 1, a + stride, a + stride + 1]);
        }
    }
    let submesh = inf_mesh::SubMesh {
        name: "dense".into(),
        vertices,
        indices,
        material_slot: Some(0),
        skin: Vec::new(),
    };
    inf_mesh::MeshAsset::new(vec![submesh], vec!["Default".into()])
}

/// The vgeom-demo scene: `GRID × GRID` instances of the dense mesh asset spread
/// across an XZ plane, each an entity with a [`MeshRef`] whose `asset` points at
/// [`VGEOM_DEMO_MESH_GUID`] (plus a placeholder `Cube` primitive for the editor
/// viewport, which cannot render asset geometry yet), + a sun. The `.inf_lvl`
/// stays small (one asset, many light instance records).
pub fn vgeom_demo_scene() -> SceneDoc {
    use inf_ecs::components::{Light, LightKind, Material, MeshRef, Primitive, Transform};
    use inf_ecs::math::{Color, Vec3d};

    let mut doc = SceneDoc::new();
    doc.set_title("Vgeom Demo");

    // Sun.
    doc.create_with_guid(VGEOM_DEMO_SUN_GUID, SpawnKind::Empty, "Sun", None);
    insert!(
        doc,
        VGEOM_DEMO_SUN_GUID,
        Transform {
            translation: Vec3d::ZERO,
            rotation: Vec3d::new(-50.0, -30.0, 0.0),
            scale: Vec3d::ONE,
        }
    );
    insert!(
        doc,
        VGEOM_DEMO_SUN_GUID,
        Light {
            kind: LightKind::Directional,
            color: Color::WHITE,
            intensity: 3.0,
            ..Default::default()
        }
    );

    // The instance grid. `spacing` tiles the 2-unit-wide meshes edge-to-edge with a
    // small gap, spreading them over the XZ plane so a ground-level camera sees only
    // a small fraction (frustum + LOD) — the cull-ratio gate.
    let grid = VGEOM_DEMO_GRID;
    let spacing = 2.4f64;
    let offset = (grid as f64 - 1.0) * 0.5 * spacing;
    for j in 0..grid {
        for i in 0..grid {
            let idx = (j * grid + i) as u128;
            let guid = Uuid::from_u128(VGEOM_DEMO_INSTANCE_BASE + idx);
            let name = format!("Tile {i}x{j}");
            doc.create_with_guid(guid, SpawnKind::Empty, &name, None);
            insert!(
                doc,
                guid,
                Transform {
                    translation: Vec3d::new(
                        i as f64 * spacing - offset,
                        0.0,
                        j as f64 * spacing - offset,
                    ),
                    rotation: Vec3d::ZERO,
                    scale: Vec3d::ONE,
                }
            );
            insert!(
                doc,
                guid,
                MeshRef {
                    primitive: Primitive::Cube,
                    asset: Some(VGEOM_DEMO_MESH_GUID),
                }
            );
            // A subtle per-tile tint so the instances read as distinct content.
            let t = idx as f32 / (grid * grid) as f32;
            insert!(
                doc,
                guid,
                Material {
                    base_color: Color::new(0.45 + 0.3 * t, 0.5, 0.65 - 0.3 * t, 1.0),
                    metallic: 0.0,
                    roughness: 0.7,
                    emissive: Color::new(0.0, 0.0, 0.0, 1.0),
                    ..Default::default()
                }
            );
        }
    }

    doc.world_mut().propagate();
    doc.mark_saved();
    doc
}

/// The repo-root `samples/vgeom-demo/` directory.
pub fn vgeom_demo_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../samples/vgeom-demo")
}

/// Write the committed vgeom-demo files from the generators (regeneration path):
/// the `.inf_lvl` (+ sidecar), the dense `.inf_mesh` (+ its inf_asset sidecar so
/// the `MeshRef.asset` refs resolve through the AssetDb / cooked pack), + README.
pub fn write_vgeom_demo() -> Result<(), String> {
    let dir = vgeom_demo_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("mkdir: {e}"))?;

    let doc = vgeom_demo_scene();
    crate::scene::serialize::save(
        &doc,
        &dir.join("VgeomDemo.inf_lvl"),
        Some(VGEOM_DEMO_LEVEL_GUID),
    )?;

    // The dense `.inf_mesh` payload + its inf_asset sidecar (STABLE mesh GUID the
    // level's MeshRef.asset points at; the cook derives its `.inf_vmesh` beside it).
    let mesh_bytes =
        inf_asset::encode(&vgeom_demo_mesh()).map_err(|e| format!("encode mesh: {e}"))?;
    let mesh_path = dir.join("Dense.inf_mesh");
    std::fs::write(&mesh_path, &mesh_bytes).map_err(|e| format!("write mesh: {e}"))?;
    inf_asset::AssetSidecar::new(
        inf_asset::AssetId(VGEOM_DEMO_MESH_GUID),
        inf_asset::AssetKind::Mesh,
        inf_asset::ContentHash::of(&mesh_bytes),
    )
    .save(&mesh_path)
    .map_err(|e| format!("write mesh sidecar: {e}"))?;

    std::fs::write(dir.join("README.md"), VGEOM_DEMO_README)
        .map_err(|e| format!("write readme: {e}"))?;
    Ok(())
}

const VGEOM_DEMO_README: &str = "# Vgeom Demo (Phase-13 gate scene)\n\n\
Generated by `inf_editor_core::samples::vgeom_demo_scene` — the P13.4 gate scene: a\n\
grid of `GRID × GRID` instances of ONE dense displaced mesh asset, so the SOURCE\n\
triangle count across instances exceeds 10M while the `.inf_lvl` stays tiny (every\n\
instance references the single `Dense.inf_mesh` by GUID).\n\n\
- `VgeomDemo.inf_lvl` — the scene (schema v7): instance transforms + `MeshRef.asset`\n\
  refs + materials + a sun.\n\
- `Dense.inf_mesh` — the shared ~33k-triangle displaced grid. The cook derives its\n\
  `.inf_vmesh` meshlet DAG (via the `MeshRef.asset` dependency-closure edge) and\n\
  ships both in the pack.\n\n\
The gate (`runtime/inf-player/tests/vgeom_gate.rs`): byte-identical save/reload; a\n\
cooked load where the total source triangles ≥ 10M and the GPU meshlet cull leaves\n\
only a small fraction of meshlets visible from a ground-level camera (deterministic\n\
across runs); the SAME pack with vgeom OFF renders through the classic discrete-LOD\n\
fallback (a far camera picks a coarser level than a near one); and the auto-tier\n\
disables vgeom on the Low tier. GPU parts skip cleanly with no adapter.\n\n\
Regenerate with `INF_BLESS_SAMPLES=1 cargo test -p inf-editor-core samples`.\n";

// -- Streamed-terrain gate scene (P16.3b2) -----------------------------------
//
// The camera-driven-streaming gate scene: a terrain that lives ENTIRELY in a
// `.inf_terrain` asset (the level carries an empty working set plus the asset
// ref), a "Walker" entity the sim scripts across it, a sun and a camera.
//
// The `.inf_lvl` is committed; the `.inf_terrain` is **generated into the
// fixture's content directory** by [`write_streamed_terrain_asset`] rather than
// committed, because it is ~100 KB of derived bytes a pure generator reproduces
// exactly (the same reasoning that keeps the vgeom demo's mesh small).

pub const STREAMED_TERRAIN_LEVEL_GUID: Uuid = Uuid::from_u128(0x8416_0000);
pub const STREAMED_TERRAIN_TERRAIN_GUID: Uuid = Uuid::from_u128(0x8416_0001);
pub const STREAMED_TERRAIN_WALKER_GUID: Uuid = Uuid::from_u128(0x8416_0002);
pub const STREAMED_TERRAIN_SUN_GUID: Uuid = Uuid::from_u128(0x8416_0003);
pub const STREAMED_TERRAIN_CAMERA_GUID: Uuid = Uuid::from_u128(0x8416_0004);
/// The **asset** GUID of the generated `World.inf_terrain`, stamped into its
/// inf_asset sidecar. Stable so the level's `Terrain.asset` ref resolves through
/// the AssetDb / the cooked pack, and the cook's level -> terrain edge ships it.
pub const STREAMED_TERRAIN_ASSET_GUID: Uuid = Uuid::from_u128(0x8416_00AA);

/// Samples per tile side. Small enough that 256 pages stay ~100 KB, large enough
/// that a page is a real page.
pub const STREAMED_TERRAIN_RESOLUTION: u32 = 9;
/// Level-0 metres per sample => a 16 m tile span.
pub const STREAMED_TERRAIN_MPS: f64 = 2.0;
/// Level-0 tiles per side: a 16 x 16 grid => **256 m of world**, far wider than
/// any single render-wants radius (`RENDER_LOD0_RADIUS_TILES` x 16 m = 40 m), so
/// the camera genuinely pages tiles in and out as it moves. 16 is a power of two,
/// so the pyramid closes cleanly: 256 -> 64 -> 16 -> 4 pages, i.e. **three coarse
/// levels** (the gate needs at least two).
pub const STREAMED_TERRAIN_TILES: i32 = 16;

/// World edge length of the generated terrain (metres).
pub fn streamed_terrain_world_size() -> f64 {
    (STREAMED_TERRAIN_RESOLUTION as f64 - 1.0)
        * STREAMED_TERRAIN_MPS
        * STREAMED_TERRAIN_TILES as f64
}

/// The generated terrain's analytic height at world `(x, z)`.
///
/// Built from [`inf_math::psin64`] / [`inf_math::pcos64`], never `std` trig: the
/// P14 law -- `std` transcendentals are not bit-portable, and this function's
/// output ends up in bytes a cook on one OS and a run on another must agree about.
pub fn streamed_terrain_height(x: f64, z: f64) -> f64 {
    8.0 * inf_math::psin64(x * 0.04) * inf_math::pcos64(z * 0.035)
        + 2.0 * inf_math::psin64((x + z) * 0.11)
}

/// The authored level-0 heightfield the `.inf_terrain` is built from.
pub fn streamed_terrain_data() -> inf_terrain::TerrainData {
    let mut t = inf_terrain::TerrainData::new(STREAMED_TERRAIN_RESOLUTION, STREAMED_TERRAIN_MPS);
    for tz in 0..STREAMED_TERRAIN_TILES {
        for tx in 0..STREAMED_TERRAIN_TILES {
            t.author_tile((tx, tz), streamed_terrain_height);
        }
    }
    t
}

/// The `.inf_terrain` payload: the level-0 grid plus its full LOD pyramid.
///
/// A pure function of the generators, so two builds are byte-identical (the
/// `.inf_terrain` layout is deterministic by construction -- see
/// `inf_terrain::asset`).
pub fn streamed_terrain_asset() -> inf_terrain::TerrainAsset {
    let data = streamed_terrain_data();
    let pyramid = inf_terrain::build_pyramid(&data, inf_terrain::PyramidOptions::default());
    inf_terrain::build_terrain_asset(&data, &pyramid, inf_terrain::PyramidOptions::default())
        .expect("streamed-terrain asset builds")
}

/// Write `World.inf_terrain` (+ its inf_asset sidecar with the stable asset GUID)
/// into `dir` -- the gate's **fixture setup**, and the P16.4 import wizard's model.
///
/// Goes through [`inf_terrain::write_terrain_asset`], the one sanctioned writer:
/// the bytes on disk are the raw payload image, never a framed `inf_asset::encode`
/// (which would knock every tile off its 16-byte boundary). The sidecar hashes
/// exactly the bytes written, so the cook packs them verbatim.
pub fn write_streamed_terrain_asset(dir: &std::path::Path) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("mkdir: {e}"))?;
    let asset = streamed_terrain_asset();
    let path = dir.join("World.inf_terrain");
    let bytes = inf_terrain::write_terrain_asset(&path, &asset)
        .map_err(|e| format!("write terrain asset: {e}"))?;
    inf_asset::AssetSidecar::new(
        inf_asset::AssetId(STREAMED_TERRAIN_ASSET_GUID),
        inf_asset::AssetKind::Terrain,
        inf_asset::ContentHash::of(bytes),
    )
    .save(&path)
    .map_err(|e| format!("write terrain sidecar: {e}"))
}

/// Build the streamed-terrain [`SceneDoc`].
///
/// The `Terrain` component carries **no tiles**: its `data` is an empty working
/// set configured on the asset's grid, and `asset` points at the `.inf_terrain`.
/// That is the whole point -- the level stays kilobytes while the world is
/// 256 m x 256 m of paged heightfield.
pub fn streamed_terrain_scene() -> SceneDoc {
    use inf_ecs::components::{Camera, Light, LightKind, Terrain};

    let mut doc = SceneDoc::new();
    doc.set_title("Streamed Terrain");
    let world_size = streamed_terrain_world_size();

    // -- The streamed terrain, anchored at the world origin (so world XZ ==
    //    terrain-local XZ and the height probe is the bare generator). --
    doc.create_with_guid(
        STREAMED_TERRAIN_TERRAIN_GUID,
        SpawnKind::Empty,
        "Terrain",
        None,
    );
    insert!(
        doc,
        STREAMED_TERRAIN_TERRAIN_GUID,
        Transform::from_translation(DVec3::ZERO)
    );
    {
        let mut terrain = Terrain::configured(STREAMED_TERRAIN_RESOLUTION, STREAMED_TERRAIN_MPS);
        terrain.asset = Some(STREAMED_TERRAIN_ASSET_GUID);
        terrain.macro_variation = 0.2;
        debug_assert!(terrain.data.is_empty(), "a streamed terrain ships no tiles");
        insert!(doc, STREAMED_TERRAIN_TERRAIN_GUID, terrain);
    }

    // -- The walker: the entity the SIM scripts across the terrain. Its position
    //    is what `sim_wants` derives level-0 residency from, and the gate probes
    //    `terrain.height_at` under it. --
    doc.create_with_guid(
        STREAMED_TERRAIN_WALKER_GUID,
        SpawnKind::Empty,
        "Walker",
        None,
    );
    insert!(
        doc,
        STREAMED_TERRAIN_WALKER_GUID,
        Transform::from_translation(streamed_terrain_walk_point(0))
    );
    // A character controller is what makes the Walker a **terrain observer**
    // (`inf_player::terrain_stream::observes_terrain`): sim residency follows the
    // things that walk on the ground, not everything with a transform. With no
    // `RigidBody3D`/`Collider3D` beside it the physics bridge skips the entity
    // entirely, so this marks intent without simulating anything.
    insert!(
        doc,
        STREAMED_TERRAIN_WALKER_GUID,
        inf_ecs::components::CharacterController3D::default()
    );

    // -- A directional sun. --
    doc.create_with_guid(STREAMED_TERRAIN_SUN_GUID, SpawnKind::Empty, "Sun", None);
    insert!(
        doc,
        STREAMED_TERRAIN_SUN_GUID,
        Transform {
            translation: inf_ecs::math::Vec3d::new(0.0, 80.0, 0.0),
            rotation: inf_ecs::math::Vec3d::new(-50.0, -30.0, 0.0),
            scale: inf_ecs::math::Vec3d::new(1.0, 1.0, 1.0),
        }
    );
    insert!(
        doc,
        STREAMED_TERRAIN_SUN_GUID,
        Light {
            kind: LightKind::Directional,
            color: Color::new(1.0, 0.97, 0.9, 1.0),
            intensity: 2.5,
            ..Default::default()
        }
    );

    // -- A camera over the middle of the world. --
    doc.create_with_guid(
        STREAMED_TERRAIN_CAMERA_GUID,
        SpawnKind::Empty,
        "Camera",
        None,
    );
    insert!(
        doc,
        STREAMED_TERRAIN_CAMERA_GUID,
        Transform::from_translation(DVec3::new(world_size * 0.5, 60.0, world_size * 0.5))
    );
    insert!(doc, STREAMED_TERRAIN_CAMERA_GUID, Camera::default());

    doc.world_mut().propagate();
    doc.mark_saved();
    doc
}

/// The scripted **sim** walk: step `i`'s world position for the Walker entity.
///
/// A diagonal crossing of the whole 256 m world, so the sim's level-0
/// neighbourhood really does page in and out. Deterministic and camera-free --
/// this is the trace the "sim determinism vs camera" gate compares.
pub fn streamed_terrain_walk_point(step: usize) -> DVec3 {
    let world_size = streamed_terrain_world_size();
    let t = step as f64 * 1.5;
    let x = (t % world_size).clamp(0.0, world_size);
    let z = ((t * 0.7) % world_size).clamp(0.0, world_size);
    DVec3::new(x, streamed_terrain_height(x, z), z)
}

/// Scripted **camera** path A: a straight sweep along +X at mid-Z.
pub fn streamed_terrain_camera_a(step: usize) -> DVec3 {
    let world_size = streamed_terrain_world_size();
    let x = (step as f64 * 3.0) % world_size;
    DVec3::new(x, 40.0, world_size * 0.5)
}

/// Scripted **camera** path B: an orbit around the world centre -- deliberately
/// nothing like [`streamed_terrain_camera_a`], so "the sim ignores the camera" is
/// tested against a genuinely different residency history, not a variation of the
/// same one.
pub fn streamed_terrain_camera_b(step: usize) -> DVec3 {
    let world_size = streamed_terrain_world_size();
    let a = step as f64 * 0.21;
    let r = world_size * 0.35;
    DVec3::new(
        world_size * 0.5 + r * inf_math::pcos64(a),
        40.0,
        world_size * 0.5 + r * inf_math::psin64(a),
    )
}

/// The repo-root `samples/streamed-terrain/` directory.
pub fn streamed_terrain_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../samples/streamed-terrain")
}

/// Write the committed streamed-terrain files (regeneration path): the `.inf_lvl`
/// (+ sidecar) and the README. The `.inf_terrain` itself is **not** committed --
/// see [`write_streamed_terrain_asset`].
pub fn write_streamed_terrain() -> Result<(), String> {
    let dir = streamed_terrain_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("mkdir: {e}"))?;
    crate::scene::serialize::save(
        &streamed_terrain_scene(),
        &dir.join("StreamedTerrain.inf_lvl"),
        Some(STREAMED_TERRAIN_LEVEL_GUID),
    )?;
    std::fs::write(dir.join("README.md"), STREAMED_TERRAIN_README)
        .map_err(|e| format!("write readme: {e}"))?;
    Ok(())
}

const STREAMED_TERRAIN_README: &str = "# Streamed Terrain (P16.3 gate scene)\n\n\
Generated by `inf_editor_core::samples::streamed_terrain_scene` -- the P16.3b2 gate\n\
scene for **camera-driven terrain streaming**.\n\n\
- `StreamedTerrain.inf_lvl` -- the scene (schema v9). Its `Terrain` carries an EMPTY\n\
  working set plus `asset = World.inf_terrain`, so the level stays kilobytes while\n\
  the world is 256 m x 256 m of paged heightfield. A `Walker` entity is what the\n\
  gate scripts across the terrain.\n\
- `World.inf_terrain` -- NOT committed. It is ~100 KB of derived bytes that\n\
  `samples::write_streamed_terrain_asset` reproduces exactly, so the gate generates\n\
  it into the fixture's Content directory (16x16 level-0 pages of 9^2 samples at\n\
  2 m, plus three coarse pyramid levels: 256 -> 64 -> 16 -> 4).\n\n\
## The doctrine this scene exists to pin\n\n\
The fixed-step sim's results must never depend on camera-driven residency. Sim\n\
wants (level-0 pages around the sim's own entities) load synchronously at the\n\
fixed-step boundary into the ECS `Terrain`'s data; render wants (the camera's\n\
quadtree cut) load into a separate working set inside the streamer that no entity\n\
references.\n\n\
The gate (`runtime/inf-player/tests/streamed_terrain.rs`):\n\n\
1. the cook ships the `.inf_terrain` through the level->terrain edge, UNCOMPRESSED\n\
   (streaming-class), so tiles page zero-copy out of the mapping;\n\
2. a headless run over a scripted camera path reproduces a byte-identical\n\
   resident-set trace AND rendered-frame (projected terrain) trace across two runs;\n\
3. the SAME scripted sim under two COMPLETELY different camera paths produces a\n\
   byte-identical sim trace -- the doctrine, as an executable assertion;\n\
4. PIE == shipping: the cooked-pack path and the editor-doc path stream the same\n\
   terrain to the same sim trace and the same resident set.\n\n\
Regenerate with `INF_BLESS_SAMPLES=1 cargo test -p inf-editor-core samples`.\n";

// -- Partitioned-world gate scene (P16.5) ------------------------------------
//
// The world-partition gate scene: a 4 x 4 grid of 128 m cells, one authored prop
// per cell, a persistent manager + sun, and a "Walker" carrying a
// `StreamingSource` that the gate scripts along the +X row. Everything is
// committed as ONE `.inf_lvl` (the editor stays single-document); the `.inf_part`
// is DERIVED by the cook, exactly as `.inf_vmesh` is.

pub const PARTITIONED_LEVEL_GUID: Uuid = Uuid::from_u128(0x8416_0500);
pub const PARTITIONED_WALKER_GUID: Uuid = Uuid::from_u128(0x8416_0501);
pub const PARTITIONED_MANAGER_GUID: Uuid = Uuid::from_u128(0x8416_0502);
pub const PARTITIONED_SUN_GUID: Uuid = Uuid::from_u128(0x8416_0503);
/// A prop parented under the cell-(3,3) prop — proves a hierarchy streams as one
/// unit even though its own transform sits nowhere near its parent's cell.
pub const PARTITIONED_CHILD_GUID: Uuid = Uuid::from_u128(0x8416_0504);

/// Cell edge length (metres). Small enough that a scripted walk crosses several
/// cells in a short run; large enough to be a plausible authoring unit.
pub const PARTITIONED_CELL_SIZE_M: f64 = 128.0;
/// Cells per side: 4 x 4 = 16, comfortably over the gate's "at least 3 x 3".
pub const PARTITIONED_GRID: i32 = 4;
/// Level activation radius (metres) — under one cell, so exactly the cells the
/// walker is standing on/next to activate and the trace has real churn.
pub const PARTITIONED_ACTIVATION_RADIUS_M: f64 = 32.0;
/// Level prefetch margin (metres). Cells within `activation + margin` may be
/// decoded ahead of need; it can never change WHICH cells activate.
pub const PARTITIONED_PREFETCH_MARGIN_M: f64 = 192.0;

/// World edge length of the partitioned grid (metres).
pub fn partitioned_world_size() -> f64 {
    PARTITIONED_CELL_SIZE_M * PARTITIONED_GRID as f64
}

/// The stable GUID of the prop authored in cell `(cx, cz)`.
pub fn partitioned_prop_guid(cx: i32, cz: i32) -> Uuid {
    Uuid::from_u128(0x8416_0600 + (cz * PARTITIONED_GRID + cx) as u128)
}

/// The world position of the prop authored in cell `(cx, cz)` — the cell's
/// centre, so it is unambiguously inside exactly one cell.
pub fn partitioned_prop_position(cx: i32, cz: i32) -> DVec3 {
    DVec3::new(
        (cx as f64 + 0.5) * PARTITIONED_CELL_SIZE_M,
        0.0,
        (cz as f64 + 0.5) * PARTITIONED_CELL_SIZE_M,
    )
}

/// The scripted **sim** walk: step `i`'s world position for the Walker.
///
/// Straight along +X through the centres of row `z = 0`, one third of a cell per
/// step, so the walk crosses every cell in the row and the activation trace has
/// real transitions. Deterministic and camera-free — this is the trace the gates
/// compare, and (by the doctrine) the only thing residency may depend on.
pub fn partitioned_walk_point(step: usize) -> DVec3 {
    let span = partitioned_world_size();
    let x = (step as f64 * (PARTITIONED_CELL_SIZE_M / 3.0)) % span;
    DVec3::new(x, 0.0, PARTITIONED_CELL_SIZE_M * 0.5)
}

/// Build the partitioned-world [`SceneDoc`].
///
/// The document is a plain, single, unpartitioned-looking level — because that is
/// what the editor authors. What makes it partitioned is one settings block; the
/// cook is what splits it, and the player is what streams it.
pub fn partitioned_world_scene() -> SceneDoc {
    use crate::scene::serialize::{LevelSettings, PartitionSettings};
    use inf_ecs::components::{AlwaysLoaded, Light, LightKind, StreamingSource};

    let mut doc = SceneDoc::new();
    doc.set_title("Partitioned World");
    doc.set_settings(LevelSettings {
        partition: PartitionSettings {
            enabled: true,
            cell_size_m: PARTITIONED_CELL_SIZE_M,
            activation_radius_m: PARTITIONED_ACTIVATION_RADIUS_M,
            prefetch_margin_m: PARTITIONED_PREFETCH_MARGIN_M,
        },
        ..LevelSettings::default()
    });

    // -- The persistent cell --
    //
    // A manager with no spatial component at all (the `Unplaced` rule), and a sun
    // explicitly marked `AlwaysLoaded` (a Light DOES occupy space, so without the
    // marker it would stream out and the world would go dark). Those two entities
    // are the whole "what is a persistent cell for" story, in the scene.
    doc.create_with_guid(PARTITIONED_MANAGER_GUID, SpawnKind::Empty, "GameMode", None);
    insert!(
        doc,
        PARTITIONED_MANAGER_GUID,
        Transform::from_translation(DVec3::ZERO)
    );

    doc.create_with_guid(PARTITIONED_SUN_GUID, SpawnKind::Empty, "Sun", None);
    insert!(
        doc,
        PARTITIONED_SUN_GUID,
        Transform {
            translation: inf_ecs::math::Vec3d::new(0.0, 80.0, 0.0),
            rotation: inf_ecs::math::Vec3d::new(-50.0, -30.0, 0.0),
            scale: inf_ecs::math::Vec3d::new(1.0, 1.0, 1.0),
        }
    );
    insert!(
        doc,
        PARTITIONED_SUN_GUID,
        Light {
            kind: LightKind::Directional,
            color: Color::new(1.0, 0.97, 0.9, 1.0),
            intensity: 2.5,
            ..Default::default()
        }
    );
    insert!(doc, PARTITIONED_SUN_GUID, AlwaysLoaded);

    // -- The streaming source: the entity residency is derived FROM. --
    //
    // It carries a `StreamingSource` (which is also what makes it persistent —
    // a source that could stream itself out is a bootstrap paradox), and
    // `radius_m: 0` so the LEVEL's activation radius is what governs; the gate
    // then has one knob to reason about.
    doc.create_with_guid(PARTITIONED_WALKER_GUID, SpawnKind::Empty, "Walker", None);
    insert!(
        doc,
        PARTITIONED_WALKER_GUID,
        Transform::from_translation(partitioned_walk_point(0))
    );
    insert!(
        doc,
        PARTITIONED_WALKER_GUID,
        StreamingSource { radius_m: 0.0 }
    );

    // -- One authored prop per cell (a cube at the cell centre). --
    for cz in 0..PARTITIONED_GRID {
        for cx in 0..PARTITIONED_GRID {
            let guid = partitioned_prop_guid(cx, cz);
            doc.create_with_guid(guid, SpawnKind::Cube, &format!("Prop {cx},{cz}"), None);
            insert!(
                doc,
                guid,
                Transform::from_translation(partitioned_prop_position(cx, cz))
            );
        }
    }

    // -- A child of the far-corner prop, authored a whole world away. --
    //
    // Its own transform would bin it into cell (0,0); the partitioner assigns it
    // its ROOT's cell instead, so the pair never splits. The gate asserts it
    // appears and disappears together with its parent.
    doc.create_with_guid(
        PARTITIONED_CHILD_GUID,
        SpawnKind::Cube,
        "Far Child",
        Some(partitioned_prop_guid(
            PARTITIONED_GRID - 1,
            PARTITIONED_GRID - 1,
        )),
    );
    insert!(
        doc,
        PARTITIONED_CHILD_GUID,
        Transform::from_translation(DVec3::new(-2000.0, 0.0, -2000.0))
    );

    doc.world_mut().propagate();
    doc.mark_saved();
    doc
}

/// The repo-root `samples/partitioned-world/` directory.
pub fn partitioned_world_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../samples/partitioned-world")
}

/// Write the committed partitioned-world files (regeneration path).
pub fn write_partitioned_world() -> Result<(), String> {
    let dir = partitioned_world_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("mkdir: {e}"))?;
    crate::scene::serialize::save(
        &partitioned_world_scene(),
        &dir.join("PartitionedWorld.inf_lvl"),
        Some(PARTITIONED_LEVEL_GUID),
    )?;
    std::fs::write(dir.join("README.md"), PARTITIONED_WORLD_README)
        .map_err(|e| format!("write readme: {e}"))?;
    Ok(())
}

const PARTITIONED_WORLD_README: &str = "# Partitioned World (P16.5 gate scene)\n\n\
Generated by `inf_editor_core::samples::partitioned_world_scene` -- the P16.5 gate\n\
scene for **world partition / level streaming**.\n\n\
- `PartitionedWorld.inf_lvl` -- the scene (schema v10). ONE document, as the editor\n\
  always authors: a 4x4 grid of cubes at the centres of 128 m cells, a `GameMode`\n\
  with no spatial component, a sun marked `AlwaysLoaded`, a `Walker` carrying a\n\
  `StreamingSource`, and a child of the far-corner prop authored a whole world\n\
  away from it. What makes it partitioned is one settings block\n\
  (`partition.enabled`), nothing else.\n\
- `PartitionedWorld.inf_part` -- NOT committed. It is DERIVED by the cook (like\n\
  `.inf_vmesh`): the cook bins the entities into a persistent cell + 16 grid cells\n\
  and writes them to a pack entry whose GUID is a pure function of the level's.\n\n\
## The doctrine this scene exists to pin\n\n\
Cell streaming decides which entities EXIST, so residency must be a function of sim\n\
state alone. Wants come from `StreamingSource` entities read at the fixed-step\n\
boundary -- never a camera -- and activation/deactivation happen only at that sync\n\
point, in ascending cell order. Loading may run ahead (the prefetch margin); a cell\n\
that reaches its activation step unloaded blocks the step. So the margin buys\n\
latency and can never move a result.\n\n\
## The v1 boundaries, stated rather than discovered\n\n\
- The PERSISTENT cell is the world at step 0: the level builder spawns it BEFORE\n\
  blueprint actors bind, so a persistent entity's `ActorClass` ticks normally. An\n\
  entity that streams IN does NOT gain a ticking blueprint in v1 (the actor map is\n\
  fixed at `RuntimeSim` construction). Mark such an entity `AlwaysLoaded`.\n\
- Runtime-spawned entities are never despawned by streaming; a statically-placed\n\
  one is evicted with its BIRTH cell, wherever a script has since moved it.\n\
- A cook-time reference from one cell to another is a cook WARNING, never a\n\
  promotion: residency must not depend on the reference graph.\n\
- The editor's in-process Simulate runs the whole document unpartitioned (single\n\
  document in v1); PIE and a shipped build both stream, and those two are what the\n\
  parity gate compares.\n\n\
The gate (`runtime/inf-player/tests/partitioned_world.rs`):\n\n\
1. the cook emits ONE `.inf_part` entry, UNCOMPRESSED, deterministic across\n\
   rebuilds, and the cooked `.inf_lvl` carries no entities;\n\
2. two headless runs of the scripted walk produce an identical activation trace;\n\
3. the SAME walk under two DIFFERENT prefetch margins produces a byte-identical\n\
   sim trace -- the doctrine, as an executable assertion;\n\
4. PIE == shipping: the cooked-pack path and the editor-document path stream the\n\
   same cells to the same sim trace;\n\
5. non-partitioned regression: the platformer sample's pack is byte-identical\n\
   whether or not partitioning exists;\n\
6. an entity in a far cell does not exist until the walker approaches -- then it\n\
   exists with its authored transform.\n\n\
Regenerate with `INF_BLESS_SAMPLES=1 cargo test -p inf-editor-core samples`.\n";

// -- Phase 16 gate scene: the composed world (P16.6) --------------------------
//
// The closing gate of Phase 16 puts every piece of the phase in ONE level:
//
//   * a **wizard-imported streamed terrain** — built by the P16.4a chunked
//     importer from a synthetic 16-bit heightmap at `meters_per_sample = 8`, so
//     the world is kilometres wide and the `.inf_terrain` carries a real LOD
//     pyramid (three levels);
//   * a **partitioned world on top of it** — a 4 x 4 grid of 2 km cells with an
//     authored prop in each, a persistent manager, and an `AlwaysLoaded` sun;
//   * a **second, inline terrain** off to one side — the multi-terrain half of
//     P16.6, proving two terrains render (and stream, and pick) side by side in
//     PIE == shipping;
//   * a scripted `StreamingSource` walk, and a camera path that deliberately goes
//     somewhere else.
//
// Both terrain entities are marked `AlwaysLoaded`. A `Terrain` component
// "occupies space" (`inf_scene::partition::occupies_space`), so without the
// marker the partitioner would bin the terrain entity into one cell and the
// ground would blink out of existence as the player walked away. That is a v1
// boundary of world partition, not a bug — stated here and in the sample's README
// rather than discovered.
//
// The `.inf_lvl` is committed; the `.inf_terrain` is **generated** into the
// fixture's content directory by [`write_phase16_terrain_asset`], because it is
// ~5 MB of derived bytes a pure generator reproduces exactly (the same reasoning
// as the streamed-terrain sample, one order of magnitude up).

pub const PHASE16_LEVEL_GUID: Uuid = Uuid::from_u128(0x8416_0900);
/// The streamed terrain (asset-backed, the wizard-imported one).
pub const PHASE16_TERRAIN_GUID: Uuid = Uuid::from_u128(0x8416_0901);
/// The second terrain: inline tiles in the level, no asset.
pub const PHASE16_INLINE_TERRAIN_GUID: Uuid = Uuid::from_u128(0x8416_0902);
/// The scripted walker: a `StreamingSource` **and** a terrain observer.
pub const PHASE16_WALKER_GUID: Uuid = Uuid::from_u128(0x8416_0903);
pub const PHASE16_SUN_GUID: Uuid = Uuid::from_u128(0x8416_0904);
pub const PHASE16_MANAGER_GUID: Uuid = Uuid::from_u128(0x8416_0905);
pub const PHASE16_CAMERA_GUID: Uuid = Uuid::from_u128(0x8416_0906);
/// The asset GUID stamped into the generated `Phase16World.inf_terrain` sidecar.
pub const PHASE16_TERRAIN_ASSET_GUID: Uuid = Uuid::from_u128(0x8416_09AA);

/// Source heightmap edge, in samples. `1025 = 8 x 128 + 1`, so the lattice closes
/// exactly on 8 x 8 shared-edge tiles with no clamped final row.
///
/// This is the CI-affordable stand-in for the wizard's headline case. A literal
/// 16 k x 16 k source is ~268 M samples and produces a ~1 GB payload however it is
/// tiled — the sample count, not the tiling, is what costs — so it cannot run in a
/// CI job that also cooks it into a pack twice. The **shape** of the import is the
/// real one (chunked row-band decode, banded pyramid, `meters_per_sample` >> 1,
/// kilometre-scale extent); the 16 k pass itself lives in
/// `terrain_import::huge_heightmap_16k_imports`, which is `#[ignore]`d for exactly
/// the same reason and run by hand when the importer is profiled.
pub const PHASE16_SOURCE_SAMPLES: u32 = 1025;
/// Samples per tile side of the imported terrain.
pub const PHASE16_TILE_RESOLUTION: u32 = 129;
/// World metres per sample: the "8 m spacing turns a big heightmap into a
/// continent" case the wizard exists for (>= 8, per the Phase 16 goal).
pub const PHASE16_MPS: f64 = 8.0;
/// Elevation the source's full scale maps to.
pub const PHASE16_MAX_HEIGHT: f64 = 400.0;

/// Partition cell edge (metres): 2 km, so the 8.2 km world is a 4 x 4 grid.
pub const PHASE16_CELL_SIZE_M: f64 = 2048.0;
/// Cells per side (4 x 4 = 16, comfortably over the gate's "at least 3 x 3").
pub const PHASE16_GRID: i32 = 4;
/// Level activation radius (metres) — under one cell, so the walk really churns.
pub const PHASE16_ACTIVATION_RADIUS_M: f64 = 512.0;
/// Level prefetch margin (metres). Look-ahead only: it can never change WHICH
/// cells activate, which is one of the properties the gate asserts.
pub const PHASE16_PREFETCH_MARGIN_M: f64 = 2048.0;

/// The inline terrain's grid: 4 x 4 pages of 9^2 samples at 4 m — a few kilobytes
/// in the `.inf_lvl`, and a completely different grid from the streamed one, which
/// is the point (a shared per-tile GPU cache key would collide immediately).
pub const PHASE16_INLINE_RESOLUTION: u32 = 9;
pub const PHASE16_INLINE_MPS: f64 = 4.0;
pub const PHASE16_INLINE_TILES: i32 = 4;

/// World edge length of the streamed terrain (metres).
pub fn phase16_world_size() -> f64 {
    (PHASE16_SOURCE_SAMPLES as f64 - 1.0) * PHASE16_MPS
}

/// World origin of the **inline** terrain: north of the streamed one, clear of it
/// in XZ so the two never overlap and the gate can tell their pixels apart.
pub fn phase16_inline_origin() -> DVec3 {
    DVec3::new(0.0, 0.0, -1024.0)
}

/// The normalized `[0, 1]` source value at source sample `(i, j)`.
///
/// Built from [`inf_math::psin64`] / [`inf_math::pcos64`], never `std` trig: the
/// P14 law — `std` transcendentals are not bit-portable, and this function's
/// output becomes bytes that a cook on one OS and a run on another must agree
/// about, through a `.inf_terrain` the gate hashes.
pub fn phase16_source_sample(i: u32, j: u32) -> f64 {
    let x = i as f64;
    let z = j as f64;
    let v = 0.5
        + 0.25 * inf_math::psin64(x * 0.011) * inf_math::pcos64(z * 0.009)
        + 0.15 * inf_math::psin64((x + z) * 0.021)
        + 0.05 * inf_math::pcos64((x - 2.0 * z) * 0.047);
    v.clamp(0.0, 1.0)
}

/// The synthetic 16-bit source heightmap the import consumes, PNG-encoded — the
/// file a user would drop on the Terrain Import wizard.
pub fn phase16_source_png() -> Result<Vec<u8>, String> {
    let n = PHASE16_SOURCE_SAMPLES;
    let mut samples = Vec::with_capacity((n * n) as usize);
    for j in 0..n {
        for i in 0..n {
            samples.push((phase16_source_sample(i, j) * u16::MAX as f64).round() as u16);
        }
    }
    inf_terrain::encode_png16(&inf_terrain::HeightImage {
        width: n,
        height: n,
        samples,
    })
    .map_err(|e| format!("encode source heightmap: {e}"))
}

/// The wizard settings the gate's terrain is imported with — the same
/// [`TerrainImportSettings`](crate::assets::terrain_import::TerrainImportSettings)
/// block the Terrain Import wizard fills in and writes into the asset's sidecar.
pub fn phase16_import_settings() -> crate::assets::terrain_import::TerrainImportSettings {
    crate::assets::terrain_import::TerrainImportSettings {
        tile_resolution: PHASE16_TILE_RESOLUTION,
        meters_per_sample: PHASE16_MPS,
        min_height: 0.0,
        max_height: PHASE16_MAX_HEIGHT,
        float_meters: false,
        // Grow into +X/+Z from the world origin, so world XZ == terrain-local XZ
        // and every probe in the gate is the bare generator.
        center: false,
        ..Default::default()
    }
}

/// Import the gate's `.inf_terrain` **through the wizard's own path**: the
/// wizard's settings block mapped onto Ring-0 `HeightmapImport`, then
/// [`inf_terrain::import_heightmap_reader`] — the chunked, row-band importer
/// P16.4a built, which never materializes the sample grid.
///
/// The only thing the wizard adds on top is a project to commit into and a
/// generated GUID; the fixture supplies a stable GUID instead (so the level's
/// `Terrain.asset` ref resolves), exactly as `write_streamed_terrain_asset` does.
pub fn phase16_terrain_asset() -> Result<inf_terrain::TerrainAsset, String> {
    let png = phase16_source_png()?;
    let probe = inf_terrain::probe_heightmap_bytes(&png).map_err(|e| format!("probe: {e}"))?;
    let settings = phase16_import_settings();
    let import = settings.to_import(probe.width, probe.height);
    let opts = inf_terrain::ChunkedImportOptions {
        pyramid: settings.pyramid(),
        // The committed sample is byte-pinned, so it takes the pre-Wave-G
        // defaults explicitly: no georeferenced placement, no no-data policy.
        // Spelling them out rather than `..Default::default()` is what makes a
        // future default change fail here instead of silently re-blessing the
        // committed bytes.
        world_origin: glam::DVec3::ZERO,
        nodata: inf_terrain::NodataHandling::NONE,
    };
    let (asset, _report) = inf_terrain::import_heightmap_reader(
        std::io::Cursor::new(png),
        import,
        opts,
        &mut |_| {},
        &|| false,
    )
    .map_err(|e| format!("chunked import: {e}"))?;
    Ok(asset)
}

/// Write `Phase16World.inf_terrain` (+ its `inf_asset` sidecar with the stable
/// asset GUID) into `dir` — the gate's **fixture setup**.
pub fn write_phase16_terrain_asset(dir: &std::path::Path) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("mkdir: {e}"))?;
    let asset = phase16_terrain_asset()?;
    let path = dir.join("Phase16World.inf_terrain");
    let bytes = inf_terrain::write_terrain_asset(&path, &asset)
        .map_err(|e| format!("write terrain asset: {e}"))?;
    inf_asset::AssetSidecar::new(
        inf_asset::AssetId(PHASE16_TERRAIN_ASSET_GUID),
        inf_asset::AssetKind::Terrain,
        inf_asset::ContentHash::of(bytes),
    )
    .save(&path)
    .map_err(|e| format!("write terrain sidecar: {e}"))
}

/// The **inline** second terrain's authored heightfield — small, entity-local,
/// and carried in the `.inf_lvl` itself.
pub fn phase16_inline_terrain_data() -> inf_terrain::TerrainData {
    let mut t = inf_terrain::TerrainData::new(PHASE16_INLINE_RESOLUTION, PHASE16_INLINE_MPS);
    for tz in 0..PHASE16_INLINE_TILES {
        for tx in 0..PHASE16_INLINE_TILES {
            t.author_tile((tx, tz), |x, z| {
                12.0 + 3.0 * inf_math::psin64(x * 0.05) * inf_math::pcos64(z * 0.05)
            });
        }
    }
    t.clear_dirty();
    t
}

/// The stable GUID of the prop authored in cell `(cx, cz)`.
pub fn phase16_prop_guid(cx: i32, cz: i32) -> Uuid {
    Uuid::from_u128(0x8416_0A00 + (cz * PHASE16_GRID + cx) as u128)
}

/// The world position of the prop authored in cell `(cx, cz)` — the cell centre,
/// so it is unambiguously inside exactly one cell.
pub fn phase16_prop_position(cx: i32, cz: i32) -> DVec3 {
    DVec3::new(
        (cx as f64 + 0.5) * PHASE16_CELL_SIZE_M,
        0.0,
        (cz as f64 + 0.5) * PHASE16_CELL_SIZE_M,
    )
}

/// The scripted **sim** walk: step `i`'s world position for the walker.
///
/// A diagonal crossing of the whole 8.2 km world at a third of a cell per step, so
/// cells activate/deactivate repeatedly AND the terrain's level-0 neighbourhood
/// slides. Deterministic and camera-free — this is the trace the gates compare,
/// and (by the doctrine) the only thing residency may depend on.
pub fn phase16_walk_point(step: usize) -> DVec3 {
    let span = phase16_world_size();
    let t = step as f64 * (PHASE16_CELL_SIZE_M / 3.0);
    let x = (t % span).clamp(0.0, span);
    let z = ((t * 0.5) % span).clamp(0.0, span);
    DVec3::new(x, 0.0, z)
}

/// Scripted **camera** path A: a straight sweep along +X at mid-Z, deliberately
/// unrelated to the walk.
pub fn phase16_camera_a(step: usize) -> DVec3 {
    let span = phase16_world_size();
    let x = (step as f64 * 512.0) % span;
    DVec3::new(x, 600.0, span * 0.5)
}

/// Scripted **camera** path B: an orbit of the world centre — nothing like
/// [`phase16_camera_a`], so "the sim ignores the camera" is tested against a
/// genuinely different residency history.
pub fn phase16_camera_b(step: usize) -> DVec3 {
    let span = phase16_world_size();
    let a = step as f64 * 0.23;
    let r = span * 0.35;
    DVec3::new(
        span * 0.5 + r * inf_math::pcos64(a),
        600.0,
        span * 0.5 + r * inf_math::psin64(a),
    )
}

/// Build the Phase 16 gate [`SceneDoc`] — the composed world.
pub fn phase16_world_scene() -> SceneDoc {
    use crate::scene::serialize::{LevelSettings, PartitionSettings};
    use inf_ecs::components::{
        AlwaysLoaded, Camera, CharacterController3D, Light, LightKind, StreamingSource, Terrain,
    };

    let mut doc = SceneDoc::new();
    doc.set_title("Phase 16 World");
    doc.set_settings(LevelSettings {
        partition: PartitionSettings {
            enabled: true,
            cell_size_m: PHASE16_CELL_SIZE_M,
            activation_radius_m: PHASE16_ACTIVATION_RADIUS_M,
            prefetch_margin_m: PHASE16_PREFETCH_MARGIN_M,
        },
        ..LevelSettings::default()
    });

    // -- Terrain 1: the wizard-imported, streamed one. --
    //
    // AlwaysLoaded: a Terrain occupies space, so the partitioner would otherwise
    // bin this entity into cell (0,0) and the ground would vanish 2 km in.
    doc.create_with_guid(PHASE16_TERRAIN_GUID, SpawnKind::Empty, "Terrain", None);
    insert!(
        doc,
        PHASE16_TERRAIN_GUID,
        Transform::from_translation(DVec3::ZERO)
    );
    {
        let mut terrain = Terrain::configured(PHASE16_TILE_RESOLUTION, PHASE16_MPS);
        terrain.asset = Some(PHASE16_TERRAIN_ASSET_GUID);
        terrain.macro_variation = 0.2;
        debug_assert!(terrain.data.is_empty(), "a streamed terrain ships no tiles");
        insert!(doc, PHASE16_TERRAIN_GUID, terrain);
    }
    insert!(doc, PHASE16_TERRAIN_GUID, AlwaysLoaded);

    // -- Terrain 2: inline tiles, its own grid, its own place in the world. --
    doc.create_with_guid(
        PHASE16_INLINE_TERRAIN_GUID,
        SpawnKind::Empty,
        "Inline Terrain",
        None,
    );
    insert!(
        doc,
        PHASE16_INLINE_TERRAIN_GUID,
        Transform::from_translation(phase16_inline_origin())
    );
    {
        let mut terrain = Terrain::configured(PHASE16_INLINE_RESOLUTION, PHASE16_INLINE_MPS);
        terrain.data = phase16_inline_terrain_data();
        // A visibly different material so the two terrains are told apart on
        // screen as well as in the DTO.
        terrain.layers[0].albedo = Color::new(0.55, 0.28, 0.18, 1.0);
        terrain.macro_variation = 0.0;
        insert!(doc, PHASE16_INLINE_TERRAIN_GUID, terrain);
    }
    insert!(doc, PHASE16_INLINE_TERRAIN_GUID, AlwaysLoaded);

    // -- The persistent cell: an unplaced manager + an AlwaysLoaded sun. --
    doc.create_with_guid(PHASE16_MANAGER_GUID, SpawnKind::Empty, "GameMode", None);
    insert!(
        doc,
        PHASE16_MANAGER_GUID,
        Transform::from_translation(DVec3::ZERO)
    );

    doc.create_with_guid(PHASE16_SUN_GUID, SpawnKind::Empty, "Sun", None);
    insert!(
        doc,
        PHASE16_SUN_GUID,
        Transform {
            translation: inf_ecs::math::Vec3d::new(0.0, 800.0, 0.0),
            rotation: inf_ecs::math::Vec3d::new(-50.0, -30.0, 0.0),
            scale: inf_ecs::math::Vec3d::new(1.0, 1.0, 1.0),
        }
    );
    insert!(
        doc,
        PHASE16_SUN_GUID,
        Light {
            kind: LightKind::Directional,
            color: Color::new(1.0, 0.97, 0.9, 1.0),
            intensity: 2.5,
            ..Default::default()
        }
    );
    insert!(doc, PHASE16_SUN_GUID, AlwaysLoaded);

    // -- The walker: cell streaming's source AND the terrain's observer. --
    //
    // One entity carrying both roles is deliberate: it is what makes the two
    // streamers' want sets move together, which is the composed case this gate
    // exists to cover. `radius_m: 0` defers to the level's activation radius.
    doc.create_with_guid(PHASE16_WALKER_GUID, SpawnKind::Empty, "Walker", None);
    insert!(
        doc,
        PHASE16_WALKER_GUID,
        Transform::from_translation(phase16_walk_point(0))
    );
    insert!(doc, PHASE16_WALKER_GUID, StreamingSource { radius_m: 0.0 });
    insert!(doc, PHASE16_WALKER_GUID, CharacterController3D::default());

    // -- One authored prop per cell (a cube at the cell centre). --
    for cz in 0..PHASE16_GRID {
        for cx in 0..PHASE16_GRID {
            let guid = phase16_prop_guid(cx, cz);
            doc.create_with_guid(guid, SpawnKind::Cube, &format!("Prop {cx},{cz}"), None);
            insert!(
                doc,
                guid,
                Transform::from_translation(phase16_prop_position(cx, cz))
            );
        }
    }

    // -- A camera over the middle of the world. --
    doc.create_with_guid(PHASE16_CAMERA_GUID, SpawnKind::Empty, "Camera", None);
    insert!(
        doc,
        PHASE16_CAMERA_GUID,
        Transform::from_translation(phase16_camera_a(0))
    );
    insert!(doc, PHASE16_CAMERA_GUID, Camera::default());
    insert!(doc, PHASE16_CAMERA_GUID, AlwaysLoaded);

    doc.world_mut().propagate();
    doc.mark_saved();
    doc
}

/// The repo-root `samples/phase16-world/` directory.
pub fn phase16_world_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../samples/phase16-world")
}

/// Write the committed Phase 16 gate files (regeneration path): the `.inf_lvl`
/// (+ sidecar) and the README. The `.inf_terrain` is **not** committed — see
/// [`write_phase16_terrain_asset`].
pub fn write_phase16_world() -> Result<(), String> {
    let dir = phase16_world_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("mkdir: {e}"))?;
    crate::scene::serialize::save(
        &phase16_world_scene(),
        &dir.join("Phase16World.inf_lvl"),
        Some(PHASE16_LEVEL_GUID),
    )?;
    std::fs::write(dir.join("README.md"), PHASE16_WORLD_README)
        .map_err(|e| format!("write readme: {e}"))?;
    Ok(())
}

const PHASE16_WORLD_README: &str = "# Phase 16 World (the phase gate scene)\n\n\
Generated by `inf_editor_core::samples::phase16_world_scene` -- the **composed**\n\
gate scene for Phase 16 (world scale & streaming). Everything the phase built, in\n\
one level.\n\n\
- `Phase16World.inf_lvl` -- the scene (schema v10). A partitioned 4x4 grid of 2 km\n\
  cells with a prop in each, a `GameMode` with no spatial component, an\n\
  `AlwaysLoaded` sun and camera, a `Walker` that is BOTH the cell-streaming\n\
  `StreamingSource` and the terrain observer, and **two terrains**.\n\
- `Phase16World.inf_terrain` -- NOT committed. ~5 MB of derived bytes that\n\
  `samples::write_phase16_terrain_asset` reproduces exactly by running the P16.4a\n\
  **chunked importer** over a synthetic 1025^2 16-bit heightmap at 8 m/sample:\n\
  8x8 level-0 pages of 129^2 samples over 8.2 km of world, plus two coarse pyramid\n\
  levels (64 -> 16 -> 4).\n\n\
## The two terrains\n\n\
`Terrain` streams from the `.inf_terrain`; `Inline Terrain` carries its tiles in\n\
the level, on a completely different grid (9^2 samples at 4 m) and at a different\n\
world origin. Both render, both are picked against, and both survive the cook --\n\
which is P16.6's multi-terrain deliverable end to end.\n\n\
Both are marked `AlwaysLoaded`. A `Terrain` component *occupies space*\n\
(`inf_scene::partition::occupies_space`), so without the marker the partitioner\n\
would bin the terrain entity into the ONE cell holding its origin -- and the ground\n\
would despawn under the player as they walked out of it. That is a v1 boundary of\n\
world partition, and the cook says so: `partition::streamed_terrains` raises an\n\
advisory naming the terrain, its cell, its `.inf_terrain` and the remedy, so a user\n\
authoring their own partitioned level is told rather than left to discover it.\n\
Binning a terrain by its real footprint instead of its origin is the deferred fix.\n\n\
## Scale, honestly\n\n\
The Phase 16 goal names a 16k x 16k source. That is ~268 M samples and ~1 GB of\n\
payload however it is tiled, so it cannot run in a CI job that also cooks it into\n\
a pack -- the *shape* of the import here is the real one (chunked row-band decode,\n\
banded pyramid, 8 m spacing, kilometre-scale extent) at a size CI can afford. The\n\
literal 16k pass is `terrain_import::huge_heightmap_16k_imports`, `#[ignore]`d and\n\
run by hand when the importer is profiled.\n\n\
## The gate (`runtime/inf-player/tests/phase16_gate.rs`)\n\n\
1. full determinism across two runs -- sim trace, cell activation timeline, and\n\
   terrain render-cut trace;\n\
2. cooked == uncooked (PIE == shipping) on all three traces, with TWO terrains\n\
   projected;\n\
3. residency stays under the configured ceilings (`inf_player::budget`) for the\n\
   whole run -- terrain bytes, cell bytes, active cells;\n\
4. the fixed-step ms budget holds while both streamers are live;\n\
5. the sim trace is invariant under the camera path AND under the prefetch\n\
   margin -- the composed doctrine proof: neither the renderer's residency nor\n\
   the streamer's look-ahead can reach a fixed step.\n\n\
Regenerate with `INF_BLESS_SAMPLES=1 cargo test -p inf-editor-core samples`.\n";

// -- Phase 18 gate scene: meshlets + GI + GPU scatter, composed (P18.6) -------
//
// The Phase-18-closing gate scene. Everything the phase built, in one level:
//
//   * **P18.1/P18.2 meshlets** -- a grid of standing slabs, every one a
//     `MeshRef.asset` reference to the vgeom-demo's `Dense.inf_mesh` BY GUID
//     ([`VGEOM_DEMO_MESH_GUID`]). The mesh binary is deliberately NOT duplicated
//     here; the gate copies both sample directories into its throwaway project so
//     the cook derives one `.inf_vmesh` from the one committed mesh. Slabs stand
//     upright (a 90 deg rotation about X) precisely so they OCCLUDE -- an occluder
//     set is what makes the two-pass HZB proof and the scatter occlusion counter
//     non-vacuous, and a field of flat plates would provide neither.
//   * **P18.4 GI v2** -- a `Terrain` (four authored tiles, real relief) plus a
//     `TimeOfDay` + `SkyAtmosphere` authority running a fast clock, so the sun
//     sweeps and the amortized probe sweep is exercised against a moving key
//     light rather than a frozen one.
//   * **P18.5 GPU scatter** -- a `PcgVolume` carrying the bulk (100k+ instances,
//     see [`PHASE18_SCATTER_INSTANCES`]) and a small `Foliage` entity for the
//     painted-scatter path. The split is a size decision, stated rather than
//     discovered: a `PcgVolume`'s `evaluated` cache is a DERIVED cache that is
//     never persisted (it is recomputed on load from the committed `.inf_pcg`),
//     so 100k instances cost the `.inf_lvl` nothing, while `Foliage::instances`
//     IS persisted -- 100k of those would be a megabyte of committed level.
//
// The committed `.inf_lvl` therefore stays ~15 KB while the loaded world carries
// six figures of scatter, which is the same "small level, huge scene" property
// the vgeom demo has for triangles.

pub const PHASE18_LEVEL_GUID: Uuid = Uuid::from_u128(0x8418_0000);
pub const PHASE18_TERRAIN_GUID: Uuid = Uuid::from_u128(0x8418_0001);
pub const PHASE18_PCG_GUID: Uuid = Uuid::from_u128(0x8418_0002);
pub const PHASE18_FOLIAGE_GUID: Uuid = Uuid::from_u128(0x8418_0003);
pub const PHASE18_SKY_GUID: Uuid = Uuid::from_u128(0x8418_0004);
pub const PHASE18_SUN_GUID: Uuid = Uuid::from_u128(0x8418_0005);
pub const PHASE18_CAMERA_GUID: Uuid = Uuid::from_u128(0x8418_0006);
/// The **asset** GUID of the committed `GroundCover.inf_pcg`, so the level's
/// `PcgVolume.graph` ref resolves through the AssetDb / cooked pack and the
/// cook's level->pcg dependency edge ships the graph.
pub const PHASE18_PCG_ASSET_GUID: Uuid = Uuid::from_u128(0x8418_00AA);
/// Base GUID for the meshlet slab entities (add the flat grid index).
const PHASE18_SLAB_BASE: u128 = 0x8418_0100;

/// Terrain samples per tile side. `17` closes a tile on a 16-cell lattice.
pub const PHASE18_TERRAIN_RESOLUTION: u32 = 17;
/// World metres per terrain sample -- 16 m, so ONE tile spans 256 m and four
/// tiles cover the whole 512 m sample world in ~4.6 KB of committed heights.
pub const PHASE18_TERRAIN_MPS: f64 = 16.0;
/// Authored tiles per side (2 x 2 -- genuinely multi-tile, still tiny).
pub const PHASE18_TERRAIN_TILES: i32 = 2;

/// Meshlet slab grid side: `GRID x GRID` standing instances of the shared dense
/// mesh.
pub const PHASE18_SLAB_GRID: usize = 5;
/// Spacing between slabs, metres.
pub const PHASE18_SLAB_SPACING: f64 = 90.0;
/// Uniform scale of one slab. The dense mesh spans `[-1, 1]` in XZ, so a slab is
/// `2 * SCALE` metres wide and (standing) `2 * SCALE` metres tall.
pub const PHASE18_SLAB_SCALE: f64 = 18.0;

/// Scatter cell edge, metres (the PCG kernel's parallel granularity).
pub const PHASE18_SCATTER_CELL_SIZE: f64 = 32.0;
/// Scatter instances per m^2 at density 1.0.
pub const PHASE18_SCATTER_DENSITY: f64 = 1.6;
/// Half-extent of the scatter volume in world XZ, metres.
pub const PHASE18_SCATTER_EXTENT: f64 = 128.0;
/// Authored per-volume draw distance, metres. Deliberately **shorter** than the
/// renderer's default 400 m cull band: `ScatterSettings` clamps DOWN only, so
/// this is the one content knob that can pull the band in, and the gate's
/// `distance_culled` counter is non-vacuous because of it.
pub const PHASE18_SCATTER_DRAW_DISTANCE: f64 = 250.0;

/// Exactly how many instances [`phase18_scatter_pcg_document`] places over the
/// volume's region -- an exact, stated property of the sample rather than an
/// emergent one.
///
/// The kernel is fully specified (`scatter.rs`'s "candidate scheme"), so the
/// count is arithmetic, not luck: the region is `2 * EXTENT = 256 m` on a side and
/// cell-aligned, giving `256 / 32 = 8` cells per axis (64 cells); each cell's
/// budget is `DENSITY * CELL^2 = 1638.4`, so its jittered sub-grid is
/// `g = round(sqrt(1638.4)) = 40` a side (1600 slots); the density field is a
/// **constant 1.0**, so the hashed acceptance test `u < density` never rejects
/// (`u` is drawn from `[0, 1)`); and jitter is bounded by half a sub-cell, so no
/// candidate escapes the half-open region clip. `64 * 1600 = 102 400`.
///
/// The constant field is the deliberate part. terrain-demo already gates a
/// noise x slope sampler; what this sample needs is a *stated* instance count at
/// the gate's 100k scale, and a count that moved whenever someone retuned a noise
/// frequency would be a fixture that rots.
pub const PHASE18_SCATTER_INSTANCES: usize = 102_400;

/// Painted-foliage grid side (`GRID x GRID` persisted instances). Small on
/// purpose -- see the module note above on why the bulk is PCG.
pub const PHASE18_FOLIAGE_GRID: i32 = 4;
/// Spacing between painted foliage instances, metres (entity-local).
pub const PHASE18_FOLIAGE_SPACING: f64 = 6.0;

/// World edge length of the sample's terrain, metres (512).
pub fn phase18_world_size() -> f64 {
    PHASE18_TERRAIN_TILES as f64 * (PHASE18_TERRAIN_RESOLUTION as f64 - 1.0) * PHASE18_TERRAIN_MPS
}

/// The centre of the sample world in XZ.
pub fn phase18_world_center() -> DVec3 {
    let c = phase18_world_size() * 0.5;
    DVec3::new(c, 0.0, c)
}

/// The sample's analytic terrain height at world `(x, z)`.
///
/// Built from [`inf_math::psin64`] / [`inf_math::pcos64`], never `std` trig: the
/// P14 law -- `std` transcendentals are not bit-portable, and this function's
/// output becomes committed `.inf_lvl` bytes that a cook on one OS and a run on
/// another must agree about.
///
/// The relief is deliberately steep for the scale (~40 m peak-to-trough over a
/// 512 m world). Hills are what put scatter instances *behind* something, and a
/// gentle plain would make the HZB occlusion counter zero without anything being
/// wrong with the renderer.
pub fn phase18_height(x: f64, z: f64) -> f64 {
    14.0 * inf_math::psin64(x * 0.019) * inf_math::pcos64(z * 0.017)
        + 5.0 * inf_math::psin64((x + z) * 0.041)
}

/// The world position of the slab at grid index `(i, j)`, sitting on the ground.
pub fn phase18_slab_position(i: usize, j: usize) -> DVec3 {
    let offset = (PHASE18_SLAB_GRID as f64 - 1.0) * 0.5 * PHASE18_SLAB_SPACING;
    let c = phase18_world_center();
    let x = c.x + i as f64 * PHASE18_SLAB_SPACING - offset;
    let z = c.z + j as f64 * PHASE18_SLAB_SPACING - offset;
    // A standing slab spans +-SCALE about its centre, so lifting the centre by
    // SCALE puts its base on the ground rather than half-buried.
    DVec3::new(x, phase18_height(x, z) + PHASE18_SLAB_SCALE, z)
}

/// Build the sample's [`inf_pcg::PcgDocument`]: one layer, one rule, a **constant
/// density field** (see [`PHASE18_SCATTER_INSTANCES`] for why), two weighted kinds
/// so the projected scatter reads as varied content.
pub fn phase18_scatter_pcg_document() -> inf_pcg::PcgDocument {
    use inf_pcg::{PcgKind, PcgRule, SamplerDef};
    let rule = PcgRule {
        name: "ground-cover".into(),
        sampler: SamplerDef::Constant(1.0),
        scatter: inf_pcg::ScatterParams {
            seed: 2026_0801,
            cell_size: PHASE18_SCATTER_CELL_SIZE,
            base_density: PHASE18_SCATTER_DENSITY,
            jitter: 1.0,
            align_to_normal: false,
            scale_range: (0.6, 1.1),
            rotation: inf_pcg::RotationMode::RandomYaw,
            altitude_offset: 0.0,
        },
        kinds: vec![
            PcgKind {
                mesh: None,
                weight: 3.0,
            },
            PcgKind {
                mesh: None,
                weight: 1.0,
            },
        ],
    };
    inf_pcg::PcgDocument::single_layer("ground", vec![rule])
}

/// The committed `.inf_pcg` payload for the sample (document-only envelope -- the
/// player evaluates from its stored lowered document).
pub fn phase18_scatter_pcg_payload() -> inf_pcg::PcgAssetPayload {
    inf_pcg::PcgAssetPayload::new(phase18_scatter_pcg_document())
}

/// Build the Phase 18 gate [`SceneDoc`] -- the composed scene.
pub fn phase18_scatter_scene() -> SceneDoc {
    use inf_ecs::components::{
        Camera, Foliage, FoliageInstance, FoliagePaletteEntry, Light, LightKind, Material, MeshRef,
        PcgVolume, Primitive, SkyAtmosphere, Terrain, TimeOfDay,
    };
    use inf_ecs::math::Vec3d;

    let mut doc = SceneDoc::new();
    doc.set_title("Phase 18 Scatter");

    // -- The ground: four authored tiles at the world origin, so world XZ ==
    //    terrain-local XZ and every probe in the gate is the bare generator. --
    doc.create_with_guid(PHASE18_TERRAIN_GUID, SpawnKind::Empty, "Terrain", None);
    insert!(
        doc,
        PHASE18_TERRAIN_GUID,
        Transform::from_translation(DVec3::ZERO)
    );
    {
        let mut terrain = Terrain::configured(PHASE18_TERRAIN_RESOLUTION, PHASE18_TERRAIN_MPS);
        for tz in 0..PHASE18_TERRAIN_TILES {
            for tx in 0..PHASE18_TERRAIN_TILES {
                terrain.data.author_tile((tx, tz), phase18_height);
            }
        }
        terrain.data.clear_dirty();
        terrain.macro_variation = 0.25;
        insert!(doc, PHASE18_TERRAIN_GUID, terrain);
    }

    // -- The meshlet slabs: standing instances of the vgeom-demo's dense mesh,
    //    referenced BY GUID (the binary is not duplicated into this sample). --
    for j in 0..PHASE18_SLAB_GRID {
        for i in 0..PHASE18_SLAB_GRID {
            let idx = (j * PHASE18_SLAB_GRID + i) as u128;
            let guid = Uuid::from_u128(PHASE18_SLAB_BASE + idx);
            doc.create_with_guid(guid, SpawnKind::Empty, &format!("Slab {i}x{j}"), None);
            let p = phase18_slab_position(i, j);
            insert!(
                doc,
                guid,
                Transform {
                    translation: Vec3d::new(p.x, p.y, p.z),
                    // Stand the displaced plane up so it occludes.
                    rotation: Vec3d::new(90.0, 0.0, 0.0),
                    scale: Vec3d::splat(PHASE18_SLAB_SCALE),
                }
            );
            insert!(
                doc,
                guid,
                MeshRef {
                    primitive: Primitive::Cube,
                    asset: Some(VGEOM_DEMO_MESH_GUID),
                }
            );
            let t = idx as f32 / (PHASE18_SLAB_GRID * PHASE18_SLAB_GRID) as f32;
            insert!(
                doc,
                guid,
                Material {
                    base_color: Color::new(0.42 + 0.28 * t, 0.46, 0.62 - 0.24 * t, 1.0),
                    metallic: 0.0,
                    roughness: 0.65,
                    emissive: Color::new(0.0, 0.0, 0.0, 1.0),
                    ..Default::default()
                }
            );
        }
    }

    // -- The bulk scatter: a PCG volume over the middle 256 m of the world. --
    doc.create_with_guid(PHASE18_PCG_GUID, SpawnKind::Empty, "Ground Cover", None);
    insert!(
        doc,
        PHASE18_PCG_GUID,
        Transform::from_translation(phase18_world_center())
    );
    insert!(
        doc,
        PHASE18_PCG_GUID,
        PcgVolume {
            graph: Some(PHASE18_PCG_ASSET_GUID),
            extent: Vec2d::new(PHASE18_SCATTER_EXTENT, PHASE18_SCATTER_EXTENT),
            seed: 0,
            draw_distance: PHASE18_SCATTER_DRAW_DISTANCE,
            ..Default::default()
        }
    );

    // -- The painted scatter: a small Foliage patch (persisted instances). --
    doc.create_with_guid(PHASE18_FOLIAGE_GUID, SpawnKind::Empty, "Foliage", None);
    {
        let c = phase18_world_center();
        insert!(
            doc,
            PHASE18_FOLIAGE_GUID,
            Transform::from_translation(DVec3::new(c.x, phase18_height(c.x, c.z), c.z))
        );
        let span = (PHASE18_FOLIAGE_GRID as f64 - 1.0) * 0.5 * PHASE18_FOLIAGE_SPACING;
        let mut instances = Vec::new();
        for jz in 0..PHASE18_FOLIAGE_GRID {
            for ix in 0..PHASE18_FOLIAGE_GRID {
                let n = (jz * PHASE18_FOLIAGE_GRID + ix) as u32;
                instances.push(FoliageInstance {
                    position: Vec3d::new(
                        ix as f64 * PHASE18_FOLIAGE_SPACING - span,
                        0.0,
                        jz as f64 * PHASE18_FOLIAGE_SPACING - span,
                    ),
                    rotation: Vec3d::new(0.0, (n * 23 % 360) as f64, 0.0),
                    scale: 1.0 + (n % 3) as f64 * 0.25,
                    // Two palette slots, so both buckets of the projector's
                    // per-primitive-kind split are exercised.
                    kind: n % 2,
                });
            }
        }
        insert!(
            doc,
            PHASE18_FOLIAGE_GUID,
            Foliage {
                palette: vec![
                    FoliagePaletteEntry {
                        primitive: Primitive::Sphere,
                        tint: Color::new(0.24, 0.48, 0.20, 1.0),
                    },
                    FoliagePaletteEntry {
                        primitive: Primitive::Cone,
                        tint: Color::new(0.32, 0.40, 0.16, 1.0),
                    },
                ],
                instances,
            }
        );
    }

    // -- The sky authority: a fast clock, so GI's key light actually moves. --
    doc.create_with_guid(PHASE18_SKY_GUID, SpawnKind::Empty, "Sky", None);
    insert!(
        doc,
        PHASE18_SKY_GUID,
        Transform::from_translation(DVec3::ZERO)
    );
    insert!(
        doc,
        PHASE18_SKY_GUID,
        TimeOfDay {
            seconds: 30_000.0,
            day_of_year: 172,
            latitude_deg: 48.9,
            longitude_deg: 0.0,
            // 600x -- the same ramp rate the Phase 17 gate uses, so a short
            // scripted run sweeps the sun through a real arc.
            rate: 600.0,
        }
    );
    insert!(
        doc,
        PHASE18_SKY_GUID,
        SkyAtmosphere {
            clouds_enabled: false,
            weather_enabled: false,
            ..SkyAtmosphere::default()
        }
    );

    // -- An authored directional sun beside the sky authority. --
    doc.create_with_guid(PHASE18_SUN_GUID, SpawnKind::Empty, "Sun", None);
    insert!(
        doc,
        PHASE18_SUN_GUID,
        Transform {
            translation: Vec3d::new(0.0, 200.0, 0.0),
            rotation: Vec3d::new(-50.0, -30.0, 0.0),
            scale: Vec3d::ONE,
        }
    );
    insert!(
        doc,
        PHASE18_SUN_GUID,
        Light {
            kind: LightKind::Directional,
            color: Color::new(1.0, 0.97, 0.9, 1.0),
            intensity: 2.5,
            ..Default::default()
        }
    );

    // -- A camera low over the ground at one corner of the scatter volume. --
    doc.create_with_guid(PHASE18_CAMERA_GUID, SpawnKind::Empty, "Camera", None);
    {
        let c = phase18_world_center();
        let eye = DVec3::new(
            c.x - PHASE18_SCATTER_EXTENT,
            0.0,
            c.z - PHASE18_SCATTER_EXTENT,
        );
        insert!(
            doc,
            PHASE18_CAMERA_GUID,
            Transform::from_translation(DVec3::new(
                eye.x,
                phase18_height(eye.x, eye.z) + 4.0,
                eye.z
            ))
        );
        insert!(doc, PHASE18_CAMERA_GUID, Camera::default());
    }

    doc.world_mut().propagate();
    doc.mark_saved();
    doc
}

/// The repo-root `samples/phase18-scatter/` directory.
pub fn phase18_scatter_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../samples/phase18-scatter")
}

/// Write the committed Phase 18 gate files (regeneration path): the `.inf_lvl`
/// (+ sidecar), the `.inf_pcg` graph (+ its inf_asset sidecar so the
/// `PcgVolume.graph` ref resolves through the AssetDb / cooked pack), + README.
///
/// The dense `.inf_mesh` the slabs reference is **not** written here: it is
/// `samples/vgeom-demo/Dense.inf_mesh`, shared by GUID (see the module note).
pub fn write_phase18_scatter() -> Result<(), String> {
    let dir = phase18_scatter_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("mkdir: {e}"))?;

    crate::scene::serialize::save(
        &phase18_scatter_scene(),
        &dir.join("Phase18Scatter.inf_lvl"),
        Some(PHASE18_LEVEL_GUID),
    )?;

    let pcg_bytes = phase18_scatter_pcg_payload()
        .encode()
        .map_err(|e| format!("encode pcg: {e}"))?;
    let pcg_path = dir.join("GroundCover.inf_pcg");
    std::fs::write(&pcg_path, &pcg_bytes).map_err(|e| format!("write pcg: {e}"))?;
    inf_asset::AssetSidecar::new(
        inf_asset::AssetId(PHASE18_PCG_ASSET_GUID),
        inf_asset::AssetKind::Pcg,
        inf_asset::ContentHash::of(&pcg_bytes),
    )
    .save(&pcg_path)
    .map_err(|e| format!("write pcg sidecar: {e}"))?;

    std::fs::write(dir.join("README.md"), PHASE18_SCATTER_README)
        .map_err(|e| format!("write readme: {e}"))?;
    Ok(())
}

const PHASE18_SCATTER_README: &str = "# Phase 18 Scatter (the phase gate scene)\n\n\
Generated by `inf_editor_core::samples::phase18_scatter_scene` -- the **composed**\n\
gate scene for Phase 18 (Lumen-class GI + Nanite completion). Everything the phase\n\
built, in one level.\n\n\
- `Phase18Scatter.inf_lvl` -- the scene. A four-tile `Terrain`, a 5x5 grid of\n\
  **standing meshlet slabs**, a `PcgVolume` carrying 102 400 scatter instances, a\n\
  small painted `Foliage` patch, a `TimeOfDay` + `SkyAtmosphere` authority running\n\
  a 600x clock, an authored sun, and a camera.\n\
- `GroundCover.inf_pcg` -- the scatter graph the volume evaluates on load. Its\n\
  instances are a **derived cache**, never persisted in the level (editor\n\
  `pcg_evaluate`, shipped/PIE player `evaluate_pcg_volumes`).\n\
- `Dense.inf_mesh` -- **NOT committed here.** The slabs reference\n\
  `samples/vgeom-demo/Dense.inf_mesh` **by GUID**\n\
  (`samples::VGEOM_DEMO_MESH_GUID`); duplicating a 1 MB mesh binary to give this\n\
  sample its own copy would buy nothing but a second thing to keep in sync. The\n\
  gate (`runtime/inf-player/tests/phase18_gate.rs`) copies **both** sample\n\
  directories into its throwaway project's `Content`, so the cook sees one mesh\n\
  and derives one `.inf_vmesh` from it.\n\n\
## Why the scatter is split PCG + Foliage\n\n\
A `PcgVolume`'s `evaluated` cache is not persisted -- it is recomputed on load from\n\
the committed `.inf_pcg` -- so 102 400 instances cost the `.inf_lvl` nothing.\n\
`Foliage::instances` **is** persisted, so 100k of those would be a megabyte of\n\
committed level. The bulk is therefore PCG and the `Foliage` entity is a small\n\
16-instance patch that covers the painted path (two palette slots, so both buckets\n\
of the projector's per-primitive-kind split are exercised). The level stays ~15 KB\n\
while the loaded world carries six figures of scatter -- the same \"small level,\n\
huge scene\" property the vgeom demo has for triangles.\n\n\
## Why the density field is a constant\n\n\
The PCG kernel is fully specified, so a constant field makes the instance count\n\
arithmetic rather than luck: 8x8 cells of 32 m, `1.6 * 32^2` budget each -> a 40x40\n\
jittered sub-grid -> `64 * 1600 = 102 400`, pinned as\n\
`samples::PHASE18_SCATTER_INSTANCES`. terrain-demo already gates a noise x slope\n\
sampler; what this sample needs is a *stated* count at the 100k scale, and a count\n\
that moved whenever someone retuned a noise frequency would be a fixture that rots.\n\n\
## Why the slabs stand up\n\n\
The dense mesh is a displaced plane. Laid flat (as in vgeom-demo) it occludes\n\
almost nothing; rotated 90 deg about X it is a wall. Occluders are what make the\n\
P18.1 two-pass proof and the P18.5 `occluded` counter non-vacuous -- \"occlusion on\n\
is pixel-identical to occlusion off\" is trivially true when nothing is occluded.\n\
The terrain's relief (~40 m peak-to-trough over 512 m) is steep for the same\n\
reason: hills put scatter instances behind something.\n\n\
## The gate (`runtime/inf-player/tests/phase18_gate.rs`)\n\n\
1. the composed frame trace -- pixels, meshlet residency, GI audit and scatter\n\
   audit -- is byte-identical across two fresh renderers over a camera path;\n\
2. cooked == uncooked (PIE == shipping) on the projected trace;\n\
3. the 10.6M-triangle vgeom gate still holds with GI, scatter and streaming all\n\
   on -- P18.1's subtractive proof survives composition;\n\
4. the instance-cull counters are real (frustum / occluded / distance / mesh /\n\
   impostor all nonzero) and deterministic;\n\
5. the composed frame is inside the frame budget, with per-system costs printed;\n\
6. the golden inventory is exactly the 41 committed PNGs.\n\n\
Regenerate with `INF_BLESS_SAMPLES=1 cargo test -p inf-editor-core samples`.\n";

// ── Phase 19: the town (biomes × grammar × enterable buildings) ─────────────
//
// The composed Phase 19 gate scene. A partitioned level over a multi-tile
// biome-painted terrain, a spline road with a grammar fence running its whole
// length, and **seven building lots — one per archetype**, each a `PcgVolume`
// carrying its own tiny `.inf_pcg`.
//
// **One volume is one lot, and that is the model rather than a workaround.** A
// building's footprint is the volume's own box, exactly as a `grammar.footprint`
// span defaults to it; seven archetypes on one canvas would need a per-node lot
// offset, which is the volume's transform spelt a second way. The cost is seven
// small graphs instead of one, and the benefit is that the level reads as what
// it is: seven plots on a street.

pub const PHASE19_LEVEL_GUID: Uuid = Uuid::from_u128(0x8419_0000);
pub const PHASE19_TERRAIN_GUID: Uuid = Uuid::from_u128(0x8419_0001);
/// The road: a `Spline` **and** the `PcgVolume` whose grammar fences it. A blank
/// `grammar.spline` entity means "the entity this graph evaluates on", so the
/// two live on one actor and no GUID is typed anywhere.
pub const PHASE19_ROAD_GUID: Uuid = Uuid::from_u128(0x8419_0002);
pub const PHASE19_SKY_GUID: Uuid = Uuid::from_u128(0x8419_0003);
pub const PHASE19_SUN_GUID: Uuid = Uuid::from_u128(0x8419_0004);
pub const PHASE19_CAMERA_GUID: Uuid = Uuid::from_u128(0x8419_0005);
/// The walker whose position drives cell activation (the partition's streaming
/// source), so the gate can watch a building's cell come and go.
pub const PHASE19_WALKER_GUID: Uuid = Uuid::from_u128(0x8419_0006);

/// Asset GUID of the committed `Town.inf_biomes`.
pub const PHASE19_BIOME_SET_GUID: Uuid = Uuid::from_u128(0x8419_00A0);
/// Asset GUID of the committed `Roadside.inf_pcg` (the fence grammar).
pub const PHASE19_ROAD_PCG_GUID: Uuid = Uuid::from_u128(0x8419_00A1);
/// Base **entity** GUID of the seven building lots (add the archetype index).
const PHASE19_LOT_BASE: u128 = 0x8419_0100;
/// Base **asset** GUID of the seven per-lot `.inf_pcg` graphs.
const PHASE19_LOT_PCG_BASE: u128 = 0x8419_0200;
/// Base **entity** GUID of the streamed street lamps (add the index).
const PHASE19_LAMP_BASE: u128 = 0x8419_0300;
/// How many street lamps run the length of the road. These are the level's
/// **streamed** content — see the note on [`phase19_town_scene`].
pub const PHASE19_LAMPS: usize = 12;

/// Terrain samples per tile side (`17` closes a tile on a 16-cell lattice).
pub const PHASE19_TERRAIN_RESOLUTION: u32 = 17;
/// World metres per terrain sample — 16 m, so one tile spans 256 m.
pub const PHASE19_TERRAIN_MPS: f64 = 16.0;
/// Authored tiles per side (2 × 2 → a 512 m world).
pub const PHASE19_TERRAIN_TILES: i32 = 2;

/// Partition cell edge, metres. Deliberately **128 m, half the engine default**,
/// so the 512 m world holds sixteen cells and the seven lots land in more than
/// one of them — a gate that binned every building into cell `(0,0)` would prove
/// nothing about binning.
pub const PHASE19_CELL_SIZE_M: f64 = 128.0;
pub const PHASE19_ACTIVATION_RADIUS_M: f64 = 160.0;
pub const PHASE19_PREFETCH_MARGIN_M: f64 = 128.0;

/// The two painted biome ids: the built-up strip along the road, and the
/// meadow either side of it.
pub const PHASE19_BIOME_TOWN: u8 = 1;
pub const PHASE19_BIOME_MEADOW: u8 = 2;
/// Half-width of the painted town strip either side of the road's Z line,
/// metres.
pub const PHASE19_TOWN_HALF_WIDTH: f64 = 90.0;

/// Half-extent of one building lot in world XZ, metres. The whole footprint is
/// therefore 44 × 30 m, comfortably above every archetype's `min_room` and big
/// enough that the partition really partitions.
pub const PHASE19_LOT_EXTENT: (f64, f64) = (22.0, 15.0);
/// Metres between lot centres along the road.
///
/// **Sized to the terrain, and re-sized when the palette grew** (wave VEN1a).
/// The lots spread symmetrically about the town centre, so the outermost sits
/// `(n - 1) / 2 x pitch` out -- and the terrain is 512 m across, i.e. +-256 m.
/// At seven archetypes and 62 m that reach was 186 m and fitted; at TEN it is
/// 279 m, and the two outermost lots stood off the terrain entirely.
/// `jobs_of`'s fail-closed rule then says "no ground, no building", so `Office`
/// (the leftmost) simply placed nothing and the gate said so.
///
/// **And again at FOURTEEN** (wave EMS1), which is why this doc reads like a
/// ledger: 50 m puts thirteen gaps at 325 m of reach plus a lot's own 22 m
/// half-extent = 347 m against a ±256 m pad, so the same defect would have
/// eaten two lots at each end — silently, because a lot with no ground places
/// nothing and says nothing. 34 m puts the reach at 221 m + 22 = 243 m, inside
/// the pad with thirteen metres to spare, and leaves 24 m of clear ground
/// between two lots on the same side of the road.
///
/// The arithmetic is `(n - 1) / 2 * pitch + half_extent.x <= world_size / 2`,
/// and `the_phase19_lots_all_stand_on_the_terrain` is the arm that checks it
/// rather than this comment.
pub const PHASE19_LOT_PITCH: f64 = 34.0;
/// Metres from the road's Z line to a lot centre.
pub const PHASE19_LOT_SETBACK: f64 = 34.0;
/// Storeys every lot is pinned to, so the gate's stair walk has the same shape
/// for all ten and a retuned archetype range cannot silently make one of them
/// single-storey.
pub const PHASE19_LOT_FLOORS: u32 = 3;

/// World edge length of the sample's terrain, metres (512).
pub fn phase19_world_size() -> f64 {
    PHASE19_TERRAIN_TILES as f64 * (PHASE19_TERRAIN_RESOLUTION as f64 - 1.0) * PHASE19_TERRAIN_MPS
}

/// The centre of the sample world in XZ.
pub fn phase19_world_center() -> DVec3 {
    let c = phase19_world_size() * 0.5;
    DVec3::new(c, 0.0, c)
}

/// The sample's analytic terrain height at world `(x, z)`.
///
/// Bit-portable by construction ([`inf_math::psin64`] / [`pcos64`], never `std`
/// trig — the P14 law), because this becomes committed `.inf_lvl` bytes a cook
/// on one OS and a run on another must agree about.
///
/// Deliberately **gentle** (~6 m over 512 m) where phase18's is steep: a
/// building levels its site from one datum, so a violent slope would put a lot's
/// floor slab metres above the ground on one side. Relief enough to prove the
/// terrain is read, not enough to make the town look posted on stilts.
pub fn phase19_height(x: f64, z: f64) -> f64 {
    3.0 * inf_math::psin64(x * 0.0075) * inf_math::pcos64(z * 0.0062)
        + 1.5 * inf_math::psin64((x + z) * 0.011)
}

/// The road's Z coordinate — it runs straight along world X down the middle of
/// the world, which is what makes the painted town strip a simple band.
pub fn phase19_road_z() -> f64 {
    phase19_world_center().z
}

/// The painted biome id at world `(x, z)`: the town strip within
/// [`PHASE19_TOWN_HALF_WIDTH`] of the road, meadow beyond it.
pub fn phase19_biome_at(_x: f64, z: f64) -> u8 {
    if (z - phase19_road_z()).abs() <= PHASE19_TOWN_HALF_WIDTH {
        PHASE19_BIOME_TOWN
    } else {
        PHASE19_BIOME_MEADOW
    }
}

/// The **entity** GUID of lot `i` — one per archetype, in `ArchetypeId::ALL`
/// order, so `0..ArchetypeId::ALL.len()` (seven when this was written, fourteen
/// since wave EMS1).
pub fn phase19_lot_guid(i: usize) -> Uuid {
    Uuid::from_u128(PHASE19_LOT_BASE + i as u128)
}

/// The **asset** GUID of lot `i`'s `.inf_pcg` graph.
pub fn phase19_lot_pcg_guid(i: usize) -> Uuid {
    Uuid::from_u128(PHASE19_LOT_PCG_BASE + i as u128)
}

/// The world position of lot `i`: laid out along the road, alternating sides so
/// the street has two frontages.
///
/// Derived from `i`, never accumulated (the P17.4 exact-linear rule).
pub fn phase19_lot_position(i: usize) -> DVec3 {
    let n = inf_pcg::ArchetypeId::ALL.len();
    let span = (n as f64 - 1.0) * 0.5 * PHASE19_LOT_PITCH;
    let x = phase19_world_center().x + i as f64 * PHASE19_LOT_PITCH - span;
    let side = if i.is_multiple_of(2) { -1.0 } else { 1.0 };
    let z = phase19_road_z() + side * PHASE19_LOT_SETBACK;
    DVec3::new(x, phase19_height(x, z), z)
}

/// The **entity** GUID of street lamp `i`.
pub fn phase19_lamp_guid(i: usize) -> Uuid {
    Uuid::from_u128(PHASE19_LAMP_BASE + i as u128)
}

/// The world position of street lamp `i` — evenly along the road, derived from
/// `i` rather than accumulated.
pub fn phase19_lamp_position(i: usize) -> DVec3 {
    let size = phase19_world_size();
    let margin = 24.0;
    let x = margin + (size - 2.0 * margin) * (i as f64 + 0.5) / PHASE19_LAMPS as f64;
    let z = phase19_road_z() + if i.is_multiple_of(2) { -7.0 } else { 7.0 };
    DVec3::new(x, phase19_height(x, z), z)
}

/// The road's control points, in the road entity's **local** frame (it sits at
/// the world centre, so local X is world X).
pub fn phase19_road_points() -> Vec<DVec3> {
    let half = phase19_world_size() * 0.5 - 24.0;
    (0..5)
        .map(|i| {
            let t = i as f64 / 4.0;
            DVec3::new(-half + 2.0 * half * t, 0.0, 0.0)
        })
        .collect()
}

/// The committed `Town.inf_biomes` set: two biomes, neither carrying a `.inf_pcg`
/// of its own.
///
/// **The binding is deliberately unused here, and that is a stated choice.** A
/// biome dispatches *scatter* over the region its id owns (P19.3); a building
/// needs a *lot*, and P19.5's answer to that is the `lot` span pin, not a
/// per-biome graph. So the painted ids are what they are for in this sample —
/// vocabulary the terrain carries, gated for round-tripping — and the buildings
/// arrive through volumes.
pub fn phase19_biome_set() -> inf_terrain::BiomeSet {
    let mut set = inf_terrain::BiomeSet::new("Town");
    let mut town = inf_terrain::BiomeDef::new(PHASE19_BIOME_TOWN, "Town");
    town.color = [0.62, 0.58, 0.50, 1.0];
    town.splat_layer = Some(1);
    // P19.2 declared `structure_hint` as inert plain data "because it is what
    // P19.5 will ask a biome for". This is that ask, and it is answered
    // honestly: the hint names a real `ArchetypeId`, and a gate checks that it
    // parses. It is **advisory** — the buildings in this sample arrive through
    // volumes, because a biome owns a region and a building needs a lot.
    town.structure_hint = Some(inf_pcg::ArchetypeId::House.name().to_string());
    let mut meadow = inf_terrain::BiomeDef::new(PHASE19_BIOME_MEADOW, "Meadow");
    meadow.color = [0.30, 0.52, 0.22, 1.0];
    meadow.splat_layer = Some(0);
    set.biomes = vec![town, meadow];
    set
}

/// The roadside graph: a `grammar.rules` fence expanded along the road actor's
/// own `Spline`, merged with a light ground scatter.
pub fn phase19_road_graph() -> inf_graph::Graph {
    let reg = inf_pcg::pcg_registry();
    let mut g = inf_graph::Graph::empty();
    let mut next = 1u32;
    let add = |g: &mut inf_graph::Graph,
               next: &mut u32,
               type_id: &str,
               params: &[(&str, inf_graph::ParamValue)]| {
        let id = inf_graph::NodeId(*next);
        *next += 1;
        let mut m = inf_graph::ParamMap::new();
        for (k, v) in params {
            m.insert((*k).to_string(), v.clone());
        }
        inf_graph::apply_edits(
            g,
            &reg,
            &[inf_graph::GraphEdit::AddNode {
                id,
                type_id: type_id.into(),
                x: 0.0,
                y: 0.0,
                params: m,
            }],
        );
        id
    };
    let link = |g: &mut inf_graph::Graph, from: inf_graph::NodeId, fp: &str, to, tp: &str| {
        inf_graph::apply_edits(
            g,
            &inf_pcg::pcg_registry(),
            &[inf_graph::GraphEdit::Connect {
                link: inf_graph::Link {
                    from,
                    from_port: fp.into(),
                    to,
                    to_port: tp.into(),
                },
            }],
        );
    };
    use inf_graph::ParamValue as P;

    let rules = add(
        &mut g,
        &mut next,
        "grammar.rules",
        &[("rules", P::Text(PHASE19_FENCE_RULES.into()))],
    );
    let span = add(
        &mut g,
        &mut next,
        "grammar.spline",
        &[("samples_per_segment", P::Int(8))],
    );
    let expand = add(
        &mut g,
        &mut next,
        "grammar.expand",
        &[
            ("name", P::Text("roadside-fence".into())),
            ("seed", P::Int(1905)),
        ],
    );
    link(&mut g, rules, "out", expand, "rules");
    link(&mut g, span, "out", expand, "span");

    let density = add(
        &mut g,
        &mut next,
        "const.density",
        &[("value", P::Float(1.0))],
    );
    let scatter = add(
        &mut g,
        &mut next,
        "scatter.scatter",
        &[
            ("name", P::Text("verge".into())),
            ("cell_size", P::Float(32.0)),
            ("base_density", P::Float(0.02)),
            ("seed", P::Int(1906)),
            ("scale_min", P::Float(0.6)),
            ("scale_max", P::Float(1.2)),
        ],
    );
    link(&mut g, density, "out", scatter, "density");

    let merge = add(&mut g, &mut next, "scatter.merge", &[]);
    link(&mut g, expand, "out", merge, "a");
    link(&mut g, scatter, "out", merge, "b");
    let out = add(&mut g, &mut next, "output.pcg", &[]);
    link(&mut g, merge, "out", out, "scatter");
    g
}

/// The rule text the roadside fence expands. Posts at both ends of every span,
/// panels filling the middle, one bay in five a gate — and every module carries
/// a `collider`, so the fence is a real barrier rather than a picture of one.
const PHASE19_FENCE_RULES: &str = "\
# Roadside fence (P19.5: `collider` makes it solid).
module Post  = size 0.2 offset 0,0.75,0.1 collider 0.1,0.75,0.1
module Panel = size 2.4 offset 0,0.6,1.2  collider 0.05,0.6,1.2
module Gate  = size 2.4 offset 0,0.55,1.2 collider 0.04,0.55,1.2

Fence -> Post Bay* Post
Bay   -> Panel | Gate@0.25
";

/// Lot `i`'s graph: one archetype, one planner, straight into the sink.
pub fn phase19_lot_graph(i: usize) -> inf_graph::Graph {
    let reg = inf_pcg::pcg_registry();
    let mut g = inf_graph::Graph::empty();
    let id = inf_pcg::ArchetypeId::ALL[i];
    use inf_graph::ParamValue as P;
    let add = |g: &mut inf_graph::Graph,
               n: u32,
               type_id: &str,
               params: &[(&str, inf_graph::ParamValue)]| {
        let node = inf_graph::NodeId(n);
        let mut m = inf_graph::ParamMap::new();
        for (k, v) in params {
            m.insert((*k).to_string(), v.clone());
        }
        inf_graph::apply_edits(
            g,
            &reg,
            &[inf_graph::GraphEdit::AddNode {
                id: node,
                type_id: type_id.into(),
                x: 0.0,
                y: 0.0,
                params: m,
            }],
        );
        node
    };
    let arch = add(
        &mut g,
        1,
        "building.archetype",
        &[
            ("archetype", P::Enum(id.name().into())),
            ("floors", P::Int(PHASE19_LOT_FLOORS as i64)),
            ("furnish", P::Bool(true)),
        ],
    );
    let plan = add(
        &mut g,
        2,
        "building.plan",
        &[
            ("name", P::Text(id.name().to_lowercase())),
            ("seed", P::Int(1900 + i as i64)),
        ],
    );
    let out = add(&mut g, 3, "output.pcg", &[]);
    for (from, fp, to, tp) in [
        (arch, "out", plan, "archetype"),
        (plan, "out", out, "scatter"),
    ] {
        inf_graph::apply_edits(
            &mut g,
            &inf_pcg::pcg_registry(),
            &[inf_graph::GraphEdit::Connect {
                link: inf_graph::Link {
                    from,
                    from_port: fp.into(),
                    to,
                    to_port: tp.into(),
                },
            }],
        );
    }
    g
}

/// The `.inf_pcg` payload for a graph: the authored graph (the source of truth
/// since P19.3) plus its lowered document mirror.
fn phase19_payload(graph: &inf_graph::Graph) -> inf_pcg::PcgAssetPayload {
    let lowered = inf_pcg::lower_graph(graph, &inf_pcg::pcg_registry());
    inf_pcg::PcgAssetPayload::from_graph(graph, lowered.document)
}

/// The roadside `.inf_pcg` payload.
pub fn phase19_road_payload() -> inf_pcg::PcgAssetPayload {
    phase19_payload(&phase19_road_graph())
}

/// Lot `i`'s `.inf_pcg` payload.
pub fn phase19_lot_payload(i: usize) -> inf_pcg::PcgAssetPayload {
    phase19_payload(&phase19_lot_graph(i))
}

/// Build the Phase 19 gate [`SceneDoc`] — the composed town.
///
/// # THE `AlwaysLoaded` INTERPLAY, stated
///
/// Every `PcgVolume` in this level — the seven lots and the road — carries
/// [`AlwaysLoaded`](inf_ecs::components::AlwaysLoaded), and the reason is a real
/// engine property rather than a convenience:
///
/// **PCG evaluation is a load-time pass.** `evaluate_pcg_volumes` runs once,
/// over the world the level builder produced; cell streaming spawns entities
/// *afterwards*, and nothing re-runs evaluation for them. A `PcgVolume` binned
/// into a grid cell would therefore stream in and stay empty — a building lot
/// with no building on it. That is the standing P10.6 remainder ("evaluation
/// still runs once, at load"), restated by P19.4 and unchanged here; a batch
/// that hid it by never streaming a volume would have hidden it.
///
/// So the volumes are persistent **by declaration**, and the level's *streamed*
/// content is the twelve street lamps — ordinary placed entities that bin by
/// position, activate as the walker approaches, and give the partition arm of
/// the gate something real to be about. What the gate then asserts about the
/// buildings is the property that survives the gap: a lot's instances all lie
/// inside the lot's own footprint, so the day evaluation follows streaming they
/// are already in the right cell.
///
/// A building bigger than a cell would still be one entity in one cell. At the
/// sample's deliberately small 128 m cell a 44 m lot fits comfortably; a 400 m
/// megastructure would want either a larger cell or the same `AlwaysLoaded`
/// declaration, and that is a content decision the engine should not make.
pub fn phase19_town_scene() -> SceneDoc {
    use crate::scene::serialize::{LevelSettings, PartitionSettings};
    use inf_ecs::components::{
        AlwaysLoaded, Camera, Light, LightKind, Material, MeshRef, PcgVolume, Primitive,
        SkyAtmosphere, Spline, SplineInterp, StreamingSource, Terrain, TimeOfDay,
    };
    use inf_ecs::math::Vec3d;

    let mut doc = SceneDoc::new();
    doc.set_title("Phase 19 Town");
    doc.set_settings(LevelSettings {
        partition: PartitionSettings {
            enabled: true,
            cell_size_m: PHASE19_CELL_SIZE_M,
            activation_radius_m: PHASE19_ACTIVATION_RADIUS_M,
            prefetch_margin_m: PHASE19_PREFETCH_MARGIN_M,
        },
        ..LevelSettings::default()
    });

    // ── The ground: four authored tiles, biome-painted, always loaded. ──
    //
    // `AlwaysLoaded` for the same reason phase16's terrain carries it: a Terrain
    // occupies space, so the partitioner would bin the whole heightfield into
    // one cell and the ground would vanish as the walker left it.
    doc.create_with_guid(PHASE19_TERRAIN_GUID, SpawnKind::Empty, "Terrain", None);
    insert!(
        doc,
        PHASE19_TERRAIN_GUID,
        Transform::from_translation(DVec3::ZERO)
    );
    insert!(doc, PHASE19_TERRAIN_GUID, AlwaysLoaded);
    {
        let mut terrain = Terrain::configured(PHASE19_TERRAIN_RESOLUTION, PHASE19_TERRAIN_MPS);
        for tz in 0..PHASE19_TERRAIN_TILES {
            for tx in 0..PHASE19_TERRAIN_TILES {
                terrain.data.author_tile((tx, tz), phase19_height);
            }
        }
        // Paint the biome ids sample by sample — the same per-sample layer the
        // P19.2 brush writes, authored analytically so the sample regenerates
        // bit-identically.
        let res = PHASE19_TERRAIN_RESOLUTION;
        let step = PHASE19_TERRAIN_MPS;
        for tz in 0..PHASE19_TERRAIN_TILES {
            for tx in 0..PHASE19_TERRAIN_TILES {
                let Some(tile) = terrain.data.get_tile_mut((tx, tz)) else {
                    continue;
                };
                let origin_x = tx as f64 * (res as f64 - 1.0) * step;
                let origin_z = tz as f64 * (res as f64 - 1.0) * step;
                for j in 0..res {
                    for i in 0..res {
                        let x = origin_x + i as f64 * step;
                        let z = origin_z + j as f64 * step;
                        tile.set_biome_sample(res, i, j, phase19_biome_at(x, z));
                    }
                }
            }
        }
        terrain.data.clear_dirty();
        terrain.biome_set = Some(PHASE19_BIOME_SET_GUID);
        terrain.macro_variation = 0.2;
        insert!(doc, PHASE19_TERRAIN_GUID, terrain);
    }

    // ── The road: a Spline AND the volume whose grammar fences it. ──
    doc.create_with_guid(PHASE19_ROAD_GUID, SpawnKind::Empty, "Road", None);
    {
        let c = phase19_world_center();
        let z = phase19_road_z();
        insert!(
            doc,
            PHASE19_ROAD_GUID,
            Transform::from_translation(DVec3::new(c.x, phase19_height(c.x, z), z))
        );
        insert!(doc, PHASE19_ROAD_GUID, AlwaysLoaded);
        insert!(
            doc,
            PHASE19_ROAD_GUID,
            Spline {
                points: phase19_road_points()
                    .into_iter()
                    .map(|p| Vec3d::new(p.x, p.y, p.z))
                    .collect(),
                closed: false,
                interp: SplineInterp::Linear,
            }
        );
        let half = phase19_world_size() * 0.5;
        insert!(
            doc,
            PHASE19_ROAD_GUID,
            PcgVolume {
                graph: Some(PHASE19_ROAD_PCG_GUID),
                extent: Vec2d::new(half, PHASE19_TOWN_HALF_WIDTH),
                seed: 19,
                draw_distance: 600.0,
                ..Default::default()
            }
        );
    }

    // ── The lots: one building per archetype, one volume each. ──
    for (i, id) in inf_pcg::ArchetypeId::ALL.into_iter().enumerate() {
        let guid = phase19_lot_guid(i);
        doc.create_with_guid(
            guid,
            SpawnKind::Empty,
            &format!("Lot — {}", id.name()),
            None,
        );
        let p = phase19_lot_position(i);
        insert!(doc, guid, Transform::from_translation(p));
        // See the function docs: a streamed volume never evaluates, because
        // evaluation is a load-time pass.
        insert!(doc, guid, AlwaysLoaded);
        insert!(
            doc,
            guid,
            PcgVolume {
                graph: Some(phase19_lot_pcg_guid(i)),
                extent: Vec2d::new(PHASE19_LOT_EXTENT.0, PHASE19_LOT_EXTENT.1),
                // The volume seed folds into the pass seed, so two lots sharing
                // one archetype would still differ. Here every lot has its own
                // graph, and the seed is what makes the *plan* per-lot.
                seed: 100 + i as u32,
                draw_distance: 600.0,
                ..Default::default()
            }
        );
    }

    // ── The street lamps: the level's STREAMED content. ──
    //
    // Ordinary placed entities with no `AlwaysLoaded` marker, so the partitioner
    // bins each by its own world XZ and the cell manager activates it as the
    // walker approaches. They are what makes the partition arm of the gate about
    // something: with only persistent entities the `.inf_part` would hold one
    // cell and prove nothing.
    for i in 0..PHASE19_LAMPS {
        let guid = phase19_lamp_guid(i);
        doc.create_with_guid(guid, SpawnKind::Empty, &format!("Lamp {i}"), None);
        let p = phase19_lamp_position(i);
        insert!(
            doc,
            guid,
            Transform {
                translation: Vec3d::new(p.x, p.y + 2.4, p.z),
                rotation: Vec3d::ZERO,
                scale: Vec3d::new(0.12, 2.4, 0.12),
            }
        );
        insert!(
            doc,
            guid,
            MeshRef {
                primitive: Primitive::Cylinder,
                asset: None,
            }
        );
        insert!(
            doc,
            guid,
            Material {
                base_color: Color::new(0.18, 0.19, 0.22, 1.0),
                metallic: 0.6,
                roughness: 0.45,
                ..Default::default()
            }
        );
    }

    // ── The walker: the partition's streaming source. ──
    doc.create_with_guid(PHASE19_WALKER_GUID, SpawnKind::Empty, "Walker", None);
    {
        let p = phase19_lot_position(0);
        insert!(
            doc,
            PHASE19_WALKER_GUID,
            Transform::from_translation(DVec3::new(p.x, p.y + 1.8, p.z))
        );
        insert!(
            doc,
            PHASE19_WALKER_GUID,
            StreamingSource {
                radius_m: PHASE19_ACTIVATION_RADIUS_M,
            }
        );
    }

    // ── Sky, sun, camera. ──
    doc.create_with_guid(PHASE19_SKY_GUID, SpawnKind::Empty, "Sky", None);
    insert!(doc, PHASE19_SKY_GUID, AlwaysLoaded);
    insert!(
        doc,
        PHASE19_SKY_GUID,
        TimeOfDay {
            // 10:30 UTC — a solid, unambiguous daytime sun.
            seconds: 10.5 * 3600.0,
            rate: 0.0,
            ..Default::default()
        }
    );
    insert!(doc, PHASE19_SKY_GUID, SkyAtmosphere::default());

    doc.create_with_guid(PHASE19_SUN_GUID, SpawnKind::Empty, "Sun", None);
    insert!(doc, PHASE19_SUN_GUID, AlwaysLoaded);
    insert!(
        doc,
        PHASE19_SUN_GUID,
        Transform {
            translation: Vec3d::new(0.0, 120.0, 0.0),
            rotation: Vec3d::new(-52.0, 34.0, 0.0),
            scale: Vec3d::splat(1.0),
        }
    );
    insert!(
        doc,
        PHASE19_SUN_GUID,
        Light {
            kind: LightKind::Directional,
            intensity: 3.4,
            ..Default::default()
        }
    );

    doc.create_with_guid(PHASE19_CAMERA_GUID, SpawnKind::Empty, "Camera", None);
    insert!(doc, PHASE19_CAMERA_GUID, AlwaysLoaded);
    {
        let c = phase19_world_center();
        insert!(
            doc,
            PHASE19_CAMERA_GUID,
            Transform {
                translation: Vec3d::new(c.x - 120.0, phase19_height(c.x, c.z) + 34.0, c.z - 140.0),
                rotation: Vec3d::new(-14.0, 32.0, 0.0),
                scale: Vec3d::splat(1.0),
            }
        );
    }
    insert!(doc, PHASE19_CAMERA_GUID, Camera::default());

    doc.world_mut().propagate();
    doc.mark_saved();
    doc
}

pub fn phase19_town_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../samples/phase19-town")
}

/// Write the committed Phase 19 gate files: the `.inf_lvl` (+ sidecar), the
/// biome set, the roadside graph, the seven lot graphs (each with its inf_asset
/// sidecar so the `PcgVolume.graph` refs resolve through the AssetDb / cooked
/// pack), and the README.
pub fn write_phase19_town() -> Result<(), String> {
    let dir = phase19_town_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("mkdir: {e}"))?;

    crate::scene::serialize::save(
        &phase19_town_scene(),
        &dir.join("Phase19Town.inf_lvl"),
        Some(PHASE19_LEVEL_GUID),
    )?;

    let biome_bytes =
        inf_asset::encode(&phase19_biome_set()).map_err(|e| format!("encode biomes: {e}"))?;
    write_phase19_asset(
        &dir.join("Town.inf_biomes"),
        &biome_bytes,
        PHASE19_BIOME_SET_GUID,
        inf_asset::AssetKind::BiomeSet,
    )?;

    let road_bytes = phase19_road_payload()
        .encode()
        .map_err(|e| format!("encode road pcg: {e}"))?;
    write_phase19_asset(
        &dir.join("Roadside.inf_pcg"),
        &road_bytes,
        PHASE19_ROAD_PCG_GUID,
        inf_asset::AssetKind::Pcg,
    )?;

    for (i, id) in inf_pcg::ArchetypeId::ALL.into_iter().enumerate() {
        let bytes = phase19_lot_payload(i)
            .encode()
            .map_err(|e| format!("encode {} pcg: {e}", id.name()))?;
        write_phase19_asset(
            &dir.join(format!("Lot{}.inf_pcg", id.name())),
            &bytes,
            phase19_lot_pcg_guid(i),
            inf_asset::AssetKind::Pcg,
        )?;
    }

    std::fs::write(dir.join("README.md"), PHASE19_TOWN_README)
        .map_err(|e| format!("write readme: {e}"))?;
    Ok(())
}

/// Write one asset payload plus its `inf_asset` sidecar (the stable GUID is what
/// makes a component's asset ref resolve through the AssetDb and the cook).
fn write_phase19_asset(
    path: &std::path::Path,
    bytes: &[u8],
    guid: Uuid,
    kind: inf_asset::AssetKind,
) -> Result<(), String> {
    std::fs::write(path, bytes).map_err(|e| format!("write {}: {e}", path.display()))?;
    inf_asset::AssetSidecar::new(
        inf_asset::AssetId(guid),
        kind,
        inf_asset::ContentHash::of(bytes),
    )
    .save(path)
    .map_err(|e| format!("write sidecar for {}: {e}", path.display()))
}

const PHASE19_TOWN_README: &str = "# Phase 19 Town (the phase gate scene)\n\n\
Generated by `inf_editor_core::samples::phase19_town_scene` -- the **composed**\n\
gate scene for Phase 19 (biomes, PCG grammar & enterable structures).\n\n\
- `Phase19Town.inf_lvl` -- a **partitioned** level (128 m cells) over a four-tile\n\
  biome-painted `Terrain`, a `Spline` road with the `PcgVolume` that fences it,\n\
  **seven building lots (one per archetype)**, a walker that is the streaming\n\
  source, a `TimeOfDay` + `SkyAtmosphere` authority, a sun and a camera.\n\
- `Town.inf_biomes` -- two biomes (Town / Meadow); the strip within 90 m of the\n\
  road is painted Town, everything beyond it Meadow.\n\
- `Roadside.inf_pcg` -- a `grammar.rules` fence expanded along the road actor's\n\
  own spline, merged with a light verge scatter. Every fence module declares a\n\
  `collider`, so the fence is a barrier rather than a picture of one.\n\
- `Lot<Archetype>.inf_pcg` (x7) -- `building.archetype -> building.plan ->\n\
  output.pcg`, one per palette: Office, Apartment, Industrial, House, Estate,\n\
  Hotel, Shop.\n\n\
## Why one volume per lot\n\n\
A building's footprint is its volume's own box -- the same default a\n\
`grammar.footprint` span has. Seven archetypes on one canvas would need a per-node\n\
lot offset, which is the volume's transform spelt a second way. So the level reads\n\
as what it is: seven plots on a street, each with its own tiny graph.\n\n\
## Why the terrain is authored inline, not streamed from an `.inf_terrain`\n\n\
Phase 16 already gates heightfield tile streaming end to end\n\
(`phase16_gate`, `streamed_terrain`). What Phase 19 needs from streaming is that\n\
**buildings respect cells**, which is the *entity* partition -- so this level turns\n\
world partition on with a deliberately small 128 m cell (half the engine default)\n\
and authors its 512 m heightfield inline. Sixteen cells over the world means the\n\
seven lots land in more than one of them, which is what makes the binning\n\
assertion non-vacuous; a level whose every building sat in cell (0,0) would prove\n\
nothing.\n\n\
## Why the relief is gentle\n\n\
A building levels its site: `base_y` is sampled once, at the footprint centre, and\n\
every slab, wall and stair shares that datum. phase18's ~40 m of relief over 512 m\n\
exists to put scatter behind occluders; here it would post a 44 m lot on stilts.\n\
~6 m over 512 m is enough to prove the terrain is read and not enough to lift a\n\
floor slab off the ground.\n\n\
## The buildings are derived, so the level stays small\n\n\
`PcgVolume.evaluated` and `PcgVolume.structures` are both `#[serde(skip)]`: the\n\
instances **and the colliders** are recomputed on load from the committed\n\
`.inf_pcg` (editor `pcg_evaluate`, shipped/PIE player `evaluate_pcg_volumes`).\n\
Seven fully furnished three-storey buildings cost the `.inf_lvl` nothing.\n\n\
## The gate (`runtime/inf-player/tests/phase19_gate.rs`)\n\n\
1. full-trace determinism across two fresh loads (population + building solids +\n\
   partition residency);\n\
2. cooked == uncooked, bit for bit;\n\
3. PIE == shipping, bit for bit, on the placed set;\n\
4. **enterability** -- for one building per archetype: the room graph is connected,\n\
   every door opening's rect holds no collider, and every floor is reachable from\n\
   OUTSIDE by a graph walk through the entrance, the doors and the stair cores;\n\
5. the lots bin into the partition cells their transforms say they do;\n\
6. the composed scene loads inside the frame budget.\n\n\
Regenerate with `INF_BLESS_SAMPLES=1 cargo test -p inf-editor-core samples`.\n";

// ── Phase 20 coastal (the P20.4 gate scene) ─────────────────────────────────
//
// The plan's own done-when sentence, built: "a coastal scene — ocean plus a
// spline river fed by a lake — carries buoyant physics objects, replays
// deterministically on the physics trace, and PIE == shipping."

/// World side of the coastal terrain, metres.
pub const PHASE20_WORLD_M: f64 = 512.0;
/// Samples per terrain tile side.
pub const PHASE20_TERRAIN_RESOLUTION: u32 = 129;
/// Metres between adjacent height samples.
pub const PHASE20_TERRAIN_MPS: f64 = 4.0;
/// Terrain tiles per side (129 samples at 4 m ⇒ 512 m per tile ⇒ one tile).
pub const PHASE20_TERRAIN_TILES: u32 = 1;
/// Still-water level of the ocean, metres of world Y. Sea level is the datum the
/// whole scene is authored against.
pub const PHASE20_SEA_LEVEL_M: f64 = 0.0;
/// Still-water level of the head lake, metres.
pub const PHASE20_LAKE_LEVEL_M: f64 = 33.6;
/// Centre of the head lake in world XZ, metres.
pub const PHASE20_LAKE_CENTER: (f64, f64) = (60.0, 250.0);
/// Half-extent of the head lake in world XZ, metres.
pub const PHASE20_LAKE_HALF_M: f64 = 28.0;
/// Radius of the basin dug for the head lake, metres.
pub const PHASE20_BASIN_RADIUS_M: f64 = 30.0;
/// Depth of that basin at its centre, metres.
pub const PHASE20_BASIN_DEPTH_M: f64 = 8.0;
/// Crates floating on the sea.
pub const PHASE20_SEA_CRATES: usize = 6;
/// Crates floating on the head lake.
pub const PHASE20_LAKE_CRATES: usize = 2;

const PHASE20_LEVEL_GUID: Uuid = uuid::uuid!("20040000-0000-4000-8000-000000000001");
/// The swimmer's blueprint class asset.
pub const PHASE20_SWIMMER_ACTOR_GUID: Uuid = uuid::uuid!("20040000-0000-4000-8000-000000000002");
const PHASE20_TERRAIN_GUID: Uuid = uuid::uuid!("20040000-0000-4000-8000-000000000010");
const PHASE20_SKY_GUID: Uuid = uuid::uuid!("20040000-0000-4000-8000-000000000011");
const PHASE20_SUN_GUID: Uuid = uuid::uuid!("20040000-0000-4000-8000-000000000012");
const PHASE20_CAMERA_GUID: Uuid = uuid::uuid!("20040000-0000-4000-8000-000000000013");
/// The ocean entity.
pub const PHASE20_OCEAN_GUID: Uuid = uuid::uuid!("20040000-0000-4000-8000-000000000020");
/// The head lake entity.
pub const PHASE20_LAKE_GUID: Uuid = uuid::uuid!("20040000-0000-4000-8000-000000000021");
/// The river entity (its centreline is the `Spline` on this same entity).
pub const PHASE20_RIVER_GUID: Uuid = uuid::uuid!("20040000-0000-4000-8000-000000000022");
/// The swimming character.
pub const PHASE20_SWIMMER_GUID: Uuid = uuid::uuid!("20040000-0000-4000-8000-000000000030");
const PHASE20_CRATE_BASE: u128 = 0x2004_0000_0000_4000_8000_0000_0000_0100;

/// The `n`-th buoyant crate's GUID. `0..PHASE20_SEA_CRATES` float on the sea, the
/// rest on the head lake.
pub fn phase20_crate_guid(i: usize) -> Uuid {
    Uuid::from_u128(PHASE20_CRATE_BASE + i as u128)
}

/// Every buoyant crate, sea then lake.
pub fn phase20_crate_count() -> usize {
    PHASE20_SEA_CRATES + PHASE20_LAKE_CRATES
}

/// The **channel centre** in world Z at world `x`, metres — a gentle cubic S with
/// about ±9 m of meander, zero at both ends and at mid-span.
///
/// Polynomial, deliberately: this is committed content, and `std` trigonometry is
/// not bit-portable (the P14 LAW). Everything here is IEEE add/mul, so the same
/// terrain is authored on every machine and the `.inf_lvl` round-trips.
pub fn phase20_channel_z(x: f64) -> f64 {
    let t = (x / PHASE20_WORLD_M).clamp(0.0, 1.0);
    256.0 + 96.0 * t * (1.0 - t) * (2.0 * t - 1.0)
}

/// The coastal terrain: a 42 m headland at `x = 0` falling to −10 m at
/// `x = 512`, a valley following [`phase20_channel_z`], and a basin dug for the
/// head lake.
///
/// Sea level is `0`, so the shoreline lands near `x ≈ 414` — which is what puts
/// the ocean's shore blending, the wetness band and the river's mouth in one
/// scene.
pub fn phase20_height(x: f64, z: f64) -> f64 {
    let t = (x / PHASE20_WORLD_M).clamp(0.0, 1.0);
    let ramp = 42.0 - 52.0 * t;
    // Valley walls: quadratic, flat at the channel floor, +10 m at 60 m out.
    let d = ((z - phase20_channel_z(x)).abs() / 60.0).min(1.0);
    let mut h = ramp + 10.0 * d * d;
    // The head lake's basin.
    let (bx, bz) = PHASE20_LAKE_CENTER;
    let r2 = ((x - bx) * (x - bx) + (z - bz) * (z - bz))
        / (PHASE20_BASIN_RADIUS_M * PHASE20_BASIN_RADIUS_M);
    if r2 < 1.0 {
        h -= PHASE20_BASIN_DEPTH_M * (1.0 - r2);
    }
    h
}

/// The river's authored control points, world space.
///
/// Five knots down the channel, each 0.6 m above the valley floor so the water
/// sits *in* the valley rather than on it, ending at `y ≈ 1 m` where the channel
/// meets the sea. Monotonically descending, so the P20.1 surface advisory is
/// silent; the depth taper (1.2 → 2.0 m) lowers the bed monotonically too, so the
/// P20.4 bed advisory is silent as well — which is what makes the gate's
/// advisory-free arm a statement about the content rather than about the check.
pub fn phase20_river_points() -> Vec<DVec3> {
    [90.0f64, 170.0, 250.0, 330.0, 410.0]
        .iter()
        .map(|&x| {
            let t = x / PHASE20_WORLD_M;
            DVec3::new(x, 42.0 - 52.0 * t + 0.6, phase20_channel_z(x))
        })
        .collect()
}

/// The swimmer's blueprint class: one Tick that asks for a brisk swim forward
/// **and a full second of accumulated free fall**.
///
/// The downward component is the point. A character controller integrates gravity
/// into its own velocity and cannot tell a deliberate dive from an accumulated
/// fall, so a swim mode that honoured it symmetrically would sink forever —
/// `water::swim_motion`'s quarter-strength sink authority is what makes the
/// balance win. Authoring the fall here puts that behaviour in the *committed
/// content* rather than only in a unit test.
pub fn phase20_swimmer_class() -> BlueprintClass {
    use inf_blueprint::{BlueprintFn, EventBinding, EventKind, Expr, Lit, Param, Stmt, Ty};
    let entity = Expr::Call {
        path: vec!["vars".into(), "get".into()],
        args: vec![Expr::Lit(Lit::Str("entity".into()))],
    };
    let mut class = BlueprintClass::new("act:phase20-swimmer", "Swimmer");
    class.events = vec![EventBinding {
        event: EventKind::Tick,
        body: BlueprintFn {
            id: "tick".into(),
            name: "tick".into(),
            params: vec![Param {
                name: "dt".into(),
                ty: Ty::Float,
            }],
            ret: Ty::Unit,
            body: vec![Stmt::ExprStmt(Expr::Call {
                path: vec!["physics3d".into(), "move_and_slide".into()],
                args: vec![
                    entity,
                    Expr::Lit(Lit::Float(4.0 / 60.0)),
                    Expr::Lit(Lit::Float(-9.81 / 60.0)),
                    Expr::Lit(Lit::Float(0.0)),
                ],
            })],
        },
    }];
    class
}

/// Build the committed Phase 20 coastal [`SceneDoc`].
pub fn phase20_coastal_scene() -> SceneDoc {
    use crate::scene::serialize::LevelSettings;
    use inf_ecs::components::{
        ActorClass, BodyKind3D, Buoyancy, Camera, CharacterController3D, Collider3D,
        ColliderShape3DKind, Light, LightKind, RigidBody3D, SkyAtmosphere, Spline, SplineInterp,
        Terrain, TimeOfDay, WaterBody, WaterKind,
    };
    use inf_ecs::math::{Vec2d, Vec3d};

    let mut doc = SceneDoc::new();
    doc.set_title("Phase 20 Coastal");
    // 3D gravity is what makes a crate fall into the sea at all — and it comes
    // from **`gravity_2d.y`**, not from `gravity_3d` (`RuntimeSim::new` and its
    // `SimSession` mirror). The claim this comment used to make was the other
    // way round and was wrong; the P29.6 course found the field, and
    // `inf_packager::cook`'s `ignored_gravity_3d` advisory now says so out loud
    // when the two disagree. Both are set here, consistently, which is what
    // keeps this level quiet in the report.
    doc.set_settings(LevelSettings {
        gravity_2d: Vec2d::new(0.0, -9.81),
        gravity_3d: Vec3d::new(0.0, -9.81, 0.0),
        sim_hz: 60.0,
        ..LevelSettings::default()
    });

    // ── The coast. ──
    doc.create_with_guid(PHASE20_TERRAIN_GUID, SpawnKind::Empty, "Coast", None);
    insert!(
        doc,
        PHASE20_TERRAIN_GUID,
        Transform::from_translation(DVec3::ZERO)
    );
    {
        let mut terrain = Terrain::configured(PHASE20_TERRAIN_RESOLUTION, PHASE20_TERRAIN_MPS);
        for tz in 0..PHASE20_TERRAIN_TILES as i32 {
            for tx in 0..PHASE20_TERRAIN_TILES as i32 {
                terrain.data.author_tile((tx, tz), phase20_height);
            }
        }
        insert!(doc, PHASE20_TERRAIN_GUID, terrain);
    }

    // ── The sky: a real clock, so the sea is not a still photograph. ──
    doc.create_with_guid(PHASE20_SKY_GUID, SpawnKind::Empty, "Sky", None);
    insert!(
        doc,
        PHASE20_SKY_GUID,
        TimeOfDay {
            seconds: 11.0 * 3600.0,
            rate: 120.0,
            ..TimeOfDay::default()
        }
    );
    insert!(
        doc,
        PHASE20_SKY_GUID,
        SkyAtmosphere {
            weather_enabled: true,
            ..SkyAtmosphere::default()
        }
    );

    // ── The three bodies. ──
    doc.create_with_guid(PHASE20_OCEAN_GUID, SpawnKind::Empty, "Ocean", None);
    insert!(
        doc,
        PHASE20_OCEAN_GUID,
        WaterBody {
            kind: WaterKind::Ocean,
            level_m: PHASE20_SEA_LEVEL_M,
            wave_amplitude_m: 0.55,
            wave_length_m: 26.0,
            wave_steepness: 0.45,
            wave_count: 5,
            wave_seed: 0x0C0A_57A1,
            // Body-local wind: the sea state must not depend on where the weather
            // blend happens to be when a trace is taken.
            wind_from_weather: false,
            wind_x: 7.0,
            wind_z: -2.0,
            ..WaterBody::default()
        }
    );

    doc.create_with_guid(PHASE20_LAKE_GUID, SpawnKind::Empty, "Head Lake", None);
    insert!(
        doc,
        PHASE20_LAKE_GUID,
        Transform::from_translation(DVec3::new(
            PHASE20_LAKE_CENTER.0,
            PHASE20_LAKE_LEVEL_M,
            PHASE20_LAKE_CENTER.1
        ))
    );
    insert!(
        doc,
        PHASE20_LAKE_GUID,
        WaterBody::lake(PHASE20_LAKE_LEVEL_M, Vec2d::splat(PHASE20_LAKE_HALF_M))
    );

    // The river. Its centreline is the `Spline` on THIS SAME ENTITY (P20.1
    // composition — nothing to resolve, nothing to dangle), authored in world
    // space under an identity transform so the sample's numbers read as the map
    // coordinates they are.
    doc.create_with_guid(PHASE20_RIVER_GUID, SpawnKind::Empty, "River", None);
    insert!(
        doc,
        PHASE20_RIVER_GUID,
        Transform::from_translation(DVec3::ZERO)
    );
    insert!(
        doc,
        PHASE20_RIVER_GUID,
        WaterBody {
            river_width_start_m: 6.0,
            river_width_end_m: 14.0,
            river_depth_start_m: 1.2,
            river_depth_end_m: 2.0,
            ..WaterBody::river(6.0, 1.2, 2.2)
        }
    );
    insert!(
        doc,
        PHASE20_RIVER_GUID,
        Spline {
            points: phase20_river_points()
                .into_iter()
                .map(|p| Vec3d::new(p.x, p.y, p.z))
                .collect(),
            closed: false,
            interp: SplineInterp::CatmullRom,
        }
    );

    // ── Buoyant cargo: six crates on the sea, two on the head lake. ──
    //
    // 350…700 kg/m³ — all comfortably buoyant, at visibly different draughts, and
    // deliberately short of 1000: a near-neutral crate has almost no restoring
    // force and takes half a minute to come back up, which would make the gate a
    // test of patience rather than of buoyancy.
    for i in 0..phase20_crate_count() {
        let guid = phase20_crate_guid(i);
        doc.create_with_guid(guid, SpawnKind::Empty, &format!("Crate{i}"), None);
        let density = 350.0 + (i % PHASE20_SEA_CRATES) as f64 * 50.0;
        let pos = if i < PHASE20_SEA_CRATES {
            // Off the shore, in the open sea.
            Vec3d::new(
                460.0 + (i as f64 - 2.5) * 3.0,
                PHASE20_SEA_LEVEL_M + 2.5 + i as f64 * 0.4,
                240.0 + (i % 3) as f64 * 4.0,
            )
        } else {
            let k = i - PHASE20_SEA_CRATES;
            Vec3d::new(
                PHASE20_LAKE_CENTER.0 + (k as f64 - 0.5) * 6.0,
                PHASE20_LAKE_LEVEL_M + 2.0 + k as f64 * 0.5,
                PHASE20_LAKE_CENTER.1 + 4.0,
            )
        };
        insert!(
            doc,
            guid,
            RigidBody3D {
                kind: BodyKind3D::Dynamic,
                ..RigidBody3D::default()
            }
        );
        insert!(
            doc,
            guid,
            Collider3D {
                shape_kind: ColliderShape3DKind::Box,
                half_extents: Vec3d::splat(0.5),
                density,
                ..Collider3D::default()
            }
        );
        insert!(doc, guid, Buoyancy::of_density(density));
        insert!(
            doc,
            guid,
            Transform {
                translation: pos,
                ..Transform::IDENTITY
            }
        );
    }

    // ── The swimmer: a kinematic capsule, mostly under the surface, driven by
    //     the committed `Swimmer.inf_act`. ──
    doc.create_with_guid(PHASE20_SWIMMER_GUID, SpawnKind::Empty, "Swimmer", None);
    insert!(
        doc,
        PHASE20_SWIMMER_GUID,
        RigidBody3D {
            kind: BodyKind3D::Kinematic,
            ..RigidBody3D::default()
        }
    );
    insert!(
        doc,
        PHASE20_SWIMMER_GUID,
        Collider3D {
            shape_kind: ColliderShape3DKind::Capsule,
            half_extents: Vec3d::new(0.3, 0.6, 0.3),
            radius: 0.3,
            ..Collider3D::default()
        }
    );
    insert!(doc, PHASE20_SWIMMER_GUID, CharacterController3D::default());
    insert!(
        doc,
        PHASE20_SWIMMER_GUID,
        ActorClass(PHASE20_SWIMMER_ACTOR_GUID)
    );
    insert!(
        doc,
        PHASE20_SWIMMER_GUID,
        Transform {
            translation: Vec3d::new(455.0, PHASE20_SEA_LEVEL_M - 2.0, 262.0),
            ..Transform::IDENTITY
        }
    );

    // ── A sun and a camera looking down the coast. ──
    doc.create_with_guid(PHASE20_SUN_GUID, SpawnKind::Empty, "Sun", None);
    insert!(
        doc,
        PHASE20_SUN_GUID,
        Transform {
            translation: Vec3d::new(256.0, 120.0, 256.0),
            rotation: Vec3d::new(-48.0, -35.0, 0.0),
            scale: Vec3d::new(1.0, 1.0, 1.0),
        }
    );
    insert!(
        doc,
        PHASE20_SUN_GUID,
        Light {
            kind: LightKind::Directional,
            intensity: 3.2,
            ..Light::default()
        }
    );

    doc.create_with_guid(PHASE20_CAMERA_GUID, SpawnKind::Empty, "Camera", None);
    insert!(
        doc,
        PHASE20_CAMERA_GUID,
        Transform {
            translation: Vec3d::new(300.0, 60.0, 150.0),
            rotation: Vec3d::new(-18.0, 55.0, 0.0),
            scale: Vec3d::new(1.0, 1.0, 1.0),
        }
    );
    insert!(doc, PHASE20_CAMERA_GUID, Camera::default());

    doc.world_mut().propagate();
    doc.mark_saved();
    doc
}

/// Where the committed Phase 20 sample lives.
pub fn phase20_coastal_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../samples/phase20-coastal")
}

/// Write the committed Phase 20 gate files: the `.inf_lvl` (+ sidecar), the
/// swimmer's `.inf_act` (+ its `inf_asset` sidecar, so the `ActorClass` binding
/// resolves through the AssetDb and the cooked pack), and the README.
pub fn write_phase20_coastal() -> Result<(), String> {
    let dir = phase20_coastal_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("mkdir: {e}"))?;

    crate::scene::serialize::save(
        &phase20_coastal_scene(),
        &dir.join("Phase20Coastal.inf_lvl"),
        Some(PHASE20_LEVEL_GUID),
    )?;

    let act_bytes = encode_actor(&phase20_swimmer_class())?;
    write_phase19_asset(
        &dir.join("Swimmer.inf_act"),
        &act_bytes,
        PHASE20_SWIMMER_ACTOR_GUID,
        inf_asset::AssetKind::Blueprint,
    )?;

    std::fs::write(dir.join("README.md"), PHASE20_COASTAL_README)
        .map_err(|e| format!("write readme: {e}"))?;
    Ok(())
}

const PHASE20_COASTAL_README: &str = "# Phase 20 Coastal (the phase gate scene)\n\n\
Generated by `inf_editor_core::samples::phase20_coastal_scene` -- the **composed**\n\
gate scene for Phase 20 (water & hydrology), and the plan's own done-when sentence\n\
built: *an ocean plus a spline river fed by a lake, carrying buoyant physics\n\
objects.*\n\n\
- `Phase20Coastal.inf_lvl` -- a 512 m coastal heightfield (a 42 m headland falling\n\
  to -10 m, with a meandering valley and a dug basin), an **ocean** at sea level, a\n\
  **head lake** in the basin, a **spline river** running the valley from the lake to\n\
  the shore with a 6 -> 14 m width taper and a 1.2 -> 2.0 m depth taper, **eight\n\
  buoyant crates** (six at sea, two on the lake) and a **swimmer**.\n\
- `Swimmer.inf_act` -- one Tick calling `physics3d.move_and_slide` with a brisk\n\
  forward request and a full second of accumulated free fall, so the committed\n\
  content exercises the swim mode's asymmetric sink authority rather than only a\n\
  unit test doing so.\n\n\
## Why the terrain is authored inline\n\n\
Phase 16 gates heightfield tile streaming end to end. What Phase 20 needs from the\n\
ground is a *shoreline* -- somewhere the sea meets land, so shore blending, the\n\
wetness band and the river's mouth all appear in one scene -- and 512 m of inline\n\
heightfield is the cheapest way to have one.\n\n\
## Why the height function is a polynomial\n\n\
`std` trigonometry is not bit-portable (the P14 LAW), and this is COMMITTED\n\
content: the terrain has to author identically on every machine or the `.inf_lvl`\n\
would not round-trip byte-for-byte. The meander is a cubic, the valley walls are a\n\
quadratic and the basin is a paraboloid -- IEEE add/mul throughout.\n\n\
## Why the river's numbers are monotone\n\n\
Both cook advisories -- the P20.1 surface climb and the P20.4 authored-bed climb --\n\
must be SILENT on this scene, and the gate asserts it. An advisory that fires on\n\
the engine's own flagship sample is one nobody reads. The surface descends from\n\
~33.4 m to ~1.0 m and the depth taper widens downstream, so the bed descends too.\n\n\
## The gate (`runtime/inf-player/tests/phase20_gate.rs`)\n\n\
1. determinism -- two fresh loads of one pack agree on the whole physics trace,\n\
   bit for bit;\n\
2. **PIE == shipping** -- the cooked-pack world and the PIE-payload world float the\n\
   same crates and swim the same swimmer;\n\
3. the water is really doing something (crates settle at their draughts, the\n\
   swimmer surfaces, the river's mouth is finite);\n\
4. the cook is silent -- no water advisory on correct content;\n\
5. budget -- the composed scene builds inside the load budget and steps inside the\n\
   frame budget.\n\n\
Regenerate with `INF_BLESS_SAMPLES=1 cargo test -p inf-editor-core samples`.\n";

// ── Phase 21 gate scene: `samples/phase21-cavern` (P21.4) ────────────────────
//
// The plan's own done-when sentence, built: *on a streamed terrain, a carved cave
// system and an excavated foundation pit with displaced spoil piles, an
// underground room in the pit, saved and reloaded byte-identical, and it works in
// PIE.*
//
// Everything below is a **pure function of these constants**, evaluated with the
// same Ring-0 primitives the editor's carve and dig tools commit through
// (`VoxelOp` / `apply_surface_cut` / `SpoilPlan`). The generator is not a second
// implementation of excavation; it is the tools' arithmetic without the
// transaction, undo record and toolbar a document editor needs and a file writer
// does not.

/// The world is 128 m square: two 64 m tiles per axis at 2 m per sample.
pub const PHASE21_WORLD_M: f64 = 128.0;
pub const PHASE21_TILE_RES: u32 = 33;
pub const PHASE21_TILES: u32 = 2;
pub const PHASE21_MPS: f64 = 2.0;

/// One metre per voxel — the volume's grid and the world's therefore agree, which
/// is what lets every constant below be read as the metre it is.
pub const PHASE21_VOXEL_M: f64 = 1.0;

/// The rock body: chunk coords `[2, 3]` in X and Z, `[1, 2]` in Y — eight 16 m
/// chunks covering `x, z ∈ [32, 64)`, `y ∈ [16, 48)`, which is where every working
/// below lives. Stated as chunk indices rather than metres because that is the
/// unit the asset's size is actually charged in.
pub const PHASE21_ROCK_CHUNKS_XZ: (i32, i32) = (2, 3);
pub const PHASE21_ROCK_CHUNKS_Y: (i32, i32) = (1, 2);

const PHASE21_LEVEL_GUID: Uuid = uuid::uuid!("21040000-0000-4000-8000-000000000001");
pub const PHASE21_TERRAIN_ASSET_GUID: Uuid = uuid::uuid!("21040000-0000-4000-8000-000000000002");
pub const PHASE21_VOXEL_ASSET_GUID: Uuid = uuid::uuid!("21040000-0000-4000-8000-000000000003");
pub const PHASE21_BORER_ACTOR_GUID: Uuid = uuid::uuid!("21040000-0000-4000-8000-000000000004");

pub const PHASE21_TERRAIN_GUID: Uuid = uuid::uuid!("21040000-0000-4000-8000-000000000010");
/// The entity that carries the `VoxelVolume` **and** the borer Blueprint.
///
/// One entity, deliberately: `vars::get("entity")` is the only entity reference
/// the blueprint IR has, so a `voxel.*` node can only name the volume on its own
/// actor. Naming another entity's volume needs an entity-reference value type the
/// IR does not have — the same limit the audio and physics kits live under, and
/// the reason this is a cavern that bores itself rather than a digger standing
/// next to one.
pub const PHASE21_CAVERN_GUID: Uuid = uuid::uuid!("21040000-0000-4000-8000-000000000011");
pub const PHASE21_SKY_GUID: Uuid = uuid::uuid!("21040000-0000-4000-8000-000000000012");
pub const PHASE21_SUN_GUID: Uuid = uuid::uuid!("21040000-0000-4000-8000-000000000013");
pub const PHASE21_CAMERA_GUID: Uuid = uuid::uuid!("21040000-0000-4000-8000-000000000014");
/// A lamp standing on the underground room's floor — the thing that makes the
/// room a *room* rather than a void, and the gate's anti-vacuity handle on "the
/// room is where the generator says it is".
pub const PHASE21_LAMP_GUID: Uuid = uuid::uuid!("21040000-0000-4000-8000-000000000015");
/// A boulder resting on the rock **over the borer's drift**.
///
/// The only dynamic body in the level, and it is content rather than decoration:
/// this engine gives terrain **no rapier collider at all** (gameplay stands on it
/// through `terrain.height_at`), so the only thing holding this boulder up is the
/// voxel chunk trimesh the P21.4 bridge builds. When the borer removes the rock
/// beneath it, it falls into the trench — which is the one witness that survives
/// the whole way out to the real `--pie` pipe, because a falling body is in the
/// sim snapshot and a carved chunk is not.
pub const PHASE21_BOULDER_GUID: Uuid = uuid::uuid!("21040000-0000-4000-8000-000000000016");

/// Where the boulder starts, and how big it is. Directly over the drift at
/// `x = 50`, which the borer reaches around tick 67 of 120 — long after the drop
/// onto the rock has settled, so the two falls are distinct events in the trace.
pub const PHASE21_BOULDER_XZ: (f64, f64) = (50.0, 60.0);
pub const PHASE21_BOULDER_HALF_M: f64 = 0.5;
/// Its start height: clear of the ground so the first tick is a free fall, and
/// clear of the **trench crown** so the first landing is on solid rock rather
/// than inside the void the borer is about to open. Low enough that the 2.1 m
/// drop settles (about 40 ticks) well before the borer reaches `x = 50` (tick
/// 67), so "resting on rock" is a real, observable phase of the trace and not a
/// moment caught mid-fall.
pub const PHASE21_BOULDER_START_Y: f64 = 33.5;

// ── the ground ───────────────────────────────────────────────────────────────

/// The heightfield: a shallow ridge falling toward `+X`, plus a knoll well clear
/// of the workings so "the terrain is not flat" is true of content the dig never
/// touches.
///
/// Polynomial, deliberately: this is committed content, and `std` trigonometry is
/// not bit-portable (the P14 LAW).
pub fn phase21_height(x: f64, z: f64) -> f64 {
    let t = (x / PHASE21_WORLD_M).clamp(0.0, 1.0);
    let base = 34.0 - 8.0 * t;
    let dx = (x - 104.0) / 20.0;
    let dz = (z - 32.0) / 20.0;
    let r2 = dx * dx + dz * dz;
    if r2 < 1.0 {
        base + 6.0 * (1.0 - r2)
    } else {
        base
    }
}

// ── the workings ─────────────────────────────────────────────────────────────

/// The cave system: two swept legs from a mouth on the west slope, ending in a
/// chamber. The first leg's top clears the ground it starts under, which is what
/// opens the **mouth** — a cave with no mouth is a bubble.
pub fn phase21_cave_ops() -> Vec<inf_voxel::VoxelOp> {
    use inf_voxel::{VoxelOp, VoxelShape};
    vec![
        VoxelOp::carve(VoxelShape::Capsule {
            a: DVec3::new(34.0, 32.5, 44.0),
            b: DVec3::new(40.0, 28.0, 44.0),
            radius_m: 2.5,
        }),
        VoxelOp::carve(VoxelShape::Capsule {
            a: DVec3::new(40.0, 28.0, 44.0),
            b: DVec3::new(48.0, 25.0, 50.0),
            radius_m: 2.5,
        }),
        VoxelOp::carve(VoxelShape::Sphere {
            center: DVec3::new(52.0, 24.0, 52.0),
            radius_m: 4.5,
        }),
    ]
}

/// The **foundation pit**, resolved by the P21.3 sky rule: the top clears the
/// *highest* ground the footprint spans and the floor is [`PHASE21_PIT_DEPTH_M`]
/// below the *lowest*, so a pit dragged across a slope has no lid of surviving
/// hillside and "6 m deep" means below grade everywhere.
pub const PHASE21_PIT_CENTER_XZ: (f64, f64) = (50.0, 38.0);
pub const PHASE21_PIT_HALF_XZ: (f64, f64) = (4.0, 3.0);
pub const PHASE21_PIT_DEPTH_M: f64 = 6.0;
/// How far the cut's top clears the highest ground it spans.
pub const PHASE21_PIT_SKY_CLEARANCE_M: f64 = 2.0;

/// `(lowest, highest)` ground over the pit's footprint, sampled at one-metre
/// pitch — the same probe the tool's sky rule takes.
fn phase21_pit_ground() -> (f64, f64) {
    let (cx, cz) = PHASE21_PIT_CENTER_XZ;
    let (hx, hz) = PHASE21_PIT_HALF_XZ;
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    let steps_x = (2.0 * hx) as i32;
    let steps_z = (2.0 * hz) as i32;
    for i in 0..=steps_x {
        for j in 0..=steps_z {
            let h = phase21_height(cx - hx + i as f64, cz - hz + j as f64);
            lo = lo.min(h);
            hi = hi.max(h);
        }
    }
    (lo, hi)
}

/// The pit floor's world `y`.
pub fn phase21_pit_floor_y() -> f64 {
    phase21_pit_ground().0 - PHASE21_PIT_DEPTH_M
}

/// The pit as one axis-aligned box cut.
pub fn phase21_pit_op() -> inf_voxel::VoxelOp {
    use inf_voxel::{VoxelOp, VoxelShape};
    let (lo, hi) = phase21_pit_ground();
    let floor = lo - PHASE21_PIT_DEPTH_M;
    let top = hi + PHASE21_PIT_SKY_CLEARANCE_M;
    let (cx, cz) = PHASE21_PIT_CENTER_XZ;
    let (hx, hz) = PHASE21_PIT_HALF_XZ;
    VoxelOp::carve(VoxelShape::Box {
        center: DVec3::new(cx, 0.5 * (floor + top), cz),
        half_extents: DVec3::new(hx, 0.5 * (top - floor), hz),
    })
}

/// The underground room's floor and ceiling, world `y`.
pub const PHASE21_ROOM_FLOOR_Y: f64 = 20.0;
pub const PHASE21_ROOM_CEILING_Y: f64 = 23.0;

/// The **underground room in the pit**: a chamber directly beneath the pit's
/// footprint, plus the shaft that joins the two. The shaft's top reaches into the
/// pit and its bottom into the room, so the column over the room's centre is one
/// continuous void from the room floor to open sky — which is what "reachable"
/// means for the combined ground query.
pub fn phase21_room_ops() -> Vec<inf_voxel::VoxelOp> {
    use inf_voxel::{VoxelOp, VoxelShape};
    let (cx, cz) = PHASE21_PIT_CENTER_XZ;
    let floor = phase21_pit_floor_y();
    vec![
        VoxelOp::carve(VoxelShape::Box {
            center: DVec3::new(
                cx,
                0.5 * (PHASE21_ROOM_FLOOR_Y + PHASE21_ROOM_CEILING_Y),
                cz,
            ),
            half_extents: DVec3::new(
                PHASE21_PIT_HALF_XZ.0,
                0.5 * (PHASE21_ROOM_CEILING_Y - PHASE21_ROOM_FLOOR_Y),
                PHASE21_PIT_HALF_XZ.1,
            ),
        }),
        VoxelOp::carve(VoxelShape::Capsule {
            a: DVec3::new(cx, floor + 0.5, cz),
            b: DVec3::new(cx, PHASE21_ROOM_CEILING_Y - 1.0, cz),
            radius_m: 1.5,
        }),
    ]
}

/// Where the pit's spoil goes. Chosen explicitly rather than through
/// [`inf_voxel::default_spoil_site`] so the heap lands inside the volume's own
/// chunks: the default site stands clear of the cut's `+X` face, which for this
/// pit is past the rock body's east edge, and a pile that materialized chunks
/// outside the authored body would make the committed asset a function of the
/// pile's arithmetic rather than of the design.
pub const PHASE21_SPOIL_SITE_XZ: (f64, f64) = (42.0, 56.0);

// ── the built content ────────────────────────────────────────────────────────

/// `(the carved volume, the terrain with its hole mask, the pit's spoil report)`
/// — every working applied, in one deterministic order.
///
/// The order is the order an author would dig in and the order the numbers depend
/// on: the cave first (its mouth is on virgin ground), then the pit (whose spoil
/// counts are what the heap is made of), then the room under it, then the heap.
/// A carve is idempotent and the ops are disjoint, so the *geometry* does not
/// depend on the order — the **spoil counts** do, which is why the pit's report is
/// taken alone rather than from a running total.
pub struct Phase21Workings {
    /// The rock, with every working cut into it and the heap placed on it.
    pub volume: inf_voxel::VoxelData,
    /// The heightfield, with the hole mask every cut opened.
    pub terrain: inf_terrain::TerrainData,
    /// What the **pit** removed, per material — one side of the conservation
    /// identity, produced by the cut itself.
    pub pit_removed: [u64; inf_voxel::MATERIAL_COUNT],
    /// What the heap placed — the other side.
    pub spoil: inf_voxel::SpoilReport,
}

pub fn phase21_build() -> Phase21Workings {
    use inf_voxel::{ChunkKey, VoxelChunk, VoxelData};

    // The heightfield, authored tile by tile.
    let mut terrain = inf_terrain::TerrainData::new(PHASE21_TILE_RES, PHASE21_MPS);
    for tz in 0..PHASE21_TILES {
        for tx in 0..PHASE21_TILES {
            terrain.author_tile((tx as i32, tz as i32), phase21_height);
        }
    }

    // The rock: solid below the heightfield, air above, meeting the terrain on
    // its own plane. Signed distance in voxels; the chunk clamps it to the band.
    let mut rock = VoxelData::new(PHASE21_VOXEL_M);
    for cy in PHASE21_ROCK_CHUNKS_Y.0..=PHASE21_ROCK_CHUNKS_Y.1 {
        for cz in PHASE21_ROCK_CHUNKS_XZ.0..=PHASE21_ROCK_CHUNKS_XZ.1 {
            for cx in PHASE21_ROCK_CHUNKS_XZ.0..=PHASE21_ROCK_CHUNKS_XZ.1 {
                let key = ChunkKey::new(cx, cy, cz);
                let base = key.base_sample();
                rock.insert_chunk(
                    key,
                    VoxelChunk::from_fn(|i, j, k| {
                        let x = (base[0] + i as i32) as f64;
                        let y = (base[1] + j as i32) as f64;
                        let z = (base[2] + k as i32) as f64;
                        y - phase21_height(x, z)
                    }),
                );
            }
        }
    }

    // Every cut, through the same two Ring-0 rules a tool commits: the voxel op,
    // then the exactly-invertible surface coupling that decides which height
    // samples it opens.
    fn cut(
        data: &mut inf_voxel::VoxelData,
        t: &mut inf_terrain::TerrainData,
        op: &inf_voxel::VoxelOp,
    ) -> inf_voxel::OpReport {
        let (report, _delta) = data.apply_op(op);
        inf_voxel::apply_surface_cut(t, DVec3::ZERO, &op.shape, true);
        report
    }

    for op in phase21_cave_ops() {
        cut(&mut rock, &mut terrain, &op);
    }
    let pit = phase21_pit_op();
    let pit_report = cut(&mut rock, &mut terrain, &pit);
    for op in phase21_room_ops() {
        cut(&mut rock, &mut terrain, &op);
    }

    // The spoil: exactly what the pit removed, per material, placed as a repose
    // heap on the ground beside it. `place_spoil_into` is the same call the
    // editor's excavation transaction makes; no store is needed because this
    // volume is fully resident by construction (there is no cold half to page).
    let (sx, sz) = PHASE21_SPOIL_SITE_XZ;
    let (pit_lo, pit_hi) = pit.shape.aabb_m(0.0);
    let plan = inf_voxel::SpoilPlan::new(
        pit_report.carved,
        DVec3::new(sx, phase21_height(sx, sz), sz),
    )
    .excluding(pit_lo, pit_hi);
    let mut builder = inf_voxel::VoxelDeltaBuilder::new();
    let spoil = rock.place_spoil_into(&plan, &mut builder);

    rock.clear_dirty();
    terrain.clear_dirty();
    Phase21Workings {
        volume: rock,
        terrain,
        pit_removed: pit_report.carved,
        spoil,
    }
}

// ── the runtime borer ────────────────────────────────────────────────────────

/// Where the borer starts cutting, how far it advances per tick, and for how
/// many ticks.
///
/// A **cut-and-cover drift along `+X` at `y = 29.7`**, north of every committed
/// working (the cave chamber reaches `z ≈ 56.5`; the pit and the room stop at
/// `z = 41`), running from `x = 40` to `x = 58` — inside the rock body's
/// `[32, 64)` span at both ends.
///
/// **It breaks the surface for its whole run, and that is the point.** The ground
/// over the drift falls from 31.5 m to 30.375 m; a 2.5 m ball centred at 29.7 m
/// has its crown at 32.2 m and its floor at 27.2 m, so every tick removes rock
/// that was solid when the level was saved *and* opens height samples that were
/// closed. A borer that stayed underground would exercise the carve and never the
/// **coupling** — and "a game digging through the surface opens a mouth" is the
/// phase's own sentence, so the sample has to actually do it.
///
/// The step is **sub-voxel on purpose**: at 0.15 m per tick a 2.5 m ball advances
/// less than one sample, so consecutive cuts overlap and the per-tick volume
/// varies between zero and a few cubic metres. A trace whose every entry was the
/// same number would compare equal for the wrong reason.
pub const PHASE21_BORE_START: (f64, f64, f64) = (40.0, 29.7, 60.0);
pub const PHASE21_BORE_STEP_M: f64 = 0.15;
pub const PHASE21_BORE_RADIUS_M: f64 = 2.5;
/// Ticks the gate drives the borer for — 18 m of drift. Stated here rather than
/// in the gate so the content and the test cannot disagree about how far it goes.
pub const PHASE21_BORE_STEPS: usize = 120;

/// The XZ the borer probes each tick with the **combined** ground query
/// (`terrain.height_at`) — the underground room's centre, which the committed
/// workings have already opened all the way to the sky.
pub fn phase21_room_probe_xz() -> (f64, f64) {
    PHASE21_PIT_CENTER_XZ
}

/// The borer Blueprint: on every Tick it advances one step along `+Z`, carves a
/// ball, accumulates the volume, and records both ground queries.
///
/// **Tick, not BeginPlay.** Both hosts seed their voxel map *after* constructing
/// the sim (`SimSession::enter` … `set_voxel_volumes`; `sim_from_built` …
/// `attach_voxel_volumes`), and `BeginPlay` runs inside that construction — so a
/// carve there would see an empty map and refuse. Stated on the node kit too; the
/// sample obeys it rather than working around it.
pub fn phase21_borer_class() -> BlueprintClass {
    use inf_blueprint::{
        BinOp, BlueprintFn, EventBinding, EventKind, Expr, Lit, Param, Stmt, Ty, Variable,
    };

    let get = |name: &str| Expr::Call {
        path: vec!["vars".into(), "get".into()],
        args: vec![Expr::Lit(Lit::Str(name.into()))],
    };
    let set = |name: &str, value: Expr| {
        Stmt::ExprStmt(Expr::Call {
            path: vec!["vars".into(), "set".into()],
            args: vec![Expr::Lit(Lit::Str(name.into())), value],
        })
    };
    // `Expr::Binary`, not a `math::*` call: `math.add` / `math.mul` lower as
    // BINARY OPS (`lower::role_of` → `NodeRole::BinaryOp`), and only the unary /
    // ternary `math.*` names ever reach `dispatch_math`. Writing `math::add(a, b)`
    // into the IR by hand produces "unknown math node", which fails the whole
    // handler — and a failed handler looks, from the trace, exactly like a borer
    // that reports zero for ever.
    let math = |op: BinOp, a: Expr, b: Expr| Expr::Binary(op, Box::new(a), Box::new(b));
    let fslot = |name: &str, default: f64| Variable {
        name: name.into(),
        ty: Ty::Float,
        default: Lit::Float(default),
        exposed: false,
    };

    let (bx, by, bz) = PHASE21_BORE_START;
    let (px, pz) = phase21_room_probe_xz();

    let mut class = BlueprintClass::new("act:phase21-borer", "Tunnel Borer");
    class.variables = vec![
        // How many ticks have run — the bore's own clock, so its position is a
        // function of the trace and not of a wall clock.
        fslot("step", 0.0),
        // This tick's cut, in cubic metres.
        fslot("removed", 0.0),
        // Every tick's cut, summed — the number a game would show on a HUD.
        fslot("total", 0.0),
        // The combined ground query over the underground room.
        fslot("room_ground", 0.0),
        // The voxel-only surface at the same XZ. Equal to `room_ground` wherever
        // the terrain is holed, which is the property the pair exists to show.
        fslot("room_voxel", 0.0),
    ];
    class.events = vec![EventBinding {
        event: EventKind::Tick,
        body: BlueprintFn {
            id: "tick".into(),
            name: "tick".into(),
            params: vec![Param {
                name: "dt".into(),
                ty: Ty::Float,
            }],
            ret: Ty::Unit,
            body: vec![
                set(
                    "removed",
                    Expr::Call {
                        path: vec!["voxel".into(), "carve_sphere".into()],
                        args: vec![
                            get("entity"),
                            // x = start.x + step * PHASE21_BORE_STEP_M
                            math(
                                BinOp::Add,
                                Expr::Lit(Lit::Float(bx)),
                                math(
                                    BinOp::Mul,
                                    get("step"),
                                    Expr::Lit(Lit::Float(PHASE21_BORE_STEP_M)),
                                ),
                            ),
                            Expr::Lit(Lit::Float(by)),
                            Expr::Lit(Lit::Float(bz)),
                            Expr::Lit(Lit::Float(PHASE21_BORE_RADIUS_M)),
                        ],
                    },
                ),
                set("total", math(BinOp::Add, get("total"), get("removed"))),
                set(
                    "step",
                    math(BinOp::Add, get("step"), Expr::Lit(Lit::Float(1.0))),
                ),
                set(
                    "room_ground",
                    Expr::Call {
                        path: vec!["terrain".into(), "height_at".into()],
                        args: vec![Expr::Lit(Lit::Float(px)), Expr::Lit(Lit::Float(pz))],
                    },
                ),
                set(
                    "room_voxel",
                    Expr::Call {
                        path: vec!["voxel".into(), "ground_height".into()],
                        args: vec![Expr::Lit(Lit::Float(px)), Expr::Lit(Lit::Float(pz))],
                    },
                ),
            ],
        },
    }];
    class
}

// ── the scene ────────────────────────────────────────────────────────────────

/// Build the committed Phase 21 [`SceneDoc`].
///
/// The terrain is **asset-backed** (`Terrain.asset`), not inline, and that is
/// forced rather than chosen: scene schema v19 pins `TerrainTileFrozenV3`, which
/// has no hole rows, so an inline terrain cannot persist a hole mask and saving
/// this level would seal every cave mouth in it (`inf_voxel::inline_hole_advisory`
/// is the detector; the carve tools refuse to create the state at all). The
/// working set is written into the document anyway and stripped on save by
/// `serialize::strip_streamed_terrain`, exactly as a wizard-imported terrain is —
/// which is why the PIE wire had to learn to carry `.inf_terrain` bytes this batch.
pub fn phase21_cavern_scene() -> SceneDoc {
    use crate::scene::serialize::LevelSettings;
    use inf_ecs::components::{
        ActorClass, BodyKind3D, Camera, Collider3D, ColliderShape3DKind, Light, LightKind,
        RigidBody3D, SkyAtmosphere, Terrain, TimeOfDay, VoxelVolume,
    };
    use inf_ecs::math::{Vec2d, Vec3d};

    let terrain_data = phase21_build().terrain;

    let mut doc = SceneDoc::new();
    doc.set_title("Phase 21 Cavern");
    doc.set_settings(LevelSettings {
        gravity_2d: Vec2d::new(0.0, -9.81),
        gravity_3d: Vec3d::new(0.0, -9.81, 0.0),
        sim_hz: 60.0,
        ..LevelSettings::default()
    });

    // ── The ground, carved. ──
    doc.create_with_guid(PHASE21_TERRAIN_GUID, SpawnKind::Empty, "Ridge", None);
    insert!(
        doc,
        PHASE21_TERRAIN_GUID,
        Transform::from_translation(DVec3::ZERO)
    );
    insert!(
        doc,
        PHASE21_TERRAIN_GUID,
        Terrain {
            meters_per_sample: PHASE21_MPS,
            tile_resolution: PHASE21_TILE_RES,
            data: terrain_data,
            asset: Some(PHASE21_TERRAIN_ASSET_GUID),
            ..Terrain::default()
        }
    );

    // ── The rock, and the script that keeps boring into it. ──
    doc.create_with_guid(PHASE21_CAVERN_GUID, SpawnKind::Empty, "Cavern", None);
    insert!(
        doc,
        PHASE21_CAVERN_GUID,
        Transform::from_translation(DVec3::ZERO)
    );
    insert!(
        doc,
        PHASE21_CAVERN_GUID,
        VoxelVolume {
            asset: Some(PHASE21_VOXEL_ASSET_GUID),
            // Must equal the asset's own scale or the cook advises about it —
            // `voxel_scale_mismatches` exists because the two are separately
            // authored and a mismatch draws a cave at the wrong size.
            voxel_size_m: PHASE21_VOXEL_M,
            runtime_carve: true,
        }
    );
    insert!(
        doc,
        PHASE21_CAVERN_GUID,
        ActorClass(PHASE21_BORER_ACTOR_GUID)
    );

    // ── A lamp on the underground room's floor. ──
    //
    // Content, not decoration: it is the one entity whose authored `y` is the
    // room's floor, so a gate can compare "where the ground query says the floor
    // is" against "where the level says the floor is" without either number being
    // derived from the other.
    doc.create_with_guid(PHASE21_LAMP_GUID, SpawnKind::Empty, "Room Lamp", None);
    {
        let (px, pz) = phase21_room_probe_xz();
        insert!(
            doc,
            PHASE21_LAMP_GUID,
            Transform::from_translation(DVec3::new(px, PHASE21_ROOM_FLOOR_Y, pz))
        );
    }
    insert!(
        doc,
        PHASE21_LAMP_GUID,
        Light {
            kind: LightKind::Point,
            intensity: 12.0,
            range: 14.0,
            color: inf_ecs::math::Color::new(1.0, 0.86, 0.62, 1.0),
            ..Light::default()
        }
    );

    // ── The boulder over the drift. ──
    //
    // The level's only dynamic body. Terrain has no collider in this engine, so it
    // is resting on the **voxel rock** and nothing else; when the borer takes that
    // rock away it falls into the trench. That is what makes "the carve reached
    // the colliders" observable through the PIE pipe, which streams poses and
    // knows nothing about chunks.
    doc.create_with_guid(PHASE21_BOULDER_GUID, SpawnKind::Empty, "Boulder", None);
    {
        let (bx, bz) = PHASE21_BOULDER_XZ;
        insert!(
            doc,
            PHASE21_BOULDER_GUID,
            Transform::from_translation(DVec3::new(bx, PHASE21_BOULDER_START_Y, bz))
        );
    }
    insert!(
        doc,
        PHASE21_BOULDER_GUID,
        RigidBody3D {
            kind: BodyKind3D::Dynamic,
            ..RigidBody3D::default()
        }
    );
    insert!(
        doc,
        PHASE21_BOULDER_GUID,
        Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: Vec3d::splat(PHASE21_BOULDER_HALF_M),
            ..Collider3D::default()
        }
    );

    // ── Sky + sun + camera, so the scene opens somewhere sensible. ──
    doc.create_with_guid(PHASE21_SKY_GUID, SpawnKind::Empty, "Sky", None);
    insert!(
        doc,
        PHASE21_SKY_GUID,
        TimeOfDay {
            seconds: 10.0 * 3600.0,
            rate: 0.0,
            ..TimeOfDay::default()
        }
    );
    insert!(doc, PHASE21_SKY_GUID, SkyAtmosphere::default());

    doc.create_with_guid(PHASE21_SUN_GUID, SpawnKind::Empty, "Sun", None);
    insert!(
        doc,
        PHASE21_SUN_GUID,
        Light {
            kind: LightKind::Directional,
            intensity: 3.0,
            ..Light::default()
        }
    );
    insert!(
        doc,
        PHASE21_SUN_GUID,
        Transform::from_translation(DVec3::new(0.0, 80.0, 0.0))
    );

    doc.create_with_guid(PHASE21_CAMERA_GUID, SpawnKind::Empty, "Camera", None);
    insert!(doc, PHASE21_CAMERA_GUID, Camera::default());
    insert!(
        doc,
        PHASE21_CAMERA_GUID,
        Transform::from_translation(DVec3::new(20.0, 44.0, 44.0))
    );

    doc.world_mut().propagate();
    doc.mark_saved();
    doc
}

// ── the committed files ──────────────────────────────────────────────────────

/// `samples/phase21-cavern`, resolved from this crate's manifest dir.
pub fn phase21_cavern_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../samples/phase21-cavern")
}

/// The `.inf_terrain` the level's `Terrain.asset` names — the **only** place the
/// hole mask can live (schema v19's frozen tile record has no hole rows).
pub fn phase21_terrain_asset() -> Result<inf_terrain::TerrainAsset, String> {
    let data = phase21_build().terrain;
    let opts = inf_terrain::PyramidOptions::default();
    let pyramid = inf_terrain::build_pyramid(&data, opts);
    inf_terrain::build_terrain_asset(&data, &pyramid, opts)
        .map_err(|e| format!("build terrain asset: {e}"))
}

/// The `.inf_voxel` the level's `VoxelVolume.asset` names.
pub fn phase21_voxel_asset() -> Result<inf_voxel::VoxelAsset, String> {
    let volume = phase21_build().volume;
    inf_voxel::build_voxel_asset(&volume).map_err(|e| format!("build voxel asset: {e}"))
}

/// Write every committed Phase 21 gate file: the `.inf_lvl` (+ sidecar), the
/// `.inf_terrain` and `.inf_voxel` (+ their `inf_asset` sidecars, so the level's
/// refs resolve through the AssetDb *and* the cooked pack), the borer's
/// `.inf_act`, and the README.
pub fn write_phase21_cavern() -> Result<(), String> {
    let dir = phase21_cavern_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("mkdir: {e}"))?;

    crate::scene::serialize::save(
        &phase21_cavern_scene(),
        &dir.join("Phase21Cavern.inf_lvl"),
        Some(PHASE21_LEVEL_GUID),
    )?;

    let terrain = phase21_terrain_asset()?;
    let tpath = dir.join("Phase21Cavern.inf_terrain");
    let tbytes = inf_terrain::write_terrain_asset(&tpath, &terrain)
        .map_err(|e| format!("write terrain asset: {e}"))?;
    inf_asset::AssetSidecar::new(
        inf_asset::AssetId(PHASE21_TERRAIN_ASSET_GUID),
        inf_asset::AssetKind::Terrain,
        inf_asset::ContentHash::of(tbytes),
    )
    .save(&tpath)
    .map_err(|e| format!("write terrain sidecar: {e}"))?;

    let voxel = phase21_voxel_asset()?;
    write_phase19_asset(
        &dir.join("Cavern.inf_voxel"),
        voxel.as_bytes(),
        PHASE21_VOXEL_ASSET_GUID,
        inf_asset::AssetKind::VoxelVolume,
    )?;

    let act_bytes = encode_actor(&phase21_borer_class())?;
    write_phase19_asset(
        &dir.join("Borer.inf_act"),
        &act_bytes,
        PHASE21_BORER_ACTOR_GUID,
        inf_asset::AssetKind::Blueprint,
    )?;

    std::fs::write(dir.join("README.md"), PHASE21_CAVERN_README)
        .map_err(|e| format!("write readme: {e}"))?;
    Ok(())
}

const PHASE21_CAVERN_README: &str = "# Phase 21 Cavern (the phase gate scene)\n\n\
Generated by `inf_editor_core::samples::phase21_cavern_scene` -- the **composed**\n\
gate scene for Phase 21 (volumetric terrain), and the plan's own done-when\n\
sentence built: *on a streamed terrain, carve a cave system and excavate a\n\
foundation pit with displaced soil piles, build an underground room in the pit,\n\
save and reload byte-identical, and it works in PIE.*\n\n\
- `Phase21Cavern.inf_lvl` -- a 128 m ridge (2 x 2 tiles at 2 m/sample) with an\n\
  **asset-backed** terrain, one `VoxelVolume` entity carrying the rock body and\n\
  the borer Blueprint, a lamp standing on the underground room's floor, and a\n\
  sky/sun/camera.\n\
- `Phase21Cavern.inf_terrain` -- the heightfield **and its hole mask**. The mask\n\
  is why the terrain is asset-backed rather than inline: scene schema v19's\n\
  frozen tile record has no hole rows, so an inline terrain cannot persist one\n\
  and saving would seal every cave mouth.\n\
- `Cavern.inf_voxel` -- eight 16 m SDF chunks of rock, with the cave system, the\n\
  foundation pit, the underground room + its shaft, and the pit's **spoil heap**\n\
  cut into them. The heap holds exactly the pit's per-material voxel count, with\n\
  no bulking factor: a 1:1 identity is a gate, a 1.25x fudge is a number nobody\n\
  can test.\n\
- `Borer.inf_act` -- the runtime-carving Blueprint. Every Tick it advances 0.5 m\n\
  along +Z and carves a 2 m ball with `voxel.carve_sphere`, accumulating the\n\
  cubic metres removed and recording both ground queries. It runs on **Tick**\n\
  rather than BeginPlay because both hosts seed their voxel map after building\n\
  the sim, so a BeginPlay carve would find no volume.\n\n\
Exercised by `runtime/inf-player/tests/phase21_gate.rs`.\n\n\
Regenerate with `INF_BLESS_SAMPLES=1 cargo test -p inf-editor-core samples`.\n";

// ── Phase 22 gate scene: `samples/phase22-playground` (P22.4) ────────────────
//
// The phase's own done-when sentence, built: *a playground scene shows footprints
// and tyre tracks in snow and sand, bending grass, and a car and a multi-storey
// building destroyed by Blueprint-triggered explosions with debris physics —
// deterministic on the replay trace, PIE == shipping.*
//
// Everything below is a **pure function of these constants**, so the gate imports
// them rather than restating them and the fixture and the test cannot drift.

/// The world is 128 m square: two 64 m tiles per axis at 2 m per sample — the
/// P21 arrangement, because the streaming machinery it exercises is the same and
/// a second set of numbers would be a second thing to reason about.
pub const PHASE22_WORLD_M: f64 = 128.0;
pub const PHASE22_TILE_RES: u32 = 33;
pub const PHASE22_TILES: u32 = 2;
pub const PHASE22_MPS: f64 = 2.0;

const PHASE22_LEVEL_GUID: Uuid = uuid::uuid!("22040000-0000-4000-8000-000000000001");
pub const PHASE22_TERRAIN_ASSET_GUID: Uuid = uuid::uuid!("22040000-0000-4000-8000-000000000002");
/// The `.inf_mesh` the tower block and its control twin both reference. **One
/// mesh, two actors** — which is also what makes the control a control: the two
/// differ in `runtime_destruct` and in nothing else, not even in what they are
/// made of.
pub const PHASE22_BLOCK_MESH_GUID: Uuid = uuid::uuid!("22040000-0000-4000-8000-000000000003");
/// The car chassis `.inf_mesh`.
pub const PHASE22_CHASSIS_MESH_GUID: Uuid = uuid::uuid!("22040000-0000-4000-8000-000000000004");
/// The demolition Blueprint class, carried by the tower **and** by the control.
pub const PHASE22_DEMO_ACTOR_GUID: Uuid = uuid::uuid!("22040000-0000-4000-8000-000000000005");
/// The wreck Blueprint class, carried by the car chassis.
pub const PHASE22_WRECK_ACTOR_GUID: Uuid = uuid::uuid!("22040000-0000-4000-8000-000000000006");
/// The roller Blueprint class — the thing that leaves the tracks.
pub const PHASE22_ROLLER_ACTOR_GUID: Uuid = uuid::uuid!("22040000-0000-4000-8000-000000000007");
/// The collapse's `.inf_audio` clip.
pub const PHASE22_CLIP_GUID: Uuid = uuid::uuid!("22040000-0000-4000-8000-000000000008");

pub const PHASE22_TERRAIN_GUID: Uuid = uuid::uuid!("22040000-0000-4000-8000-000000000010");
/// The multi-storey block: the actor the phase's headline is about.
pub const PHASE22_TOWER_GUID: Uuid = uuid::uuid!("22040000-0000-4000-8000-000000000011");
/// Its twin, identical in every way except `runtime_destruct: false` — the
/// both-ways half of the flag gate, standing in the same level under the same
/// charge.
pub const PHASE22_CONTROL_GUID: Uuid = uuid::uuid!("22040000-0000-4000-8000-000000000012");
/// The car's chassis: a destructible dynamic body with four wheels on revolute
/// joints. **A prop, not a vehicle** — this engine has no vehicle controller, so
/// it rolls and settles and is then blown up, which is all the phase claims.
pub const PHASE22_CHASSIS_GUID: Uuid = uuid::uuid!("22040000-0000-4000-8000-000000000013");
pub const PHASE22_WHEEL_GUIDS: [Uuid; 4] = [
    uuid::uuid!("22040000-0000-4000-8000-000000000014"),
    uuid::uuid!("22040000-0000-4000-8000-000000000015"),
    uuid::uuid!("22040000-0000-4000-8000-000000000016"),
    uuid::uuid!("22040000-0000-4000-8000-000000000017"),
];
/// The roller: the moving body that crosses the snow and the sand.
pub const PHASE22_ROLLER_GUID: Uuid = uuid::uuid!("22040000-0000-4000-8000-000000000018");
/// The grass the deformation field bends (P22.1's scatter-bend half).
pub const PHASE22_GRASS_GUID: Uuid = uuid::uuid!("22040000-0000-4000-8000-000000000019");
pub const PHASE22_SKY_GUID: Uuid = uuid::uuid!("22040000-0000-4000-8000-00000000001a");
pub const PHASE22_SUN_GUID: Uuid = uuid::uuid!("22040000-0000-4000-8000-00000000001b");
pub const PHASE22_CAMERA_GUID: Uuid = uuid::uuid!("22040000-0000-4000-8000-00000000001c");

// ── the ground ───────────────────────────────────────────────────────────────

/// The heightfield: **flat where anything stands**, plus one knoll well clear of
/// every actor so "the terrain is not a plane" is true of content nothing
/// touches.
///
/// Flat on purpose, and it is the design rather than laziness. Every claim this
/// sample makes is about *what the ground is made of* (which splat layer answers
/// a contact) and about *what falls onto it*; a slope would add a variable to
/// both — a rut on a hill is a rut plus a slide, and a tower on a hill is a
/// tower on a ramp — for no extra coverage. The P21 sample is the one that
/// exercises terrain shape.
///
/// Polynomial, deliberately: this is committed content, and `std` trigonometry is
/// not bit-portable (the P14 LAW).
pub fn phase22_height(x: f64, z: f64) -> f64 {
    let dx = (x - 108.0) / 16.0;
    let dz = (z - 20.0) / 16.0;
    let r2 = dx * dx + dz * dz;
    if r2 < 1.0 {
        5.0 * (1.0 - r2)
    } else {
        0.0
    }
}

/// The engine's default palette indices this sample paints with. Their *physics*
/// is `inf_terrain::deform::LAYER_RESPONSE`, which maps **by index**: layer 3 is
/// the deep, non-recovering one (snow), layer 2 the loose one that slumps over
/// minutes (sand/dirt/mud), layer 0 the one that barely dents and bends instead
/// (grass). The sample paints the indices whose archetype it means, which is the
/// only way to mean anything until an authored response table exists.
pub const PHASE22_GRASS_LAYER: u8 = 0;
pub const PHASE22_SAND_LAYER: u8 = 2;
pub const PHASE22_SNOW_LAYER: u8 = 3;

/// The snow band, in world `z`.
pub const PHASE22_SNOW_Z: (f64, f64) = (40.0, 56.0);
/// The sand band, in world `z`, immediately downstream of the snow one so a
/// single straight run crosses both — and **ending four metres before the roller
/// does**, so the trail has a far end on ground that barely dents. A run that
/// stopped inside the band would leave the last stamp still being pressed, and
/// "the trail ends where the roller stopped" would be untestable.
pub const PHASE22_SAND_Z: (f64, f64) = (56.0, 64.0);

/// Which splat layer is painted at world `z`.
pub fn phase22_layer_at(z: f64) -> u8 {
    if z >= PHASE22_SNOW_Z.0 && z < PHASE22_SNOW_Z.1 {
        PHASE22_SNOW_LAYER
    } else if z >= PHASE22_SAND_Z.0 && z < PHASE22_SAND_Z.1 {
        PHASE22_SAND_LAYER
    } else {
        PHASE22_GRASS_LAYER
    }
}

/// The heightfield **and** its splat weights, authored tile by tile.
pub fn phase22_terrain_data() -> inf_terrain::TerrainData {
    let mut data = inf_terrain::TerrainData::new(PHASE22_TILE_RES, PHASE22_MPS);
    for tz in 0..PHASE22_TILES as i32 {
        for tx in 0..PHASE22_TILES as i32 {
            data.author_tile((tx, tz), phase22_height);
        }
    }
    // The paint. One layer at full weight per sample — a hard edge rather than a
    // blend, because the *response* is chosen by `dominant_layer_at` and a
    // 50/50 sample would make which archetype answers a rounding question.
    let res = PHASE22_TILE_RES;
    let mps = PHASE22_MPS;
    for tz in 0..PHASE22_TILES as i32 {
        for tx in 0..PHASE22_TILES as i32 {
            let o = data.tile_origin_xz((tx, tz));
            let tile = data.get_or_create_tile((tx, tz));
            tile.ensure_weights(res);
            for j in 0..res {
                let wz = o.y + j as f64 * mps;
                let mut w = [0u8; 4];
                w[phase22_layer_at(wz) as usize] = 255;
                for i in 0..res {
                    tile.set_weight_sample(res, i, j, w);
                }
            }
        }
    }
    data.clear_dirty();
    data
}

// ── the meshes ───────────────────────────────────────────────────────────────

/// Subdivisions per box face. **Not taste — a cook threshold.**
///
/// `VgeomCookOptions::min_triangles` is 2048, and a mesh under it draws in the
/// shipped build as a *placeholder cube* with a cook advisory saying so. Arm (e)
/// of the phase gate asserts the cook is silent on this sample, so every mesh in
/// it has to clear the bar: 6 faces × 2 × 16² = **3 072 triangles**, which is
/// also a perfectly reasonable tessellation for a building somebody is going to
/// shatter.
const PHASE22_BOX_SUBDIV: u32 = 16;

/// A tessellated axis-aligned box as an [`inf_mesh::MeshAsset`], spanning
/// `[min, max]` in its own local space.
///
/// Local space matters twice over. The chunk geometry the cook derives is in
/// *this* frame and the actor's placement maps it into the world, so a box whose
/// local `y` runs `0 ..= h` is a building whose entity sits **on** the ground at
/// its authored translation rather than half-buried in it — and the fracture's
/// ground bonds are decided by where the chunks actually are.
fn phase22_box_mesh(min: [f32; 3], max: [f32; 3], material: &str) -> inf_mesh::MeshAsset {
    let n = PHASE22_BOX_SUBDIV;
    let mut vertices: Vec<inf_mesh::MeshVertex> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    // (axis, sign): the six faces, in a fixed order so the asset is a pure
    // function of its bounds.
    for axis in 0..3usize {
        for sign in [-1.0f32, 1.0] {
            let (u_axis, v_axis) = ((axis + 1) % 3, (axis + 2) % 3);
            let base = vertices.len() as u32;
            let mut normal = [0.0f32; 3];
            normal[axis] = sign;
            for j in 0..=n {
                for i in 0..=n {
                    let fu = i as f32 / n as f32;
                    let fv = j as f32 / n as f32;
                    let mut p = [0.0f32; 3];
                    p[axis] = if sign < 0.0 { min[axis] } else { max[axis] };
                    p[u_axis] = min[u_axis] + fu * (max[u_axis] - min[u_axis]);
                    p[v_axis] = min[v_axis] + fv * (max[v_axis] - min[v_axis]);
                    vertices.push(inf_mesh::MeshVertex {
                        position: p,
                        normal,
                        uv: [fu, fv],
                        tangent: [1.0, 0.0, 0.0, 1.0],
                    });
                }
            }
            let stride = n + 1;
            for j in 0..n {
                for i in 0..n {
                    let a = base + j * stride + i;
                    indices.extend_from_slice(&[
                        a,
                        a + stride,
                        a + 1,
                        a + 1,
                        a + stride,
                        a + stride + 1,
                    ]);
                }
            }
        }
    }
    inf_mesh::MeshAsset::new(
        vec![inf_mesh::SubMesh {
            name: "box".into(),
            vertices,
            indices,
            material_slot: Some(0),
            skin: Vec::new(),
        }],
        vec![material.into()],
    )
}

/// The block's footprint and height, metres — **three storeys of three metres**,
/// which is what makes the structural solve do real work: the top storey's chunks
/// reach the ground only *through* the storeys under them, so taking the middle
/// out drops the top.
pub const PHASE22_TOWER_HALF_XZ: f64 = 3.0;
pub const PHASE22_TOWER_HEIGHT_M: f64 = 9.0;

/// The multi-storey block's mesh: a 6 m × 9 m × 6 m tessellated box standing on
/// its own local `y = 0`.
///
/// **Authored geometry, not the P19 grammar, and the reason is structural.** A
/// `Destructible` names no asset of its own: the cook fractures the **one mesh**
/// its actor's `MeshRef.asset` points at (the strength memo's §5 — a second
/// reference would be a second authority for the same fact). The P19 grammar
/// produces a *population*: scattered solids placed as instances, with no merged
/// mesh anywhere in the pipeline. Wiring it would therefore have meant building a
/// grammar→single-mesh bake first, which is a modelling feature and belongs to
/// Phase 23, not to a destruction batch. So this is a box, and saying so is
/// better than implying a building generator was used.
pub fn phase22_tower_mesh() -> inf_mesh::MeshAsset {
    let h = PHASE22_TOWER_HALF_XZ as f32;
    phase22_box_mesh(
        [-h, 0.0, -h],
        [h, PHASE22_TOWER_HEIGHT_M as f32, h],
        "Masonry",
    )
}

/// The car chassis: a 4 m × 1 m × 2 m tessellated box, its local origin at its
/// own centre so the revolute wheel anchors read as offsets from the middle of
/// the car.
pub fn phase22_chassis_mesh() -> inf_mesh::MeshAsset {
    phase22_box_mesh([-2.0, -0.5, -1.0], [2.0, 0.5, 1.0], "Steel")
}

// ── the material classes ─────────────────────────────────────────────────────

/// The block is **masonry**: `docs/memos/p22-strength.md`'s 2–4 MPa class, and
/// brick's density. A wall an explosion opens and a footstep does not.
pub const PHASE22_TOWER_STRENGTH_PA: f64 = 2.5e6;
pub const PHASE22_TOWER_DENSITY: f64 = 1900.0;
/// Chunks the block fractures into. Twenty-four over three storeys is eight a
/// storey, which is the coarsest chunking that still has *interior* chunks — a
/// piece with neighbours on every side is what makes the support graph a graph
/// rather than a stack.
pub const PHASE22_TOWER_CHUNKS: u32 = 24;

/// The chassis is **thin pressed steel**, and this is the one number in the
/// sample that is a judgement rather than a table lookup.
///
/// The memo's steel row is 4e8 Pa, which is bulk steel: at that strength a
/// chassis bond over a ~0.5 m² face costs 200 kJ and a car needs a tactical
/// warhead. The memo's model is a *bulk* failure stress and a monocoque is not
/// bulk — it is a shell that folds — so pricing a car body at its material's
/// yield stress would be an arithmetic answer to the wrong question. 2e7 Pa is
/// the reinforced-concrete row, chosen because it is the value that makes a car
/// take roughly a grenade: strong enough that the same charge that opens the
/// block does not vaporise it, weak enough that it comes apart.
pub const PHASE22_CHASSIS_STRENGTH_PA: f64 = 2.0e7;
pub const PHASE22_CHASSIS_DENSITY: f64 = 7850.0;
pub const PHASE22_CHASSIS_CHUNKS: u32 = 12;

// ── the placements ───────────────────────────────────────────────────────────

/// Where the block stands. Clear of the roller's lane and of the car.
pub const PHASE22_TOWER_XZ: (f64, f64) = (60.0, 24.0);
/// Where its control twin stands: 20 m away, on the same flat ground, under the
/// same script.
pub const PHASE22_CONTROL_XZ: (f64, f64) = (84.0, 24.0);
/// The car's chassis centre. Its underside is 0.3 m off the ground so the first
/// step is a short settle onto the terrain heightfield rather than a jolt.
pub const PHASE22_CAR_XZ: (f64, f64) = (36.0, 20.0);
pub const PHASE22_CHASSIS_Y: f64 = 0.8;
/// Wheel radius, and where the four of them hang off the chassis in its own
/// local frame.
pub const PHASE22_WHEEL_RADIUS_M: f64 = 0.4;
/// Rubber, kg/m³ — authored on the wheels' colliders rather than left at
/// `Collider3D::density`'s default.
///
/// That default is `1.0`, which is **rapier's mass placeholder and not a material
/// density** (the finding P20.2's buoyancy work was built around, and the reason
/// chunk mass comes from `Destructible::density_kg_m3` instead). A 0.4 m sphere at
/// 1.0 kg/m³ weighs 268 grams, so any impulse sized for a chunk sends it into
/// orbit — which is precisely what the phase-22 gate caught on its first run.
pub const PHASE22_WHEEL_DENSITY: f64 = 1100.0;
/// The intact chassis' collider density, kg/m³.
///
/// A car is mostly air: ~1200 kg inside a 4 × 1 × 2 m box is ~150 kg/m³, and that
/// is the number a *settling prop* should have. It is deliberately not the
/// `Destructible::density_kg_m3` of 7850 that its **chunks** use — that one is the
/// steel the body is made of, which is the right density for a fragment of it and
/// the wrong one for the hollow shell as a whole.
pub const PHASE22_CHASSIS_COLLIDER_DENSITY: f64 = 150.0;
pub fn phase22_wheel_offsets() -> [DVec3; 4] {
    [
        DVec3::new(-1.4, -0.5, -1.1),
        DVec3::new(-1.4, -0.5, 1.1),
        DVec3::new(1.4, -0.5, -1.1),
        DVec3::new(1.4, -0.5, 1.1),
    ]
}

/// The roller's lane: it starts on grass just short of the snow, runs **+Z**
/// through the snow band and out into the sand one, and stops on grass again.
pub const PHASE22_ROLLER_X: f64 = 24.0;
pub const PHASE22_ROLLER_START_Z: f64 = 36.0;
pub const PHASE22_ROLLER_RADIUS_M: f64 = 0.5;
/// Metres per second along `+Z`. At 60 Hz and [`PHASE22_STEPS`] the run covers
/// 32 m: grass → snow (16 m) → sand (16 m) → grass, so both deformable
/// archetypes are crossed and both ends of the trail are on ground that barely
/// dents.
pub const PHASE22_ROLLER_SPEED_M_S: f64 = 8.0;
/// The downward speed the roller is held at.
///
/// **It is pressed into the ground, not balanced on it**, and that is the whole
/// reason this is a constant rather than a zero. `physics3d.set_velocity` writes
/// all three components, so a `vy` of zero would leave the body neither falling
/// nor settling — it would ride wherever the solver last left it, and the
/// contact band it has to be inside is 6 cm *above* the surface and 50 cm below
/// it (`inf_ecs::deform::CONTACT_BAND_{ABOVE,BELOW}_M`). Entering the band from
/// below makes the contact eight times as tolerant, and a metre a second is far
/// too slow to tunnel through a heightfield at 60 Hz.
pub const PHASE22_ROLLER_SINK_M_S: f64 = 1.0;

// ── the script ───────────────────────────────────────────────────────────────

/// Steps the gate traces. Argued against the *window law* at the gate; the number
/// lives here so the content and the test cannot disagree about how long the run
/// is.
pub const PHASE22_STEPS: usize = 240;
/// The step the charge goes off on.
///
/// Late enough that the car has settled onto its wheels and the roller is deep in
/// the snow band — so the pre-trigger prefix is a *settled* world rather than a
/// falling one — and early enough that [`PHASE22_STEPS`] leaves 1.4 s of collapse
/// and debris settle after it.
pub const PHASE22_TRIGGER_STEP: f64 = 60.0;
/// …and the step it stops on. The charge is spent over
/// `PHASE22_TRIGGER_STEP ..= PHASE22_TRIGGER_END` rather than in one tick,
/// because damage is **not banked**: `apply_damage` spends what it can on the
/// bonds that exist *now*, and a single blow at t=60 could only ever take the
/// chunks that were cheapest at t=60. Six ticks of charge is a demolition
/// sequence, and it is what drives the structural solve into doing progressive
/// work rather than one-shot removal.
pub const PHASE22_TRIGGER_END: f64 = 65.0;

/// Joules the charge delivers to the block per tick.
///
/// Sized against the block's own numbers rather than picked: a chunk of the
/// 324 m³ block is ~13.5 m³, so its faces are of order 5 m² and one bond costs
/// `2.5e6 × 5 × 1e-3 ≈ 12 kJ`; a chunk with three or four neighbours (plus, at
/// the base, a ground bond over its own `volume^(2/3)`) is therefore 40–60 kJ to
/// liberate. 80 kJ a tick over six ticks is ~480 kJ, which **opens the building
/// without finishing it** — around ten of the twenty-four pieces.
///
/// That is deliberate, and it is the number the first cut got wrong. At 400 kJ a
/// tick the charge simply removed every chunk directly, which looks the same in a
/// screenshot and is a strictly weaker demonstration: nothing is left standing
/// for the structural solve to have an opinion about, and the level's audit
/// counter for progressive collapse reads zero. A charge that leaves two thirds
/// of the block standing is the one that has a *structure* in it afterwards.
/// (`docs/memos/p22-strength.md`'s own sanity check: a grenade is a few kJ, a
/// rocket some tens; this is a demolition charge.)
pub const PHASE22_TOWER_CHARGE_J: f64 = 8.0e4;
/// Joules the wreck script delivers to the car per tick, over the same window.
///
/// **One blow, at [`PHASE22_TRIGGER_STEP`] — not a per-tick charge.** (An earlier
/// draft of this comment described "25 kJ a tick over the window", which was the
/// design before the wreck script became a single instant; it never shipped, and
/// leaving it here would have told the next reader to look for a loop that is not
/// there.) The block's demolition string is the multi-tick one; a car bomb is one
/// bang, for the reason [`phase22_at_trigger`] gives.
///
/// The chassis is far stronger per unit area (2e7 Pa against the block's 2.5e6)
/// and far smaller (0.67 m³ a chunk, faces of order 0.7 m²), so a bond is ~14 kJ
/// and a chunk 30–45 kJ.
///
/// **30 kJ, measured against the shipped asset, and both neighbouring values are
/// wrong for instructive reasons.** The chassis' cheapest chunk costs 25 136 J to
/// liberate.
///
/// * At **25 000 J** the charge spent *nothing*. Damage is not banked: energy
///   that cannot break the cheapest bond set is simply not absorbed, the
///   Blueprint reports 0 J as a legal value, and the level looks untouched. The
///   only thing that noticed was this gate.
/// * At **40 000 J** it took the whole car in one blow. Breaking a chunk makes
///   its neighbours cheaper — their bond to it is gone — so a charge with slack
///   in it **cascades**, and a cascade that reaches every chunk leaves the
///   structural solve with nothing to do.
///
/// 30 kJ takes the cheapest chunk and stops with ~5 kJ unspent. What happens next
/// is the demonstration: a car body is not supported by static geometry, so the
/// eleven chunks still attached are hanging in mid-air on four **dynamic** wheels
/// — which the support rule refuses to count — and the structural solve drops
/// them on the same step. One bond's worth of explosive, eleven chunks of
/// collapse.
pub const PHASE22_CAR_CHARGE_J: f64 = 3.0e4;
/// The blast: newton-seconds at one metre, and its radius in metres.
///
/// It is fired from the car's own place, so it reaches the car's debris and its
/// four wheels and nothing else — the block is 25 m away, well outside. That is
/// deliberate: an explosion that moved every loose body in the level would make
/// "the block's chunks moved" a statement about the blast rather than about the
/// collapse.
///
/// **8 kN·s, and the first cut had 60 — fired every tick.** The number has to be
/// read against the masses it acts on, and the smallest of those is a wheel: a
/// 0.4 m sphere at [`PHASE22_WHEEL_DENSITY`] is ~295 kg, and it sits ~1.8 m from
/// the charge where the inverse-square falloff leaves about a third of the
/// one-metre value. 8 kN·s is therefore ~2.5 kN·s on a wheel, i.e. about 8 m/s —
/// a wheel that is thrown and lands.
///
/// **The two numbers the first cut got wrong were the size AND the count**, and
/// separating them matters because only one is about the constant. At 60 kN·s a
/// single impulse is ~19 kN·s on a wheel, i.e. **64 m/s** — already absurd, and
/// it ballistically reaches ~210 m, not kilometres. What the gate actually
/// measured was a wheel at **y = 60.8 m still climbing**, because the impulse was
/// fired on *every one of the six charge ticks*: six times 64 m/s, minus what
/// gravity took back. So the honest account is "60 kN·s six times over put a
/// wheel past 60 m and rising", and the fix was both halves —
/// [`phase22_at_trigger`] made it one bang, and this constant made that bang the
/// right size. (An earlier version of this paragraph said "13 km up", which is
/// neither the measurement nor consistent with its own 60 m/s; it is corrected
/// rather than deleted because the *lesson* — size a blast against the lightest
/// body in its radius — was paid for and is worth keeping accurate.)
pub const PHASE22_BLAST_NS: f64 = 8.0e3;
pub const PHASE22_BLAST_RADIUS_M: f64 = 12.0;

// ── the Blueprints ───────────────────────────────────────────────────────────

/// `vars.get("entity")` — the only entity reference the Blueprint IR has, and
/// therefore the reason every `destruct.*` script in this sample sits on the
/// actor it breaks (the P21 borer's limitation, unchanged).
fn phase22_self() -> Expr {
    Expr::Call {
        path: vec!["vars".into(), "get".into()],
        args: vec![Expr::Lit(Lit::Str("entity".into()))],
    }
}

fn phase22_get(name: &str) -> Expr {
    Expr::Call {
        path: vec!["vars".into(), "get".into()],
        args: vec![Expr::Lit(Lit::Str(name.into()))],
    }
}

fn phase22_set(name: &str, value: Expr) -> Stmt {
    Stmt::ExprStmt(Expr::Call {
        path: vec!["vars".into(), "set".into()],
        args: vec![Expr::Lit(Lit::Str(name.into())), value],
    })
}

fn phase22_slot(name: &str, default: f64) -> inf_blueprint::Variable {
    inf_blueprint::Variable {
        name: name.into(),
        ty: inf_blueprint::Ty::Float,
        default: Lit::Float(default),
        exposed: false,
    }
}

fn phase22_tick_params() -> Vec<inf_blueprint::Param> {
    vec![inf_blueprint::Param {
        name: "dt".into(),
        ty: inf_blueprint::Ty::Float,
    }]
}

/// `step >= TRIGGER_STEP && step <= TRIGGER_END` — the scripted window.
///
/// Written against the actor's **own tick counter** rather than a clock, so the
/// charge goes off on the same fixed step in the editor preview, in the shipped
/// build and in a replay. That is the whole reason the counter exists.
fn phase22_in_window() -> Expr {
    use inf_blueprint::BinOp;
    Expr::Binary(
        BinOp::And,
        Box::new(Expr::Binary(
            BinOp::Ge,
            Box::new(phase22_get("step")),
            Box::new(Expr::Lit(Lit::Float(PHASE22_TRIGGER_STEP))),
        )),
        Box::new(Expr::Binary(
            BinOp::Le,
            Box::new(phase22_get("step")),
            Box::new(Expr::Lit(Lit::Float(PHASE22_TRIGGER_END))),
        )),
    )
}

/// The `Destroyed` handler both destructible classes share: record the chunk
/// count, **count** the firings (latching would make "once" uncheckable — the
/// P22.3 lesson), and play the actor's `AudioSource`.
///
/// The audio hook is the phase's own "audio/VFX event hooks" line, half of it:
/// there is no particle system in this engine so a break makes no dust, and that
/// is ledgered rather than faked — but the sound is real, goes through the P12
/// command queue, and fires exactly once.
fn phase22_destroyed_binding() -> EventBinding {
    use inf_blueprint::BinOp;
    EventBinding {
        event: EventKind::Destroyed,
        body: BlueprintFn {
            id: "destroyed".into(),
            name: "destroyed".into(),
            params: EventKind::Destroyed.signature(),
            ret: Ty::Unit,
            body: vec![
                phase22_set("destroyed_chunks", phase22_get("chunks")),
                phase22_set(
                    "destroyed_fires",
                    Expr::Binary(
                        BinOp::Add,
                        Box::new(phase22_get("destroyed_fires")),
                        Box::new(Expr::Lit(Lit::Float(1.0))),
                    ),
                ),
                Stmt::ExprStmt(Expr::Call {
                    path: vec!["audio".into(), "play".into()],
                    args: vec![phase22_self()],
                }),
            ],
        },
    }
}

/// The variables every `destruct.*` script in this sample keeps. Shared so the
/// tower, its control and the car report through the same names and the gate can
/// read any of them with one accessor.
fn phase22_destruct_vars() -> Vec<inf_blueprint::Variable> {
    vec![
        // The actor's own tick counter — the script's clock.
        phase22_slot("step", 0.0),
        // Joules this tick's charge actually consumed.
        phase22_slot("absorbed", 0.0),
        // …and their sum, which is what a refusal makes stay at zero for ever.
        phase22_slot("total", 0.0),
        // The two queries, read AFTER the charge in the same handler.
        phase22_slot("intact", -1.0),
        phase22_slot("chunks", -1.0),
        // What `Destroyed` saw, and how often it saw it.
        phase22_slot("destroyed_chunks", -1.0),
        phase22_slot("destroyed_fires", 0.0),
    ]
}

/// **The demolition charge** — carried by the multi-storey block AND, unchanged,
/// by its control twin.
///
/// One class on two actors is what makes the control a control: the two entities
/// differ in `Destructible::runtime_destruct` and in nothing else at all, so a
/// difference in outcome cannot be a difference in script.
pub fn phase22_demolition_class() -> BlueprintClass {
    use inf_blueprint::BinOp;
    let mut class = BlueprintClass::new("act:phase22-demolition", "Demolition Charge");
    class.variables = phase22_destruct_vars();
    class.events = vec![
        EventBinding {
            event: EventKind::Tick,
            body: BlueprintFn {
                id: "tick".into(),
                name: "tick".into(),
                params: phase22_tick_params(),
                ret: Ty::Unit,
                body: vec![
                    Stmt::If {
                        cond: phase22_in_window(),
                        then_body: vec![
                            phase22_set(
                                "absorbed",
                                Expr::Call {
                                    path: vec!["destruct".into(), "apply_damage".into()],
                                    args: vec![
                                        phase22_self(),
                                        Expr::Lit(Lit::Float(PHASE22_TOWER_CHARGE_J)),
                                    ],
                                },
                            ),
                            phase22_set(
                                "total",
                                Expr::Binary(
                                    BinOp::Add,
                                    Box::new(phase22_get("total")),
                                    Box::new(phase22_get("absorbed")),
                                ),
                            ),
                        ],
                        else_body: Vec::new(),
                    },
                    phase22_set(
                        "step",
                        Expr::Binary(
                            BinOp::Add,
                            Box::new(phase22_get("step")),
                            Box::new(Expr::Lit(Lit::Float(1.0))),
                        ),
                    ),
                    phase22_set(
                        "intact",
                        Expr::Call {
                            path: vec!["destruct".into(), "is_intact".into()],
                            args: vec![phase22_self()],
                        },
                    ),
                    phase22_set(
                        "chunks",
                        Expr::Call {
                            path: vec!["destruct".into(), "chunk_count".into()],
                            args: vec![phase22_self()],
                        },
                    ),
                ],
            },
        },
        phase22_destroyed_binding(),
    ];
    class
}

/// `step == TRIGGER_STEP` — a single scripted instant.
///
/// The car's charge fires **once**, unlike the block's demolition string, and
/// both halves of that are the physics rather than a preference. An explosion is
/// one impulse: firing it on every tick of a six-tick window multiplies its
/// momentum by six, and on the lightest body in the radius — a wheel — the gate
/// measured that as sixty metres of altitude. And the *damage* has to be one blow
/// too, because a car body is not supported by static geometry: the first chunk
/// that comes off makes the rest unsupported, so a second tick of charge would be
/// spending joules on bonds the structural solve had already broken for free.
fn phase22_at_trigger() -> Expr {
    use inf_blueprint::BinOp;
    Expr::Binary(
        BinOp::Eq,
        Box::new(phase22_get("step")),
        Box::new(Expr::Lit(Lit::Float(PHASE22_TRIGGER_STEP))),
    )
}

/// **The car bomb** — damage *and* the radial impulse, at one scripted instant
/// inside the block's charge window.
///
/// The blast is fired from the car's own place and reaches [`PHASE22_BLAST_RADIUS_M`],
/// which covers the chassis' own debris and its four wheels and stops well short
/// of the block 25 m away. That containment is deliberate: an explosion that
/// moved every loose body in the level would make "the block's chunks moved" a
/// statement about the blast rather than about the collapse.
pub fn phase22_wreck_class() -> BlueprintClass {
    use inf_blueprint::BinOp;
    let (cx, cz) = PHASE22_CAR_XZ;
    let mut class = BlueprintClass::new("act:phase22-wreck", "Car Bomb");
    class.variables = {
        let mut v = phase22_destruct_vars();
        // How many bodies the blast pushed — the one number that says the impulse
        // reached the debris it had just made.
        v.push(phase22_slot("hit", -1.0));
        v
    };
    class.events = vec![
        EventBinding {
            event: EventKind::Tick,
            body: BlueprintFn {
                id: "tick".into(),
                name: "tick".into(),
                params: phase22_tick_params(),
                ret: Ty::Unit,
                body: vec![
                    Stmt::If {
                        cond: phase22_at_trigger(),
                        then_body: vec![
                            phase22_set(
                                "absorbed",
                                Expr::Call {
                                    path: vec!["destruct".into(), "apply_damage".into()],
                                    args: vec![
                                        phase22_self(),
                                        Expr::Lit(Lit::Float(PHASE22_CAR_CHARGE_J)),
                                    ],
                                },
                            ),
                            phase22_set(
                                "total",
                                Expr::Binary(
                                    BinOp::Add,
                                    Box::new(phase22_get("total")),
                                    Box::new(phase22_get("absorbed")),
                                ),
                            ),
                            phase22_set(
                                "hit",
                                Expr::Call {
                                    path: vec!["destruct".into(), "radial_impulse".into()],
                                    args: vec![
                                        Expr::Lit(Lit::Float(cx)),
                                        Expr::Lit(Lit::Float(PHASE22_CHASSIS_Y)),
                                        Expr::Lit(Lit::Float(cz)),
                                        Expr::Lit(Lit::Float(PHASE22_BLAST_NS)),
                                        Expr::Lit(Lit::Float(PHASE22_BLAST_RADIUS_M)),
                                    ],
                                },
                            ),
                        ],
                        else_body: Vec::new(),
                    },
                    phase22_set(
                        "step",
                        Expr::Binary(
                            BinOp::Add,
                            Box::new(phase22_get("step")),
                            Box::new(Expr::Lit(Lit::Float(1.0))),
                        ),
                    ),
                    phase22_set(
                        "intact",
                        Expr::Call {
                            path: vec!["destruct".into(), "is_intact".into()],
                            args: vec![phase22_self()],
                        },
                    ),
                    phase22_set(
                        "chunks",
                        Expr::Call {
                            path: vec!["destruct".into(), "chunk_count".into()],
                            args: vec![phase22_self()],
                        },
                    ),
                ],
            },
        },
        phase22_destroyed_binding(),
    ];
    class
}

/// **The roller** — the moving body that leaves the tracks.
///
/// It drives itself with `physics3d.set_velocity` on every Tick: `+Z` at
/// [`PHASE22_ROLLER_SPEED_M_S`] and a steady [`PHASE22_ROLLER_SINK_M_S`]
/// downward, so it is *pressed into* the surface rather than balanced on it (see
/// that constant). Its body has `gravity_scale = 0`, which is what makes the run
/// a scripted probe: the only vertical input is the one the script writes, so the
/// trail is a function of the script and of the ground and of nothing else.
pub fn phase22_roller_class() -> BlueprintClass {
    use inf_blueprint::BinOp;
    let mut class = BlueprintClass::new("act:phase22-roller", "Roller");
    class.variables = vec![phase22_slot("step", 0.0)];
    class.events = vec![EventBinding {
        event: EventKind::Tick,
        body: BlueprintFn {
            id: "tick".into(),
            name: "tick".into(),
            params: phase22_tick_params(),
            ret: Ty::Unit,
            body: vec![
                Stmt::ExprStmt(Expr::Call {
                    path: vec!["physics3d".into(), "set_velocity".into()],
                    args: vec![
                        phase22_self(),
                        Expr::Lit(Lit::Float(0.0)),
                        Expr::Lit(Lit::Float(-PHASE22_ROLLER_SINK_M_S)),
                        Expr::Lit(Lit::Float(PHASE22_ROLLER_SPEED_M_S)),
                    ],
                }),
                phase22_set(
                    "step",
                    Expr::Binary(
                        BinOp::Add,
                        Box::new(phase22_get("step")),
                        Box::new(Expr::Lit(Lit::Float(1.0))),
                    ),
                ),
            ],
        },
    }];
    class
}

// ── the scene ────────────────────────────────────────────────────────────────

/// Build the committed Phase 22 [`SceneDoc`].
///
/// The terrain is **asset-backed** (`Terrain.asset`), like Phase 21's and for the
/// neighbouring reason: this is the streamed-terrain configuration, the one whose
/// tiles page in from a `.inf_terrain` and whose sim-resident level-0 tiles the
/// P22.3 bridge turns into heightfield colliders. Everything in this level that
/// falls, lands on those.
pub fn phase22_playground_scene() -> SceneDoc {
    use crate::scene::serialize::LevelSettings;
    use inf_ecs::components::{
        ActorClass, AudioSource, BodyKind3D, Camera, Collider3D, ColliderShape3DKind, Destructible,
        Foliage, FoliageInstance, FoliagePaletteEntry, Joint3D, JointKind3D, Light, LightKind,
        Material, MeshRef, Primitive, RigidBody3D, SkyAtmosphere, Terrain, TimeOfDay,
    };
    use inf_ecs::math::{Color, Vec2d, Vec3d};

    let mut doc = SceneDoc::new();
    doc.set_title("Phase 22 Playground");
    doc.set_settings(LevelSettings {
        gravity_2d: Vec2d::new(0.0, -9.81),
        gravity_3d: Vec3d::new(0.0, -9.81, 0.0),
        sim_hz: 60.0,
        ..LevelSettings::default()
    });

    // ── The ground: flat, painted snow and sand in two bands. ──
    doc.create_with_guid(PHASE22_TERRAIN_GUID, SpawnKind::Empty, "Ground", None);
    insert!(
        doc,
        PHASE22_TERRAIN_GUID,
        Transform::from_translation(DVec3::ZERO)
    );
    insert!(
        doc,
        PHASE22_TERRAIN_GUID,
        Terrain {
            meters_per_sample: PHASE22_MPS,
            tile_resolution: PHASE22_TILE_RES,
            data: phase22_terrain_data(),
            asset: Some(PHASE22_TERRAIN_ASSET_GUID),
            ..Terrain::default()
        }
    );

    // ── The multi-storey block, and its control twin. ──
    for (guid, name, xz, permitted) in [
        (PHASE22_TOWER_GUID, "Block", PHASE22_TOWER_XZ, true),
        (
            PHASE22_CONTROL_GUID,
            "Block (Runtime Destruct Off)",
            PHASE22_CONTROL_XZ,
            false,
        ),
    ] {
        doc.create_with_guid(guid, SpawnKind::Empty, name, None);
        insert!(
            doc,
            guid,
            Transform::from_translation(DVec3::new(xz.0, 0.0, xz.1))
        );
        insert!(
            doc,
            guid,
            MeshRef {
                primitive: Primitive::Cube,
                asset: Some(PHASE22_BLOCK_MESH_GUID),
            }
        );
        insert!(
            doc,
            guid,
            Material {
                base_color: Color::new(0.62, 0.58, 0.52, 1.0),
                roughness: 0.85,
                ..Material::default()
            }
        );
        // Static: a building is level geometry until it is not. Its chunks become
        // dynamic bodies at the break; the actor's own collider disappears in the
        // same atomic swap.
        insert!(
            doc,
            guid,
            Collider3D {
                shape_kind: ColliderShape3DKind::Box,
                half_extents: Vec3d::new(
                    PHASE22_TOWER_HALF_XZ,
                    PHASE22_TOWER_HEIGHT_M * 0.5,
                    PHASE22_TOWER_HALF_XZ
                ),
                offset: Vec3d::new(0.0, PHASE22_TOWER_HEIGHT_M * 0.5, 0.0),
                ..Collider3D::default()
            }
        );
        insert!(
            doc,
            guid,
            Destructible {
                fracture_seed: 22,
                chunk_count: PHASE22_TOWER_CHUNKS,
                strength: PHASE22_TOWER_STRENGTH_PA,
                density_kg_m3: PHASE22_TOWER_DENSITY,
                runtime_destruct: permitted,
            }
        );
        insert!(doc, guid, ActorClass(PHASE22_DEMO_ACTOR_GUID));
        insert!(
            doc,
            guid,
            AudioSource {
                clip: Some(PHASE22_CLIP_GUID),
                looping: false,
                ..AudioSource::default()
            }
        );
    }

    // ── The car: a destructible chassis on four revolute wheels. ──
    let (cx, cz) = PHASE22_CAR_XZ;
    doc.create_with_guid(PHASE22_CHASSIS_GUID, SpawnKind::Empty, "Car Chassis", None);
    insert!(
        doc,
        PHASE22_CHASSIS_GUID,
        Transform::from_translation(DVec3::new(cx, PHASE22_CHASSIS_Y, cz))
    );
    insert!(
        doc,
        PHASE22_CHASSIS_GUID,
        MeshRef {
            primitive: Primitive::Cube,
            asset: Some(PHASE22_CHASSIS_MESH_GUID),
        }
    );
    insert!(
        doc,
        PHASE22_CHASSIS_GUID,
        Material {
            base_color: Color::new(0.30, 0.36, 0.48, 1.0),
            metallic: 0.9,
            roughness: 0.35,
            ..Material::default()
        }
    );
    insert!(
        doc,
        PHASE22_CHASSIS_GUID,
        RigidBody3D {
            kind: BodyKind3D::Dynamic,
            ..RigidBody3D::default()
        }
    );
    insert!(
        doc,
        PHASE22_CHASSIS_GUID,
        Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: Vec3d::new(2.0, 0.5, 1.0),
            density: PHASE22_CHASSIS_COLLIDER_DENSITY,
            ..Collider3D::default()
        }
    );
    insert!(
        doc,
        PHASE22_CHASSIS_GUID,
        Destructible {
            fracture_seed: 22,
            chunk_count: PHASE22_CHASSIS_CHUNKS,
            strength: PHASE22_CHASSIS_STRENGTH_PA,
            density_kg_m3: PHASE22_CHASSIS_DENSITY,
            runtime_destruct: true,
        }
    );
    insert!(
        doc,
        PHASE22_CHASSIS_GUID,
        ActorClass(PHASE22_WRECK_ACTOR_GUID)
    );

    for (i, offset) in phase22_wheel_offsets().into_iter().enumerate() {
        let guid = PHASE22_WHEEL_GUIDS[i];
        doc.create_with_guid(guid, SpawnKind::Empty, &format!("Wheel {i}"), None);
        insert!(
            doc,
            guid,
            Transform::from_translation(DVec3::new(cx, PHASE22_CHASSIS_Y, cz) + offset)
        );
        insert!(
            doc,
            guid,
            MeshRef {
                primitive: Primitive::Sphere,
                asset: None,
            }
        );
        insert!(
            doc,
            guid,
            Material {
                base_color: Color::new(0.09, 0.09, 0.10, 1.0),
                roughness: 0.95,
                ..Material::default()
            }
        );
        insert!(
            doc,
            guid,
            RigidBody3D {
                kind: BodyKind3D::Dynamic,
                ..RigidBody3D::default()
            }
        );
        insert!(
            doc,
            guid,
            Collider3D {
                shape_kind: ColliderShape3DKind::Sphere,
                radius: PHASE22_WHEEL_RADIUS_M,
                density: PHASE22_WHEEL_DENSITY,
                ..Collider3D::default()
            }
        );
        // The hinge. `other_anchor` is the wheel's seat in the CHASSIS' local
        // frame, `local_anchor` the wheel's own centre — so the joint describes
        // the car's geometry and not the pair's current world poses.
        insert!(
            doc,
            guid,
            Joint3D {
                other: inf_ecs::refs::EntityRef::new(PHASE22_CHASSIS_GUID),
                kind: JointKind3D::Revolute,
                local_anchor: Vec3d::ZERO,
                other_anchor: Vec3d::new(offset.x, offset.y, offset.z),
                axis: Vec3d::new(0.0, 0.0, 1.0),
                ..Joint3D::default()
            }
        );
    }

    // ── The roller. ──
    doc.create_with_guid(PHASE22_ROLLER_GUID, SpawnKind::Empty, "Roller", None);
    insert!(
        doc,
        PHASE22_ROLLER_GUID,
        Transform::from_translation(DVec3::new(
            PHASE22_ROLLER_X,
            PHASE22_ROLLER_RADIUS_M,
            PHASE22_ROLLER_START_Z
        ))
    );
    insert!(
        doc,
        PHASE22_ROLLER_GUID,
        MeshRef {
            primitive: Primitive::Sphere,
            asset: None,
        }
    );
    insert!(
        doc,
        PHASE22_ROLLER_GUID,
        RigidBody3D {
            kind: BodyKind3D::Dynamic,
            // The script owns the vertical: see `PHASE22_ROLLER_SINK_M_S`.
            gravity_scale: 0.0,
            fixed_rotation: true,
            ..RigidBody3D::default()
        }
    );
    insert!(
        doc,
        PHASE22_ROLLER_GUID,
        Collider3D {
            shape_kind: ColliderShape3DKind::Sphere,
            radius: PHASE22_ROLLER_RADIUS_M,
            ..Collider3D::default()
        }
    );
    insert!(
        doc,
        PHASE22_ROLLER_GUID,
        ActorClass(PHASE22_ROLLER_ACTOR_GUID)
    );

    // ── The grass the tracks bend. ──
    //
    // A lattice straddling the roller's lane, so instances stand both inside the
    // trail and clear of it: the bend shader reads the same deformation window the
    // terrain displacement does, and a strip entirely inside the rut would bend
    // uniformly and prove nothing.
    doc.create_with_guid(PHASE22_GRASS_GUID, SpawnKind::Empty, "Grass", None);
    insert!(
        doc,
        PHASE22_GRASS_GUID,
        Transform::from_translation(DVec3::new(PHASE22_ROLLER_X, 0.0, 56.0))
    );
    {
        let mut instances = Vec::new();
        for j in 0..PHASE22_GRASS_ROWS {
            for i in 0..PHASE22_GRASS_COLS {
                let x = -6.0 + i as f64 * (12.0 / (PHASE22_GRASS_COLS - 1) as f64);
                let z = -16.0 + j as f64 * (32.0 / (PHASE22_GRASS_ROWS - 1) as f64);
                instances.push(FoliageInstance {
                    position: Vec3d::new(x, 0.0, z),
                    // Euler degrees, deterministic from the lattice index — no RNG
                    // in committed content.
                    rotation: Vec3d::new(0.0, ((i * 37 + j * 11) % 360) as f64, 0.0),
                    scale: 0.8 + ((i * 7 + j * 3) % 5) as f64 * 0.1,
                    kind: 0,
                });
            }
        }
        insert!(
            doc,
            PHASE22_GRASS_GUID,
            Foliage {
                palette: vec![FoliagePaletteEntry {
                    primitive: Primitive::Cone,
                    tint: Color::new(0.30, 0.48, 0.22, 1.0),
                }],
                instances,
            }
        );
    }

    // ── Sky + sun + camera, so the scene opens somewhere sensible. ──
    doc.create_with_guid(PHASE22_SKY_GUID, SpawnKind::Empty, "Sky", None);
    insert!(
        doc,
        PHASE22_SKY_GUID,
        TimeOfDay {
            seconds: 9.0 * 3600.0,
            rate: 0.0,
            ..TimeOfDay::default()
        }
    );
    insert!(doc, PHASE22_SKY_GUID, SkyAtmosphere::default());

    doc.create_with_guid(PHASE22_SUN_GUID, SpawnKind::Empty, "Sun", None);
    insert!(
        doc,
        PHASE22_SUN_GUID,
        Light {
            kind: LightKind::Directional,
            intensity: 3.0,
            ..Light::default()
        }
    );
    insert!(
        doc,
        PHASE22_SUN_GUID,
        Transform::from_translation(DVec3::new(0.0, 80.0, 0.0))
    );

    doc.create_with_guid(PHASE22_CAMERA_GUID, SpawnKind::Empty, "Camera", None);
    insert!(doc, PHASE22_CAMERA_GUID, Camera::default());
    insert!(
        doc,
        PHASE22_CAMERA_GUID,
        Transform::from_translation(DVec3::new(30.0, 18.0, 8.0))
    );

    doc.world_mut().propagate();
    doc.mark_saved();
    doc
}

/// The grass lattice. Small on purpose: the bend claim is about the *seam*, and
/// 21 × 21 instances is a legible strip that costs nothing to cook.
pub const PHASE22_GRASS_COLS: u32 = 21;
pub const PHASE22_GRASS_ROWS: u32 = 21;

// ── the committed files ──────────────────────────────────────────────────────

/// `samples/phase22-playground`, resolved from this crate's manifest dir.
pub fn phase22_playground_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../samples/phase22-playground")
}

/// The `.inf_terrain` the level's `Terrain.asset` names.
pub fn phase22_terrain_asset() -> Result<inf_terrain::TerrainAsset, String> {
    let data = phase22_terrain_data();
    let opts = inf_terrain::PyramidOptions::default();
    let pyramid = inf_terrain::build_pyramid(&data, opts);
    inf_terrain::build_terrain_asset(&data, &pyramid, opts)
        .map_err(|e| format!("build terrain asset: {e}"))
}

/// The collapse's `.inf_audio` clip — the same deterministic tone the physics
/// playground commits, so no binary fixture is needed.
pub fn phase22_audio_asset() -> inf_audio::AudioAsset {
    playground_audio_asset()
}

/// Write every committed Phase 22 gate file.
pub fn write_phase22_playground() -> Result<(), String> {
    let dir = phase22_playground_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("mkdir: {e}"))?;

    crate::scene::serialize::save(
        &phase22_playground_scene(),
        &dir.join("Phase22Playground.inf_lvl"),
        Some(PHASE22_LEVEL_GUID),
    )?;

    let terrain = phase22_terrain_asset()?;
    let tpath = dir.join("Phase22Playground.inf_terrain");
    let tbytes = inf_terrain::write_terrain_asset(&tpath, &terrain)
        .map_err(|e| format!("write terrain asset: {e}"))?;
    inf_asset::AssetSidecar::new(
        inf_asset::AssetId(PHASE22_TERRAIN_ASSET_GUID),
        inf_asset::AssetKind::Terrain,
        inf_asset::ContentHash::of(tbytes),
    )
    .save(&tpath)
    .map_err(|e| format!("write terrain sidecar: {e}"))?;

    for (file, mesh, guid) in [
        (
            "Block.inf_mesh",
            phase22_tower_mesh(),
            PHASE22_BLOCK_MESH_GUID,
        ),
        (
            "Chassis.inf_mesh",
            phase22_chassis_mesh(),
            PHASE22_CHASSIS_MESH_GUID,
        ),
    ] {
        let bytes = inf_asset::encode(&mesh).map_err(|e| format!("encode {file}: {e}"))?;
        write_phase19_asset(&dir.join(file), &bytes, guid, inf_asset::AssetKind::Mesh)?;
    }

    for (file, class, guid) in [
        (
            "Demolition.inf_act",
            phase22_demolition_class(),
            PHASE22_DEMO_ACTOR_GUID,
        ),
        (
            "Wreck.inf_act",
            phase22_wreck_class(),
            PHASE22_WRECK_ACTOR_GUID,
        ),
        (
            "Roller.inf_act",
            phase22_roller_class(),
            PHASE22_ROLLER_ACTOR_GUID,
        ),
    ] {
        let bytes = encode_actor(&class)?;
        write_phase19_asset(
            &dir.join(file),
            &bytes,
            guid,
            inf_asset::AssetKind::Blueprint,
        )?;
    }

    write_anim_asset(
        &dir,
        "Collapse.inf_audio",
        PHASE22_CLIP_GUID,
        inf_asset::AssetKind::Audio,
        &phase22_audio_asset(),
    )?;

    std::fs::write(dir.join("README.md"), PHASE22_PLAYGROUND_README)
        .map_err(|e| format!("write readme: {e}"))?;
    Ok(())
}

const PHASE22_PLAYGROUND_README: &str = "# Phase 22 Playground (the phase gate scene)\n\n\
Generated by `inf_editor_core::samples::phase22_playground_scene` -- the\n\
**composed** gate scene for Phase 22 (dynamic world: deformation & destruction),\n\
and the plan's own done-when sentence built: *a playground scene shows footprints\n\
and tyre tracks in snow and sand, bending grass, and a car and a multi-storey\n\
building destroyed by Blueprint-triggered explosions with debris physics --\n\
deterministic on the replay trace, PIE == shipping.*\n\n\
## What is in it\n\n\
- `Phase22Playground.inf_lvl` -- a 128 m square of **asset-backed (streamed)**\n\
  terrain, flat where anything stands, painted in two bands: **snow** (layer 3,\n\
  the deep non-recovering archetype) over `z` 40-56 and **sand** (layer 2, the\n\
  loose one that slumps over minutes) over `z` 56-64, grass everywhere else --\n\
  so the roller's straight run crosses both and ends back on ground that barely\n\
  dents, which is what makes 'the trail has a far end' a testable claim.\n\
- `Phase22Playground.inf_terrain` -- that heightfield and its splat weights. The\n\
  terrain is asset-backed because this is the **streamed** configuration, and\n\
  because the P22.3 physics bridge turns sim-resident level-0 tiles into\n\
  heightfield colliders: everything in this level that falls, lands on those.\n\
- `Block.inf_mesh` -- a 6 x 9 x 6 m tessellated box, three storeys of three\n\
  metres, 3 072 triangles. **Authored geometry, not the P19 grammar**: a\n\
  `Destructible` fractures the ONE mesh its actor's `MeshRef.asset` names, and the\n\
  grammar produces a scattered *population* with no merged mesh anywhere in the\n\
  pipeline -- so using it would have meant building a grammar-to-mesh bake first,\n\
  which is a modelling feature and belongs to Phase 23.\n\
- `Chassis.inf_mesh` -- the car's 4 x 1 x 2 m body, likewise tessellated (both\n\
  meshes clear the cook's 2 048-triangle vgeom threshold, because the gate's\n\
  cook-silence arm would otherwise fail on a sub-threshold advisory).\n\
- `Demolition.inf_act` -- the charge. On Tick, over steps 60-65, it spends\n\
  80 kJ a tick on its own actor's bonds and then reports `is_intact` and\n\
  `chunk_count`; its `Destroyed` handler counts its firings and plays the actor's\n\
  `AudioSource`. **Two actors carry this one class**: the block, and a control\n\
  twin 24 m away whose `Destructible::runtime_destruct` is `false`. They differ in\n\
  that flag and in nothing else -- not even in what they are made of -- which is\n\
  what makes the control a control.\n\
- `Wreck.inf_act` -- the car bomb: ONE instant at step 60, 30 kJ of damage plus\n\
  an 8 kN.s radial impulse at the car's own place with a 12 m radius. Both\n\
  'once' and the size are the physics: an explosion is one impulse, and firing\n\
  it on every tick of a six-tick window multiplied its momentum by six -- which\n\
  the gate measured as a wheel 60 m up. The blast reaches the chassis' debris\n\
  and its four wheels and stops well short of the block, so 'the block's chunks\n\
  moved' stays a statement about the collapse.\n\
  30 kJ is one chunk's worth of bonds, and what happens next is the point: a car\n\
  body is not supported by static geometry, so the eleven chunks still attached\n\
  are hanging on four DYNAMIC wheels the support rule refuses to count, and the\n\
  structural solve drops the lot.\n\
- `Roller.inf_act` -- the thing that leaves the tracks. Every Tick it writes its\n\
  own velocity (`physics3d.set_velocity`): 8 m/s along +Z and a steady 1 m/s\n\
  down, so it is *pressed into* the surface rather than balanced on it. Its body\n\
  has `gravity_scale = 0`, which makes the run a scripted probe: the only\n\
  vertical input is the one the script writes.\n\
- `Collapse.inf_audio` -- the short deterministic tone the `Destroyed` handler\n\
  plays (the same generator the physics playground commits, so no binary fixture\n\
  is needed).\n\
- A **car**: a destructible chassis with four wheels on revolute joints. It is a\n\
  **prop, not a vehicle** -- this engine has no vehicle controller -- so it\n\
  settles onto its wheels and is then blown up, which is all the phase claims.\n\
  Its colliders carry **authored densities** (rubber 1100 for the wheels, 150 for\n\
  the hollow chassis shell). If you copy one thing out of this sample, copy that:\n\
  `Collider3D::density` defaults to **1.0 kg/m3**, which is rapier's mass\n\
  placeholder and lighter than air. A 0.4 m wheel at the default weighs 268\n\
  grams, and an impulse sized against 5-tonne fracture chunks throws it out of\n\
  the level. (A chunk's mass never comes from there -- it comes from\n\
  `Destructible::density_kg_m3`, which exists for exactly this reason.)\n\
- A **grass strip** straddling the roller's lane, so instances stand both inside\n\
  the trail and clear of it: the P22.1 bend shader reads the same deformation\n\
  window the terrain displacement does, and a strip entirely inside the rut would\n\
  bend uniformly and prove nothing.\n\n\
## What it is NOT\n\n\
No dust and no smoke: this engine has no particle system, so a break makes a\n\
sound and no visual effect. That is ledgered rather than faked. The sub-chunk\n\
**rubble** P22.4 adds is real, but it is instanced render dressing derived from\n\
the detach events -- it is not simulated and never feeds back into the sim.\n\n\
Destruction is **not persisted**: rubble dies with the session, and saving after\n\
a break produces a byte-identical `.inf_lvl`\n\
(`inf-editor-core`'s `simulate_destruction_not_persisted`).\n\n\
## The gate (`runtime/inf-player/tests/phase22_gate.rs`)\n\n\
Two loads bit-identical; cooked == uncooked **up to the charge** (a dev directory\n\
has no `.inf_fracture` by design) and demonstrably different past it; PIE ==\n\
shipping on the full trace (deformation field bytes + every destructible's\n\
`FractureState` bits + the fixed step's audit counters + ground probes) compared\n\
as raw bits, per step; a REAL `--pie` subprocess against the in-process\n\
reference; the world assertions (the charge OPENED the block and left a\n\
structure behind, the car fractured and the structural solve dropped what the\n\
charge did not pay for, the trail is where the roller ran and nowhere else, the\n\
debris budget held, the control twin survived the identical charge); budgets;\n\
pool-size determinism; and the save-after-damage identity.\n\n\
Regenerate with `INF_BLESS_SAMPLES=1 cargo test -p inf-editor-core samples`.\n";

// ── Phase 23 gate scene: `samples/phase23-workshop` (P23.6) ──────────────────
//
// **The smallest sample in the tree, and deliberately so.** Every other phase
// gate ships a level whose *content* is the claim. This phase's claim is about a
// PIPELINE — model, unwrap, save, live-update, undo, replay — so the committed
// level's job is only to reference a mesh and be simulatable while that mesh is
// edited underneath it. Anything more would be content the gate does not read.
//
// What IS load-bearing is `Prop.inf_mesh`: it is the **import baseline**, the
// state undo has to land on byte for byte (gate arm (e)), so it is committed and
// locked like every other sample payload.

const PHASE23_LEVEL_GUID: Uuid = uuid::uuid!("23060000-0000-4000-8000-000000000001");
/// The mesh the gate edits. Committed as a plain cube — the "cube primitive"
/// the phase's done-when sentence starts from.
pub const PHASE23_MESH_GUID: Uuid = uuid::uuid!("23060000-0000-4000-8000-000000000002");
pub const PHASE23_SUN_GUID: Uuid = uuid::uuid!("23060000-0000-4000-8000-000000000010");
pub const PHASE23_GROUND_GUID: Uuid = uuid::uuid!("23060000-0000-4000-8000-000000000011");
/// The actor that draws the edited mesh and falls while it is being edited.
pub const PHASE23_PROP_GUID: Uuid = uuid::uuid!("23060000-0000-4000-8000-000000000012");

/// The baseline prop's edge, metres.
pub const PHASE23_PROP_SIZE_M: f64 = 2.0;
/// Steps the gate's Simulate arm traces: 1.5 s at 60 Hz, which is fall, contact
/// and settle. The save is spliced at [`PHASE23_SAVE_AT`] — mid-fall, where a
/// perturbation would be loudest.
pub const PHASE23_STEPS: usize = 90;
pub const PHASE23_SAVE_AT: usize = 30;
/// How far the gate extrudes the prop's lid, metres.
pub const PHASE23_EXTRUDE_M: f64 = 1.5;
/// The bevel the gate rounds the extruded rim with, metres.
pub const PHASE23_BEVEL_M: f64 = 0.2;

/// The committed baseline: a cube, written through the DCC's own writer.
///
/// Through `inf_dcc::to_mesh_asset` rather than hand-built triangles, because the
/// gate's undo arm compares this against what the kernel produces after a full
/// undo — and a hand-built asset would differ in its corner interning order
/// alone, which is part of the format (the P23.3 LAW).
pub fn phase23_baseline_mesh() -> inf_mesh::MeshAsset {
    let (asset, _) = inf_dcc::to_mesh_asset(
        &inf_dcc::cube(PHASE23_PROP_SIZE_M),
        &inf_dcc::ExportOptions::default(),
    );
    asset
}

/// The workshop level: a sun, a static floor, and one dynamic prop that
/// **references the mesh under edit**.
///
/// The reference is what makes the live-update and edit-during-Simulate arms mean
/// anything: the entity draws the very asset the gate rewrites, while its
/// collider is authored and therefore *cannot* be reached by a mesh edit. That
/// pair — visibly coupled, physically independent — is the whole shape of the
/// asset-scoped ruling.
pub fn phase23_workshop_scene() -> SceneDoc {
    use inf_ecs::components::{
        BodyKind3D, Collider3D, ColliderShape3DKind, Light, LightKind, Material, MeshRef,
        Primitive, RigidBody3D, Transform,
    };
    use inf_ecs::math::{Color, Vec3d};

    let mut doc = SceneDoc::new();
    doc.set_title("Phase 23 Workshop");

    doc.create_with_guid(PHASE23_SUN_GUID, SpawnKind::Empty, "Sun", None);
    insert!(
        doc,
        PHASE23_SUN_GUID,
        Transform {
            translation: Vec3d::ZERO,
            rotation: Vec3d::new(-50.0, -30.0, 0.0),
            scale: Vec3d::ONE,
        }
    );
    insert!(
        doc,
        PHASE23_SUN_GUID,
        Light {
            kind: LightKind::Directional,
            color: Color::WHITE,
            intensity: 3.0,
            ..Default::default()
        }
    );

    doc.create_with_guid(PHASE23_GROUND_GUID, SpawnKind::Empty, "Floor", None);
    insert!(
        doc,
        PHASE23_GROUND_GUID,
        Transform {
            translation: Vec3d::new(0.0, -0.5, 0.0),
            rotation: Vec3d::ZERO,
            scale: Vec3d::ONE,
        }
    );
    insert!(
        doc,
        PHASE23_GROUND_GUID,
        RigidBody3D {
            kind: BodyKind3D::Static,
            ..Default::default()
        }
    );
    insert!(
        doc,
        PHASE23_GROUND_GUID,
        Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: Vec3d::new(20.0, 0.5, 20.0),
            ..Default::default()
        }
    );

    doc.create_with_guid(PHASE23_PROP_GUID, SpawnKind::Empty, "Prop", None);
    insert!(
        doc,
        PHASE23_PROP_GUID,
        Transform {
            translation: Vec3d::new(0.0, 4.0, 0.0),
            rotation: Vec3d::ZERO,
            scale: Vec3d::ONE,
        }
    );
    insert!(
        doc,
        PHASE23_PROP_GUID,
        MeshRef {
            primitive: Primitive::Cube,
            asset: Some(PHASE23_MESH_GUID),
        }
    );
    insert!(doc, PHASE23_PROP_GUID, Material::default());
    insert!(
        doc,
        PHASE23_PROP_GUID,
        RigidBody3D {
            kind: BodyKind3D::Dynamic,
            ..Default::default()
        }
    );
    // Authored density, not rapier's 1 kg/m³ placeholder — the P20.2/P22.4
    // finding, met once more. A prop that weighs 8 grams settles differently.
    insert!(
        doc,
        PHASE23_PROP_GUID,
        Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: Vec3d::splat(PHASE23_PROP_SIZE_M * 0.5),
            density: 500.0,
            ..Default::default()
        }
    );

    doc.world_mut().propagate();
    doc.mark_saved();
    doc
}

/// Faces every one of whose vertices lies on the plane `y`.
fn phase23_faces_at_y(mesh: &inf_dcc::Mesh, y: f64) -> Vec<inf_dcc::FaceId> {
    mesh.face_ids()
        .filter(|&f| {
            mesh.face_verts(f).is_some_and(|vs| {
                vs.iter()
                    .all(|&v| mesh.position(v).is_some_and(|p| (p.y - y).abs() < 1e-9))
            })
        })
        .collect()
}

/// The lowest-id **vertical** half-edge of the extruded wall: origin on the
/// original lid plane, destination directly above it on the new one.
///
/// Vertical is what makes the loop cut go anywhere: the ring steps across each
/// quad to the edge *opposite* the one it entered on, so entering a wall on its
/// vertical edge walks around all four walls and closes. Entering on a
/// horizontal one would step off the strip immediately.
fn phase23_wall_seed(mesh: &inf_dcc::Mesh, low: f64, high: f64) -> Option<inf_dcc::HalfId> {
    mesh.half_ids().find(|&h| {
        let (Some(a), Some(b)) = (
            mesh.origin(h).and_then(|v| mesh.position(v)),
            mesh.twin(h)
                .and_then(|t| mesh.origin(t))
                .and_then(|v| mesh.position(v)),
        ) else {
            return false;
        };
        (a.y - low).abs() < 1e-9
            && (b.y - high).abs() < 1e-9
            && (a.x - b.x).abs() < 1e-9
            && (a.z - b.z).abs() < 1e-9
    })
}

/// The rim: canonical edges on plane `y` that separate the **cap** (a face whose
/// every vertex is on that plane) from anything else. Four, in a cycle.
///
/// The recipe does **not** bevel all four, and that is the P23.6 ruling rather
/// than a taste: two bevels meeting at a corner each offset it into their far
/// face, both land in the same place on a right angle, and the exact weld in
/// `from_mesh_asset` fuses them into an edge used twice. `Op::BevelEdges` now
/// refuses it as [`inf_dcc::OpError::BevelCoincidentVertex`], so the recipe takes
/// the largest **disjoint** subset — see [`phase23_disjoint_pair`].
pub fn phase23_rim_edges(mesh: &inf_dcc::Mesh, y: f64) -> Vec<inf_dcc::HalfId> {
    let on_plane = |f: inf_dcc::FaceId| {
        mesh.face_verts(f).is_some_and(|vs| {
            vs.iter()
                .all(|&v| mesh.position(v).is_some_and(|p| (p.y - y).abs() < 1e-9))
        })
    };
    mesh.half_ids()
        .filter(|&h| inf_dcc::canonical_edge(mesh, h) == Some(h))
        .filter(|&h| {
            let Some(t) = mesh.twin(h) else { return false };
            let (Some(Some(a)), Some(Some(b))) = (mesh.face_of(h), mesh.face_of(t)) else {
                return false;
            };
            on_plane(a) != on_plane(b)
        })
        .collect()
}

/// The first two edges of `edges` that share **no endpoint** — the largest set a
/// bevel can take from a cycle, and the one the kernel will accept.
///
/// Lowest-id first and then the first later edge that does not touch it: a pure
/// function of the mesh, so the journal it produces is the same on any machine.
pub fn phase23_disjoint_pair(
    mesh: &inf_dcc::Mesh,
    edges: &[inf_dcc::HalfId],
) -> Option<[inf_dcc::HalfId; 2]> {
    let ends = |h: inf_dcc::HalfId| Some((mesh.origin(h)?, mesh.dest(h)?));
    for (i, &a) in edges.iter().enumerate() {
        let (a0, a1) = ends(a)?;
        for &b in &edges[i + 1..] {
            let (b0, b1) = ends(b)?;
            if b0 != a0 && b0 != a1 && b1 != a0 && b1 != a1 {
                return Some([a, b]);
            }
        }
    }
    None
}

/// **The modelling recipe the gate performs**, as one function so the gate and
/// its replay probe cannot drift apart.
///
/// Extrude the lid, loop-cut the walls it made, bevel two opposite edges of the
/// new rim — the three ops the phase's done-when sentence names, in the order the
/// topology allows: the loop cut needs the extrusion's **quad** walls (an edge
/// ring is only defined across quads), and beveling the rim first would replace
/// the very edges the ring walks.
///
/// **Two edges and not four**, because the kernel refuses the other two: bevels
/// that meet at a corner place two vertices at one position, which the exact weld
/// on the way back in fuses into a non-manifold edge. The gate asserts both — the
/// pair that works, and the four that are refused (see
/// [`inf_dcc::OpError::BevelCoincidentVertex`]).
///
/// Applied through [`inf_dcc::MeshSession::apply`], so what comes out is the
/// product's journal — the same `Vec<Op>` a session in the panel would hold, and
/// the thing arm (f) replays from cold.
pub fn phase23_model_prop(session: &mut inf_dcc::MeshSession) -> Result<(), String> {
    use inf_dcc::Op;
    let top_y = PHASE23_PROP_SIZE_M * 0.5 + PHASE23_EXTRUDE_M;
    phase23_extrude_and_cut(session)?;
    let rim = phase23_rim_edges(session.mesh(), top_y);
    if rim.len() != 4 {
        return Err(format!("expected a four-edge rim, found {}", rim.len()));
    }
    let pair = phase23_disjoint_pair(session.mesh(), &rim)
        .ok_or_else(|| "the rim has no two edges that do not meet".to_string())?;
    session
        .apply(Op::BevelEdges {
            edges: pair.to_vec(),
            amount: PHASE23_BEVEL_M,
            // The committed baseline is pinned byte-for-byte by the P23 gate, so
            // it stays on the single-segment construction Wave D proved to be
            // bit-identical to P23.4's. A segment count here would move the pin.
            segments: 1,
        })
        .map_err(|e| format!("bevel: {e}"))?;
    Ok(())
}

/// The recipe's first two ops — extrude the lid, loop-cut the walls it made.
///
/// Split out so the gate's refusal arm can reach the state where the rim exists
/// *before* anything has been beveled, without restating either op. A gate that
/// rebuilt the first half by hand would be gating a copy (P23.4's LAW).
pub fn phase23_extrude_and_cut(session: &mut inf_dcc::MeshSession) -> Result<(), String> {
    use inf_dcc::Op;
    let lid_y = PHASE23_PROP_SIZE_M * 0.5;
    let top_y = lid_y + PHASE23_EXTRUDE_M;

    let lid = phase23_faces_at_y(session.mesh(), lid_y);
    if lid.len() != 2 {
        return Err(format!(
            "expected the cube's lid to arrive as two triangles, found {}",
            lid.len()
        ));
    }
    session
        .apply(Op::ExtrudeFaces {
            faces: lid,
            distance: PHASE23_EXTRUDE_M,
        })
        .map_err(|e| format!("extrude: {e}"))?;

    let seed = phase23_wall_seed(session.mesh(), lid_y, top_y)
        .ok_or_else(|| "the extrude left no vertical wall edge to ring".to_string())?;
    session
        .apply(Op::LoopCut {
            half: seed,
            cuts: 1,
        })
        .map_err(|e| format!("loop cut: {e}"))?;

    Ok(())
}

/// The seam set the gate cuts before unwrapping: every canonical edge whose two
/// faces are **not coplanar**.
///
/// The gate picks its own seams because the product does not: seam marking is a
/// 3D-view act and auto-seaming is a named Phase 23 remainder. What this set buys
/// is a prop cut into *flat* charts, which is what makes "fold count zero" a
/// statement about the solver rather than about how curved the prop happens to
/// be.
pub fn phase23_seam_edges(mesh: &inf_dcc::Mesh) -> Vec<inf_dcc::HalfId> {
    mesh.half_ids()
        .filter(|&h| inf_dcc::canonical_edge(mesh, h) == Some(h))
        .filter(|&h| {
            let Some(t) = mesh.twin(h) else { return false };
            let (Some(Some(a)), Some(Some(b))) = (mesh.face_of(h), mesh.face_of(t)) else {
                return false;
            };
            inf_dcc::face_normal(mesh, a)
                .zip(inf_dcc::face_normal(mesh, b))
                .is_none_or(|(na, nb)| na.dot(nb) < 0.999)
        })
        .collect()
}

/// **What the unwrap must reach on this prop.**
///
/// Machine epsilon on every chart, not a tolerance. Before the P23.6 bevel
/// ruling the recipe beveled all four rim edges and left one chart stalled at
/// 1.6e-2 — which was the *same* defect three symptoms deep, since the n-gons
/// whose fan triangulation ill-conditioned that solve were the ones the corner
/// coincidence produced. Refusing the bevel that cannot be saved removed the
/// stalled chart with it, so the bound is no longer a bound: it is zero
/// tolerance, and a chart that stalls again fails here.
pub const PHASE23_CONVERGED: f64 = 1.0e-12;

/// Cut the seams and unwrap — the gate's arm (b), through the product path
/// (`SetEdgeSeam` ops, then the solver's own `Op::Unwrap`).
pub fn phase23_unwrap_prop(
    session: &mut inf_dcc::MeshSession,
) -> Result<inf_dcc::UnwrapReport, String> {
    for half in phase23_seam_edges(session.mesh()) {
        session
            .apply(inf_dcc::Op::SetEdgeSeam { half, seam: true })
            .map_err(|e| format!("seam: {e}"))?;
    }
    let unwrapped = inf_dcc::unwrap(session.mesh()).map_err(|e| format!("unwrap: {e}"))?;
    session
        .apply(unwrapped.op)
        .map_err(|e| format!("apply unwrap: {e}"))?;
    Ok(unwrapped.report)
}

/// `samples/phase23-workshop`, resolved from this crate's manifest dir.
pub fn phase23_workshop_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../samples/phase23-workshop")
}

/// Write every committed Phase 23 gate file.
pub fn write_phase23_workshop() -> Result<(), String> {
    let dir = phase23_workshop_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("mkdir: {e}"))?;

    crate::scene::serialize::save(
        &phase23_workshop_scene(),
        &dir.join("Phase23Workshop.inf_lvl"),
        Some(PHASE23_LEVEL_GUID),
    )?;

    let bytes =
        inf_asset::encode(&phase23_baseline_mesh()).map_err(|e| format!("encode mesh: {e}"))?;
    let path = dir.join("Prop.inf_mesh");
    std::fs::write(&path, &bytes).map_err(|e| format!("write mesh: {e}"))?;
    inf_asset::AssetSidecar::new(
        inf_asset::AssetId(PHASE23_MESH_GUID),
        inf_asset::AssetKind::Mesh,
        inf_asset::ContentHash::of(&bytes),
    )
    .save(&path)
    .map_err(|e| format!("write mesh sidecar: {e}"))?;

    std::fs::write(dir.join("README.md"), PHASE23_WORKSHOP_README)
        .map_err(|e| format!("write readme: {e}"))?;
    Ok(())
}

const PHASE23_WORKSHOP_README: &str = "# Phase 23 Workshop (embedded-DCC gate scene)\n\n\
Generated by `inf_editor_core::samples::phase23_workshop_scene` -- the P23.6 gate\n\
scene, and the smallest sample in the tree on purpose. Every other phase gate ships a\n\
level whose *content* is the claim; this phase's claim is about a **pipeline** (model,\n\
unwrap, save, live-update, undo, deterministic replay), so the level's job is only to\n\
reference a mesh and keep simulating while that mesh is edited underneath it.\n\n\
- `Phase23Workshop.inf_lvl` -- a sun, a static floor, and one dynamic prop whose\n\
  `MeshRef.asset` names the mesh below. **Visibly coupled, physically independent**:\n\
  the prop draws the asset the gate rewrites, and its collider is authored, so a mesh\n\
  edit cannot reach the simulation. That pair is the whole shape of the P23.1\n\
  asset-scoped ruling.\n\
- `Prop.inf_mesh` -- the **import baseline**: a 2 m cube written through the DCC's own\n\
  `to_mesh_asset`. It is committed rather than generated at test time because the gate's\n\
  undo arm compares a fully-undone journal against it *byte for byte*, and a hand-built\n\
  asset would differ in its corner interning order alone (part of the format, per the\n\
  P23.3 law).\n\n\
## The gate (`runtime/inf-player/tests/phase23_gate.rs`)\n\n\
Nine arms, all driven through the product op path rather than hand-built meshes:\n\
(a) model a prop -- extrude, then loop cut, then bevel (that order: an edge ring is only\n\
defined across quads, so the cut needs the extrusion walls and a bevel first would replace\n\
the very edges the ring walks). The bevel takes TWO OPPOSITE rim edges and not all four:\n\
bevels that MEET at a corner each offset it into their far face, on a right angle both land\n\
at one position, and the reader's exact weld fuses them into a non-manifold edge -- so\n\
`Op::BevelEdges` refuses that, and the gate asserts both the refusal and the pair that\n\
works. Then the journal is replayed twice, bit-identically;\n\
(b) seams + unwrap: every corner inside the unit square, ZERO folds, and the convergence\n\
field at machine epsilon on EVERY chart; (c) save as a standard asset\n\
(`.inf_mesh` decodes, the derived `.inf_vmesh` decodes, the sidecar hash matches the\n\
bytes); (d) live update -- a store that resolved the mesh BEFORE the edit re-keys after\n\
it, and a pack cooked after the save carries the new bytes; (e) undo the whole journal\n\
back to the baseline byte-for-byte, redo back to the edited state; (f) determinism, the\n\
journal replayed in a fresh SUBPROCESS; (g) the edit-during-Simulate headline, at gate\n\
level; (h) budgets (`LOAD_BUDGET_MS` for opens and saves); (i) cook advisories.\n\n\
## What it is NOT\n\n\
There is no blueprint here, and no scripted actor. The Simulate arm needs a world that\n\
*moves* while a save lands in the middle of it, and gravity is the least interesting\n\
thing that does -- which is what you want in a trace whose whole job is to be identical\n\
to a control run.\n\n\
The prop is small, and that is honest rather than lazy: a hand-modelled prop is a few\n\
dozen triangles, which is **below the cook's `[vgeom] min_triangles`**. The gate asserts\n\
the advisory the cook draws about exactly that, because it is the truth about this whole\n\
class of asset -- see the Phase 23 completion block in `docs/ROADMAP.md`.\n\n\
Regenerate with `INF_BLESS_SAMPLES=1 cargo test -p inf-editor-core samples`.\n";

// ── Phase 29 gate scene: `samples/phase29-locomotion` (P29.6) ───────────────
//
// **The obstacle course**, and it is an obstacle course rather than a level
// because of what the catalogue amendment asks of it: *"P29.6's course must
// force every catalogue mode in its one deterministic replay, so the (pose,
// mode) trace certifies the catalogue and not a subset."* Every block below
// exists because one mode cannot be reached without it, and the stations are in
// the order a scripted character meets them:
//
// | z | station | the mode it forces |
// |---|---|---|
// | 0–10 | open floor | `Grounded` at all three gaits |
// | 11–17 | a roof at 1.4 m | `Crouch` (and the standing-up refusal under it) |
// | 18–44 | open floor, 26 m of it | `Prone`, `Slide`, `Roll`, `Dive`, `FallFree` |
// | 46–52 | four 20 cm risers + a landing | autostep, and `FallControlled` off its edge |
// | 64–70 | a 1 m ledge | `Mantle`, LOW class |
// | 70–76 | a 3 m ledge | `Mantle`, HIGH class |
// | 76–82 | a 5 m ledge | the drop that makes a `Ragdoll` landing |
// | 100–120 | a 3 m pool | `SwimSurface` and `SwimUnder` |
//
// The open stretch is 26 m because a **slide** is entered from a sprint at
// 4 m/s and a sprint accelerates from a standstill: a station with two metres of
// runway in it certifies nothing, which is how the first draft of this course
// reported a catalogue with no slide in it.
//
// `FallFree` is a jump and needs no geometry; `Driving` and `Flying` are typed
// refusals until P29.7 and the gate asserts them AS refusals.
//
// **The character is the wizard's own output**, generated here with fixed GUIDs
// rather than through `build_character` (which mints fresh ones): the same
// template rig, the same locomotion set, the same `inf_anim::derive_clip` pass
// and the same `inf_anim::propose` machine. That is what makes this sample the
// first committed **derived** content in the repository — P29.4's and P29.5's
// "no committed content is derived" remainder, closed.

/// The committed level's GUID.
const PHASE29_LEVEL_GUID: Uuid = Uuid::from_u128(0x8409_0000);
const PHASE29_HERO_GUID: Uuid = Uuid::from_u128(0x8409_0001);
const PHASE29_SUN_GUID: Uuid = Uuid::from_u128(0x8409_0002);
const PHASE29_WATER_GUID: Uuid = Uuid::from_u128(0x8409_0003);
/// The asset GUIDs, stable so the committed level's references resolve.
const PHASE29_SKELETON_GUID: Uuid = Uuid::from_u128(0x8409_00a0);
const PHASE29_MESH_GUID: Uuid = Uuid::from_u128(0x8409_00a1);
const PHASE29_IDLE_GUID: Uuid = Uuid::from_u128(0x8409_00a2);
const PHASE29_WALK_GUID: Uuid = Uuid::from_u128(0x8409_00a3);
const PHASE29_RUN_GUID: Uuid = Uuid::from_u128(0x8409_00a4);
const PHASE29_SM_GUID: Uuid = Uuid::from_u128(0x8409_00a5);
const PHASE29_ACTOR_GUID: Uuid = Uuid::from_u128(0x8409_00a6);
/// The course's static blocks start here and run consecutively.
const PHASE29_BLOCK_BASE: u128 = 0x8409_0100;
/// The committed car's chassis (P29.7), and its four wheels after it.
const PHASE29_CAR_GUID: Uuid = Uuid::from_u128(0x8409_0004);
const PHASE29_WHEEL_BASE: u128 = 0x8409_0010;

/// The character's own height, metres — the number every capsule dimension and
/// the camera's pivot are derived from.
pub const PHASE29_HEIGHT_M: f64 = 1.8;

/// Where the character's FEET start, in world space.
pub fn phase29_start() -> DVec3 {
    DVec3::new(0.0, 0.0, -4.0)
}

/// One static block of the course: a centre, a half-extent and a name.
///
/// Boxes only, and axis-aligned. A slope is not needed for any mode the
/// catalogue names — a slide is entered from *sprint plus crouch* and not from
/// a gradient (`step_one`'s own rule) — and every box being axis-aligned is what
/// makes the course's geometry checkable by arithmetic in the gate rather than
/// by a physics query.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Phase29Block {
    pub name: &'static str,
    pub centre: DVec3,
    pub half: DVec3,
}

impl Phase29Block {
    /// The block's top surface, metres.
    pub fn top(&self) -> f64 {
        self.centre.y + self.half.y
    }
}

/// **The course**, in the order a character meets it.
pub fn phase29_blocks() -> Vec<Phase29Block> {
    let b = |name, centre: DVec3, half: DVec3| Phase29Block { name, centre, half };
    let mut out = vec![
        // The floor, from behind the start to the lip of the pool.
        b(
            "floor",
            DVec3::new(0.0, -0.5, 45.0),
            DVec3::new(8.0, 0.5, 55.0),
        ),
        // A roof low enough to refuse a standing capsule (1.8 m) and clear a
        // crouched one (1.2 m).
        b(
            "low roof",
            DVec3::new(0.0, 1.6, 14.0),
            DVec3::new(8.0, 0.2, 3.0),
        ),
    ];
    // Four 20 cm risers — the flight P29.3's autostep arm is built on, here as
    // committed content rather than as a fixture.
    for i in 0..4 {
        let top = 0.2 * (i + 1) as f64;
        out.push(b(
            "riser",
            DVec3::new(0.0, top * 0.5, 46.0 + 0.5 * i as f64),
            DVec3::new(8.0, top * 0.5, 0.25),
        ));
    }
    out.push(b(
        "landing",
        DVec3::new(0.0, 0.4, 50.0),
        DVec3::new(8.0, 0.4, 2.0),
    ));
    // The three ledges: 1 m (low mantle), 3 m (high mantle), 5 m (the drop that
    // classifies as a ragdoll).
    out.push(b(
        "ledge 1 m",
        DVec3::new(0.0, 0.5, 67.0),
        DVec3::new(8.0, 0.5, 3.0),
    ));
    out.push(b(
        "ledge 3 m",
        DVec3::new(0.0, 1.5, 73.0),
        DVec3::new(8.0, 1.5, 3.0),
    ));
    out.push(b(
        "ledge 5 m",
        DVec3::new(0.0, 2.5, 79.0),
        DVec3::new(8.0, 2.5, 3.0),
    ));
    // The pool floor, three metres under the water's surface.
    out.push(b(
        "pool floor",
        DVec3::new(0.0, -3.5, 110.0),
        DVec3::new(8.0, 0.5, 10.0),
    ));
    // **The beach** (P29.7): a staircase out of the pool.
    //
    // Before it the course simply ended in the water — the P29.6 driver's last
    // stage was a catch-all that swam forward for ever, so nothing noticed that
    // the pool's far wall was a vertical three metres a swimmer can never climb.
    //
    // The shape is decided by one number: **a surface swimmer floats with its
    // feet at about −1.07 m** (the capsule settles at −0.2 and is 0.87 from
    // centre to sole), and it does **not** autostep, because rapier's autostep
    // needs a grounded character. So the first thing it can stand on has to be
    // below that, and the water has to end before anything above it does. Two
    // wrong versions were measured first: a beach at −1.2/−0.8/−0.4 left the
    // character pressed against a shelf face at z = 115.14, and one starting at
    // −0.9 left it against the first wall at z = 113.21.
    //
    // So the pool stays flat, the water body ends at z = 120, and the bank the
    // swimmer drops onto is at −1.35 — a 28 cm fall out of the water — with
    // seven **19 cm** steps up to road level after it. Not 40 cm: the 45 cm
    // `step_height_m` default is the mover's ceiling and a 40 cm rise stopped
    // the character dead at z = 126.71, which is the third measurement this
    // beach cost. The course's own stair station is 20 cm a tread and works, so
    // the beach is that stair, seven times.
    out.push(b(
        "bank low",
        DVec3::new(0.0, -1.85, 122.0),
        DVec3::new(8.0, 0.5, 2.0),
    ));
    for i in 0..7 {
        let top = -1.35 + 1.35 * (i + 1) as f64 / 7.0;
        out.push(b(
            "bank step",
            DVec3::new(0.0, top - 0.5, 124.5 + i as f64),
            DVec3::new(8.0, 0.5, 0.5),
        ));
    }
    // ── P29.7: the road, the hump and the apron ──
    //
    // Wide enough (24 m) that a steered car has somewhere to go, and long
    // enough that the drive segment is a drive rather than a lurch: the rig
    // accelerates at about 6.7 m/s², so the sixty metres before the hump is
    // where it reaches the speed the jump needs.
    out.push(b(
        "road",
        DVec3::new(0.0, -0.5, 201.0),
        DVec3::new(16.0, 0.5, 70.0),
    ));
    // **Kerbs.** A 1 m wall down each side, because a car is not a character:
    // the first drive segment steered off the edge of a 24 m road at 14 m/s and
    // spent three thousand steps falling to y = −3 000, which is a replay whose
    // last half certifies nothing. A wall is what a road has.
    for x in [-16.5_f64, 16.5] {
        out.push(b(
            "kerb",
            DVec3::new(x, 0.5, 201.0),
            DVec3::new(0.5, 1.0, 70.0),
        ));
    }
    // **The jump.** A hump rather than a ramp or a gap: a ramp is a rotated box
    // whose lip never quite meets the road, and a gap is a failure mode (a car
    // that misses it is at the bottom of a hole for the rest of the replay). A
    // 20 cm hump the wheels roll over launches the whole rig on its own
    // suspension at speed, which is what an airborne vehicle looks like anyway.
    out.push(b(
        "hump",
        DVec3::new(0.0, 0.1, 200.0),
        DVec3::new(16.0, 0.1, 1.5),
    ));
    // The apron the flight segment lands on. Generous in **both** axes: a flying
    // character banks into its turns and therefore travels sideways, and a first
    // draft 40 m wide put it off the edge and into a fall that ran to the end of
    // the replay. It overlaps the road so a character that lands short is still
    // on geometry.
    out.push(b(
        "apron",
        DVec3::new(0.0, -0.5, 295.0),
        DVec3::new(30.0, 0.5, 30.0),
    ));
    out
}

/// Where the committed car's **chassis** starts, world metres.
///
/// On the road, past the far bank, and far enough along it that the character
/// walks to it rather than starting inside it. The `y` is the placement an
/// author makes: wheels touching the road with the suspension fully extended,
/// which is `wheel offset + wheel radius` above the surface.
pub fn phase29_car_start() -> DVec3 {
    DVec3::new(0.0, PHASE29_WHEEL_Y.abs() + PHASE29_WHEEL_RADIUS_M, 138.0)
}

/// The committed car's chassis half-extents, metres — 4 × 1 × 2 m.
pub const PHASE29_CAR_HALF: inf_ecs::math::Vec3d = inf_ecs::math::Vec3d::new(2.0, 0.5, 1.0);
/// Its density, kg/m³. A hollow shell, not its material: 150 over 8 m³ is
/// 1 200 kg, which is a small road car. (`Collider3D::density`'s own note.)
pub const PHASE29_CAR_DENSITY: f64 = 150.0;
/// The wheel centres' height in the chassis frame, metres — at full extension.
///
/// −0.75 rather than −0.5 for **ground clearance**: the chassis is a metre tall,
/// so its underside sits `0.75 + 0.35 − 0.5 − settle` above the road, which is
/// 45 cm. At −0.5 the underside was at 20 cm, exactly the height of the hump,
/// and the car drove up onto its own belly and stopped there for four thousand
/// steps. A wheel that pokes out below the bodywork is also what a car looks
/// like.
pub const PHASE29_WHEEL_Y: f64 = -0.75;
/// The wheel radius, metres.
pub const PHASE29_WHEEL_RADIUS_M: f64 = 0.35;

/// The four wheel mounts, in the chassis frame — front pair first (`+Z` is
/// forward, and `WheelMount::steered` is a sign test on exactly this).
pub fn phase29_wheel_mounts() -> [inf_ecs::math::Vec3d; 4] {
    [
        inf_ecs::math::Vec3d::new(-0.9, PHASE29_WHEEL_Y, 1.4),
        inf_ecs::math::Vec3d::new(0.9, PHASE29_WHEEL_Y, 1.4),
        inf_ecs::math::Vec3d::new(-0.9, PHASE29_WHEEL_Y, -1.4),
        inf_ecs::math::Vec3d::new(0.9, PHASE29_WHEEL_Y, -1.4),
    ]
}

/// The pool's surface elevation and half-extent (XZ) — the P20 water body the
/// swim modes need.
pub fn phase29_pool() -> (f64, glam::DVec2, DVec3) {
    (
        0.0,
        glam::DVec2::new(8.0, 10.0),
        DVec3::new(0.0, 0.0, 110.0),
    )
}

/// The wizard's spec for this course's character — the DEFAULT biped, so the
/// sample demonstrates what an author gets from the wizard rather than a shape
/// tuned to make the gate pass.
pub fn phase29_spec() -> (
    inf_anim::BodyPlan,
    inf_anim::BodyParams,
    inf_anim::locomotion::GaitParams,
) {
    (
        // **Pinned to the canonical-vocabulary biped** (SK1a). This sample's
        // committed bytes are its whole point — `.inf_skel`, three `.inf_anim`
        // clips index-bound to it, an `.inf_sm`, a controller counting footstep
        // markers by name and a `.inf_lvl` — and `BodyPlan::Biped` became the
        // 161-bone mannequin. Following that here would re-bless every one of
        // them for a wave whose subject is the substrate, not this course. The
        // mannequin gets its own end-to-end arm in `phase24_wizard`.
        inf_anim::BodyPlan::BipedCanonical,
        inf_anim::BodyParams {
            height_m: PHASE29_HEIGHT_M,
            ..inf_anim::BodyParams::default()
        },
        inf_anim::locomotion::GaitParams::default(),
    )
}

/// The character's rig, from the template generator.
pub fn phase29_skeleton() -> inf_anim::SkeletonAsset {
    let (plan, params, _) = phase29_spec();
    inf_anim::build_template(plan, &params).expect("the default biped builds")
}

/// The gait ladder this creature's clips are derived and proposed against.
///
/// **Its own, not ALS's.** The generator's walk is around 0.65 m/s, which on the
/// ported 1.65 / 3.75 / 6.5 ladder tiers as an *idle* (the P29.5 reading), and a
/// proposal over that clusters the whole set into one state. The wizard passes
/// the generator's own numbers for exactly this reason and so does the sample.
pub fn phase29_ladder(set: &inf_anim::LocomotionSet) -> [f32; 3] {
    [
        set.walk_speed_m_s as f32,
        set.run_speed_m_s as f32,
        (set.run_speed_m_s * 1.75) as f32,
    ]
}

/// The three **derived** clips, and the ladder they were measured against.
///
/// This is the sample's headline as content: `samples/` has never carried a clip
/// with a root-motion track, a distance track, foot-plant sync markers or a
/// `W_Gait` channel on it, which is the "no committed content is derived"
/// remainder P29.4 and P29.5 both recorded.
pub fn phase29_clips() -> (inf_anim::LocomotionSet, [f32; 3]) {
    let (plan, _, gait) = phase29_spec();
    let rig = phase29_skeleton();
    let mut set = inf_anim::locomotion::build_locomotion(plan, &rig, &gait)
        .expect("the default gait generates");
    let ladder = phase29_ladder(&set);
    let opts = inf_anim::DeriveOptions {
        gait_speeds_mps: ladder,
        ..inf_anim::DeriveOptions::default()
    };
    for clip in [&mut set.idle, &mut set.walk, &mut set.run] {
        let (derived, _) =
            inf_anim::derive_clip(clip, &rig, &opts).expect("a generated cycle measures");
        *clip = derived;
    }
    (set, ladder)
}

/// The **proposed** machine — `inf_anim::propose` over the derived clips, which
/// is what the wizard writes and therefore what the sample commits.
pub fn phase29_machine() -> inf_anim::StateMachine {
    let (set, ladder) = phase29_clips();
    let facts = vec![
        inf_anim::propose::facts_of("idle", *PHASE29_IDLE_GUID.as_bytes(), &set.idle, ladder),
        inf_anim::propose::facts_of("walk", *PHASE29_WALK_GUID.as_bytes(), &set.walk, ladder),
        inf_anim::propose::facts_of("run", *PHASE29_RUN_GUID.as_bytes(), &set.run, ladder),
    ];
    inf_anim::propose::propose_machine(
        &facts,
        &inf_anim::propose::ProposalOptions {
            gait_speeds_mps: ladder,
            ..Default::default()
        },
    )
    .expect("three derived cycles propose")
    .machine
}

/// The character's controller — the wizard's own, with this sample's id.
pub fn phase29_controller() -> BlueprintClass {
    let footsteps: Vec<String> = inf_anim::DerivedNames::of_skeleton(&phase29_skeleton())
        .event_markers
        .into_iter()
        .collect();
    // The state the controller times is a function of the machine, not a guess
    // (P29.6 audit, A16) — for this creature's own ladder that is `run`.
    let motion = crate::character::motion_state_of(&phase29_machine());
    crate::character::controller_class_for("Hero", &footsteps, &motion)
}

/// The blocky mannequin body, one box per bone.
pub fn phase29_body() -> inf_mesh::MeshAsset {
    crate::character::block_body_mesh(&phase29_skeleton())
}

/// The committed course scene.
pub fn phase29_locomotion_scene() -> SceneDoc {
    use inf_ecs::components::{
        ActorClass, AnimStateMachine, BodyKind3D, CharacterController3D, CharacterMovement,
        Collider3D, ColliderShape3DKind, Light, LightKind, RigidBody3D, SkeletalMesh, WaterBody,
    };

    let mut doc = SceneDoc::new();
    doc.set_title("Phase 29 Locomotion Course");
    // **The 3D solver's gravity comes from `gravity_2d.y`**, which is the
    // convention every 3D sample in this tree follows (`RuntimeSim::new` builds
    // its 3D bridge with `DVec3::new(0, gravity.y, 0)`). It matters here because
    // the ragdoll's limbs are the first DYNAMIC bodies this course has: a
    // character carries its own `CharacterMovement::gravity_mps2` and falls
    // whatever the world says, so a level with no gravity looks perfectly fine
    // right up until something is let go of. The `.inf_lvl`'s own `gravity_3d`
    // field is authored, serialized and read by NOTHING — see the P29.6 ledger.
    doc.set_settings(crate::scene::serialize::LevelSettings {
        gravity_2d: Vec2d::new(0.0, -9.81),
        ..crate::scene::serialize::LevelSettings::default()
    });

    for (i, block) in phase29_blocks().into_iter().enumerate() {
        let guid = Uuid::from_u128(PHASE29_BLOCK_BASE + i as u128);
        doc.create_with_guid(guid, SpawnKind::Cube, block.name, None);
        insert!(doc, guid, Transform::from_translation(block.centre));
        insert!(
            doc,
            guid,
            RigidBody3D {
                kind: BodyKind3D::Static,
                ..Default::default()
            }
        );
        insert!(
            doc,
            guid,
            Collider3D {
                shape_kind: ColliderShape3DKind::Box,
                half_extents: inf_ecs::math::Vec3d::new(block.half.x, block.half.y, block.half.z),
                ..Default::default()
            }
        );
    }

    // ── the water ──
    let (level, half, centre) = phase29_pool();
    doc.create_with_guid(PHASE29_WATER_GUID, SpawnKind::Empty, "Pool", None);
    insert!(doc, PHASE29_WATER_GUID, Transform::from_translation(centre));
    insert!(
        doc,
        PHASE29_WATER_GUID,
        WaterBody::lake(level, Vec2d::new(half.x, half.y))
    );

    // ── the character, as the wizard makes one ──
    //
    // A capsule derived from the creature's own height, a kinematic body, the
    // movement component with the catalogue defaults, the proposed machine, the
    // controller — and the transform at the capsule's CENTRE with the feet on
    // the floor, which is P29.6's character-space ruling applied to committed
    // content.
    let radius = (PHASE29_HEIGHT_M * 0.15).clamp(0.1, 0.5);
    let half_h = (PHASE29_HEIGHT_M * 0.5 - radius).max(0.05);
    doc.create_with_guid(PHASE29_HERO_GUID, SpawnKind::Empty, "Hero", None);
    let feet = phase29_start();
    insert!(
        doc,
        PHASE29_HERO_GUID,
        Transform::from_translation(DVec3::new(feet.x, feet.y + half_h + radius, feet.z))
    );
    insert!(
        doc,
        PHASE29_HERO_GUID,
        SkeletalMesh {
            mesh: Some(PHASE29_MESH_GUID),
            skeleton: Some(PHASE29_SKELETON_GUID),
        }
    );
    insert!(
        doc,
        PHASE29_HERO_GUID,
        AnimStateMachine {
            sm: Some(PHASE29_SM_GUID),
            ..Default::default()
        }
    );
    insert!(
        doc,
        PHASE29_HERO_GUID,
        RigidBody3D {
            kind: BodyKind3D::Kinematic,
            ..Default::default()
        }
    );
    insert!(
        doc,
        PHASE29_HERO_GUID,
        Collider3D {
            shape_kind: ColliderShape3DKind::Capsule,
            half_extents: inf_ecs::math::Vec3d::new(radius, half_h, radius),
            radius,
            ..Default::default()
        }
    );
    insert!(doc, PHASE29_HERO_GUID, CharacterController3D::default());
    insert!(
        doc,
        PHASE29_HERO_GUID,
        CharacterMovement {
            player_controlled: true,
            stand_half_height_m: half_h,
            crouch_half_height_m: (half_h * 0.5).max(0.05),
            prone_half_height_m: (radius * 0.6).max(0.03),
            ..Default::default()
        }
    );
    insert!(doc, PHASE29_HERO_GUID, ActorClass(PHASE29_ACTOR_GUID));

    // ── the car (P29.7) ──
    //
    // **Nothing here is a vehicle field**, because there is no such field: a
    // chassis is a dynamic body with a collider and a wheel is a direct child
    // carrying a sphere `Collider3D` with `sensor: true` and no body of its own.
    // `inf_ecs::vehicle::wheel_of` is the one recogniser, and this generator and
    // the physics bridge both read it — so the level and the simulation cannot
    // disagree about what a wheel is.
    //
    // The density is authored (150 kg/m³ over 8 m³ = 1 200 kg): rapier's 1.0
    // placeholder would make this car weigh eight kilograms, which is the
    // fifth catch of that law in this repository.
    doc.create_with_guid(PHASE29_CAR_GUID, SpawnKind::Empty, "Car", None);
    insert!(
        doc,
        PHASE29_CAR_GUID,
        Transform::from_translation(phase29_car_start())
    );
    insert!(
        doc,
        PHASE29_CAR_GUID,
        inf_ecs::components::MeshRef {
            primitive: inf_ecs::components::Primitive::Cube,
            asset: None,
        }
    );
    insert!(
        doc,
        PHASE29_CAR_GUID,
        inf_ecs::components::Material::default()
    );
    insert!(
        doc,
        PHASE29_CAR_GUID,
        RigidBody3D {
            kind: BodyKind3D::Dynamic,
            // A car does not spin on its own axis for want of damping; the
            // suspension supplies the rest of the resistance.
            angular_damping: 0.5,
            ..Default::default()
        }
    );
    insert!(
        doc,
        PHASE29_CAR_GUID,
        Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: PHASE29_CAR_HALF,
            density: PHASE29_CAR_DENSITY,
            friction: 0.5,
            ..Default::default()
        }
    );
    for (i, mount) in phase29_wheel_mounts().into_iter().enumerate() {
        let guid = Uuid::from_u128(PHASE29_WHEEL_BASE + i as u128);
        doc.create_with_guid(guid, SpawnKind::Empty, "Wheel", Some(PHASE29_CAR_GUID));
        insert!(doc, guid, Transform::from_translation(mount.to_dvec3()));
        insert!(
            doc,
            guid,
            inf_ecs::components::MeshRef {
                primitive: inf_ecs::components::Primitive::Sphere,
                asset: None,
            }
        );
        insert!(doc, guid, inf_ecs::components::Material::default());
        insert!(
            doc,
            guid,
            Collider3D {
                shape_kind: ColliderShape3DKind::Sphere,
                radius: PHASE29_WHEEL_RADIUS_M,
                sensor: true,
                ..Default::default()
            }
        );
    }

    // ── a sun, so the course is visible in PIE ──
    doc.create_with_guid(PHASE29_SUN_GUID, SpawnKind::Empty, "Sun", None);
    insert!(
        doc,
        PHASE29_SUN_GUID,
        Transform {
            translation: inf_ecs::math::Vec3d::new(0.0, 40.0, 0.0),
            rotation: inf_ecs::math::Vec3d::new(-50.0, -30.0, 0.0),
            scale: inf_ecs::math::Vec3d::new(1.0, 1.0, 1.0),
        }
    );
    insert!(
        doc,
        PHASE29_SUN_GUID,
        Light {
            kind: LightKind::Directional,
            color: Color::new(1.0, 0.97, 0.9, 1.0),
            intensity: 2.5,
            ..Default::default()
        }
    );

    doc.world_mut().propagate();
    doc.mark_saved();
    doc
}

/// The `(guid, class)` actor list for a headless Simulate of the course.
pub fn phase29_actors() -> Vec<(Uuid, BlueprintClass)> {
    vec![(PHASE29_HERO_GUID, phase29_controller())]
}

/// The character's GUID — the gate's subject, and the camera's.
pub fn phase29_hero() -> Uuid {
    PHASE29_HERO_GUID
}

/// The committed car's chassis GUID (P29.7) — the drive segment's subject.
pub fn phase29_car() -> Uuid {
    PHASE29_CAR_GUID
}

/// Its four wheel GUIDs, in the order [`phase29_wheel_mounts`] lists them.
pub fn phase29_wheels() -> [Uuid; 4] {
    [
        Uuid::from_u128(PHASE29_WHEEL_BASE),
        Uuid::from_u128(PHASE29_WHEEL_BASE + 1),
        Uuid::from_u128(PHASE29_WHEEL_BASE + 2),
        Uuid::from_u128(PHASE29_WHEEL_BASE + 3),
    ]
}

/// The asset GUIDs, so a gate can resolve the committed files without scanning.
pub fn phase29_asset_guids() -> Phase29Assets {
    Phase29Assets {
        skeleton: PHASE29_SKELETON_GUID,
        mesh: PHASE29_MESH_GUID,
        idle: PHASE29_IDLE_GUID,
        walk: PHASE29_WALK_GUID,
        run: PHASE29_RUN_GUID,
        machine: PHASE29_SM_GUID,
        actor: PHASE29_ACTOR_GUID,
        level: PHASE29_LEVEL_GUID,
    }
}

/// See [`phase29_asset_guids`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Phase29Assets {
    pub skeleton: Uuid,
    pub mesh: Uuid,
    pub idle: Uuid,
    pub walk: Uuid,
    pub run: Uuid,
    pub machine: Uuid,
    pub actor: Uuid,
    pub level: Uuid,
}

/// The repo-root `samples/phase29-locomotion/` directory.
pub fn phase29_locomotion_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../samples/phase29-locomotion")
}

/// Write the committed course, its character and the three text files.
pub fn write_phase29_locomotion() -> Result<(), String> {
    let dir = phase29_locomotion_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("mkdir: {e}"))?;

    crate::scene::serialize::save(
        &phase29_locomotion_scene(),
        &dir.join("Phase29Locomotion.inf_lvl"),
        Some(PHASE29_LEVEL_GUID),
    )?;

    let skel = phase29_skeleton();
    let put = |name: &str,
               guid: Uuid,
               kind: inf_asset::AssetKind,
               bytes: Vec<u8>|
     -> Result<(), String> {
        let path = dir.join(name);
        std::fs::write(&path, &bytes).map_err(|e| format!("write {name}: {e}"))?;
        inf_asset::AssetSidecar::new(
            inf_asset::AssetId(guid),
            kind,
            inf_asset::ContentHash::of(&bytes),
        )
        .save(&path)
        .map_err(|e| format!("write {name} sidecar: {e}"))
    };

    put(
        "Hero.inf_skel",
        PHASE29_SKELETON_GUID,
        inf_asset::AssetKind::Skeleton,
        inf_asset::encode(&skel).map_err(|e| format!("encode skeleton: {e}"))?,
    )?;
    put(
        "Hero Body.inf_mesh",
        PHASE29_MESH_GUID,
        inf_asset::AssetKind::Mesh,
        inf_asset::encode(&phase29_body()).map_err(|e| format!("encode body: {e}"))?,
    )?;

    let (set, _) = phase29_clips();
    let skel_bytes = *PHASE29_SKELETON_GUID.as_bytes();
    for (name, guid, clip) in [
        ("Hero Idle.inf_anim", PHASE29_IDLE_GUID, &set.idle),
        ("Hero Walk.inf_anim", PHASE29_WALK_GUID, &set.walk),
        ("Hero Run.inf_anim", PHASE29_RUN_GUID, &set.run),
    ] {
        put(
            name,
            guid,
            inf_asset::AssetKind::AnimClip,
            inf_asset::encode(&inf_anim::AnimClipAsset::new(
                clip.clone(),
                Some(skel_bytes),
            ))
            .map_err(|e| format!("encode {name}: {e}"))?,
        )?;
    }

    let machine = phase29_machine();
    put(
        "Hero Locomotion.inf_sm",
        PHASE29_SM_GUID,
        inf_asset::AssetKind::StateMachine,
        inf_asset::encode(&inf_anim::StateMachineAsset::new(
            machine.clone(),
            Some(skel_bytes),
        ))
        .map_err(|e| format!("encode machine: {e}"))?,
    )?;
    put(
        "Hero Controller.inf_act",
        PHASE29_ACTOR_GUID,
        inf_asset::AssetKind::Blueprint,
        encode_actor(&phase29_controller())?,
    )?;

    // ── the text an author owns (pillar S1) ──
    crate::sm_text::write_text(&dir.join("Hero Locomotion.inf_sm"), &machine)
        .map_err(|e| format!("write machine text: {e}"))?;
    std::fs::write(
        dir.join("camera.toml"),
        inf_ecs::camera::CameraTuning::default().to_toml()?,
    )
    .map_err(|e| format!("write camera table: {e}"))?;
    std::fs::write(
        dir.join("input.toml"),
        toml::to_string_pretty(&inf_input::default_map())
            .map_err(|e| format!("encode bindings: {e}"))?,
    )
    .map_err(|e| format!("write bindings: {e}"))?;

    std::fs::write(dir.join("README.md"), PHASE29_README)
        .map_err(|e| format!("write readme: {e}"))?;
    Ok(())
}

const PHASE29_README: &str = "# Phase 29 Locomotion (the P29.6 gate scene)\n\n\
An **obstacle course**, and it is a course rather than a level because of what\n\
the movement-catalogue amendment asks of it: *P29.6's course must force every\n\
catalogue mode in its one deterministic replay, so the (pose, mode) trace\n\
certifies the catalogue and not a subset.* Every block in\n\
`Phase29Locomotion.inf_lvl` exists because one mode cannot be reached without\n\
it -- a 1.4 m roof to crouch under, four 20 cm risers to autostep up, ledges at\n\
1 m and 3 m for the two mantle height classes, a 5 m one for the drop a landing\n\
classifies as a ragdoll, and a 3 m pool to swim in and under.\n\n\
## The character is the wizard's own output\n\n\
`Hero.inf_skel`, `Hero Body.inf_mesh`, the three cycles and\n\
`Hero Locomotion.inf_sm` are what `inf_editor_core::character::build_character`\n\
produces from the default biped -- generated here with fixed GUIDs so the\n\
committed bytes are reproducible, but through the same doors: the template rig,\n\
the rig-derived locomotion set, `inf_anim::derive_clip` at the import door and\n\
`inf_anim::propose` over what the derivation measured.\n\n\
**The three cycles are the repository's first committed DERIVED content.** They\n\
carry a root-motion track, a distance track, foot-plant sync markers, footstep\n\
notifies and six curve channels -- none of which any committed clip had before,\n\
which is the remainder P29.4 and P29.5 both wrote down.\n\n\
## The text beside them is the point\n\n\
- `Hero Locomotion.inf_sm.txt` -- the machine, as text (pillar S1). One value\n\
  per line, conditions as expressions, and `phase29_gate`'s one-line-diff arm\n\
  edits exactly one of those lines and measures what changes.\n\
- `camera.toml` -- the locomotion camera's table. A camera is not sim state, so\n\
  it has no home in the scene schema and lives here instead.\n\
- `input.toml` -- the bindings, in the format the shipped player already reads\n\
  beside a level.\n\n\
## What the gate does with it\n\n\
`runtime/inf-player/tests/phase29_gate.rs`: PIE == shipping byte-for-byte on the\n\
(pose, mode) trace with every mode named, bit-exact replay across two\n\
independent cooks, Blueprint-versus-transpiled parity over a course segment\n\
driven through the `anim.*` kit, the one-line-diff demonstration, and a camera\n\
trace that is deterministic and is NOT part of the sim trace.\n\n\
Regenerate with `INF_BLESS_SAMPLES=1 cargo test -p inf-editor-core samples`.\n";

// ── the island's city fixture (I3) ───────────────────────────────────────────
//
// The wave's benchmark, and the shape the certification's IB-2 is about: not
// "seven buildings cost 28 % of a frame" but "what does a THOUSAND do". Every
// piece of it is generated — the samples law — so every number the gate prints
// is a property of a rule rather than of hand-authored bytes.

/// Blocks on a side. `CITY_BLOCKS² × lots-per-block` is the building count.
pub const CITY_BLOCKS: u32 = 10;
/// A downtown block, in metres. 100 × 60 cuts into 5 × 2 lots at
/// [`CITY_FRONTAGE_M`] of frontage and [`CITY_DEPTH_M`] of depth.
pub const CITY_BLOCK_M: (f64, f64) = (100.0, 60.0);
/// Block pitch: the block plus the street between two of them — 40 m of street
/// on either axis.
pub const CITY_PITCH_M: (f64, f64) = (140.0, 100.0);
/// Street frontage per lot.
pub const CITY_FRONTAGE_M: f64 = 20.0;
/// Lot depth. Two rows back to back on a 60 m block.
pub const CITY_DEPTH_M: f64 = 30.0;
/// Yard on every side of every lot.
pub const CITY_SETBACK_M: f64 = 1.5;
/// Storeys. Two, so the fixture is a *city* rather than a stress test of the
/// floor stack: IB-2's subject is the building count.
pub const CITY_FLOORS: u32 = 2;
/// Metres a scripted drive-through advances per fixed step. 15 m/s at 60 Hz is
/// a car in town, and it crosses an `inf_ecs::BAND_LATTICE_M` lattice cell every
/// 64 steps — so a 240-step run re-bands several times rather than never.
pub const CITY_DRIVE_STEP_M: f64 = 0.25;
/// Fixed steps the gate's drive-through runs. 480 at 60 Hz is eight seconds and
/// 120 m — one whole block pitch, so the band leaves a block as well as entering
/// one. A shorter run is an *approach* rather than a drive-through, and an
/// approach only ever exercises the band growing.
pub const CITY_STEPS: usize = 480;

pub const CITY_LEVEL_GUID: Uuid = Uuid::from_u128(0x8430_0000);
pub const CITY_PCG_GUID: Uuid = Uuid::from_u128(0x8430_0001);
pub const CITY_ROAD_MESH_GUID: Uuid = Uuid::from_u128(0x8430_0002);
pub const CITY_SUN_GUID: Uuid = Uuid::from_u128(0x8430_0003);
pub const CITY_DRIVER_GUID: Uuid = Uuid::from_u128(0x8430_0004);
pub const CITY_ROAD_GUID: Uuid = Uuid::from_u128(0x8430_0005);
const CITY_BLOCK_BASE: u128 = 0x8430_1000;

/// Block `i`'s stable GUID.
pub fn city_block_guid(i: u32) -> Uuid {
    Uuid::from_u128(CITY_BLOCK_BASE + u128::from(i))
}

/// Block `i`'s centre in world XZ, with the grid centred on the origin.
pub fn city_block_centre(i: u32) -> DVec2 {
    let (cx, cz) = (i % CITY_BLOCKS, i / CITY_BLOCKS);
    DVec2::new(
        (f64::from(cx) - f64::from(CITY_BLOCKS - 1) * 0.5) * CITY_PITCH_M.0,
        (f64::from(cz) - f64::from(CITY_BLOCKS - 1) * 0.5) * CITY_PITCH_M.1,
    )
}

/// The city's ground: **flat, at zero**, and that is a decision.
///
/// A terrain would make every building's datum a height query and every one of
/// the gate's numbers a statement about the terrain sampler as much as about the
/// band. IB-2's subject is the collider count, so the ground is held constant
/// and the measurement has one variable. `phase19-town` is the composed scene
/// with real terrain, biomes and streaming, and it stays that.
pub fn city_ground(_x: f64, _z: f64) -> f64 {
    0.0
}

/// Where the driver is at `step` — a straight run down the middle street, west
/// to east, at [`CITY_DRIVE_STEP_M`] per fixed step.
///
/// Scripted rather than input-driven so the trace is a function of the level
/// alone — the phase-16 gate's own discipline, applied to a drive-through.
pub fn city_drive_point(step: u64) -> DVec3 {
    // Starts on the **westernmost block's own centre line**, not outside the
    // city: a run that begins in open ground spends its first half approaching,
    // and an approach only ever exercises the band growing.
    let half = f64::from(CITY_BLOCKS - 1) * 0.5 * CITY_PITCH_M.0;
    DVec3::new(-half + step as f64 * CITY_DRIVE_STEP_M, 1.0, 0.0)
}

/// **The city's one graph** — every block volume points at it and differs only
/// by its own `PcgVolume::seed`.
///
/// `grammar.footprint` gives the block its rectangle, `building.lots` cuts it
/// into lots, `building.plan` stands one building on each. One asset rather than
/// a hundred, because a district's variety comes from the *seed* and a hundred
/// near-identical graphs would be a hundred things to keep in step.
pub fn city_block_graph() -> inf_graph::Graph {
    let reg = inf_pcg::pcg_registry();
    let mut g = inf_graph::Graph::empty();
    use inf_graph::ParamValue as P;
    let add = |g: &mut inf_graph::Graph,
               n: u32,
               type_id: &str,
               params: &[(&str, inf_graph::ParamValue)]| {
        let node = inf_graph::NodeId(n);
        let mut m = inf_graph::ParamMap::new();
        for (k, v) in params {
            m.insert((*k).to_string(), v.clone());
        }
        inf_graph::apply_edits(
            g,
            &reg,
            &[inf_graph::GraphEdit::AddNode {
                id: node,
                type_id: type_id.into(),
                x: 0.0,
                y: 0.0,
                params: m,
            }],
        );
        node
    };
    let block = add(
        &mut g,
        1,
        "grammar.footprint",
        &[
            ("size_x", P::Float(CITY_BLOCK_M.0)),
            ("size_z", P::Float(CITY_BLOCK_M.1)),
        ],
    );
    let lots = add(
        &mut g,
        2,
        "building.lots",
        &[
            ("frontage", P::Float(CITY_FRONTAGE_M)),
            ("depth", P::Float(CITY_DEPTH_M)),
            ("jitter", P::Float(0.1)),
            ("setback", P::Float(CITY_SETBACK_M)),
            ("min_area", P::Float(40.0)),
        ],
    );
    let arch = add(
        &mut g,
        3,
        "building.archetype",
        &[
            (
                "archetype",
                P::Enum(inf_pcg::ArchetypeId::Office.name().into()),
            ),
            ("floors", P::Int(i64::from(CITY_FLOORS))),
            ("furnish", P::Bool(false)),
        ],
    );
    let plan = add(
        &mut g,
        4,
        "building.plan",
        &[
            ("name", P::Text("block".into())),
            ("seed", P::Int(30)),
            // The volume's own datum, because the level carries no terrain: a
            // `Terrain` lookup would fail closed and the city would be empty.
            ("ground", P::Enum("Span".into())),
        ],
    );
    let out = add(&mut g, 5, "output.pcg", &[]);
    for (from, fp, to, tp) in [
        (block, "out", lots, "block"),
        (lots, "out", plan, "lots"),
        (arch, "out", plan, "archetype"),
        (plan, "out", out, "scatter"),
    ] {
        inf_graph::apply_edits(
            &mut g,
            &reg,
            &[inf_graph::GraphEdit::Connect {
                link: inf_graph::Link {
                    from,
                    from_port: fp.into(),
                    to,
                    to_port: tp.into(),
                },
            }],
        );
    }
    g
}

/// The street grid as a **vector layer** — the input `RoadGraph::from_layer`
/// takes, which is I2's import door and not a shortcut past it.
///
/// One polyline per street *segment*, split at every crossing, because
/// `RoadGraph` derives its junctions from segment ENDPOINTS: a street digitised
/// as one feature passing *through* a crossing creates no node there. That is
/// I2's own carried bound, and the reason its junction fixture splits its
/// through-road.
pub fn city_road_layer() -> inf_gis::GeoLayer {
    use inf_gis::{Attr, GeoFeature, GeoGeometry, GeoLayer, LayerKind};
    let n = CITY_BLOCKS;
    let xs: Vec<f64> = (0..=n)
        .map(|k| (f64::from(k) - f64::from(n) * 0.5) * CITY_PITCH_M.0)
        .collect();
    let zs: Vec<f64> = (0..=n)
        .map(|k| (f64::from(k) - f64::from(n) * 0.5) * CITY_PITCH_M.1)
        .collect();

    let mut features = Vec::new();
    let mut seg = |a: DVec3, b: DVec3| {
        let mut f = GeoFeature::new(GeoGeometry::Polyline {
            points: vec![a, b],
            closed: false,
        });
        f.attributes
            .insert("highway".to_string(), Attr::Text("residential".to_string()));
        features.push(f);
    };
    for &x in &xs {
        for w in zs.windows(2) {
            seg(DVec3::new(x, 0.0, w[0]), DVec3::new(x, 0.0, w[1]));
        }
    }
    for &z in &zs {
        for w in xs.windows(2) {
            seg(DVec3::new(w[0], 0.0, z), DVec3::new(w[1], 0.0, z));
        }
    }
    GeoLayer {
        name: "City Streets".to_string(),
        kind: LayerKind::Roads,
        features,
        source_crs: "EPSG:32610".to_string(),
        advisories: Vec::new(),
        skipped: Vec::new(),
    }
}

/// The road **mesh**, through I2's own `RoadGraph` → `build_surface` →
/// `surface_to_mesh` door.
/// # The step is [`CITY_ROAD_STEP_M`], not the 1 m default, and that is a
/// measurement rather than a saving
///
/// `build_surface` resamples both axes at the **ground's** pitch, because what
/// the step buys is conformance to the terrain's chord between samples — I2's
/// own finding, and the reason the default is a metre. This city's ground is
/// [`city_ground`]: flat, at zero. A plane has no chord error at any step, so
/// every extra vertex is a vertex that says exactly what its neighbour said.
///
/// At the default the committed mesh is **213 941 vertices / 15.1 MB**; at 20 m
/// it is a fraction of that, and
/// `the_citys_streets_conform_to_their_flat_ground_at_any_step` measures the
/// deviation at **0.000000 m** so the trade is stated rather than assumed. The
/// day this fixture grows a terrain the step goes back to the terrain's pitch,
/// and that arm is what will say so.
pub const CITY_ROAD_STEP_M: f64 = 20.0;

pub fn city_road_mesh() -> Result<inf_mesh::MeshAsset, String> {
    let layer = city_road_layer();
    let graph = inf_gis::RoadGraph::from_layer(&layer);
    let opts = inf_gis::SurfaceOptions {
        ground_step_m: CITY_ROAD_STEP_M,
        ..Default::default()
    };
    let surface = inf_gis::build_surface(&graph, &opts, &mut |x, z| Some(city_ground(x, z)));
    let (mesh, _report) = inf_gis::surface_to_mesh(&surface, DVec3::ZERO)
        .map_err(|e| format!("the city's roads did not build a surface: {e}"))?;
    Ok(mesh)
}

/// `samples/phase30-city`.
pub fn city_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../samples/phase30-city")
}

/// The city level: one `PcgVolume` per block, a driver that is the sim's
/// streaming source, the road mesh, and a sun.
pub fn city_scene() -> SceneDoc {
    use inf_ecs::components::{Light, LightKind, MeshRef, PcgVolume, StreamingSource, Transform};
    use inf_ecs::math::{Color, Vec2d, Vec3d};

    let mut doc = SceneDoc::new();
    doc.set_title("Island City");

    doc.create_with_guid(CITY_SUN_GUID, SpawnKind::Empty, "Sun", None);
    insert!(
        doc,
        CITY_SUN_GUID,
        Transform {
            translation: Vec3d::ZERO,
            rotation: Vec3d::new(-52.0, -34.0, 0.0),
            scale: Vec3d::ONE,
        }
    );
    insert!(
        doc,
        CITY_SUN_GUID,
        Light {
            kind: LightKind::Directional,
            color: Color::WHITE,
            intensity: 3.0,
            ..Default::default()
        }
    );

    // **The driver IS the band's anchor**, and it carries `StreamingSource`
    // rather than a marker of its own: the collider band reads exactly the set
    // P16's cell activation reads, so a level cannot arrange for the two to
    // disagree about where the simulation is.
    doc.create_with_guid(CITY_DRIVER_GUID, SpawnKind::Empty, "Driver", None);
    let start = city_drive_point(0);
    insert!(
        doc,
        CITY_DRIVER_GUID,
        Transform {
            translation: Vec3d::new(start.x, start.y, start.z),
            rotation: Vec3d::ZERO,
            scale: Vec3d::ONE,
        }
    );
    insert!(doc, CITY_DRIVER_GUID, StreamingSource { radius_m: 256.0 });

    doc.create_with_guid(CITY_ROAD_GUID, SpawnKind::Empty, "Streets", None);
    insert!(doc, CITY_ROAD_GUID, Transform::IDENTITY);
    insert!(
        doc,
        CITY_ROAD_GUID,
        MeshRef {
            asset: Some(CITY_ROAD_MESH_GUID),
            ..Default::default()
        }
    );

    for i in 0..(CITY_BLOCKS * CITY_BLOCKS) {
        let guid = city_block_guid(i);
        let c = city_block_centre(i);
        doc.create_with_guid(guid, SpawnKind::Empty, &format!("Block {i}"), None);
        insert!(
            doc,
            guid,
            Transform {
                translation: Vec3d::new(c.x, 0.0, c.y),
                rotation: Vec3d::ZERO,
                scale: Vec3d::ONE,
            }
        );
        insert!(
            doc,
            guid,
            PcgVolume {
                graph: Some(CITY_PCG_GUID),
                extent: Vec2d::new(CITY_BLOCK_M.0 * 0.5, CITY_BLOCK_M.1 * 0.5),
                // The block's own seed is what makes a hundred volumes sharing
                // one graph a hundred different blocks.
                seed: i,
                ..Default::default()
            }
        );
    }
    doc
}

/// Write every committed city file.
pub fn write_city() -> Result<(), String> {
    let dir = city_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("mkdir: {e}"))?;

    crate::scene::serialize::save(
        &city_scene(),
        &dir.join("City.inf_lvl"),
        Some(CITY_LEVEL_GUID),
    )?;

    let graph = city_block_graph();
    let lowered = inf_pcg::lower_graph(&graph, &inf_pcg::pcg_registry());
    if !lowered.ok {
        return Err(format!(
            "the city's block graph does not lower: {:?}",
            lowered.issues
        ));
    }
    let pcg = inf_pcg::PcgAssetPayload::from_graph(&graph, lowered.document);
    let bytes = inf_asset::encode(&pcg).map_err(|e| format!("encode .inf_pcg: {e}"))?;
    write_phase19_asset(
        &dir.join("CityBlock.inf_pcg"),
        &bytes,
        CITY_PCG_GUID,
        inf_asset::AssetKind::Pcg,
    )?;

    let mesh = city_road_mesh()?;
    let bytes = inf_asset::encode(&mesh).map_err(|e| format!("encode road mesh: {e}"))?;
    write_phase19_asset(
        &dir.join("CityRoads.inf_mesh"),
        &bytes,
        CITY_ROAD_MESH_GUID,
        inf_asset::AssetKind::Mesh,
    )?;

    std::fs::write(dir.join("README.md"), CITY_README).map_err(|e| format!("write readme: {e}"))?;
    Ok(())
}

const CITY_README: &str = "# The island city (wave I3's benchmark)\n\n\
Generated by `inf_editor_core::samples::city_scene` and its siblings. **A thousand\n\
buildings on subdivided blocks**, which is the scale the AAA-readiness\n\
certification's IB-2 is about: not \"seven buildings cost 28 % of a frame\" but what\n\
a city does.\n\n\
- `City.inf_lvl` -- a sun, the street mesh, a **Driver** carrying\n\
  `StreamingSource` (the collider band's anchor, and the same component P16's\n\
  cell activation reads), and 100 `PcgVolume` blocks that all point at the one\n\
  graph below and differ only by their own `seed`.\n\
- `CityBlock.inf_pcg` -- `grammar.footprint -> building.lots -> building.plan`.\n\
  IB-2c's subdivision node is what makes one `building.plan` node ten buildings;\n\
  before it, one node was one building and two cities meant thousands of\n\
  hand-authored nodes.\n\
- `CityRoads.inf_mesh` -- the street grid through I2's own import door\n\
  (`RoadGraph::from_layer -> build_surface -> surface_to_mesh`), one polyline per\n\
  segment split at every crossing because `RoadGraph` derives its junctions from\n\
  segment ENDPOINTS.\n\n\
## The ground is flat, on purpose\n\n\
A terrain would make every building's datum a height query and every measured\n\
number a statement about the terrain sampler as much as about the band. IB-2's\n\
subject is the collider count, so the ground is held constant and the measurement\n\
has one variable. `phase19-town` is the composed scene with real terrain, biomes\n\
and streaming, and it stays that.\n\n\
## What the gate does with it\n\n\
`runtime/inf-player/tests/city_scale.rs`: the banded collider count against the\n\
`STREAMED_STEP_BUDGET_MS` the phase-16 gate holds a streamed world to, the\n\
unbanded alternative priced in the same run, the building count and the\n\
subdivision's world proof, and **PIE == shipping on a scripted drive-through**\n\
-- because the collider band is a function of sim state, and a band that read a\n\
camera would make two hosts simulate different worlds.\n\n\
Regenerate with `INF_BLESS_SAMPLES=1 cargo test -p inf-editor-core samples`.\n";

// ── the fps instrument's scene (island wave I4) ─────────────────────────────
//
// **Composed, not committed.** Everything below builds a level out of content
// that is already in the tree — the phase-30 city, a streamed terrain, and the
// phase-29 wizard character — and nothing below writes a file into `samples/`.
// A frame-time instrument needs *one* scene that is honestly heavy, and the
// honest way to get one is to put the heaviest things this engine already ships
// into the same frame rather than to author a new benchmark that flatters it.

/// The instrument level's GUID.
const ISLAND_FRAME_LEVEL_GUID: Uuid = Uuid::from_u128(0x8431_0000);
/// The streamed terrain entity under the city.
const ISLAND_FRAME_TERRAIN_GUID: Uuid = Uuid::from_u128(0x8431_0001);
/// …and its `.inf_terrain` asset.
const ISLAND_FRAME_TERRAIN_ASSET_GUID: Uuid = Uuid::from_u128(0x8431_00AA);

/// Samples per side of the instrument terrain's source heightmap.
///
/// `2·1024 + 1`: a whole number of 128 m tiles with the shared edge, so the
/// import produces exactly [`ISLAND_FRAME_TERRAIN_TILES`]² level-0 pages and no
/// partial row.
pub const ISLAND_FRAME_SOURCE_SAMPLES: u32 = 2049;
/// Samples per tile side — the same 129 the phase-16 gate streams.
pub const ISLAND_FRAME_TILE_RESOLUTION: u32 = 129;
/// Metres per sample. 1 m, so a tile spans 128 m and the terrain's LOD cut turns
/// over several times across the city rather than covering it in one page.
pub const ISLAND_FRAME_MPS: f64 = 1.0;
/// Level-0 tiles per side (`(2049 − 1) / (129 − 1)`).
pub const ISLAND_FRAME_TERRAIN_TILES: u32 =
    (ISLAND_FRAME_SOURCE_SAMPLES - 1) / (ISLAND_FRAME_TILE_RESOLUTION - 1);

/// The terrain's world span, metres (2 048 m — the city plus a kilometre of
/// margin in each direction).
pub fn island_frame_terrain_span_m() -> f64 {
    f64::from(ISLAND_FRAME_SOURCE_SAMPLES - 1) * ISLAND_FRAME_MPS
}

/// Where the terrain entity stands, so its `+X/+Z` grid is **centred on the
/// city**: the city occupies x ∈ [−630, 630] and z ∈ [−450, 450].
pub fn island_frame_terrain_origin() -> DVec3 {
    let half = island_frame_terrain_span_m() * 0.5;
    DVec3::new(-half, 0.0, -half)
}

/// The instrument terrain's source heightmap: **constant, and that is the whole
/// point**.
///
/// A rolling terrain would move every building's datum and the composed level
/// would stop being the city wave I3 measured — 370 468 solids, 1 000 buildings,
/// 6 067 banded colliders. Held flat at zero, the ground under the city is a
/// *real streamed terrain doing real streaming work* whose height answer is
/// byte-identical to [`city_ground`], so the instrument renders exactly the city
/// the ledger describes with ground beneath it, and
/// `the_ground_under_the_city_changes_no_building` says so rather than assuming
/// it.
///
/// Flat costs the measurement nothing: a terrain page is `res²` heights whatever
/// they are, the clipmap draws the same vertices, and the fragment work does not
/// vary with slope.
fn island_frame_source_png() -> Result<Vec<u8>, String> {
    let n = ISLAND_FRAME_SOURCE_SAMPLES;
    inf_terrain::encode_png16(&inf_terrain::HeightImage {
        width: n,
        height: n,
        samples: vec![0u16; (n as usize) * (n as usize)],
    })
    .map_err(|e| format!("encode instrument heightmap: {e}"))
}

/// Import the instrument's `.inf_terrain` through the **same chunked door** the
/// Terrain Import wizard uses (`phase16_terrain_asset`'s path, different grid).
pub fn island_frame_terrain_asset() -> Result<inf_terrain::TerrainAsset, String> {
    let png = island_frame_source_png()?;
    let probe = inf_terrain::probe_heightmap_bytes(&png).map_err(|e| format!("probe: {e}"))?;
    let settings = crate::assets::terrain_import::TerrainImportSettings {
        tile_resolution: ISLAND_FRAME_TILE_RESOLUTION,
        meters_per_sample: ISLAND_FRAME_MPS,
        min_height: 0.0,
        // A one-metre band with every sample at its floor: heights are exactly
        // 0.0, which is what makes the ground query agree with `city_ground` bit
        // for bit rather than nearly.
        max_height: 1.0,
        float_meters: false,
        center: false,
        ..Default::default()
    };
    let import = settings.to_import(probe.width, probe.height);
    let opts = inf_terrain::ChunkedImportOptions {
        pyramid: settings.pyramid(),
        world_origin: glam::DVec3::ZERO,
        nodata: inf_terrain::NodataHandling::NONE,
    };
    let (asset, _report) = inf_terrain::import_heightmap_reader(
        std::io::Cursor::new(png),
        import,
        opts,
        &mut |_| {},
        &|| false,
    )
    .map_err(|e| format!("chunked import: {e}"))?;
    Ok(asset)
}

/// Write the instrument terrain (+ its sidecar) into a scaffolded project's
/// `Content`. **Fixture setup — never `samples/`.**
pub fn write_island_frame_terrain(dir: &std::path::Path) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("mkdir: {e}"))?;
    let asset = island_frame_terrain_asset()?;
    let path = dir.join("IslandFrame.inf_terrain");
    let bytes = inf_terrain::write_terrain_asset(&path, &asset)
        .map_err(|e| format!("write terrain asset: {e}"))?;
    inf_asset::AssetSidecar::new(
        inf_asset::AssetId(ISLAND_FRAME_TERRAIN_ASSET_GUID),
        inf_asset::AssetKind::Terrain,
        inf_asset::ContentHash::of(bytes),
    )
    .save(&path)
    .map_err(|e| format!("write terrain sidecar: {e}"))
}

/// The character asset files the instrument borrows from `samples/phase29-locomotion`.
///
/// The *level* is not among them: the instrument composes its own, and copying
/// a second `.inf_lvl` into the project would give the cook two startup levels
/// to choose between.
pub fn island_frame_character_files() -> [&'static str; 15] {
    [
        "Hero Body.inf_mesh",
        "Hero Body.inf_mesh.toml",
        "Hero Controller.inf_act",
        "Hero Controller.inf_act.toml",
        "Hero Idle.inf_anim",
        "Hero Idle.inf_anim.toml",
        "Hero Locomotion.inf_sm",
        "Hero Locomotion.inf_sm.toml",
        "Hero Run.inf_anim",
        "Hero Run.inf_anim.toml",
        "Hero Walk.inf_anim",
        "Hero Walk.inf_anim.toml",
        "Hero.inf_skel",
        "Hero.inf_skel.toml",
        "Hero Locomotion.inf_sm.txt",
    ]
}

/// The instrument character's GUID — the entity a camera follows, and the one
/// the skinned pass draws.
pub fn island_frame_hero() -> Uuid {
    PHASE29_HERO_GUID
}

/// Where the character stands: on the drive-through's own line, one block east
/// of its start, so the flythrough passes it rather than starting on top of it.
pub fn island_frame_hero_feet() -> DVec3 {
    let p = city_drive_point(0);
    DVec3::new(p.x + CITY_PITCH_M.0, 0.0, p.z + 3.0)
}

/// **The instrument's level**: the phase-30 city, a streamed terrain beneath it,
/// and the phase-29 wizard character standing on the middle street.
///
/// Everything a shipping frame of this engine has to do at once — a thousand
/// grammar buildings in a hundred banded volumes, a real road mesh, a paging
/// heightfield, a skinned character with a live state machine, and a sun — in
/// one scene, so "what does a frame cost" has a single answer instead of six
/// per-feature ones.
///
/// The character is built the way the P24 wizard builds one (a capsule derived
/// from the creature's own height, feet on the floor, the movement component
/// carrying its own half-heights) — the same construction
/// `phase29_locomotion_scene` commits, because a benchmark character assembled
/// differently from a shipped one measures a character nobody ships.
pub fn island_frame_scene() -> SceneDoc {
    use inf_ecs::components::{
        ActorClass, AlwaysLoaded, AnimStateMachine, BodyKind3D, CharacterController3D,
        CharacterMovement, Collider3D, ColliderShape3DKind, RigidBody3D, SkeletalMesh, Terrain,
    };

    let mut doc = city_scene();
    doc.set_title("Island Frame");

    // ── the ground the city has never had ──
    //
    // `AlwaysLoaded` for `phase16_world_scene`'s reason: a Terrain occupies
    // space, so a partitioner would bin the whole heightfield into one cell.
    doc.create_with_guid(ISLAND_FRAME_TERRAIN_GUID, SpawnKind::Empty, "Ground", None);
    insert!(
        doc,
        ISLAND_FRAME_TERRAIN_GUID,
        Transform::from_translation(island_frame_terrain_origin())
    );
    {
        let mut terrain = Terrain::configured(ISLAND_FRAME_TILE_RESOLUTION, ISLAND_FRAME_MPS);
        terrain.asset = Some(ISLAND_FRAME_TERRAIN_ASSET_GUID);
        debug_assert!(terrain.data.is_empty(), "a streamed terrain ships no tiles");
        insert!(doc, ISLAND_FRAME_TERRAIN_GUID, terrain);
    }
    insert!(doc, ISLAND_FRAME_TERRAIN_GUID, AlwaysLoaded);

    // ── the character, as the wizard makes one ──
    let radius = (PHASE29_HEIGHT_M * 0.15).clamp(0.1, 0.5);
    let half_h = (PHASE29_HEIGHT_M * 0.5 - radius).max(0.05);
    let feet = island_frame_hero_feet();
    doc.create_with_guid(PHASE29_HERO_GUID, SpawnKind::Empty, "Hero", None);
    insert!(
        doc,
        PHASE29_HERO_GUID,
        Transform::from_translation(DVec3::new(feet.x, feet.y + half_h + radius, feet.z))
    );
    insert!(
        doc,
        PHASE29_HERO_GUID,
        SkeletalMesh {
            mesh: Some(PHASE29_MESH_GUID),
            skeleton: Some(PHASE29_SKELETON_GUID),
        }
    );
    insert!(
        doc,
        PHASE29_HERO_GUID,
        AnimStateMachine {
            sm: Some(PHASE29_SM_GUID),
            ..Default::default()
        }
    );
    insert!(
        doc,
        PHASE29_HERO_GUID,
        RigidBody3D {
            kind: BodyKind3D::Kinematic,
            ..Default::default()
        }
    );
    insert!(
        doc,
        PHASE29_HERO_GUID,
        Collider3D {
            shape_kind: ColliderShape3DKind::Capsule,
            half_extents: inf_ecs::math::Vec3d::new(radius, half_h, radius),
            radius,
            ..Default::default()
        }
    );
    insert!(doc, PHASE29_HERO_GUID, CharacterController3D::default());
    insert!(
        doc,
        PHASE29_HERO_GUID,
        CharacterMovement {
            player_controlled: true,
            stand_half_height_m: half_h,
            crouch_half_height_m: (half_h * 0.5).max(0.05),
            prone_half_height_m: (radius * 0.6).max(0.03),
            ..Default::default()
        }
    );
    insert!(doc, PHASE29_HERO_GUID, ActorClass(PHASE29_ACTOR_GUID));
    insert!(doc, PHASE29_HERO_GUID, AlwaysLoaded);

    doc.world_mut().propagate();
    doc.mark_saved();
    doc
}

/// Save the instrument's level (+ sidecar) into a scaffolded project's
/// `Content`. **Fixture setup — never `samples/`.**
pub fn write_island_frame_level(dir: &std::path::Path) -> Result<(), String> {
    crate::scene::serialize::save(
        &island_frame_scene(),
        &dir.join("IslandFrame.inf_lvl"),
        Some(ISLAND_FRAME_LEVEL_GUID),
    )
    .map(|_| ())
}

// ── the island's gameplay fixture (I6) ──────────────────────────────────────
//
// One house the grammar built, one weapon on the floor, one destructible target
// and one hero — the smallest world in which every verb the owner's mandate
// names can be forced by a scripted trace.
//
// **Small on purpose.** The city fixture is a thousand buildings and 19 790
// doorways, and a gate that walked a character across it would spend its whole
// budget travelling. What a gameplay gate needs is one of each thing, close
// enough together that a script can reach them all — and the city's own numbers
// are measured where they belong, by `city_scale`.

/// The gameplay fixture's level.
pub const GAMEPLAY_LEVEL_GUID: Uuid = Uuid::from_u128(0x8460_0000);
/// Its one-house PCG graph.
pub const GAMEPLAY_PCG_GUID: Uuid = Uuid::from_u128(0x8460_0001);
/// The hero's Blueprint class — the authoring door for every item, door and
/// health value in this level.
pub const GAMEPLAY_ACTOR_GUID: Uuid = Uuid::from_u128(0x8460_0002);
/// The sun.
pub const GAMEPLAY_SUN_GUID: Uuid = Uuid::from_u128(0x8460_0003);
/// The ground slab.
pub const GAMEPLAY_GROUND_GUID: Uuid = Uuid::from_u128(0x8460_0004);
/// The hero.
pub const GAMEPLAY_HERO_GUID: Uuid = Uuid::from_u128(0x8460_0005);
/// The volume the house is grown in.
pub const GAMEPLAY_HOUSE_GUID: Uuid = Uuid::from_u128(0x8460_0006);
/// The destructible a bullet is fired at.
pub const GAMEPLAY_TARGET_GUID: Uuid = Uuid::from_u128(0x8460_0007);
/// The mesh the target fractures from.
pub const GAMEPLAY_TARGET_MESH_GUID: Uuid = Uuid::from_u128(0x8460_0008);

/// The house's footprint, metres.
pub const GAMEPLAY_HOUSE_M: (f64, f64) = (14.0, 10.0);
/// Where the hero starts, world metres — outside the house, facing `+Z`.
pub const GAMEPLAY_HERO_START: (f64, f64, f64) = (0.0, 0.0, -9.0);
/// Where the rifle lies, world metres.
pub const GAMEPLAY_RIFLE_AT: (f64, f64, f64) = (0.0, 0.4, -7.4);
/// Where the destructible target stands, world metres.
pub const GAMEPLAY_TARGET_AT: (f64, f64, f64) = (-6.0, 1.0, -9.0);

/// The hero's own body, joules — two rifle rounds' worth, so the gate can name
/// what it takes to stop one.
pub const GAMEPLAY_HERO_J: f64 = 2000.0;

/// **The item catalogue this level authors**, as the name-keyed TOML the
/// `item.define` node takes.
///
/// A `const` rather than a file, because it rides the Blueprint's own bytes —
/// see `inf_blueprint::nodekit`'s `gameplay_nodes` for why that is the only
/// surface that reaches Simulate, PIE and a cooked pack at once.
///
/// **Two weapons, and the second one is the gate's** (I6 audit): a wheel with
/// one weapon in the bag cycles back onto it, so "the wheel changed the equipped
/// weapon" was a claim a wheel that did nothing satisfied perfectly — measured,
/// deleting the wheel's whole consumer left `phase30_gameplay_gate` green. The
/// pistol is what makes the notch observable, and its numbers differ from the
/// rifle's in every field a trace can see so a cycle cannot be mistaken for a
/// no-op.
pub const GAMEPLAY_ITEMS_TOML: &str = concat!(
    "[rifle]\n",
    "label = \"Rifle\"\n",
    "stack_max = 1\n",
    "mass_kg = 3.6\n",
    "[rifle.weapon]\n",
    "damage_j = 1700.0\n",
    "rounds_per_minute = 600.0\n",
    "magazine = 30\n",
    "reserve = 120\n",
    "reload_s = 2.0\n",
    "spread_deg = 0.0\n",
    "range_m = 400.0\n",
    "automatic = true\n",
    "\n",
    "[pistol]\n",
    "label = \"Pistol\"\n",
    "stack_max = 1\n",
    "mass_kg = 0.9\n",
    "[pistol.weapon]\n",
    "damage_j = 600.0\n",
    "rounds_per_minute = 400.0\n",
    "magazine = 12\n",
    "reserve = 36\n",
    "reload_s = 1.4\n",
    "spread_deg = 0.0\n",
    "range_m = 60.0\n",
    "automatic = false\n",
    "\n",
    "[bandage]\n",
    "label = \"Bandage\"\n",
    "stack_max = 5\n",
    "mass_kg = 0.1\n",
);

/// **The doors this level hangs by hand**, as the `door.spawn` node's TOML.
///
/// **Four** of them, and they are the four the grammar cannot give a gate: a
/// front door the script opens, walks through and locks from the inside; a yard
/// gate it locks and kicks in from the outside; a shed door a sprint breaches;
/// and a hatch a dive goes through. The house's own doorways are the grammar's
/// and are hung without any of this.
///
/// *Written with `\n` escapes, and the I6 audit is why: the first draft carried
/// thirty-one **literal** newlines inside its string literals — the fifteenth
/// `chr(92)` catch's own shape, a scripted edit that resolved the escape before
/// the file was written. It was content-identical only because `.gitattributes`
/// forces `*.rs text eol=lf`; on a checkout without that rule every line of this
/// document would have arrived at the TOML parser with a `\r` on it. The escape
/// does not depend on a checkout rule.*
pub const GAMEPLAY_DOORS_TOML: &str = concat!(
    "[front]\n",
    "label = \"front door\"\n",
    "hinge = [-0.45, 1.05, -6.0]\n",
    "closed_yaw_deg = 90.0\n",
    "inside_yaw_deg = 0.0\n",
    "open_limit_deg = -95.0\n",
    "locked = false\n",
    "\n",
    "[gate]\n",
    "label = \"yard gate\"\n",
    "hinge = [7.55, 1.05, -9.45]\n",
    "closed_yaw_deg = 0.0\n",
    "inside_yaw_deg = 90.0\n",
    "open_limit_deg = 95.0\n",
    "locked = true\n",
    "\n",
    "[shed]\n",
    "label = \"shed door\"\n",
    "hinge = [17.55, 1.05, -9.45]\n",
    "closed_yaw_deg = 0.0\n",
    "inside_yaw_deg = 90.0\n",
    "open_limit_deg = 95.0\n",
    "locked = true\n",
    "\n",
    "[hatch]\n",
    "label = \"hatch\"\n",
    "hinge = [27.55, 1.05, -9.45]\n",
    "closed_yaw_deg = 0.0\n",
    "inside_yaw_deg = 90.0\n",
    "open_limit_deg = 95.0\n",
    "locked = false\n",
);

/// Where each hand-hung door's PROMPT is, world metres — the point a script
/// walks to and a gate measures from.
///
/// Half a leaf-width along its closed facing from the hinge, which is
/// `inf_ecs::door::prompt_position`'s own arithmetic. The three yard doors run
/// east along the hero's own row so one script can reach them all in a line.
pub const GAMEPLAY_FRONT_DOOR_AT: (f64, f64, f64) = (0.0, 1.05, -6.0);
/// The locked gate a kick opens.
pub const GAMEPLAY_GATE_AT: (f64, f64, f64) = (7.55, 1.05, -9.0);
/// The locked shed door a sprint goes through.
pub const GAMEPLAY_SHED_AT: (f64, f64, f64) = (17.55, 1.05, -9.0);
/// The shut hatch a dive goes through.
pub const GAMEPLAY_HATCH_AT: (f64, f64, f64) = (27.55, 1.05, -9.0);

/// The repo-root `samples/phase30-gameplay/` directory.
pub fn gameplay_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../samples/phase30-gameplay")
}

/// **The file the engine's gunshot lives in** (wave WPN1), beside the fixture
/// that fires it.
///
/// `inf_editor_core::settlement::VENUE_MUSIC_FILE`'s shape and its argument: a
/// clip an engine constant names by GUID
/// ([`inf_ecs::weapon::WEAPON_REPORT_CLIP`]) has to have the same GUID every
/// time, or the committed bytes are a different set of files on every build.
///
/// It sits here rather than in the settlement library because a gunshot is not
/// settlement content and there is no weapon library to put it in; the gameplay
/// fixture is the one committed level in this tree that fires a weapon, and a
/// second copy in `samples/harbour-heist` would be the same bytes twice.
pub const GAMEPLAY_REPORT_FILE: &str = "Report.inf_audio";

/// **The committed gunshot**, as an [`inf_audio::AudioAsset`].
///
/// A short deterministic tone, generated rather than recorded, on
/// [`playground_audio_asset`]'s own terms — a committed `.inf_audio` needs no
/// binary fixture, and a clip a test can regenerate is a clip a reviewer can
/// diff. **Eight hundred samples at 8 kHz is a tenth of a second**, which is a
/// tenth of the venue loop and is the one thing about it that is right: a
/// gunshot is a transient, and a report that outlasted its own weapon's cycle
/// time at 600 rpm would overlap itself for ever. It is not a gunshot and is not
/// pretending to be. What it makes true is the thing the wave needs true: the
/// `Play` a shot issues **names a clip that resolves**.
pub fn gameplay_report_asset() -> inf_audio::AudioAsset {
    inf_audio::AudioAsset::from_encoded(tone_wav(800, 8000), inf_audio::AudioFormat::Wav)
        .expect("tone wav decodes")
}

/// One `House`, grown on the volume's own datum — the level carries no terrain,
/// so a `Terrain` lookup would fail closed and the house would be nothing.
pub fn gameplay_house_graph() -> inf_graph::Graph {
    let reg = inf_pcg::pcg_registry();
    let mut g = inf_graph::Graph::empty();
    use inf_graph::ParamValue as P;
    let add = |g: &mut inf_graph::Graph,
               n: u32,
               type_id: &str,
               params: &[(&str, inf_graph::ParamValue)]| {
        let node = inf_graph::NodeId(n);
        let mut m = inf_graph::ParamMap::new();
        for (k, v) in params {
            m.insert((*k).to_string(), v.clone());
        }
        inf_graph::apply_edits(
            g,
            &reg,
            &[inf_graph::GraphEdit::AddNode {
                id: node,
                type_id: type_id.into(),
                x: 0.0,
                y: 0.0,
                params: m,
            }],
        );
        node
    };
    let plot = add(
        &mut g,
        1,
        "grammar.footprint",
        &[
            ("size_x", P::Float(GAMEPLAY_HOUSE_M.0)),
            ("size_z", P::Float(GAMEPLAY_HOUSE_M.1)),
        ],
    );
    // **ONE lot filling the whole plot.** The subdivider is the only thing that
    // produces the `lots` a `building.plan` takes, and a frontage as wide as the
    // footprint gives it exactly one — which is what "one house" means here.
    let lots = add(
        &mut g,
        2,
        "building.lots",
        &[
            ("frontage", P::Float(GAMEPLAY_HOUSE_M.0)),
            ("depth", P::Float(GAMEPLAY_HOUSE_M.1)),
            ("jitter", P::Float(0.0)),
            ("setback", P::Float(0.0)),
            ("min_area", P::Float(20.0)),
        ],
    );
    let arch = add(
        &mut g,
        3,
        "building.archetype",
        &[
            (
                "archetype",
                P::Enum(inf_pcg::ArchetypeId::House.name().into()),
            ),
            ("floors", P::Int(1)),
            ("furnish", P::Bool(false)),
        ],
    );
    let plan = add(
        &mut g,
        4,
        "building.plan",
        &[
            ("name", P::Text("house".into())),
            ("seed", P::Int(11)),
            ("ground", P::Enum("Span".into())),
        ],
    );
    let out = add(&mut g, 5, "output.pcg", &[]);
    for (from, fp, to, tp) in [
        (plot, "out", lots, "block"),
        (lots, "out", plan, "lots"),
        (arch, "out", plan, "archetype"),
        (plan, "out", out, "scatter"),
    ] {
        inf_graph::apply_edits(
            &mut g,
            &reg,
            &[inf_graph::GraphEdit::Connect {
                link: inf_graph::Link {
                    from,
                    from_port: fp.into(),
                    to,
                    to_port: tp.into(),
                },
            }],
        );
    }
    g
}

/// **The level's own authoring**: on `BeginPlay`, define the catalogue, hang the
/// two hand-placed doors, put a rifle on the floor and give the hero a body.
///
/// Everything a gameplay level needs that the scene cannot carry, in the one
/// place that rides `.inf_act` bytes to all three hosts.
pub fn gameplay_controller() -> BlueprintClass {
    let mut class = BlueprintClass::new("act:phase30-gameplay", "Gameplay Author");
    class.variables = vec![Variable {
        name: "entity".into(),
        ty: Ty::Int,
        default: Lit::Int(0),
        exposed: false,
    }];
    let me = || Expr::Call {
        path: vec!["vars".into(), "get".into()],
        args: vec![Expr::Lit(Lit::Str("entity".into()))],
    };
    let f = |v: f64| Expr::Lit(Lit::Float(v));
    let s = |v: &str| Expr::Lit(Lit::Str(v.into()));
    let call = |path: &[&str], args: Vec<Expr>| {
        Stmt::ExprStmt(Expr::Call {
            path: path.iter().map(|p| (*p).to_string()).collect(),
            args,
        })
    };
    class.events = vec![EventBinding {
        event: EventKind::BeginPlay,
        body: BlueprintFn {
            id: "begin".into(),
            name: "begin".into(),
            params: Vec::new(),
            ret: Ty::Unit,
            body: vec![
                call(&["item", "define"], vec![s(GAMEPLAY_ITEMS_TOML)]),
                call(&["door", "spawn"], vec![s(GAMEPLAY_DOORS_TOML)]),
                call(
                    &["item", "spawn_pickup"],
                    vec![
                        s("rifle"),
                        f(GAMEPLAY_RIFLE_AT.0),
                        f(GAMEPLAY_RIFLE_AT.1),
                        f(GAMEPLAY_RIFLE_AT.2),
                        Expr::Lit(Lit::Int(1)),
                    ],
                ),
                call(
                    &["item", "give"],
                    vec![me(), s("bandage"), Expr::Lit(Lit::Int(3))],
                ),
                // The second weapon, in the bag from the start — so the gate's
                // wheel station has something to cycle TO. See
                // `GAMEPLAY_ITEMS_TOML` for the measurement that put it here.
                // The order is load-bearing: bandages take slot 0 and this takes
                // slot 1, so the rifle the script picks up takes slot 2 and the
                // panel's focus walk is a fixed number of presses.
                call(
                    &["item", "give"],
                    vec![me(), s("pistol"), Expr::Lit(Lit::Int(1))],
                ),
                call(&["health", "set"], vec![me(), f(GAMEPLAY_HERO_J)]),
            ],
        },
    }];
    class
}

/// The gameplay fixture's scene.
pub fn gameplay_scene() -> SceneDoc {
    use inf_ecs::components::{
        BodyKind3D, CharacterController3D, CharacterMovement, Collider3D, ColliderShape3DKind,
        Destructible, Light, LightKind, MeshRef, PcgVolume, RigidBody3D, StreamingSource,
        Transform,
    };
    use inf_ecs::math::{Color, Vec2d, Vec3d};

    let mut doc = SceneDoc::new();
    doc.set_title("Island Gameplay");

    doc.create_with_guid(GAMEPLAY_SUN_GUID, SpawnKind::Empty, "Sun", None);
    insert!(
        doc,
        GAMEPLAY_SUN_GUID,
        Transform {
            translation: Vec3d::ZERO,
            rotation: Vec3d::new(-52.0, -34.0, 0.0),
            scale: Vec3d::ONE,
        }
    );
    insert!(
        doc,
        GAMEPLAY_SUN_GUID,
        Light {
            kind: LightKind::Directional,
            color: Color::WHITE,
            intensity: 3.0,
            ..Default::default()
        }
    );

    // The ground: one static slab, because the level carries no terrain and a
    // character with nothing under it falls out of the world.
    doc.create_with_guid(GAMEPLAY_GROUND_GUID, SpawnKind::Empty, "Ground", None);
    insert!(
        doc,
        GAMEPLAY_GROUND_GUID,
        Transform {
            translation: Vec3d::new(0.0, -0.5, 0.0),
            rotation: Vec3d::ZERO,
            scale: Vec3d::ONE,
        }
    );
    insert!(
        doc,
        GAMEPLAY_GROUND_GUID,
        RigidBody3D {
            kind: BodyKind3D::Static,
            ..Default::default()
        }
    );
    insert!(
        doc,
        GAMEPLAY_GROUND_GUID,
        Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: Vec3d::new(60.0, 0.5, 60.0),
            ..Default::default()
        }
    );

    // The house, grown by the grammar — its doorways become doors with no
    // authoring at all, which is clause 1 of the mandate.
    doc.create_with_guid(GAMEPLAY_HOUSE_GUID, SpawnKind::Empty, "House", None);
    insert!(doc, GAMEPLAY_HOUSE_GUID, Transform::IDENTITY);
    insert!(
        doc,
        GAMEPLAY_HOUSE_GUID,
        PcgVolume {
            graph: Some(GAMEPLAY_PCG_GUID),
            extent: Vec2d::new(GAMEPLAY_HOUSE_M.0 * 0.5, GAMEPLAY_HOUSE_M.1 * 0.5),
            seed: 1,
            ..Default::default()
        }
    );

    // The destructible: a box a rifle round is fired at, so the gate can watch
    // joules cross the P22 door.
    doc.create_with_guid(GAMEPLAY_TARGET_GUID, SpawnKind::Empty, "Target", None);
    insert!(
        doc,
        GAMEPLAY_TARGET_GUID,
        Transform {
            translation: Vec3d::new(
                GAMEPLAY_TARGET_AT.0,
                GAMEPLAY_TARGET_AT.1,
                GAMEPLAY_TARGET_AT.2
            ),
            rotation: Vec3d::ZERO,
            scale: Vec3d::ONE,
        }
    );
    insert!(
        doc,
        GAMEPLAY_TARGET_GUID,
        RigidBody3D {
            kind: BodyKind3D::Static,
            ..Default::default()
        }
    );
    insert!(
        doc,
        GAMEPLAY_TARGET_GUID,
        Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: Vec3d::new(1.0, 1.0, 1.0),
            ..Default::default()
        }
    );
    insert!(
        doc,
        GAMEPLAY_TARGET_GUID,
        MeshRef {
            asset: Some(GAMEPLAY_TARGET_MESH_GUID),
            ..Default::default()
        }
    );
    insert!(
        doc,
        GAMEPLAY_TARGET_GUID,
        Destructible {
            // Soft enough that a rifle round moves it: a 5 MPa masonry block
            // shares square-metre faces and costs kilojoules a bullet does not
            // have. The number is the fixture's, and the gate prints what it
            // measured against it.
            strength: 2.0e4,
            ..Default::default()
        }
    );

    // The hero.
    let half_h = 0.9;
    let radius = 0.3;
    doc.create_with_guid(GAMEPLAY_HERO_GUID, SpawnKind::Empty, "Hero", None);
    insert!(
        doc,
        GAMEPLAY_HERO_GUID,
        Transform {
            translation: Vec3d::new(
                GAMEPLAY_HERO_START.0,
                GAMEPLAY_HERO_START.1 + half_h + radius,
                GAMEPLAY_HERO_START.2
            ),
            rotation: Vec3d::ZERO,
            scale: Vec3d::ONE,
        }
    );
    insert!(
        doc,
        GAMEPLAY_HERO_GUID,
        RigidBody3D {
            kind: BodyKind3D::Kinematic,
            ..Default::default()
        }
    );
    insert!(
        doc,
        GAMEPLAY_HERO_GUID,
        Collider3D {
            shape_kind: ColliderShape3DKind::Capsule,
            half_extents: Vec3d::new(radius, half_h, radius),
            radius,
            ..Default::default()
        }
    );
    insert!(doc, GAMEPLAY_HERO_GUID, CharacterController3D::default());
    insert!(
        doc,
        GAMEPLAY_HERO_GUID,
        CharacterMovement {
            player_controlled: true,
            stand_half_height_m: half_h,
            crouch_half_height_m: (half_h * 0.5).max(0.05),
            prone_half_height_m: (radius * 0.6).max(0.03),
            ..Default::default()
        }
    );
    insert!(doc, GAMEPLAY_HERO_GUID, ActorClass(GAMEPLAY_ACTOR_GUID));
    // The hero is the band's anchor, exactly as the city's Driver is: the
    // collider band reads the set P16's cell activation reads, so a level cannot
    // arrange for the two to disagree about where the simulation is.
    insert!(doc, GAMEPLAY_HERO_GUID, StreamingSource { radius_m: 128.0 });

    doc.world_mut().propagate();
    doc
}

const GAMEPLAY_README: &str = concat!(
    "# samples/phase30-gameplay\n",
    "\n",
    "The island's gameplay fixture (wave I6): one grammar-built house, two\n",
    "hand-hung doors, a rifle on the floor, a destructible target and one hero.\n",
    "\n",
    "Everything a level cannot carry in its scene is authored by the hero's own\n",
    "Blueprint on `BeginPlay` - the item catalogue, the two doors, the pickup and\n",
    "the hero's own health. That is the one authoring surface which reaches the\n",
    "editor's Simulate, a PIE payload AND a cooked pack with no schema move; see\n",
    "`inf_blueprint::nodekit`'s `gameplay_nodes` for the accounting.\n",
    "\n",
    "Beside them, since wave WPN1, **one `.inf_audio`**: the gunshot every round\n",
    "leaving a barrel names by GUID (`inf_ecs::weapon::WEAPON_REPORT_CLIP`). It is\n",
    "engine content that happens to live here because this is the one committed\n",
    "level in the tree that fires a weapon.\n",
    "\n",
    "The gate over it is `runtime/inf-player/tests/phase30_gameplay_gate.rs`.\n",
    "\n",
    "Generated - do not hand-edit. Regenerate with:\n",
    "\n",
    "```sh\n",
    "INF_BLESS_SAMPLES=1 cargo test -p inf-editor-core samples\n",
    "```\n",
);

/// Write every committed gameplay-fixture file.
pub fn write_gameplay() -> Result<(), String> {
    let dir = gameplay_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("mkdir: {e}"))?;

    crate::scene::serialize::save(
        &gameplay_scene(),
        &dir.join("Gameplay.inf_lvl"),
        Some(GAMEPLAY_LEVEL_GUID),
    )?;

    let graph = gameplay_house_graph();
    let lowered = inf_pcg::lower_graph(&graph, &inf_pcg::pcg_registry());
    if !lowered.ok {
        return Err(format!(
            "the gameplay house graph does not lower: {:?}",
            lowered.issues
        ));
    }
    let pcg = inf_pcg::PcgAssetPayload::from_graph(&graph, lowered.document);
    write_phase19_asset(
        &dir.join("GameplayHouse.inf_pcg"),
        &inf_asset::encode(&pcg).map_err(|e| format!("encode pcg: {e}"))?,
        GAMEPLAY_PCG_GUID,
        inf_asset::AssetKind::Pcg,
    )?;

    write_phase19_asset(
        &dir.join("Target.inf_mesh"),
        &inf_asset::encode(&phase22_box_mesh(
            [-1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0],
            "mat:target",
        ))
        .map_err(|e| format!("encode target: {e}"))?,
        GAMEPLAY_TARGET_MESH_GUID,
        inf_asset::AssetKind::Mesh,
    )?;

    write_phase19_asset(
        &dir.join("Gameplay.inf_act"),
        &encode_actor(&gameplay_controller())?,
        GAMEPLAY_ACTOR_GUID,
        inf_asset::AssetKind::Blueprint,
    )?;

    // **The gunshot** (wave WPN1), with the GUID `inf_ecs::weapon` names it by —
    // the VEN1b club-loop precedent verbatim.
    write_anim_asset(
        &dir,
        GAMEPLAY_REPORT_FILE,
        inf_ecs::weapon::WEAPON_REPORT_CLIP,
        inf_asset::AssetKind::Audio,
        &gameplay_report_asset(),
    )?;

    std::fs::write(dir.join("README.md"), GAMEPLAY_README)
        .map_err(|e| format!("write readme: {e}"))?;
    Ok(())
}

// -- the starter character (SK1c) -------------------------------------------

/// The repo-root `samples/starter-character/` directory.
pub fn starter_character_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../samples/starter-character")
}

/// The repo-root `samples/ground/` directory — the engine's ground library
/// (TER2a clause 3). Beside the starter character rather than inside a sample,
/// because it is engine content: the island binds it, a template can, and any
/// project can drag it in.
pub fn ground_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../samples")
        .join(crate::ground::GROUND_FOLDER)
}

/// What the committed starter character is called -- the prefix on every one of
/// its files, and the name a level that spawns it shows in the Outliner.
pub const STARTER_CHARACTER_NAME: &str = "Starter";

/// **The starter character's eight asset GUIDs**, fixed so the committed bytes
/// are reproducible.
///
/// A character's assets name each other by GUID, so a build with minted ids is a
/// different set of files every time and nothing committed could be locked
/// against the door that wrote it. See
/// [`crate::character::build_character_with_ids`].
pub fn starter_character_ids() -> crate::character::CharacterIds {
    let id = |n: u128| Some(inf_asset::AssetId(Uuid::from_u128(0x5C10_00A0 + n)));
    crate::character::CharacterIds {
        skeleton: id(0),
        material: id(1),
        mesh: id(2),
        idle: id(3),
        walk: id(4),
        run: id(5),
        machine: id(6),
        actor: id(7),
    }
}

/// **The spec the New Character wizard opens with**, and therefore what ships.
///
/// `CharacterSpec::default()` with a name on it, and that is the whole point:
/// the committed starter character is not a shape tuned to look good in a
/// screenshot, it is *what an author gets by pressing the button*. If the
/// wizard's defaults move, this moves with them and the bless re-writes the
/// folder -- which is the loud version of a starter asset going stale.
///
/// `BodyPlan::Biped` has been the 161-bone mannequin since SK1a, so this is a
/// full rig with hands, twist bones, IK handles, a role table and (since SK1c) a
/// grip catalogue.
pub fn starter_character_spec() -> crate::character::CharacterSpec {
    crate::character::CharacterSpec {
        name: STARTER_CHARACTER_NAME.to_string(),
        ..Default::default()
    }
}

/// **Build the starter character and hand back every file it wrote**, as
/// `(file name, bytes)` sorted by name, **with the advisories the wizard
/// reported**.
///
/// # One generator, and this is it
///
/// Every other sample in this file is written by a pure generator that mirrors
/// what a tool would produce, with `the_showcase_character_matches_the_wizard_door`
/// standing between the mirror and the tool. A character is too big for a mirror
/// -- eight assets, a heat solve, a derivation and a proposal -- so this one does
/// not mirror anything: it runs the **wizard's own door** into a scratch project
/// and copies the result out. The committed bytes are literally
/// `build_character`'s output, and the byte lock is therefore a lock on the
/// wizard rather than on a copy of it.
///
/// # The advisories come back with it
///
/// The wizard reports what it could not do perfectly, and this build reports
/// one: 35 of the generated body's vertices cannot see a deform bone (SK1b's
/// measurement -- caps buried inside a neighbouring shell) and keep their seed
/// bone through the generator's rigid prior, which is the right answer.
/// Swallowing it here would make a *new* advisory invisible, and refusing on it
/// would make the folder unbuildable for a bound the engine already carries by
/// name -- so it is handed back, and
/// `the_starter_character_builds_clean_and_reproducibly` pins the whole list by
/// content.
///
/// The scratch directory is removed on the way out, including on failure.
#[allow(clippy::type_complexity)]
pub fn starter_character_files() -> Result<(Vec<(String, Vec<u8>)>, Vec<String>), String> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let scratch = std::env::temp_dir().join(format!(
        "inf-starter-character-{}-{}",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    // A stale directory from a killed run would put `unique_path` on
    // `Starter (1).inf_skel` and quietly produce a different file set.
    let _ = std::fs::remove_dir_all(&scratch);
    std::fs::create_dir_all(&scratch).map_err(|e| format!("scratch dir: {e}"))?;
    let out = starter_character_into(
        &scratch,
        &starter_character_spec(),
        &starter_character_ids(),
    );
    let _ = std::fs::remove_dir_all(&scratch);
    out
}

/// **THE FEMALE COMMITTED BODY** (wave CHAR1a.2) — the same wizard, the same
/// rig contract, a different set of measured proportions.
///
/// # Why a second FOLDER and not a second body in the first one
///
/// A body is generated FROM its rig: `manny::build_manny` places every joint
/// from [`inf_anim::BodyParams`] and the mesh is tubes around those joints, so a
/// female body needs a female-proportioned *skeleton* — and committing only a second
/// `.inf_mesh` beside the male's would be a female skin bound to a male bind
/// pose — the shape would be right and every pose would be wrong. The rig is
/// what differs, so the character is what is committed: eight assets, the same
/// **161 joint names in the same order**, which is the interchange contract this
/// wave exists to hold. `samples/starter-character/`'s own generator also
/// *deletes* anything it did not write, so a second body in that folder could
/// not survive a bless.
///
/// # Where the numbers come from — every one measured, none invented
///
/// Read off `SKM_Quinn`'s exported glTF (`skins[0].joints` composed to world
/// bind translations, and the mesh's own vertex bounds), 2026-09-05:
///
/// | `BodyParams` field | Quinn, measured | here |
/// |---|---|---|
/// | height (vertex bounds) | **1.8017 m** | 1.8017 |
/// | `hip_height_ratio` (`pelvis` y / h) | 0.9872 / 1.8017 = **0.5480** | 0.5480 |
/// | `shoulder_height_ratio` (`spine_05` y / h) | 1.3956 / 1.8017 = **0.7746** | 0.7746 |
/// | `head_height_ratio` (`head` y / h) | 1.6253 / 1.8017 = **0.9021** | 0.9021 |
/// | `shoulder_width_m` (`upperarm_l` ↔ `upperarm_r`) | **0.3211 m** | 0.3211 |
/// | `hip_width_m` (`thigh_l` ↔ `thigh_r`) | **0.2231 m** | 0.2231 |
/// | `arm_length_ratio` (shoulder→wrist chain / h) | 0.5319 / 1.8017 = **0.2952** | 0.2952 |
/// | `upper_limb_ratio` | arm **0.5094**, leg **0.5231** | 0.52 |
///
/// `upper_limb_ratio` is one parameter over two chains and cannot equal both, so
/// the default 0.52 is kept — it lies between the two measurements — rather than
/// picking a side and calling it measured.
///
/// **The shape difference is real and it is Quinn's**: against Manny's own
/// measurements (0.3802 m shoulders, 0.1994 m hips at 1.8054 m) she is 15.5 %
/// narrower across the shoulders and 12.1 % wider across the hips *as a fraction
/// of height*. That is the whole point of a second body.
///
/// **What is NOT changed here, and is a finding rather than a decision**:
/// `BodyParams::default()` — the male starter and every wizard default — carries
/// `arm_length_ratio: 0.42` where BOTH mannequins measure 0.30 (Manny 0.3048,
/// Quinn 0.2952). A 1.75 m person's shoulder-to-wrist is about 52 cm; 0.42
/// makes it 73 cm. It is the `HAND_OF_HEIGHT` defect one bone up and it is
/// carried with its numbers rather than fixed here, because moving a wizard
/// default re-blesses the male body, its three derived clips, its machine and
/// two pinned advisories — a bless this wave is not the place for.
pub fn starter_character_f_ids() -> crate::character::CharacterIds {
    let id = |n: u128| Some(inf_asset::AssetId(Uuid::from_u128(0x5C10_00B0 + n)));
    crate::character::CharacterIds {
        skeleton: id(0),
        material: id(1),
        mesh: id(2),
        idle: id(3),
        walk: id(4),
        run: id(5),
        machine: id(6),
        actor: id(7),
    }
}

/// What the female committed character is called — the prefix on every one of
/// its files.
pub const STARTER_CHARACTER_F_NAME: &str = "Starter_F";

/// The repo-root `samples/starter-character-f/` directory.
pub fn starter_character_f_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../samples/starter-character-f")
}

/// The spec the female body is built from — see [`starter_character_f_ids`] for
/// where every number was measured.
pub fn starter_character_f_spec() -> crate::character::CharacterSpec {
    crate::character::CharacterSpec {
        name: STARTER_CHARACTER_F_NAME.to_string(),
        params: inf_anim::BodyParams {
            height_m: 1.8017,
            hip_height_ratio: 0.5480,
            shoulder_height_ratio: 0.7746,
            head_height_ratio: 0.9021,
            shoulder_width_m: 0.3211,
            hip_width_m: 0.2231,
            arm_length_ratio: 0.2952,
            ..inf_anim::BodyParams::default()
        },
        ..Default::default()
    }
}

/// [`starter_character_files`] for the female body.
#[allow(clippy::type_complexity)]
pub fn starter_character_f_files() -> Result<(Vec<(String, Vec<u8>)>, Vec<String>), String> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let scratch = std::env::temp_dir().join(format!(
        "inf-starter-character-f-{}-{}",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&scratch);
    std::fs::create_dir_all(&scratch).map_err(|e| format!("scratch dir: {e}"))?;
    let out = starter_character_into(
        &scratch,
        &starter_character_f_spec(),
        &starter_character_f_ids(),
    );
    let _ = std::fs::remove_dir_all(&scratch);
    out
}

/// Write the committed female character.
pub fn write_starter_character_f() -> Result<(), String> {
    write_character_folder(
        &starter_character_f_dir(),
        starter_character_f_files()?,
        STARTER_CHARACTER_F_README,
    )
}

/// The half of [`starter_character_files`] that can fail, so the scratch
/// directory is removed on both paths.
#[allow(clippy::type_complexity)]
fn starter_character_into(
    scratch: &std::path::Path,
    spec: &crate::character::CharacterSpec,
    ids: &crate::character::CharacterIds,
) -> Result<(Vec<(String, Vec<u8>)>, Vec<String>), String> {
    let mut project =
        crate::assets::AssetProject::open(scratch).map_err(|e| format!("scratch project: {e}"))?;
    let build = crate::character::build_character_with_ids(&mut project, spec, ids)
        .map_err(|e| format!("the starter character does not build: {e}"))?;
    let dir = scratch.join(crate::character::CHARACTER_FOLDER);
    let mut files: Vec<(String, Vec<u8>)> = Vec::new();
    for entry in std::fs::read_dir(&dir).map_err(|e| format!("read scratch: {e}"))? {
        let entry = entry.map_err(|e| format!("read scratch: {e}"))?;
        if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        let bytes = std::fs::read(entry.path()).map_err(|e| format!("read {name}: {e}"))?;
        files.push((name, bytes));
    }
    // `read_dir` order is the filesystem's, and on NTFS that is not the sort
    // order (the P26 finding). The committed set is compared name by name, so it
    // is sorted here rather than at every reader.
    files.sort_by(|a, b| a.0.cmp(&b.0));
    Ok((files, build.warnings))
}

/// Write the committed starter character.
pub fn write_starter_character() -> Result<(), String> {
    write_character_folder(
        &starter_character_dir(),
        starter_character_files()?,
        STARTER_CHARACTER_README,
    )
}

/// Write one committed character folder: the wizard's files, its README, and
/// nothing else (wave CHAR1a.2 — two folders now share this).
#[allow(clippy::type_complexity)]
fn write_character_folder(
    dir: &std::path::Path,
    built: (Vec<(String, Vec<u8>)>, Vec<String>),
    readme: &str,
) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("mkdir: {e}"))?;
    let (files, _advisories) = built;
    // Anything the generator no longer writes goes, so a renamed asset does not
    // leave its predecessor behind for the sidecar scan to promote.
    for entry in std::fs::read_dir(dir).map_err(|e| format!("read samples dir: {e}"))? {
        let entry = entry.map_err(|e| format!("read samples dir: {e}"))?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if name == "README.md" || files.iter().any(|(n, _)| *n == name) {
            continue;
        }
        if entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
            std::fs::remove_file(entry.path()).map_err(|e| format!("remove {name}: {e}"))?;
        }
    }
    for (name, bytes) in &files {
        std::fs::write(dir.join(name), bytes).map_err(|e| format!("write {name}: {e}"))?;
    }
    std::fs::write(dir.join("README.md"), readme).map_err(|e| format!("write readme: {e}"))?;
    Ok(())
}

const STARTER_CHARACTER_README: &str = concat!(
    "# Starter character\n",
    "\n",
    "**The engine's committed starter character** - the exact eight assets the\n",
    "New Character wizard writes for its own default spec, on the 161-bone\n",
    "mannequin (`BodyPlan::Biped`).\n",
    "\n",
    "| file | what it is |\n",
    "|---|---|\n",
    "| `Starter.inf_skel` | the rig: 161 bones, role table, twist drivers, IK handles, hand cones and the grip catalogue |\n",
    "| `Starter_Body.inf_mesh` | the generated body, heat-weighted onto the rig |\n",
    "| `Starter_Skin.inf_mat` | a neutral matte dielectric, named as the body's material dependency |\n",
    "| `Starter_Idle.inf_anim`, `Starter_Walk.inf_anim`, `Starter_Run.inf_anim` | the generated, **derived** cycles |\n",
    "| `Starter_Locomotion.inf_sm` | the machine proposed from what the derivation measured, with the `Mask_AimOffset` upper-body profile on it |\n",
    "| `Starter_Locomotion.inf_sm.txt` | its reviewable text face |\n",
    "| `Starter_Controller.inf_act` | the Blueprint class the character binds |\n",
    "| `camera.toml` / `input.toml` | the camera table and the bindings |\n",
    "\n",
    "Two things ship it: `ProjectTemplate::starter_content` scaffolds it into\n",
    "every new 3D project, and `samples/island*/island.toml` names it under\n",
    "`[content]` so the island's hero is this character rather than a capsule.\n",
    "\n",
    "Generated - do not hand-edit. Regenerate with:\n",
    "\n",
    "```sh\n",
    "INF_BLESS_SAMPLES=1 cargo test -p inf-editor-core samples\n",
    "```\n",
);

const STARTER_CHARACTER_F_README: &str = concat!(
    "# Starter character (female)\n",
    "\n",
    "**The engine's second committed body** - the same New Character wizard, the\n",
    "same 161-bone rig and the same eight assets as `../starter-character/`, built\n",
    "from a different set of MEASURED proportions.\n",
    "\n",
    "Every number in the spec was read off `SKM_Quinn`'s exported glTF on\n",
    "2026-09-05 (joint world bind translations + the mesh's own vertex bounds):\n",
    "1.8017 m tall, hips at 0.5480, shoulders at 0.7746, head at 0.9021, a\n",
    "0.3211 m shoulder span and a 0.2231 m hip span. Against Manny's own\n",
    "measurements she is 15.5% narrower across the shoulders and 12.1% wider\n",
    "across the hips as a fraction of height.\n",
    "\n",
    "**Nothing from Unreal is in these files.** The mannequin was MEASURED; the\n",
    "geometry is this engine's own generator, vertex for vertex.\n",
    "\n",
    "Both bodies publish the same 161 joint names in the same order, which is\n",
    "what lets one clip play on either — see `char1a_gate.rs`.\n",
    "\n",
    "Generated - do not hand-edit. Regenerate with:\n",
    "\n",
    "```sh\n",
    "INF_BLESS_SAMPLES=1 cargo test -p inf-editor-core samples\n",
    "```\n",
);

#[cfg(test)]
mod tests {
    use super::*;

    /// **Every phase-19 lot stands on the terrain** (wave EMS1).
    ///
    /// The layout spreads one lot per archetype about the town centre, so the
    /// reach grows with the palette — and it has now outgrown the pad **twice**
    /// (7 → 10 at VEN1a, 10 → 14 here). The failure mode is silent both times:
    /// a lot with no ground under it places nothing at all, because `jobs_of`
    /// fails closed, so the sample simply comes out with two fewer buildings and
    /// no message anywhere. `PHASE19_LOT_PITCH`'s doc records the arithmetic;
    /// this is the arm that runs it, so the next wave to append an archetype
    /// finds out here rather than in a gate ten minutes into a player build.
    #[test]
    fn the_phase19_lots_all_stand_on_the_terrain() {
        let half = phase19_world_size() * 0.5;
        let n = inf_pcg::ArchetypeId::ALL.len();
        for i in 0..n {
            let p = phase19_lot_position(i);
            for (axis, v, extent) in [
                ("x", p.x, PHASE19_LOT_EXTENT.0),
                ("z", p.z, PHASE19_LOT_EXTENT.1),
            ] {
                let c = phase19_world_center();
                let centre = if axis == "x" { c.x } else { c.z };
                let reach = (v - centre).abs() + extent;
                assert!(
                    reach <= half,
                    "lot {i} ({}) reaches {reach:.1} m from the centre on {axis} \
                     against a {half:.1} m pad — it stands off the terrain, and \
                     a lot with no ground under it places NOTHING and says so \
                     nowhere",
                    inf_pcg::ArchetypeId::ALL[i].name()
                );
            }
        }
        // …and the lots do not stand inside one another, or the pitch has been
        // shrunk past the point where the town is a street.
        for i in 0..n.saturating_sub(2) {
            let gap = (phase19_lot_position(i + 2).x - phase19_lot_position(i).x).abs();
            assert!(
                gap > 2.0 * PHASE19_LOT_EXTENT.0,
                "lots {i} and {} on the same side of the road are {gap:.1} m \
                 apart and are {:.1} m wide",
                i + 2,
                2.0 * PHASE19_LOT_EXTENT.0
            );
        }
    }

    // ── Phase 23 workshop shape (P23.6) ────────────────────────────────────

    /// The recipe runs, and every op in it does real work. Here rather than only
    /// in the gate because a topology refusal is a *content* failure and this is
    /// where the content lives — the gate would report it as "the prop did not
    /// model", 10 minutes into a player build.
    #[test]
    fn the_workshop_recipe_models_a_prop() {
        let import = inf_dcc::from_mesh_asset(&phase23_baseline_mesh()).expect("baseline reads");
        assert_eq!(
            import.mesh.face_count(),
            12,
            "a cube arrives as 12 triangles"
        );
        assert_eq!(import.report.boundary_edges, 0, "closed");
        let mut session = inf_dcc::MeshSession::new(import.mesh);

        phase23_model_prop(&mut session).expect("the recipe models");
        assert_eq!(session.ops().len(), 3, "extrude, loop cut, bevel");
        assert!(
            session.mesh().face_count() > 20,
            "the recipe left {} faces — it did not build anything",
            session.mesh().face_count()
        );
        assert_eq!(inf_dcc::validate(session.mesh()), Ok(()));
        // Still a closed solid, and taller by the extrude (less the bevel's
        // chamfer, which cuts the rim rather than the cap).
        assert!(session
            .mesh()
            .vert_ids()
            .filter_map(|v| session.mesh().position(v))
            .any(|p| (p.y - (PHASE23_PROP_SIZE_M * 0.5 + PHASE23_EXTRUDE_M)).abs() < 1e-9));

        let report = phase23_unwrap_prop(&mut session).expect("the recipe unwraps");
        assert_eq!(report.flipped, 0, "a flat-charted prop folds nowhere");
        let stalled: Vec<usize> = report
            .charts
            .iter()
            .enumerate()
            .filter(|(_, c)| c.convergence >= PHASE23_CONVERGED)
            .map(|(i, _)| i)
            .collect();
        assert!(
            stalled.is_empty(),
            "charts {stalled:?} did not converge (worst {})",
            report.worst_convergence
        );
        assert!(report.seams > 0, "no seam was cut");
    }

    // ── Phase 21 cavern shape (P21.4) ──────────────────────────────────────

    /// **The sample really is the done-when sentence.** Asserted on the
    /// *generators* rather than on the committed bytes, so a change to the design
    /// fails here with a readable number instead of as a byte diff.
    #[test]
    fn the_cavern_sample_is_actually_a_cavern() {
        use glam::DVec2;
        let w = phase21_build();
        let (vol, terrain, spoil) = (&w.volume, &w.terrain, w.spoil);

        // Eight chunks — the authored rock body, and no more: a heap that
        // materialized chunks outside it would show up here first.
        assert_eq!(vol.chunk_count(), 8, "the rock body grew or shrank");

        // The cave has a MOUTH: the heightfield is holed where the first leg
        // crosses it, and un-holed on ground the dig never reached.
        assert!(terrain.has_holes());
        assert!(terrain.is_hole_at(DVec2::new(34.0, 44.0)), "no cave mouth");
        assert!(terrain.is_hole_at(DVec2::new(50.0, 38.0)), "no open pit");
        assert!(
            !terrain.is_hole_at(DVec2::new(100.0, 30.0)),
            "the knoll on the far side of the map was holed"
        );
        // …and the knoll is really there, so "the terrain is not flat" is content
        // rather than an assumption.
        let knoll = phase21_height(104.0, 32.0);
        assert!(
            knoll - phase21_height(104.0, 96.0) > 5.0,
            "the knoll is flat: {knoll}"
        );

        // The pit is open to the sky over its whole footprint (the P21.3 rule):
        // its top clears the highest ground it spans, its floor is the full depth
        // below the lowest.
        let (lo, hi) = phase21_pit_ground();
        assert!(hi > lo, "a pit on perfectly flat ground proves nothing");
        let (plo, phi) = phase21_pit_op().shape.aabb_m(0.0);
        assert!(phi.y > hi, "the pit has a lid of surviving hillside");
        assert_eq!(plo.y, lo - PHASE21_PIT_DEPTH_M);
        assert_eq!(plo.y, phase21_pit_floor_y());

        // The underground room is UNDER the pit and REACHABLE: the topmost voxel
        // surface over its centre is its own floor, which means the column above
        // it is continuous void all the way to the sky.
        let (px, pz) = phase21_room_probe_xz();
        assert_eq!(vol.surface_y_at(px, pz), Some(PHASE21_ROOM_FLOOR_Y));
        assert!(PHASE21_ROOM_CEILING_Y < phase21_pit_floor_y());
        // The combined query agrees, because the pit holed the ground above it.
        assert_eq!(
            inf_voxel::ground_height_at(
                &[(terrain, DVec3::ZERO)],
                &std::collections::BTreeMap::from([(0u8, vol.clone())]),
                px,
                pz,
            ),
            Some(PHASE21_ROOM_FLOOR_Y)
        );

        // CONSERVATION: the heap holds exactly what the pit removed, **per
        // material**, with no shortfall and no bulking factor. Both sides come
        // from the cuts that produced them, not from each other.
        assert_eq!(spoil.shortfall, [0; inf_voxel::MATERIAL_COUNT]);
        assert!(spoil.total_placed() > 0, "nothing was displaced");
        assert!(spoil.is_exact(), "the heap did not hold the whole pit");
        assert_eq!(
            spoil.placed, w.pit_removed,
            "the heap and the pit disagree about how much soil there is"
        );

        // The borer's drift is in SOLID ROCK for its whole run — a trace of
        // carving, not of a script reporting zero — and its crown is ABOVE the
        // ground for its whole run, so every tick opens the heightfield too.
        let (bx, by, bz) = PHASE21_BORE_START;
        let end_x = bx + PHASE21_BORE_STEPS as f64 * PHASE21_BORE_STEP_M;
        let mut steps = 0;
        let mut x = bx;
        while x <= end_x {
            assert!(
                vol.is_solid_at(DVec3::new(x, by, bz)),
                "the bore centre is in air at x = {x}"
            );
            let ground = phase21_height(x, bz);
            assert!(
                by + PHASE21_BORE_RADIUS_M > ground,
                "at x = {x} the bore's crown ({}) is under the ground ({ground}) — it \
                 cuts a sealed tube and never opens a mouth",
                by + PHASE21_BORE_RADIUS_M
            );
            assert!(
                by < ground,
                "at x = {x} the bore's centre is above grade, so it is cutting air"
            );
            steps += 1;
            x += 1.0;
        }
        assert!(steps > 10, "the drift is too short to prove anything");

        // The boulder starts ABOVE the rock (so it falls onto it) and sits over
        // the drift (so the borer eventually takes that rock away). Both halves,
        // because a boulder resting beside the drift never witnesses anything.
        let (blx, blz) = PHASE21_BOULDER_XZ;
        assert!(
            PHASE21_BOULDER_START_Y - PHASE21_BOULDER_HALF_M > phase21_height(blx, blz),
            "the boulder starts inside the ground"
        );
        assert!(
            blx > bx && blx < end_x,
            "the boulder at x = {blx} is not over the drift [{bx}, {end_x}]"
        );
        assert!(
            (blz - bz).abs() < PHASE21_BORE_RADIUS_M,
            "the boulder is {} m off the drift's centreline",
            (blz - bz).abs()
        );
        assert!(
            vol.is_solid_at(DVec3::new(blx, phase21_height(blx, blz) - 0.5, blz)),
            "there is no rock under the boulder to hold it up"
        );
        // …and it starts ABOVE the trench crown, so its first landing is on rock
        // the borer has not reached yet. Starting inside the trench's future void
        // would make "it fell when the rock went" unobservable: it would already
        // be on the floor.
        assert!(
            PHASE21_BOULDER_START_Y - PHASE21_BOULDER_HALF_M > by + PHASE21_BORE_RADIUS_M,
            "the boulder starts inside the trench the borer will open"
        );
        // …and it stays inside the authored chunks.
        let east = (PHASE21_ROCK_CHUNKS_XZ.1 + 1) as f64 * inf_voxel::CHUNK_DIM as f64;
        assert!(end_x + PHASE21_BORE_RADIUS_M < east, "{end_x} vs {east}");
    }

    #[test]
    fn coyote_class_round_trips() {
        let class = coyote_class();
        // The committed `.inf_act` encoding round-trips exactly.
        let bytes = encode_actor(&class).unwrap();
        assert_eq!(decode_actor(&bytes).unwrap(), class);
        // The handlers the Simulate loop fires are present.
        assert!(class.handler(&EventKind::Tick).is_some());
        assert!(class.handler(&EventKind::BeginPlay).is_some());
    }

    #[test]
    fn platformer_scene_saves_and_reloads_byte_identical() {
        // The P3 discipline applied to the full 2D content: save→load→save is
        // byte-identical.
        let doc = platformer_scene();
        let file1 = crate::scene::serialize::to_scene_file(&doc);
        let bytes1 = crate::scene::serialize::encode(&file1).unwrap();

        let mut doc2 = SceneDoc::new();
        crate::scene::serialize::apply_to_doc(
            &mut doc2,
            &crate::scene::serialize::decode(&bytes1).unwrap(),
        );
        let bytes2 =
            crate::scene::serialize::encode(&crate::scene::serialize::to_scene_file(&doc2))
                .unwrap();
        assert_eq!(
            bytes1, bytes2,
            "platformer scene must round-trip byte-identically"
        );

        // The player carries every physics + sprite component.
        let file = crate::scene::serialize::to_scene_file(&doc2);
        let player = file
            .entities
            .iter()
            .find(|r| r.guid == PLAYER_GUID)
            .expect("player present");
        assert!(player.sprite.is_some());
        // The reloaded scene keeps the tilemap ground strip.
        let tiles = file
            .entities
            .iter()
            .find(|r| r.guid == GROUND_TILES_GUID)
            .unwrap();
        assert_eq!(tiles.tilemap.as_ref().unwrap().get_tile(0, 0), 1);
    }

    #[test]
    fn firstperson_scene_saves_and_reloads_byte_identical() {
        let doc = firstperson_scene();
        let bytes1 =
            crate::scene::serialize::encode(&crate::scene::serialize::to_scene_file(&doc)).unwrap();
        let mut doc2 = SceneDoc::new();
        crate::scene::serialize::apply_to_doc(
            &mut doc2,
            &crate::scene::serialize::decode(&bytes1).unwrap(),
        );
        let bytes2 =
            crate::scene::serialize::encode(&crate::scene::serialize::to_scene_file(&doc2))
                .unwrap();
        assert_eq!(
            bytes1, bytes2,
            "first-person template scene must round-trip byte-identically"
        );
        // The player, camera, ground, and sun all survive (4 entities).
        assert_eq!(
            crate::scene::serialize::to_scene_file(&doc2).entities.len(),
            4
        );
    }

    #[test]
    fn hybrid_scene_saves_and_reloads_byte_identical() {
        let doc = hybrid_scene();
        let bytes1 =
            crate::scene::serialize::encode(&crate::scene::serialize::to_scene_file(&doc)).unwrap();
        let mut doc2 = SceneDoc::new();
        crate::scene::serialize::apply_to_doc(
            &mut doc2,
            &crate::scene::serialize::decode(&bytes1).unwrap(),
        );
        let bytes2 =
            crate::scene::serialize::encode(&crate::scene::serialize::to_scene_file(&doc2))
                .unwrap();
        assert_eq!(
            bytes1, bytes2,
            "hybrid template scene must round-trip byte-identically"
        );
        // The two billboard sprites survive with their modes.
        let file = crate::scene::serialize::to_scene_file(&doc2);
        let sph = file
            .entities
            .iter()
            .find(|r| r.guid == HYBRID_SPRITE_SPHERE_GUID)
            .unwrap();
        assert_eq!(
            sph.sprite.as_ref().unwrap().billboard,
            BillboardMode::Spherical
        );
    }

    #[test]
    fn vgeom_demo_saves_and_reloads_byte_identical() {
        let doc = vgeom_demo_scene();
        let bytes1 =
            crate::scene::serialize::encode(&crate::scene::serialize::to_scene_file(&doc)).unwrap();
        assert_eq!(
            bytes1[0],
            crate::scene::serialize::SCHEMA_VERSION as u8,
            "vgeom-demo writes at the current schema"
        );

        let mut doc2 = SceneDoc::new();
        crate::scene::serialize::apply_to_doc(
            &mut doc2,
            &crate::scene::serialize::decode(&bytes1).unwrap(),
        );
        let bytes2 =
            crate::scene::serialize::encode(&crate::scene::serialize::to_scene_file(&doc2))
                .unwrap();
        assert_eq!(
            bytes1, bytes2,
            "vgeom-demo scene must round-trip byte-identically"
        );

        // Every instance carries a MeshRef.asset pointing at the shared mesh GUID.
        let file = crate::scene::serialize::to_scene_file(&doc2);
        let inst: Vec<_> = file
            .entities
            .iter()
            .filter(|r| r.mesh.as_ref().and_then(|m| m.asset).is_some())
            .collect();
        assert_eq!(inst.len(), VGEOM_DEMO_GRID * VGEOM_DEMO_GRID);
        assert!(inst
            .iter()
            .all(|r| r.mesh.as_ref().unwrap().asset == Some(VGEOM_DEMO_MESH_GUID)));
    }

    #[test]
    fn vgeom_demo_exceeds_10m_source_triangles() {
        // The dense mesh's triangle count times the instance grid is the source
        // triangle budget the phase gate demands (≥ 10M).
        let mesh = vgeom_demo_mesh();
        let per = mesh.triangle_count() as u64;
        assert_eq!(per, (2 * VGEOM_DEMO_MESH_N * VGEOM_DEMO_MESH_N) as u64);
        let total = per * (VGEOM_DEMO_GRID * VGEOM_DEMO_GRID) as u64;
        assert_eq!(total, vgeom_demo_source_triangles());
        assert!(
            total >= 10_000_000,
            "gate needs 10M+ source triangles, got {total}"
        );
        // Above the cook's default vmesh derivation threshold.
        assert!(per >= 2048);
    }

    /// **EVERY LEVEL THE EDITOR BOOTS HAS SOMEBODY IN IT** (wave GTA1).
    ///
    /// Asked through the runtime's own door — `inf_ecs::movement::camera_subject`
    /// — rather than by counting components, because that function is what
    /// decides the answer at run time: it returns `None` on a level with no
    /// `player_controlled` character, and every host then keeps its own view. So
    /// pressing Play on a fresh project showed an author their furniture from an
    /// overhead orthographic camera, with nothing that responded to a key.
    ///
    /// The second half is what stops this from being a component-presence test:
    /// each of the three 3D starter levels must name the **committed** starter
    /// character's assets, because those are the seventeen files
    /// `ProjectTemplate::starter_content` scaffolds beside them. A pawn wired to
    /// minted guids is a pawn that draws as a placeholder cube in the project it
    /// ships in.
    #[test]
    fn every_starter_level_has_a_player_controlled_character() {
        let mut boot = SceneDoc::new();
        crate::scene::demo::build(&mut boot);
        let ids = starter_character_ids();
        let want = |id: Option<inf_asset::AssetId>| id.expect("fixed").0;

        for (name, doc, character) in [
            (
                "the editor's boot document",
                boot,
                crate::scene::demo::DEMO_PLAYER_GUID,
            ),
            ("blank-3d", blank3d_scene(), BLANK3D_PLAYER_GUID),
            ("hybrid-2.5d", hybrid_scene(), HYBRID_PLAYER_GUID),
            ("first-person", firstperson_scene(), FP_PLAYER_GUID),
        ] {
            let subject = inf_ecs::movement::camera_subject(doc.world());
            assert_eq!(
                subject,
                Some(character),
                "{name} has no player-controlled character, so Play falls back to \
                 the overhead camera and nothing in it moves"
            );
            // The first-person template's pawn is a bare capsule on purpose — a
            // first-person player has no visible body — so the asset check is for
            // the three that spawn the committed character.
            if character == FP_PLAYER_GUID {
                continue;
            }
            let world = doc.world();
            let e = world.entity_of(character).expect("the character exists");
            let sm = world
                .world()
                .get::<inf_ecs::components::SkeletalMesh>(e)
                .expect("the character has a skeletal mesh");
            assert_eq!(sm.skeleton, Some(want(ids.skeleton)), "{name}: wrong rig");
            assert_eq!(sm.mesh, Some(want(ids.mesh)), "{name}: wrong body");
            let machine = world
                .world()
                .get::<inf_ecs::components::AnimStateMachine>(e)
                .and_then(|m| m.sm);
            assert_eq!(machine, Some(want(ids.machine)), "{name}: wrong machine");
            let class = world.world().get::<ActorClass>(e).map(|c| c.0);
            assert_eq!(class, Some(want(ids.actor)), "{name}: wrong controller");
        }
    }

    /// **AND THE GROUND HOLDS IT UP** (wave GTA1 audit).
    ///
    /// The arm above asks `camera_subject` and stops there, which is exactly
    /// what a pawn with nothing under it passes. All four of these ground planes
    /// were a `MeshRef` and a `Material` and nothing else, and
    /// `inf_physics::d3::ecs`'s sync walks the entities carrying a body **or** a
    /// collider and `continue`s on the rest — so the ground reached the solver as
    /// nothing, while the character the wave placed on it is gravity-driven. The
    /// first-person template's own README, written in the same wave, says *"press
    /// Play and WASD moves the Player"*; what it moved was a body in free fall.
    ///
    /// Measured on this arm, one second of Simulate at 60 Hz: **4.9868 m of
    /// fall on every one of the four** with no collider under the plane, still
    /// accelerating toward `terminal_velocity_mps` (53), against **−0.0201 m**
    /// with the slab under it — a 2 cm *rise*, which is the kinematic
    /// controller's own ground snap and the reason the standing bound is 5 cm
    /// rather than zero.
    ///
    /// The **control is the half that makes this an assertion about the ground**
    /// rather than about gravity: the same document with every non-pawn collider
    /// taken back off must drop the pawn, measured in the same run. Without it a
    /// level with no gravity at all would pass, and so would one whose physics
    /// bridge never ran.
    #[test]
    fn every_starter_level_gives_its_pawn_something_to_stand_on() {
        use inf_ecs::components::Collider3D;

        /// The fixed step the templates author, and one second of it.
        const HZ: f64 = 60.0;
        const STEPS: u32 = 60;

        fn pawn_y(doc: &SceneDoc, guid: Uuid) -> f64 {
            let e = doc.entity_of(guid).expect("the pawn exists");
            doc.world()
                .world()
                .get::<Transform>(e)
                .expect("the pawn has a transform")
                .translation
                .y
        }

        /// How far the level's pawn falls in one second, in metres (positive =
        /// down). `strip_ground` removes every collider that is not the pawn's,
        /// which is the control.
        fn fall_of(doc: &mut SceneDoc, strip_ground: bool) -> f64 {
            let subject =
                inf_ecs::movement::camera_subject(doc.world()).expect("the level has a pawn");
            if strip_ground {
                for guid in doc.order().to_vec() {
                    if guid == subject {
                        continue;
                    }
                    if let Some(e) = doc.entity_of(guid) {
                        doc.world_mut()
                            .world_mut()
                            .entity_mut(e)
                            .remove::<Collider3D>();
                    }
                }
            }
            let gravity = crate::simulate::SimSession::gravity_of(doc);
            // ANTI-VACUITY: a document with no gravity would pass the standing
            // assertion for the wrong reason, and the control would then measure
            // nothing at all.
            assert!(
                gravity.d3.y < -1.0,
                "the document authors no 3D gravity ({}), so neither half of this \
                 arm is measuring anything",
                gravity.d3.y
            );
            let before = pawn_y(doc, subject);
            let mut sim =
                crate::simulate::SimSession::enter_with_gravity(doc, Vec::new(), gravity, HZ);
            for _ in 0..STEPS {
                sim.step_once(doc, crate::simulate::SimInput::default());
            }
            before - pawn_y(doc, subject)
        }

        let boot = || {
            let mut d = SceneDoc::new();
            crate::scene::demo::build(&mut d);
            d
        };
        for (name, make) in [
            ("the editor's boot document", &boot as &dyn Fn() -> SceneDoc),
            ("blank-3d", &blank3d_scene),
            ("hybrid-2.5d", &hybrid_scene),
            ("first-person", &firstperson_scene),
        ] {
            let stood = fall_of(&mut make(), false);
            let control = fall_of(&mut make(), true);
            eprintln!("{name}: falls {stood:.4} m on its ground, {control:.4} m without it");
            assert!(
                stood.abs() < 0.05,
                "{name}: its pawn fell {stood:.4} m in one second — the ground under \
                 it has no collider, so Play drops the player through the world"
            );
            assert!(
                control > 1.0,
                "{name}: the control fell only {control:.4} m with every collider \
                 removed, so the standing assertion above is not measuring the ground"
            );
        }
    }

    /// Regenerate the committed files under `INF_BLESS_SAMPLES=1`; otherwise
    /// assert the committed bytes still match the generators (fixture lock).
    #[test]
    fn committed_sample_matches_generators() {
        if std::env::var("INF_BLESS_SAMPLES").is_ok() {
            write_sample().expect("regenerate sample");
            write_hybrid_template().expect("regenerate hybrid template");
            write_firstperson_template().expect("regenerate first-person template");
            write_blank3d_template().expect("regenerate blank-3d template");
            write_platformer_template().expect("regenerate 2d-platformer template");
            write_terrain_demo().expect("regenerate terrain demo");
            write_character_demo().expect("regenerate character demo");
            write_physics_playground().expect("regenerate physics playground");
            write_vgeom_demo().expect("regenerate vgeom demo");
            write_streamed_terrain().expect("regenerate streamed terrain");
            write_partitioned_world().expect("regenerate partitioned world");
            write_phase16_world().expect("regenerate phase16 world");
            write_phase18_scatter().expect("regenerate phase18 scatter");
            write_phase19_town().expect("regenerate phase19 town");
            write_phase20_coastal().expect("regenerate phase20 coastal");
            write_phase21_cavern().expect("regenerate phase21 cavern");
            write_phase22_playground().expect("regenerate phase22 playground");
            write_phase23_workshop().expect("regenerate phase23 workshop");
            write_phase29_locomotion().expect("regenerate phase29 locomotion");
            // Before the island: the island's hero IS this character, and its
            // levels name these eight GUIDs.
            write_starter_character().expect("regenerate the starter character");
            // The FEMALE body (wave CHAR1a.2): the second committed body on the
            // same rig contract. Beside the male rather than after the island,
            // for the same reason — a project that scaffolds one scaffolds both.
            write_starter_character_f().expect("regenerate the female starter character");
            // …and before the island for the same reason: the island's four
            // TerrainLayers name four of these material GUIDs (TER2a clause 3).
            crate::ground::write_ground_library(&ground_dir())
                .expect("regenerate the ground library");
            // …and the three things that stand on it, whose GUIDs the island's
            // committed `.inf_pcg` names (TER2a clause 5).
            crate::cover::write_cover_library(&ground_dir()).expect("regenerate the cover library");
            // …and before the island for the third time (wave I8a): every
            // settlement block's `PcgVolume.graph` names one of these seven
            // zone GUIDs, so a level written against a stale library is 172
            // blocks bound to nothing.
            crate::settlement::write_settlement_library(&crate::settlement::settlement_dir())
                .expect("regenerate the settlement zone library");
            write_city().expect("regenerate the island city");
            write_gameplay().expect("regenerate the island gameplay fixture");
            crate::heist::write_heist().expect("regenerate the harbour heist mission");
            crate::island::write_island_levels().expect("regenerate the island levels");
            eprintln!("samples: regenerated {}", sample_dir().display());
            return;
        }
        let dir = sample_dir();
        let lvl = dir.join("Platformer.inf_lvl");
        let act = dir.join("Coyote.inf_act");
        if !lvl.exists() || !act.exists() {
            // First run before blessing: don't fail CI spuriously.
            eprintln!("SKIP: committed sample not present yet ({})", dir.display());
            return;
        }
        let want_lvl = crate::scene::serialize::encode(&crate::scene::serialize::to_scene_file(
            &platformer_scene(),
        ))
        .unwrap();
        let got_lvl = std::fs::read(&lvl).unwrap();
        assert_eq!(
            got_lvl, want_lvl,
            "committed .inf_lvl drifted from the generator"
        );

        let want_act = encode_actor(&coyote_class()).unwrap();
        let got_act = std::fs::read(&act).unwrap();
        assert_eq!(
            got_act, want_act,
            "committed .inf_act drifted from the generator"
        );

        // First-person template lock: the committed `.inf_lvl` still matches the
        // generator (skips gracefully before the first bless).
        let fpdir = firstperson_template_dir();
        let fplvl = fpdir.join("FirstPerson.inf_lvl");
        if fplvl.exists() {
            let want_lvl = crate::scene::serialize::encode(
                &crate::scene::serialize::to_scene_file(&firstperson_scene()),
            )
            .unwrap();
            assert_eq!(
                std::fs::read(&fplvl).unwrap(),
                want_lvl,
                "committed first-person .inf_lvl drifted from the generator"
            );
        }

        // Blank-3D template lock (IB-7: the default template ships a boot level).
        let blvl = blank3d_template_dir().join("Blank.inf_lvl");
        if blvl.exists() {
            let want_lvl = crate::scene::serialize::encode(
                &crate::scene::serialize::to_scene_file(&blank3d_scene()),
            )
            .unwrap();
            assert_eq!(
                std::fs::read(&blvl).unwrap(),
                want_lvl,
                "committed blank-3d .inf_lvl drifted from the generator"
            );
        }

        // 2D-platformer template lock — level AND the actor its `ActorClass`
        // binds, because a template that scaffolds one without the other
        // scaffolds a player that does not move.
        let ptdir = platformer_template_dir();
        let ptlvl = ptdir.join("Platformer.inf_lvl");
        let ptact = ptdir.join("Coyote.inf_act");
        if ptlvl.exists() && ptact.exists() {
            let want_lvl = crate::scene::serialize::encode(
                &crate::scene::serialize::to_scene_file(&platformer_scene()),
            )
            .unwrap();
            assert_eq!(
                std::fs::read(&ptlvl).unwrap(),
                want_lvl,
                "committed 2d-platformer template .inf_lvl drifted from the generator"
            );
            assert_eq!(
                std::fs::read(&ptact).unwrap(),
                encode_actor(&coyote_class()).unwrap(),
                "committed 2d-platformer template .inf_act drifted from the generator"
            );
            // The template's level payload is the sample's payload — one
            // generator, two blessed copies — while the two sidecars name
            // DIFFERENT level GUIDs so both can live in one asset database.
            assert_eq!(
                std::fs::read(&ptlvl).unwrap(),
                std::fs::read(dir.join("Platformer.inf_lvl")).unwrap(),
                "the template and the sample no longer share one generator"
            );
            assert_ne!(
                PLATFORMER_TEMPLATE_LEVEL_GUID, LEVEL_GUID,
                "the template level and the sample level must not share a GUID"
            );
            // …and the ACTOR SIDECAR names the GUID the level's `ActorClass`
            // binds. The bytes above are locked to the generator; the sidecar was
            // not, and the I1 audit measured that its GUID could be changed to
            // anything with nothing in the workspace going red — which scaffolds
            // a project whose player does not move.
            let side =
                inf_asset::AssetSidecar::load(&ptact).expect("the template actor has a sidecar");
            assert_eq!(
                side.guid,
                inf_asset::AssetId(COYOTE_ASSET_GUID),
                "the template's Coyote.inf_act.toml names a different asset from \
                 the one `platformer_scene`'s ActorClass binds"
            );
            assert_eq!(
                side.content_hash,
                inf_asset::ContentHash::of(&encode_actor(&coyote_class()).unwrap()),
                "the template actor's sidecar hash does not describe the payload \
                 beside it"
            );
        }

        // Terrain-demo lock: the committed `.inf_lvl` + `.inf_pcg` still match the
        // generators (skips gracefully before the first bless).
        let tdir = terrain_demo_dir();
        let tlvl = tdir.join("TerrainDemo.inf_lvl");
        let tpcg = tdir.join("Scatter.inf_pcg");
        if tlvl.exists() && tpcg.exists() {
            let want_lvl = crate::scene::serialize::encode(
                &crate::scene::serialize::to_scene_file(&terrain_demo_scene()),
            )
            .unwrap();
            assert_eq!(
                std::fs::read(&tlvl).unwrap(),
                want_lvl,
                "committed terrain-demo .inf_lvl drifted from the generator"
            );
            assert_eq!(
                std::fs::read(&tpcg).unwrap(),
                terrain_demo_pcg_payload().encode().unwrap(),
                "committed terrain-demo .inf_pcg drifted from the generator"
            );
        }

        // Character-demo lock: the committed `.inf_lvl` + anim assets still match
        // the generators (skips gracefully before the first bless).
        let cdir = character_demo_dir();
        let clvl = cdir.join("Character.inf_lvl");
        if clvl.exists() {
            let want_lvl = crate::scene::serialize::encode(
                &crate::scene::serialize::to_scene_file(&character_demo_scene()),
            )
            .unwrap();
            assert_eq!(
                std::fs::read(&clvl).unwrap(),
                want_lvl,
                "committed character-demo .inf_lvl drifted from the generator"
            );
            assert_eq!(
                std::fs::read(cdir.join("Locomotion.inf_sm")).unwrap(),
                inf_asset::encode(&inf_anim::StateMachineAsset::new(
                    character_demo_state_machine(),
                    Some(*CHARACTER_DEMO_SKELETON_GUID.as_bytes()),
                ))
                .unwrap(),
                "committed character-demo .inf_sm drifted from the generator"
            );
            // P24.1: the `.inf_skel` too. It is the ONE committed skeleton in the
            // tree, so it is the only thing that would have caught the
            // `SkeletonAsset` v1 → v2 wire change by *bytes* — and it was unlocked
            // while its `.inf_lvl` and `.inf_sm` siblings were not.
            assert_eq!(
                std::fs::read(cdir.join("Character.inf_skel")).unwrap(),
                inf_asset::encode(&inf_anim::SkeletonAsset::new(character_demo_skeleton()))
                    .unwrap(),
                "committed character-demo .inf_skel drifted from the generator"
            );
            // **P29.2: the three `.inf_anim` clips, which this lock had never
            // covered.** Exactly the gap the note above records for the
            // `.inf_skel` — its siblings were locked and it was not — met a
            // second time, in the other direction: `.inf_anim` moved to v2 in
            // P29.2 (the clip channel model) and the three committed clips were
            // stale v1 bytes that no assertion in this repository could see.
            // Locked now, so the next schema move is a red test rather than a
            // silent `SchemaTooOld` in whatever loads them first.
            let skel_bytes = *CHARACTER_DEMO_SKELETON_GUID.as_bytes();
            for (file, clip) in [
                ("Idle.inf_anim", character_demo_idle_clip()),
                ("Run.inf_anim", character_demo_run_clip()),
                ("Jump.inf_anim", character_demo_jump_clip()),
            ] {
                assert_eq!(
                    std::fs::read(cdir.join(file)).unwrap(),
                    inf_asset::encode(&inf_anim::AnimClipAsset::new(clip, Some(skel_bytes)))
                        .unwrap(),
                    "committed character-demo {file} drifted from the generator"
                );
            }
        }

        // Physics-playground lock: the committed v6 `.inf_lvl` + the two
        // `.inf_audio` clips still match the generators (skips before first bless).
        let pdir = physics_playground_dir();
        let plvl = pdir.join("Playground.inf_lvl");
        if plvl.exists() {
            let want_lvl = crate::scene::serialize::encode(
                &crate::scene::serialize::to_scene_file(&physics_playground_scene()),
            )
            .unwrap();
            assert_eq!(
                std::fs::read(&plvl).unwrap(),
                want_lvl,
                "committed physics-playground .inf_lvl drifted from the generator"
            );
            let want_audio = inf_asset::encode(&playground_audio_asset()).unwrap();
            assert_eq!(
                std::fs::read(pdir.join("Spinner.inf_audio")).unwrap(),
                want_audio,
                "committed Spinner.inf_audio drifted from the generator"
            );
            assert_eq!(
                std::fs::read(pdir.join("Sensor.inf_audio")).unwrap(),
                want_audio,
                "committed Sensor.inf_audio drifted from the generator"
            );
        }

        // Vgeom-demo lock: the committed `.inf_lvl` + dense `.inf_mesh` still match
        // the generators (skips gracefully before the first bless).
        let vdir = vgeom_demo_dir();
        let vlvl = vdir.join("VgeomDemo.inf_lvl");
        let vmesh = vdir.join("Dense.inf_mesh");
        if vlvl.exists() && vmesh.exists() {
            let want_lvl = crate::scene::serialize::encode(
                &crate::scene::serialize::to_scene_file(&vgeom_demo_scene()),
            )
            .unwrap();
            assert_eq!(
                std::fs::read(&vlvl).unwrap(),
                want_lvl,
                "committed vgeom-demo .inf_lvl drifted from the generator"
            );
            let want_mesh = inf_asset::encode(&vgeom_demo_mesh()).unwrap();
            assert_eq!(
                std::fs::read(&vmesh).unwrap(),
                want_mesh,
                "committed Dense.inf_mesh drifted from the generator"
            );
        }

        // Partitioned-world lock (P16.5): only the `.inf_lvl` is committed — the
        // `.inf_part` is DERIVED by the cook (like `.inf_vmesh`).
        let pwdir = partitioned_world_dir();
        let pwlvl = pwdir.join("PartitionedWorld.inf_lvl");
        if pwlvl.exists() {
            let want_lvl = crate::scene::serialize::encode(
                &crate::scene::serialize::to_scene_file(&partitioned_world_scene()),
            )
            .unwrap();
            assert_eq!(
                std::fs::read(&pwlvl).unwrap(),
                want_lvl,
                "committed partitioned-world .inf_lvl drifted from the generator"
            );
        }

        // Streamed-terrain lock (P16.3b2): only the `.inf_lvl` is committed — the
        // `.inf_terrain` is generated into a fixture's Content dir by the gate.
        let sdir = streamed_terrain_dir();
        let slvl = sdir.join("StreamedTerrain.inf_lvl");
        if slvl.exists() {
            let want_lvl = crate::scene::serialize::encode(
                &crate::scene::serialize::to_scene_file(&streamed_terrain_scene()),
            )
            .unwrap();
            assert_eq!(
                std::fs::read(&slvl).unwrap(),
                want_lvl,
                "committed streamed-terrain .inf_lvl drifted from the generator"
            );
        }

        // Phase 16 gate lock (P16.6): only the `.inf_lvl` is committed — the
        // `.inf_terrain` is imported into a fixture's Content dir by the gate, and
        // the `.inf_part` is derived by the cook.
        let p16dir = phase16_world_dir();
        let p16lvl = p16dir.join("Phase16World.inf_lvl");
        if p16lvl.exists() {
            let want_lvl = crate::scene::serialize::encode(
                &crate::scene::serialize::to_scene_file(&phase16_world_scene()),
            )
            .unwrap();
            assert_eq!(
                std::fs::read(&p16lvl).unwrap(),
                want_lvl,
                "committed phase16-world .inf_lvl drifted from the generator"
            );
        }

        // Phase 18 gate lock (P18.6): the `.inf_lvl` + the `.inf_pcg` are
        // committed. The dense `.inf_mesh` the slabs reference is NOT — it is
        // vgeom-demo's, shared by GUID, and locked by that sample's own arm above.
        let p18dir = phase18_scatter_dir();
        let p18lvl = p18dir.join("Phase18Scatter.inf_lvl");
        let p18pcg = p18dir.join("GroundCover.inf_pcg");
        if p18lvl.exists() && p18pcg.exists() {
            let want_lvl = crate::scene::serialize::encode(
                &crate::scene::serialize::to_scene_file(&phase18_scatter_scene()),
            )
            .unwrap();
            assert_eq!(
                std::fs::read(&p18lvl).unwrap(),
                want_lvl,
                "committed phase18-scatter .inf_lvl drifted from the generator"
            );
            assert_eq!(
                std::fs::read(&p18pcg).unwrap(),
                phase18_scatter_pcg_payload().encode().unwrap(),
                "committed phase18-scatter .inf_pcg drifted from the generator"
            );
        }

        // Phase 19 gate lock (P19.5): the `.inf_lvl`, the biome set, the roadside
        // graph and all seven lot graphs are committed.
        let p19dir = phase19_town_dir();
        let p19lvl = p19dir.join("Phase19Town.inf_lvl");
        if p19lvl.exists() {
            let want_lvl = crate::scene::serialize::encode(
                &crate::scene::serialize::to_scene_file(&phase19_town_scene()),
            )
            .unwrap();
            assert_eq!(
                std::fs::read(&p19lvl).unwrap(),
                want_lvl,
                "committed phase19-town .inf_lvl drifted from the generator"
            );
            assert_eq!(
                std::fs::read(p19dir.join("Town.inf_biomes")).unwrap(),
                inf_asset::encode(&phase19_biome_set()).unwrap(),
                "committed phase19-town .inf_biomes drifted from the generator"
            );
            assert_eq!(
                std::fs::read(p19dir.join("Roadside.inf_pcg")).unwrap(),
                phase19_road_payload().encode().unwrap(),
                "committed phase19-town roadside .inf_pcg drifted from the generator"
            );
            for (i, id) in inf_pcg::ArchetypeId::ALL.into_iter().enumerate() {
                let path = p19dir.join(format!("Lot{}.inf_pcg", id.name()));
                assert_eq!(
                    std::fs::read(&path).unwrap(),
                    phase19_lot_payload(i).encode().unwrap(),
                    "committed phase19-town {} .inf_pcg drifted from the generator",
                    id.name()
                );
            }
        }

        // Phase 20 coastal lock (P20.4): the level AND the swimmer's blueprint.
        // The `.inf_act` is included because the level's `ActorClass` binding is
        // a GUID — a drifted class would still bind, and the swimmer would
        // silently stop swimming.
        let p20dir = phase20_coastal_dir();
        let p20lvl = p20dir.join("Phase20Coastal.inf_lvl");
        if p20lvl.exists() {
            let want_lvl = crate::scene::serialize::encode(
                &crate::scene::serialize::to_scene_file(&phase20_coastal_scene()),
            )
            .unwrap();
            assert_eq!(
                std::fs::read(&p20lvl).unwrap(),
                want_lvl,
                "committed phase20-coastal .inf_lvl drifted from the generator"
            );
            assert_eq!(
                std::fs::read(p20dir.join("Swimmer.inf_act")).unwrap(),
                encode_actor(&phase20_swimmer_class()).unwrap(),
                "committed phase20-coastal .inf_act drifted from the generator"
            );
        }

        // Phase 21 cavern lock (P21.4): the level, the borer's blueprint, AND
        // both derived assets. The `.inf_voxel` and `.inf_terrain` are locked
        // here and not merely regenerated because the workings ARE the sample —
        // a drifted carve is a different cave, a different hole mask and a
        // different spoil heap, all of which the level itself is silent about.
        let p21dir = phase21_cavern_dir();
        let p21lvl = p21dir.join("Phase21Cavern.inf_lvl");
        if p21lvl.exists() {
            let want_lvl = crate::scene::serialize::encode(
                &crate::scene::serialize::to_scene_file(&phase21_cavern_scene()),
            )
            .unwrap();
            assert_eq!(
                std::fs::read(&p21lvl).unwrap(),
                want_lvl,
                "committed phase21-cavern .inf_lvl drifted from the generator"
            );
            assert_eq!(
                std::fs::read(p21dir.join("Borer.inf_act")).unwrap(),
                encode_actor(&phase21_borer_class()).unwrap(),
                "committed phase21-cavern .inf_act drifted from the generator"
            );
            assert_eq!(
                std::fs::read(p21dir.join("Cavern.inf_voxel")).unwrap(),
                phase21_voxel_asset().unwrap().into_bytes(),
                "committed phase21-cavern .inf_voxel drifted from the generator"
            );
            assert_eq!(
                std::fs::read(p21dir.join("Phase21Cavern.inf_terrain")).unwrap(),
                phase21_terrain_asset().unwrap().into_bytes(),
                "committed phase21-cavern .inf_terrain drifted from the generator"
            );
        }

        // Phase 22 playground lock (P22.4): the level, the two `.inf_mesh`
        // assets, the three blueprints and the terrain. The meshes are locked and
        // not merely regenerated for the same reason the cavern's volume is — a
        // drifted mesh is a different *fracture*, because the cook chunks whatever
        // geometry it is handed, and the level says nothing about that.
        let p22dir = phase22_playground_dir();
        let p22lvl = p22dir.join("Phase22Playground.inf_lvl");
        if p22lvl.exists() {
            let want_lvl = crate::scene::serialize::encode(
                &crate::scene::serialize::to_scene_file(&phase22_playground_scene()),
            )
            .unwrap();
            assert_eq!(
                std::fs::read(&p22lvl).unwrap(),
                want_lvl,
                "committed phase22-playground .inf_lvl drifted from the generator"
            );
            for (file, class) in [
                ("Demolition.inf_act", phase22_demolition_class()),
                ("Wreck.inf_act", phase22_wreck_class()),
                ("Roller.inf_act", phase22_roller_class()),
            ] {
                assert_eq!(
                    std::fs::read(p22dir.join(file)).unwrap(),
                    encode_actor(&class).unwrap(),
                    "committed phase22-playground {file} drifted from the generator"
                );
            }
            for (file, mesh) in [
                ("Block.inf_mesh", phase22_tower_mesh()),
                ("Chassis.inf_mesh", phase22_chassis_mesh()),
            ] {
                assert_eq!(
                    std::fs::read(p22dir.join(file)).unwrap(),
                    inf_asset::encode(&mesh).unwrap(),
                    "committed phase22-playground {file} drifted from the generator"
                );
            }
            assert_eq!(
                std::fs::read(p22dir.join("Phase22Playground.inf_terrain")).unwrap(),
                phase22_terrain_asset().unwrap().into_bytes(),
                "committed phase22-playground .inf_terrain drifted from the generator"
            );
        }

        // Phase 23 workshop lock (P23.6). The `.inf_mesh` is locked for a reason
        // the others are not: it is the **import baseline** the gate's undo arm
        // compares against byte for byte, so a drifted baseline would not fail
        // here as a byte diff — it would fail there as "undo went to the wrong
        // place", which is a much worse place to learn it.
        let p23dir = phase23_workshop_dir();
        let p23lvl = p23dir.join("Phase23Workshop.inf_lvl");
        if p23lvl.exists() {
            assert_eq!(
                std::fs::read(&p23lvl).unwrap(),
                crate::scene::serialize::encode(&crate::scene::serialize::to_scene_file(
                    &phase23_workshop_scene()
                ))
                .unwrap(),
                "committed phase23-workshop .inf_lvl drifted from the generator"
            );
            assert_eq!(
                std::fs::read(p23dir.join("Prop.inf_mesh")).unwrap(),
                inf_asset::encode(&phase23_baseline_mesh()).unwrap(),
                "committed phase23-workshop Prop.inf_mesh drifted from the generator"
            );
        }

        // ── Phase 29 locomotion lock (P29.6 audit, A8) ─────────────────────
        //
        // The wave added `write_phase29_locomotion()` to the bless branch above
        // and no arm down here, which made the showcase the ONLY sample in the
        // tree that regenerates on demand and is never checked. Everything the
        // P29.6 ledger claims about that folder — the clips are derived, the
        // machine is the proposal, the controller is a function of the rig, the
        // character is a real character — is a claim about *these bytes*, and
        // `phase29_gate` loads them off disk and simulates whatever it finds. So
        // a change to `phase29_locomotion_scene`, to `edit_create_character`'s
        // capsule arithmetic or to the derivation's encoding would have moved
        // the committed content and the trace with it, together, silently.
        //
        // Every file, not a representative one: the level, the rig, the mesh,
        // the three derived clips, the machine, the controller and the three
        // text faces.
        let p29dir = phase29_locomotion_dir();
        let p29lvl = p29dir.join("Phase29Locomotion.inf_lvl");
        if p29lvl.exists() {
            assert_eq!(
                std::fs::read(&p29lvl).unwrap(),
                crate::scene::serialize::encode(&crate::scene::serialize::to_scene_file(
                    &phase29_locomotion_scene()
                ))
                .unwrap(),
                "committed phase29-locomotion .inf_lvl drifted from the generator"
            );
            assert_eq!(
                std::fs::read(p29dir.join("Hero.inf_skel")).unwrap(),
                inf_asset::encode(&phase29_skeleton()).unwrap(),
                "committed phase29-locomotion Hero.inf_skel drifted from the generator"
            );
            assert_eq!(
                std::fs::read(p29dir.join("Hero Body.inf_mesh")).unwrap(),
                inf_asset::encode(&phase29_body()).unwrap(),
                "committed phase29-locomotion Hero Body.inf_mesh drifted from the generator"
            );
            let skel_bytes = *PHASE29_SKELETON_GUID.as_bytes();
            let (set, _) = phase29_clips();
            for (file, clip) in [
                ("Hero Idle.inf_anim", &set.idle),
                ("Hero Walk.inf_anim", &set.walk),
                ("Hero Run.inf_anim", &set.run),
            ] {
                assert_eq!(
                    std::fs::read(p29dir.join(file)).unwrap(),
                    inf_asset::encode(&inf_anim::AnimClipAsset::new(
                        clip.clone(),
                        Some(skel_bytes)
                    ))
                    .unwrap(),
                    "committed phase29-locomotion {file} drifted from the generator — \
                     the clip is DERIVED, so this also catches a change in the \
                     derivation the gate would otherwise absorb"
                );
            }
            let machine = phase29_machine();
            assert_eq!(
                std::fs::read(p29dir.join("Hero Locomotion.inf_sm")).unwrap(),
                inf_asset::encode(&inf_anim::StateMachineAsset::new(
                    machine.clone(),
                    Some(skel_bytes)
                ))
                .unwrap(),
                "committed phase29-locomotion .inf_sm drifted from the proposal"
            );
            assert_eq!(
                std::fs::read(p29dir.join("Hero Controller.inf_act")).unwrap(),
                encode_actor(&phase29_controller()).unwrap(),
                "committed phase29-locomotion .inf_act drifted from the generator"
            );
            // The three text faces, which are the ones an author edits and the
            // gate's one-line-diff arm reads. Read as STRINGS so a line-ending
            // rewrite fails here by content rather than by an opaque byte count.
            assert_eq!(
                std::fs::read_to_string(p29dir.join("Hero Locomotion.inf_sm.txt")).unwrap(),
                inf_anim::to_toml(&machine),
                "committed phase29-locomotion .inf_sm.txt drifted from the machine"
            );
            assert_eq!(
                std::fs::read_to_string(p29dir.join("camera.toml")).unwrap(),
                inf_ecs::camera::CameraTuning::default().to_toml().unwrap(),
                "committed phase29-locomotion camera.toml drifted from the ALS table"
            );
            assert_eq!(
                std::fs::read_to_string(p29dir.join("input.toml")).unwrap(),
                toml::to_string_pretty(&inf_input::default_map()).unwrap(),
                "committed phase29-locomotion input.toml drifted from `default_map`"
            );
            assert_eq!(
                std::fs::read_to_string(p29dir.join("README.md")).unwrap(),
                PHASE29_README,
                "committed phase29-locomotion README drifted from the generator"
            );
        }

        // The island city lock (I3). The `.inf_pcg` and the `.inf_mesh` are
        // locked beside the level for the same reason `phase22-playground` locks
        // its meshes: the level says nothing about what the graph GROWS or what
        // the streets look like, so a drifted graph would not fail here as a byte
        // diff — it would fail in the gate as "the city is a different size",
        // which is a much worse place to learn it.
        let citydir = city_dir();
        let citylvl = citydir.join("City.inf_lvl");
        if citylvl.exists() {
            assert_eq!(
                std::fs::read(&citylvl).unwrap(),
                crate::scene::serialize::encode(&crate::scene::serialize::to_scene_file(
                    &city_scene()
                ))
                .unwrap(),
                "committed City.inf_lvl drifted from the generator"
            );
            let graph = city_block_graph();
            let lowered = inf_pcg::lower_graph(&graph, &inf_pcg::pcg_registry());
            assert!(
                lowered.ok,
                "the city's graph stopped lowering: {:?}",
                lowered.issues
            );
            assert_eq!(
                std::fs::read(citydir.join("CityBlock.inf_pcg")).unwrap(),
                inf_asset::encode(&inf_pcg::PcgAssetPayload::from_graph(
                    &graph,
                    lowered.document
                ))
                .unwrap(),
                "committed CityBlock.inf_pcg drifted from the generator"
            );
            assert_eq!(
                std::fs::read(citydir.join("CityRoads.inf_mesh")).unwrap(),
                inf_asset::encode(&city_road_mesh().unwrap()).unwrap(),
                "committed CityRoads.inf_mesh drifted from the I2 road door"
            );
            assert_eq!(
                std::fs::read_to_string(citydir.join("README.md")).unwrap(),
                CITY_README,
                "committed city README drifted from the generator"
            );
        }

        // The Harbour Heist mission (SCRIPT3). The level, the vault graph and the
        // README are locked here; the **mission itself** is not, because it is
        // hand-authored (`crate::heist`'s module header states the ruling) and
        // its own arms live beside it — it compiles clean, and its sidecar
        // describes its bytes.
        let hdir = crate::heist::heist_dir();
        let hlvl = hdir.join("HarbourHeist.inf_lvl");
        if hlvl.exists() {
            assert_eq!(
                std::fs::read(&hlvl).unwrap(),
                crate::scene::serialize::encode(&crate::scene::serialize::to_scene_file(
                    &crate::heist::heist_scene()
                ))
                .unwrap(),
                "committed HarbourHeist.inf_lvl drifted from the generator"
            );
            for (graph, file) in [
                (crate::heist::heist_vault_graph(), "HarbourVault.inf_pcg"),
                (
                    crate::heist::heist_housing_graph(),
                    "HarbourHousing.inf_pcg",
                ),
            ] {
                let lowered = inf_pcg::lower_graph(&graph, &inf_pcg::pcg_registry());
                assert!(lowered.ok, "{file}'s graph stopped lowering");
                assert_eq!(
                    std::fs::read(hdir.join(file)).unwrap(),
                    inf_asset::encode(&inf_pcg::PcgAssetPayload::from_graph(
                        &graph,
                        lowered.document
                    ))
                    .unwrap(),
                    "committed {file} drifted from the generator"
                );
            }
        }

        // The island levels (I7). The `.inf_lvl` and the `.inf_pcg` are locked
        // together for the city's own reason: the level says nothing about what
        // the biome binding GROWS, so a drifted cover document would not fail
        // here as a byte diff -- it would fail in the gate as "the island has a
        // different population", which is a much worse place to learn it.
        //
        // A recipe whose derived layers have not been written yet is SKIPPED
        // rather than failed, exactly as the platformer sample above is: a fresh
        // checkout that has not run `inf island route` has no design to author a
        // level from.
        for rel in crate::island::ISLAND_RECIPES {
            let Some(design) = crate::island::committed_design(rel) else {
                eprintln!("SKIP: no committed island design at {rel}");
                continue;
            };
            let dir = crate::island::repo_root()
                .join(rel)
                .parent()
                .unwrap()
                .to_path_buf();
            let slug = inf_island::slug(&design.recipe.name);
            let lvl = dir.join(format!("{slug}.inf_lvl"));
            if !lvl.exists() {
                eprintln!("SKIP: {} has not been blessed yet", lvl.display());
                continue;
            }
            assert_eq!(
                std::fs::read(&lvl).unwrap(),
                crate::scene::serialize::encode(&crate::scene::serialize::to_scene_file(
                    &crate::island::island_scene(&design)
                ))
                .unwrap(),
                "committed {} drifted from the island generator",
                lvl.display()
            );
            assert_eq!(
                std::fs::read(dir.join(format!("{slug}Cover.inf_pcg"))).unwrap(),
                inf_asset::encode(&crate::island::island_cover_payload(
                    design.recipe.seed_for("cover")
                ))
                .unwrap(),
                "committed {slug}Cover.inf_pcg drifted from the generator"
            );
            // The STREET LAYER (wave ROAD1b), locked to the level for the same
            // reason the `.inf_pcg` is: it is derived from the same blocks the
            // level writes, through `inf_ecs::traffic::streets_of_blocks`, and
            // a drifted layer would not fail here as a byte diff — it would
            // fail in the island build as "the town is paved somewhere else",
            // which is a much worse place to learn it.
            let streets = dir.join(&design.recipe.roads.streets);
            if streets.exists() {
                let mut want = Vec::new();
                let tmp = std::env::temp_dir().join(format!("{slug}-road1b-streets.geojson"));
                inf_island::layers::write_streets(
                    &tmp,
                    &design.anchor,
                    &crate::island::island_street_spans(&design),
                )
                .expect("the street layer writes");
                std::fs::File::open(&tmp)
                    .and_then(|mut f| std::io::Read::read_to_end(&mut f, &mut want))
                    .expect("read it back");
                let _ = std::fs::remove_file(&tmp);
                assert_eq!(
                    std::fs::read(&streets).unwrap(),
                    want,
                    "committed {} drifted from the island generator",
                    streets.display()
                );
            } else {
                eprintln!("SKIP: {} has not been blessed yet", streets.display());
            }
        }

        // The starter character (SK1c). Every file, not a representative one --
        // phase29's rule, and here it is load-bearing twice over: eight assets
        // name each other by GUID, so a drift in one sidecar is a dangling
        // reference in another file rather than a diff you can read; and the
        // island's hero binds three of these GUIDs, so a silent regeneration is
        // a level that boots with no rig.
        // **BOTH committed bodies** (wave CHAR1a.2). One loop rather than two
        // copies: the female body is the same eight assets naming each other by
        // GUID, both island recipes copy the set, and every 3D template
        // scaffolds it — so every argument for locking the male's bytes is an
        // argument for locking hers.
        for (scdir, built, readme, marker) in [
            (
                starter_character_dir(),
                starter_character_files(),
                STARTER_CHARACTER_README,
                "Starter.inf_skel",
            ),
            (
                starter_character_f_dir(),
                starter_character_f_files(),
                STARTER_CHARACTER_F_README,
                "Starter_F.inf_skel",
            ),
        ] {
            if !scdir.join(marker).exists() {
                eprintln!("SKIP: {} has not been blessed yet", scdir.display());
                continue;
            }
            let (want, _) = built.expect("the committed character builds");
            let mut have: Vec<String> = std::fs::read_dir(&scdir)
                .unwrap()
                .filter_map(|e| e.ok())
                .filter(|e| e.file_type().map(|t| t.is_file()).unwrap_or(false))
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .filter(|n| n != "README.md")
                .collect();
            have.sort();
            let names: Vec<String> = want.iter().map(|(n, _)| n.clone()).collect();
            assert_eq!(
                have,
                names,
                "the committed character in {} is not the file SET the wizard \
                 writes -- an extra file here is one the asset scan promotes \
                 under a minted GUID, and a missing one is a dangling reference",
                scdir.display()
            );
            for (name, bytes) in &want {
                assert_eq!(
                    &std::fs::read(scdir.join(name)).unwrap(),
                    bytes,
                    "committed {name} drifted from the wizard"
                );
            }
            assert_eq!(
                std::fs::read_to_string(scdir.join("README.md")).unwrap(),
                readme,
                "committed README in {} drifted from the generator",
                scdir.display()
            );
        }

        // **The ground library (TER2a clause 3).** Every file, for the starter
        // character's own two reasons and a third: these are the first `.inf_tex`
        // files this repository has held, they are seven megabytes of them, and
        // they are BYTE-LOCKED ON EVERY LEG — which is the whole reason the
        // synthesis is a transcendental-free CPU generator rather than the P7
        // GPU bake. A leg that produced different bytes would be a leg whose
        // ground streams different pages, and it would fail here.
        let gdir = ground_dir();
        if gdir.join("Ground_Grass.inf_mat").exists() {
            let want = crate::ground::ground_library().expect("the ground library builds");
            let mut have: Vec<String> = std::fs::read_dir(&gdir)
                .unwrap()
                .filter_map(|e| e.ok())
                .filter(|e| e.file_type().map(|t| t.is_file()).unwrap_or(false))
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .filter(|n| n != "README.md")
                .collect();
            have.sort();
            let mut names = crate::ground::ground_files();
            names.extend(crate::cover::cover_files());
            names.sort();
            assert_eq!(
                have, names,
                "the committed ground library is not the file SET the generators \
                 write -- an extra file here is one the asset scan promotes under \
                 a minted GUID, and a missing one is a terrain layer or a scatter \
                 kind bound to nothing"
            );
            for f in &crate::cover::cover_library().expect("the cover library builds") {
                assert_eq!(
                    std::fs::read(gdir.join(&f.name)).unwrap(),
                    f.payload,
                    "committed cover {} drifted from the generator",
                    f.name
                );
                let side = inf_asset::AssetSidecar::load(&gdir.join(&f.name))
                    .unwrap_or_else(|e| panic!("{} has no sidecar: {e}", f.name));
                assert_eq!(
                    side.to_toml().unwrap(),
                    f.sidecar.to_toml().unwrap(),
                    "committed cover {}'s sidecar drifted",
                    f.name
                );
            }
            for f in &want {
                assert_eq!(
                    std::fs::read(gdir.join(&f.name)).unwrap(),
                    f.payload,
                    "committed ground {} drifted from the generator",
                    f.name
                );
                let side = inf_asset::AssetSidecar::load(&gdir.join(&f.name))
                    .unwrap_or_else(|e| panic!("{} has no sidecar: {e}", f.name));
                assert_eq!(
                    side.to_toml().unwrap(),
                    f.sidecar.to_toml().unwrap(),
                    "committed ground {}'s sidecar drifted",
                    f.name
                );
            }
        } else {
            eprintln!("SKIP: the ground library has not been blessed yet");
        }

        // **The settlement zone library (wave I8a).** Every file, and the file
        // SET as well as the bytes — for the ground library's own reason: an
        // extra `.inf_pcg` here is one the asset scan promotes under a minted
        // GUID, and a missing one is a hundred and seventy settlement blocks
        // whose `PcgVolume.graph` resolves to nothing.
        let sdir = crate::settlement::settlement_dir();
        if sdir.join("Zone_Office.inf_pcg").exists() {
            let mut have: Vec<String> = std::fs::read_dir(&sdir)
                .unwrap()
                .filter_map(|e| e.ok())
                .filter(|e| e.file_type().map(|t| t.is_file()).unwrap_or(false))
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .filter(|n| n != "README.md")
                .collect();
            have.sort();
            assert_eq!(
                have,
                crate::settlement::settlement_files(),
                "the committed settlement library is not the file SET the \
                 generator writes"
            );
            for a in inf_pcg::ArchetypeId::ALL {
                let p = sdir.join(crate::settlement::zone_file_name(a));
                let want = inf_asset::encode(
                    &crate::settlement::zone_payload(a).expect("the zone document lowers"),
                )
                .expect("the zone document encodes");
                assert_eq!(
                    std::fs::read(&p).unwrap(),
                    want,
                    "committed {} drifted from the generator",
                    p.display()
                );
                let side = inf_asset::AssetSidecar::load(&p)
                    .unwrap_or_else(|e| panic!("{} has no sidecar: {e}", p.display()));
                assert_eq!(
                    side.guid.0,
                    crate::settlement::zone_guid(a),
                    "the {} zone's committed GUID is not the derived one — every \
                     block's `PcgVolume.graph` names the derived one",
                    a.name()
                );
            }
            assert_eq!(
                std::fs::read_to_string(sdir.join("README.md")).unwrap(),
                crate::settlement::SETTLEMENT_README,
                "committed settlement README drifted from the generator"
            );
        } else {
            eprintln!("SKIP: the settlement library has not been blessed yet");
        }
    }

    /// **The starter character builds, builds CLEAN, and builds the same twice.**
    ///
    /// Three claims the byte lock above cannot make on its own:
    ///
    /// * it builds at all from the wizard's *default* spec, on the 161-bone
    ///   mannequin -- which is the rig `build_locomotion` addresses by name, so
    ///   this is the arm that goes red the day a bone is renamed;
    /// * it builds with **exactly one advisory, and it is a known one**. The
    ///   wizard reports what it could not do perfectly (vertices that could not
    ///   see a bone, joints outside the mesh, a cycle that could not be
    ///   measured), and the build hands the list back rather than swallowing it
    ///   or refusing on it: refusing would make the folder unbuildable for a
    ///   bound the engine already carries by name, and swallowing would make a
    ///   *new* advisory invisible. So the list is pinned by CONTENT — a count of
    ///   one says nothing about which one. *(The SK1c audit corrected this
    ///   bullet: it used to claim the build had no warnings and that
    ///   `starter_character_into` refused on a non-empty list. Neither was true
    ///   of the code below it, which has always asserted exactly one.)*;
    /// * it is **reproducible**. The whole committed folder rests on the build
    ///   being a pure function of the spec and the ids -- a heat solve, a
    ///   derivation and a proposal all in the middle of it -- and nothing else
    ///   in the tree checks that. Two builds, compared file by file.
    #[test]
    fn the_starter_character_builds_clean_and_reproducibly() {
        let (a, warnings) = starter_character_files().expect("the starter character builds");
        let (b, _) = starter_character_files().expect("the starter character builds twice");
        // **The advisory list, pinned by content.** Exactly one, and it is
        // SK1b's carried bound rather than something new: some of the generated
        // body's kernel vertices are caps buried inside a neighbouring shell, so
        // the visibility oracle cannot reach them and they keep the bone the
        // generator seeded them with -- which is right. Pinned by CONTENT and
        // not by count, because a count of one says nothing about which one, and
        // asserted rather than allowed-to-be-empty, because "no warnings" could
        // only be bought by silencing this one.
        //
        // **Wave CHAR1a moved the number**, because it raised the body's
        // tessellation. The written mesh went from 1 247 vertices / 1 498
        // triangles to **3 867 / 5 718**; the advisory counts the KERNEL cage
        // the visibility oracle runs over, which went from **795 to 2 905**,
        // and the unreached count with it from **35 to 102** -- 4.40% of that
        // cage before, **3.51%** after. A denser cage buries a smaller share of
        // itself, which is the claim worth pinning and the reason the number
        // below is spelled out rather than made a wildcard.
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(
            warnings[0].starts_with("102 of the generated body's vertices"),
            "the starter character's advisory changed: {warnings:?}"
        );
        assert_eq!(
            a.len(),
            19,
            "expected eight payloads, eight sidecars and three text files, got {:?}",
            a.iter().map(|(n, _)| n).collect::<Vec<_>>()
        );
        assert_eq!(
            a, b,
            "two builds of the same spec under the same ids produced different \
             bytes -- the committed folder cannot be re-blessed"
        );
        // The eight GUIDs really landed: a sidecar is TOML, so the id is
        // readable, and a build that quietly minted its own would still produce
        // two identical runs only by luck.
        let ids = starter_character_ids();
        for (slot, file) in [
            (ids.skeleton, "Starter.inf_skel.toml"),
            (ids.material, "Starter_Skin.inf_mat.toml"),
            (ids.mesh, "Starter_Body.inf_mesh.toml"),
            (ids.idle, "Starter_Idle.inf_anim.toml"),
            (ids.walk, "Starter_Walk.inf_anim.toml"),
            (ids.run, "Starter_Run.inf_anim.toml"),
            (ids.machine, "Starter_Locomotion.inf_sm.toml"),
            (ids.actor, "Starter_Controller.inf_act.toml"),
        ] {
            let want = slot.expect("every starter id is fixed").0.to_string();
            let text = a
                .iter()
                .find(|(n, _)| n == file)
                .map(|(_, b)| String::from_utf8_lossy(b).into_owned())
                .unwrap_or_else(|| panic!("no sidecar {file}"));
            assert!(
                text.contains(&want),
                "{file} does not carry the fixed GUID {want}:\n{text}"
            );
        }
    }

    /// **Both islands' recipes name the WHOLE starter character** (SK1c audit,
    /// M3).
    ///
    /// The character reaches a built island project through the recipe's
    /// `[content]` list — seventeen `../starter-character/…` lines in each of
    /// `samples/island/island.toml` and `samples/island-fixture/island.toml`.
    /// That is a **third** place the character's identity is written down, after
    /// the folder itself and `inf_project::STARTER_CHARACTER`, and the wave's own
    /// ruling about `pie_sim`'s resolver ("a table of file names here would be a
    /// second place the character's identity is written down") is the argument
    /// against having one at all.
    ///
    /// It cannot be removed — a recipe is a committed authoring document and
    /// `include_bytes!` is not available to it — so it is *checked*. The island
    /// gate covers five of the seventeen by consequence (the payload's rig,
    /// machine, three clips and class); the skin material and the machine's text
    /// face are covered by nothing, and a dropped sidecar is a project whose rig
    /// resolves and whose clips do not.
    ///
    /// Read off disk on both sides, so the two cannot drift.
    #[test]
    fn both_island_recipes_name_the_whole_starter_character() {
        const SKIPPED: [&str; 3] = ["README.md", "camera.toml", "input.toml"];
        // **BOTH committed bodies** (wave CHAR1a.2), each against its own path
        // prefix. Folded into one arm rather than copied into a second, so the
        // day a third body lands there is one list to grow — and note the
        // prefixes are checked separately, because `../starter-character-f/X`
        // also starts with nothing the male prefix matches and a single
        // `strip_prefix` would have silently seen zero female files.
        let folders = [
            ("../starter-character/", starter_character_dir()),
            ("../starter-character-f/", starter_character_f_dir()),
        ];
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../samples");
        for (prefix, dir) in &folders {
            let mut on_disk: Vec<String> = std::fs::read_dir(dir)
                .unwrap_or_else(|e| panic!("no {}: {e}", dir.display()))
                .filter_map(|e| e.ok())
                .filter(|e| e.file_type().map(|t| t.is_file()).unwrap_or(false))
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .filter(|n| !SKIPPED.contains(&n.as_str()))
                .collect();
            on_disk.sort();
            assert!(!on_disk.is_empty(), "{} is empty", dir.display());

            for recipe in ["island/island.toml", "island-fixture/island.toml"] {
                let path = root.join(recipe);
                let text = std::fs::read_to_string(&path)
                    .unwrap_or_else(|e| panic!("no {}: {e}", path.display()));
                let mut named: Vec<String> = text
                    .lines()
                    .filter_map(|l| {
                        let l = l.trim().trim_start_matches('"').trim_end_matches(',');
                        let l = l.trim_end_matches('"');
                        l.strip_prefix(prefix).map(str::to_string)
                    })
                    .collect();
                named.sort();
                assert_eq!(
                    named, on_disk,
                    "{recipe}'s `[content]` list is not {prefix} — a missing \
                     entry is an island that boots a body whose rig resolves \
                     and whose clips do not"
                );
            }
        }
    }

    /// **Both islands' recipes name the WHOLE settlement zone library** (wave
    /// I8a) — the starter character's arm above, one library over.
    ///
    /// Same argument, and the consequence of a gap is louder: a `[content]` list
    /// missing one zone document is a project where every block of that
    /// archetype has a `PcgVolume.graph` nothing resolves, so a whole district
    /// evaluates to nothing and no error is raised anywhere — a `PcgVolume`
    /// whose graph is missing is a volume that scatters zero instances, which is
    /// indistinguishable from a volume over bare ground.
    #[test]
    fn both_island_recipes_name_the_whole_settlement_library() {
        let dir = crate::settlement::settlement_dir();
        if !dir.join("Zone_Office.inf_pcg").exists() {
            eprintln!("SKIP: the settlement library has not been blessed yet");
            return;
        }
        let mut on_disk: Vec<String> = std::fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("no {}: {e}", dir.display()))
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().map(|t| t.is_file()).unwrap_or(false))
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n != "README.md")
            .collect();
        on_disk.sort();
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../samples");
        for recipe in ["island/island.toml", "island-fixture/island.toml"] {
            let path = root.join(recipe);
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("no {}: {e}", path.display()));
            let mut named: Vec<String> = text
                .lines()
                .filter_map(|l| {
                    let l = l.trim().trim_start_matches('"').trim_end_matches(',');
                    let l = l.trim_end_matches('"');
                    l.strip_prefix("../settlement/").map(str::to_string)
                })
                .collect();
            named.sort();
            assert_eq!(
                named, on_disk,
                "{recipe}'s `[content]` list is not the settlement zone library — \
                 a missing entry is a district bound to a document nothing can \
                 resolve, which evaluates to nothing and says nothing"
            );
        }
    }

    /// **The README names files that are actually there** (SK1c audit, M2).
    ///
    /// The wave's own headline defect was a text face written at a path rebuilt
    /// from a *display* name, beside a payload `sanitize` had renamed — and the
    /// README it shipped alongside the fix made the same mistake one layer up:
    /// six of its eight rows named `Starter Body.inf_mesh`, `Starter Skin.inf_mat`
    /// and four more with **spaces**, and the folder contains `Starter_Body.inf_mesh`
    /// and friends. Nothing noticed, because the README is byte-locked against
    /// the constant that produces it and a byte lock cannot see a wrong name.
    ///
    /// So every file the README's table names in its first column is asserted to
    /// be one the wizard actually wrote. The table is the readable face of a
    /// generated folder; a face that names a file nobody can open is worse than
    /// no face, which is the argument the wave made about `sm_text`.
    #[test]
    fn the_starter_character_readme_names_files_that_exist() {
        let (files, _) = starter_character_files().expect("the starter character builds");
        let have: std::collections::BTreeSet<&str> =
            files.iter().map(|(n, _)| n.as_str()).collect();
        let mut named = 0usize;
        for line in STARTER_CHARACTER_README.lines() {
            if !line.starts_with("| `") {
                continue;
            }
            // The first column only: the second is prose, and prose is allowed
            // to name a type or a function in backticks.
            let cell = line[1..].split('|').next().unwrap_or("");
            for (i, span) in cell.split('`').enumerate() {
                if i % 2 == 0 {
                    continue;
                }
                named += 1;
                assert!(
                    have.contains(span),
                    "the README names `{span}`, which the wizard does not write. \
                     It writes: {:?}",
                    have
                );
            }
        }
        // …and it names most of them, so a table somebody emptied fails here
        // rather than passing on nothing.
        assert!(
            named >= have.len() / 2,
            "the README's table names only {named} of the {} files in the folder",
            have.len()
        );
    }

    /// **The committed starter character IS what the wizard opens with.**
    ///
    /// The sentence the whole folder rests on, and the one a byte lock cannot
    /// make: `samples/starter-character` is not a curated shape, it is the
    /// output of pressing New Character and accepting every default. So the
    /// spec it is built from is asserted to be `CharacterSpec::default()` field
    /// by field, with the name as the single deliberate difference.
    ///
    /// When the wizard's defaults move, this stays green and the byte lock goes
    /// red — which is the right pair: the folder is re-blessed, and the diff is
    /// the new default character rather than a silent staleness.
    #[test]
    fn the_starter_character_is_what_the_wizard_opens_with() {
        let spec = starter_character_spec();
        let d = crate::character::CharacterSpec::default();
        assert_eq!(
            spec.plan, d.plan,
            "the starter character is not the default body plan"
        );
        assert_eq!(
            spec.params, d.params,
            "the starter proportions are not the defaults"
        );
        assert_eq!(spec.gait, d.gait, "the starter gait is not the default");
        assert_eq!(
            spec.mesh, d.mesh,
            "the starter character fits a supplied mesh"
        );
        assert_eq!(
            spec.plan,
            inf_anim::BodyPlan::Biped,
            "the wizard's default plan is no longer the mannequin, so the \
             committed starter character is not a 161-bone rig"
        );
        assert_ne!(
            spec.name, d.name,
            "the only difference from the default spec should be the name"
        );
        assert_eq!(spec.name, STARTER_CHARACTER_NAME);
    }

    /// **The starter character is a RIG, not a puppet** -- the properties the
    /// island's hero and the New Character wizard both depend on.
    ///
    /// A byte lock says the bytes did not move. It says nothing about whether
    /// they were ever right, and this folder ships as the answer to "what does a
    /// character in this engine look like", so the answer is asserted.
    #[test]
    fn the_starter_character_is_the_mannequin_with_everything_on_it() {
        let (files, _) = starter_character_files().expect("the starter character builds");
        let payload = |name: &str| {
            files
                .iter()
                .find(|(n, _)| n == name)
                .map(|(_, b)| b.clone())
                .unwrap_or_else(|| panic!("no {name}"))
        };
        let rig: inf_anim::SkeletonAsset =
            inf_asset::decode(&payload("Starter.inf_skel")).expect("the committed rig decodes");
        assert_eq!(
            rig.skeleton.len(),
            161,
            "the starter rig is not the mannequin"
        );
        assert_eq!(rig.roles.len(), 161, "a rig with no role table guesses");
        assert!(!rig.twists.is_empty(), "no twist drivers");
        assert!(!rig.ik_follow.is_empty(), "no IK handles to follow");
        assert!(
            !rig.grips.is_empty(),
            "the starter rig carries no grip catalogue, so nothing it picks up \
             can close a hand on it"
        );
        // The hand solver's two prerequisites, asserted where the asset is
        // rather than three passes downstream where the symptom is "the fingers
        // did not move".
        let roles = rig.role_index();
        for side in [inf_anim::BoneSide::Left, inf_anim::BoneSide::Right] {
            let hand = roles
                .first(inf_anim::BoneRoleKind::Hand, side)
                .unwrap_or_else(|| panic!("no {side:?} hand role"));
            assert!(
                inf_anim::hand_of(&rig.skeleton, roles, hand).is_some(),
                "the {side:?} hand has no derivable digits"
            );
            assert!(
                inf_anim::arm_chain(&rig.skeleton, roles, side).is_some(),
                "the {side:?} arm has no solvable chain"
            );
        }

        let body: inf_mesh::MeshAsset = inf_asset::decode(&payload("Starter_Body.inf_mesh"))
            .expect("the committed body decodes");
        assert!(
            body.submeshes.iter().all(|m| m.is_skinned()),
            "a submesh of the starter body carries no skin stream, so it does not deform"
        );
        assert!(
            !body.material_slots.is_empty(),
            "the starter body names no material slot"
        );

        let machine: inf_anim::StateMachineAsset =
            inf_asset::decode(&payload("Starter_Locomotion.inf_sm"))
                .expect("the committed machine decodes");
        assert!(
            machine.machine.states.len() >= 3,
            "a locomotion machine with fewer than three states is not one"
        );
        assert!(
            machine
                .machine
                .profiles
                .iter()
                .any(|p| p.name == crate::character::AIM_MASK),
            "the starter machine carries no `{}` profile",
            crate::character::AIM_MASK
        );
    }

    /// **The city's graph really cuts blocks into lots** (IB-2c), checked at the
    /// generator rather than only in the gate.
    ///
    /// The fixture's whole value is its size, and its size is a *derived*
    /// number: a lowering that silently dropped the `lots` pin would still
    /// produce a level, a `.inf_pcg` and a hundred volumes — and a city of one
    /// building per block. That is a hundred buildings where the gate expects a
    /// thousand, and every measured number would be off by ten with nothing to
    /// see.
    #[test]
    fn the_citys_graph_subdivides_its_blocks() {
        let lowered = inf_pcg::lower_graph(&city_block_graph(), &inf_pcg::pcg_registry());
        assert!(lowered.ok, "{:?}", lowered.issues);
        assert_eq!(lowered.buildings.len(), 1, "one pass per block volume");
        let pass = &lowered.buildings[0];
        let rules = pass
            .lots
            .expect("the block graph's `building.lots` node must reach the pass");
        assert_eq!(rules.frontage_m, CITY_FRONTAGE_M);
        assert_eq!(rules.depth_m, CITY_DEPTH_M);
        assert_eq!(rules.setback_m, CITY_SETBACK_M);
        assert!(
            pass.lot.is_some(),
            "the block's own footprint is the lot span"
        );

        // …and the pass really fans out, on the volume the level places.
        let cx = inf_pcg::GrammarContext {
            entity: Some(city_block_guid(0)),
            center: DVec3::ZERO,
            extent: DVec2::new(CITY_BLOCK_M.0 * 0.5, CITY_BLOCK_M.1 * 0.5),
            seed_offset: 0,
        };
        let lots = inf_pcg::building::oriented_lots_of(pass, &inf_pcg::NoSplines, &cx);
        assert_eq!(
            lots.len(),
            10,
            "a {} x {} block at {CITY_FRONTAGE_M} m frontage and {CITY_DEPTH_M} m \
             depth is 5 x 2 lots; got {}",
            CITY_BLOCK_M.0,
            CITY_BLOCK_M.1,
            lots.len()
        );
        let total = u32::try_from(lots.len()).unwrap() * CITY_BLOCKS * CITY_BLOCKS;
        assert!(
            total >= 1_000,
            "the city is {total} buildings, and the brief asked for a thousand"
        );
        println!(
            "I3 city: {} blocks x {} lots = {total} buildings",
            CITY_BLOCKS * CITY_BLOCKS,
            lots.len()
        );
    }

    /// The streets go through **I2's own import door**, and the door is what
    /// finds the junctions.
    #[test]
    fn the_citys_streets_are_a_road_graph_with_real_junctions() {
        let layer = city_road_layer();
        let graph = inf_gis::RoadGraph::from_layer(&layer);
        assert!(
            graph.skipped.is_empty(),
            "the road door skipped {:?}",
            graph.skipped
        );
        // (n+1) streets each way, each split into n segments.
        let want = 2 * (CITY_BLOCKS + 1) * CITY_BLOCKS;
        assert_eq!(graph.segments.len(), want as usize);
        // Every crossing is a node, and the interior ones are four-way. A street
        // digitised as ONE feature through its crossings would give 2(n+1) nodes
        // of degree 1 instead — which is I2's carried bound, and this is the
        // assertion that says the fixture does not walk into it.
        let interior = graph.junctions().filter(|j| j.degree() == 4).count();
        assert_eq!(
            interior,
            ((CITY_BLOCKS - 1) * (CITY_BLOCKS - 1)) as usize,
            "the interior crossings are not four-way; the layer is not split at \
             its junctions"
        );
        let mesh = city_road_mesh().expect("the streets build a surface");
        let verts: usize = mesh.submeshes.iter().map(|s| s.vertices.len()).sum();
        assert!(verts > 0, "the streets built no vertices");
        println!(
            "I3 city streets: {} segments, {interior} four-way junctions, {verts} \
             road vertices",
            graph.segments.len()
        );
    }

    /// **The coarse step costs nothing on flat ground, and here is the number.**
    ///
    /// `build_surface` resamples at the ground's pitch because what the step buys
    /// is conformance to the terrain's chord between samples (I2's own finding).
    /// This fixture's ground is a plane, which has no chord, so
    /// `CITY_ROAD_STEP_M` is 20 m rather than the 1 m default — 213 941 vertices
    /// down to a fraction. The saving is only legitimate while the claim below
    /// holds, so the claim is measured rather than assumed, and the day the city
    /// grows a terrain this arm fails and the step goes back to its pitch.
    #[test]
    fn the_citys_streets_conform_to_their_flat_ground_at_any_step() {
        let layer = city_road_layer();
        let graph = inf_gis::RoadGraph::from_layer(&layer);
        let build = |step: f64| {
            let opts = inf_gis::SurfaceOptions {
                ground_step_m: step,
                ..Default::default()
            };
            inf_gis::build_surface(&graph, &opts, &mut |x, z| Some(city_ground(x, z)))
        };
        let coarse = build(CITY_ROAD_STEP_M);
        let mut worst = 0.0f64;
        for ribbon in coarse.parts.values() {
            for p in &ribbon.vertices {
                let want = city_ground(p.x, p.z) + inf_gis::DEFAULT_ROAD_LIFT_M;
                worst = worst.max((p.y - want).abs());
            }
        }
        let fine = build(1.0);
        let n = |s: &inf_gis::RoadSurface| s.vertex_count();
        println!(
            "I3 city streets: worst deviation {worst:.6} m at a {CITY_ROAD_STEP_M} m \
             step; {} vertices against {} at the 1 m default ({:.0}x)",
            n(&coarse),
            n(&fine),
            n(&fine) as f64 / n(&coarse).max(1) as f64
        );
        assert!(
            worst < 1e-9,
            "the streets miss their own flat ground by {worst} m at a \
             {CITY_ROAD_STEP_M} m step — the coarse step is no longer free and \
             this fixture needs the ground's pitch back"
        );
        // ANTI-VACUITY: the two steps really are different surfaces, so the
        // claim above is about a coarsening rather than about nothing.
        assert!(
            n(&fine) > n(&coarse) * 8,
            "the 1 m step built {} vertices against the coarse {} — this arm is \
             not comparing two different resamplings",
            n(&fine),
            n(&coarse)
        );
    }

    /// **The showcase's character is the WIZARD's character** (P29.6 audit, A8).
    ///
    /// `phase29_locomotion_scene` hand-copies `SceneDoc::edit_create_character`'s
    /// capsule arithmetic and component set, because it needs
    /// `create_with_guid` and the wizard's door does not take one. They are
    /// equivalent today and nothing kept them so — while the sample's own README
    /// says its assets "are what `build_character` produces from the default
    /// biped". A byte pin cannot see that: it compares the committed bytes to
    /// the same copy that wrote them.
    ///
    /// So the two are compared to each other, field by field, at the same
    /// height.
    #[test]
    fn the_showcase_character_matches_the_wizard_door() {
        use inf_ecs::components::{
            CharacterController3D, CharacterMovement, Collider3D, ColliderShape3DKind, RigidBody3D,
            Transform,
        };
        let feet = phase29_start();
        let mut doc = SceneDoc::new();
        let guid = doc.edit_create_character(
            "Hero",
            Uuid::from_u128(1),
            Uuid::from_u128(2),
            Uuid::from_u128(3),
            feet,
            None,
            PHASE29_HEIGHT_M,
        );
        let door = doc.world().entity_of(guid).expect("the wizard spawned it");
        let sample_doc = phase29_locomotion_scene();
        let sample = sample_doc
            .world()
            .entity_of(PHASE29_HERO_GUID)
            .expect("the sample has a hero");

        let dw = doc.world().world();
        let sw = sample_doc.world().world();
        assert_eq!(
            dw.get::<Collider3D>(door)
                .map(|c| (c.shape_kind, c.half_extents, c.radius)),
            sw.get::<Collider3D>(sample)
                .map(|c| (c.shape_kind, c.half_extents, c.radius)),
            "the showcase's capsule is not the one the wizard would build"
        );
        assert_eq!(
            dw.get::<Transform>(door).map(|t| t.translation),
            sw.get::<Transform>(sample).map(|t| t.translation),
            "the showcase places its character at a different height than the \
             wizard does for the same feet"
        );
        let cm = |c: Option<&CharacterMovement>| {
            c.map(|c| {
                (
                    c.player_controlled,
                    c.stand_half_height_m,
                    c.crouch_half_height_m,
                    c.prone_half_height_m,
                )
            })
        };
        assert_eq!(
            cm(dw.get::<CharacterMovement>(door)),
            cm(sw.get::<CharacterMovement>(sample)),
            "the showcase's movement component is not the wizard's"
        );
        assert_eq!(
            dw.get::<RigidBody3D>(door).map(|b| b.kind),
            sw.get::<RigidBody3D>(sample).map(|b| b.kind)
        );
        assert!(
            dw.get::<CharacterController3D>(door).is_some()
                && sw.get::<CharacterController3D>(sample).is_some(),
            "one of the two has no character controller"
        );
        // …and the fixture is not vacuous: a capsule with real dimensions.
        let c = sw.get::<Collider3D>(sample).expect("a capsule");
        assert_eq!(c.shape_kind, ColliderShape3DKind::Capsule);
        assert!(c.radius > 0.1 && c.half_extents.y > 0.3, "{c:?}");
    }

    // ── Phase 22 playground shape (P22.4) ──────────────────────────────────

    /// **The sample really is the done-when sentence.** Asserted on the
    /// *generators* rather than on the committed bytes, so a change to the design
    /// fails here with a readable number instead of as a byte diff.
    #[test]
    fn the_playground_sample_is_actually_a_playground() {
        use glam::DVec2;
        let data = phase22_terrain_data();

        // Two bands of deformable ground, painted where the design says, with
        // grass on both sides of them.
        assert_eq!(
            data.dominant_layer_at(DVec2::new(PHASE22_ROLLER_X, PHASE22_SNOW_Z.0 + 4.0)),
            Some(PHASE22_SNOW_LAYER),
            "the snow band is not painted snow"
        );
        assert_eq!(
            data.dominant_layer_at(DVec2::new(PHASE22_ROLLER_X, PHASE22_SAND_Z.0 + 4.0)),
            Some(PHASE22_SAND_LAYER),
            "the sand band is not painted sand"
        );
        assert_eq!(
            data.dominant_layer_at(DVec2::new(PHASE22_ROLLER_X, PHASE22_SNOW_Z.0 - 8.0)),
            Some(PHASE22_GRASS_LAYER)
        );
        assert_eq!(
            data.dominant_layer_at(DVec2::new(PHASE22_ROLLER_X, PHASE22_SAND_Z.1 + 8.0)),
            Some(PHASE22_GRASS_LAYER)
        );

        // The ground everything stands on is flat, and the knoll is somewhere
        // nothing stands.
        for (x, z) in [
            PHASE22_TOWER_XZ,
            PHASE22_CONTROL_XZ,
            PHASE22_CAR_XZ,
            (PHASE22_ROLLER_X, PHASE22_ROLLER_START_Z),
        ] {
            assert_eq!(
                phase22_height(x, z),
                0.0,
                "({x}, {z}) is not on the flat working ground"
            );
        }
        assert!(
            phase22_height(108.0, 20.0) > 4.0,
            "the knoll is missing, so the terrain is a plane"
        );

        // Both meshes clear the cook's vgeom threshold, or the cook is not silent.
        assert!(phase22_tower_mesh().triangle_count() >= 2048);
        assert!(phase22_chassis_mesh().triangle_count() >= 2048);
        // …and the block really is three storeys of a metre-scale building: at
        // least three chunk layers deep by volume, so the support graph has an
        // interior rather than being a single course of blocks.
        let storeys = (PHASE22_TOWER_HEIGHT_M / 3.0).floor();
        assert!(storeys >= 3.0, "the block is {storeys} storeys tall");
        let per_storey = PHASE22_TOWER_CHUNKS as f64 / storeys;
        assert!(
            per_storey >= 6.0,
            "{per_storey} chunk(s) a storey — too coarse for a chunk to have \
             neighbours on every side"
        );

        // The two block actors differ in the FLAG and in nothing else: same mesh,
        // same class, same material constants.
        let doc = phase22_playground_scene();
        let get = |guid: uuid::Uuid| {
            let e = doc.entity_of(guid).expect("the actor exists");
            let w = doc.world().world();
            (
                *w.get::<inf_ecs::components::Destructible>(e).unwrap(),
                *w.get::<inf_ecs::components::MeshRef>(e).unwrap(),
                *w.get::<inf_ecs::components::ActorClass>(e).unwrap(),
            )
        };
        let (dt, mt, at) = get(PHASE22_TOWER_GUID);
        let (dc, mc, ac) = get(PHASE22_CONTROL_GUID);
        assert_eq!(mt.asset, mc.asset, "the twins reference different meshes");
        assert_eq!(at.0, ac.0, "the twins run different Blueprint classes");
        assert!(dt.runtime_destruct && !dc.runtime_destruct);
        assert_eq!(
            inf_ecs::components::Destructible {
                runtime_destruct: dc.runtime_destruct,
                ..dt
            },
            dc,
            "the control twin differs from the block in more than the flag"
        );
    }

    // ── Phase 20 coastal shape (P20.4) ─────────────────────────────────────

    /// The scene really is a **coast**: the terrain crosses sea level, the river
    /// descends the whole way, and the head lake actually holds water.
    ///
    /// Asserted on the generators rather than on the written bytes so it fails on
    /// the day someone retunes the numbers, not on the day someone forgets to
    /// bless — and every claim the gate makes about the content rests on one of
    /// these.
    #[test]
    fn the_coastal_sample_is_actually_coastal() {
        // A shoreline exists: land at the headland, sea floor at the far end.
        assert!(phase20_height(0.0, 256.0) > 30.0);
        assert!(phase20_height(PHASE20_WORLD_M, 256.0) < PHASE20_SEA_LEVEL_M);
        // …and it is crossed exactly once along the channel (a monotone ramp).
        let mut crossings = 0;
        let mut prev = phase20_height(0.0, phase20_channel_z(0.0));
        for i in 1..=512 {
            let x = i as f64;
            let h = phase20_height(x, phase20_channel_z(x));
            if (prev > PHASE20_SEA_LEVEL_M) != (h > PHASE20_SEA_LEVEL_M) {
                crossings += 1;
            }
            prev = h;
        }
        assert_eq!(crossings, 1, "the channel must meet the sea exactly once");

        // The head lake's basin really holds its level: the ground at the centre
        // is below it, and the ground at the rectangle's corner is above it.
        let (lx, lz) = PHASE20_LAKE_CENTER;
        assert!(phase20_height(lx, lz) < PHASE20_LAKE_LEVEL_M - 1.0);
        assert!(
            phase20_height(lx + PHASE20_LAKE_HALF_M, lz + PHASE20_LAKE_HALF_M)
                > PHASE20_LAKE_LEVEL_M,
            "the lake spills out of its own rectangle"
        );

        // The river descends monotonically, and its mouth is at the shore.
        let pts = phase20_river_points();
        assert_eq!(pts.len(), 5);
        for w in pts.windows(2) {
            assert!(w[1].y < w[0].y, "the river climbs: {:?}", w);
        }
        assert!(pts.last().unwrap().y > PHASE20_SEA_LEVEL_M);
        assert!(pts.last().unwrap().y < 2.0, "the mouth is not at the shore");
        // Each knot sits just above the valley floor it runs in.
        for p in &pts {
            let ground = phase20_height(p.x, p.z);
            assert!(
                (p.y - ground - 0.6).abs() < 1e-9,
                "knot {p:?} is not 0.6 m over its own valley floor ({ground})"
            );
        }
    }

    // ── Phase 16 gate scene shape (P16.6) ──────────────────────────────────

    /// The wizard-imported `.inf_terrain` really is what the gate needs: a
    /// kilometre-scale extent at `meters_per_sample >= 8`, at least three LOD
    /// levels, and a world far wider than any single render-wants radius — and it
    /// is a **pure function of the generator**, so the fixture setup, the cook and
    /// a second machine all produce the same bytes.
    #[test]
    fn phase16_terrain_imports_through_the_wizard_path() {
        let asset = phase16_terrain_asset().expect("the chunked import succeeds");
        let r = asset.reader();
        assert_eq!(r.tile_resolution(), PHASE16_TILE_RESOLUTION);
        assert_eq!(r.meters_per_sample(), PHASE16_MPS);
        // The phase goal names >= 8 m/sample; pinned as a const check so a
        // future retune of the sample cannot quietly drop below it.
        const _: () = assert!(PHASE16_MPS >= 8.0);
        assert!(
            r.lod_levels() >= 3,
            "need level 0 + at least two coarse levels, got {}",
            r.lod_levels()
        );
        // 1025 samples at 128 cells/tile ⇒ 8 x 8 level-0 pages, 8.2 km of world.
        let level0 = r.keys().filter(|k| k.is_lod0()).count();
        assert_eq!(level0, 64, "level-0 lattice");
        assert!(
            (phase16_world_size() - 8192.0).abs() < 1e-9,
            "world is {} m",
            phase16_world_size()
        );

        // Schema v2 (P16.6): the asset records the pyramid options it was built
        // with, so a sculpt write-back can never re-shape it.
        assert_eq!(
            r.pyramid_options(),
            Some(phase16_import_settings().pyramid()),
            "the imported asset must record its wizard pyramid settings"
        );

        // The cut is genuinely partial: the world is many tile spans wide.
        let span = (PHASE16_TILE_RESOLUTION as f64 - 1.0) * PHASE16_MPS;
        assert!(phase16_world_size() > 4.0 * span);

        // Byte-deterministic: two imports of the same source agree exactly.
        let again = phase16_terrain_asset().expect("import twice");
        assert_eq!(asset.as_bytes(), again.as_bytes());
    }

    /// The composed scene carries **two** terrains — one streamed, one inline —
    /// plus the partition block and the dual-role walker the gate scripts.
    #[test]
    fn phase16_scene_composes_terrain_and_partition() {
        use inf_ecs::components::{AlwaysLoaded, StreamingSource, Terrain};

        let doc = phase16_world_scene();
        let file = crate::scene::serialize::to_scene_file(&doc);
        assert!(file.settings.partition.enabled);
        assert_eq!(file.settings.partition.cell_size_m, PHASE16_CELL_SIZE_M);
        // The gate needs at least a 3x3 grid.
        const _: () = assert!(PHASE16_GRID >= 3);

        let terrains: Vec<_> = file
            .entities
            .iter()
            .filter(|e| e.terrain.is_some())
            .collect();
        assert_eq!(terrains.len(), 2, "the gate scene has TWO terrains");
        for e in &terrains {
            assert!(
                e.always_loaded.is_some(),
                "a Terrain occupies space, so it must be AlwaysLoaded or it streams out"
            );
        }

        let world = doc.world();
        let w = world.world();
        let streamed = world.entity_of(PHASE16_TERRAIN_GUID).expect("streamed");
        let t = w.get::<Terrain>(streamed).expect("terrain component");
        assert_eq!(t.asset, Some(PHASE16_TERRAIN_ASSET_GUID));
        assert!(t.data.is_empty(), "a streamed terrain ships no tiles");

        let inline = world
            .entity_of(PHASE16_INLINE_TERRAIN_GUID)
            .expect("inline");
        let t2 = w.get::<Terrain>(inline).expect("terrain component");
        assert!(t2.asset.is_none(), "the second terrain is inline");
        assert_eq!(
            t2.data.tile_count(),
            (PHASE16_INLINE_TILES * PHASE16_INLINE_TILES) as usize
        );
        // Two DIFFERENT grids — the case a coordinate-only GPU cache key breaks on.
        assert_ne!(t.data.tile_resolution(), t2.data.tile_resolution());
        assert_ne!(t.data.meters_per_sample(), t2.data.meters_per_sample());

        // The walker is both the streaming source and a terrain observer.
        let walker = world.entity_of(PHASE16_WALKER_GUID).expect("walker");
        assert!(w.get::<StreamingSource>(walker).is_some());
        assert!(w
            .get::<inf_ecs::components::CharacterController3D>(walker)
            .is_some());
        assert!(
            w.get::<AlwaysLoaded>(walker).is_none(),
            "a source is already persistent"
        );

        // The scripted walk crosses the whole world, and the camera paths differ.
        let walk: Vec<_> = (0..24).map(phase16_walk_point).collect();
        assert!(walk.iter().map(|p| p.x).fold(0.0f64, f64::max) > phase16_world_size() * 0.5);
        assert_ne!(phase16_camera_a(7), phase16_camera_b(7));

        // Save → load → save is byte-identical (the P3 discipline).
        let bytes1 = crate::scene::serialize::encode(&file).unwrap();
        let mut doc2 = SceneDoc::new();
        crate::scene::serialize::apply_to_doc(
            &mut doc2,
            &crate::scene::serialize::decode(&bytes1).unwrap(),
        );
        let bytes2 =
            crate::scene::serialize::encode(&crate::scene::serialize::to_scene_file(&doc2))
                .unwrap();
        assert_eq!(
            bytes1, bytes2,
            "phase16 scene must round-trip byte-identically"
        );
    }

    // ── Streamed-terrain sample shape (P16.3b2) ────────────────────────────

    /// The generated `.inf_terrain` really is what the gate needs: at least two
    /// coarse pyramid levels, and a level-0 grid wider than any single render
    /// wants radius — otherwise "the camera pages tiles" would never be exercised.
    #[test]
    fn streamed_terrain_asset_has_a_pyramid_and_outgrows_the_wants_radius() {
        let asset = streamed_terrain_asset();
        let reader = asset.reader();
        assert!(
            reader.lod_levels() >= 3,
            "need level 0 + at least two coarse levels, got {}",
            reader.lod_levels()
        );
        let level0 = reader.keys().filter(|k| k.is_lod0()).count();
        assert_eq!(
            level0,
            (STREAMED_TERRAIN_TILES * STREAMED_TERRAIN_TILES) as usize
        );
        assert_eq!(reader.tile_resolution(), STREAMED_TERRAIN_RESOLUTION);
        assert_eq!(reader.meters_per_sample(), STREAMED_TERRAIN_MPS);

        // The world is far wider than the finest streaming radius, so a cut over
        // it is genuinely partial.
        let span = (STREAMED_TERRAIN_RESOLUTION as f64 - 1.0) * STREAMED_TERRAIN_MPS;
        assert!(streamed_terrain_world_size() > 4.0 * span);

        // The payload is a pure function of the generators (the cook, and the
        // fixture setup, must be able to reproduce it byte for byte).
        assert_eq!(asset.as_bytes(), streamed_terrain_asset().as_bytes());

        // The level ships NO tiles — the whole point of the asset ref.
        let doc = streamed_terrain_scene();
        let (data, _) = doc
            .terrain_data_and_origin(STREAMED_TERRAIN_TERRAIN_GUID)
            .expect("terrain entity present");
        assert!(data.is_empty(), "a streamed level must ship no tiles");
    }

    /// The two scripted camera paths must really be different, or gate (c) would
    /// prove nothing.
    /// The partitioned-world scene survives the P3 discipline (save→load→save is
    /// byte-identical) **and** carries the partition settings + the two v10
    /// component markers the gate depends on.
    #[test]
    fn partitioned_world_saves_and_reloads_byte_identical() {
        use inf_ecs::components::{AlwaysLoaded, StreamingSource};
        let doc = partitioned_world_scene();
        let file = crate::scene::serialize::to_scene_file(&doc);
        let bytes1 = crate::scene::serialize::encode(&file).unwrap();
        let mut back = SceneDoc::new();
        crate::scene::serialize::apply_to_doc(
            &mut back,
            &crate::scene::serialize::decode(&bytes1).unwrap(),
        );
        let bytes2 =
            crate::scene::serialize::encode(&crate::scene::serialize::to_scene_file(&back))
                .unwrap();
        assert_eq!(bytes1, bytes2, "save→load→save must be byte-identical");

        // The settings block is what makes this a partitioned level.
        let settings = back.settings();
        assert!(settings.partition.enabled);
        assert_eq!(settings.partition.cell_size_m, PARTITIONED_CELL_SIZE_M);
        assert_eq!(
            settings.partition.activation_radius_m,
            PARTITIONED_ACTIVATION_RADIUS_M
        );

        // One prop per cell, plus the far child, plus the three persistent-ish
        // entities (manager / sun / walker).
        let n = (PARTITIONED_GRID * PARTITIONED_GRID) as usize;
        assert_eq!(file.entities.len(), n + 4);

        let w = back.world();
        let src = w.entity_of(PARTITIONED_WALKER_GUID).expect("walker");
        assert!(w.world().get::<StreamingSource>(src).is_some());
        let sun = w.entity_of(PARTITIONED_SUN_GUID).expect("sun");
        assert!(w.world().get::<AlwaysLoaded>(sun).is_some());
        // The far child really is parented to the far-corner prop.
        let child = w.entity_of(PARTITIONED_CHILD_GUID).expect("child");
        let parent = w
            .entity_of(partitioned_prop_guid(
                PARTITIONED_GRID - 1,
                PARTITIONED_GRID - 1,
            ))
            .expect("far prop");
        assert_eq!(w.parent_of(child), Some(parent));
    }

    /// The scripted walk really crosses cells — otherwise the streaming gate
    /// would be asserting over a world that never streams.
    #[test]
    fn partitioned_walk_crosses_every_cell_in_its_row() {
        use std::collections::BTreeSet;
        let seen: BTreeSet<i32> = (0..PARTITIONED_GRID as usize * 3)
            .map(|i| {
                let p = partitioned_walk_point(i);
                (p.x / PARTITIONED_CELL_SIZE_M).floor() as i32
            })
            .collect();
        assert_eq!(
            seen,
            (0..PARTITIONED_GRID).collect::<BTreeSet<i32>>(),
            "the walk must visit every cell of row z=0"
        );
        // …and each prop sits unambiguously inside exactly one cell.
        for cz in 0..PARTITIONED_GRID {
            for cx in 0..PARTITIONED_GRID {
                let p = partitioned_prop_position(cx, cz);
                assert_eq!((p.x / PARTITIONED_CELL_SIZE_M).floor() as i32, cx);
                assert_eq!((p.z / PARTITIONED_CELL_SIZE_M).floor() as i32, cz);
            }
        }
    }

    #[test]
    fn streamed_terrain_camera_paths_diverge() {
        let a: Vec<_> = (0..40).map(streamed_terrain_camera_a).collect();
        let b: Vec<_> = (0..40).map(streamed_terrain_camera_b).collect();
        assert_ne!(a, b);
        let far = a
            .iter()
            .zip(&b)
            .map(|(p, q)| (*p - *q).length())
            .fold(0.0f64, f64::max);
        assert!(far > streamed_terrain_world_size() * 0.2, "paths too close");
        // And the walk crosses the world (so sim residency really slides).
        let walk: Vec<_> = (0..120).map(streamed_terrain_walk_point).collect();
        let dx = walk.last().unwrap().x - walk[0].x;
        assert!(
            dx.abs() > streamed_terrain_world_size() * 0.5,
            "walk too short"
        );
    }

    // ── Character-demo gate test (a): byte-identical save/reload ────────────

    /// GATE (a) — the P3 discipline applied to the character demo: save → load →
    /// save is byte-identical (a genuine schema-v5 payload), and the reloaded doc
    /// keeps the full P11 animation/character component set on the character.
    #[test]
    fn character_demo_saves_and_reloads_byte_identical() {
        use inf_ecs::components::{
            AnimStateMachine, CharacterController3D, Collider3D, RigidBody3D, RootMotion,
            SkeletalMesh,
        };

        let doc = character_demo_scene();
        let bytes1 =
            crate::scene::serialize::encode(&crate::scene::serialize::to_scene_file(&doc)).unwrap();
        assert_eq!(
            bytes1[0],
            crate::scene::serialize::SCHEMA_VERSION as u8,
            "character-demo writes at the current schema"
        );

        let mut doc2 = SceneDoc::new();
        crate::scene::serialize::apply_to_doc(
            &mut doc2,
            &crate::scene::serialize::decode(&bytes1).unwrap(),
        );
        let bytes2 =
            crate::scene::serialize::encode(&crate::scene::serialize::to_scene_file(&doc2))
                .unwrap();
        assert_eq!(
            bytes1, bytes2,
            "character-demo save→load→save must be byte-identical"
        );

        // The character keeps every persisted anim/character component + its refs.
        let ce = doc2.entity_of(CHARACTER_DEMO_CHARACTER_GUID).unwrap();
        let w = doc2.world().world();
        assert_eq!(
            w.get::<SkeletalMesh>(ce).unwrap().skeleton,
            Some(CHARACTER_DEMO_SKELETON_GUID)
        );
        assert_eq!(
            w.get::<AnimStateMachine>(ce).unwrap().sm,
            Some(CHARACTER_DEMO_SM_GUID)
        );
        assert!(w.get::<RootMotion>(ce).is_some());
        assert!(w.get::<CharacterController3D>(ce).is_some());
        assert!(w.get::<Collider3D>(ce).is_some());
        assert!(w.get::<RigidBody3D>(ce).is_some());
    }

    /// The committed anim assets decode + the state machine references the three
    /// committed clip GUIDs (the cook's SM→clip closure walks exactly these).
    #[test]
    fn character_demo_state_machine_references_its_clips() {
        use inf_anim::state_machine::Motion;
        let sm = character_demo_state_machine();
        let clip_of = |i: usize| match &sm.states[i].motion {
            Motion::Clip(c) => uuid::Uuid::from_bytes(*c),
            _ => panic!("expected a clip motion"),
        };
        assert_eq!(clip_of(0), CHARACTER_DEMO_IDLE_CLIP_GUID);
        assert_eq!(clip_of(1), CHARACTER_DEMO_RUN_CLIP_GUID);
        assert_eq!(clip_of(2), CHARACTER_DEMO_JUMP_CLIP_GUID);
        // The actor blueprint round-trips through its committed encoding.
        let class = character_demo_class();
        assert_eq!(decode_actor(&encode_actor(&class).unwrap()).unwrap(), class);
    }

    /// **The measurement behind P29.1's `exit_time` ruling, kept as an arm.**
    ///
    /// P24.1 ledgered that wiring a period resolver "would move every existing
    /// machine's transition timing", and that sentence was the reason the field
    /// stayed inert for four phases. Measured before it was wired: **no authored
    /// machine in this repository gates on `exit_time` at all** — not this
    /// sample (which is what the committed `Locomotion.inf_sm` is blessed from),
    /// and not the wizard's generated locomotion, which is every machine an
    /// Infini Engine project gets without an author typing one. The retiming
    /// was real in principle and empty in fact.
    ///
    /// This is an assertion rather than a paragraph because the claim has an
    /// expiry date: the first authored `exit_time` in committed content makes it
    /// false, and the ledger entry in ROADMAP section 13 has to be rewritten
    /// rather than quietly outlived. That is what this arm is for.
    #[test]
    fn no_authored_machine_gates_on_exit_time_so_the_live_resolver_retimed_nothing() {
        use inf_anim::locomotion::{build_locomotion, locomotion_machine, GaitParams};
        use inf_anim::{BodyParams, BodyPlan};

        let gated = |sm: &inf_anim::StateMachine| -> Vec<(usize, f64)> {
            sm.transitions
                .iter()
                .enumerate()
                .filter_map(|(i, t)| t.exit_time.map(|x| (i, x)))
                .collect()
        };

        let demo = character_demo_state_machine();
        assert!(
            gated(&demo).is_empty(),
            "the character-demo machine now gates on exit_time {:?} — the live \
             period resolver retimes it, and ROADMAP section 13's ruling is stale",
            gated(&demo)
        );

        // The wizard's machine: every project's default locomotion.
        let rig = inf_anim::build_template(BodyPlan::Biped, &BodyParams::default()).unwrap();
        let set = build_locomotion(BodyPlan::Biped, &rig, &GaitParams::default()).unwrap();
        let generated = locomotion_machine(&set, [1; 16], [2; 16], [3; 16]);
        assert!(
            gated(&generated).is_empty(),
            "the generated locomotion machine now gates on exit_time {:?}",
            gated(&generated)
        );

        // NOT VACUOUS: these machines have transitions to have found one on.
        assert_eq!(demo.transitions.len(), 5);
        assert_eq!(generated.transitions.len(), 4);
    }

    // ── Terrain-demo gate test (a): byte-identical save/reload ─────────────

    /// GATE (a) — the P3 discipline applied to the terrain-demo: save → load →
    /// save is byte-identical, and the reloaded doc keeps the terrain (heights +
    /// materialized splat weights) and the PCG volume's graph ref.
    #[test]
    fn terrain_demo_saves_and_reloads_byte_identical() {
        use inf_ecs::components::{PcgVolume, Terrain};

        let doc = terrain_demo_scene();
        let bytes1 =
            crate::scene::serialize::encode(&crate::scene::serialize::to_scene_file(&doc)).unwrap();
        assert_eq!(
            bytes1[0],
            crate::scene::serialize::SCHEMA_VERSION as u8,
            "terrain-demo writes at the current schema"
        );

        let mut doc2 = SceneDoc::new();
        crate::scene::serialize::apply_to_doc(
            &mut doc2,
            &crate::scene::serialize::decode(&bytes1).unwrap(),
        );
        let bytes2 =
            crate::scene::serialize::encode(&crate::scene::serialize::to_scene_file(&doc2))
                .unwrap();
        assert_eq!(
            bytes1, bytes2,
            "terrain-demo save→load→save must be byte-identical"
        );

        // Terrain survives with a value probe matching the generator function.
        let te = doc2.entity_of(TERRAIN_DEMO_TERRAIN_GUID).unwrap();
        let terrain = doc2.world().world().get::<Terrain>(te).unwrap();
        assert!(
            terrain.data.tile_count() >= 4,
            "multi-tile terrain persists"
        );
        let probe = terrain
            .data
            .height_at(glam::DVec2::new(16.0, 16.0))
            .unwrap();
        assert!(
            (probe - terrain_demo_height(16.0, 16.0)).abs() < 1e-3,
            "height probe {probe} matches the generator function"
        );
        assert!(
            terrain.data.tiles().any(|(_, t)| !t.weights_are_default()),
            "painted (materialized) splat weights persist"
        );

        // The PCG volume keeps its graph ref (its evaluated cache is not persisted).
        let pe = doc2.entity_of(TERRAIN_DEMO_PCG_GUID).unwrap();
        let vol = doc2.world().world().get::<PcgVolume>(pe).unwrap();
        assert_eq!(vol.graph, Some(TERRAIN_DEMO_PCG_ASSET_GUID));
        assert!(vol.evaluated.is_empty());
    }

    /// The demo's PCG graph, evaluated over the demo terrain, places a few hundred
    /// instances — the in-editor reference the runtime gate matches against.
    #[test]
    fn terrain_demo_pcg_scatters_a_few_hundred_instances() {
        use inf_pcg::height::FnHeight;
        use inf_pcg::Region;

        let doc = terrain_demo_scene();
        let te = doc.entity_of(TERRAIN_DEMO_TERRAIN_GUID).unwrap();
        let data = doc
            .world()
            .world()
            .get::<inf_ecs::components::Terrain>(te)
            .unwrap()
            .data
            .clone();
        let provider = FnHeight::new(move |x, z| data.height_at(glam::DVec2::new(x, z)));
        let region = Region::from_xz(0.0, 0.0, TERRAIN_DEMO_SPAN, TERRAIN_DEMO_SPAN);
        let insts = inf_pcg::evaluate(&terrain_demo_pcg_document(), &provider, region);
        assert!(
            insts.len() > 100,
            "expected a few hundred instances, got {}",
            insts.len()
        );
        // Deterministic across two evaluations.
        let insts2 = inf_pcg::evaluate(&terrain_demo_pcg_document(), &provider, region);
        assert_eq!(insts, insts2);
    }

    // ── Physics-playground gate scene (P12.4) ──────────────────────────────

    /// The P3 discipline applied to the playground: save → load → save is
    /// byte-identical (a genuine schema-v6 payload), and the reloaded doc keeps the
    /// joints (incl. the `other` entity refs), the audio sources (incl. `clip`
    /// refs), the listener, and the collision-layer / CCD collider fields.
    #[test]
    fn physics_playground_saves_and_reloads_byte_identical() {
        use inf_ecs::components::{
            AudioListener, AudioSource, Collider3D, Joint3D, JointKind3D, RigidBody3D,
        };

        let doc = physics_playground_scene();
        let bytes1 =
            crate::scene::serialize::encode(&crate::scene::serialize::to_scene_file(&doc)).unwrap();
        assert_eq!(
            bytes1[0],
            crate::scene::serialize::SCHEMA_VERSION as u8,
            "physics-playground writes at the current schema"
        );

        let mut doc2 = SceneDoc::new();
        crate::scene::serialize::apply_to_doc(
            &mut doc2,
            &crate::scene::serialize::decode(&bytes1).unwrap(),
        );
        let bytes2 =
            crate::scene::serialize::encode(&crate::scene::serialize::to_scene_file(&doc2))
                .unwrap();
        assert_eq!(
            bytes1, bytes2,
            "physics-playground save→load→save must be byte-identical"
        );

        let w = doc2.world().world();
        // The motorized spinner joint persists with its motor + other ref.
        let we = doc2.entity_of(PLAYGROUND_SPINNER_WHEEL_GUID).unwrap();
        let sj = w.get::<Joint3D>(we).expect("spinner joint persists");
        assert_eq!(sj.kind, JointKind3D::Revolute);
        assert_eq!(
            sj.other,
            inf_ecs::EntityRef::new(PLAYGROUND_SPINNER_HUB_GUID)
        );
        assert!(sj.motor_enabled);
        assert_eq!(sj.motor_target_vel, 8.0);
        // The spinner's autoplay/looping/occluded AudioSource persists.
        let src = w.get::<AudioSource>(we).expect("spinner audio persists");
        assert_eq!(src.clip, Some(PLAYGROUND_SPINNER_CLIP_GUID));
        assert!(src.autoplay && src.looping && src.occlusion && src.spatial);
        // The CCD bullet's collider/body fields persist.
        let be = doc2.entity_of(PLAYGROUND_BULLET_GUID).unwrap();
        assert!(w.get::<RigidBody3D>(be).unwrap().ccd_enabled);
        // The ghost pair's collision-layer filter persists (empty filter).
        let ge = doc2.entity_of(PLAYGROUND_GHOST_A_GUID).unwrap();
        assert_eq!(w.get::<Collider3D>(ge).unwrap().collision_filter, 0);
        // The sensor plate is a persisted trigger volume.
        let se = doc2.entity_of(PLAYGROUND_SENSOR_GUID).unwrap();
        assert!(w.get::<Collider3D>(se).unwrap().sensor);
        // The camera carries the active listener.
        let ce = doc2.entity_of(PLAYGROUND_CAMERA_GUID).unwrap();
        assert!(w.get::<AudioListener>(ce).unwrap().active);
        // The ragdoll produced 8 bodies + 7 joints (its descs mapped to components).
        let ragdoll_joints = (0..8)
            .filter_map(|i| doc2.entity_of(Uuid::from_u128(PLAYGROUND_RAGDOLL_BASE_GUID + i)))
            .filter(|&e| w.get::<Joint3D>(e).is_some())
            .count();
        assert_eq!(ragdoll_joints, 7, "ragdoll wires 7 parent joints");
    }
}
