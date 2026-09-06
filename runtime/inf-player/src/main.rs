//! Infini Engine standalone player — binary entry.
//!
//! Modes (see [`inf_player::args`]):
//!
//! - **windowed** (default, or `--demo` / `--level`) — open a winit window and
//!   play; the fixed-step gameplay + interpolated rendering live in the library
//!   ([`inf_player::run`]).
//! - **headless** (`--headless --run-frames N [--assert-exit]`) — no window/GPU;
//!   run the sim N steps and print the determinism hash. The CI smoke path.
//! - **pie** (`--pie [--tick-hz N]`) — play-in-editor subprocess: speak the
//!   length-prefixed bincode protocol from `inf_runtime::pie` on stdin/stdout
//!   (stdout is protocol-only; logs go to stderr). Unchanged from Spike D; P9.4
//!   builds the editor side out on top of it.
//! - **embed-probe** (`--embed-probe`, Windows) — the Spike D cross-process
//!   window-embedding probe (proves the P9.4 "PIE window in the viewport hole"
//!   plan). Unchanged.
//!
//! `--pie` and `--embed-probe` are handled here (not via [`inf_player::run`])
//! because they own process stdio / native windows: installing the tracing
//! subscriber that tees logs to stdout would corrupt the PIE protocol stream.
//!
//! # No console window (wave FIX1)
//!
//! A **shipped** player is a game, and a game that opens a black console window
//! beside itself is a defect an author cannot explain away. `inf-studio`'s own
//! `main.rs` has carried this attribute since Phase 1; the player never did, so
//! the Play button spawned one every session — the second half of the same
//! defect is `inf_editor_core::pie`'s `CREATE_NO_WINDOW`, which is what keeps a
//! *debug* player spawned by the editor from allocating one too.
//!
//! **Gated on `debug_assertions`, exactly like the editor's**, because this
//! binary is also the CLI: `--headless --run-frames N` prints the determinism
//! hash to stdout and CI reads it, and CI builds debug. The two `println!` calls
//! in this crate are both on that headless path (`lib.rs`), so the windowed and
//! `--pie` paths — the only ones a release build reaches without an inherited
//! stdout — write nothing to stdout that could fail.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::process::ExitCode;

use inf_player::args::{Args, Mode};
use inf_runtime::pie::{read_msg, write_msg, EditorToPlayer, PlayerToEditor, PIE_PROTOCOL_VERSION};
use inf_runtime::World;

#[cfg(windows)]
mod embed_probe;

fn main() -> ExitCode {
    let args = match Args::from_env() {
        Ok(args) => args,
        Err(msg) => {
            eprintln!("inf-player: {msg}");
            return ExitCode::FAILURE;
        }
    };

    match args.mode {
        Mode::EmbedProbe => {
            #[cfg(windows)]
            {
                embed_probe::run()
            }
            #[cfg(not(windows))]
            {
                eprintln!("inf-player: --embed-probe is Windows-only");
                ExitCode::FAILURE
            }
        }
        Mode::Pie => run_pie(args.tick_hz),
        Mode::Windowed | Mode::Headless => inf_player::run(args),
    }
}

/// The active PIE runtime: nothing yet, the Spike-D toy world, or the **real**
/// content world driven by `RuntimeSim` (the P9.4 PIE==shipping path).
enum Active {
    None,
    Toy(World),
    Real {
        sim: Box<inf_player::runtime_sim::RuntimeSim>,
        frame: u64,
        /// **The input host, made by the FIRST input frame and not before**
        /// (wave FIX1).
        ///
        /// Lazily, and that is the whole design. `PieInputHost::open` reads the
        /// player's own settings file and applies it to the sim — which is right
        /// for a session somebody is driving and wrong for the nine existing
        /// subprocess gates, whose reference is `scene_trace` and which have
        /// never had a `PlayerUi`. A session that sends only `Step` therefore
        /// runs byte for byte what it ran before this wave; a session that sends
        /// an `Input` frame gets the shipped input path, and its in-process twin
        /// opens the same door.
        input: Option<Box<inf_player::pie_drive::PieInputHost>>,
    },
}

/// The PIE loop (Spike D + P9.4): a reader thread turns stdin frames into channel
/// messages; the main loop applies control and either streams the toy world (auto)
/// or steps the real content world (step-driven, deterministic). A `LoadScene`
/// requesting a window hands off to the windowed player. Stdout carries protocol
/// frames only.
fn run_pie(tick_hz: u32) -> ExitCode {
    use std::sync::mpsc;
    use std::time::Duration;

    let (tx, rx) = mpsc::channel::<EditorToPlayer>();
    // **A DECODE FAULT IS NOT AN END OF STREAM** (P24.4, closing the P24.1
    // carried entry).
    //
    // The loop below used to be `while let Ok(msg) = read_msg(..)`, which treated
    // every error the same: a cleanly closed pipe and a frame this build cannot
    // read both broke the loop, dropped the channel, and produced "editor closed
    // the channel; exiting" plus `ExitCode::SUCCESS`. That is exactly what a
    // version-skewed editor produces — an OLDER build writes a positional
    // `ScenePayload` with fewer tail fields, `decode_from_slice` runs off the end
    // *inside* `read_msg`, and `check_version` (which lives one call further in,
    // in `build_world_from_payload`) never runs. To the editor the refusal looked
    // like the user pressing Stop.
    //
    // A second channel rather than a `Result` element type, deliberately: `rx` is
    // handed to `run_pie_window` for the embedded path, and widening the message
    // type would put a fault arm into a window loop that cannot answer one.
    // Bound, stated: a decode fault AFTER the handoff to the windowed player
    // keeps the old behaviour. The fault that matters — the `LoadScene` frame
    // itself — arrives before any handoff, which is where this reports it.
    let (fault_tx, fault_rx) = mpsc::channel::<String>();
    std::thread::spawn(move || {
        let mut stdin = std::io::stdin().lock();
        loop {
            match read_msg::<EditorToPlayer>(&mut stdin) {
                Ok(msg) => {
                    if tx.send(msg).is_err() {
                        break;
                    }
                }
                // The editor went away: the ordinary end of a session.
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                // Anything else is a stream this build cannot read.
                Err(e) => {
                    let _ = fault_tx.send(e.to_string());
                    break;
                }
            }
        }
    });

    let mut stdout = std::io::stdout().lock();
    if write_msg(
        &mut stdout,
        &PlayerToEditor::Ready {
            protocol: PIE_PROTOCOL_VERSION,
        },
    )
    .is_err()
    {
        return ExitCode::FAILURE;
    }
    // The console measurement rides the line the session already wrote, so the
    // editor's real-host arm can read it back off a real subprocess's stderr
    // without a protocol frame. `none` is the shipped answer; see
    // `inf_player::win_host` and `inf_editor_core::pie::player_command`.
    eprintln!(
        "inf-player: PIE session ready (tick-hz {tick_hz}, console {})",
        inf_player::win_host::console_report()
    );

    let mut active = Active::None;
    let mut paused = false;
    let mut tick_duration = if tick_hz == 0 {
        Duration::ZERO
    } else {
        Duration::from_secs_f64(1.0 / tick_hz as f64)
    };

    loop {
        let mut disconnected = false;
        loop {
            match rx.try_recv() {
                Ok(msg) => match handle_msg(
                    msg,
                    &mut active,
                    &mut paused,
                    &mut tick_duration,
                    &mut stdout,
                ) {
                    Control::Continue => {}
                    Control::Exit(code) => return code,
                    Control::RunWindow(payload) => return run_pie_window(*payload, rx, stdout),
                },
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    disconnected = true;
                    break;
                }
            }
        }
        if disconnected {
            return report_stream_end(&fault_rx, &mut stdout);
        }

        // Auto-advance a running runtime (toy streams; real is step-driven so it
        // only auto-runs once Resumed).
        let running = !paused && !matches!(active, Active::None);
        if running {
            if let Some(code) = advance_and_report(&mut active, &mut stdout) {
                return code;
            }
            if !tick_duration.is_zero() {
                std::thread::sleep(tick_duration);
            }
        } else {
            match rx.recv_timeout(Duration::from_millis(20)) {
                Ok(msg) => match handle_msg(
                    msg,
                    &mut active,
                    &mut paused,
                    &mut tick_duration,
                    &mut stdout,
                ) {
                    Control::Continue => {}
                    Control::Exit(code) => return code,
                    Control::RunWindow(payload) => return run_pie_window(*payload, rx, stdout),
                },
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    return report_stream_end(&fault_rx, &mut stdout);
                }
            }
        }
    }
}

/// **Why the stream ended**, and what to tell the editor about it (P24.4).
///
/// Two outcomes, and telling them apart is the whole of the P24.1 carried entry:
///
/// * the reader thread posted a **fault** — a frame arrived that this build
///   could not decode. Almost always a version-skewed pair: the envelope is
///   positional bincode, so an older editor's `ScenePayload` is a short read.
///   The editor gets a `PlayerToEditor::Error` **naming the schema**, and the
///   process exits non-zero, because a refusal that exits 0 is indistinguishable
///   from the user pressing Stop.
/// * no fault — the pipe closed cleanly. The editor went away; exit 0, exactly
///   as before.
///
/// The fault is posted *before* the reader drops its sender, so by the time this
/// observes a disconnected channel the fault (if there is one) is already in its
/// own queue.
fn report_stream_end(
    faults: &std::sync::mpsc::Receiver<String>,
    stdout: &mut impl std::io::Write,
) -> ExitCode {
    match faults.try_recv() {
        Ok(detail) => {
            let message = format!(
                "cannot decode a PIE frame ({detail}) — the editor and the player \
                 disagree about the message SCHEMA (this build speaks scene payload \
                 v{}); rebuild both from the same commit",
                inf_runtime::pie::SCENE_PAYLOAD_VERSION
            );
            eprintln!("inf-player: {message}");
            let _ = write_msg(stdout, &PlayerToEditor::Error { message });
            ExitCode::FAILURE
        }
        Err(_) => {
            eprintln!("inf-player: editor closed the channel; exiting");
            ExitCode::SUCCESS
        }
    }
}

/// What a handled control message asks the loop to do next.
enum Control {
    Continue,
    Exit(ExitCode),
    /// A `LoadScene` requested a window: leave the headless loop and run the
    /// windowed player over `payload` (embedded / new-window PIE).
    ///
    /// Boxed: a `ScenePayload` carries every streamed asset the level references
    /// and grows each time the envelope does, so inlining it would make *every*
    /// `Control` — overwhelmingly `Continue` — that big.
    RunWindow(Box<inf_runtime::pie::ScenePayload>),
}

/// Advance the active runtime one fixed step and stream a `Frame`. `Some(code)`
/// means the editor closed stdout — exit cleanly.
fn advance_and_report(active: &mut Active, stdout: &mut impl std::io::Write) -> Option<ExitCode> {
    let frame = match active {
        Active::Toy(world) => {
            world.step();
            PlayerToEditor::Frame {
                frame: world.frame,
                state_hash: world.state_hash(),
                actors: world.actor_states(),
            }
        }
        Active::Real { sim, frame, input } => {
            // FIX1: whatever the last input frame resolved to, or nothing when
            // no frame has ever arrived. `RuntimeInput::default()` used to be
            // written here unconditionally, which meant a session that held a
            // key and then pressed Resume ran with its hands off the controls.
            let held = input
                .as_ref()
                .map(|h| h.held())
                .unwrap_or_else(inf_player::runtime_sim::RuntimeInput::default);
            sim.step_once(held);
            *frame += 1;
            PlayerToEditor::Frame {
                frame: *frame,
                state_hash: inf_player::step_state_hash(sim),
                actors: Vec::new(),
            }
        }
        Active::None => return None,
    };
    if write_msg(stdout, &frame).is_err() {
        eprintln!("inf-player: editor closed stdout; exiting");
        return Some(ExitCode::SUCCESS);
    }
    None
}

/// Apply one control message.
fn handle_msg(
    msg: EditorToPlayer,
    active: &mut Active,
    paused: &mut bool,
    tick_duration: &mut std::time::Duration,
    stdout: &mut impl std::io::Write,
) -> Control {
    let reply = match msg {
        EditorToPlayer::Load(snapshot) => {
            eprintln!(
                "inf-player: loading toy level '{}' ({} actors)",
                snapshot.level,
                snapshot.actors.len()
            );
            let loaded = PlayerToEditor::Loaded {
                level: snapshot.level.clone(),
                actor_count: snapshot.actors.len(),
            };
            *active = Active::Toy(World::from_snapshot(&snapshot));
            *paused = false; // the toy world streams
            loaded
        }
        EditorToPlayer::LoadScene(payload) => {
            eprintln!(
                "inf-player: loading real scene '{}' ({} class(es), windowed {})",
                payload.label,
                payload.classes.len(),
                payload.windowed
            );
            if payload.tick_hz == 0 {
                *tick_duration = std::time::Duration::ZERO;
            } else {
                *tick_duration = std::time::Duration::from_secs_f64(1.0 / payload.tick_hz as f64);
            }
            if payload.windowed {
                // Already boxed on the wire (P22.3) — hand the same allocation on.
                return Control::RunWindow(payload);
            }
            // `sim_from_payload` is the ONE PIE boot seam (P21.4): it makes every
            // attachment a PIE session needs — cells, voxel volumes, state
            // machines, root-motion clips, audio — so this path cannot drift from
            // the in-process `scene_trace` reference the gates compare against.
            match inf_player::sim_from_payload(&payload) {
                Ok(inf_player::PayloadSim {
                    sim,
                    label: level,
                    actor_count,
                    // A headless PIE session draws nothing, so it builds no
                    // render host and the level's block has nowhere to go. The
                    // WINDOWED branch above returned before this match and is
                    // where it is used.
                    render: _,
                }) => {
                    *active = Active::Real {
                        sim: Box::new(sim),
                        frame: 0,
                        input: None,
                    };
                    // Real headless PIE is step-driven (deterministic) until
                    // Resumed — so the trace gate reads exactly N frames.
                    *paused = true;
                    PlayerToEditor::Loaded { level, actor_count }
                }
                Err(e) => {
                    eprintln!("inf-player: build scene failed: {e}");
                    PlayerToEditor::Error { message: e }
                }
            }
        }
        EditorToPlayer::Pause => {
            *paused = true;
            PlayerToEditor::Paused
        }
        EditorToPlayer::Resume => {
            *paused = false;
            PlayerToEditor::Resumed
        }
        EditorToPlayer::Step { count } => {
            for _ in 0..count {
                if let Some(code) = advance_and_report(active, stdout) {
                    return Control::Exit(code);
                }
            }
            state_report(active, *paused)
        }
        EditorToPlayer::Eject => {
            // v1: release input possession. The player keeps running; a true
            // camera hand-back is a documented follow-up (no input is possessed
            // in headless PIE, so this is a clean ack).
            eprintln!("inf-player: eject — input possession released");
            PlayerToEditor::Ejected
        }
        // ── wave FIX1, protocol 3 ──
        EditorToPlayer::Input(input_frame) => {
            let Active::Real { sim, frame, input } = active else {
                // A toy world has no binding table and no character; saying so
                // is better than pretending the frame landed.
                return reply(
                    stdout,
                    PlayerToEditor::Error {
                        message: "an input frame needs a real scene (LoadScene first)".into(),
                    },
                );
            };
            let host = input.get_or_insert_with(|| {
                Box::new(inf_player::pie_drive::PieInputHost::open(
                    inf_player::ui::settings_dir(),
                    inf_player::input::default_map(),
                    sim,
                ))
            });
            let steps = host.apply(sim, &input_frame);
            *frame += u64::from(steps);
            // ONE `Frame` per applied input frame, carrying the hash after its
            // last step. A trace that wants a hash per step sends `steps: 1`,
            // which is the ordinary case and the one every gate uses.
            PlayerToEditor::Frame {
                frame: *frame,
                state_hash: inf_player::step_state_hash(sim),
                actors: Vec::new(),
            }
        }
        EditorToPlayer::Probe { guid } => {
            let named = guid.map(uuid::Uuid::from_bytes);
            let probe = match active {
                Active::Real { sim, frame, input } => inf_player::pie_drive::world_probe(
                    sim,
                    input.as_deref().map(|h| h.ui()),
                    *frame,
                    named,
                ),
                // Nothing real is loaded: an empty probe is the honest answer and
                // it is distinguishable from a loaded one by `entities == 0`.
                _ => inf_runtime::pie::WorldProbe::default(),
            };
            PlayerToEditor::Probe(Box::new(probe))
        }
        EditorToPlayer::SetViewport(_rect) => {
            // Headless PIE has no window; the embedded/windowed path applies the
            // rect. Ack silently by reporting current state.
            state_report(active, *paused)
        }
        EditorToPlayer::Stop => {
            let _ = write_msg(stdout, &PlayerToEditor::Stopped);
            return Control::Exit(ExitCode::SUCCESS);
        }
        EditorToPlayer::InjectPanic => {
            panic!("deliberate PIE panic (injected by editor)");
        }
    };
    if write_msg(stdout, &reply).is_err() {
        return Control::Exit(ExitCode::SUCCESS);
    }
    Control::Continue
}

/// Write one reply and keep going (or stop, if the editor closed stdout). The
/// early-return shape `handle_msg`'s FIX1 arms need, since the function's own
/// tail already writes `reply`.
fn reply(stdout: &mut impl std::io::Write, msg: PlayerToEditor) -> Control {
    if write_msg(stdout, &msg).is_err() {
        return Control::Exit(ExitCode::SUCCESS);
    }
    Control::Continue
}

/// A `State` report reflecting the active runtime's frame count + pause flag.
fn state_report(active: &Active, paused: bool) -> PlayerToEditor {
    let (running, frame) = match active {
        Active::None => (false, 0),
        Active::Toy(w) => (true, w.frame),
        Active::Real { frame, .. } => (true, *frame),
    };
    PlayerToEditor::State(inf_runtime::pie::PlayerState {
        running,
        paused,
        frame,
        last_error: None,
    })
}

/// Run the **windowed** PIE player over a real scene payload (embedded /
/// new-window). Blocks in the winit loop, reporting its window handle and
/// obeying Pause/Resume/Stop/Eject/SetViewport control frames. Needs a GPU +
/// display, so — like every GPU path — it is compile-checked in CI and
/// human-verified live.
fn run_pie_window(
    payload: inf_runtime::pie::ScenePayload,
    rx: std::sync::mpsc::Receiver<EditorToPlayer>,
    mut stdout: impl std::io::Write + 'static,
) -> ExitCode {
    // The same one seam the headless PIE path takes (P21.4) — see `run_pie`.
    let inf_player::PayloadSim {
        sim,
        label,
        actor_count,
        render,
    } = match inf_player::sim_from_payload(&payload) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("inf-player: build scene failed: {e}");
            let _ = write_msg(&mut stdout, &PlayerToEditor::Error { message: e });
            return ExitCode::FAILURE;
        }
    };
    let title = format!("Infini Engine (PIE) — {label}");
    if write_msg(
        &mut stdout,
        &PlayerToEditor::Loaded {
            level: label,
            actor_count,
        },
    )
    .is_err()
    {
        return ExitCode::SUCCESS;
    }
    match inf_player::window::run_pie(
        title,
        1280,
        720,
        sim,
        inf_player::input::default_map(),
        rx,
        Box::new(stdout),
        // Wave FIX2 (`ScenePayload` v13): the derived meshlet DAGs the payload
        // names by PATH — what a rigid `MeshRef.asset` is drawn from. Same shape,
        // same reason as every line below it, and the last of the class: without
        // it `run_pie` built an empty `VmeshRegistry` of its own and the island's
        // four road meshes drew placeholder cubes at the world origin while the
        // editor drew a paved street.
        std::sync::Arc::new(inf_player::vmeshes_from_payload(&payload)),
        // Wave FIX2 / TER2b's open item: the authored meshes a level's scatter
        // kinds name, taken out of the same `meshes` vector the skinned store
        // reads — a *use* of a payload field, not a bump.
        std::sync::Arc::new(inf_player::scatter_meshes_from_payload(&payload)),
        std::sync::Arc::new(inf_player::voxel::VoxelRegistry::from_payload(
            &payload.voxels,
        )),
        // P24.1 (`ScenePayload` v7): the skeletal bytes a windowed PIE session
        // has never had. Same shape, same reason as the voxel line above.
        std::sync::Arc::new(inf_player::skinned::SkinnedRegistry::from_payload(
            &payload.meshes,
            &payload.skeletons,
            &payload.clips,
            // …and the MACHINES (wave CHAR1a audit). Without them every `.inf_sm`
            // lookup missed in Play and the crowd stood in its bind pose.
            &payload.machines,
        )),
        // P26.4 (`ScenePayload` v8): the derived material records + the `.inf_tex`
        // containers a PIE session's surfaces sample. Same shape, same reason as
        // the two lines above — through `materials_from_payload`, which is the
        // payload half of the SAME lookup rule `PackLevelSource::material_content`
        // is the pack half of.
        std::sync::Arc::new(inf_player::materials_from_payload(&payload)),
        // Wave CERT1: the level's own render block, decoded from the payload's
        // `level_bytes` by `sim_from_payload`. This argument used to be
        // `RenderSettingsRecord::default()` inside `run_pie` itself, which made
        // the editor's Play button a DIFFERENT renderer from the shipped build
        // of the same level.
        render,
    ) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("inf-player: PIE window error: {e}");
            ExitCode::FAILURE
        }
    }
}
