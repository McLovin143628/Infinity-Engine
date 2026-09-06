//! **`inf-import`** — the headless Infini side of the Unreal bridge (wave
//! ASSET0, clause 2).
//!
//! ```text
//! inf-import --manifest <manifest.json> --into <project-dir>
//!            [--pack <name>]…            only these packs
//!            [--max-texture <n>]         ceiling on a texture's longest side (0 = source)
//!            [--dest <subfolder>]        under <project>/Content (default "UE")
//!            [--bind <Stem>=<key>]…      write a material at a committed GUID
//!            [--no-meshes]               materials and textures only
//!            [--character-lods <n>]      LOD rungs to store per character (default 3)
//!            [--retarget-to <objpath>]   the rig every clip is retargeted onto
//!            [--rebind-character <key>]  write that body at the starter GUIDs
//!            [--dry-run]                 read the manifest, write nothing
//! ```
//!
//! Everything it does is [`inf_editor_core::assets::ue_import`]; this file is
//! argument parsing and a report. See that module for the PBR remap, the clamp
//! and what a rebind is for.

use std::path::PathBuf;
use std::process::ExitCode;

use inf_editor_core::assets::ue_import::{import_manifest, UeImportOptions};
use inf_editor_core::assets::AssetProject;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--help" || a == "-h") || args.is_empty() {
        print_help();
        return ExitCode::SUCCESS;
    }
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("inf-import: {e}");
            ExitCode::FAILURE
        }
    }
}

fn print_help() {
    println!(
        "inf-import {} — Unreal → Infini asset bridge (the import side)\n\n\
         USAGE:\n  \
             inf-import --manifest <manifest.json> --into <project-dir>\n             \
             [--pack <name>]… [--max-texture <n>] [--dest <sub>]\n             \
             [--bind <Stem>=<material-key>]… [--no-meshes]\n             \
             [--character-lods <n>] [--retarget-to <objpath>] [--dry-run]\n\n\
         The manifest is written by tools/ue-export/export.py. A --bind writes\n\
         an imported material at the GUID the committed ground library assigns\n\
         that stem, so a committed level picks it up without naming licensed\n\
         content: e.g. --bind Road_Asphalt=<the asphalt material's key>.\n\n\
         NOTHING THIS WRITES MAY BE COMMITTED. It goes into a project's Content,\n\
         which for the island is outside this repository -- and since the ASSET0\n\
         audit that is a door rather than a sentence: an --into inside the engine\n\
         checkout is REFUSED before the first texture is decoded.",
        env!("CARGO_PKG_VERSION")
    );
}

fn run(args: &[String]) -> Result<(), String> {
    let mut manifest: Option<PathBuf> = None;
    let mut into: Option<PathBuf> = None;
    let mut dry = false;
    let mut opts = UeImportOptions::default();
    let mut i = 0;
    while i < args.len() {
        let take = |i: &mut usize| -> Result<String, String> {
            *i += 1;
            args.get(*i)
                .cloned()
                .ok_or_else(|| format!("{} needs a value", args[*i - 1]))
        };
        match args[i].as_str() {
            "--manifest" => manifest = Some(PathBuf::from(take(&mut i)?)),
            "--into" => into = Some(PathBuf::from(take(&mut i)?)),
            "--pack" => opts.packs.push(take(&mut i)?),
            "--dest" => opts.dest = take(&mut i)?,
            "--max-texture" => {
                let v = take(&mut i)?;
                opts.max_texture = v
                    .parse()
                    .map_err(|_| format!("--max-texture wants a number, got {v:?}"))?;
            }
            "--bind" => {
                let v = take(&mut i)?;
                let (stem, key) = v
                    .split_once('=')
                    .ok_or_else(|| format!("--bind wants <Stem>=<material-key>, got {v:?}"))?;
                opts.rebinds.push((stem.to_string(), key.to_string()));
            }
            "--no-meshes" => opts.meshes = false,
            "--character-lods" => {
                let v = take(&mut i)?;
                opts.character_lods = v
                    .parse()
                    .map_err(|_| format!("--character-lods wants a number, got {v:?}"))?;
            }
            "--retarget-to" => opts.retarget_to = Some(take(&mut i)?),
            "--rebind-character" => opts.rebind_character = Some(take(&mut i)?),
            "--dry-run" => dry = true,
            other => return Err(format!("unknown option {other:?}")),
        }
        i += 1;
    }
    let manifest = manifest.ok_or("--manifest is required")?;
    let into = into.ok_or("--into is required")?;
    let content = if into.join("Content").is_dir() {
        into.join("Content")
    } else {
        into.clone()
    };
    println!("inf-import: manifest {}", manifest.display());
    println!("inf-import: content  {}", content.display());
    println!(
        "inf-import: packs {} · max-texture {} · dest {} · meshes {} · binds {}",
        if opts.packs.is_empty() {
            "(all)".to_string()
        } else {
            opts.packs.join(",")
        },
        opts.max_texture,
        opts.dest,
        opts.meshes,
        opts.rebinds.len()
    );
    if dry {
        println!("inf-import: --dry-run, nothing written");
        return Ok(());
    }

    let started = std::time::Instant::now();
    let mut project = AssetProject::open(&content).map_err(|e| e.to_string())?;
    let report = import_manifest(&mut project, &manifest, &opts).map_err(|e| e.to_string())?;
    let secs = started.elapsed().as_secs_f64();

    for a in &report.advisories {
        println!("inf-import: ADVISORY {a}");
    }
    for (key, id) in &report.materials {
        println!("inf-import: material {id}  {key}");
    }
    for (key, id, rungs, tris) in &report.meshes {
        println!("inf-import: mesh     {id}  {tris:>7} tris, {rungs} source rungs  {key}");
    }
    for (key, mesh, skel, rungs, tris, joints) in &report.skeletal {
        println!(
            "inf-import: body     {mesh}  {tris:>7} tris, {rungs} rungs, {joints} joints, \
             skeleton {}  {key}",
            skel.map(|s| s.to_string()).unwrap_or_else(|| "NONE".into())
        );
    }
    for (key, id, tracks) in &report.clips {
        println!("inf-import: clip     {id}  {tracks:>4} tracks  {key}");
    }
    for (pack, licence, ship) in &report.licences {
        println!(
            "inf-import: LICENCE  {pack} [{}] {licence}",
            if *ship { "MAY SHIP" } else { "LOCAL ONLY" }
        );
    }
    for (stem, id) in &report.rebinds {
        println!("inf-import: REBOUND  {stem} -> {id}");
    }
    for f in &report.fixtures {
        println!(
            "inf-import: fixture  {} at ({:.2}, {:.2}, {:.2}) m, {:.0} cd, {:.1} m range, sRGB8 {:?}",
            f.name,
            f.offset_m[0],
            f.offset_m[1],
            f.offset_m[2],
            f.intensity,
            f.range_m,
            f.color_srgb8
        );
    }
    println!(
        "inf-import: {} materials, {} textures, {} meshes, {} bodies, {} clips, \
         {} fixtures, {:.1} MB, {secs:.1} s",
        report.materials.len(),
        report.textures.len(),
        report.meshes.len(),
        report.skeletal.len(),
        report.clips.len(),
        report.fixtures.len(),
        report.bytes as f64 / 1_048_576.0,
    );
    Ok(())
}
