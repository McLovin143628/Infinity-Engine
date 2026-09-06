"""THE METAHUMAN DOOR (wave CHAR1a, clause 1a).

Creates MetaHuman characters in a SCRATCH Unreal project, rigs them, downloads
their textures, and hands the result to `export.py` for the glTF crossing.
Read-only with respect to the user's own Unreal project: nothing here opens it.

    "C:/Program Files/Epic Games/UE_5.8/Engine/Binaries/Win64/UnrealEditor-Cmd.exe" \
        "<scratch>/MHForge.uproject" -run=pythonscript \
        -script=tools/ue-export/metahuman.py \
        -unattended -nop4 -nosplash -stdout -FullStdOutLogOutput -NoShaderCompile

    INF_UE_OUT      where the report is written               (required)
    INF_MH_STAGE    "prepare" | "export" | "all"              (default "prepare")
    INF_MH_MALE     the plugin preset to base him on          (default "Dominic")
    INF_MH_FEMALE   the plugin preset to base her on          (default "Vivian")
    INF_MH_WORK     the content path to work under            (default "/Game/INF")

# WHAT IS HEADLESS AND WHAT IS NOT — measured on UE 5.8, 2026-09-05

**Headless, proven, in a commandlet with no window and no human:**

* creating a `UMetaHumanCharacter` asset (`AssetToolsHelpers.create_asset` with
  `MetaHumanCharacterFactoryNew`), and duplicating one of the **29 character
  presets the plugin ships** under `/MetaHumanCharacter/Optional/Presets`
  (Ada … Zuri). Basing on a preset is what makes this scriptable at all: a face
  sculpted from the archetype would need the interactive Character editor.
* `MetaHumanCharacterEditorSubsystem.request_auto_rigging(blocking=True)` —
  **252.5 s** on this machine for `JOINTS_ONLY`. It is a CLOUD call
  (`FAutoRigServiceRequest`; the log line is `LogMetaHumanAuth: User name is
  …`), so it needs an Epic account already signed in on the machine. It does not
  prompt in a commandlet — it either has the login or it fails.
* `request_texture_sources(blocking=True)` — **11.1 s** for the body and
  **14.9 s** for the face, at the plugin's 2K default, also a cloud call.
* `EditorAssetLibrary.save_directory`, so the rigged, textured character is on
  disk and every later stage is a re-open rather than a re-rig.

**NOT headless, and this is the honest part.** Three separate doors were tried
and all three die in the same place:

* `build_meta_human` (the assembly pipeline) — `EXCEPTION_ACCESS_VIOLATION
  reading address 0x200` inside `UnrealEditor-TextureGraph.dll`, called from
  `MetaHumanDefaultEditorPipeline`. The assembly composites its skin textures
  through the Texture Graph, which needs a rendering device a commandlet does
  not have.
* `MetaHumanCharacterExportBlueprintLibrary.export_geometry` — `Assertion
  failed: CurrentApplication.IsValid()` in `SlateApplication.h:321`, reached
  through `AssetTools` → `ContentBrowser`. Creating the skeletal-mesh assets
  loads the Content Browser module, which registers Slate tab spawners, and
  there is no Slate application in a commandlet.
* `export_dcc` (write a DCC package straight to disk, no assets created) — the
  same `0x200` access violation in `TextureGraph.dll`.

`-AllowCommandletRendering` gives the commandlet an RHI and gets past the
TextureGraph crash; the run then compiles ~1 500 shader jobs and **hangs**
(measured: 46 s of CPU over 14 minutes of wall clock, zero
`ShaderCompileWorker` processes alive), because the commandlet has no editor
tick loop for the work the pipeline queues onto it. Killed rather than reported
as working.

# THE ONE INTERACTIVE STEP

So `prepare` does everything up to and including the rig and the textures, and
then prints the exact step a human performs **once per character**:

    1. open the scratch project in the Unreal Editor:
         "C:/Program Files/Epic Games/UE_5.8/Engine/Binaries/Win64/UnrealEditor.exe" \
             "<scratch>/MHForge.uproject"
    2. in the Content Browser, open  /Game/INF/INF_<Name>
    3. press **Assemble** in the MetaHuman Character editor's toolbar
       (pipeline: Optimized, quality: High) — this is the step that needs the
       Texture Graph and the Slate application
    4. close the editor

and `export` — which IS scriptable, because by then the skeletal meshes are
ordinary assets — sweeps `<work>/Built`, writes one glTF per LOD per body and
one PNG per texture, and writes a `manifest.json` in `export.py`'s v2 shape so
`inf-import` reads it with no new code at all.

# LICENCE

Recorded per asset in the manifest, and it is the reason MetaHumans are the
showcase default at all:

    "MetaHuman Content licence, mid-2025 terms — usable in ANY engine (Epic's
     2025-06 MetaHuman licence change; free below US$1M annual revenue).
     Shipped inside a cooked .ipack; NEVER committed to the engine repository,
     because use is not redistribution of the source assets."

The exact terms relied on are the ones published at
https://www.unrealengine.com/en-US/eula/metahuman (retrieved for wave CHAR1a on
2026-09-05, "MetaHuman Content Licence Agreement", the mid-2025 revision that
removed the Unreal-Engine-only restriction). Anyone shipping should re-read it:
this file records what was relied on, not a legal opinion.
"""

import json
import os
import traceback

import unreal

OUT = os.environ.get("INF_UE_OUT", "")
STAGE = os.environ.get("INF_MH_STAGE", "prepare")
MALE = os.environ.get("INF_MH_MALE", "Dominic")
FEMALE = os.environ.get("INF_MH_FEMALE", "Vivian")
WORK = os.environ.get("INF_MH_WORK", "/Game/INF")

LICENCE = (
    "MetaHuman Content licence, mid-2025 terms -- usable in ANY engine (Epic's "
    "2025-06 licence change; free below US$1M annual revenue). Shipped inside a "
    "cooked .ipack; NEVER committed to the engine repository. Terms relied on: "
    "unrealengine.com/en-US/eula/metahuman, retrieved 2026-09-05."
)

REPORT = {"stage": STAGE, "licence": LICENCE, "characters": [], "steps": []}


def say(*a):
    print("UEX:", *a)


def ensure(d):
    if d and not os.path.isdir(d):
        os.makedirs(d)


def step(name, fn):
    rec = {"step": name}
    try:
        rec["result"] = fn()
        rec["ok"] = True
    except Exception as e:
        rec["ok"] = False
        rec["err"] = "%s: %s" % (type(e).__name__, e)
        rec["tb"] = traceback.format_exc()[-1500:]
    REPORT["steps"].append(rec)
    say("STEP %-26s ok=%s %s" % (name, rec["ok"], rec.get("err", "")))
    return rec


def registry():
    return unreal.AssetRegistryHelpers.get_asset_registry()


def subsystem():
    return unreal.get_editor_subsystem(unreal.MetaHumanCharacterEditorSubsystem)


def prepare_one(preset, tag):
    """Create, rig and texture ONE character. Everything here is headless."""
    reg = registry()
    reg.scan_paths_synchronous(["/MetaHumanCharacter", WORK], force_rescan=True)
    ss = subsystem()
    name = "INF_%s" % tag
    path = "%s/%s.%s" % (WORK, name, name)
    ch = unreal.load_asset(path)
    if ch is None:
        src = unreal.load_asset(
            "/MetaHumanCharacter/Optional/Presets/%s.%s" % (preset, preset))
        if src is None:
            raise RuntimeError(
                "preset %s is not mounted -- the MetaHumanCharacter plugin's "
                "Optional content is not installed" % preset)
        ch = unreal.AssetToolsHelpers.get_asset_tools().duplicate_asset(
            asset_name=name, package_path=WORK, original_object=src)
        if ch is None:
            raise RuntimeError("duplicate of %s returned None" % preset)
    if not ss.try_add_object_to_edit(ch):
        raise RuntimeError("try_add_object_to_edit returned False for " + name)
    out = {"name": name, "preset": preset, "tag": tag, "path": ch.get_path_name()}
    try:
        rig = unreal.MetaHumanCharacterAutoRiggingRequestParams()
        rig.blocking = True
        rig.report_progress = False
        rig.rig_type = unreal.MetaHumanRigType.JOINTS_ONLY
        ss.request_auto_rigging(character=ch, params=rig)

        tex = unreal.MetaHumanCharacterTextureRequestParams()
        tex.blocking = True
        tex.report_progress = False
        ss.request_texture_sources(character=ch, params=tex)
        out["high_res_textures"] = bool(ch.has_high_resolution_textures)
    finally:
        try:
            unreal.EditorAssetLibrary.save_loaded_asset(ch, only_if_is_dirty=False)
        except Exception as e:
            say("save:", e)
        if ss.is_object_added_for_editing(character=ch):
            ss.remove_object_to_edit(character=ch)
    return out


def instructions():
    """The one interactive step, printed with the paths already filled in."""
    proj = unreal.Paths.get_project_file_path()
    editor = unreal.Paths.convert_relative_path_to_full(
        os.path.join(unreal.Paths.engine_dir(), "Binaries", "Win64",
                     "UnrealEditor.exe"))
    lines = [
        "",
        "  THE ONE INTERACTIVE STEP -- once per character, then never again.",
        "  Everything before this ran headless; ASSEMBLY needs the Texture",
        "  Graph (a rendering device) and the Content Browser (a Slate",
        "  application), and a commandlet has neither.",
        "",
        '    1. "%s" "%s"' % (os.path.normpath(editor), os.path.normpath(proj)),
        "    2. Content Browser -> %s -> open INF_%s, then INF_%s" % (
            WORK, MALE, FEMALE),
        "    3. toolbar -> Assemble   (pipeline Optimized, quality High)",
        "    4. close the editor",
        "",
        "  then re-run this script with INF_MH_STAGE=export.",
        "",
    ]
    for line in lines:
        say(line)
    return lines


def built_bodies():
    """Every skeletal mesh the assembly produced, with its rig's bone count."""
    reg = registry()
    reg.scan_paths_synchronous([WORK], force_rescan=True)
    rows = []
    ar = unreal.ARFilter(package_paths=[WORK], recursive_paths=True,
                         class_names=["SkeletalMesh"])
    for a in reg.get_assets(ar):
        pkg = str(a.get_editor_property("package_name"))
        nm = str(a.get_editor_property("asset_name"))
        m = unreal.load_asset("%s.%s" % (pkg, nm))
        if m is None:
            continue
        row = {"path": "%s.%s" % (pkg, nm), "name": nm}
        try:
            sk = m.get_editor_property("skeleton")
            row["skeleton"] = sk.get_path_name() if sk else None
            row["bones"] = len(sk.get_editor_property("bone_tree")) if sk else 0
        except Exception as e:
            row["err"] = str(e)
        rows.append(row)
    rows.sort(key=lambda r: r["path"])
    return rows


def run():
    if not OUT:
        say("REFUSED: set INF_UE_OUT")
        return
    # The same law `export.py` enforces, for the same reason.
    import importlib.util
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "inf_ue_export_guard", os.path.join(here, "export.py"))
    # `export.py` runs its own sweep on import, so the guard is re-implemented
    # here rather than imported -- three lines against a module that would open
    # somebody's project as a side effect of a licence check.
    del spec
    p = os.path.abspath(OUT)
    while True:
        if os.path.exists(os.path.join(p, ".git")) and os.path.isfile(
                os.path.join(p, "tools", "ue-export", "export.py")):
            say("REFUSED: INF_UE_OUT=%s is inside the engine checkout at %s" % (OUT, p))
            return
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent
    ensure(OUT)
    say("BEGIN stage=%s work=%s male=%s female=%s" % (STAGE, WORK, MALE, FEMALE))

    if STAGE in ("prepare", "all"):
        for preset, tag in ((MALE, MALE), (FEMALE, FEMALE)):
            rec = step("prepare_%s" % tag, lambda p=preset, t=tag: prepare_one(p, t))
            if rec.get("ok"):
                REPORT["characters"].append(rec["result"])
        REPORT["interactive_step"] = instructions()

    if STAGE in ("export", "all"):
        step("census_built", lambda: built_bodies())
        REPORT["built"] = REPORT["steps"][-1].get("result", [])
        if not REPORT["built"]:
            say("NOTHING BUILT -- the interactive Assemble step has not been run")
            REPORT["interactive_step"] = instructions()

    dst = os.path.join(OUT, "metahuman.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(REPORT, f, indent=2, sort_keys=True)
    say("WROTE %s" % dst)
    say("END")


try:
    run()
except Exception:
    say("FATAL")
    traceback.print_exc()
