"""THE METAHUMAN DOOR (wave CHAR1a, clause 1a).

Creates MetaHuman characters in a SCRATCH Unreal project, rigs them, downloads
their textures, and hands the result to `export.py` for the glTF crossing.
Read-only with respect to the user's own Unreal project: nothing here opens it.

    "C:/Program Files/Epic Games/UE_5.8/Engine/Binaries/Win64/UnrealEditor-Cmd.exe" \
        "<scratch>/MHForge.uproject" -run=pythonscript \
        -script=tools/ue-export/metahuman.py \
        -unattended -nop4 -nosplash -stdout -FullStdOutLogOutput -NoShaderCompile

    INF_UE_OUT      where the report is written               (required)
    INF_MH_STAGE    "prepare"|"assemble"|"combine"|"slots"|"export"
                                                       (default "prepare")
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


BUILD_ROOT = os.environ.get("INF_MH_BUILT", "/Game/INF/Built")
COMMON_ROOT = os.environ.get("INF_MH_COMMON", "/Game/INF/Common")


def assemble_one(tag):
    """ASSEMBLE one prepared character. Needs a LIVE editor -- see the module
    doc: the pipeline composites its skin through the Texture Graph (an RHI)
    and creates its assets through the Content Browser (a Slate application),
    and a commandlet has neither. Driven from the outside by `mh_remote.py`,
    which boots `UnrealEditor.exe` with a window and speaks the Python
    plugin's remote-execution protocol to it.

    Idempotent: a character whose `Built` folder already holds a skeletal mesh
    is reported rather than rebuilt, so a re-run after a partial failure costs
    a registry scan and nothing else.
    """
    ss = subsystem()
    name = "INF_%s" % tag
    ch = unreal.load_asset("%s/%s.%s" % (WORK, name, name))
    if ch is None:
        raise RuntimeError("%s/%s does not exist -- run the prepare stage first"
                           % (WORK, name))
    out = {"name": name, "tag": tag, "built_path": "%s/%s" % (BUILD_ROOT, name)}
    if not ss.try_add_object_to_edit(ch):
        raise RuntimeError("try_add_object_to_edit returned False for " + name)
    try:
        out["can_build"] = bool(ss.can_build_meta_human(ch, True))
        params = unreal.MetaHumanCharacterEditorBuildParameters()
        params.set_editor_property(
            "pipeline_type", unreal.MetaHumanDefaultPipelineType.OPTIMIZED)
        params.set_editor_property(
            "pipeline_quality", unreal.MetaHumanQualityLevel.HIGH)
        params.set_editor_property("absolute_build_path", BUILD_ROOT)
        params.set_editor_property("common_folder_path", COMMON_ROOT)
        params.set_editor_property("name_override", name)
        params.set_editor_property("enable_wardrobe_item_validation", False)
        ss.build_meta_human(ch, params)
    finally:
        if ss.is_object_added_for_editing(character=ch):
            ss.remove_object_to_edit(character=ch)
    unreal.EditorAssetLibrary.save_directory(BUILD_ROOT, only_if_is_dirty=False,
                                             recursive=True)
    unreal.EditorAssetLibrary.save_directory(COMMON_ROOT, only_if_is_dirty=False,
                                             recursive=True)
    out["built"] = [r for r in built_bodies()
                    if r["path"].startswith(BUILD_ROOT + "/")]
    return out


COMBINE_ROOT = os.environ.get("INF_MH_COMBINED", "/Game/INF/Combined")


def _skeletal_meshes_under(prefix):
    """Every `USkeletalMesh` under a content prefix, path-sorted."""
    reg = registry()
    try:
        reg.scan_paths_synchronous([prefix], force_rescan=True)
    except Exception:
        pass
    rows = []
    ar = unreal.ARFilter(package_paths=[prefix], recursive_paths=True,
                         class_names=["SkeletalMesh"])
    for a in reg.get_assets(ar):
        pkg = str(a.get_editor_property("package_name"))
        nm = str(a.get_editor_property("asset_name"))
        rows.append("%s.%s" % (pkg, nm))
    rows.sort()
    return rows


def _slot_table(mesh):
    """`[(slot name, material path, material)]` for one skeletal mesh."""
    out = []
    try:
        for slot in mesh.get_editor_property("materials"):
            mi = slot.get_editor_property("material_interface")
            nm = str(slot.get_editor_property("material_slot_name"))
            out.append((nm, mi.get_path_name() if mi is not None else None, mi))
    except Exception as e:
        say("slots of %s: %s" % (mesh.get_name(), e))
    return out


def combine_one(tag):
    """**THE COMBINED FACE+BODY MESH** (wave CHAR1a.3, clause 1).

    `UMetaHumanCharacterExportBlueprintLibrary.ExportGeometry` with
    `bFullBodySkeletalMesh` routes to
    `MetaHumanCharacterEditorSubsystem::CreateCombinedFaceAndBodyMesh`, which is
    the merge this engine would otherwise have had to write: ONE skeletal mesh
    on ONE skeleton carrying both the body's joints and the face's, with the
    neck seam welded by the tool that authored both halves.

    Wave CHAR1a.2 recorded `export_geometry` as failing with
    `Assertion failed: CurrentApplication.IsValid()` -- measured **in a
    commandlet**, which has no Slate application. This runs in a LIVE editor
    (see `mh_remote.py`), which has one. The retry is the whole point.

    # The materials, and why they have to be re-bound

    The combined mesh comes out of the tool wearing a CLAY material on its
    slots. The real materials exist -- they are the ones the ASSEMBLED
    character's own `SKM_*_BodyMesh` and `SKM_*_FaceMesh` wear -- so the slots
    are re-bound **by slot name** from those two meshes here, in UE, before the
    glTF crossing. Doing it here rather than at import keeps the bridge and the
    importer exactly what they were for the mannequins: a skeletal mesh with N
    material slots, N of which name real materials with real textures.

    Every slot is reported -- matched by name, already correct, or left clay --
    because a silent clay slot is a grey patch on a face.
    """
    ss = subsystem()
    name = "INF_%s" % tag
    ch = unreal.load_asset("%s/%s.%s" % (WORK, name, name))
    if ch is None:
        raise RuntimeError("%s/%s does not exist -- run prepare+assemble first"
                           % (WORK, name))
    out = {"name": name, "tag": tag, "project_path": COMBINE_ROOT}
    before = set(_skeletal_meshes_under(COMBINE_ROOT))
    if not ss.try_add_object_to_edit(ch):
        raise RuntimeError("try_add_object_to_edit returned False for " + name)
    try:
        params = unreal.MetaHumanGeometryExportParams()
        params.set_editor_property("project_path", COMBINE_ROOT)
        params.set_editor_property("head_skeletal_mesh", False)
        params.set_editor_property("body_skeletal_mesh", False)
        params.set_editor_property("full_body_skeletal_mesh", True)
        params.set_editor_property("overwrite_existing_assets", True)
        unreal.MetaHumanCharacterExportBlueprintLibrary.export_geometry(ch, params)
    finally:
        if ss.is_object_added_for_editing(character=ch):
            ss.remove_object_to_edit(character=ch)
    unreal.EditorAssetLibrary.save_directory(COMBINE_ROOT, only_if_is_dirty=False,
                                             recursive=True)
    after = _skeletal_meshes_under(COMBINE_ROOT)
    out["created"] = [p for p in after if p not in before]
    out["all_under_root"] = after
    # The combined mesh: whichever asset under the root names this character.
    # Named rather than taken at index 0, because the root accumulates one
    # asset per character and a run for Vivian must not re-bind Dominic's.
    mine = [p for p in after if tag.lower() in p.lower()]
    if not mine:
        raise RuntimeError("export_geometry produced no skeletal mesh naming %s "
                           "under %s (found %s)" % (tag, COMBINE_ROOT, after))
    path = mine[0]
    out["combined"] = path
    mesh = unreal.load_asset(path)
    if mesh is None:
        raise RuntimeError("combined mesh %s did not load" % path)
    sk = mesh.get_editor_property("skeleton")
    out["skeleton"] = sk.get_path_name() if sk else None
    out["bones"] = len(sk.get_editor_property("bone_tree")) if sk else 0
    try:
        out["lods"] = int(mesh.get_num_lods())
    except Exception:
        out["lods"] = None
    try:
        b = mesh.get_bounds()
        out["bounds_extent_cm"] = [float(b.box_extent.x), float(b.box_extent.y),
                                   float(b.box_extent.z)]
    except Exception as e:
        out["bounds_error"] = str(e)

    # -- the real materials, by slot name --------------------------------
    donors, donor_order = _donor_table(name)
    out["donor_slots"] = [{"mesh": a, "slot": b, "material": c}
                          for a, b, c in donor_order]

    out["slots_before"] = [{"slot": s, "material": p} for s, p, _ in _slot_table(mesh)]
    out.update(_write_udim_slots(mesh, donors, path))
    return out


# Which DONOR slot supplies each UDIM tile of the combined mesh, in tile order
# (index 0 = tile 1001, index 1 = tile 1002).
#
# Measured, wave CHAR1a.3 AUDIT: `CreateCombinedFaceAndBodyMesh` keeps the two
# halves' own texture atlases and packs them as UDIM -- the FACE half's uv lands
# in tile 1001 and the BODY half's in tile 1002 (34 514 triangles above the neck
# in 1001, 60 816 below it in 1002, zero triangles straddling a tile, measured on
# the imported geometry of both characters). The tool then hands the result ONE
# material slot, because a UE material has one atlas and UE has nothing to say
# two with. The consequence, photographed before this was written: the head
# sampled `T_Body_BC` -- a body atlas carrying hands, feet and underwear and NO
# FACE -- and both MetaHumans stood on the island with blank, featureless heads.
UDIM_DONOR_SLOTS = ["head_LOD1_shader_shader", "body_shader_shader"]


def _write_udim_slots(mesh, donors, path):
    """**Declare the combined mesh's UDIM atlases as MATERIAL SLOTS, in tile
    order** (wave CHAR1a.3 audit).

    One slot per UDIM tile, named `udim_1001`, `udim_1002`, ... and bound to the
    donor material whose atlas that tile IS. The importer measures the tiles off
    the geometry and splits the mesh into one section per tile, so slot *k* is
    the material of tile 1001 + *k* and neither end has to guess.

    # Why the slots are written in TILE order and not in UE's

    Because there is no UE order to preserve. The combined mesh has ONE section
    covering both atlases, so whatever material sits in slot 0 is drawn over the
    whole body in UE -- the head material or the body material, both equally
    wrong, because UE cannot draw a two-atlas mesh either. This asset exists only
    as an export source; the tile order is the only reading under which its slots
    are true, and it is the reading the bridge and the importer now share.

    A donor slot that does not exist leaves its tile UNBOUND and is reported;
    nothing is guessed and nothing falls back to slot 0.
    """
    out = {}
    arr = []
    rows = []
    for tile, donor_slot in enumerate(UDIM_DONOR_SLOTS):
        hit = donors.get(donor_slot)
        mi = hit[1] if hit else None
        sm = unreal.SkeletalMaterial()
        sm.set_editor_property("material_interface", mi)
        sm.set_editor_property("material_slot_name",
                               unreal.Name("udim_%d" % (1001 + tile)))
        arr.append(sm)
        rows.append({"tile": 1001 + tile, "slot": "udim_%d" % (1001 + tile),
                     "donor_slot": donor_slot,
                     "material": mi.get_path_name() if mi else None})
    out["slots_after"] = rows
    out["udim_slots"] = rows
    out["clay_left"] = [r["slot"] for r in rows
                        if r["material"] and "clay" in r["material"].lower()]
    out["udim_unbound"] = [r["donor_slot"] for r in rows if r["material"] is None]
    try:
        mesh.set_editor_property("materials", arr)
        unreal.EditorAssetLibrary.save_loaded_asset(mesh, only_if_is_dirty=False)
    except Exception as e:
        out["rebind_error"] = str(e)
        say("REBIND FAILED on %s: %s" % (path, e))
    return out


def _donor_table(name):
    """`{slot name: (material path, material)}` over the assembled character's
    own body and face meshes -- UE's own slot names, which is the only place the
    head material is identified reliably.

    (The glTF crossing cannot do it. UE's exporter writes a mesh's materials
    DEDUPLICATED and its primitives' `material` indices do not survive a
    `LODMaterialMap`: measured on the FACE mesh, whose 23 860-triangle head
    primitive comes out of the exporter referencing the material named
    `MI_Teeth_Baked`.)
    """
    donors = {}
    order = []
    for src in _skeletal_meshes_under("%s/%s" % (BUILD_ROOT, name)):
        m = unreal.load_asset(src)
        if m is None:
            continue
        for slot_name, mat_path, mi in _slot_table(m):
            if mi is None:
                continue
            donors.setdefault(slot_name, (mat_path, mi))
            order.append((src, slot_name, mat_path))
    return donors, order


def slots_one(tag):
    """**Re-declare an EXISTING combined mesh's UDIM slots** (wave CHAR1a.3
    audit) -- `combine_one` without `export_geometry`, so it runs in a
    commandlet.

    `export_geometry` needs a Slate application and the combined assets already
    exist; re-running it to change a material array would cost a live editor for
    nothing.
    """
    name = "INF_%s" % tag
    out = {"name": name, "tag": tag, "project_path": COMBINE_ROOT}
    under = _skeletal_meshes_under(COMBINE_ROOT)
    mine = [p for p in under if tag.lower() in p.lower()]
    if not mine:
        raise RuntimeError("no combined mesh naming %s under %s (found %s)"
                           % (tag, COMBINE_ROOT, under))
    path = mine[0]
    out["combined"] = path
    mesh = unreal.load_asset(path)
    if mesh is None:
        raise RuntimeError("combined mesh %s did not load" % path)
    donors, order = _donor_table(name)
    out["donor_slots"] = [{"mesh": a, "slot": b, "material": c}
                          for a, b, c in order]
    out["slots_before"] = [{"slot": s, "material": p}
                           for s, p, _ in _slot_table(mesh)]
    out.update(_write_udim_slots(mesh, donors, path))
    return out


def probe_materials(tag):
    """Every texture parameter of every material the assembled and combined
    meshes wear, with its compression and sRGB flag -- the table
    `ue_import::role_to_planes` is extended from (wave CHAR1a.3, clause 5).

    Probed rather than guessed, exactly as CHAR1a.2 probed `M_Mannequin`: a
    MetaHuman material's parameter names are the pipeline's, not the
    mannequin's, and a bridge that classifies them by substring puts a cavity
    map in the normal slot.
    """
    name = "INF_%s" % tag
    rows = []
    seen = set()
    for src in (_skeletal_meshes_under("%s/%s" % (BUILD_ROOT, name))
                + _skeletal_meshes_under(COMBINE_ROOT)):
        m = unreal.load_asset(src)
        if m is None:
            continue
        for slot_name, mat_path, mi in _slot_table(m):
            if mi is None or mat_path in seen:
                continue
            seen.add(mat_path)
            row = {"mesh": src, "slot": slot_name, "material": mat_path,
                   "class": mi.get_class().get_name(), "textures": []}
            try:
                par = mi.get_editor_property("parent")
                row["parent"] = par.get_path_name() if par else None
            except Exception:
                row["parent"] = None
            try:
                for p in mi.get_editor_property("texture_parameter_values"):
                    pn = str(p.get_editor_property("parameter_info")
                             .get_editor_property("name"))
                    t = p.get_editor_property("parameter_value")
                    if t is None:
                        row["textures"].append({"param": pn, "texture": None})
                        continue
                    row["textures"].append({
                        "param": pn,
                        "texture": t.get_path_name(),
                        "compression": str(t.get_editor_property("compression_settings")),
                        "srgb": bool(t.get_editor_property("srgb")),
                        "w": int(t.blueprint_get_size_x()),
                        "h": int(t.blueprint_get_size_y()),
                    })
            except Exception as e:
                row["error"] = str(e)
            rows.append(row)
    rows.sort(key=lambda r: (r["material"], r["slot"]))
    return rows


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
    reg.scan_paths_synchronous([WORK, BUILD_ROOT], force_rescan=True)
    rows = []
    ar = unreal.ARFilter(package_paths=[WORK, BUILD_ROOT],
                         recursive_paths=True,
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

    if STAGE in ("assemble", "all"):
        for tag in (MALE, FEMALE):
            rec = step("assemble_%s" % tag, lambda t=tag: assemble_one(t))
            if rec.get("ok"):
                REPORT["characters"].append(rec["result"])

    if STAGE in ("combine", "all"):
        for tag in (MALE, FEMALE):
            rec = step("combine_%s" % tag, lambda t=tag: combine_one(t))
            if rec.get("ok"):
                REPORT["characters"].append(rec["result"])
        for tag in (MALE, FEMALE):
            step("probe_materials_%s" % tag, lambda t=tag: probe_materials(t))

    if STAGE in ("slots", "all"):
        for tag in (MALE, FEMALE):
            rec = step("slots_%s" % tag, lambda t=tag: slots_one(t))
            if rec.get("ok"):
                REPORT["characters"].append(rec["result"])

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
