"""THE UNREAL -> INFINI EXPORT SIDE OF THE BRIDGE (wave ASSET0, clause 1).

Read-only. This script never writes into the Unreal project: it opens it
headless, walks the packs named in `PACKS` below, and writes glTF, PNG and one
`manifest.json` into an output directory OUTSIDE it -- and outside the engine
checkout, which it REFUSES rather than assumes (see `engine_checkout_above`).
`inf-import --manifest` is the other side; it is its own binary, not a
subcommand of `inf`, and it refuses the same destinations.

    "C:/Program Files/Epic Games/UE_5.8/Engine/Binaries/Win64/UnrealEditor-Cmd.exe" \
        "<project>.uproject" -run=pythonscript -script=tools/ue-export/export.py \
        -unattended -nop4 -nosplash -stdout -FullStdOutLogOutput -NoShaderCompile \
        -EnablePlugins=GLTFExporter

`-EnablePlugins=GLTFExporter` is not optional and is the reason this works at
all: the glTF exporter ships with the engine (Engine/Plugins/Enterprise) and is
DISABLED in a default project, and enabling it on the command line is what let
this bridge be built without editing somebody's `.uproject`. Measured: the
plugin mounts, `unreal.GLTFExportOptions` appears, and the project is untouched.

Configuration is by environment variable, because `-run=pythonscript` has no
argv to give a script:

    INF_UE_OUT     the output directory                (required)
    INF_UE_MODE    "plan" (census only) or "export"    (default "export")
    INF_UE_PACKS   comma-separated pack names to do    (default: all of PACKS)
    INF_UE_MAXTEX  skip textures wider than this       (default 8192)

Every line it prints is prefixed `UEX:` so the interesting half of a 30 MB
Unreal log is one `grep`.

# One glTF per LOD, and why

`UGLTFExportOptions` has `default_level_of_detail` and no "export every LOD":
the exporter writes ONE mesh per file. So a mesh that ships four rungs is
exported four times under four names and the manifest binds them into one
ladder. That is not a workaround for a missing feature -- it is the shape the
importer wants anyway, because `.inf_mesh` stores a rung as a whole mesh.

# The UE LOD census this was written against

93 meshes sampled across these packs: 60 ship ONE LOD (Downtown_West props and
buildings, the modular set, furniture), 21 ship four (Megascans curbs at about
8:1, vehicles), 12 ship six or seven (Fab foliage, down to an eight-triangle
card). So the ladder is real content for a fifth of what is here and has to be
GENERATED for the rest, which is `inf import`'s half.

# LICENSING

`PACKS` records what is known about each pack's licence and says "unknown" when
nothing is known -- there is no licence file anywhere under the project's
`Content`, so every claim here would otherwise be a guess. Nothing this script
writes may enter the engine repository; see `docs/memos/island-progress.md`.
"""

import datetime
import json
import os
import re
import traceback

import unreal


def say(*a):
    print("UEX:", *a)


# ── what to export ───────────────────────────────────────────────────────────
#
# A pack is a name, a licence note, and a list of SELECTORS. A selector is
# either an exact `/Game/...` object path or a `prefix` + `classes` + `limit`
# sweep. Exact paths where the choice matters (which asphalt is THE road),
# sweeps where it does not (every awning).
#
# `surface` marks a selector whose materials are tiling SURFACES rather than a
# mesh's own skin -- the importer binds those to terrain layers, road ribbons
# and grammar walls, and it needs to be told which is which because a Megascans
# material instance looks identical either way.

PACKS = [
    {
        "name": "MS_AsphaltEss",
        "license": "unknown - Quixel/Megascans via Fab. Megascans content "
                   "bundled with Unreal is licensed for use IN Unreal Engine "
                   "projects; conversion for shipping elsewhere is NOT "
                   "established. Verify on the Fab page before shipping.",
        "select": [
            {"prefix": "/Game/MS_AsphaltEss", "classes": ["MaterialInstanceConstant"],
             "match": r"(?i)asphalt", "limit": 3, "surface": True},
        ],
    },
    {
        "name": "MS_CityCurbs",
        "license": "unknown - Quixel/Megascans via Fab. See MS_AsphaltEss.",
        "select": [
            {"prefix": "/Game/MS_CityCurbs", "classes": ["StaticMesh"],
             "match": r"(?i)curb.*_LOD0$", "limit": 2},
        ],
    },
    {
        "name": "MS_BrickV1",
        "license": "unknown - Quixel/Megascans via Fab. See MS_AsphaltEss.",
        "select": [
            {"prefix": "/Game/MS_BrickV1", "classes": ["MaterialInstanceConstant"],
             "match": r"(?i)brick", "limit": 2, "surface": True},
        ],
    },
    {
        "name": "MS_ConcreteV1",
        "license": "unknown - Quixel/Megascans via Fab. See MS_AsphaltEss.",
        "select": [
            {"prefix": "/Game/MS_ConcreteV1", "classes": ["MaterialInstanceConstant"],
             "match": r"(?i)concrete", "limit": 2, "surface": True},
        ],
    },
    {
        "name": "MS_CementV1",
        "license": "unknown - Quixel/Megascans via Fab. See MS_AsphaltEss.",
        "select": [
            {"prefix": "/Game/MS_CementV1", "classes": ["MaterialInstanceConstant"],
             "match": r"(?i)(cement|concrete)", "limit": 2, "surface": True},
        ],
    },
    {
        # THE GROUND (wave ASSET0, clause 5) -- the 51 km2 the player actually
        # stands on. Two of the island's four `TerrainLayer`s can be replaced
        # from these packs and two cannot: there is **no photographed grass and
        # no sand anywhere in this project**. `MS_PristineGr` sounds like the
        # first and is Pristine GRANITE (measured: Baltic Brown, Juparana Brown,
        # French Cream), so it is not in this list. Grass and sand stay
        # synthesised, and that is stated in the wave ledger rather than papered
        # over with a moss.
        "name": "MS_MountainSl",
        "license": "unknown - Quixel/Megascans via Fab. See MS_AsphaltEss.",
        "select": [
            {"prefix": "/Game/MS_MountainSl", "classes": ["MaterialInstanceConstant"],
             "match": r"(?i)(rock|cliff|slope|scree)", "limit": 2, "surface": True},
        ],
    },
    {
        "name": "MS_MossEss",
        "license": "unknown - Quixel/Megascans via Fab. See MS_AsphaltEss.",
        "select": [
            {"prefix": "/Game/MS_MossEss", "classes": ["MaterialInstanceConstant"],
             "match": r"(?i)(moss|forest|floor|litter|ground)", "limit": 2, "surface": True},
        ],
    },
    {
        "name": "Downtown_West",
        "license": "unknown - Unreal Marketplace / Fab pack in this project. "
                   "Verify on its Fab page before shipping.",
        "select": [
            {"prefix": "/Game/Downtown_West/Assets/props/props_light_post",
             "classes": ["StaticMesh"], "limit": 4},
            # The Blueprints, for their LIGHTS -- see `add_blueprint`. The
            # meshes they reference are pulled in as a side effect, which is
            # why this selector sits after the mesh one and not instead of it.
            {"prefix": "/Game/Downtown_West/Assets/props/props_light_post",
             "classes": ["Blueprint"], "limit": 2},
            {"prefix": "/Game/Downtown_West/Assets/awnings",
             "classes": ["StaticMesh"], "limit": 3},
            {"prefix": "/Game/Downtown_West/Assets/building_a",
             "classes": ["StaticMesh"], "match": r"(?i)(window|door|pillar)",
             "limit": 4},
        ],
    },
    {
        "name": "AdvancedRealisticGlass",
        "license": "unknown - Unreal Marketplace / Fab pack in this project. "
                   "Verify on its Fab page before shipping.",
        # PARAMETERS ONLY. The glass master is a node graph this engine has no
        # importer for; what crosses the bridge is the numbers an author set on
        # its instances, which is what a PBR opacity/roughness/IOR block needs.
        "select": [
            {"prefix": "/Game/AdvancedRealisticGlass/Materials",
             "classes": ["MaterialInstanceConstant"], "match": r"(?i)clean",
             "limit": 2, "no_textures": True},
        ],
    },
]

# ── the CHARACTER packs (wave CHAR1a) ────────────────────────────────────────
#
# Skeletal meshes and animation clips, kept in their own list because they carry
# a different licence position from every surface pack above and because the
# selectors are exact rather than swept: which mannequin is THE mannequin is a
# decision, not a sample.
#
# **The licence rows here are the ones the manifest carries per asset.** They are
# not hedged ("unknown") like the surface packs, because for these three the
# position IS established and it is different for each:
#
#   * the UE5 mannequins are Epic's own engine content ("UE-Only Content" in the
#     EULA) -- usable inside an Unreal Engine project, NOT licensed for shipping
#     in another engine. So they cross the bridge as a **dev reference and
#     stand-in**, live under `island-build/project/Content/UE/Mannequins/`, and
#     are never cooked into a shipped pack and never committed.
#   * ALS-Community-UE5 is MIT (the plugin's own LICENSE file, "Copyright (c)
#     2020 Doga Can Yanikoglu & LongmireLocomotion"). MIT permits use in any
#     engine with the notice preserved -- so these clips MAY ship, and the
#     notice travels in the manifest row and in the import sidecar.
#   * MetaHumans are exported by `metahuman.py`, not this file, and carry their
#     own row.
#
# `skeletal` records the SKIN; `clips` records ANIMATION. They are separate
# because UE separates them: a `USkeletalMesh` and a `UAnimSequence` are two
# assets sharing a `USkeleton`, and the glTF exporter writes each on its own.

MANNEQUIN_DIR = "/Game/ControlRig/Characters/Mannequins"

CHARACTERS = [
    {
        "name": "UE5_Mannequins",
        "license": "Unreal Engine EULA -- Epic 'UE-Only Content'. Licensed for "
                   "use in Unreal Engine projects; NOT licensed to ship in "
                   "another engine. Imported as a DEV REFERENCE and stand-in "
                   "only: local to island-build, never cooked into a shipped "
                   "pack, never committed to the engine repository.",
        "ship": False,
        "skeletal": [
            MANNEQUIN_DIR + "/Meshes/SKM_Manny.SKM_Manny",
            MANNEQUIN_DIR + "/Meshes/SKM_Quinn.SKM_Quinn",
        ],
        "clips": [
            {"prefix": MANNEQUIN_DIR + "/Animations", "limit": 32},
        ],
    },
    {
        # **THE ASSEMBLED METAHUMANS** (wave CHAR1a.2). Not a fixed asset list
        # like the mannequins': the assembly writes its meshes under
        # `<work>/Built/<Name>/{Body,Face}/SKM_*`, so this pack names a PREFIX
        # and takes whatever the pipeline produced. It is empty in the user's own
        # project and non-empty in the scratch `MHForge` one, which is exactly
        # where `metahuman.py` builds them — the same script, run in whichever
        # project has the assets.
        #
        # LICENCE: MetaHuman Content licence, mid-2025 terms. `ship` is TRUE and
        # that is the difference from the mannequins: Epic's 2025-06 change
        # licenses MetaHuman content for use in ANY engine below US$1M annual
        # revenue, so these may be cooked into a shipped `.ipack`. They still may
        # never be COMMITTED — use is not redistribution of the source assets.
        "name": "MetaHumans",
        "license": "MetaHuman Content licence, mid-2025 terms -- usable in ANY "
                   "engine (Epic's 2025-06 licence change; free below US$1M "
                   "annual revenue). Shipped inside a cooked .ipack; NEVER "
                   "committed to the engine repository, because use is not "
                   "redistribution. Terms relied on: "
                   "unrealengine.com/en-US/eula/metahuman, retrieved 2026-09-05.",
        "ship": True,
        "skeletal": [],
        "skeletal_prefix": [
            {"prefix": "/Game/INF/Built", "match": r"^SKM_.*_(Body|Face)Mesh$",
             "limit": 8},
        ],
        "clips": [],
    },
    {
        "name": "ALS_Community",
        "license": "MIT License, Copyright (c) 2020 Doga Can Yanikoglu & "
                   "LongmireLocomotion (ALS-Community-UE5/LICENSE). Permits use "
                   "in any engine provided the copyright notice and this "
                   "permission notice are preserved; the notice travels in this "
                   "manifest row and in each imported asset's sidecar.",
        "ship": True,
        "skeletal": [],
        "clips": [
            {"prefix": "/ALSV4_CPP/AdvancedLocomotionV4/CharacterAssets/"
                       "MannequinSkeleton/AnimationExamples", "limit": 200},
        ],
    },
]

OUT = os.environ.get("INF_UE_OUT", "")
MODE = os.environ.get("INF_UE_MODE", "export")
ONLY = [p for p in os.environ.get("INF_UE_PACKS", "").split(",") if p]
MAXTEX = int(os.environ.get("INF_UE_MAXTEX", "8192"))

# ── the map-kind classifier ──────────────────────────────────────────────────
#
# A texture's ROLE, not its format. Two sources of truth, in this order:
#
#   1. the material parameter it is bound to. Every Megascans instance in this
#      project parents `Standard_MasterMaterial` and names its slots `albedo`,
#      `normal`, `roughness`, `displacement` -- measured, not assumed.
#   2. the texture's own `compression_settings` and `srgb`, which UE sets at
#      import and which cannot lie about a normal map (`TC_Normalmap`).
#
# The asset NAME is the last resort, because a name is a convention and the
# packs here use four of them.

PARAM_KIND = {
    "albedo": "albedo", "basecolor": "albedo", "base color": "albedo",
    "diffuse": "albedo", "color": "albedo", "colour": "albedo",
    "normal": "normal", "normalmap": "normal",
    "roughness": "roughness", "rough": "roughness",
    "metallic": "metallic", "metalness": "metallic",
    "specular": "specular",
    "ao": "ao", "occlusion": "ao", "ambientocclusion": "ao",
    "orm": "orm", "arm": "orm", "packed": "orm",
    "opacity": "opacity", "alpha": "opacity", "mask": "opacity",
    "emissive": "emissive", "emission": "emissive",
    "displacement": "displacement", "height": "displacement",
    # Wave CHAR1a: the UE5 mannequin material names its albedo slot
    # "Base Texture", which none of the keys above reach.
    "base texture": "albedo", "basetex": "albedo", "base": "albedo",

    # ── M_Mannequin, READ (wave CHAR1a.2) ───────────────────────────────────
    #
    # The mannequin binds EIGHT textures and this bridge was placing four of
    # them wrongly, because the substring fallback below is a net with holes in
    # it. Probed in UE on 2026-09-05 (`MaterialEditingLibrary
    # .get_texture_parameter_names` + the bound texture's compression/sRGB), and
    # the channel meanings confirmed by a census of the exported PNGs:
    #
    # | parameter | texture | what it is | what it USED to classify as |
    # |---|---|---|---|
    # | `Normal` | `T_*_N` | the tangent-space normal (TC_NORMALMAP) | normal (right) |
    # | `BNormal` | `T_*_BN` | a SECOND normal | **normal** — three maps raced for one slot |
    # | `Tangent` | `T_*_Tan` | the anisotropy tangent field, not a normal at all | **normal** — same race |
    # | `Base Texture` | `T_*_D` | albedo, sRGB | albedo (fixed at CHAR1a) |
    # | `LogoTexture` | `T_UE_Logo_M` | a grayscale logo decal | **metallic**, from the `_m$` name rule |
    # | `MSR_tex` | `T_*_MSR_MSK` | metallic / specular / roughness, packed | unknown (CHAR1a carried 76) |
    # | `AnisoAOPaintMaskTex` | `T_*_AS?AO?MASK_MSK` | anisotropy / AO / paint mask | `ao` — but from the WRONG channel |
    # | `CCCCRTex` | `T_*_CCRCCPlastic_MSK` | clearcoat + clearcoat roughness | unknown |
    #
    # The channel census (16-bit-free 8-bit PNGs, every 17th texel of
    # `T_Manny_01_*`, 4096²) is what settles the packing rather than the name:
    #
    # * `MSR_MSK` R is BIMODAL (p25 0, median 247, max 255) — a metal mask;
    #   G sits at 92-118 — UE's 0.5 specular constant; B is 0/116/255 — the
    #   roughness. R = metallic, G = specular, B = roughness, exactly as the
    #   name spells it.
    # * `ASAOPMASK_MSK` R has **mean 3.5 of 255** (anisotropy, ~0 everywhere),
    #   G has mean 244 with creases darker (the AO), B is bimodal 0/255 (the
    #   paint mask). The importer read plane R, so the hero's occlusion was
    #   **0.014** and its ambient term was multiplied away — the single biggest
    #   reason the body read dark and streaky in CHAR1a's frames.
    # * `T_Manny_01_N` is nearly FLAT (R,G in 122..130 of 255) and
    #   `T_Manny_01_BN` carries the real relief (R,G over the full 0..255). The
    #   engine has ONE normal slot and it goes to the parameter the material
    #   calls `Normal`; `BNormal` is reported as unplaced rather than silently
    #   winning a race, which is what it was doing.
    "normal": "normal",
    "bnormal": "normal_second",
    "tangent": "tangent",
    "logotexture": "decal",
    "msrtex": "msr",
    "anisoaopaintmasktex": "aniso_ao_paint",
    "ccccrtex": "clearcoat",
}

NAME_KIND = [
    (r"(?i)(_n$|_normal|normallod|_nrm)", "normal"),
    (r"(?i)(_orm$|_arm$|_packed)", "orm"),
    (r"(?i)(_r$|_rough)", "roughness"),
    (r"(?i)(_m$|_metal)", "metallic"),
    (r"(?i)(_ao$|occlusion)", "ao"),
    (r"(?i)(_o$|_opacity|_alpha|_mask)", "opacity"),
    (r"(?i)(_e$|_emissive|_emission)", "emissive"),
    # **`_d$` is DIFFUSE, not displacement** (wave CHAR1a). It used to be the
    # displacement rule, and the cost was measured on the mannequin:
    # `T_Manny_01_D` is bound to the parameter literally named "Base Texture"
    # and is sRGB -- two independent statements that it is an albedo -- and it
    # was crossing the bridge as a displacement map, which this engine has no
    # slot for, so it was dropped. The hero drew with a normal map and no base
    # colour: a grey, noisy body in every frame. Displacement keeps the
    # unambiguous spellings.
    (r"(?i)(_disp|_height|_dsp$)", "displacement"),
    (r"(?i)(_bc$|_d$|_albedo|_basecolor|_diffuse|_col)", "albedo"),
]


def kind_of_texture(tex, param_name):
    """The ROLE a texture plays, from its parameter first and its name last.

    **The exact match must be tried against every key before any substring is**
    (wave CHAR1a.2). `BNormal` contains `normal`, so the loop below classified
    the mannequin's second normal as its first one and two maps raced for a slot
    that holds one. Exactness is not a shortcut here, it is the rule.
    """
    p = (param_name or "").strip().lower().replace("_", "").replace(" ", "")
    if p in PARAM_KIND:
        return PARAM_KIND[p]
    for key, kind in PARAM_KIND.items():
        if key.replace(" ", "") in p:
            return kind
    try:
        cs = str(tex.get_editor_property("compression_settings"))
        if "NORMALMAP" in cs.upper():
            return "normal"
    except Exception:
        pass
    name = tex.get_name()
    for pat, kind in NAME_KIND:
        if re.search(pat, name):
            return kind
    try:
        if bool(tex.get_editor_property("srgb")):
            return "albedo"
    except Exception:
        pass
    return "unknown"


# ── collectors ───────────────────────────────────────────────────────────────

REG = unreal.AssetRegistryHelpers.get_asset_registry()
TEXTURES = {}    # object path -> record
MATERIALS = {}   # object path -> record
MESHES = {}      # object path -> record
ERRORS = []


def key_of(path):
    """A manifest key: the object path with the separators a filename allows."""
    return path.replace("/Game/", "").replace("/", "_").replace(".", "_")


def rel(*parts):
    return "/".join(parts)


def ensure(d):
    if d and not os.path.isdir(d):
        os.makedirs(d)


def export_task(obj, filename, exporter=None, options=None):
    task = unreal.AssetExportTask()
    task.set_editor_property("object", obj)
    task.set_editor_property("filename", filename)
    task.set_editor_property("automated", True)
    task.set_editor_property("prompt", False)
    task.set_editor_property("replace_identical", True)
    if exporter is not None:
        task.set_editor_property("exporter", exporter)
    if options is not None:
        task.set_editor_property("options", options)
    return bool(unreal.Exporter.run_asset_export_task(task))


def add_texture(tex, param_name, pack):
    """Record a texture and (in export mode) write its PNG."""
    path = tex.get_path_name()
    if path in TEXTURES:
        # A texture reached through two materials keeps the FIRST role it was
        # given: the parameter that named it is better evidence than the second
        # parameter that named it, and a slot collision would otherwise flip
        # the kind depending on iteration order -- which is exactly the kind of
        # nondeterminism a manifest must not carry.
        return TEXTURES[path]["key"]
    w = tex.blueprint_get_size_x()
    h = tex.blueprint_get_size_y()
    kind = kind_of_texture(tex, param_name)
    try:
        srgb = bool(tex.get_editor_property("srgb"))
    except Exception:
        srgb = kind in ("albedo", "emissive")
    k = key_of(path)
    rec = {
        "key": k, "source": path, "pack": pack, "map": kind,
        "width": w, "height": h, "srgb": srgb, "param": param_name,
        "file": None,
    }
    TEXTURES[path] = rec
    if max(w, h) > MAXTEX:
        rec["skipped"] = "%dx%d exceeds INF_UE_MAXTEX %d" % (w, h, MAXTEX)
        say("  tex SKIP %s (%dx%d > %d)" % (tex.get_name(), w, h, MAXTEX))
        return k
    if MODE == "export":
        name = rel("textures", k + ".png")
        dst = os.path.join(OUT, name.replace("/", os.sep))
        ensure(os.path.dirname(dst))
        # An 8K PNG is about 90 seconds of encode and this project has forty of
        # them, so a re-run that only wants the meshes must not pay for the
        # textures again. Keyed on the file being there and non-empty, which is
        # the same "the product still exists" test the engine's own import cache
        # uses -- and the output directory is ours, so nothing else writes here.
        if os.path.isfile(dst) and os.path.getsize(dst) > 0:
            rec["file"] = name
            rec["bytes"] = os.path.getsize(dst)
            rec["reused"] = True
            say("  tex %-9s %5dx%-5d srgb=%-5s %s (already exported)" %
                (kind, w, h, srgb, tex.get_name()))
            return k
        ok = export_task(tex, dst, unreal.TextureExporterPNG())
        if ok and os.path.isfile(dst):
            rec["file"] = name
            rec["bytes"] = os.path.getsize(dst)
        else:
            rec["skipped"] = "the PNG exporter refused it"
            ERRORS.append("texture %s did not export" % path)
    say("  tex %-9s %5dx%-5d srgb=%-5s %s" % (kind, w, h, srgb, tex.get_name()))
    return k


def add_material(mat, pack, surface=False, no_textures=False):
    path = mat.get_path_name()
    if path in MATERIALS:
        return MATERIALS[path]["key"]
    rec = {
        "key": key_of(path), "source": path, "pack": pack, "surface": surface,
        "parent": None, "maps": {}, "scalars": {}, "vectors": {},
        "base_color": [1.0, 1.0, 1.0, 1.0], "metallic": 0.0, "roughness": 1.0,
        "emissive": [0.0, 0.0, 0.0], "opacity": 1.0, "blend": "opaque",
        "two_sided": False,
    }
    MATERIALS[path] = rec

    try:
        parent = mat.get_editor_property("parent")
        if parent:
            rec["parent"] = parent.get_path_name()
    except Exception:
        pass

    if isinstance(mat, unreal.MaterialInstanceConstant):
        try:
            for p in mat.get_editor_property("scalar_parameter_values"):
                n = str(p.get_editor_property("parameter_info").get_editor_property("name"))
                rec["scalars"][n] = float(p.get_editor_property("parameter_value"))
        except Exception as e:
            ERRORS.append("scalars on %s: %s" % (path, e))
        try:
            for p in mat.get_editor_property("vector_parameter_values"):
                n = str(p.get_editor_property("parameter_info").get_editor_property("name"))
                v = p.get_editor_property("parameter_value")
                rec["vectors"][n] = [float(v.r), float(v.g), float(v.b), float(v.a)]
        except Exception as e:
            ERRORS.append("vectors on %s: %s" % (path, e))
        if not no_textures:
            try:
                for p in mat.get_editor_property("texture_parameter_values"):
                    n = str(p.get_editor_property("parameter_info").get_editor_property("name"))
                    t = p.get_editor_property("parameter_value")
                    if t is None:
                        continue
                    k = add_texture(t, n, pack)
                    kind = TEXTURES[t.get_path_name()]["map"]
                    # First writer wins for the same reason `add_texture` keeps
                    # the first role: two slots claiming "albedo" is an authoring
                    # accident and picking by iteration order is not a decision.
                    rec["maps"].setdefault(kind, k)
            except Exception as e:
                ERRORS.append("textures on %s: %s" % (path, e))

    # The scalar block a PBR importer actually needs, pulled out of the
    # parameter soup by NAME. Everything is kept in `scalars`/`vectors` as well,
    # so a parameter this table does not know about is still on the far side.
    def sca(*names):
        for n in names:
            for have, v in rec["scalars"].items():
                if have.strip().lower().replace(" ", "") == n:
                    return v
        return None

    def vec(*names):
        for n in names:
            for have, v in rec["vectors"].items():
                if have.strip().lower().replace(" ", "") == n:
                    return v
        return None

    v = vec("basecolor", "color", "colour", "albedo", "tint", "basecolour")
    if v:
        rec["base_color"] = v
    for name, field in (("metallic", "metallic"), ("roughness", "roughness"),
                        ("opacity", "opacity")):
        s = sca(name)
        if s is not None:
            rec[field] = s
    v = vec("emissive", "emissivecolor", "emission")
    if v:
        rec["emissive"] = v[:3]
    if rec["opacity"] < 1.0 or "opacity" in rec["maps"]:
        rec["blend"] = "blend"
    try:
        bm = str(mat.get_base_material().get_editor_property("blend_mode"))
        if "MASKED" in bm.upper():
            rec["blend"] = "masked"
        elif "TRANSLUCENT" in bm.upper():
            rec["blend"] = "blend"
    except Exception:
        pass
    say(" mat %-6s %-40s maps=%s" % ("surf" if surface else "skin",
                                     mat.get_name(),
                                     ",".join(sorted(rec["maps"]))))
    return rec["key"]


def socket_list(sm):
    """A mesh's sockets, in UE centimetres.

    `StaticMesh.sockets` is a PROTECTED editor property and refuses
    `get_editor_property` (measured). A transient component answers instead,
    which is the same list read through the blueprint API the engine does
    expose. The manifest carries the raw UE numbers and the importer converts,
    so there is one conversion in the bridge rather than one on each side.
    """
    out = []
    try:
        comp = unreal.StaticMeshComponent()
        comp.set_static_mesh(sm)
        for n in comp.get_all_socket_names():
            t = comp.get_socket_transform(n, unreal.RelativeTransformSpace.RTS_COMPONENT)
            loc = t.translation
            rot = t.rotation.rotator()
            out.append({
                "name": str(n),
                "location_cm": [float(loc.x), float(loc.y), float(loc.z)],
                "rotation_deg": [float(rot.roll), float(rot.pitch), float(rot.yaw)],
            })
    except Exception as e:
        ERRORS.append("sockets on %s: %s" % (sm.get_path_name(), e))
    return out


def add_mesh(sm, pack):
    path = sm.get_path_name()
    if path in MESHES:
        return MESHES[path]["key"]
    k = key_of(path)
    nlods = sm.get_num_lods()
    try:
        nanite = bool(sm.get_editor_property("nanite_settings").enabled)
    except Exception:
        nanite = False
    bb = sm.get_bounding_box()
    rec = {
        "key": k, "source": path, "pack": pack, "nanite": nanite,
        "lods": [], "material_slots": [], "sockets": socket_list(sm),
        "bounds_cm": {
            "min": [float(bb.min.x), float(bb.min.y), float(bb.min.z)],
            "max": [float(bb.max.x), float(bb.max.y), float(bb.max.z)],
        },
        "collision_primitives": 0,
    }
    MESHES[path] = rec
    try:
        bodysetup = sm.get_editor_property("body_setup")
        if bodysetup:
            ag = bodysetup.get_editor_property("agg_geom")
            rec["collision_primitives"] = (
                len(ag.get_editor_property("box_elems"))
                + len(ag.get_editor_property("sphere_elems"))
                + len(ag.get_editor_property("convex_elems"))
                + len(ag.get_editor_property("sphyl_elems")))
    except Exception:
        pass

    # The material SLOTS, in slot order -- the order `.inf_mesh`'s
    # `material_slots` is indexed by, so it has to be the order UE reports and
    # not the order a set iterates.
    try:
        for slot in sm.get_editor_property("static_materials"):
            mi = slot.get_editor_property("material_interface")
            if mi is None:
                rec["material_slots"].append(None)
                continue
            rec["material_slots"].append(add_material(mi, pack))
    except Exception as e:
        ERRORS.append("slots on %s: %s" % (path, e))

    # The AUTHORED screen sizes -- a pack's own opinion about when a rung takes
    # over, and the only thing about a LOD that is a DECISION rather than a
    # consequence. The triangle counts are deliberately not taken here: the
    # importer counts them off the glTF it actually reads, which is the number
    # that describes the rung this bridge produced.
    #
    # `StaticMesh.source_models` is not an exposed property and
    # `get_editor_subsystem(StaticMeshEditorSubsystem)` returns **None** in a
    # commandlet -- both measured. `EditorStaticMeshLibrary` is deprecated and
    # still answers, so it is tried, and a `-1` here means "the importer
    # chooses", which it is able to do.
    #
    # **MEASURED AT THE ASSET0 AUDIT: it is -1 for every rung this project
    # ships** -- 18 of 18 across nine packs, and 8 of 8 on a fresh re-export of
    # MS_CityCurbs. So the sidecar census the LOD ruling defers to carries rung
    # COUNTS and no thresholds, and the importer says so rather than leaving a
    # column of sentinels to be discovered by whoever tries to use it.
    screen = {}
    try:
        sizes = unreal.EditorStaticMeshLibrary.get_lod_screen_sizes(sm)
        for i, v in enumerate(sizes):
            screen[i] = float(v)
    except Exception as e:
        ERRORS.append("screen sizes on %s: %s" % (path, e))

    for lod in range(nlods):
        entry = {"level": lod, "file": None}
        entry["screen_size"] = screen.get(lod, -1.0)
        if MODE == "export":
            name = rel("meshes", "%s_LOD%d.gltf" % (k, lod))
            dst = os.path.join(OUT, name.replace("/", os.sep))
            ensure(os.path.dirname(dst))
            opts = unreal.GLTFExportOptions()
            opts.set_editor_property("default_level_of_detail", lod)
            # Geometry only. Baking UE's material graphs into glTF textures
            # would produce a second, worse copy of maps this manifest already
            # carries at source resolution, and `bake_material_inputs` is the
            # switch that does it.
            #
            # It is an **enum** in 5.8, not a bool, and passing `False` raises
            # "Failed to convert type 'bool' to property 'BakeMaterialInputs'"
            # -- which the exporter reports as a failed TASK, so the run
            # exported 79 textures and zero meshes before this was measured.
            try:
                opts.set_editor_property("bake_material_inputs",
                                         unreal.GLTFMaterialBakeMode.DISABLED)
            except Exception as e:
                ERRORS.append("bake_material_inputs: %s" % e)
            opts.set_editor_property("export_vertex_colors", True)
            opts.set_editor_property("export_lights", False)
            opts.set_editor_property("export_cameras", False)
            try:
                # `unreal.GLTFExporter` is ABSTRACT and instantiating it
                # fails the task (measured). The concrete per-asset
                # exporter is not exposed to Python, so the exporter is
                # left unset and UE picks one by the FILENAME EXTENSION --
                # which is why `dst` ends in `.gltf` and must keep doing so.
                ok = export_task(sm, dst, None, opts)
            except Exception as e:
                ok = False
                ERRORS.append("gltf %s LOD%d: %s" % (path, lod, e))
            if ok and os.path.isfile(dst):
                entry["file"] = name
                entry["bytes"] = os.path.getsize(dst)
            else:
                ERRORS.append("gltf %s LOD%d did not write" % (path, lod))
        rec["lods"].append(entry)
    say("mesh %-46s lods=%d nanite=%-5s slots=%d sockets=%d ss=%s" %
        (sm.get_name(), nlods, nanite, len(rec["material_slots"]),
         len(rec["sockets"]),
         ",".join("%.3f" % e["screen_size"] for e in rec["lods"])))
    return k


FIXTURES = {}   # blueprint object path -> record


def add_blueprint(bp, pack):
    """A prop Blueprint's LIGHTS and the mesh they hang off -- a FIXTURE.

    This is the socket clause, and it is not on the sockets. `SM_lightpost_*`
    reports zero of them (measured); what carries the lamp in Downtown_West is
    `BP_lightpost_a`, a Blueprint whose construction script parents a
    `PointLightComponent` to a `StaticMeshComponent` at a relative offset. So
    the thing worth crossing the bridge is the Blueprint's component tree, and
    a "socket" in this manifest is *where a light sits relative to a mesh*.

    `SimpleConstructionScript` is a protected property, so the subobject
    subsystem answers instead. It is an EDITOR subsystem and this runs in a
    commandlet with the editor loaded, which is the arrangement
    `-run=pythonscript` gives.
    """
    path = bp.get_path_name()
    if path in FIXTURES:
        return FIXTURES[path]["key"]
    rec = {"key": key_of(path), "source": path, "pack": pack,
           "lights": [], "meshes": []}
    FIXTURES[path] = rec
    try:
        sub = unreal.get_engine_subsystem(unreal.SubobjectDataSubsystem)
        handles = sub.k2_gather_subobject_data_for_blueprint(bp)
        # HANDLE -> DATA -> OBJECT. `get_object` takes the DATA and refuses a
        # handle ("Failed to convert parameter 'data'", measured), and the
        # gather returns each component TWICE, so the names are deduplicated --
        # a fixture with two of the same lamp would otherwise be a fact about
        # the walk rather than about the prop.
        seen = set()
        for h in handles:
            d = unreal.SubobjectDataBlueprintFunctionLibrary.get_data(h)
            obj = unreal.SubobjectDataBlueprintFunctionLibrary.get_object(d)
            if obj is None or obj.get_name() in seen:
                continue
            seen.add(obj.get_name())
            if isinstance(obj, unreal.LightComponent):
                t = obj.get_relative_transform()
                loc, rot = t.translation, t.rotation.rotator()
                light = {
                    "name": obj.get_name(),
                    "kind": ("spot" if isinstance(obj, unreal.SpotLightComponent)
                             else "point"),
                    "location_cm": [float(loc.x), float(loc.y), float(loc.z)],
                    "rotation_deg": [float(rot.roll), float(rot.pitch),
                                     float(rot.yaw)],
                }
                for prop, out in (("intensity", "intensity"),
                                  ("attenuation_radius", "radius_cm"),
                                  ("outer_cone_angle", "outer_cone_deg"),
                                  ("inner_cone_angle", "inner_cone_deg")):
                    try:
                        light[out] = float(obj.get_editor_property(prop))
                    except Exception:
                        pass
                try:
                    c = obj.get_editor_property("light_color")
                    light["color_srgb8"] = [int(c.r), int(c.g), int(c.b)]
                except Exception:
                    pass
                rec["lights"].append(light)
            elif isinstance(obj, unreal.StaticMeshComponent):
                sm = obj.get_editor_property("static_mesh")
                if sm is None:
                    continue
                t = obj.get_relative_transform()
                loc = t.translation
                rec["meshes"].append({
                    "mesh": add_mesh(sm, pack),
                    "component": obj.get_name(),
                    "location_cm": [float(loc.x), float(loc.y), float(loc.z)],
                })
    except Exception as e:
        ERRORS.append("blueprint %s: %s" % (path, e))
        say("  ! blueprint %s: %s" % (bp.get_name(), e))
    say("  bp %-40s lights=%d meshes=%d" %
        (bp.get_name(), len(rec["lights"]), len(rec["meshes"])))
    return rec["key"]


# ── skeletal meshes and clips (wave CHAR1a) ──────────────────────────────────

SKELETAL = {}   # object path -> record
CLIPS = {}      # object path -> record


def gltf_facts(dst):
    """What the written glTF actually contains -- read back, not asserted.

    The joint NAMES are the interchange contract this whole wave rests on (the
    161 mannequin bone names bind a body to our rig), and the only place they
    can be read in a commandlet is the file the exporter just wrote:
    `USkeleton` exposes `bone_tree` to Python as an opaque list whose length is
    the bone count and whose elements have no name (measured). So the manifest
    carries what the artifact carries.
    """
    out = {"bytes": os.path.getsize(dst) if os.path.isfile(dst) else 0}
    try:
        with open(dst, "r", encoding="utf-8") as f:
            j = json.load(f)
    except Exception as e:
        out["read_error"] = str(e)
        return out
    nodes = j.get("nodes", [])
    skins = j.get("skins", [])
    out["nodes"] = len(nodes)
    out["skins"] = len(skins)
    out["animations"] = len(j.get("animations", []))
    if skins:
        joints = skins[0].get("joints", [])
        out["joints"] = len(joints)
        out["inverse_bind"] = "inverseBindMatrices" in skins[0]
        out["joint_names"] = [nodes[i].get("name", "") if i < len(nodes) else ""
                              for i in joints]
    # TRIANGLES, counted off the accessors -- the number a LOD ladder is
    # actually about. UE's own `screen_size` is its auto-compute sentinel (-1)
    # for every rung in this project, so the rung a ladder KEEPS is chosen by
    # this number and not by the pack's opinion. Measured on SKM_Manny: rungs 0
    # and 1 are both 92 178 triangles -- the mannequin's LOD1 is a copy of its
    # LOD0 -- so "the first three rungs" would have stored one of them twice.
    acc = j.get("accessors", [])
    tris = 0
    verts = 0
    for mesh in j.get("meshes", []):
        for prim in mesh.get("primitives", []):
            idx = prim.get("indices")
            if idx is not None and idx < len(acc):
                tris += acc[idx].get("count", 0) // 3
            pos = prim.get("attributes", {}).get("POSITION")
            if pos is not None and pos < len(acc):
                verts += acc[pos].get("count", 0)
    out["triangles"] = tris
    out["vertices"] = verts
    prims = (j.get("meshes") or [{}])[0].get("primitives", [])
    out["primitives"] = len(prims)
    if prims:
        attrs = sorted(prims[0].get("attributes", {}).keys())
        out["attributes"] = attrs
        # UE writes EIGHT influences per vertex (JOINTS_0+JOINTS_1). Recorded
        # because the importer keeps four and the loss has to be measurable.
        out["influence_sets"] = len([a for a in attrs if a.startswith("JOINTS_")])
    return out


def skeletal_export_options(lod):
    opts = unreal.GLTFExportOptions()
    opts.set_editor_property("default_level_of_detail", lod)
    try:
        opts.set_editor_property("bake_material_inputs",
                                 unreal.GLTFMaterialBakeMode.DISABLED)
    except Exception as e:
        ERRORS.append("bake_material_inputs: %s" % e)
    # THE THREE THAT MAKE A SKIN A SKIN. `export_vertex_skin_weights` writes
    # JOINTS_n/WEIGHTS_n and the `skins` array; without it UE writes the bind
    # pose as a static mesh and the file looks fine and animates never.
    for prop, val in (("export_vertex_skin_weights", True),
                      ("export_animation_sequences", True),
                      ("export_morph_targets", False),
                      ("make_skinned_meshes_root", False),
                      ("export_vertex_colors", True),
                      ("export_lights", False),
                      ("export_cameras", False)):
        try:
            opts.set_editor_property(prop, val)
        except Exception as e:
            ERRORS.append("%s: %s" % (prop, e))
    return opts


def skeletal_lod_count(sm):
    """UE 5.8 exposes no `lod_info` on `USkeletalMesh` to Python (measured:
    "Failed to find property 'lod_info'"), so this asks the three doors that
    do answer and records which one did."""
    for fn in ("get_num_lods",):
        try:
            return int(getattr(sm, fn)()), fn
        except Exception:
            pass
    try:
        return int(unreal.EditorSkeletalMeshLibrary.get_lod_count(sm)), "EditorSkeletalMeshLibrary"
    except Exception:
        pass
    try:
        return int(sm.get_editor_property("lod_num")), "lod_num"
    except Exception:
        pass
    return 1, "assumed"


def add_skeletal_mesh(path, pack):
    if path in SKELETAL:
        return SKELETAL[path]["key"]
    sm = unreal.load_asset(path)
    if sm is None:
        ERRORS.append("skeletal mesh not loaded: %s" % path)
        return None
    k = key_of(path)
    nlods, via = skeletal_lod_count(sm)
    rec = {"key": k, "source": path, "pack": pack, "lods": [],
           "material_slots": [], "lod_count_via": via}
    SKELETAL[path] = rec
    try:
        sk = sm.get_editor_property("skeleton")
        rec["skeleton"] = sk.get_path_name() if sk else None
        rec["bones"] = len(sk.get_editor_property("bone_tree")) if sk else 0
    except Exception as e:
        ERRORS.append("skeleton of %s: %s" % (path, e))
    try:
        b = sm.get_bounds()
        rec["bounds_extent_cm"] = [float(b.box_extent.x), float(b.box_extent.y),
                                   float(b.box_extent.z)]
    except Exception as e:
        ERRORS.append("bounds of %s: %s" % (path, e))
    try:
        for slot in sm.get_editor_property("materials"):
            mi = slot.get_editor_property("material_interface")
            rec["material_slots"].append(
                add_material(mi, pack) if mi is not None else None)
    except Exception as e:
        ERRORS.append("slots on %s: %s" % (path, e))
    for lod in range(nlods):
        entry = {"level": lod, "file": None}
        if MODE == "export":
            name = rel("skeletal", "%s_LOD%d.gltf" % (k, lod))
            dst = os.path.join(OUT, name.replace("/", os.sep))
            ensure(os.path.dirname(dst))
            try:
                ok = export_task(sm, dst, None, skeletal_export_options(lod))
            except Exception as e:
                ok = False
                ERRORS.append("gltf %s LOD%d: %s" % (path, lod, e))
            if ok and os.path.isfile(dst):
                entry["file"] = name
                entry.update(gltf_facts(dst))
            else:
                ERRORS.append("skeletal gltf %s LOD%d did not write" % (path, lod))
        rec["lods"].append(entry)
    say("skel %-40s lods=%d bones=%s slots=%d joints=%s" %
        (sm.get_name(), nlods, rec.get("bones"), len(rec["material_slots"]),
         rec["lods"][0].get("joints") if rec["lods"] else "-"))
    return k


def add_clip(pkg, name, pack):
    path = "%s.%s" % (pkg, name)
    if path in CLIPS:
        return CLIPS[path]["key"]
    seq = unreal.load_asset(path)
    if seq is None:
        ERRORS.append("clip not loaded: %s" % path)
        return None
    k = key_of(path)
    rec = {"key": k, "source": path, "pack": pack, "name": name, "file": None}
    CLIPS[path] = rec
    try:
        sk = seq.get_editor_property("skeleton")
        rec["skeleton"] = sk.get_path_name() if sk else None
        rec["skeleton_bones"] = len(sk.get_editor_property("bone_tree")) if sk else 0
    except Exception as e:
        ERRORS.append("skeleton of %s: %s" % (path, e))
    for prop, out in (("sequence_length", "seconds"),
                      ("number_of_sampled_keys", "keys"),
                      ("rate_scale", "rate_scale")):
        try:
            rec[out] = float(seq.get_editor_property(prop))
        except Exception:
            pass
    if MODE == "export":
        fname = rel("clips", "%s.gltf" % k)
        dst = os.path.join(OUT, fname.replace("/", os.sep))
        ensure(os.path.dirname(dst))
        try:
            ok = export_task(seq, dst, None, skeletal_export_options(0))
        except Exception as e:
            ok = False
            ERRORS.append("gltf clip %s: %s" % (path, e))
        if ok and os.path.isfile(dst):
            rec["file"] = fname
            rec.update(gltf_facts(dst))
        else:
            ERRORS.append("clip gltf %s did not write" % path)
    return k


def run_characters():
    """The character sweep. Separate from `run`'s pack loop because a character
    pack names exact assets and a surface pack samples a prefix."""
    packs = []
    for pack in CHARACTERS:
        if ONLY and pack["name"] not in ONLY:
            continue
        say("CHARPACK %s" % pack["name"])
        packs.append({"name": pack["name"], "license": pack["license"],
                      "ship": pack.get("ship", False), "selectors": []})
        for path in pack.get("skeletal", []):
            try:
                add_skeletal_mesh(path, pack["name"])
            except Exception as e:
                ERRORS.append("%s: %s" % (path, e))
                traceback.print_exc()
        # **Discovered skeletal meshes** (wave CHAR1a.2), for a pack whose assets
        # are WRITTEN by a pipeline rather than shipped under known names -- the
        # assembled MetaHumans. Sorted then truncated, exactly as the surface
        # packs' selectors are and for the same reason: an asset registry's order
        # is a function of a scan and this manifest has to be the same manifest
        # twice.
        for sel in pack.get("skeletal_prefix", []):
            pat = sel.get("match")
            limit = sel.get("limit", 8)
            try:
                REG.scan_paths_synchronous([sel["prefix"]], force_rescan=True)
            except Exception as e:
                ERRORS.append("scan %s: %s" % (sel["prefix"], e))
            hits = []
            for a in REG.get_assets_by_path(sel["prefix"], recursive=True):
                if str(a.asset_class_path.asset_name) != "SkeletalMesh":
                    continue
                name = str(a.asset_name)
                if pat and not re.search(pat, name):
                    continue
                hits.append((name, str(a.package_name)))
            hits.sort()
            hits = hits[:limit]
            packs[-1]["selectors"].append({
                "prefix": sel["prefix"], "match": pat, "limit": limit,
                "chosen": [h[1] for h in hits],
            })
            for name, pkg in hits:
                try:
                    add_skeletal_mesh("%s.%s" % (pkg, name), pack["name"])
                except Exception as e:
                    ERRORS.append("%s: %s" % (pkg, e))
                    traceback.print_exc()
        for sel in pack.get("clips", []):
            limit = sel.get("limit", 64)
            pat = sel.get("match")
            try:
                REG.scan_paths_synchronous([sel["prefix"]], force_rescan=True)
            except Exception as e:
                ERRORS.append("scan %s: %s" % (sel["prefix"], e))
            hits = []
            for a in REG.get_assets_by_path(sel["prefix"], recursive=True):
                if str(a.asset_class_path.asset_name) != "AnimSequence":
                    continue
                name = str(a.asset_name)
                if pat and not re.search(pat, name):
                    continue
                hits.append((name, str(a.package_name)))
            hits.sort()
            hits = hits[:limit]
            packs[-1]["selectors"].append({
                "prefix": sel["prefix"], "match": pat, "limit": limit,
                "chosen": [h[1] for h in hits],
            })
            for name, pkg in hits:
                try:
                    add_clip(pkg, name, pack["name"])
                except Exception as e:
                    ERRORS.append("%s: %s" % (pkg, e))
            say("  clips %-24s %d" % (sel["prefix"].split("/")[-1], len(hits)))
    return packs


# ── the sweep ────────────────────────────────────────────────────────────────

def engine_checkout_above(path):
    """The Infini engine checkout `path` sits inside, or None.

    A directory is the checkout when it holds BOTH a `.git` and this very
    script -- the second marker on purpose, so a user's own game repository is
    not mistaken for the engine's. Mirrors
    `inf_editor_core::assets::ue_import::engine_checkout_above`, which guards
    the import side of the same law.
    """
    p = os.path.abspath(path)
    while True:
        if os.path.exists(os.path.join(p, ".git")) and os.path.isfile(
                os.path.join(p, "tools", "ue-export", "export.py")):
            return p
        parent = os.path.dirname(p)
        if parent == p:
            return None
        p = parent


def run():
    if not OUT:
        say("REFUSED: set INF_UE_OUT to an output directory outside the project")
        return
    # **THE LICENCE LAW, AS A DOOR** (ASSET0 audit). Everything below this line
    # is Marketplace/Fab/Megascans content whose licence for use outside Unreal
    # is unestablished, and the one place it may never land is the PUBLIC engine
    # repository. `INF_UE_OUT` used to be believed rather than checked.
    repo = engine_checkout_above(OUT)
    if repo:
        say("REFUSED: INF_UE_OUT=%s is inside the engine checkout at %s." % (OUT, repo))
        say("         Nothing this script writes may enter that repository -- the "
            "packs are licensed content. Choose a directory outside it.")
        return
    ensure(OUT)
    say("BEGIN mode=%s out=%s maxtex=%d" % (MODE, OUT, MAXTEX))
    t0 = datetime.datetime.now(datetime.timezone.utc)
    packs = []
    for pack in PACKS:
        if ONLY and pack["name"] not in ONLY:
            continue
        say("PACK %s" % pack["name"])
        packs.append({"name": pack["name"], "license": pack["license"],
                      "selectors": []})
        for sel in pack["select"]:
            wanted = set(sel.get("classes", ["StaticMesh"]))
            pat = sel.get("match")
            limit = sel.get("limit", 8)
            hits = []
            for a in REG.get_assets_by_path(sel["prefix"], recursive=True):
                cls = str(a.asset_class_path.asset_name)
                if cls not in wanted:
                    continue
                name = str(a.asset_name)
                if pat and not re.search(pat, name):
                    continue
                hits.append((name, str(a.package_name), cls))
            # Sorted, then truncated. An asset registry's order is a function of
            # a scan and this manifest has to be the same manifest twice.
            hits.sort()
            hits = hits[:limit]
            packs[-1]["selectors"].append({
                "prefix": sel["prefix"], "match": pat, "limit": limit,
                "chosen": [h[1] for h in hits],
            })
            for name, pkg, cls in hits:
                try:
                    obj = unreal.load_asset(pkg)
                    if obj is None:
                        ERRORS.append("could not load %s" % pkg)
                        continue
                    if cls == "StaticMesh":
                        add_mesh(obj, pack["name"])
                    elif cls == "Blueprint":
                        add_blueprint(obj, pack["name"])
                    else:
                        add_material(obj, pack["name"],
                                     surface=sel.get("surface", False),
                                     no_textures=sel.get("no_textures", False))
                except Exception as e:
                    ERRORS.append("%s: %s" % (pkg, e))
                    say("  ! %s: %s" % (pkg, e))
                    traceback.print_exc()

    packs.extend(run_characters())

    t1 = datetime.datetime.now(datetime.timezone.utc)
    manifest = {
        # **v2 (wave CHAR1a): `skeletal_meshes` and `clips` are new SECTIONS.**
        # A v1 reader ignores them (every field is `#[serde(default)]` on the
        # Rust side) but would then report success having imported no character,
        # so the version is bumped and the importer's guard is what refuses --
        # the same law the `.ipack` header carries: a newer container is
        # rejected by name, and a reader that grows an arm keys it on the
        # version rather than reinterpreting the bytes.
        "schema_version": 2,
        "generator": "tools/ue-export/export.py",
        "engine": unreal.SystemLibrary.get_engine_version(),
        "project": unreal.Paths.get_project_file_path(),
        "exported_utc": t0.isoformat(),
        "seconds": (t1 - t0).total_seconds(),
        "mode": MODE,
        "units": "meshes are glTF (metres, Y up, right handed) as UE's exporter "
                 "writes them; socket transforms are RAW UE centimetres in UE's "
                 "own frame and the importer converts them",
        "packs": packs,
        "meshes": sorted(MESHES.values(), key=lambda r: r["key"]),
        "skeletal_meshes": sorted(SKELETAL.values(), key=lambda r: r["key"]),
        "clips": sorted(CLIPS.values(), key=lambda r: r["key"]),
        "materials": sorted(MATERIALS.values(), key=lambda r: r["key"]),
        "fixtures": sorted(FIXTURES.values(), key=lambda r: r["key"]),
        "textures": sorted(TEXTURES.values(), key=lambda r: r["key"]),
        "errors": ERRORS,
    }
    path = os.path.join(OUT, "manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    tex_bytes = sum(t.get("bytes", 0) for t in TEXTURES.values())
    say("WROTE %s" % path)
    say("TOTALS meshes=%d lods=%d skeletal=%d skel_lods=%d clips=%d "
        "materials=%d textures=%d fixtures=%d "
        "lights=%d texture_MB=%.1f errors=%d seconds=%.1f" % (
            len(MESHES), sum(len(m["lods"]) for m in MESHES.values()),
            len(SKELETAL), sum(len(m["lods"]) for m in SKELETAL.values()),
            len(CLIPS),
            len(MATERIALS), len(TEXTURES), len(FIXTURES),
            sum(len(f["lights"]) for f in FIXTURES.values()),
            tex_bytes / 1048576.0, len(ERRORS), (t1 - t0).total_seconds()))
    for e in ERRORS:
        say("ERROR %s" % e)
    say("END")


run()
