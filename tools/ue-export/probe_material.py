"""WHAT A UE MASTER MATERIAL DOES WITH A CHANNEL, PROBED (wave CHAR1a.3, clause E).

Read-only, commandlet-safe, and it never writes to the project it opens.

    "C:/Program Files/Epic Games/UE_5.8/Engine/Binaries/Win64/UnrealEditor-Cmd.exe" \
        "<project>.uproject" -run=pythonscript \
        -script=tools/ue-export/probe_material.py \
        -unattended -nop4 -nosplash -stdout -FullStdOutLogOutput -NoShaderCompile

    INF_UE_OUT       where the JSON report is written           (required)
    INF_PROBE_MATS   comma-separated /Game/... object paths     (required)

# WHY

Wave CHAR1a.2 probed `M_Mannequin`'s PARAMETER NAMES and mapped its eight
textures onto this engine's slots. It did not probe the GRAPH, and the parameter
names alone do not answer the question the frames raise: UE's Manny is a light
grey android and ours is a black glossy one, and the measured cause is that
`T_*_MSR_MSK`'s R plane (median **247**) reaches this engine's `metallic`
directly. Whether that is what UE does with it is a property of the node graph
between `MSR_tex` and the `Metallic` input, not of the texture.

So this walks the master material's expressions, records what each is and what
it is connected to, and reports the chain that reaches each material input. Five
doors are tried for the expression list because UE 5.5 moved `UMaterial::
Expressions` into an editor-only `MaterialExpressionCollection`, and which door
answers is itself a fact worth recording rather than guessing at.

# LICENCE

Nothing is exported: this writes a JSON description of a graph's SHAPE, which is
a measurement about the reference project and not its content. The output
directory is refused if it is inside the engine checkout, exactly as
`export.py`'s is.
"""

import json
import os
import traceback

import unreal

OUT = os.environ.get("INF_UE_OUT", "")
MATS = [m for m in os.environ.get("INF_PROBE_MATS", "").split(",") if m]


def say(*a):
    print("UEX:", *a)


def engine_checkout_above(path):
    p = os.path.abspath(path)
    while True:
        if os.path.exists(os.path.join(p, ".git")) and os.path.isfile(
                os.path.join(p, "tools", "ue-export", "export.py")):
            return p
        parent = os.path.dirname(p)
        if parent == p:
            return None
        p = parent


def expressions_of(mat):
    """The material's expression list, and WHICH door answered."""
    tries = []

    def attempt(name, fn):
        try:
            v = fn()
            if v is None:
                tries.append((name, "None"))
                return None
            v = list(v)
            tries.append((name, "%d expressions" % len(v)))
            return v
        except Exception as e:
            tries.append((name, "%s: %s" % (type(e).__name__, str(e)[:120])))
            return None

    got = attempt("MaterialEditingLibrary.get_material_expressions",
                  lambda: unreal.MaterialEditingLibrary.get_material_expressions(mat))
    if got is None:
        got = attempt("editor_only_data.expression_collection.expressions",
                      lambda: mat.get_editor_property("editor_only_data")
                      .get_editor_property("expression_collection")
                      .get_editor_property("expressions"))
    if got is None:
        got = attempt("expression_collection.expressions",
                      lambda: mat.get_editor_property("expression_collection")
                      .get_editor_property("expressions"))
    if got is None:
        got = attempt("expressions",
                      lambda: mat.get_editor_property("expressions"))
    if got is None:
        got = attempt("get_expressions", lambda: mat.get_expressions())
    return got or [], tries


def describe(ex):
    """One expression: its class, its identifying parameter, its inputs."""
    row = {"class": ex.get_class().get_name()}
    for prop in ("parameter_name", "texture", "desc", "const", "r", "g", "b", "a",
                 "mask_r", "mask_g", "mask_b", "mask_a", "default_value",
                 "const_a", "const_b", "material_function"):
        try:
            v = ex.get_editor_property(prop)
        except Exception:
            continue
        if v is None:
            continue
        if hasattr(v, "get_path_name"):
            v = v.get_path_name()
        elif not isinstance(v, (int, float, bool, str)):
            v = str(v)
        row[prop] = v
    # Every `FExpressionInput`-shaped property, with what it points at.
    ins = {}
    for prop in ("a", "b", "input", "rgb", "alpha", "coordinates", "tex",
                 "base", "exponent", "scale", "value", "index", "x", "y", "z",
                 "expression"):
        try:
            v = ex.get_editor_property(prop)
        except Exception:
            continue
        try:
            e = v.get_editor_property("expression")
        except Exception:
            continue
        if e is not None:
            ins[prop] = e.get_class().get_name()
    if ins:
        row["inputs"] = ins
    return row


def probe(path):
    obj = unreal.load_asset(path)
    if obj is None:
        return {"material": path, "error": "did not load"}
    base = obj
    parents = []
    while isinstance(base, unreal.MaterialInstance):
        parents.append(base.get_path_name())
        base = base.get_editor_property("parent")
        if base is None:
            return {"material": path, "parents": parents,
                    "error": "instance chain has no base material"}
    out = {"material": path, "base": base.get_path_name(), "parents": parents,
           "class": obj.get_class().get_name()}
    for prop in ("blend_mode", "shading_model", "two_sided"):
        try:
            out[prop] = str(base.get_editor_property(prop))
        except Exception:
            pass
    # The SCALAR/VECTOR/SWITCH parameters and their DEFAULTS on the base, and
    # the overrides the instance sets -- because "Metallic is a parameter that
    # defaults below 1" is one of the hypotheses this probe exists to settle.
    def names(fn):
        try:
            return [str(n) for n in fn(base)]
        except Exception:
            return []
    lib = unreal.MaterialEditingLibrary
    out["scalar_defaults"] = {}
    for n in names(lib.get_scalar_parameter_names):
        try:
            out["scalar_defaults"][n] = float(
                lib.get_material_default_scalar_parameter_value(base, n))
        except Exception:
            pass
    out["switch_defaults"] = {}
    for n in names(lib.get_static_switch_parameter_names):
        try:
            out["switch_defaults"][n] = bool(
                lib.get_material_default_static_switch_parameter_value(base, n))
        except Exception:
            pass
    out["vector_defaults"] = {}
    for n in names(lib.get_vector_parameter_names):
        try:
            v = lib.get_material_default_vector_parameter_value(base, n)
            out["vector_defaults"][n] = [float(v.r), float(v.g), float(v.b), float(v.a)]
        except Exception:
            pass
    if isinstance(obj, unreal.MaterialInstanceConstant):
        out["instance_scalars"] = {}
        try:
            for p in obj.get_editor_property("scalar_parameter_values"):
                n = str(p.get_editor_property("parameter_info")
                        .get_editor_property("name"))
                out["instance_scalars"][n] = float(
                    p.get_editor_property("parameter_value"))
        except Exception:
            pass
        out["instance_switches"] = {}
        try:
            for p in obj.get_editor_property("static_parameters").get_editor_property(
                    "static_switch_parameters"):
                n = str(p.get_editor_property("parameter_info")
                        .get_editor_property("name"))
                out["instance_switches"][n] = bool(p.get_editor_property("value"))
        except Exception:
            pass
    exprs, tries = expressions_of(base)
    out["expression_doors"] = ["%s -> %s" % (a, b) for a, b in tries]
    out["expressions"] = [describe(e) for e in exprs]
    out["expression_count"] = len(exprs)
    return out


def run():
    if not OUT:
        say("REFUSED: set INF_UE_OUT")
        return
    repo = engine_checkout_above(OUT)
    if repo:
        say("REFUSED: INF_UE_OUT=%s is inside the engine checkout at %s" % (OUT, repo))
        return
    if not os.path.isdir(OUT):
        os.makedirs(OUT)
    rows = []
    for m in MATS:
        say("PROBE %s" % m)
        try:
            rows.append(probe(m))
        except Exception as e:
            rows.append({"material": m, "error": "%s: %s" % (type(e).__name__, e)})
            traceback.print_exc()
    dst = os.path.join(OUT, "material_graph.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, sort_keys=True)
    say("WROTE %s" % dst)
    for r in rows:
        say("  %-46s expressions=%s doors=%s" % (
            r.get("material", "?").split(".")[-1], r.get("expression_count"),
            r.get("expression_doors")))
    say("END")


run()
