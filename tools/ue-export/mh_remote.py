"""THE LIVE-EDITOR DOOR (wave CHAR1a.2).

Runs on the HOST python (not inside Unreal). It boots the Unreal Editor **with a
window** on the scratch project, waits for it, and then drives
`tools/ue-export/metahuman.py` inside that live editor over the Python plugin's
own remote-execution protocol
(`Engine/Plugins/Experimental/PythonScriptPlugin/Content/Python/remote_execution.py`).

    python tools/ue-export/mh_remote.py --stage assemble --out <dir>

# WHY THIS EXISTS

Wave CHAR1a proved the MetaHuman *preparation* headless (duplicate a preset,
auto-rig, download textures) and proved that **assembly is not**: three
commandlet doors die in the same two places -- `EXCEPTION_ACCESS_VIOLATION` at
`0x200` inside `UnrealEditor-TextureGraph.dll` (the pipeline composites skin
through the Texture Graph, which needs a rendering device) and
`Assertion failed: CurrentApplication.IsValid()` in `SlateApplication.h:321`
(it creates its assets through the Content Browser, which needs a Slate
application). `-AllowCommandletRendering` gets past the first and then hangs on
the second, because a commandlet has no editor tick loop.

A *live* editor has both. So instead of asking a human to press **Assemble**,
this script starts the editor the human would have started, and types into it.
Nobody has to touch the window; it is closed again when the work is done.

# THE PROTOCOL, AND WHAT IT NEEDS

`bRemoteExecution=True` under
`[/Script/PythonScriptPlugin.PythonScriptPluginSettings]` in the project's
`Config/DefaultEngine.ini` -- `--ensure-config` writes it if it is missing. The
editor then joins a UDP multicast group (239.0.0.1:6766 by default, TTL 0 =
this host only) and announces itself; this script discovers that node, asks it
to open a TCP command channel back, and sends `ExecuteFile` commands over it.

The command channel is SYNCHRONOUS: `run_command` blocks until the editor's
game thread finishes the statement and answers. An assembly that takes ten
minutes therefore takes ten minutes here, with no polling and no guessing --
which is the whole reason this is better than watching a log.

# WHAT IT REFUSES

The same licence law the rest of the bridge carries: `--out` may not be inside
the engine checkout (a directory holding both `.git` and
`tools/ue-export/export.py`). Nothing from Unreal is ever written there.
"""

import argparse
import io
import os
import subprocess
import sys
import time

ENGINE = os.environ.get(
    "INF_UE_ENGINE", r"C:\Program Files\Epic Games\UE_5.8\Engine")
EDITOR = os.path.join(ENGINE, "Binaries", "Win64", "UnrealEditor.exe")
REMOTE_PY = os.path.join(ENGINE, "Plugins", "Experimental", "PythonScriptPlugin",
                         "Content", "Python")
HERE = os.path.dirname(os.path.abspath(__file__))

CONFIG_SECTION = "[/Script/PythonScriptPlugin.PythonScriptPluginSettings]"
CONFIG_LINES = [
    "bDeveloperMode=True",
    "bRemoteExecution=True",
    "RemoteExecutionMulticastGroupEndpoint=239.0.0.1:6766",
    "RemoteExecutionMulticastBindAddress=127.0.0.1",
    "RemoteExecutionMulticastTtl=0",
]


def say(*a):
    print("MHR:", *a, flush=True)


def engine_checkout_above(path):
    """The Infini engine checkout `path` sits inside, or None (mirrors
    `export.py`'s guard, and refuses for the same reason)."""
    p = os.path.abspath(path)
    while True:
        if os.path.exists(os.path.join(p, ".git")) and os.path.isfile(
                os.path.join(p, "tools", "ue-export", "export.py")):
            return p
        parent = os.path.dirname(p)
        if parent == p:
            return None
        p = parent


def ensure_config(uproject):
    """Write the remote-execution settings into the project's DefaultEngine.ini
    if they are not already there. Returns True when the file was changed."""
    ini = os.path.join(os.path.dirname(uproject), "Config", "DefaultEngine.ini")
    if not os.path.isdir(os.path.dirname(ini)):
        os.makedirs(os.path.dirname(ini))
    text = io.open(ini, encoding="utf-8").read() if os.path.isfile(ini) else ""
    if "bRemoteExecution=True" in text:
        return False
    text = text.rstrip("\n") + "\n\n" + CONFIG_SECTION + "\n" + \
        "\n".join(CONFIG_LINES) + "\n"
    io.open(ini, "w", encoding="utf-8", newline="\n").write(text)
    return True


def launch(uproject, log):
    """Start the editor with a window, logging where we can read it.

    **NOT detached, and that is a measured requirement, not a style.** The first
    attempt at this door used `DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP` with
    stdout on the null device — the tidy way to start a long-lived GUI child. The
    editor booted to `LogShaderCompilers: Current jobs: 206, Batch size: 9` and
    then **stopped**: 58 s of CPU in 21 minutes, 0.14 s of CPU in an 8 s sample,
    no file I/O, zero `ShaderCompileWorker` processes alive and **no top-level
    window at all** (`EnumWindows` over the pid returned nothing). Relaunched with
    ordinary inherited handles it spawned **twelve** workers inside 20 seconds and
    reached its window. A detached editor cannot start its own compile workers, and
    the failure is silent — it looks exactly like the commandlet hang wave CHAR1a
    recorded, which is why it is written down here rather than remembered.

    **`-noxgeshadercompile`, and it was PROSE and not a flag until the CHAR1a
    audit.** The wave measured the third boot behaviour and wrote it into its
    ledger — *"`-noxgeshadercompile` -> MainWindowTitle in ~90 s, 267 s of
    CPU"* — and then did not put the token in this argv, so the tool as
    committed still selected the XGE (IncrediBuild) controller
    (`LogShaderCompilers: Display: Using XGE Controller for shader compilation`),
    which never dispatches on this machine and which the editor waits for for
    ever with no error. Carried item 90 says this flag *is* in `mh_remote.py`;
    it was not, and a grep for it over the whole repository found two hits, both
    in the memo's prose. It is here now.
    """
    argv = [EDITOR, uproject, "-nosplash", "-nop4", "-noxgeshadercompile",
            "-abslog=" + os.path.abspath(log)]
    say("LAUNCH", " ".join(argv))
    return subprocess.Popen(argv)


def connect(timeout_s):
    """Discover the live editor and open a command channel to it."""
    sys.path.insert(0, REMOTE_PY)
    import remote_execution  # noqa: E402  (only importable once the path is set)

    rx = remote_execution.RemoteExecution()
    rx.start()
    deadline = time.time() + timeout_s
    node = None
    while time.time() < deadline:
        nodes = rx.remote_nodes
        if nodes:
            node = nodes[0]
            break
        time.sleep(1.0)
    if node is None:
        rx.stop()
        raise RuntimeError(
            "no Unreal node answered the multicast ping in %d s -- is "
            "bRemoteExecution=True and is the editor past its boot?" % timeout_s)
    say("NODE", node)
    rx.open_command_connection(node["node_id"])
    return rx, remote_execution


def run_script(rx, mod, script):
    """Run a literal multi-statement script inside the editor and print it."""
    res = rx.run_command(script, unattended=True,
                         exec_mode=mod.MODE_EXEC_FILE, raise_on_failure=False)
    for line in res.get("output") or []:
        say("  UE[%s] %s" % (line.get("type"), line.get("output", "").rstrip()))
    say("SUCCESS=%s RESULT=%s" % (res.get("success"), res.get("result")))
    return res


def stage_script(stage, out, work, male, female):
    """The literal script the editor executes: set the environment the tool
    reads, then execute that very file. One source of truth — the stage bodies
    live in `metahuman.py` (and in `export.py` for the glTF crossing) and this
    file never copies them.

    The `export-gltf` stage runs **`export.py`**, not `metahuman.py`: once a
    MetaHuman is assembled its meshes are ordinary `USkeletalMesh` assets, and
    the bridge that already exports skeletal meshes, their LOD ladders, their
    materials and their textures is the one that should export these. Nothing
    about the crossing is MetaHuman-specific, which is the point.
    """
    tool = "export.py" if stage == "export-gltf" else "metahuman.py"
    mh = os.path.join(HERE, tool).replace("\\", "/")
    return "\n".join([
        "import os",
        "os.environ['INF_UE_OUT'] = r'''%s'''" % out,
        "os.environ['INF_MH_STAGE'] = '%s'" % stage,
        "os.environ['INF_MH_WORK'] = '%s'" % work,
        "os.environ['INF_MH_MALE'] = '%s'" % male,
        "os.environ['INF_MH_FEMALE'] = '%s'" % female,
        "os.environ['INF_UE_MODE'] = 'export'",
        "os.environ['INF_UE_PACKS'] = 'MetaHumans'",
        # `metahuman.py` reads `__file__` (its licence guard walks up from its own
        # directory). A remote `ExecuteFile` of a literal has no `__file__` at
        # all — the first live run died on `NameError: name '__file__' is not
        # defined` — so it is bound here, to the path the source actually came
        # from, rather than the guard being weakened to cope.
        "__file__ = r'''%s'''" % mh,
        "_src = open(r'''%s''', encoding='utf-8').read()" % mh,
        "exec(compile(_src, r'''%s''', 'exec'))" % mh,
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "..", "ue-staging", "MHForge", "MHForge.uproject"))
    ap.add_argument("--stage", default="assemble")
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="/Game/INF")
    ap.add_argument("--male", default="Dominic")
    ap.add_argument("--female", default="Vivian")
    ap.add_argument("--log", default=None)
    ap.add_argument("--boot-timeout", type=int, default=900)
    ap.add_argument("--no-launch", action="store_true")
    ap.add_argument("--keep-open", action="store_true")
    args = ap.parse_args()

    uproject = os.path.abspath(args.project)
    out = os.path.abspath(args.out)
    repo = engine_checkout_above(out)
    if repo:
        say("REFUSED: --out=%s is inside the engine checkout at %s" % (out, repo))
        return 2
    if not os.path.isdir(out):
        os.makedirs(out)

    changed = ensure_config(uproject)
    say("CONFIG", "written" if changed else "already enabled")

    proc = None
    if not args.no_launch:
        log = args.log or os.path.join(out, "mh_editor.log")
        proc = launch(uproject, log)

    t0 = time.time()
    rx, mod = connect(args.boot_timeout)
    say("CONNECTED after %.1f s" % (time.time() - t0))
    try:
        t1 = time.time()
        res = run_script(rx, mod, stage_script(
            args.stage, out, args.work, args.male, args.female))
        say("STAGE %s took %.1f s" % (args.stage, time.time() - t1))
        ok = bool(res.get("success"))
        if not args.keep_open:
            say("QUITTING the editor")
            try:
                rx.run_command("import unreal; unreal.SystemLibrary.quit_editor()",
                               unattended=True, exec_mode=mod.MODE_EXEC_STATEMENT,
                               raise_on_failure=False)
            except Exception as e:
                say("quit:", e)
        return 0 if ok else 1
    finally:
        try:
            rx.stop()
        except Exception:
            pass
        if proc is not None and not args.keep_open:
            for _ in range(60):
                if proc.poll() is not None:
                    break
                time.sleep(1.0)
            if proc.poll() is None:
                say("editor did not exit; terminating")
                proc.terminate()


if __name__ == "__main__":
    sys.exit(main())
