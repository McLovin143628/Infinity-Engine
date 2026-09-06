# THE DEMO LOOP (wave FIX1) — build the editor, boot it on the showcase island,
# press its own Play button, drive the game, and photograph the result.
#
# A wave that ends in a green battery has proved the tests agree with the code.
# It has not proved that the editor opens, that Play plays, or that the character
# walks. Every wave from FIX1 onward ends here. See tools/demo/README.md.
param(
    [string]$OutDir = "",
    [switch]$SkipBuild,
    [switch]$KeepOpen,
    [int]$Port = 9222,
    # "embedded" reparents the player into the viewport hole; "window" is the
    # roadmap-sanctioned Play in New Window. Both must move the hero, so both
    # are drivable from here.
    [ValidateSet("embedded", "window")][string]$PlayMode = "embedded",
    [int]$BootWaitS = 60,
    [int]$PieWaitS = 240,
    [int]$LoadSettleS = 20,
    # **The floor that makes this a GATE and not a report** (audit FIX1). The
    # wave that wrote this script printed HERO MOVED and exited 0 whatever the
    # number was -- including the runs it later found had moved 0.000 m, which
    # were noticed by a person reading the log. Twelve metres is what a held W
    # buys in the seconds this script allows; five is clear of a settle, a slide
    # or a camera drift and far below anything a walking character does.
    [double]$MinMetres = 5.0,
    # **How long the EDITOR is given to finish streaming** before the frame a
    # claim about the editor is made on. CHAR1a photographed the viewport at
    # 27/52 and could not tell an unresolved material from a dropped one.
    [int]$EditorSettleS = 45,
    # Place the second committed body beside the pawn before the editor frame,
    # in the DOCUMENT only. See tools/demo/place.mjs for why it is not saved.
    [bool]$PlaceFemale = $true
)

$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"
$repo = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$release = Join-Path $repo "target\release"
$exe = Join-Path $release "inf-studio.exe"
if ($OutDir -eq "") {
    $OutDir = Join-Path $env:TEMP ("inf-demo-" + (Get-Date -Format "yyyyMMdd-HHmmss"))
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$log = Join-Path $OutDir "demo.log"
$heroCsv = Join-Path $OutDir "hero.csv"
$shot = Join-Path $PSScriptRoot "screenshot.ps1"

function Say([string]$text) {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $text
    Write-Output $line
    Add-Content -Path $log -Value $line
}

$failed = $false

Say "repo    $repo"
Say "mode    $PlayMode"
Say "out     $OutDir"

# ── 0. nothing of ours may be running ────────────────────────────────────────
#
#    The island's pack is memory-mapped and a build that tries to replace a
#    RUNNING executable fails as a sharing violation, which MSVC reports as
#    LNK1104 and which reads like a disk problem. Refuse early and say why.
$running = Get-Process -ErrorAction SilentlyContinue |
    Where-Object { $_.ProcessName -in @("inf-studio", "inf-player") }
if ($running) {
    Say ("REFUSED: these are already running -> " + (($running | ForEach-Object { "$($_.ProcessName)/$($_.Id)" }) -join ", "))
    Say "Close the editor first, or pass -SkipBuild to photograph what is already built."
    exit 2
}

# ── 1. build ─────────────────────────────────────────────────────────────────
if (-not $SkipBuild) {
    # THE PLAYER FIRST, and it is not optional: `npx tauri build` builds the
    # EDITOR. `find_player_bin` looks for `inf-player.exe` beside the editor it
    # is running from, so a demo that built only the editor would press Play on
    # whatever player happened to be in `target/release` -- which on a dev
    # machine is the one from the wave before last, and which is exactly the
    # trap this comment exists to keep the next person out of.
    Say "building: cargo build --release -p inf-player"
    & cargo build --release -p inf-player 2>&1 | ForEach-Object { Add-Content -Path $log -Value $_ }
    if ($LASTEXITCODE -ne 0) {
        Say "PLAYER BUILD FAILED (exit $LASTEXITCODE) — see $log"
        exit 3
    }
    Say "building: npx tauri build --no-bundle"
    Push-Location (Join-Path $repo "editor\studio")
    # `cargo build --release -p inf-studio` is NOT the same thing and produces an
    # editor that loads the DEV url: the frontend has to be built and embedded,
    # which is what the tauri CLI does.
    & npx tauri build --no-bundle 2>&1 | ForEach-Object { Add-Content -Path $log -Value $_ }
    $code = $LASTEXITCODE
    Pop-Location
    if ($code -ne 0) {
        Say "BUILD FAILED (exit $code) — see $log"
        exit 3
    }
    Say "build ok"
}
if (-not (Test-Path $exe)) {
    Say "REFUSED: no editor at $exe"
    exit 3
}
$playerExe = Join-Path $release "inf-player.exe"
if (-not (Test-Path $playerExe)) {
    Say ("REFUSED: no player at $playerExe - build it with: cargo build --release -p inf-player")
    exit 3
}
Say ("editor  {0} ({1:N1} MB, built {2})" -f $exe, ((Get-Item $exe).Length / 1MB), (Get-Item $exe).LastWriteTime)
Say ("player  {0} ({1:N1} MB, built {2})" -f $playerExe, ((Get-Item $playerExe).Length / 1MB), (Get-Item $playerExe).LastWriteTime)

# ── 2. launch, from the executable's OWN directory ───────────────────────────
#
#    The boot ladder discovers the showcase by walking up from the running
#    executable, so the working directory is load-bearing: launched from
#    elsewhere the editor opens the start screen instead of the island.
$env:INF_WEBVIEW_DEBUG_PORT = "$Port"
$env:INF_PIE_HERO_LOG = $heroCsv
$proc = Start-Process -FilePath $exe -WorkingDirectory $release -PassThru
Say "launched pid $($proc.Id); waiting up to $BootWaitS s for the shell"

$booted = $false
for ($i = 0; $i -lt $BootWaitS; $i++) {
    Start-Sleep -Seconds 1
    if ($proc.HasExited) { Say "EDITOR EXITED with $($proc.ExitCode)"; exit 4 }
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/json" -UseBasicParsing -TimeoutSec 2
        if ($r.StatusCode -eq 200) { $booted = $true; Say "debug port open after $($i + 1) s"; break }
    } catch { }
}
if (-not $booted) { Say "debug port never opened; falling back to a fixed wait"; Start-Sleep -Seconds 15 }
# The shell paints its panels after the port opens; give the island's document a
# moment to land in the Outliner before the first frame is taken.
Start-Sleep -Seconds 10

& powershell -NoProfile -ExecutionPolicy Bypass -File $shot -Out (Join-Path $OutDir "01-editor.png") -WindowTitle "Infini" -Foreground |
    ForEach-Object { Say $_ }

# ── 2b. the SETTLED editor frame, and the female body in it ──────────────────
#
#    **The mid-stream frame is not the editor's answer** (CHAR1a's own caveat):
#    the first shot above was taken while the toolbar still read `Loading
#    world... 27/52`, so a body that had not resolved its material yet looked
#    exactly like a body whose material was dropped. This one waits for the
#    stream and is the frame a claim about the editor may be made on.
Say "waiting $EditorSettleS s for the editor's own streaming to settle"
Start-Sleep -Seconds $EditorSettleS
if ($PlaceFemale -and (Get-Command node -ErrorAction SilentlyContinue)) {
    Say "placing the FEMALE committed body beside the pawn (document only; never saved)"
    & node (Join-Path $PSScriptRoot "place.mjs") $Port 2>&1 | ForEach-Object { Say "  cdp: $_" }
    if ($LASTEXITCODE -ne 0) { Say "  place.mjs exit $LASTEXITCODE" }
    Start-Sleep -Seconds 3
}
& powershell -NoProfile -ExecutionPolicy Bypass -File $shot -Out (Join-Path $OutDir "01b-editor-settled.png") -WindowTitle "Infini" -Foreground |
    ForEach-Object { Say $_ }

# ── 3. press Play ────────────────────────────────────────────────────────────
$pressed = $false
if (Get-Command node -ErrorAction SilentlyContinue) {
    Say "pressing Play over CDP"
    & node (Join-Path $PSScriptRoot "play.mjs") $Port 8 $PlayMode 2>&1 | ForEach-Object { Say "  cdp: $_" }
    if ($LASTEXITCODE -eq 0) { $pressed = $true } else { Say "  cdp failed (exit $LASTEXITCODE)" }
} else {
    Say "node is not on the PATH"
}
if (-not $pressed) {
    # The fallback: the Play cluster's first button on a maximized 1080p window.
    Add-Type -AssemblyName System.Windows.Forms
    if ($PlayMode -ne "embedded") {
        Say "REFUSED: -PlayMode window needs the CDP path (there is no coordinate for a menu item)"
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        exit 6
    }
    Say "pressing Play by coordinate (1220, 49)"
    $wshell = New-Object -ComObject wscript.shell
    $wshell.AppActivate($proc.Id) | Out-Null
    Start-Sleep -Milliseconds 600
    [System.Windows.Forms.Cursor]::Position = New-Object System.Drawing.Point(1220, 49)
    Start-Sleep -Milliseconds 200
    Add-Type @"
using System;
using System.Runtime.InteropServices;
public class InfClick {
  [DllImport("user32.dll")] public static extern void mouse_event(uint f, uint x, uint y, uint d, IntPtr e);
}
"@
    [InfClick]::mouse_event(0x0002, 0, 0, 0, [IntPtr]::Zero)
    [InfClick]::mouse_event(0x0004, 0, 0, 0, [IntPtr]::Zero)
}

# ── 4. wait for the player ───────────────────────────────────────────────────
Say "waiting up to $PieWaitS s for inf-player.exe"
$player = $null
for ($i = 0; $i -lt $PieWaitS; $i++) {
    $player = Get-Process -Name "inf-player" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($player) { Say "player pid $($player.Id) after $($i + 1) s"; break }
    if ($proc.HasExited) { Say "EDITOR EXITED with $($proc.ExitCode)"; exit 4 }
    Start-Sleep -Seconds 1
}
if (-not $player) {
    Say "NO PLAYER after $PieWaitS s"
    & powershell -NoProfile -ExecutionPolicy Bypass -File $shot -Out (Join-Path $OutDir "02-no-player.png") | ForEach-Object { Say $_ }
    if (-not $KeepOpen) { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue }
    exit 5
}

# A console window is the defect this wave closed; look for one belonging to
# either process while both are alive.
$consoles = Get-Process -ErrorAction SilentlyContinue |
    Where-Object { $_.ProcessName -eq "conhost" -or $_.ProcessName -eq "WindowsTerminal" } |
    Where-Object { $_.MainWindowTitle -like "*inf-player*" }
Say ("console windows named inf-player: " + $(if ($consoles) { ($consoles | ForEach-Object { $_.MainWindowTitle }) -join "; " } else { "none" }))

Say "letting the level stream for $LoadSettleS s"
Start-Sleep -Seconds $LoadSettleS

# ── 5. drive it, and photograph two seconds apart ────────────────────────────
Add-Type @"
using System;
using System.Runtime.InteropServices;
public class InfInput {
  [StructLayout(LayoutKind.Sequential)] struct KEYBDINPUT { public ushort wVk, wScan; public uint dwFlags, time; public IntPtr dwExtraInfo; }
  [StructLayout(LayoutKind.Sequential)] struct INPUT { public uint type; public KEYBDINPUT ki; public int pad1, pad2; }
  [DllImport("user32.dll", SetLastError = true)] static extern uint SendInput(uint n, INPUT[] p, int cb);
  [DllImport("user32.dll")] static extern void mouse_event(uint f, uint x, uint y, uint d, IntPtr e);
  [DllImport("user32.dll")] public static extern bool SetCursorPos(int x, int y);
  [StructLayout(LayoutKind.Sequential)] public struct RECT { public int Left, Top, Right, Bottom; }
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
  [DllImport("user32.dll")] static extern IntPtr GetForegroundWindow();
  [DllImport("user32.dll")] public static extern IntPtr GetParent(IntPtr h);
  [DllImport("user32.dll")] static extern uint GetWindowThreadProcessId(IntPtr h, out uint pid);
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
  [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int cmd);
  // Which window the keyboard is going to, and who owns it.
  public static string Foreground() {
    IntPtr h = GetForegroundWindow();
    uint pid; GetWindowThreadProcessId(h, out pid);
    return string.Format("hwnd=0x{0:x} pid={1}", h.ToInt64(), pid);
  }
  const uint KEYEVENTF_SCANCODE = 0x0008, KEYEVENTF_KEYUP = 0x0002;
  static void Key(ushort scan, bool down) {
    INPUT[] i = new INPUT[1];
    i[0].type = 1;
    i[0].ki.wScan = scan;
    i[0].ki.dwFlags = KEYEVENTF_SCANCODE | (down ? 0u : KEYEVENTF_KEYUP);
    SendInput(1, i, Marshal.SizeOf(typeof(INPUT)));
  }
  public static void Down(ushort scan) { Key(scan, true); }
  public static void Up(ushort scan) { Key(scan, false); }
  public static void Click(int x, int y) {
    SetCursorPos(x, y);
    mouse_event(0x0002, 0, 0, 0, IntPtr.Zero);
    mouse_event(0x0004, 0, 0, 0, IntPtr.Zero);
  }
  // A screenshot CANNOT answer "is the cursor hidden" -- `CopyFromScreen` does
  // not draw one either way -- so the author's second sentence needs the OS's
  // own answer. CURSOR_SHOWING is 0x1.
  [StructLayout(LayoutKind.Sequential)] struct CURSORINFO { public int cbSize, flags; public IntPtr hCursor; public int x, y; }
  [DllImport("user32.dll")] static extern bool GetCursorInfo(ref CURSORINFO pci);
  public static string CursorState() {
    CURSORINFO ci = new CURSORINFO();
    ci.cbSize = Marshal.SizeOf(typeof(CURSORINFO));
    if (!GetCursorInfo(ref ci)) return "unknown";
    return ((ci.flags & 0x1) != 0) ? "SHOWING" : "hidden";
  }
}
"@

# WHERE TO CLICK, and it is not a detail.
#
#    An EMBEDDED player is a `WS_CHILD` with no main window of its own, so the
#    middle of the screen is the middle of the viewport hole and clicking there
#    hands it the keyboard (mouse messages are routed by hit-test, key messages
#    by focus -- the whole of the FIX1 finding).
#
#    A NEW-WINDOW player is a separate top-level window that does NOT cover the
#    screen. Clicking the screen's centre there lands on the maximized EDITOR
#    behind it, which takes the foreground back and leaves the game unfocused --
#    measured, and it is why this wave's first new-window run reported 0.000 m
#    with the editor's own Outliner showing `Selected 1`. So the click goes to
#    the player's own rectangle when it has one.
$screen = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
$target = New-Object InfInput+RECT
#    "Has the player a window of its own" is THREE questions, not one, and the
#    first version asked only the first. An EMBEDDED player still reports a
#    `MainWindowHandle` -- a 16x16 stub at (0,0), which is what winit leaves
#    behind once the editor has reparented the real one -- so a size test and a
#    parent test go with it. Without them the driver took the new-window branch
#    on an embedded session, raised a 16-pixel window, sent W to nothing and
#    reported 0.000 m.
$hasWindow = $false
$player.Refresh()
if ($player.MainWindowHandle -ne [IntPtr]::Zero) {
    $gotRect = [InfInput]::GetWindowRect($player.MainWindowHandle, [ref]$target)
    $isTop = [InfInput]::GetParent($player.MainWindowHandle) -eq [IntPtr]::Zero
    $wide = ($target.Right - $target.Left) -ge 200 -and ($target.Bottom - $target.Top) -ge 200
    $hasWindow = $gotRect -and $isTop -and $wide
    if (-not $hasWindow) {
        Say ("the player's reported window is {0}x{1}, top-level={2} — treating it as embedded" -f ($target.Right - $target.Left), ($target.Bottom - $target.Top), $isTop)
    }
}
Say ("foreground before the click: " + [InfInput]::Foreground() + " (editor pid $($proc.Id), player pid $($player.Id))")
if ($hasWindow) {
    $cx = [int](($target.Left + $target.Right) / 2)
    $cy = [int](($target.Top + $target.Bottom) / 2)
    # **NO CLICK.** A session with its own window already holds the keyboard --
    # `take_keyboard_focus` takes it when the window is created and the line
    # above says so -- and a synthetic click into it is not a no-op: measured
    # over eight runs, five of them handed the foreground to the EDITOR on the
    # click and the hero then moved 0.000 m, while the three that kept it moved
    # 12.1-12.8 m. The click exists to give an EMBEDDED player the focus its
    # reparented child window is denied; a top-level one needs raising, not
    # clicking.
    Say ("the player owns its own window [{0},{1} {2}x{3}]; raising it rather than clicking into it" -f $target.Left, $target.Top, ($target.Right - $target.Left), ($target.Bottom - $target.Top))
    [InfInput]::ShowWindow($player.MainWindowHandle, 5) | Out-Null   # SW_SHOW
    [InfInput]::SetForegroundWindow($player.MainWindowHandle) | Out-Null
    [InfInput]::SetCursorPos($cx, $cy) | Out-Null
    Start-Sleep -Milliseconds 400
} else {
    Say "clicking the viewport hole at the screen centre (the player has no window of its own)"
    [InfInput]::Click([int]($screen.Width / 2), [int]($screen.Height / 2))
}
Start-Sleep -Milliseconds 800

Say ("foreground after the click:  " + [InfInput]::Foreground())
Say ("cursor while the game has the window: " + [InfInput]::CursorState())

# **FOUR NAMED FRAMES, not two anonymous ones** (wave CHAR1a.2). A wave that is
# asked whether the idle looks right cannot answer with a picture of a run. The
# order is idle → walk → run because it is also the order the locomotion machine
# transitions in, so a bad transition shows as a frame that does not match its
# name.
Say "IDLE: no input for 2 s"
Start-Sleep -Seconds 2
& powershell -NoProfile -ExecutionPolicy Bypass -File $shot -Out (Join-Path $OutDir "04-pie-idle.png") | ForEach-Object { Say $_ }

Say "WALK: W tapped in 120 ms bursts, so the machine stays under the run threshold"
for ($i = 0; $i -lt 6; $i++) {
    [InfInput]::Down(0x11); Start-Sleep -Milliseconds 120; [InfInput]::Up(0x11)
    Start-Sleep -Milliseconds 260
}
[InfInput]::Down(0x11)
Start-Sleep -Milliseconds 350
& powershell -NoProfile -ExecutionPolicy Bypass -File $shot -Out (Join-Path $OutDir "05-pie-walk.png") | ForEach-Object { Say $_ }
[InfInput]::Up(0x11)

Say "holding W"
[InfInput]::Down(0x11)   # scancode: W
Start-Sleep -Milliseconds 900
& powershell -NoProfile -ExecutionPolicy Bypass -File $shot -Out (Join-Path $OutDir "02-pie-a.png") | ForEach-Object { Say $_ }
Start-Sleep -Seconds 2
& powershell -NoProfile -ExecutionPolicy Bypass -File $shot -Out (Join-Path $OutDir "03-pie-b.png") | ForEach-Object { Say $_ }
& powershell -NoProfile -ExecutionPolicy Bypass -File $shot -Out (Join-Path $OutDir "06-pie-run.png") | ForEach-Object { Say $_ }
Say ("cursor two seconds in: " + [InfInput]::CursorState())
[InfInput]::Up(0x11)
Say "released W"

# One more after the release: the street the hero ran into, where the crowd is,
# which is where a second body shows if the level offers one.
Start-Sleep -Seconds 2
& powershell -NoProfile -ExecutionPolicy Bypass -File $shot -Out (Join-Path $OutDir "07-pie-street.png") | ForEach-Object { Say $_ }

# ── 6. what the hero did, in metres ──────────────────────────────────────────
if (Test-Path $heroCsv) {
    $rows = Get-Content $heroCsv | Where-Object { $_ -match "^[0-9]" }
    if ($rows.Count -ge 2) {
        $a = $rows[0].Split(","); $b = $rows[-1].Split(",")
        $dx = [double]$b[2] - [double]$a[2]
        $dz = [double]$b[4] - [double]$a[4]
        $d = [math]::Sqrt($dx * $dx + $dz * $dz)
        Say ("hero first : t={0} ({1}, {2}, {3}) {4} speed {5}" -f $a[0], $a[2], $a[3], $a[4], $a[5], $a[6])
        Say ("hero last  : t={0} ({1}, {2}, {3}) {4} speed {5}" -f $b[0], $b[2], $b[3], $b[4], $b[5], $b[6])
        Say ("HERO MOVED {0:N3} m over {1} samples" -f $d, $rows.Count)
        if ($d -lt $MinMetres) {
            Say ("PLAY DID NOT PLAY: the hero moved {0:N3} m against a {1:N1} m floor" -f $d, $MinMetres)
            $failed = $true
        }
    } else {
        Say "hero.csv has $($rows.Count) row(s) — the player wrote no positions"
        $failed = $true
    }
    # What the player said about the keyboard, echoed where the number is, so a
    # session that moved is read beside the reason it could.
    foreach ($line in (Get-Content $heroCsv | Where-Object { $_ -match "^# keyboard focus" })) {
        Say ("player: " + $line.Substring(2))
    }
} else {
    Say "no hero.csv at $heroCsv"
    $failed = $true
}

Say ("windows now: " + ((Get-Process | Where-Object { $_.MainWindowTitle -ne "" -and ($_.ProcessName -like "inf*") } |
    ForEach-Object { "$($_.ProcessName)[$($_.Id)] '$($_.MainWindowTitle)'" }) -join " | "))

# ── 7. close ─────────────────────────────────────────────────────────────────
if ($KeepOpen) {
    Say "left running (pid $($proc.Id)); the island's pack stays mapped until you close it"
} else {
    Say "closing"
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
    Get-Process -Name "inf-player" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 1
    # The control for the two readings above: a cursor that is hidden here as
    # well is a cursor this script cannot see, not one the game took.
    Say ("cursor after the session ended: " + [InfInput]::CursorState())
    Say ("still running: " + $(if (Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.ProcessName -in @("inf-studio", "inf-player") }) { "YES" } else { "none" }))
}
if ($failed) {
    # **Non-zero, and that is the point** (audit FIX1). A demo loop that always
    # exits 0 is a screenshot service. This one is the last gate before a wave
    # is called done, so it fails the way a gate fails.
    Say "done (FAILED) — $OutDir"
    exit 7
}
Say "done — $OutDir"
