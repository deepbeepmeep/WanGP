# =====================================================================
#  ____   ___  _   _ _____ _____   ____   ___  _   _ _____ _____
# / ___| / _ \| \ | | ____|_   _| |  _ \ / _ \| \ | | ____|_   _|
# \___ \| | | |  \| |  _|   | |   | |_) | | | |  \| |  _|   | |
#  ___) | |_| | |\  | |___  | |   |  _ <| |_| | |\  | |___  | |
# |____/ \___/|_| \_|_____| |_|   |_| \_\\___/|_| \_|_____| |_|
#
#  Wan2GP - LOCAL ONLY SETUP
#  - Keine HuggingFace-Downloads
#  - Lokale Modellpfade aus docs/docs/models.txt
#  - Nur LTX / Flux / Minimax / Moss sichtbar
# =====================================================================

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "   WAN2GP LOCAL-ONLY SETUP" -ForegroundColor Green
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host ""

# ---------------------------------------------------------------------
# 0) Repo-Root automatisch erkennen
# ---------------------------------------------------------------------
$root = (Get-Location).Path

if (-not (Test-Path (Join-Path $root "wgp.py"))) {
    $candidate = Join-Path $root "Wan2GP"
    if (Test-Path (Join-Path $candidate "wgp.py")) {
        $root = $candidate
    }
}

if (-not (Test-Path (Join-Path $root "wgp.py"))) {
    throw "wgp.py nicht gefunden. Bitte im Wan2GP-Repo-Ordner ausfuehren."
}

Write-Host "[OK] Repo-Root erkannt:" -ForegroundColor Green
Write-Host "     $root" -ForegroundColor White
Write-Host ""

Set-Location $root

# ---------------------------------------------------------------------
# 1) files_locator.py patchen
# ---------------------------------------------------------------------
$fl = Join-Path $root "shared\utils\files_locator.py"

if (-not (Test-Path $fl)) {
    throw "Datei nicht gefunden: $fl"
}

$flSrc = Get-Content $fl -Raw

if ($null -eq $flSrc) {
    throw "Konnte files_locator.py nicht lesen: $fl"
}

if ($flSrc -notmatch "def is_local_only") {

    $anchor = 'default_checkpoints_paths = ["ckpts", "."]'

    if (-not $flSrc.Contains($anchor)) {
        throw "Anker in files_locator.py nicht gefunden. Moeglicherweise hat sich das Repo geaendert."
    }

    $block = @'
default_checkpoints_paths = ["ckpts", "."]

# --- LOCAL-ONLY MODE (no downloads, only local models) ---
import re as _re
_local_only = os.environ.get("WGP_LOCAL_ONLY", "").strip().lower() in ("1", "true", "yes")

def set_local_only(v):
    global _local_only
    _local_only = bool(v)

def is_local_only():
    return _local_only

def load_local_paths_from_file(paths_file="models.txt"):
    """Extract local model root folders from a models.txt file."""
    roots = []
    if not os.path.isfile(paths_file):
        return roots
    with open(paths_file, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            m = _re.match(
                r"^([A-Za-z]:[\\/].+?[\\/](?:checkpoints|diffusion_models|loras|vae|text_encoders|LLM|unet|model_patches|controlnet|clip|vae_approx))[\\/]",
                line,
                _re.I
            )
            if m and m.group(1) not in roots:
                roots.append(m.group(1))
    return roots
# ----------------------------------------------------------
'@

    $flSrc = $flSrc.Replace($anchor, $block.TrimEnd())
    Set-Content -Path $fl -Value $flSrc -Encoding UTF8 -NoNewline

    Write-Host "[OK] files_locator.py gepatcht" -ForegroundColor Green
}
else {
    Write-Host "[SKIP] files_locator.py ist bereits gepatcht" -ForegroundColor Yellow
}

Write-Host ""

# ---------------------------------------------------------------------
# 2) wgp.py patchen
# ---------------------------------------------------------------------
$wgp = Join-Path $root "wgp.py"

if (-not (Test-Path $wgp)) {
    throw "Datei nicht gefunden: $wgp"
}

$wgpSrc = Get-Content $wgp -Raw

if ($null -eq $wgpSrc) {
    throw "Konnte wgp.py nicht lesen: $wgp"
}

# 2a) Lokale Modellpfade aus models.txt einbinden
if ($wgpSrc -notmatch "LOCAL-ONLY MODE: add roots") {

    $oldPaths = @'
checkpoints_paths = server_config.get("checkpoints_paths", None)
if checkpoints_paths is None: checkpoints_paths = server_config["checkpoints_paths"] = fl.default_checkpoints_paths
fl.set_checkpoints_paths(checkpoints_paths)
'@

    $newPaths = @'
checkpoints_paths = server_config.get("checkpoints_paths", None)
if checkpoints_paths is None: checkpoints_paths = server_config["checkpoints_paths"] = fl.default_checkpoints_paths

# --- LOCAL-ONLY MODE: add roots parsed from docs/docs/models.txt ---
if server_config.get("use_models_txt", True):
    _wgp_root = os.path.dirname(os.path.abspath(__file__))
    extra_roots = fl.load_local_paths_from_file(os.path.join(_wgp_root, "docs", "docs", "models.txt"))
    expanded = [r for r in extra_roots if os.path.isdir(r)]
    if len(expanded) > 0:
        checkpoints_paths = expanded + list(checkpoints_paths)
        server_config["checkpoints_paths"] = checkpoints_paths

if server_config.get("local_only", False):
    fl.set_local_only(True)
# -------------------------------------------------------------------

fl.set_checkpoints_paths(checkpoints_paths)
'@

    if (-not $wgpSrc.Contains($oldPaths)) {
        throw "Anker fuer checkpoints_paths in wgp.py nicht gefunden."
    }

    $wgpSrc = $wgpSrc.Replace($oldPaths, $newPaths)
    Write-Host "[OK] wgp.py: lokale Modellpfade aktiviert" -ForegroundColor Green
}
else {
    Write-Host "[SKIP] wgp.py: Modellpfad-Patch vorhanden" -ForegroundColor Yellow
}

# 2b) Downloads in process_files_def blockieren
if ($wgpSrc -notmatch "\[LOCAL-ONLY\] Downloads deaktiviert") {

    $oldPFD = @'
def process_files_def(repoId = None, sourceFolderList = None, fileList = None, targetFolderList = None):
    if targetFolderList is None:
'@

    $newPFD = @'
def process_files_def(repoId = None, sourceFolderList = None, fileList = None, targetFolderList = None):
    if fl.is_local_only():
        print(f"[LOCAL-ONLY] Downloads deaktiviert - ueberspringe HF-Repo '{repoId}'. Nur lokale Modelle werden verwendet.")
        return
    if targetFolderList is None:
'@

    if (-not $wgpSrc.Contains($oldPFD)) {
        throw "Anker fuer process_files_def in wgp.py nicht gefunden."
    }

    $wgpSrc = $wgpSrc.Replace($oldPFD, $newPFD)
    Write-Host "[OK] wgp.py: process_files_def-Downloads blockiert" -ForegroundColor Green
}
else {
    Write-Host "[SKIP] process_files_def-Patch vorhanden" -ForegroundColor Yellow
}

# 2c) Downloads in download_file blockieren
if ($wgpSrc -notmatch "\[LOCAL-ONLY\] Download deaktiviert") {

    $oldDL = @'
def download_file(url,filename):
    url = url.split("|")[0]
'@

    $newDL = @'
def download_file(url,filename):
    if fl.is_local_only():
        raise Exception(f"[LOCAL-ONLY] Download deaktiviert fuer '{url}'. Bitte lege die Datei manuell in deinen lokalen Modellordnern ab.")
    url = url.split("|")[0]
'@

    if (-not $wgpSrc.Contains($oldDL)) {
        throw "Anker fuer download_file in wgp.py nicht gefunden."
    }

    $wgpSrc = $wgpSrc.Replace($oldDL, $newDL)
    Write-Host "[OK] wgp.py: download_file blockiert" -ForegroundColor Green
}
else {
    Write-Host "[SKIP] download_file-Patch vorhanden" -ForegroundColor Yellow
}

Set-Content -Path $wgp -Value $wgpSrc -Encoding UTF8 -NoNewline

Write-Host ""

# ---------------------------------------------------------------------
# 3) Nicht gewuenschte Modelle ausblenden
# ---------------------------------------------------------------------
$finetunes = Join-Path $root "finetunes"
$defaults  = Join-Path $root "defaults"

New-Item -ItemType Directory -Force -Path $finetunes | Out-Null

$hideJson = "{`n  `"model`": { `"visible`": false }`n}`n"

Get-ChildItem -Path $finetunes -Filter "*.json" -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -notlike "_backup_*" } |
    Remove-Item -Force

$keepRegex = '^(ltx|flux|pi_flux2|minimax|moss)'
$hidden = 0

if (Test-Path $defaults) {

    Get-ChildItem -Path $defaults -Filter "*.json" | ForEach-Object {

        $name = $_.BaseName

        if ($name -notmatch $keepRegex) {

            $bak = Join-Path $finetunes "_backup_$name.json"

            if (-not (Test-Path $bak)) {
                Copy-Item $_.FullName $bak
            }

            Set-Content -Path (Join-Path $finetunes "$name.json") -Value $hideJson -Encoding UTF8 -NoNewline
            $hidden++
        }
    }

    Write-Host "[OK] $hidden Modelle ausgeblendet" -ForegroundColor Green
    Write-Host "     Sichtbar bleiben: ltx*, flux*, pi_flux2*, minimax*, moss*" -ForegroundColor Cyan
}
else {
    Write-Host "[WARN] defaults-Ordner nicht gefunden: $defaults" -ForegroundColor Yellow
}

Write-Host ""

# ---------------------------------------------------------------------
# 4) wgp_config.json setzen
# ---------------------------------------------------------------------
$cfgFile = Join-Path $root "wgp_config.json"

if (Test-Path $cfgFile) {
    $cfg = Get-Content $cfgFile -Raw | ConvertFrom-Json
}
else {
    $cfg = [pscustomobject]@{}
}

$cfg | Add-Member -NotePropertyName local_only   -NotePropertyValue $true  -Force
$cfg | Add-Member -NotePropertyName use_models_txt -NotePropertyValue $true -Force

$cfg | ConvertTo-Json -Depth 10 | Set-Content -Path $cfgFile -Encoding UTF8

Write-Host "[OK] wgp_config.json gesetzt:" -ForegroundColor Green
Write-Host "     local_only=true" -ForegroundColor White
Write-Host "     use_models_txt=true" -ForegroundColor White
Write-Host ""

# ---------------------------------------------------------------------
# 5) Syntax-Check
# ---------------------------------------------------------------------
Write-Host "[..] Python-Syntax-Check laeuft ..." -ForegroundColor Cyan

python -m py_compile $wgp $fl

if ($LASTEXITCODE -ne 0) {
    throw "Python-Syntaxfehler nach Patch."
}

Write-Host "[OK] Python-Syntax-Check bestanden" -ForegroundColor Green
Write-Host ""

Write-Host "=====================================================" -ForegroundColor Green
Write-Host " FERTIG - LOCAL-ONLY MODE AKTIV" -ForegroundColor Green
Write-Host "=====================================================" -ForegroundColor Green
Write-Host ""
Write-Host " Start:" -ForegroundColor Cyan
Write-Host "   python wgp.py --listen" -ForegroundColor White
Write-Host ""
Write-Host " Extra-Schutz ueber Umgebungsvariable:" -ForegroundColor Cyan
Write-Host "   `$env:WGP_LOCAL_ONLY='1'; python wgp.py" -ForegroundColor White
Write-Host ""
Write-Host " Rueckgaengig:" -ForegroundColor DarkGray
Write-Host "   - finetunes\_backup_*.json wiederherstellen" -ForegroundColor DarkGray
Write-Host "   - wgp_config.json: local_only=false" -ForegroundColor DarkGray
Write-Host "   - git checkout -- wgp.py shared/utils/files_locator.py" -ForegroundColor DarkGray
Write-Host ""