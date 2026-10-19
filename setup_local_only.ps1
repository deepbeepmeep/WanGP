# =====================================================================
# setup_local_only.ps1  -  Wan2GP auf NUR lokale Modelle umstellen
# (LTX, Flux, Minimax, Moss - alle Wan 2.1 / 2.2 / 2.x ausgeblendet)
#
# Ausfuehren in PowerShell:
#   cd D:\claudeAgent\WanGP
#   Set-ExecutionPolicy -Scope Process Bypass -Force
#   .\setup_local_only.ps1
# =====================================================================

$ErrorActionPreference = "Stop"
$root = "D:\claudeAgent\WanGP"
if (-not (Test-Path "$root\wgp.py")) { throw "Wan2GP nicht gefunden unter $root" }
Set-Location $root

# ---------------------------------------------------------------------
# 1) Patch: shared/utils/files_locator.py  (Local-Only-Modus + models.txt-Pfade)
# ---------------------------------------------------------------------
$fl = "$root\shared\utils\files_locator.py"
$flSrc = Get-Content $fl -Raw

if ($flSrc -notmatch "def is_local_only") {
    $anchor = 'default_checkpoints_paths = ["ckpts", "."]'
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
    """Extract model root folders (e.g. D:\ComfyUiCuda\models\checkpoints) from a models.txt file."""
    roots = []
    if not os.path.isfile(paths_file):
        return roots
    with open(paths_file, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            m = _re.match(r"^([A-Za-z]:[\\/].+?[\\/](?:checkpoints|diffusion_models|loras|vae|text_encoders|LLM|unet|model_patches|controlnet|clip|vae_approx))[\\/]", line, _re.I)
            if m and m.group(1) not in roots:
                roots.append(m.group(1))
    return roots
# ----------------------------------------------------------
'@
    $flSrc = $flSrc.Replace($anchor, $block.TrimEnd())
    Set-Content $fl $flSrc -Encoding UTF8 -NoNewline
    Write-Host "[OK] files_locator.py gepatcht (LOCAL-ONLY + models.txt-Parser)" -ForegroundColor Green
} else {
    Write-Host "[SKIP] files_locator.py bereits gepatcht" -ForegroundColor Yellow
}

# ---------------------------------------------------------------------
# 2) Patch: wgp.py  (checkpoints_paths aus docs/docs/models.txt + local_only)
# ---------------------------------------------------------------------
$wgp = "$root\wgp.py"
$wgpSrc = Get-Content $wgp -Raw

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

if ($wgpSrc -notmatch "LOCAL-ONLY MODE: add roots") {
    if ($wgpSrc.Contains($oldPaths)) {
        $wgpSrc = $wgpSrc.Replace($oldPaths, $newPaths)
        Write-Host "[OK] wgp.py: Modellpfade aus models.txt werden eingebunden" -ForegroundColor Green
    } else { throw "wgp.py: Anker fuer checkpoints_paths nicht gefunden (Update?)" }
} else { Write-Host "[SKIP] wgp.py Pfad-Patch vorhanden" -ForegroundColor Yellow }

# Downloads blockieren (process_files_def)
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
if ($wgpSrc -notmatch "\[LOCAL-ONLY\] Downloads deaktiviert") {
    if ($wgpSrc.Contains($oldPFD)) { $wgpSrc = $wgpSrc.Replace($oldPFD, $newPFD); Write-Host "[OK] wgp.py: process_files_def-Downloads blockiert" -ForegroundColor Green }
    else { throw "wgp.py: Anker fuer process_files_def nicht gefunden" }
} else { Write-Host "[SKIP] process_files_def-Patch vorhanden" -ForegroundColor Yellow }

# Downloads blockieren (download_file)
$oldDL = @'
def download_file(url,filename):
    url = url.split("|")[0]
'@
$newDL = @'
def download_file(url,filename):
    if fl.is_local_only():
        raise Exception(f"[LOCAL-ONLY] Download deaktiviert fuer '{url}'. Bitte lege die Datei manuell in deinen lokalen Modellordern ab (siehe docs/docs/models.txt).")
    url = url.split("|")[0]
'@
if ($wgpSrc -notmatch "\[LOCAL-ONLY\] Download deaktiviert") {
    if ($wgpSrc.Contains($oldDL)) { $wgpSrc = $wgpSrc.Replace($oldDL, $newDL); Write-Host "[OK] wgp.py: download_file blockiert" -ForegroundColor Green }
    else { throw "wgp.py: Anker fuer download_file nicht gefunden" }
} else { Write-Host "[SKIP] download_file-Patch vorhanden" -ForegroundColor Yellow }

Set-Content $wgp $wgpSrc -Encoding UTF8 -NoNewline

# ---------------------------------------------------------------------
# 3) Alle Nicht-LTX/Nicht-Flux/Nicht-Minimax/Nicht-Moss-Modelle ausblenden
#    (finetunes/<name>.json ueberschreibt visible=false im Merge)
# ---------------------------------------------------------------------
New-Item -ItemType Directory -Force "$root\finetunes" | Out-Null
$hideJson = "{`n  `"model`": { `"visible`": false }`n}`n"
# Vorher aufraeumen: alte Hide-Dateien (ohne _backup_-Praefix) loeschen,
# damit ein erneut ausgefuehrtes Skript nichts veraltetes versteckt
Get-ChildItem "$root\finetunes\*.json" -Exclude "_backup_*" | Remove-Item -Force
$keepRegex = '^(ltx|flux|pi_flux2|minimax|moss)'
$hidden = 0
Get-ChildItem "$root\defaults\*.json" | ForEach-Object {
    $name = $_.BaseName
    if ($name -notmatch $keepRegex) {
        # Backup nur beim ersten Mal anlegen
        $bak = "$root\finetunes\_backup_$name.json"
        if (-not (Test-Path $bak)) { Copy-Item $_.FullName $bak }
        Set-Content "$root\finetunes\$name.json" $hideJson -Encoding UTF8 -NoNewline
        $hidden++
    }
}
Write-Host "[OK] $hidden Modelle ausgeblendet (wan*, hunyuan*, qwen*, ace_step*, alpha*, animate* etc.)" -ForegroundColor Green
Write-Host "     Sichtbar bleiben: ltx*, flux*, pi_flux2*, minimax*, moss*" -ForegroundColor Cyan

# ---------------------------------------------------------------------
# 4) wgp_config.json: local_only + use_models_txt setzen
# ---------------------------------------------------------------------
$cfgFile = "$root\wgp_config.json"
if (Test-Path $cfgFile) {
    $cfg = Get-Content $cfgFile -Raw | ConvertFrom-Json
} else {
    $cfg = [pscustomobject]@{}
}
$cfg | Add-Member -NotePropertyName local_only   -NotePropertyValue $true  -Force
$cfg | Add-Member -NotePropertyName use_models_txt -NotePropertyValue $true -Force
$cfg | ConvertTo-Json -Depth 10 | Set-Content $cfgFile -Encoding UTF8
Write-Host "[OK] wgp_config.json: local_only=true, use_models_txt=true" -ForegroundColor Green

# ---------------------------------------------------------------------
# 5) Syntax-Check der gepatchten Dateien
# ---------------------------------------------------------------------
python -m py_compile "$root\wgp.py" "$root\shared\utils\files_locator.py"
if ($LASTEXITCODE -ne 0) { throw "Python-Syntaxfehler nach Patch - Abbruch!" }
Write-Host "[OK] Syntax-Check bestanden" -ForegroundColor Green

Write-Host ""
Write-Host "===============================================================" -ForegroundColor Green
Write-Host " FERTIG! Start mit:" -ForegroundColor Green
Write-Host "   python wgp.py --listen" -ForegroundColor White
Write-Host " oder Umgebungsvariable (zusicherer Schutz):" -ForegroundColor White
Write-Host "   `$env:WGP_LOCAL_ONLY='1'; python wgp.py" -ForegroundColor White
Write-Host "===============================================================" -ForegroundColor Green
Write-Host " Rueckgaengig machen: finetunes\*_backup_*.json loeschen," -ForegroundColor DarkGray
Write-Host " .\setup_local_only.ps1 -Undo nicht verfuegbar - stattdessen" -ForegroundColor DarkGray
Write-Host " wgp_config.json local_only=false setzen und Patches entfernen." -ForegroundColor DarkGray
