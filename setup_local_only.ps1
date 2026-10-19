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
#  Helper: Text BOM-frei als UTF-8 schreiben
#  Hintergrund: "Set-Content -Encoding UTF8" schreibt in PowerShell 5.1
#  ein BOM, und "Get-Content -Raw" liest UTF-8 ohne BOM als ANSI -
#  beides hat wgp.py schon einmal zerlegt (BOM-Crash + Zeichenschaden).
# ---------------------------------------------------------------------
function Write-TextNoBom([string]$Path, [string]$Text) {
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $Text, $utf8NoBom)
}

# ---------------------------------------------------------------------
# 1) files_locator.py patchen
# ---------------------------------------------------------------------
$fl = Join-Path $root "shared\utils\files_locator.py"

if (-not (Test-Path $fl)) {
    throw "Datei nicht gefunden: $fl"
}

$flSrc = Get-Content $fl -Raw -Encoding UTF8

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
    Write-TextNoBom $fl $flSrc

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

$wgpSrc = Get-Content $wgp -Raw -Encoding UTF8

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

Write-TextNoBom $wgp $wgpSrc

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

            Write-TextNoBom (Join-Path $finetunes "$name.json") $hideJson
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

# Schluessel, die wgp.py erwartet - fehlt einer, gibt es einen KeyError
# (attention_mode war der Ausloeser des urspruenglichen Crashes).
$cfgDefaults = [ordered]@{
    attention_mode                        = "auto"
    transformer_types                     = @()
    transformer_quantization              = "int8"
    text_encoder_quantization             = "int8"
    lm_decoder_engine                     = ""
    save_path                             = "outputs"
    image_save_path                       = "outputs"
    audio_save_path                       = "outputs"
    compile                               = ""
    metadata_type                         = "metadata"
    boost                                 = 1
    enable_int8_kernels                   = 1
    clear_file_list                       = 5
    keep_intermediate_sliding_windows     = 1
    enable_4k_resolutions                 = 0
    max_reserved_loras                    = -1
    vae_config                            = 0
    profile                               = 4
    video_profile                         = 4
    image_profile                         = 4
    audio_profile                         = 3.5
    model_hierarchy_type                  = 1
    mmaudio_mode                          = 0
    mmaudio_persistence                   = 1
    rife_version                          = "v4"
    prompt_enhancer_quantization          = "quanto_int8"
    save_queue_if_crash                   = 1
    queue_color_scheme                    = "pastel"
    process_queues_when_browser_unfocused = 1
    local_only                            = $true
    use_models_txt                        = $true
}

if (Test-Path $cfgFile) {
    try {
        $cfgRaw = Get-Content $cfgFile -Raw -Encoding UTF8
        if ([string]::IsNullOrWhiteSpace($cfgRaw)) { throw "Datei ist leer" }
        $cfg = $cfgRaw | ConvertFrom-Json -ErrorAction Stop
        if ($null -eq $cfg) { throw "kein JSON-Objekt" }
        if ($cfg -is [System.Array]) { throw "Top-Level ist eine Liste" }
    }
    catch {
        $bak = "$($cfgFile).corrupt.bak"
        Copy-Item $cfgFile $bak -Force
        Write-Host "[WARN] wgp_config.json unlesbar: $($_.Exception.Message)" -ForegroundColor Yellow
        Write-Host "[WARN] gesichert als: $bak" -ForegroundColor Yellow
        $cfg = [pscustomobject]@{}
    }
}
else {
    $cfg = [pscustomobject]@{}
}

$added = 0
foreach ($k in $cfgDefaults.Keys) {
    if ($null -eq $cfg.PSObject.Properties[$k]) {
        $cfg | Add-Member -NotePropertyName $k -NotePropertyValue $cfgDefaults[$k] -Force
        $added++
    }
}

$cfg.local_only     = $true
$cfg.use_models_txt = $true

Write-TextNoBom $cfgFile ($cfg | ConvertTo-Json -Depth 20)

Write-Host "[OK] wgp_config.json geschrieben (BOM-frei):" -ForegroundColor Green
if ($added -gt 0) {
    Write-Host "     $added fehlende Schluessel ergaenzt" -ForegroundColor White
}
Write-Host "     attention_mode=$($cfg.attention_mode)" -ForegroundColor White
Write-Host "     local_only=true / use_models_txt=true" -ForegroundColor White
Write-Host ""

# ---------------------------------------------------------------------
# 5) Torch/CUDA pruefen: sm_120 (Blackwell / RTX 50x0) unterstuetzt?
# ---------------------------------------------------------------------
$py = Join-Path $root "venv\Scripts\python.exe"

if (-not (Test-Path $py)) {
    Write-Host "[WARN] venv nicht gefunden: $py" -ForegroundColor Yellow
    Write-Host "       Torch-Pruefung uebersprungen." -ForegroundColor Yellow
}
elseif ($env:WGP_SKIP_TORCH_FIX -eq "1") {
    Write-Host "[SKIP] Torch-Pruefung abgeschaltet (WGP_SKIP_TORCH_FIX=1)" -ForegroundColor Yellow
}
else {
    $torchCheck = @(
        'import sys',
        'try:',
        '    import torch',
        'except Exception as exc:',
        '    print("NO_TORCH", exc)',
        '    sys.exit(3)',
        'archs = torch.cuda.get_arch_list()',
        'print("TORCH", torch.__version__)',
        'print("ARCHS", ",".join(archs))',
        'ok = any(a.startswith("sm_120") or a.startswith("compute_120") for a in archs)',
        'print("SM120_OK" if ok else "SM120_MISSING")',
        'sys.exit(0 if ok else 2)'
    )
    $torchCheckFile = Join-Path $env:TEMP "wgp_torch_check.py"
    Write-TextNoBom $torchCheckFile ($torchCheck -join "`n")

    Write-Host "[..] Pruefe Torch-Build auf sm_120 (Blackwell) ..." -ForegroundColor Cyan
    & $py $torchCheckFile
    $torchState = $LASTEXITCODE

    if ($torchState -eq 0) {
        Write-Host "[OK] Torch unterstuetzt sm_120" -ForegroundColor Green
    }
    elseif ($torchState -eq 2) {
        Write-Host "[WARN] Dieser Torch-Build kennt sm_120 nicht (RTX 50x0 / Blackwell)." -ForegroundColor Yellow
        Write-Host "       cu124 reicht dafuer nicht - installiere aus cu128:" -ForegroundColor Yellow
        Write-Host "       pip uninstall -y torch torchvision torchaudio" -ForegroundColor DarkGray
        Write-Host "       pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128" -ForegroundColor DarkGray
        & $py -m pip uninstall -y torch torchvision torchaudio
        & $py -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[WARN] Installation aus cu128 fehlgeschlagen - bitte manuell nachziehen." -ForegroundColor Yellow
        }
        else {
            & $py $torchCheckFile
            if ($LASTEXITCODE -eq 0) {
                Write-Host "[OK] Torch mit sm_120 installiert" -ForegroundColor Green
            }
            else {
                Write-Host "[WARN] sm_120 weiterhin nicht in der Arch-Liste - GPU-Warnung bleibt moeglich." -ForegroundColor Yellow
            }
        }
    }
    else {
        Write-Host "[WARN] Torch-Pruefung nicht moeglich (ExitCode $torchState)." -ForegroundColor Yellow
    }
}

Write-Host ""

# ---------------------------------------------------------------------
# 6) Syntax-Check
# ---------------------------------------------------------------------
Write-Host "[..] Python-Syntax-Check laeuft ..." -ForegroundColor Cyan

$pythonExe = if (Test-Path $py) { $py } else { "python" }
& $pythonExe -m py_compile $wgp $fl

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
