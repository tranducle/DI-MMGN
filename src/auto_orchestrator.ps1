# auto_orchestrator.ps1 — Tự động launch experiments khi dependencies xong.
# Chạy ngầm (persistent), poll mỗi 5 phút.
#
# Logic:
#   1. Đợi CONCAT (retrain_concat.py) xong
#   2. Launch eval_retention.py (Gap 2)
#   3. Đợi retention xong
#   4. Launch train_fusion_ablation.py (Gap 3)
#   5. Đợi fusion xong
#   6. Đợi A2 (train_snapshot_mm.py) xong
#   7. Log tất cả results

$srcDir = "C:\Users\Tran Duc Le\Documents\DO_A_PAPER_Jul_30_Skill_Update\DO_A_PAPER_Jul_30_Skill_Update\Papers\TRANG-PAPER-1\src"
$logFile = "C:\Users\Tran Duc Le\Documents\DO_A_PAPER_Jul_30_Skill_Update\DO_A_PAPER_Jul_30_Skill_Update\Papers\TRANG-PAPER-1\8_Project_Management\auto_orchestrator.log"
$checkInterval = 300  # 5 minutes

function Log-Msg {
    param([string]$Msg)
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Msg"
    Write-Host $line
    Add-Content -Path $logFile -Value $line -ErrorAction SilentlyContinue
}

function Is-Running {
    param([string]$ScriptName)
    $procs = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like "*$ScriptName*" -and $_.CommandLine -notlike "*Get-CimInstance*" -and $_.CommandLine -notlike "*auto_orchestrator*" }
    return [bool]$procs
}

function Is-Done {
    param([string]$ResultsPath, [string]$DoneKey = "done")
    if (-not (Test-Path $ResultsPath)) { return $false }
    $content = Get-Content $ResultsPath -Raw -ErrorAction SilentlyContinue
    return ($content -match '"done"\s*:\s*true')
}

function Launch-Experiment {
    param([string]$ScriptName, [string]$LogFile)
    Log-Msg "LAUNCHING: $ScriptName"
    Push-Location $srcDir
    $logPath = Join-Path "..\8_Project_Management" $LogFile
    # Use Start-Process to run independently
    $proc = Start-Process -FilePath "powershell" `
        -ArgumentList "-NoProfile -Command `"Set-Location '$srcDir'; python -u $ScriptName 2>&1 | Tee-Object -FilePath '$logPath'`" `
        -WindowStyle Hidden -PassThru
    Pop-Location
    Log-Msg "STARTED: $ScriptName (PID=$($proc.Id))"
    return $proc.Id
}

Log-Msg "========================================"
Log-Msg "AUTO ORCHESTRATOR STARTED"
Log-Msg "Monitoring: CONCAT → Retention → Fusion"
Log-Msg "Check interval: ${checkInterval}s"
Log-Msg "========================================"

# ── Phase 1: Wait for CONCAT to finish ──
Log-Msg "PHASE 1: Waiting for CONCAT (retrain_concat.py)..."
while (-not (Is-Done "..\8_Project_Management\phase5b_concat_results_raw.json")) {
    if (Is-Running "retrain_concat") {
        Start-Sleep -Seconds $checkInterval
    } else {
        # Process not running and not done — check if crashed
        Start-Sleep -Seconds 30
        if (-not (Is-Running "retrain_concat") -and -not (Is-Done "..\8_Project_Management\phase5b_concat_results_raw.json")) {
            Log-Msg "WARNING: CONCAT not running and not done — may have crashed. Waiting for supervisor restart..."
            Start-Sleep -Seconds 60
        }
    }
}
Log-Msg "PHASE 1 DONE: CONCAT completed!"

# Read CONCAT best result
$concatResults = Get-Content "..\8_Project_Management\phase5b_concat_results_raw.json" -Raw | ConvertFrom-Json
if ($concatResults.summary.none) {
    Log-Msg "CONCAT none mean: $($concatResults.summary.none.mean)"
} elseif ($concatResults.summary.soft) {
    Log-Msg "CONCAT soft mean: $($concatResults.summary.soft.mean)"
}

# ── Phase 2: Launch Retention (Gap 2) ──
Log-Msg "PHASE 2: Launching Retention evaluation..."
$retentionPid = Launch-Experiment "eval_retention.py" "phase5d_retention_train.log"

# Wait for retention to finish
Log-Msg "Waiting for Retention to complete..."
while (-not (Is-Done "..\8_Project_Management\phase5d_retention_results_raw.json")) {
    Start-Sleep -Seconds $checkInterval
}
Log-Msg "PHASE 2 DONE: Retention completed!"

# Read retention result
$retentionResults = Get-Content "..\8_Project_Management\phase5d_retention_results_raw.json" -Raw -ErrorAction SilentlyContinue
if ($retentionResults) {
    $ret = $retentionResults | ConvertFrom-Json
    Log-Msg "Retention: F1=$($ret.final_f1) gate=$($ret.gate_pass)"
}

# ── Phase 3: Launch Fusion Ablation (Gap 3) ──
Log-Msg "PHASE 3: Launching Fusion ablation..."
$fusionPid = Launch-Experiment "train_fusion_ablation.py" "phase5e_train.log"

# Wait for fusion to finish
Log-Msg "Waiting for Fusion ablation to complete..."
while (-not (Is-Done "..\8_Project_Management\phase5e_fusion_results_raw.json")) {
    Start-Sleep -Seconds $checkInterval
}
Log-Msg "PHASE 3 DONE: Fusion ablation completed!"

# ── Phase 4: Check A2 (if still running) ──
if (Is-Running "train_snapshot_mm") {
    Log-Msg "PHASE 4: Waiting for A2 Snapshot-MM..."
    while (Is-Running "train_snapshot_mm") {
        Start-Sleep -Seconds $checkInterval
    }
}
Log-Msg "ALL EXPERIMENTS COMPLETE!"

# ── Summary ──
Log-Msg "========================================"
Log-Msg "ALL EXPERIMENTS COMPLETE"
Log-Msg "Results:"
Log-Msg "  CONCAT:     ..\8_Project_Management\phase5b_concat_results_raw.json"
Log-Msg "  Retention:  ..\8_Project_Management\phase5d_retention_results_raw.json"
Log-Msg "  Fusion:     ..\8_Project_Management\phase5e_fusion_results_raw.json"
Log-Msg "  A2 SnapMM:  ..\8_Project_Management\phase5c_snapshot_mm_results_raw.json"
Log-Msg "========================================"
Log-Msg "Next step: patch all results into paper.tex → recompile → postflight"
