# auto_orchestrator_v2.ps1 — Simplified, no Start-Process quoting issues.
# Runs retention then fusion SEQUENTIALLY after CONCAT completes.
$srcDir = "C:\Users\Tran Duc Le\Documents\DO_A_PAPER_Jul_30_Skill_Update\DO_A_PAPER_Jul_30_Skill_Update\Papers\TRANG-PAPER-1\src"
$logFile = "C:\Users\Tran Duc Le\Documents\DO_A_PAPER_Jul_30_Skill_Update\DO_A_PAPER_Jul_30_Skill_Update\Papers\TRANG-PAPER-1\8_Project_Management\auto_orchestrator.log"

function Log($msg) {
    $line = "[$(Get-Date -Format 'HH:mm:ss')] $msg"
    Write-Host $line
    Add-Content $logFile $line
}

Log "ORCHESTRATOR V2 STARTED"

# ── Phase 1: Wait for CONCAT ──
Log "Waiting for CONCAT done:true..."
$concatResults = Join-Path $srcDir "..\8_Project_Management\phase5b_concat_results_raw.json"
while ($true) {
    if ((Test-Path $concatResults) -and ((Get-Content $concatResults -Raw) -match '"done":\s*true')) { break }
    Start-Sleep 300
}
Log "CONCAT DONE!"

# ── Phase 2: Run Retention ──
Log "Launching eval_retention.py..."
Set-Location $srcDir
python -u eval_retention.py 2>&1 | Tee-Object -FilePath "..\8_Project_Management\phase5d_retention_train.log"
Log "RETENTION DONE (exit $LASTEXITCODE)"

# ── Phase 3: Run Fusion Ablation ──
Log "Launching train_fusion_ablation.py..."
python -u train_fusion_ablation.py 2>&1 | Tee-Object -FilePath "..\8_Project_Management\phase5e_train.log"
Log "FUSION DONE (exit $LASTEXITCODE)"

Log "ALL AUTO EXPERIMENTS COMPLETE"
