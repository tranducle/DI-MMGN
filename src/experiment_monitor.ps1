# experiment_monitor.ps1 — Auto-check + restart for all training experiments.
# Runs as a persistent background process. Polls every 5 minutes.
# Logs to experiment_monitor.log.

$srcDir = Join-Path $PSScriptRoot "" | Split-Path | Split-Path | Join-Path -ChildPath "src"
# Fallback: hardcode relative to repo root
if (-not (Test-Path $srcDir)) { $srcDir = "Papers\TRANG-PAPER-1\src" }
$logDir = "Papers\TRANG-PAPER-1\8_Project_Management"
$monitorLog = Join-Path $logDir "experiment_monitor.log"

$experiments = @(
    @{ Name="phase4b";  Script="rerun_stochastic.py";     Results="$logDir\phase4b_results_raw.json" }
    @{ Name="phase4d";  Script="ablation_textonly.py";    Results="$logDir\phase4d_textonly_raw.json" }
    @{ Name="phase5a1"; Script="train_text_baseline.py";  Results="$logDir\phase5_text_baselines_raw.json" }
)

function Write-Monitor {
    param([string]$Msg)
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Msg"
    Write-Host $line
    Add-Content -Path $monitorLog -Value $line -ErrorAction SilentlyContinue
}

function Test-Done {
    param([string]$ResultsPath)
    if (-not (Test-Path $ResultsPath)) { return $false }
    $content = Get-Content $ResultsPath -Raw -ErrorAction SilentlyContinue
    return ($content -match '"done"\s*:\s*true')
}

function Find-Process {
    param([string]$ScriptName)
    $procs = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like "*$ScriptName*" -and $_.CommandLine -notlike "*Get-CimInstance*" -and $_.CommandLine -notlike "*experiment_monitor*" }
    return $procs
}

function Restart-Experiment {
    param([string]$ScriptName)
    $logFile = Join-Path $logDir "$($ScriptName -replace '\.py','')_train.log"
    # If no specific log, use a generic one
    switch ($ScriptName) {
        "rerun_stochastic.py"    { $logFile = "$logDir\phase4b_train.log" }
        "ablation_textonly.py"   { $logFile = "$logDir\phase4d_train.log" }
        "train_text_baseline.py" { $logFile = "$logDir\phase5_train.log" }
    }
    Set-Location $srcDir
    $proc = Start-Process -FilePath "python" -ArgumentList "-u", $ScriptName -NoNewWindow -PassThru `
        -RedirectStandardError "$logFile.err" -RedirectStandardOutput "$logFile.out" 2>$null
    # Actually use the Tee pattern via cmd
    if (-not $proc) {
        # Fallback: simple Start-Process
        $proc = Start-Process -FilePath "python" -ArgumentList "-u $ScriptName" -NoNewWindow -PassThru 2>$null
    }
    return $proc
}

Write-Monitor "=== Experiment monitor started ==="
Write-Monitor "Monitoring: $($experiments.Name -join ', ')"
Write-Monitor "Poll interval: 300s (5 min)"

while ($true) {
    foreach ($exp in $experiments) {
        $name = $exp.Name
        $script = $exp.Script
        $results = $exp.Results

        # Skip if done
        if (Test-Done $results) {
            continue  # silently skip completed experiments
        }

        # Check if process is alive
        $procs = Find-Process $script
        if ($procs) {
            # Process alive — check for NaN/errors in recent log
            $logFile = switch ($script) {
                "rerun_stochastic.py"    { "$logDir\phase4b_train.log" }
                "ablation_textonly.py"   { "$logDir\phase4d_train.log" }
                "train_text_baseline.py" { "$logDir\phase5_train.log" }
            }
            if (Test-Path $logFile) {
                $tail = Get-Content $logFile -Tail 20 -ErrorAction SilentlyContinue
                $hasError = $tail | Where-Object { $_ -match 'Traceback|CUDA error|RuntimeError|nan loss|OutOfMemory' }
                if ($hasError) {
                    Write-Monitor "WARNING: $name — error detected in log:"
                    $hasError | ForEach-Object { Write-Monitor "  $_" }
                    # Kill and restart
                    $procs | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
                    Write-Monitor "ACTION: killed $name, restarting in 30s..."
                    Start-Sleep -Seconds 30
                    Restart-Experiment $script | Out-Null
                    Write-Monitor "ACTION: $name restarted"
                }
            }
        } else {
            # Process not found — check if it should be running
            $logFile = switch ($script) {
                "rerun_stochastic.py"    { "$logDir\phase4b_train.log" }
                "ablation_textonly.py"   { "$logDir\phase4d_train.log" }
                "train_text_baseline.py" { "$logDir\phase5_train.log" }
            }
            $logAge = 999
            if (Test-Path $logFile) {
                $logAge = ((Get-Date) - (Get-Item $logFile).LastWriteTime).TotalMinutes
            }
            if ($logAge -gt 5) {
                Write-Monitor "ALERT: $name — process not found, log $([math]::Round($logAge,1))min old — RESTARTING"
                Restart-Experiment $script | Out-Null
                Write-Monitor "ACTION: $name restarted"
            }
        }
    }
    Start-Sleep -Seconds 300
}
