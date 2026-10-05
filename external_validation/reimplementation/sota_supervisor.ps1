$ErrorActionPreference = 'Continue'

$Base = 'C:\Users\Tran Duc Le\Documents\LOCAL-CODING-AGENT-WIN\DO-A-PAPER-v2\Papers\[IEEE ACCESS] - DI-MMGN_REVIEWED_20260923_1357\DI-MMGN-v4\external_validation\reimplementation'
$Project = 'C:\Users\Tran Duc Le\Documents\LOCAL-CODING-AGENT-WIN\DO-A-PAPER-v2\Papers\[IEEE ACCESS] - DI-MMGN_REVIEWED_20260923_1357\DI-MMGN-v4'
$Runner = Join-Path $Project 'src\run_python_logged_v4.py'
$Logs = Join-Path $Base 'logs'
$State = Join-Path $Base 'SOTA_SUPERVISOR_STATE.json'
$Py = 'python'
New-Item -ItemType Directory -Force -Path $Logs | Out-Null

function Set-State([string]$Stage,[string]$Status,[string]$Note) {
    $obj = [ordered]@{
        updated_at = (Get-Date).ToString('o')
        status = $Status
        stage = $Stage
        note = $Note
    }
    $obj | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $State -Encoding UTF8
    Write-Output ("STATE|{0}|{1}|{2}" -f $Stage,$Status,$Note)
}

function Invoke-LoggedPython([string]$Name,[string]$Script,[string[]]$ArgList) {
    $log = Join-Path $Logs ($Name + '.log')
    Write-Output ("STEP_START|{0}|{1}" -f $Name,(Get-Date).ToString('o'))
    & $Py -u $Runner --log $log $Py -u $Script @ArgList
    $rc = $LASTEXITCODE
    Write-Output ("STEP_END|{0}|rc={1}|{2}" -f $Name,$rc,(Get-Date).ToString('o'))
    return [pscustomobject]@{ rc=$rc; log=$log }
}

function Wait-GpuFree([string]$Why) {
    Set-State 'WAIT_GPU' 'WAITING' $Why
    $okCount = 0
    for($i=0; $i -lt 4320; $i++) {
        $busyPy = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
            $_.Name -eq 'python.exe' -and $_.CommandLine -match '(?i)TRAINDNSAGUARD|phase8_runner|run-matrix|train_eval\.py|mlm_pretrain\.py|smoke_cached\.py|pretrain_html\.py'
        }).Count
        $line = (& nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>$null)
        $mem = 99999; $util = 100
        if($line) {
            $parts = $line.Trim().Split(',')
            if($parts.Count -ge 2) {
                [int]::TryParse($parts[0].Trim(), [ref]$mem) | Out-Null
                [int]::TryParse($parts[1].Trim(), [ref]$util) | Out-Null
            }
        }
        # This workstation has ~33-41% baseline GPU utilization from WDDM/UI even when no ML job is active.
        # Treat it as idle only when no heavy Python training command is present and VRAM is at the measured GUI baseline.
        if($busyPy -eq 0 -and $mem -le 3500 -and $util -le 50) { $okCount++ } else { $okCount=0 }
        if(($i % 10) -eq 0) { Write-Output ("GPU_WAIT|busyPy={0}|memMiB={1}|util={2}|stable={3}" -f $busyPy,$mem,$util,$okCount) }
        if($okCount -ge 3) {
            Write-Output ("GPU_READY|memMiB={0}|util={1}" -f $mem,$util)
            return
        }
        Start-Sleep -Seconds 10
    }
    throw "GPU did not become safely available within 12 hours."
}

function Test-OfficialRun([string]$Dir) {
    $r = Join-Path $Dir 'result.json'
    $p = Join-Path $Dir 'test_predictions.jsonl'
    if(!(Test-Path -LiteralPath $r) -or !(Test-Path -LiteralPath $p)) { return $false }
    try {
        $j = Get-Content -Raw -LiteralPath $r | ConvertFrom-Json
        $n = @(Get-Content -LiteralPath $p).Count
        return ($j.official -eq $true -and $n -eq 936)
    } catch { return $false }
}

try {
    Set-State 'INIT' 'RUNNING' 'SOTA extension supervisor initialized; waiting for shared GPU safely.'

    # S2B: worst-case cached 4-window pages, try batch 4 first.
    Wait-GpuFree 'Waiting before worst-case DefacementFusion cached smoke.'
    Set-State 'S2B_CACHED_FULLWINDOW_SMOKE' 'RUNNING' 'Trying physical batch 4 with four DOM windows/page in FP32; AMP was rejected by the scientific gradient-finiteness gate.'
    $smoke = Join-Path $Base 'defacementfusion_2025\smoke_cached.py'
    $x = Invoke-LoggedPython 'S2B_cached_b4_fp32' $smoke @('--batch-size','4','--no-amp')
    $DefBatch = 4
    if($x.rc -ne 0) {
        $txt = Get-Content -Raw -LiteralPath $x.log -ErrorAction SilentlyContinue
        if($txt -match '(?i)out of memory|CUDA.*memory') {
            Write-Output 'S2B_FALLBACK|batch4 OOM confirmed; retrying batch2.'
            Wait-GpuFree 'Waiting after batch4 OOM cleanup.'
            $x = Invoke-LoggedPython 'S2B_cached_b2_fp32' $smoke @('--batch-size','2','--no-amp')
            if($x.rc -ne 0) { throw 'S2B batch2 fallback failed.' }
            $DefBatch = 2
        } else { throw 'S2B failed for a non-OOM reason; refusing silent fallback.' }
    }
    $DefAccum = [int](16 / $DefBatch)
    Set-State 'S2B_CACHED_FULLWINDOW_SMOKE' 'PASS' ("Worst-case cached FP32 smoke passed; physical batch {0}, accumulation {1}." -f $DefBatch,$DefAccum)

    # S3C: MLM sanity with effective batch 16.
    Wait-GpuFree 'Waiting before MLM sanity.'
    Set-State 'S3C_HTML_MLM_SANITY' 'RUNNING' 'Two-epoch 64-window sanity; no validation/test HTML.'
    $mlm = Join-Path $Base 'defacementfusion_2025\mlm_pretrain.py'
    $MlmBatch = 8; $MlmAccum = 2
    $sanityCk = Join-Path $Base 'defacementfusion_2025\pretrain\sanity\html_mlm_sanity.pt'
    $x = Invoke-LoggedPython 'S3C_mlm_sanity_b8' $mlm @('--epochs','2','--limit-windows','64','--physical-batch-size','8','--accum-steps','2','--out',$sanityCk)
    if($x.rc -ne 0) {
        $txt = Get-Content -Raw -LiteralPath $x.log -ErrorAction SilentlyContinue
        if($txt -match '(?i)out of memory|CUDA.*memory') {
            Write-Output 'S3C_FALLBACK|batch8 OOM confirmed; retrying batch4.'
            Wait-GpuFree 'Waiting after MLM batch8 OOM cleanup.'
            $MlmBatch=4; $MlmAccum=4
            $x = Invoke-LoggedPython 'S3C_mlm_sanity_b4' $mlm @('--epochs','2','--limit-windows','64','--physical-batch-size','4','--accum-steps','4','--out',$sanityCk)
            if($x.rc -ne 0) { throw 'MLM sanity batch4 fallback failed.' }
        } else { throw 'MLM sanity failed for non-OOM reason.' }
    }
    $sanityResult = Join-Path (Split-Path $sanityCk -Parent) 'result.json'
    $sj = Get-Content -Raw -LiteralPath $sanityResult | ConvertFrom-Json
    if($sj.status -ne 'PASS' -or $sj.history.Count -ne 2) { throw 'MLM sanity result missing/invalid.' }
    $l1=[double]$sj.history[0].avg_mlm_loss; $l2=[double]$sj.history[1].avg_mlm_loss
    if([double]::IsNaN($l1) -or [double]::IsNaN($l2) -or $l2 -gt (1.5*$l1)) { throw 'MLM sanity loss is non-finite or explosively divergent.' }
    Set-State 'S3C_HTML_MLM_SANITY' 'PASS' ("MLM sanity passed; effective batch 16 via {0}x{1}; loss {2:F4}->{3:F4}." -f $MlmBatch,$MlmAccum,$l1,$l2)

    # Official 2021 task-specific comparator, three seeds.
    Wait-GpuFree 'Waiting before 2021 comparator official matrix.'
    Set-State 'OFFICIAL_BILSTM_EFFICIENTNET_2021' 'RUNNING' 'Three official seeds on frozen family-blocked LWDED-v4.'
    $r2021 = Join-Path $Base 'bilstm_efficientnet_2021\train_eval.py'
    foreach($seed in @(42,43,44)) {
        $dir = Join-Path $Base ("bilstm_efficientnet_2021\runs\official\seed_{0}" -f $seed)
        if(Test-OfficialRun $dir) { Write-Output ("SKIP_VALID|2021|seed={0}" -f $seed); continue }
        $x = Invoke-LoggedPython ("official_2021_seed{0}" -f $seed) $r2021 @('--official','--seed',"$seed",'--epochs','10','--batch-size','16','--workers','4')
        if($x.rc -ne 0) {
            $txt=Get-Content -Raw -LiteralPath $x.log -ErrorAction SilentlyContinue
            if($txt -match '(?i)DataLoader worker|BrokenPipe|pickle') {
                Write-Output ("2021_WORKER_FALLBACK|seed={0}|workers0" -f $seed)
                $x = Invoke-LoggedPython ("official_2021_seed{0}_w0" -f $seed) $r2021 @('--official','--seed',"$seed",'--epochs','10','--batch-size','16','--workers','0')
            }
        }
        if($x.rc -ne 0) { throw ("2021 official seed {0} failed." -f $seed) }
    }
    $verify = Join-Path $Base 'common\verify_official_sota.py'
    $verify2021 = Join-Path $Base 'S4_2021_RESULT_INTEGRITY.json'
    $x = Invoke-LoggedPython 'S4_verify_2021' $verify @('--model','bilstm_efficientnet_2021','--out',$verify2021)
    if($x.rc -ne 0) { throw '2021 result integrity gate failed.' }
    Set-State 'OFFICIAL_BILSTM_EFFICIENTNET_2021' 'PASS' 'Three seeds complete; 936 aligned test predictions/seed; recomputation integrity PASS.'

    # Full HTMLDeface2vec-style MLM pretraining.
    Wait-GpuFree 'Waiting before full 5-epoch HTML MLM pretraining.'
    Set-State 'FULL_HTML_MLM_PRETRAIN' 'RUNNING' 'Paper-specified 5 epochs, LR 2e-5, effective batch 16, pretrain families only.'
    $fullCk = Join-Path $Base 'defacementfusion_2025\pretrain\official\html_deface_mlm_final.pt'
    $fullResult = Join-Path (Split-Path $fullCk -Parent) 'result.json'
    $needMlm=$true
    if((Test-Path -LiteralPath $fullCk) -and (Test-Path -LiteralPath $fullResult)) {
        try {
            $fj=Get-Content -Raw -LiteralPath $fullResult|ConvertFrom-Json
            if($fj.status -eq 'PASS' -and $fj.history.Count -eq 5) {$needMlm=$false}
        } catch {}
    }
    if($needMlm) {
        $x = Invoke-LoggedPython 'full_html_mlm_pretrain' $mlm @('--epochs','5','--physical-batch-size',"$MlmBatch",'--accum-steps',"$MlmAccum",'--out',$fullCk)
        if($x.rc -ne 0) { throw 'Full HTML MLM pretraining failed.' }
    } else { Write-Output 'SKIP_VALID|full_html_mlm_pretrain' }
    $fj=Get-Content -Raw -LiteralPath $fullResult|ConvertFrom-Json
    if($fj.status -ne 'PASS' -or $fj.history.Count -ne 5) { throw 'Full MLM checkpoint/result gate invalid.' }
    Set-State 'FULL_HTML_MLM_PRETRAIN' 'PASS' 'Five-epoch MLM checkpoint complete from pretrain families only.'

    # Official DefacementFusion matrix.
    Wait-GpuFree 'Waiting before DefacementFusion official matrix.'
    Set-State 'OFFICIAL_DEFACEMENTFUSION_2025' 'RUNNING' ("Three seeds, end-to-end fine-tuning, effective batch 16 via {0}x{1}." -f $DefBatch,$DefAccum)
    $rdf = Join-Path $Base 'defacementfusion_2025\train_eval.py'
    foreach($seed in @(42,43,44)) {
        $dir = Join-Path $Base ("defacementfusion_2025\runs\official\seed_{0}" -f $seed)
        if(Test-OfficialRun $dir) { Write-Output ("SKIP_VALID|DefacementFusion|seed={0}" -f $seed); continue }
        $x = Invoke-LoggedPython ("official_deffusion_seed{0}_fp32" -f $seed) $rdf @('--official','--seed',"$seed",'--epochs','10','--physical-batch-size',"$DefBatch",'--accum-steps',"$DefAccum",'--workers','0','--html-pretrain',$fullCk,'--no-amp')
        if($x.rc -ne 0) {
            $txt=Get-Content -Raw -LiteralPath $x.log -ErrorAction SilentlyContinue
            if($txt -match '(?i)out of memory|CUDA.*memory' -and $DefBatch -gt 2) {
                Write-Output ("DEFFUSION_OOM_FALLBACK|seed={0}|batch4_to_batch2|effective_batch16_preserved" -f $seed)
                Wait-GpuFree ("Waiting after DefacementFusion OOM seed {0}." -f $seed)
                $DefBatch=2; $DefAccum=8
                $x = Invoke-LoggedPython ("official_deffusion_seed{0}_fp32_b2" -f $seed) $rdf @('--official','--seed',"$seed",'--epochs','10','--physical-batch-size','2','--accum-steps','8','--workers','0','--html-pretrain',$fullCk,'--no-amp')
            }
        }
        if($x.rc -ne 0) { throw ("DefacementFusion official seed {0} failed after allowed mechanical repair paths." -f $seed) }
    }

    $verifyBoth = Join-Path $Base 'S4_SOTA_RESULT_INTEGRITY.json'
    $x = Invoke-LoggedPython 'S4_verify_both' $verify @('--model','both','--out',$verifyBoth)
    if($x.rc -ne 0) { throw 'Final SOTA result integrity gate failed.' }
    Set-State 'S4_SOTA_RESULT_INTEGRITY' 'PASS' 'Both comparators: 3/3 seeds, aligned 936-pair predictions, exact metric recomputation.'

    # Paired comparison vs DI-MMGN.
    $sumScript = Join-Path $Base 'common\summarize_sota_comparison.py'
    $x = Invoke-LoggedPython 'S5_sota_comparison_summary' $sumScript @()
    if($x.rc -ne 0) { throw 'SOTA comparison summary failed.' }
    Set-State 'COMPLETE' 'COMPLETE' 'SOTA extension pipeline complete: two task-specific comparators + paired family-bootstrap summary.'
    exit 0
}
catch {
    Set-State 'FAILED' 'FAILED' $_.Exception.Message
    Write-Error $_
    exit 1
}
