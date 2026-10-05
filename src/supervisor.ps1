$ErrorActionPreference='Stop'
$Project='.'
$Src=Join-Path $Project 'src'
$Work='D:\RESEARCH\DI_MM_V4'
$Gates=Join-Path $Work 'gates'
$State=Join-Path $Gates 'V4_SUPERVISOR_STATE.json'
$Server='.'
$Jobs=Join-Path $Server '.mcp-state\persistent-jobs'
$AcqDone=Join-Path $Jobs 'dimmgn-v4-acquisition-v1.done.json'
$AcqState=Join-Path $Jobs 'dimmgn-v4-acquisition-v1.json'
$Run='D:\RESEARCH\DI_MM_V4\run_official'

function Save-State([string]$status,[string]$stage,[string]$note=''){
  New-Item -ItemType Directory -Force -Path $Gates | Out-Null
  [ordered]@{updated_at=(Get-Date).ToString('o');status=$status;stage=$stage;note=$note}|ConvertTo-Json -Depth 5|Set-Content -LiteralPath $State -Encoding UTF8
  Write-Output ('STATE|'+$status+'|'+$stage+'|'+$note)
}
function Run-Stage([string]$name,[scriptblock]$block){
  Save-State 'RUNNING' $name ''
  & $block
  if($LASTEXITCODE -ne 0){ throw ('Stage failed: '+$name+' exit='+$LASTEXITCODE) }
}
function Stop-Gated([string]$stage,[string]$note){
  Save-State 'BLOCKED' $stage $note
  exit 0
}

Save-State 'RUNNING' 'WAIT_PRIMARY_ACQUISITION' 'Waiting for persistent primary Wayback acquisition.'
$start=Get-Date
while(!(Test-Path -LiteralPath $AcqDone)){
  if(((Get-Date)-$start).TotalHours -gt 8){Stop-Gated 'WAIT_PRIMARY_ACQUISITION' 'Acquisition did not finish within 8 hours.'}
  Start-Sleep -Seconds 30
}
$d=Get-Content -Raw -LiteralPath $AcqDone|ConvertFrom-Json
if([int]$d.exit_code -ne 0){Stop-Gated 'PRIMARY_ACQUISITION' ('Primary acquisition process failed exit='+$d.exit_code)}

Save-State 'RUNNING' 'G0_PRIMARY' 'Running data provenance gate.'
python (Join-Path $Src 'gate_g0_data_provenance.py')
$g0exit=$LASTEXITCODE
if($g0exit -ne 0){
  Save-State 'RUNNING' 'G0_NETWORK_REPAIR_DECISION' 'Primary G0 did not pass; evaluating pre-frozen network-repair addendum.'
  python (Join-Path $Src 'network_repair_decision.py')
  $repair=$LASTEXITCODE
  if($repair -eq 0){
    Save-State 'RUNNING' 'ACQUISITION_NETWORK_REPAIR' 'One permitted single-worker repair pass; validity criteria unchanged.'
    python -u (Join-Path $Src 'acquire_wayback.py') --workers 1
    if($LASTEXITCODE -ne 0){Stop-Gated 'ACQUISITION_NETWORK_REPAIR' 'Repair acquisition process failed.'}
    Save-State 'RUNNING' 'G0_POST_REPAIR' 'Re-evaluating G0 after the one permitted network repair.'
    python (Join-Path $Src 'gate_g0_data_provenance.py')
    if($LASTEXITCODE -ne 0){Stop-Gated 'G0_POST_REPAIR' 'G0 remains below frozen thresholds after permitted repair.'}
  } else {
    Stop-Gated 'G0_PRIMARY' 'G0 failed and failure pattern does not qualify for network-only repair.'
  }
}

Run-Stage 'G1A_FAMILY_SPLIT' { python (Join-Path $Src 'build_family_split.py') }
Run-Stage 'G1B_VARIANTS' { python -u (Join-Path $Src 'generate_variants.py') }
Run-Stage 'G1_PAIRS' { python (Join-Path $Src 'build_pairs_and_g1.py') }

Run-Stage 'TEXT_MODALITY' { python -u (Join-Path $Src 'build_text_features.py') }
Run-Stage 'DOM_MODALITY' { python -u (Join-Path $Src 'build_dom_graphs.py') }
Run-Stage 'HTTP_MODALITY' { python -u (Join-Path $Src 'build_http_features.py') }
Run-Stage 'VISUAL_MODALITY' { python -u (Join-Path $Src 'build_visual_features.py') }
Run-Stage 'G1F_MODALITY_INTEGRITY' { python (Join-Path $Src 'gate_modality_integrity.py') }

Save-State 'RUNNING' 'PREPARE_RUN' 'Preparing isolated official run root.'
& (Join-Path $Src 'prepare_run.ps1')
if($LASTEXITCODE -ne 0){Stop-Gated 'PREPARE_RUN' 'Official run-root preparation failed.'}
Run-Stage 'FB4_SMOKE' { python (Join-Path $Src 'smoke.py') --run-root $Run --out (Join-Path $Gates 'FB4_smoke.json') }

Save-State 'RUNNING' 'OFFICIAL_30_RUN_MATRIX' 'All data and smoke gates passed. Starting gated 30-run matrix.'
& (Join-Path $Src 'run_matrix.ps1') -RunRoot $Run
if($LASTEXITCODE -ne 0){Stop-Gated 'OFFICIAL_30_RUN_MATRIX' 'Matrix stopped at an execution or scientific gate failure.'}
Save-State 'COMPLETE' 'G6_RESULT_INTEGRITY' 'LWDED-v4 acquisition, regeneration, smoke, 30-run matrix, and G2/G3/G4/G6 gates completed.'
