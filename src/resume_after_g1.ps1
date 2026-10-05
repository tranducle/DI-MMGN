$ErrorActionPreference='Stop'
$Project='.'
$Src=Join-Path $Project 'src'
$Work='D:\RESEARCH\DI_MM_V4'
$Gates=Join-Path $Work 'gates'
$State=Join-Path $Gates 'V4_SUPERVISOR_STATE.json'
$Run=Join-Path $Work 'run_official'

function Save-State([string]$status,[string]$stage,[string]$note=''){
  [ordered]@{updated_at=(Get-Date).ToString('o');status=$status;stage=$stage;note=$note}|ConvertTo-Json -Depth 5|Set-Content -LiteralPath $State -Encoding UTF8
  Write-Output ('STATE|'+$status+'|'+$stage+'|'+$note)
}
function Require-Pass([string]$path,[string]$name){
  if(!(Test-Path -LiteralPath $path)){throw ($name+' report missing: '+$path)}
  $x=Get-Content -Raw -LiteralPath $path|ConvertFrom-Json
  if($x.verdict -ne 'PASS'){throw ($name+' not PASS: '+$x.verdict)}
}
function Run-Stage([string]$name,[scriptblock]$block){
  Save-State 'RUNNING' $name ''
  & $block
  if($LASTEXITCODE -ne 0){throw ('Stage failed: '+$name+' exit='+$LASTEXITCODE)}
}

Require-Pass (Join-Path $Gates 'G0_data_provenance.json') 'G0'
Require-Pass (Join-Path $Gates 'G1a_family_split_integrity.json') 'G1A'
Require-Pass (Join-Path $Gates 'G1b_variant_generation.json') 'G1B'
Require-Pass (Join-Path $Gates 'G1_preprocessing_integrity.json') 'G1'

Run-Stage 'TEXT_MODALITY' { python -u (Join-Path $Src 'build_text_features.py') }
Run-Stage 'DOM_MODALITY' { python -u (Join-Path $Src 'build_dom_graphs.py') }
Run-Stage 'HTTP_MODALITY' { python -u (Join-Path $Src 'build_http_features.py') }
Run-Stage 'VISUAL_MODALITY' { python -u (Join-Path $Src 'build_visual_features.py') }
Run-Stage 'G1F_MODALITY_INTEGRITY' { python (Join-Path $Src 'gate_modality_integrity.py') }

Save-State 'RUNNING' 'PREPARE_RUN' 'Preparing isolated official run root.'
& (Join-Path $Src 'prepare_run.ps1')
if($LASTEXITCODE -ne 0){throw 'Official run-root preparation failed.'}

Run-Stage 'FB4_SMOKE' { python (Join-Path $Src 'smoke.py') --run-root $Run --out (Join-Path $Gates 'FB4_smoke.json') }

Save-State 'RUNNING' 'OFFICIAL_30_RUN_MATRIX' 'All data and smoke gates passed. Starting gated 30-run matrix.'
& (Join-Path $Src 'run_matrix.ps1') -RunRoot $Run
if($LASTEXITCODE -ne 0){throw 'Matrix stopped at an execution or scientific gate failure.'}

Save-State 'COMPLETE' 'G6_RESULT_INTEGRITY' 'LWDED-v4 modality regeneration, smoke, 30-run matrix, and G2/G3/G4/G6 gates completed.'
