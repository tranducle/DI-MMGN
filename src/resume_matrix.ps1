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

Require-Pass (Join-Path $Gates 'G1f_modality_integrity.json') 'G1F'
Require-Pass (Join-Path $Gates 'FB4_smoke.json') 'FB4'
if(!(Test-Path -LiteralPath (Join-Path $Run 'V4_RUN_MANIFEST.json'))){throw 'V4 run manifest missing'}

$existing=@(
  Get-ChildItem -LiteralPath (Join-Path $Run 'results\checkpoints') -File -ErrorAction SilentlyContinue
  Get-ChildItem -LiteralPath (Join-Path $Run 'results\predictions') -File -ErrorAction SilentlyContinue
  Get-ChildItem -LiteralPath (Join-Path $Run 'results') -File -Filter '*_results.json' -ErrorAction SilentlyContinue
)
if(@($existing).Count -gt 0){throw ('Refusing clean matrix restart because result artifacts already exist: '+@($existing).Count)}

Save-State 'RUNNING' 'OFFICIAL_30_RUN_MATRIX' 'Restarting clean matrix after PowerShell stderr-handling repair; G1F and FB4 remain PASS.'
& (Join-Path $Src 'run_matrix.ps1') -RunRoot $Run
if($LASTEXITCODE -ne 0){throw ('Matrix stopped at execution or scientific gate failure, exit='+$LASTEXITCODE)}
Save-State 'COMPLETE' 'G6_RESULT_INTEGRITY' '30-run matrix and G2/G3/G4/G6 gates completed.'
