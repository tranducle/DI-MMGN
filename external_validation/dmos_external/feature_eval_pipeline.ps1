$ErrorActionPreference='Stop'
$Base='C:\Users\Tran Duc Le\Documents\LOCAL-CODING-AGENT-WIN\DO-A-PAPER-v2\Papers\[IEEE ACCESS] - DI-MMGN_REVIEWED_20260923_1357\DI-MMGN-v4\external_validation\dmos_external'
$Py='python'
$Runner='C:\Users\Tran Duc Le\Documents\LOCAL-CODING-AGENT-WIN\DO-A-PAPER-v2\Papers\[IEEE ACCESS] - DI-MMGN_REVIEWED_20260923_1357\DI-MMGN-v4\src\run_python_logged_v4.py'
$Logs=Join-Path $Base 'logs'
New-Item -ItemType Directory -Force -Path $Logs | Out-Null

function Run-Py([string]$Name,[string]$Script,[string[]]$ArgList){
  $log=Join-Path $Logs ($Name+'.log')
  & $Py -u $Runner --log $log $Py -u $Script @ArgList
  if($LASTEXITCODE -ne 0){throw ("{0} failed rc={1}" -f $Name,$LASTEXITCODE)}
}
function Wait-GpuFree(){
  $stable=0
  for($i=0;$i -lt 2160;$i++){
    $busy=@(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
      $_.Name -eq 'python.exe' -and $_.CommandLine -match '(?i)train_eval|mlm_pretrain|phase8_runner|TRAINDNSAGUARD'
    }).Count
    $g=& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null
    $mem=99999
    if($g){[void][int]::TryParse(($g|Select-Object -First 1).Trim(),[ref]$mem)}
    if($busy -eq 0 -and $mem -le 3500){$stable++}else{$stable=0}
    if($stable -ge 3){return}
    Start-Sleep -Seconds 10
  }
  throw 'GPU did not become free within wait window.'
}

$g1='D:\RESEARCH\DI_MM_V4\external_candidates\DMoS_external\G1C_DMoS_TEMPORAL_COHORT.json'
if(!(Test-Path -LiteralPath $g1)){throw 'G1C missing'}
$j=Get-Content -Raw -LiteralPath $g1|ConvertFrom-Json
if($j.verdict -notin @('PASS_FULL_COHORT','PASS_ATTACK_RECALL_ONLY')){throw ('G1C does not authorize features: '+$j.verdict)}

Run-Py 'G2_build_external_cohort' (Join-Path $Base 'build_external_cohort.py') @()
$g2=Get-Content -Raw -LiteralPath 'D:\RESEARCH\DI_MM_V4\external_candidates\DMoS_external\G2_COHORT_BUILD.json'|ConvertFrom-Json
if($g2.status -ne 'PASS'){throw 'G2 cohort build did not PASS'}

Wait-GpuFree
Run-Py 'G3_build_external_features' (Join-Path $Base 'build_external_features.py') @()
$g3=Get-Content -Raw -LiteralPath 'D:\RESEARCH\DI_MM_V4\external_candidates\DMoS_external\G3_EXTERNAL_FEATURES.json'|ConvertFrom-Json
if($g3.verdict -ne 'PASS'){throw 'G3 feature gate did not PASS'}

Wait-GpuFree
Run-Py 'G4_external_frozen_eval' (Join-Path $Base 'evaluate_external.py') @()
$g4=Get-Content -Raw -LiteralPath 'D:\RESEARCH\DI_MM_V4\external_candidates\DMoS_external\G4_EXTERNAL_FROZEN_EVAL.json'|ConvertFrom-Json
if($g4.verdict -ne 'PASS'){throw 'G4 frozen evaluation did not PASS'}

Run-Py 'G5_external_summary' (Join-Path $Base 'summarize_external.py') @()
$g5=Get-Content -Raw -LiteralPath 'D:\RESEARCH\DI_MM_V4\external_candidates\DMoS_external\G5_EXTERNAL_EVIDENCE.json'|ConvertFrom-Json
if($g5.verdict -notin @('PASS_FULL_EXTERNAL','PASS_ATTACK_RECALL_EXTERNAL')){throw ('G5 unexpected verdict '+$g5.verdict)}
Write-Output ('FEATURE_EVAL_COMPLETE|'+$g5.verdict)
exit 0
