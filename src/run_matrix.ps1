param([string]$RunRoot='D:\RESEARCH\DI_MM_V4\run_official')
$ErrorActionPreference='Stop'
$V4Project='.'
$GateRoot='D:\RESEARCH\DI_MM_V4\gates'
$Smoke=Join-Path $GateRoot 'FB4_smoke.json'
$GateScript=Join-Path $V4Project 'src\result_gate.py'
$Runner=Join-Path $V4Project 'src\run_python_logged.py'
$Logs=Join-Path $RunRoot 'results\logs'
New-Item -ItemType Directory -Force -Path $Logs | Out-Null
if(!(Test-Path -LiteralPath $Smoke)){throw 'FB4 smoke report missing'}
$sm=Get-Content -Raw -LiteralPath $Smoke|ConvertFrom-Json
if($sm.verdict -ne 'PASS'){throw ('FB4 not PASS: '+$sm.verdict)}
Set-Location -LiteralPath $RunRoot
function Stage([string]$Name,[scriptblock]$Block){
  Write-Output ('STAGE_START|'+$Name+'|'+(Get-Date).ToString('o'))
  & $Block
  if($LASTEXITCODE -ne 0){throw ('Stage failed '+$Name+' exit='+$LASTEXITCODE)}
  Write-Output ('STAGE_END|'+$Name+'|'+(Get-Date).ToString('o'))
}
Stage 'STATIC_BASELINES' {
  python -u $Runner --log (Join-Path $Logs 'static_baselines.log') python -u src\run_v3_static_baselines.py --models graphsage gat gin snapshot_mm --seeds 42 43 44
}
Stage 'TEXT_BASELINE' {
  python -u $Runner --log (Join-Path $Logs 'text_baseline.log') python -u src\run_v3_text_baseline.py --seeds 42 43 44
}
Stage 'G2_BASELINE_SANITY' {
  python $GateScript --run-root $RunRoot --stage baseline --out (Join-Path $GateRoot 'G2_baseline_sanity.json')
}
Stage 'CORE_PROPOSED_CONTROL' {
  python -u $Runner --log (Join-Path $Logs 'core_proposed_control.log') python -u src\run_v3_proposed.py --runs concat_none concat_soft zero_delta_none --seeds 42 43 44
}
Stage 'G3_METHOD_MECHANISM' {
  python $GateScript --run-root $RunRoot --stage core --out (Join-Path $GateRoot 'G3_method_mechanism.json')
}
Stage 'FUSION_ABLATIONS' {
  python -u $Runner --log (Join-Path $Logs 'fusion_ablations.log') python -u src\run_v3_proposed.py --runs gated_none crossattn_none --seeds 42 43 44
}
Stage 'G4_ABLATION_VALIDITY' {
  python $GateScript --run-root $RunRoot --stage fusion --out (Join-Path $GateRoot 'G4_ablation_validity.json')
}
Stage 'G6_RESULT_INTEGRITY' {
  python $GateScript --run-root $RunRoot --stage final --out (Join-Path $GateRoot 'G6_result_integrity.json')
}
$Dest=Join-Path $V4Project 'results'
New-Item -ItemType Directory -Force -Path $Dest | Out-Null
foreach($f in @('baseline_results.json','text_baseline_results.json','proposed_results.json')){
  $p=Join-Path $RunRoot ('results\'+$f); if(Test-Path -LiteralPath $p){Copy-Item -Force -LiteralPath $p -Destination $Dest}
}
robocopy (Join-Path $RunRoot 'results\predictions') (Join-Path $Dest 'predictions') /E | Out-Null
if($LASTEXITCODE -gt 7){throw "prediction sync failed $LASTEXITCODE"}
robocopy (Join-Path $RunRoot 'results\logs') (Join-Path $Dest 'logs') /E | Out-Null
if($LASTEXITCODE -gt 7){throw "log sync failed $LASTEXITCODE"}
$ck=Get-ChildItem -LiteralPath (Join-Path $RunRoot 'results\checkpoints') -File | Sort-Object Name | ForEach-Object {[ordered]@{name=$_.Name;bytes=[int64]$_.Length;sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLower();working_path=$_.FullName}}
[ordered]@{generated_at=(Get-Date).ToString('o');count=@($ck).Count;checkpoints=@($ck)}|ConvertTo-Json -Depth 6|Set-Content -LiteralPath (Join-Path $Dest 'checkpoint_manifest.json') -Encoding UTF8
Copy-Item -Force -LiteralPath (Join-Path $RunRoot 'V4_RUN_MANIFEST.json') -Destination $Dest
Write-Output ('V4_MATRIX_COMPLETE|'+$Dest)
$global:LASTEXITCODE=0
