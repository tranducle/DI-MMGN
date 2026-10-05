$ErrorActionPreference='SilentlyContinue'
$Base='C:\Users\Tran Duc Le\Documents\LOCAL-CODING-AGENT-WIN\DO-A-PAPER-v2\Papers\[IEEE ACCESS] - DI-MMGN_REVIEWED_20260923_1357\DI-MMGN-v4\external_validation\reimplementation'
$JobState='C:\Users\Tran Duc Le\Documents\LOCAL-CODING-AGENT-WIN\chatgpt-local-coder\.mcp-state\persistent-jobs\dimmgn-sota-supervisor-v4.json'
$StageState=Join-Path $Base 'SOTA_SUPERVISOR_STATE.json'
$OutDir=Join-Path $Base 'monitor'
$Csv=Join-Path $OutDir 'sota_resource_telemetry.csv'
$Json=Join-Path $OutDir 'SOTA_RESOURCE_MONITOR_STATE.json'
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
if(!(Test-Path -LiteralPath $Csv)){
  'timestamp,supervisor_alive,stage,status,gpu_mem_mib,gpu_util_pct,gpu_temp_c,gpu_power_w,official_2021_results,official_2021_predictions,official_deffusion_results,official_deffusion_predictions,mlm_official_result' | Set-Content -LiteralPath $Csv -Encoding ASCII
}
for($i=0;$i -lt 2880;$i++){
  $now=(Get-Date).ToString('o')
  $supPid=0
  if(Test-Path -LiteralPath $JobState){try{$j=Get-Content -Raw -LiteralPath $JobState|ConvertFrom-Json; $supPid=[int]$j.pid}catch{}}
  $alive=[bool](Get-Process -Id $supPid -ErrorAction SilentlyContinue)
  $stage='UNKNOWN';$status='UNKNOWN'
  if(Test-Path -LiteralPath $StageState){try{$s=Get-Content -Raw -LiteralPath $StageState|ConvertFrom-Json;$stage=[string]$s.stage;$status=[string]$s.status}catch{}}
  $gpu=& nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits 2>$null
  $mem='';$util='';$temp='';$power=''
  if($gpu){$x=$gpu.Trim().Split(','); if($x.Count -ge 4){$mem=$x[0].Trim();$util=$x[1].Trim();$temp=$x[2].Trim();$power=$x[3].Trim()}}
  $r21=@(Get-ChildItem -LiteralPath (Join-Path $Base 'bilstm_efficientnet_2021\runs\official') -Filter 'result.json' -File -Recurse -ErrorAction SilentlyContinue).Count
  $p21=@(Get-ChildItem -LiteralPath (Join-Path $Base 'bilstm_efficientnet_2021\runs\official') -Filter 'test_predictions.jsonl' -File -Recurse -ErrorAction SilentlyContinue).Count
  $rdf=@(Get-ChildItem -LiteralPath (Join-Path $Base 'defacementfusion_2025\runs\official') -Filter 'result.json' -File -Recurse -ErrorAction SilentlyContinue).Count
  $pdf=@(Get-ChildItem -LiteralPath (Join-Path $Base 'defacementfusion_2025\runs\official') -Filter 'test_predictions.jsonl' -File -Recurse -ErrorAction SilentlyContinue).Count
  $mlm=[bool](Test-Path -LiteralPath (Join-Path $Base 'defacementfusion_2025\pretrain\official\result.json'))
  ('{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12}' -f $now,$alive,$stage,$status,$mem,$util,$temp,$power,$r21,$p21,$rdf,$pdf,$mlm) | Add-Content -LiteralPath $Csv -Encoding ASCII
  $o=[ordered]@{updated_at=$now;supervisor_pid=$supPid;supervisor_alive=$alive;stage=$stage;status=$status;gpu_memory_mib=$mem;gpu_util_pct=$util;gpu_temp_c=$temp;gpu_power_w=$power;official_2021_results=$r21;official_2021_predictions=$p21;official_deffusion_results=$rdf;official_deffusion_predictions=$pdf;mlm_official_result=$mlm}
  $o|ConvertTo-Json -Depth 4|Set-Content -LiteralPath $Json -Encoding UTF8
  if(!$alive){break}
  Start-Sleep -Seconds 30
}
