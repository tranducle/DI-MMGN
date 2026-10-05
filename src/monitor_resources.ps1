param(
  [string]$SupervisorJobKey='dimmgn-v4-supervisor-v4',
  [int]$IntervalSeconds=60,
  [int]$StallSamples=10
)
$ErrorActionPreference='SilentlyContinue'
$Server='.'
$Jobs=Join-Path $Server '.mcp-state\persistent-jobs'
$JobState=Join-Path $Jobs ($SupervisorJobKey+'.json')
$DoneState=Join-Path $Jobs ($SupervisorJobKey+'.done.json')
$Root='D:\RESEARCH\DI_MM_V4'
$GateState=Join-Path $Root 'gates\V4_SUPERVISOR_STATE.json'
$OutDir=Join-Path $Root 'monitor'
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$Csv=Join-Path $OutDir 'resource_telemetry.csv'
$State=Join-Path $OutDir 'RESOURCE_MONITOR_STATE.json'
if(!(Test-Path -LiteralPath $Csv)){
  'timestamp,stage,supervisor_status,health,gpu_util_pct,gpu_mem_used_mb,gpu_mem_total_mb,gpu_temp_c,gpu_power_w,cpu_pct,available_mem_mb,screenshots,visual_vectors,checkpoints,predictions' | Set-Content -LiteralPath $Csv -Encoding ASCII
}
$started=Get-Date
$lastStage=''
$lastProgress=-1
$stagnant=0
while($true){
  $now=Get-Date
  $supStatus='Unknown'
  $supPid=$null
  if(Test-Path -LiteralPath $JobState){
    try{
      $j=Get-Content -Raw -LiteralPath $JobState|ConvertFrom-Json
      $supStatus=[string]$j.status
      $supPid=[int]$j.pid
    }catch{}
  }
  $stage='Unknown'
  if(Test-Path -LiteralPath $GateState){
    try{$stage=[string](Get-Content -Raw -LiteralPath $GateState|ConvertFrom-Json).stage}catch{}
  }

  $gpu=@('','','','','')
  try{
    $line=& nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits 2>$null | Select-Object -First 1
    if($line){$gpu=@($line -split ',' | ForEach-Object {$_.Trim()})}
  }catch{}
  $cpu=''
  $mem=''
  try{
    $cs=Get-Counter '\Processor(_Total)\% Processor Time','\Memory\Available MBytes' -SampleInterval 1 -MaxSamples 1
    foreach($x in $cs.CounterSamples){
      if($x.Path -match 'processor'){$cpu=[math]::Round($x.CookedValue,1)}
      elseif($x.Path -match 'memory'){$mem=[math]::Round($x.CookedValue,0)}
    }
  }catch{}

  $shots=(Get-ChildItem (Join-Path $Root 'dataset_pipeline\v4\screenshots') -File -Recurse -Filter '*.png' -ErrorAction SilentlyContinue|Measure-Object).Count
  $visual=(Get-ChildItem (Join-Path $Root 'dataset_pipeline\v4\visuals') -File -Recurse -Filter '*.npy' -ErrorAction SilentlyContinue|Measure-Object).Count
  $ck=(Get-ChildItem (Join-Path $Root 'run_official\results\checkpoints') -File -ErrorAction SilentlyContinue|Measure-Object).Count
  $pred=(Get-ChildItem (Join-Path $Root 'run_official\results\predictions') -File -Filter '*.json' -ErrorAction SilentlyContinue|Measure-Object).Count

  $progress=$shots+$visual+$ck+$pred
  if($stage -eq $lastStage -and $progress -eq $lastProgress){$stagnant++}else{$stagnant=0}
  $lastStage=$stage
  $lastProgress=$progress
  $health=if($stagnant -ge $StallSamples){'STALL_SUSPECTED'}else{'OK'}

  $row=@(
    $now.ToString('o'),$stage,$supStatus,$health,$gpu[0],$gpu[1],$gpu[2],$gpu[3],$gpu[4],
    $cpu,$mem,$shots,$visual,$ck,$pred
  ) -join ','
  Add-Content -LiteralPath $Csv -Value $row -Encoding ASCII

  $alive=$false
  if($supPid){$alive=[bool](Get-Process -Id $supPid -ErrorAction SilentlyContinue)}
  $doneExists=Test-Path -LiteralPath $DoneState
  [ordered]@{
    updated_at=$now.ToString('o')
    started_at=$started.ToString('o')
    supervisor_job_key=$SupervisorJobKey
    supervisor_status=$supStatus
    supervisor_pid=$supPid
    supervisor_alive=$alive
    done_marker_exists=$doneExists
    stage=$stage
    health=$health
    stagnant_samples=$stagnant
    gpu_util_pct=$gpu[0]
    gpu_mem_used_mb=$gpu[1]
    gpu_mem_total_mb=$gpu[2]
    gpu_temp_c=$gpu[3]
    gpu_power_w=$gpu[4]
    cpu_pct=$cpu
    available_mem_mb=$mem
    screenshots=$shots
    visual_vectors=$visual
    checkpoints=$ck
    predictions=$pred
  }|ConvertTo-Json -Depth 5|Set-Content -LiteralPath $State -Encoding UTF8

  if($doneExists -or !$alive){
    break
  }
  Start-Sleep -Seconds $IntervalSeconds
}
