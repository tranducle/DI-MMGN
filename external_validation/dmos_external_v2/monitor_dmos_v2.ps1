param([string]$SupervisorJobKey='dimmgn-dmos-v2-supervisor-v1')
$ErrorActionPreference='Continue'
$Proj='C:\Users\Tran Duc Le\Documents\LOCAL-CODING-AGENT-WIN\DO-A-PAPER-v2\Papers\[IEEE ACCESS] - DI-MMGN_REVIEWED_20260923_1357\DI-MMGN-v4\external_validation\dmos_external_v2'
$Root='D:\RESEARCH\DI_MM_V4\external_candidates\DMoS_external_v2'
$Server='C:\Users\Tran Duc Le\Documents\LOCAL-CODING-AGENT-WIN\chatgpt-local-coder'
$Out=Join-Path $Proj 'monitor\DMOS_V2_MONITOR_STATE.json'
$Csv=Join-Path $Proj 'monitor\dmos_v2_monitor.csv'
if(!(Test-Path -LiteralPath $Csv)){Set-Content -LiteralPath $Csv -Value 'timestamp,supervisor_pid,alive,stage,status,smoke_rows,primary_rows,secondary_rows,positive_pairs,positive_hosts' -Encoding ascii}
$dead=0
while($true){
  $sfile=Join-Path $Server ('.mcp-state\persistent-jobs\'+$SupervisorJobKey+'.json')
  $pidv=$null;$alive=$false
  if(Test-Path -LiteralPath $sfile){
    try{$js=Get-Content -Raw -LiteralPath $sfile|ConvertFrom-Json;$pidv=[int]$js.pid;if(Get-Process -Id $pidv -ErrorAction SilentlyContinue){$alive=$true}}catch{}
  }
  $stage='UNKNOWN';$status='UNKNOWN'
  $state=Join-Path $Proj 'DMOS_V2_SUPERVISOR_STATE.json'
  if(Test-Path -LiteralPath $state){try{$z=Get-Content -Raw -LiteralPath $state|ConvertFrom-Json;$stage=$z.stage;$status=$z.status}catch{}}
  function Count-Lines($p){if(Test-Path -LiteralPath $p){return @(Get-Content -LiteralPath $p|Where-Object{$_.Trim()}).Count};return 0}
  $sm=Count-Lines (Join-Path $Root 'V2_G1A_smoke.jsonl')
  $pr=Count-Lines (Join-Path $Root 'V2_G1B_primary_salvage.jsonl')
  $se=Count-Lines (Join-Path $Root 'V2_G1C_secondary_salvage.jsonl')
  $pos=0;$ph=0
  foreach($gname in @('V2_G1C_FINAL_COHORT_GATE.json','V2_G1B_PRIMARY_GATE.json')){
    $gp=Join-Path $Root $gname
    if(Test-Path -LiteralPath $gp){try{$g=Get-Content -Raw -LiteralPath $gp|ConvertFrom-Json;$pos=$g.positive_pairs;$ph=$g.positive_host_groups;break}catch{}}
  }
  $obj=[ordered]@{timestamp=(Get-Date).ToString('o');supervisor_pid=$pidv;alive=$alive;stage=$stage;status=$status;smoke_rows=$sm;primary_rows=$pr;secondary_rows=$se;positive_pairs=$pos;positive_hosts=$ph}
  $obj|ConvertTo-Json -Depth 5|Set-Content -LiteralPath $Out -Encoding UTF8
  Add-Content -LiteralPath $Csv -Value ("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9}" -f $obj.timestamp,$pidv,$alive,$stage,$status,$sm,$pr,$se,$pos,$ph)
  if($alive){$dead=0}else{$dead++}
  if($dead -ge 3){break}
  Start-Sleep -Seconds 60
}
