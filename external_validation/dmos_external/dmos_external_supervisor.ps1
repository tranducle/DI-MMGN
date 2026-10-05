$ErrorActionPreference='Stop'
$Base='C:\Users\Tran Duc Le\Documents\LOCAL-CODING-AGENT-WIN\DO-A-PAPER-v2\Papers\[IEEE ACCESS] - DI-MMGN_REVIEWED_20260923_1357\DI-MMGN-v4\external_validation\dmos_external'
$Py='python'
$Runner='C:\Users\Tran Duc Le\Documents\LOCAL-CODING-AGENT-WIN\DO-A-PAPER-v2\Papers\[IEEE ACCESS] - DI-MMGN_REVIEWED_20260923_1357\DI-MMGN-v4\src\run_python_logged_v4.py'
$State=Join-Path $Base 'DMOS_SUPERVISOR_STATE.json'
$Logs=Join-Path $Base 'logs'
New-Item -ItemType Directory -Force -Path $Logs | Out-Null

function Set-State([string]$Stage,[string]$Status,[string]$Note){
  $o=[ordered]@{updated_at=(Get-Date).ToString('o');stage=$Stage;status=$Status;note=$Note}
  $o|ConvertTo-Json -Depth 4|Set-Content -LiteralPath $State -Encoding UTF8
  Write-Output ("STATE|{0}|{1}|{2}" -f $Stage,$Status,$Note)
}
function Run-Py([string]$Name,[string]$Script,[string[]]$ArgList){
  $log=Join-Path $Logs ($Name+'.log')
  & $Py -u $Runner --log $log $Py -u $Script @ArgList
  return [ordered]@{rc=$LASTEXITCODE;log=$log}
}
function Run-WithNetworkRetry([string]$Name,[string]$Script,[string[]]$ArgList){
  $waits=@(60,180,300)
  for($i=0;$i -lt 3;$i++){
    $x=Run-Py ($Name+"_try"+($i+1)) $Script $ArgList
    if($x.rc -eq 0 -or $x.rc -eq 5){return $x}
    Write-Output ("MECHANICAL_RETRY|{0}|rc={1}|wait={2}" -f $Name,$x.rc,$waits[$i])
    Start-Sleep -Seconds $waits[$i]
  }
  return $x
}

try {
  Set-State 'G0_DMoS_PROVENANCE' 'RUNNING' 'Checking frozen DMoS provenance gate.'
  $g0='D:\RESEARCH\DI_MM_V4\external_candidates\DMoS_external\G0_DMoS_PROVENANCE.json'
  if(!(Test-Path -LiteralPath $g0)){throw 'G0 artifact missing'}
  $j=Get-Content -Raw -LiteralPath $g0|ConvertFrom-Json
  if($j.status -ne 'PASS'){throw ('G0 not PASS: '+$j.status)}
  Set-State 'G0_DMoS_PROVENANCE' 'PASS' ("500 release records; {0} non-empty; {1} inferred URLs; {2} host groups." -f $j.nonempty,$j.unique_inferred_urls,$j.unique_host_hints)

  $discover=Join-Path $Base 'wayback_discover.py'
  Set-State 'G1A_WAYBACK_FEASIBILITY' 'RUNNING' 'Stratified 60-page same-URL predecessor feasibility probe with 429-aware backoff.'
  $x=Run-WithNetworkRetry 'G1A_wayback_probe' $discover @('--mode','probe','--sample-size','60','--max-captures','4')
  if($x.rc -eq 5){
    Set-State 'G1A_WAYBACK_FEASIBILITY' 'FAIL_REDESIGN' 'Predeclared feasibility threshold not met; scientific forward pipeline stopped.'
    exit 5
  }
  if($x.rc -ne 0){throw ('G1A mechanical failure after retry, rc='+$x.rc)}
  $g1a=Get-Content -Raw -LiteralPath 'D:\RESEARCH\DI_MM_V4\external_candidates\DMoS_external\G1A_wayback_probe.json'|ConvertFrom-Json
  if($g1a.verdict -notin @('PASS','PASS_WITH_WARNINGS')){throw ('Unexpected G1A verdict '+$g1a.verdict)}
  Set-State 'G1A_WAYBACK_FEASIBILITY' $g1a.verdict ("Coverage={0:P1}; host groups={1}." -f [double]$g1a.coverage,[int]$g1a.host_groups_with_capture)

  Set-State 'G1B_WAYBACK_DISCOVERY' 'RUNNING' 'Full 498-page same-URL predecessor discovery.'
  $x=Run-WithNetworkRetry 'G1B_wayback_full' $discover @('--mode','full','--max-captures','4')
  if($x.rc -ne 0){throw ('G1B failed after retry, rc='+$x.rc)}
  $g1b=Get-Content -Raw -LiteralPath 'D:\RESEARCH\DI_MM_V4\external_candidates\DMoS_external\G1B_wayback_discovery.json'|ConvertFrom-Json
  Set-State 'G1B_WAYBACK_DISCOVERY' 'PASS' ("Discovered pre-collection captures for {0}/{1} pages across {2} hosts." -f $g1b.with_pre_capture,$g1b.records,$g1b.host_groups_with_capture)

  $fetch=Join-Path $Base 'fetch_screen_wayback.py'
  Set-State 'G1C_DMoS_TEMPORAL_COHORT' 'RUNNING' 'Fetching archived HTML and applying frozen clean-screen.'
  $x=Run-WithNetworkRetry 'G1C_fetch_screen' $fetch @()
  if($x.rc -eq 5){
    Set-State 'G1C_DMoS_TEMPORAL_COHORT' 'FAIL_REDESIGN' 'Temporal cohort did not meet predeclared minimum; external claim not authorized.'
    exit 5
  }
  if($x.rc -ne 0){throw ('G1C failed after retry, rc='+$x.rc)}
  $g1c=Get-Content -Raw -LiteralPath 'D:\RESEARCH\DI_MM_V4\external_candidates\DMoS_external\G1C_DMoS_TEMPORAL_COHORT.json'|ConvertFrom-Json
  Set-State 'G1C_DMoS_TEMPORAL_COHORT' $g1c.verdict ("positive={0}/{1} hosts; benign={2}/{3} hosts" -f $g1c.positive_pairs,$g1c.positive_host_groups,$g1c.benign_pairs,$g1c.benign_host_groups)

  $feature=Join-Path $Base 'feature_eval_pipeline.ps1'
  Set-State 'WAIT_FEATURE_EVAL_PIPELINE' 'RUNNING' 'Temporal cohort gate passed; waiting for feature/evaluation pipeline file if not already present.'
  for($i=0;$i -lt 720;$i++){
    if(Test-Path -LiteralPath $feature){
      Set-State 'FEATURE_EVAL_PIPELINE' 'RUNNING' 'Launching frozen feature extraction and external evaluation.'
      & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $feature
      if($LASTEXITCODE -ne 0){throw ('feature_eval_pipeline failed rc='+$LASTEXITCODE)}
      Set-State 'COMPLETE' 'COMPLETE' 'DMoS external-validation pipeline completed through final evidence gate.'
      exit 0
    }
    Start-Sleep -Seconds 60
  }
  throw 'Feature/eval pipeline file not created within supervisor wait window.'
}
catch {
  Set-State 'FAILED' 'FAILED' $_.Exception.Message
  Write-Error $_
  exit 1
}
