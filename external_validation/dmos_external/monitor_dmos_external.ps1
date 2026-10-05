$ErrorActionPreference='SilentlyContinue'
$Base='C:\Users\Tran Duc Le\Documents\LOCAL-CODING-AGENT-WIN\DO-A-PAPER-v2\Papers\[IEEE ACCESS] - DI-MMGN_REVIEWED_20260923_1357\DI-MMGN-v4\external_validation\dmos_external'
$Job='C:\Users\Tran Duc Le\Documents\LOCAL-CODING-AGENT-WIN\chatgpt-local-coder\.mcp-state\persistent-jobs\dimmgn-dmos-external-supervisor-v1.json'
$State=Join-Path $Base 'DMOS_SUPERVISOR_STATE.json'
$Out=Join-Path $Base 'monitor'
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$Json=Join-Path $Out 'DMOS_MONITOR_STATE.json'
$Csv=Join-Path $Out 'dmos_monitor.csv'
if(!(Test-Path -LiteralPath $Csv)){'timestamp,alive,stage,status,probe_records,full_records,candidate_records,positive_pairs,benign_pairs,g3_exists,g4_exists,g5_exists'|Set-Content -LiteralPath $Csv -Encoding ASCII}
for($i=0;$i -lt 4320;$i++){
 $supPid=0;if(Test-Path -LiteralPath $Job){try{$j=Get-Content -Raw -LiteralPath $Job|ConvertFrom-Json;$supPid=[int]$j.pid}catch{}}
 $alive=[bool](Get-Process -Id $supPid -ErrorAction SilentlyContinue)
 $stage='UNKNOWN';$status='UNKNOWN';if(Test-Path -LiteralPath $State){try{$s=Get-Content -Raw -LiteralPath $State|ConvertFrom-Json;$stage=$s.stage;$status=$s.status}catch{}}
 $probe=0;$full=0;$cand=0;$pos=0;$ben=0
 $pp='D:\RESEARCH\DI_MM_V4\external_candidates\DMoS_external\wayback_probe_discovery.jsonl'; if(Test-Path -LiteralPath $pp){$probe=@(Get-Content -LiteralPath $pp).Count}
 $fp='D:\RESEARCH\DI_MM_V4\external_candidates\DMoS_external\wayback_full_discovery.jsonl'; if(Test-Path -LiteralPath $fp){$full=@(Get-Content -LiteralPath $fp).Count}
 $cp='D:\RESEARCH\DI_MM_V4\external_candidates\DMoS_external\dmos_temporal_candidates.jsonl'; if(Test-Path -LiteralPath $cp){$cand=@(Get-Content -LiteralPath $cp).Count}
 $g1='D:\RESEARCH\DI_MM_V4\external_candidates\DMoS_external\G1C_DMoS_TEMPORAL_COHORT.json';if(Test-Path -LiteralPath $g1){try{$x=Get-Content -Raw -LiteralPath $g1|ConvertFrom-Json;$pos=$x.positive_pairs;$ben=$x.benign_pairs}catch{}}
 $g3=Test-Path -LiteralPath 'D:\RESEARCH\DI_MM_V4\external_candidates\DMoS_external\G3_EXTERNAL_FEATURES.json'
 $g4=Test-Path -LiteralPath 'D:\RESEARCH\DI_MM_V4\external_candidates\DMoS_external\G4_EXTERNAL_FROZEN_EVAL.json'
 $g5=Test-Path -LiteralPath 'D:\RESEARCH\DI_MM_V4\external_candidates\DMoS_external\G5_EXTERNAL_EVIDENCE.json'
 $now=(Get-Date).ToString('o')
 $o=[ordered]@{updated_at=$now;supervisor_pid=$supPid;alive=$alive;stage=$stage;status=$status;probe_records=$probe;full_records=$full;candidate_records=$cand;positive_pairs=$pos;benign_pairs=$ben;g3_features=$g3;g4_eval=$g4;g5_evidence=$g5}
 $o|ConvertTo-Json|Set-Content -LiteralPath $Json -Encoding UTF8
 ('{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11}' -f $now,$alive,$stage,$status,$probe,$full,$cand,$pos,$ben,$g3,$g4,$g5)|Add-Content -LiteralPath $Csv -Encoding ASCII
 if(!$alive){break};Start-Sleep -Seconds 30
}
