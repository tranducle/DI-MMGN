param()

$ErrorActionPreference='Stop'
$Proj='C:\Users\Tran Duc Le\Documents\LOCAL-CODING-AGENT-WIN\DO-A-PAPER-v2\Papers\[IEEE ACCESS] - DI-MMGN_REVIEWED_20260923_1357\DI-MMGN-v4\external_validation\dmos_external_v2'
$Root='D:\RESEARCH\DI_MM_V4\external_candidates\DMoS_external_v2'
$State=Join-Path $Proj 'DMOS_V2_SUPERVISOR_STATE.json'
$Logs=Join-Path $Proj 'logs'
New-Item -ItemType Directory -Force -Path $Logs | Out-Null

function Write-State([string]$Stage,[string]$Status,[string]$Note=''){
    $o=[ordered]@{updated_at=(Get-Date).ToString('o');status=$Status;stage=$Stage;note=$Note;supervisor_pid=$PID}
    $o|ConvertTo-Json -Depth 6|Set-Content -LiteralPath $State -Encoding UTF8
    Write-Output ("STATE stage={0} status={1} note={2}" -f $Stage,$Status,$Note)
}
function Run-Python([string]$Name,[string]$Script,[string[]]$ArgList,[int]$Retries=3){
    $log=Join-Path $Logs ($Name+'.log')
    for($try=1;$try -le $Retries;$try++){
        Write-Host ("CHILD START name={0} try={1} script={2} args={3}" -f $Name,$try,$Script,($ArgList -join ' '))
        & python $Script @ArgList 2>&1 | ForEach-Object {
            Write-Host $_
            $_ | Out-File -LiteralPath $log -Append -Encoding utf8
        }
        $code=$LASTEXITCODE
        Write-Host ("CHILD EXIT name={0} try={1} code={2}" -f $Name,$try,$code)
        if($code -eq 0){return 0}
        if($code -eq 5){return 5}
        if($try -lt $Retries){
            $wait=@(60,180,300)[$try-1]
            Write-Host ("MECHANICAL RETRY name={0} wait={1}s" -f $Name,$wait)
            Start-Sleep -Seconds $wait
        }
    }
    return $code
}

try {
    Write-State 'V2_G0_PROTOCOL' 'RUNNING' 'verify frozen protocol and v1 provenance'
    $v1Gate='D:\RESEARCH\DI_MM_V4\external_candidates\DMoS_external\G1C_DMoS_TEMPORAL_COHORT.json'
    if(!(Test-Path -LiteralPath $v1Gate)){throw 'missing v1 G1C gate'}
    $v1=Get-Content -Raw -LiteralPath $v1Gate|ConvertFrom-Json
    if($v1.verdict -ne 'FAIL_REDESIGN'){throw ('unexpected v1 verdict '+$v1.verdict)}
    $protocol=Join-Path $Proj 'DMOS_V2_PROTOCOL.json'
    $expected=(Get-Content -Raw -LiteralPath (Join-Path $Proj 'DMOS_V2_PROTOCOL.sha256')).Trim().ToLower()
    $actual=(Get-FileHash -Algorithm SHA256 -LiteralPath $protocol).Hash.ToLower()
    if($actual -ne $expected){throw 'protocol hash mismatch'}
    $c=Run-Python 'V2_G0_pools' (Join-Path $Proj 'build_v2_pools.py') @() 1
    if($c -ne 0){throw 'pool provenance failed'}
    $g0=Get-Content -Raw -LiteralPath (Join-Path $Root 'V2_G0_POOL_PROVENANCE.json')|ConvertFrom-Json
    if($g0.verdict -ne 'PASS'){throw 'V2 G0 pool gate failed'}
    Write-State 'V2_G0_PROTOCOL' 'PASS' ('protocol_sha256='+$actual)

    Write-State 'V2_G1A_SMOKE' 'RUNNING' '5-record mechanical deep-history smoke'
    $smoke=Join-Path $Root 'V2_G1A_smoke.jsonl'
    $smokeArgs=@('--pool',(Join-Path $Root 'pools\primary_salvage_pool.jsonl'),'--out',$smoke,'--ids','dmos_0102,dmos_0103,dmos_0393,dmos_0394,dmos_0395','--max-captures','24','--years','10')
    $c=Run-Python 'V2_G1A_smoke' (Join-Path $Proj 'deep_history_salvage.py') $smokeArgs 3
    if($c -ne 0){throw ('smoke child failed code '+$c)}
    $sr=@(Get-Content -LiteralPath $smoke|Where-Object{$_.Trim()}|ForEach-Object{$_|ConvertFrom-Json})
    $timeline=@($sr|Where-Object{$_.timeline_unique_captures -gt 0}).Count
    $replay=0
    foreach($r in $sr){$replay+=@($r.assessed_captures_v2|Where-Object{$_.http_status -eq 200}).Count}
    $sv=if($sr.Count -eq 5 -and $timeline -ge 1 -and $replay -ge 1){'PASS'}else{'FAIL'}
    $sg=[ordered]@{gate='V2_G1A_SMOKE';verdict=$sv;records=$sr.Count;records_with_timeline=$timeline;successful_replay_200=$replay;ids=@($sr|ForEach-Object{$_.id})}
    $sg|ConvertTo-Json -Depth 6|Set-Content -LiteralPath (Join-Path $Root 'V2_G1A_SMOKE.json') -Encoding UTF8
    if($sv -ne 'PASS'){throw 'V2 G1A smoke gate failed'}
    Write-State 'V2_G1A_SMOKE' 'PASS' ("records={0}; timeline={1}; replay200={2}" -f $sr.Count,$timeline,$replay)

    Write-State 'V2_G1B_PRIMARY_SALVAGE' 'RUNNING' 'deep-history salvage on frozen 99-record primary pool'
    $primary=Join-Path $Root 'V2_G1B_primary_salvage.jsonl'
    $pa=@('--pool',(Join-Path $Root 'pools\primary_salvage_pool.jsonl'),'--out',$primary,'--max-captures','24','--years','10')
    $c=Run-Python 'V2_G1B_primary_salvage' (Join-Path $Proj 'deep_history_salvage.py') $pa 3
    if($c -ne 0){throw ('primary salvage mechanical failure code '+$c)}
    $c=Run-Python 'V2_G1B_primary_gate' (Join-Path $Proj 'summarize_v2_gate.py') @('--stage','primary','--primary',$primary) 1
    $pg=Get-Content -Raw -LiteralPath (Join-Path $Root 'V2_G1B_PRIMARY_GATE.json')|ConvertFrom-Json
    if($pg.verdict -eq 'FAIL_REDESIGN_DMoS'){
        Write-State 'V2_G1B_PRIMARY_SALVAGE' 'FAIL_REDESIGN' ("positive={0}; hosts={1}; new_primary={2}" -f $pg.positive_pairs,$pg.positive_host_groups,$pg.new_primary_positive)
        exit 5
    }
    Write-State 'V2_G1B_PRIMARY_SALVAGE' 'PASS' ("verdict={0}; positive={1}; hosts={2}; new_primary={3}" -f $pg.verdict,$pg.positive_pairs,$pg.positive_host_groups,$pg.new_primary_positive)

    $secondary=''
    if($pg.verdict -eq 'CONTINUE_SECONDARY'){
        Write-State 'V2_G1C_SECONDARY_SALVAGE' 'RUNNING' 'predeclared continuation criterion satisfied'
        $secondary=Join-Path $Root 'V2_G1C_secondary_salvage.jsonl'
        $sa=@('--pool',(Join-Path $Root 'pools\secondary_salvage_pool.jsonl'),'--out',$secondary,'--max-captures','24','--years','10')
        $c=Run-Python 'V2_G1C_secondary_salvage' (Join-Path $Proj 'deep_history_salvage.py') $sa 3
        if($c -ne 0){throw ('secondary salvage mechanical failure code '+$c)}
        $c=Run-Python 'V2_G1C_final_gate' (Join-Path $Proj 'summarize_v2_gate.py') @('--stage','final','--primary',$primary,'--secondary',$secondary) 1
    } else {
        Write-State 'V2_G1C_FINAL_COHORT' 'RUNNING' 'primary salvage already meets external attack gate'
        $c=Run-Python 'V2_G1C_final_gate' (Join-Path $Proj 'summarize_v2_gate.py') @('--stage','final','--primary',$primary) 1
    }
    $fg=Get-Content -Raw -LiteralPath (Join-Path $Root 'V2_G1C_FINAL_COHORT_GATE.json')|ConvertFrom-Json
    if($fg.verdict -eq 'FAIL_REDESIGN_DMoS'){
        Write-State 'V2_G1C_FINAL_COHORT' 'FAIL_REDESIGN' ("positive={0}; hosts={1}; benign={2}; benign_hosts={3}" -f $fg.positive_pairs,$fg.positive_host_groups,$fg.benign_pairs,$fg.benign_host_groups)
        exit 5
    }
    Write-State 'V2_G1C_FINAL_COHORT' 'PASS' ("verdict={0}; positive={1}; hosts={2}; benign={3}; benign_hosts={4}" -f $fg.verdict,$fg.positive_pairs,$fg.positive_host_groups,$fg.benign_pairs,$fg.benign_host_groups)

    Write-State 'V2_G2_COHORT_BUILD' 'RUNNING' 'freeze manifests before model inference'
    $ba=@('--primary',$primary)
    if($secondary){$ba+=@('--secondary',$secondary)}
    $c=Run-Python 'V2_G2_cohort_build' (Join-Path $Proj 'build_v2_cohort.py') $ba 1
    if($c -ne 0){throw ('cohort build failed '+$c)}
    $c=Run-Python 'V2_label_qa_package' (Join-Path $Proj 'generate_label_qa_package.py') @() 1
    if($c -ne 0){throw ('QA package generation failed '+$c)}
    Write-State 'V2_G2_COHORT_BUILD' 'PASS' 'manifests hashed; QA package generated before model inference'

    Write-State 'V2_G2_5_MODALITY_MATCHED' 'RUNNING' 'frozen internal missing-modality robustness controls'
    $im=Join-Path $Root 'V2_G2_5_INTERNAL_MODALITY_MATCHED.json'
    if(!(Test-Path -LiteralPath $im)){
        $c=Run-Python 'V2_G2_5_internal_masks' (Join-Path $Proj 'internal_modality_control.py') @() 2
        if($c -ne 0){throw ('internal modality control failed '+$c)}
    }
    $ig=Get-Content -Raw -LiteralPath $im|ConvertFrom-Json
    if($ig.verdict -ne 'PASS'){throw 'internal modality control gate failed'}
    Write-State 'V2_G2_5_MODALITY_MATCHED' 'PASS' 'no-HTTP and text+DOM frozen controls ready'

    Write-State 'V2_G3_EXTERNAL_FEATURES' 'RUNNING' 'build frozen external features'
    $c=Run-Python 'V2_G3_features' (Join-Path $Proj 'build_external_features_v2.py') @() 2
    if($c -ne 0){throw ('G3 feature build failed '+$c)}
    $g3=Get-Content -Raw -LiteralPath (Join-Path $Root 'V2_G3_EXTERNAL_FEATURES.json')|ConvertFrom-Json
    if($g3.verdict -ne 'PASS'){throw 'G3 gate failed'}
    Write-State 'V2_G3_EXTERNAL_FEATURES' 'PASS' ("visual_mode={0}; eval_pairs={1}" -f $g3.visual_mode,$g3.eval_pairs)

    Write-State 'V2_G4_FROZEN_EVAL' 'RUNNING' 'evaluate seeds 42/43/44; threshold=0.5; no DMoS tuning'
    $c=Run-Python 'V2_G4_eval' (Join-Path $Proj 'evaluate_external_v2.py') @() 2
    if($c -ne 0){throw ('G4 eval failed '+$c)}
    $g4=Get-Content -Raw -LiteralPath (Join-Path $Root 'V2_G4_EXTERNAL_FROZEN_EVAL.json')|ConvertFrom-Json
    if($g4.verdict -ne 'PASS'){throw 'G4 gate failed'}
    Write-State 'V2_G4_FROZEN_EVAL' 'PASS' ("pairs={0}; positive={1}; benign={2}" -f $g4.external_pairs,$g4.external_positive,$g4.external_benign)

    Write-State 'V2_G5_STATISTICS' 'RUNNING' 'host+seed bootstrap and modality-matched generalization gap'
    $c=Run-Python 'V2_G5_summary' (Join-Path $Proj 'summarize_external_v2.py') @() 1
    if($c -ne 0){throw ('G5 summary failed '+$c)}
    $g5=Get-Content -Raw -LiteralPath (Join-Path $Root 'V2_G5_EXTERNAL_EVIDENCE.json')|ConvertFrom-Json
    Write-State 'COMPLETE_EXPERIMENTS_LABEL_QA_PENDING' 'COMPLETE' ("G5={0}; pairs={1}; hosts={2}; final manuscript promotion waits for independent label-QA review" -f $g5.verdict,$g5.pairs,$g5.host_groups)
    exit 0
}
catch {
    $msg=$_.Exception.Message
    Write-Output ("SUPERVISOR_ERROR "+$msg)
    Write-State 'SUPERVISOR' 'FAILED_MECHANICAL' $msg
    exit 2
}
