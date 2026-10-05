$ErrorActionPreference='Stop'
$Reviewed='..'
$V4Project=Join-Path $Reviewed 'DI-MMGN-v4'
$Staging=Join-Path $Reviewed 'DI-MMGN-v3-release-staging'
$Work='D:\RESEARCH\DI_MM_V4'
$Data=Join-Path $Work 'dataset_pipeline\v4'
$Gate=Join-Path $Work 'gates\G1f_modality_integrity.json'
$SplitSource=Join-Path $Work 'split\family_splits.json'
$SplitTarget=Join-Path $Data 'splits.json'
$Run=Join-Path $Work 'run_official'
$Tmp=Join-Path $Work 'run_official.tmp'

if(!(Test-Path -LiteralPath $Gate)){throw 'G1f gate missing'}
$g=Get-Content -Raw -LiteralPath $Gate|ConvertFrom-Json
if($g.verdict -ne 'PASS'){throw ('G1f not PASS: '+$g.verdict)}
$pairHash=[string]$g.unified_pairs_sha256
$snapHash=[string]$g.snapshot_manifest_sha256

if(!(Test-Path -LiteralPath $SplitSource)){throw 'family split artifact missing'}
Copy-Item -Force -LiteralPath $SplitSource -Destination $SplitTarget

if(Test-Path -LiteralPath $Tmp){Remove-Item -LiteralPath $Tmp -Recurse -Force}
New-Item -ItemType Directory -Force -Path $Tmp | Out-Null
robocopy $Staging $Tmp /E /XD dataset_pipeline results __pycache__ /XF *.pyc | Out-Null
if($LASTEXITCODE -gt 7){throw "staging copy failed $LASTEXITCODE"}

New-Item -ItemType Directory -Force -Path (Join-Path $Tmp 'dataset_pipeline'),(Join-Path $Tmp 'results\checkpoints'),(Join-Path $Tmp 'results\predictions'),(Join-Path $Tmp 'results\logs') | Out-Null
New-Item -ItemType Junction -Path (Join-Path $Tmp 'dataset_pipeline\v4') -Target $Data | Out-Null
Copy-Item -Force -LiteralPath (Join-Path $V4Project 'src\dimmgn_dataset.py') -Destination (Join-Path $Tmp 'src\data\dimmgn_dataset.py')

$files=@('src\run_v3_proposed.py','src\run_v3_static_baselines.py','src\run_v3_text_baseline.py')
$nl=[Environment]::NewLine
foreach($rel in $files){
  $fp=Join-Path $Tmp $rel
  $t=Get-Content -Raw -LiteralPath $fp
  $t=$t.Replace('data.dimmgn_dataset_v3','data.dimmgn_dataset')
  $t=$t.Replace('DimmgnPairDatasetV3','DimmgnPairDatasetV4').Replace('DimmgnCollateV3','DimmgnCollateV4')
  $t=$t.Replace('"v3"','"v4"').Replace("'v3'","'v4'")
  $t=$t.Replace('LWDED-v3','LWDED-v4')
  $t=$t.Replace('241805316dc2b22a08c8047e57cacbeeb1ec1349c1904cbd79f4c71b14c78b96',$pairHash)

  if($rel -eq 'src\run_v3_proposed.py'){
    $t=$t.Replace('num_workers=0','num_workers=4')
    $t=$t.Replace('pin_memory=torch.cuda.is_available())','pin_memory=torch.cuda.is_available(),persistent_workers=True)')
  }
  if($rel -eq 'src\run_v3_static_baselines.py'){
    $t=$t.Replace('num_workers=0','num_workers=4')
    $t=$t.Replace('pin_memory=torch.cuda.is_available(),','pin_memory=torch.cuda.is_available(),'+$nl+'            persistent_workers=True,')
  }

  Set-Content -LiteralPath $fp -Value $t -Encoding UTF8
}

$cfg=Join-Path $Tmp 'src\config\experiment_v3.yaml'
$c=Get-Content -Raw -LiteralPath $cfg
$c=$c -replace 'dataset_version:\s*LWDED-v3','dataset_version: LWDED-v4'
$c=$c -replace 'unified_pairs:\s*"[^"]+"','unified_pairs: "../dataset_pipeline/v4/unified_pairs.jsonl"'
$c=$c -replace 'splits:\s*"[^"]+"','splits: "../dataset_pipeline/v4/splits.json"'
$c=$c -replace 'official_pairs_sha256:\s*[0-9a-f]{64}',('official_pairs_sha256: '+$pairHash)
$textProv=Get-Content -Raw -LiteralPath (Join-Path $Data 'text_provenance.json') | ConvertFrom-Json
$textHash=[string]$textProv.text_manifest_sha256
$c=$c -replace 'text_manifest_sha256:\s*[0-9a-f]{64}',('text_manifest_sha256: '+$textHash)
Set-Content -LiteralPath $cfg -Value $c -Encoding UTF8

$required=@(
  (Join-Path $Data 'unified_pairs.jsonl'),
  (Join-Path $Data 'splits.json'),
  (Join-Path $Data 'snapshot_manifest.jsonl'),
  (Join-Path $Data 'text_provenance.json'),
  (Join-Path $Data 'visual_provenance.json'),
  (Join-Path $Data 'text_sequence_cache.npy'),
  (Join-Path $Data 'text_sequence_mask.npy'),
  (Join-Path $Data 'text_sequence_index.json')
)
foreach($rp in $required){if(!(Test-Path -LiteralPath $rp)){throw ('required artifact missing: '+$rp)}}

$checkFiles=@(
  (Join-Path $Tmp 'src\run_v3_proposed.py'),
  (Join-Path $Tmp 'src\run_v3_static_baselines.py'),
  (Join-Path $Tmp 'src\run_v3_text_baseline.py'),
  (Join-Path $Tmp 'src\data\dimmgn_dataset.py')
)
python -m py_compile $checkFiles
if($LASTEXITCODE -ne 0){throw 'Python compile preflight failed'}

$legacyHits=Select-String -Path $checkFiles -Pattern 'data\.dimmgn_dataset_v3|DimmgnPairDatasetV3|DimmgnCollateV3|dataset_pipeline.+v3' -ErrorAction SilentlyContinue
if($legacyHits){throw ('legacy v3 data wiring remains: '+(($legacyHits|ForEach-Object {$_.Path+':'+$_.LineNumber}) -join ', '))}

$cfgObj=Get-Content -Raw -LiteralPath $cfg | python -c "import sys,yaml,json; x=yaml.safe_load(sys.stdin.read()); print(json.dumps(x))"
if($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($cfgObj)){throw 'YAML config parse preflight failed'}

Get-ChildItem -LiteralPath (Join-Path $Tmp 'results') -File -Recurse -ErrorAction SilentlyContinue | Remove-Item -Force

$man=[ordered]@{
 prepared_at=(Get-Date).ToString('o')
 dataset_version='LWDED-v4'
 pair_sha256=$pairHash
 snapshot_manifest_sha256=$snapHash
 split_sha256=(Get-FileHash -LiteralPath $SplitTarget -Algorithm SHA256).Hash.ToLower()
 source_staging=$Staging
 data_root=$Data
 run_root=$Run
 execution_tuning=[ordered]@{
   proposed_num_workers=4
   static_num_workers=4
   text_num_workers=0
   pin_memory=$true
   persistent_workers_multimodal=$true
   batch_size_unchanged=$true
 }
 code=@{}
}
foreach($rel in $files+@('src\data\dimmgn_dataset.py','src\config\experiment_v3.yaml')){
  $fp=Join-Path $Tmp $rel
  $man.code[$rel]=(Get-FileHash -LiteralPath $fp -Algorithm SHA256).Hash.ToLower()
}
$man|ConvertTo-Json -Depth 8|Set-Content -LiteralPath (Join-Path $Tmp 'V4_RUN_MANIFEST.json') -Encoding UTF8

if(Test-Path -LiteralPath $Run){Remove-Item -LiteralPath $Run -Recurse -Force}
Move-Item -LiteralPath $Tmp -Destination $Run
Write-Output ('V4_RUN_PREPARED|'+$Run)
