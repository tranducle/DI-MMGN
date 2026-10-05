import json
from pathlib import Path
ROOT=Path(r"D:\RESEARCH\DI_MM_V4")
V4=ROOT/"dataset_pipeline"/"v4"
SNAPS=V4/"snapshot_manifest.jsonl"
SHOTS=V4/"screenshots"
OUT=ROOT/"visual_repair"/"missing_visual_rows.jsonl"
OUT.parent.mkdir(parents=True,exist_ok=True)
rows=[json.loads(x) for x in SNAPS.read_text(encoding="utf-8").splitlines() if x.strip()]
rows.sort(key=lambda r:(r["domain"],r["snapshot_id"]))
missing=[r for r in rows if not (SHOTS/r["domain"]/f"{r['snapshot_id']}.png").exists()]
with OUT.open("w",encoding="utf-8",newline="\n") as f:
    for r in missing:
        f.write(json.dumps(r,sort_keys=True,separators=(",",":"))+"\n")
print(json.dumps({"total_snapshots":len(rows),"existing_screenshots":len(rows)-len(missing),"missing":len(missing),"manifest":str(OUT)},indent=2))
