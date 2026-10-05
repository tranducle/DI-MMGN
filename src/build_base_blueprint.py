import json, hashlib
from pathlib import Path
from collections import Counter

SRC = Path(r".\dataset_pipeline\v3\text_manifest.jsonl")
OUT = Path(r"D:\RESEARCH\DI_MM_V4\blueprint")
OUT.mkdir(parents=True, exist_ok=True)
rows=[]
for line in SRC.read_text(encoding="utf-8-sig").splitlines():
    if not line.strip():
        continue
    r=json.loads(line)
    sid=r["snapshot_id"]
    sh=r["source_html"].replace("\\","/")
    if sid.startswith("raw__") and "_legit_" not in sid and sh.startswith("raw_data/"):
        ts=Path(sh).stem
        rows.append({
            "domain":r["domain"].lower().strip("."),
            "requested_timestamp":ts,
            "v3_snapshot_id":sid,
            "v3_source_html":sh,
            "v3_source_html_sha256":r["source_html_sha256"],
            "v3_text_embedding_sha256":r["text_embedding_sha256"],
        })
rows.sort(key=lambda x:(x["domain"],x["requested_timestamp"]))
assert len(rows)==751, len(rows)
assert len({(x["domain"],x["requested_timestamp"]) for x in rows})==751
out=OUT/"base_snapshots.jsonl"
with out.open("w",encoding="utf-8",newline="\n") as f:
    for r in rows:
        f.write(json.dumps(r,sort_keys=True,separators=(",",":"))+"\n")
h=hashlib.sha256(out.read_bytes()).hexdigest()
summary={
    "source":str(SRC),
    "source_sha256":hashlib.sha256(SRC.read_bytes()).hexdigest(),
    "base_snapshot_count":len(rows),
    "domain_count":len(set(x["domain"] for x in rows)),
    "per_domain_min":min(Counter(x["domain"] for x in rows).values()),
    "per_domain_max":max(Counter(x["domain"] for x in rows).values()),
    "blueprint_path":str(out),
    "blueprint_sha256":h,
}
(OUT/"base_blueprint_summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
print(json.dumps(summary,indent=2))
