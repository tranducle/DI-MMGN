import json, pathlib, hashlib

ROOT=pathlib.Path(r"D:\RESEARCH\DI_MM_V4")
V4=ROOT/"dataset_pipeline"/"v4"
OUT=ROOT/"external_candidates"/"sota_static"
OUT.mkdir(parents=True,exist_ok=True)

pairs=[json.loads(x) for x in (V4/"unified_pairs.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
snaps=[json.loads(x) for x in (V4/"snapshot_manifest.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
smap={(r["domain"],r["snapshot_id"]):r for r in snaps}

rows=[]
missing=[]
for p in pairs:
    domain=p["domain"]
    sid=p["snapshot_id"]["t2"]
    s=smap.get((domain,sid))
    if s is None:
        missing.append((p["pair_id"],domain,sid,"snapshot_manifest"))
        continue
    html=ROOT/s["html_path"]
    shot=V4/"screenshots"/domain/f"{sid}.png"
    if not html.exists(): missing.append((p["pair_id"],domain,sid,"html"))
    if not shot.exists(): missing.append((p["pair_id"],domain,sid,"screenshot"))
    rows.append({
        "pair_id":p["pair_id"],
        "split":p["split"],
        "domain":domain,
        "family_id":p["family_id"],
        "label":1 if p["label"]=="defaced" else 0,
        "label_name":p["label"],
        "attack_type":p.get("attack_type"),
        "source_type":p.get("source_type"),
        "snapshot_id":sid,
        "html_path":str(html),
        "screenshot_path":str(shot),
        "html_sha256":s["html_sha256"],
    })

out=OUT/"lwded_v4_static_current_manifest.jsonl"
with out.open("w",encoding="utf-8",newline="\n") as f:
    for r in rows:
        f.write(json.dumps(r,sort_keys=True,separators=(",",":"))+"\n")

h=hashlib.sha256(out.read_bytes()).hexdigest()
summary={
    "rows":len(rows),
    "missing_count":len(missing),
    "missing_sample":missing[:20],
    "split_counts":{s:sum(r["split"]==s for r in rows) for s in ["pretrain","val","test"]},
    "label_counts":{s:{str(y):sum(r["split"]==s and r["label"]==y for r in rows) for y in [0,1]} for s in ["pretrain","val","test"]},
    "unique_current_snapshots":len({(r["domain"],r["snapshot_id"]) for r in rows}),
    "manifest_sha256":h,
    "manifest":str(out),
}
(OUT/"lwded_v4_static_current_manifest_summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
print(json.dumps(summary,indent=2))
raise SystemExit(0 if not missing and len(rows)==4678 else 3)
