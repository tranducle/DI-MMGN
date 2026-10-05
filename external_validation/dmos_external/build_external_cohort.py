import hashlib, json, pathlib
ROOT=pathlib.Path(r"D:\RESEARCH\DI_MM_V4\external_candidates\DMoS_external")
CAND=ROOT/"dmos_temporal_candidates.jsonl"
GATE=ROOT/"G1C_DMoS_TEMPORAL_COHORT.json"

def sha_file(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for c in iter(lambda:f.read(1024*1024),b""): h.update(c)
    return h.hexdigest()

g=json.loads(GATE.read_text(encoding="utf-8"))
if g["verdict"] not in ("PASS_FULL_COHORT","PASS_ATTACK_RECALL_ONLY"):
    raise SystemExit("G1C does not authorize cohort build: "+g["verdict"])
rows=[json.loads(x) for x in CAND.read_text(encoding="utf-8").splitlines() if x.strip()]
pairs=[]; snaps={}

def add_snap(domain,sid,path,source,meta):
    key=domain+"|"+sid
    if key not in snaps:
        p=pathlib.Path(path)
        if not p.exists() or p.stat().st_size==0: raise FileNotFoundError(p)
        snaps[key]={"domain":domain,"snapshot_id":sid,"html_path":str(p),"html_sha256":sha_file(p),
                    "source_type":source,**meta}
    return sid

for r in rows:
    domain=r["host_hint"]
    if r.get("positive_pair_eligible"):
        a=r["positive_t1"]; b=r["positive_t2"]
        s1=f"wb__{a['timestamp']}__{r['id']}"
        s2=f"dmos_real__{r['id']}"
        add_snap(domain,s1,a["archive_html_path"],"wayback_clean",{"timestamp":a["timestamp"],"dmos_id":r["id"]})
        add_snap(domain,s2,b["html_path"],"dmos_real_defaced",{"timestamp":r["collection_iso"],"dmos_id":r["id"]})
        pairs.append({"pair_id":"pos_"+r["id"],"label":"defaced","domain":domain,"host_group":domain,
                      "source_type":"dmos_real_temporal","t1":s1,"t2":s2,"dmos_id":r["id"],
                      "collection_iso":r["collection_iso"],"positive_keyword_hits":r["positive_keyword_hits"]})
    if g["verdict"]=="PASS_FULL_COHORT" and r.get("benign_pair_eligible"):
        a=r["benign_t1"]; b=r["benign_t2"]
        s1=f"wb__{a['timestamp']}__{r['id']}"
        s2=f"wb__{b['timestamp']}__{r['id']}"
        add_snap(domain,s1,a["archive_html_path"],"wayback_clean",{"timestamp":a["timestamp"],"dmos_id":r["id"]})
        add_snap(domain,s2,b["archive_html_path"],"wayback_clean",{"timestamp":b["timestamp"],"dmos_id":r["id"]})
        pairs.append({"pair_id":"ben_"+r["id"],"label":"legitimate","domain":domain,"host_group":domain,
                      "source_type":"wayback_clean_to_clean_external","t1":s1,"t2":s2,"dmos_id":r["id"]})

snap_rows=sorted(snaps.values(),key=lambda x:(x["domain"],x["snapshot_id"]))
pair_rows=sorted(pairs,key=lambda x:x["pair_id"])
sm=ROOT/"external_snapshot_manifest.jsonl"; pm=ROOT/"external_pairs.jsonl"
with sm.open("w",encoding="utf-8",newline="\n") as f:
    for x in snap_rows:f.write(json.dumps(x,sort_keys=True,separators=(",",":"))+"\n")
with pm.open("w",encoding="utf-8",newline="\n") as f:
    for x in pair_rows:f.write(json.dumps(x,sort_keys=True,separators=(",",":"))+"\n")
summary={
  "status":"PASS",
  "cohort_mode":g["verdict"],
  "snapshots":len(snap_rows),"pairs":len(pair_rows),
  "positive_pairs":sum(x["label"]=="defaced" for x in pair_rows),
  "benign_pairs":sum(x["label"]=="legitimate" for x in pair_rows),
  "host_groups":len({x["host_group"] for x in pair_rows}),
  "snapshot_manifest_sha256":sha_file(sm),"pair_manifest_sha256":sha_file(pm),
}
(ROOT/"G2_COHORT_BUILD.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
print(json.dumps(summary,indent=2))
