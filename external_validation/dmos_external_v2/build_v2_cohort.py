import argparse, hashlib, json, pathlib
V1=pathlib.Path(r"D:\RESEARCH\DI_MM_V4\external_candidates\DMoS_external")
ROOT=pathlib.Path(r"D:\RESEARCH\DI_MM_V4\external_candidates\DMoS_external_v2")
P=ROOT/"pools"

def read(p):
    q=pathlib.Path(p)
    if not q.exists(): return []
    return [json.loads(x) for x in q.read_text(encoding="utf-8").splitlines() if x.strip()]

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for c in iter(lambda:f.read(1024*1024),b""):h.update(c)
    return h.hexdigest()

ap=argparse.ArgumentParser();ap.add_argument("--primary",required=True);ap.add_argument("--secondary",default="");args=ap.parse_args()
gatep=ROOT/"V2_G1C_FINAL_COHORT_GATE.json"
if not gatep.exists(): gatep=ROOT/"V2_G1B_PRIMARY_GATE.json"
g=json.loads(gatep.read_text(encoding="utf-8"))
if g["verdict"] not in ("PASS_ATTACK_MIN","PASS_ATTACK_STRONG","PASS_FULL_COHORT"):
    raise SystemExit("cohort gate does not authorize build: "+g["verdict"])

v1full={r["id"]:r for r in read(V1/"dmos_temporal_candidates.jsonl")}
primary=read(args.primary);secondary=read(args.secondary) if args.secondary else []
salv={r["id"]:r for r in primary+secondary}
existing=read(P/"existing_positive.jsonl")
pairs=[];snaps={}

def add_snap(domain,sid,path,source,meta):
    key=domain+"|"+sid
    if key not in snaps:
        p=pathlib.Path(path)
        if not p.exists() or p.stat().st_size==0: raise FileNotFoundError(p)
        snaps[key]={"domain":domain,"snapshot_id":sid,"html_path":str(p),"html_sha256":sha(p),"source_type":source,**meta}
    return sid

for e in existing:
    r=v1full[e["id"]]
    a=r["positive_t1"]; domain=r["host_hint"]
    s1=f"wbv1__{a['timestamp']}__{r['id']}";s2=f"dmos_real__{r['id']}"
    add_snap(domain,s1,a["archive_html_path"],"wayback_clean_v1",{"timestamp":a["timestamp"],"dmos_id":r["id"]})
    add_snap(domain,s2,r["html_path"],"dmos_real_defaced",{"timestamp":r["collection_iso"],"dmos_id":r["id"]})
    pairs.append({"pair_id":"pos_"+r["id"],"label":"defaced","domain":domain,"host_group":domain,"source_type":"dmos_real_temporal","t1":s1,"t2":s2,"dmos_id":r["id"],"protocol_source":"v1_inherited"})

for r in salv.values():
    if r.get("positive_pair_eligible_v2"):
        a=r["positive_t1_v2"];domain=r["host_hint"]
        s1=f"wbv2__{a['timestamp']}__{r['id']}";s2=f"dmos_real__{r['id']}"
        add_snap(domain,s1,a["archive_html_path"],"wayback_clean_v2",{"timestamp":a["timestamp"],"dmos_id":r["id"]})
        add_snap(domain,s2,r["html_path"],"dmos_real_defaced",{"timestamp":r["collection_iso"],"dmos_id":r["id"]})
        pairs.append({"pair_id":"pos_"+r["id"],"label":"defaced","domain":domain,"host_group":domain,"source_type":"dmos_real_temporal","t1":s1,"t2":s2,"dmos_id":r["id"],"protocol_source":"v2_deep_history"})
    if g["verdict"]=="PASS_FULL_COHORT" and r.get("benign_pair_eligible_v2"):
        a=r["benign_t1_v2"];b=r["benign_t2_v2"];domain=r["host_hint"]
        s1=f"wbv2__{a['timestamp']}__{r['id']}";s2=f"wbv2__{b['timestamp']}__{r['id']}"
        add_snap(domain,s1,a["archive_html_path"],"wayback_clean_v2",{"timestamp":a["timestamp"],"dmos_id":r["id"]})
        add_snap(domain,s2,b["archive_html_path"],"wayback_clean_v2",{"timestamp":b["timestamp"],"dmos_id":r["id"]})
        pairs.append({"pair_id":"ben_"+r["id"],"label":"legitimate","domain":domain,"host_group":domain,"source_type":"wayback_clean_to_clean_external","t1":s1,"t2":s2,"dmos_id":r["id"],"protocol_source":"v2_deep_history"})

pairs=sorted({x["pair_id"]:x for x in pairs}.values(),key=lambda x:x["pair_id"])
snaps=sorted(snaps.values(),key=lambda x:(x["domain"],x["snapshot_id"]))
sm=ROOT/"external_snapshot_manifest_v2.jsonl";pm=ROOT/"external_pairs_v2.jsonl"
with sm.open("w",encoding="utf-8",newline="\n") as f:
    for x in snaps:f.write(json.dumps(x,sort_keys=True,separators=(",",":"))+"\n")
with pm.open("w",encoding="utf-8",newline="\n") as f:
    for x in pairs:f.write(json.dumps(x,sort_keys=True,separators=(",",":"))+"\n")
rep={"gate":"V2_G2_COHORT_BUILD","verdict":"PASS","cohort_gate":g["verdict"],"snapshots":len(snaps),"pairs":len(pairs),
     "positive_pairs":sum(x["label"]=="defaced" for x in pairs),"benign_pairs":sum(x["label"]=="legitimate" for x in pairs),
     "host_groups":len({x["host_group"] for x in pairs}),"snapshot_manifest_sha256":sha(sm),"pair_manifest_sha256":sha(pm)}
(ROOT/"V2_G2_COHORT_BUILD.json").write_text(json.dumps(rep,indent=2)+"\n",encoding="utf-8")
print(json.dumps(rep,indent=2))
