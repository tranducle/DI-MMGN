import argparse, hashlib, json, pathlib
ROOT=pathlib.Path(r"D:\RESEARCH\DI_MM_V4\external_candidates\DMoS_external_v2")
P=ROOT/"pools"

def rows(p):
    q=pathlib.Path(p)
    if not q.exists(): return []
    return [json.loads(x) for x in q.read_text(encoding="utf-8").splitlines() if x.strip()]

def sha(p):
    q=pathlib.Path(p)
    h=hashlib.sha256()
    with q.open("rb") as f:
        for c in iter(lambda:f.read(1024*1024),b""):h.update(c)
    return h.hexdigest()

def norm_existing(r):
    return {"id":r["id"],"host_hint":r["host_hint"],"positive":True,"benign":bool(len(r.get("v1_assessed_captures",[]))>=2),
            "positive_t1":None,"source":"v1_existing"}

def norm_salvage(r):
    return {"id":r["id"],"host_hint":r["host_hint"],"positive":bool(r.get("positive_pair_eligible_v2")),
            "benign":bool(r.get("benign_pair_eligible_v2")),"positive_t1":r.get("positive_t1_v2"),"source":"v2_salvage"}

ap=argparse.ArgumentParser()
ap.add_argument("--stage",choices=["primary","final"],required=True)
ap.add_argument("--primary",required=True)
ap.add_argument("--secondary",default="")
args=ap.parse_args()

existing=rows(P/"existing_positive.jsonl")
primary=rows(args.primary)
secondary=rows(args.secondary) if args.secondary else []

alln=[norm_existing(r) for r in existing]+[norm_salvage(r) for r in primary]+[norm_salvage(r) for r in secondary]
pos=[r for r in alln if r["positive"]]
ben=[r for r in alln if r["benign"]]
ph=sorted({r["host_hint"] for r in pos}); bh=sorted({r["host_hint"] for r in ben})
p_new=sum(bool(r.get("positive_pair_eligible_v2")) for r in primary)

if args.stage=="primary":
    if len(pos)>=100 and len(ph)>=8: verdict="PASS_ATTACK_STRONG"
    elif len(pos)>=50 and len(ph)>=5: verdict="PASS_ATTACK_MIN"
    elif p_new>=10: verdict="CONTINUE_SECONDARY"
    else: verdict="FAIL_REDESIGN_DMoS"
    out=ROOT/"V2_G1B_PRIMARY_GATE.json"
else:
    if len(pos)>=100 and len(ph)>=8 and len(ben)>=100 and len(bh)>=8: verdict="PASS_FULL_COHORT"
    elif len(pos)>=100 and len(ph)>=8: verdict="PASS_ATTACK_STRONG"
    elif len(pos)>=50 and len(ph)>=5: verdict="PASS_ATTACK_MIN"
    else: verdict="FAIL_REDESIGN_DMoS"
    out=ROOT/"V2_G1C_FINAL_COHORT_GATE.json"

report={"gate":"V2_G1B_PRIMARY_GATE" if args.stage=="primary" else "V2_G1C_FINAL_COHORT_GATE",
        "verdict":verdict,"existing_v1_positive":len(existing),"primary_records_processed":len(primary),
        "secondary_records_processed":len(secondary),"new_primary_positive":p_new,
        "positive_pairs":len(pos),"positive_host_groups":len(ph),"positive_hosts":ph,
        "benign_pairs":len(ben),"benign_host_groups":len(bh),"benign_hosts":bh,
        "thresholds":{"attack_min":{"positive_pairs":50,"hosts":5},"attack_strong":{"positive_pairs":100,"hosts":8},
                      "full":{"positive_pairs":100,"positive_hosts":8,"benign_pairs":100,"benign_hosts":8},
                      "continue_secondary_if_new_primary_positive_at_least":10},
        "inputs":{"primary":args.primary,"primary_sha256":sha(args.primary) if pathlib.Path(args.primary).exists() else None,
                  "secondary":args.secondary or None,"secondary_sha256":sha(args.secondary) if args.secondary and pathlib.Path(args.secondary).exists() else None}}
out.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
print(json.dumps(report,indent=2))
if verdict=="FAIL_REDESIGN_DMoS": raise SystemExit(5)
