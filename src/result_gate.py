import argparse, hashlib, json, math
from pathlib import Path

SEEDS={"42","43","44"}
METRICS=["defaced_f1","macro_f1","precision","attack_recall","legitimate_specificity","accuracy"]

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for c in iter(lambda:f.read(1024*1024),b""):h.update(c)
    return h.hexdigest()

def pair_test_ids(root):
    p=root/"dataset_pipeline"/"v4"/"unified_pairs.jsonl"
    rows=[json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    return [r["pair_id"] for r in rows if r["split"]=="test"]

def check_metrics(m,n):
    for k in METRICS:
        v=m.get(k)
        if not isinstance(v,(int,float)) or not math.isfinite(float(v)) or not 0<=float(v)<=1:return False,k
    if int(m.get("n",-1))!=n:return False,"n"
    return True,""

def validate_run(root,run,name,seed,test_ids,errors,warnings):
    ok,bad=check_metrics(run.get("test",{}),len(test_ids))
    if not ok: errors.append(f"{name} seed {seed}: bad metric {bad}")
    rel=run.get("predictions")
    if not rel: errors.append(f"{name} seed {seed}: predictions path missing"); return None
    p=root/rel
    if not p.exists(): errors.append(f"{name} seed {seed}: predictions missing {p}"); return None
    if run.get("predictions_sha256") and sha(p)!=run["predictions_sha256"]: errors.append(f"{name} seed {seed}: prediction hash mismatch")
    x=json.loads(p.read_text(encoding="utf-8-sig")); pr=x.get("predictions",{})
    ids=pr.get("pair_id",[]); labels=pr.get("label",[]); preds=pr.get("pred",[]); probs=pr.get("prob",[])
    if ids!=test_ids: errors.append(f"{name} seed {seed}: test pair ordering/content mismatch")
    if not (len(ids)==len(labels)==len(preds)==len(probs)==len(test_ids)): errors.append(f"{name} seed {seed}: prediction length mismatch")
    if probs and not all(isinstance(z,(int,float)) and math.isfinite(float(z)) and 0<=float(z)<=1 for z in probs): errors.append(f"{name} seed {seed}: invalid probabilities")
    if preds and len(set(preds))<2: warnings.append(f"{name} seed {seed}: single-class predictions")
    return x

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--run-root",required=True); ap.add_argument("--stage",choices=["baseline","core","fusion","final"],required=True); ap.add_argument("--out",required=True)
    a=ap.parse_args(); root=Path(a.run_root).resolve(); out=Path(a.out).resolve(); out.parent.mkdir(parents=True,exist_ok=True)
    test_ids=pair_test_ids(root); errors=[]; warnings=[]; evidence={"test_n":len(test_ids)}
    bp=root/"results"/"baseline_results.json"; tp=root/"results"/"text_baseline_results.json"; pp=root/"results"/"proposed_results.json"

    if a.stage in ("baseline","final"):
        if not bp.exists(): errors.append("baseline_results.json missing")
        else:
            br=json.loads(bp.read_text(encoding="utf-8-sig")); modes=["graphsage","gat","gin","snapshot_mm"]; n=0
            for mode in modes:
                runs=br.get("runs",{}).get(mode,{})
                if set(runs)!=SEEDS: errors.append(f"{mode}: seeds incomplete {sorted(runs)}")
                for seed,run in runs.items(): validate_run(root,run,mode,seed,test_ids,errors,warnings); n+=1
            evidence["static_baseline_runs"]=n
        if not tp.exists(): errors.append("text_baseline_results.json missing")
        else:
            tr=json.loads(tp.read_text(encoding="utf-8-sig")); runs=tr.get("runs",{}); n=0
            if set(runs)!=SEEDS: errors.append(f"text_bilstm seeds incomplete {sorted(runs)}")
            for seed,run in runs.items(): validate_run(root,run,"text_bilstm",seed,test_ids,errors,warnings); n+=1
            evidence["text_baseline_runs"]=n

    if a.stage in ("core","fusion","final"):
        if not pp.exists(): errors.append("proposed_results.json missing")
        else:
            pr=json.loads(pp.read_text(encoding="utf-8-sig"))
            modes={"core":["concat_none","concat_soft","zero_delta_none"],"fusion":["gated_none","crossattn_none"],"final":["concat_none","concat_soft","zero_delta_none","gated_none","crossattn_none"]}[a.stage]
            cache={}
            for mode in modes:
                runs=pr.get("runs",{}).get(mode,{})
                if set(runs)!=SEEDS: errors.append(f"{mode}: seeds incomplete {sorted(runs)}")
                for seed,run in runs.items(): cache[(mode,seed)]=validate_run(root,run,mode,seed,test_ids,errors,warnings)
            evidence["proposed_runs_checked"]=sum(len(pr.get("runs",{}).get(m,{})) for m in modes)
            if a.stage in ("core","final"):
                diffs={}
                for seed in SEEDS:
                    c=pr.get("runs",{}).get("concat_none",{}).get(seed,{})
                    z=pr.get("runs",{}).get("zero_delta_none",{}).get(seed,{})
                    if c.get("trainable_params")!=z.get("trainable_params"): errors.append(f"seed {seed}: matched control parameter counts differ")
                    xc=cache.get(("concat_none",seed)); xz=cache.get(("zero_delta_none",seed))
                    if xc and xz:
                        pc=xc["predictions"]["prob"]; pz=xz["predictions"]["prob"]
                        diffs[seed]=max(abs(float(u)-float(v)) for u,v in zip(pc,pz))
                evidence["matched_control_max_probability_difference"]=diffs
                if diffs and all(v<=1e-12 for v in diffs.values()): errors.append("matched control mechanism collapsed: CONCAT-none and ZERO-Delta probabilities identical across every seed")

    if a.stage=="final":
        total=0
        if bp.exists(): total+=sum(len(x) for x in json.loads(bp.read_text(encoding="utf-8-sig")).get("runs",{}).values())
        if tp.exists(): total+=len(json.loads(tp.read_text(encoding="utf-8-sig")).get("runs",{}))
        if pp.exists(): total+=sum(len(x) for x in json.loads(pp.read_text(encoding="utf-8-sig")).get("runs",{}).values())
        evidence["total_run_records"]=total
        if total!=30: errors.append(f"expected 30 runs, found {total}")

    verdict="FAIL_REPAIR" if errors else ("PASS_WITH_WARNINGS" if warnings else "PASS")
    rep={"gate_id":{"baseline":"G2_baseline_sanity","core":"G3_method_mechanism","fusion":"G4_ablation_validity","final":"G6_result_integrity"}[a.stage],"stage":a.stage,"verdict":verdict,"errors":errors,"warnings":warnings,"evidence":evidence,"next_step_allowed":not errors}
    out.write_text(json.dumps(rep,indent=2)+"\n",encoding="utf-8"); print(json.dumps(rep,indent=2))
    return 0 if not errors else 3
if __name__=="__main__": raise SystemExit(main())
