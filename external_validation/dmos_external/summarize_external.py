import json, pathlib, random, statistics
import numpy as np
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score, confusion_matrix

ROOT=pathlib.Path(r"D:\RESEARCH\DI_MM_V4\external_candidates\DMoS_external")
SEEDS=[42,43,44]; REPS=20000; RNG=20261001

def calc(rows):
    y=np.array([r["y"] for r in rows],dtype=int);p=np.array([r["pred"] for r in rows],dtype=int)
    out={"n":len(rows),"attack_recall":float(recall_score(y,p,pos_label=1,zero_division=0)),
         "defaced_f1":float(f1_score(y,p,pos_label=1,zero_division=0)),
         "precision":float(precision_score(y,p,pos_label=1,zero_division=0)),
         "accuracy":float(accuracy_score(y,p))}
    if len(np.unique(y))>1:
        tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel()
        out.update({"macro_f1":float(f1_score(y,p,average="macro",zero_division=0)),
                    "legitimate_specificity":float(tn/(tn+fp)) if tn+fp else None,
                    "tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)})
    else:
        out.update({"macro_f1":None,"legitimate_specificity":None})
    return out

g1=json.loads((ROOT/"G1C_DMoS_TEMPORAL_COHORT.json").read_text(encoding="utf-8"))
g4=json.loads((ROOT/"G4_EXTERNAL_FROZEN_EVAL.json").read_text(encoding="utf-8"))
pred={}
for s in SEEDS:
    rows=[json.loads(x) for x in (ROOT/"external_predictions"/f"dmos_external_seed{s}.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    pred[s]=rows

hosts=sorted({r["host_group"] for r in pred[42]})
metrics_by_seed={s:calc(pred[s]) for s in SEEDS}
internal={s:g4["seeds"][str(s)]["internal_modality_matched"] for s in SEEDS}

# Hierarchical bootstrap: resample seeds and host groups with replacement.
rng=random.Random(RNG)
metric_names=["attack_recall","defaced_f1","accuracy"]
if g1["verdict"]=="PASS_FULL_COHORT": metric_names+=["macro_f1","legitimate_specificity"]
dist={m:[] for m in metric_names}
byseedhost={s:{h:[r for r in pred[s] if r["host_group"]==h] for h in hosts} for s in SEEDS}
for _ in range(REPS):
    sampled_seeds=[rng.choice(SEEDS) for _ in SEEDS]
    sampled_hosts=[rng.choice(hosts) for _ in hosts]
    vals={m:[] for m in metric_names}
    for s in sampled_seeds:
        rr=[]
        for h in sampled_hosts: rr.extend(byseedhost[s][h])
        mm=calc(rr)
        for m in metric_names: vals[m].append(mm[m])
    for m in metric_names:dist[m].append(float(np.mean(vals[m])))
ci={}
for m,x in dist.items():
    a=np.asarray(x,float)
    ci[m]={"bootstrap_mean":float(a.mean()),"ci95":[float(np.quantile(a,.025)),float(np.quantile(a,.975))]}

means={}
for m in metric_names:
    a=[metrics_by_seed[s][m] for s in SEEDS]
    means[m]={"mean":float(np.mean(a)),"sd":float(np.std(a,ddof=1))}

report={
  "gate":"G5_EXTERNAL_EVIDENCE",
  "verdict":"PASS_FULL_EXTERNAL" if g1["verdict"]=="PASS_FULL_COHORT" else "PASS_ATTACK_RECALL_EXTERNAL",
  "cohort_verdict":g1["verdict"],"visual_mode":g4["visual_mode"],"mask":g4["mask"],
  "seeds":SEEDS,"host_groups":len(hosts),"pairs":len(pred[42]),
  "metrics_by_seed":metrics_by_seed,"external_summary":means,
  "hierarchical_bootstrap":{"replicates":REPS,"rng_seed":RNG,"unit":"seed + DMoS host group","metrics":ci},
  "internal_modality_matched_by_seed":internal,
  "internal_modality_matched_macro_f1_mean":float(np.mean([internal[s]["macro_f1"] for s in SEEDS])) if internal[42].get("macro_f1") is not None else None,
  "claim_boundary":"Independent external real-defacement validation with no DMoS retraining; missing HTTP masked. Full binary metrics only if PASS_FULL_COHORT; otherwise attack-recall-only."
}
(ROOT/"G5_EXTERNAL_EVIDENCE.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
print(json.dumps(report,indent=2))
