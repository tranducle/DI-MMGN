import json, pathlib, random, statistics
import numpy as np
from sklearn.metrics import recall_score, f1_score, confusion_matrix

ROOT=pathlib.Path(r"D:\RESEARCH\DI_MM_V4\external_candidates\DMoS_external_v2")
SEEDS=[42,43,44]; REPS=20000; RNG=20261002
g1=json.loads((ROOT/"V2_G1C_FINAL_COHORT_GATE.json").read_text(encoding="utf-8"))
g4=json.loads((ROOT/"V2_G4_EXTERNAL_FROZEN_EVAL.json").read_text(encoding="utf-8"))
pred={}
for s in SEEDS:
    p=ROOT/"external_predictions_v2"/f"dmos_external_seed{s}.jsonl"
    pred[s]=[json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]

hosts=sorted({r["host_group"] for r in pred[42]})
if not hosts: raise SystemExit("no host groups")

def attack_recall(rows):
    y=np.asarray([r["y"] for r in rows],int);p=np.asarray([r["pred"] for r in rows],int)
    return float(recall_score(y,p,pos_label=1,zero_division=0))

def full_metrics(rows):
    y=np.asarray([r["y"] for r in rows],int);p=np.asarray([r["pred"] for r in rows],int)
    tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel()
    return {"macro_f1":float(f1_score(y,p,average="macro",zero_division=0)),
            "attack_recall":float(tp/(tp+fn)) if tp+fn else None,
            "legitimate_specificity":float(tn/(tn+fp)) if tn+fp else None,
            "tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)}

by_seed={s:attack_recall(pred[s]) for s in SEEDS}
internal={s:g4["seeds"][str(s)]["internal_modality_matched"]["attack_recall"] for s in SEEDS}
per_host={s:{h:[r for r in pred[s] if r["host_group"]==h] for h in hosts} for s in SEEDS}

rng=random.Random(RNG);dist=[];gap=[]
for _ in range(REPS):
    ss=[rng.choice(SEEDS) for _ in SEEDS]
    hh=[rng.choice(hosts) for _ in hosts]
    er=[];ig=[]
    for s in ss:
        rr=[]
        for h in hh: rr.extend(per_host[s][h])
        er.append(attack_recall(rr))
        ig.append(internal[s])
    ev=float(np.mean(er)); iv=float(np.mean(ig))
    dist.append(ev); gap.append(ev-iv)
a=np.asarray(dist,float);b=np.asarray(gap,float)
report={
  "gate":"V2_G5_EXTERNAL_EVIDENCE",
  "verdict":"PASS_FULL_EXTERNAL" if g1["verdict"]=="PASS_FULL_COHORT" else "PASS_ATTACK_RECALL_EXTERNAL",
  "cohort_verdict":g1["verdict"],"visual_mode":g4["visual_mode"],"mask":g4["mask"],
  "seeds":SEEDS,"pairs":len(pred[42]),"host_groups":len(hosts),
  "attack_recall_by_seed":{str(k):v for k,v in by_seed.items()},
  "attack_recall_mean":float(np.mean(list(by_seed.values()))),
  "attack_recall_sd":float(np.std(list(by_seed.values()),ddof=1)),
  "internal_modality_matched_attack_recall_by_seed":{str(k):v for k,v in internal.items()},
  "internal_modality_matched_attack_recall_mean":float(np.mean(list(internal.values()))),
  "hierarchical_bootstrap":{"replicates":REPS,"rng_seed":RNG,"unit":"seed + DMoS host group",
       "external_attack_recall":{"bootstrap_mean":float(a.mean()),"ci95":[float(np.quantile(a,.025)),float(np.quantile(a,.975))]},
       "external_minus_internal_attack_recall":{"bootstrap_mean":float(b.mean()),"ci95":[float(np.quantile(b,.025)),float(np.quantile(b,.975))]}},
  "per_host_attack_recall":{h:{str(s):attack_recall(per_host[s][h]) for s in SEEDS} for h in hosts},
  "claim_boundary":"Independent real-defacement temporal attack-recall evidence with frozen DI-MMGN and no DMoS retraining. Full binary metrics are permitted only when cohort verdict is PASS_FULL_COHORT. Final manuscript claim promotion remains conditional on the label-QA review package."
}
if g1["verdict"]=="PASS_FULL_COHORT":
    report["full_binary_by_seed"]={str(s):full_metrics(pred[s]) for s in SEEDS}
(ROOT/"V2_G5_EXTERNAL_EVIDENCE.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
print(json.dumps(report,indent=2))
