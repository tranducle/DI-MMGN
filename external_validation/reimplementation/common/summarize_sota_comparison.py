import json, pathlib, statistics as st, numpy as np
from sklearn.metrics import f1_score

BASE=pathlib.Path(r"C:\Users\Tran Duc Le\Documents\LOCAL-CODING-AGENT-WIN\DO-A-PAPER-v2\Papers\[IEEE ACCESS] - DI-MMGN_REVIEWED_20260923_1357\DI-MMGN-v4\external_validation\reimplementation")
DIM=pathlib.Path(r"D:\RESEARCH\DI_MM_V4\run_official\results\predictions")
MANIFEST=pathlib.Path(r"D:\RESEARCH\DI_MM_V4\external_candidates\sota_static\lwded_v4_static_current_manifest.jsonl")
SEEDS=[42,43,44]; B=20000; RNG=20260930

def read_jsonl(p): return [json.loads(x) for x in pathlib.Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]

def load_dimmgn(seed):
    x=json.loads((DIM/f"concat_none_seed{seed}.json").read_text(encoding="utf-8-sig"))["predictions"]
    return [{"pair_id":x["pair_id"][i],"label":int(x["label"][i]),"pred":int(x["pred"][i])} for i in range(len(x["pair_id"]))]

def load_comp(name,seed):
    return read_jsonl(BASE/name/"runs"/"official"/f"seed_{seed}"/"test_predictions.jsonl")

test=[r for r in read_jsonl(MANIFEST) if r["split"]=="test"]
fam={r["pair_id"]:r["family_id"] for r in test}
ids=[r["pair_id"] for r in test]
families=sorted({r["family_id"] for r in test})

def macro(rows):
    return float(f1_score([r["label"] for r in rows],[r["pred"] for r in rows],average="macro",zero_division=0))

def family_counts(rows):
    by={f:[0,0,0,0] for f in families} # tn,fp,fn,tp
    for r in rows:
        y=int(r["label"]); p=int(r["pred"]); c=by[fam[r["pair_id"]]]
        if y==0 and p==0:c[0]+=1
        elif y==0 and p==1:c[1]+=1
        elif y==1 and p==0:c[2]+=1
        else:c[3]+=1
    return np.asarray([by[f] for f in families],dtype=np.int64)

def mf_from_counts(c):
    tn,fp,fn,tp=[c[...,i].astype(float) for i in range(4)]
    f1p=np.divide(2*tp,2*tp+fp+fn,out=np.zeros_like(tp),where=(2*tp+fp+fn)!=0)
    f1n=np.divide(2*tn,2*tn+fp+fn,out=np.zeros_like(tn),where=(2*tn+fp+fn)!=0)
    return (f1p+f1n)/2

systems={"dimmgn_concat_none":{s:load_dimmgn(s) for s in SEEDS}}
for n in ["bilstm_efficientnet_2021","defacementfusion_2025"]:
    systems[n]={s:load_comp(n,s) for s in SEEDS}

# Alignment checks.
for n,runs in systems.items():
    for s,rows in runs.items():
        if [r["pair_id"] for r in rows]!=ids: raise RuntimeError(f"{n} seed{s} pair order mismatch")

summary={}
for n,runs in systems.items():
    vals=[macro(runs[s]) for s in SEEDS]
    summary[n]={"macro_f1_by_seed":dict(zip(map(str,SEEDS),vals)),"mean":st.mean(vals),"sd":st.stdev(vals)}

rng=np.random.default_rng(RNG)
seed_draw=rng.integers(0,3,size=(B,3))
fam_draw=rng.integers(0,len(families),size=(B,3,len(families)))

counts={}
for n,runs in systems.items():
    counts[n]=np.stack([family_counts(runs[s]) for s in SEEDS],axis=0)

comparisons={}
for comp in ["bilstm_efficientnet_2021","defacementfusion_2025"]:
    seed_delta=np.asarray([macro(systems["dimmgn_concat_none"][s])-macro(systems[comp][s]) for s in SEEDS])
    bd=np.empty(B,dtype=float)
    for start in range(0,B,1000):
        end=min(B,start+1000)
        sd=seed_draw[start:end]; fd=fam_draw[start:end]
        a=counts["dimmgn_concat_none"][sd[:,:,None],fd].sum(axis=(1,2))
        b=counts[comp][sd[:,:,None],fd].sum(axis=(1,2))
        bd[start:end]=mf_from_counts(a)-mf_from_counts(b)
    comparisons[comp]={
        "mean_seed_delta":float(seed_delta.mean()),"sd_seed_delta":float(seed_delta.std(ddof=1)),
        "positive_seeds":int((seed_delta>0).sum()),
        "bootstrap_mean_delta":float(bd.mean()),
        "bootstrap_ci95":[float(np.quantile(bd,0.025)),float(np.quantile(bd,0.975))],
        "bootstrap_prob_delta_gt_0":float(np.mean(bd>0))
    }

out={"status":"PASS","test_pairs":len(ids),"families":len(families),"seeds":SEEDS,
     "bootstrap":{"replicates":B,"rng_seed":RNG,"unit":"paired seed + held-out registrable-domain family"},
     "summary":summary,"comparisons_vs_dimmgn":comparisons}
path=BASE/"S5_SOTA_COMPARISON_SUMMARY.json"; path.write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")
print(json.dumps(out,indent=2),flush=True)
