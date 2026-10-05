import argparse, json, pathlib, hashlib
from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score,confusion_matrix

ROOT=pathlib.Path(r"C:\Users\Tran Duc Le\Documents\LOCAL-CODING-AGENT-WIN\DO-A-PAPER-v2\Papers\[IEEE ACCESS] - DI-MMGN_REVIEWED_20260923_1357\DI-MMGN-v4\external_validation\reimplementation")
MANIFEST=pathlib.Path(r"D:\RESEARCH\DI_MM_V4\external_candidates\sota_static\lwded_v4_static_current_manifest.jsonl")
SEEDS=[42,43,44]

def load_jsonl(p):
    return [json.loads(x) for x in pathlib.Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]

def metrics(rows):
    y=[int(r["label"]) for r in rows]; p=[int(r["pred"]) for r in rows]
    tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel()
    return {
        "macro_f1":float(f1_score(y,p,average="macro",zero_division=0)),
        "defaced_f1":float(f1_score(y,p,pos_label=1,average="binary",zero_division=0)),
        "accuracy":float(accuracy_score(y,p)),
        "precision":float(precision_score(y,p,zero_division=0)),
        "attack_recall":float(recall_score(y,p,zero_division=0)),
        "legitimate_specificity":float(tn/(tn+fp)) if tn+fp else float("nan"),
        "tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)
    }

def close(a,b,tol=1e-12):
    return abs(float(a)-float(b))<=tol

def check_model(name):
    if name=="bilstm_efficientnet_2021":
        d=ROOT/name/"runs"/"official"; field="test_fusion"
    elif name=="defacementfusion_2025":
        d=ROOT/name/"runs"/"official"; field="test"
    else: raise ValueError(name)
    test_manifest=[r for r in load_jsonl(MANIFEST) if r["split"]=="test"]
    ref_ids=[r["pair_id"] for r in test_manifest]
    ref_labels=[int(r["label"]) for r in test_manifest]
    out={"model":name,"status":"PASS","seeds":{}}
    for seed in SEEDS:
        rd=d/f"seed_{seed}"
        rp=rd/"result.json"; pp=rd/"test_predictions.jsonl"
        if not rp.exists() or not pp.exists():
            out["status"]="FAIL"; out["seeds"][str(seed)]={"error":"missing result or predictions"}; continue
        res=json.loads(rp.read_text(encoding="utf-8"))
        rows=load_jsonl(pp)
        errs=[]
        if len(rows)!=936: errs.append(f"prediction_count={len(rows)}")
        if [r["pair_id"] for r in rows]!=ref_ids: errs.append("pair_id/order mismatch")
        if [int(r["label"]) for r in rows]!=ref_labels: errs.append("label mismatch")
        if len(set(r["pair_id"] for r in rows))!=len(rows): errs.append("duplicate pair ids")
        rec=metrics(rows); stored=res.get(field,{})
        diffs={k:abs(float(rec[k])-float(stored[k])) for k in ["macro_f1","defaced_f1","accuracy","precision","attack_recall","legitimate_specificity"] if k in stored}
        if any(v>1e-12 for v in diffs.values()): errs.append("stored/recomputed metric mismatch")
        if any(not (0.0<=rec[k]<=1.0) for k in ["macro_f1","defaced_f1","accuracy","precision","attack_recall","legitimate_specificity"]):
            errs.append("metric out of range")
        out["seeds"][str(seed)]={"status":"PASS" if not errs else "FAIL","metrics":rec,"max_metric_diff":max(diffs.values()) if diffs else None,
                                 "errors":errs,"prediction_path":str(pp)}
        if errs: out["status"]="FAIL"
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--model",choices=["bilstm_efficientnet_2021","defacementfusion_2025","both"],default="both")
    ap.add_argument("--out",default=str(ROOT/"S4_SOTA_RESULT_INTEGRITY.json")); args=ap.parse_args()
    names=["bilstm_efficientnet_2021","defacementfusion_2025"] if args.model=="both" else [args.model]
    result={"gate":"S4_SOTA_RESULT_INTEGRITY","status":"PASS","models":{}}
    for n in names:
        x=check_model(n); result["models"][n]=x
        if x["status"]!="PASS": result["status"]="FAIL"
    pathlib.Path(args.out).write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2),flush=True)
    raise SystemExit(0 if result["status"]=="PASS" else 3)
if __name__=="__main__": main()
