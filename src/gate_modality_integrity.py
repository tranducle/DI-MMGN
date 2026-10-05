import hashlib, json
from collections import Counter
from pathlib import Path
import numpy as np
import torch

ROOT=Path(r"D:\RESEARCH\DI_MM_V4")
V4=ROOT/"dataset_pipeline"/"v4"
SNAPS=V4/"snapshot_manifest.jsonl"
PAIRS=V4/"unified_pairs.jsonl"
REPORT=ROOT/"gates"/"G1f_modality_integrity.json"

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for c in iter(lambda:f.read(1024*1024),b""):h.update(c)
    return h.hexdigest()

def main():
    rows=[json.loads(x) for x in SNAPS.read_text(encoding="utf-8").splitlines() if x.strip()]
    pairs=[json.loads(x) for x in PAIRS.read_text(encoding="utf-8").splitlines() if x.strip()]
    errors=[]; counts=Counter()
    keys=set()
    for r in rows:
        key=(r["domain"],r["snapshot_id"])
        if key in keys: errors.append({"key":key,"error":"duplicate_snapshot_key"})
        keys.add(key)
        domain,sid=r["domain"],r["snapshot_id"]
        tp=V4/"text"/domain/f"{sid}.npy"
        gp=V4/"graphs"/domain/f"{sid}.pt"
        vp=V4/"visuals"/domain/f"{sid}.npy"
        hp=V4/"http"/domain/f"{sid}.npy"
        try:
            a=np.load(tp); counts["text"]+=1
            if a.shape!=(384,) or not np.isfinite(a).all(): errors.append({"key":key,"error":"bad_text","shape":list(a.shape)})
        except Exception as e: errors.append({"key":key,"error":"text:"+type(e).__name__})
        try:
            g=torch.load(gp,map_location="cpu",weights_only=False); counts["graph"]+=1
            x=g["x"]; ei=g["edge_index"]
            if x.ndim!=2 or x.shape[1]!=70 or x.shape[0]<1 or ei.ndim!=2 or ei.shape[0]!=2 or not torch.isfinite(x).all():
                errors.append({"key":key,"error":"bad_graph","x_shape":list(x.shape),"ei_shape":list(ei.shape)})
        except Exception as e: errors.append({"key":key,"error":"graph:"+type(e).__name__})
        try:
            a=np.load(vp); counts["visual"]+=1
            if a.shape!=(512,) or not np.isfinite(a).all(): errors.append({"key":key,"error":"bad_visual","shape":list(a.shape)})
        except Exception as e: errors.append({"key":key,"error":"visual:"+type(e).__name__})
        try:
            a=np.load(hp); counts["http"]+=1
            if a.shape!=(32,) or not np.isfinite(a).all(): errors.append({"key":key,"error":"bad_http","shape":list(a.shape)})
        except Exception as e: errors.append({"key":key,"error":"http:"+type(e).__name__})
    # sequence baseline bundle
    seqp=V4/"text_sequence_cache.npy"; maskp=V4/"text_sequence_mask.npy"; idxp=V4/"text_sequence_index.json"
    seq_info={}
    try:
        seq=np.load(seqp,mmap_mode="r"); mask=np.load(maskp,mmap_mode="r"); idx=json.loads(idxp.read_text(encoding="utf-8"))
        seq_info={"seq_shape":list(seq.shape),"mask_shape":list(mask.shape),"index_count":len(idx),"seq_sha256":sha(seqp),"mask_sha256":sha(maskp),"index_sha256":sha(idxp)}
        if tuple(seq.shape)!=(len(rows),512,384): errors.append({"error":"bad_sequence_shape","shape":list(seq.shape)})
        if tuple(mask.shape)!=(len(rows),512): errors.append({"error":"bad_mask_shape","shape":list(mask.shape)})
        if len(idx)!=len(rows): errors.append({"error":"bad_index_count","count":len(idx)})
        missing=[f"{d}|{sid}" for d,sid in keys if f"{d}|{sid}" not in idx]
        if missing: errors.append({"error":"sequence_index_missing","count":len(missing),"sample":missing[:20]})
    except Exception as e:
        errors.append({"error":"sequence_bundle:"+type(e).__name__+":"+str(e)})
    # pair endpoint paths and split consistency
    missing_paths=[]; bad_pair=0
    for p in pairs:
        for fld,dim in [("text_embedding",384),("delta_dom_graph",70),("delta_visual",512),("delta_http",32)]:
            for end in ("t1","t2"):
                fp=V4.parent/p[fld][end]
                if not fp.exists(): missing_paths.append(str(fp))
        if p["snapshot_id"]["t1"]==p["snapshot_id"]["t2"]: bad_pair+=1
    # Diagnostic diversity statistics. These are reported before training to detect
    # representation collapse without changing the frozen scientific protocol.
    diversity={}
    for name,subdir,suffix in [
        ("text","text",".npy"),
        ("visual","visuals",".npy"),
        ("http","http",".npy"),
        ("graph","graphs",".pt"),
    ]:
        hashes=[]
        for r in rows:
            fp=V4/subdir/r["domain"]/(r["snapshot_id"]+suffix)
            if fp.exists():
                hashes.append(sha(fp))
        freq=Counter(hashes)
        diversity[name]={
            "artifact_count":len(hashes),
            "unique_sha256_count":len(freq),
            "max_identical_count":max(freq.values()) if freq else 0,
            "max_identical_fraction":(max(freq.values())/len(hashes)) if hashes else None,
        }

    warnings=[]
    for name,d in diversity.items():
        if d["artifact_count"] and d["unique_sha256_count"]<=1:
            warnings.append(f"{name} representation collapsed to one exact artifact")
        if d["max_identical_fraction"] is not None and d["max_identical_fraction"]>0.95:
            warnings.append(f"{name} has >95% byte-identical artifacts")

    criteria={
        "all_snapshot_modalities_valid":len(errors)==0,
        "counts_match_snapshot_count":all(counts[k]==len(rows) for k in ["text","graph","visual","http"]),
        "all_pair_paths_exist":not missing_paths,
        "no_self_pairs":bad_pair==0,
        "no_exact_single_artifact_collapse":all(d["unique_sha256_count"]>1 for d in diversity.values()),
    }
    verdict="PASS" if all(criteria.values()) else "FAIL_REPAIR"
    rep={"gate_id":"G1f_modality_integrity","verdict":verdict,"snapshot_count":len(rows),"pair_count":len(pairs),"modality_counts":dict(counts),"sequence_bundle":seq_info,"diversity":diversity,"warnings":warnings,"missing_pair_paths_count":len(missing_paths),"missing_pair_path_sample":missing_paths[:30],"self_pair_count":bad_pair,"errors":errors[:100],"criteria":criteria,"snapshot_manifest_sha256":sha(SNAPS),"unified_pairs_sha256":sha(PAIRS),"decision":"proceed_to_training_smoke" if verdict=="PASS" else "stop_and_repair_modalities"}
    REPORT.write_text(json.dumps(rep,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(rep,indent=2))
    return 0 if verdict=="PASS" else 3
if __name__=="__main__": raise SystemExit(main())
