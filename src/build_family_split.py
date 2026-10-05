import hashlib, json
from collections import defaultdict
from pathlib import Path
import tldextract

ROOT=Path(r"D:\RESEARCH\DI_MM_V4")
G0=ROOT/"gates"/"G0_data_provenance.json"
INP=ROOT/"gates"/"G0_included_base_snapshots.jsonl"
OUT_DIR=ROOT/"split"
OUT_DIR.mkdir(parents=True,exist_ok=True)
MANIFEST=OUT_DIR/"base_snapshots_family_split.jsonl"
SPLITS=OUT_DIR/"family_splits.json"
REPORT=ROOT/"gates"/"G1a_family_split_integrity.json"
TARGET={"pretrain":0.70,"val":0.10,"test":0.20}
ORDER=["pretrain","val","test"]
MAX_ABS_FRACTION_DEVIATION=0.03

extract=tldextract.TLDExtract(cache_dir=str(ROOT/"psl_cache"),suffix_list_urls=None)

def family_id(domain):
    x=extract(domain.lower().strip("."))
    return f"{x.domain}.{x.suffix}" if x.domain and x.suffix else domain.lower().strip(".")

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def best_subset(items, target):
    """Deterministic subset-sum nearest target.

    items: list[(family_id, weight)] sorted lexicographically.
    Returns tuple(selected family ids), achieved sum.
    """
    dp={0:()}
    for fid,w in items:
        prior=list(dp.items())
        for s,path in prior:
            ns=s+w
            cand=path+(fid,)
            if ns not in dp or cand<dp[ns]:
                dp[ns]=cand
    best_sum=min(dp, key=lambda s:(abs(s-target), s>target, abs(s), dp[s]))
    return dp[best_sum], best_sum

def main():
    g0=json.loads(G0.read_text(encoding="utf-8"))
    if g0.get("verdict")!="PASS":
        raise SystemExit("G0 is not PASS")

    rows=[json.loads(x) for x in INP.read_text(encoding="utf-8").splitlines() if x.strip()]
    families=defaultdict(list)
    for r in rows:
        fid=family_id(r["domain"])
        q=dict(r)
        q["family_id"]=fid
        families[fid].append(q)

    items=sorted((fid,len(rs)) for fid,rs in families.items())
    n=len(rows)

    val_target=round(n*TARGET["val"])
    test_target=round(n*TARGET["test"])

    val_families,val_count=best_subset(items,val_target)
    val_set=set(val_families)
    remaining=[x for x in items if x[0] not in val_set]

    test_families,test_count=best_subset(remaining,test_target)
    test_set=set(test_families)
    pretrain_families=tuple(fid for fid,_ in remaining if fid not in test_set)
    pretrain_count=n-val_count-test_count

    fam_assign={}
    for fid in pretrain_families: fam_assign[fid]="pretrain"
    for fid in val_families: fam_assign[fid]="val"
    for fid in test_families: fam_assign[fid]="test"

    assigned={
        "pretrain":list(pretrain_families),
        "val":list(val_families),
        "test":list(test_families),
    }

    out=[]
    for r in rows:
        q=dict(r)
        fid=family_id(q["domain"])
        q["family_id"]=fid
        q["split"]=fam_assign[fid]
        out.append(q)

    out.sort(key=lambda r:(ORDER.index(r["split"]),r["family_id"],r["domain"],r["effective_capture_timestamp"],r["requested_timestamp"]))
    with MANIFEST.open("w",encoding="utf-8",newline="\n") as f:
        for r in out:
            f.write(json.dumps(r,sort_keys=True,separators=(",",":"))+"\n")

    split_obj={
        "targets":TARGET,
        "target_counts":{"pretrain":n-val_target-test_target,"val":val_target,"test":test_target},
        "achieved_counts":{"pretrain":pretrain_count,"val":val_count,"test":test_count},
        "allocation_method":"deterministic family-level integer subset optimization",
        "tie_break":"lexicographic family id",
        "family_to_split":dict(sorted(fam_assign.items())),
        "families":{s:sorted(assigned[s]) for s in ORDER},
    }
    SPLITS.write_text(json.dumps(split_obj,indent=2)+"\n",encoding="utf-8")

    famsets={s:set(assigned[s]) for s in ORDER}
    domsets={s:set(r["domain"] for r in out if r["split"]==s) for s in ORDER}
    overlaps={"family":{},"domain":{}}
    for i,a in enumerate(ORDER):
        for b in ORDER[i+1:]:
            overlaps["family"][f"{a}__{b}"]=sorted(famsets[a]&famsets[b])
            overlaps["domain"][f"{a}__{b}"]=sorted(domsets[a]&domsets[b])

    summary={}
    maxdev=0.0
    for s in ORDER:
        c=sum(1 for r in out if r["split"]==s)
        frac=c/n if n else 0.0
        dev=frac-TARGET[s]
        maxdev=max(maxdev,abs(dev))
        summary[s]={
            "snapshots":c,
            "fraction":frac,
            "target_fraction":TARGET[s],
            "fraction_delta":dev,
            "domains":len(domsets[s]),
            "families":len(famsets[s]),
        }

    criteria={
        "zero_family_overlap":all(not v for v in overlaps["family"].values()),
        "zero_domain_overlap":all(not v for v in overlaps["domain"].values()),
        "max_abs_fraction_deviation_le_0.03":maxdev<=MAX_ABS_FRACTION_DEVIATION,
        "all_rows_preserved":len(out)==len(rows),
        "all_families_assigned_once":len(fam_assign)==len(families),
    }
    verdict="PASS" if all(criteria.values()) else "FAIL_REPAIR"

    report={
        "gate_id":"G1a_family_split_integrity",
        "verdict":verdict,
        "input_G0_sha256":sha(INP),
        "base_snapshot_count":n,
        "family_count":len(families),
        "domain_count":len(set(r["domain"] for r in out)),
        "allocation_method":"deterministic family-level integer subset optimization",
        "target_counts":{"pretrain":n-val_target-test_target,"val":val_target,"test":test_target},
        "achieved_counts":{"pretrain":pretrain_count,"val":val_count,"test":test_count},
        "split_summary":summary,
        "overlaps":overlaps,
        "max_abs_fraction_deviation":maxdev,
        "criteria":criteria,
        "manifest_path":str(MANIFEST),
        "manifest_sha256":sha(MANIFEST),
        "split_path":str(SPLITS),
        "split_sha256":sha(SPLITS),
        "decision":"proceed_to_variant_generation" if verdict=="PASS" else "stop_and_repair_split",
    }
    REPORT.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))
    return 0 if verdict=="PASS" else 3

if __name__=="__main__":
    raise SystemExit(main())
