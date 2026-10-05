import hashlib, json
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(r"D:\RESEARCH\DI_MM_V4")
V4=ROOT/"dataset_pipeline"/"v4"
SNAPS=V4/"snapshot_manifest.jsonl"
G1A=ROOT/"gates"/"G1a_family_split_integrity.json"
G1B=ROOT/"gates"/"G1b_variant_generation.json"
PAIRS=V4/"unified_pairs.jsonl"
REPORT=ROOT/"gates"/"G1_preprocessing_integrity.json"
ORDER=["pretrain","val","test"]
SYNTH_TYPES=["benign_attr","benign_wrapper","overt","hidden","semantic","structural"]

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def mod_paths(domain,sid):
    base=f"v4"
    return {
        "text_embedding":f"{base}/text/{domain}/{sid}.npy",
        "dom_graph":f"{base}/graphs/{domain}/{sid}.pt",
        "visual":f"{base}/visuals/{domain}/{sid}.npy",
        "http":f"{base}/http/{domain}/{sid}.npy",
    }

def pair_id(domain,t1,t2,label,attack,source):
    s="|".join([domain,t1,t2,label,attack or "",source])
    return hashlib.sha256(s.encode()).hexdigest()[:20]

def make_pair(domain,family,split,t1,t2,label,attack,source):
    a=mod_paths(domain,t1); b=mod_paths(domain,t2)
    return {
        "pair_id":pair_id(domain,t1,t2,label,attack,source),
        "domain":domain,"family_id":family,"split":split,
        "label":label,"attack_type":attack,"source_type":source,
        "snapshot_id":{"t1":t1,"t2":t2},
        "text_embedding":{"t1":a["text_embedding"],"t2":b["text_embedding"]},
        "delta_dom_graph":{"t1":a["dom_graph"],"t2":b["dom_graph"]},
        "delta_visual":{"t1":a["visual"],"t2":b["visual"]},
        "delta_http":{"t1":a["http"],"t2":b["http"]},
    }

def main():
    if json.loads(G1A.read_text(encoding="utf-8")).get("verdict")!="PASS":
        raise SystemExit("G1a not PASS")
    if json.loads(G1B.read_text(encoding="utf-8")).get("verdict")!="PASS":
        raise SystemExit("G1b not PASS")
    snaps=[json.loads(x) for x in SNAPS.read_text(encoding="utf-8").splitlines() if x.strip()]
    by_domain=defaultdict(list)
    lookup={}
    for s in snaps:
        by_domain[s["domain"]].append(s)
        lookup[(s["domain"],s["snapshot_id"])]=s
    pairs=[]
    raw_count=0
    organic_count=0
    for domain,rows in sorted(by_domain.items()):
        raw=[r for r in rows if r["variant_type"]=="raw"]
        raw.sort(key=lambda r:(r["effective_capture_timestamp"],r["requested_timestamp"],r["snapshot_id"]))
        raw_count+=len(raw)
        # organic legitimate temporal transitions
        for a,b in zip(raw,raw[1:]):
            if a["split"]!=b["split"] or a["family_id"]!=b["family_id"]:
                raise RuntimeError("same-domain raw snapshots crossed split/family")
            pairs.append(make_pair(domain,a["family_id"],a["split"],a["snapshot_id"],b["snapshot_id"],"legitimate",None,"wayback_clean_to_clean"))
            organic_count+=1
        # synthetic pairs from each raw base
        for r in raw:
            ts=r["requested_timestamp"]
            for typ in SYNTH_TYPES:
                sid=f"{typ}__{ts}"
                v=lookup.get((domain,sid))
                if not v:
                    raise RuntimeError(f"missing variant {domain} {sid}")
                if v["split"]!=r["split"] or v["family_id"]!=r["family_id"]:
                    raise RuntimeError("variant crossed split/family")
                if typ.startswith("benign_"):
                    pairs.append(make_pair(domain,r["family_id"],r["split"],r["snapshot_id"],sid,"legitimate",None,"synthetic_benign"))
                else:
                    pairs.append(make_pair(domain,r["family_id"],r["split"],r["snapshot_id"],sid,"defaced",typ,"synthetic_defaced"))
    pairs.sort(key=lambda r:(ORDER.index(r["split"]),r["family_id"],r["domain"],r["pair_id"]))
    with PAIRS.open("w",encoding="utf-8",newline="\n") as f:
        for r in pairs:
            f.write(json.dumps(r,sort_keys=True,separators=(",",":"))+"\n")
    # Publish the frozen family split alongside the v4 dataset for training provenance.
    split_src=ROOT/"split"/"family_splits.json"
    split_dst=V4/"splits.json"
    split_dst.write_bytes(split_src.read_bytes())

    ids=[r["pair_id"] for r in pairs]
    famsets={s:set(r["family_id"] for r in pairs if r["split"]==s) for s in ORDER}
    domsets={s:set(r["domain"] for r in pairs if r["split"]==s) for s in ORDER}
    snapsets={s:set((r["domain"],x) for r in pairs if r["split"]==s for x in r["snapshot_id"].values()) for s in ORDER}
    overlaps={"family":{},"domain":{},"snapshot":{}}
    for i,a in enumerate(ORDER):
        for b in ORDER[i+1:]:
            k=f"{a}__{b}"
            overlaps["family"][k]=sorted(famsets[a]&famsets[b])
            overlaps["domain"][k]=sorted(domsets[a]&domsets[b])
            overlaps["snapshot"][k]=len(snapsets[a]&snapsets[b])
    split_summary={}
    for s in ORDER:
        rr=[r for r in pairs if r["split"]==s]
        split_summary[s]={
            "pairs":len(rr),
            "labels":dict(Counter(r["label"] for r in rr)),
            "attack_types":{str(k):v for k,v in Counter(r.get("attack_type") for r in rr).items()},
            "source_types":dict(Counter(r["source_type"] for r in rr)),
            "domains":len(domsets[s]),"families":len(famsets[s]),
        }
    expected=raw_count*6+organic_count
    criteria={
        "pair_count_matches_construction":len(pairs)==expected,
        "pair_ids_unique":len(ids)==len(set(ids)),
        "zero_family_overlap":all(not v for v in overlaps["family"].values()),
        "zero_domain_overlap":all(not v for v in overlaps["domain"].values()),
        "zero_snapshot_overlap":all(v==0 for v in overlaps["snapshot"].values()),
        "both_labels_in_every_split":all(set(split_summary[s]["labels"])=={"legitimate","defaced"} for s in ORDER),
        "all_four_attack_types_in_every_split":all(all(t in split_summary[s]["attack_types"] for t in ["overt","hidden","semantic","structural"]) for s in ORDER),
    }
    verdict="PASS" if all(criteria.values()) else "FAIL_REPAIR"
    report={
        "gate_id":"G1_preprocessing_integrity",
        "verdict":verdict,
        "raw_base_snapshots":raw_count,
        "organic_clean_to_clean_pairs":organic_count,
        "synthetic_pairs":raw_count*6,
        "total_pairs":len(pairs),
        "expected_pairs":expected,
        "split_summary":split_summary,
        "overlaps":overlaps,
        "criteria":criteria,
        "unified_pairs":str(PAIRS),
        "unified_pairs_sha256":sha(PAIRS),
        "snapshot_manifest_sha256":sha(SNAPS),
        "decision":"proceed_to_modality_generation" if verdict=="PASS" else "stop_and_repair_preprocessing",
    }
    REPORT.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))
    return 0 if verdict=="PASS" else 3

if __name__=="__main__":
    raise SystemExit(main())
