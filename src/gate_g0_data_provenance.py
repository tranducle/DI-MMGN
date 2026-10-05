import hashlib, json, re
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(r"D:\RESEARCH\DI_MM_V4")
BLUEPRINT=ROOT/"blueprint"/"base_snapshots.jsonl"
META_ROOT=ROOT/"acquisition"/"meta"
OUT_DIR=ROOT/"gates"
OUT_DIR.mkdir(parents=True,exist_ok=True)
REPORT=OUT_DIR/"G0_data_provenance.json"
INCLUDED=OUT_DIR/"G0_included_base_snapshots.jsonl"

MIN_SNAPSHOTS=600
MIN_ELIGIBLE_DOMAINS=150
MIN_COVERAGE=0.80
MAX_DUP_DOMAINS=5
MAX_DUP_FRACTION=0.02

def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()

def canonical_json_sha(obj):
    return sha256_bytes(json.dumps(obj,sort_keys=True,separators=(",",":")).encode())

def looks_html(b):
    h=b[:4096].lower()
    return b"<html" in h or b"<!doctype html" in h

def main():
    blueprint=[json.loads(x) for x in BLUEPRINT.read_text(encoding="utf-8").splitlines() if x.strip()]
    errors=[]
    warnings=[]
    valid=[]
    missing_meta=0
    bad=[]
    seen_keys=set()
    for row in blueprint:
        key=(row["domain"],row["requested_timestamp"])
        if key in seen_keys:
            errors.append(f"duplicate blueprint key {key}")
            continue
        seen_keys.add(key)
        mp=META_ROOT/row["domain"]/f"{row['requested_timestamp']}.json"
        if not mp.exists():
            missing_meta+=1
            continue
        try:
            m=json.loads(mp.read_text(encoding="utf-8"))
        except Exception as e:
            bad.append({"key":key,"reason":"meta_parse","detail":str(e)})
            continue
        if m.get("status")!="success":
            continue
        rp=ROOT/m["raw_path"]
        hp=ROOT/m["headers_path"]
        reasons=[]
        if not rp.exists(): reasons.append("raw_missing")
        if not hp.exists(): reasons.append("headers_missing")
        if reasons:
            bad.append({"key":key,"reason":";".join(reasons)})
            continue
        body=rp.read_bytes()
        if len(body)!=int(m.get("content_bytes",-1)): reasons.append("byte_count")
        if sha256_bytes(body)!=m.get("sha256"): reasons.append("raw_sha256")
        if int(m.get("http_status",-1))!=200: reasons.append("http_status")
        if not looks_html(body): reasons.append("html_signal")
        ect=m.get("effective_capture_timestamp")
        if not (isinstance(ect,str) and re.fullmatch(r"\d{14}",ect)): reasons.append("capture_timestamp")
        try:
            hobj=json.loads(hp.read_text(encoding="utf-8"))
            if canonical_json_sha(hobj)!=m.get("response_headers_sha256"): reasons.append("headers_sha256")
        except Exception:
            reasons.append("headers_parse")
        if reasons:
            bad.append({"key":key,"reason":";".join(reasons)})
            continue
        valid.append(m)

    hash_domains=defaultdict(set)
    hash_count=Counter()
    for m in valid:
        h=m["sha256"]
        hash_domains[h].add(m["domain"])
        hash_count[h]+=1
    suspicious=[]
    denom=max(len(valid),1)
    for h,n in hash_count.items():
        d=len(hash_domains[h])
        frac=n/denom
        if d>MAX_DUP_DOMAINS or frac>MAX_DUP_FRACTION:
            suspicious.append({"sha256":h,"records":n,"distinct_domains":d,"fraction":frac})
    suspicious_hashes={x["sha256"] for x in suspicious}
    included=[m for m in valid if m["sha256"] not in suspicious_hashes]
    per_domain=Counter(m["domain"] for m in included)
    eligible_domains=sorted([d for d,n in per_domain.items() if n>=2])
    eligible_set=set(eligible_domains)
    final=[m for m in included if m["domain"] in eligible_set]
    final.sort(key=lambda x:(x["domain"],x["effective_capture_timestamp"],x["requested_timestamp"]))

    with INCLUDED.open("w",encoding="utf-8",newline="\n") as f:
        for m in final:
            f.write(json.dumps(m,sort_keys=True,separators=(",",":"))+"\n")

    coverage=len(final)/len(blueprint) if blueprint else 0
    criteria={
        "final_valid_snapshots":{"value":len(final),"threshold":MIN_SNAPSHOTS,"pass":len(final)>=MIN_SNAPSHOTS},
        "eligible_domains_ge2":{"value":len(eligible_domains),"threshold":MIN_ELIGIBLE_DOMAINS,"pass":len(eligible_domains)>=MIN_ELIGIBLE_DOMAINS},
        "snapshot_coverage_fraction":{"value":coverage,"threshold":MIN_COVERAGE,"pass":coverage>=MIN_COVERAGE},
        "duplicate_blueprint_keys":{"value":len(errors),"threshold":0,"pass":len(errors)==0},
        "integrity_failures":{"value":len(bad),"threshold":"reported, excluded","pass":True},
        "placeholder_hash_groups":{"value":len(suspicious),"threshold":"excluded before coverage test","pass":True},
    }
    verdict="PASS" if all(x["pass"] for x in criteria.values()) else "FAIL_REDESIGN"
    report={
        "gate_id":"G0_data_provenance",
        "verdict":verdict,
        "blueprint_count":len(blueprint),
        "metadata_missing":missing_meta,
        "integrity_valid_before_placeholder_filter":len(valid),
        "placeholder_hash_groups":suspicious,
        "placeholder_records_excluded":sum(x["records"] for x in suspicious),
        "valid_after_placeholder_filter":len(included),
        "eligible_domain_rule":"at least 2 valid raw snapshots",
        "eligible_domain_count":len(eligible_domains),
        "final_included_snapshot_count":len(final),
        "final_coverage_fraction":coverage,
        "criteria":criteria,
        "integrity_failure_samples":bad[:50],
        "included_manifest":str(INCLUDED),
        "included_manifest_sha256":sha256_bytes(INCLUDED.read_bytes()),
        "decision":"proceed_to_family_split" if verdict=="PASS" else "stop_and_redesign_acquisition",
    }
    REPORT.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))
    return 0 if verdict=="PASS" else 3

if __name__=="__main__":
    raise SystemExit(main())
