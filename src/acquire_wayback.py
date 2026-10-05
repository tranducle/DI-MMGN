import argparse, concurrent.futures, hashlib, json, os, re, threading, time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests

ROOT=Path(r"D:\RESEARCH\DI_MM_V4")
BLUEPRINT=ROOT/"blueprint"/"base_snapshots.jsonl"
RAW=ROOT/"acquisition"/"raw"
META=ROOT/"acquisition"/"meta"
HEADERS=ROOT/"acquisition"/"headers"
LOG=ROOT/"acquisition"/"acquisition_events.jsonl"
MANIFEST=ROOT/"acquisition"/"acquisition_manifest.jsonl"
SUMMARY=ROOT/"acquisition"/"acquisition_summary.json"

UA="Mozilla/5.0 (compatible; LWDED-v4-research/1.0; archived-web reproducibility study)"
TIMEOUT=45
MAX_BYTES=10*1024*1024
MIN_BYTES=512
RETRIES=3
WORKERS=4
lock=threading.Lock()

def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()

def atomic_write_bytes(path,b):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_bytes(b)
    os.replace(tmp,path)

def atomic_write_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    os.replace(tmp,path)

def variants(domain):
    out=[f"https://{domain}/"]
    if not domain.startswith("www."):
        out.append(f"https://www.{domain}/")
    out.append(f"http://{domain}/")
    if not domain.startswith("www."):
        out.append(f"http://www.{domain}/")
    return out

def capture_ts(url):
    m=re.search(r"/web/(\d{14})[^/]*/",url or "")
    return m.group(1) if m else None

def looks_html(b,headers):
    ct=(headers.get("content-type") or "").lower()
    head=b[:4096].lower()
    return ("html" in ct) or (b"<html" in head) or (b"<!doctype html" in head)

def fetch_one(row,force=False):
    domain=row["domain"]
    ts=row["requested_timestamp"]
    key=f"{domain}/{ts}"
    mp=META/domain/f"{ts}.json"
    if mp.exists() and not force:
        try:
            old=json.loads(mp.read_text(encoding="utf-8"))
            if old.get("status")=="success":
                return old
        except Exception:
            pass
    event={"key":key,"domain":domain,"requested_timestamp":ts,"attempts":[]}
    sess=requests.Session()
    sess.headers.update({"User-Agent":UA,"Accept":"text/html,application/xhtml+xml;q=0.9,*/*;q=0.1"})
    for source_url in variants(domain):
        replay=f"https://web.archive.org/web/{ts}id_/{source_url}"
        for attempt in range(1,RETRIES+1):
            t0=time.time()
            rec={"source_url":source_url,"replay_url":replay,"attempt":attempt}
            try:
                resp=sess.get(replay,timeout=TIMEOUT,allow_redirects=True,stream=True)
                chunks=[]
                total=0
                too_big=False
                for chunk in resp.iter_content(chunk_size=65536):
                    if not chunk: continue
                    total+=len(chunk)
                    if total>MAX_BYTES:
                        too_big=True
                        break
                    chunks.append(chunk)
                body=b"".join(chunks)
                rec.update({
                    "http_status":resp.status_code,
                    "effective_url":resp.url,
                    "effective_capture_timestamp":capture_ts(resp.url),
                    "bytes":len(body),
                    "too_big":too_big,
                    "elapsed_seconds":round(time.time()-t0,3),
                })
                valid=(resp.status_code==200 and not too_big and len(body)>=MIN_BYTES and looks_html(body,resp.headers))
                rec["valid"]=bool(valid)
                event["attempts"].append(rec)
                if valid:
                    raw_path=RAW/domain/f"{ts}.html"
                    header_path=HEADERS/domain/f"{ts}.json"
                    atomic_write_bytes(raw_path,body)
                    hdr={str(k).lower():str(v) for k,v in resp.headers.items()}
                    hobj={
                        "requested_replay_url":replay,
                        "effective_url":resp.url,
                        "response_headers":hdr,
                        "history":[{"status":h.status_code,"url":h.url,"headers":{str(k).lower():str(v) for k,v in h.headers.items()}} for h in resp.history],
                    }
                    atomic_write_json(header_path,hobj)
                    meta={
                        **row,
                        "status":"success",
                        "requested_url":source_url,
                        "requested_replay_url":replay,
                        "effective_wayback_url":resp.url,
                        "effective_capture_timestamp":capture_ts(resp.url),
                        "http_status":resp.status_code,
                        "content_bytes":len(body),
                        "sha256":sha256_bytes(body),
                        "response_headers_sha256":sha256_bytes(json.dumps(hobj,sort_keys=True,separators=(",",":")).encode()),
                        "retrieved_at_utc":datetime.now(timezone.utc).isoformat(),
                        "raw_path":str(raw_path.relative_to(ROOT)).replace("\\","/"),
                        "headers_path":str(header_path.relative_to(ROOT)).replace("\\","/"),
                        "attempt_count":len(event["attempts"]),
                    }
                    atomic_write_json(mp,meta)
                    with lock:
                        LOG.parent.mkdir(parents=True,exist_ok=True)
                        with LOG.open("a",encoding="utf-8") as f:
                            f.write(json.dumps({"event":"success","meta":meta},sort_keys=True)+"\n")
                    return meta
            except Exception as e:
                rec.update({"error":type(e).__name__+": "+str(e),"elapsed_seconds":round(time.time()-t0,3),"valid":False})
                event["attempts"].append(rec)
            time.sleep(0.35*attempt)
    meta={
        **row,
        "status":"failed",
        "retrieved_at_utc":datetime.now(timezone.utc).isoformat(),
        "attempt_count":len(event["attempts"]),
        "attempts":event["attempts"],
    }
    atomic_write_json(mp,meta)
    with lock:
        LOG.parent.mkdir(parents=True,exist_ok=True)
        with LOG.open("a",encoding="utf-8") as f:
            f.write(json.dumps({"event":"failed","key":key,"attempts":event["attempts"]},sort_keys=True)+"\n")
    return meta

def rebuild_manifest(blueprint):
    metas=[]
    for row in blueprint:
        p=META/row["domain"]/f"{row['requested_timestamp']}.json"
        if p.exists():
            try: metas.append(json.loads(p.read_text(encoding="utf-8")))
            except Exception: pass
    metas.sort(key=lambda x:(x["domain"],x["requested_timestamp"]))
    MANIFEST.parent.mkdir(parents=True,exist_ok=True)
    with MANIFEST.open("w",encoding="utf-8",newline="\n") as f:
        for m in metas:
            f.write(json.dumps(m,sort_keys=True,separators=(",",":"))+"\n")
    good=[m for m in metas if m.get("status")=="success"]
    per={}
    for m in good: per[m["domain"]]=per.get(m["domain"],0)+1
    eligible=[d for d,n in per.items() if n>=2]
    summary={
        "blueprint_count":len(blueprint),
        "metadata_records":len(metas),
        "success_count":len(good),
        "failed_count":sum(1 for m in metas if m.get("status")=="failed"),
        "snapshot_coverage_fraction":len(good)/len(blueprint) if blueprint else 0,
        "success_domain_count":len(per),
        "eligible_domain_count_ge2":len(eligible),
        "eligible_domains":sorted(eligible),
        "manifest_sha256":hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "gate_thresholds":{"minimum_valid_raw_snapshots":600,"minimum_eligible_domains":150,"minimum_snapshot_coverage_fraction":0.80},
    }
    summary["gate_G0_precheck"]="PASS" if (summary["success_count"]>=600 and summary["eligible_domain_count_ge2"]>=150 and summary["snapshot_coverage_fraction"]>=0.80) else "FAIL_REPAIR"
    atomic_write_json(SUMMARY,summary)
    return summary

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--limit",type=int,default=0)
    ap.add_argument("--workers",type=int,default=WORKERS)
    ap.add_argument("--force",action="store_true")
    args=ap.parse_args()
    rows=[json.loads(x) for x in BLUEPRINT.read_text(encoding="utf-8").splitlines() if x.strip()]
    if args.limit>0: rows=rows[:args.limit]
    print(f"ACQUIRE rows={len(rows)} workers={args.workers}",flush=True)
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs=[ex.submit(fetch_one,r,args.force) for r in rows]
        for i,fut in enumerate(concurrent.futures.as_completed(futs),1):
            try:
                m=fut.result()
                results.append(m)
                if i%10==0 or i==len(futs):
                    ok=sum(1 for x in results if x.get("status")=="success")
                    print(f"progress {i}/{len(futs)} success={ok}",flush=True)
            except Exception as e:
                print("WORKER_ERROR",repr(e),flush=True)
    full=[json.loads(x) for x in BLUEPRINT.read_text(encoding="utf-8").splitlines() if x.strip()]
    summary=rebuild_manifest(full)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=="__main__":
    main()
