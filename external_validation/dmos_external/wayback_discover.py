import argparse, datetime as dt, hashlib, json, pathlib, random, time
from urllib.parse import urlparse, urlunparse
import requests
from bs4 import BeautifulSoup

MANIFEST=pathlib.Path(r"D:\RESEARCH\DI_MM_V4\external_candidates\DMoS_external\dmos_real500_manifest.jsonl")
KEYWORDS=pathlib.Path(r"D:\RESEARCH\DI_MM_V4\external_candidates\DMoS\container\keywords_eng.list")
OUT=pathlib.Path(r"D:\RESEARCH\DI_MM_V4\external_candidates\DMoS_external")
OUT.mkdir(parents=True,exist_ok=True)

UA="DI-MMGN academic external-validation study; Wayback archival retrieval; contact via local research workflow"
SESSION=requests.Session()
SESSION.headers.update({"User-Agent":UA})

def load_rows():
    return [json.loads(x) for x in MANIFEST.read_text(encoding="utf-8").splitlines() if x.strip()]

def visible_text(path):
    raw=pathlib.Path(path).read_bytes()
    if not raw: return ""
    soup=BeautifulSoup(raw,"html.parser")
    for t in soup(["script","style","noscript","template"]): t.decompose()
    return " ".join(soup.stripped_strings).lower()

KWS=[x.strip().lower() for x in KEYWORDS.read_text(encoding="utf-8",errors="ignore").splitlines() if x.strip()]

def kw_hits_text(text):
    return sum(1 for k in KWS if k in text)

def url_variants(u):
    if not u: return []
    p=urlparse(u)
    vals=[]
    for scheme in [p.scheme or "http","https" if (p.scheme or "http")=="http" else "http"]:
        q=p._replace(scheme=scheme,fragment="")
        s=urlunparse(q)
        if s not in vals: vals.append(s)
    return vals

def parse_ts(s):
    return dt.datetime.strptime(s,"%Y%m%d%H%M%S").replace(tzinfo=dt.timezone.utc)

def fmt_ts(x):
    return x.astimezone(dt.timezone.utc).strftime("%Y%m%d%H%M%S")

def request_available(url,target,max_retry=8):
    api="https://archive.org/wayback/available"
    backoff=20
    for attempt in range(max_retry):
        try:
            r=SESSION.get(api,params={"url":url,"timestamp":target},timeout=(10,25))
            if r.status_code==429:
                ra=r.headers.get("Retry-After")
                wait=max(backoff,int(ra) if (ra and ra.isdigit()) else 0)
                print(f"WAYBACK_429 wait={wait}s url={url}",flush=True)
                time.sleep(wait); backoff=min(backoff*2,300); continue
            if 500 <= r.status_code < 600:
                print(f"WAYBACK_{r.status_code} wait={backoff}s url={url}",flush=True)
                time.sleep(backoff); backoff=min(backoff*2,180); continue
            r.raise_for_status()
            return r.json(),None
        except Exception as e:
            if attempt==max_retry-1: return None,repr(e)
            print(f"WAYBACK_ERR attempt={attempt+1} wait={backoff}s err={type(e).__name__}",flush=True)
            time.sleep(backoff); backoff=min(backoff*2,180)
    return None,"retry_exhausted"

def walk_pre_captures(row,max_caps=4,years_back=3):
    coll=dt.datetime.fromisoformat(row["collection_iso"]).astimezone()
    # archive timestamps are UTC; treat preserved mtime as local archive time then convert.
    coll_utc=coll.astimezone(dt.timezone.utc)
    oldest=coll_utc-dt.timedelta(days=365*years_back)
    captures=[]
    errors=[]
    for variant in url_variants(row["inferred_url"]):
        target=coll_utc-dt.timedelta(minutes=1)
        seen=set()
        for step in range(max_caps*3):
            if target < oldest or len(captures)>=max_caps: break
            data,err=request_available(variant,fmt_ts(target))
            if err:
                errors.append({"url":variant,"target":fmt_ts(target),"error":err})
                break
            snap=((data or {}).get("archived_snapshots") or {}).get("closest")
            if not snap:
                # Move substantially earlier and try again.
                target-=dt.timedelta(days=90)
                continue
            ts=snap.get("timestamp")
            if not ts or ts in seen:
                target-=dt.timedelta(days=30); continue
            seen.add(ts)
            try: t=parse_ts(ts)
            except Exception:
                target-=dt.timedelta(days=30); continue
            if t >= coll_utc:
                target-=dt.timedelta(days=30); continue
            if t < oldest: break
            captures.append({
                "timestamp":ts,
                "url":snap.get("url") or variant,
                "status":snap.get("status"),
                "available_api_variant":variant,
                "gap_days":(coll_utc-t).total_seconds()/86400.0,
            })
            target=t-dt.timedelta(days=1)
            time.sleep(1.5)
        if captures: break
    # chronological newest first, unique ts/url
    uniq={}
    for c in captures: uniq[(c["timestamp"],c["url"])]=c
    captures=sorted(uniq.values(),key=lambda x:x["timestamp"],reverse=True)[:max_caps]
    return captures,errors

def stratified(rows,n,seed=20261001):
    elig=[r for r in rows if r["html_bytes"]>0 and r.get("inferred_url")]
    byhost={}
    for r in elig: byhost.setdefault(r["host_hint"],[]).append(r)
    rng=random.Random(seed)
    selected=[]
    # one per host first
    for h in sorted(byhost):
        xs=byhost[h][:]
        rng.shuffle(xs); selected.append(xs[0])
    rest=[r for r in elig if r not in selected]
    # emphasize URL-source diversity without seeing Wayback outcomes
    rng.shuffle(rest)
    while len(selected)<min(n,len(elig)) and rest:
        selected.append(rest.pop())
    return selected[:n]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=["probe","full"],required=True)
    ap.add_argument("--sample-size",type=int,default=60)
    ap.add_argument("--max-captures",type=int,default=4)
    args=ap.parse_args()
    rows=load_rows()
    # Compute positive keyword hits deterministically before acquisition.
    for r in rows:
        r["positive_keyword_hits"]=kw_hits_text(visible_text(r["html_path"])) if r["html_bytes"]>0 else 0
    work=stratified(rows,args.sample_size) if args.mode=="probe" else [r for r in rows if r["html_bytes"]>0 and r.get("inferred_url")]
    out=OUT/(f"wayback_{args.mode}_discovery.jsonl")
    done={}
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                x=json.loads(line); done[x["id"]]=x
    with out.open("a",encoding="utf-8",newline="\n") as f:
        for i,r in enumerate(work,1):
            if r["id"] in done:
                print(f"DISCOVERY_SKIP {i}/{len(work)} {r['id']}",flush=True); continue
            caps,errs=walk_pre_captures(r,args.max_captures)
            rec={**r,"captures":caps,"discovery_errors":errs}
            f.write(json.dumps(rec,sort_keys=True,separators=(",",":"))+"\n"); f.flush()
            print(f"DISCOVERY {i}/{len(work)} id={r['id']} host={r['host_hint']} caps={len(caps)} kw={r['positive_keyword_hits']}",flush=True)
            time.sleep(1.5)
    recs=[json.loads(x) for x in out.read_text(encoding="utf-8").splitlines() if x.strip()]
    withcap=[r for r in recs if r.get("captures")]
    hosts=sorted({r["host_hint"] for r in withcap})
    coverage=len(withcap)/max(1,len(recs))
    if args.mode=="probe":
        if coverage>=0.20 and len(hosts)>=5: verdict="PASS"
        elif coverage>=0.10 and len(hosts)>=3: verdict="PASS_WITH_WARNINGS"
        else: verdict="FAIL_REDESIGN"
    else:
        verdict="DISCOVERY_COMPLETE"
    summary={
        "gate":"G1A_WAYBACK_FEASIBILITY" if args.mode=="probe" else "G1B_WAYBACK_DISCOVERY",
        "verdict":verdict,
        "records":len(recs),"with_pre_capture":len(withcap),"coverage":coverage,
        "host_groups_with_capture":len(hosts),"hosts":hosts,
        "positive_keyword_signal":sum(r.get("positive_keyword_hits",0)>0 for r in recs),
        "output":str(out),
    }
    sp=OUT/(f"G1A_wayback_probe.json" if args.mode=="probe" else "G1B_wayback_discovery.json")
    sp.write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2),flush=True)
    if args.mode=="probe" and verdict=="FAIL_REDESIGN": raise SystemExit(5)

if __name__=="__main__": main()
