import argparse, datetime as dt, hashlib, json, pathlib, time
from urllib.parse import urlsplit, urlunsplit
import requests
from bs4 import BeautifulSoup

ROOT=pathlib.Path(r"D:\RESEARCH\DI_MM_V4\external_candidates\DMoS_external_v2")
KEYWORDS=pathlib.Path(r"D:\RESEARCH\DI_MM_V4\external_candidates\DMoS\container\keywords_eng.list")
KWS=[x.strip().lower() for x in KEYWORDS.read_text(encoding="utf-8",errors="ignore").splitlines() if x.strip()]
SESSION=requests.Session()
SESSION.headers.update({"User-Agent":"DI-MMGN academic archival external-validation study; contact via paper authors"})
CDX="https://web.archive.org/cdx/search/cdx"

def sha_bytes(b): return hashlib.sha256(b).hexdigest()

def visible(raw):
    s=BeautifulSoup(raw,"html.parser")
    for t in s(["script","style","noscript","template"]): t.decompose()
    return " ".join(s.stripped_strings).lower()

def kw_hits(text): return sum(1 for k in KWS if k in text)

def variants(url):
    q=urlsplit(url)
    if not q.scheme or not q.netloc:
        return [url]
    vals=[]
    def add(scheme,host,path,query):
        u=urlunsplit((scheme,host,path,query,""))
        if u not in vals: vals.append(u)
    add(q.scheme,q.netloc,q.path,q.query)
    add("https" if q.scheme=="http" else "http",q.netloc,q.path,q.query)
    host=q.netloc
    toggled=host[4:] if host.lower().startswith("www.") else "www."+host
    add(q.scheme,toggled,q.path,q.query)
    add("https" if q.scheme=="http" else "http",toggled,q.path,q.query)
    if not q.query:
        p=q.path or "/"
        p2=p[:-1] if len(p)>1 and p.endswith("/") else (p+"/" if not p.endswith("/") else p)
        add(q.scheme,q.netloc,p2,q.query)
        add("https" if q.scheme=="http" else "http",q.netloc,p2,q.query)
    return vals

def get_retry(url,params=None,timeout=(10,40),max_retry=7,kind="GET"):
    backoff=10
    last=None
    for a in range(max_retry):
        try:
            r=SESSION.get(url,params=params,timeout=timeout,allow_redirects=True)
            if r.status_code==429 or 500<=r.status_code<600:
                wait=backoff
                ra=r.headers.get("Retry-After")
                if ra and ra.isdigit(): wait=max(wait,int(ra))
                print(f"{kind}_{r.status_code} attempt={a+1} wait={wait}s",flush=True)
                time.sleep(wait); backoff=min(backoff*2,180); last=f"HTTP {r.status_code}"; continue
            r.raise_for_status()
            return r,None
        except Exception as e:
            last=repr(e)
            if a==max_retry-1: break
            print(f"{kind}_ERR attempt={a+1} wait={backoff}s err={type(e).__name__}",flush=True)
            time.sleep(backoff); backoff=min(backoff*2,180)
    return None,last or "retry_exhausted"

def parse_iso(s):
    return dt.datetime.fromisoformat(s.replace("Z","+00:00")).replace(tzinfo=None)

def ymd(d): return d.strftime("%Y%m%d%H%M%S")

def subtract_years(d,years):
    try: return d.replace(year=d.year-years)
    except ValueError: return d.replace(year=d.year-years,day=28)

def query_variant(url,from_ts,to_ts):
    params=[("url",url),("output","json"),("fl","timestamp,original,statuscode,digest"),("filter","statuscode:200"),
            ("from",from_ts),("to",to_ts),("collapse","digest"),("limit","5000")]
    r,err=get_retry(CDX,params=params,kind="CDX")
    if r is None: return [],err
    try:
        j=r.json()
        if not isinstance(j,list) or len(j)<1: return [],None
        hdr=j[0]
        out=[]
        for row in j[1:]:
            if len(row)!=len(hdr): continue
            x=dict(zip(hdr,row))
            if x.get("timestamp") and x.get("original"):
                out.append(x)
        return out,None
    except Exception as e:
        return [],"cdx_parse:"+repr(e)

def select_uniform(caps,k):
    caps=sorted(caps,key=lambda x:x["timestamp"])
    n=len(caps)
    if n<=k: return caps
    idx=[]
    for i in range(k):
        j=round(i*(n-1)/(k-1))
        if j not in idx: idx.append(j)
    return [caps[j] for j in idx]

def discover(r,max_caps,years):
    coll=parse_iso(r["collection_iso"])
    start=subtract_years(coll,years)
    merged=[];errors=[];vars_=variants(r["inferred_url"])
    for u in vars_:
        rows,err=query_variant(u,ymd(start),ymd(coll))
        if err: errors.append({"variant":u,"error":err})
        for x in rows:
            x={**x,"query_variant":u}
            merged.append(x)
        time.sleep(0.5)
    uniq={}
    for x in sorted(merged,key=lambda z:z["timestamp"]):
        key=(x.get("digest") or "",x["timestamp"],x["original"])
        if x.get("digest"):
            key=("digest",x["digest"])
        uniq[key]=x
    allcaps=sorted(uniq.values(),key=lambda x:x["timestamp"])
    selected=select_uniform(allcaps,max_caps)
    return vars_,allcaps,selected,errors

def replay(cap,current_sha,outdir):
    ts=cap["timestamp"]; original=cap["original"]
    replay_url=f"https://web.archive.org/web/{ts}id_/{original}"
    rr,err=get_retry(replay_url,kind="REPLAY")
    if rr is None:
        return {**cap,"replay_url":replay_url,"fetch_error":err,"clean_eligible":False}
    raw=rr.content
    txt=visible(raw)
    hits=kw_hits(txt)
    sh=sha_bytes(raw)
    outdir.mkdir(parents=True,exist_ok=True)
    fp=outdir/f"{ts}.html"
    if not fp.exists(): fp.write_bytes(raw)
    clean=bool(rr.status_code==200 and len(raw)>=1000 and len(txt)>=100 and hits==0 and sh!=current_sha)
    return {**cap,"replay_url":replay_url,"final_url":rr.url,"http_status":rr.status_code,
            "content_type":rr.headers.get("content-type"),"archive_html_path":str(fp),
            "archive_bytes":len(raw),"archive_sha256":sh,"visible_chars":len(txt),
            "keyword_hits":hits,"fetch_error":None,"clean_eligible":clean}

def load_rows(p):
    return [json.loads(x) for x in pathlib.Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--pool",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--limit",type=int,default=0)
    ap.add_argument("--ids",default="")
    ap.add_argument("--max-captures",type=int,default=24)
    ap.add_argument("--years",type=int,default=10)
    args=ap.parse_args()
    rows=load_rows(args.pool)
    ids=[x for x in args.ids.split(",") if x]
    if ids:
        order={x:i for i,x in enumerate(ids)}
        rows=sorted([r for r in rows if r["id"] in order],key=lambda r:order[r["id"]])
    if args.limit>0: rows=rows[:args.limit]
    out=pathlib.Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    done={}
    if out.exists():
        for z in out.read_text(encoding="utf-8").splitlines():
            if z.strip():
                q=json.loads(z);done[q["id"]]=q
    mode="a" if out.exists() else "w"
    with out.open(mode,encoding="utf-8",newline="\n") as f:
        for i,r in enumerate(rows,1):
            if r["id"] in done:
                print(f"SALVAGE_SKIP {i}/{len(rows)} {r['id']}",flush=True);continue
            vars_,allcaps,selected,disc_err=discover(r,args.max_captures,args.years)
            assessed=[];clean=[]
            for cap in sorted(selected,key=lambda x:x["timestamp"],reverse=True):
                a=replay(cap,r["html_sha256"],ROOT/"html"/r["id"])
                assessed.append(a)
                if a.get("clean_eligible"): clean.append(a)
                print(f"REPLAY_RESULT id={r['id']} ts={cap['timestamp']} clean={int(a.get('clean_eligible',False))} kw={a.get('keyword_hits')} bytes={a.get('archive_bytes')}",flush=True)
                time.sleep(1.5)
                if len(clean)>=2: break
            clean=sorted(clean,key=lambda x:x["timestamp"],reverse=True)
            rec={**r,"url_variants":vars_,"discovery_errors":disc_err,
                 "timeline_unique_captures":len(allcaps),"selected_capture_count":len(selected),
                 "selected_captures":selected,"assessed_captures_v2":assessed,
                 "clean_capture_count":len(clean),
                 "positive_pair_eligible_v2":bool(r.get("positive_keyword_hits",0)>=1 and len(clean)>=1),
                 "positive_t1_v2":clean[0] if clean else None,
                 "positive_t2_v2":{"html_path":r["html_path"],"html_sha256":r["html_sha256"],"collection_iso":r["collection_iso"]} if clean else None,
                 "benign_pair_eligible_v2":bool(len(clean)>=2),
                 "benign_t1_v2":clean[1] if len(clean)>=2 else None,
                 "benign_t2_v2":clean[0] if len(clean)>=2 else None}
            f.write(json.dumps(rec,sort_keys=True,separators=(",",":"))+"\n");f.flush()
            print(f"SALVAGE {i}/{len(rows)} id={r['id']} timeline={len(allcaps)} selected={len(selected)} assessed={len(assessed)} clean={len(clean)} pos={int(rec['positive_pair_eligible_v2'])} benign={int(rec['benign_pair_eligible_v2'])}",flush=True)

if __name__=="__main__": main()
