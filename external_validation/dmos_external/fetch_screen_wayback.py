import argparse, hashlib, json, pathlib, time
import requests
from bs4 import BeautifulSoup

ROOT=pathlib.Path(r"D:\RESEARCH\DI_MM_V4\external_candidates\DMoS_external")
KEYWORDS=pathlib.Path(r"D:\RESEARCH\DI_MM_V4\external_candidates\DMoS\container\keywords_eng.list")
HTMLDIR=ROOT/"wayback_html"
HTMLDIR.mkdir(parents=True,exist_ok=True)
KWS=[x.strip().lower() for x in KEYWORDS.read_text(encoding="utf-8",errors="ignore").splitlines() if x.strip()]
SESSION=requests.Session()
SESSION.headers.update({"User-Agent":"DI-MMGN academic external-validation study; archival retrieval"})

def sha256_bytes(b): return hashlib.sha256(b).hexdigest()

def visible_text_bytes(raw):
    soup=BeautifulSoup(raw,"html.parser")
    for t in soup(["script","style","noscript","template"]): t.decompose()
    return " ".join(soup.stripped_strings).lower()

def kw_hits(text): return sum(1 for k in KWS if k in text)

def fetch_replay(ts,url,max_retry=8):
    replay=f"https://web.archive.org/web/{ts}id_/{url}"
    backoff=20
    for attempt in range(max_retry):
        try:
            r=SESSION.get(replay,timeout=(10,35),allow_redirects=True)
            if r.status_code==429:
                ra=r.headers.get("Retry-After"); wait=max(backoff,int(ra) if (ra and ra.isdigit()) else 0)
                print(f"REPLAY_429 wait={wait}s",flush=True); time.sleep(wait); backoff=min(backoff*2,300); continue
            if 500 <= r.status_code < 600:
                print(f"REPLAY_{r.status_code} wait={backoff}s",flush=True); time.sleep(backoff); backoff=min(backoff*2,180); continue
            r.raise_for_status()
            return r.content,{"final_url":r.url,"status_code":r.status_code,"content_type":r.headers.get("content-type")},None
        except Exception as e:
            if attempt==max_retry-1: return None,None,repr(e)
            print(f"REPLAY_ERR attempt={attempt+1} wait={backoff}s err={type(e).__name__}",flush=True)
            time.sleep(backoff); backoff=min(backoff*2,180)
    return None,None,"retry_exhausted"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--discovery",default=str(ROOT/"wayback_full_discovery.jsonl"))
    args=ap.parse_args()
    disc=pathlib.Path(args.discovery)
    rows=[json.loads(x) for x in disc.read_text(encoding="utf-8").splitlines() if x.strip()]
    out=ROOT/"dmos_temporal_candidates.jsonl"
    done={}
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                x=json.loads(line); done[x["id"]]=x
    with out.open("a",encoding="utf-8",newline="\n") as f:
        for i,r in enumerate(rows,1):
            if r["id"] in done:
                print(f"FETCH_SKIP {i}/{len(rows)} {r['id']}",flush=True); continue
            assessed=[]
            for cap in r.get("captures",[]):
                ts=cap["timestamp"]
                snap_url=cap.get("url") or ""
                marker=f"/web/{ts}/"
                url=snap_url.split(marker,1)[1] if marker in snap_url else (cap.get("available_api_variant") or snap_url)
                fp=HTMLDIR/f"{r['id']}__{ts}.html"
                meta=None; err=None
                if fp.exists() and fp.stat().st_size>0:
                    raw=fp.read_bytes(); meta={"reused":True}
                else:
                    raw,meta,err=fetch_replay(ts,url)
                    if raw:
                        fp.write_bytes(raw)
                    time.sleep(2.0)
                if not raw:
                    assessed.append({**cap,"fetch_error":err,"clean_eligible":False}); continue
                txt=visible_text_bytes(raw)
                hits=kw_hits(txt)
                item={**cap,
                      "archive_html_path":str(fp),
                      "archive_bytes":len(raw),
                      "archive_sha256":sha256_bytes(raw),
                      "visible_chars":len(txt),
                      "keyword_hits":hits,
                      "fetch_meta":meta,
                      "clean_eligible":bool(len(raw)>=1000 and len(txt)>=100 and hits==0)}
                assessed.append(item)
            clean=sorted([x for x in assessed if x.get("clean_eligible")],key=lambda x:x["timestamp"],reverse=True)
            pos_ok=bool(r.get("positive_keyword_hits",0)>=1 and len(clean)>=1)
            benign_ok=len(clean)>=2
            rec={**r,"assessed_captures":assessed,
                 "positive_pair_eligible":pos_ok,
                 "positive_t1":clean[0] if pos_ok else None,
                 "positive_t2":{"html_path":r["html_path"],"html_sha256":r["html_sha256"],"collection_iso":r["collection_iso"]} if pos_ok else None,
                 "benign_pair_eligible":benign_ok,
                 "benign_t1":clean[1] if benign_ok else None,
                 "benign_t2":clean[0] if benign_ok else None}
            f.write(json.dumps(rec,sort_keys=True,separators=(",",":"))+"\n"); f.flush()
            print(f"FETCH {i}/{len(rows)} id={r['id']} caps={len(assessed)} clean={len(clean)} pos={int(pos_ok)} benign={int(benign_ok)}",flush=True)
    recs=[json.loads(x) for x in out.read_text(encoding="utf-8").splitlines() if x.strip()]
    pos=[r for r in recs if r.get("positive_pair_eligible")]
    ben=[r for r in recs if r.get("benign_pair_eligible")]
    ph=sorted({r["host_hint"] for r in pos}); bh=sorted({r["host_hint"] for r in ben})
    if len(pos)>=100 and len(ph)>=8 and len(ben)>=100 and len(bh)>=8:
        verdict="PASS_FULL_COHORT"
    elif len(pos)>=50 and len(ph)>=5:
        verdict="PASS_ATTACK_RECALL_ONLY"
    else:
        verdict="FAIL_REDESIGN"
    summary={
        "gate":"G1C_DMoS_TEMPORAL_COHORT",
        "verdict":verdict,
        "source_records":len(recs),
        "positive_pairs":len(pos),"positive_host_groups":len(ph),"positive_hosts":ph,
        "benign_pairs":len(ben),"benign_host_groups":len(bh),"benign_hosts":bh,
        "predeclared_full_threshold":{"positive_pairs":100,"positive_hosts":8,"benign_pairs":100,"benign_hosts":8},
        "predeclared_recall_only_threshold":{"positive_pairs":50,"positive_hosts":5},
        "output":str(out),
    }
    (ROOT/"G1C_DMoS_TEMPORAL_COHORT.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2),flush=True)
    if verdict=="FAIL_REDESIGN": raise SystemExit(5)

if __name__=="__main__": main()

