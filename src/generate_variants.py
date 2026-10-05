import hashlib, json, os, random, re, shutil
from pathlib import Path
from bs4 import BeautifulSoup, Comment

ROOT=Path(r"D:\RESEARCH\DI_MM_V4")
G1=ROOT/"gates"/"G1a_family_split_integrity.json"
BASE=ROOT/"split"/"base_snapshots_family_split.jsonl"
V4=ROOT/"dataset_pipeline"/"v4"
HTML=V4/"html"
MANIFEST=V4/"snapshot_manifest.jsonl"
SUMMARY=ROOT/"gates"/"G1b_variant_generation.json"
SEED=20260928

MESSAGES=[
    "This page is no longer under its original control.",
    "The original content on this page has been replaced.",
    "This page has been altered without authorization.",
    "The published content on this page has been changed.",
    "The original page content is temporarily unavailable.",
]

def hseed(*parts):
    h=hashlib.sha256(("|".join(map(str,parts))+"|"+str(SEED)).encode()).digest()
    return int.from_bytes(h[:8],"big")

def sha(b):
    return hashlib.sha256(b).hexdigest()

def parse(raw):
    try:
        return BeautifulSoup(raw,"lxml")
    except Exception:
        return BeautifulSoup(raw,"html.parser")

def ensure_html_body(s):
    if s.html is None:
        html=s.new_tag("html")
        for x in list(s.contents): html.append(x.extract())
        s.append(html)
    if s.body is None:
        body=s.new_tag("body")
        for x in list(s.html.contents):
            if getattr(x,"name",None) not in ("head",):
                body.append(x.extract())
        s.html.append(body)
    return s

def common_normalize(s):
    s=ensure_html_body(s)
    for c in s.find_all(string=lambda x:isinstance(x,Comment)):
        c.extract()
    for tag in s.find_all(True):
        if tag.attrs:
            tag.attrs=dict(sorted(tag.attrs.items(),key=lambda kv:kv[0]))
    return s

def serialize(s):
    return str(s).encode("utf-8",errors="ignore")

def direct_body_tags(s):
    return [x for x in s.body.children if getattr(x,"name",None)]

def benign_attr(raw,rng):
    s=common_normalize(parse(raw))
    imgs=s.find_all("img")
    if imgs:
        k=max(1,min(len(imgs),max(1,len(imgs)//5)))
        idx=list(range(len(imgs))); rng.shuffle(idx)
        for i in idx[:k]:
            if "loading" not in imgs[i].attrs:
                imgs[i]["loading"]="lazy"
    return serialize(s)

def benign_wrapper(raw,rng):
    s=common_normalize(parse(raw))
    kids=direct_body_tags(s)
    if len(kids)>=2:
        start=rng.randrange(0,max(1,len(kids)-1))
        length=min(max(2,1+len(kids)//4),len(kids)-start)
        chosen=kids[start:start+length]
        wrapper=s.new_tag("section")
        chosen[0].insert_before(wrapper)
        for x in chosen: wrapper.append(x.extract())
    else:
        wrapper=s.new_tag("section")
        while s.body.contents:
            wrapper.append(s.body.contents[0].extract())
        s.body.append(wrapper)
    return serialize(s)

def overt(raw,rng):
    s=common_normalize(parse(raw))
    banner=s.new_tag("div")
    msg=MESSAGES[rng.randrange(len(MESSAGES))]
    h=s.new_tag("h1"); h.string=msg
    p=s.new_tag("p"); p.string="Please verify the page contents before relying on this site."
    banner.append(h); banner.append(p)
    s.body.insert(0,banner)
    kids=direct_body_tags(s)
    # Remove a bounded subset of original visible sections, never the injected banner.
    originals=[x for x in kids if x is not banner]
    rng.shuffle(originals)
    remove_n=min(len(originals)//3,4)
    for x in originals[:remove_n]: x.decompose()
    return serialize(s)

def hidden(raw,rng):
    s=common_normalize(parse(raw))
    box=s.new_tag("div")
    box["hidden"]=""
    span=s.new_tag("span"); span.string=MESSAGES[rng.randrange(len(MESSAGES))]
    frame=s.new_tag("iframe"); frame["src"]="about:blank"; frame["hidden"]=""
    box.append(span); box.append(frame)
    s.body.append(box)
    return serialize(s)

def semantic(raw,rng):
    s=common_normalize(parse(raw))
    msg=MESSAGES[rng.randrange(len(MESSAGES))]
    candidates=[]
    if s.title: candidates.append(s.title)
    candidates.extend(s.find_all(["h1","h2","p"],limit=20))
    if candidates:
        rng.shuffle(candidates)
        k=max(1,min(3,len(candidates)))
        for i,t in enumerate(candidates[:k]):
            t.clear()
            t.append(msg if i==0 else "The expected information on this page has been changed.")
    else:
        p=s.new_tag("p"); p.string=msg; s.body.insert(0,p)
    return serialize(s)

def structural(raw,rng):
    s=common_normalize(parse(raw))
    kids=direct_body_tags(s)
    if len(kids)>=3:
        start=rng.randrange(0,max(1,len(kids)-2))
        length=min(max(3,len(kids)//3),len(kids)-start)
        chosen=kids[start:start+length]
        holder=s.new_tag("div")
        inner=s.new_tag("div")
        chosen[0].insert_before(holder)
        holder.append(inner)
        for x in reversed(chosen): inner.append(x.extract())
    else:
        holder=s.new_tag("div")
        while s.body.contents:
            holder.append(s.body.contents[0].extract())
        s.body.append(holder)
    overlay=s.new_tag("aside")
    overlay["role"]="status"
    p=s.new_tag("p"); p.string=MESSAGES[rng.randrange(len(MESSAGES))]
    overlay.append(p)
    s.body.insert(0,overlay)
    return serialize(s)

OPS={
    "benign_attr":("legitimate",None,"synthetic_benign",benign_attr),
    "benign_wrapper":("legitimate",None,"synthetic_benign",benign_wrapper),
    "overt":("defaced","overt","synthetic_defaced",overt),
    "hidden":("defaced","hidden","synthetic_defaced",hidden),
    "semantic":("defaced","semantic","synthetic_defaced",semantic),
    "structural":("defaced","structural","synthetic_defaced",structural),
}

def main():
    g1=json.loads(G1.read_text(encoding="utf-8"))
    if g1.get("verdict")!="PASS":
        raise SystemExit("G1a is not PASS")
    bases=[json.loads(x) for x in BASE.read_text(encoding="utf-8").splitlines() if x.strip()]
    rows=[]
    failures=[]
    exact_same=0
    for n,b in enumerate(bases,1):
        domain=b["domain"]; ts=b["requested_timestamp"]
        raw_src=ROOT/b["raw_path"]
        raw=raw_src.read_bytes()
        domdir=HTML/domain
        domdir.mkdir(parents=True,exist_ok=True)
        raw_id=f"raw__{ts}"
        raw_dst=domdir/f"{raw_id}.html"
        if not raw_dst.exists() or sha(raw_dst.read_bytes())!=sha(raw):
            raw_dst.write_bytes(raw)
        rows.append({
            "domain":domain,"family_id":b["family_id"],"split":b["split"],
            "snapshot_id":raw_id,"variant_type":"raw","pair_label":None,"attack_type":None,
            "source_type":"wayback_raw","base_snapshot_id":raw_id,
            "html_path":str(raw_dst.relative_to(ROOT)).replace("\\","/"),
            "html_sha256":sha(raw),"bytes":len(raw),
            "requested_timestamp":ts,"effective_capture_timestamp":b["effective_capture_timestamp"],
            "acquisition_meta_path":str((Path(b["raw_path"]).parent.parent.parent/"meta"/domain/f"{ts}.json")).replace("\\","/") if False else None,
        })
        for typ,(label,attack,source,fn) in OPS.items():
            rng=random.Random(hseed(domain,ts,typ))
            try:
                out=fn(raw,rng)
                sid=f"{typ}__{ts}"
                p=domdir/f"{sid}.html"
                p.write_bytes(out)
                if sha(out)==sha(raw): exact_same+=1
                rows.append({
                    "domain":domain,"family_id":b["family_id"],"split":b["split"],
                    "snapshot_id":sid,"variant_type":typ,"pair_label":label,"attack_type":attack,
                    "source_type":source,"base_snapshot_id":raw_id,
                    "html_path":str(p.relative_to(ROOT)).replace("\\","/"),
                    "html_sha256":sha(out),"bytes":len(out),
                    "requested_timestamp":ts,"effective_capture_timestamp":b["effective_capture_timestamp"],
                    "base_html_sha256":sha(raw),
                })
            except Exception as e:
                failures.append({"domain":domain,"timestamp":ts,"variant_type":typ,"error":type(e).__name__+": "+str(e)})
        if n%100==0: print("generated",n,"/",len(bases),flush=True)
    rows.sort(key=lambda r:(r["domain"],r["requested_timestamp"],r["snapshot_id"]))
    with MANIFEST.open("w",encoding="utf-8",newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r,sort_keys=True,separators=(",",":"))+"\n")
    expected=len(bases)*7
    criteria={
        "all_expected_snapshots_generated":len(rows)==expected,
        "no_generation_failures":len(failures)==0,
        "all_html_paths_exist":all((ROOT/r["html_path"]).exists() for r in rows),
        "all_defaced_variants_differ_from_base":all(r["html_sha256"]!=r.get("base_html_sha256") for r in rows if r["pair_label"]=="defaced"),
        "benign_wrapper_differs_from_base":all(r["html_sha256"]!=r.get("base_html_sha256") for r in rows if r["variant_type"]=="benign_wrapper"),
    }
    verdict="PASS" if all(criteria.values()) else "FAIL_REPAIR"
    report={
        "gate_id":"G1b_variant_generation",
        "verdict":verdict,
        "base_snapshots":len(bases),
        "expected_snapshots":expected,
        "generated_snapshots":len(rows),
        "exact_same_variant_count":exact_same,
        "failures":failures[:50],
        "criteria":criteria,
        "snapshot_manifest":str(MANIFEST),
        "snapshot_manifest_sha256":hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "decision":"proceed_to_pair_construction_and_modalities" if verdict=="PASS" else "stop_and_repair_variants",
    }
    SUMMARY.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))
    return 0 if verdict=="PASS" else 3

if __name__=="__main__":
    raise SystemExit(main())
