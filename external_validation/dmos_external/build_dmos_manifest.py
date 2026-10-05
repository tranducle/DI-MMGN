import hashlib, json, pathlib, re
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup

SRC=pathlib.Path(r"D:\RESEARCH\DI_MM_V4\external_candidates\DMoS\real500_extracted\real_en_defacement_500")
OUT=pathlib.Path(r"D:\RESEARCH\DI_MM_V4\external_candidates\DMoS_external")
OUT.mkdir(parents=True,exist_ok=True)

def sha256(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for c in iter(lambda:f.read(1024*1024),b""): h.update(c)
    return h.hexdigest()

def host_from_rel(rel):
    return rel.parts[0][:-5] if len(rel.parts)==1 and rel.parts[0].endswith(".html") else rel.parts[0]

def extract_url(path, rel):
    raw=path.read_bytes()
    if not raw: return None,"empty"
    soup=BeautifulSoup(raw,"html.parser")
    host=host_from_rel(rel)
    # Build a release-path base URL for resolving relative canonicals.
    if len(rel.parts)==1:
        base_url=f"http://{host}/"
    else:
        base_url="http://"+host+"/"+"/".join(rel.parts[1:])
    # 1 canonical (absolute or relative)
    for link in soup.find_all("link"):
        relv=link.get("rel") or []
        if isinstance(relv,str): relv=[relv]
        if any(str(x).lower()=="canonical" for x in relv):
            href=(link.get("href") or "").strip()
            if href:
                u=urljoin(base_url,href)
                if u.startswith(("http://","https://")):
                    return u,"canonical"
    # 2 og:url
    for m in soup.find_all("meta"):
        key=(m.get("property") or m.get("name") or "").lower()
        if key=="og:url":
            u=(m.get("content") or "").strip()
            if u.startswith(("http://","https://")): return u,"og:url"
    # 3 ability.nyu.edu known query encoding, directly evidenced by in-page links
    if host=="ability.nyu.edu" and len(rel.parts)>=3 and rel.parts[1]=="js" and rel.parts[-1].startswith("index_key="):
        q=rel.parts[-1][len("index_key="):]
        if q.endswith(".html"): q=q[:-5]
        # first underscore separates category from slug in released filename
        if "_" in q:
            cat,slug=q.split("_",1)
            return f"http://ability.nyu.edu/js/?key={cat}/{slug}","release-path+inpage-pattern"
    # 4 fallback path reconstruction
    if len(rel.parts)==1:
        return f"http://{host}/","root-fallback"
    parts=list(rel.parts[1:])
    return "http://"+host+"/"+"/".join(parts),"path-fallback"

rows=[]
for p in sorted(SRC.rglob("*")):
    if not p.is_file(): continue
    rel=p.relative_to(SRC)
    u,src=extract_url(p,rel)
    host=host_from_rel(rel)
    parsed=urlparse(u) if u else None
    rows.append({
        "id":f"dmos_{len(rows):04d}",
        "relative_path":str(rel).replace("\\","/"),
        "html_path":str(p),
        "html_bytes":p.stat().st_size,
        "html_sha256":sha256(p),
        "collection_mtime":p.stat().st_mtime,
        "collection_date":p.stat().st_mtime_ns,
        "collection_iso":__import__("datetime").datetime.fromtimestamp(p.stat().st_mtime).isoformat(),
        "host_hint":host,
        "inferred_url":u,
        "url_source":src,
        "url_host":parsed.hostname if parsed else None,
    })

manifest=OUT/"dmos_real500_manifest.jsonl"
with manifest.open("w",encoding="utf-8",newline="\n") as f:
    for r in rows: f.write(json.dumps(r,sort_keys=True,separators=(",",":"))+"\n")

from collections import Counter
summary={
    "status":"PASS" if len(rows)==500 and sum(r["html_bytes"]>0 for r in rows)>=498 else "FAIL",
    "rows":len(rows),
    "nonempty":sum(r["html_bytes"]>0 for r in rows),
    "empty":sum(r["html_bytes"]==0 for r in rows),
    "unique_inferred_urls":len({r["inferred_url"] for r in rows if r["inferred_url"]}),
    "url_source_counts":dict(Counter(r["url_source"] for r in rows)),
    "collection_dates":dict(Counter(r["collection_iso"][:10] for r in rows)),
    "unique_host_hints":len({r["host_hint"] for r in rows}),
    "manifest_sha256":sha256(manifest),
    "manifest":str(manifest),
}
(OUT/"G0_DMoS_PROVENANCE.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
print(json.dumps(summary,indent=2))
