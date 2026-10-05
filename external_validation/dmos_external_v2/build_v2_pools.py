import hashlib, json, pathlib

V1=pathlib.Path(r"D:\RESEARCH\DI_MM_V4\external_candidates\DMoS_external")
V2=pathlib.Path(r"D:\RESEARCH\DI_MM_V4\external_candidates\DMoS_external_v2")
POOLS=V2/"pools"
POOLS.mkdir(parents=True,exist_ok=True)

src=V1/"dmos_temporal_candidates.jsonl"
rows=[json.loads(x) for x in src.read_text(encoding="utf-8").splitlines() if x.strip()]
existing=[r for r in rows if r.get("positive_pair_eligible")]
primary=[r for r in rows if r.get("positive_keyword_hits",0)>=1 and len(r.get("assessed_captures",[]))>0 and not r.get("positive_pair_eligible")]
secondary=[r for r in rows if r.get("positive_keyword_hits",0)>=1 and len(r.get("assessed_captures",[]))==0]
current_pos=[r for r in rows if r.get("positive_keyword_hits",0)>=1]

expected={"existing":9,"primary":99,"secondary":309,"current_positive":417}
actual={"existing":len(existing),"primary":len(primary),"secondary":len(secondary),"current_positive":len(current_pos)}
if actual!=expected:
    raise SystemExit(f"POOL_PROVENANCE_MISMATCH actual={actual} expected={expected}")

def slim(r):
    return {
        "id":r["id"],"host_hint":r["host_hint"],"inferred_url":r["inferred_url"],
        "html_path":r["html_path"],"html_sha256":r["html_sha256"],
        "collection_iso":r["collection_iso"],"collection_mtime":r["collection_mtime"],
        "positive_keyword_hits":r.get("positive_keyword_hits",0),
        "url_source":r.get("url_source"),"relative_path":r.get("relative_path"),
        "v1_captures":r.get("captures",[]),
        "v1_assessed_captures":r.get("assessed_captures",[])
    }

def write(name,arr):
    p=POOLS/name
    with p.open("w",encoding="utf-8",newline="\n") as f:
        for r in arr:
            f.write(json.dumps(slim(r),sort_keys=True,separators=(",",":"))+"\n")
    return p

p_existing=write("existing_positive.jsonl",existing)
p_primary=write("primary_salvage_pool.jsonl",primary)
p_secondary=write("secondary_salvage_pool.jsonl",secondary)

def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda:f.read(1024*1024),b""): h.update(c)
    return h.hexdigest()

summary={"gate":"V2_G0_POOL_PROVENANCE","verdict":"PASS","source":str(src),"source_sha256":sha(src),"counts":actual,
         "files":{"existing":{"path":str(p_existing),"sha256":sha(p_existing)},"primary":{"path":str(p_primary),"sha256":sha(p_primary)},"secondary":{"path":str(p_secondary),"sha256":sha(p_secondary)}}}
(V2/"V2_G0_POOL_PROVENANCE.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
print(json.dumps(summary,indent=2))
