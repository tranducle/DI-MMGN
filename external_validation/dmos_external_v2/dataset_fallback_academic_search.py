import dataclasses, json, pathlib, sys
sys.path.insert(0,r"C:\Users\Tran Duc Le\Documents\LOCAL-CODING-AGENT-WIN\DO-A-PAPER-v3")
from src.tools import literature_tools as lt

OUT=pathlib.Path(r"C:\Users\Tran Duc Le\Documents\LOCAL-CODING-AGENT-WIN\DO-A-PAPER-v2\Papers\[IEEE ACCESS] - DI-MMGN_REVIEWED_20260923_1357\DI-MMGN-v4\external_validation\dmos_external_v2")
queries=[
  "website defacement detection dataset temporal snapshots benchmark",
  "web defacement dataset Zone-H HTML screenshot normal webpages",
  "website defacement real dataset public archive URL timestamp",
  "webpage defacement detection dataset Alexa Zone-H 96100",
  "web defacement monitoring dataset longitudinal archive"
]
sources=[
  ("openalex",lt.search_openalex_sync),
  ("semantic_scholar",lt.search_semantic_scholar_sync),
  ("scopus",lt.search_scopus_sync),
  ("wos",getattr(lt,"search_wos_sync",None)),
]
raw=[];errors=[]
for q in queries:
    for name,fn in sources:
        if fn is None: continue
        try:
            pp=fn(q,limit=20,year_from=2010,year_to=2026)
            for p in pp:
                d=dataclasses.asdict(p);d["search_query"]=q;d["search_source"]=name;raw.append(d)
            print(f"SEARCH {name} q={q!r} n={len(pp)}",flush=True)
        except Exception as e:
            errors.append({"source":name,"query":q,"error":repr(e)})
            print(f"SEARCH_ERROR {name} q={q!r} err={type(e).__name__}:{e}",flush=True)

dedup={}
for d in raw:
    key=(d.get("doi") or "").lower().strip() or (d.get("title") or "").lower().strip()
    if not key: continue
    if key not in dedup: dedup[key]=d
papers=list(dedup.values())
terms=["dataset","defacement","zone-h","archive","temporal","snapshot","html","webpage","benchmark"]
def score(d):
    title=(d.get("title") or "").lower()
    text=(title+" "+(d.get("abstract") or "")).lower()
    return sum(2 if x in title else 1 for x in terms if x in text)
papers.sort(key=lambda d:(score(d),d.get("citation_count") or 0,d.get("year") or 0),reverse=True)

(OUT/"DATASET_FALLBACK_ACADEMIC_SEARCH.json").write_text(
    json.dumps({"queries":queries,"years":"2010-2026","databases":[x[0] for x in sources],
                "papers":papers,"errors":errors},indent=2,ensure_ascii=False)+"\n",encoding="utf-8")

lines=[
"# Dataset Discovery and Fit Register — academic-search pass","",
"Search years: 2010–2026.",
"Databases attempted: OpenAlex, Semantic Scholar, Scopus, Web of Science via project literature_tools.py.","",
"All candidates remain **candidate_needs_access_or_license_check** until raw data, URLs/timestamps, labels, and license are independently verified.","",
"| # | Year | Candidate paper | DOI | Search source | Fit note |",
"|---:|---:|---|---|---|---|"
]
for i,d in enumerate(papers[:40],1):
    title=(d.get("title") or "").replace("|","/")
    doi=d.get("doi") or ""
    text=((d.get("title") or "")+" "+(d.get("abstract") or "")).lower()
    fit=[]
    if "zone-h" in text:fit.append("Zone-H/real-defacement provenance")
    if "dataset" in text:fit.append("dataset described")
    if "temporal" in text or "longitudinal" in text or "snapshot" in text:fit.append("possible temporal signal")
    if "screenshot" in text or "image" in text:fit.append("multimodal possible")
    if not fit:fit.append("needs manual fit check")
    lines.append(f"| {i} | {d.get('year') or ''} | {title} | {doi} | {d.get('search_source')} | {'; '.join(fit)} |")
lines+=["","## Required access/fit verification",
"For any promising candidate verify: raw HTML availability; per-page source URL; capture timestamp; benign/defaced label provenance; same-site predecessor feasibility; license/terms; redistribution constraints; baseline/metric compatibility.",
"","## Search errors","~~~json",json.dumps(errors,indent=2,ensure_ascii=False),"~~~"]
(OUT/"DATASET_DISCOVERY_AND_FIT_REGISTER.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
print(json.dumps({"unique_papers":len(papers),"errors":len(errors),"top_titles":[p.get("title") for p in papers[:10]]},indent=2,ensure_ascii=False))
