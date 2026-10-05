import dataclasses, json, pathlib, sys, re
sys.path.insert(0,r"C:\\Users\\Tran Duc Le\\Documents\\LOCAL-CODING-AGENT-WIN\\DO-A-PAPER-v3")
from src.tools import literature_tools as lt

OUT=pathlib.Path(r"C:\Users\Tran Duc Le\Documents\LOCAL-CODING-AGENT-WIN\DO-A-PAPER-v2\Papers\[IEEE ACCESS] - DI-MMGN_REVIEWED_20260923_1357\DI-MMGN-v4\external_validation\dataset_search_v3")
queries=[
 "website defacement dataset Zone-H raw HTML screenshots",
 "website defacement detection dataset Alexa Zone-H HTML image",
 "defaced webpages dataset raw HTML",
 "web defacement dataset normal defaced webpages",
 "website defacement benchmark dataset",
 "website defacement temporal dataset archive",
 "hacked webpage archive dataset website defacement",
 "website defacement dataset Zenodo",
 "website defacement screenshots dataset",
 "website defacement dataset URL timestamp",
 "Zone-H website defacement dataset 96100",
 "web defacement real-world dataset machine learning"
]
sources=[
 ("openalex",lt.search_openalex_sync),
 ("semantic_scholar",lt.search_semantic_scholar_sync),
 ("scopus",lt.search_scopus_sync),
 ("google_scholar",lt.search_google_scholar),
]
raw=[]; errors=[]
for q in queries:
    for name,fn in sources:
        try:
            kwargs=dict(query=q,limit=15,year_from=2010)
            if name!="google_scholar": kwargs["year_to"]=2026
            pp=fn(**kwargs)
            print(f"SEARCH source={name} q={q!r} n={len(pp)}",flush=True)
            for p in pp:
                d=dataclasses.asdict(p)
                d["search_source"]=name
                d["search_query"]=q
                raw.append(d)
        except Exception as e:
            errors.append({"source":name,"query":q,"error":repr(e)})
            print(f"ERROR source={name} q={q!r} err={type(e).__name__}:{e}",flush=True)

dedup={}
for d in raw:
    key=(d.get("doi") or "").lower().strip() or re.sub(r"\W+"," ",(d.get("title") or "").lower()).strip()
    if not key: continue
    if key not in dedup:
        dedup[key]=d
    else:
        old=dedup[key]
        seen=set((old.get("search_sources") or [old.get("search_source")]))
        seen.add(d.get("search_source"))
        old["search_sources"]=sorted(x for x in seen if x)

papers=list(dedup.values())
def score(d):
    t=((d.get("title") or "")+" "+(d.get("abstract") or "")).lower()
    s=0
    terms={
      "defacement":8,"defaced":6,"website":2,"webpage":2,"dataset":5,"data set":5,
      "zone-h":7,"zone h":7,"alexa":5,"html":5,"screenshot":5,"image":2,
      "archive":5,"archival":5,"timestamp":4,"url":3,"temporal":5,"longitudinal":4,
      "zenodo":3,"github":3,"public":2,"available":2
    }
    for k,w in terms.items():
        if k in t:s+=w
    if "phishing" in t:s-=5
    if "deepfake" in t:s-=8
    if "rumor" in t:s-=8
    if "graph" in t and "defacement" not in t:s-=4
    return s
papers.sort(key=lambda d:(score(d),d.get("citation_count") or 0,d.get("year") or 0),reverse=True)
for d in papers:d["fit_score_raw"]=score(d)

(OUT/"academic_dataset_search_raw.json").write_text(json.dumps({"queries":queries,"sources":[x[0] for x in sources],"errors":errors,"papers":papers},indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
lines=[
 "# DI-MMGN External Dataset Academic Discovery — Pass 1","",
 "Goal: discover independent website-defacement datasets that can support temporal or at least real-world external validation.",
 "",
 "Hard requirements for full temporal fit: raw HTML or renderable page content; source URL/site identity; capture/event timestamp; trustworthy defaced/benign labels or reconstructable clean predecessor; obtainable license/access.",
 "",
 f"Unique academic records: {len(papers)}. Search errors: {len(errors)}.","",
 "| Rank | Year | Candidate paper | DOI | Source | Heuristic fit |",
 "|---:|---:|---|---|---|---:|"
]
for i,d in enumerate(papers[:60],1):
    lines.append(f"| {i} | {d.get('year') or ''} | {(d.get('title') or '').replace('|','/')} | {d.get('doi') or ''} | {d.get('search_source') or ''} | {d['fit_score_raw']} |")
lines += ["","## Search errors","~~~json",json.dumps(errors,indent=2,ensure_ascii=False),"~~~"]
(OUT/"academic_dataset_search_pass1.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
print(json.dumps({"unique_records":len(papers),"errors":len(errors),"top":[{"title":p.get("title"),"year":p.get("year"),"doi":p.get("doi"),"score":p.get("fit_score_raw")} for p in papers[:20]]},indent=2,ensure_ascii=False))
