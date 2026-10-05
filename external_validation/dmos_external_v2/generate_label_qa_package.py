import csv, json, pathlib, re
from bs4 import BeautifulSoup
ROOT=pathlib.Path(r"D:\RESEARCH\DI_MM_V4\external_candidates\DMoS_external_v2")
pairs=[json.loads(x) for x in (ROOT/"external_pairs_v2.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
snaps={x["domain"]+"|"+x["snapshot_id"]:x for x in [json.loads(z) for z in (ROOT/"external_snapshot_manifest_v2.jsonl").read_text(encoding="utf-8").splitlines() if z.strip()]}

def text(p):
    raw=pathlib.Path(p).read_bytes();s=BeautifulSoup(raw,"html.parser")
    for t in s(["script","style","noscript","template"]):t.decompose()
    return " ".join(s.stripped_strings)[:1200]

out=ROOT/"V2_LABEL_QA_PACKAGE.csv"
with out.open("w",encoding="utf-8-sig",newline="") as f:
    w=csv.DictWriter(f,fieldnames=["pair_id","host_group","label","t1_timestamp","t1_path","t2_path","t1_text_preview","t2_text_preview","reviewer1","reviewer2","adjudication","notes"])
    w.writeheader()
    for q in pairs:
        a=snaps[q["domain"]+"|"+q["t1"]];b=snaps[q["domain"]+"|"+q["t2"]]
        w.writerow({"pair_id":q["pair_id"],"host_group":q["host_group"],"label":q["label"],"t1_timestamp":a.get("timestamp"),
                    "t1_path":a["html_path"],"t2_path":b["html_path"],"t1_text_preview":text(a["html_path"]),"t2_text_preview":text(b["html_path"]),
                    "reviewer1":"","reviewer2":"","adjudication":"","notes":""})
rep={"gate":"V2_LABEL_QA_PACKAGE","verdict":"PENDING_INDEPENDENT_REVIEW","pairs":len(pairs),"path":str(out),
     "rule":"Pair selection is frozen before predictions; later review must not silently drop outcome-unfavorable pairs. Material label disputes block claim promotion."}
(ROOT/"V2_LABEL_QA_GATE.json").write_text(json.dumps(rep,indent=2)+"\n",encoding="utf-8")
print(json.dumps(rep,indent=2))
