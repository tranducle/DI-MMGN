import hashlib, json, pathlib, sys, time
import torch
HERE=pathlib.Path(__file__).resolve(); REIMPL=HERE.parents[1]
sys.path.insert(0,str(REIMPL/"common"))
from static_dataset import WordTokenizer, load_manifest, visible_text_from_html

MANIFEST=r"D:\RESEARCH\DI_MM_V4\external_candidates\sota_static\lwded_v4_static_current_manifest.jsonl"
OUT=pathlib.Path(r"D:\RESEARCH\DI_MM_V4\external_candidates\sota_static\bilstm_text_tokens_128.pt")

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for c in iter(lambda:f.read(1024*1024),b""):h.update(c)
    return h.hexdigest()

rows=load_manifest(MANIFEST,None)
tok=WordTokenizer.load(HERE.parent/"vocab.json")
ids=[]; t0=time.time()
for i,r in enumerate(rows,1):
    text=visible_text_from_html(r["html_path"])
    ids.append(tok.encode(text,128))
    if i%100==0 or i==len(rows):
        print(f"TEXT_CACHE_PROGRESS {i}/{len(rows)} elapsed={time.time()-t0:.1f}s",flush=True)
obj={"input_ids":torch.tensor(ids,dtype=torch.int32),
     "pair_id":[r["pair_id"] for r in rows],
     "split":[r["split"] for r in rows],
     "metadata":{"rows":len(rows),"max_len":128,"vocab_size_including_pad":len(tok.word_index)+1,
                 "manifest":MANIFEST,"elapsed_seconds":time.time()-t0}}
OUT.parent.mkdir(parents=True,exist_ok=True); torch.save(obj,OUT)
summary={**obj["metadata"],"cache":str(OUT),"cache_bytes":OUT.stat().st_size,"cache_sha256":sha256(OUT)}
OUT.with_suffix(".summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
print(json.dumps(summary,indent=2),flush=True)
