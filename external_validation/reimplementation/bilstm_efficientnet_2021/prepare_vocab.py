import argparse, collections, json, pathlib, sys, time
HERE=pathlib.Path(__file__).resolve()
COMMON=HERE.parents[1]/"common"
sys.path.insert(0,str(COMMON))
from static_dataset import WordTokenizer, load_manifest, visible_text_from_html

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--manifest",default=r"D:\RESEARCH\DI_MM_V4\external_candidates\sota_static\lwded_v4_static_current_manifest.jsonl")
    ap.add_argument("--out",default=str(HERE.parent/"vocab.json"))
    ap.add_argument("--max-words",type=int,default=None)
    args=ap.parse_args()

    rows=load_manifest(args.manifest,"pretrain")
    tok=WordTokenizer(max_words=args.max_words)
    cnt=collections.Counter()
    t0=time.time()
    for i,r in enumerate(rows,1):
        text=visible_text_from_html(r["html_path"])
        cnt.update(tok._tokens(text))
        if i%100==0 or i==len(rows):
            print(f"VOCAB_PROGRESS {i}/{len(rows)} unique={len(cnt)} elapsed={time.time()-t0:.1f}s",flush=True)
    words=[w for w,n in cnt.most_common() if n>=tok.min_freq and w!=tok.oov_token]
    if tok.max_words is not None:
        words=words[:max(0,tok.max_words-2)]
    tok.word_index={tok.oov_token:1}
    for w in words:
        tok.word_index[w]=len(tok.word_index)+1
    tok.save(args.out)
    print(json.dumps({"pretrain_rows":len(rows),"vocab_size_including_pad":len(tok.word_index)+1,"out":args.out,"elapsed_seconds":time.time()-t0},indent=2),flush=True)
if __name__=="__main__": main()
