import argparse, hashlib, json, os, re, time
from pathlib import Path

from bs4 import UnicodeDammit
from playwright.sync_api import sync_playwright

ROOT=Path(r"D:\RESEARCH\DI_MM_V4")
V4=ROOT/"dataset_pipeline"/"v4"
SNAPS=V4/"snapshot_manifest.jsonl"
SHOTS=V4/"screenshots"
SHARD_DIR=ROOT/"visual_render_shards"
VIEWPORT={"width":1280,"height":720}

def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()

CSP_META='<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; img-src data:; style-src \'unsafe-inline\' data:; font-src data:; media-src data:; frame-src \'none\'; connect-src \'none\';">'
STABLE_STYLE='<style>*,*::before,*::after{animation:none!important;transition:none!important;caret-color:transparent!important}html{scroll-behavior:auto!important}</style>'

def decode_html_bytes(raw):
    text=UnicodeDammit(raw,is_html=True).unicode_markup
    if text is None:
        text=raw.decode("utf-8",errors="replace")
    text=text.encode("utf-8",errors="replace").decode("utf-8")
    text="".join(ch for ch in text if ch in "\t\n\r" or ord(ch)>=32)
    m=re.search(r"<head\b[^>]*>",text,flags=re.IGNORECASE)
    inject=CSP_META+STABLE_STYLE
    if m:
        text=text[:m.end()]+inject+text[m.end():]
    else:
        text=inject+text
    return text

def new_page(context):
    page=context.new_page()
    page.set_default_timeout(10000)
    return page

def render_one(page,row):
    path=(ROOT/row["html_path"]).resolve()
    raw=path.read_bytes()
    html=decode_html_bytes(raw)
    page.set_content(html,wait_until="domcontentloaded",timeout=10000)
    png=page.screenshot(type="png",full_page=False,animations="disabled",timeout=10000)
    return png

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--shard-index",type=int,required=True)
    ap.add_argument("--num-shards",type=int,required=True)
    ap.add_argument("--limit",type=int,default=0)
    args=ap.parse_args()
    if not (0<=args.shard_index<args.num_shards):
        raise SystemExit("invalid shard arguments")

    rows=[json.loads(x) for x in SNAPS.read_text(encoding="utf-8").splitlines() if x.strip()]
    rows.sort(key=lambda r:(r["domain"],r["snapshot_id"]))
    selected=[r for i,r in enumerate(rows) if i%args.num_shards==args.shard_index]
    if args.limit>0:
        selected=selected[:args.limit]

    SHOTS.mkdir(parents=True,exist_ok=True)
    SHARD_DIR.mkdir(parents=True,exist_ok=True)
    manifest=SHARD_DIR/f"render_shard_{args.shard_index:02d}.jsonl"
    report=SHARD_DIR/f"render_shard_{args.shard_index:02d}_report.json"
    events=[]
    errors=[]
    browser_version=None
    started=time.time()

    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True)
        browser_version=browser.version
        context=browser.new_context(viewport=VIEWPORT,device_scale_factor=1.0,java_script_enabled=False)
        context.set_default_timeout(10000)
        page=new_page(context)
        for i,row in enumerate(selected,1):
            out=SHOTS/row["domain"]/f"{row['snapshot_id']}.png"
            out.parent.mkdir(parents=True,exist_ok=True)
            err=None
            try:
                png=render_one(page,row)
                out.write_bytes(png)
                events.append({
                    "domain":row["domain"],
                    "snapshot_id":row["snapshot_id"],
                    "screenshot_path":str(out.relative_to(ROOT)).replace("\\","/"),
                    "screenshot_sha256":sha_bytes(png),
                    "bytes":len(png),
                })
            except Exception as e:
                err=type(e).__name__+": "+str(e)
                errors.append({"domain":row["domain"],"snapshot_id":row["snapshot_id"],"error":err})
                try:
                    page.close()
                except Exception:
                    pass
                page=new_page(context)
            if i%100==0 or i==len(selected):
                print(f"render shard={args.shard_index} {i}/{len(selected)} errors={len(errors)}",flush=True)
        try:
            page.close()
        except Exception:
            pass
        context.close()
        browser.close()

    events.sort(key=lambda r:(r["domain"],r["snapshot_id"]))
    with manifest.open("w",encoding="utf-8",newline="\n") as f:
        for r in events:
            f.write(json.dumps(r,sort_keys=True,separators=(",",":"))+"\n")
    result={
        "shard_index":args.shard_index,
        "num_shards":args.num_shards,
        "selected":len(selected),
        "generated":len(events),
        "errors":errors[:100],
        "error_count":len(errors),
        "browser_version":browser_version,
        "render_policy":"UnicodeDammit decoded HTML -> Playwright set_content; Chromium headless; JavaScript disabled; all external network blocked; fixed 1280x720 viewport; device scale 1",
        "manifest_path":str(manifest),
        "manifest_sha256":hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "elapsed_seconds":time.time()-started,
        "verdict":"PASS" if len(events)==len(selected) and not errors else "FAIL_REPAIR",
    }
    report.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2),flush=True)
    return 0 if result["verdict"]=="PASS" else 3

if __name__=="__main__":
    raise SystemExit(main())
