import argparse, hashlib, json, re, time
from pathlib import Path
from bs4 import UnicodeDammit
from playwright.sync_api import sync_playwright

ROOT=Path(r"D:\RESEARCH\DI_MM_V4")
V4=ROOT/"dataset_pipeline"/"v4"
SHOTS=V4/"screenshots"
MISSING=ROOT/"visual_repair"/"missing_visual_rows.jsonl"
OUTDIR=ROOT/"visual_repair"
VIEWPORT={"width":1280,"height":720}
TIMEOUTS=[10000,30000,60000]
CSP_META='<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; img-src data:; style-src \'unsafe-inline\' data:; font-src data:; media-src data:; frame-src \'none\'; connect-src \'none\';">'
STABLE_STYLE='<style>*,*::before,*::after{animation:none!important;transition:none!important;caret-color:transparent!important}html{scroll-behavior:auto!important}</style>'

def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()

def decode_html_bytes(raw):
    text=UnicodeDammit(raw,is_html=True).unicode_markup
    if text is None:
        text=raw.decode("utf-8",errors="replace")
    text=text.encode("utf-8",errors="replace").decode("utf-8")
    text="".join(ch for ch in text if ch in "\t\n\r" or ord(ch)>=32)
    inject=CSP_META+STABLE_STYLE
    m=re.search(r"<head\b[^>]*>",text,flags=re.IGNORECASE)
    return text[:m.end()]+inject+text[m.end():] if m else inject+text

def open_browser(p):
    browser=p.chromium.launch(headless=True)
    context=browser.new_context(viewport=VIEWPORT,device_scale_factor=1.0,java_script_enabled=False)
    page=context.new_page()
    return browser,context,page

def close_all(browser,context,page):
    for obj in (page,context,browser):
        try:
            if obj is not None: obj.close()
        except Exception:
            pass

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--shard-index",type=int,required=True)
    ap.add_argument("--num-shards",type=int,required=True)
    args=ap.parse_args()
    rows=[json.loads(x) for x in MISSING.read_text(encoding="utf-8").splitlines() if x.strip()]
    selected=[r for i,r in enumerate(rows) if i%args.num_shards==args.shard_index]
    OUTDIR.mkdir(parents=True,exist_ok=True)
    events=[]; errors=[]
    started=time.time()
    with sync_playwright() as p:
        browser=context=page=None
        browser,context,page=open_browser(p)
        for i,row in enumerate(selected,1):
            out=SHOTS/row["domain"]/f"{row['snapshot_id']}.png"
            if out.exists():
                events.append({"domain":row["domain"],"snapshot_id":row["snapshot_id"],"status":"already_present","screenshot_path":str(out.relative_to(ROOT)).replace("\\","/"),"screenshot_sha256":hashlib.sha256(out.read_bytes()).hexdigest()})
                continue
            raw=(ROOT/row["html_path"]).read_bytes()
            html=decode_html_bytes(raw)
            success=False
            attempts=[]
            for attempt,timeout in enumerate(TIMEOUTS,1):
                try:
                    page.set_default_timeout(timeout)
                    t0=time.time()
                    page.set_content(html,wait_until="domcontentloaded",timeout=timeout)
                    png=page.screenshot(type="png",full_page=False,animations="disabled",timeout=timeout)
                    out.parent.mkdir(parents=True,exist_ok=True)
                    out.write_bytes(png)
                    attempts.append({"attempt":attempt,"timeout_ms":timeout,"elapsed_seconds":time.time()-t0,"status":"success"})
                    events.append({"domain":row["domain"],"snapshot_id":row["snapshot_id"],"status":"repaired","attempt":attempt,"screenshot_path":str(out.relative_to(ROOT)).replace("\\","/"),"screenshot_sha256":sha_bytes(png),"bytes":len(png),"attempts":attempts})
                    success=True
                    break
                except Exception as e:
                    attempts.append({"attempt":attempt,"timeout_ms":timeout,"status":"error","error":type(e).__name__+": "+str(e)})
                    close_all(browser,context,page)
                    browser=context=page=None
                    if attempt < len(TIMEOUTS):
                        try:
                            browser,context,page=open_browser(p)
                        except Exception as be:
                            attempts.append({"attempt":attempt,"status":"browser_reopen_error","error":type(be).__name__+": "+str(be)})
                            time.sleep(1)
            if not success:
                errors.append({"domain":row["domain"],"snapshot_id":row["snapshot_id"],"html_path":row["html_path"],"attempts":attempts})
                if browser is None:
                    try: browser,context,page=open_browser(p)
                    except Exception: pass
            if i%20==0 or i==len(selected):
                print(f"repair shard={args.shard_index} {i}/{len(selected)} repaired={sum(1 for e in events if e['status']=='repaired')} errors={len(errors)}",flush=True)
        close_all(browser,context,page)
    report={"shard_index":args.shard_index,"num_shards":args.num_shards,"selected":len(selected),"events":events,"error_count":len(errors),"errors":errors,"elapsed_seconds":time.time()-started,"verdict":"PASS" if not errors else "FAIL_REPAIR"}
    rp=OUTDIR/f"repair_shard_{args.shard_index:02d}_report.json"
    rp.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"shard_index":args.shard_index,"selected":len(selected),"error_count":len(errors),"verdict":report["verdict"],"report":str(rp)},indent=2),flush=True)
    return 0 if not errors else 3
if __name__=="__main__":
    raise SystemExit(main())
