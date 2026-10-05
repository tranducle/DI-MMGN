import json,time
from pathlib import Path
from bs4 import UnicodeDammit
from playwright.sync_api import sync_playwright

ROOT=Path(r"D:\RESEARCH\DI_MM_V4")
rows=[json.loads(x) for x in (ROOT/"dataset_pipeline"/"v4"/"snapshot_manifest.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
rows.sort(key=lambda r:(r["domain"],r["snapshot_id"]))
row=rows[0]
path=(ROOT/row["html_path"]).resolve()
print("STEP row",row["domain"],row["snapshot_id"],path.stat().st_size,flush=True)
raw=path.read_bytes()
print("STEP read",len(raw),flush=True)
text=UnicodeDammit(raw,is_html=True).unicode_markup
if text is None:
    text=raw.decode("utf-8",errors="replace")
text=text.encode("utf-8",errors="replace").decode("utf-8")
print("STEP decoded",len(text),flush=True)
with sync_playwright() as p:
    print("STEP playwright",flush=True)
    browser=p.chromium.launch(headless=True)
    print("STEP browser",browser.version,flush=True)
    context=browser.new_context(viewport={"width":1280,"height":720},device_scale_factor=1.0,java_script_enabled=False)
    context.set_default_timeout(5000)
    print("STEP context",flush=True)
    page=context.new_page()
    print("STEP page",flush=True)
    t=time.time()
    page.set_content(text,wait_until="domcontentloaded",timeout=5000)
    print("STEP set_content",time.time()-t,flush=True)
    t=time.time()
    png=page.screenshot(type="png",full_page=False,animations="disabled",timeout=5000)
    print("STEP screenshot",len(png),time.time()-t,flush=True)
    out=ROOT/"visual_diag.png"
    out.write_bytes(png)
    print("STEP wrote",out,flush=True)
    page.close();context.close();browser.close()
print("DONE",flush=True)
