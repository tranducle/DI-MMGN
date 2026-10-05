import json,re,time
from pathlib import Path
from bs4 import UnicodeDammit
from playwright.sync_api import sync_playwright

ROOT=Path(r"D:\RESEARCH\DI_MM_V4")
rows=[json.loads(x) for x in (ROOT/"dataset_pipeline"/"v4"/"snapshot_manifest.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
rows.sort(key=lambda r:(r["domain"],r["snapshot_id"]))
row=rows[0]
path=(ROOT/row["html_path"]).resolve()
raw=path.read_bytes()
text=UnicodeDammit(raw,is_html=True).unicode_markup or raw.decode("utf-8",errors="replace")
text=text.encode("utf-8",errors="replace").decode("utf-8")
CSP='<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; img-src data:; style-src \'unsafe-inline\' data:; font-src data:; media-src data:; frame-src \'none\'; connect-src \'none\';">'
m=re.search(r"<head\b[^>]*>",text,flags=re.IGNORECASE)
text=text[:m.end()]+CSP+text[m.end():] if m else CSP+text
print("STEP decoded+csp",len(text),flush=True)
with sync_playwright() as p:
    print("STEP playwright",flush=True)
    browser=p.chromium.launch(headless=True)
    print("STEP browser",browser.version,flush=True)
    context=browser.new_context(viewport={"width":1280,"height":720},device_scale_factor=1.0,java_script_enabled=False)
    context.set_default_timeout(5000)
    page=context.new_page()
    print("STEP page",flush=True)
    t=time.time(); page.set_content(text,wait_until="domcontentloaded",timeout=5000); print("STEP set",time.time()-t,flush=True)
    t=time.time(); page.add_style_tag(content="*,*::before,*::after{animation:none!important;transition:none!important;caret-color:transparent!important}html{scroll-behavior:auto!important}"); print("STEP style",time.time()-t,flush=True)
    t=time.time(); png=page.screenshot(type="png",full_page=False,animations="disabled",timeout=5000); print("STEP shot",len(png),time.time()-t,flush=True)
    (ROOT/"visual_diag_csp.png").write_bytes(png)
    page.close();context.close();browser.close()
print("DONE",flush=True)
