import hashlib, io, json, platform, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from playwright.sync_api import sync_playwright
from transformers import CLIPModel, CLIPProcessor
import transformers

ROOT=Path(r"D:\RESEARCH\DI_MM_V4")
V4=ROOT/"dataset_pipeline"/"v4"
SNAPS=V4/"snapshot_manifest.jsonl"
OUT=V4/"visuals"
SHOTS=V4/"screenshots"
REPORT=ROOT/"gates"/"G1e_visual_generation.json"
PROV=V4/"visual_provenance.json"

MODEL_ID="openai/clip-vit-base-patch32"
COMMIT="3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
BATCH=16
VIEWPORT={"width":1280,"height":720}

def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()

def render_batch(rows):
    pngs=[]; meta=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True)
        version=browser.version
        context=browser.new_context(viewport=VIEWPORT,device_scale_factor=1.0,java_script_enabled=False)
        def route_handler(route):
            u=route.request.url.lower()
            if u.startswith("file:") or u.startswith("data:") or u.startswith("about:"):
                route.continue_()
            else:
                route.abort()
        context.route("**/*",route_handler)
        page=context.new_page()
        page.set_default_timeout(10000)
        for r in rows:
            path=(ROOT/r["html_path"]).resolve()
            err=None
            try:
                page.goto(path.as_uri(),wait_until="domcontentloaded",timeout=10000)
                try:
                    page.add_style_tag(content="*,*::before,*::after{animation:none!important;transition:none!important;caret-color:transparent!important}html{scroll-behavior:auto!important}")
                except Exception:
                    pass
                png=page.screenshot(type="png",full_page=False,animations="disabled")
            except Exception as e:
                err=type(e).__name__+": "+str(e)
                raw=path.read_text(encoding="utf-8",errors="ignore")
                page.set_content(raw,wait_until="domcontentloaded",timeout=10000)
                png=page.screenshot(type="png",full_page=False,animations="disabled")
            pngs.append(png)
            meta.append({"domain":r["domain"],"snapshot_id":r["snapshot_id"],"render_fallback_error":err})
        page.close(); context.close(); browser.close()
    return pngs,meta,version

def main():
    rows=[json.loads(x) for x in SNAPS.read_text(encoding="utf-8").splitlines() if x.strip()]
    OUT.mkdir(parents=True,exist_ok=True); SHOTS.mkdir(parents=True,exist_ok=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor=CLIPProcessor.from_pretrained(MODEL_ID,revision=COMMIT)
    model=CLIPModel.from_pretrained(MODEL_ID,revision=COMMIT)
    model.eval().to(device)
    actual=getattr(model.config,"_commit_hash",None)
    if actual and actual!=COMMIT: raise RuntimeError(f"CLIP commit mismatch {actual} != {COMMIT}")
    errors=[]; fallback_count=0; generated=0; browser_version=None; shot_manifest=[]
    started=time.time()
    for start in range(0,len(rows),BATCH):
        batch=rows[start:start+BATCH]
        try:
            pngs,rmeta,browser_version=render_batch(batch)
            imgs=[Image.open(io.BytesIO(b)).convert("RGB") for b in pngs]
            inputs=processor(images=imgs,return_tensors="pt")
            pix=inputs["pixel_values"].to(device)
            with torch.inference_mode():
                feats=model.get_image_features(pixel_values=pix)
                feats=feats/feats.norm(dim=-1,keepdim=True).clamp(min=1e-12)
                arr=feats.detach().float().cpu().numpy()
            if arr.shape!=(len(batch),512) or not np.isfinite(arr).all(): raise RuntimeError(f"bad CLIP output {arr.shape}")
            for j,r in enumerate(batch):
                out=OUT/r["domain"]/f"{r['snapshot_id']}.npy"; out.parent.mkdir(parents=True,exist_ok=True)
                shot=SHOTS/r["domain"]/f"{r['snapshot_id']}.png"; shot.parent.mkdir(parents=True,exist_ok=True)
                np.save(out,arr[j].astype(np.float32)); shot.write_bytes(pngs[j])
                if rmeta[j]["render_fallback_error"]: fallback_count+=1
                shot_manifest.append({"domain":r["domain"],"snapshot_id":r["snapshot_id"],"screenshot_path":str(shot.relative_to(ROOT)).replace("\\","/"),"screenshot_sha256":sha_bytes(pngs[j]),"visual_path":str(out.relative_to(ROOT)).replace("\\","/"),"fallback_error":rmeta[j]["render_fallback_error"]})
                generated+=1
        except Exception as e:
            for r in batch: errors.append({"domain":r["domain"],"snapshot_id":r["snapshot_id"],"error":type(e).__name__+": "+str(e)})
        print("visual",min(start+BATCH,len(rows)),"/",len(rows),"errors",len(errors),flush=True)
    sm=V4/"visual_manifest.jsonl"
    shot_manifest.sort(key=lambda x:(x["domain"],x["snapshot_id"]))
    with sm.open("w",encoding="utf-8",newline="\n") as f:
        for r in shot_manifest:f.write(json.dumps(r,sort_keys=True,separators=(",",":"))+"\n")
    criteria={"all_visuals_generated":generated==len(rows),"no_batch_errors":not errors}
    verdict="PASS" if all(criteria.values()) else "FAIL_REPAIR"
    prov={"gate_id":"G1e_visual_generation","verdict":verdict,"model_id":MODEL_ID,"model_commit_hash":actual,"expected_commit_hash":COMMIT,"model_mode":"eval","normalization":"L2-normalized CLIP image_features","output_dim":512,"render_policy":"Playwright Chromium headless, JavaScript disabled, external network blocked, fixed 1280x720 viewport, device scale 1","browser_version":browser_version,"playwright_screenshot_saved":True,"snapshots":len(rows),"generated":generated,"fallback_render_count":fallback_count,"errors":errors[:50],"criteria":criteria,"visual_manifest_sha256":hashlib.sha256(sm.read_bytes()).hexdigest(),"torch_version":torch.__version__,"transformers_version":transformers.__version__,"python_version":platform.python_version(),"cuda_device":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"elapsed_seconds":time.time()-started}
    PROV.write_text(json.dumps(prov,indent=2)+"\n",encoding="utf-8")
    REPORT.write_text(json.dumps(prov,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(prov,indent=2))
    return 0 if verdict=="PASS" else 3
if __name__=="__main__": raise SystemExit(main())
