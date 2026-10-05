import hashlib, json, math, re, time
from pathlib import Path
import numpy as np

ROOT=Path(r"D:\RESEARCH\DI_MM_V4")
V4=ROOT/"dataset_pipeline"/"v4"
SNAPS=V4/"snapshot_manifest.jsonl"
OUT=V4/"http"
REPORT=ROOT/"gates"/"G1d_http_generation.json"

FEATURES=[
"status_is_2xx","redirect_count_div5","log_content_bytes_div_log10mb","log_orig_content_length_div_log10mb",
"content_type_html","content_type_text","charset_present","orig_server_present",
"cache_control_present","expires_present","etag_present","last_modified_present",
"content_encoding_present","transfer_encoding_present","set_cookie_present","csp_present",
"hsts_present","x_frame_options_present","x_content_type_options_present","referrer_policy_present",
"permissions_policy_present","acao_present","location_present","vary_present",
"accept_ranges_present","age_present","x_powered_by_present","via_present",
"orig_header_count_div64","response_header_count_div64","effective_capture_hour_div23","requested_effective_delta_days_div31"
]
assert len(FEATURES)==32
SCHEMA_SHA=hashlib.sha256(json.dumps(FEATURES,separators=(",",":")).encode()).hexdigest()

def norm_log_bytes(v):
    try:v=max(0,int(v))
    except:return 0.0
    return min(math.log1p(v)/math.log(10*1024*1024+1),1.0)

def feature_vec(meta,hobj):
    hdr={str(k).lower():str(v) for k,v in hobj.get("response_headers",{}).items()}
    orig={k[len("x-archive-orig-"):]:v for k,v in hdr.items() if k.startswith("x-archive-orig-")}
    ct=(orig.get("content-type") or hdr.get("content-type") or "").lower()
    history=hobj.get("history",[])
    req=str(meta["requested_timestamp"]); eff=str(meta.get("effective_capture_timestamp") or req)
    try:
        from datetime import datetime
        a=datetime.strptime(req,"%Y%m%d%H%M%S"); b=datetime.strptime(eff,"%Y%m%d%H%M%S")
        delta=min(abs((b-a).total_seconds())/86400.0/31.0,1.0)
        hour=b.hour/23.0
    except Exception:
        delta=0.0; hour=0.0
    def has(name):
        return 1.0 if name in orig or name in hdr else 0.0
    x=np.array([
        1.0 if 200<=int(meta.get("http_status",0))<300 else 0.0,
        min(len(history)/5.0,1.0),
        norm_log_bytes(meta.get("content_bytes",0)),
        norm_log_bytes(orig.get("content-length",0)),
        1.0 if "html" in ct else 0.0,
        1.0 if "text" in ct else 0.0,
        1.0 if "charset=" in ct else 0.0,
        1.0 if "server" in orig else 0.0,
        has("cache-control"),has("expires"),has("etag"),has("last-modified"),
        has("content-encoding"),has("transfer-encoding"),has("set-cookie"),has("content-security-policy"),
        has("strict-transport-security"),has("x-frame-options"),has("x-content-type-options"),has("referrer-policy"),
        has("permissions-policy"),has("access-control-allow-origin"),has("location"),has("vary"),
        has("accept-ranges"),has("age"),has("x-powered-by"),has("via"),
        min(len(orig)/64.0,1.0),min(len(hdr)/64.0,1.0),hour,delta
    ],dtype=np.float32)
    assert x.shape==(32,)
    return x

def main():
    rows=[json.loads(x) for x in SNAPS.read_text(encoding="utf-8").splitlines() if x.strip()]
    OUT.mkdir(parents=True,exist_ok=True)
    cache={}; errors=[]; generated=0
    started=time.time()
    for i,r in enumerate(rows,1):
        domain=r["domain"]; ts=r["requested_timestamp"]
        key=(domain,ts)
        try:
            if key not in cache:
                mp=ROOT/"acquisition"/"meta"/domain/f"{ts}.json"
                hp=ROOT/"acquisition"/"headers"/domain/f"{ts}.json"
                meta=json.loads(mp.read_text(encoding="utf-8"))
                hobj=json.loads(hp.read_text(encoding="utf-8"))
                cache[key]=feature_vec(meta,hobj)
            x=cache[key]
            out=OUT/domain/f"{r['snapshot_id']}.npy"; out.parent.mkdir(parents=True,exist_ok=True)
            np.save(out,x)
            generated+=1
        except Exception as e:
            errors.append({"domain":domain,"snapshot_id":r["snapshot_id"],"error":type(e).__name__+": "+str(e)})
        if i%500==0 or i==len(rows): print("http",i,"/",len(rows),"errors",len(errors),flush=True)
    criteria={"all_http_vectors_generated":generated==len(rows),"no_errors":not errors}
    verdict="PASS" if all(criteria.values()) else "FAIL_REPAIR"
    rep={"gate_id":"G1d_http_generation","verdict":verdict,"snapshots":len(rows),"generated":generated,"errors":errors[:50],"feature_names":FEATURES,"feature_schema_sha256":SCHEMA_SHA,"unique_base_response_vectors":len(cache),"criteria":criteria,"elapsed_seconds":time.time()-started}
    REPORT.write_text(json.dumps(rep,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(rep,indent=2))
    return 0 if verdict=="PASS" else 3
if __name__=="__main__": raise SystemExit(main())
