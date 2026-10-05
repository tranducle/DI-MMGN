import json
from collections import Counter
from pathlib import Path
ROOT=Path(r"D:\RESEARCH\DI_MM_V4")
META=ROOT/"acquisition"/"meta"
OUT=ROOT/"gates"/"G0_network_repair_decision.json"
network=0; total=0; classes=Counter(); failed=0
for p in META.rglob("*.json"):
    x=json.loads(p.read_text(encoding="utf-8"))
    if x.get("status")!="failed": continue
    failed+=1
    for a in x.get("attempts",[]):
        total+=1
        if "error" in a:
            typ=a["error"].split(":",1)[0]; classes[typ]+=1
            if typ in {"ConnectionError","ReadTimeout","ConnectTimeout","Timeout","TimeoutError"}: network+=1
        elif "http_status" in a:
            st=int(a["http_status"]); classes[f"HTTP_{st}"]+=1
            if st>=500: network+=1
frac=network/total if total else 0
trigger=failed>0 and total>0 and frac>=0.5
rep={"failed_records":failed,"failed_attempts":total,"network_or_5xx_attempts":network,"network_fraction":frac,"attempt_classes":dict(classes),"repair_trigger":trigger,"criterion":"network_or_5xx_fraction >= 0.5"}
OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(rep,indent=2)+"\n",encoding="utf-8"); print(json.dumps(rep,indent=2))
raise SystemExit(0 if trigger else 4)
