import json, os, subprocess, sys, time
from pathlib import Path

ROOT=Path(r"D:\RESEARCH\DI_MM_V4")
PROJECT=Path(r".")
REPAIR=PROJECT/"src"/"repair_visual_shard.py"
MISSING=ROOT/"visual_repair"/"missing_visual_rows.jsonl"
OUTDIR=ROOT/"visual_repair"
LOGDIR=OUTDIR/"logs"
SHOTS=ROOT/"dataset_pipeline"/"v4"/"screenshots"
GATE=ROOT/"gates"/"G1e_visual_repair.json"
NUM_SHARDS=4

def count_png():
    return sum(1 for _ in SHOTS.rglob("*.png")) if SHOTS.exists() else 0

def main():
    rows=[json.loads(x) for x in MISSING.read_text(encoding="utf-8").splitlines() if x.strip()]
    LOGDIR.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    env["PYTHONUNBUFFERED"]="1"
    procs=[]; handles=[]
    started=time.time()
    for i in range(NUM_SHARDS):
        out=(LOGDIR/f"repair_{i:02d}.out.log").open("w",encoding="utf-8")
        err=(LOGDIR/f"repair_{i:02d}.err.log").open("w",encoding="utf-8")
        handles += [out,err]
        cmd=[sys.executable,"-u",str(REPAIR),"--shard-index",str(i),"--num-shards",str(NUM_SHARDS)]
        p=subprocess.Popen(cmd,cwd=str(ROOT),env=env,stdout=out,stderr=err)
        procs.append((i,p))
        print(f"LAUNCHED repair_shard={i} pid={p.pid}",flush=True)
    last=-1
    while True:
        done=sum(1 for _,p in procs if p.poll() is not None)
        c=count_png()
        if c!=last:
            print(f"REPAIR_PROGRESS screenshots={c}/4865 workers_done={done}/{NUM_SHARDS}",flush=True)
            last=c
        if done==NUM_SHARDS:
            break
        time.sleep(5)
    for h in handles: h.close()
    reports=[]; failures=[]
    for i,p in procs:
        rp=OUTDIR/f"repair_shard_{i:02d}_report.json"
        if not rp.exists():
            failures.append({"shard":i,"reason":"report_missing","exit_code":p.returncode})
            continue
        r=json.loads(rp.read_text(encoding="utf-8"))
        reports.append(r)
        if r.get("verdict")!="PASS" or p.returncode!=0:
            failures.append({"shard":i,"reason":"repair_failed","exit_code":p.returncode,"error_count":r.get("error_count")})
    final_count=count_png()
    verdict="PASS" if not failures and final_count==4865 else "FAIL_REPAIR"
    gate={
        "gate_id":"G1e_visual_repair",
        "verdict":verdict,
        "initial_missing":len(rows),
        "repair_workers":NUM_SHARDS,
        "final_screenshot_count":final_count,
        "expected_screenshot_count":4865,
        "failures":failures,
        "shard_reports":[str(OUTDIR/f"repair_shard_{i:02d}_report.json") for i in range(NUM_SHARDS)],
        "elapsed_seconds":time.time()-started,
        "decision":"proceed_to_clip_embedding" if verdict=="PASS" else "stop_and_repair_remaining_visuals",
    }
    GATE.write_text(json.dumps(gate,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(gate,indent=2),flush=True)
    return 0 if verdict=="PASS" else 3
if __name__=="__main__":
    raise SystemExit(main())
