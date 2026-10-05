import json, os, shutil, subprocess, sys, time
from pathlib import Path

ROOT=Path(r"D:\RESEARCH\DI_MM_V4")
V4=ROOT/"dataset_pipeline"/"v4"
SNAPS=V4/"snapshot_manifest.jsonl"
SHOTS=V4/"screenshots"
VISUALS=V4/"visuals"
SHARDS=ROOT/"visual_render_shards"
LOGS=ROOT/"visual_parallel_logs"
GATES=ROOT/"gates"
PROJECT=Path(r".")
RENDER=PROJECT/"src"/"render_visual_shard.py"
EMBED=PROJECT/"src"/"embed_visuals.py"
NUM_SHARDS=4

def count_files(root,suffix):
    return sum(1 for _ in root.rglob("*"+suffix)) if root.exists() else 0

def clean_path(p):
    if p.is_dir():
        shutil.rmtree(p,ignore_errors=True)
    elif p.exists():
        p.unlink()

def main():
    rows=[json.loads(x) for x in SNAPS.read_text(encoding="utf-8").splitlines() if x.strip()]
    n=len(rows)
    if n!=4865:
        raise RuntimeError(f"unexpected snapshot count {n}")

    for p in [SHOTS,VISUALS,SHARDS,LOGS]:
        clean_path(p)
        p.mkdir(parents=True,exist_ok=True)
    for p in [V4/"visual_manifest.jsonl",V4/"visual_provenance.json",GATES/"G1e_visual_generation.json"]:
        if p.exists(): p.unlink()

    procs=[]
    files=[]
    env=os.environ.copy()
    env["PYTHONUNBUFFERED"]="1"
    started=time.time()
    for i in range(NUM_SHARDS):
        out=(LOGS/f"render_shard_{i:02d}.out.log").open("w",encoding="utf-8")
        err=(LOGS/f"render_shard_{i:02d}.err.log").open("w",encoding="utf-8")
        files.extend([out,err])
        cmd=[sys.executable,"-u",str(RENDER),"--shard-index",str(i),"--num-shards",str(NUM_SHARDS)]
        p=subprocess.Popen(cmd,stdout=out,stderr=err,cwd=str(ROOT),env=env)
        procs.append((i,p))
        print(f"LAUNCHED shard={i} pid={p.pid}",flush=True)

    last=-1
    failed=None
    while True:
        done=0
        for i,p in procs:
            rc=p.poll()
            if rc is not None:
                done+=1
                if rc!=0 and failed is None:
                    failed=(i,rc)
        c=count_files(SHOTS,".png")
        if c!=last:
            print(f"RENDER_PROGRESS screenshots={c}/{n} workers_done={done}/{NUM_SHARDS}",flush=True)
            last=c
        if failed is not None:
            for _,p in procs:
                if p.poll() is None:
                    p.terminate()
            for _,p in procs:
                try:p.wait(timeout=15)
                except Exception:p.kill()
            raise RuntimeError(f"render shard failed {failed}")
        if done==NUM_SHARDS:
            break
        time.sleep(10)

    for f in files:
        f.close()

    reports=[]
    for i in range(NUM_SHARDS):
        rp=SHARDS/f"render_shard_{i:02d}_report.json"
        if not rp.exists():
            raise RuntimeError(f"missing shard report {rp}")
        r=json.loads(rp.read_text(encoding="utf-8"))
        reports.append(r)
        if r.get("verdict")!="PASS":
            raise RuntimeError(f"shard {i} report not PASS: {r.get('verdict')}")
    selected=sum(int(r["selected"]) for r in reports)
    generated=sum(int(r["generated"]) for r in reports)
    screenshot_count=count_files(SHOTS,".png")
    if selected!=n or generated!=n or screenshot_count!=n:
        raise RuntimeError(f"render coverage mismatch selected={selected} generated={generated} screenshots={screenshot_count} expected={n}")

    render_elapsed=time.time()-started
    print(f"RENDER_PASS screenshots={screenshot_count} elapsed={render_elapsed:.1f}s",flush=True)

    embed_start=time.time()
    rc=subprocess.run([sys.executable,"-u",str(EMBED)],cwd=str(ROOT),env=env).returncode
    if rc!=0:
        raise RuntimeError(f"CLIP embedding failed exit={rc}")
    embed_elapsed=time.time()-embed_start

    gate=GATES/"G1e_visual_generation.json"
    if not gate.exists():
        raise RuntimeError("G1e visual gate missing after embedding")
    g=json.loads(gate.read_text(encoding="utf-8"))
    if g.get("verdict")!="PASS":
        raise RuntimeError(f"G1e visual gate not PASS: {g.get('verdict')}")
    visual_count=count_files(VISUALS,".npy")
    if visual_count!=n:
        raise RuntimeError(f"visual vector count {visual_count} != {n}")

    report={
        "gate_id":"G1e_visual_parallel_supervision",
        "verdict":"PASS",
        "snapshot_count":n,
        "render_workers":NUM_SHARDS,
        "render_reports":[str((SHARDS/f"render_shard_{i:02d}_report.json")) for i in range(NUM_SHARDS)],
        "screenshots":screenshot_count,
        "visual_vectors":visual_count,
        "render_elapsed_seconds":render_elapsed,
        "clip_elapsed_seconds":embed_elapsed,
        "total_elapsed_seconds":time.time()-started,
        "g1e_visual_generation":str(gate),
    }
    (GATES/"G1e_visual_parallel_supervision.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2),flush=True)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
