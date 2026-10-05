import argparse, os, subprocess, sys
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--log",required=True)
    ap.add_argument("cmd",nargs=argparse.REMAINDER)
    args=ap.parse_args()
    if not args.cmd:
        raise SystemExit("missing child command")
    log=Path(args.log)
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open("a",encoding="utf-8",errors="replace") as f:
        f.write("\n=== CHILD START ===\n")
        f.write("CMD: "+" ".join(args.cmd)+"\n")
        f.flush()
        p=subprocess.Popen(
            args.cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=os.environ.copy(),
        )
        assert p.stdout is not None
        for line in p.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            f.write(line)
            f.flush()
        rc=p.wait()
        f.write(f"=== CHILD EXIT {rc} ===\n")
        f.flush()
    return rc

if __name__=="__main__":
    raise SystemExit(main())
