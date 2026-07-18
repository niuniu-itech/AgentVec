"""Local orchestrator: upload difftest harness to server, run it, print JSON summary."""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ssh_helper import run, put, HOSTS

LOCAL = os.path.dirname(os.path.abspath(__file__))
RROOT = "/path/to/agentvec_lab/difftest"
GCC = HOSTS["server"]["riscv_gcc"]
QEMU = HOSTS["server"]["qemu"]

def upload_tree():
    for dirpath, _, files in os.walk(LOCAL):
        for fn in files:
            if fn.endswith((".pyc",)) or "/build/" in dirpath.replace("\\", "/"):
                continue
            lp = os.path.join(dirpath, fn)
            rel = os.path.relpath(lp, LOCAL).replace("\\", "/")
            put("server", lp, f"{RROOT}/{rel}")
    print("uploaded ->", RROOT)

def main():
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    upload_tree()
    cmd = (f"cd {RROOT} && python3 runner.py --root {RROOT} --gcc {GCC} --qemu {QEMU} "
           f"--seeds {seeds} --neg-control")
    rc, out, err = run("server", cmd, timeout=1800)
    print("rc", rc)
    if err.strip():
        print("STDERR:", err[:800])
    print(out)

if __name__ == "__main__":
    main()
