"""Paramiko-based SSH/SFTP helper for AgentVec experiments (anonymized).

The experiments compile RVV kernels on a build host (x86 with a RISC-V cross-gcc +
qemu-riscv64) and, for on-hardware performance numbers, on a real RISC-V board. No
credentials are bundled with this artifact: provide your own hosts via environment
variables (key-based auth recommended).

    export AGENTVEC_SERVER_HOST=...   AGENTVEC_SERVER_USER=...
    export AGENTVEC_SERVER_KEY=~/.ssh/id_ed25519        # or AGENTVEC_SERVER_PASSWORD=...
    export AGENTVEC_SERVER_GCC=/path/to/riscv64-...-gcc  AGENTVEC_SERVER_QEMU=/path/to/qemu-riscv64
    # optional board (performance runs):
    export AGENTVEC_BOARD_HOST=...    AGENTVEC_BOARD_USER=...    AGENTVEC_BOARD_KEY=...

Usage:
    from ssh_helper import run, put, get, HOSTS
    rc, out, err = run("server", "uname -a")
"""
import os
import paramiko

HOSTS = {
    "server": {
        "host": os.environ.get("AGENTVEC_SERVER_HOST", ""),
        "user": os.environ.get("AGENTVEC_SERVER_USER", ""),
        "password": os.environ.get("AGENTVEC_SERVER_PASSWORD"),
        "key": os.environ.get("AGENTVEC_SERVER_KEY"),
        "riscv_gcc": os.environ.get("AGENTVEC_SERVER_GCC", "riscv64-linux-gnu-gcc"),
        "qemu": os.environ.get("AGENTVEC_SERVER_QEMU", "qemu-riscv64"),
    },
    "board": {
        "host": os.environ.get("AGENTVEC_BOARD_HOST", ""),
        "user": os.environ.get("AGENTVEC_BOARD_USER", ""),
        "password": os.environ.get("AGENTVEC_BOARD_PASSWORD"),
        "key": os.environ.get("AGENTVEC_BOARD_KEY"),
    },
}


def _client(name):
    cfg = HOSTS[name]
    if not cfg.get("host"):
        raise RuntimeError(
            f"No host configured for '{name}'. Set AGENTVEC_{name.upper()}_HOST / _USER / "
            f"_KEY (or _PASSWORD); see the module docstring and README.")
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    kw = dict(username=cfg["user"], timeout=20, banner_timeout=20, auth_timeout=20)
    if cfg.get("key"):
        kw.update(key_filename=os.path.expanduser(cfg["key"]), look_for_keys=True, allow_agent=True)
    elif cfg.get("password"):
        kw.update(password=cfg["password"], look_for_keys=False, allow_agent=False)
    else:
        kw.update(look_for_keys=True, allow_agent=True)   # fall back to ssh-agent / default keys
    c.connect(cfg["host"], **kw)
    return c


def run(name, cmd, timeout=120):
    """Run a command; return (exit_code, stdout, stderr)."""
    c = _client(name)
    try:
        _, so, se = c.exec_command(cmd, timeout=timeout)
        out = so.read().decode("utf-8", "replace")
        err = se.read().decode("utf-8", "replace")
        rc = so.channel.recv_exit_status()
        return rc, out, err
    finally:
        c.close()


def put(name, local, remote):
    c = _client(name)
    try:
        sftp = c.open_sftp()
        rdir = os.path.dirname(remote)
        if rdir:
            run(name, f"mkdir -p {rdir}")
        sftp.put(local, remote)
        sftp.close()
    finally:
        c.close()


def get(name, remote, local):
    c = _client(name)
    try:
        os.makedirs(os.path.dirname(local) or ".", exist_ok=True)
        sftp = c.open_sftp()
        sftp.get(remote, local)
        sftp.close()
    finally:
        c.close()


if __name__ == "__main__":
    for name in ("server", "board"):
        try:
            rc, out, err = run(name, "echo HOST=$(hostname); uname -m; nproc 2>/dev/null || echo nproc?", timeout=25)
            print(f"[{name}] rc={rc}\n{out}{err}".strip())
        except Exception as e:
            print(f"[{name}] not configured / unreachable: {type(e).__name__}: {e}")
        print("-" * 50)
