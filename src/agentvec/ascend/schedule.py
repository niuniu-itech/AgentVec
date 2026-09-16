"""Bounds for the released Ascend310P1 vector templates."""
from dataclasses import asdict
import math

from .air import Pattern


def check_schedule(air, sched):
    if any(type(v) is not int for v in asdict(sched).values()):
        raise ValueError("schedule fields must be integers")
    if not 1 <= sched.block_dim <= 8 or sched.buffer_num not in (1, 2):
        raise ValueError("schedule exceeds the eight-block profile or queue-depth range")
    if (sched.tile_m, sched.tile_n, sched.tile_k) != (64, 64, 64):
        raise ValueError("these matrix tile fields are reserved; this backend does not apply them")
    if sched.tile_len < 64 or sched.tile_len % 64:
        raise ValueError("tile_len must be a positive multiple of 64")
    if air.s_algo.pattern == Pattern.MAP:
        queues = 3 if air.name == "saxpy" else 2
        ub = queues * sched.buffer_num * sched.tile_len * 4
    elif air.s_algo.pattern == Pattern.REDUCE:
        queues = 2 if air.s_algo.reduce_pre == "ab" else 1
        ub = (queues * sched.buffer_num + 3) * sched.tile_len * 4 + 32
    else:
        ub = 0  # Matrix storage depends on m/n/k and is checked by check_shape.
    if ub > 262144:
        raise ValueError("schedule exceeds the 256 KiB UB budget")
    if air.s_algo.pattern not in (Pattern.MAP, Pattern.REDUCE) and (sched.tile_len, sched.buffer_num) != (1024, 2):
        raise ValueError("matrix kernels have fixed inner-buffer scheduling")
    return ub


def check_shape(air, sched, sizes, iterations):
    check_schedule(air, sched)
    p = air.s_algo.pattern
    expected = 1 if p in (Pattern.MAP, Pattern.REDUCE) else 2 if p == Pattern.GEMV else 3
    if len(sizes) != expected or any(type(n) is not int or n < 1 or n > 67108864 for n in sizes):
        raise ValueError("invalid extent count or positive-size domain")
    if type(iterations) is not int or not 1 <= iterations <= 100000:
        raise ValueError("iterations must be in 1..100000")
    if p in (Pattern.MAP, Pattern.REDUCE):
        if sizes[0] % (sched.block_dim * sched.tile_len):
            raise ValueError("L1 size must be divisible by blocks * tile_len; no truncation is allowed")
    elif p == Pattern.GEMV:
        m, n = sizes
        if n % 64 or m % (8 * sched.block_dim):
            raise ValueError("GEMV requires n % 64 = 0 and m % (8 * blocks) = 0")
        if (4 * n + 4 * (m // sched.block_dim)) * 4 + 544 > 262144:
            raise ValueError("GEMV rows and buffers exceed UB")
    else:
        m, n, k = sizes
        if m % sched.block_dim or n % 8 or k % (8 if p == Pattern.GEMM else 64):
            raise ValueError("matrix shape violates the registered vector alignment")
        if p == Pattern.SYRK and m != n:
            raise ValueError("SYRK requires m = n")
        ub = (5 * n + k) * 4 if p == Pattern.GEMM else (4 * k + 3 * n) * 4 + 544
        if ub > 262144:
            raise ValueError("matrix row buffers exceed UB")
    if math.prod(sizes) > 2**34:
        raise ValueError("problem exceeds the bounded validation workload")


def legal_schedules(air):
    from .lowering import Sched
    candidates = [Sched()]
    if air.s_algo.pattern in (Pattern.MAP, Pattern.REDUCE):
        candidates = [Sched(block_dim=b, tile_len=t, buffer_num=q)
                      for b in (1, 4, 8) for t in (1024, 2048, 4096, 8192) for q in (1, 2)]
    admitted = []
    for sched in candidates:
        try:
            check_schedule(air, sched)
            admitted.append(sched)
        except ValueError:
            pass
    return admitted
