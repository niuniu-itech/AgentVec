# Performance Metric Suite — cuBLAS → Ascend310P1 (AscendC)

Measured on the target device (Ascend310P1, CANN 8.0.RC3). Each kernel is the
deterministically-lowered V-AIR/P-AIR realization; correctness is differential vs a
double-precision oracle (all VERIFIED ops pass). Methodology below is reported
explicitly so the numbers are interpretable rather than headline-only.

## Methodology (the metric suite)

| metric | definition |
|--------|-----------|
| latency | mean µs/call, warmup excluded, ≥20–30 timed iters |
| working set | bytes touched per call; **L2-resident if < 16 MiB** (Ascend310P1 L2), else **DRAM-resident** |
| effective bandwidth | bytes_moved / time (reads+writes counted once) |
| bandwidth efficiency | achieved BW / **measured DRAM ceiling** |
| compute | FLOPs / time (axpy 2n, dot 2n, gemv 2mn, gemm 2mnk) |
| compute efficiency | achieved / **vector-fp32 peak** = 1.024 TFLOP/s (8 vec cores × 64 fp32 × 2 × 1.0 GHz) |
| arithmetic intensity | FLOP / byte → roofline regime (memory- vs compute-bound) |
| P-AIR speedup | autotuned-best schedule / default schedule |

**Measured DRAM bandwidth ceiling = 170.7 GB/s** (scopy, 128 MiB working set; ≈84 % of
the ~204 GB/s LPDDR theoretical). This is the roofline ceiling for memory-bound ops.

> Why sizing matters: at n=2²⁰ the saxpy working set is 12 MiB < 16 MiB L2, so it reads
> 644 GB/s — that is **L2 bandwidth, not DRAM**. The suite sweeps sizes to separate the two.

## L1 BLAS (memory-bound; AI ≤ 0.5 FLOP/byte) — bandwidth vs the 170.7 GB/s ceiling

| op | n=2²⁰ (12 MiB, L2) | n=2²² (DRAM) | n=2²⁴ (DRAM) | DRAM BW eff. |
|----|--------------------|--------------|--------------|--------------|
| scopy | 477 GB/s (L2) | 164 GB/s | 170.1 GB/s | **99.7 %** |
| sscal | 555 GB/s (L2) | 163 GB/s | 170.7 GB/s | **100 %** |
| saxpy | 644 GB/s (L2) | 166 GB/s | 171.0 GB/s | **100 %** |
| sdot | 386 GB/s (L2) | 170 GB/s | 178.1 GB/s | **104 %**¹ |
| sasum | 227 GB/s (L2) | 446 GB/s (16 MiB≈L2) | 175.9 GB/s | **103 %**¹ |
| snrm2 | 228 GB/s (L2) | 451 GB/s (16 MiB≈L2) | 175.8 GB/s | **103 %**¹ |

¹ slightly >100 % because the reduction reads one array (the ceiling was measured on a
2-array copy); the DRAM-resident result is at the memory-bandwidth roofline either way.
All L1 ops are **bit-exact** (maxrel = 0) at every size.

**Takeaway:** at DRAM-resident sizes every L1 kernel saturates memory bandwidth
(~100 % of the 170.7 GB/s ceiling) — optimal for memory-bound operators.

## L2 GEMV (memory-bound, AI = 0.5 FLOP/byte)

| m×n | working set | latency | GFLOP/s | eff. BW | maxrel |
|-----|-------------|---------|---------|---------|--------|
| 1024² | 4 MiB (L2) | 115 µs | 18.2 | 36.4 GB/s | 4.9e-7 |
| 2048² | 16 MiB | 354 µs | 23.7 | 47.5 GB/s | ~1e-6 |
| 4096² | 64 MiB (DRAM) | 1239 µs | 27.1 | 54.2 GB/s | ~1e-6 |

GEMV is bandwidth-bound by A (read once). Effective BW is below the L1 ceiling because the
per-row reduction carries cross-pipe sync overhead — the natural P-AIR optimization target
(wider row tiling / fewer barriers).

## L3 GEMM (compute-bound, AI = 43–85 FLOP/byte) — vs vector-fp32 peak (1.024 TFLOP/s)

| m×n×k | latency | GFLOP/s | % vector peak | regime |
|-------|---------|---------|---------------|--------|
| 256³ | 1.54 ms | 21.8 | 2.1 % | compute-bound |
| 512³ | 6.04 ms | 44.4 | 4.3 % | compute-bound |
| 1024³ | 24.0 ms | 89.4 | 8.7 % | compute-bound |

GFLOP/s scales with size (more reuse amortizes the rank-1 loop). This is the **vector-unit**
realization (Muls+Add); the Cube engine (16×16×16 fp16 MACs) is the P-AIR upgrade path that
would lift this by ~1–2 orders of magnitude — left as the documented next step.

## P-AIR (performance alignment) — schedule autotuning

Align(I,H) bounds the schedule space to a sound subset B (tile ≤ UB, block_dim ≤ cores,
double-buffer legal); the model orders it (largest-tile-first); the autotuner measures.

- **saxpy, L2-resident (n=2²⁰):** tile 256→8192 sweeps **53.8 → 664.6 GB/s = 12.3×**; the
  prior hits the optimum on trial 1 (Proposition 1: monotone-safe).
- **saxpy, DRAM-resident (n=2²⁴):** default t1024 158.9 → best t8192 170.8 GB/s = **1.08×**
  (already bandwidth-saturated, so tile choice matters less — the suite makes this visible).

The lesson the metric suite encodes: tile autotuning is decisive in the cache regime and
near-flat once DRAM bandwidth saturates — both are reported, not just the flattering one.

Raw data: `artifacts/metrics.json`. Per-op P-AIR records: `artifacts/<op>.pair.json`.
