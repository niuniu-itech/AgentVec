# Baseline comparison — with vs without the method (Ascend310P1, DRAM-resident)

`naive` = single AI core, small tile, no double-buffer (a direct/unoptimized port). `V-AIR` = deterministic 8-core tiled lowering. `P-AIR` = autotuned schedule.

| op | lvl | metric | naive (no method) | V-AIR (default) | P-AIR (best) | V-AIR speedup | P-AIR speedup |
|----|-----|--------|-------------------|-----------------|--------------|---------------|---------------|
| saxpy | L1 | GBps | 8.0 (25094us) | 158.6 (1269us) | 170.8 (1179us) | 19.8x | 21.3x |
| sdot | L1 | GBps | 5.7 (23624us) | 158.0 (849us) | 178.1 (754us) | 27.8x | 31.3x |
| sgemv | L2 | GFLOPs | 3.5 (9714us) | 27.1 (1239us) | 27.1 (1241us) | 7.8x | 7.8x |
| sgemm | L3 | GFLOPs | 5.6 (48108us) | 44.5 (6038us) | 44.5 (6037us) | 8.0x | 8.0x |

**Reading it:** the naive direct port (single core, no pipeline) is **8–31× slower**. For
memory-bound L1, V-AIR's multi-core tiling + double-buffering give ~20–28× (more than the
8× core count, because pipelining also hides DMA latency); P-AIR's tile autotuning adds the
rest. For compute-bound sgemm the gain tracks the 8 cores (~8×); the schedule is already at
the per-core optimum, so P-AIR is flat there.

**Vendor-library baseline (aclnn):** not available on this image — the toolkit ships the
inference runtime (`libnnopbase`) but **not** the high-level `aclnn` op API
(`libopapi`/`aclnnop/aclnn_matmul.h`), so a cuBLAS-vs-vendor-op comparison can't be run
here. The naive port above is therefore the "without-method" reference. (On a full CANN
install, drop an `aclnnMatmul` call into `baseline.py` for the vendor ceiling — expect it to
beat the vector sgemm by 1–2 orders via the Cube engine, which is the documented P-AIR upgrade.)
