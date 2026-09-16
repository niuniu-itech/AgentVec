# cuBLAS -> Ascend310P1 — V-AIR correctness & default-schedule performance

fp32, differential vs a double-precision oracle. L1 n=2^20; sgemv 1024x1024; sgemm 512x512x512. Default schedule b8/t1024/q2 (P-AIR autotuned bests in PAIR_SUMMARY.md).

| op | lvl | pattern | V-AIR | maxrel | us/call | throughput |
|----|-----|---------|-------|--------|---------|------------|
| saxpy | L1 | map | PASS | 0e+00 | 64.55 | 194.9 GB/s |
| sscal | L1 | map | PASS | 0e+00 | 45.87 | 182.9 GB/s |
| scopy | L1 | map | PASS | 0e+00 | 45.75 | 183.4 GB/s |
| sdot | L1 | reduce | PASS | 0e+00 | 48.58 | 172.7 GB/s |
| sasum | L1 | reduce | PASS | 0e+00 | 31.37 | 133.7 GB/s |
| snrm2 | L1 | reduce | PASS | 0e+00 | 31.29 | 134.1 GB/s |
| sgemv | L2 | gemv | PASS | 5e-07 | 112.98 | 18.6 GFLOP/s |
| sgemm | L3 | gemm | PASS | 9e-07 | 6029.16 | 44.5 GFLOP/s |
| sgemm_nt | L3 | gemm_nt | PASS | 3e-07 | 18218.50 | 14.7 GFLOP/s |
| ssyrk | L3 | syrk | PASS | 2e-07 | 18225.10 | 14.7 GFLOP/s |
| strsv | L2 | trsv | VETO | - | - | - |
| strsm | L3 | trsm | VETO | - | - | - |
