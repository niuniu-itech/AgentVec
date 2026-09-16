# P-AIR autotuning summary (Ascend310P1)

Align(I,H) bounds the schedule space to a sound subset B; the model orders it; the on-device autotuner keeps the measured best (Proposition 1: monotone-safe).

| op | lvl | pattern | best schedule | best us | best GB/s or GFLOPs |
|----|-----|---------|---------------|---------|----------------------|
| saxpy | L1 | map | b8_t8192_q2 | 18.64 | 675.2 |
| sscal | L1 | map | b8_t8192_q2 | 14.30 | 586.7 |
| scopy | L1 | map | b8_t8192_q2 | 14.58 | 575.4 |
| sdot | L1 | reduce | b8_t4096_q2 | 19.20 | 436.9 |
| sasum | L1 | reduce | b8_t4096_q2 | 17.46 | 240.2 |
| snrm2 | L1 | reduce | b8_t4096_q2 | 18.15 | 231.1 |
| sgemv | L2 | gemv | b8_t1024_q2 | 113.16 | 18.5 |
| sgemm | L3 | gemm | b8_t1024_q2 | 6026.68 | 44.5 |
| strsv | L2 | trsv | - | - | - |
