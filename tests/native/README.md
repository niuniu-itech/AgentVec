# Native and QEMU checks

Generate an edge-case executable after installing AgentVec:

```bash
python tests/native/generate_edge_cases.py --output /tmp/agentvec-tests/edges.c
gcc -O3 -ffp-contract=off -march=rv64gcv /tmp/agentvec-tests/edges.c -lm -o /tmp/agentvec-tests/edges
/tmp/agentvec-tests/edges
```

The test covers every length from 0 through 258, empty reductions, extreme
finite min/max values, absolute value, and modular i32 arithmetic. It uses
independently written expected computations, not the scalar AIR interpreter.

For cross-VLEN checks, cross-compile statically and run the binary with QEMU at
VLEN 128, 256, 512, and 1024. A native K1 run validates its installed VLEN only.

The guarded double-precision norm has its own executable:

```bash
gcc -O3 -march=rv64gcv src/agentvec/nrm2_guarded_dispatch.c -lm -o /tmp/agentvec-tests/nrm2
/tmp/agentvec-tests/nrm2
```

These are correctness checks. They do not reproduce paper timing or LLM pass rates.

`generate_blas_cases.py --output /tmp/agentvec-tests/blas.c` produces compile
coverage for all ten f64 BLAS templates and independent runtime checks for DOT,
ASUM, NRM2, MIN and MAX at unit stride and stride two. Compile and run it with the
same flags as the edge test. This does not link or benchmark an OpenBLAS expert.
