# Paper-to-code coverage

This map describes the public implementation, not a claim that every paper result
has been independently reproduced with this revision.

| Paper component | Code | Coverage |
| --- | --- | --- |
| AIR intent and typed expressions | `air.py`, `air_from_formula.py` | f32/i32 map and reduce; complete expression parsing; unsupported forms rejected |
| V-AIR state transitions | `guard.py`, `pipeline.py` | CHECKED after contract checks; VERIFIED only after build and all independent checks |
| Independent evidence extraction | registered source contracts in `pipeline.py`; view-specific experiment drivers | No general C/PTX/SASS dependence, alias or bound extractor in the package |
| Deterministic VLA lowering | `lowering.py` | Unit-stride LMUL=1 core; separate f64 BLAS emitters in `blas_lower.py` |
| P-AIR feasibility | `symbolic_align.py`, `schedules.py` | Explicit K1 resource rules and identifier-only ranking |
| GEMM tuning | `experiments/drivers/gemm_opt.py`, `run_baseline_B.py`, `run_gemm_opt.py` | Study-specific driver; requires external OpenBLAS extraction/build inputs |
| Ranked convergence | `experiments/drivers/run_C.py` | Replays recorded board measurements; fresh ranking is not fresh target timing |
| Guarded NRM2 | `nrm2_guarded_dispatch.c` | Unit-stride finite domain, overflow/underflow fallback and native negative controls |
| Source stress and non-source views | `experiments/drivers/run_study_expanded.py`, `run_study_l2_expanded.py`, `run_s04_repair_sourcefree.py` | Separate experiment adapters and inputs; not one general source compiler |
| Composite DAGs | `dag.py`, `dag_lowering.py`, `dag_pipeline.py` | Typed f32 map/reduction graphs, scalar broadcasts, SSA/dependency/shape checks, topological lowering and single-use map fusion; independent references for four composites |
| AscendC external-validity probe | `ascend/`, `benchmarks/ascend/`, `results/ascend310p/` | Ten registered f32 templates, two dependence vetoes, host references, CANN runner and archived source/measurement materials; current device acceptance is distinct from historical records |
| Differential numerical policy | `difftest.py` | Exact integers; explicit relative/absolute tolerances and optional additional ULP bound |

The six-kernel CLI checks registered SAXPY, SCAL, COPY, DOT, ASUM and NRM2. It is a
working subset of the method, not the complete 80-instance or multi-model campaign.
Caller-provided contracts assume valid buffers and disjoint arrays over `0 <= i < n`;
the generic generated function does not inspect pointer provenance at runtime.

The paper describes bounded-ULP and relative-error acceptance. Existing released
specifications often contain only `rtol`. Such a record must be described as a
relative/absolute check unless an explicit `max_ulps` is supplied. A passing dynamic
test is evidence for its tested domain, not a symbolic proof over all inputs.

`difftest.py` reports two different quantities: per-check pass rate and all-checks
kernel acceptance. `spr` in a new runner record is binary per kernel. Do not combine
this denominator with older experiment summaries that aggregate dynamic checks.

Further implementation priorities are independent source-evidence extraction with
provenance, general tensor-shaped DAG nodes, and common V-AIR/P-AIR orchestration
across the remaining specialized adapters. Recorded numerical results are retained
unchanged; private machine paths in imported materials are replaced by placeholders.
Changed lowering or tolerance policies require new measurements.
