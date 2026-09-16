# Research experiments

`drivers/` contains one copy of each study driver. `scripts/` keeps compatibility
entry points. These drivers depend on prepared external inputs; they are not all
covered by the installed six-kernel migration command.

Before using a driver:

1. Set `AGENTVEC_EXPERIMENT_DIR` to an external run workspace.
2. Copy the required differential inputs into its `difftest/` directory. For remote
   execution, deploy `src/agentvec/difftest.py` as `runner.py` and the harness too.
3. Copy required recorded JSON inputs from `results/rvv/` when running a replay.
4. For OpenBLAS measurements, prepare the extracted operators, `common.h`, benchmark
   drivers and RVV library under `AGENTVEC_EXTRACT_ROOT` on the build host. Those
   external build products are not shipped by this repository.
5. Configure the server/board as described in [configuration](../docs/configuration.md),
   and execute from the external run directory so relative products stay there.

| Study | Entry points |
| --- | --- |
| Registered intent vs direct code | `run_study`, `run_study_reduce` |
| Expanded L1/L2 stress cases | `run_study_expanded`, `run_study_l2_expanded` |
| Noisy/source-free inputs | `run_s04_repair_sourcefree` |
| Assembly, trace and composite views | `exp_assembly_view`, `exp_trace_view`, `exp_composite`, `exp_layernorm` |
| BLAS performance and dispersion | `run_l1_board`, `run_l1_stats`, `run_l2_ops` |
| GEMM schedule sweep | `run_gemm_opt`, `run_baseline_B` |
| Recorded schedule ranking replay | `run_C` |

For example, after preparing the workspace and adding the repository root and its
`src` directory to `PYTHONPATH`, run `python -m experiments.drivers.run_C`. This
asks the configured models to rank recorded candidates; it does not remeasure them.

Some view-specific drivers contain purpose-built templates and historical model
selections. Review their inputs and output contracts before a full campaign. New
core checks deliberately veto unsupported f64/gather/DAG proposals; use the relevant
specialized study adapter rather than bypassing the core guard.
