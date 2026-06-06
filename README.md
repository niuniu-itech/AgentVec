# AgentVec — Reproducibility Artifact

Anonymous artifact for the paper **"AgentVec: Intent-Driven Source-to-Source Migration
for RISC-V VLA via Agentic-Symbolic Alignment."** It contains the migration pipeline,
the AIR schema and deterministic lowering, the symbolic Guard, all agent prompts, the
differential-testing harness, the cached model responses, the measured result data, and
the figure-regeneration scripts.

> This repository is anonymized for double-blind review. It bundles **no credentials**;
> hosts are supplied via environment variables (see below).

## Layout

```
experiments/
  agentvec/            # the pipeline
    air.py, air_from_formula.py    # AIR four-tuple schema + formula -> AIR
    guard.py                       # symbolic Guard (dependence / alias / type checks)
    lowering.py, blas_lower.py     # deterministic AIR -> RVV (VLA) lowering
    obfuscate.py                   # IM / DCI / type obfuscation (anti-memorization)
    llm_backend.py                 # Intent-Proposer / pure-LLM prompts + backend (reads SF_KEY)
    pipeline.py                    # end-to-end driver
    run_*.py, exp_*.py             # the individual experiments
    run_baseline_mini.py           # baseline head-to-head (GCC autovec / VecTrans / pure-LLM / AgentVec)
    _llm_cache/                    # cached model responses -> reproduce WITHOUT any API key
    _gen/                          # generated kernels
    _*.json                        # measured results (L1 stats, GEMM, cost, baseline mini-study, ...)
  lab/
    ssh_helper.py                  # SSH/SFTP helper (env-var hosts, no bundled credentials)
    difftest/                      # differential-testing harness (compile + QEMU across VLEN)
      runner.py, harness.h, kernels/
```

## Requirements

- Python 3.9+ with `paramiko` and `matplotlib`.
- A build host reachable over SSH with a **RISC-V cross GCC (14.3)** and **qemu-riscv64**
  (functional + cross-VLEN testing). Optionally a real RVV board for wall-clock numbers.

## Configure your hosts (no credentials are bundled)

```bash
export AGENTVEC_SERVER_HOST=your.host        AGENTVEC_SERVER_USER=you
export AGENTVEC_SERVER_KEY=~/.ssh/id_ed25519               # or AGENTVEC_SERVER_PASSWORD=...
export AGENTVEC_SERVER_GCC=/path/to/riscv64-...-gcc
export AGENTVEC_SERVER_QEMU=/path/to/qemu-riscv64
# optional, for on-board performance:
export AGENTVEC_BOARD_HOST=your.board        AGENTVEC_BOARD_USER=you   AGENTVEC_BOARD_KEY=...
```

## Reproduce

**Without any API key (cache replay).** The Intent Proposer is a pluggable backend whose
recovered intents are cached on disk; the Guard, lowering, build, and test stages are
deterministic. With `experiments/agentvec/_llm_cache/` present, the reduction studies
reproduce identical kernels and pass rates without contacting any model.

**Fresh model calls.** Provide an OpenAI-compatible key for the open models:
```bash
export SF_KEY=...        # never written to disk; read from the environment only
```

**Baseline head-to-head** (compile / SPR across VLEN / vector-instruction count):
```bash
cd experiments/agentvec && python -u run_baseline_mini.py
```

## Paper → code → data

Every quantitative result maps to a script and, where applicable, a cached data file.
Scripts marked **†** use a real RVV board for wall-clock numbers; without that hardware the
QEMU harness still reproduces every *correctness* (SPR / cross-VLEN) result, and the listed
`_*.json` carry the measured performance values.

| Result in the paper | Script(s) | Data file |
|---|---|---|
| L1 Migration Speedup + per-kernel dispersion | `run_l1_stats.py` † | `_l1_stats.json` |
| Real-operator validation (MS / SPR) | `run_l1.py`, `run_l1_board.py` † | `_l1board.json` |
| Cross-VLEN scalability | `run_l1.py` + `lab/difftest/runner.py` | (harness) |
| Multi-model pure-LLM vs AgentVec (+ robustness) | `run_study_reduce.py`, `run_pure_llm.py`, `run_agentvec_arm.py` | `_llm_cache/` |
| Guard ablation | `run_study_reduce.py` (w/o-Guard arm) | `_llm_cache/` |
| Obfuscation (anti-memorization) | `obfuscate.py`, `run_study_reduce.py`, `run_obf_gemm.py` | `_gemm_obf.json` |
| Multi-view (assembly / trace) | `exp_assembly_view.py`, `exp_trace_view.py` | (harness) |
| Composite (softmax / layer-norm) | `exp_composite.py`, `exp_layernorm.py` | (harness) |
| GEMM frontier comparison | `run_gemm_mm.py`, `run_gemm_stats.py` † | `_gemm_stats.json`, `_gemm_mm.json` |
| Closed-loop refinement | `run_gemv_refine.py` †, `gemm_autotune.py` †, `run_l1_lmul.py` † | `_gemv_refine.json`, `_autotune.json`, `_l1_lmul.json` |
| Generation cost (tokens) | `measure_cost.py` | `_cost.json` |
| Baseline head-to-head (intro teaser + Related Work) | `run_baseline_mini.py` | `_baseline_mini.json` |

The framework itself: AIR schema (`air.py`, `air_from_formula.py`); symbolic Guard (`guard.py`);
deterministic lowering (`lowering.py`, `blas_lower.py`); prompts + backend (`llm_backend.py`);
end-to-end driver (`pipeline.py`).

## Notes

- `_llm_cache/` holds model *outputs* (RVV code / recovered intents), not credentials.
- A reviewer with no RVV board can still reproduce all correctness (SPR / cross-VLEN) numbers
  under QEMU; the **†** scripts additionally need the board for wall-clock performance.
- Raw per-run board logs are large and omitted here; the summarized `_*.json` files and the
  differential-testing harness reproduce every reported number.

## License

MIT — see `LICENSE`.
