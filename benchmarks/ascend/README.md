# Ascend probe entry points

Install the package from the repository root. Generation requires no device:

```bash
agentvec ascend --output /external/ascend/generated
python benchmarks/ascend/run_comparison.py --output /external/ascend/comparison
python benchmarks/ascend/tune.py --op saxpy --budget 6 --output /external/ascend/tuning
```

Add `--execute local` on an Ascend host, or `--execute ascend` with SSH environment
variables set. The comparison uses fixed naive, default and paper-selected
schedules at matched sizes. The tuning driver enumerates admitted tuples and
selects a best candidate only after new target measurements. Its optional
`--ranking` accepts `{"ranked_ids":["b8_t8192_q2", ...]}`; IDs cannot alter a tuple.

`paper_sources/` retains 14 archived kernel/host/CMake project directories.
These are historical sources; the installed backend generates updated projects
with stricter argument and result checks. Original CANN host harnesses may round
unsupported dimensions. Use the recorded dimensions for historical reconstruction,
or the new CLI for explicit rejection of unsupported workloads.

`validated_sources/` contains the ten projects executed for release 0.3.1, including
their standalone runners. Copy a project into an external working directory before
rebuilding so source folders remain free of generated binaries:

```bash
python /external/saxpy/runner.py --root /external/saxpy --setup /path/to/setup.sh
```

The matching current-device records and logs are in `results/validation/ascend/0.3.1/`.

`source_contracts/` contains 12 cuBLAS signature/semantics descriptions. It is not
a complete CUDA benchmark suite. Archived direct-generation candidates and logs
are under `results/ascend310p/paper/llm_baseline/`; these include intentionally
failing programs and are not automatically built by CI.

See [the Ascend guide](../../docs/ascend.md) for SDK requirements, numerical and
timing boundaries, and the distinction between kernel acceptance and safe vetoes.
