# Benchmark results and timing

`results/rvv/` holds the downloaded measurement summaries. They are preserved as
recorded and are not overwritten by tests or new runs. Research execution entry
points are listed in [experiments/README.md](../experiments/README.md).

The new migration CLI runs correctness tests, not performance benchmarks.
The native edge suite tests mathematical and tail behavior. Neither establishes
expert-relative Migration Speedup or reproduces LLM population statistics.

For a paper performance rerun, match operator, dtype, sizes, stride, input domain,
thread/core affinity, compiler flags and timing boundary. Report warm-up and repeat
counts alongside raw rounds. Include packing and dispatch overhead when those are
part of the selected pipeline. Keep inference, compilation and target execution
time separate. Use the same equivalence policy for candidate and baseline.

Migration Speedup is `(target_expert / migrated) / (source_expert / source_input)`
when all terms are latencies. A source efficiency of one is an assumption requiring
the matching source/expert records, not a universal cross-machine conversion.

Summarize new correctness runs without changing their denominator:

```bash
python benchmarks/summarize_runs.py /external/run/migration.json --output /external/summary.csv
```
