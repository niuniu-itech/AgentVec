# Changelog

## 0.3.1

- Set an explicit Release build type in the Ascend runner and exported CMake
  projects. CANN 8.0.RC3 requires this value when merging device objects.
- Retain the first failed build and subsequent device acceptance records outside
  the source checkout; publish sanitized validation summaries with source hashes.
- Validate all ten emitted AscendC templates on Ascend310P1 / CANN 8.0.RC3 across
  30 shapes and 90 rounds. Keep both triangular-solve vetoes outside the kernel
  acceptance count. Include the executed projects and individual build/run logs.

## 0.3.0

- Add typed scalar/vector AIR DAGs, dependency and extent checks, topological RVV
  and scalar lowering, and single-use map fusion.
- Add independent Softmax, LayerNorm, RMSNorm and MinMax references, plus a
  branch/join benchmark with two outputs and runtime storage checks.
- Add an AscendC adapter for ten registered f32 BLAS contracts and two explicit
  dependence vetoes, with schedule bounds, generated host references, CANN build
  projects and a source-bound remote validation runner.
- Restore the archived Ascend paper projects, measurements and direct-generation
  outcomes with a source manifest. Preserve historical measurements separately
  from current code-validation evidence.
- Support environment-configured SiliconFlow intent proposals in both new paths.
  The compiler checks the actual proposed intent against caller-owned rules.
- Add Ascend schedule comparison and measured selection commands, public examples,
  validation summaries, packaging rules and CI coverage.

At 0.3.0, the DAG path had native RVV correctness evidence and the Ascend generator
had export and CPU test coverage. Ascend device acceptance was still pending at
that checkpoint. This does not claim a new full-paper performance campaign.
