# Release validation

`0.3.1.json` identifies the current source files and validation environment.
`0.3.0.json` retains the earlier checkpoint before authenticated Ascend access.
`dag/` contains native RVV reports whose generated code is unchanged in 0.3.1.
`ascend/0.3.1/` contains fresh CANN build logs and per-round target results.

Ten emitted AscendC templates passed independent host-reference checks on
Ascend310P1 with CANN 8.0.RC3: three shapes per operator, three rounds per shape,
and 30 launches per timed batch. All 90 rounds passed. `strsv` and `strsm` veto
before lowering and are excluded from those counts. The executed projects are in
`benchmarks/ascend/validated_sources/`; hashes bind the records to these sources.
The runtime's SoC query confirmed Ascend310P1, despite `npu-smi` returning a DCMI
initialization error for this login.

The first native build exposed an empty CMake build type. Setting Release fixed
CANN's device-object merge step; both the runner and exported CMake now set it.
The CPU suite has 158 passing tests, including compiled scalar references.

The retained RVV checks cover:

- Four composite graphs, each staged and fused: 2,112 input cases and 610,464
  numerical/storage checks passed on native RVV.
- A branch/join graph with scalar broadcast and two outputs, staged and fused:
  132 cases and 38,694 checks passed.
- A deliberately incorrect Softmax proposal compiled, then failed all 76,308
  numerical checks. This verifies that code generation cannot establish VERIFIED.

Ascend timings use `host_batch_submit_sync`. CPU reduction finalization and its
transfer are outside the measured kernel batch. Historical paper evidence remains
in `results/ascend310p/paper/` and is not replaced by these new measurements.

Only one native RVV machine was used. The checks do not establish a multi-VLEN
hardware result, a new LLM acceptance rate, or reproduction of all paper timings.
Detailed local logs and build products remain outside the source release.
