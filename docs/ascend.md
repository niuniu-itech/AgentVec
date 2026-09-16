# AscendC release

The release contains a registered Ascend310P1 vector backend, complete project
generation, host references, a CANN build/test runner, and the source/measurement
materials recovered from the paper's Ascend probe.

## Layout

| Path | Contents |
| --- | --- |
| `src/agentvec/ascend/` | AIR adapter, source registry, contract checks, schedule bounds, AscendC/host/CMake generation, runner |
| `benchmarks/ascend/paper_sources/` | Archived device/host/CMake projects for 14 measured or exploratory configurations |
| `benchmarks/ascend/validated_sources/` | Ten projects executed for current Ascend device acceptance |
| `benchmarks/ascend/source_contracts/` | Registered cuBLAS signatures and source-semantics descriptions |
| `results/ascend310p/paper/` | Preserved metrics, AIR/validation/schedule records and direct-generation outcomes |
| `results/validation/ascend/0.3.1/` | Current build logs and 90 target-validation rounds |

`source_contracts` files are semantic descriptions, not complete CUDA benchmark
programs. They are not advertised as a general CUDA parser or independent static
dependency analyzer. Core RVV graphs and the registered Ascend BLAS adapter have
different supported scopes.

## Prerequisites

The paper's target is Ascend310P1 with CANN 8.0.RC3. Install a compatible CANN
toolkit and driver, Python 3.10+, CMake 3.16+, and the host C++ toolchain required
by that toolkit. The SDK, device driver and compiler are not redistributed.
Generation and unit tests work without CANN; compilation/execution require it.
The exported standalone runner also works with Python 3.9 on the target; installing
the full package and generating projects require Python 3.10 or newer.

The build script locates `ascendc.cmake` under the toolkit's `compiler`, `tools`,
or `aarch64-linux` tree. Set `ASCEND_CANN_PACKAGE_PATH` to override the toolkit
root. If the installation requires additional compiler include/library variables,
put them in a private setup script and pass `--setup /path/to/setup.sh`.
`--cmake` accepts an installed CMake executable path. No user-specific compiler
shim, home directory or password is embedded in the release.

Generated projects and the runner default to `Release`. CANN 8.0.RC3's device
object merge script fails if the build type is empty. On a host with limited root
disk space, set `AGENTVEC_REMOTE_ROOT` to a scratch filesystem with enough room for
the build trees and keep the setup script outside the public repository.

## Generate, build and test

```bash
agentvec ascend --output /external/ascend-generated
agentvec ascend --ops saxpy,sdot,sgemv,sgemm --profile paper \
  --rounds 3 --iterations 30 --execute local --output /external/ascend-paper
agentvec ascend --ops saxpy --backend siliconflow --model "$SILICONFLOW_MODEL" \
  --execute ascend --output /external/ascend-model
```

For SSH execution, set `AGENTVEC_ASCEND_HOST`, `_USER`, `_KEY` (or `_PASSWORD`),
and optionally `_PORT`, then replace `--execute local` with `--execute ascend`.
Each remote run uses a unique directory. All reports and logs are downloaded to
the specified local output directory. Example variables are in `.env.example`.

Model proposals may supply only algorithm slots; source contracts and schedules
remain caller-owned. A supplied `--proposal intent.json` follows the same checks.
Set `SILICONFLOW_API_KEY` in the environment; no key is included in the package.

The commands in [benchmarks/ascend](../benchmarks/ascend/README.md) compare matched
workloads under naive, default and archived schedule choices. `tune.py` enumerates
admitted schedules, accepts optional identifier-only ranking, and chooses a best
candidate only from successful target measurements.

The registry has twelve input contracts. Ten have code-generation templates:
`saxpy`, `scopy`, `sscal`, `sdot`, `sasum`, `snrm2`, `sgemv`, `sgemm`, `sgemm_nt`
and `ssyrk`. `strsv` and `strsm` are explicit dependence vetoes and emit no kernel.
A veto is not a successful numerical test and is not included as an emitted
kernel in a correctness denominator.

Each accepted project has a device source, independent host reference,
`CMakeLists.txt`, `air.json`, `project.json` and a standalone runner. The smoke
profile checks three sizes; the paper profile includes L1 `2^24`, GEMV `4096²`
and GEMM `512³`. Each case retains individual rounds and their median latency.

```bash
python /external/ascend-generated/saxpy/runner.py \
  --root /external/ascend-generated/saxpy
```

Generation establishes CHECKED only. VERIFIED requires a successful build,
successful exit codes, all requested cases/rounds passing, exact dimension
agreement, finite error/timing statistics and matching executed source hashes.
Unsupported shapes are rejected rather than rounded to a different workload.
The templates are f32 and use vector instructions; GEMM is not a Cube implementation.

## Timing and original evidence

The reported boundary is `host_batch_submit_sync`: resident device buffers,
one warmup, a batch of kernel launches, and one stream synchronization. Allocation,
host/device transfer, compilation and host oracle computation are outside the
timed region. Reduction kernels write per-block partials; CPU finalization and
its transfer are outside the timed kernel batch. These results must not be
presented as complete device-resident reductions or as CUDA-normalized efficiency.

Matrix timing repeats the same update operation on resident output storage,
matching the archived experiment; numerical validation uses the first execution
against the original output. New timing records state their profile and boundary.

The archived paper materials are not rewritten when the new generator changes.
Their old `VERIFIED` fields may denote static admission; use the accompanying
device logs for historical correctness. The stable metric suite measures eight
operators, while the adapter contains ten emit-capable contracts. The fair direct
generation aggregate has 16 tasks and five PASS outcomes. None of these archived
counts is a new-device acceptance claim for this revision.

Native validation and current-source build status are recorded separately in the
release verification summary. No Ascend 350 or vendor-library performance claim
is made by this package.
