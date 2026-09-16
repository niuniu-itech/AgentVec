# AIR graphs

The graph path accepts arbitrary acyclic compositions of the supported f32 maps,
scalar maps, broadcasts and sum/min/max contractions. The four bundled composites
are examples, not dispatch names for hand-written composite kernels. `candidate.c`
is generated from the actual proposed nodes and edges.

## Contract and proposal

Supply storage and numerical rules separately from the model response:

```json
{"inputs":{"x":"n","gain":"scalar"},"outputs":{"y":"n","total":"scalar"},
 "dtype":"f32","alias":"disjoint","domain":"finite","min_n":1,"max_n":16777216,
 "rtol":0.0002,"atol":0.00002}
```

```json
{"pattern":"dag","nodes":[
  {"name":"scaled","pattern":"map","formula":"x*gain"},
  {"name":"y","pattern":"map","formula":"scaled+1"},
  {"name":"total","pattern":"contract","formula":"y","reduce_op":"sum"}
]}
```

The expression language includes named reads, `n`, finite constants, `+ - * /`,
`exp`, `sqrt`, `abs`, `min` and `max`. Expressions are parsed completely; Python
attributes, indexing, arbitrary calls and statements are rejected. `compose` is
accepted as a graph-container spelling. A vector expression has extent `n`;
scalar values broadcast when combined with vectors. Reductions produce scalars.

Names have one definition. A node cannot overwrite an input or another node, so
cross-stage write-after-read hazards cannot enter the admitted SSA representation.
Undefined references, cycles, unused nodes and output-extent mismatches are vetoed.
The checker derives topological order from dependencies rather than trusting the
proposal's listing order. Serialized status fields cannot authorize lowering.

## Generate and validate

```bash
agentvec dag --example softmax --fuse-maps --output /external/dag-softmax
agentvec dag --contract contract.json --proposal graph.json \
  --output /external/custom-dag
agentvec dag --example layernorm --fuse-maps --execute board \
  --output /external/layernorm-native
agentvec dag --example softmax --backend siliconflow --model "$SILICONFLOW_MODEL" \
  --execute board --output /external/softmax-model
```

`--example` supplies independent input/output rules and a separately written
double-precision reference. With `--proposal` or `--backend siliconflow`, the
supplied graph replaces the example graph. It is never discarded in favor of a
known-correct implementation. A shape-correct but semantically wrong proposal can
be CHECKED and subsequently fail the independent reference.

Custom graphs can be exported through the same compiler. They require a caller's
independent reference before claiming semantic validation; the command does not
manufacture an oracle from the proposed graph. The scalar target (`--target scalar`)
is a lowering/debugging aid, not an independent intent oracle.

The generated ABI is:

```c
int agentvec_dag(const float *const *inputs, float *const *outputs, size_t n);
```

Pointer order follows the contract's input/output key order. Buffers must hold
the declared number of floats. Return values are 0 for execution, 1 for invalid
arguments, 2 for overlapping output storage, and 3 for scratch allocation failure.
The caller owns buffers; temporary arrays are allocated and released internally.
Read-only inputs may share storage; output ranges must be disjoint from all inputs
and other outputs. The function does not infer allocation sizes from pointers.

## Lowering and fusion

RVV maps use runtime `vsetvl`. Reductions retain inactive accumulator lanes with
tail-undisturbed intrinsics. Scalar nodes and broadcasts preserve graph dependency
order. `exp` uses per-lane `expf`, since there is no native RVV exponential
instruction. Arithmetic is compiled without fast-math or implicit FMA contraction.

`--fuse-maps` substitutes a vector map into its sole consumer only when that value
is read once and is not an exposed graph output. Branched producers stay materialized.
This can fuse map-to-map and map-to-reduction edges without changing a shared value's
lifetime. The lowering record lists every fused edge and the remaining stage order.

The LayerNorm example centers values around a finite minimum before calculating
their mean and variance. This avoids large-offset accumulation error without
loosening the validation tolerance. Each example is checked on 22 lengths, three
seeds and four input regimes (264 cases), including constant vectors, small values,
large offsets, output canaries and rejected zero-length/alias calls.

The present graph type has scalar and common-`n` vector extents. General tensor
shapes, scans, indexed accesses and arbitrary matrix contractions are outside this
backend. Matrix and BLAS study adapters remain separate. Native correctness checks
are not a rerun of the paper's historical model acceptance percentages.
