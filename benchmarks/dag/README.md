# Composite DAG validation

The four examples in `agentvec dag` use independent scalar oracles for Softmax,
LayerNorm, RMSNorm and MinMax normalization. Run each with and without
`--fuse-maps`; use `--execute board` after configuring the RVV connection.

`validate_branch.py` additionally exercises an out-of-order branch/join graph,
scalar broadcasting, two outputs, reduction tails and rejected output aliasing:

```bash
python benchmarks/dag/validate_branch.py --execute board --output /tmp/branch-staged
python benchmarks/dag/validate_branch.py --execute board --fuse-maps --output /tmp/branch-fused
```

Use `--execute local --target scalar` to test the same graph with a local C11
compiler. These commands measure correctness, not a paper performance speedup.
