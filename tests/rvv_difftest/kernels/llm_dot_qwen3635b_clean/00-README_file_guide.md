# Kernel package: llm_dot_qwen3635b_clean

Category: pure LLM / direct generated RVV

This folder is one runnable difftest kernel package. The fixed files have these roles:

- air.json: AgentVec intermediate intent record. It describes recovered operator semantics and constraints, such as map/reduce pattern, element expression, dtype, dependency, aliasing, and memory access. It appears only in AgentVec/AIR packages.
- rvv.c: RVV implementation under test. In AgentVec packages this is emitted by AIR plus lowering template; in pure-LLM/direct packages this is model-written RVV code; in baseline packages this is the related-method implementation.
- scalar.c: scalar reference implementation, used as the oracle for differential testing against rvv.c.
- spec.json: test metadata consumed by the harness, including kernel name, dtype, reduction flag, tolerance tier, relative tolerance, and tested sizes.

Suggested reading order:
1. spec.json: see what this package tests.
2. air.json, if present: inspect AgentVec intent before lowering.
3. rvv.c: inspect generated/direct vector code.
4. scalar.c: inspect the oracle used for correctness comparison.
