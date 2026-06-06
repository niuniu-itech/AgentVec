#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 0
#include <riscv_vector.h>
code block
     - Must start with includes and exact signature
     - No explanation

2.  **Identify Key RVV 1.0 Concepts:**
   - `vsetvl` is used to get the actual vector length for the current execution environment.
   - `__riscv_vsetvl_e32m1(n, 1)` returns the number of elements per vector group for `e32m1` (32-bit float, LMUL=1).
   - Strip-mining loop: `while (vl > 0) { ... vl = __riscv_vsetvl_e32m1(remaining, 1); ... }`
   - Intrinsics needed:
     - `vle32_v_f32m1` or `vle32_v_f32m1`? Wait, RVV 1.0 uses `vle32_v_f32m1` for loading. Actually, the standard naming is `vle32_v_f32m1` or `vle32_v_f32m1`? Let's check RVV 1.0 spec: `vle32_v_f32m1` is correct for LMUL=1. Actually, it's `vle32_v_f32m1` or `vle32_v_f32m1`? The standard is `vle32_v_f32m1` (load element 32-bit, float, m1). Wait, RVV 1.0 uses `vle32_v_f32m1`? Let's verify: `vle32_v_f32m1` is correct. Actually, it's `vle32_v_f32m1` or `vle32_v_f32m1`? The format is `vle<width>_v_<type><lmul>`. So `vle32_v_f32m1`.
     - `vfmul.vf_f32m1` for `2.0f * a[i]` (scalar-vector multiply)
     - `vadd.vv_f32m1` for adding `b[i]`
     - `vse32_v_f32m1` for storing
   - Wait, `vfmul.vf` takes a scalar and a vector. The scalar is the second operand in RVV 1.0? Actually, RVV 1.0 intrinsics: `vfmul_vf_f32m1(vfloat32m1_t op1, float op2, size_t vl)`? Let's check the exact signature. In RVV 1.0, it's `vfmul_vf_f32m1(vfloat32m1_t op1, float op2, size_t vl)`. Wait, the order might be `vfmul_vf_f32m1(vfloat32m1_t op1, float op2, size_t vl)`. Actually, the standard is `vfmul_vf_f32m1(vfloat32m1_t op1, float op2, size_t vl)`. Let's verify with RISC-V V spec: `vfmul.vf vd, vs2, rs1` -> `vfmul_vf_f32m1(vfloat32m1_t op1, float op2, size_t vl)`. Yes.
   - Alternatively, I can load `a`, multiply by 2.0f, load `b`, add, store.
   - Let's check intrinsic names carefully:
     - Load: `vle32_v_f32m1`
     - Store: `vse32_v_f32m1`
     - Mul: `vfmul_vf_f32m1`
     - Add: `vadd_vv_f32m1`
   - Wait, RVV 1.0 intrinsics naming convention: `v<op>_<type><lmul>_<type><lmul>_<type><lmul>`? Actually, it's `v<op>_<type><lmul>_<type><lmul>_<type><lmul>` for 3-operand, but for 2-operand it's `v<op>_<type><lmul>_<type><lmul>`. For scalar-vector: `vfmul_vf_f32m1`.
   - Let's verify with a known reference or just use standard names. I'll stick to `vle32_v_f32m1`, `vfmul_vf_f32m1`, `vadd_vv_f32m1`, `vse32_v_f32m1`.
   - Actually, RVV 1.0 uses `vle32_v_f32m1` and `vse32_v_f32m1`.
   - Wait, the type for LMUL=1 is `vfloat32m1_t`.
   - Let's construct the loop:
#include "harness.h"
