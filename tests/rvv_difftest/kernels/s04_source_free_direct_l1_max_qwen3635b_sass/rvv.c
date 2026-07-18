#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
#define OUT_FACTOR 1
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    int vl = __riscv_vsetvl_e32m1(n);
    float acc = 0.0f;
    
    while (vl > 0) {
        vfloat32m1_t va = vle32_v_f32m1(a, vl);
        vfloat32m1_t vb = vle32_v_f32m1(b, vl);
        vfloat32m1_t vc = vle32_v_f32m1(c, vl);
        
        // Perform element-wise operations and accumulate max
        // Note: The evidence suggests a reduction (red.max.f32) but RVV 1.0 
        // requires explicit reduction intrinsics. We'll compute a local max 
        // per vector chunk and accumulate to a scalar accumulator.
        vfloat32m1_t sum = vadd_vv_f32m1(va, vb);
        sum = vadd_vv_f32m1(sum, vc);
        
        // Compute max of the current vector chunk
        float chunk_max = 0.0f;
        for (int i = 0; i < vl; i++) {
            if (sum[i] > chunk_max) {
                chunk_max = sum[i];
            }
        }
        
        if (chunk_max > acc) {
            acc = chunk_max;
        }
        
        a += vl;
        b += vl;
        c += vl;
        vl = __riscv_vsetvl_e32m1(n - (a - (const float *)0)); // Simplified for strip mining
        // Correct strip mining loop condition:
        // We need to track remaining elements. Let's rewrite loop properly.
    }
    
    // Since the evidence is ambiguous about the exact operation (max of sums vs sum of maxes),
    // and RVV doesn't have a direct "red.max" that accumulates across strips into a single scalar
    // without a loop, we assume the intent is to find the maximum value in the result of a+b+c.
    // However, the evidence shows STG.E out[0], acc, implying a single output.
    // Let's implement a standard strip-mined loop that computes the max of (a[i]+b[i]+c[i]).
    
    // Re-implementing cleanly:
    int remaining = n;
    const float *pa = a;
    const float *pb = b;
    const float *pc = c;
    float global_max = -1e30f; // Initialize to a very small number
    
    while (remaining > 0) {
        vl = __riscv_vsetvl_e32m1(remaining);
        vfloat32m1_t va = vle32_v_f32m1(pa, vl);
        vfloat32m1_t vb = vle32_v_f32m1(pb, vl);
        vfloat32m1_t vc = vle32_v_f32m1(pc, vl);
        
        vfloat32m1_t res = vadd_vv_f32m1(va, vb);
        res = vadd_vv_f32m1(res, vc);
        
        // Find max in this vector chunk
        float local_max = -1e30f;
        for (int i = 0; i < vl; i++) {
            if (res[i] > local_max) {
                local_max = res[i];
            }
        }
        
        if (local_max > global_max) {
            global_max = local_max;
        }
        
        pa += vl;
        pb += vl;
        pc += vl;
        remaining -= vl;
    }
    
    out[0] = global_max;
}
#include "harness.h"
