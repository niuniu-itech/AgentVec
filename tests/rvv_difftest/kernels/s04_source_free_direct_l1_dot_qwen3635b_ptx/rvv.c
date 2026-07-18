#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
#define OUT_FACTOR 1
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    int vl = __riscv_vsetvl_e32m1(n);
    float acc = 0.0f;
    vfloat32m1_t va, vb;
    vfloat32m1_t vsum = vmv_v_f32_m1(0.0f);
    
    while (vl > 0) {
        va = vle32_v_f32_m1(a, vl);
        vb = vle32_v_f32_m1(b, vl);
        vsum = vfmacc_vf_f32_m1(vsum, va, vb);
        a += vl;
        b += vl;
        n -= vl;
        vl = __riscv_vsetvl_e32m1(n);
    }
    
    // Reduce vector sum to scalar
    float *tmp = &acc;
    vse32_v_f32_m1(tmp, vsum, __riscv_vsetvl_e32m1(1));
    // Note: vfmacc accumulates into vsum, but we need a scalar result.
    // The evidence shows a reduction (red.add.f32). 
    // vfloat32m1_t reduction requires vlseg or specific reduction intrinsics.
    // However, with m1, we can just sum the elements.
    // Let's use a simpler approach for m1: accumulate in a scalar loop or use vreduce.
    // Since vreduce is not always available in basic RVV 1.0 without specific extensions,
    // and the evidence implies a single accumulator, let's do a scalar reduction of the vector.
    
    // Actually, vfmacc_vf accumulates a scalar into a vector. The evidence does:
    // mul.f32 t,a[i],b[i]; red.add.f32 acc,t
    // This is a dot product.
    
    // Correct approach for dot product with RVV:
    // 1. Load vectors
    // 2. Multiply vectors
    // 3. Reduce the resulting vector to a scalar
    
    // Let's restart the implementation properly.
}

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    int vl = __riscv_vsetvl_e32m1(n);
    float acc = 0.0f;
    
    while (vl > 0) {
        vfloat32m1_t va = vle32_v_f32_m1(a, vl);
        vfloat32m1_t vb = vle32_v_f32_m1(b, vl);
        vfloat32m1_t vmul = vmul_vv_f32_m1(va, vb, vl);
        
        // Reduce vmul to scalar acc
        // Since we are using m1, we can store to a temporary array and sum, 
        // or use vreduce_add_vf if available. 
        // Standard RVV 1.0 includes vreduce_add_vf for m1.
        acc += vreduce_add_vf_f32_m1(vmul);
        
        a += vl;
        b += vl;
        n -= vl;
        vl = __riscv_vsetvl_e32m1(n);
    }
    
    *out = acc;
}
#include "harness.h"
