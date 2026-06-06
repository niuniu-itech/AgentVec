#include <riscv_vector.h>

void kern(int n, double a, const double *x, double *y) {
    // Process 8 doubles at a time using vfloat64m8_t (8-lane vector)
    // Strip-mining with __riscv_vsetvl_e64m8 for vector-length agnosticity
    
    int i = 0;
    for (; i < n; ) {
        // Determine the number of elements to process in this chunk
        // vfloat64m8_t has 8 elements per vector
        size_t vl = __riscv_vsetvl_e64m8(n - i);
        
        if (vl == 0) break;
        
        // Load vector a (broadcast scalar a to vector)
        vfloat64m8_t va = __riscv_vfmv_v_f_f64m8(a, vl);
        
        // Load vector x
        vfloat64m8_t vx = __riscv_vle_v_f64m8(x + i, vl);
        
        // Load vector y
        vfloat64m8_t vy = __riscv_vle_v_f64m8(y + i, vl);
        
        // Fused multiply-add: vy = va * vx + vy
        vy = __riscv_vfmadd_vv_f64m8(vy, va, vx, vl);
        
        // Store result back to y
        __riscv_vse_v_f64m8(y + i, vy, vl);
        
        i += (int)vl;
    }
}