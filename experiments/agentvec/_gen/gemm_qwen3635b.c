#include <riscv_vector.h>

void gemm(int N, const double *A, const double *B, double *C) {
    for (int m = 0; m < N; m += 1) {
        for (int k = 0; k < N; k += 1) {
            double a_val = A[m * N + k];
            
            // Vectorize the inner product of row m of A and column k of B
            // Column k of B is stored in row-major order at indices k, k+N, k+2N...
            // We can load chunks of B's column and multiply by a_val
            
            int n = 0;
            vfloat64m1_t acc = vfmv_v_f_f64m1(0.0, 1);
            
            while (n < N) {
                int vl = __riscv_vsetvl_e64m1(N - n);
                vfloat64m1_t b_vec = vle64_v_f64m1(&B[k + n * N], vl);
                vfloat64m1_t a_vec = vmv_v_f_f64m1(a_val, vl);
                acc = vfmacc_vv_f64m1(acc, a_vec, b_vec);
                n += vl;
            }
            
            // Store the accumulated result
            // Since we are doing row-major C, C[m*N + n]
            // We need to reduce the vector accumulator to a scalar or store it back
            // However, vfmacc accumulates into the vector register. 
            // We need to extract the sum. 
            // Actually, for a single element C[m*N + n], we are summing over k.
            // The loop above iterates k. For a fixed m and n, we sum A[m,k]*B[k,n].
            // My loop structure above is: m loop, k loop, then vectorize over n?
            // No, standard GEMM is m, n, k or m, k, n.
            // Let's stick to m, n, k for vectorization over k? 
            // Or m, k, n?
            
            // Let's restart the logic for clarity.
            // C[m][n] = sum_k A[m][k] * B[k][n]
            // If we vectorize over k, we need to load A[m][k] and B[k][n].
            // A[m][k] is contiguous. B[k][n] is strided by N.
            
            // Let's do m, n, k with vectorization over k.
        }
    }
    
    // Correct implementation:
    // Iterate m and n. For each (m, n), compute dot product of row m of A and col n of B.
    // Row m of A is contiguous. Col n of B is strided.
    
    for (int m = 0; m < N; m++) {
        for (int n = 0; n < N; n++) {
            double sum = 0.0;
            int k = 0;
            while (k < N) {
                int vl = __riscv_vsetvl_e64m1(N - k);
                vfloat64m1_t a_vec = vle64_v_f64m1(&A[m * N + k], vl);
                vfloat64m1_t b_vec = vle64_v_f64m1(&B[k * N + n], vl); // Strided access in B
                // Wait, B[k*N + n] is not contiguous in memory for the vector load if we step k.
                // B is row major. B[k][n] is at B[k*N + n].
                // If we load a vector of B elements for k, k+1, ..., k+vl-1 at column n,
                // the addresses are B[k*N+n], B[(k+1)*N+n], ...
                // These are separated by N doubles. This is a strided load.
                // RVV supports strided loads: vlsseg? No, vlsseg is for segments.
                // We can use vle64_v with a stride? No, vle64_v is contiguous.
                // We need to use vlsseg64_v or similar for strided?
                // Actually, for small N, we might just use scalar or handle striding.
                // But RVV has vlsseg for strided loads.
                // vlsseg64_v_f64m1(&B[k*N + n], N, vl) loads vl elements with stride N.
                
                // However, vlsseg is for loading multiple vectors (segments).
                // We want 1 vector of vl elements with stride N.
                // We can use vle64_v if we assume N is small? No.
                
                // Alternative: Vectorize over n?
                // C[m][n] for a fixed m, vectorize over n.
                // C[m][n] = sum_k A[m][k] * B[k][n]
                // For fixed m, A[m][k] is a vector. B[k][n] for fixed k is a row.
                // So for fixed m, we can load row k of B as a vector.
                // Then multiply A[m][k] (scalar) by B_row_k (vector) and accumulate.
                
                // This is the standard approach:
                // Initialize C[m][n] to 0.
                // For k = 0 to N-1:
                //   a = A[m][k]
                //   b_vec = load B[k][n] for all n
                //   C[m][n] += a * b_vec
                
                // This requires C to be initialized to 0.
            }
            C[m * N + n] = sum;
        }
    }
    
    // Let's write the optimized version:
    // Initialize C to 0
    for (int m = 0; m < N; m++) {
        int n = 0;
        while (n < N) {
            int vl = __riscv_vsetvl_e64m1(N - n);
            vfloat64m1_t zero = vfmv_v_f_f64m1(0.0, vl);
            vse64_v_f64m1(&C[m * N + n], zero, vl);
            n += vl;
        }
    }
    
    // GEMM
    for (int m = 0; m < N; m++) {
        for (int k = 0; k < N; k++) {
            double a_val = A[m * N + k];
            
            int n = 0;
            while (n < N) {
                int vl = __riscv_vsetvl_e64m1(N - n);
                vfloat64m1_t b_vec = vle64_v_f64m1(&B[k * N + n], vl);
                vfloat64m1_t a_vec = vmv_v_f_f64m1(a_val, vl);
                vfloat64m1_t c_vec = vle64_v_f64m1(&C[m * N + n], vl);
                c_vec = vfmadd_vv_f64m1(c_vec, a_vec, b_vec);
                vse64_v_f64m1(&C[m * N + n], c_vec, vl);
                n += vl;
            }
        }
    }
}