#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 0
#define A_IS_MATRIX 1
#define B_IS_MATRIX 0
#define C_IS_MATRIX 0
#define OUT_IS_MATRIX 0
#define SYMMETRIC_A 1
#define LOWER_TRIANGULAR_A 0
#define BANDED_A_RADIUS -1
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)c;
    
    // Precompute the lower-triangular mask pattern for each row i.
    // For row i, we need to access a[i*n + j] for j <= i, and a[j*n + i] for j > i.
    // Since a stores the lower triangle in row-major order, a[i*n + j] is valid for j <= i.
    // For j > i, the value is a[j*n + i] (which is the same as a[i + j*n] in memory layout of lower triangle stored row-major? 
    // Wait: "a stores the lower triangle of a symmetric n by n matrix in row-major slots"
    // This means a is an array of size n*n? Or just the lower triangle elements?
    // The source code accesses a[i*n+j] when j<=i and a[j*n+i] when j>i.
    // This implies a is a full n*n array where the lower triangle is stored in its natural positions,
    // and the upper triangle values are retrieved from the lower triangle positions (symmetric).
    // So a is treated as an n*n matrix in row-major order.
    
    // We will process one row of the output at a time.
    // For each row i, we compute out[i] = sum_{j=0}^{n-1} v_{i,j} * b[j]
    // where v_{i,j} = a[i*n + j] if j <= i, else a[j*n + i].
    
    // To vectorize efficiently, we can process multiple rows or use strip mining on the inner loop.
    // Given the dependency on i for the mask, it's easier to process one row at a time with vectorization over j.
    
    for (int i = 0; i < n; i++) {
        float s = 0.0f;
        
        // We need to compute sum_{j=0}^{n-1} v_{i,j} * b[j]
        // Let's use strip mining for the inner loop over j.
        
        int vl;
        vfloat32m1_t b_v, v_v, acc_v;
        vfloat32m1_t acc_sum = vfmv_v_f_f32m1(0.0f, 1); // Not quite right, need to accumulate
        
        // Actually, let's accumulate into a vector accumulator and then reduce.
        // But since we're doing one row, we can just accumulate into a scalar if n is small,
        // but we want to vectorize.
        
        // Strategy: Process chunks of j.
        // For each chunk, we need to load b[j], load v_{i,j}, multiply, and add.
        
        // Load b vector
        // We'll use vsetvl to get the active vector length.
        
        // Initialize accumulator for this row
        float row_sum = 0.0f;
        
        // Process in strips
        int j = 0;
        while (j < n) {
            vl = __riscv_vsetvl_e32m1(n - j);
            
            // Load b[j:j+vl]
            vfloat32m1_t b_chunk = vle32_v_f32m1(b + j, vl);
            
            // Compute v_{i,j} for j in [j, j+vl)
            // For each k in [0, vl), index = j + k
            // If (j+k) <= i, v = a[i*n + j + k]
            // Else v = a[(j+k)*n + i]
            
            // We need to generate the mask for j+k <= i
            // Create a mask for the current chunk
            // mask[k] = (j+k) <= i
            
            // Generate indices for the current chunk
            // We can use vrgather or compute on the fly.
            
            // Let's create a vector of indices for the current chunk
            // vindex = j, j+1, ..., j+vl-1
            vint32m1_t idx = vadd_vx_i32m1(vfredosum_vs_f32m1_f32m1_x2, 0, vl); // This doesn't work directly
            
            // Better approach: use vadd_vx to create indices
            // First, create a zero vector and add index
            // Actually, we can use vrgather_vv or just compute the values.
            
            // Let's try a different approach: precompute the mask and gather.
            
            // Create a vector of offsets 0..vl-1
            vint32m1_t offsets = vadd_vx_i32m1(vl, 0); // This is not standard. Let's use a loop to create it? No, we need vector ops.
            
            // Standard way to create index vector:
            // vint32m1_t idx = vadd_vx_i32m1(vl, 0); // vl is int, not vector.
            
            // Let's use vadd_vx with a zero vector
            vint32m1_t zero_idx = vmv_v_x_i32m1(0, vl);
            vint32m1_t idx = vadd_vx_i32m1(zero_idx, vl, vl); // No, vadd_vx takes scalar
            
            // Correct: vadd_vx_i32m1(vd, vs2, vl) adds scalar vs2 to each element of vs1
            // So vadd_vx_i32m1(zero_idx, vl, vl) is wrong syntax.
            // vadd_vx_i32m1(vd, vs2, vl) -> vd = vs1 + vs2
            // We want idx[k] = k for k in 0..vl-1
            // So we can start with zero and add... but we need to add 0,1,2...
            
            // Alternative: use vrgather_vv with a precomputed index vector? No.
            
            // Let's use a simpler method: since we're doing strip mining, we can just handle the logic carefully.
            
            // Create index vector: start with 0, then add 1, 2, ... vl-1
            // We can do this by: idx = vadd_vx_i32m1(vl, 0) is not right.
            
            // Actually, in RVV, we can create an index vector using:
            // vint32m1_t idx = vadd_vx_i32m1(vl, 0); // This is incorrect.
            
            // Let's use the fact that we can add a scalar to a vector.
            // Start with a zero vector, then we need to add 0,1,2,... which is not a scalar.
            
            // Better: use vadd_vv with a vector of 0,1,2,... but we don't have that.
            
            // Alternative approach: compute the values directly without explicit index vector.
            
            // For each element in the chunk, we need to know if (j+k) <= i.
            // We can create a mask:
            // mask[k] = (j+k) <= i  <=>  k <= i - j
            
            int limit = i - j;
            vbool8_t mask = vmv_v_x_v_b8(vl, 0); // Not quite.
            
            // Create mask: for k in 0..vl-1, mask[k] = (j+k) <= i
            // This is equivalent to k <= i - j
            // We can create a vector of k values and compare.
            
            // Let's create a vector of k values (0, 1, ..., vl-1)
            // We can do this by starting with 0 and adding 1 repeatedly? No.
            
            // Standard trick: use vadd_vx with a sequence? No.
            
            // Let's use vrgather_vv with a pointer to an index array? No, we want vector agnostic.
            
            // Actually, we can create the index vector by:
            // vint32m1_t idx = vadd_vx_i32m1(vl, 0); // This is wrong.
            
            // Let's try: 
            // vint32m1_t idx = vmv_v_x_i32m1(0, vl);
            // Then we need to add 0,1,2,... to it. This is not a scalar addition.
            
            // Alternative: use a loop for small vl? No, we want vectorization.
            
            // Let's use the following approach:
            // We'll create the mask by comparing a vector of indices with i-j.
            // To create the index vector, we can use vadd_vx in a clever way? No.
            
            // Actually, RVV has vadd_vx which adds a scalar. We need to add 0,1,2,...
            
            // Let's use a different strategy: process the row in two parts:
            // Part 1: j from 0 to min(i, n-1) where j <= i, so v = a[i*n + j]
            // Part 2: j from i+1 to n-1 where j > i, so v = a[j*n + i]
            
            // This splits the inner loop into two vectorizable parts.
            
            // Part 1: j in [0, min(i, n-1)]
            int end1 = (i < n) ? i : n - 1;
            if (end1 >= 0) {
                int j1 = 0;
                while (j1 <= end1) {
                    vl = __riscv_vsetvl_e32m1(end1 - j1 + 1);
                    vfloat32m1_t b_chunk = vle32_v_f32m1(b + j1, vl);
                    
                    // Load a[i*n + j1 : j1+vl]
                    vfloat32m1_t a_chunk = vle32_v_f32m1(a + i * n + j1, vl);
                    
                    // Multiply and accumulate
                    vfloat32m1_t prod = vmul_vv_f32m1(a_chunk, b_chunk, vl);
                    
                    // Reduce and add to row_sum
                    float chunk_sum = __riscv_vfredosums_vs_f32m1_f32_m1(prod, row_sum, vl);
                    row_sum = chunk_sum;
                    
                    j1 += vl;
                }
            }
            
            // Part 2: j in [i+1, n-1]
            int start2 = i + 1;
            if (start2 < n) {
                int j2 = start2;
                while (j2 < n) {
                    vl = __riscv_vsetvl_e32m1(n - j2);
                    
                    // Load b[j2 : j2+vl]
                    vfloat32m1_t b_chunk = vle32_v_f32m1(b + j2, vl);
                    
                    // Load a[j2*n + i : (j2+vl-1)*n + i]
                    // These are not contiguous in memory! a[j*n + i] for j = j2, j2+1, ...
                    // The stride is n.
                    
                    // We need to gather these values.
                    // vfloat32m1_t a_chunk = vle32_v_f32m1(a + j2 * n + i, vl); // This is wrong, they are not contiguous.
                    
                    // We need to use vrgather or load with stride.
                    // RVV has vle32_v with stride? No, vle32_v loads contiguous.
                    // We can use vrgather_vv to gather.
                    
                    // Create index vector for the positions: j2*n + i, (j2+1)*n + i, ...
                    // index[k] = (j2 + k) * n + i = j2*n + i + k*n
                    
                    // Create a vector of k values: 0, 1, ..., vl-1
                    vint32m1_t k_vec = vmv_v_x_i32m1(0, vl);
                    // Add k to k_vec? No, we need 0,1,2,...
                    // k_vec already has 0, but we need to add 0,1,2,... to it? No, k_vec should be 0,1,2,...
                    
                    // Let's create k_vec properly:
                    // k_vec[k] = k for k in 0..vl-1
                    // We can do this by: k_vec = vadd_vx_i32m1(vl, 0) is wrong.
                    
                    // Actually, we can create it by starting with 0 and adding 1,2,... but that's not scalar.
                    
                    // Alternative: use vadd_vv with a vector of 0,1,2,... but we don't have it.
                    
                    // Let's use a different approach: since the stride is n, we can load with stride if n is small?
                    // No, we want vector agnostic.
                    
                    // We'll use vrgather_vv:
                    // First, create the base indices: j2*n + i, (j2+1)*n + i, ...
                    // base[k] = (j2 + k) * n + i
                    
                    // Create a vector of (j2 + k) for k in 0..vl-1
                    // j2_vec[k] = j2 + k
                    vint32m1_t j2_vec = vadd_vx_i32m1(vl, j2); // j2_vec[k] = 0 + j2 = j2? No.
                    // vadd_vx_i32m1(vd, vs2, vl) -> vd = vs1 + vs2
                    // We want j2_vec[k] = j2 + k
                    // So we need vs1 to be 0
#include "harness.h"
