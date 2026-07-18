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
    // Operator inference:
    // The evidence describes a Level-2 operator (matrix-vector multiply variant).
    // "select lower-triangle A[max(i,j),min(i,j)]" implies accessing a symmetric matrix A
    // stored in lower-triangular format (or accessing it such that we always use the lower triangle).
    // However, standard Level-2 BLAS is y = alpha*op(A)*x + beta*y.
    // The evidence says: acc=0; for col j: select lower-triangle A[max(i,j),min(i,j)]; fma acc,A,x[j]; st out[i],acc.
    // This looks like computing a row of the result of a matrix-vector product where the matrix is symmetric.
    // Specifically, if A is symmetric, A[i][j] = A[j][i].
    // The access pattern A[max(i,j), min(i,j)] suggests we are treating the matrix as if it were stored in a way
    // that we always index the lower triangle. Let's assume A is a dense matrix but we only use the lower triangle
    // for the computation, or A is stored in a packed format.
    // Given "source-free" and "infer operator", and the specific indexing, this is likely:
    // out[i] = sum_{j=0}^{n-1} A_sym[i][j] * b[j]
    // where A_sym[i][j] = A[max(i,j)][min(i,j)].
    // This is equivalent to: out[i] = sum_{j=0}^{i} A[i][j]*b[j] + sum_{j=i+1}^{n-1} A[j][i]*b[j].
    // This is a symmetric matrix-vector multiplication.

    int vl = vsetvlmax_e32_m1();
    int vlre = vsetvlred_e32_m1(); // Not standard, use vsetvl

    for (int i = 0; i < n; i++) {
        float acc = 0.0f;
        
        // We need to compute the dot product of the i-th row of the symmetric matrix with vector b.
        // Row i of symmetric matrix A:
        // For j <= i: element is A[i][j]
        // For j > i: element is A[j][i]
        
        // Vector length agnostic strip mining for the inner loop over j
        int j = 0;
        int vn;
        
        // Process in chunks of VL
        while (j < n) {
            vn = vsetvl_e32_m1(n - j);
            vfloat32m1_t va, vb, vres;
            
            // Load b[j...j+vn-1]
            vb = vle32_v_f32_m1(&b[j], vn);
            
            // We need to load A elements.
            // Case 1: j + vn <= i. All j in this chunk are <= i.
            //   We need A[i][j...j+vn-1].
            // Case 2: j > i. All j in this chunk are > i.
            //   We need A[j...j+vn-1][i].
            // Case 3: Chunk straddles i.
            
            // Since we are doing scalar accumulation per row 'i', and the access pattern for A depends on j relative to i,
            // and j changes, this is not a simple vector load of a contiguous row unless we handle the straddle.
            // However, RVV allows masked loads.
            
            // Let's create a mask for j <= i and j > i within the current chunk.
            // But wait, the index 'i' is fixed for the outer loop.
            // The indices into A are:
            // If j <= i: row i, col j.
            // If j > i: row j, col i.
            
            // This access pattern is not contiguous in memory for A in the general case for a chunk.
            // However, if we assume A is stored in row-major order:
            // A[i][j] is at offset i*n + j.
            // A[j][i] is at offset j*n + i.
            
            // This requires gathering. RVV has vrgather.
            
            // Let's use vrgather.
            // We need indices for the gather.
            // For the current chunk starting at j with length vn:
            // Indices for A:
            // If j <= i: indices are i*n + j, i*n + j + 1, ...
            // If j > i: indices are j*n + i, (j+1)*n + i, ...
            
            // This is complex to vectorize efficiently with simple gathers if the stride changes.
            // However, note that for a fixed i, the "row" index for the lower triangle access is:
            // max(i, j).
            // If j <= i, max(i,j) = i.
            // If j > i, max(i,j) = j.
            
            // Let's split the loop into two parts: j <= i and j > i.
            // Part 1: j from 0 to min(i, n-1)
            // Part 2: j from i+1 to n-1
            
            // This avoids the straddle and complex masking within a single vector op.
        }
        
        // Let's rewrite the inner loop logic properly with strip mining for the two segments.
        
        // Segment 1: j from 0 to min(i, n-1)
        int end1 = (i < n) ? i : n - 1;
        int j1 = 0;
        while (j1 <= end1) {
            int vn1 = vsetvl_e32_m1(end1 - j1 + 1);
            vfloat32m1_t vb1 = vle32_v_f32_m1(&b[j1], vn1);
            
            // Indices for A: row i, cols j1...j1+vn1-1
            // We can use vrgather with a generated index vector.
            // Index = i * n + j
            // We need a vector of indices: i*n + j1, i*n + j1 + 1, ...
            
            // Generate index vector
            // vindex_t is not directly available in standard RVV C API for gather in this way easily without intrinsics.
            // However, we can use vadd_vx_vx_vx to create indices if we had a base.
            // Actually, vrgather_vv_vx requires a base pointer and an index vector.
            // Let's assume we can generate the index vector.
            
            // In standard RVV intrinsics, vrgather_vv_vx is not always present.
            // Alternative: If A is symmetric and we are doing y = A*x, and A is stored in lower triangle packed?
            // The problem says "source-free", "infer operator".
            // "select lower-triangle A[max(i,j),min(i,j)]"
            // This implies A is a full matrix in memory, but we only access the lower triangle part.
            // If A is row-major, A[i][j] for j<=i is contiguous.
            // A[j][i] for j>i is strided (stride n).
            
            // Let's handle Segment 1 (j <= i): Contiguous access A[i][j].
            if (vn1 > 0) {
                vfloat32m1_t va1 = vle32_v_f32_m1(&a[i * n + j1], vn1);
                vfloat32m1_t vprod = vmul_vv_vx_vx_vx_vx(vb1, va1, vn1); // fma acc, A, x
                // acc += sum(vprod)
                // We need to reduce.
                // Since we are accumulating into a scalar 'acc', we can do a vector reduce.
                // But we are in a loop. Let's accumulate into a vector accumulator and reduce at the end of the segment?
                // No, 'acc' is scalar. We can accumulate into a vector register and reduce at the end of the entire row computation.
            }
            j1 += vn1;
        }
        
        // Segment 2: j from i+1 to n-1
        int start2 = i + 1;
        if (start2 < n) {
            int j2 = start2;
            while (j2 < n) {
                int vn2 = vsetvl_e32_m1(n - j2);
                vfloat32m1_t vb2 = vle32_v_f32_m1(&b[j2], vn2);
                
                // Indices for A: row j, col i.
                // A[j][i] is at offset j*n + i.
                // This is strided access with stride n.
                // We can use vlsseg or vrgather.
                // vrgather_vv_vx_vx_vx: vrgather_vv_vx_vx_vx(vd, vs2, vs1, vm)
                // We need an index vector: j2*n + i, (j2+1)*n + i, ...
                // This is an arithmetic progression.
                
                // Generate index vector:
                // idx = j2*n + i + k * n for k in 0..vn2-1
                // idx = (j2*n + i) + k * n
                // We can use vadd_vx_vx_vx_vx to add k*n to base?
                // Or use vrgather with a precomputed index vector?
                // Generating index vector:
                // vindex_t idx = vadd_vx_vx_vx_vx(vsetvl_e32_m1(vn2) ... )
                // This is getting complicated.
                
                // Let's use vrgather_vv_vx_vx_vx if available, or assume a helper.
                // For the sake of the exercise, let's assume we can gather.
                
                // Base address: &a[j2 * n + i]
                // Stride: n
                // We can use vlsseg_vf2 if we were loading pairs, but we need single floats.
                
                // Let's use vrgather.
                // vfloat32m1_t va2 = vrgather_vv_vx_vx_vx(&a[j2 * n + i], idx_vec, vn2);
                
                // Since we don't have idx_vec, let's create it.
                // idx_vec = vadd_vx_vx_vx_vx(vset_vx_vx_vx_vx(vn2, j2*n + i), vset_vx_vx_vx_vx(vn2, 0), n) ? No.
                // idx_vec = vadd_vv_vx_vx_vx_vx(vset_vx_vx_vx_vx(vn2, j2*n + i), vseq_vx_vx_vx_vx(vn2, 0, n))
                
                // This is too verbose for a simple migration.
                // Let's assume the compiler handles the gather or we use a simpler approach if n is small.
                // But we must use RVV.
                
                // Let's write the code assuming vrgather is available and we can generate indices.
                
                // To keep it clean, let's just do the scalar fallback for the strided part if vectorization is too hard?
                // No, the prompt asks for RVV.
                
                // Let's use a simpler assumption:
                // If the matrix is stored in column-major, then A[j][i] is contiguous for fixed i and varying j.
                // But the evidence says "A[max(i,j),min(i,j)]".
                // If A is column-major, A[j][i] is at offset j + i*n.
                // A[i][j] is at offset i + j*n.
                // This doesn't help with the max/min logic directly for contiguous access.
                
                // Let's stick to row-major and use vrgather.
                
                // Generate index vector for segment 2
                // Base index: j2 * n + i
                // Step: n
                // We need a vector of indices: base, base+n, base+2n, ...
                
                // vindex_t idx2 = vadd_vx_vx_vx_vx(vset_vx_vx_vx_vx(vn2, j2 * n + i), vseq_vx_vx_vx_vx(vn2, 0, n), n);
                // This is not standard.
                
                // Let's use a different approach.
                // Since this is a "Level-2 matrix/vector operator", and the evidence is sparse,
                // maybe the "lower-triangle" access implies a packed storage?
                // If A is packed in lower triangle, then A[i][j] for j<=i is at a specific offset.
                // But the evidence says "A[max(i,j),min(i,j)]", which implies accessing the full matrix A.
                
                // Let's assume the simplest case:
                // We can't easily vectorize the strided part with standard intrinsics without gather.
                // Let's use vrgather_vv_vx_vx_vx.
                
                // We need to generate the index vector.
                // Let's assume a helper function or intrinsic exists.
                
                // For the purpose of this code block, I will write the structure.
                
                vfloat32m1_t va2;
                // Placeholder for gather
                // va2 = vrgather_vv_vx_vx_vx(&a[j2 * n + i], idx2, vn2);
                
                // Since I cannot generate idx2 easily without more intrinsics,
                // let's assume the compiler can optimize the loop if we write it clearly.
                // But we must emit RVV code.
                
                // Let's use a scalar loop for the strided part if vectorization is not feasible?
                // No, the prompt asks for RVV.
                
                // Let's assume we can use vlsseg_vf2 to load pairs and interleave? No.
                
                // Let's try to use vrgather with a generated index vector using vadd_vx_vx_vx_vx and vseq.
                // vseq_vx_vx_vx_vx(vn2, 0, n) creates 0, n, 2n, ...
                // vset_vx_vx_vx_vx(vn2, j2 * n + i) creates base, base, ...
                // vadd_vv_vx_vx_vx_vx adds them.
                
                // This is the way.
                
                vint32m1_t idx2 = vadd_vv_vx_vx_vx_vx(
                    vset_vx_vx_vx_vx(vn2, j2 * n + i),
                    vseq_vx_vx_vx_vx(vn2, 0, n),
                    n
                );
                
                va2 = vrgather_vv_vx_vx_vx(&a[0], idx2, vn2);
                
                vfloat32m1_t vprod2 = vmul_vv_vx_vx_vx_vx(vb2, va2, vn2);
                
                // Accumulate vprod2 into a vector accumulator for the row?
                // We need to reduce vprod2 to a scalar and add to acc.
                // But we are in a loop.
                // Let's accumulate into a vector register 'vacc' for the whole row.
            }
        }
    }
    
    // The above logic is flawed because 'acc' is scalar and we are accumulating into it.
    // We should accumulate into a vector register for the entire
#include "harness.h"
