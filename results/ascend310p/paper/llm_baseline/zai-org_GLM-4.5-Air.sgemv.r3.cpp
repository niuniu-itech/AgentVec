#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void sgemv(GM_ADDR A, GM_ADDR x, GM_ADDR y, float alpha, float beta, uint32_t m, uint32_t n) {
    int32_t block_idx = GetBlockIdx();
    int32_t block_num = GetBlockNum();
    
    // Calculate the number of rows each block should process
    int32_t rows_per_block = (m + block_num - 1) / block_num;
    int32_t start_row = block_idx * rows_per_block;
    int32_t end_row = min((block_idx + 1) * rows_per_block, m);
    int32_t rows_to_process = end_row - start_row;
    
    // Create GlobalTensor objects
    GlobalTensor<float> global_A(A);
    GlobalTensor<float> global_x(x);
    GlobalTensor<float> global_y(y);
    
    // Allocate tensors in UB
    LocalTensor<float> local_x;
    LocalTensor<float> local_y;
    LocalTensor<float> local_A;
    LocalTensor<float> local_y_out;
    
    // Initialize local tensors
    TensorInit(local_x, n);
    TensorInit(local_y, 1);
    TensorInit(local_y_out, 1);
    TensorInit(local_A, n);
    
    // Copy x from GM to UB
    DataCopy(local_x, global_x, n);
    
    // Process rows
    for (int32_t row = start_row; row < end_row; row++) {
        // Create sub-tensors for the current row
        GlobalTensor<float> global_A_row = global_A.GetSubTensor(row * n, n);
        GlobalTensor<float> global_y_row = global_y.GetSubTensor(row, 1);
        
        // Copy current row of A and y from GM to UB
        DataCopy(local_A, global_A_row, n);
        DataCopy(local_y, global_y_row, 1);
        
        // Compute dot product
        float sum = 0.0f;
        
        // Process in chunks for better performance
        const int32_t UNROLL_FACTOR = 4;
        int32_t unrolled_iters = n / UNROLL_FACTOR;
        
        // Unrolled loop for better performance
        for (int32_t i = 0; i < unrolled_iters; i++) {
            sum += local_A.GetValue(i * UNROLL_FACTOR) * local_x.GetValue(i * UNROLL_FACTOR);
            sum += local_A.GetValue(i * UNROLL_FACTOR + 1) * local_x.GetValue(i * UNROLL_FACTOR + 1);
            sum += local_A.GetValue(i * UNROLL_FACTOR + 2) * local_x.GetValue(i * UNROLL_FACTOR + 2);
            sum += local_A.GetValue(i * UNROLL_FACTOR + 3) * local_x.GetValue(i * UNROLL_FACTOR + 3);
        }
        
        // Handle remaining elements
        for (int32_t i = unrolled_iters * UNROLL_FACTOR; i < n; i++) {
            sum += local_A.GetValue(i) * local_x.GetValue(i);
        }
        
        // Apply alpha and beta
        sum *= alpha;
        float y_val = local_y.GetValue(0) * beta + sum;
        local_y_out.SetValue(0, y_val);
        
        // Copy result back to GM
        DataCopy(global_y_row, local_y_out, 1);
    }
}