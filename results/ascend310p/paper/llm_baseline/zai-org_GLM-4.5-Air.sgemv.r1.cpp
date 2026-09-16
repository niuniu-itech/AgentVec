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
    
    // Allocate tensors in UB
    LocalTensor<float> local_x;
    LocalTensor<float> local_y;
    LocalTensor<float> local_A;
    LocalTensor<float> local_y_out;
    
    // Allocate double buffers
    LocalTensor<float> local_A_buf[2];
    LocalTensor<float> local_y_buf[2];
    
    // Create pipe for double buffering
    TQue<float, 2> que_A;
    TQue<float, 2> que_y;
    
    // Create pipe for data transfer
    TPipe pipe;
    pipe.InitBuffer(que_A, local_A_buf, 2);
    pipe.InitBuffer(que_y, local_y_buf, 2);
    
    // Allocate memory in UB
    int32_t ub_size = GetUbSize();
    int32_t tile_size = n * sizeof(float); // Size of one row of A
    int32_t max_tiles = ub_size / (tile_size + n * sizeof(float) + m * sizeof(float));
    max_tiles = min(max_tiles, 16); // Limit to prevent overflow
    
    // Initialize local tensors
    TensorInit(local_x, n);
    TensorInit(local_y, n);
    TensorInit(local_y_out, rows_to_process);
    TensorInit(local_A_buf[0], n);
    TensorInit(local_A_buf[1], n);
    TensorInit(local_y_buf[0], n);
    TensorInit(local_y_buf[1], n);
    
    // Copy x from GM to UB
    DataCopy(local_x, x, n, 0);
    
    // Process in tiles for better UB utilization
    for (int32_t row = start_row; row < end_row; row += max_tiles) {
        int32_t current_tile_size = min(max_tiles, end_row - row);
        
        // Double buffering for A and y
        for (int32_t buf_idx = 0; buf_idx < 2; buf_idx++) {
            // Copy current tile of A from GM to UB
            int32_t offset_A = row * n * sizeof(float);
            DataCopy(local_A_buf[buf_idx], A, current_tile_size * n, offset_A);
            
            // Copy corresponding y from GM to UB
            int32_t offset_y = row * sizeof(float);
            DataCopy(local_y_buf[buf_idx], y, current_tile_size, offset_y);
            
            // Enqueue in pipe
            que_A.Enque(local_A_buf[buf_idx]);
            que_y.Enque(local_y_buf[buf_idx]);
        }
        
        // Process the tile
        for (int32_t tile_row = 0; tile_row < current_tile_size; tile_row++) {
            // Get current row from pipe
            local_A = que_A.Deque();
            local_y = que_y.Deque();
            
            // Compute dot product: alpha * sum(A[i*n+j]*x[j])
            float sum = 0.0f;
            for (int32_t j = 0; j < n; j++) {
                sum += local_A.GetValue(j) * local_x.GetValue(j);
            }
            sum *= alpha;
            
            // Apply beta factor and add to y
            float y_val = local_y.GetValue(0) * beta + sum;
            local_y_out.SetValue(tile_row, y_val);
            
            // Enqueue back for next iteration
            que_A.Enque(local_A);
            que_y.Enque(local_y);
        }
    }
    
    // Copy results back to GM
    int32_t output_offset = start_row * sizeof(float);
    DataCopy(y, local_y_out, rows_to_process, output_offset);
}