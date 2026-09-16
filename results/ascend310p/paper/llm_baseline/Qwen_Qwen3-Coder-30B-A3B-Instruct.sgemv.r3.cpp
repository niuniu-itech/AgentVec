#include "kernel_operator.h"
using namespace AscendC;

#define TILE_M 16
#define TILE_N 64
#define VEC_SIZE 16

extern "C" __global__ __aicore__ void sgemv(GM_ADDR A, GM_ADDR x, GM_ADDR y, float alpha, float beta, uint32_t m, uint32_t n) {
    // Shared memory for double buffering
    LocalTensor<float> buf_a = LocalTensor<float>::AllocLocalTensor<TILE_M * TILE_N>();
    LocalTensor<float> buf_x = LocalTensor<float>::AllocLocalTensor<TILE_N>();
    
    // Get block information
    int32_t block_id = GetBlockIdx();
    int32_t block_num = GetBlockNum();
    
    // Process rows in chunks
    for (int32_t row_start = block_id * TILE_M; row_start < m; row_start += block_num * TILE_M) {
        // Load x vector
        int32_t x_load_size = (n + VEC_SIZE - 1) / VEC_SIZE;
        for (int32_t i = 0; i < x_load_size; ++i) {
            int32_t x_offset = i * VEC_SIZE;
            int32_t x_size = min(VEC_SIZE, n - x_offset);
            if (x_size > 0) {
                DataCopy(buf_x + x_offset, x + x_offset, x_size * sizeof(float));
            }
        }
        
        // Process TILE_M rows
        for (int32_t row = 0; row < TILE_M && (row_start + row) < m; ++row) {
            // Load A row
            int32_t a_row_offset = (row_start + row) * n;
            int32_t a_load_size = (n + VEC_SIZE - 1) / VEC_SIZE;
            for (int32_t i = 0; i < a_load_size; ++i) {
                int32_t a_offset = i * VEC_SIZE;
                int32_t a_size = min(VEC_SIZE, n - a_offset);
                if (a_size > 0) {
                    DataCopy(buf_a + row * TILE_N + a_offset, A + a_row_offset + a_offset, a_size * sizeof(float));
                }
            }
            
            // Compute dot product for this row
            float sum = 0.0f;
            for (int32_t j = 0; j < n; j += VEC_SIZE) {
                auto vec_a = vec_load<float, VEC_SIZE>(buf_a + row * TILE_N + j);
                auto vec_x = vec_load<float, VEC_SIZE>(buf_x + j);
                sum += vec_reduce_sum<float, VEC_SIZE>(vec_a * vec_x);
            }
            
            // Load existing y value
            float y_val = 0.0f;
            DataCopy(&y_val, y + row_start + row, sizeof(float));
            
            // Apply alpha and beta
            float result = alpha * sum + beta * y_val;
            
            // Store result back
            DataCopy(y + row_start + row, &result, sizeof(float));
        }
    }
}