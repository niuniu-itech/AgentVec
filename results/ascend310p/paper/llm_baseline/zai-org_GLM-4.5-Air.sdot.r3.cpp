#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void sdot(GM_ADDR x, GM_ADDR y, GM_ADDR partials, uint32_t total) {
    int32_t block_idx = GetBlockIdx();
    int32_t block_num = GetBlockNum();
    int32_t block_size = (total + block_num - 1) / block_num;
    int32_t start_idx = block_idx * block_size;
    int32_t end_idx = min(start_idx + block_size, total);
    
    LocalTensor<float> x_local;
    LocalTensor<float> y_local;
    
    // Allocate local tensors
    x_local.Alloc(256);
    y_local.Alloc(256);
    
    // Initialize partial sum for this block
    float block_sum = 0.0f;
    
    // Process data in chunks
    for (int32_t i = start_idx; i < end_idx; i += 256) {
        int32_t chunk_size = min(256, end_idx - i);
        
        // Copy data to local buffer
        DataCopy(x_local, x + i, chunk_size, 1);
        DataCopy(y_local, y + i, chunk_size, 1);
        
        // Compute dot product for this chunk
        for (int32_t j = 0; j < chunk_size; j++) {
            block_sum = block_sum + (x_local[j] * y_local[j]);
        }
    }
    
    // Store partial result
    *(float*)(partials + block_idx * 8 * sizeof(float)) = block_sum;
}