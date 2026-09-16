#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void saxpy(GM_ADDR x, GM_ADDR y, GM_ADDR out, float alpha, uint32_t total) {
    // Get block information
    uint32_t blockId = GetBlockIdx();
    uint32_t blockNum = GetBlockNum();
    
    // Each block processes total / blockNum elements
    uint32_t stride = total / blockNum;
    uint32_t start = blockId * stride;
    uint32_t end = (blockId == blockNum - 1) ? total : (blockId + 1) * stride;
    uint32_t len = end - start;
    
    // Define tensor dimensions
    const int32_t VEC_LEN = 16; // 16 floats per vector operation
    const int32_t TILE_SIZE = 256; // 256 elements per tile
    
    // Local tensor for input data
    LocalTensor<float> local_x = LocalTensor<float>::AllocLocalTensor<float>(TILE_SIZE);
    LocalTensor<float> local_y = LocalTensor<float>::AllocLocalTensor<float>(TILE_SIZE);
    LocalTensor<float> local_out = LocalTensor<float>::AllocLocalTensor<float>(TILE_SIZE);
    
    // Process data in tiles
    for (uint32_t offset = 0; offset < len; offset += TILE_SIZE) {
        uint32_t current_tile_size = min(TILE_SIZE, len - offset);
        uint32_t actual_offset = start + offset;
        
        // Load data
        DataCopy(local_x, x + actual_offset, current_tile_size, 0);
        DataCopy(local_y, y + actual_offset, current_tile_size, 0);
        
        // Vectorized computation
        for (int32_t i = 0; i < current_tile_size; i += VEC_LEN) {
            // Load vectors
            auto vec_x = local_x.Load(i);
            auto vec_y = local_y.Load(i);
            
            // Compute: out = alpha * x + y
            auto vec_alpha = vec_x * alpha;
            auto vec_result = vec_alpha + vec_y;
            
            // Store result
            local_out.Store(i, vec_result);
        }
        
        // Store result back to global memory
        DataCopy(out + actual_offset, local_out, current_tile_size, 0);
    }
}