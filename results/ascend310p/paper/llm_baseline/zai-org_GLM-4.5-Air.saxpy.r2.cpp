#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void saxpy(GM_ADDR x, GM_ADDR y, GM_ADDR out, float alpha, uint32_t total) {
    // Initialize tensors
    LocalTensor<float> x_local;
    LocalTensor<float> y_local;
    LocalTensor<float> out_local;
    
    // Create tensor descriptors
    TensorDesc x_desc = TensorDesc(DATA_TYPE_FLOAT32, {total}, {sizeof(float)});
    TensorDesc y_desc = TensorDesc(DATA_TYPE_FLOAT32, {total}, {sizeof(float)});
    TensorDesc out_desc = TensorDesc(DATA_TYPE_FLOAT32, {total}, {sizeof(float)});
    
    // Initialize local tensors
    x_local = LocalTensor<float>(x_desc);
    y_local = LocalTensor<float>(y_desc);
    out_local = LocalTensor<float>(out_desc);
    
    // Create pipe for double buffering
    TPipe pipe;
    TQue<QuePosition::VEC, BUFFER_TYPE::P1> x_que(1, 1);
    TQue<QuePosition::VEC, BUFFER_TYPE::P1> y_que(1, 1);
    TQue<QuePosition::VEC, BUFFER_TYPE::P1> out_que(1, 1);
    
    pipe.Queue(x_que);
    pipe.Queue(y_que);
    pipe.Queue(out_que);
    
    // Calculate tile size based on unified buffer size
    uint32_t ub_size = 256 * 1024; // 256 KB
    uint32_t tile_size = ub_size / (3 * sizeof(float)); // Divide by 3 tensors
    tile_size = (tile_size < 1024) ? 1024 : tile_size; // Minimum tile size
    
    // Calculate number of tiles
    uint32_t num_tiles = (total + tile_size - 1) / tile_size;
    
    // Get block info
    uint32_t block_idx = GetBlockIdx();
    uint32_t block_num = GetBlockNum();
    
    // Calculate tiles per block
    uint32_t tiles_per_block = (num_tiles + block_num - 1) / block_num;
    uint32_t start_tile = block_idx * tiles_per_block;
    uint32_t end_tile = min(start_tile + tiles_per_block, num_tiles);
    
    // Process tiles
    for (uint32_t tile = start_tile; tile < end_tile; ++tile) {
        uint32_t tile_start = tile * tile_size;
        uint32_t tile_end = min(tile_start + tile_size, total);
        uint32_t tile_size_actual = tile_end - tile_start;
        
        // Double buffering
        for (int i = 0; i < 2; ++i) {
            // Copy data from global to local memory
            DataCopy(x_local, GlobalTensor<float>(x, x_desc), tile_start, tile_size_actual);
            DataCopy(y_local, GlobalTensor<float>(y, y_desc), tile_start, tile_size_actual);
            
            // Wait for copy to complete
            __aicore__::sync_all();
            
            // Compute saxpy: out = alpha * x + y
            // Vectorized computation
            int vec_size = tile_size_actual / 4;
            int remaining = tile_size_actual % 4;
            
            // Process in vector chunks
            for (int j = 0; j < vec_size; ++j) {
                float4 x_vec = x_local[j];
                float4 y_vec = y_local[j];
                float4 out_vec = alpha * x_vec + y_vec;
                out_local[j] = out_vec;
            }
            
            // Process remaining elements
            for (int j = vec_size * 4; j < tile_size_actual; ++j) {
                out_local[j] = alpha * x_local[j] + y_local[j];
            }
            
            // Copy result back to global memory
            DataCopy(GlobalTensor<float>(out, out_desc), out_local, tile_start, tile_size_actual);
            
            // Wait for copy to complete
            __aicore__::sync_all();
        }
    }
}