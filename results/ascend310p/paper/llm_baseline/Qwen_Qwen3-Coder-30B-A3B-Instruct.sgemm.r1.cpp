#include "kernel_operator.h"
using namespace AscendC;

#define TILE_M 16
#define TILE_N 16
#define TILE_K 16
#define UNIFIED_BUFFER_SIZE (TILE_M * TILE_K + TILE_K * TILE_N)
#define DBUF_NUM 2

extern "C" __global__ __aicore__ void sgemm(GM_ADDR A, GM_ADDR B, GM_ADDR C, float alpha, float beta, uint32_t m, uint32_t n, uint32_t k) {
    // Shared memory for tiles
    LocalTensor<float> a_tile = LocalTensor<float>::AllocLocalTensor<TILE_M, TILE_K>();
    LocalTensor<float> b_tile = LocalTensor<float>::AllocLocalTensor<TILE_K, TILE_N>();
    LocalTensor<float> c_tile = LocalTensor<float>::AllocLocalTensor<TILE_M, TILE_N>();
    
    // Unified buffer for double buffering
    LocalTensor<float> unified_buffer = LocalTensor<float>::AllocLocalTensor<UNIFIED_BUFFER_SIZE>();
    
    // Initialize C tile
    for (int i = 0; i < TILE_M; i++) {
        for (int j = 0; j < TILE_N; j++) {
            c_tile(i, j) = 0.0f;
        }
    }
    
    // Get block info
    int block_id = GetBlockIdx();
    int block_num = GetBlockNum();
    
    // Loop over tiles of k dimension
    for (int k_block = 0; k_block < k; k_block += TILE_K) {
        // Load A tile
        for (int i = 0; i < TILE_M; i++) {
            for (int j = 0; j < TILE_K; j++) {
                int global_i = block_id * TILE_M + i;
                int global_j = k_block + j;
                if (global_i < m && global_j < k) {
                    a_tile(i, j) = *(float*)(A + (global_i * k + global_j) * sizeof(float));
                } else {
                    a_tile(i, j) = 0.0f;
                }
            }
        }
        
        // Load B tile
        for (int i = 0; i < TILE_K; i++) {
            for (int j = 0; j < TILE_N; j++) {
                int global_i = k_block + i;
                int global_j = j;
                if (global_i < k && global_j < n) {
                    b_tile(i, j) = *(float*)(B + (global_i * n + global_j) * sizeof(float));
                } else {
                    b_tile(i, j) = 0.0f;
                }
            }
        }
        
        // Compute C tile
        for (int i = 0; i < TILE_M; i++) {
            for (int j = 0; j < TILE_N; j++) {
                float sum = 0.0f;
                for (int t = 0; t < TILE_K; t++) {
                    sum += a_tile(i, t) * b_tile(t, j);
                }
                c_tile(i, j) += sum;
            }
        }
    }
    
    // Apply alpha and beta
    for (int i = 0; i < TILE_M; i++) {
        for (int j = 0; j < TILE_N; j++) {
            int global_i = block_id * TILE_M + i;
            int global_j = j;
            if (global_i < m && global_j < n) {
                float c_val = *(float*)(C + (global_i * n + global_j) * sizeof(float));
                float result = alpha * c_tile(i, j) + beta * c_val;
                *(float*)(C + (global_i * n + global_j) * sizeof(float)) = result;
            }
        }
    }
}