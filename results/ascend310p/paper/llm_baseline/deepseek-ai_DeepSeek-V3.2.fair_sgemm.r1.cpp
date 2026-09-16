#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFNUM = 2;
constexpr int32_t TILE_M = 32;
constexpr int32_t TILE_N = 32;
constexpr int32_t TILE_K = 32;
constexpr int32_t VEC_LEN = 8;

class K_SGEMM {
public:
    __aicore__ inline K_SGEMM() {}
    
    __aicore__ inline void Init(GM_ADDR A, GM_ADDR B, GM_ADDR C, float alpha, float beta,
                                uint32_t m, uint32_t n, uint32_t k) {
        this->alpha = alpha;
        this->beta = beta;
        this->m = m;
        this->n = n;
        this->k = k;
        
        // Split work across 8 cores
        uint32_t total_tiles_m = (m + TILE_M - 1) / TILE_M;
        uint32_t total_tiles_n = (n + TILE_N - 1) / TILE_N;
        uint32_t total_tiles = total_tiles_m * total_tiles_n;
        
        uint32_t tiles_per_core = (total_tiles + GetBlockNum() - 1) / GetBlockNum();
        start_tile = GetBlockIdx() * tiles_per_core;
        end_tile = min(start_tile + tiles_per_core, total_tiles);
        
        // Initialize global tensors
        Ag.SetGlobalBuffer((__gm__ float*)A, m * k);
        Bg.SetGlobalBuffer((__gm__ float*)B, k * n);
        Cg.SetGlobalBuffer((__gm__ float*)C, m * n);
        
        // Initialize pipe buffers
        pipe.InitBuffer(qA, BUFNUM, TILE_M * TILE_K * sizeof(float));
        pipe.InitBuffer(qB, BUFNUM, TILE_K * TILE_N * sizeof(float));
        pipe.InitBuffer(qC, BUFNUM, TILE_M * TILE_N * sizeof(float));
        pipe.InitBuffer(tmp, TILE_M * TILE_N * sizeof(float));
    }
    
    __aicore__ inline void Process() {
        uint32_t tiles_m = (m + TILE_M - 1) / TILE_M;
        uint32_t tiles_n = (n + TILE_N - 1) / TILE_N;
        
        for (uint32_t tile_idx = start_tile; tile_idx < end_tile; ++tile_idx) {
            uint32_t tile_m = tile_idx / tiles_n;
            uint32_t tile_n = tile_idx % tiles_n;
            
            uint32_t m_start = tile_m * TILE_M;
            uint32_t n_start = tile_n * TILE_N;
            uint32_t m_end = min(m_start + TILE_M, m);
            uint32_t n_end = min(n_start + TILE_N, n);
            uint32_t m_len = m_end - m_start;
            uint32_t n_len = n_end - n_start;
            
            // Initialize accumulator to zero
            LocalTensor<float> acc = tmp.Get<float>();
            Duplicate(acc, 0.0f, TILE_M * TILE_N);
            
            // Process K dimension in tiles
            uint32_t tiles_k = (k + TILE_K - 1) / TILE_K;
            for (uint32_t tile_k = 0; tile_k < tiles_k; ++tile_k) {
                uint32_t k_start = tile_k * TILE_K;
                uint32_t k_end = min(k_start + TILE_K, k);
                uint32_t k_len = k_end - k_start;
                
                // Double buffering: load next tile while computing current
                if (tile_k < tiles_k - 1) {
                    LocalTensor<float> nextA = qA.AllocTensor<float>();
                    LocalTensor<float> nextB = qB.AllocTensor<float>();
                    
                    // Load next A tile
                    uint32_t a_offset = m_start * k + k_start + TILE_K;
                    uint32_t a_count = m_len * k_len;
                    DataCopy(nextA, Ag[a_offset], a_count);
                    qA.EnQue(nextA);
                    
                    // Load next B tile  
                    uint32_t b_offset = (k_start + TILE_K) * n + n_start;
                    uint32_t b_count = k_len * n_len;
                    DataCopy(nextB, Bg[b_offset], b_count);
                    qB.EnQue(nextB);
                }
                
                // Load current tiles
                LocalTensor<float> curA, curB;
                if (tile_k == 0) {
                    curA = qA.AllocTensor<float>();
                    curB = qB.AllocTensor<float>();
                    
                    uint32_t a_offset = m_start * k + k_start;
                    uint32_t a_count = m_len * k_len;
                    DataCopy(curA, Ag[a_offset], a_count);
                    
                    uint32_t b_offset = k_start * n + n_start;
                    uint32_t b_count = k_len * n_len;
                    DataCopy(curB, Bg[b_offset], b_count);
                } else {
                    curA = qA.DeQue<float>();
                    curB = qB.DeQue<float>();
                }
                
                // Matrix multiplication: acc += A * B
                for (uint32_t i = 0; i < m_len; ++i) {
                    for (uint32_t j = 0; j < n_len; j += VEC_LEN) {
                        // Load B column vector
                        LocalTensor<float> b_vec = tmp.Get<float>();
                        for (uint32_t v = 0; v < VEC_LEN; ++v) {
                            if (j + v < n_len) {
                                b_vec.SetValue(v, curB.GetValue(i * n_len + j + v));
                            }
                        }
                        
                        // Compute dot product with A row
                        for (uint32_t t = 0; t < k_len; ++t) {
                            float a_val = curA.GetValue(i * k_len + t);
                            LocalTensor<float> temp = tmp.Get<float>();
                            Muls(temp, b_vec, a_val, VEC_LEN);
                            
                            // Add to accumulator
                            uint32_t acc_idx = i * TILE_N + j;
                            LocalTensor<float> acc_vec = tmp.Get<float>();
                            for (uint32_t v = 0; v < VEC_LEN; ++v) {
                                if (j + v < n_len) {
                                    float current = acc.GetValue(acc_idx + v);
                                    float add_val = temp.GetValue(v);
                                    acc.SetValue(acc_idx + v, current + add_val);
                                }
                            }
                        }
                    }
                }
                
                if (tile_k > 0) {
                    qA.FreeTensor(curA);
                    qB.FreeTensor(curB);
                }
            }
            
            // Apply alpha and beta
            LocalTensor<float> c_tile = qC.AllocTensor<float>();
            
            // Load original C tile
            uint32_t c_offset = m_start * n + n_start;
            uint32_t c_count = m_len * n_len;
            DataCopy(c_tile, Cg[c_offset], c_count);
            
            // C = alpha * acc + beta * C
            for (uint32_t i = 0; i < m_len; ++i) {
                for (uint32_t j = 0; j < n_len; j += VEC_LEN) {
                    uint32_t idx = i * TILE_N + j;
                    
                    // Load accumulator vector
                    LocalTensor<float> acc_vec = tmp.Get<float>();
                    for (uint32_t v = 0; v < VEC_LEN; ++v) {
                        if (j + v < n_len) {
                            acc_vec.SetValue(v, acc.GetValue(idx + v));
                        }
                    }
                    
                    // Load C vector
                    LocalTensor<float> c_vec = tmp.Get<float>();
                    for (uint32_t v = 0; v < VEC_LEN; ++v) {
                        if (j + v < n_len) {
                            c_vec.SetValue(v, c_tile.GetValue(idx + v));
                        }
                    }
                    
                    // Compute: alpha * acc + beta * C
                    LocalTensor<float> temp1 = tmp.Get<float>();
                    Muls(temp1, acc_vec, alpha, VEC_LEN);
                    
                    LocalTensor<float> temp2 = tmp.Get<float>();
                    Muls(temp2, c_vec, beta, VEC_LEN);
                    
                    LocalTensor<float> result = tmp.Get<float>();
                    Add(result, temp1, temp2, VEC_LEN);
                    
                    // Store back to c_tile
                    for (uint32_t v = 0; v < VEC_LEN; ++v) {
                        if (j + v < n_len) {
                            c_tile.SetValue(idx + v, result.GetValue(v));
                        }
                    }
                }
            }
            
            // Write back to global memory
            DataCopy(Cg[c_offset], c_tile, c_count);
            qC.FreeTensor(c_tile);
        }
    }
    
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFNUM> qA, qB;
    TQue<QuePosition::VECOUT, BUFNUM> qC;
    TBuf<QuePosition::VECCALC> tmp;
    
    GlobalTensor<float> Ag, Bg, Cg;
    float alpha, beta;
    uint32_t m, n, k;
    uint32_t start_tile, end_tile;
};

extern "C" __global__ __aicore__ void sgemm(GM_ADDR A, GM_ADDR B, GM_ADDR C, 
                                           float alpha, float beta, 
                                           uint32_t m, uint32_t n, uint32_t k) {
    K_SGEMM sgemm_kernel;
    sgemm_kernel.Init(A, B, C, alpha, beta, m, n, k);
    sgemm_kernel.Process();
}