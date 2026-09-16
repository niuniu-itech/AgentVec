#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFNUM = 2;
constexpr int32_t TILE_M = 64;
constexpr int32_t TILE_N = 64;
constexpr int32_t TILE_K = 32;

class KSGEMM {
public:
    __aicore__ inline KSGEMM() {}
    
    __aicore__ inline void Init(GM_ADDR A, GM_ADDR B, GM_ADDR C, float alpha, float beta, uint32_t m, uint32_t n, uint32_t k) {
        this->alpha = alpha;
        this->beta = beta;
        this->m = m;
        this->n = n;
        this->k = k;
        
        block_m = (m + GetBlockNum() - 1) / GetBlockNum();
        block_n = n;
        block_k = k;
        
        Ag.SetGlobalBuffer((__gm__ float*)A, m * k);
        Bg.SetGlobalBuffer((__gm__ float*)B, k * n);
        Cg.SetGlobalBuffer((__gm__ float*)C, m * n);
        
        pipe.InitBuffer(qA, BUFNUM, TILE_M * TILE_K * sizeof(float));
        pipe.InitBuffer(qB, BUFNUM, TILE_K * TILE_N * sizeof(float));
        pipe.InitBuffer(qC, BUFNUM, TILE_M * TILE_N * sizeof(float));
        pipe.InitBuffer(tmp, TILE_M * TILE_N * sizeof(float));
    }
    
    __aicore__ inline void Process(uint32_t block_idx) {
        uint32_t block_m_start = block_idx * block_m;
        uint32_t block_m_end = (block_idx + 1) * block_m > m ? m : (block_idx + 1) * block_m;
        
        for (uint32_t tile_m_start = block_m_start; tile_m_start < block_m_end; tile_m_start += TILE_M) {
            uint32_t tile_m = (tile_m_start + TILE_M) > block_m_end ? (block_m_end - tile_m_start) : TILE_M;
            
            for (uint32_t tile_n_start = 0; tile_n_start < n; tile_n_start += TILE_N) {
                uint32_t tile_n = (tile_n_start + TILE_N) > n ? (n - tile_n_start) : TILE_N;
                
                // Initialize accumulator tile
                LocalTensor<float> acc = tmp.Get<float>();
                Duplicate(acc, 0.0f, tile_m * tile_n);
                
                // Process k dimension
                for (uint32_t tile_k_start = 0; tile_k_start < k; tile_k_start += TILE_K) {
                    uint32_t tile_k = (tile_k_start + TILE_K) > k ? (k - tile_k_start) : TILE_K;
                    
                    // Double buffering for A and B
                    LocalTensor<float> tileA, tileB;
                    
                    // Load A tile
                    tileA = qA.AllocTensor<float>();
                    DataCopy(tileA, Ag[tile_m_start * k + tile_k_start], tile_m * tile_k);
                    qA.EnQue(tileA);
                    
                    // Load B tile
                    tileB = qB.AllocTensor<float>();
                    DataCopy(tileB, Bg[tile_k_start * n + tile_n_start], tile_k * tile_n);
                    qB.EnQue(tileB);
                    
                    // Dequeue and compute
                    tileA = qA.DeQue<float>();
                    tileB = qB.DeQue<float>();
                    
                    // Matrix multiplication
                    for (uint32_t i = 0; i < tile_m; i++) {
                        for (uint32_t j = 0; j < tile_n; j++) {
                            float sum = 0.0f;
                            for (uint32_t t = 0; t < tile_k; t++) {
                                sum += tileA.GetValue(i * tile_k + t) * tileB.GetValue(t * tile_n + j);
                            }
                            float current = acc.GetValue(i * tile_n + j);
                            acc.SetValue(i * tile_n + j, current + sum);
                        }
                    }
                    
                    // Free tensors
                    qA.FreeTensor(tileA);
                    qB.FreeTensor(tileB);
                }
                
                // Apply alpha and beta
                LocalTensor<float> tileC = qC.AllocTensor<float>();
                if (beta != 0.0f) {
                    DataCopy(tileC, Cg[tile_m_start * n + tile_n_start], tile_m * tile_n);
                    for (uint32_t i = 0; i < tile_m; i++) {
                        for (uint32_t j = 0; j < tile_n; j++) {
                            float acc_val = acc.GetValue(i * tile_n + j);
                            float c_val = tileC.GetValue(i * tile_n + j);
                            acc.SetValue(i * tile_n + j, alpha * acc_val + beta * c_val);
                        }
                    }
                } else {
                    for (uint32_t i = 0; i < tile_m; i++) {
                        for (uint32_t j = 0; j < tile_n; j++) {
                            float acc_val = acc.GetValue(i * tile_n + j);
                            acc.SetValue(i * tile_n + j, alpha * acc_val);
                        }
                    }
                }
                
                // Store result
                DataCopy(Cg[tile_m_start * n + tile_n_start], acc, tile_m * tile_n);
                qC.EnQue(tileC);
                tileC = qC.DeQue<float>();
                qC.FreeTensor(tileC);
            }
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
    uint32_t block_m, block_n, block_k;
};

extern "C" __global__ __aicore__ void sgemm(GM_ADDR A, GM_ADDR B, GM_ADDR C, float alpha, float beta, uint32_t m, uint32_t n, uint32_t k) {
    KSGEMM op;
    op.Init(A, B, C, alpha, beta, m, n, k);
    uint32_t block_idx = GetBlockIdx();
    op.Process(block_idx);
}