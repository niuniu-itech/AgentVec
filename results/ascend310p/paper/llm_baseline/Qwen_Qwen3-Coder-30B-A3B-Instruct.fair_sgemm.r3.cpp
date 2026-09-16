#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFNUM = 2;
constexpr int32_t TILE_M = 16;
constexpr int32_t TILE_N = 16;
constexpr int32_t TILE_K = 16;

class KSGEMM {
public:
  __aicore__ inline KSGEMM() {}
  __aicore__ inline void Init(GM_ADDR A, GM_ADDR B, GM_ADDR C, float alpha, float beta, uint32_t m, uint32_t n, uint32_t k) {
    this->alpha = alpha;
    this->beta = beta;
    this->m = m;
    this->n = n;
    this->k = k;
    
    blockM = m / GetBlockNum();
    blockStart = blockM * GetBlockIdx();
    
    // Initialize global tensors
    Ag.SetGlobalBuffer((__gm__ float*)A + blockStart * k, blockM * k);
    Bg.SetGlobalBuffer((__gm__ float*)B, k * n);
    Cg.SetGlobalBuffer((__gm__ float*)C + blockStart * n, blockM * n);
    
    // Initialize pipe buffers
    pipe.InitBuffer(qa, BUFNUM, TILE_M * TILE_K * sizeof(float));
    pipe.InitBuffer(qb, BUFNUM, TILE_K * TILE_N * sizeof(float));
    pipe.InitBuffer(qc, BUFNUM, TILE_M * TILE_N * sizeof(float));
    pipe.InitBuffer(tmp, TILE_M * TILE_N * sizeof(float));
  }
  
  __aicore__ inline void Process() {
    // Initialize accumulator
    LocalTensor<float> acc = tmp.Get<float>();
    Duplicate(acc, 0.0f, TILE_M * TILE_N);
    
    // Loop over tiles of k dimension
    uint32_t numTilesK = k / TILE_K;
    for (uint32_t tileK = 0; tileK < numTilesK; ++tileK) {
      // Load tile from A
      LocalTensor<float> Al = qa.AllocTensor<float>();
      DataCopy(Al, Ag[tileK * TILE_M * TILE_K], TILE_M * TILE_K);
      qa.EnQue(Al);
      
      // Load tile from B
      LocalTensor<float> Bl = qb.AllocTensor<float>();
      DataCopy(Bl, Bg[tileK * TILE_K * n], TILE_K * TILE_N);
      qb.EnQue(Bl);
      
      // Synchronize
      Al = qa.DeQue<float>();
      Bl = qb.DeQue<float>();
      
      // Compute GEMM tile
      LocalTensor<float> Cl = qc.AllocTensor<float>();
      MatMul(Cl, Al, Bl, TILE_M, TILE_N, TILE_K);
      qc.EnQue(Cl);
      
      // Free tensors
      qa.FreeTensor(Al);
      qb.FreeTensor(Bl);
      
      // Accumulate
      Cl = qc.DeQue<float>();
      Add(acc, acc, Cl, TILE_M * TILE_N);
      qc.FreeTensor(Cl);
    }
    
    // Scale and add beta * C
    LocalTensor<float> Ctmp = qc.AllocTensor<float>();
    DataCopy(Ctmp, Cg[0], TILE_M * TILE_N);
    qc.EnQue(Ctmp);
    Ctmp = qc.DeQue<float>();
    
    // Apply alpha and beta
    Muls(acc, acc, alpha, TILE_M * TILE_N);
    Muls(Ctmp, Ctmp, beta, TILE_M * TILE_N);
    Add(acc, acc, Ctmp, TILE_M * TILE_N);
    
    // Store result
    DataCopy(Cg[0], acc, TILE_M * TILE_N);
    qc.FreeTensor(Ctmp);
  }
  
private:
  TPipe pipe;
  TQue<QuePosition::VECIN, BUFNUM> qa, qb;
  TQue<QuePosition::VECOUT, BUFNUM> qc;
  TBuf<QuePosition::VECCALC> tmp;
  
  GlobalTensor<float> Ag, Bg, Cg;
  float alpha, beta;
  uint32_t m, n, k, blockM, blockStart;
  
  __aicore__ inline void MatMul(LocalTensor<float>& C, const LocalTensor<float>& A, const LocalTensor<float>& B, 
                                uint32_t m, uint32_t n, uint32_t k) {
    // Use vectorized operations for better performance
    for (uint32_t i = 0; i < m; ++i) {
      for (uint32_t j = 0; j < n; ++j) {
        float sum = 0.0f;
        for (uint32_t t = 0; t < k; ++t) {
          sum += A.GetValue(i * k + t) * B.GetValue(t * n + j);
        }
        C.SetValue(i * n + j, sum);
      }
    }
  }
};

extern "C" __global__ __aicore__ void sgemm(GM_ADDR A, GM_ADDR B, GM_ADDR C, float alpha, float beta, uint32_t m, uint32_t n, uint32_t k) {
  KSGEMM kernel;
  kernel.Init(A, B, C, alpha, beta, m, n, k);
  kernel.Process();
}