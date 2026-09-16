#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFNUM = 2;
constexpr int32_t TILE_M = 16;
constexpr int32_t TILE_N = 256;

class KSGEMV {
public:
  __aicore__ inline KSGEMV() {}
  __aicore__ inline void Init(GM_ADDR A, GM_ADDR x, GM_ADDR y, float alpha, float beta, uint32_t m, uint32_t n) {
    this->alpha = alpha;
    this->beta = beta;
    this->m = m;
    this->n = n;
    
    blockId = GetBlockIdx();
    blockNum = GetBlockNum();
    
    // Each block handles one row of output
    rowsPerBlock = (m + blockNum - 1) / blockNum;
    startRow = blockId * rowsPerBlock;
    endRow = (blockId == blockNum - 1) ? m : (blockId + 1) * rowsPerBlock;
    rowsHandled = endRow - startRow;
    
    // Initialize global tensors
    Ag.SetGlobalBuffer((__gm__ float*)A, m * n);
    xg.SetGlobalBuffer((__gm__ float*)x, n);
    yg.SetGlobalBuffer((__gm__ float*)y, m);
    
    // Initialize pipe buffers
    pipe.InitBuffer(qA, BUFNUM, TILE_M * TILE_N * sizeof(float));
    pipe.InitBuffer(qx, BUFNUM, TILE_N * sizeof(float));
    pipe.InitBuffer(qy, BUFNUM, TILE_M * sizeof(float));
  }
  
  __aicore__ inline void Process() {
    // Process rows assigned to this block
    for (uint32_t row = 0; row < rowsHandled; row += TILE_M) {
      uint32_t currentRow = startRow + row;
      uint32_t rowsToProcess = (rowsHandled - row) < TILE_M ? (rowsHandled - row) : TILE_M;
      
      // Allocate tensors for this tile
      LocalTensor<float> Al = qA.AllocTensor<float>();
      LocalTensor<float> xl = qx.AllocTensor<float>();
      
      // Load A tile (row-major)
      uint32_t loadRows = (m - currentRow) < TILE_M ? (m - currentRow) : TILE_M;
      loadRows = loadRows < rowsToProcess ? loadRows : rowsToProcess;
      uint32_t loadBytes = loadRows * n * sizeof(float);
      
      // Align to 32-byte boundary
      uint32_t alignedBytes = (loadBytes + 31) & ~31;
      DataCopy(Al, Ag[currentRow * n], alignedBytes / sizeof(float));
      
      // Load x vector
      DataCopy(xl, xg[0], n);
      
      // Compute partial dot products
      LocalTensor<float> tmpRes = qy.AllocTensor<float>();
      Duplicate(tmpRes, 0.0f, TILE_M);
      
      // Perform dot product for each row
      for (uint32_t i = 0; i < loadRows; i++) {
        LocalTensor<float> rowVec = Al[i * n];
        LocalTensor<float> partial = qy.AllocTensor<float>();
        Mul(partial, rowVec, xl, n);
        
        // Reduction
        float sum = 0.0f;
        ReduceSum<float>(&sum, partial, n);
        
        // Accumulate
        tmpRes[i] = tmpRes[i] + sum;
        
        qy.FreeTensor(partial);
      }
      
      // Apply alpha and beta
      for (uint32_t i = 0; i < loadRows; i++) {
        float yVal = 0.0f;
        DataCopy(&yVal, yg[currentRow + i], 1);
        tmpRes[i] = alpha * tmpRes[i] + beta * yVal;
      }
      
      // Store result
      DataCopy(yg[currentRow], tmpRes, loadRows);
      
      // Free tensors
      qA.FreeTensor(Al);
      qx.FreeTensor(xl);
      qy.FreeTensor(tmpRes);
    }
  }

private:
  TPipe pipe;
  TQue<QuePosition::VECIN, BUFNUM> qA, qx;
  TQue<QuePosition::VECOUT, BUFNUM> qy;
  
  GlobalTensor<float> Ag, xg, yg;
  
  float alpha, beta;
  uint32_t m, n;
  uint32_t blockId, blockNum;
  uint32_t startRow, endRow, rowsHandled;
  uint32_t rowsPerBlock;
};

extern "C" __global__ __aicore__ void sgemv(GM_ADDR A, GM_ADDR x, GM_ADDR y, float alpha, float beta, uint32_t m, uint32_t n) {
  KSGEMV kernel;
  kernel.Init(A, x, y, alpha, beta, m, n);
  kernel.Process();
}