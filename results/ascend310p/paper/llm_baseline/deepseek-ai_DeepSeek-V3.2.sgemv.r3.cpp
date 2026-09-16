#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t TILE_M = 256;
constexpr int32_t TILE_N = 256;

extern "C" __global__ __aicore__ void sgemv(GM_ADDR A, GM_ADDR x, GM_ADDR y, float alpha, float beta, uint32_t m, uint32_t n)
{
    LocalTensor<float> localA[BUFFER_NUM];
    LocalTensor<float> localX[BUFFER_NUM];
    LocalTensor<float> localY;
    LocalTensor<float> localAcc;
    
    TPipe pipe;
    
    uint32_t blockIdx = GetBlockIdx();
    uint32_t blockNum = GetBlockNum();
    
    uint32_t mPerBlock = (m + blockNum - 1) / blockNum;
    uint32_t mStart = blockIdx * mPerBlock;
    uint32_t mEnd = min(mStart + mPerBlock, m);
    uint32_t mTileNum = (mEnd - mStart + TILE_M - 1) / TILE_M;
    
    pipe.InitBuffer(localA[0], TILE_M * TILE_N * sizeof(float));
    pipe.InitBuffer(localA[1], TILE_M * TILE_N * sizeof(float));
    pipe.InitBuffer(localX[0], TILE_N * sizeof(float));
    pipe.InitBuffer(localX[1], TILE_N * sizeof(float));
    pipe.InitBuffer(localY, TILE_M * sizeof(float));
    pipe.InitBuffer(localAcc, TILE_M * sizeof(float));
    
    for (uint32_t mTileIdx = 0; mTileIdx < mTileNum; ++mTileIdx) {
        uint32_t mCurrent = mStart + mTileIdx * TILE_M;
        uint32_t mValid = min(TILE_M, mEnd - mCurrent);
        
        uint64_t yOffset = mCurrent * sizeof(float);
        
        DataCopy(localY, y + yOffset, mValid);
        
        for (uint32_t i = 0; i < TILE_M; ++i) {
            localAcc.SetValue(i, 0.0f);
        }
        
        uint32_t nTileNum = (n + TILE_N - 1) / TILE_N;
        uint32_t pipeIdx = 0;
        
        for (uint32_t nTileIdx = 0; nTileIdx < nTileNum; ++nTileIdx) {
            uint32_t nCurrent = nTileIdx * TILE_N;
            uint32_t nValid = min(TILE_N, n - nCurrent);
            
            uint64_t aOffset = (mCurrent * n + nCurrent) * sizeof(float);
            uint64_t xOffset = nCurrent * sizeof(float);
            
            pipe.Wait();
            
            DataCopy(localA[pipeIdx], A + aOffset, mValid, nValid, TILE_N, n - nValid);
            DataCopy(localX[pipeIdx], x + xOffset, nValid);
            
            if (nTileIdx > 0) {
                uint32_t prevPipeIdx = 1 - pipeIdx;
                
                for (uint32_t i = 0; i < mValid; ++i) {
                    float acc = localAcc.GetValue(i);
                    for (uint32_t j = 0; j < nValid; ++j) {
                        float aVal = localA[prevPipeIdx].GetValue(i * TILE_N + j);
                        float xVal = localX[prevPipeIdx].GetValue(j);
                        acc += aVal * xVal;
                    }
                    localAcc.SetValue(i, acc);
                }
            }
            
            pipeIdx = 1 - pipeIdx;
        }
        
        pipe.Wait();
        
        uint32_t lastPipeIdx = 1 - pipeIdx;
        for (uint32_t i = 0; i < mValid; ++i) {
            float acc = localAcc.GetValue(i);
            for (uint32_t j = 0; j < min(TILE_N, n - (nTileNum - 1) * TILE_N); ++j) {
                float aVal = localA[lastPipeIdx].GetValue(i * TILE_N + j);
                float xVal = localX[lastPipeIdx].GetValue(j);
                acc += aVal * xVal;
            }
            localAcc.SetValue(i, acc);
        }
        
        for (uint32_t i = 0; i < mValid; ++i) {
            float acc = localAcc.GetValue(i);
            float yVal = localY.GetValue(i);
            float result = alpha * acc + beta * yVal;
            localY.SetValue(i, result);
        }
        
        DataCopy(y + yOffset, localY, mValid);
    }
}