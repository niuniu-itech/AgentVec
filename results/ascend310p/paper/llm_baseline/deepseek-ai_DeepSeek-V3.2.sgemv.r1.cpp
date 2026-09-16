#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t TILE_M = 256;
constexpr int32_t TILE_N = 256;
constexpr int32_t VEC_LEN = 256;

extern "C" __global__ __aicore__ void sgemv(GM_ADDR A, GM_ADDR x, GM_ADDR y, float alpha, float beta, uint32_t m, uint32_t n)
{
    LocalTensor<float> localA[BUFFER_NUM];
    LocalTensor<float> localX[BUFFER_NUM];
    LocalTensor<float> localY;
    LocalTensor<float> localAcc;
    
    TPipe pipe;
    TBuf<TPosition::VECCALC> vecCalcBuf;
    
    uint32_t blockIdx = GetBlockIdx();
    uint32_t blockNum = GetBlockNum();
    
    uint32_t mPerBlock = (m + blockNum - 1) / blockNum;
    uint32_t mStart = blockIdx * mPerBlock;
    uint32_t mEnd = min(mStart + mPerBlock, m);
    uint32_t mTileNum = (mEnd - mStart + TILE_M - 1) / TILE_M;
    
    uint32_t nPerBlock = (n + blockNum - 1) / blockNum;
    uint32_t nStart = blockIdx * nPerBlock;
    uint32_t nEnd = min(nStart + nPerBlock, n);
    uint32_t nTileNum = (nEnd - nStart + TILE_N - 1) / TILE_N;
    
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
        DataCopyParams yCopyParams;
        yCopyParams.blockCount = mValid;
        yCopyParams.blockSize = sizeof(float);
        yCopyParams.dstStride = 0;
        yCopyParams.srcStride = 0;
        
        DataCopy(localY, y, yCopyParams, yOffset, 0);
        
        vecCalcBuf.Clean();
        for (uint32_t i = 0; i < TILE_M; ++i) {
            localAcc.SetValue(i, 0.0f);
        }
        
        uint32_t pipeIdx = 0;
        for (uint32_t nTileIdx = 0; nTileIdx < nTileNum; ++nTileIdx) {
            uint32_t nCurrent = nStart + nTileIdx * TILE_N;
            uint32_t nValid = min(TILE_N, nEnd - nCurrent);
            
            uint64_t aOffset = (mCurrent * n + nCurrent) * sizeof(float);
            uint64_t xOffset = nCurrent * sizeof(float);
            
            DataCopyParams aCopyParams;
            aCopyParams.blockCount = mValid * nValid;
            aCopyParams.blockSize = sizeof(float);
            aCopyParams.dstStride = (TILE_N - nValid) * sizeof(float);
            aCopyParams.srcStride = (n - nValid) * sizeof(float);
            
            DataCopyParams xCopyParams;
            xCopyParams.blockCount = nValid;
            xCopyParams.blockSize = sizeof(float);
            xCopyParams.dstStride = 0;
            xCopyParams.srcStride = 0;
            
            pipe.WaitAll();
            
            DataCopy(localA[pipeIdx], A, aCopyParams, aOffset, 0);
            DataCopy(localX[pipeIdx], x, xCopyParams, xOffset, 0);
            
            if (nTileIdx > 0) {
                uint32_t prevPipeIdx = 1 - pipeIdx;
                
                for (uint32_t i = 0; i < mValid; i += VEC_LEN) {
                    uint32_t iValid = min(VEC_LEN, mValid - i);
                    LocalTensor<float> aVec = localA[prevPipeIdx][i * TILE_N];
                    LocalTensor<float> xVec = localX[prevPipeIdx];
                    LocalTensor<float> accVec = localAcc[i];
                    
                    vecCalcBuf.CalcVector(accVec, aVec, xVec, iValid, nValid);
                }
            }
            
            pipeIdx = 1 - pipeIdx;
        }
        
        pipe.WaitAll();
        
        uint32_t lastPipeIdx = 1 - pipeIdx;
        for (uint32_t i = 0; i < mValid; i += VEC_LEN) {
            uint32_t iValid = min(VEC_LEN, mValid - i);
            LocalTensor<float> aVec = localA[lastPipeIdx][i * TILE_N];
            LocalTensor<float> xVec = localX[lastPipeIdx];
            LocalTensor<float> accVec = localAcc[i];
            
            vecCalcBuf.CalcVector(accVec, aVec, xVec, iValid, nValid);
        }
        
        vecCalcBuf.Wait();
        
        for (uint32_t i = 0; i < mValid; ++i) {
            float acc = localAcc.GetValue(i);
            float yVal = localY.GetValue(i);
            float result = alpha * acc + beta * yVal;
            localY.SetValue(i, result);
        }
        
        DataCopyParams yOutParams;
        yOutParams.blockCount = mValid;
        yOutParams.blockSize = sizeof(float);
        yOutParams.dstStride = 0;
        yOutParams.srcStride = 0;
        
        DataCopy(y, localY, yOutParams, yOffset, 0);
    }
}