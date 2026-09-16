#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BLOCK_SIZE = 8;
constexpr int32_t MTILE = 256;
constexpr int32_t NTILE = 256;
constexpr int32_t KTILE = 64;
constexpr int32_t M_TILES_PER_CORE = 4;
constexpr int32_t N_TILES_PER_CORE = 4;
constexpr int32_t K_TILES_PER_CORE = 2;
constexpr int32_t BUFFER_NUM = 2;

extern "C" __global__ __aicore__ void sgemm(GM_ADDR A, GM_ADDR B, GM_ADDR C, float alpha, float beta, uint32_t m, uint32_t n, uint32_t k) {
    LocalTensor<float> localA[BUFFER_NUM];
    LocalTensor<float> localB[BUFFER_NUM];
    LocalTensor<float> localC;
    LocalTensor<float> accum[M_TILES_PER_CORE][N_TILES_PER_CORE];
    
    TPipe pipe;
    TBuf<TPosition::VECCALC> commBuf;
    
    uint32_t blockIdx = GetBlockIdx();
    uint32_t blockNum = GetBlockNum();
    
    uint32_t mTiles = (m + MTILE - 1) / MTILE;
    uint32_t nTiles = (n + NTILE - 1) / NTILE;
    uint32_t kTiles = (k + KTILE - 1) / KTILE;
    
    uint32_t totalTiles = mTiles * nTiles;
    uint32_t tilesPerBlock = (totalTiles + blockNum - 1) / blockNum;
    uint32_t tileStart = blockIdx * tilesPerBlock;
    uint32_t tileEnd = min(tileStart + tilesPerBlock, totalTiles);
    
    for (int32_t i = 0; i < M_TILES_PER_CORE; ++i) {
        for (int32_t j = 0; j < N_TILES_PER_CORE; ++j) {
            accum[i][j] = commBuf.Get<float>();
        }
    }
    
    for (int32_t bufIdx = 0; bufIdx < BUFFER_NUM; ++bufIdx) {
        localA[bufIdx] = commBuf.Get<float>();
        localB[bufIdx] = commBuf.Get<float>();
    }
    
    localC = commBuf.Get<float>();
    
    for (uint32_t tileIdx = tileStart; tileIdx < tileEnd; ++tileIdx) {
        uint32_t mTileIdx = tileIdx / nTiles;
        uint32_t nTileIdx = tileIdx % nTiles;
        
        uint32_t mStart = mTileIdx * MTILE;
        uint32_t nStart = nTileIdx * NTILE;
        uint32_t mSize = min(MTILE, m - mStart);
        uint32_t nSize = min(NTILE, n - nStart);
        
        for (int32_t i = 0; i < M_TILES_PER_CORE; ++i) {
            for (int32_t j = 0; j < N_TILES_PER_CORE; ++j) {
                accum[i][j].Init(0.0f);
            }
        }
        
        pipe.InitBuffer(localA[0], MTILE * KTILE * sizeof(float));
        pipe.InitBuffer(localB[0], KTILE * NTILE * sizeof(float));
        pipe.InitBuffer(localA[1], MTILE * KTILE * sizeof(float));
        pipe.InitBuffer(localB[1], KTILE * NTILE * sizeof(float));
        pipe.InitBuffer(localC, MTILE * NTILE * sizeof(float));
        
        uint32_t copyA = 0;
        uint32_t copyB = 0;
        
        for (uint32_t kTileIdx = 0; kTileIdx < kTiles; ++kTileIdx) {
            uint32_t kStart = kTileIdx * KTILE;
            uint32_t kSize = min(KTILE, k - kStart);
            
            uint32_t bufIdx = kTileIdx % BUFFER_NUM;
            uint32_t nextBufIdx = (kTileIdx + 1) % BUFFER_NUM;
            
            if (kTileIdx < kTiles - 1) {
                pipe.InitBuffer(localA[nextBufIdx], MTILE * KTILE * sizeof(float));
                pipe.InitBuffer(localB[nextBufIdx], KTILE * NTILE * sizeof(float));
            }
            
            if (kTileIdx < kTiles) {
                CopyParams copyAParam = { .blockCount = mSize, .blockLen = kSize * sizeof(float), .srcStride = k * sizeof(float), .dstStride = KTILE * sizeof(float) };
                pipe.EnqueueCopy(copyA, A + (mStart * k + kStart) * sizeof(float), localA[bufIdx], copyAParam);
                
                CopyParams copyBParam = { .blockCount = kSize, .blockLen = nSize * sizeof(float), .srcStride = n * sizeof(float), .dstStride = NTILE * sizeof(float) };
                pipe.EnqueueCopy(copyB, B + (kStart * n + nStart) * sizeof(float), localB[bufIdx], copyBParam);
            }
            
            pipe.Wait(copyA);
            pipe.Wait(copyB);
            
            for (int32_t i = 0; i < M_TILES_PER_CORE; ++i) {
                for (int32_t j = 0; j < N_TILES_PER_CORE; ++j) {
                    uint32_t mSubStart = i * (MTILE / M_TILES_PER_CORE);
                    uint32_t nSubStart = j * (NTILE / N_TILES_PER_CORE);
                    uint32_t mSubSize = min(MTILE / M_TILES_PER_CORE, mSize - mSubStart);
                    uint32_t nSubSize = min(NTILE / N_TILES_PER_CORE, nSize - nSubStart);
                    
                    if (mSubSize > 0 && nSubSize > 0 && kSize > 0) {
                        for (uint32_t ki = 0; ki < kSize; ++ki) {
                            float* aPtr = localA[bufIdx].GetPtr(mSubStart * KTILE + ki);
                            float* bPtr = localB[bufIdx].GetPtr(ki * NTILE + nSubStart);
                            float* cPtr = accum[i][j].GetPtr(0);
                            
                            for (uint32_t mi = 0; mi < mSubSize; ++mi) {
                                float aVal = aPtr[mi * KTILE];
                                for (uint32_t ni = 0; ni < nSubSize; ++ni) {
                                    cPtr[mi * (NTILE / N_TILES_PER_CORE) + ni] += aVal * bPtr[ni];
                                }
                            }
                        }
                    }
                }
            }
        }
        
        pipe.InitBuffer(localC, MTILE * NTILE * sizeof(float));
        
        if (beta != 0.0f) {
            CopyParams copyCParam = { .blockCount = mSize, .blockLen = nSize * sizeof(float), .srcStride = n * sizeof(float), .dstStride = NTILE * sizeof(float) };
            pipe.EnqueueCopy(localC, C + (mStart * n + nStart) * sizeof(float), localC, copyCParam);
            pipe.Wait(localC);
        }
        
        for (int32_t i = 0; i < M_TILES_PER_CORE; ++i) {
            for (int32_t j = 0; j < N_TILES_PER_CORE; ++j) {
                uint32_t mSubStart = i * (MTILE / M_TILES_PER_CORE);
                uint32_t nSubStart = j * (NTILE / N_TILES_PER_CORE);
                uint32_t mSubSize = min(MTILE / M_TILES_PER_CORE, mSize - mSubStart);
                uint32_t nSubSize = min(NTILE / N_TILES_PER_CORE, nSize - nSubStart);
                
                if (mSubSize > 0 && nSubSize > 0) {
                    float* accumPtr = accum[i][j].GetPtr(0);
                    float* localCPtr = localC.GetPtr(mSubStart * NTILE + nSubStart);
                    
                    for (uint32_t mi = 0; mi < mSubSize; ++mi) {
                        for (uint32_t ni = 0; ni < nSubSize; ++ni) {
                            uint32_t idx = mi * (NTILE / N_TILES_PER_CORE) + ni;
                            float result = alpha * accumPtr[idx];
                            if (beta != 0.0f) {
                                result += beta * localCPtr[mi * NTILE + ni];
                            }
                            localCPtr[mi * NTILE + ni] = result;
                        }
                    }
                }
            }
        }
        
        CopyParams copyOutParam = { .blockCount = mSize, .blockLen = nSize * sizeof(float), .srcStride = NTILE * sizeof(float), .dstStride = n * sizeof(float) };
        pipe.EnqueueCopy(C + (mStart * n + nStart) * sizeof(float), localC, copyOutParam);
        pipe.WaitAll();
    }
}