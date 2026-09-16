#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void sdot(GM_ADDR x, GM_ADDR y, GM_ADDR partials, uint32_t total) {
    uint32_t blockNum = GetBlockNum();
    uint32_t blockIdx = GetBlockIdx();

    uint32_t base = total / blockNum;
    uint32_t rem  = total % blockNum;
    uint32_t blockCount = base + (blockIdx < rem ? 1 : 0);
    uint32_t blockOffset = blockIdx * base + (blockIdx < rem ? blockIdx : rem);

    GlobalTensor<float> xGm, yGm, partialsGm;
    xGm.SetGlobalBuffer((__gm__ float*)x + blockOffset);
    yGm.SetGlobalBuffer((__gm__ float*)y + blockOffset);
    partialsGm.SetGlobalBuffer((__gm__ float*)partials);

    TPipe pipe;
    TQue<QuePosition::VECIN, 2> xQueue;
    TQue<QuePosition::VECIN, 2> yQueue;
    TBuf<QuePosition::VECCALC> workBuf;
    TBuf<QuePosition::VECCALC> resBuf;

    constexpr uint32_t MAX_TILE_FLOATS = 8192; // 32 KB per tile
    uint32_t vecCount = blockCount & ~7U;
    uint32_t tileFloats = (vecCount > MAX_TILE_FLOATS) ? MAX_TILE_FLOATS : vecCount;

    pipe.InitBuffer(resBuf, 8 * sizeof(float));
    if (tileFloats > 0) {
        pipe.InitBuffer(xQueue, 2, tileFloats * sizeof(float));
        pipe.InitBuffer(yQueue, 2, tileFloats * sizeof(float));
        pipe.InitBuffer(workBuf, tileFloats * sizeof(float));
    }

    LocalTensor<float> outLocal = resBuf.Get<float>();
    Duplicate(outLocal, 0.0f, 8);

    if (blockCount == 0) {
        DataCopy(partialsGm[blockIdx * 8], outLocal, 8);
        return;
    }

    uint32_t scalarCount = blockCount - vecCount;
    float blockSum = 0.0f;

    if (vecCount > 0) {
        uint32_t tile = tileFloats;
        uint32_t fullTiles = vecCount / tile;
        uint32_t fullCount = fullTiles * tile;
        uint32_t remainderVec = vecCount - fullCount;

        // Pre-load the first tile to start the double-buffer pipeline.
        LocalTensor<float> xCur = xQueue.AllocTensor<float>();
        LocalTensor<float> yCur = yQueue.AllocTensor<float>();
        DataCopy(xCur, xGm[0], tile);
        DataCopy(yCur, yGm[0], tile);
        xQueue.EnQue(xCur);
        yQueue.EnQue(yCur);

        for (uint32_t i = 0; i < fullTiles; ++i) {
            xCur = xQueue.DeQue<float>();
            yCur = yQueue.DeQue<float>();

            // Load next tile on MTE while the vector unit works on the current one.
            if (i + 1 < fullTiles) {
                LocalTensor<float> xNext = xQueue.AllocTensor<float>();
                LocalTensor<float> yNext = yQueue.AllocTensor<float>();
                DataCopy(xNext, xGm[(i + 1) * tile], tile);
                DataCopy(yNext, yGm[(i + 1) * tile], tile);
                xQueue.EnQue(xNext);
                yQueue.EnQue(yNext);
            }

            Mul(xCur, xCur, yCur, tile);
            LocalTensor<float> workLocal = workBuf.Get<float>();
            ReduceSum(outLocal, xCur, workLocal, tile);
            PipeBarrier<PIPE_V>();
            blockSum += outLocal.GetValue(0);

            xQueue.FreeTensor(xCur);
            yQueue.FreeTensor(yCur);
        }

        if (remainderVec > 0) {
            LocalTensor<float> xLocal = xQueue.AllocTensor<float>();
            LocalTensor<float> yLocal = yQueue.AllocTensor<float>();
            DataCopy(xLocal, xGm[fullCount], remainderVec);
            DataCopy(yLocal, yGm[fullCount], remainderVec);
            xQueue.EnQue(xLocal);
            yQueue.EnQue(yLocal);

            xLocal = xQueue.DeQue<float>();
            yLocal = yQueue.DeQue<float>();
            Mul(xLocal, xLocal, yLocal, remainderVec);
            LocalTensor<float> workLocal = workBuf.Get<float>();
            ReduceSum(outLocal, xLocal, workLocal, remainderVec);
            PipeBarrier<PIPE_V>();
            blockSum += outLocal.GetValue(0);

            xQueue.FreeTensor(xLocal);
            yQueue.FreeTensor(yLocal);
        }
    }

    // Scalar tail (at most 7 elements).
    for (uint32_t i = 0; i < scalarCount; ++i) {
        blockSum += xGm.GetValue(vecCount + i) * yGm.GetValue(vecCount + i);
    }

    outLocal.SetValue(0, blockSum);
    PipeBarrier<PIPE_ALL>();
    DataCopy(partialsGm[blockIdx * 8], outLocal, 8);
}