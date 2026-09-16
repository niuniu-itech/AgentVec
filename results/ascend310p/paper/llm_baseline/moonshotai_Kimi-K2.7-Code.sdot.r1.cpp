#include "kernel_operator.h"
using namespace AscendC;

class KernelSdot {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR partials, uint32_t total) {
        uint32_t blockNum = GetBlockNum();
        uint32_t blockIdx = GetBlockIdx();

        uint32_t base = total / blockNum;
        uint32_t rem  = total % blockNum;
        blockCount_ = base + (blockIdx < rem ? 1 : 0);
        uint32_t blockOffset = blockIdx * base + (blockIdx < rem ? blockIdx : rem);

        xGm.SetGlobalBuffer((__gm__ float*)x + blockOffset);
        yGm.SetGlobalBuffer((__gm__ float*)y + blockOffset);
        partialsGm.SetGlobalBuffer((__gm__ float*)partials);

        constexpr uint32_t MAX_TILE_FLOATS = 8192; // 32 KB per buffer
        uint32_t vecCount = blockCount_ & ~7U;
        tileFloats_ = (vecCount > MAX_TILE_FLOATS) ? MAX_TILE_FLOATS : vecCount;

        pipe.InitBuffer(resBuf_, 8 * sizeof(float));
        if (tileFloats_ > 0) {
            pipe.InitBuffer(xQueue_, 2, tileFloats_ * sizeof(float));
            pipe.InitBuffer(yQueue_, 2, tileFloats_ * sizeof(float));
            pipe.InitBuffer(workBuf_, tileFloats_ * sizeof(float));
        }
    }

    __aicore__ inline void Process() {
        uint32_t idx = GetBlockIdx();
        LocalTensor<float> outLocal = resBuf_.Get<float>();
        Duplicate(outLocal, 0.0f, 8);

        if (blockCount_ == 0) {
            DataCopy(partialsGm[idx * 8], outLocal, 8);
            return;
        }

        uint32_t vecCount    = blockCount_ & ~7U;
        uint32_t scalarCount = blockCount_ - vecCount;
        float blockSum = 0.0f;

        if (vecCount > 0) {
            uint32_t tile         = tileFloats_;
            uint32_t fullTiles    = vecCount / tile;
            uint32_t fullCount    = fullTiles * tile;
            uint32_t remainderVec = vecCount - fullCount;

            // Pre-load the first tile to start the double-buffer pipeline.
            LocalTensor<float> xCur = xQueue_.AllocTensor<float>();
            LocalTensor<float> yCur = yQueue_.AllocTensor<float>();
            DataCopy(xCur, xGm[0], tile);
            DataCopy(yCur, yGm[0], tile);
            xQueue_.EnQue(xCur);
            yQueue_.EnQue(yCur);

            for (uint32_t i = 0; i < fullTiles; ++i) {
                xCur = xQueue_.DeQue<float>();
                yCur = yQueue_.DeQue<float>();

                // Load next tile on MTE while the vector unit works on the current one.
                if (i + 1 < fullTiles) {
                    LocalTensor<float> xNext = xQueue_.AllocTensor<float>();
                    LocalTensor<float> yNext = yQueue_.AllocTensor<float>();
                    DataCopy(xNext, xGm[(i + 1) * tile], tile);
                    DataCopy(yNext, yGm[(i + 1) * tile], tile);
                    xQueue_.EnQue(xNext);
                    yQueue_.EnQue(yNext);
                }

                Mul(xCur, xCur, yCur, tile);
                LocalTensor<float> workLocal = workBuf_.Get<float>();
                ReduceSum(outLocal, xCur, workLocal, tile);
                PipeBarrier<PIPE_V>();
                blockSum += outLocal.GetValue(0);

                xQueue_.FreeTensor(xCur);
                yQueue_.FreeTensor(yCur);
            }

            if (remainderVec > 0) {
                LocalTensor<float> xLocal = xQueue_.AllocTensor<float>();
                LocalTensor<float> yLocal = yQueue_.AllocTensor<float>();
                DataCopy(xLocal, xGm[fullCount], remainderVec);
                DataCopy(yLocal, yGm[fullCount], remainderVec);
                xQueue_.EnQue(xLocal);
                yQueue_.EnQue(yLocal);

                xLocal = xQueue_.DeQue<float>();
                yLocal = yQueue_.DeQue<float>();
                Mul(xLocal, xLocal, yLocal, remainderVec);
                LocalTensor<float> workLocal = workBuf_.Get<float>();
                ReduceSum(outLocal, xLocal, workLocal, remainderVec);
                PipeBarrier<PIPE_V>();
                blockSum += outLocal.GetValue(0);

                xQueue_.FreeTensor(xLocal);
                yQueue_.FreeTensor(yLocal);
            }
        }

        // Scalar tail (at most 7 elements).
        for (uint32_t i = 0; i < scalarCount; ++i) {
            blockSum += xGm.GetValue(vecCount + i) * yGm.GetValue(vecCount + i);
        }

        outLocal.SetValue(0, blockSum);
        PipeBarrier<PIPE_ALL>();
        DataCopy(partialsGm[idx * 8], outLocal, 8);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 2> xQueue_;
    TQue<QuePosition::VECIN, 2> yQueue_;
    TBuf<QuePosition::VECCALC> workBuf_;
    TBuf<QuePosition::VECCALC> resBuf_;

    GlobalTensor<float> xGm;
    GlobalTensor<float> yGm;
    GlobalTensor<float> partialsGm;

    uint32_t blockCount_;
    uint32_t tileFloats_;
};

extern "C" __global__ __aicore__ void sdot(GM_ADDR x, GM_ADDR y, GM_ADDR partials, uint32_t total) {
    KernelSdot op;
    op.Init(x, y, partials, total);
    op.Process();
}