#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFNUM  = 2;
constexpr int32_t TILELEN = 1024;   // must be a multiple of 8
constexpr int32_t REDLANE = 64;     // ReduceSum lane width
constexpr int32_t PARTLEN = 8;      // 32 B aligned partial slot

class KSDot {
public:
    __aicore__ inline KSDot() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR partials, uint32_t total) {
        blockNum = GetBlockNum();
        blockIdx = GetBlockIdx();
        this->total = total;

        xg.SetGlobalBuffer((__gm__ float*)x, total);
        yg.SetGlobalBuffer((__gm__ float*)y, total);
        pg.SetGlobalBuffer((__gm__ float*)partials, blockNum * PARTLEN);

        // Equal contiguous slices.  Caller must ensure total is a multiple of
        // blockNum*8 so every DataCopy is 32-byte aligned.
        blockLen   = total / blockNum;
        blockStart = blockIdx * blockLen;
        numTile    = (blockLen + TILELEN - 1) / TILELEN;

        pipe.InitBuffer(qx, BUFNUM, TILELEN * sizeof(float));
        pipe.InitBuffer(qy, BUFNUM, TILELEN * sizeof(float));
        pipe.InitBuffer(accBuf,  REDLANE * sizeof(float));
        pipe.InitBuffer(workBuf, REDLANE * sizeof(float));
        pipe.InitBuffer(partBuf, PARTLEN * sizeof(float));
    }

    __aicore__ inline void Process() {
        LocalTensor<float> acc  = accBuf.Get<float>();
        LocalTensor<float> work = workBuf.Get<float>();
        LocalTensor<float> part = partBuf.Get<float>();

        Duplicate(acc, 0.0f, REDLANE);

        // Prologue: prefetch tile 0
        LocalTensor<float> xCur = qx.AllocTensor<float>();
        LocalTensor<float> yCur = qy.AllocTensor<float>();
        uint32_t sz0 = TileSize(0);
        DataCopy(xCur, xg[blockStart], sz0);
        DataCopy(yCur, yg[blockStart], sz0);
        qx.EnQue(xCur);
        qy.EnQue(yCur);

        for (uint32_t t = 0; t < numTile; ++t) {
            // Prefetch next tile while the current one is being reduced.
            if (t + 1 < numTile) {
                LocalTensor<float> xNext = qx.AllocTensor<float>();
                LocalTensor<float> yNext = qy.AllocTensor<float>();
                uint32_t nextStart = blockStart + (t + 1) * TILELEN;
                uint32_t nextSz    = TileSize(t + 1);
                DataCopy(xNext, xg[nextStart], nextSz);
                DataCopy(yNext, yg[nextStart], nextSz);
                qx.EnQue(xNext);
                qy.EnQue(yNext);
            }

            xCur = qx.DeQue<float>();
            yCur = qy.DeQue<float>();

            uint32_t curSz = TileSize(t);

            Mul(xCur, xCur, yCur, curSz);
            Accumulate(acc, xCur, curSz);

            qx.FreeTensor(xCur);
            qy.FreeTensor(yCur);
        }

        Duplicate(part, 0.0f, PARTLEN);
        ReduceSum<float>(part, acc, work, REDLANE);

        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(pg[blockIdx * PARTLEN], part, PARTLEN);
    }

private:
    __aicore__ inline uint32_t TileSize(uint32_t tileIdx) const {
        uint32_t start = blockStart + tileIdx * TILELEN;
        uint32_t end   = start + TILELEN;
        return (end <= blockStart + blockLen) ? TILELEN : (blockStart + blockLen - start);
    }

    __aicore__ inline void Accumulate(LocalTensor<float>& acc,
                                      LocalTensor<float>& src,
                                      uint32_t n) {
        uint32_t offset = 0;
        for (; offset + REDLANE <= n; offset += REDLANE) {
            Add(acc, acc, src[offset], REDLANE);
        }
        uint32_t rem = n - offset;
        if (rem > 0) {
            Add(acc, acc, src[offset], rem);
        }
    }

    TPipe pipe;
    TQue<QuePosition::VECIN, BUFNUM> qx, qy;
    TBuf<QuePosition::VECCALC> accBuf, workBuf, partBuf;
    GlobalTensor<float> xg, yg, pg;
    uint32_t total, blockLen, blockStart, numTile, blockNum, blockIdx;
};

extern "C" __global__ __aicore__ void sdot(GM_ADDR x, GM_ADDR y, GM_ADDR partials, uint32_t total) {
    KSDot op;
    op.Init(x, y, partials, total);
    op.Process();
}