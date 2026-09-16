#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFNUM   = 2;
constexpr int32_t TILELEN  = 1024;   // must be a multiple of 8
constexpr int32_t REDLANE  = 64;     // reduction width for ReduceSum
constexpr int32_t PARTLEN  = 8;      // 32 B slot per block

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

        numChunks = (total + TILELEN - 1) / TILELEN;

        pipe.InitBuffer(qx,    BUFNUM, TILELEN * sizeof(float));
        pipe.InitBuffer(qy,    BUFNUM, TILELEN * sizeof(float));
        pipe.InitBuffer(accBuf, 1,     REDLANE * sizeof(float));
        pipe.InitBuffer(workBuf, 1,    REDLANE * sizeof(float));
        pipe.InitBuffer(partBuf, 1,    PARTLEN * sizeof(float));
        pipe.InitBuffer(outBuf, 1,     PARTLEN * sizeof(float));
    }

    __aicore__ inline void Process() {
        LocalTensor<float> acc  = accBuf.Get<float>();
        LocalTensor<float> work = workBuf.Get<float>();
        LocalTensor<float> part = partBuf.Get<float>();
        Duplicate(acc, 0.0f, REDLANE);

        uint32_t c = blockIdx;
        bool hasCur = false;
        uint32_t curC = 0;

        // Double-buffered round-robin over aligned TILELEN chunks.
        while (c < numChunks || hasCur) {
            if (c < numChunks) {
                LocalTensor<float> xn = qx.AllocTensor<float>();
                LocalTensor<float> yn = qy.AllocTensor<float>();
                uint32_t start = c * TILELEN;
                uint32_t sz    = (start + TILELEN <= total) ? TILELEN : (total - start);

                DataCopy(xn, xg[start], sz);
                DataCopy(yn, yg[start], sz);
                qx.EnQue(xn);
                qy.EnQue(yn);

                if (!hasCur) {
                    curC   = c;
                    hasCur = true;
                    c     += blockNum;
                    continue;
                }
            }

            // Compute the tile that was prefetched earlier.
            LocalTensor<float> xl = qx.DeQue<float>();
            LocalTensor<float> yl = qy.DeQue<float>();
            uint32_t start = curC * TILELEN;
            uint32_t sz    = (start + TILELEN <= total) ? TILELEN : (total - start);

            Mul(xl, xl, yl, sz);          // xl <- xl * yl
            Accumulate(acc, xl, sz);      // fold into 64-lane accumulator

            qx.FreeTensor(xl);
            qy.FreeTensor(yl);

            if (c < numChunks) {
                curC = c;
                c   += blockNum;
            } else {
                hasCur = false;
            }
        }

        // Final reduction of the 64-lane accumulator to one scalar.
        Duplicate(part, 0.0f, PARTLEN);
        ReduceSum<float>(part, acc, work, REDLANE);

        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        float sum = part.GetValue(0);

        // Write the block partial to a 32B-aligned slot.
        LocalTensor<float> out = outBuf.Get<float>();
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        Duplicate(out, sum, PARTLEN);

        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(pg[blockIdx * PARTLEN], out, PARTLEN);
    }

private:
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
    TBuf<QuePosition::VECCALC> accBuf, workBuf, partBuf, outBuf;
    GlobalTensor<float> xg, yg, pg;
    uint32_t total, numChunks, blockNum, blockIdx;
};

extern "C" __global__ __aicore__ void sdot(GM_ADDR x, GM_ADDR y, GM_ADDR partials, uint32_t total) {
    KSDot op;
    op.Init(x, y, partials, total);
    op.Process();
}