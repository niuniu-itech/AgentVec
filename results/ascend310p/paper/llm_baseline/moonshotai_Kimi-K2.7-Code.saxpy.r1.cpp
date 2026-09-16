#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t TILE_LEN = 8192; // elements, 32-byte aligned count for fp32

extern "C" __global__ __aicore__ void saxpy(GM_ADDR x, GM_ADDR y, GM_ADDR out, float alpha, uint32_t total) {
    GlobalTensor<float> xGm;
    GlobalTensor<float> yGm;
    GlobalTensor<float> outGm;
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));
    outGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(out));

    uint32_t blockNum = GetBlockNum();
    uint32_t blockIdx = GetBlockIdx();
    uint64_t blockStart64 = (uint64_t)blockIdx * (uint64_t)total / (uint64_t)blockNum;
    uint64_t blockEnd64   = (uint64_t)(blockIdx + 1) * (uint64_t)total / (uint64_t)blockNum;
    uint32_t blockStart = (uint32_t)blockStart64;
    uint32_t blockEnd   = (uint32_t)blockEnd64;
    if (blockStart >= blockEnd) {
        return;
    }

    TPipe pipe;
    TQue<TPosition::VECIN, 2> qx;
    TQue<TPosition::VECIN, 2> qy;
    TQue<TPosition::VECOUT, 2> qout;
    TBuf<TPosition::VECCALC> tmpBuf;
    pipe.InitBuffer(qx,    TILE_LEN * sizeof(float));
    pipe.InitBuffer(qy,    TILE_LEN * sizeof(float));
    pipe.InitBuffer(qout,  TILE_LEN * sizeof(float));
    pipe.InitBuffer(tmpBuf, TILE_LEN * sizeof(float));

    uint32_t offset = blockStart;

    // Scalar prologue to reach 32-byte alignment
    uint32_t prefix = ((8U - (offset & 7U)) & 7U);
    if (prefix > blockEnd - offset) {
        prefix = blockEnd - offset;
    }
    for (uint32_t i = 0; i < prefix; ++i) {
        uint32_t idx = offset + i;
        outGm.SetValue(idx, alpha * xGm.GetValue(idx) + yGm.GetValue(idx));
    }
    offset += prefix;

    uint32_t remaining  = blockEnd - offset;
    uint32_t tileCount  = remaining / TILE_LEN;
    uint32_t tailCount  = remaining % TILE_LEN;

    // Double-buffered pipeline: load / compute / store
    for (int32_t stage = 0; stage < (int32_t)tileCount + 2; ++stage) {
        int32_t loadIdx  = stage;
        int32_t compIdx  = stage - 1;
        int32_t storeIdx = stage - 2;

        if (loadIdx >= 0 && loadIdx < (int32_t)tileCount) {
            uint32_t curOffset = offset + (uint32_t)loadIdx * TILE_LEN;
            LocalTensor<float> xLocal = qx.AllocTensor<float>();
            LocalTensor<float> yLocal = qy.AllocTensor<float>();
            DataCopy(xLocal, xGm[curOffset], TILE_LEN);
            DataCopy(yLocal, yGm[curOffset], TILE_LEN);
            qx.EnQue(xLocal);
            qy.EnQue(yLocal);
        }

        if (compIdx >= 0 && compIdx < (int32_t)tileCount) {
            uint32_t curOffset = offset + (uint32_t)compIdx * TILE_LEN;
            LocalTensor<float> xLocal = qx.DeQue<float>();
            LocalTensor<float> yLocal = qy.DeQue<float>();
            LocalTensor<float> outLocal = qout.AllocTensor<float>();
            LocalTensor<float> tmp = tmpBuf.Get<float>();
            Muls(tmp, xLocal, alpha, TILE_LEN);
            Add(outLocal, tmp, yLocal, TILE_LEN);
            qout.EnQue(outLocal);
            qx.FreeTensor(xLocal);
            qy.FreeTensor(yLocal);
        }

        if (storeIdx >= 0 && storeIdx < (int32_t)tileCount) {
            uint32_t curOffset = offset + (uint32_t)storeIdx * TILE_LEN;
            LocalTensor<float> outLocal = qout.DeQue<float>();
            DataCopy(outGm[curOffset], outLocal, TILE_LEN);
            qout.FreeTensor(outLocal);
        }
    }

    // Vector tail (32-byte aligned portion)
    uint32_t tailOffset = offset + tileCount * TILE_LEN;
    uint32_t vecTail = tailCount & ~7U;
    if (vecTail > 0) {
        LocalTensor<float> xLocal = qx.AllocTensor<float>();
        LocalTensor<float> yLocal = qy.AllocTensor<float>();
        DataCopy(xLocal, xGm[tailOffset], vecTail);
        DataCopy(yLocal, yGm[tailOffset], vecTail);
        LocalTensor<float> outLocal = qout.AllocTensor<float>();
        LocalTensor<float> tmp = tmpBuf.Get<float>();
        Muls(tmp, xLocal, alpha, static_cast<int32_t>(vecTail));
        Add(outLocal, tmp, yLocal, static_cast<int32_t>(vecTail));
        DataCopy(outGm[tailOffset], outLocal, vecTail);
        qx.FreeTensor(xLocal);
        qy.FreeTensor(yLocal);
        qout.FreeTensor(outLocal);
    }

    // Scalar remainder (at most 7 elements)
    uint32_t scalarTail = tailCount - vecTail;
    for (uint32_t i = 0; i < scalarTail; ++i) {
        uint32_t idx = tailOffset + vecTail + i;
        outGm.SetValue(idx, alpha * xGm.GetValue(idx) + yGm.GetValue(idx));
    }
}