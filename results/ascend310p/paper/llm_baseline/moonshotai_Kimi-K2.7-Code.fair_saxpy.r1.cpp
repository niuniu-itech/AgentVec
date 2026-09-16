#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFNUM = 2;
constexpr int32_t TILELEN = 1024;

class Saxpy {
public:
    __aicore__ inline Saxpy() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR out,
                                float alpha, uint32_t total) {
        alpha_ = alpha;
        total_ = total;

        const uint32_t blockNum = GetBlockNum();
        const uint32_t idx      = GetBlockIdx();

        // Split work in multiples of 8 floats to keep GM offsets aligned.
        const uint32_t base      = (total / blockNum) & ~7u;
        const uint32_t rem       = total - base * blockNum;
        const uint32_t extra8    = rem & ~7u;
        const uint32_t leftover  = rem & 7u;
        const uint32_t extraBlocks = extra8 >> 3;

        blockStart_ = idx * base + (idx < extraBlocks ? idx * 8u : extra8);
        blockLen_   = base + (idx < extraBlocks ? 8u : 0u);

        // The remaining <8 elements are handled by block 0 as a tail.
        hasTail_     = (leftover > 0) && (idx == 0);
        tailStart_   = total - leftover;
        tailLeftover_ = leftover;

        xg_.SetGlobalBuffer((__gm__ float*)x, total);
        yg_.SetGlobalBuffer((__gm__ float*)y, total);
        zg_.SetGlobalBuffer((__gm__ float*)out, total);

        pipe_.InitBuffer(qx_, BUFNUM, TILELEN * sizeof(float));
        pipe_.InitBuffer(qy_, BUFNUM, TILELEN * sizeof(float));
        pipe_.InitBuffer(qz_, BUFNUM, TILELEN * sizeof(float));
    }

    __aicore__ inline void Process() {
        const uint32_t tileLen = static_cast<uint32_t>(TILELEN);
        const uint32_t numTile = blockLen_ / tileLen;
        const uint32_t rem     = blockLen_ % tileLen;

        if (numTile > 0) {
            ProcessFullTiles(numTile);
        }
        if (rem > 0) {
            ProcessRemainder(blockStart_ + numTile * tileLen, rem);
        }
        if (hasTail_) {
            ProcessTail();
        }
    }

private:
    __aicore__ inline void ProcessFullTiles(uint32_t count) {
        const uint32_t tileLen = static_cast<uint32_t>(TILELEN);

        // Prefetch first input tile(s).
        LocalTensor<float> x0 = qx_.AllocTensor<float>();
        LocalTensor<float> y0 = qy_.AllocTensor<float>();
        DataCopy(x0, xg_[blockStart_], tileLen);
        DataCopy(y0, yg_[blockStart_], tileLen);
        qx_.EnQue(x0);
        qy_.EnQue(y0);

        if (count > 1) {
            LocalTensor<float> x1 = qx_.AllocTensor<float>();
            LocalTensor<float> y1 = qy_.AllocTensor<float>();
            DataCopy(x1, xg_[blockStart_ + tileLen], tileLen);
            DataCopy(y1, yg_[blockStart_ + tileLen], tileLen);
            qx_.EnQue(x1);
            qy_.EnQue(y1);
        }

        // Double-buffered pipeline: V compute overlaps with MTE2/MTE3.
        for (uint32_t i = 0; i < count; ++i) {
            LocalTensor<float> xl = qx_.DeQue<float>();
            LocalTensor<float> yl = qy_.DeQue<float>();
            LocalTensor<float> zl = qz_.AllocTensor<float>();

            Muls(zl, xl, alpha_, tileLen);
            Add(zl, zl, yl, tileLen);
            qz_.EnQue(zl);

            qx_.FreeTensor(xl);
            qy_.FreeTensor(yl);

            // Prefetch input tile i+2.
            if (i + 2 < count) {
                LocalTensor<float> xn = qx_.AllocTensor<float>();
                LocalTensor<float> yn = qy_.AllocTensor<float>();
                DataCopy(xn, xg_[blockStart_ + (i + 2) * tileLen], tileLen);
                DataCopy(yn, yg_[blockStart_ + (i + 2) * tileLen], tileLen);
                qx_.EnQue(xn);
                qy_.EnQue(yn);
            }

            // Drain output tile i-1.
            if (i > 0) {
                LocalTensor<float> zo = qz_.DeQue<float>();
                DataCopy(zg_[blockStart_ + (i - 1) * tileLen], zo, tileLen);
                qz_.FreeTensor(zo);
            }
        }

        // Drain the final output tile.
        LocalTensor<float> zo = qz_.DeQue<float>();
        DataCopy(zg_[blockStart_ + (count - 1) * tileLen], zo, tileLen);
        qz_.FreeTensor(zo);
    }

    __aicore__ inline void ProcessRemainder(uint32_t offset, uint32_t len) {
        LocalTensor<float> xl = qx_.AllocTensor<float>();
        LocalTensor<float> yl = qy_.AllocTensor<float>();
        DataCopy(xl, xg_[offset], len);
        DataCopy(yl, yg_[offset], len);
        qx_.EnQue(xl);
        qy_.EnQue(yl);

        xl = qx_.DeQue<float>();
        yl = qy_.DeQue<float>();
        LocalTensor<float> zl = qz_.AllocTensor<float>();
        Muls(zl, xl, alpha_, len);
        Add(zl, zl, yl, len);
        qz_.EnQue(zl);

        qx_.FreeTensor(xl);
        qy_.FreeTensor(yl);

        zl = qz_.DeQue<float>();
        DataCopy(zg_[offset], zl, len);
        qz_.FreeTensor(zl);
    }

    __aicore__ inline void ProcessTail() {
        LocalTensor<float> xl = qx_.AllocTensor<float>();
        LocalTensor<float> yl = qy_.AllocTensor<float>();
        DataCopy(xl, xg_[tailStart_], 8);
        DataCopy(yl, yg_[tailStart_], 8);
        qx_.EnQue(xl);
        qy_.EnQue(yl);

        xl = qx_.DeQue<float>();
        yl = qy_.DeQue<float>();

        // Zero the out-of-bound x lanes so they contribute 0 to the result.
        for (uint32_t i = tailLeftover_; i < 8; ++i) {
            xl.SetValue(i, 0.0f);
        }
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);

        LocalTensor<float> zl = qz_.AllocTensor<float>();
        Muls(zl, xl, alpha_, 8);
        Add(zl, zl, yl, 8);
        qz_.EnQue(zl);

        qx_.FreeTensor(xl);
        qy_.FreeTensor(yl);

        zl = qz_.DeQue<float>();
        DataCopy(zg_[tailStart_], zl, 8);
        qz_.FreeTensor(zl);
    }

    TPipe pipe_;
    TQue<QuePosition::VECIN,  BUFNUM> qx_;
    TQue<QuePosition::VECIN,  BUFNUM> qy_;
    TQue<QuePosition::VECOUT, BUFNUM> qz_;

    GlobalTensor<float> xg_;
    GlobalTensor<float> yg_;
    GlobalTensor<float> zg_;

    float    alpha_;
    uint32_t total_;
    uint32_t blockStart_;
    uint32_t blockLen_;
    uint32_t tailStart_;
    uint32_t tailLeftover_;
    bool     hasTail_;
};

extern "C" __global__ __aicore__ void saxpy(GM_ADDR x, GM_ADDR y, GM_ADDR out,
                                            float alpha, uint32_t total) {
    Saxpy op;
    op.Init(x, y, out, alpha, total);
    op.Process();
}