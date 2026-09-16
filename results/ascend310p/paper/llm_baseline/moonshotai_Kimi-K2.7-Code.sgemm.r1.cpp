#include "kernel_operator.h"
using namespace AscendC;

constexpr uint32_t BN = 128;
constexpr uint32_t BK = 128;
constexpr uint32_t RM = 64;

__aicore__ inline uint32_t MinU(uint32_t a, uint32_t b) { return (a < b) ? a : b; }

__aicore__ inline void LoadA(LocalTensor<float>& dst, GlobalTensor<float>& src,
                             uint32_t i, uint32_t k0, uint32_t bk, uint32_t k) {
    DataCopy(dst, src + i * k + k0, bk);
}

__aicore__ inline void LoadBTile(LocalTensor<float>& dst, GlobalTensor<float>& src,
                                 uint32_t k0, uint32_t n0, uint32_t bk, uint32_t bn,
                                 uint32_t n) {
    for (uint32_t t = 0; t < bk; ++t) {
        DataCopy(dst[t * BN], src + (k0 + t) * n + n0, bn);
    }
}

__aicore__ inline void LoadCRow(LocalTensor<float>& dst, GlobalTensor<float>& src,
                                uint32_t i, uint32_t n0, uint32_t bn, uint32_t n) {
    DataCopy(dst, src + i * n + n0, bn);
}

__aicore__ inline void StoreCRow(GlobalTensor<float>& dst, LocalTensor<float>& src,
                                 uint32_t i, uint32_t n0, uint32_t bn, uint32_t n) {
    DataCopy(dst + i * n + n0, src, bn);
}

__aicore__ inline void AccumulateRow(LocalTensor<float>& cRow, LocalTensor<float>& aRow,
                                     LocalTensor<float>& bTile, LocalTensor<float>& tmp,
                                     uint32_t bk, uint32_t bn) {
    for (uint32_t t = 0; t < bk; ++t) {
        float a = aRow.GetValue(t);
        Muls(tmp, bTile[t * BN], a, bn);
        Add(cRow, cRow, tmp, bn);
    }
}

extern "C" __global__ __aicore__ void sgemm(GM_ADDR A, GM_ADDR B, GM_ADDR C,
                                            float alpha, float beta,
                                            uint32_t m, uint32_t n, uint32_t k) {
    GlobalTensor<float> gA, gB, gC;
    gA.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(A));
    gB.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(B));
    gC.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(C));

    if (m == 0 || n == 0) return;

    uint32_t blockIdx = GetBlockIdx();
    uint32_t blockNum = GetBlockNum();
    uint32_t rowsPerBlock = (m + blockNum - 1) / blockNum;
    uint32_t rowStart = blockIdx * rowsPerBlock;
    uint32_t rowEnd = MinU(rowStart + rowsPerBlock, m);
    if (rowStart >= m) return;

    TPipe pipe;
    TQue<QuePosition::A1, 2> qB;
    TQue<QuePosition::B1, 2> qA;
    TBuf<QuePosition::VECCALC> calcBuf;
    pipe.InitBuffer(qB, 2, BK * BN * sizeof(float));
    pipe.InitBuffer(qA, 2, BK * sizeof(float));
    pipe.InitBuffer(calcBuf, (RM * BN + BN) * sizeof(float));

    LocalTensor<float> cTile = calcBuf.Get<float>(0);
    LocalTensor<float> tmp  = calcBuf.Get<float>(RM * BN);

    if (k == 0) {
        for (uint32_t i = rowStart; i < rowEnd; ++i) {
            for (uint32_t n0 = 0; n0 < n; n0 += BN) {
                uint32_t bn = MinU(BN, n - n0);
                LocalTensor<float> cRow = cTile;
                if (beta == 0.0f) {
                    Duplicate(cRow, 0.0f, bn);
                } else {
                    LoadCRow(cRow, gC, i, n0, bn, n);
                    if (beta != 1.0f) Muls(cRow, cRow, beta, bn);
                }
                StoreCRow(gC, cRow, i, n0, bn, n);
            }
        }
        return;
    }

    for (uint32_t i0 = rowStart; i0 < rowEnd; i0 += RM) {
        uint32_t rm = MinU(RM, rowEnd - i0);

        for (uint32_t n0 = 0; n0 < n; n0 += BN) {
            uint32_t bn = MinU(BN, n - n0);

            for (uint32_t r = 0; r < rm; ++r) {
                uint32_t i = i0 + r;
                LocalTensor<float> cRow = cTile[r * BN];
                if (beta == 0.0f) {
                    Duplicate(cRow, 0.0f, bn);
                } else if (beta == 1.0f) {
                    LoadCRow(cRow, gC, i, n0, bn, n);
                } else {
                    LoadCRow(tmp, gC, i, n0, bn, n);
                    Muls(cRow, tmp, beta, bn);
                }
            }

            if (alpha == 0.0f) {
                for (uint32_t r = 0; r < rm; ++r) {
                    StoreCRow(gC, cTile[r * BN], i0 + r, n0, bn, n);
                }
                continue;
            }

            uint32_t kTiles = (k + BK - 1) / BK;

            {
                uint32_t bk0 = MinU(BK, k);
                LocalTensor<float> b0 = qB.AllocTensor<float>();
                LoadBTile(b0, gB, 0, n0, bk0, bn, n);
                qB.EnQue(b0);
            }

            for (uint32_t kt = 0; kt < kTiles; ++kt) {
                uint32_t k0 = kt * BK;
                uint32_t bk = MinU(BK, k - k0);

                LocalTensor<float> bCur = qB.DeQue<float>();

                if (kt + 1 < kTiles) {
                    uint32_t k1 = (kt + 1) * BK;
                    uint32_t bk1 = MinU(BK, k - k1);
                    LocalTensor<float> bNext = qB.AllocTensor<float>();
                    LoadBTile(bNext, gB, k1, n0, bk1, bn, n);
                    qB.EnQue(bNext);
                }

                if (rm > 0) {
                    LocalTensor<float> a0 = qA.AllocTensor<float>();
                    LoadA(a0, gA, i0, k0, bk, k);
                    if (alpha != 1.0f) Muls(a0, a0, alpha, bk);
                    qA.EnQue(a0);
                }

                for (uint32_t r = 0; r < rm; ++r) {
                    uint32_t i = i0 + r;
                    LocalTensor<float> aCur = qA.DeQue<float>();

                    if (r + 1 < rm) {
                        LocalTensor<float> aNext = qA.AllocTensor<float>();
                        LoadA(aNext, gA, i0 + r + 1, k0, bk, k);
                        if (alpha != 1.0f) Muls(aNext, aNext, alpha, bk);
                        qA.EnQue(aNext);
                    }

                    LocalTensor<float> cRow = cTile[r * BN];
                    AccumulateRow(cRow, aCur, bCur, tmp, bk, bn);
                    qA.FreeTensor(aCur);
                }

                qB.FreeTensor(bCur);
            }

            for (uint32_t r = 0; r < rm; ++r) {
                StoreCRow(gC, cTile[r * BN], i0 + r, n0, bn, n);
            }
        }
    }
}