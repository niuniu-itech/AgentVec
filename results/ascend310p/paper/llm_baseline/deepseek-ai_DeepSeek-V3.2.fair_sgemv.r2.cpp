#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFNUM = 2;
constexpr int32_t TILE_M = 256;
constexpr int32_t TILE_N = 256;
constexpr int32_t REDUCE_LEN = 64;

class K_SGEMV {
public:
    __aicore__ inline K_SGEMV() {}
    
    __aicore__ inline void Init(GM_ADDR A, GM_ADDR x, GM_ADDR y, 
                                float alpha, float beta, 
                                uint32_t m, uint32_t n) {
        this->alpha = alpha;
        this->beta = beta;
        this->m = m;
        this->n = n;
        
        uint32_t totalTiles = (m + TILE_M - 1) / TILE_M;
        uint32_t tilesPerBlock = (totalTiles + GetBlockNum() - 1) / GetBlockNum();
        startM = GetBlockIdx() * tilesPerBlock * TILE_M;
        endM = startM + tilesPerBlock * TILE_M;
        if (endM > m) endM = m;
        
        if (startM >= endM) {
            validBlock = false;
            return;
        }
        validBlock = true;
        
        uint32_t blockRows = endM - startM;
        
        Ag.SetGlobalBuffer((__gm__ float*)A + startM * n, blockRows * n);
        xg.SetGlobalBuffer((__gm__ float*)x, n);
        yg.SetGlobalBuffer((__gm__ float*)y + startM, blockRows);
        yOut.SetGlobalBuffer((__gm__ float*)y + startM, blockRows);
        
        pipe.InitBuffer(inQueueA, BUFNUM, TILE_M * TILE_N * sizeof(float));
        pipe.InitBuffer(inQueueX, BUFNUM, TILE_N * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFNUM, TILE_M * sizeof(float));
        pipe.InitBuffer(calcBuf, REDUCE_LEN * sizeof(float));
    }
    
    __aicore__ inline void Process() {
        if (!validBlock) return;
        
        uint32_t blockRows = endM - startM;
        uint32_t tileMCount = (blockRows + TILE_M - 1) / TILE_M;
        uint32_t tileNCount = (n + TILE_N - 1) / TILE_N;
        
        for (uint32_t tileMIdx = 0; tileMIdx < tileMCount; ++tileMIdx) {
            uint32_t rowsInTile = TILE_M;
            if (tileMIdx == tileMCount - 1) {
                rowsInTile = blockRows - tileMIdx * TILE_M;
            }
            uint32_t rowOffset = tileMIdx * TILE_M;
            
            LocalTensor<float> accTile = outQueueY.AllocTensor<float>();
            Duplicate(accTile, 0.0f, TILE_M);
            
            if (beta != 0.0f) {
                LocalTensor<float> yTile = calcBuf.Get<float>();
                DataCopy(yTile, yg[rowOffset], rowsInTile);
                SetFlag(PIPE_MTE2, PIPE_S);
                WaitFlag(PIPE_MTE2, PIPE_S);
                
                for (uint32_t i = 0; i < rowsInTile; ++i) {
                    float yVal = yTile.GetValue(i);
                    yTile.SetValue(i, beta * yVal);
                }
                
                SetFlag(PIPE_S, PIPE_V);
                WaitFlag(PIPE_S, PIPE_V);
                Add(accTile, accTile, yTile, rowsInTile);
            }
            
            for (uint32_t tileNIdx = 0; tileNIdx < tileNCount; ++tileNIdx) {
                uint32_t colsInTile = TILE_N;
                if (tileNIdx == tileNCount - 1) {
                    colsInTile = n - tileNIdx * TILE_N;
                }
                uint32_t colOffset = tileNIdx * TILE_N;
                
                LocalTensor<float> aTile = inQueueA.AllocTensor<float>();
                LocalTensor<float> xTile = inQueueX.AllocTensor<float>();
                
                uint32_t aCopyLen = rowsInTile * colsInTile;
                uint32_t aSrcOffset = rowOffset * n + colOffset;
                DataCopy(aTile, Ag[aSrcOffset], aCopyLen);
                
                DataCopy(xTile, xg[colOffset], colsInTile);
                
                inQueueA.EnQue(aTile);
                inQueueX.EnQue(xTile);
                
                aTile = inQueueA.DeQue<float>();
                xTile = inQueueX.DeQue<float>();
                
                LocalTensor<float> partial = calcBuf.Get<float>();
                Duplicate(partial, 0.0f, REDUCE_LEN);
                
                for (uint32_t row = 0; row < rowsInTile; ++row) {
                    float sum = 0.0f;
                    uint32_t aRowStart = row * TILE_N;
                    
                    for (uint32_t col = 0; col < colsInTile; col += REDUCE_LEN) {
                        uint32_t len = REDUCE_LEN;
                        if (col + REDUCE_LEN > colsInTile) {
                            len = colsInTile - col;
                        }
                        LocalTensor<float> aSlice = aTile[aRowStart + col];
                        LocalTensor<float> xSlice = xTile[col];
                        
                        LocalTensor<float> prod = partial;
                        Mul(prod, aSlice, xSlice, len);
                        
                        float rowSum = 0.0f;
                        if (len <= REDUCE_LEN) {
                            Duplicate(partial, 0.0f, 8);
                            ReduceSum(partial, prod, len);
                            SetFlag(PIPE_V, PIPE_S);
                            WaitFlag(PIPE_V, PIPE_S);
                            rowSum = partial.GetValue(0);
                        }
                        sum += rowSum;
                    }
                    
                    partial.SetValue(0, sum);
                    SetFlag(PIPE_S, PIPE_V);
                    WaitFlag(PIPE_S, PIPE_V);
                    
                    float accVal = accTile.GetValue(row);
                    accTile.SetValue(row, accVal + alpha * partial.GetValue(0));
                }
                
                inQueueA.FreeTensor(aTile);
                inQueueX.FreeTensor(xTile);
            }
            
            outQueueY.EnQue(accTile);
            
            accTile = outQueueY.DeQue<float>();
            DataCopy(yOut[rowOffset], accTile, rowsInTile);
            outQueueY.FreeTensor(accTile);
        }
    }
    
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFNUM> inQueueA;
    TQue<QuePosition::VECIN, BUFNUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFNUM> outQueueY;
    TBuf<QuePosition::VECCALC> calcBuf;
    
    GlobalTensor<float> Ag;
    GlobalTensor<float> xg;
    GlobalTensor<float> yg;
    GlobalTensor<float> yOut;
    
    float alpha;
    float beta;
    uint32_t m;
    uint32_t n;
    uint32_t startM;
    uint32_t endM;
    bool validBlock;
};

extern "C" __global__ __aicore__ void sgemv(GM_ADDR A, GM_ADDR x, GM_ADDR y, 
                                            float alpha, float beta, 
                                            uint32_t m, uint32_t n) {
    K_SGEMV sgemvKernel;
    sgemvKernel.Init(A, x, y, alpha, beta, m, n);
    sgemvKernel.Process();
}