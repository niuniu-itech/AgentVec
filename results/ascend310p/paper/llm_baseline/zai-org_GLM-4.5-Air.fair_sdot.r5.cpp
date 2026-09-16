#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFNUM = 2;
constexpr int32_t TILELEN = 256;  // Must be multiple of 8 for alignment
constexpr int32_t VECNUM = 8;     // Vector width for reduction

class KDot {
public:
    __aicore__ inline KDot() {}
    
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR partials, uint32_t total) {
        blockLen = (total + GetBlockNum() - 1) / GetBlockNum();
        startIdx = blockLen * GetBlockIdx();
        endIdx = ((startIdx + blockLen) < total) ? (startIdx + blockLen) : total;
        actualLen = endIdx - startIdx;
        numTile = (actualLen + TILELEN - 1) / TILELEN;
        
        xg.SetGlobalBuffer((__gm__ float*)x + startIdx, actualLen);
        yg.SetGlobalBuffer((__gm__ float*)y + startIdx, actualLen);
        pg.SetGlobalBuffer((__gm__ float*)partials + GetBlockIdx() * 8, 8);
        
        pipe.InitBuffer(qx, BUFNUM, TILELEN * sizeof(float));
        pipe.InitBuffer(qy, BUFNUM, TILELEN * sizeof(float));
        pipe.InitBuffer(tmp, VECNUM * sizeof(float));
    }
    
    __aicore__ inline void Process() {
        // Initialize partial sum for this block
        LocalTensor<float> partialSum = tmp.Get<float>();
        Duplicate(partialSum, 0.0f, VECNUM);
        
        for (uint32_t i = 0; i < numTile; i++) {
            uint32_t currentTileLen = (TILELEN < (actualLen - i * TILELEN)) ? TILELEN : (actualLen - i * TILELEN);
            
            // Double buffering for input
            LocalTensor<float> xl = qx.AllocTensor<float>();
            LocalTensor<float> yl = qy.AllocTensor<float>();
            
            DataCopy(xl, xg[i * TILELEN], currentTileLen);
            DataCopy(yl, yg[i * TILELEN], currentTileLen);
            
            qx.EnQue(xl);
            qy.EnQue(yl);
            
            xl = qx.DeQue<float>();
            yl = qy.DeQue<float>();
            
            // Compute element-wise multiplication
            LocalTensor<float> product = tmp.Get<float>();
            Mul(product, xl, yl, currentTileLen);
            
            // Reduce the tile into partialSum
            for (uint32_t j = 0; j < currentTileLen; j += VECNUM) {
                uint32_t vecLen = (VECNUM < (currentTileLen - j)) ? VECNUM : (currentTileLen - j);
                LocalTensor<float> productSlice = tmp.Get<float>();
                DataCopy(productSlice, product, vecLen);
                
                // Perform reduction using vector operations
                for (uint32_t k = 0; k < vecLen; k++) {
                    float val = productSlice.GetValue(k);
                    partialSum.SetValue(k, partialSum.GetValue(k) + val);
                }
                
                // Sync between vector and scalar operations
                SetFlag<HardEvent::V_S>(0);
                WaitFlag<HardEvent::V_S>(0);
            }
            
            qx.FreeTensor(xl);
            qy.FreeTensor(yl);
        }
        
        // Final reduction across vector lanes
        LocalTensor<float> temp = tmp.Get<float>();
        for (uint32_t i = 1; i < VECNUM; i++) {
            float val = partialSum.GetValue(i);
            temp.SetValue(0, val);
            
            // Sync before scalar operation
            SetFlag<HardEvent::V_S>(0);
            WaitFlag<HardEvent::V_S>(0);
            
            Add(partialSum, partialSum, temp, 1);
            
            // Sync after vector operation
            SetFlag<HardEvent::S_V>(0);
            WaitFlag<HardEvent::S_V>(0);
        }
        
        // Write the final partial sum to global memory
        // Only the first lane has the correct sum
        LocalTensor<float> result = tmp.Get<float>();
        DataCopy(result, partialSum, 1);
        
        // Sync before writing to GM
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        
        // Use SetValue to write to global tensor
        pg.SetValue(0, result.GetValue(0));
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFNUM> qx, qy;
    TBuf<QuePosition::VECCALC> tmp;
    
    GlobalTensor<float> xg, yg, pg;
    uint32_t blockLen, startIdx, endIdx, actualLen, numTile;
};

extern "C" __global__ __aicore__ void sdot(GM_ADDR x, GM_ADDR y, GM_ADDR partials, uint32_t total) {
    KDot op;
    op.Init(x, y, partials, total);
    op.Process();
}