#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFNUM = 2;
constexpr int32_t TILELEN = 1024;

class KSaxpy {
public:
    __aicore__ inline KSaxpy() {}
    
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR out, float alpha, uint32_t total) {
        blockLen = total / GetBlockNum();
        numTile = (blockLen + TILELEN - 1) / TILELEN;
        
        xg.SetGlobalBuffer((__gm__ float*)x + blockLen * GetBlockIdx(), blockLen);
        yg.SetGlobalBuffer((__gm__ float*)y + blockLen * GetBlockIdx(), blockLen);
        outg.SetGlobalBuffer((__gm__ float*)out + blockLen * GetBlockIdx(), blockLen);
        
        pipe.InitBuffer(qx, BUFNUM, TILELEN * sizeof(float));
        pipe.InitBuffer(qy, BUFNUM, TILELEN * sizeof(float));
        pipe.InitBuffer(qout, BUFNUM, TILELEN * sizeof(float));
        
        // Set alpha value for all tiles
        this->alpha = alpha;
    }
    
    __aicore__ inline void Process() {
        for (uint32_t i = 0; i < numTile; i++) {
            uint32_t currentTileLen = (i == numTile - 1) ? (blockLen - i * TILELEN) : TILELEN;
            
            // Double buffering for input tensors
            LocalTensor<float> xl = qx.AllocTensor<float>();
            LocalTensor<float> yl = qy.AllocTensor<float>();
            
            DataCopy(xl, xg[i * TILELEN], currentTileLen);
            DataCopy(yl, yg[i * TILELEN], currentTileLen);
            
            qx.EnQue(xl);
            qy.EnQue(yl);
            
            // Get tensors from queue (sync with MTE2_V)
            xl = qx.DeQue<float>();
            yl = qy.DeQue<float>();
            
            // Allocate output tensor
            LocalTensor<float> outl = qout.AllocTensor<float>();
            
            // Perform saxpy operation: outl = alpha * xl + yl
            Muls(outl, xl, alpha, currentTileLen);
            Add(outl, outl, yl, currentTileLen);
            
            // Enqueue output tensor
            qout.EnQue<float>(outl);
            
            // Free input tensors
            qx.FreeTensor(xl);
            qy.FreeTensor(yl);
            
            // Get output tensor from queue (sync with V_MTE3)
            outl = qout.DeQue<float>();
            
            // Copy result to global memory
            DataCopy(outg[i * TILELEN], outl, currentTileLen);
            
            // Free output tensor
            qout.FreeTensor(outl);
        }
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFNUM> qx;
    TQue<QuePosition::VECIN, BUFNUM> qy;
    TQue<QuePosition::VECOUT, BUFNUM> qout;
    
    GlobalTensor<float> xg;
    GlobalTensor<float> yg;
    GlobalTensor<float> outg;
    
    uint32_t blockLen;
    uint32_t numTile;
    float alpha;
};

extern "C" __global__ __aicore__ void saxpy(GM_ADDR x, GM_ADDR y, GM_ADDR out, float alpha, uint32_t total) {
    KSaxpy op;
    op.Init(x, y, out, alpha, total);
    op.Process();
}