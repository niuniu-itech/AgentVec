#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFNUM = 2;
constexpr int32_t TILELEN = 1024;

class KSaxpy {
public:
    __aicore__ inline KSaxpy() {}
    
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR out, float alpha, uint32_t total) {
        this->alpha = alpha;
        
        // Split work across 8 blocks
        uint32_t blockLen = total / GetBlockNum();
        uint32_t remainder = total % GetBlockNum();
        
        // Handle remainder by giving first 'remainder' blocks one extra element
        if (GetBlockIdx() < remainder) {
            blockLen += 1;
            blockStart = GetBlockIdx() * blockLen;
        } else {
            blockStart = GetBlockIdx() * blockLen + remainder;
        }
        
        numTile = blockLen / TILELEN;
        tileRemainder = blockLen % TILELEN;
        
        // Set global buffers with proper offsets
        xg.SetGlobalBuffer((__gm__ float*)x + blockStart, blockLen);
        yg.SetGlobalBuffer((__gm__ float*)y + blockStart, blockLen);
        outg.SetGlobalBuffer((__gm__ float*)out + blockStart, blockLen);
        
        // Initialize pipe buffers for double-buffering
        pipe.InitBuffer(qx, BUFNUM, TILELEN * sizeof(float));
        pipe.InitBuffer(qy, BUFNUM, TILELEN * sizeof(float));
        pipe.InitBuffer(qout, BUFNUM, TILELEN * sizeof(float));
    }
    
    __aicore__ inline void Process() {
        // Process full tiles
        for (uint32_t i = 0; i < numTile; ++i) {
            // Allocate and copy input tiles
            LocalTensor<float> xl = qx.AllocTensor<float>();
            LocalTensor<float> yl = qy.AllocTensor<float>();
            
            DataCopy(xl, xg[i * TILELEN], TILELEN);
            DataCopy(yl, yg[i * TILELEN], TILELEN);
            
            qx.EnQue(xl);
            qy.EnQue(yl);
            
            // Dequeue for computation
            xl = qx.DeQue<float>();
            yl = qy.DeQue<float>();
            
            // Allocate output tile
            LocalTensor<float> outl = qout.AllocTensor<float>();
            
            // Compute saxpy: out = alpha * x + y
            // First compute alpha * x
            Muls(outl, xl, alpha, TILELEN);
            // Then add y
            Add(outl, outl, yl, TILELEN);
            
            qout.EnQue(outl);
            
            // Free input tensors
            qx.FreeTensor(xl);
            qy.FreeTensor(yl);
            
            // Dequeue and copy output
            outl = qout.DeQue<float>();
            DataCopy(outg[i * TILELEN], outl, TILELEN);
            qout.FreeTensor(outl);
        }
        
        // Process remainder (if any)
        if (tileRemainder > 0) {
            // Ensure remainder is multiple of 8 for alignment
            uint32_t alignedRemainder = ((tileRemainder + 7) / 8) * 8;
            
            // Allocate and copy remainder
            LocalTensor<float> xl = qx.AllocTensor<float>();
            LocalTensor<float> yl = qy.AllocTensor<float>();
            
            // Copy actual data
            DataCopy(xl, xg[numTile * TILELEN], tileRemainder);
            DataCopy(yl, yg[numTile * TILELEN], tileRemainder);
            
            // Allocate output for remainder
            LocalTensor<float> outl = qout.AllocTensor<float>();
            
            // Compute saxpy for remainder
            Muls(outl, xl, alpha, tileRemainder);
            Add(outl, outl, yl, tileRemainder);
            
            // Copy back remainder
            DataCopy(outg[numTile * TILELEN], outl, tileRemainder);
            
            // Free tensors
            qx.FreeTensor(xl);
            qy.FreeTensor(yl);
            qout.FreeTensor(outl);
        }
    }
    
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFNUM> qx, qy;
    TQue<QuePosition::VECOUT, BUFNUM> qout;
    GlobalTensor<float> xg, yg, outg;
    float alpha;
    uint32_t blockStart;
    uint32_t numTile;
    uint32_t tileRemainder;
};

extern "C" __global__ __aicore__ void saxpy(GM_ADDR x, GM_ADDR y, GM_ADDR out, float alpha, uint32_t total) {
    KSaxpy saxpyKernel;
    saxpyKernel.Init(x, y, out, alpha, total);
    saxpyKernel.Process();
}