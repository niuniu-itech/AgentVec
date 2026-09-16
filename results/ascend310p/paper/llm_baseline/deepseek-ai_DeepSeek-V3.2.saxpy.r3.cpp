#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t TILE_LENGTH = 256;

class KernelSaxpy {
public:
    __aicore__ inline KernelSaxpy() {}
    
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR out, float alpha, uint32_t total)
    {
        this->alpha = alpha;
        this->total = total;
        
        ASSERT(GetBlockNum() == 8);
        
        xGm.SetGlobalBuffer((__gm__ float*)x + GetBlockIdx() * TILE_LENGTH, TILE_LENGTH);
        yGm.SetGlobalBuffer((__gm__ float*)y + GetBlockIdx() * TILE_LENGTH, TILE_LENGTH);
        outGm.SetGlobalBuffer((__gm__ float*)out + GetBlockIdx() * TILE_LENGTH, TILE_LENGTH);
        
        pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(float));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, TILE_LENGTH * sizeof(float));
        pipe.InitBuffer(outQueue, BUFFER_NUM, TILE_LENGTH * sizeof(float));
        
        alphaLocal = alpha;
    }
    
    __aicore__ inline void Process()
    {
        int32_t loopCount = total / (TILE_LENGTH * GetBlockNum());
        int32_t remainder = total % (TILE_LENGTH * GetBlockNum());
        int32_t tileNum = loopCount + (GetBlockIdx() < remainder ? 1 : 0);
        
        for (int32_t i = 0; i < tileNum; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }
    
private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        LocalTensor<float> yLocal = inQueueY.AllocTensor<float>();
        
        DataCopy(xLocal, xGm[progress * TILE_LENGTH * GetBlockNum()], TILE_LENGTH);
        DataCopy(yLocal, yGm[progress * TILE_LENGTH * GetBlockNum()], TILE_LENGTH);
        
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    
    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = inQueueY.DeQue<float>();
        LocalTensor<float> outLocal = outQueue.AllocTensor<float>();
        
        uint32_t repeatTimes = TILE_LENGTH / 64;
        
        for (uint32_t i = 0; i < repeatTimes; i++) {
            float* xPtr = xLocal.GetPtr(i * 64);
            float* yPtr = yLocal.GetPtr(i * 64);
            float* outPtr = outLocal.GetPtr(i * 64);
            
            for (uint32_t j = 0; j < 64; j += 8) {
                float8 xVec, yVec, outVec;
                xVec = vload(0, xPtr + j);
                yVec = vload(0, yPtr + j);
                
                outVec = xVec * alphaLocal;
                outVec = outVec + yVec;
                
                vstore(outVec, 0, outPtr + j);
            }
        }
        
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
        outQueue.EnQue(outLocal);
    }
    
    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<float> outLocal = outQueue.DeQue<float>();
        DataCopy(outGm[progress * TILE_LENGTH * GetBlockNum()], outLocal, TILE_LENGTH);
        outQueue.FreeTensor(outLocal);
    }
    
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    
    GlobalTensor<float> xGm;
    GlobalTensor<float> yGm;
    GlobalTensor<float> outGm;
    
    float alpha;
    float alphaLocal;
    uint32_t total;
};

extern "C" __global__ __aicore__ void saxpy(GM_ADDR x, GM_ADDR y, GM_ADDR out, float alpha, uint32_t total)
{
    KernelSaxpy op;
    op.Init(x, y, out, alpha, total);
    op.Process();
}