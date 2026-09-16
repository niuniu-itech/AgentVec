#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t BLOCK_SIZE = 8;
constexpr int32_t TILE_LENGTH = 256;
constexpr int32_t VEC_CORE_NUM = 8;
constexpr int32_t ALIGN_SIZE = 32;
constexpr int32_t ELEM_SIZE = sizeof(float);

extern "C" __global__ __aicore__ void sdot(GM_ADDR x, GM_ADDR y, GM_ADDR partials, uint32_t total) {
    LocalTensor<float> localX[BUFFER_NUM];
    LocalTensor<float> localY[BUFFER_NUM];
    TBuf<TPosition::VECCALC> reduceBuf;
    TPipe pipe;
    
    int32_t blockIdx = GetBlockIdx();
    int32_t blockNum = GetBlockNum();
    uint32_t totalElements = total;
    
    uint32_t tileNum = totalElements / (blockNum * TILE_LENGTH) + 
                      (totalElements % (blockNum * TILE_LENGTH) ? 1 : 0);
    uint32_t tileLength = TILE_LENGTH;
    
    for (int32_t i = 0; i < BUFFER_NUM; ++i) {
        localX[i] = pipe.AllocTensor<float>(TILE_LENGTH);
        localY[i] = pipe.AllocTensor<float>(TILE_LENGTH);
    }
    
    uint32_t tileLoop = tileNum;
    uint32_t tileCount = 0;
    uint32_t offset = blockIdx * TILE_LENGTH;
    
    float blockSum = 0.0f;
    
    pipe.InitBuffer(reduceBuf, TILE_LENGTH * ELEM_SIZE);
    
    for (int32_t i = 0; i < BUFFER_NUM && tileCount < tileLoop; ++i) {
        uint32_t copySize = (offset + TILE_LENGTH <= totalElements) ? 
                           TILE_LENGTH : totalElements - offset;
        
        if (copySize > 0) {
            DataCopy(localX[i], x + offset * ELEM_SIZE, copySize);
            DataCopy(localY[i], y + offset * ELEM_SIZE, copySize);
        }
        
        offset += blockNum * TILE_LENGTH;
        ++tileCount;
    }
    
    for (uint32_t i = 0; i < tileLoop; ++i) {
        int32_t bufferIdx = i % BUFFER_NUM;
        
        if (i + BUFFER_NUM < tileLoop) {
            uint32_t copySize = (offset + TILE_LENGTH <= totalElements) ? 
                               TILE_LENGTH : totalElements - offset;
            
            if (copySize > 0) {
                DataCopy(localX[bufferIdx], x + offset * ELEM_SIZE, copySize);
                DataCopy(localY[bufferIdx], y + offset * ELEM_SIZE, copySize);
            }
            
            offset += blockNum * TILE_LENGTH;
        }
        
        uint32_t processSize = ((blockIdx * TILE_LENGTH + i * blockNum * TILE_LENGTH + TILE_LENGTH) <= totalElements) ?
                              TILE_LENGTH : totalElements - (blockIdx * TILE_LENGTH + i * blockNum * TILE_LENGTH);
        
        if (processSize > 0) {
            float tileSum = 0.0f;
            
            for (int32_t vecIdx = 0; vecIdx < VEC_CORE_NUM; ++vecIdx) {
                uint32_t vecLen = TILE_LENGTH / VEC_CORE_NUM;
                uint32_t vecOffset = vecIdx * vecLen;
                
                if (vecOffset < processSize) {
                    uint32_t validLen = (vecOffset + vecLen <= processSize) ? vecLen : processSize - vecOffset;
                    
                    LocalTensor<float> xVec = localX[bufferIdx];
                    LocalTensor<float> yVec = localY[bufferIdx];
                    
                    for (uint32_t j = 0; j < validLen; ++j) {
                        float xVal = xVec.GetValue(vecOffset + j);
                        float yVal = yVec.GetValue(vecOffset + j);
                        tileSum += xVal * yVal;
                    }
                }
            }
            
            blockSum += tileSum;
        }
    }
    
    uint32_t partialsOffset = blockIdx * BLOCK_SIZE * ELEM_SIZE;
    float* partialsPtr = reinterpret_cast<float*>(partials + partialsOffset);
    *partialsPtr = blockSum;
}