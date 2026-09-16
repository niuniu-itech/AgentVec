#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void sdot(GM_ADDR x, GM_ADDR y, GM_ADDR partials, uint32_t total) {
    // Get block information
    int32_t blockId = GetBlockIdx();
    int32_t blockNum = GetBlockNum();
    
    // Each block processes total/blockNum elements
    uint32_t elementsPerBlock = total / blockNum;
    uint32_t startIdx = blockId * elementsPerBlock;
    uint32_t endIdx = (blockId == blockNum - 1) ? total : (blockId + 1) * elementsPerBlock;
    
    // Local tensor for input data
    LocalTensor<float> local_x = NewLocalTensor<float>(elementsPerBlock);
    LocalTensor<float> local_y = NewLocalTensor<float>(elementsPerBlock);
    
    // Local tensor for partial sum
    LocalTensor<float> local_sum = NewLocalTensor<float>(1);
    local_sum(0) = 0.0f;
    
    // Load data in chunks to unified buffer
    uint32_t chunkSize = 1024;
    uint32_t numChunks = (elementsPerBlock + chunkSize - 1) / chunkSize;
    
    for (uint32_t chunk = 0; chunk < numChunks; ++chunk) {
        uint32_t offset = chunk * chunkSize;
        uint32_t currentChunkSize = (offset + chunkSize > elementsPerBlock) ? 
                                   elementsPerBlock - offset : chunkSize;
        
        // Load chunk of data
        DataCopy(local_x, x + startIdx + offset, currentChunkSize, 0);
        DataCopy(local_y, y + startIdx + offset, currentChunkSize, 0);
        
        // Vectorized computation
        for (uint32_t i = 0; i < currentChunkSize; ++i) {
            local_sum(0) += local_x(i) * local_y(i);
        }
    }
    
    // Reduce within block
    float result = local_sum(0);
    
    // Store partial result
    DataCopy(partials + blockId * 8, &result, 1, 0);
}