#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFNUM = 2;
constexpr int32_t TILELEN = 2048; // 2048 floats = 8192 bytes, aligned to 32 bytes (2048 % 8 == 0)
constexpr int32_t REDUCE_LANES = 64; // ReduceSum works reliably for <=64 floats

class KSDot {
public:
    __aicore__ inline KSDot() {}
    
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR partials, uint32_t total) {
        uint32_t blockNum = GetBlockNum();
        uint32_t blockIdx = GetBlockIdx();
        
        // Split work across blocks
        uint32_t blockLen = total / blockNum;
        uint32_t remainder = total % blockNum;
        
        // Handle remainder distribution
        if (blockIdx < remainder) {
            blockStart = blockIdx * (blockLen + 1);
            blockLen += 1;
        } else {
            blockStart = blockIdx * blockLen + remainder;
        }
        
        this->blockLen = blockLen;
        numTile = (blockLen + TILELEN - 1) / TILELEN;
        
        // Set up global tensors for input vectors
        xg.SetGlobalBuffer((__gm__ float*)x + blockStart, blockLen);
        yg.SetGlobalBuffer((__gm__ float*)y + blockStart, blockLen);
        
        // Set up global tensor for partial result (8-stride for alignment)
        partialsG.SetGlobalBuffer((__gm__ float*)partials + blockIdx * 8, 1);
        
        // Initialize pipe buffers
        pipe.InitBuffer(qx, BUFNUM, TILELEN * sizeof(float));
        pipe.InitBuffer(qy, BUFNUM, TILELEN * sizeof(float));
        pipe.InitBuffer(qmul, BUFNUM, TILELEN * sizeof(float));
        pipe.InitBuffer(tmp, REDUCE_LANES * sizeof(float)); // Scratch for reduction
        
        // Initialize accumulator with zeros
        Duplicate(acc, 0.0f, REDUCE_LANES);
    }
    
    __aicore__ inline void Process() {
        // Process tiles with double buffering
        for (uint32_t tileIdx = 0; tileIdx < numTile + BUFNUM - 1; ++tileIdx) {
            // Load phase (if there are more tiles to load)
            if (tileIdx < numTile) {
                // Load tile for x
                LocalTensor<float> xl = qx.AllocTensor<float>();
                uint32_t copyLen = GetTileLen(tileIdx);
                DataCopy(xl, xg[tileIdx * TILELEN], copyLen);
                qx.EnQue(xl);
                
                // Load tile for y
                LocalTensor<float> yl = qy.AllocTensor<float>();
                DataCopy(yl, yg[tileIdx * TILELEN], copyLen);
                qy.EnQue(yl);
            }
            
            // Compute phase (if we have loaded enough tiles)
            if (tileIdx >= 1 && tileIdx <= numTile) {
                // Get input tensors for computation
                LocalTensor<float> xl = qx.DeQue<float>();
                LocalTensor<float> yl = qy.DeQue<float>();
                uint32_t computeLen = GetTileLen(tileIdx - 1);
                
                // Allocate buffer for multiplication results
                LocalTensor<float> mul = qmul.AllocTensor<float>();
                
                // Compute element-wise product: mul = x * y
                Mul(mul, xl, yl, computeLen);
                qmul.EnQue(mul);
                
                // Free input tensors
                qx.FreeTensor(xl);
                qy.FreeTensor(yl);
            }
            
            // Reduce phase (if we have computed enough tiles)
            if (tileIdx >= BUFNUM) {
                LocalTensor<float> mul = qmul.DeQue<float>();
                uint32_t reduceLen = GetTileLen(tileIdx - BUFNUM);
                ReduceTile(mul, reduceLen);
                qmul.FreeTensor(mul);
            }
        }
        
        // Final reduction of accumulated values
        FinalReduce();
        
        // Write partial result to global memory
        WritePartial();
    }
    
private:
    __aicore__ inline uint32_t GetTileLen(uint32_t tileIdx) {
        uint32_t remaining = blockLen - tileIdx * TILELEN;
        return (remaining > TILELEN) ? TILELEN : remaining;
    }
    
    __aicore__ inline void ReduceTile(LocalTensor<float>& tile, uint32_t len) {
        LocalTensor<float> scratch = tmp.Get<float>();
        
        // For tiles larger than REDUCE_LANES, fold into smaller vector
        if (len > REDUCE_LANES) {
            // First REDUCE_LANES elements
            for (int32_t i = 0; i < REDUCE_LANES; ++i) {
                scratch.SetValue(i, tile.GetValue(i));
            }
            
            // Sync before reading scalar values
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            
            // Add remaining elements
            for (uint32_t i = REDUCE_LANES; i < len; ++i) {
                int32_t idx = i % REDUCE_LANES;
                float val = scratch.GetValue(idx) + tile.GetValue(i);
                scratch.SetValue(idx, val);
            }
            
            // Sync before vector operations
            SetFlag<HardEvent::S_V>(EVENT_ID1);
            WaitFlag<HardEvent::S_V>(EVENT_ID1);
            
            // Add to accumulator
            Add(acc, acc, scratch, REDUCE_LANES);
        } else {
            // For small tiles, add directly to accumulator
            Add(acc, acc, tile, len);
        }
    }
    
    __aicore__ inline void FinalReduce() {
        LocalTensor<float> scratch = tmp.Get<float>();
        
        // Sync before scalar access of accumulator
        SetFlag<HardEvent::V_S>(EVENT_ID2);
        WaitFlag<HardEvent::V_S>(EVENT_ID2);
        
        // Perform final reduction using scalar operations
        float sum = 0.0f;
        for (int32_t i = 0; i < REDUCE_LANES; ++i) {
            sum += acc.GetValue(i);
        }
        
        finalSum = sum;
    }
    
    __aicore__ inline void WritePartial() {
        // Create local tensor for the partial sum
        LocalTensor<float> partialLocal;
        partialLocal.SetValue(0, finalSum);
        
        // Sync before MTE3 operation
        SetFlag<HardEvent::S_MTE3>(EVENT_ID3);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID3);
        
        // Copy to global memory
        DataCopy(partialsG, partialLocal, 1);
    }
    
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFNUM> qx, qy;
    TQue<QuePosition::VECOUT, BUFNUM> qmul;
    TBuf<QuePosition::VECCALC> tmp;
    
    GlobalTensor<float> xg, yg, partialsG;
    LocalTensor<float> acc;
    
    uint32_t blockStart, blockLen, numTile;
    float finalSum = 0.0f;
};

extern "C" __global__ __aicore__ void sdot(GM_ADDR x, GM_ADDR y, GM_ADDR partials, uint32_t total) {
    KSDot dot;
    dot.Init(x, y, partials, total);
    dot.Process();
}