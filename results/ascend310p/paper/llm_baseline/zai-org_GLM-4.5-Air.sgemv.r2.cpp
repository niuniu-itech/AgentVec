#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void sgemv(GM_ADDR A, GM_ADDR x, GM_ADDR y, float alpha, float beta, uint32_t m, uint32_t n) {
    int32_t block_idx = GetBlockIdx();
    int32_t block_num = GetBlockNum();
    
    // Calculate the number of rows each block should process
    int32_t rows_per_block = (m + block_num - 1) / block_num;
    int32_t start_row = block_idx * rows_per_block;
    int32_t end_row = min((block_idx + 1) * rows_per_block, m);
    int32_t rows_to_process = end_row - start_row;
    
    // Create GlobalTensor objects
    GlobalTensor<float> global_A(A);
    GlobalTensor<float> global_x(x);
    GlobalTensor<float> global_y(y);
    
    // Allocate tensors in UB
    LocalTensor<float> local_x;
    LocalTensor<float> local_y;
    LocalTensor<float> local_A;
    LocalTensor<float> local_y_out;
    
    // Allocate double buffers
    LocalTensor<float> local_A_buf[2];
    LocalTensor<float> local_y_buf[2];
    
    // Create pipe for double buffering
    TQue<float, 2> que_A;
    TQue<float, 2> que_y;
    
    // Create pipe for data transfer
    TPipe pipe;
    pipe.InitBuffer(que_A, local_A_buf, 2);
    pipe.InitBuffer(que_y, local_y_buf, 2);
    
    // Initialize local tensors
    TensorInit(local_x, n);
    TensorInit(local_y, 1);
    TensorInit(local_y_out, 1);
    TensorInit(local_A_buf[0], n);
    TensorInit(local_A_buf[1], n);
    TensorInit(local_y_buf[0], 1);
    TensorInit(local_y_buf[1], 1);
    
    // Copy x from GM to UB
    DataCopy(local_x, global_x, n);
    
    // Process rows
    for (int32_t row = start_row; row < end_row; row++) {
        // Create sub-tensors for the current row
        GlobalTensor<float> global_A_row = global_A.GetSubTensor(row * n, n);
        GlobalTensor<float> global_y_row = global_y.GetSubTensor(row, 1);
        
        // Double buffering for A and y
        for (int32_t buf_idx = 0; buf_idx < 2; buf_idx++) {
            // Copy current row of A and y from GM to UB
            DataCopy(local_A_buf[buf_idx], global_A_row, n);
            DataCopy(local_y_buf[buf_idx], global_y_row, 1);
            
            // Enqueue in pipe
            que_A.Enque(local_A_buf[buf_idx]);
            que_y.Enque(local_y_buf[buf_idx]);
        }
        
        // Process the row
        local_A = que_A.Deque();
        local_y = que_y.Deque();
        
        // Compute dot product using vectorization
        const int32_t VEC_SIZE = 16;
        int32_t num_vecs = n / VEC_SIZE;
        Vec<float, VEC_SIZE> sum_vec = 0;
        
        // Process in vector chunks
        for (int32_t v = 0; v < num_vecs; v++) {
            Vec<float, VEC_SIZE> a_vec = local_A.template Load<VEC_SIZE>(v * VEC_SIZE);
            Vec<float, VEC_SIZE> x_vec = local_x.template Load<VEC_SIZE>(v * VEC_SIZE);
            sum_vec = sum_vec + a_vec * x_vec;
        }
        
        // Handle remaining elements
        float sum = sum_vec.template ReduceSum<VEC_SIZE>();
        for (int32_t j = num_vecs * VEC_SIZE; j < n; j++) {
            sum += local_A.GetValue(j) * local_x.GetValue(j);
        }
        
        // Apply alpha and beta
        sum *= alpha;
        float y_val = local_y.GetValue(0) * beta + sum;
        local_y_out.SetValue(0, y_val);
        
        // Copy result back to GM
        DataCopy(global_y_row, local_y_out, 1);
        
        // Enqueue back for next iteration
        que_A.Enque(local_A);
        que_y.Enque(local_y);
    }
}