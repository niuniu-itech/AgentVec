#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFNUM = 2;
constexpr int32_t TILELEN = 1024;

class KDot {
public:
  __aicore__ inline KDot() {}
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR partials, uint32_t total) {
    blockLen = total / GetBlockNum();
    numTile = blockLen / TILELEN;
    xg.SetGlobalBuffer((__gm__ float*)x + blockLen*GetBlockIdx(), blockLen);
    yg.SetGlobalBuffer((__gm__ float*)y + blockLen*GetBlockIdx(), blockLen);
    partialsg.SetGlobalBuffer((__gm__ float*)partials + GetBlockIdx() * 8, 1);
    pipe.InitBuffer(qx, BUFNUM, TILELEN*sizeof(float));
    pipe.InitBuffer(qy, BUFNUM, TILELEN*sizeof(float));
    pipe.InitBuffer(tmp, TILELEN*sizeof(float));
  }
  __aicore__ inline void Process() {
    LocalTensor<float> acc = tmp.Get<float>();
    Duplicate(acc, 0.0f, 8);
    for (uint32_t i=0; i<numTile; i++) {
      LocalTensor<float> xl = qx.AllocTensor<float>();
      LocalTensor<float> yl = qy.AllocTensor<float>();
      DataCopy(xl, xg[i*TILELEN], TILELEN);
      DataCopy(yl, yg[i*TILELEN], TILELEN);
      qx.EnQue(xl);
      qy.EnQue(yl);
      xl = qx.DeQue<float>();
      yl = qy.DeQue<float>();
      LocalTensor<float> tmp_mul = tmp.Get<float>();
      Mul(tmp_mul, xl, yl, TILELEN);
      Add(acc, acc, tmp_mul, 8);
      qx.FreeTensor(xl);
      qy.FreeTensor(yl);
    }
    float result = 0.0f;
    ReduceSum<float>(result, acc, 8, 8);
    LocalTensor<float> result_tensor = tmp.Get<float>();
    result_tensor.SetValue(0, result);
    DataCopy(partialsg, result_tensor, 1);
  }
private:
  TPipe pipe;
  TQue<QuePosition::VECIN, BUFNUM> qx, qy;
  TBuf<QuePosition::VECCALC> tmp;
  GlobalTensor<float> xg, yg, partialsg;
  uint32_t blockLen, numTile;
};

extern "C" __global__ __aicore__ void sdot(GM_ADDR x, GM_ADDR y, GM_ADDR partials, uint32_t total) {
  KDot kernel;
  kernel.Init(x, y, partials, total);
  kernel.Process();
}