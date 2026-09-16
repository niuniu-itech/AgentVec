#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFNUM = 2, TILELEN = 1024;
class KSaxpy {
public:
  __aicore__ inline KSaxpy() {}
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR out, float alpha, uint32_t total) {
    blockLen = total / GetBlockNum(); numTile = blockLen / TILELEN;
    xg.SetGlobalBuffer((__gm__ float*)x + blockLen*GetBlockIdx(), blockLen);
    yg.SetGlobalBuffer((__gm__ float*)y + blockLen*GetBlockIdx(), blockLen);
    zg.SetGlobalBuffer((__gm__ float*)out + blockLen*GetBlockIdx(), blockLen);
    pipe.InitBuffer(qx, BUFNUM, TILELEN*sizeof(float));
    pipe.InitBuffer(qy, BUFNUM, TILELEN*sizeof(float));
    pipe.InitBuffer(qz, BUFNUM, TILELEN*sizeof(float));
    alphaVal = alpha;
  }
  __aicore__ inline void Process() {
    for (uint32_t i=0;i<numTile;i++){
      LocalTensor<float> xl=qx.AllocTensor<float>(), yl=qy.AllocTensor<float>();
      DataCopy(xl, xg[i*TILELEN], TILELEN); DataCopy(yl, yg[i*TILELEN], TILELEN);
      qx.EnQue(xl); qy.EnQue(yl);
      xl=qx.DeQue<float>(); yl=qy.DeQue<float>();
      LocalTensor<float> zl=qz.AllocTensor<float>();
      Muls(zl, xl, alphaVal, TILELEN);
      Add(zl, zl, yl, TILELEN);
      qz.EnQue<float>(zl);
      qx.FreeTensor(xl); qy.FreeTensor(yl);
      zl=qz.DeQue<float>(); DataCopy(zg[i*TILELEN], zl, TILELEN); qz.FreeTensor(zl);
    }
  }
private:
  TPipe pipe; TQue<QuePosition::VECIN,BUFNUM> qx,qy; TQue<QuePosition::VECOUT,BUFNUM> qz;
  GlobalTensor<float> xg,yg,zg; uint32_t blockLen,numTile;
  float alphaVal;
};

extern "C" __global__ __aicore__ void saxpy(GM_ADDR x, GM_ADDR y, GM_ADDR out, float alpha, uint32_t total) {
  KSaxpy op;
  op.Init(x, y, out, alpha, total);
  op.Process();
}