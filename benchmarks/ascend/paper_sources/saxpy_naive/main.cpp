// AUTO-GENERATED differential-test harness for saxpy (MAP/axpy)
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include "acl/acl.h"
#include "aclrtlaunch_saxpy.h"
#define CK(x) do{aclError e=(x); if(e!=ACL_SUCCESS){printf("ACLERR %d @%d\n",e,__LINE__);return 2;}}while(0)
int main(int argc,char**argv){
    uint32_t n = (argc>1)?(uint32_t)atol(argv[1]):(1u<<20);
    int iters = (argc>2)?atoi(argv[2]):50;
    const uint32_t blockDim = 1;
    n = (n/(blockDim*256))*(blockDim*256); if(n==0)n=blockDim*256;
    float alpha=1.5f; size_t bytes=(size_t)n*sizeof(float);
    float*xh=(float*)malloc(bytes),*yh=(float*)malloc(bytes),*oh=(float*)malloc(bytes);
    for(uint32_t i=0;i<n;i++){xh[i]=(float)((int)(i%17)-8)*0.125f; yh[i]=(float)((int)(i%13)-6)*0.25f;}
    CK(aclInit(nullptr)); CK(aclrtSetDevice(0)); aclrtStream st; CK(aclrtCreateStream(&st));
    void*xd,*yd,*od; CK(aclrtMalloc(&xd,bytes,ACL_MEM_MALLOC_HUGE_FIRST));
    CK(aclrtMalloc(&yd,bytes,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMalloc(&od,bytes,ACL_MEM_MALLOC_HUGE_FIRST));
    CK(aclrtMemcpy(xd,bytes,xh,bytes,ACL_MEMCPY_HOST_TO_DEVICE));
    CK(aclrtMemcpy(yd,bytes,yh,bytes,ACL_MEMCPY_HOST_TO_DEVICE));
    CK(ACLRT_LAUNCH_KERNEL(saxpy)(blockDim,st,xd,yd,od,alpha,n)); CK(aclrtSynchronizeStream(st)); // warmup
    auto t0=std::chrono::high_resolution_clock::now();
    for(int it=0;it<iters;it++) CK(ACLRT_LAUNCH_KERNEL(saxpy)(blockDim,st,xd,yd,od,alpha,n));
    CK(aclrtSynchronizeStream(st));
    auto t1=std::chrono::high_resolution_clock::now();
    double us=std::chrono::duration<double,std::micro>(t1-t0).count()/iters;
    CK(aclrtMemcpy(oh,bytes,od,bytes,ACL_MEMCPY_DEVICE_TO_HOST));
    double maxrel=0; for(uint32_t i=0;i<n;i++){double r=alpha*xh[i]+yh[i]; double e=fabs((double)oh[i]-r)/(fabs(r)+1e-12); if(e>maxrel)maxrel=e;}
    double gbps=3.0*(double)n*sizeof(float)/(us*1e-6)/1e9;
    int pass = maxrel < 1e-5;
    printf("RESULT op=saxpy pattern=map n=%u maxrel=%.3e pass=%d us=%.3f GBps=%.2f\n",n,maxrel,pass,us,gbps);
    aclrtFree(xd);aclrtFree(yd);aclrtFree(od);free(xh);free(yh);free(oh);
    aclrtDestroyStream(st);aclrtResetDevice(0);aclFinalize(); return pass?0:1;
}
