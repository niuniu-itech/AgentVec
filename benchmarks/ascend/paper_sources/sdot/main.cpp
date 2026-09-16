// AUTO-GENERATED differential-test harness for sdot (REDUCE sum/pre=ab)
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include "acl/acl.h"
#include "aclrtlaunch_sdot.h"
#define CK(x) do{aclError e=(x); if(e!=ACL_SUCCESS){printf("ACLERR %d @%d\n",e,__LINE__);return 2;}}while(0)
int main(int argc,char**argv){
    uint32_t n=(argc>1)?(uint32_t)atol(argv[1]):(1u<<20);
    int iters=(argc>2)?atoi(argv[2]):50;
    const uint32_t blockDim=8;
    n=(n/(blockDim*1024))*(blockDim*1024); if(n==0)n=blockDim*1024;
    size_t bytes=(size_t)n*sizeof(float);
    float*xh=(float*)malloc(bytes),*yh=(float*)malloc(bytes);
    for(uint32_t i=0;i<n;i++){xh[i]=(float)((int)(i%17)-8)*0.0625f; yh[i]=(float)((int)(i%13)-6)*0.125f;}
    CK(aclInit(nullptr)); CK(aclrtSetDevice(0)); aclrtStream st; CK(aclrtCreateStream(&st));
    void*xd,*yd,*pd; size_t pbytes=(size_t)blockDim*8*sizeof(float);
    CK(aclrtMalloc(&xd,bytes,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMalloc(&yd,bytes,ACL_MEM_MALLOC_HUGE_FIRST));
    CK(aclrtMalloc(&pd,pbytes,ACL_MEM_MALLOC_HUGE_FIRST));
    CK(aclrtMemcpy(xd,bytes,xh,bytes,ACL_MEMCPY_HOST_TO_DEVICE));
    CK(aclrtMemcpy(yd,bytes,yh,bytes,ACL_MEMCPY_HOST_TO_DEVICE));
    CK(ACLRT_LAUNCH_KERNEL(sdot)(blockDim,st,xd,yd,pd,n)); CK(aclrtSynchronizeStream(st)); // warmup
    auto t0=std::chrono::high_resolution_clock::now();
    for(int it=0;it<iters;it++) CK(ACLRT_LAUNCH_KERNEL(sdot)(blockDim,st,xd,yd,pd,n));
    CK(aclrtSynchronizeStream(st));
    auto t1=std::chrono::high_resolution_clock::now();
    double us=std::chrono::duration<double,std::micro>(t1-t0).count()/iters;
    float*ph=(float*)malloc(pbytes); CK(aclrtMemcpy(ph,pbytes,pd,pbytes,ACL_MEMCPY_DEVICE_TO_HOST));
    double dev=0; for(uint32_t b=0;b<blockDim;b++) dev+=ph[b*8]; double devr=(dev);
    double acc=0; for(uint32_t i=0;i<n;i++) acc+=xh[i]*yh[i]; double ref=acc;
    double maxrel=fabs(devr-ref)/(fabs(ref)+1e-12);
    double gbps=2.0*(double)n*sizeof(float)/(us*1e-6)/1e9;
    int pass = maxrel < 1e-3;
    printf("RESULT op=sdot pattern=reduce n=%u maxrel=%.3e pass=%d us=%.3f GBps=%.2f\n",n,maxrel,pass,us,gbps);
    aclrtFree(xd);aclrtFree(yd);aclrtFree(pd);free(xh);free(yh);free(ph);
    aclrtDestroyStream(st);aclrtResetDevice(0);aclFinalize(); return pass?0:1;
}
