// AUTO-GENERATED differential-test harness for sgemv (GEMV)
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include "acl/acl.h"
#include "aclrtlaunch_sgemv.h"
#define CK(x) do{aclError e=(x); if(e!=ACL_SUCCESS){printf("ACLERR %d @%d\n",e,__LINE__);return 2;}}while(0)
int main(int argc,char**argv){
    uint32_t m=(argc>1)?(uint32_t)atol(argv[1]):1024;
    uint32_t n=(argc>2)?(uint32_t)atol(argv[2]):1024;
    int iters=(argc>3)?atoi(argv[3]):50;
    const uint32_t blockDim=8;
    if(n%64) n=(n/64)*64; if(n==0)n=64;                 // inner reduce folds in 64-lanes
    if((m/blockDim)%8){ m=((m/blockDim)/8*8)*blockDim; } if(m==0)m=blockDim*8;  // rows/core mult of 8
    float alpha=1.25f, beta=0.5f;
    size_t Ab=(size_t)m*n*sizeof(float), xb=(size_t)n*sizeof(float), yb=(size_t)m*sizeof(float);
    float*Ah=(float*)malloc(Ab),*xh=(float*)malloc(xb),*yh=(float*)malloc(yb),*yo=(float*)malloc(yb);
    for(size_t i=0;i<(size_t)m*n;i++) Ah[i]=(float)((int)(i%13)-6)*0.05f;
    for(uint32_t j=0;j<n;j++) xh[j]=(float)((int)(j%7)-3)*0.1f;
    for(uint32_t i=0;i<m;i++){ yh[i]=(float)((int)(i%5)-2)*0.2f; yo[i]=yh[i]; }
    CK(aclInit(nullptr)); CK(aclrtSetDevice(0)); aclrtStream st; CK(aclrtCreateStream(&st));
    void*Ad,*xd,*yd; CK(aclrtMalloc(&Ad,Ab,ACL_MEM_MALLOC_HUGE_FIRST));
    CK(aclrtMalloc(&xd,xb,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMalloc(&yd,yb,ACL_MEM_MALLOC_HUGE_FIRST));
    CK(aclrtMemcpy(Ad,Ab,Ah,Ab,ACL_MEMCPY_HOST_TO_DEVICE));
    CK(aclrtMemcpy(xd,xb,xh,xb,ACL_MEMCPY_HOST_TO_DEVICE));
    CK(aclrtMemcpy(yd,yb,yh,yb,ACL_MEMCPY_HOST_TO_DEVICE));
    CK(ACLRT_LAUNCH_KERNEL(sgemv)(blockDim,st,Ad,xd,yd,alpha,beta,m,n)); CK(aclrtSynchronizeStream(st));
    CK(aclrtMemcpy(yh,yb,yd,yb,ACL_MEMCPY_DEVICE_TO_HOST)); // correctness on fresh y
    double maxabs=0, maxref=0; for(uint32_t i=0;i<m;i++){ double s=0; for(uint32_t j=0;j<n;j++) s+=(double)Ah[i*n+j]*xh[j];
        double r=alpha*s+beta*(double)yo[i]; double e=fabs((double)yh[i]-r);
        if(e>maxabs)maxabs=e; if(fabs(r)>maxref)maxref=fabs(r); }
    double maxrel=maxabs/(maxref+1e-9);  // norm-wise rel error (gemv has near-zero rows)
    CK(aclrtMemcpy(yd,yb,yo,yb,ACL_MEMCPY_HOST_TO_DEVICE)); // reset y for timing
    auto t0=std::chrono::high_resolution_clock::now();
    for(int it=0;it<iters;it++) CK(ACLRT_LAUNCH_KERNEL(sgemv)(blockDim,st,Ad,xd,yd,alpha,beta,m,n));
    CK(aclrtSynchronizeStream(st));
    auto t1=std::chrono::high_resolution_clock::now();
    double us=std::chrono::duration<double,std::micro>(t1-t0).count()/iters;
    double gflops=2.0*(double)m*n/(us*1e-6)/1e9;
    double gbps=((double)m*n+ n + m)*sizeof(float)/(us*1e-6)/1e9;
    int pass = maxrel < 2e-3;
    printf("RESULT op=sgemv pattern=gemv n=%u m=%u maxrel=%.3e pass=%d us=%.3f GFLOPs=%.2f GBps=%.2f\n",
           n,m,maxrel,pass,us,gflops,gbps);
    aclrtFree(Ad);aclrtFree(xd);aclrtFree(yd);free(Ah);free(xh);free(yh);free(yo);
    aclrtDestroyStream(st);aclrtResetDevice(0);aclFinalize(); return pass?0:1;
}
