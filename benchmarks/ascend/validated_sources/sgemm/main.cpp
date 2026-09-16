// AUTO-GENERATED differential-test harness for sgemm (GEMM)
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cerrno>
#include "acl/acl.h"
#include "aclrtlaunch_sgemm.h"
#define CK(x) do{aclError e=(x); if(e!=ACL_SUCCESS){printf("ACLERR %d @%d\n",e,__LINE__);return 2;}}while(0)
static uint32_t parse_size(const char *s) {
    if (!s || *s<'0' || *s>'9') std::exit(2);
    errno=0; char *end=nullptr; unsigned long long v=strtoull(s,&end,10);
    if(errno || *end || v==0 || v>67108864) std::exit(2);
    return (uint32_t)v;
}
int main(int argc,char**argv){
    uint32_t m=(argc>1)?parse_size(argv[1]):512;
    uint32_t n=(argc>2)?parse_size(argv[2]):512;
    uint32_t k=(argc>3)?parse_size(argv[3]):512;
    int iters=(argc>4)?(int)parse_size(argv[4]):10;
    const uint32_t blockDim=8;
    if(m%blockDim) return 2;
    if(n%8) return 2; if(k%8) return 2; if((5ull*n+k)*4>262144) return 2;
    if(iters<1 || iters>100000) return 2;
    float alpha=1.1f, beta=0.3f;
    size_t Ab=(size_t)m*k*4, Bb=(size_t)k*n*4, Cb=(size_t)m*n*4;
    float*Ah=(float*)malloc(Ab),*Bh=(float*)malloc(Bb),*Ch=(float*)malloc(Cb),*Co=(float*)malloc(Cb); if(!Ah||!Bh||!Ch||!Co)return 3;
    for(size_t i=0;i<(size_t)m*k;i++)Ah[i]=(float)((int)(i%11)-5)*0.03f;
    for(size_t i=0;i<(size_t)k*n;i++)Bh[i]=(float)((int)(i%7)-3)*0.05f;
    for(size_t i=0;i<(size_t)m*n;i++){Ch[i]=(float)((int)(i%5)-2)*0.1f; Co[i]=Ch[i];}
    CK(aclInit(nullptr)); CK(aclrtSetDevice(0)); aclrtStream st; CK(aclrtCreateStream(&st));
    void*Ad,*Bd,*Cd; CK(aclrtMalloc(&Ad,Ab,ACL_MEM_MALLOC_HUGE_FIRST));
    CK(aclrtMalloc(&Bd,Bb,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMalloc(&Cd,Cb,ACL_MEM_MALLOC_HUGE_FIRST));
    CK(aclrtMemcpy(Ad,Ab,Ah,Ab,ACL_MEMCPY_HOST_TO_DEVICE));
    CK(aclrtMemcpy(Bd,Bb,Bh,Bb,ACL_MEMCPY_HOST_TO_DEVICE));
    CK(aclrtMemcpy(Cd,Cb,Ch,Cb,ACL_MEMCPY_HOST_TO_DEVICE));
    CK(ACLRT_LAUNCH_KERNEL(sgemm)(blockDim,st,Ad,Bd,Cd,alpha,beta,m,n,k)); CK(aclrtSynchronizeStream(st));
    CK(aclrtMemcpy(Ch,Cb,Cd,Cb,ACL_MEMCPY_DEVICE_TO_HOST)); // correctness
    double maxabs=0,maxref=0; for(uint32_t i=0;i<m;i++)for(uint32_t j=0;j<n;j++){ double s=0;
        for(uint32_t kk=0;kk<k;kk++) s+=(double)Ah[i*k+kk]*Bh[kk*n+j];
        double r=alpha*s+beta*(double)Co[i*n+j]; if(!std::isfinite(Ch[i*n+j]) || !std::isfinite(r)){maxabs=INFINITY;continue;} double e=fabs((double)Ch[i*n+j]-r);
        if(e>maxabs)maxabs=e; if(fabs(r)>maxref)maxref=fabs(r); }
    double maxrel=maxabs/(maxref+1e-9);
    CK(aclrtMemcpy(Cd,Cb,Co,Cb,ACL_MEMCPY_HOST_TO_DEVICE)); // reset C for timing
    auto t0=std::chrono::steady_clock::now();
    for(int it=0;it<iters;it++) CK(ACLRT_LAUNCH_KERNEL(sgemm)(blockDim,st,Ad,Bd,Cd,alpha,beta,m,n,k));
    CK(aclrtSynchronizeStream(st));
    auto t1=std::chrono::steady_clock::now();
    double us=std::chrono::duration<double,std::micro>(t1-t0).count()/iters;
    double gflops=2.0*(double)m*n*k/(us*1e-6)/1e9;
    int pass = std::isfinite(maxrel) && maxrel < 2e-3;
    printf("RESULT op=sgemm pattern=gemm n=%u m=%u k=%u maxrel=%.3e pass=%d us=%.3f GFLOPs=%.2f GBps=%.2f\n",
           n,m,k,maxrel,pass,us,gflops,0.0);
    aclrtFree(Ad);aclrtFree(Bd);aclrtFree(Cd);free(Ah);free(Bh);free(Ch);free(Co);
    aclrtDestroyStream(st);aclrtResetDevice(0);aclFinalize(); return pass?0:1;
}
