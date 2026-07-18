/* Standalone GEMM study (C = A*B, row-major, square N) for the gap/overhead analysis.
 *  - scalar:    plain triple loop (reference / correctness oracle)
 *  - agentvec:  AgentVec-style naive VLA vectorization (recovered intent: map-reduce over k,
 *               vectorized across the n dimension; vsetvl, no register tiling)
 *  - tiled:     register-tiled (4 C-rows reused per B load) -- represents the expert micro-arch
 *               optimization that AgentVec does NOT yet emit (the admitted gap).
 * Reports GFLOP/s for each and verifies correctness. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <riscv_vector.h>

static double wall(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }

static void gemm_scalar(int N, const double*A, const double*B, double*C){
    for(int i=0;i<N;i++) for(int j=0;j<N;j++){ double s=0; for(int k=0;k<N;k++) s+=A[i*N+k]*B[k*N+j]; C[i*N+j]=s; }
}

/* AgentVec naive VLA: for each row i, accumulate C[i][:] += A[i][k]*B[k][:] vectorized over j */
static void gemm_agentvec(int N, const double*A, const double*B, double*C){
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++) C[i*N+j]=0.0;
        for(int k=0;k<N;k++){
            double a=A[i*N+k];
            const double*Brow=&B[k*N]; double*Crow=&C[i*N];
            for(int j=0;j<N;){
                size_t vl=__riscv_vsetvl_e64m8(N-j);
                vfloat64m8_t vc=__riscv_vle64_v_f64m8(Crow+j,vl);
                vfloat64m8_t vb=__riscv_vle64_v_f64m8(Brow+j,vl);
                vc=__riscv_vfmacc_vf_f64m8(vc,a,vb,vl);
                __riscv_vse64_v_f64m8(Crow+j,vc,vl);
                j+=vl;
            }
        }
    }
}

/* register-tiled: 4 rows of C share each B[k][:] load (the expert micro-arch trick) */
static void gemm_tiled(int N, const double*A, const double*B, double*C){
    memset(C,0,sizeof(double)*N*N);
    int i=0;
    for(; i+4<=N; i+=4){
        for(int k=0;k<N;k++){
            double a0=A[(i+0)*N+k],a1=A[(i+1)*N+k],a2=A[(i+2)*N+k],a3=A[(i+3)*N+k];
            const double*Brow=&B[k*N];
            for(int j=0;j<N;){
                size_t vl=__riscv_vsetvl_e64m8(N-j);
                vfloat64m8_t vb=__riscv_vle64_v_f64m8(Brow+j,vl);
                vfloat64m8_t c0=__riscv_vle64_v_f64m8(&C[(i+0)*N+j],vl);
                vfloat64m8_t c1=__riscv_vle64_v_f64m8(&C[(i+1)*N+j],vl);
                vfloat64m8_t c2=__riscv_vle64_v_f64m8(&C[(i+2)*N+j],vl);
                vfloat64m8_t c3=__riscv_vle64_v_f64m8(&C[(i+3)*N+j],vl);
                c0=__riscv_vfmacc_vf_f64m8(c0,a0,vb,vl); c1=__riscv_vfmacc_vf_f64m8(c1,a1,vb,vl);
                c2=__riscv_vfmacc_vf_f64m8(c2,a2,vb,vl); c3=__riscv_vfmacc_vf_f64m8(c3,a3,vb,vl);
                __riscv_vse64_v_f64m8(&C[(i+0)*N+j],c0,vl); __riscv_vse64_v_f64m8(&C[(i+1)*N+j],c1,vl);
                __riscv_vse64_v_f64m8(&C[(i+2)*N+j],c2,vl); __riscv_vse64_v_f64m8(&C[(i+3)*N+j],c3,vl);
                j+=vl;
            }
        }
    }
    for(; i<N; i++) for(int k=0;k<N;k++){ double a=A[i*N+k]; for(int j=0;j<N;j++) C[i*N+j]+=a*B[k*N+j]; }
}

static double maxdiff(int N,const double*X,const double*Y){ double d=0; for(int i=0;i<N*N;i++){double e=fabs(X[i]-Y[i]); if(e>d)d=e;} return d; }

int main(int argc,char**argv){
    int N=argc>1?atoi(argv[1]):256; int reps=argc>2?atoi(argv[2]):3;
    double*A=malloc(8.0*N*N),*B=malloc(8.0*N*N),*C0=malloc(8.0*N*N),*C1=malloc(8.0*N*N),*C2=malloc(8.0*N*N);
    for(int i=0;i<N*N;i++){A[i]=(double)((i*1103515245+12345)%1000)/1000.0-0.5; B[i]=(double)((i*16807+7)%1000)/1000.0-0.5;}
    double flop=2.0*N*N*N;
    gemm_scalar(N,A,B,C0);
    double t,best;
    best=1e9; for(int r=0;r<reps;r++){t=wall();gemm_agentvec(N,A,B,C1);t=wall()-t; if(t<best)best=t;} double g_av=flop/best*1e-9, d_av=maxdiff(N,C0,C1);
    best=1e9; for(int r=0;r<reps;r++){t=wall();gemm_tiled(N,A,B,C2);t=wall()-t; if(t<best)best=t;} double g_ti=flop/best*1e-9, d_ti=maxdiff(N,C0,C2);
    best=1e9; for(int r=0;r<reps;r++){t=wall();gemm_scalar(N,A,B,C0);t=wall()-t; if(t<best)best=t;} double g_sc=flop/best*1e-9;
    printf("N=%d  scalar=%.3f GF/s  agentvec(naive VLA)=%.3f GF/s (err %.1e)  tiled=%.3f GF/s (err %.1e)\n",
           N,g_sc,g_av,d_av,g_ti,d_ti);
    printf("  AgentVec speedup vs scalar: %.2fx ; tiling headroom (tiled/agentvec): %.2fx\n", g_av/g_sc, g_ti/g_av);
    return 0;
}
