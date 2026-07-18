#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
#define OUT_FACTOR 1
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)c;
    float sum = 0.0f;
    int i = 0;
    vfloat32m1_t vsum = __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvl_e32m1(n));

    while (i < n) {
        int vl = __riscv_vsetvl_e32m1(n - i);
        vfloat32m1_t va = __riscv_vle32_v_f32m1(a + i, vl);
        vfloat32m1_t vb = __riscv_vle32_v_f32m1(b + i, vl);
        vfloat32m1_t vprod = __riscv_vfmul_vv_f32m1(va, vb, vl);
        vsum = __riscv_vfadd_vv_f32m1(vsum, vprod, vl);
        i += vl;
    }

    vfloat32m1_t vshift = __riscv_vslide1down_vf_f32m1(vsum, 0.0f, __riscv_vsetvl_e32m1(n));
    vfloat32m1_t vsum2 = __riscv_vfadd_vv_f32m1(vsum, vshift, __riscv_vsetvl_e32m1(n));
    vfloat32m1_t vshift2 = __riscv_vslide1down_vf_f32m1(vsum2, 0.0f, __riscv_vsetvl_e32m1(n));
    vfloat32m1_t vsum4 = __riscv_vfadd_vv_f32m1(vsum2, vshift2, __riscv_vsetvl_e32m1(n));
    vfloat32m1_t vshift3 = __riscv_vslide1down_vf_f32m1(vsum4, 0.0f, __riscv_vsetvl_e32m1(n));
    vfloat32m1_t vsum8 = __riscv_vfadd_vv_f32m1(vsum4, vshift3, __riscv_vsetvl_e32m1(n));
    vfloat32m1_t vshift4 = __riscv_vslide1down_vf_f32m1(vsum8, 0.0f, __riscv_vsetvl_e32m1(n));
    vfloat32m1_t vsum16 = __riscv_vfadd_vv_f32m1(vsum8, vshift4, __riscv_vsetvl_e32m1(n));
    vfloat32m1_t vshift5 = __riscv_vslide1down_vf_f32m1(vsum16, 0.0f, __riscv_vsetvl_e32m1(n));
    vfloat32m1_t vsum32 = __riscv_vfadd_vv_f32m1(vsum16, vshift5, __riscv_vsetvl_e32m1(n));
    vfloat32m1_t vshift6 = __riscv_vslide1down_vf_f32m1(vsum32, 0.0f, __riscv_vsetvl_e32m1(n));
    vfloat32m1_t vsum64 = __riscv_vfadd_vv_f32m1(vsum32, vshift6, __riscv_vsetvl_e32m1(n));
    vfloat32m1_t vshift7 = __riscv_vslide1down_vf_f32m1(vsum64, 0.0f, __riscv_vsetvl_e32m1(n));
    vfloat32m1_t vsum128 = __riscv_vfadd_vv_f32m1(vsum64, vshift7, __riscv_vsetvl_e32m1(n));
    vfloat32m1_t vshift8 = __riscv_vslide1down_vf_f32m1(vsum128, 0.0f, __riscv_vsetvl_e32m1(n));
    vfloat32m1_t vsum256 = __riscv_vfadd_vv_f32m1(vsum128, vshift8, __riscv_vsetvl_e32m1(n));

    sum = __riscv_vfmv_f_s_f32m1_f32(vsum256);
    out[0] = sum;
}
#include "harness.h"
