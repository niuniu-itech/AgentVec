#include <riscv_vector.h>
#include <math.h>

void layernorm(int n, const float *x, float *y) {
    // 1. Compute mean
    float sum = 0.0f;
    int i = 0;
    size_t vl;
    while (i < n) {
        vl = __riscv_vsetvl_e32m8(n - i);
        vfloat32m8_t vx = __riscv_vle32_v_f32m8(x + i, vl);
        vfloat32m1_t vsum = __riscv_vfredusum_vs_f32m8_f32m1(vx, __riscv_vfmv_s_f_f32m1(0.0f, 1), vl);
        sum += __riscv_vfmv_f_s_f32m1_f32(vsum);
        i += vl;
    }
    float mean = sum / n;

    // 2. Compute variance
    float var_sum = 0.0f;
    i = 0;
    vfloat32m1_t vmean = __riscv_vfmv_v_f_f32m1(mean, 1);
    while (i < n) {
        vl = __riscv_vsetvl_e32m8(n - i);
        vfloat32m8_t vx = __riscv_vle32_v_f32m8(x + i, vl);
        vfloat32m8_t vdiff = __riscv_vfsub_vv_f32m8(vx, __riscv_vfmv_v_f_f32m8(mean, vl), vl);
        vfloat32m8_t vsq = __riscv_vfmul_vv_f32m8(vdiff, vdiff, vl);
        vfloat32m1_t vred = __riscv_vfredusum_vs_f32m8_f32m1(vsq, __riscv_vfmv_s_f_f32m1(0.0f, 1), vl);
        var_sum += __riscv_vfmv_f_s_f32m1_f32(vred);
        i += vl;
    }
    float var = var_sum / n;

    // 3. Compute rstd and apply normalization
    float rstd = 1.0f / sqrtf(var + 1e-5f);
    i = 0;
    while (i < n) {
        vl = __riscv_vsetvl_e32m8(n - i);
        vfloat32m8_t vx = __riscv_vle32_v_f32m8(x + i, vl);
        vfloat32m8_t vdiff = __riscv_vfsub_vf_f32m8(vx, mean, vl);
        vfloat32m8_t vy = __riscv_vfmul_vf_f32m8(vdiff, rstd, vl);
        __riscv_vse32_v_f32m8(y + i, vy, vl);
        i += vl;
    }
}