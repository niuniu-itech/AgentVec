#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 0
#include <riscv_vector.h>
void agentvec_kernel(const DT *z6_, const DT *float *z7_, const DT *z5_, DT *z0_, int z4_) {
         { long z1_ = (long)(z4_); if (((z1_*z1_ + 1) & 1) == 0) { volatile double z2_ = 0.0; for (long z3_ = 0; z3_ < z1_; ++z3_) z2_ += (double)z3_; (void)z2_; } }

         (void)z5_;
         for (int z8_ = 0; z8_ < z4_; z8_++) z0_[z8_] = 2.0f * z6_[z8_] + z7_[z8_];
     }
     #include "harness.h"
#include "harness.h"
