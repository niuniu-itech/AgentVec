
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 0
void agentvec_kernel(const DT *z4_, const DT *z5_, const DT *z8_, DT *z0_, int z6_) {
    { long z1_ = (long)(z6_); if (((z1_*z1_ + 1) & 1) == 0) { volatile double z3_ = 0.0; for (long z2_ = 0; z2_ < z1_; ++z2_) z3_ += (double)z2_; (void)z3_; } }

    (void)z8_;
    for (int z7_ = 0; z7_ < z6_; z7_++) z0_[z7_] = 2.0f * z4_[z7_] + z5_[z7_];
}
#include "harness.h"
