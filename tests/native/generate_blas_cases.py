"""Generate compile coverage for f64 BLAS emitters and runtime reduction checks."""
import argparse
from pathlib import Path
from agentvec.blas_lower import EMITTERS, L1Spec, emit_l1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    code = [emit_l1(L1Spec("check_" + kind, "f64", kind)).replace(
        '#include "common.h"', 'typedef long BLASLONG;') for kind in EMITTERS]
    code.append(r'''
int main(void) {
    double a[516], b[516];
    int checks = 0, failures = 0;
    for (int i = 0; i < 516; i++) { a[i] = (i % 17) - 8; b[i] = (i % 11) - 5; }
    for (int stride = 1; stride <= 2; stride++) {
        for (int n = 0; n <= 258; n++) {
            double dot = 0, asum = 0, squares = 0, lo = INFINITY, hi = -INFINITY;
            for (int i = 0; i < n; i++) {
                double x = a[i*stride];
                dot += x*b[i*stride]; asum += fabs(x); squares += x*x;
                lo = fmin(lo, x); hi = fmax(hi, x);
            }
            failures += check_dot(n, a, stride, b, stride) != dot; checks++;
            failures += check_asum(n, a, stride) != asum; checks++;
            failures += fabs(check_nrm2(n, a, stride)-sqrt(squares)) > 1e-12; checks++;
            failures += check_max(n, a, stride) != (n ? hi : 0); checks++;
            failures += check_min(n, a, stride) != (n ? lo : 0); checks++;
        }
    }
    printf("f64 reduction checks=%d failures=%d\n", checks, failures);
    return failures ? 1 : 0;
}
''')
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(code), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
