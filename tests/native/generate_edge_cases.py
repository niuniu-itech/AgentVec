"""Generate a native RVV test for vector tails, identities, abs and i32 wrapping."""
import argparse
from pathlib import Path
from agentvec.air import AIR, CPhy, Dep, Expr, Pattern, SAlgo
from agentvec.air_from_formula import air_from_intent
from agentvec.lowering import emit_rvv_kernel


def source():
    cases = {
        "map_abs": dict(pattern="map", dtype="f32", formula="abs(a)"),
        "reduce_sum": dict(pattern="reduce", dtype="f32", reduce_op="sum", elem="a"),
        "reduce_min": dict(pattern="reduce", dtype="f32", reduce_op="min", elem="a"),
        "reduce_max": dict(pattern="reduce", dtype="f32", reduce_op="max", elem="a"),
        "integer_map": dict(pattern="map", dtype="i32", formula="a*3+b"),
        "integer_sum": dict(pattern="reduce", dtype="i32", reduce_op="sum", elem="a"),
    }
    code = []
    for name, intent in cases.items():
        emitted = emit_rvv_kernel(air_from_intent(intent))
        emitted = emitted.replace("agentvec_kernel", name).replace('#include "harness.h"', '')
        code.append("#undef DT\n#undef DT_IS_FLOAT\n#undef REDUCE\n" + emitted)
    code.append(r'''
#include <stdio.h>
#include <string.h>
int main(void) {
    float a[258], b[258] = {0}, c[258] = {0}, out[258];
    int32_t ia[258], ib[258], ic[258] = {0}, io[258];
    int failures = 0, checks = 0;
    for (int i = 0; i < 258; i++) {
        a[i] = (float)((i % 17) - 8);
        ia[i] = INT32_MAX - i;
        ib[i] = i + 7;
    }
    for (int n = 0; n <= 258; n++) {
        float sum = 0, lo = INFINITY, hi = -INFINITY;
        uint32_t isum = 0;
        for (int i = 0; i < n; i++) {
            sum += a[i]; lo = fminf(lo, a[i]); hi = fmaxf(hi, a[i]);
            isum += (uint32_t)ia[i];
        }
        reduce_sum(a, b, c, out, n); failures += out[0] != sum; checks++;
        reduce_min(a, b, c, out, n); failures += out[0] != lo; checks++;
        reduce_max(a, b, c, out, n); failures += out[0] != hi; checks++;
        integer_sum(ia, ib, ic, io, n); failures += (uint32_t)io[0] != isum; checks++;
        map_abs(a, b, c, out, n);
        integer_map(ia, ib, ic, io, n);
        for (int i = 0; i < n; i++) {
            failures += out[i] != fabsf(a[i]); checks++;
            failures += (uint32_t)io[i] != (uint32_t)ia[i]*3u + (uint32_t)ib[i]; checks++;
        }
    }
    for (int i = 0; i < 258; i++) a[i] = -3.3e38f;
    reduce_max(a, b, c, out, 258); failures += out[0] != -3.3e38f; checks++;
    for (int i = 0; i < 258; i++) a[i] = 3.3e38f;
    reduce_min(a, b, c, out, 258); failures += out[0] != 3.3e38f; checks++;
    printf("checks=%d failures=%d\n", checks, failures);
    return failures ? 1 : 0;
}
''')
    return "\n".join(code)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source(), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
