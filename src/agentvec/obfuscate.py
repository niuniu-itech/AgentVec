"""Semantics-preserving obfuscation of C kernels (the paper's Sec 4.1 transforms),
used to STRIP the obvious intent cues so that intent recovery is non-trivial.

  IM  Identifier Mangling : rename meaningful identifiers -> opaque tokens.
  DCI Dead-Code Insertion : inject inert blocks under always-false opaque predicates.
  T   Type upconversion   : alias the element type behind a typedef.

These preserve behavior (verified by recompiling and diffing outputs). After
obfuscation, names/types no longer reveal that a kernel is, say, axpy; the model
must rely on data-flow, not surface cues.
"""
import re, sys

C_KEYWORDS = {
    "int","double","float","void","for","while","if","else","return","size_t","const",
    "struct","typedef","unsigned","long","char","static","inline","sizeof","BLASLONG",
    "do","switch","case","break","continue","FLOAT","register","volatile","int32_t","int64_t",
}
PP_WORDS = {"define","include","ifndef","ifdef","endif","pragma","undef","elif","error","line","if"}
PROTECTED = {"agentvec_kernel","main","printf","malloc","free","memcpy",
             "DT","DT_IS_FLOAT","REDUCE","harness",
             # libc math: these are OPERATIONS, not cosmetic name cues -> keep visible
             "fabs","fabsf","sqrt","sqrtf","exp","expf","log","logf","pow","powf",
             "fmin","fminf","fmax","fmaxf","sin","sinf","cos","cosf","tanh","tanhf","fma","fmaf"}


def _mask(src):
    """Replace string/char literals, comments, and preprocessor lines with placeholders."""
    ph = {}
    def sub(m):
        k = f"__MASK{len(ph)}__"; ph[k] = m.group(0); return k
    s = re.sub(r"/\*.*?\*/", sub, src, flags=re.S)      # block comments
    s = re.sub(r"//[^\n]*", sub, s)                       # line comments
    s = re.sub(r"(?m)^\s*#.*$", sub, s)                  # preprocessor lines (before strings)
    s = re.sub(r'"[^"]*"', sub, s)                        # remaining string literals
    return s, ph


def _unmask(s, ph):
    for _ in range(6):                                    # restore nested placeholders to fixpoint
        for k, v in ph.items():
            s = s.replace(k, v)
        if "__MASK" not in s:
            break
    return s


def identifier_mangling(src: str) -> str:
    masked, ph = _mask(src)
    skip = C_KEYWORDS | PP_WORDS | PROTECTED
    cands = []
    for i in sorted(set(re.findall(r"\b[A-Za-z_]\w*\b", masked)), key=len, reverse=True):
        if i in skip or i.startswith("__MASK") or i.startswith("__riscv") or i.startswith("riscv_"):
            continue
        if re.match(r"^v(float|int|uint|bool)\w*", i):   # rvv vector types
            continue
        cands.append(i)
    mapping, k = {}, 0
    for i in cands:
        if i not in mapping:
            mapping[i] = f"z{k}_"; k += 1
    out = masked
    for o, n in mapping.items():
        out = re.sub(rf"\b{re.escape(o)}\b", n, out)
    return _unmask(out, ph)


def dead_code_insertion(src: str) -> str:
    dead = ("    { long _o = (long)(n); if (((_o*_o + 1) & 1) == 0) { "
            "volatile double _d = 0.0; for (long _q = 0; _q < _o; ++_q) _d += (double)_q; "
            "(void)_d; } }\n")
    m = re.search(r"(agentvec_kernel\s*\([^)]*\)\s*\{)", src)
    return src[:m.end()] + "\n" + dead + src[m.end():] if m else src


def type_upconversion(src: str) -> str:
    """Alias a raw element type behind a typedef (only when the kernel declares it
    directly, e.g. real OpenBLAS kernels using `double`). DT-macro kernels already
    alias their type, so T is a no-op there."""
    if "#define DT" in src:           # difftest kernels already alias via DT
        return src
    if re.search(r"\bdouble\b", src) and "typedef" not in src:
        body = re.sub(r"\bdouble\b", "real_t", src)
        return "typedef double real_t;\n" + body
    return src


def strip_comments(src: str) -> str:
    s = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    s = re.sub(r"//[^\n]*", "", s)
    return s


def obfuscate(src: str, im=True, dci=True, t=True) -> str:
    out = strip_comments(src)        # comments leak intent -> remove them
    if t:   out = type_upconversion(out)
    if dci: out = dead_code_insertion(out)
    if im:  out = identifier_mangling(out)
    return out


if __name__ == "__main__":
    sys.stdout.write(obfuscate(open(sys.argv[1]).read()))
