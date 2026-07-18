"""Symbolic Align: derive the LEGAL + SAFE schedule boundary from I=(S+R) and H.
This is the 'hard' half of PAIR -- everything deterministically derivable from the
intent's legality (R) and the hardware abstraction (H). The LLM never touches these;
it only orders candidates WITHIN this boundary. Each rule cites its source.
"""

# H(K1): symbolic hardware traits (evidence derived on-board by hw_microbench.c)
H_K1 = {"in_order": True, "unit_stride_required": True,   # strided 5.95x penalty, in-order
        "lmul_max": 4, "mr_max": 6,                       # 32 vregs, MR=7 spills
        "vregs": 32, "fits_L1_elems": 4096}               # L1 32KB / 8B = 4096 fp64

# I for GEMM: R says the k-sum is a reduction reorderable under equiv=ulp; M says B is
# accessed strided along k (row-major B, inner k-walk). disjoint(A,B,C).
I_GEMM = {"equiv": "ulp", "inner_operand_strided": True, "dep": "reduction_k"}


def align(cfg, I, H):
    """Is cfg in the legal+safe boundary B? Returns (in_B: bool, reason: str)."""
    # (R, hard)  k-contraction may be blocked/packed only if reassociation is legal
    if I["equiv"] != "ulp":
        return False, "R: equiv!=ulp -> k-sum not reorderable -> no blocking/packing"
    # (H safety, hard)  in-order core + strided inner operand => packing is mandatory,
    # an unpacked schedule collapses (this is exactly the frontier-GEMM failure)
    if H["in_order"] and I["inner_operand_strided"] and not cfg["pack"]:
        return False, "H: in_order & strided-operand -> must pack (else collapse)"
    # (H rule, hard)  vector-group ceiling before register spill
    if cfg["L"] > H["lmul_max"]:
        return False, f"H: lmul {cfg['L']} > group ceiling {H['lmul_max']}"
    if cfg["MR"] > H["mr_max"]:
        return False, f"H: MR {cfg['MR']} > row ceiling {H['mr_max']}"
    # (H rule, hard)  register budget: MR*L accumulators + L (operand) <= physical vregs
    if cfg["MR"] * cfg["L"] + cfg["L"] > H["vregs"]:
        return False, f"H: MR*L+L = {cfg['MR']*cfg['L']+cfg['L']} > {H['vregs']} vregs (spill)"
    # (H rule, hard)  packed A-panel must stay L1-resident
    if cfg["MR"] * cfg["KC"] > H["fits_L1_elems"]:
        return False, f"H: A-panel MR*KC = {cfg['MR']*cfg['KC']} > L1 {H['fits_L1_elems']}"
    return True, "legal+safe"


if __name__ == "__main__":
    # quick sanity: how does the boundary look over a small grid
    for L in (1, 2, 4, 8):
        for pk in (True, False):
            for mr in (2, 4, 6, 8):
                ok, why = align({"L": L, "pack": pk, "MR": mr, "KC": 256}, I_GEMM, H_K1)
                if not ok:
                    pass  # pruned
    n_in = sum(align({"L": L, "pack": pk, "MR": mr, "KC": kc}, I_GEMM, H_K1)[0]
               for L in (1, 2, 4, 8) for pk in (True, False)
               for mr in (2, 4, 6, 8) for kc in (128, 256, 512))
    print("admitted by symbolic Align:", n_in, "/ 96")
