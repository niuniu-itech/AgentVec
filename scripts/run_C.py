"""HARD PROBLEM 3, part 2 -- C (symbolic Align + LLM ordering).
B already pruned A(96) to 27 legal+safe configs and kept the optimum. C asks the
HONEST residual question: can the LLM ORDER those 27 so the autotuner hits the
optimum in far fewer MEASURED trials? (Its marginal value is faster convergence,
NOT more pruning -- symbolic already did the pruning.)
We reuse the board GFLOP/s already measured for all 27 (in _baseline_B.json), so C
needs only LLM calls. Metric: trials-to-optimum under LLM order vs B(27) vs A(96).
Run:  SF_KEY=... python -u run_C.py
"""
import os, sys, json
import llm_backend as LB

HERE = os.path.dirname(__file__)
B = json.load(open(os.path.join(HERE, "_baseline_B.json")))
A_OPT = B["__summary__"]["A_opt_gflops"]            # 2.740
# the 27 admitted, with their measured GFLOP/s
CFG = [{"id": i, "L": r["cfg"]["L"], "MR": r["cfg"]["MR"], "KC": r["cfg"]["KC"], "gflops": r["gflops"]}
       for i, (t, r) in enumerate(sorted((t, r) for t, r in B.items()
                                          if isinstance(r, dict) and r.get("in_B") and r.get("status") == "ok"))]
assert len(CFG) == 27, f"expected 27 admitted configs, got {len(CFG)}"

H = ("Target hardware H (abstracted): in-order core (no OoO latency hiding); fp64 FMA is the binding "
     "resource at ~1/cycle so FEWER instructions per MAC wins; 32 vector registers; L1=32KB; "
     "all listed configs are already legal & L1-resident & register-safe (packed).")
INTENT = "GEMM C=A*B (f64), packed register-tiled. You ONLY rank schedules; you do not change the math."


def prompt():
    lines = "\n".join(f"  id={c['id']}: lmul={c['L']}, MR={c['MR']} rows, KC={c['KC']}" for c in CFG)
    return f"""{INTENT}

{H}

Here are 27 legal+safe schedule configs. Rank ALL of them best-first by EXPECTED GFLOP/s on H,
reasoning from H (which lmul/MR/KC this in-order, issue-bound core will run fastest).
Configs:
{lines}

Output ONLY JSON: {{"ranked_ids": [<all 27 ids, best first>], "why_top": "<1 sentence: why your #1>"}}"""


def convergence(order):
    """best-so-far GFLOP/s after trying configs in `order`; rank where optimum first reached."""
    gf = {c["id"]: c["gflops"] for c in CFG}
    best, curve, rank_opt = 0.0, [], None
    for k, cid in enumerate(order, 1):
        best = max(best, gf.get(cid, 0.0))
        curve.append(round(best, 3))
        if rank_opt is None and best >= 0.99 * A_OPT:
            rank_opt = k
    return curve, rank_opt


def main():
    models = [m for m in ("DeepSeek-V4", "Qwen3.6-35B", "GLM-5") if m in LB.MODELS]
    out = {"A_opt": A_OPT, "n_admitted": len(CFG), "models": {}}
    for model in models:
        try:
            j = LB.extract_json(LB.chat(LB.MODELS[model], prompt(), max_tokens=1500, timeout=160))
            order = [int(x) for x in (j or {}).get("ranked_ids", [])]
        except Exception as e:
            out["models"][model] = {"error": str(e)[:60]}; print(f"{model}: ERR {e}"); continue
        # append any missing ids (model dropped some) so convergence is well-defined
        order = order + [c["id"] for c in CFG if c["id"] not in order]
        curve, rank_opt = convergence(order)
        top = sorted([c for c in CFG if c["id"] == order[0]], key=lambda c: -c["gflops"])
        out["models"][model] = {"ranked": order, "rank_to_opt": rank_opt,
                                "best_top1": curve[0], "best_top3": max(curve[:3]),
                                "best_top5": max(curve[:5]), "why_top": (j or {}).get("why_top", ""),
                                "top1_cfg": top[0] if top else None}
        print(f"[{model}] rank_to_opt={rank_opt}/27  top1={curve[0]}  top3={max(curve[:3])}  "
              f"top5={max(curve[:5])}  (A_opt={A_OPT})  why='{(j or {}).get('why_top','')[:70]}'", flush=True)
    json.dump(out, open(os.path.join(HERE, "_C_ordering.json"), "w"), indent=2, ensure_ascii=False)

    print("\n============ THREE-WAY: trials to reach the optimum (2.740 GFLOP/s) ============")
    print(f"  A  full search        : up to 96 trials (whole inflated space)")
    print(f"  B  symbolic Align      : 27 trials (exhaustive within the legal+safe boundary)")
    for model in models:
        m = out["models"].get(model, {})
        if "rank_to_opt" in m:
            print(f"  C  +{model:12} ordering: {m['rank_to_opt']} trials  "
                  f"(best@top1={m['best_top1']}, @top3={m['best_top3']})")
    print("\nHonest read: C's value is FASTER CONVERGENCE within B, not more pruning.")
    print("saved _C_ordering.json")


if __name__ == "__main__":
    main()
