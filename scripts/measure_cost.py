"""Measure the COST of producing a kernel: tokens + wall-clock, pure-LLM (writes the
whole RVV kernel) vs AgentVec arm (writes only the structured intent). SiliconFlow
returns exact token usage. thinking=False for fairness. [i/N] progress."""
import os, sys, json, time, urllib.request
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lab"))
import llm_backend as LB, obfuscate

BASE = "https://api.siliconflow.cn/v1"
ORACLE = {
    "dot":  "void k(const float*a,const float*b,float*out,int n){float s=0;for(int i=0;i<n;i++)s+=a[i]*b[i];out[0]=s;}",
    "asum": "void k(const float*a,float*out,int n){float s=0;for(int i=0;i<n;i++)s+=fabsf(a[i]);out[0]=s;}",
    "nrm2": "void k(const float*a,float*out,int n){float s=0;for(int i=0;i<n;i++)s+=a[i]*a[i];out[0]=sqrtf(s);}",
}
MODELS = {k: LB.MODELS[k] for k in ("DeepSeek-V4", "Qwen3.6-35B", "GLM-5")}


def call(model_id, prompt, max_tokens):
    key = os.environ["SF_KEY"]
    body = json.dumps({"model": model_id, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0.0, "enable_thinking": False}).encode()
    req = urllib.request.Request(BASE + "/chat/completions", data=body,
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    t = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=120))
    dt = time.time() - t
    u = r.get("usage", {})
    return u.get("completion_tokens", 0), u.get("total_tokens", 0), dt


def main():
    ops = list(ORACLE); total = len(MODELS) * len(ops) * 2; idx = 0
    agg = {"pure": {"comp": [], "tot": [], "t": []}, "av": {"comp": [], "tot": [], "t": []}}
    for short, mid in MODELS.items():
        for op in ops:
            src = ORACLE[op]
            for arm, prompt, mx in (
                ("pure", LB.PURE_LLM_PROMPT.format(src=src), 2000),
                ("av",   LB.AIR_PROMPT.format(src=src), 600)):
                idx += 1
                print(f"[{idx}/{total}] {short} {op} {arm} ...", flush=True)
                try:
                    comp, tot, dt = call(mid, prompt, mx)
                    agg[arm]["comp"].append(comp); agg[arm]["tot"].append(tot); agg[arm]["t"].append(dt)
                    print(f"    completion={comp} total={tot} time={dt:.1f}s", flush=True)
                except Exception as e:
                    print(f"    ERR {type(e).__name__} {str(e)[:60]}", flush=True)
    def avg(x): return sum(x) / len(x) if x else 0
    print("\n=== COST per kernel (avg over 3 models x 3 reduction ops) ===")
    for arm, name in (("pure", "Pure-LLM (writes RVV kernel)"), ("av", "AgentVec (writes intent only)")):
        a = agg[arm]
        print(f"{name:32} completion_tokens={avg(a['comp']):6.0f}  total_tokens={avg(a['tot']):6.0f}  time={avg(a['t']):5.1f}s")
    if agg["pure"]["comp"] and agg["av"]["comp"]:
        print(f"\nAgentVec uses {avg(agg['pure']['comp'])/max(avg(agg['av']['comp']),1):.1f}x FEWER completion tokens, "
              f"{avg(agg['pure']['t'])/max(avg(agg['av']['t']),0.1):.1f}x less time.")
    json.dump(agg, open(os.path.join(os.path.dirname(__file__), "_cost.json"), "w"))


if __name__ == "__main__":
    main()
