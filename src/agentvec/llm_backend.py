"""Multi-model LLM backend over SiliconFlow (OpenAI-compatible) for the
pure-LLM-vs-AgentVec study. Key is read from env SF_KEY (never written to disk)."""
import os, json, re, time, urllib.request

BASE = "https://api.siliconflow.cn/v1"
MODELS = {
    "DeepSeek-V4":  "deepseek-ai/DeepSeek-V4-Flash",
    "DeepSeek-V4-Pro": "deepseek-ai/DeepSeek-V4-Pro",
    "Qwen3.6-35B":  "Qwen/Qwen3.6-35B-A3B",
    "Qwen3.6-35B-A3B": "Qwen/Qwen3.6-35B-A3B",
    "GLM-5":        "Pro/zai-org/GLM-5",
    "GLM-5.2":      "zai-org/GLM-5.2",
    "Kimi-K2.5":    "Pro/moonshotai/Kimi-K2.5",
}

MODEL_FALLBACKS = {
    "deepseek-ai/DeepSeek-V4-Flash": "deepseek-ai/DeepSeek-V4-Pro",
    "Pro/zai-org/GLM-5": "zai-org/GLM-5.2",
}


_CACHE_DIR = os.path.join(os.path.dirname(__file__), "_llm_cache")


def _cache_path(model_id, prompt, max_tokens):
    import hashlib
    h = hashlib.sha256(f"{model_id}|{max_tokens}|{prompt}".encode()).hexdigest()[:24]
    return os.path.join(_CACHE_DIR, h + ".txt")


def chat(model_id, prompt, max_tokens=3000, temperature=0.0, timeout=180, use_cache=True, thinking=False):
    # thinking=False => uniform direct-output mode for a fair pure-LLM comparison across
    # thinking/non-thinking models. cache keyed on (model, max_tokens, thinking, prompt).
    if use_cache:
        cp = _cache_path(model_id, f"think{int(thinking)}|" + prompt, max_tokens)
        if os.path.exists(cp):
            return open(cp, encoding="utf-8").read()
    delay = float(os.environ.get("SF_DELAY", "0"))
    if delay > 0:
        time.sleep(delay)
    key = os.environ["SF_KEY"]
    current_model = model_id
    for attempt in range(6):
        try:
            payload = {"model": current_model, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": temperature,
                       "enable_thinking": thinking}          # SiliconFlow hybrid-model switch
            body = json.dumps(payload).encode()
            req2 = urllib.request.Request(BASE + "/chat/completions", data=body,
                                          headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req2, timeout=timeout))
            msg = r["choices"][0]["message"]
            content = msg.get("content") or msg.get("reasoning_content") or ""
            if use_cache and content:
                os.makedirs(_CACHE_DIR, exist_ok=True)
                open(_cache_path(model_id, f"think{int(thinking)}|" + prompt, max_tokens), "w", encoding="utf-8").write(content)
            return content
        except Exception as e:
            if attempt == 5:
                raise
            msg = str(e)
            if current_model in MODEL_FALLBACKS and any(x in msg.lower() for x in ("404", "not found", "model")):
                current_model = MODEL_FALLBACKS[current_model]
                continue
            if "429" in msg or "Too Many Requests" in msg:
                time.sleep(float(os.environ.get("SF_429_SLEEP", "45")) * (attempt + 1))
            else:
                time.sleep(3 + attempt * 3)


def extract_c(text):
    """Pull the C function from a model reply (prefer a ```c block; else the function)."""
    m = re.search(r"```(?:c|cpp|C)?\s*(.*?)```", text, re.S)
    code = m.group(1) if m else text
    # keep from the first #include or function signature
    i = code.find("#include")
    j = code.find("void agentvec_kernel")
    start = min([x for x in (i, j) if x >= 0] or [0])
    return code[start:].strip()


def extract_json(text):
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S) or re.search(r"(\{.*\})", text, re.S)
    return json.loads(m.group(1)) if m else None


AIR_PROMPT = """Analyze this CPU kernel and recover ONLY its algorithmic intent (ignore obfuscated
names; reason from data flow).
```c
{src}
```
The inputs are the arrays in the source's parameter order, named a, b, c; the output array is out,
length n. Output ONLY a JSON object, no prose:
- elementwise map: {{"pattern":"map","dtype":"f32","formula":"out = <expr over a,b,c, constants, + - * abs()>"}}
- reduction:       {{"pattern":"reduce","dtype":"f32","reduce_op":"sum|max|min","elem":"<per-element expr: a, a*b, abs(a), or a*a>","postproc":"none|sqrt"}}
Examples:
  out = 2*a + b            -> {{"pattern":"map","dtype":"f32","formula":"out = 2*a + b"}}
  dot product sum(a*b)     -> {{"pattern":"reduce","dtype":"f32","reduce_op":"sum","elem":"a*b","postproc":"none"}}
  sum of abs values        -> {{"pattern":"reduce","dtype":"f32","reduce_op":"sum","elem":"abs(a)","postproc":"none"}}
  euclidean norm sqrt(sum a*a) -> {{"pattern":"reduce","dtype":"f32","reduce_op":"sum","elem":"a*a","postproc":"sqrt"}}
JSON only."""


PURE_LLM_PROMPT = """You are migrating a CPU kernel to RISC-V Vector (RVV 1.0), vector-length-agnostic.

Source kernel (semantics-preserving; names may be obfuscated):
```c
{src}
```

Rewrite it as EXACTLY this function, computing the SAME result:
  void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n)
The inputs are arrays a,b,c (in the source's parameter order) and output array out of length n.
Requirements:
- Use <riscv_vector.h> intrinsics with __riscv_vsetvl_e32m1 strip-mining so it works at ANY VLEN
  (vector-length-agnostic). Do NOT hardcode a lane count.
- Output ONLY the function in a ```c code block: it must start with the includes you need and the
  exact signature above. No explanation."""
