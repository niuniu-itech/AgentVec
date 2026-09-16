"""SiliconFlow chat client with explicit model selection and request-scoped caching."""
import hashlib
import json
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.request

BASE = "https://api.siliconflow.cn/v1"
# Aliases retained for historical studies; availability is provider-dependent.
MODELS = {
    "DeepSeek-V4":  "deepseek-ai/DeepSeek-V4-Flash",
    "DeepSeek-V4-Pro": "deepseek-ai/DeepSeek-V4-Pro",
    "Qwen3.6-35B":  "Qwen/Qwen3.6-35B-A3B",
    "Qwen3.6-35B-A3B": "Qwen/Qwen3.6-35B-A3B",
    "GLM-5":        "Pro/zai-org/GLM-5",
    "GLM-5.2":      "zai-org/GLM-5.2",
    "Kimi-K2.5":    "Pro/moonshotai/Kimi-K2.5",
}


def _settings():
    base = os.environ.get("SILICONFLOW_BASE_URL", BASE).rstrip("/")
    cache = Path((os.environ.get("AGENTVEC_CACHE_DIR") or str(Path.home() / ".cache" / "agentvec")))
    return base, cache


def _cache_path(model_id, prompt, max_tokens, temperature=0.0, thinking=False):
    base, cache = _settings()
    record = dict(endpoint=base, model=model_id, prompt=prompt, max_tokens=max_tokens,
                  temperature=temperature, thinking=thinking)
    digest = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    return cache / (digest + ".json")


def chat(model_id, prompt, max_tokens=3000, temperature=0.0, timeout=180, use_cache=True, thinking=False):
    """Return completion text; never substitute another model after an API error."""
    if not model_id:
        raise ValueError("an explicit model ID is required")
    base, _ = _settings()
    cache_path = _cache_path(model_id, prompt, max_tokens, temperature, thinking)
    if use_cache and cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))["content"]
    key = os.environ.get("SILICONFLOW_API_KEY") or os.environ.get("SF_KEY")
    if not key:
        raise RuntimeError("Set SILICONFLOW_API_KEY (or SF_KEY) in your environment")
    payload = dict(model=model_id, messages=[dict(role="user", content=prompt)],
                   max_tokens=max_tokens, temperature=temperature, enable_thinking=thinking)
    delay = float(os.environ.get("SF_DELAY", "0"))
    if delay:
        time.sleep(min(max(delay, 0), 30))
    for attempt in range(3):
        request = urllib.request.Request(base + "/chat/completions", data=json.dumps(payload).encode(),
                  headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            started = time.monotonic()
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.load(response)
            content = data["choices"][0]["message"].get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("provider returned no completion text")
            if use_cache:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                record = dict(content=content, requested_model=model_id, response_model=data.get("model"),
                              usage=data.get("usage"), elapsed_s=time.monotonic()-started,
                              endpoint=base, temperature=temperature, max_tokens=max_tokens, thinking=thinking)
                cache_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
            return content
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise RuntimeError(f"chat request failed with HTTP {exc.code}; model was not changed") from None
            time.sleep(2 ** attempt)
    raise RuntimeError("chat request failed")

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
    """Accept a JSON object, optionally enclosed in one Markdown code fence."""
    if not isinstance(text, str):
        raise ValueError("completion is not text")
    value = text.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", value, re.S)
    if fence:
        value = fence.group(1)
    try:
        result = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("completion must contain one JSON object") from exc
    if not isinstance(result, dict):
        raise ValueError("completion must be a JSON object")
    return result


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
