import io
import json
import urllib.error
import pytest
from agentvec import llm_backend as backend
from agentvec.schedules import enumerate_schedules, validate_ranking
from agentvec.symbolic_align import align, H_K1, I_GEMM


def test_cache_distinguishes_decoding_and_endpoint(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENTVEC_CACHE_DIR", str(tmp_path))
    first = backend._cache_path("model", "prompt", 128, 0.0)
    assert first != backend._cache_path("model", "prompt", 128, 0.5)
    assert first != backend._cache_path("model", "prompt", 128, 0.0, True)
    monkeypatch.setenv("SILICONFLOW_BASE_URL", "https://example.invalid/v1")
    assert first != backend._cache_path("model", "prompt", 128, 0.0)


def test_api_does_not_silently_switch_models(monkeypatch):
    monkeypatch.setenv("SILICONFLOW_API_KEY", "test-key")
    calls = []
    def fail(request, **kwargs):
        calls.append(json.loads(request.data)["model"])
        raise urllib.error.HTTPError(request.full_url, 404, "model not found", {}, None)
    monkeypatch.setattr(backend.urllib.request, "urlopen", fail)
    with pytest.raises(RuntimeError, match="model was not changed"):
        backend.chat("requested-model", "prompt", use_cache=False)
    assert calls == ["requested-model"]


def test_cached_record_keeps_provider_metadata_without_key(monkeypatch, tmp_path):
    monkeypatch.setenv("SILICONFLOW_API_KEY", "secret-test-value")
    monkeypatch.setenv("AGENTVEC_CACHE_DIR", str(tmp_path))
    result = dict(model="resolved-model", usage=dict(completion_tokens=3), choices=[dict(message=dict(content='{"ok":true}'))])
    monkeypatch.setattr(backend.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(json.dumps(result).encode()))
    assert backend.chat("requested", "prompt") == '{"ok":true}'
    text = next(tmp_path.glob("*.json")).read_text()
    assert "secret-test-value" not in text
    assert json.loads(text)["response_model"] == "resolved-model"


@pytest.mark.parametrize("text", ['{} {}', '[]', 'answer: {"a": 1}', '{bad json}'])
def test_malformed_completion_is_rejected(text):
    with pytest.raises(ValueError):
        backend.extract_json(text)


def test_ranker_cannot_invent_or_repeat_candidates():
    result = validate_ranking([2, 999, 2, True, "1"], [0, 1, 2])
    assert result["order"] == [2, 0, 1]
    assert result["ignored"] == [999, 2, True, "1"]
    assert result["appended"] == [0, 1]


def test_hardware_boundary_has_27_admitted_schedules():
    candidates = enumerate_schedules()
    assert len(candidates) == 27
    assert all(align(c["schedule"], I_GEMM, H_K1)[0] for c in candidates)
    assert not align(dict(L=-1, MR=4, KC=128, pack=True), I_GEMM, H_K1)[0]
