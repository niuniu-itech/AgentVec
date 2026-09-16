import json
from agentvec.pipeline import main, oracle, accepts_test
from agentvec import llm_backend


def test_manual_generation_is_checked_not_verified(tmp_path):
    output = tmp_path / "manual"
    assert main(["--output", str(output)]) == 0
    report = json.loads((output / "migration.json").read_text())
    assert report["verified_count"] == 0
    assert len(report["cases"]) == 6
    assert all(case["status"] == "CHECKED" for case in report["cases"].values())
    assert (output / "runner.py").exists()


def test_invalid_llm_proposal_is_vetoed_without_generated_candidate(monkeypatch, tmp_path):
    monkeypatch.setattr(llm_backend, "chat", lambda *a, **k: '{}')
    output = tmp_path / "veto"
    assert main(["--backend", "siliconflow", "--model", "test", "--ops", "saxpy", "--output", str(output)]) == 1
    report = json.loads((output / "migration.json").read_text())
    assert report["cases"]["saxpy"]["status"] == "VETO"
    assert not list(output.rglob("rvv.c"))


def test_independent_scalar_oracles_do_not_require_vector_extension():
    for op in ("dot", "asum", "nrm2", "saxpy", "scal", "copy"):
        assert "riscv_vector.h" not in oracle(op)


def test_verified_flag_needs_complete_checks_and_matching_sources():
    record = dict(candidate_sha256="candidate", source_sha256="oracle", spec_sha256="spec")
    test = dict(verified=True, planned=12, total=12, passed=12,
                source_sha256={"rvv.c": "candidate", "scalar.c": "oracle", "spec.json": "spec"})
    assert accepts_test(record, test, 0)
    assert not accepts_test(record, dict(test, passed=11), 0)
    assert not accepts_test(record, dict(test, planned=0, total=0, passed=0), 0)
    assert not accepts_test(record, dict(test, source_sha256={"rvv.c": "other"}), 0)
    assert not accepts_test(record, test, 2)
