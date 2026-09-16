"""End-to-end proposal handling without an accelerator or live model."""
import json

from agentvec.dag_pipeline import main


def test_emit_custom_branch_graph(tmp_path):
    contract = tmp_path / 'contract.json'
    contract.write_text(json.dumps(dict(inputs={"x":"n","gain":"scalar"},outputs={"y":"n","s":"scalar"})))
    proposal = tmp_path / 'proposal.json'
    proposal.write_text(json.dumps(dict(pattern="dag", nodes=[
        dict(name="y",pattern="map",formula="x*gain+1"),
        dict(name="s",pattern="reduce",formula="y",reduce_op="sum")
    ])))
    output = tmp_path / 'run'
    assert main(['--contract',str(contract),'--proposal',str(proposal),'--output',str(output)]) == 0
    record = json.loads((output/'migration.json').read_text())
    assert record['status'] == 'CHECKED'
    assert not (output/'reference.c').exists()


def test_model_result_is_used_instead_of_the_example_graph(tmp_path,monkeypatch):
    from agentvec import llm_backend
    monkeypatch.setattr(llm_backend,'chat',lambda *a,**kw: '{"pattern":"dag","nodes":[{"name":"y","pattern":"map","formula":"x+17"}]}')
    out = tmp_path/'proposal'
    assert main(['--example','softmax','--backend','siliconflow','--model','test', '--output',str(out)]) == 0
    record = json.loads((out/'air.json').read_text())
    assert record['nodes'][0]['formula'] == 'x+17'
    # Static checks establish shape/storage legality, not recovered source semantics.
    assert record['v_meta']['status'] == 'CHECKED'
    assert 'exp(' in (out/'reference.c').read_text()


def test_invalid_model_dag_is_vetoed_without_fallback(tmp_path,monkeypatch):
    from agentvec import llm_backend
    monkeypatch.setattr(llm_backend,'chat',lambda *a,**kw: '{"pattern":"dag","nodes":[{"name":"y","pattern":"map","formula":"unknown"}]}')
    out = tmp_path/'veto'
    assert main(['--example','softmax','--backend','siliconflow','--model','test','--output',str(out)]) == 1
    assert json.loads((out/'migration.json').read_text())['status'] == 'VETO'
    assert not (out/'candidate.c').exists()


def test_custom_contract_and_local_static_example_have_distinct_oracles(tmp_path):
    out = tmp_path/'example'
    assert main(['--example','rmsnorm','--target','scalar','--output',str(out)]) == 0
    assert (out/'candidate.c').read_text() != (out/'reference.c').read_text()
    assert json.loads((out/'migration.json').read_text())['status'] == 'CHECKED'
