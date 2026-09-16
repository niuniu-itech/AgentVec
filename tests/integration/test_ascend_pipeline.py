"""Exercise Ascend proposal-to-export boundaries without a CANN installation."""
import json

from agentvec.ascend.pipeline import main
from agentvec.ascend.registry import recover


def test_registered_export_reports_vetoes_separately(tmp_path):
    assert main(['--output', str(tmp_path)]) == 0
    cases = json.loads((tmp_path/'migration.json').read_text())['cases']
    assert sum(r['status']=='CHECKED' for r in cases.values()) == 10
    assert {op for op, r in cases.items() if r['status']=='VETO'} == {'strsv','strsm'}
    assert not (tmp_path/'strsv').exists()


def test_wrong_model_algorithm_is_not_replaced(tmp_path, monkeypatch):
    algorithm = json.loads(recover('sscal').to_json())['s_algo']
    algorithm['elem_op'] = {'op':'load','src':'x'}
    monkeypatch.setattr('agentvec.llm_backend.chat', lambda *a, **k: json.dumps({'s_algo':algorithm}))
    main(['--ops','sscal','--backend','siliconflow','--model','test-model','--output',str(tmp_path)])
    record = json.loads((tmp_path/'migration.json').read_text())['cases']['sscal']
    assert record['status'] == 'VETO'
    assert not (tmp_path/'sscal').exists()
    assert (tmp_path/'sscal.response.txt').exists()


def test_supplied_intent_exports_only_after_registered_check(tmp_path):
    algorithm = json.loads(recover('saxpy').to_json())['s_algo']
    proposal = tmp_path/'proposal.json'
    proposal.write_text(json.dumps({'s_algo':algorithm}),encoding='utf-8')
    root = tmp_path/'export'
    assert main(['--ops','saxpy','--proposal',str(proposal),'--output',str(root)]) == 0
    record = json.loads((root/'migration.json').read_text())['cases']['saxpy']
    assert record['status'] == 'CHECKED'
    assert record['intent_source'] == 'supplied proposal'
    assert (root/'saxpy/saxpy.cpp').exists()
