"""Execute scalar lowering against independent composite references when GCC is available."""
import shutil

import pytest

from agentvec.dag import DagAIR
from agentvec.dag_examples import EXAMPLES, example, harness_source, reference_source
from agentvec.dag_lowering import DagSchedule, emit_dag
from agentvec.dag_runner import run


pytestmark = pytest.mark.skipif(not shutil.which('gcc'), reason='requires a C11 compiler')


def prepare(root, name, graph, fused):
    candidate, _ = emit_dag(graph, 'scalar', DagSchedule(fused))
    for filename, source in {'candidate.c':candidate, 'reference.c':reference_source(name),
                             'harness.c':harness_source(graph.contract)}.items():
        (root/filename).write_text(source, encoding='utf-8', newline='\n')


@pytest.mark.parametrize('name', EXAMPLES)
@pytest.mark.parametrize('fused', [False, True])
def test_compiled_composite_matches_independent_oracle(tmp_path, name, fused):
    prepare(tmp_path, name, example(name), fused)
    result = run(tmp_path, target='scalar')
    assert result['verified'], result
    assert result['statistics']['cases'] == 264


def test_compiled_wrong_intent_fails_independent_oracle(tmp_path):
    graph = DagAIR.from_proposal({'pattern':'dag','nodes':[{'name':'y','pattern':'map','formula':'x+17'}]},
                                example('softmax').contract)
    prepare(tmp_path, 'softmax', graph, False)
    result = run(tmp_path, target='scalar')
    assert result['build_exit'] == 0
    assert result['status'] == 'DIFF_FAILED'
    assert result['verified'] is False
    assert result['statistics']['failures'] > 0
