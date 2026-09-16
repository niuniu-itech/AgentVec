"""Published examples must conform to the schema offered to model callers."""
import json
from pathlib import Path

import jsonschema
import pytest

from agentvec.dag_examples import EXAMPLES

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = json.loads((ROOT/'schemas/dag.schema.json').read_text())


@pytest.mark.parametrize('name', EXAMPLES)
def test_published_graph_example_matches_schema(name):
    proposal = json.loads((ROOT/'examples/dag'/(name+'.json')).read_text())
    jsonschema.validate(proposal, SCHEMA)


@pytest.mark.parametrize('pattern,reduce_op', [('map','sum'),('reduce',None)])
def test_schema_rejects_mismatched_reduction_slot(pattern, reduce_op):
    proposal = {'pattern':'dag','nodes':[{'name':'y','pattern':pattern,'formula':'x','reduce_op':reduce_op}]}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(proposal, SCHEMA)
