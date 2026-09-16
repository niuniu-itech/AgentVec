"""Ascend template admission, exact dimensions and result parsing."""
from dataclasses import replace
import json
import pytest

from agentvec.ascend.air import AIR, Status
from agentvec.ascend.guard import guard
from agentvec.ascend.lowering import Sched, emit_project
from agentvec.ascend.pipeline import export
from agentvec.ascend.registry import ALL_OPS, recover
from agentvec.ascend.runner import parse_result
from agentvec.ascend.schedule import check_schedule, check_shape


@pytest.mark.parametrize("op", ALL_OPS)
def test_registered_admission_has_no_unmeasured_verified_state(op):
    air = recover(op)
    ok, _ = guard(air)
    assert ok == (op not in ("strsv", "strsm"))
    assert air.v_meta.status == (Status.CHECKED if ok else Status.DRAFT)


def test_mismatched_semantics_and_forged_status_rejected():
    air = recover("saxpy")
    air.s_algo.elem_op.args.reverse()
    air.v_meta.status = Status.VERIFIED
    with pytest.raises(ValueError, match="semantics"):
        emit_project(air, Sched())
    raw = json.loads(recover("scopy").to_json())
    raw["v_meta"]["status"] = "VERIFIED"
    assert AIR.from_json(json.dumps(raw)).v_meta.status == Status.DRAFT


@pytest.mark.parametrize("field,value", [("dtype","f16"),("dtype","i32"),("dep_distance",1)])
def test_reject_contract_drift(field,value):
    air = recover("scopy")
    setattr(air.c_phy,field,value)
    assert not guard(air)[0]


@pytest.mark.parametrize("schedule", [Sched(block_dim=9),Sched(tile_len=1000),Sched(tile_len=16384),
                                      Sched(tile_m=32),Sched(buffer_num=3)])
def test_schedule_caps(schedule):
    with pytest.raises(ValueError):
        check_schedule(recover("saxpy"),schedule)


@pytest.mark.parametrize("op,sizes", [("saxpy",[1000]),("sgemv",[65,64]),("sgemv",[64,63]),
                                      ("sgemm",[65,64,64]),("sgemm_nt",[64,64,65]),
                                      ("ssyrk",[64,128,64])])
def test_shapes_are_rejected_instead_of_rounded(op,sizes):
    with pytest.raises(ValueError):
        check_shape(recover(op),Sched(),sizes,10)


@pytest.mark.parametrize("op", ALL_OPS[:-2])
def test_complete_project_export(tmp_path,op):
    record = export(op,tmp_path/op,Sched(),rounds=2)
    assert record["status"] == "CHECKED"
    assert len(record["source_sha256"]) == 4
    assert 'set(CMAKE_BUILD_TYPE "Release"' in (tmp_path/op/'CMakeLists.txt').read_text()
    host = (tmp_path/op/'main.cpp').read_text()
    assert "std::isfinite(maxrel)" in host
    assert "parse_size" in host
    assert "n=(n/" not in host and "n = (n/" not in host
    assert "m=(m/blockDim)" not in host and "n=m;" not in host


@pytest.mark.parametrize("line", [
    "RESULT op=saxpy pass=1 n=8192 maxrel=nan us=1",
    "RESULT op=saxpy pass=1 n=8192 maxrel=0 us=inf",
    "RESULT op=saxpy pass=1 n=4096 maxrel=0 us=1",
    "RESULT op=sdot pass=1 n=8192 maxrel=0 us=1",
    "RESULT op=saxpy pass=0 n=8192 maxrel=0 us=1",
    "RESULT op=saxpy pass=1 n=8192 maxrel=0 us=0",
])
def test_bad_measurement_cannot_be_accepted(line):
    with pytest.raises(ValueError):
        parse_result(line,'saxpy',[8192])


def test_model_proposal_cannot_supply_rules_or_change_scalar_operation():
    from agentvec.ascend.proposer import from_proposal
    algorithm = json.loads(recover('sscal').to_json())['s_algo']
    assert guard(from_proposal('sscal', {'s_algo':algorithm}))[0]
    algorithm['elem_op'] = {'op':'load','src':'x'}
    assert not guard(from_proposal('sscal', {'s_algo':algorithm}))[0]
    with pytest.raises(ValueError):
        from_proposal('sscal', {'s_algo':algorithm, 'v_meta':{'status':'VERIFIED'}})
