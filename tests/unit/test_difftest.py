import math
import struct
import pytest
from agentvec.difftest import compare, read_vals, ulp_distance, output_length


@pytest.mark.parametrize("reference,candidate", [(0.0, math.nan), (math.nan, math.nan),
    (math.inf, -math.inf), (1.0, math.inf), (math.inf, 1.0)])
def test_nonfinite_mismatches_never_pass(reference, candidate):
    assert not compare([reference], [candidate], True, 1e-6)[0]


def test_equal_infinities_and_signed_zero():
    assert compare([math.inf, -math.inf, 0.0], [math.inf, -math.inf, -0.0], True, 0)[0]


def test_relative_error_is_scaled_by_reference():
    assert not compare([1.0], [2.0], True, 0.6, 0)[0]


def test_ulp_limit_is_additional_to_relative_absolute_limit():
    next_float = struct.unpack("<f", struct.pack("<I", 0x3F800001))[0]
    assert ulp_distance(1.0, next_float) == 1
    assert not compare([1.0], [next_float], True, 1.0, max_ulps=0)[0]
    assert compare([1.0], [next_float], True, 1e-6, max_ulps=1)[0]
    assert not compare([1.0], [next_float], True, 0, 0, max_ulps=10)[0]


def test_output_length_and_truncated_file(tmp_path):
    assert not compare([1], [], False, 0)[0]
    output = tmp_path / "out.bin"
    output.write_bytes(b"abc")
    with pytest.raises(ValueError):
        read_vals(output, True)
    output.write_bytes(struct.pack("<d", 1.25))
    assert read_vals(output, True, "f64") == [1.25]


@pytest.mark.parametrize("rtol", [-1, math.nan, math.inf])
def test_invalid_tolerance(rtol):
    with pytest.raises(ValueError):
        compare([1.0], [1.0], True, rtol)


def test_legacy_multi_output_reference_shape():
    assert output_length({"reduce": False}, "#define OUT_FACTOR 2\n", 7) == 14
    assert output_length({"reduce": False}, "#define OUT_IS_MATRIX 1\n", 7) == 49
    assert output_length({"reduce": True}, "", 0) == 1
