import numpy as np

from pycutfem.jit import _merge_static_arrays


def test_merge_static_arrays_handles_scalar_ndarray_values():
    """
    Regression: parametric constants are represented as 0d NumPy arrays.
    _merge_static_arrays must not index shape[0] on these arrays.
    """
    target_eids = np.asarray([1, 2], dtype=np.int32)

    old_static = {
        "eids": np.asarray([1], dtype=np.int32),
        "foo": np.asarray([10.0], dtype=np.float64),
        "jit_const_0": np.asarray(1.5, dtype=np.float64),  # 0d array
    }
    new_static = {
        "eids": np.asarray([2], dtype=np.int32),
        "foo": np.asarray([20.0], dtype=np.float64),
        "jit_const_0": np.asarray(2.0, dtype=np.float64),  # 0d array
    }

    merged = _merge_static_arrays(target_eids, old_static, new_static)

    assert np.array_equal(merged["eids"], target_eids)
    assert merged["foo"].shape == (2,)
    assert merged["foo"][0] == 10.0
    assert merged["foo"][1] == 20.0

    assert isinstance(merged["jit_const_0"], np.ndarray)
    assert merged["jit_const_0"].shape == ()
    assert float(merged["jit_const_0"]) == 2.0

