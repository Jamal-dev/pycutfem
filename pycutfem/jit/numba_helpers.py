import numba
import numpy as np


@numba.njit(cache=True)
def dot_grad_grad_mixed(a, b, flag, dtype):
    """
    Matmul for grad(test/trial) · grad(trial/test).
    """
    k_comps, n_basis_a, d_comps = a.shape
    d_comps_b, n_basis_b, l_comps = b.shape
    if d_comps != d_comps_b:
        raise ValueError("Gradient dimensions do not match")
    a_flat = np.ascontiguousarray(a).reshape(k_comps * n_basis_a, d_comps)
    b_flat = np.ascontiguousarray(b).reshape(d_comps_b, n_basis_b * l_comps)
    res_flat = a_flat @ b_flat
    res_tmp = res_flat.reshape(k_comps, n_basis_a, n_basis_b, l_comps)
    if flag == 1:
        return np.transpose(res_tmp, (0, 2, 1, 3))
    return res_tmp


@numba.njit(cache=True)
def contract_last_first(a, b, dtype):
    """
    Generic contraction over the last axis of a and first axis of b.
    """
    if a.shape[-1] != b.shape[0]:
        raise ValueError("Dot dimension mismatch")
    left_dim = 1
    for dim in a.shape[:-1]:
        left_dim *= dim
    right_dim = 1
    for dim in b.shape[1:]:
        right_dim *= dim
    a_flat = np.ascontiguousarray(a).reshape(left_dim, a.shape[-1])
    b_flat = np.ascontiguousarray(b).reshape(a.shape[-1], right_dim)
    dot_flat = a_flat @ b_flat
    out_shape = a.shape[:-1] + b.shape[1:]
    return dot_flat.reshape(out_shape)


@numba.njit(cache=True)
def dot_mixed_const(a, b, dtype):
    """
    Mixed basis (k,n,m) dotted with constant vector (k,) -> (n,m).
    """
    k_comps = a.shape[0]
    res = np.zeros((a.shape[1], a.shape[2]), dtype=dtype)
    for comp in range(k_comps):
        res += a[comp] * b[comp]
    return res


@numba.njit(cache=True)
def dot_const_mixed(a, b, dtype):
    """
    Constant vector (k,) dotted with mixed basis (k,n,m) -> (n,m).
    """
    k_comps = b.shape[0]
    res = np.zeros((b.shape[1], b.shape[2]), dtype=dtype)
    for comp in range(k_comps):
        res += b[comp] * a[comp]
    return res


@numba.njit(cache=True)
def dot_vector_trial_grad_test(trial_vec, grad_test, dtype):
    """
    Vector trial · grad(test) -> mixed tensor (d, n_test, n_trial).
    """
    k_vec = trial_vec.shape[0]
    n_trial = trial_vec.shape[1]
    n_test = grad_test.shape[1]
    d = grad_test.shape[2]
    res = np.zeros((d, n_test, n_trial), dtype=dtype)
    for comp in range(k_vec):
        vec_vals = trial_vec[comp]
        grad_block = grad_test[comp]
        for j in range(d):
            grad_col = grad_block[:, j]
            res[j] += np.outer(grad_col, vec_vals)
    return res


@numba.njit(cache=True)
def basis_dot_const_vector(basis, const_vec, dtype):
    """
    Basis (k,n) dotted with constant vector (k,) -> (1,n).
    """
    n_locs = basis.shape[1]
    res = np.zeros((1, n_locs), dtype=dtype)
    for n in range(n_locs):
        res[0, n] = np.dot(basis[:, n], const_vec)
    return res


@numba.njit(cache=True)
def const_vector_dot_basis(const_vec, basis, dtype):
    """
    Constant vector (k,) dotted with basis (k,n) -> (1,n).
    """
    n_locs = basis.shape[1]
    res = np.zeros((1, n_locs), dtype=dtype)
    for n in range(n_locs):
        res[0, n] = np.dot(basis[:, n], const_vec)
    return res


@numba.njit(cache=True)
def columnwise_dot(a_mat, b_mat, dtype):
    """
    Column-wise dot products between two (k,n) arrays -> (1,n).
    """
    n_cols = a_mat.shape[1]
    res = np.zeros((1, n_cols), dtype=dtype)
    for n in range(n_cols):
        acc = 0.0
        for k in range(a_mat.shape[0]):
            acc += a_mat[k, n] * b_mat[k, n]
        res[0, n] = acc
    return res


@numba.njit(cache=True)
def hessian_dot_vector(hessian, vec, dtype):
    """
    Hessian (basis or value) dotted with spatial vector.
    Returns (k, n, d1) for basis, (k, d1) for value.
    """
    vec_contig = np.ascontiguousarray(vec)
    if hessian.ndim == 4:
        k_comps, n_locs, d1, _ = hessian.shape
        res = np.zeros((k_comps, n_locs, d1), dtype=dtype)
        for kk in range(k_comps):
            for n in range(n_locs):
                res[kk, n, :] = np.dot(hessian[kk, n], vec_contig)
        return res
    elif hessian.ndim == 3:
        k_comps, d1, _ = hessian.shape
        res = np.zeros((k_comps, d1), dtype=dtype)
        for kk in range(k_comps):
            res[kk, :] = np.dot(hessian[kk], vec_contig)
        return res
    else:
        raise ValueError("Unsupported Hessian shape for hessian_dot_vector")


@numba.njit(cache=True)
def vector_dot_hessian_basis(vec, hessian, dtype):
    """
    Vector dotted with Hessian basis (k,n,d1,d2).
    Handles component and spatial vector variants.
    """
    k_comps, n_locs, d1, d2 = hessian.shape
    vlen = vec.shape[0]
    vec_contig = np.ascontiguousarray(vec)
    if vlen == k_comps and k_comps > 1:
        res = np.zeros((d1, n_locs, d2), dtype=dtype)
        for n in range(n_locs):
            for j in range(d1):
                res[j, n, :] = np.dot(vec_contig, hessian[:, n, j, :])
        return res
    elif vlen == d1 and k_comps == 1:
        res = np.zeros((1, n_locs, d2), dtype=dtype)
        for n in range(n_locs):
            res[0, n, :] = np.dot(vec_contig, hessian[0, n])
        return res
    else:
        raise ValueError("vector·Hessian(basis): incompatible shapes")


@numba.njit(cache=True)
def vector_dot_hessian_value(vec, hessian, dtype):
    """
    Vector dotted with Hessian value (k,d1,d2).
    Handles component and spatial vector variants.
    """
    k_comps, d1, d2 = hessian.shape
    vlen = vec.shape[0]
    vec_contig = np.ascontiguousarray(vec)
    if vlen == k_comps and k_comps > 1:
        res = np.zeros((d1, d2), dtype=dtype)
        for j in range(d1):
            res[j, :] = np.dot(vec_contig, hessian[:, j, :])
        return res
    elif vlen == d1 and k_comps == 1:
        res = np.zeros((1, d2), dtype=dtype)
        res[0, :] = np.dot(vec_contig, hessian[0])
        return res
    else:
        raise ValueError("vector·Hessian(value): incompatible shapes")


@numba.njit(cache=True)
def inner_grad_grad(test_var, trial_var, dtype):
    """
    Inner product of gradient bases -> (n_test, n_trial).
    """
    n_test = test_var.shape[1]
    n_trial = trial_var.shape[1]
    res = np.zeros((n_test, n_trial), dtype=dtype)
    for k in range(test_var.shape[0]):
        res += test_var[k] @ trial_var[k].T
    return res


@numba.njit(cache=True)
def inner_hessian_hessian(test_var, trial_var, dtype):
    """
    Inner product of Hessian bases -> (n_test, n_trial).
    """
    n_test = test_var.shape[1]
    n_trial = trial_var.shape[1]
    res = np.zeros((n_test, n_trial), dtype=dtype)
    for k in range(test_var.shape[0]):
        test_flat = test_var[k].reshape(n_test, -1)
        trial_flat = trial_var[k].reshape(n_trial, -1)
        res += test_flat @ trial_flat.T
    return res


class DotHelpers:
    dot_grad_grad_mixed = staticmethod(dot_grad_grad_mixed)
    contract_last_first = staticmethod(contract_last_first)
    dot_mixed_const = staticmethod(dot_mixed_const)
    dot_const_mixed = staticmethod(dot_const_mixed)
    dot_vector_trial_grad_test = staticmethod(dot_vector_trial_grad_test)
    basis_dot_const_vector = staticmethod(basis_dot_const_vector)
    const_vector_dot_basis = staticmethod(const_vector_dot_basis)
    columnwise_dot = staticmethod(columnwise_dot)
    hessian_dot_vector = staticmethod(hessian_dot_vector)
    vector_dot_hessian_basis = staticmethod(vector_dot_hessian_basis)
    vector_dot_hessian_value = staticmethod(vector_dot_hessian_value)


class InnerHelpers:
    inner_grad_grad = staticmethod(inner_grad_grad)
    inner_hessian_hessian = staticmethod(inner_hessian_hessian)
