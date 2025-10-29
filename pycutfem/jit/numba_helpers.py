import numba
import numpy as np
use_type = np.float64

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
    if b.ndim == 0:
        raise ValueError("contract_last_first: second operand must have at least 1 dimension")
    if a.shape[-1] != b.shape[0]:
        raise ValueError("Dot dimension mismatch")
    left_dim = 1
    for idx in range(a.ndim - 1):
        left_dim *= a.shape[idx]
    right_dim = 1
    for idx in range(1, b.ndim):
        right_dim *= b.shape[idx]
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
def dot_grad_basis_vector(grad_basis, vec, dtype):
    """
    Gradient basis (k, n, d) dotted with spatial vector (d,) -> (k, n).
    """
    k_comps, n_locs, d_dim = grad_basis.shape
    if vec.shape[0] != d_dim:
        raise ValueError("Vector length does not match gradient spatial dimension")
    res = np.zeros((k_comps, n_locs), dtype=dtype)
    for k in range(k_comps):
        for d in range(d_dim):
            res[k] += grad_basis[k, :, d] * vec[d]
    return res


@numba.njit(cache=True)
def vector_dot_grad_basis(vec, grad_basis, dtype):
    """
    Vector (component) dotted with gradient basis (k, n, d).
    Returns (1, n) for scalar fields or (d, n) when k > 1.
    """
    k_comps, n_locs, d_dim = grad_basis.shape
    vlen = vec.shape[0]
    if k_comps == 1 and vlen == d_dim:
        res = np.zeros((1, n_locs), dtype=dtype)
        for d in range(d_dim):
            res[0] += vec[d] * grad_basis[0, :, d]
        return res
    if vlen == k_comps:
        res = np.zeros((d_dim, n_locs), dtype=dtype)
        for d in range(d_dim):
            for k in range(k_comps):
                res[d] += vec[k] * grad_basis[k, :, d]
        return res
    raise ValueError("vector·grad basis: incompatible shapes")


@numba.njit(cache=True)
def dot_grad_basis_with_grad_value(grad_basis, grad_value, dtype):
    """
    Grad(basis) (k, n, d) dotted with grad(value) (k, d) -> (k, n, k).
    """
    k_comps, n_locs, d_dim = grad_basis.shape
    if grad_value.shape[0] != k_comps or grad_value.shape[1] != d_dim:
        raise ValueError("Gradient value shape incompatible with basis gradient")
    res = np.zeros((k_comps, n_locs, k_comps), dtype=dtype)
    for n in range(n_locs):
        a_slice = grad_basis[:, n, :].copy()
        res[:, n, :] = a_slice @ grad_value
    return res


@numba.njit(cache=True)
def dot_grad_value_with_grad_basis(grad_value, grad_basis, dtype):
    """
    Grad(value) (k, d) dotted with grad(basis) (k, n, d) -> (k, n, d).
    """
    k_comps, n_locs, d_dim = grad_basis.shape
    if grad_value.shape[0] != k_comps or grad_value.shape[1] != d_dim:
        raise ValueError("Gradient value shape incompatible with basis gradient")
    res = np.zeros((k_comps, n_locs, d_dim), dtype=dtype)
    for n in range(n_locs):
        for i in range(k_comps):
            for d in range(d_dim):
                acc = 0.0
                for k in range(k_comps):
                    acc += grad_value[i, k] * grad_basis[k, n, d]
                res[i, n, d] = acc
    return res


@numba.njit(cache=True)
def dot_mass_test_trial(test_vec, trial_vec, dtype):
    """
    Compute Test.T @ Trial for mass matrices.
    """
    return test_vec.T.copy() @ trial_vec


@numba.njit(cache=True)
def dot_mass_trial_test(trial_vec, test_vec, dtype):
    """
    Compute Trial.T @ Test for mass matrices.
    """
    return trial_vec.T.copy() @ test_vec


@numba.njit(cache=True)
def dot_grad_func_trial_vec(grad_func, trial_vec, dtype):
    """
    Compute grad(Function) @ Trial vector basis.
    """
    return grad_func @ trial_vec


@numba.njit(cache=True)
def dot_trial_vec_grad_func(trial_vec, grad_func, dtype):
    """
    Compute Trial vector basis @ grad(Function).T.
    """
    return grad_func.T.copy() @ trial_vec


@numba.njit(cache=True)
def dot_vec_vec(vec_a, vec_b, dtype):
    """
    Dot product between two vectors.
    """
    return np.dot(vec_a, vec_b)


@numba.njit(cache=True)
def dot_grad_grad_value(grad_a, grad_b, dtype):
    """
    Compute grad(value) @ grad(value).
    """
    return grad_a @ grad_b


@numba.njit(cache=True)
def dot_value_with_grad(value_vec, grad_mat, dtype):
    """
    Dot product between value vector (k,) and gradient matrix (k,d).
    """
    return np.dot(value_vec, grad_mat)


@numba.njit(cache=True)
def dot_grad_with_value(grad_mat, value_vec, dtype):
    """
    Dot product between gradient matrix (k,d) and value vector (k,).
    """
    return np.dot(grad_mat, value_vec)


@numba.njit(cache=True)
def mul_scalar(scalar, array, dtype):
    """
    Multiply scalar with array.
    """
    return float(scalar) * array


@numba.njit(cache=True)
def _flatten_to_1d(array_like, dtype):
    """
    Convert scalar/array-like input to contiguous 1D array of dtype.
    """
    arr = np.ascontiguousarray(array_like)
    target = np.dtype(dtype)
    arr = arr.astype(target)
    if arr.ndim == 0:
        res = np.empty(1, dtype=target)
        res[0] = arr.item()
        return res
    return arr.reshape(arr.size)


@numba.njit(cache=True)
def _ensure_matrix(array_like, dtype):
    """
    Convert scalar/vector/array-like input to contiguous 2D array of dtype.
    """
    arr = np.ascontiguousarray(array_like)
    target = np.dtype(dtype)
    arr = arr.astype(target)
    if arr.ndim == 0:
        res = np.zeros((1, 1), dtype=target)
        res[0, 0] = arr.item()
        return res
    if arr.ndim == 1:
        return arr.reshape(arr.shape[0], 1)
    if arr.ndim == 2:
        return arr
    # For higher-dimensional inputs, flatten trailing axes
    rows = arr.shape[0]
    cols = arr.size // rows
    reshaped = arr.reshape(rows, cols)
    res = np.zeros((rows, cols), dtype=target)
    for i in range(rows):
        for j in range(cols):
            res[i, j] = reshaped[i, j]
    return res


@numba.njit(cache=True)
def trace_matrix_value(matrix, dtype):
    """
    Trace of a dense square matrix -> scalar.
    """
    return float(np.trace(np.ascontiguousarray(matrix)))


@numba.njit(cache=True)
def trace_basis_tensor(tensor, dtype):
    """
    Trace of a basis tensor (k, n, k) -> (1, n).
    """
    k_comps, n_locs, _ = tensor.shape
    res = np.zeros((1, n_locs), dtype=dtype)
    for j in range(n_locs):
        acc = 0.0
        for i in range(k_comps):
            acc += tensor[i, j, i]
        res[0, j] = acc
    return res


@numba.njit(cache=True)
def trace_mixed_tensor(tensor, dtype):
    """
    Trace of a mixed tensor (k, n_test, n_trial, k) -> (1, n_test, n_trial).
    """
    k_comps, n_test, n_trial, d_dim = tensor.shape
    res = np.zeros((1, n_test, n_trial), dtype=dtype)
    n_diag = min(k_comps, d_dim)
    for nt in range(n_test):
        for tr in range(n_trial):
            acc = 0.0
            for i in range(n_diag):
                acc += tensor[i, nt, tr, i]
            res[0, nt, tr] = acc
    return res


@numba.njit(cache=True)
def transpose_grad_tensor(tensor, dtype):
    """
    Transpose vector gradient tensor (k,n,d) assuming square k==d.
    """
    k_comps, n_locs, d_dim = tensor.shape
    res = np.zeros((k_comps, n_locs, d_dim), dtype=dtype)
    for n in range(n_locs):
        res[:, n, :] = tensor[:, n, :].T
    return res


@numba.njit(cache=True)
def transpose_mixed_grad_tensor(tensor, dtype):
    """
    Transpose mixed gradient tensor (k,n,m,d) by swapping component/spatial axes.
    """
    k_dim, n_rows, n_cols, d_dim = tensor.shape
    res = np.zeros((k_dim, n_rows, n_cols, d_dim), dtype=dtype)
    for i in range(k_dim):
        for j in range(d_dim):
            res[i, :, :, j] = tensor[j, :, :, i]
    return res


@numba.njit(cache=True)
def transpose_hessian_tensor(tensor, dtype):
    """
    Transpose Hessian tensor (k,n,d,d) swapping the last two axes.
    """
    return tensor.swapaxes(2, 3).copy()


@numba.njit(cache=True)
def transpose_matrix(matrix, dtype):
    """
    Transpose a 2D matrix.
    """
    return matrix.T.copy()


@numba.njit(cache=True)
def scatter_tensor_to_union(values, mapping, n_union, dtype):
    """
    Scatter local rows (m, ...) into a union-sized tensor (n_union, ...).
    """
    out_shape = (n_union,) + values.shape[1:]
    res = np.zeros(out_shape, dtype=dtype)
    m = values.shape[0]
    for j in range(m):
        idx = mapping[j]
        if 0 <= idx < n_union:
            res[idx] = values[j]
    return res


@numba.njit(cache=True)
def compute_physical_hessian(d20, d11, d02, d10, d01, j_inv, hx, hy, dtype):
    """
    Compute physical Hessian for a single component.
    """
    nloc = d20.shape[0]
    d_dim = j_inv.shape[0]
    res = np.zeros((nloc, d_dim, d_dim), dtype=dtype)
    for j in range(nloc):
        href00 = d20[j]
        href01 = d11[j]
        href11 = d02[j]
        href = np.zeros((d_dim, d_dim), dtype=dtype)
        href[0, 0] = href00
        href[0, 1] = href01
        href[1, 0] = href01
        href[1, 1] = href11
        core = j_inv.T @ (href @ j_inv)
        res[j] = core + d10[j] * hx + d01[j] * hy
    return res


@numba.njit(cache=True)
def compute_physical_laplacian(d20, d11, d02, d10, d01, j_inv, hx, hy, dtype):
    """
    Compute physical Laplacian entries for scalar component.
    """
    nloc = d20.shape[0]
    res = np.zeros(nloc, dtype=dtype)
    for j in range(nloc):
        href = np.zeros((2, 2), dtype=dtype)
        href[0, 0] = d20[j]
        href[0, 1] = d11[j]
        href[1, 0] = d11[j]
        href[1, 1] = d02[j]
        core = j_inv.T @ (href @ j_inv)
        hphys = core + d10[j] * hx + d01[j] * hy
        res[j] = hphys[0, 0] + hphys[1, 1]
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
def const_vector_dot_basis_1d(const_vec, basis, dtype):
    """
    Constant vector (k,) dotted with basis (k,n) -> (n,).
    """
    n_locs = basis.shape[1]
    k_comps = basis.shape[0]
    res = np.zeros(n_locs, dtype=dtype)
    for n in range(n_locs):
        acc = 0.0
        for k in range(k_comps):
            acc += const_vec[k] * basis[k, n]
        res[n] = acc
    return res


@numba.njit(cache=True)
def scalar_basis_times_vector(scalar_basis, vector_vals, dtype):
    """
    Scalar basis (1,n) or (n,) times vector components (k,) -> (k,n).
    """
    if scalar_basis.ndim == 2:
        basis_vals = scalar_basis[0]
    else:
        basis_vals = scalar_basis
    n_vec = vector_vals.shape[0]
    n_basis = basis_vals.shape[0]
    res = np.zeros((n_vec, n_basis), dtype=dtype)
    for k in range(n_vec):
        res[k, :] = basis_vals * vector_vals[k]
    return res


@numba.njit(cache=True)
def matrix_times_scalar_basis(matrix_vals, scalar_basis, dtype):
    """
    Matrix (m, n) times scalar basis row (1,p) -> (1,p).
    """
    if scalar_basis.ndim == 2:
        phi_row = scalar_basis[0]
    else:
        phi_row = scalar_basis
    n_trial = phi_row.shape[0]
    m_rows, n_cols = matrix_vals.shape
    res = np.zeros((1, n_trial), dtype=dtype)
    for i in range(m_rows):
        accum = np.zeros(n_trial, dtype=dtype)
        for j in range(n_cols):
            coeff = matrix_vals[i, j]
            if coeff != 0.0:
                accum += coeff * phi_row
        res[0] += accum
    return res


@numba.njit(cache=True)
def scalar_vector_outer_product(scalar_vals, vector_vals, dtype):
    """
    Scalar values (n,) times vector (k,) -> (k,n).
    """
    n_basis = scalar_vals.shape[0]
    n_vec = vector_vals.shape[0]
    res = np.zeros((n_vec, n_basis), dtype=dtype)
    for k in range(n_vec):
        for n in range(n_basis):
            res[k, n] = scalar_vals[n] * vector_vals[k]
    return res


@numba.njit(cache=True)
def scalar_trial_times_grad_test(grad_test, trial_vals, dtype):
    """
    Scalar Trial basis (n_trial,) times grad(Test) (k,n_test,d) -> (k,n_test,n_trial,d).
    """
    k_comps, n_test, d_dim = grad_test.shape
    n_trial = trial_vals.shape[0]
    res = np.zeros((k_comps, n_test, n_trial, d_dim), dtype=dtype)
    for comp in range(k_comps):
        for dim in range(d_dim):
            row = grad_test[comp, :, dim]
            for nt in range(n_test):
                res[comp, nt, :, dim] = row[nt] * trial_vals
    return res


@numba.njit(cache=True)
def grad_trial_times_scalar_test(grad_trial, test_vals, dtype):
    """
    Grad(Trial) (k,n_trial,d) times scalar Test (n_test,) -> (k,n_test,n_trial,d).
    """
    k_comps, n_trial, d_dim = grad_trial.shape
    n_test = test_vals.shape[0]
    res = np.zeros((k_comps, n_test, n_trial, d_dim), dtype=dtype)
    for comp in range(k_comps):
        for dim in range(d_dim):
            col = grad_trial[comp, :, dim]
            for nt in range(n_test):
                res[comp, nt, :, dim] = test_vals[nt] * col
    return res


@numba.njit(cache=True)
def scale_mixed_basis_with_coeffs(mixed_basis, coeffs, dtype):
    """
    Mixed basis (k_mixed, n_rows, n_cols) scaled by coeffs (k_out, d_cols)
    -> (k_out, n_rows, n_cols, d_cols).
    """
    k_mixed, n_rows, n_cols = mixed_basis.shape
    k_out, d_cols = coeffs.shape
    res = np.zeros((k_out, n_rows, n_cols, d_cols), dtype=dtype)
    for i in range(k_out):
        for j in range(d_cols):
            coeff = coeffs[i, j]
            if coeff != 0.0:
                for comp in range(k_mixed):
                    res[i, :, :, j] += mixed_basis[comp] * coeff
    return res


@numba.njit(cache=True)
def trace_times_identity(trace_vals, identity, dtype):
    """
    Trace data (flattened) times identity matrix (k,d) -> (k, n_rows, d).
    """
    flat = _flatten_to_1d(trace_vals, dtype)
    identity_mat = _ensure_matrix(identity, dtype)
    n_rows = flat.shape[0]
    k_comps, d_dim = identity_mat.shape
    res = np.zeros((k_comps, n_rows, d_dim), dtype=dtype)
    for i in range(k_comps):
        for j in range(d_dim):
            coeff = identity_mat[i, j]
            for r in range(n_rows):
                res[i, r, j] = flat[r] * coeff
    return res


@numba.njit(cache=True)
def identity_times_trace_matrix(identity, trace_matrix, dtype):
    """
    Identity (k,d) times trace matrix (n_rows,n_cols) -> (k,n_rows,n_cols,d).
    """
    identity_mat = _ensure_matrix(identity, dtype)
    base = _ensure_matrix(trace_matrix, dtype)
    n_rows, n_cols = base.shape
    k_comps, d_dim = identity_mat.shape
    res = np.zeros((k_comps, n_rows, n_cols, d_dim), dtype=dtype)
    for i in range(k_comps):
        for r in range(n_rows):
            for c in range(n_cols):
                val = base[r, c]
                for j in range(d_dim):
                    res[i, r, c, j] = val * identity_mat[i, j]
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
def inner_grad_function_grad_test(function_grad, test_grad, dtype):
    """
    Inner product between grad(Function) (k,d) and grad(Test) basis (k,n,d) -> (n,).
    """
    k_comps, n_locs, d_dim = test_grad.shape
    if function_grad.shape[0] != k_comps or function_grad.shape[1] != d_dim:
        raise ValueError("Gradient(Function) shape incompatible with grad(Test)")
    res = np.zeros(n_locs, dtype=dtype)
    for comp in range(k_comps):
        res += test_grad[comp] @ function_grad[comp]
    return res


@numba.njit(cache=True)
def inner_hessian_function_hessian_test(function_hess, test_hess, dtype):
    """
    Inner product between Hess(Function) (k,d,d) and Hess(Test) basis (k,n,d,d) -> (n,).
    """
    k_comps, n_locs, d_dim, d_dim2 = test_hess.shape
    if (function_hess.shape[0] != k_comps or
            function_hess.shape[1] != d_dim or
            function_hess.shape[2] != d_dim2):
        raise ValueError("Hessian(Function) shape incompatible with Hessian(Test)")
    res = np.zeros(n_locs, dtype=dtype)
    for comp in range(k_comps):
        for n in range(n_locs):
            acc = 0.0
            for i in range(d_dim):
                for j in range(d_dim2):
                    acc += test_hess[comp, n, i, j] * function_hess[comp, i, j]
            res[n] += acc
    return res


@numba.njit(cache=True)
def inner_mixed_grad_const(mixed_grad, grad_const, dtype):
    """
    Inner product of mixed gradient (k, n_test, n_trial, d)
    with gradient of const/value (k, d) -> (n_test, n_trial).
    """
    k_comps, n_test, n_trial, d_dim = mixed_grad.shape
    if grad_const.shape[0] != k_comps or grad_const.shape[1] != d_dim:
        raise ValueError("Gradient(const) shape incompatible with mixed gradient")
    res = np.zeros((n_test, n_trial), dtype=dtype)
    for comp in range(k_comps):
        for d in range(d_dim):
            res += mixed_grad[comp, :, :, d] * grad_const[comp, d]
    return res


@numba.njit(cache=True)
def inner_grad_const_mixed(grad_const, mixed_grad, dtype):
    """
    Inner product of grad(const/value) (k, d) with mixed gradient (k, n_test, n_trial, d)
    -> (n_test, n_trial).
    """
    k_comps, n_test, n_trial, d_dim = mixed_grad.shape
    if grad_const.shape[0] != k_comps or grad_const.shape[1] != d_dim:
        raise ValueError("Gradient(const) shape incompatible with mixed gradient")
    res = np.zeros((n_test, n_trial), dtype=dtype)
    for comp in range(k_comps):
        grad_vec = grad_const[comp]
        for i in range(n_test):
            block = mixed_grad[comp, i]
            res[i, :] += block @ grad_vec
    return res


@numba.njit(cache=True)
def inner_grad_basis_grad_const(grad_basis, grad_const, dtype):
    """
    Inner product of grad(Test/Trial) basis (k, n, d) with grad(const/value) (k, d) -> (n,).
    """
    k_comps, n_locs, d_dim = grad_basis.shape
    if grad_const.shape[0] != k_comps or grad_const.shape[1] != d_dim:
        raise ValueError("Gradient(const) shape incompatible with grad basis")
    res = np.zeros(n_locs, dtype=dtype)
    for n in range(n_locs):
        acc = 0.0
        for comp in range(k_comps):
            for d in range(d_dim):
                acc += grad_basis[comp, n, d] * grad_const[comp, d]
        res[n] = acc
    return res


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


@numba.njit(cache=True, inline='always')
def _binary_add_or_sub(x, y, sign):
    """sign = +1.0 for add, -1.0 for subtract."""
    return x + sign * y





# In numba_helpers.py

@numba.njit(cache=True)
def binary_add_generic(a, b, dtype):
    """
    Elementwise addition for all standard broadcasting.
    Uses Numba's native ufunc support.
    """
    return a + b



@numba.njit(cache=True)
def binary_add_3_4(a, b, dtype):
    """
    Specialized elementwise addition for:
    (k, n, d) + (k, n, m, d) -> (k, n, m, d)
    Uses reshape + ufunc. (Very fast to compile)
    """
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)

    k = a_arr.shape[0]
    n = a_arr.shape[1]
    
    # Reshape (k, n, d) -> (k, n, 1, d) and let Numba broadcast
    return a_arr.reshape(k, n, 1, -1) + b_arr


@numba.njit(cache=True)
def binary_add_4_3(a, b, dtype):
    """
    Specialized elementwise addition for:
    (k, n, m, d) + (k, n, d) -> (k, n, m, d)
    Uses reshape + ufunc. (Very fast to compile)
    """
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)

    k = b_arr.shape[0]
    n = b_arr.shape[1]
    
    # Reshape (k, n, d) -> (k, n, 1, d) and let Numba broadcast
    return a_arr + b_arr.reshape(k, n, 1, -1)


@numba.njit(cache=True)
def binary_sub_generic(a, b, dtype):
    """
    Elementwise subtraction for all standard broadcasting.
    Uses Numba's native ufunc support.
    """
    return a - b


@numba.njit(cache=True)
def binary_sub_3_4(a, b, dtype):
    """
    Specialized elementwise subtraction for:
    (k, n, d) - (k, n, m, d) -> (k, n, m, d)
    Uses reshape + ufunc. (Very fast to compile)
    """
    a_arr = np.array(a, dtype=dtype, copy=False)
    b_arr = np.array(b, dtype=dtype, copy=False)
    
    k = a_arr.shape[0]
    n = a_arr.shape[1]
    
    # Reshape (k, n, d) -> (k, n, 1, d) and let Numba broadcast
    return a_arr.reshape(k, n, 1, -1) - b_arr


@numba.njit(cache=True)
def binary_sub_4_3(a, b, dtype):
    """
    Specialized elementwise subtraction for:
    (k, n, m, d) - (k, n, d) -> (k, n, m, d)
    Uses reshape + ufunc. (Very fast to compile)
    """
    a_arr = np.array(a, dtype=dtype, copy=False)
    b_arr = np.array(b, dtype=dtype, copy=False)
    
    k = b_arr.shape[0]
    n = b_arr.shape[1]
    
    # Reshape (k, n, d) -> (k, n, 1, d) and let Numba broadcast
    return a_arr - b_arr.reshape(k, n, 1, -1)


class BinaryOpsHelpers:
    """Backward compatibility shim for legacy imports."""
    binary_add_generic = staticmethod(binary_add_generic)
    binary_add_3_4 = staticmethod(binary_add_3_4)
    binary_add_4_3 = staticmethod(binary_add_4_3)
    binary_sub_generic = staticmethod(binary_sub_generic)
    binary_sub_3_4 = staticmethod(binary_sub_3_4)
    binary_sub_4_3 = staticmethod(binary_sub_4_3)
    


    # --------------------------
# IR LoadVariable helper routines
# --------------------------

@numba.njit(cache=True)
def load_variable_qp(u_e, phi_q):
    """Evaluate u_h at a quadrature point."""
    ndof = phi_q.shape[0]
    if u_e.ndim == 1:
        acc = 0.0
        for a in range(ndof):
            acc += u_e[a] * phi_q[a]
        return acc
    ncomp = u_e.shape[1]
    out = np.empty(ncomp, dtype=use_type)
    for i in range(ncomp):
        acc = 0.0
        for a in range(ndof):
            acc += u_e[a, i] * phi_q[a]
        out[i] = acc
    return out


@numba.njit(cache=True)
def gradient_qp(u_e, grad_phi_q):
    """Evaluate ∇u_h at a quadrature point."""
    ndof = grad_phi_q.shape[0]
    dim = grad_phi_q.shape[1]
    if u_e.ndim == 1:
        g = np.zeros(dim, dtype=use_type)
        for d in range(dim):
            acc = 0.0
            for a in range(ndof):
                acc += u_e[a] * grad_phi_q[a, d]
            g[d] = acc
        return g
    ncomp = u_e.shape[1]
    G = np.zeros((dim, ncomp), dtype=use_type)
    for d in range(dim):
        for i in range(ncomp):
            acc = 0.0
            for a in range(ndof):
                acc += u_e[a, i] * grad_phi_q[a, d]
            G[d, i] = acc
    return G


@numba.njit(cache=True)
def laplacian_qp(u_e, lap_phi_q):
    """Evaluate Δu_h at a quadrature point."""
    ndof = lap_phi_q.shape[0]
    if u_e.ndim == 1:
        acc = 0.0
        for a in range(ndof):
            acc += u_e[a] * lap_phi_q[a]
        return acc
    ncomp = u_e.shape[1]
    out = np.empty(ncomp, dtype=use_type)
    for i in range(ncomp):
        acc = 0.0
        for a in range(ndof):
            acc += u_e[a, i] * lap_phi_q[a]
        out[i] = acc
    return out


@numba.njit(cache=True)
def hessian_qp(u_e, hess_phi_q):
    """Evaluate ∇²u_h at a quadrature point."""
    ndof = hess_phi_q.shape[0]
    dim = hess_phi_q.shape[1]
    if u_e.ndim == 1:
        H = np.zeros((dim, dim), dtype=use_type)
        for i in range(dim):
            for j in range(dim):
                acc = 0.0
                for a in range(ndof):
                    acc += u_e[a] * hess_phi_q[a, i, j]
                H[i, j] = acc
        return H
    ncomp = u_e.shape[1]
    H = np.zeros((ncomp, dim, dim), dtype=use_type)
    for c in range(ncomp):
        for i in range(dim):
            for j in range(dim):
                acc = 0.0
                for a in range(ndof):
                    acc += u_e[a, c] * hess_phi_q[a, i, j]
                H[c, i, j] = acc
    return H


@numba.njit(cache=True)
def collapse_hessian_to_value(h_tensor, coeffs, dtype):
    """Compatibility wrapper delegating to hessian_qp."""
    return hessian_qp(coeffs, h_tensor, dtype)


@numba.njit(cache=True)
def collapse_vector_to_value(vector_vals, coeffs, dtype):
    """Compatibility wrapper delegating to laplacian_qp."""
    return laplacian_qp(coeffs, vector_vals, dtype)



BinaryOpHelpers = BinaryOpsHelpers

class AssemblyHelpers:
    """Backward compatibility shim for legacy assembly helpers."""
    pass

for _helper_name in (
    "trace_times_identity",
    "identity_times_trace_matrix",
    "columnwise_dot",
    "hessian_dot_vector",
    "vector_dot_hessian_basis",
    "vector_dot_hessian_value",
    "scale_mixed_basis_with_coeffs",
    "matrix_times_scalar_basis",
    "scalar_vector_outer_product",
    "scalar_basis_times_vector",
    "scalar_trial_times_grad_test",
    "grad_trial_times_scalar_test",
    "trace_matrix_value",
    "trace_basis_tensor",
    "trace_mixed_tensor",
    "transpose_grad_tensor",
    "transpose_mixed_grad_tensor",
    "transpose_hessian_tensor",
    "transpose_matrix",
    "mul_scalar",
    "dot_grad_basis_vector",
    "vector_dot_grad_basis",
    "dot_grad_basis_with_grad_value",
    "dot_grad_value_with_grad_basis",
    "dot_grad_func_trial_vec",
    "dot_trial_vec_grad_func",
    "dot_vec_vec",
    "dot_grad_grad_value",
    "dot_value_with_grad",
    "dot_grad_with_value",
    "compute_physical_hessian",
    "compute_physical_laplacian",
    "load_variable_qp",
    "gradient_qp",
    "laplacian_qp",
    "hessian_qp",
):
    setattr(AssemblyHelpers, _helper_name, staticmethod(globals()[_helper_name]))

class IRLoadVariableHelpers:
    """Backward compatibility shim for legacy load-variable helpers."""
    LoadVariable = staticmethod(load_variable_qp)
    IRGradient = staticmethod(gradient_qp)
    laplacian = staticmethod(laplacian_qp)
    hessian = staticmethod(hessian_qp)

# ---------------------------------------------------------------------------
# Public aliases (module-level) for the helper routines
# ---------------------------------------------------------------------------

evaluate_field_value = load_variable_qp
evaluate_field_gradient = gradient_qp
evaluate_field_laplacian = laplacian_qp
evaluate_field_hessian = hessian_qp
