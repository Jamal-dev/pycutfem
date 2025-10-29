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
    Mixed basis (k, n, m) dotted with constant vector (k,) -> (n, m).
    Vectorized via (k*nm) matvec.
    """
    k = a.shape[0]
    nm = a.shape[1] * a.shape[2]
    a2 = np.ascontiguousarray(a).reshape(k, nm)
    return (a2.T @ np.ascontiguousarray(b)).reshape(a.shape[1], a.shape[2])



@numba.njit(cache=True)
def dot_const_mixed(a, b, dtype):
    """
    Constant vector (k,) dotted with mixed basis (k, n, m) -> (n, m).
    Vectorized via (1 x k) @ (k x nm).
    """
    k = b.shape[0]
    nm = b.shape[1] * b.shape[2]
    b2 = np.ascontiguousarray(b).reshape(k, nm)
    return (np.ascontiguousarray(a) @ b2).reshape(b.shape[1], b.shape[2])



@numba.njit(cache=True)
def dot_vector_trial_grad_test(trial_vec, grad_test, dtype):
    """
    Vector trial (k, n_trial) · grad(test) (k, n_test, d) -> (d, n_test, n_trial).
    For each spatial dim j: res[j] = grad_test[:, :, j].T @ trial_vec
    """
    k_vec, n_trial = trial_vec.shape
    _, n_test, d = grad_test.shape
    res = np.empty((d, n_test, n_trial), dtype=dtype)
    G = np.ascontiguousarray(grad_test)   # (k, n_test, d)
    V = np.ascontiguousarray(trial_vec)   # (k, n_trial)
    for j in range(d):
        # (k, n_test) -> (n_test, k)
        Gj = np.ascontiguousarray(G[:, :, j].T)
        res[j] = Gj @ V
    return res


@numba.njit(cache=True)
def dot_grad_basis_vector(grad_basis, vec, dtype):
    """
    Gradient basis (k, n, d) dotted with spatial vector (d,) -> (k, n).
    Vectorized via (k*n, d) @ (d,)
    """
    k, n, d = grad_basis.shape
    G = np.ascontiguousarray(grad_basis).reshape(k * n, d)
    out = G @ np.ascontiguousarray(vec)
    return out.reshape(k, n)


@numba.njit(cache=True)
def vector_dot_grad_basis(vec, grad_basis, dtype):
    """
    Vector (component) dotted with gradient basis (k, n, d).
    Returns (1, n) for scalar fields or (d, n) when len(vec)==k.
    """
    k, n, d = grad_basis.shape
    vlen = vec.shape[0]
    if k == 1 and vlen == d:
        # (n,d) @ (d,) -> (n,)
        res = np.empty((1, n), dtype=dtype)
        res[0] = np.ascontiguousarray(grad_basis[0]) @ np.ascontiguousarray(vec)
        return res
    if vlen == k:
        # sum_k vec[k] * grad_basis[k, :, d]  -> reshape to do one BLAS
        A = np.ascontiguousarray(grad_basis).reshape(k, n * d)         # (k, n*d)
        tmp = np.ascontiguousarray(vec) @ A                             # (n*d,)
        return tmp.reshape(n, d).T.copy()                               # (d, n)
    raise ValueError("vector·grad basis: incompatible shapes")


@numba.njit(cache=True)
def dot_grad_basis_with_grad_value(grad_basis, grad_value, dtype):
    """
    Grad(basis) (k, n, d) dotted with grad(value) (k, d) -> (k, n, k).
    For each n: (k,d) @ (d,k)  (transpose on grad_value).
    """
    k, n, d = grad_basis.shape
    if grad_value.shape[0] != k or grad_value.shape[1] != d:
        raise ValueError("Gradient value shape incompatible with basis gradient")
    res = np.empty((k, n, k), dtype=dtype)
    GVt = np.ascontiguousarray(grad_value)  # (k, d)
    for ii in range(n):
        res[:, ii, :] = np.ascontiguousarray(grad_basis[:, ii, :]) @ GVt
    return res


@numba.njit(cache=True)
def dot_grad_value_with_grad_basis(grad_value, grad_basis, dtype):
    """
    Grad(value) (k, k) (k==d) dotted with grad(basis) (k, n, k) -> (k, n, k).
    Loop only over 'n' with BLAS inside: res[:, n, :] = grad_value @ grad_basis[:, n, :]
    """
    k, n, d = grad_basis.shape
    if grad_value.shape[0] != k or grad_value.shape[1] != k or d != k:
        raise ValueError("Gradient value shape incompatible with grad basis (expect square k==d)")
    res = np.empty((k, n, k), dtype=dtype)
    GV = np.ascontiguousarray(grad_value)
    Gb = np.ascontiguousarray(grad_basis)
    for ii in range(n):
        Gb_slice = np.ascontiguousarray(Gb[:, ii, :])
        res[:, ii, :] = GV @ Gb_slice
    return res


@numba.njit(cache=True)
def dot_vec_vec(vec_a, vec_b, dtype):
    """Dot product between two vectors."""
    return float(np.dot(vec_a, vec_b))

@numba.njit(cache=True)
def dot_value_with_grad(value_vec, grad_mat, dtype):
    """
    Value (k,) · grad (k,d) -> (d,)
    """
    return np.ascontiguousarray(value_vec) @ np.ascontiguousarray(grad_mat)

@numba.njit(cache=True)
def dot_grad_with_value(grad_mat, value_vec, dtype):
    """
    Grad (k,d) · value (k,) -> (d,)
    """
    return np.ascontiguousarray(grad_mat) @ np.ascontiguousarray(value_vec)


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
    Vectorized over the last two axes; sum diagonal blocks.
    """
    k, n_test, n_trial, d_dim = tensor.shape
    n_diag = k if k < d_dim else d_dim
    res = np.zeros((1, n_test, n_trial), dtype=dtype)
    # Sum tensor[i, :, :, i] across i
    for i in range(n_diag):
        res[0] += tensor[i, :, :, i]
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
def inner_grad_function_grad_test(function_grad, test_grad, dtype):
    """
    Inner product between grad(Function) (k,d) and grad(Test) basis (k,n,d) -> (n,).
    Vectorized by flattening (k*d): (n x kd) @ (kd,)
    """
    k_comps, n_locs, d_dim = test_grad.shape
    if function_grad.shape[0] != k_comps or function_grad.shape[1] != d_dim:
        raise ValueError("Gradient(Function) shape incompatible with grad(Test)")
    
    # Vectorized implementation:
    # (k, n, d) -> (n, k, d)
    test_grad_T = test_grad.transpose(1, 0, 2)
    
    # --- THIS IS THE FIX ---
    # Ensure the array is contiguous *before* reshaping
    test_grad_contig = np.ascontiguousarray(test_grad_T)
    # -----------------------

    # Reshape (n, k, d) -> (n, k*d)
    test_grad_flat = test_grad_contig.reshape(n_locs, k_comps * d_dim)
    
    # Reshape (k, d) -> (k*d,)
    func_grad_flat = function_grad.reshape(k_comps * d_dim)
    
    # Perform (n, k*d) @ (k*d,) -> (n,)
    res = test_grad_flat @ func_grad_flat
    
    return res

@numba.njit(cache=True)
def basis_dot_const_vector(basis, const_vec, dtype):
    """
    Basis (k,n) dotted with constant vector (k,) -> (1,n).
    Vectorized: (n,k) @ (k,)
    """
    res = np.empty((1, basis.shape[1]), dtype=dtype)
    res[0] = np.ascontiguousarray(basis).T @ np.ascontiguousarray(const_vec)
    return res


@numba.njit(cache=True)
def const_vector_dot_basis(const_vec, basis, dtype):
    """
    Constant vector (k,) dotted with basis (k,n) -> (1,n).
    (same as above, order swapped)
    """
    res = np.empty((1, basis.shape[1]), dtype=dtype)
    res[0] = np.ascontiguousarray(basis).T @ np.ascontiguousarray(const_vec)
    return res


@numba.njit(cache=True)
def const_vector_dot_basis_1d(const_vec, basis, dtype):
    """
    Constant vector (k,) dotted with basis (k,n) -> (n,).
    """
    return np.ascontiguousarray(basis).T @ np.ascontiguousarray(const_vec)



@numba.njit(cache=True)
def scalar_basis_times_vector(scalar_basis, vector_vals, dtype):
    """
    Scalar basis (1,n) or (n,) times vector components (k,) -> (k,n).
    Broadcasting outer product.
    """
    if scalar_basis.ndim == 2:
        phi = scalar_basis[0]
    else:
        phi = scalar_basis
    return np.ascontiguousarray(vector_vals)[:, None] * np.ascontiguousarray(phi)[None, :]



@numba.njit(cache=True)
def matrix_times_scalar_basis(matrix_vals, scalar_basis, dtype):
    """
    Matrix (m, n) times scalar basis row (1,p) -> (1,p).
    Algebraically equals sum(matrix_vals) * scalar_basis.
    """
    if scalar_basis.ndim == 2:
        phi = scalar_basis[0]
    else:
        phi = scalar_basis
    s = float(np.sum(matrix_vals))
    res = np.empty((1, phi.shape[0]), dtype=dtype)
    res[0] = s * phi
    return res


@numba.njit(cache=True)
def scalar_vector_outer_product(scalar_vals, vector_vals, dtype):
    """
    Scalar values (n,) times vector (k,) -> (k,n).
    Broadcasting outer product.
    """
    return np.ascontiguousarray(vector_vals)[:, None] * np.ascontiguousarray(scalar_vals)[None, :]



@numba.njit(cache=True)
def scalar_trial_times_grad_test(grad_test, trial_vals, dtype):
    """
    Scalar Trial (n_trial,) times grad(Test) (k,n_test,d) -> (k,n_test,n_trial,d).
    Vectorized with broadcasting.
    """
    return (np.ascontiguousarray(grad_test)[:, :, None, :] *
            np.ascontiguousarray(trial_vals)[None, None, :, None])



@numba.njit(cache=True)
def grad_trial_times_scalar_test(grad_trial, test_vals, dtype):
    """
    Grad(Trial) (k,n_trial,d) times scalar Test (n_test,) -> (k,n_test,n_trial,d).
    Vectorized with broadcasting.
    """
    return (np.ascontiguousarray(grad_trial)[:, None, :, :] *
            np.ascontiguousarray(test_vals)[None, :, None, None])



@numba.njit(cache=True)
def scale_mixed_basis_with_coeffs(mixed_basis, coeffs, dtype):
    """
    Mixed basis (k_mixed, n_rows, n_cols) scaled by coeffs (k_out, d_cols)
    -> (k_out, n_rows, n_cols, d_cols).
    Since coeffs are independent of k_mixed, this is:
       coeffs[:,None,None,:] * sum_k mixed_basis[k]
    """
    base = np.sum(np.ascontiguousarray(mixed_basis), axis=0)  # (n_rows, n_cols)
    return (np.ascontiguousarray(coeffs)[:, None, None, :] *
            base[None, :, :, None])


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
    Vectorized sum over axis=0.
    """
    res = np.empty((1, a_mat.shape[1]), dtype=dtype)
    res[0] = np.sum(np.ascontiguousarray(a_mat) * np.ascontiguousarray(b_mat), axis=0)
    return res


@numba.njit(cache=True)
def hessian_dot_vector(hessian, vec, dtype):
    """
    Hessian (basis or value) dotted with spatial vector.
    Basis: (k,n,d1,d2) -> (k,n,d1)    Value: (k,d1,d2) -> (k,d1)
    """
    v = np.ascontiguousarray(vec)
    if hessian.ndim == 4:
        k, n, d1, _ = hessian.shape
        out = np.empty((k, n, d1), dtype=dtype)
        H = np.ascontiguousarray(hessian)
        for kk in range(k):
            for nn in range(n):
                out[kk, nn] = H[kk, nn] @ v
        return out
    elif hessian.ndim == 3:
        k, d1, _ = hessian.shape
        out = np.empty((k, d1), dtype=dtype)
        H = np.ascontiguousarray(hessian)
        for kk in range(k):
            out[kk] = H[kk] @ v
        return out
    else:
        raise ValueError("Unsupported Hessian shape for hessian_dot_vector")



@numba.njit(cache=True)
def vector_dot_hessian_basis(vec, hessian, dtype):
    """
    Vector dotted with Hessian basis (k,n,d1,d2).
    vlen==k: res[j] = vec @ hessian[:, :, j, :]   (batched via reshape)
    vlen==d1 & k==1: res[0,n,:] = vec @ hessian[0,n]
    """
    k, n, d1, d2 = hessian.shape
    v = np.ascontiguousarray(vec)
    H = np.ascontiguousarray(hessian)
    if v.shape[0] == k and k > 1:
        out = np.empty((d1, n, d2), dtype=dtype)
        for j in range(d1):
            
            # Slicing H[:, :, j, :] creates a non-contiguous 3D array.
            # We must make it contiguous before reshaping.
            H_slice = H[:, :, j, :]
            H_slice_contig = np.ascontiguousarray(H_slice)
            Hj = H_slice_contig.reshape(k, n * d2)      # (k, n*d2)

            tmp = v @ Hj                                # (n*d2,)
            out[j] = tmp.reshape(n, d2)
        return out
    elif v.shape[0] == d1 and k == 1:
        out = np.empty((1, n, d2), dtype=dtype)
        for nn in range(n):
            out[0, nn] = v @ H[0, nn]
        return out
    else:
        raise ValueError("vector·Hessian(basis): incompatible shapes")

@numba.njit(cache=True)
def vector_dot_hessian_value(vec, hessian, dtype):
    """
    Vector dotted with Hessian value (k,d1,d2):
    vlen==k: res[j,:] = vec @ hessian[:, j, :]
    vlen==d1 & k==1:  res[0,:] = vec @ hessian[0]
    """
    k, d1, d2 = hessian.shape
    v = np.ascontiguousarray(vec)
    H = np.ascontiguousarray(hessian)
    if v.shape[0] == k and k > 1:
        out = np.empty((d1, d2), dtype=dtype)
        for j in range(d1):
            Hj = np.ascontiguousarray(H[:, j, :])
            out[j] = v @ Hj
        return out
    elif v.shape[0] == d1 and k == 1:
        out = np.empty((1, d2), dtype=dtype)
        out[0] = v @ H[0]
        return out
    else:
        raise ValueError("vector·Hessian(value): incompatible shapes")

# ---------- “inner(·,·)” building blocks ----------

@numba.njit(cache=True)
def inner_hessian_function_hessian_test(function_hess, test_hess, dtype):
    """
    Inner product between Hess(Function) (k,d,d) and Hess(Test) (k,n,d,d) -> (n,).
    Vectorized by flattening the (d*d) block and doing (n x dd) @ (dd,)
    """
    k, n, d, d2 = test_hess.shape
    if function_hess.shape[0] != k or function_hess.shape[1] != d or function_hess.shape[2] != d2:
        raise ValueError("Hessian(Function) shape incompatible with Hessian(Test)")
    res = np.zeros(n, dtype=dtype)
    dd = d * d2
    fh = np.ascontiguousarray(function_hess).reshape(k, dd)
    th = np.ascontiguousarray(test_hess).reshape(k, n, dd)
    for comp in range(k):
        res += th[comp] @ fh[comp]
    return res


@numba.njit(cache=True)
def inner_mixed_grad_const(mixed_grad, grad_const, dtype):
    """
    Inner of mixed grad (k, n_test, n_trial, d) with grad(const) (k, d) -> (n_test, n_trial).
    Vectorized by flattening (k*d) and a single matvec.
    """
    k, n_test, n_trial, d = mixed_grad.shape
    # (k, n_test, n_trial, d) -> (n_test, n_trial, k, d)
    A_transposed = np.ascontiguousarray(mixed_grad).transpose(1, 2, 0, 3)
    # Ensure the array is contiguous *before* reshaping
    A_contig = np.ascontiguousarray(A_transposed)
    # (n_test, n_trial, k, d) -> (n_test * n_trial, k * d)
    A = A_contig.reshape(n_test * n_trial, k * d)
    b = np.ascontiguousarray(grad_const).reshape(k * d)
    return (A @ b).reshape(n_test, n_trial)


@numba.njit(cache=True)
def inner_grad_const_mixed(grad_const, mixed_grad, dtype):
    """
    Inner of grad(const/value) (k, d) with mixed grad (k, n_test, n_trial, d) -> (n_test, n_trial).
    Same vectorization as above.
    """
    k, n_test, n_trial, d = mixed_grad.shape
    A_trasposed = np.ascontiguousarray(mixed_grad).transpose(1, 2, 0, 3)
    A_contg = np.ascontiguousarray(A_trasposed)
    A = A_contg.reshape(n_test * n_trial, k * d)
    b = np.ascontiguousarray(grad_const).reshape(k * d)
    return (A @ b).reshape(n_test, n_trial)


@numba.njit(cache=True)
def inner_grad_basis_grad_const(grad_basis, grad_const, dtype):
    """
    Inner grad(basis) (k, n, d) with grad(const/value) (k, d) -> (n,).
    Vectorized by flattening (k*d): (n x kd) @ (kd,)
    """
    k, n, d = grad_basis.shape
    A = np.ascontiguousarray(grad_basis).transpose(1, 0, 2).reshape(n, k * d)
    b = np.ascontiguousarray(grad_const).reshape(k * d)
    return A @ b


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
    """
    Evaluate u_h at a quadrature point.
    u_e: (ndof,) or (ndof, ncomp); phi_q: (ndof,)
    """
    if u_e.ndim == 1:
        # (ndof,) @ (ndof,) -> scalar
        return float(np.dot(u_e, phi_q))
    # (ndof, ncomp).T @ (ndof,) -> (ncomp, ndof) @ (ndof,) -> (ncomp,)
    return (u_e.T @ phi_q).astype(u_e.dtype)


@numba.njit(cache=True)
def gradient_qp(u_e, grad_phi_q):
    """
    Evaluate ∇u_h at a qp. grad_phi_q: (ndof, dim)
    Returns: scalar -> (dim,), vector -> (dim, ncomp)
    """
    if u_e.ndim == 1:
        # (ndof, dim).T @ (ndof,) -> (dim, ndof) @ (ndof,) -> (dim,)
        return (grad_phi_q.T @ u_e).astype(u_e.dtype)
    # (ndof, dim).T @ (ndof, ncomp) -> (dim, ndof) @ (ndof, ncomp) -> (dim, ncomp)
    return (grad_phi_q.T @ u_e).astype(u_e.dtype)


@numba.njit(cache=True)
def laplacian_qp(u_e, lap_phi_q):
    """
    Evaluate Δu_h at a qp. lap_phi_q: (ndof,)
    Returns: scalar -> float, vector -> (ncomp,)
    """
    if u_e.ndim == 1:
        # (ndof,) @ (ndof,) -> scalar
        return float(np.dot(u_e, lap_phi_q))
    # (ndof, ncomp).T @ (ndof,) -> (ncomp, ndof) @ (ndof,) -> (ncomp,)
    return (u_e.T @ lap_phi_q).astype(u_e.dtype)


@numba.njit(cache=True)
def hessian_qp(u_e, hess_phi_q):
    """
    Evaluate ∇²u_h at a qp.
    hess_phi_q: (ndof, dim, dim)
    Returns: scalar -> (dim,dim), vector -> (ncomp, dim, dim)
    """
    ndof = hess_phi_q.shape[0]
    dim  = hess_phi_q.shape[1]
    
    # (ndof, dim, dim) -> (ndof, dim*dim)
    Hflat = np.ascontiguousarray(hess_phi_q).reshape(ndof, dim * dim)
    
    if u_e.ndim == 1:
        # (ndof, dd).T @ (ndof,) -> (dd, ndof) @ (ndof,) -> (dd,)
        out = (Hflat.T @ u_e).reshape(dim, dim)
        return out.astype(u_e.dtype)
    
    # (ndof, ncomp).T @ (ndof, dd) -> (ncomp, ndof) @ (ndof, dd) -> (ncomp, dd)
    ncomp = u_e.shape[1]
    out2 = (u_e.T @ Hflat).reshape(ncomp, dim, dim)
    return out2.astype(u_e.dtype)


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
