import numba
import numpy as np
use_type = np.float64
DEBUG = False
@numba.njit(cache=True, fastmath=True)
def ghost_grad_jump_penalty_scalar(
    Ke, w, cell_h, normals, grad_pos, grad_neg, gamma, dtype
):
    """
    Assemble ghost gradient-jump penalty for scalar fields.
    Ke      : (nE, n_union, n_union)
    w       : (nE, n_q) physical quadrature weights
    cell_h  : (nE,) background cell diameter
    normals : (nE, n_q, 2)
    grad_pos/grad_neg : (nE, n_q, n_union, 2)
    """
    if DEBUG:print("ghost_grad_jump_penalty_scalar")
    nE, n_q, n_union, _ = grad_pos.shape
    for e in range(nE):
        h_e = cell_h[e]
        for q in range(n_q):
            wq = w[e, q] * gamma * h_e
            if wq == 0.0:
                continue
            n0 = normals[e, q, 0]
            n1 = normals[e, q, 1]
            jump = np.empty(n_union, dtype=dtype)
            for i in range(n_union):
                gx_pos = grad_pos[e, q, i, 0]
                gy_pos = grad_pos[e, q, i, 1]
                gx_neg = grad_neg[e, q, i, 0]
                gy_neg = grad_neg[e, q, i, 1]
                jump[i] = (gx_pos - gx_neg) * n0 + (gy_pos - gy_neg) * n1
            for i in range(n_union):
                vi = jump[i] * wq
                for j in range(n_union):
                    Ke[e, i, j] += vi * jump[j]


@numba.njit(cache=True, fastmath=True)
def ghost_grad_jump_penalty_vector(
    Ke, w, cell_h, normals, grad_pos, grad_neg, gamma, dtype
):
    """
    Assemble ghost gradient-jump penalty for vector-valued fields.
    grad_pos/grad_neg : (nE, n_q, n_comp, n_union, 2)
    """
    if DEBUG:print("ghost_grad_jump_penalty_vector")
    nE, n_q, n_comp, n_union, _ = grad_pos.shape
    for e in range(nE):
        h_e = cell_h[e]
        for q in range(n_q):
            wq = w[e, q] * gamma * h_e
            if wq == 0.0:
                continue
            n0 = normals[e, q, 0]
            n1 = normals[e, q, 1]
            jump_n = np.empty((n_union, n_comp), dtype=dtype)
            for i in range(n_union):
                for k in range(n_comp):
                    gx_pos = grad_pos[e, q, k, i, 0]
                    gy_pos = grad_pos[e, q, k, i, 1]
                    gx_neg = grad_neg[e, q, k, i, 0]
                    gy_neg = grad_neg[e, q, k, i, 1]
                    jump_n[i, k] = (gx_pos - gx_neg) * n0 + (gy_pos - gy_neg) * n1
            for i in range(n_union):
                for j in range(n_union):
                    s = 0.0
                    for k in range(n_comp):
                        s += jump_n[i, k] * jump_n[j, k]
                    Ke[e, i, j] += wq * s

@numba.njit(cache=True)
def dot_grad_grad_mixed(a, b, flag, dtype):
    """
    Matmul for grad(test/trial) · grad(trial/test).
    """
    if DEBUG: print("dot_grad_grad_mixed")
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
    if DEBUG: print("contract_last_first")
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
def contract_first_first(a, b, dtype):
    """
    Generic contraction over the first axis of a and first axis of b.
    """
    if DEBUG: print("contract_first_first")
    if b.ndim == 0:
        raise ValueError("contract_first_first: second operand must have at least 1 dimension")

    if b.ndim == 1:
        b_flat = np.ascontiguousarray(b)
    elif b.ndim == 2 and (b.shape[0] == 1 or b.shape[1] == 1):
        b_flat = np.ascontiguousarray(b).reshape(b.size)
    else:
        raise ValueError("contract_first_first: second operand must be a 1D vector or row/column matrix")

    if a.shape[0] != b_flat.shape[0]:
        raise ValueError("Dot dimension mismatch")

    # --- Setup for 'a' ---
    left_dim = 1
    for idx in range(1, a.ndim):
        left_dim *= a.shape[idx]
    # a_flat has shape (k, m*n)
    a_flat = np.ascontiguousarray(a).reshape(a.shape[0], left_dim)
    
    # --- Setup for 'b'  ---
    a_flat_t = np.ascontiguousarray(a_flat.T)
    # dot_flat is (m*n, k) @ (k,) -> (m*n,)
    dot_flat = a_flat_t @ b_flat
    # out_shape is (m, n)
    out_shape = a.shape[1:]
    # (m*n,).reshape(m, n) -> Correct
    return dot_flat.reshape(out_shape)



@numba.njit(cache=True)
def dot_mixed_const(a, b, dtype):
    """
    Mixed basis (k, n, m) dotted with constant vector (k,) -> (n, m).
    Vectorized via (k*nm) matvec.
    """
    if DEBUG: print("dot_mixed_const")
    if b.ndim == 2:
        if b.shape[0] == 1 or b.shape[1] == 1:
            b = np.ascontiguousarray(b).reshape(b.size)
        else:
            raise ValueError("dot_mixed_const: constant operand must be 1D or row/column matrix")
    elif b.ndim != 1:
        raise ValueError("dot_mixed_const: constant operand must be 1D or row/column matrix")
    k = a.shape[0]
    nm = a.shape[1] * a.shape[2]
    a2 = np.ascontiguousarray(a).reshape(k, nm)
    a2_t = np.ascontiguousarray(a2.T)
    return (a2_t @ np.ascontiguousarray(b)).reshape(a.shape[1], a.shape[2])



@numba.njit(cache=True)
def dot_const_mixed(a, b, dtype):
    """
    Constant vector (k,) dotted with mixed basis (k, n, m) -> (n, m).
    Vectorized via (1 x k) @ (k x nm).
    """
    if DEBUG: print("dot_const_mixed")
    if a.ndim == 2:
        if a.shape[0] == 1 or a.shape[1] == 1:
            a = np.ascontiguousarray(a).reshape(a.size)
        else:
            raise ValueError("dot_const_mixed: constant operand must be 1D or row/column matrix")
    elif a.ndim != 1:
        raise ValueError("dot_const_mixed: constant operand must be 1D or row/column matrix")
    k = b.shape[0]
    nm = b.shape[1] * b.shape[2]
    b2 = np.ascontiguousarray(b).reshape(k, nm)
    return (np.ascontiguousarray(a) @ b2).reshape(b.shape[1], b.shape[2])



@numba.njit(cache=True)
def dot_vec_grad_components(vec_basis, grad_basis, swap_roles, dtype):
    """
    Contract vector basis components with gradient basis components.

    vec_basis : (k, n_vec)
    grad_basis:
      - scalar gradient basis: (d, n_grad)
      - vector gradient basis: (k, n_grad, d)

    Canonical ordering is rows=test, cols=trial. `swap_roles=True` means the
    left vector basis is the test side and the right gradient basis is the
    trial side.
    """
    if DEBUG: print("dot_vec_grad_components")
    if grad_basis.ndim == 2:
        if vec_basis.shape[0] != grad_basis.shape[0]:
            raise ValueError("dot_vec_grad_components: scalar-gradient component mismatch")
        if swap_roles:
            return np.ascontiguousarray(vec_basis.T) @ np.ascontiguousarray(grad_basis)
        return np.ascontiguousarray(grad_basis.T) @ np.ascontiguousarray(vec_basis)

    k_vec, n_vec = vec_basis.shape
    k_grad, n_grad, d_dim = grad_basis.shape
    if k_vec != k_grad:
        raise ValueError("dot_vec_grad_components: vector/gradient component mismatch")

    n_rows = n_vec if swap_roles else n_grad
    n_cols = n_grad if swap_roles else n_vec
    out = np.zeros((d_dim, n_rows, n_cols), dtype=dtype)
    vec_basis_c = np.ascontiguousarray(vec_basis)
    grad_basis_c = np.ascontiguousarray(grad_basis)
    for r in range(d_dim):
        if swap_roles:
            out[r] = np.ascontiguousarray(vec_basis_c.T) @ np.ascontiguousarray(grad_basis_c[:, :, r])
        else:
            out[r] = np.ascontiguousarray(grad_basis_c[:, :, r].T) @ vec_basis_c
    return out


@numba.njit(cache=True)
def dot_vector_trial_grad_test(trial_vec, grad_test, dtype):
    """
    Vector trial (k, n_trial) · grad(test) (k, n_test, d) -> (d, n_test, n_trial).
    For each spatial dim j: res[j] = grad_test[:, :, j].T @ trial_vec
    """
    if DEBUG: print("dot_vector_trial_grad_test")
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
    Gradient basis dotted with spatial vector.

    Supports:
      (k, n, d) -> (k, n)
      (d, n)    -> (1, n)
      (n, d)    -> (1, n)
    """
    if DEBUG: print("dot_grad_basis_vector")
    if grad_basis.ndim == 2:
        if grad_basis.shape[0] == vec.shape[0]:
            res = np.empty((1, grad_basis.shape[1]), dtype=dtype)
            grad_basis_t = np.ascontiguousarray(grad_basis.T)
            res[0] = grad_basis_t @ np.ascontiguousarray(vec)
            return res
        if grad_basis.shape[1] == vec.shape[0]:
            res = np.empty((1, grad_basis.shape[0]), dtype=dtype)
            res[0] = np.ascontiguousarray(grad_basis) @ np.ascontiguousarray(vec)
            return res
        raise ValueError("dot_grad_basis_vector: incompatible rank-2 basis/vector shapes")
    k, n, d = grad_basis.shape
    G = np.ascontiguousarray(grad_basis).reshape(k * n, d)
    out = G @ np.ascontiguousarray(vec)
    return np.ascontiguousarray(out.reshape(k, n))


@numba.njit(cache=True)
def vector_dot_grad_basis(vec, grad_basis, dtype):
    """
    Vector (component) dotted with gradient basis (k, n, d).
    Returns (1, n) for scalar fields or (d, n) when len(vec)==k.
    """
    if DEBUG: print("vector_dot_grad_basis")
    if grad_basis.ndim == 2:
        if grad_basis.shape[0] == vec.shape[0]:
            res = np.empty((1, grad_basis.shape[1]), dtype=dtype)
            grad_basis_t = np.ascontiguousarray(grad_basis.T)
            res[0] = grad_basis_t @ np.ascontiguousarray(vec)
            return res
        if grad_basis.shape[1] == vec.shape[0]:
            res = np.empty((1, grad_basis.shape[0]), dtype=dtype)
            res[0] = np.ascontiguousarray(grad_basis) @ np.ascontiguousarray(vec)
            return res
        raise ValueError("vector_dot_grad_basis: incompatible rank-2 basis/vector shapes")
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
def vector_dot_grad_value(vec_basis, grad_value, dtype):
    """
    Vector basis (k, n) dotted with grad(value) (k, d) -> (d, n).
    res[p, j] = sum_i vec_basis[i, j] * grad_value[i, p]
    """
    if DEBUG: print("vector_dot_grad_value")
    k, n = vec_basis.shape
    if grad_value.shape[0] == 1 and k == grad_value.shape[1]:
        res = np.empty((1, n), dtype=dtype)
        g = np.ascontiguousarray(grad_value[0])
        V = np.ascontiguousarray(vec_basis)
        for j in range(n):
            col = np.ascontiguousarray(V[:, j])
            acc = 0.0
            for i in range(k):
                acc += col[i] * g[i]
            res[0, j] = acc
        return res
    if grad_value.shape[0] != k:
        raise ValueError("vector·grad value: incompatible shapes")
    d = grad_value.shape[1]
    res = np.empty((d, n), dtype=dtype)
    V = np.ascontiguousarray(vec_basis)   # (k, n)
    G = np.ascontiguousarray(grad_value)  # (k, d)
    for j in range(n):
        col = np.ascontiguousarray(V[:, j])
        for p in range(d):
            acc = 0.0
            for i in range(k):
                acc += col[i] * G[i, p]
            res[p, j] = acc
    return res


@numba.njit(cache=True)
def dot_grad_basis_with_grad_value(grad_basis, grad_value, dtype):
    """
    Grad(basis) (k, n, d) dotted with grad(value) (k, d) -> (k, n, k).
    For each n: (k,d) @ (d,k)  (transpose on grad_value).
    """
    if DEBUG: print("dot_grad_basis_with_grad_value")
    k, n, d = grad_basis.shape
    if grad_value.shape[0] != k or grad_value.shape[1] != d:
        raise ValueError("Gradient value shape incompatible with basis gradient")
    res = np.empty((k, n, k), dtype=dtype)
    GV = np.ascontiguousarray(grad_value)  # (k, d)
    for ii in range(n):
        res[:, ii, :] = np.ascontiguousarray(grad_basis[:, ii, :]) @ GV
    return res


@numba.njit(cache=True)
def dot_grad_value_with_grad_basis(grad_value, grad_basis, dtype):
    """
    Grad(value) (k, k) (k==d) dotted with grad(basis) (k, n, k) -> (k, n, k).
    Loop only over 'n' with BLAS inside: res[:, n, :] = grad_value @ grad_basis[:, n, :]
    """
    if DEBUG: print("dot_grad_value_with_grad_basis")
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
    if DEBUG: print("dot_vec_vec")
    return float(np.dot(np.ascontiguousarray(vec_a), np.ascontiguousarray(vec_b)))

@numba.njit(cache=True)
def dot_value_with_grad(value_vec, grad_mat, dtype):
    """
    Value (k,) · grad (k,d) -> (d,)
    """
    if DEBUG: print("dot_value_with_grad")
    return np.ascontiguousarray(value_vec) @ np.ascontiguousarray(grad_mat)

@numba.njit(cache=True)
def dot_grad_with_value(grad_mat, value_vec, dtype):
    """
    Grad (k,d) · value (k,) -> (d,)
    """
    if DEBUG: print("dot_grad_with_value")
    return np.ascontiguousarray(grad_mat) @ np.ascontiguousarray(value_vec)


@numba.njit(cache=True)
def dot_grad_func_trial_vec(grad_func, trial_vec, dtype):
    """
    Compute grad(Function) @ Trial vector basis.
    """
    if DEBUG: print("dot_grad_func_trial_vec")
    return np.ascontiguousarray(grad_func) @ np.ascontiguousarray(trial_vec)


@numba.njit(cache=True)
def dot_trial_vec_grad_func(trial_vec, grad_func, dtype):
    """
    Compute dot(Trial vector basis, grad(Function)).
    """
    if DEBUG: print("dot_trial_vec_grad_func")
    if grad_func.shape[0] == 1 and trial_vec.shape[0] == grad_func.shape[1]:
        return np.ascontiguousarray(grad_func) @ np.ascontiguousarray(trial_vec)
    return np.ascontiguousarray(grad_func.T.copy()) @ np.ascontiguousarray(trial_vec)


@numba.njit(cache=True)
def dot_vec_vec(vec_a, vec_b, dtype):
    """
    Dot product between two vectors.
    """
    if DEBUG: print("dot_vec_vec")
    return np.dot(np.ascontiguousarray(vec_a), np.ascontiguousarray(vec_b))


@numba.njit(cache=True)
def dot_grad_grad_value(grad_a, grad_b, dtype):
    """
    Compute grad(value) @ grad(value).
    """
    if DEBUG: print("dot_grad_grad_value")
    return np.ascontiguousarray(grad_a) @ np.ascontiguousarray(grad_b)





@numba.njit(cache=True)
def mul_scalar(scalar, array, dtype):
    """
    Multiply scalar with array.
    """
    # if DEBUG: print("mul_scalar")
    return float(scalar) * array


@numba.njit(cache=True)
def _flatten_to_1d(array_like, dtype):
    """
    Convert scalar/array-like input to contiguous 1D array of dtype.
    """
    # if DEBUG: print("_flatten_to_1d")
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
    # if DEBUG: print("_ensure_matrix")
    arr = np.asarray(array_like, dtype=dtype)
    if arr.ndim == 0:
        res = np.zeros((1, 1), dtype=dtype)
        res[0, 0] = arr.item()
    elif arr.ndim == 1:
        res = arr.reshape(arr.shape[0], 1)
    else:
        rows = arr.shape[0]
        cols = arr.size // rows
        res = arr.reshape(rows, cols)
    return np.ascontiguousarray(res)


@numba.njit(cache=True)
def trace_matrix_value(matrix, dtype):
    """
    Trace of a dense square matrix -> scalar.
    """
    # if DEBUG: print("trace_matrix_value")
    return float(np.trace(np.ascontiguousarray(matrix)))


@numba.njit(cache=True)
def trace_basis_tensor(tensor, dtype):
    """
    Trace of a basis tensor (k, n, k) -> (1, n).
    """
    # if DEBUG: print("trace_basis_tensor")
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
    if DEBUG: print("trace_mixed_tensor")
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
    # if DEBUG: print("transpose_grad_tensor")
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
    if DEBUG: print("transpose_mixed_grad_tensor")
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
    if DEBUG: print("transpose_hessian_tensor")
    return tensor.swapaxes(2, 3).copy()


@numba.njit(cache=True)
def transpose_matrix(matrix, dtype):
    """
    Transpose a 2D matrix.
    """
    # if DEBUG: print("transpose_matrix")
    return matrix.T.copy()


@numba.njit(cache=True)
def swap_mixed_basis_tensor(tensor, dtype):
    """
    Swap the two basis axes of a mixed tensor.
    Supports:
      (n_test, n_trial) <-> (n_trial, n_test)
      (free..., n_test, n_trial, free...) with basis axes in slots 1/2
    """
    if tensor.ndim == 2:
        return np.ascontiguousarray(tensor.T.copy())
    if tensor.ndim >= 3:
        return np.ascontiguousarray(np.swapaxes(tensor, 1, 2).copy())
    raise ValueError(f"swap_mixed_basis_tensor: unsupported ndim={tensor.ndim}")


@numba.njit(cache=True)
def scatter_tensor_to_union(values, mapping, n_union, dtype):
    """
    Scatter local rows (m, ...) into a union-sized tensor (n_union, ...).
    """
    if DEBUG: print("scatter_tensor_to_union")
    out_shape = (n_union,) + values.shape[1:]
    res = np.zeros(out_shape, dtype=dtype)
    m = values.shape[0]
    for j in range(m):
        idx = mapping[j]
        if 0 <= idx < n_union:
            res[idx] = values[j]
    return res


@numba.njit(cache=True, fastmath=True)
def pad_basis_to_union(local, mapping, n_union, s0, s1, dtype):
    """
    Pad a 1D local basis/derivative vector to the union layout using a side map.
    """
    if DEBUG: print("pad_basis_to_union")
    if n_union == local.shape[0]:
        return local.copy()
    m = mapping.shape[0]
    loc_vec = local if local.shape[0] == m else local[s0:s1]
    out = np.zeros(n_union, dtype=dtype)
    for j in range(m):
        idx = mapping[j]
        if 0 <= idx < n_union:
            out[idx] = loc_vec[j]
    return out


@numba.njit(cache=True, fastmath=True)
def pushforward_grad_to_union(d10, d01, j_inv, mapping, n_union, s0, s1, dtype):
    """
    Push forward (d10,d01) with J^{-1} and scatter to the union layout.
    """
    if DEBUG: print("pushforward_grad_to_union")
    grad_loc = np.ascontiguousarray(np.stack((d10, d01), axis=1)) @ np.ascontiguousarray(j_inv.copy())
    if grad_loc.shape[0] == n_union:
        return grad_loc
    return scatter_tensor_to_union(grad_loc[s0:s1], mapping, n_union, dtype)


@numba.njit(cache=True)
def compute_physical_hessian(d20, d11, d02, d10, d01, j_inv, hx, hy, dtype):
    """
    Compute physical Hessian for a single component.
    """
    if DEBUG: print("compute_physical_hessian")
    nloc = d20.shape[0]
    d_dim = j_inv.shape[0]
    j_inv_c = np.ascontiguousarray(j_inv)
    j_inv_t = np.ascontiguousarray(j_inv.T.copy())
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
        core = j_inv_t @ (np.ascontiguousarray(href) @ j_inv_c)
        res[j] = core + d10[j] * hx + d01[j] * hy
    return res


@numba.njit(cache=True)
def compute_physical_laplacian(d20, d11, d02, d10, d01, j_inv, hx, hy, dtype):
    """
    Compute physical Laplacian entries for scalar component.
    """
    if DEBUG: print("compute_physical_laplacian")
    nloc = d20.shape[0]
    j_inv_c = np.ascontiguousarray(j_inv)
    j_inv_t = np.ascontiguousarray(j_inv.T.copy())
    res = np.zeros(nloc, dtype=dtype)
    for j in range(nloc):
        href = np.zeros((2, 2), dtype=dtype)
        href[0, 0] = d20[j]
        href[0, 1] = d11[j]
        href[1, 0] = d11[j]
        href[1, 1] = d02[j]
        core = j_inv_t @ (np.ascontiguousarray(href) @ j_inv_c)
        hphys = core + d10[j] * hx + d01[j] * hy
        res[j] = hphys[0, 0] + hphys[1, 1]
    return res


@numba.njit(cache=True)
def pushforward_d3(d10, d01, d20, d11, d02, d30, d21, d12, d03, A, Hx, Hy, Tx0, Tx1, axes, dtype):
    """
    Exact third-order pullback using inverse-map jets.
    Returns a (nloc,) array for the derivative specified by 'axes'.
    """
    if DEBUG: print("pushforward_d3")
    nloc = d20.shape[0]
    res = np.zeros(nloc, dtype=dtype)
    for j in range(nloc):
        s = 0.0
        for a in (0, 1):
            for b in (0, 1):
                for c in (0, 1):
                    ones = (a == 1) + (b == 1) + (c == 1)
                    if ones == 0:
                        g3 = d30[j]
                    elif ones == 1:
                        g3 = d21[j]
                    elif ones == 2:
                        g3 = d12[j]
                    else:
                        g3 = d03[j]
                    s += g3 * A[a, axes[0]] * A[b, axes[1]] * A[c, axes[2]]
        for a in (0, 1):
            for b in (0, 1):
                if a == 0 and b == 0:
                    g2 = d20[j]
                elif a != b:
                    g2 = d11[j]
                else:
                    g2 = d02[j]
                Hb = Hx if b == 0 else Hy
                s += g2 * (
                    A[a, axes[0]] * Hb[axes[1], axes[2]]
                    + A[a, axes[1]] * Hb[axes[0], axes[2]]
                    + A[a, axes[2]] * Hb[axes[0], axes[1]]
                )
        s += d10[j] * Tx0[axes[0], axes[1], axes[2]] + d01[j] * Tx1[axes[0], axes[1], axes[2]]
        res[j] = s
    return res


@numba.njit(cache=True)
def pushforward_d4(
    d10,
    d01,
    d20,
    d11,
    d02,
    d30,
    d21,
    d12,
    d03,
    d40,
    d31,
    d22,
    d13,
    d04,
    A,
    Hx,
    Hy,
    Tx0,
    Tx1,
    Qx0,
    Qx1,
    axes,
    dtype,
):
    """
    Exact fourth-order pullback using inverse-map jets.
    Returns a (nloc,) array for the derivative specified by 'axes'.
    """
    if DEBUG: print("pushforward_d4")
    nloc = d20.shape[0]
    res = np.zeros(nloc, dtype=dtype)
    for j in range(nloc):
        s = 0.0
        for a in (0, 1):
            for b in (0, 1):
                for c in (0, 1):
                    for d in (0, 1):
                        ones = (a == 1) + (b == 1) + (c == 1) + (d == 1)
                        if ones == 0:
                            g4 = d40[j]
                        elif ones == 1:
                            g4 = d31[j]
                        elif ones == 2:
                            g4 = d22[j]
                        elif ones == 3:
                            g4 = d13[j]
                        else:
                            g4 = d04[j]
                        s += g4 * A[a, axes[0]] * A[b, axes[1]] * A[c, axes[2]] * A[d, axes[3]]
        for a in (0, 1):
            for b in (0, 1):
                for c in (0, 1):
                    ones = a + b + c
                    if ones == 0:
                        g3v = d30[j]
                    elif ones == 1:
                        g3v = d21[j]
                    elif ones == 2:
                        g3v = d12[j]
                    else:
                        g3v = d03[j]
                    for holder in (0, 1, 2):
                        hb = [a, b, c][holder]
                        Hb = Hx if hb == 0 else Hy
                        others = [a, b, c][:holder] + [a, b, c][holder + 1 :]
                        for p in range(4):
                            for q in range(p + 1, 4):
                                r = [0, 1, 2, 3]
                                r.remove(p)
                                r.remove(q)
                                s += g3v * Hb[axes[p], axes[q]] * A[others[0], axes[r[0]]] * A[others[1], axes[r[1]]]
        for a in (0, 1):
            for b in (0, 1):
                if a == 0 and b == 0:
                    g2v = d20[j]
                elif a != b:
                    g2v = d11[j]
                else:
                    g2v = d02[j]
                Ha = Hx if a == 0 else Hy
                Hb = Hx if b == 0 else Hy
                s += g2v * (
                    Ha[axes[0], axes[1]] * Hb[axes[2], axes[3]]
                    + Ha[axes[0], axes[2]] * Hb[axes[1], axes[3]]
                    + Ha[axes[0], axes[3]] * Hb[axes[1], axes[2]]
                )
                Tb = Tx0 if b == 0 else Tx1
                Ta = Tx0 if a == 0 else Tx1
                for p in range(4):
                    rest = [axes[i] for i in range(4) if i != p]
                    s += g2v * (A[a, axes[p]] * Tb[rest[0], rest[1], rest[2]] + A[b, axes[p]] * Ta[rest[0], rest[1], rest[2]])
        s += d10[j] * Qx0[axes[0], axes[1], axes[2], axes[3]] + d01[j] * Qx1[axes[0], axes[1], axes[2], axes[3]]
        res[j] = s
    return res

@numba.njit(cache=True)
def dot_mass_test_trial(test_vec, trial_vec, dtype):
    """
    Compute Test.T @ Trial for mass matrices.
    Accepts basis/value carriers whose basis axis is the second axis for ndim >= 3.
    Flattens all non-basis/free axes into the contraction dimension and always returns (n, n).
    """
    if DEBUG: print("dot_mass_test_trial")
    # Normalize to 2-D with contraction/free axes along axis 0 and basis along axis 1.
    if test_vec.ndim == 1:
        test_arr = test_vec.reshape(1, test_vec.shape[0])
    elif test_vec.ndim == 2:
        test_arr = np.ascontiguousarray(test_vec)
    elif test_vec.ndim == 3:
        # Basis carriers use layout (free0, n, free1).
        test_arr = np.ascontiguousarray(test_vec.transpose(0, 2, 1)).reshape(
            test_vec.shape[0] * test_vec.shape[2],
            test_vec.shape[1],
        )
    elif test_vec.ndim == 4:
        # Basis tensor carriers use layout (free0, n, free1, free2).
        test_arr = np.ascontiguousarray(test_vec.transpose(0, 2, 3, 1)).reshape(
            test_vec.shape[0] * test_vec.shape[2] * test_vec.shape[3],
            test_vec.shape[1],
        )
    else:
        raise ValueError(f"dot_mass_test_trial: unsupported test_vec ndim={test_vec.ndim}")

    if trial_vec.ndim == 1:
        trial_arr = trial_vec.reshape(1, trial_vec.shape[0])
    elif trial_vec.ndim == 2:
        trial_arr = np.ascontiguousarray(trial_vec)
    elif trial_vec.ndim == 3:
        trial_arr = np.ascontiguousarray(trial_vec.transpose(0, 2, 1)).reshape(
            trial_vec.shape[0] * trial_vec.shape[2],
            trial_vec.shape[1],
        )
    elif trial_vec.ndim == 4:
        trial_arr = np.ascontiguousarray(trial_vec.transpose(0, 2, 3, 1)).reshape(
            trial_vec.shape[0] * trial_vec.shape[2] * trial_vec.shape[3],
            trial_vec.shape[1],
        )
    else:
        raise ValueError(f"dot_mass_test_trial: unsupported trial_vec ndim={trial_vec.ndim}")

    return test_arr.T.copy() @ trial_arr
@numba.njit(cache=True)
def dot_mass_trial_test(trial_vec, test_vec, dtype):
    """
    Compute Trial.T @ Test for mass matrices.
    """
    if DEBUG: print("dot_mass_trial_test")
    if trial_vec.ndim == 1:
        trial_arr = trial_vec.reshape(1, trial_vec.shape[0])
    elif trial_vec.ndim == 2:
        trial_arr = np.ascontiguousarray(trial_vec)
    elif trial_vec.ndim == 3:
        trial_arr = np.ascontiguousarray(trial_vec.transpose(0, 2, 1)).reshape(
            trial_vec.shape[0] * trial_vec.shape[2],
            trial_vec.shape[1],
        )
    elif trial_vec.ndim == 4:
        trial_arr = np.ascontiguousarray(trial_vec.transpose(0, 2, 3, 1)).reshape(
            trial_vec.shape[0] * trial_vec.shape[2] * trial_vec.shape[3],
            trial_vec.shape[1],
        )
    else:
        raise ValueError(f"dot_mass_trial_test: unsupported trial_vec ndim={trial_vec.ndim}")

    if test_vec.ndim == 1:
        test_arr = test_vec.reshape(1, test_vec.shape[0])
    elif test_vec.ndim == 2:
        test_arr = np.ascontiguousarray(test_vec)
    elif test_vec.ndim == 3:
        test_arr = np.ascontiguousarray(test_vec.transpose(0, 2, 1)).reshape(
            test_vec.shape[0] * test_vec.shape[2],
            test_vec.shape[1],
        )
    elif test_vec.ndim == 4:
        test_arr = np.ascontiguousarray(test_vec.transpose(0, 2, 3, 1)).reshape(
            test_vec.shape[0] * test_vec.shape[2] * test_vec.shape[3],
            test_vec.shape[1],
        )
    else:
        raise ValueError(f"dot_mass_trial_test: unsupported test_vec ndim={test_vec.ndim}")

    return trial_arr.T.copy() @ test_arr

@numba.njit(cache=True)
def inner_grad_function_grad_test(function_grad, test_grad, dtype):
    """
    Inner product between grad(Function) (k,d) and grad(Test) basis (k,n,d) -> (n,).
    Vectorized by flattening (k*d): (n x kd) @ (kd,)
    """
    if DEBUG: print("inner_grad_function_grad_test")
    if test_grad.ndim == 2:
        if function_grad.ndim == 2:
            if function_grad.shape[0] != 1 or function_grad.shape[1] != test_grad.shape[0]:
                raise ValueError("Gradient(Function) shape incompatible with scalar grad(Test)")
            test_grad_t = np.ascontiguousarray(test_grad.T)
            return test_grad_t @ np.ascontiguousarray(function_grad[0])
        if function_grad.ndim == 1:
            if function_grad.shape[0] != test_grad.shape[0]:
                raise ValueError("Gradient(Function) shape incompatible with scalar grad(Test)")
            test_grad_t = np.ascontiguousarray(test_grad.T)
            return test_grad_t @ np.ascontiguousarray(function_grad)
        raise ValueError("Gradient(Function) shape incompatible with scalar grad(Test)")
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
    Rank-1 basis carrier dotted with constant vector -> basis row.

    Supports canonical basis storage `(k, n)`, free-last planner views `(n, k)`,
    and carried rank-1 tensor basis layouts `(k, n, d)` where the basis axis is 1.
    """
    if DEBUG: print("basis_dot_const_vector")
    vec = np.ravel(np.ascontiguousarray(const_vec))
    if basis.ndim == 1:
        if basis.shape[0] != vec.shape[0]:
            raise ValueError(
                f"basis_dot_const_vector: incompatible shapes {basis.shape} and {const_vec.shape}"
            )
        res = np.empty((1, 1), dtype=dtype)
        res[0, 0] = basis @ vec
        return res
    if basis.ndim == 2:
        if basis.shape[0] == vec.shape[0]:
            res = np.empty((1, basis.shape[1]), dtype=dtype)
            basis_t = np.ascontiguousarray(basis.T)
            res[0] = basis_t @ vec
            return res
        if basis.shape[1] == vec.shape[0]:
            res = np.empty((1, basis.shape[0]), dtype=dtype)
            basis_c = np.ascontiguousarray(basis)
            res[0] = basis_c @ vec
            return res
    if basis.ndim == 3:
        basis_flat = np.ascontiguousarray(basis.transpose(0, 2, 1)).reshape(
            basis.shape[0] * basis.shape[2],
            basis.shape[1],
        )
        if basis_flat.shape[0] == vec.shape[0]:
            res = np.empty((1, basis.shape[1]), dtype=dtype)
            res[0] = basis_flat.T @ vec
            return res
    raise ValueError(
        f"basis_dot_const_vector: incompatible shapes {basis.shape} and {const_vec.shape}"
    )


@numba.njit(cache=True)
def const_vector_dot_basis(const_vec, basis, dtype):
    """
    Constant vector dotted with rank-1 basis carrier -> basis row.
    """
    if DEBUG: print("const_vector_dot_basis")
    return basis_dot_const_vector(basis, const_vec, dtype)


@numba.njit(cache=True)
def const_vector_dot_basis_1d(const_vec, basis, dtype):
    """
    Constant vector dotted with rank-1 basis carrier -> 1D basis vector.
    """
    if DEBUG: print("const_vector_dot_basis_1d")
    vec = np.ravel(np.ascontiguousarray(const_vec))
    if basis.ndim == 1:
        if basis.shape[0] == vec.shape[0]:
            return np.asarray([basis @ vec], dtype=dtype)
    if basis.ndim == 2:
        if basis.shape[0] == vec.shape[0]:
            basis_t = np.ascontiguousarray(basis.T)
            return basis_t @ vec
        if basis.shape[1] == vec.shape[0]:
            basis_c = np.ascontiguousarray(basis)
            return basis_c @ vec
    if basis.ndim == 3:
        basis_flat = np.ascontiguousarray(basis.transpose(0, 2, 1)).reshape(
            basis.shape[0] * basis.shape[2],
            basis.shape[1],
        )
        if basis_flat.shape[0] == vec.shape[0]:
            return basis_flat.T @ vec
    raise ValueError(
        f"const_vector_dot_basis_1d: incompatible shapes {const_vec.shape} and {basis.shape}"
    )



@numba.njit(cache=True)
def scalar_basis_times_vector(scalar_basis, vector_vals, dtype):
    """
    Scalar basis (1,n) or (n,) times vector components (k,) -> (k,n).
    Broadcasting outer product.
    """
    if DEBUG: print("scalar_basis_times_vector")
    if scalar_basis.ndim == 2:
        phi = scalar_basis[0]
    else:
        phi = scalar_basis
    return np.ascontiguousarray(vector_vals)[:, None] * np.ascontiguousarray(phi)[None, :]


@numba.njit(cache=True)
def scalar_basis_times_vector_as_grad_tensor(scalar_basis, vector_vals, dtype):
    """
    Scalar basis (1,n) or (n,) times a spatial vector (d,) -> (d,n).

    This keeps scalar-gradient promotions in the same carried layout as
    grad(trial_scalar): a rank-1 spatial object over basis columns.
    """
    if DEBUG:
        print("scalar_basis_times_vector_as_grad_tensor")
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
    if DEBUG: print("matrix_times_scalar_basis")
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
    if DEBUG: print("scalar_vector_outer_product")
    return np.ascontiguousarray(vector_vals)[:, None] * np.ascontiguousarray(scalar_vals)[None, :]

@numba.njit(cache=True)
def vector_trial_times_scalar_test(trial_vec, test_scalar, dtype):
    """
    Vector Trial basis (k,n_trial) times scalar Test basis (1,n_test) or (n_test,)
    -> mixed carried tensor (k, n_test, n_trial) in canonical test/trial order.
    """
    if DEBUG: print("vector_trial_times_scalar_test")
    phi_test = test_scalar[0] if test_scalar.ndim == 2 else test_scalar
    return (
        np.ascontiguousarray(trial_vec)[:, None, :]
        * np.ascontiguousarray(phi_test)[None, :, None]
    )

@numba.njit(cache=True)
def vector_test_times_scalar_trial(test_vec, trial_scalar, dtype):
    """
    Vector Test basis (k,n_test) times scalar Trial basis (1,n_trial) or (n_trial,)
    -> mixed carried tensor (k, n_test, n_trial) in canonical test/trial order.
    """
    if DEBUG: print("vector_test_times_scalar_trial")
    phi_trial = trial_scalar[0] if trial_scalar.ndim == 2 else trial_scalar
    return (
        np.ascontiguousarray(test_vec)[:, :, None]
        * np.ascontiguousarray(phi_trial)[None, None, :]
    )

@numba.njit(cache=True)
def vector_vector_outer_product(vec_a, vec_b, dtype):
    """
    Outer product between two vectors:
      (k_a,) ⊗ (k_b,) -> (k_a, k_b)
    """
    if DEBUG: print("vector_vector_outer_product")
    return np.ascontiguousarray(vec_a)[:, None] * np.ascontiguousarray(vec_b)[None, :]




@numba.njit(cache=True)
def scalar_trial_times_grad_test(grad_test, trial_vals, dtype):
    """
    Scalar Trial (n_trial,) times grad(Test) (k,n_test,d) -> (k,n_test,n_trial,d).
    Vectorized with broadcasting.
    """
    if DEBUG: print("scalar_trial_times_grad_test")
    return (np.ascontiguousarray(grad_test)[:, :, None, :] *
            np.ascontiguousarray(trial_vals)[None, None, :, None])


@numba.njit(cache=True)
def scalar_trial_times_basis_test(basis_test, trial_vals, dtype):
    """
    Scalar Trial (n_trial,) times rank-2 Test basis (r0,n_test,r1)
    -> mixed rank-2 tensor (r0,n_test,n_trial,r1).
    """
    if DEBUG: print("scalar_trial_times_basis_test")
    B = np.ascontiguousarray(basis_test)
    phi_trial = trial_vals[0] if trial_vals.ndim == 2 else trial_vals
    basis_view = B.reshape((B.shape[0], B.shape[1], 1) + B.shape[2:])
    trial_view = np.ascontiguousarray(phi_trial).reshape(
        (1, 1, phi_trial.shape[0]) + (1,) * (B.ndim - 2)
    )
    return basis_view * trial_view



@numba.njit(cache=True)
def grad_trial_times_scalar_test(grad_trial, test_vals, dtype):
    """
    Grad(Trial) (k,n_trial,d) times scalar Test (n_test,) -> (k,n_test,n_trial,d).
    Vectorized with broadcasting.
    """
    if DEBUG: print("grad_trial_times_scalar_test")
    return (np.ascontiguousarray(grad_trial)[:, None, :, :] *
            np.ascontiguousarray(test_vals)[None, :, None, None])


@numba.njit(cache=True)
def basis_trial_times_scalar_test(basis_trial, test_vals, dtype):
    """
    Rank-2 Trial basis (r0,n_trial,r1) times scalar Test (n_test,)
    -> mixed rank-2 tensor (r0,n_test,n_trial,r1).
    """
    if DEBUG: print("basis_trial_times_scalar_test")
    B = np.ascontiguousarray(basis_trial)
    phi_test = test_vals[0] if test_vals.ndim == 2 else test_vals
    basis_view = B.reshape((B.shape[0], 1, B.shape[1]) + B.shape[2:])
    test_view = np.ascontiguousarray(phi_test).reshape(
        (1, phi_test.shape[0], 1) + (1,) * (B.ndim - 2)
    )
    return test_view * basis_view



@numba.njit(cache=True)
def scale_mixed_basis_with_coeffs(mixed_basis, coeffs, dtype):
    """
    Mixed basis (k_mixed, n_rows, n_cols) scaled by coeffs (k_out, d_cols)
    -> (k_out, n_rows, n_cols, d_cols).
    Since coeffs are independent of k_mixed, this is:
       coeffs[:,None,None,:] * sum_k mixed_basis[k]
    """
    if DEBUG: print("scale_mixed_basis_with_coeffs")
    base = np.sum(np.ascontiguousarray(mixed_basis), axis=0)  # (n_rows, n_cols)
    return (np.ascontiguousarray(coeffs)[:, None, None, :] *
            base[None, :, :, None])


@numba.njit(cache=True)
def trace_times_identity(trace_vals, identity, dtype):
    """
    Trace data (flattened) times identity matrix (k,d) -> (k, n_rows, d).
    """
    if DEBUG: print("trace_times_identity")
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
    if DEBUG: print("identity_times_trace_matrix")
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
    Column-wise dot products between two basis/value arrays -> (1,n).
    Accepts inputs shaped (k,n), (k,), or (k,1). If one operand has a single
    column, it is broadcast to the other's column count.
    """
    if DEBUG: print("columnwise_dot")
    A = _ensure_matrix(a_mat, dtype)
    B = _ensure_matrix(b_mat, dtype)
    if A.shape[0] != B.shape[0]:
        raise ValueError("columnwise_dot: incompatible row counts")
    n_cols = A.shape[1] if A.shape[1] > B.shape[1] else B.shape[1]
    if A.shape[1] == 1 and n_cols > 1:
        A_b = np.empty((A.shape[0], n_cols), dtype=dtype)
        for j in range(n_cols):
            for i in range(A.shape[0]):
                A_b[i, j] = A[i, 0]
        A = A_b
    if B.shape[1] == 1 and n_cols > 1:
        B_b = np.empty((B.shape[0], n_cols), dtype=dtype)
        for j in range(n_cols):
            for i in range(B.shape[0]):
                B_b[i, j] = B[i, 0]
        B = B_b
    res = np.empty((1, n_cols), dtype=dtype)
    res[0] = np.sum(np.ascontiguousarray(A) * np.ascontiguousarray(B), axis=0)
    return res


@numba.njit(cache=True)
def hessian_dot_vector(hessian, vec, dtype):
    """
    Hessian (basis or value) dotted with spatial vector.
    Basis: (k,n,d1,d2) -> (k,n,d1)    Value: (k,d1,d2) -> (k,d1)
    """
    if DEBUG: print("hessian_dot_vector")
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
    if DEBUG: print("vector_dot_hessian_basis")
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
    if DEBUG: print("vector_dot_hessian_value")
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
def inner_rank2_value_rank2_basis(value_rank2, basis_rank2, dtype):
    """
    Inner product between a rank-2 value tensor ``(r0, r1)`` and a rank-2
    basis carrier ``(r0, n, r1)`` -> ``(n,)``.

    This is the semantic post-dot path for contractions such as ``inner(H·c, V·c)``
    and ``inner(c·H, c·V)`` once the shared dot planner has collapsed one Hessian
    axis and the result is no longer a full Hessian.
    """
    if DEBUG: print("inner_rank2_value_rank2_basis")
    V = np.ascontiguousarray(value_rank2)
    B = np.ascontiguousarray(basis_rank2)
    if V.ndim != 2 or B.ndim != 3:
        raise ValueError("inner_rank2_value_rank2_basis: expected value (r0,r1) and basis (r0,n,r1)")
    r0, r1 = V.shape
    if B.shape[0] != r0 or B.shape[2] != r1:
        raise ValueError("inner_rank2_value_rank2_basis: incompatible rank-2 shapes")
    n = B.shape[1]
    out = np.zeros(n, dtype=dtype)
    for i in range(r0):
        out += B[i] @ V[i]
    return out


@numba.njit(cache=True)
def inner_rank2_basis_rank2_value(basis_rank2, value_rank2, dtype):
    if DEBUG: print("inner_rank2_basis_rank2_value")
    return inner_rank2_value_rank2_basis(value_rank2, basis_rank2, dtype)


@numba.njit(cache=True)
def inner_hessian_function_hessian_test(function_hess, test_hess, dtype):
    """
    Inner product between Hess(Function) (k,d,d) and Hess(Test) (k,n,d,d) -> (n,).
    Vectorized by flattening the (d*d) block and doing (n x dd) @ (dd,)
    """
    if DEBUG: print("inner_hessian_function_hessian_test")
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
    if DEBUG: print("inner_mixed_grad_const")
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
    if DEBUG: print("inner_grad_const_mixed")
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
    if DEBUG: print("inner_grad_basis_grad_const")
    if grad_basis.ndim == 2:
        if grad_const.ndim == 2:
            if grad_const.shape[0] != 1 or grad_const.shape[1] != grad_basis.shape[0]:
                raise ValueError("inner_grad_basis_grad_const: incompatible scalar gradient shapes")
            grad_basis_t = np.ascontiguousarray(grad_basis.T)
            return grad_basis_t @ np.ascontiguousarray(grad_const[0])
        if grad_const.ndim == 1:
            if grad_const.shape[0] != grad_basis.shape[0]:
                raise ValueError("inner_grad_basis_grad_const: incompatible scalar gradient shapes")
            grad_basis_t = np.ascontiguousarray(grad_basis.T)
            return grad_basis_t @ np.ascontiguousarray(grad_const)
        raise ValueError("inner_grad_basis_grad_const: incompatible scalar gradient shapes")
    k, n, d = grad_basis.shape
    # Make contiguous *after* transpose so reshape is legal for numba
    A = np.ascontiguousarray(grad_basis.transpose(1, 0, 2)).reshape(n, k * d)
    b = np.ascontiguousarray(grad_const).reshape(k * d)
    return A @ b


@numba.njit(cache=True)
def inner_grad_grad(test_var, trial_var, dtype):
    """
    Inner product of gradient bases -> (n_test, n_trial).
    """
    if DEBUG: print("inner_grad_grad")
    if test_var.ndim == 2 and trial_var.ndim == 2:
        if test_var.shape[0] != trial_var.shape[0]:
            raise ValueError("inner_grad_grad: scalar gradient bases must share spatial dimension")
        test_var_t = np.ascontiguousarray(test_var.T)
        return test_var_t @ np.ascontiguousarray(trial_var)
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
    if DEBUG: print("inner_hessian_hessian")
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
    # if DEBUG: print("_binary_add_or_sub")
    return x + sign * y





# In numba_helpers.py

@numba.njit(cache=True)
def binary_add_generic(a, b, dtype):
    """
    Elementwise addition for all standard broadcasting.
    Uses Numba's native ufunc support.
    """
    # if DEBUG: print("binary_add_generic")
    return a + b



@numba.njit(cache=True)
def binary_add_3_4(a, b, dtype):
    """
    Specialized elementwise addition for:
    (k, n, d) + (k, n, m, d) -> (k, n, m, d)
    Uses reshape + ufunc. (Very fast to compile)
    """
    # if DEBUG: print("binary_add_3_4")
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
    # if DEBUG: print("binary_add_4_3")
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
    # if DEBUG: print("binary_sub_generic")
    return a - b


@numba.njit(cache=True)
def binary_sub_3_4(a, b, dtype):
    """
    Specialized elementwise subtraction for:
    (k, n, d) - (k, n, m, d) -> (k, n, m, d)
    Uses reshape + ufunc. (Very fast to compile)
    """
    # if DEBUG: print("binary_sub_3_4")
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
    # if DEBUG: print("binary_sub_4_3")
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
    # if DEBUG: print("load_variable_qp")
    u_c = np.ascontiguousarray(u_e)
    phi_c = np.ascontiguousarray(phi_q)
    if u_c.ndim == 1:
        # (ndof,) @ (ndof,) -> scalar
        return float(np.dot(u_c, phi_c))
    # (ndof, ncomp).T @ (ndof,) -> (ncomp, ndof) @ (ndof,) -> (ncomp,)
    u_e_T = u_c.T.copy()
    return u_e_T @ phi_c


@numba.njit(cache=True)
def gradient_qp(u_e, grad_phi_q):
    """
    Evaluate ∇u_h at a qp. grad_phi_q: (ndof, dim)
    Returns: scalar -> (dim,), vector -> (dim, ncomp)
    """
    # if DEBUG: print("gradient_qp")
    grad_T = np.ascontiguousarray(grad_phi_q.T)
    u_c = np.ascontiguousarray(u_e)
    if u_c.ndim == 1:
        # (ndof, dim).T @ (ndof,) -> (dim, ndof) @ (ndof,) -> (dim,)
        return grad_T @ u_c
    # (ndof, dim).T @ (ndof, ncomp) -> (dim, ndof) @ (ndof, ncomp) -> (dim, ncomp)
    return grad_T @ u_c


@numba.njit(cache=True)
def laplacian_qp(u_e, lap_phi_q):
    """
    Evaluate Δu_h at a qp. lap_phi_q: (ndof,)
    Returns: scalar -> float, vector -> (ncomp,)
    """
    if u_e.ndim == 1:
        # (ndof,) @ (ndof,) -> scalar
        return float(np.dot(np.ascontiguousarray(u_e), np.ascontiguousarray(lap_phi_q)))
    # (ndof, ncomp).T @ (ndof,) -> (ncomp, ndof) @ (ndof,) -> (ncomp,)
    return (np.ascontiguousarray(u_e.T) @ np.ascontiguousarray(lap_phi_q)).astype(u_e.dtype)


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
    
    u_c = np.ascontiguousarray(u_e)

    if u_e.ndim == 1:
        # (ndof, dd).T @ (ndof,) -> (dd, ndof) @ (ndof,) -> (dd,)
        Hflat_T = np.ascontiguousarray(Hflat.T)
        out = (Hflat_T @ u_c).reshape(dim, dim)
        return out.astype(u_c.dtype)
    
    # (ndof, ncomp).T @ (ndof, dd) -> (ncomp, ndof) @ (ndof, dd) -> (ncomp, dd)
    ncomp = u_e.shape[1]
    ue_T = np.ascontiguousarray(u_c.T)
    out2 = (ue_T @ Hflat).reshape(ncomp, dim, dim)
    return out2.astype(u_c.dtype)


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
    "vector_trial_times_scalar_test",
    "vector_test_times_scalar_trial",
    "vector_vector_outer_product",
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
    "vector_dot_grad_value",
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
    "pushforward_d3",
    "pushforward_d4",
    "load_variable_qp",
    "gradient_qp",
    "laplacian_qp",
    "hessian_qp",
    "ghost_grad_jump_penalty_scalar",
    "ghost_grad_jump_penalty_vector",
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
