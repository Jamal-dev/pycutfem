import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from pycutfem.nonmatching import apply_dirichlet_increment


def test_apply_dirichlet_increment_enforces_values():
    rng = np.random.default_rng(0)
    n = 12
    A = rng.standard_normal((n, n))
    K = sp.csr_matrix(A.T @ A + 0.1 * np.eye(n))  # SPD-ish
    R = rng.standard_normal(n)
    U = rng.standard_normal(n)

    rows = [0, 3, 7]
    vals = {i: float(rng.standard_normal()) for i in rows}

    Kc, rhs = apply_dirichlet_increment(K, R, U, vals)
    dU = spla.spsolve(Kc.tocsc(), rhs)
    U_new = U + np.asarray(dU, float)

    for i in rows:
        assert abs(float(U_new[i]) - float(vals[i])) < 1.0e-10

    free = [i for i in range(n) if i not in rows]
    res_new = K @ np.asarray(dU, float) + np.asarray(R, float)
    assert float(np.linalg.norm(res_new[free], ord=np.inf)) < 1.0e-10

