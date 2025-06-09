import numpy as np
from pycutfem.fem.reference import get_reference

def test_q2_partition_of_unity():
    ref = get_reference("quad", 2)  # Q2
    rng = np.random.default_rng(0)
    for _ in range(20):
        xi, eta = rng.uniform(-1, 1, size=2)
        N = ref.shape(xi, eta)
        assert np.isclose(N.sum(), 1.0, atol=1e-12)
        G = ref.grad(xi, eta)
        assert np.allclose(G.sum(axis=0), 0.0, atol=1e-12)
