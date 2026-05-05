import numpy as np
import pytest


def _python_qjac(coords: np.ndarray, xi: np.ndarray, eta: np.ndarray, *, poly_order: int):
    """Reference J, detJ, invJ via pycutfem reference gradients."""
    from pycutfem.fem.reference import get_reference

    coords = np.asarray(coords, dtype=float)
    xi = np.asarray(xi, dtype=float)
    eta = np.asarray(eta, dtype=float)
    assert coords.ndim == 3 and coords.shape[-1] == 2
    assert xi.shape == eta.shape and xi.ndim == 2 and xi.shape[0] == coords.shape[0]

    ref = get_reference("quad", int(poly_order))
    nE, nQ = xi.shape
    nLoc = coords.shape[1]
    dN = np.empty((nE, nQ, nLoc, 2), dtype=float)
    for e in range(nE):
        for q in range(nQ):
            dN[e, q] = np.asarray(ref.grad(float(xi[e, q]), float(eta[e, q])), dtype=float)

    J = np.einsum("eki,eqkj->eqij", coords, dN, optimize=True)
    det = J[..., 0, 0] * J[..., 1, 1] - J[..., 0, 1] * J[..., 1, 0]
    inv = np.empty_like(J)
    inv[..., 0, 0] = J[..., 1, 1]
    inv[..., 0, 1] = -J[..., 0, 1]
    inv[..., 1, 0] = -J[..., 1, 0]
    inv[..., 1, 1] = J[..., 0, 0]
    inv /= det[..., None, None]
    return J, det, inv


@pytest.mark.parametrize("poly_order,nloc", [(1, 4), (2, 9)])
def test_quad_jacobian_det_inv_matches_python(poly_order, nloc, tmp_path, monkeypatch):
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / "pycutfem_jit_cache"))

    # Deterministic, non-degenerate affine mapping.
    rng = np.random.default_rng(1)
    A = rng.normal(size=(2, 2))
    if np.linalg.det(A) < 0.0:
        A[:, 0] *= -1.0
    b = rng.normal(size=(2,))

    if poly_order == 1:
        xi_nodes = np.array([-1.0, 1.0, -1.0, 1.0])
        eta_nodes = np.array([-1.0, -1.0, 1.0, 1.0])
    else:
        grid = np.array([-1.0, 0.0, 1.0])
        xi_nodes = np.tile(grid, 3)
        eta_nodes = np.repeat(grid, 3)

    # Simple reference-to-physical embedding.
    X = np.stack(((xi_nodes + 1.0) * 0.5, (eta_nodes + 1.0) * 0.5), axis=1)  # (nloc,2)
    X = (X @ A.T) + b[None, :]
    coords = X.reshape(1, nloc, 2)

    xi = rng.uniform(-1.0, 1.0, size=(1, 7))
    eta = rng.uniform(-1.0, 1.0, size=(1, 7))

    from pycutfem.jit.cpp_backend.precompute_geom import quad_jacobian_det_inv

    J_cpp, det_cpp, inv_cpp = quad_jacobian_det_inv(coords, xi, eta, poly_order)
    J_py, det_py, inv_py = _python_qjac(coords, xi, eta, poly_order=poly_order)

    assert np.allclose(J_cpp, J_py, rtol=1e-12, atol=1e-12)
    assert np.allclose(det_cpp, det_py, rtol=1e-12, atol=1e-12)
    assert np.allclose(inv_cpp, inv_py, rtol=1e-12, atol=1e-12)


def test_precompute_ghost_factors_cpp_backend_parity(tmp_path, monkeypatch):
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / "pycutfem_jit_cache"))

    from pycutfem.core.mesh import Mesh
    from pycutfem.core.levelset import AffineLevelSet
    from pycutfem.core.dofhandler import DofHandler, clear_caches
    from pycutfem.fem.mixedelement import MixedElement
    from pycutfem.utils.meshgen import structured_quad

    poly_order = 2
    nodes, elements_connectivity, edge_connectivity, corner_nodes = structured_quad(
        2.0, 1.0, nx=16, ny=4, poly_order=poly_order
    )
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elements_connectivity,
        edges_connectivity=edge_connectivity,
        elements_corner_nodes=corner_nodes,
        element_type="quad",
        poly_order=poly_order,
    )
    level_set = AffineLevelSet(a=1.0, b=0.0, c=-1.03)
    mesh.classify_elements(level_set)
    mesh.build_interface_segments(level_set)
    mesh.classify_edges(level_set)
    ghost = mesh.edge_bitset("ghost")
    assert ghost.cardinality() > 0

    dh = DofHandler(MixedElement(mesh, field_specs={"u": poly_order}), method="cg")
    ghost_ids = np.asarray(ghost.to_indices(), dtype=np.int32)
    derivs = {(1, 0), (0, 1), (2, 0), (1, 1), (0, 2)}

    # Python/NumPy geometry
    clear_caches()
    monkeypatch.delenv("PYCUTFEM_PRECOMPUTE_GEOM_BACKEND", raising=False)
    ref = dh.precompute_ghost_factors(ghost_ids, 6, level_set, derivs, reuse=False)

    # C++ geometry
    clear_caches()
    monkeypatch.setenv("PYCUTFEM_PRECOMPUTE_GEOM_BACKEND", "cpp")
    cpp = dh.precompute_ghost_factors(ghost_ids, 6, level_set, derivs, reuse=False)

    for key in ("detJ_pos", "detJ_neg", "J_inv_pos", "J_inv_neg"):
        assert np.allclose(cpp[key], ref[key], rtol=1e-11, atol=1e-12), key
