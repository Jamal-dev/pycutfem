import numpy as np
import scipy.sparse.linalg as spla

from pycutfem.assembly.dg_global import assemble_dg
from pycutfem.core.mesh import Mesh
from pycutfem.fem import transform
from pycutfem.fem.reference import get_reference
from pycutfem.integration import volume
from pycutfem.utils.meshgen import structured_quad


def _u_exact(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)


def _rhs(x, y):
    return 2.0 * (np.pi ** 2) * _u_exact(x, y)


def _compute_l2_error(mesh, uh, u_exact_func, *, n_comp=1):
    total_error_sq = 0.0
    total_area = 0.0

    ref = get_reference(mesh.element_type, mesh.poly_order)
    pts, wts = volume(mesh.element_type, 2 * mesh.poly_order + 2)
    n_loc = len(ref.shape(0, 0))
    n_eldof = n_loc * n_comp

    for eid, elem in enumerate(mesh.elements_list):
        dofs = np.arange(n_eldof) + eid * n_eldof
        uh_element = uh[dofs]

        elem_error_sq = 0.0
        for (xi, eta), w in zip(pts, wts):
            N = ref.shape(xi, eta)
            J = transform.jacobian(mesh, eid, (xi, eta))
            detJ = abs(np.linalg.det(J))
            x_phys = transform.x_mapping(mesh, eid, (xi, eta))

            u_exact_at_pt = np.array(u_exact_func(*x_phys)).flatten()
            uh_at_pt = np.zeros(n_comp)
            for c in range(n_comp):
                uh_local_comp = uh_element[c * n_loc:(c + 1) * n_loc]
                uh_at_pt[c] = N @ uh_local_comp

            error_vec = uh_at_pt - u_exact_at_pt
            elem_error_sq += w * detJ * (error_vec @ error_vec)

        total_error_sq += elem_error_sq
        total_area += mesh.areas()[eid]

    return np.sqrt(total_error_sq / total_area)


def solve_poisson_dg(*, symmetry=1, nx=8, ny=8, poly_order=1, alpha=None):
    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=nx, ny=ny, poly_order=poly_order)
    mesh = Mesh(
        nodes,
        element_connectivity=elems,
        edges_connectivity=None,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=poly_order,
    )
    if alpha is None:
        alpha = 10.0 * (poly_order + 1) ** 2

    K, F = assemble_dg(
        mesh,
        alpha=float(alpha),
        symmetry=int(symmetry),
        dirichlet=_u_exact,
        rhs=_rhs,
    )
    uh = spla.spsolve(K, F)
    return _compute_l2_error(mesh, uh, _u_exact, n_comp=1)


def main():
    fluxes = {"sipg": 1, "iipg": 0, "nipg": -1}
    for name, sym in fluxes.items():
        err = solve_poisson_dg(symmetry=sym, nx=8, ny=8, poly_order=1)
        print(f"{name.upper()}: L2 error = {err:.3e}")


if __name__ == "__main__":
    main()
