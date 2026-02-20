import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.jit import KernelRunner
from pycutfem.ufl.expressions import VectorFunction
from pycutfem.utils.meshgen import structured_quad


def _tag_unit_square_boundaries(mesh: Mesh, *, tol: float = 1.0e-12) -> None:
    mesh.tag_boundary_edges(
        {
            "left": lambda x, y: abs(x - 0.0) <= tol,
            "right": lambda x, y: abs(x - 1.0) <= tol,
            "bottom": lambda x, y: abs(y - 0.0) <= tol,
            "top": lambda x, y: abs(y - 1.0) <= tol,
        }
    )


def test_kernelrunner_resolves_vector_component_coeff_names():
    """
    Regression test: some kernels may request vector coefficient components by
    their full component symbol name (e.g. "vS_n_vS_x") even when the solver only
    provides the parent VectorFunction under its base name ("vS_n").
    """

    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=1, ny=1, poly_order=2)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=2,
    )
    _tag_unit_square_boundaries(mesh)

    me = MixedElement(
        mesh,
        field_specs={
            "vS_x": 2,
            "vS_y": 2,
        },
    )
    dh = DofHandler(me, method="cg")

    vS_n = VectorFunction("vS_n", ["vS_x", "vS_y"], dof_handler=dh)
    vS_n.nodal_values[:] = np.arange(vS_n.nodal_values.size, dtype=float) + 1.0

    def _kernel(u_vS_n_vS_x_loc, u_vS_n_vS_y_loc):
        return u_vS_n_vS_x_loc, u_vS_n_vS_y_loc

    param_order = ["u_vS_n_vS_x_loc", "u_vS_n_vS_y_loc"]
    runner = KernelRunner(_kernel, param_order, ir_sequence=[], dof_handler=dh)
    got_x, got_y = runner(functions={"vS_n": vS_n}, static_args={})

    total_dofs = int(dh.total_dofs)
    full_vec = np.zeros(total_dofs, dtype=float)
    full_vec[vS_n._g_dofs] = vS_n.nodal_values

    gdofs_map = np.vstack([dh.get_elemental_dofs(eid) for eid in range(mesh.n_elements)]).astype(np.int32)
    sl_x = dh.get_field_slice("vS_x")
    sl_y = dh.get_field_slice("vS_y")

    comp_x = np.zeros_like(full_vec)
    comp_x[sl_x] = full_vec[sl_x]
    comp_y = np.zeros_like(full_vec)
    comp_y[sl_y] = full_vec[sl_y]

    exp_x = comp_x[gdofs_map]
    exp_y = comp_y[gdofs_map]

    assert np.array_equal(got_x, exp_x)
    assert np.array_equal(got_y, exp_y)

