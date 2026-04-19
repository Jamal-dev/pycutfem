import numpy as np

from pycutfem.utils.meshgen import structured_quad
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.core.dofhandler import DofHandler
from pycutfem.ufl.expressions import Function, VectorFunction
from pycutfem.io.vtk import export_vtk


def test_export_vtk_skips_missing_dof_to_node_entries(tmp_path):
    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=1, ny=1, poly_order=1)
    mesh = Mesh(nodes, elems, elements_corner_nodes=corners, element_type="quad", poly_order=1)
    me = MixedElement(mesh, field_specs={"ux": 1, "uy": 1, "p": 1})
    dh = DofHandler(me, method="cg")

    u = VectorFunction("u", ["ux", "uy"], dof_handler=dh)
    p = Function("p", "p", dof_handler=dh)

    u.nodal_values[:] = np.arange(len(u.nodal_values), dtype=float)
    p.nodal_values[:] = np.linspace(0.0, 1.0, len(p.nodal_values))

    # Simulate a sparse/incomplete node map such as the accepted-step
    # diagnostic export path encountered on the quasi-static Seboldt branch.
    missing_scalar_gdof = int(p._g_dofs[0])
    missing_vector_gdof = int(u._g_dofs[0])
    dh._dof_to_node_map.pop(missing_scalar_gdof, None)
    dh._dof_to_node_map.pop(missing_vector_gdof, None)

    out = tmp_path / "sparse_map.vtu"
    export_vtk(
        str(out),
        mesh=mesh,
        dof_handler=dh,
        functions={"u": u, "p": p},
    )

    assert out.exists()
