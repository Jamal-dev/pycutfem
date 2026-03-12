from pathlib import Path

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.jit.cpp_backend import compile_backend_cpp
from pycutfem.ufl.expressions import Constant, FacetNormal, TestFunction as ScalarTestFunction, dot
from pycutfem.ufl.measures import ds
from pycutfem.utils.meshgen import structured_quad


def test_cpp_codegen_does_not_pool_vectorxd_temporaries(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / "pycutfem_jit_cache"))

    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=1, ny=1, poly_order=1)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=1,
    )
    me = MixedElement(mesh, field_specs={"u": 1})
    dh = DofHandler(me, method="cg")

    v = ScalarTestFunction("u", dof_handler=dh)
    n = FacetNormal()
    form = (Constant(2.0) * dot(n, n) * v) * ds(metadata={"q": 4})

    runner, _ = compile_backend_cpp(form.integrand, dh, me, on_facet=form.measure.on_facet)
    mod_name = runner.kernel.__module__
    cpp_path = Path(tmp_path / "pycutfem_jit_cache" / "cpp" / f"{mod_name}.cpp")
    src = cpp_path.read_text(errors="replace")

    assert "Eigen::VectorXd _v[" not in src
