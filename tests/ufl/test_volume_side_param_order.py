import pytest

from pycutfem.utils.meshgen import structured_quad
from pycutfem.core.mesh import Mesh
from pycutfem.core.levelset import AffineLevelSet
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.core.dofhandler import DofHandler
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.ufl.expressions import VectorTrialFunction, VectorTestFunction, inner, grad
from pycutfem.ufl.measures import dx, dGhost
from pycutfem.jit import compile_multi


def _build_mesh():
    nodes, elements, edges, corners = structured_quad(
        Lx=1.0,
        Ly=1.0,
        nx=6,
        ny=4,
        poly_order=1,
    )
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elements,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=1,
    )
    return mesh


def _build_space(dh):
    space = FunctionSpace(name="u_pos", field_names=["u_pos_x", "u_pos_y"], dim=1, side="+")
    u = VectorTrialFunction(space, dof_handler=dh)
    v = VectorTestFunction(space, dof_handler=dh)
    return u, v


def _volume_kernels(dh, me, backend):
    u, v = _build_space(dh)
    form = inner(grad(u), grad(v)) * dx
    kernels = compile_multi(form, dof_handler=dh, mixed_element=me, backend=backend)
    return [ker for ker in kernels if ker.domain == "volume"]


def _ghost_kernels(dh, me, mesh, level_set, backend):
    u, v = _build_space(dh)
    ghost = mesh.edge_bitset("ghost")
    form = inner(grad(u), grad(v)) * dGhost(
        defined_on=ghost,
        level_set=level_set,
        metadata={"q": 2},
    )
    kernels = compile_multi(form, dof_handler=dh, mixed_element=me, backend=backend)
    return [ker for ker in kernels if ker.domain == "ghost_edge"]


@pytest.mark.parametrize("backend", ("jit", "cpp"))
def test_volume_side_does_not_require_sided_statics(monkeypatch, backend):
    monkeypatch.setenv("PYCUTFEM_JIT_BACKEND", "")
    mesh = _build_mesh()
    me = MixedElement(mesh, field_specs={"u_pos_x": 1, "u_pos_y": 1})
    dh = DofHandler(me, method="cg")

    kernels = _volume_kernels(dh, me, backend)
    assert kernels, "Expected at least one volume kernel for the test form."

    for ker in kernels:
        param_order = ker.runner.param_order
        assert "J_inv_pos" not in param_order
        assert "J_inv_neg" not in param_order
        assert not any(p.endswith("__pos_loc") or p.endswith("__neg_loc") for p in param_order)


@pytest.mark.parametrize("backend", ("jit", "cpp"))
def test_ghost_side_requires_sided_statics(monkeypatch, backend):
    monkeypatch.setenv("PYCUTFEM_JIT_BACKEND", "")
    mesh = _build_mesh()
    level_set = AffineLevelSet(a=1.0, b=0.0, c=-0.45)
    mesh.classify_elements(level_set)
    mesh.classify_edges(level_set)
    ghost = mesh.edge_bitset("ghost")
    assert ghost.cardinality() > 0, "Expected ghost edges for the sided-ghost test."

    me = MixedElement(mesh, field_specs={"u_pos_x": 1, "u_pos_y": 1})
    dh = DofHandler(me, method="cg")

    kernels = _ghost_kernels(dh, me, mesh, level_set, backend)
    assert kernels, "Expected at least one ghost-edge kernel for the test form."

    for ker in kernels:
        param_order = ker.runner.param_order
        assert "J_inv_pos" in param_order
