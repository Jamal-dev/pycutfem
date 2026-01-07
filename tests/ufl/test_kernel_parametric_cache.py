import numpy as np
import pytest

from pycutfem.core.mesh import Mesh
from pycutfem.core.dofhandler import DofHandler
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.utils.bitset import BitSet
from pycutfem.utils.meshgen import structured_quad

from pycutfem.jit.cache import KernelCache
from pycutfem.jit.ir import LoadConstant, strip_side_metadata
from pycutfem.jit.visitor import IRGenerator

from pycutfem.ufl.analytic import Analytic, x as x_ana
from pycutfem.ufl.expressions import (
    Constant,
    Derivative,
    ElementWiseConstant,
    FacetNormal,
    Jump,
    TestFunction,
    TrialFunction,
    dot,
    grad,
    inner,
    restrict,
)


def _make_dh(*, nx: int, ny: int, poly_order: int = 1):
    nodes, elems, edges, corners = structured_quad(
        1.0, 1.0, nx=nx, ny=ny, poly_order=poly_order
    )
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=poly_order,
    )
    me = MixedElement(mesh, field_specs={"u": poly_order})
    dh = DofHandler(me, method="cg")
    return dh, me


def _kernel_hash(expr, *, mixed_element, on_facet: bool) -> str:
    ir = IRGenerator().generate(expr)
    ir = strip_side_metadata(ir, on_facet=on_facet)
    assert not any(isinstance(op, LoadConstant) for op in ir)
    return KernelCache._hash_ir(ir, mixed_element.signature())


def _base_integrand(dh: DofHandler, kind: str):
    if kind == "volume":
        u = TrialFunction(field_name="u", name="u", dof_handler=dh)
        v = TestFunction(field_name="u", name="v", dof_handler=dh)
        return inner(grad(u), grad(v)), False

    if kind in {"interface", "interior_edge"}:
        u_pos = TrialFunction(field_name="u", name="u", dof_handler=dh, side="+")
        u_neg = TrialFunction(field_name="u", name="u", dof_handler=dh, side="-")
        v_pos = TestFunction(field_name="u", name="v", dof_handler=dh, side="+")
        v_neg = TestFunction(field_name="u", name="v", dof_handler=dh, side="-")
        return inner(Jump(u_pos, u_neg), Jump(v_pos, v_neg)), True

    if kind == "ghost":
        u_pos = TrialFunction(field_name="u", name="u", dof_handler=dh, side="+")
        u_neg = TrialFunction(field_name="u", name="u", dof_handler=dh, side="-")
        v_pos = TestFunction(field_name="u", name="v", dof_handler=dh, side="+")
        v_neg = TestFunction(field_name="u", name="v", dof_handler=dh, side="-")
        jump_u = Jump(u_pos, u_neg)
        jump_v = Jump(v_pos, v_neg)
        return Derivative(jump_u, 1, 0) * Derivative(jump_v, 1, 0), True

    if kind == "edge":
        u = TrialFunction(field_name="u", name="u", dof_handler=dh)
        v = TestFunction(field_name="u", name="v", dof_handler=dh)
        n = FacetNormal()
        return dot(grad(u), n) * v, True

    raise ValueError(f"Unknown kernel kind: {kind!r}")


@pytest.mark.parametrize(
    "kind",
    [
        "volume",
        "interface",
        "ghost",
        "edge",
        "interior_edge",
    ],
)
def test_kernel_hash_ignores_scalar_multipliers(kind):
    dh, me = _make_dh(nx=2, ny=2, poly_order=1)
    base, on_facet = _base_integrand(dh, kind)
    h1 = _kernel_hash(1.5 * base, mixed_element=me, on_facet=on_facet)
    h2 = _kernel_hash(2.0 * base, mixed_element=me, on_facet=on_facet)
    assert h1 == h2


@pytest.mark.parametrize("kind", ["volume", "interface", "ghost", "edge", "interior_edge"])
def test_constant_value_change_does_not_change_kernel_hash(kind):
    dh, me = _make_dh(nx=2, ny=2, poly_order=1)
    base, on_facet = _base_integrand(dh, kind)
    alpha = Constant(1.5)
    h1 = _kernel_hash(alpha * base, mixed_element=me, on_facet=on_facet)
    alpha.value = 2.0
    h2 = _kernel_hash(alpha * base, mixed_element=me, on_facet=on_facet)
    assert h1 == h2


@pytest.mark.parametrize("kind", ["volume", "interface", "ghost", "edge", "interior_edge"])
def test_kernel_hash_mesh_independent_resolution(kind):
    dh1, me1 = _make_dh(nx=2, ny=2, poly_order=1)
    dh2, me2 = _make_dh(nx=4, ny=4, poly_order=1)
    base1, on_facet = _base_integrand(dh1, kind)
    base2, _ = _base_integrand(dh2, kind)
    h1 = _kernel_hash(1.5 * base1, mixed_element=me1, on_facet=on_facet)
    h2 = _kernel_hash(1.5 * base2, mixed_element=me2, on_facet=on_facet)
    assert h1 == h2


def test_kernel_hash_ignores_domain_mask_values():
    dh, me = _make_dh(nx=2, ny=2, poly_order=1)
    u = TrialFunction(field_name="u", name="u", dof_handler=dh)
    v = TestFunction(field_name="u", name="v", dof_handler=dh)

    n_elems = int(me.mesh.n_elements)
    bs_all = BitSet(np.ones(n_elems, dtype=bool))
    bs_alt = BitSet((np.arange(n_elems) % 2) == 0)

    base1 = inner(grad(restrict(u, bs_all)), grad(v))
    base2 = inner(grad(restrict(u, bs_alt)), grad(v))
    h1 = _kernel_hash(base1, mixed_element=me, on_facet=False)
    h2 = _kernel_hash(base2, mixed_element=me, on_facet=False)
    assert h1 == h2


def test_kernel_hash_ignores_elementwise_constant_values():
    dh, me = _make_dh(nx=2, ny=2, poly_order=1)
    u = TrialFunction(field_name="u", name="u", dof_handler=dh)
    v = TestFunction(field_name="u", name="v", dof_handler=dh)
    base = inner(grad(u), grad(v))

    n_elems = int(me.mesh.n_elements)
    ewc1 = ElementWiseConstant(np.ones(n_elems, dtype=float))
    ewc2 = ElementWiseConstant(np.linspace(0.0, 1.0, n_elems))

    h1 = _kernel_hash(ewc1 * base, mixed_element=me, on_facet=False)
    h2 = _kernel_hash(ewc2 * base, mixed_element=me, on_facet=False)
    assert h1 == h2


def test_kernel_hash_ignores_analytic_identity():
    dh, me = _make_dh(nx=2, ny=2, poly_order=1)
    v = TestFunction(field_name="u", name="v", dof_handler=dh)
    a1 = Analytic(x_ana)
    a2 = Analytic(x_ana)

    h1 = _kernel_hash(a1 * v, mixed_element=me, on_facet=False)
    h2 = _kernel_hash(a2 * v, mixed_element=me, on_facet=False)
    assert h1 == h2
