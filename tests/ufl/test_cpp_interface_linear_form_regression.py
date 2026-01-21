import numpy as np
import pytest

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.levelset import AffineLevelSet
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.expressions import FacetNormal, Constant, Function, VectorTestFunction, VectorTrialFunction, dot, Neg, Pos
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.ufl.measures import dInterface
from pycutfem.utils.fsi_fully_eulerian import build_measures, make_domain_sets
from pycutfem.utils.meshgen import structured_quad


def test_cpp_interface_linear_form_p_times_dotn_v_not_zero(monkeypatch, tmp_path) -> None:
    """
    Regression for a C++ backend bug where certain dInterface linear forms were
    assembled as identically zero due to an active-field ordering mismatch.

    Minimal failing form (on Γ):  ∫ p * (n · v) dS
    """
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / "pycutfem_jit_cache"))

    poly_order = 1
    nodes, elems, edges, corners = structured_quad(
        2.0,
        2.0,
        nx=1,
        ny=1,
        poly_order=poly_order,
        offset=(-1.0, -1.0),
    )
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=poly_order,
    )

    ls = AffineLevelSet(a=1.0, b=0.0, c=0.0)  # interface x=0
    mesh.classify_elements(ls)
    mesh.classify_edges(ls)
    mesh.build_interface_segments(ls)

    domains = make_domain_sets(mesh, use_aligned_interface=False)
    _, _, dGamma, _, _ = build_measures(mesh, ls, domains, qvol=3)

    me = MixedElement(mesh, field_specs={"u_pos_x": 1, "u_pos_y": 1, "p_pos_": 1})
    dh = DofHandler(me, method="cg")

    velocity = FunctionSpace(name="velocity", field_names=["u_pos_x", "u_pos_y"], dim=1, side="+")
    v = VectorTestFunction(space=velocity, dof_handler=dh)
    p = Function(name="p", field_name="p_pos_", dof_handler=dh, side="+")
    p.nodal_values.fill(1.234)

    n = FacetNormal()
    L = (p * dot(n, v)) * dGamma

    _, F_py = assemble_form(Equation(None, L), dof_handler=dh, bcs=[], backend="python")
    try:
        _, F_cpp = assemble_form(Equation(None, L), dof_handler=dh, bcs=[], backend="cpp")
    except Exception as exc:
        pytest.skip(f"cpp backend unavailable: {exc}")

    dofs_u_x = np.asarray(dh.get_field_slice("u_pos_x"), dtype=int)
    dofs_u_y = np.asarray(dh.get_field_slice("u_pos_y"), dtype=int)

    assert np.linalg.norm(F_cpp[dofs_u_x], ord=np.inf) > 1.0e-12
    assert np.allclose(F_cpp[dofs_u_x], F_py[dofs_u_x], rtol=1e-12, atol=1e-12)
    assert np.allclose(F_cpp[dofs_u_y], 0.0, rtol=0.0, atol=1e-12)


def test_cpp_interface_bilinear_form_jump_penalty_matches_python(monkeypatch, tmp_path) -> None:
    """
    Regression for an interface-only bilinear form mismatch where per-field side maps
    were created in already-compressed union indices and then remapped a second time
    during static-arg compression, silently dropping the (-) side DOFs.

    Minimal failing form (on Γ):  ∫ ( [u] · [v] ) dS,  with [u] = Neg(u-) - Pos(u+)
    """
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / "pycutfem_jit_cache"))

    poly_order = 1
    nodes, elems, edges, corners = structured_quad(
        2.0,
        2.0,
        nx=1,
        ny=1,
        poly_order=poly_order,
        offset=(-1.0, -1.0),
    )
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=poly_order,
    )

    ls = AffineLevelSet(a=1.0, b=0.0, c=0.0)  # interface x=0 cuts the single cell
    mesh.classify_elements(ls)
    mesh.classify_edges(ls)
    mesh.build_interface_segments(ls)

    cut = mesh.element_bitset("cut")
    dGamma = dInterface(defined_on=cut, level_set=ls, metadata={"q": 4, "derivs": {(0, 0)}})

    # Include unused middle fields so active-col compression shifts later slices.
    me = MixedElement(
        mesh,
        field_specs={
            "u_pos_x": 1,
            "u_pos_y": 1,
            "p_pos_": 1,  # unused
            "u_neg_x": 1,
            "u_neg_y": 1,
            "p_neg_": 1,  # unused
            "lm": ":number:",  # unused
        },
    )
    dh = DofHandler(me, method="cg")

    vel_pos = FunctionSpace(name="vel_pos", field_names=["u_pos_x", "u_pos_y"], dim=1, side="+")
    vel_neg = FunctionSpace(name="vel_neg", field_names=["u_neg_x", "u_neg_y"], dim=1, side="-")
    u_pos = VectorTrialFunction(space=vel_pos, dof_handler=dh, side="+")
    v_pos = VectorTestFunction(space=vel_pos, dof_handler=dh, side="+")
    u_neg = VectorTrialFunction(space=vel_neg, dof_handler=dh, side="-")
    v_neg = VectorTestFunction(space=vel_neg, dof_handler=dh, side="-")

    jump_u = Neg(u_neg) - Pos(u_pos)
    jump_v = Neg(v_neg) - Pos(v_pos)
    a = (Constant(1.0) * dot(jump_u, jump_v)) * dGamma

    K_py, _ = assemble_form(Equation(a, None), dof_handler=dh, bcs=[], backend="python")
    try:
        K_cpp, _ = assemble_form(Equation(a, None), dof_handler=dh, bcs=[], backend="cpp")
    except Exception as exc:
        pytest.skip(f"cpp backend unavailable: {exc}")

    D = (K_cpp - K_py).tocsr()
    max_abs = float(np.max(np.abs(D.data))) if D.nnz else 0.0
    assert max_abs < 1.0e-12
