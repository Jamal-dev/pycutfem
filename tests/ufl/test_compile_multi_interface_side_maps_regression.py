import numpy as np
import pytest

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.levelset import AffineLevelSet
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.jit import _active_columns, _active_field_order, compile_multi
from pycutfem.jit.ir import strip_side_metadata
from pycutfem.jit.visitor import IRGenerator
from pycutfem.ufl.expressions import Constant, Neg, Pos, VectorTestFunction, VectorTrialFunction, dot
from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.ufl.measures import dInterface
from pycutfem.utils.meshgen import structured_quad


@pytest.mark.parametrize("backend", ("jit", "cpp"))
def test_compile_multi_interface_per_field_side_maps_are_not_double_remapped(backend, monkeypatch, tmp_path) -> None:
    """
    Regression for a fragile `compile_multi` path where element-based per-field side maps
    (pos_map_<field>/neg_map_<field>) could be constructed in full indices but then remapped
    incorrectly (or twice) during active-field compression, silently dropping DOFs.

    This reproduces the inactive-middle-fields scenario:
      u_pos_* , (unused p_pos_) , u_neg_* , (unused p_neg_) , (unused lm)
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

    ls = AffineLevelSet(a=1.0, b=0.0, c=0.0)  # x=0 cuts the single cell
    mesh.classify_elements(ls)
    mesh.classify_edges(ls)
    mesh.build_interface_segments(ls)

    cut = mesh.element_bitset("cut")
    dGamma = dInterface(defined_on=cut, level_set=ls, metadata={"q": 4, "derivs": {(0, 0)}})

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
    form = (Constant(1.0) * dot(jump_u, jump_v)) * dGamma

    kernels = compile_multi(form, dof_handler=dh, mixed_element=me, backend=backend)
    ifc_elem = [k for k in kernels if k.domain == "interface" and k.static_args.get("entity_kind") == "element"]
    assert ifc_elem, "Expected at least one element-based interface kernel."
    ker = ifc_elem[0]

    # Compute the expected packed indexing for this kernel from IR.
    ir = IRGenerator().generate(form.integrand)
    ir = strip_side_metadata(ir, on_facet=form.measure.on_facet)
    active_fields = _active_field_order(ir, me)
    active_cols = _active_columns(me, active_fields)

    full_n = int(me.n_dofs_local)
    col_map = -np.ones(full_n, dtype=np.int32)
    for new_i, old_i in enumerate(active_cols):
        col_map[int(old_i)] = int(new_i)

    def _expected_map(field: str) -> np.ndarray:
        sl = me.component_dof_slices[field]
        full_idx = np.arange(int(sl.start), int(sl.stop), dtype=np.int32)
        exp = col_map[full_idx]
        assert np.all(exp >= 0), f"Field {field!r} unexpectedly inactive in active_cols."
        return exp

    static = ker.static_args
    for key, field in (
        ("pos_map_u_pos_x", "u_pos_x"),
        ("pos_map_u_pos_y", "u_pos_y"),
        ("neg_map_u_neg_x", "u_neg_x"),
        ("neg_map_u_neg_y", "u_neg_y"),
    ):
        assert key in static, f"Missing required side map '{key}'."
        arr = np.asarray(static[key], dtype=np.int32)
        assert arr.ndim == 2 and arr.shape[0] >= 1
        exp = _expected_map(field)
        assert np.array_equal(arr[0], exp), f"{key} mismatch: got {arr[0].tolist()} expected {exp.tolist()}"

