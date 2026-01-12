import numpy as np
import pytest


def test_precompute_unified_matches_separate_turek_fsi2(tmp_path, monkeypatch):
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / "pycutfem_jit_cache"))

    from pycutfem.core.dofhandler import clear_caches
    from pycutfem.utils.turek_fsi2 import build_turek_fsi2_setup

    setup = build_turek_fsi2_setup(mesh_size=0.025, poly_order=2)
    dh = setup.dof_handler
    mesh = setup.mesh
    ls = setup.level_set

    qvol = 6
    qifc = 8
    qghost = 6
    qfacet_patch = 6
    derivs_cut = {(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)}
    derivs_ghost = {(1, 0), (0, 1), (2, 0), (1, 1), (0, 2)}
    derivs_facet_patch = {(1, 0), (0, 1)}

    cut_ids = np.asarray(mesh.element_bitset("cut").to_indices(), dtype=np.int32)
    ghost_only_ids = np.asarray(mesh.edge_bitset("ghost").to_indices(), dtype=np.int32)
    ifc_edge_ids = np.asarray(mesh.edge_bitset("interface").to_indices(), dtype=np.int32)
    ghost_ids = np.asarray((mesh.edge_bitset("ghost") | mesh.edge_bitset("interface")).to_indices(), dtype=np.int32)

    clear_caches()
    sep_cut_plus = dh.precompute_cut_volume_factors(cut_ids, qvol, derivs_cut, ls, side="+", reuse=False)
    sep_cut_minus = dh.precompute_cut_volume_factors(cut_ids, qvol, derivs_cut, ls, side="-", reuse=False)
    sep_ifc = dh.precompute_interface_factors(cut_ids, qifc, ls, reuse=False)
    sep_ghost = dh.precompute_ghost_factors(ghost_only_ids, qghost, ls, derivs_ghost, allow_interface=False, reuse=False)
    sep_facet_patch = dh.precompute_facet_patch_factors(
        facet_ids=ghost_only_ids,
        qdeg=qfacet_patch,
        level_set=ls,
        derivs=derivs_facet_patch,
        allow_interface=False,
        reuse=False,
    )
    sep_aligned = None
    if ifc_edge_ids.size:
        sep_aligned = dh.precompute_ghost_factors(ifc_edge_ids, qifc, ls, derivs_ghost, allow_interface=True, reuse=False)

    clear_caches()
    uni = dh.precompute_unified_factors(
        level_set=ls,
        qvol=qvol,
        qifc=qifc,
        qghost=qghost,
        qfacet_patch=qfacet_patch,
        cut_element_ids=cut_ids,
        interface_element_ids=cut_ids,
        ghost_edge_ids=ghost_ids,
        facet_patch_edge_ids=ghost_only_ids,
        derivs_cut=derivs_cut,
        derivs_ghost=derivs_ghost,
        derivs_facet_patch=derivs_facet_patch,
        allow_interface=True,
        include_volume=False,
        reuse=False,
    )

    assert set(uni.keys()) >= {"cut_plus", "cut_minus", "interface", "ghost", "facet_patch"}

    for key in ("detJ", "J_inv", "qw", "phis"):
        assert np.allclose(uni["cut_plus"][key], sep_cut_plus[key], rtol=1e-11, atol=1e-12), key
        assert np.allclose(uni["cut_minus"][key], sep_cut_minus[key], rtol=1e-11, atol=1e-12), key

    for key in ("qw", "phis", "normals"):
        assert np.allclose(uni["interface"][key], sep_ifc[key], rtol=1e-11, atol=1e-12), key

    # Ghost edges are split internally; in this setup interface edges can be empty.
    assert np.allclose(uni["ghost"]["detJ"], sep_ghost["detJ"], rtol=1e-11, atol=1e-12)
    assert np.allclose(uni["ghost"]["J_inv"], sep_ghost["J_inv"], rtol=1e-11, atol=1e-12)
    if "aligned_interface" in uni:
        assert sep_aligned is not None
        assert np.allclose(uni["aligned_interface"]["detJ"], sep_aligned["detJ"], rtol=1e-11, atol=1e-12)

    for key in ("detJ", "J_inv", "qw", "phis"):
        assert np.allclose(uni["facet_patch"][key], sep_facet_patch[key], rtol=1e-11, atol=1e-12), key


def test_precompute_unified_empty_domains(tmp_path, monkeypatch):
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / "pycutfem_jit_cache"))

    from pycutfem.core.mesh import Mesh
    from pycutfem.core.levelset import AffineLevelSet
    from pycutfem.core.dofhandler import DofHandler
    from pycutfem.fem.mixedelement import MixedElement
    from pycutfem.utils.meshgen import structured_quad

    nodes, elements, edges, corners = structured_quad(1.0, 1.0, nx=4, ny=4, poly_order=1)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elements,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=1,
    )

    level_set = AffineLevelSet(a=0.0, b=0.0, c=1.0)  # strictly positive -> no cuts
    mesh.classify_elements(level_set)
    mesh.build_interface_segments(level_set)
    mesh.classify_edges(level_set)

    dh = DofHandler(MixedElement(mesh, field_specs={"u": 1}), method="cg")

    uni = dh.precompute_unified_factors(
        level_set=level_set,
        qvol=2,
        qifc=2,
        qghost=2,
        qfacet_patch=2,
        cut_element_ids=np.zeros((0,), dtype=np.int32),
        interface_element_ids=np.zeros((0,), dtype=np.int32),
        ghost_edge_ids=np.zeros((0,), dtype=np.int32),
        facet_patch_edge_ids=np.zeros((0,), dtype=np.int32),
        derivs_cut={(0, 0)},
        derivs_ghost=set(),
        derivs_facet_patch=set(),
        allow_interface=False,
        include_volume=False,
        reuse=False,
    )
    assert uni["cut_plus"]["eids"].size == 0
    assert uni["cut_minus"]["eids"].size == 0
    assert uni["interface"]["eids"].size == 0
    assert uni["ghost"]["eids"].size == 0
    assert uni["facet_patch"]["eids"].size == 0


def test_compile_multi_uses_unified_precompute(monkeypatch, tmp_path):
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / "pycutfem_jit_cache"))
    monkeypatch.setenv("PYCUTFEM_UNIFIED_PRECOMPUTE", "1")

    from pycutfem.core.mesh import Mesh
    from pycutfem.core.levelset import AffineLevelSet
    from pycutfem.core.dofhandler import DofHandler, clear_caches
    from pycutfem.fem.mixedelement import MixedElement
    from pycutfem.utils.meshgen import structured_quad
    from pycutfem.jit import compile_multi
    from pycutfem.ufl.expressions import TrialFunction, TestFunction, Jump
    from pycutfem.ufl.measures import dx, dGhost, dFacetPatch
    from pycutfem.ufl.forms import Equation

    poly_order = 2
    nodes, elements, edges, corners = structured_quad(2.0, 1.0, nx=10, ny=3, poly_order=poly_order)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elements,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=poly_order,
    )

    ls1 = AffineLevelSet(a=1.0, b=0.0, c=-1.03)
    mesh.classify_elements(ls1)
    mesh.build_interface_segments(ls1)
    mesh.classify_edges(ls1)
    ghost = mesh.edge_bitset("ghost")
    assert ghost.cardinality() > 0

    dh = DofHandler(MixedElement(mesh, field_specs={"u": poly_order}), method="cg")

    counts = {"unified": 0, "cut": 0, "ghost": 0, "facet_patch": 0}
    orig_unified = dh.precompute_unified_factors
    orig_cut = dh.precompute_cut_volume_factors
    orig_ghost = dh.precompute_ghost_factors
    orig_facet_patch = dh.precompute_facet_patch_factors

    def wrap_unified(*args, **kwargs):
        counts["unified"] += 1
        return orig_unified(*args, **kwargs)

    def wrap_cut(*args, **kwargs):
        counts["cut"] += 1
        return orig_cut(*args, **kwargs)

    def wrap_ghost(*args, **kwargs):
        counts["ghost"] += 1
        return orig_ghost(*args, **kwargs)

    def wrap_facet_patch(*args, **kwargs):
        counts["facet_patch"] += 1
        return orig_facet_patch(*args, **kwargs)

    dh.precompute_unified_factors = wrap_unified  # type: ignore[method-assign]
    dh.precompute_cut_volume_factors = wrap_cut  # type: ignore[method-assign]
    dh.precompute_ghost_factors = wrap_ghost  # type: ignore[method-assign]
    dh.precompute_facet_patch_factors = wrap_facet_patch  # type: ignore[method-assign]

    u_pos = TrialFunction(field_name="u", name="u_pos", dof_handler=dh, side="+")
    v_pos = TestFunction(field_name="u", name="v_pos", dof_handler=dh, side="+")
    u_neg = TrialFunction(field_name="u", name="u_neg", dof_handler=dh, side="-")
    v_neg = TestFunction(field_name="u", name="v_neg", dof_handler=dh, side="-")

    q = 6
    # Two cut-volume integrals per side with different derivative requirements.
    a = (
        (u_pos * v_pos) * dx(level_set=ls1, metadata={"side": "+", "q": q, "derivs": {(0, 0)}})
        + (u_pos * v_pos) * dx(level_set=ls1, metadata={"side": "+", "q": q, "derivs": {(0, 0), (1, 0), (0, 1)}})
        + (u_neg * v_neg) * dx(level_set=ls1, metadata={"side": "-", "q": q, "derivs": {(0, 0)}})
        + (u_neg * v_neg) * dx(level_set=ls1, metadata={"side": "-", "q": q, "derivs": {(0, 0), (2, 0), (1, 1), (0, 2)}})
    )

    # Two ghost-edge integrals with different derivative requirements.
    jump_u = Jump(u_pos, u_neg)
    jump_v = Jump(v_pos, v_neg)
    a += (jump_u * jump_v) * dGhost(defined_on=ghost, level_set=ls1, metadata={"q": q, "derivs": {(1, 0), (0, 1)}})
    a += (jump_u * jump_v) * dGhost(defined_on=ghost, level_set=ls1, metadata={"q": q, "derivs": {(2, 0), (1, 1), (0, 2)}})

    # Two facet-patch integrals (supported derivative orders only).
    a += (jump_u * jump_v) * dFacetPatch(defined_on=ghost, level_set=ls1, metadata={"q": q, "derivs": {(0, 0)}})
    a += (jump_u * jump_v) * dFacetPatch(defined_on=ghost, level_set=ls1, metadata={"q": q, "derivs": {(1, 0), (0, 1)}})

    kernels = compile_multi(Equation(a, None), dof_handler=dh, mixed_element=dh.mixed_element, backend="jit")

    # Refresh with a fresh-but-equivalent level set object to avoid hitting the per-kernel unified cache.
    counts.update({"unified": 0, "cut": 0, "ghost": 0, "facet_patch": 0})
    clear_caches()
    ls2 = AffineLevelSet(a=1.0, b=0.0, c=-1.03)
    mesh.classify_elements(ls2)
    mesh.build_interface_segments(ls2)
    mesh.classify_edges(ls2)
    for k in kernels:
        if getattr(k, "level_set", None) is not None:
            k.refresh(ls2)

    assert counts["unified"] >= 1
    # Unified bundle computes cut +/- once each and ghost once.
    assert counts["cut"] <= 2
    assert counts["ghost"] <= 1
    assert counts["facet_patch"] <= 1
