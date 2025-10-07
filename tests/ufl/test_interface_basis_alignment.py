import numpy as np


def test_interface_basis_matches_python(monkeypatch, tmp_path):
    cache_dir = tmp_path / "jitcache"
    cache_dir.mkdir()

    from pycutfem.jit import cache as jit_cache_module
    monkeypatch.setattr(jit_cache_module.KernelCache, "_CACHE_DIR", cache_dir, raising=False)

    from examples.debug import stokes_backend_diagnostics as diag
    from pycutfem.ufl.compilers import FormCompiler
    from pycutfem.jit import compile_backend
    from pycutfem.ufl.helpers_jit import _build_jit_kernel_args

    problem = diag.build_stokes_problem(with_deformation=True)
    dh = problem.dh
    me = dh.mixed_element

    fc = FormCompiler(dh, backend="jit")
    int_penalty = problem.equation.a.integrals[5]
    qdeg = fc._find_q_order(int_penalty)

    cut_ids = problem.mesh.element_bitset("cut").to_indices()
    if len(cut_ids) == 0:
        raise RuntimeError("Expected at least one cut element in the benchmark setup")

    cut_ids_array = np.asarray(cut_ids, dtype=np.int32)
    precomputed = dh.precompute_interface_factors(
        cut_ids_array,
        qdeg,
        problem.level_set,
        deformation=problem.deformation,
        reuse=False,
    )

    runner, ir = compile_backend(int_penalty.integrand, dh, me, on_facet=True)
    static_args = _build_jit_kernel_args(
        ir,
        int_penalty.integrand,
        me,
        qdeg,
        dh,
        param_order=runner.param_order,
        pre_built=precomputed,
    )

    field_side = (
        ("u_pos_x", "pos"),
        ("u_pos_y", "pos"),
        ("u_neg_x", "neg"),
        ("u_neg_y", "neg"),
    )

    for field, side in field_side:
        preferred = f"r00_{field}_{side}"
        if preferred in static_args:
            basis_jit = static_args[preferred]
            basis_ref = precomputed.get(preferred, precomputed[f"b_{field}"])
        else:
            fallback = f"b_{field}"
            assert fallback in static_args, f"missing {preferred} and {fallback} in JIT kernel arguments"
            basis_jit = static_args[fallback]
            basis_ref = precomputed[fallback]
        np.testing.assert_allclose(basis_jit, basis_ref, atol=1e-12, rtol=1e-12)
