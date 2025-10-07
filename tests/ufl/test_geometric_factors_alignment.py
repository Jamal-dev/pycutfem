import numpy as np


def test_precompute_geometric_factors_matches_python(monkeypatch, tmp_path):
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

    integral = problem.equation.a.integrals[0]  # dx_pos
    fc = FormCompiler(dh, backend="jit")
    qdeg = fc._find_q_order(integral)

    geo = dh.precompute_geometric_factors(
        qdeg,
        level_set=problem.level_set,
        deformation=problem.deformation,
        reuse=False,
    )

    runner, ir = compile_backend(integral.integrand, dh, me, on_facet=False)

    eids = geo["eids"]
    assert eids.size > 0
    gdofs_map = np.vstack([dh.get_elemental_dofs(int(e)) for e in eids]).astype(np.int32)

    prebuilt = dict(geo)
    prebuilt["gdofs_map"] = gdofs_map
    prebuilt["eids"] = eids.astype(np.int32)

    static_args = _build_jit_kernel_args(
        ir,
        integral.integrand,
        me,
        qdeg,
        dh,
        gdofs_map=gdofs_map,
        param_order=runner.param_order,
        pre_built=prebuilt,
    )

    for key in ("qp_phys", "qw", "detJ", "J_inv"):
        np.testing.assert_allclose(static_args[key], geo[key], atol=1e-12, rtol=1e-12)

