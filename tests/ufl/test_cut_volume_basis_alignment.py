import numpy as np
import re


def _basis_keys(static_args):
    pattern = re.compile(r"^d\d\d_")
    for key in static_args:
        if key.startswith("b_") or key.startswith("g_") or pattern.match(key):
            yield key


def test_cut_volume_basis_matches_python(monkeypatch, tmp_path):
    cache_dir = tmp_path / "jitcache"
    cache_dir.mkdir()

    from pycutfem.jit import cache as jit_cache_module
    monkeypatch.setattr(jit_cache_module.KernelCache, "_CACHE_DIR", cache_dir, raising=False)

    from examples.debug import stokes_backend_diagnostics as diag
    from pycutfem.ufl.compilers import FormCompiler
    from pycutfem.ufl.helpers import required_multi_indices
    from pycutfem.jit import compile_backend
    from pycutfem.ufl.helpers_jit import _build_jit_kernel_args

    problem = diag.build_stokes_problem(with_deformation=True)
    dh = problem.dh
    me = dh.mixed_element

    # Pick the positive-side volume integral (dx_pos)
    int_vol = problem.equation.a.integrals[0]
    side = int_vol.measure.metadata.get("side", "+")

    fc = FormCompiler(dh, backend="jit")
    qdeg = fc._find_q_order(int_vol)
    req_derivs = required_multi_indices(int_vol.integrand) | {(0, 0)}

    cut_bs = problem.mesh.element_bitset("cut")
    precomputed = dh.precompute_cut_volume_factors(
        cut_bs,
        qdeg,
        req_derivs,
        problem.level_set,
        side=side,
        deformation=problem.deformation,
        reuse=False,
    )

    eids = precomputed.get("eids", np.array([], dtype=np.int32))
    assert eids.size > 0, "expected cut elements in benchmark setup"

    runner, ir = compile_backend(int_vol.integrand, dh, me, on_facet=False)

    gdofs_map = np.vstack([dh.get_elemental_dofs(int(e)) for e in eids]).astype(np.int32)
    prebuilt = dict(precomputed)
    prebuilt["gdofs_map"] = gdofs_map
    prebuilt["eids"] = eids.astype(np.int32)

    static_args = _build_jit_kernel_args(
        ir,
        int_vol.integrand,
        me,
        qdeg,
        dh,
        gdofs_map=gdofs_map,
        param_order=runner.param_order,
        pre_built=prebuilt,
    )

    for key in _basis_keys(static_args):
        assert key in precomputed, f"precompute missing {key}"
        np.testing.assert_allclose(static_args[key], precomputed[key], atol=1e-12, rtol=1e-12)
