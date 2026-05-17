from __future__ import annotations

import numpy as np

from pycutfem.mor import (
    AffineStateUpdateSpec,
    BoundConstraintSpec,
    NativeAdjointDWRSpec,
    NativeGnatTargetSpec,
    NativeKernelReference,
    NativeReducedArtifact,
    NativeSparseMatrix,
    NativeStateArraySpec,
    StateTransactionSpec,
    SymbolicStateUpdateKernelSpec,
    apply_affine_state_updates,
    load_native_reduced_artifact,
)


def test_affine_and_symbolic_state_update_specs_validate_and_apply() -> None:
    affine = AffineStateUpdateSpec(
        name="qstate",
        basis=np.array([[1.0, 0.0], [0.5, -1.0], [0.0, 2.0]], dtype=float),
        offset=np.array([0.25, -0.5, 1.0], dtype=float),
    )
    symbolic = SymbolicStateUpdateKernelSpec(
        name="dvms_refresh",
        kernel_id="state_kernel",
        abi="native-state-v1",
        param_order=("qstate", "u_loc"),
        target_names=("qstate",),
        argument_map={"u_loc": "reduced_state"},
    )
    transaction = StateTransactionSpec(
        state_arrays=(NativeStateArraySpec(name="qstate", shape=(3,), role="quadrature"),),
        affine_updates=(affine,),
        symbolic_updates=(symbolic,),
    )

    values = apply_affine_state_updates(transaction.affine_updates, np.array([2.0, -0.5], dtype=float))

    np.testing.assert_allclose(values["qstate"], np.array([2.25, 1.0, 0.0]))
    payload = transaction.to_native_dict()
    assert payload["restore_on_reject"]
    assert payload["symbolic_updates"][0]["kernel_id"] == "state_kernel"


def test_native_reduced_artifact_roundtrip_preserves_sparse_target_and_kernel_metadata(tmp_path) -> None:
    lift = NativeSparseMatrix.from_dense(np.array([[1.0, 0.0, -0.5], [0.0, 2.0, 0.25]], dtype=float))
    artifact = NativeReducedArtifact(
        problem_id="generic_problem",
        trial_basis=np.array([[1.0, 0.0], [0.5, 1.0], [0.0, -1.0]], dtype=float),
        offset=np.array([0.0, 1.0, -1.0], dtype=float),
        residual_kernel=NativeKernelReference(
            kernel_id="residual_kernel",
            abi="native-kernel-v1",
            param_order=("gdofs_map", "u_loc"),
            metadata={"form": "residual"},
        ),
        tangent_kernel=NativeKernelReference(
            kernel_id="tangent_kernel",
            abi="native-kernel-v1",
            param_order=("gdofs_map", "u_loc"),
            metadata={"form": "tangent"},
        ),
        target=NativeGnatTargetSpec(
            row_dofs=np.array([0, 2, 5], dtype=np.int64),
            element_ids=np.array([1, 4], dtype=np.int64),
            row_weights=np.array([1.0, 0.5, 2.0], dtype=float),
            lift=lift,
            objective="gnat",
            metadata={"sampler": "unit"},
        ),
        bound_constraints=BoundConstraintSpec(
            rows=np.array([1, 2], dtype=np.int64),
            lower=np.array([0.0, -np.inf], dtype=float),
            upper=np.array([1.0, 0.5], dtype=float),
            row_scaling=np.array([2.0, 1.0], dtype=float),
            metadata={"fields": ("alpha", "phi")},
        ),
        adjoint_dwr=NativeAdjointDWRSpec(
            qoi_name="alpha_mass",
            qoi_kernel=NativeKernelReference(
                kernel_id="qoi_kernel",
                abi="native-kernel-v1",
                param_order=("gdofs_map", "alpha_loc"),
            ),
            qoi_current_gradient_kernel=NativeKernelReference(
                kernel_id="qoi_gradient_kernel",
                abi="native-kernel-v1",
                param_order=("gdofs_map", "alpha_loc"),
            ),
            adjoint_basis=np.array([[1.0], [0.5], [0.0]], dtype=float),
            solver_options={"backend": "cpp", "rcond": 1.0e-12},
            estimator_options={"effectivity_bounds": (0.5, 2.0)},
            certification={"passed": False, "reason": "not_run"},
            metadata={"type": "dwr"},
        ),
        solver_options={"max_iterations": 8, "line_search": True},
        metadata={"basis_signature": "test"},
    )
    path = tmp_path / "native_reduced_artifact.npz"

    artifact.save(path)
    loaded = load_native_reduced_artifact(path)

    assert loaded.problem_id == artifact.problem_id
    assert loaded.residual_kernel.kernel_id == "residual_kernel"
    assert loaded.tangent_kernel is not None
    assert loaded.tangent_kernel.metadata["form"] == "tangent"
    assert loaded.target is not None
    assert isinstance(loaded.target.lift, NativeSparseMatrix)
    np.testing.assert_allclose(loaded.target.lift.to_dense(), lift.to_dense())
    np.testing.assert_allclose(loaded.trial_basis, artifact.trial_basis)
    np.testing.assert_allclose(loaded.offset, artifact.offset)
    assert loaded.bound_constraints is not None
    np.testing.assert_array_equal(loaded.bound_constraints.rows, np.array([1, 2], dtype=np.int64))
    np.testing.assert_allclose(loaded.bound_constraints.row_scaling, np.array([2.0, 1.0]))
    reduced_bounds = loaded.bound_constraints.reduce(trial_basis=loaded.trial_basis, offset=loaded.offset)
    assert reduced_bounds.n_constraints == 2
    assert loaded.adjoint_dwr is not None
    assert loaded.adjoint_dwr.qoi_name == "alpha_mass"
    assert loaded.adjoint_dwr.qoi_current_gradient_kernel is not None
    assert loaded.adjoint_dwr.qoi_current_gradient_kernel.kernel_id == "qoi_gradient_kernel"
    np.testing.assert_allclose(loaded.adjoint_dwr.adjoint_basis, np.array([[1.0], [0.5], [0.0]], dtype=float))
    assert loaded.adjoint_dwr.estimator_options["effectivity_bounds"] == (0.5, 2.0)
    assert loaded.solver_options["line_search"] is True
