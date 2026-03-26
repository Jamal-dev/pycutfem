from types import SimpleNamespace

import numpy as np
import pytest

from pycutfem.ufl.expressions import HdivFunction, Transpose, VectorTrialFunction, _expr_shape, grad
from pycutfem.ufl.helpers import GradOpInfo, VecOpInfo
from pycutfem.ufl.tensor_algebra import (
    AxisSpace,
    AxisLabel,
    BasisLabel,
    DotKernelCase,
    ExpressionMeta,
    FieldSource,
    OperationKind,
    ProvenanceKind,
    ProvenanceSignature,
    TensorAxis,
    TensorRuleEngine,
    TensorSignature,
    OperandTransform,
)


def test_infer_signature_for_scalar_gradient_basis_is_spatial_vector_basis():
    grad_trial = GradOpInfo(np.zeros((2, 3), dtype=float), role="trial", is_rhs=False)

    sig = TensorRuleEngine.infer_signature(grad_trial)

    assert sig.storage_kind == "grad"
    assert sig.basis_axes == (BasisLabel.TRIAL,)
    assert len(sig.free_axes) == 1
    assert sig.free_axes[0].label == AxisLabel.COMPONENT
    assert sig.free_axes[0].size == 2


def test_infer_signature_for_vector_gradient_basis_tracks_component_and_derivative_axes():
    grad_test = GradOpInfo(np.zeros((2, 4, 2), dtype=float), role="test", is_rhs=False)

    sig = TensorRuleEngine.infer_signature(grad_test)

    assert sig.basis_axes == (BasisLabel.TEST,)
    assert [axis.label for axis in sig.free_axes] == [AxisLabel.COMPONENT, AxisLabel.DERIVATIVE]
    assert [axis.size for axis in sig.free_axes] == [2, 2]


def test_dot_plan_distinguishes_vector_dot_scalar_gradient_from_grad_dot_vector():
    vec_val = VecOpInfo(np.array([1.0, -2.0], dtype=float), role="function", is_rhs=False)
    grad_scalar = GradOpInfo(np.array([[3.0, 4.0]], dtype=float), role="function", is_rhs=False)
    grad_vector = GradOpInfo(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float), role="function", is_rhs=False)

    scalar_plan = TensorRuleEngine.plan_dot(vec_val, grad_scalar)
    vector_plan = TensorRuleEngine.plan_dot(grad_vector, vec_val)

    assert scalar_plan.kind == OperationKind.DOT_VECTOR_VECTOR
    assert scalar_plan.result.tensor_rank == 0
    assert vector_plan.kind == OperationKind.DOT_TENSOR_TENSOR
    assert [axis.label for axis in vector_plan.result.free_axes] == [AxisLabel.COMPONENT]


def test_dot_plan_for_grad_trial_and_vector_test_is_mixed_and_canonical():
    grad_trial = GradOpInfo(np.zeros((2, 5, 2), dtype=float), role="trial", is_rhs=False)
    v_test = VecOpInfo(np.zeros((2, 7), dtype=float), role="test", is_rhs=False)

    plan = TensorRuleEngine.plan_dot(grad_trial, v_test)
    lowering = TensorRuleEngine.plan_dot_lowering(grad_trial, v_test)

    assert plan.kind == OperationKind.DOT_TENSOR_TENSOR
    assert plan.result.role == "mixed"
    assert plan.result.basis_axes == (BasisLabel.TEST, BasisLabel.TRIAL)
    assert [axis.label for axis in plan.result.free_axes] == [AxisLabel.COMPONENT]
    assert lowering.lhs_storage.free_axis_positions == (0, 2)
    assert lowering.lhs_storage.basis_axis_positions == (1,)
    assert lowering.result_storage.stored_shape == (2, 7, 5)


def test_dot_lowering_storage_for_scalar_gradient_basis_is_exact_and_axis_driven():
    vec_value = VecOpInfo(np.array([1.0, -2.0], dtype=float), role="function", is_rhs=False)
    grad_basis = GradOpInfo(np.zeros((2, 5), dtype=float), role="test", is_rhs=False)

    lowering = TensorRuleEngine.plan_dot_lowering(vec_value, grad_basis)

    assert lowering.algebra.result.tensor_rank == 0
    assert lowering.rhs_storage.free_axis_positions == (0,)
    assert lowering.rhs_storage.basis_axis_positions == (1,)
    assert lowering.result_storage.stored_shape == (1, 5)


def test_dot_kernel_plan_classifies_basis_gradient_and_value_vector():
    grad_trial = GradOpInfo(np.zeros((2, 5, 2), dtype=float), role="trial", is_rhs=False)
    vec_value = VecOpInfo(np.array([1.0, -2.0], dtype=float), role="function", is_rhs=False)

    kernel_plan = TensorRuleEngine.plan_dot_kernel(grad_trial, vec_value)

    assert kernel_plan.case == DotKernelCase.BASIS_GRAD_DOT_VALUE_VECTOR
    assert kernel_plan.lowering.result_storage.stored_shape == (2, 5)


def test_dot_kernel_plan_classifies_value_gradient_and_basis_vector():
    grad_value = GradOpInfo(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float), role="function", is_rhs=False)
    vec_trial = VecOpInfo(np.zeros((2, 6), dtype=float), role="trial", is_rhs=False)

    kernel_plan = TensorRuleEngine.plan_dot_kernel(grad_value, vec_trial)

    assert kernel_plan.case == DotKernelCase.VALUE_GRAD_DOT_BASIS_VECTOR
    assert kernel_plan.lowering.result_storage.stored_shape == (2, 6)


def test_dot_kernel_plan_classifies_basis_gradient_and_basis_vector_as_mixed():
    grad_trial = GradOpInfo(np.zeros((2, 5, 2), dtype=float), role="trial", is_rhs=False)
    vec_test = VecOpInfo(np.zeros((2, 7), dtype=float), role="test", is_rhs=False)

    kernel_plan = TensorRuleEngine.plan_dot_kernel(grad_trial, vec_test)

    assert kernel_plan.case == DotKernelCase.BASIS_GRAD_DOT_BASIS_VECTOR
    assert kernel_plan.lowering.result_storage.stored_shape == (2, 7, 5)


def test_dot_kernel_plan_preserves_gradient_semantics_from_provenance_across_chained_dot():
    transported_grad_basis = SimpleNamespace(
        var_name="Fg_test",
        role="test",
        shape=(2, 5),
        is_gradient=False,
        is_vector=True,
        is_hessian=False,
        expression_meta=ExpressionMeta(
            tensor=TensorSignature(
                free_axes=(TensorAxis(AxisSpace.PHYSICAL, AxisLabel.COMPONENT, 2),),
                basis_axes=(BasisLabel.TEST,),
                storage_kind="dot_result",
                raw_shape=(2, 5),
                role="test",
                source="dot",
            ),
            provenance=ProvenanceSignature(
                (
                    FieldSource("u", ("ux", "uy"), ProvenanceKind.FIELD_COMPONENTS, 0, "test", "dot"),
                    FieldSource("u", ("ux", "uy"), ProvenanceKind.DERIVATIVE_CHANNELS, 1, "test", "dot"),
                )
            ),
        ),
    )
    vec_trial = VecOpInfo(np.zeros((2, 7), dtype=float), role="trial", is_rhs=False)

    kernel_plan = TensorRuleEngine.plan_dot_kernel(transported_grad_basis, vec_trial)

    assert kernel_plan.case == DotKernelCase.BASIS_GRAD_DOT_BASIS_VECTOR


def test_product_lowering_storage_for_scalar_basis_times_gradient_value_is_exact():
    scalar_trial = VecOpInfo(np.zeros((1, 6), dtype=float), role="trial", is_rhs=False)
    grad_value = GradOpInfo(np.array([[0.5, -0.25]], dtype=float), role="function", is_rhs=False)

    lowering = TensorRuleEngine.plan_product_lowering(scalar_trial, grad_value)

    assert lowering.result_storage.stored_shape == (2, 6)


def test_sum_lowering_canonicalizes_legacy_scalar_gradient_basis_storage():
    grad_basis_canonical = GradOpInfo(np.zeros((2, 5), dtype=float), role="trial", is_rhs=False)
    grad_basis_legacy = GradOpInfo(np.zeros((1, 5, 2), dtype=float), role="trial", is_rhs=False)

    assert grad_basis_legacy.shape == (2, 5)

    lowering = TensorRuleEngine.plan_sum_lowering(grad_basis_canonical, grad_basis_legacy)

    assert lowering.lhs_transform == OperandTransform.NONE
    assert lowering.rhs_transform == OperandTransform.NONE
    assert lowering.result_storage.stored_shape == (2, 5)


def test_grad_addition_uses_shared_sum_lowering_and_returns_canonical_shape():
    grad_basis_canonical = GradOpInfo(np.zeros((2, 5), dtype=float), role="trial", is_rhs=False)
    grad_basis_legacy = GradOpInfo(np.zeros((1, 5, 2), dtype=float), role="trial", is_rhs=False)

    result = grad_basis_canonical + grad_basis_legacy

    assert isinstance(result, GradOpInfo)
    assert result.shape == (2, 5)


def test_product_plan_promotes_scalar_basis_times_scalar_gradient_value():
    scalar_trial = VecOpInfo(np.zeros((1, 6), dtype=float), role="trial", is_rhs=False)
    grad_value = GradOpInfo(np.array([[0.5, -0.25]], dtype=float), role="function", is_rhs=False)

    plan = TensorRuleEngine.plan_product(scalar_trial, grad_value)

    assert plan.kind == OperationKind.PRODUCT_PROMOTE
    assert plan.result.role == "trial"
    assert plan.result.basis_axes == (BasisLabel.TRIAL,)
    assert [axis.label for axis in plan.result.free_axes] == [AxisLabel.DERIVATIVE]


def test_product_plan_for_vector_times_vector_is_dyad():
    lhs = VecOpInfo(np.array([1.0, -2.0], dtype=float), role="function", is_rhs=False)
    rhs = VecOpInfo(np.array([0.5, 3.0], dtype=float), role="function", is_rhs=False)

    plan = TensorRuleEngine.plan_product(lhs, rhs)

    assert plan.kind == OperationKind.PRODUCT_OUTER
    assert plan.result.tensor_rank == 2
    assert [axis.label for axis in plan.result.free_axes] == [AxisLabel.COMPONENT, AxisLabel.COMPONENT]
    assert [axis.size for axis in plan.result.free_axes] == [2, 2]


def test_product_plan_for_vector_times_matrix_appends_all_free_axes():
    lhs = VecOpInfo(np.array([1.0, -2.0], dtype=float), role="function", is_rhs=False)
    rhs = SimpleNamespace(
        var_name="B",
        role="value",
        shape=(2, 2),
        is_gradient=False,
        is_vector=False,
        is_hessian=False,
    )

    plan = TensorRuleEngine.plan_product(lhs, rhs)

    assert plan.kind == OperationKind.PRODUCT_TENSOR
    assert plan.result.tensor_rank == 3
    assert [axis.size for axis in plan.result.free_axes] == [2, 2, 2]


def test_codegen_stack_item_inference_for_scalar_gradient_basis_uses_same_semantics():
    item = SimpleNamespace(
        var_name="g_alpha",
        role="trial",
        shape=(2, -1),
        is_gradient=True,
        is_vector=False,
        is_hessian=False,
    )

    sig = TensorRuleEngine.infer_signature(item)

    assert sig.storage_kind == "grad"
    assert sig.basis_axes == (BasisLabel.TRIAL,)
    assert [axis.label for axis in sig.free_axes] == [AxisLabel.COMPONENT]


def test_codegen_stack_item_inference_for_value_gradient_with_shape_1_d_is_rank_1():
    item = SimpleNamespace(
        var_name="hdotn",
        role="value",
        shape=(1, 2),
        is_gradient=True,
        is_vector=False,
        is_hessian=False,
    )

    sig = TensorRuleEngine.infer_signature(item)

    assert sig.storage_kind == "grad"
    assert sig.tensor_rank == 1
    assert [axis.label for axis in sig.free_axes] == [AxisLabel.DERIVATIVE]
    assert [axis.size for axis in sig.free_axes] == [2]


def test_codegen_stack_item_inference_for_mixed_basis_and_mixed_gradient():
    mixed_basis = SimpleNamespace(
        var_name="m_uv",
        role="mixed",
        shape=(2, -1, -1),
        is_gradient=False,
        is_vector=False,
        is_hessian=False,
    )
    mixed_grad = SimpleNamespace(
        var_name="mg_uv",
        role="mixed",
        shape=(2, -1, -1, 2),
        is_gradient=True,
        is_vector=False,
        is_hessian=False,
    )

    basis_sig = TensorRuleEngine.infer_signature(mixed_basis)
    grad_sig = TensorRuleEngine.infer_signature(mixed_grad)

    assert basis_sig.basis_axes == (BasisLabel.TEST, BasisLabel.TRIAL)
    assert [axis.label for axis in basis_sig.free_axes] == [AxisLabel.COMPONENT]
    assert grad_sig.basis_axes == (BasisLabel.TEST, BasisLabel.TRIAL)
    assert [axis.label for axis in grad_sig.free_axes] == [AxisLabel.COMPONENT, AxisLabel.DERIVATIVE]


def test_dot_plan_for_codegen_vector_with_scalar_gradient_basis_collapses_to_scalar_basis():
    vec_value = SimpleNamespace(
        var_name="u_val",
        role="value",
        shape=(2,),
        is_gradient=False,
        is_vector=True,
        is_hessian=False,
    )
    grad_basis = SimpleNamespace(
        var_name="g_p",
        role="test",
        shape=(1, -1, 2),
        is_gradient=True,
        is_vector=False,
        is_hessian=False,
    )

    plan = TensorRuleEngine.plan_dot(vec_value, grad_basis)

    assert plan.kind == OperationKind.DOT_VECTOR_VECTOR
    assert plan.result.tensor_rank == 0
    assert plan.result.basis_axes == (BasisLabel.TEST,)
    assert plan.result.role == "test"


def test_dot_plan_for_codegen_grad_basis_with_vector_keeps_remaining_component_axis():
    grad_basis = SimpleNamespace(
        var_name="g_u",
        role="trial",
        shape=(2, -1, 2),
        is_gradient=True,
        is_vector=False,
        is_hessian=False,
    )
    vec_value = SimpleNamespace(
        var_name="u_val",
        role="value",
        shape=(2,),
        is_gradient=False,
        is_vector=True,
        is_hessian=False,
    )

    plan = TensorRuleEngine.plan_dot(grad_basis, vec_value)

    assert plan.kind == OperationKind.DOT_TENSOR_TENSOR
    assert plan.result.tensor_rank == 1
    assert [axis.label for axis in plan.result.free_axes] == [AxisLabel.COMPONENT]
    assert plan.result.basis_axes == (BasisLabel.TRIAL,)


def test_expression_meta_distinguishes_vector_field_components_from_scalar_gradient_channels():
    vec_basis = VecOpInfo(
        np.zeros((2, 5), dtype=float),
        role="test",
        field_names=["ux", "uy"],
        parent_name="u",
        is_rhs=False,
    )
    grad_scalar = GradOpInfo(
        np.zeros((1, 5, 2), dtype=float),
        role="test",
        field_names=["p"],
        parent_name="p",
        is_rhs=False,
    )

    vec_meta = TensorRuleEngine.infer_expression_meta(vec_basis)
    grad_meta = TensorRuleEngine.infer_expression_meta(grad_scalar)

    assert vec_meta.tensor.tensor_rank == 1
    assert grad_meta.tensor.tensor_rank == 1
    assert vec_meta.provenance.sources[0].kind == ProvenanceKind.FIELD_COMPONENTS
    assert vec_meta.provenance.sources[0].fields == ("ux", "uy")
    assert grad_meta.provenance.sources[0].kind == ProvenanceKind.DERIVATIVE_CHANNELS
    assert grad_meta.provenance.sources[0].fields == ("p",)
    assert grad_meta.provenance.sources[0].derivative_depth == 1


def test_expr_shape_recognizes_hdiv_functions_as_vectors_before_generic_function_case():
    v = HdivFunction(name="v", field_name="v")
    gv = grad(v)

    assert _expr_shape(v) == (2,)
    assert _expr_shape(gv) == (2, 2)


def test_transpose_rejects_rank_1_vector_expressions():
    vector_space = SimpleNamespace(name="u", field_names=["ux", "uy"], side="")

    with pytest.raises(TypeError):
        Transpose(VectorTrialFunction(vector_space))


def test_sum_plan_allows_scalar_gradient_plus_vector_basis_and_merges_provenance():
    vec_basis = VecOpInfo(
        np.zeros((2, 5), dtype=float),
        role="test",
        field_names=["ux", "uy"],
        parent_name="u",
        is_rhs=False,
    )
    grad_scalar = GradOpInfo(
        np.zeros((1, 5, 2), dtype=float),
        role="test",
        field_names=["p"],
        parent_name="p",
        is_rhs=False,
    )

    plan = TensorRuleEngine.plan_sum(vec_basis, grad_scalar)

    assert plan.kind == OperationKind.SUM_GENERIC
    assert plan.result.tensor.tensor_rank == 1
    assert plan.result.tensor.basis_axes == (BasisLabel.TEST,)
    assert len(plan.result.provenance.sources) == 2
    assert {src.kind for src in plan.result.provenance.sources} == {
        ProvenanceKind.FIELD_COMPONENTS,
        ProvenanceKind.DERIVATIVE_CHANNELS,
    }


def test_sum_plan_rejects_mixing_test_and_trial_basis_spaces():
    grad_test = GradOpInfo(np.zeros((1, 4, 2), dtype=float), role="test", field_names=["p"], parent_name="p", is_rhs=False)
    grad_trial = GradOpInfo(np.zeros((1, 4, 2), dtype=float), role="trial", field_names=["p"], parent_name="p", is_rhs=False)

    try:
        TensorRuleEngine.plan_sum(grad_test, grad_trial)
    except TypeError as exc:
        assert "basis signatures" in str(exc)
    else:
        raise AssertionError("Expected plan_sum to reject adding test and trial basis tensors.")


def test_sum_plan_broadcasts_scalar_basis_row_and_scalar_basis_vector_shapes():
    row_basis = VecOpInfo(np.zeros((1, 5), dtype=float), role="test", is_rhs=False)
    vec_basis = VecOpInfo(np.zeros((5,), dtype=float), role="test_n", is_rhs=True)

    plan = TensorRuleEngine.plan_sum(row_basis, vec_basis)

    assert plan.result.tensor.tensor_rank == 0
    assert plan.result.tensor.basis_axes == (BasisLabel.TEST,)
    assert plan.result.tensor.raw_shape == (1, 5)


def test_infer_signature_prefers_stored_expression_meta_over_legacy_layout_guessing():
    stored_meta = ExpressionMeta(
        tensor=TensorSignature(
            free_axes=(TensorAxis(AxisSpace.PHYSICAL, AxisLabel.DERIVATIVE, 2),),
            basis_axes=(BasisLabel.TEST,),
            storage_kind="grad",
            raw_shape=(1, -1, 2),
            role="test",
            source="stored",
        ),
        provenance=ProvenanceSignature(
            (
                FieldSource("p", ("p",), ProvenanceKind.DERIVATIVE_CHANNELS, 1, "test", "stored"),
            )
        ),
    )
    item = SimpleNamespace(
        var_name="lifted_grad",
        role="test",
        shape=(-1, 2),
        is_gradient=False,
        is_vector=False,
        is_hessian=False,
        expression_meta=stored_meta,
    )

    sig = TensorRuleEngine.infer_signature(item)
    meta = TensorRuleEngine.infer_expression_meta(item)

    assert sig.storage_kind == "grad"
    assert sig.tensor_rank == 1
    assert sig.free_axes[0].label == AxisLabel.DERIVATIVE
    assert meta is stored_meta


def test_product_lowering_uses_derivative_provenance_for_promoted_scalar_gradient_vectors():
    scalar_basis = SimpleNamespace(
        var_name="phi_p",
        role="test",
        shape=(1, 5),
        is_gradient=False,
        is_vector=False,
        is_hessian=False,
    )
    promoted_grad_vec = SimpleNamespace(
        var_name="grad_p_val",
        role="value",
        shape=(2,),
        is_gradient=False,
        is_vector=True,
        is_hessian=False,
        expression_meta=ExpressionMeta(
            tensor=TensorSignature(
                free_axes=(TensorAxis(AxisSpace.PHYSICAL, AxisLabel.DERIVATIVE, 2),),
                basis_axes=(),
                storage_kind="vec",
                raw_shape=(2,),
                role="value",
                source="stored",
            ),
            provenance=ProvenanceSignature(
                (
                    FieldSource("p", ("p",), ProvenanceKind.DERIVATIVE_CHANNELS, 1, "value", "stored"),
                )
            ),
        ),
    )

    lowering = TensorRuleEngine.plan_product_lowering(scalar_basis, promoted_grad_vec)

    assert lowering.result.role == "test"
    assert lowering.result.is_gradient
    assert not lowering.result.is_vector
    assert any(src.kind == ProvenanceKind.DERIVATIVE_CHANNELS for src in lowering.meta.provenance.sources)
