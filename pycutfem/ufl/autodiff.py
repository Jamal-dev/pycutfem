from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from pycutfem.ufl.analytic import Analytic
from pycutfem.ufl.expressions import (
    Acos,
    Acosh,
    Asin,
    Asinh,
    Atan,
    Atanh,
    Avg,
    CellDiameter,
    Cofactor,
    Constant,
    Cos,
    Cosh,
    Derivative,
    Determinant,
    Div,
    DivOperation,
    Dot,
    ElementWiseConstant,
    Exp,
    Expression,
    FacetNormal,
    Function,
    Grad,
    HdivFunction,
    HdivFunctionComponent,
    HdivTestFunction,
    HdivTestFunctionComponent,
    HdivTrialFunction,
    Hessian,
    Heaviside,
    Identity,
    Inner,
    Integral,
    Inverse,
    Jump,
    Laplacian,
    Log,
    MeshSize,
    Neg,
    NodalFunction,
    NormalComponent,
    Outer,
    Pos,
    PositivePart,
    Power,
    Prod,
    Restriction,
    Side,
    Sin,
    Sinh,
    Sub,
    Sum,
    Tan,
    Tanh,
    TestFunction,
    Trace,
    Transpose,
    TrialFunction,
    VectorFunction,
    VectorTestFunction,
    VectorTrialFunction,
    _expr_shape,
    _is_zero_expression_exact,
    _zero_expression,
    avg,
    cof,
    det,
    div,
    dot,
    grad,
    inner,
    laplacian,
    log,
)
from pycutfem.ufl.forms import Equation, Form


class GateauxDerivativeError(RuntimeError):
    """Base error for native pycutfem Gateaux differentiation."""


class UnsupportedGateauxNodeError(GateauxDerivativeError):
    """Raised when the native autodiff visitor encounters an unsupported node."""


class NonDifferentiableGateauxNodeError(GateauxDerivativeError):
    """Raised for hard non-smooth operators that depend on active coefficients."""


@dataclass(frozen=True)
class _CoefficientDirectionPair:
    coefficient: Expression
    direction: Expression


def _zero_like(expr: Expression) -> Expression:
    return _zero_expression(_expr_shape(expr))


def _is_zero(expr: Expression | None) -> bool:
    return expr is None or _is_zero_expression_exact(expr)


def _is_scalar_expr(expr: Expression) -> bool:
    return _expr_shape(expr) == ()


def _same_shape(a: Expression, b: Expression) -> bool:
    return _expr_shape(a) == _expr_shape(b)


def _component(expr: Expression, idx: int):
    try:
        return expr[idx]
    except Exception as exc:  # pragma: no cover - defensive
        raise GateauxDerivativeError(
            f"Direction {expr!r} does not expose component {idx} required by autodiff."
        ) from exc


def _direction_components(expr: Expression, count: int) -> list[Expression]:
    if hasattr(expr, "components"):
        comps = list(getattr(expr, "components"))
        if len(comps) == int(count):
            return comps
    return [_component(expr, i) for i in range(int(count))]


def _normalize_pairs(
    coefficients,
    directions,
) -> tuple[_CoefficientDirectionPair, ...]:
    coeff_seq = coefficients
    dir_seq = directions
    if isinstance(coeff_seq, Expression):
        coeff_seq = [coeff_seq]
    if isinstance(dir_seq, Expression):
        dir_seq = [dir_seq]
    coeff_seq = tuple(coeff_seq or ())
    dir_seq = tuple(dir_seq or ())
    if len(coeff_seq) != len(dir_seq):
        raise GateauxDerivativeError(
            f"Coefficient/direction arity mismatch: {len(coeff_seq)} vs {len(dir_seq)}."
        )
    if not coeff_seq:
        raise GateauxDerivativeError("At least one coefficient/direction pair is required.")
    pairs: list[_CoefficientDirectionPair] = []
    for coefficient, direction in zip(coeff_seq, dir_seq):
        if not isinstance(coefficient, Expression):
            raise GateauxDerivativeError(
                f"Coefficient {coefficient!r} is not a pycutfem Expression."
            )
        if not isinstance(direction, Expression):
            raise GateauxDerivativeError(
                f"Direction {direction!r} is not a pycutfem Expression."
            )
        if not _same_shape(coefficient, direction):
            raise GateauxDerivativeError(
                "Coefficient/direction shape mismatch for "
                f"{coefficient!r} and {direction!r}: "
                f"{_expr_shape(coefficient)!r} vs {_expr_shape(direction)!r}."
            )
        if isinstance(coefficient, Function) and not isinstance(
            coefficient, (VectorFunction, HdivFunction, TrialFunction, TestFunction)
        ):
            if not isinstance(direction, TrialFunction):
                raise GateauxDerivativeError(
                    f"Scalar coefficient {coefficient!r} must map to a TrialFunction direction, "
                    f"got {type(direction).__name__}."
                )
        if isinstance(coefficient, VectorFunction) and not isinstance(direction, VectorTrialFunction):
            raise GateauxDerivativeError(
                f"Vector coefficient {coefficient!r} must map to a VectorTrialFunction direction, "
                f"got {type(direction).__name__}."
            )
        if isinstance(coefficient, HdivFunction) and not isinstance(direction, HdivTrialFunction):
            raise GateauxDerivativeError(
                f"H(div) coefficient {coefficient!r} must map to an HdivTrialFunction direction, "
                f"got {type(direction).__name__}."
            )
        pairs.append(_CoefficientDirectionPair(coefficient=coefficient, direction=direction))
    return tuple(pairs)


class _GateauxContext:
    def __init__(self, pairs: Sequence[_CoefficientDirectionPair]):
        self.pairs = tuple(pairs)
        self._direct_by_id = {id(pair.coefficient): pair.direction for pair in self.pairs}
        self._vector_component_by_id: dict[int, Expression] = {}
        for pair in self.pairs:
            coeff = pair.coefficient
            direction = pair.direction
            if isinstance(coeff, VectorFunction):
                comps = list(getattr(coeff, "components", []))
                dcomps = _direction_components(direction, len(comps))
                for comp, dcomp in zip(comps, dcomps):
                    self._vector_component_by_id[id(comp)] = dcomp

    def direction_for_leaf(self, expr: Expression) -> Expression | None:
        direction = self._direct_by_id.get(id(expr))
        if direction is not None:
            return direction
        direction = self._vector_component_by_id.get(id(expr))
        if direction is not None:
            return direction
        if isinstance(expr, Function) and getattr(expr, "_parent_vector", None) is not None:
            parent = getattr(expr, "_parent_vector", None)
            parent_direction = self._direct_by_id.get(id(parent))
            if parent_direction is not None:
                return _component(parent_direction, int(getattr(expr, "_component_index")))
        if isinstance(expr, HdivFunctionComponent):
            parent_direction = self._direct_by_id.get(id(expr.parent))
            if parent_direction is not None:
                return _component(parent_direction, int(expr.component_index))
        return None


_ZERO_LEAF_TYPES = (
    Constant,
    Identity,
    Analytic,
    ElementWiseConstant,
    CellDiameter,
    MeshSize,
    FacetNormal,
    NormalComponent,
    NodalFunction,
    Function,
    VectorFunction,
    HdivFunction,
    HdivFunctionComponent,
    VectorTestFunction,
    HdivTestFunction,
    HdivTestFunctionComponent,
    TrialFunction,
    TestFunction,
)


def _sum_like(base: Expression, *terms: Expression) -> Expression:
    kept = [term for term in terms if not _is_zero(term)]
    if not kept:
        return _zero_like(base)
    out = kept[0]
    for term in kept[1:]:
        out = out + term
    return out


def _has_expr_children(expr: Expression) -> bool:
    for value in expr.__dict__.values():
        if isinstance(value, Expression):
            return True
        if isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, Expression):
                    return True
    return False


def _differentiate_expression(expr: Expression, ctx: _GateauxContext) -> Expression:
    mapped_direction = ctx.direction_for_leaf(expr)
    if mapped_direction is not None:
        return mapped_direction

    if isinstance(expr, _ZERO_LEAF_TYPES):
        return _zero_like(expr)
    if isinstance(expr, HdivTrialFunction):
        return _zero_like(expr)
    if isinstance(expr, VectorTrialFunction):
        return _zero_like(expr)

    if isinstance(expr, Sum):
        return _sum_like(expr, _differentiate_expression(expr.a, ctx), _differentiate_expression(expr.b, ctx))
    if isinstance(expr, Sub):
        da = _differentiate_expression(expr.a, ctx)
        db = _differentiate_expression(expr.b, ctx)
        if _is_zero(da) and _is_zero(db):
            return _zero_like(expr)
        if _is_zero(db):
            return da
        if _is_zero(da):
            return -db
        return da - db
    if isinstance(expr, Prod):
        da = _differentiate_expression(expr.a, ctx)
        db = _differentiate_expression(expr.b, ctx)
        if _is_zero(da) and _is_zero(db):
            return _zero_like(expr)
        if _is_zero(da):
            return expr.a * db
        if _is_zero(db):
            return da * expr.b
        return _sum_like(expr, da * expr.b, expr.a * db)
    if isinstance(expr, Div):
        da = _differentiate_expression(expr.a, ctx)
        db = _differentiate_expression(expr.b, ctx)
        if _is_zero(da) and _is_zero(db):
            return _zero_like(expr)
        numer = _zero_like(expr.a) if _is_zero(da) else da * expr.b
        if not _is_zero(db):
            numer = numer - expr.a * db
        return numer / (expr.b * expr.b)
    if isinstance(expr, Outer):
        da = _differentiate_expression(expr.a, ctx)
        db = _differentiate_expression(expr.b, ctx)
        if _is_zero(da) and _is_zero(db):
            return _zero_like(expr)
        left = _zero_like(expr) if _is_zero(da) else Outer(da, expr.b)
        right = _zero_like(expr) if _is_zero(db) else Outer(expr.a, db)
        return _sum_like(expr, left, right)
    if isinstance(expr, Dot):
        da = _differentiate_expression(expr.a, ctx)
        db = _differentiate_expression(expr.b, ctx)
        if _is_zero(da) and _is_zero(db):
            return _zero_like(expr)
        left = _zero_like(expr) if _is_zero(da) else dot(da, expr.b)
        right = _zero_like(expr) if _is_zero(db) else dot(expr.a, db)
        return _sum_like(expr, left, right)
    if isinstance(expr, Inner):
        da = _differentiate_expression(expr.a, ctx)
        db = _differentiate_expression(expr.b, ctx)
        if _is_zero(da) and _is_zero(db):
            return _zero_like(expr)
        left = _zero_like(expr) if _is_zero(da) else inner(da, expr.b)
        right = _zero_like(expr) if _is_zero(db) else inner(expr.a, db)
        return _sum_like(expr, left, right)
    if isinstance(expr, Pos):
        dchild = _differentiate_expression(expr.operand, ctx)
        return _zero_like(expr) if _is_zero(dchild) else Pos(dchild)
    if isinstance(expr, Neg):
        dchild = _differentiate_expression(expr.operand, ctx)
        return _zero_like(expr) if _is_zero(dchild) else Neg(dchild)
    if isinstance(expr, Restriction):
        dchild = _differentiate_expression(expr.operand, ctx)
        return _zero_like(expr) if _is_zero(dchild) else Restriction(dchild, expr.domain)
    if isinstance(expr, Side):
        dchild = _differentiate_expression(expr.f, ctx)
        return _zero_like(expr) if _is_zero(dchild) else Side(dchild, expr.side)
    if isinstance(expr, Avg):
        dchild = _differentiate_expression(expr.v, ctx)
        return _zero_like(expr) if _is_zero(dchild) else avg(dchild)
    if isinstance(expr, Jump):
        dpos = _differentiate_expression(expr.u_pos, ctx)
        dneg = _differentiate_expression(expr.u_neg, ctx)
        if _is_zero(dpos) and _is_zero(dneg):
            return _zero_like(expr)
        return Jump(dpos, dneg)
    if isinstance(expr, Transpose):
        dchild = _differentiate_expression(expr.A, ctx)
        return _zero_like(expr) if _is_zero(dchild) else Transpose(dchild)
    if isinstance(expr, Grad):
        dchild = _differentiate_expression(expr.operand, ctx)
        return _zero_like(expr) if _is_zero(dchild) else grad(dchild)
    if isinstance(expr, Hessian):
        dchild = _differentiate_expression(expr.operand, ctx)
        return _zero_like(expr) if _is_zero(dchild) else Hessian(dchild)
    if isinstance(expr, Laplacian):
        dchild = _differentiate_expression(expr.operand, ctx)
        return _zero_like(expr) if _is_zero(dchild) else laplacian(dchild)
    if isinstance(expr, DivOperation):
        dchild = _differentiate_expression(expr.operand, ctx)
        return _zero_like(expr) if _is_zero(dchild) else div(dchild)
    if isinstance(expr, Derivative):
        dchild = _differentiate_expression(expr.f, ctx)
        return _zero_like(expr) if _is_zero(dchild) else Derivative(dchild, *expr.order)

    if isinstance(expr, Power):
        da = _differentiate_expression(expr.a, ctx)
        db = _differentiate_expression(expr.b, ctx)
        if _is_zero(da) and _is_zero(db):
            return _zero_like(expr)
        if not _is_scalar_expr(expr.a) or not _is_scalar_expr(expr.b):
            raise UnsupportedGateauxNodeError(
                f"Power autodiff currently requires scalar base and exponent, got "
                f"{_expr_shape(expr.a)!r} and {_expr_shape(expr.b)!r}."
            )
        if _is_zero(db):
            if isinstance(expr.b, Constant):
                p = float(expr.b.value)
                return Constant(p) * (expr.a ** Constant(p - 1.0)) * da
            return expr.b * (expr.a ** (expr.b - Constant(1.0))) * da
        return (expr.a ** expr.b) * (db * log(expr.a) + expr.b * da / expr.a)

    if isinstance(expr, Log):
        darg = _differentiate_expression(expr.operand, ctx)
        return _zero_like(expr) if _is_zero(darg) else darg / expr.operand
    if isinstance(expr, Exp):
        darg = _differentiate_expression(expr.operand, ctx)
        return _zero_like(expr) if _is_zero(darg) else Exp(expr.operand) * darg
    if isinstance(expr, Tanh):
        darg = _differentiate_expression(expr.operand, ctx)
        if _is_zero(darg):
            return _zero_like(expr)
        th = Tanh(expr.operand)
        return (Constant(1.0) - th * th) * darg
    if isinstance(expr, Sin):
        darg = _differentiate_expression(expr.operand, ctx)
        return _zero_like(expr) if _is_zero(darg) else Cos(expr.operand) * darg
    if isinstance(expr, Cos):
        darg = _differentiate_expression(expr.operand, ctx)
        return _zero_like(expr) if _is_zero(darg) else (-Sin(expr.operand)) * darg
    if isinstance(expr, Tan):
        darg = _differentiate_expression(expr.operand, ctx)
        if _is_zero(darg):
            return _zero_like(expr)
        tn = Tan(expr.operand)
        return (Constant(1.0) + tn * tn) * darg
    if isinstance(expr, Asin):
        darg = _differentiate_expression(expr.operand, ctx)
        if _is_zero(darg):
            return _zero_like(expr)
        return darg / ((Constant(1.0) - expr.operand * expr.operand) ** Constant(0.5))
    if isinstance(expr, Acos):
        darg = _differentiate_expression(expr.operand, ctx)
        if _is_zero(darg):
            return _zero_like(expr)
        return -(darg / ((Constant(1.0) - expr.operand * expr.operand) ** Constant(0.5)))
    if isinstance(expr, Atan):
        darg = _differentiate_expression(expr.operand, ctx)
        if _is_zero(darg):
            return _zero_like(expr)
        return darg / (Constant(1.0) + expr.operand * expr.operand)
    if isinstance(expr, Sinh):
        darg = _differentiate_expression(expr.operand, ctx)
        return _zero_like(expr) if _is_zero(darg) else Cosh(expr.operand) * darg
    if isinstance(expr, Cosh):
        darg = _differentiate_expression(expr.operand, ctx)
        return _zero_like(expr) if _is_zero(darg) else Sinh(expr.operand) * darg
    if isinstance(expr, Asinh):
        darg = _differentiate_expression(expr.operand, ctx)
        if _is_zero(darg):
            return _zero_like(expr)
        return darg / ((Constant(1.0) + expr.operand * expr.operand) ** Constant(0.5))
    if isinstance(expr, Acosh):
        darg = _differentiate_expression(expr.operand, ctx)
        if _is_zero(darg):
            return _zero_like(expr)
        return darg / (
            ((expr.operand - Constant(1.0)) ** Constant(0.5))
            * ((expr.operand + Constant(1.0)) ** Constant(0.5))
        )
    if isinstance(expr, Atanh):
        darg = _differentiate_expression(expr.operand, ctx)
        if _is_zero(darg):
            return _zero_like(expr)
        return darg / (Constant(1.0) - expr.operand * expr.operand)

    if isinstance(expr, PositivePart):
        darg = _differentiate_expression(expr.operand, ctx)
        if _is_zero(darg):
            return _zero_like(expr)
        raise NonDifferentiableGateauxNodeError(
            "Gateaux derivative of PositivePart is intentionally unsupported. "
            "Use a smooth approximation or supply a manual Jacobian."
        )
    if isinstance(expr, Heaviside):
        darg = _differentiate_expression(expr.operand, ctx)
        if _is_zero(darg):
            return _zero_like(expr)
        raise NonDifferentiableGateauxNodeError(
            "Gateaux derivative of Heaviside is intentionally unsupported. "
            "Use a smooth approximation or supply a manual Jacobian."
        )

    if isinstance(expr, Trace):
        dchild = _differentiate_expression(expr.A, ctx)
        return _zero_like(expr) if _is_zero(dchild) else Trace(dchild)
    if isinstance(expr, Cofactor):
        if _expr_shape(expr.A) != (2, 2):
            raise UnsupportedGateauxNodeError(
                f"Cofactor autodiff is implemented only for 2x2 tensors, got {_expr_shape(expr.A)!r}."
            )
        dchild = _differentiate_expression(expr.A, ctx)
        return _zero_like(expr) if _is_zero(dchild) else cof(dchild)
    if isinstance(expr, Determinant):
        if _expr_shape(expr.A) != (2, 2):
            raise UnsupportedGateauxNodeError(
                f"Determinant autodiff is implemented only for 2x2 tensors, got {_expr_shape(expr.A)!r}."
            )
        dchild = _differentiate_expression(expr.A, ctx)
        return _zero_like(expr) if _is_zero(dchild) else inner(cof(expr.A), dchild)
    if isinstance(expr, Inverse):
        if _expr_shape(expr.A) != (2, 2):
            raise UnsupportedGateauxNodeError(
                f"Inverse autodiff is implemented only for 2x2 tensors, got {_expr_shape(expr.A)!r}."
            )
        dchild = _differentiate_expression(expr.A, ctx)
        if _is_zero(dchild):
            return _zero_like(expr)
        inv_a = Inverse(expr.A)
        return -(dot(inv_a, dot(dchild, inv_a)))

    if _has_expr_children(expr):
        child_derivatives = []
        for value in expr.__dict__.values():
            if isinstance(value, Expression):
                child_derivatives.append(_differentiate_expression(value, ctx))
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Expression):
                        child_derivatives.append(_differentiate_expression(item, ctx))
        if child_derivatives and all(_is_zero(item) for item in child_derivatives):
            return _zero_like(expr)

    raise UnsupportedGateauxNodeError(
        f"Gateaux derivative is not implemented for {type(expr).__name__}: {expr!r}"
    )


def gateaux_derivative(
    obj,
    coefficients,
    directions,
    *,
    strict: bool = True,
):
    """Differentiate a pycutfem expression/form with respect to selected coefficients."""
    pairs = _normalize_pairs(coefficients, directions)
    ctx = _GateauxContext(pairs)

    if isinstance(obj, Integral):
        dintegrand = _differentiate_expression(obj.integrand, ctx)
        if strict and _is_zero(dintegrand):
            return Integral(_zero_like(obj.integrand), obj.measure)
        return Integral(dintegrand, obj.measure)

    if isinstance(obj, Form):
        derived_integrals = []
        for integral in obj.integrals:
            dintegral = gateaux_derivative(integral, coefficients, directions, strict=strict)
            if isinstance(dintegral, Integral):
                if _is_zero(dintegral.integrand):
                    continue
                derived_integrals.append(dintegral)
            else:  # pragma: no cover - defensive
                raise GateauxDerivativeError(
                    f"Unexpected derivative object {type(dintegral).__name__} for integral {integral!r}."
                )
        return Form(derived_integrals)

    if isinstance(obj, Equation):
        return Equation(
            gateaux_derivative(obj.a, coefficients, directions, strict=strict) if obj.a is not None else None,
            gateaux_derivative(obj.L, coefficients, directions, strict=strict) if obj.L is not None else None,
        )

    if isinstance(obj, Expression):
        return _differentiate_expression(obj, ctx)

    raise GateauxDerivativeError(
        f"Unsupported object type for Gateaux differentiation: {type(obj).__name__}."
    )


def linearize_form(
    residual_form: Form | Integral,
    coefficients,
    directions,
    *,
    strict: bool = True,
) -> Form:
    out = gateaux_derivative(residual_form, coefficients, directions, strict=strict)
    if not isinstance(out, Form):
        if isinstance(out, Integral):
            return Form([out])
        raise GateauxDerivativeError(
            f"linearize_form expected a Form/Integral result, got {type(out).__name__}."
        )
    return out


def linearize_equation(
    residual_form: Form | Integral,
    coefficients,
    directions,
    *,
    strict: bool = True,
) -> Equation:
    jacobian_form = linearize_form(
        residual_form,
        coefficients,
        directions,
        strict=strict,
    )
    return Equation(jacobian_form, residual_form)


__all__ = [
    "GateauxDerivativeError",
    "UnsupportedGateauxNodeError",
    "NonDifferentiableGateauxNodeError",
    "gateaux_derivative",
    "linearize_form",
    "linearize_equation",
]
