from __future__ import annotations

from pycutfem.ufl.expressions import ElementWiseConstant, Expression


class CellStateCoefficient(ElementWiseConstant):
    """
    UFL-visible per-cell coefficient backed by a mutable state field.

    The wrapped state array is intentionally mutated in-place so compiled
    kernels can refresh values without changing the expression structure.
    """

    def __init__(self, field, *, jit_name: str | None = None):
        self.field = field
        super().__init__(field.values)
        self._jit_name = str(jit_name or field.name)
        self._preserve_runtime_structure = True
        # State coefficients are runtime data; keep their cache token tied to
        # structure/name rather than to the current numeric values.
        self._cache_token = f"cell_state:{self._jit_name}"

    def __repr__(self) -> str:
        shape = "scalar" if self.tensor_shape == () else f"tensor{self.tensor_shape}"
        return f"CellStateCoefficient(name={self._jit_name!r}, {shape})"


class QuadratureStateCoefficient(Expression):
    """
    UFL-visible per-quadrature-point coefficient backed by a mutable state field.

    Values are indexed by `(entity_id, q)` during assembly. The numerical array
    is runtime-mutable and must not affect kernel cache identity.
    """

    _is_quadrature_state_coefficient = True

    def __init__(self, field, *, jit_name: str | None = None, values=None, tensor_shape=None):
        self.field = field
        self.values = field.values if values is None else values
        self.layout = field.layout
        self.tensor_shape = tuple(field.tensor_shape if tensor_shape is None else tensor_shape)
        self._jit_name = str(jit_name or field.name)
        self._preserve_runtime_structure = True
        self._cache_token = (
            f"quadrature_state:{self._jit_name}:{self.layout.signature}:{self.tensor_shape}"
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return self.tensor_shape

    @property
    def cache_token(self) -> str:
        return self._cache_token

    def value_on_entity_qp(self, entity_id: int, q: int):
        return self.values[int(entity_id), int(q)]

    def slice_component(self, idx):
        index = tuple(idx) if isinstance(idx, tuple) else (int(idx),)
        return QuadratureStateCoefficient(
            self.field,
            jit_name=self._jit_name,
            values=self.values[(slice(None), slice(None)) + index],
            tensor_shape=self.tensor_shape[len(index):],
        )

    def __repr__(self) -> str:
        shape = "scalar" if self.tensor_shape == () else f"tensor{self.tensor_shape}"
        return f"QuadratureStateCoefficient(name={self._jit_name!r}, {shape})"
