from pycutfem.ufl import Constant
from pycutfem.ufl.expressions import Expression, Restriction
from pycutfem.ufl.helpers import _find_all_restrictions
from pycutfem.utils.bitset import BitSet


class _DummyScalar(Expression):
    num_components = 1


def test_find_all_restrictions_handles_deep_expression_iteratively() -> None:
    domain = BitSet([True, False, True])
    expr = Restriction(_DummyScalar(), domain)
    for _ in range(2000):
        expr = expr + Constant(0.0)

    restrictions = _find_all_restrictions(expr)

    assert len(restrictions) == 1
    assert restrictions[0].domain is domain
