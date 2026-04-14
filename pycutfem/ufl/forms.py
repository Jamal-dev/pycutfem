from dataclasses import dataclass
from pycutfem.ufl.expressions import Expression, Integral, Sum, Sub, Prod, Constant
from typing import Callable, List, Dict, Sequence, Any
import numbers

class Form(Expression):
    """Represents the sum of several integrals that make up one side of a weak form."""
    def __init__(self, integrals: list):
        self.integrals = []
        for term in integrals:
            if isinstance(term, Form): self.integrals.extend(term.integrals)
            elif isinstance(term, Integral): self.integrals.append(term)
            else:
                raise TypeError(f"A Form can only be constructed from Integral or other Form objects, not {type(term)}")

    def __add__(self, other):
        if isinstance(other, Integral):
            return Form(self.integrals + [other])
        if isinstance(other, Form):
            return Form(self.integrals + other.integrals)
        raise TypeError(f"Can only add an Integral or Form to a Form, not {type(other)}")

    def __repr__(self):
        if not self.integrals:
            return "Form()"
        pieces = []
        for i, integral in enumerate(self.integrals, start=1):
            pieces.append(f"  [{i}] {integral!r}")
        return "Form(\n" + ",\n".join(pieces) + "\n)"

    __str__ = __repr__

    def __sub__(self, other):
        # Use the __neg__ method to create negated versions of the terms to be subtracted.
        if isinstance(other, (Integral, Form)):
             return self.__add__(other.__neg__())
        raise TypeError("Can only subtract an Integral or Form from a Form.")

    def __neg__(self):
        # Create a new Form where each integral's integrand is negated.
        return Form([integral.__neg__() for integral in self.integrals])

    def __mul__(self, other):
        """
        Scale a form by a scalar factor by scaling each integral's integrand.

        Notes
        -----
        - Prefer `form * Constant(c)` over `Constant(c) * form` since the left
          operand's `__mul__` may eagerly build a generic expression tree.
        """
        if isinstance(other, (Integral, Form)):
            raise TypeError("Multiplying two forms/integrals is not supported.")
        if isinstance(other, numbers.Real):
            other = Constant(float(other))
        if not isinstance(other, Expression):
            return NotImplemented
        return Form([Integral(other * I.integrand, I.measure) for I in self.integrals])

    def __rmul__(self, other):
        if isinstance(other, numbers.Real):
            return self.__mul__(other)
        if isinstance(other, Expression):
            # See note in __mul__: Constant * Form will usually not reach here.
            return self.__mul__(other)
        return NotImplemented

    def __eq__(self, other):
        """Handles cases like `my_form == None`."""
        return Equation(self, other)

    def __req__(self, other):
        """
        Handles reverse-equals for cases like `None == my_form`. [THE FIX]
        
        This method is called by Python when `other == self` is evaluated and
        `other` does not have a specific `__eq__` method for a Form. Here,
        `other` is the left-hand side (e.g., None) and `self` is the Form object.
        """
        return Equation(other, self)

class Equation:
    def _valid_form(self, a):
        # Accept: None, Form, Integral. Treat numeric zero as "empty".
        if a is None:
            return None
        if isinstance(a, Form):
            return a
        if isinstance(a, Integral):
            return Form([a])
        # Constant is an Expression *and* a numbers.Number; handle it before numbers.Real
        if isinstance(a, Constant):
            if float(a) == 0.0:
                return None
            raise TypeError(
                "Bare Constant on an Equation side is ambiguous. "
                "Multiply by a Measure to make an Integral (e.g. Constant(c)*dx) "
                "or wrap it into your form appropriately."
            )
        # Plain numerics: only allow exact zero to mean 'no form'
        if isinstance(a, numbers.Real):
            if float(a) == 0.0:
                return None
            raise TypeError(
                "Numeric nonzero on an Equation side is not a Form. "
                "Wrap it with Constant(...) and a Measure (e.g. Constant(c)*dx)."
            )
        raise TypeError(
            f"Equation sides must be Form, Integral, Constant/0.0, or None, not {type(a)}"
        )
    def __init__(self, a:Form, L:Form):
        # Allow a side to be None, otherwise ensure it's a Form object.
        self.a = self._valid_form(a)
        self.L = self._valid_form(L)


@dataclass(frozen=True)
class CondensedQuadratureLocalSystem:
    """
    Compiler-level description of a local system with quadrature-local hidden
    state eliminated through a Schur complement.

    The base FE system is assembled from `base_form_or_equation`. The hidden
    state is local to quadrature points and described by:

    - `coupling_left[a]`   : local FE row block C_a(x_q)
    - `coupling_right[a]`  : local FE column block B_a(x_q)
    - `hidden_jacobian[a][b]` : local hidden-state Jacobian G_ab(x_q)
    - `hidden_residual[a]` : local hidden-state residual r_a(x_q)

    The compiler assembles the condensed local correction

        K_hat = K_base + sign * sum_q C^T G^{-1} B
        F_hat = F_base + sign * sum_q C^T G^{-1} r

    where `sign=-1.0` recovers the usual Schur-complement elimination.

    Notes
    -----
    - The hidden-state dimension is given by `len(hidden_residual)`.
    - Each coupling entry must be a scalar UFL expression whose pointwise
      evaluation yields one local FE vector.
    - The hidden Jacobian/residual entries must be coefficient-only scalar UFL
      expressions evaluated at the supplied quadrature layout.
    """

    base_form_or_equation: Any
    coupling_left: Sequence[Expression]
    coupling_right: Sequence[Expression]
    hidden_jacobian: Sequence[Sequence[Expression]]
    hidden_residual: Sequence[Expression]
    quadrature_layout: Any
    sign: float = -1.0

    def __post_init__(self) -> None:
        n_hidden = int(len(tuple(self.hidden_residual)))
        if n_hidden <= 0:
            raise ValueError("CondensedQuadratureLocalSystem requires at least one hidden-state residual entry.")
        if int(len(tuple(self.coupling_left))) != n_hidden:
            raise ValueError("coupling_left must have the same length as hidden_residual.")
        if int(len(tuple(self.coupling_right))) != n_hidden:
            raise ValueError("coupling_right must have the same length as hidden_residual.")
        rows = tuple(tuple(row) for row in self.hidden_jacobian)
        if int(len(rows)) != n_hidden:
            raise ValueError("hidden_jacobian must be square with side length len(hidden_residual).")
        for row in rows:
            if int(len(row)) != n_hidden:
                raise ValueError("hidden_jacobian must be square with side length len(hidden_residual).")

class BoundaryCondition:
    def __init__(self, field: str, method: str, domain_tag: str, value: Callable):
        self.field = field
        m = method.lower()
        if m not in ("dirichlet", "neumann"):
            raise ValueError("BC method must be 'dirichlet' or 'neumann'")
        self.method = m
        self.domain_tag = domain_tag
        self.value = value

def assemble_form(equation: Equation, dof_handler, bcs=[], quad_order=None, 
                  assembler_hooks=None,backend='jit', **kwargs):
    """
    High-level function to assemble a weak form into a matrix and vector.
    """
    if not isinstance(equation, Equation):
        raise Warning(
            "assemble_form expects a pycutfem.ufl.forms.Equation; "
            "did you accidentally write None == form? This will give just 0s"
        )
    from pycutfem.ufl.compilers import FormCompiler
    if kwargs.get('quad_degree') is not None:
        quad_order = kwargs['quad_degree']
    
    # We no longer need to preprocess the form.
    # The compiler will handle the list of integrals directly.
    compiler = FormCompiler(dof_handler, quad_order, assembler_hooks=assembler_hooks, backend=backend)
    
    # This runs the full assembly process. K and F are created, and if hooks
    # are present, compiler.ctx['scalar_results'] is populated.
    K, F = compiler.assemble(equation, bcs)

    # After assembly, check the compiler's context for scalar results.
    # If the user provided hooks and those hooks produced results, return them.
    if assembler_hooks and 'scalar_results' in compiler.ctx:
        return compiler.ctx['scalar_results']
    
    # Otherwise, return the standard system matrix and vector.
    return K, F
