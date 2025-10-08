# pycutfem/jit/visitor.py
from pycutfem.ufl.expressions import (
    Expression, Constant, TestFunction, TrialFunction, Function,
    VectorTestFunction, VectorTrialFunction, VectorFunction,
    Sum, Sub, Prod, Div as UflDiv, Inner as UflInner, Dot as UflDot, 
    Grad as UflGrad, DivOperation, Derivative, FacetNormal, Jump, Pos, Neg,
    ElementWiseConstant, Transpose as UFLTranspose, CellDiameter as UFLCellDiameter,
    NormalComponent, Restriction, Power as UFLPower, Trace as UFLTrace,
    Hessian as UFLHessian, Laplacian as UFLLaplacian
)
from pycutfem.ufl.analytic import Analytic
from pycutfem.jit.ir import (
    LoadVariable, LoadConstant, LoadConstantArray, LoadElementWiseConstant as LoadEWC_IR,
    LoadAnalytic, LoadFacetNormal, Grad, Div, BinaryOp, Inner, Dot, Store, Transpose,
    CellDiameter, LoadFacetNormalComponent, CheckDomain, Trace,
    Hessian as IRHessian, Laplacian as IRLaplacian
)
from dataclasses import replace
import logging

logger = logging.getLogger(__name__)

def _find_form_type(node: Expression) -> str:
    """
    Analyzes the expression to determine if it's for a matrix (LHS),
    a vector (RHS), or a scalar functional.
    """
    has_trial = node.find_first(lambda n: getattr(n, 'is_trial', False))
    has_test = node.find_first(lambda n: getattr(n, 'is_test', False))

    if has_trial and has_test:
        return 'matrix'
    if has_test:
        return 'vector'
    return 'functional'

class IRGenerator:
    """
    Walks a UFL expression tree and generates a linear Intermediate
    Representation (IR) in Reverse Polish Notation.
    """
    def __init__(self):
        self.ir_sequence = []

    def generate(self, node: Expression) -> list:
        """
        Public method to generate the IR for a given UFL expression.
        """
        self.ir_sequence.clear()
        form_type = _find_form_type(node)
        
        self._visit(node) # Initial call with default side=""

        if form_type == 'functional':
            self.ir_sequence.append(Store(dest_name="Je", store_type="functional"))
        else:
            dest_name = "Ke" if form_type == 'matrix' else "Fe"
            self.ir_sequence.append(Store(dest_name=dest_name, store_type=form_type))

        return self.ir_sequence

    def _visit(self, node: Expression, side: str = ""):
        """
        Recursive post-order traversal to generate IR. The `side` parameter
        is crucial for correctly handling Jump/Pos/Neg expressions.
        This version is reordered to handle operators before leaf nodes.
        """
        # --- 1. Unary Operators that change context ---
        if isinstance(node, Pos):
            self._visit(node.operand, side="+")
            return

        if isinstance(node, Neg):
            self._visit(node.operand, side="-")
            return

        # --- 2. Binary Operators ---
        if isinstance(node, Jump):
            # Force side on both operands. If they’re already Pos/Neg wrappers,
            # just visit them; otherwise visit with explicit side.
            def _visit_with_side(expr, sgn):
                from pycutfem.ufl.expressions import Pos, Neg
                if isinstance(expr, (Pos, Neg)):
                    self._visit(expr)
                else:
                    self._visit(expr, side=sgn)
            _visit_with_side(node.u_pos, "+")   # u(+)
            _visit_with_side(node.u_neg, "-")   # u(-)
            self.ir_sequence.append(BinaryOp(op_symbol='-'))
            return

        if isinstance(node, Restriction):
            # Visit the operand first (post-order traversal)
            self._visit(node.operand, side=side)
            # Then, add the instruction to check the domain tag.
            # The code generator will use this to conditionally zero the result.
            self.ir_sequence.append(CheckDomain(bitset_id=id(node.domain)))
            return
        if isinstance(node, UFLTrace):
            # First, visit the operand of the trace. This will execute all
            # operations inside tr() and leave a tensor on the stack.
            self._visit(node.A, side=side)
            # Then, append the instruction to take the trace of that tensor.
            self.ir_sequence.append(Trace())
            return
        
        if isinstance(node, (Sum, Sub, Prod, UflDiv, UFLPower)):
            self._visit(node.a, side=side)
            self._visit(node.b, side=side)
            if isinstance(node, UflDiv):
                op_symbol = '/'
            else:
                # op_symbol = {Sum: '+', Sub: '-', Prod: '*'}.get(type(node), '/')
                op_symbol = {Sum: '+', Sub: '-', Prod: '*', UFLPower: '**'}.get(type(node), '/')
            self.ir_sequence.append(BinaryOp(op_symbol=op_symbol))
            return
            
        if isinstance(node, UFLCellDiameter):           # ==> your Expression
            self.ir_sequence.append(CellDiameter())     # push scalar at runtime
            return
        if isinstance(node, (UflInner, UflDot)):
            self._visit(node.a, side=side)
            self._visit(node.b, side=side)
            self.ir_sequence.append(Inner() if isinstance(node, UflInner) else Dot())
            return

        # --- 3. Unary Operators that modify their operand ---
        if isinstance(node, UflGrad):
            op = node.operand
            if isinstance(op, Jump):
                # grad(jump(u)) = grad(u(+)) - grad(u(-))
                self._visit(UflGrad(op.u_pos), side='+')
                self._visit(UflGrad(op.u_neg), side='-')
                self.ir_sequence.append(BinaryOp(op_symbol='-'))
                return
            self._visit(op, side=side)
            self.ir_sequence.append(Grad())
            return
        
        if isinstance(node, UFLHessian):
            op = node.operand
            if isinstance(op, Jump):
                self._visit(UFLHessian(op.u_pos), side="+")
                self._visit(UFLHessian(op.u_neg), side="-")
                self.ir_sequence.append(BinaryOp(op_symbol='-'))
                return
            self._visit(op, side=side)
            self.ir_sequence.append(IRHessian())
            return

        if isinstance(node, UFLLaplacian):
            op = node.operand
            if isinstance(op, Jump):
                self._visit(UFLLaplacian(op.u_pos), side="+")
                self._visit(UFLLaplacian(op.u_neg), side="-")
                self.ir_sequence.append(BinaryOp(op_symbol='-'))
                return
            self._visit(op, side=side)
            self.ir_sequence.append(IRLaplacian())
            return
            
        if isinstance(node, DivOperation):
            op = node.operand
            if isinstance(op, Jump):
                # Special case: Derivative of a Jump(u) is D(u(+)) - D(u(-)).
                # This correctly dispatches to the Derivative visitor again.
                self._visit(DivOperation(op.u_pos), side='+')
                self._visit(DivOperation(op.u_neg), side='-')
                self.ir_sequence.append(BinaryOp(op_symbol='-'))
                return
            self._visit(node.operand, side=side)
            self.ir_sequence.append(Grad()) 
            self.ir_sequence.append(Div())
            return
        
        if isinstance(node, Derivative):
            operand = node.f
            while isinstance(operand, Restriction):
                operand = operand.operand
            deriv_order = node.order

            if isinstance(operand, Pos):
                self._visit(Derivative(operand.operand, *deriv_order), side='+')
                return
            if isinstance(operand, Neg):
                self._visit(Derivative(operand.operand, *deriv_order), side='-')
                return

            # --- SPECIAL CASE: Derivative of a Jump ---
            # This is the key to handling [[D(u)]] = D(u_pos) - D(u_neg).
            if isinstance(operand, Jump):
                # We create two new temporary UFL nodes and visit them.
                # This correctly dispatches to this Derivative visitor again,
                # but with Pos(u) and Neg(u) as the new operands.
                self._visit(Derivative(operand.u_pos, *deriv_order),side='+')
                self._visit(Derivative(operand.u_neg, *deriv_order),side='-')
                self.ir_sequence.append(BinaryOp(op_symbol='-'))
                return

            # --- GENERIC CASE: Derivative of a standard field ---
            # At this point, the operand MUST be a leaf-like node (Function, etc.)
            # because all operators that could wrap it have been handled above.
            is_vec = isinstance(operand, (VectorTestFunction, VectorTrialFunction, VectorFunction))
            role = 'test' if getattr(operand, 'is_test', False) else 'trial' if getattr(operand, 'is_trial', False) else 'function'
            # Safely get field_names or field_name
            field_names = getattr(operand, 'field_names', None) or [getattr(operand, 'field_name')]
            name = getattr(getattr(operand, 'space', operand), 'name', 'anon')


            self.ir_sequence.append(
                LoadVariable(name=name, role=role, is_vector=is_vec,
                             deriv_order=deriv_order, field_names=field_names,
                             side=side,
                             field_sides=getattr(operand, "field_sides", None))
            )
            return

        # --- 4. Leaf Nodes ---
        while isinstance(node, Restriction):
            node = node.operand

        if isinstance(node, (TestFunction, TrialFunction, Function, VectorTestFunction, VectorTrialFunction, VectorFunction)):
            is_vec = isinstance(node, (VectorTestFunction, VectorTrialFunction, VectorFunction))
            role = 'test' if getattr(node, 'is_test', False) else 'trial' if getattr(node, 'is_trial', False) else 'function'
            field_names = getattr(node, 'field_names', None) or [getattr(node, 'field_name')]
            name = getattr(getattr(node, 'space', node), 'name', 'anon')
            field_sides = getattr(node, "field_sides", None)
            self.ir_sequence.append(LoadVariable(name=name, role=role, is_vector=is_vec, field_names=field_names, side=side, field_sides=field_sides))
            return
            
        if isinstance(node, Constant):
            if node.dim == 0:
                self.ir_sequence.append(LoadConstant(value=float(node.value)))
            else:
                name = f"const_arr_{id(node)}"
                self.ir_sequence.append(LoadConstantArray(name=name, shape=node.shape))
            return
        if isinstance(node, UFLTranspose):
            # Transpose is a no-op in IR, just push the top-of-stack tensor.
            self._visit(node.A, side=side)
            self.ir_sequence.append(Transpose())
            return

        if isinstance(node, FacetNormal):
            self.ir_sequence.append(LoadFacetNormal())
            return
        if isinstance(node, NormalComponent):
            self.ir_sequence.append(LoadFacetNormalComponent(idx=node.idx))
            return

        if isinstance(node, ElementWiseConstant):
            name = f"ewc_{id(node)}"
            # This was a bug, creating a UFL node instead of an IR node
            self.ir_sequence.append(LoadEWC_IR(name=name, tensor_shape=node.tensor_shape))
            return

        if isinstance(node, Analytic):
            self.ir_sequence.append(LoadAnalytic(func_id=id(node), func_ref=node.eval,
                                                 tensor_shape=getattr(node, "tensor_shape", ())))
            return

        # If we reach here, the node type is not supported.
        raise TypeError(f"UFL expression node '{type(node).__name__}' is not supported by the JIT compiler.")
