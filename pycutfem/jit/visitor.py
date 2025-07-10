# pycutfem/jit/visitor.py
from pycutfem.ufl.expressions import (
    Expression, Constant, TestFunction, TrialFunction, Function,
    VectorTestFunction, VectorTrialFunction, VectorFunction,
    Sum, Sub, Prod, Div as UflDiv, Inner as UflInner, Dot as UflDot, 
    Grad as UflGrad, DivOperation, Derivative, FacetNormal, Jump, Pos, Neg,
    ElementWiseConstant
)
from pycutfem.ufl.analytic import Analytic
from pycutfem.jit.ir import (
    LoadVariable, LoadConstant, LoadElementWiseConstant, LoadAnalytic,
    LoadFacetNormal, Grad, Div, PosOp, NegOp, BinaryOp, Inner, Dot, Store,
    LoadConstantArray
)

import logging
logger = logging.getLogger(__name__)



def _find_form_type(node: Expression) -> str:
    """
    Analyzes the expression to determine if it's for a matrix (LHS) or
    a vector (RHS).
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
        
        if form_type == 'functional':
            raise NotImplementedError("JIT for scalar functionals is not yet implemented.")

        self._visit(node)

        dest_name = "Ke" if form_type == 'matrix' else "Fe"
        self.ir_sequence.append(Store(dest_name=dest_name, store_type=form_type))

        return self.ir_sequence

    def _visit(self, node: Expression):
        """
        Recursive post-order traversal to generate IR.
        This function dispatches to the correct visitor based on node type.
        """
        # --- Leaf Nodes & Special Variables ---
        if isinstance(node, (TestFunction)):
            is_vec = False
            name = node.space.name if hasattr(node, 'space') else node.field_name
            self.ir_sequence.append(LoadVariable(name=name, role='test', is_vector=is_vec, field_names=[node.field_name]))
            return
            
        if isinstance(node, (TrialFunction)):
            is_vec = False
            name = node.space.name if hasattr(node, 'space') else node.field_name
            self.ir_sequence.append(LoadVariable(name=name, role='trial', is_vector=is_vec, field_names= [node.field_name]))
            return
        if isinstance(node, (Function)):
            is_vec = False
            role = 'function'
            name = node.space.name if hasattr(node, 'space') else node.name
            # Pass the component field names into the IR node
            self.ir_sequence.append(LoadVariable(name=name, role=role, is_vector=is_vec, field_names=[node.field_name]))
            return
        if isinstance(node, (VectorFunction)):
            is_vec = True
            role = 'function'
            name = node.space.name if hasattr(node, 'space') else node.name
            # Pass the component field names into the IR node
            self.ir_sequence.append(LoadVariable(name=name, role=role, is_vector=is_vec, field_names=node.field_names))
            return

        if isinstance(node, (VectorTestFunction, VectorTrialFunction)):
            is_vec = True
            role = 'test' if node.is_test else 'trial' 
            name = node.space.name if hasattr(node, 'space') else node.name
            # Pass the component field names into the IR node
            self.ir_sequence.append(LoadVariable(name=name, role=role, is_vector=is_vec, field_names=node.field_names))
            return
            
        if isinstance(node, Constant):
            if node.dim == 0:
                self.ir_sequence.append(LoadConstant(value=float(node.value)))
            else:
                # For array constants, we treat them as runtime arguments
                # identified by their object ID to ensure uniqueness.
                name = f"const_arr_{id(node)}"
                self.ir_sequence.append(LoadConstantArray(name=name, shape=node.shape))
            return

        if isinstance(node, FacetNormal):
            self.ir_sequence.append(LoadFacetNormal())
            return
        
        elif isinstance(node, ElementWiseConstant):
            name = f"ewc_{id(node)}"
            self.ir_sequence.append(
                LoadElementWiseConstant(name=name,
                                        tensor_shape=node.tensor_shape)   # NEW
            )
            return


        # --- Unary Operators ---
        if isinstance(node, UflGrad):
            self._visit(node.operand)
            self.ir_sequence.append(Grad())
            return
            
        if isinstance(node, DivOperation):
            self._visit(node.operand)
            self.ir_sequence.append(Grad()) 
            self.ir_sequence.append(Div())
            return
        
        if isinstance(node, Pos):
            self._visit(node.operand)
            self.ir_sequence.append(PosOp())
            return

        if isinstance(node, Neg):
            self._visit(node.operand)
            self.ir_sequence.append(NegOp())
            return
        
        if isinstance(node, Derivative):
            operand = node.f
            deriv_order = node.order
            
            is_vec = isinstance(operand, (VectorTestFunction, VectorTrialFunction, VectorFunction))
            role = 'test' if operand.is_test else 'trial' if operand.is_trial else 'function'
            name = operand.space.name if hasattr(operand, 'space') else operand.name if hasattr(operand, 'name') else operand.field_name
            
            self.ir_sequence.append(LoadVariable(name=name, role=role, is_vector=is_vec, deriv_order=deriv_order))
            return

        # --- Binary Operators ---
        if isinstance(node, (Sum, Sub, Prod, UflDiv)):
            self._visit(node.a)
            if isinstance(node, UflDiv):
                # We handle division by multiplying by the reciprocal.
                # This requires visiting a new Prod(Constant(1.0), node.b) tree.
                # For simplicity, we assume b is a Constant for now.
                if not isinstance(node.b, Constant):
                    raise NotImplementedError("JIT compilation for division by non-constants is not supported.")
                self._visit(Constant(1.0 / node.b.value))
                op_symbol = '*'
            else:
                self._visit(node.b)
                op_symbol = '+' if isinstance(node, Sum) else '-' if isinstance(node, Sub) else '*'
            self.ir_sequence.append(BinaryOp(op_symbol=op_symbol))
            return
            
        if isinstance(node, UflInner):
            self._visit(node.a)
            self._visit(node.b)
            self.ir_sequence.append(Inner())
            return
            
        if isinstance(node, UflDot):
            self._visit(node.a)
            self._visit(node.b)
            self.ir_sequence.append(Dot())
            return

        if isinstance(node, Jump):
            self._visit(node.u_pos)
            self._visit(node.u_neg)
            self.ir_sequence.append(BinaryOp(op_symbol='-'))
            return
        elif isinstance(node, Analytic):
            uid = f"ana_{id(node)}"
            self.ir_sequence.append(LoadAnalytic(func_id=id(node), func_ref=node.eval))
            return

        raise TypeError(f"UFL expression node '{type(node).__name__}' is not supported by the JIT compiler.")