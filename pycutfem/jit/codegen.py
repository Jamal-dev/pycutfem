# pycutfem/jit/codegen.py
import textwrap
from dataclasses import dataclass, field

from matplotlib.pylab import f
from xarray import as_variable
from .ir import (
    LoadVariable, LoadConstant, LoadConstantArray, LoadElementWiseConstant,
    LoadAnalytic, LoadFacetNormal, Grad, Div, PosOp, NegOp,
    BinaryOp, Inner, Dot, Store
)
import numpy as np

# Numba is imported inside the generated kernel string
# import numba

@dataclass(frozen=True, slots=True)
class StackItem:
    """Holds metadata for an item on the code generation stack."""
    var_name: str
    # role can be 'test', 'trial', or 'value' (a computed quantity)
    role: str
    # shape of the data: e.g., () for scalar, (k,) for vector value,
    # (n,) for scalar basis, (k, n) for vector basis,
    # (n, d) for scalar grad basis, (k, n, d) for vector grad basis
    shape: tuple
    is_gradient: bool = field(default=False)
    is_vector: bool = field(default=False)
    # Stores the field name(s) to look up basis functions or coefficients
    field_names: list = field(default_factory=list)
    # Stores the name of the parent Function/VectorFunction
    parent_name: str = ""


class NumbaCodeGen:
    """
    Translates a linear IR sequence into a Numba-Python kernel source string.
    """
    def __init__(self, nopython: bool = True, mixed_element=None):
        """
        Initializes the code generator.
        
        Args:
            nopython (bool): Whether to use nopython mode for Numba.
            mixed_element: An INSTANCE of the MixedElement class, not the class type.
        """
        self.nopython = nopython
        if mixed_element is None:
            raise ValueError("NumbaCodeGen requires an instance of a MixedElement.")
        self.me = mixed_element
        self.n_dofs_local = self.me.n_dofs_local
        self.spatial_dim = self.me.mesh.spatial_dim

    def generate_source(self, ir_sequence: list, kernel_name: str):
        body_lines = []
        stack = []
        var_counter = 0
        required_args = set()
        # Track names of Function/VectorFunction objects that provide coefficients
        solution_func_names = set()

        def new_var(prefix="tmp"):
            nonlocal var_counter
            var_counter += 1
            return f"{prefix}_{var_counter}"

        # --- Main IR processing loop ---
        for op in ir_sequence:
            # --- LOAD OPERATIONS ---
            if isinstance(op, LoadVariable):
                # This handles both simple variables and UFL Derivatives,
                # which the visitor translates to a LoadVariable with a deriv_order.
                operand = op
                deriv_order = op.deriv_order
                # This now correctly gets the field names from the IR
                field_names = operand.field_names

                # Determine the name of the basis/derivative array needed
                def get_arg_name(field_name):
                    if deriv_order == (0, 0): return f"b_{field_name}"
                    return f"d{deriv_order[0]}{deriv_order[1]}_{field_name}"

                basis_vars = [f"{get_arg_name(fname)}_q" for fname in field_names]
                for fname in field_names:
                    required_args.add(get_arg_name(fname))

                if operand.role in ('test', 'trial'):
                    if not operand.is_vector:
                        var_name = new_var("basis_reshaped")
                        body_lines.append(f"{var_name} = {basis_vars[0]}[np.newaxis, :]")
                        shape = (1, self.n_dofs_local)
                    else:
                        var_name = new_var("basis_stack")
                        body_lines.append(f"{var_name} = np.stack(({', '.join(basis_vars)}))")
                        shape = (len(field_names), self.n_dofs_local)
                    stack.append(StackItem(var_name=var_name, role=operand.role, shape=shape, is_vector=operand.is_vector, field_names=field_names, parent_name=operand.name))

                elif operand.role == 'function':
                    # For functions, the coefficients are associated with the main function name
                    solution_func_names.add(operand.name)
                    val_var = new_var(f"{operand.name}_val")
                    
                    if operand.is_vector:
                        # val_at_q = [dot(basis_ux, coeffs_u_k), dot(basis_uy, coeffs_u_k), ...]
                        comp_vals = [f"np.dot({bvar}, u_{operand.name}_loc)" for bvar in basis_vars]
                        body_lines.append(f"{val_var} = np.array([{', '.join(comp_vals)}])")
                        shape = (len(field_names),)
                    else: # scalar
                        body_lines.append(f"{val_var} = np.dot({basis_vars[0]}, u_{operand.name}_loc)")
                        shape = ()
                    stack.append(StackItem(var_name=val_var, role='value', shape=shape, is_vector=operand.is_vector, field_names=field_names, parent_name=operand.name))
                else:
                    raise TypeError(f"Unknown role '{operand.role}' for LoadVariable/Derivative")

            elif isinstance(op, LoadConstant):
                stack.append(StackItem(var_name=str(op.value), role='value', shape=(), is_vector=False, field_names=[]))
            
            elif isinstance(op, LoadConstantArray):
                required_args.add(op.name)
                # The constant array is passed in as a list. Convert it to a
                # NumPy array inside the kernel for Numba compatibility.
                np_array_var = new_var("const_np_arr")
                body_lines.append(f"{np_array_var} = np.array({op.name}, dtype=np.float64)")
                stack.append(StackItem(var_name=np_array_var, role='value', shape=(-1,), is_vector=True, field_names=[]))


            # --- UNARY OPERATORS ---
            elif isinstance(op, Grad):
                a = stack.pop()
                grad_var = new_var("grad")
                
                def get_grad_arg_name(field_name): return f"g_{field_name}"
                
                grad_basis_vars = [f"{get_grad_arg_name(fname)}_q" for fname in a.field_names]
                for fname in a.field_names: required_args.add(get_grad_arg_name(fname))

                if a.role in ('test', 'trial'):
                    phys_grad_vars = []
                    for i, fname in enumerate(a.field_names):
                        phys_grad_var = new_var(f"phys_grad_{fname}")
                        body_lines.append(f"{phys_grad_var} = {grad_basis_vars[i]} @ J_inv_q.transpose()")
                        phys_grad_vars.append(phys_grad_var)
                    
                    if not a.is_vector:
                        var_name = new_var("grad_reshaped")
                        body_lines.append(f"{var_name} = {phys_grad_vars[0]}[np.newaxis, :, :]")
                        shape = (1, self.n_dofs_local, self.spatial_dim)
                    else:
                        var_name = new_var("grad_stack")
                        body_lines.append(f"{var_name} = np.stack(({', '.join(phys_grad_vars)}))")
                        shape = (len(a.field_names), self.n_dofs_local, self.spatial_dim)
                    stack.append(StackItem(var_name=var_name, role=a.role, shape=shape, is_gradient=True, is_vector=False, field_names=a.field_names, parent_name=a.parent_name))

                elif a.role == 'value': # grad(function)
                    # grad(f)(x_q) = sum_i u_i * grad(phi_i)(x_q)
                    grad_val_comps = []
                    for i, fname in enumerate(a.field_names):
                        phys_grad_basis = new_var(f"phys_grad_basis_{fname}")
                        body_lines.append(f"{phys_grad_basis} = {grad_basis_vars[i]} @ J_inv_q.transpose()")
                        grad_val_comp = new_var(f"grad_val_{fname}")
                        # Use the parent name to get the correct coefficient array
                        body_lines.append(f"{grad_val_comp} = {phys_grad_basis}.T @ u_{a.parent_name}_loc")
                        grad_val_comps.append(grad_val_comp)
                    
                    if not a.is_vector:
                        var_name = grad_val_comps[0]
                        shape = (self.spatial_dim,)
                    else:
                        var_name = new_var("grad_val_stack")
                        body_lines.append(f"{var_name} = np.stack(({', '.join(grad_val_comps)}))")
                        shape = (len(a.field_names), self.spatial_dim)
                    stack.append(StackItem(var_name=var_name, role='value', shape=shape, is_gradient=True, is_vector=False, field_names=a.field_names, parent_name=a.parent_name))
                else:
                    raise NotImplementedError(f"Grad not implemented for role {a.role}")
            
            elif isinstance(op, Div):
                a = stack.pop()

                if not a.is_gradient:
                    raise TypeError("Div can only be applied to a gradient quantity.")

                div_var = new_var("div")

                # ---------------------------------------------------------------
                # 1)  Divergence of basis gradients (Test / Trial)  →  scalar (1,n)
                #     a.var_name shape: (k , n_loc , d)
                # ---------------------------------------------------------------
                if a.role in ("test", "trial") and a.shape[0] ==2:
                    body_lines.append("# Div(basis) → scalar basis (1,n_loc)")

                    body_lines += [
                        f"n_loc  = {a.var_name}.shape[1]",
                        f"n_vec  = {a.var_name}.shape[0]",     # components k
                        f"{div_var} = np.zeros((1, n_loc), dtype=np.float64)",
                        f"for j in range(n_loc):",            # local basis index
                        f"    tmp = 0.0",
                        f"    for k in range(n_vec):",        # component index
                        f"        tmp += {a.var_name}[k, j, k]",   #  ∂φ_k / ∂x_k
                        f"    {div_var}[0, j] = tmp",
                    ]

                    stack.append(
                        StackItem(
                            var_name=div_var,
                            role=a.role,
                            shape=(1, a.shape[1]),            # (1 , n_loc)
                            is_vector=False,
                            field_names=a.field_names,
                            parent_name=a.parent_name,
                        )
                    )

                # ---------------------------------------------------------------
                # 2)  Divergence of a gradient VALUE  (Function / VectorFunction)
                #     a.var_name shape: (k , d)
                # ---------------------------------------------------------------
                elif a.role == "value":
                    body_lines.append("# Div(value)  – vector → scalar   or   tensor → vector")

                    body_lines += [
                        f"n_comp = {a.var_name}.shape[0]",    # number of components  (k)
                        f"n_dim  = {a.var_name}.shape[1]",    # spatial dimension     (d)",

                        # ---------------- vector field: k == d  → scalar
                        f"if n_comp == n_dim:",
                        f"    {div_var} = 0.0",
                        f"    for k in range(n_dim):",
                        f"        {div_var} += {a.var_name}[k, k]",

                        # ---------------- tensor field: k != d  → vector (k,)
                        f"else:",
                        f"    {div_var} = np.zeros((n_comp,), dtype=np.float64)",
                        f"    for k in range(n_comp):",
                        f"        tmp = 0.0",
                        f"        for d in range(n_dim):",
                        f"            tmp += {a.var_name}[k, d]",
                        f"        {div_var}[k] = tmp",
                    ]

                    # meta-data for the StackItem
                    body_lines.append(f"_is_vec = n_comp != n_dim")
                    body_lines.append(f"_shape  = () if not _is_vec else (n_comp,)")

                    stack.append(
                        StackItem(
                            var_name    = div_var,
                            role        = "value",
                            shape       = () if a.shape[0] == a.shape[1] else (a.shape[1],),
                            is_vector   = a.shape[1] != a.shape[0],
                            is_gradient = False,
                            field_names = a.field_names,
                        )
                    )

                # ---------------------------------------------------------------
                # 3)  Anything else is not defined
                # ---------------------------------------------------------------
                else:
                    raise NotImplementedError(
                        f"Div not implemented for role '{a.role}' with gradient=True."
                    )

            elif isinstance(op, PosOp):
                a = stack.pop()
                res_var = new_var("pos")
                body_lines.append(f"{res_var} = {a.var_name} if phi_q >= 0.0 else np.zeros_like({a.var_name})")
                stack.append(a._replace(var_name=res_var))

            elif isinstance(op, NegOp):
                a = stack.pop()
                res_var = new_var("neg")
                body_lines.append(f"{res_var} = {a.var_name} if phi_q < 0.0 else np.zeros_like({a.var_name})")
                stack.append(a._replace(var_name=res_var))

            # --- BINARY OPERATORS ---
            elif isinstance(op, Inner):
                b = stack.pop(); a = stack.pop()
                res_var = new_var("inner")
                if a.role in ('test', 'trial') and b.role in ('test', 'trial'): # LHS
                    if a.is_gradient and b.is_gradient:
                        body_lines.append(f'# Inner(Grad, Grad): stiffness matrix')
                        # This is equivalent to einsum("knd,kmd->nm", a, b)
                        body_lines.append(f'{res_var} = np.zeros(({a.shape[1]}, {b.shape[1]}))')
                        body_lines.append(f'for k in range({a.shape[0]}):')
                        # Use .copy() to ensure contiguous arrays and avoid performance warnings
                        body_lines.append(f'    a_k = {a.var_name}[k].copy()')
                        body_lines.append(f'    b_k = {b.var_name}[k].copy()')
                        body_lines.append(f'    {res_var} += a_k @ b_k.T')
                    else:
                        body_lines.append(f'# Inner(Vec, Vec): mass matrix')
                        body_lines.append(f'{res_var} = {a.var_name}.T @ {b.var_name}')
                elif a.role == 'value' and b.role == 'test': # RHS
                    body_lines.append(f'# RHS: Inner(Function, Test)')
                    # a is (k,d) , b is (k,n,d), 
                    if a.is_gradient and b.is_gradient:
                        # a is (k,d) and b is  (k,n,d)  --> (n)
                        body_lines.append(f'# RHS: Inner(Grad(Function), Grad(Test))')
                        body_lines += [
                            f'n_locs = {b.shape[1]}; n_vec_comps = {a.shape[0]};',
                            # f'print(f"a.shape: {{{a.var_name}.shape}}, b.shape: {{{b.var_name}.shape}}")',
                            f'{res_var} = np.zeros((n_locs))',
                            f'for n in range(n_locs):',
                            f"    {res_var}[n] = np.sum({a.var_name} * {b.var_name}[:,n,:])"
                        ]
                    elif a.is_vector and  b.is_vector:
                        body_lines.append(f'# RHS: Inner(Function, Test)')
                        # a is (k), b is (k,n) -> (n,)
                        body_lines += [
                            f'n_locs = {b.shape[1]}; n_vec_comps = {b.shape[0]};',
                            f'{res_var} = np.zeros((n_locs))',
                            f'for n in range(n_locs):',
                            f'    {res_var}[n] = np.dot({a.var_name}, {b.var_name}[:,n])'
                        ]
                    else:
                        raise NotImplementedError(f"Inner not implemented for roles {a.role}/{b.role}")
                        
                    
                elif a.role == 'value' and b.role == 'value':
                    if a.is_vector and b.is_vector:
                        body_lines.append(f'# Inner(Value, Value): dot product')
                        body_lines.append(f'{res_var} = np.dot({a.var_name}, {b.var_name})')
                    if a.is_gradient and b.is_gradient:
                        body_lines += [
                            f'# Inner(Grad(Value), Grad(Value)): stiffness matrix',
                            f'# (k,d) @ (k,d) -> (k,k)',
                            f'{res_var} = {a.var_name} @ {b.var_name}.T',]
                
                else:
                    raise NotImplementedError(f"JIT Inner not implemented for roles {a.role}/{b.role}")
                stack.append(StackItem(var_name=res_var, role='value', shape=(), is_vector=False, field_names=[]))

            # ------------------------------------------------------------------
            # DOT   — special-cased branches for advection / mass terms --------
            # ------------------------------------------------------------------
            elif isinstance(op, Dot):
                b = stack.pop()
                a = stack.pop()
                res_var = new_var("dot")

                # Advection term: dot(grad(u_trial), u_k)
                if a.role == 'trial' and a.is_gradient and b.role == 'value' and b.is_vector:
                    body_lines.append(f"# Advection: dot(grad(Trial), Function)")
                    # body_lines.append(f"{res_var} = np.einsum('knd,d->kn', {a.var_name}, {b.var_name})")
                    body_lines += [
                        f"n_vec_comps = {a.var_name}.shape[0];n_locs = {a.var_name}.shape[1];n_spatial_dim = {a.var_name}.shape[2];",
                        f"{res_var} = np.zeros((n_vec_comps, n_locs), dtype=np.float64)",
                        f"for k in range(n_vec_comps):",
                        f"    {res_var}[k] = {b.var_name} @ {a.var_name}[k].T ",
                        # f"assert {res_var}.shape == (2, 22), f'result shape mismatch {res_var}.shape with {{(n_vec_comps, n_locs)}}'"
                    ]
                    stack.append(StackItem(var_name=res_var, role='trial', shape=(a.shape[0], a.shape[1]), is_vector=True, is_gradient=False, field_names=a.field_names, parent_name=a.parent_name))
               
                # Final advection term: dot(advection_vector_trial, v_test)
                elif a.role == 'trial' and a.is_vector and b.role == 'test' and b.is_vector:
                     body_lines.append(f"# Mass: dot(Trial, Test)")
                    #  body_lines.append(f"assert ({a.var_name}.shape == (2,22) and {b.var_name}.shape == (2,22)), 'Trial and Test to have the same shape'")
                     body_lines.append(f"{res_var} = {b.var_name}.T @ {a.var_name}")
                     stack.append(StackItem(var_name=res_var, role='value', shape=(), is_vector=False, field_names=[]))
                
                # ---------------------------------------------------------------------
                # dot( grad(u_k) ,  u_trial )  ← convection term (Function gradient · Trial)
                # ---------------------------------------------------------------------
                elif a.role == 'value' and a.is_gradient and b.role == 'trial' and b.is_vector:
                    body_lines.append("# Advection: dot(grad(Function), Trial)")
                    body_lines += [
                        f"{res_var}   = {a.var_name} @ {b.var_name}",
                        
                    ]
                    stack.append(StackItem(var_name=res_var, role='trial',
                                        shape=(b.shape[0], b.shape[1]), is_vector=True,
                                        is_gradient=False, field_names=b.field_names,
                                        parent_name=b.parent_name))

                # ---------------------------------------------------------------------
                # dot( u_trial ,  grad(u_k) )   ← transpose of the previous
                # ---------------------------------------------------------------------
                elif a.role == 'trial' and a.is_vector and b.role == 'value' and b.is_gradient:
                    body_lines.append("# Advection: dot(Trial, grad(Function))")
                    body_lines += [
                        f"{res_var} = {b.var_name} @ {a.var_name}",
                    ]
                    stack.append(StackItem(var_name=res_var, role='trial',
                                        shape=(a.shape[0], a.shape[1]), is_vector=True,
                                        is_gradient=False, field_names=a.field_names,
                                        parent_name=a.parent_name))

                # ---------------------------------------------------------------------
                # dot( u_k ,  u_k )             ← |u_k|², scalar
                # ---------------------------------------------------------------------
                elif a.role == 'value' and a.is_vector and b.role == 'value' and b.is_vector:
                    body_lines.append("# Non-linear term: dot(Function, Function)")
                    body_lines.append(f"{res_var} = np.dot({a.var_name}, {b.var_name})")
                    stack.append(StackItem(var_name=res_var, role='value',
                                        shape=(), is_vector=False))

                # ---------------------------------------------------------------------
                # dot( u_k ,  grad(u_trial) )   ← usually zero for skew-symm forms
                # ---------------------------------------------------------------------
                elif a.role == 'value' and a.is_vector and b.role == 'trial' and b.is_gradient:
                    body_lines.append("# dot(Function, grad(Trial))")
                    body_lines += [
                        f"n_vec_comps = {b.var_name}.shape[0];",
                        f"n_locs      = {b.var_name}.shape[1];",
                        f"{res_var}   = np.zeros((n_vec_comps, n_locs), dtype=np.float64)",
                        # einsum: f"{res_var} = np.einsum('d,kld->kl', {a.var_name}, {b.var_name})",
                        f"for k in range(n_vec_comps):",
                        f"    {res_var}[k] = {a.var_name} @ {b.var_name}[k].T",
                    ]
                    stack.append(StackItem(var_name=res_var, role='trial',
                                        shape=(b.shape[0], b.shape[1]), is_vector=True,
                                        is_gradient=False, field_names=b.field_names,
                                        parent_name=b.parent_name))

                # ---------------------------------------------------------------------
                # dot( u_test ,  u_trial )      ← mass-matrix block
                # ---------------------------------------------------------------------
                elif a.role == 'test' and a.is_vector and b.role == 'trial' and b.is_vector:
                    body_lines.append("# Mass: dot(Test, Trial)")
                    body_lines.append(f"{res_var} = {a.var_name}.T @ {b.var_name}")
                    stack.append(StackItem(var_name=res_var, role='value',
                                        shape=(), is_vector=False))

                
                # ---------------------------------------------------------------------
                # dot( scalar ,  u_trial;u_test;u_k )     ← e.g. scalar constant time function
                # ---------------------------------------------------------------------
                elif (a.role == 'const' or a.role == 'value') and not a.is_vector and not a.is_gradient:
                    # a is scalar, b is vector (trial/test/function)
                    body_lines.append("# Scalar constant: dot(scalar, Function/Trial/Test)")
                    body_lines.append(f"{res_var} = float({a.var_name}) * {b.var_name}")
                    stack.append(StackItem(var_name=res_var, role=b.role,
                                        shape=b.shape, is_vector=b.is_vector,
                                        is_gradient=b.is_gradient, field_names=b.field_names,
                                        parent_name=b.parent_name))
                # ---------------------------------------------------------------------
                # dot( u_trial;u_test;u_k, scalar )     ← e.g. scalar constant time function
                # ---------------------------------------------------------------------
                elif (b.role == 'const' or b.role == 'value') and not b.is_vector and not b.is_gradient:
                    # a is vector (trial/test/function), b is scalar
                    body_lines.append("# Scalar constant: dot(Function/Trial/Test, scalar)")
                    body_lines.append(f"{res_var} = float({b.var_name}) * {a.var_name}")
                    stack.append(StackItem(var_name=res_var, role=a.role,
                                        shape=a.shape, is_vector=a.is_vector,
                                        is_gradient=a.is_gradient, field_names=a.field_names,
                                        parent_name=a.parent_name))
                
                # ---------------------------------------------------------------------
                # dot( u_k ,  grad(u_k) )     ← e.g. rhs advection term
                # ---------------------------------------------------------------------
                elif a.role == 'value' and a.is_vector and b.role == 'value' and b.is_gradient:
                    body_lines.append("# RHS: dot(Function, grad(Function)) (k).(k,d) ->k")
                    body_lines += [
                        f"{res_var}   = {b.var_name} @ {a.var_name}",
                    ]
                    stack.append(StackItem(var_name=res_var, role='const',
                                        shape=( a.shape[0],), is_vector=True,
                                        is_gradient=False, field_names=[]))
                # ---------------------------------------------------------------------
                # dot( grad(u_k) ,  u_k )     ← e.g. rhs advection term
                # ---------------------------------------------------------------------
                elif a.role == 'value' and a.is_gradient and b.role == 'value' and b.is_vector:
                    body_lines.append("# RHS: dot(grad(Function), Function) (k,d).(k) -> k")
                    body_lines += [
                        f"{res_var}   = {a.var_name} @ {b.var_name}",
                    ]
                    stack.append(StackItem(var_name=res_var, role='const',
                                        shape=(a.shape[0], ), is_vector=True,
                                        is_gradient=False, field_names=[]))
                # ---------------------------------------------------------------------
                # dot( np.array ,  u_test )     ← e.g. body-force · test
                # ---------------------------------------------------------------------
                elif a.role == 'const' and a.is_vector and b.role == 'test' and b.is_vector:
                    # a (k) and b (k,n)
                    body_lines.append("# Constant body-force: dot(const-vec, Test)")
                    body_lines += [
                        f"n_locs = {b.shape[1]}; n_vec_comps = {a.shape[0]};",
                        f"{res_var} = np.zeros((n_locs), dtype=np.float64)",
                        f"for n in range(n_locs):",
                        f"    {res_var}[n] = np.dot({a.var_name}, {b.var_name}[:,n])"
                    ]
                    stack.append(StackItem(var_name=res_var, role='value',
                                        shape=(), is_vector=False))
                
                # ---------------------------------------------------------------------
                # dot( u_k ,  u_test )          ← load-vector term
                # ---------------------------------------------------------------------
                elif a.role == 'value' and a.is_vector and b.role == 'test' and b.is_vector:
                    body_lines.append("# RHS: dot(Function, Test)")
                    # body_lines.append(f"print(f'a.shape: {{{a.var_name}.shape}}, b.shape: {{{b.var_name}.shape}}')")
                    body_lines.append(f"{res_var} = {b.var_name}.T @ {a.var_name}")
                    stack.append(StackItem(var_name=res_var, role='value',
                                        shape=(), is_vector=False,is_gradient=False))
                
                else:
                    raise NotImplementedError(f"Dot not implemented for roles {a.role}/{b.role} with shapes {a.shape}/{b.shape} with vectoors {a.is_vector}/{b.is_vector} and gradients {a.is_gradient}/{b.is_gradient}")

            
            elif isinstance(op, BinaryOp):
                 b = stack.pop(); a = stack.pop()
                 res_var = new_var("res")
                 # -------------------------------------
                 # ------------ PRODUCT ---------------
                 # -------------------------------------
                 if op.op_symbol == '*':
                    body_lines.append(f"# Product: {a.role} * {b.role}")
                    # -----------------------------------------------------------------
                    # 0. scalar:   scalar   *  scalar/np.ndarray    →  scalar/np.ndarray 
                    # -----------------------------------------------------------------
                    if ((a.role == 'const' or a.role=='value') and (b.role == 'const' or b.role=='value')  
                        and not a.is_vector and not b.is_vector 
                            and not a.is_gradient and not b.is_gradient):
                        body_lines.append("# Product: scalar * scalar/np.ndarry → scalar/np.ndarray")
                        body_lines += [
                            f"if np.isscalar({a.var_name}) and (np.isscalar({b.var_name}) or isinstance({b.var_name},np.ndarray)):",
                            f"    {res_var} = {a.var_name} * {b.var_name}",
                            f"elif np.isscalar({b.var_name}) and (np.isscalar({a.var_name}) or isinstance({a.var_name},np.ndarray)):",
                            f"    {res_var} = {b.var_name} * {a.var_name}",
                            f"else:",
                            f"    raise ValueError(f'Both operands must be scalars for this operation. "
                            f"Received: {a.var_name} ({{type({a.var_name})}}), {b.var_name} ({{type({b.var_name})}}) "
                            f"and shapes {{getattr({a.var_name}, \"shape\", None)}}/{{getattr({b.var_name}, \"shape\", None)}}')"

                        ]
                        stack.append(StackItem(var_name=res_var, role='const', shape=(), is_vector=False, field_names=[]))
                        
                    # -----------------------------------------------------------------
                    # 01. Vector, Tensor:   scalar   *  Vector/Tensor    →  Vector/Tensor 
                    # -----------------------------------------------------------------
                    elif ((a.role == 'const' or a.role=='value')  
                         and 
                          (not a.is_vector and not a.is_gradient)) :
                        body_lines.append("# Product: scalar * Vector/Tensor → Vector/Tensor")
                        # a is scalar, b is vector/tensor
                        body_lines += [
                            # f"print(f'a.shape: {{{a.var_name}.shape}}, b.shape: {{{b.var_name}.shape}}')",
                            # f"print(f'a.role: {a.role}, b.role: {b.role}')",
                            # f"print(f'a: {{{a.var_name}}}')",
                            f"if np.isscalar({a.var_name}):",
                            f"    {res_var} = {a.var_name} * {b.var_name}[0]" if b.role == 'test' else f"    {res_var} = {a.var_name} * {b.var_name}",
                            f"else:",
                            f"    raise ValueError('First operand must be a scalar for this operation.')",
                        ]
                        stack.append(StackItem(var_name=res_var, role=b.role,
                                            shape=b.shape, is_vector=b.is_vector,
                                            is_gradient=b.is_gradient,
                                            field_names=b.field_names,
                                            parent_name=b.parent_name))
                        
                    elif ((b.role == 'const' or b.role=='value')
                          and 
                        (not b.is_vector and not b.is_gradient)):
                        body_lines.append("# Product: Vector/Tensor * scalar → Vector/Tensor")
                        # b is scalar, a is vector/tensor
                        body_lines += [
                            # f"print(f'a.shape: {{{a.var_name}.shape}}, b.shape: {{{b.var_name}.shape}}')",
                            # f"print(f'a.role: {a.role}, b.role: {b.role}')",
                            # f"print(f'b: {{{b.var_name}}}')",
                            f"if np.isscalar({b.var_name}):",      
                            f"    {res_var} = {b.var_name} * {a.var_name}[0]" if a.role == 'test' else f"    {res_var} = {b.var_name} * {a.var_name}",
                            f"else:",
                            f"    raise ValueError('Second operand must be a scalar for this operation.')",
                        ]
                        stack.append(StackItem(var_name=res_var, role=a.role,
                                            shape=a.shape, is_vector=a.is_vector,
                                            is_gradient=a.is_gradient,
                                            field_names=a.field_names,
                                            parent_name=a.parent_name))
                        
                    
                    # -----------------------------------------------------------------
                    # 1. LHS block:   scalar test  *  scalar trial   →  outer product
                    # -----------------------------------------------------------------
                    elif (not a.is_gradient and not b.is_gradient and
                        ((a.role, b.role) == ("test", "trial") or
                        (a.role, b.role) == ("trial", "test"))):

                        body_lines.append("# Product: scalar Test × scalar Trial → (n_loc,n_loc)")

                        # orient rows = test , columns = trial
                        test_var  = a if a.role == "test"  else b
                        trial_var = b if a.role == "test"  else a

                        body_lines += [
                            f"{res_var} = {test_var.var_name}.T @ {trial_var.var_name}",  # (n_loc, n_loc)
                        ]
                        stack.append(StackItem(var_name=res_var, role='value',
                                            shape=(), is_vector=False))
                    # -----------------------------------------------------------------
                    # 2. RHS load:   scalar / vector Function  *  scalar Test
                    #                (u_k or c)                ·  φ_v
                    # -----------------------------------------------------------------
                    elif (b.role == "test" and not b.is_vector
                        and a.role == "value" and not a.is_vector
                        and not a.is_gradient and not b.is_gradient):
                        body_lines.append("# Load: scalar Function × scalar Test → (n_loc,)")

                        body_lines.append(f"{res_var} = {a.var_name} * {b.var_name}")   # (n_loc,)
                        stack.append(StackItem(var_name=res_var, role='value',
                                            shape=(), is_vector=False))

                    # symmetric orientation
                    elif (a.role == "test" and not a.is_vector
                        and b.role == "value" and not b.is_vector):
                        body_lines.append("# Load: scalar Test × scalar Function → (n_loc,)")

                        body_lines.append(f"{res_var} = {b.var_name} * {a.var_name}")   # (n_loc,)
                        stack.append(StackItem(var_name=res_var, role='value',
                                            shape=(), is_vector=False))
                    # -----------------------------------------------------------------
                    # 4. Anything else is ***not implemented yet*** – fail fast
                    # -----------------------------------------------------------------
                    else:
                        raise NotImplementedError(
                            f"Product not implemented for roles {a.role}/{b.role} "
                            f"with vector flags {a.is_vector}/{b.is_vector} "
                            f"and gradient flags {a.is_gradient}/{b.is_gradient}."
                            f" Shapes: {a.shape}/{b.shape}"
                        )
                 # -----------------------------------------------------------------
                 # -------------  ADDITION / SUBTRACTION  (+  -)  ------------------
                 # -----------------------------------------------------------------
                 elif op.op_symbol in ('+', '-'):
                    sym = op.op_symbol                    # '+' or '-'
                    body_lines.append(f"# {'Addition' if sym=='+' else 'Subtraction'}")

                    # ---------- helper to choose the resulting role ----------------
                    def _merge_role(ra: str, rb: str) -> str:
                        if 'trial' in (ra, rb):  return 'trial'
                        if 'test'  in (ra, rb):  return 'test'
                        if 'value' in (ra, rb):  return 'value'
                        return 'const'

                    # ----------------------------------------------------------------
                    # CASE A – broadcast with a *scalar* const/value on one side
                    # ----------------------------------------------------------------
                    scalar_left  = (a.shape == () and not a.is_vector and not a.is_gradient)
                    scalar_right = (b.shape == () and not b.is_vector and not b.is_gradient)

                    if scalar_left ^ scalar_right:
                        # orient:  non_scalar  ±  scalar
                        non_scal, scal   = (b, a) if scalar_left else (a, b)
                        op_left,  op_rgt = (scal, non_scal) if scalar_left else (non_scal, scal)

                        body_lines.append("# broadcast scalar with non-scalar")
                        body_lines.append(f"{res_var} = {op_left.var_name} {sym} {op_rgt.var_name}")

                        stack.append(StackItem(
                            var_name    = res_var,
                            role        = non_scal.role,          # keep the structural role
                            shape       = non_scal.shape,
                            is_vector   = non_scal.is_vector,
                            is_gradient = non_scal.is_gradient,
                            field_names = non_scal.field_names
                        ))

                    # ----------------------------------------------------------------
                    # CASE B – identical shapes & flags  (original rule)
                    # ----------------------------------------------------------------
                    elif a.shape == b.shape and a.is_vector == b.is_vector and a.is_gradient == b.is_gradient:
                        same_shape = a.shape == b.shape
                        same_vec   = a.is_vector   == b.is_vector
                        same_grad  = a.is_gradient == b.is_gradient

                        if same_shape and same_vec and same_grad:
                            res_role = _merge_role(a.role, b.role)
                            body_lines.append(f"{res_var} = {a.var_name} {sym} {b.var_name}")

                            stack.append(StackItem(
                                var_name    = res_var,
                                role        = res_role,
                                shape       = a.shape,
                                is_vector   = a.is_vector,
                                is_gradient = a.is_gradient,
                                field_names = a.field_names if a.role in ('trial', 'test')
                                                        else b.field_names
                            ))
                    else:
                        raise NotImplementedError(
                            f"'{sym}' not implemented for roles {a.role}/{b.role} "
                            f"with shapes {a.shape}/{b.shape} "
                            f"or mismatched vector / gradient flags "
                            f"({a.is_vector},{b.is_vector}) – "
                            f"({a.is_gradient},{b.is_gradient})"
                        )
                 # -----------------------------------------------------------------
                 # ------------------  DIVISION  ( /  )  ---------------------------
                 # -----------------------------------------------------------------
                 elif op.op_symbol == '/':
                    body_lines.append("# Division")
                    # divide *anything* by a scalar constant (const in denominator)
                    if (b.role == 'const' or b.role == 'value') and not b.is_vector and np.isscalar(b.shape):
                        body_lines.append(f"{res_var} = {a.var_name} / float({b.var_name})")
                        stack.append(StackItem(var_name=res_var, role=a.role,
                                            shape=a.shape, is_vector=a.is_vector,
                                            is_gradient=a.is_gradient, field_names=a.field_names,
                                            parent_name=a.parent_name))
                    elif (a.role == 'const' or a.role == 'value') and not a.is_vector and np.isscalar(a.shape):
                        body_lines.append(f"{res_var} = float({a.var_name}) / {b.var_name}")
                        stack.append(StackItem(var_name=res_var, role=b.role,
                                            shape=b.shape, is_vector=b.is_vector,
                                            is_gradient=b.is_gradient, field_names=b.field_names,
                                            parent_name=b.parent_name))
                    else:
                        raise NotImplementedError(
                            f"Division not implemented for roles {a.role}/{b.role} "
                            f"with shapes {a.shape}/{b.shape}")

                    

            # --- STORE ---
            elif isinstance(op, Store):
                integrand = stack.pop()
                if op.store_type == 'matrix':
                    body_lines.append(f"Ke += {integrand.var_name} * w_q")
                else:
                    body_lines.append(f"Fe += {integrand.var_name} * w_q")
            
            else:
                raise NotImplementedError(f"Opcode {type(op).__name__} not handled in JIT codegen.")

        source, param_order = self._build_kernel_string(
            kernel_name, body_lines, required_args, solution_func_names
        )
        return source, {}, param_order


    def _build_kernel_string(
            self, kernel_name: str,
            body_lines: list,
            required_args: set,
            solution_func_names: set
            , DEBUG: bool = False
        ):
        """
        Build complete kernel source code with parallel assembly.
        """
        # Add coefficients for all function fields to the required arguments
        for name in solution_func_names:
            required_args.add(f"u_{name}_coeffs")

        param_order = [
            "gdofs_map", "node_coords", "element_nodes",
            "qp_phys", "qw", "detJ", "J_inv", "normals", "phis",
            *sorted(list(required_args))
        ]
        param_order_literal = ", ".join(f"'{arg}'" for arg in param_order)

        # Create the unpacking block for the solution coefficients
        coeffs_unpack_block = "\n".join(
            f"        u_{name}_loc = u_{name}_coeffs[gdofs_e]"
            for name in sorted(list(solution_func_names))
        )
        
        basis_unpack_block = "\n".join(
            f"            {arg}_q = {arg}[q]"
            for arg in sorted(list(required_args))
            if arg.startswith(("b_", "d_", "g_"))
        )

        body_code_block = "\n".join(
            f"            {line}" for line in body_lines if line.strip()
        )

        decorator = ""
        if not DEBUG:
            decorator = "@numba.njit(parallel=True, fastmath=True)"
        final_kernel_src = f"""
import numba
import numpy as np
PARAM_ORDER = [{param_order_literal}]
{decorator}
def {kernel_name}(
        {", ".join(param_order)}
    ):
    num_elements        = gdofs_map.shape[0]
    n_dofs_per_element  = gdofs_map.shape[1]
    
    # Allocate storage for the dense local matrices and vectors
    K_values = np.zeros((num_elements, n_dofs_per_element, n_dofs_per_element), dtype=np.float64)
    F_values = np.zeros((num_elements, n_dofs_per_element), dtype=np.float64)

    for e in numba.prange(num_elements):
        # These are now local to the thread for this element
        Ke = np.zeros((n_dofs_per_element, n_dofs_per_element), dtype=np.float64)
        Fe = np.zeros(n_dofs_per_element, dtype=np.float64)
        gdofs_e = gdofs_map[e]

{coeffs_unpack_block}

        for q in range(qw.shape[1]):
            x_q, w_q, J_inv_q = qp_phys[e, q], qw[e, q], J_inv[e, q]
            normal_q = normals[e, q] if normals is not None else np.zeros(2)
            phi_q    = phis[e, q] if phis is not None else 0.0
{basis_unpack_block}
{body_code_block}

        # Store the final computed local matrix/vector for this element
        K_values[e] = Ke
        F_values[e] = Fe
                
    return K_values, F_values
""".lstrip()

        return final_kernel_src, param_order
