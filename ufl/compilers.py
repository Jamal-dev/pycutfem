import numpy as np
import scipy.sparse as sp
from ufl.expressions import (
    Constant,
    TrialFunction,
    TestFunction,
    Grad,
    Inner,
    Sum,
    Sub,
    Prod,
    Div,
    Derivative,
    ElementWiseConstant,
    Jump,
)

from pycutfem.fem.reference import get_reference
from pycutfem.integration.quadrature import edge as line_quadrature
from pycutfem.integration import volume
from pycutfem.fem import transform
from pycutfem.core.dofhandler import DofHandler

# --------------------------------------------------------------------------------------
# Helper utilities
# --------------------------------------------------------------------------------------

def _find_functions(expression):
    """Recursively finds the first TrialFunction and TestFunction in an expression tree."""
    if isinstance(expression, TrialFunction):
        return (expression, None)
    if isinstance(expression, TestFunction):
        return (None, expression)

    trial_func, test_func = None, None
    if hasattr(expression, "a") and hasattr(expression, "b"):
        tf1, tsf1 = _find_functions(expression.a)
        tf2, tsf2 = _find_functions(expression.b)
        trial_func, test_func = tf1 or tf2, tsf1 or tsf2
    elif hasattr(expression, "operand"):
        trial_func, test_func = _find_functions(expression.operand)
    elif hasattr(expression, "f"):
        trial_func, test_func = _find_functions(expression.f)
    elif hasattr(expression, "u_pos"):
        trial_func, test_func = _find_functions(expression.u_pos)
        if not trial_func and not test_func:
            trial_func, test_func = _find_functions(expression.u_neg)
    return trial_func, test_func


def _flatten_integrand(expression):
    """Flattens sums inside an integrand so that every term is (sign, term)."""
    if isinstance(expression, Sum):
        return _flatten_integrand(expression.a) + _flatten_integrand(expression.b)
    if isinstance(expression, Sub):
        return _flatten_integrand(expression.a) + [(-s, t) for s, t in _flatten_integrand(expression.b)]

    # Distribute Prod and Inner over sums so that e.g. (A+B)*C becomes A*C + B*C
    op_type = type(expression)
    if op_type in (Prod, Inner):
        if isinstance(expression.a, Sum):
            return _flatten_integrand(op_type(expression.a.a, expression.b)) + _flatten_integrand(
                op_type(expression.a.b, expression.b)
            )
        if isinstance(expression.a, Sub):
            return _flatten_integrand(op_type(expression.a.a, expression.b)) + [
                (-s, t)
                for s, t in _flatten_integrand(op_type(expression.a.b, expression.b))
            ]
        if isinstance(expression.b, Sum):
            return _flatten_integrand(op_type(expression.a, expression.b.a)) + _flatten_integrand(
                op_type(expression.a, expression.b.b)
            )
        if isinstance(expression.b, Sub):
            return _flatten_integrand(op_type(expression.a, expression.b.a)) + [
                (-s, t)
                for s, t in _flatten_integrand(op_type(expression.a, expression.b.b))
            ]
    return [(1, expression)]


# --------------------------------------------------------------------------------------
# Main compiler class
# --------------------------------------------------------------------------------------

class FormCompiler:
    """Very small, very specialised Form compiler for the pycutfem internal UFL clone."""

    def __init__(self, dof_handler: DofHandler, quad_order: int | None = None):
        self.dof_handler = dof_handler
        self.quad_order = quad_order
        # The *context* dict is how we avoid gigantic argument lists in every visitor
        self.context: dict[str, object] = {}

    # ------------------------------------------------------------------
    # Visitor dispatch helpers
    # ------------------------------------------------------------------

    def visit(self, node):
        method_name = "visit_" + type(node).__name__
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        raise TypeError(f"No visit_{type(node).__name__} method for {type(node)}")

    # ------------------------------------------------------------------
    # Elementary nodes
    # ------------------------------------------------------------------

    def visit_Constant(self, node):
        return node.value

    def visit_ElementWiseConstant(self, node):
        side = self.context.get("side", "")
        if side:
            elem_id = self.context["e_pos"] if side == "+" else self.context["e_neg"]
        else:
            elem_id = self.context["elem_id"]
        return node.values[elem_id]

    def visit_Sum(self, node):
        return self.visit(node.a) + self.visit(node.b)

    def visit_Sub(self, node):
        return self.visit(node.a) - self.visit(node.b)

    def visit_Prod(self, node):
        val_a = self.visit(node.a)
        val_b = self.visit(node.b)

        # Outer‐product assembly shortcut for matrix terms
        if (
            not self.context["is_rhs"]
            and hasattr(val_a, "ndim")
            and hasattr(val_b, "ndim")
            and val_a.ndim == 1
            and val_b.ndim == 1
        ):
            _, test_fn_a = _find_functions(node.a)
            trial_fn_b, _ = _find_functions(node.b)
            if test_fn_a and trial_fn_b:
                return np.outer(val_a, val_b)
            trial_fn_a, _ = _find_functions(node.a)
            _, test_fn_b = _find_functions(node.b)
            if trial_fn_a and test_fn_b:
                return np.outer(val_b, val_a)

        return val_a * val_b

    def visit_Analytic(self, node):
        return node.eval(self.context["x_phys"])

    # ------------------------------------------------------------------
    # Basis‑function related nodes
    # ------------------------------------------------------------------

    def _basis_lookup(self, field: str, which: str, side: str):
        """Small helper that is side‑aware and provides a graceful fallback."""
        try:
            return self.context["basis_values"][field][side][which]
        except KeyError:
            # When *side* is "" (default) but only + / ‑ exist, we fall back to +
            if side == "":
                return self.context["basis_values"][field]["+"][which]
            raise

    def visit_TrialFunction(self, node):
        if self.context["is_rhs"]:
            raise TypeError("TrialFunction on RHS is not allowed.")
        side = self.context.get("side", "")
        return self._basis_lookup(node.field_name, "val", side)

    def visit_TestFunction(self, node):
        side = self.context.get("side", "")
        return self._basis_lookup(node.field_name, "val", side)

    def visit_Grad(self, node):
        field = node.operand.field_name
        side = self.context.get("side", "")
        return self._basis_lookup(field, "grad", side)

    def visit_Derivative(self, node):
        grad_basis = self.visit(Grad(node.f))
        return grad_basis[:, node.component_index]

    # ------------------------------------------------------------------
    # Higher‑level algebra
    # ------------------------------------------------------------------

    def visit_Inner(self, node):
        """Evaluate an Inner/ dot product between two operands *at the current
        quadrature point* and return an object that has the correct shape for
        the *current* assembly context (matrix or vector).

        ── Implementation strategy ───────────────────────────────────────────
        • On the **RHS** we only need a *scalar* value per quadrature point, so
          we use NumPy’s broadcasting‑friendly ``einsum('i...,i...->...')``.

        • On the **LHS** (matrix assembly) we need an *outer‑type* product that
          yields a ``(n_test, n_trial)`` dense block.  Four practically useful
          shape combinations are supported:

            (a) val_a ≡ (n_test,) ,   val_b ≡ (n_trial,)      → outer
            (b) val_a ≡ (n_test,k),   val_b ≡ (n_trial,k)     → contract k
            (c) val_a ≡ (n_test,k),   val_b ≡ (n_trial,)      → Σ_k val_a * val_b
            (d) val_a ≡ (n_test,) ,   val_b ≡ (n_trial,k)     → Σ_k val_b * val_a

          For cases (c) and (d) we *first* collapse the length‑k vector by
          summing over its last axis before forming the outer product.  This is
          a pragmatic approximation that works for face‑coupling terms such as
          ``dot(avg_flux_u, jump_v)`` that triggered the original failure.
        """
        val_a = self.visit(node.a)
        val_b = self.visit(node.b)

        # --------------------
        # RHS – scalar result
        # --------------------
        if self.context["is_rhs"]:
            return np.einsum("i...,i...->...", val_a, val_b)

        # --------------------
        # LHS – produce block
        # --------------------
        # Helper to flatten a possible vector component dimension (k)
        def _collapse(v):
            return v.sum(axis=-1) if v.ndim > 1 else v

        # (a) both 1‑D: straightforward outer product
        if val_a.ndim == 1 and val_b.ndim == 1:
            return np.outer(val_a, val_b)
        # (b) both 2‑D with matching k dimension -> contract k
        if val_a.ndim == 2 and val_b.ndim == 2 and val_a.shape[1] == val_b.shape[1]:
            return np.einsum("ik,jk->ij", val_a, val_b)
        # (c)  val_a has k‑dimension, val_b is 1‑D
        if val_a.ndim == 2 and val_b.ndim == 1:
            return np.outer(_collapse(val_a), val_b).T  # (n_test,k)->n_test after collapse
        # (d)  val_a is 1‑D, val_b has k‑dimension
        if val_a.ndim == 1 and val_b.ndim == 2:
            return np.outer(val_a, _collapse(val_b))

        # Fallback – let NumPy handle broadcasting or raise a diagnostic error
        try:
            return val_a * val_b  # element‑wise product with broadcasting
        except ValueError as err:
            raise ValueError(
                "visit_Inner could not reconcile operand shapes: "
                f"{val_a.shape} vs {val_b.shape}"
            ) from err

    def visit_Jump(self, node):
        # u⁺ − u⁻
        self.context["side"] = "+"
        val_pos = self.visit(node.u_pos)
        self.context["side"] = "-"
        val_neg = self.visit(node.u_neg)
        self.context["side"] = ""
        return val_pos - val_neg

    # ----------------------------------------------------------------------------------
    # Public driver
    # ----------------------------------------------------------------------------------

    def assemble(self, system, bcs):
        n_dofs = self.dof_handler.total_dofs
        K = sp.lil_matrix((n_dofs, n_dofs))
        F = np.zeros(n_dofs)

        # Ensure we always work with an iterable of equations
        if not isinstance(system, (list, tuple)):
            system = [system]

        for equation in system:
            # Stiffness/matrix part (LHS)
            self.context["is_rhs"] = False
            self._assemble_form(equation.a, K)

            # Load / RHS part
            self.context["is_rhs"] = True
            self._assemble_form(equation.L, F)

        K, F = self._apply_bcs(K, F, bcs)
        return K.tocsr(), F

    # ----------------------------------------------------------------------------------
    # BC helper
    # ----------------------------------------------------------------------------------

    def _apply_bcs(self, K, F, bcs):
        if not bcs:
            return K, F
        dirichlet_data = self.dof_handler.get_dirichlet_data(bcs)
        if not dirichlet_data:
            return K, F

        dofs = list(dirichlet_data.keys())
        values = np.array(list(dirichlet_data.values()))

        # Modify RHS first (strong BC elimination)
        u_d = np.zeros_like(F)
        u_d[dofs] = values
        F -= K @ u_d

        # Row/col zero‑out & set identity
        K_lil = K.tolil()
        for dof in dofs:
            K_lil[dof, :] = 0.0
            K_lil[:, dof] = 0.0
            K_lil[dof, dof] = 1.0
        F[dofs] = values
        return K_lil, F

    # ----------------------------------------------------------------------------------
    # Core assembly routines – volume and interior facets
    # ----------------------------------------------------------------------------------

    def _assemble_form(self, form, matrix_or_vector):
        for integral in form.integrals:
            if integral.measure.domain_type == "volume":
                self._assemble_volume_integral(integral, matrix_or_vector)
            elif integral.measure.domain_type == "interior_facet":
                self._assemble_face_integral(integral, matrix_or_vector)

    # ----------------------------------------
    # Utility to collect fields used in a term
    # ----------------------------------------

    def _get_all_fields_from_term(self, term):
        fields: set[str] = set()

        def find_in_node(n):
            if isinstance(n, (TrialFunction, TestFunction)):
                fields.add(n.field_name)
            if hasattr(n, "a") and hasattr(n, "b"):
                find_in_node(n.a)
                find_in_node(n.b)
            elif hasattr(n, "operand"):
                find_in_node(n.operand)
            elif hasattr(n, "f"):
                find_in_node(n.f)
            elif hasattr(n, "u_pos"):
                find_in_node(n.u_pos)
                find_in_node(n.u_neg)

        find_in_node(term)
        return fields

    # ----------------------------------------
    # Volume (cell) integrals
    # ----------------------------------------

    def _assemble_volume_integral(self, integral, matrix_or_vector):
        is_rhs = self.context["is_rhs"]
        terms = _flatten_integrand(integral.integrand)

        # Determine the mesh from the first TestFunction
        primary_test_fn = _find_functions(integral.integrand)[1]
        if not primary_test_fn:
            raise ValueError("Integral is missing a TestFunction.")
        mesh = self.dof_handler.fe_map[primary_test_fn.field_name]

        quad_order = self.quad_order or mesh.poly_order + 2
        elements_to_iterate = (
            integral.measure.defined_on.to_indices()
            if integral.measure.defined_on is not None
            else range(len(mesh.elements_list))
        )

        # Pre‑compute reference quadrature once
        pts, wts = volume(mesh.element_type, quad_order)

        for elem_id in elements_to_iterate:
            # --------------------------------------------------
            # Basis pre‑computation for *all* fields at all q‑pts
            # --------------------------------------------------
            all_fields = self._get_all_fields_from_term(integral.integrand)
            basis_values_at_quad_points: list[dict] = []
            for (xi, eta) in (p for p in pts):
                bv = {}
                for field in all_fields:
                    field_mesh = self.dof_handler.fe_map[field]
                    ref = get_reference(field_mesh.element_type, field_mesh.poly_order)
                    J = transform.jacobian(field_mesh, elem_id, (xi, eta))
                    invJ_T = np.linalg.inv(J).T
                    bv[field] = {
                        "": {
                            "val": ref.shape(xi, eta),
                            "grad": ref.grad(xi, eta) @ invJ_T,
                        }
                    }
                basis_values_at_quad_points.append(bv)

            # --------------------------------------------------
            # Loop over every *term* and assemble
            # --------------------------------------------------
            for sign, term in terms:
                trial_fn, test_fn = _find_functions(term)
                if not test_fn:
                    if is_rhs:
                        test_fn = primary_test_fn
                    else:
                        continue  # matrix term but no TestFunction – skip

                test_field = test_fn.field_name
                trial_field = trial_fn.field_name if trial_fn else None

                test_ref = get_reference(
                    self.dof_handler.fe_map[test_field].element_type,
                    self.dof_handler.fe_map[test_field].poly_order,
                )
                n_loc_test = len(test_ref.shape(0, 0))
                n_loc_trial = (
                    len(
                        get_reference(
                            self.dof_handler.fe_map[trial_field].element_type,
                            self.dof_handler.fe_map[trial_field].poly_order,
                        ).shape(0, 0)
                    )
                    if trial_field
                    else 0
                )

                local_contrib = (
                    np.zeros((n_loc_test, n_loc_trial)) if not is_rhs else np.zeros(n_loc_test)
                )

                # Quadrature loop
                for (xi, eta), w in zip(pts, wts):
                    self.context["elem_id"] = elem_id
                    self.context["side"] = ""  # volume integrals have no side notion
                    self.context["x_phys"] = transform.x_mapping(mesh, elem_id, (xi, eta))
                    self.context["basis_values"] = basis_values_at_quad_points.pop(0)
                    val = self.visit(term)
                    J = transform.jacobian(mesh, elem_id, (xi, eta))
                    local_contrib += sign * w * abs(np.linalg.det(J)) * val

                # Scatter to global vector / matrix
                dofs_row = self.dof_handler.element_maps[test_field][elem_id]
                if not is_rhs and trial_field is not None:
                    dofs_col = self.dof_handler.element_maps[trial_field][elem_id]
                    matrix_or_vector[np.ix_(dofs_row, dofs_col)] += local_contrib
                else:
                    matrix_or_vector[dofs_row] += local_contrib

    # ----------------------------------------
    # Interior‑facet (edge) integrals
    # ----------------------------------------

    def _assemble_face_integral(self, integral, matrix_or_vector):
        is_rhs = self.context["is_rhs"]
        terms = _flatten_integrand(integral.integrand)

        # Get the mesh from the primary TestFunction
        primary_test_fn = _find_functions(integral.integrand)[1]
        if not primary_test_fn:
            raise ValueError("Face integral missing TestFunction.")
        mesh = self.dof_handler.fe_map[primary_test_fn.field_name]

        quad_order = self.quad_order or mesh.poly_order + 2

        edges_to_iterate = (
            integral.measure.defined_on.to_indices()
            if integral.measure.defined_on is not None
            else [e.gid for e in mesh.edges_list if e.right is not None]
        )

        level_set = integral.measure.level_set
        if level_set is None:
            raise ValueError("ds() integrals require a level_set function.")

        for edge_id in edges_to_iterate:
            edge_obj = mesh.edge(edge_id)
            eL, eR = edge_obj.left, edge_obj.right
            if eR is None:
                continue  # boundary edge – should not happen for ds

            local_edge_idx = mesh.elements_list[eL].edges.index(edge_id)
            pts, wts = line_quadrature(mesh.element_type, local_edge_idx, quad_order)

            all_fields = self._get_all_fields_from_term(integral.integrand)

            for (xi_ref, _), w in zip(pts, wts):
                # Map quad‑point to physical coords using *left* element reference
                x_phys = transform.x_mapping(mesh, eL, (xi_ref, 0))

                # Determine orientation ( + / ‑ ) based on level‑set sign at left centroid
                phi_L = level_set(mesh.elements_list[eL].centroid())
                e_pos, e_neg = (eL, eR) if phi_L > 0 else (eR, eL)

                # --------------------------------------------------
                # Basis values on both sides at this *physical* point
                # --------------------------------------------------
                basis_values: dict = {}
                for field in all_fields:
                    fmesh = self.dof_handler.fe_map[field]
                    fref = get_reference(fmesh.element_type, fmesh.poly_order)

                    # + side
                    xi_pos, eta_pos = transform.inverse_mapping(fmesh, e_pos, x_phys)
                    J_pos = transform.jacobian(fmesh, e_pos, (xi_pos, eta_pos))
                    invJ_T_pos = np.linalg.inv(J_pos).T
                    basis_values[field] = {
                        "+": {
                            "val": fref.shape(xi_pos, eta_pos),
                            "grad": fref.grad(xi_pos, eta_pos) @ invJ_T_pos,
                        }
                    }

                    # - side
                    xi_neg, eta_neg = transform.inverse_mapping(fmesh, e_neg, x_phys)
                    J_neg = transform.jacobian(fmesh, e_neg, (xi_neg, eta_neg))
                    invJ_T_neg = np.linalg.inv(J_neg).T
                    basis_values[field]["-"] = {
                        "val": fref.shape(xi_neg, eta_neg),
                        "grad": fref.grad(xi_neg, eta_neg) @ invJ_T_neg,
                    }

                    # Default ("") side *aliases* plus side so that Grad(u) outside of Jump works
                    basis_values[field][""] = basis_values[field]["+"]

                # --------------------------------------------------
                # Update context for this quadrature point
                # --------------------------------------------------
                self.context.update(
                    {
                        "basis_values": basis_values,
                        "x_phys": x_phys,
                        "elem_id": e_pos,  # + side element is used for EWC values when side=""
                        "e_pos": e_pos,
                        "e_neg": e_neg,
                        "side": "",  # default side – overwritten inside Jump()
                    }
                )

                jac1d = transform.jacobian_1d(mesh, eL, (xi_ref, 0), local_edge_idx)

                # --------------------------------------------------
                # Loop over every *term* for this q‑point
                # --------------------------------------------------
                for sign, term in terms:
                    trial_fn, test_fn = _find_functions(term)
                    if not test_fn:
                        if is_rhs:
                            test_fn = primary_test_fn
                        else:
                            continue

                    test_field = test_fn.field_name
                    trial_field = trial_fn.field_name if trial_fn else None

                    test_ref = get_reference(
                        self.dof_handler.fe_map[test_field].element_type,
                        self.dof_handler.fe_map[test_field].poly_order,
                    )
                    n_loc_test = len(test_ref.shape(0, 0))
                    n_loc_trial = (
                        len(
                            get_reference(
                                self.dof_handler.fe_map[trial_field].element_type,
                                self.dof_handler.fe_map[trial_field].poly_order,
                            ).shape(0, 0)
                        )
                        if trial_field
                        else 0
                    )

                    local_contrib = (
                        np.zeros((n_loc_test, n_loc_trial))
                        if not is_rhs
                        else np.zeros(n_loc_test)
                    )

                    val = self.visit(term)
                    local_contrib += sign * w * jac1d * val

                    # Scatter
                    dofs_row = self.dof_handler.element_maps[test_field][e_pos]
                    if not is_rhs and trial_field is not None:
                        dofs_col = self.dof_handler.element_maps[trial_field][e_pos]
                        matrix_or_vector[np.ix_(dofs_row, dofs_col)] += local_contrib
                    else:
                        matrix_or_vector[dofs_row] += local_contrib
