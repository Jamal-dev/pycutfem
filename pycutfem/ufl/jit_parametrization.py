"""
Deterministic, value-independent naming for JIT kernel parameters.

Goal: the kernel cache key and generated code should depend only on the
expression structure ("physics"), not on runtime data such as:
- numeric literal values / Constant.value
- element-wise constant arrays
- BitSet / Restriction masks
- Analytic object identity

Both the IR generator and static-argument builders use this module to
agree on stable parameter names/ids.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict


_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _as_valid_identifier(value: Any) -> str | None:
    if value is None:
        return None
    name = str(value)
    if not _IDENT_RE.match(name):
        raise ValueError(
            f"Invalid JIT parameter name {name!r}; must be a valid identifier."
        )
    return name


@dataclass(slots=True)
class JitParametrization:
    const_name_by_id: Dict[int, str]
    const_by_name: Dict[str, Any]
    ewc_name_by_id: Dict[int, str]
    ewc_by_name: Dict[str, Any]
    analytic_id_by_id: Dict[int, int]
    analytic_by_id: Dict[int, Any]
    domain_token_by_id: Dict[int, str]
    domain_by_token: Dict[str, Any]


_RESERVED_PARAM_NAMES = {
    # Common kernel args
    "gdofs_map",
    "node_coords",
    "element_nodes",
    "qp_phys",
    "qref",
    "qp_ref",
    "qw",
    "detJ",
    "J_inv",
    "J_inv_pos",
    "J_inv_neg",
    "normals",
    "phis",
    "h_arr",
    "eids",
    "entity_kind",
    "owner_id",
    "owner_pos_id",
    "owner_neg_id",
    "pos_eids",
    "neg_eids",
    "pos_map",
    "neg_map",
    # Hessian / higher-order mapping tensors
    "Hxi0",
    "Hxi1",
    "Txi0",
    "Txi1",
    "Qxi0",
    "Qxi1",
    "pos_Hxi0",
    "pos_Hxi1",
    "neg_Hxi0",
    "neg_Hxi1",
    "pos_Txi0",
    "pos_Txi1",
    "neg_Txi0",
    "neg_Txi1",
    "pos_Qxi0",
    "pos_Qxi1",
    "neg_Qxi0",
    "neg_Qxi1",
}


def build_jit_parametrization(expr) -> JitParametrization:
    """
    Compute stable parameter names/ids for dynamic inputs inside *expr*.

    The ordering is derived from a deterministic DFS walk (`_find_all`) which
    is shared across the codebase, making the mapping reproducible across
    runs as long as the expression structure is unchanged.
    """
    from pycutfem.ufl.expressions import Constant, ElementWiseConstant, Restriction
    from pycutfem.ufl.analytic import Analytic
    from pycutfem.ufl.helpers import _find_all

    used: set[str] = set(_RESERVED_PARAM_NAMES)

    # ------------------------------------------------------------------
    # Constants (scalar + tensor)
    # ------------------------------------------------------------------
    const_name_by_id: Dict[int, str] = {}
    const_by_name: Dict[str, Constant] = {}
    const_nodes = _find_all(expr, Constant)
    auto_idx = 0
    for c in const_nodes:
        raw = getattr(c, "_jit_name", None) or getattr(c, "jit_name", None)
        name = _as_valid_identifier(raw) if raw else None
        if not name:
            name = f"jit_const_{auto_idx}"
            auto_idx += 1
        base = name
        suffix = 1
        while name in used:
            suffix += 1
            name = f"{base}_{suffix}"
        used.add(name)
        const_name_by_id[id(c)] = name
        const_by_name[name] = c

    # ------------------------------------------------------------------
    # Element-wise constants
    # ------------------------------------------------------------------
    ewc_name_by_id: Dict[int, str] = {}
    ewc_by_name: Dict[str, ElementWiseConstant] = {}
    ewc_nodes = _find_all(expr, ElementWiseConstant)
    auto_idx = 0
    for ewc in ewc_nodes:
        raw = getattr(ewc, "_jit_name", None) or getattr(ewc, "jit_name", None)
        name = _as_valid_identifier(raw) if raw else None
        if not name:
            name = f"jit_ewc_{auto_idx}"
            auto_idx += 1
        base = name
        suffix = 1
        while name in used:
            suffix += 1
            name = f"{base}_{suffix}"
        used.add(name)
        ewc_name_by_id[id(ewc)] = name
        ewc_by_name[name] = ewc

    # ------------------------------------------------------------------
    # Analytic functions (indexed, not named)
    # ------------------------------------------------------------------
    analytic_id_by_id: Dict[int, int] = {}
    analytic_by_id: Dict[int, Analytic] = {}
    for idx, ana in enumerate(_find_all(expr, Analytic)):
        analytic_id_by_id[id(ana)] = idx
        analytic_by_id[idx] = ana

    # ------------------------------------------------------------------
    # Restriction domains (BitSets) – tokenized by first occurrence.
    # ------------------------------------------------------------------
    domain_token_by_id: Dict[int, str] = {}
    domain_by_token: Dict[str, Any] = {}
    d_idx = 0
    for r in _find_all(expr, Restriction):
        dom = r.domain
        dom_id = id(dom)
        if dom_id in domain_token_by_id:
            continue
        token = f"jit_dom_{d_idx}"
        d_idx += 1
        domain_token_by_id[dom_id] = token
        domain_by_token[token] = dom

    return JitParametrization(
        const_name_by_id=const_name_by_id,
        const_by_name=const_by_name,
        ewc_name_by_id=ewc_name_by_id,
        ewc_by_name=ewc_by_name,
        analytic_id_by_id=analytic_id_by_id,
        analytic_by_id=analytic_by_id,
        domain_token_by_id=domain_token_by_id,
        domain_by_token=domain_by_token,
    )

