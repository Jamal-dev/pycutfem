from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp

from pycutfem.core.dofhandler import DofHandler
from pycutfem.fem import transform
from pycutfem.integration.quadrature import gauss_legendre
from pycutfem.ufl.compilers import FormCompiler
from pycutfem.ufl.helpers import lhs_num, required_multi_indices
from pycutfem.jit.kernel_args import _build_jit_kernel_args, _scatter_element_contribs

from .interface import NonMatchingInterface

log = logging.getLogger(__name__)


def _form_rank(expr) -> int:
    """Return 0 (functional), 1 (linear) or 2 (bilinear)."""
    from pycutfem.ufl.expressions import (
        TrialFunction,
        VectorTrialFunction,
        HdivTrialFunction,
        TestFunction,
        VectorTestFunction,
        HdivTestFunction,
    )

    has_trial = expr.find_first(lambda n: isinstance(n, (TrialFunction, VectorTrialFunction, HdivTrialFunction))) is not None
    has_test = expr.find_first(lambda n: isinstance(n, (TestFunction, VectorTestFunction, HdivTestFunction))) is not None
    return 2 if (has_trial and has_test) else 1 if has_test else 0


def _field_orders(dh: DofHandler) -> dict[str, int]:
    me = getattr(dh, "mixed_element", None)
    if me is None:
        raise ValueError("DofHandler must be MixedElement-backed.")
    orders = getattr(me, "get_field_orders", None)
    if callable(orders):
        return {str(k): int(v) for k, v in (orders() or {}).items()}
    return {str(k): int(v) for k, v in getattr(me, "_field_orders", {}).items()}


def _fields_in_expression(expr) -> list[str]:
    from pycutfem.ufl.expressions import Expression

    fields: set[str] = set()
    visited: set[int] = set()

    def dfs(node):
        nid = id(node)
        if nid in visited:
            return
        visited.add(nid)

        fns = getattr(node, "field_names", None)
        if fns:
            for f in fns:
                if f:
                    fields.add(str(f))
        fn = getattr(node, "field_name", None)
        if fn:
            fields.add(str(fn))

        for v in getattr(node, "__dict__", {}).values():
            if isinstance(v, Expression):
                dfs(v)
            elif isinstance(v, (list, tuple)):
                for vv in v:
                    if isinstance(vv, Expression):
                        dfs(vv)

    dfs(expr)
    return sorted(fields)


@dataclass(frozen=True, slots=True)
class _PrecomputedInterface:
    geo: dict[str, Any]
    pos_offset: int
    neg_offset: int


def precompute_nonmatching_interface_factors_multimesh(
    *,
    interface: NonMatchingInterface,
    dh_pos: DofHandler,
    dh_neg: DofHandler,
    qdeg: int,
    derivs: set[tuple[int, int]],
    ordering: str = "pos_neg",
) -> _PrecomputedInterface:
    """
    Precompute geometry + sided reference-derivative tables for a nonmatching
    interface coupling between two *different* meshes (e.g. quad ↔ tri).

    This is similar to `DofHandler.precompute_nonmatching_interface_factors`,
    but does not require `interface.mesh_pos is interface.mesh_neg`.
    """
    qdeg = int(qdeg)
    ordering = str(ordering).strip().lower()
    if ordering not in {"pos_neg", "neg_pos"}:
        raise ValueError("ordering must be 'pos_neg' or 'neg_pos'")

    n_seg = int(interface.n_segments())
    if n_seg <= 0:
        return _PrecomputedInterface(geo={"eids": np.empty((0,), dtype=np.int32)}, pos_offset=0, neg_offset=0)

    if dh_pos.mixed_element is None or dh_neg.mixed_element is None:
        raise ValueError("Both dh_pos and dh_neg must be MixedElement-backed.")

    me_pos = dh_pos.mixed_element
    me_neg = dh_neg.mixed_element

    fields_pos = list(getattr(me_pos, "field_names", ()))
    fields_neg = list(getattr(me_neg, "field_names", ()))
    if set(fields_pos) != set(fields_neg):
        raise ValueError(f"Field mismatch: pos has {fields_pos}, neg has {fields_neg}")
    fields = [f for f in fields_pos if f in fields_neg]

    fam_pos = getattr(me_pos, "_field_families", {}) or {}
    fam_neg = getattr(me_neg, "_field_families", {}) or {}
    for f in fields:
        if fam_pos.get(f) != fam_neg.get(f):
            raise ValueError(f"Field family mismatch for {f!r}: pos={fam_pos.get(f)!r}, neg={fam_neg.get(f)!r}")
        if fam_pos.get(f) == "RT":
            raise NotImplementedError("Multimesh nonmatching interface precompute does not yet support RT/H(div) fields.")

    n_pos = int(dh_pos.total_dofs)
    n_neg = int(dh_neg.total_dofs)
    if ordering == "pos_neg":
        pos_offset = 0
        neg_offset = n_pos
    else:
        neg_offset = 0
        pos_offset = n_neg

    # 1D quadrature on overlap segments
    xi_1d, w_1d = gauss_legendre(int(qdeg))
    xi_1d = np.asarray(xi_1d, dtype=float)
    w_1d = np.asarray(w_1d, dtype=float)
    nQ = int(xi_1d.size)

    P0 = np.asarray(interface.P0, dtype=float)
    P1 = np.asarray(interface.P1, dtype=float)
    mid = 0.5 * (P0 + P1)
    half = 0.5 * (P1 - P0)
    seg_J = np.linalg.norm(half, axis=1)
    qp_phys = mid[:, None, :] + xi_1d[None, :, None] * half[:, None, :]
    qw = w_1d[None, :] * seg_J[:, None]

    n_vec = np.asarray(interface.n, dtype=float)
    nn = np.linalg.norm(n_vec, axis=1)
    nn = np.where(nn > 0.0, nn, 1.0)
    n_unit = n_vec / nn[:, None]
    normals = np.broadcast_to(n_unit[:, None, :], (n_seg, nQ, 2)).copy()

    pos_eids = np.asarray(interface.pos_elem_ids, dtype=np.int32).ravel()
    neg_eids = np.asarray(interface.neg_elem_ids, dtype=np.int32).ravel()

    # Union DOF maps (per segment entity)
    union_lists: list[np.ndarray] = []
    max_union = 0
    for i in range(n_seg):
        pos_dofs = np.asarray(dh_pos.get_elemental_dofs(int(pos_eids[i])), dtype=np.int64) + int(pos_offset)
        neg_dofs = np.asarray(dh_neg.get_elemental_dofs(int(neg_eids[i])), dtype=np.int64) + int(neg_offset)
        union = np.unique(np.concatenate((pos_dofs, neg_dofs)))
        union_lists.append(np.asarray(union, dtype=np.int64))
        max_union = max(int(max_union), int(union.size))

    n_union = int(max_union) if max_union > 0 else 1
    gdofs_map = np.empty((n_seg, n_union), dtype=np.int32)
    for i in range(n_seg):
        union = union_lists[i]
        if union.size == 0:
            gdofs_map[i, :] = 0
            continue
        pad = int(union[-1])
        gdofs_map[i, :] = pad
        gdofs_map[i, : union.size] = union.astype(np.int32)

    # Side-local union maps (for coefficient gathering via KernelRunner)
    n_loc_pos = int(len(dh_pos.get_elemental_dofs(int(pos_eids[0]))))
    n_loc_neg = int(len(dh_neg.get_elemental_dofs(int(neg_eids[0]))))
    pos_map = -np.ones((n_seg, n_loc_pos), dtype=np.int32)
    neg_map = -np.ones((n_seg, n_loc_neg), dtype=np.int32)

    for i in range(n_seg):
        union = union_lists[i]
        if union.size == 0:
            continue
        pos_dofs = np.asarray(dh_pos.get_elemental_dofs(int(pos_eids[i])), dtype=np.int64) + int(pos_offset)
        neg_dofs = np.asarray(dh_neg.get_elemental_dofs(int(neg_eids[i])), dtype=np.int64) + int(neg_offset)
        pos_map[i, : pos_dofs.size] = np.searchsorted(union, pos_dofs).astype(np.int32)
        neg_map[i, : neg_dofs.size] = np.searchsorted(union, neg_dofs).astype(np.int32)

    # Per-field maps and membership masks
    maps_by_field: dict[str, np.ndarray] = {}
    masks_by_field: dict[str, np.ndarray] = {}
    for fld in fields:
        # local basis count per side (constant across elements for a fixed p)
        nlp = int(len(dh_pos.element_maps[fld][int(pos_eids[0])]))
        nln = int(len(dh_neg.element_maps[fld][int(neg_eids[0])]))
        pm = np.empty((n_seg, nlp), dtype=np.int32)
        nm = np.empty((n_seg, nln), dtype=np.int32)
        pos_mask = np.zeros((n_seg, n_union), dtype=float)
        neg_mask = np.zeros((n_seg, n_union), dtype=float)
        for i in range(n_seg):
            union = union_lists[i]
            pos_loc = np.asarray(dh_pos.element_maps[fld][int(pos_eids[i])], dtype=np.int64) + int(pos_offset)
            neg_loc = np.asarray(dh_neg.element_maps[fld][int(neg_eids[i])], dtype=np.int64) + int(neg_offset)
            pm[i, :] = np.searchsorted(union, pos_loc).astype(np.int32)
            nm[i, :] = np.searchsorted(union, neg_loc).astype(np.int32)
            cols = pm[i]
            ok = (cols >= 0) & (cols < n_union)
            if np.any(ok):
                pos_mask[i, cols[ok]] = 1.0
            cols = nm[i]
            ok = (cols >= 0) & (cols < n_union)
            if np.any(ok):
                neg_mask[i, cols[ok]] = 1.0
        maps_by_field[f"pos_map_{fld}"] = pm
        maps_by_field[f"neg_map_{fld}"] = nm
        masks_by_field[f"restrict_mask_{fld}_pos"] = pos_mask
        masks_by_field[f"restrict_mask_{fld}_neg"] = neg_mask

    # Geometry per side
    xi_pos = np.empty((n_seg, nQ), dtype=float)
    eta_pos = np.empty((n_seg, nQ), dtype=float)
    xi_neg = np.empty((n_seg, nQ), dtype=float)
    eta_neg = np.empty((n_seg, nQ), dtype=float)
    detJ_pos = np.empty((n_seg, nQ), dtype=float)
    detJ_neg = np.empty((n_seg, nQ), dtype=float)
    J_inv_pos = np.empty((n_seg, nQ, 2, 2), dtype=float)
    J_inv_neg = np.empty((n_seg, nQ, 2, 2), dtype=float)

    mesh_pos = interface.mesh_pos
    mesh_neg = interface.mesh_neg
    for i in range(n_seg):
        pe = int(pos_eids[i])
        ne = int(neg_eids[i])
        for q in range(nQ):
            xq = np.asarray(qp_phys[i, q], dtype=float)

            s, t = transform.inverse_mapping(mesh_pos, pe, xq)
            xi_pos[i, q] = float(s)
            eta_pos[i, q] = float(t)
            Jp = np.asarray(transform.jacobian(mesh_pos, pe, (float(s), float(t))), dtype=float)
            detp = float(Jp[0, 0] * Jp[1, 1] - Jp[0, 1] * Jp[1, 0])
            detJ_pos[i, q] = detp
            invd = 1.0 / (detp + 1e-300)
            J_inv_pos[i, q, 0, 0] = Jp[1, 1] * invd
            J_inv_pos[i, q, 0, 1] = -Jp[0, 1] * invd
            J_inv_pos[i, q, 1, 0] = -Jp[1, 0] * invd
            J_inv_pos[i, q, 1, 1] = Jp[0, 0] * invd

            s, t = transform.inverse_mapping(mesh_neg, ne, xq)
            xi_neg[i, q] = float(s)
            eta_neg[i, q] = float(t)
            Jn = np.asarray(transform.jacobian(mesh_neg, ne, (float(s), float(t))), dtype=float)
            detn = float(Jn[0, 0] * Jn[1, 1] - Jn[0, 1] * Jn[1, 0])
            detJ_neg[i, q] = detn
            invd = 1.0 / (detn + 1e-300)
            J_inv_neg[i, q, 0, 0] = Jn[1, 1] * invd
            J_inv_neg[i, q, 0, 1] = -Jn[0, 1] * invd
            J_inv_neg[i, q, 1, 0] = -Jn[1, 0] * invd
            J_inv_neg[i, q, 1, 1] = Jn[0, 0] * invd

    detJ = 0.5 * (detJ_pos + detJ_neg)
    J_inv = 0.5 * (J_inv_pos + J_inv_neg)

    # CellDiameter() support for JIT: owner_id indexes into element-size vector h_arr.
    h_pos = np.sqrt(np.asarray(mesh_pos.areas_list, dtype=float))
    h_neg = np.sqrt(np.asarray(mesh_neg.areas_list, dtype=float))
    h_arr = np.concatenate([h_pos, h_neg]).astype(np.float64)
    pos_elem_offset = int(h_pos.size)
    owner_id = np.empty((n_seg,), dtype=np.int32)
    h_face = np.empty((n_seg,), dtype=np.float64)
    for i in range(n_seg):
        hp = float(h_pos[int(pos_eids[i])] if h_pos.size else 1.0)
        hn = float(h_neg[int(neg_eids[i])] if h_neg.size else 1.0)
        if hp <= hn:
            owner_id[i] = int(pos_eids[i])
            h_face[i] = hp
        else:
            owner_id[i] = int(pos_elem_offset + int(neg_eids[i]))
            h_face[i] = hn

    # Sided reference-derivative tables
    derivs_eff = set(tuple((int(dx), int(dy))) for (dx, dy) in (derivs or set()))
    derivs_eff.add((0, 0))

    basis_tables: dict[str, np.ndarray] = {}
    for fld in fields:
        nlp = int(len(dh_pos.element_maps[fld][int(pos_eids[0])]))
        nln = int(len(dh_neg.element_maps[fld][int(neg_eids[0])]))
        for dx, dy in sorted(derivs_eff):
            tab_pos = np.empty((n_seg, nQ, nlp), dtype=float)
            tab_neg = np.empty((n_seg, nQ, nln), dtype=float)
            for i in range(n_seg):
                for q in range(nQ):
                    tab_pos[i, q, :] = me_pos._eval_scalar_deriv(fld, float(xi_pos[i, q]), float(eta_pos[i, q]), int(dx), int(dy))
                    tab_neg[i, q, :] = me_neg._eval_scalar_deriv(fld, float(xi_neg[i, q]), float(eta_neg[i, q]), int(dx), int(dy))
            basis_tables[f"r{dx}{dy}_{fld}_pos"] = tab_pos
            basis_tables[f"r{dx}{dy}_{fld}_neg"] = tab_neg

    geo: dict[str, Any] = {
        "domain": "nonmatching_interface",
        "eids": np.arange(n_seg, dtype=np.int32),
        "qp_phys": qp_phys,
        "qw": qw,
        "normals": normals,
        "xi_pos": xi_pos,
        "eta_pos": eta_pos,
        "xi_neg": xi_neg,
        "eta_neg": eta_neg,
        "gdofs_map": gdofs_map,
        "pos_map": pos_map,
        "neg_map": neg_map,
        "J_inv_pos": J_inv_pos,
        "J_inv_neg": J_inv_neg,
        "detJ_pos": detJ_pos,
        "detJ_neg": detJ_neg,
        "detJ": detJ,
        "J_inv": J_inv,
        "owner_id": owner_id,
        "h_arr": h_arr,
        # Python backend CellDiameter override
        "h_face": h_face,
        # convenience
        "owner_pos_id": pos_eids,
        "owner_neg_id": neg_eids,
    }
    geo.update(basis_tables)
    geo.update(maps_by_field)
    geo.update(masks_by_field)

    return _PrecomputedInterface(geo=geo, pos_offset=int(pos_offset), neg_offset=int(neg_offset))


def assemble_nonmatching_interface_form(
    integrand,
    *,
    interface: NonMatchingInterface,
    dh_pos: DofHandler,
    dh_neg: DofHandler,
    quad_order: int | None = None,
    backend: str = "python",
    ordering: str = "pos_neg",
) -> tuple[sp.csr_matrix | None, np.ndarray | None]:
    """
    Assemble a nonmatching-interface integral written in UFL between two meshes.

    The returned system is ordered as:
      - ordering='pos_neg': [U_pos, U_neg]
      - ordering='neg_pos': [U_neg, U_pos]
    """
    backend = str(backend).strip().lower()
    rank = _form_rank(integrand)
    if rank not in (1, 2):
        raise NotImplementedError("Only linear and bilinear nonmatching-interface forms are supported.")

    fields_used = _fields_in_expression(integrand)
    if not fields_used:
        fields_used = list(getattr(dh_pos, "field_names", []))

    # Derivatives needed (reference-space tables)
    derivs = set(required_multi_indices(integrand))
    derivs |= {(0, 0)}
    max_total = max((int(dx) + int(dy) for (dx, dy) in derivs), default=0)
    for p in range(max_total + 1):
        for dx in range(p + 1):
            derivs.add((dx, p - dx))

    # Quadrature order heuristic
    if quad_order is None:
        orders_pos = _field_orders(dh_pos)
        orders_neg = _field_orders(dh_neg)
        max_p = 1
        for f in fields_used:
            max_p = max(max_p, int(orders_pos.get(f, 1)), int(orders_neg.get(f, 1)))
        quad_order = int(2 * max_p + 2)
    qdeg = int(max(int(quad_order), 2 * max_total + 4))

    # Precompute geometry + tables
    pre = precompute_nonmatching_interface_factors_multimesh(
        interface=interface,
        dh_pos=dh_pos,
        dh_neg=dh_neg,
        qdeg=qdeg,
        derivs=derivs,
        ordering=ordering,
    )
    geo = pre.geo
    eids = np.asarray(geo.get("eids", []), dtype=np.int32)
    if eids.size == 0:
        n_total = int(dh_pos.total_dofs) + int(dh_neg.total_dofs)
        if rank == 2:
            return sp.csr_matrix((n_total, n_total), dtype=float), None
        return None, np.zeros(n_total, dtype=float)

    n_total = int(dh_pos.total_dofs) + int(dh_neg.total_dofs)
    rhs = (rank == 1)

    if backend == "python":
        compiler = FormCompiler(dh_pos, backend="python")
        compiler.ctx["rhs"] = rhs
        compiler.ctx["is_interface"] = True
        compiler.ctx["is_ghost"] = False
        compiler.ctx["measure_side"] = None

        mat = sp.lil_matrix((n_total, n_total), dtype=float) if not rhs else None
        vec = np.zeros(n_total, dtype=float) if rhs else None

        qp_phys = np.asarray(geo["qp_phys"], dtype=float)
        qw = np.asarray(geo["qw"], dtype=float)
        normals = np.asarray(geo["normals"], dtype=float)
        pos_eids = np.asarray(geo["owner_pos_id"], dtype=np.int32)
        neg_eids = np.asarray(geo["owner_neg_id"], dtype=np.int32)
        h_face = np.asarray(geo.get("h_face", np.ones(len(eids))), dtype=float)

        for ei in range(int(eids.size)):
            pos_eid = int(pos_eids[ei])
            neg_eid = int(neg_eids[ei])

            pos_dofs = np.asarray(dh_pos.get_elemental_dofs(pos_eid), dtype=int) + int(pre.pos_offset)
            neg_dofs = np.asarray(dh_neg.get_elemental_dofs(neg_eid), dtype=int) + int(pre.neg_offset)
            global_dofs = np.unique(np.concatenate([pos_dofs, neg_dofs]))
            pos_map = np.searchsorted(global_dofs, pos_dofs)
            neg_map = np.searchsorted(global_dofs, neg_dofs)

            pos_map_by_field: dict[str, np.ndarray] = {}
            neg_map_by_field: dict[str, np.ndarray] = {}
            pos_union_mask_by_field: dict[str, np.ndarray] = {}
            neg_union_mask_by_field: dict[str, np.ndarray] = {}
            for fld in fields_used:
                if fld not in dh_pos.element_maps or fld not in dh_neg.element_maps:
                    continue
                dofs_p = np.asarray(dh_pos.element_maps[fld][pos_eid], dtype=int) + int(pre.pos_offset)
                dofs_n = np.asarray(dh_neg.element_maps[fld][neg_eid], dtype=int) + int(pre.neg_offset)
                mp = np.searchsorted(global_dofs, dofs_p)
                mn = np.searchsorted(global_dofs, dofs_n)
                pos_map_by_field[fld] = mp
                neg_map_by_field[fld] = mn
                m = np.zeros(len(global_dofs), dtype=float)
                m[np.asarray(mp, dtype=int)] = 1.0
                pos_union_mask_by_field[fld] = m
                m = np.zeros(len(global_dofs), dtype=float)
                m[np.asarray(mn, dtype=int)] = 1.0
                neg_union_mask_by_field[fld] = m

            acc = np.zeros(len(global_dofs), float) if rhs else np.zeros((len(global_dofs), len(global_dofs)), float)
            for q in range(int(qp_phys.shape[1])):
                wq = float(qw[ei, q])
                if wq == 0.0:
                    continue

                # Provide sided basis/derivative rows directly (field-local).
                basis_values = {"+": {}, "-": {}}
                for fld in fields_used:
                    pos_block = {}
                    neg_block = {}
                    for dx, dy in derivs:
                        kpos = f"r{dx}{dy}_{fld}_pos"
                        kneg = f"r{dx}{dy}_{fld}_neg"
                        if kpos in geo:
                            pos_block[(dx, dy)] = np.asarray(geo[kpos][ei, q], dtype=float)
                        if kneg in geo:
                            neg_block[(dx, dy)] = np.asarray(geo[kneg][ei, q], dtype=float)
                    if pos_block:
                        basis_values["+"][fld] = pos_block
                    if neg_block:
                        basis_values["-"][fld] = neg_block

                compiler.ctx.update(
                    {
                        "eid": 0,
                        "pos_eid": pos_eid,
                        "neg_eid": neg_eid,
                        "normal": np.asarray(normals[ei, q], float),
                        "x_phys": np.asarray(qp_phys[ei, q], float),
                        "phi_val": 0.0,
                        "global_dofs": global_dofs,
                        "pos_map": pos_map,
                        "neg_map": neg_map,
                        "pos_map_by_field": pos_map_by_field,
                        "neg_map_by_field": neg_map_by_field,
                        "pos_union_mask_by_field": pos_union_mask_by_field,
                        "neg_union_mask_by_field": neg_union_mask_by_field,
                        "use_union_local_dofs": True,
                        "basis_values": basis_values,
                        "h_val": float(h_face[ei]),
                    }
                )

                val = compiler._visit(integrand)
                arr = lhs_num(val)
                arr = np.asarray(arr)
                if rhs and arr.ndim == 2 and 1 in arr.shape:
                    arr = arr.reshape(-1)
                acc += wq * arr

            if rhs:
                np.add.at(vec, global_dofs, acc)
            else:
                rr, cc = np.meshgrid(global_dofs, global_dofs, indexing="ij")
                mat[rr, cc] += acc

        return (mat.tocsr(), None) if not rhs else (None, vec)

    if backend not in {"jit", "cpp", "c++"}:
        raise ValueError("backend must be 'python', 'jit', or 'cpp'.")

    # JIT/CPP path: compile a facet kernel and execute it on the segment set.
    if backend in {"cpp", "c++"}:
        from pycutfem.jit.cpp_backend import compile_backend_cpp as _compile
    else:
        from pycutfem.jit import compile_backend as _compile

    runner, ir = _compile(integrand, dh_pos, dh_pos.mixed_element, on_facet=True)

    geo = dict(geo)  # ensure mutable
    geo["is_interface"] = True
    geo["is_ghost"] = False

    kernel_args = _build_jit_kernel_args(
        ir=ir,
        expression=integrand,
        mixed_element=dh_pos.mixed_element,
        q_order=int(qdeg),
        dof_handler=dh_pos,
        gdofs_map=np.asarray(geo["gdofs_map"], dtype=np.int32),
        param_order=runner.param_order,
        pre_built=geo,
    )

    if ("node_coords" in runner.param_order) and ("node_coords" not in kernel_args):
        kernel_args["node_coords"] = np.asarray(dh_pos.mixed_element.mesh.nodes_x_y_pos, dtype=float)
    if ("element_nodes" in runner.param_order) and ("element_nodes" not in kernel_args):
        kernel_args["element_nodes"] = np.asarray(dh_pos.mixed_element.mesh.elements_connectivity, dtype=np.int64)

    current_funcs: dict[str, Any] = {}
    K_ent, F_ent, J_ent = runner(current_funcs, kernel_args)

    if rhs:
        vec = np.zeros(n_total, dtype=float)
        _scatter_element_contribs(
            K_ent,
            F_ent,
            J_ent,
            np.asarray(geo["eids"], dtype=np.int32),
            np.asarray(geo["gdofs_map"], dtype=np.int32),
            vec,
            {"rhs": True},
            integrand,
            hook=None,
        )
        return None, vec

    mat = sp.lil_matrix((n_total, n_total), dtype=float)
    _scatter_element_contribs(
        K_ent,
        F_ent,
        J_ent,
        np.asarray(geo["eids"], dtype=np.int32),
        np.asarray(geo["gdofs_map"], dtype=np.int32),
        mat,
        {"rhs": False},
        integrand,
        hook=None,
    )
    return mat.tocsr(), None
