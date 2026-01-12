from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from pycutfem.core.sideconvention import SIDE
from pycutfem.ufl.measures import dx
from pycutfem.ufl.expressions import TestFunction, TrialFunction
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.helpers_geom import phi_eval


def _heaviside(phi: float) -> float:
    return 1.0 if SIDE.is_pos(float(phi), tol=SIDE.tol) else 0.0


@dataclass(frozen=True)
class _OldElementCache:
    base_gdofs: np.ndarray
    u_base: np.ndarray
    a_enr: np.ndarray
    node_phi: np.ndarray
    kind: str | None


def _eval_old_scalar_on_element(
    *,
    dh_old,
    u_old: np.ndarray,
    field: str,
    eid: int,
    xi: float,
    eta: float,
    x_phys: np.ndarray,
    level_set_old,
    cache: dict[tuple[int, str], _OldElementCache],
) -> float:
    base_dh = getattr(dh_old, "base", dh_old)
    me = base_dh.mixed_element
    n0 = int(me._n_basis[field])

    key = (int(eid), str(field))
    c = cache.get(key)
    if c is None:
        base_gdofs = np.asarray(base_dh.element_maps[field][int(eid)], dtype=int)
        if base_gdofs.size != n0:
            raise ValueError(f"Unexpected local basis size for field '{field}': {base_gdofs.size} != {n0}")
        u_base = np.asarray(u_old[base_gdofs], dtype=float)

        kind = None
        a_enr = np.zeros((n0,), dtype=float)
        node_phi = np.zeros((n0,), dtype=float)

        # Only XFEMDofHandler carries enrichment info.
        spec = getattr(dh_old, "enrichment_spec", None)
        kind_by_field = getattr(spec, "kind_by_field", None) if spec is not None else None
        if isinstance(kind_by_field, dict):
            kind = kind_by_field.get(field)

        if kind is not None:
            coords = np.asarray(base_dh.get_all_dof_coords(), dtype=float)
            for j, gd in enumerate(base_gdofs.tolist()):
                node_phi[j] = float(phi_eval(level_set_old, coords[int(gd)], eid=int(eid), mesh=me.mesh))
                enr = getattr(dh_old, "enrichment_map", {}).get((field, int(gd)))
                if enr is not None:
                    a_enr[j] = float(u_old[int(enr)])

        c = _OldElementCache(
            base_gdofs=base_gdofs,
            u_base=u_base,
            a_enr=a_enr,
            node_phi=node_phi,
            kind=kind,
        )
        cache[key] = c

    N = np.asarray(me._eval_scalar_basis(field, float(xi), float(eta)), dtype=float).ravel()
    if N.size != c.u_base.size:
        raise ValueError(f"Basis mismatch for field '{field}': {N.size} != {c.u_base.size}")

    uh = float(N @ c.u_base)
    if c.kind is None:
        return uh

    # Short-circuit: no enriched DOFs present on this element.
    if not np.any(c.a_enr):
        return uh

    phi_x = float(phi_eval(level_set_old, np.asarray(x_phys, dtype=float), eid=int(eid), mesh=me.mesh))
    if c.kind == "heaviside":
        Hx = _heaviside(phi_x)
        Hi = np.asarray([_heaviside(p) for p in c.node_phi], dtype=float)
        alpha = Hx - Hi
    elif c.kind == "abs":
        alpha = abs(phi_x) - np.abs(c.node_phi)
    else:
        raise ValueError(f"Unknown enrichment kind '{c.kind}' for field '{field}'.")

    uh += float((N * alpha) @ c.a_enr)
    return uh


def l2_project_moving_interface(
    *,
    dh_old,
    u_old: np.ndarray,
    level_set_old,
    dh_new,
    level_set_new,
    field: str,
    q: int = 4,
    backend: str = "python",
) -> np.ndarray:
    """
    L2-project a scalar field from an old (possibly XFEM-enriched) space to a new
    (possibly XFEM-enriched) space when the level set (interface) changes.

    Solves: find u_new such that for all v_new,
        ∫ (u_new - u_old) v_new dx = 0
    using the new interface split Ω=Ω⁺∪Ω⁻ (dx(side='+') + dx(side='-')).
    """
    field = str(field)
    if backend != "python":
        raise NotImplementedError("Moving-interface projection currently supports backend='python' only.")

    base_new = getattr(dh_new, "base", dh_new)
    me_new = base_new.mixed_element
    mesh = me_new.mesh

    # Assemble mass matrix in the new space.
    u = TrialFunction(field, dof_handler=dh_new)
    v = TestFunction(field, dof_handler=dh_new)
    dx_pos = dx(level_set=level_set_new, metadata={"side": "+", "q": int(q)})
    dx_neg = dx(level_set=level_set_new, metadata={"side": "-", "q": int(q)})
    M, _ = assemble_form(Equation((u * v) * dx_pos + (u * v) * dx_neg, None), dof_handler=dh_new, backend="python")
    if not sp.issparse(M):
        M = sp.csr_matrix(M)
    M = M.tocsr()

    # Assemble RHS by quadrature in Python, evaluating the old FE function at QPs.
    b = np.zeros(int(getattr(dh_new, "total_dofs")), dtype=float)
    old_cache: dict[tuple[int, str], _OldElementCache] = {}

    # Cache old/new cut sets (avoid relying on mutable mesh tag state).
    _, _, cut_new_ids = mesh.classify_elements(level_set_new)
    _, _, cut_old_ids = mesh.classify_elements(level_set_old)
    cut_new_set = {int(e) for e in np.asarray(cut_new_ids, dtype=int).tolist()}
    cut_old_set = {int(e) for e in np.asarray(cut_old_ids, dtype=int).tolist()}

    # Reference quadrature for full elements on each side.
    from pycutfem.integration.quadrature import volume as vol_rule
    from pycutfem.utils.bitset import BitSet

    qp_ref, _ = vol_rule(mesh.element_type, int(q))
    qp_ref = np.asarray(qp_ref, dtype=float)

    # Geometric factors (physical QPs + physical weights) for full elements.
    geo = base_new.precompute_geometric_factors(int(q), level_set=None, reuse=True)
    qp_phys_all = np.asarray(geo["qp_phys"], dtype=float)
    qw_all = np.asarray(geo["qw"], dtype=float)

    # New-basis cache per element: (gdofs, H_i at nodes, kind, is_cut_new)
    new_cache: dict[int, tuple[np.ndarray, np.ndarray, str | None, bool]] = {}

    def _new_basis_and_dofs(eid: int, xi: float, eta: float, x_phys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ent = new_cache.get(int(eid))
        if ent is None:
            base_gdofs = np.asarray(base_new.element_maps[field][int(eid)], dtype=int)
            coords = np.asarray(base_new.get_all_dof_coords(), dtype=float)
            node_phi = np.asarray(
                [float(phi_eval(level_set_new, coords[int(gd)], eid=int(eid), mesh=mesh)) for gd in base_gdofs],
                dtype=float,
            )
            H_i = np.asarray([_heaviside(p) for p in node_phi], dtype=float)
            spec = getattr(dh_new, "enrichment_spec", None)
            kind_by_field = getattr(spec, "kind_by_field", None) if spec is not None else None
            kind = kind_by_field.get(field) if isinstance(kind_by_field, dict) else None
            is_cut_new = int(eid) in cut_new_set

            if is_cut_new and kind is not None and hasattr(dh_new, "get_elemental_dofs_xfem"):
                gdofs = np.asarray(dh_new.get_elemental_dofs_xfem(int(eid)), dtype=int)
            else:
                gdofs = np.asarray(dh_new.get_elemental_dofs(int(eid)), dtype=int)
            ent = (gdofs, H_i, kind, is_cut_new)
            new_cache[int(eid)] = ent

        gdofs, H_i, kind, is_cut_new = ent
        N = np.asarray(me_new._eval_scalar_basis(field, float(xi), float(eta)), dtype=float).ravel()
        if not (is_cut_new and kind is not None):
            return gdofs, N

        phi_x = float(phi_eval(level_set_new, np.asarray(x_phys, dtype=float), eid=int(eid), mesh=mesh))
        if kind == "heaviside":
            Hx = _heaviside(phi_x)
            alpha = Hx - H_i
        elif kind == "abs":
            raise NotImplementedError("Moving-interface projection for kind='abs' is not implemented yet.")
        else:
            raise ValueError(f"Unknown enrichment kind '{kind}' for field '{field}'.")

        return gdofs, np.concatenate([N, N * alpha], axis=0)

    # ------------------------------------------------------------------
    # (A) Elements NOT cut by the OLD interface: standard quadrature
    # ------------------------------------------------------------------
    for eid in range(int(mesh.n_elements)):
        if int(eid) in cut_old_set:
            continue
        for qid, (xi, eta) in enumerate(qp_ref.tolist()):
            w = float(qw_all[int(eid), int(qid)])
            if w == 0.0:
                continue
            x_phys = qp_phys_all[int(eid), int(qid), :]
            u_q = _eval_old_scalar_on_element(
                dh_old=dh_old,
                u_old=u_old,
                field=field,
                eid=int(eid),
                xi=float(xi),
                eta=float(eta),
                x_phys=x_phys,
                level_set_old=level_set_old,
                cache=old_cache,
            )
            gdofs, phi_new = _new_basis_and_dofs(int(eid), float(xi), float(eta), x_phys)
            np.add.at(b, gdofs, w * u_q * phi_new)

    # ------------------------------------------------------------------
    # (B) Elements cut by the OLD interface: cut quadrature on Ω⁺/Ω⁻ (old)
    # ------------------------------------------------------------------
    if cut_old_set:
        cut_old_mask = np.zeros((int(mesh.n_elements),), dtype=bool)
        for e in cut_old_set:
            cut_old_mask[int(e)] = True
        cut_old_bs = BitSet(cut_old_mask)

        base_old = getattr(dh_old, "base", dh_old)
        req_derivs = {(0, 0)}
        for side_old in ("+", "-"):
            geo_cut_old = base_old.precompute_cut_volume_factors(
                cut_old_bs,
                int(q),
                req_derivs,
                level_set_old,
                side=side_old,
                reuse=True,
                deformation=None,
            )
            eids = np.asarray(geo_cut_old.get("eids", []), dtype=int)
            if eids.size == 0:
                continue
            qp_phys = np.asarray(geo_cut_old["qp_phys"], dtype=float)
            qw = np.asarray(geo_cut_old["qw"], dtype=float)
            qref = np.asarray(geo_cut_old.get("qref"), dtype=float)

            nE, nQ = int(qw.shape[0]), int(qw.shape[1])
            for ie in range(nE):
                eid = int(eids[ie])
                for iq in range(nQ):
                    w = float(qw[ie, iq])
                    if w == 0.0:
                        continue
                    xi, eta = float(qref[ie, iq, 0]), float(qref[ie, iq, 1])
                    x_phys = qp_phys[ie, iq, :]
                    u_q = _eval_old_scalar_on_element(
                        dh_old=dh_old,
                        u_old=u_old,
                        field=field,
                        eid=eid,
                        xi=xi,
                        eta=eta,
                        x_phys=x_phys,
                        level_set_old=level_set_old,
                        cache=old_cache,
                    )
                    gdofs, phi_new = _new_basis_and_dofs(eid, xi, eta, x_phys)
                    np.add.at(b, gdofs, w * u_q * phi_new)

    # Solve M u_new = b.
    u_new = spla.spsolve(M, b)
    return np.asarray(u_new, dtype=float)
