from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

from pycutfem.core.dofhandler import DofHandler, LinearConstraints
from pycutfem.core.geometry import hansbo_cut_ratio
from pycutfem.fem import transform


@dataclass(frozen=True)
class AgFEMMap:
    side: str
    ghost_eids: np.ndarray
    active_eids: np.ndarray
    root_eids: np.ndarray
    ghost_to_root: Dict[int, int]
    ghost_to_dist: Dict[int, int]


class AgFEMMapper:
    """
    Aggregation-style mapper for cut-cell conditioning.

    Phase-1 scope:
    - classify ghost cut elements based on small Hansbo cut ratio (theta)
    - pick a robust root element by BFS on element adjacency
    - optionally build linear constraints u_slave = sum_j w_j u_master_j by
      polynomial extrapolation from the root element.
    """

    def __init__(self, dh):
        """
        Parameters
        ----------
        dh:
            Either a base :class:`~pycutfem.core.dofhandler.DofHandler` or an
            :class:`~pycutfem.xfem.dofhandler.XFEMDofHandler` wrapper.
        """
        try:
            from pycutfem.xfem.dofhandler import XFEMDofHandler  # local import to avoid cycles
        except Exception:
            XFEMDofHandler = None  # type: ignore

        if not isinstance(dh, DofHandler) and (XFEMDofHandler is None or not isinstance(dh, XFEMDofHandler)):
            raise TypeError("AgFEMMapper expects a DofHandler or XFEMDofHandler.")
        if getattr(dh, "mixed_element", None) is None:
            raise RuntimeError("AgFEMMapper requires a MixedElement-backed DofHandler.")
        self.dh = dh

    def classify_ghost_elements(
        self,
        level_set,
        *,
        side: str,
        theta_min: float = 0.05,
        tol: float = 1e-12,
        defined_on=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (ghost_eids, active_eids) for the chosen side.

        - ghost: cut elements with theta(side) < theta_min
        - active: full elements on that side + cut elements with theta(side) >= theta_min
        """
        side = str(side).strip()
        if side not in {"+", "-"}:
            raise ValueError("side must be '+' or '-'.")

        mesh = self.dh.mixed_element.mesh
        mesh.classify_elements(level_set, tol=tol)
        cut_mask = np.asarray(mesh.element_bitset("cut").mask, dtype=bool)
        inside_mask = np.asarray(mesh.element_bitset("inside").mask, dtype=bool) if "inside" in getattr(mesh, "_elem_bitsets", {}) else None
        outside_mask = np.asarray(mesh.element_bitset("outside").mask, dtype=bool) if "outside" in getattr(mesh, "_elem_bitsets", {}) else None

        theta = hansbo_cut_ratio(mesh, level_set, side=side, tol=tol)
        theta_min = float(theta_min)

        ghost_mask = cut_mask & (theta < theta_min)

        if side == "+":
            full_mask = outside_mask if outside_mask is not None else (~cut_mask)
        else:
            full_mask = inside_mask if inside_mask is not None else (~cut_mask)
        active_mask = full_mask | (cut_mask & (theta >= theta_min))

        if defined_on is not None:
            try:
                allowed = np.asarray(defined_on.to_indices(), dtype=int)
            except Exception:
                arr = np.asarray(defined_on)
                allowed = np.nonzero(arr)[0].astype(int) if arr.dtype == bool else arr.astype(int)
            allow_mask = np.zeros(mesh.n_elements, dtype=bool)
            allow_mask[allowed] = True
            ghost_mask &= allow_mask
            active_mask &= allow_mask

        ghost_eids = np.nonzero(ghost_mask)[0].astype(int)
        active_eids = np.nonzero(active_mask)[0].astype(int)
        return ghost_eids, active_eids

    def build_aggregation_map(
        self,
        level_set,
        *,
        side: str,
        theta_min: float = 0.05,
        tol: float = 1e-12,
        defined_on=None,
    ) -> AgFEMMap:
        mesh = self.dh.mixed_element.mesh
        ghost_eids, active_eids = self.classify_ghost_elements(
            level_set, side=side, theta_min=theta_min, tol=tol, defined_on=defined_on
        )
        active_set = set(int(e) for e in active_eids.tolist())
        nbrs = mesh.neighbors()

        # Prefer a *full* element root (non-cut) when available; fall back to any active.
        cut_mask = np.asarray(mesh.element_bitset("cut").mask, dtype=bool)
        inside_mask = np.asarray(mesh.element_bitset("inside").mask, dtype=bool) if "inside" in getattr(mesh, "_elem_bitsets", {}) else None
        outside_mask = np.asarray(mesh.element_bitset("outside").mask, dtype=bool) if "outside" in getattr(mesh, "_elem_bitsets", {}) else None
        if str(side) == "+":
            full_mask = outside_mask if outside_mask is not None else (~cut_mask)
        else:
            full_mask = inside_mask if inside_mask is not None else (~cut_mask)

        if defined_on is not None:
            try:
                allowed = np.asarray(defined_on.to_indices(), dtype=int)
            except Exception:
                arr = np.asarray(defined_on)
                allowed = np.nonzero(arr)[0].astype(int) if arr.dtype == bool else arr.astype(int)
            allow_mask = np.zeros(mesh.n_elements, dtype=bool)
            allow_mask[allowed] = True
            full_mask &= allow_mask
        prefer_set = set(np.nonzero(full_mask)[0].astype(int).tolist())

        ghost_to_root: Dict[int, int] = {}
        ghost_to_dist: Dict[int, int] = {}

        # Map each ghost element individually to its nearest root.
        #
        # IMPORTANT: Do *not* aggregate the whole ghost component onto a single root.
        # For typical geometries (e.g. a ring of cut elements) that would constrain
        # all ghost DOFs from one interior element, leading to huge errors.
        for ge in ghost_eids.tolist():
            ge = int(ge)
            root, dist = self._bfs_find_root_with_dist(ge, prefer_set, nbrs)
            if root is None:
                root, dist = self._bfs_find_root_with_dist(ge, active_set, nbrs)
            if root is None:
                raise RuntimeError(f"AgFEM: could not find a root for ghost element {ge} on side {side}.")
            ghost_to_root[ge] = int(root)
            ghost_to_dist[ge] = int(dist if dist is not None else 0)

        return AgFEMMap(
            side=str(side),
            ghost_eids=np.asarray(ghost_eids, dtype=int),
            active_eids=np.asarray(active_eids, dtype=int),
            root_eids=np.asarray([ghost_to_root[int(ge)] for ge in ghost_eids.tolist()], dtype=int),
            ghost_to_root=ghost_to_root,
            ghost_to_dist=ghost_to_dist,
        )

    @staticmethod
    def _bfs_find_root_with_dist(
        start: int, target_set: set[int], nbrs: Sequence[Sequence[int]]
    ) -> tuple[int | None, int | None]:
        start = int(start)
        if start in target_set:
            return start, 0
        seen = {start}
        q: List[tuple[int, int]] = [(start, 0)]
        while q:
            eid, dist = q.pop(0)
            for nb in nbrs[int(eid)]:
                nb = int(nb)
                if nb in seen:
                    continue
                if nb in target_set:
                    return nb, dist + 1
                seen.add(nb)
                q.append((nb, dist + 1))
        return None, None

    @staticmethod
    def _bfs_find_root_multi(starts: Sequence[int], target_set: set[int], nbrs: Sequence[Sequence[int]]) -> int | None:
        starts_i = [int(s) for s in starts]
        for s in starts_i:
            if s in target_set:
                return int(s)
        seen = set(starts_i)
        q: List[int] = list(starts_i)
        while q:
            eid = int(q.pop(0))
            for nb in nbrs[int(eid)]:
                nb = int(nb)
                if nb in seen:
                    continue
                if nb in target_set:
                    return nb
                seen.add(nb)
                q.append(nb)
        return None

    # ------------------------------------------------------------------
    # Constraint construction (polynomial extrapolation)
    # ------------------------------------------------------------------
    def build_constraints(
        self,
        aggregation: AgFEMMap,
        *,
        fields: Sequence[str],
        ghost_dofs: Sequence[int] | None = None,
        master_selector: str = "all_non_ghost",
        exclude_dofs: Sequence[int] | None = None,
    ) -> LinearConstraints:
        """
        Build LinearConstraints E such that u_full = E @ u_master.

        For each selected ghost DOF (slave), define it as a linear combination
        of DOFs from its root element by evaluating root shape functions at the
        ghost DOF coordinate.
        """
        dh = self.dh
        mesh = dh.mixed_element.mesh
        inactive: set[int]
        if exclude_dofs is None:
            inactive = set(int(d) for d in (getattr(dh, "dof_tags", {}) or {}).get("inactive", set()))
        else:
            inactive = set(int(d) for d in exclude_dofs)

        fields = [str(f).strip() for f in fields if str(f).strip()]
        fields_set = set(fields)

        # Determine slave set
        if ghost_dofs is not None:
            slaves = np.asarray([int(d) for d in ghost_dofs if int(d) not in inactive], dtype=int)
        else:
            # Only constrain DOFs that live exclusively on ghost elements:
            # if a DOF also appears in any active element for this side,
            # it remains a master to avoid self-referential constraints.
            active_dofs: set[int] = set()
            for ae in np.asarray(getattr(aggregation, "active_eids", []), dtype=int).tolist():
                for f in fields:
                    for gd in dh.element_maps[f][int(ae)]:
                        active_dofs.add(int(gd))

            s: set[int] = set()
            for ge in np.asarray(aggregation.ghost_eids, dtype=int).tolist():
                for f in fields:
                    for gd in dh.element_maps[f][int(ge)]:
                        gdi = int(gd)
                        if gdi in inactive:
                            continue
                        if gdi not in active_dofs:
                            s.add(gdi)
            slaves = np.asarray(sorted(s), dtype=int)

        slaves_set = set(int(d) for d in slaves.tolist())

        # Determine masters
        if master_selector != "all_non_ghost":
            raise ValueError("Only master_selector='all_non_ghost' is implemented.")
        masters = np.asarray(
            [i for i in range(int(dh.total_dofs)) if (i not in slaves_set and i not in inactive)],
            dtype=int,
        )
        master_index = {int(gd): i for i, gd in enumerate(masters.tolist())}

        # Sparse matrix triplets for E
        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []

        # Identity rows for masters
        for gd in masters.tolist():
            rows.append(int(gd))
            cols.append(master_index[int(gd)])
            data.append(1.0)

        # Slaves: extrapolate from root element
        node_coords = dh.get_all_dof_coords()
        slave_to_master: Dict[int, List[Tuple[int, float]]] = {}

        def _field_for_dof(gdof: int) -> str | None:
            try:
                fld = dh._dof_to_node_map[int(gdof)][0]
                return str(fld)
            except Exception:
                return None

        def _fallback_nearest_master(
            sd: int, x: np.ndarray, root_gdofs: np.ndarray
        ) -> List[Tuple[int, float]]:
            """Robust fallback: inject the nearest *available master* DOF from the root."""
            if root_gdofs.size == 0:
                return []
            coords = np.asarray(node_coords[root_gdofs], dtype=float)
            d2 = np.sum((coords - x[None, :]) ** 2, axis=1)
            order = np.argsort(d2)
            for j in order.tolist():
                mg = int(root_gdofs[int(j)])
                if mg in master_index:
                    return [(mg, 1.0)]
            return []

        for sd in slaves.tolist():
            sd = int(sd)
            fld_sd = _field_for_dof(sd)
            if fld_sd is None or fld_sd not in fields_set:
                continue

            # Identify which ghost element/root to use for this slave DOF.
            # If a DOF is shared by multiple ghost elements, pick the nearest root
            # (fewest adjacency hops) to avoid constraining large regions from one root.
            best_root: int | None = None
            best_dist: int | None = None
            for g_eid, r_eid in aggregation.ghost_to_root.items():
                g_eid = int(g_eid)
                if sd not in dh.element_maps[fld_sd][g_eid]:
                    continue
                dist = int(getattr(aggregation, "ghost_to_dist", {}).get(g_eid, 0))
                if best_root is None or dist < int(best_dist):
                    best_root = int(r_eid)
                    best_dist = dist

            if best_root is None:
                # If caller provided ghost_dofs explicitly, allow skipping unmapped
                continue
            re = int(best_root)

            x = np.asarray(node_coords[int(sd)], dtype=float)

            root_gdofs = np.asarray(dh.element_maps[fld_sd][int(re)], dtype=int)
            weights_accum: List[Tuple[int, float]] = []
            if root_gdofs.size:
                # Evaluate basis at the ghost point in the root element reference.
                xi_eta = None
                try:
                    xi_eta = transform.inverse_mapping(mesh, int(re), x)
                except Exception:
                    xi_eta = None
                if xi_eta is not None:
                    xi, eta = xi_eta
                    Nloc = dh.mixed_element._eval_scalar_basis(fld_sd, float(xi), float(eta))
                    if Nloc.shape[0] == root_gdofs.shape[0]:
                        for j, mg in enumerate(root_gdofs.tolist()):
                            mg = int(mg)
                            col = master_index.get(mg)
                            if col is None:
                                continue
                            w = float(Nloc[int(j)])
                            if abs(w) < 1e-14:
                                continue
                            rows.append(sd)
                            cols.append(col)
                            data.append(w)
                            weights_accum.append((mg, w))

            if not weights_accum:
                weights_accum = _fallback_nearest_master(sd, x, root_gdofs)
                for mg, w in weights_accum:
                    col = master_index.get(int(mg))
                    if col is None:
                        continue
                    rows.append(sd)
                    cols.append(col)
                    data.append(float(w))

            if not weights_accum:
                raise RuntimeError(f"AgFEM: could not build constraint weights for slave DOF {sd}.")
            slave_to_master[sd] = weights_accum

        E = sp.csr_matrix((data, (rows, cols)), shape=(int(dh.total_dofs), int(masters.shape[0])))
        return LinearConstraints(E, masters, slave_to_master)
