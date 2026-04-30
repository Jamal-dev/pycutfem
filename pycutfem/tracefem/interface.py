from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from pycutfem.core.mesh import Mesh


@dataclass(frozen=True, slots=True)
class TraceLinkInterface:
    """Discrete trace/link integration domain.

    ``TraceLinkInterface`` represents a set of one-dimensional trace entities
    whose test and trial values are defined by explicit quadrature/station
    tables.  It is intentionally different from ``NonMatchingInterface``:
    there is no common-refinement segment with one geometric owner on each
    side.  Each trace entity is an element-like integration row with its own
    global DOF union, sided local maps and sided trace basis tables.

    Required table contract
    -----------------------
    The ``factors`` mapping must contain at least:

    - ``eids`` with shape ``(n_entities,)``;
    - ``qref`` or ``qp_ref`` with shape ``(n_qp, 1)``;
    - ``qw_ref`` with shape ``(n_qp,)``;
    - ``qp_phys`` and ``normals`` with shape ``(n_entities, n_qp, 2)``;
    - ``qw`` with shape ``(n_entities, n_qp)``;
    - ``gdofs_map`` with shape ``(n_entities, n_trace_dofs)``;
    - ``owner_id``, ``owner_pos_id``, ``owner_neg_id`` with length
      ``n_entities`` when expressions need element-wise quantities;
    - sided maps/tables such as ``pos_map_<field>``, ``neg_map_<field>`` and
      ``r00_<field>_<side>`` when a UFL form uses ``Pos``, ``Neg`` or sided
      derivatives.

    The backend copies the table mapping and stamps ``domain_type`` as
    ``"trace_link"``.  Invalid or inconsistent shapes fail at construction or
    precompute time; no fallback geometry is inferred silently.
    """

    mesh: Mesh
    factors: Mapping[str, object]
    name: str = "trace_link"

    def __post_init__(self) -> None:
        normalized = self._normalized_factors()
        object.__setattr__(self, "factors", normalized)

    def n_entities(self) -> int:
        return int(np.asarray(self.factors["eids"], dtype=np.int32).reshape(-1).shape[0])

    def n_segments(self) -> int:
        """Compatibility alias for line-entity count."""

        return self.n_entities()

    def precomputed_factors(self) -> dict[str, object]:
        return dict(self.factors)

    def _normalized_factors(self) -> dict[str, object]:
        if self.mesh is None:
            raise ValueError("TraceLinkInterface requires a mesh.")
        factors = dict(self.factors or {})
        required = ("eids", "qw_ref", "qp_phys", "qw", "normals", "gdofs_map")
        missing = [key for key in required if key not in factors]
        if missing:
            raise ValueError(f"TraceLinkInterface factors are missing required keys: {missing}.")
        if "qref" not in factors and "qp_ref" not in factors:
            raise ValueError("TraceLinkInterface factors require 'qref' or 'qp_ref'.")

        qref = np.asarray(factors.get("qref", factors.get("qp_ref")), dtype=float)
        if qref.ndim == 1:
            qref = qref.reshape(-1, 1)
        if qref.ndim != 2 or int(qref.shape[1]) != 1:
            raise ValueError(f"TraceLinkInterface qref must have shape (n_qp, 1), got {qref.shape}.")
        qw_ref = np.asarray(factors["qw_ref"], dtype=float).reshape(-1)
        if int(qref.shape[0]) != int(qw_ref.shape[0]):
            raise ValueError(
                "TraceLinkInterface qref/qw_ref length mismatch: "
                f"{int(qref.shape[0])} vs {int(qw_ref.shape[0])}."
            )

        eids = np.asarray(factors["eids"], dtype=np.int32).reshape(-1)
        n_entities = int(eids.shape[0])
        n_qp = int(qref.shape[0])
        qp_phys = np.asarray(factors["qp_phys"], dtype=float)
        qw = np.asarray(factors["qw"], dtype=float)
        normals = np.asarray(factors["normals"], dtype=float)
        gdofs_map = np.asarray(factors["gdofs_map"], dtype=np.int64)

        _require_shape("qp_phys", qp_phys, (n_entities, n_qp, 2))
        _require_shape("qw", qw, (n_entities, n_qp))
        _require_shape("normals", normals, (n_entities, n_qp, 2))
        if gdofs_map.ndim != 2 or int(gdofs_map.shape[0]) != n_entities:
            raise ValueError(
                "TraceLinkInterface gdofs_map must have shape "
                f"(n_entities, n_trace_dofs), got {gdofs_map.shape}."
            )

        for key in ("owner_id", "owner_pos_id", "owner_neg_id", "qstate_owner_id", "qstate_entity_id"):
            if key in factors:
                arr = np.asarray(factors[key], dtype=np.int32).reshape(-1)
                if int(arr.shape[0]) != n_entities:
                    raise ValueError(
                        f"TraceLinkInterface {key} must have length {n_entities}, got {int(arr.shape[0])}."
                    )
                factors[key] = arr

        factors["eids"] = eids
        factors["qref"] = qref
        factors["qp_ref"] = qref
        factors["qw_ref"] = qw_ref
        factors["qp_phys"] = qp_phys
        factors["qw"] = qw
        factors["normals"] = normals
        factors["gdofs_map"] = gdofs_map
        factors.setdefault("owner_id", eids.copy())
        factors.setdefault("owner_pos_id", np.asarray(factors["owner_id"], dtype=np.int32).copy())
        factors.setdefault("owner_neg_id", np.asarray(factors["owner_id"], dtype=np.int32).copy())
        factors.setdefault("qstate_entity_id", eids.copy())
        factors.setdefault("qstate_owner_id", eids.copy())
        factors["entity_kind"] = "edge"
        factors["domain"] = "trace_link"
        factors["domain_type"] = "trace_link"
        return factors


def _require_shape(name: str, arr: np.ndarray, shape: tuple[int, ...]) -> None:
    if tuple(int(v) for v in arr.shape) != tuple(int(v) for v in shape):
        raise ValueError(f"TraceLinkInterface {name} must have shape {shape}, got {arr.shape}.")
