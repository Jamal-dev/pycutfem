from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from pycutfem.core.mesh import Mesh
from pycutfem.integration.quadrature import gauss_legendre, gauss_lobatto

from .interface import TraceLinkInterface


@dataclass(frozen=True, slots=True)
class TraceStationEntity2D:
    """One 2D station-based trace/fracture entity.

    The entity stores the two FE traces explicitly: station coordinates and
    global DOFs on the negative and positive sides.  This class deliberately
    does not allocate or enrich DOFs.  Fracture insertion therefore remains
    explicit: the caller must provide the side DOFs that the inserted trace
    should couple.
    """

    entity_id: int
    negative_coords: Sequence[Sequence[float]]
    positive_coords: Sequence[Sequence[float]]
    negative_dofs: Mapping[str, Sequence[int]]
    positive_dofs: Mapping[str, Sequence[int]]
    owner_neg_id: int = -1
    owner_pos_id: int = -1
    state_source_id: int | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        entity_id = int(self.entity_id)
        neg = _as_point_table("negative_coords", self.negative_coords)
        pos = _as_point_table("positive_coords", self.positive_coords)
        if neg.shape != pos.shape:
            raise ValueError(
                "TraceStationEntity2D negative_coords and positive_coords must have the same shape."
            )
        if int(neg.shape[0]) < 2:
            raise ValueError("TraceStationEntity2D requires at least two stations.")
        _validate_station_geometry(neg, pos)

        neg_dofs = _normalize_dof_map("negative_dofs", self.negative_dofs, int(neg.shape[0]))
        pos_dofs = _normalize_dof_map("positive_dofs", self.positive_dofs, int(neg.shape[0]))
        if set(neg_dofs) != set(pos_dofs):
            raise ValueError(
                "TraceStationEntity2D negative_dofs and positive_dofs must contain the same field names."
            )

        object.__setattr__(self, "entity_id", entity_id)
        object.__setattr__(self, "negative_coords", tuple(map(tuple, neg.tolist())))
        object.__setattr__(self, "positive_coords", tuple(map(tuple, pos.tolist())))
        object.__setattr__(self, "negative_dofs", neg_dofs)
        object.__setattr__(self, "positive_dofs", pos_dofs)
        object.__setattr__(self, "owner_neg_id", int(self.owner_neg_id))
        object.__setattr__(self, "owner_pos_id", int(self.owner_pos_id))
        if self.state_source_id is not None:
            object.__setattr__(self, "state_source_id", int(self.state_source_id))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def n_stations(self) -> int:
        return int(np.asarray(self.negative_coords, dtype=float).shape[0])

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(self.negative_dofs.keys())

    def negative_coords_array(self) -> np.ndarray:
        return np.asarray(self.negative_coords, dtype=float)

    def positive_coords_array(self) -> np.ndarray:
        return np.asarray(self.positive_coords, dtype=float)

    def midpoint_coords(self) -> np.ndarray:
        return 0.5 * (self.negative_coords_array() + self.positive_coords_array())


@dataclass(frozen=True, slots=True)
class TraceFractureNetwork2D:
    """Fixed-mesh trace-fracture network.

    A network is only a topology/table description.  The bulk mesh is not
    remeshed when entities are inserted or extended; new entities are appended
    as trace-link rows and later assembled with ``dTraceLink``.
    """

    mesh: Mesh
    entities: Sequence[TraceStationEntity2D] = ()
    field_names: Sequence[str] | None = None
    name: str = "trace_fracture_network_2d"

    def __post_init__(self) -> None:
        if self.mesh is None:
            raise ValueError("TraceFractureNetwork2D requires a mesh.")
        entities = tuple(self.entities or ())
        ids = [int(e.entity_id) for e in entities]
        if len(ids) != len(set(ids)):
            raise ValueError(f"TraceFractureNetwork2D entity ids must be unique, got {ids}.")

        if self.field_names is None:
            field_names = entities[0].field_names if entities else ()
        else:
            field_names = tuple(str(name) for name in self.field_names)
        if len(field_names) != len(set(field_names)):
            raise ValueError(f"TraceFractureNetwork2D field_names must be unique, got {field_names}.")

        if entities:
            n_stations = entities[0].n_stations
            for entity in entities:
                if entity.n_stations != n_stations:
                    raise ValueError(
                        "TraceFractureNetwork2D requires one station count per network. "
                        "Build separate TraceLinkInterface objects for mixed station counts."
                    )
                if tuple(entity.field_names) != field_names:
                    raise ValueError(
                        "TraceFractureNetwork2D entity field order mismatch: "
                        f"expected {field_names}, got {entity.field_names} for entity {entity.entity_id}."
                    )

        object.__setattr__(self, "entities", entities)
        object.__setattr__(self, "field_names", field_names)
        object.__setattr__(self, "name", str(self.name))

    @property
    def n_entities(self) -> int:
        return len(self.entities)

    @property
    def entity_ids(self) -> tuple[int, ...]:
        return tuple(int(entity.entity_id) for entity in self.entities)

    def insert(self, entity: TraceStationEntity2D) -> "TraceFractureNetwork2D":
        """Return a new network with ``entity`` appended."""

        if int(entity.entity_id) in set(self.entity_ids):
            raise ValueError(f"Trace fracture entity id {entity.entity_id} already exists.")
        field_names = tuple(self.field_names or entity.field_names)
        if tuple(entity.field_names) != field_names:
            raise ValueError(
                f"Inserted trace entity fields {entity.field_names} do not match network fields {field_names}."
            )
        return TraceFractureNetwork2D(
            mesh=self.mesh,
            entities=(*self.entities, entity),
            field_names=field_names,
            name=self.name,
        )

    def replace(
        self,
        *,
        remove_entity_ids: Sequence[int] = (),
        add_entities: Sequence[TraceStationEntity2D] = (),
    ) -> "TraceFractureNetwork2D":
        """Return a new network with selected entities removed and new ones inserted."""

        remove = {int(entity_id) for entity_id in remove_entity_ids}
        kept = tuple(entity for entity in self.entities if int(entity.entity_id) not in remove)
        out = TraceFractureNetwork2D(
            mesh=self.mesh,
            entities=kept,
            field_names=tuple(self.field_names or ()),
            name=self.name,
        )
        for entity in add_entities:
            out = out.insert(entity)
        return out

    def make_tip_extension(
        self,
        *,
        parent_entity_id: int,
        new_entity_id: int,
        negative_dofs: Mapping[str, Sequence[int]],
        positive_dofs: Mapping[str, Sequence[int]],
        length: float,
        width: float | None = None,
        direction: Sequence[float] | None = None,
        tip: str = "end",
        state_source_id: int | None = None,
        owner_neg_id: int | None = None,
        owner_pos_id: int | None = None,
    ) -> TraceStationEntity2D:
        """Create a new two-station extension from an existing fracture tip.

        The returned entity is not inserted automatically.  Explicit DOF maps
        are required because fixed-mesh trace-FEM propagation must not hide DOF
        creation behind a topology update.
        """

        parent = self.entity(parent_entity_id)
        length = float(length)
        if length <= 0.0:
            raise ValueError("Trace fracture extension length must be positive.")
        tip_key = str(tip).strip().lower()
        if tip_key not in {"start", "end"}:
            raise ValueError("Trace fracture extension tip must be 'start' or 'end'.")

        neg = parent.negative_coords_array()
        pos = parent.positive_coords_array()
        mids = 0.5 * (neg + pos)
        tangent = mids[-1] - mids[0]
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm <= 1.0e-14:
            raise ValueError(f"Parent trace entity {parent_entity_id} has zero tangent length.")
        tangent = tangent / tangent_norm
        if tip_key == "start":
            tangent = -tangent
            tip_mid = mids[0]
            side_gap = pos[0] - neg[0]
        else:
            tip_mid = mids[-1]
            side_gap = pos[-1] - neg[-1]

        if direction is not None:
            tangent = np.asarray(direction, dtype=float).reshape(2)
            tangent_norm = float(np.linalg.norm(tangent))
            if tangent_norm <= 1.0e-14:
                raise ValueError("Trace fracture extension direction must be nonzero.")
            tangent = tangent / tangent_norm

        side_width = float(np.linalg.norm(side_gap)) if width is None else float(width)
        if side_width <= 0.0:
            raise ValueError("Trace fracture extension width must be positive.")
        normal = np.asarray([-tangent[1], tangent[0]], dtype=float)
        if float(np.dot(normal, side_gap)) < 0.0:
            normal = -normal

        end_mid = tip_mid + length * tangent
        half = 0.5 * side_width
        neg_coords = np.vstack((tip_mid - half * normal, end_mid - half * normal))
        pos_coords = np.vstack((tip_mid + half * normal, end_mid + half * normal))
        return TraceStationEntity2D(
            entity_id=int(new_entity_id),
            negative_coords=neg_coords,
            positive_coords=pos_coords,
            negative_dofs=negative_dofs,
            positive_dofs=positive_dofs,
            owner_neg_id=parent.owner_neg_id if owner_neg_id is None else int(owner_neg_id),
            owner_pos_id=parent.owner_pos_id if owner_pos_id is None else int(owner_pos_id),
            state_source_id=parent.entity_id if state_source_id is None else int(state_source_id),
        )

    def entity(self, entity_id: int) -> TraceStationEntity2D:
        for entity in self.entities:
            if int(entity.entity_id) == int(entity_id):
                return entity
        raise KeyError(f"Trace fracture entity id {entity_id} does not exist.")

    def to_trace_link_interface(self, *, quadrature: str = "lobatto") -> TraceLinkInterface:
        """Build a backend-ready ``TraceLinkInterface`` for ``dTraceLink``."""

        if not self.entities:
            xi, w_ref = _reference_quadrature(2, quadrature)
            return TraceLinkInterface(
                mesh=self.mesh,
                factors=_empty_factors(xi, w_ref),
                name=self.name,
            )

        n_entities = len(self.entities)
        n_stations = int(self.entities[0].n_stations)
        field_names = tuple(self.field_names or ())
        n_fields = len(field_names)
        if n_fields == 0:
            raise ValueError("TraceFractureNetwork2D needs at least one field to build a trace-link domain.")

        neg = np.asarray([entity.negative_coords_array() for entity in self.entities], dtype=float)
        pos = np.asarray([entity.positive_coords_array() for entity in self.entities], dtype=float)
        mids = 0.5 * (neg + pos)
        station_s, tangents, normals = _station_geometry_batch(neg, pos)
        xi, w_ref = _reference_quadrature(n_stations, quadrature)
        q_s, q_w = _quadrature_on_station_intervals(station_s, xi, w_ref)
        phi, dphi_ds = _lagrange_basis_and_derivative_batch(station_s, q_s)
        n_q = int(q_s.shape[1])
        qp_phys = mids[:, 0:1, :] + q_s[:, :, None] * tangents[:, None, :]
        gdofs_map = self._gdofs_map(field_names)

        factors: dict[str, object] = {
            "trace_kind": "fracture_station",
            "eids": np.arange(n_entities, dtype=np.int32),
            "trace_entity_id": np.asarray(self.entity_ids, dtype=np.int32),
            "trace_state_source_id": np.asarray(
                [-1 if entity.state_source_id is None else int(entity.state_source_id) for entity in self.entities],
                dtype=np.int32,
            ),
            "qref": np.asarray(xi, dtype=float).reshape(-1, 1),
            "qp_ref": np.asarray(xi, dtype=float).reshape(-1, 1),
            "qw_ref": np.asarray(w_ref, dtype=float),
            "qp_phys": np.asarray(qp_phys, dtype=float),
            "qw": np.asarray(q_w, dtype=float),
            "normals": np.broadcast_to(normals[:, None, :], (n_entities, n_q, 2)).copy(),
            "xi_pos": np.zeros((n_entities, n_q), dtype=float),
            "eta_pos": np.zeros((n_entities, n_q), dtype=float),
            "xi_neg": np.zeros((n_entities, n_q), dtype=float),
            "eta_neg": np.zeros((n_entities, n_q), dtype=float),
            "gdofs_map": gdofs_map,
            "J_inv_pos": _identity_jacobian_batch(n_entities, n_q),
            "J_inv_neg": _identity_jacobian_batch(n_entities, n_q),
            "J_inv": _identity_jacobian_batch(n_entities, n_q),
            "detJ_pos": np.ones((n_entities, n_q), dtype=float),
            "detJ_neg": np.ones((n_entities, n_q), dtype=float),
            "detJ": np.ones((n_entities, n_q), dtype=float),
            "phis": np.zeros((n_entities, n_q), dtype=float),
            "h_arr": _mesh_element_lengths(self.mesh),
            "entity_kind": "edge",
            "domain": "trace_link",
            "domain_type": "trace_link",
            "owner_pos_id": np.asarray([entity.owner_pos_id for entity in self.entities], dtype=np.int32),
            "owner_neg_id": np.asarray([entity.owner_neg_id for entity in self.entities], dtype=np.int32),
            "owner_id": np.asarray([entity.owner_pos_id for entity in self.entities], dtype=np.int32),
            "qstate_entity_id": np.arange(n_entities, dtype=np.int32),
            "qstate_owner_id": np.arange(n_entities, dtype=np.int32),
        }
        _add_field_tables(
            factors,
            field_names=field_names,
            phi=phi,
            dphi_ds=dphi_ds,
            tangents=tangents,
            n_stations=n_stations,
            n_fields=n_fields,
            gdofs_map=gdofs_map,
        )
        return TraceLinkInterface(mesh=self.mesh, factors=factors, name=self.name)

    def _gdofs_map(self, field_names: tuple[str, ...]) -> np.ndarray:
        rows: list[list[int]] = []
        for entity in self.entities:
            row: list[int] = []
            for side in ("negative", "positive"):
                dofs = entity.negative_dofs if side == "negative" else entity.positive_dofs
                for station in range(entity.n_stations):
                    for field_name in field_names:
                        row.append(int(dofs[field_name][station]))
            rows.append(row)
        return np.asarray(rows, dtype=np.int64)


@dataclass(frozen=True, slots=True)
class TraceFracturePropagationSettings2D:
    """Small deterministic propagation criterion for trace-fracture planning."""

    damage_threshold: float
    propagation_length: float
    propagation_width: float
    correction_tolerance: float = 0.0

    @classmethod
    def from_kratos_fractures_data(cls, data: Mapping[str, object]) -> "TraceFracturePropagationSettings2D":
        """Create settings from a Kratos ``FracturesData.json`` dictionary."""

        fracture_data = data.get("fracture_data", data)
        if not isinstance(fracture_data, Mapping):
            raise ValueError("FracturesData must contain a mapping under 'fracture_data'.")
        required = ("propagation_damage", "propagation_length", "propagation_width")
        missing = [key for key in required if key not in fracture_data]
        if missing:
            raise ValueError(f"FracturesData fracture_data is missing required keys: {missing}.")
        return cls(
            damage_threshold=float(fracture_data["propagation_damage"]),
            propagation_length=float(fracture_data["propagation_length"]),
            propagation_width=float(fracture_data["propagation_width"]),
            correction_tolerance=float(fracture_data.get("correction_tolerance", 0.0)),
        )

    def __post_init__(self) -> None:
        if float(self.damage_threshold) < 0.0:
            raise ValueError("damage_threshold must be nonnegative.")
        if float(self.propagation_length) <= 0.0:
            raise ValueError("propagation_length must be positive.")
        if float(self.propagation_width) <= 0.0:
            raise ValueError("propagation_width must be positive.")


@dataclass(frozen=True, slots=True)
class TraceFractureExtensionPlan2D:
    """Topology-only plan for extending one trace-fracture tip."""

    parent_entity_id: int
    tip: str
    start: tuple[float, float]
    end: tuple[float, float]
    negative_coords: tuple[tuple[float, float], tuple[float, float]]
    positive_coords: tuple[tuple[float, float], tuple[float, float]]
    max_damage: float


def plan_fracture_extensions_from_damage(
    network: TraceFractureNetwork2D,
    damage_state: np.ndarray,
    settings: TraceFracturePropagationSettings2D,
    *,
    tip: str = "end",
) -> tuple[TraceFractureExtensionPlan2D, ...]:
    """Return deterministic extension plans for entities above the damage limit.

    The function creates geometry plans only.  A caller still has to allocate or
    select the side DOFs for each planned extension and call
    ``TraceFractureNetwork2D.insert`` with the resulting entity.
    """

    damage = np.asarray(damage_state, dtype=float)
    if damage.ndim < 1 or int(damage.shape[0]) != network.n_entities:
        raise ValueError(
            "damage_state first dimension must match the number of trace-fracture entities "
            f"({network.n_entities}), got {damage.shape}."
        )
    tip_key = str(tip).strip().lower()
    if tip_key not in {"start", "end"}:
        raise ValueError("tip must be 'start' or 'end'.")

    plans: list[TraceFractureExtensionPlan2D] = []
    for idx, entity in enumerate(network.entities):
        max_damage = float(np.max(damage[idx]))
        if max_damage < float(settings.damage_threshold):
            continue
        neg = entity.negative_coords_array()
        pos = entity.positive_coords_array()
        mids = 0.5 * (neg + pos)
        tangent = mids[-1] - mids[0]
        tangent /= max(float(np.linalg.norm(tangent)), 1.0e-300)
        if tip_key == "start":
            tangent = -tangent
            start_mid = mids[0]
            side_gap = pos[0] - neg[0]
        else:
            start_mid = mids[-1]
            side_gap = pos[-1] - neg[-1]
        normal = np.asarray([-tangent[1], tangent[0]], dtype=float)
        if float(np.dot(normal, side_gap)) < 0.0:
            normal = -normal
        end_mid = start_mid + float(settings.propagation_length) * tangent
        half = 0.5 * float(settings.propagation_width)
        neg_coords = np.vstack((start_mid - half * normal, end_mid - half * normal))
        pos_coords = np.vstack((start_mid + half * normal, end_mid + half * normal))
        plans.append(
            TraceFractureExtensionPlan2D(
                parent_entity_id=int(entity.entity_id),
                tip=tip_key,
                start=(float(start_mid[0]), float(start_mid[1])),
                end=(float(end_mid[0]), float(end_mid[1])),
                negative_coords=(
                    (float(neg_coords[0, 0]), float(neg_coords[0, 1])),
                    (float(neg_coords[1, 0]), float(neg_coords[1, 1])),
                ),
                positive_coords=(
                    (float(pos_coords[0, 0]), float(pos_coords[0, 1])),
                    (float(pos_coords[1, 0]), float(pos_coords[1, 1])),
                ),
                max_damage=max_damage,
            )
        )
    return tuple(plans)


def transfer_trace_quadrature_state(
    old_network: TraceFractureNetwork2D,
    new_network: TraceFractureNetwork2D,
    old_state: np.ndarray,
    *,
    default: float | np.ndarray = 0.0,
) -> np.ndarray:
    """Transfer entity-indexed quadrature state across a topology update.

    Existing entity ids are copied exactly.  New entities copy from
    ``state_source_id`` when it points to an old entity; otherwise they are
    initialized from ``default``.  Only the leading entity axis is remapped; the
    quadrature/state trailing shape must stay the same.
    """

    old = np.asarray(old_state, dtype=float)
    if old.ndim < 1 or int(old.shape[0]) != old_network.n_entities:
        raise ValueError(
            "old_state first dimension must match old_network.n_entities "
            f"({old_network.n_entities}), got {old.shape}."
        )
    trailing_shape = old.shape[1:]
    default_arr = np.broadcast_to(np.asarray(default, dtype=float), trailing_shape)
    out = np.empty((new_network.n_entities, *trailing_shape), dtype=float)
    old_index = {int(entity.entity_id): idx for idx, entity in enumerate(old_network.entities)}
    for new_idx, entity in enumerate(new_network.entities):
        source = int(entity.entity_id)
        if source not in old_index and entity.state_source_id is not None:
            source = int(entity.state_source_id)
        if source in old_index:
            out[new_idx] = old[old_index[source]]
        else:
            out[new_idx] = default_arr
    return out


def _as_point_table(name: str, raw: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(raw, dtype=float)
    if arr.ndim != 2 or int(arr.shape[1]) != 2:
        raise ValueError(f"{name} must have shape (n_stations, 2), got {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite coordinates.")
    return arr


def _normalize_dof_map(name: str, raw: Mapping[str, Sequence[int]], n_stations: int) -> dict[str, tuple[int, ...]]:
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError(f"{name} must be a non-empty mapping field_name -> station DOFs.")
    out: dict[str, tuple[int, ...]] = {}
    for field_name, values in raw.items():
        key = str(field_name)
        arr = np.asarray(values, dtype=np.int64).reshape(-1)
        if int(arr.shape[0]) != int(n_stations):
            raise ValueError(
                f"{name}[{key!r}] must have {n_stations} station DOFs, got {int(arr.shape[0])}."
            )
        if np.any(arr < 0):
            raise ValueError(f"{name}[{key!r}] contains negative DOF ids.")
        out[key] = tuple(int(v) for v in arr.tolist())
    return out


def _validate_station_geometry(neg: np.ndarray, pos: np.ndarray) -> None:
    mids = 0.5 * (neg + pos)
    tangent = mids[-1] - mids[0]
    length = float(np.linalg.norm(tangent))
    if length <= 1.0e-14:
        raise ValueError("Trace station geometry is degenerate: first and last midpoints coincide.")
    tangent = tangent / length
    station_s = np.einsum("ni,i->n", mids - mids[0:1, :], tangent, optimize=True)
    if np.any(np.diff(station_s) <= 1.0e-14):
        raise ValueError("Trace station midpoints must be strictly ordered along the tangent.")


def _station_geometry_batch(neg: np.ndarray, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mids = 0.5 * (neg + pos)
    tangent = mids[:, -1, :] - mids[:, 0, :]
    length = np.linalg.norm(tangent, axis=1)
    if np.any(length <= 1.0e-14):
        raise ValueError("Trace station geometry contains a degenerate entity.")
    tangent = tangent / length[:, None]
    side_gap = pos[:, 0, :] - neg[:, 0, :]
    normal = np.stack((-tangent[:, 1], tangent[:, 0]), axis=1)
    flip = np.einsum("ei,ei->e", normal, side_gap, optimize=True) < 0.0
    normal[flip] *= -1.0
    station_s = np.einsum("eni,ei->en", mids - mids[:, 0:1, :], tangent, optimize=True)
    if np.any(np.diff(station_s, axis=1) <= 1.0e-14):
        raise ValueError("Trace station midpoints must be strictly ordered along the tangent.")
    return station_s, tangent, normal


def _reference_quadrature(n_stations: int, quadrature: str) -> tuple[np.ndarray, np.ndarray]:
    rule = str(quadrature or "lobatto").strip().lower().replace("-", "_")
    if rule in {"lobatto", "gll", "gauss_lobatto"}:
        return gauss_lobatto(max(2, int(n_stations)))
    if rule in {"gauss", "gauss_legendre"}:
        return gauss_legendre(max(2, int(n_stations) + 1))
    raise ValueError(f"Unsupported trace-fracture quadrature rule {quadrature!r}.")


def _quadrature_on_station_intervals(
    station_s: np.ndarray,
    xi: np.ndarray,
    w_ref: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    station_s = np.asarray(station_s, dtype=float)
    xi = np.asarray(xi, dtype=float).reshape(-1)
    w_ref = np.asarray(w_ref, dtype=float).reshape(-1)
    center = 0.5 * (station_s[:, 0] + station_s[:, -1])
    half = 0.5 * (station_s[:, -1] - station_s[:, 0])
    q_s = center[:, None] + half[:, None] * xi[None, :]
    q_w = half[:, None] * w_ref[None, :]
    return q_s, q_w


def _lagrange_basis_and_derivative_batch(
    station_s: np.ndarray,
    q_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    station_s = np.asarray(station_s, dtype=float)
    q_s = np.asarray(q_s, dtype=float)
    n_entities, n_stations = station_s.shape
    n_q = int(q_s.shape[1])
    phi = np.ones((n_entities, n_q, n_stations), dtype=float)
    dphi = np.zeros_like(phi)
    for e in range(n_entities):
        nodes = station_s[e]
        for q in range(n_q):
            s = float(q_s[e, q])
            for i in range(n_stations):
                value = 1.0
                for j in range(n_stations):
                    if j == i:
                        continue
                    value *= (s - nodes[j]) / (nodes[i] - nodes[j])
                phi[e, q, i] = value
                deriv = 0.0
                for m in range(n_stations):
                    if m == i:
                        continue
                    term = 1.0 / (nodes[i] - nodes[m])
                    for j in range(n_stations):
                        if j == i or j == m:
                            continue
                        term *= (s - nodes[j]) / (nodes[i] - nodes[j])
                    deriv += term
                dphi[e, q, i] = deriv
    return phi, dphi


def _identity_jacobian_batch(n_entities: int, n_q: int) -> np.ndarray:
    eye = np.eye(2, dtype=float)
    return np.broadcast_to(eye, (int(n_entities), int(n_q), 2, 2)).copy()


def _mesh_element_lengths(mesh: Mesh) -> np.ndarray:
    n_elements = int(getattr(mesh, "n_elements", 0) or 0)
    return np.asarray([mesh.element_char_length(i) for i in range(n_elements)], dtype=float)


def _empty_factors(xi: np.ndarray, w_ref: np.ndarray) -> dict[str, object]:
    qref = np.asarray(xi, dtype=float).reshape(-1, 1)
    n_q = int(qref.shape[0])
    z1 = np.empty((0, n_q), dtype=float)
    z2 = np.empty((0, n_q, 2), dtype=float)
    z22 = np.empty((0, n_q, 2, 2), dtype=float)
    return {
        "trace_kind": "fracture_station",
        "eids": np.empty((0,), dtype=np.int32),
        "trace_entity_id": np.empty((0,), dtype=np.int32),
        "trace_state_source_id": np.empty((0,), dtype=np.int32),
        "qref": qref,
        "qp_ref": qref,
        "qw_ref": np.asarray(w_ref, dtype=float),
        "qp_phys": z2,
        "qw": z1,
        "normals": z2,
        "xi_pos": z1,
        "eta_pos": z1,
        "xi_neg": z1,
        "eta_neg": z1,
        "gdofs_map": np.empty((0, 0), dtype=np.int64),
        "J_inv_pos": z22,
        "J_inv_neg": z22,
        "J_inv": z22,
        "detJ_pos": z1,
        "detJ_neg": z1,
        "detJ": z1,
        "phis": z1,
        "h_arr": np.empty((0,), dtype=float),
        "entity_kind": "edge",
        "domain": "trace_link",
        "domain_type": "trace_link",
        "owner_pos_id": np.empty((0,), dtype=np.int32),
        "owner_neg_id": np.empty((0,), dtype=np.int32),
        "owner_id": np.empty((0,), dtype=np.int32),
        "qstate_entity_id": np.empty((0,), dtype=np.int32),
        "qstate_owner_id": np.empty((0,), dtype=np.int32),
    }


def _add_field_tables(
    factors: dict[str, object],
    *,
    field_names: tuple[str, ...],
    phi: np.ndarray,
    dphi_ds: np.ndarray,
    tangents: np.ndarray,
    n_stations: int,
    n_fields: int,
    gdofs_map: np.ndarray,
) -> None:
    n_entities = int(np.asarray(gdofs_map).shape[0])
    n_union = int(np.asarray(gdofs_map).shape[1]) if n_entities else 0
    stations = np.arange(int(n_stations), dtype=np.int32)
    neg_all = np.arange(0, int(n_stations) * int(n_fields), dtype=np.int32)
    pos_all = np.arange(
        int(n_stations) * int(n_fields),
        2 * int(n_stations) * int(n_fields),
        dtype=np.int32,
    )
    factors["neg_map"] = np.broadcast_to(neg_all, (n_entities, neg_all.size)).copy()
    factors["pos_map"] = np.broadcast_to(pos_all, (n_entities, pos_all.size)).copy()
    dphi_dx = dphi_ds * tangents[:, None, 0:1]
    dphi_dy = dphi_ds * tangents[:, None, 1:2]
    for field_index, field_name in enumerate(field_names):
        neg_map = (stations * int(n_fields) + int(field_index)).astype(np.int32)
        pos_map = (int(n_stations) * int(n_fields) + stations * int(n_fields) + int(field_index)).astype(np.int32)
        factors[f"neg_map_{field_name}"] = np.broadcast_to(neg_map, (n_entities, neg_map.size)).copy()
        factors[f"pos_map_{field_name}"] = np.broadcast_to(pos_map, (n_entities, pos_map.size)).copy()
        for side_name, side_map in (("neg", neg_map), ("pos", pos_map)):
            mask = np.zeros((n_entities, n_union), dtype=float)
            if side_map.size:
                mask[:, side_map] = 1.0
            factors[f"restrict_mask_{field_name}_{side_name}"] = mask
            factors[f"r00_{field_name}_{side_name}"] = np.asarray(phi, dtype=float)
            factors[f"r10_{field_name}_{side_name}"] = np.asarray(dphi_dx, dtype=float)
            factors[f"r01_{field_name}_{side_name}"] = np.asarray(dphi_dy, dtype=float)

        b_union = np.zeros((n_entities, int(phi.shape[1]), n_union), dtype=float)
        d10_union = np.zeros_like(b_union)
        d01_union = np.zeros_like(b_union)
        if pos_map.size:
            b_union[:, :, pos_map] = phi
            d10_union[:, :, pos_map] = dphi_dx
            d01_union[:, :, pos_map] = dphi_dy
        factors[f"b_{field_name}"] = b_union
        factors[f"d10_{field_name}"] = d10_union
        factors[f"d01_{field_name}"] = d01_union
        factors[f"g_{field_name}"] = np.stack((d10_union, d01_union), axis=-1)
