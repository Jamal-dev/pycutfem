"""Backend-dispatched paired 2D U-Pl interface element helpers.

The local algebra mirrors the Kratos Poromechanics small-strain interface
blocks for the two-node-per-side case, but it is implemented as a paired line
trace with an arbitrary equal number of stations on both sides. The public
helpers return standard pycutfem local-assembly batches and can run through the
same ``python``, ``jit`` and ``cpp`` backend names used by the rest of the repo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import scipy.sparse as sp

from pycutfem.integration.quadrature import gauss_legendre, gauss_lobatto
from pycutfem.nonmatching.interface import NonMatchingInterface
from pycutfem.operators import CallbackLocalAssemblyOperator, LocalAssemblyWorkset
from pycutfem.tracefem import TraceLinkInterface
from pycutfem.ufl.compilers import LocalAssemblyBatch
from pycutfem.ufl.expressions import Constant, FacetNormal, Neg, Pos, grad, heaviside, pos_part
from pycutfem.ufl.forms import Equation
from pycutfem.ufl.measures import dNonmatchingInterface, dTraceLink


QuadratureRule = Literal["lobatto", "gauss"]
BackendName = Literal["python", "jit", "cpp"]
InterfacePermeabilityLaw = Literal["fracture", "link"]


def _named_constant(value, name: str | None = None, *, dim: int | None = None, preserve: bool = True):
    """Create or annotate a JIT-stable constant used by interface UFL forms."""

    if isinstance(value, Constant):
        c = value
    elif hasattr(value, "__dict__") and not isinstance(value, np.ndarray):
        return value
    else:
        c = Constant(value, dim=dim) if dim is not None else Constant(value)
    if name:
        c._jit_name = name
    if preserve:
        c._preserve_runtime_structure = True
    return c


@dataclass(frozen=True)
class UPlInterfaceMaterial2D:
    """Material data for a 2D small-strain U-Pl cohesive interface."""

    normal_stiffness: float
    shear_stiffness: float
    porosity: float
    biot_coefficient: float
    bulk_modulus_solid: float
    bulk_modulus_liquid: float
    dynamic_viscosity_liquid: float
    initial_joint_width: float
    transversal_permeability_coefficient: float = 0.0
    penalty_stiffness: float = 1.0
    density_solid: float = 0.0
    density_liquid: float = 0.0
    thickness: float = 1.0
    young_modulus: float | None = None
    critical_displacement: float | None = None
    yield_stress: float | None = None
    damage_threshold: float | None = None
    friction_coefficient: float | None = None

    @property
    def biot_modulus_inverse(self) -> float:
        alpha = float(self.biot_coefficient)
        phi = float(self.porosity)
        return (alpha - phi) / float(self.bulk_modulus_solid) + phi / float(self.bulk_modulus_liquid)

    @property
    def dynamic_viscosity_inverse(self) -> float:
        viscosity = float(self.dynamic_viscosity_liquid)
        if viscosity <= 0.0:
            raise ValueError("dynamic_viscosity_liquid must be positive.")
        return 1.0 / viscosity

    @property
    def mixture_density(self) -> float:
        phi = float(self.porosity)
        return phi * float(self.density_liquid) + (1.0 - phi) * float(self.density_solid)


@dataclass(frozen=True)
class PairedUPlInterface2D:
    """A paired 2D U-Pl interface trace.

    ``negative_*`` and ``positive_*`` arrays must use the same station order
    along the interface. For a Kratos 2D4 interface this means converting the
    native geometry order ``[negative_start, negative_end, positive_end,
    positive_start]`` into paired order ``negative=[start,end]`` and
    ``positive=[start,end]`` before constructing this object.
    """

    negative_coords: tuple[tuple[float, float], ...]
    positive_coords: tuple[tuple[float, float], ...]
    negative_dofs: tuple[tuple[int, int, int], ...]
    positive_dofs: tuple[tuple[int, int, int], ...]

    def __post_init__(self) -> None:
        n = len(self.negative_coords)
        if n < 2:
            raise ValueError("At least two interface stations are required.")
        if len(self.positive_coords) != n or len(self.negative_dofs) != n or len(self.positive_dofs) != n:
            raise ValueError("Negative and positive interface sides must have the same station count.")
        for row in (*self.negative_dofs, *self.positive_dofs):
            if len(row) != 3:
                raise ValueError("Each interface dof row must be (ux, uy, p).")

    @property
    def n_stations(self) -> int:
        return len(self.negative_coords)

    @property
    def n_nodes(self) -> int:
        return 2 * self.n_stations

    def coords_array(self) -> np.ndarray:
        return np.vstack(
            [
                np.asarray(self.negative_coords, dtype=float),
                np.asarray(self.positive_coords, dtype=float),
            ]
        )

    def local_dofs(self) -> np.ndarray:
        """Return global DOFs in local node order with per-node ``ux, uy, p``."""

        return np.asarray([*self.negative_dofs, *self.positive_dofs], dtype=int).reshape(-1)


@dataclass(frozen=True)
class PairedUPlInterfaceLocalSystem2D:
    """Local interface contribution in interleaved ``(ux, uy, p)`` order."""

    matrix: np.ndarray
    rhs: np.ndarray
    local_dofs: np.ndarray
    joint_widths: np.ndarray
    backend: str = "python"


@dataclass(frozen=True)
class PairedUPlInterfaceNewtonBatch2D:
    """Kratos Newton tangent/residual batch for paired U-Pl interfaces."""

    K_elem: np.ndarray
    R_elem: np.ndarray
    gdofs_map: np.ndarray
    state_next: np.ndarray
    joint_widths: np.ndarray
    backend: str = "python"


@dataclass(frozen=True)
class BilinearCohesiveUFLResponse2D:
    """UFL expression bundle for Kratos' 2D bilinear cohesive law."""

    D00: object
    D01: object
    D10: object
    D11: object
    stress_t: object
    stress_n: object
    equivalent: object
    loading: object
    state_next: object
    open_flag: object


@dataclass(frozen=True)
class UPlInterfaceUFLSystem2D:
    """Symbolic U-Pl interface forms on a mesh-backed paired interface."""

    lhs_form: object
    rhs_form: object
    equation: Equation
    measure: object
    interface: object
    stiffness_lhs: object
    pressure_coupling_lhs: object
    rate_coupling_lhs: object
    storage_lhs: object
    permeability_lhs: object
    rate_coupling_rhs: object
    storage_rhs: object
    permeability_law: str = "fracture"


@dataclass(frozen=True)
class UPlInterfaceKratosNewtonUFLSystem2D:
    """Symbolic Kratos Newton cohesive U-Pl interface system."""

    tangent_form: object
    residual_form: object
    equation: Equation
    measure: object
    interface: object
    response: BilinearCohesiveUFLResponse2D
    state_update_expr: object
    stiffness_tangent: object
    pressure_coupling_tangent: object
    rate_coupling_tangent: object
    storage_tangent: object
    permeability_tangent: object
    displacement_residual: object
    pressure_coupling_residual: object
    storage_residual: object
    permeability_residual: object
    permeability_law: str = "fracture"


def paired_upl_interface_to_nonmatching_interface_2d(
    interface: PairedUPlInterface2D,
    *,
    mesh,
    negative_element_ids: int | Sequence[int],
    positive_element_ids: int | Sequence[int],
) -> NonMatchingInterface:
    """Represent a paired trace as a pycutfem nonmatching-interface object.

    The trace is split between consecutive paired stations. Normals are seeded
    from the station order and are reoriented by pycutfem precompute to point
    from the negative owner element to the positive owner element.
    """

    neg = np.asarray(interface.negative_coords, dtype=float)
    pos = np.asarray(interface.positive_coords, dtype=float)
    mids = 0.5 * (neg + pos)
    p0 = np.asarray(mids[:-1], dtype=float)
    p1 = np.asarray(mids[1:], dtype=float)
    n_segments = int(p0.shape[0])
    if n_segments <= 0:
        raise ValueError("At least one paired interface segment is required.")

    neg_ids = _expand_segment_ids(negative_element_ids, n_segments, "negative_element_ids")
    pos_ids = _expand_segment_ids(positive_element_ids, n_segments, "positive_element_ids")

    normals = np.empty((n_segments, 2), dtype=float)
    side_gap = np.asarray(pos[:-1] - neg[:-1], dtype=float)
    for i in range(n_segments):
        gap = side_gap[i]
        if float(np.linalg.norm(gap)) <= 1.0e-14:
            tangent = p1[i] - p0[i]
            length = float(np.linalg.norm(tangent))
            if length <= 1.0e-14:
                raise ValueError("Degenerate paired interface segment.")
            tangent = tangent / length
            gap = np.asarray([-tangent[1], tangent[0]], dtype=float)
        normals[i] = gap / max(float(np.linalg.norm(gap)), 1.0e-300)

    edge_ids = np.arange(n_segments, dtype=np.int32)
    return NonMatchingInterface(
        mesh_neg=mesh,
        mesh_pos=mesh,
        neg_edge_ids=edge_ids,
        pos_edge_ids=edge_ids.copy(),
        neg_elem_ids=neg_ids,
        pos_elem_ids=pos_ids,
        P0=p0,
        P1=p1,
        n=normals,
        h_neg=np.ones(n_segments, dtype=float),
        h_pos=np.ones(n_segments, dtype=float),
    )


def paired_upl_interfaces_to_trace_link_interface_2d(
    interfaces: Sequence[PairedUPlInterface2D],
    *,
    mesh,
    negative_element_ids: int | Sequence[int],
    positive_element_ids: int | Sequence[int],
    quadrature: QuadratureRule = "lobatto",
) -> TraceLinkInterface:
    """Build a Trace-FEM link domain using Kratos station shape functions.

    Kratos' finite-width U-Pl interface elements are discrete trace/link
    entities: the negative and positive traces are line/station shape
    functions on paired nodes, not volume traces on a common-refinement facet.
    This helper returns a first-class ``TraceLinkInterface`` so UFL forms can
    be assembled with ``dTraceLink`` through the python, jit and cpp backends
    without coupling extra bulk-element DOFs.
    """

    interfaces = tuple(interfaces)
    if not interfaces:
        empty = np.empty((0,), dtype=np.int32)
        empty2 = np.empty((0, 2), dtype=float)
        xi, w_ref = _reference_quadrature_for_station_trace(2, quadrature)
        factors = _empty_station_trace_factors(xi, w_ref, domain_type="trace_link")
        return TraceLinkInterface(mesh=mesh, factors=factors, name="upl_station_trace")

    n_stations = interfaces[0].n_stations
    if any(iface.n_stations != n_stations for iface in interfaces):
        raise ValueError("Station-trace UFL interface batches require one station count.")
    n_entities = len(interfaces)
    neg_ids = _expand_segment_ids(negative_element_ids, n_entities, "negative_element_ids")
    pos_ids = _expand_segment_ids(positive_element_ids, n_entities, "positive_element_ids")

    neg = np.asarray([iface.negative_coords for iface in interfaces], dtype=float)
    pos = np.asarray([iface.positive_coords for iface in interfaces], dtype=float)
    mids = 0.5 * (neg + pos)
    station_s, tangents, normals, lengths = _interface_station_geometry_batch(neg, pos)
    xi, w_ref = _reference_quadrature_for_station_trace(n_stations, quadrature)
    q_s, q_w = _quadrature_on_station_intervals(station_s, n_stations, quadrature)
    phi, dphi_ds = _lagrange_basis_and_derivative_batch(station_s, q_s)
    qp_phys = mids[:, 0:1, :] + q_s[:, :, None] * tangents[:, None, :]
    n_q = int(q_s.shape[1])
    gdofs_map = np.asarray([iface.local_dofs() for iface in interfaces], dtype=np.int64)

    factors: dict[str, object] = {
        "trace_kind": "paired_station",
        "eids": np.arange(n_entities, dtype=np.int32),
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
        "h_arr": np.asarray(
            [mesh.element_char_length(int(e)) for e in range(int(mesh.n_elements))],
            dtype=float,
        ),
        "entity_kind": "edge",
        "domain": "trace_link",
        "domain_type": "trace_link",
        "owner_pos_id": pos_ids.astype(np.int32, copy=False),
        "owner_neg_id": neg_ids.astype(np.int32, copy=False),
        "owner_id": pos_ids.astype(np.int32, copy=False),
        "qstate_entity_id": np.arange(n_entities, dtype=np.int32),
        "qstate_owner_id": np.arange(n_entities, dtype=np.int32),
    }
    _add_station_trace_field_tables(
        factors,
        phi=phi,
        dphi_ds=dphi_ds,
        tangents=tangents,
        n_stations=n_stations,
        gdofs_map=gdofs_map,
    )

    return TraceLinkInterface(mesh=mesh, factors=factors, name="upl_station_trace")


def paired_upl_interfaces_to_station_trace_nonmatching_interface_2d(
    interfaces: Sequence[PairedUPlInterface2D],
    *,
    mesh,
    negative_element_ids: int | Sequence[int],
    positive_element_ids: int | Sequence[int],
    quadrature: QuadratureRule = "lobatto",
) -> NonMatchingInterface:
    """Compatibility wrapper for the previous station-trace nonmatching path.

    New code should use ``paired_upl_interfaces_to_trace_link_interface_2d`` and
    assemble the resulting domain with ``dTraceLink``.
    """

    trace = paired_upl_interfaces_to_trace_link_interface_2d(
        interfaces,
        mesh=mesh,
        negative_element_ids=negative_element_ids,
        positive_element_ids=positive_element_ids,
        quadrature=quadrature,
    )
    factors = trace.precomputed_factors()
    factors["domain"] = "nonmatching_interface"
    factors["domain_type"] = "nonmatching_interface"
    n_entities = trace.n_entities()
    empty_ids = np.arange(n_entities, dtype=np.int32)
    qp_phys = np.asarray(factors["qp_phys"], dtype=float)
    normals = np.asarray(factors["normals"], dtype=float)
    return NonMatchingInterface(
        mesh_neg=mesh,
        mesh_pos=mesh,
        neg_edge_ids=empty_ids.copy(),
        pos_edge_ids=empty_ids.copy(),
        neg_elem_ids=np.asarray(factors.get("owner_neg_id", empty_ids), dtype=np.int32),
        pos_elem_ids=np.asarray(factors.get("owner_pos_id", empty_ids), dtype=np.int32),
        P0=np.asarray(qp_phys[:, 0, :], dtype=float) if n_entities else np.empty((0, 2), dtype=float),
        P1=np.asarray(qp_phys[:, -1, :], dtype=float) if n_entities else np.empty((0, 2), dtype=float),
        n=np.asarray(normals[:, 0, :], dtype=float) if n_entities else np.empty((0, 2), dtype=float),
        h_neg=np.ones(n_entities, dtype=float),
        h_pos=np.ones(n_entities, dtype=float),
        precomputed_factors=factors,
    )


def build_upl_interface_ufl_system_2d(
    *,
    u_trial,
    p_trial,
    u_test,
    p_test,
    u_prev,
    p_prev,
    u_current,
    material: UPlInterfaceMaterial2D,
    interface: object,
    dt: float,
    theta_u: float = 1.0,
    theta_p: float = 1.0,
    quadrature: QuadratureRule = "lobatto",
    quadrature_order: int | None = None,
    permeability_law: InterfacePermeabilityLaw = "fracture",
) -> UPlInterfaceUFLSystem2D:
    """Build the Kratos-style 2D U-Pl interface as one symbolic UFL system."""

    if not hasattr(dt, "__dict__") and float(dt) <= 0.0:
        raise ValueError("dt must be positive.")
    if (not hasattr(theta_u, "__dict__") and float(theta_u) <= 0.0) or (
        not hasattr(theta_p, "__dict__") and float(theta_p) <= 0.0
    ):
        raise ValueError("theta_u and theta_p must be positive.")
    if permeability_law not in {"fracture", "link"}:
        raise ValueError("permeability_law must be 'fracture' or 'link'.")

    q_rule = "gauss_lobatto" if quadrature == "lobatto" else "gauss"
    if quadrature_order is None:
        q_order = 2 if q_rule == "gauss_lobatto" else 3
    else:
        q_order = int(quadrature_order)
    if isinstance(interface, TraceLinkInterface):
        dGamma = dTraceLink(metadata={"q": q_order, "trace": interface, "quadrature": q_rule})
    else:
        dGamma = dNonmatchingInterface(metadata={"q": q_order, "interface": interface, "quadrature": q_rule})

    n = FacetNormal()
    thickness = _named_constant(material.thickness, "interface_thickness")
    alpha = _named_constant(material.biot_coefficient, "interface_biot_coef")
    invM = _named_constant(material.biot_modulus_inverse, "interface_inv_biot_modulus")
    mu_inv = _named_constant(material.dynamic_viscosity_inverse, "interface_mu_inv")
    dt_c = _named_constant(dt, "interface_dt")
    theta_u_c = _named_constant(theta_u, "interface_theta_u")
    theta_p_c = _named_constant(theta_p, "interface_theta_p")
    one = _named_constant(1.0, "interface_one", preserve=False)
    minus_one = _named_constant(-1.0, "interface_minus_one", preserve=False)
    twelve = _named_constant(12.0, "interface_twelve", preserve=False)
    velocity = one / (theta_u_c * dt_c)
    dt_pressure = one / (theta_p_c * dt_c)
    shear_stiff = _named_constant(material.shear_stiffness, "interface_shear_stiffness")
    normal_stiff = _named_constant(material.normal_stiffness, "interface_normal_stiffness")
    penalty = _named_constant(material.penalty_stiffness, "interface_penalty_stiffness")
    initial_width = _named_constant(material.initial_joint_width, "interface_initial_joint_width")
    transversal_perm = _named_constant(
        material.transversal_permeability_coefficient,
        "interface_transversal_permeability",
    )

    normal_rel = _jump_normal_component(u_current, n)
    width = pos_part(initial_width + normal_rel)
    tangential_perm = width * width / twelve
    transverse_perm = tangential_perm if permeability_law == "link" else transversal_perm
    normal_stiff_eff = normal_stiff * (one + (penalty - one) * heaviside(minus_one * normal_rel))

    jump_u_n = _jump_normal_component(u_trial, n)
    jump_v_n = _jump_normal_component(u_test, n)
    jump_u_t = _jump_tangent_component(u_trial, n)
    jump_v_t = _jump_tangent_component(u_test, n)
    jump_prev_n = _jump_normal_component(u_prev, n)

    p_avg = _avg_trace(p_trial)
    q_avg = _avg_trace(p_test)
    p_prev_avg = _avg_trace(p_prev)
    jump_p = Pos(p_trial) - Neg(p_trial)
    jump_q = Pos(p_test) - Neg(p_test)
    dp_ds = _avg_tangential_derivative(p_trial, n)
    dq_ds = _avg_tangential_derivative(p_test, n)

    stiffness_lhs = thickness * (shear_stiff * jump_u_t * jump_v_t + normal_stiff_eff * jump_u_n * jump_v_n) * dGamma
    pressure_coupling_lhs = minus_one * thickness * alpha * p_avg * jump_v_n * dGamma
    rate_coupling_lhs = thickness * velocity * alpha * q_avg * jump_u_n * dGamma
    storage_lhs = thickness * dt_pressure * invM * width * p_avg * q_avg * dGamma
    permeability_lhs = (
        thickness
        * mu_inv
        * (tangential_perm * width * dp_ds * dq_ds + transverse_perm * width * jump_p * jump_q)
        * dGamma
    )

    rate_coupling_rhs = thickness * velocity * alpha * q_avg * jump_prev_n * dGamma
    storage_rhs = thickness * dt_pressure * invM * width * p_prev_avg * q_avg * dGamma
    lhs_form = stiffness_lhs + pressure_coupling_lhs + rate_coupling_lhs + storage_lhs + permeability_lhs
    rhs_form = rate_coupling_rhs + storage_rhs
    return UPlInterfaceUFLSystem2D(
        lhs_form=lhs_form,
        rhs_form=rhs_form,
        equation=Equation(lhs_form, rhs_form),
        measure=dGamma,
        interface=interface,
        stiffness_lhs=stiffness_lhs,
        pressure_coupling_lhs=pressure_coupling_lhs,
        rate_coupling_lhs=rate_coupling_lhs,
        storage_lhs=storage_lhs,
        permeability_lhs=permeability_lhs,
        rate_coupling_rhs=rate_coupling_rhs,
        storage_rhs=storage_rhs,
        permeability_law=str(permeability_law),
    )


def build_upl_link_interface_ufl_system_2d(**kwargs) -> UPlInterfaceUFLSystem2D:
    """Build Kratos' 2D link-interface permeability law with the UFL interface form."""

    kwargs["permeability_law"] = "link"
    return build_upl_interface_ufl_system_2d(**kwargs)


def build_upl_elastic_kratos_newton_interface_ufl_system_2d(
    *,
    u_trial,
    p_trial,
    u_test,
    p_test,
    u_current,
    p_current,
    velocity_current,
    p_rate_current,
    material: UPlInterfaceMaterial2D,
    interface: object,
    dt: float,
    theta_u: float = 1.0,
    theta_p: float = 1.0,
    quadrature: QuadratureRule = "lobatto",
    quadrature_order: int | None = None,
    permeability_law: InterfacePermeabilityLaw = "fracture",
) -> UPlInterfaceKratosNewtonUFLSystem2D:
    """Build Kratos' elastic cohesive 2D U-Pl interface Newton block in UFL."""

    if not hasattr(dt, "__dict__") and float(dt) <= 0.0:
        raise ValueError("dt must be positive.")
    if (not hasattr(theta_u, "__dict__") and float(theta_u) <= 0.0) or (
        not hasattr(theta_p, "__dict__") and float(theta_p) <= 0.0
    ):
        raise ValueError("theta_u and theta_p must be positive.")
    if permeability_law not in {"fracture", "link"}:
        raise ValueError("permeability_law must be 'fracture' or 'link'.")

    q_rule = "gauss_lobatto" if quadrature == "lobatto" else "gauss"
    if quadrature_order is None:
        q_order = 2 if q_rule == "gauss_lobatto" else 3
    else:
        q_order = int(quadrature_order)
    if isinstance(interface, TraceLinkInterface):
        dGamma = dTraceLink(metadata={"q": q_order, "trace": interface, "quadrature": q_rule})
    else:
        dGamma = dNonmatchingInterface(metadata={"q": q_order, "interface": interface, "quadrature": q_rule})

    n = FacetNormal()
    thickness = _named_constant(material.thickness, "interface_elastic_newton_thickness")
    alpha = _named_constant(material.biot_coefficient, "interface_elastic_newton_biot_coef")
    invM = _named_constant(material.biot_modulus_inverse, "interface_elastic_newton_inv_biot_modulus")
    mu_inv = _named_constant(material.dynamic_viscosity_inverse, "interface_elastic_newton_mu_inv")
    dt_c = _named_constant(dt, "interface_elastic_newton_dt")
    theta_u_c = _named_constant(theta_u, "interface_elastic_newton_theta_u")
    theta_p_c = _named_constant(theta_p, "interface_elastic_newton_theta_p")
    one = _named_constant(1.0, "interface_elastic_newton_one", preserve=False)
    minus_one = _named_constant(-1.0, "interface_elastic_newton_minus_one", preserve=False)
    twelve = _named_constant(12.0, "interface_elastic_newton_twelve", preserve=False)
    initial_width = _named_constant(material.initial_joint_width, "interface_elastic_newton_initial_joint_width")
    shear_stiff = _named_constant(material.shear_stiffness, "interface_elastic_newton_shear_stiffness")
    normal_stiff = _named_constant(material.normal_stiffness, "interface_elastic_newton_normal_stiffness")
    penalty = _named_constant(material.penalty_stiffness, "interface_elastic_newton_penalty_stiffness")
    transversal_perm = _named_constant(
        material.transversal_permeability_coefficient,
        "interface_elastic_newton_transversal_permeability",
    )
    velocity_coeff = one / (theta_u_c * dt_c)
    dt_pressure_coeff = one / (theta_p_c * dt_c)

    du_t = _jump_tangent_component(u_trial, n)
    du_n = _jump_normal_component(u_trial, n)
    v_t = _jump_tangent_component(u_test, n)
    v_n = _jump_normal_component(u_test, n)
    u_current_t = _jump_tangent_component(u_current, n)
    u_current_n = _jump_normal_component(u_current, n)
    velocity_n = _jump_normal_component(velocity_current, n)

    p_avg = _avg_trace(p_trial)
    q_avg = _avg_trace(p_test)
    p_current_avg = _avg_trace(p_current)
    p_rate_avg = _avg_trace(p_rate_current)
    jump_p = Pos(p_trial) - Neg(p_trial)
    jump_q = Pos(p_test) - Neg(p_test)
    jump_p_current = Pos(p_current) - Neg(p_current)
    dp_ds = _avg_tangential_derivative(p_trial, n)
    dq_ds = _avg_tangential_derivative(p_test, n)
    dp_current_ds = _avg_tangential_derivative(p_current, n)

    width = pos_part(initial_width + u_current_n)
    k_tangential = width * width / twelve
    k_transverse = k_tangential if permeability_law == "link" else transversal_perm
    normal_stiff_eff = normal_stiff * (one + (penalty - one) * heaviside(minus_one * u_current_n))
    stress_t = shear_stiff * u_current_t
    stress_n = normal_stiff_eff * u_current_n

    stiffness_tangent = thickness * (shear_stiff * v_t * du_t + normal_stiff_eff * v_n * du_n) * dGamma
    pressure_coupling_tangent = minus_one * thickness * alpha * p_avg * v_n * dGamma
    rate_coupling_tangent = thickness * velocity_coeff * alpha * q_avg * du_n * dGamma
    storage_tangent = thickness * dt_pressure_coeff * invM * width * p_avg * q_avg * dGamma
    permeability_tangent = (
        thickness
        * mu_inv
        * (k_tangential * width * dp_ds * dq_ds + k_transverse * width * jump_p * jump_q)
        * dGamma
    )

    displacement_residual = minus_one * thickness * (stress_t * v_t + stress_n * v_n) * dGamma
    pressure_coupling_residual = thickness * alpha * (p_current_avg * v_n - q_avg * velocity_n) * dGamma
    storage_residual = minus_one * thickness * invM * width * p_rate_avg * q_avg * dGamma
    permeability_residual = (
        minus_one
        * thickness
        * mu_inv
        * (
            k_tangential * width * dp_current_ds * dq_ds
            + k_transverse * width * jump_p_current * jump_q
        )
        * dGamma
    )

    tangent_form = (
        stiffness_tangent
        + pressure_coupling_tangent
        + rate_coupling_tangent
        + storage_tangent
        + permeability_tangent
    )
    residual_form = (
        displacement_residual
        + pressure_coupling_residual
        + storage_residual
        + permeability_residual
    )
    return UPlInterfaceKratosNewtonUFLSystem2D(
        tangent_form=tangent_form,
        residual_form=residual_form,
        equation=Equation(tangent_form, residual_form),
        measure=dGamma,
        interface=interface,
        response=None,
        state_update_expr=None,
        stiffness_tangent=stiffness_tangent,
        pressure_coupling_tangent=pressure_coupling_tangent,
        rate_coupling_tangent=rate_coupling_tangent,
        storage_tangent=storage_tangent,
        permeability_tangent=permeability_tangent,
        displacement_residual=displacement_residual,
        pressure_coupling_residual=pressure_coupling_residual,
        storage_residual=storage_residual,
        permeability_residual=permeability_residual,
        permeability_law=str(permeability_law),
    )


def build_upl_kratos_newton_interface_ufl_system_2d(
    *,
    u_trial,
    p_trial,
    u_test,
    p_test,
    u_current,
    p_current,
    velocity_current,
    p_rate_current,
    state,
    material: UPlInterfaceMaterial2D,
    interface: object,
    dt: float,
    theta_u: float = 1.0,
    theta_p: float = 1.0,
    quadrature: QuadratureRule = "lobatto",
    quadrature_order: int | None = None,
    permeability_law: InterfacePermeabilityLaw = "fracture",
) -> UPlInterfaceKratosNewtonUFLSystem2D:
    """Build Kratos' nonlinear 2D U-Pl cohesive interface Newton block in UFL."""

    if not hasattr(dt, "__dict__") and float(dt) <= 0.0:
        raise ValueError("dt must be positive.")
    if (not hasattr(theta_u, "__dict__") and float(theta_u) <= 0.0) or (
        not hasattr(theta_p, "__dict__") and float(theta_p) <= 0.0
    ):
        raise ValueError("theta_u and theta_p must be positive.")
    if permeability_law not in {"fracture", "link"}:
        raise ValueError("permeability_law must be 'fracture' or 'link'.")
    _require_bilinear_cohesive_material(material)

    q_rule = "gauss_lobatto" if quadrature == "lobatto" else "gauss"
    if quadrature_order is None:
        q_order = 2 if q_rule == "gauss_lobatto" else 3
    else:
        q_order = int(quadrature_order)
    if isinstance(interface, TraceLinkInterface):
        dGamma = dTraceLink(metadata={"q": q_order, "trace": interface, "quadrature": q_rule})
    else:
        dGamma = dNonmatchingInterface(metadata={"q": q_order, "interface": interface, "quadrature": q_rule})

    n = FacetNormal()
    thickness = _named_constant(material.thickness, "interface_newton_thickness")
    alpha = _named_constant(material.biot_coefficient, "interface_newton_biot_coef")
    invM = _named_constant(material.biot_modulus_inverse, "interface_newton_inv_biot_modulus")
    mu_inv = _named_constant(material.dynamic_viscosity_inverse, "interface_newton_mu_inv")
    dt_c = _named_constant(dt, "interface_newton_dt")
    theta_u_c = _named_constant(theta_u, "interface_newton_theta_u")
    theta_p_c = _named_constant(theta_p, "interface_newton_theta_p")
    one = _named_constant(1.0, "interface_newton_one", preserve=False)
    minus_one = _named_constant(-1.0, "interface_newton_minus_one", preserve=False)
    twelve = _named_constant(12.0, "interface_newton_twelve", preserve=False)
    initial_width = _named_constant(material.initial_joint_width, "interface_newton_initial_joint_width")
    transversal_perm = _named_constant(
        material.transversal_permeability_coefficient,
        "interface_newton_transversal_permeability",
    )
    velocity_coeff = one / (theta_u_c * dt_c)
    dt_pressure_coeff = one / (theta_p_c * dt_c)

    du_t = _jump_tangent_component(u_trial, n)
    du_n = _jump_normal_component(u_trial, n)
    v_t = _jump_tangent_component(u_test, n)
    v_n = _jump_normal_component(u_test, n)
    u_current_t = _jump_tangent_component(u_current, n)
    u_current_n = _jump_normal_component(u_current, n)
    velocity_n = _jump_normal_component(velocity_current, n)

    p_avg = _avg_trace(p_trial)
    q_avg = _avg_trace(p_test)
    p_current_avg = _avg_trace(p_current)
    p_rate_avg = _avg_trace(p_rate_current)
    jump_p = Pos(p_trial) - Neg(p_trial)
    jump_q = Pos(p_test) - Neg(p_test)
    jump_p_current = Pos(p_current) - Neg(p_current)
    dp_ds = _avg_tangential_derivative(p_trial, n)
    dq_ds = _avg_tangential_derivative(p_test, n)
    dp_current_ds = _avg_tangential_derivative(p_current, n)

    width = pos_part(initial_width + u_current_n)
    k_tangential = width * width / twelve
    k_transverse = k_tangential if permeability_law == "link" else transversal_perm

    response = bilinear_cohesive_2d_ufl_response(
        material,
        tangential_jump=u_current_t,
        normal_jump=u_current_n,
        state=state,
    )

    stiffness_tangent = thickness * (
        v_t * (response.D00 * du_t + response.D01 * du_n)
        + v_n * (response.D10 * du_t + response.D11 * du_n)
    ) * dGamma
    pressure_coupling_tangent = minus_one * thickness * alpha * p_avg * v_n * dGamma
    rate_coupling_tangent = thickness * velocity_coeff * alpha * q_avg * du_n * dGamma
    storage_tangent = thickness * dt_pressure_coeff * invM * width * p_avg * q_avg * dGamma
    permeability_tangent = (
        thickness
        * mu_inv
        * (k_tangential * width * dp_ds * dq_ds + k_transverse * width * jump_p * jump_q)
        * dGamma
    )

    displacement_residual = (
        minus_one * thickness * (response.stress_t * v_t + response.stress_n * v_n) * dGamma
    )
    pressure_coupling_residual = thickness * alpha * (p_current_avg * v_n - q_avg * velocity_n) * dGamma
    storage_residual = minus_one * thickness * invM * width * p_rate_avg * q_avg * dGamma
    permeability_residual = (
        minus_one
        * thickness
        * mu_inv
        * (
            k_tangential * width * dp_current_ds * dq_ds
            + k_transverse * width * jump_p_current * jump_q
        )
        * dGamma
    )

    tangent_form = (
        stiffness_tangent
        + pressure_coupling_tangent
        + rate_coupling_tangent
        + storage_tangent
        + permeability_tangent
    )
    residual_form = (
        displacement_residual
        + pressure_coupling_residual
        + storage_residual
        + permeability_residual
    )
    return UPlInterfaceKratosNewtonUFLSystem2D(
        tangent_form=tangent_form,
        residual_form=residual_form,
        equation=Equation(tangent_form, residual_form),
        measure=dGamma,
        interface=interface,
        response=response,
        state_update_expr=response.state_next,
        stiffness_tangent=stiffness_tangent,
        pressure_coupling_tangent=pressure_coupling_tangent,
        rate_coupling_tangent=rate_coupling_tangent,
        storage_tangent=storage_tangent,
        permeability_tangent=permeability_tangent,
        displacement_residual=displacement_residual,
        pressure_coupling_residual=pressure_coupling_residual,
        storage_residual=storage_residual,
        permeability_residual=permeability_residual,
        permeability_law=str(permeability_law),
    )


def _reference_quadrature_for_station_trace(
    n_stations: int,
    quadrature: QuadratureRule,
) -> tuple[np.ndarray, np.ndarray]:
    if quadrature == "lobatto":
        return gauss_lobatto(max(2, int(n_stations)))
    if quadrature == "gauss":
        return gauss_legendre(max(2, int(n_stations) + 1))
    raise ValueError(f"Unknown interface quadrature rule '{quadrature}'.")


def _empty_station_trace_factors(
    xi: np.ndarray,
    w_ref: np.ndarray,
    *,
    domain_type: str = "nonmatching_interface",
) -> dict[str, object]:
    qref = np.asarray(xi, dtype=float).reshape(-1, 1)
    n_q = int(qref.shape[0])
    z1 = np.empty((0, n_q), dtype=float)
    z2 = np.empty((0, n_q, 2), dtype=float)
    z22 = np.empty((0, n_q, 2, 2), dtype=float)
    return {
        "trace_kind": "paired_station",
        "eids": np.empty((0,), dtype=np.int32),
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
        "domain": str(domain_type),
        "domain_type": str(domain_type),
        "owner_pos_id": np.empty((0,), dtype=np.int32),
        "owner_neg_id": np.empty((0,), dtype=np.int32),
        "owner_id": np.empty((0,), dtype=np.int32),
        "qstate_entity_id": np.empty((0,), dtype=np.int32),
        "qstate_owner_id": np.empty((0,), dtype=np.int32),
    }


def _identity_jacobian_batch(n_entities: int, n_q: int) -> np.ndarray:
    eye = np.eye(2, dtype=float)
    return np.broadcast_to(eye, (int(n_entities), int(n_q), 2, 2)).copy()


def _add_station_trace_field_tables(
    factors: dict[str, object],
    *,
    phi: np.ndarray,
    dphi_ds: np.ndarray,
    tangents: np.ndarray,
    n_stations: int,
    gdofs_map: np.ndarray,
) -> None:
    n_entities = int(np.asarray(gdofs_map).shape[0])
    n_union = int(np.asarray(gdofs_map).shape[1]) if n_entities else 0
    stations = np.arange(int(n_stations), dtype=np.int32)
    neg_all = np.arange(0, 3 * int(n_stations), dtype=np.int32)
    pos_all = np.arange(3 * int(n_stations), 6 * int(n_stations), dtype=np.int32)
    factors["neg_map"] = np.broadcast_to(neg_all, (n_entities, neg_all.size)).copy()
    factors["pos_map"] = np.broadcast_to(pos_all, (n_entities, pos_all.size)).copy()

    field_offsets = {"ux": 0, "uy": 1, "p": 2}
    dphi_dx = dphi_ds * tangents[:, None, 0:1]
    dphi_dy = dphi_ds * tangents[:, None, 1:2]
    for fld, offset in field_offsets.items():
        neg_map = (3 * stations + int(offset)).astype(np.int32)
        pos_map = (3 * int(n_stations) + 3 * stations + int(offset)).astype(np.int32)
        factors[f"neg_map_{fld}"] = np.broadcast_to(neg_map, (n_entities, neg_map.size)).copy()
        factors[f"pos_map_{fld}"] = np.broadcast_to(pos_map, (n_entities, pos_map.size)).copy()
        for side_name, side_map in (("neg", neg_map), ("pos", pos_map)):
            mask = np.zeros((n_entities, n_union), dtype=float)
            if side_map.size:
                mask[:, side_map] = 1.0
            factors[f"restrict_mask_{fld}_{side_name}"] = mask

        for side_name in ("neg", "pos"):
            factors[f"r00_{fld}_{side_name}"] = np.asarray(phi, dtype=float)
            factors[f"r10_{fld}_{side_name}"] = np.asarray(dphi_dx, dtype=float)
            factors[f"r01_{fld}_{side_name}"] = np.asarray(dphi_dy, dtype=float)

        b_union = np.zeros((n_entities, int(phi.shape[1]), n_union), dtype=float)
        d10_union = np.zeros_like(b_union)
        d01_union = np.zeros_like(b_union)
        if pos_map.size:
            b_union[:, :, pos_map] = phi
            d10_union[:, :, pos_map] = dphi_dx
            d01_union[:, :, pos_map] = dphi_dy
        factors[f"b_{fld}"] = b_union
        factors[f"d10_{fld}"] = d10_union
        factors[f"d01_{fld}"] = d01_union
        factors[f"g_{fld}"] = np.stack((d10_union, d01_union), axis=-1)


def _expand_segment_ids(raw: int | Sequence[int], n_segments: int, name: str) -> np.ndarray:
    arr = np.asarray([raw] if np.isscalar(raw) else raw, dtype=np.int32).reshape(-1)
    if arr.size == 1:
        return np.full(n_segments, int(arr[0]), dtype=np.int32)
    if arr.size != n_segments:
        raise ValueError(f"{name} must be a scalar or have one entry per interface segment.")
    return arr.astype(np.int32, copy=False)


def _avg_trace(expr):
    half = _named_constant(0.5, "interface_average_half", preserve=False)
    return half * (Pos(expr) + Neg(expr))


def _jump_normal_component(vec, normal):
    return (Pos(vec[0]) - Neg(vec[0])) * normal[0] + (Pos(vec[1]) - Neg(vec[1])) * normal[1]


def _jump_tangent_component(vec, normal):
    return (Pos(vec[0]) - Neg(vec[0])) * normal[1] - (Pos(vec[1]) - Neg(vec[1])) * normal[0]


def _tangential_derivative(expr, normal):
    return grad(expr)[0] * normal[1] - grad(expr)[1] * normal[0]


def _avg_tangential_derivative(expr, normal):
    half = _named_constant(0.5, "interface_tangent_average_half", preserve=False)
    return half * (_tangential_derivative(Pos(expr), normal) + _tangential_derivative(Neg(expr), normal))


def build_paired_upl_interface_local_system_2d(
    interface: PairedUPlInterface2D,
    material: UPlInterfaceMaterial2D,
    *,
    dt: float,
    theta_u: float = 1.0,
    theta_p: float = 1.0,
    previous_solution: np.ndarray | None = None,
    current_solution: np.ndarray | None = None,
    quadrature: QuadratureRule = "lobatto",
    backend: BackendName = "python",
    permeability_law: InterfacePermeabilityLaw = "fracture",
) -> PairedUPlInterfaceLocalSystem2D:
    """Build one local interface system using the selected backend."""

    payload = _build_interface_payload(
        [interface],
        material,
        dt=dt,
        theta_u=theta_u,
        theta_p=theta_p,
        previous_solution=previous_solution,
        current_solution=current_solution,
        quadrature=quadrature,
        permeability_law=permeability_law,
    )
    batch = _build_local_batch_from_payload(payload, n_entities=1, backend=backend, need_matrix=True)
    if batch.K_elem is None or batch.F_elem is None:
        raise RuntimeError("Interface local-system assembly did not return both matrix and RHS.")
    return PairedUPlInterfaceLocalSystem2D(
        matrix=np.asarray(batch.K_elem[0], dtype=float),
        rhs=np.asarray(batch.F_elem[0], dtype=float),
        local_dofs=np.asarray(batch.gdofs_map[0], dtype=int),
        joint_widths=np.asarray(payload["joint_widths"][0], dtype=float),
        backend=str(backend),
    )


def build_paired_upl_interface_local_batch_2d(
    interfaces: Sequence[PairedUPlInterface2D],
    material: UPlInterfaceMaterial2D,
    *,
    dt: float,
    theta_u: float = 1.0,
    theta_p: float = 1.0,
    previous_solution: np.ndarray | None = None,
    current_solution: np.ndarray | None = None,
    quadrature: QuadratureRule = "lobatto",
    backend: BackendName = "python",
    need_matrix: bool = True,
    permeability_law: InterfacePermeabilityLaw = "fracture",
) -> LocalAssemblyBatch:
    """Return a pycutfem local-assembly batch for one or more paired interfaces."""

    payload = _build_interface_payload(
        interfaces,
        material,
        dt=dt,
        theta_u=theta_u,
        theta_p=theta_p,
        previous_solution=previous_solution,
        current_solution=current_solution,
        quadrature=quadrature,
        permeability_law=permeability_law,
    )
    return _build_local_batch_from_payload(
        payload,
        n_entities=len(interfaces),
        backend=backend,
        need_matrix=need_matrix,
    )


def build_paired_upl_interface_kratos_newton_batch_2d(
    interfaces: Sequence[PairedUPlInterface2D],
    material: UPlInterfaceMaterial2D,
    *,
    dt: float,
    theta_u: float,
    theta_p: float,
    previous_solution: np.ndarray,
    current_solution: np.ndarray,
    velocity_solution: np.ndarray,
    dt_pressure_solution: np.ndarray,
    state_variables: np.ndarray | None = None,
    quadrature: QuadratureRule = "lobatto",
    permeability_law: InterfacePermeabilityLaw = "fracture",
    backend: BackendName = "python",
) -> PairedUPlInterfaceNewtonBatch2D:
    """Return Kratos-style Newton tangent and residual for paired interfaces.

    This mirrors ``UPlSmallStrainInterfaceElement<2,4>::CalculateAll``: the
    matrix is the tangent at ``current_solution`` and ``R_elem`` is the residual
    assembled on that same iterate. The absolute linear helper is kept separate
    because Kratos' bilinear cohesive law is not an absolute secant system.
    """

    if backend != "python":
        raise NotImplementedError(
            "The nonlinear paired interface residual is currently a vectorized Python local kernel. "
            "Use build_upl_interface_ufl_system_2d for backend-dispatched linear interface assembly."
        )
    _require_bilinear_cohesive_material(material)
    payload = _build_interface_payload(
        interfaces,
        material,
        dt=dt,
        theta_u=theta_u,
        theta_p=theta_p,
        previous_solution=previous_solution,
        current_solution=current_solution,
        quadrature=quadrature,
        permeability_law=permeability_law,
    )
    gdofs_map = np.asarray(payload["gdofs_map"], dtype=int)
    velocity_values = _extract_batch_values(velocity_solution, gdofs_map)
    dt_pressure_values = _extract_batch_values(dt_pressure_solution, gdofs_map)
    K_elem, R_elem, state_next, joint_widths = _assemble_kratos_newton_interface_numpy(
        payload,
        material=material,
        velocity_values=velocity_values,
        dt_pressure_values=dt_pressure_values,
        state_variables=state_variables,
    )
    return PairedUPlInterfaceNewtonBatch2D(
        K_elem=K_elem,
        R_elem=R_elem,
        gdofs_map=gdofs_map,
        state_next=state_next,
        joint_widths=joint_widths,
        backend=str(backend),
    )


def _build_local_batch_from_payload(
    payload: dict[str, np.ndarray | float | str],
    *,
    n_entities: int,
    backend: BackendName,
    need_matrix: bool,
) -> LocalAssemblyBatch:
    element_ids = np.arange(int(n_entities), dtype=np.int32)
    workset = LocalAssemblyWorkset(
        solver=None,
        coeffs=None,
        need_matrix=bool(need_matrix),
        backend=str(backend),
        element_ids=element_ids,
        gdofs_map=np.asarray(payload["gdofs_map"], dtype=int),
        payload=payload,
        metadata={"entity_kind": "paired_upl_interface_2d"},
    )
    K_elem, F_elem = _dispatch_interface_kernel(workset)
    return LocalAssemblyBatch(
        K_elem=None if K_elem is None else np.asarray(K_elem, dtype=float),
        F_elem=np.asarray(F_elem, dtype=float),
        element_ids=element_ids,
        gdofs_map=np.asarray(payload["gdofs_map"], dtype=int),
        entity_kind="paired_upl_interface_2d",
    )


def build_paired_upl_interface_operator_2d(
    interfaces: Sequence[PairedUPlInterface2D],
    material: UPlInterfaceMaterial2D,
    *,
    dt: float,
    theta_u: float = 1.0,
    theta_p: float = 1.0,
    quadrature: QuadratureRule = "lobatto",
    permeability_law: InterfacePermeabilityLaw = "fracture",
) -> CallbackLocalAssemblyOperator:
    """Create a solver runtime operator for paired U-Pl interface blocks.

    The solver's backend selects the local kernel. Per-iteration solutions are
    supplied through the coefficient dictionary:

    - ``previous_solution``: accepted previous-step global vector
    - ``current_solution``: current iterate/global vector; defaults to previous
    """

    def _workset_builder(*, solver, coeffs, need_matrix: bool):
        coeffs = dict(coeffs or {})
        payload = _build_interface_payload(
            interfaces,
            material,
            dt=dt,
            theta_u=theta_u,
            theta_p=theta_p,
            previous_solution=coeffs.get("previous_solution"),
            current_solution=coeffs.get("current_solution"),
            quadrature=quadrature,
            permeability_law=permeability_law,
        )
        return {
            "backend": str(getattr(solver, "backend", "python")),
            "element_ids": np.arange(len(interfaces), dtype=np.int32),
            "gdofs_map": np.asarray(payload["gdofs_map"], dtype=int),
            "payload": payload,
            "need_matrix": bool(need_matrix),
        }

    return CallbackLocalAssemblyOperator(
        workset_builder=_workset_builder,
        python_kernel=_python_interface_kernel,
        jit_kernel=_unsupported_standalone_backend,
        cpp_kernel=_unsupported_standalone_backend,
    )


def assemble_paired_upl_interface_2d(
    matrix,
    rhs: np.ndarray,
    interface: PairedUPlInterface2D,
    material: UPlInterfaceMaterial2D,
    *,
    dt: float,
    theta_u: float = 1.0,
    theta_p: float = 1.0,
    previous_solution: np.ndarray | None = None,
    current_solution: np.ndarray | None = None,
    quadrature: QuadratureRule = "lobatto",
    backend: BackendName = "python",
    permeability_law: InterfacePermeabilityLaw = "fracture",
) -> PairedUPlInterfaceLocalSystem2D:
    """Assemble one paired interface contribution into a global matrix/RHS."""

    local = build_paired_upl_interface_local_system_2d(
        interface,
        material,
        dt=dt,
        theta_u=theta_u,
        theta_p=theta_p,
        previous_solution=previous_solution,
        current_solution=current_solution,
        quadrature=quadrature,
        backend=backend,
        permeability_law=permeability_law,
    )
    dofs = local.local_dofs
    if sp.issparse(matrix):
        matrix[np.ix_(dofs, dofs)] = matrix[np.ix_(dofs, dofs)] + local.matrix
    else:
        rows, cols = np.meshgrid(dofs, dofs, indexing="ij")
        np.add.at(matrix, (rows, cols), local.matrix)
    np.add.at(rhs, dofs, local.rhs)
    return local


def _dispatch_interface_kernel(workset: LocalAssemblyWorkset):
    backend = str(workset.backend).lower()
    if backend == "python":
        return _python_interface_kernel(workset)
    if backend in {"jit", "cpp"}:
        raise NotImplementedError(
            "The standalone paired interface assembler is a Python reference path. "
            "Use build_upl_interface_ufl_system_2d on a mesh-backed interface for backend-dispatched "
            "python/jit/cpp assembly."
        )
    raise ValueError(f"Unsupported interface backend {workset.backend!r}.")


def _build_interface_payload(
    interfaces: Sequence[PairedUPlInterface2D],
    material: UPlInterfaceMaterial2D,
    *,
    dt: float,
    theta_u: float,
    theta_p: float,
    previous_solution: np.ndarray | None,
    current_solution: np.ndarray | None,
    quadrature: QuadratureRule,
    permeability_law: InterfacePermeabilityLaw,
) -> dict[str, np.ndarray | float | str]:
    interfaces = tuple(interfaces)
    if not interfaces:
        raise ValueError("At least one paired interface is required.")
    n_stations = interfaces[0].n_stations
    if any(iface.n_stations != n_stations for iface in interfaces):
        raise ValueError("Batched paired interfaces must share the same station count.")
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if theta_u <= 0.0 or theta_p <= 0.0:
        raise ValueError("theta_u and theta_p must be positive.")
    if permeability_law not in {"fracture", "link"}:
        raise ValueError("permeability_law must be 'fracture' or 'link'.")

    neg = np.asarray([iface.negative_coords for iface in interfaces], dtype=float)
    pos = np.asarray([iface.positive_coords for iface in interfaces], dtype=float)
    gdofs_map = np.asarray([iface.local_dofs() for iface in interfaces], dtype=np.int64)
    station_s, tangents, normals, lengths = _interface_station_geometry_batch(neg, pos)
    q_s, q_w = _quadrature_on_station_intervals(station_s, n_stations, quadrature)
    phi, dphi_ds = _lagrange_basis_and_derivative_batch(station_s, q_s)
    local_values_prev = _extract_batch_values(previous_solution, gdofs_map)
    if current_solution is None:
        local_values_current = local_values_prev
    else:
        local_values_current = _extract_batch_values(current_solution, gdofs_map)

    n_nodes = 2 * n_stations
    u_prev = local_values_prev.reshape(len(interfaces), n_nodes, 3)[:, :, :2].reshape(len(interfaces), 2 * n_nodes)
    p_prev = local_values_prev.reshape(len(interfaces), n_nodes, 3)[:, :, 2]
    u_current = local_values_current.reshape(len(interfaces), n_nodes, 3)[:, :, :2].reshape(len(interfaces), 2 * n_nodes)

    scalars = np.asarray(
        [
            1.0 / (float(theta_u) * float(dt)),
            1.0 / (float(theta_p) * float(dt)),
            float(material.biot_coefficient),
            float(material.biot_modulus_inverse),
            float(material.dynamic_viscosity_inverse),
            float(material.initial_joint_width),
            float(material.transversal_permeability_coefficient),
            float(material.shear_stiffness),
            float(material.normal_stiffness),
            float(material.penalty_stiffness),
            float(material.thickness),
        ],
        dtype=float,
    )
    initial_widths = np.full((len(interfaces), q_s.shape[1]), float(material.initial_joint_width), dtype=float)
    joint_widths = _joint_widths_from_state(
        phi=phi,
        normals=normals,
        u_current=u_current,
        initial_width=float(material.initial_joint_width),
    )
    return {
        "gdofs_map": gdofs_map,
        "phi": np.ascontiguousarray(phi, dtype=float),
        "dphi_ds": np.ascontiguousarray(dphi_ds, dtype=float),
        "weights": np.ascontiguousarray(q_w, dtype=float),
        "tangents": np.ascontiguousarray(tangents, dtype=float),
        "normals": np.ascontiguousarray(normals, dtype=float),
        "lengths": np.ascontiguousarray(lengths, dtype=float),
        "u_prev": np.ascontiguousarray(u_prev, dtype=float),
        "p_prev": np.ascontiguousarray(p_prev, dtype=float),
        "u_current": np.ascontiguousarray(u_current, dtype=float),
        "current_values": np.ascontiguousarray(local_values_current, dtype=float),
        "scalars": scalars,
        "initial_joint_widths": initial_widths,
        "joint_widths": np.ascontiguousarray(joint_widths, dtype=float),
        "quadrature": str(quadrature),
        "permeability_law": str(permeability_law),
    }


def _python_interface_kernel(workset: LocalAssemblyWorkset):
    return _assemble_interface_numpy(workset.payload, need_matrix=bool(workset.need_matrix))


def _unsupported_standalone_backend(workset: LocalAssemblyWorkset):
    raise NotImplementedError(
        f"The standalone paired interface operator does not implement backend={workset.backend!r}. "
        "Use build_upl_interface_ufl_system_2d with dNonmatchingInterface for backend-dispatched assembly."
    )


def _assemble_interface_numpy(payload, *, need_matrix: bool):
    phi = np.asarray(payload["phi"], dtype=float)
    dphi = np.asarray(payload["dphi_ds"], dtype=float)
    weights = np.asarray(payload["weights"], dtype=float)
    tangents = np.asarray(payload["tangents"], dtype=float)
    normals = np.asarray(payload["normals"], dtype=float)
    u_prev = np.asarray(payload["u_prev"], dtype=float)
    p_prev = np.asarray(payload["p_prev"], dtype=float)
    u_current = np.asarray(payload["u_current"], dtype=float)
    scalars = np.asarray(payload["scalars"], dtype=float)

    velocity, dt_pressure, alpha, invM, mu_inv, initial_width, transversal_perm, shear_stiff, normal_stiff, penalty, thickness = scalars
    n_entities, n_q, n_stations = phi.shape
    n_nodes = 2 * n_stations
    n_dofs = 3 * n_nodes

    phi_nodes = np.concatenate((phi, phi), axis=2)
    dphi_nodes = np.concatenate((dphi, dphi), axis=2)
    jump_sign = np.concatenate((-np.ones(n_stations), np.ones(n_stations))).astype(float)
    jump_phi = phi_nodes * jump_sign.reshape(1, 1, n_nodes)
    pressure_shape = 0.5 * phi_nodes
    grad0 = 0.5 * dphi_nodes
    grad1 = jump_phi

    u_current_nodes = u_current.reshape(n_entities, n_nodes, 2)
    jump_u = np.einsum("eqm,ema->eqa", jump_phi, u_current_nodes, optimize=True)
    normal_rel = np.einsum("eqa,ea->eq", jump_u, normals, optimize=True)
    joint_width = np.maximum(0.0, initial_width + normal_rel)
    normal_stiff_q = normal_stiff * np.where(normal_rel < 0.0, penalty, 1.0)
    weighted = weights * thickness

    K = np.zeros((n_entities, n_dofs, n_dofs), dtype=float) if need_matrix else None
    F = np.zeros((n_entities, n_dofs), dtype=float)

    pcols = 3 * np.arange(n_nodes, dtype=int) + 2
    storage = dt_pressure * invM * np.einsum(
        "eqi,eqj,eq->eij",
        pressure_shape,
        pressure_shape,
        joint_width * weighted,
        optimize=True,
    )
    k_tangential = joint_width * joint_width / 12.0
    permeability_law = str(payload.get("permeability_law", "fracture"))
    k_transverse = k_tangential if permeability_law == "link" else transversal_perm
    permeability = mu_inv * (
        np.einsum("eqi,eqj,eq->eij", grad0, grad0, k_tangential * joint_width * weighted, optimize=True)
        + np.einsum("eqi,eqj,eq->eij", grad1, grad1, k_transverse * joint_width * weighted, optimize=True)
    )

    if K is not None:
        K[:, pcols[:, None], pcols[None, :]] += storage + permeability

    for comp in range(2):
        ucols = 3 * np.arange(n_nodes, dtype=int) + comp
        uvec = jump_phi * normals[:, None, comp].reshape(n_entities, 1, 1)
        kup = -alpha * np.einsum("eqi,eqj,eq->eij", uvec, pressure_shape, weighted, optimize=True)
        kpu = -velocity * np.swapaxes(kup, 1, 2)
        F[:, pcols] += np.einsum("eij,ej->ei", kpu, u_prev.reshape(n_entities, n_nodes, 2)[:, :, comp], optimize=True)
        if K is not None:
            K[:, ucols[:, None], pcols[None, :]] += kup
            K[:, pcols[:, None], ucols[None, :]] += kpu

    F[:, pcols] += np.einsum("eij,ej->ei", storage, p_prev, optimize=True)

    if K is not None:
        for a in range(2):
            rows = 3 * np.arange(n_nodes, dtype=int) + a
            for b in range(2):
                cols = 3 * np.arange(n_nodes, dtype=int) + b
                d_ab = (
                    shear_stiff * tangents[:, a].reshape(n_entities, 1) * tangents[:, b].reshape(n_entities, 1)
                    + normal_stiff_q * normals[:, a].reshape(n_entities, 1) * normals[:, b].reshape(n_entities, 1)
                )
                block = np.einsum("eqi,eqj,eq->eij", jump_phi, jump_phi, d_ab * weighted, optimize=True)
                K[:, rows[:, None], cols[None, :]] += block

    return K, F


def _require_bilinear_cohesive_material(material: UPlInterfaceMaterial2D) -> None:
    missing = [
        name
        for name in (
            "young_modulus",
            "critical_displacement",
            "yield_stress",
            "damage_threshold",
            "friction_coefficient",
        )
        if getattr(material, name) is None
    ]
    if missing:
        raise ValueError(f"Bilinear cohesive interface material is missing {', '.join(missing)}.")


def bilinear_cohesive_2d_ufl_response(
    material: UPlInterfaceMaterial2D,
    *,
    tangential_jump,
    normal_jump,
    state,
) -> BilinearCohesiveUFLResponse2D:
    """Return Kratos-compatible bilinear cohesive law components as UFL expressions.

    The branch semantics intentionally mirror ``_bilinear_cohesive_2d_response``:
    the open branch is active for positive normal jump, loading uses
    ``equivalent >= state`` with the repo's strict ``Heaviside`` convention, and
    contact friction uses Kratos' non-symmetric tangent.
    """

    _require_bilinear_cohesive_material(material)
    young = _named_constant(material.young_modulus, "interface_cohesive_young")
    crit = _named_constant(material.critical_displacement, "interface_cohesive_critical_displacement")
    yield_stress = _named_constant(material.yield_stress, "interface_cohesive_yield_stress")
    threshold = _named_constant(material.damage_threshold, "interface_cohesive_damage_threshold")
    friction = _named_constant(material.friction_coefficient, "interface_cohesive_friction")
    eps = _named_constant(1.0e-20, "interface_cohesive_friction_eps", preserve=False)
    one = _named_constant(1.0, "interface_cohesive_one", preserve=False)
    zero = _named_constant(0.0, "interface_cohesive_zero", preserve=False)
    half = _named_constant(0.5, "interface_cohesive_half", preserve=False)
    minus_one = _named_constant(-1.0, "interface_cohesive_minus_one", preserve=False)

    t = tangential_jump
    n = normal_jump
    s = threshold + pos_part(state - threshold)
    s2 = s * s
    s3 = s2 * s
    crit2 = crit * crit
    crit3 = crit2 * crit

    open_flag = heaviside(n)
    contact_flag = one - open_flag
    equivalent_open = ((t * t + n * n) ** half) / crit
    equivalent_contact = (pos_part(t) + pos_part(minus_one * t)) / crit
    equivalent = open_flag * equivalent_open + contact_flag * equivalent_contact
    loading = one - heaviside(s - equivalent)

    base = yield_stress / ((one - threshold) * crit)
    secant = yield_stress * (one - s) / ((one - threshold) * crit * s)
    normal_contact = young / (threshold * crit)

    d00_loading = base * ((one - s) / s - (t * t) / (crit2 * s3))
    d11_open_loading = base * ((one - s) / s - (n * n) / (crit2 * s3))
    d01_damage = minus_one * yield_stress * t * n / ((one - threshold) * crit3 * s3)

    sign_t = heaviside(t) - heaviside(minus_one * t)
    sign_t_eps = heaviside(t - eps) - heaviside(minus_one * t - eps)

    d00_open = loading * d00_loading + (one - loading) * secant
    d01_open = loading * d01_damage
    d11_open = loading * d11_open_loading + (one - loading) * secant

    d00_contact = loading * d00_loading + (one - loading) * secant
    d01_contact = loading * (d01_damage - normal_contact * friction * sign_t_eps) + (
        one - loading
    ) * (minus_one * normal_contact * friction * sign_t)
    d11_contact = normal_contact

    D00 = open_flag * d00_open + contact_flag * d00_contact
    D01 = open_flag * d01_open + contact_flag * d01_contact
    D10 = open_flag * d01_open + contact_flag * zero
    D11 = open_flag * d11_open + contact_flag * d11_contact

    stress_open_t = secant * t
    stress_open_n = secant * n
    stress_contact_n = normal_contact * n
    stress_contact_t = secant * t - friction * stress_contact_n * sign_t
    stress_t = open_flag * stress_open_t + contact_flag * stress_contact_t
    stress_n = open_flag * stress_open_n + contact_flag * stress_contact_n

    equivalent_capped = equivalent - pos_part(equivalent - one)
    state_next = loading * equivalent_capped + (one - loading) * s

    return BilinearCohesiveUFLResponse2D(
        D00=D00,
        D01=D01,
        D10=D10,
        D11=D11,
        stress_t=stress_t,
        stress_n=stress_n,
        equivalent=equivalent,
        loading=loading,
        state_next=state_next,
        open_flag=open_flag,
    )


def _assemble_kratos_newton_interface_numpy(
    payload,
    *,
    material: UPlInterfaceMaterial2D,
    velocity_values: np.ndarray,
    dt_pressure_values: np.ndarray,
    state_variables: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    phi = np.asarray(payload["phi"], dtype=float)
    dphi = np.asarray(payload["dphi_ds"], dtype=float)
    weights = np.asarray(payload["weights"], dtype=float)
    tangents = np.asarray(payload["tangents"], dtype=float)
    normals = np.asarray(payload["normals"], dtype=float)
    current_values = np.asarray(payload["current_values"], dtype=float)
    scalars = np.asarray(payload["scalars"], dtype=float)

    (
        velocity,
        dt_pressure_coeff,
        alpha,
        invM,
        mu_inv,
        initial_width,
        transversal_perm,
        _shear_stiff,
        _normal_stiff,
        _penalty,
        thickness,
    ) = scalars
    n_entities, n_q, n_stations = phi.shape
    n_nodes = 2 * n_stations
    n_dofs = 3 * n_nodes
    if state_variables is None:
        state = np.full((n_entities, n_q), float(material.damage_threshold), dtype=float)
    else:
        state = np.asarray(state_variables, dtype=float)
        if state.shape != (n_entities, n_q):
            raise ValueError(f"state_variables must have shape {(n_entities, n_q)}, got {state.shape}.")

    phi_nodes = np.concatenate((phi, phi), axis=2)
    dphi_nodes = np.concatenate((dphi, dphi), axis=2)
    jump_sign = np.concatenate((-np.ones(n_stations), np.ones(n_stations))).astype(float)
    jump_phi = phi_nodes * jump_sign.reshape(1, 1, n_nodes)
    pressure_shape = 0.5 * phi_nodes
    grad_tangent = 0.5 * dphi_nodes
    grad_transverse = jump_phi

    current_nodes = current_values.reshape(n_entities, n_nodes, 3)
    u_current = current_nodes[:, :, :2]
    p_current = current_nodes[:, :, 2]
    velocity_nodes = np.asarray(velocity_values, dtype=float).reshape(n_entities, n_nodes, 3)[:, :, :2]
    dt_pressure_nodes = np.asarray(dt_pressure_values, dtype=float).reshape(n_entities, n_nodes, 3)[:, :, 2]

    K = np.zeros((n_entities, n_dofs, n_dofs), dtype=float)
    R = np.zeros((n_entities, n_dofs), dtype=float)
    state_next = state.copy()
    joint_widths = np.zeros((n_entities, n_q), dtype=float)
    pcols = 3 * np.arange(n_nodes, dtype=int) + 2
    ucols = np.empty(2 * n_nodes, dtype=int)
    for node in range(n_nodes):
        ucols[2 * node] = 3 * node
        ucols[2 * node + 1] = 3 * node + 1

    permeability_law = str(payload.get("permeability_law", "fracture"))
    density = material.mixture_density
    body = np.zeros(2, dtype=float)

    for e in range(n_entities):
        rotation = np.vstack((tangents[e], normals[e]))
        for q in range(n_q):
            weight = float(weights[e, q]) * thickness
            jump = jump_phi[e, q]
            n_p = pressure_shape[e, q]
            grad_np = np.column_stack((grad_tangent[e, q], grad_transverse[e, q]))
            rel_u = np.einsum("i,ia->a", jump, u_current[e], optimize=True)
            strain = rotation @ rel_u
            joint_width = max(0.0, initial_width + float(strain[1]))
            joint_widths[e, q] = joint_width
            is_open = joint_width > initial_width
            D, stress, equivalent, loading = _bilinear_cohesive_2d_response(
                material,
                tangential_jump=float(strain[0]),
                normal_jump=float(strain[1]),
                state=float(state[e, q]),
                is_open=bool(is_open),
            )
            if loading:
                state_next[e, q] = min(equivalent, 1.0)

            B = np.zeros((2, 2 * n_nodes), dtype=float)
            for node in range(n_nodes):
                B[:, 2 * node : 2 * node + 2] = jump[node] * rotation
            K[e][np.ix_(ucols, ucols)] += B.T @ D @ B * weight
            R[e, ucols] += -B.T @ stress * weight

            if density != 0.0 and np.any(body):
                for node in range(n_nodes):
                    R[e, 3 * node : 3 * node + 2] += density * jump[node] * body * joint_width * weight

            u_normal = np.zeros(2 * n_nodes, dtype=float)
            for node in range(n_nodes):
                u_normal[2 * node : 2 * node + 2] = jump[node] * normals[e]
            K_up = -alpha * np.outer(u_normal, n_p) * weight
            K[e][np.ix_(ucols, pcols)] += K_up
            K[e][np.ix_(pcols, ucols)] += -velocity * K_up.T

            pressure_gp = float(np.dot(n_p, p_current[e]))
            normal_velocity_gp = float(np.dot(u_normal, velocity_nodes[e].reshape(2 * n_nodes)))
            R[e, ucols] += alpha * u_normal * pressure_gp * weight
            R[e, pcols] += -alpha * n_p * normal_velocity_gp * weight

            storage = invM * np.outer(n_p, n_p) * joint_width * weight
            K[e][np.ix_(pcols, pcols)] += dt_pressure_coeff * storage
            R[e, pcols] += -storage @ dt_pressure_nodes[e]

            k_tangential = joint_width * joint_width / 12.0
            k_transverse = k_tangential if permeability_law == "link" else transversal_perm
            local_perm = np.diag([k_tangential, k_transverse])
            permeability = mu_inv * (grad_np @ local_perm @ grad_np.T) * joint_width * weight
            K[e][np.ix_(pcols, pcols)] += permeability
            R[e, pcols] += -permeability @ p_current[e]

    return K, R, state_next, joint_widths


def _bilinear_cohesive_2d_response(
    material: UPlInterfaceMaterial2D,
    *,
    tangential_jump: float,
    normal_jump: float,
    state: float,
    is_open: bool,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    young = float(material.young_modulus)
    crit = float(material.critical_displacement)
    yield_stress = float(material.yield_stress)
    threshold = float(material.damage_threshold)
    friction = float(material.friction_coefficient)
    state = max(float(state), threshold)
    t = float(tangential_jump)
    n = float(normal_jump)
    base = yield_stress / ((1.0 - threshold) * crit)
    secant = yield_stress / (crit * state) * (1.0 - state) / (1.0 - threshold)
    D = np.zeros((2, 2), dtype=float)
    stress = np.zeros(2, dtype=float)

    if is_open:
        equivalent = float(np.sqrt(t * t + n * n) / crit)
        loading = equivalent >= state
        if loading:
            denom = crit * crit * state * state * state
            D[0, 0] = base * ((1.0 - state) / state - (t * t) / denom)
            D[1, 1] = base * ((1.0 - state) / state - (n * n) / denom)
            D[0, 1] = -yield_stress * t * n / (
                (1.0 - threshold) * crit * crit * crit * state * state * state
            )
            D[1, 0] = D[0, 1]
        else:
            D[0, 0] = secant
            D[1, 1] = secant
        stress[0] = secant * t
        stress[1] = secant * n
        return D, stress, equivalent, loading

    equivalent = abs(t) / crit
    loading = equivalent >= state
    normal_contact = young / (threshold * crit)
    if loading:
        denom = crit * crit * state * state * state
        D[0, 0] = base * ((1.0 - state) / state - (t * t) / denom)
        D[1, 1] = normal_contact
        if t > 1.0e-20:
            D[0, 1] = -yield_stress * t * n / (
                (1.0 - threshold) * crit * crit * crit * state * state * state
            ) - normal_contact * friction
        elif t < -1.0e-20:
            D[0, 1] = -yield_stress * t * n / (
                (1.0 - threshold) * crit * crit * crit * state * state * state
            ) + normal_contact * friction
    else:
        D[0, 0] = secant
        D[1, 1] = normal_contact
        if t > 0.0:
            D[0, 1] = -normal_contact * friction
        elif t < 0.0:
            D[0, 1] = normal_contact * friction

    stress[1] = normal_contact * n
    if t > 0.0:
        stress[0] = secant * t - friction * stress[1]
    elif t < 0.0:
        stress[0] = secant * t + friction * stress[1]
    else:
        stress[0] = 0.0
    return D, stress, equivalent, loading


def _extract_batch_values(global_solution: np.ndarray | None, gdofs_map: np.ndarray) -> np.ndarray:
    if global_solution is None:
        return np.zeros_like(gdofs_map, dtype=float)
    values = np.asarray(global_solution, dtype=float)
    if values.ndim != 1:
        raise ValueError("global_solution must be one-dimensional.")
    if int(np.max(gdofs_map)) >= values.size:
        raise ValueError("global_solution is smaller than the requested interface DOFs.")
    return values[np.asarray(gdofs_map, dtype=int)]


def _joint_widths_from_state(
    *,
    phi: np.ndarray,
    normals: np.ndarray,
    u_current: np.ndarray,
    initial_width: float,
) -> np.ndarray:
    n_entities, _n_q, n_stations = phi.shape
    n_nodes = 2 * n_stations
    phi_nodes = np.concatenate((phi, phi), axis=2)
    jump_sign = np.concatenate((-np.ones(n_stations), np.ones(n_stations))).astype(float)
    jump_phi = phi_nodes * jump_sign.reshape(1, 1, n_nodes)
    u_current_nodes = np.asarray(u_current, dtype=float).reshape(n_entities, n_nodes, 2)
    jump_u = np.einsum("eqm,ema->eqa", jump_phi, u_current_nodes, optimize=True)
    normal_rel = np.einsum("eqa,ea->eq", jump_u, normals, optimize=True)
    return np.maximum(0.0, float(initial_width) + normal_rel)


def _interface_station_geometry_batch(
    negative_coords: np.ndarray,
    positive_coords: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    neg = np.asarray(negative_coords, dtype=float)
    pos = np.asarray(positive_coords, dtype=float)
    mids = 0.5 * (neg + pos)
    tangent_raw = mids[:, -1, :] - mids[:, 0, :]
    lengths = np.linalg.norm(tangent_raw, axis=1)
    if np.any(lengths <= 1.0e-14):
        raise ValueError("Degenerate interface: first and last station midpoints coincide.")
    tangents = tangent_raw / lengths[:, None]
    normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))
    station_s = np.einsum("eni,ei->en", mids - mids[:, 0:1, :], tangents, optimize=True)
    if np.any(np.diff(station_s, axis=1) <= 1.0e-14):
        raise ValueError("Interface stations must be strictly ordered along the tangent.")
    return station_s, tangents, normals, lengths


def _quadrature_on_station_intervals(
    station_s: np.ndarray,
    n_stations: int,
    quadrature: QuadratureRule,
) -> tuple[np.ndarray, np.ndarray]:
    if quadrature == "lobatto":
        xi, w = gauss_lobatto(max(2, n_stations))
    elif quadrature == "gauss":
        xi, w = gauss_legendre(max(2, n_stations + 1))
    else:
        raise ValueError(f"Unknown interface quadrature rule '{quadrature}'.")
    center = 0.5 * (station_s[:, 0] + station_s[:, -1])
    half = 0.5 * (station_s[:, -1] - station_s[:, 0])
    return center[:, None] + half[:, None] * xi[None, :], half[:, None] * w[None, :]


def _lagrange_basis_and_derivative_batch(nodes: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nodes = np.asarray(nodes, dtype=float)
    points = np.asarray(points, dtype=float)
    n_entities, n_nodes = nodes.shape
    n_q = points.shape[1]
    phi = np.ones((n_entities, n_q, n_nodes), dtype=float)
    dphi = np.zeros_like(phi)
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                phi[:, :, i] *= (points - nodes[:, j, None]) / (nodes[:, i, None] - nodes[:, j, None])
        total = np.zeros((n_entities, n_q), dtype=float)
        for k in range(n_nodes):
            if k == i:
                continue
            term = np.broadcast_to(
                1.0 / (nodes[:, i, None] - nodes[:, k, None]),
                (n_entities, n_q),
            ).copy()
            for j in range(n_nodes):
                if j != i and j != k:
                    term *= (points - nodes[:, j, None]) / (nodes[:, i, None] - nodes[:, j, None])
            total += term
        dphi[:, :, i] = total
    return phi, dphi
