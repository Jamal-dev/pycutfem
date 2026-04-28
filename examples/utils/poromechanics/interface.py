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
from pycutfem.ufl.compilers import LocalAssemblyBatch
from pycutfem.ufl.expressions import Constant, FacetNormal, Neg, Pos, grad, heaviside, pos_part
from pycutfem.ufl.forms import Equation
from pycutfem.ufl.measures import dNonmatchingInterface


QuadratureRule = Literal["lobatto", "gauss"]
BackendName = Literal["python", "jit", "cpp"]
InterfacePermeabilityLaw = Literal["fracture", "link"]


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
class UPlInterfaceUFLSystem2D:
    """Symbolic U-Pl interface forms on a mesh-backed paired interface."""

    lhs_form: object
    rhs_form: object
    equation: Equation
    measure: object
    interface: NonMatchingInterface
    stiffness_lhs: object
    pressure_coupling_lhs: object
    rate_coupling_lhs: object
    storage_lhs: object
    permeability_lhs: object
    rate_coupling_rhs: object
    storage_rhs: object
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
    interface: NonMatchingInterface,
    dt: float,
    theta_u: float = 1.0,
    theta_p: float = 1.0,
    quadrature: QuadratureRule = "lobatto",
    quadrature_order: int | None = None,
    permeability_law: InterfacePermeabilityLaw = "fracture",
) -> UPlInterfaceUFLSystem2D:
    """Build the Kratos-style 2D U-Pl interface as one symbolic UFL system."""

    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if theta_u <= 0.0 or theta_p <= 0.0:
        raise ValueError("theta_u and theta_p must be positive.")
    if permeability_law not in {"fracture", "link"}:
        raise ValueError("permeability_law must be 'fracture' or 'link'.")

    q_rule = "gauss_lobatto" if quadrature == "lobatto" else "gauss"
    if quadrature_order is None:
        q_order = 2 if q_rule == "gauss_lobatto" else 3
    else:
        q_order = int(quadrature_order)
    dGamma = dNonmatchingInterface(metadata={"q": q_order, "interface": interface, "quadrature": q_rule})

    n = FacetNormal()
    thickness = Constant(float(material.thickness))
    alpha = Constant(float(material.biot_coefficient))
    invM = Constant(float(material.biot_modulus_inverse))
    mu_inv = Constant(float(material.dynamic_viscosity_inverse))
    velocity = Constant(1.0 / (float(theta_u) * float(dt)))
    dt_pressure = Constant(1.0 / (float(theta_p) * float(dt)))
    shear_stiff = Constant(float(material.shear_stiffness))
    normal_stiff = Constant(float(material.normal_stiffness))
    penalty = Constant(float(material.penalty_stiffness))
    initial_width = Constant(float(material.initial_joint_width))
    transversal_perm = Constant(float(material.transversal_permeability_coefficient))

    normal_rel = _jump_normal_component(u_current, n)
    width = pos_part(initial_width + normal_rel)
    tangential_perm = width * width / Constant(12.0)
    transverse_perm = tangential_perm if permeability_law == "link" else transversal_perm
    normal_stiff_eff = normal_stiff * (Constant(1.0) + (penalty - Constant(1.0)) * heaviside(-normal_rel))

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
    pressure_coupling_lhs = -thickness * alpha * p_avg * jump_v_n * dGamma
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


def _expand_segment_ids(raw: int | Sequence[int], n_segments: int, name: str) -> np.ndarray:
    arr = np.asarray([raw] if np.isscalar(raw) else raw, dtype=np.int32).reshape(-1)
    if arr.size == 1:
        return np.full(n_segments, int(arr[0]), dtype=np.int32)
    if arr.size != n_segments:
        raise ValueError(f"{name} must be a scalar or have one entry per interface segment.")
    return arr.astype(np.int32, copy=False)


def _avg_trace(expr):
    return Constant(0.5) * (Pos(expr) + Neg(expr))


def _jump_normal_component(vec, normal):
    return (Pos(vec[0]) - Neg(vec[0])) * normal[0] + (Pos(vec[1]) - Neg(vec[1])) * normal[1]


def _jump_tangent_component(vec, normal):
    return (Pos(vec[0]) - Neg(vec[0])) * normal[1] - (Pos(vec[1]) - Neg(vec[1])) * normal[0]


def _tangential_derivative(expr, normal):
    return grad(expr)[0] * normal[1] - grad(expr)[1] * normal[0]


def _avg_tangential_derivative(expr, normal):
    return Constant(0.5) * (_tangential_derivative(Pos(expr), normal) + _tangential_derivative(Neg(expr), normal))


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
