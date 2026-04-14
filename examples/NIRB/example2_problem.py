from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import gmsh  # type: ignore
except Exception:  # pragma: no cover - optional dependency at import time
    gmsh = None

from pycutfem.core.levelset import BeamLevelSet, MinLevelSet
from pycutfem.ufl.expressions import Constant, FacetNormal, Identity, cof, det, dot, grad, inner, inv, trace
from pycutfem.ufl.forms import BoundaryCondition
from pycutfem.ufl.measures import dS, dx
from pycutfem.utils.bitset import BitSet

from examples.fsi_dealii_reference import ALE_Helpers, NSE_ALE
from examples.fsi_dealii_reference import tag_fluid_solid_interface_edges as _tag_fluid_solid_interface_edges

from .double_flap_reference import DoubleFlapReference, load_double_flap_reference


def _named_constant(name: str, value, *, dim: int | None = None) -> Constant:
    const = value if isinstance(value, Constant) else Constant(value, dim=dim)
    setattr(const, "_jit_name", str(name))
    return const


_EX2_HALF = _named_constant("example2_half", 0.5)
_EX2_ONE = _named_constant("example2_one", 1.0)
_EX2_TWO = _named_constant("example2_two", 2.0)
_EX2_TWO_THIRDS = _named_constant("example2_two_thirds", 2.0 / 3.0)


@dataclass(frozen=True)
class DoubleFlapGeometry:
    channel_length: float
    channel_height: float
    cylinder_center: tuple[float, float]
    cylinder_radius: float
    solid_x0: float
    solid_x1: float
    solid_y0: float
    solid_y1: float
    base_height: float
    arm_width: float
    inlet_ramp_end_time: float
    inlet_tag: str = "inlet"
    outlet_tag: str = "outlet"
    walls_tag: str = "walls"
    cylinder_tag: str = "cylinder"
    interface_tag: str = "structure_interface"
    clamp_tag: str = "structure_clamp"

    @property
    def solid_width(self) -> float:
        return float(self.solid_x1 - self.solid_x0)

    @property
    def solid_height(self) -> float:
        return float(self.solid_y1 - self.solid_y0)

    @property
    def left_arm_x1(self) -> float:
        return float(self.solid_x0 + self.arm_width)

    @property
    def right_arm_x0(self) -> float:
        return float(self.solid_x1 - self.arm_width)

    @property
    def gap_width(self) -> float:
        return float(self.right_arm_x0 - self.left_arm_x1)

    def contains_solid_point(self, x: float, y: float, *, tol: float = 1.0e-12) -> bool:
        x = float(x)
        y = float(y)
        if x < self.solid_x0 - tol or x > self.solid_x1 + tol:
            return False
        if y < self.solid_y0 - tol or y > self.solid_y1 + tol:
            return False
        if y <= self.solid_y0 + self.base_height + tol:
            return True
        return x <= self.left_arm_x1 + tol or x >= self.right_arm_x0 - tol

    def level_set(self):
        left_arm = BeamLevelSet(
            center=(self.solid_x0 + 0.5 * self.arm_width, self.solid_y0 + 0.5 * self.solid_height),
            Lb=self.arm_width,
            Hb=self.solid_height,
        )
        right_arm = BeamLevelSet(
            center=(self.solid_x1 - 0.5 * self.arm_width, self.solid_y0 + 0.5 * self.solid_height),
            Lb=self.arm_width,
            Hb=self.solid_height,
        )
        base = BeamLevelSet(
            center=(0.5 * (self.solid_x0 + self.solid_x1), self.solid_y0 + 0.5 * self.base_height),
            Lb=self.solid_width,
            Hb=self.base_height,
        )
        return MinLevelSet(left_arm, right_arm, base)

    def inlet_velocity(self, y: float, t: float, *, reference_velocity: float) -> float:
        y = float(y)
        t = float(t)
        if y <= 0.0 or y >= self.channel_height:
            return 0.0
        profile = 4.0 * float(reference_velocity) * y * (self.channel_height - y) / (self.channel_height**2)
        if t >= self.inlet_ramp_end_time:
            return profile
        phase = max(t, 0.0) / max(self.inlet_ramp_end_time, 1.0e-14)
        return 0.5 * (1.0 - math.cos(math.pi * phase)) * profile


def _group_consecutive(values: np.ndarray, *, gap_tol: float) -> list[tuple[float, float]]:
    if values.size == 0:
        return []
    values = np.sort(np.asarray(values, dtype=float))
    groups: list[tuple[float, float]] = []
    start = float(values[0])
    prev = float(values[0])
    for value in values[1:]:
        value = float(value)
        if value - prev > gap_tol:
            groups.append((start, prev))
            start = value
        prev = value
    groups.append((start, prev))
    return groups


def _infer_geometry_from_reference(reference: DoubleFlapReference) -> DoubleFlapGeometry:
    solid_coords = np.asarray(list(reference.solid.nodes.values()), dtype=float)
    solid_x0 = float(solid_coords[:, 0].min())
    solid_x1 = float(solid_coords[:, 0].max())
    solid_y0 = float(solid_coords[:, 1].min())
    solid_y1 = float(solid_coords[:, 1].max())

    x_levels = np.unique(np.round(solid_coords[:, 0], 6))
    y_levels = np.unique(np.round(solid_coords[:, 1], 6))
    x_gap_tol = max(1.5e-2, 5.0 * float(np.median(np.diff(x_levels))) if x_levels.size > 1 else 1.5e-2)
    y_tol = max(2.0e-3, 2.0 * float(np.median(np.diff(y_levels))) if y_levels.size > 1 else 2.0e-3)

    upper_coords = solid_coords[solid_coords[:, 1] >= solid_y0 + 0.5 * (solid_y1 - solid_y0)]
    if upper_coords.size == 0:
        raise RuntimeError("Could not infer the DoubleFlap arm geometry from the reference solid mesh.")
    upper_xs = np.unique(np.round(upper_coords[:, 0], 6))
    upper_components = _group_consecutive(upper_xs, gap_tol=x_gap_tol)
    if len(upper_components) != 2:
        raise RuntimeError(
            f"Expected two arm components above the base, found {len(upper_components)} components: {upper_components}"
    )
    left_arm = upper_components[0]
    right_arm = upper_components[1]
    arm_width = 0.5 * ((left_arm[1] - solid_x0) + (solid_x1 - right_arm[0]))
    gap_x0 = solid_x0 + arm_width
    gap_x1 = solid_x1 - arm_width

    base_rows: list[float] = []
    for y in y_levels:
        band = solid_coords[np.abs(solid_coords[:, 1] - y) <= y_tol]
        if band.size == 0:
            continue
        has_gap_fill = np.any((band[:, 0] >= gap_x0 + x_gap_tol) & (band[:, 0] <= gap_x1 - x_gap_tol))
        if has_gap_fill:
            base_rows.append(float(y))
    if not base_rows:
        raise RuntimeError("Could not infer the DoubleFlap base height from the reference solid mesh.")
    base_height = max(base_rows) - solid_y0

    return DoubleFlapGeometry(
        channel_length=float(reference.channel_length),
        channel_height=float(reference.channel_height),
        cylinder_center=tuple(float(v) for v in reference.cylinder_center),
        cylinder_radius=float(reference.cylinder_radius),
        solid_x0=solid_x0,
        solid_x1=solid_x1,
        solid_y0=solid_y0,
        solid_y1=solid_y1,
        base_height=float(base_height),
        arm_width=float(arm_width),
        inlet_ramp_end_time=float(reference.inlet_ramp_end_time),
    )


def load_geometry(reference_root: str | Path | None = None) -> DoubleFlapGeometry:
    return _infer_geometry_from_reference(load_double_flap_reference(reference_root))


def classify_fluid_solid(mesh, geometry: DoubleFlapGeometry, *, tol: float = 1.0e-9) -> tuple[BitSet, BitSet]:
    coords = mesh.nodes_x_y_pos[mesh.corner_connectivity].mean(axis=1)
    solid_mask = np.array(
        [geometry.contains_solid_point(float(x), float(y), tol=tol) for x, y in np.asarray(coords, dtype=float)],
        dtype=bool,
    )
    tags = np.where(solid_mask, "solid", "fluid")
    for element, tag in zip(mesh.elements_list, tags):
        element.tag = str(tag)
    mesh._elem_bitsets = {"fluid": BitSet(tags == "fluid"), "solid": BitSet(tags == "solid")}
    return mesh._elem_bitsets["fluid"], mesh._elem_bitsets["solid"]


def tag_interface_edges(mesh, geometry: DoubleFlapGeometry) -> int:
    return _tag_fluid_solid_interface_edges(mesh, tag=geometry.interface_tag)


def retag_boundaries(mesh, geometry: DoubleFlapGeometry, *, tol: float = 1.0e-6, overwrite: bool = False) -> None:
    cx, cy = geometry.cylinder_center
    for edge in mesh.edges_list:
        if edge.right is not None:
            continue
        if getattr(edge, "tag", "") and not overwrite:
            continue
        mid_x, mid_y = mesh.nodes_x_y_pos[list(edge.nodes)].mean(axis=0)
        x = float(mid_x)
        y = float(mid_y)
        tag = ""
        edge_pts = mesh.nodes_x_y_pos[list(edge.all_nodes if edge.all_nodes else edge.nodes)]
        r_edge = np.linalg.norm(edge_pts - np.asarray([[cx, cy]], dtype=float), axis=1)
        if abs(x) <= tol:
            tag = geometry.inlet_tag
        elif abs(x - geometry.channel_length) <= tol:
            tag = geometry.outlet_tag
        elif (
            edge_pts[:, 0].max() <= cx + geometry.cylinder_radius + 6.0 * tol
            and edge_pts[:, 0].min() >= cx - geometry.cylinder_radius - 6.0 * tol
            and edge_pts[:, 1].max() <= cy + geometry.cylinder_radius + 6.0 * tol
            and edge_pts[:, 1].min() >= cy - geometry.cylinder_radius - 6.0 * tol
            and np.min(np.abs(r_edge - geometry.cylinder_radius)) <= 6.0 * tol
        ):
            tag = geometry.cylinder_tag
        elif abs(y - geometry.solid_y0) <= tol and geometry.solid_x0 - tol <= x <= geometry.solid_x1 + tol:
            tag = geometry.clamp_tag
        elif abs(y) <= tol or abs(y - geometry.channel_height) <= tol:
            tag = geometry.walls_tag
        edge.tag = tag
    mesh.rebuild_edge_bitsets()


def _curve_center(curve_tag: int) -> tuple[float, float]:
    com = gmsh.model.occ.getCenterOfMass(1, int(curve_tag))
    return float(com[0]), float(com[1])


def _curve_bbox(curve_tag: int) -> tuple[float, float, float, float]:
    x0, y0, _z0, x1, y1, _z1 = gmsh.model.getBoundingBox(1, int(curve_tag))
    return float(x0), float(y0), float(x1), float(y1)


def build_conforming_mesh(
    path: str | Path,
    *,
    geometry: DoubleFlapGeometry,
    mesh_size: float,
    order: int = 1,
    view: bool = False,
) -> Path:
    if gmsh is None:
        raise RuntimeError("gmsh is required to build the DoubleFlap conforming mesh.")

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    gmsh.initialize()
    try:
        gmsh.model.add("double_flap_conforming")
        occ = gmsh.model.occ

        channel = occ.addRectangle(0.0, 0.0, 0.0, geometry.channel_length, geometry.channel_height)
        cylinder = occ.addDisk(
            geometry.cylinder_center[0],
            geometry.cylinder_center[1],
            0.0,
            geometry.cylinder_radius,
            geometry.cylinder_radius,
        )
        left_arm = occ.addRectangle(
            geometry.solid_x0,
            geometry.solid_y0,
            0.0,
            geometry.arm_width,
            geometry.solid_height,
        )
        right_arm = occ.addRectangle(
            geometry.right_arm_x0,
            geometry.solid_y0,
            0.0,
            geometry.arm_width,
            geometry.solid_height,
        )
        base = occ.addRectangle(
            geometry.solid_x0,
            geometry.solid_y0,
            0.0,
            geometry.solid_width,
            geometry.base_height,
        )

        solid_dimtags, _ = occ.fuse([(2, left_arm)], [(2, right_arm), (2, base)], removeObject=True, removeTool=True)
        solid_surfaces = [tag for dim, tag in solid_dimtags if dim == 2]
        fluid_dimtags, _ = occ.cut([(2, channel)], [(2, cylinder)] + solid_dimtags, removeObject=True, removeTool=False)
        fluid_surfaces = [tag for dim, tag in fluid_dimtags if dim == 2]

        occ.synchronize()

        if not fluid_surfaces or not solid_surfaces:
            raise RuntimeError("Failed to build the DoubleFlap fluid/solid surface sets.")

        gmsh.model.addPhysicalGroup(2, fluid_surfaces, tag=1)
        gmsh.model.setPhysicalName(2, 1, "fluid")
        gmsh.model.addPhysicalGroup(2, solid_surfaces, tag=2)
        gmsh.model.setPhysicalName(2, 2, "solid")

        fluid_curves = sorted(
            {
                abs(int(curve_tag))
                for surf in fluid_surfaces
                for dim, curve_tag in gmsh.model.getBoundary([(2, int(surf))], oriented=False)
                if dim == 1
            }
        )
        solid_curves = sorted(
            {
                abs(int(curve_tag))
                for surf in solid_surfaces
                for dim, curve_tag in gmsh.model.getBoundary([(2, int(surf))], oriented=False)
                if dim == 1
            }
        )

        boundary_groups: dict[str, list[int]] = {
            geometry.inlet_tag: [],
            geometry.outlet_tag: [],
            geometry.walls_tag: [],
            geometry.cylinder_tag: [],
            geometry.interface_tag: [],
            geometry.clamp_tag: [],
        }
        tol = max(1.0e-6, 0.1 * float(mesh_size))
        cx, cy = geometry.cylinder_center

        for curve in fluid_curves:
            x_mid, y_mid = _curve_center(curve)
            x0, y0, x1, y1 = _curve_bbox(curve)
            if abs(x_mid) <= tol:
                boundary_groups[geometry.inlet_tag].append(curve)
            elif abs(x_mid - geometry.channel_length) <= tol:
                boundary_groups[geometry.outlet_tag].append(curve)
            elif (
                x0 >= cx - geometry.cylinder_radius - tol
                and x1 <= cx + geometry.cylinder_radius + tol
                and y0 >= cy - geometry.cylinder_radius - tol
                and y1 <= cy + geometry.cylinder_radius + tol
            ):
                boundary_groups[geometry.cylinder_tag].append(curve)
            elif abs(y_mid) <= tol or abs(y_mid - geometry.channel_height) <= tol:
                boundary_groups[geometry.walls_tag].append(curve)

        for curve in solid_curves:
            x_mid, y_mid = _curve_center(curve)
            if abs(y_mid - geometry.solid_y0) <= tol and geometry.solid_x0 - tol <= x_mid <= geometry.solid_x1 + tol:
                boundary_groups[geometry.clamp_tag].append(curve)
            else:
                boundary_groups[geometry.interface_tag].append(curve)

        physical_ids = {
            geometry.inlet_tag: 11,
            geometry.walls_tag: 12,
            geometry.outlet_tag: 13,
            geometry.cylinder_tag: 14,
            geometry.interface_tag: 15,
            geometry.clamp_tag: 16,
        }
        for name, curves in boundary_groups.items():
            if not curves:
                continue
            tag = gmsh.model.addPhysicalGroup(1, sorted(set(int(curve) for curve in curves)), tag=physical_ids[name])
            gmsh.model.setPhysicalName(1, tag, name)

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", float(mesh_size))
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", float(mesh_size))
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.SecondOrderLinear", 1)

        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(int(order))
        if view:
            try:
                gmsh.fltk.initialize()
                gmsh.fltk.run()
            except Exception:
                pass
        gmsh.write(str(target))
        return target
    finally:
        gmsh.finalize()


def _neo_hookean_pk1(F, mu_s: Constant, lambda_s: Constant):
    c10 = _EX2_HALF * mu_s
    kappa = lambda_s + _EX2_TWO_THIRDS * mu_s
    I2 = Identity(2)
    C = dot(F.T, F)
    A = inv(C)
    I1 = trace(C)
    J = det(F)
    S_iso = (_EX2_TWO * c10 / J) * (I2 - _EX2_HALF * I1 * A)
    p = kappa * (J - _EX2_ONE)
    S_vol = J * p * A
    S = S_iso + S_vol
    return dot(F, S)


def _neo_hookean_delta_pk1(F, grad_dd, mu_s: Constant, lambda_s: Constant):
    c10 = _EX2_HALF * mu_s
    kappa = lambda_s + _EX2_TWO_THIRDS * mu_s
    I2 = Identity(2)
    C = dot(F.T, F)
    A = inv(C)
    I1 = trace(C)
    J = det(F)

    S_iso = (_EX2_TWO * c10 / J) * (I2 - _EX2_HALF * I1 * A)
    p = kappa * (J - _EX2_ONE)
    S_vol = J * p * A
    S = S_iso + S_vol

    delta_C = dot(grad_dd.T, F) + dot(F.T, grad_dd)
    tr_delta_C = trace(delta_C)
    tr_A_delta_C = trace(dot(A, delta_C))

    mu2 = _EX2_TWO * c10
    term_I = I2 - _EX2_HALF * I1 * A
    delta_S_iso = (
        -(mu2 / (_EX2_TWO * J)) * tr_A_delta_C * term_I
        + (mu2 / J) * (-_EX2_HALF * tr_delta_C * A + _EX2_HALF * I1 * dot(dot(A, delta_C), A))
    )

    alpha = kappa * J * (J - _EX2_ONE)
    delta_S_vol = (
        _EX2_HALF * kappa * J * (_EX2_TWO * J - _EX2_ONE) * tr_A_delta_C * A
        - alpha * dot(dot(A, delta_C), A)
    )
    delta_S = delta_S_iso + delta_S_vol
    return dot(grad_dd, S) + dot(F, delta_S)


def build_jac(
    *,
    uk,
    u_prev,
    dk,
    d_prev,
    pk,
    p_prev,
    du,
    dd,
    dp,
    test_v,
    test_w,
    test_q,
    timestep: Constant,
    theta: Constant,
    rho_f: Constant,
    mu_f: Constant,
    rho_s: Constant,
    lambda_s: Constant,
    mu_s: Constant,
    alpha_u: Constant,
    stab_eps: Constant,
    p_gauge: Constant | None = None,
    fluid_bs=None,
    solid_bs=None,
    outlet_bs=None,
    quad_order: int = 4,
):
    del p_prev, stab_eps

    timestep = _named_constant("example2_dt", timestep)
    theta = _named_constant("example2_theta", theta)
    rho_f = _named_constant("example2_rho_f", rho_f)
    if isinstance(mu_f, Constant):
        mu_f = _named_constant("example2_mu_f", mu_f)
    rho_s = _named_constant("example2_rho_s", rho_s)
    lambda_s = _named_constant("example2_lambda_s", lambda_s)
    mu_s = _named_constant("example2_mu_s", mu_s)
    alpha_u = _named_constant("example2_alpha_u", alpha_u)
    if p_gauge is not None:
        p_gauge = _named_constant("example2_p_gauge", p_gauge)

    dx_f = dx(defined_on=fluid_bs, metadata={"q": quad_order})
    dx_s = dx(defined_on=solid_bs, metadata={"q": quad_order})
    dS_outlet = dS(defined_on=outlet_bs, metadata={"q": quad_order})
    n = FacetNormal()

    F = ALE_Helpers.get_F(grad(dk))
    J = ALE_Helpers.get_J(F)
    Finv = ALE_Helpers.get_F_inv(F)
    F_inv_T = Finv.T
    pI = pk * Identity(2)
    pI_LinP_trial = dp * Identity(2)

    F_old = ALE_Helpers.get_F(grad(d_prev))
    J_old = ALE_Helpers.get_J(F_old)

    grad_dd = grad(dd)
    J_F_inv_T_LinU = cof(grad_dd)
    J_LinU = ALE_Helpers.get_J_LinU(F, grad_dd)
    Finv_LinU = ALE_Helpers.get_F_inv_LinU(Finv, grad_dd)

    test_grad_v = grad(test_v)

    grad_uk = grad(uk)
    grad_uk_T = grad_uk.T
    sigma_ALE = NSE_ALE.get_stress_fluid_ALE_direct(mu_f, pI, grad_uk, Finv, grad_uk_T, F_inv_T)
    acc_term_jac = NSE_ALE.get_acceleration_term_LinAll(J, J_old, J_LinU, uk, u_prev, du, rho_f)
    convection_fluid_v = NSE_ALE.get_Convection_LinAll_short(
        grad(du),
        du,
        J,
        J_LinU,
        Finv,
        Finv_LinU,
        uk,
        grad_uk,
        rho_f,
    )
    convection_fluid_d = NSE_ALE.get_Convection_u_LinAll_short(
        grad(du),
        dd,
        J,
        J_LinU,
        Finv,
        Finv_LinU,
        dk,
        grad_uk,
        rho_f,
    )
    convection_fluid_u_old = NSE_ALE.get_Convection_u_old_LinAll_short(
        grad(du),
        J,
        J_LinU,
        Finv,
        Finv_LinU,
        d_prev,
        grad_uk,
        rho_f,
    )
    stress_fluid_term_1 = NSE_ALE.get_stress_fluid_ALE_1st_term_LinAll_short(
        pI,
        F_inv_T,
        J_F_inv_T_LinU,
        pI_LinP_trial,
        J,
    )
    stress_visc_no_mu = dot(grad_uk, Finv) + dot(F_inv_T, grad_uk_T)
    stress_fluid_term_2 = NSE_ALE.get_stress_fluid_ALE_2nd_term_LinAll_short(
        J_F_inv_T_LinU,
        stress_visc_no_mu,
        grad_uk,
        grad(du),
        Finv,
        Finv_LinU,
        J,
        mu_f,
    )

    jac_mass_du = dot(acc_term_jac, test_v)
    jac_convection_du = timestep * theta * dot(convection_fluid_v, test_v)
    jac_convection_du += -dot(convection_fluid_d, test_v)
    jac_convection_du += dot(convection_fluid_u_old, test_v)
    jac_diffusion_du = timestep * inner(stress_fluid_term_1, test_grad_v)
    jac_diffusion_du += timestep * theta * inner(stress_fluid_term_2, test_grad_v)
    jac_biharmonic_dd = -alpha_u / (J * J) * J_LinU * inner(grad(dk), grad(test_w)) + alpha_u / J * inner(
        grad(dd), grad(test_w)
    )
    jac_incompressibility_dp = NSE_ALE.get_Incompressibility_ALE_LinV_optimized(grad_uk, grad(du), F, grad_dd) * test_q

    volume_terms_fluid = (
        jac_mass_du
        + jac_convection_du
        + jac_diffusion_du
        + jac_biharmonic_dd
        + jac_incompressibility_dp
    ) * dx_f
    if p_gauge is not None:
        volume_terms_fluid += (p_gauge * dp * test_q) * dx_f

    neuman_term = NSE_ALE.get_stress_fluid_ALE_3rd_term_LinAll_short(
        Finv,
        Finv_LinU,
        grad_uk,
        grad(du),
        mu_f,
        J,
        J_F_inv_T_LinU,
    )
    out_flow_jac = -timestep * theta * dot(dot(neuman_term, n), test_v) * dS_outlet

    solid_stress_LinU = _neo_hookean_delta_pk1(F, grad_dd, mu_s, lambda_s)
    jac_solid = (
        rho_s * dot(du, test_v)
        + timestep * theta * inner(solid_stress_LinU, grad(test_v))
        + rho_s * dot(dd, test_w)
        - rho_s * timestep * theta * dot(du, test_w)
        + dp * test_q
    ) * dx_s
    if p_gauge is not None:
        jac_solid += (p_gauge * dp * test_q) * dx_s

    return volume_terms_fluid + out_flow_jac + jac_solid


def build_residual(
    *,
    uk,
    u_prev,
    dk,
    d_prev,
    pk,
    p_prev,
    v_test,
    w_test,
    q_test,
    dt: Constant,
    theta: Constant,
    rho_f: Constant,
    mu_f: Constant,
    rho_s: Constant,
    lambda_s: Constant,
    mu_s: Constant,
    alpha_u: Constant,
    stab_eps: Constant,
    p_gauge: Constant | None = None,
    fluid_bs=None,
    solid_bs=None,
    outlet_bs=None,
    quad_order: int = 4,
):
    del p_prev, stab_eps

    dt = _named_constant("example2_dt", dt)
    theta = _named_constant("example2_theta", theta)
    rho_f = _named_constant("example2_rho_f", rho_f)
    if isinstance(mu_f, Constant):
        mu_f = _named_constant("example2_mu_f", mu_f)
    rho_s = _named_constant("example2_rho_s", rho_s)
    lambda_s = _named_constant("example2_lambda_s", lambda_s)
    mu_s = _named_constant("example2_mu_s", mu_s)
    alpha_u = _named_constant("example2_alpha_u", alpha_u)
    if p_gauge is not None:
        p_gauge = _named_constant("example2_p_gauge", p_gauge)

    dx_f = dx(defined_on=fluid_bs, metadata={"q": quad_order})
    dx_s = dx(defined_on=solid_bs, metadata={"q": quad_order})
    dS_outlet = dS(defined_on=outlet_bs, metadata={"q": quad_order})
    n = FacetNormal()

    grad_v = grad(uk)
    grad_d = grad(dk)
    grad_v_old = grad(u_prev)
    grad_d_old = grad(d_prev)
    F = ALE_Helpers.get_F(grad_d)
    Finv = ALE_Helpers.get_F_inv(F)
    J = ALE_Helpers.get_J(F)
    F_old = ALE_Helpers.get_F(grad_d_old)
    Finv_old = ALE_Helpers.get_F_inv(F_old)
    J_old = ALE_Helpers.get_J(F_old)
    pI = pk * Identity(2)
    J_theta = theta * J + (_EX2_ONE - theta) * J_old

    acc_term = rho_f * J_theta * inner(uk - u_prev, v_test)
    convection_fluid = rho_f * J * dot(dot(grad_v, Finv), uk)
    convection_fluid_with_u = rho_f * J * dot(dot(grad_v, Finv), dk)
    convection_fluid_with_u_old = rho_f * J * dot(dot(grad_v, Finv), d_prev)
    old_convection_fluid = rho_f * J_old * dot(dot(grad_v_old, Finv_old), u_prev)
    convec_term = (
        dt * theta * dot(convection_fluid, v_test)
        + dt * (_EX2_ONE - theta) * dot(old_convection_fluid, v_test)
        - dot(convection_fluid_with_u - convection_fluid_with_u_old, v_test)
    )
    pressure_term = dt * inner(-(J * dot(pI, Finv.T)), grad(v_test))

    sigma_ALE = NSE_ALE.get_stress_fluid_except_pressure_ALE(mu_f, grad_v, Finv)
    sigma_ALE_old = NSE_ALE.get_stress_fluid_except_pressure_ALE(mu_f, grad_v_old, Finv_old)
    stress_fluid_viscous = J * dot(sigma_ALE, Finv.T)
    stress_fluid_viscous_old = J_old * dot(sigma_ALE_old, Finv_old.T)
    stress_term = dt * theta * inner(stress_fluid_viscous, grad(v_test))
    stress_term += dt * (_EX2_ONE - theta) * inner(stress_fluid_viscous_old, grad(v_test))

    biharmonic_term = alpha_u / J * inner(grad(dk), grad(w_test))
    incompressibility_term = NSE_ALE.get_Incompressibility_ALE(uk, F) * q_test

    residual_fluid = (
        acc_term
        + convec_term
        + pressure_term
        + stress_term
        + biharmonic_term
        + incompressibility_term
    ) * dx_f
    if p_gauge is not None:
        residual_fluid += (p_gauge * pk * q_test) * dx_f

    sigma_ALE_tilde = mu_f * dot(Finv.T, grad_v.T)
    sigma_ALE_tilde_old = mu_f * dot(Finv_old.T, grad_v_old.T)
    stress_fluid_transpose = J * dot(sigma_ALE_tilde, Finv.T)
    stress_fluid_transpose_old = J_old * dot(sigma_ALE_tilde_old, Finv_old.T)
    out_flow = (
        -dt * theta * dot(dot(stress_fluid_transpose, n), v_test)
        - dt * (_EX2_ONE - theta) * dot(dot(stress_fluid_transpose_old, n), v_test)
    )
    residual_outlet = out_flow * dS_outlet

    P_solid = _neo_hookean_pk1(F, mu_s, lambda_s)
    P_solid_old = _neo_hookean_pk1(F_old, mu_s, lambda_s)
    residual_solid = (
        rho_s * inner(uk - u_prev, v_test)
        + dt * theta * inner(P_solid, grad(v_test))
        + dt * (_EX2_ONE - theta) * inner(P_solid_old, grad(v_test))
        + rho_s * inner(dk - d_prev, w_test)
        - rho_s * dt * theta * inner(uk, w_test)
        - rho_s * dt * (_EX2_ONE - theta) * inner(u_prev, w_test)
        + pk * q_test
    ) * dx_s
    if p_gauge is not None:
        residual_solid += (p_gauge * pk * q_test) * dx_s

    return residual_fluid + residual_outlet + residual_solid


def build_bcs(*, geometry: DoubleFlapGeometry, reference_velocity: float):
    zero = lambda x, y, t=0.0: 0.0

    def inlet_u(x, y, t=0.0):
        return geometry.inlet_velocity(y, t, reference_velocity=reference_velocity)

    vel_bcs = [
        BoundaryCondition("ux", "dirichlet", geometry.inlet_tag, inlet_u),
        BoundaryCondition("uy", "dirichlet", geometry.inlet_tag, zero),
        BoundaryCondition("ux", "dirichlet", geometry.walls_tag, zero),
        BoundaryCondition("uy", "dirichlet", geometry.walls_tag, zero),
        BoundaryCondition("ux", "dirichlet", geometry.cylinder_tag, zero),
        BoundaryCondition("uy", "dirichlet", geometry.cylinder_tag, zero),
        BoundaryCondition("ux", "dirichlet", geometry.clamp_tag, zero),
        BoundaryCondition("uy", "dirichlet", geometry.clamp_tag, zero),
    ]
    disp_bcs = [
        BoundaryCondition("dx", "dirichlet", geometry.inlet_tag, zero),
        BoundaryCondition("dy", "dirichlet", geometry.inlet_tag, zero),
        BoundaryCondition("dx", "dirichlet", geometry.outlet_tag, zero),
        BoundaryCondition("dy", "dirichlet", geometry.outlet_tag, zero),
        BoundaryCondition("dx", "dirichlet", geometry.walls_tag, zero),
        BoundaryCondition("dy", "dirichlet", geometry.walls_tag, zero),
        BoundaryCondition("dx", "dirichlet", geometry.cylinder_tag, zero),
        BoundaryCondition("dy", "dirichlet", geometry.cylinder_tag, zero),
        BoundaryCondition("dx", "dirichlet", geometry.clamp_tag, zero),
        BoundaryCondition("dy", "dirichlet", geometry.clamp_tag, zero),
    ]
    bcs = vel_bcs + disp_bcs
    bcs_homog = [BoundaryCondition(b.field, b.method, b.domain_tag, zero) for b in bcs]
    return bcs, bcs_homog
