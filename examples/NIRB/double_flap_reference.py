from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def default_double_flap_root() -> Path:
    return Path(__file__).resolve().parents[2] / ".tmp" / "nirb_benchmarks" / "DoubleFlap"


@dataclass(frozen=True)
class SubModelPart:
    name: str
    node_ids: tuple[int, ...]
    element_ids: tuple[int, ...]
    condition_ids: tuple[int, ...]


@dataclass(frozen=True)
class MDPAMesh:
    path: Path
    element_block: str
    condition_blocks: tuple[str, ...]
    nodes: dict[int, tuple[float, float]]
    elements: dict[int, tuple[int, ...]]
    conditions: dict[int, tuple[int, ...]]
    submodelparts: dict[str, SubModelPart]

    def submodelpart_coords(self, name: str) -> np.ndarray:
        node_ids = self.submodelparts[name].node_ids
        return np.asarray([self.nodes[node_id] for node_id in node_ids], dtype=float)


@dataclass(frozen=True)
class DoubleFlapReference:
    root: Path
    fluid: MDPAMesh
    solid: MDPAMesh
    fluid_project_parameters: dict[str, Any]
    solid_project_parameters: dict[str, Any]
    coupling_parameters: dict[str, Any]
    channel_length: float
    channel_height: float
    solid_bbox: tuple[float, float, float, float]
    interface_node_count: int
    interface_max_mismatch: float
    clamp_node_count: int
    fluid_time_step: float
    solid_time_step: float
    end_time: float
    inlet_ramp_end_time: float
    inlet_modulus_ramp: str
    inlet_modulus_steady: str
    density: float
    kinematic_viscosity: float
    max_velocity: float
    cylinder_center: tuple[float, float]
    cylinder_radius: float

    @property
    def interface_coords_fluid(self) -> np.ndarray:
        return self.fluid.submodelpart_coords("NoSlip2D_Interface")

    @property
    def interface_coords_solid(self) -> np.ndarray:
        return self.solid.submodelpart_coords("StructureInterface2D_Struc_Fsi")

    @property
    def clamp_coords(self) -> np.ndarray:
        return self.solid.submodelpart_coords("DISPLACEMENT_BCDisp")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["root"] = str(self.root)
        data["fluid"] = {
            "path": str(self.fluid.path),
            "element_block": self.fluid.element_block,
            "condition_blocks": list(self.fluid.condition_blocks),
            "submodelparts": {
                name: {
                    "node_count": len(part.node_ids),
                    "element_count": len(part.element_ids),
                    "condition_count": len(part.condition_ids),
                }
                for name, part in self.fluid.submodelparts.items()
            },
        }
        data["solid"] = {
            "path": str(self.solid.path),
            "element_block": self.solid.element_block,
            "condition_blocks": list(self.solid.condition_blocks),
            "submodelparts": {
                name: {
                    "node_count": len(part.node_ids),
                    "element_count": len(part.element_ids),
                    "condition_count": len(part.condition_ids),
                }
                for name, part in self.solid.submodelparts.items()
            },
        }
        return data

    def write_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return target

    def plot_geometry(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        figure, axis = plt.subplots(figsize=(9, 4))
        fluid_coords = np.asarray(list(self.fluid.nodes.values()), dtype=float)
        solid_coords = np.asarray(list(self.solid.nodes.values()), dtype=float)
        axis.scatter(fluid_coords[:, 0], fluid_coords[:, 1], s=3, c="#c9d5e3", label="Fluid nodes")
        axis.scatter(solid_coords[:, 0], solid_coords[:, 1], s=5, c="#d95f02", label="Solid nodes")

        interface = self.interface_coords_solid
        clamp = self.clamp_coords
        axis.scatter(interface[:, 0], interface[:, 1], s=12, c="#1b9e77", label="Interface")
        axis.scatter(clamp[:, 0], clamp[:, 1], s=12, c="#7570b3", label="Clamp")

        circle = plt.Circle(self.cylinder_center, self.cylinder_radius, fill=False, color="#444444", lw=1.5)
        axis.add_patch(circle)
        axis.set_aspect("equal")
        axis.set_xlim(-0.02, self.channel_length + 0.02)
        axis.set_ylim(-0.02, self.channel_height + 0.02)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_title("DoubleFlap Reference Geometry")
        axis.legend(loc="upper right")
        figure.tight_layout()
        figure.savefig(target, dpi=200)
        plt.close(figure)
        return target


def reynolds_to_mean_velocity(reynolds: float, *, kinematic_viscosity: float, cylinder_diameter: float) -> float:
    return float(reynolds) * float(kinematic_viscosity) / float(cylinder_diameter)


def _parse_mdpa(path: str | Path) -> MDPAMesh:
    source = Path(path)
    lines = source.read_text(encoding="utf-8").splitlines()

    nodes: dict[int, tuple[float, float]] = {}
    elements: dict[int, tuple[int, ...]] = {}
    conditions: dict[int, tuple[int, ...]] = {}
    submodelparts: dict[str, SubModelPart] = {}
    condition_blocks: list[str] = []
    element_block = ""

    section = ""
    current_part = ""
    part_nodes: list[int] = []
    part_elements: list[int] = []
    part_conditions: list[int] = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("//"):
            continue

        if line == "Begin Nodes":
            section = "nodes"
            continue
        if line == "End Nodes":
            section = ""
            continue

        if line.startswith("Begin Elements "):
            tokens = line.split()
            element_block = tokens[2] if len(tokens) >= 3 else ""
            section = "elements"
            continue
        if line == "End Elements":
            section = ""
            continue

        if line.startswith("Begin Conditions "):
            tokens = line.split()
            if len(tokens) >= 3:
                condition_blocks.append(tokens[2])
            section = "conditions"
            continue
        if line == "End Conditions":
            section = ""
            continue

        if line.startswith("Begin SubModelPart "):
            if current_part:
                submodelparts[current_part] = SubModelPart(
                    name=current_part,
                    node_ids=tuple(part_nodes),
                    element_ids=tuple(part_elements),
                    condition_ids=tuple(part_conditions),
                )
            current_part = line.split()[2]
            part_nodes = []
            part_elements = []
            part_conditions = []
            section = "submodelpart"
            continue
        if line == "End SubModelPart":
            if current_part:
                submodelparts[current_part] = SubModelPart(
                    name=current_part,
                    node_ids=tuple(part_nodes),
                    element_ids=tuple(part_elements),
                    condition_ids=tuple(part_conditions),
                )
            current_part = ""
            part_nodes = []
            part_elements = []
            part_conditions = []
            section = ""
            continue

        if line == "Begin SubModelPartNodes":
            section = "submodelpart_nodes"
            continue
        if line == "End SubModelPartNodes":
            section = "submodelpart"
            continue
        if line == "Begin SubModelPartElements":
            section = "submodelpart_elements"
            continue
        if line == "End SubModelPartElements":
            section = "submodelpart"
            continue
        if line == "Begin SubModelPartConditions":
            section = "submodelpart_conditions"
            continue
        if line == "End SubModelPartConditions":
            section = "submodelpart"
            continue

        if section == "nodes":
            parts = line.split()
            nodes[int(parts[0])] = (float(parts[1]), float(parts[2]))
            continue
        if section == "elements":
            parts = line.split()
            elements[int(parts[0])] = tuple(int(value) for value in parts[2:])
            continue
        if section == "conditions":
            parts = line.split()
            conditions[int(parts[0])] = tuple(int(value) for value in parts[2:])
            continue
        if section == "submodelpart_nodes":
            part_nodes.append(int(line.split()[0]))
            continue
        if section == "submodelpart_elements":
            part_elements.append(int(line.split()[0]))
            continue
        if section == "submodelpart_conditions":
            part_conditions.append(int(line.split()[0]))
            continue

    if current_part and current_part not in submodelparts:
        submodelparts[current_part] = SubModelPart(
            name=current_part,
            node_ids=tuple(part_nodes),
            element_ids=tuple(part_elements),
            condition_ids=tuple(part_conditions),
        )

    return MDPAMesh(
        path=source,
        element_block=element_block,
        condition_blocks=tuple(condition_blocks),
        nodes=nodes,
        elements=elements,
        conditions=conditions,
        submodelparts=submodelparts,
    )


def _load_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    # The benchmark parameter files use JSON-with-comments syntax.
    text = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    return json.loads(text)


def _sorted_rows(coords: np.ndarray) -> np.ndarray:
    if coords.size == 0:
        return coords.reshape(0, 2)
    keys = np.lexsort((np.round(coords[:, 1], 12), np.round(coords[:, 0], 12)))
    return np.asarray(coords[keys], dtype=float)


def _cylinder_geometry(fluid: MDPAMesh) -> tuple[tuple[float, float], float]:
    coords = fluid.submodelpart_coords("NoSlip2D_Cylinder")
    center = coords.mean(axis=0)
    radii = np.linalg.norm(coords - center[None, :], axis=1)
    return (float(center[0]), float(center[1])), float(radii.mean())


def load_double_flap_reference(root: str | Path | None = None) -> DoubleFlapReference:
    root_path = default_double_flap_root() if root is None else Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"DoubleFlap reference directory not found: {root_path}")

    fluid = _parse_mdpa(root_path / "Double_Flap_Mesh" / "testing_22_Fluid.mdpa")
    solid = _parse_mdpa(root_path / "Double_Flap_Mesh" / "solid_double_mesh_Structural.mdpa")
    fluid_json = _load_json(root_path / "ProjectParametersCFD.json")
    solid_json = _load_json(root_path / "ProjectParametersCSM.json")
    coupling_json = _load_json(root_path / "DoubleFlap_fsi_parameters_ROM.json")
    fluid_materials_json = _load_json(root_path / "FluidMaterials.json")

    fluid_coords = np.asarray(list(fluid.nodes.values()), dtype=float)
    solid_coords = np.asarray(list(solid.nodes.values()), dtype=float)
    channel_length = float(fluid_coords[:, 0].max())
    channel_height = float(fluid_coords[:, 1].max())
    solid_bbox = (
        float(solid_coords[:, 0].min()),
        float(solid_coords[:, 0].max()),
        float(solid_coords[:, 1].min()),
        float(solid_coords[:, 1].max()),
    )

    interface_fluid = _sorted_rows(fluid.submodelpart_coords("NoSlip2D_Interface"))
    interface_solid = _sorted_rows(solid.submodelpart_coords("StructureInterface2D_Struc_Fsi"))
    if interface_fluid.shape != interface_solid.shape:
        raise ValueError(
            "Fluid and solid interface groups have different sizes: "
            f"{interface_fluid.shape} vs {interface_solid.shape}"
        )
    pairwise = np.linalg.norm(interface_fluid[:, None, :] - interface_solid[None, :, :], axis=2)
    mismatch = np.maximum(pairwise.min(axis=1), pairwise.min(axis=0))
    cylinder_center, cylinder_radius = _cylinder_geometry(fluid)

    bc_processes = fluid_json["processes"]["boundary_conditions_process_list"]
    inlet_ramp = bc_processes[0]["Parameters"]
    inlet_steady = bc_processes[1]["Parameters"]
    material = fluid_materials_json["properties"][0]["Material"]["Variables"]
    density = float(material["DENSITY"])
    dynamic_viscosity = float(material["DYNAMIC_VISCOSITY"])
    kinematic_viscosity = dynamic_viscosity / max(density, 1.0e-14)

    return DoubleFlapReference(
        root=root_path,
        fluid=fluid,
        solid=solid,
        fluid_project_parameters=fluid_json,
        solid_project_parameters=solid_json,
        coupling_parameters=coupling_json,
        channel_length=channel_length,
        channel_height=channel_height,
        solid_bbox=solid_bbox,
        interface_node_count=int(interface_solid.shape[0]),
        interface_max_mismatch=float(mismatch.max(initial=0.0)),
        clamp_node_count=len(solid.submodelparts["DISPLACEMENT_BCDisp"].node_ids),
        fluid_time_step=float(
            fluid_json["solver_settings"]["fluid_solver_settings"]["time_stepping"]["time_step"]
        ),
        solid_time_step=float(solid_json["solver_settings"]["time_stepping"]["time_step"]),
        end_time=float(fluid_json["problem_data"]["end_time"]),
        inlet_ramp_end_time=float(inlet_ramp["interval"][1]),
        inlet_modulus_ramp=str(inlet_ramp["modulus"]),
        inlet_modulus_steady=str(inlet_steady["modulus"]),
        density=density,
        kinematic_viscosity=kinematic_viscosity,
        max_velocity=2.5,
        cylinder_center=cylinder_center,
        cylinder_radius=cylinder_radius,
    )
