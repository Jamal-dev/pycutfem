"""
Example showing how to import a 2D Gmsh mesh with a rectangular domain that
contains a circular hole.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List

import gmsh

from pycutfem.utils.gmsh_loader import mesh_from_gmsh


def build_rectangle_with_hole(
    msh_path: Path,
    *,
    width: float = 2.0,
    height: float = 1.0,
    hole_radius: float = 0.25,
    mesh_size: float = 0.05,
) -> None:
    gmsh.initialize()
    try:
        gmsh.model.add("rect_with_hole")
        rect_tag = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, width, height)
        hole_tag = gmsh.model.occ.addDisk(width * 0.5, height * 0.5, 0.0, hole_radius, hole_radius)
        cut_entities, _ = gmsh.model.occ.cut(
            [(2, rect_tag)],
            [(2, hole_tag)],
            removeObject=True,
            removeTool=True,
        )
        if not cut_entities:
            raise RuntimeError("Boolean difference failed when creating the hole.")
        surface_tag = cut_entities[0][1]
        gmsh.model.occ.synchronize()

        gmsh.model.addPhysicalGroup(2, [surface_tag], tag=1)
        gmsh.model.setPhysicalName(2, 1, "domain")

        boundary = gmsh.model.getBoundary([(2, surface_tag)], oriented=False, recursive=False)
        outer_curves: List[int] = []
        inner_curves: List[int] = []
        for dim, tag in boundary:
            if dim != 1:
                continue
            xmin, ymin, _, xmax, ymax, _ = gmsh.model.occ.getBoundingBox(dim, tag)
            cx = 0.5 * (xmin + xmax)
            cy = 0.5 * (ymin + ymax)
            if (cx - width * 0.5) ** 2 + (cy - height * 0.5) ** 2 < (hole_radius * 0.9) ** 2:
                inner_curves.append(tag)
            else:
                outer_curves.append(tag)
        if outer_curves:
            outer_tag = gmsh.model.addPhysicalGroup(1, outer_curves)
            gmsh.model.setPhysicalName(1, outer_tag, "outer_wall")
        if inner_curves:
            inner_tag = gmsh.model.addPhysicalGroup(1, inner_curves)
            gmsh.model.setPhysicalName(1, inner_tag, "inner_wall")

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
        gmsh.model.mesh.generate(2)
        msh_path.parent.mkdir(parents=True, exist_ok=True)
        gmsh.write(str(msh_path))
    finally:
        gmsh.finalize()


def main():
    width = 2.0
    height = 1.0
    hole_radius = 0.25
    mesh_size = 0.05

    msh_path = Path("examples/meshes/rectangle_with_hole.msh")
    build_rectangle_with_hole(
        msh_path,
        width=width,
        height=height,
        hole_radius=hole_radius,
        mesh_size=mesh_size,
    )

    mesh = mesh_from_gmsh(msh_path)
    area = float(mesh.areas_list.sum())
    expected = width * height - math.pi * hole_radius**2
    rel_err = abs(area - expected) / expected

    inner_edges = sum(1 for edge in mesh.edges_list if edge.tag and "inner_wall" in edge.tag)
    outer_edges = sum(1 for edge in mesh.edges_list if edge.tag and "outer_wall" in edge.tag)

    print(f"Mesh imported from {msh_path}")
    print(mesh)
    print(f"Geometric area  : {area:.6f}")
    print(f"Expected area   : {expected:.6f}")
    print(f"Relative error  : {rel_err:.2e}")
    print(f"Outer edge tags : {outer_edges}")
    print(f"Inner edge tags : {inner_edges}")


if __name__ == "__main__":
    main()
