"""
Utilities for generating quadrilateral lid-driven cavity meshes with Gmsh.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import gmsh


def _classify_edge(tag: int, bounds: Tuple[float, float, float, float, float, float], *, L: float, H: float) -> str:
    """
    Determine which cavity wall a boundary curve belongs to using its bounding box.
    """
    xmin, ymin, _, xmax, ymax, _ = bounds
    tol = 1e-6
    if abs(ymin) < tol and abs(ymax) < tol:
        return "bottom_wall"
    if abs(ymin - H) < tol and abs(ymax - H) < tol:
        return "top_lid"
    if abs(xmin) < tol and abs(xmax) < tol:
        return "left_wall"
    if abs(xmin - L) < tol and abs(xmax - L) < tol:
        return "right_wall"
    raise RuntimeError(f"Unable to classify boundary curve {tag} (bounds={bounds}) for a {L}x{H} cavity.")


def build_caity_quad_mesh(
    output: Path,
    *,
    L: float = 1.0,
    H: float = 1.0,
    nx: int = 30,
    ny: int = 30,
    element_order: int = 2,
    visualize: bool = False,
) -> Path:
    """
    Generate a structured quadrilateral lid-driven cavity mesh.

    Args:
        output: Destination ``.msh`` file path.
        L, H: Physical cavity dimensions.
        nx, ny: Number of elements along the x and y directions.
        element_order: Geometric order assigned to Gmsh elements (1 or 2).
        visualize: Launch the Gmsh GUI before writing the mesh when True.
    """
    output = Path(output)
    gmsh.initialize()
    try:
        gmsh.model.add("lid_driven_cavity")
        rect_tag = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, L, H)
        gmsh.model.occ.synchronize()

        gmsh.model.addPhysicalGroup(2, [rect_tag], tag=1)
        gmsh.model.setPhysicalName(2, 1, "fluid")

        boundary_entities = gmsh.model.getBoundary([(2, rect_tag)], oriented=False, recursive=False)
        curve_tags = [tag for dim, tag in boundary_entities if dim == 1]
        if len(curve_tags) != 4:
            raise RuntimeError("Expected four boundary curves for the rectangular cavity.")

        transfinite_counts: Dict[str, int] = {
            "bottom_wall": nx + 1,
            "top_lid": nx + 1,
            "left_wall": ny + 1,
            "right_wall": ny + 1,
        }
        curve_groups: Dict[str, list[int]] = {name: [] for name in transfinite_counts}
        for dim, curve in boundary_entities:
            if dim != 1:
                continue
            bounds = gmsh.model.occ.getBoundingBox(1, curve)
            name = _classify_edge(curve, bounds, L=L, H=H)
            gmsh.model.mesh.setTransfiniteCurve(curve, transfinite_counts[name])
            curve_groups[name].append(curve)

        gmsh.model.mesh.setTransfiniteSurface(rect_tag)
        gmsh.option.setNumber("Mesh.Algorithm", 8)  # Frontal quad
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(element_order)

        for name, tags in curve_groups.items():
            if not tags:
                continue
            tag = gmsh.model.addPhysicalGroup(1, tags)
            gmsh.model.setPhysicalName(1, tag, name)

        if visualize:
            gmsh.fltk.initialize()
            gmsh.fltk.run()

        output.parent.mkdir(parents=True, exist_ok=True)
        gmsh.write(str(output))
        return output
    finally:
        gmsh.finalize()


# Backwards compatibility: keep the old function name available for callers.
build_cavity_quad_mesh = build_caity_quad_mesh

__all__ = ["build_caity_quad_mesh", "build_cavity_quad_mesh"]
