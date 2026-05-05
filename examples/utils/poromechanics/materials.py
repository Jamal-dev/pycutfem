"""Material data for example-level 2D U-Pl poromechanics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class UPlMaterial2D:
    """Small-strain isotropic 2D U-Pl material data.

    The storage coefficient follows Kratos Poromechanics when
    ``storage_inverse`` is not supplied:

        1/M = (alpha - phi) / K_s + phi / K_f

    Existing pycutfem/FEniCS consolidation examples often prescribe ``M``
    directly. For those cases, pass ``storage_inverse=1.0 / M`` to keep the
    benchmark exactly unchanged.
    """

    young_modulus: float
    poisson_ratio: float
    porosity: float
    biot_coefficient: float
    permeability_xx: float
    permeability_yy: float | None = None
    permeability_xy: float = 0.0
    dynamic_viscosity_liquid: float = 1.0
    bulk_modulus_solid: float | None = None
    bulk_modulus_liquid: float | None = None
    storage_inverse: float | None = None
    density_solid: float = 0.0
    density_liquid: float = 0.0
    thickness: float = 1.0

    @property
    def lambda_(self) -> float:
        nu = float(self.poisson_ratio)
        E = float(self.young_modulus)
        return E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    @property
    def mu(self) -> float:
        nu = float(self.poisson_ratio)
        E = float(self.young_modulus)
        return E / (2.0 * (1.0 + nu))

    @property
    def biot_modulus_inverse(self) -> float:
        if self.storage_inverse is not None:
            return float(self.storage_inverse)
        if self.bulk_modulus_solid is None or self.bulk_modulus_liquid is None:
            raise ValueError(
                "Either storage_inverse or both bulk_modulus_solid and "
                "bulk_modulus_liquid must be provided."
            )
        alpha = float(self.biot_coefficient)
        phi = float(self.porosity)
        return (alpha - phi) / float(self.bulk_modulus_solid) + phi / float(self.bulk_modulus_liquid)

    @property
    def mixture_density(self) -> float:
        phi = float(self.porosity)
        return phi * float(self.density_liquid) + (1.0 - phi) * float(self.density_solid)

    @property
    def liquid_density(self) -> float:
        return float(self.density_liquid)

    @property
    def permeability_matrix(self) -> np.ndarray:
        kyy = self.permeability_xx if self.permeability_yy is None else self.permeability_yy
        return np.asarray(
            [
                [float(self.permeability_xx), float(self.permeability_xy)],
                [float(self.permeability_xy), float(kyy)],
            ],
            dtype=float,
        )

    @property
    def darcy_conductivity_matrix(self) -> np.ndarray:
        viscosity = float(self.dynamic_viscosity_liquid)
        if viscosity <= 0.0:
            raise ValueError("dynamic_viscosity_liquid must be positive.")
        return self.permeability_matrix / viscosity
