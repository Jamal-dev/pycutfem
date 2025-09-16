# pycutfem/core/sideconvention.py
from dataclasses import dataclass

@dataclass
class SideConvention:
    """
    Defines the convention for classifying points relative to a level-set function (phi).
    """
    # If True, the "+" side corresponds to phi >= 0. If False, it corresponds to phi <= 0.
    pos_is_phi_nonnegative: bool = True
    # If True, points on the interface (abs(phi) <= tol) are considered part of the "+" side.
    zero_belongs_to_pos: bool = True
    # If True, points on the interface (abs(phi) <= tol) are considered part of the "−" side.
    zero_belongs_to_neg: bool = False
    # The tolerance for determining if a point is on the interface.
    tol: float = 1e-12

    def is_pos(self, phi: float, tol: float = None) -> bool:
        """Checks if a point is on the positive side according to the convention."""
        if tol is None:
            tol = self.tol
        # Handle the interface case first
        if abs(phi) <= tol:
            return self.zero_belongs_to_pos
        # Handle the bulk domain case
        if self.pos_is_phi_nonnegative:
            return phi > 0.0
        else:
            return phi < 0.0

    def is_neg(self, phi: float, tol: float = None) -> bool:
        """
        Checks if a point is on the negative side according to the convention.

        This is now independent of `is_pos` for points on the interface.
        """
        if tol is None:
            tol = self.tol
        # Handle the interface case based on its own independent flag
        if abs(phi) <= tol:
            return self.zero_belongs_to_neg
        # Handle the bulk domain case (which is always the opposite of the positive bulk)
        if self.pos_is_phi_nonnegative:
            return phi < 0.0
        else:
            return phi > 0.0
    def is_zero(self, phi: float, tol: float = None) -> bool:
        """Checks if a point is on the interface according to the tolerance."""
        if tol is None:
            tol = self.tol
        return abs(phi) <= tol

    def label(self, phi: float, tol: float = None) -> str:
        """Returns a label ('+', '−', 'interface', or 'gap') for a given phi value."""
        if tol is None:
            tol = self.tol

        is_p = self.is_pos(phi, tol)
        is_n = self.is_neg(phi, tol)

        if is_p and is_n:
            return 'interface'  # Belongs to both sides
        elif is_p:
            return '+'
        elif is_n:
            return '-'
        else:
            return 'gap'  # Belongs to neither side

# Global, editable in one place:
SIDE = SideConvention(
    pos_is_phi_nonnegative=True,
    zero_belongs_to_pos=False,
    zero_belongs_to_neg=False # Default: interface does not belong to the negative side
)

# def set_side_convention(
#     *,
#     pos_is_phi_nonnegative: bool = True,
#     zero_belongs_to_pos: bool = True,
#     zero_belongs_to_neg: bool = False
# ):
#     """
#     Updates the global side convention.
#     """
#     global SIDE
#     SIDE = SideConvention(
#         pos_is_phi_nonnegative=pos_is_phi_nonnegative,
#         zero_belongs_to_pos=zero_belongs_to_pos,
#         zero_belongs_to_neg=zero_belongs_to_neg
#     )