# pycutfem/core/sideconvention.py
from dataclasses import dataclass

@dataclass
class SideConvention:
    # If True: "+" means "phi >= 0"; if False: "+" means "phi <= 0"
    pos_is_phi_nonnegative: bool = True
    # tie-breaker at exactly 0 (within tol) goes to "+" if True, else to "−"
    zero_belongs_to_pos: bool = True
    tol: float = 1e-16


    def is_pos(self, phi: float, tol=None) -> bool:
        if tol is None:
            tol = self.tol
        # normalize sign with tolerance
        if abs(phi) <= tol:
            return self.zero_belongs_to_pos
        if self.pos_is_phi_nonnegative:
            return phi > 0.0
        else:
            return phi < 0.0

    def is_neg(self, phi: float, tol=None) -> bool:
        if tol is None:
            tol = self.tol
        return not self.is_pos(phi, tol)

    def label(self, phi: float, tol=None) -> str:
        if tol is None:
            tol = self.tol
        return '+' if self.is_pos(phi, tol) else '-'

# Global, editable in one place:
SIDE = SideConvention(
    pos_is_phi_nonnegative=True,   # ← set False to flip to NG’s convention
    zero_belongs_to_pos=True       # ← choose where φ≈0 goes
)

def set_side_convention(*, pos_is_phi_nonnegative: bool = True, zero_belongs_to_pos: bool = True):
    global SIDE
    SIDE = SideConvention(pos_is_phi_nonnegative, zero_belongs_to_pos)
