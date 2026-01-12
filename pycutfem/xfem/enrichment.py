from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

from pycutfem.ufl.helpers import HelpersFieldAware as _hfa


def alpha_from_side_masks(pos_mask: np.ndarray, neg_mask: np.ndarray, *, side: str) -> np.ndarray:
    """
    Compute the shifted-Heaviside enrichment multiplier α per local DOF.

    α_i = H(side) - H(φ(x_i))

    where H(side) is 1 on '+' integration, 0 on '-' integration.
    Interface DOFs (pos_mask==neg_mask==1) are assigned α=0 on both sides.
    """
    side = str(side).strip()
    if side not in {"+", "-"}:
        raise ValueError("side must be '+' or '-'.")
    pos_mask = np.asarray(pos_mask, dtype=float).ravel()
    neg_mask = np.asarray(neg_mask, dtype=float).ravel()
    if pos_mask.shape != neg_mask.shape:
        raise ValueError("pos_mask and neg_mask must have the same shape.")

    interface = (pos_mask > 0.5) & (neg_mask > 0.5)
    H_int = 1.0 if side == "+" else 0.0

    # H_i: 1 on positive side, 0 on negative side; for interface nodes set to H_int.
    H_i = pos_mask.copy()
    if side == "-":
        H_i[interface] = 0.0
    # side '+' keeps H_i=1 for interface (already true).

    alpha = H_int - H_i
    # ensure interface alpha is exactly 0
    alpha[interface] = 0.0
    return alpha


def build_alpha_by_field(
    dh,
    fields: Sequence[str],
    eid: int,
    level_set,
    *,
    side: str,
    tol: float = 0.0,
) -> Dict[str, np.ndarray]:
    """
    Convenience: compute α arrays per field for an element.
    """
    pos_masks, neg_masks = _hfa.build_side_masks_by_field(dh, fields, int(eid), level_set, tol=tol)
    out: Dict[str, np.ndarray] = {}
    for f in fields:
        pm = pos_masks.get(f)
        nm = neg_masks.get(f)
        if pm is None or nm is None:
            continue
        out[f] = alpha_from_side_masks(pm, nm, side=side)
    return out

