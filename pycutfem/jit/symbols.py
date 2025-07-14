# pycutfem/jit/symbols.py
SENTINEL     = "__CFM__"
TAIL         = "__ARR"
POS_SUFFIX   = "__pos"
NEG_SUFFIX   = "__neg"

def encode(field: str, side: str | None) -> str:
    """
    side ∈ {None, "+", "-"}  →  interior / exterior
    """
    core = f"{SENTINEL}{field}"
    if   side == "+": core += POS_SUFFIX
    elif side == "-": core += NEG_SUFFIX
    return core + TAIL

def decode_coeff(sym: str) -> tuple[str, str | None]:
    """
    Decode the simplified CutFEM coefficient names produced after the
    recent refactor, e.g.

        u_u_k_loc
        u_beta_loc
        u_u_neg_loc
        u_u_pos_loc_loc        (after extra rewriting)
        u_u_k_loc_e            (after element expansion)

    Returns: (field_name, side) with side ∈ {None, "pos", "neg"}.
    """

    # ---- 1. remove *one* leading 'u_' wrapper ---------------------------
    if sym.startswith("u_"):
        sym = sym[2:]

    # ---- 2. drop every trailing '_loc' or '_loc_e' ----------------------
    while sym.endswith("_loc_e") or sym.endswith("_loc"):
        sym = sym[:-6] if sym.endswith("_loc_e") else sym[:-4]

    # ---- 3. pick off the side suffix ------------------------------------
    side = None
    if sym.endswith(POS_SUFFIX):
        sym, side = sym[:-len(POS_SUFFIX)], "pos"
    elif sym.endswith(NEG_SUFFIX):
        sym, side = sym[:-len(NEG_SUFFIX)], "neg"

    return sym, side
