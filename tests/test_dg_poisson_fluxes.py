import pytest

from examples.dg_poisson_fluxes import solve_poisson_dg


@pytest.mark.parametrize(
    "symmetry, tol",
    [
        (1, 0.12),   # SIPG
        (0, 0.20),   # IIPG
        (-1, 0.20),  # NIPG
    ],
)
def test_dg_poisson_fluxes(symmetry, tol):
    err = solve_poisson_dg(symmetry=symmetry, nx=6, ny=6, poly_order=1)
    assert err < tol
