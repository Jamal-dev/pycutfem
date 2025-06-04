import numpy as np
from pycutfem.integration import quadrature as q

def integrate_ref_tri(func, order):
    pts, wts = q.volume('tri', order)
    fvals = np.array([func(xy) for xy in pts])
    return (fvals * wts).sum()

def integrate_ref_quad(func, order):
    pts, wts = q.volume('quad', order)
    fvals = np.array([func(xy) for xy in pts])
    return (fvals * wts).sum()

def test_constant_volume():
    for et in ('tri','quad'):
        pts,wts=q.volume(et,3)
        area = wts.sum()
        exact = 0.5 if et=='tri' else 4.0
        assert np.isclose(area, exact, rtol=1e-12)

def test_linear_exact_tri():
    # ∫_T r dA  over reference triangle  = 1/6
    val = integrate_ref_tri(lambda xy: xy[0], order=4)
    assert np.isclose(val, 1/6, rtol=1e-12)

def test_edge_rule_quad():
    pts,wts = q.edge('quad', 0, 3)
    # integrate 1 over bottom edge [-1,1] so length 2
    length = (wts).sum()
    assert np.isclose(length, 2.0, rtol=1e-12)
