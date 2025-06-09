import numpy as np
from pycutfem.fem.operators import jump, average, inner
def test_jump_average_inner():
    gradL = np.array([[1,0],[0,1]])
    gradR = np.array([[0,1],[1,0]])
    n = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
    j = jump(gradL, gradR, n)
    a = average(gradL, gradR)
    assert j.shape == (2,)
    assert a.shape == (2,2)
    g = inner(gradL, gradL)
    assert g.shape == (2,)
