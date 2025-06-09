from pycutfem.fem.reference import get_reference
import numpy as np
def test_shape_sum():
    tri=get_reference('tri',1)
    N=tri.shape(1/3,1/3)
    assert np.isclose(N.sum(),1.0)
