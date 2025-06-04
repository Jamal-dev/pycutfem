"""pycutfem.fem.operators.inner"""
import numpy as np
def inner(a,b):
    a=np.asarray(a); b=np.asarray(b)
    if a.shape!=b.shape:
        raise ValueError('shape mismatch')
    if a.ndim==1:
        return a*b
    flat_a=a.reshape(a.shape[0], -1)
    flat_b=b.reshape(b.shape[0], -1)
    return np.einsum('ij,ij->i', flat_a, flat_b)
