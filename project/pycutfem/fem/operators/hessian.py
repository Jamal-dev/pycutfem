"""pycutfem.fem.operators.hessian"""
import numpy as np
def laplacian(hess):
    return hess[:,0,0]+hess[:,1,1]
def sym_grad(grad):
    return 0.5*(grad+grad.T)
