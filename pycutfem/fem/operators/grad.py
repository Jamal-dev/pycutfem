"""pycutfem.fem.operators.grad"""
import numpy as np
def jump(grad_left, grad_right, normal):
    return np.dot(grad_left, normal)-np.dot(grad_right, normal)
def average(grad_left, grad_right):
    return 0.5*(grad_left+grad_right)
