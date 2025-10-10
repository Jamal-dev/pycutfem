"""Quick comparison of ghost-penalty mass and diffusion terms between
PyCutFEM and NGSolve on a simple structured mesh.

Run with the desired backend (python/jit) by setting
    PYCUTFEM_BACKEND=python python examples/debug/compare_ghost_terms.py
"""

import numpy as np
from ngsolve import CoefficientFunction, InnerProduct, Grad, TaskManager
from examples.compare_with_ngsolve import (
    setup_pc,
    setup_ng,
    PC_DX,
    NG_DX,
    assemble_and_energy_pc,
    assemble_and_energy_ng,
    pc_grad,
    pc_inner,
    pc_jump,
)


def setup_const_vectors(ng_setup):
    const_vec = CoefficientFunction((1, 1))
    const_scalar = CoefficientFunction(1.0)
    with TaskManager():
        # velocity components (neg, pos)
        ng_setup['gfu'].components[0].components[0].Set(const_vec)
        ng_setup['gfv'].components[0].components[0].Set(const_vec)
        ng_setup['gfu'].components[0].components[1].Set(const_vec)
        ng_setup['gfv'].components[0].components[1].Set(const_vec)
        # pressure components
        ng_setup['gfu'].components[1].components[0].Set(const_scalar)
        ng_setup['gfv'].components[1].components[0].Set(const_scalar)
        ng_setup['gfu'].components[1].components[1].Set(const_scalar)
        ng_setup['gfv'].components[1].components[1].Set(const_scalar)
        # number space (lagrange multiplier)
        ng_setup['gfu'].components[2].Set(const_scalar)
        ng_setup['gfv'].components[2].Set(const_scalar)


def compare_ghost_terms(backend='python'):
    maxh, order, R = 0.125, 2, 2.0/3.0
    use_quad = True

    pc = setup_pc(maxh, order, R, L=2.0, H=2.0, use_quad=use_quad)
    ng = setup_ng(maxh, order, R, L=2.0, H=2.0, use_quad=use_quad)

    quad_order = 8
    pc_dx = PC_DX(quad_order, pc['level_set'], pc['es'])
    ng_dx = NG_DX(quad_order, ng['lsetp1'], ng['ci'], ghost_facets=ng['ghost_facets'])

    setup_const_vectors(ng)
    total_dofs = pc['dh'].total_dofs
    u_vec = np.ones(total_dofs)
    v_vec = np.ones(total_dofs)

    tests = {
        'ghost_mass': {
            'pc_form': pc_inner(pc_jump(pc['up']), pc_jump(pc['vp'])) * pc_dx.ghost(),
            'ng_form': (
                InnerProduct(ng['u'][0] - ng['u'][0].Other(),
                              ng['v'][0] - ng['v'][0].Other()) * ng_dx.ghost(side='-') +
                InnerProduct(ng['u'][1] - ng['u'][1].Other(),
                              ng['v'][1] - ng['v'][1].Other()) * ng_dx.ghost(side='+')
            )
        },
        'ghost_diffusion': {
            'pc_form': pc_inner(pc_grad(pc_jump(pc['up'])),
                                pc_grad(pc_jump(pc['vp']))) * pc_dx.ghost(),
            'ng_form': (
                InnerProduct(Grad(ng['u'][0]) - Grad(ng['u'][0].Other()),
                              Grad(ng['v'][0]) - Grad(ng['v'][0].Other())) * ng_dx.ghost(side='-') +
                InnerProduct(Grad(ng['u'][1]) - Grad(ng['u'][1].Other()),
                              Grad(ng['v'][1]) - Grad(ng['v'][1].Other())) * ng_dx.ghost(side='+')
            )
        },
    }

    for name, forms in tests.items():
        pc_energy = assemble_and_energy_pc(forms['pc_form'], pc['dh'], u_vec, v_vec, backend=backend)
        bf = BilinearForm(ng['WhG'])
        bf += forms['ng_form']
        ng_energy = assemble_and_energy_ng(bf, ng['gfu'], ng['gfv'])
        diff = abs(pc_energy - ng_energy)
        print(f"{name}: PC={pc_energy:+.12e}  NG={ng_energy:+.12e}  |diff|={diff:.3e}")


if __name__ == "__main__":
    import os
    backend = os.getenv('PYCUTFEM_BACKEND', 'python').lower()
    compare_ghost_terms(backend=backend)
