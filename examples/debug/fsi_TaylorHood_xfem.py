#!/usr/bin/env python
# coding: utf-8

# # Fluid-structure interaction with Taylor-Hood elements
# 
# In this section we will use Taylor-Hood elements for discretizing the Navier-Stokes equations and Lagrangian elements for the elastic wave equations. We recall that, [see introduction of FSI](fsi_intro.ipynb), the Navier-Stokes equations in ALE form read
# \begin{align*}
# &\int_{\Omega^f}J(\rho\frac{\partial \hat{u}}{\partial t}\cdot\hat{v}+\rho((\hat{u}-\dot{d})\cdot\nabla)\hat{u}F^{-1}\cdot\hat{v}+2\rho\nu \mathrm{sym}(\nabla\hat{u}F^{-1}):\mathrm{sym}(\nabla\hat{v}F^{-1})-\mathrm{tr}(\nabla\hat{v}F^{-1})p)\,dx = 0&&\qquad \forall \hat{v},\\
# &\int_{\Omega^f}J\,\mathrm{tr}(\nabla\hat{u}F^{-1})q\,dx = 0&&\qquad \forall q.
# \end{align*}
# and the elastic wave equation as first order system
# \begin{align*}
# & \int_{\Omega^s}\frac{\partial d}{\partial t}\cdot v \,dx = \int_{\Omega^s}u\cdot v\,dx &&\qquad \forall v,\\
# &\int_{\Omega^s}\rho\frac{\partial u}{\partial t}\cdot w+(F\Sigma):\nabla w\,dx = 0&&\qquad \forall w.
# \end{align*}
# 
# We will consider the following benchmark proposed in [<a href="https://doi.org/10.1007/3-540-34596-5_15">Turek, Hron. Proposal for numerical benchmarking of fluid-structure interaction between an elastic object and laminar incompressible flow. <i> In: Fluid-Structure Interaction: Modelling, Simulation, Optimisation</i>, 2006</a>]. It is based on the classical flow around cylinder benchmark [<a href="https://doi.org/10.1007/978-3-322-89849-4_39">Schäfer, Turek, Durst, Krause, Rannacher. Benchmark computations of laminar flow around a cylinder. <i> In: Flow simulation with high-performance computers II</i>, 1996</a>], where additionally an elastic flag is "glued" behind the obstacle.
# 
# <img src="figures/turek_solid_fluid_domain.png" width="400" align="center"/>
# 
# We choose the space dependent function $h(x)$ in the deformation extension problem to be large close to the elastic solid's tip and decreases with the distance.

# In[ ]:


from ngsolve import *
from netgen.occ import *
from ngsolve.webgui import Draw
import ipywidgets as widgets

tau = 0.004
tend = 5
order = 3

rhos, nus, mus, rhof, nuf, U = 1e4, 0.4, 0.5 * 1e6, 1e3, 1e-3, 1
ls = 2 * mus * nus / (1 - 2 * nus)

# Parabolic inflow profile at the inlet
par = Parameter(0)
u_inflow = par * CoefficientFunction((4 * U * 1.5 * y * (0.41 - y) / (0.41 * 0.41), 0))


# magnitude of inflow velocity over time
def Force(t):
    if t < 0:
        return 0
    elif t < 2:
        return (1 - cos(pi / 2.0 * t)) / 2.0
    else:
        return 1


# directly start with Stokes solution instead of increasing inflow
start_with_stokes = True


def GenerateMesh(order, maxh=0.2):
    circle = Circle((0.2, 0.2), r=0.05).Face()
    circle.edges.name = "circ"
    fluid = Rectangle(2.5, 0.41).Face()
    fluid.faces.name = "fluid"
    fluid.edges.Min(X).name = "inlet"
    fluid.edges.Max(X).name = "outlet"
    fluid.edges.Min(Y).name = "wall"
    fluid.edges.Max(Y).name = "wall"
    solid = (
        MoveTo(0.248989794855664, 0.19).Rectangle(0.6 - 0.248989794855664, 0.02).Face()
    )
    solid.faces.name = "solid"
    solid.edges.name = "interface"
    solid.edges.Min(X).name = "circ_inner"

    domain_fluid = (fluid - circle) - solid
    domain = Glue([domain_fluid, solid])

    mesh = Mesh(OCCGeometry(domain, dim=2).GenerateMesh(maxh=maxh))
    mesh.Curve(order)

    return mesh


mesh = GenerateMesh(order=order)
Draw(mesh);


# For the spatial discretization we will use the Taylor-Hood elements for the Navier-Stokes equations and also H1-conforming elements for the elastic wave equation. Thus, we can use one global space for the velocity and the displacement. With the definedon flag, we can tell the pressure space, that it lives only on the fluid domain.

# In[ ]:


V = VectorH1(mesh, order=order, dirichlet="inlet|wall|circ|circ_inner")
Q = H1(mesh, order=order - 1, definedon="fluid")
D = VectorH1(mesh, order=order, dirichlet="inlet|wall|outlet|circ|circ_inner")

X = V * Q * D
Y = V * Q
(u, p, d), (v, q, w) = X.TnT()

gf_solution = GridFunction(X)
gf_solution_old = GridFunction(X)

velocity, pressure, deformation = gf_solution.components
velocity_old, pressure_old, deformation_old = gf_solution_old.components

gradu_old = Grad(velocity_old)
gradd_old = Grad(deformation_old)

I = Id(mesh.dim)


def CalcStresses(A):
    F = A + I
    C = F.trans * F
    E = 0.5 * (C - I)
    J = Det(F)
    Finv = Inv(F)
    return (F, C, E, J, Finv)


F, C, E, J, Finv = CalcStresses(Grad(d))
F_old, C_old, E_old, J_old, Finv_old = CalcStresses(gradd_old)


def Stress(mat):
    return mus * mat + ls / 2 * Trace(mat) * I


# For the time discretization we will use the Crank-Nicolson method
# 
# \begin{align*}
# \int_{t^n}^{t^{n+1}} f(s)\,ds \approx \frac{\tau}{2}(f(t^{n+1})+f(t^n)).
# \end{align*}
# 
# Only the pressure constraint is handled with implicit Euler.

# In[ ]:


# For Stokes problem, check_unused=False to avoid warning not solving on the solid
stokes = BilinearForm(Y, symmetric=True, check_unused=False)
stokes += (
    nuf * rhof * 2 * InnerProduct(Sym(Grad(u)), Sym(Grad(v)))
    - div(u) * q
    - div(v) * p
    - 1e-8 * p * q
) * dx("fluid")
stokes.Assemble()

true_compile = False

bfa = BilinearForm(X, symmetric=False, check_unused=False, condense=True)
########################### Fluid: Navier-Stokes ##########################
# M du/dt
bfa += (rhof / tau * (InnerProduct(0.5 * (J + J_old) * (u - velocity_old), v))).Compile(
    true_compile, wait=True
) * dx("fluid")
# symmetric stress div (eps u)
bfa += (
    0.5
    * rhof
    * nuf
    * (
        InnerProduct(J * 2 * Sym(Grad(u) * Finv), Sym(Grad(v) * Finv))
        + InnerProduct(J_old * 2 * Sym(gradu_old * Finv_old), (Grad(v) * Finv_old))
    )
).Compile(true_compile, wait=True) * dx("fluid")
# Convection and mesh-velocity
bfa += (
    0.5
    * rhof
    * (
        InnerProduct(J * (Grad(u) * Finv) * (u - (d - deformation_old) / tau), v)
        + InnerProduct(
            J_old
            * (gradu_old * Finv_old)
            * (velocity_old - (d - deformation_old) / tau),
            v,
        )
    )
).Compile(true_compile, wait=True) * dx("fluid")
# Pressure/Constraint implicit
bfa += (-J * (Trace(Grad(v) * Finv) * p + Trace(Grad(u) * Finv) * q)).Compile(
    true_compile, wait=True
) * dx("fluid")

########################### Solid: elastic wave ##########################
# M du/dt
bfa += (rhos / tau * InnerProduct(u - velocity_old, v)).Compile(
    true_compile, wait=True
) * dx("solid")
# Material law
bfa += (InnerProduct(F * Stress(E) + F_old * Stress(E_old), Grad(v))).Compile(
    true_compile, wait=True
) * dx("solid")
# dd/dt = u
bfa += (InnerProduct(u + velocity_old - 2.0 / tau * (d - deformation_old), w)).Compile(
    true_compile, wait=True
) * dx("solid")


########################## Deformation extension ##########################
def minCF(a, b):
    return IfPos(a - b, b, a)


gf_dist = GridFunction(H1(mesh, order=2))
gf_dist.Set(
    minCF(
        (x - 0.6) * (x - 0.6) + (y - 0.19) * (y - 0.19),
        (x - 0.6) * (x - 0.6) + (y - 0.21) * (y - 0.21),
    )
)


def NeoHookExt(C, mu=1, lam=1):
    return 0.5 * mu * (Trace(C - I) + 2 * mu / lam * Det(C) ** (-lam / 2 / mu) - 1)


bfa += Variation(
    (1e-20 * mus * (1 / sqrt(gf_dist * gf_dist + 1e-12)) * NeoHookExt(C)).Compile(
        true_compile, wait=True
    )
    * dx("fluid")
)

Draw(1 / sqrt(gf_dist * gf_dist + 1e-12), mesh, "h(x)", min=0.1, max=100, order=3);


# To increase the inflow velocity depending on time, we have to extend the new velocity into the domain, before solving the system. This can be done by solving a Stokes problem with the new velocity as dirichlet data only on the fluid domain. To tell the solver on which domain it should work, we have to define the according degrees of freedom, which is done in terms of bitarrays.

# In[ ]:


bt_stokes = Y.FreeDofs() & ~Y.GetDofs(mesh.Materials("solid"))
bt_stokes &= ~Y.GetDofs(mesh.Boundaries("wall|inlet|circ|interface|circ_inner"))
# set all pressure dofs as active for Stokes
bt_stokes[V.ndof :] = True

gf_stokes = GridFunction(Y)
res_stokes = gf_stokes.vec.CreateVector()
inv_stokes = stokes.mat.Inverse(bt_stokes, inverse="sparsecholesky")


# Reset and draw.

# In[ ]:


t = 0
i = 0


# Calculate quantities of interest
def CalcForces(disp_x, disp_y):
    dmidx, dmidy = deformation(0.6, 0.2)
    disp_x.append(dmidx)
    disp_y.append(dmidy)
    return


disp_x = [0]
disp_y = [0]
times = [0]

if start_with_stokes:
    par.Set(1)
    gf_stokes.components[0].Set(u_inflow, definedon=mesh.Boundaries("inlet"))
    res_stokes.data = stokes.mat * gf_stokes.vec
    gf_stokes.vec.data -= inv_stokes * res_stokes
    velocity.vec.data = gf_stokes.components[0].vec
    pressure.vec.data = gf_stokes.components[1].vec
else:
    gf_solution.vec[:] = 0
gf_solution_old.vec.data = gf_solution.vec

scene_u = Draw(
    velocity,
    mesh.Materials("fluid"),
    "velocity",
    deformation=deformation,
    order=3,
    max=2,
)
scene_p = Draw(pressure, mesh.Materials("fluid"), "pressure", deformation=deformation)
scene_d = Draw(deformation, mesh, "deformation")


gf_history = GridFunction(X, multidim=0)


# Time loop.

# In[ ]:


tw = widgets.Text(value="t = 0")
display(tw)

with TaskManager():
    while t < tend - tau / 2.0:
        t += tau

        # update inflow by extending inflow profile as Stokes solution
        if t < 2 + tau / 2.0 and not start_with_stokes:
            par.Set(Force(t) - Force(t - tau))
            gf_stokes.components[0].Set(
                u_inflow, BND, definedon=mesh.Boundaries("inlet")
            )
            res_stokes.data = stokes.mat * gf_stokes.vec
            gf_stokes.vec.data -= inv_stokes * res_stokes
            velocity.vec.data += gf_stokes.components[0].vec
            pressure.vec.data += gf_stokes.components[1].vec

        solvers.Newton(bfa, gf_solution, maxit=10, maxerr=1e-2, printing=False)

        if i % 10 == 0:
            scene_u.Redraw()
            scene_d.Redraw()
            scene_p.Redraw()

        if i % 20 == 0:
            gf_history.AddMultiDimComponent(gf_solution.vec)

        times.append(t)
        CalcForces(disp_x, disp_y)

        gf_solution_old.vec.data = gf_solution.vec

        i += 1
        tw.value = f"t = {round(t,5)}"


# In[ ]:


Draw(
    gf_history.components[0],
    mesh.Materials("fluid"),
    animate=True,
    min=0,
    max=2,
    autoscale=True,
    deformation=gf_history.components[2],
    order=3,
);


# Draw results over time. They become periodic after some time.

# In[ ]:


import matplotlib.pyplot as plt

plt.plot(times, disp_x)
plt.xlabel("time")
plt.ylabel("disp_x")
plt.grid(True)
plt.show()

plt.plot(times, disp_y)
plt.xlabel("time")
plt.ylabel("disp_y")
plt.grid(True)
plt.show()


# Using a Taylor-Hood discretization for the Navier-Stokes equations has the draw back of not exactly divergence-free velocity fields
# 
# \begin{align*}
# \int_{\Omega}\mathrm{div}(u)\,q\,dx = 0 \quad\forall q\quad\nRightarrow \quad\mathrm{div}(u)\equiv 0.
# \end{align*}
# 
# Using H(div)-conforming Brezzi-Douglas-Marini or Raviart-Thomas finite elements yield to exactly divergence-free solutions. The ALE transformation and interface coupling becomes a bit more involved:
# [<a href="https://doi.org/10.1016/j.compstruc.2020.106402">Neunteufel, Schöberl. Fluid-structure interaction with H(div)-conforming finite elements. <i>Computers \& Structures</i>, (2021).</a>], see [Notebook FSI with H(div)-conforming HDG](fsi_HDiv.ipynb).

# In[ ]:




