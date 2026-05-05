import time
start_time = time.time()

from netgen.geom2d import SplineGeometry
from netgen.meshing import MeshingParameters
from ngsolve import *
from xfem import *
from xfem.lsetcurv import *
from math import pi
ngsglobals.msg_level = 2
# SetNumThreads(24)

# ---------------------------------- PARAMETERS ----------------------------------
nu = 1e-3
U_m = 1.5

dt_inv = 250
t_end = 8.0
t = Parameter(0.0)

h_max = 0.08
k = 2
gamma_N = 40
gamma_GP = 0.01

pReg = 1e-8
inverse = "sparsecholesky"
ric_tol = 1e-10
ric_max_it = 10
ric_print_resid = False

filename_functionals = "CutSchaeferTurek2D3_Functionals_hmax{}k{}gammaN{}gammaGP{}SBDF1dtinv{}.txt".format(
	h_max,k,gamma_N,gamma_GP,dt_inv)

# ------------------------------- BACKGROUND MESH --------------------------------
geo = SplineGeometry()
p1,p2,p3,p4,p5,p6 = [ geo.AppendPoint(x,y) for x,y in [(0,0), (0.7,0), (2.2,0), (2.2,0.41), (0.7,0.41), (0,0.41)] ]
geo.Append (["line", p1, p2], leftdomain=1, rightdomain=0, bc="wall")
geo.Append (["line", p2, p5], leftdomain=1, rightdomain=2)
geo.Append (["line", p5, p6], leftdomain=1, rightdomain=0, bc="wall")
geo.Append (["line", p6, p1], leftdomain=1, rightdomain=0, bc="inlet")
geo.Append (["line", p2, p3], leftdomain=2, rightdomain=0, bc="wall")
geo.Append (["line", p3, p4], leftdomain=2, rightdomain=0, bc="outlet")
geo.Append (["line", p4, p5], leftdomain=2, rightdomain=0, bc="wall")

geo.SetDomainMaxH(1, h_max/6)

with TaskManager():
	ngmesh = geo.GenerateMesh(maxh=h_max)
	mesh = Mesh(ngmesh)

# ----------------------------------- LEVELSET -----------------------------------
levelset = 0.05**2 - (x-0.2)**2 - (y-0.2)**2

lset_meshadap = LevelSetMeshAdaptation(mesh, order=k, discontinuous_qn=True)
deformation = lset_meshadap.CalcDeformation(levelset)
mesh.SetDeformation(deformation)
lsetp1 = lset_meshadap.lset_p1

lset_neg = { "levelset" : lsetp1, "domain_type" : NEG, "subdivlvl" : 0}
lset_if  = { "levelset" : lsetp1, "domain_type" : IF , "subdivlvl" : 0}

# ----------------------------------- FE SPACE -----------------------------------
V = VectorH1(mesh, order=k, dirichlet="inlet|wall")
Q = H1(mesh,order=k-1)
X = FESpace([V,Q], dgjumps=True)

gfu = GridFunction(X)
gfu.vec[:] = 0.0
vel,pre = gfu.components[0],gfu.components[1]
Draw(vel,mesh,"velocity")
Draw(pre,mesh,"pressure")

# ----------------------------- MESH & DOF MARKINGS ------------------------------
ci = CutInfo(mesh,lsetp1)
active_el = ci.GetElementsOfType(HASNEG)
cut_el = ci.GetElementsOfType(IF)

active_dofs = GetDofsOfElements(X,active_el)
active_dofs &= X.FreeDofs()

facets_gp = GetFacetsWithNeighborTypes(mesh, a=active_el, b=cut_el, use_and=True)

# ------------------------------- (BI)LINEAR FORMS -------------------------------
(u,p),(v,q) = X.TnT()

h = specialcf.mesh_size
n_lset = 1.0/Norm(grad(lsetp1)) * grad(lsetp1)

n_steps = int(t_end*dt_inv+0.5)
dt =  t_end/n_steps

#--------------------------------
mass = InnerProduct(u,v)

stokes = nu*InnerProduct(grad(u),grad(v)) - p*div(v) - q*div(u)

nitsche = -nu*InnerProduct(grad(u)*n_lset,v)
nitsche += -nu*InnerProduct(grad(v)*n_lset,u)
nitsche += nu*(gamma_N*k*k/h)*InnerProduct(u,v)
nitsche += p*InnerProduct(v,n_lset)
nitsche += q*InnerProduct(u,n_lset)

ghost_penalty = nu*gamma_GP*(1/(h**2))*(u-u.Other())*(v-v.Other())
ghost_penalty += -(1/nu)*gamma_GP*(p-p.Other())*(q-q.Other())

convect = InnerProduct(grad(vel)*vel,v)

# ------------------------------ MATRICES & VECTORS ------------------------------
with TaskManager():
	mStar = BilinearForm(X,symmetric=True)
	mStar += SymbolicBFI(lset_neg, form = mass + dt*stokes)
	mStar += SymbolicBFI(lset_if, form = dt*nitsche)
	mStar += SymbolicFacetPatchBFI(form = dt*ghost_penalty, skeleton=False,definedonelements=facets_gp)
	mStar.Assemble()

with TaskManager():
	mStarReg = BilinearForm(X,symmetric=True)
	mStarReg += SymbolicBFI(lset_neg, form = mass + dt*stokes - dt*pReg*p*q)
	mStarReg += SymbolicBFI(lset_if, form = dt*nitsche)
	mStarReg += SymbolicFacetPatchBFI(form = dt*ghost_penalty, skeleton=False,definedonelements=facets_gp)
	
	mStarReg.Assemble()
	inv = mStarReg.mat.Inverse(active_dofs, inverse=inverse)

with TaskManager():
	m = BilinearForm(X,symmetric=True)
	m += SymbolicBFI(lset_neg, form=mass)
	m.Assemble()

with TaskManager():
	c = LinearForm(X)
	c += SymbolicLFI(lset_neg, form=convect)

# ---------------------------- FUNCTIONAL EVALUATION -----------------------------
stress = InnerProduct(nu*grad(u)*n_lset,v) - InnerProduct(p*n_lset,v)

with TaskManager():
	a = BilinearForm(X,symmetric=False)
	a += SymbolicBFI(lset_if, form=stress)
	a.Assemble()

drag_x_test, drag_y_test = GridFunction(X), GridFunction(X)
drag_x_test.components[0].Set(CoefficientFunction((1.0,0.0)))
drag_y_test.components[0].Set(CoefficientFunction((0.0,1.0)))


fid = open(filename_functionals, "w")
fid.write("time\tdrag\tlift\tpdiff\n")
fid.close()

def CompFunctionalsAndWritToFile():
	drag = -2.0/(0.1*(2*U_m/3)**2)*a(gfu,drag_x_test)
	lift = -2.0/(0.1*(2*U_m/3)**2)*a(gfu,drag_y_test)
	pdiff = pre(0.15,0.2) - pre(0.25,0.2)

	fid = open(filename_functionals, "a")
	fid.write("{:02.4f}\t{}\t{}\t{}\n".format(t.Get(),drag,lift,pdiff))
	fid.close()

# --------------------------------- INFLOW DATA ----------------------------------
u_inflow = sin(pi*t/8)*CoefficientFunction((4*U_m*y*(0.41-y)/(0.41**2), 0.0))

# -------------------------------- TIME STEPPING ---------------------------------
res = gfu.vec.CreateVector()

with TaskManager():
	for it in range(1,n_steps+1):
		t.Set(it*dt)
		print("t = {}".format(t.Get()), end="\r")

		c.Assemble()
		res.data = m.mat*gfu.vec - dt*c.vec

		vel.Set(u_inflow, definedon=mesh.Boundaries("inlet"))
		res.data += -mStar.mat*gfu.vec

		gfu.vec.data += solvers.PreconditionedRichardson(a=mStar, rhs=res, pre=inv, freedofs=active_dofs,
														  maxit=ric_max_it, tol=ric_tol, dampfactor=1.0,
														  printing=ric_print_resid)

		CompFunctionalsAndWritToFile()
		Redraw(blocking=True)

end_time = time.time() - start_time
print("\n ---------- Total time: {:02.0f}:{:02.0f}:{:02.0f}:{:02.0f} ----------".format(end_time//(24*60*60),end_time%(24*60*60)//(60*60), end_time%3600//60, end_time%60 ) )