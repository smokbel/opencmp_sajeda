from ngsolve import *
from ngsolve.comp import IntegrationRuleSpace

import numpy as np

# viscosity
nu = 0.001

# timestepping parameters
tau = 0.001
tend = 10

# mesh = Mesh("cylinder.vol")
from netgen.geom2d import SplineGeometry
geo = SplineGeometry()
geo.AddRectangle( (0, 0), (2, 0.41), bcs = ("wall", "outlet", "wall", "inlet"))
geo.AddCircle ( (0.2, 0.2), r=0.05, leftdomain=0, rightdomain=1, bc="cyl")
mesh = Mesh( geo.GenerateMesh(maxh=0.08))

mesh.Curve(3)

V = H1(mesh,order=3, dirichlet="wall|cyl|inlet")
Q = H1(mesh,order=2)

X = FESpace([V,V,Q])

ux,uy,p = X.TrialFunction()
vx,vy,q = X.TestFunction()

div_u = grad(ux)[0]+grad(uy)[1]
div_v = grad(vx)[0]+grad(vy)[1]

stokes = nu*grad(ux)*grad(vx)+nu*grad(uy)*grad(vy)+div_u*q+div_v*p - 1e-10*p*q
a = BilinearForm(X)
a += SymbolicBFI(stokes)
a.Assemble()

# nothing here ...
f = LinearForm(X)
f.Assemble()

# gridfunction for the solution
gfu = GridFunction(X)

# parabolic inflow at bc=1:
uin = 1.5*4*y*(0.41-y)/(0.41*0.41)
gfu.components[0].Set(uin, definedon=mesh.Boundaries("inlet"))

# solve Stokes problem for initial conditions:
inv_stokes = a.mat.Inverse(X.FreeDofs())

res = f.vec.CreateVector()
res.data = f.vec - a.mat*gfu.vec
gfu.vec.data += inv_stokes * res


# matrix for implicit Euler
mstar = BilinearForm(X)
mstar += SymbolicBFI(ux*vx+uy*vy + tau*stokes)
mstar.Assemble()
inv = mstar.mat.Inverse(X.FreeDofs(), inverse="sparsecholesky")

# the non-linear term
conv = BilinearForm(X, flags = { "nonassemble" : True })
conv += SymbolicBFI( CoefficientFunction( (ux,uy) ) * (grad(ux)*vx+grad(uy)*vy) )

# for visualization
velocity = CoefficientFunction (gfu.components[0:2])
Draw (Norm(velocity), mesh, "velocity", sd=3)
Draw (gfu.components[2], mesh, "pressure", sd=3)

print("done")

# IntegralRuleSpace step:

fesir_u = IntegrationRuleSpace(mesh=mesh, order=3)
fesir_p = IntegrationRuleSpace(mesh=mesh, order=2)

irs_p = fesir_p.GetIntegrationRules()
irs_u = fesir_u.GetIntegrationRules()

gfuir_ux = GridFunction(fesir_u)
gfuir_p = GridFunction(fesir_p)

gfuir_ux.Interpolate(gfu.components[0])
gfuir_p.Interpolate(gfu.components[2])

for i in range(len(gfuir_p.vec.data)):
    gfuir_p.vec.data[i] = np.random.uniform(0,1)

###########
# L2 PROJECTION BACK TO GFU -> GETTING TNT FUNCTIONS FROM MIXED FES ABOVE
###########
# mass = BilinearForm(p*q*dx).Assemble().mat   # JS: leads to singular matrix
mass = BilinearForm(p*q*dx + ux*vx*dx + uy*vy*dx).Assemble().mat
invmass = mass.Inverse(inverse="sparsecholesky")

SetTestoutFile("test.out")
rhs = LinearForm(gfuir_p*q*dx(intrules=irs_p), printelvec=True)
rhs.Assemble()
print ("last rhs =", rhs.vec[-15:-1])
# gfu.components[2].vec.data = invmass * rhs.vec
gfu.vec.data = invmass * rhs.vec     # JS: mass and inverse have full size -> store to full vector

# See some of the data that the new gfu has:
print("This is the new gfu data when we get the test and trial functions from the mixed FES:")
print("This is the new gfu data when we get the test and trial functions from the mixed FES:")
for i in range(15):
    print(gfu.components[2].vec.data[i])


##########
# L2 PROJECTION BACK TO GFU -> GETTING TNT FUNCTIONS INDIVIDUALLY...
###########
# p,q = V.TnT() # HERE: This p,q works, while the one above (coming from mixed FES) does not
p,q = Q.TnT() #  JS: must be from space Q
mass = BilinearForm(p*q*dx).Assemble().mat
invmass = mass.Inverse(inverse="sparsecholesky")

rhs = LinearForm(gfuir_p*q*dx(intrules=irs_p), printelvec=True)
rhs.Assemble()
print ("last rhs =", rhs.vec[-15:-1])

gfu.components[2].vec.data = invmass * rhs.vec

# See some of the data that the new gfu has:
print("This is the new gfu data when we get the test and trial functions individually:")
for i in range(15):
    print(gfu.components[2].vec.data[i])