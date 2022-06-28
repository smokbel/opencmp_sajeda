from ngsolve import *
import netgen.gui
from MGPreconditioner import *
from netgen.geom2d import SplineGeometry
from ngsolve.comp import LinearForm
from ngsolve.la import EigenValues_Preconditioner


tau = 1e-6
levelsc = 5
penalty = 10*4**levelsc

geo = SplineGeometry()
geo.AddRectangle( (0, 0), (2, 0.41), bcs = ("wall", "outlet", "wall", "inlet"))
mesh = Mesh(geo.GenerateMesh(maxh=0.2))
Draw (mesh)

#refine mesh levelsc times for MGPreconditioner
for l in range(levelsc):
    mesh.Refine()


V = VectorH1(mesh, order=2)
V.SetOrder(TRIG,3)
V.Update()

#add additional H1, order=1 space for MGPreconditioner
Vscal = H1(mesh, order=1)

Q = L2(mesh, order=1)

u,v = V.TnT()
s,t = Vscal.TnT()
p,q = Q.TnT()

# bubble
def Phi1(): return IfPos(((x-0.2)**2+(y-0.2)**2 - 0.05**2), 0, 1)
def Phi2(): return IfPos(((x-0.7)**2+(y-0.1)**2 - 0.02**2), 0, 1)
def Phi3(): return IfPos(((x-0.8)**2+(y-0.3)**2 - 0.02**2), 0, 1)
def SumPhi(): return Phi1()+Phi2()+Phi3()
def Phi(): return IfPos(SumPhi()-0.5, penalty, 0)


A = BilinearForm(InnerProduct(Grad(u),Grad(v))*dx + Phi()*u*v*dx + 1e8*u*v*ds("wall|inlet")).Assemble()

#A bilinear form defined on H1
Ascal = BilinearForm(InnerProduct(Grad(s),Grad(t))*dx + Phi()*s*t*dx + 1e8*s*t*ds("wall|inlet")).Assemble()

B = BilinearForm(div(u)*q*dx).Assemble()

#C bilinear form for pressure-penalty inside the bubble
#C = BilinearForm(IfPos(Phi(),-0.001,0)*p*q*dx).Assemble()

#MP bilinear form to use for the preconditioner for pressure block
# MP = BilinearForm(p*q*dx + IfPos(Phi(),1,0)*p*q*dx).Assemble()
MP = BilinearForm(p*q*dx).Assemble()

g = LinearForm(Q).Assemble()

gfu = GridFunction(V, name="u")
gfp = GridFunction(Q, name="p")
uin = CoefficientFunction((1.5*4*y*(0.41-y)/(0.41*0.41), 0))
#gfu.Set(uin, definedon=mesh.Boundaries("inlet"))


# f = LinearForm(CoefficientFunction((0,x-0.5)) * v * dx + 1e8 * uin*v*ds("inlet")).Assemble()
f = LinearForm(1e8 * uin*v*ds("inlet")).Assemble()

#############################
#       preconditioner:     #
preAscal = ProjectedMG(Vscal, Ascal.mat, Vscal.mesh.levels-1)


#print condition number kappa
lam = list(EigenValues_Preconditioner(Ascal.mat, preAscal))
print ("preAscal: lammin, lammax=", lam[0], lam[-1], "kappa=", lam[-1]/lam[0])


preMP = MP.mat.Inverse()
embx, emby = V.embeddings
emblo = Embedding(embx.width, IntRange(0, Vscal.ndof))
# preAscal = ... multigrid-precond for ascal-bifrom in Vscal
preA = embx@emblo @ preAscal @ emblo.T @ embx.T + emby@emblo @ preAscal @ emblo.T @ emby.T

preAGS = GS(V, A.mat, preA)

lam = list(EigenValues_Preconditioner(A.mat, preAGS))
print("preAGS: lammin, lammax=", lam[0], lam[-1], "kappa=", lam[-1] / lam[0])

pre = BlockMatrix([[preAGS, None], [None, preMP]])



#############################
#       solve system        #
#
# K = BlockMatrix([[A.mat, B.mat.T], [B.mat, C.mat]])
K = BlockMatrix([[A.mat, B.mat.T], [B.mat, None]])
rhs = BlockVector ( [f.vec, g.vec] )
sol = BlockVector( [gfu.vec, gfp.vec] )

solvers.MinRes (mat=K, pre=pre, rhs=rhs, sol=sol, initialize=False, maxsteps=500)
#solvers.GMRes (A=K, pre=C, b=rhs, x=sol, tol=tau)

Draw(gfu, mesh, "v", sd=0)
Draw(div(gfu), mesh, "div_v")
Draw(gfp, mesh, "p")


