from ngsolve import *

class MGPreconditioner(BaseMatrix):
    def __init__(self, fes, level, mat, coarsepre, inv):
        super(MGPreconditioner, self).__init__()
        self.fes = fes
        self.level = level
        self.mat = mat
        self.coarsepre = coarsepre
        if level > 0:
            self.localpre = mat.CreateSmoother(fes.FreeDofs())
        elif inv:
            self.localpre = mat.Inverse(fes.FreeDofs())
        else:
            self.localpre = 0


    def Mult(self, d, w):
        if self.level == 0:
            w.data = self.localpre * d
            return
        w[:] = 0
        self.localpre.Smooth(w, d)
        res = (d - self.mat * w).Evaluate()
        self.fes.Prolongation().Restrict(self.level, res)
        wc = res.CreateVector()
        nc = self.coarsepre.height
        wc[IntRange(0, nc)] = (self.coarsepre * res[0:nc]).Evaluate()
        self.fes.Prolongation().Prolongate(self.level, wc)
        w += wc
        self.localpre.SmoothBack(w, d)

    def Height(self):
        return self.localpre.height

    def Width(self):
        return self.localpre.height

    def CreateColVector(self):
        return self.localpre.CreateColVector()

    def CreateRowVector(self):
        return self.localpre.CreateRowVector()


class ProjectedMG(BaseMatrix):
    def __init__(self, fes, mat, level):
        super(ProjectedMG, self).__init__()
        self.fes = fes
        self.level = level
        self.mat = mat
        if level > 0:
            self.prol = fes.Prolongation().CreateMatrix(level)
            self.rest = self.prol.CreateTranspose()
            coarsemat = self.rest @ mat @ self.prol  # multiply matrices
            self.localpre = mat.CreateSmoother(fes.FreeDofs())

            self.coarsepre = ProjectedMG(fes, coarsemat, level - 1)
        else:
            self.localpre = mat.Inverse(fes.FreeDofs())

    def Mult(self, d, w):
        if self.level == 0:
            w.data = self.localpre * d
            return
        w[:] = 0
        self.localpre.Smooth(w, d)
        res = (d - self.mat * w).Evaluate()
        w += self.prol @ self.coarsepre @ self.rest * res
        self.localpre.SmoothBack(w, d)

    def Height(self):
        return self.localpre.height

    def Width(self):
        return self.localpre.width

    def CreateColVector(self):
        return self.localpre.CreateColVector()

    def CreateRowVector(self):
        return self.localpre.CreateRowVector()

class GS(BaseMatrix):
    def __init__ (self, fes, mat, pre):
        super(GS, self).__init__()
        self.mat = mat
        self.fes = fes
        self.localpre = mat.CreateSmoother(fes.FreeDofs())
        self.coarsepre = pre

    def Mult (self, d, w):
        w[:] = 0
        self.localpre.Smooth(w, d)
        res = (d - self.mat * w).Evaluate()
        w += self.coarsepre * res
        self.localpre.SmoothBack(w, d)

    def Height (self):
        return self.localpre.height
    def Width (self):
        return self.localpre.height