import numpy
from dolfin import *
from petsc4py import PETSc
import time

def darcy(mesh):
    "Mixed H(div) x L^2 formulation of Poisson/Darcy."
    
    V = FiniteElement("RT", mesh.ufl_cell(), 1)
    Q = FiniteElement("DG", mesh.ufl_cell(), 0)
    W = FunctionSpace(mesh, V*Q)
    
    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)

    a = (dot(u, v) + div(u)*q + div(v)*p)*dx
    f = Constant(-1.0)
    L = f*q*dx 

    A = assemble(a)
    b = assemble(L)
    w = Function(W)

    return A, b, w

def solve_darcy(n):

    mesh = UnitCubeMesh(n, n, n)
    print mesh.num_cells()

    PETScOptions.set("darcy_ksp_type", "gmres")
    PETScOptions.set("darcy_pc_type", "fieldsplit")
    PETScOptions.set("darcy_pc_fieldsplit_type", "schur")
    PETScOptions.set("darcy_pc_fieldsplit_schur_fact_type", "full")

    PETScOptions.set("darcy_fieldsplit_0_ksp_type", "cg")
    PETScOptions.set("darcy_fieldsplit_0_pc_type", "ilu")
    PETScOptions.set("darcy_fieldsplit_0_ksp_rtol", 1.e-12)

    PETScOptions.set("darcy_fieldsplit_1_ksp_type", "cg")
    PETScOptions.set("darcy_fieldsplit_1_pc_type", "none")
    PETScOptions.set("darcy_fieldsplit_1_ksp_rtol", 1.e-12)

    A, b, w = darcy(mesh)
    #w.vector()[:] = numpy.random.rand(b.size())

    
    solver = PETScKrylovSolver() # Will be overwritten
    solver.parameters["error_on_nonconvergence"] = True
    solver.parameters["relative_tolerance"] = 1.e-10
    solver.parameters["convergence_norm_type"] = "preconditioned"
    solver.parameters["monitor_convergence"] = True
    solver.parameters["report"] = True

    solver.set_operator(A)

    # Extrct the KSP (Krylov Solver P) from the solver
    ksp = solver.ksp()
    ksp.setOptionsPrefix("darcy_")
    ksp.setFromOptions()

    W = w.function_space()
    u_dofs = W.sub(0).dofmap().dofs()
    p_dofs = W.sub(1).dofmap().dofs()
    u_is = PETSc.IS().createGeneral(u_dofs)
    p_is = PETSc.IS().createGeneral(p_dofs)
    fields = [("0", u_is), ("1", p_is)]
    ksp.pc.setFieldSplitIS(*fields)
    
   # Give this KSP a name (darcy_) 
    #ksp.setUp()
    
    iterations = solver.solve(w.vector(), b)
    print "#iterations = ", iterations

    return w
    
if __name__ == "__main__":

    sizes = [2, 4, 8, 16]
    for n in sizes:
        
        w = solve_darcy(n)

    (u, p) = w.split(deepcopy=True)
    plot(u)
    plot(p)
    interactive()
