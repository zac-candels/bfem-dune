from scipy.sparse import linalg
from math import sqrt
import matplotlib.pyplot as plt
import pygmsh
#from dune.alugrid import aluConformGrid as GridView
from dune.ufl import Constant, DirichletBC
import dune
import scipy
import numpy as np
import os

WORKDIR = os.getcwd()
outDirName = os.path.join(WORKDIR, "felb")
os.makedirs(outDirName, exist_ok=True)

import ufl

Q = 9
Tfinal = 1500
dt = 0.01
numSteps = int(Tfinal/dt)
L_x = 32
L_y = 32
nx = 10
ny = 10
h = min(L_x/nx, L_y/ny)

forceDensity = np.array([2.6041666e-5, 0.0])
rho_init = dune.ufl.Constant(1.0)
nu = 1/3
c_s = np.sqrt(1/3)
tau = 1

u_max = forceDensity[0]*L_y**2/(8*rho_init*nu)


dgf_text = """\
DGF
Interval
0.0 0.0
32.0 32.0
10 10
#
PERIODICFACETRANSFORMATION
1 0, 0 1 + 32 0
#
"""
with open("periodic_box.dgf", "w") as f:
    f.write(dgf_text)
    
with open("periodic_box.dgf", "w") as f:
    f.write(dgf_text)
    
from dune.grid import reader
from dune.alugrid import aluCubeGrid as leafGridView

domain = (reader.dgf, "periodic_box.dgf")
mesh = leafGridView(domain, dimgrid=2)

xi = [
        dune.ufl.Constant((0.0,  0.0)),
        dune.ufl.Constant((1.0,  0.0)),
        dune.ufl.Constant((0.0,  1.0)),
        dune.ufl.Constant((-1.0,  0.0)),
        dune.ufl.Constant((0.0, -1.0)),
        dune.ufl.Constant((1.0,  1.0)),
        dune.ufl.Constant((-1.0,  1.0)),
        dune.ufl.Constant((-1.0, -1.0)),
        dune.ufl.Constant((1.0, -1.0)),
    ]

# Corresponding weights
w = np.array([
    4/9,
    1/9, 1/9, 1/9, 1/9,
    1/36, 1/36, 1/36, 1/36
])

#%% Create function space and finite-element functions for the solution

V = dune.fem.space.lagrange(mesh, order=1)

V_vec = dune.fem.space.lagrange(mesh, order=1, dimRange=2)

v = ufl.TestFunction(V)

f_trial = ufl.TrialFunction(V)
f_n = []
f_star = []
f_nP1 = []

initFn = V.interpolate(0, "initFn")
for idx in range(Q):
    #f_n.append( V.interpolate(0, "f_n{}".format(idx) ) )
    f_star.append( initFn.copy(name=f"f_star{idx}") )
    f_nP1.append( initFn.copy(name=f"f_nP1{idx}") )

x = ufl.SpatialCoordinate(V)

t = dune.ufl.Constant(0.0)


#%% Define functions to compute density, velocity, body force
# and equilibrium distributions

def getDens(f_n):
    return sum(f_n)

def getVel(f_n, forceDensity):
    density = sum(f_n)
    
    momentum = f_n[0] * xi[0]
    for idx in range(1, Q):
        momentum += f_n[idx] * xi[idx]
        
    velTerm1 = momentum/density 
    
    F = dune.ufl.Constant((forceDensity[0], forceDensity[1]))
    velTerm2 = F * dt / (2 * density)
        
    return velTerm1 + velTerm2 



def fEquilInit(vel_idx, forceDensity):
    
    vel_0 = -dune.ufl.Constant( (forceDensity[0]*dt/(2*rho_init),
                               forceDensity[1]*dt/2*rho_init) )
    
    c = xi[vel_idx]
    c_dot_u = ufl.inner(c, vel_0)
    return w[vel_idx] * rho_init * (
        1
        + c_dot_u / c_s**2 
        + c_dot_u**2 / (2*c_s**4) 
        - ufl.inner(vel_0, vel_0) / (2*c_s**2)
        )



def f_equil(f_n, vel_idx, forceDensity, wallMask=None):
    density = sum(f_n)

    # Compute velocity at each DoF
    velocity = getVel(f_n, forceDensity)

    if wallMask is not None:
        velocity = velocity * wallMask   # forces velocity -> 0 on wall DoFs

    velSquared = ufl.inner(velocity, velocity)
    c_dot_u = ufl.inner(velocity, xi[vel_idx])
    feq = w[vel_idx] * density * (1 + 3*c_dot_u + 4.5*c_dot_u**2 - 1.5*velSquared)
    return feq



def body_Force(vel, vel_idx, Force_density):
    prefactor = w[vel_idx]
    inverse_cs2 = 1 / c_s**2
    inverse_cs4 = 1 / c_s**4

    xi_dot_prod_F = xi[vel_idx][0]*Force_density[0]\
        + xi[vel_idx][1]*Force_density[1]

    u_dot_prod_F = vel[0]*Force_density[0] + vel[1]*Force_density[1]

    xi_dot_u = xi[vel_idx][0]*vel[0] + xi[vel_idx][1]*vel[1]

    Force = prefactor*(inverse_cs2*(xi_dot_prod_F - u_dot_prod_F)
                       + inverse_cs4*xi_dot_u*xi_dot_prod_F)

    return Force


for idx in range(Q):
    print("idx = ", idx)
    f_n.append( V.interpolate(
        fEquilInit(idx, forceDensity),
        name=f"f_n{idx}") )
    
#%% Define BCs

u_D = 1

dbcBottom = dune.ufl.DirichletBC(V, u_D, x[1] < 1e-8)
dbcTop = dune.ufl.DirichletBC(V, u_D, abs(L_y - x[1]) <  1e-8 )

bottomFn = V.interpolate(
    lambda x: 1.0 if abs(x[1]) < 1e-8 else 0.0,
    name="bottomFn"
)

topFn = V.interpolate(
    lambda x: 1.0 if abs(L_y - x[1]) < 1e-8 else 0.0,
    name="topFn"
)

bottomDoFs = np.where(bottomFn.as_numpy != 0)[0]
topDoFs = np.where(topFn.as_numpy != 0)[0]

wallMask = V.interpolate(
    1.0 - bottomFn - topFn,
    name="wallMask"
)
    

#%% Define linear and bilinear forms

bilinFormsStream = []
linear_forms_stream = []

bilinFormsColl = []
linear_forms_collision = []

opp_idx = {0: 0, 1: 3, 2: 4, 3: 1, 4: 2, 5: 7, 6: 8, 7: 5, 8: 6}

for idx in range(Q):

    bilinFormsStream.append(f_trial * v * ufl.dx)
    bilinFormsColl.append(f_trial*v*ufl.dx)

    double_dot_product_term = -0.5*dt**2 * ufl.inner(xi[idx], ufl.grad(f_star[idx]))\
        * ufl.inner(xi[idx], ufl.grad(v)) * ufl.dx

    dot_product_force_term = 0.5*dt**2 * ufl.inner(xi[idx], ufl.grad(v))\
        * body_Force(getVel(f_star, forceDensity), idx, forceDensity) * ufl.dx


    lin_form_idx = f_star[idx]*v*ufl.dx\
        - dt*v*ufl.inner(xi[idx], ufl.grad(f_star[idx]))*ufl.dx\
        + dt*v*body_Force(getVel(f_star, forceDensity), idx, forceDensity)*ufl.dx\
        + double_dot_product_term\
        + dot_product_force_term
        
    lin_form_coll = (f_n[idx] - dt/(tau) * (f_n[idx] - f_equil(f_n, idx, forceDensity, wallMask))) * v * ufl.dx

    linear_forms_stream.append(lin_form_idx)
    linear_forms_collision.append(lin_form_coll)

# Assemble matrices for first step
sysMatStream = []
sysMatColl = []
rhs_vec_streaming = [0]*Q
rhs_vec_collision = [0]*Q
for idx in range(Q):
    sysMatStream.append(dune.fem.assemble(bilinFormsStream[idx]))
    #if (idx == 0) or (idx == 1) or (idx == 3):
    #    sysMatStream.append(dune.fem.assemble(bilinFormsStream[idx]))
    #elif (idx == 5) or (idx == 2) or (idx == 6):
    #    sysMatStream.append(dune.fem.assemble([bilinFormsStream[idx], dbcBottom]))
    #elif (idx== 4) or (idx == 7) or (idx == 8) :
    #    sysMatStream.append(dune.fem.assemble([bilinFormsStream[idx], dbcTop]))
        
    sysMatColl.append(dune.fem.assemble(bilinFormsColl[idx]))
 
rhsVecStreaming = [0]*Q
rhsVecCollision = [0]*Q
    
sysMatCollNumpy = sysMatColl[0].as_numpy 
collSolver = scipy.sparse.linalg.factorized(sysMatCollNumpy)

streamSolvers = []
for idx in range(Q):
    sysMatStreamNumpy = sysMatStream[idx].as_numpy
    streamSolvers.append(scipy.sparse.linalg.factorized(sysMatStreamNumpy))

vel_expr = getVel(f_n, forceDensity)

ux_expr = vel_expr[0]
uy_expr = vel_expr[1]

ux = V.interpolate(ux_expr, name="ux")    
vtk = mesh.sequencedVTK(
    "felb",
    pointdata=[ux])

y = ufl.SpatialCoordinate(V)[1]
        
u_exact_expr = u_max * (
    1.0 - (2.0 * y / L_y - 1.0)**2
)

u_exact = V.interpolate(
    u_exact_expr,
    name="u_exact"
)

xi_arr = np.array([[0,0],[1,0],[0,1],[-1,0],[0,-1],
                    [1,1],[-1,1],[-1,-1],[1,-1]], dtype=float)

wallDoFs = np.concatenate([bottomDoFs, topDoFs])

#%% Start time-stepping

t = 0.0
print("about to enter time loop \n\n\n")
for n in range(numSteps):
    t += dt


    f_vals = np.array([f_n[idx].as_numpy for idx in range(Q)])   # shape (Q, n_dofs)

    rho = f_vals.sum(axis=0)
    ux = (xi_arr[:, 0, None] * f_vals).sum(axis=0) / rho + forceDensity[0]*dt/(2*rho)
    uy = (xi_arr[:, 1, None] * f_vals).sum(axis=0) / rho + forceDensity[1]*dt/(2*rho)

    # Enforce zero velocity at wall DoFs before it's used in feq
    ux[wallDoFs] = 0.0
    uy[wallDoFs] = 0.0

    cu = xi_arr[:, 0, None]*ux + xi_arr[:, 1, None]*uy   # shape (Q, n_dofs)
    u2 = ux**2 + uy**2

    feq = w[:, None] * rho * (1 + 3*cu + 4.5*cu**2 - 1.5*u2)

    f_star_np = f_vals - (dt/tau) * (f_vals - feq)

    for idx in range(Q):
        f_star[idx].as_numpy[:] = f_star_np[idx, :]

        
    for idx in range(Q):
        rhsVecStreaming[idx] = (dune.fem.assemble(linear_forms_stream[idx]))
        #rhsVecStreaming[idx].as_numpy[bottomDoFs]\
        #    = f_star[opp_idx[idx]].as_numpy[bottomDoFs]
            
        #rhsVecStreaming[idx].as_numpy[topDoFs]\
        #        = f_star[opp_idx[idx]].as_numpy[topDoFs]
                
        b = rhsVecStreaming[idx].as_numpy
        
        sysMatStreamNumpy = sysMatStream[idx].as_numpy
        f_nP1[idx].as_numpy[:] = streamSolvers[idx](b)

    # Update previous solutions

    for idx in range(Q):
        f_n[idx].assign(f_nP1[idx])
        
    
    
    if n % 100 == 0:
        
        
        print("\n\n n = ", n, "writing to file \n\n")
        
        vel_expr = getVel(f_n, forceDensity)
        
        ux_expr = vel_expr[0]
        uy_expr = vel_expr[1]
        
        ux_fn = V.interpolate(ux_expr, name="ux")
        #uy = V.interpolate(uy_expr, name="uy")
        vtk()
        
        nu_lbm = tau / 3.0

        u_max = forceDensity[0] * L_y**2 / (
            8.0 * 1.0 * nu_lbm
        )

        
        # ------------------------------------------------------------
        # Compute error and maximum velocity
        # ------------------------------------------------------------
        
        u_exact_np = u_exact.as_numpy
        
        
        vel_expr = getVel(f_n, forceDensity)
        #ux.interpolate(vel_expr[0])
        
        ux_np = ux_fn.as_numpy
        
        print(
            f"n={n}, t={t:.4f}, "
            f"max(ux)={np.max(ux_np):.12e}, "
            f"min(ux)={np.min(ux_np):.12e}",
            flush=True
        )
        
        mesh.writeVTK(
            os.path.join(outDirName, f"ux_noBC_{n:06d}"),
            pointdata=[ux_fn]
        )
        error = np.linalg.norm(u_exact_np - ux_np)
        max_u = np.max(ux_np)
                
                
            
        
    
    
    

