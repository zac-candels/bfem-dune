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
import time

absolute_start_time = time.time()


WORKDIR = os.getcwd()
outDirName = os.path.join(WORKDIR, "felb-faster-new")
os.makedirs(outDirName, exist_ok=True)

import ufl

Q = 9
Tfinal = 1500
dt = 0.005
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
    f_star.append( initFn.copy() )
    f_nP1.append( initFn.copy() )

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
                               forceDensity[1]*dt/(2*rho_init)) )
    
    c = xi[vel_idx]
    c_dot_u = ufl.inner(c, vel_0)
    return w[vel_idx] * rho_init * (
        1
        + c_dot_u / c_s**2 
        + c_dot_u**2 / (2*c_s**4) 
        - ufl.inner(vel_0, vel_0) / (2*c_s**2)
        )



def f_equil(vel_idx, velocity, density, velSquared, forceDensity):

    # Compute ci . u for this direction
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
        fEquilInit(idx, forceDensity) ) )
    
#%% Define BCs

u_D = 1

dbcBottom = dune.ufl.DirichletBC(V, u_D, x[1] < 1e-8)
dbcTop = dune.ufl.DirichletBC(V, u_D, abs(L_y - x[1]) <  1e-8 )

bottomFn = V.interpolate(
    lambda x: 1.0 if abs(x[1]) < 1e-8 else 0.0, name="bottomFn")

topFn = V.interpolate(
    lambda x: 1.0 if abs(L_y - x[1]) < 1e-8 else 0.0, name="topFn")

bottomDoFs = np.where(bottomFn.as_numpy != 0)[0]
topDoFs = np.where(topFn.as_numpy != 0)[0]
    

#%% Define linear and bilinear forms

bilinFormsStream = []
linear_forms_stream = []

bilinFormsColl = []
linear_forms_collision = []

opp_idx = {0: 0, 1: 3, 2: 4, 3: 1, 4: 2, 5: 7, 6: 8, 7: 5, 8: 6}

density = getDens(f_n)
velocity_n = getVel(f_n, forceDensity)
velStar_n = getVel(f_star, forceDensity)

velSquared = ufl.inner(velocity_n, velocity_n)

for idx in range(Q):

    bilinFormsStream.append(f_trial * v * ufl.dx)
    bilinFormsColl.append(f_trial*v*ufl.dx)

    double_dot_product_term = -0.5*dt**2 * ufl.inner(xi[idx], ufl.grad(f_star[idx]))\
        * ufl.inner(xi[idx], ufl.grad(v)) * ufl.dx

    dot_product_force_term = 0.5*dt**2 * ufl.inner(xi[idx], ufl.grad(v))\
        * body_Force(velStar_n, idx, forceDensity) * ufl.dx


    lin_form_idx = f_star[idx]*v*ufl.dx\
        - dt*v*ufl.inner(xi[idx], ufl.grad(f_star[idx]))*ufl.dx\
        + dt*v*body_Force(velStar_n, idx, forceDensity)*ufl.dx\
        + double_dot_product_term\
        + dot_product_force_term
        
    f_eq_idx = f_equil(
        idx,
        velocity_n,
        density,
        velSquared,
        forceDensity)
        
    lin_form_coll = (f_n[idx] - dt/(tau) * (f_n[idx] - f_eq_idx) )*v*ufl.dx

    linear_forms_stream.append(lin_form_idx)
    linear_forms_collision.append(lin_form_coll)

# Assemble matrices for first step
sysMatStream = []
sysMatColl = []
rhs_vec_streaming = [0]*Q
rhs_vec_collision = [0]*Q
for idx in range(Q):
    #sysMatStream.append(dune.fem.assemble(bilinFormsStream[idx]))
    if (idx == 0) or (idx == 1) or (idx == 3):
        sysMatStream.append(dune.fem.assemble(bilinFormsStream[idx]))
    elif (idx == 5) or (idx == 2) or (idx == 6):
        sysMatStream.append(dune.fem.assemble([bilinFormsStream[idx], dbcBottom]))
    elif (idx== 4) or (idx == 7) or (idx == 8) :
        sysMatStream.append(dune.fem.assemble([bilinFormsStream[idx], dbcTop]))
        
    sysMatColl.append(dune.fem.assemble(bilinFormsColl[idx]))
 
rhsVecStreaming = []
rhsVecCollision = []
    
sysMatCollNumpy = sysMatColl[0].as_numpy 
collSolver = scipy.sparse.linalg.factorized(sysMatCollNumpy)

streamSolvers = []
f_nP1_arrays = []
f_n_arrays = []
f_star_arrays= []
for idx in range(Q):
    sysMatStreamNumpy = sysMatStream[idx].as_numpy
    streamSolvers.append(scipy.sparse.linalg.factorized(sysMatStreamNumpy))
    f_nP1_arrays.append(f_nP1[idx].as_numpy)
    f_n_arrays.append(f_n[idx].as_numpy)
    f_star_arrays.append(f_star[idx].as_numpy)
    
collisionOps = []
streamingOps = []
for idx in range(Q):
    collForm = linear_forms_collision[idx]
    streamForm = linear_forms_stream[idx]
    collisionOps.append( dune.fem.operator.galerkin(collForm - f_trial*v*ufl.dx) )
    streamingOps.append( dune.fem.operator.galerkin(streamForm - f_trial*v*ufl.dx) )
    
    rhsVecCollision.append(V.zero.copy())
    rhsVecStreaming.append(V.zero.copy())
    
    
    

vel_expr = getVel(f_n, forceDensity)

ux_expr = vel_expr[0]
uy_expr = vel_expr[1]

y = ufl.SpatialCoordinate(V)[1]
        
u_exact_expr = u_max * (
    1.0 - (2.0 * y / L_y - 1.0)**2
)

u_exact = V.interpolate(
    u_exact_expr)

ux = V.interpolate(0, name='ux')

xi_arr = np.array([[0,0],[1,0],[0,1],[-1,0],[0,-1],
                   [1,1],[-1,1],[-1,-1],[1,-1]], dtype=float)
    

#%% Start time-stepping

about_to_enter_loop_time = time.time()
print("time to enter time loop:", about_to_enter_loop_time - absolute_start_time)
t = 0.0
print("about to enter time loop \n\n\n")
for n in range(numSteps):
    t += dt
    

    collisionTimeStart = time.time()
    # Do collision
    for idx in range(Q):
        collisionOps[idx](V.zero, rhsVecCollision[idx])
        
        b = rhsVecCollision[idx].as_numpy 
        
        f_star_arrays[idx][:] = collSolver(b)
    collisionTimeEnd = time.time()
    #print("collision time = ", collisionTimeEnd - collisionTimeStart)
    
        

    for idx in range(Q):
        
        if (idx==2) or (idx==5) or (idx==6):
            streamAssembleTimeStart = time.time()
            
            streamingOps[idx](V.zero, rhsVecStreaming[idx])
            rhsVecStreaming[idx].as_numpy[bottomDoFs]\
                = f_star[opp_idx[idx]].as_numpy[bottomDoFs]
            streamAssembleTimeEnd = time.time()
            #print("time to assemble streaming = ", streamAssembleTimeEnd - streamAssembleTimeStart, "\n\n")
                
            b = rhsVecStreaming[idx].as_numpy
            
            streamSolveTimeStart = time.time()
            f_n_arrays[idx][:] = streamSolvers[idx](b)
            streamSolveTimeEnd = time.time()
            #print("time to solve streaming = ", streamSolveTimeEnd - streamSolveTimeStart)
            
            
            
        elif (idx==4) or (idx == 7) or (idx==8):
            streamingOps[idx](V.zero, rhsVecStreaming[idx])
            
            rhsVecStreaming[idx].as_numpy[topDoFs]\
                    = f_star[opp_idx[idx]].as_numpy[topDoFs]
                    
            b = rhsVecStreaming[idx].as_numpy
            
            f_n_arrays[idx][:] = streamSolvers[idx](b)
            
            
        else:
            streamingOps[idx](V.zero, rhsVecStreaming[idx])
            b = rhsVecStreaming[idx].as_numpy

            f_n_arrays[idx][:] = streamSolvers[idx](b)
        
        # if idx == 2:
        #     print(
        #         "bottom bounceback error f2-f4:",
        #         np.max(np.abs(
        #             f_nP1[2].as_numpy[bottomDoFs]
        #             - f_star[4].as_numpy[bottomDoFs]
        #         ))
        #     )

        # if idx == 5:
        #     print(
        #         "bottom bounceback error f5-f7:",
        #         np.max(np.abs(
        #             f_nP1[5].as_numpy[bottomDoFs]
        #             - f_star[7].as_numpy[bottomDoFs]
        #         ))
        #     )

        # if idx == 6:
        #     print(
        #         "bottom bounceback error f6-f8:",
        #         np.max(np.abs(
        #             f_nP1[6].as_numpy[bottomDoFs]
        #             - f_star[8].as_numpy[bottomDoFs]
        #         ))
        #     )
            
        # if idx == 4:
        #     print(
        #         "top bounceback error f4-f2:",
        #         np.max(np.abs(
        #             f_nP1[4].as_numpy[topDoFs]
        #             - f_star[2].as_numpy[topDoFs]
        #         ))
        #     )
                    
    
    if n % 5000 == 0:
        
        #ux = V.interpolate(ux_expr, name="ux")
        #uy = V.interpolate(uy_expr, name="uy")
        
        nu_lbm = tau / 3.0

        u_max = forceDensity[0] * L_y**2 / (
            8.0 * 1.0 * nu_lbm
        )

        
        # ------------------------------------------------------------
        # Compute error and maximum velocity
        # ------------------------------------------------------------
        
        u_exact_np = u_exact.as_numpy
        
        
        density = np.zeros_like(f_n_arrays[0])

        for j in range(Q):
            density += f_n_arrays[j]
        
        momentum_x = np.zeros_like(density)
        momentum_y = np.zeros_like(density)
        
        for j in range(Q):
            momentum_x += f_n_arrays[j] * float(xi_arr[j][0])
            momentum_y += f_n_arrays[j] * float(xi_arr[j][1])
        
        ux_array = momentum_x / density
        uy_array = momentum_y / density
        
        # Force correction
        ux_array += forceDensity[0] * dt / (2.0 * density)
        uy_array += forceDensity[1] * dt / (2.0 * density)
        
        ux.as_numpy[:] = ux_array
                
        print(
            f"n={n}, t={t:.4f}, "
            f"max(ux)={np.max(ux.as_numpy):.12e}, "
            f"min(ux)={np.min(ux.as_numpy):.12e}",
            flush=True
        )
        
        mesh.writeVTK(
            os.path.join(outDirName, f"ux_modBC_{n:06d}"),
            pointdata=[ux]
        )
        error = np.linalg.norm(u_exact_np - ux.as_numpy)
        max_u = np.max(ux.as_numpy)
                
                
            
        
    
    
    
