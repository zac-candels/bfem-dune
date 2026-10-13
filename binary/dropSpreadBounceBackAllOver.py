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

WORKDIR = os.getcwd()
outDirName = os.path.join(WORKDIR, "drop-spread-bounceback-all-over")
os.makedirs(outDirName, exist_ok=True)

import ufl

R0 = 2
initDropDiam = 2*R0
A = 0.5
kappa = 0.02
interfaceThickness = np.sqrt(kappa/A)
M_tilde = 10
theta_deg = 30
theta = theta_deg * np.pi / 180

Q = 9
Tfinal = 1500
dt = 0.001
beta_mass_diff = 0.1*dt
numSteps = int(Tfinal/dt)
L_x = 16
L_y = 4


xc, yc = L_x/2, R0 - 0.6*R0

rho_init = 1.0
nu = 1/3
c_s = np.sqrt(1/3)
tau = 0.1

dgf_text = """\
DGF
Interval
0.0 0.0
16.0 4.0
60 20
#
BOUNDARYDOMAIN
1 0.0 0.0 0.0 4.0
2 16.0 0.0 16.0 4.0
3 0.0 0.0 16.0 0.0
4 0.0 4.0 16.0 4.0
default 1
#
"""

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

V_vec3 = dune.fem.space.lagrange(mesh, order=1, dimRange=3)
vel3 = V_vec3.interpolate(dune.ufl.Constant((0, 0, 0)), name="vel")

v = ufl.TestFunction(V)

phi_trial = ufl.TrialFunction(V)
mu_trial = ufl.TrialFunction(V)
f_trial = ufl.TrialFunction(V)
f_n = []
f_star = []
f_nP1 = []

initFn = V.interpolate(0, "initFn")
for idx in range(Q):
    #f_n.append( V.interpolate(0, "f_n{}".format(idx) ) )
    f_star.append( initFn.copy(name=f"f_star{idx}") )
    f_nP1.append( initFn.copy(name=f"f_nP1{idx}") )
    
phi_nP1 = initFn.copy(name="phi_nP1")
mu_n = initFn.copy(name="mu_n")

x = ufl.SpatialCoordinate(V)

phiInitExpr = -ufl.tanh( (ufl.sqrt(pow(x[0]-xc,2) + pow(x[1]-yc,2)) - R0)\
                        / (ufl.sqrt(2)*interfaceThickness) )

phi_n = V.interpolate(phiInitExpr, name="phi_n")
mass_init = dune.fem.integrate((phi_n + 1) / 2)
mass_diff = dune.ufl.Constant(1e-9)

forceDensity = -phi_n * ufl.grad(mu_n)

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
    velTerm2 = forceDensity * dt / (2 * density)
    return velTerm1 + velTerm2



def fEquilInit(vel_idx):
    
    forceDensity = dune.ufl.Constant((0.0, 0.0))
    
    vel_0 = -dune.ufl.Constant( (0.0, 0.0) )
    
    c = xi[vel_idx]
    c_dot_u = ufl.inner(c, vel_0)
    return w[vel_idx] * rho_init * (
        1
        + c_dot_u / c_s**2 
        + c_dot_u**2 / (2*c_s**4) 
        - ufl.inner(vel_0, vel_0) / (2*c_s**2)
        )




def f_equil(vel_idx, velocity, density, velSquared):

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
        fEquilInit(idx),
        name=f"f_n{idx}") )
    
#%% Define BCs

print("starting to create BCs\n\n")

u_D = 1

dbcBottom = dune.ufl.DirichletBC(V, u_D, x[1] < 1e-8)
dbcTop = dune.ufl.DirichletBC(V, u_D, abs(L_y - x[1]) <  1e-8 )

dbcLeft = dune.ufl.DirichletBC(V, u_D, x[0] < 1e-8)
dbcRight = dune.ufl.DirichletBC(V, u_D, abs(L_x - x[0]) < 1e-8)


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

leftFn = V.interpolate(
    lambda x: 1.0 if abs(x[0]) < 1e-8 else 0.0,
    name="leftFn"
)

rightFn = V.interpolate(
    lambda x: 1.0 if abs(L_x - x[0]) < 1e-8 else 0.0,
    name="rightFn"
)

leftDoFs = np.where(leftFn.as_numpy != 0)[0]
rightDoFs = np.where(rightFn.as_numpy != 0)[0]

    
print("done making BCs\n\n")

#%% Define linear and bilinear forms


bilinFormsStream = []
linear_forms_stream = []

bilinFormsColl = []
linear_forms_collision = []


bilin_form_AC = phi_trial * v * ufl.dx
bilin_form_mu = mu_trial * v * ufl.dx

print("going to create ufl functions for density, vel, velStar \n\n")

density = getDens(f_n)
velocity_n = getVel(f_n, forceDensity)
velStar_n = getVel(f_star, forceDensity)
velSquared = ufl.inner(velocity_n, velocity_n)

print("about to start making AC forms \n\n")

lin_form_AC = phi_n * v * ufl.dx - dt*v*ufl.dot(velocity_n, ufl.grad(phi_n))*ufl.dx\
    - dt*M_tilde*v*mu_n*ufl.dx - (beta_mass_diff/dt)*mass_diff*ufl.sqrt( ufl.dot(ufl.grad(phi_n), ufl.grad(phi_n)) )*v*ufl.dx\
        - 0.5*dt**2 * ufl.dot(velocity_n, ufl.grad(v)) * ufl.dot(velocity_n, ufl.grad(phi_n)) *ufl.dx

lin_form_mu =  A* phi_n*(phi_n**2 - 1)*v*ufl.dx\
    + kappa*ufl.dot(ufl.grad(phi_n),ufl.grad(v))*ufl.dx\
        + kappa/(np.sqrt(2)*interfaceThickness)*np.cos(theta)*(phi_n**2-1)*v*ufl.ds(3)

print("finished making AC forms \n\n")
opp_idx = {0: 0, 1: 3, 2: 4, 3: 1, 4: 2, 5: 7, 6: 8, 7: 5, 8: 6}

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
        velSquared)
        
    lin_form_coll = (f_n[idx] - dt/(tau) * (f_n[idx] - f_eq_idx) )*v*ufl.dx

    linear_forms_stream.append(lin_form_idx)
    linear_forms_collision.append(lin_form_coll)
    
print("finished making LB forms\n\n")

# Assemble matrices for first step
sysMatStream = []
sysMatColl = []
for idx in range(Q):

    # No wall BC for rest population
    if idx == 0:
        sysMatStream.append(
            dune.fem.assemble(bilinFormsStream[idx])
        )

    # Populations pointing toward bottom wall
    elif idx in (2, 5, 6):
        sysMatStream.append(
            dune.fem.assemble([bilinFormsStream[idx], dbcBottom])
        )

    # Populations pointing toward top wall
    elif idx in (4, 7, 8):
        sysMatStream.append(
            dune.fem.assemble([bilinFormsStream[idx], dbcTop])
        )

    # Population pointing toward right wall
    elif idx in (1, 5, 8):
        sysMatStream.append(
            dune.fem.assemble([bilinFormsStream[idx], dbcRight])
        )

    # Population pointing toward left wall
    elif idx in (3, 6, 7):
        sysMatStream.append(
            dune.fem.assemble([bilinFormsStream[idx], dbcLeft])
        )

sysMatColl = dune.fem.assemble(bilinFormsColl[0])

rhsVecStreaming = []
rhsVecCollision = []

collisionOps = []
streamingOps = []
for idx in range(Q):
    collForm = linear_forms_collision[idx]
    streamForm = linear_forms_stream[idx]
    collisionOps.append( dune.fem.operator.galerkin(collForm - f_trial*v*ufl.dx) )
    streamingOps.append( dune.fem.operator.galerkin(streamForm - f_trial*v*ufl.dx) )
    
    rhsVecCollision.append(V.zero.copy())
    rhsVecStreaming.append(V.zero.copy())

phi_form_op = dune.fem.operator.galerkin(lin_form_AC - f_trial*v*ufl.dx)
mu_form_op = dune.fem.operator.galerkin(lin_form_mu - f_trial*v*ufl.dx)
phiMassFormOp = dune.fem.operator.galerkin((phi_n+1)/2*v*ufl.dx - f_trial*v*ufl.dx)
rhs_AC_fn = V.zero.copy()
rhs_Mu_fn = V.zero.copy()
    
    
    
sysMatCollNumpy = sysMatColl.as_numpy 
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
    

#rint("about to start time-stepping\n\n")
for n in range(numSteps):

    phi_form_op(V.zero, rhs_AC_fn)
    mu_form_op(V.zero, rhs_Mu_fn)
    
    #print("assembled rhs vecs for phi, mu \n\n")

    
    # Do collision
    for idx in range(Q):
        collisionOps[idx](V.zero, rhsVecCollision[idx])
        
        b = rhsVecCollision[idx].as_numpy 
        
        f_star_arrays[idx][:] = collSolver(b)
    
    #print("did collision \n\n")
        
    #print("about to start streaming")
    
    bottom_idx = (2, 5, 6)
    top_idx    = (4, 7, 8)
    left_idx   = (3, 6, 7)
    right_idx  = (1, 5, 8)
    
    for idx in range(Q):
    
        streamingOps[idx](V.zero, rhsVecStreaming[idx])
    
        b = rhsVecStreaming[idx].as_numpy
    
        # Bottom wall: bounce back populations
        if idx in bottom_idx:
            b[bottomDoFs] = f_star[opp_idx[idx]].as_numpy[bottomDoFs]
    
        # Top wall: bounce back populations
        if idx in top_idx:
            b[topDoFs] = f_star[opp_idx[idx]].as_numpy[topDoFs]
    
        # Left wall: bounce back populations
        if idx in left_idx:
            b[leftDoFs] = f_star[opp_idx[idx]].as_numpy[leftDoFs]
    
        # Right wall: bounce back populations
        if idx in right_idx:
            b[rightDoFs] = f_star[opp_idx[idx]].as_numpy[rightDoFs]
    
        f_n_arrays[idx][:] = streamSolvers[idx](b)
              
          
    #print("about to solve for phi, mu")
    phi_nP1.as_numpy[:] = collSolver(rhs_AC_fn.as_numpy)
    mu_n.as_numpy[:] = collSolver(rhs_Mu_fn.as_numpy)
    
    phi_n.assign(phi_nP1)
    
    
    mass_n = dune.fem.integrate((phi_n + 1) / 2)
    mass_diff.assign( mass_n - mass_init )
    print("mass_n = ", mass_n)
    
    phi_a = phi_n.as_numpy

    mass_error = mass_n - mass_init

    print(
        "phi finite:",
        np.isfinite(phi_a).all(),
        "min:", np.nanmin(phi_a),
        "max:", np.nanmax(phi_a),
        "nan:", np.isnan(phi_a).sum(),
        "inf:", np.isinf(phi_a).sum()
    )
        
    mass_diff.assign(mass_error)
    
    print("mass_error =", repr(mass_error))
    print("type       =", type(mass_error))
    print("isNumber   =", dune.ufl.isNumber(mass_error))
    print("float      =", float(mass_error))
    print("isNumber(float) =", dune.ufl.isNumber(float(mass_error)))
    
    mass_diff.assign(mass_error)

    
    if n % 100 == 0:
        
        print("n = ", n, "writing to file \n\n")
        
        vel_expr = getVel(f_n, forceDensity)
        vel3.interpolate(ufl.as_vector([vel_expr[0], vel_expr[1], 0]))

        mesh.writeVTK(os.path.join(outDirName, f"vel_{n:06d}"), pointdata=[vel3])
        
        mesh.writeVTK(os.path.join(outDirName, f"phi_{n:06d}"),
            pointdata=[phi_n])
    
    
