import os
os.environ["DUNE_LOG_LEVEL"] = "ERROR"

from mpi4py import MPI
from scipy.sparse import linalg
from math import sqrt
import matplotlib.pyplot as plt
#import pygmsh
#from dune.alugrid import aluConformGrid as GridView
from dune.ufl import Constant, DirichletBC
import dune
import dune.alugrid
import scipy
import numpy as np
import os
import time
from dune.common import comm



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

L_x = 16
L_y = 4
Nx = 40
Ny = 10

#%% Prepare output
WORKDIR = os.getcwd()
outDirName = os.path.join(WORKDIR, f"no_gmsh")
os.makedirs(outDirName, exist_ok=True)


log_file = open(outDirName + "/simulation_log.txt", "w")
log_file.write(f"{'n' :>15}"
               f"{'r_cl':>15}"
               f"{'max ||u||':>15}"
               f"{'theta':>15}\n")
log_file.flush()

xc, yc = L_x/2, R0 - 0.6*R0

rho_init = 1.0
nu = 1/3
c_s = np.sqrt(1/3)
tau = 0.1

domain = dune.grid.cartesianDomain(
    [0.0, 0.0],
    [L_x, L_y],
    [Nx, Ny])

mesh = dune.fem.view.adaptiveLeafGridView(
    dune.alugrid.aluConformGrid(domain, lbMethod=9))

mesh.hierarchicalGrid.loadBalance()

h_min = np.inf

for element in mesh.elements:
    corners = np.asarray(element.geometry.corners)

    h_element = max(
        np.linalg.norm(a - b)
        for i, a in enumerate(corners)
        for b in corners[i + 1:]
    )

    h_min = min(h_min, h_element)

h_min = h_min*(1/2**3)

Tfinal = 1500
dt = 0.05*h_min 
beta_mass_diff = 0.1*dt
numSteps = int(Tfinal/dt)
# Corresponding weights
w = np.array([
    4/9,
    1/9, 1/9, 1/9, 1/9,
    1/36, 1/36, 1/36, 1/36
])

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
    if comm.rank == 0:
        print("idx = ", idx, flush=True)
    f_n.append( V.interpolate(
        fEquilInit(idx),
        name=f"f_n{idx}") )
    

# Function to evaluate FE function at a point 

from dune.common import FieldVector

def evaluate_at(u, gridView, point):
    point = np.asarray(point, dtype=float)
    local_u = u.localFunction()

    for element in gridView.elements:
        geometry = element.geometry
        local_point = geometry.local(point)

        if geometry.checkInside(local_point):
            local_u.bind(element)
            try:
                return float(local_u(local_point))
            finally:
                local_u.unbind()

    raise ValueError(f"Point {point} is outside the grid")



# Function to track contact line
def trackCL(phi_n, dof_coordinates, h):
    """
    Estimate the left contact-line position from nodal values of phi_n.

    Parameters
    ----------
    phi_n : DUNE-FemPy discrete function
        Order parameter.
    dof_coordinates : array-like, shape (N, 2)
        Coordinates corresponding to phi_n.as_numpy, in the same DOF order.
    h : float
        Mesh length scale used for filtering.
    """
    values = np.asarray(phi_n.as_numpy).reshape(-1)
    coords = np.asarray(dof_coordinates)

    # Restrict to the lower part of the domain.
    mask = coords[:, 1] < 4 * h

    # Select approximate interface points.
    mask &= (values > -0.2) & (values < 0.2)

    candidates = coords[mask]
    candidate_values = values[mask]

    if comm.sum(float(len(candidates))) < 2:
        raise ValueError(
            "Not enough interface points to estimate the contact line."
        )

    # Exclude points near the left contact line.
    min_x = comm.min(candidates[:, 0].min() if len(candidates) else np.inf)
    keep_rightCL = candidates[:, 0] > min_x + 7 * h

    max_x = comm.max(candidates[:, 0].max() if len(candidates) else -np.inf)
    keep_leftCL = candidates[:, 0] < max_x - 7*h

    candidates_rightCL = candidates[keep_rightCL]
    candidates_leftCL = candidates[keep_leftCL]

    # Leftmost and rightmost remaining points (global).
    left_x = comm.min(candidates_leftCL[:, 0].min() if len(candidates_leftCL) else np.inf)
    right_x = comm.max(candidates_rightCL[:, 0].max() if len(candidates_rightCL) else -np.inf)

    return left_x, right_x


# Function to compute contact angle 
def computeContactAngle(c_n, h, Cn, mesh):
    angles = []
    n_vec = np.array([0.0, -1.0])

    local_c = c_n.localFunction()

    try:
        for element in mesh.elements:
            geometry = element.geometry

            # Physical element centre and its reference coordinate
            midpoint = np.asarray(geometry.center, dtype=float)
            local_point = geometry.toLocal(midpoint.tolist())

            local_c.bind(element)
            try:
                value = float(np.asarray(local_c(local_point)).reshape(-1)[0])

                # Keep points near the substrate and interface
                if midpoint[1] >= 5*h:
                    continue
                if not (-0.5 < value < 0.5):
                    continue

                # Gradient with respect to physical coordinates
                grad_c = np.asarray(local_c.jacobian(local_point)).reshape(-1)

                grad_norm = np.linalg.norm(grad_c)
                if grad_norm < 1e-14:
                    continue

                cos_theta = np.dot(grad_c, n_vec) / grad_norm
                angles.append(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

            finally:
                local_c.unbind()

    finally:
        pass

    n_angles = comm.sum(float(len(angles)))
    if n_angles == 0:
        raise ValueError(
            "No valid interfacial points found near the substrate."
        )

    # Preserve the original exclusion of the left contact line.
    # For that exclusion, store (midpoint, angle) pairs before averaging.
    return np.degrees(comm.sum(float(np.sum(angles))) / n_angles)

    
#%% Define BCs

if comm.rank == 0:
    print("starting to create BCs\n\n", flush=True)

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
    
if comm.rank == 0:
    print("done making BCs\n\n", flush=True)

def wall_dofs():
    y = V.interpolate(x[1], name="y_dof").as_numpy
    bottom = np.where(np.abs(y) < 1e-8)[0]
    top    = np.where(np.abs(L_y - y) < 1e-8)[0]
    return bottom, top

#%% Define linear and bilinear forms


bilinFormsStream = []
linear_forms_stream = []

bilinFormsColl = []
linear_forms_collision = []


bilin_form_AC = phi_trial * v * ufl.dx
bilin_form_mu = mu_trial * v * ufl.dx


#print("going to create ufl functions for density, vel, velStar \n\n")

density = getDens(f_n)
velocity_n = getVel(f_n, forceDensity)
velStar_n = getVel(f_star, forceDensity)
velSquared = ufl.inner(velocity_n, velocity_n)

#print("about to start making AC forms \n\n")

lin_form_AC =  - dt*v*ufl.dot(velocity_n, ufl.grad(phi_n))*ufl.dx\
    - dt*M_tilde*v*mu_n*ufl.dx - (beta_mass_diff/dt)*mass_diff*ufl.sqrt( ufl.dot(ufl.grad(phi_n), ufl.grad(phi_n)) )*v*ufl.dx\
        - 0.5*dt**2 * ufl.dot(velocity_n, ufl.grad(v)) * ufl.dot(velocity_n, ufl.grad(phi_n)) *ufl.dx

lin_form_mu =  A* phi_n*(phi_n**2 - 1)*v*ufl.dx\
    + kappa*ufl.dot(ufl.grad(phi_n),ufl.grad(v))*ufl.dx\
        + kappa/(np.sqrt(2)*interfaceThickness)*np.cos(theta)*(phi_n**2-1)*v*ufl.ds(3)

#print("finished making AC forms \n\n")
opp_idx = {0: 0, 1: 3, 2: 4, 3: 1, 4: 2, 5: 7, 6: 8, 7: 5, 8: 6}

for idx in range(Q):

    bilinFormsStream.append(f_trial * v * ufl.dx)
    bilinFormsColl.append(f_trial*v*ufl.dx)

    double_dot_product_term = -0.5*dt**2 * ufl.inner(xi[idx], ufl.grad(f_star[idx]))\
        * ufl.inner(xi[idx], ufl.grad(v)) * ufl.dx

    dot_product_force_term = 0.5*dt**2 * ufl.inner(xi[idx], ufl.grad(v))\
        * body_Force(velStar_n, idx, forceDensity) * ufl.dx


    lin_form_idx = - dt*v*ufl.inner(xi[idx], ufl.grad(f_star[idx]))*ufl.dx\
        + dt*v*body_Force(velStar_n, idx, forceDensity)*ufl.dx\
        + double_dot_product_term\
        + dot_product_force_term
        
    f_eq_idx = f_equil(
        idx,
        velocity_n,
        density,
        velSquared)
        
    lin_form_coll = ( - dt/(tau) * (f_n[idx] - f_eq_idx) )*v*ufl.dx

    linear_forms_stream.append(lin_form_idx)
    linear_forms_collision.append(lin_form_coll)
   
if comm.rank == 0:
    print("finished making LB forms\n\n", flush=True)

# Assemble matrices for first step
sysMatStream = []
sysMatColl = []
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

zeroFn = V.zero.copy()
    
    
    
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
 
def build_solvers():
    # 1. Get local DOF coordinates on this rank
    x_local = V.interpolate(x[0], name="x_dof").as_numpy.copy()
    y_local = V.interpolate(x[1], name="y_dof").as_numpy.copy()
    N_local = len(x_local)

    # 2. Gather coordinates from all ranks globally using mpi4py
    x_global_list = MPI.COMM_WORLD.allgather(x_local)
    y_global_list = MPI.COMM_WORLD.allgather(y_local)

    x_full = np.concatenate(x_global_list)
    y_full = np.concatenate(y_global_list)
    N_global = len(x_full)

    # 3. Apply periodicity on the global coordinate set
    x_periodic_global = x_full.copy()
    x_periodic_global[np.abs(x_periodic_global - L_x) < 1e-10] = 0.0

    keys = np.column_stack((np.round(x_periodic_global, 10), np.round(y_full, 10)))
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    nPeriodic = len(unique_keys)

    # 4. Build the GLOBAL projection matrix P_global (shape: N_global x nPeriodic)
    rows_global = np.arange(N_global)
    cols_global = inverse
    data_global = np.ones(N_global)
    P_global = scipy.sparse.coo_matrix((data_global, (rows_global, cols_global)), shape=(N_global, nPeriodic)).tocsr()

    # 5. Determine which global indices belong to this local rank
    local_sizes = MPI.COMM_WORLD.allgather(N_local)
    start_idx = sum(local_sizes[:comm.rank])
    end_idx = start_idx + N_local

    # Extract the local slice of P for this rank
    P_local = P_global[start_idx:end_idx, :]

    # 6. Handle lumped mass diagonal globally and slice it locally
    lumpOp = dune.fem.operator.galerkin(1.0 * v * ufl.dx - f_trial * v * ufl.dx)
    lumpVec = V.zero.copy()
    lumpOp(zeroFn, lumpVec)

    mass_diag_local = lumpVec.as_numpy.copy()
    mass_diag_global = np.concatenate(MPI.COMM_WORLD.allgather(mass_diag_local))

    # Periodic lumped mass diagonal computed globally, then used locally
    mass_diag_periodic = np.asarray(P_global.T @ mass_diag_global).ravel()

    if np.any(mass_diag_periodic <= 0):
        raise ValueError("Periodic lumped mass contains non-positive entries")

    return P_local, mass_diag_local, mass_diag_periodic

P, mass_diag, mass_diag_periodic = build_solvers()

bottomDoFs, topDoFs = wall_dofs()
    

#%% Time-stepping 

if comm.rank == 0:
    print("about to start time-stepping\n\n", flush=True)
    
for n in range(numSteps):
    
    #print("n = ", n)

    phi_form_op(zeroFn, rhs_AC_fn)
    mu_form_op(zeroFn, rhs_Mu_fn)
    
    #print("assembled rhs vecs for phi, mu \n\n")

    
    # Do collision
    for idx in range(Q):
        #print("idx = ", idx)
        collisionOps[idx](zeroFn, rhsVecCollision[idx])
    
        # Full DUNE RHS
        b_full = rhsVecCollision[idx].as_numpy
    
        # Reduce RHS to periodic space
        b_periodic = comm.sum(P.T @ b_full)
    
        # Solve periodic system
        f_star_periodic = b_periodic / mass_diag_periodic
    
        # Expand back to full DUNE space
        f_star_arrays[idx][:] = P @ f_star_periodic + f_n_arrays[idx][:]
        

    
    #print("did collision \n\n")
        
    # Stream all populations
    for idx in range(Q):
        streamingOps[idx](zeroFn, rhsVecStreaming[idx])
    
        b_full = rhsVecStreaming[idx].as_numpy
        b_periodic = comm.sum(P.T @ b_full)
        f_n_periodic = b_periodic / mass_diag_periodic
    
        f_n_arrays[idx][:] = P @ f_n_periodic + f_star_arrays[idx][:]
    
    # Bottom wall: reflect populations into the fluid
    for idx in (2, 5, 6):
        f_n_arrays[idx][bottomDoFs] = (
            f_star_arrays[opp_idx[idx]][bottomDoFs])
    
    # Top wall
    for idx in (4, 7, 8):
        f_n_arrays[idx][topDoFs] = (
            f_star_arrays[opp_idx[idx]][topDoFs])
            
            
          
    #print("about to solve for phi, mu")
    b_phi_full = rhs_AC_fn.as_numpy

    b_phi_periodic = comm.sum(P.T @ b_phi_full)

    phi_periodic = b_phi_periodic / mass_diag_periodic

    phi_nP1.as_numpy[:] = ( P @ phi_periodic) + phi_n.as_numpy[:]
    
    b_mu_full = rhs_Mu_fn.as_numpy

    b_mu_periodic = comm.sum(P.T @ b_mu_full)

    mu_periodic = b_mu_periodic / mass_diag_periodic

    mu_n.as_numpy[:] = (P @ mu_periodic)
    
    phi_n.assign(phi_nP1)
    
    
    mass_n = dune.fem.integrate((phi_n + 1) / 2)
    
    #print("mass_n = ", mass_n)
    mass_diff.assign( mass_n - mass_init )
    
    if n % 10 == 0:
    
        marker = dune.fem.GridMarker(
            1 - phi_n**2,
            refineTolerance=0.6,
            coarsenTolerance=0.2,
            minLevel=0,
            maxLevel=5)
    
        dune.fem.gridAdapt(
            marker,
            f_n
            + f_star
            + rhsVecCollision
            + rhsVecStreaming
            + [phi_n, mu_n, zeroFn, rhs_AC_fn, rhs_Mu_fn])
    
        # IMPORTANT:
        # Never use the old .as_numpy references after gridAdapt.
    
        f_star_arrays = [
            f_star[idx].as_numpy for idx in range(Q)]
    
        f_n_arrays = [
            f_n[idx].as_numpy for idx in range(Q)]
    
        # Recompute boundary DOF indices
        bottomDoFs = np.where(
            bottomFn.as_numpy != 0)[0]
    
        topDoFs = np.where(
            topFn.as_numpy != 0)[0]
    
        # Rebuild periodic matrix/factorizations
        P, mass_diag, mass_diag_periodic = build_solvers()
        
        bottomDoFs, topDoFs = wall_dofs()
        
    
    if n % 200 == 0:
        
        if comm.rank == 0:
            print("n = ", n, "writing to file", flush=True)
        #print("mass_diff type:", type(mass_diff))
        
        #print("mass_n type:", type(mass_n), "\n\n")
        
        if n > 200:
            theta_avg = computeContactAngle(phi_n, h_min, interfaceThickness, mesh)
            
            if comm.rank == 0:
                print("theta = ", theta_avg, flush=True)
            
            xh = V.interpolate(x[0], name="x")
            yh = V.interpolate(x[1], name="y")
            dof_coords = np.column_stack([xh.as_numpy, yh.as_numpy])  
            left_cl, right_cl = trackCL(phi_n, dof_coords, h_min)
            
            if comm.rank == 0:
                print("left_cl = ", left_cl, " right_cl = ", right_cl, flush=True)
        
        vel_expr = getVel(f_n, forceDensity)
        vel3.interpolate(ufl.as_vector([vel_expr[0], vel_expr[1], 0]))
        
        v_array = vel3.as_numpy.reshape(-1, 3)

        vel_mag = np.linalg.norm(v_array, axis=1)
        
        if comm.rank == 0:
            print("Min speed:", vel_mag.min(), flush=True)
            print("Max speed:", vel_mag.max(), flush=True)
            
            print("Max KE:", vel_mag.max()**2, flush=True)
        
        if n > 200:
            if comm.rank == 0:
                log_file.write(f"{n:15d}"
                            f"{right_cl:15.3f}"
                            f"{vel_mag.max():15.8g}"
                            f"{theta_avg:15.2f}\n")
                log_file.flush()

        mesh.writeVTK(os.path.join(outDirName, f"vel_{n:06d}"), pointdata=[vel3])
        
        mesh.writeVTK(os.path.join(outDirName, f"phi_{n:06d}"),
            pointdata=[phi_n])
    
    