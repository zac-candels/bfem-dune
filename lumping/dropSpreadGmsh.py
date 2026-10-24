import os
os.environ["DUNE_LOG_LEVEL"] = "ERROR"

from scipy.sparse import linalg
from math import sqrt
import matplotlib.pyplot as plt
import gmsh
#import pygmsh
#from dune.alugrid import aluConformGrid as GridView
from dune.ufl import Constant, DirichletBC
import dune
import dune.alugrid
import scipy
import numpy as np
import os
import time



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
outDirName = os.path.join(WORKDIR, f"gmsh_with_refinement")
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

def create_sinusoidal_mesh(
    L_x,
    L_y,
    surface_amplitude=0.0,
    surface_freq=None,
    n_surface=200,
    lc=0.1,
    filename="sinusoidal_wall.msh"):
    """
    Create a 2D Gmsh mesh with a sinusoidal bottom boundary.

    Parameters
    ----------
    L_x : float
        Domain length.
    L_y : float
        Domain height.
    surface_amplitude : float
        Amplitude of the sinusoidal bottom boundary.
    surface_freq : float or None
        Frequency of the sinusoidal boundary.
    n_surface : int
        Number of segments along the bottom boundary.
    lc : float
        Characteristic mesh element size.
    filename : str
        Output mesh filename.

    Returns
    -------
    str
        Path to the generated mesh file.
    """

    if surface_freq is None:
        surface_freq = L_x / (8.0 * np.pi)

    xs = np.linspace(0.0, L_x, n_surface + 1)

    ys = surface_amplitude * np.cos(surface_freq * (xs - L_x / 2.0))

    gmsh.initialize()

    try:
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)

        gmsh.model.add("sinusoidal_wall")

        geo = gmsh.model.geo

        # Bottom boundary points
        bottom_points = [
            geo.addPoint(float(x), float(y), 0.0, lc)
            for x, y in zip(xs, ys)
        ]

        # Top corners
        top_right = geo.addPoint(L_x, L_y, 0.0, lc)
        top_left = geo.addPoint(0.0, L_y, 0.0, lc)

        # Bottom boundary segments
        wall_lines = [geo.addLine(bottom_points[i],
                                  bottom_points[i + 1]) for i in range(n_surface)]

        # Right, top, and left boundaries
        right_line = geo.addLine(bottom_points[-1], top_right)

        top_line = geo.addLine(top_right, top_left)

        left_line = geo.addLine(top_left, bottom_points[0])

        # Enclosed fluid domain
        loop = geo.addCurveLoop(wall_lines + [right_line, top_line, left_line])

        surface = geo.addPlaneSurface([loop])

        geo.synchronize()

        # Physical IDs used for boundary conditions
        gmsh.model.addPhysicalGroup(1, wall_lines, 3)
        
        gmsh.model.setPhysicalName(1, 3, "wall")

        gmsh.model.addPhysicalGroup(1, [right_line, left_line], 4)
        
        gmsh.model.setPhysicalName(1, 4, "sides")

        gmsh.model.addPhysicalGroup(1, [top_line], 5)
        
        gmsh.model.setPhysicalName(1, 5, "top")

        gmsh.model.addPhysicalGroup(2, [surface], 1)
        
        gmsh.model.setPhysicalName(2, 1, "fluid")

        # Generate and save the 2D mesh
        gmsh.model.mesh.generate(2)
        gmsh.write(filename)

    finally:
        gmsh.finalize()

    return filename


#%% Generate the Gmsh mesh

print("\nSTEP 1: Starting mesh generation", flush=True)

mesh_file = create_sinusoidal_mesh(
    L_x=L_x,
    L_y=L_y,
    surface_amplitude=0.0,
    surface_freq=L_x / (8.0 * np.pi),
    n_surface=40,
    lc=0.3)

print("\nSTEP 2: Mesh generation completed", flush=True)


# Import the Gmsh mesh and wrap it in an ALUGrid leaf view.
from dune.grid import reader
from dune.alugrid import aluConformGrid
from dune.fem.view import adaptiveLeafGridView

grid_description = (reader.gmsh, mesh_file)

grid = aluConformGrid(grid_description, dimgrid=2)
mesh = adaptiveLeafGridView(grid)


h_min = np.inf

for element in mesh.elements:
    corners = np.asarray(element.geometry.corners)

    h_element = max(
        np.linalg.norm(a - b)
        for i, a in enumerate(corners)
        for b in corners[i + 1:])

    h_min = min(h_min, h_element)

h_min = h_min*(1/2**3)

Tfinal = 1500
dt = 0.003
dt_c = dune.ufl.Constant(dt, name="dt")
beta_mass_diff = 0.1*dt_c
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

pts = [(8.0, 0.8), (8.0, 2.8), (8.0, 3.5), (3.0, 2.0), (5.5, 0.1)]  # drop centre, interface top, bulk, bulk, near wall

def getDens(f_n):
    return sum(f_n)

def getVel(f_n, forceDensity):
    density = sum(f_n)
    momentum = f_n[0] * xi[0]
    for idx in range(1, Q):
        momentum += f_n[idx] * xi[idx]
    velTerm1 = momentum/density 
    velTerm2 = forceDensity * dt_c / (2 * density)
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
    

# Function to evaluate FE function at a point 

from dune.common import FieldVector

def evaluate_at(u, gridView, point, eps=1e-10):
    point = np.asarray(point, dtype=float)
    local_u = u.localFunction()
    for element in gridView.elements:
        lp = np.asarray(element.geometry.toLocal(point.tolist()), dtype=float)
        # reference triangle: xi >= 0, eta >= 0, xi + eta <= 1
        if lp[0] >= -eps and lp[1] >= -eps and lp[0] + lp[1] <= 1 + eps:
            local_u.bind(element)
            try:
                return float(np.asarray(local_u(lp.tolist())).reshape(-1)[0])
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

    if len(candidates) < 2:
        raise ValueError(
            "Not enough interface points to estimate the contact line."
        )

    # Exclude points near the left contact line.
    min_x = candidates[:, 0].min()
    keep_rightCL = candidates[:, 0] > min_x + 7 * h
    
    max_x = candidates[:, 0].max()
    keep_leftCL = candidates[:, 0] < max_x - 7*h

    candidates_rightCL = candidates[keep_rightCL]
    candidate_values_rightCL = candidate_values[keep_rightCL]
    
    candidates_leftCL = candidates[keep_leftCL]
    candidate_values_leftCL = candidate_values[keep_leftCL]



    # Leftmost and rightmost remaining points.
    left_idx = np.argmin(candidates_leftCL[:, 0])
    right_idx = np.argmax(candidates_rightCL[:, 0])

    left_x = candidates_leftCL[left_idx, 0]
    left_phi = candidate_values_leftCL[left_idx]

    right_x = candidates_rightCL[right_idx, 0]
    right_phi = candidate_values_rightCL[right_idx]

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
                if midpoint[1] >= 2*h:
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

    if not angles:
        raise ValueError(
            "No valid interfacial points found near the substrate."
        )

    # Preserve the original exclusion of the left contact line.
    # For that exclusion, store (midpoint, angle) pairs before averaging.
    return np.degrees(np.mean(angles))

    
#%% Define BCs

print("starting to create BCs\n\n")

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
    
print("done making BCs\n\n")

def wall_dofs():
    y = V.interpolate(x[1], name="y_dof").as_numpy
    bottom = np.where(np.abs(y) < 1e-8)[0]
    top    = np.where(np.abs(L_y - y) < 1e-8)[0]
    return bottom, top

#%% Define linear and bilinear forms

substrate = ufl.conditional(x[1] < 1e-8, 1.0, 0.0)

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

lin_form_AC =  - dt_c*v*ufl.dot(velocity_n, ufl.grad(phi_n))*ufl.dx\
    - dt_c*M_tilde*v*mu_n*ufl.dx - (beta_mass_diff/dt_c)*mass_diff*ufl.sqrt( ufl.dot(ufl.grad(phi_n), ufl.grad(phi_n)) )*v*ufl.dx\
        - 0.5*dt_c**2 * ufl.dot(velocity_n, ufl.grad(v)) * ufl.dot(velocity_n, ufl.grad(phi_n)) *ufl.dx

lin_form_mu = A*phi_n*(phi_n**2 - 1)*v*ufl.dx \
    + kappa*ufl.dot(ufl.grad(phi_n), ufl.grad(v))*ufl.dx \
    + substrate*kappa/(np.sqrt(2)*interfaceThickness)*np.cos(theta)*(phi_n**2 - 1)*v*ufl.ds


print("finished making AC forms \n\n")
opp_idx = {0: 0, 1: 3, 2: 4, 3: 1, 4: 2, 5: 7, 6: 8, 7: 5, 8: 6}

for idx in range(Q):

    bilinFormsStream.append(f_trial * v * ufl.dx)
    bilinFormsColl.append(f_trial*v*ufl.dx)

    double_dot_product_term = -0.5*dt_c**2 * ufl.inner(xi[idx], ufl.grad(f_star[idx]))\
        * ufl.inner(xi[idx], ufl.grad(v)) * ufl.dx

    dot_product_force_term = 0.5*dt_c**2 * ufl.inner(xi[idx], ufl.grad(v))\
        * body_Force(velStar_n, idx, forceDensity) * ufl.dx


    lin_form_idx = - dt_c*v*ufl.inner(xi[idx], ufl.grad(f_star[idx]))*ufl.dx\
        + dt_c*v*body_Force(velStar_n, idx, forceDensity)*ufl.dx\
        + double_dot_product_term\
        + dot_product_force_term
        
    f_eq_idx = f_equil(
        idx,
        velocity_n,
        density,
        velSquared)
        
    lin_form_coll = ( - dt_c/(tau) * (f_n[idx] - f_eq_idx) )*v*ufl.dx

    linear_forms_stream.append(lin_form_idx)
    linear_forms_collision.append(lin_form_coll)
    
print("finished making LB forms\n\n")

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
    x_dof = V.interpolate(
        x[0],
        name="x_dof"
    ).as_numpy.copy()

    y_dof = V.interpolate(
        x[1],
        name="y_dof"
    ).as_numpy.copy()

    N = len(x_dof)

    # -----------------------------------------------------
    # x periodicity only
    # -----------------------------------------------------

    x_periodic = x_dof.copy()

    x_periodic[
        np.abs(x_periodic - L_x) < 1e-10
    ] = 0.0

    # -----------------------------------------------------
    # y is NOT periodic
    # -----------------------------------------------------

    keys = np.column_stack((
        np.round(x_periodic, 10),
        np.round(y_dof, 10)
    ))

    unique_keys, inverse = np.unique(
        keys,
        axis=0,
        return_inverse=True
    )

    nPeriodic = len(unique_keys)

    rows = np.arange(N)
    cols = inverse
    data = np.ones(N)

    P = scipy.sparse.coo_matrix((data, (rows, cols)),
                                shape=(N, nPeriodic)).tocsr()
    
    sysMatStream = []
    sysMatColl = []

    massMatConsistent = scipy.sparse.csr_matrix(
    dune.fem.operator.galerkin(
        f_trial * v * ufl.dx
    ).linear().as_numpy)

    # Full-space lumped mass diagonal
    mass_diag = np.asarray(massMatConsistent.sum(axis=1)).ravel()
    
    # Periodic lumped mass diagonal
    mass_diag_periodic = np.asarray(P.T @ mass_diag).ravel()
    
    if np.any(mass_diag_periodic <= 0):
        raise ValueError("Periodic lumped mass contains non-positive entries")
    
    return P, mass_diag, mass_diag_periodic

P, mass_diag, mass_diag_periodic = build_solvers()

bottomDoFs, topDoFs = wall_dofs()
    

#%% Time-stepping 

print("about to start time-stepping\n\n")
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
        b_periodic = P.T @ b_full
    
        # Solve periodic system
        f_star_periodic = b_periodic / mass_diag_periodic
    
        # Expand back to full DUNE space
        f_star_arrays[idx][:] = P @ f_star_periodic + f_n_arrays[idx][:]
        

    
    #print("did collision \n\n")
        
    # Stream all populations
    for idx in range(Q):
        streamingOps[idx](zeroFn, rhsVecStreaming[idx])
    
        b_full = rhsVecStreaming[idx].as_numpy
        b_periodic = P.T @ b_full
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

    b_phi_periodic = P.T @ b_phi_full

    phi_periodic = b_phi_periodic / mass_diag_periodic

    phi_nP1.as_numpy[:] = ( P @ phi_periodic) + phi_n.as_numpy[:]
    
    b_mu_full = rhs_Mu_fn.as_numpy

    b_mu_periodic = P.T @ b_mu_full

    mu_periodic = b_mu_periodic / mass_diag_periodic

    mu_n.as_numpy[:] = (P @ mu_periodic)
    
    phi_n.assign(phi_nP1)
    
    
    mass_n = dune.fem.integrate((phi_n + 1) / 2)
    
    #print("mass_n = ", mass_n)
    mass_diff.assign( mass_n - mass_init )
    
    # ============================================================
    # AMR: adapt the grid and rebuild mesh-dependent objects
    # ============================================================
    
    
    if n == 500:
    
        # --------------------------------------------------------
        # 1. Mark and adapt the grid.
        #
        # Transfer persistent solution fields.
        # Do NOT transfer temporary RHS vectors; recreate them below.
        # --------------------------------------------------------
    
        marker = dune.fem.GridMarker(
            1 - phi_n**2,
            refineTolerance=0.6,
            coarsenTolerance=0.2,
            minLevel=0,
            maxLevel=1
        )
    
        dune.fem.gridAdapt(
            marker,
            f_n
            + f_star
            + [phi_n, phi_nP1, mu_n]
        )
    
        # --------------------------------------------------------
        # 2. Refresh ALL NumPy views after adaptation.
        # --------------------------------------------------------
    
        f_n_arrays = [
            f_n[i].as_numpy for i in range(Q)
        ]
    
        f_star_arrays = [
            f_star[i].as_numpy for i in range(Q)
        ]
    
        phi_n_array = phi_n.as_numpy
        phi_nP1_array = phi_nP1.as_numpy
        mu_n_array = mu_n.as_numpy
    
        # --------------------------------------------------------
        # 3. Recreate temporary functions on the adapted grid.
        # --------------------------------------------------------
    
        zeroFn = V.zero.copy()
    
        rhsVecCollision = [
            V.zero.copy() for _ in range(Q)
        ]
    
        rhsVecStreaming = [
            V.zero.copy() for _ in range(Q)
        ]
    
        rhs_AC_fn = V.zero.copy()
        rhs_Mu_fn = V.zero.copy()
    
        # --------------------------------------------------------
        # 4. Rebuild the collision and streaming operators.
        # --------------------------------------------------------
    
        collisionOps = []
        streamingOps = []
    
        for i in range(Q):
    
            collisionOps.append(
                dune.fem.operator.galerkin(
                    linear_forms_collision[i]
                    - f_trial * v * ufl.dx
                )
            )
    
            streamingOps.append(
                dune.fem.operator.galerkin(
                    linear_forms_stream[i]
                    - f_trial * v * ufl.dx
                )
            )
    
        # --------------------------------------------------------
        # 5. Rebuild the phase-field operators.
        # --------------------------------------------------------
    
        phi_form_op = dune.fem.operator.galerkin(
            lin_form_AC - f_trial * v * ufl.dx
        )
    
        mu_form_op = dune.fem.operator.galerkin(
            lin_form_mu - f_trial * v * ufl.dx
        )
    
        # --------------------------------------------------------
        # 6. Rebuild the periodic projection and lumped masses.
        # --------------------------------------------------------
    
        P, mass_diag, mass_diag_periodic = build_solvers()
    
        # --------------------------------------------------------
        # 7. Recalculate wall DOFs on the adapted grid.
        # Do not use bottomFn.as_numpy or topFn.as_numpy here.
        # --------------------------------------------------------
    
        bottomDoFs, topDoFs = wall_dofs()
    
        # --------------------------------------------------------
        # 8. Verify dimensions and transferred values.
        # --------------------------------------------------------
    

    
    if n % 50 == 0:
        
        print("n = ", n, "writing to file")
        #print("mass_diff type:", type(mass_diff))
        
        #print("mass_n type:", type(mass_n), "\n\n")
        
        if n > 1000:
            theta_avg = computeContactAngle(phi_n, h_min, interfaceThickness, mesh)
            print("theta = ", theta_avg)
            
            xh = V.interpolate(x[0], name="x")
            yh = V.interpolate(x[1], name="y")
            dof_coords = np.column_stack([xh.as_numpy, yh.as_numpy])  
            left_cl, right_cl = trackCL(phi_n, dof_coords, h_min)
            
            print("left_cl = ", left_cl, " right_cl = ", right_cl)
        
        vel_expr = getVel(f_n, forceDensity)
        vel3.interpolate(ufl.as_vector([vel_expr[0], vel_expr[1], 0]))
        
        v_array = vel3.as_numpy.reshape(-1, 3)

        vel_mag = np.linalg.norm(v_array, axis=1)
        
        print("Min speed:", vel_mag.min())
        print("Max speed:", vel_mag.max())
        
        print("Max KE:", vel_mag.max()**2)
        
        if n > 1000:
            log_file.write(f"{n:15d}"
                        f"{right_cl:15.3f}"
                        f"{vel_mag.max():15.8g}"
                        f"{theta_avg:15.2f}\n")
            log_file.flush()

        mesh.writeVTK(os.path.join(outDirName, f"vel_{n:06d}"), pointdata=[vel3])
        
        mesh.writeVTK(os.path.join(outDirName, f"phi_{n:06d}"),
            pointdata=[phi_n])
    
    