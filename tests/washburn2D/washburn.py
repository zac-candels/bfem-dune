import os
os.environ["DUNE_LOG_LEVEL"] = "ERROR"
from mpi4py import MPI
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
from collections import defaultdict
from dune.common import comm
import ufl
from mpi4py import MPI

R0 = 2
initDropDiam = 2*R0
A = 0.5
kappa = 0.02
interfaceThickness = np.sqrt(kappa/A)
M_tilde = 10
theta_deg = 70
theta = theta_deg * np.pi / 180

Q = 9

xi_values = [
    (0.0,  0.0),
    (1.0,  0.0),
    (0.0,  1.0),
    (-1.0, 0.0),
    (0.0, -1.0),
    (1.0,  1.0),
    (-1.0, 1.0),
    (-1.0, -1.0),
    (1.0, -1.0),
]

L_x = 16
L_y = 2
Nx = 35
Ny = 8

xc = 2

rho_init = 1.0
nu = 1/3
c_s = np.sqrt(1/3)

# Pressure reservoirs: equal reference pressure for spontaneous imbibition
rho_in = 1.0
rho_out = 1.0

p_in = c_s**2 * rho_in
p_out = c_s**2 * rho_out

tau_1 = 1     # relaxation time in the phi = +1 phase (the drop, as phi_init = -tanh(x - xc))
tau_2 = 0.9    # relaxation time in the phi = -1 phase

domain = dune.grid.cartesianDomain(
    [0.0, 0.0],
    [L_x, L_y],
    [Nx, Ny])

mesh = dune.fem.view.adaptiveLeafGridView(
    dune.alugrid.aluConformGrid(domain))
mesh.hierarchicalGrid.loadBalance()

h_min = np.inf

for element in mesh.elements:
    corners = np.asarray(element.geometry.corners)

    h_element = max(
        np.linalg.norm(a - b)
        for i, a in enumerate(corners)
        for b in corners[i + 1:])

    h_min = min(h_min, h_element)

h_min = comm.min(h_min)
h_min = h_min*(1/2**3)

Tfinal = 1500
dt_mult_fac = 0.001
dt = dt_mult_fac*h_min
dt_c = dune.ufl.Constant(dt, name="dt")
beta_mass_diff = 0.01*dt_c
numSteps = int(Tfinal/dt)


# ---------------- Output ----------------
WORKDIR = os.getcwd()
outDirName = os.path.join(WORKDIR, f"viscRatio_{tau_1/tau_2:1g}_dt{dt:1g}")
os.makedirs(outDirName, exist_ok=True)

log_file = open(outDirName + "/simulation_log.txt", "w")
if comm.rank == 0:
    log_file.write(f"{'n':>15}{'r_cl':>15}{'max ||u||':>15}{'theta':>15}\n")
    log_file.flush()

# D2Q9 weights
w = np.array([4/9]
             + [1/9]*4
             + [1/36]*4)

xi = [dune.ufl.Constant(c) for c in xi_values]


#%% Create finite-element space

if comm.rank == 0:
    print("\nSTEP 8: Creating finite-element space", flush=True)

V = dune.fem.space.lagrange(mesh, order=1)

x = ufl.SpatialCoordinate(V)

if comm.rank == 0:
    print("\nSTEP 9: Finite-element space created successfully", flush=True)

#%% Compute boundary DOF normals

def build_bounceback_dofs(mesh, V, xi_values, Q, tol=1e-8):
    """
    Build bounce-back DOF sets for the wall boundaries.

    DOF indices are taken from DUNE's Lagrange space, so they
    correspond to the ordering in f_n[idx].as_numpy.
    """

    # Coordinates in DUNE's actual DOF ordering
    x_dof = V.interpolate(x[0], name="x_dof_bb").as_numpy.copy()
    y_dof = V.interpolate(x[1], name="y_dof_bb").as_numpy.copy()

    dof_by_key = {tuple(np.round(np.array([xx, yy]) / tol).astype(np.int64)): i
                  for i, (xx, yy) in enumerate(zip(x_dof, y_dof))}

    # Accumulate boundary normals at each DUNE DOF
    normals_sum = defaultdict(lambda: np.zeros(2))

    for element in mesh.elements:
        for intersection in mesh.intersections(element):

            if not intersection.boundary:
                continue

            # In 2D the intersection is a 1D reference element
            normal = np.asarray(
                intersection.unitOuterNormal([0.5]),
                dtype=float
            )

            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue

            normal /= norm

            # Skip the x = 0 and x = L_x boundaries (periodic): their
            # normals have no y component.
            if abs(normal[1]) < 1e-8:
                continue

            for point in np.asarray(
                intersection.geometry.corners, dtype=float
            ):
                key = tuple(
                    np.round(point / tol).astype(np.int64)
                )

                dof = dof_by_key.get(key)

                if dof is not None:
                    normals_sum[dof] += normal

    # Normalize the averaged normal at each boundary DOF
    boundary_normals = {}

    for dof, normal_sum in normals_sum.items():
        norm = np.linalg.norm(normal_sum)

        if norm > 1e-12:
            boundary_normals[dof] = normal_sum / norm

    # Build population-specific bounce-back index arrays
    bouncebackDofs = {idx: [] for idx in range(1, Q)}

    ctol = 0.2
    for dof, normal in boundary_normals.items():
        for idx in range(1, Q):
            c = np.asarray(xi_values[idx])
            if np.dot(normal, c)/np.linalg.norm(c) < -ctol:
                bouncebackDofs[idx].append(dof)

    bouncebackDofs = {
        idx: np.asarray(dofs, dtype=int)
        for idx, dofs in bouncebackDofs.items()}

    return bouncebackDofs

#%% Calculate boundary vertex normals

bouncebackDofs = build_bounceback_dofs(mesh, V, xi_values, Q)


#%% Create function space and finite-element functions for the solution

V = dune.fem.space.lagrange(mesh, order=1)

# ParaView's Glyph filter needs 3-component vectors, so the velocity is
# stored with a zero z-component even though the simulation is 2D.
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
    f_star.append( initFn.copy(name=f"f_star{idx}") )
    f_nP1.append( initFn.copy(name=f"f_nP1{idx}") )

phi_nP1 = initFn.copy(name="phi_nP1")
mu_n = initFn.copy(name="mu_n")
dens_n = initFn.copy(name="dens_n")

x = ufl.SpatialCoordinate(V)

phiInitExpr = -ufl.tanh( (x[0]-xc )\
                        / (ufl.sqrt(2)*interfaceThickness) )

phi_n = V.interpolate(phiInitExpr, name="phi_n")
mass_init = dune.fem.integrate((phi_n + 1) / 2)
mass_diff = dune.ufl.Constant(1e-9)

tau =(1 + phi_n)*tau_1/2 + (1 - phi_n)*tau_2/2

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

    vel_0 = dune.ufl.Constant((0.0, 0.0))

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
    Estimate the left and right contact-line positions (in x) from
    nodal values of phi_n.

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

    # Exclude points near the opposite contact line.
    min_x = candidates[:, 0].min()
    keep_rightCL = candidates[:, 0] > min_x + 7 * h

    max_x = candidates[:, 0].max()
    keep_leftCL = candidates[:, 0] < max_x - 7*h

    candidates_rightCL = candidates[keep_rightCL]
    candidates_leftCL = candidates[keep_leftCL]

    # Leftmost and rightmost remaining points.
    left_idx = np.argmin(candidates_leftCL[:, 0])
    right_idx = np.argmax(candidates_rightCL[:, 0])

    left_x = candidates_leftCL[left_idx, 0]
    right_x = candidates_rightCL[right_idx, 0]

    return left_x, right_x


# Function to compute contact angle
def computeContactAngle(c_n, h, Cn, mesh):
    angles = []
    n_vec = np.array([0.0, -1.0])

    local_c = c_n.localFunction()

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

    if not angles:
        raise ValueError(
            "No valid interfacial points found near the substrate."
        )

    return np.degrees(np.mean(angles))


#%% Define BCs

if comm.rank == 0:
    print("starting to create BCs\n\n", flush=True)

def wall_dofs():
    y = V.interpolate(x[1], name="y_dof").as_numpy
    bottom = np.where(np.abs(y) < 1e-8)[0]
    top    = np.where(np.abs(L_y - y) < 1e-8)[0]
    return bottom, top

if comm.rank == 0:
    print("done making BCs\n\n", flush=True)

#%% Define linear and bilinear forms

bilinFormsStream = []
linear_forms_stream = []

bilinFormsColl = []
linear_forms_collision = []


bilin_form_AC = phi_trial * v * ufl.dx
bilin_form_mu = mu_trial * v * ufl.dx

if comm.rank == 0:
    print("going to create ufl functions for density, vel, velStar \n\n", flush=True)

density = getDens(f_n)
velocity_n = getVel(f_n, forceDensity)
velStar_n = getVel(f_star, forceDensity)
velSquared = ufl.inner(velocity_n, velocity_n)

if comm.rank == 0:
    print("about to start making AC forms \n\n", flush=True)

lin_form_AC =  - dt*v*ufl.dot(velocity_n, ufl.grad(phi_n))*ufl.dx\
    - dt*M_tilde*v*mu_n*ufl.dx - 0.5*dt**2 * ufl.dot(velocity_n, ufl.grad(v)) * ufl.dot(velocity_n, ufl.grad(phi_n)) *ufl.dx

# Wetting acts on the wall only
wall_ind = ufl.conditional(ufl.And(x[0] > 1e-6, x[0] < L_x - 1e-6), 1.0, 0.0)

lin_form_mu = A*phi_n*(phi_n**2 - 1)*v*ufl.dx \
    + kappa*ufl.dot(ufl.grad(phi_n), ufl.grad(v))*ufl.dx \
    + wall_ind*kappa/(np.sqrt(2)*interfaceThickness)*np.cos(theta) \
      *(phi_n**2 - 1)*v*ufl.ds

wall_len  = dune.fem.assemble(f_trial*v*wall_ind*ufl.ds).as_numpy.sum()
total_len = dune.fem.assemble(f_trial*v*ufl.ds).as_numpy.sum()

if comm.rank == 0:
    print("wall:", wall_len, " rest:", total_len - wall_len, flush=True)
    print("finished making AC forms \n\n", flush=True)

opp_idx = {i: xi_values.index((-xi_values[i][0], -xi_values[i][1]))
           for i in range(Q)}

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

# Assemble operators for first step
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


f_nP1_arrays = []
f_n_arrays = []
f_star_arrays = []
for idx in range(Q):
    f_nP1_arrays.append(f_nP1[idx].as_numpy)
    f_n_arrays.append(f_n[idx].as_numpy)
    f_star_arrays.append(f_star[idx].as_numpy)
    
    
def apply_pressure_boundary_conditions():

    # Current density at the inlet
    rho_boundary = np.zeros(len(inletDofs))

    for i in range(Q):
        rho_boundary += f_n_arrays[i][inletDofs]

    # Correct the density without changing momentum
    delta_rho = rho_in - rho_boundary

    for i in range(Q):
        f_n_arrays[i][inletDofs] += w[i] * delta_rho


    # Current density at the outlet
    rho_boundary = np.zeros(len(outletDofs))

    for i in range(Q):
        rho_boundary += f_n_arrays[i][outletDofs]

    # Correct the density without changing momentum
    delta_rho = rho_out - rho_boundary

    for i in range(Q):
        f_n_arrays[i][outletDofs] += w[i] * delta_rho
    
    
def build_pressure_boundary_dofs():

    x_dof = V.interpolate(
        x[0], name="x_pressure_dof"
    ).as_numpy.copy()

    y_dof = V.interpolate(
        x[1], name="y_pressure_dof"
    ).as_numpy.copy()

    tol = 1e-8

    # Exclude the four wall corners. They belong to the
    # horizontal-wall boundary treatment.
    inlet = np.where(
        (np.abs(x_dof) < tol) &
        (y_dof > tol) &
        (y_dof < L_y - tol)
    )[0]

    outlet = np.where(
        (np.abs(x_dof - L_x) < tol) &
        (y_dof > tol) &
        (y_dof < L_y - tol)
    )[0]

    return inlet, outlet


inletDofs, outletDofs = build_pressure_boundary_dofs()

def build_solvers():

    # 1. Local DOF coordinates
    x_local = V.interpolate(x[0], name="x_dof").as_numpy.copy()
    y_local = V.interpolate(x[1], name="y_dof").as_numpy.copy()

    N_local = len(x_local)

    # 2. Gather coordinates across MPI ranks
    x_full = np.concatenate(
        MPI.COMM_WORLD.allgather(x_local)
    )
    y_full = np.concatenate(
        MPI.COMM_WORLD.allgather(y_local)
    )

    N_global = len(x_full)

    # 3. Identify coincident DOFs shared between MPI ranks.
    #    Do NOT identify x=0 with x=L_x.
    keys = np.column_stack((
        np.round(x_full, 10),
        np.round(y_full, 10)
    ))

    unique_keys, inverse = np.unique(
        keys, axis=0, return_inverse=True
    )

    nGlobal = len(unique_keys)

    # 4. Global projection matrix
    P_global = scipy.sparse.coo_matrix(
        (
            np.ones(N_global),
            (np.arange(N_global), inverse)
        ),
        shape=(N_global, nGlobal)
    ).tocsr()

    # 5. Local rows of projection matrix
    local_sizes = MPI.COMM_WORLD.allgather(N_local)

    start_idx = sum(local_sizes[:comm.rank])
    end_idx = start_idx + N_local

    P_local = P_global[start_idx:end_idx, :]

    # 6. Lumped mass matrix
    lumpOp = dune.fem.operator.galerkin(
        1.0 * v * ufl.dx - f_trial * v * ufl.dx
    )

    lumpVec = V.zero.copy()
    lumpOp(zeroFn, lumpVec)

    mass_diag_local = lumpVec.as_numpy.copy()

    mass_diag_full = np.concatenate(
        MPI.COMM_WORLD.allgather(mass_diag_local)
    )

    mass_diag_global = np.asarray(
        P_global.T @ mass_diag_full
    ).ravel()

    if np.any(mass_diag_global <= 0):
        raise ValueError(
            "Non-periodic lumped mass contains non-positive entries"
        )

    return P_local, mass_diag_local, mass_diag_global


P, mass_diag, mass_diag_global = build_solvers()

bottomDoFs, topDoFs = wall_dofs()

#%% Refine mesh once before time-stepping commences

def adapt_mesh():
    global f_star_arrays, f_n_arrays, P, mass_diag, mass_diag_global, bouncebackDofs
    marker = dune.fem.GridMarker(1 - phi_n**2, refineTolerance=0.6,
                                 coarsenTolerance=0.2, minLevel=0, maxLevel=4)
    dune.fem.gridAdapt(marker, f_n + f_star + rhsVecCollision + rhsVecStreaming
                       + [phi_n, mu_n, zeroFn, rhs_AC_fn, rhs_Mu_fn])
    f_star_arrays = [f.as_numpy for f in f_star]
    f_n_arrays    = [f.as_numpy for f in f_n]
    P, mass_diag, mass_diag_global = build_solvers()
    bouncebackDofs = build_bounceback_dofs(mesh, V, xi_values, Q)

for _ in range(8):                       # before the time loop
    phi_n.interpolate(phiInitExpr)       # re-sample exact profile on current mesh
    adapt_mesh()
phi_n.interpolate(phiInitExpr)
mass_init = dune.fem.integrate((phi_n + 1)/2)


#%% Time-stepping

if comm.rank == 0:
    print("about to start time-stepping\n\n")

for n in range(numSteps):

    phi_form_op(zeroFn, rhs_AC_fn)
    mu_form_op(zeroFn, rhs_Mu_fn)

    # Do collision
    for idx in range(Q):
        collisionOps[idx](zeroFn, rhsVecCollision[idx])

        # Full DUNE RHS
        b_full = rhsVecCollision[idx].as_numpy

        # Reduce RHS to periodic space
        b_periodic = comm.sum(P.T @ b_full)

        # Solve periodic system
        f_star_periodic = b_periodic / mass_diag_global

        # Expand back to full DUNE space
        f_star_arrays[idx][:] = P @ f_star_periodic + f_n_arrays[idx][:]

    # Stream all populations
    for idx in range(Q):
        streamingOps[idx](zeroFn, rhsVecStreaming[idx])

        b_full = rhsVecStreaming[idx].as_numpy
        b_periodic = comm.sum( P.T @ b_full )
        f_n_periodic = b_periodic / mass_diag_global

        f_n_arrays[idx][:] = P @ f_n_periodic + f_star_arrays[idx][:]
        
    apply_pressure_boundary_conditions()

    # Bounce-back
    for idx in range(1, Q):
        dofs = bouncebackDofs[idx]
        f_n_arrays[idx][dofs] = (
            f_star_arrays[opp_idx[idx]][dofs])

    # Phase field and chemical potential
    b_phi_full = rhs_AC_fn.as_numpy
    b_phi_periodic = comm.sum( P.T @ b_phi_full )
    phi_periodic = b_phi_periodic / mass_diag_global
    phi_nP1.as_numpy[:] = ( P @ phi_periodic) + phi_n.as_numpy[:]

    b_mu_full = rhs_Mu_fn.as_numpy
    b_mu_periodic = comm.sum( P.T @ b_mu_full )
    mu_periodic = b_mu_periodic / mass_diag_global
    mu_n.as_numpy[:] = (P @ mu_periodic)

    phi_n.assign(phi_nP1)

    mass_n = dune.fem.integrate((phi_n + 1) / 2)
    mass_diff.assign( mass_n - mass_init )

    if n % 200 == 0:

        marker = dune.fem.GridMarker(
            1 - phi_n**2,
            refineTolerance=0.6,
            coarsenTolerance=0.2,
            minLevel=0,
            maxLevel=4)

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

        # Rebuild periodic matrix / lumped mass
        P, mass_diag, mass_diag_global = build_solvers()
        
        inletDofs, outletDofs = build_pressure_boundary_dofs()

        bouncebackDofs = build_bounceback_dofs(mesh, V, xi_values, Q)

    if n % 2000 == 0:

        if comm.rank == 0:
            print("n = ", n, "writing to file", flush=True)

        if n > 3000:
            theta_avg = 1 #computeContactAngle(phi_n, h_min, interfaceThickness, mesh)

            if comm.rank == 0:
                print("theta = ", theta_avg, flush=True)

            xh = V.interpolate(x[0], name="x")
            yh = V.interpolate(x[1], name="y")
            dof_coords = np.column_stack([xh.as_numpy, yh.as_numpy])
            left_cl, right_cl = 1, 1 #trackCL(phi_n, dof_coords, h_min)

            if comm.rank == 0:
                print("left_cl = ", left_cl, " right_cl = ", right_cl, flush=True)

        vel_expr = getVel(f_n, forceDensity)
        vel3.interpolate(ufl.as_vector([vel_expr[0], vel_expr[1], 0.0]))
        
        dens_expr = getDens(f_n)
        dens_n.interpolate(dens_expr)

        v_array = vel3.as_numpy.reshape(-1, 3)

        vel_mag = np.linalg.norm(v_array, axis=1)

        if comm.rank == 0:
            print("Min speed:", vel_mag.min(), flush=True)
            print("Max speed:", vel_mag.max(), flush=True)

            print("Max KE:", vel_mag.max()**2, flush=True)

        if n > 3000:
            if comm.rank == 0:
                log_file.write(f"{n:15d}"
                            f"{right_cl:15.3f}"
                            f"{vel_mag.max():15.8g}"
                            f"{theta_avg:15.2f}\n")
                log_file.flush()

        mesh.writeVTK(os.path.join(outDirName, f"vel_{n:06d}"), pointdata=[vel3])

        mesh.writeVTK(os.path.join(outDirName, f"phi_{n:06d}"),
            pointdata=[phi_n])
        
        mesh.writeVTK(os.path.join(outDirName, f"dens_{n:06d}"),
            pointdata=[dens_n])
