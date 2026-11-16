import os
os.environ["DUNE_LOG_LEVEL"] = "ERROR"

from mpi4py import MPI
from scipy.sparse import linalg
from math import sqrt
import matplotlib.pyplot as plt
from dune.ufl import Constant, DirichletBC
import dune
import dune.alugrid
import scipy
import numpy as np
import os
import time
from dune.common import comm

import ufl

# ----------------------------------------------------------------------
# Physical / numerical parameters
# ----------------------------------------------------------------------
A = 2
kappa = 0.02
interfaceThickness = np.sqrt(kappa/A)
M_tilde = 10
theta_deg = 60.0          # equilibrium contact angle (measured in the liquid, phi = +1)
theta = theta_deg * np.pi / 180
tau = 0.05

Q = 9

L_x = 28
L_y = 4
Nx = 70
Ny = 10

# --- Capillary bridge geometry -----------------------------------------
# The bridge spans the gap between the two walls (height H = L_y).
# W_bridge is the half-width of the bridge AT THE WALLS (i.e. the contact
# lines start at x = xc -/+ W_bridge). The menisci are circular arcs that
# meet both walls at the equilibrium angle theta, so the bridge starts
# in static equilibrium.
xc = L_x/2
W_bridge = 6.0

# --- Wall velocities (lattice units; keep U << c_s ~ 0.577) -------------
# Top wall slides in +x with constant speed U. The bottom wall is at rest by
# default. U_bottom = -U_top gives an antisymmetric (zero net flux, no drift)
# Couette flow, which is cleaner in a closed box.
U = 0.02
u_wall_top = np.array([U, 0.0])
u_wall_bottom = np.array([0.0, 0.0])

rho_init = 1.0
c_s = np.sqrt(1/3)

# Estimates for the capillary number (assumes nu = c_s^2 * tau)
sigma = (2*np.sqrt(2)/3)*np.sqrt(kappa*A)
mu_visc = rho_init*c_s**2*tau
Ca = mu_visc*U/sigma

#%% Prepare output
WORKDIR = os.getcwd()
outDirName = os.path.join(
    WORKDIR, "test")#f"bridge_theta{theta_deg:1g}_U{U:g}_A{A}_tau{tau}")
os.makedirs(outDirName, exist_ok=True)

log_file = open(outDirName + "/simulation_log.txt", "w")
log_file.write(f"{'n':>10}{'t':>12}"
               f"{'xCL_BL':>11}{'xCL_BR':>11}{'xCL_TL':>11}{'xCL_TR':>11}"
               f"{'th_BL':>9}{'th_BR':>9}{'th_TL':>9}{'th_TR':>9}"
               f"{'max||u||':>13}\n")
log_file.flush()

if comm.rank == 0:
    print(f"sigma = {sigma:.5g}, mu = {mu_visc:.5g}, U = {U}, "
          f"Ca = mu*U/sigma = {Ca:.4g}, Ma = {U/c_s:.3g}", flush=True)

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

Tfinal = 4500
dt = 0.05*h_min
beta_mass_diff = 0.1*dt
numSteps = int(Tfinal/dt)

# Corresponding weights
w = np.array([
    4/9,
    1/9, 1/9, 1/9, 1/9,
    1/36, 1/36, 1/36, 1/36
])

# Numeric lattice directions (used for the moving-wall correction)
c_np = np.array([
    [0, 0],
    [1, 0], [0, 1], [-1, 0], [0, -1],
    [1, 1], [-1, 1], [-1, -1], [1, -1],
], dtype=float)

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
    f_star.append( initFn.copy(name=f"f_star{idx}") )
    f_nP1.append( initFn.copy(name=f"f_nP1{idx}") )

phi_nP1 = initFn.copy(name="phi_nP1")
mu_n = initFn.copy(name="mu_n")

x = ufl.SpatialCoordinate(V)

# ----------------------------------------------------------------------
# Initial condition: capillary bridge (liquid = phi = +1 between the walls)
# ----------------------------------------------------------------------
# By symmetry about x = xc we work with X = |x - xc| and describe the right
# meniscus only. For theta < 90 deg the meniscus is an arc of radius
#       R = H / (2 cos(theta))
# whose centre sits at X = W_bridge + R sin(theta), y = H/2. The signed
# distance d is > 0 in the liquid. min_value(.., cx - X) removes the
# spurious "liquid" region on the far side of the circle.
H_gap = L_y
X = abs(x[0] - xc)

if theta_deg < 90.0 - 1e-6:
    R_arc = H_gap/(2*np.cos(theta))
    cx_arc = W_bridge + R_arc*np.sin(theta)
    dist_c = ufl.sqrt((X - cx_arc)**2 + (x[1] - H_gap/2)**2)
    d_signed = ufl.min_value(dist_c - R_arc, cx_arc - X)
else:
    # theta >= 90 deg: start from a straight-sided column and let the
    # wetting boundary condition relax it.
    d_signed = W_bridge - X

phiInitExpr = ufl.tanh(d_signed / (ufl.sqrt(2)*interfaceThickness))
phiInitExpr = 1

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


# Initial velocity field: linear (Couette) profile between the two walls.
# Set INIT_COUETTE = False to start from rest (impulsive start of the wall).
INIT_COUETTE = True

def fEquilInit(vel_idx):

    if INIT_COUETTE:
        eta = x[1]/L_y
        vel_0 = ufl.as_vector((
            (1 - eta)*float(u_wall_bottom[0]) + eta*float(u_wall_top[0]),
            0.0))
    else:
        vel_0 = ufl.as_vector((0.0, 0.0))

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


#%% Contact-line tracking and contact-angle measurement
# Four contact lines: BL, BR (bottom wall, left/right) and TL, TR (top wall).

def trackCL(phi_n, dof_coords, wallDoFs):
    """
    Nodal estimate of the left/right contact-line positions on one wall:
    leftmost / rightmost wall node with phi > 0. Accuracy ~ local mesh size.
    """
    values = np.asarray(phi_n.as_numpy).reshape(-1)[wallDoFs]
    xw = dof_coords[wallDoFs, 0]

    liquid = values > 0
    left_x = comm.min(xw[liquid].min() if liquid.any() else np.inf)
    right_x = comm.max(xw[liquid].max() if liquid.any() else -np.inf)

    return left_x, right_x


def computeContactAngles(c_n, h, mesh, x_split_bottom, x_split_top):
    """
    Mean contact angle at each of the four contact lines, measured in the
    liquid (phi = +1) as the angle between grad(phi) and the outward wall
    normal. Elements within 5h of a wall with |phi| < 0.5 are used; they are
    assigned to the left/right contact line by comparing their x-coordinate
    with x_split_* (midpoint of the two contact lines on that wall).
    """
    keys = ("BL", "BR", "TL", "TR")
    sums = {k: 0.0 for k in keys}
    counts = {k: 0.0 for k in keys}

    local_c = c_n.localFunction()

    for element in mesh.elements:
        geometry = element.geometry

        midpoint = np.asarray(geometry.center, dtype=float)

        if midpoint[1] < 5*h:
            wall, n_vec, x_split = "B", np.array([0.0, -1.0]), x_split_bottom
        elif midpoint[1] > L_y - 5*h:
            wall, n_vec, x_split = "T", np.array([0.0, 1.0]), x_split_top
        else:
            continue

        side = "L" if midpoint[0] < x_split else "R"
        key = wall + side

        local_point = geometry.toLocal(midpoint.tolist())

        local_c.bind(element)
        try:
            value = float(np.asarray(local_c(local_point)).reshape(-1)[0])

            if not (-0.5 < value < 0.5):
                continue

            grad_c = np.asarray(local_c.jacobian(local_point)).reshape(-1)
            grad_norm = np.linalg.norm(grad_c)
            if grad_norm < 1e-14:
                continue

            cos_theta = np.dot(grad_c, n_vec) / grad_norm
            sums[key] += np.arccos(np.clip(cos_theta, -1.0, 1.0))
            counts[key] += 1.0
        finally:
            local_c.unbind()

    angles = {}
    for k in keys:
        n_k = comm.sum(counts[k])
        s_k = comm.sum(sums[k])
        angles[k] = np.degrees(s_k/n_k) if n_k > 0 else np.nan
    return angles


#%% Wall DoFs

def wall_dofs():
    y = V.interpolate(x[1], name="y_dof").as_numpy
    bottom = np.where(np.abs(y) < 1e-8)[0]
    top    = np.where(np.abs(L_y - y) < 1e-8)[0]

    x_coord = V.interpolate(x[0], name="x_dof").as_numpy
    left = np.where(np.abs(x_coord) < 1e-8)[0]
    right    = np.where(np.abs(L_x - x_coord) < 1e-8)[0]
    return bottom, top, left, right

#%% Define BCs (matrix Dirichlet rows, kept from the original script)

if comm.rank == 0:
    print("starting to create BCs\n\n", flush=True)

u_D = 1

dbcBottom = dune.ufl.DirichletBC(V, u_D, x[1] < 1e-8)
dbcTop = dune.ufl.DirichletBC(V, u_D, abs(L_y - x[1]) <  1e-8 )

if comm.rank == 0:
    print("done making BCs\n\n", flush=True)

#%% Define linear and bilinear forms

bilinFormsStream = []
linear_forms_stream = []

bilinFormsColl = []
linear_forms_collision = []

bilin_form_AC = phi_trial * v * ufl.dx
bilin_form_mu = mu_trial * v * ufl.dx

density = getDens(f_n)
velocity_n = getVel(f_n, forceDensity)
velStar_n = getVel(f_star, forceDensity)
velSquared = ufl.inner(velocity_n, velocity_n)

lin_form_AC =  - dt*v*ufl.dot(velocity_n, ufl.grad(phi_n))*ufl.dx\
    - dt*M_tilde*v*mu_n*ufl.dx - (beta_mass_diff/dt)*mass_diff*ufl.sqrt( ufl.dot(ufl.grad(phi_n), ufl.grad(phi_n)) )*v*ufl.dx\
        - 0.5*dt**2 * ufl.dot(velocity_n, ufl.grad(v)) * ufl.dot(velocity_n, ufl.grad(phi_n)) *ufl.dx

# Wetting boundary term. Boundary ids of dune.grid.cartesianDomain in 2D:
# 1 = x-min, 2 = x-max, 3 = y-min (bottom), 4 = y-max (top).
# The bridge touches BOTH plates, so the wetting term is applied on ds(3)
# and ds(4) with the same equilibrium angle.
wetting_term = kappa/(np.sqrt(2)*interfaceThickness)*np.cos(theta)*(phi_n**2-1)*v

lin_form_mu =  A* phi_n*(phi_n**2 - 1)*v*ufl.dx\
    + kappa*ufl.dot(ufl.grad(phi_n),ufl.grad(v))*ufl.dx\
        + wetting_term*ufl.ds(3)\
        + wetting_term*ufl.ds(4)

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

rhsVecCollision = []
rhsVecStreaming = []

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
rhs_AC_fn = V.zero.copy()
rhs_Mu_fn = V.zero.copy()

zeroFn = V.zero.copy()

f_star_arrays = [f.as_numpy for f in f_star]
f_n_arrays = [f.as_numpy for f in f_n]

def build_solvers():

    lumpOp = dune.fem.operator.galerkin(
        1.0 * v * ufl.dx - f_trial * v * ufl.dx
    )

    lumpVec = V.zero.copy()
    lumpOp(zeroFn, lumpVec)

    mass_diag_local = lumpVec.as_numpy.copy()

    return mass_diag_local

mass_diag_local = build_solvers()

bottomDoFs, topDoFs, leftDoFs, rightDoFs = wall_dofs()


#%% Mesh adaptation

def adapt_mesh():
    # NOTE: the wall-DoF index arrays must be declared global here, otherwise
    # the refreshed arrays are only bound to local names and the time loop
    # keeps using stale indices after the pre-refinement.
    global f_star_arrays, f_n_arrays, mass_diag_local
    global bottomDoFs, topDoFs, leftDoFs, rightDoFs
    marker = dune.fem.GridMarker(1 - phi_n**2, refineTolerance=0.15,
                                  coarsenTolerance=0.15, minLevel=0, maxLevel=5)
    dune.fem.gridAdapt(marker, f_n + f_star + rhsVecCollision + rhsVecStreaming
                        + [phi_n, mu_n, zeroFn, rhs_AC_fn, rhs_Mu_fn])

    # Never use the old .as_numpy references after gridAdapt.
    f_star_arrays = [f.as_numpy for f in f_star]
    f_n_arrays    = [f.as_numpy for f in f_n]
    mass_diag_local = build_solvers()
    bottomDoFs, topDoFs, leftDoFs, rightDoFs = wall_dofs()

# Refine mesh around the interface before time-stepping commences
for _ in range(8):
    phi_n.interpolate(phiInitExpr)       # re-sample exact profile on current mesh
    adapt_mesh()
phi_n.interpolate(phiInitExpr)
mass_init = dune.fem.integrate((phi_n + 1)/2)


#%% Time-stepping

def moving_wall_term(idx, u_wall):
    """Ladd correction 2 w_i rho_0 (c_i . u_w) / c_s^2 for the reflected population i."""
    return 2.0*w[idx]*rho_init*float(c_np[idx] @ u_wall)/c_s**2

if comm.rank == 0:
    print("about to start time-stepping\n\n", flush=True)

for n in range(numSteps):

    phi_form_op(zeroFn, rhs_AC_fn)
    mu_form_op(zeroFn, rhs_Mu_fn)

    # Do collision
    for idx in range(Q):
        collisionOps[idx](zeroFn, rhsVecCollision[idx])
        b_local = rhsVecCollision[idx].as_numpy

        f_star_local = b_local / mass_diag_local

        f_star_arrays[idx][:] = f_star_local + f_n_arrays[idx][:]

    # Stream all populations
    for idx in range(Q):
        streamingOps[idx](zeroFn, rhsVecStreaming[idx])
        b_local = rhsVecStreaming[idx].as_numpy

        f_n_local = b_local / mass_diag_local

        f_n_arrays[idx][:] = f_n_local + f_star_arrays[idx][:]

    # Bottom wall (velocity u_wall_bottom): reflect populations into the fluid
    for idx in (2, 5, 6):
        f_n_arrays[idx][bottomDoFs] = (
            f_star_arrays[opp_idx[idx]][bottomDoFs]
            + moving_wall_term(idx, u_wall_bottom))

    # Top wall (MOVING with velocity u_wall_top): bounce-back + momentum term
    for idx in (4, 7, 8):
        f_n_arrays[idx][topDoFs] = (
            f_star_arrays[opp_idx[idx]][topDoFs]
            + moving_wall_term(idx, u_wall_top))

    # Left wall (at rest)
    for idx in (5, 1, 8):
        f_n_arrays[idx][leftDoFs] = f_star_arrays[opp_idx[idx]][leftDoFs]

    # Right wall (at rest)
    for idx in (6, 3, 7):
        f_n_arrays[idx][rightDoFs] = f_star_arrays[opp_idx[idx]][rightDoFs]

    b_phi_local = rhs_AC_fn.as_numpy

    phi_local = b_phi_local / mass_diag_local

    phi_nP1.as_numpy[:] = phi_local + phi_n.as_numpy[:]

    phi_n.assign(phi_nP1)

    b_mu_local = rhs_Mu_fn.as_numpy

    mu_local = b_mu_local / mass_diag_local

    mu_n.as_numpy[:] = mu_local

    mass_n = dune.fem.integrate((phi_n + 1) / 2)

    mass_diff.assign( mass_n - mass_init )

    if n % 10 == 0:
        adapt_mesh()

    if n % 100 == 0:

        if comm.rank == 0:
            print("n = ", n, "writing to file", flush=True)

        xh = V.interpolate(x[0], name="x")
        yh = V.interpolate(x[1], name="y")
        dof_coords = np.column_stack([xh.as_numpy, yh.as_numpy])

        xBL, xBR = trackCL(phi_n, dof_coords, bottomDoFs)
        xTL, xTR = trackCL(phi_n, dof_coords, topDoFs)

        angles = computeContactAngles(
            phi_n, h_min, mesh,
            x_split_bottom=0.5*(xBL + xBR),
            x_split_top=0.5*(xTL + xTR))

        vel_expr = getVel(f_n, forceDensity)
        vel3.interpolate(ufl.as_vector([vel_expr[0], vel_expr[1], 0]))

        v_array = vel3.as_numpy.reshape(-1, 3)
        vel_mag = np.linalg.norm(v_array, axis=1)
        max_speed = comm.max(float(vel_mag.max()))

        if comm.rank == 0:
            print(f"CL x:  BL={xBL:.3f} BR={xBR:.3f} TL={xTL:.3f} TR={xTR:.3f}",
                  flush=True)
            print("angles:", {k: round(float(a), 2) for k, a in angles.items()},
                  flush=True)
            print("Max speed:", max_speed, flush=True)

            if n > 200:
                log_file.write(f"{n:10d}{n*dt:12.4f}"
                               f"{xBL:11.3f}{xBR:11.3f}{xTL:11.3f}{xTR:11.3f}"
                               f"{angles['BL']:9.2f}{angles['BR']:9.2f}"
                               f"{angles['TL']:9.2f}{angles['TR']:9.2f}"
                               f"{max_speed:13.6g}\n")
                log_file.flush()

        mesh.writeVTK(os.path.join(outDirName, f"vel_{n:06d}"), pointdata=[vel3])

        mesh.writeVTK(os.path.join(outDirName, f"phi_{n:06d}"),
            pointdata=[phi_n])