import os
os.environ["DUNE_LOG_LEVEL"] = "ERROR"

import numpy as np
import ufl
import dune
import dune.alugrid
import dune.fem
from dune.common import comm

# ---------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------
R0 = 2
A = 0.5
kappa = 0.02
interfaceThickness = np.sqrt(kappa / A)
M_tilde = 10
theta_deg = 30
theta = theta_deg * np.pi / 180

Q = 9
L_x, L_y = 16, 4
Nx, Ny = 40, 10

rho_init = 1.0
c_s = np.sqrt(1 / 3)
tau = 0.1

ADAPT_EVERY = 10      # grid adaptation interval
OUT_EVERY = 10        # vtk / velocity output interval
DIAG_STEPS = 20       # print global diagnostics every step for the first steps
USE_SYNC = False       # call .communicate() on DoF vectors after NumPy writes, if available

xc, yc = L_x / 2, R0 - 0.6 * R0

# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------
outDirName = os.path.join(os.getcwd(), "test2")
os.makedirs(outDirName, exist_ok=True)

log_file = None
if comm.rank == 0:
    log_file = open(os.path.join(outDirName, "simulation_log.txt"), "w")
    log_file.write(f"{'n':>15}{'r_cl':>15}{'max ||u||':>15}{'theta':>15}\n")
    log_file.flush()

# ---------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------
domain = dune.grid.cartesianDomain([0.0, 0.0], [L_x, L_y], [Nx, Ny])
mesh = dune.fem.view.adaptiveLeafGridView(
    dune.alugrid.aluConformGrid(domain, lbMethod=9))
mesh.hierarchicalGrid.loadBalance()

h_min = np.inf
for element in mesh.elements:
    corners = np.asarray(element.geometry.corners)
    h_element = max(np.linalg.norm(a - b)
                    for i, a in enumerate(corners) for b in corners[i + 1:])
    h_min = min(h_min, h_element)
h_min = comm.min(h_min) * (1 / 2**3)     # same value on every rank

Tfinal = 1500
dt = 0.05 * h_min
beta_mass_diff = 0.1 * dt
numSteps = int(Tfinal / dt)

w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])

xi = [dune.ufl.Constant(c) for c in [
    (0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0),
    (1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)]]

opp_idx = {0: 0, 1: 3, 2: 4, 3: 1, 4: 2, 5: 7, 6: 8, 7: 5, 8: 6}

# ---------------------------------------------------------------------
# Spaces and functions
# ---------------------------------------------------------------------
V = dune.fem.space.lagrange(mesh, order=1)
V_vec3 = dune.fem.space.lagrange(mesh, order=1, dimRange=3)
vel3 = V_vec3.interpolate(dune.ufl.Constant((0, 0, 0)), name="vel")

v = ufl.TestFunction(V)
f_trial = ufl.TrialFunction(V)
x = ufl.SpatialCoordinate(V)

initFn = V.interpolate(0, "initFn")
f_n, f_star = [], []
for idx in range(Q):
    f_star.append(initFn.copy(name=f"f_star{idx}"))

phi_nP1 = initFn.copy(name="phi_nP1")
mu_n = initFn.copy(name="mu_n")

phiInitExpr = -ufl.tanh(
    (ufl.sqrt(pow(x[0] - xc, 2) + pow(x[1] - yc, 2)) - R0)
    / (ufl.sqrt(2) * interfaceThickness))
phi_n = V.interpolate(phiInitExpr, name="phi_n")

mass_init = dune.fem.integrate((phi_n + 1) / 2)
mass_diff = dune.ufl.Constant(1e-9)

forceDensity = -phi_n * ufl.grad(mu_n)

# ---------------------------------------------------------------------
# LB building blocks
# ---------------------------------------------------------------------
def getDens(f):
    return sum(f)


def getVel(f, forceDensity):
    density = sum(f)
    momentum = f[0] * xi[0]
    for idx in range(1, Q):
        momentum += f[idx] * xi[idx]
    return momentum / density + forceDensity * dt / (2 * density)


def fEquilInit(vel_idx):
    vel_0 = dune.ufl.Constant((0.0, 0.0))
    c = xi[vel_idx]
    c_dot_u = ufl.inner(c, vel_0)
    return w[vel_idx] * rho_init * (
        1 + c_dot_u / c_s**2 + c_dot_u**2 / (2 * c_s**4)
        - ufl.inner(vel_0, vel_0) / (2 * c_s**2))


def f_equil(vel_idx, velocity, density, velSquared):
    c_dot_u = ufl.inner(velocity, xi[vel_idx])
    return w[vel_idx] * density * (1 + 3 * c_dot_u + 4.5 * c_dot_u**2 - 1.5 * velSquared)


def body_Force(vel, vel_idx, Force_density):
    inverse_cs2 = 1 / c_s**2
    inverse_cs4 = 1 / c_s**4
    xi_dot_F = xi[vel_idx][0] * Force_density[0] + xi[vel_idx][1] * Force_density[1]
    u_dot_F = vel[0] * Force_density[0] + vel[1] * Force_density[1]
    xi_dot_u = xi[vel_idx][0] * vel[0] + xi[vel_idx][1] * vel[1]
    return w[vel_idx] * (inverse_cs2 * (xi_dot_F - u_dot_F) + inverse_cs4 * xi_dot_u * xi_dot_F)


for idx in range(Q):
    if comm.rank == 0:
        print("idx = ", idx, flush=True)
    f_n.append(V.interpolate(fEquilInit(idx), name=f"f_n{idx}"))

# ---------------------------------------------------------------------
# Diagnostics helpers (rank-local; only used gated by n > 1000)
# ---------------------------------------------------------------------
def trackCL(phi_n, dof_coordinates, h):
    values = np.asarray(phi_n.as_numpy).reshape(-1)
    coords = np.asarray(dof_coordinates)

    mask = coords[:, 1] < 4 * h
    mask &= (values > -0.2) & (values < 0.2)
    candidates = coords[mask]

    if len(candidates) < 2:
        raise ValueError("Not enough interface points to estimate the contact line.")

    min_x = candidates[:, 0].min()
    keep_rightCL = candidates[:, 0] > min_x + 7 * h
    max_x = candidates[:, 0].max()
    keep_leftCL = candidates[:, 0] < max_x - 7 * h

    candidates_rightCL = candidates[keep_rightCL]
    candidates_leftCL = candidates[keep_leftCL]

    left_x = candidates_leftCL[np.argmin(candidates_leftCL[:, 0]), 0]
    right_x = candidates_rightCL[np.argmax(candidates_rightCL[:, 0]), 0]
    return left_x, right_x


def computeContactAngle(c_n, h, mesh):
    angles = []
    n_vec = np.array([0.0, -1.0])
    local_c = c_n.localFunction()

    for element in mesh.elements:
        geometry = element.geometry
        midpoint = np.asarray(geometry.center, dtype=float)
        local_point = geometry.toLocal(midpoint.tolist())

        local_c.bind(element)
        try:
            value = float(np.asarray(local_c(local_point)).reshape(-1)[0])
            if midpoint[1] >= 2 * h:
                continue
            if not (-0.5 < value < 0.5):
                continue
            grad_c = np.asarray(local_c.jacobian(local_point)).reshape(-1)
            grad_norm = np.linalg.norm(grad_c)
            if grad_norm < 1e-14:
                continue
            cos_theta = np.dot(grad_c, n_vec) / grad_norm
            angles.append(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
        finally:
            local_c.unbind()

    if not angles:
        raise ValueError("No valid interfacial points found near the substrate.")
    return np.degrees(np.mean(angles))


def wall_dofs():
    y = V.interpolate(x[1], name="y_dof").as_numpy
    bottom = np.where(np.abs(y) < 1e-8)[0]
    top = np.where(np.abs(L_y - y) < 1e-8)[0]
    return bottom, top


# ---------------------------------------------------------------------
# Forms and operators
# ---------------------------------------------------------------------
density = getDens(f_n)
velocity_n = getVel(f_n, forceDensity)
velStar_n = getVel(f_star, forceDensity)
velSquared = ufl.inner(velocity_n, velocity_n)

lin_form_AC = (
    - dt * v * ufl.dot(velocity_n, ufl.grad(phi_n)) * ufl.dx
    - dt * M_tilde * v * mu_n * ufl.dx
    - (beta_mass_diff / dt) * mass_diff
      * ufl.sqrt(ufl.dot(ufl.grad(phi_n), ufl.grad(phi_n))) * v * ufl.dx
    - 0.5 * dt**2 * ufl.dot(velocity_n, ufl.grad(v))
      * ufl.dot(velocity_n, ufl.grad(phi_n)) * ufl.dx)

lin_form_mu = (
    A * phi_n * (phi_n**2 - 1) * v * ufl.dx
    + kappa * ufl.dot(ufl.grad(phi_n), ufl.grad(v)) * ufl.dx
    + kappa / (np.sqrt(2) * interfaceThickness) * np.cos(theta)
      * (phi_n**2 - 1) * v * ufl.ds(3))

collisionOps, streamingOps = [], []
rhsVecCollision, rhsVecStreaming = [], []

for idx in range(Q):
    double_dot_product_term = (
        -0.5 * dt**2 * ufl.inner(xi[idx], ufl.grad(f_star[idx]))
        * ufl.inner(xi[idx], ufl.grad(v)) * ufl.dx)

    dot_product_force_term = (
        0.5 * dt**2 * ufl.inner(xi[idx], ufl.grad(v))
        * body_Force(velStar_n, idx, forceDensity) * ufl.dx)

    lin_form_idx = (
        - dt * v * ufl.inner(xi[idx], ufl.grad(f_star[idx])) * ufl.dx
        + dt * v * body_Force(velStar_n, idx, forceDensity) * ufl.dx
        + double_dot_product_term
        + dot_product_force_term)

    f_eq_idx = f_equil(idx, velocity_n, density, velSquared)
    lin_form_coll = (- dt / tau * (f_n[idx] - f_eq_idx)) * v * ufl.dx

    collisionOps.append(dune.fem.operator.galerkin(lin_form_coll - f_trial * v * ufl.dx))
    streamingOps.append(dune.fem.operator.galerkin(lin_form_idx - f_trial * v * ufl.dx))
    rhsVecCollision.append(V.zero.copy())
    rhsVecStreaming.append(V.zero.copy())

phi_form_op = dune.fem.operator.galerkin(lin_form_AC - f_trial * v * ufl.dx)
mu_form_op = dune.fem.operator.galerkin(lin_form_mu - f_trial * v * ufl.dx)
rhs_AC_fn = V.zero.copy()
rhs_Mu_fn = V.zero.copy()
zeroFn = V.zero.copy()

# Lumped mass computed through the SAME operator path as the right-hand sides
# (applies  int v dx  to the zero function), so it gets the same treatment at
# partition borders as the rhs vectors do.
lump_op = dune.fem.operator.galerkin(1.0 * v * ufl.dx - f_trial * v * ufl.dx)


def lumped_mass():
    out = V.zero.copy()
    lump_op(zeroFn, out)
    return out.as_numpy.copy()


def report_mass(label):
    local = float(mass_diag.sum())
    total = comm.sum(local)
    print(f"[{label}] rank {comm.rank}: local sum(mass_diag) = {local:.6f}", flush=True)
    if comm.rank == 0:
        print(f"[{label}] total = {total:.6f} (domain area = {L_x * L_y}; "
              f"> area means shared border DoFs are accumulated)", flush=True)


# Optional border synchronisation after NumPy writes
SYNC_AVAILABLE = USE_SYNC and hasattr(phi_n, "communicate")
if comm.rank == 0:
    print("DoF communicate() available:", hasattr(phi_n, "communicate"),
          "| using it:", SYNC_AVAILABLE, flush=True)


def sync(fns):
    if SYNC_AVAILABLE:
        for f in fns:
            f.communicate()


# ---------------------------------------------------------------------
# Setup before time loop
# ---------------------------------------------------------------------
mass_diag = lumped_mass()
if np.any(mass_diag <= 0):
    raise ValueError("Lumped mass contains non-positive entries")
report_mass("init")

f_n_arrays = [f.as_numpy for f in f_n]
f_star_arrays = [f.as_numpy for f in f_star]
bottomDoFs, topDoFs = wall_dofs()

vtk = mesh.sequencedVTK(os.path.join(outDirName, "solution"), pointdata=[phi_n])

# Everything transferred on adaptation / load balancing
def transfer_list():
    return (f_n + f_star + rhsVecCollision + rhsVecStreaming
            + [phi_n, mu_n, zeroFn, rhs_AC_fn, rhs_Mu_fn])


# ---------------------------------------------------------------------
# Time stepping
# ---------------------------------------------------------------------
if comm.rank == 0:
    print("about to start time-stepping\n\n", flush=True)

for n in range(numSteps):

    phi_form_op(zeroFn, rhs_AC_fn)
    mu_form_op(zeroFn, rhs_Mu_fn)

    # Collision
    for idx in range(Q):
        collisionOps[idx](zeroFn, rhsVecCollision[idx])
        f_star_arrays[idx][:] = rhsVecCollision[idx].as_numpy / mass_diag + f_n_arrays[idx]
    sync(f_star)

    # Streaming
    for idx in range(Q):
        streamingOps[idx](zeroFn, rhsVecStreaming[idx])
        f_n_arrays[idx][:] = rhsVecStreaming[idx].as_numpy / mass_diag + f_star_arrays[idx]

    # Bottom wall: bounce-back
    for idx in (2, 5, 6):
        f_n_arrays[idx][bottomDoFs] = f_star_arrays[opp_idx[idx]][bottomDoFs]
    # Top wall
    for idx in (4, 7, 8):
        f_n_arrays[idx][topDoFs] = f_star_arrays[opp_idx[idx]][topDoFs]
    sync(f_n)

    # Phase field and chemical potential
    phi_nP1.as_numpy[:] = rhs_AC_fn.as_numpy / mass_diag + phi_n.as_numpy
    mu_n.as_numpy[:] = rhs_Mu_fn.as_numpy / mass_diag
    sync([phi_nP1, mu_n])

    phi_n.assign(phi_nP1)

    mass_n = dune.fem.integrate((phi_n + 1) / 2)
    if not np.isfinite(mass_n):
        raise RuntimeError(f"Non-finite mass at step {n}: {mass_n}")
    mass_diff.assign(mass_n - mass_init)

    # Global diagnostics (same on every rank)
    if n < DIAG_STEPS or n % OUT_EVERY == 0:
        pmax = comm.max(float(np.abs(phi_n.as_numpy).max()))
        e2 = dune.fem.integrate(phi_n**2)
        if comm.rank == 0:
            print(f"n = {n:6d}  max|phi| = {pmax:.12f}  int phi^2 = {e2:.12f}  "
                  f"mass = {mass_n:.12f}", flush=True)

    # Adaptation + load balancing
    if n % ADAPT_EVERY == 0:
        marker = dune.fem.GridMarker(
            1 - phi_n**2,
            refineTolerance=0.6,
            coarsenTolerance=0.2,
            minLevel=0,
            maxLevel=5)

        dune.fem.gridAdapt(marker, transfer_list())
        dune.fem.loadBalance(transfer_list())

        # DoF numbering has changed: refresh everything that depends on it
        f_n_arrays = [f.as_numpy for f in f_n]
        f_star_arrays = [f.as_numpy for f in f_star]
        mass_diag = lumped_mass()
        if np.any(mass_diag <= 0):
            raise ValueError("Lumped mass contains non-positive entries")
        bottomDoFs, topDoFs = wall_dofs()

        if n < DIAG_STEPS or n % (10 * ADAPT_EVERY) == 0:
            report_mass(f"adapt n={n}")

    # Output
    if n % OUT_EVERY == 0:
        if comm.rank == 0:
            print("n = ", n, "writing to file", flush=True)

        if n > 1000:
            theta_avg = computeContactAngle(phi_n, h_min, mesh)
            xh = V.interpolate(x[0], name="x")
            yh = V.interpolate(x[1], name="y")
            dof_coords = np.column_stack([xh.as_numpy, yh.as_numpy])
            left_cl, right_cl = trackCL(phi_n, dof_coords, h_min)
            if comm.rank == 0:
                print("theta = ", theta_avg, flush=True)
                print("left_cl = ", left_cl, " right_cl = ", right_cl, flush=True)

        vel_expr = getVel(f_n, forceDensity)
        vel3.interpolate(ufl.as_vector([vel_expr[0], vel_expr[1], 0]))
        vel_mag = np.linalg.norm(vel3.as_numpy.reshape(-1, 3), axis=1)
        vmax = comm.max(float(vel_mag.max()))     # global, not rank-local

        if comm.rank == 0:
            print("Max speed (global):", vmax, flush=True)
            print("Max KE:", vmax**2, flush=True)
            if n > 1000:
                log_file.write(f"{n:15d}{right_cl:15.3f}{vmax:15.8g}{theta_avg:15.2f}\n")
                log_file.flush()

        vtk()