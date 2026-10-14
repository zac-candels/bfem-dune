import os
import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
import dune
import dune.alugrid
import ufl
from dune.ufl import Constant

# ---------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------
R0 = 2.0
A = 0.5
kappa = 0.02
interfaceThickness = np.sqrt(kappa / A)
M_tilde = 10.0
theta_deg = 30.0
theta = np.deg2rad(theta_deg)

Q = 9
Tfinal = 1500.0
dt = 0.0005
beta_mass_diff = 0.1 * dt
numSteps = int(Tfinal / dt)

L_x, L_y = 16.0, 4.0
Nx, Ny = 40, 10
rho_init = 1.0
c_s = np.sqrt(1.0 / 3.0)
tau = 0.1

print("defined constants", flush=True)

outDirName = os.path.join(os.getcwd(), "newRefinement")
os.makedirs(outDirName, exist_ok=True)

xc, yc = L_x / 2.0, R0 - 0.6 * R0

print("about to make domain")
domain = dune.grid.cartesianDomain([0.0, 0.0], [L_x, L_y], [Nx, Ny])
mesh = dune.fem.view.adaptiveLeafGridView(dune.alugrid.aluConformGrid(domain))
print("created mesh")

h_min = np.inf
for element in mesh.elements:
    corners = np.asarray(element.geometry.corners)
    h_el = max(np.linalg.norm(a - b)
               for i, a in enumerate(corners)
               for b in corners[i + 1:])
    h_min = min(h_min, h_el)
h_min /= 2**3

w = np.array([4/9, 1/9, 1/9, 1/9, 1/9,
              1/36, 1/36, 1/36, 1/36])

xi = [
    Constant((0.0, 0.0)), Constant((1.0, 0.0)),
    Constant((0.0, 1.0)), Constant((-1.0, 0.0)),
    Constant((0.0, -1.0)), Constant((1.0, 1.0)),
    Constant((-1.0, 1.0)), Constant((-1.0, -1.0)),
    Constant((1.0, -1.0))
]
opp_idx = {0: 0, 1: 3, 2: 4, 3: 1, 4: 2, 5: 7, 6: 8, 7: 5, 8: 6}

print("created constants", flush=True)

# ---------------------------------------------------------------------
# Space and functions
# ---------------------------------------------------------------------
V = dune.fem.space.lagrange(mesh, order=1)
V_vec3 = dune.fem.space.lagrange(mesh, order=1, dimRange=3)
v = ufl.TestFunction(V)
f_trial = ufl.TrialFunction(V)
x = ufl.SpatialCoordinate(V)

print("created function spaces and trial/test functions", flush=True)

print("A: starting zeroFn interpolation", flush=True)
zeroFn = V.interpolate(0.0, name="zero")
print("B: zeroFn created", flush=True)

f_n = []
f_star = []

for i in range(Q):
    print(f"C: creating f_n[{i}]", flush=True)
    f_n.append(V.interpolate(0.0, name=f"f_n{i}"))

    print(f"D: creating f_star[{i}]", flush=True)
    f_star.append(V.interpolate(0.0, name=f"f_star{i}"))

print("E: all population functions created", flush=True)

print("F: building phase-field expression", flush=True)
phi_expr = -ufl.tanh(
    (ufl.sqrt((x[0] - xc)**2 + (x[1] - yc)**2) - R0)
    / (ufl.sqrt(2.0) * interfaceThickness)
)
print("G: phase-field expression built", flush=True)

print("H: interpolating phi_n", flush=True)
phi_n = V.interpolate(phi_expr, name="phi")
print("I: phi_n created", flush=True)

print("J: interpolating phi_nP1", flush=True)
phi_nP1 = V.interpolate(phi_expr, name="phi_next")
print("K: phi_nP1 created", flush=True)

print("L: interpolating mu_n", flush=True)
mu_n = V.interpolate(0.0, name="mu")
print("M: mu_n created", flush=True)

print("N: interpolating vel3", flush=True)
vel3 = V_vec3.interpolate(
    Constant((0.0, 0.0, 0.0)), name="vel"
)
print("O: vel3 created", flush=True)

print("initialized FE functions", flush=True)


def getDens(populations):
    return sum(populations)

def getVel(populations, force_density):
    density = sum(populations)
    momentum = populations[0] * xi[0]
    for i in range(1, Q):
        momentum += populations[i] * xi[i]
    return momentum / density + force_density * dt / (2.0 * density)

def fEquilInit(i):
    u0 = Constant((0.0, 0.0))
    cu = ufl.inner(xi[i], u0)
    return w[i] * rho_init * (
        1.0 + cu / c_s**2 + cu**2 / (2.0*c_s**4)
        - ufl.inner(u0, u0) / (2.0*c_s**2)
    )

def f_equil(i, velocity, density, velocity_squared):
    cu = ufl.inner(velocity, xi[i])
    return w[i] * density * (
        1.0 + 3.0*cu + 4.5*cu**2 - 1.5*velocity_squared
    )

def body_Force(vel, i, force_density):
    inv_cs2 = 1.0 / c_s**2
    inv_cs4 = 1.0 / c_s**4
    ciF = xi[i][0]*force_density[0] + xi[i][1]*force_density[1]
    uF = vel[0]*force_density[0] + vel[1]*force_density[1]
    ciu = xi[i][0]*vel[0] + xi[i][1]*vel[1]
    return w[i] * (inv_cs2*(ciF-uF) + inv_cs4*ciu*ciF)

for i in range(Q):
    f_n[i].interpolate(fEquilInit(i))

mass_init = dune.fem.integrate((phi_n + 1.0) / 2.0)
mass_diff = Constant(1e-9)
forceDensity = -phi_n * ufl.grad(mu_n)

print("about to start wall_dofs", flush=True)

# ---------------------------------------------------------------------
# Wall DOFs and periodic reduction
# ---------------------------------------------------------------------
def wall_dofs():
    yy = V.interpolate(x[1], name="y_dof").as_numpy.copy()
    return np.where(np.abs(yy) < 1e-8)[0], np.where(np.abs(yy-L_y) < 1e-8)[0]

def periodic_matrix():
    xx = V.interpolate(x[0], name="x_dof").as_numpy.copy()
    yy = V.interpolate(x[1], name="y_dof").as_numpy.copy()
    N = len(xx)
    xx[np.abs(xx-L_x) < 1e-10] = 0.0
    keys = np.column_stack((np.round(xx, 10), np.round(yy, 10)))
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    P = scipy.sparse.coo_matrix(
        (np.ones(N), (np.arange(N), inverse)),
        shape=(N, int(inverse.max())+1)
    ).tocsr()
    return P

# ---------------------------------------------------------------------
# Lumped mass matrix: row-sum lumping of the consistent FE mass matrix.
# The reduced lumped matrix is rebuilt after every grid adaptation.
# ---------------------------------------------------------------------
mass_form = f_trial * v * ufl.dx
def build_lumped_solver(P):
    M_consistent = scipy.sparse.csr_matrix(
        dune.fem.assemble(mass_form).as_numpy
    )
    lumped_diagonal = np.asarray(M_consistent.sum(axis=1)).reshape(-1)
    if np.any(lumped_diagonal <= 0):
        raise RuntimeError("Non-positive row-sum lumped mass entry found.")
    M_lumped = scipy.sparse.diags(lumped_diagonal, format="csr")
    M_reduced = (P.T @ M_lumped @ P).tocsc()
    solver = scipy.sparse.linalg.factorized(M_reduced)
    return lumped_diagonal, solver

def lumped_solve(rhs_full, P, solver):
    return np.asarray(P @ solver(np.asarray(P.T @ rhs_full).reshape(-1))).reshape(-1)

print("defined mass form", flush=True)

# ---------------------------------------------------------------------
# Weak right-hand sides. The mass terms are deliberately removed from
# these forms and replaced by M_L @ nodal_values during each step.
# ---------------------------------------------------------------------
density = getDens(f_n)
velocity_n = getVel(f_n, forceDensity)
velStar_n = getVel(f_star, forceDensity)
velSquared = ufl.inner(velocity_n, velocity_n)

# Allen-Cahn: phi mass term is handled with the lumped matrix.
lin_AC_nonmass = (
    -dt * v * ufl.dot(velocity_n, ufl.grad(phi_n)) * ufl.dx
    -dt * M_tilde * v * mu_n * ufl.dx
    -(beta_mass_diff/dt) * mass_diff
      * ufl.sqrt(ufl.dot(ufl.grad(phi_n), ufl.grad(phi_n))) * v * ufl.dx
    -0.5 * dt**2 * ufl.dot(velocity_n, ufl.grad(v))
      * ufl.dot(velocity_n, ufl.grad(phi_n)) * ufl.dx
)
lin_mu = (
    A * phi_n * (phi_n**2 - 1.0) * v * ufl.dx
    + kappa * ufl.dot(ufl.grad(phi_n), ufl.grad(v)) * ufl.dx
    + kappa/(np.sqrt(2.0)*interfaceThickness) * np.cos(theta)
      * (phi_n**2 - 1.0) * v * ufl.ds(3)
)

# Population RHS forms with the f_star mass term excluded.
stream_nonmass = []
collision_nonmass = []
for i in range(Q):
    force_term = body_Force(velStar_n, i, forceDensity)
    stream_nonmass.append(
        -dt * v * ufl.inner(xi[i], ufl.grad(f_star[i])) * ufl.dx
        +dt * v * force_term * ufl.dx
        -0.5 * dt**2 * ufl.inner(xi[i], ufl.grad(f_star[i]))
          * ufl.inner(xi[i], ufl.grad(v)) * ufl.dx
        +0.5 * dt**2 * ufl.inner(xi[i], ufl.grad(v))
          * force_term * ufl.dx
    )
    feq = f_equil(i, velocity_n, density, velSquared)
    collision_nonmass.append(
        (f_n[i] - dt/tau * (f_n[i] - feq)) * v * ufl.dx
    )

stream_ops = [
    dune.fem.operator.galerkin(stream_nonmass[i] - f_trial*v*ufl.dx)
    for i in range(Q)
]
AC_op = dune.fem.operator.galerkin(lin_AC_nonmass - f_trial*v*ufl.dx)
mu_op = dune.fem.operator.galerkin(lin_mu - f_trial*v*ufl.dx)

print("defined linear forms", flush=True)

# With the same lumped mass on both sides, the collision step is the
# nodal BGK relaxation itself; no linear solve is required for collision.
collision_targets = []
for i in range(Q):
    feq = f_equil(i, velocity_n, density, velSquared)
    collision_targets.append(
        f_n[i] - dt/tau * (f_n[i] - feq)
    )
collision_target_functions = [
    V.interpolate(0.0, name=f"collision_target{i}") for i in range(Q)
]

stream_rhs = [V.zero.copy() for _ in range(Q)]
rhs_AC = V.zero.copy()
rhs_mu = V.zero.copy()

# Keep references to current arrays only; refresh after grid adaptation.
def refresh_arrays():
    return ([f.as_numpy for f in f_n],
            [f.as_numpy for f in f_star])

P = periodic_matrix()
lumped_diag, lumped_solver = build_lumped_solver(P)
bottomDoFs, topDoFs = wall_dofs()
f_n_arrays, f_star_arrays = refresh_arrays()


def trackCL(phi, dof_coordinates, h):
    values = np.asarray(phi.as_numpy).reshape(-1)
    coords = np.asarray(dof_coordinates)
    mask = (coords[:, 1] < 4*h) & (values > -0.2) & (values < 0.2)
    candidates = coords[mask]
    if len(candidates) < 2:
        raise ValueError("Not enough interface points to estimate contact lines.")
    min_x, max_x = candidates[:, 0].min(), candidates[:, 0].max()
    left_candidates = candidates[candidates[:, 0] < max_x - 7*h]
    right_candidates = candidates[candidates[:, 0] > min_x + 7*h]
    if len(left_candidates) == 0 or len(right_candidates) == 0:
        raise ValueError("Could not separate left/right contact-line candidates.")
    return left_candidates[:, 0].min(), right_candidates[:, 0].max()

def computeContactAngle(phi, h, mesh_view):
    angles = []
    n_vec = np.array([0.0, -1.0])
    local_phi = phi.localFunction()
    for element in mesh_view.elements:
        geometry = element.geometry
        midpoint = np.asarray(geometry.center, dtype=float)
        if midpoint[1] >= 2*h:
            continue
        local_point = geometry.toLocal(midpoint.tolist())
        local_phi.bind(element)
        try:
            value = float(np.asarray(local_phi(local_point)).reshape(-1)[0])
            if not (-0.5 < value < 0.5):
                continue
            grad_phi = np.asarray(local_phi.jacobian(local_point)).reshape(-1)
            grad_norm = np.linalg.norm(grad_phi)
            if grad_norm < 1e-14:
                continue
            cosine = np.dot(grad_phi, n_vec) / grad_norm
            angles.append(np.arccos(np.clip(cosine, -1.0, 1.0)))
        finally:
            local_phi.unbind()
    if not angles:
        raise ValueError("No valid interfacial points found near the substrate.")
    return np.degrees(np.mean(angles))

# ---------------------------------------------------------------------
# Time loop
# ---------------------------------------------------------------------
print("Starting time-stepping", flush=True)
for n in range(numSteps):
    # Update expressions that depend on current fields through the operators.
    AC_op(zeroFn, rhs_AC)
    mu_op(zeroFn, rhs_mu)

    # Collision: M_L f_star = M_L [f_n - dt/tau (f_n - f_eq)].
    # Since M_L is diagonal and invertible, this is the nodal BGK update.
    for i in range(Q):
        collision_target_functions[i].interpolate(collision_targets[i])
        f_star_arrays[i][:] = collision_target_functions[i].as_numpy

    # Streaming/advection. Replace the f_star mass term by M_L f_star.
    for i in range(Q):
        stream_ops[i](zeroFn, stream_rhs[i])
        rhs = stream_rhs[i].as_numpy.copy()
        rhs += lumped_diag * f_star_arrays[i]
        f_new = lumped_solve(rhs, P, lumped_solver)

        # Enforce halfway bounce-back nodally for the outgoing wall links.
        if i in (2, 5, 6):
            f_new[bottomDoFs] = f_star_arrays[opp_idx[i]][bottomDoFs]
        elif i in (4, 7, 8):
            f_new[topDoFs] = f_star_arrays[opp_idx[i]][topDoFs]
        f_n_arrays[i][:] = f_new

    # Phase field: mass term M_L phi_n plus non-mass terms.
    rhs_phi = rhs_AC.as_numpy.copy() + lumped_diag * phi_n.as_numpy.copy()
    phi_nP1.as_numpy[:] = lumped_solve(rhs_phi, P, lumped_solver)

    # Chemical potential equation has no input mass term on the RHS.
    rhs_chemical = rhs_mu.as_numpy.copy()
    mu_n.as_numpy[:] = lumped_solve(rhs_chemical, P, lumped_solver)
    phi_n.assign(phi_nP1)

    mass_now = dune.fem.integrate((phi_n + 1.0) / 2.0)
    mass_diff.assign(mass_now - mass_init)

    if n % 10 == 0:
        marker = dune.fem.GridMarker(
            1.0 - phi_n**2,
            refineTolerance=0.6,
            coarsenTolerance=0.2,
            minLevel=0,
            maxLevel=5
        )
        dune.fem.gridAdapt(
            marker,
            f_n + f_star + collision_target_functions + stream_rhs
            + [phi_n, mu_n, zeroFn, rhs_AC, rhs_mu, phi_nP1]
        )
        P = periodic_matrix()
        lumped_diag, lumped_solver = build_lumped_solver(P)
        bottomDoFs, topDoFs = wall_dofs()
        f_n_arrays, f_star_arrays = refresh_arrays()

    if n % 2000 == 0:
        print(f"n = {n}, mass difference = {float(mass_diff)}")
        if n > 0:
            try:
                print("theta =", computeContactAngle(phi_n, h_min, mesh))
                xh = V.interpolate(x[0], name="x_contact").as_numpy.copy()
                yh = V.interpolate(x[1], name="y_contact").as_numpy.copy()
                dof_coords = np.column_stack((xh, yh))
                left_cl, right_cl = trackCL(phi_n, dof_coords, h_min)
                print("left_cl =", left_cl, "right_cl =", right_cl)
            except ValueError as err:
                print("Contact-line diagnostic skipped:", err)
        vel_expr = getVel(f_n, forceDensity)
        vel3.interpolate(ufl.as_vector([vel_expr[0], vel_expr[1], 0.0]))
        mesh.writeVTK(os.path.join(outDirName, f"vel_{n:06d}"), pointdata=[vel3])
        mesh.writeVTK(os.path.join(outDirName, f"phi_{n:06d}"), pointdata=[phi_n])

print("Finished.")
