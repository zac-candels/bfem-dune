import os
os.environ["DUNE_LOG_LEVEL"] = "ERROR"
import numpy as np
import gmsh
import meshio
from mpi4py import MPI
from dune.common import comm
from dune.ufl import Constant, DirichletBC
import dune
import dune.alugrid
from dune.alugrid import aluConformGrid
from dune.fem.view import adaptiveLeafGridView
import ufl
from collections import defaultdict
import scipy

mpi_comm = MPI.COMM_WORLD

# ---------------- Parameters ----------------
R0 = 2.5
L_x, L_y = 4*R0, 6*R0               # 2D domain: no z direction
radius = R0                          # circle radius
xc, yc = L_x/2, L_y/2 + 1.5*R0       # droplet centre

A = 0.5
kappa = 0.02
interfaceThickness = np.sqrt(kappa/A)
M_tilde = 10
theta_deg = 30
theta = theta_deg*np.pi/180

# D2Q9 lattice (ordering chosen so that the opposite of odd idx is idx+1)
Q = 9
xi_values = [
    (0.0,  0.0),
    (1.0,  0.0), (-1.0,  0.0),
    (0.0,  1.0), (0.0, -1.0),
    (1.0,  1.0), (-1.0, -1.0),
    (1.0, -1.0), (-1.0,  1.0),
]

rho_init = 1.0
nu = 1/3
c_s = np.sqrt(1/3)
tau = 0.1


# ---------------- Mesh generation ----------------
def create_periodic_circle_mesh(L_x, L_y, radius,
                                h_fine, h_coarse,
                                periodic_x=True,
                                filename="periodic_circle.msh"):
    """
    Rectangle [0,L_x]x[y_min,L_y] with a circular hole
    (2D analogue of the box-with-sphere mesh).

    Physical groups:
        surface 1 : fluid
        curves    : 2 bottom (y=y_min), 3 top (y=L_y), 4 left (x=0),
                    5 right (x=L_x), 6 circle
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.model.add("periodic_domain_with_circle")

    occ = gmsh.model.occ
    cx, cy = L_x/2, L_y/2 - 0.2*R0

    y_min = 0.0
    rect = occ.addRectangle(0, y_min, 0, L_x, L_y - y_min)
    disk = occ.addDisk(cx, cy, 0, radius, radius)
    cut, _ = occ.cut([(2, rect)], [(2, disk)])
    occ.synchronize()
    surface = cut[0][1]

    # Classify boundary curves by bounding box
    tol = 1e-6

    faces = {
        "left": [],
        "right": [],
        "bottom": [],    # the circle may split the bottom edge into pieces
        "top": [],
        "circle": [],
    }

    for dim, c in gmsh.model.getBoundary([(2, surface)], oriented=False):
        xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(1, c)

        if abs(xmax - xmin) < tol:                 # vertical line
            if abs(xmin) < tol:
                faces["left"].append(c)
            elif abs(xmin - L_x) < tol:
                faces["right"].append(c)

        elif abs(ymax - ymin) < tol:               # horizontal line
            if abs(ymin - y_min) < tol:
                faces["bottom"].append(c)
            elif abs(ymin - L_y) < tol:
                faces["top"].append(c)

        else:                                      # curved
            faces["circle"].append(c)

    for name, curves in faces.items():
        if len(curves) == 0:
            raise RuntimeError(f"Could not identify boundary curve: {name}")
    if len(faces["left"]) != 1 or len(faces["right"]) != 1:
        raise RuntimeError("Expected exactly one left and one right curve")

    # Exact periodicity in x (slave curve, master curve, master -> slave map)
    if periodic_x:
        gmsh.model.mesh.setPeriodic(
            1, faces["right"], faces["left"],
            [1, 0, 0, L_x, 0, 1, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 1]
        )

    # Physical groups
    gmsh.model.addPhysicalGroup(2, [surface], tag=1)
    gmsh.model.addPhysicalGroup(1, faces["bottom"], tag=2)
    gmsh.model.addPhysicalGroup(1, faces["top"], tag=3)
    gmsh.model.addPhysicalGroup(1, faces["left"], tag=4)
    gmsh.model.addPhysicalGroup(1, faces["right"], tag=5)
    gmsh.model.addPhysicalGroup(1, faces["circle"], tag=6)

    # Refinement near the circle
    field = gmsh.model.mesh.field
    dist = field.add("Distance")
    field.setNumbers(dist, "CurvesList", faces["circle"])
    field.setNumber(dist, "Sampling", 200)

    thr = field.add("Threshold")
    field.setNumber(thr, "InField", dist)
    field.setNumber(thr, "SizeMin", h_fine)
    field.setNumber(thr, "SizeMax", h_coarse)
    field.setNumber(thr, "DistMin", 0.5*radius)
    field.setNumber(thr, "DistMax", 1.2*radius)
    field.setAsBackgroundMesh(thr)
    
    # inside create_periodic_sphere_mesh, after the existing Threshold `thr`
    f_drop = field.add("MathEval")
    field.setString(f_drop, "F",
        f"Abs(Sqrt((x-{xc})^2+(y-{yc})^2)-{R0})")
    
    h_interface = 0.3
    thr_drop = field.add("Threshold")
    field.setNumber(thr_drop, "InField", f_drop)
    field.setNumber(thr_drop, "SizeMin", h_interface)   # ~0.5 for maxLevel=7
    field.setNumber(thr_drop, "SizeMax", h_coarse)
    field.setNumber(thr_drop, "DistMin", 1.0)           # band you want at full resolution
    field.setNumber(thr_drop, "DistMax", 3.0)
    
    bg = field.add("Min")
    field.setNumbers(bg, "FieldsList", [thr, thr_drop])
    field.setAsBackgroundMesh(bg)                       # replace the setAsBackgroundMesh(thr)

    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    gmsh.model.mesh.generate(2)
    gmsh.write(filename)
    gmsh.finalize()
    return filename


mesh_file = "periodic_circle.msh"

h_fine = 0.2
if comm.rank == 0:
    print("\nSTEP 1: Starting mesh generation", flush=True)
    # 2D allows much finer meshes than 3D.
    create_periodic_circle_mesh(L_x, L_y, radius,
                                h_fine=h_fine*radius, h_coarse=1.5*radius,
                                periodic_x=False,
                                filename=mesh_file)
    print("\nSTEP 2: Mesh generation completed", flush=True)

MPI.COMM_WORLD.Barrier()

# ---------------- Build the dune grid ----------------
if mpi_comm.rank == 0:
    m = meshio.read(mesh_file)
    points = m.points[:, :2]
    tris = m.cells_dict["triangle"]

    # keep only vertices used by a triangle, and renumber
    used, inverse = np.unique(tris, return_inverse=True)
    vertices = points[used]
    simplices = inverse.reshape(tris.shape).astype(np.int64)
else:
    vertices = np.zeros((0, 2))
    simplices = np.zeros((0, 3), dtype=np.int64)

grid = aluConformGrid({"vertices": vertices, "simplices": simplices}, dimgrid=2)
mesh = adaptiveLeafGridView(grid)
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
dt_mult_fac = 0.01
dt = dt_mult_fac*h_min
dt_c = dune.ufl.Constant(dt, name="dt")
beta_mass_diff = 0.01*dt_c
numSteps = int(Tfinal/dt)


# ---------------- Output ----------------
WORKDIR = os.getcwd()
outDirName = os.path.join(WORKDIR, f"coarsened")
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
    - dt*M_tilde*v*mu_n*ufl.dx - (beta_mass_diff/dt)*mass_diff*ufl.sqrt( ufl.dot(ufl.grad(phi_n), ufl.grad(phi_n)) )*v*ufl.dx\
        - 0.5*dt**2 * ufl.dot(velocity_n, ufl.grad(v)) * ufl.dot(velocity_n, ufl.grad(phi_n)) *ufl.dx

# Wetting acts on the wall only: exclude the top and the x = 0, x = L_x
# boundaries
wall_ind = ufl.conditional(
    ufl.And(x[1] < L_y - 1e-6,
            ufl.And(x[0] > 1e-6, x[0] < L_x - 1e-6)),
    1.0, 0.0)

lin_form_mu = A*phi_n*(phi_n**2 - 1)*v*ufl.dx \
    + kappa*ufl.dot(ufl.grad(phi_n), ufl.grad(v))*ufl.dx \
    + wall_ind*kappa/(np.sqrt(2)*interfaceThickness)*np.cos(theta) \
      *(phi_n**2 - 1)*v*ufl.ds

wall_len  = dune.fem.assemble(f_trial*v*wall_ind*ufl.ds).as_numpy.sum()
total_len = dune.fem.assemble(f_trial*v*ufl.ds).as_numpy.sum()

if comm.rank == 0:
    print("wall:", wall_len, " rest:", total_len - wall_len, flush=True)
    print("finished making AC forms \n\n", flush=True)

opp_idx = {0: 0}
for idx in range(1, Q):
    opp_idx[idx] = idx + 1 if idx % 2 == 1 else idx - 1

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


def build_solvers():

    lumpOp = dune.fem.operator.galerkin(
        1.0 * v * ufl.dx - f_trial * v * ufl.dx
    )

    lumpVec = V.zero.copy()
    lumpOp(zeroFn, lumpVec)

    mass_diag_local = lumpVec.as_numpy.copy()

    return mass_diag_local

mass_diag_local = build_solvers()

bottomDoFs, topDoFs = wall_dofs()

#%% Refine mesh once before time-stepping commences

def adapt_mesh():
    global f_star_arrays, f_n_arrays, mass_diag_local, bouncebackDofs
    marker = dune.fem.GridMarker(1 - phi_n**2, refineTolerance=0.1,
                                  coarsenTolerance=0, minLevel=0, maxLevel=4)
    dune.fem.gridAdapt(marker, f_n + f_star + rhsVecCollision + rhsVecStreaming
                        + [phi_n, mu_n, zeroFn, rhs_AC_fn, rhs_Mu_fn])
    f_star_arrays = [f.as_numpy for f in f_star]
    f_n_arrays    = [f.as_numpy for f in f_n]
    mass_diag_local = build_solvers()
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
        b_local = rhsVecCollision[idx].as_numpy

        f_star_local = b_local / mass_diag_local
        
        f_star_arrays[idx][:] = f_star_local + f_n_arrays[idx][:]
        
    
    #print("did collision \n\n")
        
    # Stream all populations
    for idx in range(Q):
        streamingOps[idx](zeroFn, rhsVecStreaming[idx])
        b_local = rhsVecStreaming[idx].as_numpy

        f_n_local = b_local / mass_diag_local
        
        f_n_arrays[idx][:] = f_n_local + f_star_arrays[idx][:]
    

    for idx in range(1, Q):
        dofs = bouncebackDofs[idx]
        f_n_arrays[idx][dofs] = (
            f_star_arrays[opp_idx[idx]][dofs])
            
            
          
    b_phi_local = rhs_AC_fn.as_numpy

    phi_local = b_phi_local / mass_diag_local
    
    phi_nP1.as_numpy[:] = phi_local + phi_n.as_numpy[:]
    
    phi_n.assign(phi_nP1)

    b_mu_local = rhs_Mu_fn.as_numpy

    mu_local = b_mu_local / mass_diag_local
    
    mu_n.as_numpy[:] = mu_local
    
    
    mass_n = dune.fem.integrate((phi_n + 1) / 2)
    
    #print("mass_n = ", mass_n)
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
        mass_diag_local = build_solvers()

        bouncebackDofs = build_bounceback_dofs(mesh, V, xi_values, Q)

    if n % 1000 == 0:

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