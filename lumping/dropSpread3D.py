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
from collections import defaultdict



import ufl

R0 = 2
initDropDiam = 2*R0
A = 0.5
kappa = 0.02
interfaceThickness = np.sqrt(kappa/A)
M_tilde = 10
theta_deg = 30
theta = theta_deg * np.pi / 180

# D3Q19 lattice
Q = 19

xi_values = [
    (0.0,  0.0,  0.0),
    (1.0,  0.0,  0.0),
    (-1.0, 0.0,  0.0),
    (0.0,  1.0,  0.0),
    (0.0, -1.0,  0.0),
    (0.0,  0.0,  1.0),
    (0.0,  0.0, -1.0),
    (1.0,  1.0,  0.0),
    (-1.0, -1.0, 0.0),
    (1.0, -1.0,  0.0),
    (-1.0, 1.0,  0.0),
    (1.0,  0.0,  1.0),
    (-1.0, 0.0, -1.0),
    (1.0,  0.0, -1.0),
    (-1.0, 0.0,  1.0),
    (0.0,  1.0,  1.0),
    (0.0, -1.0, -1.0),
    (0.0,  1.0, -1.0),
    (0.0, -1.0,  1.0),
]

L_x = 16
L_y = 4
L_z = 8
Nx = 40
Ny = 10
Nz = 20

#%% Prepare output
WORKDIR = os.getcwd()
outDirName = os.path.join(WORKDIR, f"3d")
os.makedirs(outDirName, exist_ok=True)


log_file = open(outDirName + "/simulation_log.txt", "w")
log_file.write(f"{'n' :>15}"
               f"{'r_cl':>15}"
               f"{'max ||u||':>15}"
               f"{'theta':>15}\n")
log_file.flush()

xc, yc, zc = L_x/2, R0 - 0.6*R0, L_z/2

rho_init = 1.0
nu = 1/3
c_s = np.sqrt(1/3)
tau = 0.1

def create_sinusoidal_mesh(
    L_x,
    L_y,
    L_z,
    surface_amplitude=0.0,
    surface_freq=None,
    n_surface=200,
    lc=0.1,
    filename="sinusoidal_wall.msh"):
    """
    Create a 3D Gmsh mesh with a sinusoidal bottom boundary.

    The sinusoidal profile in the (x, y) plane is extruded along z.

    Parameters
    ----------
    L_x : float
        Domain length.
    L_y : float
        Domain height.
    L_z : float
        Domain depth.
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

        # Enclosed 2D profile in the (x, y) plane at z = 0
        loop = geo.addCurveLoop(wall_lines + [right_line, top_line, left_line])

        surface = geo.addPlaneSurface([loop])

        # Extrude the profile along z to get the 3D fluid domain
        # extruded = [top surface (z = L_z), volume, lateral surfaces ...]
        extruded = geo.extrude([(2, surface)], 0.0, 0.0, L_z)

        geo.synchronize()

        top_z_surface = extruded[0][1]
        volume = extruded[1][1]
        lateral = [tag for dim, tag in extruded[2:]]

        # Classify the lateral surfaces by their bounding boxes
        wall_surfaces = []
        left_surface = None
        right_surface = None
        top_surfaces = []

        for s in lateral:
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, s)

            if abs(xmax - xmin) < 1e-6:
                # x = 0 or x = L_x
                if abs(xmin) < 1e-6:
                    left_surface = s
                else:
                    right_surface = s
            elif abs(ymax - ymin) < 1e-6 and abs(ymin - L_y) < 1e-6:
                top_surfaces.append(s)
            else:
                wall_surfaces.append(s)

        # Periodicity in x and z (the right / z = L_z surfaces are copies
        # of the left / z = 0 surfaces), so that the periodic DOFs match
        gmsh.model.mesh.setPeriodic(
            2, [right_surface], [left_surface],
            [1, 0, 0, L_x,
             0, 1, 0, 0,
             0, 0, 1, 0,
             0, 0, 0, 1])

        gmsh.model.mesh.setPeriodic(
            2, [top_z_surface], [surface],
            [1, 0, 0, 0,
             0, 1, 0, 0,
             0, 0, 1, L_z,
             0, 0, 0, 1])

        # Physical IDs used for boundary conditions
        gmsh.model.addPhysicalGroup(2, wall_surfaces, 3)
        
        gmsh.model.setPhysicalName(2, 3, "wall")

        gmsh.model.addPhysicalGroup(2, [right_surface, left_surface], 4)
        
        gmsh.model.setPhysicalName(2, 4, "sides")

        gmsh.model.addPhysicalGroup(2, top_surfaces, 5)
        
        gmsh.model.setPhysicalName(2, 5, "top")

        gmsh.model.addPhysicalGroup(2, [surface, top_z_surface], 6)
        
        gmsh.model.setPhysicalName(2, 6, "zfaces")

        gmsh.model.addPhysicalGroup(3, [volume], 1)
        
        gmsh.model.setPhysicalName(3, 1, "fluid")

        # Generate and save the 3D mesh
        gmsh.model.mesh.generate(3)
        gmsh.write(filename)

    finally:
        gmsh.finalize()

    return filename


#%% Generate the Gmsh mesh

print("\nSTEP 1: Starting mesh generation", flush=True)

mesh_file = create_sinusoidal_mesh(
    L_x=L_x,
    L_y=L_y,
    L_z=L_z,
    surface_amplitude=1.0,
    surface_freq=L_x / (8.0 * np.pi),
    n_surface=40,
    lc=0.3)

print("\nSTEP 2: Mesh generation completed", flush=True)

import meshio
from dune.alugrid import aluConformGrid
from dune.fem.view import adaptiveLeafGridView

msh = meshio.read(mesh_file)

vertices = np.asarray(msh.points, dtype=float)
simplices = np.asarray(msh.cells_dict["tetra"], dtype=int)

grid_description = {"vertices": vertices, "simplices": simplices}

grid = aluConformGrid(grid_description, dimgrid=3)
mesh = adaptiveLeafGridView(grid)

#from dune.fem.function import boundaryFunction
#boundaryFunction(grid).plot()

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
dt = 0.0001*h_min
dt_c = dune.ufl.Constant(dt, name="dt")
beta_mass_diff = 0.01*dt_c
numSteps = int(Tfinal/dt)
# Corresponding weights
w = np.array([1/3]
             + [1/18]*6
             + [1/36]*12)

xi = [dune.ufl.Constant(c) for c in xi_values]


#%% Create finite-element space

print("\nSTEP 8: Creating finite-element space", flush=True)

V = dune.fem.space.lagrange(mesh, order=1)

x = ufl.SpatialCoordinate(V)

print("\nSTEP 9: Finite-element space created successfully", flush=True)

#%% Compute boundary DOF normals

def build_bounceback_dofs(mesh, V, xi_values, Q, tol=1e-8):
    """
    Build bounce-back DOF sets for the lower wall.

    DOF indices are taken from DUNE's Lagrange space, so they
    correspond to the ordering in f_n[idx].as_numpy.
    """

    # Coordinates in DUNE's actual DOF ordering
    x_dof = V.interpolate(x[0], name="x_dof_bb").as_numpy.copy()
    y_dof = V.interpolate(x[1], name="y_dof_bb").as_numpy.copy()
    z_dof = V.interpolate(x[2], name="z_dof_bb").as_numpy.copy()

    dof_by_key = {tuple(np.round(np.array([xx, yy, zz]) / tol).astype(np.int64)): i
        for i, (xx, yy, zz) in enumerate(zip(x_dof, y_dof, z_dof))}

    # Accumulate boundary normals at each DUNE DOF
    normals_sum = defaultdict(lambda: np.zeros(3))

    for element in mesh.elements:
        for intersection in mesh.intersections(element):

            if not intersection.boundary:
                continue

            normal = np.asarray(
                intersection.unitOuterNormal([1/3, 1/3]),
                dtype=float
            )

            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue

            normal /= norm

            # Keep the wall-facing boundary only (skips the x = 0, x = L_x
            # and z = 0, z = L_z faces, whose normals have no y component).
            # The curved wall's outward normal has negative y component.
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

    for dof, normal in boundary_normals.items():
        for idx in range(1, Q):
            if np.dot(normal, xi_values[idx]) < 0:
                bouncebackDofs[idx].append(dof)

    bouncebackDofs = {
        idx: np.asarray(dofs, dtype=int)
        for idx, dofs in bouncebackDofs.items()}

    return bouncebackDofs

#%% Calculate boundary vertex normals

bouncebackDofs = build_bounceback_dofs(mesh, V, xi_values, Q)



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

phiInitExpr = -ufl.tanh( (ufl.sqrt(pow(x[0]-xc,2) + pow(x[1]-yc,2) + pow(x[2]-zc,2)) - R0)\
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
    
    forceDensity = dune.ufl.Constant((0.0, 0.0, 0.0))
    
    vel_0 = -dune.ufl.Constant( (0.0, 0.0, 0.0) )
    
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
        + xi[vel_idx][1]*Force_density[1]\
        + xi[vel_idx][2]*Force_density[2]

    u_dot_prod_F = vel[0]*Force_density[0] + vel[1]*Force_density[1]\
        + vel[2]*Force_density[2]

    xi_dot_u = xi[vel_idx][0]*vel[0] + xi[vel_idx][1]*vel[1]\
        + xi[vel_idx][2]*vel[2]

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
    dof_coordinates : array-like, shape (N, 3)
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
    n_vec = np.array([0.0, -1.0, 0.0])

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

lin_form_AC =  - dt*v*ufl.dot(velocity_n, ufl.grad(phi_n))*ufl.dx\
    - dt*M_tilde*v*mu_n*ufl.dx - (beta_mass_diff/dt)*mass_diff*ufl.sqrt( ufl.dot(ufl.grad(phi_n), ufl.grad(phi_n)) )*v*ufl.dx\
        - 0.5*dt**2 * ufl.dot(velocity_n, ufl.grad(v)) * ufl.dot(velocity_n, ufl.grad(phi_n)) *ufl.dx

# Wetting acts on the wall only: exclude the top, the x = 0, x = L_x
# faces and the z = 0, z = L_z faces
wall_ind = ufl.conditional(
    ufl.And(x[1] < L_y - 1e-6,
            ufl.And(x[0] > 1e-6,
                    ufl.And(x[0] < L_x - 1e-6,
                            ufl.And(x[2] > 1e-6, x[2] < L_z - 1e-6)))),
    1.0, 0.0)

lin_form_mu = A*phi_n*(phi_n**2 - 1)*v*ufl.dx \
    + kappa*ufl.dot(ufl.grad(phi_n), ufl.grad(v))*ufl.dx \
    + wall_ind*kappa/(np.sqrt(2)*interfaceThickness)*np.cos(theta) \
      *(phi_n**2 - 1)*v*ufl.ds

wall_len  = dune.fem.assemble(f_trial*v*wall_ind*ufl.ds).as_numpy.sum()
total_len = dune.fem.assemble(f_trial*v*ufl.ds).as_numpy.sum()
print("wall:", wall_len, " rest:", total_len - wall_len)   # expect wall ~17.9*L_z

print("finished making AC forms \n\n")
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
    
print("finished making LB forms\n\n")

# Assemble matrices for first step
sysMatStream = []
sysMatColl = []
for idx in range(Q):
    #sysMatStream.append(dune.fem.assemble(bilinFormsStream[idx]))
    if xi_values[idx][1] == 0:
        sysMatStream.append(dune.fem.assemble(bilinFormsStream[idx]))
    elif xi_values[idx][1] > 0:
        sysMatStream.append(dune.fem.assemble([bilinFormsStream[idx], dbcBottom]))
    else:
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
    x_dof = V.interpolate(x[0], name="x_dof").as_numpy.copy()

    y_dof = V.interpolate(x[1], name="y_dof").as_numpy.copy()

    z_dof = V.interpolate(x[2], name="z_dof").as_numpy.copy()

    N = len(x_dof)

    # -----------------------------------------------------
    # x and z periodicity
    # -----------------------------------------------------

    x_periodic = x_dof.copy()

    x_periodic[np.abs(x_periodic - L_x) < 1e-10] = 0.0

    z_periodic = z_dof.copy()

    z_periodic[np.abs(z_periodic - L_z) < 1e-10] = 0.0

    # -----------------------------------------------------
    # y is NOT periodic
    # -----------------------------------------------------

    keys = np.column_stack((np.round(x_periodic, 10),
                            np.round(y_dof, 10),
                            np.round(z_periodic, 10)))

    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)

    nPeriodic = len(unique_keys)

    rows = np.arange(N)
    cols = inverse
    data = np.ones(N)

    P = scipy.sparse.coo_matrix((data, (rows, cols)),
                                shape=(N, nPeriodic)).tocsr()
    
    sysMatStream = []
    sysMatColl = []

    massMatConsistent = scipy.sparse.csr_matrix(
    dune.fem.operator.galerkin(f_trial * v * ufl.dx).linear().as_numpy)

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
    
    # # Bottom wall: reflect populations into the fluid
    # for idx in (2, 5, 6):
    #     f_n_arrays[idx][bottomDoFs] = (
    #         f_star_arrays[opp_idx[idx]][bottomDoFs])
    
    # # Top wall
    # for idx in (4, 7, 8):
    #     f_n_arrays[idx][topDoFs] = (
    #         f_star_arrays[opp_idx[idx]][topDoFs])
        
    for idx in range(1, Q):
        dofs = bouncebackDofs[idx]
        f_n_arrays[idx][dofs] = (
            f_star_arrays[opp_idx[idx]][dofs])
            
            
          
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
    
    phi_a, mu_a = phi_n.as_numpy, mu_n.as_numpy
    rho = sum(f_n_arrays)

    
    if n % 10 == 0:
    
        marker = dune.fem.GridMarker(
            1 - phi_n**2,
            refineTolerance=0.6,
            coarsenTolerance=0.2,
            minLevel=0,
            maxLevel=2)
    
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
        
        bouncebackDofs = build_bounceback_dofs(mesh, V, xi_values, Q)
        
    
    if n % 50 == 0:
        
        print("n = ", n, "writing to file")
        #print("mass_diff type:", type(mass_diff))
        
        #print("mass_n type:", type(mass_n), "\n\n")
        
        if n > 3000:
            theta_avg = computeContactAngle(phi_n, h_min, interfaceThickness, mesh)
            print("theta = ", theta_avg)
            
            xh = V.interpolate(x[0], name="x")
            yh = V.interpolate(x[1], name="y")
            zh = V.interpolate(x[2], name="z")
            dof_coords = np.column_stack([xh.as_numpy, yh.as_numpy, zh.as_numpy])  
            left_cl, right_cl = trackCL(phi_n, dof_coords, h_min)
            
            print("left_cl = ", left_cl, " right_cl = ", right_cl)
        
        vel_expr = getVel(f_n, forceDensity)
        vel3.interpolate(ufl.as_vector([vel_expr[0], vel_expr[1], vel_expr[2]]))
        
        v_array = vel3.as_numpy.reshape(-1, 3)

        vel_mag = np.linalg.norm(v_array, axis=1)
        
        print("Min speed:", vel_mag.min())
        print("Max speed:", vel_mag.max())
        
        print("Max KE:", vel_mag.max()**2)
        
        if n > 3000:
            log_file.write(f"{n:15d}"
                        f"{right_cl:15.3f}"
                        f"{vel_mag.max():15.8g}"
                        f"{theta_avg:15.2f}\n")
            log_file.flush()

        mesh.writeVTK(os.path.join(outDirName, f"vel_{n:06d}"), pointdata=[vel3])
        
        mesh.writeVTK(os.path.join(outDirName, f"phi_{n:06d}"),
            pointdata=[phi_n])