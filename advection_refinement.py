import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt

import dune
import dune.fem
import ufl
import dune.alugrid


# ============================================================
# Parameters
# ============================================================

T = 10.0
dt = 0.0005
numSteps = int(T / dt)

Lx = 2.0
Ly = 2.0

Nx = 15
Ny = 15

order = 2

# Adapt the mesh every N time steps
adaptEvery = 10

# Maximum refinement level
maxLevel = 4

# Tolerance used when identifying periodic DOFs
tol = 1e-10


# ============================================================
# Parameters appearing in the UFL form
# ============================================================

t = dune.ufl.Constant(0.0)
c = dune.ufl.Constant((1.0, 0.0))


# ============================================================
# Create mesh
# ============================================================

domain = dune.grid.cartesianDomain(
    [0.0, 0.0],
    [Lx, Ly],
    [Nx, Ny]
)

mesh = dune.fem.view.adaptiveLeafGridView(
    dune.alugrid.aluConformGrid(domain)
)

print("Initial cells:", mesh.size(0))


# ============================================================
# Initial condition
# ============================================================

def initial_condition(x):

    return (
        1.0
        - ufl.exp(
            -1.0
            / (
                4.0
                * (
                    (x[0] - 1.0)**2
                    + (x[1] - 1.0)**2
                )
            )
        )
    )


# ============================================================
# FINITE ELEMENT SPACE
#
# Create this ONCE.
#
# adaptiveLeafGridView allows the underlying DoF layout to
# change when gridAdapt() is called.
# ============================================================

fnSpace = dune.fem.space.lagrange(
    mesh,
    order=order
)


# ============================================================
# UFL objects
#
# Create these ONCE.
# ============================================================

uTrial = ufl.TrialFunction(fnSpace)
v = ufl.TestFunction(fnSpace)

x = ufl.SpatialCoordinate(fnSpace)


# ============================================================
# Solution
#
# Create this ONCE.
# ============================================================

u_n = fnSpace.interpolate(
    initial_condition(x),
    name="u_n"
)


# ============================================================
# Forms
#
# Create these ONCE.
#
# IMPORTANT:
# These symbolic expressions refer to the adaptive FEM space.
# They are not reconstructed after grid adaptation.
# ============================================================

bilinForm = (
    uTrial * v * ufl.dx
)

linForm = (
    u_n * v * ufl.dx
    - dt
      * ufl.dot(c, ufl.grad(u_n))
      * v
      * ufl.dx
    - 0.5
      * dt**2
      * ufl.dot(c, ufl.grad(u_n))
      * ufl.dot(c, ufl.grad(v))
      * ufl.dx
)


# ============================================================
# DUNE FEM operator
#
# Create this ONCE.
#
# This is the important part for avoiding repeated C++
# operator generation/compilation.
# ============================================================

rhsOp = dune.fem.operator.galerkin(
    linForm - uTrial * v * ufl.dx
)


# ============================================================
# DUNE vectors
#
# These are also persistent discrete functions associated with
# fnSpace.
#
# We pass them to gridAdapt() so their storage is updated when
# the mesh changes.
# ============================================================

rhsFn = fnSpace.zero.copy()

zeroFn = fnSpace.zero.copy()


# ============================================================
# Build ONLY the mesh-dependent SciPy objects
#
# This function is called initially and after adaptation.
#
# It does NOT construct:
#   fnSpace
#   TrialFunction
#   TestFunction
#   SpatialCoordinate
#   UFL forms
#   rhsOp
#
# Those objects are created only once above.
# ============================================================

def build_linear_solver():

    # --------------------------------------------------------
    # Assemble current mass matrix
    #
    # The mathematical form has not changed, but its numerical
    # matrix has changed because the mesh/DoF layout changed.
    # --------------------------------------------------------

    mat = dune.fem.assemble(
        [bilinForm]
    )

    A = scipy.sparse.csr_matrix(
        mat.as_numpy.copy()
    )

    # --------------------------------------------------------
    # Coordinates of CURRENT DoFs
    # --------------------------------------------------------

    x_dof = fnSpace.interpolate(
        x[0],
        name="x_dof"
    ).as_numpy.copy()

    y_dof = fnSpace.interpolate(
        x[1],
        name="y_dof"
    ).as_numpy.copy()

    N = len(x_dof)

    # --------------------------------------------------------
    # Apply periodic coordinate mapping
    #
    # x = Lx -> x = 0
    # y = Ly -> y = 0
    # --------------------------------------------------------

    x_periodic = x_dof.copy()
    y_periodic = y_dof.copy()

    x_periodic[
        np.abs(x_periodic - Lx) < tol
    ] = 0.0

    y_periodic[
        np.abs(y_periodic - Ly) < tol
    ] = 0.0

    # --------------------------------------------------------
    # Create periodic equivalence classes
    # --------------------------------------------------------

    keys = np.column_stack(
        (
            np.round(x_periodic, 10),
            np.round(y_periodic, 10)
        )
    )

    unique_keys, inverse = np.unique(
        keys,
        axis=0,
        return_inverse=True
    )

    nPeriodic = len(unique_keys)

    # --------------------------------------------------------
    # Prolongation matrix
    #
    # u_full = P @ u_periodic
    # --------------------------------------------------------

    rows = np.arange(N)
    cols = inverse
    data = np.ones(N)

    P = scipy.sparse.coo_matrix(
        (data, (rows, cols)),
        shape=(N, nPeriodic)
    ).tocsr()

    # --------------------------------------------------------
    # Periodically reduced mass matrix
    #
    # A_periodic = P^T A P
    # --------------------------------------------------------

    A_periodic = (
        P.T @ A @ P
    ).tocsc()

    # --------------------------------------------------------
    # Factorize
    #
    # This is the expensive SciPy operation.
    #
    # It is done ONLY when the mesh changes.
    # --------------------------------------------------------

    solver = scipy.sparse.linalg.factorized(
        A_periodic
    )

    return P, solver


# ============================================================
# Initial numerical matrix/factorization
# ============================================================

P, solver = build_linear_solver()


print("Initial DOFs:", fnSpace.size)


# ============================================================
# VTK
# ============================================================

vtk = mesh.sequencedVTK(
    "advection",
    pointdata=[u_n]
)


# ============================================================
# Time loop
# ============================================================

for n in range(numSteps):

    # --------------------------------------------------------
    # Advance time
    # --------------------------------------------------------

    t.value += dt

    # --------------------------------------------------------
    # Assemble RHS using the SAME rhsOp
    #
    # No UFL expression is rebuilt here.
    # --------------------------------------------------------

    rhsOp(
        zeroFn,
        rhsFn
    )

    # --------------------------------------------------------
    # Periodic reduction
    #
    # b_periodic = P^T b
    # --------------------------------------------------------

    b_periodic = (
        P.T @ rhsFn.as_numpy
    )

    # --------------------------------------------------------
    # Solve
    # --------------------------------------------------------

    u_periodic = solver(
        b_periodic
    )

    # --------------------------------------------------------
    # Expand periodic solution
    #
    # u_full = P u_periodic
    # --------------------------------------------------------

    u_full = (
        P @ u_periodic
    )

    # --------------------------------------------------------
    # Put solution back into DUNE function
    # --------------------------------------------------------

    u_n.as_numpy[:] = u_full

    # --------------------------------------------------------
    # Adapt mesh
    # --------------------------------------------------------

    if (n + 1) % adaptEvery == 0:

        print(
            f"Step {n+1}: "
            f"adapting, cells = {mesh.size(0)}, "
            f"DOFs = {fnSpace.size}"
        )

        # ----------------------------------------------------
        # Refinement indicator
        # ----------------------------------------------------

        marker = dune.fem.GridMarker(
            u_n,
            refineTolerance=0.9,
            coarsenTolerance=0.4,
            minLevel=0,
            maxLevel=maxLevel
        )

        # ----------------------------------------------------
        # Adapt the grid.
        #
        # IMPORTANT:
        #
        # u_n, rhsFn and zeroFn are passed to gridAdapt so DUNE
        # updates their storage when the DoF layout changes.
        #
        # We do NOT recreate fnSpace.
        # We do NOT recreate uTrial.
        # We do NOT recreate v.
        # We do NOT recreate x.
        # We do NOT recreate the UFL forms.
        # We do NOT recreate rhsOp.
        # ----------------------------------------------------

        dune.fem.gridAdapt(
            marker,
            [u_n, rhsFn, zeroFn]
        )

        print(
            f"             new cells = {mesh.size(0)}, "
            f"new DOFs = {fnSpace.size}"
        )

        # ----------------------------------------------------
        # The DUNE FEM objects above survive adaptation.
        #
        # However, our SciPy objects contain the old DoF
        # numbering and therefore MUST be rebuilt.
        # ----------------------------------------------------

        P, solver = build_linear_solver()

    # --------------------------------------------------------
    # VTK output
    # --------------------------------------------------------

    if n % 10 == 0:
        vtk()


print()
print("Simulation finished.")