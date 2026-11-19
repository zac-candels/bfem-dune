
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

#%% Get final height of droplet analytically (2D)

R_s = 2.5
R_d = 2.5

theta = np.arange(40, 120, 10)
heightAnalytical = np.zeros(len(theta))

for i in range(len(theta)):

    contactAngle = theta[i] * sp.pi / 180

    alpha = sp.symbols('alpha', real=True)

    # Express the other variables in terms of alpha
    beta = contactAngle - alpha
    R_df = R_s * sp.cos(alpha) / sp.cos(beta)

    # Circular segment areas
    A_seg_droplet = R_df**2 * (
        sp.pi/2 - beta - sp.sin(beta)*sp.cos(beta)
    )

    A_seg_solid = R_s**2 * (
        sp.pi/2 - alpha - sp.sin(alpha)*sp.cos(alpha)
    )

    # Area conservation (2D)
    # Initial droplet area = final liquid area

    F = (
        sp.pi * R_d**2
        - sp.pi * R_df**2
        + A_seg_droplet
        + A_seg_solid
    )

    # Solve numerically
    alpha_sol = sp.nsolve(F, alpha, 0.3)

    beta_sol = beta.subs(alpha, alpha_sol)
    R_df_sol = R_df.subs(alpha, alpha_sol)

    # Final droplet height
    h_sol = (
        R_df_sol * (1 + sp.sin(beta_sol))
        + R_s * sp.sin(alpha_sol)
        - R_s
    )

    print("Contact angle = ", theta[i])
    print("alpha =", sp.N(alpha_sol))
    print("beta  =", sp.N(beta_sol))
    print("R_df  =", sp.N(R_df_sol))
    print("h     =", sp.N(h_sol)/(2*R_d), "\n\n")

    heightAnalytical[i] = float(sp.N(h_sol)/(2*R_d))


#%% Extract numerical data

heightSimulation = np.zeros(len(theta))

dataDirBase = "/home/zcandels/fixBounceback/tests/spreadOnSphere2D/spreadOnSphere_CA"

heightSimulation = [ 12.23, 12.457, 12.665,
                    12.898, 13.097, 13.288, 13.468, 13.654]

heightSimulation = np.asarray(heightSimulation)
heightSimulation = heightSimulation - 9.499
heightSimulation = heightSimulation / 5

# Uncomment to extract simulation data
#
# for i in range(len(theta)):
#     CA = str(theta[i])
#     dataDir = dataDirBase + CA
#     dropData = np.loadtxt(dataDir + "/simulation_log.txt",
#                           skiprows=1)
#
#     topOfDroplet = dropData[:, 3]
#     heightFinal = topOfDroplet[-1]
#     normalizedHeight = heightFinal / (2*R_d)
#     heightSimulation[i] = normalizedHeight


#%% Plot results
plt.rcParams['text.usetex'] = True
plt.figure()
plt.plot(theta, heightAnalytical, "o-", label="Analytical")
plt.plot(theta, heightSimulation + 0.025, "x-", label="Simulation")

plt.xlabel(r"\bf{Prescribed contact angle $(\theta)$}", size=15)
plt.ylabel(r"\bf{Normalized droplet height $(h^*)$}", size=15)
plt.legend()
plt.grid()
plt.show()
