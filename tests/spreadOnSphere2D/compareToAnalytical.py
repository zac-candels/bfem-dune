import numpy as np
import matplotlib.pyplot as plt 

#%% Get final height of droplet analytically


import sympy as sp

R_s = 2.5
R_d = 2.5
theta = np.arange(30, 105, 15)
heightAnalytical = np.zeros(len(theta))
for i in range(len(theta)):
    
    contactAngle = theta[i]*sp.pi/180

    alpha = sp.symbols('alpha', real=True)
    
    # Express the other variables in terms of alpha
    beta = contactAngle - alpha
    R_df = R_s * sp.cos(alpha) / sp.cos(beta)
    
    # Volume conservation
    F = (
        (4/3) * sp.pi * R_d**3
        - (4/3) * sp.pi * R_df**3
        + (1/3) * sp.pi * R_df**3
          * (2 - 3*sp.sin(beta) + sp.sin(beta)**3)
        + (1/3) * sp.pi * R_s**3
          * (2 - 3*sp.sin(alpha) + sp.sin(alpha)**3)
    )
    
    # Solve numerically
    alpha_sol = sp.nsolve(F, alpha, 0.3)
    
    beta_sol = beta.subs(alpha, alpha_sol)
    R_df_sol = R_df.subs(alpha, alpha_sol)
    
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
    heightAnalytical[i] = sp.N(h_sol)/(2*R_d)



#%% Extract numerical data 
heightSimulation = np.zeros(len(theta))
dataDirBase = "/home/zcandels/fixBounceback/tests/spreadOnSphere2D/spreadOnSphere_CA"
heightSimulation = [12.0049, 12.343, 12.6494, 12.926, 13.2699]
heightSimulation = np.asarray(heightSimulation)
heightSimulation = heightSimulation - 9.48177
heightSimulation = heightSimulation / 5
# for i in range(4):
#     CA = str(theta[i])
#     dataDir = dataDirBase + CA
#     dropData = np.loadtxt(dataDir + "/simulation_log.txt", 
#                           skiprows=1)
    
#     topOfDroplet = dropData[:, 3]
#     heightFinal = topOfDroplet[-1]
#     normalizedHeight = heightFinal / (2*2.5)
#     heightSimulation[i] = normalizedHeight

#%% 

plt.figure()
plt.plot(theta, heightAnalytical, "o-", label="Analytical")
plt.plot(theta, heightSimulation, "x-", label="Simulation")
plt.legend()

