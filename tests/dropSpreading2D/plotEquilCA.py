import numpy as np
import matplotlib.pyplot as plt

thetas = np.arange(30, 165, 15)*np.pi/180
wettingPotentialDiscrete = np.cos(thetas)

wettingPotential = np.linspace(-1, 1, 200)
theta = np.arccos(wettingPotential)*180/np.pi

tauPoint01 = np.array([30.07, 44.35, 59.66, 75.099, 90.3699, 105.676, 118.46, 132.67, 146.49])

tauPoint1 = np.array([33.44, 47.52, 61.84, 76.37, 91.48, 106.34, 118.18, 132.076, 145.23])

tau1 = np.array([28.85, 43.05, 56.96, 71.48, 85.93, 99.72, 112.93, 125.54, 136.62])

plt.rcParams['text.usetex'] = True
plt.figure()
plt.plot(-wettingPotential, theta, label="Theory", color="orange")
plt.plot(-wettingPotentialDiscrete, tauPoint01, 'o', label=r"$\tau=0.01$")
plt.plot(-wettingPotentialDiscrete, tauPoint1, '+', label=r"$\tau=0.1$", color="black")
plt.xlabel(r"$\Omega$",size=18)
plt.ylabel(r"$\theta_e({\circ})$", size=18)
plt.legend()


def computeContactAngle(h, D):
    return 2 * np.arctan(2*h/D)*180/np.pi 


tauPoint01_height = [1.359, 1.659, 1.944, 2.2089, 2.455, 2.69, 2.858, 3.058, 3.232]

tauPoint01_diam = [13.027-2.91, 12.05-3.91, 11.39-4.61, 10.874-5.127,
                   10.455-5.5766, 10.05-5.972, 9.673-6.27, 9.316-6.636, 8.956-7.01 ]


tauPoint1_height = [1.45096, 1.737, 1.9953, 2.236, 2.476, 2.7,
                    2.855, 3.05, 3.21]

tauPoint1_diam = [12.77 - 3.1086, 11.932-4.04, 11.3279-4.666, 10.834-5.152,
                  10.414-5.588, 10.03-5.986, 9.699-6.28, 9.337-6.626, 8.99-6.98]

for i in range(len(tauPoint01_height)):
    print("CA = ", computeContactAngle(tauPoint01_height[i], tauPoint01_diam[i]))
    
    