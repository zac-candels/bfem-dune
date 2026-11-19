import numpy as np
import matplotlib.pyplot as plt 

def fit_tanners_law(t, R, t_min=None, t_max=None):
    """
    Fit R(t) = K * t^(1/7) using linear least squares.

    Parameters
    ----------
    t : array
        Simulation times.
    R : array
        Contact-line radii.
    t_min, t_max : float
        Optional fitting interval.

    Returns
    -------
    K : float
        Best-fit prefactor.
    R_fit : array
        Fitted contact-line radii.
    """

    t = np.asarray(t, dtype=float)
    R = np.asarray(R, dtype=float)

    mask = (t > 0) & np.isfinite(t) & np.isfinite(R)

    if t_min is not None:
        mask &= t >= t_min

    if t_max is not None:
        mask &= t <= t_max

    t_fit = t[mask]
    R_data = R[mask]

    x = t_fit**(1/7)

    # Linear least-squares fit through the origin
    K = np.sum(x * R_data) / np.sum(x**2)

    R_fit = K * t**(1/7)

    return K, R_fit

A_vals = 0.1
kappa = 0.02
sigma = np.sqrt(8*A_vals/9)
plt.figure(figsize=(8, 5))

fileName = "/home/zcandels/tannersLaw/tests/tannerLaw2D/M1_A1_tauL0.1_viscRatio50/simulation_log.txt"
dataArr = np.loadtxt(fileName, skiprows = 1, usecols=(0, 1, 2, 3))

times = dataArr[:, 0]
clPos = dataArr[:, 1]
times = times[:] - times[0]
clPos = clPos[:] - clPos[0]

label = rf"$\sigma$ = {sigma:.2g}"

plt.plot(times[120:], clPos[120:], 'o',
     markersize=3,
     label=label)

# plt.figure()
# plt.plot(times, clPos)
# plt.plot(times, 1.834*times**(1/10))




# Example usage:
K, R_fit = fit_tanners_law(times, clPos,
                           t_min=500)

print(f"Tanner prefactor K = {K:.6f}")



plt.plot(times[120:], K*times[120:]**(1/7), '--',
         label=fr'Tanner fit: $\alpha t^{{1/7}}$')

plt.xlabel('$t^*$', size=15)
plt.ylabel(r'$r_{cl}^*$', size=15)
plt.legend()
plt.grid()
plt.show()