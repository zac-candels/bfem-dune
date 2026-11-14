import numpy as np
import matplotlib.pyplot as plt 


dataArr = np.loadtxt("/home/zcandels/bfem-dune/tests/tannerLaw2D/tannersLaw/simulation_log.txt",
                     skiprows = 1)

times = dataArr[:, 0]
clPos = dataArr[:, 1]
#times = times[:1000]
times = times - times[0]
#clPos = clPos[:1000]
clPos = clPos - clPos[0]

# plt.figure()
# plt.plot(times, clPos)
# plt.plot(times, 1.834*times**(1/10))

def fit_tanners_law(t, R, t_min=None, t_max=None):
    """
    Fit R(t) = K * t^(1/10) using linear least squares.

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

    x = t_fit**(1/10)

    # Linear least-squares fit through the origin
    K = np.sum(x * R_data) / np.sum(x**2)

    R_fit = K * t**(1/10)

    return K, R_fit


# Example usage:
K, R_fit = fit_tanners_law(times, clPos,
                           t_min=500)

print(f"Tanner prefactor K = {K:.6f}")

plt.figure(figsize=(8, 5))
plt.plot(times, clPos, 'o',
         markersize=3, label='Simulation')
plt.plot(times, R_fit, '--',
         label=fr'Tanner fit: $R={K:.4f}t^{{1/10}}$')

plt.xlabel('Time')
plt.ylabel('Contact line radius')
plt.legend()
plt.grid()
plt.show()