import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import vonmises


def circular_gau_filter(H, a_ax, kappa: float = 12.6):
    """
    Filter the 3-d histogram only along the circular dimension (the third axis).
    Parameters
    ----------
    H : ndarray
        3-d array with shape (Nx, Ny, Na). Histogram count of the x, y, angle in the bins.
    a_ax : ndarray
        1-d array with shape (Na, ). Value (or middle points) of the angle bins.
    kappa : float
        Kappa of the von-mise distribution. It controls the concentration. Recommend 4*np.pi.

    Returns
    -------
    Hout : ndarray
        3-d array with shape (Nx, Ny, Na). Smoothed histogram of x, y, angle. The sum remains the same.
    """

    da = a_ax[1] - a_ax[0]
    Nx, Ny, Na = H.shape
    Np = Nx * Ny
    H_flat = H.reshape(Np, Na).T  # (Na, Np)
    a_axs = np.stack([a_ax] * Np).T  # (Na, Np)
    Hout = np.zeros(H_flat.shape)  # (Na, Np)

    for i in range(len(a_ax)):
        Hout += vonmises.pdf(a_axs, kappa, loc=a_ax[i], scale=1) * H_flat[i, :].reshape(1, Np) * da
    Hout = Hout.T.reshape(Nx, Ny, Na)
    return Hout


if __name__ == '__main__':
    a_ax = np.linspace(-np.pi, np.pi, 36)
    foo = np.zeros((10, 10, a_ax.shape[0]))

    foo[5, 6, 18] = 1
    foo[5, 6, 10] = 2
    foo[5, 5, 18] = 1

    foo_out = circular_gau_filter(foo, a_ax, kappa=4 * np.pi)
    print(foo_out[5, 5, :].sum())
    print(foo_out[5, 6, :].sum())
    fig, ax = plt.subplots(facecolor='w')
    ax.plot(a_ax, foo_out[5, 5, :], marker='x')
    ax.plot(a_ax, foo_out[5, 6, :])
    fig.savefig('try4.png')

    # import os
    # os.makedirs('try', exist_ok=True)
    # for i in range(32):
    #
    #     fig, ax = plt.subplots(facecolor='w')
    #     im = ax.pcolormesh(foo_out[:, :, i], vmin=0, vmax=foo_out.max(), cmap='jet')
    #     plt.colorbar(im)
    #     fig.savefig('try/%d.jpg'%i)
