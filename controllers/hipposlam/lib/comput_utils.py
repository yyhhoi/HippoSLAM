import numpy as np
from scipy.stats import vonmises


def cmean(x, w=None):
    """
    Compute the circular mean and mean resultant vector length of the circular data x.

    Parameters
    ----------
    x : ndarray
        1-d array with circular values in radians. Expecting [-pi, pi).
    w : ndarray or None
        1-d array with the same shape as x. Weighting of each value in computing the mean. Need to be normalized.

    Returns
    -------
    mu : float
        Mean angle in radian
    R : float
        Mean resultant vector length in range (0, 1].

    """
    N = x.shape[0]
    complex_vec = np.exp(1j * x)
    if w is None:
        meanvec = np.sum(complex_vec) / N
    else:
        w = w / np.sum(w)
        meanvec = np.sum(w * complex_vec)
    R = np.abs(meanvec)
    mu = np.angle(meanvec)
    return mu, R



def opposite_angle(x):
    assert x <= np.pi
    assert x >= -np.pi

    if x > 0:
        out = -(2*np.pi - (x + np.pi) )
    else:
        out = x + np.pi
    return out

def divide_ignore(a: np.ndarray, b: np.ndarray):
    e1, e2, e3, e4 = np.isinf(a).sum(), np.isnan(a).sum(), np.isinf(b).sum(), np.isnan(b).sum()
    if np.any(np.array([e1, e2, e3, e4]) > 0):
        raise ValueError('No invalid values (nan and inf) in the inputs a, b are allowed.')
    else:
        with np.errstate(divide='ignore', invalid='ignore'):
            c = a / b
        c[np.isinf(c)] = 0
        c[np.isnan(c)] = 0
    return c

def midedges(edges):
    return (edges[:-1] + edges[1:]) / 2


class Arena:
    def __init__(self, xmin, xmax, ymin, ymax, amin, amax, dp, da, bodysd):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.amin = amin
        self.amax = amax
        self.dp = dp
        self.da = da
        self.bodysd = bodysd
        self.bodysd_ind = self.bodysd/self.dp
        self.xedges = np.arange(xmin, xmax+dp, dp)
        self.yedges = np.arange(ymin, ymax+dp, dp)
        self.aedges = np.arange(-np.pi, np.pi+da, da)
        self.xedm = midedges(self.xedges)
        self.yedm = midedges(self.yedges)
        self.aedm = midedges(self.aedges)

    def compute_histogram3d(self, x, y, a):
        data3d = np.stack([x, y, a]).T  # (T, 3)
        H3d, edges3d = np.histogramdd(data3d, bins=[self.xedges, self.yedges, self.aedges])
        return H3d, edges3d

    def compute_ratemap(self, occ, spcounts):
        return divide_ignore(spcounts, occ)

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


def MLM(p, d, n, t, minerr=0.01):
    """
    MLM (Caccuci et al., 2004) is not used since it produced diverging tuning curves for some examples.

    Parameters
    ----------
    p : ndarray
        1-d array with shape (I, ). Firing rate in each position bin i.
    d : ndarray
        1-d array with shape (J, ). Firing rate in each direction bin j.
    n : ndarray
        2-d array with shape (I, J). Number of spike counts in each position and direction bin
    t : ndarray
        2-d array with shape (I, J). Dwell time in each position and direction bin
    minerr : float
        Error tolerance of MLM iteration.
    Returns
    -------

    """
    I, J = p.shape[0], d.shape[0]

    p_out = p.copy()
    d_out = d.copy()
    Np = np.sum(p * n.sum(axis=1))
    Nd = np.sum(d * n.sum(axis=0))
    err = 1000
    i = 0
    while err > minerr:
        if (i+1) % 100 == 0:
            print(err)
        p_est = divide_ignore(np.mean(n, axis=1), np.mean(d_out.reshape(1, J) * t, axis=1))
        d_est = divide_ignore(np.mean(n, axis=0), np.mean(p_out.reshape(I, 1) * t, axis=0))

        Np_est = np.sum(p_est * n.sum(axis=1))
        Nd_est = np.sum(d_est * n.sum(axis=0))
        p_est_norm = p_est / Np_est * Np
        d_est_norm = d_est / Nd_est * Nd
        # error
        errp = np.mean(np.abs(p_est_norm - p_out))
        errd = np.mean(np.abs(d_est_norm - d_out))
        err = errd + errp

        # update
        p_out = p_est_norm
        d_out = d_est_norm

        i += 1
    return p_out, d_out


