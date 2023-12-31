import numpy as np


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