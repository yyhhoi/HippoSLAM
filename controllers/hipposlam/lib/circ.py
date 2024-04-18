import numpy as np


def center_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def cdiff(alpha, beta):
    """
    Difference between pairs :math:`x_i-y_i` around the circle,
    computed efficiently.

    :param alpha:  sample of circular random variable
    :param beta:   sample of circular random variable
    :return: distance between the pairs
    """
    return center_angle(alpha - beta)


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
