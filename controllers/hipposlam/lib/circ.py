import numpy as np
def center_angle(angle):
    return (angle + np.pi) % (2*np.pi) - np.pi


def cdiff(alpha, beta):
    """
    Difference between pairs :math:`x_i-y_i` around the circle,
    computed efficiently.

    :param alpha:  sample of circular random variable
    :param beta:   sample of circular random variable
    :return: distance between the pairs
    """
    return center_angle(alpha - beta)