import numpy as np
from .comput_utils import cmean, opposite_angle

def compute_steering(ds_vals, ds_maxs, ds_angles, epsilon=0.1):


    ds_amp = ds_maxs - ds_vals

    if np.all(ds_amp < epsilon):
        steering_a = 0
    else:
        steering_a, _ = cmean(ds_angles, ds_amp)
        steering_a = opposite_angle(steering_a)

    return steering_a



def convert_steering_to_wheelspeed(a, k):
    """

    Parameters
    ----------
    a : float
        [-pi, pi). Starting from the front of the robot at 0, counter-clockwise
    k : float
        Wheel speed.
    Returns
    -------

    """
    assert a <= np.pi
    assert a >= -np.pi
    if a > 0:  # Left side
        vl = np.cos(a) * k
        vr = k
    elif a < 0:  # right side
        vl = k
        vr = np.cos(a) * k
    else:
        vl = k
        vr = k
    return vl, vr







