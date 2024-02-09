import pickle

import numpy as np


def save_pickle(save_path, data):
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)

def read_pickle(read_path):
    with open(read_path, 'rb') as f:
        data = pickle.load(f)
    return data


def breakroom_avoidance_policy(x, y, dsval, noise=0.3):

    a = 0
    if dsval < 95:
        if x > 2:  # First room
            a = 2  # right
        elif (x < 2 and x > -2.7) and (y < 1.45):  # middle room, lower half
            a = 2  # right

        elif (x < 2 and x > -2.7) and (y > 1.45):  # middle room, upper half
            a = 1  # left
        else:  # Final Room
            a = 1  # left
    else:
        a = 0

    if np.random.rand() < noise:
        a = int(np.random.randint(0, 3))

    return a