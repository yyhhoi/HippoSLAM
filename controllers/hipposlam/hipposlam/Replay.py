from random import sample
from typing import Tuple

import numpy as np
import torch


class ReplayMemory:
    def __init__(self, max_size):
        # deque object that we've used for 'episodic_memory' is not suitable for random sampling
        # here, we instead use a fix-size array to implement 'buffer'
        self.buffer = [None] * max_size
        self.max_size = max_size
        self.index = 0
        self.size = 0

    def push(self, obj):
        self.buffer[self.index] = obj
        self.size = min(self.size + 1, self.max_size)
        self.index = (self.index + 1) % self.max_size

    def sample(self, batch_size):
        indices = sample(range(self.size), batch_size)
        return [self.buffer[index] for index in indices]

    def __len__(self):
        return self.size

    def reset(self):
        self.buffer = [None] * self.max_size
        self.index = 0
        self.size = 0

    def from_offline_np(self, data):
        """

        Parameters
        ----------
        data : list or tuple
            list of 2-d numpy arrays with shape (time, data_dim) recording the (s, a, s', r, end) of a whole episode.
        Returns
        -------

        """
        alltrajs = np.vstack(data)  # (time, data_dim=15)
        for traj in alltrajs:
            self.push(torch.from_numpy(traj).to(torch.float))



class ReplayMemoryAWAC(ReplayMemory):
    def __init__(self, max_size):
        super().__init__(max_size)
        self.indexes = None

    def sample(self, batch_size):
        assert self.indexes is not None
        if self.size <= batch_size:
            indices = [ind for ind in range(self.size)]
        else:
            indices = sample(range(self.size), batch_size)
        data = torch.vstack([self.buffer[index] for index in indices])
        s = data[:, self.indexes[0][0]:self.indexes[0][1]]  # (Nsamp, obs_dim)
        a = data[:, self.indexes[1][0]:self.indexes[1][1]].to(torch.int64)  # (Nsamp, act_dim)
        snext = data[:, self.indexes[2][0]:self.indexes[2][1]]  # (Nsamp, obs_dim)
        r = data[:, self.indexes[3][0]:self.indexes[3][1]]  # (Nsamp, 1)
        end = data[:, self.indexes[4][0]:self.indexes[4][1]]  # (Nsamp, 1)
        data_tuple = (s, a, snext, r, end)
        return data_tuple

    def specify_data_tuple(self, s: Tuple[int, int], a: Tuple[int, int], snext: Tuple[int, int], r: Tuple[int, int],
                           end: Tuple[int, int]):
        self.indexes = (s, a, snext, r, end)
        return None


class ReplayMemoryA2C(ReplayMemory):
    def __init__(self, max_size):
        super().__init__(max_size)
        self.indexes = None

    def sample(self, batch_size):
        assert self.indexes is not None
        if self.size <= batch_size:
            indices = [ind for ind in range(self.size)]
        else:
            indices = sample(range(self.size), batch_size)
        data = torch.vstack([self.buffer[index] for index in indices])
        s = data[:, self.indexes[0][0]:self.indexes[0][1]]  # (Nsamp, obs_dim)
        a = data[:, self.indexes[1][0]:self.indexes[1][1]].to(torch.int64)  # (Nsamp, 1)
        G = data[:, self.indexes[2][0]:self.indexes[2][1]]  # (Nsamp, 1)
        data_tuple = (s, a, G)
        return data_tuple

    def specify_data_tuple(self, s: Tuple[int, int], a: Tuple[int, int], G: Tuple[int, int]):
        self.indexes = (s, a, G)
        return None
