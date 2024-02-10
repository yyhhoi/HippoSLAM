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


    def save_buffer_torch(self, pth:str):
        torch.save(torch.vstack(self.buffer), pth)

    def load_buffer_torch(self, pth: str):
        """

        Parameters
        ----------
        pth : str
            Path to the data containing (NumSamples, data_dim) torch tensor (float32)

        """
        alltrajs = torch.load(pth)
        for traj in alltrajs :
            self.push(traj)



class ReplayMemoryAWAC(ReplayMemory):
    def __init__(self, max_size, discrete_obs = None):
        super().__init__(max_size)
        self.sliceinds = None  # reserved
        self.discrete_obs = discrete_obs  # None or int

    def sample(self, batch_size):
        assert self.sliceinds is not None
        if self.size <= batch_size:
            indices = [ind for ind in range(self.size)]
        else:
            indices = sample(range(self.size), batch_size)
        data = torch.vstack([self.buffer[index] for index in indices])
        s = data[:, self.sliceinds[0][0]:self.sliceinds[0][1]]  # (Nsamp, obs_dim) float32, or (Nsamp, 1) float32 if discrete
        a = data[:, self.sliceinds[1][0]:self.sliceinds[1][1]].to(torch.int64)  # (Nsamp, act_dim) int64
        snext = data[:, self.sliceinds[2][0]:self.sliceinds[2][1]]  # (Nsamp, obs_dim) float32, or (Nsamp, 1) float32 if discrete
        r = data[:, self.sliceinds[3][0]:self.sliceinds[3][1]]  # (Nsamp, 1) float32
        end = data[:, self.sliceinds[4][0]:self.sliceinds[4][1]]  # (Nsamp, 1) float32

        if self.discrete_obs:
            s2 = torch.nn.functional.one_hot(s.to(torch.int64), num_classes=self.discrete_obs).squeeze()  # (Nsamp, obs_classes)
            s2next = torch.nn.functional.one_hot(snext.to(torch.int64), num_classes=self.discrete_obs).squeeze() # (Nsamp, obs_classes)
            data_tuple = (s2.to(torch.float32), a, s2next.to(torch.float32), r, end)
        else:
            data_tuple = (s, a, snext, r, end)
        return data_tuple

    def specify_data_tuple(self, s: Tuple[int, int], a: Tuple[int, int], snext: Tuple[int, int], r: Tuple[int, int],
                           end: Tuple[int, int]):
        self.sliceinds = (s, a, snext, r, end)
        return None


class ReplayMemoryA2C(ReplayMemory):
    def __init__(self, max_size):
        super().__init__(max_size)
        self.sliceinds = None

    def sample(self, batch_size):
        assert self.sliceinds is not None
        if self.size <= batch_size:
            indices = [ind for ind in range(self.size)]
        else:
            indices = sample(range(self.size), batch_size)
        data = torch.vstack([self.buffer[index] for index in indices])
        s = data[:, self.sliceinds[0][0]:self.sliceinds[0][1]]  # (Nsamp, obs_dim)
        a = data[:, self.sliceinds[1][0]:self.sliceinds[1][1]].to(torch.int64)  # (Nsamp, 1)
        G = data[:, self.sliceinds[2][0]:self.sliceinds[2][1]]  # (Nsamp, 1)
        data_tuple = (s, a, G)
        return data_tuple

    def specify_data_tuple(self, s: Tuple[int, int], a: Tuple[int, int], G: Tuple[int, int]):
        self.sliceinds = (s, a, G)
        return None
