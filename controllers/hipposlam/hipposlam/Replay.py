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


class ReplayMemoryCat(ReplayMemory):
    def __init__(self, max_size):
        super().__init__(max_size)
        self.indexes = None

    def sample(self, batch_size):
        assert self.indexes is not None
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


class Replayer:
    def __init__(self, buffer=None, max_buffer_size=100, seed=None, verbose=False):
        if buffer:
            self.buffer = buffer
        else:
            self.buffer = []
        self.max_buffer_size = max_buffer_size  # Number of episodes.
        self.indexes = None
        self.seed = seed
        self.verbose = verbose

    def append(self, traj):
        """

        Parameters
        ----------
        traj : torch.tensor with dtype=torch.float32
            2-d torch tensor with (time, data_dim). data dim is the shape of the (s, a, s', r, end).

        Returns
        -------

        """
        self.buffer.append(traj)
        if len(self.buffer) > self.max_buffer_size:
            self.buffer.pop(0)
        return None

    def specify_data_tuple(self, s: Tuple[int, int], a: Tuple[int, int], snext: Tuple[int, int], r: Tuple[int, int],
                           end: Tuple[int, int]):
        self.indexes = (s, a, snext, r, end)
        return None

    def sample(self, batch_size):

        assert self.indexes is not None
        Ntrajs = len(self.buffer)
        assert Ntrajs > 0

        data = torch.vstack(self.buffer)  #  -> (samples, data_dim)
        Nsamp = data.shape[0]
        if self.verbose:
            print('The replay buffer has %d samples'%Nsamp)

        if self.seed is not None:
            np.random.seed(self.seed)
            # torch.manual_seed(self.seed)
            self.seed += 1
        # randinds = torch.randperm(Nsamp)
        if Nsamp <= batch_size:
            randinds = np.arange(Nsamp)
        else:
            randinds = np.random.choice(Nsamp, size=batch_size, replace=False)


        s = data[:, self.indexes[0][0]:self.indexes[0][1]]  # (Nsamp, obs_dim)
        a = data[:, self.indexes[1][0]:self.indexes[1][1]].to(torch.int64)  # (Nsamp, act_dim)
        snext = data[:, self.indexes[2][0]:self.indexes[2][1]]  # (Nsamp, obs_dim)
        r = data[:, self.indexes[3][0]:self.indexes[3][1]]  # (Nsamp, 1)
        end = data[:, self.indexes[4][0]:self.indexes[4][1]]  # (Nsamp, 1)

        data_tuple = (s[randinds, :], a[randinds, :], snext[randinds, :],
                      r[randinds, :], end[randinds, :])
        return data_tuple




