import numpy as np
import torch
from torch import nn
from typing import Tuple

class TensorConvertor:
    def __init__(self, obs_dim, act_dim):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.data_dim = self.obs_dim + self.act_dim + self.obs_dim + 2

    @staticmethod
    def select_action_tensor(aprob):
        """
        aprob : Tensor (N, act_dim), float

        Return:
        out_a : Tensor (N, ), int64
        """
        out_a = torch.multinomial(aprob, 1).reshape(aprob.shape[0])
        return out_a
    @staticmethod
    def a2onehot_tensor(a, act_dim):
        """
        a : Tensor (N, ), int
        act_dim: int

        Return:
        avec : Tensor (N, act_dim), float
        """
        avec = nn.functional.one_hot(a, act_dim).float()
        return avec

class DataLoader:
    def __init__(self, buffer=None, batch_size=64, max_buffer_size=100, seed=None):
        if buffer:
            self.buffer = buffer
        else:
            self.buffer = []
        self.batch_size = batch_size
        self.max_buffer_size = max_buffer_size  # Number of episodes.
        self.indexes = None
        self.specify_data_tuple((0, 6), (6, 9), (9, 15), (15, 16), (16, 17))  # for OmniscientLearner()
        self.seed = seed

    def append(self, traj):
        """

        Parameters
        ----------
        traj : ndarray
            2-d numpy array with (time, data_dim). data dim is the shape of the (s, a, s', r, end).

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

    def sample(self):

        assert self.indexes is not None
        Ntrajs = len(self.buffer)
        assert Ntrajs > 0

        all_trajs = np.vstack(self.buffer)  #  -> (samples, data_dim)
        Nsamp = all_trajs.shape[0]
        # print('The replay buffer has %d samples'%Nsamp)

        if self.seed is not None:
            np.random.seed(self.seed)
            self.seed += 1
        randinds = np.random.permutation(Nsamp)

        data = torch.from_numpy(all_trajs).to(torch.float32)
        s = data[:, self.indexes[0][0]:self.indexes[0][1]]
        a = data[:, self.indexes[1][0]:self.indexes[1][1]]
        snext = data[:, self.indexes[2][0]:self.indexes[2][1]]
        r = data[:, self.indexes[3][0]:self.indexes[3][1]]
        end = data[:, self.indexes[4][0]:self.indexes[4][1]]

        inds = np.arange(0, Nsamp, self.batch_size)
        inds = np.append(inds, Nsamp)

        for i in range(len(inds) - 1):
            selected_inds = randinds[inds[i]:inds[i+1]]
            data_tuple = (s[selected_inds, :], a[selected_inds, :], snext[selected_inds, :],
                          r[selected_inds, :].squeeze(), end[selected_inds, :].squeeze())

            yield data_tuple



def model_train(s, a, s_next, r, end,
                actor, critic, delayed_critic, gamma, lamb=1, beta=0.01, AWAC=False):
    """

    Parameters
    ----------
    s : torch.tensor. (N, obs_dim). torch.float32
    a : torch.tensor. (N, act_dim). torch.float32
    s_next : torch.tensor. (N, obs_dim). torch.float32
    r : torch.tensor. (N, ). torch.float32
    end : torch.tensor. (N, ). torch.float32
    actor
    critic
    delayed_critic
    gamma
    lamb
    AWAC

    Returns
    -------

    """

    act_dim = a.shape[1]
    qall = critic(s)  # -> (N, act_dim) float
    aprob = actor(s)  # -> (N, act_dim) float
    q = torch.sum(torch.square(qall * a), dim=1)  # -> (N, )

    with torch.no_grad():
        aprob_next = actor(s_next)  # -> (N, act_dim) float
        a_next = TensorConvertor.select_action_tensor(aprob_next)  # -> (N, ) int64
        a_next_vec = TensorConvertor.a2onehot_tensor(a_next, act_dim)  # -> (N, act_dim) float
        qall_next = delayed_critic(s_next)  # -> (N, act_dim) float
        q_next = torch.sum(torch.square(qall_next * a_next_vec), dim=1)  # -> (N,) float
        q_next = q_next * end # -> (N,) float

        # Compute Value
        v = critic.compute_value(s, aprob)  # -> (N,) float
        A = q - v  # -> (N, ) float

    y = r + gamma * q_next  # -> (N,)
    critic_loss = torch.sum(torch.square(q - y))  # -> scalar
    entropy = beta * - torch.sum(torch.sum(torch.log(aprob + 1e-5) * aprob, dim=1), dim=0)
    if AWAC:
        actor_loss = -torch.sum(torch.log(torch.sum(aprob * a, dim=1)) * torch.exp(A/lamb)) - entropy  # -> scalar
    else:
        actor_loss = -torch.sum(torch.log(torch.sum(aprob * a, dim=1)) * A) - entropy # -> scalar


    return critic_loss, actor_loss



