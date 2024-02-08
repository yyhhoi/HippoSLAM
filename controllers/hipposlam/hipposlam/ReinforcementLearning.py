import numpy as np
import torch
from torch import nn
from typing import Tuple
import torch.nn.functional as F
from torch.distributions import Categorical

def soft_update(net, net_target, tau):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)


def compute_discounted_returns(r, end, q_val, gamma=0.99):
    """

    Parameters
    ----------
    r : iterable
        1-d array with shape (t, ). Immediate reward at time t of the episode after the action.
    end : iterable
        1-d array with shape (t, ). Floating number of 0.0 or 1.0 indicating if the action results in episode termination.
    q_val : float
        Value estimate at the last time step. = r[-1] if end[-1] == 1. Else, it serves as estimate for bootstrapping.
    gamma : float
        Future discounting factor.

    Returns
    -------

    """
    N = len(r)
    allG = np.zeros(N)
    for i in range(N-1, -1, -1):
        q_val = r[i] + gamma * q_val * (1-end[i])
        allG[i] = q_val
    return allG

from random import sample


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
        print('The replay buffer has %d samples'%Nsamp)

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


def model_train_A2C(s, a, G, actor, critic):
    """

    Parameters
    ----------
    s : torch.tensor. (N, obs_dim). torch.float32
    a : torch.tensor. (N, act_dim). torch.float32
    G : torch.tensor. (N, ). torch.float32
        Return. Approximation of the action values sampled by policy.
    actor
    critic
    Returns
    -------

    """
    v = critic(s)  # -> (N, 1) float
    aprob = actor(s)  # -> (N, act_dim) float
    A = G - v.squeeze()  # -> (N,)
    critic_loss = torch.sum(torch.square(A))  # -> scalar
    actor_loss = -torch.sum(torch.log(torch.sum(aprob * a, dim=1) + 1e-7) * A.detach())  # -> scalar
    return critic_loss, actor_loss


class AWAC(nn.Module):
    # Courtesy to https://github.com/Junyoungpark/Pytorch-AWAC
    def __init__(self,
                 critic: nn.Module,  # Q(s,a): Map (N, obs_dim) to (N, act_dim)
                 critic_target: nn.Module,
                 actor: nn.Module,  # pi(a|s): Map (N, obs_dim) to (N, act_dim)
                 lam: float = 0.3,  # Lagrangian parameter
                 tau: float = 5 * 1e-3,
                 gamma: float = 0.9,
                 num_action_samples: int = 1,
                 critic_lr: float = 3 * 1e-4,
                 actor_lr: float = 3 * 1e-4,
                 weight_decay: float = 1e-5,
                 use_adv: bool = False):
        super(AWAC, self).__init__()

        self.critic = critic
        self.critic_target = critic_target
        self.critic_target.load_state_dict(critic.state_dict())
        self.critic_opt = torch.optim.Adam(params=self.critic.parameters(), lr=critic_lr, weight_decay=weight_decay)

        self.actor = actor
        self.actor_opt = torch.optim.Adam(params=self.actor.parameters(), lr=actor_lr, weight_decay=weight_decay)

        assert lam > 0, "Lagrangian parameter 'lam' requires to be strictly larger than 0.0"
        self.lam = lam
        self.tau = tau
        self.gamma = gamma
        self.num_action_samples = num_action_samples
        self.use_adv = use_adv

    def get_action(self, state, num_samples: int = 1):
        logits = self.actor(state)  # (N, obs_dim) -> (N, act_dim)
        try:
            dist = Categorical(logits=logits)
        except:
            breakpoint()
        return dist.sample(sample_shape=[num_samples]).T  # -> (N, 1)

    def update_critic(self, state, action, next_states, reward, dones):
        """

        Parameters
        ----------
        state : tensor (float32)
            Shape = (N, obs_dim) tensor.
        action : tensor (int64)
            Shape = (N, 1) tensor with index numbers of the actions.
        next_states : tensor (float32)
            Shape = (N, obs_dim) tensor.
        reward : tensor (float32)
            Shape = (N, 1)
        dones : tensor (float32)
            Shape = (N, 1)

        Returns
        -------

        """

        with torch.no_grad():
            # breakpoint()
            qs = self.critic_target(next_states)  # (N, obs_dim) -> (N, act_dim)
            sampled_as = self.get_action(next_states, self.num_action_samples)  # (N, obs_dim) -> (N, N_action_samples)
            mean_qsa = qs.gather(1, sampled_as).mean(dim=-1, keepdims=True)  # (N, N_action_samples) -> (N, 1)
            q_target = reward + self.gamma * mean_qsa * (1 - dones)  # (N, 1)
        q_val = self.critic(state).gather(1, action)  # (N, obs_dim) and (N, 1) -> (N, 1)
        loss = F.mse_loss(q_val, q_target)  # (N, 1) -> scalar
        self.critic_opt.zero_grad()
        loss.backward()
        self.critic_opt.step()

        # target network update
        soft_update(self.critic, self.critic_target, self.tau)

        return loss

    def update_actor(self, state, action):
        """

        Parameters
        ----------
        state : tensor (float32)
            Shape = (N, obs_dim) tensor.
        action : tensor (int64)
            Shape = (N, 1) tensor with index numbers of the actions.

        Returns
        -------

        """
        logits = self.actor(state)  # (N, obs_dim) -> (N, act_dim)
        log_prob = Categorical(logits=logits).log_prob(action.squeeze()).view(-1, 1)  # -> (N, 1)

        with torch.no_grad():
            if self.use_adv:
                qs = self.critic_target(state)  # (N, obs_dim) -> (N, act_dim)
                action_probs = F.softmax(logits, dim=-1)  # (N, act_dim) -> (N, act_dim)
                vs = (qs * action_probs).sum(dim=-1, keepdims=True)  # -> (N, 1)
                qas = qs.gather(1, action)  # -> (N, 1)
                adv = qas - vs  # -> (N, 1)
            else:
                adv = self.critic_target(state).gather(1, action)

            weight_term = torch.exp(1.0 / self.lam * adv)  # -> (N, 1)

        loss = (log_prob * weight_term).mean() * -1  # -> scalar

        self.actor_opt.zero_grad()
        loss.backward()
        self.actor_opt.step()
        return loss


class DQN(nn.Module):

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 qnet: nn.Module,
                 qnet_target: nn.Module,
                 lr: float,
                 gamma: float,
                 epsilon: float):
        """
        :param state_dim: input state dimension
        :param action_dim: action dimension
        :param qnet: main q network
        :param qnet_target: target q network
        :param lr: learning rate
        :param gamma: discount factor of MDP
        :param epsilon: E-greedy factor
        """

        super(DQN, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.qnet = qnet
        self.lr = lr
        self.gamma = gamma
        self.opt = torch.optim.Adam(params=self.qnet.parameters(), lr=lr)
        self.register_buffer('epsilon', torch.ones(1,) * epsilon)

        # target network related
        qnet_target.load_state_dict(qnet.state_dict())
        self.qnet_target = qnet_target
        self.criteria = nn.SmoothL1Loss()

    def get_action(self, state):
        qs = self.qnet(state)
        prob = np.random.uniform(0.0, 1.0, 1)
        if torch.from_numpy(prob).float() <= self.epsilon:  # random
            action = np.random.choice(range(self.action_dim))
        else:  # greedy
            action = qs.argmax(dim=-1)
        return int(action)

    def update(self, state, action, reward, next_state, done):
        s, a, r, ns = state, action, reward, next_state

        # compute Q-Learning target with 'target network'
        with torch.no_grad():
            q_max, _ = self.qnet_target(ns).max(dim=-1, keepdims=True)
            q_target = r + self.gamma * q_max * (1 - done)

        q_val = self.qnet(s).gather(1, a)
        loss = self.criteria(q_val, q_target)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()


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
        # v = critic.compute_value(s, aprob)  # -> (N,) float
        v = torch.sum(qall, dim=1)/act_dim # -> (N,) float
        A = q - v  # -> (N, ) float

    y = r + gamma * q_next  # -> (N,)
    critic_loss = torch.sum(torch.square(q - y))  # -> scalar
    entropy = beta * - torch.sum(torch.sum(torch.log(aprob + 1e-5) * aprob, dim=1), dim=0)
    if AWAC:
        actor_loss = -torch.sum(torch.log(torch.sum(aprob * a, dim=1)) * torch.exp(A/lamb)) - entropy  # -> scalar
    else:
        actor_loss = -torch.sum(torch.log(torch.sum(aprob * a, dim=1)) * A) - entropy # -> scalar


    return critic_loss, actor_loss



