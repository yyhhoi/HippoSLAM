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

class A2C(nn.Module):
    def __init__(self, critic,
                        actor,
                        critic_target=None,
                        gamma=0.99,
                        critic_lr=1e-3,
                        actor_lr=1e-3,
                        weight_decay=0):
        super(A2C, self).__init__()
        self.critic = critic
        self.critic_target = critic_target
        if self.critic_target:
            self.critic_target.load_state_dict(critic.state_dict())
        self.critic_opt = torch.optim.Adam(params=self.critic.parameters(), lr=critic_lr, weight_decay=weight_decay)
        self.actor = actor
        self.actor_opt = torch.optim.Adam(params=self.actor.parameters(), lr=actor_lr, weight_decay=weight_decay)
        self.gamma = gamma



    def get_action(self, state):
        """

        Parameters
        ----------
        state : Tensor (N, obs_dim) torch.float32

        Returns
        -------
        Action : Tensor (N, 1) torch.int64

        """

        with torch.no_grad():
            logits = self.actor(state)  # (N, obs_dim) -> (N, act_dim)
            dist = Categorical(logits=logits)
            cosa, sina = state[0, 4].item(), state[0, 5].item()
            print('x=%0.3f, y=%0.3f, cos = %0.2f, sin = %0.4f, angle = %0.4f'%(state[0, 0].item(), state[0, 1].item(), cosa, sina, np.angle(cosa + 1j * sina)))
            prob = torch.exp(dist.log_prob(torch.tensor([0, 1, 2])))
            print('F = %0.4f, L = %0.4f, R = %0.4f'%(prob[0], prob[1], prob[2]))
            return dist.sample(sample_shape=[1]).T  # -> (N, 1)

    def update_networks(self, state, action, G):
        """
        Parameters
        ----------
        state : Tensor. (N, obs_dim). torch.float32
        action : Tensor. (N, 1). torch.int64
        G : torch.tensor. (N, 1). torch.float32
        """

        critic_loss = self.update_critic(state, G)
        actor_loss = self.update_actor(state, action, G)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        if self.critic_target:
            soft_update(self.critic, self.critic_target, 5e-3)
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        return critic_loss, actor_loss

    def update_critic(self, state, G):
        """

        Parameters
        ----------
        state : Tensor. (N, obs_dim). torch.float32
        action : Tensor. (N, 1). torch.int64

        Returns
        -------
        loss : Tensor. Scalar
        """
        v = self.critic(state)  # -> (N, 1) float
        A = G - v  # -> (N, 1)
        critic_loss = torch.sum(torch.square(A))  # -> scalar
        return critic_loss

    def update_actor(self, state, action, G):
        """

        Parameters
        ----------
        state : Tensor. (N, obs_dim). torch.float32
        action : Tensor. (N, 1). torch.int64
        G : torch.tensor. (N, 1). torch.float32

        Returns
        -------
        loss : Tensor. Scalar
        """

        logits = self.actor(state)  # (N, obs_dim) -> (N, act_dim)
        log_prob = Categorical(logits=logits).log_prob(action.squeeze()).view(-1, 1)  # -> (N, 1)
        with torch.no_grad():
            if self.critic_target:
                v = self.critic_target(state)  # -> (N, 1) float
            else:
                v = self.critic(state)  # -> (N, 1) float
            A = G - v  # -> (N, 1)
        loss = (log_prob * A).mean() * -1  # -> scalar
        return loss

    def save_checkpoint(self, pth):
        ckpt_dict = {
            'critic_state_dict': self.critic.state_dict(),
            'actor_state_dict': self.actor.state_dict(),
            'critic_opt_state_dict': self.critic_opt.state_dict(),
            'actor_opt_state_dict': self.actor_opt.state_dict(),
            'gamma': self.gamma,
        }
        if self.critic_target:
            ckpt_dict['critic_target_state_dict'] = self.critic_target.state_dict()
        torch.save(ckpt_dict, pth)

    def load_checkpoint(self, pth):
        checkpoint = torch.load(pth)
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic_opt.load_state_dict(checkpoint['critic_opt_state_dict'])
        self.actor_opt.load_state_dict(checkpoint['actor_opt_state_dict'])
        self.gamma = checkpoint['gamma']
        if self.critic_target:
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])



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
                 clip_max_norm: float = 1.0,
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
        self.clip_max_norm = clip_max_norm
        self.use_adv = use_adv

    def get_action(self, state, num_samples: int = 1):
        with torch.no_grad():
            logits = self.actor(state)  # (N, obs_dim) -> (N, act_dim)
            dist = Categorical(logits=logits)
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
        loss : torch.tensor (scalar)

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
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.clip_max_norm)
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
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_max_norm)
        self.actor_opt.step()
        return loss

    def save_checkpoint(self, pth):
        ckpt_dict = {
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_state_dict': self.actor.state_dict(),
            'critic_opt_state_dict': self.critic_opt.state_dict(),
            'actor_opt_state_dict': self.actor_opt.state_dict(),
            'lam': self.lam,
            'tau': self.tau,
            'gamma': self.gamma,
            'num_action_samples': self.num_action_samples,
            'clip_max_norm': self.clip_max_norm,
            'use_adv': self.use_adv,
        }
        torch.save(ckpt_dict, pth)

    def load_checkpoint(self, pth):
        checkpoint = torch.load(pth)
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic_opt.load_state_dict(checkpoint['critic_opt_state_dict'])
        self.actor_opt.load_state_dict(checkpoint['actor_opt_state_dict'])
        self.lam = checkpoint['lam']
        self.tau = checkpoint['tau']
        self.gamma = checkpoint['gamma']
        self.num_action_samples = checkpoint['num_action_samples']
        self.clip_max_norm = checkpoint['clip_max_norm']
        self.use_adv = checkpoint['use_adv']




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

class BioQ:

    def __init__(self, N=100, act_dim=3, lr=0.025, beta=5, gamma=0.9):

        self.N = N
        self.act_dim = act_dim
        self.lr = lr
        self.beta = beta
        self.gamma = gamma

        self.m = np.ones((self.N, self.act_dim))
        # self.m = np.random.uniform(-0.1, 0.1, size=(self.N, self.act_dim))
        self.m = self.m / np.linalg.norm(self.m, axis=1, keepdims=True)
        self.l = np.zeros(self.N)
        self.w = np.zeros(self.N)

        self.iter = 1
        self.counts = np.ones((self.N, self.act_dim))

    def get_action(self, state:int):
        # ucb = np.sqrt(np.log(self.iter)/self.counts[state, :])
        # print('Counts = ', list(np.around(self.counts[state, :], 3)))
        expo = self.beta * self.m[state, :]
        aprob = self._softmax(expo)
        a = int(np.random.choice(np.arange(self.act_dim), p=aprob))
        self.iter += 1
        self.counts[state, a] += 1
        return a, aprob

    def expand(self, state):
        if state > self.N-1:
            dnum = state+1 - self.N
            self.l = np.concatenate([self.l, np.zeros(dnum)])
            self.w = np.concatenate([self.w, np.zeros(dnum)])
            # self.m = np.vstack([self.m, np.random.uniform(-0.1, 0.1, size=(dnum, self.act_dim))])
            self.m = np.vstack([self.m, np.ones((dnum, self.act_dim))])
            self.counts = np.vstack([self.counts, np.ones(-1, 1, size=(dnum, self.act_dim))])
    def update(self, state, action, action_prob, reward, next_state, done):
        self.l[state] = self.l[state] * self.gamma
        self.l[state] = self.l[state] + 1

        # if done:
        #     td = reward - self.w[state]
        # else:
        td = self.w[next_state] + reward - self.w[state]

        self.w = self.w + self.lr * td * self.l

        avec = np.zeros(self.act_dim)
        avec[action] = 1
        self.m[state, :] = self.m[state, :] + self.lr * td * (avec - action_prob)
        self.m[state, :] = self.m[state, :]/np.linalg.norm(self.m[state, :])

    def _softmax(self, m):
        aprob = np.exp(m) / np.sum(np.exp(m))
        return aprob


