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


