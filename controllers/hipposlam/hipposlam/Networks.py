import torch
from torch import nn
class ActorModel(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(ActorModel, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = 64
        self.actor_model = torch.nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(self.hidden_dim, self.act_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        aprob = self.actor_model(x)
        return aprob

class CriticModel(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(CriticModel, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = 64
        self.critic_model = torch.nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(self.hidden_dim, self.act_dim),
        )

    def forward(self, svec):
        q = self.critic_model(svec)  # (N, obs_dim) -> (N, act_dim)
        return q


    def compute_value(self, svec, aprob):
        with torch.no_grad():
            q_all = self.forward(svec)  # (N, obs_dim) -> (N, act_dim)
            v = torch.sum(q_all * aprob, dim=1)  # -> (N, )

        return v