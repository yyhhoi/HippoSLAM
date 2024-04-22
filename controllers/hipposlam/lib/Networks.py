import torch
from torch import nn
class ActorModel(nn.Module):
    def __init__(self, obs_dim, act_dim, logit=False):
        super(ActorModel, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = 64
        if logit:
            self.actor_model = torch.nn.Sequential(
                nn.Linear(self.obs_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.act_dim),
            )
        else:
            self.actor_model = torch.nn.Sequential(
                nn.Linear(self.obs_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.act_dim),
                nn.Softmax(dim=1)
            )

    def forward(self, x):
        aprob = self.actor_model(x)
        return aprob

class ValueCriticModel(nn.Module):
    def __init__(self, obs_dim):
        super(ValueCriticModel, self).__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = 64
        self.critic_model = torch.nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )
    def forward(self, s):
        v = self.critic_model(s)  # (N, obs_dim) -> (N, 1)
        return v


class QCriticModel(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(QCriticModel, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = 64
        self.critic_model = torch.nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
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


class MLP(nn.Module):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 num_neurons: list = [64, 32],
                 hidden_act: str = 'ReLU',
                 out_act: str = 'Identity'):
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_neurons = num_neurons
        self.hidden_act = getattr(nn, hidden_act)()
        self.out_act = getattr(nn, out_act)()

        input_dims = [input_dim] + num_neurons
        output_dims = num_neurons + [output_dim]

        self.layers = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(input_dims, output_dims)):
            is_last = True if i == len(input_dims) - 1 else False
            self.layers.append(nn.Linear(in_dim, out_dim))
            if is_last:
                self.layers.append(self.out_act)
            else:
                self.layers.append(self.hidden_act)

    def forward(self, xs):
        for layer in self.layers:
            xs = layer(xs)
        return xs
