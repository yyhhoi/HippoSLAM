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



class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, bottleneck_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LeakyReLU(),
        )

        self.bottleneck_mu = nn.Sequential(
            nn.Linear(self.hidden_dim, self.bottleneck_dim),
        )

        self.bottleneck_logvar = nn.Sequential(
            nn.Linear(self.hidden_dim, self.bottleneck_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.bottleneck_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.input_dim),
        )

    def reparametrize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """

        Parameters
        ----------
        x : torch.tensor
            (batch_size, input_dim) torch.flaot32. Image Embeddings.

        Returns
        -------
        y : torch.tensor
            (batch_size, input_dim) torch.flaot32. Reconstructed image Embeddings.
        mu : torch.tensor
            (batch_size, bottleneck_dim) torch.flaot32. Means.
        logvar : torch.tensor
            (batch_size, bottleneck_dim) torch.flaot32. Log variance.
        """
        x = self.encoder(x)  # (batch_size, input_dim) -> (batch_size, hidden_dim)
        mu = self.bottleneck_mu(x)  # (batch_size, hidden_dim) -> (batch_size, bottleneck_dim)
        logvar = self.bottleneck_logvar(x)  # (batch_size, hidden_dim) -> (batch_size, bottleneck_dim)
        sampled = self.reparametrize(mu, logvar)  # (batch_size, bottleneck_dim)
        y = self.decoder(sampled)  # (batch_size, bottleneck_dim) -> (batch_size, input_dim)
        return y, mu, logvar

    def sample(self, Nsamps):
        with torch.no_grad():
            sampled = torch.randn((Nsamps, self.hidden_dim))
            y = self.decoder(sampled)
        return y





