# Paths
from os.path import join

import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from hipposlam.Networks import ActorModel, QCriticModel, MLP
from hipposlam.Replay import Replayer, ReplayMemoryCat
from hipposlam.ReinforcementLearning import AWAC


env = gym.make("CartPole-v1")

obs_dim = 4
act_dim = 2

gamma = 0.9
lam = 1.0
# critic = QCriticModel(obs_dim, act_dim)
# critic_target = QCriticModel(obs_dim, act_dim)
# actor = ActorModel(obs_dim, act_dim, logit=True)

critic = MLP(4, 2,
           num_neurons=[128,128],
           out_act='ReLU')
critic_target = MLP(4, 2,
                  num_neurons=[128,128],
                  out_act='ReLU')
actor = MLP(4, 2, num_neurons=[128,64])


batch_size = 1024
memory = ReplayMemoryCat(max_size=500000)
datainds = np.cumsum([0, obs_dim, 1, obs_dim, 1, 1])
memory.specify_data_tuple(s=(datainds[0], datainds[1]), a=(datainds[1], datainds[2]),
                          snext=(datainds[2], datainds[3]), r=(datainds[3], datainds[4]), end=(datainds[4], datainds[5]))
agent = AWAC(critic, critic_target, actor,
             lam=lam,
             gamma=gamma,
             num_action_samples=10,
             critic_lr=3e-4,
             actor_lr=3e-4,
             weight_decay=0,
             use_adv=True)

Niters = 8000
awac_cum_rs = []
critic_losses, actor_losses = [], []
for n_epi in range(Niters):
    print('\rEpisode %d/%d'%(n_epi, Niters), flush=True, end='')
    s, _ = env.reset()
    cum_r = 0
    traj = []
    truncated = False
    while True and (truncated is False):

        s = torch.from_numpy(s).to(torch.float32).view(-1, obs_dim) # (1, obs_dim)
        a = int(agent.get_action(s).squeeze())  # tensor (1, 1) -> int
        snext, r, done, truncated, info = env.step(a)
        experience = torch.concat([
            s.squeeze(), torch.tensor([a]), torch.tensor(snext), torch.tensor([r]), torch.tensor([done])
        ])
        memory.push(experience)
        s = snext
        cum_r += 1
        if done:
            awac_cum_rs.append(cum_r)
            break
    if len(memory) >= batch_size:
        _s, _a, _snext, _r, _end = memory.sample(batch_size)
        critic_loss = agent.update_critic(_s, _a, _snext, _r, _end)
        actor_loss = agent.update_actor(_s, _a)
        critic_losses.append(critic_loss.detach())
        actor_losses.append(actor_loss.detach())
        # for _s, _a, _snext, _r, _end in memory.sample(batch_size):
        #     critic_loss = agent.update_critic(_s, _a, _snext, _r, _end)
        #     actor_loss = agent.update_actor(_s, _a)
        #     critic_losses.append(critic_loss.detach())
        #     actor_losses.append(actor_loss.detach())
        #     break




fig, ax = plt.subplots(1, 3, figsize=(10, 3))
ax[0].plot(awac_cum_rs)
ax[0].set_ylabel('Cum R')
ax[1].plot(critic_losses)
ax[1].set_ylabel('Critic loss')
ax[2].plot(actor_losses)
ax[2].set_ylabel('Actor loss')
fig.savefig('AWAC.png', dpi=200)
plt.show()
