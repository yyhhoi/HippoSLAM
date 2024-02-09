from os.path import join

import numpy as np
import torch
from matplotlib import pyplot as plt

from hipposlam.ReinforcementLearning import AWAC
from hipposlam.Replay import ReplayMemoryAWAC
from hipposlam.Networks import MLP
from hipposlam.utils import read_pickle


# Paths
save_dir = join('data', 'Omniscient')
offline_data_pth = join(save_dir, 'naive_controller_data.pickle')
save_chpt_pth = join(save_dir, 'NaiveControllerLayer3CHPT.pt')
loss_records_pth = join(save_dir, 'NaiveControllerLayer3LOSS.png')

# Paramters
obs_dim = 6
act_dim = 3
gamma = 0.99
lam = 1
use_adv = True
batch_size = 1024
max_buffer_size = 20000
Niters = 50000

# Initialize Networks
critic = MLP(obs_dim, act_dim, [128, 128, 128])
critic_target = MLP(obs_dim, act_dim, [128, 128, 128])
actor = MLP(obs_dim, act_dim, [128, 128, 64])


# Initialize Replay buffer
memory = ReplayMemoryAWAC(max_size=max_buffer_size)
datainds = np.cumsum([0, obs_dim, 1, obs_dim, 1, 1])
memory.specify_data_tuple(s=(datainds[0], datainds[1]), a=(datainds[1], datainds[2]),
                          snext=(datainds[2], datainds[3]), r=(datainds[3], datainds[4]),
                          end=(datainds[4], datainds[5]))

# Initialize agent
agent = AWAC(critic, critic_target, actor,
             lam=lam,
             gamma=gamma,
             num_action_samples=10,
             critic_lr=2e-4,  # 5e-4
             actor_lr=2e-4,  # 5e-4
             weight_decay=5e-5,
             use_adv=True)

# Load offline data and add to replay buffer
data = read_pickle(offline_data_pth)
memory.from_offline_np(data['traj'])  # (time, data_dim=15)
print('Replay buffer has %d samples'%(len(memory)))

# Training
agent.train()
closs_list, aloss_list = [], []
for i in range(Niters):
    _s, _a, _snext, _r, _end = memory.sample(batch_size)
    critic_loss = agent.update_critic(_s, _a, _snext, _r, _end)
    actor_loss = agent.update_actor(_s, _a)
    closs = critic_loss.item()
    aloss = actor_loss.item()
    closs_list.append(closs)
    aloss_list.append(aloss)
    if i % 100 == 0:
        print('Training %d/%d. C/A Loss = %0.6f, %0.6f' % (i, Niters, closs, aloss))

# Save checkpoints
agent.save_checkpoint(save_chpt_pth)

# Plot loss records
fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
ax[0].plot(closs_list, linewidth=1)
ax[0].set_ylabel('Critic Loss')
ax[1].plot(aloss_list, linewidth=1)
ax[1].set_ylabel('Actor Loss')
fig.tight_layout()
fig.savefig(loss_records_pth, dpi=300)