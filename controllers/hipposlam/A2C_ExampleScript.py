# Paths
from os.path import join

import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from hipposlam.Networks import ActorModel, ValueCriticModel
from hipposlam.ReinforcementLearning import TensorConvertor, model_train_A2C, \
    compute_discounted_returns

env = gym.make("CartPole-v1")

obs_dim = 4
act_dim = 2
data_dim = obs_dim + act_dim + obs_dim + 2
gamma = 0.99
beta = 0.
lamb = 1
TC = TensorConvertor(obs_dim, act_dim)
critic = ValueCriticModel(obs_dim)
actor = ActorModel(obs_dim, act_dim)
optimizer_C = torch.optim.Adam(critic.parameters(), lr=1e-3, weight_decay=1e-5)
optimizer_A = torch.optim.Adam(actor.parameters(), lr=1e-3, weight_decay=1e-5)

Niters = 500

dims = np.cumsum([0, obs_dim, act_dim, obs_dim, 1, 1])
rewards_alleps = []
maxtimesteps = 300
for i in range(Niters):
    print('Episode %d/%d' % (i, Niters))
    s = env.reset()
    s = torch.from_numpy(s).reshape(1, -1).to(torch.float32)

    # ================== Model Unroll ===============================

    done = False
    alls = []
    alla = []
    allr = []
    allend = []
    total_reward = 0
    actor.eval()
    critic.eval()
    t = 0
    while (done is False) and (t < maxtimesteps):
        with torch.no_grad():
            aprob = actor(s)  # (1, obs_dim) -> (1, act_dim)
            a = TC.select_action_tensor(aprob)  # (1, act_dim) -> (1, ) int
            s_next, r, done, _ = env.step(a.item())
        a_onehot = np.zeros(act_dim)
        a_onehot[a.item()] = 1
        alls.append(s.squeeze().numpy())
        alla.append(a_onehot)
        allr.append(r)
        allend.append(done)
        s = torch.from_numpy(s_next).reshape(1, -1).to(torch.float32)
        total_reward += r
        t += 1
    with torch.no_grad():
        last_v = critic(s)


    rewards_alleps.append(total_reward)
    alls = torch.from_numpy(np.vstack(alls)).to(torch.float32)
    alla = torch.from_numpy(np.vstack(alla)).to(torch.float32)
    allG = compute_discounted_returns(allr, allend, last_v.squeeze().detach().item(), gamma)
    allG = torch.from_numpy(allG).to(torch.float32)
    # ================== Model Train ===============================
    actor.train()
    critic.train()
    all_closs, all_aloss = [], []
    closs, aloss = model_train_A2C(alls, alla, allG, actor, critic)

    optimizer_C.zero_grad()
    closs.backward()
    optimizer_C.step()

    optimizer_A.zero_grad()
    aloss.backward()
    optimizer_A.step()

    clossmu = closs.item()
    alossmu = aloss.item()

plt.scatter(np.arange(len(rewards_alleps)), rewards_alleps, s=2)
plt.title("Total reward per episode (episodic)")
plt.ylabel("reward")
plt.xlabel("episode")
plt.show()