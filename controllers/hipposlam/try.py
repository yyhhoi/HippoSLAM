import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
from hipposlam.ReinforcementLearning import BioQ
from stable_baselines3 import PPO
# from stable_baselines3.common.env_util import env

# vec_env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)
#
# model = PPO("MlpPolicy", vec_env, verbose=1)
# model.learn(total_timesteps=20000)
#
# Niters = 1000
# allt = np.zeros(Niters)
# allr = np.zeros(Niters)
# for i in tqdm(range(Niters)):
#     done = False
#     obs, _ = vec_env.reset()
#     t = 0
#     while (done is False):
#         action, _states = model.predict(obs)
#
#         obsnext, rewards, done, term, info = vec_env.step(int(action))
#         obs = obsnext
#         t += 1
#     allt[i] = t
#     allr[i] = rewards
#
# print('Win rate = ', allr.mean())
# fig, ax = plt.subplots(1, 2, figsize=(8, 4))
#
# ax[0].plot(allt)
# ax[1].plot(allr)
# ax[1].plot(gaussian_filter1d(allr, sigma=100))
#
# fig.savefig('FrozenLake.png', dpi=200)

env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)

agent = BioQ(N = 16, act_dim=4)
Niters = 50000
allt = np.zeros(Niters)
allr = np.zeros(Niters)
for i in tqdm(range(Niters)):
    done = False
    truncated = False
    t = 0
    s, _ = env.reset()
    while (done is False) and (truncated is False) :
        a, aprob = agent.get_action(s)
        snext, r, done, truncated, info = env.step(a)
        if np.random.rand() < 0.1:
            a = np.random.randint(0, 4, size=1).squeeze()

        agent.update(int(s), int(a), aprob, r, int(snext), done)
        s = snext
        t += 1
    allt[i] = t
    allr[i] = r

print('Win rate = ', allr.mean())
fig, ax = plt.subplots(1, 2, figsize=(8, 4))

ax[0].plot(allt)
ax[1].plot(allr)
ax[1].plot(gaussian_filter1d(allr, sigma=100))

fig.savefig('FrozenLake.png', dpi=200)
