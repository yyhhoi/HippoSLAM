import numpy as np
import gym
import matplotlib.pyplot as plt
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor


class SimpleQ_CMD(gym.Env):
    def __init__(self, max_episode_steps=1000):
        super().__init__()

        # Open AI Gym generic
        high = np.array(
            [3, 3],
            dtype=np.float32
        )

        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float64)
        self.state = None
        self.spec = gym.envs.registration.EnvSpec(id='WeBotsQ-v0', max_episode_steps=max_episode_steps)



        # Action - 'up', 'right', 'down', 'left'
        self.action_space = gym.spaces.Discrete(4)
        self.move_d = 0.5
        self._action_to_direction = {
            0: np.array([0, self.move_d]),
            1: np.array([self.move_d, 0]),
            2: np.array([0, -self.move_d]),
            3: np.array([-self.move_d, 0]),
        }


    def get_obs(self):

        return self.state

    def reset(self):

        self.state = np.array([0.0, -1.0])

        # Open AI Gym generic
        return self.get_obs()

    def step(self, action):
        # Execute the action
        new_state = self.state + self._action_to_direction[action]

        new_x, new_y = new_state

        if (new_x > 2) or (new_x < -2) or (new_y > 2) or (new_y < -2):
            # print('Current pos: ', np.around(self.get_obs(), 4))
            # print('Next pos: ', np.around(new_pos, 4))
            # print('Pass')
            pass
        else:
            self.state = new_state


        # Done
        done = bool(
            ((self.state[0] < 2.4) & (self.state[0] > 0)) & ((self.state[1] < 2.4) & (self.state[1] > 0))
        )

        # Reward
        reward = 1 if done else -1

        return self.get_obs(), reward, done, {}


def train():
    env = SimpleQ_CMD()
    check_env(env)

    env = Monitor(env, '/log/')


    # Train
    model = DQN('MlpPolicy', env, verbose=1, exploration_initial_eps=0.5)
    model.learn(total_timesteps=1e5)

    model.save('SimpleQ_CMD')

    eplen = np.array(env.get_episode_lengths())
    np.save('eplen.npy', eplen)




def replay():
    env = SimpleQ_CMD()
    obs = env.reset()

    model = DQN.load('SimpleQ_CMD', env=env)

    all_obs = []
    for _ in range(100000):
        action, _states = model.predict(obs)
        obs, reward, done, _ = env.step(int(action))
        print('Obs = ',obs, ', Action = ', action, ', Reward = ',  reward, ', Done=', done)
        all_obs.append(obs)
        if done:
            break

    eplen = np.load('eplen.npy')
    plt.plot(np.log10(eplen))
    plt.show()


    all_obs = np.stack(all_obs)
    plt.plot(all_obs[:, 0], all_obs[:, 1])
    plt.show()

if __name__ == '__main__':
    import os

    os.environ['KMP_DUPLICATE_LIB_OK'] = "True"
    train()
    # replay()