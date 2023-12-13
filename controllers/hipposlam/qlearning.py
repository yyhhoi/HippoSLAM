"""tabular_qlearning controller."""

# You may need to import some classes of the controller module. Ex:
from controller import Robot, Motor
from controller import Supervisor
import gym
import numpy as np
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import DQN
# import os
#
# os.environ["WEBOTS_HOME"] = r"C:\Users\Hoi\AppData\Local\Programs\Webots"
# os.environ["PYTHONPATH"] += r"${WEBOTS_HOME}\lib\controller\python"
# os.environ["PYTHONIOENCODING"] += r"UTF-8"


class SimpleQ(Supervisor, gym.Env):
    def __init__(self, max_episode_steps=1000):
        super().__init__()

        # Open AI Gym generic
        high = np.array(
            [3, 3],
            dtype=np.float64
        )
        
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float64)
        self.state = None
        self.spec = gym.envs.registration.EnvSpec(id='WeBotsQ-v0', max_episode_steps=max_episode_steps)

        # Supervisor
        self.supervis = self.getSelf()

        # Robot
        self.leftMotor = self.getDevice('rear left wheel motor')
        self.rightMotor = self.getDevice('rear right wheel motor')
        self.MAX_SPEED = 10*2 # 6.28 * 2

        # Action - 'Forward', 'back', 'left', 'right'
        self.action_space = gym.spaces.Discrete(4)
        self.move_d = 1 * self.MAX_SPEED
        self._action_to_direction = {
            0: np.array([self.move_d, self.move_d]),
            1: np.array([-self.move_d, -self.move_d]),
            2: np.array([self.move_d, self.move_d*0.4]),
            3: np.array([self.move_d*0.4, self.move_d])
        }

        # Environment specific
        self.__timestep = int(self.getBasicTimeStep())

        # Tools
        self.keyboard = self.getKeyboard()
        self.keyboard.enable(self.__timestep)

    def wait_keyboard(self):
        while self.keyboard.getKey() != ord('Y'):
            super().step(self.__timestep)


    def get_obs(self):
        
        return np.array(self.supervis.getField('translation').getSFVec3f())[:2]
    
    def reset(self):
        # Reset the simulation
        self.simulationResetPhysics()
        self.simulationReset()
        super().step(self.__timestep)

        # Reset position and velocity
        translation_field = self.supervis.getField('translation')
        translation_field.setSFVec3f([0, -0.5, 0])
        self.leftMotor.setVelocity(0)
        self.rightMotor.setVelocity(0)
        self.leftMotor.setPosition(float('inf'))
        self.rightMotor.setPosition(float('inf'))

        # Internals
        super().step(self.__timestep)

        # Open AI Gym generic
        return self.get_obs()

    def step(self, action):
        # # Execute the action
        # translation_field = self.supervis.getField('translation')
        # current_pos = np.array(translation_field.getSFVec3f())
        # new_pos = current_pos + self._action_to_direction[action]
        # new_x, new_y = new_pos[0], new_pos[1]
        #
        # if (new_x > 2) or (new_x < -2) or (new_y > 2) or (new_y < -2):
        #     # print('Current pos: ', np.around(self.get_obs(), 4))
        #     # print('Next pos: ', np.around(new_pos, 4))
        #     # print('Pass')
        #     pass
        # else:
        #     translation_field.setSFVec3f(list(new_pos))
        leftd, rightd = self._action_to_direction[action]
        self.leftMotor.setVelocity(leftd)
        self.rightMotor.setVelocity(rightd)


        super().step(self.__timestep * 25)

        new_x, new_y = self.get_obs()
        # Done
        done = bool(
            ((new_x < 2.4) & (new_x > 0)) & ((new_y < 2.4) & (new_y > 0))
        )

        # Reward
        reward = 1 if done else -1
        if done:
            print('Goal reached!')

        return self.get_obs(), reward, done, {}


    def step_unmoved(self):
        return super().step(self.__timestep)


def main():
    # Initialize the environment

    env = SimpleQ()
    env.reset()
    env.step(0)
    env.step(1)
    env.step(2)
    env.step(3)
    # check_env(env)

    # # Train
    # model = DQN('MlpPolicy', env, verbose=1, exploration_initial_eps=0.5, learning_rate=0.01)
    # model.learn(total_timesteps=1e5)
    #
    # # Replay
    # print('Training is finished, press `Y` for replay...')
    # env.wait_keyboard()
    #
    # obs = env.reset()
    # for _ in range(100000):
    #     action, _states = model.predict(obs)
    #     obs, reward, done, info = env.step(action)
    #     print(obs, reward, done, info)
    #     if done:
    #         obs = env.reset()

    # print(env.get_obs())
    # steps = [0] * 6 + [1] * 4
    # for ai in steps:
    #     new_pos, reward, done, _ = env.step(ai)
    #     print('State ori:', np.around(new_pos, 4),
    #           '\nStateReal: ', np.around(env.get_obs(), 4),
    #           '\nReward: ', reward,
    #           '\nDone: ', done)



if __name__ == '__main__':
    main()
