"""tabular_qlearning controller."""

# You may need to import some classes of the controller module. Ex:
from controller import Robot, Motor
from controller import Supervisor
import gym
import numpy as np
# from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
# import os
#
# os.environ["WEBOTS_HOME"] = r"C:\Users\Hoi\AppData\Local\Programs\Webots"
# os.environ["PYTHONPATH"] += r"${WEBOTS_HOME}\lib\controller\python"
# os.environ["PYTHONIOENCODING"] += r"UTF-8"


class SimpleQ(Supervisor, gym.Env):
    def __init__(self, max_episode_steps=1000):
        super().__init__()

        # X < 2

        # Open AI Gym generic
        lowBox = np.array([-7, -3, -1, -1, -1, -2 * np.pi], dtype=np.float64)
        highBox = np.array([7, 5, 1, 1, 1, 2 * np.pi], dtype=np.float64)

        self.observation_space = gym.spaces.Box(lowBox, highBox, dtype=np.float64)
        self.state = None
        self.spec = gym.envs.registration.EnvSpec(id='WeBotsQ-v0', max_episode_steps=max_episode_steps)

        # Supervisor
        self.supervis = self.getSelf()
        self.translation_field = self.supervis.getField('translation')
        self.rotation_field = self.supervis.getField('rotation')

        # Wheels
        self.leftMotor1 = self.getDevice('wheel1')
        self.leftMotor2 = self.getDevice('wheel3')
        self.rightMotor1 = self.getDevice('wheel2')
        self.rightMotor2 = self.getDevice('wheel4')
        self.MAX_SPEED = 15

        # Action - 'Forward', 'back', 'left', 'right'
        self.action_space = gym.spaces.Discrete(3)
        self.turn_steps = 10
        self.forward_steps = 20
        self.move_d = self.MAX_SPEED * 2 / 3
        self._action_to_direction = {
            0: np.array([self.move_d, self.move_d]),  # Forward
            1: np.array([-self.move_d, self.move_d]),  # Left turn
            2: np.array([self.move_d, -self.move_d]),  # Right turn
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
        rotx, roty, rotz, rota = self._get_rotation()
        x, y, z = self._get_translation()
        return np.array([x, y, rotx, roty, rotz, rota])

    def reset(self):
        # Reset the simulation
        self.simulationResetPhysics()
        self.simulationReset()
        super().step(self.__timestep)

        # Reset position and velocity
        self._set_translation(4.18, 2.82, 0.07)
        self._set_rotation(0, 0, -1, 1.57)
        for motor in [self.leftMotor1, self.leftMotor2, self.rightMotor1, self.rightMotor2]:
            motor.setVelocity(0)
            motor.setPosition(float('inf'))

        # Internals
        super().step(self.__timestep)

        # Open AI Gym generic
        return self.get_obs()

    def step(self, action):

        leftd, rightd = self._action_to_direction[action]
        self.leftMotor1.setVelocity(leftd)
        self.leftMotor2.setVelocity(leftd)
        self.rightMotor1.setVelocity(rightd)
        self.rightMotor2.setVelocity(rightd)
        numsteps = self.forward_steps if action == 0 else self.turn_steps
        for counter in range(numsteps):
            super().step(self.__timestep)

        new_x, new_y, rotx, roty, _, _ = self.get_obs()
        # Done
        done = bool(new_x < 2)

        # Reward
        reward = 1 if done else 0
        if done:
            print('Goal reached!')

        if (np.abs(rotx) > 0.5) or (np.abs(roty) > 0.5):
            self.reset()

        return self.get_obs(), reward, done, {}

    def _get_translation(self):
        return self.translation_field.getSFVec3f()
    def _get_rotation(self):
        return self.rotation_field.getSFRotation()
    def _set_translation(self, x, y, z):
        self.translation_field.setSFVec3f([x, y, z])
        return None
    def _set_rotation(self, rotx, roty, rotz, rota):
        self.rotation_field.setSFRotation([rotx, roty, rotz, rota])
        return None


def main():
    # Initialize the environment

    env = SimpleQ()
    env.reset()
    # for i in range(18):
    #     env.step(0)
    # for i in range(3):
    #     env.step(2)
    # for i in range(10):
    #     env.step(0)
    #

    # Train
    model = PPO('MlpPolicy', env, verbose=1, learning_rate=0.01)
    model.learn(total_timesteps=2e5)
    #
    # Replay
    print('Training is finished, press `Y` for replay...')
    env.wait_keyboard()
    #
    obs = env.reset()
    for _ in range(100000):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        print(obs, reward, done, info)
        if done:
            obs = env.reset()

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
