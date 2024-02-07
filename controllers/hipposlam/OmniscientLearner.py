"""tabular_qlearning controller."""

# You may need to import some classes of the controller module. Ex:
from controller import Robot, Motor
from controller import Supervisor
import gym
import numpy as np
# from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from controllers.hipposlam.hipposlam.sequences import Sequences, HippoLearner


import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class OmniscientLearner(Supervisor, gym.Env):
    def __init__(self, max_episode_steps=1000, io_pth="data/statemaps.txt"):
        super().__init__()

        # Open AI Gym generic
        # x, y, rotx, roty, rotz, rota
        lowBox = np.array([-7, -3, -1, -1, -1, -2 * np.pi], dtype=np.float32)
        highBox = np.array([7,  5,  1,  1,  1,  2 * np.pi], dtype=np.float32)
        self.obs_dim = 6
        self.observation_space = gym.spaces.Box(lowBox, highBox, shape=(self.obs_dim,))

        self.state = None
        self.spec = gym.envs.registration.EnvSpec(id='WeBotsQ-v0', max_episode_steps=max_episode_steps)

        # Supervisor
        self.supervis = self.getSelf()
        self.translation_field = self.supervis.getField('translation')
        self.rotation_field = self.supervis.getField('rotation')
        self.fallen = False
        self.fallen_seq = 0

        # Self position
        self.x, self.y = None, None
        self.stuck_m = 0
        self.stuck_epsilon = 0.0001
        self.stuck_thresh = 8

        # Environment specific
        self.__timestep = int(self.getBasicTimeStep())  # default 32ms
        self.thetastep = self.__timestep * 32  # 32 * 32 = 1024 ms

        # Wheels
        self.leftMotor1 = self.getDevice('wheel1')
        self.leftMotor2 = self.getDevice('wheel3')
        self.rightMotor1 = self.getDevice('wheel2')
        self.rightMotor2 = self.getDevice('wheel4')
        self.MAX_SPEED = 15

        # Action - 'Forward', 'back', 'left', 'right'
        self.action_space = gym.spaces.Discrete(3)
        self.turn_steps = self.thetastep
        self.forward_steps = self.thetastep
        self.move_d = self.MAX_SPEED * 2 / 3
        self._action_to_direction = {
            0: np.array([self.move_d, self.move_d]),  # Forward
            1: np.array([-self.move_d, self.move_d]) * 0.5,  # Left turn
            2: np.array([self.move_d, -self.move_d]) * 0.5,  # Right turn
        }

        # Camera
        self.camera_timestep = self.thetastep
        self.cam = self.getDevice('camera')
        self.cam.enable(self.camera_timestep)
        self.cam.recognitionEnable(self.camera_timestep)
        self.cam_width = self.cam.getWidth()
        self.cam_height = self.cam.getHeight()

        # hippoSlam
        self.fpos_dict = dict()
        self.obj_dist = 2  # in meters
        R, L = 5, 10
        self.seq = Sequences(R=R, L=L, reobserve=False)
        self.HL = HippoLearner(R, L, L)

        # Tools
        self.keyboard = self.getKeyboard()
        self.keyboard.enable(self.__timestep)

        # Data I/O
        self.io_pth = io_pth

    def get_obs(self):
        new_x, new_y, _ = self._get_translation()
        rotx, roty, rotz, rota = self._get_rotation()
        obs = np.array([new_x, new_y, rotx, roty, rotz, rota])
        return obs

    def reset(self):
        # Reset the simulation
        self.simulationResetPhysics()
        self.simulationReset()
        super().step(self.__timestep)

        self.translation_field = self.supervis.getField('translation')
        self.rotation_field = self.supervis.getField('rotation')

        # Reset position and velocity
        x = np.random.uniform(3.45, 6.3, size=1)
        y = np.random.uniform(1.35, 3.85, size=1)
        a = np.random.uniform(-np.pi, np.pi, size=1)
        self.stuck_m = 0
        self._set_translation(x, y, 0.07)  # 4.18, 2.82, 0.07
        self._set_rotation(0, 0, -1, a)  # 0, 0, -1, 1.57
        x, y, _ = self._get_translation()
        self.x, self.y = x, y
        for motor in [self.leftMotor1, self.leftMotor2, self.rightMotor1, self.rightMotor2]:
            motor.setVelocity(0)
            motor.setPosition(float('inf'))
        # Reset hipposlam
        self.seq.reset_activity()

        # Infer the first step
        obs = self.get_obs()

        # Internals
        super().step(self.__timestep)

        return obs

    def step(self, action):

        leftd, rightd = self._action_to_direction[action]
        self.leftMotor1.setVelocity(leftd)
        self.leftMotor2.setVelocity(leftd)
        self.rightMotor1.setVelocity(rightd)
        self.rightMotor2.setVelocity(rightd)
        super().step(self.thetastep)


        obs = self.get_obs()
        new_x, new_y, rotx, roty, rotz, rota = obs


        # Win condition
        win = bool((new_x < 2) or (new_y < 0))
        if win:
            print('\n================== Robot has reached the goal =================================\n')
            reward, done = 1, True
        else:
            reward, done = 0, False


        # Stuck detection
        dpos = np.sqrt((new_x-self.x)**2 + (new_y - self.y)**2)
        stuck_count = dpos < self.stuck_epsilon
        self.stuck_m = 0.9 * self.stuck_m + stuck_count * 1.0
        stuck = self.stuck_m > self.stuck_thresh
        self.x, self.y = new_x, new_y
        if stuck:
            print("\n================== Robot is stuck =================================\n")
            reward, done = -1, True


        # Fallen detection
        fallen = (np.abs(rotx) > 0.5) | (np.abs(roty) > 0.5)
        if fallen:
            print('\n================== Robot has fallen %s=============================\n'%(str(fallen)))
            print('Rotations = %0.4f, %0.4f, %0.4f, %0.4f '%(rotx, roty, rotz, rota))
            print('Abs x and y = %0.4f, %0.4f'%(np.abs(rotx), (np.abs(roty))))
            reward, done = -1, True
            if self.fallen:
                self.fallen_seq += 1
            if self.fallen_seq > 5:
                breakpoint()
        self.fallen = fallen


        return obs, reward, done, {}


    def _get_translation(self):
        return self.translation_field.getSFVec3f()
    def _get_rotation(self):
        return self.rotation_field.getSFRotation()
    def _set_translation(self, x, y, z):
        self.translation_field.setSFVec3f([x, y, z])
        return None

    def _set_rotation(self, rotx, roty, rotz, rot):
        self.rotation_field.setSFRotation([rotx, roty, rotz, rot])
        return None


def main():
    # Initialize the environment

    env = OmniscientLearner()
    env.reset()

    while True:
        key = env.keyboard.getKey()
        if key == ord("w"):
            env.step(0)
        elif key == ord("a"):
            env.step(1)
        elif key == ord("d"):
            env.step(2)


    # model = PPO('MlpPolicy', env, verbose=1, learning_rate=0.01)
    # model.learn(total_timesteps=2e5)


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
