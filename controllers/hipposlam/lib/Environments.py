import copy
import os
from os.path import join

import numpy as np
import torch
from skimage.io import imsave

from controller import Supervisor
import gymnasium as gym

class BreakRoom(Supervisor, gym.Env):
    def __init__(self, max_episode_steps=300, use_ds=True, spawn='all', goal='hard'):
        super().__init__()
        # ====================== To be defined by child class ========================================
        self.obs_dim = None
        self.observation_space = None
        # ============================================================================================
        self.spec = gym.envs.registration.EnvSpec(id='WeBotsQ-v0', max_episode_steps=max_episode_steps)
        self.spawn_mode = spawn  # 'all' or 'start'
        self.goal_mode = goal  # 'easy' or 'hard'
        self.use_ds = use_ds
        self.x_norm = 6  # Normalize value for neural network training
        self.y_norm = 4  # Normalize value for neural network training
        # Environment specific
        self.__timestep = int(self.getBasicTimeStep())  # default 32ms
        self.thetastep = self.__timestep * 32  # 32 * 32 = 1024 ms
        self.r_bonus_counts = [0] * 4
        self.t = 0  # reset to 0, +1 every time self.step() is called
        self.maxt = max_episode_steps
        # Supervisor
        self.supervis = self.getSelf()
        self.translation_field = self.supervis.getField('translation')
        self.translation_field.enableSFTracking(self.__timestep)
        self.rotation_field = self.supervis.getField('rotation')
        self.rotation_field.enableSFTracking(self.__timestep)

        # Self position
        self.x, self.y = None, None
        self.rotz, self.rota = None, None
        self.stuck_m = 0
        self.stuck_epsilon = 0.001
        self.stuck_thresh = 8.5
        self.fallen = False
        self.fallen_seq = 0
        self.fallen_thresh = 0.6

        # Distance sensor or bumper
        if self.use_ds:
            self.ds = []
            self.ds = self.getDevice('ds2')  # Front sensor
            self.ds.enable(self.__timestep)

        # Wheels
        self.leftMotor1 = self.getDevice('wheel1')
        self.leftMotor2 = self.getDevice('wheel3')
        self.rightMotor1 = self.getDevice('wheel2')
        self.rightMotor2 = self.getDevice('wheel4')
        self.MAX_SPEED = 15
        # Action - 'Forward', 'left', 'right'
        self.act_dim = 4
        self.action_space = gym.spaces.Discrete(self.act_dim)
        self.turn_steps = self.thetastep
        self.forward_steps = self.thetastep
        self.move_d = self.MAX_SPEED
        self._action_to_direction = {
            0: np.array([self.move_d, self.move_d]),  # Forward
            1: np.array([0, self.move_d]) * 0.5,  # Left turn
            2: np.array([self.move_d, 0]) * 0.5,  # Right turn
            3: np.array([-self.move_d, -self.move_d]) * 0.2,  # Right turn
        }

    def get_obs(self):
        new_x, new_y, _ = self._get_translation()
        rotx, roty, rotz, rota = self._get_rotation()
        dsval = self.ds.getValue() / 100
        norma = np.sign(rotz) * rota
        cosa, sina = np.cos(norma), np.sin(norma)
        obs = np.array([new_x / self.x_norm, new_y / self.y_norm, rotx, roty, cosa, sina, dsval]).astype(np.float32)
        return obs

    def steptime(self, mul=1):
        super().step(self.__timestep * mul)

    def reset(self, seed=None):
        # Reset the simulation
        self.simulationResetPhysics()
        self.simulationReset()
        super().step(self.__timestep)
        # Reset attributes
        self.stuck_m = 0
        self.t = 0
        self.r_bonus_counts = [0] * 4
        # Reset position and velocity
        x, y, a = self._reset_pose()
        self.x, self.y = x, y
        self.rotz, self.rota = -1, a
        self.init_wheels()
        super().step(self.__timestep * 5)
        # Infer the first step
        obs = self.get_obs()
        # Internals
        super().step(self.__timestep)
        return obs, {}

    def step(self, action):
        leftd, rightd = self._action_to_direction[action]
        self.leftMotor1.setVelocity(leftd)
        self.leftMotor2.setVelocity(leftd)
        self.rightMotor1.setVelocity(rightd)
        self.rightMotor2.setVelocity(rightd)
        super().step(self.thetastep)
        new_x, new_y, new_z = self._get_translation()
        rotx, roty, rotz, rota = self._get_rotation()
        r_bonus = self._get_intermediate_reward(new_x, new_y)

        # Win condition
        win = self._check_goal(new_x, new_y)
        if win:
            print('\n================== Robot has reached the goal =================================\n')
            reward, terminated, truncated = 1, True, False
        else:
            reward, terminated, truncated = 0 + r_bonus, False, False

        # Stuck detection
        dpos = np.sqrt((new_x - self.x) ** 2 + (new_y - self.y) ** 2)
        stuck_count = dpos < self.stuck_epsilon
        self.stuck_m = 0.9 * self.stuck_m + stuck_count * 1.0
        stuck = self.stuck_m > self.stuck_thresh
        self.x, self.y = new_x, new_y
        self.rotz = rotz
        self.rota = rota
        if stuck:
            print("\n================== Robot is stuck =================================\n")
            print('stuck_m = %0.4f' % (self.stuck_m))
            reward, terminated, truncated = -0.01, False, True

        # Fallen detection
        fallen = (np.abs(rotx) > self.fallen_thresh) | (np.abs(roty) > self.fallen_thresh)
        if fallen:
            print('\n================== Robot has fallen =============================\n')
            print('Rotations = %0.4f, %0.4f, %0.4f, %0.4f ' % (rotx, roty, rotz, rota))
            reward, terminated, truncated = -.01, False, True
            if self.fallen:
                self.fallen_seq += 1
            if self.fallen_seq > 5:
                self.fallen_seq = 0
                self._set_rotation(0, 0, -1, 1.57)
                breakpoint()
        self.fallen = fallen
        # Timelimit reached
        timeout = self.t > self.maxt
        if timeout:
            print('\n================== Time out =============================')
            reward, terminated, truncated = -0.01, False, True
        self.t += 1
        # Info
        info = {'last_r': reward, 'terminated': int(terminated), 'truncated': int(truncated), 'stuck': int(stuck),
                'fallen': int(fallen), 'timeout': int(timeout)}
        return self.get_obs(), reward, terminated, truncated, info

    def init_wheels(self):
        for motor in [self.leftMotor1, self.leftMotor2, self.rightMotor1, self.rightMotor2]:
            motor.setVelocity(0)
            motor.setPosition(float('inf'))

    def _get_translation(self):
        return self.translation_field.getSFVec3f()

    def _get_rotation(self):
        return self.rotation_field.getSFRotation()

    def _get_heading(self):
        _, _, rotz, rota = self.rotation_field.getSFRotation()
        return np.sign(rotz) * rota

    def _set_translation(self, x, y, z):
        self.translation_field.setSFVec3f([x, y, z])
        return None

    def _set_rotation(self, rotx, roty, rotz, rot):
        self.rotation_field.setSFRotation([rotx, roty, rotz, rot])
        return None

    def _check_goal(self, x, y):
        if self.goal_mode == 'easy':
            win = bool((x < 2) or (y < 0))
        elif self.goal_mode == 'hard':
            xgood = x < -4
            ygood = (y > -2) & (y < -0.5)
            win = bool(xgood and ygood)
        else:
            raise ValueError()
        return win

    def _get_intermediate_reward(self, x, y):
        r_bonus = 0
        if (y < -1.2) and (self.r_bonus_counts[0] < 1):
            r_bonus = 0.1
            self.r_bonus_counts[0] += 1
        if (x < 2) and (self.r_bonus_counts[1] < 1):
            r_bonus = 0.3
            self.r_bonus_counts[1] += 1
        if (x < 2) and (y > 1.3) and (self.r_bonus_counts[2] < 1):
            r_bonus = 0.5
            self.r_bonus_counts[2] += 1
        if (x < -3.3) and (self.r_bonus_counts[3] < 1):
            r_bonus = 0.7
            self.r_bonus_counts[3] += 1
        if r_bonus > 0:
            print('Partial goal arrived! R bonus = %0.2f' % (r_bonus))
        return r_bonus

    def _spawn(self):
        if self.spawn_mode == 'start':
            x = np.random.uniform(3.45, 6.3, size=1)
            y = np.random.uniform(1.35, 3.85, size=1)
            a = np.random.uniform(-np.pi, np.pi, size=1)
            z = 0.07
        elif self.spawn_mode == 'all':
            room = int(np.random.randint(0, 3))
            a = np.random.uniform(-np.pi, np.pi, size=1)
            z = 0.03
            if room == 0:  # First room
                x = np.random.uniform(3.45, 5.34, size=1)
                y = np.random.uniform(-2.25, 4.60, size=1)
            elif room == 1:  # Middle room
                x = np.random.uniform(-1.35, 1.16, size=1)
                y = np.random.uniform(-1.75, 3.92, size=1)
            elif room == 2:  # transition between Middle and final room
                x = np.random.uniform(-4.7, 1.68, size=1)
                y = np.random.uniform(2.25, 4.03, size=1)
            else:
                raise ValueError()
        else:
            raise ValueError()
        return float(x), float(y), z, float(a)

    def _reset_pose(self):
        x, y, z, a = self._spawn()
        self._set_translation(x, y, z)  # 4.18, 2.82, 0.07
        self._set_rotation(0, 0, -1, a)  # 0, 0, -1, 1.57
        return x, y, a


class OmniscientLearner(BreakRoom):
    def __init__(self, max_episode_steps=1000, use_ds=True, spawn='all', goal='hard'):
        super(OmniscientLearner, self).__init__(max_episode_steps, use_ds, spawn, goal)
        lowBox = np.array([-7, -3, -1, -1, -1, -1, 0], dtype=np.float32)
        highBox = np.array([7, 5, 1, 1, 1, 1, 1], dtype=np.float32)
        self.obs_dim = 7
        self.observation_space = gym.spaces.Box(lowBox, highBox, shape=(self.obs_dim,))
