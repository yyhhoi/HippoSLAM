import copy
import os
from os.path import join
from pprint import PrettyPrinter

import numpy as np
import torch
from skimage.io import imsave
from sklearn.decomposition import IncrementalPCA
from stable_baselines3.common.callbacks import CheckpointCallback

from controller import Supervisor
import gymnasium as gym

from .Sequences import Sequences, StateDecoder, StateTeacher
from .utils import save_pickle, read_pickle, Recorder
from .vision import WebotImageConvertor, MobileNetEmbedder
from .VAE import VAELearner

class BreakRoom(Supervisor, gym.Env):
    def __init__(self, max_episode_steps=300, use_ds=True,  spawn='all', goal='hard'):
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
        dsval = self.ds.getValue()/100
        norma = np.sign(rotz) * rota
        cosa, sina = np.cos(norma), np.sin(norma)
        obs = np.array([new_x/self.x_norm, new_y/self.y_norm, rotx, roty, cosa, sina, dsval]).astype(np.float32)
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
            print('stuck_m = %0.4f'%(self.stuck_m))
            reward, terminated, truncated = -0.01, False, True

        # Fallen detection
        fallen = (np.abs(rotx) > self.fallen_thresh) | (np.abs(roty) > self.fallen_thresh)
        if fallen:
            print('\n================== Robot has fallen =============================\n')
            print('Rotations = %0.4f, %0.4f, %0.4f, %0.4f '%(rotx, roty, rotz, rota))
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
        info = {'last_r': reward, 'terminated': int(terminated), 'truncated': int(truncated), 'stuck':int(stuck),
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
        if (y < -1.2) and (self.r_bonus_counts[0]<1):
            r_bonus = 0.1
            self.r_bonus_counts[0] += 1
        if (x < 2) and (self.r_bonus_counts[1]<1):
            r_bonus = 0.3
            self.r_bonus_counts[1] += 1
        if (x < 2) and (y > 1.3) and (self.r_bonus_counts[2]<1):
            r_bonus = 0.5
            self.r_bonus_counts[2] += 1
        if (x < -3.3) and (self.r_bonus_counts[3]<1):
            r_bonus = 0.7
            self.r_bonus_counts[3] += 1
        if r_bonus > 0:
            print('Partial goal arrived! R bonus = %0.2f'%(r_bonus))
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
    def __init__(self, max_episode_steps=1000, use_ds=True,  spawn='all', goal='hard'):
        super(OmniscientLearner, self).__init__(max_episode_steps, use_ds, spawn, goal)
        lowBox = np.array([-7, -3, -1, -1, -1, -1, 0], dtype=np.float32)
        highBox = np.array([7,  5,  1,  1, 1, 1, 1], dtype=np.float32)
        self.obs_dim = 7
        self.observation_space = gym.spaces.Box(lowBox, highBox, shape=(self.obs_dim,))

class Forest(Supervisor, gym.Env):
    def __init__(self, maxt=300):
        super().__init__()

        # ====================== To be defined by child class ========================================
        self.obs_dim = None
        self.observation_space = None
        # ============================================================================================

        self.spec = gym.envs.registration.EnvSpec(id='WeBotsQ-v0', max_episode_steps=maxt)

        # Environment specific
        self.__timestep = int(self.getBasicTimeStep())  # default 32ms
        self.thetastep = self.__timestep * 32  # 32 * 32 = 1024 ms
        self.r_bonus_counts = [0] * 4
        self.t = 0  # reset to 0, +1 every time self.step() is called.
        self.maxt = maxt

        # Supervisor
        self.supervis = self.getSelf()
        self.translation_field = self.supervis.getField('translation')
        self.translation_field.enableSFTracking(self.__timestep*12)
        self.rotation_field = self.supervis.getField('rotation')
        self.rotation_field.enableSFTracking(self.__timestep * 12)
        self.supervis.enableContactPointsTracking(self.__timestep)


        # Self position
        self.x, self.y = None, None
        self.rotz, self.rota = None, None
        self.stuck_m = 0
        self.stuck_epsilon = 0.001
        self.stuck_thresh = 8.5
        self.fallen = False
        self.fallen_seq = 0
        self.fallen_thresh = 1


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
        norma = np.sign(rotz) * rota
        cosa, sina = np.cos(norma), np.sin(norma)
        obs = np.array([new_x, new_y, rotx, roty, cosa, sina]).astype(np.float32)
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
            print('stuck_m = %0.4f'%(self.stuck_m))
            reward, terminated, truncated = -0.01, False, True


        # Fallen detection
        fallen = (np.abs(rotx) > self.fallen_thresh) | (np.abs(roty) > self.fallen_thresh)
        if fallen:
            print('\n================== Robot has fallen =============================\n')
            print('Rotations = %0.4f, %0.4f, %0.4f, %0.4f '%(rotx, roty, rotz, rota))
            reward, terminated, truncated = -.01, False, True
            if self.fallen:
                self.fallen_seq += 1
            if self.fallen_seq > 5:
                self.fallen_seq = 0
                self._set_rotation(0, 0, -1, 1.57)
                # breakpoint()
        self.fallen = fallen

        # Timelimit reached
        timeout = self.t > self.maxt
        if timeout:
            print('\n================== Time out =============================')
            reward, terminated, truncated = -0.01, False, True
        self.t += 1

        # Info
        info = {'last_r': reward, 'terminated': int(terminated), 'truncated': int(truncated), 'stuck':int(stuck),
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
        if (x < -13) and (y < -6.8):
            return True


    def _get_intermediate_reward(self, x, y):
        r_bonus = 0

        if (x < -4) and (y < 4) and (self.r_bonus_counts[0]<1):
            r_bonus = 0.3
            self.r_bonus_counts[0] += 1

        if (x < -8) and (y < 0) and (self.r_bonus_counts[1]<1):
            r_bonus = 0.5
            self.r_bonus_counts[1] += 1
        if (x < -11) and (y < -4) and (self.r_bonus_counts[2]<1):
            r_bonus = 0.7
            self.r_bonus_counts[2] += 1
        if r_bonus > 0:
            print('Partial goal arrived! R bonus = %0.2f'%(r_bonus))
        return r_bonus


    def _spawn(self):

        pts = np.array([
            (-3, 5, 0.07),
            (-11.6, 7.17, 0.2),
            (-0.6, -3.43, 0.18),
        ])
        pti = np.random.choice(len(pts))
        x, y, z = pts[pti]

        a = np.random.uniform(-np.pi, np.pi, size=1)
        return float(x), float(y), z, float(a)

    def _reset_pose(self):
        x, y, z, a = self._spawn()
        self._set_translation(x, y, z)  # 4.18, 2.82, 0.07
        self._set_rotation(0, 0, -1, a)  # 0, 0, -1, 1.57

        return x, y, a

class ImageSampler(Forest):

    def __init__(self):
        super().__init__(maxt=300)

        # Camera
        self.camera_timestep = int(self.getBasicTimeStep())
        self.cam = self.getDevice('camera')
        self.cam.enable(self.camera_timestep)
        self.cam_width = self.cam.getWidth()
        self.cam_height = self.cam.getHeight()

        self.stuck_m = 0
        self.stuck_epsilon = 0.001
        self.stuck_thresh = 8
        self.fallen = False
        self.fallen_seq = 0
        self.fallen_thresh = 0.6

        self.translation_field.enableSFTracking(int(self.getBasicTimeStep()))
        self.rotation_field.enableSFTracking(int(self.getBasicTimeStep()))

        # image sampling specifics
        self.save_img_dir = 'data\VAE\imgs3'
        os.makedirs(self.save_img_dir, exist_ok=True)
        self.save_annotation_pth = r'data\VAE\annotations3.csv'
        self.c = 0
        # self.sim = 0  # 1 if the state is similar to the previous sample, 0 if not
        # self.x_prev, self.y_prev, _ = self._get_translation()
        # self.a_prev = self._get_rotation()
        # self.dist_thresh = 2
        # self.adiff_thresh = np.pi / 6
        if not os.path.exists(self.save_annotation_pth):
            with open(self.save_annotation_pth, 'w') as f:
                f.write('c,t,x,y,a\n')


    def get_obs(self):


        if (self.t % 5 == 0):
            # print('Save image and data. c=%d'%self.c)
            img_bytes = self.cam.getImage()
            img = np.array(bytearray(img_bytes)).reshape(self.cam_height, self.cam_width, 4)
            img = img[:, :, [2, 1, 0, 3]]

            save_img_pth = join(self.save_img_dir, f'{self.c}.png')
            imsave(save_img_pth, img)

            x, y, _ = self._get_translation()
            a = self._get_heading()

            txt = f'{self.c},{self.t:d},{x:0.6f},{y:0.6f},{a:0.6f}'
            print(txt)
            with open(self.save_annotation_pth, 'a') as f:
                f.write(f'{txt}\n')

            self.c += 1


        return 0


# class OmniscientLearner(Forest):
#     def __init__(self, max_episode_steps=1000, use_ds=True,  spawn='all'):
#         super(OmniscientLearner, self).__init__(max_episode_steps, use_ds, spawn)
#         lowBox = np.array([-16.5, -8.7, -1, -1, -1, -1], dtype=np.float32)
#         highBox = np.array([8.7,  17,  1,  1, 1, 1], dtype=np.float32)
#         self.obs_dim = 6
#         self.observation_space = gym.spaces.Box(lowBox, highBox, shape=(self.obs_dim,))




class EmbeddingLearner(Forest):
    def __init__(self, embedding_dim, maxt=1000):
        super(EmbeddingLearner, self).__init__(maxt)
        lowBox = np.ones(embedding_dim).astype(np.float32) * -10.0
        highBox = np.ones(embedding_dim).astype(np.float32) * 10.0
        self.obs_dim = embedding_dim
        self.observation_space = gym.spaces.Box(lowBox, highBox, shape=(self.obs_dim,))


        # Camera
        self.camera_timestep = self.thetastep
        self.cam = self.getDevice('camera')
        self.cam.enable(self.camera_timestep)
        self.cam_width = self.cam.getWidth()
        self.cam_height = self.cam.getHeight()

        # Embedding
        self.imgconverter = WebotImageConvertor(self.cam_height, self.cam_width)
        self.imgembedder = MobileNetEmbedder()


    def get_obs(self):

        img_bytes = self.cam.getImage()
        img_tensor = self.imgconverter.to_torch_RGB(img_bytes)
        embedding = self.imgembedder.infer_embedding(img_tensor)
        return embedding.numpy()


class StateMapLearner(Forest):
    def __init__(self, R=5, L=10, maxt=1000, max_hipposlam_states=500,
                 save_hipposlam_pth=None, save_trajdata_pth=None):
        super(StateMapLearner, self).__init__(maxt)
        self.observation_space = gym.spaces.Discrete(max_hipposlam_states)

        # Camera
        self.camera_timestep = self.thetastep
        self.cam = self.getDevice('camera')
        self.cam.enable(self.camera_timestep)
        self.cam.recognitionEnable(self.camera_timestep)
        self.cam_width = self.cam.getWidth()
        self.cam_height = self.cam.getHeight()

        # hippoSlam
        self.save_hipposlam_pth = save_hipposlam_pth
        self.fpos_dict = dict()
        self.obj_dist = 3  # in meters
        self.hipposeq = Sequences(R=R, L=L, reobserve=False)
        self.hippomap = StateDecoder(R=R, L=L, maxN=max_hipposlam_states)
        self.hippomap.set_lowSthresh(0.2)
        self.save_trajdata_pth = save_trajdata_pth

        # I/O
        if self.save_trajdata_pth:
            self.SW = Recorder('t', 'x', 'y', 'a', 'sid', 'r', 'terminated', 'truncated', 'fsigma')
        else:
            self.SW = None

    def get_obs_base(self):
        id_list = self.recognize_objects()
        self.hipposeq.step(id_list)
        sid, Snodes = self.hippomap.infer_state(self.hipposeq.X)
        if (self.hippomap.reach_maximum() is False) and (self.hippomap.learn_mode):
            self.hippomap.learn_unsupervised(self.hipposeq.X)
        # print('Interred state = %d   / %d  , val = %0.2f. F = %d, Xsum=%0.4f'%(sid+1, self.hippomap.N, Snodes[sid], self.hippomap.current_F, self.hipposeq.X.sum()))
        return sid, Snodes

    def get_obs(self):
        sid, Snodes = self.get_obs_base()
        return sid

    def reset(self, seed=None):
        # Reset the simulation
        self.simulationResetPhysics()
        self.simulationReset()
        self.steptime()

        # Reset attributes
        self.stuck_m = 0
        self.t = 0
        self.r_bonus_counts = [0] * 4

        # Reset position and velocity
        x, y, a = self._reset_pose()
        self.x, self.y = x, y
        self.rotz, self.rota = -1, a
        self.init_wheels()
        self.steptime(5)

        # Reset hipposlam
        self.hipposeq.reset_activity()
        self.hippomap.reset()
        self.t = 0
        print('HippoSLAM Num. states = %d. Num. Feature nodes = %d' % (self.hippomap.N, self.hippomap.current_F))
        if self.save_hipposlam_pth:
            self.save_hipposlam(self.save_hipposlam_pth)

        # Save writer
        if self.save_trajdata_pth and (len(self.SW.records_dict) > 0):
            self.SW.append_to_pickle(self.save_trajdata_pth)
        self.SW.clear_records_dict()

        # Infer the first step
        obs = self.get_obs()

        # Internals
        self.steptime()

        return obs, {}


    def step(self, action):
        obs, reward, terminated, truncated, info = super(StateMapLearner, self).step(action)
        info['Nstates'] = self.hippomap.N

        if self.save_trajdata_pth:
            x, y, _ = super()._get_translation()
            a = super()._get_heading()
            fsigma_to_store = copy.deepcopy({key:val for key, val in self.hipposeq.f_sigma.items() if len(val) > 0})
            self.SW.record(t=self.t, x=x, y=y, a=a, sid=obs, r=reward, terminated=terminated, truncated=truncated, fsigma=fsigma_to_store)

        return obs, reward, terminated, truncated, info

    def recognize_objects(self, only_close=False):
        objs = self.cam.getRecognitionObjects()
        idlist = [obj.getId() for obj in objs]

        # Distance from robot to the objects
        x, y, z = self._get_translation()
        IDlist_out = []
        dist_list = []
        for objid in idlist:

            # Obtain object position
            obj_node = self.getFromId(objid)
            objpos = obj_node.getPosition()

            # Store object positions
            if str(objid) not in self.fpos_dict:
                fpos_key = '%d'%objid
                self.fpos_dict[fpos_key] = objpos
                # print('Insert Id=%s with position ' % (fpos_key), objpos)

            # Compute distance
            dist = np.sqrt((x - objpos[0]) ** 2 + (y - objpos[1])**2)
            dist_list.append(dist)
            if dist < 1:
                IDlist_out.append('%d_t' % (objid))
            elif (dist < self.obj_dist) and (dist > 1):
                # print('Close object %d added'%(objid))
                IDlist_out.append('%d_c'%(objid))
            else:
                if not only_close:
                    IDlist_out.append('%d_f'%(objid))

        # id_list = []
        # for c in closeIDlist:
        #     for f in farIDlist:
        #         id_list.append("%s_%s_%d"%(c, f, bumped))

        return IDlist_out

    def save_hipposlam(self, pth):
        save_pickle(pth, dict(hipposeq=self.hipposeq, hippomap=self.hippomap, fpos=self.fpos_dict))

    def load_hipposlam(self, pth):
        hippodata = read_pickle(pth)
        self.hippomap = hippodata['hippomap']
        self.hipposeq = hippodata['hipposeq']
        if 'fpos' in hippodata:
            self.fpos_dict = hippodata['fpos']
        return hippodata


class StateMapLearnerEmbedding(StateMapLearner):
    def __init__(self, R=5, L=10, maxt=1000, max_hipposlam_states=500,
                 save_hipposlam_pth=None, save_trajdata_pth=None):


        super().__init__(R, L, maxt, max_hipposlam_states,
                         save_hipposlam_pth = save_hipposlam_pth, save_trajdata_pth= save_trajdata_pth)
        # Embedding
        self.imgconverter = WebotImageConvertor(self.cam_height, self.cam_width)
        self.imgembedder = MobileNetEmbedder()
        self.hippomap.set_lowSthresh(0.8)
        self.n_components = 50
        self.pca = IncrementalPCA(n_components=self.n_components)
        self.embedding_buffer = []

        self.vaelearner = VAELearner(
            input_dim=576,
            hidden_dims=[400, 200, 100, 50, 25],
            kld_mul=0.01,
            lr=0.001,
            lr_gamma=0.95,
            weight_decay=0
        )

        self.vaelearner.load_checkpoint('data\VAE\model\kldmul=0.010000_bottle25\ckpt_kldmul=0.010000_bottle25.pt')
        self.vaelearner.vae.eval()

    def get_obs_base(self):
        id_list = self.recognize_objects()
        self.hipposeq.step(id_list)
        sid, Snodes = self.hippomap.infer_state(self.hipposeq.X)


        # Image embedding
        if (self.hippomap.learn_mode) and (self.t % 5 == 0) and (self.hippomap.current_F > 0):
            far_ids = list(self.hipposeq.far_fids.values())
            img_bytes = self.cam.getImage()
            img_tensor = self.imgconverter.to_torch_RGB(img_bytes)
            embedding = self.imgembedder.infer_embedding(img_tensor)
            with torch.no_grad():
                _, mu, _ = self.vaelearner.vae(embedding.unsqueeze(0))
            vae_embed = mu.squeeze().detach().numpy().copy()
            self.hippomap.learn_embedding(self.hipposeq.X, vae_embed, far_ids=far_ids)
            # self.embedding_buffer.append(embedding.numpy().copy())
            # if len(self.embedding_buffer) > self.n_components:
            #     print('Partially fitting PCA')
            #     self.pca.partial_fit(np.stack(self.embedding_buffer))
            #     self.embedding_buffer = []
            #     print(f'Total exlained variance = {self.pca.explained_variance_ratio_.sum()}')
            #
            # try:
            #     embedding_PC = self.pca.transform(embedding.reshape(1, -1))
            #     self.hippomap.learn_embedding(self.hipposeq.X, embedding_PC.squeeze(), far_ids=far_ids)
            # except NotFittedError:
            #     pass


        # print('Interred state = %d   / %d  , val = %0.2f. F = %d, Xsum=%0.4f'%(sid+1, self.hippomap.N, Snodes[sid], self.hippomap.current_F, self.hipposeq.X.sum()))
        return sid, Snodes

    def save_hipposlam(self, pth):
        print('Saving HippoSLAM at %s' % pth)
        save_pickle(pth, dict(hipposeq=self.hipposeq, hippomap=self.hippomap,
                              fpos=self.fpos_dict, pca=self.pca))

    def load_hipposlam(self, pth):
        hippodata = super().load_hipposlam(pth)
        self.pca = hippodata['pca']


class StateMapLearnerTaught(StateMapLearner):


    def __init__(self, R=5, L=10, maxt=1000, save_hipposlam_pth=None, save_trajdata_pth=None):

        super(StateMapLearnerTaught, self).__init__(R, L, maxt, 1,
                                                    save_hipposlam_pth = save_hipposlam_pth, save_trajdata_pth= save_trajdata_pth)

        self.xbound = (-6.4, 8.4)
        self.ybound = (-8.6, 16.2)
        self.dp = 2  # 2
        self.da = 2 * np.pi / 8  # 12
        self.hippoteach = StateTeacher(self.xbound, self.ybound, self.dp, self.da)
        self.max_Nstates = self.hippoteach.Nstates
        self.only_close = False

        # Over-write parent's attributes
        self.hippomap = StateDecoder(R=R, L=L, maxN=self.max_Nstates)
        self.observation_space = gym.spaces.Discrete(self.max_Nstates)


    def get_obs_base(self):

        # Teacher
        x, y, _ = self._get_translation()
        _, _, rotz, rota = self._get_rotation()
        a = np.sign(rotz) * rota
        sidgt = self.hippoteach.lookup_xya((x, y, a))


        id_list = self.recognize_objects(only_close=self.only_close)
        self.hipposeq.step(id_list)
        sidpred, Snodes = self.hippomap.infer_state(self.hipposeq.X)



        if self.hipposeq.X.sum() < 1e-6:
            return sidpred, Snodes

        # print('GroundTruthState Storage = \n', self.hippoteach.pred2gt_map)
        if self.hippoteach.match_groundtruth_storage(sidgt):
            # print(f'GroundTruthState {sidgt} found in storage')

            if self.hippoteach.pred2gt_map[sidpred] == sidgt:
                msg = 'Match    '
                _ = self.hippomap.learn_supervised(self.hipposeq.X, sid=sidpred)
            else:
                msg = 'NOT Match'
                _ = self.hippomap.learn_supervised(self.hipposeq.X, sid=self.hippoteach.gt2pred_map[sidgt])

            # print(
                # f'{msg} InferredState {sidpred}/{self.hippomap.N} (mappedGT={self.hippoteach.pred2gt_map[sidpred]} vs {sidgt}), val={Snodes[sidpred]}')

        else:
            # print(f'GroundTruthState {sidgt} not found')
            sidpred = self.hippomap.learn_supervised(self.hipposeq.X)
            Snodes = np.zeros(self.hippomap.N)
            Snodes[sidpred] = 1
            self.hippoteach.store_prediction_mapping(sidpred, sidgt)
            self.hippoteach.store_groundtruth_mapping(sidgt, sidpred)
            # print(f'New inferred state {sidpred} created.')

        # print()
        return sidpred, Snodes

        # sidgt = int(np.random.choice(self.hippoteach.Nstates))
        # _ = np.zeros(self.hippoteach.Nstates)
        # return sidgt, _

    def save_hipposlam(self, pth):
        print('Saving HippoSLAM at %s' % pth)
        save_pickle(pth, dict(hipposeq=self.hipposeq, hippomap=self.hippomap, hippoteach=self.hippoteach,
                              fpos=self.fpos_dict))

    def load_hipposlam(self, pth):
        hippodata = super().load_hipposlam(pth)
        self.hippoteach = hippodata['hippoteach']







class StateMapLearnerSnodes(StateMapLearner):
    def __init__(self, R=5, L=10, maxt=1000, max_hipposlam_states=500,
                 save_hipposlam_pth=None):
        super(StateMapLearnerSnodes, self).__init__(R, L, maxt, max_hipposlam_states,
                                                    save_hipposlam_pth=save_hipposlam_pth)
        self.obs_dim = max_hipposlam_states
        lowBox = np.zeros(self.obs_dim).astype(np.float32)
        highBox = np.ones(self.obs_dim).astype(np.float32)
        self.observation_space = gym.spaces.Box(lowBox, highBox, shape=(self.obs_dim,))


    def get_obs(self):
        sid, Snodes = self.get_obs_base()
        Snodesvec = np.zeros(self.obs_dim).astype(np.float32)
        Snodesvec[:len(Snodes)] = Snodes
        return Snodesvec


