import copy
import os
from os.path import join

import numpy as np
from skimage.io import imsave

from controller import Supervisor
import gymnasium as gym

from .Sequences import Sequences, StateDecoder, StateTeacher
from .utils import save_pickle, read_pickle, Recorder
from .vision import WebotImageConvertor, MobileNetEmbedder

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
        self.translation_field.enableSFTracking(self.__timestep)
        self.rotation_field = self.supervis.getField('rotation')
        self.rotation_field.enableSFTracking(self.__timestep)

        # Self position
        self.x, self.y = None, None
        self.rotz, self.rota = None, None
        self.stuck_m = 0
        self.stuck_epsilon = 0.001
        self.stuck_thresh = 8
        self.fallen = False
        self.fallen_seq = 0
        self.fallen_thresh = 0.6

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
                # breakpoint()
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
        if (x < -13) and (y < -6.8):
            return True

    def _get_intermediate_reward(self, x, y):
        r_bonus = 0

        if (x < -4) and (y < 4) and (self.r_bonus_counts[0] < 1):
            r_bonus = 0.3
            self.r_bonus_counts[0] += 1

        if (x < -8) and (y < 0) and (self.r_bonus_counts[1] < 1):
            r_bonus = 0.5
            self.r_bonus_counts[1] += 1
        if (x < -11) and (y < -4) and (self.r_bonus_counts[2] < 1):
            r_bonus = 0.7
            self.r_bonus_counts[2] += 1
        if r_bonus > 0:
            print('Partial goal arrived! R bonus = %0.2f' % (r_bonus))
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


class StateMapLearner(Forest):
    def __init__(self, R=5, L=20, maxt=1000, max_hipposlam_states=500,
                 save_hipposlam_pth=None, save_trajdata_pth=None, save_img_dir=None):
        super(StateMapLearner, self).__init__(maxt)
        self.observation_space = gym.spaces.Discrete(max_hipposlam_states)
        self.save_trajdata_pth = save_trajdata_pth
        self.save_img_dir = save_img_dir
        self.c = 0  # global counter for image saving

        # Camera
        self.camera_timestep = int(self.getBasicTimeStep())
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
        self.hippomap = StateDecoder(R=R, L=L, maxN=max_hipposlam_states, area_norm=False)
        self.hippomap.set_lowSthresh(0.2)
        self.current_embedid = 0

        # I/O
        if self.save_trajdata_pth:
            self.SW = Recorder('t', 'x', 'y', 'a', 'sid', 'r', 'terminated', 'truncated', 'fsigma', 'embedid')
        else:
            self.SW = None

    def get_obs_base(self):
        id_list = self.recognize_objects()
        self.hipposeq.step(id_list)

        sid, Snodes = self.hippomap.infer_state(self.hipposeq.X)
        if (self.hippomap.reach_maximum() is False) and (self.hippomap.learn_mode):
            self.hippomap.learn_unsupervised(self.hipposeq.X)

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

        # Reset lib
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
            fsigma_to_store = copy.deepcopy({key: val for key, val in self.hipposeq.f_sigma.items() if len(val) > 0})
            self.SW.record(t=self.t, x=x, y=y, a=a, sid=obs, r=reward, terminated=terminated, truncated=truncated,
                           fsigma=fsigma_to_store, embedid=self.current_embedid)

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
                fpos_key = '%d' % objid
                self.fpos_dict[fpos_key] = objpos
                # print('Insert Id=%s with position ' % (fpos_key), objpos)

            # Compute distance
            dist = np.sqrt((x - objpos[0]) ** 2 + (y - objpos[1]) ** 2)
            dist_list.append(dist)
            if dist < 1:
                IDlist_out.append('%d_t' % (objid))
            elif (dist < self.obj_dist) and (dist > 1):
                # print('Close object %d added'%(objid))
                IDlist_out.append('%d_c' % (objid))
            else:
                if not only_close:
                    IDlist_out.append('%d_f' % (objid))

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

    def save_image(self, save_img_pth):
        img_bytes = self.cam.getImage()
        img = np.array(bytearray(img_bytes)).reshape(self.cam_height, self.cam_width, 4)  # BGRA
        img = img[:, :, [2, 1, 0, 3]]  # RGBA
        imsave(save_img_pth, img)


class StateMapLearnerImageSaver(StateMapLearner):

    def __init__(self, R=5, L=20, maxt=1000, max_hipposlam_states=500,
                 save_hipposlam_pth=None, save_trajdata_pth=None, save_img_dir=None):
        super().__init__(R=R, L=L, maxt=maxt, max_hipposlam_states=max_hipposlam_states,
                         save_hipposlam_pth=save_hipposlam_pth, save_trajdata_pth=save_trajdata_pth,
                         save_img_dir=save_img_dir)

        # I/O
        if self.save_trajdata_pth:
            self.SW = Recorder('t', 'x', 'y', 'a', 'id_list', 'c')
        else:
            self.SW = None


    def get_obs_base(self):
        id_list = self.recognize_objects()
        self.hipposeq.step(id_list)


        if self.save_trajdata_pth:
            x, y, _ = super()._get_translation()
            a = super()._get_heading()
            fsigma_to_store = copy.deepcopy({key: val for key, val in self.hipposeq.f_sigma.items() if len(val) > 0})
            self.SW.record(t=self.t, x=x, y=y, a=a, id_list=id_list, c=self.c)

        if (self.t % 5) == 0:
            self.save_image(join(self.save_img_dir, f'{self.c}_{self.t}.png'))
            self.c += 1

        sid = 0
        Snodes = np.array([0])
        return sid, Snodes

    def step(self, action):
        obs, reward, terminated, truncated, info = super(StateMapLearner, self).step(action)
        info['Nstates'] = self.hippomap.N

        return obs, reward, terminated, truncated, info



class StateMapLearnerUmapEmbedding(StateMapLearner):
    def __init__(self, R=5, L=20, maxt=1000, max_hipposlam_states=1000,
                 save_hipposlam_pth=None, save_trajdata_pth=None):
        from .Embeddings import load_parametric_umap_model
        super().__init__(R, L, maxt, max_hipposlam_states,
                         save_hipposlam_pth=save_hipposlam_pth, save_trajdata_pth=save_trajdata_pth)
        # Embedding
        self.imgconverter = WebotImageConvertor(self.cam_height, self.cam_width)
        self.imgembedder = MobileNetEmbedder()
        self.hippomap.set_lowSthresh(0.98)
        print('Loading Umap')
        load_umap_dir = r'D:\data\OfflineStateMapLearner_IdList3\base\assets\umap_params'
        self.umap_model, self.umins, self.umaxs = load_parametric_umap_model(load_umap_dir)

    def get_obs_base(self):
        img_bytes = self.cam.getImage()
        img_tensor = self.imgconverter.to_torch_RGB(img_bytes)
        embedding = self.imgembedder.infer_embedding(img_tensor)
        umap_embed = self.umap_model.transform(embedding.unsqueeze(0).numpy()).squeeze()
        sid, Snodes = self.hippomap.simple_umap_state_assignment(
            umap_embed, self.umins, self.umaxs)
        self.current_embedid = sid


        # id_list = self.recognize_objects()
        # self.hipposeq.step(id_list)
        # sid, Snodes = self.hippomap.infer_state(self.hipposeq.X)
        #
        # # Image embedding
        # if (self.hippomap.current_F > 0):
        #     img_bytes = self.cam.getImage()
        #     img_tensor = self.imgconverter.to_torch_RGB(img_bytes)
        #     embedding = self.imgembedder.infer_embedding(img_tensor)
        #     umap_embed = self.umap_model.transform(embedding.unsqueeze(0).numpy()).squeeze()
        #     self.current_embedid = self.hippomap.learn_embedding(self.hipposeq.X, umap_embed, self.umins, self.umaxs,
        #                                                          far_ids=None)

        return sid, Snodes


class StateMapLearnerTaught(StateMapLearner):

    def __init__(self, R=5, L=10, maxt=1000, save_hipposlam_pth=None, save_trajdata_pth=None):

        super(StateMapLearnerTaught, self).__init__(R, L, maxt, 1,
                                                    save_hipposlam_pth=save_hipposlam_pth,
                                                    save_trajdata_pth=save_trajdata_pth)


        self.xbound = (-16.4, 8.4)
        self.ybound = (-8.6, 16.2)

        self.dp = 2  # 2
        self.da = 2 * np.pi / 12  # 12
        self.hippoteach = StateTeacher(self.xbound, self.ybound, self.dp, self.da)
        self.max_Nstates = self.hippoteach.Nstates
        self.only_close = False

        # Over-write parent's attributes
        self.hippomap = StateDecoder(R=R, L=L, maxN=self.max_Nstates, area_norm=False)
        self.observation_space = gym.spaces.Discrete(self.max_Nstates)

    def get_obs_base(self):

        # Teacher
        x, y, _ = self._get_translation()
        a = self._get_heading()
        sidgt = self.hippoteach.lookup_xya((x, y, a))

        id_list = self.recognize_objects(only_close=self.only_close)
        self.hipposeq.step(id_list)
        sidpred, Snodes = self.hippomap.infer_state(self.hipposeq.X)

        if self.hipposeq.X.sum() < 1e-6:
            return sidpred, Snodes

        # print('GroundTruthState Storage = \n', self.hippoteach.pred2gt_map)
        if self.hippoteach.match_groundtruth_storage(sidgt):

            print(f'GroundTruthState {sidgt} found in storage')

            if self.hippoteach.pred2gt_map[sidpred] == sidgt:
                msg = 'Match    '
                _ = self.hippomap.learn_supervised(self.hipposeq.X, sid=sidpred, far_ids=None)
            else:
                msg = 'NOT Match'
                _ = self.hippomap.learn_supervised(self.hipposeq.X, sid=self.hippoteach.gt2pred_map[sidgt],
                                                   far_ids=None)

            # print(
            # f'{msg} InferredState {sidpred}/{self.hippomap.N} (mappedGT={self.hippoteach.pred2gt_map[sidpred]} vs {sidgt}), val={Snodes[sidpred]}')

        else:
            print(f'GroundTruthState {sidgt} not found')
            sidpred = self.hippomap.learn_supervised(self.hipposeq.X, sid=None, far_ids=None)
            Snodes = np.zeros(self.hippomap.N)
            Snodes[sidpred] = 1
            self.hippoteach.store_prediction_mapping(sidpred, sidgt)
            self.hippoteach.store_groundtruth_mapping(sidgt, sidpred)
            # print(f'New inferred state {sidpred} created.')

        ## Debug: Compare prediction and ground truth
        sidgt_frompred = self.hippoteach.map_pred_to_gt(sidpred)
        predx, predy, preda = self.hippoteach.sid2xya(sidgt_frompred)
        print('GT/Pred: x=%02.2f/%02.2f, y=%02.2f/%02.2f, a=%03.2f/%03.2f' % (
        x, predx, y, predy, np.rad2deg(a), np.rad2deg(preda)))
        ## ============================================

        self.current_embedid = sidgt

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
