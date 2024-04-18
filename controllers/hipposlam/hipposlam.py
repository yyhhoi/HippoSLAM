import numpy as np
import torch
from matplotlib import pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback

from torch import nn

from hipposlam_lib.Networks import MLP
# from hipposlam_lib.ReinforcementLearning import AWAC, A2C, compute_discounted_returns
from hipposlam_lib.Replay import ReplayMemoryAWAC, ReplayMemoryA2C
from hipposlam_lib.utils import breakroom_avoidance_policy, save_pickle, Recorder, read_pickle
from hipposlam_lib.Environments import StateMapLearner, StateMapLearnerTaught, EmbeddingLearner, \
    StateMapLearnerUmapEmbedding, StateMapLearnerImageSaver
from hipposlam_lib.Environments import ImageSampler, StateMapLearnerVAEEmbedding, OmniscientForest
from os.path import join
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Notes
# - Fully superviosed: PPO3_FullySupervised_lr1_dp3_da8
# - half-supervised: PPO5_OnlyCorrectLearnt_lr1_dp2_da12: Not successful aver. R ~ 0.2
# - half-supervised: PPO4_HalfSupervised_lr1_dp3_da8: Not successful aver. R ~ 0.438
# - unsupervised:
def SB_PPO_Train():
    # Modes
    load_model = False
    save_model = True
    hippomap_learn = True
    model_class = PPO

    # Paths
    save_dir = join('data', 'StateMapLearnerUmapEmbedding_test')
    os.makedirs(save_dir, exist_ok=True)
    load_model_name = ''
    save_model_name = 'PPO1'
    load_hipposlam_pth = join(save_dir, '%s_hipposlam.pickle' % load_model_name)
    load_model_pth = join(save_dir, '%s.zip'%(load_model_name))
    save_hipposlam_pth = join(save_dir, '%s_hipposlam.pickle' % save_model_name)
    save_model_pth = join(save_dir, '%s.zip' % (save_model_name))
    save_record_pth = join(save_dir, '%s' % save_model_name)
    save_trajdata_pth = join(save_dir, '%s_trajdata.pickle' % save_model_name)
    # save_trajdata_pth = None

    # Environment
    env = StateMapLearnerUmapEmbedding(R=5, L=20, max_hipposlam_states=1000,
                                     save_hipposlam_pth=save_hipposlam_pth, save_trajdata_pth=save_trajdata_pth)
    # env = StateMapLearnerTaught(R=5, L=20,
    #                                  save_hipposlam_pth=save_hipposlam_pth, save_trajdata_pth=save_trajdata_pth)
    # env = OmniscientForest(maxt=1000, save_trajdata_pth=save_trajdata_pth)

    info_keywords = ('Nstates', 'last_r', 'terminated', 'truncated', 'stuck', 'fallen', 'timeout')
    env = Monitor(env, save_record_pth, info_keywords=info_keywords)
    check_env(env)


    # Load models
    if load_model:
        # env.unwrapped.load_hipposlam(load_hipposlam_pth)
        # env.unwrapped.hippomap.learn_mode = hippomap_learn
        # print('Hipposamp learn mode = ', str(env.unwrapped.hippomap.learn_mode))
        # print('Loading hippomap. There are %d states in the hippomap' % (len(env.hippomap.sid2embed)))
        model = model_class.load(load_model_pth, env=env)
        # env.hippomap.area_norm = False
    else:
        model = model_class("MlpPolicy", env, verbose=1)

    # Train
    model.learn(total_timesteps=80000)

    # Save models
    if save_model:
        model.save(save_model_pth)
        # env.unwrapped.save_hipposlam(save_hipposlam_pth)

    # print('After training, there are %d states in the hippomap' % (env.hippomap.N))


def main():
    SB_PPO_Train()
    return None

if __name__ == '__main__':
    main()
