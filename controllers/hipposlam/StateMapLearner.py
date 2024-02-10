"""tabular_qlearning controller."""

import numpy as np
import torch
from torch import nn

from controllers.hipposlam.hipposlam.Networks import MLP
from controllers.hipposlam.hipposlam.ReinforcementLearning import AWAC, BioQ
from controllers.hipposlam.hipposlam.Replay import ReplayMemoryAWAC
from hipposlam.utils import breakroom_avoidance_policy, save_pickle, PerformanceRecorder, read_pickle
from hipposlam.Environments import StateMapLearner
from os.path import join
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def bio_learning():
    # Tags
    load_previous_hipposlam = False

    # Paths
    model_name = 'BioLearning'
    save_replay_pth = join('data', 'StateMapLearner', 'AvoidanceReplayBuffer_%s.pickle'%(model_name))
    load_hipposlam_pth = join('data', 'StateMapLearner', 'hipposlam_%s.pickle'%(model_name))
    save_hipposlam_pth = join('data', 'StateMapLearner', 'hipposlam_%s.pickle'%(model_name))
    save_bioq_pth = join('data', 'StateMapLearner', 'bioq_%s.pickle'%(model_name))
    save_record_pth = join('data', 'StateMapLearner', '%s_Performance.csv'%(model_name))

    # Environment
    env = StateMapLearner(spawn='start', goal='easy')
    if load_previous_hipposlam:
        env.load_hipposlam(load_hipposlam_pth)

    # RL agent
    agent = BioQ(300)

    # Record data and episodes
    PR = PerformanceRecorder(save_record_pth)
    data = {'episodes':[], 'end_r':[], 't':[], 'traj':[]}
    cum_win = 0
    maxtimeout = 300
    while cum_win <= 100:
        print('Iter %d, cum_win = %d'%(len(data['end_r']), cum_win))
        s = env.reset()
        explist = []
        trajlist = []
        t = 0
        while True:

            # Policy
            x, y = env.x, env.y
            rotz, rota = env.rotz, env.rota
            agent.expand(s)
            a, aprob = agent.get_action(s)
            print('s = %d, a = %d, aprob = '%(s, a), list(np.around(aprob, 3)))
            print('v = %d, m = ' % (agent.w[s]), list(np.around(agent.m[s, :], 3)))

            # Step
            snext, r, done, info = env.step(a)

            # Learn
            agent.update(int(s), a, aprob, r, int(snext), done)


            # Store data
            explist.append(np.array([s, a, snext, r, done]))
            trajlist.append(np.array([x, y, np.sign(rotz)*rota, s]))
            PR.record(t, r)

            # Increment
            s = snext
            t += 1

            # Termination condition
            if done or (t >= maxtimeout):
                msg = "Done" if done else "Timeout"
                print(msg)
                break

        # Store data
        data['episodes'].append(np.vstack(explist))
        data['traj'].append(np.vstack(trajlist))
        data['end_r'].append(r)
        data['t'].append(t)
        if r > 0:
            cum_win += 1
        print()

    # Saving
    env.save_hipposlam(save_hipposlam_pth)
    save_pickle(save_replay_pth, data)
    PR.to_csv()
    save_pickle(save_bioq_pth, agent)

def fine_tune_trained_model():
    # Modes
    load_expert_buffer = True
    save_replay_buffer = False
    save_this_hipposlam = True
    hipposlam_learn_mode = True


    # Paths

    offline_data_pth = join('data', 'StateMapLearner', 'AvoidanceReplayBuffer.pickle')
    load_ckpt_pth = join('data', 'StateMapLearner', 'NaiveControllerCKPT.pt')
    # load_ckpt_pth = join('data', 'StateMapLearner', 'FineTuned1.pt')
    load_hipposlam_pth = join('data', 'StateMapLearner', 'hipposlam.pickle')
    save_model_name = 'FineTuned1'
    save_ckpt_pth = join('data', 'StateMapLearner', '%s.pt'%save_model_name)
    save_buffer_pth = join('data', 'StateMapLearner', 'ReplayBuffer_%s.pt'%save_model_name)
    save_hipposlam_pth = join('data', 'StateMapLearner', 'hipposlam_%s.pickle'%save_model_name)

    # Parameters
    obs_dim = 100
    act_dim = 3
    gamma = 0.99
    lam = 1
    batch_size = 1024
    max_buffer_size = 10000

    # Initialize models
    critic = MLP(obs_dim, act_dim, [128, 128])
    critic_target = MLP(obs_dim, act_dim, [128, 128])
    actor = MLP(obs_dim, act_dim, [128, 64])
    agent = AWAC(critic, critic_target, actor,
                 lam=lam,
                 gamma=gamma,
                 num_action_samples=10,
                 critic_lr=1e-4,  # 5e-4
                 actor_lr=1e-4,  # 5e-4
                 weight_decay=0,
                 use_adv=True)
    agent.load_checkpoint(load_ckpt_pth)

    # Initialize Replay buffer
    memory = ReplayMemoryAWAC(max_size=max_buffer_size, discrete_obs=obs_dim)
    datainds = np.cumsum([0, 1, 1, 1, 1, 1])
    memory.specify_data_tuple(s=(datainds[0], datainds[1]), a=(datainds[1], datainds[2]),
                              snext=(datainds[2], datainds[3]), r=(datainds[3], datainds[4]),
                              end=(datainds[4], datainds[5]))


    # Load expert data and add to replay buffer
    if load_expert_buffer:
        data = read_pickle(offline_data_pth)
        memory.from_offline_np(data['episodes'])  # (time, data_dim=15)
        print('Load Expert buffer. Replay buffer has %d samples' % (len(memory)))


    # Environment
    env = StateMapLearner(spawn='start', goal='hard')
    env.load_hipposlam(load_hipposlam_pth)
    env.hippomap.learn_mode = hipposlam_learn_mode


    # Unroll
    Niters = 300
    maxtimeout = 300
    for i in range(Niters):
        print('Episode %d/%d, NumHippomapStates=%d'%(i, Niters, env.hippomap.N))
        s = env.reset()
        t = 0
        agent.eval()
        explist = []
        while True:

            # Policy
            s_onehot = nn.functional.one_hot(torch.LongTensor([s]), num_classes=obs_dim).to(
                torch.float32)  # (1, obs_classes)
            a = int(agent.get_action(s_onehot).squeeze())  # tensor (1, 1) -> int

            # Step
            snext, r, done, info = env.step(a)

            # Store data
            explist.append(torch.tensor([s, a, snext, r, done]).to(torch.float32))

            # Increment
            s = snext
            t += 1

            # Termination condition
            if done or (t >= maxtimeout):
                msg = "Done" if done else "Timeout"
                print(msg)
                break

        # Store memory
        for exp in explist:
            memory.push(exp)

        # Training
        agent.train()
        if len(memory) > 1:
            _s, _a, _snext, _r, _end = memory.sample(batch_size)
            critic_loss = agent.update_critic(_s, _a, _snext, _r, _end)
            actor_loss = agent.update_actor(_s, _a)
            closs = critic_loss.item()
            aloss = actor_loss.item()
            print('Training finished. C/A Loss = %0.6f, %0.6f' % (closs, aloss))
    agent.save_checkpoint(save_ckpt_pth)

    if save_replay_buffer:
        memory.save_buffer_torch(save_buffer_pth)
    if save_this_hipposlam:
        env.save_hipposlam(save_hipposlam_pth)


def evaluate_trained_model():

    save_this_hipposlam = False
    hipposlam_learn = False

    # Paths
    load_hipposlam_pth = join('data', 'StateMapLearner', 'hipposlam.pickle')
    save_hipposlam_pth = join('data', 'StateMapLearner', 'hipposlam2.pickle')
    load_ckpt_pth = join('data', 'StateMapLearner', 'NavieControllerCKPT.pt')
    save_record_pth = join('data', 'StateMapLearner', 'NavieControllerCKPT_Performance.csv')

    # Parameters
    obs_dim = 100
    act_dim = 3
    gamma = 0.99
    lam = 1

    # Initialize models
    critic = MLP(obs_dim, act_dim, [128, 128])
    critic_target = MLP(obs_dim, act_dim, [128, 128])
    actor = MLP(obs_dim, act_dim, [128, 64])
    agent = AWAC(critic, critic_target, actor,
                 lam=lam,
                 gamma=gamma,
                 num_action_samples=10,
                 critic_lr=1e-4,
                 actor_lr=1e-4,
                 weight_decay=0,
                 use_adv=True)
    agent.load_checkpoint(load_ckpt_pth)
    agent.eval()

    # Environment
    env = StateMapLearner(spawn='start', goal='hard')
    env.load_hipposlam(load_hipposlam_pth)
    env.hipposlam.learn = hipposlam_learn
    PR = PerformanceRecorder(save_record_pth)

    # Unroll
    Niters = 100
    maxtimeout = 300
    for i in range(Niters):
        print('Episode %d/%d'%(i, Niters))
        s = env.reset()
        t = 0
        while True:
            # Policy
            s_onehot = nn.functional.one_hot(torch.LongTensor([s]), num_classes=obs_dim).to(torch.float32) # (1, obs_classes)
            a = int(agent.get_action(s_onehot).squeeze())  # tensor (1, 1) -> int

            # Step
            snext, r, done, info = env.step(a)

            # Record
            PR.record(t, r)

            # Increment
            s = snext
            t += 1

            # Termination
            if done or (t >= maxtimeout):
                msg = "Done" if done else "Timeout"
                print(msg)
                break
    PR.to_csv()
    if save_this_hipposlam:
        env.save_hipposlam(save_hipposlam_pth)



def naive_avoidance():
    # Tags
    load_previous_hipposlam = False

    # Paths
    save_replay_pth = join('data', 'StateMapLearner', 'AvoidanceReplayBuffer.pickle')
    load_hipposlam_pth = join('data', 'StateMapLearner', 'hipposlam.pickle')
    save_hipposlam_pth = join('data', 'StateMapLearner', 'hipposlam.pickle')
    save_record_pth = join('data', 'StateMapLearner', 'NaiveAvoidanceController_Performance.csv')

    # Environment
    env = StateMapLearner(spawn='start')
    if load_previous_hipposlam:
        env.load_hipposlam(load_hipposlam_pth)

    # Record data and episodes
    PR = PerformanceRecorder(save_record_pth)
    data = {'episodes':[], 'end_r':[], 't':[], 'traj':[]}
    cum_win = 0
    maxtimeout = 300
    while cum_win <= 50:
        print('Iter %d, cum_win = %d'%(len(data['end_r']), cum_win))
        s = env.reset()
        explist = []
        trajlist = []
        t = 0
        while True:

            # Policy
            x, y = env.x, env.y
            rotz, rota = env.rotz, env.rota
            a = breakroom_avoidance_policy(x, y, env.ds.getValue(), 0.3)

            # Step
            snext, r, done, info = env.step(a)

            # Store data
            explist.append(np.array([s, a, snext, r, done]))
            trajlist.append(np.array([x, y, np.sign(rotz)*rota, s]))
            PR.record(t, r)

            # Increment
            s = snext
            t += 1

            # Termination condition
            if done or (t >= maxtimeout):
                msg = "Done" if done else "Timeout"
                print(msg)
                break

        # Store data
        data['episodes'].append(np.vstack(explist))
        data['traj'].append(np.vstack(trajlist))
        data['end_r'].append(r)
        data['t'].append(t)
        if r > 0:
            cum_win += 1
        print()

    # Saving
    env.save_hipposlam(save_hipposlam_pth)
    save_pickle(save_replay_pth, data)
    PR.to_csv()


def main():
    # naive_avoidance()
    # evaluate_trained_model()
    # fine_tune_trained_model()
    bio_learning()
    return None

if __name__ == '__main__':
    main()
