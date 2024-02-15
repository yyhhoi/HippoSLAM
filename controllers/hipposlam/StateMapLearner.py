"""tabular_qlearning controller."""

import numpy as np
import torch
from matplotlib import pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env

from torch import nn

from hipposlam.Networks import MLP
from hipposlam.ReinforcementLearning import AWAC, A2C, compute_discounted_returns
from hipposlam.Replay import ReplayMemoryAWAC, ReplayMemoryA2C
from hipposlam.utils import breakroom_avoidance_policy, save_pickle, PerformanceRecorder, read_pickle
from hipposlam.Environments import StateMapLearner, StateMapLearnerForest
from os.path import join
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'



def SB_PPO_Train():
    # Modes
    load_model = True
    save_model = True
    hippomap_learn = True

    # Paths
    save_dir = join('data', 'StateMapLearnerForest_R5L20')
    os.makedirs(save_dir, exist_ok=True)
    load_model_name = 'PPO6'
    save_model_name = 'PPO7'
    load_hipposlam_pth = join(save_dir, '%s_hipposlam.pickle' % load_model_name)
    load_model_pth = join(save_dir, '%s.zip'%(load_model_name))
    save_hipposlam_pth = join(save_dir, '%s_hipposlam.pickle' % save_model_name)
    save_model_pth = join(save_dir, '%s.zip' % (save_model_name))
    save_record_pth = join(save_dir, '%s_TrainRecords.csv' % save_model_name)

    # Environment
    env = StateMapLearnerForest(R=5, L=20, spawn='start', goal='hard', max_episode_steps=500, max_hipposlam_states=500, use_ds=False, use_bumper=True)
    info_keywords = ('Nstates', 'last_r', 'terminated', 'truncated', 'stuck', 'fallen', 'timeout')
    env = Monitor(env, save_record_pth, info_keywords=info_keywords)
    check_env(env)

    # Load models
    if load_model:
        env.unwrapped.load_hipposlam(load_hipposlam_pth)
        env.unwrapped.hippomap.learn_mode = hippomap_learn
        print('Hipposamp learn mode = ', str(env.unwrapped.hippomap.learn_mode))
        print('Loading hippomap. There are %d states in the hippomap' % (env.hippomap.N))
        model = PPO.load(load_model_pth, env=env)
    else:
        env = Monitor(env, save_record_pth, info_keywords=info_keywords)
        model = PPO("MlpPolicy", env, verbose=1)

    # Train
    model.learn(total_timesteps=25000)

    # Save models
    if save_model:
        model.save(save_model_pth)
        env.unwrapped.save_hipposlam(save_hipposlam_pth)

    print('After training, there are %d states in the hippomap' % (env.hippomap.N))


def SB_PPO_Eval():
    # Modes
    hippomap_learn = False

    # Paths
    save_dir = join('data', 'StateMapLearner')
    load_model_name = 'PPO'
    # load_model_name = 'PPO11_NoLearnMap'
    load_hipposlam_pth = join(save_dir, '%s_hipposlam.pickle' % load_model_name)
    load_model_pth = join(save_dir, '%s.zip'%(load_model_name))
    save_record_pth = join(save_dir, '%s_EvalRecords.csv' % load_model_name)

    # Environment
    env = StateMapLearner(spawn='start', goal='hard', max_hipposlam_states=500, use_ds=False)


    # Load models
    env.load_hipposlam(load_hipposlam_pth)
    env.hippomap.learn_mode = hippomap_learn
    print('Loading hippomap. There are %d states in the hippomap' % (env.hippomap.N))

    # Monitor
    env = Monitor(env, save_record_pth, info_keywords=('last_r',))

    # RL model
    model = PPO.load(load_model_pth, env=env)

    # Eval
    vec_env = model.get_env()
    obs = vec_env.reset()
    for i in range(100):
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, done, info = vec_env.step(action)

            # Termination
            if done:
                print('Done')
                break

def OnlineA2C():

    # Modes
    load_model = False
    load_hipposlam = False
    save_this_hipposlam = True
    hipposlam_learn_mode = True


    # Paths
    save_dir = join('data', 'StateMapLearner')
    load_model_name = ''
    save_model_name = 'OnlineA2C1'
    load_model_pth = join(save_dir, '%s.pt'%load_model_name)
    load_hipposlam_pth = join(save_dir, '%s_hipposlam.pickle'%load_model_name)
    save_model_pth = join(save_dir, '%s.pt'%save_model_name)
    save_hipposlam_pth = join(save_dir, 'hipposlam_%s.pickle'%save_model_name)
    save_record_pth = join(save_dir, '%s_Records.csv'%save_model_name)
    save_plot_pth = join(save_dir, '%s_LOSS.png'% save_model_name)

    # Parameters
    obs_dim = 100
    act_dim = 3
    gamma = 0.9

    # Initialize models
    critic = MLP(obs_dim, 1, [128, 128])
    actor = MLP(obs_dim, act_dim, [128, 64])
    critic_target = MLP(obs_dim, 1, [128, 128])
    agent = A2C(critic, actor, critic_target=critic_target,
                 gamma=gamma,
                 critic_lr=1e-3,
                 actor_lr=1e-3,
                 weight_decay=0)
    if load_model:
        agent.load_checkpoint(load_model_pth)

    # Initialize Replay buffer
    memory = ReplayMemoryA2C(discrete_obs=obs_dim)
    datainds = np.cumsum([0, obs_dim, 1, 1])
    memory.specify_data_tuple(s=(datainds[0], datainds[1]), a=(datainds[1], datainds[2]),
                              G=(datainds[2], datainds[3]))

    # Environment
    env = StateMapLearner(spawn='start', goal='easy', use_ds=False)
    if load_hipposlam:
        env.load_hipposlam(load_hipposlam_pth)
    env.hippomap.learn_mode = hipposlam_learn_mode
    PR = PerformanceRecorder('i', 't', 'r', 'closs', 'aloss')

    # Unrolle
    Niters = 200
    maxtimeout = 300
    for i in range(Niters):
        print('Episode %d/%d'%(i, Niters))
        s = env.reset()
        t = 0
        agent.eval()
        explist = []
        r_end_list = []
        while True:

            # Policy
            s_onehot = nn.functional.one_hot(torch.LongTensor([s]), num_classes=obs_dim).to(
                torch.float32)  # (1, obs_classes)
            a = int(agent.get_action(s_onehot).squeeze())  # tensor (1, 1) -> int

            # Step
            snext, r, done, info = env.step(a)

            # Store data
            explist.append(torch.concat([s.squeeze(), torch.tensor([a])]).to(torch.float))
            r_end_list.append(torch.tensor([r, done]).to(torch.float32))

            # Increment
            s = snext
            t += 1

            # Termination
            if done or (t >= maxtimeout):
                msg = "Done" if done else "Timeout"
                print(msg)
                break

        PR.record(i=i, t=t, r=r)

        # Last v
        with torch.no_grad():
            s = torch.from_numpy(s).to(torch.float32).view(-1, obs_dim)
            last_v = critic(s)

        # Training
        agent.train()
        exptmp = torch.vstack(explist)
        r_end = torch.vstack(r_end_list)
        G = compute_discounted_returns(r_end[:, 0], r_end[:, 1], last_v.squeeze().detach().item(), gamma)
        print(np.around(G, 2))
        exp = torch.hstack([exptmp, torch.from_numpy(G).to(torch.float32).view(-1, 1)])
        _s, _a, _G = memory.online_process(exp)
        critic_loss, actor_loss = agent.update_networks(_s, _a, _G)
        closs = critic_loss.item()
        aloss = actor_loss.item()
        PR.record(closs=closs, aloss=aloss)
        print('Training finished. C/A Loss = %0.6f, %0.6f' % (closs, aloss))


    # Saving
    agent.save_checkpoint(save_model_pth)
    PR.to_csv(save_record_pth)
    if save_this_hipposlam:
        env.save_hipposlam(save_hipposlam_pth)

    # Plotting
    win_mask = PR.records_df['r'] == 1
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    ax[0].plot(PR.records_df['closs'])
    ax[0].set_title('Win rate = %0.4f'% win_mask.mean())
    ax[1].plot(PR.records_df['aloss'])
    ax[1].set_title('Traj time = %0.4f' % PR.records_df[win_mask]['t'].median())
    fig.savefig(save_plot_pth, dpi=200)


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

    # Saving
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
    SB_PPO_Train()
    # SB_PPO_Eval()
    return None

if __name__ == '__main__':
    main()
