"""tabular_qlearning controller."""
import os
import numpy as np
import torch
from os.path import join


from matplotlib import pyplot as plt

from hipposlam.Replay import ReplayMemoryAWAC, ReplayMemoryA2C
from hipposlam.utils import save_pickle, read_pickle, breakroom_avoidance_policy, Recorder
from hipposlam.Networks import MLP
from hipposlam.ReinforcementLearning import AWAC, A2C, compute_discounted_returns
from hipposlam.Environments import OmniscientLearner
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def SB_PPO_Train():
    # Modes
    load_model = True
    save_model = True

    # Paths
    save_dir = join('data', 'OmniscientLearnerForest')
    os.makedirs(save_dir, exist_ok=True)
    load_model_name = 'PPO4'
    save_model_name = 'PPO5'
    load_model_pth = join(save_dir, '%s.zip'%(load_model_name))
    save_model_pth = join(save_dir, '%s.zip' % (save_model_name))
    save_record_pth = join(save_dir, '%s_TrainRecords.csv' % save_model_name)

    # Environment
    env = OmniscientLearnerForest(spawn='start', goal='hard', max_episode_steps=500, use_ds=False, use_bumper=True)
    info_keywords = ('last_r', 'terminated', 'truncated', 'stuck', 'fallen', 'timeout')
    env = Monitor(env, save_record_pth, info_keywords=info_keywords)
    check_env(env)

    # Load models
    if load_model:
        model = PPO.load(load_model_pth, env=env)
    else:
        model = PPO("MlpPolicy", env, verbose=1, )

    # Train
    model.learn(total_timesteps=25000)

    # Save models
    if save_model:
        model.save(save_model_pth)




def StartOnlineA2C():
    # Modes
    load_model = False

    # Paths
    save_dir = join('data', 'OmniscientLearner')
    load_model_name = ''
    save_model_name = 'OnlineA2C1'
    load_model_pth = join(save_dir, '%s_CHPT.pt' % load_model_name)
    save_ckpt_pth = join(save_dir, '%s_CHPT.pt' % save_model_name)
    save_record_pth = join(save_dir, '%s_Records.csv'%save_model_name)
    save_plot_pth = join(save_dir, '%s_LOSS.png'% save_model_name)

    # Parameters
    obs_dim = 6
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
    memory = ReplayMemoryA2C()
    datainds = np.cumsum([0, obs_dim, 1, 1])
    memory.specify_data_tuple(s=(datainds[0], datainds[1]), a=(datainds[1], datainds[2]),
                              G=(datainds[2], datainds[3]))

    # Environment
    env = OmniscientLearner(spawn='start', goal='easy', use_ds=False)
    PR = Recorder('i', 't', 'r', 'closs', 'aloss')

    # Unrolle
    Niters = 500
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
            s = torch.from_numpy(s).to(torch.float32).view(-1, obs_dim)  # (1, obs_dim)
            a = int(agent.get_action(s).squeeze())  # tensor (1, 1) -> int

            # Step
            snext, r, done, info = env.step(a)
            r -= t/maxtimeout


            # Store data
            explist.append(torch.concat([s.squeeze(), torch.tensor([a])]).to(torch.float))
            r_end_list.append(torch.tensor([r, done]).to(torch.float32))

            # Increment
            s = snext
            t += 1

            # Termination
            if done or (t >= maxtimeout):
                msg = "Done" if done else "Timeout"
                print(msg )
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
        # print(np.around(G, 2))
        exp = torch.hstack([exptmp, torch.from_numpy(G).to(torch.float32).view(-1, 1)])
        _s, _a, _G = memory.online_process(exp)
        critic_loss, actor_loss = agent.update_networks(_s, _a, _G)
        closs = critic_loss.item()
        aloss = actor_loss.item()
        PR.record(closs=closs, aloss=aloss)
        print('Training finished. C/A Loss = %0.6f, %0.6f' % (closs, aloss))


    # Saving
    agent.save_checkpoint(save_ckpt_pth)
    PR.to_csv(save_record_pth)

    # Plotting
    win_mask = PR.records_df['r'] == 1
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    ax[0].plot(PR.records_df['closs'])
    ax[0].set_title('Win rate = %0.4f'% win_mask.mean())
    ax[1].plot(PR.records_df['aloss'])
    ax[1].set_title('Traj time = %0.4f' % PR.records_df[win_mask]['t'].median())
    fig.savefig(save_plot_pth, dpi=200)



def naive_avoidance():
    # Paths
    save_dir = join('data', 'OmniscientLearner')
    os.makedirs(save_dir, exist_ok=True)
    model_name = 'NaiveController'
    save_replay_pth = join('data', 'OmniscientLearner', '%s_ReplayBuffer.pickle'%model_name)
    save_record_pth = join('data', 'OmniscientLearner', '%s_Performance.csv'%model_name)

    # Environment
    env = OmniscientLearner(spawn='start', goal='hard')

    # Record data and epsidoes
    PRtraj = Recorder('i', 't', 'r', 'done')

    data = {'episodes':[], 'end_r':[], 't':[]}
    cum_win = 0
    i = 0
    maxtimeout = 300
    while cum_win <= 100:
        print('Iter %d, cum_win = %d'%(len(data['end_r']), cum_win))
        s, _ = env.reset()
        exp_list = []
        t = 0
        while True:

            # Policy
            dsval = env.ds.getValue()
            a = breakroom_avoidance_policy(s[0], s[1], dsval, 0.3)

            # Step
            snext, r, done, truncated, info = env.step(a)

            # Store data
            try:
                experience = np.concatenate([
                    s, np.array([a]), snext, np.array([r]), np.array([done])
                ])
            except:
                breakpoint()
            exp_list.append(experience)


            # Increment
            s = snext
            t += 1

            # Termination condition
            if done or (t >= maxtimeout) or truncated:
                msg = "Done" if done else "Timeout/Stuck"
                print(msg)
                break

        PRtraj.record(i=i, t=t, r=r, done=int(done))
        # Store data
        data['episodes'].append(np.vstack(exp_list))
        data['end_r'].append(r)
        data['t'].append(t)
        if r > 0:
            cum_win += 1
        i += 1

    # Saving
    save_pickle(save_replay_pth, data)
    PRtraj.to_csv(save_record_pth)

def fine_tune_trained_model():
    # Modes
    load_expert_buffer = False
    save_replay_buffer = False

    # Paths
    save_dir = join('data', 'OmniscientLearner')
    os.makedirs(save_dir, exist_ok=True)
    load_model_name = 'OfflineTrained'
    save_model_name = 'Finetuned1'
    offline_data_pth = join(save_dir, 'NaiveController_ReplayBuffer.pickle')
    load_ckpt_pth = join(save_dir, '%s_CKPT.pt' % load_model_name)
    save_ckpt_pth = join(save_dir, '%s_CKPT.pt' % save_model_name)
    save_buffer_pth = join(save_dir, '%s_ReplayBuffer.pt' % save_model_name)
    save_record_pth = join(save_dir, '%s_Records.csv'%save_model_name)
    save_plot_pth = join(save_dir, '%s_LOSS.png'% save_model_name)

    # Parameters
    obs_dim = 7
    act_dim = 4
    gamma = 0.99
    lam = 1
    batch_size = 1024
    max_buffer_size = 5000

    # Initialize models
    critic = MLP(obs_dim, act_dim, [128, 128])
    critic_target = MLP(obs_dim, act_dim, [128, 128])
    actor = MLP(obs_dim, act_dim, [128, 64])
    agent = AWAC(critic, critic_target, actor,
                 lam=lam,
                 gamma=gamma,
                 num_action_samples=100,
                 critic_lr=5e-4,
                 actor_lr=5e-4,
                 weight_decay=0,
                 use_adv=True)
    agent.load_checkpoint(load_ckpt_pth)

    # Initialize Replay buffer
    memory = ReplayMemoryAWAC(max_size=max_buffer_size)
    datainds = np.cumsum([0, obs_dim, 1, obs_dim, 1, 1])
    memory.specify_data_tuple(s=(datainds[0], datainds[1]), a=(datainds[1], datainds[2]),
                              snext=(datainds[2], datainds[3]), r=(datainds[3], datainds[4]),
                              end=(datainds[4], datainds[5]))


    # Load expert data and add to replay buffer
    if load_expert_buffer:
        data = read_pickle(offline_data_pth)
        memory.from_offline_np(data['episodes'])  # (time, data_dim)
        print('Load Expert buffer. Replay buffer has %d samples' % (len(memory)))


    # Environment
    env = OmniscientLearner(spawn='start', goal='hard')
    PR = Recorder('i', 't', 'r', 'done', 'closs', 'aloss')


    Niters = 500
    maxtimeout = 300
    for i in range(Niters):
        print('Episode %d/%d'%(i, Niters))
        s, _ = env.reset()
        t = 0
        agent.eval()
        explist = []
        while True:

            # Policy
            s = torch.from_numpy(s).to(torch.float32).view(-1, obs_dim)  # (1, obs_dim)
            a = int(agent.get_action(s).squeeze())  # tensor (1, 1) -> int


            # Step
            snext, r, done, truncated, info = env.step(a)

            # Store data
            experience = torch.concat([
                s.squeeze(), torch.tensor([a]), torch.tensor(snext), torch.tensor([r]), torch.tensor([done])
            ])
            explist.append(experience.to(torch.float))

            # Increment
            s = snext
            t += 1

            # Termination
            if done or (t >= maxtimeout) or truncated:
                msg = "Done" if done else "Timeout/Truncated"
                print(msg)
                break

        # Store memory
        for exp in explist:
            memory.push(exp)
        PR.record(i=i, t=t, r=r, done=int(done))


        # Training
        agent.train()
        if (len(memory) > 1):
            _s, _a, _snext, _r, _end = memory.sample(batch_size)
            critic_loss = agent.update_critic(_s, _a, _snext, _r, _end)
            actor_loss = agent.update_actor(_s, _a)
            closs = critic_loss.item()
            aloss = actor_loss.item()
            PR.record(closs=closs, aloss=aloss)
            print('Training finished. C/A Loss = %0.6f, %0.6f' % (closs, aloss))

    # Saving
    agent.save_checkpoint(save_ckpt_pth)
    if save_replay_buffer:
        memory.save_buffer_torch(save_buffer_pth)
    PR.to_csv(save_record_pth)

    # Plotting
    win_mask = PR.records_df['r'] == 1
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    ax[0].plot(PR.records_df['closs'])
    ax[0].set_title('Win rate = %0.4f'% win_mask.mean())
    ax[1].plot(PR.records_df['aloss'])
    ax[1].set_title('Traj time = %0.4f' % PR.records_df[win_mask]['t'].median())
    fig.savefig(save_plot_pth, dpi=200)

def evaluate_trained_model():

    # Paths
    save_dir = join('data', 'OmniscientLearner')
    model_name = 'Finetuned1'
    load_ckpt_pth = join(save_dir, '%s_CHPT.pt'%(model_name))
    save_record_pth = join(save_dir, '%s.csv'%(model_name))

    # Parameters
    obs_dim = 8
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
                 critic_lr=5e-4,
                 actor_lr=5e-4,
                 weight_decay=0,
                 use_adv=True)
    agent.load_checkpoint(load_ckpt_pth)
    agent.eval()


    # Environment
    env = OmniscientLearner(spawn='start', goal='hard')
    PRtraj = Recorder('i', 't', 'r')

    # Unroll
    Niters = 100
    maxtimeout = 300
    for i in range(Niters):
        print('Episode %d/%d'%(i, Niters))
        s = env.reset()
        t = 0
        while True:

            # Policy
            s = torch.from_numpy(s).to(torch.float32).view(-1, obs_dim)  # (1, obs_dim)
            a = int(agent.get_action(s).squeeze())  # tensor (1, 1) -> int

            # Step
            snext, r, done, info = env.step(a)

            # Record
            PRtraj.record(i=i, t=t, r=r)

            # Increment
            s = snext
            t += 1

            # Termination
            if done or (t >= maxtimeout):
                msg = "Done" if done else "Timeout"
                print(msg)
                break

    # Saving
    PRtraj.to_csv()


def main():
    naive_avoidance()
    # evaluate_trained_model()
    # fine_tune_trained_model()
    # StartOnlineA2C()
    # SB_PPO_Train()
if __name__ == '__main__':
    main()
