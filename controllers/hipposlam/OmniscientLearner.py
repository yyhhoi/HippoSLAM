"""tabular_qlearning controller."""
import os
import numpy as np
import torch
from os.path import join


from matplotlib import pyplot as plt

from lib.Replay import ReplayMemoryAWAC, ReplayMemoryA2C
from lib.utils import save_pickle, read_pickle, breakroom_avoidance_policy, Recorder
from lib.Networks import MLP
from lib.ReinforcementLearning import AWAC, A2C, compute_discounted_returns
from lib.Environments import OmniscientLearner
from ImitationOmniscientTraining import AWAC_offline_imitation
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def random_agent():
    # =============================================================================
    # ============================= Modifiable ====================================
    # =============================================================================
    # Project identifiers
    experiment = 'ImitationLearningDemo'
    run = 'RandomAgent'

    # Parameters of the run
    NumEndsRequired = 100  # Number of episode until the simulation stops
    MaxTimeOut = 1000  # Maximum time steps for each episode
    # =============================================================================
    # =============================================================================

    # Paths
    save_dir = join('data', experiment, run)
    os.makedirs(save_dir, exist_ok=True)
    save_trajall_pth = join(save_dir, 'Records.csv')

    # Environment
    env = OmniscientLearner(spawn='start', goal='hard')

    # Record trajectory data
    PRtrajall = Recorder('i', 't', 'x', 'y', 'cosa', 'sina', 'r', 'done', 'truncated')

    cum_end = 0
    i = 0
    while cum_end <= NumEndsRequired:
        print('Iter %d, cum_end = %d'%(i, cum_end))
        s, _ = env.reset()
        t = 0
        while True:

            # Policy
            a = int(np.random.choice(4))

            # Step
            snext, r, done, truncated, info = env.step(a)

            # Store data
            PRtrajall.record(i=i, t=t, x=s[0], y=s[1], cosa=s[4], sina=s[5], r=r, done=int(done), truncated=int(truncated))


            # Increment
            s = snext
            t += 1

            # Termination condition
            if done or (t >= MaxTimeOut) or truncated:
                msg = "Done" if done else "Timeout/Stuck"
                cum_end += 1
                print(msg)
                break



        if r > 0:
            cum_end += 1
        i += 1

    # Saving
    PRtrajall.to_csv(save_trajall_pth)

def naive_avoidance():
    # =============================================================================
    # ============================= Modifiable ====================================
    # =============================================================================
    # Project identifiers
    experiment = 'ImitationLearningDemo'
    run = 'NaiveAvoidance'

    # Parameters of the run
    NumWinsRequired = 50  # Number of wins until the simulation stops
    MaxTimeOut = 500  # Maximum time steps for each episode
    # =============================================================================
    # =============================================================================

    # Paths
    save_dir = join('data', experiment, run)
    os.makedirs(save_dir, exist_ok=True)
    save_replay_pth = join(save_dir, 'ReplayBuffer.pickle')
    save_trajall_pth = join(save_dir, 'Records.csv')

    # Environment
    env = OmniscientLearner(spawn='start', goal='hard')
    x_norm = env.x_norm
    y_norm = env.y_norm

    # Record trajectory data
    PRtrajall = Recorder('i', 't', 'x', 'y', 'cosa', 'sina', 'r', 'done', 'truncated')

    data = {'episodes':[], 'end_r':[], 't':[]}
    cum_win = 0
    i = 0
    while cum_win <= NumWinsRequired:
        print('Iter %d, cum_win = %d'%(i, cum_win))
        s, _ = env.reset()
        exp_list = []
        t = 0
        while True:

            # Policy
            dsval = env.ds.getValue()
            a = breakroom_avoidance_policy(s[0]*x_norm, s[1]*y_norm, dsval, 0.3)

            # Step
            snext, r, done, truncated, info = env.step(a)

            # Store data
            PRtrajall.record(i=i, t=t, x=s[0], y=s[1], cosa=s[4], sina=s[5], r=r, done=int(done), truncated=int(truncated))

            # Store Replay Data for AWAC
            experience = np.concatenate([
                s, np.array([a]), snext, np.array([r]), np.array([done])
            ])
            exp_list.append(experience)

            # Increment
            s = snext
            t += 1

            # Termination condition
            if done or (t >= MaxTimeOut) or truncated:
                msg = "Done" if done else "Timeout/Stuck"
                if done:
                    cum_win += 1

                print(msg)
                break

        # PRtraj.record(i=i, t=t, r=r, done=int(done))
        # Store data
        data['episodes'].append(np.vstack(exp_list))
        data['end_r'].append(r)
        data['t'].append(t)

        i += 1

    # Saving
    save_pickle(save_replay_pth, data)
    PRtrajall.to_csv(save_trajall_pth)







def fine_tune_trained_model():
    # =============================================================================
    # ============================= Modifiable ====================================
    # =============================================================================
    # Project identifiers
    experiment = 'ImitationLearningDemo'
    load_run = 'Finetuned1'
    save_run = 'Finetuned2'

    # Parameters of the run
    load_expert_buffer = True
    save_replay_buffer = True
    obs_dim = 7
    act_dim = 4
    gamma = 0.99
    lam = 1
    batch_size = 1024
    max_buffer_size = 10000
    Niters = 1000
    MaxTimeOut = 300
    # =============================================================================
    # =============================================================================

    # Paths
    save_dir = join('data', experiment, save_run)
    load_dir = join('data', experiment, load_run)
    os.makedirs(save_dir, exist_ok=True)
    load_replay_pth = join('data', experiment, 'NaiveAvoidance', 'ReplayBuffer.pickle')
    load_ckpt_pth = join(load_dir, 'CKPT.pt')
    save_replay_pth = join(save_dir, 'ReplayBuffer.pickle')
    save_ckpt_pth = join(save_dir, 'CKPT.pt')
    save_record_pth = join(save_dir, 'Records.csv')
    save_plot_pth = join(save_dir, 'LOSS.png')


    # Initialize models
    critic = MLP(obs_dim, act_dim, [128, 128], hidden_act='Tanh')
    critic_target = MLP(obs_dim, act_dim, [128, 128], hidden_act='Tanh')
    actor = MLP(obs_dim, act_dim, [128, 64], hidden_act='Tanh')
    agent = AWAC(critic, critic_target, actor,
                 lam=lam,
                 gamma=gamma,
                 num_action_samples=100,  # 10
                 critic_lr=1e-4,  # 5e-4
                 actor_lr=1e-4,  # 5e-4
                 weight_decay=0,
                 clip_max_norm=1.0,
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
        # data = memory.load_buffer_torch(load_replay_pth)
        data = read_pickle(load_replay_pth)
        memory.from_offline_np(data['episodes'])  # (time, data_dim)
        print('Load Expert buffer. Replay buffer has %d samples' % (len(memory)))


    # Environment
    env = OmniscientLearner(spawn='start', goal='hard')
    PR = Recorder('i', 't', 'r', 'done', 'closs', 'aloss')



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
            if done or (t >= MaxTimeOut) or truncated:
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
        memory.save_buffer_torch(save_replay_pth)
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

    # =============================================================================
    # ============================= Modifiable ====================================
    # =============================================================================
    # Project identifiers
    experiment = 'ImitationLearningDemo'
    run = 'Finetuned1'
    load_ckpt_run = 'OfflineImitation'

    # Parameters of the run
    obs_dim = 7
    act_dim = 4
    gamma = 0.99
    lam = 1
    Niters = 300
    MaxTimeOut = 300
    # =============================================================================
    # =============================================================================


    # Paths
    save_dir = join('data', experiment, run)
    os.makedirs(save_dir, exist_ok=True)
    load_ckpt_pth = join('data', experiment, load_ckpt_run, 'CKPT.pt')
    save_trajall_pth = join(save_dir, 'Records.csv')


    # Initialize models
    critic = MLP(obs_dim, act_dim, [128, 128], hidden_act='Tanh')
    critic_target = MLP(obs_dim, act_dim, [128, 128], hidden_act='Tanh')
    actor = MLP(obs_dim, act_dim, [128, 64], hidden_act='Tanh')
    agent = AWAC(critic, critic_target, actor,
                 lam=lam,
                 gamma=gamma,
                 num_action_samples=100,  # 10
                 critic_lr=1e-4,  # 5e-4
                 actor_lr=1e-4,  # 5e-4
                 weight_decay=0,
                 clip_max_norm=1.0,
                 use_adv=True)
    agent.load_checkpoint(load_ckpt_pth)
    agent.eval()

    # Environment
    env = OmniscientLearner(spawn='start', goal='hard')
    PRtrajall = Recorder('i', 't', 'x', 'y', 'cosa', 'sina', 'r', 'done', 'truncated')

    # Unroll

    for i in range(Niters):
        print('Episode %d/%d'%(i, Niters))
        s, _ = env.reset()
        t = 0
        while True:

            # Policy
            s = torch.from_numpy(s).to(torch.float32).view(-1, obs_dim)  # (1, obs_dim)
            a = int(agent.get_action(s).squeeze())  # tensor (1, 1) -> int

            # Step
            snext, r, done, truncated, info = env.step(a)

            # Store data
            stmp = s.numpy().squeeze()
            PRtrajall.record(i=i, t=t, x=stmp[0].item(), y=stmp[1], cosa=stmp[4], sina=stmp[5], r=r, done=int(done), truncated=int(truncated))


            # Increment
            s = snext
            t += 1

            # Termination
            if done or (t >= MaxTimeOut) or truncated:
                msg = "Done" if done else "Timeout"
                print(msg)
                break

    # Saving
    PRtrajall.to_csv(save_trajall_pth)


def main():
    # naive_avoidance()
    # random_agent()
    # AWAC_offline_imitation()
    fine_tune_trained_model()

    # evaluate_trained_model()

    # StartOnlineA2C()
    # SB_PPO_Train()

if __name__ == '__main__':
    main()
