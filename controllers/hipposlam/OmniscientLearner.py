"""tabular_qlearning controller."""
import os
import numpy as np
import torch
from os.path import join

from matplotlib import pyplot as plt

from hipposlam.Replay import ReplayMemoryAWAC
from hipposlam.utils import save_pickle, read_pickle, breakroom_avoidance_policy, PerformanceRecorder
from hipposlam.Networks import ActorModel, MLP, QCriticModel
from hipposlam.ReinforcementLearning import AWAC
from hipposlam.Environments import OmniscientLearner
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def learn_by_AWAC():
    # Paths
    save_dir = join('data', 'Omniscient')
    loss_recorder_pth = join(save_dir, 'AC_loss.txt')
    with open(loss_recorder_pth, 'w') as f:
        f.write('critic_loss,actor_loss,t,r\n')

    env = OmniscientLearner()
    obs_dim = 6
    act_dim = 3
    gamma = 0.9
    lam = 1

    critic = QCriticModel(obs_dim, act_dim)
    critic_target = QCriticModel(obs_dim, act_dim)
    actor = ActorModel(obs_dim, act_dim, logit=True)

    batch_size = 1024
    memory = ReplayMemoryAWAC(max_size=10000)
    datainds = np.cumsum([0, obs_dim, 1, obs_dim, 1, 1])
    memory.specify_data_tuple(s=(datainds[0], datainds[1]), a=(datainds[1], datainds[2]),
                              snext=(datainds[2], datainds[3]), r=(datainds[3], datainds[4]),
                              end=(datainds[4], datainds[5]))
    agent = AWAC(critic, critic_target, actor,
                 lam=lam,
                 gamma=gamma,
                 num_action_samples=10,
                 critic_lr=1e-3,
                 actor_lr=1e-3,
                 weight_decay=0,
                 use_adv=True)

    # agent.load_state_dict(torch.load(join(save_dir, 'AWAC.pickle')))

    Niters = 500
    maxtimeout = 100

    all_t = []
    for i in range(Niters):
        print('Episode %d/%d'%(i,Niters))
        s = env.reset()
        candidates = []
        t = 0
        while True:

            s = torch.from_numpy(s).to(torch.float32).view(-1, obs_dim)  # (1, obs_dim)
            a = int(agent.get_action(s).squeeze())  # tensor (1, 1) -> int
            snext, r, done, info = env.step(a)

            experience = torch.concat([
                s.squeeze(), torch.tensor([a]), torch.tensor(snext).to(torch.float),
                torch.tensor([r]).to(torch.float), torch.tensor([done]).to(torch.float)
            ])
            candidates.append(experience)
            s = snext
            t += 1
            if done or (t >= maxtimeout):
                msg = "Done" if done else "Timeout"
                print(msg)
                all_t.append(t)
                break

        if t < maxtimeout:
            for exp in candidates:
                memory.push(exp)
        if len(memory) > 1:
            _s, _a, _snext, _r, _end = memory.sample(batch_size)
            critic_loss = agent.update_critic(_s, _a, _snext, _r, _end)
            actor_loss = agent.update_actor(_s, _a)
            closs = critic_loss.item()
            aloss = actor_loss.item()
            print('Training finished. C/A Loss = %0.6f, %0.6f' % (closs, aloss))
            with open(loss_recorder_pth, 'a') as f:
                f.write('%0.6f,%0.6f,%d,%0.1f\n' % (closs, aloss, t, r))

    # torch.save(agent.state_dict(), join(save_dir, 'AWAC.pickle'))

    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d
    df = pd.read_csv(loss_recorder_pth)
    critic_loss = df['critic_loss'].to_numpy()
    actor_loss = df['actor_loss'].to_numpy()
    t = df['t'].to_numpy()
    r = df['r'].to_numpy()
    critic_loss_gau = gaussian_filter1d(critic_loss, sigma=10)
    actor_loss_gau = gaussian_filter1d(actor_loss, sigma=10)
    t_gau = gaussian_filter1d(t, sigma=10)
    r_gau = gaussian_filter1d(r, sigma=10)
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax = ax.ravel()
    ax[0].plot(critic_loss)
    ax[0].plot(critic_loss_gau)
    ax[1].plot(actor_loss)
    ax[1].plot(actor_loss_gau)
    ax[2].plot(t)
    ax[2].plot(t_gau)
    ax[3].plot(r)
    ax[3].plot(r_gau)
    fig.savefig(join(save_dir, 'AWAC.png'), dpi=300)



def naive_avoidance():
    # Paths
    model_name = 'NaiveController'
    save_replay_pth = join('data', 'OmniscientLearner', '%s_ReplayBuffer.pickle'%model_name)
    save_record_pth = join('data', 'OmniscientLearner', '%s_Performance.csv'%model_name)

    # Environment
    env = OmniscientLearner(spawn='start', goal='hard')

    # Record data and epsidoes
    PRtraj = PerformanceRecorder('i', 't', 'r')

    data = {'episodes':[], 'end_r':[], 't':[]}
    cum_win = 0
    i = 0
    maxtimeout = 300
    while cum_win <= 100:
        print('Iter %d, cum_win = %d'%(len(data['end_r']), cum_win))
        s = env.reset()
        exp_list = []
        t = 0
        while True:

            # Policy
            dsval = env.ds.getValue()
            a = breakroom_avoidance_policy(s[0], s[1], dsval, 0.3)

            # Step
            snext, r, done, info = env.step(a)

            # Store data
            experience = np.concatenate([
                s, np.array([a]), snext, np.array([r]), np.array([done])
            ])
            exp_list.append(experience)
            PRtraj.record(i=i, t=t, r=r)

            # Increment
            s = snext
            t += 1

            # Termination condition
            if done or (t >= maxtimeout):
                msg = "Done" if done else "Timeout"
                print(msg)
                break

        # Store data
        data['episodes'].append(np.vstack(exp_list))
        data['end_r'].append(r)
        data['t'].append(t)
        if r > 0:
            cum_win += 1
        i += 1
        print()

    # Saving
    save_pickle(save_replay_pth, data)
    PRtraj.to_csv()

def fine_tune_trained_model():
    # Modes
    load_expert_buffer = False
    save_replay_buffer = False
    online = False

    # Paths
    save_dir = join('data', 'OmniscientLearner')
    load_model_name = 'OfflineTrainedComplex'
    save_model_name = 'FinetunedComplex1'
    offline_data_pth = join(save_dir, 'NaiveController_ReplayBuffer.pickle')
    load_ckpt_pth = join(save_dir, '%s_CHPT.pt' % load_model_name)
    save_ckpt_pth = join(save_dir, '%s_CHPT.pt' % save_model_name)
    save_buffer_pth = join(save_dir, '%s_ReplayBuffer.pt' % save_model_name)
    save_record_pth = join(save_dir, '%s_Records.csv'%save_model_name)
    save_plot_pth = join(save_dir, '%s_LOSS.png'% save_model_name)

    # Parameters
    obs_dim = 8
    act_dim = 3
    gamma = 0.9
    lam = 1
    batch_size = 1024
    max_buffer_size = 10000

    # Initialize models
    critic = MLP(obs_dim, act_dim, [128, 128, 128])
    critic_target = MLP(obs_dim, act_dim, [128, 128, 128])
    actor = MLP(obs_dim, act_dim, [128, 128, 64])
    agent = AWAC(critic, critic_target, actor,
                 lam=lam,
                 gamma=gamma,
                 num_action_samples=10,
                 critic_lr=1e-4,
                 actor_lr=1e-4,
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
        memory.from_offline_np(data['episodes'])  # (time, data_dim=15)
        print('Load Expert buffer. Replay buffer has %d samples' % (len(memory)))


    # Environment
    env = OmniscientLearner(spawn='start', goal='hard')
    PR = PerformanceRecorder('i', 't', 'r', 'closs', 'aloss')


    Niters = 500
    maxtimeout = 300
    for i in range(Niters):
        print('Episode %d/%d'%(i, Niters))
        s = env.reset()
        t = 0
        agent.eval()
        explist = []
        while True:

            # Policy
            s = torch.from_numpy(s).to(torch.float32).view(-1, obs_dim)  # (1, obs_dim)
            a = int(agent.get_action(s).squeeze())  # tensor (1, 1) -> int


            # Step
            snext, r, done, info = env.step(a)

            # Store data
            experience = torch.concat([
                s.squeeze(), torch.tensor([a]), torch.tensor(snext), torch.tensor([r]), torch.tensor([done])
            ])
            explist.append(experience.to(torch.float))

            # Increment
            s = snext
            t += 1

            # Termination
            if done or (t >= maxtimeout):
                msg = "Done" if done else "Timeout"
                print(msg)
                break

        # Store memory
        if not online:
            for exp in explist:
                memory.push(exp)
        PR.record(i=i, t=t, r=r)


        # Training
        agent.train()
        if (len(memory) > 1) or online:
            if online:
                exptmp = torch.vstack(explist)
                _s, _a, _snext, _r, _end = memory.online_process(exptmp)
            else:
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
    PRtraj = PerformanceRecorder('i', 't', 'r')

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
    # naive_avoidance()
    # evaluate_trained_model()
    fine_tune_trained_model()

if __name__ == '__main__':
    main()
