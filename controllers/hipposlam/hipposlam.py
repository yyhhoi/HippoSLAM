"""tabular_qlearning controller."""
import os
import numpy as np
import gymnasium as gym
import torch
from os.path import join

from hipposlam.Replay import ReplayMemoryAWAC
from hipposlam.utils import save_pickle
from hipposlam.sequences import Sequences, HippoLearner
from hipposlam.Networks import ActorModel, MLP, QCriticModel
from hipposlam.ReinforcementLearning import AWAC
from controller import Keyboard, Supervisor
# from stable_baselines3.common.env_checker import check_env
# from stable_baselines3 import PPO
# from controllers.hipposlam.hipposlam.sequences import Sequences, HippoLearner
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
        self.game_mode = 'hard'  # 'easy' or 'hard'

        # Supervisor
        self.supervis = self.getSelf()
        self.translation_field = self.supervis.getField('translation')
        self.rotation_field = self.supervis.getField('rotation')
        self.fallen = False
        self.fallen_seq = 0

        # Self position
        self.x, self.y = None, None
        self.rotz, self.rota = None, None
        self.stuck_m = 0
        self.stuck_epsilon = 0.0001
        self.stuck_thresh = 7.5

        # Environment specific
        self.__timestep = int(self.getBasicTimeStep())  # default 32ms
        self.thetastep = self.__timestep * 32  # 32 * 32 = 1024 ms

        # Distance sensor
        self.ds = []
        self.ds = self.getDevice('ds2')  # Front sensor
        self.ds.enable(self.__timestep * 4)

        # Wheels
        self.leftMotor1 = self.getDevice('wheel1')
        self.leftMotor2 = self.getDevice('wheel3')
        self.rightMotor1 = self.getDevice('wheel2')
        self.rightMotor2 = self.getDevice('wheel4')
        self.MAX_SPEED = 15

        # Action - 'Forward', 'back', 'left', 'right'
        self.act_dim = 3
        self.action_space = gym.spaces.Discrete(self.act_dim)
        self.turn_steps = self.thetastep
        self.forward_steps = self.thetastep
        self.move_d = self.MAX_SPEED * 2 / 3
        self._action_to_direction = {
            0: np.array([self.move_d, self.move_d]),  # Forward
            1: np.array([-self.move_d, self.move_d]) * 0.2,  # Left turn
            2: np.array([self.move_d, -self.move_d]) * 0.2,  # Right turn
        }

        # hippoSlam
        self.fpos_dict = dict()
        self.obj_dist = 2  # in meters
        R, L = 5, 10
        self.seq = Sequences(R=R, L=L, reobserve=False)
        self.HL = HippoLearner(R, L, L)


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
        x, y, a = self._spawn()
        # x = np.random.uniform(3.45, 6.3, size=1)
        # y = np.random.uniform(1.35, 3.85, size=1)
        # a = np.random.uniform(-np.pi, np.pi, size=1)
        self.stuck_m = 0
        self._set_translation(x, y, 0.07)  # 4.18, 2.82, 0.07
        self._set_rotation(0, 0, -1, a)  # 0, 0, -1, 1.57
        x, y, _ = self._get_translation()
        self.x, self.y = x, y
        self.rotz, self.rota = -1, a
        self.init_wheels()

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
        out = super().step(self.thetastep)

        obs = self.get_obs()
        new_x, new_y, rotx, roty, rotz, rota = obs

        # Win condition
        win = self._check_goal(new_x, new_y)
        if win:
            print('\n================== Robot has reached the goal =================================\n')
            reward, done = 1, True
        else:
            reward, done = 0, False


        # Stuck detection
        dpos = np.sqrt((new_x-self.x)**2 + (new_y - self.y)**2 + (rotz*rota - self.rotz*self.rota)**2)
        stuck_count = dpos < self.stuck_epsilon
        self.stuck_m = 0.9 * self.stuck_m + stuck_count * 1.0
        stuck = self.stuck_m > self.stuck_thresh
        self.x, self.y = new_x, new_y
        self.rotz = rotz
        self.rota = rota

        if stuck:
            print("\n================== Robot is stuck =================================\n")
            print('stuck_m = %0.4f'%(self.stuck_m))
            reward, done = -1, True


        # Fallen detection
        fallen = (np.abs(rotx) > 0.4) | (np.abs(roty) > 0.4)
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


        return obs, reward, done, {'robot':out}

    def steptime(self):
        return super().step(self.__timestep)

    def init_wheels(self):
        for motor in [self.leftMotor1, self.leftMotor2, self.rightMotor1, self.rightMotor2]:
            motor.setVelocity(0)
            motor.setPosition(float('inf'))



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

    def _check_goal(self, x, y):
        if self.game_mode == 'easy':
            win = bool((x < 2) or (y < 0))
        elif self.game_mode == 'hard':
            xgood = x < -4
            ygood = (y > -2) & (y < -0.5)
            win = bool(xgood and ygood)
        else:
            raise ValueError()
        return win

    def _spawn(self):
        if self.game_mode == 'easy':
            x = np.random.uniform(3.45, 6.3, size=1)
            y = np.random.uniform(1.35, 3.85, size=1)
            a = np.random.uniform(-np.pi, np.pi, size=1)
        elif self.game_mode == 'hard':
            room = int(np.random.randint(0, 3))
            a = np.random.uniform(-np.pi, np.pi, size=1)

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

        return x, y, a


def key2action(key):
    if key == 315:
        return 0
    elif key == 314:
        return 1
    elif key == 316:
        return 2
    else:
        return None

def learn_by_AWAC():
    # Paths
    save_dir = join('data', 'Omniscient')
    loss_recorder_pth = join(save_dir, 'AC_loss.txt')
    with open(loss_recorder_pth, 'w') as f:
        f.write('critic_loss,actor_loss,t,r\n')

    env = OmniscientLearner()
    obs_dim = 6
    act_dim = 3
    gamma = 0.99
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
    def get_turn_direction(x, y):
        if x > 2:  # First room
            a = 2  # right
        elif (x < 2 and x > -2.7) and (y < 1.45):  # middle room, lower half
            a = 2  # right

        elif (x < 2 and x > -2.7) and (y > 1.45):  # middle room, upper half
            a = 1  # left
        else:  # Final Room
            a = 1  # left
        return a

    env = OmniscientLearner()
    data = {'traj':[], 'end_r':[], 't':[]}
    cum_win = 0
    while cum_win <= 100:
        print('Iter %d, cum_win = %d'%(len(data['end_r']), cum_win))
        s = env.reset()
        trajlist = []
        done = False
        r = None
        t = 0
        while done is False:

            # Policy
            dsval = env.ds.getValue()
            if dsval < 95:
                a = get_turn_direction(s[0], s[1])
            else:
                a = 0
            if np.random.rand() < 0.3:
                a = int(np.random.randint(0, 3))

            # Step
            snext, r, done, info = env.step(a)

            # Store data
            experience = np.concatenate([
                s, np.array([a]), snext, np.array([r]), np.array([done])
            ])
            trajlist.append(experience)

            s = snext
            t += 1

        # Store data
        traj = np.vstack(trajlist)
        data['traj'].append(traj)
        data['end_r'].append(r)
        data['t'].append(t)
        if r > 0:
            cum_win += 1

    save_pickle(join('data', 'Omniscient', 'naive_controller_data.pickle'), data)

def evaluate_trained_model():

    # Paths
    ckpt_pth = join('data', 'Omniscient', 'NaiveControllerCHPT.pt')

    # Parameters
    obs_dim = 6
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
    agent.load_checkpoint(ckpt_pth)
    agent.eval()

    # Unroll
    env = OmniscientLearner()
    Niters = 100
    all_t = []
    maxtimeout = 300
    for i in range(Niters):
        print('Episode %d/%d'%(i, Niters))
        s = env.reset()
        t = 0
        while True:

            s = torch.from_numpy(s).to(torch.float32).view(-1, obs_dim)  # (1, obs_dim)
            a = int(agent.get_action(s).squeeze())  # tensor (1, 1) -> int
            snext, r, done, info = env.step(a)
            s = snext
            t += 1
            if done or (t >= maxtimeout):
                msg = "Done" if done else "Timeout"
                print(msg)
                all_t.append(t)
                break
def main():
    # naive_avoidance()
    evaluate_trained_model()


if __name__ == '__main__':
    main()
