"""tabular_qlearning controller."""
import os
import numpy as np
import gymnasium as gym
import torch
from os.path import join

from controllers.hipposlam.hipposlam.Replay import ReplayMemoryCat
from hipposlam.sequences import Sequences, HippoLearner
from hipposlam.Networks import ActorModel, MLP, QCriticModel
from hipposlam.ReinforcementLearning import TensorConvertor, DataLoader, model_train, AWAC
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

        # Supervisor
        self.supervis = self.getSelf()
        self.translation_field = self.supervis.getField('translation')
        self.rotation_field = self.supervis.getField('rotation')
        self.fallen = False
        self.fallen_seq = 0

        # Self position
        self.x, self.y = None, None
        self.stuck_m = 0
        self.stuck_epsilon = 0.0001
        self.stuck_thresh = 7.5

        # Environment specific
        self.__timestep = int(self.getBasicTimeStep())  # default 32ms
        self.thetastep = self.__timestep * 32  # 32 * 32 = 1024 ms

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
        heading = np.sign(rotz) * rota
        obs = np.array([new_x, new_y, rotx, roty, np.cos(heading), np.sin(heading)])
        return obs

    def reset(self):
        # Reset the simulation
        self.simulationResetPhysics()
        self.simulationReset()
        super().step(self.__timestep)

        self.translation_field = self.supervis.getField('translation')
        self.rotation_field = self.supervis.getField('rotation')

        # Reset position and velocity
        x = np.random.uniform(3.45, 6.3, size=1)
        y = np.random.uniform(1.35, 3.85, size=1)
        a = np.random.uniform(-np.pi, np.pi, size=1)
        self.stuck_m = 0
        self._set_translation(x, y, 0.07)  # 4.18, 2.82, 0.07
        self._set_rotation(0, 0, -1, a)  # 0, 0, -1, 1.57
        x, y, _ = self._get_translation()
        self.x, self.y = x, y
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
        win = bool((new_x < 2) or (new_y < 0))
        if win:
            print('\n================== Robot has reached the goal =================================\n')
            reward, done = 1, True
        else:
            reward, done = 0, False


        # Stuck detection
        dpos = np.sqrt((new_x-self.x)**2 + (new_y - self.y)**2)
        stuck_count = dpos < self.stuck_epsilon
        self.stuck_m = 0.9 * self.stuck_m + stuck_count * 1.0
        stuck = self.stuck_m > self.stuck_thresh
        self.x, self.y = new_x, new_y
        if stuck:
            print("\n================== Robot is stuck =================================\n")
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





def key2action(key):
    if key == 315:
        return 0
    elif key == 314:
        return 1
    elif key == 316:
        return 2
    else:
        return None

def expert_demo():
    # Paths
    save_dir = join('data', 'Omniscient')
    save_pth = join(save_dir, 'expert_demo.npy')
    os.makedirs(save_dir, exist_ok=True)

    # Initialize the environment
    env = OmniscientLearner()
    demo_data = []
    NumTrajs = 3


    obs = env.reset()
    keyboard = Keyboard()
    keyboard.enable(int(env.getBasicTimeStep()))

    while len(demo_data) < NumTrajs:
        print('Completing %d/%d demo' %(len(demo_data), NumTrajs))
        traj = []
        while env.steptime() != -1:
            key = keyboard.getKey()
            act = key2action(key)
            if act is not None:
                next_obs, reward, done, info = env.step(act)
                data_tuple = (*obs, act, *next_obs, reward, done)
                traj.append(data_tuple)
                print(np.around(data_tuple, 2))
                obs = next_obs
                if done:
                    obs = env.reset()
                    print('Traj with %d steps added to demo data'%(len(traj)))
                    break
            else:
                env.init_wheels()

        demo_data.append(np.asarray(traj))
    print('Demo finished, save data at %s'%(save_pth))
    np.save(save_pth, np.vstack(demo_data))



def main():
    # expert_demo()

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

    batch_size = 128
    memory = ReplayMemoryCat(max_size=10000)
    datainds = np.cumsum([0, obs_dim, 1, obs_dim, 1, 1])
    memory.specify_data_tuple(s=(datainds[0], datainds[1]), a=(datainds[1], datainds[2]),
                              snext=(datainds[2], datainds[3]), r=(datainds[3], datainds[4]),
                              end=(datainds[4], datainds[5]))
    agent = AWAC(critic, critic_target, actor,
                 lam=lam,
                 gamma=gamma,
                 num_action_samples=10,
                 critic_lr=3e-4,
                 actor_lr=3e-4,
                 weight_decay=0,
                 use_adv=True)

    Niters = 200
    maxtimeout = 300

    all_t = []
    for i in range(Niters):
        print('Episode %d/%d'%(i,Niters))
        s = env.reset()

        truncated = False
        t = 0
        while True and (truncated is False):

            s = torch.from_numpy(s).to(torch.float32).view(-1, obs_dim)  # (1, obs_dim)
            a = int(agent.get_action(s).squeeze())  # tensor (1, 1) -> int
            snext, r, done, info = env.step(a)

            experience = torch.concat([
                s.squeeze(), torch.tensor([a]), torch.tensor(snext).to(torch.float),
                torch.tensor([r]).to(torch.float), torch.tensor([done]).to(torch.float)
            ])
            memory.push(experience)
            s = snext
            t += 1
            if done:
                all_t.append(t)
                break

        if len(memory) >= batch_size:
            _s, _a, _snext, _r, _end = memory.sample(batch_size)
            critic_loss = agent.update_critic(_s, _a, _snext, _r, _end)
            actor_loss = agent.update_actor(_s, _a)

            closs = critic_loss.item()
            aloss = actor_loss.item()
            print('Training finished. C/A Loss = %0.6f, %0.6f' % (closs, aloss))
            with open(loss_recorder_pth, 'a') as f:
                f.write('%0.6f,%0.6f,%d,%0.1f\n' % (closs, aloss, t, r))

    torch.save(agent.state_dict(), join(save_dir, 'AWAC.pickle'))

if __name__ == '__main__':
    main()
