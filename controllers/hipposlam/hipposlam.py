"""tabular_qlearning controller."""
import os
import numpy as np
import gym
import torch
from os.path import join
from hipposlam.sequences import Sequences, HippoLearner
from hipposlam.Networks import ActorModel, CriticModel
from hipposlam.ReinforcementLearning import TensorConvertor, DataLoader, model_train
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
        fallen = (np.abs(rotx) > 0.5) | (np.abs(roty) > 0.5)
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
        f.write('critic_loss,actor_loss,t\n')

    env = OmniscientLearner()
    obs_dim = 6
    act_dim = 3
    gamma = 0.99
    beta = 0.05
    lamb = 1
    TC = TensorConvertor(obs_dim, act_dim)
    critic = CriticModel(obs_dim, act_dim)
    delayed_critic = CriticModel(obs_dim, act_dim)
    delayed_critic.load_state_dict(critic.state_dict())
    actor = ActorModel(obs_dim, act_dim)
    optimizer_A = torch.optim.Adam(actor.parameters(), lr=1e-4, weight_decay=1e-5)
    optimizer_C = torch.optim.Adam(critic.parameters(), lr=1e-4, weight_decay=1e-5)

    Niters = 200
    maxtimeout = 300

    DL = DataLoader(max_buffer_size=50, seed=0)


    for i in range(Niters):
        print('Episode %d/%d'%(i,Niters))
        s = env.reset()
        s = torch.from_numpy(s).reshape(1, -1).to(torch.float32)

        # ================== Model Unroll ===============================
        print('Model Unrolling')
        actor.eval()
        critic.eval()
        done = False
        traj = []
        t = 0
        while (done is False):
            with torch.no_grad():
                aprob = actor(s)  # (1, obs_dim) -> (1, act_dim)
                a = TC.select_action_tensor(aprob)  # (1, act_dim) -> (1, ) int
            s_next, r, done, _ = env.step(a.item())
            t += 1
            if t > maxtimeout:
                r, done = 0, True
                print('TimeOut %d. Reset. Episode not added to the buffer.'%(maxtimeout))
            a_onehot = np.zeros(act_dim)
            a_onehot[a.item()] = 1
            data_duple = (*s.squeeze().numpy(), *a_onehot, *s_next, r, done*1.0)
            # print(np.around(data_duple, 3))
            traj.append(data_duple)
            s = torch.from_numpy(s_next).reshape(1, -1).to(torch.float32)
        if (r != 0):
            print("Episode added to the buffer. Time = %d"%(t))
            data_to_append = np.vstack(traj)
            DL.append(data_to_append)

        # ================== Model Train ===============================
        print('Buffer has %d trajs'%(len(DL.buffer)))
        if len(DL.buffer) < 1:
            continue
        actor.train()
        critic.train()
        all_closs, all_aloss =[], []
        for s, a, s_next, r, end in DL.sample():
            closs, aloss = model_train(s, a, s_next, r, end, actor, critic, delayed_critic, gamma, beta, lamb, False)

            delayed_critic.load_state_dict(critic.state_dict())

            optimizer_C.zero_grad()
            closs.backward()
            optimizer_C.step()

            optimizer_A.zero_grad()
            aloss.backward()
            optimizer_A.step()

        clossmu = closs.item()
        alossmu = aloss.item()
        print('Training finished. C/A Loss = %0.6f, %0.6f'%(clossmu, alossmu))
        with open(loss_recorder_pth, 'a') as f:
            f.write('%0.6f,%0.6f,%d'%(clossmu, alossmu, t))

    torch.save(critic.state_dict(), join(save_dir, 'critic.pickle'))
    torch.save(delayed_critic.state_dict(), join(save_dir, 'delayed_critic.pickle'))
    torch.save(actor.state_dict(), join(save_dir, 'actor.pickle'))

if __name__ == '__main__':
    main()
