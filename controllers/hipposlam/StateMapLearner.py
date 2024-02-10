"""tabular_qlearning controller."""

import numpy as np

from hipposlam.utils import breakroom_avoidance_policy, save_pickle
from hipposlam.Environments import StateMapLearner
from os.path import join
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def naive_avoidance():
    # Tags
    load_previous_hipposlam = False

    # Paths
    project_name = 'StateMapLearner'
    save_dir = join('data', project_name)
    os.makedirs(save_dir, exist_ok=True)
    save_replay_pth = join(save_dir, 'AvoidanceReplayBuffer.pickle')
    load_hipposlam_pth = join(save_dir, 'hipposlam.pickle')
    save_hipposlam_pth = join(save_dir, 'hipposlam.pickle')




    env = StateMapLearner(spawn='start')
    if load_previous_hipposlam:
        env.load_hipposlam(load_hipposlam_pth)
        breakpoint()
    data = {'episodes':[], 'end_r':[], 't':[], 'traj':[]}

    cum_win = 0
    while cum_win <= 50:
        print('Iter %d, cum_win = %d'%(len(data['end_r']), cum_win))
        s = env.reset()
        explist = []
        trajlist = []
        done = False
        r = None
        t = 0
        while done is False:

            # Policy
            x, y = env.x, env.y
            rotz, rota = env.rotz, env.rota
            a = breakroom_avoidance_policy(x, y, env.ds.getValue(), 0.3)

            # Step
            snext, r, done, info = env.step(a)

            # Store data
            explist.append(np.array([s, a, snext, r, done]))
            trajlist.append(np.array([x, y, np.sign(rotz)*rota, s]))

            s = snext
            t += 1

        # Store data
        data['episodes'].append(np.vstack(explist))
        data['traj'].append(np.vstack(trajlist))
        data['end_r'].append(r)
        data['t'].append(t)
        if r > 0:
            cum_win += 1
        print()
    env.save_hipposlam(save_hipposlam_pth)
    save_pickle(save_replay_pth, data)



def main():
    naive_avoidance()

    return None

if __name__ == '__main__':
    main()
