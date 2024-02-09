"""tabular_qlearning controller."""

import numpy as np

from hipposlam.utils import breakroom_avoidance_policy, save_pickle
from hipposlam.Environments import StateMapLearner
from os.path import join
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def naive_avoidance():
    # Paths
    save_dir = join('data', 'StateMapLearner')
    os.makedirs(save_dir, exist_ok=True)
    save_hipposeq_pth = join(save_dir, 'HippoSLAMseq.pickle')
    save_hippomap_pth = join(save_dir, 'HippoSLAMmap.pickle')



    env = StateMapLearner(spawn='start')
    data = {'traj':[], 'end_r':[], 't':[]}
    cum_win = 0
    while cum_win <= 5:
        print('Iter %d, cum_win = %d'%(len(data['end_r']), cum_win))
        s = env.reset()
        trajlist = []
        done = False
        r = None
        t = 0
        while done is False:

            # Policy
            x, y, z = env._get_translation()
            a = breakroom_avoidance_policy(x, y, env.ds.getValue(), 0.3)

            # Step
            snext, r, done, info = env.step(a)

            # Store data
            trajlist.append(np.array([s, a, snext, r, done]))

            s = snext
            t += 1

        # Store data
        traj = np.vstack(trajlist)
        data['traj'].append(traj)
        data['end_r'].append(r)
        data['t'].append(t)
        if r > 0:
            cum_win += 1

    save_pickle(save_hipposeq_pth, env.hipposeq)
    save_pickle(save_hippomap_pth, env.hippomap)



def main():
    naive_avoidance()

    return None

if __name__ == '__main__':
    main()
