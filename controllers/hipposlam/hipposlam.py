
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from lib.Environments import StateMapLearnerTaught, StateMapLearnerUmapEmbedding
from os.path import join
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def SB_PPO_Train(agent_type, simt=10000, experiment_dir='data/OnlineAnalysis'):
    # Parameters
    load_model = False  # After saving a model checkpoint, set it as True to load it.
    save_model = True
    hippomap_learn = True  # If you think you don't need to create/learn more new state, set it as False.
    load_model_name = ''  # Model checkpoint name that you want to load.
    save_model_name = 'PPO1'  # Model checkpoint name that you want to save as, for loading it and continue training next time.


    # Paths
    run_dir = join(experiment_dir, agent_type)
    os.makedirs(run_dir, exist_ok=True)
    load_hipposlam_pth = join(run_dir, '%s_hipposlam.pickle' % load_model_name)
    load_model_pth = join(run_dir, '%s.zip'%(load_model_name))
    load_umap_dir = 'data/OfflineAnalysis/OfflineStateMapLearner/umap_params'
    save_hipposlam_pth = join(run_dir, '%s_hipposlam.pickle' % save_model_name)
    save_model_pth = join(run_dir, '%s.zip' % (save_model_name))
    save_record_pth = join(run_dir, '%s' % save_model_name)
    save_trajdata_pth = join(run_dir, '%s_trajdata.pickle' % save_model_name)


    # Environment
    if agent_type == 'UmapDirect' or agent_type == 'RegressedToUmapState':
        env = StateMapLearnerUmapEmbedding(agent_type=agent_type, load_umap_dir=load_umap_dir, R=5, L=20,
                                           max_hipposlam_states=1000, save_hipposlam_pth=save_hipposlam_pth,
                                           save_trajdata_pth=save_trajdata_pth)
    elif agent_type == 'RegressedToTrueState':
        env = StateMapLearnerTaught(R=5, L=20, save_hipposlam_pth=save_hipposlam_pth,
                                    save_trajdata_pth=save_trajdata_pth)
    else:
        raise NotImplementedError('Agent type should be either "UmapDirect", "RegressedToUmapState" or "RegressedToTrueState"')


    info_keywords = ('Nstates', 'last_r', 'terminated', 'truncated', 'stuck', 'fallen', 'timeout')
    env = Monitor(env, save_record_pth, info_keywords=info_keywords)
    check_env(env)
    # Load models
    if load_model:
        env.unwrapped.load_hipposlam(load_hipposlam_pth)
        env.unwrapped.hippomap.learn_mode = hippomap_learn
        print('Hipposamp learn mode = ', str(env.unwrapped.hippomap.learn_mode))
        print('Loading hippomap. There are %d states in the hippomap' % (len(env.hippomap.sid2embed)))
        model = PPO.load(load_model_pth, env=env)
        env.hippomap.area_norm = False
    else:
        model = PPO("MlpPolicy", env, verbose=1)

    # Train
    model.learn(total_timesteps=simt)

    # Save models
    if save_model:
        model.save(save_model_pth)


def main():
    experiment_dir = join('data', 'OnlineAnalysis')
    agent_type = 'UmapDirect'
    agent_type = 'RegressedToUmapState'
    agent_type = 'RegressedToTrueState'
    SB_PPO_Train(agent_type=agent_type, simt=10000, experiment_dir=experiment_dir)
    return None

if __name__ == '__main__':
    main()
