import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'hipposlam')))

from lib.OfflinePipelines import ImageSampling
from os.path import join


def main(run_dir):
    # Paths
    os.makedirs(run_dir, exist_ok=True)

    ImageSampling(run_dir)



if __name__ == '__main__':

    experiment_dir = join('..', 'hipposlam', 'data', 'OfflineAnalysis2')
    run_name = 'OfflineStateMapLearner'
    run_dir = join(experiment_dir, run_name)
    main(run_dir)
