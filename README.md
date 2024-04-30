# HippoSLAM

The work investigates a SLAM algorithm using the spatial representation supported by hippocampal sequences. 

The idea originates from the following study: 

Leibold C. A model for navigation in unknown environments based on a reservoir of hippocampal sequences. Neural Networks. 2020 Apr;124:328–42.

 


# How to use the repo?

- For building the python environment and running the controller in Webots, see [How to run the controller scripts in Ubuntu](How to run the controller scripts in Ubuntu.md).
- After you have successfully run the controller scripts, you could try out other analysis scripts in [controllers/hipposlam](controllers/hipposlam). You might need the Webots python API. For that, you need to add the Webots library to your path. See [Webots documentation](https://cyberbotics.com/doc/guide/using-your-ide#pycharm) for more details. Alternatively, you can add the lines below at the beginning of the python scripts you are going to run:
```
import sys
sys.path.insert(0, r'...\Webots\lib\controller\python')
```
- More detailed explanations of the controller and analysis scripts are located at [controllers/hipposlam/README.md](controllers/hipposlam/README.md).

# Data file

For my colleagues at BCF, the data file is located at atlas:/home/yiu/hipposlam/data.zip . Simply unzip it and put the folder "data" to [controllers/hipposlam](controllers/hipposlam). The states of my analyses and training data (such as sampled images for training the UMAP) can then be recovered. 

However, due to changes in directory names, the pickled HippoSLAM models can no longer be loaded. You would need to train the RL and HippoSLAM models from scratch again. However, the analysis scripts will run.