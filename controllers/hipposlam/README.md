# HippoSLAM controller and analysis


## [hipposlam.py](hipposlam.py)

Webots controller file. The simulation modes include

- UmapDirect
- RegressedToTrueState
- RegressedToUmapState
  
The script requires Webots simulator to run. Behaviour data and reinforcement learning model are saved checkpoint by checkpoint.

For "UmapDirect" and "RegressedToUmapState", a pre-trained Umap model is required. It can be obtained by the pipeline in [OfflineStateMapLearner.py](OfflineStateMapLearner.py).


## [OfflineStateMapLearner.py](OfflineStateMapLearner.py):
    
An offline analysis pipeline to evaluate the spatial representation of the hipposlam, without reinforcement learning, following the steps:

1. Sample images from the environment using a random agent.
2. Infer MobileNetV3 embeddings from the sampled images.
3. Train UMAP embeddings from the MobileNetV3 embeddings.
4. Use the UMAP embeddings and the random agent's trajectory to train the state decoder (J) in the HippoSLAM, as if the robot went through the images in real time. 
5. Visualize the spatial specificity of the UMAP embeddings and HippoSLAM predicted states.  

Except step (1), the pipeline does not require Webots simulation (offline).


## [StateMapper_Analysis.ipynb](StateMapper_Analysis.ipynb)

Analysis notebook performing the following:

1. Evaluate the reinforcement learning performances of "UmapDirect", "RegressedToTrueState" and "RegressedToUmapState".
2. Plot the final poses of the robot when it fails the episodes in "RegressedToUmapState".
   - Result: The episode fails mainly because of losing balance and tripping, or getting stuck somewhere (such as the at the fence).
   
3. Plot the accuracy of hipposlam prediction to the Umap embedding states in "RegressedToUmapState".
   - Result: The wining and failing episodes have similar accuracy. The accuracy of hipposlam prediction is not responsible to the poorer performance.
4. Plot trajectories of the robot during winning episodes.
5. Plot the robot's poses at hipposlam predicted states and Umap states.
   - Result: both hipposlam predicted states and Umap states occur at roughly the same robot poses. Again, the poorer performance in "RegressedToUmapState" cannot be attributed to hipposlam prediction accuracy.
6. Plot spatial (positional and directional) specificity of hipposlam predicted states (or modified to umap embedding states).

## [SIFT_PatternMatchingExample.ipynb](SIFT_PatternMatchingExample.ipynb)

Example code for performing feature matching with SIFT.


# Tips to speed up the simulation

1. In the simulation, action and state inference are performed every 1024ms (duration of a theta cycle). However, by default, the camera feed is updated at the world's basic time step (32ms). It causes a huge cost of compute. One way to speed by the simulation speed up (up to >10x) is by modifying [StateMapLearner.camera_timestep](lib/Environments.py#L246) to be ~64-512ms. However, it could cause some mismatch between the timing of step() function and update of camera image, causing the action to be sampled based on a wrong image. You would need to ensure in the .step() function, the camera image is updated BEFORE the state is inferred.
2. At [StateMapLearnerUmapEmbedding.get_obs_base()](lib/Environments.py#L458), the MobileNetV3 and UMAP do not need to infer the image every step. Their inferences are relatively expensive compared to HippoSLAM state decoder, and hence, the MobileNetV3 and UMAP can be called every 2-5th time, while HippoSLAM can be called every time. In this case, the HippoSLAM will infer the state without a teacher from the new image embedding, going blind, using its 'working memory' to decode the state. The computation speed comes at the cost of lower HippoSLAM state decoding accuracy. I recommend inferring with MobileNetV3 and UMAP every 2th time step. More than 2th time step, the accuracy significantly degrades.          