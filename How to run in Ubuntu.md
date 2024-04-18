This documentation guides how to run the simulations in Ubuntu.

Prerequisites:
- Anaconda

You can also use pyenv for creating a python environment. 


# Installing Webots
1. Go to https://github.com/cyberbotics/webots/releases
2. It is recommanded to choose the R2023b version, on which the project is developed.
3. Download and install "webots_2023b_amd64.deb" (or other versions depending on individual cases).

# Clone the Repo
```bash
$ git clone https://github.com/yyhhoi/HippoSLAM/tree/onlyhipposlam
```

# Set up Conda environment
Go to the directory of the repo (./HippoSLAM) and install the packages with pip:
```bash
$ cd HippoSLAM
$ conda create -n hipposlam python=3.11.7
$ conda activate hipposlam
$ pip install -r requirements.txt --ignore-requires-python
```

# Run a  script in Webots
### Open Webots and load the world file
1. In the top menu bar, click "File" 
2. Open "World" 
3. Open "HippoSLAM/worlds/outdoor.wbt"


### Use our Conda environment in the Webots simulation
1. In the top menu bar, click "Preferences" 
2. in "Python command", enter "\<PATH\>/anaconda3/envs/hipposlam/bin/python3"
3. The path above should be your hipposlam conda environment executable. Replace \<PATH\> with the path to your Anaconda directory. By default it is the home directory.
   
	
### Select the python controller script for our robot.
1. On the left sidebar, select and expand the robot item named "DEF AGENT Robot". 
2. Click "controller" field in the expanded tree. 
3. The controller name will be shown in the field below. By default, it is "<extern>", which means the controller script is provided by an external IDE. 
4. To simply try out the script, click "Select..." and change it to "hipposlam", which will runs the script at "controllers/hipposlam/hipposlam.py" within the simulator.

### Run the script
1. At the top bar, click the play-forward icon to run the script with normal speed (press "Ctrl + 2" in Ubuntu).
2. To run the script faster , click the fast-forward icon  (press "Ctrl + 3" in Ubuntu).
3. If the script runs successfully, which means the environment has set up correctly.

### (Recommended) Use your own IDE
1. To use your own IDE, see the [official documentation](https://cyberbotics.com/doc/guide/using-your-ide). 
2. It is better to use your own IDE, since Webots only looks for the controller script at "controller_name/controller_name.py". To run other scripts for your analysis, you would need your IDE imported with Webots controller libraries.

