"""four_wheeled_collision_avoidance controller.
Note: different distance = different sequence instance

"""
import sys
from os.path import join
import os
from hipposlam.sequences import Sequences
from hipposlam.utils import save_pickle
from controller import Robot, Display, Supervisor
import numpy as np

# Project tags and paths
save_tag = True
project_tag = 'avoidance'
save_dir = join('data', project_tag)
os.makedirs(save_dir, exist_ok=True)

# create the Robot instance.
robot = Supervisor()
agent_node = robot.getFromDef('AGENT')
translation_field = agent_node.getField('translation')
rotation_field = agent_node.getField('rotation')

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())
thetastep = 128
print('Basic time step = %d ms, Theta time step = %d ms'%(timestep, thetastep))

# Get Camera
cam = robot.getDevice('camera')
cam.enable(timestep)
cam.recognitionEnable(timestep)
# cam.enableRecognitionSegmentation()
width = cam.getWidth()
height = cam.getHeight()

# Get Display
display = robot.getDevice('display')

# Get wheels and Initialize wheel speed
wheels = []
wheelsNames = ['wheel1', 'wheel2', 'wheel3', 'wheel4']
for name in wheelsNames:
    wheels.append(robot.getDevice(name))
for i in range(4):
    wheels[i].setPosition(float('inf'))
    wheels[i].setVelocity(0)

# Get distance sensors
ds = []
for i in range(1, 13):
    ds_each = robot.getDevice('ds%d'%i)

    ds.append(ds_each)
    ds[i-1].enable(timestep)

# Get touch sensor
bumpers = []
bumper_names = ['touchsensor_front', 'touchsensor_back']
for name in bumper_names:
    bumper = robot.getDevice(name)
    bumpers.append(bumper)
    bumper.enable(timestep)

# Obstacle avoidance (OA)
OA_counter = timestep * 5
OA_dsThresh = 95

# Stuck-detection (STUCK) and recuse
pos = np.array(translation_field.getSFVec3f())
STUCK_thresh = 2000  # in ms
STUCK_epsilon = 1e-3
STUCK_counttime = 0
STUCK_recuseMaxT = 2000  # in ms
STUCK_recusettime = STUCK_recuseMaxT
STUCK_LR_counter = 0
STUCK_LR_thresh = 1000

# Change-direction (CHANGEDIR)
CHANGEDIR_count = np.array([0, 0])
CHANGEDIR_thresh = np.array([30, 30])
CHANGEDIR_mat = np.array([
    [1, -1],  # Avoidance speed coefficients
    [-0.3, -0.6]  # STUCK recuse speed coefficients
])

# HippoSLAM
seq = Sequences(R=5, L=10)

# Other parameters
base_speed = 15
timeCounter = 0
timeCounter2 = 0
global_count = 0
navmodes = [True, False]  # [Obstacle avoidance, Stuck recuse]

# Data Storage
data = dict(
    t = [],
    x = [],
    y = [],
    z = [],
    a = [],
    objID = [],
    X = [],

)

# Get position of feature nodes
obj_node = robot.getFromId(2919)
obj_pos = obj_node.getPosition()
obj_tpyename = obj_node.getTypeName()
print(obj_tpyename, obj_pos)



while True:
    sim_state = robot.step(timestep)

    if sim_state == -1:
        print('Reset pressed')
        if save_tag:
            # Store time-indexed data
            save_pth = join(save_dir, 'traj.pickle')
            save_pickle(save_pth, data)
            print('Traj data saved at ' + save_pth)

            # meta data
            meta = dict(stored_f=seq.stored_f, pos=dict())
            for key in seq.stored_f.keys():
                obj_node = robot.getFromId(key)
                meta[pos][key] = obj_node.getPosition()
            save_pth = join(save_dir, 'meta.pickle')
            save_pickle(save_pth, data)
            print('Meta data saved at ' + save_pth)


        sys.exit(0)


    angle = rotation_field.getSFRotation()[3]
    leftSpeed = base_speed
    rightSpeed = base_speed
    ds_vals = np.array([ds[i].getValue() for i in range(12)])
    new_pos = np.array(translation_field.getSFVec3f())


    # Obstacle avoidance
    if navmodes[0]:
        # left_blocked = ds_vals[0] < OA_dsThresh
        # right_blocked = ds_vals[2] < OA_dsThresh
        blocked = np.any(ds_vals[[0, 1, 2, 3, 11]] < OA_dsThresh)
        # # print(ds_vals)
        # if left_blocked and right_blocked:
        #     # Both left and right are obstructed, direction random.
        #     # print('Obstacle avoidance BACK')
        #     leftSpeed = base_speed * CHANGEDIR_mat[1, 0]
        #     rightSpeed = base_speed * CHANGEDIR_mat[1, 1]
        #     STUCK_LR_counter += timestep
        #     print('Left Right both obstructed, STUCK_LR_counter=%d'%(STUCK_LR_counter))
        # elif left_blocked:
        #     # Left side obstructed, turn right
        #     # print('Obstacle avoidance turn right')
        #     leftSpeed = base_speed * 1
        #     rightSpeed = base_speed * -1
        # elif right_blocked:
        #     # right side obstructed, turn left
        #     # print('Obstacle avoidance turn left')
        #     leftSpeed = base_speed * -1
        #     rightSpeed = base_speed * 1

        if blocked:
            leftSpeed = base_speed * CHANGEDIR_mat[0, 0]
            rightSpeed = base_speed * CHANGEDIR_mat[0, 1]

        else:  # Unobstructed
            # print('Obstacle avoidance forward')
            leftSpeed = base_speed
            rightSpeed = base_speed


        # Stuck detection
        dpos = new_pos - pos
        dpos_val = np.mean(np.abs(dpos))
        # print('Dpos = %0.6f, Count = %0.4f'% (dpos_val, STUCK_counttime))
        if dpos_val < STUCK_epsilon:
            STUCK_counttime += timestep

        if (STUCK_counttime > STUCK_thresh) or (STUCK_LR_counter > STUCK_LR_thresh) or (bumpers[0].getValue() >0):
            navmodes = [False, True]

    
    # Stuck recuse
    if navmodes[1]:
        print('Stuck recuse')
        if (STUCK_recusettime > 0) and (bumpers[1].getValue() < 1):
            # Start recusing from STUCK if there was not recuse attempted before
            leftSpeed = base_speed * CHANGEDIR_mat[1, 0]
            rightSpeed = base_speed * CHANGEDIR_mat[1, 1]
            STUCK_recusettime = STUCK_recusettime - timestep
        else:
            # After recuse time has passed, reset the recuse counter
            STUCK_counttime = 0
            STUCK_LR_counter = 0
            STUCK_recusettime = STUCK_recuseMaxT
            # Switch to object avoidance mode
            navmodes = [True, False]


    # CHANGEDIR behaviour
    for i in range(2):
        if CHANGEDIR_count[i] > CHANGEDIR_thresh[i]:
            
            np.random.seed(global_count)
            randvec = np.random.permutation(2)
            CHANGEDIR_mat[i, :] = CHANGEDIR_mat[i, :][randvec]
            CHANGEDIR_count[i] = 0
            np.random.seed(global_count)
            CHANGEDIR_thresh[i] = np.random.randint(10, 100)
            # print('Change behaviour %d!, new mat\n'%(i), CHANGEDIR_mat)
            # print('New Thresh = ', CHANGEDIR_thresh)

    # Theta
    timediff = timeCounter - timeCounter2
    if timediff >= thetastep:
        # print('Front=%d, Back=%d'% ( bumpers[0].getValue(), bumpers[1].getValue() ) )
        objs = cam.getRecognitionObjects()
        idlist = [obj.getId() for obj in objs]
        seq.step(idlist)

        data["t"].append(timeCounter)
        data["x"].append(new_pos[0])
        data["y"].append(new_pos[1])
        data["z"].append(new_pos[2])
        data["a"].append(angle)
        data["objID"].append(idlist)
        data["X"].append(seq.X)

        timeCounter2 = timeCounter
        pass


    # Update wheels
    wheels[0].setVelocity(leftSpeed)
    wheels[1].setVelocity(rightSpeed)
    wheels[2].setVelocity(leftSpeed)
    wheels[3].setVelocity(rightSpeed)

    CHANGEDIR_count += 1
    pos = new_pos
    global_count += 1
    timeCounter += timestep

    pass

