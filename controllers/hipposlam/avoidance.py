"""four_wheeled_collision_avoidance controller.
Note: different distance = different sequence instance

"""
import sys
from os.path import join
import os
from hipposlam.sequences import Sequences
from hipposlam.utils import save_pickle
from controller import Supervisor
import numpy as np


# Project tags and paths
save_tag = False
reobserve = False
project_tag = 'avoidance_NoReObserve'
save_dir = join('data', project_tag)
os.makedirs(save_dir, exist_ok=True)
img_dir = join(save_dir, 'imgs')
os.makedirs(img_dir, exist_ok=True)

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
camera_timestep = 64
cam = robot.getDevice('camera')
cam.enable(camera_timestep)
cam.recognitionEnable(camera_timestep)
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
CHANGEDIR_count = 0
CHANGEDIR_thresh = 5e3  # every 5s
CHANGEDIR_mat = np.array([
    [1, -1],  # Avoidance speed coefficients
    [-0.3, -0.6]  # STUCK recuse speed coefficients
])

# HippoSLAM
seq = Sequences(R=5, L=10, reobserve=reobserve)

# Other parameters
base_speed = 15
timeCounter = 0
timeCounter_theta = 0
timeCounter_cam = 0
global_count = 0
navmodes = [True, False]  # [Obstacle avoidance, Stuck recuse]
floor_diag = (7.7 **2 + 12.9 ** 2) ** (1/2)
dist_sep = floor_diag / 7


# Data Storage
data = dict(
    t = [],
    x = [],
    y = [],
    z = [],
    a = [],
    objID = [],
    objID_dist= [],
    f_sigma=[],
    X = [],

)

fpos_dict = dict()

while True:
    sim_state = robot.step(timestep)

    if sim_state == -1:
        print('Reset pressed')
        if save_tag:
            # Store time-indexed data
            save_pth = join(save_dir, 'traj.pickle')
            save_pickle(save_pth, data)
            print('Traj data saved at ' + save_pth)

            meta = dict(stored_f=seq.stored_f, fpos=fpos_dict, dist_level=dist_level)
            save_pth = join(save_dir, 'meta.pickle')
            save_pickle(save_pth, meta)
            print('Meta data saved at ' + save_pth)


        sys.exit(0)


    angle = rotation_field.getSFRotation()[3]
    leftSpeed = base_speed
    rightSpeed = base_speed
    ds_vals = np.array([ds[i].getValue() for i in range(12)])
    new_pos = np.array(translation_field.getSFVec3f())


    # Obstacle avoidance
    if navmodes[0]:
        blocked = np.any(ds_vals[[0, 1, 2, 3, 11]] < OA_dsThresh)


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
    if CHANGEDIR_count > CHANGEDIR_thresh:
        np.random.seed(global_count)
        negval = np.random.uniform(-1, 0)
        np.random.seed(global_count + 1)
        posval = np.random.uniform(0, 1)
        avoidance_vals = np.array([posval, negval])
        np.random.seed(global_count + 2)
        avoidance_vals = avoidance_vals[np.random.permutation(2)]
        np.random.seed(global_count + 3)
        stuck_negvals = np.random.uniform(-1, -0.1, size=2)
        CHANGEDIR_mat = np.vstack([avoidance_vals, stuck_negvals])
        CHANGEDIR_count = 0
    else:
        np.random.seed(global_count)
        CHANGEDIR_count += np.random.randint(10, 100)

    # Theta
    timediff = timeCounter - timeCounter_theta
    if timediff >= thetastep:

        # Get object ids
        objs = cam.getRecognitionObjects()
        idlist = [obj.getId() for obj in objs]

        # Distance from robot to the object
        id2list = []
        for objid in idlist:

            # Obtain object position
            obj_node = robot.getFromId(objid)
            objpos = obj_node.getPosition()

            # Store object positions
            if str(objid) not in fpos_dict:
                fpos_key = '%d'%objid
                fpos_dict[fpos_key] = objpos
                print('Insert Id=%s with position ' % (fpos_key), objpos)

            # Compute distance
            dist = np.sqrt((new_pos[0] - objpos[0]) ** 2 + (new_pos[1] - objpos[1])**2)
            dist_level = int(dist / dist_sep)  # discretized distance
            id2list.append('%d_%d'%(objid, dist_level))

        # print('step ', id2list)
        seq.step(id2list)

        data["t"].append(timeCounter)
        data["x"].append(new_pos[0])
        data["y"].append(new_pos[1])
        data["z"].append(new_pos[2])
        data["a"].append(angle)
        data["objID"].append(idlist)
        data["objID_dist"].append(id2list)
        data['f_sigma'].append(seq.f_sigma.copy())
        data["X"].append(seq.X)



        timeCounter_theta = timeCounter
        pass

    # Store image
    if timeCounter - timeCounter_cam >= 1000:
        imgpth = join(img_dir, '%dms.jpg' % (timeCounter))
        img = cam.getImage()
        cam.saveImage(imgpth, 100)
        timeCounter_cam = timeCounter

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

