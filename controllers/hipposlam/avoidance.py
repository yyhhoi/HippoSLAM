"""four_wheeled_collision_avoidance controller.
Note: different distance = different sequence instance

"""
import sys
from os.path import join
import os
from hipposlam.sequences import Sequences
from hipposlam.utils import save_pickle
from controller import Supervisor
from hipposlam.vision import SiftMemory
from hipposlam.kinematics import compute_steering, convert_steering_to_wheelacceleration_nonlinear


import numpy as np


# Project tags and paths
save_tag = True
reobserve = False
project_tag = 'Avoidance_CombinedCues_theta1024'
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
thetastep = 1024
print('Basic time step = %d ms, Theta time step = %d ms'%(timestep, thetastep))

# Get Camera
camera_timestep = thetastep
cam = robot.getDevice('camera')
cam.enable(camera_timestep)
cam.recognitionEnable(camera_timestep)
width = cam.getWidth()
height = cam.getHeight()
# cam.enableRecognitionSegmentation()

# Sift scene recognition
newMemoryThresh = 0.1
memoryActivationThresh = 0.2
SM = SiftMemory(newMemoryThresh=newMemoryThresh,
                memoryActivationThresh=memoryActivationThresh)

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


# Get distance sensors and their orientations
ds = []
ds_names = []
ds_angles = []
ds_maxs = []
robot_children_field = agent_node.getField('children')
num_fields = robot_children_field   .getCount()
print('There are %d fields in Robot'%(num_fields))
for i in range(num_fields):
    node = robot_children_field.getMFNode(i)
    nodetypename = node.getTypeName()
    print('%d Node type = %s'%(i, nodetypename))

    if (i >= 8) and (i <= 20):
        ds_rotation_field = node.getField('rotation')
        name_field = node.getField('name')
        dsangle = ds_rotation_field.getSFRotation()[3]
        dsname = node.getField('name').getSFString()
        print('Name = %s, angle = %0.4f'%(dsname, dsangle))
        ds_names.append(dsname)
        ds_angles.append(dsangle)
        ds_each = robot.getDevice(dsname)
        ds_each.enable(timestep)
        ds.append(ds_each)
        ds_maxs.append(ds_each.getMaxValue())
ds_angles = np.array(ds_angles)
ds_maxs = np.array(ds_maxs)


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
STUCK_thresh = 3000  # in ms
STUCK_epsilon = 1e-3
STUCK_counttime = 0
STUCK_recuseMaxT = 1000  # in ms
STUCK_recusettime = STUCK_recuseMaxT
STUCK_LR_counter = 0
STUCK_LR_thresh = 1000

# Change-direction (CHANGEDIR)
CHANGEDIR_count = 0
CHANGEDIR_thresh = 10e3  # every ?s
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
dist_sep = floor_diag / 4
obj_dist = 2

# Data Storage
data = dict(
    t = [],
    x = [],
    y = [],
    z = [],
    rotx = [],
    roty = [],
    rotz = [],
    rota = [],
    objID = [],
    objID_dist= [],
    f_sigma=[],
    X = [],

)

fpos_dict = dict()
vl = 10
vr = 10
dt = timestep * 1e-3
acce_k = 40
ds_epsilon = 0.1
while True:

    sim_state = robot.step(timestep)
    time_s = robot.getTime()

    if sim_state == -1:
        print('Reset pressed')
        if save_tag:
            # Store time-indexed data
            save_pth = join(save_dir, 'traj.pickle')
            save_pickle(save_pth, data)
            print('Traj data saved at ' + save_pth)

            meta = dict(stored_f=seq.stored_f, fpos=fpos_dict, dist_level=dist_sep, seqR=seq.R, seqL=seq.L, obj_dist=obj_dist)
            save_pth = join(save_dir, 'meta.pickle')
            save_pickle(save_pth, meta)
            print('Meta data saved at ' + save_pth)


        sys.exit(0)


    rotx, roty, rotz, rota = rotation_field.getSFRotation()
    leftSpeed = base_speed
    rightSpeed = base_speed
    ds_vals = np.array([ds[i].getValue() for i in range(12)])
    new_pos = np.array(translation_field.getSFVec3f())

    if (np.abs(rotx) > 0.5) or (np.abs(roty) > 0.5):
        print(rotx, roty, rotz, rota)
    #     robot.simulationResetPhysics()
    #     robot.simulationReset()

    # Obstacle avoidance
    if navmodes[0]:
        steering_a, ds_amp = compute_steering(ds_vals, ds_maxs, ds_angles, epsilon=ds_epsilon)
        ds_mask = ds_amp > ds_epsilon
        if np.any(ds_mask):
            ds_amp_mean = np.mean(ds_amp[ds_amp > ds_epsilon])

            acce_amp = (acce_k ) * ( ds_amp_mean / 100 )
        else:
            ds_amp_mean = 0
            acce_amp = 0

        np.random.seed(global_count)
        acce_noise = np.random.normal(0, 5, size=2)

        left_acce, right_acce = convert_steering_to_wheelacceleration_nonlinear(steering_a, acce_amp)
        vl_nat = (10-vl)
        vr_nat = (10-vr)
        dvl = vl_nat + left_acce  + acce_noise[0]
        dvr = vr_nat + right_acce + acce_noise[1]
        vl += dvl * dt
        vr += dvr * dt
        vl = np.clip(vl, a_min=-15, a_max=15)
        vr = np.clip(vr, a_min=-15, a_max=15)
        leftSpeed, rightSpeed = vl, vr

        # print('Steering %0.2f, ds_amp_mean=%0.4f, acce_amp=%0.2f'%(np.rad2deg(steering_a), ds_amp_mean, acce_amp))
        # print('vl_nat=%0.4f, vr_nat=%0.4f'%(vl_nat, vr_nat))
        # print('left_acce=%0.4f, right_acce=%0.4f'%(left_acce, right_acce))
        # print('dvl=%0.4f, dvr=%0.4f'%(dvl, dvr))
        # print('leftSpeed=%0.2f, rightSpeed=%0.2f'%(leftSpeed, rightSpeed))


        # Stuck detection
        dpos = new_pos - pos
        dpos_val = np.mean(np.abs(dpos))
        if dpos_val < STUCK_epsilon:
            STUCK_counttime += timestep
            # print('Stuck added by ', timestep)
        # print('Stuck countimt = ', STUCK_counttime, ', dpos = ', dpos_val, ', bumper=', bumpers[0].getValue())
        if (STUCK_counttime > STUCK_thresh) or (bumpers[0].getValue() >0):
            navmodes = [False, True]

    
    # Stuck recuse
    if navmodes[1]:
        # print('Stuck recuse: %d'%(STUCK_recusettime))
        if (STUCK_recusettime > 0):
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
            vl, vr = 10, 10


    # CHANGEDIR behaviour
    if CHANGEDIR_count > CHANGEDIR_thresh:
        # print('Change mat')
        np.random.seed(global_count)
        negval = np.random.uniform(-1, 0)
        np.random.seed(global_count + 1)
        posval = np.random.uniform(0, 1)
        avoidance_vals = np.array([posval, negval])
        np.random.seed(global_count + 2)
        avoidance_vals = avoidance_vals[np.random.permutation(2)]
        np.random.seed(global_count + 3)
        stuck_ranvec = np.random.permutation(2)
        stuck_vals = np.array([0.5, -0.5])[stuck_ranvec]
        CHANGEDIR_mat = np.vstack([avoidance_vals, stuck_vals])
        CHANGEDIR_count = 0
    else:
        np.random.seed(global_count)
        CHANGEDIR_count += np.random.randint(10, 100)

    # Theta
    timediff = timeCounter - timeCounter_theta
    time_ms = int(time_s * 1e3)
    if (time_ms % thetastep) == 0:
        # print('Time = %0.2f ms'%(time_ms))

        # # ======================= Sift scene recognition ======================================
        # imgobj = cam.getImage()
        # imgtmp = np.frombuffer(imgobj, np.uint8).reshape((cam.getHeight(), cam.getWidth(), 4))
        # gray = cv.cvtColor(imgtmp, cv.COLOR_BGRA2GRAY)
        # np.save(join(img_dir, '%d.npy'%(time_ms)), gray)
        # idlist, maxscore, noveltag = SM.observe(gray)
        # seq.step(idlist)
        # print('Time = %d ms, Thresh=%0.2f' % (timeCounter, SM.newMemoryThresh))
        # print('Num Obs: ', len(SM.obs_list))
        # print('Activated ID: ', idlist)
        # print('Match score: ', maxscore)
        # print('Descriptor sizes:\n', [len(des) for des in SM.obs_list])
        # ========================================================================================


        # # =========================== Bulti-in object recognition ===========================
        # Get object ids
        objs = cam.getRecognitionObjects()
        idlist = [obj.getId() for obj in objs]

        # Distance from robot to the object
        id2list = []
        closeIDlist = []
        farIDlist = []
        for objid in idlist:

            # Obtain object position
            obj_node = robot.getFromId(objid)
            objpos = obj_node.getPosition()

            # Store object positions
            if str(objid) not in fpos_dict:
                fpos_key = '%d'%objid
                fpos_dict[fpos_key] = objpos
                # print('Insert Id=%s with position ' % (fpos_key), objpos)

            # Compute distance
            dist = np.sqrt((new_pos[0] - objpos[0]) ** 2 + (new_pos[1] - objpos[1])**2)
            # print('Dist = %0.2f, obj_dist = %0.2f'%(dist, obj_dist))
            if dist < obj_dist:
                # print('Close object %d added'%(objid))
                closeIDlist.append('%d'%(objid))
            else:
                # print('Distant object %d added' % (objid))
                farIDlist.append('%d'%objid)

        close_to_dist_list = []
        for c in closeIDlist:
            for d in farIDlist:
                cd = c + "_" + d
                close_to_dist_list.append(cd)
            # dist_level = int(dist / dist_sep)  # discretized distance
            # id2list.append('%d_%d'%(objid, dist_level))
        seq.step(close_to_dist_list)
        # # ========================================================================================

        data["t"].append(time_ms)
        data["x"].append(new_pos[0])
        data["y"].append(new_pos[1])
        data["z"].append(new_pos[2])
        data["rotx"].append(rotx)
        data["roty"].append(roty)
        data["rotz"].append(rotz)
        data["rota"].append(rota)
        data["objID"].append(idlist)
        data["objID_dist"].append(id2list)
        data['f_sigma'].append(seq.f_sigma.copy())
        data["X"].append(seq.X)
        timeCounter_theta = timeCounter
        pass

    # # Store image
    # if timeCounter - timeCounter_cam >= 1000:
    #     imgpth = join(img_dir, '%dms.jpg' % (timeCounter))
    #     img = cam.getImage()
    #     cam.saveImage(imgpth, 100)
    #     timeCounter_cam = timeCounter

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

