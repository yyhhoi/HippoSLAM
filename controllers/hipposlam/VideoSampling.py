from controller import Supervisor, Camera
import sys
import numpy as np
import os
from os.path import join
import skvideo.io

TIME_STEP = 64
robot = Supervisor()
ds = []
dsNames = ['front right distance sensor', 'front left distance sensor']
for i in range(2):
    ds.append(robot.getDevice(dsNames[i]))
    ds[i].enable(TIME_STEP)
wheels = []
wheelsNames = ['front left wheel motor',
               'front right wheel motor',
               'rear left wheel motor',
               'rear right wheel motor']
for i in range(4):
    wheels.append(robot.getDevice(wheelsNames[i]))
    wheels[i].setPosition(float('inf'))
    wheels[i].setVelocity(0.0)


# position-based stuck-detection (STUCK)
agent_node = robot.getFromDef('AGENT')
translation_field = agent_node.getField('translation')
pos = np.array(translation_field.getSFVec3f())
STUCK_thresh = 10
STUCK_epsilon = 1e-4
STUCK_count = 0
STUCK_duration = 20


# Rotation
rotation_field = agent_node.getField('rotation')

# Change-direction (CHANGEDIR)
CHANGEDIR_count = np.array([0, 0])
CHANGEDIR_thresh = np.array([30, 30])
CHANGEDIR_mat = np.array([
    [1, -1],  # Avoidance speed coefficients
    [-0.5, -1]  # STUCK fallback speed coefficients
])

# Camera
cam = robot.getDevice('camera rgb')
cam.enable(TIME_STEP)
# writer = skvideo.io.FFmpegWriter("SimVideo.mp4")

avoid_thresh = 1
base_speed = 15
global_count = 0
desktop_dir = os.environ['DESKTOP_PATH']
data_dir = join(desktop_dir, 'imgs')
pos_data, img_data = [], []
while True:
    sim_state = robot.step(TIME_STEP)

    if sim_state == -1:
        print('Reset pressed')
        pos_data = np.stack(pos_data)
        img_data = np.stack(img_data)
        np.savez_compressed(join(data_dir, 'data.npz'), pos=pos_data, img=img_data)
        sys.exit(0)

    ds_r, ds_l = ds[0].getValue(), ds[1].getValue()
    leftSpeed = base_speed
    rightSpeed = base_speed


    # Stuck detection
    new_pos = np.array(translation_field.getSFVec3f())
    dpos = new_pos - pos
    dpos_val = np.mean(np.abs(dpos))
    if dpos_val < STUCK_epsilon:
        STUCK_count += 1

    if (STUCK_count > STUCK_thresh) and (STUCK_duration > 0):
        leftSpeed = base_speed * CHANGEDIR_mat[1, 0]
        rightSpeed = base_speed * CHANGEDIR_mat[1, 1]
        STUCK_duration = STUCK_duration-1

    if (STUCK_count > STUCK_thresh) and (STUCK_duration == 0):
        STUCK_count = 0
        STUCK_duration = 10

    # Obstacle avoidance
    if STUCK_duration == 10:
        if (ds_r < avoid_thresh) or (ds_l < avoid_thresh):
            leftSpeed = base_speed * CHANGEDIR_mat[0, 0]
            rightSpeed = base_speed * CHANGEDIR_mat[0, 1]

    # CHANGEDIR behaviour
    for i in range(2):
        if CHANGEDIR_count[i] > CHANGEDIR_thresh[i]:
            # np.random.seed(global_count)
            randvec = np.random.permutation(2)
            CHANGEDIR_mat[i, :] = CHANGEDIR_mat[i, :][randvec]
            CHANGEDIR_count[i] = 0
            # np.random.seed(global_count)
            CHANGEDIR_thresh[i] = np.random.randint(10, 100)
            # print('Change behaviour %d!, new mat\n'%(i), CHANGEDIR_mat)
            # print('New Thresh = ', CHANGEDIR_thresh)


    # CAMERA
    if global_count %4 == 0:
        img = cam.getImage()
        imgarr = np.frombuffer(img, np.uint8).reshape((cam.getHeight(), cam.getWidth(), 4))
        img_data.append(imgarr)
        rothd = rotation_field.getSFRotation()
        cxyza = np.array([global_count, pos[0], pos[1], pos[2], rothd[3]])
        pos_data.append(cxyza)

    # Update
    wheels[0].setVelocity(leftSpeed)
    wheels[1].setVelocity(rightSpeed)
    wheels[2].setVelocity(leftSpeed)
    wheels[3].setVelocity(rightSpeed)
    global_count += 1
    CHANGEDIR_count += 1
    pos = new_pos
