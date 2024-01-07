"""epuck_avoid_collision controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot

# create the Robot instance.
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Speeds
MAX_SPEED = 6.28

# Get wheels and Initialize wheel speed
wheels = []
wheelsNames = ['left wheel motor', 'right wheel motor']
for name in wheelsNames:
    wheels.append(robot.getDevice(name))
for i in range(2):
    wheels[i].setPosition(float('inf'))
    wheels[i].setVelocity(0)


# Main loop:
# - perform simulation steps until Webots is stopping the controller
a = 1
v = 0
dt = 32 * 1e-3
while robot.step(timestep) != -1:

    v += a * dt
    print('timet', robot.getTime())
    print('v = ', v)
    for i in range(2):

        wheels[i].setVelocity(v)
        print('Velocity %d = %0.4f' % (i, wheels[i].getVelocity()))


# Enter here exit cleanup code.
