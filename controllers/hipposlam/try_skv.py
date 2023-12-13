import numpy as np
import skvideo.io
writer = skvideo.io.FFmpegWriter("TestVideo.mp4")

x = np.linspace(-1, 1, 640)
y = np.linspace(-1, 1, 480)
xx, yy = np.meshgrid(x, y)
zz = np.exp(-(xx ** 2 + yy ** 2))
zz = zz/zz.max() * 255
print(zz.shape)
img = np.zeros((zz.shape[0], zz.shape[1], 3)).astype(np.uint8)
zz_R = zz * 0.8
zz_B = zz * 0.2
zz_G = zz * 0.2

img[:, :, 0] = zz_R
img[:, :, 1] = zz_B
img[:, :, 2] = zz_G

for i in range(100):

    writer.writeFrame(img)
writer.close()