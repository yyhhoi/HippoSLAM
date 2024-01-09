import numpy as np
from hipposlam.utils import read_pickle
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join
import os
import matplotlib as mpl
from matplotlib import cm
from tqdm import tqdm



project_tag = 'Avoidance'
data_dir = join('data', project_tag)
plot_dir = join('plots', project_tag, 'ratemaps')
os.makedirs(plot_dir, exist_ok=True)
trajdata = read_pickle(join(data_dir, 'traj.pickle'))
metadata = read_pickle(join(data_dir, 'meta.pickle'))
stored_f = metadata['stored_f']
f_pos = metadata['fpos']


trajdf = pd.DataFrame(trajdata)
trajdf['X_Nrow'] = trajdf['X'].apply(lambda x : x.shape[0])
trajdf['a'] = trajdf['rota'] * trajdf['rotz']
print('Max x row ', trajdf['X_Nrow'].max())
trajdf

from scipy.ndimage import gaussian_filter
from hipposlam.comput_utils import divide_ignore
def midedges(edges):
    return (edges[:-1] + edges[1:]) / 2


class BayesianDecoder:
    def __init__(self, xmin, xmax, ymin, ymax, dp, bodysd):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.dp = dp
        self.bodysd = bodysd
        self.bodysd_ind = self.bodysd/self.dp
        self.xedges = np.arange(xmin, xmax+dp, dp)
        self.yedges = np.arange(ymin, ymax+dp, dp)
        self.xedm = midedges(self.xedges)
        self.yedm = midedges(self.yedges)

    def compute_occupancy(self, x, y):

        H2d, _, _ = np.histogram2d(x, y, bins=(self.xedges, self.yedges))
        H2d_smooth = gaussian_filter(H2d, sigma=self.bodysd_ind, mode='constant', cval=0)
        return H2d, H2d_smooth

    def compute_spikecounts(self, xsp, ysp):
        H2d, _, _ = np.histogram2d(xsp, ysp, bins=(self.xedges, self.yedges))
        H2d_smooth = gaussian_filter(H2d, sigma=self.bodysd_ind, mode='constant', cval=0)
        return H2d, H2d_smooth

    def compute_ratemap(self, occ, spcounts):
        return divide_ignore(spcounts, occ)

# Compute occpancy
bodysd = 0.15  # body length of the robot = 0.3m
dp = 0.1
xmin = int(np.floor(trajdf['x'].min() - bodysd))
xmax = int(np.ceil(trajdf['x'].max() + bodysd))
ymin = int(np.floor(trajdf['y'].min() - bodysd))
ymax = int(np.ceil(trajdf['y'].max() + bodysd))
print(xmin, xmax, ymin, ymax)

BD = BayesianDecoder(xmin, xmax, ymin, ymax, dp, bodysd)


occ, occ_gau = BD.compute_occupancy(trajdf['x'].to_numpy(), trajdf['y'].to_numpy())
# fig, ax = plt.subplots(1, 2, figsize=(10, 4))
# im0 = ax[0].pcolormesh(BD.xedges, BD.yedges, occ.T)
# im1 = ax[1].pcolormesh(BD.xedges, BD.yedges, occ_gau.T)
# plt.colorbar(im0, ax=ax[0])
# plt.colorbar(im1, ax=ax[1])


Num_Fnodes = trajdf['X_Nrow'].max()

xdict = dict()
ydict = dict()
adict = dict()
fposdict = dict()
for i in range(trajdf.shape[0]):

    Xmat = trajdf['X'][i]
    x = trajdf['x'][i]
    y = trajdf['y'][i]
    a = trajdf['a'][i]
    active_rowIDs = np.where(np.sum(Xmat, axis=1) > 0)[0]

    for rowid in active_rowIDs:
        nodekey = [str(k) for k, v in stored_f.items() if v == rowid][0]
        fposdict[rowid] = f_pos[nodekey.split('_')[0]]


        if rowid in xdict:
            xdict[rowid].append(x)
            ydict[rowid].append(y)
            adict[rowid].append(a)

        else:
            xdict[rowid] = [x]
            ydict[rowid] = [y]
            adict[rowid] = [a]


num_fnodes = trajdf['X'][trajdf.shape[0]-1].shape[0]

for i in tqdm(range(num_fnodes)):
    xsp = xdict[i]
    ysp = ydict[i]

    spmap, spmap_gau = BD.compute_spikecounts(xsp, ysp)

    ratemap = BD.compute_ratemap(occ_gau, spmap_gau)

    fig, ax = plt.subplots(2, 2, figsize=(8, 6), sharex=True, sharey=True, facecolor='w')
    im0 = ax[0, 0].imshow(occ_gau.T)
    plt.colorbar(im0, ax=ax[0, 0])

    # ax[0, 1].plot(trajdf['x'], trajdf['y'])
    # im1 = ax[0, 1].scatter(xsp, ysp, color='r', marker='x', alpha=0.2)
    # plt.colorbar(im1, ax=ax[0, 1])

    im2 = ax[1, 0].imshow(spmap_gau.T)
    # ax[1, 0].plot(trajdf['x'], trajdf['y'], alpha=0.2, color='w')
    plt.colorbar(im2, ax=ax[1, 0])

    im3 = ax[1, 1].imshow(ratemap.T)
    # ax[1, 1].plot(trajdf['x'], trajdf['y'], alpha=0.2, color='w')
    plt.colorbar(im3, ax=ax[1, 1])
    fig.tight_layout()
    plt.show()
    break



