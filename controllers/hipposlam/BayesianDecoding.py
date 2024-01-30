import numpy as np
from hipposlam.utils import read_pickle
from hipposlam.sequences import Sequences
from hipposlam.comput_utils import circular_gau_filter, divide_ignore, midedges, Arena
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join
import os
import matplotlib as mpl
from matplotlib import cm
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from scipy.special import factorial
def logPois(r, k, epsilon=1e-6):
    out = k * np.log(r+epsilon) - (r+epsilon) - factorial(k)
    return out




# Paths and data===============
debug_plot_tag = True
project_tag = 'Avoidance'
data_dir = join('data', project_tag)
plot_dir = join('plots', project_tag, 'BayesianDecoding')
os.makedirs(plot_dir, exist_ok=True)
trajdata = read_pickle(join(data_dir, 'traj.pickle'))
metadata = read_pickle(join(data_dir, 'meta.pickle'))
metadata['seqR'] = 5 # For now. Remove this line next time after seqR is stored in the metadata
metadata['seqL'] = 10 # For now. Remove this line next time after seqL is stored in the metadata
seqR = metadata['seqR']
seqL = metadata['seqL']
fkey2id_dict = metadata['stored_f']
id2fkey_dict = {val:key for key, val in fkey2id_dict.items()}
f_pos = metadata['fpos']
trajdf = pd.DataFrame(trajdata)
trajdf['X_Nrow'] = trajdf['X'].apply(lambda x : x.shape[0])
trajdf['a'] = trajdf['rota'] * trajdf['rotz']


# Occpancy ===========================
bodysd = 0.15  # body length of the robot = 0.3m
dp = 0.1
da = 2*np.pi/18
xmin = np.floor(trajdf['x'].min() * 10) / 10
xmax = np.ceil(trajdf['x'].max() * 10) / 10
ymin = np.floor(trajdf['y'].min() * 10) / 10
ymax = np.ceil(trajdf['y'].max() * 10) / 10
amin, amax = -np.pi, np.pi

BD = Arena(xmin, xmax, ymin, ymax, amin, amax, dp, da, bodysd)
occ, edges3d = BD.compute_histogram3d(trajdf['x'].to_numpy(), trajdf['y'].to_numpy(), trajdf['a'].to_numpy())
occ_p = occ.sum(axis=2)
occ_a = occ.sum(axis=0).sum(axis=0)
if debug_plot_tag:
    fig = plt.figure(figsize=(8, 4), facecolor='w')
    ax1 = fig.add_axes([0, 0, 0.4, 0.9])
    ax2 = fig.add_axes([0.6, 0, 0.3, 0.6], polar=True)
    cbar_ax = fig.add_axes([0.41, 0, 0.04, 0.9])
    im1 = ax1.pcolormesh(BD.xedges, BD.yedges, occ_p.T, cmap='hot')
    cb = fig.colorbar(im1, cax=cbar_ax)
    ax2.bar(BD.aedm, occ_a, width=da)
    fig.savefig(join(plot_dir, 'Occupancy.png'))

# Organize Spike Data ====================
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

    if Xmat.shape[0] < 1:
        continue
    fnode_ids, sigma_ids = Sequences.X2sigma(Xmat, seqR, sigma_state=False)

    for fnode_id, sigma_id in zip(fnode_ids, sigma_ids):
        nodekey = id2fkey_dict[fnode_id]
        ensem_key = '%s-%d'%(nodekey, sigma_id)

        fposdict[ensem_key] = f_pos[nodekey.split('_')[0]]

        if ensem_key in xdict:
            xdict[ensem_key].append(x)
            ydict[ensem_key].append(y)
            adict[ensem_key].append(a)

        else:
            xdict[ensem_key] = [x]
            ydict[ensem_key] = [y]
            adict[ensem_key] = [a]



# Compute Rate Map =================
plot_ratemap = True
plot_dir_ratemap = join(plot_dir, 'ratemaps')
os.makedirs(plot_dir_ratemap, exist_ok=True)
print('Compute ratemaps')
print('Save at ', plot_dir_ratemap)
num_ensem = len(xdict.keys())
id2ensemkey = [ensem_key for ensem_key in xdict.keys()]
ensemkey2id = dict()
for i, ensem_key in enumerate(id2ensemkey):
    ensemkey2id[ensem_key] = i

all_ratemaps = np.zeros((num_ensem, BD.xedm.shape[0], BD.yedm.shape[0], BD.aedm.shape[0]))

for i in tqdm(range(num_ensem)):

    ensem_key = id2ensemkey[i]
    xsp = xdict[ensem_key]
    ysp = ydict[ensem_key]
    asp = adict[ensem_key]
    fpos = fposdict[ensem_key]

    Hsp3d, _ = BD.compute_histogram3d(xsp, ysp, asp)
    ratemap_3d = BD.compute_ratemap(occ, Hsp3d)
    maptmp = gaussian_filter(ratemap_3d, sigma=BD.bodysd_ind, mode='constant', cval=0, axes=(0, 1))
    ratemap_3d_gau = circular_gau_filter(maptmp, a_ax=BD.aedm, kappa=4*np.pi)
    all_ratemaps[i, :, :, :] = ratemap_3d_gau

    if debug_plot_tag:
        ratemap_pos = ratemap_3d_gau.mean(axis=2)
        ratemap_a = ratemap_3d_gau.mean(axis=0).mean(axis=0)

        fig = plt.figure(figsize=(8, 4), facecolor='w')
        ax1 = fig.add_axes([0.05, 0.2, 0.4, 0.7])
        ax2 = fig.add_axes([0.55, 0.2, 0.4, 0.6], polar=True)
        cbar_ax = fig.add_axes([0.46, 0.2, 0.03, 0.7])


        im1 = ax1.pcolormesh(BD.xedges, BD.yedges, ratemap_pos.T, cmap='jet')
        ax1.scatter(fpos[0], fpos[1], color='g', s=100)
        r = 2
        ax1.quiver(xsp, ysp, r* np.cos(asp), r*np.sin(asp), color='r', alpha=0.5, scale=75)
        ax1.set_xlim(xmin, xmax)
        ax1.set_ylim(ymin, ymax)
        ax1.set_title(ensem_key)
        cb = fig.colorbar(im1, cax=cbar_ax)

        ax2.bar(BD.aedm, ratemap_3d.mean(axis=0).mean(axis=0), width=da, alpha=0.5)
        ax2.plot(BD.aedm, ratemap_a, c='k')
        # ax2.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        # ax2.set_xticks(np.arange(-np.pi, np.pi + np.pi/4, np.pi/4), minor=True)
        # ax2.set_xticklabels(['-180', '-90', '0', '90', '180'])

        fig.savefig(join(plot_dir_ratemap, '%d.png'%(i)), dpi=200)
        plt.close(fig)
        # raise ValueError
# Bayesian inference =============================

xML, yML, aML, trajidML = [], [], [], []
for i in tqdm(range(trajdf.shape[0])):
    Xmat = trajdf['X'][i]

    if (Xmat.shape[0] < 1):
        continue

    if (i % 10 == 0):

        act_vec = np.zeros(num_ensem)
        fnode_ids, sigma_ids = Sequences.X2sigma(Xmat, seqR, sigma_state=False)
        for fnode_id, sigma_id in zip(fnode_ids, sigma_ids):
            nodekey = id2fkey_dict[fnode_id]
            ensem_key = '%s-%d'%(nodekey, sigma_id)
            ratemap_id = ensemkey2id[ensem_key]
            act_vec[ratemap_id] = 1

        logL = np.zeros((BD.xedm.shape[0], BD.yedm.shape[0], BD.aedm.shape[0]))
        for j in range(num_ensem):
            ratemap = all_ratemaps[j, :, :, :]
            k = act_vec[j]
            ratemap_epsilon = ratemap + 1e-7
            out = k * np.log(ratemap_epsilon) - ratemap_epsilon - factorial(k)
            logL += out

        maxid1D_logL = np.argmax(logL)
        maxid3D_logL = np.unravel_index(maxid1D_logL, logL.shape)
        xML.append(BD.xedm[maxid3D_logL[0]])
        yML.append(BD.yedm[maxid3D_logL[1]])
        aML.append(BD.aedm[maxid3D_logL[2]])
        trajidML.append(i)

BDresults = pd.DataFrame(dict(xML=xML, yML=yML, aML=aML,
             xGT=trajdf['x'][trajidML].to_list(), yGT=trajdf['y'][trajidML].to_list(), aGT=trajdf['a'][trajidML].to_list()))

BDresults.to_pickle(join(data_dir, 'inferences.pickle'))


