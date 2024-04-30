import matplotlib.pyplot as plt
import numpy as np


def set_arena(ax):
    xbound = np.array((-16, 8)) * 1.1
    ybound = np.array((-8, 16)) * 1.1
    goal_loc = (-14.87, -7.38)

    ax.set_xlim(*xbound)
    ax.set_ylim(*ybound)
    ax.scatter(*goal_loc, c='cyan', marker='D', zorder=3.1, s=100, alpha=0.5)
    return ax

def plot_spatial_specificity_base(ax0, ax1, xya, aedges, xbound, ybound, title='' ):
    ax0.quiver(xya[:, 0], xya[:, 1], np.cos(xya[:, 2]), np.sin(xya[:, 2]), scale=30)
    set_arena(ax0)
    ax0.set_title(title)
    abins ,_  = np.histogram(xya[:, 2], bins=aedges)
    ax1.bar(aedges[1:]/2 + aedges[:-1]/2, abins, width=aedges[1] -aedges[0], fill=False)
    ax1.scatter(0, 0, c='k')
    ax1.axis('off')


def plot_spatial_specificity(xya, aedges, xbound, ybound, title='', figsize=(8, 3)):
    fig = plt.figure(figsize=figsize)
    ax0 = fig.add_subplot(1, 2, 1)
    ax1 = fig.add_subplot(1, 2, 2, projection='polar')
    plot_spatial_specificity_base(ax0, ax1, xya, aedges, xbound, ybound, title=title)
    return fig, [ax0, ax1]



def compare_spatial_specificity(xya_targs, xya_pred, pred_offsets, aedges, xbound, ybound, title=''):
    fig = plt.figure(figsize=(12, 6))
    ax0 = fig.add_subplot(2, 4, 1)
    ax1 = fig.add_subplot(2, 4, 2, projection='polar')

    ax_pred = []
    for i in range(3, 3 + 5):
        ax_pred.append(fig.add_subplot(2, 4, i))

    # Plot Embedding poses
    ax0.quiver(xya_targs[:, 0], xya_targs[:, 1], np.cos(xya_targs[:, 2]), np.sin(xya_targs[:, 2]), scale=30)
    ax0.set_title(title)
    abins, _ = np.histogram(xya_targs[:, 2], bins=aedges)
    ax1.bar(aedges[1:] / 2 + aedges[:-1] / 2, abins, width=aedges[1] - aedges[0])
    ax1.scatter(0, 0, c='k')
    ax1.axis('off')

    # Plot predicted poses
    for i in range(5):
        mask = pred_offsets == i
        ax_pred[i].quiver(xya_pred[:, 0][mask], xya_pred[:, 1][mask], np.cos(xya_pred[:, 2][mask]),
                          np.sin(xya_pred[:, 2][mask]), scale=30)
        ax_pred[i].set_title('offset=%d, num=%d' % (i, mask.sum()))

    for axeach in [ax0] + ax_pred:
        axeach.set_xlim(*xbound)
        axeach.set_ylim(*ybound)

    return fig, [ax0, ax1] + ax_pred



