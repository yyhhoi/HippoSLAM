import numpy as np


def cmean(x, w=None):
    """
    Compute the circular mean and mean resultant vector length of the circular data x.

    Parameters
    ----------
    x : ndarray
        1-d array with circular values in radians. Expecting [-pi, pi).
    w : ndarray or None
        1-d array with the same shape as x. Weighting of each value in computing the mean. Need to be normalized.

    Returns
    -------
    mu : float
        Mean angle in radian
    R : float
        Mean resultant vector length in range (0, 1].

    """
    N = x.shape[0]
    complex_vec = np.exp(1j * x)
    if w is None:
        meanvec = np.sum(complex_vec) / N
    else:
        w = w / np.sum(w)
        meanvec = np.sum(w * complex_vec)
    R = np.abs(meanvec)
    mu = np.angle(meanvec)
    return mu, R



def opposite_angle(x):
    assert x <= np.pi
    assert x >= -np.pi

    if x > 0:
        out = -(2*np.pi - (x + np.pi) )
    else:
        out = x + np.pi
    return out

def divide_ignore(a: np.ndarray, b: np.ndarray):
    e1, e2, e3, e4 = np.isinf(a).sum(), np.isnan(a).sum(), np.isinf(b).sum(), np.isnan(b).sum()
    if np.any(np.array([e1, e2, e3, e4]) > 0):
        raise ValueError('No invalid values (nan and inf) in the inputs a, b are allowed.')
    else:
        with np.errstate(divide='ignore', invalid='ignore'):
            c = a / b
        c[np.isinf(c)] = 0
        c[np.isnan(c)] = 0
    return c

def midedges(edges):
    return (edges[:-1] + edges[1:]) / 2

def MLM(p, d, n, t, minerr=0.01):
    """

    Parameters
    ----------
    p : ndarray
        1-d array with shape (I, ). Firing rate in each position bin i.
    d : ndarray
        1-d array with shape (J, ). Firing rate in each direction bin j.
    n : ndarray
        2-d array with shape (I, J). Number of spike counts in each position and direction bin
    t : ndarray
        2-d array with shape (I, J). Dwell time in each position and direction bin
    minerr : float
        Error tolerance of MLM iteration.
    Returns
    -------

    """
    I, J = p.shape[0], d.shape[0]

    p_out = p.copy()
    d_out = d.copy()
    err = 1000
    i = 0
    while err > minerr:
        if i % 100 == 0:
            print(err)
        p_est = divide_ignore(np.mean(n, axis=1), np.mean(d_out.reshape(1, J) * t, axis=1))
        d_est = divide_ignore(np.mean(n, axis=0), np.mean(p_out.reshape(I, 1) * t, axis=0))

        # error
        errd = np.mean(np.abs(d_est - d_out))
        errp = np.mean(np.abs(p_est - p_out))
        err = errd + errp

        # update
        p_out = p_est
        d_out = d_est

        i += 1

    return p_out, d_out

class DirectionerMLM:
    def __init__(self, pos, hd, dt, sp_binwidth, a_binwidth, minerr=0.001, verbose=False):
        """
        Parameters
        ----------
        pos : ndarray
            xy coordinates of trajectory. Shape = (time, 2).
        hd : ndarray
            Heading in range (-pi, pi). Shape = (time, )
        dt : scalar
            Time duration represented by each sample of occupancy.

        sp_binwidth : scalar
            Bin width of xy space. Recommended 0.05 of the range.
        a_binwidth : scalar
            Bin width of angular distribution. Should be 2pi/36
        minerr : scalar
            Error tolerance of MLM iteration. Default 0.001.
        """
        self.minerr = minerr
        self.verbose = verbose
        # Binning
        self.xbins = np.arange(pos[:, 0].min(), pos[:, 0].max() + sp_binwidth, step=sp_binwidth)
        self.ybins = np.arange(pos[:, 1].min(), pos[:, 1].max() + sp_binwidth, step=sp_binwidth)
        self.abins = np.arange(-np.pi, np.pi + a_binwidth, step=a_binwidth)
        self.aedm = (self.abins[:-1] + self.abins[1:])
        self.abind = self.abins[1] - self.abins[0]

        # Behavioral
        data3d = np.concatenate([pos, hd.reshape(-1, 1)], axis=1)
        bins3d, edges3d = np.histogramdd(data3d, bins=[self.xbins, self.ybins, self.abins])
        self.tocc = bins3d * dt

    def get_directionality(self, possp, hdsp):
        """

        Parameters
        ----------
        possp : ndarray
            xy coordinates at spike times. Shape = (spiketime, 2)
        hdsp : ndarray
            Heading at spike times. Shape = (spiketime, )

        Returns
        -------

        """
        datasp3d = np.concatenate([possp, hdsp.reshape(-1, 1)], axis=1)
        nspk, edgessp3d = np.histogramdd(datasp3d, bins=[self.xbins, self.ybins, self.abins])
        totspks = np.sum(nspk)

        directionality = np.ones(self.tocc.shape[2]) / self.tocc.shape[2] * np.sqrt(totspks)
        positionality = np.ones((self.tocc.shape[0], self.tocc.shape[1])) / self.tocc.shape[0] / self.tocc.shape[
            1] * np.sqrt(totspks)

        err = 2
        iter = 0
        with np.errstate(divide='ignore', invalid='ignore'):
            while err > self.minerr:

                # pi
                tmp = np.nansum(directionality.reshape(1, 1, -1) * self.tocc, axis=2)
                ptmp = np.nansum(nspk, axis=2) / tmp
                ptmp[np.isinf(ptmp)] = np.nan

                # dj
                tmp = np.nansum(
                    np.nansum(positionality.reshape(positionality.shape[0], positionality.shape[1], 1) * self.tocc,
                              axis=0), axis=0)
                dtmp = np.nansum(np.nansum(nspk, axis=0), axis=0) / tmp
                dtmp[np.isinf(dtmp)] = np.nan

                # nfac
                # nfac = np.nansum(np.nansum(dtmp.reshape(1, 1, -1) * self.tocc, axis=2) * ptmp)
                # dtmp_norm = dtmp * np.sqrt(totspks / nfac)
                # ptmp_norm = ptmp * np.sqrt(totspks / nfac)

                dtmp_norm = dtmp / np.nansum(dtmp)
                ptmp_norm = ptmp / np.nansum(ptmp)

                # error
                errd = np.nanmean(directionality - dtmp_norm) ** 2
                errp = np.nanmean(positionality - ptmp_norm) ** 2
                err = np.sqrt(errd + errp)
                if self.verbose:
                    print('\r Error = %0.5f' % err, end="", flush=True)
                # update
                positionality = ptmp_norm
                directionality = dtmp_norm
                iter += 1

        directionality = directionality / np.nansum(directionality)
        nonanmask = ~np.isnan(directionality)
        fieldangle = circmean(self.aedm[nonanmask], directionality[nonanmask], d=self.abind)
        fieldR = resultant_vector_length(self.aedm[nonanmask], directionality[nonanmask], d=self.abind)
        return fieldangle, fieldR, directionality