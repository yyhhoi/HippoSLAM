import numpy as np
from .circ import cdiff
from scipy.signal import convolve
from collections import OrderedDict
from .comput_utils import midedges


def createX(R, F, K, stored_f, f_sigma):
    X = np.zeros((F, K), dtype=int)

    for key, sigmalist in f_sigma.items():
        rowid = stored_f[key]
        for sigma in sigmalist:
            start_colid = sigma - 1  # sigma - 1
            X[rowid, start_colid:start_colid + R] = 1

    return X

class Sequences:
    def __init__(self, R: int, L: int, reobserve: bool = True):
        self.R = R
        self.L = L
        self.X_Ncol = R + L - 1
        self.X = np.zeros((0, self.X_Ncol))
        self.stored_f = dict()  # mapped to the row id of X. ObjectID (str/int) to RowID of X (int)
        self.f_sigma = dict()  # ObjectID (str/int) to sigma (int)
        self.far_fids = dict()
        self.current_f = []
        self.num_f = 0
        self.iter = 0
        self.reobserve = reobserve

    def step(self, f: list):
        self.clear_end_state()
        self.observe_f(f)
        self.propagate_sigma_update_X()
        self.current_f = f
        self.iter += 1

    def observe_f(self, f: list):
        for f_each in f:
            if (f_each in self.stored_f) and (f_each in self.current_f):
                # The stored feature node is still in the view.
                if self.reobserve:
                    self.f_sigma[f_each].append(0)  # Sigma stars with 0 because of the self.propagate_sigma() function.

            elif (f_each in self.stored_f) and (f_each not in self.current_f):
                # The feature node exited the view before and now re-enters
                # Start a new instance of the feature node
                self.f_sigma[f_each].append(0)

            elif f_each not in self.stored_f:
                # A new feature node is found
                self.stored_f[f_each] = self.num_f
                if f_each[-1] == 'f':
                    self.far_fids[f_each] = self.num_f
                self.f_sigma[f_each] = [0]
                self.num_f += 1
            else:
                ValueError('Unknown case. Feature nodes not recognised.')
    def reset_activity(self):
        for f_each in self.f_sigma.keys():
            self.f_sigma[f_each] = []
        self.current_f = []
        self.X = np.zeros((self.num_f, self.X_Ncol), dtype=int)
        self.iter = 0

    def propagate_sigma_update_X(self):
        self.X = np.zeros((self.num_f, self.X_Ncol), dtype=int)
        for f_each in self.stored_f.keys():
            num_instances = len(self.f_sigma[f_each])
            for f_instance in range(num_instances):
                self.f_sigma[f_each][f_instance] += 1
                rowid = self.stored_f[f_each]
                start_colid = self.f_sigma[f_each][f_instance] - 1  # sigma - 1
                self.X[rowid, start_colid:start_colid + self.R] = 1

    def clear_end_state(self):
        # Clear the instance when sigma reaches L
        for f_each in self.stored_f.keys():
            num_instances = len(self.f_sigma[f_each])
            non_terminal_sigmas = []
            for f_instance in range(num_instances):
                sigma = self.f_sigma[f_each][f_instance]
                if sigma < self.L:
                    non_terminal_sigmas.append(sigma)
            self.f_sigma[f_each] = non_terminal_sigmas

    @staticmethod
    def X2sigma(Xmat, R, sigma_state=False):
        kernel = np.ones((1, R)).astype(int)
        sigma_mat = convolve(Xmat.astype(int), kernel, mode='valid')

        actnum = R-1 if sigma_state else 1e-8

        fnode_rids, sigma_states = np.where(sigma_mat > actnum)

        # sigma should be in the range of [1, L], hence sigma_states + 1
        return fnode_rids, sigma_states + 1


class MatrixJ:

    def __init__(self, N, F, K):
        self.mat = np.zeros((N, F, K)).astype(float)


    def expand_F(self, num:int):
        if num > 0:
            mat_to_append = np.zeros((self.mat.shape[0], num, self.mat.shape[2]))
            self.mat = np.append(self.mat, mat_to_append, axis=1)
        return self.mat.shape[1]

    def expand_N(self, num:int):
        if num > 0:
            mat_to_append = np.zeros((num, self.mat.shape[1], self.mat.shape[2]))
            self.mat = np.append(self.mat, mat_to_append, axis=0)
        return self.mat.shape[0]

    def increment(self, X, target_n:int):

        assert (X.shape[0] == self.mat.shape[1])  # same F
        assert (X.shape[1] == self.mat.shape[2])  # same K
        self.mat[target_n, :, :] = self.mat[target_n, :, :] + X


    def normalize_slice(self, sid, area=False):
        N, F, K = self.mat.shape
        assert N > 0
        assert F > 0
        if area:
            norm = np.sum(self.mat[sid, :, :])
        else:
            norm = np.sqrt(np.sum(self.mat[sid, :, :] ** 2))
        if norm == 0:
            norm = 1
        self.mat[sid, :, :] = self.mat[sid, :, :] / norm

    def normalize_slice_cuetypes(self, sid, far_ids, area=False):
        N, F, K = self.mat.shape
        assert N > 0
        assert F > 0
        if far_ids is None:
            self.normalize_slice(sid, area=area)
        else:
            allFids = set(i for i in range(F))
            close_ids = list(allFids.difference(far_ids))
            far_norm = np.sum(self.mat[sid, far_ids, :]) * 2
            close_norm = np.sum(self.mat[sid, close_ids, :]) * 2
            if far_norm == 0:
                far_norm = 1
            if close_norm == 0:
                close_norm = 1
            self.mat[sid, far_ids, :] = self.mat[sid, far_ids, :] / far_norm
            self.mat[sid, close_ids, :] = self.mat[sid, close_ids, :] / close_norm


class StateDecoder:
    def __init__(self, R, L, maxN=500, lr=1, area_norm=False):
        self.R = R
        self.K = R + L - 1  # Number of columns of decoder matrix J
        self.N = 0  # Number of state nodes
        self.maxN = maxN  # When N reaches the maximum, no new node will be expanded.
        self.learn_mode = True  # Whether expanding new state is allowed.
        self.current_F = 0
        self.current_Sid = 0  # Current ID of the state node
        self.current_Sval = 0
        self.lowSThresh = 0.1
        self.J = MatrixJ(self.N, self.current_F,
                         self.K)  # MatrixJ mapping J.reshape(N, -1) @ X.flatten() = State vector
        self.lr = lr

        self.area_norm = area_norm


        # Embedding
        self.sid2embed = []

    def set_lowSthresh(self, s):
        self.lowSThresh = s
    @property
    def area_norm(self):
        return self._area_norm
    @area_norm.setter
    def area_norm(self, value: bool):
        self._area_norm = value


    def learn_embedding(self, X, e_new, emins, emaxs, far_ids):
        """

        Parameters
        ----------
        X : ndarray
            Shape = (F, K). From Sequence.X
        e_new : ndarray
            Shape = (Embed_dim, )
        emins : ndarray
            Shape = (Embed_dim, )
        emaxs : ndarray
            Shape = (Embed_dim, )
        far_ids : None or list

        Returns
        -------
        index of the embedding vector in the storage, i.e. self.sid2embed
        """


        assert self.N == len(self.sid2embed)

        # print('Far ids = \n', far_ids)

        # Initial condition
        if len(self.sid2embed) == 0:
            self.current_Sid = self.learn_supervised(X, sid=None, far_ids=far_ids)
            self.sid2embed.append(e_new.copy())
            return 0


        e_mat = np.stack(self.sid2embed)  # -> (Nstates, Embed_dim)
        e_new_norm = (e_new - emins) / (emaxs - emins)
        e_mat_norm = (e_mat - emins) / (emaxs - emins)
        sim_measure = 1 - np.sqrt(np.sum(np.square(e_new_norm.reshape(1, -1) - e_mat_norm), axis=1)) / np.sqrt(2)

        maxid = np.argmax(sim_measure)
        maxsimval = sim_measure[maxid]

        print('maxsimval = ', maxsimval)

        if (maxsimval < self.lowSThresh) and (not self.reach_maximum()):  # Not matching any existing embeddings
            # Create a new state, and remember the embedding
            _ = self.learn_supervised(X, sid=None, far_ids=far_ids)
            self.sid2embed.append(e_new.copy())
            print(f'Learn new state = {self.N-1}')

        else:
            if maxid != self.current_Sid:
                print(f'Learn old state = {self.current_Sid} to {maxid}')
            else:
                print(f'Correct state prediction {self.current_Sid} to {maxid}')
            _ = self.learn_supervised(X, sid=maxid, far_ids=far_ids)


        return maxid

    def learn_supervised(self, X, sid=None, far_ids=None):
        """

        Parameters
        ----------
        X : ndarray
            2-d array with shape (F, K). F = Number of feature nodes. K = R + L - 1.

        Returns
        -------

        """

        # Create a new state node
        if sid is None:
            self.N = self.J.expand_N(1)
            sid = self.N - 1
        # Increment to the specified sid (supervised)
        self.J.increment(X * self.lr, sid)
        self.J.normalize_slice_cuetypes(sid, far_ids, self._area_norm)
        return sid

    def learn_unsupervised(self, X):
        """

        Parameters
        ----------
        X : ndarray
            2-d array with shape (F, K). F = Number of feature nodes. K = R + L - 1.

        Returns
        -------

        """

        # Conditions trigger creating a new experience node
        N_tag = self.N < 1
        S_tag = self.current_Sval <= self.lowSThresh

        Xarea = np.sum(X)

        if Xarea > 1e-6:
            if N_tag or S_tag:
                # Create a new state node
                self.N = self.J.expand_N(1)
                # Associate X with the  N-th node
                X_norm = X / Xarea
                self.J.increment(X_norm * self.lr, self.N - 1)

    def update_F(self, X):
        dF = X.shape[0] - self.J.mat.shape[1]
        self.current_F = self.J.expand_F(dF)

    def infer_state(self, X):
        # Extend self.J.mat to the shape of X
        self.update_F(X)

        if self.N == 0:
            self.current_Sid = 0
            Snodes = np.zeros(1)

        elif X.sum() < 1e-6:
            # print('Inference: X is zero')
            # if No feature nodes were observed, use the previous state as inferred result
            Snodes = np.zeros(self.N)
            # self.current_Sid = 0    # Comment this line to use the previous state as inference result
            self.current_Sval = 0

        else:
            Snodes = self.infer_func(self.J.mat, X)
            if np.sum(Snodes) < 1e-6:
                # print('Inference: all Snodes are zeros')
                self.current_Sval = 0
            else:
                self.current_Sid = np.argmax(Snodes)
                self.current_Sval = np.max(Snodes)

        return self.current_Sid, Snodes
    def reset(self):
        self.current_Sid = 0  # Current ID of the state node
        self.current_Sval = 0

    def reach_maximum(self):
        return self.N >= self.maxN


    def infer_func(self, Jmat, X):
        Snodes = Jmat.reshape(self.N, self.current_F * self.K) @ X.flatten()
        return Snodes





class StateTeacher:
    def __init__(self, xbound, ybound, dp, da):
        self.xmin, self.xmax = xbound
        self.ymin, self.ymax = ybound
        self.amin, self.amax = -np.pi, np.pi
        self.dp = dp
        self.da = da
        self.xedges = np.arange(self.xmin, self.xmax + self.dp, self.dp)
        self.yedges = np.arange(self.ymin, self.ymax + self.dp, self.dp)
        self.aedges = np.arange(self.amin, self.amax + self.da, self.da)

        self.xax = midedges(self.xedges)
        self.yax = midedges(self.yedges)
        self.aax = midedges(self.aedges)
        self.xx, self.yy, self.aa = np.meshgrid(self.xax, self.yax, self.aax, indexing='ij')
        self.xx1d, self.yy1d, self.aa1d = self.xx.flatten(), self.yy.flatten(), self.aa.flatten()
        self.Nstates = self.xx1d.shape[0]
        self.pred2gt_map = dict()
        self.gt2pred_map = dict()

    def store_sid(self, sid):
        self.past_sids.append(sid)
        return None

    def store_prediction_mapping(self, sid_pred, sid_gt):
        self.pred2gt_map[sid_pred] = sid_gt

    def store_groundtruth_mapping(self, sid_gt, sid_pred):
        self.gt2pred_map[sid_gt] = sid_pred


    def map_pred_to_gt(self, sid_pred):
        return self.pred2gt_map[sid_pred]
    def map_gt_to_pred(self, sid_gt):
        return self.gt2pred_map[sid_gt]
    def match_prediction_storage(self, sid_pred):
        if sid_pred in self.pred2gt_map:
            return True
        else:
            return False

    def match_groundtruth_storage(self, sid_gt):
        if sid_gt in self.gt2pred_map:
            return True
        else:
            return False


    def lookup_xya(self, vals):
        xval, yval, aval = vals
        xind = self._lookup_xy(xval, self.xax)
        yind = self._lookup_xy(yval, self.yax)
        aind = self._lookup_a(aval, self.aax)
        inds = (xind, yind, aind)
        sid = np.ravel_multi_index(inds, dims=self.xx.shape)
        return sid

    def sid2xya(self, sid):
        return (self.xx1d[sid], self.yy1d[sid], self.aa1d[sid])

    def _lookup_xy(self, val, ax):
        return np.argmin(np.square(val - ax))

    def _lookup_a(self, val, ax):
        return np.argmin(np.abs(cdiff(val, ax)))