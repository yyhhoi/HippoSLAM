import numpy as np
from scipy.signal import convolve


def DiceCoef(Jmat, X):
    """
    Parameters
    ----------
    Jmat : ndarray
        Shape (N, F, K) with values [0, 1]
    X : ndarray
        Shape (F, K) with values [0, 1].

    Returns
    -------
    Dice_Coefficient: ndarray
        Shape (N, )

    """
    N, F, K = Jmat.shape

    DC = (Jmat.reshape(N, -1) * X.reshape(1, -1)).sum(axis=1) / np.clip(Jmat.reshape(N, -1) + X.reshape(1, -1), a_min=0,
                                                                        a_max=1).sum(axis=1)
    return DC


class Sequences:
    def __init__(self, R: int, L: int, reobserve: bool = True):
        self.R = R
        self.L = L
        self.X_Ncol = R + L - 1
        self.X = np.zeros((0, self.X_Ncol))
        self.stored_f = dict()  # mapped to the row id of X. ObjectID (str/int) to RowID of X (int)
        self.f_sigma = dict()  # ObjectID (str/int) to sigma (int)
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
            stored_f_keys = self.stored_f.keys()
            if (f_each in stored_f_keys) and (f_each in self.current_f):
                # The stored feature node is still in the view.
                if self.reobserve:
                    self.f_sigma[f_each].append(0)  # Sigma stars with 0 because of the self.propagate_sigma() function.

            elif (f_each in stored_f_keys) and (f_each not in self.current_f):
                # The feature node exited the view before and now re-enters
                # Start a new instance of the feature node
                self.f_sigma[f_each].append(0)

            elif f_each not in self.stored_f.keys():
                # A new feature node is found
                self.stored_f[f_each] = self.num_f
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
        kernel = np.ones((1, R))
        sigma_mat = convolve(Xmat, kernel, mode='valid')

        actnum = R-1 if sigma_state else 0

        fnode_rids, sigma_states = np.where(sigma_mat > actnum)

        # sigma should be in the range of [1, L], hence sigma_states + 1
        return fnode_rids, sigma_states + 1


class MatrixJ:

    def __init__(self, N, F, K):
        self.mat = np.zeros((N, F, K))


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

    def normalize(self, area=False):
        N, F, K = self.mat.shape
        assert N > 0
        assert F > 0
        if area:
            norm = np.sum(np.sum(self.mat, axis=1), axis=1)
        else:
            norm = np.sqrt(np.sum(np.sum(self.mat ** 2, axis=1), axis=1))
        self.mat = self.mat / norm.reshape(norm.shape[0], 1, 1)


class StateDecoder:
    def __init__(self, R, L, maxN=500, infer_mode='Dice'):
        self.R = R
        self.K = R + L - 1  # Number of columns of decoder matrix J
        self.N = 0  # Number of state nodes
        self.maxN = maxN  # When N reaches the maximum, no new node will be expanded.
        self.learn_mode = True  # Whether expanding new state is allowed.
        self.infer_mode = infer_mode  # "Dice" or "Sum"
        self.current_F = 0
        self.current_Sid = 0  # Current ID of the state node
        self.current_Sval = 0
        self.lowSThresh = 0.1
        self.J = MatrixJ(self.N, self.current_F,
                         self.K)  # MatrixJ mapping J.reshape(N, -1) @ X.flatten() = State vector


    def learn(self, X):
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
        if N_tag:
            X_tag = True
        else:
            # filtermask = self.J.mat[self.current_Sid, :, :] < 1e-6
            # X_tag = np.any(X[filtermask]) > 0
            PreviousQuietMask = self.J.mat[self.current_Sid, :, :].sum(axis=1) < 1e-6
            PreviousQuietIDs = np.where(PreviousQuietMask)[0]
            X_tag = np.any(X[PreviousQuietIDs, :] > 0)

        if (N_tag & (Xarea > 1e-6)) or (S_tag and X_tag):
            # Create a new state node
            self.N = self.J.expand_N(1)
            # Associate X with the  N-th node
            X_norm = X / Xarea
            self.J.increment(X_norm, self.N - 1)


    def learn2(self, X):
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
                if self.infer_mode == "Sum":
                    # Associate X with the  N-th node
                    X_norm = X / Xarea
                    self.J.increment(X_norm, self.N - 1)
                elif self.infer_mode == "Dice":
                    self.J.increment(X, self.N - 1)


    def infer_state(self, X):
        # Extend self.J.mat to the shape of X
        newF = X.shape[0]
        F_tag = newF > self.current_F
        if F_tag:
            dF = newF - self.J.mat.shape[1]
            self.current_F = self.J.expand_F(dF)

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
        if self.infer_mode == "Sum":
            Snodes = Jmat.reshape(self.N, self.current_F * self.K) @ X.flatten()
        elif self.infer_mode == "Dice":
            Snodes = DiceCoef(Jmat, X)
        else:
            raise ValueError("StateDecoder.infer_mode must be either 'Sum' or 'Dice'.")
        return Snodes



