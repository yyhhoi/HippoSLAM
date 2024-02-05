import numpy as np
from scipy.signal import convolve


class Sequences:
    def __init__(self, R: int, L: int, reobserve: bool = True):
        self.R = R
        self.L = L
        self.X_Ncol = R + L - 1
        self.X = np.zeros((0, self.X_Ncol))
        self.stored_f = dict()  # mapped to the row id of X
        self.f_sigma = dict()
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
    def reset(self):
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

    def normalize(self):
        N, F, K = self.mat.shape
        assert N > 0
        assert F > 0
        areas = np.sqrt(np.sum(np.sum(self.mat ** 2, axis=1), axis=1))
        # areas = np.sum(np.sum(self.mat, axis=1), axis=1)
        self.mat = self.mat / areas.reshape(areas.shape[0], 1, 1)


class HippoLearner:
    def __init__(self, R, L, NL):
        self.K = R + L - 1  # Number of columns of J

        self.current_F = 0
        self.previous_F = 0  # Number of feature nodes when the learning started
        self.N = 0  # Number of state nodes
        self.J = MatrixJ(self.N, self.current_F, self.K)  # MatrixJ mapping J.reshape(N, -1) @ X.flatten() = State vector
        self.learn_mode = False
        self.NL = NL  #  Number of learning steps.
        self.learn_l = 0  # the l-th learning step
        self.learn_S = 0
        self.current_S = 0
        self.current_Sval = 1

    def step(self, X):
        """

        Parameters
        ----------
        X : ndarray
            2-d array with shape (F, K). F = Number of feature nodes. K = R + L - 1.

        Returns
        -------

        """
        newF = X.shape[0]
        # Conditions trigger learning
        # if (newF > self.current_F) or ((self.current_Sval < 0.9) and X.sum() > 1):
        if (newF > self.current_F):
            if  (self.learn_l == 0) and (self.learn_mode==False) :
                self.learn_mode = True
                # Create a new state node
                self.N = self.J.expand_N(1)

        # Once learning is triggered
        if self.learn_mode:
            # Extend self.J.mat to the shape of X
            dF = newF - self.J.mat.shape[1]
            self.current_F = self.J.expand_F(dF)

            # Associate X and J to N-th node
            self.J.increment(X, self.N-1)

            self.learn_l += 1
            self.learn_S = self.N-1

            if self.learn_l == (self.NL-1):
                self.J.normalize()
                self.learn_l = 0
                self.learn_mode = False


    def infer_state(self, X):
        if self.learn_mode:
            self.current_S = self.learn_S
            Snodes = np.zeros(self.N)
            Snodes[self.current_S] = 1
            self.current_Sval = 1

        else:

            if self.N == 0:
                Snodes = np.ones(1)

            elif X.sum() < 1e-6:
                # if No feature nodes were observed, use the previous state as inferred result
                Snodes = np.zeros(self.N)
                Snodes[self.current_S] = 1
                self.current_Sval = 1

            else:
                Snodes = self.J.mat.reshape(self.N, self.current_F * self.K) @ X.flatten()
                self.current_S = np.argmax(Snodes)
                self.current_Sval = np.max(Snodes)

        return self.current_S, Snodes




