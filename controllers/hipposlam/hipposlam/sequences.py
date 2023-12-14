import numpy as np


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

