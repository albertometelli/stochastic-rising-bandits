from math import floor, log, sqrt

import numpy as np
from agent import Agent


class R_less_UCB(Agent):
    """R-less-UCB algorithm"""

    def __init__(self):
        super().__init__()
        self.sigma = 0.1

        # parameters: delta, h
        self.delta_function = lambda t : t ** -4 if t > 0 else 0.01
        self.window_function = lambda n : floor(n/4)

    def set_arms_number(self, K :int):
        super().set_arms_number(K)
        # efficient update parameters
        self.h = np.zeros(self.K)
        self.a = np.zeros(self.K)
        self.b = np.zeros(self.K)
        self.c = np.zeros(self.K)
        self.d = np.zeros(self.K)

        self.mu_hat = np.ones(self.K)
        self.gamma_hat = np.zeros(self.K)
        self.upper_bound = np.zeros(self.K)


    def select_arm(self) -> int:
        # first pulls
        if self.time < 2*self.K:
            return self.time % self.K

        ub = self.upper_bound
        max_arm_ids = np.where(ub == np.max(ub))[0]
        return np.random.choice(max_arm_ids)


    def new_observation(self, arm: int, value: float) -> None:
        super().new_observation(arm, value)
        self.__update_ubs(arm)


    def __update_ubs(self, arm_pulled):
        for k in range(self.K):
            n = int(self.number_of_pulls[k])
            h = self.window_function(n)     # window size h = f(n)
            delta = self.delta_function(self.time)
            t =  self.time + 1

            ## update with efficient update rules
            if k == arm_pulled:         #I have to update a,b,c,d only if the arm was pulled
                if h == self.h[k]:      #h_n == h_{n-1}
                    self.a[k] += self.observations[k][n-1] - self.observations[k][n-1-h]
                    self.b[k] += self.observations[k][n-1-h] - self.observations[k][n-1-2*h]
                    self.c[k] += (n-1)*self.observations[k][n-1] - (n-1-h)*self.observations[k][n-1-h]
                    self.d[k] += (n-1-h)*self.observations[k][n-1-h] - (n-1-2*h)*self.observations[k][n-1-2*h]
                else:
                    self.a[k] += self.observations[k][n-1]
                    self.b[k] += self.observations[k][n-2*h]
                    self.c[k] += (n-1)*self.observations[k][n-1]
                    self.d[k] += (n-2*h)*self.observations[k][n-2*h]

                self.mu_hat[k] = ( (2*h-n+1)*(self.a[k]-self.b[k]) + (self.c[k]-self.d[k]) ) / (h**2) if h > 0 else 0
                self.gamma_hat[k] = (t-n)*(self.a[k]-self.b[k]) / (h**2) if h > 0 else 0
                self.h[k] = h

            #I have to update the confidence factors even if the arm was not pulled
            if h > 0:
                #beta_mu = self.sigma * sqrt( 6*log(2/delta) / h )
                #beta_gamma = 2*self.sigma*(t-n) * sqrt( log(2/delta) / (h**3) )
                
                beta_new = self.sigma*(t-n+h-1) * sqrt( 10*log(1/delta) / (h**3) )
                beta = beta_new
                self.upper_bound[k] = self.mu_hat[k] + self.gamma_hat[k] + beta
            else:
                self.upper_bound[k] = 1


    def copy(self):
        return R_less_UCB()


    def __str__(self) -> str:
        return "R-less-UCB"
