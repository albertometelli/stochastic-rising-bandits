from math import ceil, e, gamma, log, sqrt
from random import choices

import numpy as np
from agent import Agent


class Rexp3(Agent):
    """Rexp3 algorithm"""

    def __init__(self) -> None:
        super().__init__()
        self.reset()
        self.sigma = 0.001


    def set_horizon(self, T: int):
        super().set_horizon(T)
        self.T = T


    def set_arms_number(self, K: int):
        super().set_arms_number(K)
        self.delta = ceil( (self.K * log(self.K))**(1/3) * (self.T / (self.sigma*self.T)) ** (2/3) )
        self.gamma = min(1, sqrt( self.K*log(self.K) / ( (e-1)*self.delta) ))
        self.previous_arm = -1
        self.previous_outcome = -1

        self.p = np.zeros(self.K)
        self.w = np.zeros(shape=(self.K,2))


    def select_arm(self) -> int:
        x = {}
        tau = (self.phase-1)*self.delta
        
        if self.phase == 0 or self.time > tau + self.delta:
            self.phase += 1
            for k in range(self.K):
                self.w[k,1] = 1
        else:
            for k in range(self.K):
                x[k] = self.previous_outcome / self.p[k] if k == self.previous_arm else 0
                self.w[k,1] = self.w[k,0] * (e ** (self.gamma * x[k] / self.K))

        # compute bound
        for k in range(self.K):
            others_w = 0
            for a in range(self.K):
                if a != k:
                    others_w += self.w[a,1]
            self.p[k] = (1-self.gamma) * self.w[k,1] / others_w + self.gamma / self.K

        # arm selection
        best_arm = np.random.choice(self.K, p=(self.p / np.sum(self.p)))

        self.w[:,0] = self.w[:,1]   # previous = current

        return best_arm


    def new_observation(self, arm: int, value: float) -> None:
        p = max(0,min(1,value))
        outcome = np.random.binomial(1,p)
        self.previous_arm = arm
        self.previous_outcome = outcome

        super().new_observation(arm, value)


    def reset(self):
        super().reset()
        self.phase = 0


    def copy(self):
        return Rexp3()


    def __str__(self) -> str:
        return "Rexp3"
