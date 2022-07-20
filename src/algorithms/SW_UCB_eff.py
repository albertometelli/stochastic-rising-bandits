from math import log, sqrt, floor

import numpy as np
from agent import Agent


class SW_UCB_eff(Agent):
    """SW-UCB algorithm (efficient version, lower time complexity)"""

    def __init__(self) -> None:
        super().__init__()
        self.B = 1  # maximum reward value

    
    def set_horizon(self, T: int):
        super().set_horizon(T)
        self.outcomes = np.zeros(self.T)
        self.tau = floor( 4 * sqrt(T * log(T)) )
        self.gamma = 1 - 1 / sqrt(self.T) / 4
        self.xi = 0.6


    def set_arms_number(self, K: int):
        super().set_arms_number(K)
        self.cumul_r_in_window = np.zeros(self.K)
        self.inverse_cumul_r_in_window = np.zeros(self.K)
        self.upper_bound = np.zeros(self.K)
        self.N_t = np.zeros(self.K)
        self.X_line = np.zeros(self.K)

    
    def select_arm(self) -> int:
        if self.time < self.K:  # first pulls
            return self.time

        max_arm_ids = np.where(self.upper_bound == np.max(self.upper_bound))[0]
        return np.random.choice(max_arm_ids)


    def new_observation(self, arm: int, value: float) -> None:
        p = max(0, min(1, value))
        self.outcomes[self.time] = np.random.binomial(1,p)

        self.cumul_r_in_window[arm] += self.outcomes[self.time]
        self.inverse_cumul_r_in_window[arm] += 1 - self.outcomes[self.time]

        if self.time > self.tau:
            self.cumul_r_in_window[arm] -= self.outcomes[self.time - self.tau]
            self.inverse_cumul_r_in_window[arm] -= 1 - self.outcomes[self.time - self.tau]

        if self.time >= self.K:
            self.__update_ubs()

        super().new_observation(arm, value)


    def __update_ubs(self):
        for k in range(self.K):
            val = self.cumul_r_in_window[k] / (self.cumul_r_in_window[k] + self.inverse_cumul_r_in_window[k])
            confidence = sqrt( self.xi * log(min(self.time, self.tau)) / (self.cumul_r_in_window[k] + self.inverse_cumul_r_in_window[k]) )
            self.upper_bound[k] = val + confidence
            

    def copy(self):
        return SW_UCB_eff()


    def __str__(self) -> str:
        return "SW-UCB"
