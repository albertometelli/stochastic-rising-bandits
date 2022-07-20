from math import log, ceil, floor

import numpy as np
from agent import Agent


class SW_KL_UCB(Agent):
    """SW-KL-UCB algorithm"""

    def __init__(self) -> None:
        super().__init__()
        self.c = 3
    
    def set_horizon(self, T: int):
        super().set_horizon(T)
        self.sigma = 0.001
        self.tau = ceil(self.sigma ** (-4/5))   # sliding window size


    def set_arms_number(self, K: int):
        super().set_arms_number(K)
        self.upper_bound = np.zeros(self.K)
        self.cumul_observation = np.zeros(self.K)
        self.inverse_cumul_observation = np.zeros(self.K)
        self.outcomes = np.zeros(self.T)


    def select_arm(self) -> int:
        # first pulls
        for k in range(self.K):
            if self.cumul_observation[k] + self.inverse_cumul_observation[k] == 0:
                return k

        # other pulls
        max_arm_ids = np.where(self.upper_bound == np.max(self.upper_bound))[0]
        return np.random.choice(max_arm_ids)


    def new_observation(self, arm: int, value: float) -> None:
        p = max(0, min(1, value))
        self.outcomes[self.time] = np.random.binomial(1,p)

        self.cumul_observation[arm] += self.outcomes[self.time]
        self.inverse_cumul_observation[arm] += 1-self.outcomes[self.time]

        if self.time > self.tau:
            self.cumul_observation[arm] -= self.outcomes[self.time - self.tau]
            self.inverse_cumul_observation[arm] -= 1-self.outcomes[self.time - self.tau]

        if self.time < self.K:
            super().new_observation(arm, value)
            return

        for k in range(self.K):
            curr_p = self.cumul_observation[k] / (self.cumul_observation[k] + self.inverse_cumul_observation[k])
            threshold = self.c * log(log(min(self.time,self.tau))) + log(min(self.time, self.tau))
            if curr_p == 1 or threshold < 0:
                self.upper_bound[k] = 1
            else:
                self.upper_bound[k] = self.__max_kldiv(curr_p, threshold / (self.cumul_observation[k] + self.inverse_cumul_observation[k]),1e-10)

        super().new_observation(arm, value)



    def __max_kldiv(self, p, limit, tol):
        p = max(p, tol)
        p = min(p, 1-tol)
        a = p
        b = 1-tol

        dive = 1

        div_a = p * log(p / a) + (1 - p) * log((1-p) / (1-a)) - limit
        assert(div_a < 0)

        div_b = p * log(p / b) + (1 - p) * log((1-p) / (1-b)) - limit

        if(div_a * div_b < 0):
            max_iter = 1

            while (abs(dive) > tol and max_iter < 10000):
                x = (a + b) / 2
                dive = p * log(p / x) + (1 - p) * log((1-p) / (1-x)) - limit

                if(dive * div_a > 0):
                    a = x
                else:
                    b = x

                max_iter = max_iter + 1
        else:
            x = b

        return x


    def copy(self):
        return SW_KL_UCB()


    def __str__(self) -> str:
        return "SW-KL-UCB"
