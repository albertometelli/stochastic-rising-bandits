from math import log

import numpy as np
from agent import Agent


class KL_UCB(Agent):
    """KL-UCB algorithm"""

    def __init__(self) -> None:
        super().__init__()
        self.c = 3

    
    def set_horizon(self, T: int):
        super().set_horizon(T)


    def set_arms_number(self, K: int):
        super().set_arms_number(K)
        self.upper_bound = np.zeros(self.K)
        self.cumul_observation = np.zeros(self.K)

    
    def select_arm(self) -> int:
        if self.time < self.K:  # first pulls
            return self.time

        max_arm_ids = np.where(self.upper_bound == np.max(self.upper_bound))[0]
        return np.random.choice(max_arm_ids)


    def new_observation(self, arm: int, value: float) -> None:
        self.cumul_observation[arm] += value
        if self.time >= self.K:
            self.__update_ubs()
        super().new_observation(arm, value)


    def __update_ubs(self):
        limit = self.c * log(log(self.T)) + log(self.T)
        for k in range(self.K):
            curr_p = self.cumul_observation[k] / self.time
            self.upper_bound[k] = self.__max_kldiv(curr_p, limit / self.time, 1e-10)


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
        return KL_UCB()

    def __str__(self) -> str:
        return "KL-UCB"
