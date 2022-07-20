from math import ceil

import numpy as np
from agent import Agent


class SW_TS_eff(Agent):
    """SW-TS algorithm (efficient version, lower time complexity)"""
    
        
    def __init__(self) -> None:
        super().__init__()

        self.beta = 0.5


    def set_horizon(self, T: int):
        super().set_horizon(T)
        self.tau = ceil(self.T ** (1 - self.beta))


    def set_arms_number(self, K :int):
        super().set_arms_number(K)
        self.distributions = np.array([lambda : np.random.uniform()] * K)    # uniform priors
        self.theta = np.zeros(K)                                      # samples
        self.bernoulli_rewards = np.array([-1] * self.T)                     # TS needs rewards to be in {0,1}
        self.pulls_in_window = np.zeros(self.K)
        self.cumul_reward_in_window = np.zeros(self.K)


    def select_arm(self) -> int:
        for k in range(self.K):
            self.distributions[k] = lambda : np.random.beta(self.pulls_in_window[k] + 1, self.pulls_in_window[k] - self.cumul_reward_in_window[k] + 1)
            self.theta[k] = self.distributions[k]()
        
        return np.argmax(self.theta)


    def new_observation(self, arm: int, value: float) -> None:
        p = max(0, min(1, value))
        self.bernoulli_rewards[self.time] = np.random.binomial(1, p)

        self.pulls_in_window[arm] += 1
        self.cumul_reward_in_window[arm] += self.bernoulli_rewards[self.time]
        
        for k in range(self.K):
            if self.pulled_arm_ids[self.time - self.tau] == k:
                self.pulls_in_window[k] -= 1
                self.cumul_reward_in_window[k] -= self.bernoulli_rewards[self.time - self.tau]

        super().new_observation(arm, value)


    def copy(self):
        return SW_TS_eff()


    def __str__(self) -> str:
        return "SW-TS"
