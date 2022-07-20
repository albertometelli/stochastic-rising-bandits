from math import log, sqrt

import numpy as np
from agent import Agent


class Ser4(Agent):
    """Ser4 algorithm"""
    
    def __init__(self) -> None:
        super().__init__()


    def set_arms_number(self, K: int):
        super().set_arms_number(K)

        self.n_active_arms = self.K
        self.current_arm = 0

        self.previous_mu = np.zeros(self.K)
        self.last_outcome = np.zeros(self.K)
        self.active_arms_idxs = np.ones(self.K, dtype=bool)

        self.delta = 1 / self.T
        self.epsilon = 1 / (self.T * self.K)
        self.phi = sqrt(self.T / (self.T * self.K * sqrt(self.K*self.T)))
        self.tau = 1


    def set_horizon(self, T: int):
        super().set_horizon(T)


    def select_arm(self) -> int:
        u = {}
        stop_round_robin = False

        # round robin
        if self.n_active_arms == 1:             # there is only one arm active
            for k in range(self.K):
                if self.active_arms_idxs[k]:
                    best_arm = k

        elif self.current_arm < self.K:           # select each arm once (if active)
            while self.current_arm < self.K and not self.active_arms_idxs[self.current_arm]:
                self.current_arm += 1
            if self.current_arm < self.K:
                best_arm = self.current_arm
                self.current_arm += 1
            else:
                stop_round_robin = True

        else:
            stop_round_robin = True

        # after round robin
        if stop_round_robin:
            # select each arm once and compute the means of the active arms
            for k in range(self.K):
                if not self.active_arms_idxs[k]:
                    u[k] = -1
                else:
                    u[k] = (self.tau - 1) * self.previous_mu[k] / self.tau + self.last_outcome[k] / self.tau

            best_arm = 0
            for k in range(self.K):
                if u[k] > u[best_arm]:
                    best_arm = k


            # arm elimination
            threshold = 2*sqrt( log(2*self.n_active_arms*self.tau / self.delta) )
            to_be_removed = 0
            for k in range(self.K):
                if self.active_arms_idxs[k] and u[best_arm] - u[k] + self.epsilon >= threshold:
                    self.active_arms_idxs[k] == False
                    to_be_removed += 1

            # move to next phase
            for k in range(self.K):
                self.previous_mu[k] = u[k]
            self.tau += 1
            self.n_active_arms = max(self.n_active_arms - to_be_removed, 0)
            self.current_arm = 0    # restart

            # with prob. phi restart the algo
            if np.random.random() < self.phi:
                self.tau = 1
                self.n_active_arms = self.K
                self.current_arm = 0
                self.active_arms_idxs = np.ones(self.K, dtype=bool)
                self.previous_mu = np.zeros(self.K)
                self.last_outcome = np.zeros(self.K)

        return best_arm


    def new_observation(self, arm: int, value: float) -> None:
        p = max(0, min(1, value))
        outcome = np.random.binomial(1,p)
        self.last_outcome[arm] = outcome

        super().new_observation(arm, value)
    

    def copy(self):
        return Ser4()


    def __str__(self) -> str:
        return "Ser4"
