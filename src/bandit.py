from typing import Callable

import numpy as np
from tqdm import tqdm

from arm import Arm
from reward_model import RewardModel


class Bandit:
    """Bandit environment"""

    def __init__(self, restless=False, custom_arms :"list[Arm]"=[]) -> None:
        self.restless = restless

        self.arms = custom_arms.copy()
        self.arm_number = len(self.arms)
        self.noise = self.arms[0].reward_model.noise_type if self.arm_number > 0 else RewardModel.bernoulli_noise["type"]  # all arms have same noise

        self.oracle_result = None
        self.best_arm = None
        self.time = 0


    def pull_arm(self, i :int) -> "tuple(float, float)":
        """pull_arm : pulls arm i of the bandit and receive (true reward, observation)

        Args:
            i (int): arm of the bandit to be pulled
        """
        self.time += 1
        return self.arms[i].pull(self.time - 1)


    def oracle(self, T :int) -> dict:
        """oracle this method computes the oracle policy
        
        rested bandit = oracle constant policy (which always pulls the arm with highest cumulative reward at T)

        restless bandit = oracle greedy policy (whcih pulls the best available arm at each time step t)

        Args:
            T (int): time horizon

        Returns:
            dict: contains "rewards" (np.ndarray), "observations" (np.ndarray), "pulls" (np.ndarray) at each time step
        """

        if not self.oracle_result or T != self.oracle_result["T"]:
            self.T = T
            best_rewards = np.zeros(T)
            best_observations = np.zeros(T)
            best_cumul_rewards = 0
            best_id = -1
            oracle_pulls = np.zeros(len(self.arms), dtype=int)

            if self.restless:
                self.best_arm = None    # restless bandit: best arm can change
                for t in tqdm(range(T), desc="Oracle"):
                    pulls_at_t = [arm.pull(t) for arm in self.arms]
                    best_arm_at_t = np.argmax(list(map(lambda s : s[0], pulls_at_t)))
                    best_rewards[t] = pulls_at_t[best_arm_at_t][0]
                    best_observations[t] = pulls_at_t[best_arm_at_t][1]
                    oracle_pulls[best_arm_at_t] += 1
            else:
                for arm in tqdm(self.arms, desc="Oracle"):
                    pulls = [arm.pull(t) for t in range(T)]
                    rewards = np.array(list(map(lambda s : s[0], pulls)))
                    observations = np.array(list(map(lambda s : s[1], pulls)))

                    cumul_rewards = np.sum(rewards)
                    if cumul_rewards > best_cumul_rewards:
                        best_id = arm.arm_id
                        best_cumul_rewards = cumul_rewards
                        best_rewards = rewards.copy()
                        best_observations = observations.copy()
                oracle_pulls[best_id] = T
                self.best_arm = best_id # rested bandit best arm

            self.oracle_result = {
                "T" : T,
                "pulls" : oracle_pulls,
                "rewards" : best_rewards,
                "observations" : best_observations
            }

        self.reset()

        return self.oracle_result


    def reset(self) -> None:
        """resets the bandit (and all its arms)"""
        self.time = 0
        for arm in self.arms:
            arm.reset()


    def get_lambdas(self) -> "list[Callable]":
        """returns a list of all the reward functions of the bandit arms"""
        return list(map(lambda arm: arm.reward_model.reward_function, self.arms))


    def load_lambdas(self, lambdas :list, noise = RewardModel.beta_noise):
        """replace the arms of the bandit with new arms with given reward functions and noise"""
        if self.arm_number > 0:
            print("WARNING: overwriting existing arms")
            self.arms = []
            self.arm_number = 0
        
        for l in lambdas:
            r = RewardModel()
            r.load_model(l, noise, self.arm_number)
            self.arms.append(Arm(self.arm_number,r))
            self.arm_number += 1


    def create_random_arms(self, K :int, T :int, only_exp=False, only_poly=False, global_optimum=False):
        """randomly create arms for the bandit (reward functions, noise)

        Args:
            K (int): number of arms to be created
            T (int): time horizon (used to tune how the reward functions are created)
            only_exp (bool, optional): draw functions from the exponential family only. Defaults to False.
            only_poly (bool, optional): draw functions from the polynomial family only. Defaults to False.
            global_optimum (bool, optional): all the functions reach the global optimum 1. Defaults to False.
        """
        for _ in range(K):
            m = RewardModel()
            m.random_model(T, only_exp=only_exp, only_poly=only_poly, rewards_reach_1=global_optimum)
            self.arms.append(Arm(self.arm_number, m, self.restless))
            self.arm_number += 1

        self.noise = self.arms[0].reward_model.noise_type if self.arm_number > 0 else RewardModel.bernoulli_noise["type"]  # all arms have same noise


    def copy(self):
        """create a copy of the bandit (without history of pulls)"""
        arms_copy = [arm.copy() for arm in self.arms]
        bandit_copy = Bandit(self.restless, arms_copy)
        bandit_copy.best_arm = self.best_arm
        bandit_copy.oracle_result = self.oracle_result.copy() if self.oracle_result != None else None
        return bandit_copy


    def __str__(self) -> str:
        s = "Bandit:\n{}\nnoise {}\n\n".format("restless" if self.restless else "rested", self.noise) + \
            f"K = {len(self.arms)}\n"

        for arm in self.arms:
            s += f"{arm}\n"
        
        s += "-" * 40 + f"\nT = {self.T}\n\nbest arm = arm {self.best_arm}"
        
        p = self.oracle_result["pulls"]
        s += f"\noracle pulls = {p}\n" + "-" * 40

        return s