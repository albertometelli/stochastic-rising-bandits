from itertools import accumulate

import numpy as np
from tqdm import tqdm

import options
from agent import Agent
from bandit import Bandit


class Experiment:
    """container of a single iteration of a specific algorithm on a specific bandit"""

    def __init__(self, bandit :Bandit, agent :Agent, T :int) -> None:
        self.bandit = bandit
        self.agent = agent
        self.name = str(agent)
        self.T = T

        oracle_result = bandit.oracle(T)
        self.oracle_pulls = oracle_result["pulls"]
        self.best_expected_rewards = oracle_result["rewards"]
        self.best_observed_rewards = oracle_result["observations"]


    def run(self, tqdm_=True):
        """run self.agent on self.bandit until horizon T
        
            this is the method in which the interaction agent-environment is handled
        """

        rewards = np.zeros(self.T)
        self.observations = []
        self.agent.set_horizon(self.T)
        self.agent.set_arms_number(len(self.bandit.arms))
        iterator = tqdm(range(self.T), desc=f"{self.agent}") if tqdm_ else range(self.T)
        for time in iterator:
            arm_to_be_pulled = self.agent.select_arm()                           # ask the agent which arm to pull
            reward, observation = self.bandit.pull_arm(arm_to_be_pulled)    # pull the arm, get (reward, observation)
            self.agent.new_observation(arm=arm_to_be_pulled, value=observation)  # provide observation to the agent
            rewards[time] = reward                  # keeping track of the true rewards (unknown to the agent)
            self.observations.append( (time, arm_to_be_pulled, observation) )

        # compute regret
        expected_regrets = self.best_expected_rewards - rewards
        observed_regrets = self.best_observed_rewards - list(map(lambda x : x[2], self.observations))
        self.cumulative_expected_regret = np.array([*accumulate(expected_regrets)])
        self.cumulative_observed_regret = np.array([*accumulate(observed_regrets)])

        # number of pulls by arm
        self.pulls_by_arm = self.agent.number_of_pulls.copy()


    def get_regret(self) -> "tuple[float,float]":
        """returns expected cumulative regret, observed cmulative regret"""
        step = options.STEP
        e_r = self.cumulative_expected_regret[::step].copy()
        o_r = self.cumulative_observed_regret[::step].copy()
        return e_r, o_r


    def get_pulls(self) -> np.ndarray:
        """returns a list of how many times each arm has been pulled"""
        return self.pulls_by_arm.copy()


    def get_observations(self) -> list:
        """returns a list of the observed rewards"""
        step = options.STEP
        return self.observations[::step].copy()


    def __str__(self):
        s = "\n" + str(self.agent)
        s += f"\n pulls: {list(self.agent.number_of_pulls)}"
        s += f"\n expected regret: {self.cumulative_expected_regret[-1]}"
        s += f"\n observed regret: {self.cumulative_observed_regret[-1]}\n"

        return s
