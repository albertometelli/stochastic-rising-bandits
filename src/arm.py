import numpy as np
from reward_model import RewardModel

class Arm:
    """Class representing an arm of the bandit"""

    def __init__(self, id :int, model :RewardModel, restless=False):
        self.arm_id = id
        self.reward_model = model   # model generating rewards, i.e. \mu_i(.)
        self.restless = restless

        self.pulls_number = 0   # next free place in the list == list length


    def reset(self):
        """resets the arm history"""
        self.pulls_number = 0


    def pull(self, t) -> "tuple[float,float]":
        """pull generates and returns next (true reward, observation)

        parameter t (time instant) is significative only if the arm operates in a restless setting

        Returns:
            tuple: true_reward, observation
        """
        x = t if self.restless else self.pulls_number
        next_reward = self.reward_model.reward_function(x)
        next_observation = self.reward_model.observation_function(x)

        self.pulls_number += 1

        return (next_reward, next_observation)


    def copy(self):
        """create & return a copy of the arm"""
        return Arm(self.arm_id, self.reward_model, self.restless)


    def __str__(self):
        return f"arm {self.arm_id} : {self.reward_model.description}"