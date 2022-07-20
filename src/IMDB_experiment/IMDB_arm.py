import numpy as np
from arm import Arm

from IMDB_experiment.IMDB_environment import IMDB_Environment
from IMDB_experiment.strategy import Strategy


def loss_function(prediction, target):
    """not used in the experiments"""
    return target * np.log(prediction) + (1-target) * np.log(1-prediction)

class IMDB_Arm(Arm):
    """class which represents an arm of the bandit for the IMDB experiment (handles base algorithm - IMDB dataset interaction)"""

    def __init__(self, arm_id :int, algo :Strategy, env :IMDB_Environment, T :int):
        self.arm_id = arm_id
        self.algo = algo
        self.env = env
        self.T = T
        self.reward_curve = np.zeros(T)
        self.reset()

    def reset(self):
        self.time = 0
        self.algo.reset()
        self.env.shuffle_indexes()


    def load_reward_curve(self, rewards :np.ndarray, T):
        """load average reward curves from ./data"""
        self.reward_curve = rewards[:T]


    def pull(self) -> "tuple[float,float]":
        """perform an iteration and returns expected (average) and observed reward (prediction result)"""
 
        x, t = self.env.get_next_point()
        prediction = self.algo.predict(x)

        # loss = loss_function(prediction,t)

        self.algo.prediction_result(x, prediction, t)

        self.time +=1

        return self.reward_curve[self.time-1], int((prediction > 0.5) == t)


    def copy(self):
        """create & return a copy of the arm"""
        
        return IMDB_Arm(self.arm_id, self.algo.copy(), self.env.copy(), self.T)


    def __str__(self):
        return f"arm {self.arm_id} \t: {self.algo}"
