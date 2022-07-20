import numpy as np

class Agent:
    """abstract class, father of all the bandit-algorithms"""

    def set_horizon(self, T :int):
        """tell the algorithm how long is the horizon

        Args:
            T (int): horizon of the problem
        """
        self.time = 0
        self.T = T
        self.pulled_arm_ids :np.ndarray = -np.ones(self.T, dtype = int)      # id of arm pulled at each round


    def set_arms_number(self, K :int):
        """tell the algorithm how many arms the bandit has

        Args:
            K (int): number of arms of the bandit we are working on
        """
        self.K = K
        self.observations = np.array([[float('nan')] * self.T] * self.K)  # list of observations received for each arm
        self.number_of_pulls :np.ndarray = np.zeros(K, dtype=int)      # number of pulls of each arm


    def select_arm(self) -> int:
        """select_arm select which arm to be pulled at next iteration

        Returns:
            (int): id of the arm to be pulled
        """
        pass


    def new_observation(self, arm :int, value :float) -> None:
        """new_observation tell the agent the reward observed by pulling an arm of the bandit

        Args:
            arm (int): id of the arm related to the observation (pulled)
            value (float): observed reward from that arm
        """
        self.observations[arm, int(self.number_of_pulls[arm])] = value
        self.pulled_arm_ids[self.time] = arm
        self.number_of_pulls[arm] += 1
        self.time += 1


    def reset(self):
        """reset resets the agent"""
        self.K = None
        self.observations = None
        self.pulled_arm_ids = None
        self.number_of_pulls = None
        self.time = 0
        

    def copy(self):
        """create an empty copy of the agent"""
        raise NotImplementedError


    def __str__(self) -> str:
        return "abstract agent"