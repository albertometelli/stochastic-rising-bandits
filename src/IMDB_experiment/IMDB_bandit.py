import numpy as np
import options
from bandit import Bandit

from IMDB_experiment.IMDB_arm import IMDB_Arm

class IMDB_Bandit(Bandit):
    """bandit class for the IMDB experiment"""

    def __init__(self, arms :"list[IMDB_Arm]" = []) -> None:
        self.arms = arms
        self.arm_number = len(self.arms)
        self.oracle_result = None
        self.time = 0


    def pull_arm(self, i :int) -> "tuple[float,float]":
        """pull_arm : pulls arm i of the bandit and returns a tuple (avg reward, observed reward)"""
        self.time += 1
        return self.arms[i].pull()


    def load_arms_reward_curves(self, path, T):
        """load average reward curves of each arm (from /data)"""
        for arm in self.arms:
            try:
                arm.load_reward_curve(np.load(f"{path}/{arm.algo}_avg.npy"),T)
            except IOError:
                print(f"cannot find {path}/{arm.algo}_avg.npy")
                exit(1)


    def oracle(self, T :int) -> dict:
        """oracle constant policy (rested bandit), computes & returns best arm, best cumul observations, best avg cumul reward"""
        if not self.oracle_result or T != self.oracle_result["T"]:
            best_arm_id = None
            best_value = None
            for arm in self.arms:
                curr_value = np.sum(arm.reward_curve)
                if best_arm_id is None or curr_value > best_value:
                    best_value = curr_value
                    best_arm_id = arm.arm_id
            
            self.best_arm = best_arm_id
            self.oracle_result = {
                "T" : T,
                "pulls" : np.array([T if arm_id == self.best_arm else 0 for arm_id in range(len(self.arms))]),
                "observations" : self.arms[self.best_arm].reward_curve,
                "rewards" : self.arms[self.best_arm].reward_curve,
            }

            self.reset()

        return self.oracle_result


    def reset(self) -> None:
        self.time = 0
        for arm in self.arms:
            arm.reset()


    def __str__(self) -> str:
        s = f"\nIMDB EXPERIMENT:\nfeatures = {options.FEATURES}\npoints = {options.DATA_SIZE}\n"

        for arm in self.arms:
            s += f"{arm}\n"
        
        s += "-" * 40 + f"\n\nbest arm = arm {self.best_arm}"

        return s
