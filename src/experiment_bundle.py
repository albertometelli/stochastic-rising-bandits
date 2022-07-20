import gc
import sys

import numpy as np
from tqdm import tqdm

import options
from agent import Agent
from bandit import Bandit
from experiment import Experiment
from utils import (aggregate_pulls_and_regrets, create_folders, plot_pulls,
                   plot_rewards, save_npy)


class ExperimentBundle:
    """class used to perform sequential iterations of many algorithms over many bandits"""

    def __init__(self, agents :"list[Agent]") -> None:
        self.agents = list(map(lambda agent: agent.copy(), agents))
        self.names = list(map(lambda agent : str(agent), agents))


    def run(self, bandits :"list[Bandit]"=[], plot_everything=False, special_folder=""):
        """performs an iteration of all the self.agents over the provided bandits"""

        n = len(bandits)
        for test_number in range(n):

            curr_bandit = bandits[test_number].copy()  # load bandit

            for curr_T in options.T_HORIZON:
                path = f"{options.TEST_DIR}{special_folder}/{test_number}_{curr_T}"
                create_folders(path, list(map(lambda agent : str(agent), self.agents)))

                plot_rewards(curr_bandit, curr_T, savefig=f"{path}")

                results = {}
                for agent in tqdm(self.agents, desc=f"\ntest {test_number}, T = {curr_T}"):
                    e = Experiment(curr_bandit, agent, curr_T)
                    e.run()

                    regrets = e.get_regret()
                    results[str(agent)] = {
                        "experiment" : str(e),
                        "cumul_expected_regret" : regrets[0],
                        "cumul_observed_regret" : regrets[1],
                        "pulls" : e.get_pulls()
                    }
                
                    if plot_everything:
                        plot_pulls(e, savefig=f"{path}/{agent}")

                    agent.reset()
                    curr_bandit.reset()
                    del e
                    gc.collect()

                self.log_test(path, curr_bandit, results, curr_T)
                aggregate_pulls_and_regrets(path, results)


    def log_test(self, path :str, bandit :Bandit, results :dict, T :int) -> None:
        s = str(bandit)

        for a in self.names:
            s += "-" * 40 + results[a]["experiment"]

        filename = f"{path}/summary.txt"
        with open(filename, "a") as f, np.printoptions(precision = 2, suppress = True, threshold = sys.maxsize):
                f.write(s+"\n")

        for a in results:
            save_npy(a, results[a], path)
