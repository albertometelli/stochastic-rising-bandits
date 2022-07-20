import gc
import sys

import numpy as np
import options
from agent import Agent
from experiment import Experiment
from experiment_bundle import ExperimentBundle
from utils import (aggregate_pulls_and_regrets, create_folders,
                   plot_rewards_IMDB, save_npy)

from IMDB_experiment.IMDB_bandit import IMDB_Bandit

class IMDB_Experiment(ExperimentBundle):
    """class which handles a run of the IMDB experiment"""

    def __init__(self, agents :"list[Agent]") -> None:
        self.agents = agents
        self.names = list(map(lambda agent : str(agent), agents))


    def run(self, bandit :IMDB_Bandit, T=0):
        """run each algorithm (agent) on the bandit, save & plot the results"""

        path = f"{options.TEST_DIR}"
        create_folders(path, list(map(lambda agent : str(agent), self.agents)))
        
        plot_rewards_IMDB(bandit, T, savefig=f"{path}")

        results = {}
        for agent in self.agents:
            e = Experiment(bandit, agent, T)
            e.run()
            
            regrets = e.get_regret()
            results[str(agent)] = {
                "experiment" : str(e),
                "cumul_expected_regret" : regrets[0],
                "cumul_observed_regret" : regrets[1],
                "pulls" : e.get_pulls()
            }

            agent.reset()
            bandit.reset()
            del e
            gc.collect()

        self.log_test_IMDB(path, bandit, results, T)
        aggregate_pulls_and_regrets(path,results)


    def log_test_IMDB(self, path :str, bandit :IMDB_Bandit, results :dict, T :int) -> None:
        """log to file the results"""
        s = str(bandit)
        
        s+= f"T = {T}"

        for a in self.names:
            pulls = results[a]["pulls"]
            obs_regret = results[a]["cumul_observed_regret"]
            regret = results[a]["cumul_expected_regret"]
            s += f"\n{a}\n\n"
            s += f"arm pulls = {pulls}\n"
            s += f"expected cumul regret = {regret[-1]}\n"
            s += f"observed cumul regret = {obs_regret[-1]}\n"
            s += "-" * 40

            save_npy(a, results[a], path)

        filename = f"{path}/summary.txt"
        with open(filename, "a") as f, np.printoptions(precision = 0, suppress = True, threshold = sys.maxsize):
            f.write(s+"\n")
