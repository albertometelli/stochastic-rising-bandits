import re
from math import ceil
from os import makedirs

import matplotlib.pyplot as plt
import numpy as np

import options
from bandit import Bandit
from experiment import Experiment
from IMDB_experiment.IMDB_bandit import IMDB_Bandit
from plot import Curve, Custom_Plotter

STEP = options.STEP
PLOTTER = Custom_Plotter()

def plot_pulls(e :Experiment, savefig="") -> None:
    """plot pulls of each arm"""
    arms = e.bandit.arms
    for k in range(len(arms)):
        obs = e.get_observations()
        time = np.arange(0, e.T, STEP)

        # dashed line: projection of future pulls
        projection = Curve(time, np.array([arms[k].reward_model.reward_function(t) for t in time]), linestyle="--r", description="projected rewards")
        obs_k = np.array([o[2] for o in obs if o[1] == k])
        if e.bandit.restless:
            obs_t = np.array([o[0] for i,o in enumerate(obs) if o[1] == k])
        else:
            obs_t = np.array([i*STEP for i,o in enumerate(obs) if o[1] == k])

        observations_curve = Curve(obs_t, obs_k, linestyle=".b", description="observations")
        PLOTTER.plot_graph("pulls", [observations_curve, projection], title=f"arm: {k}\n{arms[k].reward_model.description}")
    
    if savefig:
        PLOTTER.save_pulls(f"{savefig}", str(e.agent))     #savefig==path/to/test/dir

    PLOTTER.reset()


def plot_rewards(bandit :Bandit, T :int, savefig="") -> None:
    """plot reward functions"""
    rewards = []
    for k,arm in enumerate(bandit.arms):
        time = np.arange(0,T,STEP)
        rewards.append(Curve(time, np.array([arm.reward_model.reward_function(t) for t in time]), linestyle="", description=k))
    PLOTTER.plot_graph("reward_functions", rewards, title="reward functions")
    
    if savefig:
        PLOTTER.save_reward_functions(savefig)     #savefig==path/to/test/dir

    PLOTTER.reset()


def plot_rewards_IMDB(bandit :IMDB_Bandit, T :int, savefig="") -> None:
    """plot reward functions of IMDB experiment"""
    rewards = []
    for k,arm in enumerate(bandit.arms):
        time = np.arange(0,T,STEP)
        rewards.append(Curve(time, np.array([arm.reward_curve[t] for t in time]), linestyle="", description=str(arm)))
    PLOTTER.plot_graph("reward_functions", rewards, title="reward functions")
    
    if savefig:
        PLOTTER.save_reward_functions(savefig)     #savefig==path/to/test/dir

    PLOTTER.reset()


def plot_learning_functions_IMDB(algos :"list[str]", functions :dict, savefig="") -> None:
    """"""
    curves = []
    for algo in algos:
        f = functions[algo]
        curves.append(Curve(np.arange(f.size), f, description=algo, linestyle=""))
    
    PLOTTER.plot_graph("reward_functions", curves, title="average learning functions")
    PLOTTER.save_reward_functions(savefig)
    PLOTTER.reset()


def aggregate_pulls_and_regrets(path :str, results :dict) -> None:
    """plot regrets and pulls in a single plot"""
    algos = list(results.keys())

    # 1: plot pulls of each algo
    X = 2
    fig, axs = plt.subplots(X, max(ceil(len(algos) / X),2))
    fig.suptitle("Pulls")
    for i, a in enumerate(algos):
        pull_history = results[a]["pulls"]
        axs[i%X, i//X].bar(np.arange(pull_history.size), pull_history, width=0.1)
        axs[i%X, i//X].set_title(str(a))
        axs[i%X, i//X].yaxis.set_ticklabels([])
    fig.set_size_inches(15.5, 8.5)
    plt.savefig(f"{path}/pulls.png", dpi=240)


    # 2: plot regret of each algo
    plt.figure()
    plt.title("Regrets")
    for i, a in enumerate(algos):
        regret = results[a]["cumul_expected_regret"]
        plt.plot(np.arange(regret.size)*options.STEP, regret)
    plt.legend(algos)
    fig = plt.gcf()
    fig.set_size_inches(15.5, 8.5)
    plt.savefig(f"{path}/regrets.png", dpi=240)


def create_folders(path :str, names :"list[str]", create_agent_subfodlers=True) -> None:
    """create folders structure"""
    if create_agent_subfodlers:
        for name in names:
            makedirs(f"{path}/{name}", exist_ok=True)


def parse_list(string :str) -> list:
    """generate a list of ints from a string like '[1,2,3,4]'"""
    nums = re.findall("\d+", string)
    return [int(n) for n in nums]


def save_npy(name :str, result :dict, path :str) -> None:
    np.save(f"{path}/{name}/pulls", result["pulls"])          # save the pull history
    np.save(f"{path}/{name}/regret", result["cumul_expected_regret"])        # save the cumulative regret
