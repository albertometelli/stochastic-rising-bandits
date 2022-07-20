import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np


def find_positioning(regrets):
    pos = np.zeros(len(regrets))
    sorted_r = regrets.copy()
    sorted_r.sort()
    for i in range(pos.size):
        try:
            pos[i] = np.where(sorted_r == regrets[i])[0]
        except ValueError:
            print("all the algorithms have achieved the same regret, try increasing the horizon")
            exit(1)
    
    return pos + 1

parser = ArgumentParser(description="perform the ranking of the experiments")
parser.add_argument("-path", help='path to the folder containing all the bandits previously ran',metavar="experiments/my_exp/", required=True)
parser.add_argument("-N", help='number of experiments to compute the ranking from', required=True)
parser.add_argument("-T", help='horizon (must be the same for all the experiments)', required=True)
parser.add_argument("--step", help='step', default=1000)

args, _ = parser.parse_known_args()

N = int(args.N)
T = int(args.T)
step = int(args.step)
IN = args.path

# EXTRACT DATA from single runs
path_0 = f"{IN}/0_{T}_avg"
algos = [algo for algo in os.listdir(path_0) if os.path.isdir(f"{path_0}/{algo}") and algo != "regrets"]
algos.sort()

regrets = {}
for i in range(N):  # cycle all the bandits
    algo_res = []

    for k, a in enumerate(algos):   # extract regret of each algo
        r = np.load(f"{IN}/{i}_{T}_avg/{a}/regret_avg.npy")
        algo_res.append(r[-1])

    regrets[i] = algo_res

positions = np.zeros((N,len(algos)))
for j in range(N):
    positions[j] = find_positioning(regrets[j])

final_positions = np.average(positions, 0)
final_positions_err = 2 * np.sqrt(np.var(positions,0) / N)

print("RANKING RESULTS")
for i,a in enumerate(algos):
    print(a, round(final_positions[i],2), "\u00b1", round(final_positions_err[i],2))


## BOXPLOT of all algo_positions
fig, axs = plt.subplots(2,4)
fig.suptitle("Ranking of each algo in each experiment")
for i, algo in enumerate(algos):
    axs[i % 2, i // 2].set_title(algo)
    c = np.zeros(len(algos))
    unique_pos, counts = np.unique(positions[:,i], return_counts=True)
    c[unique_pos.astype(int)-1] += counts
    axs[i % 2, i // 2].bar(np.arange(c.size)+1, c, width=0.1)
plt.savefig("ranking_full.png",dpi=180)
