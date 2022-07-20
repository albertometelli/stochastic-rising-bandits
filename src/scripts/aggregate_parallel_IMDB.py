import os
from argparse import ArgumentParser
from shutil import copyfile

import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib

parser = ArgumentParser(description="aggregate results of different experiment runs")
parser.add_argument("-name", help='name of the experiment',metavar="2021-09-16-17:26", required=True)
parser.add_argument("-path", help='path/to/parallel/experiments/dir',metavar="experiments/parallel/12345", required=True)
parser.add_argument("-N", help='number of experiments', required=True)
parser.add_argument("-T", help='horizon', required=True)
parser.add_argument("-output", help='path/to/output/dir', required=True)
parser.add_argument("--step", help='step', default=100)
parser.add_argument("-labels", help='arms labels', required=True)

args, _ = parser.parse_known_args()

name = args.name
N = int(args.N)
T = int(args.T)
step = int(args.step)
IN = args.path
OUT = f"{args.output}/{name}_avg"

# EXTRACT DATA from single runs
magic_string = IN + "/{}/{}"
path_1 = magic_string.format(1,name)
algos = [algo for algo in os.listdir(path_1) if os.path.isdir(f"{path_1}/{algo}") and algo != "regrets"]
algos.sort()
all_data = {}
K = np.load(f"{path_1}/{algos[0]}/pulls.npy").size
X_LABELS = args.labels.split(",")
if X_LABELS[0] == ".":
    X_LABELS = [f"nn_{i}" for i in range(1, K+1)]

for a in algos:
    r = np.zeros(shape=(N,T // step))
    p = np.zeros(shape=(N,K))
    for i in range(1,N+1):
        path = magic_string.format(i, name) + f"/{a}"
        r[i-1] = np.load(f"{path}/regret.npy")
        p[i-1] = np.load(f"{path}/pulls.npy")

    all_data[a] = {
        "regret_avg" : np.average(r,0),
        "regret_var" : np.var(r,0),
        "pulls_avg" : np.average(p,0),
        "pulls_var" : np.var(p,0)
    }

# PLOT
os.makedirs(OUT, exist_ok=True)
copyfile(f"{path_1}/reward_functions.png", f"{OUT}/reward_functions.png")

fig = plt.figure()
fig.set_size_inches(15.5,8.5)
plt.title(f"cumulative expected regret\n{N} simulations")
for a in algos:
    avg = all_data[a]["regret_avg"]
    err = 2 * np.sqrt(all_data[a]["regret_var"] / N)
    plt.plot(np.arange(T,step=step), avg)
    plt.fill_between(np.arange(T,step=step), avg+err, avg-err, alpha=0.1)
plt.legend(algos)
plt.savefig(f"{OUT}/regrets_avg.png",dpi=240)
tikzplotlib.clean_figure()
tikzplotlib.save(f"{OUT}/regrets_avg.tex")


fig = plt.figure()
fig.set_size_inches(15.5,8.5)
plt.title(f"pulls by arm\n{N} simulations")
xx = 0
for idx,a in enumerate(algos):
    pulls = all_data[a]["pulls_avg"]
    err = 2 * np.sqrt(all_data[a]["pulls_var"] / N)
    plt.errorbar(x=np.arange(K)+xx, y=pulls, yerr=err, capsize=3,
        marker="s", markersize=5, linestyle="None")
    xx += 0.05
plt.xticks(np.arange(K), X_LABELS, rotation=45)
plt.tight_layout()
plt.legend(algos)
plt.savefig(f"{OUT}/pulls_avg.png",dpi=240)
tikzplotlib.clean_figure()
tikzplotlib.save(f"{OUT}/pulls_avg.tex")


# SAVE AVGs
for a in algos:
    os.makedirs(f"{OUT}/{a}", exist_ok=True)
    np.save(f"{OUT}/{a}/regret_avg", all_data[a]["regret_avg"])
    np.save(f"{OUT}/{a}/regret_var", all_data[a]["regret_var"])
    np.save(f"{OUT}/{a}/pulls_avg", all_data[a]["pulls_avg"])
    np.save(f"{OUT}/{a}/pulls_var", all_data[a]["pulls_var"])
