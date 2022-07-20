from argparse import ArgumentParser
from datetime import datetime
from os import listdir, makedirs

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

import options
from IMDB_experiment.IMDB_arm import IMDB_Arm
from IMDB_experiment.IMDB_environment import IMDB_Environment
from utils import plot_learning_functions_IMDB

# script to generate the average learning curves of the base algorithms for the classification problem on the IMDB dataset

PATH = "IMDB_learning_curves"

def create_arm(id, algo) -> IMDB_Arm:
    env = IMDB_Environment(options.DATA_SIZE, options.FEATURES, options.T_IMDB)
    return IMDB_Arm(id, algo, env, options.T_IMDB)


def pull_arm_till_T(arm :IMDB_Arm, T :int):
    arm_copy = arm.copy()
    observations = np.array([arm_copy.pull()[1] for _ in range(T)])
    arm_copy.reset()
    return observations


def generate_learning_curves(arms :"list[IMDB_Arm]", T :int, N :int, n_parall :int):
    makedirs(PATH, exist_ok=True)
    with open(f"{PATH}/summary.txt", "a") as f:
        f.write(f"N : {options.N_TESTS_IMDB}\nT : {options.T_IMDB}\n# FEAT : {options.FEATURES}\n# points : {options.DATA_SIZE}\n")

    observed_rewards = 0
    for arm in arms:
        observed_rewards = np.array( Parallel(n_jobs=n_parall)(delayed(pull_arm_till_T)(arm,T) for _ in tqdm(range(N), desc=f"{arm.algo}")) )   # shape (N,T)

        avg_rewards = np.average(observed_rewards, axis=0)
        var_rewards = np.var(observed_rewards, axis=0)
       
        np.save(f"{PATH}/{arm.algo}_avg", avg_rewards)
        np.save(f"{PATH}/{arm.algo}_var", var_rewards)


# parsing argv
description = f"generate the learning curves of the strategies for the IMDB problem"
parser = ArgumentParser(description=description)
parser.add_argument("-N", help='number of runs to average', metavar=options.N_TESTS_IMDB, required=True)
parser.add_argument("-feat", help='set the number of features to be used', metavar=options.FEATURES, required=True)
parser.add_argument("-size", help='set the dataset size', metavar=options.DATA_SIZE, required=True)
parser.add_argument("-T", help='set the time horizon', metavar=options.T_IMDB, required=True)
parser.add_argument("-workers", help='set the number of threads to parallelize the execution', metavar=4, required=True)
parser.add_argument("--dir", help='create a specific directory', metavar="")

args, _ = parser.parse_known_args()

options.N_TESTS_IMDB = int(args.N)
options.FEATURES = int(args.feat)
options.DATA_SIZE = int(args.size)
options.T_IMDB = int(args.T)
n_workers = int(args.workers)

if args.dir != None and args.dir != "":
    PATH = f"{PATH}/{args.dir}"
else:
    PATH = f"{PATH}/{datetime.today().strftime('%Y-%m-%d_%H:%M')}"

strategies = options.IMDB_STRATEGIES

already_generated = listdir(options.IMDB_DATA_DIR)
arms = []
for i, strategy in enumerate(strategies):
    if f"{strategy}_avg.npy" not in already_generated:
        arms.append(create_arm(i, strategies[strategy]))

# generate & save learning curves
generate_learning_curves(arms, options.T_IMDB, options.N_TESTS_IMDB, n_workers)

# plot the curves
algos_str = []
functions = {}
for arm in arms:
    a = str(arm.algo)
    algos_str.append(a)
    functions[a] = np.load(f"{PATH}/{a}_avg.npy")

plot_learning_functions_IMDB(algos_str, functions, f"{PATH}")
