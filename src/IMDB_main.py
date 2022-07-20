from argparse import ArgumentParser
from os import listdir

import options
from algorithms.KL_UCB import KL_UCB
from algorithms.Rexp3 import Rexp3
from algorithms.Ser4 import Ser4
from algorithms.SW_KLUCB import SW_KL_UCB
from algorithms.SW_TS_eff import SW_TS_eff
from algorithms.SW_UCB_eff import SW_UCB_eff
from algorithms.R_ed_UCB import R_ed_UCB
from IMDB_experiment.IMDB_arm import IMDB_Arm
from IMDB_experiment.IMDB_bandit import IMDB_Bandit
from IMDB_experiment.IMDB_environment import IMDB_Environment
from IMDB_experiment.IMDB_experiment import IMDB_Experiment
from strategies.NN import NN
from strategies.adagrad import ADAGRAD
from strategies.logistic_regression import LogisticRegression
from strategies.naive_bayes import NaiveBayes
from strategies.OGD import OGD

def create_arm(id, algo) -> IMDB_Arm:
    env = IMDB_Environment(options.DATA_SIZE, options.FEATURES, options.T_IMDB)
    return IMDB_Arm(id, algo, env, options.T_IMDB)


# parsing argv
description = "run one experiment using the IMDB dataset"
parser = ArgumentParser(description=description)
parser.add_argument("--test-dir", help='folder in which the tests will be saved', metavar=options.TEST_DIR, default=options.TEST_DIR)
parser.add_argument("--features", help='set the number of features to be used', metavar=options.FEATURES, default=options.FEATURES,)
parser.add_argument("--set-size", help='set dataset size', metavar=options.DATA_SIZE, default=options.DATA_SIZE,)
parser.add_argument("--T", help='use this time horizon', metavar=options.T_IMDB, default=options.T_IMDB)
parser.add_argument("--step", help='plot 1 point every "step" time instants', metavar=100, default=100)
parser.add_argument("-force-strategies", help='use only the strategies [comma separated] ("*" to use all)', required=True)
parser.add_argument("-exp-name", help='name of the experiment', required=True)

args, _ = parser.parse_known_args()

options.FEATURES = int(args.features)
options.DATA_SIZE = int(args.set_size)
options.T_IMDB = int(args.T)
options.STEP = int(args.step)
options.TEST_DIR = f"{args.test_dir}/{args.exp_name}"
strat = args.force_strategies

algos = {
    "r_ed_ucb" : R_ed_UCB(IMDB=True),
    "sw_ucb" : SW_UCB_eff(),
    "sw_ts" : SW_TS_eff(),
    "rexp3" : Rexp3(),
    "ser4" : Ser4(),
    "kl_ucb" : KL_UCB(),
    "sw_klucb" : SW_KL_UCB(),
}

strategies = options.IMDB_STRATEGIES

arms = []
if strat != ".":
    strat = strat.split(",")
    for i, s in enumerate(strat):
        arms.append(create_arm(i, strategies[s]))
else:
    for i, s in enumerate(strategies):
        arms.append(create_arm(i, strategies[s]))

bandit = IMDB_Bandit(arms)
bandit.load_arms_reward_curves(path = options.IMDB_DATA_DIR, T=options.T_IMDB)

tester = IMDB_Experiment(list(algos.values()))
tester.run(bandit=bandit, T = options.T_IMDB)
