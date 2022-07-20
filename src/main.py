import re
from argparse import ArgumentParser
from random import shuffle

import model
import options
from algorithms.KL_UCB import KL_UCB
from algorithms.R_ed_UCB import R_ed_UCB
from algorithms.R_less_UCB import R_less_UCB
from algorithms.Rexp3 import Rexp3
from algorithms.Ser4 import Ser4
from algorithms.SW_KLUCB import SW_KL_UCB
from algorithms.SW_TS_eff import SW_TS_eff
from algorithms.SW_UCB_eff import SW_UCB_eff
from bandit import Bandit
from experiment_bundle import ExperimentBundle
from utils import parse_list

# parsing argv
description = "perform many executions of bandits with different algorithms"
parser = ArgumentParser(description=description)
parser.add_argument("--test-dir", help='folder in which the tests will be saved', metavar=options.TEST_DIR, default=options.TEST_DIR)
parser.add_argument("--T", help='run tests using this time horizons set', metavar=options.T_HORIZON, default=options.T_HORIZON)
parser.add_argument("--restless", help='create restless bandits only (default: rested bandit only)', action='store_true')
parser.add_argument("--only-UCB", help='use only R-less/ed-UCB, no comparison with other algos (default: False)', action='store_true')
parser.add_argument("--step", help='plot 1 point every "step" time instants', metavar=options.STEP, default=options.STEP)
parser.add_argument("--light-plot", help='do not plot all the pulls of each arm (Default False)', action='store_false')
parser.add_argument("-exp-name", help='name of the experiment')
parser.add_argument("--n-random", help='run tests on this number of randomly generated bandits (from the ones saved in model.py)', metavar=2)
parser.add_argument("--specific-bandits", help='run tests on the specific bandit(s) (indexed in model.py)', metavar='[0,1,2]')
parser.add_argument("--print-all-bandits", help='print all the available bandits from model.py', action='store_true')

args, _ = parser.parse_known_args()

with open('src/model.py', 'r') as f:
    buffer = f.read()
num_of_bandits = int( re.findall( "\d+", re.findall("\d+\s?:\s?\[", buffer)[-1] )[0] )

if args.print_all_bandits:
        print(buffer[buffer.index("{"):])
        exit(1)
else:
    if not args.exp_name:
        parser.print_usage()
        print("error: required argument -exp-name")
        exit(1)

if args.specific_bandits:    
    indexes = parse_list(str(args.specific_bandits))
else:
    if args.n_random and int(args.n_random) > 0:
        indexes = list(range(num_of_bandits))
        indexes = indexes[-int(args.n_random):]
    else:
        parser.print_usage()
        print("Either set --n-random or --specific-bandit")
        exit(1)

options.RESTLESS = args.restless
options.TEST_DIR = f"{args.test_dir}/{args.exp_name}"
options.T_HORIZON = parse_list(str(args.T))
options.STEP = min(min(options.T_HORIZON), int(args.step))

## algorithms
R_less_ed_ucb = R_less_UCB() if options.RESTLESS else R_ed_UCB()
sw_ts = SW_TS_eff()
sw_ucb = SW_UCB_eff()
rexp3 = Rexp3()
kl_ucb = KL_UCB()
sw_kl_ucb = SW_KL_UCB()
ser4 = Ser4()

algos = [R_less_ed_ucb, sw_ts, sw_ucb, kl_ucb, sw_kl_ucb, rexp3, ser4] if not args.only_UCB else [R_less_ed_ucb]

bandits = []
for idx in indexes:
    b = Bandit(restless=options.RESTLESS)
    b.load_lambdas(model.lambdas[idx])
    bandits.append(b)

tester = ExperimentBundle(algos)
tester.run(bandits, args.light_plot)
