#!/bin/bash

if [ $# -lt 5 ]
then
    echo "arguments needed: runs to average for each bandit, how many random bandits to test, 'rested'/'restless', horizon, parallel_workers, experiment_name"
    exit 1
fi

args=("$@")

M_g=${args[0]}
N_g=${args[1]}
less_ed_g="${args[2]}"
T_g=${args[3]}
N_PARALL_g=${args[4]}
dir_name_g="${args[5]}"

source src/run/clean_model.sh

source src/run/run_random_bandits.sh $M_g $N_g $less_ed_g $T_g $N_PARALL_g $dir_name_g "YES"

python3 src/scripts/ranking.py -path "experiments/$dir_name_g" -N $N_g -T $T_g

rm -r experiments/parallel/*/${dir_name_g}
