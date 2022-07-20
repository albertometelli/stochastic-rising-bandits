#!/bin/bash

run_my_script() {
    dir="experiments/parallel/$1"

    if [ "$5" == "restless" ]
    then
        python3 src/main.py --test-dir $dir --n-random $2 --T $3 -exp-name $4 --light-plot --restless
    else
        python3 src/main.py --test-dir $dir --n-random $2 --T $3 -exp-name $4 --light-plot
    fi
}
export -f run_my_script

if [ $# -lt 6 ]
then
    echo "arguments needed: runs to average, how many random bandits to test, 'rested'/'restless', horizon, parallel_workers, experiment_name"
    exit 1
fi

args=("$@")
N=${args[0]}
N_random=${args[1]}
less_ed="${args[2]}"
T=${args[3]}
N_PARALL=${args[4]}
exp_name="${args[5]}"
is_rank="${args[6]}"

if [ "$less_ed" != "restless" ] && [ "$less_ed" != "rested" ]
then
    echo "you should specify either 'restless' or 'rested'"
    exit 1
fi

python3 src/generate_random_bandits.py -N $N_random

if [ $N -lt $N_PARALL ]
then
    N_PARALL=$N
fi

I=$(($N/$N_PARALL))

echo "running ${less_ed} experiments"
# run multicores
for ((i=1;i<=I;i++))
do
    start=$((($i-1)*$N_PARALL +1))
    range=$(($i*$N_PARALL))
    EXP=$(seq $start $range)
    echo running iterations $EXP
    parallel -j $N_PARALL -k --lb run_my_script ::: $EXP ::: $N_random ::: $T ::: $exp_name ::: $less_ed
done

# run residuals
x=$(($I*$N_PARALL))
if [ $N -gt $x ]
then
    EXP=$(seq $(($x+1)) $N)
    echo running iterations $EXP
    parallel -j $(($N-$x)) -k --lb run_my_script ::: $EXP ::: $N_random ::: $T ::: $exp_name ::: $less_ed
fi

for ((j=0;j<$N_random;j++))
do
    python3 src/scripts/aggregate_parallel.py -N $N -path "experiments/parallel" -output "experiments" -name "${exp_name}/${j}_${T}" -T $T
done

# do not remove if I'm ranking
if [ "$is_rank" != "YES" ]
then
    rm -r experiments/parallel/*/${exp_name}
fi