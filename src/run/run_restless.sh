#!/bin/bash

run_my_script() {
    dir="experiments/parallel/$1"

    python3 src/main.py --test-dir $dir --specific-bandits $2 --T $3 -exp-name $4 --light-plot --restless
}
export -f run_my_script


if [ $# -lt 5 ]
then
    echo "arguments needed: runs to average, which bandits to test (from model.py), horizon, parallel_workers, experiment_name (without /)"
    exit 1
fi

args=("$@")
N=${args[0]}
custom=${args[1]}
T=${args[2]}
N_PARALL=${args[3]}
exp_name="${args[4]}"

if [ $N -lt $N_PARALL ]
then
    N_PARALL=$N
fi

I=$(($N/$N_PARALL))

echo "running RESTLESS experiments"
# run multicores
for ((i=1;i<=I;i++))
do
    start=$((($i-1)*$N_PARALL +1))
    range=$(($i*$N_PARALL))
    EXP=$(seq $start $range)
    echo running iterations $EXP
    parallel -j $N_PARALL -k --lb run_my_script ::: $EXP ::: $custom ::: $T ::: $exp_name
done

# run residuals
x=$(($I*$N_PARALL))
if [ $N -gt $x ]
then
    EXP=$(seq $(($x+1)) $N)
    echo running iterations $EXP
    parallel -j $(($N-$x)) -k --lb run_my_script ::: $EXP ::: $custom ::: $T ::: $exp_name
fi


res="${custom//[^,]}"
n=$((${#res}+1))
for ((j=0;j<$n;j++))
do
    python3 src/scripts/aggregate_parallel.py -N $N -path "experiments/parallel" -output "experiments" -name "${exp_name}/${j}_${T}" -T $T
done

rm -r experiments/parallel/*/${exp_name}