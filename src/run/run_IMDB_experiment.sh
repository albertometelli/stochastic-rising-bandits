#!/bin/bash

run_my_script() {
    dir="IMDB_experiments/parallel/$1"
    
    python3 src/IMDB_main.py --test-dir $dir  --T $2 --set-size 50000 --feat $3 --step 100 -force-strategies $4 -exp-name $5
}
export -f run_my_script


if [ $# -lt 3 ]
then
    echo "arguments needed: runs to average, parallel workers, experiment name"
    echo "you can additionally provide the name of the strategies to use, otherwise the ones presented in the paper are used"
    exit 1
fi

args=("$@")
N=${args[0]}
T=50000
feat=1000
N_PARALL=${args[1]}
exp_name=${args[2]}

if [ -z "${args[3]}" ]
then
    strategies="lr0003,ogd01,nn1,nn2,nn6,nn7,nn11"
else
    strategies="${args[3]}"
fi


I=$(($N/$N_PARALL))

echo "running IMDB experiments"
# run multicores
for ((i=1;i<=I;i++))
do
    start=$((($i-1)*$N_PARALL +1))
    range=$(($i*$N_PARALL))
    EXP=$(seq $start $range)
    echo running iterations $EXP
    parallel -j $N_PARALL -k --lb run_my_script ::: $EXP ::: $T ::: $feat ::: $strategies ::: $exp_name
done

# run residuals
x=$(($I*$N_PARALL))
if [ $N -gt $x ]
then
    EXP=$(seq $(($x+1)) $N)
    echo running iterations $EXP
    parallel -j $(($N-$x)) -k --lb run_my_script ::: $EXP ::: $T ::: $feat ::: $strategies ::: $exp_name
fi

python3 src/scripts/aggregate_parallel_IMDB.py -N $N -path IMDB_experiments/parallel -output "IMDB_experiments" -name $exp_name -T $T -labels $strategies

# rm -r IMDB_experiments/parallel/*/${exp_name}
