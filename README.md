# Stochastic Rising Bandits

This repository contains the code to run the experiments of the paper "Stochastic Rising Bandits"

- Paper: https://proceedings.mlr.press/v162/metelli22a.html

To cite this repository in publications:

    
	@InProceedings{pmlr-v162-metelli22a,
	  title = 	 {Stochastic Rising Bandits},
	  author =       {Metelli, Alberto Maria and Trov{\`o}, Francesco and Pirola, Matteo and Restelli, Marcello},
	  booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
	  pages = 	 {15421--15457},
	  year = 	 {2022},
	  editor = 	 {Chaudhuri, Kamalika and Jegelka, Stefanie and Song, Le and Szepesvari, Csaba and Niu, Gang and Sabato, Sivan},
	  volume = 	 {162},
	  series = 	 {Proceedings of Machine Learning Research},
	  month = 	 {17--23 Jul},
	  publisher =    {PMLR},
	  pdf = 	 {https://proceedings.mlr.press/v162/metelli22a/metelli22a.pdf},
	  url = 	 {https://proceedings.mlr.press/v162/metelli22a.html},
	  abstract = 	 {This paper is in the field of stochastic Multi-Armed Bandits (MABs), i.e., those sequential selection techniques able to learn online using only the feedback given by the chosen option (a.k.a. arm). We study a particular case of the rested and restless bandits in which the arms’ expected payoff is monotonically non-decreasing. This characteristic allows designing specifically crafted algorithms that exploit the regularity of the payoffs to provide tight regret bounds. We design an algorithm for the rested case (R-ed-UCB) and one for the restless case (R-less-UCB), providing a regret bound depending on the properties of the instance and, under certain circumstances, of $\widetilde{\mathcal{O}}(T^{\frac{2}{3}})$. We empirically compare our algorithms with state-of-the-art methods for non-stationary MABs over several synthetically generated tasks and an online model selection problem for a real-world dataset. Finally, using synthetic and real-world data, we illustrate the effectiveness of the proposed approaches compared with state-of-the-art algorithms for the non-stationary bandits.}
	}


# Experiments
Details on how to reproduce the experiments presented in the paper or perform new simulations.

Notice that we have provided in ```src/model.py```, the definition of the bandits used to perform the experiments presented in the paper (bandit "0" is the 15 arms bandit, while bandit "1" is the 2 arms one)

First of all, don't forget to ```chmod +x src/run/*```

## Experiments presented in the paper
### 15 arms bandit
To reproduce the 15 arms bandit experiment presented in the paper (restless or rested) run:
```
./src/run/run_restless.sh 500 0 50000 <parallel_cores> <out_folder_name>
./src/run/run_rested.sh 500 0 50000 <parallel_cores> <out_folder_name>
```
'out_folder_name' cannot contain "/"

### 2 arms bandit
To reproduce the 2 arms rested bandit experiment presented in the paper run:
```
./src/run/run_rested.sh 500 1 50000 <parallel_cores> <out_folder_name>
```

### IMDB experiment
To reproduce the IMDB experiment presented in the paper run:
```
./src/run/run_IMDB_experiment.sh 30 <parallel_cores> <out_folder_name>
```

## Further simulations
If you want to perform either random or customized experiments

### random experiments
```
./src/run/run_random_bandits.sh <M> <N> <'restless'/'rested'> <horizon> <parallel_cores> <out_folder_name>
```
If you want to perform a rank process instead, you have to run:
```
./src/run/rank.sh <M> <N> <'restless'/'rested> <horizon> <parallel_workers> <experiment_name>
```
where N is the number of distinct random bandits (rested or restless) the algorithms will be used onto and M is the number of times the runs will be repeated in order to provide an average result (total: N*M runs over N bandits, each of which is averaged over M times)

### customized experiment
first of all you have to find out the available bandits:
  - option 1: open ```src/model.py``` and look at the bandits available (you can even add your own bandit if you follow the correct syntax)
  - option 2: run ```python3 src/main.py --print-all-bandits```
take note of the _indexes_ of the bandits you want to run and, accordingly to the setting (restless or rested) run:
```
./src/run/run_restless.sh <runs_to_average> <indexes> <horizon> <parallel_cores> <out_folder_name>
./src/run/run_rested.sh <runs_to_average> <indexes> <horizon> <parallel_cores> <out_folder_name>
```
'indexes' must be provided comma separated (without blanks)
i.e.
```
./src/run/run_rested.sh <runs_to_average> 1,2,3 <horizon> <parallel_cores> <out_folder_name>
```

## Detailed scripts description
- ### ```run_restless.sh``` 
    perform a restless bandit experiment, choosing one (or more, providing a comma separation) of the existing bandits (in ```src/model.py```) by simply providing its (their) identification number(s)
    
    the script works as follows:
    ```
    ./src/run/run_restless.sh <runs_to_average> <bandits_ids> <horizon> <parallel_cores> <out_folder_name>
    ```
- ### ```run_rested.sh``` 
    perform a rested bandit experiment, choosing one (or more) of the existing bandits (in ```src/model.py```) by simply providing its (their) identification number(s)
    
    the script works as follows:
    ```
    ./src/run/run_rested.sh <runs_to_average> <bandits_ids> <horizon> <parallel_cores> <out_folder_name>
    ```
- ### ```run_random_bandits.sh```
    perform experiments on randomly generated bandits
    
    the script works as follows:
    ```
    ./src/run/run_random_bandits.sh <M> <N> <'rested'/'restless'> <horizon> <parallel_cores> <out_folder_name>
    ```
    where N is the number of distinct random bandits (rested or restless) the algorithms will be used onto, and M is the number of times the runs will be repeated in order to provide an average result (total: N*M runs over N bandits, each of which is averaged over M times)

    Moreover we suggest the usage of an horizon of at least 5,000 to avoid unsignificative reward functions (i.e. always zero)

- ### ```run_IMDB_experiment.sh```
    perform the IMDB experiment detailed in the paper
    
    the script works as follows:
    ```
    ./src/run/run_IMDB_experiment.sh <runs_to_average> <parallel_cores> <out_folder_name> [<base_algos>]
    ```
    the argument 'base_algos' is optional (if omitted the script uses the base algorithms presented in the paper) and can be a list (comma separated); you can also use "." to use them all.
    You can find the list of all the available base algorithms in ```src/options.py```
 
    i.e.
    ```
    ./src/run/run_IMDB_experiment.sh <runs_to_average> <parallel_cores> <out_folder_name> lr2,ogd3
    ```

- ### ```rank.sh```
    run N distinct random experiments (rested or restless) each of which averaged M times, then perform a ranking process over the results

    the script works as follows:
    ```
    ./src/run/rank.sh <N> <M> <'restless'/'rested> <horizon> <parallel_workers> <experiment_name>
    ```
- ### ```clean_model.sh```
    restore ```src/model.py``` to the default configuration (i.e. erase every randomly generated bandit and keep only the 15 arms bandit and the 2 arms one)

    simply run ```./src/run/clean_model.sh```
