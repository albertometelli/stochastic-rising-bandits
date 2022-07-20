from random import randint

from strategies.adagrad import ADAGRAD
from strategies.logistic_regression import LogisticRegression
from strategies.naive_bayes import NaiveBayes
from strategies.NN import NN
from strategies.OGD import OGD

TEST_DIR = "experiments"            # default test dir


## EXPERIMENTS
T_HORIZON = [50000]                 # horizon extraction pool
LOAD_BANDIT = [1,2]                 # run tests on those bandits
STEP = 100                          # save&plot 1 point every STEP time instants

RESTLESS = False        # restless or rested bandit

## IMDB EXPERIMENTS
N_TESTS_IMDB = 1000             # reward curves generation runs to average

FEATURES = 1000                # number of features to use
DATA_SIZE = 50000               # number of points in the dataset
T_IMDB = 50000                  # time horizon


IMDB_DATA_DIR = "data/learning curves/npy"

IMDB_STRATEGIES = {
    "lr0001" : LogisticRegression(FEATURES, 0.001),
    "lr05" : LogisticRegression(FEATURES, 0.5),
    "lr025" : LogisticRegression(FEATURES, 0.25),
    "lr0003" : LogisticRegression(FEATURES, 0.003),
    "lr1" : LogisticRegression(FEATURES, 1),
    "lr:0.1" : LogisticRegression(FEATURES, 0.1),

    "nb" : NaiveBayes(),

    "ada1" : ADAGRAD(FEATURES, 1),
    "ada05" : ADAGRAD(FEATURES, 0.5),
    "ada025" : ADAGRAD(FEATURES, 0.25),
    "ada0003" : ADAGRAD(FEATURES, 0.003),
    "ada0001" : ADAGRAD(FEATURES, 0.001),
    "ada005" : ADAGRAD(FEATURES, 0.05),

    "ogd1" : OGD(FEATURES, 1),
    "ogd05" : OGD(FEATURES, 0.5),
    "ogd00001" : OGD(FEATURES, 0.0001),
    "ogd025" : OGD(FEATURES, 0.25),
    "ogd0003" : OGD(FEATURES, 0.003),
    "ogd01" : OGD(FEATURES, 0.1),

    "nn1" : NN(nn_shape=(1,)),
    "nn2" : NN(nn_shape=(2,)),
    "nn3" : NN(nn_shape=(5,)),
    "nn4" : NN(nn_shape=(10,)),
    "nn5" : NN(nn_shape=(1,1)),    
    "nn6" : NN(nn_shape=(2,2)),    
    "nn7" : NN(nn_shape=(2,2,2)),    
    "nn8" : NN(nn_shape=(5,5)),    
    "nn9" : NN(nn_shape=(10,5,2)),    
    "nn10" : NN(nn_shape=(10,5,2)),
    "nn11" : NN(nn_shape=(1,1,2)),
}

