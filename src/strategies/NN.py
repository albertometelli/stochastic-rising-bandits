import numpy as np
from IMDB_experiment.strategy import Strategy
from sklearn.exceptions import NotFittedError
from sklearn.neural_network import MLPClassifier


class NN(Strategy):
    """Neural Network"""

    def __init__(self, nn_shape :tuple, learning_rate = 0.001):
        """define the shape of the network (hidden layers); i.e. (3,5,) means the nn will have 2 hidden layers with 3 and 5 neurons respectively"""
        self.learning_rate = learning_rate
        self.nn_shape = nn_shape
        self.reset()


    def predict(self, x) -> float:
        try:
            return max( min( self.classifier.predict_proba(x.reshape(1,-1)) [0,1], 1-1e-4), 1e-4)
        except NotFittedError:
            return 0.5


    def prediction_result(self, x, prediction, target):
        self.classifier.partial_fit(x.reshape(1,-1), [target], classes=[0,1])


    def reset(self):
        self.classifier = MLPClassifier(hidden_layer_sizes=self.nn_shape, learning_rate="constant", learning_rate_init=self.learning_rate)


    def copy(self):
        """create & return a copy of NN"""
        return NN(self.nn_shape)


    def __str__(self) -> str:
        return f"NN: {self.nn_shape}"

