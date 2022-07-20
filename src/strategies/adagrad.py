import numpy as np
from IMDB_experiment.strategy import Strategy
from scipy.special import expit


class ADAGRAD(Strategy):
    """not used in the final experiments"""

    def __init__(self, n_features :int, alpha=1):
        self.w_0 = np.ones(n_features)/n_features
        self.n_feature = n_features
        self.alpha = alpha
        self.w_t = self.w_0
        self.eps = 1e-8
        self.reset()


    def predict(self, x) -> float:
        def sigmoid(x):
            return min( max( expit(x), 1e-4 ), 1-1e-4 )

        y = sigmoid( np.dot(self.w_t, x) )
        return y


    def prediction_result(self, x, prediction, target):
        gradient = x * (-target * (1 - prediction) + (1 - target) * prediction)
        self.V += gradient**2
        self.w_t = self.w_t - self.alpha * gradient/(self.V + self.eps)**0.5


    def reset(self) -> None:
        self.w_t = self.w_0
        self.V = np.zeros(shape=self.n_feature)


    def copy(self):
        """create & return a copy of ADAGRAD"""
        return ADAGRAD(self.n_feature, self.alpha)


    def __str__(self):
        return f"ADAGRAD alpha = {self.alpha}"
