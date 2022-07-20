import numpy as np
from IMDB_experiment.strategy import Strategy
from scipy.special import expit


class OGD(Strategy):
    """Online Gradient Descent"""

    def __init__(self, n_features :int, beta_0 = 0.5):
        self.beta_0 = beta_0
        self.n_features = n_features
        self.w_0 = np.ones(n_features) / n_features
        self.reset()


    def predict(self, x) -> float:
        def sigmoid(x):
            return min( max( expit(x), 1e-4 ), 1-1e-4 )

        y = sigmoid( np.dot(self.w_t, x) )
        
        return y


    def prediction_result(self, x, prediction, target):
        self.t += 1
        gradient = x * (-target * (1 - prediction) + (1 - target) * prediction)
        y_t1 = self.w_t - self.beta_0 * gradient / self.t**0.5
        self.w_t = self.project(y_t1)


    def project(self,v):
        return np.minimum(np.maximum(v, -2), 2)


    def reset(self):
        self.w_t = self.w_0
        self.t = 0


    def copy(self):
        """create & return a copy of OGD"""
        return OGD(self.n_features, self.beta_0)


    def __str__(self) -> str:
        return f"OGD beta = {self.beta_0}"
